// The pieces shared by the multivariate distributions (laplace::distrib); the
// design is in note/distrib_multivariate.md, decided with Giovanni on
// 2026-11-15. distrib/core.hpp (the univariate structure) is not changed.
//
// A multivariate distrib has one vector parameter mu in R^d, which varies by
// unit, and "the rest", the parameters theta of its structure. It holds its
// structure by value (copied into kernels) and splits the work in two:
//
//   state_size(), scratch_size()  doubles of the state and of the working scratch
//   theta_state(theta, w)         writes into w[0, state_size()) what depends on
//                                 theta only (log det, the matrix of the quadratic
//                                 form, the factor) and returns a small state that
//                                 points into w and at theta; it uses
//                                 w[state_size(), state_size() + scratch_size())
//                                 as scratch
//   logpdf(state, y, mu)          the log-density of the unit y
//   logpdf_dmu_dk<A, K>(state, y, mu, idx, out, w)
//                                 A derivatives in mu and K in theta along the
//                                 sorted tuple idx (ignored when K = 0), A + K <= 6;
//                                 out is a scalar (A = 0), a d-vector (A = 1), a
//                                 packed symmetric d x d matrix (A = 2, by rows of
//                                 the lower triangle, entry (i, j) at
//                                 i (i + 1)/2 + j), a sym_tensor over d (A >= 3,
//                                 the sorted tuples in lexicographic order);
//                                 returns false, out untouched, when the block is
//                                 zero by structure; w has scratch_size() doubles
//   logpdf_dy_dk<J, K>            the same in y, (-1)^J times the mu block
//   normalization                 proper, degenerate or improper (below)
//
// The state stays valid while w and theta do. With a theta shared by the units
// the state is computed once; with a theta per unit each unit computes its own.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/pdmat/core.hpp>

#include <concepts>
#include <vector>

namespace laplace::distrib {

/**
 * \ingroup distrib_multivariate
 * \brief Normalization regime allowed by a multivariate distribution's matrix parameter.
 *
 * Distinguishes three regimes for the quadratic/precision or covariance matrix:
 * - `proper`: a positive definite matrix in either the covariance or precision parametrization.
 * - `degenerate`: a singular covariance matrix (a proper distribution supported on \f$\mu + \operatorname{col}(\Sigma)\f$).
 * - `improper`: a singular precision matrix (a density flat along the null space of \f$Q\f$).
 */
// What the matrix parameter allows (note/distrib_multivariate.md section 4):
// proper, a positive definite matrix in either parametrization; degenerate, a
// singular covariance (a proper law on mu + col(Sigma)); improper, a singular
// precision (flat along its null space).
enum class distrib_normalization : int { proper, degenerate, improper };

/**
 * \ingroup distrib_multivariate
 * \brief Block selector for the expected Fisher information matrix and its derivatives.
 *
 * Partitions the parameter vector into the location vector \f$\mu\f$ and the
 * matrix-structure parameter vector \f$\theta\f$:
 * - `vector_vector`: the \f$(\mu, \mu)\f$ block.
 * - `vector_rest`: the cross \f$(\mu, \theta)\f$ block.
 * - `rest_rest`: the \f$(\theta, \theta)\f$ block.
 */
// The blocks of the expected information: (mu, mu), (mu, theta), (theta, theta).
enum class distrib_mv_info_block : int { vector_vector, vector_rest, rest_rest };

/**
 * \ingroup distrib_multivariate
 * \brief Concept satisfied by multivariate distribution descriptors in \c laplace::distrib.
 *
 * Requires a compile-time `normalization` constant, an associated `state_type`,
 * dimension and buffer-size queries (`dim`, `n_params`, `state_size`,
 * `scratch_size`), parameter precomputation (`theta_state`), `logpdf`, and
 * first-order derivative calls `logpdf_dmu_dk<1, 0>` and `logpdf_dmu_dk<0, 1>`.
 *
 * \tparam D Multivariate distribution type to check.
 */
template <class D>
concept multivariate_distrib =
    requires(const D d, const double* p, double* w, const int* idx, const typename D::state_type& st) {
      { D::normalization } -> std::convertible_to<distrib_normalization>;
      { d.dim() } -> std::convertible_to<int>;
      { d.n_params() } -> std::convertible_to<int>;
      { d.state_size() } -> std::convertible_to<int>;
      { d.scratch_size() } -> std::convertible_to<int>;
      { d.theta_state(p, w) } -> std::same_as<typename D::state_type>;
      { d.logpdf(st, p, p) } -> std::same_as<double>;
      { d.template logpdf_dmu_dk<1, 0>(st, p, p, idx, w, w) } -> std::same_as<bool>;
      { d.template logpdf_dmu_dk<0, 1>(st, p, p, idx, w, w) } -> std::same_as<bool>;
    };

/**
 * \ingroup distrib_multivariate
 * \brief Host-side sample summary used to compute starting parameter values.
 *
 * Stores the sample size, dimension, sample mean, and sample covariance matrix
 * with divisor \f$n\f$ (packed by rows of the lower triangle).
 */
// What the starting values of a fit read from a sample of n units of
// dimension d: the mean and the covariance with divisor n, packed by rows of
// the lower triangle.
struct distrib_mv_sample_summary {
  /// Number of units \f$n\f$, as a double.
  double count = 0.0;
  /// Vector dimension \f$d\f$ of each unit.
  int dim = 0;
  /// Sample mean vector of length `dim`.
  std::vector<double> mean;
  /// Sample covariance matrix with divisor \f$n\f$, packed by rows of the lower triangle (length `d*(d+1)/2`).
  std::vector<double> covariance;
};

/**
 * \ingroup distrib_multivariate
 * \brief Compute a host-side sample mean and covariance summary from stacked units.
 *
 * Uses a two-pass algorithm on host memory: first computing the sample mean
 * across the `n` units (unit `i` at `y[i * d .. i * d + d - 1]`), then
 * accumulating centred cross products with divisor `n`. With `n = 0` the
 * mean and the covariance are zero.
 *
 * \param y Pointer to `n * d` contiguous sample values (host memory).
 * \param n Number of units.
 * \param d Dimension of each unit.
 * \return Populated `distrib_mv_sample_summary`.
 */
// The summary of y, n units of d values each, unit i at y[i d .. i d + d - 1]
// (host). Two passes: the mean, then the centred cross products.
inline distrib_mv_sample_summary summarize_mv_sample(const double* y, int n, int d) {
  distrib_mv_sample_summary s;
  s.count = n;
  s.dim = d;
  s.mean.assign(d, 0.0);
  s.covariance.assign(pdmat::packed_size(d), 0.0);
  if (n == 0) return s;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) s.mean[k] += y[i * d + k];
  for (int k = 0; k < d; ++k) s.mean[k] /= n;
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < d; ++a) {
      const double ra = y[i * d + a] - s.mean[a];
      for (int b = 0; b <= a; ++b) s.covariance[pdmat::packed_index(a, b)] += ra * (y[i * d + b] - s.mean[b]);
    }
  for (double& c : s.covariance) c /= n;
  return s;
}

/**
 * \ingroup distrib_multivariate
 * \brief Concept for a positive semidefinite matrix structure whose null space is constant in \f$\theta\f$ and supplied.
 *
 * Refines `pdmat::psdmat_structure` by requiring `S::constant_null_space` and
 * `S::supplies_null_space` to be `true`, so that `null_space_basis()` returns
 * a fixed orthonormal basis of the null space. Both the degenerate
 * specialization of `mvgaussian1_distrib` and the improper specialization of
 * `mvgaussian2_distrib` exist over structures satisfying this concept (such as
 * `pdmat::weighted_sum_psdmat`, whereas `pdmat::reduced_rank_psdmat` is excluded).
 *
 * \tparam S Matrix structure type to check.
 * \see mvgaussian1_distrib, mvgaussian2_distrib, distrib_multivariate
 */
template <class S>
concept constant_null_space_psdmat = pdmat::psdmat_structure<S> && S::constant_null_space && S::supplies_null_space;

/**
 * \ingroup distrib_mvgaussian1
 * \brief Precomputed state of a rank-deficient multivariate Gaussian at fixed \f$\theta\f$.
 *
 * Written by `theta_state()` (and, for the improper specialization of
 * `mvgaussian2_distrib`, augmented by `constrained_state()`) into caller-owned
 * workspace `w`. Stores \f$\log\lvert M(\theta)\rvert_+\f$, pointers into `w`
 * for the packed quadratic-form matrix and optional shifted Cholesky factor and
 * inverse, the shift \f$c\f$, optional structure pseudo-determinant state, and
 * optional \f$\theta\f$-derivative table. Both `w` and `theta` must remain
 * alive while the state is in use.
 *
 * \see mvgaussian1_distrib, mvgaussian2_distrib, distrib_multivariate
 */
struct mvgaussian_rank_deficient_state {
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer to caller-owned matrix structure parameter vector \f$\theta\f$ of length `n_params()`.
   */
  const double* theta;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Log pseudo-determinant \f$\log\lvert M(\theta)\rvert_+\f$ (logarithm of the product of the \f$\rho\f$ nonzero eigenvalues).
   */
  double log_det;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the packed lower-triangular matrix of the quadratic form: \f$\Sigma^+\f$ (degenerate) or \f$Q\f$ (improper).
   */
  const double* quadratic;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the packed lower Cholesky factor \f$L\f$ of \f$M + cP\f$, or `nullptr` when produced by `theta_state()` of the improper class.
   */
  const double* factor = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding packed \f$(M + cP)^{-1}\f$, or `nullptr` when produced by `theta_state()` of the improper class.
   */
  const double* shifted_inverse = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Fixed shift \f$c = \operatorname{tr} M / \rho\f$ evaluated at `theta` (`1.0` if \f$\rho = 0\f$ or \f$\operatorname{tr} M \le 0\f$).
   */
  double shift = 1.0;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer into caller workspace holding the structure's evaluated log pseudo-determinant state (read by `log_pdet_dk_at()`), or `nullptr`.
   */
  double* pdet = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Pointer to the precomputed \f$\theta\f$-derivative table attached by `with_theta_table()`, or `nullptr` if none.
   */
  const double* table = nullptr;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Largest \f$\theta\f$-derivative order covered by `table` (`0` if none).
   */
  int table_order = 0;
};

namespace detail_mv {

inline constexpr double log_two_pi = 1.8378770664093454835606594728112353;

// r' P r with r = y - mu and P packed symmetric
LAPLACE_HD double packed_quadratic(int d, const double* P, const double* y, const double* mu) {
  double acc = 0.0;
  for (int i = 0; i < d; ++i) {
    const double ri = y[i] - mu[i];
    double t = 0.5 * P[pdmat::packed_index(i, i)] * ri;
    for (int j = 0; j < i; ++j) t += P[pdmat::packed_index(i, j)] * (y[j] - mu[j]);
    acc += ri * t;
  }
  return 2.0 * acc;
}

// out = c P r, r = y - mu
LAPLACE_HD void packed_times_residual(int d, const double* P, const double* y, const double* mu, double c,
                                             double* out) {
  for (int i = 0; i < d; ++i) {
    double t = 0.0;
    for (int j = 0; j < d; ++j) t += P[i >= j ? pdmat::packed_index(i, j) : pdmat::packed_index(j, i)] * (y[j] - mu[j]);
    out[i] = c * t;
  }
}

// tr(A B) for packed symmetric A and B
LAPLACE_HD double packed_trace_product(int d, const double* A, const double* B) {
  double diag = 0.0, off = 0.0;
  for (int i = 0; i < d; ++i) {
    diag += A[pdmat::packed_index(i, i)] * B[pdmat::packed_index(i, i)];
    for (int j = 0; j < i; ++j) off += A[pdmat::packed_index(i, j)] * B[pdmat::packed_index(i, j)];
  }
  return diag + 2.0 * off;
}

// The lower Cholesky factor of a packed positive definite matrix, in place, by
// rows (Cholesky-Banachiewicz).
LAPLACE_HD void packed_cholesky_in_place(int d, double* a) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = a[pdmat::packed_index(i, j)];
      for (int k = 0; k < j; ++k) s -= a[pdmat::packed_index(i, k)] * a[pdmat::packed_index(j, k)];
      a[pdmat::packed_index(i, j)] = i == j ? Kokkos::sqrt(s) : s / a[pdmat::packed_index(j, j)];
    }
}

// L with M = L L': the structure's own when it has one, else from M
template <class S>
LAPLACE_HD void structure_factor(const S& s, const double* theta, double* l, double* w) {
  if constexpr (S::closed_factor) {
    s.cholesky(theta, l, w);
  } else {
    s.matrix(theta, l, w);
    packed_cholesky_in_place(s.dim, l);
  }
}

// d^k M (Inv false) or d^k M^-1 (Inv true) along idx[0..k-1], k known at run
// time (k = 0: the matrix itself); false when zero by structure
template <bool Inv, int K, class S>
LAPLACE_HD bool structure_matrix_dk(const S& s, const double* t, const int* idx, double* out, double* w) {
  if constexpr (K == 0) {
    if constexpr (Inv) s.inverse(t, out, w);
    else s.matrix(t, out, w);
    return true;
  } else {
    int i[K];
    for (int q = 0; q < K; ++q) i[q] = idx[q];
    if constexpr (Inv) return s.template inverse_dk<K>(t, i, out, w);
    else return s.template matrix_dk<K>(t, i, out, w);
  }
}
template <bool Inv, class S>
LAPLACE_HD bool structure_matrix_runtime(const S& s, int k, const double* t, const int* idx, double* out, double* w) {
  switch (k) {
    case 0: return structure_matrix_dk<Inv, 0>(s, t, idx, out, w);
    case 1: return structure_matrix_dk<Inv, 1>(s, t, idx, out, w);
    case 2: return structure_matrix_dk<Inv, 2>(s, t, idx, out, w);
    case 3: return structure_matrix_dk<Inv, 3>(s, t, idx, out, w);
    case 4: return structure_matrix_dk<Inv, 4>(s, t, idx, out, w);
    case 5: return structure_matrix_dk<Inv, 5>(s, t, idx, out, w);
    default: return structure_matrix_dk<Inv, 6>(s, t, idx, out, w);
  }
}

template <int K, class S>
LAPLACE_HD double structure_log_det_dk(const S& s, const double* t, const int* idx, double* w) {
  if constexpr (K == 0) {
    return s.log_det(t, w);
  } else {
    int i[K];
    for (int q = 0; q < K; ++q) i[q] = idx[q];
    return s.template log_det_dk<K>(t, i, w);
  }
}
template <class S>
LAPLACE_HD double structure_log_det_runtime(const S& s, int k, const double* t, const int* idx, double* w) {
  switch (k) {
    case 0: return structure_log_det_dk<0>(s, t, idx, w);
    case 1: return structure_log_det_dk<1>(s, t, idx, w);
    case 2: return structure_log_det_dk<2>(s, t, idx, w);
    case 3: return structure_log_det_dk<3>(s, t, idx, w);
    case 4: return structure_log_det_dk<4>(s, t, idx, w);
    case 5: return structure_log_det_dk<5>(s, t, idx, w);
    default: return structure_log_det_dk<6>(s, t, idx, w);
  }
}

// merge of two sorted tuples into out (sorted), returns its length
LAPLACE_HD int merge_sorted(const int* a, int na, const int* b, int nb, int* out) {
  int i = 0, j = 0, k = 0;
  while (i < na || j < nb) out[k++] = (j >= nb || (i < na && a[i] <= b[j])) ? a[i++] : b[j++];
  return k;
}

}  // namespace detail_mv
}  // namespace laplace::distrib
