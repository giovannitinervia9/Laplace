// toeplitz_pdmat: the p x p Toeplitz correlation matrix M_ij = m_|i-j|, m_0 = 1,
// parametrized by its p - 1 partial autocorrelations phi_k = tanh(theta_(k-1)),
// k = 1..p-1 (note/pdmat_structures.md; interface in core.hpp). Every real
// theta gives a positive definite M. Below c_k = sech(theta_(k-1)) =
// (1 - phi_k^2)^(1/2).
//
// M and L by the normalized lattice. For the process x_t with correlations m,
// let e_k(t) and b_k(t) be the forward and backward prediction errors of order
// k (x_t less its projection on x_(t-1), ..., x_(t-k), and x_(t-k) less its
// projection on x_(t-k+1), ..., x_t), both of variance v_k = prod_(j<=k) c_j^2,
// and E_k(t), B_k(t) the same divided by v_k^(1/2). With eps_t = E_t(t), the
// standardized innovations, x_t = sum_s L_ts eps_s with L the Cholesky factor of
// M, and the Durbin-Levinson lattice e_k(t) = e_(k-1)(t) - phi_k b_(k-1)(t-1),
// b_k(t) = b_(k-1)(t-1) - phi_k e_(k-1)(t), solved for the lower order, is the
// rotation
//   E_(k-1)(t) = c_k E_k(t) + phi_k B_(k-1)(t-1),
//   B_k(t)     = c_k B_(k-1)(t-1) - phi_k E_k(t),
// orthogonal since c_k^2 + phi_k^2 = 1. At each time t it runs from E_t(t) =
// eps_t down to E_0(t) = B_0(t) = x_t, k = t, ..., 1, and stores the B_k(t)
// for time t + 1. Each column s of L, the coefficients of eps_s, follows the
// same rotations from 1 at time s; m_t = L_t0 (M = L L^T and L_0s = delta_0s).
// The rotations keep the small factors of the result as products: the
// derivative of m_n along phi_k is v_(k-1) times a polynomial in phi, which
// the Durbin-Levinson recursion of m forms as a difference of terms of order
// one and loses at |phi_j| near 1 (5e7 ulpN at theta_0 = 6, against 3.3 for the
// lattice; docs/numerics.md).
//
// M^-1 by the Durbin-Levinson coefficients, with v_0 = 1, for k = 1..p-1:
//   phi_(k,k) = phi_k,  phi_(k,j) = phi_(k-1,j) - phi_k phi_(k-1,k-j)  (j < k).
// With A the unit lower triangular matrix whose row t holds A_(t,t-j) =
// -phi_(t,j) and V = diag(v_0, ..., v_(p-1)), A M A^T = V, so
//   M^-1 = A^T V^-1 A, M^-1_ij = sum_(t >= i) A_ti A_tj / v_t  (i >= j),
// with 1/v_t = prod_(a<t) cosh^2 theta_a and its derivatives in closed form,
// never divided; and
//   log det M = sum_k (p - k) log(1 - phi_k^2) = -2 sum_k (p - k) log cosh theta_(k-1).
//
// Derivatives along a sorted tuple idx of K indices: every quantity keeps its
// derivatives along the 2^K sub-masks of the tuple, and each product is
// differentiated by Leibniz's rule over the subsets (as cholesky_subtuple in
// bijector/coupled.hpp); the elementary factors tanh, sech and cosh^2 of
// theta_a have their derivatives written directly in theta. Equal indices of the
// tuple are interchangeable, so only one mask of each class is computed (see
// classes below): a tuple with repeated indices costs prod (m_v + 1) masks, m_v
// the multiplicities, and a pure tuple of order K only K + 1. A quantity of time
// or stage t depends on theta_0..theta_(t-1) only, so a mask that holds a later
// index is zero and is not computed. At order K each product costs about 3^K
// operations: about 2 p^2 products for M, 2 p^3/3 for L, p^3/6 for M^-1.
//
// For p = 1, M = [1], there are no parameters and no tuples. For p = 2,
// M = [[1, tanh theta_0], [., 1]], the matrix of ar1_pdmat(2); with theta =
// (t, 0, ..., 0) M is the AR(1) matrix of ar1_pdmat(p) for every p.
#pragma once

#include <laplace/bijector/coupled.hpp>
#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>
#include <laplace/pdmat/detail/tuple_classes.hpp>

namespace laplace::pdmat {

namespace detail_toeplitz {

namespace dc = laplace::bijector::detail_coupled;
namespace dp = detail_pdmat;

// the positions q with idx[q] < k, the first ones since idx is sorted, as a mask
LAPLACE_HD_FORCEINLINE int below_mask(const int* idx, int K, int k) {
  int c = 0;
  while (c < K && idx[c] < k) ++c;
  return (1 << c) - 1;
}

// the size of S if every position of S holds the index a, else -1
LAPLACE_HD_FORCEINLINE int order_on(const int* idx, int K, int S, int a) {
  int n = 0;
  for (int q = 0; q < K; ++q)
    if (S & (1 << q)) {
      if (idx[q] != a) return -1;
      ++n;
    }
  return n;
}

// Equal indices of the tuple are interchangeable: one mask per class of equal
// sub-masks, Leibniz over the classes (detail/tuple_classes.hpp).
template <int K>
using classes = dp::tuple_classes<K>;

enum class elementary { tanh, cosh2, sech };

// d^n f(theta), n = 0..6, for the elementary functions f of the recursion
LAPLACE_HD double elementary_d(elementary f, int n, double theta, const dp::tanh_parts& t) {
  switch (f) {
    case elementary::tanh: return n == 0 ? t.T : dp::tanh_power_d(n, 1, 0, t);
    case elementary::cosh2: {
      if (n == 0) {
        const double c = Kokkos::cosh(theta);
        return c * c;
      }
      // cosh^2 = (1 + cosh 2 theta)/2
      const double h = 0.5 * dp::ipow(2.0, n);
      return h * (n % 2 == 1 ? Kokkos::sinh(2.0 * theta) : Kokkos::cosh(2.0 * theta));
    }
    default: return dp::tanh_power_d(n, 0, 1, t);
  }
}

// e[a NM + S] = d_S f(theta_a), a = 0..p-2
template <int K>
LAPLACE_HD void fill_elementary(int p, const double* theta, const int* idx, elementary f, double* e) {
  constexpr int NM = 1 << K;
  for (int a = 0; a + 1 < p; ++a) {
    const auto t = dp::tanh_of(theta[a]);
    double d[K + 1];
    d[0] = elementary_d(f, 0, theta[a], t);
    bool used = false;
    for (int q = 0; q < K; ++q) used = used || idx[q] == a;
    for (int n = 1; n <= K; ++n) d[n] = used ? elementary_d(f, n, theta[a], t) : 0.0;
    for (int S = 0; S < NM; ++S) {
      const int n = order_on(idx, K, S, a);
      e[a * NM + S] = n < 0 ? 0.0 : d[n];
    }
  }
}

// c_t = prod_(a<t) f(theta_a), t = 0..p-1, from e of fill_elementary
template <int K>
LAPLACE_HD void product_chain(int p, const int* idx, const classes<K>& cl, const double* e, double* c) {
  constexpr int NM = 1 << K;
  for (int S = 0; S < NM; ++S) c[S] = S == 0 ? 1.0 : 0.0;
  for (int t = 1; t < p; ++t) {
    const int low = below_mask(idx, K, t);
    for (int S = 0; S < NM; ++S)
      c[t * NM + S] = (S & ~low) ? 0.0 : cl.rep[S] != S ? c[t * NM + cl.rep[S]] : cl.lb(c + (t - 1) * NM, e + (t - 1) * NM, S);
  }
}

// the first of the p (p - 1)/2 coefficients phi_(k,j), j = 1..k, of row k
LAPLACE_HD_FORCEINLINE int row_start(int k) { return k * (k - 1) / 2; }

// the Durbin-Levinson coefficients: r[(row_start(k) + j - 1) NM + S] = d_S phi_(k,j)
template <int K>
LAPLACE_HD void coefficients(int p, const int* idx, const classes<K>& cl, const double* phi, double* r) {
  constexpr int NM = 1 << K;
  for (int k = 1; k < p; ++k) {
    const int low = below_mask(idx, K, k);
    double* R = r + row_start(k) * NM;
    const double* Rp = r + row_start(k - 1) * NM;
    const double* f = phi + (k - 1) * NM;
    for (int S = 0; S < NM; ++S) {
      if (S & ~low) {
        for (int j = 0; j < k; ++j) R[j * NM + S] = 0.0;
        continue;
      }
      R[(k - 1) * NM + S] = f[S];
      if (cl.rep[S] != S) {
        for (int j = 1; j < k; ++j) R[(j - 1) * NM + S] = R[(j - 1) * NM + cl.rep[S]];
        continue;
      }
      for (int j = 1; j < k; ++j) R[(j - 1) * NM + S] = Rp[(j - 1) * NM + S] - cl.lb(f, Rp + (k - j - 1) * NM, S);
    }
  }
}

// scratch: two elementary arrays, the coefficients, two chains of p and one
// strict lower triangle, each entry with its NM masks (the lattice needs less)
LAPLACE_HD constexpr int scratch_entries(int p) { return 2 * (p - 1) + p * (p - 1) + 2 * p; }

// column s of L along the tuple by the normalized lattice: out[packed_index(t, s)]
// = d_F L_ts, t = s..p-1. B holds B_k at the previous time, k = 0..p-1, and
// E, En the forward error before and after a rotation, each with its NM masks.
template <int K>
LAPLACE_HD void lattice_column(int p, int s, const int* idx, const classes<K>& cl, const double* phi, const double* c, double* B, double* E,
                               double* En, double* out) {
  constexpr int NM = 1 << K, F = NM - 1;
  for (int e = 0; e < s * NM; ++e) B[e] = 0.0;
  for (int t = s; t < p; ++t) {
    const int low = below_mask(idx, K, t);
    for (int S = 0; S < NM; ++S) E[S] = (t == s && S == 0) ? 1.0 : 0.0;
    for (int k = t; k >= 1; --k) {
      const double* f = phi + (k - 1) * NM;
      const double* g = c + (k - 1) * NM;
      const double* b = B + (k - 1) * NM;  // B_(k-1)(t-1)
      double* bn = B + k * NM;             // B_k(t), over B_k(t-1), read at k + 1
      for (int S = 0; S < NM; ++S) {
        if (S & ~low) {
          bn[S] = En[S] = 0.0;
          continue;
        }
        if (cl.rep[S] != S) {
          bn[S] = bn[cl.rep[S]], En[S] = En[cl.rep[S]];
          continue;
        }
        bn[S] = cl.lb(g, b, S) - cl.lb(f, E, S);
        En[S] = cl.lb(g, E, S) + cl.lb(f, b, S);
      }
      double* x = E;
      E = En, En = x;
    }
    for (int S = 0; S < NM; ++S) B[S] = E[S];  // B_0(t) = x_t
    out[packed_index(t, s)] = E[F];
  }
}

// M along the tuple into out: column 0 of L, then its bands
template <int K>
LAPLACE_HD void matrix_masks(int p, const double* theta, const int* idx, double* out, double* w) {
  constexpr int NM = 1 << K;
  const classes<K> cl(idx);
  double* phi = w;
  double* c = phi + (p - 1) * NM;
  double* B = c + (p - 1) * NM;
  double* E = B + p * NM;
  fill_elementary<K>(p, theta, idx, elementary::tanh, phi);
  fill_elementary<K>(p, theta, idx, elementary::sech, c);
  lattice_column<K>(p, 0, idx, cl, phi, c, B, E, E + NM, out);
  for (int k = 0; k < p; ++k)
    for (int j = 1; j + k < p; ++j) out[packed_index(j + k, j)] = out[packed_index(k, 0)];
}

// M^-1 = A^T V^-1 A along the tuple into out
template <int K>
LAPLACE_HD void inverse_masks(int p, const double* theta, const int* idx, double* out, double* w) {
  constexpr int NM = 1 << K, F = NM - 1;
  const classes<K> cl(idx);
  double* phi = w;
  double* c = phi + (p - 1) * NM;
  double* r = c + (p - 1) * NM;
  double* iv = r + row_start(p) * NM;  // 1/v_t
  double* G = iv + 2 * p * NM;         // G_tj = A_tj / v_t, j < t
  fill_elementary<K>(p, theta, idx, elementary::tanh, phi);
  fill_elementary<K>(p, theta, idx, elementary::cosh2, c);
  coefficients<K>(p, idx, cl, phi, r);
  product_chain<K>(p, idx, cl, c, iv);
  for (int t = 1; t < p; ++t) {
    const int low = below_mask(idx, K, t);
    for (int j = 0; j < t; ++j) {
      double* g = G + dc::strict(t, j) * NM;
      const double* a = r + (row_start(t) + t - j - 1) * NM;
      for (int S = 0; S < NM; ++S) g[S] = (S & ~low) ? 0.0 : cl.rep[S] != S ? g[cl.rep[S]] : -cl.lb(a, iv + t * NM, S);
    }
  }
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double x = i == j ? iv[i * NM + F] : G[dc::strict(i, j) * NM + F];
      for (int t = i + 1; t < p; ++t) x -= cl.lb(r + (row_start(t) + t - i - 1) * NM, G + dc::strict(t, j) * NM, F);
      out[packed_index(i, j)] = x;
    }
}

// L along the tuple into out, one column at a time
template <int K>
LAPLACE_HD void cholesky_masks(int p, const double* theta, const int* idx, double* out, double* w) {
  constexpr int NM = 1 << K;
  const classes<K> cl(idx);
  double* phi = w;
  double* c = phi + (p - 1) * NM;
  double* B = c + (p - 1) * NM;
  double* E = B + p * NM;
  fill_elementary<K>(p, theta, idx, elementary::tanh, phi);
  fill_elementary<K>(p, theta, idx, elementary::sech, c);
  for (int s = 0; s < p; ++s) lattice_column<K>(p, s, idx, cl, phi, c, B, E, E + NM, out);
}

}  // namespace detail_toeplitz

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with Toeplitz correlation structure.
 *
 * Maps \f$q = p - 1\f$ unconstrained parameters \f$\theta_0, \dots, \theta_{p-2}\f$ (\f$p \ge 1\f$) to a \f$p \times p\f$
 * Toeplitz correlation matrix \f$M_{ij}(\theta) = m_{|i - j|}\f$ (\f$m_0 = 1\f$) with partial autocorrelations
 * \f$\varphi_k = \tanh\theta_{k-1} \in (-1, 1)\f$ for \f$k = 1, \dots, p - 1\f$ (\f$M = [1]\f$ at \f$p = 1\f$).
 *
 * \see pdmat, pdmat_orders, pdmat_structure, ar1_pdmat
 */
struct toeplitz_pdmat : pdmat_orders<toeplitz_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;

  /// \brief Compile-time descriptor: \c true, diagonal entries are identically 1.
  static constexpr bool unit_diagonal = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form inverse is implemented.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Cholesky factor is implemented.
  static constexpr bool closed_factor = true;
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  toeplitz_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs a Toeplitz structured matrix of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit toeplitz_pdmat(int p) : dim(p) {}

  /**
   * \ingroup pdmat
   * \brief Number of parameters \f$q = p - 1\f$.
   * \return Number of parameters \f$p - 1\f$ (\f$0\f$ for \f$p = 1\f$).
   */
  LAPLACE_HD int n_params() const { return dim - 1; }

  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch any function needs: \f$64(p^2 + 3p - 2)\f$ doubles for \f$p \ge 2\f$, \f$0\f$ for \f$p = 1\f$.
   * \return Required scratch buffer size in doubles.
   */
  LAPLACE_HD int scratch_size() const {
    return dim == 1 ? 0 : bijector::detail_coupled::max_masks * detail_toeplitz::scratch_entries(dim);
  }

  /**
   * \ingroup pdmat
   * \brief Computes starting parameters \f$\theta\f$ from a target packed covariance matrix \f$\hat{M}\f$.
   *
   * Host-only function. Forms the correlation matrix of \f$\hat{M}\f$ after flooring
   * diagonal entries at \f$10^{-6}\f$ times the largest, averages each subdiagonal \f$k\f$
   * to obtain \f$m_k\f$, recovers partial autocorrelations \f$\varphi_k\f$ via the
   * Durbin-Levinson recursion (Durbin, 1960), clamps each to \f$[-0.99, 0.99]\f$, and
   * sets \f$\theta_{k-1} = \operatorname{atanh}\varphi_k\f$.
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output array of \c n_params() doubles inside the domain.
   *
   * Accuracy: recovers \f$\theta\f$ from `matrix()` within \f$1.7 \times 10^{-16}\f$ on gcc 14.2/OpenMP (test bound \f$10^{-13}\f$).
   *
   * \see matrix(), pdmat
   */
  // host: m_k the mean correlation on diagonal k of m_hat, then the partial
  // autocorrelations by the Durbin-Levinson recursion, each clamped
  void params_from_matrix(const double* m_hat, double* theta) const {
    const auto r = detail_pdmat::correlation_of(dim, m_hat);
    std::vector<double> mk(dim, 1.0), a(dim, 0.0), b(dim, 0.0);
    for (int k = 1; k < dim; ++k) {
      double s = 0.0;
      for (int i = k; i < dim; ++i) s += r[packed_index(i, i - k)];
      mk[k] = s / (dim - k);
    }
    double v = 1.0;
    for (int k = 1; k < dim; ++k) {
      double num = mk[k];
      for (int j = 1; j < k; ++j) num -= a[j] * mk[k - j];
      const double phi = detail_pdmat::clamp_rho(num / v);
      for (int j = 1; j < k; ++j) b[j] = a[j] - phi * a[k - j];
      for (int j = 1; j < k; ++j) a[j] = b[j];
      a[k] = phi;
      v *= 1.0 - phi * phi;
      theta[k - 1] = std::atanh(phi);
    }
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed Toeplitz correlation matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ij} = m_{|i - j|}\f$ with \f$m_t = L_{t0}\f$ via normalized lattice rotations (\f$M = [1]\f$ for \f$p = 1\f$).
   * Maximum error over orders 0–6: 163 ulpN on gcc and 135 ulpN on CUDA; test bound 2 ulpN at order 0.
   *
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$ (unused for \f$p = 1\f$).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused for \f$p = 1\f$).
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const {
    if (dim == 1) {
      m[0] = 1.0;
      return;
    }
    detail_toeplitz::matrix_masks<0>(dim, theta, nullptr, m, w);
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * Computes \f$\log \det M(\theta) = -2 \sum_{a=0}^{p-2} (p - 1 - a)\log\cosh\theta_a\f$ (\f$0.0\f$ for \f$p = 1\f$).
   * Maximum error over orders 0–6: 22.8 ulp on gcc and 19.9 ulp on CUDA; test bound 3 ulp at order 0.
   *
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$ (unused for \f$p = 1\f$).
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused).
   * \return Log-determinant of \f$M(\theta)\f$.
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const {
    double s = 0.0;
    for (int a = 0; a + 1 < dim; ++a) {
      const double x = theta[a] < 0.0 ? -theta[a] : theta[a];
      const double log_cosh = x + Kokkos::log1p(detail_pdmat::m::exp(-2.0 * x)) - 0.69314718055994530942;
      s += (dim - 1 - a) * log_cosh;
    }
    return -2.0 * s;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Evaluates \f$M^{-1} = A^\top V^{-1} A\f$ from Durbin-Levinson coefficients \f$\varphi_{t, j}\f$
   * and \f$1/v_t = \prod_{a < t} \cosh^2\theta_a\f$ (\f$M^{-1} = [1]\f$ for \f$p = 1\f$).
   * Maximum error over orders 0–6: 10.0 ulpN on gcc and 13.9 ulpN on CUDA; test bound 15 ulpN at order 0.
   *
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$ (unused for \f$p = 1\f$).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused for \f$p = 1\f$).
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, double* w) const {
    if (dim == 1) {
      m[0] = 1.0;
      return;
    }
    detail_toeplitz::inverse_masks<0>(dim, theta, nullptr, m, w);
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Evaluates \f$L\f$ satisfying \f$M = L L^\top\f$ column by column via normalized lattice rotations (\f$L = [1]\f$ for \f$p = 1\f$).
   * Maximum error over orders 0–6: 10.9 ulpN on gcc and 10.3 ulpN on CUDA; test bound 3 ulpN at order 0.
   *
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$ (unused for \f$p = 1\f$).
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused for \f$p = 1\f$).
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, double* w) const {
    if (dim == 1) {
      l[0] = 1.0;
      return;
    }
    detail_toeplitz::cholesky_masks<0>(dim, theta, nullptr, l, w);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Differentiates column 0 of the normalized lattice by Leibniz's rule over sub-masks; returns \c false for \f$p = 1\f$.
   * Maximum error over orders 1–6: 163 ulpN on gcc and 135 ulpN on CUDA; test bounds 12 to 244 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, p - 2\}\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    if (dim == 1) return false;
    detail_toeplitz::matrix_masks<K>(dim, theta, idx, m, w);
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$.
   *
   * For \f$p \ge 2\f$ and a pure tuple of index \f$a\f$, computes \f$-2(p - 1 - a)\tanh^{(K-1)}(\theta_a)\f$;
   * returns \f$0.0\f$ for \f$p = 1\f$ or on mixed tuples.
   * Maximum error over orders 1–6: 22.8 ulp on gcc and 19.9 ulp on CUDA; test bounds 2 to 35 ulp.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, p - 2\}\f$.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused).
   * \return Value of the derivative, or \f$0.0\f$ if \f$p = 1\f$ or on a mixed tuple.
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], [[maybe_unused]] double* w) const {
    if (dim == 1 || idx[0] != idx[K - 1]) return 0.0;
    const int a = idx[0];
    return -2.0 * (dim - 1 - a) * detail_pdmat::tanh_power_d(K - 1, 1, 0, detail_pdmat::tanh_of(theta[a]));
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * Differentiates \f$M^{-1} = A^\top V^{-1} A\f$ over sub-masks; returns \c false for \f$p = 1\f$.
   * Maximum error over orders 1–6: 10.0 ulpN on gcc and 13.9 ulpN on CUDA; test bounds 14 to 21 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, p - 2\}\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    if (dim == 1) return false;
    detail_toeplitz::inverse_masks<K>(dim, theta, idx, m, w);
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$.
   *
   * Differentiates the normalized lattice column by column over sub-masks; returns \c false for \f$p = 1\f$.
   * Maximum error over orders 1–6: 10.9 ulpN on gcc and 10.3 ulpN on CUDA; test bounds 7 to 17 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p - 1\f$ parameters \f$\theta_0, \dots, \theta_{p-2}\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, p - 2\}\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, const int (&idx)[K], double* l, double* w) const {
    if (dim == 1) return false;
    detail_toeplitz::cholesky_masks<K>(dim, theta, idx, l, w);
    return true;
  }
};

static_assert(pdmat_structure<toeplitz_pdmat>);

}  // namespace laplace::pdmat
