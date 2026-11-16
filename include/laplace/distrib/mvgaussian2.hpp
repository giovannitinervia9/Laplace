// mvgaussian2_distrib<S>: the multivariate Gaussian in the precision form,
// Q = M(theta) a structured positive definite matrix S (laplace::pdmat);
// note/distrib_multivariate.md, interface in distrib/multivariate_core.hpp.
//
//   l = -d/2 log 2 pi + 1/2 log det Q - 1/2 r' Q r,   r = y - mu.
//
// The state holds log det Q (the structure's log_det), Q (its matrix) and the
// factor L, Q = L L' (its cholesky, or the Cholesky factorization of its matrix
// when the structure has none), and points at theta. The derivative blocks in
// (mu, theta) come from matrix_dk and log_det_dk of the structure; Q^-1 is
// needed only by the expected information in theta and by covariance().
#pragma once

#include <laplace/distrib/detail/mvgaussian_common.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_multivariate
 * \brief Precomputed state for \c mvgaussian2_distrib at fixed matrix parameters \f$\theta\f$.
 *
 * Holds the parameter pointer together with \f$\log\det Q(\theta)\f$ and
 * pointers into caller-supplied storage for packed \f$Q(\theta)\f$ and the
 * lower-triangular Cholesky factor \f$L\f$ satisfying \f$Q = L L^\top\f$.
 */
struct mvgaussian2_state {
  /// Matrix structure parameter vector \f$\theta\f$ (caller-owned).
  const double* theta;       // the parameters of the structure (the caller's)
  /// Log determinant \f$\log\det Q(\theta)\f$.
  double log_det;            // log det Q
  /// Packed lower-triangular precision matrix \f$Q(\theta)\f$.
  const double* quadratic;   // Q, packed
  /// Packed lower-triangular Cholesky factor \f$L\f$ with \f$Q = L L^\top\f$.
  const double* factor;      // L with Q = L L', packed
};

/**
 * \ingroup distrib_multivariate
 * \brief Multivariate Gaussian distribution parameterized by precision \f$Q(\theta)\f$.
 *
 * Models \f$Y \sim \mathcal{N}_d(\mu, Q(\theta)^{-1})\f$ with
 * \f$Q(\theta)\f$ defined by a positive definite matrix structure `S`
 * satisfying `pdmat::pdmat_structure`. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 */
template <pdmat::pdmat_structure S>
struct mvgaussian2_distrib {
  /// Positive definite matrix structure parameterizing the precision \f$Q(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite precision).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Precomputed state type returned by `theta_state`.
  using state_type = mvgaussian2_state;

  /// Default constructor.
  mvgaussian2_distrib() = default;
  /**
   * \brief Construct from a matrix structure instance.
   * \param s Positive definite matrix structure defining \f$Q(\theta)\f$.
   */
  LAPLACE_HD explicit mvgaussian2_distrib(const S& s) : structure(s) {}

  /**
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  LAPLACE_HD int dim() const { return structure.dim; }
  /**
   * \brief Return the number of precision parameters in \f$\theta\f$.
   * \return Parameter count `structure.n_params()`.
   */
  LAPLACE_HD int n_params() const { return structure.n_params(); }
  /**
   * \brief Return the number of doubles needed to store `quadratic` and `factor` in `mvgaussian2_state`.
   * \return `2 * pdmat::packed_size(structure.dim)`.
   */
  LAPLACE_HD int state_size() const { return 2 * pdmat::packed_size(structure.dim); }
  /**
   * \brief Return total workspace doubles needed for `theta_state` and evaluation/derivative calls.
   * \return `2 * pdmat::packed_size(structure.dim) + structure.scratch_size()`.
   */
  LAPLACE_HD int scratch_size() const { return 2 * pdmat::packed_size(structure.dim) + structure.scratch_size(); }

  /**
   * \brief Precompute \f$\log\det Q(\theta)\f$, \f$Q(\theta)\f$, and Cholesky factor \f$L\f$ (\f$Q = L L^\top\f$).
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `quadratic` and `factor` while the state is in use.
   * \return Populated `mvgaussian2_state`.
   */
  LAPLACE_HD mvgaussian2_state theta_state(const double* theta, double* w) const {
    return detail_mvgaussian::theta_state<true, mvgaussian2_state>(structure, theta, w);
  }
  /**
   * \brief Evaluate the multivariate Gaussian log density \f$\log p(y \mid \mu, \theta)\f$.
   *
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$-\frac{d}{2}\log(2\pi) + \frac{1}{2}\log\det Q - \frac{1}{2}(y-\mu)^\top Q(y-\mu)\f$.
   */
  LAPLACE_HD double logpdf(const mvgaussian2_state& st, const double* y, const double* mu) const {
    return detail_mvgaussian::logpdf<true>(structure, st, y, mu);
  }
  /**
   * \brief Evaluate partial derivatives of the log density with respect to \f$\mu\f$ (order `A`) and \f$\theta\f$ (order `K`).
   *
   * Supports \f$A, K \ge 0\f$ with \f$A + K \le 6\f$. Blocks with \f$A \ge 3\f$
   * and structurally zero blocks return `false` without modifying `out`.
   *
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), or `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if `A <= 2` and the block is not structurally zero; `false` otherwise.
   */
  template <int A, int K>
  LAPLACE_HD bool logpdf_dmu_dk(const mvgaussian2_state& st, const double* y, const double* mu, const int* idx,
                                double* out, double* w) const {
    return detail_mvgaussian::logpdf_dmu_dk<true, A, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate partial derivatives of the log density with respect to \f$y\f$ (order `J`) and \f$\theta\f$ (order `K`).
   *
   * Supports \f$J, K \ge 0\f$ with \f$J + K \le 6\f$. Blocks with \f$J \ge 3\f$
   * and structurally zero blocks return `false` without modifying `out`.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `J + K <= 6`).
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), or `pdmat::packed_size(dim())` (`J = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if `J <= 2` and the block is not structurally zero; `false` otherwise.
   */
  template <int J, int K>
  LAPLACE_HD bool logpdf_dy_dk(const mvgaussian2_state& st, const double* y, const double* mu, const int* idx,
                               double* out, double* w) const {
    return detail_mvgaussian::logpdf_dy_dk<true, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate the expected Fisher information matrix block or its `K`-th derivative with respect to \f$\theta\f$.
   *
   * Supports \f$K \in \{0, \dots, 4\}\f$ across `vector_vector`, `vector_rest`, and `rest_rest`.
   *
   * \tparam K Derivative order with respect to \f$\theta\f$ (`0 <= K <= 4`).
   * \param st Precomputed state from `theta_state`.
   * \param block Block of the expected information matrix.
   * \param ab Two-element index pair `(a, b)` selecting an entry within `rest_rest` (unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\theta\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (packed lower-triangular) for `vector_vector`, or length `1` (`out[0]`) for `rest_rest`.
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `vector_rest` or structurally zero derivatives.
   */
  template <int K>
  LAPLACE_HD bool expected_info_dk(const mvgaussian2_state& st, distrib_mv_info_block block, const int* ab,
                                   const int* idx, double* out, double* w) const {
    return detail_mvgaussian::expected_info_dk<true, K>(structure, st, block, ab, idx, out, w);
  }
  /**
   * \brief Draw one sample \f$y \sim \mathcal{N}_d(\mu, Q^{-1})\f$ using counter-based normal deviates.
   *
   * Computes \f$y = \mu + L^{-\top} z\f$ by backward substitution with `st.factor`
   * (\f$Q = L L^\top\f$) and independent standard normals generated via
   * `standard_normal_by_inversion` and `counter_uniform_source`.
   *
   * \param st Precomputed state from `theta_state`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   */
  LAPLACE_HD void rng(const mvgaussian2_state& st, const double* mu, std::uint64_t seed, std::uint64_t unit,
                      double* y) const {
    detail_mvgaussian::rng<true>(structure, st, mu, seed, unit, y);
  }
  /**
   * \brief Compute starting matrix parameters `theta` on the host from a sample summary.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample`.
   * \param theta Output parameter vector of length `n_params()`.
   */
  // (host) theta from a sample summary (summarize_mv_sample)
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    detail_mvgaussian::start_value<true>(structure, summary, theta);
  }
  /**
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out`.
   *
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   */
  LAPLACE_HD void mean(const double* mu, double* out) const {
    for (int k = 0; k < structure.dim; ++k) out[k] = mu[k];
  }
  /**
   * \brief Write the covariance matrix \f$Q(\theta)^{-1}\f$ in packed lower-triangular order.
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  // Q^-1, packed; w has structure.scratch_size() doubles
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const { structure.inverse(theta, out, w); }
};

}  // namespace laplace::distrib
