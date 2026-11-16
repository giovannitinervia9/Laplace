// mvgaussian1_distrib<S>: the multivariate Gaussian in the covariance form,
// Sigma = M(theta) a structured positive definite matrix S (laplace::pdmat);
// note/distrib_multivariate.md, interface in distrib/multivariate_core.hpp.
//
//   l = -d/2 log 2 pi - 1/2 log det Sigma - 1/2 r' Sigma^-1 r,   r = y - mu.
//
// The state holds log det Sigma (the structure's log_det), Sigma^-1 (its
// inverse) and the factor L, Sigma = L L' (its cholesky, or the Cholesky
// factorization of its matrix when the structure has none), and points at
// theta. The derivative blocks in (mu, theta) come from inverse_dk and
// log_det_dk of the structure; those of order 3 and more in mu are zero.
#pragma once

#include <laplace/distrib/detail/mvgaussian_common.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_multivariate
 * \brief Precomputed state for \c mvgaussian1_distrib at fixed matrix parameters \f$\theta\f$.
 *
 * Holds the parameter pointer together with \f$\log\det\Sigma(\theta)\f$ and
 * pointers into caller-supplied storage for packed \f$\Sigma(\theta)^{-1}\f$ and
 * the lower-triangular Cholesky factor \f$L\f$ satisfying \f$\Sigma = L L^\top\f$.
 */
struct mvgaussian1_state {
  /// Matrix structure parameter vector \f$\theta\f$ (caller-owned).
  const double* theta;       // the parameters of the structure (the caller's)
  /// Log determinant \f$\log\det\Sigma(\theta)\f$.
  double log_det;            // log det Sigma
  /// Packed lower-triangular precision matrix \f$\Sigma(\theta)^{-1}\f$.
  const double* quadratic;   // Sigma^-1, packed
  /// Packed lower-triangular Cholesky factor \f$L\f$ with \f$\Sigma = L L^\top\f$.
  const double* factor;      // L with Sigma = L L', packed
  /**
   * \ingroup distrib_multivariate
   * \brief Pointer to the precomputed \f$\theta\f$-derivative table attached by `with_theta_table()`, or `nullptr` if none.
   */
  const double* table = nullptr;  // the theta table (with_theta_table), or none
  /**
   * \ingroup distrib_multivariate
   * \brief Largest derivative order covered by the attached \f$\theta\f$-derivative table (0 if none).
   */
  int table_order = 0;            // its largest order
};

/**
 * \ingroup distrib_multivariate
 * \brief Multivariate Gaussian distribution parameterized by covariance \f$\Sigma(\theta)\f$.
 *
 * Models \f$Y \sim \mathcal{N}_d(\mu, \Sigma(\theta))\f$ with
 * \f$\Sigma(\theta)\f$ defined by a positive definite matrix structure `S`
 * satisfying `pdmat::pdmat_structure`. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 */
template <pdmat::pdmat_structure S>
struct mvgaussian1_distrib {
  /// Positive definite matrix structure parameterizing the covariance \f$\Sigma(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite covariance).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Precomputed state type returned by `theta_state`.
  using state_type = mvgaussian1_state;

  /// Default constructor.
  mvgaussian1_distrib() = default;
  /**
   * \brief Construct from a matrix structure instance.
   * \param s Positive definite matrix structure defining \f$\Sigma(\theta)\f$.
   */
  LAPLACE_HD explicit mvgaussian1_distrib(const S& s) : structure(s) {}

  /**
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  LAPLACE_HD int dim() const { return structure.dim; }
  /**
   * \brief Return the number of covariance parameters in \f$\theta\f$.
   * \return Parameter count `structure.n_params()`.
   */
  LAPLACE_HD int n_params() const { return structure.n_params(); }
  /**
   * \brief Return the number of doubles needed to store `quadratic` and `factor` in `mvgaussian1_state`.
   * \return `2 * pdmat::packed_size(structure.dim)`.
   */
  LAPLACE_HD int state_size() const { return 2 * pdmat::packed_size(structure.dim); }
  /**
   * \brief Return total workspace doubles needed for `theta_state` and evaluation/derivative calls.
   * \return `2 * pdmat::packed_size(structure.dim) + structure.scratch_size()`.
   */
  LAPLACE_HD int scratch_size() const { return 2 * pdmat::packed_size(structure.dim) + structure.scratch_size(); }
  /**
   * \ingroup distrib_multivariate
   * \brief Return workspace doubles needed in `w` for `expected_info_dk()`.
   *
   * Returns `2 * pdmat::packed_size(dim()) + structure.scratch_size()`, or the
   * larger of that size and `113 * dim() * dim() + pdmat::packed_size(dim()) + structure.scratch_size()`
   * when `S::closed_factor` is `true`.
   *
   * \return Number of `double` elements required in `w` by `expected_info_dk()`.
   * \see expected_info_dk(), scratch_size(), distrib_multivariate
   */
  // doubles of w of expected_info_dk
  LAPLACE_HD int info_scratch_size() const { return detail_mvgaussian::info_scratch_size<false>(structure); }

  /**
   * \brief Precompute \f$\log\det\Sigma(\theta)\f$, \f$\Sigma(\theta)^{-1}\f$, and Cholesky factor \f$L\f$ (\f$\Sigma = L L^\top\f$).
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `quadratic` and `factor` while the state is in use.
   * \return Populated `mvgaussian1_state`.
   */
  LAPLACE_HD mvgaussian1_state theta_state(const double* theta, double* w) const {
    return detail_mvgaussian::theta_state<false, mvgaussian1_state>(structure, theta, w);
  }
  /**
   * \brief Evaluate the multivariate Gaussian log density \f$\log p(y \mid \mu, \theta)\f$.
   *
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$-\frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(y-\mu)^\top\Sigma^{-1}(y-\mu)\f$.
   */
  LAPLACE_HD double logpdf(const mvgaussian1_state& st, const double* y, const double* mu) const {
    return detail_mvgaussian::logpdf<false>(structure, st, y, mu);
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
  LAPLACE_HD bool logpdf_dmu_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx,
                                double* out, double* w) const {
    return detail_mvgaussian::logpdf_dmu_dk<false, A, K>(structure, st, y, mu, idx, out, w);
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
  LAPLACE_HD bool logpdf_dy_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx,
                               double* out, double* w) const {
    return detail_mvgaussian::logpdf_dy_dk<false, J, K>(structure, st, y, mu, idx, out, w);
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
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `vector_rest` or structurally zero derivatives.
   */
  template <int K>
  LAPLACE_HD bool expected_info_dk(const mvgaussian1_state& st, distrib_mv_info_block block, const int* ab,
                                   const int* idx, double* out, double* w) const {
    return detail_mvgaussian::expected_info_dk<false, K>(structure, st, block, ab, idx, out, w);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Equals \f$\sum_{j=1}^{k_{\max}} \binom{q + j - 1}{j}\f$ for \f$k_{\max} \ge 1\f$
   * (0 if \f$k_{\max} \le 0\f$), where \f$q\f$ is `n_params()`, covering all sorted
   * parameter-index tuples of orders 1 through `k_max` in lexicographic order.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of table entries.
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  // The theta table: the derivatives d_T W and d_T log det M for the sorted
  // tuples of orders 1..k_max, filled once per theta and read by the unit
  // functions through the state (note in detail/mvgaussian_common.hpp).
  // entries and doubles of a table of orders 1..k_max
  LAPLACE_HD int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Each entry stores `pdmat::packed_size(dim()) + 2` doubles: a structural-nonzero
   * flag, \f$\partial_T \log\det\Sigma\f$, and packed \f$\partial_T \Sigma^{-1}\f$.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   * \see theta_table_entries(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  LAPLACE_HD int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table.
   *
   * Writes the structural-nonzero flag, \f$\partial_T \log\det\Sigma\f$, and
   * packed \f$\partial_T \Sigma^{-1}\f$ for the sorted tuple \f$T\f$ at index
   * `entry` among orders 1 through `k_max` in lexicographic order. Each entry
   * can be filled independently across threads.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table(), distrib_multivariate
   */
  // fills entry `entry` (0 <= entry < theta_table_entries(k_max)); w has scratch_size() doubles
  LAPLACE_HD void fill_theta_table_entry(const mvgaussian1_state& st, int k_max, int entry, double* table, double* w) const {
    detail_mvgaussian::fill_theta_table_entry<false>(structure, st, k_max, entry, table, w);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * Sets `table` and `table_order` on the returned state so that `logpdf_dmu_dk()`,
   * `logpdf_dy_dk()`, and the `vector_vector` block of `expected_info_dk()` read
   * precomputed derivatives for orders up to `k_max`. The buffer `table` must
   * remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order covered by `table`.
   * \param table Precomputed table buffer of `theta_table_size(k_max)` doubles filled by `fill_theta_table_entry()`.
   * \return Copy of `st` with `table` and `table_order` set.
   * \see fill_theta_table_entry(), theta_table_entries(), theta_table_size(), distrib_multivariate
   */
  // the state with the filled table attached
  LAPLACE_HD mvgaussian1_state with_theta_table(const mvgaussian1_state& st, int k_max, const double* table) const {
    mvgaussian1_state t = st;
    t.table = table;
    t.table_order = k_max;
    return t;
  }
  /**
   * \brief Draw one sample \f$y \sim \mathcal{N}_d(\mu, \Sigma)\f$ using counter-based normal deviates.
   *
   * Computes \f$y = \mu + L z\f$ from `st.factor` (\f$\Sigma = L L^\top\f$) and
   * independent standard normals generated via `standard_normal_by_inversion` and
   * `counter_uniform_source`.
   *
   * \param st Precomputed state from `theta_state`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   */
  LAPLACE_HD void rng(const mvgaussian1_state& st, const double* mu, std::uint64_t seed, std::uint64_t unit,
                      double* y) const {
    detail_mvgaussian::rng<false>(structure, st, mu, seed, unit, y);
  }
  /**
   * \brief Compute starting matrix parameters `theta` on the host from a sample summary.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample`.
   * \param theta Output parameter vector of length `n_params()`.
   */
  // (host) theta from a sample summary (summarize_mv_sample)
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    detail_mvgaussian::start_value<false>(structure, summary, theta);
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
   * \brief Write the covariance matrix \f$\Sigma(\theta)\f$ in packed lower-triangular order.
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  // Sigma, packed; w has structure.scratch_size() doubles
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const { structure.matrix(theta, out, w); }
};

}  // namespace laplace::distrib
