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
#include <laplace/distrib/detail/mvgaussian_rank_deficient.hpp>

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
 * \brief Multivariate Gaussian distribution in the precision parametrization \f$Q = M(\theta)\f$.
 *
 * Two partial specializations: the proper class, for a positive definite structure
 * (`pdmat::pdmat_structure`), and the improper class, for a positive semidefinite structure
 * with a constant null space (`constant_null_space_psdmat`).
 *
 * \tparam S Matrix structure parameterizing the precision.
 */
template <class S>
struct mvgaussian2_distrib;

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
struct mvgaussian2_distrib<S> {
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
   * \ingroup distrib_multivariate
   * \brief Return workspace doubles needed in `w` for `expected_info_dk()`.
   *
   * \return `2 * pdmat::packed_size(structure.dim) + structure.scratch_size()`.
   * \see expected_info_dk(), scratch_size(), distrib_multivariate
   */
  // doubles of w of expected_info_dk
  LAPLACE_HD int info_scratch_size() const { return detail_mvgaussian::info_scratch_size<true>(structure); }

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
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `vector_rest` or structurally zero derivatives.
   */
  template <int K>
  LAPLACE_HD bool expected_info_dk(const mvgaussian2_state& st, distrib_mv_info_block block, const int* ab,
                                   const int* idx, double* out, double* w) const {
    return detail_mvgaussian::expected_info_dk<true, K>(structure, st, block, ab, idx, out, w);
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
   * flag, \f$\partial_T \log\det Q\f$, and packed \f$\partial_T Q\f$.
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
   * Writes the structural-nonzero flag, \f$\partial_T \log\det Q\f$, and
   * packed \f$\partial_T Q\f$ for the sorted tuple \f$T\f$ at index `entry`
   * among orders 1 through `k_max` in lexicographic order. Each entry can be
   * filled independently across threads.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table(), distrib_multivariate
   */
  // fills entry `entry` (0 <= entry < theta_table_entries(k_max)); w has scratch_size() doubles
  LAPLACE_HD void fill_theta_table_entry(const mvgaussian2_state& st, int k_max, int entry, double* table, double* w) const {
    detail_mvgaussian::fill_theta_table_entry<true>(structure, st, k_max, entry, table, w);
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
  LAPLACE_HD mvgaussian2_state with_theta_table(const mvgaussian2_state& st, int k_max, const double* table) const {
    mvgaussian2_state t = st;
    t.table = table;
    t.table_order = k_max;
    return t;
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

/**
 * \ingroup distrib_multivariate
 * \brief Improper (intrinsic) multivariate Gaussian distribution parameterized by a singular precision \f$Q(\theta)\f$ with constant null space.
 *
 * Models an improper density flat along the null space of \f$Q(\theta)\f$ of rank \f$\rho\f$,
 * with log-density
 * \f$\ell_2 = -\frac{\rho}{2}\log 2\pi + \frac{1}{2}\log\lvert Q\rvert_+ - \frac{1}{2} r^\top Q r\f$
 * (\f$r = y - \mu\f$), counting only the \f$\rho\f$ range dimensions in the
 * normalizing constant. Draws, constrained covariance, and expected Fisher
 * information belong to the constrained proper law \f$U_0^\top(y - \mu) = 0\f$
 * with covariance \f$Q^+\f$. Host only, as required by `S`. Satisfies
 * `multivariate_distrib`.
 *
 * \tparam S Positive semidefinite matrix structure satisfying `constant_null_space_psdmat`.
 * \see mvgaussian_rank_deficient_state, constant_null_space_psdmat, distrib_multivariate
 */
template <constant_null_space_psdmat S>
struct mvgaussian2_distrib<S> {
  /// \name Improper class: singular precision with a constant null space
  ///@{
  /**
   * \ingroup distrib_multivariate
   * \brief Positive semidefinite matrix structure parameterizing the precision \f$Q(\theta)\f$.
   */
  S structure;

  /**
   * \ingroup distrib_multivariate
   * \brief Normalization regime (`distrib_normalization::improper`: singular precision).
   */
  static constexpr distrib_normalization normalization = distrib_normalization::improper;
  /**
   * \ingroup distrib_multivariate
   * \brief Precomputed state type returned by `theta_state()` and `constrained_state()`.
   */
  using state_type = mvgaussian_rank_deficient_state;

  /**
   * \ingroup distrib_multivariate
   * \brief Construct from a positive semidefinite matrix structure instance.
   * \param s Matrix structure defining \f$Q(\theta)\f$.
   */
  explicit mvgaussian2_distrib(const S& s) : structure(s) {}

  /**
   * \ingroup distrib_multivariate
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  int dim() const { return structure.dim; }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of precision parameters in \f$\theta\f$.
   * \return Parameter count `structure.n_params()`.
   */
  int n_params() const { return structure.n_params(); }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the constant rank \f$\rho\f$ of \f$Q(\theta)\f$.
   * \return Rank `structure.rank()`.
   */
  int rank() const { return structure.rank(); }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of `double` elements stored in `w` for the precomputed state.
   * \return `3 * pdmat::packed_size(dim()) + structure.scratch_size()`.
   * \see theta_state(), constrained_state(), scratch_size(), distrib_multivariate
   */
  int state_size() const { return detail_mvgaussian::rank_deficient_state_size(structure); }
  /**
   * \ingroup distrib_multivariate
   * \brief Return working scratch doubles required after the state for `theta_state()`, `constrained_state()`, derivative blocks, and table filling.
   * \return `dim() * dim() + structure.scratch_size()` (or `pdmat::packed_size(dim()) + structure.scratch_size()` when larger).
   * \see state_size(), info_scratch_size(), distrib_multivariate
   */
  int scratch_size() const { return detail_mvgaussian::rank_deficient_scratch<true>(structure); }
  /**
   * \ingroup distrib_multivariate
   * \brief Return workspace doubles required in `w` by `expected_info_dk()`.
   * \return `structure.scratch_size()`.
   * \see expected_info_dk(), scratch_size(), distrib_multivariate
   */
  int info_scratch_size() const { return detail_mvgaussian::rank_deficient_info_scratch<true>(structure); }

  /**
   * \ingroup distrib_multivariate
   * \brief Precompute \f$\log\lvert Q(\theta)\rvert_+\f$ and packed \f$Q(\theta)\f$ without a Cholesky factor.
   *
   * Records \f$\log\lvert Q(\theta)\rvert_+\f$ (via `structure.log_pdet_state()`
   * when available) and packed \f$Q(\theta)\f$, leaving `factor` and
   * `shifted_inverse` as `nullptr`. Use `constrained_state()` when draws or
   * constrained covariances are needed.
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back the returned state while it is in use.
   * \return Populated `mvgaussian_rank_deficient_state`.
   * \see constrained_state(), mvgaussian_rank_deficient_state, state_size(), distrib_multivariate
   */
  mvgaussian_rank_deficient_state theta_state(const double* theta, double* w) const {
    return detail_mvgaussian::rank_deficient_state<true>(structure, theta, w, false);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Precompute the state of `theta_state()` together with the Cholesky factor of \f$Q + cP\f$ and \f$(Q + cP)^{-1}\f$.
   *
   * Adds the packed lower Cholesky factor \f$L\f$ of \f$Q + cP\f$ and packed
   * \f$(Q + cP)^{-1}\f$ with \f$P = U_0 U_0^\top\f$ and fixed shift
   * \f$c = \operatorname{tr} Q / \rho\f$ (`1.0` if \f$\rho = 0\f$ or
   * \f$\operatorname{tr} Q \le 0\f$). Required by `constrained_rng()` and
   * `constrained_covariance()`.
   *
   * \param theta Parameter vector of length `n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back the returned state while it is in use.
   * \return Populated `mvgaussian_rank_deficient_state` with `factor` and `shifted_inverse` set.
   * \see theta_state(), constrained_rng(), constrained_covariance(), distrib_multivariate
   */
  mvgaussian_rank_deficient_state constrained_state(const double* theta, double* w) const {
    return detail_mvgaussian::rank_deficient_state<true>(structure, theta, w, true);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Evaluate the improper Gaussian log-density.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$-\frac{\rho}{2}\log 2\pi + \frac{1}{2}\log\lvert Q\rvert_+ - \frac{1}{2} r^\top Q r\f$ with \f$r = y - \mu\f$.
   * \see logpdf_dmu_dk(), theta_state(), distrib_multivariate
   */
  double logpdf(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu) const {
    return detail_mvgaussian::rank_deficient_logpdf<true>(structure, st, y, mu);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Evaluate partial derivatives of the log-density with respect to \f$\mu\f$ (order `A`) and \f$\theta\f$ (order `K`).
   *
   * Supports \f$A, K \ge 0\f$ with \f$A + K \le 6\f$, following the proper
   * precision formulas with \f$Q\f$ and \f$\log\lvert Q\rvert_+\f$ (and
   * `logpdf()` for \f$A = 0, K = 0\f$). Blocks with \f$A \ge 3\f$ or
   * structurally zero derivatives return `false` without modifying `out`.
   *
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), or `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when `A >= 3` or the block vanishes identically by structure.
   * \see logpdf(), logpdf_dy_dk(), with_theta_table(), distrib_multivariate
   */
  template <int A, int K>
  bool logpdf_dmu_dk(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu, const int* idx,
                     double* out, double* w) const {
    if constexpr (A == 0 && K == 0) {
      out[0] = logpdf(st, y, mu);
      return true;
    } else {
      return detail_mvgaussian::logpdf_dmu_dk<true, A, K>(
          detail_mvgaussian::rank_deficient_view<S>(structure, st), st, y, mu, idx, out, w);
    }
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Evaluate partial derivatives of the log-density with respect to \f$y\f$ (order `J`) and \f$\theta\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dmu_dk<J, K>()` for \f$J + K \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `J + K <= 6`).
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), or `pdmat::packed_size(dim())` (`J = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when `J >= 3` or the block vanishes identically by structure.
   * \see logpdf_dmu_dk(), distrib_multivariate
   */
  template <int J, int K>
  bool logpdf_dy_dk(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu, const int* idx,
                    double* out, double* w) const {
    if (!logpdf_dmu_dk<J, K>(st, y, mu, idx, out, w)) return false;
    if constexpr (J % 2 == 1) {
      const int n = J == 1 ? dim() : pdmat::packed_size(dim());
      for (int k = 0; k < n; ++k) out[k] = -out[k];
    }
    return true;
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Evaluate the expected Fisher information block of the constrained proper law or its `K`-th derivative in \f$\theta\f$.
   *
   * Supports \f$0 \le K \le 4\f$ for structures satisfying
   * \f$\partial_{ab} Q = \delta_{ab}\,\partial_a Q\f$ (such as
   * `pdmat::weighted_sum_psdmat`). `vector_vector` writes packed \f$Q\f$ or
   * its \f$\theta\f$-derivatives; `vector_rest` returns `false` without
   * modifying `out`; `rest_rest` writes \f$\partial_T I_{ab}\f$ for the
   * constrained law \f$U_0^\top(y - \mu) = 0\f$, where
   * \f$I_{ab} = -\frac{1}{2}\partial_{ab}\log\lvert Q\rvert_+ + \frac{1}{2}\delta_{ab}\partial_a\log\lvert Q\rvert_+\f$.
   *
   * \tparam K Derivative order with respect to \f$\theta\f$ (`0 <= K <= 4`).
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param block Block of the expected information matrix.
   * \param ab Two-element index pair `(a, b)` selecting an entry within `rest_rest` (unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\theta\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (`vector_vector`) or length `1` (`rest_rest`).
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `vector_rest` or structurally zero derivatives.
   * \see info_scratch_size(), distrib_mv_info_block, distrib_multivariate
   */
  template <int K>
    requires detail_mvgaussian::exp_linear_structure<S>
  bool expected_info_dk(const mvgaussian_rank_deficient_state& st, distrib_mv_info_block block, const int* ab,
                        const int* idx, double* out, double* w) const {
    return detail_mvgaussian::rank_deficient_info<true, K>(structure, st, block, ab, idx, out, w);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   * \see theta_table_entries(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table.
   *
   * Entries must be filled one at a time on the host because they share the
   * structure's evaluated pseudo-determinant state in `st.pdet`.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table(), distrib_multivariate
   */
  void fill_theta_table_entry(const mvgaussian_rank_deficient_state& st, int k_max, int entry, double* table,
                              double* w) const {
    detail_mvgaussian::fill_theta_table_entry<true>(detail_mvgaussian::rank_deficient_view<S>(structure, st), st,
                                                    k_max, entry, table, w);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * The buffer `table` must remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param k_max Largest derivative order covered by `table`.
   * \param table Precomputed table buffer of `theta_table_size(k_max)` doubles filled by `fill_theta_table_entry()`.
   * \return Copy of `st` with `table` and `table_order` set.
   * \see fill_theta_table_entry(), theta_table_entries(), theta_table_size(), distrib_multivariate
   */
  mvgaussian_rank_deficient_state with_theta_table(const mvgaussian_rank_deficient_state& st, int k_max,
                                                   const double* table) const {
    mvgaussian_rank_deficient_state t = st;
    t.table = table;
    t.table_order = k_max;
    return t;
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Draw one sample \f$y = \mu + (I - P) L^{-\top} z\f$ from the constrained proper law \f$U_0^\top(y - \mu) = 0\f$.
   *
   * Uses `cst.factor` (\f$Q + cP = L L^\top\f$) and independent standard normal
   * deviates \f$z\f$ from `standard_normal_by_inversion()` and
   * `counter_uniform_source` `{seed, unit}`, yielding covariance \f$Q^+\f$ with
   * zero null-space component.
   *
   * \param cst Precomputed state from `constrained_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   * \param tmp Caller scratch buffer of at least `dim()` doubles.
   * \see constrained_state(), constrained_covariance(), distrib_multivariate
   */
  void constrained_rng(const mvgaussian_rank_deficient_state& cst, const double* mu, std::uint64_t seed,
                       std::uint64_t unit, double* y, double* tmp) const {
    detail_mvgaussian::rank_deficient_rng<true>(structure, cst, mu, seed, unit, y, tmp);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Write the covariance \f$Q^+ = (Q + cP)^{-1} - P/c\f$ of the constrained law in packed lower-triangular order.
   * \param cst Precomputed state from `constrained_state()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \see constrained_state(), constrained_rng(), distrib_multivariate
   */
  void constrained_covariance(const mvgaussian_rank_deficient_state& cst, double* out) const {
    detail_mvgaussian::rank_deficient_view<S>(structure, cst).inverse(cst.theta, out, nullptr);
  }
  /**
   * \ingroup distrib_multivariate
   * \brief Compute starting matrix parameters `theta` on the host from a sample summary.
   *
   * Projects the sample covariance \f$C\f$ from `summary` onto the range as
   * \f$C_p = (I - P) C (I - P)\f$ without an eigenvalue floor, forms its
   * pseudo-inverse \f$C_p^+ = (C_p + cP)^{-1} - P/c\f$ with
   * \f$c = \operatorname{tr} C_p / \rho\f$, and calls `structure.params_from_matrix()`.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `n_params()`.
   * \see summarize_mv_sample(), distrib_mv_sample_summary, distrib_multivariate
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    detail_mvgaussian::rank_deficient_start_value<true>(structure, summary, theta);
  }
  ///@}
};

}  // namespace laplace::distrib
