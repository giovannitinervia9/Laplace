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

#include <laplace/distrib/detail/mvgaussian_cdf.hpp>
#include <laplace/distrib/detail/mvgaussian_common.hpp>
#include <laplace/distrib/detail/mvgaussian_rank_deficient.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_mvgaussian1
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
   * \ingroup distrib_mvgaussian1
   * \brief Pointer to the precomputed \f$\theta\f$-derivative table attached by `with_theta_table()`, or `nullptr` if none.
   */
  const double* table = nullptr;  // the theta table (with_theta_table), or none
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Largest derivative order covered by the attached \f$\theta\f$-derivative table (0 if none).
   */
  int table_order = 0;            // its largest order
};

/**
 * \ingroup distrib_mvgaussian1
 * \brief Multivariate Gaussian distribution in the covariance parametrization \f$\Sigma = M(\theta)\f$.
 *
 * Two partial specializations: the proper class, for a positive definite structure
 * (`pdmat::pdmat_structure`), and the degenerate class, for a positive semidefinite structure
 * with a constant null space (`constant_null_space_psdmat`).
 *
 * \tparam S Matrix structure parameterizing the covariance.
 */
template <class S>
struct mvgaussian1_distrib;

/**
 * \ingroup distrib_mvgaussian1
 * \brief Multivariate Gaussian distribution parameterized by covariance \f$\Sigma(\theta)\f$.
 *
 * Models \f$Y \sim \mathcal{N}_d(\mu, \Sigma(\theta))\f$ with
 * \f$\Sigma(\theta)\f$ defined by a positive definite matrix structure `S`
 * satisfying `pdmat::pdmat_structure`. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 */
template <pdmat::pdmat_structure S>
struct mvgaussian1_distrib<S> {
  /// Positive definite matrix structure parameterizing the covariance \f$\Sigma(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite covariance).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Kind of the law (`continuous`: a density, with derivatives in \f$y\f$).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of parameters: \f$\mu\f$ and \f$\Sigma\f$.
  static constexpr int n_params = 2;
  /// Descriptors of \f$\mu\f$ (dense) and \f$\Sigma\f$ (indexed, the parameters \f$\theta\f$ of its structure).
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Sigma", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth}};
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
  /// Number of \f$y\f$ coordinates of the derivatives in \f$y\f$ (`dim()`: the unit is not constrained).
  LAPLACE_HD int y_coord_size() const { return structure.dim; }
  /**
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params`.
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for the matrix.
   */
  LAPLACE_HD int param_size(int j) const { return j == 0 ? structure.dim : structure.n_params(); }
  /**
   * \brief Return where the coordinates of parameter `j` start in their array (\f$x\f$ for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params`.
   * \return 0: \f$\mu\f$ starts \f$x\f$ and the matrix starts \f$\theta\f$.
   */
  LAPLACE_HD int param_offset(int j) const { return (void)j, 0; }
  /**
   * \brief Return the number of indexed coordinates, the parameters \f$\theta\f$ of the structure.
   * \return `structure.n_params()`.
   */
  LAPLACE_HD int theta_size() const { return structure.n_params(); }
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
   * \ingroup distrib_mvgaussian1
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
   * \param theta Parameter vector of length `theta_size()`.
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
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (`A >= 0`; omitted means 0).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `theta_size()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), or `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if `A <= 2` and the block is not structurally zero; `false` otherwise.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx,
                            double* out, double* w) const {
    return detail_mvgaussian::logpdf_dk<false, K, detail_mv::dense_order<A...>()>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate partial derivatives of the log density with respect to \f$y\f$ (order `J`) and \f$\theta\f$ (order `K`).
   *
   * Supports \f$J, K \ge 0\f$ with \f$J + K \le 6\f$. Blocks with \f$J \ge 3\f$
   * and structurally zero blocks return `false` without modifying `out`.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `J + K <= 6`).
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`, `J + K + A <= 6`; omitted means 0). With `A > 0`
   *         the output is the Kronecker product, the \f$y\f$ factor first, of the packed tensors of order `J` and `A`
   *         over \f$d\f$, and `w` holds `scratch_size() + sym_tensor_size(dim(), J + A)` doubles.
   * \param st Precomputed state from `theta_state`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `theta_size()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), or `pdmat::packed_size(dim())` (`J = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if `J <= 2` and the block is not structurally zero; `false` otherwise.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx,
                               double* out, double* w) const {
    if constexpr (detail_mv::dense_order<A...>() > 0)
      return detail_mv::location_dy_dk<J, K, detail_mv::dense_order<A...>()>(*this, st, y, mu, idx, out, w);
    return detail_mvgaussian::logpdf_dy_dk<false, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate the expected Fisher information matrix block or its `K`-th derivative with respect to \f$\theta\f$.
   *
   * Supports \f$K \in \{0, \dots, 4\}\f$ across `(0, 0)`, `(0, theta_slot)`, and `(theta_slot, theta_slot)`.
   *
   * \tparam K Derivative order with respect to \f$\theta\f$ (`0 <= K <= 4`).
   * \tparam A Derivative orders in the dense parameters (all zero here: any positive order returns `false`).
   * \param st Precomputed state from `theta_state`.
   * \param slot_a Slot of the row parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param slot_b Slot of the column parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param ab Two-element index pair `(a, b)` selecting an entry within `(theta_slot, theta_slot)` (unused otherwise).
   * \param x Dense parameters (unused: the information does not depend on \f$\mu\f$).
   * \param idx Sorted parameter indices of length `K` in \f$\theta\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (packed lower-triangular) for `(0, 0)`, or length `1` (`out[0]`) for `(theta_slot, theta_slot)`.
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `(0, theta_slot)` or structurally zero derivatives.
   */
  template <int K, int... A>
  LAPLACE_HD bool expected_info_dk(const mvgaussian1_state& st, int slot_a, int slot_b, const int* ab,
                                   const double* x, const int* idx, double* out, double* w) const {
    // the information does not depend on mu: every derivative in it is zero
    if constexpr ((0 + ... + A) > 0) {
      return false;
    } else {
      (void)x;
      return detail_mvgaussian::expected_info_dk<false, K>(structure, st, slot_a, slot_b, ab, idx, out, w);
    }
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Equals \f$\sum_{j=1}^{k_{\max}} \binom{q + j - 1}{j}\f$ for \f$k_{\max} \ge 1\f$
   * (0 if \f$k_{\max} \le 0\f$), where \f$q\f$ is `theta_size()`, covering all sorted
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
   * \ingroup distrib_mvgaussian1
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
   * \ingroup distrib_mvgaussian1
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
   * \ingroup distrib_mvgaussian1
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * Sets `table` and `table_order` on the returned state so that `logpdf_dk()`,
   * `logpdf_dy_dk()`, and the `(0, 0)` block of `expected_info_dk()` read
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
   * \param theta Output parameter vector of length `theta_size()`.
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
   * \param theta Parameter vector of length `theta_size()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  // Sigma, packed; w has structure.scratch_size() doubles
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const { structure.matrix(theta, out, w); }

  /// \name Distribution function (optional members, detected by `mv_has_cdf`)
  ///@{
  /**
   * \brief Doubles of `w` needed by `log_cdf`, `log_ccdf`, `log_rect_prob` and their exponentials.
   */
  LAPLACE_HD int cdf_value_scratch_size() const { return detail_mvgaussian_cdf::value_scratch(structure.dim); }
  /**
   * \brief Doubles of `w` needed by the derivatives of orders `K` in \f$\theta\f$ and `A` in \f$\mu\f$ (or in \f$y\f$ and \f$\mu\f$ together).
   * \param K Order in \f$\theta\f$.
   * \param A Order in \f$\mu\f$ plus order in \f$y\f$.
   */
  LAPLACE_HD int cdf_scratch_size(int K, int A = 0) const {
    return detail_mvgaussian_cdf::deriv_scratch(structure, K, A) + sym_tensor_size(structure.dim, A);
  }
  /// \brief \f$\log\Pr(Y \le y)\f$ (componentwise). `w` has `cdf_value_scratch_size()` doubles.
  LAPLACE_HD double log_cdf(const mvgaussian1_state& st, const double* y, const double* mu, double* w) const {
    return detail_mvgaussian_cdf::log_value<false, detail_mvgaussian_cdf::region::lower>(structure, st, y, nullptr, mu, w);
  }
  /// \brief \f$\log\Pr(Y > y)\f$ (componentwise). `w` has `cdf_value_scratch_size()` doubles.
  LAPLACE_HD double log_ccdf(const mvgaussian1_state& st, const double* y, const double* mu, double* w) const {
    return detail_mvgaussian_cdf::log_value<false, detail_mvgaussian_cdf::region::upper>(structure, st, y, nullptr, mu, w);
  }
  /// \brief \f$\log\Pr(a < Y \le b)\f$; infinite limits allowed. `w` has `cdf_value_scratch_size()` doubles.
  LAPLACE_HD double log_rect_prob(const mvgaussian1_state& st, const double* a, const double* b, const double* mu, double* w) const {
    return detail_mvgaussian_cdf::log_value<false, detail_mvgaussian_cdf::region::rect>(structure, st, a, b, mu, w);
  }
  /// \brief \f$\Pr(Y \le y)\f$.
  LAPLACE_HD double cdf(const mvgaussian1_state& st, const double* y, const double* mu, double* w) const {
    return math::exp(log_cdf(st, y, mu, w));
  }
  /// \brief \f$\Pr(Y > y)\f$.
  LAPLACE_HD double ccdf(const mvgaussian1_state& st, const double* y, const double* mu, double* w) const {
    return math::exp(log_ccdf(st, y, mu, w));
  }
  /**
   * \brief Derivatives of \f$\log\Pr(Y \le y)\f$ of order `K` in \f$\theta\f$ along `idx` and `A` in \f$\mu\f$.
   *
   * Same layout as `logpdf_dk`; \f$A + K \le 6\f$. `w` has `cdf_scratch_size(K, A)` doubles.
   */
  template <int K, int... A>
  LAPLACE_HD bool log_cdf_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx, double* out,
                             double* w) const {
    return detail_mvgaussian_cdf::log_dk<false, detail_mvgaussian_cdf::region::lower, K, detail_mv::dense_order<A...>()>(
        structure, st, y, nullptr, mu, idx, out, w);
  }
  /// \brief Derivatives of \f$\log\Pr(Y > y)\f$, as `log_cdf_dk`.
  template <int K, int... A>
  LAPLACE_HD bool log_ccdf_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx, double* out,
                              double* w) const {
    return detail_mvgaussian_cdf::log_dk<false, detail_mvgaussian_cdf::region::upper, K, detail_mv::dense_order<A...>()>(
        structure, st, y, nullptr, mu, idx, out, w);
  }
  /// \brief Derivatives of \f$\log\Pr(a < Y \le b)\f$, as `log_cdf_dk`.
  template <int K, int... A>
  LAPLACE_HD bool log_rect_prob_dk(const mvgaussian1_state& st, const double* a, const double* b, const double* mu, const int* idx,
                                   double* out, double* w) const {
    return detail_mvgaussian_cdf::log_dk<false, detail_mvgaussian_cdf::region::rect, K, detail_mv::dense_order<A...>()>(
        structure, st, a, b, mu, idx, out, w);
  }
  /**
   * \brief Derivatives of \f$\log\Pr(Y \le y)\f$ of order `J` in \f$y\f$, `K` in \f$\theta\f$ and `A` in \f$\mu\f$.
   *
   * \f$\log F\f$ depends on \f$y - \mu\f$ only, so \f$\partial_y = -\partial_\mu\f$: the output is the
   * Kronecker product (the \f$y\f$ factor first) of the packed tensors of orders `J` and `A`, as `logpdf_dy_dk`.
   * `w` has `cdf_scratch_size(K, J + A)` doubles.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool log_cdf_dy_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx, double* out,
                                double* w) const {
    return detail_mvgaussian_cdf::log_dy_dk<false, detail_mvgaussian_cdf::region::lower, J, K, detail_mv::dense_order<A...>()>(
        structure, st, y, mu, idx, out, w);
  }
  /// \brief Derivatives of \f$\log\Pr(Y > y)\f$ in \f$y\f$, \f$\theta\f$ and \f$\mu\f$, as `log_cdf_dy_dk`.
  template <int J, int K, int... A>
  LAPLACE_HD bool log_ccdf_dy_dk(const mvgaussian1_state& st, const double* y, const double* mu, const int* idx, double* out,
                                 double* w) const {
    return detail_mvgaussian_cdf::log_dy_dk<false, detail_mvgaussian_cdf::region::upper, J, K, detail_mv::dense_order<A...>()>(
        structure, st, y, mu, idx, out, w);
  }
  ///@}
};

/**
 * \ingroup distrib_mvgaussian1
 * \brief Degenerate multivariate Gaussian distribution parameterized by a singular covariance \f$\Sigma(\theta)\f$ with constant null space.
 *
 * Models a proper Gaussian law on the affine subspace \f$\mu + \operatorname{col}\Sigma(\theta)\f$
 * of rank \f$\rho\f$, with log-density
 * \f$\ell_1 = -\frac{\rho}{2}\log 2\pi - \frac{1}{2}\log\lvert\Sigma\rvert_+ - \frac{1}{2} r^\top \Sigma^+ r\f$
 * (\f$r = y - \mu\f$) on the support and \f$-\infty\f$ off it. Host only, as
 * required by `S`. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive semidefinite matrix structure satisfying `constant_null_space_psdmat`.
 * \see mvgaussian_rank_deficient_state, constant_null_space_psdmat, distrib_multivariate
 */
template <constant_null_space_psdmat S>
struct mvgaussian1_distrib<S> {
  /// \name Degenerate class: singular covariance with a constant null space
  ///@{
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Positive semidefinite matrix structure parameterizing the covariance \f$\Sigma(\theta)\f$.
   */
  S structure;

  /**
   * \ingroup distrib_mvgaussian1
   * \brief Normalization regime (`distrib_normalization::degenerate`: singular covariance).
   */
  static constexpr distrib_normalization normalization = distrib_normalization::degenerate;
  /// Kind of the law (`continuous`: a density, with derivatives in \f$y\f$).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Number of parameters: \f$\mu\f$ and \f$\Sigma\f$.
   */
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Descriptors of \f$\mu\f$ (dense) and \f$\Sigma\f$ (indexed, the parameters \f$\theta\f$ of its structure).
   */
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Sigma", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth}};
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Precomputed state type returned by `theta_state()`.
   */
  using state_type = mvgaussian_rank_deficient_state;

  /**
   * \ingroup distrib_mvgaussian1
   * \brief Construct from a positive semidefinite matrix structure instance.
   * \param s Matrix structure defining \f$\Sigma(\theta)\f$.
   */
  explicit mvgaussian1_distrib(const S& s) : structure(s) {}

  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  int dim() const { return structure.dim; }
  /// Number of \f$y\f$ coordinates of the derivatives in \f$y\f$ (`dim()`: the unit is not constrained).
  int y_coord_size() const { return structure.dim; }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params`.
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for the matrix.
   */
  int param_size(int j) const { return j == 0 ? structure.dim : structure.n_params(); }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return where the coordinates of parameter `j` start in their array (\f$x\f$ for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params`.
   * \return 0: \f$\mu\f$ starts \f$x\f$ and the matrix starts \f$\theta\f$.
   */
  int param_offset(int j) const { return (void)j, 0; }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of indexed coordinates, the parameters \f$\theta\f$ of the structure.
   * \return `structure.n_params()`.
   */
  int theta_size() const { return structure.n_params(); }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the constant rank \f$\rho\f$ of \f$\Sigma(\theta)\f$.
   * \return Rank `structure.rank()`.
   */
  int rank() const { return structure.rank(); }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of `double` elements stored in `w` for the precomputed state.
   * \return `3 * pdmat::packed_size(dim()) + structure.scratch_size()`.
   * \see theta_state(), scratch_size(), distrib_multivariate
   */
  int state_size() const { return detail_mvgaussian::rank_deficient_state_size(structure); }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return working scratch doubles required after the state for `theta_state()`, derivative blocks, and table filling.
   * \return `66 * pdmat::packed_size(dim()) + dim() * dim() + structure.scratch_size()`.
   * \see state_size(), info_scratch_size(), distrib_multivariate
   */
  int scratch_size() const { return detail_mvgaussian::rank_deficient_scratch<false>(structure); }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return workspace doubles required in `w` by `expected_info_dk()`.
   * \return `65 * pdmat::packed_size(dim()) + dim() * dim() + structure.scratch_size()`.
   * \see expected_info_dk(), scratch_size(), distrib_multivariate
   */
  int info_scratch_size() const { return detail_mvgaussian::rank_deficient_info_scratch<false>(structure); }

  /**
   * \ingroup distrib_mvgaussian1
   * \brief Precompute \f$\log\lvert\Sigma(\theta)\rvert_+\f$, \f$\Sigma^+\f$, and the Cholesky factor of \f$\Sigma + cP\f$.
   *
   * Forms \f$\Sigma^+ = (\Sigma + cP)^{-1} - P/c\f$ with \f$P = U_0 U_0^\top\f$
   * and fixed shift \f$c = \operatorname{tr}\Sigma / \rho\f$ (`1.0` if
   * \f$\rho = 0\f$ or \f$\operatorname{tr}\Sigma \le 0\f$), and records the
   * structure's evaluated log pseudo-determinant state in `w` when available.
   *
   * \param theta Parameter vector of length `theta_size()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back the returned state while it is in use.
   * \return Populated `mvgaussian_rank_deficient_state`.
   * \see mvgaussian_rank_deficient_state, state_size(), scratch_size(), distrib_multivariate
   */
  mvgaussian_rank_deficient_state theta_state(const double* theta, double* w) const {
    return detail_mvgaussian::rank_deficient_state<false>(structure, theta, w, true);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Evaluate the degenerate Gaussian log-density.
   *
   * Applies the support test \f$\lVert U_0^\top r\rVert > 64\,d\,\varepsilon\,\lVert r\rVert\f$
   * with \f$r = y - \mu\f$ and \f$\varepsilon = 2^{-52}\f$, returning \f$-\infty\f$
   * off the support \f$\mu + \operatorname{col}\Sigma\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$-\frac{\rho}{2}\log 2\pi - \frac{1}{2}\log\lvert\Sigma\rvert_+ - \frac{1}{2} r^\top \Sigma^+ r\f$ on the support, or \f$-\infty\f$ off it.
   * \see logpdf_dk(), theta_state(), distrib_multivariate
   */
  double logpdf(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu) const {
    return detail_mvgaussian::rank_deficient_logpdf<false>(structure, st, y, mu);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Evaluate partial derivatives of the log-density with respect to \f$\mu\f$ (order `A`) and \f$\theta\f$ (order `K`).
   *
   * Supports \f$A, K \ge 0\f$ with \f$A + K \le 6\f$. For \f$A = 0, K = 0\f$,
   * delegates to `logpdf()` (including the support test). Higher-order blocks
   * differentiate the projected density on \f$\mu + \operatorname{col}\Sigma\f$
   * without testing support, using \f$\Sigma^+\f$, \f$\log\lvert\Sigma\rvert_+\f$,
   * and \f$\partial_T \Sigma^+ = \partial_T(\Sigma + cP)^{-1}\f$. Blocks with
   * \f$A \ge 3\f$ or structurally zero derivatives return `false` without
   * modifying `out`.
   *
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (`A >= 0`; omitted means 0).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `theta_size()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), or `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when `A >= 3` or the block vanishes identically by structure.
   * \see logpdf(), logpdf_dy_dk(), with_theta_table(), distrib_multivariate
   */
  template <int K, int... A>
  bool logpdf_dk(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu, const int* idx,
                 double* out, double* w) const {
    if constexpr (detail_mv::dense_order<A...>() == 0 && K == 0) {
      out[0] = logpdf(st, y, mu);
      return true;
    } else {
      return detail_mvgaussian::logpdf_dk<false, K, detail_mv::dense_order<A...>()>(
          detail_mvgaussian::rank_deficient_view<S>(structure, st), st, y, mu, idx, out, w);
    }
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Evaluate partial derivatives of the log-density with respect to \f$y\f$ (order `J`) and \f$\theta\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dk<K, J>()` for \f$J + K \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$\theta\f$ (`K >= 0`, `J + K <= 6`).
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`, `J + K + A <= 6`; omitted means 0). With `A > 0`
   *         the output is the Kronecker product, the \f$y\f$ factor first, of the packed tensors of order `J` and `A`
   *         over \f$d\f$, and `w` holds `scratch_size() + sym_tensor_size(dim(), J + A)` doubles.
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q - 1\}\f$, \f$q\f$ = `theta_size()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), or `pdmat::packed_size(dim())` (`J = 2`, packed lower-triangular).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when `J >= 3` or the block vanishes identically by structure.
   * \see logpdf_dk(), distrib_multivariate
   */
  template <int J, int K, int... A>
  bool logpdf_dy_dk(const mvgaussian_rank_deficient_state& st, const double* y, const double* mu, const int* idx,
                    double* out, double* w) const {
    if constexpr (detail_mv::dense_order<A...>() > 0)
      return detail_mv::location_dy_dk<J, K, detail_mv::dense_order<A...>()>(*this, st, y, mu, idx, out, w);
    if (!logpdf_dk<K, J>(st, y, mu, idx, out, w)) return false;
    if constexpr (J % 2 == 1) {
      const int n = J == 1 ? dim() : pdmat::packed_size(dim());
      for (int k = 0; k < n; ++k) out[k] = -out[k];
    }
    return true;
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Evaluate the expected Fisher information block or its `K`-th derivative with respect to \f$\theta\f$.
   *
   * Supports \f$0 \le K \le 4\f$ for structures satisfying
   * \f$\partial_{ab}\Sigma = \delta_{ab}\,\partial_a\Sigma\f$ (such as
   * `pdmat::weighted_sum_psdmat`). `(0, 0)` writes packed \f$\Sigma^+\f$
   * or its \f$\theta\f$-derivatives; `(0, theta_slot)` returns `false` without
   * modifying `out`; `(theta_slot, theta_slot)` writes \f$\partial_T I_{ab}\f$ where
   * \f$I_{ab} = -\frac{1}{2}\partial_{ab}\log\lvert\Sigma\rvert_+ + \frac{1}{2}\delta_{ab}\partial_a\log\lvert\Sigma\rvert_+\f$.
   *
   * \tparam K Derivative order with respect to \f$\theta\f$ (`0 <= K <= 4`).
   * \tparam A Derivative orders in the dense parameters (all zero here: any positive order returns `false`).
   * \param st Precomputed state from `theta_state()`.
   * \param slot_a Slot of the row parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param slot_b Slot of the column parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param ab Two-element index pair `(a, b)` selecting an entry within `(theta_slot, theta_slot)` (unused otherwise).
   * \param x Dense parameters (unused: the information does not depend on \f$\mu\f$).
   * \param idx Sorted parameter indices of length `K` in \f$\theta\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (`(0, 0)`) or length `1` (`(theta_slot, theta_slot)`).
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `(0, theta_slot)` or structurally zero derivatives.
   * \see info_scratch_size(), theta_slot, distrib_multivariate
   */
  template <int K, int... A>
    requires detail_mvgaussian::exp_linear_structure<S>
  bool expected_info_dk(const mvgaussian_rank_deficient_state& st, int slot_a, int slot_b, const int* ab,
                        const double* x, const int* idx, double* out, double* w) const {
    // the information does not depend on mu: every derivative in it is zero
    if constexpr ((0 + ... + A) > 0) {
      return false;
    } else {
      (void)x;
      return detail_mvgaussian::rank_deficient_info<false, K>(structure, st, slot_a, slot_b, ab, idx, out, w);
    }
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   * \see theta_table_entries(), fill_theta_table_entry(), with_theta_table(), distrib_multivariate
   */
  int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table.
   *
   * Entries must be filled one at a time on the host because they share the
   * structure's evaluated pseudo-determinant state in `st.pdet`.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table(), distrib_multivariate
   */
  void fill_theta_table_entry(const mvgaussian_rank_deficient_state& st, int k_max, int entry, double* table,
                              double* w) const {
    detail_mvgaussian::fill_theta_table_entry<false>(detail_mvgaussian::rank_deficient_view<S>(structure, st), st,
                                                     k_max, entry, table, w);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * The buffer `table` must remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()`.
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
   * \ingroup distrib_mvgaussian1
   * \brief Draw one sample \f$y = \mu + (I - P) L z\f$ from the degenerate Gaussian.
   *
   * Uses `st.factor` (\f$\Sigma + cP = L L^\top\f$) and independent standard
   * normal deviates \f$z\f$ from `standard_normal_by_inversion()` and
   * `counter_uniform_source` `{seed, unit}`, yielding covariance
   * \f$(I - P)(\Sigma + cP)(I - P) = \Sigma\f$ with zero null-space component.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   * \param tmp Caller scratch buffer of at least `dim()` doubles.
   * \see theta_state(), covariance(), distrib_multivariate
   */
  void rng(const mvgaussian_rank_deficient_state& st, const double* mu, std::uint64_t seed, std::uint64_t unit,
           double* y, double* tmp) const {
    detail_mvgaussian::rank_deficient_rng<false>(structure, st, mu, seed, unit, y, tmp);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Compute starting matrix parameters `theta` on the host from a sample summary.
   *
   * Projects the sample covariance \f$C\f$ from `summary` onto the range as
   * \f$(I - P) C (I - P)\f$ without applying an eigenvalue floor, then calls
   * `structure.params_from_matrix()`.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `theta_size()`.
   * \see summarize_mv_sample(), distrib_mv_sample_summary, distrib_multivariate
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    detail_mvgaussian::rank_deficient_start_value<false>(structure, summary, theta);
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out`.
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   * \see covariance(), distrib_multivariate
   */
  void mean(const double* mu, double* out) const {
    for (int k = 0; k < structure.dim; ++k) out[k] = mu[k];
  }
  /**
   * \ingroup distrib_mvgaussian1
   * \brief Write the covariance matrix \f$\Sigma(\theta)\f$ in packed lower-triangular order.
   * \param theta Parameter vector of length `theta_size()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   * \see mean(), rng(), distrib_multivariate
   */
  void covariance(const double* theta, double* out, double* w) const { structure.matrix(theta, out, w); }
  ///@}
};

}  // namespace laplace::distrib
