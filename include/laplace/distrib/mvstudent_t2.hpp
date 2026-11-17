// mvstudent_t2_distrib<S>: the multivariate Student t with location mu, inverse
// scale matrix Q = M(theta) (a positive definite structure S of laplace::pdmat)
// and degrees of freedom nu > 0; note/distrib_multivariate.md section 15, the
// formulas in detail/mvstudent_t_common.hpp. "The rest" is (theta, nu), nu last.
//
//   l = c_d(nu) + 1/2 log det Q - (nu + d)/2 log1p(delta/nu),  delta = r' Q r.
//
// Q is the inverse of the scale matrix, the precision of Y only up to
// (nu - 2)/nu: Var Y = nu/(nu - 2) Q^-1 for nu > 2. Proper class only (a
// singular Q would be the improper class, not built).
#pragma once

#include <laplace/distrib/detail/mvstudent_t_common.hpp>
#include <laplace/distrib/mvgaussian2.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_mvstudent_t2
 * \brief Multivariate Student t distribution parameterized by location \f$\mu\f$, inverse scale matrix \f$Q(\theta)\f$, and degrees of freedom \f$\nu\f$.
 *
 * Models \f$Y \sim t_d(\mu, Q(\theta)^{-1}, \nu)\f$ with inverse scale matrix
 * \f$Q(\theta) = M(\theta)\f$ defined by a positive definite matrix structure `S`
 * satisfying `pdmat::pdmat_structure`, and "the rest" parameter vector
 * \f$(\theta_0, \dots, \theta_{q-1}, \nu)\f$ with \f$\nu > 0\f$ at the last index \f$q\f$.
 * Proper class only. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 * \see mvstudent_t_state, mvstudent_t1_distrib, distrib_mvstudent_t2, distrib_multivariate
 */
template <pdmat::pdmat_structure S>
struct mvstudent_t2_distrib {
  /// Positive definite matrix structure parameterizing the inverse scale matrix \f$Q(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite inverse scale matrix).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Precomputed state type returned by `theta_state()`.
  using state_type = mvstudent_t_state<mvgaussian2_state>;

  /// Default constructor.
  mvstudent_t2_distrib() = default;
  /**
   * \brief Construct from a matrix structure instance.
   * \param s Positive definite matrix structure defining \f$Q(\theta)\f$.
   */
  LAPLACE_HD explicit mvstudent_t2_distrib(const S& s) : structure(s) {}

  /**
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  LAPLACE_HD int dim() const { return structure.dim; }
  /**
   * \brief Return the number of non-location parameters: the structure's \f$\theta\f$ followed by \f$\nu\f$.
   * \return Parameter count `structure.n_params() + 1`.
   */
  LAPLACE_HD int n_params() const { return structure.n_params() + 1; }
  /**
   * \brief Return the number of doubles needed to store \f$Q\f$ and \f$L\f$ in `st.gauss`.
   * \return `2 * pdmat::packed_size(structure.dim)`.
   */
  LAPLACE_HD int state_size() const { return 2 * pdmat::packed_size(structure.dim); }
  /**
   * \brief Return working scratch doubles required after the state for `theta_state()`, derivative blocks, and table filling.
   * \return Maximum of `2 * pdmat::packed_size(dim()) + structure.scratch_size()` and `17 * pdmat::packed_size(dim()) + 128 * (dim() + 1) + structure.scratch_size()`.
   */
  LAPLACE_HD int scratch_size() const {
    const int g = 2 * pdmat::packed_size(structure.dim) + structure.scratch_size();
    const int u = detail_mvstudent_t::unit_scratch(structure);
    return g > u ? g : u;
  }
  /**
   * \brief Return workspace doubles needed in `w` for `expected_info_dk()`.
   * \return `2 * pdmat::packed_size(dim()) + structure.scratch_size()`, or the larger of that size and `113 * dim() * dim() + pdmat::packed_size(dim()) + structure.scratch_size()` when `S::closed_factor` is `true`.
   * \see expected_info_dk(), scratch_size(), distrib_mvstudent_t2
   */
  LAPLACE_HD int info_scratch_size() const { return detail_mvstudent_t::info_scratch<true>(structure); }

  /**
   * \brief Precompute the matrix state (`gauss`), \f$\nu\f$, dimension \f$d\f$, and \f$c_d^{(k)}(\nu)\f$ for \f$k = 0, \dots, 6\f$.
   *
   * \param theta Parameter vector of length `n_params()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated `mvstudent_t_state<mvgaussian2_state>`.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const {
    state_type st{detail_mvgaussian::theta_state<true, mvgaussian2_state>(structure, theta, w),
                  theta[structure.n_params()], structure.dim, {}};
    detail_mvstudent_t::nu_constant_dk(st.nu, st.rho, 6, st.cnu);
    return st;
  }
  /**
   * \brief Evaluate the multivariate Student t log-density \f$\ell_2(y; \mu, \theta, \nu)\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$c_d(\nu) + \frac{1}{2}\log\det Q - \frac{\nu + d}{2}\log(1 + \delta/\nu)\f$ with \f$\delta = (y - \mu)^\top Q (y - \mu)\f$.
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* mu) const {
    return detail_mvstudent_t::logpdf_value<true>(structure, st, y, mu);
  }
  /**
   * \brief Evaluate partial derivatives of the log-density with respect to \f$\mu\f$ (order `A`) and the rest \f$(\theta, \nu)\f$ (order `K`).
   *
   * Supports \f$A, K \ge 0\f$ with \f$A + K \le 6\f$. Indices in `idx` with value `structure.n_params()`
   * differentiate with respect to \f$\nu\f$; smaller indices differentiate with respect to \f$\theta\f$.
   *
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular), or `binomial_coef(dim() + A - 1, A)` (`A >= 3`, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   */
  template <int A, int K>
  LAPLACE_HD bool logpdf_dmu_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                                double* w) const {
    return detail_mvstudent_t::logpdf_dmu_dk<true, A, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate partial derivatives of the log-density with respect to \f$y\f$ (order `J`) and the rest \f$(\theta, \nu)\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dmu_dk<J, K>()` for \f$J + K \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`K >= 0`, `J + K <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), `pdmat::packed_size(dim())` (`J = 2`), or `binomial_coef(dim() + J - 1, J)` (`J >= 3`, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   */
  template <int J, int K>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                               double* w) const {
    return detail_mvstudent_t::logpdf_dy_dk<true, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate an expected Fisher information block or its `K`-th derivative with respect to \f$(\theta, \nu)\f$.
   *
   * Supports \f$0 \le K \le 4\f$ across `vector_vector`, `vector_rest`, and `rest_rest`.
   * Uses the traceless factor form from the Cholesky factor of \f$Q\f$ for \f$(\theta, \theta)\f$ pairs when `S::closed_factor` is `true`,
   * and the positive-term recurrence in \f$d\f$ for \f$I_{\nu\nu}\f$.
   *
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`0 <= K <= 4`).
   * \param st Precomputed state from `theta_state()`.
   * \param block Block of the expected information matrix.
   * \param ab Two-element index pair `(a, b)` in \f$\{0, \dots, q\}\f$ selecting an entry within `rest_rest` (`q = structure.n_params()` for \f$\nu\f$; unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (packed lower-triangular) for `vector_vector`, or length `1` (`out[0]`) for `rest_rest`.
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `vector_rest` or structurally zero derivatives.
   */
  template <int K>
  LAPLACE_HD bool expected_info_dk(const state_type& st, distrib_mv_info_block block, const int* ab, const int* idx,
                                   double* out, double* w) const {
    constexpr auto R = S::closed_factor ? detail_mvstudent_t::info_route::traceless
                                        : detail_mvstudent_t::info_route::gauss_proper;
    return detail_mvstudent_t::expected_info_dk<true, R, K>(structure, st, block, ab, idx, out, w);
  }
  /**
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Covers only the structure parameters \f$\theta\f$ (of length `structure.n_params()`);
   * \f$\nu\f$ is not stored in the table.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), mvgaussian2_distrib
   */
  LAPLACE_HD int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Each entry stores `pdmat::packed_size(dim()) + 2` doubles: a structural-nonzero
   * flag, \f$\partial_T \log\det Q\f$, and packed \f$\partial_T Q\f$.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   * \see theta_table_entries(), fill_theta_table_entry(), with_theta_table()
   */
  LAPLACE_HD int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table (\f$\theta\f$ only; \f$\nu\f$ is not in it).
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table()
   */
  LAPLACE_HD void fill_theta_table_entry(const state_type& st, int k_max, int entry, double* table, double* w) const {
    detail_mvgaussian::fill_theta_table_entry<true>(structure, st.gauss, k_max, entry, table, w);
  }
  /**
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * The buffer `table` must remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order covered by `table`.
   * \param table Precomputed table buffer of `theta_table_size(k_max)` doubles filled by `fill_theta_table_entry()`.
   * \return Copy of `st` with `gauss.table` and `gauss.table_order` set.
   * \see fill_theta_table_entry(), theta_table_entries(), theta_table_size()
   */
  LAPLACE_HD state_type with_theta_table(const state_type& st, int k_max, const double* table) const {
    state_type t = st;
    t.gauss.table = table;
    t.gauss.table_order = k_max;
    return t;
  }
  /**
   * \brief Draw one sample \f$y = \mu + L^{-\top} z / \sqrt{w}\f$ from the multivariate Student t distribution.
   *
   * Uses `st.gauss.factor` (\f$Q = L L^\top\f$), standard normal coordinates
   * \f$z_k\f$ at indices `0` through `d - 1` of `counter_uniform_source{seed, unit}`
   * via `standard_normal_by_inversion()`, and \f$w = 2 G / \nu\f$ where \f$G\f$ is
   * drawn at index `d` via `standard_gamma_by_inversion()`.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   */
  LAPLACE_HD void rng(const state_type& st, const double* mu, std::uint64_t seed, std::uint64_t unit, double* y) const {
    detail_mvstudent_t::rng_whitened<true>(structure, st, seed, unit, y);
    for (int k = 0; k < structure.dim; ++k) y[k] += mu[k];
  }
  /**
   * \brief Compute starting parameters `theta` on the host from a sample summary.
   *
   * Sets \f$\nu = 5.0\f$ at index `structure.n_params()` and computes the matrix
   * parameters from the floored inverse of \f$0.6\f$ times the sample covariance
   * via the Gaussian starting rule.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `n_params()`.
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    distrib_mv_sample_summary s = summary;
    for (double& v : s.covariance) v *= 0.6;
    detail_mvgaussian::start_value<true>(structure, s, theta);
    theta[structure.n_params()] = 5.0;
  }
  /**
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out` when \f$\nu > 1\f$, or quiet NaN otherwise.
   *
   * \param theta Parameter vector of length `n_params()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   */
  LAPLACE_HD void mean(const double* theta, const double* mu, double* out) const {
    const bool ok = theta[structure.n_params()] > 1.0;
    for (int k = 0; k < structure.dim; ++k) out[k] = ok ? mu[k] : Kokkos::Experimental::quiet_NaN_v<double>;
  }
  /**
   * \brief Write the covariance matrix \f$\frac{\nu}{\nu - 2}Q(\theta)^{-1}\f$ in packed lower-triangular order when \f$\nu > 2\f$, or quiet NaN otherwise.
   *
   * \param theta Parameter vector of length `n_params()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const {
    const double nu = theta[structure.n_params()];
    structure.inverse(theta, out, w);
    const double f = nu > 2.0 ? nu / (nu - 2.0) : Kokkos::Experimental::quiet_NaN_v<double>;
    for (int k = 0; k < pdmat::packed_size(structure.dim); ++k) out[k] *= f;
  }
};

}  // namespace laplace::distrib
