// mvstudent_t2_distrib<S>: the multivariate Student t with location mu, inverse
// scale matrix Q = M(theta) (a positive definite structure S of laplace::pdmat)
// and degrees of freedom nu > 0; note/distrib_multivariate.md section 15, the
// formulas in detail/mvstudent_t_common.hpp. "The rest" is (theta, nu), nu last.
//
//   l = c_d(nu) + 1/2 log det Q - (nu + d)/2 log1p(delta/nu),  delta = r' Q r.
//
// Q is the inverse of the scale matrix, the precision of Y only up to
// (nu - 2)/nu: Var Y = nu/(nu - 2) Q^-1 for nu > 2. Over a positive
// semidefinite structure with a constant null space (weighted_sum_psdmat) the
// law is improper, flat along the null space, with the rank rho in place of d
// and log|Q|_+ (note/distrib_multivariate.md section 17, the classes of
// mvgaussian2).
#pragma once

#include <laplace/distrib/detail/mvstudent_t_common.hpp>
#include <laplace/distrib/mvgaussian2.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_mvstudent_t2
 * \brief Multivariate Student t distribution in the inverse scale parametrization \f$Q = M(\theta)\f$.
 *
 * Provides two partial specializations: the proper class for a positive definite
 * structure (`pdmat::pdmat_structure`), and the improper class for a positive
 * semidefinite structure with a constant null space (`constant_null_space_psdmat`).
 *
 * \tparam S Matrix structure parameterizing the inverse scale matrix \f$Q(\theta)\f$.
 * \see mvstudent_t_state, mvstudent_t1_distrib, distrib_mvstudent_t2, distrib_multivariate
 */
template <class S>
struct mvstudent_t2_distrib;

/**
 * \ingroup distrib_mvstudent_t2
 * \brief Proper multivariate Student t distribution parameterized by location \f$\mu\f$, inverse scale matrix \f$Q(\theta)\f$, and degrees of freedom \f$\nu\f$.
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
struct mvstudent_t2_distrib<S> {
  /// Positive definite matrix structure parameterizing the inverse scale matrix \f$Q(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite inverse scale matrix).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Number of parameters: \f$\mu\f$, \f$Q\f$ and \f$\nu\f$.
  static constexpr int n_params = 3;
  /// Descriptors of \f$\mu\f$ (dense), \f$Q\f$ (indexed, the parameters \f$\theta\f$ of its structure) and \f$\nu\f$ (indexed, last in the indexed coordinates).
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Q", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth},
      {"nu", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
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
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params`.
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for the matrix, 1 for \f$\nu\f$.
   */
  LAPLACE_HD int param_size(int j) const { return j == 0 ? structure.dim : j == 1 ? structure.n_params() : 1; }
  /**
   * \brief Return where the coordinates of parameter `j` start in their array (\f$x\f$ for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params`.
   * \return 0 for \f$\mu\f$ (in \f$x\f$) and for the matrix (in \f$\theta\f$), `structure.n_params()` for \f$\nu\f$ (in \f$\theta\f$).
   */
  LAPLACE_HD int param_offset(int j) const { return j == 2 ? structure.n_params() : 0; }
  /**
   * \brief Return the number of indexed coordinates: the structure's \f$\theta\f$ followed by \f$\nu\f$.
   * \return `structure.n_params() + 1`.
   */
  LAPLACE_HD int theta_size() const { return structure.n_params() + 1; }
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
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
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
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (`A >= 0`; omitted means 0).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), `pdmat::packed_size(dim())` (`A = 2`, packed lower-triangular), or `binomial_coef(dim() + A - 1, A)` (`A >= 3`, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                            double* w) const {
    return detail_mvstudent_t::logpdf_dk<true, K, detail_mv::dense_order<A...>()>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate partial derivatives of the log-density with respect to \f$y\f$ (order `J`) and the rest \f$(\theta, \nu)\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dk<K, J>()` for \f$J + K \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`K >= 0`, `J + K <= 6`).
   * \tparam A Derivative order with respect to \f$\mu\f$ (`A >= 0`, `J + K + A <= 6`; omitted means 0). With `A > 0`
   *         the output is the Kronecker product, the \f$y\f$ factor first, of the packed tensors of order `J` and `A`
   *         over \f$d\f$, and `w` holds `scratch_size() + sym_tensor_size(dim(), J + A)` doubles.
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`J = 0`), `dim()` (`J = 1`), `pdmat::packed_size(dim())` (`J = 2`), or `binomial_coef(dim() + J - 1, J)` (`J >= 3`, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                               double* w) const {
    if constexpr (detail_mv::dense_order<A...>() > 0)
      return detail_mv::location_dy_dk<J, K, detail_mv::dense_order<A...>()>(*this, st, y, mu, idx, out, w);
    return detail_mvstudent_t::logpdf_dy_dk<true, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate an expected Fisher information block or its `K`-th derivative with respect to \f$(\theta, \nu)\f$.
   *
   * Supports \f$0 \le K \le 4\f$ across `(0, 0)`, `(0, theta_slot)`, and `(theta_slot, theta_slot)`.
   * Uses the traceless factor form from the Cholesky factor of \f$Q\f$ for \f$(\theta, \theta)\f$ pairs when `S::closed_factor` is `true`,
   * and the positive-term recurrence in \f$d\f$ for \f$I_{\nu\nu}\f$.
   *
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`0 <= K <= 4`).
   * \param st Precomputed state from `theta_state()`.
   * \param slot_a Slot of the row parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param slot_b Slot of the column parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param ab Two-element index pair `(a, b)` in \f$\{0, \dots, q\}\f$ selecting an entry within `(theta_slot, theta_slot)` (`q = structure.n_params()` for \f$\nu\f$; unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (packed lower-triangular) for `(0, 0)`, or length `1` (`out[0]`) for `(theta_slot, theta_slot)`.
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `(0, theta_slot)` or structurally zero derivatives.
   */
  template <int K>
  LAPLACE_HD bool expected_info_dk(const state_type& st, int slot_a, int slot_b, const int* ab, const int* idx,
                                   double* out, double* w) const {
    constexpr auto R = S::closed_factor ? detail_mvstudent_t::info_route::traceless
                                        : detail_mvstudent_t::info_route::gauss_proper;
    return detail_mvstudent_t::expected_info_dk<true, R, K>(structure, st, slot_a, slot_b, ab, idx, out, w);
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
   * \param theta Output parameter vector of length `theta_size()`.
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
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
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
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
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

/**
 * \ingroup distrib_mvstudent_t2
 * \brief Improper (intrinsic) multivariate Student t distribution parameterized by a singular inverse scale matrix \f$Q(\theta)\f$ with constant null space (host only).
 *
 * Models an improper density flat along the null space of \f$Q(\theta)\f$ of constant
 * rank \f$\rho\f$, with log-density
 * \f$\ell_2 = c_\rho(\nu) + \frac{1}{2}\log\lvert Q\rvert_+ - \frac{\nu + \rho}{2}\log(1 + \delta/\nu)\f$
 * (\f$\delta = r^\top Q r\f$, \f$r = y - \mu\f$). Draws, moments, and expected
 * Fisher information belong to the constrained proper law \f$U_0^\top(y - \mu) = 0\f$.
 * Runs on the host only, as required by `S`. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive semidefinite matrix structure satisfying `constant_null_space_psdmat`.
 * \see mvstudent_t_state, mvgaussian_rank_deficient_state, constant_null_space_psdmat, mvgaussian2_distrib, mvstudent_t1_distrib, distrib_mvstudent_t2
 */
template <constant_null_space_psdmat S>
struct mvstudent_t2_distrib<S> {
  /// \name Improper class: singular inverse scale matrix with a constant null space
  ///@{
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Positive semidefinite matrix structure parameterizing the inverse scale matrix \f$Q(\theta)\f$.
   */
  S structure;

  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Normalization regime (`distrib_normalization::improper`: singular inverse scale matrix).
   */
  static constexpr distrib_normalization normalization = distrib_normalization::improper;
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Number of parameters: \f$\mu\f$, \f$Q\f$, and \f$\nu\f$.
   */
  static constexpr int n_params = 3;
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Descriptors of \f$\mu\f$ (dense), \f$Q\f$ (indexed, structure parameters \f$\theta\f$), and \f$\nu\f$ (indexed, last in the indexed array).
   */
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Q", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth},
      {"nu", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Precomputed state type returned by `theta_state()` and `constrained_state()`.
   * \see mvstudent_t_state, mvgaussian_rank_deficient_state, distrib_mvstudent_t2
   */
  using state_type = mvstudent_t_state<mvgaussian_rank_deficient_state>;

  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Construct from a positive semidefinite matrix structure instance.
   * \param s Matrix structure defining \f$Q(\theta)\f$.
   * \see distrib_mvstudent_t2
   */
  explicit mvstudent_t2_distrib(const S& s) : structure(s) {}

  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   * \see rank(), distrib_mvstudent_t2
   */
  int dim() const { return structure.dim; }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params` (`0` for \f$\mu\f$, `1` for \f$Q\f$, `2` for \f$\nu\f$).
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for \f$Q\f$, and \f$1\f$ for \f$\nu\f$.
   * \see param_offset(), theta_size(), distrib_mvstudent_t2
   */
  int param_size(int j) const { return j == 0 ? structure.dim : j == 1 ? structure.n_params() : 1; }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return where the coordinates of parameter `j` start in their array (`x` for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params` (`0` for \f$\mu\f$, `1` for \f$Q\f$, `2` for \f$\nu\f$).
   * \return \f$0\f$ for \f$\mu\f$ (in `x`) and \f$Q\f$ (in \f$\theta\f$), and `structure.n_params()` for \f$\nu\f$ (in \f$\theta\f$).
   * \see param_size(), theta_size(), distrib_mvstudent_t2
   */
  int param_offset(int j) const { return j == 2 ? structure.n_params() : 0; }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the number of indexed coordinates: the structure's \f$\theta\f$ followed by \f$\nu\f$.
   * \return `structure.n_params() + 1`.
   * \see param_size(), param_offset(), distrib_mvstudent_t2
   */
  int theta_size() const { return structure.n_params() + 1; }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the constant rank \f$\rho\f$ of \f$Q(\theta)\f$.
   * \return Rank `structure.rank()`.
   * \see dim(), distrib_mvstudent_t2
   */
  int rank() const { return structure.rank(); }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the number of `double` elements stored in `w` for `st.gauss`.
   * \return `3 * pdmat::packed_size(dim()) + structure.scratch_size()`.
   * \see theta_state(), constrained_state(), scratch_size(), distrib_mvstudent_t2
   */
  int state_size() const { return detail_mvgaussian::rank_deficient_state_size(structure); }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return working scratch doubles required after the state for `theta_state()`, `constrained_state()`, derivative blocks, and table filling.
   * \return Maximum of the improper Gaussian scratch requirement and the block scratch requirement.
   * \see state_size(), info_scratch_size(), distrib_mvstudent_t2
   */
  int scratch_size() const {
    const int p = structure.dim, np = pdmat::packed_size(p);
    const int g = detail_mvgaussian::rank_deficient_scratch<true>(structure);
    const int u = 17 * np + 128 * (p + 1) + detail_mvgaussian::shifted_inverse_scratch(p) + structure.scratch_size();
    return g > u ? g : u;
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return workspace doubles required in `w` by `expected_info_dk()`.
   * \return `pdmat::packed_size(dim()) + structure.scratch_size()`.
   * \see expected_info_dk(), scratch_size(), distrib_mvstudent_t2
   */
  int info_scratch_size() const {
    return pdmat::packed_size(structure.dim) + detail_mvgaussian::rank_deficient_info_scratch<true>(structure);
  }

  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Precompute \f$\log\lvert Q\rvert_+\f$ and packed \f$Q\f$ (`gauss`, without a factor), \f$\nu\f$, rank \f$\rho\f$, and \f$c_\rho^{(k)}(\nu)\f$ for \f$k = 0, \dots, 6\f$.
   *
   * Leaves `st.gauss.factor` and `st.gauss.shifted_inverse` as `nullptr`.
   * Both `theta` and the first `state_size()` doubles of `w` must remain alive
   * while the returned state is used.
   *
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated `mvstudent_t_state<mvgaussian_rank_deficient_state>`.
   * \see constrained_state(), state_size(), distrib_mvstudent_t2
   */
  state_type theta_state(const double* theta, double* w) const { return make_state(theta, w, false); }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Precompute the state of `theta_state()` together with the Cholesky factor of \f$Q + cP\f$ and \f$(Q + cP)^{-1}\f$.
   *
   * Uses \f$P = U_0 U_0^\top\f$ and fixed shift \f$c = \operatorname{tr} Q / \rho\f$
   * (\f$1\f$ when \f$\rho = 0\f$ or \f$\operatorname{tr} Q \le 0\f$), as `mvgaussian2_distrib::constrained_state()`.
   * Required by `constrained_rng()` and `constrained_covariance()`.
   *
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated state with `gauss.factor` and `gauss.shifted_inverse` set.
   * \see theta_state(), constrained_rng(), constrained_covariance(), distrib_mvstudent_t2
   */
  state_type constrained_state(const double* theta, double* w) const { return make_state(theta, w, true); }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Evaluate the improper multivariate Student t log-density \f$\ell_2(y; \mu, \theta, \nu)\f$.
   *
   * Defined for every \f$y \in \mathbb{R}^d\f$ with no support test; invariant to shifts
   * along the null space of \f$Q\f$.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$c_\rho(\nu) + \frac{1}{2}\log\lvert Q\rvert_+ - \frac{\nu + \rho}{2}\log(1 + \delta/\nu)\f$ with \f$\delta = (y - \mu)^\top Q (y - \mu)\f$.
   * \see logpdf_dk(), theta_state(), distrib_mvstudent_t2
   */
  double logpdf(const state_type& st, const double* y, const double* mu) const {
    return detail_mvstudent_t::logpdf_value<true>(structure, st, y, mu);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Evaluate partial derivatives of the improper log-density with respect to \f$\mu\f$ (order `A`) and \f$(\theta, \nu)\f$ (order `K`).
   *
   * Evaluates the proper formulas with \f$\rho\f$, \f$Q\f$, and \f$\log\lvert Q\rvert_+\f$
   * for \f$A, K \ge 0\f$ with \f$A + K \le 6\f$, using double-double accumulation for
   * \f$A + K_\theta \ge 3\f$. Indices in `idx` equal to `structure.n_params()` differentiate
   * with respect to \f$\nu\f$; smaller indices differentiate with respect to \f$\theta\f$.
   *
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (\f$K \ge 0\f$, \f$A + K \le 6\f$).
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (\f$A \ge 0\f$; omitted means \f$0\f$).
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when \f$K = 0\f$).
   * \param out Output buffer of length \f$1\f$ (\f$A = 0\f$), `dim()` (\f$A = 1\f$), `pdmat::packed_size(dim())` (\f$A = 2\f$, packed lower-triangular), or `binomial_coef(dim() + A - 1, A)` (\f$A \ge 3\f$, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   * \see logpdf(), logpdf_dy_dk(), with_theta_table(), distrib_mvstudent_t2
   */
  template <int K, int... A>
  bool logpdf_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                 double* w) const {
    return detail_mvstudent_t::logpdf_dk<true, K, detail_mv::dense_order<A...>()>(
        detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss), st, y, mu, idx, out, w);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Evaluate partial derivatives of the improper log-density with respect to \f$y\f$ (order `J`) and \f$(\theta, \nu)\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dk<K, J>()` when \f$A = 0\f$, for \f$J + K + A \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (\f$J \ge 0\f$).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam A Derivative order with respect to \f$\mu\f$ (\f$A \ge 0\f$, \f$J + K + A \le 6\f$; omitted means \f$0\f$). With \f$A > 0\f$
   *         the output is the Kronecker product, the \f$y\f$ factor first, of the packed tensors of order `J` and `A`
   *         over \f$d\f$, and `w` holds `scratch_size() + sym_tensor_size(dim(), J + A)` doubles.
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when \f$K = 0\f$).
   * \param out Output buffer of length \f$1\f$ (\f$J = 0\f$), `dim()` (\f$J = 1\f$), `pdmat::packed_size(dim())` (\f$J = 2\f$), or `binomial_coef(dim() + J - 1, J)` (\f$J \ge 3\f$, `sym_tensor` order), or the Kronecker product size when \f$A > 0\f$.
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   * \see logpdf_dk(), distrib_mvstudent_t2
   */
  template <int J, int K, int... A>
  bool logpdf_dy_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                    double* w) const {
    if constexpr (detail_mv::dense_order<A...>() > 0)
      return detail_mv::location_dy_dk<J, K, detail_mv::dense_order<A...>()>(*this, st, y, mu, idx, out, w);
    if (!logpdf_dk<K, J>(st, y, mu, idx, out, w)) return false;
    if constexpr (J % 2 == 1) {
      const int n = J == 1 ? dim() : binomial_coef(dim() + J - 1, J);
      for (int k = 0; k < n; ++k) out[k] = -out[k];
    }
    return true;
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Evaluate the expected Fisher information block of the constrained law or its `K`-th derivative in \f$(\theta, \nu)\f$.
   *
   * Defined for \f$0 \le K \le 4\f$ on structures satisfying \f$\partial_{ab} Q = \delta_{ab}\,\partial_a Q\f$
   * (`weighted_sum_psdmat`). With \f$\kappa = (\nu + \rho)/(\nu + \rho + 2)\f$ and
   * \f$t_a = -\partial_a\log\lvert Q\rvert_+\f$, `(0, 0)` writes \f$\kappa Q\f$ and its
   * derivatives, `(0, theta_slot)` returns `false`, and `(theta_slot, theta_slot)`
   * writes \f$\kappa I^G_{ab} - t_a t_b / (2(\nu + \rho + 2))\f$ (with \f$I^G\f$ from
   * \f$\log\lvert Q\rvert_+\f$), \f$-t_a / ((\nu + \rho)(\nu + \rho + 2))\f$, or
   * \f$I_{\nu\nu}\f$ with \f$\rho\f$; with a single matrix, \f$I_{\theta\theta}\f$ evaluates
   * \f$\nu t^2 / (2\rho(\nu + \rho + 2))\f$ (\ref mvstudent_t2_improper "the improper class").
   *
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (\f$0 \le K \le 4\f$).
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param slot_a Slot of the row parameter: \f$0\f$ for \f$\mu\f$ or `theta_slot`.
   * \param slot_b Slot of the column parameter: \f$0\f$ for \f$\mu\f$ or `theta_slot`.
   * \param ab Two-element index pair `(a, b)` in \f$\{0, \dots, q\}\f$ selecting an entry within `(theta_slot, theta_slot)` (\f$q\f$ = `structure.n_params()` for \f$\nu\f$; unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$ (unused when \f$K = 0\f$).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (`(0, 0)`) or length \f$1\f$ (`(theta_slot, theta_slot)`).
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `(0, theta_slot)` or structurally zero derivatives.
   * \see info_scratch_size(), theta_slot, distrib_mvstudent_t2
   */
  template <int K>
    requires detail_mvgaussian::exp_linear_structure<S>
  bool expected_info_dk(const state_type& st, int slot_a, int slot_b, const int* ab, const int* idx,
                        double* out, double* w) const {
    return detail_mvstudent_t::expected_info_dk<true, detail_mvstudent_t::info_route::gauss_degenerate, K>(
        detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss), st, slot_a, slot_b, ab, idx, out, w);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders \f$1\f$ to `k_max`.
   *
   * Covers only the structure parameters \f$\theta\f$; \f$\nu\f$ is not stored in the table.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders \f$1\f$ to `k_max` (\f$0\f$ if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), distrib_mvstudent_t2
   */
  int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders \f$1\f$ to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   * \see theta_table_entries(), fill_theta_table_entry(), with_theta_table(), distrib_mvstudent_t2
   */
  int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table on the host, one entry at a time.
   *
   * Entries must be filled sequentially because they share the structure's evaluated
   * pseudo-determinant state in `st.gauss.pdet`.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \see theta_table_entries(), theta_table_size(), with_theta_table(), distrib_mvstudent_t2
   */
  void fill_theta_table_entry(const state_type& st, int k_max, int entry, double* table, double* w) const {
    detail_mvgaussian::fill_theta_table_entry<true>(detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss),
                                                    st.gauss, k_max, entry, table, w);
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * The buffer `table` must remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()` or `constrained_state()`.
   * \param k_max Largest derivative order covered by `table`.
   * \param table Precomputed table buffer of `theta_table_size(k_max)` doubles filled by `fill_theta_table_entry()`.
   * \return Copy of `st` with `gauss.table` and `gauss.table_order` set.
   * \see fill_theta_table_entry(), theta_table_entries(), theta_table_size(), distrib_mvstudent_t2
   */
  state_type with_theta_table(const state_type& st, int k_max, const double* table) const {
    state_type t = st;
    t.gauss.table = table;
    t.gauss.table_order = k_max;
    return t;
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Draw one sample \f$y = \mu + (I - P) L^{-\top} z / \sqrt{w}\f$ from the constrained law \f$U_0^\top(y - \mu) = 0\f$.
   *
   * Uses `cst.gauss.factor` (\f$Q + cP = L L^\top\f$), standard normal coordinates \f$z_k\f$ at indices
   * \f$0\f$ through \f$d - 1\f$ of `counter_uniform_source` `{seed, unit}` via `standard_normal_by_inversion()`,
   * and \f$w = 2 G / \nu\f$ with \f$G\f$ drawn at index \f$d\f$ by `standard_gamma_by_inversion()`; the
   * null-space component is zero.
   *
   * \param cst Precomputed state from `constrained_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   * \param tmp Caller scratch buffer of at least `dim()` doubles.
   * \see constrained_state(), constrained_mean(), constrained_covariance(), distrib_mvstudent_t2
   */
  void constrained_rng(const state_type& cst, const double* mu, std::uint64_t seed, std::uint64_t unit, double* y,
                       double* tmp) const {
    detail_mvstudent_t::rng_whitened<true>(structure, cst, seed, unit, y);
    detail_mvgaussian::null_projection(structure, y, tmp);
    for (int k = 0; k < structure.dim; ++k) y[k] += mu[k] - tmp[k];
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Write the mean \f$\mu\f$ of the constrained law \f$U_0^\top(y - \mu) = 0\f$ into `out` when \f$\nu > 1\f$, or quiet NaN in every entry otherwise.
   *
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   * \see constrained_covariance(), constrained_rng(), distrib_mvstudent_t2
   */
  void constrained_mean(const double* theta, const double* mu, double* out) const {
    const bool ok = theta[structure.n_params()] > 1.0;
    for (int k = 0; k < structure.dim; ++k) out[k] = ok ? mu[k] : std::numeric_limits<double>::quiet_NaN();
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Write the covariance \f$\frac{\nu}{\nu - 2} Q^+\f$ of the constrained law in packed lower-triangular order when \f$\nu > 2\f$, or quiet NaN otherwise.
   *
   * Computes \f$Q^+ = (Q + cP)^{-1} - P/c\f$ from `cst` and scales by \f$\nu / (\nu - 2)\f$
   * for \f$\nu > 2\f$, writing quiet NaN in every entry for \f$\nu \le 2\f$.
   *
   * \param cst Precomputed state from `constrained_state()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \see constrained_state(), constrained_mean(), constrained_rng(), distrib_mvstudent_t2
   */
  void constrained_covariance(const state_type& cst, double* out) const {
    detail_mvgaussian::rank_deficient_view<S>(structure, cst.gauss).inverse(cst.gauss.theta, out, nullptr);
    const double f = cst.nu > 2.0 ? cst.nu / (cst.nu - 2.0) : std::numeric_limits<double>::quiet_NaN();
    for (int k = 0; k < pdmat::packed_size(structure.dim); ++k) out[k] *= f;
  }
  /**
   * \ingroup distrib_mvstudent_t2
   * \brief Compute starting parameters `theta` on the host from a sample summary.
   *
   * Projects \f$0.6\f$ times the sample covariance \f$C\f$ onto the range as
   * \f$C_p = (I - P)(0.6\,C)(I - P)\f$ without an eigenvalue floor, forms its
   * pseudo-inverse \f$C_p^+ = (C_p + cP)^{-1} - P/c\f$, maps it to structure
   * parameters with `structure.params_from_matrix()`, and sets \f$\nu = 5\f$ at
   * index `structure.n_params()`.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `theta_size()`.
   * \see summarize_mv_sample(), distrib_mv_sample_summary, distrib_mvstudent_t2
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    distrib_mv_sample_summary s = summary;
    for (double& v : s.covariance) v *= 0.6;
    detail_mvgaussian::rank_deficient_start_value<true>(structure, s, theta);
    theta[structure.n_params()] = 5.0;
  }
  ///@}

 private:
  state_type make_state(const double* theta, double* w, bool with_factor) const {
    state_type st{detail_mvgaussian::rank_deficient_state<true>(structure, theta, w, with_factor),
                  theta[structure.n_params()], structure.rank(), {}};
    detail_mvstudent_t::nu_constant_dk(st.nu, st.rho, 6, st.cnu);
    return st;
  }
};

}  // namespace laplace::distrib
