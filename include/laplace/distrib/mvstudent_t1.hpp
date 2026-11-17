// mvstudent_t1_distrib<S>: the multivariate Student t with location mu, scale
// matrix Sigma = M(theta) (a structure S of laplace::pdmat) and degrees of
// freedom nu > 0; note/distrib_multivariate.md section 15, the formulas in
// detail/mvstudent_t_common.hpp. "The rest" is (theta, nu), nu last.
//
//   l = c_d(nu) - 1/2 log det Sigma - (nu + d)/2 log1p(delta/nu),  delta = r' Sigma^-1 r.
//
// Sigma is the scale matrix, not the covariance: Var Y = nu/(nu - 2) Sigma for
// nu > 2. Over a positive semidefinite structure with a constant null space
// (weighted_sum_psdmat) the law is degenerate, on mu + col(Sigma), with the rank
// rho in place of d, Sigma^+ and log|Sigma|_+ (the classes of mvgaussian1).
#pragma once

#include <laplace/distrib/detail/mvstudent_t_common.hpp>
#include <laplace/distrib/mvgaussian1.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_mvstudent_t1
 * \brief Multivariate Student t distribution in the scale parametrization \f$\Sigma = M(\theta)\f$.
 *
 * Two partial specializations: the proper class, for a positive definite structure
 * (`pdmat::pdmat_structure`), and the degenerate class, for a positive semidefinite structure
 * with a constant null space (`constant_null_space_psdmat`).
 *
 * \tparam S Matrix structure parameterizing the scale matrix \f$\Sigma(\theta)\f$.
 * \see mvstudent_t_state, mvstudent_t2_distrib, distrib_mvstudent_t1
 */
template <class S>
struct mvstudent_t1_distrib;

/**
 * \ingroup distrib_mvstudent_t1
 * \brief Proper multivariate Student t distribution parameterized by location \f$\mu\f$, scale matrix \f$\Sigma(\theta)\f$, and degrees of freedom \f$\nu\f$.
 *
 * Models \f$Y \sim t_d(\mu, \Sigma(\theta), \nu)\f$ with scale matrix \f$\Sigma(\theta) = M(\theta)\f$
 * defined by a positive definite matrix structure `S` satisfying `pdmat::pdmat_structure`,
 * and "the rest" parameter vector \f$(\theta_0, \dots, \theta_{q-1}, \nu)\f$ with \f$\nu > 0\f$
 * at the last index \f$q\f$. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 * \see mvstudent_t_state, distrib_mvstudent_t1, distrib_multivariate
 */
template <pdmat::pdmat_structure S>
struct mvstudent_t1_distrib<S> {
  /// Positive definite matrix structure parameterizing the scale matrix \f$\Sigma(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite scale matrix).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Number of parameters: \f$\mu\f$, \f$\Sigma\f$ and \f$\nu\f$.
  static constexpr int n_params = 3;
  /// Descriptors of \f$\mu\f$ (dense), \f$\Sigma\f$ (indexed, the parameters \f$\theta\f$ of its structure) and \f$\nu\f$ (indexed, last in the indexed coordinates).
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Sigma", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth},
      {"nu", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
  /// Precomputed state type returned by `theta_state()`.
  using state_type = mvstudent_t_state<mvgaussian1_state>;

  /// Default constructor.
  mvstudent_t1_distrib() = default;
  /**
   * \brief Construct from a matrix structure instance.
   * \param s Positive definite matrix structure defining \f$\Sigma(\theta)\f$.
   */
  LAPLACE_HD explicit mvstudent_t1_distrib(const S& s) : structure(s) {}

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
   * \brief Return the number of doubles needed to store \f$\Sigma^{-1}\f$ and \f$L\f$ in `st.gauss`.
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
   * \see expected_info_dk(), scratch_size(), distrib_mvstudent_t1
   */
  LAPLACE_HD int info_scratch_size() const { return detail_mvstudent_t::info_scratch<false>(structure); }

  /**
   * \brief Precompute the matrix state (`gauss`), \f$\nu\f$, dimension \f$d\f$, and \f$c_d^{(k)}(\nu)\f$ for \f$k = 0, \dots, 6\f$.
   *
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated `mvstudent_t_state<mvgaussian1_state>`.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const {
    state_type st{detail_mvgaussian::theta_state<false, mvgaussian1_state>(structure, theta, w),
                  theta[structure.n_params()], structure.dim, {}};
    detail_mvstudent_t::nu_constant_dk(st.nu, st.rho, 6, st.cnu);
    return st;
  }
  /**
   * \brief Evaluate the multivariate Student t log-density \f$\ell_1(y; \mu, \theta, \nu)\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$c_d(\nu) - \frac{1}{2}\log\det\Sigma - \frac{\nu + d}{2}\log(1 + \delta/\nu)\f$ with \f$\delta = (y - \mu)^\top \Sigma^{-1} (y - \mu)\f$.
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* mu) const {
    return detail_mvstudent_t::logpdf_value<false>(structure, st, y, mu);
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
    return detail_mvstudent_t::logpdf_dk<false, K, detail_mv::dense_order<A...>()>(structure, st, y, mu, idx, out, w);
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
    return detail_mvstudent_t::logpdf_dy_dk<false, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate an expected Fisher information block or its `K`-th derivative with respect to \f$(\theta, \nu)\f$.
   *
   * Supports \f$0 \le K \le 4\f$ across `(0, 0)`, `(0, theta_slot)`, and `(theta_slot, theta_slot)`.
   * Uses the traceless factor form for \f$(\theta, \theta)\f$ pairs when `S::closed_factor` is `true`,
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
    return detail_mvstudent_t::expected_info_dk<false, R, K>(structure, st, slot_a, slot_b, ab, idx, out, w);
  }
  /**
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Covers only the structure parameters \f$\theta\f$ (of length `structure.n_params()`);
   * \f$\nu\f$ is not stored in the table.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table(), mvgaussian1_distrib
   */
  LAPLACE_HD int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Each entry stores `pdmat::packed_size(dim()) + 2` doubles: a structural-nonzero
   * flag, \f$\partial_T \log\det\Sigma\f$, and packed \f$\partial_T \Sigma^{-1}\f$.
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
    detail_mvgaussian::fill_theta_table_entry<false>(structure, st.gauss, k_max, entry, table, w);
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
   * \brief Draw one sample \f$y = \mu + L z / \sqrt{w}\f$ from the multivariate Student t distribution.
   *
   * Uses `st.gauss.factor` (\f$\Sigma = L L^\top\f$), standard normal coordinates
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
    detail_mvstudent_t::rng_whitened<false>(structure, st, seed, unit, y);
    for (int k = 0; k < structure.dim; ++k) y[k] += mu[k];
  }
  /**
   * \brief Compute starting parameters `theta` on the host from a sample summary.
   *
   * Sets \f$\nu = 5.0\f$ at index `structure.n_params()` and computes the matrix
   * parameters from \f$0.6\f$ times the sample covariance via the Gaussian starting rule.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `theta_size()`.
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    distrib_mv_sample_summary s = summary;
    for (double& v : s.covariance) v *= 0.6;
    detail_mvgaussian::start_value<false>(structure, s, theta);
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
   * \brief Write the covariance matrix \f$\frac{\nu}{\nu - 2}\Sigma(\theta)\f$ in packed lower-triangular order when \f$\nu > 2\f$, or quiet NaN otherwise.
   *
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const {
    const double nu = theta[structure.n_params()];
    structure.matrix(theta, out, w);
    const double f = nu > 2.0 ? nu / (nu - 2.0) : Kokkos::Experimental::quiet_NaN_v<double>;
    for (int k = 0; k < pdmat::packed_size(structure.dim); ++k) out[k] *= f;
  }
};

/**
 * \ingroup distrib_mvstudent_t1
 * \brief Degenerate multivariate Student t distribution parameterized by a singular scale matrix \f$\Sigma(\theta)\f$ with constant null space (host only).
 *
 * Models a proper Student t law of rank \f$\rho\f$ on the affine subspace
 * \f$\mu + \operatorname{col}\Sigma(\theta)\f$, replacing \f$d\f$ by \f$\rho\f$
 * and replacing \f$\Sigma^{-1}\f$ and \f$\log\det\Sigma\f$ by \f$\Sigma^+\f$
 * and \f$\log\lvert\Sigma\rvert_+\f$. Satisfies `multivariate_distrib`.
 *
 * \tparam S Positive semidefinite matrix structure satisfying `constant_null_space_psdmat`.
 * \see mvstudent_t_state, mvgaussian_rank_deficient_state, constant_null_space_psdmat, distrib_mvstudent_t1
 */
template <constant_null_space_psdmat S>
struct mvstudent_t1_distrib<S> {
  /// \name Degenerate class: singular scale matrix with a constant null space
  ///@{
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Positive semidefinite matrix structure parameterizing the scale matrix \f$\Sigma(\theta)\f$.
   */
  S structure;

  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Normalization regime (`distrib_normalization::degenerate`: singular scale matrix).
   */
  static constexpr distrib_normalization normalization = distrib_normalization::degenerate;
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Number of parameters: \f$\mu\f$, \f$\Sigma\f$ and \f$\nu\f$.
   */
  static constexpr int n_params = 3;
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Descriptors of \f$\mu\f$ (dense), \f$\Sigma\f$ (indexed, the parameters \f$\theta\f$ of its structure) and \f$\nu\f$ (indexed, last in the indexed coordinates).
   */
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, infinitely_smooth},
      {"Sigma", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth},
      {"nu", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Precomputed state type returned by `theta_state()`.
   */
  using state_type = mvstudent_t_state<mvgaussian_rank_deficient_state>;

  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Construct from a positive semidefinite matrix structure instance.
   * \param s Matrix structure defining \f$\Sigma(\theta)\f$.
   */
  explicit mvstudent_t1_distrib(const S& s) : structure(s) {}

  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the observation and location dimension \f$d\f$.
   * \return Dimension `structure.dim`.
   */
  int dim() const { return structure.dim; }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params`.
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for the matrix, 1 for \f$\nu\f$.
   */
  int param_size(int j) const { return j == 0 ? structure.dim : j == 1 ? structure.n_params() : 1; }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return where the coordinates of parameter `j` start in their array (\f$x\f$ for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params`.
   * \return 0 for \f$\mu\f$ (in \f$x\f$) and for the matrix (in \f$\theta\f$), `structure.n_params()` for \f$\nu\f$ (in \f$\theta\f$).
   */
  int param_offset(int j) const { return j == 2 ? structure.n_params() : 0; }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the number of indexed coordinates: the structure's \f$\theta\f$ followed by \f$\nu\f$.
   * \return `structure.n_params() + 1`.
   */
  int theta_size() const { return structure.n_params() + 1; }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the constant rank \f$\rho\f$ of \f$\Sigma(\theta)\f$.
   * \return Rank `structure.rank()`.
   */
  int rank() const { return structure.rank(); }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the number of `double` elements stored in `w` for `st.gauss`.
   * \return `3 * pdmat::packed_size(dim()) + structure.scratch_size()`.
   * \see theta_state(), scratch_size()
   */
  int state_size() const { return detail_mvgaussian::rank_deficient_state_size(structure); }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return working scratch doubles required after the state for `theta_state()`, derivative blocks, and table filling.
   * \return Maximum of the degenerate Gaussian scratch requirement and the block/shifted-inverse scratch requirement.
   * \see state_size(), info_scratch_size()
   */
  int scratch_size() const {
    const int p = structure.dim, np = pdmat::packed_size(p);
    const int g = detail_mvgaussian::rank_deficient_scratch<false>(structure);
    const int u = 17 * np + 128 * (p + 1) + detail_mvgaussian::shifted_inverse_scratch(p) + structure.scratch_size();
    return g > u ? g : u;
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return workspace doubles required in `w` by `expected_info_dk()`.
   * \return `pdmat::packed_size(dim())` plus the degenerate Gaussian information scratch size.
   * \see expected_info_dk(), scratch_size()
   */
  int info_scratch_size() const {
    return pdmat::packed_size(structure.dim) + detail_mvgaussian::rank_deficient_info_scratch<false>(structure);
  }

  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Precompute the rank-deficient matrix state (`gauss`), \f$\nu\f$, rank \f$\rho\f$, and \f$c_\rho^{(k)}(\nu)\f$ for \f$k = 0, \dots, 6\f$.
   *
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\nu > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated `mvstudent_t_state<mvgaussian_rank_deficient_state>`.
   */
  state_type theta_state(const double* theta, double* w) const {
    state_type st{detail_mvgaussian::rank_deficient_state<false>(structure, theta, w, true),
                  theta[structure.n_params()], structure.rank(), {}};
    detail_mvstudent_t::nu_constant_dk(st.nu, st.rho, 6, st.cnu);
    return st;
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Evaluate the degenerate multivariate Student t log-density.
   *
   * Applies the support test of the degenerate Gaussian (\f$\lVert U_0^\top r\rVert > 64\,d\,\varepsilon\,\lVert r\rVert\f$),
   * returning \f$-\infty\f$ off \f$\mu + \operatorname{col}\Sigma\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$c_\rho(\nu) - \frac{1}{2}\log\lvert\Sigma\rvert_+ - \frac{\nu + \rho}{2}\log(1 + \delta/\nu)\f$ with \f$\delta = r^\top \Sigma^+ r\f$ on the support, or \f$-\infty\f$ off it.
   */
  double logpdf(const state_type& st, const double* y, const double* mu) const {
    const double g = detail_mvgaussian::rank_deficient_logpdf<false>(structure, st.gauss, y, mu);
    if (g == -std::numeric_limits<double>::infinity()) return g;
    return detail_mvstudent_t::logpdf_value<false>(structure, st, y, mu);
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Evaluate partial derivatives of the degenerate log-density with respect to \f$\mu\f$ (order `A`) and \f$(\theta, \nu)\f$ (order `K`).
   *
   * For \f$A = 0, K = 0\f$, delegates to `logpdf()` (including the support test).
   * Higher-order blocks differentiate the law on the range using \f$\Sigma^+\f$
   * and \f$\log\lvert\Sigma\rvert_+\f$ without testing support.
   *
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (`A >= 0`; omitted means 0).
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`K >= 0`, `A + K <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$, \f$q\f$ = `structure.n_params()` (unused when `K == 0`).
   * \param out Output buffer of length `1` (`A = 0`), `dim()` (`A = 1`), `pdmat::packed_size(dim())` (`A = 2`), or `binomial_coef(dim() + A - 1, A)` (`A >= 3`, `sym_tensor` order).
   * \param w Caller workspace of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` if the block vanishes identically by structure.
   */
  template <int K, int... A>
  bool logpdf_dk(const state_type& st, const double* y, const double* mu, const int* idx, double* out,
                 double* w) const {
    if constexpr (detail_mv::dense_order<A...>() == 0 && K == 0) {
      out[0] = logpdf(st, y, mu);
      return true;
    } else {
      return detail_mvstudent_t::logpdf_dk<false, K, detail_mv::dense_order<A...>()>(
          detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss), st, y, mu, idx, out, w);
    }
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Evaluate partial derivatives of the degenerate log-density with respect to \f$y\f$ (order `J`) and \f$(\theta, \nu)\f$ (order `K`).
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
   * \ingroup distrib_mvstudent_t1
   * \brief Evaluate the expected Fisher information block of the law on the range or its `K`-th derivative in \f$(\theta, \nu)\f$.
   *
   * Supports \f$0 \le K \le 4\f$ for structures satisfying \f$\partial_{ab}\Sigma = \delta_{ab}\,\partial_a\Sigma\f$,
   * using the rank \f$\rho\f$ in place of \f$d\f$ and obtaining the Gaussian \f$(\theta, \theta)\f$ part from \f$\log\lvert\Sigma\rvert_+\f$.
   *
   * \tparam K Derivative order with respect to \f$(\theta, \nu)\f$ (`0 <= K <= 4`).
   * \param st Precomputed state from `theta_state()`.
   * \param slot_a Slot of the row parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param slot_b Slot of the column parameter: 0 for \f$\mu\f$ or `theta_slot`.
   * \param ab Two-element index pair `(a, b)` in \f$\{0, \dots, q\}\f$ selecting an entry within `(theta_slot, theta_slot)` (`q = structure.n_params()` for \f$\nu\f$; unused otherwise).
   * \param idx Sorted parameter indices of length `K` in \f$\{0, \dots, q\}\f$ (unused when `K == 0`).
   * \param out Output buffer of length `pdmat::packed_size(dim())` (`(0, 0)`) or length `1` (`(theta_slot, theta_slot)`).
   * \param w Caller workspace of at least `info_scratch_size()` doubles.
   * \return `true` if evaluated; `false` for `(0, theta_slot)` or structurally zero derivatives.
   */
  template <int K>
    requires detail_mvgaussian::exp_linear_structure<S>
  bool expected_info_dk(const state_type& st, int slot_a, int slot_b, const int* ab, const int* idx,
                        double* out, double* w) const {
    return detail_mvstudent_t::expected_info_dk<false, detail_mvstudent_t::info_route::gauss_degenerate, K>(
        detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss), st, slot_a, slot_b, ab, idx, out, w);
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   */
  int theta_table_entries(int k_max) const {
    return detail_mvgaussian::table_offset(structure.n_params(), k_max + 1);
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return the number of `double` elements required for a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return `theta_table_entries(k_max) * (pdmat::packed_size(dim()) + 2)`.
   */
  int theta_table_size(int k_max) const {
    return theta_table_entries(k_max) * detail_mvgaussian::table_entry_size(structure);
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table on the host.
   *
   * Entries must be filled one at a time because they share the structure's evaluated pseudo-determinant state.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order in \f$\theta\f$ covered by `table`.
   * \param entry Zero-based entry index satisfying `0 <= entry < theta_table_entries(k_max)`.
   * \param table Caller buffer of at least `theta_table_size(k_max)` doubles.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   */
  void fill_theta_table_entry(const state_type& st, int k_max, int entry, double* table, double* w) const {
    detail_mvgaussian::fill_theta_table_entry<false>(detail_mvgaussian::rank_deficient_view<S>(structure, st.gauss),
                                                     st.gauss, k_max, entry, table, w);
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Return a copy of `st` with a filled \f$\theta\f$-derivative table attached.
   *
   * The buffer `table` must remain alive while the returned state is used.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param k_max Largest derivative order covered by `table`.
   * \param table Precomputed table buffer of `theta_table_size(k_max)` doubles filled by `fill_theta_table_entry()`.
   * \return Copy of `st` with `gauss.table` and `gauss.table_order` set.
   */
  state_type with_theta_table(const state_type& st, int k_max, const double* table) const {
    state_type t = st;
    t.gauss.table = table;
    t.gauss.table_order = k_max;
    return t;
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Draw one sample \f$y = \mu + (I - P) L z / \sqrt{w}\f$ from the degenerate Student t distribution.
   *
   * Uses `st.gauss.factor` (\f$\Sigma + cP = L L^\top\f$), standard normal coordinates
   * \f$z\f$, and \f$w = 2 G / \nu\f$ as in the proper class, projecting onto \f$\operatorname{col}\Sigma\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   * \param tmp Caller scratch buffer of at least `dim()` doubles.
   */
  void rng(const state_type& st, const double* mu, std::uint64_t seed, std::uint64_t unit, double* y,
           double* tmp) const {
    detail_mvstudent_t::rng_whitened<false>(structure, st, seed, unit, y);
    detail_mvgaussian::null_projection(structure, y, tmp);
    for (int k = 0; k < structure.dim; ++k) y[k] += mu[k] - tmp[k];
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Compute starting parameters `theta` on the host from a sample summary.
   *
   * Projects \f$0.6\f$ times the sample covariance onto the range without an eigenvalue floor,
   * maps to structure parameters, and sets \f$\nu = 5.0\f$ at index `structure.n_params()`.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `theta_size()`.
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    distrib_mv_sample_summary s = summary;
    for (double& v : s.covariance) v *= 0.6;
    detail_mvgaussian::rank_deficient_start_value<false>(structure, s, theta);
    theta[structure.n_params()] = 5.0;
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out` when \f$\nu > 1\f$, or quiet NaN otherwise.
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   */
  void mean(const double* theta, const double* mu, double* out) const {
    const bool ok = theta[structure.n_params()] > 1.0;
    for (int k = 0; k < structure.dim; ++k) out[k] = ok ? mu[k] : std::numeric_limits<double>::quiet_NaN();
  }
  /**
   * \ingroup distrib_mvstudent_t1
   * \brief Write the covariance matrix \f$\frac{\nu}{\nu - 2}\Sigma(\theta)\f$ in packed lower-triangular order when \f$\nu > 2\f$, or quiet NaN otherwise.
   * \param theta Parameter vector of length `theta_size()` with \f$\nu\f$ at index `structure.n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  void covariance(const double* theta, double* out, double* w) const {
    const double nu = theta[structure.n_params()];
    structure.matrix(theta, out, w);
    const double f = nu > 2.0 ? nu / (nu - 2.0) : std::numeric_limits<double>::quiet_NaN();
    for (int k = 0; k < pdmat::packed_size(structure.dim); ++k) out[k] *= f;
  }
  ///@}
};

}  // namespace laplace::distrib
