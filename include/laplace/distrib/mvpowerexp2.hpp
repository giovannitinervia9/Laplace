// mvpowerexp2_distrib<S>: the multivariate power exponential of Gomez,
// Gomez-Villegas and Marin (1998) with location mu, inverse scale matrix
// Q = M(theta) (a positive definite structure S of laplace::pdmat) and shape
// beta > 0; note/distrib_multivariate.md section 7, the formulas in
// detail/mvpowerexp_common.hpp. "The rest" is (theta, beta), beta last.
//
//   l = -log Z(d, beta) + 1/2 log det Q - 1/2 delta^beta,  delta = r' Q r,
//   Z(d, beta) = 2^(1 + d/(2 beta)) pi^(d/2) Gamma(1 + d/(2 beta)) / (d Gamma(d/2)).
//
// Q is the inverse of the SCALE matrix, the precision of Y only up to the
// factor of mvpowerexp1_distrib (Giovanni, 2026-11-18: the matrix is the scale
// and is not rescaled to the covariance, as mvstudent_t2_distrib's is not), so
// that beta = 1 is exactly mvgaussian2_distrib:
//
//   Var Y = Q^-1 2^(1/beta) Gamma((d + 2)/(2 beta)) / (d Gamma(d/(2 beta))),
//
// whose factor is 1 at beta = 1.
//
// beta = 1/2 with Q = sigma^-2 I gives exp(-||r||/(2 sigma)), the group lasso,
// which is why the penalty side wants this law. It is NOT the "multivariate
// Laplace" of the literature, which is a different law, a scale mixture of
// normals with a Bessel K in its density; the confusion is common.
//
// The kink is radial: delta^beta is not smooth at y = mu, and l stays
// continuous in y there up to order ceil(2 beta) - 1, or to every order when
// beta is a positive integer, where delta^beta is a polynomial in r. The
// multivariate interface carries no place for a runtime order, so the mu
// descriptor declares 0, the worst case over beta > 0.
//
// Only the proper class is written: a positive definite structure S. The
// improper class (a positive semidefinite structure with a constant null
// space, as mvgaussian2 and mvstudent_t2 carry) is not here.
#pragma once

#include <laplace/distrib/detail/mvpowerexp_common.hpp>
#include <laplace/distrib/mvgaussian2.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_mvpowerexp2
 * \brief Multivariate power exponential distribution in the inverse scale parametrization \f$Q = M(\theta)\f$.
 *
 * Models \f$Y \sim \mathrm{PE}_d(\mu, Q(\theta)^{-1}, \beta)\f$ with inverse scale matrix
 * \f$Q(\theta) = M(\theta)\f$ defined by a positive definite matrix structure `S`
 * satisfying `pdmat::pdmat_structure`, and "the rest" parameter vector
 * \f$(\theta_0, \dots, \theta_{q-1}, \beta)\f$ with \f$\beta > 0\f$ at index
 * \f$q =\f$ `structure.n_params()`. Satisfies `multivariate_distrib`.
 *
 * The log-density, with \f$r = y - \mu\f$ and \f$\delta = r^\top Q r\f$, is
 * \f[ \ell = -\log Z(d, \beta) + \tfrac{1}{2}\log\det Q - \tfrac{1}{2}\delta^\beta, \qquad
 *     Z(d, \beta) = \frac{2^{1 + d/(2\beta)}\,\pi^{d/2}\,\Gamma(1 + d/(2\beta))}{d\,\Gamma(d/2)}. \f]
 *
 * \f$Q\f$ is the inverse of the scale matrix, the precision of \f$Y\f$ only up to the
 * same factor: \f$\operatorname{Var}Y = Q^{-1} 2^{1/\beta}\Gamma(\tfrac{d+2}{2\beta}) / (d\,\Gamma(\tfrac{d}{2\beta}))\f$,
 * equal to 1 at \f$\beta = 1\f$, where the law is the Gaussian
 * `mvgaussian2_distrib`. At \f$\beta = 1/2\f$ with \f$Q = \sigma^{-2} I\f$ it is
 * \f$\exp(-\lVert r\rVert / (2\sigma))\f$, the group lasso. It is not the
 * "multivariate Laplace" of the literature, a normal scale mixture whose density
 * carries a Bessel K function.
 *
 * \tparam S Positive definite matrix structure satisfying `pdmat::pdmat_structure`.
 * \see mvpowerexp_state, mvpowerexp1_distrib, mvgaussian2_distrib, distrib_mvpowerexp2
 */
template <pdmat::pdmat_structure S>
struct mvpowerexp2_distrib {
  /// Positive definite matrix structure parameterizing the inverse scale matrix \f$Q(\theta)\f$.
  S structure{};

  /// Normalization regime (`proper`: positive definite inverse scale matrix).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Kind of the law (`continuous`: a density, with derivatives in \f$y\f$).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of parameters: \f$\mu\f$, \f$Q\f$ and \f$\beta\f$.
  static constexpr int n_params = 3;
  /**
   * \brief Descriptors of \f$\mu\f$ (dense), \f$Q\f$ (indexed, the parameters \f$\theta\f$ of its structure) and \f$\beta\f$ (indexed scalar, last).
   *
   * \f$\mu\f$ declares regularity 0, the worst case over \f$\beta > 0\f$ of the
   * radial kink at \f$y = \mu\f$; the order actually reached there is
   * \f$\lceil 2\beta\rceil - 1\f$, or every order when \f$\beta\f$ is a positive
   * integer.
   */
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::vector, -distrib_infinity, distrib_infinity, 0},
      {"Q", distrib_mv_param_kind::matrix, 0.0, 0.0, infinitely_smooth},
      {"beta", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
  /// Precomputed state type returned by `theta_state`.
  using state_type = mvpowerexp_state<mvgaussian2_state>;

  /// Default constructor.
  mvpowerexp2_distrib() = default;
  /**
   * \brief Construct from a matrix structure instance.
   * \param s Positive definite matrix structure defining \f$Q(\theta)\f$.
   */
  LAPLACE_HD explicit mvpowerexp2_distrib(const S& s) : structure(s) {}

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
   * \return `dim()` for \f$\mu\f$, `structure.n_params()` for \f$Q\f$, 1 for \f$\beta\f$.
   */
  LAPLACE_HD int param_size(int j) const { return j == 0 ? structure.dim : j == 1 ? structure.n_params() : 1; }
  /**
   * \brief Return where the coordinates of parameter `j` start in their array (\f$x\f$ for the dense, \f$\theta\f$ for the indexed).
   * \param j Parameter index in `params`.
   * \return `structure.n_params()` for \f$\beta\f$, 0 otherwise.
   */
  LAPLACE_HD int param_offset(int j) const { return j == 2 ? structure.n_params() : 0; }
  /**
   * \brief Return the number of indexed coordinates: the structure's parameters and \f$\beta\f$.
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
    const int u = detail_mvpowerexp::unit_scratch(structure);
    return g > u ? g : u;
  }

  /**
   * \brief Precompute the matrix state (`gauss`), \f$\beta\f$, the dimension, and \f$\log Z(d, \beta)\f$ with its \f$\beta\f$-derivatives to order six.
   *
   * \param theta Parameter vector of length `theta_size()` holding \f$\theta_0, \dots, \theta_{q-1}\f$ followed by \f$\beta > 0\f$ at index `structure.n_params()`.
   * \param w Caller workspace of at least `state_size() + scratch_size()` doubles; the first `state_size()` doubles back `st.gauss` while the state is in use.
   * \return Populated `mvpowerexp_state<mvgaussian2_state>`.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const {
    state_type st{detail_mvgaussian::theta_state<true, mvgaussian2_state>(structure, theta, w),
                  theta[structure.n_params()], structure.dim, {}};
    detail_mvpowerexp::log_z_dk(st.dim, st.beta, 6, st.lz);
    return st;
  }
  /**
   * \brief Evaluate the multivariate power exponential log-density \f$\ell_1(y; \mu, \theta, \beta)\f$.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param mu Location vector of length `dim()`.
   * \return Value of \f$-\log Z(d, \beta) + \frac{1}{2}\log\det Q - \frac{1}{2}\delta^\beta\f$ with \f$\delta = (y - \mu)^\top Q (y - \mu)\f$.
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* mu) const {
    return detail_mvpowerexp::logpdf_value<true>(structure, st, y, mu);
  }
  /**
   * \brief Evaluate partial derivatives of the log-density with respect to \f$\mu\f$ (order `A`) and the rest \f$(\theta, \beta)\f$ (order `K`).
   *
   * Supports \f$A, K \ge 0\f$ with \f$A + K \le 6\f$. Indices in `idx` with value `structure.n_params()`
   * differentiate with respect to \f$\beta\f$; smaller indices differentiate with respect to \f$\theta\f$.
   * At \f$y = \mu\f$ the blocks with \f$A \ge 1\f$ or with a \f$\theta\f$ index are
   * the limits of the radial kink, infinite wherever no finite limit exists.
   *
   * \tparam K Derivative order with respect to \f$(\theta, \beta)\f$ (`K >= 0`, `A + K <= 6`).
   * \tparam A Derivative order with respect to \f$\mu\f$, the one dense parameter (`A >= 0`; omitted means 0).
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
    return detail_mvpowerexp::logpdf_dk<true, K, detail_mv::dense_order<A...>()>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Evaluate partial derivatives of the log-density with respect to \f$y\f$ (order `J`) and the rest \f$(\theta, \beta)\f$ (order `K`).
   *
   * Equal to \f$(-1)^J\f$ times `logpdf_dk<K, J>()` for \f$J + K \le 6\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order with respect to \f$(\theta, \beta)\f$ (`K >= 0`, `J + K <= 6`).
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
    return detail_mvpowerexp::logpdf_dy_dk<true, J, K>(structure, st, y, mu, idx, out, w);
  }
  /**
   * \brief Return the number of entries in a \f$\theta\f$-derivative table of orders 1 to `k_max`.
   *
   * Covers only the structure parameters \f$\theta\f$ (of length `structure.n_params()`);
   * \f$\beta\f$ is not stored in the table.
   *
   * \param k_max Largest derivative order in \f$\theta\f$ stored in the table.
   * \return Total number of sorted tuple entries across orders 1 to `k_max` (`0` if `k_max <= 0`).
   * \see theta_table_size(), fill_theta_table_entry(), with_theta_table()
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
   * \brief Fill one entry of a precomputed \f$\theta\f$-derivative table (\f$\theta\f$ only; \f$\beta\f$ is not in it).
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
   * \brief Draw one sample \f$y = \mu + L^{-\top} v\f$ from the multivariate power exponential distribution.
   *
   * Uses `st.gauss.factor` (\f$Q = L L^\top\f$, so \f$\Sigma = L^{-\top}L^{-1}\f$) and
   * \f$v = s\, z / \lVert z\rVert\f$ with \f$z\f$ the standard normal coordinates
   * at indices `0` through `d - 1` of `counter_uniform_source{seed, unit}` and
   * \f$s = (2t)^{1/(2\beta)}\f$, \f$t\f$ a standard gamma of shape
   * \f$d/(2\beta)\f$ drawn at index `d`: under the law
   * \f$\lVert v\rVert^{2\beta}/2\f$ is that gamma and the direction is uniform.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param mu Location vector of length `dim()`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   */
  LAPLACE_HD void rng(const state_type& st, const double* mu, std::uint64_t seed, std::uint64_t unit, double* y) const {
    detail_mvpowerexp::rng_whitened<true>(structure, st, seed, unit, y);
    for (int k = 0; k < structure.dim; ++k) y[k] += mu[k];
  }
  /**
   * \brief Compute starting parameters `theta` on the host from a sample summary.
   *
   * Sets \f$\beta = 1\f$ at index `structure.n_params()`, where the law is the
   * Gaussian, and computes the matrix parameters from the sample covariance by
   * the Gaussian starting rule.
   *
   * \param summary Sample mean and covariance summary produced by `summarize_mv_sample()`.
   * \param theta Output parameter vector of length `theta_size()`.
   */
  void start_value(const distrib_mv_sample_summary& summary, double* theta) const {
    detail_mvgaussian::start_value<true>(structure, summary, theta);
    theta[structure.n_params()] = 1.0;
  }
  /**
   * \brief Compute starting parameters `theta` on the host from the sample itself.
   *
   * \f$\beta\f$ solves \f$d^2\,\Gamma(\tfrac{d}{2\beta})\,\Gamma(\tfrac{d+4}{2\beta}) / \Gamma(\tfrac{d+2}{2\beta})^2 = b_2\f$,
   * the law's multivariate kurtosis equated to Mardia's sample kurtosis
   * \f$b_2 = n^{-1} \sum_i ((y_i - \bar y)^\top S^{-1} (y_i - \bar y))^2\f$ (\f$S\f$ the covariance with
   * divisor \f$n\f$), by bisection on \f$\log\beta\f$ over \f$[1/4, 10]\f$ and clamped to the
   * nearer end outside it; the matrix parameters come from \f$S\f$ divided by the covariance factor
   * at that \f$\beta\f$, so that the covariance of the start equals \f$S\f$. With fewer than
   * \f$d + 2\f$ units or a singular \f$S\f$, \f$\beta = 1\f$ and the summary rule above.
   *
   * \param y Pointer to `n * dim()` values on the host, unit `i` at `y[i * dim() ...]`.
   * \param n Number of units.
   * \param theta Output parameter vector of length `theta_size()`.
   */
  void start_value(const double* y, int n, double* theta) const {
    detail_mvpowerexp::start_from_sample<true>(structure, y, n, theta);
  }
  /**
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out`.
   *
   * Every moment of the law is finite, so the mean exists for every \f$\beta > 0\f$.
   *
   * \param theta Parameter vector of length `theta_size()` (unused).
   * \param mu Location vector of length `dim()`.
   * \param out Output mean vector of length `dim()`.
   */
  LAPLACE_HD void mean(const double* theta, const double* mu, double* out) const {
    (void)theta;
    for (int k = 0; k < structure.dim; ++k) out[k] = mu[k];
  }
  /**
   * \brief Write the covariance matrix in packed lower-triangular order.
   *
   * \f$\operatorname{Var}Y = Q(\theta)^{-1} 2^{1/\beta}\,\Gamma(\tfrac{d+2}{2\beta}) / (d\,\Gamma(\tfrac{d}{2\beta}))\f$,
   * a factor equal to 1 at \f$\beta = 1\f$.
   *
   * \param theta Parameter vector of length `theta_size()` with \f$\beta\f$ at index `structure.n_params()`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `structure.scratch_size()` doubles.
   */
  LAPLACE_HD void covariance(const double* theta, double* out, double* w) const {
    structure.inverse(theta, out, w);
    const double f = detail_mvpowerexp::covariance_factor(structure.dim, theta[structure.n_params()]);
    for (int k = 0; k < pdmat::packed_size(structure.dim); ++k) out[k] *= f;
  }
};

}  // namespace laplace::distrib
