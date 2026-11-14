// lognormal1_distrib (laplace::distrib): the lognormal distribution in the mean mu (real) and the
// variance v = sigma^2 > 0 of log Y, so that Y > 0 and
//
//   l(mu, v; y) = l_gaussian2(mu, v; x) - x,   x = log y,
//
// where l_gaussian2 is gaussian2_distrib's log-density and the term -x = -log y carries no
// parameter. Every parameter derivative, the expected information and the log F and log(1 - F)
// families are therefore gaussian2_distrib's at x = log y, unchanged; this file forwards to it and
// adds only what the change of variable makes different: the value (minus log y), the quantile
// (exp of gaussian2's), the support (0, infinity), and the y-derivatives.
//
// y-derivatives. With theta = y d/dy and x = log y, y^J d^J/dy^J = theta (theta - 1) ... (theta - J + 1),
// and theta acts on a function of x as d/dx. So for any function f of x,
//
//   d^J/dy^J f(log y) = y^(-J) sum_(k = 1..J) s(J, k) f^(k)(log y),
//
// with s(J, k) the signed Stirling numbers of the first kind (a table of the coefficients is in
// tools/reference/distrib_lognormal.py, which checks the identity against SymPy). Applied to
// f = gaussian2's log-density, f^(k) is gaussian2_distrib::logpdf_dy_dk<k, K>(x) (its derivatives in
// its argument, which is x here). The term -log y adds, to the value only (K = 0), the derivative
//
//   d^J/dy^J (-log y) = (-1)^J (J - 1)! y^(-J).
//
// The coefficients for J = 1 .. 6 are written one function per J (detail_lognormal1::dx_comb_J).
//
// Expected information: gaussian2's diag(1/v, 1/(2 v^2)) in (mu, v), unchanged (the density of Y
// differs from that of log Y by a function of y alone).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::distrib {

namespace detail_lognormal1 {

// The combination sum_k s(J, k) g_k of the x-derivatives g_k = d^k/dx^k d^K/dtheta^K of gaussian2's
// log-density at x, for J = 1 .. 6: the signed Stirling numbers of the first kind s(J, k) (row J
// of the table in tools/reference/distrib_lognormal.py) multiply gaussian2's logpdf_dy_dk<k, K>.
// Each function returns that combination without the factor y^(-J).

template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_1(const gaussian2_distrib& g, double x) {
  return g.logpdf_dy_dk<1, K>(x);
}

// s(2, .) = (-1, 1)
template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_2(const gaussian2_distrib& g, double x) {
  const auto g1 = g.logpdf_dy_dk<1, K>(x), g2 = g.logpdf_dy_dk<2, K>(x);
  sym_tensor<2, K> t;
  for (int i = 0; i < sym_tensor<2, K>::size; ++i) t[i] = -g1[i] + g2[i];
  return t;
}

// s(3, .) = (2, -3, 1)
template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_3(const gaussian2_distrib& g, double x) {
  const auto g1 = g.logpdf_dy_dk<1, K>(x), g2 = g.logpdf_dy_dk<2, K>(x), g3 = g.logpdf_dy_dk<3, K>(x);
  sym_tensor<2, K> t;
  for (int i = 0; i < sym_tensor<2, K>::size; ++i) t[i] = 2.0 * g1[i] - 3.0 * g2[i] + g3[i];
  return t;
}

// s(4, .) = (-6, 11, -6, 1)
template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_4(const gaussian2_distrib& g, double x) {
  const auto g1 = g.logpdf_dy_dk<1, K>(x), g2 = g.logpdf_dy_dk<2, K>(x), g3 = g.logpdf_dy_dk<3, K>(x),
             g4 = g.logpdf_dy_dk<4, K>(x);
  sym_tensor<2, K> t;
  for (int i = 0; i < sym_tensor<2, K>::size; ++i) t[i] = -6.0 * g1[i] + 11.0 * g2[i] - 6.0 * g3[i] + g4[i];
  return t;
}

// s(5, .) = (24, -50, 35, -10, 1)
template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_5(const gaussian2_distrib& g, double x) {
  const auto g1 = g.logpdf_dy_dk<1, K>(x), g2 = g.logpdf_dy_dk<2, K>(x), g3 = g.logpdf_dy_dk<3, K>(x),
             g4 = g.logpdf_dy_dk<4, K>(x), g5 = g.logpdf_dy_dk<5, K>(x);
  sym_tensor<2, K> t;
  for (int i = 0; i < sym_tensor<2, K>::size; ++i)
    t[i] = 24.0 * g1[i] - 50.0 * g2[i] + 35.0 * g3[i] - 10.0 * g4[i] + g5[i];
  return t;
}

// s(6, .) = (-120, 274, -225, 85, -15, 1)
template <int K>
LAPLACE_HD sym_tensor<2, K> dx_comb_6(const gaussian2_distrib& g, double x) {
  const auto g1 = g.logpdf_dy_dk<1, K>(x), g2 = g.logpdf_dy_dk<2, K>(x), g3 = g.logpdf_dy_dk<3, K>(x),
             g4 = g.logpdf_dy_dk<4, K>(x), g5 = g.logpdf_dy_dk<5, K>(x), g6 = g.logpdf_dy_dk<6, K>(x);
  sym_tensor<2, K> t;
  for (int i = 0; i < sym_tensor<2, K>::size; ++i)
    t[i] = -120.0 * g1[i] + 274.0 * g2[i] - 225.0 * g3[i] + 85.0 * g4[i] - 15.0 * g5[i] + g6[i];
  return t;
}

// The y-derivatives of the lognormal1 log-density in the residual r = log y - mu. The log-density depends on
// (mu, y) only through r (and s = v), so the combination is evaluated at mu = 0 and the argument r. Callers
// that have r without the cancellation of log y - mu (lognormal2_distrib, where mu = log m - s/2 is near 4 in
// size while r may be about s/2) use this form; lognormal1_distrib itself uses r = log y - mu.
template <int J, int K>
LAPLACE_HD sym_tensor<2, K> dy_dk_residual(double y, double r, double s) {
  static_assert(J >= 0 && K >= 0 && J + K <= 6, "lognormal1_distrib: J + K must be at most 6");
  const gaussian2_distrib g{0.0, s};
  if constexpr (J == 0) {
    if constexpr (K == 0) {
      sym_tensor<2, K> t;
      t[0] = g.logpdf(r) - math::log(y);
      return t;
    } else {
      return g.logpdf_dk<K>(r);
    }
  } else {
    sym_tensor<2, K> t;
    if constexpr (J == 1) t = dx_comb_1<K>(g, r);
    else if constexpr (J == 2) t = dx_comb_2<K>(g, r);
    else if constexpr (J == 3) t = dx_comb_3<K>(g, r);
    else if constexpr (J == 4) t = dx_comb_4<K>(g, r);
    else if constexpr (J == 5) t = dx_comb_5<K>(g, r);
    else t = dx_comb_6<K>(g, r);
    // y^(-J) = 1 / y^J, the product of J factors of 1 / y (J <= 6)
    double iy = 1.0;
    for (int i = 0; i < J; ++i) iy /= y;
    for (int i = 0; i < sym_tensor<2, K>::size; ++i) t[i] *= iy;
    if constexpr (K == 0) t[0] += detail_gaussian2::sgn(J) * detail_gaussian2::fact(J - 1) * iy;
    return t;
  }
}

}  // namespace detail_lognormal1

/**
 * \ingroup distrib_lognormal1
 * \brief Lognormal distribution parameterized by the mean \f$\mu\f$ and the variance \f$v = \sigma^2 > 0\f$ of \f$\log Y\f$.
 *
 * The density of \f$Y\f$ is \f$f(y) = \varphi((\log y - \mu)/\sigma)/(\sigma y)\f$, so the log-density is
 * \f$\ell(\mu, v; y) = \ell_{\mathrm{N}}(\mu, v; \log y) - \log y\f$, with \f$\ell_{\mathrm{N}}\f$ the log-density of
 * gaussian2_distrib. Neither parameter is a moment of \f$Y\f$ (lognormal2_distrib is the parametrization in the
 * mean and variance of \f$Y\f$).
 * \see gaussian2_distrib, lognormal2_distrib, distrib_lognormal1
 */
struct lognormal1_distrib {
  /// Mean of \f$\log Y\f$, \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Variance of \f$\log Y\f$, \f$v \in (0, \infty)\f$.
  double v;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Differentiability order of the log-density in \f$y\f$ on \f$(0, \infty)\f$ (\c infinitely_smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c v.
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"v", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// The gaussian2_distrib of \f$\log Y\f$ at the same parameters: every parameter derivative is its own.
  LAPLACE_HD gaussian2_distrib base() const { return gaussian2_distrib{mu, v}; }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates the log-density \f$\log f(y; \mu, v)\f$ for \f$y > 0\f$.
   * \param y Observation point \f$y \in (0, \infty)\f$.
   * \return \f$\ell_{\mathrm{N}}(\mu, v; \log y) - \log y\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double x = math::log(y);
    return base().logpdf(x) - x;
  }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log f(y; \mu, v)\f$, equal to gaussian2's at \f$x = \log y\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    return base().logpdf_dk<K>(math::log(y));
  }
  /// First-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates \f$\partial_y^J \partial_\theta^K \log f(y; \mu, v)\f$, \f$J + K \le 6\f$, for \f$y > 0\f$.
   *
   * For \f$J \ge 1\f$ the derivative in \f$y\f$ is \f$y^{-J}\sum_k s(J, k)\,\partial_x^k\f$ of gaussian2's
   * \f$\partial_\theta^K\f$ log-density at \f$x = \log y\f$ (see the file comment), and the term \f$-\log y\f$
   * adds \f$(-1)^J (J-1)!\, y^{-J}\f$ when \f$K = 0\f$. For \f$J = 0\f$ it is the parameter derivative (or the value).
   *
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in \f$(\mu, v)\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    return detail_lognormal1::dy_dk_residual<J, K>(y, math::log(y) - mu, v);
  }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates the expected Fisher information \f$I(\mu, v) = \operatorname{diag}(1/v, 1/(2 v^2))\f$.
   * \return gaussian2's expected information (the same parameters, the same law of \f$\log Y\f$).
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const { return base().expected_info(); }
  /// First-order derivatives of the expected information (gaussian2's).
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return base().expected_info_d1(); }
  /// Second-order derivatives of the expected information (gaussian2's).
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return base().expected_info_d2(); }
  /// Third-order derivatives of the expected information (gaussian2's).
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return base().expected_info_d3(); }
  /// Fourth-order derivatives of the expected information (gaussian2's).
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return base().expected_info_d4(); }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates \f$F(y) = \Phi((\log y - \mu)/\sqrt{v})\f$ for \f$y > 0\f$, and 0 for \f$y \le 0\f$.
   * \param y Observation point.
   * \return \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : base().cdf(math::log(y)); }
  /// \f$1 - F(y)\f$ (1 for \f$y \le 0\f$).
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : base().ccdf(math::log(y)); }
  /// \f$\log F(y)\f$ (\f$-\infty\f$ for \f$y \le 0\f$).
  LAPLACE_HD double log_cdf(double y) const { return y <= 0.0 ? -distrib_infinity : base().log_cdf(math::log(y)); }
  /// \f$\log(1 - F(y))\f$ (0 for \f$y \le 0\f$).
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : base().log_ccdf(math::log(y)); }
  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates the quantile \f$F^{-1}(p) = \exp(\mu + \sqrt{v}\,\Phi^{-1}(p))\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile \f$y > 0\f$.
   */
  LAPLACE_HD double quantile(double p) const { return math::exp(base().quantile(p)); }

  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$: gaussian2's at \f$x = \log y\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    return base().log_cdf_dk<K>(math::log(y));
  }
  /**
   * \ingroup distrib_lognormal1
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$: gaussian2's at \f$x = \log y\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    return base().log_ccdf_dk<K>(math::log(y));
  }
  /// First-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /// First-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }

  /// Lower bound of the support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of the support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location: the median \f$\exp(\mu)\f$.
  LAPLACE_HD double center() const { return math::exp(mu); }
  /// Typical spread \f$\exp(\mu)\sqrt{v}\f$ (the first-order standard deviation of \f$Y\f$; a rough quadrature hint).
  LAPLACE_HD double scale() const { return math::exp(mu) * Kokkos::sqrt(v); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
