// chisq_distrib (laplace::distrib): the chi-square distribution with mu > 0 degrees of freedom,
// mu treated as a continuous parameter. It is the gamma law with shape a = mu/2 and scale 2,
// with mean mu and variance 2 mu,
//
//   l(mu; y) = (a - 1) log y - y/2 - log Gamma(a) - a log 2,   a = mu/2,   y > 0.
//
// Log-density. The value is written in the saddle-point form of gamma1_distrib with
// shape a and mean mu (u = y/mu, D(u) = u - 1 - log u, delta Stirling's error):
//
//   l = -a D(y/mu) + (1/2) log a - delta(a) - log y - (1/2) log(2 pi),
//
// which has no cancellation at large a (the closed form with log Gamma(a) cancels to
// about 1e-12 absolute at mu = 1e4, see the measurements in the test). The deviance D comes
// from detail_gamma1::deviance_unit and the error delta from laplace::math::stirling_error.
//
// Parameter derivatives. Since a = mu/2, d/dmu = (1/2) d/da, and with psi^(n) the polygamma
// functions (laplace::math::polygamma, digamma and trigamma of special.hpp):
//   K = 1:  d l/dmu = (1/2) (log(y/2) - psi(a))
//   K >= 2: d^K l/dmu^K = -2^(-K) psi^(K-1)(a).
// Expected information I = psi'(a)/4 and its K-th derivative 2^(-(K+2)) psi^(K+1)(a).
//
// The cdf is F(y) = P(a, y/2) and 1 - F = Q(a, y/2): the argument x = y/2 does not depend on
// mu and a is linear in mu, so d^K log P/dmu^K = 2^(-K) d^K log P/da^K, the pure shape
// derivative of the mixed partials of log P(a, x) (laplace::math::log_gamma_p_mixed_dK, the
// j = 0 component, through detail_gamma_cdf::mixed_order). The composition with a = mu/2 is
// a single power of 1/2: nothing cancels.
//
// The observation enters the log-density only through (a - 1) log y - y/2, so the mixed
// derivatives in y are short closed forms (logpdf_dy_dk).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/gamma_cdf.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_chisq {

LAPLACE_HD_FORCEINLINE double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^n as a double
LAPLACE_HD_FORCEINLINE double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// 2^-K
LAPLACE_HD_FORCEINLINE double half_pow(int K) {
  double s = 1.0;
  for (int k = 0; k < K; ++k) s *= 0.5;
  return s;
}

}  // namespace detail_chisq

/**
 * \ingroup distrib_chisq
 * \brief Univariate chi-square distribution with \f$\mu > 0\f$ degrees of freedom (mean \f$\mu\f$, variance \f$2\mu\f$).
 *
 * Implements the continuous density in the saddle-point form of gamma1_distrib with shape \f$a = \mu/2\f$, its
 * parameter derivatives to order 6 (polygamma closed forms), the expected information \f$\psi'(a)/4\f$ with its
 * derivatives to order 4, and the regularized incomplete gamma cumulative distribution functions in \f$(a, y/2)\f$.
 * \see distrib_chisq, distrib_core, gamma1_distrib, incomplete_gamma
 */
struct chisq_distrib {
  /// Degrees of freedom \f$\mu \in (0, \infty)\f$ (also the mean).
  double mu;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (1).
  static constexpr int n_params = 1;
  /// Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_chisq
   * \brief Evaluates the log-density \f$\log f(y; \mu)\f$ in the saddle-point form.
   * \param y Observation point \f$y\f$.
   * \return Log-density value, or \f$-\infty\f$ if \f$y \le 0\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    const double a = 0.5 * mu;
    return -a * detail_gamma1::deviance_unit(y, mu) + 0.5 * math::log(a) - math::stirling_error(a) -
           math::log(y) - 0.91893853320467274178;
  }

  /**
   * \ingroup distrib_chisq
   * \brief \f$K\f$-th derivative of \f$\log f\f$ in \f$\mu\f$, \f$\frac{1}{2}(\log(y/2) - \psi(a))\f$ for \f$K = 1\f$ and
   * \f$-2^{-K}\psi^{(K-1)}(a)\f$ for \f$K \ge 2\f$, with \f$a = \mu/2\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpdf_dk(double y) const {
    const double a = 0.5 * mu;
    sym_tensor<1, K> t;
    if constexpr (K == 1) t[0] = 0.5 * (math::log(0.5 * y) - math::digamma(a));
    else t[0] = -detail_chisq::half_pow(K) * math::polygamma(K - 1, a);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log f(y; \mu)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_chisq
   * \brief Mixed partial derivatives \f$\partial_y^J \partial_\mu^K \log f(y; \mu)\f$ with \f$J + K \le 6\f$.
   *
   * With \f$a = \mu/2\f$ and \f$c_J = (-1)^{J-1}(J-1)!/y^J\f$ (the \f$J\f$-th derivative of \f$\log y\f$), the
   * log-density has \f$\partial_y^J \log f = (a - 1) c_J\f$ for \f$J \ge 2\f$ and \f$\partial_y \log f = (a - 1)/y - 1/2\f$.
   * Since \f$\partial_\mu = \frac{1}{2}\partial_a\f$: the \f$K = 0\f$ component is \f$(a - 1) c_J\f$ (less \f$1/2\f$ for
   * \f$J = 1\f$), the \f$K = 1\f$ component \f$c_J/2\f$, and \f$0\f$ for \f$K \ge 2\f$.
   * For \f$J = 0\f$, returns the parameter derivatives (\f$K \ge 1\f$) or the value (\f$K = 0\f$).
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in \f$\mu\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<1, K> logpdf_dy_dk(double y) const {
    sym_tensor<1, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
    } else {
      const double a = 0.5 * mu;
      double iy = 1.0;  // y^-J
      for (int j = 0; j < J; ++j) iy /= y;
      const double c = detail_chisq::sgn(J - 1) * detail_chisq::fact(J - 1) * iy;
      if constexpr (K == 0) t[0] = (a - 1.0) * c - (J == 1 ? 0.5 : 0.0);
      else if constexpr (K == 1) t[0] = 0.5 * c;
      else t[0] = 0.0;
    }
    return t;
  }

  /**
   * \ingroup distrib_chisq
   * \brief Expected Fisher information \f$I(\mu) = \psi'(\mu/2)/4\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = 0.25 * math::trigamma(0.5 * mu);
    return t;
  }
  /**
   * \ingroup distrib_chisq
   * \brief \f$K\f$-th derivative of the expected information, \f$2^{-(K+2)}\psi^{(K+1)}(\mu/2)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    expected_info_tensor<1, K> t;
    t.pair[0][0] = detail_chisq::half_pow(K + 2) * math::polygamma(K + 1, 0.5 * mu);
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} I(\mu)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_chisq
   * \brief Cumulative distribution function \f$F(y) = P(\mu/2, y/2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : math::gamma_p(0.5 * mu, 0.5 * y); }
  /**
   * \ingroup distrib_chisq
   * \brief Complementary cdf \f$1 - F(y) = Q(\mu/2, y/2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : math::gamma_q(0.5 * mu, 0.5 * y); }
  /**
   * \ingroup distrib_chisq
   * \brief Logarithm of the cdf, \f$\log P(\mu/2, y/2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability (\f$-\infty\f$ for \f$y \le 0\f$).
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y <= 0.0 ? -distrib_infinity : math::log_gamma_p(0.5 * mu, 0.5 * y);
  }
  /**
   * \ingroup distrib_chisq
   * \brief Logarithm of the complementary cdf, \f$\log Q(\mu/2, y/2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability (0 for \f$y \le 0\f$).
   */
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : math::log_gamma_q(0.5 * mu, 0.5 * y); }

  /**
   * \ingroup distrib_chisq
   * \brief \f$K\f$-th derivative in \f$\mu\f$ of \f$\log F(y)\f$: \f$2^{-K}\,\partial_a^K \log P(a, y/2)\f$ with \f$a = \mu/2\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$ (NaN for \f$y \le 0\f$, where \f$\log F = -\infty\f$).
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_cdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (!(y > 0.0)) {
      t[0] = math::detail::quiet_nan();
      return t;
    }
    double v[K + 1];
    detail_gamma_cdf::mixed_order<K>(true, 0.5 * mu, 0.5 * y, v);
    t[0] = detail_chisq::half_pow(K) * v[0];
    return t;
  }
  /**
   * \ingroup distrib_chisq
   * \brief \f$K\f$-th derivative in \f$\mu\f$ of \f$\log(1 - F(y))\f$: \f$2^{-K}\,\partial_a^K \log Q(a, y/2)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$ (zero for \f$y \le 0\f$).
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_ccdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (!(y > 0.0)) return t;
    double v[K + 1];
    detail_gamma_cdf::mixed_order<K>(false, 0.5 * mu, 0.5 * y, v);
    t[0] = detail_chisq::half_pow(K) * v[0];
    return t;
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_chisq
   * \brief Quantile function \f$F^{-1}(p) = 2\,P^{-1}(\mu/2, p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return 2.0 * math::gamma_p_inv(0.5 * mu, p); }

  /// Lower bound of support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sqrt{2\mu}\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(2.0 * mu); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
