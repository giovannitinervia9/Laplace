// gamma1_distrib (laplace::distrib): the gamma distribution in its mean mu > 0
// and dispersion phi > 0 (shape s = 1/phi, scale mu phi, variance phi mu^2),
//
//   l(mu, phi; y) = s G(mu) + H(phi) - log y - log sqrt(2 pi),
//   G(mu)  = -D(y/mu),  D(z) = z - 1 - log z >= 0,
//   H(phi) = (1/2) log s - delta(s),  s = 1/phi,
//
// with delta Stirling's error (laplace::math::stirling_error). This form has
// no cancellation at large s, where log Gamma(s) and s log s nearly cancel.
//
// Derivatives, a in mu and b in phi (b the position in the packed tensor),
// with s^(b)(phi) = (-1)^b b! / phi^(b+1):
//   a >= 1:        s^(b) G^(a),  G^(a)(mu) = (-1)^a (a-1)! (mu - a y) / mu^(a+1)
//   a = 0, b >= 1: s^(b) G + H^(b),
//                  H^(b) = (-1)^b (b-1)! / (2 phi^b) - d^b delta(1/phi) / dphi^b,
// the last by Faa di Bruno's formula over delta^(k)(s), k <= b. Measured with
// mpmath (2026-11-12): within 8 ulp of H^(b) for phi in [1e-4, 1e3], b <= 6,
// although the sum of the Faa di Bruno terms alone cancels by up to 1e10,
// because delta is a small correction to the logarithm there.
//
// Expected information: I_mumu = s / mu^2, I_muphi = 0,
// I_phiphi = s^2/2 + s^4 delta''(s) (= s^4 (psi'(s) - 1/s)). Its derivatives:
// I_mumu by the product of the powers; I_phiphi = 1/(2 phi^2) + f(s(phi)),
// f(s) = s^4 delta''(s), by Faa di Bruno over f^(k), and f^(k) by Leibniz's rule
// over delta^(2..6).
//
// F(y) = P(s, s y / mu), the regularized lower incomplete gamma function. The
// derivatives of log F and log(1 - F) in (mu, phi) compose the mixed partials of
// log P(a, a c) and log Q(a, a c) in (a, c), a = 1/phi, c = y/mu
// (laplace::math::log_gamma_{p,q}_ratio_mixed_dK), by two univariate Faa di Bruno
// sums (detail_gamma_cdf::gamma1_dk): along phi the ratio x/a = c is fixed, and
// the partials in (a, x) would cancel there.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/gamma_cdf.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_gamma1 {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// D(z) = z - 1 - log z, z = y/mu: the series in w = (y - mu)/mu below |w| = 1/4,
// w - log(1 + w) above; below z = 1/2 the logarithm directly (w rounds to -1 for
// y < mu 2^-53, where log1p(w) would be -inf)
LAPLACE_HD_FORCEINLINE double deviance_unit(double y, double mu) {
  const double w = (y - mu) / mu;
  if (w < -0.5) return w - math::log(y / mu);
  if (Kokkos::fabs(w) < 0.25) {
    double p = w * w, s = 0.0;  // sum_(n >= 2) (-1)^n w^n / n
    for (int n = 2; n <= 40; ++n, p *= -w) {
      const double t = p / n;
      s += t;
      if (Kokkos::fabs(t) <= 0x1p-54 * s) break;
    }
    return s;
  }
  return w - Kokkos::log1p(w);
}

// delta^(k)(s), k = 1..N, into d[1..N]
template <int N>
LAPLACE_HD_FORCEINLINE void stirling_upto(double s, double* d) {
  namespace ds = laplace::math::detail_stirling;
  d[1] = ds::value<1>(s);
  if constexpr (N >= 2) d[2] = ds::value<2>(s);
  if constexpr (N >= 3) d[3] = ds::value<3>(s);
  if constexpr (N >= 4) d[4] = ds::value<4>(s);
  if constexpr (N >= 5) d[5] = ds::value<5>(s);
  if constexpr (N >= 6) d[6] = ds::value<6>(s);
}

// g[j] = d^j s / dphi^j = (-1)^j j! s^(j+1), j = 1..N
template <int N>
LAPLACE_HD_FORCEINLINE void reciprocal_derivs(double s, double* g) {
  double p = -s * s;  // (-1)^j s^(j+1)
  for (int j = 1; j <= N; ++j, p *= -s) g[j] = fact(j) * p;
}

}  // namespace detail_gamma1

/**
 * \ingroup distrib_gamma1
 * \brief Univariate Gamma distribution parameterized by mean \f$\mu > 0\f$ and dispersion \f$\phi > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \phi; y) = s G(\mu) + H(\phi) - \log y - \frac{1}{2}\log(2\pi)\f$
 * with shape \f$s = 1/\phi\f$ and Stirling's error, its parameter derivatives to order 6,
 * expected Fisher information, and cumulative distribution derivatives composed from fixed-ratio incomplete gamma partials.
 * \see distrib_gamma1, distrib_core, stirling, incomplete_gamma
 */
struct gamma1_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Dispersion parameter \f$\phi \in (0, \infty)\f$ (shape \f$s = 1/\phi\f$, scale \f$\mu\phi\f$).
  double phi;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Descriptors of \c mu and \c phi.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"phi", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the log-density \f$\log f(y; \mu, \phi)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-density value, or \f$-\infty\f$ if \f$y \le 0\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    const double s = 1.0 / phi;
    return -s * detail_gamma1::deviance_unit(y, mu) + 0.5 * math::log(s) - math::stirling_error(s) -
           math::log(y) - 0.91893853320467274178;
  }

  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \phi)\f$ in parameters \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    using detail_gamma1::fact;
    const double s = 1.0 / phi;
    double g[K + 1], dl[K + 1];
    detail_gamma1::reciprocal_derivs<K>(s, g);
    sym_tensor<2, K> t;
    // a >= 1: s^(b) G^(a); s^(0) = s
    double iam = 1.0 / mu;  // mu^-(a+1), from a = 1
    iam *= iam;
    for (int a = 1; a <= K; ++a, iam /= mu) {
      const int b = K - a;
      const double Ga = (a % 2 == 0 ? 1.0 : -1.0) * fact(a - 1) * (mu - a * y) * iam;
      t[b] = (b == 0 ? s : g[b]) * Ga;
    }
    // a = 0: s^(K) G + H^(K)
    detail_gamma1::stirling_upto<K>(s, dl);
    double f[K + 1];
    f[0] = 0.0;
    for (int k = 1; k <= K; ++k) f[k] = -dl[k];
    const double G = -detail_gamma1::deviance_unit(y, mu);
    double ipk = 1.0;
    for (int k = 0; k < K; ++k) ipk *= s;
    const double half_log = (K % 2 == 0 ? 0.5 : -0.5) * fact(K - 1) * ipk;
    t[K] = g[K] * G + half_log + math::detail_faa::compose<K>(f, g);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \phi)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \phi)\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\phi\phi}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double s = 1.0 / phi;
    sym_tensor<2, 2> t;
    t[0] = s / (mu * mu);
    t[2] = s * s * (0.5 + s * s * math::stirling_error_d2(s));
    return t;
  }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    using detail_gamma1::fact;
    const double s = 1.0 / phi;
    double g[K + 1];
    detail_gamma1::reciprocal_derivs<K>(s, g);
    expected_info_tensor<2, K> t;
    // I_mumu = s / mu^2: a derivatives in mu, b in phi
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      double im = 1.0 / (mu * mu);
      for (int i = 0; i < a; ++i) im /= mu;
      t.pair[0][b] = (a % 2 == 0 ? 1.0 : -1.0) * fact(a + 1) * im * (b == 0 ? s : g[b]);
    }
    // I_phiphi = 1/(2 phi^2) + f(s), f = s^4 delta''(s)
    double dl[7] = {0, 0, 0, 0, 0, 0, 0};
    namespace ds = laplace::math::detail_stirling;
    dl[2] = ds::value<2>(s);
    dl[3] = ds::value<3>(s);
    if constexpr (K >= 2) dl[4] = ds::value<4>(s);
    if constexpr (K >= 3) dl[5] = ds::value<5>(s);
    if constexpr (K >= 4) dl[6] = ds::value<6>(s);
    double f[K + 1];
    f[0] = 0.0;
    for (int k = 1; k <= K; ++k) {  // f^(k) = sum_i C(k, i) 4!/(4-i)! s^(4-i) delta^(2+k-i)
      double acc = 0.0, binom = 1.0;
      for (int i = 0; i <= k && i <= 4; ++i) {
        double sp = 1.0;
        for (int j = 0; j < 4 - i; ++j) sp *= s;
        acc += binom * (fact(4) / fact(4 - i)) * sp * dl[2 + k - i];
        binom = binom * (k - i) / (i + 1);
      }
      f[k] = acc;
    }
    double ip = 1.0;  // phi^-(K+2)
    for (int k = 0; k < K + 2; ++k) ip *= s;
    t.pair[2][K] = (K % 2 == 0 ? 0.5 : -0.5) * fact(K + 1) * ip + math::detail_faa::compose<K>(f, g);
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \phi)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the cumulative distribution function \f$F(y) = P(1/\phi, y/(\mu\phi))\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : math::gamma_p(1.0 / phi, y / (mu * phi)); }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = Q(1/\phi, y/(\mu\phi))\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : math::gamma_q(1.0 / phi, y / (mu * phi)); }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y <= 0.0 ? -distrib_infinity : math::log_gamma_p(1.0 / phi, y / (mu * phi));
  }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : math::log_gamma_q(1.0 / phi, y / (mu * phi)); }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const { return detail_gamma_cdf::gamma1_dk<K>(true, y, mu, phi); }
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const { return detail_gamma_cdf::gamma1_dk<K>(false, y, mu, phi); }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */
  /**
   * \ingroup distrib_gamma1
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu\phi\, P^{-1}(1/\phi, p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return mu * phi * math::gamma_p_inv(1.0 / phi, p); }

  /// Lower bound of support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\mu\sqrt{\phi}\f$).
  LAPLACE_HD double scale() const { return mu * Kokkos::sqrt(phi); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
