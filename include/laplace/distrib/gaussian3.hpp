// gaussian3_distrib (laplace::distrib): the normal distribution in its mean
// mu (real) and its precision tau = 1/sigma^2 > 0,
//
//   l(mu, tau; y) = -log(2 pi)/2 + log(tau)/2 - tau r^2/2,   r = y - mu.
//
// With a derivatives in mu and b in tau (a + b = K, b the position in the packed
// tensor; formulas checked with SymPy by tools/reference/gaussian23_forms.py):
//   a = 0: (1/2)(-1)^(b-1) (b-1)! / tau^b      for b >= 2, and 1/(2 tau) - r^2/2 for b = 1
//   a = 1: tau r for b = 0, r for b = 1, 0 for b >= 2
//   a = 2: -tau for b = 0, -1 for b = 1, 0 for b >= 2
//   a > 2: 0
// Expected information diag(tau, 1/(2 tau^2)); its K-th derivative in tau is
// diag(1 for K = 1 and 0 above, (-1)^K (K+1)!/(2 tau^(K+2))), and it does not depend on mu.
//
// log F = log Phi(z) and log(1 - F) = log Phi(-z) with z = r sqrt(tau) = r/sigma. Away
// from the tail, the z-derivatives L_j of log Phi go through
// mixed_power_dk with w = tau and k = -1/2. In the tail
// (log Phi(t) at t <= -4 in the sense of detail_gaussian1::tail_split) the closed parts
//   -t^2/2 = -tau r^2/2,   -log x = -log|r| - (1/2) log tau,   -w = -1/(tau r^2)
// (x = |z| and w = 1/x^2 = 1/(tau r^2)) are differentiated in (mu, tau) here, and only
// the remainder S_2 goes through mixed_power_dk; see detail_gaussian3::log_phi_tail_dk.
//
// The z-derivatives of log Phi and the tail series come from detail_gaussian1 (the
// pilot), which is shared with gaussian1_distrib and gaussian2_distrib: a function of z only.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

namespace laplace::distrib {

namespace detail_gaussian3 {

using detail_gaussian1::fact;

// x^n for a non-negative integer n
LAPLACE_HD_FORCEINLINE double ipow(double x, int n) {
  double p = 1.0;
  for (int i = 0; i < n; ++i) p *= x;
  return p;
}

// (-1)^n
LAPLACE_HD_FORCEINLINE double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// d^a/dmu^a of r^2 (r = y - mu): r^2, -2 r, 2 for a = 0, 1, 2; 0 above
LAPLACE_HD_FORCEINLINE double square_mu(int a, double r) {
  if (a == 0) return r * r;
  if (a == 1) return -2.0 * r;
  if (a == 2) return 2.0;
  return 0.0;
}

// d^b/dtau^b of tau: tau, 1 for b = 0, 1; 0 above
LAPLACE_HD_FORCEINLINE double tau_tau(int b, double tau) {
  if (b == 0) return tau;
  if (b == 1) return 1.0;
  return 0.0;
}

// The (a, b) component of the log-density without its constant, a derivatives in mu and b
// in tau, a + b >= 1, r = y - mu. Its parts: (1/2) log tau and -tau r^2/2.
LAPLACE_HD_FORCEINLINE double logpdf_comp(int a, int b, double r, double tau) {
  double out = -0.5 * square_mu(a, r) * tau_tau(b, tau);
  if (a == 0 && b >= 1) out += 0.5 * sgn(b - 1) * fact(b - 1) * ipow(1.0 / tau, b);
  return out;
}

// The closed parts of the tail route (see the file comment), with a derivatives in mu and b
// in tau, a + b >= 1; r = y - mu = z sigma.
//   A: -tau r^2/2 = -z^2/2, the half-square part;
//   B: -log|r| - (1/2) log tau (no mixed terms);
//   C: -w = -1/(tau r^2), whose mu-derivatives are (a+1)!/r^(a+2) and whose tau-derivatives are
//      (-1)^b b!/tau^(b+1).
LAPLACE_HD_FORCEINLINE double tail_part_a(int a, int b, double r, double tau) {
  return -0.5 * square_mu(a, r) * tau_tau(b, tau);
}
LAPLACE_HD_FORCEINLINE double tail_part_b(int a, int b, double r, double tau) {
  if (a == 0 && b >= 1) return -0.5 * sgn(b - 1) * fact(b - 1) * ipow(1.0 / tau, b);
  if (a >= 1 && b == 0) return fact(a - 1) / ipow(r, a);
  return 0.0;
}
LAPLACE_HD_FORCEINLINE double tail_part_c(int a, int b, double r, double tau) {
  return -fact(a + 1) / ipow(r, a + 2) * sgn(b) * fact(b) * ipow(1.0 / tau, b + 1);
}


// The tail route, for x = -sign z >= tail_split (z <= -4 for log F = log Phi(z), z >= 4 for
// log(1 - F) = log Phi(-z)). With t = sign z = -x and w = 1/x^2 = 1/(tau r^2) (r = y - mu):
//   log Phi(t) = -t^2/2 - log x - log sqrt(2 pi) - w + S_2(t),
//   S_2(t) = log(x M_0(x)) + w,   M_0(x) = int_0^inf exp(-x s - s^2/2) ds.
// The parts -t^2/2, -log x and -w are differentiated in (mu, tau) by tail_part_a, _b and _c, and
// only S_2 (of size w^2) goes through mixed_power_dk with w = tau and k = -1/2. The z-derivatives
// of S_2 are those of detail_gaussian1::tail_mills_derivs.
template <int K>
LAPLACE_HD sym_tensor<2, K> log_phi_tail_dk(double z, double sign, double sigma, double tau) {
  const double x = -sign * z, r = z * sigma;
  double L[K + 1];
  detail_gaussian1::tail_mills_derivs<K>(x, sign, L);
  sym_tensor<2, K> t = detail_location_scale::mixed_power_dk<K>(L, z, sigma, tau, -0.5);
  for (int b = 0; b <= K; ++b) {
    const int a = K - b;
    t[b] += tail_part_a(a, b, r, tau) + tail_part_b(a, b, r, tau) + tail_part_c(a, b, r, tau);
  }
  return t;
}

}  // namespace detail_gaussian3

/**
 * \ingroup distrib_gaussian3
 * \brief Univariate Gaussian (normal) distribution parameterized by mean \f$\mu\f$ and precision \f$\tau = 1/\sigma^2 > 0\f$.
 *
 * Implements the continuous normal density \f$\ell(\mu, \tau; y) = -\frac{1}{2}\log(2\pi) + \frac{1}{2}\log\tau - \frac{\tau (y - \mu)^2}{2}\f$,
 * its closed-form parameter derivatives to order 6, diagonal expected Fisher information, and the normal cumulative
 * distribution functions with asymptotic tail expansions. It is the precision parametrization of gaussian1_distrib
 * (\f$\sigma\f$ there, \f$\tau = 1/\sigma^2\f$ here).
 * \see distrib_gaussian3, distrib_core
 */
struct gaussian3_distrib {
  /// Mean parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Precision parameter \f$\tau \in (0, \infty)\f$.
  double tau;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_gaussian3
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth).
   */
  static constexpr int y_regularity = infinitely_smooth;  // regularity of log f in y (continuous only)
  /// Descriptors of \c mu and \c tau.
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"tau", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the log-density \f$\log f(y; \mu, \tau)\f$.
   *
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density \f$\frac{1}{2}\log\tau - \frac{1}{2}\log(2\pi) - \frac{\tau (y - \mu)^2}{2}\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double r = y - mu;
    return 0.5 * math::log(tau) - 0.91893853320467274178 - 0.5 * tau * r * r;
  }

  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates partial derivatives of \f$\log f(y; \mu, \tau)\f$ of order \f$K\f$ in parameters \f$(\mu, \tau)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    const double r = y - mu;
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = detail_gaussian3::logpdf_comp(K - b, b, r, tau);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \tau)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  // d^(J+K) l / dy^J dtheta^K, J derivatives in y and the symmetric tensor of order K in (mu, tau):
  // l depends on y - mu only, so d/dy = -d/dmu and the component with a derivatives in mu and b in tau
  // is (-1)^J times the (a + J, b) component of the log-density.
  // J = 0 is logpdf_dk (K >= 1) or the value (K = 0).
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \tau)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * Evaluates \f$\frac{\partial^{J+K}}{\partial y^J \partial\theta^K} \log f(y; \mu, \tau)\f$ as a symmetric tensor of order \f$K\f$ in \f$(\mu, \tau)\f$.
   * Because the log-density depends on \f$y - \mu\f$ only, \f$\frac{\partial}{\partial y} = -\frac{\partial}{\partial\mu}\f$.
   * For \f$J \ge 1\f$ the component at position \f$b\f$, with \f$a = K - b + J\f$ derivatives in \f$\mu\f$, is
   * \f$(-1)^J\f$ times the (a, b) component of the log-density, which is zero for \f$a > 2\f$.
   * For \f$J = 0\f$, returns parameter derivatives (\f$K \ge 1\f$) or the log-density value (\f$K = 0\f$).
   *
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \tau)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \see distrib_gaussian3, logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      const double r = y - mu;
      for (int b = 0; b <= K; ++b)
        t[b] = detail_gaussian3::sgn(J) * detail_gaussian3::logpdf_comp(K - b + J, b, r, tau);
      return t;
    }
  }
  /** @} */

  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \tau) = \operatorname{diag}(\tau, 1/(2 \tau^2))\f$.
   *
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\tau\tau}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double it = 1.0 / tau;
    sym_tensor<2, 2> t;
    t[0] = tau;
    t[2] = 0.5 * it * it;
    return t;
  }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \tau)\f$.
   *
   * The information does not depend on \f$\mu\f$. The (mu, mu) entry is \f$\tau\f$, whose first derivative is 1 and
   * whose higher derivatives vanish; the (tau, tau) entry is \f$1/(2 \tau^2)\f$ with K-th derivative
   * \f$(-1)^K (K+1)!/(2 \tau^{K+2})\f$. Both are stored at the position \f$b = K\f$ of the symmetric tensor.
   *
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    const double it = 1.0 / tau;
    const double s = detail_gaussian3::sgn(K) * detail_gaussian3::ipow(it, K);  // (-1)^K / tau^K
    expected_info_tensor<2, K> t;
    t.pair[0][K] = (K == 1) ? 1.0 : 0.0;
    t.pair[2][K] = 0.5 * detail_gaussian3::fact(K + 1) * s * it * it;
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \tau)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \tau)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \tau)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \tau)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \tau)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the cumulative distribution function \f$F(y) = \Phi((y - \mu)\sqrt{\tau})\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return math::Phi((y - mu) * Kokkos::sqrt(tau)); }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \Phi((\mu - y)\sqrt{\tau})\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return math::Phi((mu - y) * Kokkos::sqrt(tau)); }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log\Phi((y - \mu)\sqrt{\tau})\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const { return math::log_Phi((y - mu) * Kokkos::sqrt(tau)); }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log\Phi((\mu - y)\sqrt{\tau})\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const { return math::log_Phi((mu - y) * Kokkos::sqrt(tau)); }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu + \Phi^{-1}(p)/\sqrt{\tau}\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return mu + math::Phi_inv(p) / Kokkos::sqrt(tau); }

  // d^K log F / dtheta^K and d^K log(1 - F) / dtheta^K: log F = log Phi(z), log(1 - F) = log Phi(-z),
  // z = (y - mu) sqrt(tau) = (y - mu)/sigma, carried to (mu, tau) by mixed_power_dk (w = tau, k = -1/2);
  // at |z| >= tail_split on the side of the tail by detail_gaussian3::log_phi_tail_dk.
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \tau)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double st = Kokkos::sqrt(tau), sigma = 1.0 / st, z = (y - mu) * st;
    if (z <= -detail_gaussian1::tail_split) return detail_gaussian3::log_phi_tail_dk<K>(z, 1.0, sigma, tau);
    double L[K + 1];
    detail_gaussian1::log_phi_derivs<K>(z, 1.0, L);
    return detail_location_scale::mixed_power_dk<K>(L, z, sigma, tau, -0.5);
  }
  /**
   * \ingroup distrib_gaussian3
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \tau)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double st = Kokkos::sqrt(tau), sigma = 1.0 / st, z = (y - mu) * st;
    if (z >= detail_gaussian1::tail_split) return detail_gaussian3::log_phi_tail_dk<K>(z, -1.0, sigma, tau);
    double L[K + 1];
    detail_gaussian1::log_phi_derivs<K>(-z, -1.0, L);
    return detail_location_scale::mixed_power_dk<K>(L, z, sigma, tau, -0.5);
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \tau)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$1/\sqrt{\tau}\f$).
  LAPLACE_HD double scale() const { return 1.0 / Kokkos::sqrt(tau); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
