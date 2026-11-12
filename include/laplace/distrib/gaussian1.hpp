// gaussian1_distrib (laplace::distrib): the normal distribution in its mean
// mu and standard deviation sigma > 0,
//
//   l(mu, sigma; y) = -log sigma - log sqrt(2 pi) - z^2/2,   z = (y - mu)/sigma.
//
// With a derivatives in mu and b in sigma (a + b = K, b the position in the
// packed tensor):
//   a = 0: (-1)^b ((b-1)! - (b+1)!/2 z^2) / sigma^b
//   a = 1: (-1)^b (b+1)! z / sigma^(b+1)
//   a = 2: (-1)^(b+1) (b+1)! / sigma^(b+2)
//   a > 2: 0
// Expected information diag(1, 2)/sigma^2; its K-th derivative in sigma is
// (-1)^K (K+1)! diag(1, 2)/sigma^(K+2), and it does not depend on mu.
// log F = log Phi(z) and log(1 - F) = log Phi(-z), differentiated in z by
// laplace::math::log_Phi_dK and carried to (mu, sigma) by
// detail_location_scale::mixed_dk; in the tail (log Phi(t) at t <= -4) the parts
// -t^2/2, -log(-t) and -1/t^2 are differentiated in (mu, sigma) in closed form and
// only the remainder goes through mixed_dk (detail_gaussian1::log_phi_tail_dk).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

namespace laplace::distrib {

namespace detail_gaussian1 {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// L[j] = d^j/dz^j log Phi(sign z) at the point t = sign z, j = 1..K: (sign)^j log_Phi_dj(t)
template <int K>
LAPLACE_HD_FORCEINLINE void log_phi_derivs(double t, double sign, double* L) {
  L[1] = sign * math::log_Phi_d1(t);
  if constexpr (K >= 2) L[2] = math::log_Phi_d2(t);
  if constexpr (K >= 3) L[3] = sign * math::log_Phi_d3(t);
  if constexpr (K >= 4) L[4] = math::log_Phi_d4(t);
  if constexpr (K >= 5) L[5] = sign * math::log_Phi_d5(t);
  if constexpr (K >= 6) L[6] = math::log_Phi_d6(t);
}

// The tail route, for x = -sign z >= tail_split (z <= -4 for log F = log Phi(z), z >= 4 for
// log(1 - F) = log Phi(-z)). With t = sign z = -x and w = 1/x^2 = sigma^2/(y - mu)^2,
//   log Phi(t) = -t^2/2 - log x - log sqrt(2 pi) - w + S_2(t),
//   S_2(t) = log(x M_0(x)) + w,   M_0(x) = int_0^inf exp(-x s - s^2/2) ds.
// The first parts are differentiated in (mu, sigma) in closed form:
//   -t^2/2 = -(y - mu)^2 / (2 sigma^2), nothing beyond two derivatives in mu;
//   -log x = -log(sign (mu - y)) + log sigma, no mixed derivatives;
//   -w = -sigma^2 / (y - mu)^2, nothing beyond two derivatives in sigma;
// and only S_2, of size w^2, goes through the z-polynomials of mixed_dk. Through mixed_dk
// whole, the derivatives of log Phi in t (of size (j-1)!/x^j and larger) cancelled in every
// component that the first parts do not reach: 1.6e5 ulp of the component at order 6
// (3 derivatives in mu, 3 in sigma) at x = 27.
// Below x = 4 the parts are larger than their sum and the split loses more than the route
// through log_Phi_dK (measured on a grid of x step 0.25 from 1 to 15).
constexpr double tail_split = 4.0;

// L[j] = d^j/dz^j S_2(sign z) = sign^j eta2_j / x^j, j = 1..K, x >= tail_split.
// d^j/dt^j log(x M_0(x)) = kappa_j - (j-1)!/x^j, with kappa_j the cumulants of the density
// proportional to exp(-x s - s^2/2) on s > 0 (math::normal.hpp, log_Phi_dk) and (j-1)!/x^j
// those of x exp(-x s), its limit for large x; d^j/dt^j w = (j+1)!/x^(j+2). In the scaled
// variable v = x s the moments are mu_j = x^j m_j = rho_1 ... rho_j, rho_j = j / (1 + w rho_(j+1)),
// computed downward from the large-j root after K + 22 + ceil(300 w) steps. Every deviation
// from the exponential limit is formed directly, never as a difference:
//   rho_j = j (1 + e_j),   e_j = -w rho_(j+1) / (1 + w rho_(j+1))   (< 0),
//   e_j = -(j+1) w + f_j,  f_j = (j+1) w (w rho_(j+1) - e_(j+1)) / (1 + w rho_(j+1))   (> 0),
//   mu_n = n! (1 + eps_n),  eps_n = eps_(n-1) + e_n (1 + eps_(n-1)),
//   eps_n = -n (n+3)/2 w + eps2_n,  eps2_n = eps2_(n-1) + f_n + e_n eps_(n-1),
// and the scaled cumulants x^n kappa_n = (n-1)! + eta_n = (n-1)! - (n+1)! w + eta2_n follow
// from kappa_n = mu_n - sum_(i<n) C(n-1, i-1) kappa_i mu_(n-i) with the exponential case and
// the part linear in w subtracted exactly:
//   eta_n  = n! eps_n  - sum_(i<n) C(n-1, i-1) (n-i)! ((i-1)! eps_(n-i) + eta_i (1 + eps_(n-i))),
//   eta2_n = n! eps2_n - sum_(i<n) C(n-1, i-1) (n-i)! ((i-1)! eps2_(n-i) + eta2_i + eta_i eps_(n-i)).
template <int K>
LAPLACE_HD void tail_mills_derivs(double x, double sign, double* L) {
  const double ix = 1.0 / x, w = ix * ix;
  const int N = K + 22 + static_cast<int>(Kokkos::ceil(300.0 * w));
  const double N1 = N + 1.0;
  double rho = 2.0 * N1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * N1 * w));
  double e[K + 2] = {}, f[K + 1] = {};
  for (int j = N; j >= 1; --j) {  // rho holds rho_(j+1) until it is updated
    if (j <= K + 1) e[j] = -w * rho / (1.0 + w * rho);
    if (j <= K) f[j] = w * (j + 1) * (w * rho - e[j + 1]) / (1.0 + w * rho);
    rho = j / (1.0 + w * rho);
  }
  double eps[K + 1], eps2[K + 1], eta[K + 1], eta2[K + 1];
  eps[0] = eps2[0] = 0.0;
  for (int n = 1; n <= K; ++n) {
    eps[n] = eps[n - 1] + e[n] * (1.0 + eps[n - 1]);
    eps2[n] = eps2[n - 1] + f[n] + e[n] * eps[n - 1];
  }
  double sc = 1.0;
  for (int n = 1; n <= K; ++n) {
    double s1 = fact(n) * eps[n], s2 = fact(n) * eps2[n], binom = 1.0;  // binom = C(n - 1, i - 1)
    for (int i = 1; i < n; ++i) {
      const double c = binom * fact(n - i);
      s1 -= c * (fact(i - 1) * eps[n - i] + eta[i] * (1.0 + eps[n - i]));
      s2 -= c * (fact(i - 1) * eps2[n - i] + eta2[i] + eta[i] * eps[n - i]);
      binom = binom * (n - i) / i;
    }
    eta[n] = s1;
    eta2[n] = s2;
    sc *= sign * ix;
    L[n] = s2 * sc;
  }
}

// d^K/dmu^(K-b) dsigma^b of -(y - mu)^2 / (2 sigma^2), z = (y - mu)/sigma: the terms of
// logpdf_dk without -log sigma
template <int K>
LAPLACE_HD_FORCEINLINE sym_tensor<2, K> half_square_dk(double z, double sigma) {
  const double is = 1.0 / sigma;
  sym_tensor<2, K> t;
  double isb = 1.0;  // sigma^-b
  for (int b = 0; b <= K; ++b, isb *= is) {
    const int a = K - b;
    const double sgn = (b % 2 == 0) ? 1.0 : -1.0;
    if (a == 0) t[b] = -sgn * 0.5 * fact(b + 1) * z * z * isb;
    else if (a == 1) t[b] = sgn * fact(b + 1) * z * isb * is;
    else if (a == 2) t[b] = -sgn * fact(b + 1) * isb * is * is;
  }
  return t;
}

// the tail route above, t = sign z, x = -t. With a = K - b derivatives in mu:
//   -log x: (-sign)^K (K-1)! / (x sigma)^K at a = K, (-1)^(K-1) (K-1)! / sigma^K at b = K;
//   -w = -sigma^2 / (z sigma)^2: -(a+1)! z^-(a+2) sigma^-K times 1, 2, 2 for b = 0, 1, 2.
template <int K>
LAPLACE_HD sym_tensor<2, K> log_phi_tail_dk(double z, double sign, double sigma) {
  const double x = -sign * z;
  double L[K + 1];
  tail_mills_derivs<K>(x, sign, L);
  sym_tensor<2, K> t = detail_location_scale::mixed_dk<K>(L, z, sigma);
  const sym_tensor<2, K> q = half_square_dk<K>(z, sigma);
  for (int b = 0; b <= K; ++b) t[b] += q[b];
  double pm = fact(K - 1), ps = fact(K - 1), isk = 1.0;
  for (int k = 0; k < K; ++k) {
    pm *= -sign / (x * sigma);
    ps /= -sigma;
    isk /= sigma;
  }
  t[0] += pm;
  t[K] -= ps;
  const double iz = 1.0 / z;
  for (int b = 0; b <= 2 && b <= K; ++b) {
    double v = -fact(K - b + 1) * isk * (b == 0 ? 1.0 : 2.0) * iz * iz;
    for (int k = 0; k < K - b; ++k) v *= iz;
    t[b] += v;
  }
  return t;
}

}  // namespace detail_gaussian1

/**
 * \ingroup distrib_gaussian1
 * \brief Univariate Gaussian (normal) distribution parameterized by mean \f$\mu\f$ and standard deviation \f$\sigma > 0\f$.
 *
 * Implements the continuous normal density \f$\ell(\mu, \sigma; y) = -\log\sigma - \frac{1}{2}\log(2\pi) - \frac{(y - \mu)^2}{2\sigma^2}\f$,
 * its closed-form parameter derivatives to order 6, diagonal expected Fisher information,
 * and normal cumulative distribution functions with asymptotic tail expansions.
 * \see distrib_gaussian1, distrib_core
 */
struct gaussian1_distrib {
  /// Mean parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Standard deviation parameter \f$\sigma \in (0, \infty)\f$.
  double sigma;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_gaussian1
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth).
   */
  static constexpr int y_regularity = infinitely_smooth;  // regularity of log f in y (continuous only)
  /// Descriptors of \c mu and \c sigma.
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma)\f$.
   *
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density \f$-\log\sigma - \frac{1}{2}\log(2\pi) - \frac{(y - \mu)^2}{2\sigma^2}\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    return -math::log(sigma) - 0.91893853320467274178 - 0.5 * z * z;
  }

  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates partial derivatives of \f$\log f(y; \mu, \sigma)\f$ of order \f$K\f$ in parameters \f$(\mu, \sigma)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    using detail_gaussian1::fact;
    const double z = (y - mu) / sigma, is = 1.0 / sigma;
    sym_tensor<2, K> t;
    double isb = 1.0;  // sigma^-b
    for (int b = 0; b <= K; ++b, isb *= is) {
      const int a = K - b;
      const double sgn = (b % 2 == 0) ? 1.0 : -1.0;
      if (a == 0) t[b] = sgn * (fact(b - 1) - 0.5 * fact(b + 1) * z * z) * isb;
      else if (a == 1) t[b] = sgn * fact(b + 1) * z * isb * is;
      else if (a == 2) t[b] = -sgn * fact(b + 1) * isb * is * is;
    }
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \sigma)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  // d^(J+K) l / dy^J dtheta^K, J derivatives in y and the symmetric tensor of order K in
  // (mu, sigma): l depends on y - mu only, so d/dy = -d/dmu and the component with a
  // derivatives in mu and b in sigma is (-1)^J times the (a + J, b) component of logpdf_dk.
  // J = 0 is logpdf_dk (K >= 1) or the value (K = 0).
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \sigma)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * Evaluates \f$\frac{\partial^{J+K}}{\partial y^J \partial\theta^K} \log f(y; \mu, \sigma)\f$ as a symmetric tensor of order \f$K\f$ in \f$(\mu, \sigma)\f$.
   * Because the log-density depends on \f$y - \mu\f$ only, \f$\frac{\partial}{\partial y} = -\frac{\partial}{\partial\mu}\f$. For \f$J \ge 1\f$, letting
   * \f$a = K - b + J\f$ and \f$b\f$ be the order in \f$\sigma\f$, the component at position \f$b\f$ is
   * \f$\frac{(-1)^{J + b} (b + 1)!}{\sigma^{b + 1}} z\f$ for \f$a = 1\f$, \f$\frac{(-1)^{J + b + 1} (b + 1)!}{\sigma^{b + 2}}\f$ for \f$a = 2\f$, and 0 for \f$a > 2\f$,
   * where \f$z = \frac{y - \mu}{\sigma}\f$. For \f$J = 0\f$, returns parameter derivatives (\f$K \ge 1\f$) or the log-density value (\f$K = 0\f$).
   *
   * Agrees with 5-point finite differences in \f$y\f$ and in parameters within relative tolerance \f$10^{-8}\f$.
   *
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \sigma)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \see distrib_gaussian1, logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      using detail_gaussian1::fact;
      const double z = (y - mu) / sigma, is = 1.0 / sigma;
      const double sj = (J % 2 == 0) ? 1.0 : -1.0;
      double isb = 1.0;  // sigma^-b
      for (int b = 0; b <= K; ++b, isb *= is) {
        const int a = K - b + J;  // a >= 1
        const double sgn = (b % 2 == 0) ? 1.0 : -1.0;
        if (a == 1) t[b] = sj * sgn * fact(b + 1) * z * isb * is;
        else if (a == 2) t[b] = -sj * sgn * fact(b + 1) * isb * is * is;
      }
      return t;
    }
  }
  /** @} */

  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \sigma) = \operatorname{diag}(1/\sigma^2, 2/\sigma^2)\f$.
   *
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\sigma\sigma}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double i2 = 1.0 / (sigma * sigma);
    sym_tensor<2, 2> t;
    t[0] = i2;
    t[2] = 2.0 * i2;
    return t;
  }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    double d = detail_gaussian1::fact(K + 1) / (sigma * sigma);
    for (int k = 0; k < K; ++k) d /= -sigma;
    expected_info_tensor<2, K> t;
    t.pair[0][K] = d;        // (mu, mu), all K derivatives in sigma
    t.pair[2][K] = 2.0 * d;  // (sigma, sigma)
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \sigma)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \sigma)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the cumulative distribution function \f$F(y) = \Phi((y - \mu)/\sigma)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return math::Phi((y - mu) / sigma); }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \Phi((\mu - y)/\sigma)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return math::Phi((mu - y) / sigma); }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log\Phi((y - \mu)/\sigma)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const { return math::log_Phi((y - mu) / sigma); }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log\Phi((\mu - y)/\sigma)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const { return math::log_Phi((mu - y) / sigma); }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu + \sigma \Phi^{-1}(p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return mu + sigma * math::Phi_inv(p); }

  // d^K log F / dtheta^K and d^K log(1 - F) / dtheta^K: log F = log Phi(z), log(1 - F) = log Phi(-z),
  // carried to (mu, sigma) by detail_location_scale; at |z| >= tail_split on the side of the tail by
  // detail_gaussian1::log_phi_tail_dk
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z <= -detail_gaussian1::tail_split) return detail_gaussian1::log_phi_tail_dk<K>(z, 1.0, sigma);
    double L[K + 1];
    detail_gaussian1::log_phi_derivs<K>(z, 1.0, L);
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \ingroup distrib_gaussian1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z >= detail_gaussian1::tail_split) return detail_gaussian1::log_phi_tail_dk<K>(z, -1.0, sigma);
    double L[K + 1];
    detail_gaussian1::log_phi_derivs<K>(-z, -1.0, L);
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
