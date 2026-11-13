// laplace1_distrib (laplace::distrib): the Laplace (double exponential)
// distribution in its location mu and scale sigma > 0,
//
//   l(mu, sigma; y) = -log(2 sigma) - |z|,   z = (y - mu)/sigma.
//
// l is C^0 in mu, with a kink at mu = y (and in y at y = mu); everywhere else
// it is C^inf. log F and log(1 - F) are C^1 in mu at mu = y: their second
// derivative in mu jumps by 2/sigma^2 there.
//
// With a derivatives in mu and b in sigma (a + b = K, b the position in the
// packed tensor), away from the kink:
//   a = 0: (-1)^b ((b-1)! - b! |z|) / sigma^b
//   a = 1: (-1)^b b! sign(z) / sigma^(b+1)
//   a > 1: 0
// At the kink sign(0) = 0, the mean of the two one-sided derivatives. The
// derivatives of order a > 1 in mu carry a point mass at the kink that no
// pointwise value shows: E[d^2 l / dmu^2] = 0, while the expected information
// (the variance of the score) is diag(1, 1)/sigma^2; its K-th derivative in
// sigma is (-1)^K (K+1)! diag(1, 1)/sigma^(K+2).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_laplace1 {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// L[j] = d^j/dz^j log F_0(z), j = 1..K, for the standard Laplace cdf F_0:
// z < 0: log F_0 = z - log 2, so L_1 = 1 and the others 0; z >= 0:
// log F_0 = log(1 - e^-z/2), L_j = P_j(q) with q = 1/(2 e^z - 1), P_1 = q,
// P_(j+1)(q) = -q (1 + q) P_j'(q) (every coefficient of P_j has the sign
// (-1)^(j-1), so nothing cancels). At z = 0 the right-hand derivatives;
// log F_0 is C^1 there (L_2 jumps from 0 to -2).
template <int K>
LAPLACE_HD void log_cdf0_derivs(double z, double* L) {
  if (z < 0.0) {
    L[1] = 1.0;
    for (int j = 2; j <= K; ++j) L[j] = 0.0;
    return;
  }
  const double q = 1.0 / (2.0 * math::exp(z) - 1.0);
  double p[K + 2] = {};  // coefficients of P_j in q
  p[1] = 1.0;
  for (int j = 1; j <= K; ++j) {
    double v = 0.0, qm = 1.0;
    for (int m = 0; m <= j; ++m, qm *= q) v += p[m] * qm;
    L[j] = v;
    if (j == K) break;
    double d[K + 2] = {};  // P_j'(q)
    for (int m = 1; m <= j; ++m) d[m - 1] = m * p[m];
    for (int m = 0; m <= j + 1; ++m) p[m] = 0.0;
    for (int m = 0; m < j; ++m) {  // -q (1 + q) d
      p[m + 1] -= d[m];
      p[m + 2] -= d[m];
    }
  }
}

// from |z| = series_split on, the side of log F or log(1 - F) that is log(1 - e^(-|z|)/2)
// is differentiated term by term in its series (see log1m_half_exp_dk); nearer 0 the
// z-polynomials of log_cdf0_derivs (measured against mpmath on |z| = 0.05 .. 30: in ulp of the
// component the series is the more accurate from |z| = 0.2 on, the z-polynomials below)
constexpr double series_split = 0.25;

// d^K/dmu^(K-b) dsigma^b of log(1 - e^(-|z|)/2), |z| = s (mu - y)/sigma > 0, s = 1 for
// log(1 - F) at z < 0 and s = -1 for log F at z > 0. With p = e^(-|z|)/2,
//   log(1 - e^(-|z|)/2) = -sum_(n >= 1) p^n / n,   p^n = 2^-n exp(c_n / sigma), c_n = n s (y - mu),
// and each exponential is differentiated in closed form: with a = K - b derivatives in mu,
//   d^a/dmu^a d^b/dsigma^b exp(c/sigma) = (-n s)^a (-1)^b b! sigma^-(a+b) exp(c/sigma) L_b^(a-1)(u),
// u = -c/sigma = n |z| and L_b^(alpha) the generalized Laguerre polynomial, computed by its
// forward recurrence (k + 1) L_(k+1) = (2k + 1 + alpha - u) L_k - (k + alpha) L_(k-1),
// L_0 = 1, L_1 = 1 + alpha - u. Written as a polynomial in z, the same derivative has
// alternating terms much larger than their sum (391 ulp lost at z = -4.9, order 6); the
// recurrence has no such cancellation. Terms are added until p^n n^K (1 + n |z|)^K falls
// below 2^-60 of its value at n = 1.
template <int K>
LAPLACE_HD sym_tensor<2, K> log1m_half_exp_dk(double az, double s, double sigma) {
  const double p = 0.5 * math::exp(-az);
  double acc[K + 1] = {};  // acc[b] = sum_n n^(a-1) p^n L_b^(a-1)(n |z|)
  double ref = 1.0;
  for (int k = 0; k < K; ++k) ref *= 1.0 + az;
  double pn = 1.0;
  for (int n = 1; n <= 1000; ++n) {
    pn *= p;
    const double u = n * az;
    double npow = 1.0 / n;  // n^(a-1), a = 0 first
    double na[K + 1];
    for (int a = 0; a <= K; ++a, npow *= n) na[a] = npow;
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      // L_b^(a-1)(u). At a = 0 the order is -1, where the identity
      // L_b^(-1)(u) = -(u/b) L_(b-1)^(1)(u) (DLMF 18.5.12) removes a degeneracy of the
      // recurrence; it is worth using when the negative part of the order is large (laplace2,
      // where it goes to -b and saved a factor 32), but here it is only -1 and the identity
      // measured 12.2 ulp against 11.2 for log F and no change for log(1 - F) (2026-11-13),
      // so the plain recurrence is kept.
      const double alpha = a - 1.0;
      double l0 = 1.0, l1 = 1.0 + alpha - u;
      double lb = b == 0 ? l0 : l1;
      for (int k = 1; k < b; ++k) {
        const double l2 = ((2 * k + 1 + alpha - u) * l1 - (k + alpha) * l0) / (k + 1);
        l0 = l1;
        l1 = l2;
        lb = l2;
      }
      acc[b] += na[a] * pn * lb;
    }
    double g = pn;
    for (int k = 0; k < K; ++k) g *= n * (1.0 + u);
    if (g <= 0x1p-60 * p * ref) break;
  }
  sym_tensor<2, K> t;
  double isk = 1.0;
  for (int k = 0; k < K; ++k) isk /= sigma;
  for (int b = 0; b <= K; ++b) {
    const int a = K - b;
    double c = -fact(b) * isk;
    if (b % 2 == 1) c = -c;
    if (a % 2 == 1 && s > 0.0) c = -c;  // (-s)^a
    t[b] = c * acc[b];
  }
  return t;
}

}  // namespace detail_laplace1

/**
 * \ingroup distrib_laplace1
 * \brief Univariate Laplace (double exponential) distribution parameterized by location \f$\mu\f$ and scale \f$\sigma > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \sigma; y) = -\log(2\sigma) - \frac{|y - \mu|}{\sigma}\f$,
 * its parameter derivatives away from the kink at \f$\mu = y\f$, score-based Fisher information,
 * and cumulative distribution functions with series expansions for tails.
 * \see distrib_laplace1, distrib_core
 */
struct laplace1_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Scale parameter \f$\sigma \in (0, \infty)\f$.
  double sigma;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  // log f is C^0 in y at y = mu (y_kinks)
  /**
   * \ingroup distrib_laplace1
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (0, \f$C^0\f$ with a kink at \f$y = \mu\f$).
   */
  static constexpr int y_regularity = 0;  // regularity of log f in y (continuous only)
  /// Descriptors of \c mu and \c sigma.
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma) = -\log(2\sigma) - \frac{|y - \mu|}{\sigma}\f$.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value.
   */
  LAPLACE_HD double logpdf(double y) const {
    return -math::log(2.0 * sigma) - Kokkos::fabs(y - mu) / sigma;
  }

  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \sigma)\f$ in parameters \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    using detail_laplace1::fact;
    const double r = y - mu, is = 1.0 / sigma, az = Kokkos::fabs(r) * is;
    const double sg = r > 0.0 ? 1.0 : (r < 0.0 ? -1.0 : 0.0);
    sym_tensor<2, K> t;
    double isb = 1.0;  // sigma^-b
    for (int b = 0; b <= K; ++b, isb *= is) {
      const int a = K - b;
      const double sgn = (b % 2 == 0) ? 1.0 : -1.0;
      if (a == 0) t[b] = sgn * (fact(b - 1) - fact(b) * az) * isb;
      else if (a == 1) t[b] = sgn * fact(b) * sg * isb * is;
    }
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \sigma)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma)\f$. At \f$\mu = y\f$, uses \f$\operatorname{sgn}(0) = 0\f$.
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

  // d^(J+K) l / dy^J dtheta^K: l depends on y - mu only, so d/dy = -d/dmu and the component
  // with a derivatives in mu and b in sigma is (-1)^J times the (a + J, b) component of
  // logpdf_dk; at y = mu the convention sign(0) = 0 of logpdf_dk. J = 0 is logpdf_dk or the value.
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \sigma)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * Evaluates \f$\frac{\partial^{J+K}}{\partial y^J \partial\theta^K} \log f(y; \mu, \sigma)\f$ as a symmetric tensor of order \f$K\f$ in \f$(\mu, \sigma)\f$.
   * Because the log-density depends on \f$y - \mu\f$ only, \f$\frac{\partial}{\partial y} = -\frac{\partial}{\partial\mu}\f$. Away from the kink \f$y = \mu\f$,
   * for \f$J \ge 1\f$ letting \f$a = K - b + J\f$ and \f$b\f$ be the order in \f$\sigma\f$, the component at position \f$b\f$ is
   * \f$\frac{(-1)^{J + b} b!}{\sigma^{b + 1}} \operatorname{sgn}(z)\f$ for \f$a = 1\f$ and 0 for \f$a > 1\f$, where \f$z = \frac{y - \mu}{\sigma}\f$.
   * At the kink \f$y = \mu\f$ (\f$z = 0\f$), adopts the symmetric convention \f$\operatorname{sgn}(0) = 0\f$.
   * For \f$J = 0\f$, returns parameter derivatives (\f$K \ge 1\f$) or the log-density value (\f$K = 0\f$).
   *
   * Away from the kink, agrees with 5-point finite differences within relative tolerance \f$10^{-8}\f$.
   *
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \sigma)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \see distrib_laplace1, logpdf(), logpdf_dk(), y_kinks()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      using detail_laplace1::fact;
      const double r = y - mu, is = 1.0 / sigma;
      const double sg = r > 0.0 ? 1.0 : (r < 0.0 ? -1.0 : 0.0);
      const double sj = (J % 2 == 0) ? 1.0 : -1.0;
      double isb = 1.0;
      for (int b = 0; b <= K; ++b, isb *= is) {
        const int a = K - b + J;  // a >= 1; a > 1 gives 0 away from the kink
        const double sgn = (b % 2 == 0) ? 1.0 : -1.0;
        if (a == 1) t[b] = sj * sgn * fact(b) * sg * isb * is;
      }
      return t;
    }
  }
  /** @} */

  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \sigma) = \operatorname{diag}(1/\sigma^2, 1/\sigma^2)\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\sigma\sigma}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double i2 = 1.0 / (sigma * sigma);
    sym_tensor<2, 2> t;
    t[0] = i2;
    t[2] = i2;
    return t;
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    double d = detail_laplace1::fact(K + 1) / (sigma * sigma);
    for (int k = 0; k < K; ++k) d /= -sigma;
    expected_info_tensor<2, K> t;
    t.pair[0][K] = d;
    t.pair[2][K] = d;
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

  // F(y) = e^z/2 for z < 0, 1 - e^-z/2 otherwise; each tail written directly
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the cumulative distribution function \f$F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double z = (y - mu) / sigma;
    return z < 0.0 ? 0.5 * math::exp(z) : 1.0 - 0.5 * math::exp(-z);
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return z > 0.0 ? 0.5 * math::exp(-z) : 1.0 - 0.5 * math::exp(z);
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double z = (y - mu) / sigma;
    return z < 0.0 ? z - 0.69314718055994530942 : Kokkos::log1p(-0.5 * math::exp(-z));
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return z > 0.0 ? -z - 0.69314718055994530942 : Kokkos::log1p(-0.5 * math::exp(z));
  }
  // F^-1(p) = mu + sigma log(2p) below 1/2, mu - sigma log(2 - 2p) above (2 - 2p exact)
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates the quantile function \f$F^{-1}(p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const {
    return p < 0.5 ? mu + sigma * math::log(2.0 * p) : mu - sigma * math::log(2.0 - 2.0 * p);
  }

  // d^K log F / dtheta^K, d^K log(1 - F) / dtheta^K; log(1 - F)(z) = log F_0(-z). The side
  // where the function is log(1 - e^(-|z|)/2) (log F for z >= series_split, log(1 - F) for
  // z <= -series_split) by the series of log1m_half_exp_dk, the rest by mixed_dk
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z >= detail_laplace1::series_split) return detail_laplace1::log1m_half_exp_dk<K>(z, -1.0, sigma);
    double L[K + 1];
    detail_laplace1::log_cdf0_derivs<K>(z, L);
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    if (z <= -detail_laplace1::series_split) return detail_laplace1::log1m_half_exp_dk<K>(-z, 1.0, sigma);
    double L[K + 1];
    detail_laplace1::log_cdf0_derivs<K>(-z, L);
    for (int j = 1; j <= K; j += 2) L[j] = -L[j];
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
  // mu: kink at y, continuous (order 0); sigma: none
  /**
   * \ingroup distrib_laplace1
   * \brief Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$.
   *
   * For \f$j = 0\f$ (\c mu), returns one kink at \f$\mu = y\f$ with \c continuous_order = 0.
   * For \f$j = 1\f$ (\c sigma), reports no kinks.
   * \param j Parameter index (0 for \c mu, 1 for \c sigma).
   * \param y Observation value.
   * \return Kinks descriptor \c distrib_kinks.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    distrib_kinks k;
    if (j == 0) {
      k.count = 1;
      k.point[0] = y;
    }
    return k;
  }
  /**
   * \ingroup distrib_laplace1
   * \brief Non-smooth points in observation \f$y\f$ for parameters \f$(\mu, \sigma)\f$.
   *
   * Returns one kink at \f$y = \mu\f$ with \c continuous_order = 0.
   * \return Kinks descriptor \c distrib_kinks.
   */
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    k.count = 1;
    k.point[0] = mu;
    return k;
  }
};

}  // namespace laplace::distrib
