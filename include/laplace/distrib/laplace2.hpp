// laplace2_distrib (laplace::distrib): the Laplace (double exponential) distribution in its
// location mu and RATE lambda > 0, the same law as laplace1_distrib at sigma = 1/lambda:
//
//   l(mu, lambda; y) = log(lambda/2) - lambda |y - mu|.
//
// With r = y - mu and s = sign(r), the components of d^a_mu d^b_lambda l (a + b = K, b the
// position in the packed tensor) away from the kink are:
//   a = 0: b = 0: log(lambda/2) - lambda |r|;  b = 1: 1/lambda - |r|;  b >= 2: (-1)^(b-1) (b-1)!/lambda^b
//   a = 1: b = 0: lambda s;  b = 1: s;  b >= 2 or a >= 2: 0
// At the kink r = 0 the convention is sign(0) = 0, the mean of the two one-sided
// derivatives, as in laplace1_distrib. l is C^0 in mu (kink at mu = y) and C^inf in lambda.
//
// log F and log(1 - F) are C^1 in mu at mu = y (their second derivative in mu jumps by
// 2 lambda^2 there), C^inf in lambda. With z = lambda (y - mu), the expansion
//   log(1 - e^(-az)/2) = -sum_(n >= 1) 2^-n e^(-n az)/n,   az = lambda q, q = y - mu (log F, z > 0)
//                                                          or  q = mu - y (log(1 - F), z < 0),
// is differentiated term by term in closed form: with eps = dq/dmu (= s, the sign that laplace1
// uses) and u = n az,
//   d^a_mu d^b_lambda e^(-n lambda q) = (-n eps)^a lambda^(a-b) e^(-u) b! L_b^(a-b)(u),
// where L_b^(alpha) is the generalized Laguerre polynomial (forward recurrence, as in
// laplace1's log1m_half_exp_dk). Below |z| = 1/4 the series is replaced by the z-polynomials of
// detail_laplace1::log_cdf0_derivs carried to (mu, lambda) by mixed_power_dk, which is where
// those polynomials are accurate (laplace1's series_split).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_laplace2 {

// n! as a double (n <= 20 for the orders used here)
LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^n
LAPLACE_HD_FORCEINLINE constexpr double neg_one_pow(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// x^n for n >= 0
LAPLACE_HD_FORCEINLINE double pow_int(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}

// x^e for any integer e (negative powers by division)
LAPLACE_HD_FORCEINLINE double signed_pow(double x, int e) { return e >= 0 ? pow_int(x, e) : 1.0 / pow_int(x, -e); }

// Generalized Laguerre polynomial L_n^(alpha)(x) by its forward recurrence in n (see gumbel.hpp):
// (k + 1) L_(k+1) = (2k + 1 + alpha - x) L_k - (k + alpha) L_(k-1), L_0 = 1, L_1 = 1 + alpha - x.
LAPLACE_HD_FORCEINLINE double laguerre(int n, double alpha, double x) {
  if (n == 0) return 1.0;
  double l0 = 1.0, l1 = 1.0 + alpha - x;
  for (int k = 1; k < n; ++k) {
    const double l2 = ((2.0 * k + 1.0 + alpha - x) * l1 - (k + alpha) * l0) / (k + 1.0);
    l0 = l1;
    l1 = l2;
  }
  return l1;
}

// d^a_mu d^b_lambda l, a + b >= 1, with r = y - mu and s = sign(r) (sign(0) = 0)
LAPLACE_HD_FORCEINLINE double logpdf_component(int a, int b, double lambda, double r) {
  const double ar = Kokkos::fabs(r);
  const double s = r > 0.0 ? 1.0 : (r < 0.0 ? -1.0 : 0.0);
  if (a == 0) {
    if (b == 1) return 1.0 / lambda - ar;
    if (b >= 2) return neg_one_pow(b - 1) * fact(b - 1) / signed_pow(lambda, b);
    return 0.0;  // b = 0 is the value, not a component of order K >= 1
  }
  if (a == 1) {
    if (b == 0) return lambda * s;
    if (b == 1) return s;
  }
  return 0.0;
}

// log(1 - e^(-az)/2) and its derivatives in (mu, lambda), order K, with az > 0 and the sign
// s = dq/dmu of the relevant side (s = -1 for log F, s = +1 for log(1 - F)); see the header.
// Terms are added until the n-th term falls below 2^-60 of the first, with the stopping test of
// laplace1's log1m_half_exp_dk.
template <int K>
LAPLACE_HD sym_tensor<2, K> log1m_half_exp_lambda_dk(double az, double s, double lambda) {
  const double p = 0.5 * math::exp(-az);
  double acc[K + 1] = {};  // acc[b] = sum_n (p^n/n) (-n s)^a lambda^(a-b) b! L_b^(a-b)(n az), a = K - b
  double ref = pow_int(1.0 + az, K);
  double pn = 1.0;  // p^n
  for (int n = 1; n <= 1000; ++n) {
    pn *= p;
    const double u = n * az;
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      // b! L_b^(a-b)(u). For a < b the order a - b is a negative integer, where the forward
      // recurrence starts from terms of size b and has to cancel down to a result of size u^(b-a):
      // at a = 0 the exact value is (-u)^b / b!, and the recurrence lost 720 ulp of it at u near
      // 1/2 (measured 2026-11-13, order 6 all in lambda). The identity
      //   L_b^(-m)(u) = (-u)^m ((b - m)! / b!) L_(b-m)^(m)(u),   m = b - a <= b
      // (DLMF 18.5.12) moves the degeneracy into the explicit factor (-u)^m and leaves a
      // recurrence of positive order m, which does not cancel:
      //   b! L_b^(a-b)(u) = (-u)^(b-a) a! L_a^(b-a)(u).
      const double lag = a < b ? pow_int(-u, b - a) * fact(a) * laguerre(a, b - a, u)
                               : fact(b) * laguerre(b, a - b, u);
      acc[b] += (pn / n) * pow_int(-n * s, a) * signed_pow(lambda, a - b) * lag;
    }
    double g = pn;
    for (int k = 0; k < K; ++k) g *= n * (1.0 + u);
    if (g <= 0x1p-60 * p * ref) break;
  }
  sym_tensor<2, K> t;
  for (int b = 0; b <= K; ++b) t[b] = -acc[b];
  return t;
}

}  // namespace detail_laplace2

/**
 * \ingroup distrib_laplace2
 * \brief Univariate Laplace (double exponential) distribution parameterized by location \f$\mu\f$ and rate \f$\lambda > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \lambda; y) = \log(\lambda/2) - \lambda|y - \mu|\f$, the
 * same law as laplace1_distrib with \f$\sigma = 1/\lambda\f$, its parameter derivatives away from the kink at
 * \f$\mu = y\f$ (with \f$\operatorname{sgn}(0) = 0\f$ there), the expected Fisher information, and the
 * cumulative distribution functions.
 * \see distrib_laplace2, laplace1_distrib
 */
struct laplace2_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Rate parameter \f$\lambda \in (0, \infty)\f$ (the inverse of the scale of laplace1_distrib).
  double lambda;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_laplace2
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (0, \f$C^0\f$ with a kink at \f$y = \mu\f$).
   */
  static constexpr int y_regularity = 0;
  /// Descriptors of \c mu (\f$C^0\f$ in the log-density, \f$C^1\f$ in log F) and \c lambda (\f$C^\infty\f$).
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1},
      {"lambda", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the log-density \f$\log f(y; \mu, \lambda) = \log\lambda - \log 2 - \lambda|y - \mu|\f$.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value.
   */
  LAPLACE_HD double logpdf(double y) const {
    return math::log(lambda) - 0.69314718055994530942 - lambda * Kokkos::fabs(y - mu);
  }

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \lambda)\f$ in \f$(\mu, \lambda)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    using namespace detail_laplace2;
    const double r = y - mu;
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = logpdf_component(K - b, b, lambda, r);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \lambda)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$. At \f$\mu = y\f$, uses \f$\operatorname{sgn}(0) = 0\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \lambda)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * The log-density depends on \f$y\f$ and \f$\mu\f$ through \f$y - \mu\f$ only, so \f$\partial/\partial y = -\partial/\partial\mu\f$:
   * the component with \f$a\f$ derivatives in \f$\mu\f$ and \f$b\f$ in \f$\lambda\f$ is \f$(-1)^J\f$ times the
   * \f$(a + J, b)\f$ component of \f$\log f\f$. Away from the kink this is nonzero only for \f$J = 1\f$ and \f$a = 0\f$.
   * At the kink \f$y = \mu\f$ the convention \f$\operatorname{sgn}(0) = 0\f$ of logpdf_dk applies.
   *
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters (\f$K \ge 0\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    using namespace detail_laplace2;
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      const double r = y - mu;
      for (int b = 0; b <= K; ++b) t[b] = neg_one_pow(J) * logpdf_component(K - b + J, b, lambda, r);
      return t;
    }
  }

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \lambda) = \operatorname{diag}(\lambda^2, \lambda^{-2})\f$.
   *
   * Given \f$\lambda\f$, \f$|Y - \mu|\f$ is exponential with mean \f$1/\lambda\f$. The score in \f$\mu\f$ is \f$\lambda s\f$
   * with \f$s = \pm 1\f$, so \f$I_{\mu\mu} = \lambda^2\f$; the score in \f$\lambda\f$ is \f$1/\lambda - |r|\f$, whose
   * variance is \f$\lambda^{-2}\f$; the cross term vanishes by the symmetry of \f$s\f$. The change of parametrization
   * from laplace1_distrib confirms it: \f$I_{\lambda\lambda} = \sigma^4 I_{\sigma\sigma} = \lambda^{-2}\f$.
   * Packed as \c sym_tensor<2, 2>: \f$I_{\mu\mu}\f$ at index 0, \f$I_{\lambda\lambda}\f$ at index 2.
   * \return Symmetric tensor \c sym_tensor<2, 2>.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    sym_tensor<2, 2> t;
    t[0] = lambda * lambda;
    t[1] = 0.0;
    t[2] = 1.0 / (lambda * lambda);
    return t;
  }

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \lambda)\f$.
   *
   * \f$I_{\mu\mu} = \lambda^2\f$ has derivatives \f$2\lambda\f$ (\f$K = 1\f$), \f$2\f$ (\f$K = 2\f$) and 0 beyond;
   * \f$I_{\lambda\lambda} = \lambda^{-2}\f$ has \f$K\f$-th derivative \f$(-1)^K (K+1)! \lambda^{-(K+2)}\f$. The
   * derivatives in \f$\mu\f$ are zero.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    using namespace detail_laplace2;
    expected_info_tensor<2, K> t;
    if constexpr (K == 1) t.pair[0][K] = 2.0 * lambda;
    else if constexpr (K == 2) t.pair[0][K] = 2.0;
    t.pair[2][K] = neg_one_pow(K) * fact(K + 1) / signed_pow(lambda, K + 2);
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \lambda)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \lambda)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \lambda)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \lambda)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \lambda)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the cumulative distribution function \f$F(y)\f$, with \f$z = \lambda(y - \mu)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double z = lambda * (y - mu);
    return z < 0.0 ? 0.5 * math::exp(z) : 1.0 - 0.5 * math::exp(-z);
  }
  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double z = lambda * (y - mu);
    return z > 0.0 ? 0.5 * math::exp(-z) : 1.0 - 0.5 * math::exp(z);
  }
  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double z = lambda * (y - mu);
    return z < 0.0 ? z - 0.69314718055994530942 : Kokkos::log1p(-0.5 * math::exp(-z));
  }
  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = lambda * (y - mu);
    return z > 0.0 ? -z - 0.69314718055994530942 : Kokkos::log1p(-0.5 * math::exp(z));
  }
  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu + \log(2p)/\lambda\f$ below \f$p = 1/2\f$, and
   * \f$\mu - \log(2 - 2p)/\lambda\f$ above (with \f$2 - 2p\f$ exact there).
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const {
    return p < 0.5 ? mu + math::log(2.0 * p) / lambda : mu - math::log(2.0 - 2.0 * p) / lambda;
  }

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \lambda)\f$.
   *
   * For \f$z < -1/4\f$ the value is \f$z - \log 2\f$ and its derivatives are closed. For \f$z \ge 1/4\f$ the
   * series of log(1 - e^(-z)/2) is used (detail_laplace2::log1m_half_exp_lambda_dk); between the two the
   * z-polynomials of detail_laplace1::log_cdf0_derivs are carried by mixed_power_dk.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double z = lambda * (y - mu);
    if (z >= detail_laplace1::series_split) return detail_laplace2::log1m_half_exp_lambda_dk<K>(z, -1.0, lambda);
    double L[K + 1];
    detail_laplace1::log_cdf0_derivs<K>(z, L);
    return detail_location_scale::mixed_power_dk<K>(L, z, 1.0 / lambda, lambda, -1.0);
  }

  /**
   * \ingroup distrib_laplace2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \lambda)\f$.
   *
   * The mirror image of log_cdf_dk: for \f$z \le -1/4\f$ the series with \f$s = +1\f$; for \f$z > -1/4\f$ the
   * z-polynomials of \f$\log F_0(-z)\f$ with the odd orders negated, carried by mixed_power_dk.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double z = lambda * (y - mu);
    if (z <= -detail_laplace1::series_split) return detail_laplace2::log1m_half_exp_lambda_dk<K>(-z, 1.0, lambda);
    double L[K + 1];
    detail_laplace1::log_cdf0_derivs<K>(-z, L);
    for (int j = 1; j <= K; j += 2) L[j] = -L[j];
    return detail_location_scale::mixed_power_dk<K>(L, z, 1.0 / lambda, lambda, -1.0);
  }

  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \lambda)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread, the scale \f$1/\lambda\f$.
  LAPLACE_HD double scale() const { return 1.0 / lambda; }

  /**
   * \ingroup distrib_laplace2
   * \brief Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$.
   *
   * For \f$j = 0\f$ (\c mu), one kink at \f$\mu = y\f$ with \c continuous_order = 0. For \f$j = 1\f$ (\c lambda), none.
   * \param j Parameter index (0 for \c mu, 1 for \c lambda).
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
   * \ingroup distrib_laplace2
   * \brief Non-smooth points in observation \f$y\f$: one kink at \f$y = \mu\f$ with \c continuous_order = 0.
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
