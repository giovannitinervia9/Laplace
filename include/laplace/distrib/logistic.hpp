// logistic_distrib (laplace::distrib): the logistic distribution in its location mu and scale
// sigma > 0. With z = (y - mu)/sigma and p(z) = F(z) = 1/(1 + e^(-z)), q = 1 - p,
//
//   l(mu, sigma; y) = -log(sigma) - z - 2 log(1 + e^(-z)) = -log(sigma) + log(p q),
//   F(y) = p.
//
// l, log F and log(1 - F) are C^inf in both parameters and in y: no kinks.
//
// The z-derivatives all come from one recurrence in p. Since p' = p q = p (1 - p), the
// derivatives s_k = d^k p / dz^k, k = 0, 1, ..., satisfy
//   s_0 = p,   s_1 = p q,   s_(k+1) = (q - p) s_k - sum_(i=1)^(k-1) C(k, i) s_i s_(k-i),
// (q - p = 1 - 2p). Then, with Lambda_f = log(p q), Lambda_F = log p and Lambda_(1-F) = log q,
//   d/dz Lambda_f = q - p,           d^j/dz^j Lambda_f = -2 s_(j-1)       (j >= 2),
//   d/dz Lambda_F = q,               d^j/dz^j Lambda_F = -s_(j-1)         (j >= 2),
//   d/dz Lambda_(1-F) = -p,          d^j/dz^j Lambda_(1-F) = -s_(j-1)     (j >= 2).
// The recurrence has no alternating sums of large terms: on each side one of p, q is small
// and the products s_i s_(k-i) are smaller than the terms they add to.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_logistic {

inline constexpr double pi_value = 3.14159265358979323846;

// Fisher information of the location (1/3) and of the scale (pi^2 + 3)/9, both divided by sigma^2
inline constexpr double info_mu = 1.0 / 3.0;
inline constexpr double info_sigma = (pi_value * pi_value + 3.0) / 9.0;

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// p = F(z) and q = 1 - p, each computed where it is accurate: p from e^(-z) for z >= 0 and from
// e^z for z < 0 (no overflow in either case), q from the other exponential.
LAPLACE_HD_FORCEINLINE void pq(double z, double& p, double& q) {
  if (z >= 0.0) {
    const double e = math::exp(-z);
    p = 1.0 / (1.0 + e);
    q = e / (1.0 + e);
  } else {
    const double e = math::exp(z);
    p = e / (1.0 + e);
    q = 1.0 / (1.0 + e);
  }
}

// s[k] = d^k p / dz^k for k = 0..K-1 (s[0] = p), by the recurrence of the header comment
template <int K>
LAPLACE_HD void p_derivs(double p, double q, double* s) {
  s[0] = p;
  s[1] = p * q;
  for (int k = 1; k + 1 < K; ++k) {
    double acc = (q - p) * s[k];
    for (int i = 1; i < k; ++i) acc -= binomial_coef(k, i) * s[i] * s[k - i];
    s[k + 1] = acc;
  }
}

// L[j] = d^j/dz^j log(p q), j = 1..K (the log-density without -log sigma)
template <int K>
LAPLACE_HD void logf_derivs(double z, double* L) {
  double p, q;
  pq(z, p, q);
  double s[K + 1];
  p_derivs<K>(p, q, s);
  L[1] = q - p;
  for (int j = 2; j <= K; ++j) L[j] = -2.0 * s[j - 1];
}

// L[j] = d^j/dz^j log p, j = 1..K (log F)
template <int K>
LAPLACE_HD void log_cdf0_derivs(double z, double* L) {
  double p, q;
  pq(z, p, q);
  double s[K + 1];
  p_derivs<K>(p, q, s);
  L[1] = q;
  for (int j = 2; j <= K; ++j) L[j] = -s[j - 1];
}

// L[j] = d^j/dz^j log q, j = 1..K (log(1 - F))
template <int K>
LAPLACE_HD void log_ccdf0_derivs(double z, double* L) {
  double p, q;
  pq(z, p, q);
  double s[K + 1];
  p_derivs<K>(p, q, s);
  L[1] = -p;
  for (int j = 2; j <= K; ++j) L[j] = -s[j - 1];
}

}  // namespace detail_logistic

/**
 * \ingroup distrib_logistic
 * \brief Univariate logistic distribution parameterized by location \f$\mu\f$ and scale \f$\sigma > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \sigma; y) = -\log\sigma - z - 2\log(1 + e^{-z})\f$ with
 * \f$z = (y - \mu)/\sigma\f$, its parameter derivatives from one recurrence in \f$p = F(z)\f$, the Fisher information
 * \f$\operatorname{diag}(1/(3\sigma^2), (\pi^2 + 3)/(9\sigma^2))\f$, and the cumulative distribution function
 * \f$F(y) = 1/(1 + e^{-z})\f$.
 * \see distrib_core
 */
struct logistic_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// Scale parameter \f$\sigma \in (0, \infty)\f$.
  double sigma;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_logistic
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth, no kinks).
   */
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c sigma (both \c C^inf, neither moves the support).
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma) = -\log\sigma - |z| - 2\log(1 + e^{-|z|})\f$.
   *
   * The density is symmetric in \f$z\f$, so \f$|z|\f$ is used and \f$e^{-|z|} \le 1\f$ cannot overflow.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    const double az = Kokkos::fabs(z);
    return -math::log(sigma) - az - 2.0 * Kokkos::log1p(math::exp(-az));
  }

  /**
   * \ingroup distrib_logistic
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \sigma)\f$ in \f$(\mu, \sigma)\f$.
   *
   * The density is \f$\log f = \Lambda(z) - \log\sigma\f$ with \f$\Lambda = \log(p q)\f$. The derivatives of
   * \f$\Lambda\f$ are carried to \f$(\mu, \sigma)\f$ by \c detail_location_scale::mixed_dk, and \f$-\log\sigma\f$
   * adds \f$(-1)^K (K - 1)!\,\sigma^{-K}\f$ to the component with all \f$K\f$ indices in \f$\sigma\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double L[K + 1];
    detail_logistic::logf_derivs<K>(z, L);
    sym_tensor<2, K> t = detail_location_scale::mixed_dk<K>(L, z, sigma);
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    const double sgn = (K % 2 == 0) ? 1.0 : -1.0;
    t[K] += sgn * detail_logistic::fact(K - 1) * isk;
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
  /** @} */

  // l depends on y and mu only through y - mu, so d/dy = -d/dmu: the component with b derivatives in sigma
  // of d^(J+K) l / dy^J dtheta^K is (-1)^J times that of logpdf_dk<J + K> at position b.
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \sigma)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * Evaluates \f$\frac{\partial^{J+K}}{\partial y^J \partial\theta^K} \log f(y; \mu, \sigma)\f$ as a symmetric tensor of order
   * \f$K\f$ in \f$(\mu, \sigma)\f$. For \f$J = 0\f$ the result is \c logpdf_dk<K> (or the value for \f$K = 0\f$).
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \sigma)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \see logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<2, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<2, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<2, K> r;
      const double s = (J % 2 == 0) ? 1.0 : -1.0;
      for (int b = 0; b <= K; ++b) r[b] = s * t[b];
      return r;
    }
  }

  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \sigma) = \operatorname{diag}(1/(3\sigma^2), (\pi^2 + 3)/(9\sigma^2))\f$.
   *
   * The scale entry is \f$\int_{\mathbb{R}} (1 + z(1 - 2p))^2 f(z)\,dz / \sigma^2\f$ with the logistic density \f$f = pq\f$,
   * which equals \f$(\pi^2/9 + 4/3 - 1)/\sigma^2 = (\pi^2 + 3)/(9\sigma^2)\f$ (integration by parts, checked in
   * \c tools/reference/cauchy_logistic_forms.py).
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\sigma\sigma}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double s2 = sigma * sigma;
    sym_tensor<2, 2> t;
    t[0] = detail_logistic::info_mu / s2;
    t[2] = detail_logistic::info_sigma / s2;
    return t;
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma)\f$.
   *
   * Each diagonal entry is a constant \f$c\f$ over \f$\sigma^2\f$, so its \f$K\f$-th derivative in \f$\sigma\f$ is
   * \f$(-1)^K (K+1)!\, c/\sigma^{K+2}\f$, and the derivatives in \f$\mu\f$ vanish.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    const double f = detail_logistic::fact(K + 1) / (sigma * sigma);
    double d0 = f * detail_logistic::info_mu, d2 = f * detail_logistic::info_sigma;
    for (int k = 0; k < K; ++k) {
      d0 /= -sigma;
      d2 /= -sigma;
    }
    expected_info_tensor<2, K> t;
    t.pair[0][K] = d0;
    t.pair[2][K] = d2;
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
   * \ingroup distrib_logistic
   * \brief Evaluates the cumulative distribution function \f$F(y) = 1/(1 + e^{-z})\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double z = (y - mu) / sigma;
    double p, q;
    detail_logistic::pq(z, p, q);
    return p;
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = 1/(1 + e^{z})\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double z = (y - mu) / sigma;
    double p, q;
    detail_logistic::pq(z, p, q);
    return q;
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y) = -\operatorname{log1p}(e^{-z})\f$.
   *
   * For \f$z < 0\f$ the equal form \f$z - \operatorname{log1p}(e^{z})\f$ is used, so \f$e^{-z}\f$ never overflows.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double z = (y - mu) / sigma;
    return z < 0.0 ? z - Kokkos::log1p(math::exp(z)) : -Kokkos::log1p(math::exp(-z));
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y)) = -\operatorname{log1p}(e^{z})\f$.
   *
   * For \f$z > 0\f$ the equal form \f$-z - \operatorname{log1p}(e^{-z})\f$ is used.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return z > 0.0 ? -z - Kokkos::log1p(math::exp(-z)) : -Kokkos::log1p(math::exp(z));
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu + \sigma\log(p/(1 - p))\f$.
   *
   * The ratio \f$p/(1 - p)\f$ keeps its digits at both ends: \f$1 - p\f$ is exact for \f$p \ge 1/2\f$ (Sterbenz) and
   * \f$p\f$ itself is exact for \f$p < 1/2\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return mu + sigma * math::log(p / (1.0 - p)); }

  /**
   * \ingroup distrib_logistic
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double L[K + 1];
    detail_logistic::log_cdf0_derivs<K>(z, L);
    return detail_location_scale::mixed_dk<K>(L, z, sigma);
  }
  /**
   * \ingroup distrib_logistic
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double L[K + 1];
    detail_logistic::log_ccdf0_derivs<K>(z, L);
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
  /**
   * \ingroup distrib_logistic
   * \brief Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$: none, for both parameters.
   * \return Kinks descriptor \c distrib_kinks with \c count = 0.
   */
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /**
   * \ingroup distrib_logistic
   * \brief Non-smooth points in the observation \f$y\f$: none.
   * \return Kinks descriptor \c distrib_kinks with \c count = 0.
   */
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
};

}  // namespace laplace::distrib
