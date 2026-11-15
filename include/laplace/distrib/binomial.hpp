// binomial_distrib (laplace::distrib): the binomial distribution in its mean
// proportion mu in (0, 1) with a known size n (a trailing field of the struct, not a
// parameter), on y = 0, 1, ..., n.
//
//   l(mu; y) = log C(n, y) + y log mu + (n - y) log(1 - mu)
//
// The log-mass is evaluated in Loader's saddle-point form, which has no large cancelling
// terms at any n:
//
//   l = (1/2) log(n / (2 pi y (n - y))) + delta(n) - delta(y) - delta(n - y)
//       - n mu D(y / (n mu)) - n (1 - mu) D((n - y) / (n (1 - mu)))
//
// with delta Stirling's error and D(z) = z log z - z + 1; the ends y = 0 and y = n are
// n log(1 - mu) and n log mu. The two deviance terms take the relative deviation w = (y - n mu)
// / (n mu) of y from its mean, and (n - y) / (n (1 - mu)) - 1 = -(y - n mu) / (n (1 - mu)),
// from the one numerator y - n mu, formed by a fused multiply-add: the rounded product n mu
// would otherwise move the mass by tens of ulp at n = 1e6.
//
// Derivatives in mu: d^K l = (K-1)! [ (-1)^(K-1) y / mu^K - (n - y) / (1 - mu)^K ],
// at order 1 written (y - n mu) / (mu (1 - mu)) with the numerator by a fused multiply-add.
// Expected information I = n / (mu (1 - mu)), d^K I = n [ (-1)^K K! / mu^(K+1) + K! / (1 -
// mu)^(K+1) ]. With k = floor(y), 0 <= k < n: F(k) = 1 - I_mu(k + 1, n - k) and
// 1 - F(k) = I_mu(k + 1, n - k), the regularized incomplete beta function taken at the
// argument mu itself (1 - mu is never formed), each side computed directly.
//
// The derivatives of log F and log(1 - F) in mu are not written here: they come from the
// generic route by summing the mass (distrib_log_cdf_dk<K>, distrib_log_ccdf_dk<K> of
// generic_cdf.hpp), at a cost proportional to the number of terms of the tail summed.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_binomial {

// D(1 + w), D(z) = z log z - z + 1: the series sum_(k>=2) (-1)^k w^k / (k (k-1)) below |w| = 1/4,
// (1 + w) log1p(w) - w above
LAPLACE_HD_FORCEINLINE double deviance_unit(double w) {
  if (Kokkos::fabs(w) < 0.25) {
    double p = w * w, s = 0.0;
    for (int k = 2; k <= 40; ++k, p *= -w) {
      const double t = p / (k * (k - 1));
      s += t;
      if (Kokkos::fabs(t) <= 0x1p-54 * s) break;
    }
    return s;
  }
  return (1.0 + w) * Kokkos::log1p(w) - w;
}

}  // namespace detail_binomial

/**
 * \ingroup distrib_binomial
 * \brief Binomial discrete distribution with known size \f$n\f$, parameterized by the mean proportion \f$\mu \in (0, 1)\f$.
 *
 * Implements the log-mass \f$\ell(\mu; y) = \log\binom{n}{y} + y\log\mu + (n-y)\log(1-\mu)\f$ in the
 * saddle-point form with Stirling's error, its derivatives in \f$\mu\f$ to order 6, the expected
 * Fisher information with its derivatives to order 4, and the cumulative distribution function
 * through the regularized incomplete beta function. The size is a data constant: a trailing field,
 * not counted in \c n_params and not listed in \c params.
 * \see distrib_binomial, distrib_core, stirling, incomplete_beta
 */
struct binomial_distrib {
  /// Mean proportion \f$\mu \in (0, 1)\f$.
  double mu;
  /// Known size \f$n \ge 1\f$ (an integer held in a double), the upper end of the support.
  double size;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (1; the size is not one).
  static constexpr int n_params = 1;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {{"mu", 0.0, 1.0, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the log-mass \f$\log p(y; \mu)\f$ by the saddle-point representation.
   *
   * Returns \f$-\infty\f$ when \f$y\f$ is not an integer in \f$\{0, \dots, n\}\f$.
   * \param y Observation point.
   * \return Log-mass value.
   * \see distrib_binomial, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    const double n = size;
    if (!(y >= 0.0 && y <= n) || y != Kokkos::floor(y)) return -distrib_infinity;
    if (y == 0.0) return n * Kokkos::log1p(-mu);
    if (y == n) return n * math::log(mu);
    const double m = n - y, np = n * mu, nq = n * (1.0 - mu), e = Kokkos::fma(-n, mu, y);  // y - n mu
    return 0.5 * math::log(n / (6.28318530717958647693 * y * m)) + math::stirling_error(n) - math::stirling_error(y) -
           math::stirling_error(m) - np * detail_binomial::deviance_unit(e / np) -
           nq * detail_binomial::deviance_unit(-e / nq);
  }

  /**
   * \ingroup distrib_binomial
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu)\f$ with respect to \f$\mu\f$.
   *
   * \f$(K-1)!\,[(-1)^{K-1} y/\mu^K - (n-y)/(1-\mu)^K]\f$; near the mode \f$y = n\mu\f$ the two terms cancel
   * where the derivative is genuinely small (order 1 is written with a fused multiply-add so that it does not).
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Integer observation point in \f$\{0, \dots, n\}\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpmf_dk(double y) const {
    sym_tensor<1, K> t;
    const double q = 1.0 - mu;
    if constexpr (K == 1) {
      t[0] = Kokkos::fma(-size, mu, y) / (mu * q);
    } else {
      double a = y, b = size - y;  // y / mu^K and (n - y) / q^K
      for (int k = 1; k <= K; ++k) a /= mu, b /= q;
      double f = 1.0;  // (K-1)!
      for (int k = 2; k < K; ++k) f *= k;
      t[0] = f * (((K & 1) ? a : -a) - b);
    }
    return t;
  }
  /**
   * \name Derivatives of the log-mass
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log p(y; \mu)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> logpmf_d1(double y) const { return logpmf_dk<1>(y); }
  /// Second-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> logpmf_d2(double y) const { return logpmf_dk<2>(y); }
  /// Third-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> logpmf_d3(double y) const { return logpmf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> logpmf_d4(double y) const { return logpmf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> logpmf_d5(double y) const { return logpmf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> logpmf_d6(double y) const { return logpmf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the expected Fisher information \f$I(\mu) = n / (\mu (1 - \mu))\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = size / (mu * (1.0 - mu));
    return t;
  }
  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the \f$K\f$-th derivative of the expected Fisher information with respect to \f$\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    const double q = 1.0 - mu;
    double a = 1.0 / mu, b = 1.0 / q, f = 1.0;  // 1 / mu^(K+1), 1 / q^(K+1), K!
    for (int k = 1; k <= K; ++k) a /= mu, b /= q, f *= k;
    expected_info_tensor<1, K> t;
    t.pair[0][0] = size * f * (((K & 1) ? -a : a) + b);
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} I(\mu)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the cumulative distribution function \f$F(y) = \mathbb{P}(Y \le y)\f$.
   * \param y Observation point.
   * \return 0 for \f$y < 0\f$, 1 for \f$\lfloor y \rfloor \ge n\f$, otherwise \f$1 - I_\mu(k + 1, n - k)\f$, \f$k = \lfloor y \rfloor\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    if (y < 0.0) return 0.0;
    const double k = Kokkos::floor(y);
    return k >= size ? 1.0 : math::ibetac(k + 1.0, size - k, mu);
  }
  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point.
   * \return 1 for \f$y < 0\f$, 0 for \f$\lfloor y \rfloor \ge n\f$, otherwise \f$I_\mu(k + 1, n - k)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    if (y < 0.0) return 1.0;
    const double k = Kokkos::floor(y);
    return k >= size ? 0.0 : math::ibeta(k + 1.0, size - k, mu);
  }
  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   *
   * Returns \f$-\infty\f$ when \f$y < 0\f$ and 0 when \f$\lfloor y \rfloor \ge n\f$. The derivatives
   * in \f$\mu\f$ come from distrib_log_cdf_dk().
   * \param y Observation point.
   * \return Log-probability \f$\log F(y)\f$.
   * \see distrib_binomial, cdf()
   */
  LAPLACE_HD double log_cdf(double y) const {
    if (y < 0.0) return -distrib_infinity;
    const double k = Kokkos::floor(y);
    return k >= size ? 0.0 : math::log_ibetac(k + 1.0, size - k, mu);
  }
  /**
   * \ingroup distrib_binomial
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   *
   * Returns 0 when \f$y < 0\f$ and \f$-\infty\f$ when \f$\lfloor y \rfloor \ge n\f$.
   * \param y Observation point.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   * \see distrib_binomial, ccdf()
   */
  LAPLACE_HD double log_ccdf(double y) const {
    if (y < 0.0) return 0.0;
    const double k = Kokkos::floor(y);
    return k >= size ? -distrib_infinity : math::log_ibeta(k + 1.0, size - k, mu);
  }

  /// \ingroup distrib_binomial
  /// \brief Lower bound of the support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_binomial
  /// \brief Upper bound of the support (the size \f$n\f$).
  LAPLACE_HD double support_upper() const { return size; }
  /// \ingroup distrib_binomial
  /// \brief Central location (\f$n\mu\f$).
  LAPLACE_HD double center() const { return size * mu; }
  /// \ingroup distrib_binomial
  /// \brief Typical spread (\f$\sqrt{n\mu(1-\mu)}\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(size * mu * (1.0 - mu)); }
  /// \ingroup distrib_binomial
  /// \brief Non-smooth points in parameter \f$\mu\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
};

}  // namespace laplace::distrib
