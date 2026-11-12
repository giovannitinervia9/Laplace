// poisson_distrib (laplace::distrib): the Poisson distribution in its mean
// mu > 0, on y = 0, 1, 2, ...
//
//   l(mu; y) = y log mu - mu - log y!
//
// Derivatives: y/mu - 1 at order 1, (-1)^(K-1) (K-1)! y / mu^K at K > 1.
// Expected information 1/mu, its K-th derivative (-1)^K K! / mu^(K+1).
// F(k) = Q(k + 1, mu) and 1 - F(k) = P(k + 1, mu), the regularized
// incomplete gamma functions.
//
// Derivatives of log F and log(1 - F) in mu, to order 6: those of
// log Q(k + 1, x) and log P(k + 1, x) in x at x = mu, the components with K
// derivatives in x of laplace::math::log_gamma_{q,p}_mixed_dK.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/gamma_cdf.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_poisson {

// D(z) = z log z - z + 1, z = y/mu: the series sum_(n>=2) (-1)^n w^n / (n (n-1)) in
// w = (y - mu)/mu below |w| = 1/4, the formula above
LAPLACE_HD_FORCEINLINE double deviance_unit(double y, double mu) {
  const double w = (y - mu) / mu;
  if (Kokkos::fabs(w) < 0.25) {
    double p = w * w, s = 0.0;
    for (int n = 2; n <= 40; ++n, p *= -w) {
      const double t = p / (n * (n - 1));
      s += t;
      if (Kokkos::fabs(t) <= 0x1p-54 * s) break;
    }
    return s;
  }
  const double z = y / mu;
  return z * math::log(z) - z + 1.0;
}

}  // namespace detail_poisson

/**
 * \ingroup distrib_poisson
 * \brief Univariate Poisson discrete distribution parameterized by mean \f$\mu > 0\f$.
 *
 * Implements the discrete probability mass function \f$\ell(\mu; y) = -\mu D(y/\mu) - \frac{1}{2}\log(2\pi y) - \delta(y)\f$
 * in saddle-point form with Stirling's error, its parameter derivatives to order 6,
 * expected Fisher information, and cumulative distribution functions via regularized incomplete gamma functions.
 * \see distrib_poisson, distrib_core, stirling, incomplete_gamma
 */
struct poisson_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (1).
  static constexpr int n_params = 1;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  // saddle-point form: l = -mu D(y/mu) - log sqrt(2 pi y) - delta(y), D(z) = z log z - z + 1,
  // delta Stirling's error; y log mu - mu - log y! cancels (1e-12 absolute at mu = 1000)
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the log-mass \f$\log p(y; \mu)\f$ using the saddle-point representation.
   * \param y Integer observation point \f$y \ge 0\f$.
   * \return Log-mass value.
   */
  LAPLACE_HD double logpmf(double y) const {
    if (y == 0.0) return -mu;
    return -mu * detail_poisson::deviance_unit(y, mu) - 0.5 * math::log(6.28318530717958647693 * y) -
           math::stirling_error(y);
  }

  /**
   * \ingroup distrib_poisson
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu)\f$ with respect to \f$\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Integer observation point \f$y \ge 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpmf_dk(double y) const {
    sym_tensor<1, K> t;
    if constexpr (K == 1) {
      t[0] = y / mu - 1.0;
    } else {
      double d = y;  // (-1)^(K-1) (K-1)! y / mu^K
      for (int k = 1; k <= K; ++k) d /= mu;
      for (int k = 1; k < K; ++k) d *= -k;
      t[0] = d;
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
   * \ingroup distrib_poisson
   * \brief Evaluates the expected Fisher information \f$I(\mu) = 1/\mu\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = 1.0 / mu;
    return t;
  }
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the \f$K\f$-th derivative of the expected Fisher information with respect to \f$\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    double d = 1.0 / mu;  // (-1)^K K! / mu^(K+1)
    for (int k = 1; k <= K; ++k) d *= -k / mu;
    expected_info_tensor<1, K> t;
    t.pair[0][0] = d;
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

  // F(y) = P(Y <= y) for any real y (0 below 0)
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the cumulative distribution function \f$F(y) = \mathbb{P}(Y \le y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$F(y)\f$, equal to 0 for \f$y < 0\f$ and \f$Q(\lfloor y \rfloor + 1, \mu)\f$ for \f$y \ge 0\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    return y < 0.0 ? 0.0 : math::gamma_q(Kokkos::floor(y) + 1.0, mu);
  }
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \mathbb{P}(Y > y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$1 - F(y)\f$, equal to 1 for \f$y < 0\f$ and \f$P(\lfloor y \rfloor + 1, \mu)\f$ for \f$y \ge 0\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    return y < 0.0 ? 1.0 : math::gamma_p(Kokkos::floor(y) + 1.0, mu);
  }
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y < 0.0 ? -distrib_infinity : math::log_gamma_q(Kokkos::floor(y) + 1.0, mu);
  }
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    return y < 0.0 ? 0.0 : math::log_gamma_p(Kokkos::floor(y) + 1.0, mu);
  }

  // d^K log F / dmu^K and d^K log(1 - F) / dmu^K at y (floored); below 0, log F = -inf
  // (NaN derivatives) and log(1 - F) = 0
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ with respect to \f$\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_cdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y < 0.0) {
      t[0] = math::detail::quiet_nan();
      return t;
    }
    double v[K + 1];
    detail_gamma_cdf::mixed_order<K>(false, Kokkos::floor(y) + 1.0, mu, v);
    t[0] = v[K];
    return t;
  }
  /**
   * \ingroup distrib_poisson
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ with respect to \f$\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_ccdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y < 0.0) return t;
    double v[K + 1];
    detail_gamma_cdf::mixed_order<K>(true, Kokkos::floor(y) + 1.0, mu, v);
    t[0] = v[K];
    return t;
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of discrete support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of discrete support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sqrt{\mu}\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu); }
  /// Non-smooth points in parameter \f$\mu\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
};

}  // namespace laplace::distrib
