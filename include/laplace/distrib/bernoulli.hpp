// bernoulli_distrib (laplace::distrib): the Bernoulli distribution in its mean
// mu in (0, 1), on y = 0, 1.
//
//   l(mu; y) = y log mu + (1 - y) log(1 - mu)
//
// Derivatives of order K in mu, for K >= 1:
//   d^K l / dmu^K = y (-1)^(K-1) (K-1)! / mu^K - (1 - y) (K-1)! / (1 - mu)^K.
// Only the term of the observed y is formed. The second term has the sign
// -(K-1)! at every order: d/dmu log(1 - mu) = -1/(1 - mu) and differentiating again
// keeps the sign negative.
// Expected information I(mu) = 1/mu + 1/(1 - mu) = 1 / (mu (1 - mu)), its K-th derivative
// (-1)^K K! / mu^(K+1) + K! / (1 - mu)^(K+1).
// F(y) = 0 for y < 0, 1 - mu for 0 <= y < 1, and 1 for y >= 1; log F = log(1 - mu) and
// log(1 - F) = log mu on [0, 1), with the derivatives -(K-1)! / (1 - mu)^K and
// (-1)^(K-1) (K-1)! / mu^K.
// The quantile is 0 at p <= 1 - mu and 1 above.
//
// The mass is closed in every quantity, so no member is left to the generic routes
// (generic_cdf.hpp, generic_quantile.hpp, generic_info.hpp). log1p is used for log(1 - mu),
// so the value and the derivatives stay accurate for mu near 0 and near 1.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_bernoulli {

// (K-1)! / x^K for x > 0, K >= 1, by a loop over the factors (exact for the factorial, K
// roundings for the powers)
LAPLACE_HD_FORCEINLINE double fact_over_pow(int K, double x) {
  double d = 1.0;
  for (int k = 1; k < K; ++k) d *= k;
  for (int k = 1; k <= K; ++k) d /= x;
  return d;
}

}  // namespace detail_bernoulli

/**
 * \ingroup distrib_bernoulli
 * \brief Univariate Bernoulli discrete distribution parameterized by its mean \f$\mu \in (0, 1)\f$.
 *
 * Implements the probability mass function \f$p(y; \mu) = \mu^y (1 - \mu)^{1 - y}\f$ on \f$y \in \{0, 1\}\f$,
 * its parameter derivatives of the log-mass to order 6, the expected Fisher information
 * \f$I(\mu) = 1 / (\mu (1 - \mu))\f$, and the cumulative functions, all in closed form.
 * \see distrib_bernoulli, distrib_core, poisson_distrib
 */
struct bernoulli_distrib {
  /// Mean parameter \f$\mu \in (0, 1)\f$.
  double mu;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (1).
  static constexpr int n_params = 1;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {
      {"mu", 0.0, 1.0, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates the log-mass \f$\log p(y; \mu)\f$.
   *
   * Returns \f$\log \mu\f$ at \f$y = 1\f$, \f$\log(1 - \mu)\f$ at \f$y = 0\f$, and \f$-\infty\f$ for any other \f$y\f$.
   * \param y Observation point, \f$y \in \{0, 1\}\f$.
   * \return Log-mass value.
   * \see distrib_bernoulli, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    if (y == 1.0) return math::log(mu);
    if (y == 0.0) return Kokkos::log1p(-mu);
    return -distrib_infinity;
  }

  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu)\f$ with respect to \f$\mu\f$.
   *
   * Forms only the term of the observed \f$y\f$: \f$(-1)^{K-1} (K-1)! / \mu^K\f$ at \f$y = 1\f$ and
   * \f$-(K-1)! / (1 - \mu)^K\f$ at \f$y = 0\f$. Fills quiet NaN for any other \f$y\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point, \f$y \in \{0, 1\}\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpmf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y == 1.0) {
      double d = detail_bernoulli::fact_over_pow(K, mu);
      t[0] = K % 2 == 0 ? -d : d;
    } else if (y == 0.0) {
      t[0] = -detail_bernoulli::fact_over_pow(K, 1.0 - mu);
    } else {
      t[0] = math::detail::quiet_nan();
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
   * \ingroup distrib_bernoulli
   * \brief Evaluates the expected Fisher information \f$I(\mu) = 1 / (\mu (1 - \mu))\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = 1.0 / mu + 1.0 / (1.0 - mu);
    return t;
  }
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates the \f$K\f$-th derivative of the expected Fisher information with respect to \f$\mu\f$.
   *
   * The two terms \f$(-1)^K K! / \mu^{K+1}\f$ and \f$K! / (1 - \mu)^{K+1}\f$ are summed at the end; at
   * \f$\mu = 1/2\f$ the odd orders are exactly 0, since the two terms are equal and opposite.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    double a = 1.0 / mu;          // (-1)^K K! / mu^(K+1)
    double b = 1.0 / (1.0 - mu);  // K! / (1 - mu)^(K+1)
    for (int k = 1; k <= K; ++k) {
      a *= -k / mu;
      b *= k / (1.0 - mu);
    }
    expected_info_tensor<1, K> t;
    t.pair[0][0] = a + b;
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
   * \ingroup distrib_bernoulli
   * \brief Evaluates the cumulative distribution function \f$F(y) = \mathbb{P}(Y \le y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$0\f$ for \f$y < 0\f$, \f$1 - \mu\f$ for \f$0 \le y < 1\f$, and \f$1\f$ for \f$y \ge 1\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    if (y < 0.0) return 0.0;
    return y < 1.0 ? 1.0 - mu : 1.0;
  }
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \mathbb{P}(Y > y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$1\f$ for \f$y < 0\f$, \f$\mu\f$ for \f$0 \le y < 1\f$, and \f$0\f$ for \f$y \ge 1\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    if (y < 0.0) return 1.0;
    return y < 1.0 ? mu : 0.0;
  }
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   *
   * Uses \c log1p, so the value is accurate for \f$\mu\f$ near 0.
   * \param y Observation point \f$y\f$.
   * \return \f$-\infty\f$ for \f$y < 0\f$, \f$\log(1 - \mu)\f$ for \f$0 \le y < 1\f$, and \f$0\f$ for \f$y \ge 1\f$.
   * \see distrib_bernoulli, log_cdf_dk()
   */
  LAPLACE_HD double log_cdf(double y) const {
    if (y < 0.0) return -distrib_infinity;
    return y < 1.0 ? Kokkos::log1p(-mu) : 0.0;
  }
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$0\f$ for \f$y < 0\f$, \f$\log \mu\f$ for \f$0 \le y < 1\f$, and \f$-\infty\f$ for \f$y \ge 1\f$.
   * \see distrib_bernoulli, log_ccdf_dk()
   */
  LAPLACE_HD double log_ccdf(double y) const {
    if (y < 0.0) return 0.0;
    return y < 1.0 ? math::log(mu) : -distrib_infinity;
  }

  // The cdf is constant on each piece, so the derivatives of log F are those of log(1 - mu) on
  // [0, 1) and 0 above 1; below 0 log F = -inf and the derivatives are NaN. For log(1 - F) the
  // roles are swapped: 0 with zero derivatives below 0, NaN above 1.
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ with respect to \f$\mu\f$.
   *
   * Equal to \f$-(K-1)! / (1 - \mu)^K\f$ on \f$0 \le y < 1\f$, 0 for \f$y \ge 1\f$, and quiet NaN for \f$y < 0\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   * \see distrib_bernoulli, log_cdf()
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_cdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y < 0.0) {
      t[0] = math::detail::quiet_nan();
    } else if (y < 1.0) {
      t[0] = -detail_bernoulli::fact_over_pow(K, 1.0 - mu);
    }
    return t;
  }
  /**
   * \ingroup distrib_bernoulli
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ with respect to \f$\mu\f$.
   *
   * Equal to \f$(-1)^{K-1} (K-1)! / \mu^K\f$ on \f$0 \le y < 1\f$, 0 for \f$y < 0\f$, and quiet NaN for \f$y \ge 1\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   * \see distrib_bernoulli, log_ccdf()
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_ccdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y >= 1.0) {
      t[0] = math::detail::quiet_nan();
    } else if (y >= 0.0) {
      double d = detail_bernoulli::fact_over_pow(K, mu);
      t[0] = K % 2 == 0 ? -d : d;
    }
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

  /**
   * \ingroup distrib_bernoulli
   * \brief The quantile \f$F^{-1}(p)\f$: 0 when \f$p \le 1 - \mu\f$, 1 otherwise.
   * \param p Probability in \f$(0, 1)\f$.
   * \return The smallest \f$y\f$ with \f$F(y) \ge p\f$.
   */
  LAPLACE_HD double quantile(double p) const { return p <= 1.0 - mu ? 0.0 : 1.0; }
  /**
   * \ingroup distrib_bernoulli
   * \brief The quantile from the log-probability, \f$F^{-1}(e^{\log p})\f$.
   * \param log_p Logarithm of the probability.
   * \return 0 when \f$\log p \le \log(1 - \mu)\f$, 1 otherwise.
   */
  LAPLACE_HD double quantile_from_log(double log_p) const { return log_p <= Kokkos::log1p(-mu) ? 0.0 : 1.0; }
  /**
   * \ingroup distrib_bernoulli
   * \brief The upper quantile from the log of the complementary probability, \f$F^{-1}(1 - e^{\log q})\f$.
   * \param log_q Logarithm of \f$1 - p\f$.
   * \return 0 when \f$\log q \ge \log \mu\f$ (that is \f$p \le 1 - \mu\f$), 1 otherwise.
   */
  LAPLACE_HD double quantile_from_log_upper(double log_q) const { return log_q >= math::log(mu) ? 0.0 : 1.0; }

  /// \ingroup distrib_bernoulli
  /// \brief Lower bound of discrete support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_bernoulli
  /// \brief Upper bound of discrete support (1.0).
  LAPLACE_HD double support_upper() const { return 1.0; }
  /// \ingroup distrib_bernoulli
  /// \brief Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_bernoulli
  /// \brief Typical spread (\f$\sqrt{\mu (1 - \mu)}\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu * (1.0 - mu)); }
  /// \ingroup distrib_bernoulli
  /// \brief Non-smooth points in parameter \f$\mu\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /**
   * \ingroup distrib_bernoulli
   * \brief Starting value from a sample summary: \f$\mu = \bar{y}\f$ (`s.mean`), the maximum likelihood estimate, kept strictly inside \f$(0, 1)\f$.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c bernoulli_distrib with parameters strictly inside their open domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  // Starting value from a sample (generic_start.hpp): the sample mean, the MLE; strictly inside the domain.
  bernoulli_distrib start_value(const distrib_sample_summary& s) const { return {start_inside(s.mean, params[0])}; }

};

}  // namespace laplace::distrib
