// negbin2_distrib (laplace::distrib): the negative binomial of mean mu > 0 and dispersion
// theta > 0 with Var(Y) = mu + mu^2 / theta, on y = 0, 1, 2, ...
//
//   p(y) = Gamma(y + theta) / (Gamma(theta) y!) (theta / (theta + mu))^theta (mu / (theta + mu))^y
//
// It is the shared mass of negbin_core.hpp at a shape r = theta and a mean m = mu, written in
// Loader's saddle-point form (the form of poisson.hpp and binomial.hpp), so that no term is
// larger than the result at any count; the Poisson limit is theta -> infinity. The partials in
// (mu, theta) are assembled by detail_negbin::negbin2_partials, which composes the deviance
// D(1 + w) and Stirling's error with the rational inner maps of this parametrization,
// b1 = mu (y + theta) / (theta + mu), b2 = theta (y + theta) / (theta + mu) and the one
// numerator Delta = theta (y - mu) / (theta + mu).
//
// The cumulative function is the classical incomplete beta identity F(k) = I_p(theta, k + 1),
// p = theta / (theta + mu), with 1 - F(k) from the complement routine so that the small side is
// never formed as a difference.
//
// The derivatives of log F and log(1 - F) in (mu, theta) are not written here: they come from
// the generic route by summing the mass (distrib_log_cdf_dk<K>, distrib_log_ccdf_dk<K> of
// generic_cdf.hpp). The expected information and its derivatives likewise come from the generic
// sum over the support (distrib_expected_info<K> of generic_info.hpp), which is exact in
// I_mu_mu = 1 / (mu + mu^2 / theta) and in I_mu_theta = 0 to the accuracy of the sum.
// start_value inverts the moments, theta = mu^2 / (s^2 - mu) with the denominator floored at
// mu / 100, since an underdispersed sample puts the estimate past this family's Poisson limit.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/negbin_core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_negbin2
 * \brief Negative binomial distribution with mean \f$\mu > 0\f$ and dispersion \f$\theta > 0\f$,
 * with variance \f$\mu + \mu^2/\theta\f$.
 *
 * Mass \f$p(y) = \frac{\Gamma(y + \theta)}{\Gamma(\theta)\, y!}
 * \left(\frac{\theta}{\theta + \mu}\right)^{\theta}\left(\frac{\mu}{\theta + \mu}\right)^{y}\f$
 * on \f$y = 0, 1, 2, \dots\f$, with the Poisson distribution as the limit \f$\theta \to \infty\f$.
 * The log-mass and its derivatives to order 6 come from the shape-and-mean core at
 * \f$(r, m) = (\theta, \mu)\f$; the cumulative function is closed through the regularized
 * incomplete beta function.
 * \see distrib_negbin2, distrib_core
 */
struct negbin2_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Dispersion parameter \f$\theta \in (0, \infty)\f$; the variance is \f$\mu + \mu^2/\theta\f$.
  double theta;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Descriptors of \c mu and \c theta.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"theta", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates the log-mass \f$\log p(y; \mu, \theta)\f$.
   *
   * Written in Loader's saddle-point form, so that no term is larger than the result at any
   * count, with the two deviances taken at the single numerator
   * \f$\Delta = \theta(y - \mu)/(\theta + \mu)\f$.
   * \param y Observation point \f$y \ge 0\f$.
   * \return Log-mass value.
   * \see distrib_negbin2, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    double G[1];
    detail_negbin::negbin2_partials<0>(mu, theta, y, G);
    return G[0];
  }

  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu, \theta)\f$.
   *
   * Component \f$b\f$ of the packed tensor holds \f$\partial^K \log p / \partial\mu^{K-b}
   * \partial\theta^{b}\f$, assembled from the deviance and Stirling's error composed with the
   * rational inner maps of this parametrization.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \ge 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpmf_dk(double y) const {
    double G[detail_negbin::pair_count(K)];
    detail_negbin::negbin2_partials<K>(mu, theta, y, G);
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = G[detail_negbin::pair_index(K - b, b)];
    return t;
  }
  /**
   * \name Derivatives of the log-mass
   * Parameter derivatives of \f$\log p(y; \mu, \theta)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpmf_d1(double y) const { return logpmf_dk<1>(y); }
  /// Second-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpmf_d2(double y) const { return logpmf_dk<2>(y); }
  /// Third-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpmf_d3(double y) const { return logpmf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpmf_d4(double y) const { return logpmf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpmf_d5(double y) const { return logpmf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log p\f$ in \f$(\mu, \theta)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpmf_d6(double y) const { return logpmf_dk<6>(y); }
  /** @} */

  /// The success probability \f$p = \theta/(\theta + \mu)\f$ of the incomplete beta identity.
  LAPLACE_HD double success_prob() const { return theta / (theta + mu); }

  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates the cumulative distribution function \f$F(y) = I_p(\theta, \lfloor y \rfloor + 1)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$F(y)\f$, 0 below the support.
   */
  LAPLACE_HD double cdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 0.0 : math::ibeta(theta, k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$1 - F(y)\f$, 1 below the support.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 1.0 : math::ibetac(theta, k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates \f$\log F(y)\f$; the derivatives in the parameters come from distrib_log_cdf_dk().
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$, \f$-\infty\f$ below the support.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? -distrib_infinity : math::log_ibeta(theta, k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin2
   * \brief Evaluates \f$\log(1 - F(y))\f$; the derivatives come from distrib_log_ccdf_dk().
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$, 0 below the support.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 0.0 : math::log_ibetac(theta, k + 1.0, success_prob());
  }

  /// \ingroup distrib_negbin2
  /// \brief Lower bound of the discrete support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_negbin2
  /// \brief Upper bound of the discrete support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \ingroup distrib_negbin2
  /// \brief Central location (the mean \f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_negbin2
  /// \brief Typical spread \f$\sqrt{\mu + \mu^2/\theta}\f$.
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu + mu * mu / theta); }
  /// \ingroup distrib_negbin2
  /// \brief Non-smooth points in the parameters (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }

  /**
   * \ingroup distrib_negbin2
   * \brief Starting value from a sample summary: \f$\mu = \bar{y}\f$ (`s.mean`, the mean's moment
   * estimate and its maximum likelihood estimate) and \f$\theta = \mu^2/(s^2 - \mu)\f$
   * (`s.variance`), the moment estimate inverted from \f$\mathrm{Var} = \mu + \mu^2/\theta\f$.
   *
   * The denominator is floored at \f$\mu/100\f$, so a sample with no overdispersion
   * (\f$s^2 \le \mu\f$, where the estimate is negative or infinite) starts at
   * \f$\theta = 100\mu\f$, a variance of \f$1.01\mu\f$, instead of at an end of the domain.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c negbin2_distrib with both parameters strictly inside their open domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  negbin2_distrib start_value(const distrib_sample_summary& s) const {
    const double m = start_inside(s.mean, params[0]);
    const double floor_d = 0.01 * m, d = s.variance - m;
    return {m, start_inside(m * m / (d > floor_d ? d : floor_d), params[1])};
  }
};

}  // namespace laplace::distrib
