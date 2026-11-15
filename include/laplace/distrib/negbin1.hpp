// negbin1_distrib (laplace::distrib): the negative binomial of mean mu > 0 and dispersion
// theta > 0 with Var(Y) = mu (1 + theta), on y = 0, 1, 2, ...
//
//   p(y) = Gamma(y + r) / (Gamma(r) y!) (1 / (1 + theta))^r (theta / (1 + theta))^y,  r = mu / theta
//
// It is the shared mass of negbin_core.hpp at a shape r = mu / theta and a mean m = mu,
// written in Loader's saddle-point form (the form of poisson.hpp and binomial.hpp), so that no
// term is larger than the result at any count. The partials in (mu, theta) are assembled by
// detail_negbin::negbin1_partials.
//
// The Poisson limit is theta -> 0, where the shape grows without bound, and it decides how the
// partials are taken. The shape itself must not be the inner variable of the composition: with
// r = mu / theta the chain rule's terms are of the size of theta^(1-K) while
// d^K l / d theta^K is of the size of one, a relative loss of theta^(K-1) (measured on a first
// version written that way: order 6 at theta = 1e-6 was noise, 6e23 ulp). Every quantity is
// therefore written as a rational function of (mu, theta) that is regular at theta = 0:
// (y + r) / r = 1 + u with u = y theta / mu, b1 = (y theta + mu) / (1 + theta),
// w1 = (y - mu) / (y theta + mu), and the second deviance
//
//   bd0(r, b2) = mu (1 + kappa(theta)) - y (1 + kappa(u)) + (y - mu) / (1 + theta),
//   kappa(z) = (log1p(z) - z) / z,
//
// which follows from r log(r / b2) = (mu / theta) [log1p(theta) - log1p(u)] and
// log1p(z) = z (1 + kappa(z)), and has no 1/theta left in it. The difference of Stirling's
// errors delta(y + r) - delta(r) is taken as a function of s = theta / mu above r = 10, where
// both arguments are in the range of the asymptotic series. The corner of large mu with small
// theta (statmodels7's worst case, mu = 2242 and theta = 0.028) is in the reference grid.
//
// The cumulative function is the incomplete beta identity F(k) = I_p(r, k + 1) with
// p = 1 / (1 + theta) formed directly. The derivatives of log F and log(1 - F) in
// (mu, theta) come from the generic route by summing the mass (distrib_log_cdf_dk<K>,
// distrib_log_ccdf_dk<K> of generic_cdf.hpp), and the expected information and its
// derivatives from the generic sum over the support (distrib_expected_info<K> of
// generic_info.hpp): no block of this family's information is orthogonal or closed.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/negbin_core.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

/**
 * \ingroup distrib_negbin1
 * \brief Negative binomial distribution with mean \f$\mu > 0\f$ and dispersion \f$\theta > 0\f$,
 * with variance \f$\mu(1 + \theta)\f$.
 *
 * Mass \f$p(y) = \frac{\Gamma(y + r)}{\Gamma(r)\, y!}\left(\frac{1}{1 + \theta}\right)^{r}
 * \left(\frac{\theta}{1 + \theta}\right)^{y}\f$ with shape \f$r = \mu/\theta\f$, on
 * \f$y = 0, 1, 2, \dots\f$, with the Poisson distribution as the limit \f$\theta \to 0\f$.
 * The variance is proportional to the mean, which is what distinguishes this parametrization
 * from the quadratic one. The log-mass comes from the shape-and-mean core at
 * \f$(r, m) = (\mu/\theta, \mu)\f$ and its derivatives from the pair chain rule.
 * \see distrib_negbin1, distrib_core, chain_pair
 */
struct negbin1_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Dispersion parameter \f$\theta \in (0, \infty)\f$; the variance is \f$\mu(1 + \theta)\f$.
  double theta;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Descriptors of \c mu and \c theta.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"theta", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// The shape \f$r = \mu/\theta\f$ of the core parametrization.
  LAPLACE_HD double shape() const { return mu / theta; }
  /// The success probability \f$p = 1/(1 + \theta)\f$ of the incomplete beta identity.
  LAPLACE_HD double success_prob() const { return 1.0 / (1.0 + theta); }

  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates the log-mass \f$\log p(y; \mu, \theta)\f$.
   *
   * Written in Loader's saddle-point form, so that no term is larger than the result at any
   * count, with the deviance taken at the single numerator \f$\Delta = (y - \mu)/(1 + \theta)\f$.
   * \param y Observation point \f$y \ge 0\f$.
   * \return Log-mass value.
   * \see distrib_negbin1, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    double G[1];
    detail_negbin::negbin1_partials<0>(mu, theta, y, G);
    return G[0];
  }

  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu, \theta)\f$.
   *
   * Composes the deviance and Stirling's error with inner maps that are regular at
   * \f$\theta = 0\f$ (never with the shape \f$\mu/\theta\f$ itself); component \f$b\f$ holds
   * \f$\partial^K \log p / \partial\mu^{K-b}\partial\theta^{b}\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \ge 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpmf_dk(double y) const {
    double G[detail_negbin::pair_count(K)];
    detail_negbin::negbin1_partials<K>(mu, theta, y, G);
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

  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates the cumulative distribution function \f$F(y) = I_p(r, \lfloor y \rfloor + 1)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$F(y)\f$, 0 below the support.
   */
  LAPLACE_HD double cdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 0.0 : math::ibeta(shape(), k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$1 - F(y)\f$, 1 below the support.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 1.0 : math::ibetac(shape(), k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates \f$\log F(y)\f$; the derivatives in the parameters come from distrib_log_cdf_dk().
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$, \f$-\infty\f$ below the support.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? -distrib_infinity : math::log_ibeta(shape(), k + 1.0, success_prob());
  }
  /**
   * \ingroup distrib_negbin1
   * \brief Evaluates \f$\log(1 - F(y))\f$; the derivatives come from distrib_log_ccdf_dk().
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$, 0 below the support.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double k = Kokkos::floor(y);
    return k < 0.0 ? 0.0 : math::log_ibetac(shape(), k + 1.0, success_prob());
  }

  /// \ingroup distrib_negbin1
  /// \brief Lower bound of the discrete support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_negbin1
  /// \brief Upper bound of the discrete support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \ingroup distrib_negbin1
  /// \brief Central location (the mean \f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_negbin1
  /// \brief Typical spread \f$\sqrt{\mu(1 + \theta)}\f$.
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu * (1.0 + theta)); }
  /// \ingroup distrib_negbin1
  /// \brief Non-smooth points in the parameters (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
};

}  // namespace laplace::distrib
