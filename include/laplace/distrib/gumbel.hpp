// gumbel_distrib (laplace::distrib): the Gumbel (maximum) distribution in its location mu
// and scale sigma > 0. With z = (y - mu)/sigma,
//
//   l(mu, sigma; y) = -log(sigma) - z - e^(-z),   F(y) = exp(-e^(-z)),   log F = -e^(-z).
//
// mu is the mode of the density, not the mean: the mean is mu + gamma sigma, gamma the
// Euler constant. l, log F and log(1 - F) are C^inf in (mu, sigma) and l is C^inf in y:
// the family has no kinks.
//
// With a derivatives in mu and b in sigma (a + b = K, the position b in the packed tensor),
// the log-density is the sum of three terms, each in closed form:
//   lin  (the term -z):        -(-1)^b b! z sigma^-b   (a = 0),   (-1)^b b! sigma^-(b+1)   (a = 1),   0 (a >= 2)
//   exp  (the term -e^(-z)):   -(-1)^b b! sigma^-K e^(-z) L_b^(a-1)(z)
//   logs (the term -log sigma): (-1)^b (b-1)! sigma^-b  (a = 0, b >= 1),   0 otherwise
// where L_b^(alpha) is the generalized Laguerre polynomial, computed by its forward
// recurrence (no cancellation for the orders used here). The same closed form, with
// the opposite sign and without the other two terms, gives log F:
//   d^a_mu d^b_sigma log F = -(-1)^b b! sigma^-K e^(-z) L_b^(a-1)(z).
// log(1 - F) = -z + H(e^(-z)) for z >= 0, with H(u) = log((1 - e^(-u))/u) = sum_k h_k u^k
// (the -z term carries the closed form above and H its own series in u, so the
// subtraction 1 - u/(e^u - 1) of the z-polynomial route is never formed), and
// log(1 - F) = -sum_n e^(-n e^(-z))/n for z < 0, differentiated in z through the
// polynomials P_j of exp(-w), w = n e^(-z): d^j/dz^j exp(-w) = exp(-w) P_j(w).
// Both routes are carried to (mu, sigma) by detail_location_scale::mixed_dk.
//
// The expected information (the variance of the score (1 - V)/sigma and
// (-1 + z (1 - V))/sigma, V = e^(-z) ~ Exp(1)):
//   I = (1/sigma^2) [[1, gamma - 1], [gamma - 1, (1 - gamma)^2 + pi^2/6]],
// with gamma the Euler constant. The closed forms were derived by hand and checked by
// quadrature in tools/reference/gumbel_laplace2_forms.py.
//
// The quantile is mu - sigma log(-log p), and the value -log p is written as -log1p(p - 1)
// for p above 1/2, so that it keeps its accuracy as p tends to 1.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_gumbel {

// n! as a double (n <= 20 for the orders used here)
LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^n
LAPLACE_HD_FORCEINLINE constexpr double neg_one_pow(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// s^(-n) for n >= 0
LAPLACE_HD_FORCEINLINE double inv_pow(double s, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r /= s;
  return r;
}

// x^n for n >= 0
LAPLACE_HD_FORCEINLINE double pow_int(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}

// gamma - 1 and (1 - gamma)^2 + pi^2/6: the constants of the expected information
// (gamma the Euler constant)
LAPLACE_HD_FORCEINLINE double info_mus() { return 0.57721566490153286061 - 1.0; }
LAPLACE_HD_FORCEINLINE double info_ss() {
  const double g1 = 1.0 - 0.57721566490153286061;
  return g1 * g1 + 1.6449340668482264365;
}

// Generalized Laguerre polynomial L_n^(alpha)(x) by its forward recurrence in n:
// (k + 1) L_(k+1) = (2k + 1 + alpha - x) L_k - (k + alpha) L_(k-1), L_0 = 1, L_1 = 1 + alpha - x.
// alpha may be negative (here alpha = a - 1 with a = 0 gives -1).
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

// The coefficients h_k, k = 1..36, of H(u) = log((1 - e^(-u))/u): h_1 = -1/2 and
// h_(2n) = B_(2n)/(2n (2n)!) with B the Bernoulli numbers (h_k = 0 for odd k > 1). The
// radius of convergence is 2 pi, so the terms are below 1e-31 relative at u = 1 by k = 36 (the weights k^j of the derivatives need the extra terms: with k up to 20 the truncation alone was 3e-11 in the order-6 sum at u = 1);
// the coefficients are the exact rationals rounded to double (checked by SymPy's series in
// tools/reference/gumbel_laplace2_forms.py).
LAPLACE_HD_FORCEINLINE double h_coef(int k) {
  constexpr double h[37] = {0.0, -0.5, 0.041666666666666664, 0.0, -0.00034722222222222224, 0.0, 5.5114638447971785e-06, 0.0, -1.033399470899471e-07, 0.0, 2.08767569878681e-09, 0.0, -4.403491782239578e-11, 0.0, 9.55895466477477e-13, 0.0, -2.1185501852016142e-14, 0.0, 4.770034475709914e-16, 0.0, -1.087434349279031e-17, 0.0, 2.5040921947091955e-19, 0.0, -5.814360285755218e-21, 0.0, 1.3595027075497952e-22, 0.0, -3.1976847953705525e-24, 0.0, 7.559841507792277e-26, 0.0, -1.7952470840225633e-27, 0.0, 4.279919045926073e-29, 0.0, -1.0238874835181417e-30};
  return h[k];
}

// d^a_mu d^b_sigma of the term -z, a + b >= 1
LAPLACE_HD_FORCEINLINE double lin_component(int a, int b, double z, double sigma) {
  if (a == 0) return -neg_one_pow(b) * fact(b) * z * inv_pow(sigma, b);
  if (a == 1) return neg_one_pow(b) * fact(b) * inv_pow(sigma, b + 1);
  return 0.0;
}

// d^a_mu d^b_sigma of the term -e^(-z), e = e^(-z); this is the component of log F
LAPLACE_HD_FORCEINLINE double exp_component(int a, int b, double z, double sigma, double e) {
  return -neg_one_pow(b) * fact(b) * inv_pow(sigma, a + b) * e * laguerre(b, a - 1.0, z);
}

// d^a_mu d^b_sigma of the term -log sigma
LAPLACE_HD_FORCEINLINE double logs_component(int a, int b, double sigma) {
  if (a == 0 && b >= 1) return neg_one_pow(b) * fact(b - 1) * inv_pow(sigma, b);
  return 0.0;
}

// d^a_mu d^b_sigma l, a + b >= 1
LAPLACE_HD_FORCEINLINE double logpdf_component(int a, int b, double z, double sigma) {
  return lin_component(a, b, z, sigma) + exp_component(a, b, z, sigma, Kokkos::exp(-z)) +
         logs_component(a, b, sigma);
}

}  // namespace detail_gumbel

/**
 * \ingroup distrib_gumbel
 * \brief Univariate Gumbel (maximum) distribution in its location \f$\mu\f$ (the mode) and scale \f$\sigma > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu, \sigma; y) = -\log\sigma - z - e^{-z}\f$ with
 * \f$z = (y - \mu)/\sigma\f$, the cumulative distribution function \f$F(y) = \exp(-e^{-z})\f$, its
 * logarithm \f$\log F = -e^{-z}\f$, the parameter derivatives of \f$\ell\f$, \f$\log F\f$ and
 * \f$\log(1 - F)\f$ to order six, and the expected Fisher information in closed form.
 * \see distrib_gumbel, distrib_core
 */
struct gumbel_distrib {
  /// Location parameter \f$\mu \in (-\infty, \infty)\f$ (the mode).
  double mu;
  /// Scale parameter \f$\sigma \in (0, \infty)\f$.
  double sigma;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /**
   * \ingroup distrib_gumbel
   * \brief Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth: no kinks).
   */
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c sigma (both \f$C^\infty\f$ for \f$\log f\f$ and \f$\log F\f$, no support moves).
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma) = -\log\sigma - z - e^{-z}\f$.
   * \param y Observation point \f$y \in (-\infty, \infty)\f$.
   * \return Log-density value.
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    return -math::log(sigma) - z - math::exp(-z);
  }

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \sigma)\f$ in \f$(\mu, \sigma)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    using namespace detail_gumbel;
    const double z = (y - mu) / sigma;
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = logpdf_component(K - b, b, z, sigma);
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

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \sigma)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * The log-density depends on \f$y\f$ and \f$\mu\f$ only through \f$z = (y - \mu)/\sigma\f$, so
   * \f$\partial/\partial y = -\partial/\partial\mu\f$ and the component with \f$a\f$ derivatives in \f$\mu\f$ and
   * \f$b\f$ in \f$\sigma\f$ is \f$(-1)^J\f$ times the \f$(a + J, b)\f$ component of \f$\log f\f$.
   * There are no kinks: the result is exact everywhere.
   *
   * \tparam J Derivative order with respect to observation \f$y\f$ (\f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters (\f$K \ge 0\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    using namespace detail_gumbel;
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      const double z = (y - mu) / sigma;
      for (int b = 0; b <= K; ++b) t[b] = neg_one_pow(J) * logpdf_component(K - b + J, b, z, sigma);
      return t;
    }
  }

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the expected Fisher information matrix in \f$(\mu, \sigma)\f$.
   *
   * \f$I = \sigma^{-2} \begin{pmatrix} 1 & \gamma - 1 \\ \gamma - 1 & (1 - \gamma)^2 + \pi^2/6 \end{pmatrix}\f$,
   * \f$\gamma\f$ the Euler constant. Packed as \c sym_tensor<2, 2>: \f$I_{\mu\mu}\f$ at index 0, \f$I_{\mu\sigma}\f$
   * at index 1, \f$I_{\sigma\sigma}\f$ at index 2.
   * \return Symmetric tensor \c sym_tensor<2, 2> of the expected information.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    using namespace detail_gumbel;
    const double s2 = 1.0 / (sigma * sigma);
    sym_tensor<2, 2> t;
    t[0] = s2;
    t[1] = info_mus() * s2;
    t[2] = info_ss() * s2;
    return t;
  }

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma)\f$.
   *
   * Every entry is a constant times \f$\sigma^{-2}\f$, so its \f$K\f$-th derivative in \f$\sigma\f$ is
   * \f$(-1)^K (K+1)! \sigma^{-(K+2)}\f$ times the constant; the derivatives in \f$\mu\f$ are zero.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    using namespace detail_gumbel;
    const double d = neg_one_pow(K) * fact(K + 1) * inv_pow(sigma, K + 2);
    expected_info_tensor<2, K> t;
    t.pair[0][K] = d;
    t.pair[1][K] = info_mus() * d;
    t.pair[2][K] = info_ss() * d;
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

  // F(y) = exp(-e^(-z)): the cdf and ccdf in their own forms (1 - F through expm1)
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the cumulative distribution function \f$F(y) = \exp(-e^{-z})\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    const double z = (y - mu) / sigma;
    return math::exp(-math::exp(-z));
  }
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    const double z = (y - mu) / sigma;
    return -Kokkos::expm1(-math::exp(-z));
  }
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y) = -e^{-z}\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    const double z = (y - mu) / sigma;
    return -math::exp(-z);
  }
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   *
   * For \f$z \ge 0\f$ (\f$u = e^{-z} \le 1\f$) the value is \f$-z + H(u)\f$ with the series \f$H\f$ of
   * detail_gumbel::h_coef; for \f$z < 0\f$ it is \f$\log(1 - e^{-u})\f$ through \c log1p.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    const double z = (y - mu) / sigma;
    if (z >= 0.0) {
      const double u = math::exp(-z);
      double h = 0.0, uk = 1.0;
      for (int k = 1; k <= 36; ++k) {
        uk *= u;
        h += detail_gumbel::h_coef(k) * uk;
      }
      return -z + h;
    }
    return Kokkos::log1p(-math::exp(-math::exp(-z)));
  }
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates the quantile function \f$F^{-1}(p) = \mu - \sigma\log(-\log p)\f$.
   *
   * For \f$p > 1/2\f$ the inner value \f$-\log p\f$ is written as \f$-\log(1 + (p - 1))\f$ through \c log1p, with
   * \f$p - 1\f$ exact, so that it keeps its accuracy as \f$p\f$ tends to 1.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const {
    const double t = p <= 0.5 ? -math::log(p) : -Kokkos::log1p(p - 1.0);
    return mu - sigma * math::log(t);
  }

  // d^K log F and d^K log(1 - F) in (mu, sigma)
  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \sigma)\f$.
   *
   * The closed form is the exp term of the log-density with the opposite sign (no other terms).
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    using namespace detail_gumbel;
    const double z = (y - mu) / sigma, e = math::exp(-z);
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = exp_component(K - b, b, z, sigma, e);
    return t;
  }

  /**
   * \ingroup distrib_gumbel
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma)\f$.
   *
   * For \f$z \ge 0\f$ the term \f$-z\f$ has its closed form and \f$H(e^{-z})\f$ is carried by
   * detail_location_scale::mixed_dk from its derivatives in \f$z\f$, \f$H^{(j)}(z) = \sum_k h_k (-k)^j e^{-kz}\f$.
   * For \f$z < 0\f$ the series \f$-\sum_n e^{-n e^{-z}}/n\f$ is differentiated term by term through the
   * polynomials \f$P_j\f$ of \f$e^{-w}\f$, \f$w = n e^{-z}\f$, and carried by \c mixed_dk.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    using namespace detail_gumbel;
    const double z = (y - mu) / sigma;
    double L[K + 1];
    for (int j = 0; j <= K; ++j) L[j] = 0.0;
    sym_tensor<2, K> t;
    if (z >= 0.0) {
      const double u = math::exp(-z);
      double uk = 1.0;
      for (int k = 1; k <= 36; ++k) {
        uk *= u;
        const double hk = h_coef(k);
        if (hk == 0.0) continue;
        double kp = 1.0;  // (-k)^j
        for (int j = 1; j <= K; ++j) {
          kp *= -static_cast<double>(k);
          L[j] += hk * kp * uk;
        }
      }
      t = detail_location_scale::mixed_dk<K>(L, z, sigma);
      for (int b = 0; b <= K; ++b) t[b] += lin_component(K - b, b, z, sigma);
      return t;
    }
    // z < 0: -sum_n e^(-n e^(-z))/n, the derivative of e^(-w) in z being e^(-w) P_j(w)
    const double u = math::exp(-z);
    const double first = Kokkos::exp(-u) * pow_int(1.0 + u, K);  // scale of the n = 1 term
    for (int n = 1; n <= 400; ++n) {
      const double w = n * u;
      const double e = math::exp(-w);
      // coefficients c[m] of P_j(w), P_0 = 1, P_(j+1) = w (P_j - P_j')
      double c[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int j = 1; j <= K; ++j) {
        double d[8] = {};
        for (int m = 0; m < j; ++m) d[m + 1] = c[m] - (m + 1) * c[m + 1];
        for (int m = 0; m < 8; ++m) c[m] = d[m];
        double v = 0.0, wp = 1.0;
        for (int m = 0; m <= j; ++m, wp *= w) v += c[m] * wp;
        L[j] -= e * v / n;
      }
      if (n > 1 && e * pow_int(1.0 + w, K) / n <= 0x1p-60 * first) break;
    }
    t = detail_location_scale::mixed_dk<K>(L, z, sigma);
    return t;
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
  /// Central location (\f$\mu\f$, the mode).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }

  /**
   * \ingroup distrib_gumbel
   * \brief Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$: none for the Gumbel.
   * \param j Parameter index (0 for \c mu, 1 for \c sigma).
   * \param y Observation value (unused).
   * \return Kinks descriptor with \c count = 0.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    (void)j;
    (void)y;
    return distrib_kinks{};
  }
  /**
   * \ingroup distrib_gumbel
   * \brief Non-smooth points in observation \f$y\f$: none for the Gumbel.
   * \return Kinks descriptor with \c count = 0.
   */
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
};

}  // namespace laplace::distrib
