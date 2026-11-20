// The tails that make mcp_distrib and scad_distrib proper (laplace::distrib):
// gaussian_tail, student_tail and logistic_tail. Both penalties are flat beyond their
// last knot, which makes exp(-penalty) unnormalizable for every parameter; a tail
// replaces the flat part by an exponent that grows, and leaves the penalty untouched
// on the inner pieces. The tail is a template parameter of the two laws, whose
// default is student_tail.
//
// A tail is an exponent in x = u - k, the distance beyond the law's last knot:
//
//   H(x; theta) = kappa(theta) Psi(x / varsigma(theta)),   x >= 0,
//
// with H(0) = H'(0) = 0, so that q and q' stay continuous at the knot, and with
// varsigma a MONOMIAL in the tail's own parameters, varsigma = prod_j theta_j^(e_j):
//
//   gaussian   Psi(z) = z^2/2          varsigma = V^(1/2)      kappa = 1
//   student    Psi(z) = log(1 + z^2)   varsigma = s nu^(1/2)   kappa = (nu + 1)/2
//   logistic   Psi(z) = 2 log cosh z   varsigma = s            kappa = 1
//
// What the three differ in, beyond the weight of the tail, is the score q' far from
// the location. The penalties exist because q' = 0 beyond the knot leaves large
// deviations unpenalized; the Gaussian tail loses that, since q' = x/V grows without
// bound and beyond x = lambda V the law shrinks more than a rate-lambda Laplace. The
// logistic tail has q' -> 2/s, bounded, so with 2/s <= lambda it never shrinks more
// than that Laplace, and the Student tail has q' -> 0, which is the original
// behaviour up to O(1/x). Measured at s = 1, nu = 3: q'(10^4) is 10^4 for the
// Gaussian, 2 for the logistic and 4.0e-4 for the Student.
//
// Mixed partials. Two identities carry every component the laws need, both derived and
// checked against SymPy in tools/reference/properized_tail_forms.py.
//
// (1) Through the knot. Both laws have k = p lambda (p = gamma for MCP, a for SCAD),
//     so for any F of x = u - p lambda
//
//       d^j_u d^i_lambda d^m_p F(x) = sum_r c_r(i, m) F^(j + r)(x),
//       c_r(i, m) = C(i, m + i - r) (-1)^m m!/(r - i)! lambda^(r - i) (-p)^(r - m),
//
//     over max(i, m) <= r <= i + m; the derivatives in u only shift the order, the
//     coefficients being free of u. This is x_chain_coef below.
//
// (2) In the tail's own parameters. With D = varsigma d/dvarsigma and E = z d/dz,
//     every theta_j enters through varsigma alone, so theta_j d/dtheta_j = e_j D, and
//     the Stirling numbers of the first kind turn d^n into (theta d/dtheta)^k:
//
//       d^n1_theta1 d^n2_theta2 G = theta1^(-n1) theta2^(-n2)
//           sum_(k1,k2) s1(n1,k1) s1(n2,k2) e1^k1 e2^k2 D^(k1+k2) G,
//       D^m [varsigma^(-r) Psi^(r)(z)] = (-1)^m varsigma^(-r)
//           sum_q C(m,q) r^(m-q) sum_j S2(q,j) z^j Psi^(r+j)(z),
//
//     the second line because D acts on a function of z as -E and on varsigma^(-r) as
//     multiplication by -r, with E^q f = sum_j S2(q,j) z^j f^(j). So one component is
//     a finite sum of values of Psi with integer coefficients and powers of z, r, e_j
//     and 1/theta_j: no Bell polynomials and nothing on the stack beyond a few
//     doubles. This is psi_chain and h_mixed below.
//
// A tail therefore supplies Psi^(k) at a runtime order, its exponents e_j, its kappa
// (affine in one parameter at most), the constant T = int_0^inf exp(-H) with its own
// mixed partials, and the partial integral int_x^inf exp(-H) with its logarithm. The
// highest order of Psi reached is r + j with r <= J + K and j <= K, hence
// tail_max_psi_order.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/polygamma.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_tail {

/// Highest total order of a derivative in the parameters.
constexpr int tail_max_order = 6;
/// Highest order of \f$\Psi\f$ reached, \f$r + j\f$ with \f$r \le J + K\f$ and \f$j \le K\f$.
constexpr int tail_max_psi_order = 20;

// s1(n, k), signed: the coefficient of x^k in x (x - 1) ... (x - n + 1), built from
// p_i(x) = p_(i-1)(x) (x - (i - 1)) downward in the degree, so that each step reads
// only coefficients it has not yet written
LAPLACE_HD double stirling1(int n, int k) {
  double c[tail_max_order + 1] = {1.0};
  for (int i = 1; i <= n; ++i)
    for (int q = i; q >= 0; --q) c[q] = (q > 0 ? c[q - 1] : 0.0) - double(i - 1) * c[q];
  return c[k];
}

// S2(n, k) = (1/k!) sum_t (-1)^t C(k, t) (k - t)^n, from E^n f = sum_k S2(n,k) z^k f^(k)
LAPLACE_HD double stirling2(int n, int k) {
  if (k == 0) return n == 0 ? 1.0 : 0.0;
  double kf = 1.0;
  for (int t = 2; t <= k; ++t) kf *= double(t);
  double s = 0.0;
  for (int t = 0; t <= k; ++t) {
    double p = 1.0;
    for (int w = 0; w < n; ++w) p *= double(k - t);
    s += ((t % 2 == 0) ? 1.0 : -1.0) * double(binomial_coef(k, t)) * p;
  }
  return s / kf;
}

// x^n for a small integer n, with 0^0 = 1
LAPLACE_HD_FORCEINLINE double ipow(double x, int n) {
  double p = 1.0;
  for (int i = 0; i < n; ++i) p *= x;
  return p;
}

/// \brief Identity (1): the coefficient of \f$F^{(j+r)}\f$ in
/// \f$\partial_u^j\partial_\lambda^i\partial_p^m F(u - p\lambda)\f$.
LAPLACE_HD double x_chain_coef(int r, int i, int m, double lambda, double p) {
  const int lo = i > m ? i : m;
  if (r < lo || r > i + m) return 0.0;
  double c = double(binomial_coef(i, m + i - r)) * ((m % 2 == 0) ? 1.0 : -1.0);
  for (int t = 2; t <= m; ++t) c *= double(t);          // m!
  for (int t = 2; t <= r - i; ++t) c /= double(t);      // 1/(r - i)!
  return c * ipow(lambda, r - i) * ipow(-p, r - m);
}

/// \brief Identity (2) for \f$\Psi\f$ alone:
/// \f$\partial_x^r\partial_{\theta_1}^{n_1}\partial_{\theta_2}^{n_2}\Psi(x/\varsigma)\f$.
template <class Tail>
LAPLACE_HD double psi_chain(const Tail& tail, int r, int n1, int n2, double x) {
  const double vs = tail.varsigma();
  const double z = x / vs;
  const int nmax = n1 + n2;
  double a[tail_max_order + 1];  // a[j] multiplies Psi^(r + j)(z) z^j
  for (int j = 0; j <= nmax; ++j) a[j] = 0.0;
  for (int k1 = 0; k1 <= n1; ++k1) {
    const double c1 = stirling1(n1, k1) * ipow(tail.exponent(0), k1);
    if (c1 == 0.0) continue;
    for (int k2 = 0; k2 <= n2; ++k2) {
      const double c2 = stirling1(n2, k2) * ipow(tail.exponent(1), k2);
      if (c2 == 0.0) continue;
      const int m = k1 + k2;
      const double sgn = (m % 2 == 0) ? c1 * c2 : -c1 * c2;
      for (int q = 0; q <= m; ++q) {
        const double w = sgn * double(binomial_coef(m, q)) * ipow(double(r), m - q);
        for (int j = 0; j <= q; ++j) {
          const double s2 = stirling2(q, j);
          if (s2 != 0.0) a[j] += w * s2;
        }
      }
    }
  }
  double s = 0.0, zj = 1.0;
  for (int j = 0; j <= nmax; ++j) {
    s += a[j] * zj * tail.psi_deriv(r + j, z);
    zj *= z;
  }
  return s / (ipow(tail.param(0), n1) * ipow(tail.param(1), n2) * ipow(vs, r));
}

/// \brief One component \f$\partial_x^r\partial_{\theta_1}^{n_1}\partial_{\theta_2}^{n_2}H\f$
/// of the tail exponent \f$H = \kappa\,\Psi(x/\varsigma)\f$.
///
/// \f$\kappa\f$ is affine in the second parameter at most, so the Leibniz rule over the
/// product has two terms.
template <class Tail>
LAPLACE_HD double h_mixed(const Tail& tail, int r, int n1, int n2, double x) {
  double s = tail.kappa(0) * psi_chain(tail, r, n1, n2, x);
  if (n2 >= 1) {
    const double k1 = tail.kappa(1);
    if (k1 != 0.0) s += double(n2) * k1 * psi_chain(tail, r, n1, n2 - 1, x);
  }
  return s;
}

// G = exp(g) from the derivatives of g, the scalar exponential recursion
LAPLACE_HD void exp_from_log(int K, const double* g, double* G) {
  G[0] = math::exp(g[0]);
  for (int k = 1; k <= K; ++k) {
    double s = 0.0;
    for (int b = 0; b < k; ++b) s += double(binomial_coef(k - 1, b)) * G[b] * g[k - b];
    G[k] = s;
  }
}

}  // namespace detail_tail

/**
 * \ingroup distrib_properizing_tail
 * \brief Gaussian properizing tail, \f$H = x^2/(2V)\f$, in the variance
 * \f$V = \f$ \c tail_sigma2 \f$> 0\f$.
 *
 * The lightest of the three. \f$q'(x) = x/V\f$ grows without bound, so beyond
 * \f$x = \lambda V\f$ the law shrinks more than a rate-\f$\lambda\f$ Laplace.
 * \see distrib_properizing_tail, student_tail, logistic_tail
 */
struct gaussian_tail {
  /// Variance \f$V \in (0, \infty)\f$ of the tail.
  double tail_sigma2;

  /// Number of parameters the tail adds (1).
  static constexpr int n_tail_params = 1;
  /// Descriptor of the tail's parameter.
  static constexpr distrib_param_info tail_params[1] = {
      {"tail_sigma2", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// \f$\varsigma = \sqrt{V}\f$.
  LAPLACE_HD double varsigma() const { return Kokkos::sqrt(tail_sigma2); }
  /// Exponent of parameter \f$j\f$ in \f$\varsigma\f$ (\f$1/2\f$ for \f$V\f$, 1 beyond).
  LAPLACE_HD double exponent(int j) const { return j == 0 ? 0.5 : 1.0; }
  /// Parameter \f$j\f$ (1 beyond the tail's own).
  LAPLACE_HD double param(int j) const { return j == 0 ? tail_sigma2 : 1.0; }
  /// \f$\partial^b\kappa\f$; \f$\kappa = 1\f$ here.
  LAPLACE_HD double kappa(int b) const { return b == 0 ? 1.0 : 0.0; }
  /// \f$\Psi^{(k)}(z)\f$ of \f$\Psi(z) = z^2/2\f$.
  LAPLACE_HD double psi_deriv(int k, double z) const {
    return k == 0 ? 0.5 * z * z : (k == 1 ? z : (k == 2 ? 1.0 : 0.0));
  }

  /// \f$\partial^{n_1}_V T\f$ of \f$T = \sqrt{\pi V/2}\f$.
  LAPLACE_HD double const_mixed(int n1, int n2) const {
    if (n2 > 0) return 0.0;
    double c = 1.25331413731550025121;  // sqrt(pi / 2)
    for (int t = 0; t < n1; ++t) c *= 0.5 - double(t);
    double v = Kokkos::sqrt(tail_sigma2);
    for (int t = 0; t < n1; ++t) v /= tail_sigma2;
    return c * v;
  }
  /// \f$T = \int_0^\infty e^{-H} = \sqrt{\pi V/2}\f$.
  LAPLACE_HD double tail_const() const { return 1.25331413731550025121 * Kokkos::sqrt(tail_sigma2); }
  /// \f$\log T\f$.
  LAPLACE_HD double log_tail_const() const {
    return 0.22579135264472743236 + 0.5 * math::log(tail_sigma2);
  }
  /// \f$\int_x^\infty e^{-H} = 2\sqrt{\pi V/2}\,\Phi(-x/\sqrt{V})\f$.
  LAPLACE_HD double upper(double x) const {
    return 2.0 * tail_const() * math::Phi(-x / Kokkos::sqrt(tail_sigma2));
  }
  /// \f$\log\int_x^\infty e^{-H}\f$, kept in logarithms for a deep tail.
  LAPLACE_HD double log_upper(double x) const {
    return 0.69314718055994530942 + log_tail_const() + math::log_Phi(-x / Kokkos::sqrt(tail_sigma2));
  }
  /// Rough spread of the tail, for the quadratures.
  LAPLACE_HD double scale_hint() const { return Kokkos::sqrt(tail_sigma2); }

  /**
   * \brief Stationary points \f$z > 0\f$ of \f$t H'(z) + z - w\f$, for the prox.
   *
   * \f$H' = z/V\f$ is linear, so there is the single root \f$wV/(V + t)\f$, present
   * for \f$w > 0\f$. The shrinkage \f$w - z = wt/(V + t)\f$ grows with \f$w\f$.
   * \param w Distance \f$|v - \mu| - K\f$ of the point beyond the law's last knot.
   * \param t Step \f$t > 0\f$.
   * \param z Buffer of at least four doubles, written with the roots.
   * \return Number of roots written (0 or 1).
   */
  LAPLACE_HD int prox_roots(double w, double t, double* z) const {
    if (!(w > 0.0)) return 0;
    z[0] = w * tail_sigma2 / (tail_sigma2 + t);
    return z[0] > 0.0 ? 1 : 0;
  }

  /// Starting value from a sample summary.
  gaussian_tail start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.variance, tail_params[0])};
  }
};

/**
 * \ingroup distrib_properizing_tail
 * \brief Student properizing tail,
 * \f$H = \frac{\nu+1}{2}\log\!\big(1 + \frac{x^2}{\nu s^2}\big)\f$, in the scale
 * \f$s > 0\f$ and the degrees of freedom \f$\nu > 0\f$.
 *
 * The only one of the three whose score returns to zero, \f$q'(x) \to 0\f$, so the
 * penalty keeps the behaviour it was built for up to \f$O(1/x)\f$. The Gaussian tail is
 * its limit as \f$\nu \to \infty\f$ at fixed \f$s\f$, with \f$T \to \sqrt{\pi s^2/2}\f$
 * and the exponent \f$\to x^2/(2s^2)\f$, both to within \f$O(1/\nu)\f$.
 * \see distrib_properizing_tail, gaussian_tail, logistic_tail
 */
struct student_tail {
  /// Scale \f$s \in (0, \infty)\f$ of the tail.
  double tail_scale;
  /// Degrees of freedom \f$\nu \in (0, \infty)\f$; the tail decays as \f$x^{-(\nu+1)}\f$.
  double tail_df;

  /// Number of parameters the tail adds (2).
  static constexpr int n_tail_params = 2;
  /// Descriptors of the tail's two parameters.
  static constexpr distrib_param_info tail_params[2] = {
      {"tail_scale", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"tail_df", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// \f$\varsigma = s\sqrt{\nu}\f$.
  LAPLACE_HD double varsigma() const { return tail_scale * Kokkos::sqrt(tail_df); }
  /// Exponent of parameter \f$j\f$ in \f$\varsigma\f$ (1 for \f$s\f$, \f$1/2\f$ for \f$\nu\f$).
  LAPLACE_HD double exponent(int j) const { return j == 0 ? 1.0 : 0.5; }
  /// Parameter \f$j\f$.
  LAPLACE_HD double param(int j) const { return j == 0 ? tail_scale : tail_df; }
  /// \f$\partial^b\kappa\f$ of \f$\kappa = (\nu + 1)/2\f$, affine in \f$\nu\f$.
  LAPLACE_HD double kappa(int b) const {
    return b == 0 ? 0.5 * (tail_df + 1.0) : (b == 1 ? 0.5 : 0.0);
  }
  /**
   * \brief \f$\Psi^{(k)}(z)\f$ of \f$\Psi(z) = \log(1 + z^2)\f$.
   *
   * \f$\Psi^{(k)} = N_k(z)\,\omega^k\f$ with \f$\omega = 1/(1+z^2)\f$, \f$N_1 = 2z\f$ and
   * \f$N_{k+1} = N_k'(1+z^2) - 2kzN_k\f$, a polynomial of degree \f$k\f$. Each term is
   * assembled as \f$c_j\,(z\omega)^j\,\omega^{k-j}\f$, and \f$z\omega \le 1/2\f$ and
   * \f$\omega \le 1\f$ for every \f$z\f$, so nothing overflows at any order. The closed
   * form \f$(-1)^{k-1}(k-1)!\,2\cos(k(\pi/2 - \arctan z))/(1+z^2)^{k/2}\f$ was measured
   * against this one and rejected: it returns \f$(k-1)!\f$ times the rounding of the
   * angle where the odd orders vanish at \f$z = 0\f$, which the polynomial gives exactly.
   */
  LAPLACE_HD double psi_deriv(int k, double z) const {
    if (k == 0) return Kokkos::log1p(z * z);
    const double om = 1.0 / (1.0 + z * z);
    double c[detail_tail::tail_max_psi_order + 2], d[detail_tail::tail_max_psi_order + 2];
    for (int j = 0; j <= k + 1; ++j) c[j] = 0.0;
    c[1] = 2.0;  // N_1 = 2 z
    for (int i = 1; i < k; ++i) {
      for (int j = 0; j <= k + 1; ++j) d[j] = 0.0;
      for (int j = 0; j <= i; ++j) {
        if (c[j] == 0.0) continue;
        if (j >= 1) d[j - 1] += double(j) * c[j];
        d[j + 1] += (double(j) - 2.0 * double(i)) * c[j];
      }
      for (int j = 0; j <= k + 1; ++j) c[j] = d[j];
    }
    const double zw = z * om;
    double s = 0.0, zwj = 1.0;
    for (int j = 0; j <= k; ++j) {
      if (c[j] != 0.0) {
        double w = zwj;
        for (int t = 0; t < k - j; ++t) w *= om;
        s += c[j] * w;
      }
      zwj *= zw;
    }
    return s;
  }

  // log G, G = sqrt(pi nu) Gamma(nu/2) / (2 Gamma((nu+1)/2)), and its derivatives in nu:
  // elementary plus one polygamma difference per order
  LAPLACE_HD void log_g_derivs(int K, double* g) const {
    const double nu = tail_df;
    g[0] = 0.5 * math::log(3.14159265358979323846 * nu) + math::lgamma(0.5 * nu) -
           math::lgamma(0.5 * (nu + 1.0)) - 0.69314718055994530942;
    double f = 1.0, p = nu, h = 0.5;
    for (int k = 1; k <= K; ++k) {
      g[k] = 0.5 * ((k % 2 == 1) ? 1.0 : -1.0) * f / p +
             h * (math::polygamma(k - 1, 0.5 * nu) - math::polygamma(k - 1, 0.5 * (nu + 1.0)));
      f *= double(k);
      p *= nu;
      h *= 0.5;
    }
  }

  /// \f$\partial_s^{n_1}\partial_\nu^{n_2}T\f$ of \f$T = s\,G(\nu)\f$.
  LAPLACE_HD double const_mixed(int n1, int n2) const {
    if (n1 > 1) return 0.0;
    double g[detail_tail::tail_max_order + 1], G[detail_tail::tail_max_order + 1];
    log_g_derivs(n2, g);
    detail_tail::exp_from_log(n2, g, G);
    return (n1 == 0 ? tail_scale : 1.0) * G[n2];
  }
  /// \f$T = \int_0^\infty e^{-H} = \frac{s}{2}\sqrt{\pi\nu}\,\Gamma(\nu/2)/\Gamma(\frac{\nu+1}{2})\f$.
  LAPLACE_HD double tail_const() const { return math::exp(log_tail_const()); }
  /// \f$\log T\f$.
  LAPLACE_HD double log_tail_const() const {
    double g[detail_tail::tail_max_order + 1];
    log_g_derivs(0, g);
    return math::log(tail_scale) + g[0];
  }
  // int_x^inf exp(-H) = T I_w(nu/2, 1/2), w = varsigma^2/(varsigma^2 + x^2), from
  // v = varsigma t and then w = 1/(1 + t^2); I is the regularized incomplete beta. The
  // complement is I_w(nu/2, 1/2) = 1 - I_(1-w)(1/2, nu/2) with 1 - w = x^2/(varsigma^2 +
  // x^2), and WHICHEVER OF THE TWO ARGUMENTS IS SMALL IS FORMED DIRECTLY, never by
  // subtracting the other from one: just beyond the knot 1 - w is of the order
  // x^2/varsigma^2, which the quotient v2/(v2 + x^2) rounds away. Measured on 2026-11-19
  // at the second knot of SCAD with lambda = 3, a = 1.1, s = 0.5 and nu = 40, at
  // x = 3.3e-7: through w alone log F was out by 1.3e5 ulp of max(|log F|, 1), the whole
  // size of the increment of F across that step, and through the small argument it is
  // within one.
  /// \f$\int_x^\infty e^{-H} = T\,I_w(\nu/2, 1/2)\f$.
  LAPLACE_HD double upper(double x) const {
    const double v2 = varsigma() * varsigma(), x2 = x * x;
    if (x2 <= v2) return tail_const() * math::ibetac(0.5, 0.5 * tail_df, x2 / (v2 + x2));
    return tail_const() * math::ibeta(0.5 * tail_df, 0.5, v2 / (v2 + x2));
  }
  /// \f$\log\int_x^\infty e^{-H}\f$, kept in logarithms for a deep tail.
  LAPLACE_HD double log_upper(double x) const {
    const double v2 = varsigma() * varsigma(), x2 = x * x;
    if (x2 <= v2) return log_tail_const() + math::log_ibetac(0.5, 0.5 * tail_df, x2 / (v2 + x2));
    return log_tail_const() + math::log_ibeta(0.5 * tail_df, 0.5, v2 / (v2 + x2));
  }
  /// Rough spread of the tail, for the quadratures.
  LAPLACE_HD double scale_hint() const { return tail_scale; }

  /**
   * \brief Stationary points \f$z > 0\f$ of \f$t H'(z) + z - w\f$, for the prox.
   *
   * \f$H'(z) = 2\kappa z/(\varsigma^2 + z^2)\f$ rises to \f$\kappa/\varsigma\f$ at
   * \f$z = \varsigma\f$ and falls back to zero, so the stationary equation is the cubic
   * \f$z^3 - w z^2 + (\varsigma^2 + 2t\kappa) z - w\varsigma^2 = 0\f$, whose real roots
   * are all positive (Descartes) and number one or three: the objective can have two
   * local minima in the tail, and the caller compares their values. The roots come from
   * the trigonometric form when the discriminant is positive and from Cardano's otherwise,
   * then two Newton steps on \f$t H' + z - w\f$, which is better conditioned near a root
   * than the cubic. The shrinkage \f$w - z \to 2t\kappa/w\f$ vanishes as \f$w\f$ grows.
   * \param w Distance \f$|v - \mu| - K\f$ of the point beyond the law's last knot.
   * \param t Step \f$t > 0\f$.
   * \param z Buffer of at least four doubles, written with the roots.
   * \return Number of roots written (0 to 4).
   */
  LAPLACE_HD int prox_roots(double w, double t, double* z) const {
    if (!(w > 0.0)) return 0;
    const double vs = varsigma(), v2 = vs * vs, k = 0.5 * (tail_df + 1.0);
    const double b = -w, c = v2 + 2.0 * t * k, d = -w * v2;
    // the depressed cubic y^3 + p y + q with z = y + sh
    const double sh = -b / 3.0, p = c - b * b / 3.0;
    const double q = d + b * (2.0 * b * b - 9.0 * c) / 27.0;
    int n = 0;
    if (p < 0.0 && -4.0 * p * p * p - 27.0 * q * q > 0.0) {
      const double m = 2.0 * Kokkos::sqrt(-p / 3.0);
      const double arg = Kokkos::fmax(-1.0, Kokkos::fmin(1.0, 3.0 * q / (p * m)));
      const double th = Kokkos::acos(arg) / 3.0;
      for (int j = 0; j < 3; ++j)
        z[n++] = sh + m * Kokkos::cos(th - 2.09439510239319549231 * double(j));
    } else {
      const double r = Kokkos::sqrt(Kokkos::fmax(0.0, 0.25 * q * q + p * p * p / 27.0));
      z[n++] = sh + Kokkos::cbrt(-0.5 * q + r) + Kokkos::cbrt(-0.5 * q - r);
    }
    // The discriminant is a difference of two terms that both grow like w^6 and cancel
    // to leading order, so its computed sign is not to be trusted for a large w: at
    // w = 1e11 it comes out positive where the cubic has one real root, which sends the
    // branch above to the trigonometric form and fills z with three numbers that are not
    // roots. Whenever the root near w exists it is therefore added here on its own
    // account, whatever the branch returned: beyond z = varsigma the equation is
    // increasing, so Newton from w reaches it. A duplicate of a root already found costs
    // the caller one objective evaluation.
    if (w > vs) {
      double x = w;
      for (int it = 0; it < 60; ++it) {
        const double s = v2 + x * x, f = 2.0 * t * k * x / s + x - w;
        const double fp = 2.0 * t * k * (v2 - x * x) / (s * s) + 1.0;
        if (!(fp > 0.0)) break;
        const double xn = x - f / fp;
        if (!(xn < x) || !(xn > vs)) break;
        x = xn;
      }
      if (x > 0.0) z[n++] = x;
    }
    int out = 0;
    for (int j = 0; j < n; ++j) {
      double x = z[j];
      // Newton to rounding on t H' + z - w, which near a root is better conditioned than
      // the cubic. Two steps are not enough: measured on 2026-11-20, the root the
      // trigonometric form supplies was still 4.4e6 ulp from the converged one, which
      // showed up as a disagreement with the reference where two copies of one root
      // competed. The iteration is stopped when the step no longer moves x.
      for (int it = 0; it < 60 && x > 0.0; ++it) {
        const double s = v2 + x * x, f = 2.0 * t * k * x / s + x - w;
        const double fp = 2.0 * t * k * (v2 - x * x) / (s * s) + 1.0;
        if (fp == 0.0) break;
        const double xn = x - f / fp;
        if (xn == x || !Kokkos::isfinite(xn)) break;
        x = xn;
      }
      if (x > 0.0) z[out++] = x;
    }
    // ascending, since the caller's tie rule keeps the first candidate of equal value
    // and its own candidates below the knot are already in that order
    for (int i = 1; i < out; ++i)
      for (int j = i; j > 0 && z[j] < z[j - 1]; --j) {
        const double tmp = z[j];
        z[j] = z[j - 1], z[j - 1] = tmp;
      }
    return out;
  }

  /// Starting value from a sample summary; \f$\nu = 5\f$, light enough to have a variance.
  student_tail start_value(const distrib_sample_summary& s) const {
    return {start_inside(Kokkos::sqrt(s.variance), tail_params[0]), start_inside(5.0, tail_params[1])};
  }
};

/**
 * \ingroup distrib_properizing_tail
 * \brief Logistic properizing tail, \f$H = 2\log\cosh(x/s)\f$, in the scale \f$s > 0\f$.
 *
 * The cheapest of the three: \f$T = s\f$ and \f$\int_x^\infty e^{-H} = s(1 - \tanh(x/s))\f$,
 * both elementary at every order, which makes it the exact reference the other two are
 * checked against. Its score tends to \f$2/s\f$, so with \f$2/s \le \lambda\f$ the tail
 * never shrinks more than a rate-\f$\lambda\f$ Laplace.
 * \see distrib_properizing_tail, gaussian_tail, student_tail
 */
struct logistic_tail {
  /// Scale \f$s \in (0, \infty)\f$ of the tail.
  double tail_scale;

  /// Number of parameters the tail adds (1).
  static constexpr int n_tail_params = 1;
  /// Descriptor of the tail's parameter.
  static constexpr distrib_param_info tail_params[1] = {
      {"tail_scale", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// \f$\varsigma = s\f$.
  LAPLACE_HD double varsigma() const { return tail_scale; }
  /// Exponent of parameter \f$j\f$ in \f$\varsigma\f$ (1 throughout).
  LAPLACE_HD double exponent(int) const { return 1.0; }
  /// Parameter \f$j\f$ (1 beyond the tail's own).
  LAPLACE_HD double param(int j) const { return j == 0 ? tail_scale : 1.0; }
  /// \f$\partial^b\kappa\f$; \f$\kappa = 1\f$ here.
  LAPLACE_HD double kappa(int b) const { return b == 0 ? 1.0 : 0.0; }
  /**
   * \brief \f$\Psi^{(k)}(z)\f$ of \f$\Psi(z) = 2\log\cosh z\f$.
   *
   * \f$\Psi^{(k)} = 2\,\partial^{k-1}\tanh\f$, written in \f$g = \operatorname{sech}^2 z\f$:
   * \f$\partial^{n}\tanh = R_n(g)\f$ for odd \f$n\f$ and \f$t\,R_n(g)\f$ for even \f$n\f$,
   * with \f$R_0 = 1\f$, \f$R_{n+1} = -2gR_n'\f$ from odd to even and
   * \f$R_{n+1} = gR_n - 2g(1-g)R_n'\f$ from even to odd. Every \f$R_n\f$ with
   * \f$n \ge 1\f$ vanishes at \f$g = 0\f$, so the leading term carries the value in the
   * far tail and nothing cancels there; \f$g\f$ itself is built from
   * \f$4e^{-2|z|}/(1 + e^{-2|z|})^2\f$ rather than from \f$1 - \tanh^2\f$, which would
   * lose every digit for large \f$|z|\f$.
   */
  LAPLACE_HD double psi_deriv(int k, double z) const {
    const double az = Kokkos::fabs(z);
    const double em = math::exp(-2.0 * az);
    if (k == 0) return 2.0 * (az - 0.69314718055994530942 + Kokkos::log1p(em));
    const double g = 4.0 * em / ((1.0 + em) * (1.0 + em));
    const double t = Kokkos::tanh(z);
    const int n = k - 1;
    double r[detail_tail::tail_max_psi_order + 2], w[detail_tail::tail_max_psi_order + 2];
    for (int j = 0; j <= n + 1; ++j) r[j] = 0.0;
    r[0] = 1.0;  // R_0 = 1, with the even form t R_0
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= n + 1; ++j) w[j] = 0.0;
      if (i % 2 == 0) {  // even -> odd: R gR - 2g(1 - g) R'
        for (int j = 0; j <= i; ++j) {
          if (r[j] == 0.0) continue;
          w[j + 1] += r[j];
          if (j >= 1) {
            w[j] -= 2.0 * double(j) * r[j];
            w[j + 1] += 2.0 * double(j) * r[j];
          }
        }
      } else {  // odd -> even: R -2g R'
        for (int j = 1; j <= i + 1; ++j)
          if (r[j] != 0.0) w[j] -= 2.0 * double(j) * r[j];
      }
      for (int j = 0; j <= n + 1; ++j) r[j] = w[j];
    }
    double s = 0.0;
    for (int j = n + 1; j >= 0; --j) s = s * g + r[j];
    return 2.0 * ((n % 2 == 0) ? t * s : s);
  }

  /// \f$\partial^{n_1}_s T\f$ of \f$T = s\f$.
  LAPLACE_HD double const_mixed(int n1, int n2) const {
    if (n2 > 0) return 0.0;
    return n1 == 0 ? tail_scale : (n1 == 1 ? 1.0 : 0.0);
  }
  /// \f$T = \int_0^\infty e^{-H} = s\f$.
  LAPLACE_HD double tail_const() const { return tail_scale; }
  /// \f$\log T\f$.
  LAPLACE_HD double log_tail_const() const { return math::log(tail_scale); }
  // s (1 - tanh(x/s)) = 2 s e^(-2z)/(1 + e^(-2z)), which nothing cancels in
  /// \f$\int_x^\infty e^{-H} = s\,(1 - \tanh(x/s))\f$.
  LAPLACE_HD double upper(double x) const {
    const double em = math::exp(-2.0 * x / tail_scale);
    return 2.0 * tail_scale * em / (1.0 + em);
  }
  /// \f$\log\int_x^\infty e^{-H}\f$, kept in logarithms for a deep tail.
  LAPLACE_HD double log_upper(double x) const {
    const double z = x / tail_scale;
    return 0.69314718055994530942 + math::log(tail_scale) - 2.0 * z -
           Kokkos::log1p(math::exp(-2.0 * z));
  }
  /// Rough spread of the tail, for the quadratures.
  LAPLACE_HD double scale_hint() const { return tail_scale; }

  /**
   * \brief Stationary points \f$z > 0\f$ of \f$t H'(z) + z - w\f$, for the prox.
   *
   * \f$H'(z) = (2/s)\tanh(z/s)\f$ increases to \f$2/s\f$, so \f$t H' + z - w\f$ is
   * increasing and concave on \f$z > 0\f$ and has the single root, present for
   * \f$w > 0\f$. Newton from \f$\max(0, w - 2t/s)\f$, where the function is at most zero,
   * rises monotonically to it and is stopped when the step no longer rises, which is
   * rounding. The shrinkage \f$w - z \to 2t/s\f$, a constant.
   * \param w Distance \f$|v - \mu| - K\f$ of the point beyond the law's last knot.
   * \param t Step \f$t > 0\f$.
   * \param z Buffer of at least four doubles, written with the root.
   * \return Number of roots written (0 or 1).
   */
  LAPLACE_HD int prox_roots(double w, double t, double* z) const {
    if (!(w > 0.0)) return 0;
    const double hmax = 2.0 * t / tail_scale;
    double x = Kokkos::fmax(0.0, w - hmax);
    // ponytail: cap of 100 steps, the monotone test stops it at rounding well before
    for (int it = 0; it < 100; ++it) {
      const double th = Kokkos::tanh(x / tail_scale);
      const double f = hmax * th + x - w;
      const double fp = hmax * (1.0 - th * th) / tail_scale + 1.0;
      const double xn = x - f / fp;
      if (!(xn > x)) break;
      x = xn;
    }
    z[0] = x;
    return x > 0.0 ? 1 : 0;
  }

  /// Starting value from a sample summary; \f$s\f$ from the standard deviation.
  logistic_tail start_value(const distrib_sample_summary& s) const {
    return {start_inside(Kokkos::sqrt(s.variance), tail_params[0])};
  }
};

}  // namespace laplace::distrib
