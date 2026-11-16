// vonmises_distrib (laplace::distrib): the von Mises distribution on the circle in its mean
// direction mu (identity link on the whole real line, periodic of period 2 pi) and its circular
// standard deviation s > 0 (log link),
//
//   rho = A(kappa) = I_1(kappa)/I_0(kappa) = exp(-s^2/2),     s = sqrt(-2 log rho),
//   f(y; mu, s) = exp(kappa cos z) / (2 pi I_0(kappa)),       z = y - mu wrapped to (-pi, pi],
//
// where kappa = kappa(s) is the concentration, obtained from s by vonmises_kappa_from_s
// (vonmises_scale.hpp). s is the sigma of the wrapped normal with the same mean resultant
// length rho, and s^2 is about 1/kappa when the law is concentrated.
//
// The density is written so that nothing cancels:
//
//   l = kappa(s) S(z) - L(kappa(s)) - log(2 pi),   S(z) = cos z - 1 = -2 sin^2(z/2),
//   L(kappa) = log I_0(kappa) - kappa   (log_bessel_i0_scaled).
//
// Derivatives. l is a product of a function of s and a function of z, minus a function of s, so
// the component with a derivatives in mu and b in s is
//
//   d^a/dmu^a d^b/ds^b l = W_a(z) kappa^(b)(s) - [a = 0] d^b/ds^b L(kappa(s)),
//   W_a(z) = (-1)^a S^(a)(z) = sin z, -cos z, -sin z, cos z, sin z, -cos z   (a = 1..6).
//
// The last term is never formed through Bessel-ratio derivatives. dL/dkappa = A - 1 = -q with
// q = 1 - rho = -expm1(-s^2/2) known from s, so d/ds L(kappa(s)) = -q kappa'(s) and
// d/ds l = kappa'(s) (S + q) = kappa'(s) (cos z - rho), the textbook score. Differentiating the
// score by Leibniz' rule gives the pure s component of order K >= 1 from kappa^(1..K) and the
// derivatives of rho = exp(-s^2/2) alone (detail_vonmises_distrib::pure_s_component):
//
//   d^K/ds^K l = kappa^(K) (S + q) - sum_(j=1)^(K-1) C(K-1, j) rho^(j) kappa^(K-j).
//
// (A Faa di Bruno composition of kappa^(b) with L^(j) = A^(j-1) gives the same number; the test
// keeps it as an independent twin.) d/dy = -d/dmu on every component, so logpdf_dy_dk<J, K> is
// the table of order J + K with the sign (-1)^J, and for J >= 1 it needs only W and kappa^(b).
//
// Expected information, closed and diagonal: I_mu,mu = kappa rho, I_mu,s = 0 exactly, and
// I_ss = s^2 rho^2 / A'(kappa) = -s rho kappa'(s) (from rho' = -s rho = A'(kappa) kappa'(s)). The
// second form is the one coded: it needs no Bessel derivative and is differentiated in s by
// Leibniz' rule. As s -> 0, I_ss -> 2/s^2; as s -> infinity, I_ss -> 2 s^2 rho^2 -> 0 (a uniform
// law does not identify s).
//
// Support and cdf: the representative interval is the fixed principal branch [-pi, pi], so the
// ends do not depend on a parameter (moves_support is false) and the generic quadratures place
// their panels without a parameter-dependent end. The cdf is measured from the fixed end -pi,
//
//   F(y) = (y + pi)/(2 pi) + (1/pi) sum_(j >= 1) R_j(kappa) [sin(j (y - mu)) + sin(j (pi + mu))]/j,
//   R_j = I_j(kappa)/I_0(kappa)
//
// (the second sine is the lower limit and does not vanish). What this convention costs: F is the
// cdf of the representation on [-pi, pi), not of the law on the circle. It is continuous and
// 2 pi-periodic in mu, and analytic in mu, but it is not invariant under a rotation of the circle,
// F(y; mu) != F(y - c; mu - c), and the cut sits in y: F(-pi) = 0 and F(pi) = 1 while the law is
// the same at both. Censoring or truncating a circular response is therefore meaningful only as a
// statement about an arc starting at -pi.
//
// The series needs m = max(12, ceil(8.6 sqrt(kappa)) + 10) terms (measured with mpmath: the
// remainder sum_(j > m) 2 R_j/(pi j) is below 1e-18 for kappa from 0.01 to 1e4), capped at 512
// terms, which covers kappa <= 3400. The series gives F and 1 - F with an absolute error of a few
// eps; where either falls below cdf_series_floor, and for kappa above the cap, the tail is
// integrated instead (panels over the arc that does not hold mu, with the density normalized by
// its larger end value so that nothing overflows) and the other probability is its complement on
// the log scale. The quantile is the generic safeguarded root finder and the draw is the generic
// inversion. The derivatives of log F and log(1 - F) in the parameters are not written: the
// dispatchers take the generic route by quadrature of the density (log_cdf_dk_by_integral), which
// divides the integrand by the density at y and may overflow when the cut separates two parts of
// a very concentrated law (mu within a few s of +-pi).
//
// The arc mass in closed form. The panels of the tail integral are laid out by the decay length of the
// integrand, 1/(kappa |sin((c - mu)/2)|) at an end c, and never by s. Where the quadrature reports that
// it has not converged, the mass of the arc (a, b) is taken from the change of variable
// w = 2 sin((t - mu)/2), monotone on (mu - pi, mu + pi), which makes the exponent exactly -kappa w^2/2
// and dt = dw/sqrt(1 - w^2/4). The remaining factor g(w) = (1 - w^2/4)^(-1/2) varies on scale one while
// the Gaussian varies on 1/sqrt(kappa), so it leaves the integral evaluated at one point and what is
// left is a difference of normal tails,
//
//   M ~ g(w*) sqrt(2 pi / kappa) [Phi(-w_0 sqrt(kappa)) - Phi(-w_1 sqrt(kappa))],
//
// written through log_Phi and log1mexp so that one formula covers both regimes: for an end far from mu
// the asymptotics of Phi give the 1/(kappa w_0) of an end-dominated integral, and for an end at mu,
// where w_0 = 0, the bracket becomes the half Gaussian. The point w* is the dominating end moved in by
// one decay length, which is the first-order correction for the variation of g and is uniformly eight
// times better than taking g at the end itself (measured by tools/reference/vonmises_arc.py against an
// exact mpmath quadrature). The error falls like 1/kappa, so the closed form improves exactly where the
// quadrature degrades, and the caller chooses by what the quadrature reports about itself rather than by
// a threshold in kappa. The two routes meet at w_0 sqrt(kappa) of order one, one Gaussian width from the
// mode, where both are at their worst: there F is accurate to about 3e-10 instead of the 1e-15 reached
// elsewhere, and closing that corner needs a second order in the expansion of g.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/vonmises_scale.hpp>
#include <laplace/math/bessel_i.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <vector>

namespace laplace::distrib {

namespace detail_vonmises_distrib {

inline constexpr double pi_value = 3.14159265358979323846;
inline constexpr double two_pi_value = 6.28318530717958647692;
inline constexpr double two_pi_lo = 2.4492935982947064e-16;  // 2 pi - two_pi_value
inline constexpr double inv_two_pi = 0.15915494309189533577;
inline constexpr double log_2pi = 1.8378770664093454836;
/// Longest cdf series, and the largest kappa for which cdf_terms stays within it.
inline constexpr int max_cdf_terms = 512;
inline constexpr double cdf_series_kappa_max = 3400.0;
/// Below this value of F or of 1 - F the series is replaced by the tail integral.
inline constexpr double cdf_series_floor = 1e-3;

// d wrapped into (-pi, pi]: exact when it is already there, else minus the nearest multiple of 2 pi
// with the product taken by fma and the low part of 2 pi subtracted.
LAPLACE_HD_FORCEINLINE double wrap_angle(double d) {
  if (d > -pi_value && d <= pi_value) return d;
  if (!(Kokkos::fabs(d) < 1e15)) return laplace::math::detail::quiet_nan();
  const double k = Kokkos::floor(d * inv_two_pi + 0.5);
  return Kokkos::fma(-k, two_pi_value, d) - k * two_pi_lo;
}

// W[a] = (-1)^a S^(a)(z), a = 1..6 (W[0] is not used)
LAPLACE_HD_FORCEINLINE void mu_table(double z, double* W) {
  const double sz = Kokkos::sin(z), cz = Kokkos::cos(z);
  W[0] = 0.0;
  W[1] = sz;
  W[2] = -cz;
  W[3] = -sz;
  W[4] = cz;
  W[5] = sz;
  W[6] = -cz;
}

// The pure s component of order K >= 1, from kap[0..K] = kappa^(0..K)(s).
template <int K>
LAPLACE_HD_FORCEINLINE double pure_s_component(double s, double z, const double* kap) {
  const double h = Kokkos::sin(0.5 * z);
  const double e0 = -2.0 * h * h - Kokkos::expm1(-0.5 * s * s);  // S + q = cos z - rho
  if constexpr (K == 1) {
    return kap[1] * e0;
  } else {
    double g[K];  // rho^(0..K-1)
    detail_vonmises::gauss_dk<K - 1>(s, 1.0, g);
    double acc = kap[K] * e0;
    for (int j = 0; j < K - 1; ++j) acc -= double(binomial_coef(K - 1, j)) * kap[j + 1] * g[K - 1 - j];
    return acc;
  }
}

// log(1 - e^x), x <= 0; -infinity for x >= 0
LAPLACE_HD_FORCEINLINE double log1mexp(double x) {
  if (!(x < 0.0)) return x == x ? -distrib_infinity : x;
  return x > -0.69314718055994530942 ? laplace::math::log(-Kokkos::expm1(x)) : Kokkos::log1p(-laplace::math::exp(x));
}

// S = (1/pi) sum_(j=1)^m R_j (sin(j A) + sin(j B)) / j, the sines by rotation recurrences
LAPLACE_HD double cdf_series_sum(double kappa, double A, double B) {
  int m = int(Kokkos::ceil(8.6 * Kokkos::sqrt(kappa))) + 10;
  if (m < 12) m = 12;
  if (m > max_cdf_terms) m = max_cdf_terms;
  double R[max_cdf_terms];
  laplace::math::bessel_i_ratios(kappa, m, R);
  const double sA = Kokkos::sin(A), cA = Kokkos::cos(A), sB = Kokkos::sin(B), cB = Kokkos::cos(B);
  double sa = sA, ca = cA, sb = sB, cb = cB, acc = 0.0;
  for (int j = 1; j <= m; ++j) {
    acc += R[j - 1] * (sa + sb) / double(j);
    const double sa2 = sa * cA + ca * sA;
    ca = ca * cA - sa * sA;
    sa = sa2;
    const double sb2 = sb * cB + cb * sB;
    cb = cb * cB - sb * sB;
    sb = sb2;
  }
  return acc / pi_value;
}

// log of the integral over [a, b] of exp(u(t)), u(t) = -2 kappa sin^2(wrapped(t - mu_w)/2), for an
// arc that does not hold mu_w in its interior: u is largest at an end, which is factored out.
// The panel width is the integrand's own decay length at the end that is
// factored out, and not the circular standard deviation s.
//
// u'(t) = -kappa sin(t - mu_w), so exp(u) falls by e over
// 1/(kappa |sin(c - mu_w)|) at the end c, with a floor of 1/sqrt(kappa) for a
// c close to mu_w, where the width is set by the curvature instead. Passing s
// was right only near the mode: at the end of an arc that does not hold mu it
// is wrong by kappa s |sin|, which at s = 1e-4 (kappa = 1e8) makes the panels
// ten thousand times too wide, the refinement never converges and the value
// comes back silently partial. That is what made the cdf of this family wrong
// in the fourth digit at s = 1e-4 on CUDA while gcc happened to stop nearer
// the truth: measured 0.841804 against 0.841345 at mu = -2, y = -1.9999,
// where the law is a Gaussian of standard deviation s and F is Phi(1).
// statmodels7 panels this same tail by the same length.
// The same arc mass in closed form, for a large kappa where the quadrature
// cannot resolve the integrand.
//
// With w = 2 sin((t - mu_w)/2), which is monotone on (mu_w - pi, mu_w + pi),
// the exponent is exactly -kappa w^2/2 and dt = dw / sqrt(1 - w^2/4), so
//
//   M = integral over [w0, w1] of exp(-kappa w^2/2) g(w) dw,  g = (1 - w^2/4)^(-1/2),
//
// where the Gaussian factor is exact and not an approximation. g varies on a
// scale of order one while the Gaussian varies on 1/sqrt(kappa), so g comes out
// of the integral at one point and what is left is a difference of standard
// normal tails,
//
//   M ~ g(w*) sqrt(2 pi / kappa) [Phi(-w0 sqrt kappa) - Phi(-w1 sqrt kappa)].
//
// Writing it with Phi covers both regimes with one formula: for an endpoint far
// from mu_w, Phi's own asymptotics reproduce the 1/(kappa w0) of an
// endpoint-dominated Laplace integral, and for an endpoint at mu_w, where
// w0 = 0, the bracket becomes the half-Gaussian. The logarithm goes through
// log_Phi, which is measured for exactly this: the far left tail.
//
// w* is the dominating endpoint moved one decay length inwards, which is the
// first-order correction for g's variation and was measured
// (tools/reference/vonmises_arc.py check) to be uniformly about eight times
// better than taking g at the endpoint itself: worst relative error on log M
// over the grid 3.6e-06, 3.2e-07, 2.9e-08, 2.6e-09, 2.4e-10, 2.2e-11, 1.9e-13
// and 1.7e-15 at kappa = 1e3 to 1e12, against 3.9e-05 down to 1.2e-14 for the
// endpoint. The error falls as 1/kappa, so this route gets better exactly
// where the quadrature gets worse, which is why the caller does not need a
// threshold: it takes this one when the quadrature does not converge.
LAPLACE_HD double log_arc_mass_closed(double kappa, double mu_w, double a, double b) {
  const double wa = 2.0 * Kokkos::sin(0.5 * wrap_angle(a - mu_w));
  const double wb = 2.0 * Kokkos::sin(0.5 * wrap_angle(b - mu_w));
  double w0 = Kokkos::fabs(wa), w1 = Kokkos::fabs(wb);
  if (w1 < w0) {
    const double t = w0;
    w0 = w1;
    w1 = t;
  }
  const double rk = Kokkos::sqrt(kappa);
  // g at the dominating endpoint moved in by one decay length, 1/(kappa w0)
  // away from mu_w and 1/sqrt(kappa) close to it
  const double step = (w0 * rk > 1.0) ? 1.0 / (kappa * w0) : 1.0 / rk;
  double ws = w0 + step;
  if (!(ws < 2.0)) ws = w0;  // never leave the chart
  const double q = 1.0 - 0.25 * ws * ws;
  const double log_g = (q > 0.0) ? -0.5 * laplace::math::log(q) : 0.0;
  // log(Phi(-w0 rk) - Phi(-w1 rk)) with the smaller tail factored out
  const double lo = laplace::math::log_Phi(-w0 * rk);
  const double hi = laplace::math::log_Phi(-w1 * rk);
  const double d = hi - lo;  // <= 0
  if (!(d < 0.0)) return -distrib_infinity;
  const double bracket = lo + log1mexp(d);
  return log_g + 0.5 * laplace::math::log(2.0 * pi_value / kappa) + bracket;
}

LAPLACE_HD double log_arc_mass(double kappa, double scale, double mu_w, double a, double b) {
  const auto u = [&](double t) {
    const double h = Kokkos::sin(0.5 * wrap_angle(t - mu_w));
    return -2.0 * kappa * h * h;
  };
  const double ua = u(a), ub = u(b);
  const double um = ua >= ub ? ua : ub;
  const double c = ua >= ub ? a : b;
  const double sl = Kokkos::fabs(Kokkos::sin(wrap_angle(c - mu_w)));
  const double rt = Kokkos::sqrt(kappa);
  double w = kappa * sl;
  if (!(w > rt)) w = rt;            // the curvature's width near the mode
  double width = 1.0 / w;
  if (!(width > 0.0) || width != width) width = scale;  // kappa not finite
  laplace::math::quad::de_quad_options o;
  o.stop = 1e-13;
  const auto f = [&](double t) { return laplace::math::exp(u(t) - um); };
  const auto r = laplace::math::quad::panels_quad(f, a, b, c, width, nullptr, 0, o);
  // Where the quadrature does not converge, the closed form above does, and
  // the two are complementary rather than ranked: the quadrature is exact at
  // moderate kappa, where the asymptotic is weakest, and the asymptotic
  // improves as 1/kappa exactly where the integrand becomes too narrow to
  // resolve. So the route is chosen by what the quadrature reports about
  // itself, not by a threshold in kappa.
  //
  // The family used to read only r.value, which is how a partial sum passed
  // for a probability: at kappa = 1e8 that returned a cdf wrong in the fourth
  // digit, 0.841804 against 0.841345, and no test saw it.
  if (r.value != r.value || !r.converged || !(r.value > 0.0))
    return log_arc_mass_closed(kappa, mu_w, a, b);
  return um + laplace::math::log(r.value);
}

struct cdf_values {
  double F, G, log_F, log_G;  // F, 1 - F and their logarithms
};

}  // namespace detail_vonmises_distrib

/**
 * \ingroup distrib_vonmises
 * \brief Von Mises distribution in the mean direction \f$\mu\f$ and the circular standard deviation \f$s\f$.
 *
 * The density is
 * \f[
 * f(y; \mu, s) = \frac{\exp(\kappa \cos z)}{2\pi I_0(\kappa)},
 * \f]
 * where \f$z = y - \mu\f$ is wrapped to \f$(-\pi, \pi]\f$ and \f$\kappa = A^{-1}(e^{-s^2/2})\f$.
 * \f$\mu\f$ is circular: the identity link on an unbounded line, identified modulo \f$2\pi\f$.
 * The support representative is \f$[-\pi, \pi]\f$ and the cdf is measured from \f$-\pi\f$.
 * \see distrib_vonmises, distrib_core, vonmises_kappa_from_s()
 */
struct vonmises_distrib {
  /// Mean direction \f$\mu \in (-\infty, \infty)\f$ (circular, period \f$2\pi\f$).
  double mu;
  /// Circular standard deviation \f$s \in (0, \infty)\f$.
  double s;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Regularity of the log-density in \f$y\f$ (analytic on the circle; no kinks).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors: mu is circular and unbounded, s is positive; neither moves the support.
  static constexpr distrib_param_info params[2] = {
      {"mu", -distrib_infinity, distrib_infinity, true, infinitely_smooth, false, infinitely_smooth},
      {"s", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_vonmises
   * \brief Log-density \f$-2\kappa\sin^2(z/2) - (\log I_0(\kappa) - \kappa) - \log(2\pi)\f$.
   *
   * Periodic of period \f$2\pi\f$ in \f$y\f$ and \f$\mu\f$; returns quiet \c NaN for
   * \f$|y - \mu| \ge 10^{15}\f$, \f$s \le 0\f$, or \c NaN inputs.
   * \param y Observation point.
   * \return Log-density value.
   * \par Measured accuracy
   * Maximum error \f$4.70\f$ scaled ulp for \f$s \le 4.5\f$ and \f$0.10\f$ for \f$s > 4.5\f$ on \c gcc and \c CUDA (test bounds \f$8\f$ and \f$8(1 + 0.25 s^2)\f$).
   * \see distrib_vonmises, logpdf_dk(), logpdf_dy_dk()
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = detail_vonmises_distrib::wrap_angle(y - mu);
    const double kappa = vonmises_kappa_from_s(s);
    const double h = Kokkos::sin(0.5 * z);
    return -2.0 * kappa * h * h - math::log_bessel_i0_scaled(kappa) - detail_vonmises_distrib::log_2pi;
  }

  /**
   * \ingroup distrib_vonmises
   * \brief Derivatives of order \f$K\f$ of the log-density in \f$(\mu, s)\f$; component \c b has \c b indices equal to \f$s\f$.
   *
   * Separates into \f$W_{K-b}(z)\,\kappa^{(b)}(s)\f$ for \f$b < K\f$ and Leibniz over
   * \f$\kappa'(s)(S(z) + q)\f$ for \f$b = K\f$; returns quiet \c NaN for \f$|y - \mu| \ge 10^{15}\f$,
   * \f$s \le 0\f$, or \c NaN inputs.
   * \tparam K Order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   * \par Measured accuracy
   * Maximum error \f$4.87\f$ to \f$5.10\f$ (\c gcc, \f$5.16\f$ \c CUDA) for \f$s \le 4.5\f$ and \f$0.49\f$ to \f$0.81\f$ for \f$s > 4.5\f$ (test bounds \f$8\f$ and \f$8(1 + 0.25 s^2)\f$).
   * \see distrib_vonmises, logpdf(), logpdf_dy_dk()
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    namespace dv = detail_vonmises_distrib;
    const double z = dv::wrap_angle(y - mu);
    double kap[K + 1];
    vonmises_kappa_from_s_upto<K>(s, kap);
    double W[7];
    dv::mu_table(z, W);
    sym_tensor<2, K> t;
    for (int b = 0; b < K; ++b) t[b] = W[K - b] * kap[b];
    t[K] = dv::pure_s_component<K>(s, z, kap);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives of \f$\log f(y; \mu, s)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivatives.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_vonmises
   * \brief Mixed derivatives, \f$J\f$ in \f$y\f$ and order \f$K\f$ in the parameters, \f$J + K \le 6\f$.
   *
   * \f$\partial_y = -\partial_\mu\f$, so for \f$J \ge 1\f$ the component with \c b indices in \f$s\f$ is
   * \f$(-1)^J W_{K-b+J}(z)\,\kappa^{(b)}(s)\f$. At \f$J = 0, K = 0\f$ returns `logpdf(y)` in `v[0]`;
   * returns quiet \c NaN for \f$|y - \mu| \ge 10^{15}\f$, \f$s \le 0\f$, or \c NaN inputs.
   * \tparam J Order in \f$y\f$, \f$0 \le J \le 6\f$.
   * \tparam K Order in \f$(\mu, s)\f$, \f$0 \le K \le 6\f$, \f$J + K \le 6\f$.
   * \param y Observation point.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of mixed derivatives.
   * \par Measured accuracy
   * Maximum error \f$0.52\f$ to \f$3.33\f$ scaled ulp for \f$s \le 4.5\f$ and \f$0.49\f$ to \f$0.60\f$ for \f$s > 4.5\f$ on \c gcc (test bounds \f$8\f$ and \f$8(1 + 0.25 s^2)\f$).
   * \see distrib_vonmises, logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    static_assert(J >= 0 && K >= 0 && J + K <= 6, "total order at most six");
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<2, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      namespace dv = detail_vonmises_distrib;
      const double z = dv::wrap_angle(y - mu);
      double W[7];
      dv::mu_table(z, W);
      double kap[K + 1];
      if constexpr (K == 0) kap[0] = vonmises_kappa_from_s(s);
      else vonmises_kappa_from_s_upto<K>(s, kap);
      const double sg = (J % 2 == 0) ? 1.0 : -1.0;
      sym_tensor<2, K> r;
      for (int b = 0; b <= K; ++b) r[b] = sg * W[K - b + J] * kap[b];
      return r;
    }
  }

  /**
   * \ingroup distrib_vonmises
   * \brief Expected information \f$\operatorname{diag}(\kappa\rho,\ -s\rho\kappa')\f$; the off-diagonal entry is exactly zero.
   *
   * Returns quiet \c NaN in the diagonal entries for \f$s \le 0\f$ or \c NaN \f$s\f$; grows like
   * \f$\operatorname{diag}(1/s^2, 2/s^2)\f$ as \f$s \to 0\f$ and tends to \f$0\f$ as \f$s \to \infty\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> of the expected Fisher information.
   * \par Measured accuracy
   * Verified within \f$10^{-12}\f$ relative against \c mpmath quadrature (`tests/distrib/test_vonmises.cpp`).
   * \see distrib_vonmises, expected_info_dk()
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    double kap[2];
    vonmises_kappa_from_s_upto<1>(s, kap);
    const double rho = math::exp(-0.5 * s * s);
    sym_tensor<2, 2> t;
    t[0] = kap[0] * rho;
    t[1] = 0.0;
    t[2] = -s * rho * kap[1];
    return t;
  }
  /**
   * \ingroup distrib_vonmises
   * \brief Derivatives of order \f$K\f$ (1 to 4) of the expected information in \f$s\f$; those in \f$\mu\f$ vanish.
   *
   * Differentiates \f$\kappa\rho\f$ and \f$-s\rho\kappa'\f$ by Leibniz's rule; returns quiet \c NaN
   * in the diagonal blocks for \f$s \le 0\f$ or \c NaN \f$s\f$ and \f$0\f$ in all \f$\mu\f$ components.
   * \tparam K Derivative order, \f$1 \le K \le 4\f$.
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   * \par Measured accuracy
   * Verified within \f$2 \times 10^{-12}\f$ relative against \c mpmath (`tests/distrib/test_vonmises.cpp`).
   * \see distrib_vonmises, expected_info()
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    static_assert(K >= 1 && K <= 4, "orders one to four");
    double kap[K + 2];
    vonmises_kappa_from_s_upto<K + 1>(s, kap);
    double g[K + 1];
    detail_vonmises::gauss_dk<K>(s, 1.0, g);
    double imm = 0.0, iss = 0.0;
    for (int j = 0; j <= K; ++j) {
      const double c = double(binomial_coef(K, j));
      imm += c * kap[j] * g[K - j];
      const double p = (j == 0) ? s * g[0] : s * g[j] + double(j) * g[j - 1];  // (s rho)^(j)
      iss -= c * p * kap[K - j + 1];
    }
    expected_info_tensor<2, K> t;
    t.pair[0][K] = imm;
    t.pair[2][K] = iss;
    return t;
  }
  /**
   * \name Derivatives of the expected information
   * Parameter derivatives of \f$I(\mu, s)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_vonmises
   * \brief Cumulative distribution function \f$F(y)\f$ measured from \f$-\pi\f$.
   *
   * Returns \f$0\f$ for \f$y \le -\pi\f$, \f$1\f$ for \f$y \ge \pi\f$, and quiet \c NaN for
   * \f$s \le 0\f$ or \c NaN inputs.
   * \param y Observation point.
   * \return Probability \f$F(y) \in [0, 1]\f$.
   * \par Measured accuracy
   * Worst error-to-tolerance ratio \f$0.24\f$ (series) and \f$0.18\f$ (tail), with \f$2.9 \times 10^{-10}\f$ relative at the \f$w_0\sqrt{\kappa} \approx 1\f$ crossover.
   * \see distrib_vonmises, ccdf(), log_cdf()
   */
  LAPLACE_HD double cdf(double y) const { return cdf_all(y).F; }
  /**
   * \ingroup distrib_vonmises
   * \brief Complementary cumulative distribution function \f$1 - F(y)\f$ on \f$[-\pi, \pi]\f$.
   *
   * Returns \f$1\f$ for \f$y \le -\pi\f$, \f$0\f$ for \f$y \ge \pi\f$, and quiet \c NaN for
   * \f$s \le 0\f$ or \c NaN inputs.
   * \param y Observation point.
   * \return Complementary probability \f$1 - F(y) \in [0, 1]\f$.
   * \par Measured accuracy
   * Worst error-to-tolerance ratio \f$0.24\f$ (series) and \f$0.18\f$ (tail) against \c mpmath quadrature.
   * \see distrib_vonmises, cdf(), log_ccdf()
   */
  LAPLACE_HD double ccdf(double y) const { return cdf_all(y).G; }
  /**
   * \ingroup distrib_vonmises
   * \brief Logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   *
   * Returns \f$-\infty\f$ for \f$y \le -\pi\f$, \f$0\f$ for \f$y \ge \pi\f$, and quiet \c NaN for
   * \f$s \le 0\f$ or \c NaN inputs.
   * \param y Observation point.
   * \return Log-probability \f$\log F(y) \le 0\f$.
   * \par Measured accuracy
   * Worst error-to-tolerance ratio \f$0.24\f$ (series) and \f$0.18\f$ (tail), with \f$1.8 \times 10^{-9}\f$ at the \f$w_0\sqrt{\kappa} \approx 1\f$ crossover.
   * \see distrib_vonmises, cdf(), log_ccdf()
   */
  LAPLACE_HD double log_cdf(double y) const { return cdf_all(y).log_F; }
  /**
   * \ingroup distrib_vonmises
   * \brief Logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   *
   * Returns \f$0\f$ for \f$y \le -\pi\f$, \f$-\infty\f$ for \f$y \ge \pi\f$, and quiet \c NaN for
   * \f$s \le 0\f$ or \c NaN inputs.
   * \param y Observation point.
   * \return Complementary log-probability \f$\log(1 - F(y)) \le 0\f$.
   * \par Measured accuracy
   * Worst error-to-tolerance ratio \f$0.24\f$ (series) and \f$0.18\f$ (tail) against \c mpmath quadrature.
   * \see distrib_vonmises, ccdf(), log_cdf()
   */
  LAPLACE_HD double log_ccdf(double y) const { return cdf_all(y).log_G; }

  /// \ingroup distrib_vonmises
  /// \brief Lower end of the representative interval (\f$-\pi\f$, exact constant).
  /// \return \f$-\pi\f$.
  LAPLACE_HD double support_lower() const { return -detail_vonmises_distrib::pi_value; }
  /// \ingroup distrib_vonmises
  /// \brief Upper end of the representative interval (\f$\pi\f$, exact constant).
  /// \return \f$\pi\f$.
  LAPLACE_HD double support_upper() const { return detail_vonmises_distrib::pi_value; }
  /// \ingroup distrib_vonmises
  /// \brief Mean direction wrapped into \f$(-\pi, \pi]\f$, where the density peaks; returns quiet \c NaN if \f$|\mu| \ge 10^{15}\f$ or \c NaN (accurate to wrapping rounding).
  /// \return Wrapped mean direction in \f$(-\pi, \pi]\f$.
  LAPLACE_HD double center() const { return detail_vonmises_distrib::wrap_angle(mu); }
  /// \ingroup distrib_vonmises
  /// \brief Typical spread (\f$s\f$, exact field value).
  /// \return Circular standard deviation \f$s\f$.
  LAPLACE_HD double scale() const { return s; }
  /// \ingroup distrib_vonmises
  /// \brief Non-smooth points in either parameter (none; exact empty \c distrib_kinks).
  /// \return Empty \c distrib_kinks struct.
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// \ingroup distrib_vonmises
  /// \brief Non-smooth points in \f$y\f$ (none; exact empty \c distrib_kinks).
  /// \return Empty \c distrib_kinks struct.
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }

  /**
   * \ingroup distrib_vonmises
   * \brief Starting value from the sample (host): the circular moment estimate.
   *
   * With \f$c, \bar s\f$ the means of \f$\cos y_i\f$ and \f$\sin y_i\f$: \f$\hat\mu = \operatorname{atan2}(\bar s, c)\f$,
   * \f$\hat\rho^2 = c^2 + \bar s^2\f$ and \f$\hat s = \sqrt{-\log\hat\rho^2}\f$, the definition of \f$s\f$ (no inversion).
   * \f$\hat s\f$ is kept in \f$[10^{-4}, 4.29]\f$: a sample with no spread (\f$\hat\rho = 1\f$) returns
   * \f$10^{-4}\f$, a perfectly uniform one (\f$\hat\rho = 0\f$) returns \f$\sqrt{-\log 10^{-8}} = 4.29\f$,
   * and an empty or non-finite sample returns \f$(0, 1)\f$. Non-finite observations are skipped.
   * \param y Sample vector on the host.
   * \return A \c vonmises_distrib with parameters strictly inside their open domains.
   * \see distrib_vonmises, distrib_start_value(), distrib_start
   */
  vonmises_distrib start_value(const std::vector<double>& y) const {
    double c = 0.0, sn = 0.0, n = 0.0;
    for (double v : y) {
      if (!std::isfinite(v)) continue;
      c += std::cos(v);
      sn += std::sin(v);
      n += 1.0;
    }
    if (n == 0.0) return {0.0, 1.0};
    c /= n;
    sn /= n;
    const double r2 = c * c + sn * sn;
    const double s_hi = std::sqrt(-std::log(1e-8)), s_lo = 1e-4;
    double s_hat = r2 > 0.0 ? std::sqrt(-std::log(r2)) : s_hi;
    if (!(s_hat < s_hi)) s_hat = s_hi;
    if (!(s_hat > s_lo)) s_hat = s_lo;
    return {start_inside(std::atan2(sn, c), params[0]), start_inside(s_hat, params[1])};
  }

 private:
  // F, 1 - F and their logarithms. Series where it is accurate, tail integral elsewhere.
  LAPLACE_HD detail_vonmises_distrib::cdf_values cdf_all(double y) const {
    namespace dv = detail_vonmises_distrib;
    dv::cdf_values r;
    if (!(y == y) || !(s > 0.0)) {
      r.F = r.G = r.log_F = r.log_G = laplace::math::detail::quiet_nan();
      return r;
    }
    if (y <= -dv::pi_value) {
      r.F = 0.0, r.G = 1.0, r.log_F = -distrib_infinity, r.log_G = 0.0;
      return r;
    }
    if (y >= dv::pi_value) {
      r.F = 1.0, r.G = 0.0, r.log_F = 0.0, r.log_G = -distrib_infinity;
      return r;
    }
    const double kappa = vonmises_kappa_from_s(s);
    const double mu_w = dv::wrap_angle(mu);
    if (kappa <= dv::cdf_series_kappa_max) {
      const double S = dv::cdf_series_sum(kappa, y - mu_w, dv::pi_value + mu_w);
      double F = (y + dv::pi_value) * dv::inv_two_pi + S;
      double G = (dv::pi_value - y) * dv::inv_two_pi - S;
      F = F < 0.0 ? 0.0 : (F > 1.0 ? 1.0 : F);
      G = G < 0.0 ? 0.0 : (G > 1.0 ? 1.0 : G);
      if (F >= dv::cdf_series_floor && G >= dv::cdf_series_floor) {
        r.F = F, r.G = G;
        r.log_F = F >= 0.5 ? Kokkos::log1p(-G) : laplace::math::log(F);
        r.log_G = G >= 0.5 ? Kokkos::log1p(-F) : laplace::math::log(G);
        return r;
      }
    }
    // the arc that does not hold mu_w is integrated, the other probability is its complement
    const double lc = -math::log_bessel_i0_scaled(kappa) - dv::log_2pi;
    if (y < mu_w) {
      r.log_F = lc + dv::log_arc_mass(kappa, s, mu_w, -dv::pi_value, y);
      r.log_G = dv::log1mexp(r.log_F);
    } else {
      r.log_G = lc + dv::log_arc_mass(kappa, s, mu_w, y, dv::pi_value);
      r.log_F = dv::log1mexp(r.log_G);
    }
    r.F = laplace::math::exp(r.log_F);
    r.G = laplace::math::exp(r.log_G);
    return r;
  }
};

}  // namespace laplace::distrib
