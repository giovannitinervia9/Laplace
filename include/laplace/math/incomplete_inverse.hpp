// Inverses of the regularized incomplete gamma and beta functions, Laplace's
// own (a, b > 0):
//
//   gamma_p_inv(a, p)        x with P(a, x) = p
//   gamma_q_inv(a, q)        x with Q(a, x) = 1 - P(a, x) = q
//   gamma_p_inv_logp(a, l)   x with log P(a, x) = l, for l <= 0 (p below the double range)
//   gamma_q_inv_logq(a, l)   x with log Q(a, x) = l
//   ibeta_inv(a, b, p)       x with I_x(a, b) = p
//   ibetac_inv(a, b, q)      x with 1 - I_x(a, b) = q
//   ibeta_inv_logp, ibetac_inv_logq   the same from log p, log q
//
// laplace::math::own returns, for the beta, the pair (x, 1 - x), each with
// its full relative precision (1 - x is not 1 minus the rounded x when x is
// near 1); laplace::math returns x.
//
// Why own: Boost.Math's inverses in double (the only precision on the GPU)
// fail in the tails: gamma_p_inv returns 1e248 for subnormal p, ibeta_inv NaN
// for subnormal p and factor-2 errors near p = 1e-44 (docs/numerics.md).
//
// Method. The side whose probability is the smaller of the two (P or Q, I or
// 1 - I) is the one solved, always from its own value, never from 1 minus the
// other: gamma_q_inv(a, q) with q < 1/2 solves Q(a, x) = q, and with q >= 1/2
// solves P(a, x) = 1 - q, which is then exact. The equation is
//
//     g(v) = log S(x(v)) - log s = 0,
//
// with S the chosen side, s its target, and v = log x for the gamma,
// v = logit x = log(x / (1 - x)) for the beta. In these variables log S is
// concave for every a, b > 0 (the densities of log X and logit X are
// log-concave, so their distribution and survival functions are too), so that
// Newton's iterates approach the root monotonically after the first step from
// any start. The derivatives are closed-form: with D = x f(x) for the gamma
// and D = x (1 - x) f(x) for the beta (f the density),
//
//     g'  = +-D / S,
//     g'' = g' (a - x - g')                  (gamma),
//     g'' = g' (a (1 - x) - b x - g')        (beta),
//
// and log D is written without cancellation for large shapes (as the
// prefactors of incomplete_gamma.hpp and incomplete_beta.hpp). Halley's step
// is taken when its correction factor 1 - g g'' / (2 g'^2) = 1 + dv h / 2
// (dv = -g / g' Newton's step, g'' = g' h) exceeds 1/2 (above 1 it
// only shortens the step, which tempers the overshoot of Newton's step from
// the convex side), Newton's otherwise; a bracket from the signs of g turns a
// step that leaves it into a bisection. x is carried in full precision and
// moved by x expm1(dv) (gamma) or by the equivalent update of the pair
// (x, 1 - x) (beta), so that the last steps change it by fractions of an ulp.
// The iteration stops
//   - when the error left after the step, predicted from g', g'' and g''' in
//     closed form (converged_after), is below eps / 16: this saves the
//     evaluation that would only confirm convergence (1.3 evaluations of the
//     forward function per call on average for the gamma, 2.5 for the beta,
//     against 2.4 and 4.4 without it, at the same accuracy);
//   - when the step is below half an ulp of x, or below 4 eps (with FMA
//     contraction on CUDA a test at eps may never be met);
//   - at the noise floor of the forward function: |g| < 1e-6 and either the
//     step no longer shrinks or it leaves a bracket that the noise has closed;
//   - after 64 steps.
// The logarithms of the forward functions keep the iteration meaningful for p
// down to the subnormals and, in the _logp versions, below the double range.
// The accuracy is that of the forward functions: the computed x solves
// own::gamma_inc or own::beta_inc to their own error (docs/numerics.md).
//
// Starting values. Gamma: DiDonato and Morris (1986, ACM TOMS 12, 377-393),
// equations 21-25 for a < 1 and 31-36 for a > 1, adapted from Boost.Math's
// find_inverse_gamma (boost/math/special_functions/detail/igamma_inverse.hpp,
// Boost Software License 1.0) and rewritten in terms of log p and log q, so
// that they hold for p below the double range. Beta: for a, b >= 1 the normal
// approximation of Abramowitz and Stegun 26.5.22 with the normal quantile of
// DiDonato and Morris' equation 32, written as a logit; otherwise the
// power-law tails x^a / (a w) and (1 - x)^b / (b w) of Numerical Recipes
// (3rd edition, invbetai), in logs. Roots below 2^-1075 return 0 without
// iterating: there P(a, x) = x^a / Gamma(a + 1) and I_x(a, b) =
// x^a / (a B(a, b)) up to a relative 1e-300. On the other side (the upper
// tail of a tiny shape, e.g. gamma_q_inv(1e-10, 0.3), whose root is about
// e^-3e9) the iteration returns 0, or 1 - x = 0 for the beta, when it reaches
// the smallest subnormal with the root still beyond it.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/log1p.hpp>

namespace laplace::math {

namespace own {

struct beta_inv_result {
  double x, y;  // x and 1 - x
};

namespace detail_inverse {

constexpr double eps = 1.1102230246251565e-16;   // 2^-53
constexpr double log_half_min = -745.13321910194122;  // log(2^-1075): smaller roots round to 0
constexpr double log_min = -744.44007192138122;   // log(2^-1074), the smallest subnormal
constexpr double log_max = 709.78271289338397;    // log of the largest double
constexpr double min_subnormal = 4.9406564584124654e-324;  // 2^-1074
constexpr double euler = 0.57721566490153286061;
constexpr double ln2 = 0.69314718055994530942;
constexpr int max_iter = 64;

// Upper normal quantile s > 0 of a small probability e^l (l <= log 1/2),
// from t = sqrt(-2 l): DiDonato and Morris (1986), equation 32, as in Boost's
// find_inverse_s. Valid for any l <= log 1/2, also below the double range.
LAPLACE_HD double normal_upper_quantile(double l) {
  const double t = Kokkos::sqrt(-2.0 * l);
  const double num = 3.31125922108741 + t * (11.6616720288968 + t * (4.28342155967104 + t * 0.213623493715853));
  const double den =
      1.0 + t * (6.61053765625462 + t * (6.40691597760039 + t * (1.27364489782223 + t * 0.3611708101884203e-1)));
  return t - num / den;
}

// DiDonato and Morris (1986), equation 25: x for large y = -log(q Gamma(a))
LAPLACE_HD double dm_eq25(double a, double y) {
  const double c1 = (a - 1.0) * math::log(y);
  const double c1_2 = c1 * c1, c1_3 = c1_2 * c1, c1_4 = c1_2 * c1_2;
  const double a_2 = a * a, a_3 = a_2 * a;
  const double c2 = (a - 1.0) * (1.0 + c1);
  const double c3 = (a - 1.0) * (-(c1_2 / 2.0) + (a - 2.0) * c1 + (3.0 * a - 5.0) / 2.0);
  const double c4 = (a - 1.0) * ((c1_3 / 3.0) - (3.0 * a - 5.0) * c1_2 / 2.0 + (a_2 - 6.0 * a + 7.0) * c1 +
                                 (11.0 * a_2 - 46.0 * a + 47.0) / 6.0);
  const double c5 = (a - 1.0) * (-(c1_4 / 4.0) + (11.0 * a - 17.0) * c1_3 / 6.0 + (-3.0 * a_2 + 13.0 * a - 13.0) * c1_2 +
                                 (2.0 * a_3 - 25.0 * a_2 + 72.0 * a - 61.0) * c1 / 2.0 +
                                 (25.0 * a_3 - 195.0 * a_2 + 477.0 * a - 379.0) / 12.0);
  const double iy = 1.0 / y;
  return y + c1 + iy * (c2 + iy * (c3 + iy * (c4 + iy * c5)));
}

// Whether the error left after a step dv of Newton's or Halley's method on g
// (in the working variable v) is predicted below eps / 16, so that no further
// evaluation is needed. With g'' = g' h and g''' = g' (h^2 - c - g' h), the
// error after the step is about (h / 2) dv^2 for Newton and
// |(g''/(2 g'))^2 - g'''/(6 g')| |dv|^3 = |h^2 / 12 + (c + g' h) / 6| |dv|^3
// for Halley (dv standing for the current error, which it approximates).
LAPLACE_HD bool converged_after(bool halley, double dv, double h, double c, double g1) {
  const double adv = Kokkos::fabs(dv);
  if (!(adv < 1e-3)) return false;
  const double e = halley ? Kokkos::fabs(h * h / 12.0 + (c + g1 * h) / 6.0) * adv * adv * adv
                          : 0.5 * Kokkos::fabs(h) * adv * adv;
  return e <= eps / 16.0;
}

// ------------------------------------------------------------------ gamma

// log Gamma(a) and log(x f(x)), f the gamma(a, 1) density, without the
// cancellation of a log x against log Gamma(a) for large a:
//     x f(x) = x^a e^-x / Gamma(a) = a exp(-a phi(x/a)) / (sqrt(2 pi a) Gamma*(a)),
//     phi(lambda) = lambda - 1 - log(lambda).
struct gamma_shape {
  double a, lga, c;
  bool large;

  LAPLACE_HD explicit gamma_shape(double a_) : a(a_) {
    using detail_igamma::half_log_two_pi;
    large = a >= 10.0;
    if (large) {
      const double lgs = detail_igamma::log_gamma_star(a);
      lga = lgs + (a - 0.5) * math::log(a) - a + half_log_two_pi;
      c = 0.5 * math::log(a) - half_log_two_pi - lgs;
    } else {
      lga = math::lgamma(a);
      c = -lga;
    }
  }

  LAPLACE_HD double log_xdens(double x) const {
    if (!large) return a * math::log(x) - x + c;
    const double lambda = x / a;
    double phi;
    if (lambda > 0.5 && lambda < 1.5) {
      phi = -boost::math::log1pmx((x - a) / a, math::detail::boost_policy());
    } else {
      phi = lambda - 1.0 - math::log(lambda);
    }
    return c - a * phi;
  }
};

// Starting value (DiDonato and Morris 1986, after Boost's find_inverse_gamma).
// lower: the target is P = e^lp (p <= 1/2); otherwise Q = e^lq (q < 1/2).
LAPLACE_HD double gamma_start(const gamma_shape& sh, bool lower, double lp, double lq) {
  const double a = sh.a;
  if (a == 1.0) return -lq;
  const double lga1 = sh.lga + math::log(a);  // log Gamma(a + 1)
  if (a < 1.0) {
    const double log_b = lq + sh.lga;  // b = q Gamma(a)
    const double b = math::exp(log_b);
    if (b > 0.6 || (b >= 0.45 && a >= 0.3)) {
      // equation 21, with (p Gamma(a + 1))^(1/a) taken from log p
      const double u = math::exp((lp + lga1) / a);
      return u / (1.0 - u / (a + 1.0));
    }
    if (a < 0.3 && b >= 0.35) {
      // equation 22
      const double t = math::exp(-euler - b);
      const double u = t * math::exp(t);
      return t * math::exp(u);
    }
    const double y = -log_b;
    if (b > 0.15 || a >= 0.3) {
      // equation 23
      const double u = y - (1.0 - a) * math::log(y);
      return y - (1.0 - a) * math::log(u) - Kokkos::log1p((1.0 - a) / (1.0 + u));
    }
    if (b > 0.1) {
      // equation 24
      const double u = y - (1.0 - a) * math::log(y);
      return y - (1.0 - a) * math::log(u) -
             math::log((u * u + 2.0 * (3.0 - a) * u + (2.0 - a) * (3.0 - a)) / (u * u + (5.0 - a) * u + 2.0));
    }
    return dm_eq25(a, y);
  }

  // a > 1: equation 31, s the normal quantile of p
  double s = normal_upper_quantile(lower ? lp : lq);
  if (lower) s = -s;
  const double s_2 = s * s, s_3 = s_2 * s, s_4 = s_2 * s_2, s_5 = s_4 * s;
  const double ra = Kokkos::sqrt(a);
  double w = a + s * ra + (s_2 - 1.0) / 3.0;
  w += (s_3 - 7.0 * s) / (36.0 * ra);
  w -= (3.0 * s_4 + 7.0 * s_2 - 16.0) / (810.0 * a);
  w += (9.0 * s_5 + 256.0 * s_3 - 433.0 * s) / (38880.0 * a * ra);
  if (a >= 500.0 && Kokkos::fabs(1.0 - w / a) < 1e-6) return w;
  if (!lower) {
    if (w < 3.0 * a) return w;
    const double D = Kokkos::fmax(2.0, a * (a - 1.0));
    const double lb = lq + sh.lga;
    if (lb < -D * 2.3) return dm_eq25(a, -lb);
    // equation 33
    const double u = -lb + (a - 1.0) * math::log(w) - Kokkos::log1p((1.0 - a) / (1.0 + w));
    return -lb + (a - 1.0) * math::log(u) - Kokkos::log1p((1.0 - a) / (1.0 + u));
  }
  double z = w;
  const double ap1 = a + 1.0, ap2 = a + 2.0;
  const double v = lp + lga1;
  if (w < 0.15 * ap1) {
    // equation 35
    z = math::exp((v + w) / a);
    double sl = Kokkos::log1p(z / ap1 * (1.0 + z / ap2));
    z = math::exp((v + z - sl) / a);
    sl = Kokkos::log1p(z / ap1 * (1.0 + z / ap2));
    z = math::exp((v + z - sl) / a);
    sl = Kokkos::log1p(z / ap1 * (1.0 + z / ap2 * (1.0 + z / (a + 3.0))));
    z = math::exp((v + z - sl) / a);
  }
  if (z <= 0.01 * ap1 || z > 0.7 * ap1) return z;
  // equation 36: S_N(a, z) = 1 + z/(a+1) + z^2/((a+1)(a+2)) + ..., to a relative 1e-4
  double sum = 1.0, partial = 1.0;
  for (int i = 1; i <= 100; ++i) {
    partial *= z / (a + i);
    sum += partial;
    if (partial < 1e-4) break;
  }
  const double ls = math::log(sum);
  z = math::exp((v + z - ls) / a);
  return z * (1.0 - (a * math::log(z) - z - v + ls) / (a - z));
}

// x with log P(a, x) = lp (lower) or log Q(a, x) = lq; the chosen side is the
// smaller one, and both logarithms are given.
LAPLACE_HD double gamma_solve(double a, bool lower, double lp, double lq) {
  const double inf = math::detail::pos_inf();
  const gamma_shape sh(a);
  if (lower && (lp + sh.lga + math::log(a)) / a < log_half_min) return 0.0;
  const double target = lower ? lp : lq;

  double x = gamma_start(sh, lower, lp, lq);
  if (!(x > 0.0) || !(x < inf)) {
    // fallbacks: the power law of the lower tail, or the upper tail
    if (lower) {
      x = math::exp(Kokkos::fmax((lp + sh.lga + math::log(a)) / a, log_min));
    } else {
      const double L = -lq;
      x = a + Kokkos::sqrt(2.0 * a * L) + L;
    }
  }
  double lo = 0.0, hi = inf;  // bracket of the root
  double step_prev = inf;
  bool prev_newton = false;
  for (int it = 0; it < max_iter; ++it) {
    const auto r = gamma_inc(a, x);
    const double ls = lower ? r.log_p : r.log_q;
    const double g = ls - target;
    if (g == 0.0 || g != g) return x;
    // lower: g increases with x; upper: g decreases
    const bool below = (g < 0.0) == lower;  // the root is to the right of x
    if (!below && x <= min_subnormal) return 0.0;  // a root below the smallest subnormal
    if (below) lo = x;
    else hi = x;
    double g1 = math::exp(sh.log_xdens(x) - ls);
    if (!lower) g1 = -g1;
    const double h = a - x - g1;  // g'' = g' h
    double dt = -g / g1;          // Newton's step in t = log x
    const double den = 1.0 + 0.5 * dt * h;
    const bool halley = den > 0.5;  // Halley's correction; den > 1 only shortens the step
    if (halley) dt /= den;
    double xn = x;
    const double t = math::log(x);
    const bool newton = below ? dt > 0.0 : dt < 0.0;  // toward the root (false for NaN)
    if (newton) {
      xn = Kokkos::fabs(dt) < 0.5 ? x + x * Kokkos::expm1(dt)
                                  : math::exp(Kokkos::fmin(Kokkos::fmax(t + dt, log_min), log_max));
      if (xn == x) return x;  // a step below half an ulp of x
    }
    const bool inside = newton && xn > lo && xn < hi;
    if (inside && converged_after(halley, dt, h, x, g1)) return xn;
    if (!inside) {
      // near the root, a step that leaves the bracket means the noise of the
      // forward function has closed it: stop
      if (Kokkos::fabs(g) < 1e-6) return x;
      // bisection in log x, or a step by a factor e toward the side not yet bracketed
      double tn;
      if (lo > 0.0 && hi < inf) tn = 0.5 * (math::log(lo) + math::log(hi));
      else tn = below ? t + 1.0 : t - 1.0;
      xn = math::exp(Kokkos::fmin(Kokkos::fmax(tn, log_min), log_max));
      if (!(xn > lo && xn < hi)) return x;  // the bracket has closed to adjacent doubles
      dt = tn - t;
    }
    const double adt = Kokkos::fabs(dt);
    if (xn == x || (inside && adt <= 4.0 * eps)) return xn;
    // the noise floor of the forward function: near the root (|g| < 1e-6)
    // Newton's and Halley's steps shrink at least quadratically
    if (inside && prev_newton && Kokkos::fabs(g) < 1e-6 && adt > 0.25 * step_prev) return xn;
    step_prev = adt;
    prev_newton = inside;
    x = xn;
  }
  return x;
}

// ------------------------------------------------------------------ beta

// log(1 + t) - t, given log(1 + t) computed from the ratio itself
LAPLACE_HD double log1pmx_from(double t, double log_ratio) {
  if (Kokkos::fabs(t) <= 0.5) return boost::math::log1pmx(t, math::detail::boost_policy());
  return log_ratio - t;
}

// log B(a, b) and log D(x) = log(x^a (1 - x)^b / B(a, b)) = log(x (1 - x) f(x)),
// for a, b >= 10 through the mean p0 = a / (a + b) as in incomplete_beta.hpp,
// with nothing cancelling.
struct beta_shape {
  double a, b, lbeta, c, p0, q0;
  bool large;

  LAPLACE_HD beta_shape(double a_, double b_) : a(a_), b(b_) {
    using detail_igamma::half_log_two_pi;
    using detail_igamma::log_gamma_star;
    const double s = a + b;
    p0 = a / s;
    q0 = b / s;
    large = a >= 10.0 && b >= 10.0;
    if (large) {
      c = 0.5 * math::log(a * b / s) - half_log_two_pi + log_gamma_star(s) - log_gamma_star(a) - log_gamma_star(b);
      lbeta = a * math::log(p0) + b * math::log(q0) - c;
    } else {
      const double B = boost::math::beta(a, b, math::detail::boost_policy());
      lbeta = (B > 0.0 && B < math::detail::pos_inf()) ? math::log(B)
                                                       : math::lgamma(a) + math::lgamma(b) - math::lgamma(a + b);
      c = -lbeta;
    }
  }

  // x and y = 1 - x both given, the smaller of the two with full precision
  LAPLACE_HD double log_d(double x, double y) const {
    const double lx = x <= 0.5 ? math::log(x) : Kokkos::log1p(-y);
    const double ly = x <= 0.5 ? Kokkos::log1p(-x) : math::log(y);
    if (!large) return a * lx + b * ly + c;
    const double u = (x - p0) / p0, v = (p0 - x) / q0;
    return a * log1pmx_from(u, lx - math::log(p0)) + b * log1pmx_from(v, ly - math::log(q0)) + c;
  }
};

// Starting value of logit x. lower: the target is I = e^lp (p <= 1/2);
// otherwise 1 - I = e^lq.
LAPLACE_HD double beta_start(const beta_shape& sh, bool lower, double lp, double lq) {
  const double a = sh.a, b = sh.b;
  if (a >= 1.0 && b >= 1.0) {
    // Abramowitz and Stegun 26.5.22: x = a / (a + b e^(2w)), y_p the upper
    // normal quantile of p
    const double s = normal_upper_quantile(lower ? lp : lq);
    const double yp = lower ? s : -s;
    const double lambda = (yp * yp - 3.0) / 6.0;
    const double ia = 1.0 / (2.0 * a - 1.0), ib = 1.0 / (2.0 * b - 1.0);
    const double h = 2.0 / (ia + ib);
    const double w = yp * Kokkos::sqrt(h + lambda) / h - (ib - ia) * (lambda + 5.0 / 6.0 - 2.0 / (3.0 * h));
    return math::log(a / b) - 2.0 * w;
  }
  // Numerical Recipes' power-law tails: I ~ x^a / (a w) below the split
  // t / w, 1 - I ~ (1 - x)^b / (b w) above, with t = p0^a / a, u = q0^b / b,
  // w = t + u
  const double log_t = a * math::log(sh.p0) - math::log(a);
  const double log_u = b * math::log(sh.q0) - math::log(b);
  const double m = Kokkos::fmax(log_t, log_u);
  const double log_w = m + Kokkos::log1p(math::exp(-Kokkos::fabs(log_t - log_u)));
  if (lp < log_t - log_w) {
    const double lx = (lp + math::log(a) + log_w) / a;
    return lx - Kokkos::log1p(-math::exp(lx));
  }
  const double ly = (lq + math::log(b) + log_w) / b;
  return Kokkos::log1p(-math::exp(ly)) - ly;
}

// the pair (x, 1 - x) from u = logit x
LAPLACE_HD beta_inv_result from_logit(double u) {
  if (u < 0.0) {
    const double e = math::exp(u);
    return {e / (1.0 + e), 1.0 / (1.0 + e)};
  }
  const double e = math::exp(-u);
  return {1.0 / (1.0 + e), e / (1.0 + e)};
}

// logit x of a pair, from the side held exactly
LAPLACE_HD double logit(const beta_inv_result& s) {
  return s.x <= 0.5 ? math::log(s.x) - Kokkos::log1p(-s.x) : Kokkos::log1p(-s.y) - math::log(s.y);
}

// s1 strictly left of s2 (x1 < x2), comparing the sides held exactly
LAPLACE_HD bool left_of(const beta_inv_result& s1, const beta_inv_result& s2) {
  if (s1.x <= 0.5 && s2.x <= 0.5) return s1.x < s2.x;
  if (s1.x > 0.5 && s2.x > 0.5) return s1.y > s2.y;
  return s1.x <= 0.5;
}

// (x, 1 - x) with log I_x(a, b) = lp (lower) or log(1 - I_x(a, b)) = lq
LAPLACE_HD beta_inv_result beta_solve(double a, double b, bool lower, double lp, double lq) {
  const double inf = math::detail::pos_inf();
  const beta_shape sh(a, b);
  if (lower && (lp + math::log(a) + sh.lbeta) / a < log_half_min) return {0.0, 1.0};
  if (!lower && (lq + math::log(b) + sh.lbeta) / b < log_half_min) return {1.0, 0.0};
  const double target = lower ? lp : lq;

  double u0 = beta_start(sh, lower, lp, lq);
  if (u0 != u0) u0 = math::log(sh.p0) - math::log(sh.q0);
  u0 = Kokkos::fmin(Kokkos::fmax(u0, log_min), -log_min);
  beta_inv_result s = from_logit(u0);
  // bracket of the root, as pairs; (0, 1) and (1, 0) until a sign is seen
  beta_inv_result lo{0.0, 1.0}, hi{1.0, 0.0};
  bool has_lo = false, has_hi = false;
  double step_prev = inf;
  bool prev_newton = false;
  for (int it = 0; it < max_iter; ++it) {
    const double x = s.x, y = s.y;
    // the forward function at the pair, from the side that is held exactly
    double log_i, log_ic;
    if (x <= 0.5) {
      const auto r = beta_inc(a, b, x);
      log_i = r.log_i;
      log_ic = r.log_ic;
    } else {
      const auto r = beta_inc(b, a, y);
      log_i = r.log_ic;
      log_ic = r.log_i;
    }
    const double ls = lower ? log_i : log_ic;
    const double g = ls - target;
    if (g == 0.0 || g != g) return s;
    const double u = x <= 0.5 ? math::log(x) - Kokkos::log1p(-x) : Kokkos::log1p(-y) - math::log(y);
    const bool below = (g < 0.0) == lower;  // the root is to the right of x
    // roots closer to 0 or 1 than the smallest subnormal
    if (!below && x <= min_subnormal) return {0.0, 1.0};
    if (below && y <= min_subnormal) return {1.0, 0.0};
    if (below) {
      lo = s;
      has_lo = true;
    } else {
      hi = s;
      has_hi = true;
    }
    double g1 = math::exp(sh.log_d(x, y) - ls);
    if (!lower) g1 = -g1;
    const double h = a * y - b * x - g1;  // g'' = g' h
    double du = -g / g1;                  // Newton's step in u = logit x
    const double den = 1.0 + 0.5 * du * h;
    const bool halley = den > 0.5;  // Halley's correction; den > 1 only shortens the step
    if (halley) du /= den;
    beta_inv_result sn = s;
    const bool newton = below ? du > 0.0 : du < 0.0;  // toward the root (false for NaN)
    if (newton) {
      if (Kokkos::fabs(du) < 0.5) {
        // logit moved by du: x' = x e^du / (1 + x (e^du - 1)), y' = y / (1 + x (e^du - 1)),
        // from the side held exactly
        if (x <= y) {
          const double E = Kokkos::expm1(du), d = 1.0 + x * E;
          sn = {x * (1.0 + E) / d, y / d};
        } else {
          const double E = Kokkos::expm1(-du), d = 1.0 + y * E;
          sn = {x / d, y * (1.0 + E) / d};
        }
        if (sn.x <= 0.5) sn.y = 1.0 - sn.x;
        else sn.x = 1.0 - sn.y;
      } else {
        sn = from_logit(Kokkos::fmin(Kokkos::fmax(u + du, log_min), -log_min));
      }
      if (sn.x == x && sn.y == y) return s;  // a step below half an ulp
    }
    const bool inside = newton && left_of(lo, sn) && left_of(sn, hi);
    if (inside && converged_after(halley, du, h, (a + b) * x * y, g1)) return sn;
    if (!inside) {
      // near the root, a step that leaves the bracket means the noise of the
      // forward function has closed it: stop
      if (Kokkos::fabs(g) < 1e-6) return s;
      // bisection in logit x, or a step of 1 in logit x toward the side not yet bracketed
      double un;
      if (has_lo && has_hi) un = 0.5 * (logit(lo) + logit(hi));
      else un = below ? u + 1.0 : u - 1.0;
      sn = from_logit(Kokkos::fmin(Kokkos::fmax(un, log_min), -log_min));
      if (!(left_of(lo, sn) && left_of(sn, hi))) return s;  // the bracket has closed to adjacent doubles
      du = un - u;
    }
    const double adu = Kokkos::fabs(du);
    if ((sn.x == x && sn.y == y) || (inside && adu <= 4.0 * eps)) return sn;
    if (inside && prev_newton && Kokkos::fabs(g) < 1e-6 && adu > 0.25 * step_prev) return sn;
    step_prev = adu;
    prev_newton = inside;
    s = sn;
  }
  return s;
}

}  // namespace detail_inverse

// ------------------------------------------------------------------ public

LAPLACE_HD double gamma_p_inv(double a, double p) {
  const double nan = math::detail::quiet_nan();
  if (!(a > 0.0) || !(a < math::detail::pos_inf()) || !(p >= 0.0) || !(p <= 1.0)) return nan;
  if (p == 0.0) return 0.0;
  if (p == 1.0) return math::detail::pos_inf();
  if (p <= 0.5) return detail_inverse::gamma_solve(a, true, math::log(p), Kokkos::log1p(-p));
  const double q = 1.0 - p;  // exact
  return detail_inverse::gamma_solve(a, false, Kokkos::log1p(-q), math::log(q));
}

LAPLACE_HD double gamma_q_inv(double a, double q) {
  const double nan = math::detail::quiet_nan();
  if (!(a > 0.0) || !(a < math::detail::pos_inf()) || !(q >= 0.0) || !(q <= 1.0)) return nan;
  if (q == 0.0) return math::detail::pos_inf();
  if (q == 1.0) return 0.0;
  if (q < 0.5) return detail_inverse::gamma_solve(a, false, Kokkos::log1p(-q), math::log(q));
  const double p = 1.0 - q;  // exact
  return detail_inverse::gamma_solve(a, true, math::log(p), Kokkos::log1p(-p));
}

LAPLACE_HD double gamma_p_inv_logp(double a, double log_p) {
  const double nan = math::detail::quiet_nan();
  if (!(a > 0.0) || !(a < math::detail::pos_inf()) || !(log_p <= 0.0)) return nan;
  if (log_p == 0.0) return math::detail::pos_inf();
  if (log_p == -math::detail::pos_inf()) return 0.0;
  if (log_p <= -detail_inverse::ln2)
    return detail_inverse::gamma_solve(a, true, log_p, Kokkos::log1p(-math::exp(log_p)));
  return detail_inverse::gamma_solve(a, false, log_p, math::log(-Kokkos::expm1(log_p)));
}

LAPLACE_HD double gamma_q_inv_logq(double a, double log_q) {
  const double nan = math::detail::quiet_nan();
  if (!(a > 0.0) || !(a < math::detail::pos_inf()) || !(log_q <= 0.0)) return nan;
  if (log_q == 0.0) return 0.0;
  if (log_q == -math::detail::pos_inf()) return math::detail::pos_inf();
  if (log_q < -detail_inverse::ln2)
    return detail_inverse::gamma_solve(a, false, Kokkos::log1p(-math::exp(log_q)), log_q);
  return detail_inverse::gamma_solve(a, true, math::log(-Kokkos::expm1(log_q)), log_q);
}

LAPLACE_HD beta_inv_result ibeta_inv(double a, double b, double p) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(a < inf) || !(b > 0.0) || !(b < inf) || !(p >= 0.0) || !(p <= 1.0)) return {nan, nan};
  if (p == 0.0) return {0.0, 1.0};
  if (p == 1.0) return {1.0, 0.0};
  if (p <= 0.5) return detail_inverse::beta_solve(a, b, true, math::log(p), Kokkos::log1p(-p));
  const double q = 1.0 - p;  // exact
  return detail_inverse::beta_solve(a, b, false, Kokkos::log1p(-q), math::log(q));
}

LAPLACE_HD beta_inv_result ibetac_inv(double a, double b, double q) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(a < inf) || !(b > 0.0) || !(b < inf) || !(q >= 0.0) || !(q <= 1.0)) return {nan, nan};
  if (q == 0.0) return {1.0, 0.0};
  if (q == 1.0) return {0.0, 1.0};
  if (q < 0.5) return detail_inverse::beta_solve(a, b, false, Kokkos::log1p(-q), math::log(q));
  const double p = 1.0 - q;  // exact
  return detail_inverse::beta_solve(a, b, true, math::log(p), Kokkos::log1p(-p));
}

LAPLACE_HD beta_inv_result ibeta_inv_logp(double a, double b, double log_p) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(a < inf) || !(b > 0.0) || !(b < inf) || !(log_p <= 0.0)) return {nan, nan};
  if (log_p == 0.0) return {1.0, 0.0};
  if (log_p == -inf) return {0.0, 1.0};
  if (log_p <= -detail_inverse::ln2)
    return detail_inverse::beta_solve(a, b, true, log_p, Kokkos::log1p(-math::exp(log_p)));
  return detail_inverse::beta_solve(a, b, false, log_p, math::log(-Kokkos::expm1(log_p)));
}

LAPLACE_HD beta_inv_result ibetac_inv_logq(double a, double b, double log_q) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(a < inf) || !(b > 0.0) || !(b < inf) || !(log_q <= 0.0)) return {nan, nan};
  if (log_q == 0.0) return {0.0, 1.0};
  if (log_q == -inf) return {1.0, 0.0};
  if (log_q < -detail_inverse::ln2)
    return detail_inverse::beta_solve(a, b, false, Kokkos::log1p(-math::exp(log_q)), log_q);
  return detail_inverse::beta_solve(a, b, true, math::log(-Kokkos::expm1(log_q)), log_q);
}

}  // namespace own

// What the rest of Laplace calls: the own versions (docs/numerics.md).
LAPLACE_HD_FORCEINLINE double gamma_p_inv(double a, double p) { return own::gamma_p_inv(a, p); }
LAPLACE_HD_FORCEINLINE double gamma_q_inv(double a, double q) { return own::gamma_q_inv(a, q); }
LAPLACE_HD_FORCEINLINE double gamma_p_inv_logp(double a, double log_p) { return own::gamma_p_inv_logp(a, log_p); }
LAPLACE_HD_FORCEINLINE double gamma_q_inv_logq(double a, double log_q) { return own::gamma_q_inv_logq(a, log_q); }
LAPLACE_HD_FORCEINLINE double ibeta_inv(double a, double b, double p) { return own::ibeta_inv(a, b, p).x; }
LAPLACE_HD_FORCEINLINE double ibetac_inv(double a, double b, double q) { return own::ibetac_inv(a, b, q).x; }
LAPLACE_HD_FORCEINLINE double ibeta_inv_logp(double a, double b, double log_p) {
  return own::ibeta_inv_logp(a, b, log_p).x;
}
LAPLACE_HD_FORCEINLINE double ibetac_inv_logq(double a, double b, double log_q) {
  return own::ibetac_inv_logq(a, b, log_q).x;
}

}  // namespace laplace::math
