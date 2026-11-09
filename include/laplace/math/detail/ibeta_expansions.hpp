// The expansions of the regularized incomplete beta function I_x(a, b) and the
// choice between them, after DiDonato and Morris (1992), "Algorithm 708:
// significant digit computation of the incomplete beta function ratios", ACM
// TOMS 18, 360-373, and Temme's uniform asymptotic expansion (Temme 1992, as
// used by DiDonato and Morris for both shapes large).
//
// Notation: a, b > 0 the shapes, s = a + b, x in (0, 1), y = 1 - x, and
// lambda = a - s x, which is positive when x lies below the mean a / s. Of x
// and y one is always exact (x is the argument; y = 1 - x is exact when
// x >= 1/2): their logarithms lx, ly are computed once from the exact one, and
// lambda from x with the sum and the product carried exactly, so that no
// expansion loses digits through a rounded 1 - x or a rounded mean.
//
// The pieces (each returns the logarithm of what it computes, so that nothing
// underflows):
//   bpser   power series in x (DiDonato-Morris BPSER, Boost ibeta_series), for
//           x small or b x small;
//   bup     I_x(a, b) - I_x(a + n, b), a finite sum (BUP, Boost ibeta_a_step);
//   bgrat   expansion for a large and b <= 1 in terms of the incomplete gamma
//           function Q(b, u) (BGRAT, Boost beta_small_b_large_a_series), with
//           Q from the branches of Laplace's own gamma_inc (gamma_q_ratio);
//   bfrac   continued fraction (BFRAC, Boost ibeta_fraction2);
//   basym   Temme's expansion for a, b > 100 near the mean (BASYM).
// The choice between them (direct_side) is that of DiDonato and Morris'
// BRATIO, except where measurement chose the continued fraction (thresholds
// below); it returns the side, I_x(a, b) or 1 - I_x(a, b), that it computes
// directly, which is never close to 1, so that the other is 1 minus it.
//
// What differs from Boost.Math (beta.hpp, ibeta_imp and the functions named
// above, read and adapted; Boost Software License 1.0): the prefactor
// x^a y^b / B(a, b) is written without cancellation (log_power below), as in
// the first own version of incomplete_beta.hpp, because in double precision
// Boost's prefactor loses the relative accuracy of the tails; and every piece
// works in logarithms.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/log1p.hpp>

// On the device direct_side is a subroutine, not inlined into the kernel that
// calls it. Measured (2026-11-09, CUDA 13.1, RTX 4070 SUPER): inlined, its
// large divergent body inside a Kokkos parallel_reduce left a warp
// unconverged at the reduction's block barrier (__syncthreads_or, BAR.RED),
// and the kernel stopped with cudaErrorIllegalInstruction; the same code in a
// parallel_for ran correctly. As a call, the warp reconverges at the return.
#if defined(__CUDACC__)
#define LAPLACE_IBX_NOINLINE __noinline__
#else
#define LAPLACE_IBX_NOINLINE
#endif

namespace laplace::math::own::detail_ibx {

constexpr double eps = 1.1102230246251565e-16;  // 2^-53

// S(q) = 1/3 + q/5 + q^2/7 + ... (q = r^2, r = atanh's argument), so that
// 2 atanh(r) = 2 r + 2 r^3 S(r^2). For 0 <= q <= 1/9 the series is cut where its
// remainder falls below 2^-60 of S: 8 terms for q <= 1/80, 12 for q <= 1/25,
// 17 up to 1/9.
LAPLACE_HD double atanh_tail(double q) {
  double S;
  if (q <= 0.0125) {
    S = 1.0 / 17.0;
  } else {
    if (q <= 0.04) {
      S = 1.0 / 25.0;
    } else {
      S = 1.0 / 35.0;
      S = S * q + 1.0 / 33.0;
      S = S * q + 1.0 / 31.0;
      S = S * q + 1.0 / 29.0;
      S = S * q + 1.0 / 27.0;
      S = S * q + 1.0 / 25.0;
    }
    S = S * q + 1.0 / 23.0;
    S = S * q + 1.0 / 21.0;
    S = S * q + 1.0 / 19.0;
    S = S * q + 1.0 / 17.0;
  }
  S = S * q + 1.0 / 15.0;
  S = S * q + 1.0 / 13.0;
  S = S * q + 1.0 / 11.0;
  S = S * q + 1.0 / 9.0;
  S = S * q + 1.0 / 7.0;
  S = S * q + 1.0 / 5.0;
  S = S * q + 1.0 / 3.0;
  return S;
}

// L(t) = log(1 + t) - t, t > -1. For -1/2 <= t <= 1 through r = t / (2 + t),
// |r| <= 1/3: log(1 + t) = 2 atanh(r) = 2 (r + r^3 S(r^2)) and
// 2 r - t = -t^2 / (2 + t), so that
//     L(t) = -t^2 / (2 + t) + 2 r^3 S(r^2),
// a sum without cancellation (the second term is at most t/6 of the first).
// Faster than Boost's log1pmx, whose series needs up to 50 divisions at
// |t| = 1/2. Elsewhere log1p(t) - t.
LAPLACE_HD double log1pmx(double t) {
  if (t < -0.5 || t > 1.0) return Kokkos::log1p(t) - t;
  const double inv = 1.0 / (2.0 + t), r = t * inv, r2 = r * r;
  return -t * t * inv + 2.0 * r * r2 * atanh_tail(r2);
}

// log(1 + t) for -1/2 <= t <= 1 as t + L(t), without a call to the libm
LAPLACE_HD_FORCEINLINE double log1p_poly(double t) { return t + log1pmx(t); }

// log Gamma*(z) = log Gamma(z) - (z - 1/2) log z + z - log sqrt(2 pi), z > 0:
// its series for z >= 10 (incomplete_gamma.hpp); below 10 the recurrence
//     log Gamma*(z) = log Gamma*(z + 1) + (z + 1/2) log(1 + 1/z) - 1,
// whose step is, with r = 1/(2z + 1) and log(1 + 1/z) = 2 atanh(r),
//     (z + 1/2) log(1 + 1/z) - 1 = r^2 S(r^2) = r^2/3 + r^4/5 + r^6/7 + ...:
// positive terms, one division per step, no cancellation (relative error of
// a few 2^-53 in the sum for z >= 1); at most 9 steps from z = 1. For z < 1
// (which the expansions do not use) the first step is taken directly, with
// some cancellation.
LAPLACE_HD double lgs(double z) {
  double acc = 0.0;
  if (z < 1.0) {
    acc = (z + 0.5) * Kokkos::log1p(1.0 / z) - 1.0;
    z += 1.0;
  }
  for (int k = 0; k < 64 && z < 10.0; ++k) {
    const double r = 1.0 / (2.0 * z + 1.0), q = r * r;
    acc += q * atanh_tail(q);
    z += 1.0;
  }
  return acc + detail_igamma::log_gamma_star(z);
}

// p + e = u v exactly (the FMA on the device; Dekker's product on the host,
// where the build does not assume FMA hardware)
LAPLACE_HD_FORCEINLINE void two_prod(double u, double v, double& p, double& e) {
  p = u * v;
  KOKKOS_IF_ON_DEVICE((e = Kokkos::fma(u, v, -p);))
  KOKKOS_IF_ON_HOST((constexpr double split = 134217729.0;  // 2^27 + 1
                     const double cu = split * u, uh = cu - (cu - u), ul = u - uh;
                     const double cv = split * v, vh = cv - (cv - v), vl = v - vh;
                     e = ((uh * vh - p) + uh * vl + ul * vh) + ul * vl;))
}

// lambda = a - (a + b) x, with a + b and (a + b) x carried exactly: near the
// mean the subtraction a - (a + b) x is then exact (Sterbenz)
LAPLACE_HD double lambda_of(double a, double b, double x) {
  const double s = a + b, bb = s - a, es = (a - (s - bb)) + (b - bb);  // s + es = a + b
  double p, ep;
  two_prod(s, x, p, ep);  // p + ep = s x
  return ((a - p) - ep) - es * x;
}

// One evaluation: shapes, x and y = 1 - x, their logarithms, lambda for these
// shapes, and which of x and y is exact (at least one is).
struct args {
  double a, b, x, y, lx, ly, lambda;
  bool xe, ye;
};

// lambda = a - (a + b) x for the shapes of p, from the exact one of x and y
LAPLACE_HD_FORCEINLINE double lambda_for(const args& p, double a, double b) {
  return p.xe ? lambda_of(a, b, p.x) : -lambda_of(b, a, p.y);
}

// the same x, y with shapes a, b
LAPLACE_HD_FORCEINLINE args reshaped(const args& p, double a, double b) {
  args q = p;
  q.a = a;
  q.b = b;
  q.lambda = lambda_for(p, a, b);
  return q;
}

// I_y(b, a) = 1 - I_x(a, b)
LAPLACE_HD_FORCEINLINE args swapped(const args& p) {
  return {p.b, p.a, p.y, p.x, p.ly, p.lx, -p.lambda, p.ye, p.xe};
}

// log Gamma(1 + a), accurate near a = 0 and 1
LAPLACE_HD_FORCEINLINE double lgamma1p(double a) {
  return Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
}

// log(e^u + e^v)
LAPLACE_HD_FORCEINLINE double log_add(double u, double v) {
  const double hi = u > v ? u : v, lo = u > v ? v : u;
  if (lo - hi < -40.0) return hi;  // e^-40 < 2^-57
  return hi + Kokkos::log1p(math::exp(lo - hi));
}

// log(Gamma(a + b) / (Gamma(b) b^a)), b >= 1, a > 0: small terms only
LAPLACE_HD double log_gamma_ratio(double a, double b) {
  const double t = a / b;
  const double l1p = t <= 1.0 ? log1p_poly(t) : Kokkos::log1p(t);
  return b * log1pmx(t) + (a - 0.5) * l1p + lgs(a + b) - lgs(b);
}

// log(x^a y^b / B(a, b)), written without cancellation:
//   a, b >= 1: with u = -lambda/a = x/p0 - 1 and v = lambda/b = y/q0 - 1
//     (p0 = a/s, q0 = b/s, so that a u + b v = 0),
//     a L(u) + b L(v) + log(sqrt(a b / s) / sqrt(2 pi)) + lgs(s) - lgs(a) - lgs(b),
//     L(t) = log(1 + t) - t; u and v come from lambda, not from a rounded mean;
//   one shape below 1, the other (b, say) at least 1, with
//     log Gamma(a) = lgamma1p(a) - log a:
//     a log(b x) + b log y + log a - lgamma1p(a) + log(Gamma(a + b) / (Gamma(b) b^a));
//   both below 1: a log x + b log y - log B(a, b), B from Boost (|log B| < 6.3
//     for shapes down to 2^-8).
LAPLACE_HD double log_power(const args& p) {
  const double a = p.a, b = p.b;
  if (a >= 1.0 && b >= 1.0) {
    const double s = a + b;
    const double u = -p.lambda / a, v = p.lambda / b;
    // L(u), L(v); below -1/2 log(1 + u) = log(x / p0), from lx (1 + u would lose a small x)
    const double Lu = u >= -0.5 ? log1pmx(u) : p.lx - math::log(a / s) - u;
    const double Lv = v >= -0.5 ? log1pmx(v) : p.ly - math::log(b / s) - v;
    return a * Lu + b * Lv + 0.5 * math::log(a * b / s) - detail_igamma::half_log_two_pi + lgs(s) - lgs(a) - lgs(b);
  }
  if (a < 1.0 && b >= 1.0) {
    const double bx = b * p.x;
    const double l_bx = bx > 1e-290 ? math::log(bx) : math::log(b) + p.lx;
    return a * l_bx + b * p.ly + math::log(a) - lgamma1p(a) + log_gamma_ratio(a, b);
  }
  if (b < 1.0 && a >= 1.0) {
    const double ay = a * p.y;
    const double l_ay = ay > 1e-290 ? math::log(ay) : math::log(a) + p.ly;
    return b * l_ay + a * p.lx + math::log(b) - lgamma1p(b) + log_gamma_ratio(b, a);
  }
  const double B = boost::math::beta(a, b, math::detail::boost_policy());
  const double log_b = (B > 0.0 && B < math::detail::pos_inf()) ? math::log(B)
                                                                 : math::lgamma(a) + math::lgamma(b) - math::lgamma(a + b);
  return a * p.lx + b * p.ly - log_b;
}

// BPSER: I_x(a, b) = x^a / B(a, b) sum_{n>=0} (1 - b)_n x^n / (n! (a + n)),
// (1 - b)_n the rising factorial. Used for x <= 1/2 with b <= 1, or b x <= 0.7.
// For a <= 1 the sum is about 1/a and B(a, b) about 1/a: the two are taken
// together, as a sum and a B(a, b), so that their logarithms do not cancel.
LAPLACE_HD double bpser(const args& p) {
  const double a = p.a, b = p.b, x = p.x;
  double term = 1.0, sum = 1.0 / a, poch = 1.0 - b;
  for (int n = 1; n < 2000; ++n) {
    term *= poch * x / n;
    poch += 1.0;
    const double c = term / (a + n);
    sum += c;
    if (Kokkos::fabs(c) <= eps * Kokkos::fabs(sum)) break;
  }
  if (a <= 1.0) {
    if (b < 1.0) {
      const double aB = a * boost::math::beta(a, b, math::detail::boost_policy());  // Gamma(1 + a) Gamma(b) / Gamma(a + b)
      if (aB > 0.0 && aB < math::detail::pos_inf()) return a * p.lx + math::log((a * sum) / aB);
    } else {
      const double bx = b * x;
      const double l_bx = bx > 1e-290 ? math::log(bx) : math::log(b) + p.lx;
      return a * l_bx - lgamma1p(a) + log_gamma_ratio(a, b) + math::log(a * sum);
    }
  }
  return log_power(p) - b * p.ly + math::log(sum);
}

// BUP: I_x(a, b) - I_x(a + n, b) = sum_{i=0}^{n-1} t_i, n >= 1, with
// t_i = x^(a+i) y^b / ((a + i) B(a + i, b)) and t_{i+1} / t_i = (a + b + i) x / (a + 1 + i).
// The terms can grow before they fall, by as much as e^(b x) (DiDonato and
// Morris scale them for this reason): the sum starts from the largest term,
// whose prefactor log_power computes without cancellation, and goes outwards,
// so that no large exponent cancels against a large sum.
LAPLACE_HD double bup(const args& p, int n) {
  const double a = p.a, b = p.b, x = p.x, y = p.y;
  const double i0 = ((a + b) * x - a - 1.0) / y;  // the terms grow while i < i0
  const int k = i0 <= 0.0 ? 0 : (i0 >= n - 1.0 ? n - 1 : static_cast<int>(Kokkos::ceil(i0)));
  const double log_tk = log_power(k == 0 ? p : reshaped(p, a + k, b)) - math::log(a + k);
  double d = 1.0, sum = 1.0;
  for (int i = k; i < n - 1; ++i) {  // t_{i+1} / t_k
    d *= (a + b + i) * x / (a + 1.0 + i);
    sum += d;
  }
  d = 1.0;
  for (int i = k; i > 0; --i) {  // t_{i-1} / t_k
    d *= (a + i) / ((a + b + i - 1.0) * x);
    sum += d;
  }
  return log_tk + math::log(sum);
}

// Q(b, u) and H / Q(b, u), H = u^b e^-u / Gamma(b), for 0 < b <= 1, u > 0: the
// branches of Laplace's own gamma_inc (incomplete_gamma.hpp, same series and
// continued fraction, same stopping tests), arranged so that BGRAT gets the
// ratio without the exp and log it would otherwise need: in the continued
// fraction branch Q = H h, h the value of the fraction, so H / Q = 1 / h.
struct q_ratio {
  double log_q;   // log Q(b, u)
  double h_over_q;  // H / Q(b, u)
};
LAPLACE_HD q_ratio gamma_q_ratio(double b, double u) {
  constexpr int max_iter = 1000000;
  if (b < 1.0 && u < 1.5) {
    // J = b sum_{n>=1} (-u)^n / (n! (b + n)), g = u^b / Gamma(b + 1), Q = -expm1(log g) - g J
    double term = 1.0, J = 0.0;
    for (int n = 1; n < 200; ++n) {
      term *= -u / n;
      const double c = term / (b + n);
      J += c;
      if (Kokkos::fabs(c) <= eps * Kokkos::fabs(J)) break;
    }
    J *= b;
    const double log_g = b * math::log(u) - lgamma1p(b);
    const double g = math::exp(log_g);
    const double q = -Kokkos::expm1(log_g) - g * J;
    return {math::log(q), b * g * math::exp(-u) / q};
  }
  const double log_d = detail_igamma::log_prefactor(b, u);  // log(u^b e^-u / Gamma(b + 1))
  if (u < b + 1.0) {
    // P = d sum_{n>=0} u^n / ((b+1)...(b+n)), Q = 1 - P
    double term = 1.0, s = 1.0;
    for (int n = 1; n < max_iter; ++n) {
      term *= u / (b + n);
      s += term;
      if (term <= eps * s) break;
    }
    const double d = math::exp(log_d);
    const double q = 1.0 - d * s;
    return {math::log(q), b * d / q};
  }
  // Legendre's continued fraction, Q = H h (modified Lentz)
  constexpr double tiny = 1e-300;
  double bb = u + 1.0 - b, c = 1.0 / tiny, d = 1.0 / bb, h = d;
  for (int i = 1; i < max_iter; ++i) {
    const double an = -i * (i - b);
    bb += 2.0;
    d = an * d + bb;
    if (Kokkos::fabs(d) < tiny) d = tiny;
    c = bb + an / c;
    if (Kokkos::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    // 4 eps: with FMA contraction (CUDA) delta can settle one ulp away from 1
    if (Kokkos::fabs(delta - 1.0) <= 4.0 * eps) break;
  }
  // H = b d, so Q = b d h
  return {log_d + math::log(b) + math::log(h), 1.0 / h};
}

// BGRAT (DiDonato and Morris, eqs. 9-9.6; Boost beta_small_b_large_a_series):
// I_x(a, b) for a >= 15, 0 < b <= 1 and x close to 1. With T = a + (b - 1)/2,
// u = -T log x,
//   I_x(a, b) = Gamma(a + b) / (Gamma(a) T^b) Q(b, u) sum_{n>=0} p_n J_n / J_0,
// J_0 = Q(b, u) / H, H = u^b e^-u / Gamma(b), and J_n, p_n from the recurrences
// of eqs. 9.4 and 9.6. The sum is normalized by J_0 so that a huge J_0 (u tiny)
// or a tiny Q (u large) do not overflow or underflow.
LAPLACE_HD double bgrat(const args& p) {
  const double a = p.a, b = p.b, lx = p.lx;
  const double T = a + 0.5 * (b - 1.0);
  const double u = -T * lx;
  const q_ratio g = gamma_q_ratio(b, u);
  // log(Gamma(a + b) / (Gamma(a) T^b)), small terms only
  const double log_r = a * log1pmx(b / a) - 0.5 * log1p_poly(b / a) + b * log1p_poly(0.5 * (b + 1.0) / T) +
                       lgs(a + b) - lgs(a);  // b / a <= 1/15 and (b + 1) / (2 T) < 1/7 here
  const double inv_j0 = g.h_over_q;  // 1 / J_0
  constexpr int nmax = 30;
  double pn[nmax + 1], inv_fact_odd[nmax + 1];  // p_n, 1 / (2n + 1)!
  pn[0] = 1.0;
  inv_fact_odd[0] = 1.0;
  const double bm1 = b - 1.0, lx2 = 0.25 * lx * lx, t4 = 4.0 * T * T;
  double jr = 1.0, sum = 1.0, lxp = 1.0, b2n = b;  // jr = J_n / J_0, lxp = (lx / 2)^(2n - 2)
  for (int n = 1; n <= nmax; ++n) {
    inv_fact_odd[n] = inv_fact_odd[n - 1] / ((2.0 * n) * (2.0 * n + 1.0));
    double acc = 0.0;
    for (int m = 1; m < n; ++m) acc += (m * b - n) * pn[n - m] * inv_fact_odd[m];
    pn[n] = acc / n + bm1 * inv_fact_odd[n];
    jr = (b2n * (b2n + 1.0) * jr + (u + b2n + 1.0) * lxp * inv_j0) / t4;
    lxp *= lx2;
    b2n += 2.0;
    const double r = pn[n] * jr;
    sum += r;
    if (Kokkos::fabs(r) <= eps * Kokkos::fabs(sum)) break;
  }
  return log_r + g.log_q + math::log(sum);
}

// BFRAC (DiDonato and Morris; Boost ibeta_fraction2): I_x(a, b) =
// x^a y^b / B(a, b) / (b_0 + a_1 / (b_1 + a_2 / (b_2 + ...))), with
//   a_m = m (a + m - 1) (a + b + m - 1) (b - m) x^2 / (a + 2m - 1)^2,
//   b_m = m + m (b - m) x / (a + 2m - 1) + (a + m) (lambda + 1 + m (2 - x)) / (a + 2m + 1),
// by the modified Lentz method; for lambda >= 0.
LAPLACE_HD double bfrac(const args& p) {
  const double a = p.a, b = p.b, x = p.x, lam = p.lambda;
  constexpr double tiny = 1e-300;
  double f = a * (lam + 1.0) / (a + 1.0);
  if (f == 0.0) f = tiny;
  double C = f, D = 0.0;
  for (int m = 1; m < 100000; ++m) {
    const double denom = a + 2.0 * m - 1.0;
    const double am = (m * (a + m - 1.0) / denom) * ((a + b + m - 1.0) / denom) * (b - m) * x * x;
    const double bm = m + m * (b - m) * x / denom + (a + m) * (lam + 1.0 + m * (2.0 - x)) / (a + 2.0 * m + 1.0);
    D = bm + am * D;
    if (Kokkos::fabs(D) < tiny) D = tiny;
    C = bm + am / C;
    if (Kokkos::fabs(C) < tiny) C = tiny;
    D = 1.0 / D;
    const double delta = C * D;
    f *= delta;
    // 4 eps: with FMA contraction (CUDA) delta can settle one ulp away from 1
    if (Kokkos::fabs(delta - 1.0) <= 4.0 * eps) break;
  }
  return log_power(p) - math::log(f);
}

// exp(z^2) erfc(z) for z >= 0, given f = z^2
LAPLACE_HD double erfcx(double z, double f) {
  if (z < 25.0) return math::exp(f) * Kokkos::erfc(z);
  const double w = 1.0 / (2.0 * f);  // asymptotic series, terms (2k - 1)!! / (2 z^2)^k
  return (1.0 - w * (1.0 - 3.0 * w * (1.0 - 5.0 * w * (1.0 - 7.0 * w)))) / (z * 1.77245385090551602730);
}

// BASYM: Temme's uniform asymptotic expansion as given by DiDonato and Morris
// (section 5), for a, b > 100 and 0 <= lambda <= 0.03 min(a, b):
//   I_x(a, b) = (2 / sqrt(pi)) e^-f exp(lgs(s) - lgs(a) - lgs(b)) sum_n d_n w0^n J_n,
// with f = -(a L(u) + b L(v)) as in log_power, J_n built from
// exp(f) erfc(sqrt f), and the coefficients d_n from the recurrences of the paper.
LAPLACE_HD double basym(const args& p) {
  constexpr int num = 20;
  constexpr double e0 = 1.12837916709551257390;  // 2 / sqrt(pi)
  constexpr double e1 = 0.35355339059327376220;  // 2^(-3/2)
  const double a = p.a, b = p.b, lam = p.lambda;
  double h, r0, r1, w0;
  if (a < b) {
    h = a / b;
    r0 = 1.0 / (1.0 + h);
    r1 = (b - a) / b;
    w0 = 1.0 / Kokkos::sqrt(a * (1.0 + h));
  } else {
    h = b / a;
    r0 = 1.0 / (1.0 + h);
    r1 = (b - a) / a;
    w0 = 1.0 / Kokkos::sqrt(b * (1.0 + h));
  }
  const double u = -lam / a, v = lam / b;
  const double f = -(a * log1pmx(u) + b * log1pmx(v));  // |u|, |v| <= 0.03 here
  const double z0 = Kokkos::sqrt(f), z = 0.5 * (z0 / e1), z2 = f + f;
  double a0[num + 2], b0[num + 2], c[num + 2], d[num + 2];  // 1-based
  a0[1] = (2.0 / 3.0) * r1;
  c[1] = -0.5 * a0[1];
  d[1] = -c[1];
  double j0 = (0.5 / e0) * erfcx(z0, f), j1 = e1;
  double sum = j0 + d[1] * w0 * j1;
  double s = 1.0, h2 = h * h, hn = 1.0, w = w0, znm1 = z, zn = z2;
  for (int n = 2; n <= num; n += 2) {
    hn *= h2;
    a0[n] = 2.0 * r0 * (1.0 + h * hn) / (n + 2.0);
    const int np1 = n + 1;
    s += hn;
    a0[np1] = 2.0 * r1 * s / (n + 3.0);
    for (int i = n; i <= np1; ++i) {
      const double r = -0.5 * (i + 1.0);
      b0[1] = r * a0[1];
      for (int m = 2; m <= i; ++m) {
        double bsum = 0.0;
        for (int j = 1; j < m; ++j) bsum += (j * r - (m - j)) * a0[j] * b0[m - j];
        b0[m] = r * a0[m] + bsum / m;
      }
      c[i] = b0[i] / (i + 1.0);
      double dsum = 0.0;
      for (int j = 1; j < i; ++j) dsum += d[i - j] * c[j];
      d[i] = -(dsum + c[i]);
    }
    j0 = e1 * znm1 + (n - 1.0) * j0;
    j1 = e1 * zn + n * j1;
    znm1 *= z2;
    zn *= z2;
    w *= w0;
    const double t0 = d[n] * w * j0;
    w *= w0;
    const double t1 = d[np1] * w * j1;
    sum += t0 + t1;
    if (Kokkos::fabs(t0) + Kokkos::fabs(t1) <= eps * sum) break;
  }
  return -f + math::log(e0) + lgs(a + b) - lgs(a) - lgs(b) + math::log(sum);
}

// The side computed directly: log_w = log I_x(a, b) when complement is false,
// log(1 - I_x(a, b)) when it is true. It is never close to 1.
struct side {
  double log_w;
  bool complement;
  int method;  // which piece computed it (for measurement): see direct_side
};

// Thresholds of the choice. BRATIO uses, for a, b > 1 and b < 40, bpser when
// b x <= 0.7 and otherwise bup followed by bpser (x <= 0.7) or by bgrat; with
// the prefactor of log_power the continued fraction is as accurate there
// (measured on 2026-11-09, docs/numerics.md: the same maximum errors on all
// four reference sets), faster on the CPU, and on the GPU it reduces the
// number of different paths in a warp. It is used instead of bpser and of
// bup + bpser everywhere, and instead of bup + bgrat for a < 500 (at a = 2501
// the fraction reached 6.5 ulp of 1 against 1.5 for bgrat). The defaults are
// these measured choices; the fields exist so that math_ibeta can vary them.
struct thresholds {
  double basym_min = 100.0;      // both shapes above it for BASYM
  double basym_lambda = 0.03;    // and lambda <= basym_lambda min(a, b)
  double bup_max_a = 0.0;        // a, b > 1, b < 40, b x > 0.7, x <= 0.7: bup + bpser if a <= this, else bfrac
  double bgrat_min_a = 500.0;    // a, b > 1, b < 40, x > 0.7: bup + bgrat if a >= this, else bfrac
  double bpser_max_bx = 0.0;     // a, b > 1, b < 40: bpser if b x <= this, else bfrac
};

// The choice, for a, b > 0 and 0 < x < 1. Methods (side::method): 1 bpser,
// 2 bpser of the complement, 3 bup + bgrat of the complement, 4 bgrat of the
// complement, 5 bup + bpser, 6 bup + bgrat, 7 bfrac, 8 basym (5 is not used
// with the default thresholds).
LAPLACE_IBX_NOINLINE LAPLACE_HD side direct_side(double a, double b, double x, const thresholds& th = thresholds{}) {
  const double y = 1.0 - x;
  const double lx = x < 0.5 ? math::log(x) : Kokkos::log1p(-y);
  const double ly = x < 0.5 ? Kokkos::log1p(-x) : math::log(y);
  args p{a, b, x, y, lx, ly, lambda_of(a, b, x), true, x >= 0.5};
  bool comp = false;
  if (Kokkos::fmin(a, b) <= 1.0) {
    if (x > 0.5) {
      p = swapped(p);
      comp = true;
    }
    const double a0 = p.a, b0 = p.b, x0 = p.x;
    const args q = swapped(p);  // I_{y0}(b0, a0) = 1 - I_{x0}(a0, b0)
    if (Kokkos::fmax(a0, b0) <= 1.0) {
      if (a0 >= Kokkos::fmin(0.2, b0) || math::exp(a0 * p.lx) <= 0.9) return {bpser(p), comp, 1};
      if (x0 >= 0.3) return {bpser(q), !comp, 2};
      return {log_add(bup(q, 20), bgrat(reshaped(q, b0 + 20.0, a0))), !comp, 3};
    }
    if (b0 <= 1.0) return {bpser(p), comp, 1};
    if (x0 >= 0.3) return {bpser(q), !comp, 2};
    if (x0 < 0.1 && math::exp(a0 * math::log(x0 * b0)) <= 0.7) return {bpser(p), comp, 1};
    if (b0 > 15.0) return {bgrat(q), !comp, 4};
    return {log_add(bup(q, 20), bgrat(reshaped(q, b0 + 20.0, a0))), !comp, 3};
  }
  // a, b > 1: compute the side below the mean (lambda >= 0)
  if (p.lambda < 0.0) {
    p = swapped(p);
    comp = true;
  }
  const double a0 = p.a, b0 = p.b, x0 = p.x;
  if (b0 < 40.0) {
    if (b0 * x0 <= th.bpser_max_bx) return {bpser(p), comp, 1};
    if (x0 <= 0.7 && a0 > th.bup_max_a) return {bfrac(p), comp, 7};
    if (x0 > 0.7 && a0 < th.bgrat_min_a) return {bfrac(p), comp, 7};
    // I_x0(a0, b0) = [I_y0(bbar, a0) - I_y0(bbar + n, a0)] + I_x0(a0, bbar), bbar in (0, 1]
    int n = static_cast<int>(b0);
    double bbar = b0 - n;
    if (bbar == 0.0) {
      --n;
      bbar = 1.0;
    }
    const double w = bup(reshaped(swapped(p), bbar, a0), n);
    args r = reshaped(p, a0, bbar);
    if (x0 <= 0.7) return {log_add(w, bpser(r)), comp, 5};
    double v = -math::detail::pos_inf();
    if (a0 <= 15.0) {
      v = bup(r, 20);
      r = reshaped(r, a0 + 20.0, bbar);
    }
    return {log_add(log_add(w, v), bgrat(r)), comp, 6};
  }
  const double mn = Kokkos::fmin(a0, b0);
  if (mn > th.basym_min && p.lambda <= th.basym_lambda * mn) return {basym(p), comp, 8};
  return {bfrac(p), comp, 7};
}

}  // namespace laplace::math::own::detail_ibx
