// The multivariate normal rectangle probability on the log scale.
//
//   P(a, b; Sigma) = Pr(a < X <= b),  X ~ N(0, Sigma),  a_i < b_i, a_i = -inf and b_i = +inf allowed.
//
// After standardization (z = x / sqrt(Sigma_ii), R the correlation matrix) the
// value is computed by the number m of constrained coordinates (those with a
// finite limit; the others are dropped):
//   m = 1  log(Phi(b) - Phi(a)) (log_interval);
//   m = 2  Genz's bivariate (Drezner and Wesolowsky 1990, Genz 2004) while its
//          result is at least 1e-3, its absolute error about 1e-16; below,
//          the principal axes X1 = al u + be v, X2 = al u - be v, the
//          outer variable the one whose inner limits move with slope at most 1,
//          the axis cut where the inner interval changes its active lines, and
//          on each piece an exponentially tilted separation of variables by a
//          fixed tanh-sinh rule (log_bvn_rect); a narrow coordinate by
//          Gauss-Legendre;
//   m >= 3 Genz's separation of variables, Gibson-Glasbey-Elston order,
//          tilted by Botev's (2017) minimax saddle point: for m <= 4 a tensor
//          tanh-sinh rule over the first m - 2 coordinates with the last two by
//          the bivariate, beyond a fixed rank-1 Korobov lattice with the
//          baker's transform over the first m - 1, the last by log_interval.
// Every route sums positive terms on the log scale (log-sum-exp), so the
// relative accuracy carries into the tail.
//
// Derivatives in the shift of the rectangle, of every order, by the recursion
// over the corner densities (log_mvn_rect_derivs); the moments and cumulants
// of the truncated law from the same tilted nodes, centred
// (log_mvn_trunc_cumulants, mvn_trunc_mean, mvn_trunc_cov); the derivatives of
// log P through those cumulants (log_mvn_rect_log_derivs).
//
// All functions are LAPLACE_HD, allocate nothing, and take their workspace from
// the caller.
#pragma once

#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/quadrature.hpp>

namespace laplace::math {

namespace detail_mvnormal {

constexpr double pi = 3.14159265358979323846;
constexpr double two_pi = 6.28318530717958647692;
constexpr double sqrt_two_pi = 2.50662827463100050242;
constexpr double half_log_2pi = 0.91893853320467274178;
constexpr double log_half = -0.69314718055994530942;

LAPLACE_HD_FORCEINLINE double inf() { return detail::pos_inf(); }
LAPLACE_HD_FORCEINLINE double nan() { return detail::quiet_nan(); }

// log(1 - exp(x)) for x <= 0
LAPLACE_HD_FORCEINLINE double log1mexp(double x) {
  return x > log_half ? math::log(-Kokkos::expm1(x)) : Kokkos::log1p(-math::exp(x));
}
// log(exp(x) + exp(y))
LAPLACE_HD_FORCEINLINE double logaddexp(double x, double y) {
  const double m = x > y ? x : y;
  if (m == -inf()) return -inf();
  return m + Kokkos::log1p(math::exp(-Kokkos::fabs(x - y)));
}

// Gauss-Legendre half rules of 6, 12 and 20 nodes (negative nodes, weights),
// mpmath at 50 digits; Genz's bivariate uses them symmetrically.
// clang-format off
#define LAPLACE_MVN_GL                                                                                        \
  -0.93246951420315203, -0.66120938646626451, -0.23861918608319691,                                           \
  0.17132449237917035, 0.36076157304813861, 0.46791393457269105,                                              \
  -0.98156063424671925, -0.90411725637047486, -0.76990267419430469, -0.58731795428661745,                    \
  -0.36783149899818019, -0.12523340851146892,                                                                 \
  0.047175336386511827, 0.10693932599531843, 0.16007832854334623, 0.20316742672306592,                       \
  0.23349253653835481, 0.24914704581340279,                                                                   \
  -0.99312859918509492, -0.96397192727791379, -0.91223442825132591, -0.83911697182221882,                    \
  -0.74633190646015079, -0.63605368072651503, -0.5108670019508271, -0.37370608871541956,                     \
  -0.22778585114164508, -0.076526521133497334,                                                                \
  0.017614007139152118, 0.040601429800386941, 0.062672048334109064, 0.083276741576704749,                    \
  0.10193011981724044, 0.11819453196151842, 0.13168863844917663, 0.14209610931838205,                        \
  0.14917298647260375, 0.15275338713072585
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_GLOBAL_TABLE double gl_dev[72] = {LAPLACE_MVN_GL};
#endif
inline constexpr double gl_host[72] = {LAPLACE_MVN_GL};
#undef LAPLACE_MVN_GL
LAPLACE_HD_FORCEINLINE double gl(int i) {
#if defined(__CUDACC__)
  KOKKOS_IF_ON_DEVICE((return gl_dev[i];))
  KOKKOS_IF_ON_HOST((return gl_host[i];))
#else
  return gl_host[i];
#endif
}
// offsets of the three rules: nodes at off, weights at off + n
LAPLACE_HD_FORCEINLINE int gl_offset(int n) { return n == 3 ? 0 : n == 6 ? 6 : 18; }

}  // namespace detail_mvnormal

/**
 * \ingroup mvnormal
 * \brief Log of the standard normal interval probability \f$\log(\Phi(b) - \Phi(a))\f$.
 *
 * Infinite limits allowed. Narrow intervals (\f$h \max(1, |m|) \le 1\f$ with
 * \f$m, h\f$ the midpoint and half-width) by 16-point Gauss-Legendre of
 * \f$\phi\f$ relative to \f$\phi(m)\f$; one-sided intervals through \f$\log\Phi\f$
 * and \f$\log(1 - e^x)\f$; intervals across zero by \f$1 - \Phi(a) - \Phi(-b)\f$.
 * The width \f$b - a\f$ is passed on its own, for a caller that knows it more
 * precisely than the difference of the rounded limits (a narrow interval moved
 * by a large shift).
 *
 * \return \f$-\infty\f$ when \f$a \ge b\f$; NaN for a NaN limit.
 */
LAPLACE_HD double log_interval(double a, double b, double width) {
  using namespace detail_mvnormal;
  if (a != a || b != b) return nan();
  if (!(a < b) || !(width > 0.0)) return -inf();
  if (a == -inf()) return log_Phi(b);
  if (b == inf()) return log_Phi(-a);
  const double h = 0.5 * width, m = a + h;
  if (h * Kokkos::fmax(1.0, Kokkos::fabs(m)) <= 1.0) {
    double s = 0.0;
    for (int i = 0; i < 16; ++i) {
      const double x = h * quad::detail::legendre16(i);
      s += quad::detail::legendre16(16 + i) * math::exp(-(m * x + 0.5 * x * x));
    }
    return -0.5 * m * m - half_log_2pi + math::log(h * s);
  }
  if (b <= 0.0) {
    const double la = log_Phi(a), lb = log_Phi(b);
    return lb + log1mexp(la - lb);
  }
  if (a >= 0.0) {
    const double la = log_Phi(-b), lb = log_Phi(-a);
    return lb + log1mexp(la - lb);
  }
  return math::log(1.0 - Phi(a) - Phi(-b));
}
/// \ingroup mvnormal
/// \brief `log_interval(a, b, b - a)`.
LAPLACE_HD double log_interval(double a, double b) { return log_interval(a, b, b - a); }

namespace detail_mvnormal {

// The point x of the interval (a, b) with Phi(x) = Phi(a) + t (Phi(b) - Phi(a)),
// from t, its complement tc = 1 - t, lpa = log Phi(a), lqb = log Phi(-b) and
// lE = log(Phi(b) - Phi(a)): through log Phi(x) or log Phi(-x), whichever is
// smaller, so that both tails keep their relative precision.
LAPLACE_HD_FORCEINLINE double interval_point(double lpa, double lqb, double lE, double t, double tc) {
  const double lp = logaddexp(lpa, math::log(t) + lE);
  const double lq = logaddexp(lqb, math::log(tc) + lE);
  return lp <= lq ? Phi_inv_logp(lp) : -Phi_inv_logp(lq);
}

// The tanh-sinh rule on (0, 1) with 2 M + 1 nodes at step h = t_max / M:
// node j gives t, tc = 1 - t and the weight.
constexpr double ts_t_max = 3.2;
LAPLACE_HD_FORCEINLINE void ts_node(int j, int M, double& t, double& tc, double& w) {
  const double h = ts_t_max / M, s = j * h;
  const double q = 0.5 * pi * Kokkos::sinh(s);
  const double e = math::exp(-2.0 * Kokkos::fabs(q));
  const double d = e / (1.0 + e);  // distance to the nearer end
  t = q < 0.0 ? d : 1.0 - d;
  tc = q < 0.0 ? 1.0 - d : d;
  w = h * 0.5 * pi * Kokkos::cosh(s) * 2.0 * e / ((1.0 + e) * (1.0 + e));
}

// running log-sum-exp
struct log_sum {
  double m = -detail::pos_inf(), s = 0.0;
  LAPLACE_HD_FORCEINLINE void add(double lv) {
    if (!(lv > -detail::pos_inf())) return;
    if (lv > m) {
      s = s * math::exp(m - lv) + 1.0;
      m = lv;
    } else {
      s += math::exp(lv - m);
    }
  }
  LAPLACE_HD_FORCEINLINE double value() const { return m + math::log(s); }
};

// Genz's bivariate upper orthant Pr(X > h, Y > k), correlation r, |r| < 1,
// h and k finite: Drezner and Wesolowsky's (1990) integral in arcsin(r) for
// |r| < 0.925, the expansion of Genz (2004, section 2.2) beyond; absolute error
// about 1e-16.
LAPLACE_HD double bvnu_genz(double h, double k, double r) {
  const double ar = Kokkos::fabs(r);
  const int n = ar < 0.3 ? 3 : ar < 0.75 ? 6 : 10;
  const int off = gl_offset(n);
  double hk = h * k, bvn = 0.0;
  if (ar < 0.925) {
    const double hs = 0.5 * (h * h + k * k), asr = Kokkos::asin(r);
    for (int i = 0; i < n; ++i) {
      const double x = gl(off + i), w = gl(off + n + i);
      for (int s = -1; s <= 1; s += 2) {
        const double sn = Kokkos::sin(0.5 * asr * (s * x + 1.0));
        bvn += w * math::exp((sn * hk - hs) / (1.0 - sn * sn));
      }
    }
    return bvn * asr / (2.0 * two_pi) + Phi(-h) * Phi(-k);
  }
  if (r < 0.0) {
    k = -k;
    hk = -hk;
  }
  if (ar < 1.0) {
    const double as = (1.0 - r) * (1.0 + r);
    double a = Kokkos::sqrt(as);
    const double bs = (h - k) * (h - k), c = (4.0 - hk) / 8.0, d = (12.0 - hk) / 16.0;
    double asr = -0.5 * (bs / as + hk);
    if (asr > -100.0) bvn = a * math::exp(asr) * (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as * as / 5.0);
    if (hk > -100.0) {
      const double b = Kokkos::sqrt(bs);
      bvn -= math::exp(-0.5 * hk) * sqrt_two_pi * Phi(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
    }
    a *= 0.5;
    for (int i = 0; i < n; ++i) {
      const double x = gl(off + i), w = gl(off + n + i);
      for (int s = -1; s <= 1; s += 2) {
        const double xs = (a * (s * x + 1.0)) * (a * (s * x + 1.0));
        const double rs = Kokkos::sqrt(1.0 - xs);
        asr = -0.5 * (bs / xs + hk);
        if (asr > -100.0)
          bvn += a * w * math::exp(asr) * (math::exp(-hk * xs / (2.0 * (1.0 + rs) * (1.0 + rs))) / rs - (1.0 + c * xs * (1.0 + d * xs)));
      }
    }
    bvn = -bvn / two_pi;
  }
  if (r > 0.0) return bvn + Phi(-(h > k ? h : k));
  if (h >= k) return -bvn;
  const double L = h < 0.0 ? Phi(k) - Phi(h) : Phi(-h) - Phi(-k);
  return L - bvn;
}

// Pr(X1 <= h, X2 <= k) by Genz's bivariate, infinite limits allowed
LAPLACE_HD double bvn_lower_genz(double h, double k, double r) {
  if (h == -inf() || k == -inf()) return 0.0;
  if (h == inf()) return k == inf() ? 1.0 : Phi(k);
  if (k == inf()) return Phi(h);
  return bvnu_genz(-h, -k, r);
}
// the rectangle by inclusion-exclusion of Genz's lower orthants: absolute
// error about 1e-16 (the fast inner probability of the outer integrals, whose
// total is not small)
LAPLACE_HD double bvn_rect_genz(double a1, double b1, double a2, double b2, double r) {
  return bvn_lower_genz(b1, b2, r) - bvn_lower_genz(a1, b2, r) - bvn_lower_genz(b1, a2, r) + bvn_lower_genz(a1, a2, r);
}

// The one-dimensional integral P = int_a^b phi(x) exp(lD(x)) dx on the log
// scale, lD the log of a log-concave function, by the separation of variables
// over a TILTED normal: N(mu, w^2) matched to the integrand's log, g(x) =
// -x^2/2 + lD(x), at its maximum x0 on [a, b] (w^2 = -1/g''(x0),
// mu = x0 + w^2 g'(x0), which is x0 when the maximum is inside), so that the
// ratio of the integrand to the tilted density is flat to second order at x0:
//   P = E int_0^1 exp(-x^2/2 + z^2/2 + lD(x) + log w) dt,  x = mu + w z,
// z the quantile of the truncated standard normal on ((a - mu)/w, (b - mu)/w].
// g' and g'' by central differences of g at a step of the local scale (they
// only choose the tilt); the maximum by safeguarded Newton.
// int_a^b phi(x) exp(lD(x)) dx over the truncated N(mu, 1) on (a, b]:
// E int_0^1 exp(-x^2/2 + z^2/2 + lD(x)) dt, z = x - mu, log scale
template <int M, class LD>
LAPLACE_HD double tilted_piece(double a, double b, double mu, const LD& lD) {
  const double al = a == -inf() ? -inf() : a - mu, be = b == inf() ? inf() : b - mu;
  const double lE = log_interval(al, be, b - a);
  if (!(lE > -inf())) return lE;
  const double lpa = al == -inf() ? -inf() : log_Phi(al);
  const double lqb = be == inf() ? -inf() : log_Phi(-be);
  log_sum acc;
  for (int j = -M; j <= M; ++j) {
    double t, tc, wt;
    ts_node(j, M, t, tc, wt);
    if (!(t > 0.0) || !(tc > 0.0)) continue;
    const double z = interval_point(lpa, lqb, lE, t, tc);
    const double xx = mu + z;
    acc.add(math::log(wt) - 0.5 * xx * xx + 0.5 * z * z + lD(xx));
  }
  return lE + acc.value();
}

template <int M, bool SplitAtMax, class LD>
LAPLACE_HD double log_tilted_integral(double a, double b, const LD& lD, const double* cuts = nullptr, int ncuts = 0) {
  auto g = [&](double x) { return -0.5 * x * x + lD(x); };
  if (a != -inf() && b != inf() && (b - a) * Kokkos::fmax(1.0, Kokkos::fmax(Kokkos::fabs(a), Kokkos::fabs(b))) <= 1.0) {
    // a short piece: the tanh-sinh rule in x itself, relative to the midpoint
    const double xm = 0.5 * (a + b), gm = g(xm), L = b - a;
    log_sum acc;
    for (int j = -M; j <= M; ++j) {
      double t, tc, wt;
      ts_node(j, M, t, tc, wt);
      if (!(t > 0.0) || !(tc > 0.0)) continue;
      const double x = t <= 0.5 ? a + L * t : b - L * tc;
      acc.add(math::log(wt * L) + g(x) - gm);
    }
    return gm - half_log_2pi + acc.value();
  }
  // a finite starting bracket and point
  double lo = a, hi = b;
  double x = a == -inf() ? (b == inf() ? 0.0 : b - 1.0) : (b == inf() ? a + 1.0 : 0.5 * (a + b));
  if (a == -inf() && b != inf() && b < 0.0) x = b;  // the lower tail: the end is close to the mode
  if (b == inf() && a != -inf() && a > 0.0) x = a;
  const double span = (a != -inf() && b != inf()) ? b - a : inf();
  double h = 1e-3 * Kokkos::fmin(1.0, span), g1 = 0.0, g2 = -1.0;
  for (int it = 0; it < 60; ++it) {
    const double gl = g(x - h), gc = g(x), gr = g(x + h);
    g1 = (gr - gl) / (2.0 * h);
    g2 = (gr - 2.0 * gc + gl) / (h * h);
    if (!(g2 < 0.0) || !(g2 > -1e300)) g2 = -1.0;
    if (g1 > 0.0) lo = x; else hi = x;
    double xn = x - g1 / g2;
    if (!(xn > lo) || !(xn < hi)) {
      // outside the bracket: bisect, or step out of an infinite side
      if (lo == -inf()) xn = x - 2.0 * (Kokkos::fabs(x) + 1.0);
      else if (hi == inf()) xn = x + 2.0 * (Kokkos::fabs(x) + 1.0);
      else xn = 0.5 * (lo + hi);
    }
    const double scale = 1.0 / Kokkos::sqrt(-g2);
    h = 1e-3 * Kokkos::fmin(scale, span);
    if (Kokkos::fabs(xn - x) <= 1e-9 * scale) {
      x = xn;
      break;
    }
    x = xn;
    // the maximum at an end of the interval
    if (a != -inf() && x <= a && g1 < 0.0) { x = a; break; }
    if (b != inf() && x >= b && g1 > 0.0) { x = b; break; }
  }
  if (a != -inf() && x < a) x = a;
  if (b != inf() && x > b) x = b;
  {
    // derivatives at x0, one-sided at an end
    const double hh = h;
    double xl = x - hh, xr = x + hh;
    if (a != -inf() && xl < a) xl = x, xr = x + 2.0 * hh;
    if (b != inf() && xr > b) xr = x, xl = x - 2.0 * hh;
    const double xm = 0.5 * (xl + xr), d = 0.5 * (xr - xl);
    const double gl = g(xl), gc = g(xm), gr = g(xr);
    g1 = (gr - gl) / (2.0 * d);
    g2 = (gr - 2.0 * gc + gl) / (d * d);
    g1 -= g2 * (xm - x);  // back to x
    if (!(g2 < 0.0) || !(g2 > -1e300)) g2 = -1.0;
  }
  // exponential tilt only (unit variance): the ratio exp(lD(x) - mu x) is then
  // log-concave with its maximum at x0, so the proposal has the heavier tails
  const double mu = x + g1;
  // split at an interior maximum: on each side the ratio decreases away from
  // x0, which the end clustering of the tanh-sinh rule resolves however narrow
  // the peak is
  // and at the caller's cuts inside (a, b), where the integrand changes
  // steeply: the ratio is monotone between consecutive cuts and the maximum
  double pt[8];
  int np = 0;
  if (SplitAtMax && (a == -inf() || x > a) && (b == inf() || x < b)) pt[np++] = x;
  for (int i = 0; i < ncuts && np < 8; ++i)
    if ((a == -inf() || cuts[i] > a) && (b == inf() || cuts[i] < b)) pt[np++] = cuts[i];
  for (int i = 1; i < np; ++i)
    for (int j = i; j > 0 && pt[j] < pt[j - 1]; --j) {
      const double t = pt[j];
      pt[j] = pt[j - 1], pt[j - 1] = t;
    }
  log_sum acc;
  for (int p = 0; p <= np; ++p) {
    const double lo = p == 0 ? a : pt[p - 1], hi = p == np ? b : pt[p];
    if (!(lo < hi)) continue;
    acc.add(tilted_piece<M>(lo, hi, mu, lD));
  }
  return acc.value();
}

// The bivariate rectangle in principal axes. X1 = al u + be v, X2 = al u - be v
// with al = sqrt((1 + r)/2), be = sqrt((1 - r)/2) and u, v independent standard
// normals. The outer variable o is v when r >= 0 and u when r < 0, so that the
// inner variable's limits, linear in o, have slopes of modulus at most 1
// (be/al or al/be): the inner probability never varies faster than phi(o).
// The inner interval is (max of two lower lines, min of two upper lines]; the
// o-axis is cut where two lower or two upper lines cross and where the interval
// closes, and each piece is a tilted one-dimensional integral.
// The outer integral over o of phi(o) times an inner probability whose first
// variable w (a standard normal independent of o) lies in (max of two lower
// lines, min of two upper lines]: lower_i(o) = c[i] + s[i] o, upper_i(o) =
// C[i] + s[i] o, wid[i] = C[i] - c[i] exactly. The o-axis is cut where two lower
// or two upper lines cross and where the interval closes; on each piece the
// active lines are fixed and inner(lo, hi, width, o) gives the log of the inner
// probability, integrated by log_tilted_integral with the caller's cuts.
template <int M, bool SplitAtMax, class Inner>
LAPLACE_HD double rot_pieces(const double* c, const double* C, const double* s, const double* wid, const Inner& inner,
                             const double* cuts, int ncuts) {
  double bp[4];
  int nb = 0;
  const double ds = s[0] - s[1];
  auto add = [&](double v) {
    if (v > -inf() && v < inf()) bp[nb++] = v;
  };
  if (ds != 0.0) {
    add((c[1] - c[0]) / ds);
    add((C[1] - C[0]) / ds);
    add((C[1] - c[0]) / ds);
    add((C[0] - c[1]) / -ds);
  }
  for (int i = 1; i < nb; ++i)
    for (int j = i; j > 0 && bp[j] < bp[j - 1]; --j) {
      const double t = bp[j];
      bp[j] = bp[j - 1], bp[j - 1] = t;
    }
  log_sum acc;
  for (int p = 0; p <= nb; ++p) {
    const double oa = p == 0 ? -inf() : bp[p - 1], ob = p == nb ? inf() : bp[p];
    if (!(oa < ob)) continue;
    // the active lines from a point inside the piece
    const double om = oa == -inf() ? (ob == inf() ? 0.0 : ob - 1.0) : ob == inf() ? oa + 1.0 : 0.5 * (oa + ob);
    const int il = (c[0] + s[0] * om >= c[1] + s[1] * om) ? 0 : 1;
    const int iu = (C[0] + s[0] * om <= C[1] + s[1] * om) ? 0 : 1;
    const double cl = c[il], sl = s[il], cu = C[iu], su = s[iu];
    const double lm = cl + sl * om, um = cu + su * om;
    if (!(lm < um)) continue;  // the interval is empty on this piece
    // the same coordinate's two lines: their distance is its width, exactly
    const double wd = il == iu ? wid[il] : -1.0;
    auto lD = [&](double o) {
      const double lo = cl == -inf() ? -inf() : cl + sl * o;
      const double hi = cu == inf() ? inf() : cu + su * o;
      const double w = wd > 0.0 && lo > -inf() && hi < inf() ? wd : hi - lo;
      return inner(lo, wd > 0.0 && lo > -inf() && hi < inf() ? lo + wd : hi, w, o);
    };
    acc.add(log_tilted_integral<M, SplitAtMax>(oa, ob, lD, cuts, ncuts));
  }
  return acc.value();
}

// the bivariate rectangle in principal axes: X1 = al u + be v, X2 = al u - be v
// with al = sqrt((1 + r)/2), be = sqrt((1 - r)/2), u and v independent standard
// normals; the outer variable is v when r >= 0 and u when r < 0, so that the
// inner limits move with slopes of modulus at most 1 (be/al or al/be): the
// inner probability never varies faster than phi of the outer variable
template <int M>
LAPLACE_HD double log_bvn_rot(double a1, double b1, double a2, double b2, double r, double w1, double w2) {
  const double al = Kokkos::sqrt(0.5 * (1.0 + r)), be = Kokkos::sqrt(0.5 * (1.0 - r));
  double c[2], C[2], s[2], wid[2];
  wid[0] = w1 / (r >= 0.0 ? al : be);
  wid[1] = w2 / (r >= 0.0 ? al : be);
  if (r >= 0.0) {
    // inner u: (a1 - be v)/al < u <= (b1 - be v)/al, (a2 + be v)/al < u <= (b2 + be v)/al
    c[0] = a1 / al, C[0] = b1 / al, s[0] = -be / al;
    c[1] = a2 / al, C[1] = b2 / al, s[1] = be / al;
  } else {
    // inner v: (a1 - al u)/be < v <= (b1 - al u)/be, (al u - b2)/be <= v < (al u - a2)/be
    c[0] = a1 / be, C[0] = b1 / be, s[0] = -al / be;
    c[1] = -b2 / be, C[1] = -a2 / be, s[1] = al / be;
  }
  auto inner = [](double lo, double hi, double w, double) { return log_interval(lo, hi, w); };
  return rot_pieces<M, false>(c, C, s, wid, inner, nullptr, 0);
}

}  // namespace detail_mvnormal

/// \ingroup mvnormal
/// \brief Threshold of the bivariate route: Genz's result is kept when it is at least this.
inline constexpr double bvn_genz_threshold = 1e-3;
/// \ingroup mvnormal
/// \brief Nodes of the tanh-sinh rule of the separation of variables: 2 M + 1.
inline constexpr int mvn_ts_half_nodes = 48;

/**
 * \ingroup mvnormal
 * \brief Log of the standard bivariate normal rectangle probability \f$\log\Pr(a_1 < X_1 \le b_1, a_2 < X_2 \le b_2)\f$.
 *
 * Unit variances, correlation \f$r \in [-1, 1]\f$, infinite limits allowed. The
 * widths \f$b_i - a_i\f$ are passed on their own (see `log_interval`).
 */
LAPLACE_HD double log_bvn_rect(double a1, double b1, double a2, double b2, double r, double w1, double w2) {
  using namespace detail_mvnormal;
  if (a1 != a1 || b1 != b1 || a2 != a2 || b2 != b2 || r != r || !(Kokkos::fabs(r) <= 1.0)) return nan();
  if (!(a1 < b1) || !(a2 < b2)) return -inf();
  const bool f1 = a1 == -inf() && b1 == inf(), f2 = a2 == -inf() && b2 == inf();
  if (f1) return log_interval(a2, b2);
  if (f2) return log_interval(a1, b1);
  if (r == 1.0) return log_interval(a1 > a2 ? a1 : a2, b1 < b2 ? b1 : b2);
  if (r == -1.0) return log_interval(a1 > -b2 ? a1 : -b2, b1 < -a2 ? b1 : -a2);
  const bool o1 = a1 == -inf() || b1 == inf(), o2 = a2 == -inf() || b2 == inf();
  double p = -1.0;
  if (o1 && o2) {
    // an orthant: upper orthant Pr(Y1 > h, Y2 > k) after sign flips
    const double h = a1 == -inf() ? -b1 : a1, k = a2 == -inf() ? -b2 : a2;
    const double rr = (a1 == -inf()) != (a2 == -inf()) ? -r : r;
    p = bvnu_genz(h, k, rr);
  } else {
    // a finite rectangle, or one coordinate one-sided and the other finite:
    // inclusion-exclusion of the lower orthants (bvn_rect_genz)
    p = bvn_rect_genz(a1, b1, a2, b2, r);
  }
  if (p >= bvn_genz_threshold) return math::log(p);
  {
    // a narrow coordinate, over which phi and the conditional probability of the
    // other barely move: 16-point Gauss-Legendre in it, relative to the midpoint,
    // the other coordinate's conditional width taken exactly
    const double s = Kokkos::sqrt((1.0 - r) * (1.0 + r));
    for (int pass = 0; pass < 2; ++pass) {
      const double A1 = pass ? a2 : a1, B1 = pass ? b2 : b1, A2 = pass ? a1 : a2, B2 = pass ? b1 : b2;
      const double W1 = pass ? w2 : w1, W2 = pass ? w1 : w2;
      if (A1 == -inf() || B1 == inf()) continue;
      const double h = 0.5 * W1, m = A1 + h;
      // the slope of log phi is |m|, that of the log conditional probability about
      // (|r|/s) times the larger of 1 and its limits' modulus
      const double lm = A2 == -inf() ? 0.0 : Kokkos::fabs((A2 - r * m) / s);
      const double hm = B2 == inf() ? 0.0 : Kokkos::fabs((B2 - r * m) / s);
      const double slope = Kokkos::fmax(Kokkos::fabs(m), Kokkos::fabs(r) / s * Kokkos::fmax(1.0, Kokkos::fmax(lm, hm)));
      if (!(h * Kokkos::fmax(1.0, slope) <= 1.0)) continue;
      const double ws = W2 / s;
      log_sum acc;
      for (int i = 0; i < 16; ++i) {
        const double x = h * quad::detail::legendre16(i), xx = m + x;
        const double lo = A2 == -inf() ? -inf() : (A2 - r * xx) / s, hi = B2 == inf() ? inf() : (B2 - r * xx) / s;
        const double lD = lo > -inf() && hi < inf() ? log_interval(lo, lo + ws, ws) : log_interval(lo, hi);
        acc.add(math::log(quad::detail::legendre16(16 + i)) - (m * x + 0.5 * x * x) + lD);
      }
      return -0.5 * m * m - half_log_2pi + math::log(h) + acc.value();
    }
  }
  return log_bvn_rot<mvn_ts_half_nodes>(a1, b1, a2, b2, r, w1, w2);
}
/// \ingroup mvnormal
/// \brief `log_bvn_rect` with the widths \f$b_i - a_i\f$.
LAPLACE_HD double log_bvn_rect(double a1, double b1, double a2, double b2, double r) {
  return log_bvn_rect(a1, b1, a2, b2, r, b1 - a1, b2 - a2);
}

namespace detail_mvnormal {

LAPLACE_HD_FORCEINLINE int pk(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

}  // namespace detail_mvnormal

/// \ingroup mvnormal
/// \brief Largest dimension of `log_mvn_rect` (the lattice generators are tabulated up to it).
inline constexpr int mvn_max_dim = 19;  // the lattice over m - 1 <= 18 coordinates, the generators' table

/**
 * \ingroup mvnormal
 * \brief Doubles of workspace needed by `log_mvn_rect` in dimension \f$k\f$.
 */
LAPLACE_HD constexpr int mvn_rect_scratch(int k) { return 5 * k * k + 16 * k; }

namespace detail_mvnormal {

// fixed rank-1 lattice: N points, the Korobov generator per dimension s of the
// lattice part (k - 2), its shift, and the baker's transform
LAPLACE_HD_FORCEINLINE double frac(double x) { return x - Kokkos::floor(x); }

// Korobov generators (tools/reference/mvnormal_lattice.py) for the primes
// 1021, 4093, 16381, 65521 and the lattice dimensions 1 .. 18 (weighted P_2
// criterion with gamma_j = 1/j^2, every generator searched)
// clang-format off
#define LAPLACE_MVN_KOROBOV                                                                      \
  1, 374, 467, 223, 223, 223, 223, 346, 208, 208, 346, 481, 228, 228, 228, 228, 228, 455,                 \
  1, 1210, 1276, 1515, 1802, 1802, 1905, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450,           \
  1, 3711, 5159, 5619, 5899, 2690, 2690, 3993, 6711, 6711, 6711, 8098, 5202, 5202, 372, 7957, 7957, 7957, \
  1, 18303, 23557, 18589, 12427, 23927, 5859, 19609, 19609, 5782, 19609, 11339, 25641, 4171, 4171, 17388, \
  17388, 17388
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_GLOBAL_TABLE int korobov_dev[72] = {LAPLACE_MVN_KOROBOV};
#endif
inline constexpr int korobov_host[72] = {LAPLACE_MVN_KOROBOV};
#undef LAPLACE_MVN_KOROBOV
LAPLACE_HD_FORCEINLINE int korobov(int i) {
#if defined(__CUDACC__)
  KOKKOS_IF_ON_DEVICE((return korobov_dev[i];))
  KOKKOS_IF_ON_HOST((return korobov_host[i];))
#else
  return korobov_host[i];
#endif
}
LAPLACE_HD_FORCEINLINE int lattice_row(int n) { return n == 1021 ? 0 : n == 4093 ? 1 : n == 16381 ? 2 : n == 65521 ? 3 : -1; }

constexpr double golden = 0.61803398874989484820;

}  // namespace detail_mvnormal

namespace detail_mvnormal {

// Cholesky of the m x m correlation C (full storage, in place: L in the lower
// part) with the Gibson-Glasbey-Elston order: at step i the coordinate of
// smallest conditional probability given the earlier ones at their truncated
// means; A, B (standardized limits) permuted with it. y and c are m doubles of
// workspace. False when C is not positive definite.
LAPLACE_HD bool gge_cholesky(int m, double* A, double* B, double* C, double* y, double* c, int* perm = nullptr) {
  for (int i = 0; i < m; ++i) c[i] = 0.0;
  if (perm)
    for (int i = 0; i < m; ++i) perm[i] = i;
  for (int i = 0; i < m; ++i) {
    int best = i;
    double best_le = inf();
    for (int j = i; j < m && i < m - 1; ++j) {
      const double sd = Kokkos::sqrt(C[j * m + j]);
      const double lo = A[j] == -inf() ? -inf() : (A[j] - c[j]) / sd, hi = B[j] == inf() ? inf() : (B[j] - c[j]) / sd;
      const double le = log_interval(lo, hi, (B[j] - A[j]) / sd);
      if (le < best_le) best_le = le, best = j;
    }
    if (best != i) {
      for (int q = 0; q < m; ++q) {
        const double t = C[i * m + q];
        C[i * m + q] = C[best * m + q], C[best * m + q] = t;
      }
      for (int q = 0; q < m; ++q) {
        const double t = C[q * m + i];
        C[q * m + i] = C[q * m + best], C[q * m + best] = t;
      }
      double t = A[i];
      A[i] = A[best], A[best] = t;
      t = B[i], B[i] = B[best], B[best] = t;
      t = c[i], c[i] = c[best], c[best] = t;
      if (perm) {
        const int q = perm[i];
        perm[i] = perm[best], perm[best] = q;
      }
    }
    const double d = C[i * m + i];
    if (!(d > 0.0)) return false;
    const double L = Kokkos::sqrt(d);
    C[i * m + i] = L;
    for (int j = i + 1; j < m; ++j) C[j * m + i] /= L;
    for (int j = i + 1; j < m; ++j)
      for (int q = i + 1; q < m; ++q) C[j * m + q] -= C[j * m + i] * C[q * m + i];
    const double lo = A[i] == -inf() ? -inf() : (A[i] - c[i]) / L, hi = B[i] == inf() ? inf() : (B[i] - c[i]) / L;
    const double le = log_interval(lo, hi, (B[i] - A[i]) / L);
    const double plo = lo == -inf() ? 0.0 : math::exp(-0.5 * lo * lo - half_log_2pi - le);
    const double phi = hi == inf() ? 0.0 : math::exp(-0.5 * hi * hi - half_log_2pi - le);
    const double yi = plo - phi;
    for (int j = i + 1; j < m; ++j) c[j] += C[j * m + i] * yi;
  }
  // the strict upper part is left over from the swaps; clear it
  for (int i = 0; i < m; ++i)
    for (int j = i + 1; j < m; ++j) C[i * m + j] = 0.0;
  return true;
}

// rho = (phi(hi) - phi(lo)) / D and kappa = d rho / d shift = (lo phi(lo) - hi phi(hi)) / D - rho^2,
// D = Phi(hi) - Phi(lo), for the interval (lo, hi]
LAPLACE_HD_FORCEINLINE void interval_rho(double lo, double hi, double width, double& rho, double& kappa) {
  const double le = log_interval(lo, hi, width);
  const double pl = lo == -inf() ? 0.0 : math::exp(-0.5 * lo * lo - half_log_2pi - le);
  const double ph = hi == inf() ? 0.0 : math::exp(-0.5 * hi * hi - half_log_2pi - le);
  rho = ph - pl;
  kappa = (lo == -inf() ? 0.0 : lo * pl) - (hi == inf() ? 0.0 : hi * ph) - rho * rho;
}

// Botev's (2017) minimax tilt: the saddle point (z, mu) in 2 (m - 1) unknowns of
//   psi(z, mu) = sum_i mu_i^2/2 - z_i mu_i + log(Phi(u_i(z) - mu_i) - Phi(l_i(z) - mu_i)),  mu_(m-1) = 0,
// l_i(z) = (A_i - sum_(j<i) L_ij z_j)/L_ii and u_i alike, by Newton's method on the
// gradient with its exact Jacobian (dense, partial pivoting), step halved while
// the gradient's norm does not decrease. work has 4 (m - 1)^2 + 8 m doubles.
LAPLACE_HD bool botev_tilt(int m, const double* A, const double* B, const double* L, double* mu, double* work) {
  const int n = m - 1, n2 = 2 * n;
  double* x = work;            // z (n), mu (n)
  double* g = x + n2;          // gradient
  double* J = g + n2;          // n2 x n2
  double* dx = J + n2 * n2;
  double* xt = dx + n2;
  double* rho = xt + n2;       // m
  double* kap = rho + m;       // m
  for (int i = 0; i < n2; ++i) x[i] = 0.0;
  auto eval = [&](const double* xx, double* gg, double* JJ) {
    const double* z = xx;
    const double* mm = xx + n;
    for (int i = 0; i < m; ++i) {
      double ci = 0.0;
      for (int j = 0; j < i; ++j) ci += L[i * m + j] * z[j];
      const double Li = L[i * m + i], mi = i < n ? mm[i] : 0.0;
      const double lo = A[i] == -inf() ? -inf() : (A[i] - ci) / Li - mi, hi = B[i] == inf() ? inf() : (B[i] - ci) / Li - mi;
      interval_rho(lo, hi, (B[i] - A[i]) / Li, rho[i], kap[i]);
    }
    double nrm = 0.0;
    for (int j = 0; j < n; ++j) {
      double v = -mm[j];
      for (int i = j + 1; i < m; ++i) v -= rho[i] * L[i * m + j] / L[i * m + i];
      gg[j] = v;
      gg[n + j] = mm[j] - z[j] - rho[j];
      nrm += gg[j] * gg[j] + gg[n + j] * gg[n + j];
    }
    if (JJ) {
      for (int q = 0; q < n2 * n2; ++q) JJ[q] = 0.0;
      for (int j = 0; j < n; ++j) {
        // g_z[j]
        JJ[j * n2 + n + j] = -1.0;
        for (int i = j + 1; i < n; ++i) JJ[j * n2 + n + i] = kap[i] * L[i * m + j] / L[i * m + i];
        for (int p = 0; p < n; ++p) {
          double v = 0.0;
          for (int i = (j > p ? j : p) + 1; i < m; ++i) v += kap[i] * L[i * m + j] * L[i * m + p] / (L[i * m + i] * L[i * m + i]);
          JJ[j * n2 + p] = v;
        }
        // g_mu[j]
        for (int p = 0; p < j; ++p) JJ[(n + j) * n2 + p] = kap[j] * L[j * m + p] / L[j * m + j];
        JJ[(n + j) * n2 + j] = -1.0;
        JJ[(n + j) * n2 + n + j] = 1.0 + kap[j];
      }
    }
    return nrm;
  };
  double nrm = eval(x, g, J);
  for (int it = 0; it < 50 && nrm > 1e-26; ++it) {
    // solve J dx = -g by Gaussian elimination with partial pivoting (J is overwritten)
    for (int i = 0; i < n2; ++i) dx[i] = -g[i];
    for (int col = 0; col < n2; ++col) {
      int piv = col;
      for (int r = col + 1; r < n2; ++r)
        if (Kokkos::fabs(J[r * n2 + col]) > Kokkos::fabs(J[piv * n2 + col])) piv = r;
      if (!(Kokkos::fabs(J[piv * n2 + col]) > 0.0)) return false;
      if (piv != col) {
        for (int q = 0; q < n2; ++q) {
          const double t = J[col * n2 + q];
          J[col * n2 + q] = J[piv * n2 + q], J[piv * n2 + q] = t;
        }
        const double t = dx[col];
        dx[col] = dx[piv], dx[piv] = t;
      }
      for (int r = col + 1; r < n2; ++r) {
        const double f = J[r * n2 + col] / J[col * n2 + col];
        if (f == 0.0) continue;
        for (int q = col; q < n2; ++q) J[r * n2 + q] -= f * J[col * n2 + q];
        dx[r] -= f * dx[col];
      }
    }
    for (int r = n2 - 1; r >= 0; --r) {
      double v = dx[r];
      for (int q = r + 1; q < n2; ++q) v -= J[r * n2 + q] * dx[q];
      dx[r] = v / J[r * n2 + r];
    }
    double step = 1.0, nt = 0.0;
    for (int h = 0; h < 30; ++h) {
      for (int i = 0; i < n2; ++i) xt[i] = x[i] + step * dx[i];
      nt = eval(xt, g, nullptr);
      if (nt == nt && nt < nrm) break;
      step *= 0.5;
    }
    if (!(nt < nrm)) break;
    for (int i = 0; i < n2; ++i) x[i] = xt[i];
    nrm = eval(x, g, J);
  }
  for (int i = 0; i < n; ++i) mu[i] = x[n + i];
  mu[n] = 0.0;
  return nrm == nrm && nrm < 1e-10;
}

}  // namespace detail_mvnormal

/// \ingroup mvnormal
/// \brief Largest dimension done by the tensor tanh-sinh rule (the lattice beyond).
inline constexpr int mvn_tensor_max_dim = 4;
/// \ingroup mvnormal
/// \brief Half the nodes of the tensor rule over one coordinate (three constrained): 2 M + 1.
inline constexpr int mvn_tensor_half_nodes_1 = 96;
/// \ingroup mvnormal
/// \brief Half the nodes per coordinate of the tensor rule over two coordinates (four constrained): 2 M + 1.
inline constexpr int mvn_tensor_half_nodes_2 = 48;
/// \ingroup mvnormal
/// \brief The lattice is tilted by Botev's minimax saddle point.
inline constexpr bool mvn_tilt = true;

/// \ingroup mvnormal
/// \brief Default number of lattice points for five or more constrained coordinates.
inline constexpr int mvn_default_lattice = 16381;

/**
 * \ingroup mvnormal
 * \brief Log of the multivariate normal rectangle probability \f$\log\Pr(a < X \le b)\f$, \f$X \sim N(0, \Sigma)\f$.
 *
 * \param k Dimension, \f$1 \le k \le\f$ `mvn_max_dim`.
 * \param a Lower limits (\f$-\infty\f$ allowed).
 * \param b Upper limits (\f$+\infty\f$ allowed).
 * \param S \f$\Sigma\f$ packed lower-triangular (`pdmat::packed_index`), positive definite.
 * \param w Workspace of `mvn_rect_scratch(k)` doubles.
 * \param n_lattice Points of the lattice for five or more constrained coordinates: 1021, 4093, 16381 or 65521.
 * \return \f$\log P\f$; \f$-\infty\f$ for an empty rectangle; \f$0\f$ when every coordinate is unconstrained;
 *         NaN for a NaN argument, a non-positive variance, \f$k\f$ out of range, or, when five or more
 *         coordinates are constrained, a lattice size other than 1021, 4093, 16381 or 65521.
 */
LAPLACE_HD double log_mvn_rect(int k, const double* a, const double* b, const double* S, double* w,
                               int n_lattice = mvn_default_lattice) {
  using namespace detail_mvnormal;
  if (k < 1 || k > mvn_max_dim) return nan();
  // standardized limits of the constrained coordinates, and their correlations
  double* A = w;
  double* B = w + k;
  int* keep = reinterpret_cast<int*>(w + 2 * k);  // k ints in k doubles
  double* C = w + 3 * k;                          // k * k
  int m = 0;
  for (int i = 0; i < k; ++i) {
    const double v = S[pk(i, i)];
    if (a[i] != a[i] || b[i] != b[i] || !(v > 0.0)) return nan();
    if (!(a[i] < b[i])) return -inf();
    if (a[i] == -inf() && b[i] == inf()) continue;
    const double sd = Kokkos::sqrt(v);
    A[m] = a[i] / sd;
    B[m] = b[i] / sd;
    keep[m++] = i;
  }
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      C[i * m + j] = S[pk(keep[i], keep[j])] / Kokkos::sqrt(S[pk(keep[i], keep[i])] * S[pk(keep[j], keep[j])]);
  if (m == 0) return 0.0;
  if (m == 1) return log_interval(A[0], B[0]);
  if (m == 2) return log_bvn_rect(A[0], B[0], A[1], B[1], C[1]);
  int row = lattice_row(n_lattice);
  if (row < 0) {
    if (m > mvn_tensor_max_dim) return nan();
    row = 0;
  }
  double* y = w + 3 * k + k * k;  // conditional means, then the lattice point
  double* c = y + k;              // sum_p L_jp y_p
  double* mu = c + k;             // the tilt
  double* work = mu + k;          // the Newton system
  if (!gge_cholesky(m, A, B, C, y, c)) return nan();
  // Botev's minimax tilt over the first n = m - 1 coordinates (mu_(m-1) = 0)
  const int n = m - 1;
  if (mvn_tilt) {
    if (!botev_tilt(m, A, B, C, mu, work)) for (int i = 0; i < m; ++i) mu[i] = 0.0;
  } else {
    for (int i = 0; i < m; ++i) mu[i] = 0.0;
  }
  // m <= mvn_tensor_max_dim: the tensor tanh-sinh rule, 2 M + 1 nodes per
  // coordinate, over the first m - 2 coordinates and the last two by the
  // bivariate; beyond, the lattice over the first m - 1 and the last by
  // log_interval (the bivariate there cost two to five times more for the same
  // error, which is the lattice's)
  const bool tensor = m <= mvn_tensor_max_dim, last_two = tensor;
  const int s = last_two ? m - 2 : n;
  // 193 nodes over one coordinate, 97 per coordinate over two: a near-singular
  // trivariate (r_12 = -0.95, det R = 0.0075) lost 3.3e-6 in log P with 49
  const int Mt = s == 1 ? mvn_tensor_half_nodes_1 : mvn_tensor_half_nodes_2, Nt1 = 2 * Mt + 1;
  long long N = n_lattice;
  if (tensor) {
    N = 1;
    for (int i = 0; i < s; ++i) N *= Nt1;
  }
  const int zg = korobov(row * 18 + s - 1);
  double s1 = 1.0, s2 = 1.0, rho = 0.0;
  if (last_two) {
    const double L1 = C[s * m + s], L21 = C[(s + 1) * m + s], L22 = C[(s + 1) * m + s + 1];
    s1 = L1;
    s2 = Kokkos::sqrt(L21 * L21 + L22 * L22);
    rho = L21 / s2;
  }
  log_sum acc;
  for (long long pt = 0; pt < N; ++pt) {
    double lw = 0.0;
    long long g = 1, rest = pt;
    for (int i = 0; i < s; ++i) {
      double ci = 0.0;
      for (int p = 0; p < i; ++p) ci += C[i * m + p] * y[p];
      const double L = C[i * m + i], mi = mu[i];
      const double lo = A[i] == -inf() ? -inf() : (A[i] - ci) / L - mi, hi = B[i] == inf() ? inf() : (B[i] - ci) / L - mi;
      const double le = log_interval(lo, hi, (B[i] - A[i]) / L);
      double t, tc;
      if (tensor) {
        double wt;
        ts_node(int(rest % Nt1) - Mt, Mt, t, tc, wt);
        rest /= Nt1;
        lw += math::log(wt);
      } else {
        const double u = frac(double((pt * g) % N) / double(N) + frac((i + 1) * golden));
        t = 1.0 - Kokkos::fabs(2.0 * u - 1.0), tc = Kokkos::fabs(2.0 * u - 1.0);
        if (!(t > 0x1p-60)) t = 0x1p-60, tc = 1.0;
        if (!(tc > 0x1p-60)) tc = 0x1p-60, t = 1.0;
      }
      const double lpa = lo == -inf() ? -inf() : log_Phi(lo), lqb = hi == inf() ? -inf() : log_Phi(-hi);
      const double zi = mi + interval_point(lpa, lqb, le, t, tc);
      y[i] = zi;
      lw += le + 0.5 * mi * mi - zi * mi;
      g = g * zg % N;
    }
    if (last_two) {
      double c1 = 0.0, c2 = 0.0;
      for (int p = 0; p < s; ++p) c1 += C[s * m + p] * y[p], c2 += C[(s + 1) * m + p] * y[p];
      const double lo1 = A[s] == -inf() ? -inf() : (A[s] - c1) / s1, hi1 = B[s] == inf() ? inf() : (B[s] - c1) / s1;
      const double lo2 = A[s + 1] == -inf() ? -inf() : (A[s + 1] - c2) / s2, hi2 = B[s + 1] == inf() ? inf() : (B[s + 1] - c2) / s2;
      const double pf = bvn_rect_genz(lo1, hi1, lo2, hi2, rho);
      lw += pf >= 1e-3 ? math::log(pf) : log_bvn_rect(lo1, hi1, lo2, hi2, rho, (B[s] - A[s]) / s1, (B[s + 1] - A[s + 1]) / s2);
    } else {
      double cl = 0.0;
      for (int p = 0; p < s; ++p) cl += C[n * m + p] * y[p];
      const double L = C[n * m + n];
      const double lo = A[n] == -inf() ? -inf() : (A[n] - cl) / L, hi = B[n] == inf() ? inf() : (B[n] - cl) / L;
      lw += log_interval(lo, hi, (B[n] - A[n]) / L);
    }
    acc.add(lw);
  }
  return tensor ? acc.value() : acc.value() - math::log(double(N));
}

// ---------------------------------------------------------------------------
// Derivatives in the shift of the rectangle.
//
// F(delta) = Pr(lo + delta < X <= hi + delta), X ~ N(0, Sigma). For a set S of
// coordinates and a choice e of a finite end c_i in {lo_i, hi_i} for each i in S,
//   G_{S,e} = phi_{Sigma_SS}(c_S) Pr(lo_T < X_T <= hi_T | X_S = c_S),  T the rest,
// the density of X_S at the corner times the conditional rectangle. With
// Q = Sigma_SS^-1, v = Q c_S, B = Sigma_TS Q and sgn(hi) = 1, sgn(lo) = -1:
//   d_j G_{S,e} = sum_{e_j} sgn(e_j) G_{S+j, e+e_j}                         (j in T)
//   d_i G_{S,e} = -v_i G_{S,e} - sum_{j in T} B_ji sum_{e_j} sgn(e_j) G_{S+j, e+e_j}   (i in S)
// (the second because the conditional limits move with c_S by -B). The
// derivative of order alpha with support U is
//   d^alpha F = sum_e prod_{i in U} sgn(e_i) d^(alpha - 1_U) G_{U,e},
// and d^beta G_{S,e} with beta supported in S follows from the second rule by
// Leibniz, v_i being linear in c_S (d_l v_i = Q_il):
//   d^beta G = -(v_i d^beta' G + sum_l beta'_l Q_il d^(beta' - e_l) G)
//              - sum_{j in T} B_ji sum_{e_j} sgn(e_j) d^beta' G_{S+j, e+e_j},   beta = beta' + e_i,
// which stays inside supports S' ⊇ S. Every G is divided by F, so the tables
// hold d^alpha F / F and nothing underflows.
// ---------------------------------------------------------------------------

namespace detail_mvnormal {

LAPLACE_HD_FORCEINLINE int binom(int n, int k) {
  if (k < 0 || n < k) return 0;
  long long r = 1;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return int(r);
}
// number of sorted tuples of length n over p symbols, and of all lengths <= n
LAPLACE_HD_FORCEINLINE int tuples(int p, int n) { return binom(p + n - 1, n); }
LAPLACE_HD_FORCEINLINE int tuples_upto(int p, int n) { return n < 0 ? 0 : binom(p + n, n); }
// lexicographic rank of a sorted tuple t of length n over p symbols
LAPLACE_HD_FORCEINLINE int tuple_rank(int p, int n, const int* t) {
  int pos = 0, prev = 0;
  for (int j = 0; j < n; ++j) {
    for (int v = prev; v < t[j]; ++v) pos += binom(p - v + n - j - 2, n - j - 1);
    prev = t[j];
  }
  return pos;
}
// the next sorted tuple; false after the last
LAPLACE_HD_FORCEINLINE bool next_tuple(int p, int n, int* t) {
  int i = n - 1;
  while (i >= 0 && t[i] == p - 1) --i;
  if (i < 0) return false;
  ++t[i];
  for (int j = i + 1; j < n; ++j) t[j] = t[i];
  return true;
}
// the next restricted growth string g of length n (a set partition, mx the
// running maximum); false after the last
LAPLACE_HD_FORCEINLINE bool next_partition_rgs(int n, int* g, int* mx) {
  for (int i = n - 1; i > 0; --i) {
    if (g[i] <= mx[i - 1]) {
      ++g[i];
      mx[i] = g[i] > mx[i - 1] ? g[i] : mx[i - 1];
      for (int j = i + 1; j < n; ++j) g[j] = 0, mx[j] = mx[i];
      return true;
    }
  }
  return false;
}
LAPLACE_HD_FORCEINLINE int popcount(unsigned x) {
  int c = 0;
  for (; x; x &= x - 1) ++c;
  return c;
}
// local index of coordinate g in the set S (bit mask)
LAPLACE_HD_FORCEINLINE int local(unsigned S, int g) { return popcount(S & ((1u << g) - 1u)); }

// dense Cholesky of an n x n symmetric matrix in place (lower), log det; false when not positive definite
LAPLACE_HD_FORCEINLINE bool chol(int n, double* a, double& log_det) {
  log_det = 0.0;
  for (int j = 0; j < n; ++j) {
    double d = a[j * n + j];
    for (int p = 0; p < j; ++p) d -= a[j * n + p] * a[j * n + p];
    if (!(d > 0.0)) return false;
    const double l = Kokkos::sqrt(d);
    a[j * n + j] = l;
    log_det += 2.0 * math::log(l);
    for (int i = j + 1; i < n; ++i) {
      double s = a[i * n + j];
      for (int p = 0; p < j; ++p) s -= a[i * n + p] * a[j * n + p];
      a[i * n + j] = s / l;
    }
  }
  return true;
}
// inverse from the Cholesky factor (lower in l), full into q
LAPLACE_HD_FORCEINLINE void chol_inverse(int n, const double* l, double* q) {
  for (int c = 0; c < n; ++c) {
    // solve L L' x = e_c
    for (int i = 0; i < n; ++i) {
      double s = i == c ? 1.0 : 0.0;
      for (int p = 0; p < i; ++p) s -= l[i * n + p] * q[p * n + c];
      q[i * n + c] = s / l[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      double s = q[i * n + c];
      for (int p = i + 1; p < n; ++p) s -= l[p * n + i] * q[p * n + c];
      q[i * n + c] = s / l[i * n + i];
    }
  }
}

}  // namespace detail_mvnormal

// ---------------------------------------------------------------------------
// The moments of the truncated normal.
//
// One dimension. W = Z | lo < Z <= hi, Z standard normal. Its mass sits near
// the end closer to the bulk, c = hi when hi <= 0, c = lo when lo >= 0, and
// W = c - Y (or c + Y) with Y >= 0 of density proportional to
// exp(-a y - y^2/2) on [0, w], a = |c|, w = hi - lo. With
//   I_j(a) = int_0^inf y^j exp(-a y - y^2/2) dy = (-1)^j M^(j)(a),
// M the Mills ratio (normal.hpp's log_mills), the normalized moments are
//   E_j(a) = I_j / I_0 = (-1)^j B_j(l_1, ..., l_j),  l_i = d^i log M (a),
// B_j the complete Bell polynomials, whose terms all have the sign of
// (-1)^j: no cancellation however large a is. A finite w subtracts the part
// beyond w, exp(-a w - w^2/2) sum_i C(j, i) w^(j - i) I_i(a + w); a narrow
// window (w max(1, a) <= 1) is summed by 16-point Gauss-Legendre instead, and
// a window across zero is the two halves from 0 (a = 0). Orders to six (the
// derivatives of log M the library has).
// ---------------------------------------------------------------------------

namespace detail_mvnormal {

// E_j(a) = E[Y^j], j = 0..N, Y with density proportional to exp(-a y - y^2/2) on y > 0, a >= 0
LAPLACE_HD_FORCEINLINE void half_moments_inf(double a, int N, double* E) {
  double l[7];
  l[1] = log_mills_d1(a);
  if (N >= 2) l[2] = log_mills_d2(a);
  if (N >= 3) l[3] = log_mills_d3(a);
  if (N >= 4) l[4] = log_mills_d4(a);
  if (N >= 5) l[5] = log_mills_d5(a);
  if (N >= 6) l[6] = log_mills_d6(a);
  // complete Bell polynomials: B_(n+1) = sum_i C(n, i) B_(n-i) l_(i+1)
  double B[7];
  B[0] = 1.0;
  for (int n = 0; n < N; ++n) {
    double s = 0.0, c = 1.0;
    for (int i = 0; i <= n; ++i) {
      s += c * B[n - i] * l[i + 1];
      c = c * (n - i) / (i + 1);
    }
    B[n + 1] = s;
  }
  for (int j = 0; j <= N; ++j) E[j] = (j % 2 ? -1.0 : 1.0) * B[j];
}

// the same on [0, w]
LAPLACE_HD_FORCEINLINE void half_moments(double a, double w, int N, double* E) {
  if (!(w < inf())) {
    half_moments_inf(a, N, E);
    return;
  }
  if (w * Kokkos::fmax(1.0, a) <= 1.0) {
    // narrow: Gauss-Legendre on [0, w]
    double S[7] = {0, 0, 0, 0, 0, 0, 0};
    const double h = 0.5 * w;
    for (int i = 0; i < 16; ++i) {
      const double y = h + h * quad::detail::legendre16(i);
      const double f = quad::detail::legendre16(16 + i) * math::exp(-a * y - 0.5 * y * y);
      double p = f;
      for (int j = 0; j <= N; ++j) S[j] += p, p *= y;
    }
    for (int j = 0; j <= N; ++j) E[j] = S[j] / S[0];
    return;
  }
  double Ea[7], Eb[7];
  half_moments_inf(a, N, Ea);
  half_moments_inf(a + w, N, Eb);
  // r = I_0(a + w) exp(-a w - w^2/2) / I_0(a): the share of the mass beyond w
  const double r = math::exp(-a * w - 0.5 * w * w + log_mills(a + w) - log_mills(a));
  for (int j = 0; j <= N; ++j) {
    double t = 0.0, c = 1.0, wp = 1.0;
    // sum_i C(j, i) w^(j - i) E_i(a + w), from i = j down
    for (int i = j; i >= 0; --i) {
      t += c * wp * Eb[i];
      c = c * i / (j - i + 1);
      wp *= w;
    }
    E[j] = (Ea[j] - r * t) / (1.0 - r);
  }
}

// W = Z | lo < Z <= hi: its mean and its central moments mu[0..N] (N <= 6)
LAPLACE_HD void trunc1_moments(double lo, double hi, int N, double& mean, double* mu) {
  if (lo > -inf() && hi < inf()) {
    const double h = 0.5 * (hi - lo), m = lo + h;
    if (h * Kokkos::fmax(1.0, Kokkos::fabs(m)) <= 1.0) {
      // narrow: Gauss-Legendre about the midpoint, then about the mean, so that
      // the odd central moments (nearly zero) are summed and not cancelled
      double f[16], x[16], S0 = 0.0, S1 = 0.0;
      for (int i = 0; i < 16; ++i) {
        x[i] = h * quad::detail::legendre16(i);
        f[i] = quad::detail::legendre16(16 + i) * math::exp(-(m * x[i] + 0.5 * x[i] * x[i]));
        S0 += f[i], S1 += f[i] * x[i];
      }
      const double d = S1 / S0;
      mean = m + d;
      for (int j = 0; j <= N; ++j) mu[j] = 0.0;
      for (int i = 0; i < 16; ++i) {
        double p = f[i] / S0;
        const double y = x[i] - d;
        for (int j = 0; j <= N; ++j) mu[j] += p, p *= y;
      }
      mu[1] = 0.0;
      return;
    }
  }
  double M[7];  // moments about c
  double c;
  if (hi <= 0.0 || lo >= 0.0) {
    const bool lower = hi <= 0.0;
    c = lower ? hi : lo;
    const double w = lower ? (lo == -inf() ? inf() : hi - lo) : (hi == inf() ? inf() : hi - lo);
    double E[7];
    half_moments(Kokkos::fabs(c), w, N, E);
    for (int j = 0; j <= N; ++j) M[j] = (lower && (j % 2)) ? -E[j] : E[j];
  } else {
    // across zero: the halves (lo, 0] and (0, hi], weighted by their masses
    c = 0.0;
    double Ep[7], Em[7];
    half_moments(0.0, hi, N, Ep);
    half_moments(0.0, -lo, N, Em);
    const double pp = Phi(hi) - 0.5, pm = 0.5 - Phi(lo);
    for (int j = 0; j <= N; ++j) M[j] = (pp * Ep[j] + pm * ((j % 2) ? -Em[j] : Em[j])) / (pp + pm);
  }
  // to the mean: mu_j = sum_i C(j, i) M_i (-d)^(j - i), d = M_1
  const double d = M[1];
  mean = c + d;
  for (int j = 0; j <= N; ++j) {
    double s = 0.0, cb = 1.0, dp = 1.0;
    for (int i = j; i >= 0; --i) {
      s += cb * M[i] * dp;
      cb = cb * i / (j - i + 1);
      dp *= -d;
    }
    mu[j] = s;
  }
  mu[1] = 0.0;
}

}  // namespace detail_mvnormal

/// \ingroup mvnormal
/// \brief Largest dimension of the derivatives of `log_mvn_rect_derivs` (subsets are bit masks).
inline constexpr int mvn_deriv_max_dim = 12;

/**
 * \ingroup mvnormal
 * \brief Size of the output of `log_mvn_rect_derivs`: every sorted tuple of length \f$0..N\f$ over \f$k\f$ coordinates.
 */
LAPLACE_HD int mvn_rect_derivs_size(int k, int N) { return detail_mvnormal::tuples_upto(k, N); }

/**
 * \ingroup mvnormal
 * \brief Doubles of workspace of `log_mvn_rect_derivs`.
 *
 * The table of every \f$\partial^\beta G_{S,e} / F\f$, the per-set conditional
 * quantities, and the workspace of the inner `log_mvn_rect`.
 */
LAPLACE_HD int mvn_rect_derivs_scratch(int k, int N) {
  using namespace detail_mvnormal;
  const int smax = k < N ? k : N;
  long long n = 0;
  // per set: offset (int), Q (s^2), B (k s), and per end choice: v (s), the table
  for (int s = 0; s <= smax; ++s) {
    const long long sets = binom(k, s);
    n += sets * (1 + s * s + k * s + (1LL << s) * (s + tuples_upto(s, N - s)));
  }
  n += (1LL << k);                  // offsets per mask
  n += 4 * k * k + 4 * k;           // conditional covariance, limits, small matrices
  n += mvn_rect_scratch(k);         // the inner probability
  return int(n);
}

/**
 * \ingroup mvnormal
 * \brief Log rectangle probability and its normalized derivatives in the shift.
 *
 * With \f$F(\delta) = \Pr(lo + \delta < X \le hi + \delta)\f$, writes to `f` the
 * values \f$\partial^\alpha F / F\f$ at \f$\delta = 0\f$ for every sorted tuple
 * \f$\alpha\f$ of length \f$n = 0..N\f$ over the \f$k\f$ coordinates, the block of
 * length \f$n\f$ starting at `tuples_upto(k, n - 1)` and ordered lexicographically.
 *
 * \return \f$\log F\f$; NaN components when \f$F = 0\f$ or an argument is NaN.
 */
LAPLACE_HD double log_mvn_rect_derivs(int k, int N, const double* lo, const double* hi, const double* S, double* f,
                                      double* w, int n_lattice = mvn_default_lattice) {
  using namespace detail_mvnormal;
  const int nf = tuples_upto(k, N);
  if (k < 1 || k > mvn_deriv_max_dim || N < 0) {
    for (int q = 0; q < nf; ++q) f[q] = nan();
    return nan();
  }
  double* inner = w;
  w += mvn_rect_scratch(k);
  const double lF = log_mvn_rect(k, lo, hi, S, inner, n_lattice);
  if (!(lF > -inf()) || lF != lF) {
    for (int q = 0; q < nf; ++q) f[q] = nan();
    return lF;
  }
  const int smax = k < N ? k : N;
  const unsigned nmask = 1u << k;
  // a coordinate can enter a set when one of its ends is finite
  unsigned can = 0;
  for (int i = 0; i < k; ++i)
    if (lo[i] != -inf() || hi[i] != inf()) can |= 1u << i;
  // layout per set S (only subsets of `can` with |S| <= smax): Q (s^2), B (k s),
  // then per end choice e (bit i of e set: the lower end of the i-th member):
  // v (s) and the table of d^beta G / F for |beta| <= N - s (lexicographic by order)
  int* off = reinterpret_cast<int*>(w);
  w += nmask;
  double* base = w;
  long long pos = 0;
  for (unsigned M = 0; M < nmask; ++M) {
    off[M] = -1;
    const int s = popcount(M);
    if ((M & ~can) || s > smax) continue;
    off[M] = int(pos);
    pos += s * s + k * s + (1LL << s) * (s + tuples_upto(s, N - s));
  }
  double* work = base + pos;  // 4 k^2 + 4 k
  auto Qp = [&](unsigned M) { return base + off[M]; };
  auto Bp = [&](unsigned M) { const int s = popcount(M); return base + off[M] + s * s; };
  auto Ep = [&](unsigned M, unsigned e) {
    const int s = popcount(M);
    return base + off[M] + s * s + k * s + (long long)e * (s + tuples_upto(s, N - s));
  };
  // the values: for each set, Q, B, the conditional covariance, and per end
  // choice v and G / F
  for (unsigned M = 0; M < nmask; ++M) {
    if (off[M] < 0) continue;
    const int s = popcount(M), t = k - s;
    int mem[mvn_deriv_max_dim], rest[mvn_deriv_max_dim];
    {
      int a = 0, b = 0;
      for (int i = 0; i < k; ++i) ((M >> i) & 1u ? mem[a++] : rest[b++]) = i;
    }
    double* Q = Qp(M);
    double* B = Bp(M);
    double* L = work;               // s x s
    double* C = work + k * k;       // packed t x t
    double* lo_t = work + 2 * k * k;
    double* hi_t = lo_t + k;
    double* cS = hi_t + k;
    double log_det = 0.0;
    for (int i = 0; i < s; ++i)
      for (int j = 0; j < s; ++j) L[i * s + j] = S[pk(mem[i], mem[j])];
    if (s > 0) {
      if (!chol(s, L, log_det)) {
        for (int q = 0; q < nf; ++q) f[q] = nan();
        return nan();
      }
      chol_inverse(s, L, Q);
    }
    // B = Sigma_TS Q, rows by global coordinate (rows of S unused)
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < s; ++i) {
        double v = 0.0;
        if (!((M >> j) & 1u))
          for (int p = 0; p < s; ++p) v += S[pk(j, mem[p])] * Q[p * s + i];
        B[j * s + i] = v;
      }
    for (int a = 0; a < t; ++a)
      for (int b = 0; b <= a; ++b) {
        double v = S[pk(rest[a], rest[b])];
        for (int p = 0; p < s; ++p) v -= B[rest[a] * s + p] * S[pk(mem[p], rest[b])];
        C[a * (a + 1) / 2 + b] = v;
      }
    for (unsigned e = 0; e < (1u << s); ++e) {
      double* V = Ep(M, e);
      double* T = V + s;
      bool ok = true;
      for (int i = 0; i < s; ++i) {
        cS[i] = (e >> i) & 1u ? lo[mem[i]] : hi[mem[i]];
        if (!(cS[i] > -inf() && cS[i] < inf())) ok = false;
      }
      if (!ok) {
        T[0] = 0.0;
        for (int i = 0; i < s; ++i) V[i] = 0.0;
        continue;
      }
      double quad = 0.0;
      for (int i = 0; i < s; ++i) {
        double v = 0.0;
        for (int p = 0; p < s; ++p) v += Q[i * s + p] * cS[p];
        V[i] = v;
        quad += cS[i] * v;
      }
      double lP = 0.0;
      if (t > 0) {
        for (int a = 0; a < t; ++a) {
          double m = 0.0;
          for (int p = 0; p < s; ++p) m += B[rest[a] * s + p] * cS[p];
          lo_t[a] = lo[rest[a]] == -inf() ? -inf() : lo[rest[a]] - m;
          hi_t[a] = hi[rest[a]] == inf() ? inf() : hi[rest[a]] - m;
        }
        lP = log_mvn_rect(t, lo_t, hi_t, C, inner, n_lattice);
      }
      T[0] = math::exp(-s * half_log_2pi - 0.5 * log_det - 0.5 * quad + lP - lF);
    }
  }
  // the derivatives of G, order by order
  int beta[2 * mvn_deriv_max_dim + 2], bp[2 * mvn_deriv_max_dim + 2], up[2 * mvn_deriv_max_dim + 2];
  for (int n = 1; n <= N; ++n) {
    for (unsigned M = 1; M < nmask; ++M) {
      if (off[M] < 0) continue;
      const int s = popcount(M);
      if (n > N - s) continue;
      int mem[mvn_deriv_max_dim];
      {
        int a = 0;
        for (int i = 0; i < k; ++i)
          if ((M >> i) & 1u) mem[a++] = i;
      }
      const double* Q = Qp(M);
      const double* B = Bp(M);
      const int ob = tuples_upto(s, n - 1), ob1 = tuples_upto(s, n - 2);
      for (unsigned e = 0; e < (1u << s); ++e) {
        double* V = Ep(M, e);
        double* T = V + s;
        if (T[0] == 0.0) {
          for (int q = 0; q < tuples(s, n); ++q) T[ob + q] = 0.0;
          continue;
        }
        for (int q = 0; q < n; ++q) beta[q] = 0;
        int q = 0;
        do {
          const int i = beta[0];
          // beta' = beta[1..n-1]
          for (int r = 1; r < n; ++r) bp[r - 1] = beta[r];
          double val = V[i] * T[ob1 + tuple_rank(s, n - 1, bp)];
          // sum over the occurrences in beta' of Q_il d^(beta' - e_l)
          for (int r = 0; r < n - 1; ++r) {
            if (r > 0 && bp[r] == bp[r - 1]) continue;
            int mult = 0;
            for (int u = r; u < n - 1 && bp[u] == bp[r]; ++u) ++mult;
            int c = 0;
            for (int u = 0; u < n - 1; ++u)
              if (u != r) up[c++] = bp[u];
            val += mult * Q[i * s + bp[r]] * T[tuples_upto(s, n - 3) + tuple_rank(s, n - 2, up)];
          }
          // the sets one larger
          double add = 0.0;
          for (int j = 0; j < k; ++j) {
            if ((M >> j) & 1u) continue;
            const unsigned M2 = M | (1u << j);
            if (off[M2] < 0) continue;
            const double bji = B[j * s + i];
            if (bji == 0.0) continue;
            const int lj = local(M2, j), s2 = s + 1;
            // beta' in the local coordinates of M2
            for (int r = 0; r < n - 1; ++r) up[r] = bp[r] >= lj ? bp[r] + 1 : bp[r];
            const int rk = tuples_upto(s2, n - 2) + tuple_rank(s2, n - 1, up);
            // e extended: insert the bit of j at position lj
            const unsigned low = e & ((1u << lj) - 1u), high = (e >> lj) << (lj + 1);
            double sj = 0.0;
            if (hi[j] != inf()) sj += Ep(M2, low | high)[s2 + rk];
            if (lo[j] != -inf()) sj -= Ep(M2, low | high | (1u << lj))[s2 + rk];
            add += bji * sj;
          }
          T[ob + q] = -val - add;
          ++q;
        } while (next_tuple(s, n, beta));
      }
    }
  }
  // assemble d^alpha F / F
  f[0] = 1.0;
  for (int n = 1; n <= N; ++n) {
    int alpha[2 * mvn_deriv_max_dim + 2];
    for (int q = 0; q < n; ++q) alpha[q] = 0;
    int q = tuples_upto(k, n - 1);
    do {
      unsigned U = 0;
      for (int r = 0; r < n; ++r) U |= 1u << alpha[r];
      double val = 0.0;
      if (off[U] >= 0) {
        const int s = popcount(U), nb = n - s;
        // beta = alpha minus one of each distinct, in local coordinates of U
        int c = 0;
        for (int r = 0; r < n; ++r)
          if (r > 0 && alpha[r] == alpha[r - 1]) beta[c++] = local(U, alpha[r]);
        const int rk = tuples_upto(s, nb - 1) + tuple_rank(s, nb, beta);
        int mem[mvn_deriv_max_dim];
        {
          int a = 0;
          for (int i = 0; i < k; ++i)
            if ((U >> i) & 1u) mem[a++] = i;
        }
        for (unsigned e = 0; e < (1u << s); ++e) {
          const double* V = Ep(U, e);
          const double g = V[s + rk];
          if (g == 0.0) continue;
          int sg = 1;
          for (int i = 0; i < s; ++i)
            if ((e >> i) & 1u) sg = -sg;
          val += sg * g;
        }
      }
      f[q++] = val;
    } while (next_tuple(k, n, alpha));
  }
  return lF;
}


// ---------------------------------------------------------------------------
// The cumulants of the truncated law T = X | lo < X <= hi, from the nodes of
// log_mvn_rect's tilted separation of variables (the last coordinate exact).
// Pass one: the weights and the mean; pass two: the central moments, each
// node contributing the products of d = E[X | node] - E[T] and the exact
// central moments of its last coordinate (trunc1_moments), so that nothing
// large is subtracted. The cumulants follow from the central moments over
// the set partitions without singleton blocks.
// ---------------------------------------------------------------------------

/// \ingroup mvnormal
/// \brief Default number of lattice points of the truncated moments for \f$k \ge 4\f$ (the covariance within 7e-4 of its scale; 16381 gives 3e-3).
inline constexpr int mvn_trunc_default_lattice = 65521;
/// \ingroup mvnormal
/// \brief Largest order of `log_mvn_trunc_cumulants`.
inline constexpr int mvn_trunc_max_order = 6;
/// \ingroup mvnormal
/// \brief Largest dimension done by the tensor rule in `log_mvn_trunc_cumulants` (the lattice beyond).
inline constexpr int mvn_trunc_tensor_max_dim = 3;
/// \ingroup mvnormal
/// \brief Half the nodes per coordinate of the tensor rule of `log_mvn_trunc_cumulants`.
inline constexpr int mvn_trunc_tensor_half_nodes = 32;

/**
 * \ingroup mvnormal
 * \brief Doubles of workspace of `log_mvn_trunc_cumulants`.
 */
LAPLACE_HD int mvn_trunc_scratch(int k, int N) {
  return mvn_rect_scratch(k) + 3 * detail_mvnormal::tuples_upto(k, N) + 8 * k + 16;
}

/**
 * \ingroup mvnormal
 * \brief Log rectangle probability and the cumulants of the truncated law over the constrained coordinates.
 *
 * With \f$T = X \mid lo < X \le hi\f$, \f$X \sim N(0, \Sigma)\f$, writes the number
 * \f$m\f$ of constrained coordinates (those with a finite limit), their indices
 * `keep[0..m)`, and to `kap` the cumulants of \f$T\f$ restricted to them: the
 * block of order \f$n = 1..N\f$ (\f$N \le\f$ `mvn_trunc_max_order`) starts at
 * `tuples_upto(m, n - 1)` and lists the sorted tuples over \f$0..m-1\f$
 * (positions in `keep`) lexicographically; order 1 is the mean.
 *
 * \return \f$\log P\f$; \f$-\infty\f$ for an empty rectangle; NaN for a NaN argument, a non-positive
 *         variance, \f$k\f$ or \f$N\f$ out of range, or, when four or more coordinates are constrained,
 *         a lattice size other than 1021, 4093, 16381 or 65521.
 */
LAPLACE_HD double log_mvn_trunc_cumulants(int k, int N, const double* lo, const double* hi, const double* S, int& m,
                                          int* keep, double* kap, double* w, int n_lattice = mvn_trunc_default_lattice) {
  using namespace detail_mvnormal;
  m = 0;
  if (k < 1 || k > mvn_max_dim || N < 1 || N > mvn_trunc_max_order) return nan();
  double* A = w;
  double* B = A + k;
  double* sd = B + k;
  double* C = sd + k;                 // m x m
  double* y = C + k * k;              // node: sampled standard variables
  double* c = y + k;                  // GGE workspace
  double* mu = c + k;                 // tilt
  double* xb = mu + k;                // conditional mean of the node
  double* mean = xb + k;              // pass-one mean (permuted coordinates)
  int* perm = reinterpret_cast<int*>(mean + k);
  double* work = mean + 2 * k;        // Newton system, 4 k^2 + 8 k
  double* mom = work + 4 * k * k + 8 * k;  // central moments, tuples_upto(m, N)
  for (int i = 0; i < k; ++i) {
    const double v = S[pk(i, i)];
    if (lo[i] != lo[i] || hi[i] != hi[i] || !(v > 0.0)) return nan();
    if (!(lo[i] < hi[i])) return -inf();
    if (lo[i] == -inf() && hi[i] == inf()) continue;
    sd[m] = Kokkos::sqrt(v);
    A[m] = lo[i] / sd[m];
    B[m] = hi[i] / sd[m];
    keep[m++] = i;
  }
  const int nt = tuples_upto(m, N);
  for (int q = 0; q < nt; ++q) kap[q] = 0.0;
  if (m == 0) return 0.0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j) C[i * m + j] = S[pk(keep[i], keep[j])] / (sd[i] * sd[j]);
  if (!gge_cholesky(m, A, B, C, y, c, perm)) return nan();
  const int n = m - 1;
  if (m > 1) {
    if (!botev_tilt(m, A, B, C, mu, work))
      for (int i = 0; i < m; ++i) mu[i] = 0.0;
  } else {
    mu[0] = 0.0;
  }
  // the nodes: the tensor tanh-sinh rule over the first n coordinates when
  // m <= mvn_trunc_tensor_max_dim, else the lattice
  const bool tensor = m <= mvn_trunc_tensor_max_dim;
  constexpr int Mt = mvn_trunc_tensor_half_nodes, Nt1 = 2 * Mt + 1;
  long long NP = 1;
  int row = 0;
  if (tensor) {
    for (int i = 0; i < n; ++i) NP *= Nt1;
  } else {
    row = lattice_row(n_lattice);
    if (row < 0) return nan();
    NP = n_lattice;
  }
  const int zg = n > 0 ? korobov(row * 18 + n - 1) : 1;
  // one node: its log weight, the conditional mean xb and the last
  // coordinate's central moments wm[0..N] (scaled by L_nn^j)
  double wm[mvn_trunc_max_order + 1];
  auto node = [&](long long pt) {
    double lw = 0.0;
    long long g = 1, rest = pt;
    for (int i = 0; i < n; ++i) {
      double ci = 0.0;
      for (int p = 0; p < i; ++p) ci += C[i * m + p] * y[p];
      const double L = C[i * m + i], mi = mu[i];
      const double l0 = A[i] == -inf() ? -inf() : (A[i] - ci) / L - mi, h0 = B[i] == inf() ? inf() : (B[i] - ci) / L - mi;
      const double le = log_interval(l0, h0, (B[i] - A[i]) / L);
      double t, tc;
      if (tensor) {
        double wt;
        ts_node(int(rest % Nt1) - Mt, Mt, t, tc, wt);
        rest /= Nt1;
        lw += math::log(wt);
      } else {
        const double u = frac(double((pt * g) % NP) / double(NP) + frac((i + 1) * golden));
        t = 1.0 - Kokkos::fabs(2.0 * u - 1.0), tc = Kokkos::fabs(2.0 * u - 1.0);
        if (!(t > 0x1p-60)) t = 0x1p-60, tc = 1.0;
        if (!(tc > 0x1p-60)) tc = 0x1p-60, t = 1.0;
        g = g * zg % NP;
      }
      const double lpa = l0 == -inf() ? -inf() : log_Phi(l0), lqb = h0 == inf() ? -inf() : log_Phi(-h0);
      const double zi = mi + interval_point(lpa, lqb, le, t, tc);
      y[i] = zi;
      lw += le + 0.5 * mi * mi - zi * mi;
    }
    for (int i = 0; i < n; ++i) {
      double v = 0.0;
      for (int p = 0; p <= i; ++p) v += C[i * m + p] * y[p];
      xb[i] = v;
    }
    double cl = 0.0;
    for (int p = 0; p < n; ++p) cl += C[n * m + p] * y[p];
    const double L = C[n * m + n];
    const double l0 = A[n] == -inf() ? -inf() : (A[n] - cl) / L, h0 = B[n] == inf() ? inf() : (B[n] - cl) / L;
    lw += log_interval(l0, h0, (B[n] - A[n]) / L);
    double wmean;
    trunc1_moments(l0, h0, N, wmean, wm);
    xb[n] = cl + L * wmean;
    double Lj = 1.0;
    for (int j = 0; j <= N; ++j) wm[j] *= Lj, Lj *= L;
    return lw;
  };
  if (!tensor) NP = n_lattice;
  // pass one: log P and the mean, the sums kept relative to the largest weight
  double lmax = -inf(), ssum = 0.0;
  for (int i = 0; i < m; ++i) mean[i] = 0.0;
  for (long long pt = 0; pt < NP; ++pt) {
    const double lw = node(pt);
    if (!(lw > -inf())) continue;
    if (lw > lmax) {
      const double r = math::exp(lmax - lw);
      ssum *= r;
      for (int i = 0; i < m; ++i) mean[i] *= r;
      lmax = lw;
    }
    const double e = math::exp(lw - lmax);
    ssum += e;
    for (int i = 0; i < m; ++i) mean[i] += e * xb[i];
  }
  if (!(ssum > 0.0)) return -inf();
  for (int i = 0; i < m; ++i) mean[i] /= ssum;
  const double lP = lmax + math::log(ssum) - (tensor ? 0.0 : math::log(double(NP)));
  // pass two: the central moments of orders 2..N
  for (int q = 0; q < nt; ++q) mom[q] = 0.0;
  if (N >= 2) {
    double pw[mvn_trunc_max_order + 1];
    int t[mvn_trunc_max_order];
    for (long long pt = 0; pt < NP; ++pt) {
      const double lw = node(pt);
      if (!(lw > -inf())) continue;
      const double e = math::exp(lw - lmax) / ssum;
      for (int i = 0; i < m; ++i) xb[i] -= mean[i];  // d
      // E[(d_n + L (W - E W))^j] = sum_i C(j, i) d_n^(j - i) wm_i
      for (int j = 0; j <= N; ++j) {
        double v = 0.0, cb = 1.0, dp = 1.0;
        for (int i = j; i >= 0; --i) {
          v += cb * wm[i] * dp;
          cb = cb * i / (j - i + 1);
          dp *= xb[n];
        }
        pw[j] = v;
      }
      for (int o = 2; o <= N; ++o) {
        for (int q = 0; q < o; ++q) t[q] = 0;
        int pos = tuples_upto(m, o - 1);
        do {
          double v = 1.0;
          int cnt = 0;
          for (int q = 0; q < o; ++q) {
            if (t[q] == n) ++cnt;
            else v *= xb[t[q]];
          }
          mom[pos++] += e * v * pw[cnt];
        } while (next_tuple(m, o, t));
      }
    }
  }
  // cumulants (permuted standardized coordinates) into the scratch after mom,
  // then unpermuted and unstandardized into kap
  double* cum = mom + nt;
  for (int i = 0; i < m; ++i) cum[1 + i] = mean[i];
  for (int o = 2; o <= N; ++o) {
    int t[mvn_trunc_max_order], g[mvn_trunc_max_order], mx[mvn_trunc_max_order], sub[mvn_trunc_max_order];
    for (int q = 0; q < o; ++q) t[q] = 0;
    int pos = tuples_upto(m, o - 1);
    do {
      for (int q = 0; q < o; ++q) g[q] = 0, mx[q] = 0;
      double v = 0.0;
      do {
        const int nb = mx[o - 1] + 1;
        bool single = false;
        double prod = 1.0;
        for (int b = 0; b < nb && !single; ++b) {
          int sz = 0;
          for (int q = 0; q < o; ++q)
            if (g[q] == b) sub[sz++] = t[q];
          if (sz < 2) single = true;
          else prod *= mom[tuples_upto(m, sz - 1) + tuple_rank(m, sz, sub)];
        }
        if (single) continue;
        double fac = 1.0;
        for (int b = 1; b < nb; ++b) fac *= -b;
        v += fac * prod;
      } while (next_partition_rgs(o, g, mx));
      cum[pos++] = v;
    } while (next_tuple(m, o, t));
  }
  // permuted position p holds original constrained coordinate perm[p]
  int inv[mvn_max_dim];
  for (int p = 0; p < m; ++p) inv[perm[p]] = p;
  for (int o = 1; o <= N; ++o) {
    int t[mvn_trunc_max_order], u[mvn_trunc_max_order];
    for (int q = 0; q < o; ++q) t[q] = 0;
    int pos = tuples_upto(m, o - 1);
    do {
      double scale = 1.0;
      for (int q = 0; q < o; ++q) u[q] = inv[t[q]], scale *= sd[t[q]];
      // sort u
      for (int a = 1; a < o; ++a)
        for (int b = a; b > 0 && u[b] < u[b - 1]; --b) {
          const int x = u[b];
          u[b] = u[b - 1], u[b - 1] = x;
        }
      kap[pos++] = scale * cum[tuples_upto(m, o - 1) + tuple_rank(m, o, u)];
    } while (next_tuple(m, o, t));
  }
  return lP;
}

/**
 * \ingroup mvnormal
 * \brief Doubles of workspace of `mvn_trunc_mean` and `mvn_trunc_cov`.
 */
LAPLACE_HD int mvn_trunc_moment_scratch(int k) {
  return mvn_trunc_scratch(k, 2) + 3 * k * k + detail_mvnormal::tuples_upto(k, 2) + 2 * k;
}

namespace detail_mvnormal {

// the moments of the whole truncated vector from those of its constrained
// part: the free coordinates (both limits infinite) are X_f = Bf X_c + E with
// Bf = Sigma_fc Sigma_cc^-1 and E independent of X_c, of covariance
// Sigma_ff - Bf Sigma_cf; Order 1 writes the mean, 2 the packed covariance
template <int Order>
LAPLACE_HD double trunc_moment(int k, const double* lo, const double* hi, const double* S, double* out, double* w,
                               int n_lattice) {
  const int nt = tuples_upto(k, 2);
  double* kap = w;
  int* keep = reinterpret_cast<int*>(kap + nt);
  int* isc = keep + k;  // position of each coordinate in keep, or -1
  double* Q = kap + nt + 2 * k;
  double* Bf = Q + k * k;
  double* L = Bf + k * k;
  double* ws = L + k * k;
  int m = 0;
  const double lP = log_mvn_trunc_cumulants(k, Order, lo, hi, S, m, keep, kap, ws, n_lattice);
  const int no = Order == 1 ? k : k * (k + 1) / 2;
  if (lP != lP || !(lP > -inf())) {
    for (int q = 0; q < no; ++q) out[q] = nan();
    return lP;
  }
  for (int i = 0; i < k; ++i) isc[i] = -1;
  for (int p = 0; p < m; ++p) isc[keep[p]] = p;
  // Sigma_cc^-1 and Bf
  for (int a = 0; a < m; ++a)
    for (int b = 0; b < m; ++b) L[a * m + b] = S[pk(keep[a], keep[b])];
  double ld;
  if (m > 0) {
    chol(m, L, ld);
    chol_inverse(m, L, Q);
  }
  for (int i = 0; i < k; ++i)
    for (int p = 0; p < m; ++p) {
      double v = 0.0;
      if (isc[i] < 0)
        for (int q = 0; q < m; ++q) v += S[pk(i, keep[q])] * Q[q * m + p];
      Bf[i * m + p] = v;
    }
  // the constrained mean and covariance, kap's order-1 and order-2 blocks
  auto mc = [&](int p) { return kap[1 + p]; };
  auto cc = [&](int p, int q) {
    const int a = p < q ? p : q, b = p < q ? q : p;
    const int t[2] = {a, b};
    return kap[tuples_upto(m, 1) + tuple_rank(m, 2, t)];
  };
  if constexpr (Order == 1) {
    for (int i = 0; i < k; ++i) {
      if (isc[i] >= 0) {
        out[i] = mc(isc[i]);
      } else {
        double v = 0.0;
        for (int p = 0; p < m; ++p) v += Bf[i * m + p] * mc(p);
        out[i] = v;
      }
    }
  } else {
    for (int i = 0; i < k; ++i)
      for (int j = 0; j <= i; ++j) {
        double v;
        const int pi = isc[i], pj = isc[j];
        if (pi >= 0 && pj >= 0) {
          v = cc(pi, pj);
        } else if (pi >= 0 || pj >= 0) {
          const int f = pi >= 0 ? j : i, c = pi >= 0 ? pi : pj;
          v = 0.0;
          for (int p = 0; p < m; ++p) v += Bf[f * m + p] * cc(p, c);
        } else {
          // Sigma_ij - (Bf Sigma_cf)_ij + (Bf Cov_cc Bf')_ij
          v = S[pk(i, j)];
          for (int p = 0; p < m; ++p) {
            v -= Bf[i * m + p] * S[pk(keep[p], j)];
            double t = 0.0;
            for (int q = 0; q < m; ++q) t += cc(p, q) * Bf[j * m + q];
            v += Bf[i * m + p] * t;
          }
        }
        out[pk(i, j)] = v;
      }
  }
  return lP;
}

}  // namespace detail_mvnormal

/**
 * \ingroup mvnormal
 * \brief Mean of the truncated normal \f$T = X \mid lo < X \le hi\f$, \f$X \sim N(0, \Sigma)\f$.
 *
 * Computed from the nodes of `log_mvn_rect`'s tilted separation of variables,
 * relative to the weighted mean itself. `w` has `mvn_trunc_moment_scratch(k)` doubles.
 * \return \f$\log P\f$.
 */
LAPLACE_HD double mvn_trunc_mean(int k, const double* lo, const double* hi, const double* S, double* mean, double* w,
                                 int n_lattice = mvn_trunc_default_lattice) {
  return detail_mvnormal::trunc_moment<1>(k, lo, hi, S, mean, w, n_lattice);
}

/**
 * \ingroup mvnormal
 * \brief Covariance of the truncated normal, packed lower-triangular.
 *
 * Accumulated centred on the mean, never as \f$\Sigma + \Sigma H \Sigma\f$ with \f$H\f$
 * the Hessian of \f$\log P\f$ (which cancels in the tail). `w` has
 * `mvn_trunc_moment_scratch(k)` doubles.
 * \return \f$\log P\f$.
 */
LAPLACE_HD double mvn_trunc_cov(int k, const double* lo, const double* hi, const double* S, double* cov, double* w,
                                int n_lattice = mvn_trunc_default_lattice) {
  return detail_mvnormal::trunc_moment<2>(k, lo, hi, S, cov, w, n_lattice);
}

// ---------------------------------------------------------------------------
// The derivatives of log F in the shift through the cumulants. With
// F(delta) = Pr(lo + delta < X <= hi + delta) and phi_Sigma(u + delta) =
// phi_Sigma(u) exp(-delta' Q u - delta' Q delta / 2), Q = Sigma^-1 (over the
// constrained coordinates, the free ones not entering F),
//   log F(delta) = log P + K_T(-Q delta) - delta' Q delta / 2,
// K_T the cumulant generating function of the truncated law, so
//   d log F = -Q kappa_1,  d^2 log F = -Q + Q kappa_2 Q,  d^n log F = (-Q)^(x n) kappa_n  (n >= 3).
// The cumulants are small where the derivatives are (in the tail they fall
// like |a|^-n) and come out of centred sums, so this route does not cancel
// where the moments of the shift do.
// ---------------------------------------------------------------------------

/**
 * \ingroup mvnormal
 * \brief Doubles of workspace of `log_mvn_rect_log_derivs`.
 */
LAPLACE_HD int mvn_rect_log_derivs_scratch(int k, int N) {
  using namespace detail_mvnormal;
  int big = 0;
  for (int j = 0; j <= N; ++j) {
    const int v = tuples(k, j) * tuples(k, N - j);
    big = v > big ? v : big;
  }
  return mvn_trunc_scratch(k, N) + tuples_upto(k, N) + 2 * k * k + 2 * k + 2 * big;
}

/**
 * \ingroup mvnormal
 * \brief Log rectangle probability and the derivatives of its log in the shift, through the cumulants of the truncated law.
 *
 * Writes \f$\partial^\alpha \log F\f$ for the sorted tuples \f$\alpha\f$ of length
 * \f$n = 1..N\f$ (\f$N \le\f$ `mvn_trunc_max_order`) over the \f$k\f$ coordinates, the
 * block of length \f$n\f$ at `tuples_upto(k, n - 1)`, lexicographic (position 0 unused).
 * `w` has `mvn_rect_log_derivs_scratch(k, N)` doubles.
 * \return \f$\log F\f$.
 */
LAPLACE_HD double log_mvn_rect_log_derivs(int k, int N, const double* lo, const double* hi, const double* S, double* out,
                                          double* w, int n_lattice = mvn_default_lattice) {
  using namespace detail_mvnormal;
  const int nf = tuples_upto(k, N);
  double* kap = w;
  int* keep = reinterpret_cast<int*>(kap + nf);
  double* Q = kap + nf + 2 * k;
  double* L = Q + k * k;
  int big = 0;
  for (int j = 0; j <= N; ++j) {
    const int v = tuples(k, j) * tuples(k, N - j);
    big = v > big ? v : big;
  }
  double* T0 = L + k * k;
  double* T1 = T0 + big;
  double* ws = T1 + big;
  int m = 0;
  const double lF = log_mvn_trunc_cumulants(k, N, lo, hi, S, m, keep, kap, ws, n_lattice);
  for (int q = 0; q < nf; ++q) out[q] = 0.0;
  if (lF != lF || !(lF > -inf())) {
    for (int q = 0; q < nf; ++q) out[q] = nan();
    return lF;
  }
  if (m == 0) return lF;
  for (int a = 0; a < m; ++a)
    for (int b = 0; b < m; ++b) L[a * m + b] = S[pk(keep[a], keep[b])];
  double ld;
  chol(m, L, ld);
  chol_inverse(m, L, Q);
  int a[mvn_trunc_max_order], b[mvn_trunc_max_order], t[mvn_trunc_max_order + 1], g[mvn_trunc_max_order];
  for (int n = 1; n <= N; ++n) {
    // T_j[(a sorted j-tuple), (b sorted (n - j)-tuple)]: kappa_n with j of its
    // indices contracted with -Q; row-major in (rank a, rank b)
    const int base = tuples_upto(m, n - 1);
    for (int q = 0; q < tuples(m, n); ++q) T0[q] = kap[base + q];
    double* src = T0;
    double* dst = T1;
    for (int j = 0; j < n; ++j) {
      // to j + 1: new[(a' = a + {a_last}), b'] = sum_c (-Q)_(a_last c) old[a, b' + {c}], a_last the largest of a'
      const int na = tuples(m, j + 1), nb = tuples(m, n - j - 1), ob = tuples(m, n - j);
      for (int q = 0; q < j + 1; ++q) a[q] = 0;
      int ra = 0;
      do {
        const int last = a[j];
        const int rold = tuple_rank(m, j, a);
        for (int q = 0; q < n - j - 1; ++q) b[q] = 0;
        int rb = 0;
        do {
          double v = 0.0;
          for (int c = 0; c < m; ++c) {
            // b + {c}, sorted
            int r = 0;
            bool ins = false;
            for (int q = 0; q < n - j - 1; ++q) {
              if (!ins && c <= b[q]) t[r++] = c, ins = true;
              t[r++] = b[q];
            }
            if (!ins) t[r++] = c;
            v -= Q[last * m + c] * src[rold * ob + tuple_rank(m, n - j, t)];
          }
          dst[ra * nb + rb] = v;
          ++rb;
        } while (n - j - 1 > 0 && next_tuple(m, n - j - 1, b));
        ++ra;
      } while (next_tuple(m, j + 1, a));
      double* x = src;
      src = dst;
      dst = x;
    }
    // src: tuples(m, n) x 1; the order-2 term -Q
    if (n == 2)
      for (int p = 0; p < m; ++p)
        for (int q = p; q < m; ++q) {
          const int tt[2] = {p, q};
          src[tuple_rank(m, 2, tt)] -= Q[p * m + q];
        }
    // scatter to the k coordinates (keep is increasing, so sorted stays sorted)
    for (int q = 0; q < n; ++q) g[q] = 0;
    int r = 0;
    do {
      for (int q = 0; q < n; ++q) t[q] = keep[g[q]];
      out[tuples_upto(k, n - 1) + tuple_rank(k, n, t)] = src[r++];
    } while (next_tuple(m, n, g));
  }
  return lF;
}

}  // namespace laplace::math
