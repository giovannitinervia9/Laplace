// Double-double arithmetic: a value is the unevaluated sum hi + lo of two
// doubles with |lo| <= ulp(hi)/2, which carries about 106 bits. Only the
// operations that Laplace's numerics use are here (sum, product, quotient,
// with doubles and with double-doubles); the algorithms are the classical ones
// of Dekker (1971) and Knuth (TAOCP vol. 2, 4.2.2), in the form given by
// Hida, Li and Bailey's QD library (the "accurate" sum).
//
// The exact product uses the FMA on the device. On the host the build does not
// assume FMA hardware (gcc without -mfma calls a slow software fma), so the
// product is Dekker's split, as in normal.hpp.
//
// Every elementary operation goes through add/sub/mul below, which on the
// CUDA device are the intrinsics __dadd_rn, __dsub_rn, __dmul_rn: nvcc never
// contracts them into an FMA. Without them nvcc fuses, after inlining, a
// product with the sum that follows it (s = p.hi + p.lo with p.hi = a b
// becomes fma(a, b, p.lo)), which breaks the exact identities the algorithms
// rest on: measured on 2026-11-10, the derivatives of order 4 to 6 of the
// Bessel ratio lost up to 1.4e4 ulp on the GPU against 2.4 on the CPU.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::math::detail::ddm {

LAPLACE_HD_FORCEINLINE double add(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}
LAPLACE_HD_FORCEINLINE double sub(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dsub_rn(a, b);
#else
  return a - b;
#endif
}
LAPLACE_HD_FORCEINLINE double mul(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}

struct dd {
  double hi, lo;
  LAPLACE_HD_FORCEINLINE constexpr dd() : hi(0.0), lo(0.0) {}
  LAPLACE_HD_FORCEINLINE constexpr dd(double h) : hi(h), lo(0.0) {}
  LAPLACE_HD_FORCEINLINE constexpr dd(double h, double l) : hi(h), lo(l) {}
};

// s + e = a + b exactly
LAPLACE_HD_FORCEINLINE dd two_sum(double a, double b) {
  const double s = add(a, b), bb = sub(s, a);
  return dd(s, add(sub(a, sub(s, bb)), sub(b, bb)));
}

// s + e = a + b exactly, for |a| >= |b| (or a = 0)
LAPLACE_HD_FORCEINLINE dd quick_two_sum(double a, double b) {
  const double s = add(a, b);
  return dd(s, sub(b, sub(s, a)));
}

// p + e = a b exactly
LAPLACE_HD_FORCEINLINE dd two_prod(double a, double b) {
  const double p = mul(a, b);
  double e;
  KOKKOS_IF_ON_DEVICE((e = Kokkos::fma(a, b, -p);))
  KOKKOS_IF_ON_HOST((constexpr double split = 134217729.0;  // 2^27 + 1
                     const double ca = split * a, ah = ca - (ca - a), al = a - ah;
                     const double cb = split * b, bh = cb - (cb - b), bl = b - bh;
                     e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;))
  return dd(p, e);
}

LAPLACE_HD_FORCEINLINE dd operator-(const dd& a) { return dd(-a.hi, -a.lo); }

LAPLACE_HD_FORCEINLINE dd operator+(const dd& a, const dd& b) {
  dd s = two_sum(a.hi, b.hi);
  const dd t = two_sum(a.lo, b.lo);
  s.lo = add(s.lo, t.hi);
  s = quick_two_sum(s.hi, s.lo);
  s.lo = add(s.lo, t.lo);
  return quick_two_sum(s.hi, s.lo);
}

LAPLACE_HD_FORCEINLINE dd operator+(const dd& a, double b) {
  dd s = two_sum(a.hi, b);
  s.lo = add(s.lo, a.lo);
  return quick_two_sum(s.hi, s.lo);
}
LAPLACE_HD_FORCEINLINE dd operator+(double a, const dd& b) { return b + a; }
LAPLACE_HD_FORCEINLINE dd operator-(const dd& a, const dd& b) { return a + (-b); }
LAPLACE_HD_FORCEINLINE dd operator-(const dd& a, double b) { return a + (-b); }
LAPLACE_HD_FORCEINLINE dd operator-(double a, const dd& b) { return (-b) + a; }

LAPLACE_HD_FORCEINLINE dd operator*(const dd& a, const dd& b) {
  dd p = two_prod(a.hi, b.hi);
  p.lo = add(p.lo, add(mul(a.hi, b.lo), mul(a.lo, b.hi)));
  return quick_two_sum(p.hi, p.lo);
}

LAPLACE_HD_FORCEINLINE dd operator*(const dd& a, double b) {
  dd p = two_prod(a.hi, b);
  p.lo = add(p.lo, mul(a.lo, b));
  return quick_two_sum(p.hi, p.lo);
}
LAPLACE_HD_FORCEINLINE dd operator*(double a, const dd& b) { return b * a; }

// a / b by three correction steps of the double quotient (QD's accurate division)
LAPLACE_HD_FORCEINLINE dd operator/(const dd& a, const dd& b) {
  const double q1 = a.hi / b.hi;
  dd r = a - b * q1;
  const double q2 = r.hi / b.hi;
  r = r - b * q2;
  const double q3 = r.hi / b.hi;
  return quick_two_sum(q1, q2) + q3;
}

LAPLACE_HD_FORCEINLINE dd operator/(const dd& a, double b) { return a / dd(b); }

// Square root by Karp's method (Karp and Markstein, "High-precision division
// and square root", ACM TOMS 1997, section 3; the form of QD's dd_real::sqrt).
// With x = 1 / sqrt(a.hi) in double and ax = a.hi x in double, the first
// approximation ax has a relative error of about 2^-53, and
//
//   sqrt(a) = ax + (a - ax^2) x / 2 + O(error^2),
//
// where the residual a - ax^2 is formed in double-double, exactly up to 2^-106 |a|.
// The second-order term is of size 2^-106 sqrt(a), so the result is accurate
// to about 106 bits. a < 0 and NaN give NaN, 0 gives 0, +inf gives +inf
// (the product 0 inf of the general formula is avoided).
LAPLACE_HD dd sqrt(const dd& a) {
  if (!(a.hi >= 0.0)) return dd(laplace::math::detail::quiet_nan());
  if (a.hi == 0.0) return dd(0.0);
  if (a.hi == laplace::math::detail::pos_inf()) return dd(a.hi);
  // outside [2^-900, 2^900] the argument is scaled by an even power of two
  // (exactly): Dekker's split in two_prod overflows above 2^996, and
  // 1 / sqrt(a.hi) overflows for subnormal a
  int k = 0;
  dd b = a;
  if (a.hi < 0x1p-900 || a.hi > 0x1p900) {
    int e = 0;
    Kokkos::frexp(a.hi, &e);
    k = e / 2;
    b = dd(Kokkos::ldexp(a.hi, -2 * k), Kokkos::ldexp(a.lo, -2 * k));
  }
  const double x = 1.0 / Kokkos::sqrt(b.hi);
  const double ax = mul(b.hi, x);
  const dd r = b - two_prod(ax, ax);
  const dd s = dd(ax) + mul(r.hi, mul(x, 0.5));
  return k == 0 ? s : dd(Kokkos::ldexp(s.hi, k), Kokkos::ldexp(s.lo, k));
}

// Exponential, the algorithm of the QD library (Hida, Li and Bailey, "Library
// for double-double and quad-double arithmetic", 2000, dd_real::exp), with
// the reduction of ln 2 split as in Cody and Waite, so that the exponent k
// is subtracted without error. The steps are:
//
//   k = round(a.hi / ln 2)                        (double)
//   t = a - k LN2_1 - k LN2_2                     (double-double; k LN2_1 is exact)
//   r = t / 2^9                                   (exact; |r| <= ln 2 / 2^10 + tiny)
//   s = exp(r) - 1 = r + r^2/2! + r^3/3! + ...    (Taylor series in double-double,
//                                                  the terms added until one falls
//                                                  below 2^-110 |s|, at most 20)
//   repeat 9 times: s <- s (s + 2)                (this is (1 + s)^2 - 1, written so
//                                                  that s keeps its relative precision)
//   exp(a) = (1 + s) 2^k                          (exact, unless the result is subnormal)
//
// LN2_1 = 0x1.62e42fefa38p-1 has 42 significant bits, so k LN2_1 is exact for
// |k| < 2^11. LN2_2 = ln 2 - LN2_1 as a double-double. Both are computed with
// mpmath at 300 bits. Overflow (a.hi > 710) gives +inf, underflow (a.hi < -746)
// gives 0, NaN gives NaN.
// e^t - 1 for |t| <= ln 2 / 2 (plus rounding), with relative precision: the
// Taylor series of e^r - 1 at r = t / 2^9 and nine steps s <- s (s + 2), which
// is (1 + s)^2 - 1 and keeps the relative precision of s
LAPLACE_HD_FORCEINLINE dd expm1_reduced(const dd& t) {
  const dd r(mul(t.hi, 0x1p-9), mul(t.lo, 0x1p-9));
  dd s = r, p = r;  // p = r^n / n! after the n-th pass
  for (int n = 2; n <= 20; ++n) {
    p = (p * r) / static_cast<double>(n);
    s = s + p;
    if (!(Kokkos::fabs(p.hi) > 0x1p-110 * Kokkos::fabs(s.hi))) break;
  }
  for (int i = 0; i < 9; ++i) s = s * (s + 2.0);
  return s;
}

LAPLACE_HD dd exp(const dd& a) {
  if (a.hi != a.hi) return dd(laplace::math::detail::quiet_nan());
  if (a.hi > 710.0) return dd(laplace::math::detail::pos_inf());
  if (a.hi < -746.0) return dd(0.0);
  constexpr double LN2_1 = 0x1.62e42fefa38p-1;
  constexpr dd LN2_2(0x1.ef35793c76730p-45, 0x1.f97b57a079a19p-103);
  constexpr double INV_LN2 = 0x1.71547652b82fep+0;
  constexpr int M = 9;

  const double k = Kokkos::floor(add(mul(a.hi, INV_LN2), 0.5));
  dd t = a - mul(k, LN2_1);
  t = t - dd(k) * LN2_2;
  (void)M;
  const dd res = expm1_reduced(t) + 1.0;
  const int ki = static_cast<int>(k);
  const int k1 = ki / 2;
  const double hi = Kokkos::ldexp(Kokkos::ldexp(res.hi, k1), ki - k1);
  if (Kokkos::isinf(hi) || hi == 0.0) return dd(hi);
  return dd(hi, Kokkos::ldexp(Kokkos::ldexp(res.lo, k1), ki - k1));
}

// e^a - 1 with relative precision: expm1_reduced for |a| <= ln 2 / 2 (where
// e^a - 1 would cancel), exp(a) - 1 beyond.
LAPLACE_HD dd expm1(const dd& a) {
  if (Kokkos::fabs(a.hi) <= 0.34657359027997264) return expm1_reduced(a);
  return exp(a) - 1.0;
}

// Logarithm by one Newton step. The argument is first split as a = m 2^e with
// m in [sqrt(1/2), sqrt(2)) (m exact, e an integer), so that log a = log m + e ln 2.
// Then, with x0 = log(m.hi) in double (laplace::math::log),
//
//   x1 = x0 + m exp(-x0) - 1      (double-double),
//
// which is the Newton step for exp(x) = m. If m exp(-x0) = 1 + eps and
// delta = log m - x0, then eps = e^delta - 1 and x1 = log m + delta^2/2 + ...,
// so the error of x1 is about delta^2/2, below 2^-107 absolute; the next terms
// of log(1 + u) are added below, which keeps the relative precision where
// log m is itself small.
// The split is needed because exp(-x0) overflows when a < 2^-1000; the
// reduction keeps exp(-x0) in [0.7, 1.4]. Then log a = x1 + e ln 2, with ln 2
// as a double-double. a = 1 gives exactly 0 (m = 1, x0 = 0, eps = 0).
// a < 0 and NaN give NaN, 0 gives -inf, +inf gives +inf.
LAPLACE_HD dd log(const dd& a) {
  if (!(a.hi >= 0.0)) return dd(laplace::math::detail::quiet_nan());
  if (a.hi == 0.0) return dd(-laplace::math::detail::pos_inf());
  if (a.hi == laplace::math::detail::pos_inf()) return dd(a.hi);
  constexpr dd LN2(0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56);
  int e = 0;
  Kokkos::frexp(a.hi, &e);
  if (Kokkos::ldexp(a.hi, -e) < 0.70710678118654752440) --e;
  const dd m(Kokkos::ldexp(a.hi, -e), Kokkos::ldexp(a.lo, -e));
  if (e == 0) {
    // near 1 the result is small and must keep its relative precision: with
    // v = m - 1 (exact), x0 = log1p(v.hi) and the Newton step for
    // expm1(x) = v, x1 = x0 - (expm1(x0) - v) / (1 + expm1(x0)), where
    // expm1 has relative precision (the step above would work with
    // m exp(-x0) ~ 1, whose double-double precision is absolute)
    const dd v = m - 1.0;
    if (Kokkos::fabs(v.hi) < 0.25) {
      const double x0 = Kokkos::log1p(v.hi);
      const dd E = expm1_reduced(dd(x0));
      return dd(x0) - (E - v) / (E + 1.0);
    }
  }
  const double x0 = laplace::math::log(m.hi);
  const dd q = m * exp(dd(-x0));
  // log q = u - u^2/2 + u^3/3 - ..., u = q - 1 of order 2^-52: the u^2 term
  // matters where log m itself is small (m near 1, x0 = 0), and u^3 / 3 keeps
  // the relative precision there
  const dd u = q - 1.0;
  const double u2 = mul(u.hi, u.hi);
  const dd x1 = dd(x0) + (u + (mul(u2, u.hi) / 3.0 - 0.5 * u2));
  return x1 + dd(static_cast<double>(e)) * LN2;
}

// log x for a double x > 0 (finite, subnormals included), as a double-double
// whose absolute error is that of log m, not |log x| 2^-53: with x = m 2^e,
// m in [sqrt(1/2), sqrt(2)), log x = e LN2_1 + (e LN2_2 + log m), where LN2_1
// has 42 significant bits, so e LN2_1 is exact for |e| < 2^11, and LN2_2 is the
// rest of ln 2 to a double. The error is about 2^-53 |log m| + 2^-53 ulp(log m)
// <= 1.2e-16 absolute: enough where the large part of a logarithm comes from
// the exponent (a^x x^a with small x, a probability far in a tail), at the
// cost of one plain log. Not a substitute for ddm::log where |log m| itself
// must carry more than 53 bits.
LAPLACE_HD dd log_split(double x) {
  constexpr double LN2_1 = 0x1.62e42fefa38p-1;
  constexpr double LN2_2 = 0x1.ef35793c76730p-45;
  // x = m 2^e from the bits (a library frexp costs more than the log on gcc);
  // subnormals are first scaled by 2^54
  std::uint64_t u = laplace::math::detail::to_bits(x);
  int e = -1022;
  if (u < (std::uint64_t(1) << 52)) {
    u = laplace::math::detail::to_bits(x * 0x1p54);
    e -= 54;
  }
  e += static_cast<int>(u >> 52);
  // m in [1, 2), then in [sqrt(1/2), sqrt(2))
  double m = laplace::math::detail::from_bits((u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL);
  e -= 1;
  if (m >= 1.41421356237309504880) {
    m *= 0.5;
    ++e;
  }
  const double ed = static_cast<double>(e);
  return two_sum(ed * LN2_1, add(mul(ed, LN2_2), laplace::math::log(m)));
}

}  // namespace laplace::math::detail::ddm
