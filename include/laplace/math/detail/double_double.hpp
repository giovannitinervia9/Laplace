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

}  // namespace laplace::math::detail::ddm
