// Helpers shared by the scalar bijectors (laplace::bijector::detail_scalar):
// constants, small polynomials and powers, and the families of auxiliary
// quantities (tanh, r and s, the cloglog inverse, the loglog polynomials)
// from which the derivatives are written. Internal.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/detail/double_double.hpp>

// The dispatch functions of scalar.hpp hold the code of every map; inlined into one kernel they
// made a single device function too large for ptxas (memory allocation failure on 2026-11-12, the
// tests of the maps with parameters). On the device they are separate functions.
#if defined(__CUDACC__)
#define LAPLACE_BIJ_NOINLINE __noinline__
#else
#define LAPLACE_BIJ_NOINLINE
#endif

namespace laplace::bijector {

namespace detail_scalar {

namespace m = laplace::math;
namespace dn = laplace::math::own::detail_normal;

constexpr double pi = 3.14159265358979323846;
constexpr double log_pi = 1.14472988584940017414;
constexpr double log_2 = 0.69314718055994530942;
constexpr double log_4 = 1.38629436111989061883;
constexpr double half_log_2pi = 0.91893853320467274178;  // log sqrt(2 pi)
constexpr double inv_sqrt_2pi = 0.39894228040143267794;
LAPLACE_HD_FORCEINLINE double nan() { return m::detail::quiet_nan(); }
LAPLACE_HD_FORCEINLINE double inf() { return m::detail::pos_inf(); }

template <int N>
LAPLACE_HD_FORCEINLINE double poly(const double (&c)[N], double x) {  // c[0] + c[1] x + ...
  double s = c[N - 1];
  for (int i = N - 2; i >= 0; --i) s = s * x + c[i];
  return s;
}

template <int K>
LAPLACE_HD_FORCEINLINE double ipow(double x) {
  double r = 1.0;
  for (int i = 0; i < K; ++i) r *= x;
  return r;
}

template <int K>
constexpr double fact() {  // K!
  double f = 1.0;
  for (int i = 2; i <= K; ++i) f *= i;
  return f;
}

template <int K>
constexpr double sgn() {  // (-1)^K
  return K % 2 == 0 ? 1.0 : -1.0;
}

// S_K(a, b) = sum_(j=0..K-1) a^(K-1-j) b^j, every term positive for a, b > 0
template <int K>
LAPLACE_HD_FORCEINLINE double geo_sum(double a, double b) {
  double s = 0.0, ap = ipow<K - 1>(a), bp = 1.0;
  const double ia = 1.0 / a;
  for (int j = 0; j < K; ++j) {
    s += ap * bp;
    ap *= ia;
    bp *= b;
  }
  return s;
}

// log(1 + x^2) without overflow of x^2
LAPLACE_HD_FORCEINLINE double log1p_sq(double x) {
  const double ax = Kokkos::fabs(x);
  if (ax < 1.0) return Kokkos::log1p(x * x);
  const double i = 1.0 / ax;
  return 2.0 * m::log(ax) + Kokkos::log1p(i * i);
}

// c a^K with the constant multiplied first, so that a partial product does not
// go subnormal before the result does
template <int K>
LAPLACE_HD_FORCEINLINE double cpow(double c, double a) {
  for (int i = 0; i < K; ++i) c *= a;
  return c;
}
// d^K/dv^K of c log v, K >= 1: c (-1)^(K-1) (K-1)! / v^K
template <int K>
LAPLACE_HD_FORCEINLINE double log_dk(double c, double v) {
  return cpow<K>(c * sgn<K - 1>() * fact<K - 1>(), 1.0 / v);
}

// ---------------------------------------------------------------- shared families

// log cosh u: below |u| = 1 as log1p(cosh u - 1), cosh u - 1 = em^2/(2 (1 + em)),
// em = expm1(|u|), since |u| + log1p(e^-2|u|) - log 2 cancels to u^2/2 there
LAPLACE_HD_FORCEINLINE double log_cosh(double u) {
  const double au = Kokkos::fabs(u);
  if (au < 1.0) {
    const double em = Kokkos::expm1(au);
    return Kokkos::log1p(em * em / (2.0 * (1.0 + em)));
  }
  return au + Kokkos::log1p(m::exp(-2.0 * au)) - log_2;
}
// tanh: T and S = 1 - T^2 from e = exp(-2|u|)
struct tanh_parts {
  double T, S;
};
LAPLACE_HD_FORCEINLINE tanh_parts tanh_of(double u) {
  const double au = Kokkos::fabs(u), e = m::exp(-2.0 * au), d = 1.0 + e;
  const double T = Kokkos::copysign(-Kokkos::expm1(-2.0 * au) / d, u);
  return {T, 4.0 * e / (d * d)};
}
// d^K tanh u = S P_K(T), K >= 1; P_K in T^2 while T^2 < 1/2, in S = 1 - T^2
// beyond, where the T^2 form cancels
template <int K>
LAPLACE_HD_FORCEINLINE double tanh_dk(double u) {
  const auto p = tanh_of(u);
  const double T = p.T, T2 = T * T, S = p.S;
  const bool small = T2 < 0.5;
  if constexpr (K == 1) return S;
  if constexpr (K == 2) return S * (-2.0 * T);
  if constexpr (K == 3) return S * (small ? 6.0 * T2 - 2.0 : 4.0 - 6.0 * S);
  if constexpr (K == 4) return S * T * (small ? 16.0 - 24.0 * T2 : 24.0 * S - 8.0);
  if constexpr (K == 5) {
    constexpr double c[] = {16.0, -120.0, 120.0};  // the same in T^2 and in S
    return S * poly(c, small ? T2 : S);
  }
  if constexpr (K == 6) {
    constexpr double a[] = {-272.0, 960.0, -720.0}, b[] = {-32.0, 480.0, -720.0};
    return S * T * (small ? poly(a, T2) : poly(b, S));
  }
}

// r = 1/hypot(1, x), s = x r
struct rs_parts {
  double r, s;
};
LAPLACE_HD_FORCEINLINE rs_parts rs_of(double x) {
  const double r = 1.0 / Kokkos::hypot(1.0, x);
  return {r, x * r};
}
// The polynomials below are in s^2 while s^2 < 1/2 and in r^2 = 1 - s^2
// beyond, where the s^2 forms cancel (their terms grow to 50 times the result).
// d^M r = r^(M+1) V_M(s)
template <int M>
LAPLACE_HD_FORCEINLINE double r_dm(double x) {
  const auto p = rs_of(x);
  const double s = p.s, s2 = s * s, r2 = p.r * p.r, rp = ipow<M + 1>(p.r);
  const bool small = s2 < 0.5;
  if constexpr (M == 0) return rp;
  if constexpr (M == 1) return rp * (-s);
  if constexpr (M == 2) return rp * (small ? 3.0 * s2 - 1.0 : 2.0 - 3.0 * r2);
  if constexpr (M == 3) return rp * s * (small ? 9.0 - 15.0 * s2 : 15.0 * r2 - 6.0);
  if constexpr (M == 4) {
    constexpr double a[] = {9.0, -90.0, 105.0}, b[] = {24.0, -120.0, 105.0};
    return rp * (small ? poly(a, s2) : poly(b, r2));
  }
  if constexpr (M == 5) {
    constexpr double a[] = {-225.0, 1050.0, -945.0}, b[] = {-120.0, 840.0, -945.0};
    return rp * s * (small ? poly(a, s2) : poly(b, r2));
  }
}
// d^M r^2 = r^(M+2) W_M(s)
template <int M>
LAPLACE_HD_FORCEINLINE double r2_dm(double x) {
  const auto p = rs_of(x);
  const double s = p.s, s2 = s * s, r2 = p.r * p.r, rp = ipow<M + 2>(p.r);
  const bool small = s2 < 0.5;
  if constexpr (M == 0) return rp;
  if constexpr (M == 1) return rp * (-2.0 * s);
  if constexpr (M == 2) return rp * (small ? 8.0 * s2 - 2.0 : 6.0 - 8.0 * r2);
  if constexpr (M == 3) return rp * s * (small ? 24.0 - 48.0 * s2 : 48.0 * r2 - 24.0);
  if constexpr (M == 4) {
    constexpr double a[] = {24.0, -288.0, 384.0}, b[] = {120.0, -480.0, 384.0};
    return rp * (small ? poly(a, s2) : poly(b, r2));
  }
  if constexpr (M == 5) {
    constexpr double a[] = {-720.0, 3840.0, -3840.0};  // the same in s^2 and in r^2
    return rp * s * poly(a, small ? s2 : r2);
  }
}
// d^M (s r) = r^(M+1) X_M(s)
template <int M>
LAPLACE_HD_FORCEINLINE double sr_dm(double x) {
  const auto p = rs_of(x);
  const double s = p.s, s2 = s * s, r2 = p.r * p.r, rp = ipow<M + 1>(p.r);
  const bool small = s2 < 0.5;
  if constexpr (M == 0) return rp * s;
  if constexpr (M == 1) return rp * (small ? 1.0 - 2.0 * s2 : 2.0 * r2 - 1.0);
  if constexpr (M == 2) return rp * s * (small ? 8.0 * s2 - 6.0 : 2.0 - 8.0 * r2);
  if constexpr (M == 3) {
    constexpr double a[] = {-6.0, 48.0, -48.0};  // the same in s^2 and in r^2
    return rp * poly(a, small ? s2 : r2);
  }
  if constexpr (M == 4) {
    constexpr double a[] = {120.0, -480.0, 384.0}, b[] = {24.0, -288.0, 384.0};
    return rp * s * (small ? poly(a, s2) : poly(b, r2));
  }
  if constexpr (M == 5) {
    constexpr double a[] = {120.0, -2160.0, 5760.0, -3840.0}, b[] = {-120.0, 2160.0, -5760.0, 3840.0};
    return rp * (small ? poly(a, s2) : poly(b, r2));
  }
}

// d^K/du^K (1 - exp(-e^u)), K >= 1
template <int K>
LAPLACE_HD_FORCEINLINE double cloglog_inv_dk(double u) {
  const double w = m::exp(u);
  if (!(w <= 1000.0)) return w > 1000.0 ? 0.0 : nan();
  const double e = m::exp(u - w);
  if constexpr (K == 1) return e;
  if constexpr (K == 2) return e * -Kokkos::expm1(u);  // 1 - w, exact near u = 0
  if constexpr (K == 3) {
    constexpr double c[] = {1.0, -3.0, 1.0};
    return e * poly(c, w);
  }
  if constexpr (K == 4) {
    constexpr double c[] = {1.0, -7.0, 6.0, -1.0};
    return e * poly(c, w);
  }
  if constexpr (K == 5) {
    constexpr double c[] = {1.0, -15.0, 25.0, -10.0, 1.0};
    return e * poly(c, w);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {1.0, -31.0, 90.0, -65.0, 15.0, -1.0};
    return e * poly(c, w);
  }
}

// phi_(s) = log(-log s) and psi(s) = -t - log(-t): K-th derivatives in s from
// t = log s and c = 1/(s t)
template <int K>
LAPLACE_HD_FORCEINLINE double C_poly(double t) {
  if constexpr (K == 1) return 1.0;
  if constexpr (K == 2) return -(1.0 + t);
  if constexpr (K == 3) {
    constexpr double c[] = {2.0, 3.0, 2.0};
    return poly(c, t);
  }
  if constexpr (K == 4) {
    constexpr double c[] = {-6.0, -12.0, -11.0, -6.0};
    return poly(c, t);
  }
  if constexpr (K == 5) {
    constexpr double c[] = {24.0, 60.0, 70.0, 50.0, 24.0};
    return poly(c, t);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {-120.0, -360.0, -510.0, -450.0, -274.0, -120.0};
    return poly(c, t);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double E_poly(double t) {
  if constexpr (K == 1) return -(1.0 + t);
  if constexpr (K == 2) {
    constexpr double c[] = {1.0, 1.0, 1.0};
    return poly(c, t);
  }
  if constexpr (K == 3) {
    constexpr double c[] = {-2.0, -3.0, -2.0, -2.0};
    return poly(c, t);
  }
  if constexpr (K == 4) {
    constexpr double c[] = {6.0, 12.0, 11.0, 6.0, 6.0};
    return poly(c, t);
  }
  if constexpr (K == 5) {
    constexpr double c[] = {-24.0, -60.0, -70.0, -50.0, -24.0, -24.0};
    return poly(c, t);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {120.0, 360.0, 510.0, 450.0, 274.0, 120.0, 120.0};
    return poly(c, t);
  }
}

// a^n for an integer n (|n| <= 12 here) by repeated multiplication or division, at most |n|
// roundings: the integer powers of softplus' sharpness
LAPLACE_HD_FORCEINLINE double pow_int(double a, int n) {
  double r = 1.0;
  if (n >= 0)
    for (int i = 0; i < n; ++i) r *= a;
  else
    for (int i = 0; i < -n; ++i) r /= a;
  return r;
}

// x^y for x > 0. On the device the platform's pow (libdevice, under 1 ns per call on the RTX
// 4070 SUPER). On the host exp(y log x) with log x from ddm::log_split (absolute error about
// 1.2e-16) and the exponent carried as a double-double: the platform's pow of the gcc build
// (Rtools) cost about 150 ns per call (measured 2026-11-12, benchmarks/bijector_scalar.cpp), against
// about 15 here. The relative error is about |y| / 2 ulp plus that of exp; outside the normal
// range of the result, and for x <= 0, non-finite x or y, the platform's pow.
LAPLACE_HD_FORCEINLINE double pow_pos(double x, double y) {
  KOKKOS_IF_ON_DEVICE((return Kokkos::pow(x, y);))
  KOKKOS_IF_ON_HOST((
    if (!(x > 0.0) || !Kokkos::isfinite(x) || !Kokkos::isfinite(y)) return Kokkos::pow(x, y);
    const m::detail::ddm::dd t = m::detail::ddm::log_split(x) * y;
    if (!(Kokkos::fabs(t.hi) < 700.0)) return Kokkos::pow(x, y);
    return m::exp(t.hi) * (1.0 + t.lo);))
}

}  // namespace detail_scalar

}  // namespace laplace::bijector
