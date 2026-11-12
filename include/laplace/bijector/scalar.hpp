// Scalar bijectors (laplace::bijector), the maps without parameters.
//
// A bijector is a bijection g: V -> U between open intervals, with inverse
// h = g^-1. Forward goes from the constrained side to the unconstrained one,
// in the direction that statistics names the map (decisions of 2026-11-12,
// note/bijectors.md): log maps (0, inf) onto R, logit maps (0, 1) onto R. A
// link evaluates h (theta = h(eta)), a transformer g (z = g(y)), a
// constraint h (theta = h(u)). Inverting a bijector (scalar_bijector::inverted) swaps
// every pair below; exp is log inverted.
//
// For each map, at a point, as member functions of scalar_bijector b:
//   b.forward(v) = g(v)                      b.inverse(u) = h(u)
//   b.forward_dK(v) = g^(K)(v), K = 1..6     b.inverse_dK(u) = h^(K)(u)
//   b.forward_log_jacobian(v) = log |g'(v)|  b.inverse_log_jacobian(u) = log |h'(u)|
//   ..._log_jacobian_dK: their derivatives, K = 1..6
//   b.inverse_distance_upper(u), b.inverse_distance_lower(u): the distances
//   of h(u) to the bounds of its range, exact where 1 - h(u) would cancel
// One function per order; each computes only its own order.
//
// Every formula is written directly for its direction; none is obtained from
// the other direction by inversion, so that each is accurate where the other
// cancels. The polynomials below were derived and checked with SymPy
// (tools/reference/bijector_forms.py, SymPy 1.14.0). Notation:
//
//   logit   p = v or expit(u), q = 1 - p, a = 1/p, b = 1/q, U = p q,
//           D = q - p = -tanh(u/2). Forward: g^(K) = (K-1)! ((-1)^(K-1) a^K + b^K),
//           written for even K as (K-1)! (2p - 1) a b S_K(a, b), with
//           S_K(a, b) = sum_j a^(K-1-j) b^j, since b^K - a^K = (b - a) S_K and
//           b - a = (2p - 1) a b; so no difference of two large terms is formed.
//           Inverse: h^(K) = U (A_K(U) + D B_K(U)): U, U D, U (1 - 6U),
//           U D (1 - 12U), U (1 - 30U + 120U^2), U D (1 - 60U + 360U^2).
//           p, q, U and D come from e = exp(-|u|), with D = -sign(u) |expm1(-|u|)|/(1 + e).
//   atanh   a = 1/(1 - v), b = 1/(1 + v); g^(K) = (K-1)!/2 (a^K + (-1)^(K-1) b^K),
//           for even K = (K-1)! v a b S_K(a, b). Inverse T = tanh u, S = 1 - T^2,
//           both from e = exp(-2|u|): h^(K) = S P_K(T), P = 1, -2T, 6T^2 - 2,
//           T (16 - 24T^2), 16 - 120T^2 + 120T^4, T (-272 + 960T^2 - 720T^4).
//   probit  z = Phi^-1(v), phi = phi(z) = v/m(z), m(z) = Phi(z)/phi(z) =
//           sqrt(pi/2) erfcx(-z/sqrt 2) (1 - v and -z above 1/2): g^(K) = Q_K(z)/phi^K, Q = 1, z, 1 + 2z^2,
//           z (7 + 6z^2), 7 + 46z^2 + 24z^4, z (127 + 326z^2 + 120z^4), all of one
//           sign in each parity, so no cancellation. Inverse:
//           h^(K) = (-1)^(K-1) He_(K-1)(u) phi(u). Forward log-Jacobian
//           -log phi(z) = z^2/2 + log sqrt(2 pi), derivatives R_K(z)/phi^K,
//           R = z, 1 + z^2, z (4 + 2z^2), 4 + 18z^2 + 6z^4, z (52 + 96z^2 + 24z^4),
//           52 + 548z^2 + 600z^4 + 120z^6.
//   cauchit z = g(v) = tan(pi (v - 1/2)), computed as -1/tan(pi v) below 1/4,
//           1/tan(pi (1 - v)) above 3/4 (1 - v exact there), so that no argument
//           near the pole is formed; g^(K) = pi^K T_K(z), T the derivative
//           polynomials of tan: 1 + z^2, 2z (1 + z^2), 2 + 8z^2 + 6z^4,
//           z (16 + 40z^2 + 24z^4), 16 + 136z^2 + 240z^4 + 120z^6,
//           z (272 + 1232z^2 + 1680z^4 + 720z^6). Inverse
//           h = atan2(1, -u)/pi (no cancellation as u -> -inf) and, with
//           r = 1/hypot(1, u) and s = u r (|s| < 1, r^2 = 1 - s^2),
//           d^m r^2 = r^(m+2) W_m(s), so h^(K) = r^(K+1) W_(K-1)(s)/pi.
//   asinh   same r, s at v: g^(K) = d^(K-1) r = r^K V_(K-1)(s); log-Jacobian
//           log r, derivatives -d^(K-1)(s r) = -r^K X_(K-1)(s). The inverse
//           log-Jacobian of cauchit is -2 d^(K-1)(s r) likewise.
//   cloglog Inverse h = -expm1(-w), w = e^u: h^(K) = exp(u - w) P_K(w),
//           P = 1, 1 - w, 1 - 3w + w^2, 1 - 7w + 6w^2 - w^3,
//           1 - 15w + 25w^2 - 10w^3 + w^4, 1 - 31w + 90w^2 - 65w^3 + 15w^4 - w^5
//           (0 once w > 1000, where exp(u - w) has underflowed). Forward
//           g(v) = phi_(1 - v), phi_(s) = log(-log s): with t = log s
//           (log1p(-v) for cloglog) and c = 1/(s t), phi_^(K)(s) = C_K(t) c^K,
//           C = 1, -(1 + t), 2 + 3t + 2t^2, -(6 + 12t + 11t^2 + 6t^3),
//           24 + 60t + 70t^2 + 50t^3 + 24t^4, -(120 + 360t + 510t^2 + 450t^3
//           + 274t^4 + 120t^5); g^(K) = (-1)^K phi_^(K)(1 - v). The forward
//           log-Jacobian is psi(s) = -t - log(-t), psi^(K) = E_K(t) c^K with
//           E_K = -C_K + (-1)^K (K-1)! t^K.
//   loglog  g(v) = -phi_(v) with s = v, t = log v; h(u) = exp(-exp(-u)) =
//           1 - h_cloglog(-u), so h^(K)(u) = (-1)^(K+1) h_cloglog^(K)(-u).
//           Forward log-Jacobian psi(v).
//   log, inverse, inverse_sq, sqrt: powers and logarithms of a = 1/v.
//
// Outside the closed domain (or NaN) every function returns NaN. On a
// bound the formulas give their limits (logit(0) = -inf). Nothing is clamped
// here: where a value saturates or a derivative under- or overflows is the
// subject of note/clamping/, still open.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

namespace laplace::bijector {

// The maps, named by their forward direction.
/**
 * \ingroup bijector_scalar
 * \brief Enumeration of parameter-free scalar bijector maps.
 *
 * Defines twelve bijections \f$g: V \to U\f$ between open intervals:
 * - \c identity : \f$g(v) = v\f$, mapping \f$(-\infty, \infty) \to (-\infty, \infty)\f$.
 * - \c log : \f$g(v) = \log v\f$, mapping \f$(0, \infty) \to (-\infty, \infty)\f$.
 * - \c logit : \f$g(v) = \log(v / (1 - v))\f$, mapping \f$(0, 1) \to (-\infty, \infty)\f$.
 * - \c probit : \f$g(v) = \Phi^{-1}(v)\f$, mapping \f$(0, 1) \to (-\infty, \infty)\f$.
 * - \c cauchit : \f$g(v) = \tan(\pi(v - 1/2))\f$, mapping \f$(0, 1) \to (-\infty, \infty)\f$.
 * - \c cloglog : \f$g(v) = \log(-\log(1 - v))\f$, mapping \f$(0, 1) \to (-\infty, \infty)\f$.
 * - \c loglog : \f$g(v) = -\log(-\log v)\f$, mapping \f$(0, 1) \to (-\infty, \infty)\f$.
 * - \c inverse : \f$g(v) = 1 / v\f$, mapping \f$(0, \infty) \to (0, \infty)\f$.
 * - \c inverse_sq : \f$g(v) = 1 / v^2\f$, mapping \f$(0, \infty) \to (0, \infty)\f$.
 * - \c sqrt : \f$g(v) = \sqrt{v}\f$, mapping \f$(0, \infty) \to (0, \infty)\f$.
 * - \c atanh : \f$g(v) = \operatorname{atanh}(v) = \frac{1}{2}(\log(1+v) - \log(1-v))\f$, mapping \f$(-1, 1) \to (-\infty, \infty)\f$.
 * - \c asinh : \f$g(v) = \operatorname{asinh}(v)\f$, mapping \f$(-\infty, \infty) \to (-\infty, \infty)\f$.
 *
 * \see bijector_scalar, scalar_bijector
 */
enum class scalar_bijector_map : int {
  identity,    // R -> R
  log,         // (0, inf) -> R
  logit,       // (0, 1) -> R
  probit,      // (0, 1) -> R, Phi^-1
  cauchit,     // (0, 1) -> R, tan(pi (v - 1/2))
  cloglog,     // (0, 1) -> R, log(-log(1 - v))
  loglog,      // (0, 1) -> R, -log(-log v)
  inverse,     // (0, inf) -> (0, inf), 1/v
  inverse_sq,  // (0, inf) -> (0, inf), 1/v^2
  sqrt,        // (0, inf) -> (0, inf)
  atanh,       // (-1, 1) -> R (the rhobit link, Fisher's z)
  asinh        // R -> R
};

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

// ---------------------------------------------------------------- the maps
// fwd<K>(v): g^(K)(v), K = 0 the value; inv<K>(u): h^(K)(u);
// fwd_lj<K>(v): d^K/dv^K log|g'(v)|; inv_lj<K>(u): d^K/du^K log|h'(u)|.

// identity
template <int K>
LAPLACE_HD_FORCEINLINE double identity_fwd(double v) {
  if constexpr (K == 0) return v;
  if constexpr (K == 1) return 1.0;
  return 0.0;
}

// log; its inverse is exp
template <int K>
LAPLACE_HD_FORCEINLINE double log_fwd(double v) {
  if constexpr (K == 0) return m::log(v);
  else return log_dk<K>(1.0, v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double log_inv(double u) {
  return m::exp(u);
}
template <int K>
LAPLACE_HD_FORCEINLINE double log_fwd_lj(double v) {  // -log v
  if constexpr (K == 0) return -m::log(v);
  else return log_dk<K>(-1.0, v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double log_inv_lj(double u) {  // u
  if constexpr (K == 0) return u;
  if constexpr (K == 1) return 1.0;
  return 0.0;
}

// logit
template <int K>
LAPLACE_HD_FORCEINLINE double logit_fwd(double v) {
  const double p = v, q = 1.0 - v;  // q exact for v >= 1/2
  if constexpr (K == 0) {
    const double lp = v < 0.5 ? m::log(p) : Kokkos::log1p(-q);
    const double lq = v < 0.5 ? Kokkos::log1p(-p) : m::log(q);
    return lp - lq;
  } else {
    const double a = 1.0 / p, b = 1.0 / q;
    if constexpr (K % 2 == 1) return fact<K - 1>() * (ipow<K>(a) + ipow<K>(b));
    else return fact<K - 1>() * (2.0 * p - 1.0) * a * b * geo_sum<K>(a, b);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double logit_fwd_lj(double v) {  // -log p - log q
  const double p = v, q = 1.0 - v;
  if constexpr (K == 0) {
    const double lp = v < 0.5 ? m::log(p) : Kokkos::log1p(-q);
    const double lq = v < 0.5 ? Kokkos::log1p(-p) : m::log(q);
    return -lp - lq;
  } else {
    const double a = 1.0 / p, b = 1.0 / q;
    if constexpr (K % 2 == 0) return fact<K - 1>() * (ipow<K>(a) + ipow<K>(b));
    else return fact<K - 1>() * (2.0 * p - 1.0) * a * b * geo_sum<K>(a, b);
  }
}
struct expit_parts {
  double p, U, D, e;
};
LAPLACE_HD_FORCEINLINE expit_parts expit_of(double u) {
  const double au = Kokkos::fabs(u), e = m::exp(-au), d = 1.0 + e;
  const double big = 1.0 / d, small = e / d;
  return {u >= 0.0 ? big : small, small * big, Kokkos::copysign(Kokkos::expm1(-au) / d, -u), e};  // D = q - p
}
template <int K>
LAPLACE_HD_FORCEINLINE double logit_inv(double u) {
  const auto x = expit_of(u);
  const double U = x.U, D = x.D;
  if constexpr (K == 0) return x.p;
  if constexpr (K == 1) return U;
  if constexpr (K == 2) return U * D;
  if constexpr (K == 3) return U * (1.0 - 6.0 * U);
  if constexpr (K == 4) return U * D * (1.0 - 12.0 * U);
  if constexpr (K == 5) {
    constexpr double c[] = {1.0, -30.0, 120.0};
    return U * poly(c, U);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {1.0, -60.0, 360.0};
    return U * D * poly(c, U);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double logit_inv_lj(double u) {  // log p + log q
  if constexpr (K == 0) {
    const auto x = expit_of(u);
    return -(Kokkos::fabs(u) + 2.0 * Kokkos::log1p(x.e));
  } else if constexpr (K == 1) {
    return expit_of(u).D;
  } else {
    return -2.0 * logit_inv<K - 1>(u);
  }
}

// probit. 1/phi(z) from v, not from the rounded z: phi(z) = v / m(z) with
// m(z) = Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2) for v <= 1/2 (and the
// mirror with 1 - v, exact there, above), since exp(z^2/2) at a rounded z
// carries a relative error z^2 eps (up to 1400 ulp at |z| = 37)
LAPLACE_HD_FORCEINLINE double probit_inv_phi(double v, double z) {
  constexpr double sqrt_half_pi = 1.25331413731550025121;  // sqrt(pi/2)
  constexpr double sqrt_half = 0.70710678118654752440;
  if (!(v > 0.0 && v < 1.0)) return inf();
  if (z <= 0.0) return sqrt_half_pi * m::erfcx(-z * sqrt_half) / v;
  return sqrt_half_pi * m::erfcx(z * sqrt_half) / (1.0 - v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double probit_fwd(double v) {
  const double z = m::Phi_inv(v);
  if constexpr (K == 0) return z;
  else {
    const double ip = probit_inv_phi(v, z), z2 = z * z;
    double Q;
    if constexpr (K == 1) Q = 1.0;
    if constexpr (K == 2) Q = z;
    if constexpr (K == 3) Q = 1.0 + 2.0 * z2;
    if constexpr (K == 4) Q = z * (7.0 + 6.0 * z2);
    if constexpr (K == 5) {
      constexpr double c[] = {7.0, 46.0, 24.0};
      Q = poly(c, z2);
    }
    if constexpr (K == 6) {
      constexpr double c[] = {127.0, 326.0, 120.0};
      Q = z * poly(c, z2);
    }
    return Q * ipow<K>(ip);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double probit_fwd_lj(double v) {  // z^2/2 + log sqrt(2 pi)
  const double z = m::Phi_inv(v);
  if constexpr (K == 0) {
    double h, l;
    dn::half_square(z, h, l);
    return (h + half_log_2pi) + l;
  } else {
    const double ip = probit_inv_phi(v, z), z2 = z * z;
    double R;
    if constexpr (K == 1) R = z;
    if constexpr (K == 2) R = 1.0 + z2;
    if constexpr (K == 3) R = z * (4.0 + 2.0 * z2);
    if constexpr (K == 4) {
      constexpr double c[] = {4.0, 18.0, 6.0};
      R = poly(c, z2);
    }
    if constexpr (K == 5) {
      constexpr double c[] = {52.0, 96.0, 24.0};
      R = z * poly(c, z2);
    }
    if constexpr (K == 6) {
      constexpr double c[] = {52.0, 548.0, 600.0, 120.0};
      R = poly(c, z2);
    }
    return R * ipow<K>(ip);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double probit_inv(double u) {
  if constexpr (K == 0) return m::Phi(u);
  else {
    const double ph = inv_sqrt_2pi * dn::exp_minus_half_square(u), u2 = u * u;
    if constexpr (K == 1) return ph;
    if constexpr (K == 2) return -u * ph;
    if constexpr (K == 3) return (u2 - 1.0) * ph;
    if constexpr (K == 4) return u * (3.0 - u2) * ph;
    if constexpr (K == 5) {
      constexpr double c[] = {3.0, -6.0, 1.0};
      return poly(c, u2) * ph;
    }
    if constexpr (K == 6) {
      constexpr double c[] = {-15.0, 10.0, -1.0};
      return u * poly(c, u2) * ph;
    }
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double probit_inv_lj(double u) {  // log phi(u)
  if constexpr (K == 0) {
    double h, l;
    dn::half_square(u, h, l);
    return -(h + half_log_2pi) - l;
  }
  if constexpr (K == 1) return -u;
  if constexpr (K == 2) return -1.0;
  return 0.0;
}

// cauchit
LAPLACE_HD_FORCEINLINE double cauchit_z(double v) {
  if (v < 0.25) return -1.0 / Kokkos::tan(pi * v);
  if (v > 0.75) return 1.0 / Kokkos::tan(pi * (1.0 - v));
  return Kokkos::tan(pi * (v - 0.5));
}
template <int K>
LAPLACE_HD_FORCEINLINE double tan_poly(double z) {  // d^K tan at tan = z
  const double z2 = z * z;
  if constexpr (K == 1) return 1.0 + z2;
  if constexpr (K == 2) return 2.0 * z * (1.0 + z2);
  if constexpr (K == 3) {
    constexpr double c[] = {2.0, 8.0, 6.0};
    return poly(c, z2);
  }
  if constexpr (K == 4) {
    constexpr double c[] = {16.0, 40.0, 24.0};
    return z * poly(c, z2);
  }
  if constexpr (K == 5) {
    constexpr double c[] = {16.0, 136.0, 240.0, 120.0};
    return poly(c, z2);
  }
  if constexpr (K == 6) {
    constexpr double c[] = {272.0, 1232.0, 1680.0, 720.0};
    return z * poly(c, z2);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double cauchit_fwd(double v) {
  const double z = cauchit_z(v);
  if constexpr (K == 0) return z;
  else return ipow<K>(pi) * tan_poly<K>(z);
}
template <int K>
LAPLACE_HD_FORCEINLINE double cauchit_fwd_lj(double v) {  // log pi + log(1 + z^2)
  const double z = cauchit_z(v);
  if constexpr (K == 0) return log_pi + log1p_sq(z);
  if constexpr (K == 1) return 2.0 * pi * z;
  if constexpr (K >= 2) return 2.0 * ipow<K>(pi) * tan_poly<K - 1>(z);
}
template <int K>
LAPLACE_HD_FORCEINLINE double cauchit_inv(double u) {
  if constexpr (K == 0) return Kokkos::atan2(1.0, -u) / pi;
  else return r2_dm<K - 1>(u) / pi;
}
template <int K>
LAPLACE_HD_FORCEINLINE double cauchit_inv_lj(double u) {  // -log pi - log(1 + u^2)
  if constexpr (K == 0) return -log_pi - log1p_sq(u);
  else return -2.0 * sr_dm<K - 1>(u);
}

// cloglog
template <int K>
LAPLACE_HD_FORCEINLINE double cloglog_fwd(double v) {
  const double t = Kokkos::log1p(-v);
  if constexpr (K == 0) return m::log(-t);
  else return sgn<K>() * C_poly<K>(t) * ipow<K>(1.0 / ((1.0 - v) * t));
}
template <int K>
LAPLACE_HD_FORCEINLINE double cloglog_fwd_lj(double v) {  // -t - log(-t)
  const double t = Kokkos::log1p(-v);
  if constexpr (K == 0) return -t - m::log(-t);
  else return sgn<K>() * E_poly<K>(t) * ipow<K>(1.0 / ((1.0 - v) * t));
}
template <int K>
LAPLACE_HD_FORCEINLINE double cloglog_inv(double u) {
  if constexpr (K == 0) return -Kokkos::expm1(-m::exp(u));
  else return cloglog_inv_dk<K>(u);
}
template <int K>
LAPLACE_HD_FORCEINLINE double cloglog_inv_lj(double u) {  // u - e^u
  if constexpr (K == 0) return u - m::exp(u);
  if constexpr (K == 1) return -Kokkos::expm1(u);
  if constexpr (K >= 2) return -m::exp(u);
}

// loglog
template <int K>
LAPLACE_HD_FORCEINLINE double loglog_fwd(double v) {
  const double t = m::log(v);
  if constexpr (K == 0) return -m::log(-t);
  else return -C_poly<K>(t) * ipow<K>(1.0 / (v * t));
}
template <int K>
LAPLACE_HD_FORCEINLINE double loglog_fwd_lj(double v) {
  const double t = m::log(v);
  if constexpr (K == 0) return -t - m::log(-t);
  else return E_poly<K>(t) * ipow<K>(1.0 / (v * t));
}
template <int K>
LAPLACE_HD_FORCEINLINE double loglog_inv(double u) {
  if constexpr (K == 0) return m::exp(-m::exp(-u));
  else return sgn<K + 1>() * cloglog_inv_dk<K>(-u);
}
template <int K>
LAPLACE_HD_FORCEINLINE double loglog_inv_lj(double u) {  // -u - e^-u
  if constexpr (K == 0) return -u - m::exp(-u);
  if constexpr (K == 1) return Kokkos::expm1(-u);
  if constexpr (K >= 2) return sgn<K + 1>() * m::exp(-u);
}

// inverse: g = h = 1/v
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_fwd(double v) {
  const double a = 1.0 / v;
  if constexpr (K == 0) return a;
  else return cpow<K + 1>(sgn<K>() * fact<K>(), a);
}
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_fwd_lj(double v) {  // -2 log v
  if constexpr (K == 0) return -2.0 * m::log(v);
  else return log_dk<K>(-2.0, v);
}

// inverse_sq: g = v^-2, h = u^-1/2
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_sq_fwd(double v) {
  const double a = 1.0 / v;
  if constexpr (K == 0) return a * a;
  else return cpow<K + 2>(sgn<K>() * fact<K + 1>(), a);
}
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_sq_fwd_lj(double v) {  // log 2 - 3 log v
  if constexpr (K == 0) return log_2 - 3.0 * m::log(v);
  else return log_dk<K>(-3.0, v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_sq_inv(double u) {
  const double r = 1.0 / Kokkos::sqrt(u);
  if constexpr (K == 0) return r;
  else {
    constexpr double c[] = {1.0, -0.5, 0.75, -1.875, 6.5625, -29.53125, 162.421875};  // prod (-1/2 - j)
    return cpow<2 * K + 1>(c[K], r);  // u^(-1/2-K)
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double inverse_sq_inv_lj(double u) {  // -log 2 - 3/2 log u
  if constexpr (K == 0) return -log_2 - 1.5 * m::log(u);
  else return log_dk<K>(-1.5, u);
}

// sqrt: g = sqrt v, h = u^2
template <int K>
LAPLACE_HD_FORCEINLINE double sqrt_fwd(double v) {
  if constexpr (K == 0) return Kokkos::sqrt(v);
  else {
    constexpr double c[] = {1.0, 0.5, -0.25, 0.375, -0.9375, 3.28125, -14.765625};  // prod (1/2 - j)
    return cpow<2 * K - 1>(c[K], 1.0 / Kokkos::sqrt(v));  // v^(1/2-K)
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double sqrt_fwd_lj(double v) {  // -log 2 - 1/2 log v
  if constexpr (K == 0) return -log_2 - 0.5 * m::log(v);
  else return log_dk<K>(-0.5, v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double sqrt_inv(double u) {
  if constexpr (K == 0) return u * u;
  if constexpr (K == 1) return 2.0 * u;
  if constexpr (K == 2) return 2.0;
  return 0.0;
}
template <int K>
LAPLACE_HD_FORCEINLINE double sqrt_inv_lj(double u) {  // log 2 + log u
  if constexpr (K == 0) return log_2 + m::log(u);
  else return log_dk<K>(1.0, u);
}

// atanh (rhobit); its inverse is tanh
template <int K>
LAPLACE_HD_FORCEINLINE double atanh_fwd(double v) {
  if constexpr (K == 0) return 0.5 * (Kokkos::log1p(v) - Kokkos::log1p(-v));
  else {
    const double a = 1.0 / (1.0 - v), b = 1.0 / (1.0 + v);
    if constexpr (K % 2 == 1) return 0.5 * fact<K - 1>() * (ipow<K>(a) + ipow<K>(b));
    else return fact<K - 1>() * v * a * b * geo_sum<K>(a, b);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double atanh_fwd_lj(double v) {  // -log(1 - v) - log(1 + v)
  if constexpr (K == 0) {  // -log(1 - v^2); the two log1p cancel to v^2 near 0
    if (Kokkos::fabs(v) < 0.5) return -Kokkos::log1p(-v * v);
    return -Kokkos::log1p(-v) - Kokkos::log1p(v);
  }
  else {
    const double a = 1.0 / (1.0 - v), b = 1.0 / (1.0 + v);
    if constexpr (K % 2 == 0) return fact<K - 1>() * (ipow<K>(a) + ipow<K>(b));
    else return 2.0 * fact<K - 1>() * v * a * b * geo_sum<K>(a, b);
  }
}
template <int K>
LAPLACE_HD_FORCEINLINE double atanh_inv(double u) {
  if constexpr (K == 0) return tanh_of(u).T;
  else return tanh_dk<K>(u);
}
template <int K>
LAPLACE_HD_FORCEINLINE double atanh_inv_lj(double u) {  // log(1 - tanh^2 u)
  if constexpr (K == 0) return -2.0 * log_cosh(u);
  if constexpr (K == 1) return -2.0 * tanh_of(u).T;
  if constexpr (K >= 2) return -2.0 * tanh_dk<K - 1>(u);
}

// asinh; its inverse is sinh
template <int K>
LAPLACE_HD_FORCEINLINE double asinh_fwd(double v) {
  if constexpr (K == 0) return Kokkos::asinh(v);
  else return r_dm<K - 1>(v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double asinh_fwd_lj(double v) {  // -1/2 log(1 + v^2)
  if constexpr (K == 0) return -0.5 * log1p_sq(v);
  else return -sr_dm<K - 1>(v);
}
template <int K>
LAPLACE_HD_FORCEINLINE double asinh_inv(double u) {
  if constexpr (K % 2 == 0) return Kokkos::sinh(u);
  else return Kokkos::cosh(u);
}
template <int K>
LAPLACE_HD_FORCEINLINE double asinh_inv_lj(double u) {  // log cosh u
  if constexpr (K == 0) return log_cosh(u);
  if constexpr (K == 1) return tanh_of(u).T;
  if constexpr (K >= 2) return tanh_dk<K - 1>(u);
}

// ---------------------------------------------------------------- dispatch

LAPLACE_HD_FORCEINLINE double dom_lo(scalar_bijector_map k) {
  switch (k) {
    case scalar_bijector_map::identity:
    case scalar_bijector_map::asinh: return -inf();
    case scalar_bijector_map::atanh: return -1.0;
    default: return 0.0;
  }
}
LAPLACE_HD_FORCEINLINE double dom_hi(scalar_bijector_map k) {
  switch (k) {
    case scalar_bijector_map::logit:
    case scalar_bijector_map::probit:
    case scalar_bijector_map::cauchit:
    case scalar_bijector_map::cloglog:
    case scalar_bijector_map::loglog:
    case scalar_bijector_map::atanh: return 1.0;
    default: return inf();
  }
}
LAPLACE_HD_FORCEINLINE double img_lo(scalar_bijector_map k) {
  switch (k) {
    case scalar_bijector_map::inverse:
    case scalar_bijector_map::inverse_sq:
    case scalar_bijector_map::sqrt: return 0.0;
    default: return -inf();
  }
}
LAPLACE_HD_FORCEINLINE double img_hi(scalar_bijector_map) { return inf(); }

LAPLACE_HD_FORCEINLINE bool inside(double x, double lo, double hi) { return x >= lo && x <= hi; }

template <int K>
LAPLACE_HD double fwd(scalar_bijector_map k, double v) {
  if (!inside(v, dom_lo(k), dom_hi(k))) return nan();
  switch (k) {
    case scalar_bijector_map::identity: return identity_fwd<K>(v);
    case scalar_bijector_map::log: return log_fwd<K>(v);
    case scalar_bijector_map::logit: return logit_fwd<K>(v);
    case scalar_bijector_map::probit: return probit_fwd<K>(v);
    case scalar_bijector_map::cauchit: return cauchit_fwd<K>(v);
    case scalar_bijector_map::cloglog: return cloglog_fwd<K>(v);
    case scalar_bijector_map::loglog: return loglog_fwd<K>(v);
    case scalar_bijector_map::inverse: return inverse_fwd<K>(v);
    case scalar_bijector_map::inverse_sq: return inverse_sq_fwd<K>(v);
    case scalar_bijector_map::sqrt: return sqrt_fwd<K>(v);
    case scalar_bijector_map::atanh: return atanh_fwd<K>(v);
    case scalar_bijector_map::asinh: return asinh_fwd<K>(v);
  }
  return nan();
}
template <int K>
LAPLACE_HD double inv(scalar_bijector_map k, double u) {
  if (!inside(u, img_lo(k), img_hi(k))) return nan();
  switch (k) {
    case scalar_bijector_map::identity: return identity_fwd<K>(u);
    case scalar_bijector_map::log: return log_inv<K>(u);
    case scalar_bijector_map::logit: return logit_inv<K>(u);
    case scalar_bijector_map::probit: return probit_inv<K>(u);
    case scalar_bijector_map::cauchit: return cauchit_inv<K>(u);
    case scalar_bijector_map::cloglog: return cloglog_inv<K>(u);
    case scalar_bijector_map::loglog: return loglog_inv<K>(u);
    case scalar_bijector_map::inverse: return inverse_fwd<K>(u);
    case scalar_bijector_map::inverse_sq: return inverse_sq_inv<K>(u);
    case scalar_bijector_map::sqrt: return sqrt_inv<K>(u);
    case scalar_bijector_map::atanh: return atanh_inv<K>(u);
    case scalar_bijector_map::asinh: return asinh_inv<K>(u);
  }
  return nan();
}
template <int K>
LAPLACE_HD double fwd_lj(scalar_bijector_map k, double v) {
  if (!inside(v, dom_lo(k), dom_hi(k))) return nan();
  switch (k) {
    case scalar_bijector_map::identity: return 0.0;
    case scalar_bijector_map::log: return log_fwd_lj<K>(v);
    case scalar_bijector_map::logit: return logit_fwd_lj<K>(v);
    case scalar_bijector_map::probit: return probit_fwd_lj<K>(v);
    case scalar_bijector_map::cauchit: return cauchit_fwd_lj<K>(v);
    case scalar_bijector_map::cloglog: return cloglog_fwd_lj<K>(v);
    case scalar_bijector_map::loglog: return loglog_fwd_lj<K>(v);
    case scalar_bijector_map::inverse: return inverse_fwd_lj<K>(v);
    case scalar_bijector_map::inverse_sq: return inverse_sq_fwd_lj<K>(v);
    case scalar_bijector_map::sqrt: return sqrt_fwd_lj<K>(v);
    case scalar_bijector_map::atanh: return atanh_fwd_lj<K>(v);
    case scalar_bijector_map::asinh: return asinh_fwd_lj<K>(v);
  }
  return nan();
}
template <int K>
LAPLACE_HD double inv_lj(scalar_bijector_map k, double u) {
  if (!inside(u, img_lo(k), img_hi(k))) return nan();
  switch (k) {
    case scalar_bijector_map::identity: return 0.0;
    case scalar_bijector_map::log: return log_inv_lj<K>(u);
    case scalar_bijector_map::logit: return logit_inv_lj<K>(u);
    case scalar_bijector_map::probit: return probit_inv_lj<K>(u);
    case scalar_bijector_map::cauchit: return cauchit_inv_lj<K>(u);
    case scalar_bijector_map::cloglog: return cloglog_inv_lj<K>(u);
    case scalar_bijector_map::loglog: return loglog_inv_lj<K>(u);
    case scalar_bijector_map::inverse: return inverse_fwd_lj<K>(u);
    case scalar_bijector_map::inverse_sq: return inverse_sq_inv_lj<K>(u);
    case scalar_bijector_map::sqrt: return sqrt_inv_lj<K>(u);
    case scalar_bijector_map::atanh: return atanh_inv_lj<K>(u);
    case scalar_bijector_map::asinh: return asinh_inv_lj<K>(u);
  }
  return nan();
}

// The distances of h(u) to the bounds of its range (the domain of g), computed
// without forming h, where h saturates at a bound other than 0 or infinity:
// 1 - h(u) of logit is expit(-u), exact while 1 - expit(u) has rounded to 0
// from u = 37.4. Elsewhere the bound minus h(u) (exact for a bound 0 or infinite).
LAPLACE_HD double inv_to_upper(scalar_bijector_map k, double u) {
  if (!inside(u, img_lo(k), img_hi(k))) return nan();
  switch (k) {
    case scalar_bijector_map::logit: return logit_inv<0>(-u);
    case scalar_bijector_map::probit: return m::Phi(-u);
    case scalar_bijector_map::cauchit: return cauchit_inv<0>(-u);  // 1 - h(u) = h(-u)
    case scalar_bijector_map::cloglog: return m::exp(-m::exp(u));
    case scalar_bijector_map::loglog: return cloglog_inv<0>(-u);   // 1 - exp(-e^-u)
    case scalar_bijector_map::atanh: {                              // 1 - tanh u
      if (u <= 0.0) return 1.0 - tanh_of(u).T;
      const double e = m::exp(-2.0 * u);
      return 2.0 * e / (1.0 + e);
    }
    default: return dom_hi(k) - inv<0>(k, u);
  }
}
LAPLACE_HD double inv_to_lower(scalar_bijector_map k, double u) {
  if (k == scalar_bijector_map::atanh) return inv_to_upper(k, -u);  // 1 + tanh u
  if (!inside(u, img_lo(k), img_hi(k))) return nan();
  return inv<0>(k, u) - dom_lo(k);
}

}  // namespace detail_scalar

// ---------------------------------------------------------------- the bijector

// A scalar bijector: the map and the inversion flag (inverted: forward is h).
// Every quantity is a member function, one per order; an inverted bijector
// swaps the two directions of every pair.
/**
 * \ingroup bijector_scalar
 * \brief Configuration and evaluation struct for scalar bijectors without parameters.
 *
 * Represents a scalar bijection \f$g: V \to U\f$ between open intervals \f$V, U \subseteq \mathbb{R}\f$,
 * with inverse \f$h = g^{-1}: U \to V\f$.
 *
 * Holds two data members:
 * - \c map : the chosen bijection (\ref scalar_bijector_map).
 * - \c inverted : whether the bijector is inverted (\c bool).
 *
 * Every numerical quantity, derivative, log-Jacobian, domain bound, and distance to bounds is a member function.
 * Inverting a bijector (\c inverted = \c true) swaps every pair of forward and inverse operations:
 * forward() evaluates \f$h\f$, inverse() evaluates \f$g\f$, forward_log_jacobian() evaluates \f$\log|h'|\f$,
 * inverse_log_jacobian() evaluates \f$\log|g'|\f$, forward_domain_lower() and forward_domain_upper()
 * return the bounds of \f$U\f$, and inverse_domain_lower() and inverse_domain_upper() return the bounds of \f$V\f$.
 *
 * Example:
 * \code{.cpp}
 * const scalar_bijector logit{scalar_bijector_map::logit};
 * double theta = logit.inverse(eta);
 * \endcode
 *
 * \see bijector_scalar, scalar_bijector_map, forward(), inverse()
 */
struct scalar_bijector {
  scalar_bijector_map map = scalar_bijector_map::identity;
  bool inverted = false;

  // g and its derivatives; h and its derivatives
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the forward map \f$g(v)\f$.
   *
   * Computes \f$u = g(v)\f$ mapping domain \f$V\f$ to image \f$U\f$.
   * When inverted, evaluates \f$h(v)\f$ instead.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return The transformed value \f$g(v)\f$, or \f$h(v)\f$ if inverted.
   *
   * \par Domain and special values
   * - Standard domain is \f$[v_{\min}, v_{\max}]\f$ given by forward_domain_lower() and forward_domain_upper().
   * - On boundaries, returns the mathematical limits (for example, \f$\mathrm{logit}(0) = -\infty\f$, \f$\mathrm{logit}(1) = +\infty\f$, \f$\log(0) = -\infty\f$).
   * - If \f$v\f$ is strictly outside the closed domain or is NaN, returns quiet NaN.
   *
   * \par Measured accuracy
   * Over orders 0 to 6, maximum error in ulpC on gcc / CUDA: log 3.84 / 3.84; logit 3.17 / 3.17; probit 6.17 / 6.17; cauchit 6.19 / 6.19; cloglog 6.73 / 3.12; loglog 4.58 / 3.23; inverse 3.92 / 3.92; inverse_sq 3.09 / 3.09; sqrt 10.9 / 10.9; atanh 2.78 / 2.78; asinh 8.55 / 9.49. The measured plain-ulp errors of the forward value itself, on gcc, are 30.9 ulp for cloglog and 1050 ulp for loglog (near its zero at \f$v = 1/e\f$), within 4.1 ulp for the other maps.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), forward_d1(), forward_log_jacobian(), forward_domain_lower(), forward_domain_upper()
   */
  LAPLACE_HD_FORCEINLINE double forward(double v) const {
    return inverted ? detail_scalar::inv<0>(map, v) : detail_scalar::fwd<0>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief First derivative of the forward map, \f$g'(v) = \frac{\mathrm{d}}{\mathrm{d}v}g(v)\f$.
   *
   * Computes the first derivative of \f$g(v)\f$ (or \f$h'(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return First derivative \f$g'(v)\f$, or \f$h'(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d2()
   */
  LAPLACE_HD_FORCEINLINE double forward_d1(double v) const {
    return inverted ? detail_scalar::inv<1>(map, v) : detail_scalar::fwd<1>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Second derivative of the forward map, \f$g''(v) = \frac{\mathrm{d}^2}{\mathrm{d}v^2}g(v)\f$.
   *
   * Computes the second derivative of \f$g(v)\f$ (or \f$h''(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Second derivative \f$g''(v)\f$, or \f$h''(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d1(), forward_d3()
   */
  LAPLACE_HD_FORCEINLINE double forward_d2(double v) const {
    return inverted ? detail_scalar::inv<2>(map, v) : detail_scalar::fwd<2>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Third derivative of the forward map, \f$g^{(3)}(v) = \frac{\mathrm{d}^3}{\mathrm{d}v^3}g(v)\f$.
   *
   * Computes the third derivative of \f$g(v)\f$ (or \f$h^{(3)}(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Third derivative \f$g^{(3)}(v)\f$, or \f$h^{(3)}(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d2(), forward_d4()
   */
  LAPLACE_HD_FORCEINLINE double forward_d3(double v) const {
    return inverted ? detail_scalar::inv<3>(map, v) : detail_scalar::fwd<3>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fourth derivative of the forward map, \f$g^{(4)}(v) = \frac{\mathrm{d}^4}{\mathrm{d}v^4}g(v)\f$.
   *
   * Computes the fourth derivative of \f$g(v)\f$ (or \f$h^{(4)}(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Fourth derivative \f$g^{(4)}(v)\f$, or \f$h^{(4)}(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d3(), forward_d5()
   */
  LAPLACE_HD_FORCEINLINE double forward_d4(double v) const {
    return inverted ? detail_scalar::inv<4>(map, v) : detail_scalar::fwd<4>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fifth derivative of the forward map, \f$g^{(5)}(v) = \frac{\mathrm{d}^5}{\mathrm{d}v^5}g(v)\f$.
   *
   * Computes the fifth derivative of \f$g(v)\f$ (or \f$h^{(5)}(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Fifth derivative \f$g^{(5)}(v)\f$, or \f$h^{(5)}(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d4(), forward_d6()
   */
  LAPLACE_HD_FORCEINLINE double forward_d5(double v) const {
    return inverted ? detail_scalar::inv<5>(map, v) : detail_scalar::fwd<5>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Sixth derivative of the forward map, \f$g^{(6)}(v) = \frac{\mathrm{d}^6}{\mathrm{d}v^6}g(v)\f$.
   *
   * Computes the sixth derivative of \f$g(v)\f$ (or \f$h^{(6)}(v)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Sixth derivative \f$g^{(6)}(v)\f$, or \f$h^{(6)}(v)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward(), forward_d5()
   */
  LAPLACE_HD_FORCEINLINE double forward_d6(double v) const {
    return inverted ? detail_scalar::inv<6>(map, v) : detail_scalar::fwd<6>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the inverse map \f$h(u) = g^{-1}(u)\f$.
   *
   * Computes \f$v = h(u)\f$ mapping image \f$U\f$ to domain \f$V\f$.
   * When inverted, evaluates \f$g(u)\f$ instead.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return The transformed value \f$h(u)\f$, or \f$g(u)\f$ if inverted.
   *
   * \par Domain and special values
   * - Standard domain is \f$[u_{\min}, u_{\max}]\f$ given by inverse_domain_lower() and inverse_domain_upper().
   * - On boundaries, returns the mathematical limits (for example, \f$\Phi(-\infty) = 0\f$, \f$\Phi(+\infty) = 1\f$).
   * - If \f$u\f$ is strictly outside the closed domain or is NaN, returns quiet NaN.
   *
   * \par Measured accuracy
   * Over orders 0 to 6, maximum error in ulpC on gcc / CUDA: log 0.574 / 0.574; logit 10.5 / 10.1; probit 2.28 / 2.28; cauchit 8.63 / 8.63; cloglog 9.98 / 7.61; loglog 11 / 8.49; inverse 3.25 / 3.25; inverse_sq 7.29 / 7.29; sqrt 0.246 / 0.246; atanh 2.67 / 1.67; asinh 0.494 / 0.622. Values (order 0) are within 4.1 ulp on gcc except where ill-conditioned.
   *
   * \see bijector_scalar, scalar_bijector, forward(), inverse_d1(), inverse_log_jacobian(), inverse_domain_lower(), inverse_domain_upper()
   */
  LAPLACE_HD_FORCEINLINE double inverse(double u) const {
    return inverted ? detail_scalar::fwd<0>(map, u) : detail_scalar::inv<0>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief First derivative of the inverse map, \f$h'(u) = \frac{\mathrm{d}}{\mathrm{d}u}h(u)\f$.
   *
   * Computes the first derivative of \f$h(u)\f$ (or \f$g'(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return First derivative \f$h'(u)\f$, or \f$g'(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d2()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d1(double u) const {
    return inverted ? detail_scalar::fwd<1>(map, u) : detail_scalar::inv<1>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Second derivative of the inverse map, \f$h''(u) = \frac{\mathrm{d}^2}{\mathrm{d}u^2}h(u)\f$.
   *
   * Computes the second derivative of \f$h(u)\f$ (or \f$g''(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Second derivative \f$h''(u)\f$, or \f$g''(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d1(), inverse_d3()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d2(double u) const {
    return inverted ? detail_scalar::fwd<2>(map, u) : detail_scalar::inv<2>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Third derivative of the inverse map, \f$h^{(3)}(u) = \frac{\mathrm{d}^3}{\mathrm{d}u^3}h(u)\f$.
   *
   * Computes the third derivative of \f$h(u)\f$ (or \f$g^{(3)}(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Third derivative \f$h^{(3)}(u)\f$, or \f$g^{(3)}(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d2(), inverse_d4()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d3(double u) const {
    return inverted ? detail_scalar::fwd<3>(map, u) : detail_scalar::inv<3>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fourth derivative of the inverse map, \f$h^{(4)}(u) = \frac{\mathrm{d}^4}{\mathrm{d}u^4}h(u)\f$.
   *
   * Computes the fourth derivative of \f$h(u)\f$ (or \f$g^{(4)}(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Fourth derivative \f$h^{(4)}(u)\f$, or \f$g^{(4)}(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d3(), inverse_d5()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d4(double u) const {
    return inverted ? detail_scalar::fwd<4>(map, u) : detail_scalar::inv<4>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fifth derivative of the inverse map, \f$h^{(5)}(u) = \frac{\mathrm{d}^5}{\mathrm{d}u^5}h(u)\f$.
   *
   * Computes the fifth derivative of \f$h(u)\f$ (or \f$g^{(5)}(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Fifth derivative \f$h^{(5)}(u)\f$, or \f$g^{(5)}(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d4(), inverse_d6()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d5(double u) const {
    return inverted ? detail_scalar::fwd<5>(map, u) : detail_scalar::inv<5>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Sixth derivative of the inverse map, \f$h^{(6)}(u) = \frac{\mathrm{d}^6}{\mathrm{d}u^6}h(u)\f$.
   *
   * Computes the sixth derivative of \f$h(u)\f$ (or \f$g^{(6)}(u)\f$ if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Sixth derivative \f$h^{(6)}(u)\f$, or \f$g^{(6)}(u)\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), inverse_d5()
   */
  LAPLACE_HD_FORCEINLINE double inverse_d6(double u) const {
    return inverted ? detail_scalar::fwd<6>(map, u) : detail_scalar::inv<6>(map, u);
  }

  // log|g'(v)| and log|h'(u)|, with their derivatives
  /**
   * \ingroup bijector_scalar
   * \brief Forward log-Jacobian, \f$\log|g'(v)|\f$.
   *
   * Computes the forward log-Jacobian \f$\log|\frac{\mathrm{d}}{\mathrm{d}v}g(v)|\f$.
   * When inverted, evaluates \f$\log|h'(v)|\f$ instead.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return The forward log-Jacobian \f$\log|g'(v)|\f$, or \f$\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Over orders 0 to 6, maximum error in ulpC on gcc / CUDA: log 3.84 / 3.84; logit 5.17 / 5.17; probit 6.57 / 6.57; cauchit 5.23 / 5.23; cloglog 4.16 / 3.34; loglog 4.84 / 3.38; inverse 3.92 / 3.92; inverse_sq 3.86 / 3.86; sqrt 2.43 / 2.43; atanh 3.53 / 5.05; asinh 5.35 / 7.85. Identity is 0.
   *
   * \see bijector_scalar, scalar_bijector, forward(), inverse_log_jacobian(), forward_log_jacobian_d1()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian(double v) const {
    return inverted ? detail_scalar::inv_lj<0>(map, v) : detail_scalar::fwd_lj<0>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief First derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|g'(v)|\f$.
   *
   * Computes the first derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return First derivative \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d2()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d1(double v) const {
    return inverted ? detail_scalar::inv_lj<1>(map, v) : detail_scalar::fwd_lj<1>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Second derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^2}{\mathrm{d}v^2}\log|g'(v)|\f$.
   *
   * Computes the second derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Second derivative \f$\frac{\mathrm{d}^2}{\mathrm{d}v^2}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}^2}{\mathrm{d}v^2}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d1(), forward_log_jacobian_d3()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d2(double v) const {
    return inverted ? detail_scalar::inv_lj<2>(map, v) : detail_scalar::fwd_lj<2>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Third derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^3}{\mathrm{d}v^3}\log|g'(v)|\f$.
   *
   * Computes the third derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Third derivative \f$\frac{\mathrm{d}^3}{\mathrm{d}v^3}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}^3}{\mathrm{d}v^3}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d2(), forward_log_jacobian_d4()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d3(double v) const {
    return inverted ? detail_scalar::inv_lj<3>(map, v) : detail_scalar::fwd_lj<3>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fourth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^4}{\mathrm{d}v^4}\log|g'(v)|\f$.
   *
   * Computes the fourth derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Fourth derivative \f$\frac{\mathrm{d}^4}{\mathrm{d}v^4}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}^4}{\mathrm{d}v^4}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d3(), forward_log_jacobian_d5()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d4(double v) const {
    return inverted ? detail_scalar::inv_lj<4>(map, v) : detail_scalar::fwd_lj<4>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fifth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^5}{\mathrm{d}v^5}\log|g'(v)|\f$.
   *
   * Computes the fifth derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Fifth derivative \f$\frac{\mathrm{d}^5}{\mathrm{d}v^5}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}^5}{\mathrm{d}v^5}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d4(), forward_log_jacobian_d6()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d5(double v) const {
    return inverted ? detail_scalar::inv_lj<5>(map, v) : detail_scalar::fwd_lj<5>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Sixth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^6}{\mathrm{d}v^6}\log|g'(v)|\f$.
   *
   * Computes the sixth derivative of the forward log-Jacobian (or of the inverse log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param v Input argument \f$v \in V\f$.
   * \return Sixth derivative \f$\frac{\mathrm{d}^6}{\mathrm{d}v^6}\log|g'(v)|\f$, or \f$\frac{\mathrm{d}^6}{\mathrm{d}v^6}\log|h'(v)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$v\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, forward_log_jacobian(), forward_log_jacobian_d5()
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d6(double v) const {
    return inverted ? detail_scalar::inv_lj<6>(map, v) : detail_scalar::fwd_lj<6>(map, v);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Inverse log-Jacobian, \f$\log|h'(u)|\f$.
   *
   * Computes the inverse log-Jacobian \f$\log|\frac{\mathrm{d}}{\mathrm{d}u}h(u)|\f$.
   * When inverted, evaluates \f$\log|g'(u)|\f$ instead.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return The inverse log-Jacobian \f$\log|h'(u)|\f$, or \f$\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Over orders 0 to 6, maximum error in ulpC on gcc / CUDA: log 0 / 0; logit 10.5 / 10.1; probit 0.832 / 0.832; cauchit 7.78 / 7.78; cloglog 0.969 / 0.969; loglog 0.883 / 0.883; inverse 3.25 / 3.25; inverse_sq 2.79 / 2.79; sqrt 2.95 / 2.95; atanh 2.67 / 1.67; asinh 1.9 / 2.27. Identity is 0.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), forward_log_jacobian(), inverse_log_jacobian_d1()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian(double u) const {
    return inverted ? detail_scalar::fwd_lj<0>(map, u) : detail_scalar::inv_lj<0>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief First derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|h'(u)|\f$.
   *
   * Computes the first derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return First derivative \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d2()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d1(double u) const {
    return inverted ? detail_scalar::fwd_lj<1>(map, u) : detail_scalar::inv_lj<1>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Second derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^2}{\mathrm{d}u^2}\log|h'(u)|\f$.
   *
   * Computes the second derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Second derivative \f$\frac{\mathrm{d}^2}{\mathrm{d}u^2}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}^2}{\mathrm{d}u^2}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d1(), inverse_log_jacobian_d3()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d2(double u) const {
    return inverted ? detail_scalar::fwd_lj<2>(map, u) : detail_scalar::inv_lj<2>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Third derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^3}{\mathrm{d}u^3}\log|h'(u)|\f$.
   *
   * Computes the third derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Third derivative \f$\frac{\mathrm{d}^3}{\mathrm{d}u^3}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}^3}{\mathrm{d}u^3}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d2(), inverse_log_jacobian_d4()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d3(double u) const {
    return inverted ? detail_scalar::fwd_lj<3>(map, u) : detail_scalar::inv_lj<3>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fourth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^4}{\mathrm{d}u^4}\log|h'(u)|\f$.
   *
   * Computes the fourth derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Fourth derivative \f$\frac{\mathrm{d}^4}{\mathrm{d}u^4}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}^4}{\mathrm{d}u^4}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d3(), inverse_log_jacobian_d5()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d4(double u) const {
    return inverted ? detail_scalar::fwd_lj<4>(map, u) : detail_scalar::inv_lj<4>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Fifth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^5}{\mathrm{d}u^5}\log|h'(u)|\f$.
   *
   * Computes the fifth derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Fifth derivative \f$\frac{\mathrm{d}^5}{\mathrm{d}u^5}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}^5}{\mathrm{d}u^5}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d4(), inverse_log_jacobian_d6()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d5(double u) const {
    return inverted ? detail_scalar::fwd_lj<5>(map, u) : detail_scalar::inv_lj<5>(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Sixth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^6}{\mathrm{d}u^6}\log|h'(u)|\f$.
   *
   * Computes the sixth derivative of the inverse log-Jacobian (or of the forward log-Jacobian if inverted).
   * Closed-form formulas for each map are given on \ref bijector_scalar.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return Sixth derivative \f$\frac{\mathrm{d}^6}{\mathrm{d}u^6}\log|h'(u)|\f$, or \f$\frac{\mathrm{d}^6}{\mathrm{d}u^6}\log|g'(u)|\f$ if inverted.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain or is NaN.
   *
   * \par Measured accuracy
   * Within 11 ulpC on gcc and CUDA across all orders; see \ref bijector_scalar.
   *
   * \see bijector_scalar, scalar_bijector, inverse_log_jacobian(), inverse_log_jacobian_d5()
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d6(double u) const {
    return inverted ? detail_scalar::fwd_lj<6>(map, u) : detail_scalar::inv_lj<6>(map, u);
  }

  // the domain of forward() and the domain of inverse(), as closed bounds
  /**
   * \ingroup bijector_scalar
   * \brief Lower bound of the forward domain interval \f$V\f$.
   *
   * Returns the infimum of the domain of forward() (the lower bound of \f$V\f$, or of \f$U\f$ if inverted).
   *
   * \return Lower bound as a \c double (may be \f$-\infty\f$).
   *
   * \see bijector_scalar, scalar_bijector, forward_domain_upper(), inverse_domain_lower()
   */
  LAPLACE_HD_FORCEINLINE double forward_domain_lower() const {
    return inverted ? detail_scalar::img_lo(map) : detail_scalar::dom_lo(map);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Upper bound of the forward domain interval \f$V\f$.
   *
   * Returns the supremum of the domain of forward() (the upper bound of \f$V\f$, or of \f$U\f$ if inverted).
   *
   * \return Upper bound as a \c double (may be \f$+\infty\f$).
   *
   * \see bijector_scalar, scalar_bijector, forward_domain_lower(), inverse_domain_upper()
   */
  LAPLACE_HD_FORCEINLINE double forward_domain_upper() const {
    return inverted ? detail_scalar::img_hi(map) : detail_scalar::dom_hi(map);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Lower bound of the inverse domain interval \f$U\f$.
   *
   * Returns the infimum of the domain of inverse() (the lower bound of \f$U\f$, or of \f$V\f$ if inverted).
   *
   * \return Lower bound as a \c double (may be \f$-\infty\f$).
   *
   * \see bijector_scalar, scalar_bijector, inverse_domain_upper(), forward_domain_lower()
   */
  LAPLACE_HD_FORCEINLINE double inverse_domain_lower() const {
    return inverted ? detail_scalar::dom_lo(map) : detail_scalar::img_lo(map);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Upper bound of the inverse domain interval \f$U\f$.
   *
   * Returns the supremum of the domain of inverse() (the upper bound of \f$U\f$, or of \f$V\f$ if inverted).
   *
   * \return Upper bound as a \c double (may be \f$+\infty\f$).
   *
   * \see bijector_scalar, scalar_bijector, inverse_domain_lower(), forward_domain_upper()
   */
  LAPLACE_HD_FORCEINLINE double inverse_domain_upper() const {
    return inverted ? detail_scalar::dom_hi(map) : detail_scalar::img_hi(map);
  }
  // true when the map is decreasing (inverse, inverse_sq); inversion keeps it
  /**
   * \ingroup bijector_scalar
   * \brief Whether the bijector map is strictly decreasing.
   *
   * Returns \c true for \c inverse and \c inverse_sq, and \c false for all other maps.
   * Inversion does not change monotonicity.
   *
   * \return \c true if strictly decreasing, \c false if strictly increasing.
   *
   * \see bijector_scalar, scalar_bijector
   */
  LAPLACE_HD_FORCEINLINE bool is_decreasing() const {
    return map == scalar_bijector_map::inverse || map == scalar_bijector_map::inverse_sq;
  }

  // The distance of inverse(u) to the upper and to the lower bound of its
  // range (forward_domain_upper(), forward_domain_lower()), computed directly
  // where the difference would cancel.
  /**
   * \ingroup bijector_scalar
   * \brief Distance of \ref laplace::bijector::scalar_bijector::inverse "inverse()" to the upper bound of its range, \ref forward_domain_upper().
   *
   * Computes the distance of \f$h(u)\f$ to the upper bound of its range, evaluated directly without
   * forming \f$h(u)\f$ where forming the difference would cancel:
   * - \c logit : \f$\operatorname{expit}(-u)\f$
   * - \c probit : \f$\Phi(-u)\f$
   * - \c cauchit : \f$h(-u)\f$
   * - \c cloglog : \f$\exp(-e^u)\f$
   * - \c loglog : \f$1 - \exp(-e^{-u})\f$
   * - \c atanh : \f$2e/(1 + e)\f$ with \f$e = \exp(-2u)\f$ for \f$u > 0\f$ (\f$1 - \tanh u\f$ for \f$u \le 0\f$)
   *
   * For other maps, evaluates the generic form \ref forward_domain_upper() \f$- h(u)\f$ (evaluating to \f$+\infty\f$
   * when the bound is infinite). When inverted, always evaluates the generic form.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return The distance to the upper bound, or quiet NaN if outside domain or NaN.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain of \ref laplace::bijector::scalar_bijector::inverse "inverse()" or is NaN.
   *
   * \par Measured accuracy
   * With ulpC scaled by \f$h'(u)\f$, maximum error over the six maps and both distances is 2.28 ulpC on gcc
   * and on CUDA (measured for probit lower distance); in plain ulps up to 130 for cloglog (upper distance)
   * where \f$\exp(-e^u)\f$ carries the rounding of \f$e^u\f$.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), forward_domain_upper(), inverse_distance_lower()
   */
  LAPLACE_HD_FORCEINLINE double inverse_distance_upper(double u) const {
    if (inverted) return forward_domain_upper() - inverse(u);
    return detail_scalar::inv_to_upper(map, u);
  }
  /**
   * \ingroup bijector_scalar
   * \brief Distance of \ref laplace::bijector::scalar_bijector::inverse "inverse()" to the lower bound of its range, \ref forward_domain_lower().
   *
   * Computes the distance of \f$h(u)\f$ to the lower bound of its range:
   * - \c atanh : \f$2e/(1 + e)\f$ with \f$e = \exp(2u)\f$ for \f$u < 0\f$ (\f$1 + \tanh u\f$ for \f$u \ge 0\f$)
   * - For \c logit, \c probit, \c cauchit, \c cloglog, \c loglog : evaluates \f$h(u)\f$ directly, since the lower bound is 0.
   *
   * For other maps, evaluates the generic form \ref laplace::bijector::scalar_bijector::inverse "inverse()" \f$- \f$ \ref forward_domain_lower() (evaluating to \f$+\infty\f$
   * when the bound is infinite). When inverted, always evaluates the generic form.
   *
   * \param u Input argument \f$u \in U\f$.
   * \return The distance to the lower bound, or quiet NaN if outside domain or NaN.
   *
   * \par Domain and special values
   * Returns quiet NaN if \f$u\f$ is strictly outside the closed domain of \ref laplace::bijector::scalar_bijector::inverse "inverse()" or is NaN.
   *
   * \par Measured accuracy
   * With ulpC scaled by \f$h'(u)\f$, maximum error over the six maps and both distances is 2.28 ulpC on gcc
   * and on CUDA (measured for probit lower distance); in plain ulps up to 173–174 for loglog (lower distance)
   * where \f$\exp(-e^{-u})\f$ carries the rounding of \f$e^{-u}\f$.
   *
   * \see bijector_scalar, scalar_bijector, inverse(), forward_domain_lower(), inverse_distance_upper()
   */
  LAPLACE_HD_FORCEINLINE double inverse_distance_lower(double u) const {
    if (inverted) return inverse(u) - forward_domain_lower();
    return detail_scalar::inv_to_lower(map, u);
  }
};

}  // namespace laplace::bijector
