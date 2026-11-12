// Scalar bijectors (laplace::bijector), the maps without parameters.
//
// A bijector is a bijection g: V -> U between open intervals, with inverse
// h = g^-1. Forward goes from the constrained side to the unconstrained one,
// in the direction that statistics names the map (decisions of 2026-11-12,
// docs/numerics.md): log maps (0, inf) onto R, logit maps (0, 1) onto R. A
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
// (decided 2026-11-12): the exact function is computed, and
// the distances to the bounds are given where 1 - h(u) would cancel.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/bijector/detail/scalar_affine.hpp>
#include <laplace/bijector/detail/scalar_shifted.hpp>
#include <laplace/bijector/detail/scalar_bounded.hpp>
#include <laplace/bijector/detail/scalar_power.hpp>
#include <laplace/bijector/detail/scalar_softplus.hpp>
#include <laplace/bijector/detail/scalar_box_cox.hpp>
#include <laplace/bijector/detail/scalar_yeo_johnson.hpp>

namespace laplace::bijector {

// The maps, named by their forward direction.
/**
 * \ingroup bijector_scalar
 * \brief Enumeration of scalar bijector maps with and without parameters.
 *
 * Defines twenty bijections \f$g: V \to U\f$ between open intervals:
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
 * - \c affine : \f$g(v) = (v - \mu)/\sigma\f$, mapping \f$(-\infty, \infty) \to (-\infty, \infty)\f$, parameters \f$\mu\f$ (\c location) and \f$\sigma \neq 0\f$ (\c scale).
 * - \c power : \f$g(v) = v^\lambda\f$, mapping \f$(0, \infty) \to (0, \infty)\f$, parameter \f$\lambda \neq 0\f$ (\c exponent).
 * - \c softplus : \f$g(v) = \frac{1}{a}\log(e^{a v} - 1)\f$, mapping \f$(0, \infty) \to (-\infty, \infty)\f$, parameter \f$a > 0\f$ (\c sharpness).
 * - \c bounded : \f$g(v) = \log(v - a) - \log(b - v)\f$, mapping \f$(a, b) \to (-\infty, \infty)\f$, parameters \f$a\f$ (\c lower) and \f$b\f$ (\c upper) with \f$a < b\f$.
 * - \c lower_bounded : \f$g(v) = \log(v - a)\f$, mapping \f$(a, \infty) \to (-\infty, \infty)\f$, parameter \f$a\f$ (\c lower).
 * - \c upper_bounded : \f$g(v) = -\log(b - v)\f$, mapping \f$(-\infty, b) \to (-\infty, \infty)\f$, parameter \f$b\f$ (\c upper).
 * - \c box_cox : \f$g(v) = (v^\lambda - 1)/\lambda\f$ (\f$\log v\f$ at \f$\lambda = 0\f$), mapping \f$(0, \infty)\f$ to \f$(-1/\lambda, \infty)\f$ for \f$\lambda > 0\f$, \f$(-\infty, -1/\lambda)\f$ for \f$\lambda < 0\f$, and \f$(-\infty, \infty)\f$ for \f$\lambda = 0\f$, parameter \f$\lambda\f$ (\c exponent).
 * - \c yeo_johnson : \f$g(v) = ((1 + v)^\lambda - 1)/\lambda\f$ for \f$v \ge 0\f$ and \f$-((1 - v)^{2 - \lambda} - 1)/(2 - \lambda)\f$ for \f$v < 0\f$, mapping \f$(-\infty, \infty)\f$ to \f$(-\infty, \infty)\f$ for \f$0 \le \lambda \le 2\f$, \f$(-\infty, -1/\lambda)\f$ for \f$\lambda < 0\f$, and \f$(1/(2 - \lambda), \infty)\f$ for \f$\lambda > 2\f$, parameter \f$\lambda\f$ (\c exponent).
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
  asinh,       // R -> R
  // maps with parameters (fields of scalar_bijector in brackets)
  affine,         // R -> R, (v - location)/scale, scale != 0
  power,          // (0, inf) -> (0, inf), v^exponent, exponent != 0; no derivatives in exponent
  softplus,       // (0, inf) -> R, log(e^(sharpness v) - 1)/sharpness, sharpness > 0
  bounded,        // (lower, upper) -> R, log(v - lower) - log(upper - v)
  lower_bounded,  // (lower, inf) -> R, log(v - lower)
  upper_bounded,  // (-inf, upper) -> R, -log(upper - v)
  box_cox,        // (0, inf) -> image depending on exponent, (v^exponent - 1)/exponent
  yeo_johnson     // R -> image depending on exponent
};

namespace detail_scalar {

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
// p1, p2: the parameters of the map, in the order of its partial derivatives
// (scalar_bijector::parameter_1/2): affine (location, scale); power, box_cox,
// yeo_johnson (exponent); softplus (sharpness); bounded (lower, upper);
// lower_bounded (lower); upper_bounded (upper); unused ones are ignored.

LAPLACE_HD_FORCEINLINE double dom_lo(scalar_bijector_map k, double p1, double p2) {
  switch (k) {
    case scalar_bijector_map::identity: return -inf();
    case scalar_bijector_map::asinh: return -inf();
    case scalar_bijector_map::atanh: return -1.0;
    case scalar_bijector_map::affine: return affine_dom_lo(p1, p2);
    case scalar_bijector_map::power: return power_dom_lo(p1, p2);
    case scalar_bijector_map::softplus: return softplus_dom_lo(p1, p2);
    case scalar_bijector_map::bounded: return bounded_dom_lo(p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_dom_lo(p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_dom_lo(p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_dom_lo(p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_dom_lo(p1, p2);
    default: return 0.0;
  }
}
LAPLACE_HD_FORCEINLINE double dom_hi(scalar_bijector_map k, double p1, double p2) {
  switch (k) {
    case scalar_bijector_map::logit: return 1.0;
    case scalar_bijector_map::probit: return 1.0;
    case scalar_bijector_map::cauchit: return 1.0;
    case scalar_bijector_map::cloglog: return 1.0;
    case scalar_bijector_map::loglog: return 1.0;
    case scalar_bijector_map::atanh: return 1.0;
    case scalar_bijector_map::affine: return affine_dom_hi(p1, p2);
    case scalar_bijector_map::power: return power_dom_hi(p1, p2);
    case scalar_bijector_map::softplus: return softplus_dom_hi(p1, p2);
    case scalar_bijector_map::bounded: return bounded_dom_hi(p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_dom_hi(p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_dom_hi(p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_dom_hi(p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_dom_hi(p1, p2);
    default: return inf();
  }
}
LAPLACE_HD_FORCEINLINE double img_lo(scalar_bijector_map k, double p1, double p2) {
  switch (k) {
    case scalar_bijector_map::inverse: return 0.0;
    case scalar_bijector_map::inverse_sq: return 0.0;
    case scalar_bijector_map::sqrt: return 0.0;
    case scalar_bijector_map::affine: return affine_img_lo(p1, p2);
    case scalar_bijector_map::power: return power_img_lo(p1, p2);
    case scalar_bijector_map::softplus: return softplus_img_lo(p1, p2);
    case scalar_bijector_map::bounded: return bounded_img_lo(p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_img_lo(p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_img_lo(p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_img_lo(p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_img_lo(p1, p2);
    default: return -inf();
  }
}
LAPLACE_HD_FORCEINLINE double img_hi(scalar_bijector_map k, double p1, double p2) {
  switch (k) {
    case scalar_bijector_map::affine: return affine_img_hi(p1, p2);
    case scalar_bijector_map::power: return power_img_hi(p1, p2);
    case scalar_bijector_map::softplus: return softplus_img_hi(p1, p2);
    case scalar_bijector_map::bounded: return bounded_img_hi(p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_img_hi(p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_img_hi(p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_img_hi(p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_img_hi(p1, p2);
    default: return inf();
  }
}

LAPLACE_HD_FORCEINLINE bool inside(double x, double lo, double hi) { return x >= lo && x <= hi; }

template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD double fwd(scalar_bijector_map k, double v, double p1, double p2) {
  if (!inside(v, dom_lo(k, p1, p2), dom_hi(k, p1, p2))) return nan();
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
    case scalar_bijector_map::affine: return affine_fwd<K>(v, p1, p2);
    case scalar_bijector_map::power: return power_fwd<K>(v, p1, p2);
    case scalar_bijector_map::softplus: return softplus_fwd<K>(v, p1, p2);
    case scalar_bijector_map::bounded: return bounded_fwd<K>(v, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_fwd<K>(v, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_fwd<K>(v, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_fwd<K>(v, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_fwd<K>(v, p1, p2);
  }
  return nan();
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD double inv(scalar_bijector_map k, double u, double p1, double p2) {
  if (!inside(u, img_lo(k, p1, p2), img_hi(k, p1, p2))) return nan();
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
    case scalar_bijector_map::affine: return affine_inv<K>(u, p1, p2);
    case scalar_bijector_map::power: return power_inv<K>(u, p1, p2);
    case scalar_bijector_map::softplus: return softplus_inv<K>(u, p1, p2);
    case scalar_bijector_map::bounded: return bounded_inv<K>(u, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_inv<K>(u, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_inv<K>(u, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_inv<K>(u, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_inv<K>(u, p1, p2);
  }
  return nan();
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD double fwd_lj(scalar_bijector_map k, double v, double p1, double p2) {
  if (!inside(v, dom_lo(k, p1, p2), dom_hi(k, p1, p2))) return nan();
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
    case scalar_bijector_map::affine: return affine_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::power: return power_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::softplus: return softplus_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::bounded: return bounded_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_fwd_lj<K>(v, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_fwd_lj<K>(v, p1, p2);
  }
  return nan();
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD double inv_lj(scalar_bijector_map k, double u, double p1, double p2) {
  if (!inside(u, img_lo(k, p1, p2), img_hi(k, p1, p2))) return nan();
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
    case scalar_bijector_map::affine: return affine_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::power: return power_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::softplus: return softplus_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::bounded: return bounded_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_inv_lj<K>(u, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_inv_lj<K>(u, p1, p2);
  }
  return nan();
}

// Partial derivatives of total order K in (argument, p1, p2), written to out[]
// in the order of bijector_partials<K>. Maps without parameters, and power
// (no derivatives in its exponent), give the pure derivative in c[0]; the
// others are 0 for maps without parameters and NaN for power.
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD void fwd_partials(scalar_bijector_map k, double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!inside(x, dom_lo(k, p1, p2), dom_hi(k, p1, p2))) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  switch (k) {
    case scalar_bijector_map::affine: affine_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::softplus: softplus_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::bounded: bounded_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::lower_bounded: lower_bounded_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::upper_bounded: upper_bounded_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::box_cox: box_cox_fwd_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::yeo_johnson: yeo_johnson_fwd_partials<K>(x, p1, p2, out); return;
    default: {
      const double fill = k == scalar_bijector_map::power ? nan() : 0.0;
      out[0] = fwd<K>(k, x, p1, p2);
      for (int i = 1; i < n; ++i) out[i] = fill;
    }
  }
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD void inv_partials(scalar_bijector_map k, double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!inside(x, img_lo(k, p1, p2), img_hi(k, p1, p2))) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  switch (k) {
    case scalar_bijector_map::affine: affine_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::softplus: softplus_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::bounded: bounded_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::lower_bounded: lower_bounded_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::upper_bounded: upper_bounded_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::box_cox: box_cox_inv_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::yeo_johnson: yeo_johnson_inv_partials<K>(x, p1, p2, out); return;
    default: {
      const double fill = k == scalar_bijector_map::power ? nan() : 0.0;
      out[0] = inv<K>(k, x, p1, p2);
      for (int i = 1; i < n; ++i) out[i] = fill;
    }
  }
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD void fwd_lj_partials(scalar_bijector_map k, double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!inside(x, dom_lo(k, p1, p2), dom_hi(k, p1, p2))) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  switch (k) {
    case scalar_bijector_map::affine: affine_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::softplus: softplus_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::bounded: bounded_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::lower_bounded: lower_bounded_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::upper_bounded: upper_bounded_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::box_cox: box_cox_fwd_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::yeo_johnson: yeo_johnson_fwd_lj_partials<K>(x, p1, p2, out); return;
    default: {
      const double fill = k == scalar_bijector_map::power ? nan() : 0.0;
      out[0] = fwd_lj<K>(k, x, p1, p2);
      for (int i = 1; i < n; ++i) out[i] = fill;
    }
  }
}
template <int K>
LAPLACE_BIJ_NOINLINE LAPLACE_HD void inv_lj_partials(scalar_bijector_map k, double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  if (!inside(x, img_lo(k, p1, p2), img_hi(k, p1, p2))) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  switch (k) {
    case scalar_bijector_map::affine: affine_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::softplus: softplus_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::bounded: bounded_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::lower_bounded: lower_bounded_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::upper_bounded: upper_bounded_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::box_cox: box_cox_inv_lj_partials<K>(x, p1, p2, out); return;
    case scalar_bijector_map::yeo_johnson: yeo_johnson_inv_lj_partials<K>(x, p1, p2, out); return;
    default: {
      const double fill = k == scalar_bijector_map::power ? nan() : 0.0;
      out[0] = inv_lj<K>(k, x, p1, p2);
      for (int i = 1; i < n; ++i) out[i] = fill;
    }
  }
}

// The distances of h(u) to the bounds of its range (the domain of g), computed
// without forming h, where h saturates at a bound other than 0 or infinity:
// 1 - h(u) of logit is expit(-u), exact while 1 - expit(u) has rounded to 0
// from u = 37.4. Elsewhere the bound minus h(u) (exact for a bound 0 or infinite).
LAPLACE_BIJ_NOINLINE LAPLACE_HD double inv_to_upper(scalar_bijector_map k, double u, double p1, double p2) {
  if (!inside(u, img_lo(k, p1, p2), img_hi(k, p1, p2))) return nan();
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
    case scalar_bijector_map::affine: return affine_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::power: return power_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::softplus: return softplus_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::bounded: return bounded_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_inv_to_upper(u, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_inv_to_upper(u, p1, p2);
    default: return dom_hi(k, p1, p2) - inv<0>(k, u, p1, p2);
  }
}
LAPLACE_BIJ_NOINLINE LAPLACE_HD double inv_to_lower(scalar_bijector_map k, double u, double p1, double p2) {
  if (k == scalar_bijector_map::atanh) return inv_to_upper(k, -u, p1, p2);  // 1 + tanh u
  if (!inside(u, img_lo(k, p1, p2), img_hi(k, p1, p2))) return nan();
  switch (k) {
    case scalar_bijector_map::affine: return affine_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::power: return power_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::softplus: return softplus_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::bounded: return bounded_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::lower_bounded: return lower_bounded_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::upper_bounded: return upper_bounded_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::box_cox: return box_cox_inv_to_lower(u, p1, p2);
    case scalar_bijector_map::yeo_johnson: return yeo_johnson_inv_to_lower(u, p1, p2);
    default: return inv<0>(k, u, p1, p2) - dom_lo(k, p1, p2);
  }
}

}  // namespace detail_scalar

// ---------------------------------------------------------------- the bijector

// The partial derivatives of total order K of a quantity of a scalar bijector in
// (argument, parameter 1, parameter 2): one component per multiset of K
// variables, n0 + n1 + n2 = K, ordered by n2 and then by n1, so that the pure
// derivative in the argument is c[0] and a map with one parameter fills the first
// K + 1 components. index(n1, n2) gives the position.
/**
 * \ingroup bijector_scalar
 * \brief Partial derivatives of total order \f$K\f$ of a bijector quantity in argument and parameters.
 *
 * Holds the derivatives \f$\frac{\partial^K}{\partial x^{n_0} \partial p_1^{n_1} \partial p_2^{n_2}} f\f$
 * with \f$n_0 + n_1 + n_2 = K\f$ in an array of size \f$(K + 1)(K + 2) / 2\f$. Components are ordered
 * by \f$n_2\f$ and then by \f$n_1\f$, accessible via index() as
 * \f$\mathrm{index}(n_1, n_2) = n_2 (K + 1) - n_2 (n_2 - 1) / 2 + n_1\f$.
 *
 * Component \c c[0] (\f$n_1 = n_2 = 0\f$) is the pure derivative in the argument. A map with one
 * parameter fills the first \f$K + 1\f$ components (\f$n_2 = 0\f$) and leaves the rest zero.
 * Maps without parameters have component \c c[0] equal to the pure derivative and the remaining
 * components zero. For \c power, components involving the exponent are NaN.
 *
 * \tparam K Total derivative order, \f$1 \le K \le 6\f$.
 * \see scalar_bijector, bijector_scalar
 */
template <int K>
struct bijector_partials {
  static constexpr int size = (K + 1) * (K + 2) / 2;
  double c[size];
  /**
   * \brief Computes the component index in array \c c for derivative orders \f$(n_1, n_2)\f$.
   * \param n1 Order of differentiation in parameter 1.
   * \param n2 Order of differentiation in parameter 2.
   * \return Index \f$n_2 (K + 1) - n_2 (n_2 - 1) / 2 + n_1\f$.
   */
  LAPLACE_HD static constexpr int index(int n1, int n2) { return n2 * (K + 1) - n2 * (n2 - 1) / 2 + n1; }
};

// The points at which a map is not smooth: count of them (0 or 1 for a scalar
// map, up to 4 for a chain), their positions, and the order up to which the map
// stays continuous there (2 for Yeo-Johnson: its third derivative jumps). At the
// point itself the derivatives are the right-hand ones.
/**
 * \ingroup bijector_scalar
 * \brief Information about points where a scalar bijector is not smooth.
 *
 * Declares the non-smooth points: \c count (0 or 1 for a single scalar map),
 * \c point (positions of the kinks, up to 4), and \c continuous_order (the order
 * up to which the map remains continuous). For Yeo-Johnson, \c count is 1 at
 * \f$v = 0\f$ (\f$u = 0\f$) with \c continuous_order equal to 2 (its third
 * derivative jumps across zero unless \f$\lambda = 1\f$). At the kink itself,
 * derivatives follow the right-hand convention.
 *
 * \see scalar_bijector::forward_kinks(), scalar_bijector::inverse_kinks(), bijector_scalar
 */
struct bijector_kinks {
  int count = 0;
  double point[4] = {0.0, 0.0, 0.0, 0.0};
  int continuous_order = 0;
};

// A scalar bijector: the map, the inversion flag (inverted: forward is h) and the
// parameters of the maps that have them; each map reads only its own. Every
// quantity is a member function, one per order; an inverted bijector swaps the
// two directions of every pair.
/**
 * \ingroup bijector_scalar
 * \brief Configuration and evaluation structure for scalar bijectors.
 *
 * Holds the map selection (\c map), an inversion flag (\c inverted), and parameters:
 * - \c location : location parameter \f$\mu\f$ for \c affine.
 * - \c scale : scale parameter \f$\sigma \neq 0\f$ for \c affine.
 * - \c exponent : power or shape parameter \f$\lambda\f$ for \c power (\f$\lambda \neq 0\f$),
 *   \c box_cox, and \c yeo_johnson.
 * - \c sharpness : sharpness parameter \f$a > 0\f$ for \c softplus.
 * - \c lower : lower bound \f$a\f$ for \c bounded and \c lower_bounded.
 * - \c upper : upper bound \f$b\f$ for \c bounded and \c upper_bounded.
 *
 * Each map reads only its own fields; other fields are ignored. Every mathematical quantity
 * is evaluated via member functions. An inverted bijector (\c inverted == true) swaps the forward
 * and inverse directions for every member function.
 *
 * \see scalar_bijector_map, bijector_partials, bijector_kinks, bijector_scalar
 */
struct scalar_bijector {
  scalar_bijector_map map = scalar_bijector_map::identity;
  bool inverted = false;
  double location = 0.0;   // affine
  double scale = 1.0;      // affine, != 0
  double exponent = 1.0;   // power (!= 0), box_cox, yeo_johnson
  double sharpness = 1.0;  // softplus, > 0
  double lower = 0.0;      // bounded, lower_bounded
  double upper = 1.0;      // bounded, upper_bounded

  // construction of the maps with parameters; power normalizes its special
  // exponents to the named maps (1 identity, 1/2 sqrt, -1 inverse, -2 inverse_sq,
  // 0 log). Invalid parameters give a bijector whose functions return NaN.
  /**
   * \ingroup bijector_scalar
   * \brief Constructs an affine scalar bijector \f$g(v) = (v - \mu) / \sigma\f$.
   * \param location_ Location parameter \f$\mu\f$.
   * \param scale_ Nonzero scale parameter \f$\sigma\f$.
   * \return Configured scalar_bijector for \c affine. If \f$\sigma = 0\f$ or parameters are non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector affine(double location_, double scale_) {
    scalar_bijector b{scalar_bijector_map::affine};
    b.location = location_;
    b.scale = scale_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a power scalar bijector \f$g(v) = v^\lambda\f$.
   *
   * Normalizes special exponents to the corresponding parameter-free maps:
   * \f$\lambda = 1\f$ to \c identity, \f$\lambda = 0.5\f$ to \c sqrt, \f$\lambda = -1\f$ to \c inverse,
   * \f$\lambda = -2\f$ to \c inverse_sq, and \f$\lambda = 0\f$ to \c log.
   * \param exponent_ Exponent \f$\lambda \neq 0\f$.
   * \return Configured scalar_bijector for \c power or the normalized map. If \f$\lambda\f$ is non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector power(double exponent_) {
    if (exponent_ == 1.0) return scalar_bijector{scalar_bijector_map::identity};
    if (exponent_ == 0.5) return scalar_bijector{scalar_bijector_map::sqrt};
    if (exponent_ == -1.0) return scalar_bijector{scalar_bijector_map::inverse};
    if (exponent_ == -2.0) return scalar_bijector{scalar_bijector_map::inverse_sq};
    if (exponent_ == 0.0) return scalar_bijector{scalar_bijector_map::log};
    scalar_bijector b{scalar_bijector_map::power};
    b.exponent = exponent_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a softplus scalar bijector \f$g(v) = \frac{1}{a}\log(e^{a v} - 1)\f$.
   * \param sharpness_ Sharpness parameter \f$a > 0\f$.
   * \return Configured scalar_bijector for \c softplus. If \f$a \le 0\f$ or non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector softplus(double sharpness_) {
    scalar_bijector b{scalar_bijector_map::softplus};
    b.sharpness = sharpness_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a Box-Cox scalar bijector \f$g(v) = (v^\lambda - 1)/\lambda\f$.
   * \param exponent_ Exponent parameter \f$\lambda\f$ (\f$\log v\f$ at \f$\lambda = 0\f$).
   * \return Configured scalar_bijector for \c box_cox. If \f$\lambda\f$ is non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector box_cox(double exponent_) {
    scalar_bijector b{scalar_bijector_map::box_cox};
    b.exponent = exponent_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a Yeo-Johnson scalar bijector.
   * \param exponent_ Exponent parameter \f$\lambda\f$.
   * \return Configured scalar_bijector for \c yeo_johnson. If \f$\lambda\f$ is non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector yeo_johnson(double exponent_) {
    scalar_bijector b{scalar_bijector_map::yeo_johnson};
    b.exponent = exponent_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a lower-bounded scalar bijector \f$g(v) = \log(v - a)\f$.
   * \param lower_ Finite lower bound \f$a\f$.
   * \return Configured scalar_bijector for \c lower_bounded. If \f$a\f$ is non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector lower_bounded(double lower_) {
    scalar_bijector b{scalar_bijector_map::lower_bounded};
    b.lower = lower_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs an upper-bounded scalar bijector \f$g(v) = -\log(b - v)\f$.
   * \param upper_ Finite upper bound \f$b\f$.
   * \return Configured scalar_bijector for \c upper_bounded. If \f$b\f$ is non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector upper_bounded(double upper_) {
    scalar_bijector b{scalar_bijector_map::upper_bounded};
    b.upper = upper_;
    return b;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Constructs a bounded scalar bijector \f$g(v) = \log(v - a) - \log(b - v)\f$.
   * \param lower_ Finite lower bound \f$a\f$.
   * \param upper_ Finite upper bound \f$b\f$ with \f$a < b\f$.
   * \return Configured scalar_bijector for \c bounded. If \f$a \ge b\f$ or bounds are non-finite, evaluations return NaN.
   * \see scalar_bijector, bijector_scalar
   */
  LAPLACE_HD static scalar_bijector bounded(double lower_, double upper_) {
    scalar_bijector b{scalar_bijector_map::bounded};
    b.lower = lower_;
    b.upper = upper_;
    return b;
  }

  // the parameters in the order of the partial derivatives (see the dispatch)
  /**
   * \ingroup bijector_scalar
   * \brief Returns the first parameter of the map in the order of partial derivatives.
   *
   * Returns \c location for \c affine; \c exponent for \c power, \c box_cox, and \c yeo_johnson;
   * \c sharpness for \c softplus; \c lower for \c bounded and \c lower_bounded;
   * \c upper for \c upper_bounded; and \f$0.0\f$ otherwise.
   * \return First parameter value.
   * \see parameter_2(), parameter_count(), bijector_scalar
   */
  LAPLACE_HD double parameter_1() const {
    switch (map) {
      case scalar_bijector_map::affine: return location;
      case scalar_bijector_map::power:
      case scalar_bijector_map::box_cox:
      case scalar_bijector_map::yeo_johnson: return exponent;
      case scalar_bijector_map::softplus: return sharpness;
      case scalar_bijector_map::bounded:
      case scalar_bijector_map::lower_bounded: return lower;
      case scalar_bijector_map::upper_bounded: return upper;
      default: return 0.0;
    }
  }
  /**
   * \ingroup bijector_scalar
   * \brief Returns the second parameter of the map in the order of partial derivatives.
   *
   * Returns \c scale for \c affine; \c upper for \c bounded; and \f$0.0\f$ otherwise.
   * \return Second parameter value.
   * \see parameter_1(), parameter_count(), bijector_scalar
   */
  LAPLACE_HD double parameter_2() const {
    switch (map) {
      case scalar_bijector_map::affine: return scale;
      case scalar_bijector_map::bounded: return upper;
      default: return 0.0;
    }
  }
  // the number of parameters with partial derivatives (0 for power)
  /**
   * \ingroup bijector_scalar
   * \brief Returns the number of parameters with partial derivatives.
   *
   * Returns 2 for \c affine and \c bounded; 1 for \c softplus, \c lower_bounded, \c upper_bounded,
   * \c box_cox, and \c yeo_johnson; and 0 for \c power and parameter-free maps.
   * \return Count of differentiated parameters (0, 1, or 2).
   * \see parameter_1(), parameter_2(), bijector_scalar
   */
  LAPLACE_HD int parameter_count() const {
    switch (map) {
      case scalar_bijector_map::affine:
      case scalar_bijector_map::bounded: return 2;
      case scalar_bijector_map::softplus:
      case scalar_bijector_map::lower_bounded:
      case scalar_bijector_map::upper_bounded:
      case scalar_bijector_map::box_cox:
      case scalar_bijector_map::yeo_johnson: return 1;
      default: return 0;
    }
  }

  // g and its derivatives; h and its derivatives
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the forward map \f$g(v)\f$.
   *
   * Maps an argument \f$v\f$ from the domain \f$V\f$ to \f$u = g(v) \in U\f$.
   * When inverted is true, evaluates the inverse map \f$h(v)\f$.
   *
   * \param v Point in the domain of the forward map.
   * \return Value \f$g(v)\f$. Outside the closed domain or for invalid parameters, returns quiet NaN.
   *         On domain boundaries, returns the mathematical limit.
   *
   * Measured accuracy: within 10.9 ulpC for parameter-free maps and within 26 ulpC for maps with parameters.
   * \see inverse(), forward_d1(), forward_log_jacobian(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward(double v) const {
    return inverted ? detail_scalar::inv<0>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<0>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the first derivative of the forward map, \f$g'(v)\f$.
   *
   * When inverted is true, evaluates \f$h'(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return First derivative \f$g'(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d2(), forward_log_jacobian(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d1(double v) const {
    return inverted ? detail_scalar::inv<1>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<1>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the second derivative of the forward map, \f$g''(v)\f$.
   *
   * When inverted is true, evaluates \f$h''(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Second derivative \f$g''(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d1(), forward_d3(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d2(double v) const {
    return inverted ? detail_scalar::inv<2>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<2>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the third derivative of the forward map, \f$g^{(3)}(v)\f$.
   *
   * When inverted is true, evaluates \f$h^{(3)}(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Third derivative \f$g^{(3)}(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d2(), forward_d4(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d3(double v) const {
    return inverted ? detail_scalar::inv<3>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<3>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fourth derivative of the forward map, \f$g^{(4)}(v)\f$.
   *
   * When inverted is true, evaluates \f$h^{(4)}(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Fourth derivative \f$g^{(4)}(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d3(), forward_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d4(double v) const {
    return inverted ? detail_scalar::inv<4>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<4>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fifth derivative of the forward map, \f$g^{(5)}(v)\f$.
   *
   * When inverted is true, evaluates \f$h^{(5)}(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Fifth derivative \f$g^{(5)}(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d4(), forward_d6(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d5(double v) const {
    return inverted ? detail_scalar::inv<5>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<5>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the sixth derivative of the forward map, \f$g^{(6)}(v)\f$.
   *
   * When inverted is true, evaluates \f$h^{(6)}(v)\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Sixth derivative \f$g^{(6)}(v)\f$, or NaN outside the closed domain.
   * \see forward(), forward_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_d6(double v) const {
    return inverted ? detail_scalar::inv<6>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd<6>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the inverse map \f$h(u) = g^{-1}(u)\f$.
   *
   * Maps an argument \f$u\f$ from the image \f$U\f$ back to \f$v = h(u) \in V\f$.
   * When inverted is true, evaluates the forward map \f$g(u)\f$.
   *
   * \param u Point in the domain of the inverse map.
   * \return Value \f$h(u)\f$. Outside the closed domain or for invalid parameters, returns quiet NaN.
   *         On domain boundaries, returns the mathematical limit.
   *
   * Measured accuracy: within 11 ulpC for parameter-free maps and within 42.8 ulpC (gcc) / 46.3 ulpC (CUDA) for maps with parameters.
   * \see forward(), inverse_d1(), inverse_log_jacobian(), inverse_distance_upper(), inverse_distance_lower(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse(double u) const {
    return inverted ? detail_scalar::fwd<0>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<0>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the first derivative of the inverse map, \f$h'(u)\f$.
   *
   * When inverted is true, evaluates \f$g'(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return First derivative \f$h'(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d2(), inverse_log_jacobian(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d1(double u) const {
    return inverted ? detail_scalar::fwd<1>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<1>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the second derivative of the inverse map, \f$h''(u)\f$.
   *
   * When inverted is true, evaluates \f$g''(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Second derivative \f$h''(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d1(), inverse_d3(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d2(double u) const {
    return inverted ? detail_scalar::fwd<2>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<2>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the third derivative of the inverse map, \f$h^{(3)}(u)\f$.
   *
   * When inverted is true, evaluates \f$g^{(3)}(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Third derivative \f$h^{(3)}(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d2(), inverse_d4(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d3(double u) const {
    return inverted ? detail_scalar::fwd<3>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<3>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fourth derivative of the inverse map, \f$h^{(4)}(u)\f$.
   *
   * When inverted is true, evaluates \f$g^{(4)}(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Fourth derivative \f$h^{(4)}(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d3(), inverse_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d4(double u) const {
    return inverted ? detail_scalar::fwd<4>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<4>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fifth derivative of the inverse map, \f$h^{(5)}(u)\f$.
   *
   * When inverted is true, evaluates \f$g^{(5)}(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Fifth derivative \f$h^{(5)}(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d4(), inverse_d6(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d5(double u) const {
    return inverted ? detail_scalar::fwd<5>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<5>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the sixth derivative of the inverse map, \f$h^{(6)}(u)\f$.
   *
   * When inverted is true, evaluates \f$g^{(6)}(u)\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Sixth derivative \f$h^{(6)}(u)\f$, or NaN outside the closed domain.
   * \see inverse(), inverse_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_d6(double u) const {
    return inverted ? detail_scalar::fwd<6>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv<6>(map, u, parameter_1(), parameter_2());
  }

  // log|g'(v)| and log|h'(u)|, with their derivatives
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the forward log-Jacobian, \f$\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates the inverse log-Jacobian \f$\log|h'(v)|\f$.
   *
   * \param v Point in the domain of forward().
   * \return Value \f$\log|g'(v)|\f$. Outside the closed domain or for invalid parameters, returns quiet NaN.
   *
   * Measured accuracy: within 6.57 ulpC for parameter-free maps and within 18.4 ulpC for maps with parameters.
   * \see forward(), forward_log_jacobian_d1(), inverse_log_jacobian(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian(double v) const {
    return inverted ? detail_scalar::inv_lj<0>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<0>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the first derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return First derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d2(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d1(double v) const {
    return inverted ? detail_scalar::inv_lj<1>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<1>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the second derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^2}{\mathrm{d}v^2}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^2}{\mathrm{d}v^2}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Second derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d1(), forward_log_jacobian_d3(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d2(double v) const {
    return inverted ? detail_scalar::inv_lj<2>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<2>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the third derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^3}{\mathrm{d}v^3}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^3}{\mathrm{d}v^3}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Third derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d2(), forward_log_jacobian_d4(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d3(double v) const {
    return inverted ? detail_scalar::inv_lj<3>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<3>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fourth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^4}{\mathrm{d}v^4}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^4}{\mathrm{d}v^4}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Fourth derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d3(), forward_log_jacobian_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d4(double v) const {
    return inverted ? detail_scalar::inv_lj<4>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<4>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fifth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^5}{\mathrm{d}v^5}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^5}{\mathrm{d}v^5}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Fifth derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d4(), forward_log_jacobian_d6(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d5(double v) const {
    return inverted ? detail_scalar::inv_lj<5>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<5>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the sixth derivative of the forward log-Jacobian, \f$\frac{\mathrm{d}^6}{\mathrm{d}v^6}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^6}{\mathrm{d}v^6}\log|h'(v)|\f$. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Sixth derivative of forward log-Jacobian, or NaN outside the closed domain.
   * \see forward_log_jacobian(), forward_log_jacobian_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_log_jacobian_d6(double v) const {
    return inverted ? detail_scalar::inv_lj<6>(map, v, parameter_1(), parameter_2())
                    : detail_scalar::fwd_lj<6>(map, v, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the inverse log-Jacobian, \f$\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates the forward log-Jacobian \f$\log|g'(u)|\f$.
   *
   * \param u Point in the domain of inverse().
   * \return Value \f$\log|h'(u)|\f$. Outside the closed domain or for invalid parameters, returns quiet NaN.
   *
   * Measured accuracy: within 10.5 ulpC for parameter-free maps and within 49.4 ulpC (gcc) / 43 ulpC (CUDA) for maps with parameters.
   * \see inverse(), inverse_log_jacobian_d1(), forward_log_jacobian(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian(double u) const {
    return inverted ? detail_scalar::fwd_lj<0>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<0>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the first derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return First derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d2(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d1(double u) const {
    return inverted ? detail_scalar::fwd_lj<1>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<1>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the second derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^2}{\mathrm{d}u^2}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^2}{\mathrm{d}u^2}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Second derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d1(), inverse_log_jacobian_d3(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d2(double u) const {
    return inverted ? detail_scalar::fwd_lj<2>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<2>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the third derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^3}{\mathrm{d}u^3}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^3}{\mathrm{d}u^3}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Third derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d2(), inverse_log_jacobian_d4(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d3(double u) const {
    return inverted ? detail_scalar::fwd_lj<3>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<3>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fourth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^4}{\mathrm{d}u^4}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^4}{\mathrm{d}u^4}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Fourth derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d3(), inverse_log_jacobian_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d4(double u) const {
    return inverted ? detail_scalar::fwd_lj<4>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<4>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the fifth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^5}{\mathrm{d}u^5}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^5}{\mathrm{d}u^5}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Fifth derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d4(), inverse_log_jacobian_d6(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d5(double u) const {
    return inverted ? detail_scalar::fwd_lj<5>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<5>(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates the sixth derivative of the inverse log-Jacobian, \f$\frac{\mathrm{d}^6}{\mathrm{d}u^6}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates \f$\frac{\mathrm{d}^6}{\mathrm{d}u^6}\log|g'(u)|\f$. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Sixth derivative of inverse log-Jacobian, or NaN outside the closed domain.
   * \see inverse_log_jacobian(), inverse_log_jacobian_d5(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_log_jacobian_d6(double u) const {
    return inverted ? detail_scalar::fwd_lj<6>(map, u, parameter_1(), parameter_2())
                    : detail_scalar::inv_lj<6>(map, u, parameter_1(), parameter_2());
  }

  // partial derivatives of total order K in (argument, parameter 1, parameter 2)
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 1 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Returns a bijector_partials<1> struct containing 3 components:
   * - \c c[0] (\f$n_1=0, n_2=0\f$): pure derivative in argument, \f$g'(v)\f$.
   * - \c c[1] (\f$n_1=1, n_2=0\f$): derivative in parameter 1, \f$\frac{\partial g}{\partial p_1}\f$.
   * - \c c[2] (\f$n_1=0, n_2=1\f$): derivative in parameter 2, \f$\frac{\partial g}{\partial p_2}\f$.
   *
   * When inverted is true, evaluates partial derivatives of the inverse map.
   * Outside the closed domain or for invalid parameters, all components are quiet NaN.
   * For \c power, components in the exponent are quiet NaN.
   * For maps without parameters, components in absent parameters are 0.0.
   *
   * Measured accuracy: within 26 ulpC on the conditioned scale.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<1> containing the partial derivatives.
   * \see forward(), forward_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<1> forward_partials_d1(double v) const {
    bijector_partials<1> r;
    if (inverted) detail_scalar::inv_partials<1>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<1>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 2 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<2> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<2> containing the partial derivatives.
   * \see forward_partials_d1(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<2> forward_partials_d2(double v) const {
    bijector_partials<2> r;
    if (inverted) detail_scalar::inv_partials<2>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<2>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 3 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<3> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<3> containing the partial derivatives.
   * \see forward_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<3> forward_partials_d3(double v) const {
    bijector_partials<3> r;
    if (inverted) detail_scalar::inv_partials<3>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<3>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 4 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<4> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<4> containing the partial derivatives.
   * \see forward_partials_d3(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<4> forward_partials_d4(double v) const {
    bijector_partials<4> r;
    if (inverted) detail_scalar::inv_partials<4>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<4>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 5 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<5> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<5> containing the partial derivatives.
   * \see forward_partials_d4(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<5> forward_partials_d5(double v) const {
    bijector_partials<5> r;
    if (inverted) detail_scalar::inv_partials<5>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<5>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 6 of the forward map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<6> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<6> containing the partial derivatives.
   * \see forward_partials_d5(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<6> forward_partials_d6(double v) const {
    bijector_partials<6> r;
    if (inverted) detail_scalar::inv_partials<6>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_partials<6>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 1 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Returns a bijector_partials<1> struct containing 3 components:
   * - \c c[0] (\f$n_1=0, n_2=0\f$): pure derivative in argument, \f$h'(u)\f$.
   * - \c c[1] (\f$n_1=1, n_2=0\f$): derivative in parameter 1, \f$\frac{\partial h}{\partial p_1}\f$.
   * - \c c[2] (\f$n_1=0, n_2=1\f$): derivative in parameter 2, \f$\frac{\partial h}{\partial p_2}\f$.
   *
   * When inverted is true, evaluates partial derivatives of the forward map.
   * Outside the closed domain or for invalid parameters, all components are quiet NaN.
   * For \c power, components in the exponent are quiet NaN.
   * For maps without parameters, components in absent parameters are 0.0.
   *
   * Measured accuracy: within 24.9 ulpC (gcc) / 39.9 ulpC (CUDA) on the conditioned scale.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<1> containing the partial derivatives.
   * \see inverse(), inverse_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<1> inverse_partials_d1(double u) const {
    bijector_partials<1> r;
    if (inverted) detail_scalar::fwd_partials<1>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<1>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 2 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<2> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<2> containing the partial derivatives.
   * \see inverse_partials_d1(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<2> inverse_partials_d2(double u) const {
    bijector_partials<2> r;
    if (inverted) detail_scalar::fwd_partials<2>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<2>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 3 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<3> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<3> containing the partial derivatives.
   * \see inverse_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<3> inverse_partials_d3(double u) const {
    bijector_partials<3> r;
    if (inverted) detail_scalar::fwd_partials<3>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<3>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 4 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<4> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<4> containing the partial derivatives.
   * \see inverse_partials_d3(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<4> inverse_partials_d4(double u) const {
    bijector_partials<4> r;
    if (inverted) detail_scalar::fwd_partials<4>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<4>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 5 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<5> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<5> containing the partial derivatives.
   * \see inverse_partials_d4(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<5> inverse_partials_d5(double u) const {
    bijector_partials<5> r;
    if (inverted) detail_scalar::fwd_partials<5>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<5>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 6 of the inverse map in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<6> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<6> containing the partial derivatives.
   * \see inverse_partials_d5(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<6> inverse_partials_d6(double u) const {
    bijector_partials<6> r;
    if (inverted) detail_scalar::fwd_partials<6>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_partials<6>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 1 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Returns a bijector_partials<1> struct containing 3 components for \f$\log|g'(v)|\f$:
   * - \c c[0] (\f$n_1=0, n_2=0\f$): pure derivative in argument, \f$\frac{\mathrm{d}}{\mathrm{d}v}\log|g'(v)|\f$.
   * - \c c[1] (\f$n_1=1, n_2=0\f$): derivative in parameter 1, \f$\frac{\partial}{\partial p_1}\log|g'(v)|\f$.
   * - \c c[2] (\f$n_1=0, n_2=1\f$): derivative in parameter 2, \f$\frac{\partial}{\partial p_2}\log|g'(v)|\f$.
   *
   * When inverted is true, evaluates partial derivatives of the inverse log-Jacobian.
   * Outside the closed domain or for invalid parameters, all components are quiet NaN.
   * For \c power, components in the exponent are quiet NaN.
   * For maps without parameters, components in absent parameters are 0.0.
   *
   * Measured accuracy: within 18.4 ulpC on the conditioned scale.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<1> containing the partial derivatives.
   * \see forward_log_jacobian(), forward_log_jacobian_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<1> forward_log_jacobian_partials_d1(double v) const {
    bijector_partials<1> r;
    if (inverted) detail_scalar::inv_lj_partials<1>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<1>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 2 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<2> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<2> containing the partial derivatives.
   * \see forward_log_jacobian_partials_d1(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<2> forward_log_jacobian_partials_d2(double v) const {
    bijector_partials<2> r;
    if (inverted) detail_scalar::inv_lj_partials<2>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<2>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 3 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<3> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<3> containing the partial derivatives.
   * \see forward_log_jacobian_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<3> forward_log_jacobian_partials_d3(double v) const {
    bijector_partials<3> r;
    if (inverted) detail_scalar::inv_lj_partials<3>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<3>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 4 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<4> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<4> containing the partial derivatives.
   * \see forward_log_jacobian_partials_d3(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<4> forward_log_jacobian_partials_d4(double v) const {
    bijector_partials<4> r;
    if (inverted) detail_scalar::inv_lj_partials<4>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<4>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 5 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<5> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<5> containing the partial derivatives.
   * \see forward_log_jacobian_partials_d4(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<5> forward_log_jacobian_partials_d5(double v) const {
    bijector_partials<5> r;
    if (inverted) detail_scalar::inv_lj_partials<5>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<5>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 6 of the forward log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<6> in index order. See \ref bijector_scalar.
   * \param v Point in the domain of forward().
   * \return Structure bijector_partials<6> containing the partial derivatives.
   * \see forward_log_jacobian_partials_d5(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<6> forward_log_jacobian_partials_d6(double v) const {
    bijector_partials<6> r;
    if (inverted) detail_scalar::inv_lj_partials<6>(map, v, parameter_1(), parameter_2(), r.c);
    else detail_scalar::fwd_lj_partials<6>(map, v, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 1 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Returns a bijector_partials<1> struct containing 3 components for \f$\log|h'(u)|\f$:
   * - \c c[0] (\f$n_1=0, n_2=0\f$): pure derivative in argument, \f$\frac{\mathrm{d}}{\mathrm{d}u}\log|h'(u)|\f$.
   * - \c c[1] (\f$n_1=1, n_2=0\f$): derivative in parameter 1, \f$\frac{\partial}{\partial p_1}\log|h'(u)|\f$.
   * - \c c[2] (\f$n_1=0, n_2=1\f$): derivative in parameter 2, \f$\frac{\partial}{\partial p_2}\log|h'(u)|\f$.
   *
   * When inverted is true, evaluates partial derivatives of the forward log-Jacobian.
   * Outside the closed domain or for invalid parameters, all components are quiet NaN.
   * For \c power, components in the exponent are quiet NaN.
   * For maps without parameters, components in absent parameters are 0.0.
   *
   * Measured accuracy: within 49.4 ulpC (gcc) / 43 ulpC (CUDA) on the conditioned scale.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<1> containing the partial derivatives.
   * \see inverse_log_jacobian(), inverse_log_jacobian_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<1> inverse_log_jacobian_partials_d1(double u) const {
    bijector_partials<1> r;
    if (inverted) detail_scalar::fwd_lj_partials<1>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<1>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 2 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<2> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<2> containing the partial derivatives.
   * \see inverse_log_jacobian_partials_d1(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<2> inverse_log_jacobian_partials_d2(double u) const {
    bijector_partials<2> r;
    if (inverted) detail_scalar::fwd_lj_partials<2>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<2>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 3 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<3> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<3> containing the partial derivatives.
   * \see inverse_log_jacobian_partials_d2(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<3> inverse_log_jacobian_partials_d3(double u) const {
    bijector_partials<3> r;
    if (inverted) detail_scalar::fwd_lj_partials<3>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<3>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 4 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<4> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<4> containing the partial derivatives.
   * \see inverse_log_jacobian_partials_d3(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<4> inverse_log_jacobian_partials_d4(double u) const {
    bijector_partials<4> r;
    if (inverted) detail_scalar::fwd_lj_partials<4>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<4>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 5 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<5> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<5> containing the partial derivatives.
   * \see inverse_log_jacobian_partials_d4(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<5> inverse_log_jacobian_partials_d5(double u) const {
    bijector_partials<5> r;
    if (inverted) detail_scalar::fwd_lj_partials<5>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<5>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Evaluates partial derivatives of total order 6 of the inverse log-Jacobian in (argument, parameter 1, parameter 2).
   *
   * Components are stored in bijector_partials<6> in index order. See \ref bijector_scalar.
   * \param u Point in the domain of inverse().
   * \return Structure bijector_partials<6> containing the partial derivatives.
   * \see inverse_log_jacobian_partials_d5(), bijector_partials, bijector_scalar
   */
  LAPLACE_HD bijector_partials<6> inverse_log_jacobian_partials_d6(double u) const {
    bijector_partials<6> r;
    if (inverted) detail_scalar::fwd_lj_partials<6>(map, u, parameter_1(), parameter_2(), r.c);
    else detail_scalar::inv_lj_partials<6>(map, u, parameter_1(), parameter_2(), r.c);
    return r;
  }

  // the domain of forward() and the domain of inverse(), as closed bounds
  /**
   * \ingroup bijector_scalar
   * \brief Returns the lower bound of the domain of forward().
   * \return Finite or infinite lower bound, or NaN for invalid parameters.
   * \see forward_domain_upper(), inverse_domain_lower(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_domain_lower() const {
    return inverted ? detail_scalar::img_lo(map, parameter_1(), parameter_2())
                    : detail_scalar::dom_lo(map, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Returns the upper bound of the domain of forward().
   * \return Finite or infinite upper bound, or NaN for invalid parameters.
   * \see forward_domain_lower(), inverse_domain_upper(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double forward_domain_upper() const {
    return inverted ? detail_scalar::img_hi(map, parameter_1(), parameter_2())
                    : detail_scalar::dom_hi(map, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Returns the lower bound of the domain of inverse().
   * \return Finite or infinite lower bound, or NaN for invalid parameters.
   * \see inverse_domain_upper(), forward_domain_lower(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_domain_lower() const {
    return inverted ? detail_scalar::dom_lo(map, parameter_1(), parameter_2())
                    : detail_scalar::img_lo(map, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Returns the upper bound of the domain of inverse().
   * \return Finite or infinite upper bound, or NaN for invalid parameters.
   * \see inverse_domain_lower(), forward_domain_upper(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_domain_upper() const {
    return inverted ? detail_scalar::dom_hi(map, parameter_1(), parameter_2())
                    : detail_scalar::img_hi(map, parameter_1(), parameter_2());
  }
  // true when the map is decreasing; inversion keeps it
  /**
   * \ingroup bijector_scalar
   * \brief Returns whether the map is strictly decreasing.
   *
   * Returns true for \c inverse, \c inverse_sq, \c power with negative exponent,
   * and \c affine with negative scale. Inversion preserves monotonicity.
   * \return True if decreasing, false if increasing.
   * \see bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE bool is_decreasing() const {
    switch (map) {
      case scalar_bijector_map::inverse:
      case scalar_bijector_map::inverse_sq: return true;
      case scalar_bijector_map::power: return exponent < 0.0;
      case scalar_bijector_map::affine: return scale < 0.0;
      default: return false;
    }
  }

  // the points at which forward(), respectively inverse(), is not smooth:
  // Yeo-Johnson at v = 0 (u = g(0) = 0), unless exponent = 1 (the identity)
  /**
   * \ingroup bijector_scalar
   * \brief Returns the non-smooth points of forward().
   *
   * For \c yeo_johnson with \f$\lambda \neq 1\f$, reports 1 kink at \f$v = 0\f$
   * with \c continuous_order equal to 2.
   * \return Structure bijector_kinks with kink count, positions, and continuous order.
   * \see inverse_kinks(), bijector_kinks, bijector_scalar
   */
  LAPLACE_HD bijector_kinks forward_kinks() const {
    bijector_kinks r;
    if (map == scalar_bijector_map::yeo_johnson && exponent != 1.0) {
      r.count = 1;
      r.point[0] = 0.0;
      r.continuous_order = 2;
    }
    return r;
  }
  /**
   * \ingroup bijector_scalar
   * \brief Returns the non-smooth points of inverse().
   *
   * Returns the kinks of the inverse map (for \c yeo_johnson with \f$\lambda \neq 1\f$,
   * 1 kink at \f$u = g(0) = 0\f$ with \c continuous_order equal to 2).
   * \return Structure bijector_kinks with kink count, positions, and continuous order.
   * \see forward_kinks(), bijector_kinks, bijector_scalar
   */
  LAPLACE_HD bijector_kinks inverse_kinks() const { return forward_kinks(); }  // g(0) = 0

  // The distance of inverse(u) to the upper and to the lower bound of its
  // range (forward_domain_upper(), forward_domain_lower()), computed directly
  // where the difference would cancel.
  /**
   * \ingroup bijector_scalar
   * \brief Distance from the inverse map \f$h(u)\f$ to the upper bound of its range.
   *
   * Evaluates \f$\sup(V) - h(u)\f$ directly without forming \f$h(u)\f$ where catastrophic
   * cancellation would otherwise occur near the bound (e.g. \f$1 - \operatorname{expit}(u) = \operatorname{expit}(-u)\f$
   * for \c logit, \f$(b - a)\operatorname{expit}(-u)\f$ for \c bounded, \f$e^{-u}\f$ for \c upper_bounded).
   * Returns \f$+\infty\f$ when the upper bound is \f$+\infty\f$.
   * When inverted is true, computes the generic difference \c forward_domain_upper() - inverse(u).
   *
   * \param u Point in the domain of inverse().
   * \return Distance to the upper bound, or NaN outside the closed domain or for invalid parameters.
   *
   * Measured accuracy: within 2.28 ulpC across all bounded maps; for \c bounded, 0.721 ulpC; for \c upper_bounded, 0.193 ulpC.
   * \see inverse(), inverse_distance_lower(), forward_domain_upper(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_distance_upper(double u) const {
    if (inverted) return forward_domain_upper() - inverse(u);
    return detail_scalar::inv_to_upper(map, u, parameter_1(), parameter_2());
  }
  /**
   * \ingroup bijector_scalar
   * \brief Distance from the inverse map \f$h(u)\f$ to the lower bound of its range.
   *
   * Evaluates \f$h(u) - \inf(V)\f$ directly without forming \f$h(u)\f$ where catastrophic
   * cancellation would otherwise occur near the bound (e.g. \f$h(u)\f$ for \c logit,
   * \f$(b - a)\operatorname{expit}(u)\f$ for \c bounded, \f$e^u\f$ for \c lower_bounded).
   * Returns \f$+\infty\f$ when the lower bound is \f$-\infty\f$.
   * When inverted is true, computes the generic difference \c inverse(u) - forward_domain_lower().
   *
   * \param u Point in the domain of inverse().
   * \return Distance to the lower bound, or NaN outside the closed domain or for invalid parameters.
   *
   * Measured accuracy: within 2.28 ulpC across all bounded maps; for \c bounded, 0.584 ulpC; for \c lower_bounded, 0.273 ulpC.
   * \see inverse(), inverse_distance_upper(), forward_domain_lower(), bijector_scalar
   */
  LAPLACE_HD_FORCEINLINE double inverse_distance_lower(double u) const {
    if (inverted) return inverse(u) - forward_domain_lower();
    return detail_scalar::inv_to_lower(map, u, parameter_1(), parameter_2());
  }
};

}  // namespace laplace::bijector
