// Scalar bijectors lower_bounded, upper_bounded (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   lower_bounded: (p1, inf) -> R, g(v) = log(v - p1); p1 = lower, p2 unused
//   upper_bounded: (-inf, p1) -> R, g(v) = -log(p1 - v), increasing; p1 = upper, p2 unused
//
// For each map NAME, with v in the domain of g and u in the domain of h = g^-1:
//   NAME_fwd<K>(v, p1, p2)      g^(K)(v), K = 0..6 (K = 0 the value)
//   NAME_inv<K>(u, p1, p2)      h^(K)(u)
//   NAME_fwd_lj<K>(v, p1, p2)   d^K/dv^K log|g'(v)|
//   NAME_inv_lj<K>(u, p1, p2)   d^K/du^K log|h'(u)|
//   NAME_fwd_partials<K>(v, p1, p2, out), NAME_inv_partials, NAME_fwd_lj_partials,
//   NAME_inv_lj_partials        the partial derivatives of total order K = 1..6 in
//                               (argument, p1, p2), out[bijector_partials<K>::index(n1, n2)]
//                               = d^K / (d arg^n0 d p1^n1 d p2^n2), n0 = K - n1 - n2;
//                               components in a parameter the map does not have are 0
//   NAME_dom_lo/hi(p1, p2), NAME_img_lo/hi(p1, p2)   the closed bounds of the domains of g and h
//   NAME_inv_to_upper/lower(u, p1, p2)   dom_hi - h(u) and h(u) - dom_lo, computed directly
// Invalid parameters: every function returns NaN. The arguments have been checked
// against the domain bounds by the dispatch; a value on a bound gives its limit.
// A bound must be finite; an infinite bound is an invalid parameter here.
//
// Formulas (2026-11-12). Both maps depend on the argument and the bound only through
// one difference, which is formed by a single subtraction (exact near the bound):
//   lower_bounded: t = v - p1 > 0. g = log t, g' = 1/t, log|g'| = -log t = -g.
//     g^(K) = (-1)^(K-1) (K-1)! / t^K (K >= 1), log|g'|^(K) = -g^(K).
//     h = p1 + e^u, so h - p1 = e^u exactly (the distance to the lower bound), and
//     h^(K) = e^u for K >= 1; log|h'| = u, whose derivatives are 1, 0, 0, ...
//     Partials in p1: g depends on v - p1 only, so d/dp1 = -d/dv on g, and the
//     component of order n1 in p1 is (-1)^n1 times the pure derivative of order K.
//     h: d/dp1 h = 1 and nothing else depends on p1, so the partials of h are e^u at
//     (n0, n1) = (K, 0), 1 at (0, 1) for K = 1, and 0 elsewhere.
//   upper_bounded: s = p1 - v > 0 (one subtraction). g = -log s, g' = 1/s = e^g, so
//     log|g'| = g exactly, and g^(K) = (K-1)! / s^K (K >= 1), the same for log|g'|.
//     h = p1 - e^(-u), so p1 - h = e^(-u) exactly (the distance to the upper bound),
//     h^(K) = (-1)^(K+1) e^(-u) for K >= 1; log|h'| = -u, whose derivatives are -1, 0, ...
//     Partials in p1: g depends on p1 - v only, so d/dp1 = -d/dv on g again, and the
//     component of order n1 in p1 is (-1)^n1 times the pure derivative; the partials
//     of h are (-1)^(K+1) e^(-u) at (K, 0), 1 at (0, 1) for K = 1, and 0.
// The partials of order K go to out in the index order of bijector_partials<K>
// (component index(n1, n2) = n2 (K + 1) - n2 (n2 - 1)/2 + n1; here n2 = 0 always, so the
// component of order n1 in p1 is out[n1]); the others are 0.
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>

namespace laplace::bijector::detail_scalar {

// a finite bound; anything else is an invalid parameter
LAPLACE_HD_FORCEINLINE bool shifted_valid(double p1) { return Kokkos::isfinite(p1); }

// the bounded maps have no second parameter, so their partials fill out[0..K] only
template <int K>
LAPLACE_HD_FORCEINLINE void shifted_zero(double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
}

// ---------------------------------------------------------------- lower_bounded

template <int K>
LAPLACE_HD double lower_bounded_fwd(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  const double t = x - p1;
  if constexpr (K == 0) return m::log(t);
  else return log_dk<K>(1.0, t);
}
template <int K>
LAPLACE_HD double lower_bounded_inv(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  if constexpr (K == 0) return p1 + m::exp(x);
  else return m::exp(x);
}
template <int K>
LAPLACE_HD double lower_bounded_fwd_lj(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  const double t = x - p1;
  if constexpr (K == 0) return -m::log(t);
  else return -log_dk<K>(1.0, t);
}
template <int K>
LAPLACE_HD double lower_bounded_inv_lj(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  if constexpr (K == 0) return x;
  else if constexpr (K == 1) return 1.0;
  else return 0.0;
}
template <int K>
LAPLACE_HD void lower_bounded_fwd_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const double f = log_dk<K>(1.0, x - p1);
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = (n1 % 2 == 0) ? f : -f;  // (-1)^n1
}
template <int K>
LAPLACE_HD void lower_bounded_inv_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  out[0] = m::exp(x);  // (n0, n1) = (K, 0)
  if constexpr (K == 1) out[1] = 1.0;  // d/dp1 h = 1, (n0, n1) = (0, 1)
}
template <int K>
LAPLACE_HD void lower_bounded_fwd_lj_partials(double x, double p1, double p2, double* out) {
  lower_bounded_fwd_partials<K>(x, p1, p2, out);
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = -out[i];  // log|g'| = -g
}
template <int K>
LAPLACE_HD void lower_bounded_inv_lj_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  if constexpr (K == 1) out[0] = 1.0;  // d/du u = 1
}
LAPLACE_HD_FORCEINLINE double lower_bounded_dom_lo(double p1, double p2) { return shifted_valid(p1) ? p1 : nan(); }
LAPLACE_HD_FORCEINLINE double lower_bounded_dom_hi(double p1, double p2) { return shifted_valid(p1) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double lower_bounded_img_lo(double p1, double p2) { return shifted_valid(p1) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double lower_bounded_img_hi(double p1, double p2) { return shifted_valid(p1) ? inf() : nan(); }
// dom_hi - h(u) = inf - h(u)
LAPLACE_HD_FORCEINLINE double lower_bounded_inv_to_upper(double u, double p1, double p2) {
  return shifted_valid(p1) ? inf() : nan();
}
// h(u) - p1 = e^u exactly
LAPLACE_HD_FORCEINLINE double lower_bounded_inv_to_lower(double u, double p1, double p2) {
  return shifted_valid(p1) ? m::exp(u) : nan();
}

// ---------------------------------------------------------------- upper_bounded

template <int K>
LAPLACE_HD double upper_bounded_fwd(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  const double s = p1 - x;
  if constexpr (K == 0) return -m::log(s);
  else return cpow<K>(fact<K - 1>(), 1.0 / s);
}
template <int K>
LAPLACE_HD double upper_bounded_inv(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  if constexpr (K == 0) return p1 - m::exp(-x);
  else return -sgn<K>() * m::exp(-x);  // (-1)^(K+1) e^(-u)
}
template <int K>
LAPLACE_HD double upper_bounded_fwd_lj(double x, double p1, double p2) {
  return upper_bounded_fwd<K>(x, p1, p2);  // log|g'| = -log s = g
}
template <int K>
LAPLACE_HD double upper_bounded_inv_lj(double x, double p1, double p2) {
  if (!shifted_valid(p1)) return nan();
  if constexpr (K == 0) return -x;
  else if constexpr (K == 1) return -1.0;
  else return 0.0;
}
template <int K>
LAPLACE_HD void upper_bounded_fwd_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const double f = cpow<K>(fact<K - 1>(), 1.0 / (p1 - x));  // g^(K), the pure derivative
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = (n1 % 2 == 0) ? f : -f;  // (-1)^n1
}
template <int K>
LAPLACE_HD void upper_bounded_inv_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  out[0] = -sgn<K>() * m::exp(-x);  // (n0, n1) = (K, 0)
  if constexpr (K == 1) out[1] = 1.0;  // d/dp1 h = 1
}
template <int K>
LAPLACE_HD void upper_bounded_fwd_lj_partials(double x, double p1, double p2, double* out) {
  upper_bounded_fwd_partials<K>(x, p1, p2, out);  // log|g'| = g
}
template <int K>
LAPLACE_HD void upper_bounded_inv_lj_partials(double x, double p1, double p2, double* out) {
  shifted_zero<K>(out);
  if (!shifted_valid(p1)) {
    constexpr int n = (K + 1) * (K + 2) / 2;
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  if constexpr (K == 1) out[0] = -1.0;  // d/du (-u) = -1
}
LAPLACE_HD_FORCEINLINE double upper_bounded_dom_lo(double p1, double p2) { return shifted_valid(p1) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double upper_bounded_dom_hi(double p1, double p2) { return shifted_valid(p1) ? p1 : nan(); }
LAPLACE_HD_FORCEINLINE double upper_bounded_img_lo(double p1, double p2) { return shifted_valid(p1) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double upper_bounded_img_hi(double p1, double p2) { return shifted_valid(p1) ? inf() : nan(); }
// p1 - h(u) = e^(-u) exactly
LAPLACE_HD_FORCEINLINE double upper_bounded_inv_to_upper(double u, double p1, double p2) {
  return shifted_valid(p1) ? m::exp(-u) : nan();
}
// h(u) - dom_lo = inf - (-inf)
LAPLACE_HD_FORCEINLINE double upper_bounded_inv_to_lower(double u, double p1, double p2) {
  return shifted_valid(p1) ? inf() : nan();
}

}  // namespace laplace::bijector::detail_scalar
