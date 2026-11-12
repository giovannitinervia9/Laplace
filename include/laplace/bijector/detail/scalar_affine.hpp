// Scalar bijector affine (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   affine: R -> R, g(v) = (v - p1)/p2, p2 != 0; p1 = location, p2 = scale
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
//
// Formulas (2026-11-12). Notation: t = v - p1 (one subtraction, exact when v and p1
// are close), q = 1/p2. Everything is linear in v, u and p1 or a power of q, so
// there is no cancellation: the value is (v - p1)/p2 and h(u) = p1 + p2 u, each
// a single product and a single sum.
//   g = (v - p1)/p2.  g' = 1/p2, g'' = 0.
//   log|g'| = -log|p2|, whose only nonzero derivatives are in p2:
//             d^K/dp2^K = (-1)^K (K-1)! / p2^K.
//   h = p1 + p2 u.    h' = p2, h'' = 0.
//   log|h'| = log|p2|, d^K/dp2^K = (-1)^(K-1) (K-1)! / p2^K (K >= 1).
//   Mixed partials of g (n0 = order in v, n1 in p1, n2 in p2):
//     n0 = 1: 1/p2 with n1 = 0, then (-1)^n2 n2! q^(n2+1); 0 for n1 > 0.
//     n0 = 0: n1 = 0: (v - p1) (-1)^n2 n2! q^(n2+1); n1 = 1: (-1)^(n2+1) n2! q^(n2+1);
//             0 for n1 >= 2.
//   Mixed partials of h (n0 in u): n0 = 0: (n1, n2) = (1, 0) gives 1, (0, 1) gives u;
//     n0 = 1: (0, 0) gives p2, (0, 1) gives 1; all others 0.
// The partials of order K are written to out in the index order of bijector_partials<K>
// (component index(n1, n2) = n2 (K + 1) - n2 (n2 - 1)/2 + n1); the other components are 0.
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>

namespace laplace::bijector::detail_scalar {

// finite location, finite nonzero scale; anything else is an invalid parameter
LAPLACE_HD_FORCEINLINE bool affine_valid(double p1, double p2) {
  return Kokkos::isfinite(p1) && Kokkos::isfinite(p2) && p2 != 0.0;
}

// position of the component (n1, n2) in bijector_partials<K> (the same formula as scalar.hpp)
LAPLACE_HD_FORCEINLINE int affine_index(int K, int n1, int n2) { return n2 * (K + 1) - n2 * (n2 - 1) / 2 + n1; }

// x^n and n! for the small n of the partials (n <= 6)
LAPLACE_HD_FORCEINLINE double affine_pow(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}
LAPLACE_HD_FORCEINLINE double affine_fact(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

// the mixed partial of g of order (n0, n1, n2) with t = v - p1
LAPLACE_HD_FORCEINLINE double affine_fwd_mixed(int n0, int n1, int n2, double t, double p2) {
  if (n0 >= 2) return 0.0;
  const double q = 1.0 / p2;
  const double sign = (n2 % 2 == 0) ? 1.0 : -1.0;  // (-1)^n2
  const double c = affine_fact(n2) * affine_pow(q, n2 + 1);
  if (n0 == 1) return n1 == 0 ? sign * c : 0.0;
  if (n1 >= 2) return 0.0;
  if (n1 == 0) return t * sign * c;
  return -sign * c;  // n1 == 1: (-1)^(n2+1) n2! q^(n2+1), no t
}

// the mixed partial of h of order (n0, n1, n2) with u
LAPLACE_HD_FORCEINLINE double affine_inv_mixed(int n0, int n1, int n2, double u, double p2) {
  if (n0 == 0) {
    if (n1 == 1 && n2 == 0) return 1.0;
    if (n1 == 0 && n2 == 1) return u;
    return 0.0;
  }
  if (n0 == 1) {
    if (n1 != 0) return 0.0;
    if (n2 == 0) return p2;
    if (n2 == 1) return 1.0;
    return 0.0;
  }
  return 0.0;
}

template <int K>
LAPLACE_HD double affine_fwd(double x, double p1, double p2) {
  if (!affine_valid(p1, p2)) return nan();
  if constexpr (K == 0) return (x - p1) / p2;
  else if constexpr (K == 1) return 1.0 / p2;
  else return 0.0;
}
template <int K>
LAPLACE_HD double affine_inv(double x, double p1, double p2) {
  if (!affine_valid(p1, p2)) return nan();
  if constexpr (K == 0) return p1 + p2 * x;
  else if constexpr (K == 1) return p2;
  else return 0.0;
}
template <int K>
LAPLACE_HD double affine_fwd_lj(double x, double p1, double p2) {
  if (!affine_valid(p1, p2)) return nan();
  if constexpr (K == 0) return -m::log(Kokkos::fabs(p2));
  else return 0.0;
}
template <int K>
LAPLACE_HD double affine_inv_lj(double x, double p1, double p2) {
  if (!affine_valid(p1, p2)) return nan();
  if constexpr (K == 0) return m::log(Kokkos::fabs(p2));
  else return 0.0;
}
template <int K>
LAPLACE_HD void affine_fwd_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!affine_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const double t = x - p1;
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) out[affine_index(K, n1, n2)] = affine_fwd_mixed(K - n1 - n2, n1, n2, t, p2);
}
template <int K>
LAPLACE_HD void affine_inv_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!affine_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  for (int n2 = 0; n2 <= K; ++n2)
    for (int n1 = 0; n1 + n2 <= K; ++n1) out[affine_index(K, n1, n2)] = affine_inv_mixed(K - n1 - n2, n1, n2, x, p2);
}
// d^K/dp2^K (-log|p2|) = (-1)^K (K-1)!/p2^K sits in the component (0, K); the rest is 0
template <int K>
LAPLACE_HD void affine_fwd_lj_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!affine_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  if constexpr (K >= 1) out[affine_index(K, 0, K)] = affine_fact(K - 1) * affine_pow(1.0 / p2, K) * ((K % 2 == 0) ? 1.0 : -1.0);
}
// d^K/dp2^K log|p2| = (-1)^(K-1) (K-1)!/p2^K sits in the component (0, K)
template <int K>
LAPLACE_HD void affine_inv_lj_partials(double x, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!affine_valid(p1, p2)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  if constexpr (K >= 1) out[affine_index(K, 0, K)] = affine_fact(K - 1) * affine_pow(1.0 / p2, K) * ((K % 2 == 1) ? 1.0 : -1.0);
}
LAPLACE_HD_FORCEINLINE double affine_dom_lo(double p1, double p2) { return affine_valid(p1, p2) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double affine_dom_hi(double p1, double p2) { return affine_valid(p1, p2) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double affine_img_lo(double p1, double p2) { return affine_valid(p1, p2) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double affine_img_hi(double p1, double p2) { return affine_valid(p1, p2) ? inf() : nan(); }
// both bounds are infinite, so both distances are +inf
LAPLACE_HD_FORCEINLINE double affine_inv_to_upper(double u, double p1, double p2) { return affine_valid(p1, p2) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double affine_inv_to_lower(double u, double p1, double p2) { return affine_valid(p1, p2) ? inf() : nan(); }

}  // namespace laplace::bijector::detail_scalar
