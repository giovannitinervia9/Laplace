// Scalar bijector power (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   power: (0, inf) -> (0, inf), g(v) = v^p1, p1 != 0; no derivatives in p1; p1 = exponent, p2 unused
//
// Written 2026-11-12. The forms, for lambda = p1 and mu = 1/lambda:
//
//   forward   g^(K)(v) = F_K v^(lambda - K), F_K = prod_(j<K) (lambda - j)
//   inverse   h^(K)(u) = G_K u^(mu - K),     G_K = prod_(j<K) (mu - j) = prod_(j<K) (1 - j lambda)/lambda
//   fwd lj    log|g'(v)| = log|lambda| + (lambda - 1) log v, derivatives (lambda - 1) log_dk
//   inv lj    log|h'(u)| = -log|lambda| + (mu - 1) log u, mu - 1 = (1 - lambda)/lambda
//
// Cancellation and range, and how each is avoided:
//  - v^(lambda - K) and u^(mu - K) are formed as pow(v, s) (1 + e log v), where
//    s + e is the exponent written as a sum of two doubles without rounding loss
//    (two_sum for lambda - K; for mu - K, mu = mu0 + (1 - mu0 lambda)/lambda with
//    mu0 = fl(1/lambda) and the residual 1 - mu0 lambda by fma). The product
//    e log v is at most 1e-12 for |log v| <= 708, so the term 1 + e log v is
//    exact to the last place. Without it the exponent rounding gives a relative
//    error of about |log v| eps.
//  - the falling factorials use fma(-j, lambda, 1) = 1 - j lambda, exact to one
//    rounding, so that mu - j near zero (lambda near 1/j) keeps its relative accuracy.
//  - the powers of the argument are never formed as v^lambda times a power of v,
//    which overflows where the result does not (lambda = 7, K = 6, v = 1e300).
//  - no derivatives in lambda (they grow without bound as lambda -> 0): the dispatch
//    fills the other components of the partials with NaN.
//
// Invalid parameters (lambda = 0, NaN, infinite): every function returns NaN.
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>

namespace laplace::bijector::detail_scalar {

LAPLACE_HD_FORCEINLINE bool power_valid(double lambda) { return lambda != 0.0 && Kokkos::isfinite(lambda); }

// the sum a + b as s + e exactly (Knuth's two-sum; no multiplication, so no FMA contraction)
struct power_sum {
  double s, e;
};
LAPLACE_HD_FORCEINLINE power_sum power_two_sum(double a, double b) {
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

// 1 + e log x for an exponent offset e, with the term skipped where it is zero
// (log 0 is infinite and 0 times infinity is not wanted at the bounds)
LAPLACE_HD_FORCEINLINE double power_offset(double e, double x) {
  if (e == 0.0 || x == 0.0) return 1.0;
  return 1.0 + e * m::log(x);
}

// F x^(s + e) for an exponent s + e, the offset e being small. The power x^s is
// formed directly when it is normal; otherwise (it overflows or underflows while
// F x^s does not, e.g. F = -0.21 at x = 2.6e-182, s = -1.7) the exponent is halved
// and the two factors are multiplied with F between them.
LAPLACE_HD_FORCEINLINE double power_scaled(double F, double x, double s, double e) {
  const double w = pow_pos(x, s);
  const double aw = Kokkos::fabs(w);
  if (aw < 1e300 && aw > 1e-290) return F * (w * power_offset(e, x));
  const double wh = pow_pos(x, 0.5 * s);
  return (F * wh) * (wh * power_offset(e, x));
}

template <int K>
LAPLACE_HD double power_fwd(double v, double p1, double p2) {
  if (!power_valid(p1)) return nan();
  if constexpr (K == 0) return pow_pos(v, p1);
  else {
    double F = 1.0;
    for (int j = 0; j < K; ++j) F *= p1 - double(j);
    if (F == 0.0) return 0.0;  // lambda an integer below K: the derivative is exactly 0
    const power_sum ex = power_two_sum(p1, -double(K));  // lambda - K = ex.s + ex.e
    return power_scaled(F, v, ex.s, ex.e);
  }
}

template <int K>
LAPLACE_HD double power_inv(double u, double p1, double p2) {
  if (!power_valid(p1)) return nan();
  const double mu0 = 1.0 / p1;
  const double dmu = Kokkos::fma(-mu0, p1, 1.0) / p1;  // mu - mu0, mu = 1/lambda exactly
  if constexpr (K == 0) return pow_pos(u, mu0) * power_offset(dmu, u);
  else {
    double G = 1.0;
    for (int j = 0; j < K; ++j) G *= Kokkos::fma(-double(j), p1, 1.0) / p1;  // mu - j = (1 - j lambda)/lambda
    const power_sum ex = power_two_sum(mu0, -double(K));  // mu - K = ex.s + (ex.e + dmu)
    return power_scaled(G, u, ex.s, ex.e + dmu);
  }
}

template <int K>
LAPLACE_HD double power_fwd_lj(double v, double p1, double p2) {  // log|lambda| + (lambda - 1) log v
  if (!power_valid(p1)) return nan();
  const double c = p1 - 1.0;
  if constexpr (K == 0) return m::log(Kokkos::fabs(p1)) + c * m::log(v);
  else return log_dk<K>(c, v);
}

template <int K>
LAPLACE_HD double power_inv_lj(double u, double p1, double p2) {  // -log|lambda| + (mu - 1) log u
  if (!power_valid(p1)) return nan();
  const double c = (1.0 - p1) / p1;  // mu - 1
  if constexpr (K == 0) return -m::log(Kokkos::fabs(p1)) + c * m::log(u);
  else return log_dk<K>(c, u);
}

LAPLACE_HD_FORCEINLINE double power_dom_lo(double p1, double p2) { return power_valid(p1) ? 0.0 : nan(); }
LAPLACE_HD_FORCEINLINE double power_dom_hi(double p1, double p2) { return power_valid(p1) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double power_img_lo(double p1, double p2) { return power_valid(p1) ? 0.0 : nan(); }
LAPLACE_HD_FORCEINLINE double power_img_hi(double p1, double p2) { return power_valid(p1) ? inf() : nan(); }
// dom_hi - h(u) = inf - h(u), and h(u) - dom_lo = h(u): the second is the value itself
LAPLACE_HD_FORCEINLINE double power_inv_to_upper(double u, double p1, double p2) {
  return power_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double power_inv_to_lower(double u, double p1, double p2) {
  return power_inv<0>(u, p1, p2);
}

}  // namespace laplace::bijector::detail_scalar
