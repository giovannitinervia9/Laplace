// Regularized incomplete beta function I_x(a, b), its complement
// 1 - I_x(a, b), and their logarithms, Laplace's own (a, b > 0, 0 <= x <= 1).
//
// Why own: in double precision Boost.Math loses all the relative accuracy of
// the far tail, because of its prefactor x^a (1 - x)^b / B(a, b)
// (docs/numerics.md). Here the prefactor is written without cancellation and
// the series, expansions and continued fraction of DiDonato and Morris (1992,
// ACM TOMS 18, 360-373) are chosen as in their BRATIO; they are in
// detail/ibeta_expansions.hpp, with the choice (direct_side). The side that is
// computed directly, I_x(a, b) or 1 - I_x(a, b), is never close to 1; the other
// is 1 minus it, and its logarithm log1p of minus it. Every piece works in
// logarithms, so that log I and log(1 - I) keep their accuracy where I or
// 1 - I underflows.
//
// History: the first own version (2026-11-09) used the prefactor and one
// continued fraction (Numerical Recipes' betacf) everywhere; with one shape
// small and the other large and x near the mean it needed many terms and its
// absolute error reached 216 ulp of 1 (1059 on the hard references). Measured
// in docs/numerics.md.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/ibeta_expansions.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::math {

namespace own {

struct beta_inc_result {
  double i, ic;          // I_x(a, b), 1 - I_x(a, b)
  double log_i, log_ic;  // their logarithms
};

LAPLACE_HD beta_inc_result beta_inc(double a, double b, double x) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(x <= 1.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == 1.0) return {1.0, 0.0, 0.0, -inf};
  const detail_ibx::side r = detail_ibx::direct_side(a, b, x);
  const double w = math::exp(r.log_w);
  const double other = 1.0 - w, log_other = Kokkos::log1p(-w);
  if (!r.complement) return {w, other, r.log_w, log_other};
  return {other, w, log_other, r.log_w};
}

}  // namespace own

// What the rest of Laplace calls: the own versions (docs/numerics.md).
LAPLACE_HD_FORCEINLINE double ibeta(double a, double b, double x) { return own::beta_inc(a, b, x).i; }
LAPLACE_HD_FORCEINLINE double ibetac(double a, double b, double x) { return own::beta_inc(a, b, x).ic; }
LAPLACE_HD_FORCEINLINE double log_ibeta(double a, double b, double x) { return own::beta_inc(a, b, x).log_i; }
LAPLACE_HD_FORCEINLINE double log_ibetac(double a, double b, double x) { return own::beta_inc(a, b, x).log_ic; }

}  // namespace laplace::math
