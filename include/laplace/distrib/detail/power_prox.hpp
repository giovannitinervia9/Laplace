// The proximal operator of a|x|^p (note/penalties.md section 5g), shared by
// powerexp_distrib (a = t s^-p) and the isotropic mvpowerexp1/2 (on the norm,
// p = 2 beta):
//
//   x*(u) = argmin_x  a |x|^p + (x - u)^2 / 2,   a >= 0, p > 0.
//
// x* has the sign of u and |x*| <= |u|. With U = |u|, p = 1 is the soft
// threshold (U - a)_+ and p = 2 is U / (1 + 2a). Otherwise a nonzero |x*| is a
// root of phi(x) = x + a p x^(p-1) - U, found by Newton from a bound on the
// side where the iteration is monotone, and stopped when it stops being so
// (the step no longer moves x in its direction), which is rounding:
//
//   1 < p < 2  phi concave and increasing; from the lower bound
//              (U / (U^(2-p) + a p))^(1/(p-1)), since x^(p-1) = U/(x^(2-p) + a p)
//              and x <= U;
//   p > 2      phi convex and increasing; from the upper bound
//              min(U, (U / (a p))^(1/(p-1)));
//   p < 1      nonconvex objective (the bridge): x* = 0 when U <= tau, with
//              x_b = (2a(1-p))^(1/(2-p)) and tau = x_b + a p x_b^(p-1) (at
//              U = tau the objective has two minimizers, 0 and x_b, and 0 is
//              taken); beyond, the larger root of phi, on the branch x > x_b
//              where phi is convex and increasing, from U.
//
// Against a 200-bit mpmath bisection (400 random (u, a) per p, 1e-8..1e8):
// within 1 ulp for p <= 0.75 and p >= 2.5, 18 ulp at p = 1.05, where the
// problem's own condition is 1/(p - 1); at most 12 Newton steps.
#pragma once

#include <laplace/core/config.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib::detail_power_prox {

// argmin_x a|x|^p + (x - u)^2/2
LAPLACE_HD double power_prox(double u, double a, double p) {
  const double U = Kokkos::fabs(u), s = u < 0.0 ? -1.0 : 1.0;
  if (!(U > 0.0)) return 0.0 * u;
  if (p == 1.0) return U <= a ? 0.0 : s * (U - a);
  if (p == 2.0) return s * U / (1.0 + 2.0 * a);
  const bool concave = p > 1.0 && p < 2.0;
  double x;
  if (p < 1.0) {
    const double xb = Kokkos::pow(2.0 * a * (1.0 - p), 1.0 / (2.0 - p));
    if (U <= xb + a * p * Kokkos::pow(xb, p - 1.0)) return 0.0;
    x = U;
  } else if (concave) {
    x = Kokkos::pow(U / (Kokkos::pow(U, 2.0 - p) + a * p), 1.0 / (p - 1.0));
    if (!(x > 0.0)) return 0.0;  // the root underflows
  } else {
    x = Kokkos::fmin(U, Kokkos::pow(U / (a * p), 1.0 / (p - 1.0)));
  }
  // ponytail: cap of 100 steps, never reached in the measurements (12 at most)
  for (int it = 0; it < 100; ++it) {
    const double xp2 = Kokkos::pow(x, p - 2.0);
    const double xn = x - (x + a * p * xp2 * x - U) / (1.0 + a * p * (p - 1.0) * xp2);
    if (concave ? !(xn > x) : !(xn < x)) break;
    x = xn;
  }
  return s * x;
}

// The prox of (1/2) (q ||y - x0||^2)^beta in R^d: with u = v - x0 and r = ||u||, the minimizer
// lies along u (for a fixed norm the quadratic term is least there) with norm
// c power_prox(r/c, t q/2, 2 beta), c = q^(-1/2) (the scalar problem in units of c, so that
// q^beta never overflows); out = x0 at r = 0. beta = 1/2 is the block soft threshold
// (1 - t sqrt(q)/(2r))_+ u.
LAPLACE_HD void isotropic_power_prox(int d, double q, double beta, const double* v, const double* x0, double t,
                                     double* out) {
  double r2 = 0.0;
  for (int i = 0; i < d; ++i) r2 += (v[i] - x0[i]) * (v[i] - x0[i]);
  const double r = Kokkos::sqrt(r2), c = 1.0 / Kokkos::sqrt(q);
  const double f = r == 0.0 ? 0.0 : c * power_prox(r / c, 0.5 * t * q, 2.0 * beta) / r;  // NaN in, NaN out
  for (int i = 0; i < d; ++i) out[i] = x0[i] + f * (v[i] - x0[i]);
}

}  // namespace laplace::distrib::detail_power_prox
