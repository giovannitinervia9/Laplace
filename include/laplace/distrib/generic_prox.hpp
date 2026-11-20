// The proximal operator in y of a continuous univariate distribution by the generic route
// (laplace::distrib; note/penalties.md section 5h).
//
//   prox_y_by_newton(d, v, t)   x* = argmin_y  -t log f(y) + (y - v)^2 / 2
//   distrib_prox_y(d, v, t)     the member prox_y when the distribution has it (a closed form),
//                               the generic route otherwise
//   has_distrib_prox_y<D>       whether either exists
//
// With l = log f, x* is the root of g(y) = y - v - t l'(y), whose derivative is g' = 1 - t l''.
// The weak-convexity constant rho = max(0, sup_y l''(y)) is declared per law by
// distrib_weak_convexity_reading<D> (weak_convexity.hpp); it is 0 for a log-concave law. When
// t rho < 1, g' >= c = 1 - t rho > 0, so g increases, the root is unique and lies at most
// |g(x)|/c from any point x: one evaluation at v brackets it, with no need of the mode. At
// t rho = 1 the bound is infinite and the bracket is found by doubling steps instead; for
// t rho > 1 the objective may have several local minima and the route returns NaN (no caller
// reaches it: the proximal methods keep t rho < 1).
//
// Inside the bracket, Newton's step on g is taken unless it leaves the bracket or fails to halve
// the step of two iterations before (the safeguard of rtsafe, Numerical Recipes 9.4), in which case
// the bracket is bisected, geometrically towards a finite end of the support when it is more than
// four times nearer to that end. It stops when the step or the bracket falls below 4 eps |y|.
//
// A finite end e of the support: with t rho <= 1 the objective is convex on the closed support, so
// when g has the sign of the interior at the point next to e (e itself may lie outside the domain of
// log f), the constrained minimum is e. A density unbounded at e (gamma with shape < 1) has
// rho = infinity, hence NaN here: the objective goes to -infinity at e and has no minimizer.
//
// Measured on 13 laws against a 200-bit minimization of the objective (note/penalties.md 5h):
// error within 2.1 ulp of the condition number of the problem, 2.4 to 10.5 evaluations of g on
// average, at most 52.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

#include <concepts>

namespace laplace::distrib {

/**
 * \ingroup distrib_generic
 * \brief The weak-convexity constant \f$\rho = \max(0, \sup_y \partial_y^2 \log f(y))\f$ of a continuous univariate law.
 *
 * Undefined for a general `D`; specialized in `weak_convexity.hpp` with
 * `LAPLACE_HD static double value(const D& d)`, which returns \f$\rho\f$ at the
 * parameters held by `d` (0 for a log-concave law, \f$+\infty\f$ when
 * \f$\partial_y^2 \log f\f$ is unbounded above).
 *
 * \tparam D Continuous univariate distribution.
 */
template <class D>
struct distrib_weak_convexity_reading;

/// \ingroup distrib_generic
/// \brief Whether `D` declares its weak-convexity constant.
template <class D>
concept has_weak_convexity_reading = requires(const D d) {
  { distrib_weak_convexity_reading<D>::value(d) } -> std::convertible_to<double>;
};

/// \ingroup distrib_generic
/// \brief Whether `D` has the optional member `prox_y(v, t)` (a closed form).
template <class D>
concept has_prox_y_member = requires(const D d, double a) { d.prox_y(a, a); };

namespace detail_prox {
template <class D>
concept generic_prox_applies = D::kind == distrib_kind::continuous && D::y_regularity >= 2 &&
                               has_weak_convexity_reading<D> &&
                               requires(const D d, double a) { d.template logpdf_dy_dk<2, 0>(a); };
}  // namespace detail_prox

/// \ingroup distrib_generic
/// \brief Whether `distrib_prox_y` exists for `D`: a member `prox_y`, or the generic route (continuous, \f$\log f\f$ at least \f$C^2\f$ in \f$y\f$, \f$\rho\f$ declared).
template <class D>
concept has_distrib_prox_y = has_prox_y_member<D> || detail_prox::generic_prox_applies<D>;

/**
 * \ingroup distrib_generic
 * \brief The weak-convexity constant \f$\rho\f$ of `d`, from `distrib_weak_convexity_reading`.
 */
template <class D>
  requires has_weak_convexity_reading<D>
LAPLACE_HD double distrib_weak_convexity(const D& d) {
  return distrib_weak_convexity_reading<D>::value(d);
}

/**
 * \ingroup distrib_generic
 * \brief \f$\arg\min_y\, -t \log f(y) + (y - v)^2/2\f$ by safeguarded Newton on \f$y - v - t\,\partial_y \log f(y) = 0\f$.
 *
 * \param d The distribution, at its parameters.
 * \param v The point, any real.
 * \param t The step, \f$t \ge 0\f$.
 * \return The minimizer, which lies in the closed support; \f$v\f$ projected on the support at \f$t = 0\f$;
 *   NaN when \f$t \rho > 1\f$, when \f$t < 0\f$ or when an argument is NaN.
 */
template <class D>
  requires detail_prox::generic_prox_applies<D>
LAPLACE_HD double prox_y_by_newton(const D& d, double v, double t) {
  constexpr double eps4 = 4.0 * 0x1p-53;
  const double nan = math::detail::quiet_nan();
  const double lo_e = d.support_lower(), hi_e = d.support_upper();
  const bool lo_fin = lo_e > -distrib_infinity, hi_fin = hi_e < distrib_infinity;
  if (!(t >= 0.0) || v != v) return nan;
  if (t == 0.0) return v < lo_e ? lo_e : (v > hi_e ? hi_e : v);
  const double c = 1.0 - t * distrib_weak_convexity(d);
  if (!(c >= 0.0)) return nan;  // t rho > 1, or rho NaN
  auto g = [&](double y) { return (y - v) - t * d.template logpdf_dy_dk<1, 0>(y)[0]; };
  // the points next to the ends, inside the support, at a distance of at least the least normal number:
  // next to 0 a subnormal y makes terms such as (k - 1)/y overflow into inf - inf
  constexpr double tiny = 0x1p-1022;
  const double in_lo = lo_fin ? Kokkos::fmax(Kokkos::nextafter(lo_e, distrib_infinity), lo_e + tiny) : lo_e;
  const double in_hi = hi_fin ? Kokkos::fmin(Kokkos::nextafter(hi_e, -distrib_infinity), hi_e - tiny) : hi_e;
  double x = v <= lo_e ? in_lo : (v >= hi_e ? in_hi : v);
  double gx = g(x);
  if (gx == 0.0) return x;
  if (gx != gx) return nan;
  const bool right = gx < 0.0;  // the root lies above x
  double lo, hi;
  {
    const double far = x - gx / c;  // the bound |x* - x| <= |g(x)|/c; infinite at c = 0 or g(x) infinite
    if (right) {
      lo = x;
      if (far < in_hi) hi = far;
      else if (hi_fin) {  // the end: the constrained minimum when g does not change sign before it
        if (g(in_hi) <= 0.0) return hi_e;
        hi = in_hi;
      } else {  // no finite bound: doubling steps
        double s = Kokkos::fmax(Kokkos::fabs(x), d.scale());
        hi = distrib_infinity;
        for (int it = 0; it < 2100 && hi == distrib_infinity; ++it, s *= 2.0) {
          const double y = lo + s;
          if (!(y < distrib_infinity)) return nan;
          if (g(y) >= 0.0) hi = y;
          else lo = y;
        }
      }
    } else {
      hi = x;
      if (far > in_lo) lo = far;
      else if (lo_fin) {
        if (g(in_lo) >= 0.0) return lo_e;
        lo = in_lo;
      } else {
        double s = Kokkos::fmax(Kokkos::fabs(x), d.scale());
        lo = -distrib_infinity;
        for (int it = 0; it < 2100 && lo == -distrib_infinity; ++it, s *= 2.0) {
          const double y = hi - s;
          if (!(y > -distrib_infinity)) return nan;
          if (g(y) <= 0.0) lo = y;
          else hi = y;
        }
      }
    }
  }
  if (!(lo <= x && x <= hi)) {  // the doubling moved the bracket away from x
    x = 0.5 * (lo + hi);
    gx = g(x);
  }
  // Newton inside the bracket, bisection when it leaves it or slows down (rtsafe)
  double dx = distrib_infinity, dxold = distrib_infinity;
  for (int it = 0; it < 200; ++it) {
    if (it > 0) gx = g(x);
    if (gx == 0.0) return x;
    if (gx != gx) return nan;
    if (gx > 0.0) hi = x;
    else lo = x;
    const double dg = 1.0 - t * d.template logpdf_dy_dk<2, 0>(x)[0];
    // an infinite g' (l'' overflowing next to an end) would give a zero step and stop the loop: bisect instead
    double nx = dg > 0.0 && dg < distrib_infinity ? x - gx / dg : nan;
    bool bisect = false;
    if (Kokkos::fabs(nx - x) <= eps4 * Kokkos::fabs(x)) {
      // a small step is accepted only when g changes sign within one more such step: next to a singular
      // end (beta at 1) Newton crawls by steps of an ulp far from the root
      const double w = nx + (gx > 0.0 ? -1.0 : 1.0) * Kokkos::fmax(Kokkos::fabs(nx - x), Kokkos::fmax(0x1p-53 * Kokkos::fabs(nx), tiny));
      if (!(lo <= w && w <= hi)) return nx;  // beyond the bracket: the root is within the step
      const double gw = g(w);
      if (gw == 0.0) return w;
      if ((gw > 0.0) != (gx > 0.0)) return nx;
      if (gw > 0.0) hi = w;
      else lo = w;
      bisect = true;
    }
    if (bisect || !(lo <= nx && nx <= hi) || nx == x || Kokkos::fabs(nx - x) > 0.5 * Kokkos::fabs(dxold)) {
      if (lo_fin && hi - lo_e > 4.0 * (lo - lo_e)) nx = lo_e + Kokkos::sqrt((lo - lo_e) * (hi - lo_e));
      else if (hi_fin && hi_e - lo > 4.0 * (hi_e - hi)) nx = hi_e - Kokkos::sqrt((hi_e - lo) * (hi_e - hi));
      else nx = 0.5 * (lo + hi);
      if (!(lo <= nx && nx <= hi) || nx == x) return x;  // the bracket has closed
    }
    dxold = dx;
    dx = nx - x;
    if (hi - lo <= eps4 * Kokkos::fmax(Kokkos::fabs(lo), Kokkos::fabs(hi))) return nx;
    x = nx;
  }
  return x;
}

/**
 * \ingroup distrib_generic
 * \brief The proximal operator in \f$y\f$: the member `prox_y` when `D` has it, `prox_y_by_newton` otherwise.
 */
template <class D>
  requires has_distrib_prox_y<D>
LAPLACE_HD double distrib_prox_y(const D& d, double v, double t) {
  if constexpr (has_prox_y_member<D>) return d.prox_y(v, t);
  else return prox_y_by_newton(d, v, t);
}

}  // namespace laplace::distrib
