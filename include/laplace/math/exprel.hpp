// exprel(x) = (e^x - 1)/x, the function that GSL and SciPy call exprel, and its
// derivatives phi^(k)(x), k = 1..6 (laplace::math):
//
//   exprel(x)       phi(x) = (e^x - 1)/x, with phi(0) = 1
//   exprel_dK(x)    phi^(K)(x) = int_0^1 t^K e^(x t) dt, K = 1..6, with phi^(K)(0) = 1/(K+1)
//
// phi^(K) is strictly positive for every real x, so the relative error in units
// of the last place is the right measure of accuracy.
//
// Methods, for x != 0. The value phi(x) is Kokkos::expm1(x)/x. The derivative
// phi^(K)(x) is one of three forms, chosen by x and K (the thresholds below):
//
//  - for 0 < x < thr_pos(K), the power series
//      phi^(K)(x) = sum_(n >= 0) x^n / (n! (n + K + 1)),
//    whose terms are all positive for x > 0, so it has no cancellation;
//  - for -thr_neg(K) < x < 0, Kummer's transformation (substitute t = 1 - s),
//      phi^(K)(x) = e^x sum_(n >= 0) (-x)^n K! / (n + K + 1)!,
//    whose terms are all positive for x < 0. The power series itself cancels
//    badly for x < 0 (its terms alternate and their sum is smaller than the sum
//    of their magnitudes by a factor e^|x| |x|^K / K!), so it is not used there;
//  - otherwise, the forward recursion from phi^(0) = phi(x),
//      phi^(j)(x) = (e^x - j phi^(j-1)(x)) / x,  j = 1..K,
//    which loses accuracy as x -> 0 (the subtraction e^x - j phi^(j-1) cancels
//    to a result of order x), and is accurate away from 0 (measured in
//    docs/numerics.md). For x >= 700 the recursion works with the scaled quantities
//    phi^(j) e^(-700) (e^(x - 700) does not overflow, and x - 700 is exact by
//    Sterbenz's lemma), and the product with e^700 at the end overflows only
//    when the exact result does not fit a double (from x about 716.5 on).
//
// The thresholds, and the measurements behind them (docs/numerics.md), are
// in thr_pos(K) and thr_neg(K) below.
#pragma once

#include <Kokkos_Core.hpp>
#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::math {

namespace detail_exprel {

constexpr double eps = 2.220446049250313e-16;
constexpr double e700 = 0x1.d945df4f8ec8ep+1009;  // exp(700), correctly rounded
constexpr double x_scaled = 700.0;                 // from here the recursion is scaled by e^-700
constexpr double x_overflow = 740.0;               // phi^(K)(x) > DBL_MAX for every K from about 716.5 on

// The thresholds per order K = 1..6 (index K; index 0 is unused): the power
// series for 0 < x < thr_pos(K), Kummer's form for -thr_neg(K) < x < 0, the
// recursion elsewhere. Chosen from the measured maximum errors per binary
// octave of |x| (docs/numerics.md): the recursion is below about 4 ulp
// from these points on, the series is below about 4.5 ulp up to thr_pos, and
// Kummer's form below about 6 ulp up to thr_neg. The power series is not used
// for x < 0, where it cancels (for orders 1, 3 and 6 its error is 35 to 143 ulp
// for |x| in [2, 4) and 3e3 to 5e4 ulp for |x| in [4, 8)).
LAPLACE_HD_FORCEINLINE double thr_pos(int k) {
  constexpr double t[7] = {0.0, 1.0, 2.0, 2.0, 4.0, 4.0, 4.0};
  return t[k];
}
LAPLACE_HD_FORCEINLINE double thr_neg(int k) {
  constexpr double t[7] = {0.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0};
  return t[k];
}

// The power series sum_n x^n / (n! (n + k + 1)), summed until a term is below
// 4 eps of the sum (iteration cap 200).
LAPLACE_HD double series(int k, double x) {
  double t = 1.0;  // x^n / n!
  double s = 1.0 / (k + 1);
  for (int n = 1; n < 200; ++n) {
    t *= x / n;
    const double term = t / (n + k + 1);
    s += term;
    if (Kokkos::fabs(term) <= 4.0 * eps * Kokkos::fabs(s)) break;
  }
  return s;
}

// Kummer's form e^x sum_n (-x)^n k! / (n + k + 1)!, for x < 0, summed with
// the same stopping test and cap; the terms are positive.
LAPLACE_HD double kummer(int k, double x) {
  const double y = -x;
  double t = 1.0 / (k + 1);  // k! / (k + 1)!
  double s = t;
  for (int n = 1; n < 200; ++n) {
    t *= y / (n + k + 1);
    s += t;
    if (t <= 4.0 * eps * s) break;
  }
  return math::exp(x) * s;
}

// The forward recursion from phi(x), for x != 0 (order k = 0 is phi itself).
LAPLACE_HD double recursion(int k, double x) {
  if (x >= x_scaled) {
    if (x > x_overflow) return detail::pos_inf();
    const double E = math::exp(x - x_scaled);  // e^(x - 700), finite
    double p = E / x;                          // phi(x) e^-700; the e^-700 term of e^x - 1 is negligible
    for (int j = 1; j <= k; ++j) p = (E - j * p) / x;
    return p * e700;
  }
  const double E = math::exp(x);
  double p = Kokkos::expm1(x) / x;
  for (int j = 1; j <= k; ++j) p = (E - j * p) / x;
  return p;
}

// phi^(k)(x) for one order k, with the method chosen by x (see the header)
LAPLACE_HD double deriv(int k, double x) {
  if (x != x) return x;
  if (x == 0.0) return 1.0 / (k + 1);
  if (x > 0.0 && x < thr_pos(k)) return series(k, x);
  if (x < 0.0 && x > -thr_neg(k)) return kummer(k, x);
  return recursion(k, x);
}

}  // namespace detail_exprel

// phi(x) = (e^x - 1)/x, phi(0) = 1
/**
 * \ingroup exprel
 * \brief Relative exponential function, \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the relative exponential function:
 * \f[
 * \varphi(x) = \frac{e^x - 1}{x}
 * \f]
 * with \f$\varphi(0) = 1\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$1.0\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 1.81 ulp (gcc) / 1.52 ulp (CUDA); test bound 2.8 ulp.
 *
 * \see exprel, exprel_d1()
 */
LAPLACE_HD double exprel(double x) {
  if (x != x) return x;
  if (x == 0.0) return 1.0;
  if (x >= detail_exprel::x_scaled) return detail_exprel::recursion(0, x);
  return Kokkos::expm1(x) / x;
}

// phi^(1)(x) .. phi^(6)(x), one function per order
/**
 * \ingroup exprel
 * \brief First derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the first derivative:
 * \f[
 * \varphi'(x) = \frac{\mathrm{d}}{\mathrm{d}x} \varphi(x) = \int_0^1 t e^{x t} \, \mathrm{d}t = \frac{e^x - \varphi(x)}{x}
 * \f]
 * with \f$\varphi'(0) = 1/2\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi'(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$0.5\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.71 ulp (gcc) / 2.64 ulp (CUDA); test bound 4.1 ulp.
 *
 * \see exprel, exprel(), exprel_d2()
 */
LAPLACE_HD double exprel_d1(double x) { return detail_exprel::deriv(1, x); }

/**
 * \ingroup exprel
 * \brief Second derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the second derivative:
 * \f[
 * \varphi''(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \varphi(x) = \int_0^1 t^2 e^{x t} \, \mathrm{d}t = \frac{e^x - 2 \varphi'(x)}{x}
 * \f]
 * with \f$\varphi''(0) = 1/3\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi''(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$1/3\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 4.11 ulp (gcc) / 4.11 ulp (CUDA); test bound 6.2 ulp.
 *
 * \see exprel, exprel(), exprel_d1(), exprel_d3()
 */
LAPLACE_HD double exprel_d2(double x) { return detail_exprel::deriv(2, x); }

/**
 * \ingroup exprel
 * \brief Third derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the third derivative:
 * \f[
 * \varphi^{(3)}(x) = \frac{\mathrm{d}^3}{\mathrm{d}x^3} \varphi(x) = \int_0^1 t^3 e^{x t} \, \mathrm{d}t = \frac{e^x - 3 \varphi''(x)}{x}
 * \f]
 * with \f$\varphi^{(3)}(0) = 1/4\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi^{(3)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$0.25\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.90 ulp (gcc) / 5.90 ulp (CUDA); test bound 8.9 ulp.
 *
 * \see exprel, exprel(), exprel_d2(), exprel_d4()
 */
LAPLACE_HD double exprel_d3(double x) { return detail_exprel::deriv(3, x); }

/**
 * \ingroup exprel
 * \brief Fourth derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * \varphi^{(4)}(x) = \frac{\mathrm{d}^4}{\mathrm{d}x^4} \varphi(x) = \int_0^1 t^4 e^{x t} \, \mathrm{d}t = \frac{e^x - 4 \varphi^{(3)}(x)}{x}
 * \f]
 * with \f$\varphi^{(4)}(0) = 1/5\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi^{(4)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$0.2\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 4.25 ulp (gcc) / 4.25 ulp (CUDA); test bound 6.4 ulp.
 *
 * \see exprel, exprel(), exprel_d3(), exprel_d5()
 */
LAPLACE_HD double exprel_d4(double x) { return detail_exprel::deriv(4, x); }

/**
 * \ingroup exprel
 * \brief Fifth derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * \varphi^{(5)}(x) = \frac{\mathrm{d}^5}{\mathrm{d}x^5} \varphi(x) = \int_0^1 t^5 e^{x t} \, \mathrm{d}t = \frac{e^x - 5 \varphi^{(4)}(x)}{x}
 * \f]
 * with \f$\varphi^{(5)}(0) = 1/6\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi^{(5)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$1/6\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.85 ulp (gcc) / 5.85 ulp (CUDA); test bound 8.8 ulp.
 *
 * \see exprel, exprel(), exprel_d4(), exprel_d6()
 */
LAPLACE_HD double exprel_d5(double x) { return detail_exprel::deriv(5, x); }

/**
 * \ingroup exprel
 * \brief Sixth derivative in \f$x\f$ of \f$\varphi(x) = (e^x - 1)/x\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * \varphi^{(6)}(x) = \frac{\mathrm{d}^6}{\mathrm{d}x^6} \varphi(x) = \int_0^1 t^6 e^{x t} \, \mathrm{d}t = \frac{e^x - 6 \varphi^{(5)}(x)}{x}
 * \f]
 * with \f$\varphi^{(6)}(0) = 1/7\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\varphi^{(6)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - \f$x = 0\f$: returns \f$1/7\f$.
 * - \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$ or \f$x > 740.0\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.92 ulp (gcc) / 6.57 ulp (CUDA); test bound 8.9 ulp.
 *
 * \see exprel, exprel(), exprel_d5()
 */
LAPLACE_HD double exprel_d6(double x) { return detail_exprel::deriv(6, x); }

}  // namespace laplace::math
