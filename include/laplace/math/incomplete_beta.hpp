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

/**
 * \ingroup incomplete_beta
 * \brief Result of \ref beta_inc() holding linear values, logarithms, and log low parts.
 *
 * Fields: `i` holds \f$I_x(a, b)\f$, `ic` holds \f$1 - I_x(a, b)\f$, `log_i` holds
 * \f$\log I_x(a, b)\f$, `log_ic` holds \f$\log(1 - I_x(a, b))\f$, and `log_i_lo` and
 * `log_ic_lo` hold the double-double low-order correction of the directly computed side.
 *
 * \see incomplete_beta, beta_inc()
 */
struct beta_inc_result {
  double i, ic;          // I_x(a, b), 1 - I_x(a, b)
  double log_i, log_ic;  // their logarithms
  // the low part of the logarithm of the side computed directly (the smaller
  // one), as a double-double; 0 for the other. The inverses use it far in a tail.
  double log_i_lo = 0.0, log_ic_lo = 0.0;
};

/**
 * \ingroup incomplete_beta
 * \brief Regularized incomplete beta function \f$I_x(a, b)\f$, complement, and logarithms in one call.
 *
 * Computes
 * \f[
 * I_x(a, b) = \frac{1}{B(a, b)} \int_0^x t^{a-1} (1 - t)^{b-1} \, \mathrm{d}t,
 * \f]
 * \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$, \f$\log I_x(a, b)\f$, \f$\log(1 - I_x(a, b))\f$, and
 * the double-double log low parts at the same cost as any single scalar call. Requires
 * \f$a, b \in (0, \infty)\f$ and \f$x \in [0, 1]\f$; invalid or NaN inputs yield quiet NaN
 * for the four primary fields and zero for the low parts (see \ref incomplete_beta).
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1]\f$.
 * \return A \ref beta_inc_result holding all four values and log low parts.
 *
 * \par Accuracy
 * Matches the scalar functions (\ref laplace::math::ibeta "ibeta()", \ref laplace::math::ibetac "ibetac()",
 * \ref laplace::math::log_ibeta "log_ibeta()", \ref laplace::math::log_ibetac "log_ibetac()").
 *
 * \see incomplete_beta, beta_inc_result, \ref laplace::math::ibeta "ibeta()", \ref laplace::math::ibetac "ibetac()"
 */
LAPLACE_HD beta_inc_result beta_inc(double a, double b, double x) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(b > 0.0) || !(x >= 0.0) || !(x <= 1.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == 1.0) return {1.0, 0.0, 0.0, -inf};
  const detail_ibx::side r = detail_ibx::direct_side(a, b, x);
  double w = math::exp(r.log_w);
  w += w * r.log_w_lo;
  const double other = 1.0 - w, log_other = Kokkos::log1p(-w);
  if (!r.complement) return {w, other, r.log_w, log_other, r.log_w_lo, 0.0};
  return {other, w, log_other, r.log_w, 0.0, r.log_w_lo};
}

}  // namespace own

// What the rest of Laplace calls: the own versions (docs/numerics.md).
/**
 * \ingroup incomplete_beta
 * \brief Regularized incomplete beta function \f$I_x(a, b)\f$.
 *
 * Returns the `i` field of `own::beta_inc(a, b, x)`. Returns \f$0.0\f$ at \f$x = 0\f$ and
 * \f$1.0\f$ at \f$x = 1\f$; see \ref incomplete_beta for shared domain rules.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1]\f$.
 * \return The value \f$I_x(a, b) \in [0, 1]\f$.
 *
 * \par Accuracy
 * Max error 2.82 ulp1 / 12.7 ulpL (gcc) and 3.80 ulp1 / 12.9 ulpL (CUDA), bound 5.7 ulp1 / 19.5 ulpL; hard regions max 3.65 ulp1 / 18.8 ulpL, bound 5.5 ulp1 / 29.5 ulpL.
 *
 * \see incomplete_beta, own::beta_inc(), ibetac(), log_ibeta(), log_ibetac()
 */
LAPLACE_HD_FORCEINLINE double ibeta(double a, double b, double x) { return own::beta_inc(a, b, x).i; }

/**
 * \ingroup incomplete_beta
 * \brief Complement of the regularized incomplete beta function \f$1 - I_x(a, b)\f$.
 *
 * Computes \f$1 - I_x(a, b) = I_{1-x}(b, a)\f$ directly from `own::beta_inc(a, b, x).ic`
 * without subtracting near unity. Returns \f$1.0\f$ at \f$x = 0\f$ and \f$0.0\f$ at \f$x = 1\f$.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1]\f$.
 * \return The value \f$1 - I_x(a, b) \in [0, 1]\f$.
 *
 * \par Accuracy
 * Max error 2.82 ulp1 / 16.9 ulpL (gcc) and 3.80 ulp1 / 12.2 ulpL (CUDA), bound 5.7 ulp1 / 25.5 ulpL; hard regions max 3.65 ulp1 / 18.4 ulpL, bound 5.5 ulp1 / 29.5 ulpL.
 *
 * \see incomplete_beta, own::beta_inc(), ibeta(), log_ibeta(), log_ibetac()
 */
LAPLACE_HD_FORCEINLINE double ibetac(double a, double b, double x) { return own::beta_inc(a, b, x).ic; }

/**
 * \ingroup incomplete_beta
 * \brief Natural logarithm of the regularized incomplete beta function \f$\log I_x(a, b)\f$.
 *
 * Evaluated on log scale without intermediate underflow via `own::beta_inc(a, b, x).log_i`.
 * Returns \f$-\infty\f$ at \f$x = 0\f$ and \f$0.0\f$ at \f$x = 1\f$.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1]\f$.
 * \return The value \f$\log I_x(a, b) \in [-\infty, 0]\f$.
 *
 * \par Accuracy
 * Max error 17 ulpL (gcc) and 12 ulpL (CUDA) on general points, 22 ulpL (gcc) and 19.7 ulpL (CUDA) on hard regions; test bound 33.0 ulpL.
 *
 * \see incomplete_beta, own::beta_inc(), ibeta(), ibetac(), log_ibetac()
 */
LAPLACE_HD_FORCEINLINE double log_ibeta(double a, double b, double x) { return own::beta_inc(a, b, x).log_i; }

/**
 * \ingroup incomplete_beta
 * \brief Natural logarithm of the complement \f$\log(1 - I_x(a, b))\f$.
 *
 * Computes \f$\log(1 - I_x(a, b)) = \log I_{1-x}(b, a)\f$ directly on log scale via
 * `own::beta_inc(a, b, x).log_ic`. Returns \f$0.0\f$ at \f$x = 0\f$ and \f$-\infty\f$ at \f$x = 1\f$.
 *
 * \param a First positive shape parameter \f$a \in (0, \infty)\f$.
 * \param b Second positive shape parameter \f$b \in (0, \infty)\f$.
 * \param x Argument \f$x \in [0, 1]\f$.
 * \return The value \f$\log(1 - I_x(a, b)) \in [-\infty, 0]\f$.
 *
 * \par Accuracy
 * Max error 10 ulpL (gcc) and 15 ulpL (CUDA) on general points, 15 ulpL (gcc) and 16 ulpL (CUDA) on hard regions; test bound 33.0 ulpL.
 *
 * \see incomplete_beta, own::beta_inc(), ibeta(), ibetac(), log_ibeta()
 */
LAPLACE_HD_FORCEINLINE double log_ibetac(double a, double b, double x) { return own::beta_inc(a, b, x).log_ic; }

}  // namespace laplace::math
