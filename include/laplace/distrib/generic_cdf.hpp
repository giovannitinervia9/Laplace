// The cdf family of a discrete distribution by summing its mass, and the
// dispatchers that take a distribution's own member when it has one
// (laplace::distrib; note/distrib.md section 4).
//
//   cdf_by_sum(d, y), ccdf_by_sum, log_cdf_by_sum, log_ccdf_by_sum
//   distrib_cdf(d, y), distrib_ccdf, distrib_log_cdf, distrib_log_ccdf
//
// With k = floor(y): below the centre the lower tail F(k) = sum_(i <= k) p(i) is
// summed downwards from k, and 1 - F(k) is its complement; from the centre up the
// upper tail 1 - F(k) = sum_(i > k) p(i) is summed upwards from k + 1, and F(k)
// is its complement. Each sum starts at its largest term, p(k) or p(k + 1), and
// adds e^(l(i) - l(start)), so that the logarithm of a tail far below the double
// range is still exact: log(tail) = l(start) + log(sum). A sum stops at the end of
// the support or when the geometric bound t r / (1 - r) on the rest (r the ratio
// of the last two terms, below 1) falls below 2^-60 of the sum: the terms
// decrease away from the mode, and the bound holds for log-concave masses.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_generic {

// log of the lower tail sum_(i <= k) (upper = false) or of the upper one sum_(i > k)
template <class D>
LAPLACE_HD double log_tail_by_sum(const D& d, double k, bool upper) {
  const double start = upper ? k + 1.0 : k, end = upper ? d.support_upper() : d.support_lower();
  const int step = upper ? 1 : -1;
  const double l0 = log_density(d, start);
  if (!(l0 > -distrib_infinity)) return l0;  // ponytail: a zero mass at the start means an empty tail (unimodal masses)
  double s = 0.0, prev = 0.0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double i = start + step * double(n);
    if (upper ? i > end : i < end) break;
    const double t = math::exp(log_density(d, i) - l0);
    s += t;
    if (n > 0) {
      const double r = t / prev;
      if (t == 0.0 || (r < 1.0 && t * r <= 0x1p-60 * s * (1.0 - r))) break;
    }
    prev = t;
  }
  return l0 + math::log(s);
}

// log F(k) or log(1 - F(k)) from the cheaper side
template <class D>
LAPLACE_HD double log_cdf_side_by_sum(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::discrete, "the cdf by summing the mass is for discrete distributions");
  const double k = Kokkos::floor(y);
  if (k < d.support_lower()) return upper ? 0.0 : -distrib_infinity;
  if (k >= d.support_upper()) return upper ? -distrib_infinity : 0.0;
  const bool lower_side = k < d.center();
  const double lt = log_tail_by_sum(d, k, !lower_side);
  if (lower_side != upper) return lt;  // the side asked for is the one summed
  return Kokkos::log1p(-math::exp(lt));
}

}  // namespace detail_generic

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of cumulative distribution function \f$\log F(y)\f$ for a discrete distribution by summation.
 *
 * For \f$k = \lfloor y \rfloor\f$, evaluates \f$\log \sum_{i \le k} p(i)\f$ by summing the lower tail downwards
 * when \f$k < \mathrm{center}()\f$, or from the upper tail complement \f$\log(1 - e^{\log(1 - F(k))})\f$
 * when \f$k \ge \mathrm{center}()\f$. Each tail factors out its dominant term to prevent underflow.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$-\infty\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * On the Poisson at \f$\mu\f$ = 0.05, 3.5, 40, 1000 (gcc), worst error 0.60, 2.64, 0.82, 5.91 ulp of \f$\max(|\log F|, 1)\f$ and \f$\max(|\log(1 - F)|, 1)\f$ over both functions.
 *
 * \see log_ccdf_by_sum(), cdf_by_sum(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double log_cdf_by_sum(const D& d, double y) { return detail_generic::log_cdf_side_by_sum(d, y, false); }

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of complementary cumulative distribution function \f$\log(1 - F(y))\f$ for a discrete distribution by summation.
 *
 * For \f$k = \lfloor y \rfloor\f$, evaluates \f$\log \sum_{i > k} p(i)\f$ by summing the upper tail upwards
 * when \f$k \ge \mathrm{center}()\f$, or from the lower tail complement \f$\log(1 - e^{\log F(k)})\f$
 * when \f$k < \mathrm{center}()\f$. Each tail factors out its dominant term to prevent underflow.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log(1 - F(y))\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$-\infty\f$.
 *
 * \par Measured accuracy
 * On the Poisson at \f$\mu\f$ = 0.05, 3.5, 40, 1000 (gcc), worst error 0.60, 2.64, 0.82, 5.91 ulp of \f$\max(|\log F|, 1)\f$ and \f$\max(|\log(1 - F)|, 1)\f$ over both functions.
 *
 * \see log_cdf_by_sum(), ccdf_by_sum(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double log_ccdf_by_sum(const D& d, double y) { return detail_generic::log_cdf_side_by_sum(d, y, true); }

/**
 * \ingroup distrib_generic
 * \brief Cumulative distribution function \f$F(y)\f$ for a discrete distribution by summation.
 *
 * Computes the exponential of log_cdf_by_sum().
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$0.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$1.0\f$.
 *
 * \par Measured accuracy
 * That of log_cdf_by_sum(), passed through the exponential.
 *
 * \see log_cdf_by_sum(), ccdf_by_sum(), distrib_cdf()
 */
template <class D>
LAPLACE_HD double cdf_by_sum(const D& d, double y) { return math::exp(log_cdf_by_sum(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Complementary cumulative distribution function \f$1 - F(y)\f$ for a discrete distribution by summation.
 *
 * Computes the exponential of log_ccdf_by_sum().
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$1 - F(y)\f$.
 *
 * \par Domain and special values
 * - If \f$\lfloor y \rfloor\f$ is below `d.support_lower()`, returns \f$1.0\f$.
 * - If \f$\lfloor y \rfloor\f$ is at least `d.support_upper()`, returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * That of log_ccdf_by_sum(), passed through the exponential.
 *
 * \see log_ccdf_by_sum(), cdf_by_sum(), distrib_ccdf()
 */
template <class D>
LAPLACE_HD double ccdf_by_sum(const D& d, double y) { return math::exp(log_ccdf_by_sum(d, y)); }

/**
 * \ingroup distrib_generic
 * \brief Cumulative distribution function \f$F(y)\f$, dispatching to specialized member or generic summation.
 *
 * Calls `d.cdf(y)` if provided by distribution `d`; otherwise calls cdf_by_sum().
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$F(y)\f$.
 *
 * \see cdf_by_sum(), distrib_ccdf(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double distrib_cdf(const D& d, double y) {
  if constexpr (requires { d.cdf(y); }) return d.cdf(y);
  else return cdf_by_sum(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Complementary cumulative distribution function \f$1 - F(y)\f$, dispatching to specialized member or generic summation.
 *
 * Calls `d.ccdf(y)` if provided by distribution `d`; otherwise calls ccdf_by_sum().
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$1 - F(y)\f$.
 *
 * \see ccdf_by_sum(), distrib_cdf(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double distrib_ccdf(const D& d, double y) {
  if constexpr (requires { d.ccdf(y); }) return d.ccdf(y);
  else return ccdf_by_sum(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of cumulative distribution function \f$\log F(y)\f$, dispatching to specialized member or generic summation.
 *
 * Calls `d.log_cdf(y)` if provided by distribution `d`; otherwise calls log_cdf_by_sum().
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log F(y)\f$.
 *
 * \see log_cdf_by_sum(), distrib_cdf(), distrib_log_ccdf()
 */
template <class D>
LAPLACE_HD double distrib_log_cdf(const D& d, double y) {
  if constexpr (requires { d.log_cdf(y); }) return d.log_cdf(y);
  else return log_cdf_by_sum(d, y);
}

/**
 * \ingroup distrib_generic
 * \brief Natural logarithm of complementary cumulative distribution function \f$\log(1 - F(y))\f$, dispatching to specialized member or generic summation.
 *
 * Calls `d.log_ccdf(y)` if provided by distribution `d`; otherwise calls log_ccdf_by_sum().
 * The choice is made at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return The value of \f$\log(1 - F(y))\f$.
 *
 * \see log_ccdf_by_sum(), distrib_ccdf(), distrib_log_cdf()
 */
template <class D>
LAPLACE_HD double distrib_log_ccdf(const D& d, double y) {
  if constexpr (requires { d.log_ccdf(y); }) return d.log_ccdf(y);
  else return log_ccdf_by_sum(d, y);
}

}  // namespace laplace::distrib
