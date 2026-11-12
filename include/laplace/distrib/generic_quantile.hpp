// The quantile of a distribution by the generic route (laplace::distrib;
// note/distrib.md section 4), from p, from log p and from log(1 - p).
//
//   quantile_by_root(d, p), _from_log(d, log_p), _from_log_upper(d, log_q)
//                         continuous: the root of the cdf
//   quantile_by_search(d, p), _from_log(...), _from_log_upper(...)
//                         discrete: the smallest k of the support with F(k) >= p
//   distrib_quantile(d, p), distrib_quantile_from_log(d, log_p),
//   distrib_quantile_from_log_upper(d, log_q)
//                         the member quantile / quantile_from_log /
//                         quantile_from_log_upper when the distribution has it,
//                         the generic route otherwise
//
// Both routes work on a logarithmic scale and on the tail that holds the
// target: with t = log p <= -log 2 they solve log F(y) = t, otherwise log(1 - F(y))
// = log q with log q = log(1 - p) (log1p(-p), or log(-expm1(log p))), so that a
// quantile deep in either tail is found to the accuracy of the log cdf there.
//
// Continuous (root): g(y) = log F(y) - t, or log q - log(1 - F(y)), increases with
// y. A bracket is found from center() by steps of scale() that double towards an
// infinite end, and towards a finite end e by distances (e - y) 2^-m with m =
// 1, 2, 4, 8, ...; then Newton's step on g (g' = f/F or f/(1 - F)) is taken when it
// stays inside the bracket, and the bracket is bisected otherwise (at the geometric
// mean of the distances to a finite end when they differ by more than a factor 4,
// so that a quantile at 1e-300 from the end takes tens of steps). It stops when
// Newton's step or the bracket falls below 4 eps |y| (or the bracket closes on an
// end), after at most 200 steps. A last Newton step on F(y) = p itself (or on
// 1 - F(y) = 1 - p) removes the error that the log scale adds in a tail.
//
// Discrete (search): the predicate F(k) >= p, read as log F(k) >= t or
// log(1 - F(k)) <= log q, is monotone in k; a bracket is found from floor(center())
// by steps of ceil(scale()) that double, then the integers are bisected.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_generic {

constexpr double log_half = -0.69314718055994530942;

// log(1 - e^x), x <= 0 (Maechler's switch at -log 2)
LAPLACE_HD_FORCEINLINE double log1mexp(double x) {
  return x > log_half ? math::log(-Kokkos::expm1(x)) : Kokkos::log1p(-math::exp(x));
}

// the root of log F(y) = t (upper = false) or log(1 - F(y)) = t (upper = true)
template <class D>
LAPLACE_HD double quantile_root_log(const D& d, double t, bool upper) {
  const double lo_end = d.support_lower(), hi_end = d.support_upper();
  const bool lo_fin = lo_end > -1e308, hi_fin = hi_end < 1e308;
  // g increases with y
  auto g = [&](double y) { return upper ? t - distrib_log_ccdf(d, y) : distrib_log_cdf(d, y) - t; };
  double x = d.center(), s = d.scale();
  if (!(x > lo_end)) x = lo_fin && hi_fin ? 0.5 * (lo_end + hi_end) : (lo_fin ? lo_end + s : hi_end - s);
  if (!(x < hi_end)) x = hi_fin && lo_fin ? 0.5 * (lo_end + hi_end) : hi_end - s;
  double gx = g(x);
  if (gx == 0.0) return x;
  if (gx != gx) return math::detail::quiet_nan();
  double lo = lo_end, hi = hi_end, glo = -distrib_infinity, ghi = distrib_infinity;  // an end is never the start
  // the bracket
  {
    const bool right = gx < 0.0;  // the root is above x
    (right ? lo : hi) = x, (right ? glo : ghi) = gx;
    const bool fin = right ? hi_fin : lo_fin;
    const double e = right ? hi_end : lo_end;
    double step = s, m = 1.0;
    for (int it = 0; it < 64; ++it) {
      double nx;
      if (fin) {
        nx = e - (e - x) * math::exp(-0.69314718055994530942 * m);  // (e - x) 2^-m from e
        m *= 2.0;
      } else {
        nx = right ? x + step : x - step;
        step *= 2.0;
      }
      if (nx == (right ? lo : hi) || !(nx > lo_end && nx < hi_end)) break;  // the end itself bounds the root
      const double gn = g(nx);
      if (gn == 0.0) return nx;
      if ((gn > 0.0) == right) {
        (right ? hi : lo) = nx, (right ? ghi : glo) = gn;
        break;
      }
      (right ? lo : hi) = nx, (right ? glo : ghi) = gn;
      if (!fin) x = nx;
    }
  }
  // Newton inside the bracket, bisection otherwise
  x = glo > -ghi ? lo : hi;
  double gc = glo > -ghi ? glo : ghi;
  for (int it = 0; it < 200; ++it) {
    if (gc == 0.0) return x;
    if (gc < 0.0) lo = x; else hi = x;
    const double lf = detail_generic::log_density(d, x);
    const double dg = math::exp(lf - (upper ? distrib_log_ccdf(d, x) : distrib_log_cdf(d, x)));
    double nx = x - gc / dg;
    const double tol = 4.0 * 0x1p-53 * Kokkos::fabs(x);
    if (nx > lo && nx < hi && dg > 0.0) {
      if (Kokkos::fabs(nx - x) <= tol) return nx;
    } else {
      if (lo_fin && lo > lo_end && hi - lo_end > 4.0 * (lo - lo_end)) nx = lo_end + Kokkos::sqrt((lo - lo_end) * (hi - lo_end));
      else if (hi_fin && hi < hi_end && hi_end - lo > 4.0 * (hi_end - hi)) nx = hi_end - Kokkos::sqrt((hi_end - lo) * (hi_end - hi));
      else nx = 0.5 * (lo + hi);
      if (!(nx > lo && nx < hi)) return x;  // the bracket has closed
    }
    if (hi - lo <= 4.0 * 0x1p-53 * Kokkos::fmax(Kokkos::fabs(lo), Kokkos::fabs(hi))) return nx;
    x = nx;
    gc = g(x);
    if (gc != gc) return math::detail::quiet_nan();
  }
  return x;
}

// the root on the log scale, then one Newton step on F(y) = p (or 1 - F(y) = q) with
// the target as given, when it is a normal number: the log cdf of a tail has an
// absolute error of a few ulps of |log F|, which is a relative error of F as large
// as eps |log F| (measured: 67 ulps of y at p = 1e-300 on gamma1(2, 0.3) without it)
template <class D>
LAPLACE_HD double quantile_root(const D& d, double t, bool upper, double target) {
  const double y = quantile_root_log(d, t, upper);
  if (!(target >= 0x1p-1022) || !(y > d.support_lower() && y < d.support_upper())) return y;
  const double f = math::exp(detail_generic::log_density(d, y));
  if (!(f > 0.0)) return y;
  const double ny = upper ? y + (distrib_ccdf(d, y) - target) / f : y - (distrib_cdf(d, y) - target) / f;
  return ny > d.support_lower() && ny < d.support_upper() ? ny : y;
}

// the smallest k of the support with log F(k) >= t (upper = false) or log(1 - F(k)) <= t (upper = true)
template <class D>
LAPLACE_HD double quantile_search(const D& d, double t, bool upper) {
  const double lo_end = d.support_lower(), hi_end = d.support_upper();
  auto ok = [&](double k) { return upper ? distrib_log_ccdf(d, k) <= t : distrib_log_cdf(d, k) >= t; };
  double x = Kokkos::floor(d.center());
  if (x < lo_end) x = lo_end;
  if (x > hi_end) x = hi_end;
  double step = Kokkos::ceil(d.scale());
  if (!(step >= 1.0)) step = 1.0;
  double lo, hi;  // ok(lo) false (lo_end - 1 counts as false), ok(hi) true
  if (ok(x)) {
    hi = x, lo = x - step;
    for (int it = 0; it < 1100; ++it, step *= 2.0, lo = hi - step) {
      if (lo < lo_end) {
        lo = lo_end - 1.0;
        break;
      }
      if (!ok(lo)) break;
      hi = lo;
    }
  } else {
    lo = x, hi = x + step;
    for (int it = 0; it < 1100; ++it, step *= 2.0, hi = lo + step) {
      if (hi >= hi_end) {
        hi = hi_end;  // F(hi_end) = 1 (an infinite end is never reached: ok() holds before)
        break;
      }
      if (ok(hi)) break;
      lo = hi;
    }
  }
  while (hi - lo > 1.0) {
    const double m = Kokkos::floor(0.5 * (lo + hi));
    if (ok(m)) hi = m; else lo = m;
  }
  return hi;
}

}  // namespace detail_generic

// F^-1(p) of a continuous distribution, by the root of its log cdf.
/**
 * \ingroup distrib_generic
 * \brief Quantile \f$F^{-1}(p)\f$ of a continuous distribution by numerical root-finding on the log-cdf scale.
 *
 * Solves \f$\log F(y) = \log p\f$ for \f$p \le 1/2\f$, or \f$\log(1 - F(y)) = \log(1 - p)\f$ for \f$p > 1/2\f$.
 * Brackets the root from `center()` by doubling steps of `scale()` towards an infinite end or geometric
 * approach towards a finite boundary, followed by guarded Newton iterations with bisection fallback, and
 * concludes with one Newton step on \f$F(y) = p\f$ (or on \f$1 - F(y) = 1 - p\f$) on the linear scale.
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param p Probability \f$p \in [0, 1]\f$.
 * \return The quantile \f$y\f$ satisfying \f$F(y) = p\f$.
 *
 * \par Domain and special values
 * - If \f$p = 0\f$, returns `d.support_lower()`.
 * - If \f$p = 1\f$, returns `d.support_upper()`.
 * - If \f$p < 0.0\f$, \f$p > 1.0\f$, or \f$p\f$ is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * Worst error 31.9 (gcc), on gamma1(2, 0.3) at \f$p = 10^{-300}\f$, in units of the change of \f$y\f$ caused by one ulp of \f$y\f$ or of \f$p\f$ (the measure is defined on the module page), over 13 values of \f$p\f$ from \f$10^{-300}\f$ to \f$1 - 10^{-15}\f$ on gaussian1, laplace and gamma1.
 *
 * \see quantile_by_root_from_log(), quantile_by_root_from_log_upper(), distrib_quantile()
 */
template <class D>
LAPLACE_HD double quantile_by_root(const D& d, double p) {
  if (!(p > 0.0)) return p == 0.0 ? d.support_lower() : math::detail::quiet_nan();
  if (!(p < 1.0)) return p == 1.0 ? d.support_upper() : math::detail::quiet_nan();
  return p <= 0.5 ? detail_generic::quantile_root(d, math::log(p), false, p)
                  : detail_generic::quantile_root(d, Kokkos::log1p(-p), true, 1.0 - p);
}
// F^-1(e^log_p) of a continuous distribution.
/**
 * \ingroup distrib_generic
 * \brief Quantile \f$F^{-1}(e^{\log p})\f$ of a continuous distribution from the log-probability \f$\log p\f$.
 *
 * Solves \f$\log F(y) = \log p\f$ for \f$\log p \le -\log 2\f$, or \f$\log(1 - F(y)) = \log(1 - e^{\log p})\f$ for \f$\log p > -\log 2\f$.
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param log_p Natural logarithm of probability \f$\log p \in (-\infty, 0]\f$.
 * \return The quantile \f$y\f$ satisfying \f$\log F(y) = \log p\f$.
 *
 * \par Domain and special values
 * - If \f$\log p = 0\f$, returns `d.support_upper()`.
 * - If \f$\log p\f$ is \f$-\infty\f$ (at most `-distrib_infinity`), returns `d.support_lower()`.
 * - If \f$\log p > 0.0\f$ or is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * The same points and worst error as quantile_by_root(): 31.9 (gcc).
 *
 * \see quantile_by_root(), quantile_by_root_from_log_upper(), distrib_quantile_from_log()
 */
template <class D>
LAPLACE_HD double quantile_by_root_from_log(const D& d, double log_p) {
  if (!(log_p < 0.0)) return log_p == 0.0 ? d.support_upper() : math::detail::quiet_nan();
  if (!(log_p > -distrib_infinity)) return d.support_lower();
  if (log_p <= detail_generic::log_half) return detail_generic::quantile_root(d, log_p, false, math::exp(log_p));
  const double lq = detail_generic::log1mexp(log_p);
  return detail_generic::quantile_root(d, lq, true, math::exp(lq));
}
// F^-1(1 - e^log_q) of a continuous distribution: the upper quantile.
/**
 * \ingroup distrib_generic
 * \brief Upper quantile \f$F^{-1}(1 - e^{\log q})\f$ of a continuous distribution from complementary log-probability \f$\log q = \log(1 - p)\f$.
 *
 * Solves \f$\log(1 - F(y)) = \log q\f$ for \f$\log q \le -\log 2\f$, or \f$\log F(y) = \log(1 - e^{\log q})\f$ for \f$\log q > -\log 2\f$.
 *
 * \tparam D Continuous distribution type.
 * \param d Distribution object.
 * \param log_q Natural logarithm of upper tail probability \f$\log(1 - p) \in (-\infty, 0]\f$.
 * \return The quantile \f$y\f$ satisfying \f$\log(1 - F(y)) = \log q\f$.
 *
 * \par Domain and special values
 * - If \f$\log q = 0\f$, returns `d.support_lower()`.
 * - If \f$\log q\f$ is \f$-\infty\f$ (at most `-distrib_infinity`), returns `d.support_upper()`.
 * - If \f$\log q > 0.0\f$ or is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * The same worst error as quantile_by_root() for \f$p \ge 10^{-15}\f$, 31.9 (gcc); at \f$\log q = -50\f$ and \f$-300\f$ the tests check that \f$\log(1 - F)\f$ at the result equals \f$\log q\f$ to a relative \f$10^{-12}\f$.
 *
 * \see quantile_by_root(), quantile_by_root_from_log(), distrib_quantile_from_log_upper()
 */
template <class D>
LAPLACE_HD double quantile_by_root_from_log_upper(const D& d, double log_q) {
  if (!(log_q < 0.0)) return log_q == 0.0 ? d.support_lower() : math::detail::quiet_nan();
  if (!(log_q > -distrib_infinity)) return d.support_upper();
  if (log_q <= detail_generic::log_half) return detail_generic::quantile_root(d, log_q, true, math::exp(log_q));
  const double lp = detail_generic::log1mexp(log_q);
  return detail_generic::quantile_root(d, lp, false, math::exp(lp));
}

// The smallest k of the support with F(k) >= p, for a discrete distribution.
/**
 * \ingroup distrib_generic
 * \brief Quantile of a discrete distribution by discrete search: smallest integer \f$k\f$ with \f$F(k) \ge p\f$.
 *
 * Uses the monotone predicate \f$\log F(k) \ge \log p\f$ (for \f$p \le 1/2\f$) or \f$\log(1 - F(k)) \le \log(1 - p)\f$ (for \f$p > 1/2\f$).
 * Brackets from \f$\lfloor \mathrm{center}() \rfloor\f$ with doubling steps of \f$\lceil \mathrm{scale}() \rceil\f$, followed by integer bisection.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param p Probability \f$p \in [0, 1]\f$.
 * \return Smallest integer \f$k\f$ within support such that \f$F(k) \ge p\f$.
 *
 * \par Domain and special values
 * - If \f$p = 0\f$, returns `d.support_lower()`.
 * - If \f$p = 1\f$, returns `d.support_upper()`.
 * - If \f$p < 0.0\f$, \f$p > 1.0\f$, or \f$p\f$ is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * Returned the exact smallest integer \f$k\f$ with \f$F(k) \ge p\f$ in every case tested on Poisson.
 *
 * \see quantile_by_search_from_log(), quantile_by_search_from_log_upper(), distrib_quantile()
 */
template <class D>
LAPLACE_HD double quantile_by_search(const D& d, double p) {
  if (!(p > 0.0)) return p == 0.0 ? d.support_lower() : math::detail::quiet_nan();
  if (!(p < 1.0)) return p == 1.0 ? d.support_upper() : math::detail::quiet_nan();
  return p <= 0.5 ? detail_generic::quantile_search(d, math::log(p), false)
                  : detail_generic::quantile_search(d, Kokkos::log1p(-p), true);
}
/**
 * \ingroup distrib_generic
 * \brief Quantile of a discrete distribution from \f$\log p\f$: smallest integer \f$k\f$ with \f$\log F(k) \ge \log p\f$.
 *
 * Uses the predicate \f$\log F(k) \ge \log p\f$ for \f$\log p \le -\log 2\f$, or \f$\log(1 - F(k)) \le \log(1 - e^{\log p})\f$ for \f$\log p > -\log 2\f$, using doubling bracket and integer bisection.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param log_p Natural logarithm of probability \f$\log p \in (-\infty, 0]\f$.
 * \return Smallest integer \f$k\f$ within support such that \f$\log F(k) \ge \log p\f$.
 *
 * \par Domain and special values
 * - If \f$\log p = 0\f$, returns `d.support_upper()`.
 * - If \f$\log p\f$ is \f$-\infty\f$ (at most `-distrib_infinity`), returns `d.support_lower()`.
 * - If \f$\log p > 0.0\f$ or is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * Identical to quantile_by_search().
 *
 * \see quantile_by_search(), quantile_by_search_from_log_upper(), distrib_quantile_from_log()
 */
template <class D>
LAPLACE_HD double quantile_by_search_from_log(const D& d, double log_p) {
  if (!(log_p < 0.0)) return log_p == 0.0 ? d.support_upper() : math::detail::quiet_nan();
  if (!(log_p > -distrib_infinity)) return d.support_lower();
  return log_p <= detail_generic::log_half ? detail_generic::quantile_search(d, log_p, false)
                                           : detail_generic::quantile_search(d, detail_generic::log1mexp(log_p), true);
}
/**
 * \ingroup distrib_generic
 * \brief Upper quantile of a discrete distribution from \f$\log q = \log(1 - p)\f$: smallest integer \f$k\f$ with \f$\log(1 - F(k)) \le \log q\f$.
 *
 * Uses the predicate \f$\log(1 - F(k)) \le \log q\f$ for \f$\log q \le -\log 2\f$, or \f$\log F(k) \ge \log(1 - e^{\log q})\f$ for \f$\log q > -\log 2\f$, using doubling bracket and integer bisection.
 *
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param log_q Natural logarithm of upper tail probability \f$\log(1 - p) \in (-\infty, 0]\f$.
 * \return Smallest integer \f$k\f$ within support such that \f$\log(1 - F(k)) \le \log q\f$.
 *
 * \par Domain and special values
 * - If \f$\log q = 0\f$, returns `d.support_lower()`.
 * - If \f$\log q\f$ is \f$-\infty\f$ (at most `-distrib_infinity`), returns `d.support_upper()`.
 * - If \f$\log q > 0.0\f$ or is NaN, returns quiet NaN.
 *
 * \par Measured accuracy
 * Identical to quantile_by_search().
 *
 * \see quantile_by_search(), quantile_by_search_from_log(), distrib_quantile_from_log_upper()
 */
template <class D>
LAPLACE_HD double quantile_by_search_from_log_upper(const D& d, double log_q) {
  if (!(log_q < 0.0)) return log_q == 0.0 ? d.support_lower() : math::detail::quiet_nan();
  if (!(log_q > -distrib_infinity)) return d.support_upper();
  return log_q <= detail_generic::log_half ? detail_generic::quantile_search(d, log_q, true)
                                           : detail_generic::quantile_search(d, detail_generic::log1mexp(log_q), false);
}

/**
 * \ingroup distrib_generic
 * \brief Quantile function \f$F^{-1}(p)\f$, dispatching to specialized member or generic route.
 *
 * Calls `d.quantile(p)` if provided by distribution `d`; otherwise calls quantile_by_root() for continuous
 * distributions or quantile_by_search() for discrete distributions at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param p Probability \f$p \in [0, 1]\f$.
 * \return The computed quantile value.
 *
 * \see quantile_by_root(), quantile_by_search(), distrib_quantile_from_log()
 */
template <class D>
LAPLACE_HD double distrib_quantile(const D& d, double p) {
  if constexpr (requires { d.quantile(p); }) return d.quantile(p);
  else if constexpr (D::kind == distrib_kind::continuous) return quantile_by_root(d, p);
  else return quantile_by_search(d, p);
}
/**
 * \ingroup distrib_generic
 * \brief Quantile from log-probability \f$\log p\f$, dispatching to specialized member or generic route.
 *
 * Calls `d.quantile_from_log(log_p)` if provided by distribution `d`; otherwise calls quantile_by_root_from_log()
 * for continuous distributions or quantile_by_search_from_log() for discrete distributions at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param log_p Natural logarithm of probability \f$\log p \in (-\infty, 0]\f$.
 * \return The computed quantile value.
 *
 * \see quantile_by_root_from_log(), quantile_by_search_from_log(), distrib_quantile()
 */
template <class D>
LAPLACE_HD double distrib_quantile_from_log(const D& d, double log_p) {
  if constexpr (requires { d.quantile_from_log(log_p); }) return d.quantile_from_log(log_p);
  else if constexpr (D::kind == distrib_kind::continuous) return quantile_by_root_from_log(d, log_p);
  else return quantile_by_search_from_log(d, log_p);
}
/**
 * \ingroup distrib_generic
 * \brief Upper quantile from complementary log-probability \f$\log q = \log(1 - p)\f$, dispatching to specialized member or generic route.
 *
 * Calls `d.quantile_from_log_upper(log_q)` if provided by distribution `d`; otherwise calls quantile_by_root_from_log_upper()
 * for continuous distributions or quantile_by_search_from_log_upper() for discrete distributions at compile time.
 *
 * \tparam D Distribution type.
 * \param d Distribution object.
 * \param log_q Natural logarithm of upper tail probability \f$\log(1 - p) \in (-\infty, 0]\f$.
 * \return The computed quantile value.
 *
 * \see quantile_by_root_from_log_upper(), quantile_by_search_from_log_upper(), distrib_quantile()
 */
template <class D>
LAPLACE_HD double distrib_quantile_from_log_upper(const D& d, double log_q) {
  if constexpr (requires { d.quantile_from_log_upper(log_q); }) return d.quantile_from_log_upper(log_q);
  else if constexpr (D::kind == distrib_kind::continuous) return quantile_by_root_from_log_upper(d, log_q);
  else return quantile_by_search_from_log_upper(d, log_q);
}

}  // namespace laplace::distrib
