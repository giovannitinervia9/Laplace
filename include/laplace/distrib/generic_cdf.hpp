// The cdf family of a discrete distribution by summing its mass, and the
// dispatchers that take a distribution's own member when it has one
// (laplace::distrib; note/distrib.md section 4).
//
//   cdf_by_sum(d, y), ccdf_by_sum, log_cdf_by_sum, log_ccdf_by_sum
//   distrib_cdf(d, y), distrib_ccdf, distrib_log_cdf, distrib_log_ccdf
//   log_cdf_dk_by_sum<K>(d, y), log_ccdf_dk_by_sum<K>  the derivatives in the
//                               parameters of log F and log(1 - F), K = 1..6
//   distrib_log_cdf_dk<K>(d, y), distrib_log_ccdf_dk<K>
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
#include <laplace/distrib/generic_info.hpp>
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

// The derivatives of a tail in the parameters. With S the tail summed,
// C = c[0..K-1] a sorted multi-index of parameters, T a sub-multiset of C (the
// bits of a mask over the positions of C) and B_T the complete Bell polynomial
// over T of the derivatives of a function (B_T = e^-g d_T e^g, B_empty = 1),
//
//   d_T S = sum_i p(i) B_T(l(i)),       l = log p,
//
// which summed as ratios to S itself would give d_T S / S directly. At high order
// in a tail those ratios are powers of the score (of size ((y - mu)/mu)^K on the
// Poisson) while d_T log S is of the size of d_T l alone, so the step from one to
// the other cancels: measured against the mpmath references of the Poisson's
// log F, up to 4e14 ulp at order 6. The sum is therefore centred on the first
// mass: with A = l(start) and m_i = l(i) - l(start),
//
//   S = e^A W,   W = sum_i e^(m_i),   d_T log S = d_T A + d_T log W,
//
// where d_T A is the mass's own derivative (exact and closed) and the derivatives
// of m_i, which vanish at the start, carry only the variation of l over the tail.
// This fills r[T] = d_T W / W (so r[empty] = 1) and l0[T] = d_T A, and returns
// log S. The walk and its stopping test are those of log_tail_by_sum, with the
// bound taken on the largest of the sums instead of the mass alone and asked for
// twice in a row: the mass decreases geometrically away from the mode and the
// |B_T| grow at most polynomially in i.
template <int K, class D>
LAPLACE_HD double tail_ratios_by_sum(const D& d, double k, bool upper, const int* c, double* r, double* l0) {
  constexpr int P = D::n_params;
  constexpr int full = (1 << K) - 1;
  const double start = upper ? k + 1.0 : k, end = upper ? d.support_upper() : d.support_lower();
  const int step = upper ? 1 : -1;
  const double la = log_density(d, start);
  for (int m = 0; m <= full; ++m) r[m] = l0[m] = 0.0;
  if (!(la > -distrib_infinity)) return la;
  log_density_partials<P, K> L0;
  L0.fill(d, start);
  for (int U = 1; U <= full; ++U) l0[U] = l_over(L0, nullptr, 0, c, U);  // d_T A, T not empty
  constexpr int n_partials = log_density_partials<P, K>::offset(K + 1);
  double A[1 << K], sabs[1 << K];
  for (int m = 0; m <= full; ++m) A[m] = sabs[m] = 0.0;
  double prev = 0.0;
  int quiet = 0;
  for (long n = 0; n < (1L << 26); ++n) {
    const double i = start + step * double(n);
    if (upper ? i > end : i < end) break;
    const double w = math::exp(log_density(d, i) - la);
    log_density_partials<P, K> L;
    L.fill(d, i);
    for (int j = 0; j < n_partials; ++j) L.v[j] -= L0.v[j];  // the derivatives of m_i
    double B[1 << K], mc[1 << K];
    B[0] = 1.0;
    for (int U = 1; U <= full; ++U) mc[U] = l_over(L, nullptr, 0, c, U);
    for (int mask = 1; mask <= full; ++mask) {  // the block that holds the lowest index of T
      const int low = mask & -mask, rest = mask ^ low;
      double acc = 0.0;
      for (int sub = rest;; sub = (sub - 1) & rest) {
        const int U = sub | low;
        acc += mc[U] * B[mask ^ U];
        if (sub == 0) break;
      }
      B[mask] = acc;
    }
    double tmax = 0.0, amax = 0.0;
    for (int m = 0; m <= full; ++m) {
      const double t = w * B[m];
      A[m] += t;
      sabs[m] += Kokkos::fabs(t);
      if (Kokkos::fabs(t) > tmax) tmax = Kokkos::fabs(t);
      if (sabs[m] > amax) amax = sabs[m];
    }
    if (n > 0 && prev > 0.0) {
      const double ratio = w / prev;
      if (w == 0.0 || (ratio < 1.0 && tmax * ratio <= 0x1p-55 * amax * (1.0 - ratio))) {
        if (++quiet == 2) break;
      } else {
        quiet = 0;
      }
    }
    prev = w;
  }
  if (!(A[0] > 0.0)) return -distrib_infinity;
  for (int m = 0; m <= full; ++m) r[m] = A[m] / A[0];
  return la + math::log(A[0]);
}

// u[T] = d_T log G from r[T] = d_T G / G, the Bell relation solved for the
// derivatives of log G: r[T] = sum over the sub-multisets U of T that hold T's
// lowest index of u[U] r[T - U], and the term U = T leaves u[T] alone.
template <int K>
LAPLACE_HD void log_tail_dk_from_ratios(const double* r, double* u) {
  constexpr int full = (1 << K) - 1;
  u[0] = 0.0;
  for (int mask = 1; mask <= full; ++mask) {
    const int low = mask & -mask, rest = mask ^ low;
    double acc = r[mask];
    if (rest != 0) {
      for (int sub = (rest - 1) & rest;; sub = (sub - 1) & rest) {
        const int U = sub | low;
        acc -= u[U] * r[mask ^ U];
        if (sub == 0) break;
      }
    }
    u[mask] = acc;
  }
}

// the forward relation, r[T] = d_T G / G = B_T(u) from u[T] = d_T log G
template <int K>
LAPLACE_HD void ratios_from_log_tail_dk(const double* u, double* r) {
  constexpr int full = (1 << K) - 1;
  r[0] = 1.0;
  for (int mask = 1; mask <= full; ++mask) {
    const int low = mask & -mask, rest = mask ^ low;
    double acc = 0.0;
    for (int sub = rest;; sub = (sub - 1) & rest) {
      const int U = sub | low;
      acc += u[U] * r[mask ^ U];
      if (sub == 0) break;
    }
    r[mask] = acc;
  }
}

// d^K log F / dtheta^K (upper = false) or d^K log(1 - F) / dtheta^K (upper = true)
// by summing the mass: one walk per multi-index over the cheaper tail S. When the
// side asked for is the other one, O = 1 - S, its ratios follow from d_T O =
// -d_T S: d_T O / O = -(S/O) d_T S / S, with d_T S / S rebuilt from the
// derivatives of log S, and are small wherever S is (the side summed is the one
// of smaller probability, so S / O <= 1).
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_side_dk_by_sum(const D& d, double y, bool upper) {
  static_assert(D::kind == distrib_kind::discrete,
                "the derivatives of the log cdf by summing the mass are for discrete distributions");
  constexpr int P = D::n_params;
  constexpr int full = (1 << K) - 1;
  sym_tensor<P, K> t;
  const double k = Kokkos::floor(y);
  const bool empty = upper ? k >= d.support_upper() : k < d.support_lower();
  if (empty) {  // the side asked for has probability 0: its logarithm is -inf
    for (int i = 0; i < sym_tensor<P, K>::size; ++i) t[i] = math::detail::quiet_nan();
    return t;
  }
  if (upper ? k < d.support_lower() : k >= d.support_upper()) return t;  // probability 1, every derivative 0
  const bool lower_side = k < d.center();
  const bool summed_is_asked = lower_side != upper;
  int c[K];
  for (int j = 0; j < K; ++j) c[j] = 0;
  for (int comp = 0; comp < sym_tensor<P, K>::size; ++comp) {
    double r[1 << K], u[1 << K], l0[1 << K];
    const double log_s = tail_ratios_by_sum<K>(d, k, !lower_side, c, r, l0);
    log_tail_dk_from_ratios<K>(r, u);                      // d_T log W
    for (int m = 1; m <= full; ++m) u[m] += l0[m];         // d_T log S = d_T A + d_T log W
    if (!summed_is_asked) {
      const double log_o = Kokkos::log1p(-math::exp(log_s));
      const double f = -math::exp(log_s - log_o);
      ratios_from_log_tail_dk<K>(u, r);
      for (int m = 1; m <= full; ++m) r[m] *= f;
      log_tail_dk_from_ratios<K>(r, u);
    }
    t[comp] = u[full];
    next_multi_index<P, K>(c);
  }
  return t;
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

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ for a discrete distribution by summation.
 *
 * With \f$k = \lfloor y \rfloor\f$ and \f$C\f$ a sorted multi-index of parameters,
 * \f$\partial_C F(k) = \sum_{i \le k} p(i) B_C(l(i))\f$, where \f$B_C\f$ is the complete Bell
 * polynomial of the derivatives of \f$l = \log p\f$ over the sub-multisets of \f$C\f$; the same
 * relation inverted gives the derivatives of \f$\log F\f$. The cheaper tail is summed and the
 * other side is taken from it by complement.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor \c sym_tensor<D::n_params, K> of derivatives.
 *
 * \par Domain and special values
 * - Quiet NaN when \f$\lfloor y \rfloor\f$ is below `d.support_lower()` (\f$\log F = -\infty\f$).
 * - Zero when \f$\lfloor y \rfloor\f$ is at least `d.support_upper()` (\f$F = 1\f$).
 *
 * \see log_ccdf_dk_by_sum(), log_cdf_by_sum(), distrib_log_cdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_cdf_dk_by_sum(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_sum<K>(d, y, false);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ for a discrete distribution by summation.
 *
 * The route of log_cdf_dk_by_sum() applied to the upper tail \f$\sum_{i > k} p(i)\f$.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \tparam D Discrete distribution type.
 * \param d Distribution object.
 * \param y Evaluation point.
 * \return Packed symmetric tensor \c sym_tensor<D::n_params, K> of derivatives.
 *
 * \par Domain and special values
 * - Zero when \f$\lfloor y \rfloor\f$ is below `d.support_lower()` (\f$1 - F = 1\f$).
 * - Quiet NaN when \f$\lfloor y \rfloor\f$ is at least `d.support_upper()` (\f$\log(1 - F) = -\infty\f$).
 *
 * \see log_cdf_dk_by_sum(), log_ccdf_by_sum(), distrib_log_ccdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> log_ccdf_dk_by_sum(const D& d, double y) {
  return detail_generic::log_cdf_side_dk_by_sum<K>(d, y, true);
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$, dispatching to the member or to the generic sum.
 *
 * Calls the distribution's own `log_cdf_dK(y)` when it has one; otherwise calls log_cdf_dk_by_sum().
 * The choice is made at compile time.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \see log_cdf_dk_by_sum(), distrib_log_ccdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> distrib_log_cdf_dk(const D& d, double y) {
  if constexpr (K == 1) {
    if constexpr (requires { d.log_cdf_d1(y); }) return d.log_cdf_d1(y);
    else return log_cdf_dk_by_sum<1>(d, y);
  } else if constexpr (K == 2) {
    if constexpr (requires { d.log_cdf_d2(y); }) return d.log_cdf_d2(y);
    else return log_cdf_dk_by_sum<2>(d, y);
  } else if constexpr (K == 3) {
    if constexpr (requires { d.log_cdf_d3(y); }) return d.log_cdf_d3(y);
    else return log_cdf_dk_by_sum<3>(d, y);
  } else if constexpr (K == 4) {
    if constexpr (requires { d.log_cdf_d4(y); }) return d.log_cdf_d4(y);
    else return log_cdf_dk_by_sum<4>(d, y);
  } else if constexpr (K == 5) {
    if constexpr (requires { d.log_cdf_d5(y); }) return d.log_cdf_d5(y);
    else return log_cdf_dk_by_sum<5>(d, y);
  } else {
    if constexpr (requires { d.log_cdf_d6(y); }) return d.log_cdf_d6(y);
    else return log_cdf_dk_by_sum<6>(d, y);
  }
}

/**
 * \ingroup distrib_generic
 * \brief Parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$, dispatching to the member or to the generic sum.
 *
 * Calls the distribution's own `log_ccdf_dK(y)` when it has one; otherwise calls log_ccdf_dk_by_sum().
 * The choice is made at compile time.
 *
 * \tparam K Derivative order (\f$1 \le K \le 6\f$).
 * \see log_ccdf_dk_by_sum(), distrib_log_cdf_dk()
 */
template <int K, class D>
LAPLACE_HD sym_tensor<D::n_params, K> distrib_log_ccdf_dk(const D& d, double y) {
  if constexpr (K == 1) {
    if constexpr (requires { d.log_ccdf_d1(y); }) return d.log_ccdf_d1(y);
    else return log_ccdf_dk_by_sum<1>(d, y);
  } else if constexpr (K == 2) {
    if constexpr (requires { d.log_ccdf_d2(y); }) return d.log_ccdf_d2(y);
    else return log_ccdf_dk_by_sum<2>(d, y);
  } else if constexpr (K == 3) {
    if constexpr (requires { d.log_ccdf_d3(y); }) return d.log_ccdf_d3(y);
    else return log_ccdf_dk_by_sum<3>(d, y);
  } else if constexpr (K == 4) {
    if constexpr (requires { d.log_ccdf_d4(y); }) return d.log_ccdf_d4(y);
    else return log_ccdf_dk_by_sum<4>(d, y);
  } else if constexpr (K == 5) {
    if constexpr (requires { d.log_ccdf_d5(y); }) return d.log_ccdf_d5(y);
    else return log_ccdf_dk_by_sum<5>(d, y);
  } else {
    if constexpr (requires { d.log_ccdf_d6(y); }) return d.log_ccdf_d6(y);
    else return log_ccdf_dk_by_sum<6>(d, y);
  }
}

}  // namespace laplace::distrib
