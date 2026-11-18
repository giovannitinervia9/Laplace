// Dawson's integral F and its derivatives to order 6 (laplace::math):
//
//   F(x) = e^(-x^2) int_0^x e^(s^2) ds,
//
// odd, bounded, with one maximum F = 0.5410442246351816 at
// x = 0.9241388730045917; F(x) ~ x near 0 and F(x) ~ 1/(2x) as x -> inf, and
// F'(x) = 1 - 2 x F(x) exactly. Needed by the normalizing constants of the
// properized MCP and SCAD penalties, where the argument is lambda sqrt(gamma/2)
// or lambda sqrt((a-1)/2) and may be large.
//
// Two routes, chosen by measurement against mpmath (docs/numerics.md, 2026-11-18):
//
//   |x| < 8   the scaled series. With A_K(x) = e^(x^2) F^(K)(x), the identity
//             A_(K+1) = A_K' - 2 x A_K turns the series of
//             A_0(x) = int_0^x e^(s^2) ds = sum_m x^(2m+1)/(m! (2m+1)) into
//
//               F^(K)(x) = e^(-x^2) sum_(m >= 0) b_K(m) x^n / m!,
//
//             with n = 2m + 1 for K even and n = 2m for K odd, and
//
//               b_0(m) =     1/(2m+1)                         b_1(m) =    -1/(2m-1)
//               b_2(m) =     4/((2m-1)(2m+1))                  b_3(m) =   -12/((2m-3)(2m-1))
//               b_4(m) =    96/((2m-3)(2m-1)(2m+1))            b_5(m) =  -480/((2m-5)(2m-3)(2m-1))
//               b_6(m) =  5760/((2m-5)(2m-3)(2m-1)(2m+1)).
//
//             One formula covers every m >= 0, with no exceptional leading
//             coefficient, and no denominator vanishes, since 2m - j is odd
//             for odd j. Each b_K keeps one sign from the first m at which all
//             its factors are positive on (m >= 1 for K <= 2, m >= 2, 3, 4 for
//             the higher orders), so the tail of every series has one sign and
//             nothing cancels inside it. The forms were derived and checked
//             against SymPy's own derivatives in tools/reference/dawson_forms.py.
//             The series is summed at the rounded w = fl(x^2) and multiplied by
//             e^(-w) at the same w: the sum behaves like e^w, so the error that
//             rounding x^2 causes in the two factors cancels to first order,
//             and the fma-corrected e^(-x^2) would instead leave it in.
//
//   |x| >= 8  the asymptotic series, differentiated term by term:
//
//               F^(K)(x) ~ (-1)^K sum_m ((2m-1)!!/2^(m+1)) (2m+1)_K x^(-(2m+1+K)),
//
//             (2m+1)_K = (2m+1)(2m+2)...(2m+K) the rising factorial. Every term
//             has the sign (-1)^K, so nothing cancels here either; the sum is
//             stopped at its smallest term, which is where an asymptotic series
//             is most accurate.
//
// Measured and rejected: the recurrence F^(K+1) = -2 (x F^(K) + K F^(K-1)) for
// the derivatives, which cancels, since the two terms of a step agree to about
// log2(x) bits, so order 1 loses 2 log2(x) bits and each further order another
// log2(x) (measured over [5/2, 8): 7.6 ulp1 at order 1 and 5.0e6 at order 6);
// the alternating Taylor series of F, which cancels above |x| = 2 (33 ulp1
// below 5/2, 1.7e26 on [5/2, 8)); Rybicki's equally spaced sum (4.2 ulp but
// 140 ns and the value only); and the continued fraction of the asymptotic
// series (1.26 ulp above 8, better than the series used there, but the value
// only and useless below).
// NaN gives NaN; F and every derivative are 0 at +-inf.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::math {

namespace detail_dawson {

// |x| at which the asymptotic series takes over, measured in benchmarks/math_dawson.cpp
constexpr double asym_from = 8.0;
constexpr int max_terms = 320;  // the scaled series needs about x^2 + 5 |x| of them

// t_m / t_(m-1) for the scaled series, which is (w/m) b_K(m)/b_K(m-1) with
// w = x^2: every denominator of b_K is a window of consecutive odd numbers, so
// the ratio of two neighbours is a single fraction and one division per term is
// enough. Numerator 2m - 1 - 2 floor((K+1)/2), denominator m (2m + 1) for K
// even and m (2m - 1) for K odd.
template <int K>
LAPLACE_HD_FORCEINLINE double scaled_ratio(double w, int m) {
  const double num = double(2 * m - 1 - 2 * ((K + 1) / 2));
  const double den = double(m) * double(K % 2 == 0 ? 2 * m + 1 : 2 * m - 1);
  return w * num / den;
}

// b_K(0), the first coefficient: 1, 1, -4, -4, 32, 32, -384 for K = 0..6
template <int K>
LAPLACE_HD_FORCEINLINE constexpr double scaled_first() {
  if constexpr (K == 0) return 1.0;
  else if constexpr (K == 1) return 1.0;
  else if constexpr (K == 2) return -4.0;
  else if constexpr (K == 3) return -4.0;
  else if constexpr (K == 4) return 32.0;
  else if constexpr (K == 5) return 32.0;
  else return -384.0;
}

// F^(K)(x) = e^(-x^2) sum_m b_K(m) x^n / m!, n = 2m + 1 (K even) or 2m (K odd)
template <int K>
LAPLACE_HD double scaled_series(double x) {
  const double w = x * x;
  double t = scaled_first<K>();
  if constexpr (K % 2 == 0) t *= x;  // the odd powers carry one factor x
  double sum = t, amax = Kokkos::fabs(t);
  for (int m = 1; m <= max_terms; ++m) {
    t *= scaled_ratio<K>(w, m);
    sum += t;
    const double at = Kokkos::fabs(t);
    if (at > amax) amax = at;
    // the tail is single-signed and decreasing from here on, so a term below
    // 2^-54 of the largest one cannot move the sum
    if (at <= 0x1p-54 * amax) break;
  }
  return math::exp(-w) * sum;
}

// t_m / t_(m-1) for the asymptotic series: (c_m/c_(m-1)) (2m+1)_K/(2m-1)_K w
// with c_m = (2m-1)!!/2^(m+1), which is w (2m + K - 1)(2m + K)/(4m)
template <int K>
LAPLACE_HD_FORCEINLINE double asym_ratio(double w, int m) {
  return w * double(2 * m + K - 1) * double(2 * m + K) / double(4 * m);
}

// F^(K)(x) ~ (-1)^K x^-(K+1) sum_m c_m (2m+1)_K x^(-2m), summed to its smallest term
template <int K>
LAPLACE_HD double asymptotic(double x) {
  const double v = 1.0 / x, w = v * v;
  double t = 0.5;                                 // c_0 (1)_K = K!/2
  for (int i = 1; i <= K; ++i) t *= double(i);
  double sum = t, prev = t;                       // the terms decrease from m = 1 for |x| >= 6
  for (int m = 1; m <= 80; ++m) {
    const double tn = t * asym_ratio<K>(w, m);
    const double at = Kokkos::fabs(tn);
    if (!(at < prev)) break;                      // the series diverges: stop at the smallest term
    t = tn;
    sum += t;
    prev = at;
    if (at <= 0x1p-54 * Kokkos::fabs(sum)) break;
  }
  double p = v;                                   // x^-(K+1); 0 at +-inf, with the sign of v
  for (int i = 0; i < K; ++i) p *= v;
  return (K % 2 == 0 ? 1.0 : -1.0) * p * sum;
}

template <int K>
LAPLACE_HD double value(double x) {
  if (Kokkos::isnan(x)) return x;
  return Kokkos::fabs(x) < asym_from ? scaled_series<K>(x) : asymptotic<K>(x);
}

}  // namespace detail_dawson

/// \brief Dawson's integral \f$F(x) = e^{-x^2} \int_0^x e^{s^2} \, \mathrm{d}s\f$.
///
/// For \f$|x| < 8\f$ the value is the scaled series \f$F(x) = e^{-x^2} \sum_{m \ge 0} b_0(m) x^{2m+1} / m!\f$
/// with \f$b_0(m) = 1/(2m+1)\f$. For \f$|x| \ge 8\f$ it is the asymptotic series
/// \f$F(x) \sim \sum_{m \ge 0} \frac{(2m-1)!!}{2^{m+1}} x^{-(2m+1)}\f$, summed to its smallest term.
/// The method and the threshold are explained on the module page \ref dawson "Dawson's integral".
///
/// \param x the argument, any real number
/// \return \f$F(x)\f$; \f$F(0) = 0\f$, \f$F\f$ is odd and bounded, with its maximum
///         \f$0.5410442246351816\f$ at \f$x = 0.9241388730045917\f$
///
/// The value is 0 at both infinities and NaN for a NaN argument. Its error is
/// 1.24 ulp1 below \f$|x| = 8\f$ and 6.50 ulp above it. Over the whole line the
/// relative error is 9.68 ulp on gcc/OpenMP and 10.35 ulp on CUDA (see \ref accuracy "How accuracy is measured").
///
/// \ingroup dawson
/// \see dawson_d1(), dawson_d2(), and the module \ref dawson "Dawson's integral"
// Dawson's integral F(x) = e^(-x^2) int_0^x e^(s^2) ds
LAPLACE_HD_FORCEINLINE double dawson(double x) { return detail_dawson::value<0>(x); }
/// \brief First derivative \f$F'(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_1(m) = -1/(2m-1)\f$ and powers \f$x^{2m}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$-1/(2x^2)\f$.
///
/// \param x the argument, any real number
/// \return \f$F'(x) = 1 - 2 x F(x)\f$, with \f$F'(0) = 1\f$ and \f$F'\f$ even
///
/// \f$F'\f$ vanishes at \f$x = 0.9241388730045917\f$, the maximum of \f$F\f$, and at its mirror
/// point. Its measured error is
/// 1.07 ulp1 below \f$|x| = 8\f$ and 5.37 ulp above it.
///
/// \ingroup dawson
/// \see dawson(), dawson_d2()
// its derivatives F^(k)(x), k = 1..6
LAPLACE_HD_FORCEINLINE double dawson_d1(double x) { return detail_dawson::value<1>(x); }
/// \brief Second derivative \f$F''(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_2(m) = 4/((2m-1)(2m+1))\f$ and powers \f$x^{2m+1}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$1/x^3\f$.
///
/// \param x the argument, any real number
/// \return \f$F''(x)\f$, with \f$F''(0) = 0\f$ and \f$F''\f$ odd
///
/// Its measured error is
/// 2.54 ulp1 below \f$|x| = 8\f$ and 6.14 ulp above it.
///
/// \ingroup dawson
/// \see dawson_d1(), dawson_d3()
LAPLACE_HD_FORCEINLINE double dawson_d2(double x) { return detail_dawson::value<2>(x); }
/// \brief Third derivative \f$F^{(3)}(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_3(m) = -12/((2m-3)(2m-1))\f$ and powers \f$x^{2m}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$-3/x^4\f$.
///
/// \param x the argument, any real number
/// \return \f$F^{(3)}(x)\f$, with \f$F^{(3)}(0) = -4\f$ and \f$F^{(3)}\f$ even
///
/// Its measured error is
/// 4.63 ulp1 below \f$|x| = 8\f$ and 6.01 ulp above it.
///
/// \ingroup dawson
/// \see dawson_d2(), dawson_d4()
LAPLACE_HD_FORCEINLINE double dawson_d3(double x) { return detail_dawson::value<3>(x); }
/// \brief Fourth derivative \f$F^{(4)}(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_4(m) = 96/((2m-3)(2m-1)(2m+1))\f$ and powers \f$x^{2m+1}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$12/x^5\f$.
///
/// \param x the argument, any real number
/// \return \f$F^{(4)}(x)\f$, with \f$F^{(4)}(0) = 0\f$ and \f$F^{(4)}\f$ odd
///
/// Its measured error is
/// 8.00 ulp1 below \f$|x| = 8\f$ and 5.78 ulp above it.
///
/// \ingroup dawson
/// \see dawson_d3(), dawson_d5()
LAPLACE_HD_FORCEINLINE double dawson_d4(double x) { return detail_dawson::value<4>(x); }
/// \brief Fifth derivative \f$F^{(5)}(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_5(m) = -480/((2m-5)(2m-3)(2m-1))\f$ and powers \f$x^{2m}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$-60/x^6\f$.
///
/// \param x the argument, any real number
/// \return \f$F^{(5)}(x)\f$, with \f$F^{(5)}(0) = 32\f$ and \f$F^{(5)}\f$ even
///
/// Its measured error is
/// 13.4 ulp1 below \f$|x| = 8\f$ and 8.27 ulp above it.
///
/// \ingroup dawson
/// \see dawson_d4(), dawson_d6()
LAPLACE_HD_FORCEINLINE double dawson_d5(double x) { return detail_dawson::value<5>(x); }
/// \brief Sixth derivative \f$F^{(6)}(x)\f$ of Dawson's integral.
///
/// Below \f$|x| = 8\f$ the scaled series has \f$b_6(m) = 5760/((2m-5)(2m-3)(2m-1)(2m+1))\f$ and powers \f$x^{2m+1}\f$.
/// From \f$|x| = 8\f$ on, the asymptotic series starts with \f$360/x^7\f$.
///
/// \param x the argument, any real number
/// \return \f$F^{(6)}(x)\f$, with \f$F^{(6)}(0) = 0\f$ and \f$F^{(6)}\f$ odd
///
/// Its measured error is
/// 89.5 ulp1 below \f$|x| = 8\f$, reached where the derivative crosses 1, and 7.95 ulp above it.
///
/// \ingroup dawson
/// \see dawson_d5(), dawson()
LAPLACE_HD_FORCEINLINE double dawson_d6(double x) { return detail_dawson::value<6>(x); }

}  // namespace laplace::math
