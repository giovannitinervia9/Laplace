// Stirling's error delta(a) and its derivatives to order 6 (laplace::math), a > 0:
//
//   delta(a) = log Gamma(a) - (a - 1/2) log a + a - log sqrt(2 pi),
//
// the remainder of Stirling's series, delta(a) ~ 1/(12 a) as a -> inf. A
// log-density with a shape a (the gamma distribution) written through delta
// has no cancellation at large a; its derivatives in a need delta^(k)(a), and
// the direct form psi^(k-1)(a) - d^k[(a - 1/2) log a - a] cancels: the result
// is about 1/(12 a^2) times its largest term.
//
// Routes, chosen by measurement against mpmath (docs/numerics.md, 2026-11-12):
//   a < 1        direct: the definition through lgamma, digamma and the
//                polygamma functions, whose pole at 0 dominates there;
//   1 <= a < 10  recurrence: delta(a) = delta(a + n) + sum_(j < n) g(a + j),
//                with a + n >= 10 and g(b) = delta(b) - delta(b + 1)
//                = (b + 1/2) log(1 + 1/b) - 1 = sum_(m >= 1) u^(2m)/(2m + 1),
//                u = 1/(2b + 1), differentiated term by term: every term of
//                every sum has the same sign, so nothing cancels;
//   a >= 10      asymptotic: delta(a) = sum_j B_2j / (2j (2j-1) a^(2j-1)),
//                j <= 20, differentiated term by term.
// Measured and rejected: Binet's integral
//   delta^(k)(a) = (-1)^k int_0^inf t^(k-1) (1/(e^t - 1) - 1/t + 1/2) e^(-a t) dt
// by the exp-sinh rule, accurate (3-7 ulp) but 7-11 microseconds per call.
// a <= 0 or NaN gives NaN; delta(inf) and its derivatives are 0.
//
// The same header carries the half-step log-gamma ratio, which delta makes free of
// cancellation:
//
//   R(a) = log Gamma(a + 1/2) - log Gamma(a),   a > 0,
//
//   lgamma_half_ratio(a)      R(a)
//   lgamma_half_ratio_dk(a)   R^(k)(a) = psi^(k-1)(a + 1/2) - psi^(k-1)(a), k = 1..6
//
// Every law whose shape enters through Gamma((nu + 1)/2)/Gamma(nu/2) reads it; the
// plain difference of two log-gammas there loses one digit per decade of a. The route
// is in the comment above half_series below.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::math {

namespace detail_stirling {

// B_2j / (2j (2j - 1)), j = 1..20: delta(a) = sum_j s_j a^(1-2j)
LAPLACE_HD_FORCEINLINE double asym_coef(int j) {
  constexpr double s[20] = {0.08333333333333333,   -0.002777777777777778, 0.0007936507936507937, -0.0005952380952380953,
                            0.0008417508417508417, -0.0019175269175269176, 0.00641025641025641,  -0.029550653594771242,
                            0.17964437236883057,   -1.3924322169059011,   13.402864044168393,    -156.84828462600203,
                            2193.1033333333335,    -36108.77125372499,    691472.268851313,      -15238221.539407415,
                            382900751.39141417,    -10882266035.784391,   347320283765.00226,    -12369602142269.275};
  return s[j];
}

template <int K>
LAPLACE_HD_FORCEINLINE constexpr double sign_k() {
  return K % 2 == 0 ? 1.0 : -1.0;
}

// the asymptotic series, differentiated K times
template <int K>
LAPLACE_HD double asymptotic(double a) {
  const double ia = 1.0 / a, w = ia * ia;
  double p = ia;  // a^(-(2j - 1 + K))
  for (int i = 0; i < K; ++i) p *= ia;
  double sum = 0.0;
  for (int j = 1; j <= 20; ++j, p *= w) {
    double r = 1.0;  // (2j - 1)(2j) ... (2j + K - 2)
    for (int i = 0; i < K; ++i) r *= 2 * j - 1 + i;
    const double term = asym_coef(j - 1) * r * p;
    sum += term;
    if (Kokkos::fabs(term) <= 0x1p-54 * Kokkos::fabs(sum)) break;
  }
  return sign_k<K>() * sum;
}

// g(b) = delta(b) - delta(b + 1) = (b + 1/2) log(1 + 1/b) - 1 = atanh(u)/u - 1
//      = sum_(m >= 1) u^(2m)/(2m + 1),  u = 1/(2b + 1),
// so g^(k)(b) = (-2)^k sum_m (2m)(2m+1)...(2m+k-1)/(2m+1) u^(2m+k): one sign, no cancellation
template <int K>
LAPLACE_HD double shift_term(double b) {
  const double u = 1.0 / (2.0 * b + 1.0), w = u * u;
  double p = w;  // u^(2m + K)
  for (int i = 0; i < K; ++i) p *= u;
  double sum = 0.0;
  for (int m = 1; m <= 80; ++m, p *= w) {
    double r = 1.0;
    for (int i = 0; i < K; ++i) r *= 2 * m + i;
    const double term = r / (2 * m + 1) * p;
    sum += term;
    if (term <= 0x1p-54 * sum) break;
  }
  double s = 1.0;
  for (int i = 0; i < K; ++i) s *= -2.0;
  return s * sum;
}

// delta^(K)(a) = delta^(K)(a + n) + sum_(j < n) g^(K)(a + j), a + n >= 10 by the series
template <int K>
LAPLACE_HD double recurrence(double a) {
  double s = 0.0;
  while (a < 10.0) {
    s += shift_term<K>(a);
    a += 1.0;
  }
  return s + asymptotic<K>(a);
}

// the definition
template <int K>
LAPLACE_HD double direct(double a) {
  if constexpr (K == 0) return math::lgamma(a) - (a - 0.5) * math::log(a) + a - 0.91893853320467274178;
  else if constexpr (K == 1) return math::digamma(a) - math::log(a) + 0.5 / a;
  else {
    double f2 = 1.0;  // (K-2)!
    for (int i = 2; i <= K - 2; ++i) f2 *= i;
    double ak1 = 1.0;  // a^(K-1)
    for (int i = 1; i < K; ++i) ak1 *= a;
    const double e = sign_k<K>() * (f2 / ak1 + f2 * (K - 1) / (2.0 * ak1 * a));
    return math::polygamma(K - 1, a) - e;
  }
}

template <int K>
LAPLACE_HD double value(double a) {
  if (!(a > 0.0)) return math::detail::quiet_nan();
  if (a < 1.0) return direct<K>(a);
  if (a < 10.0) return recurrence<K>(a);
  return asymptotic<K>(a);
}

// ---------------------------------------------------------------------------
// R(a) = log Gamma(a + 1/2) - log Gamma(a), the half-step log-gamma ratio
// ---------------------------------------------------------------------------
// Through delta the elementary part of the difference is free of cancellation:
//
//   R(a) = (1/2) log a + S(a) + delta(a + 1/2) - delta(a),
//   S(a) = a log(1 + 1/(2a)) - 1/2 = sum_(j >= 1) c_j a^-j,
//   c_j = (-1)^j / ((j + 1) 2^(j+1)),
//
// the series being the one of log(1 + u) - u at u = 1/(2a). The two deltas are each
// of order 1/(12 a) and their difference is smaller still, so nothing cancels there.
// The series is summed at a >= 2 and the recurrence
// R(a) = R(a + 1) - [log(a + 1/2) - log a] brings a smaller argument up, every term of a
// step being elementary and exact and every term of one sum having one sign.
//
// Where to stop the recurrence was measured against mpmath, at three thresholds. One is
// not available at all: at a = 1 the series needs 84 terms at order six, and the cap of 60
// left 6027 ulp there, which is how the cap was found. Ten looked attractive, since at
// a >= 10 both deltas are in their own asymptotic series and the series needs 18 terms
// instead of 39, but the measurement says the nine steps of the recurrence cost more than
// the twenty-one terms they save: region by region, 369 ns a call against 322 below
// a = 0.5 and 345 against 297 on (1, 2], the same above a = 10. So two it is, which also
// measures better, 1.42 ulp1 for the value and at most 3.96 ulp for the derivatives
// against 4.88 and 9.81, the sum of the steps of a longer walk weighing against a value
// that is small near its own zero.
//
// The direct difference of two log-gammas loses one digit per decade of a, since
// each of them is of size a log a while R is only (log a)/2: measured against
// mpmath, 210 ulp at a = 125 and 3.6e8 ulp at a = 1e8, against at most 2 ulp
// here (docs/numerics.md, 2026-11-20).

// d^K/da^K of S(a), the series summed until a term cannot move the sum
template <int K>
LAPLACE_HD double half_series(double a) {
  const double ia = 1.0 / a;
  double p = ia;  // a^-(j + K) at j = 1
  for (int i = 0; i < K; ++i) p *= ia;
  double h = 0.25;  // 2^-(j + 1) at j = 1
  double sum = 0.0;
  for (int j = 1; j <= 60; ++j) {
    double r = 1.0;  // (j)_K = j (j + 1) ... (j + K - 1)
    for (int i = 0; i < K; ++i) r *= double(j + i);
    const double t = ((j % 2) ? -1.0 : 1.0) * h * r * p / double(j + 1);
    sum += t;
    if (Kokkos::fabs(t) <= 0x1p-56 * Kokkos::fabs(sum)) break;
    h *= 0.5;
    p *= ia;
  }
  return sign_k<K>() * sum;
}

// (K - 1)!
template <int K>
LAPLACE_HD_FORCEINLINE constexpr double fact_km1() {
  double f = 1.0;
  for (int i = 2; i < K; ++i) f *= double(i);
  return f;
}

// R(a) and its derivatives for a >= 1
template <int K>
LAPLACE_HD double half_core(double a) {
  double head;
  if constexpr (K == 0) {
    head = 0.5 * math::log(a);
  } else {
    double ak = a;
    for (int i = 1; i < K; ++i) ak *= a;
    head = 0.5 * (K % 2 == 1 ? 1.0 : -1.0) * fact_km1<K>() / ak;
  }
  return head + half_series<K>(a) + (value<K>(a + 0.5) - value<K>(a));
}

// d^K/da^K [log(a + 1/2) - log a], exact
template <int K>
LAPLACE_HD double half_step(double a) {
  if constexpr (K == 0) {
    return Kokkos::log1p(0.5 / a);
  } else {
    double u = a + 0.5, v = a;
    for (int i = 1; i < K; ++i) u *= (a + 0.5), v *= a;
    return (K % 2 == 1 ? 1.0 : -1.0) * fact_km1<K>() * (1.0 / u - 1.0 / v);
  }
}

template <int K>
LAPLACE_HD double half_value(double a) {
  if (!(a > 0.0)) return math::detail::quiet_nan();
  double b = a, sum = 0.0;  // at most two steps, since a > 0
  while (b < 2.0) {
    sum += half_step<K>(b);
    b += 1.0;
  }
  return half_core<K>(b) - sum;
}

}  // namespace detail_stirling

// Stirling's error delta(a) = log Gamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)
/**
 * \ingroup stirling
 * \brief Stirling's error \f$\delta(a) = \log \Gamma(a) - (a - \tfrac{1}{2})\log a + a - \tfrac{1}{2}\log(2\pi)\f$.
 *
 * Computes Stirling's error, the remainder of Stirling's asymptotic series for \f$\log \Gamma(a)\f$:
 * \f[
 * \delta(a) = \log \Gamma(a) - \left(a - \frac{1}{2}\right) \log a + a - \frac{1}{2} \log(2\pi),
 * \f]
 * with \f$\delta(a) \sim \frac{1}{12a}\f$ as \f$a \to \infty\f$ and \f$\delta(a) \sim -\frac{1}{2}\log a\f$ as \f$a \to 0^+\f$.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of Stirling's error \f$\delta(a)\f$.
 *
 * Returns \f$1 - \frac{1}{2}\log(2\pi) \approx 0.08106146679532726\f$ at \f$a = 1\f$; returns \f$0.0\f$ at \f$a = +\infty\f$ and quiet NaN for \f$a \le 0\f$ or NaN (shared by all functions in \ref stirling "Stirling's error").
 *
 * \par Accuracy
 * Maximum error 13.2 ulp (gcc) and 12.0 ulp (CUDA); test bound 20.0 ulp.
 *
 * \see stirling, stirling_error_d1()
 */
LAPLACE_HD_FORCEINLINE double stirling_error(double a) { return detail_stirling::value<0>(a); }
// its derivatives delta^(k)(a), k = 1..6
/**
 * \ingroup stirling
 * \brief First derivative of Stirling's error, \f$\delta'(a) = \psi(a) - \log a + \frac{1}{2a}\f$.
 *
 * Computes \f$\delta'(a) = \frac{\mathrm{d}}{\mathrm{d}a} \delta(a) = \psi(a) - \log a + \frac{1}{2a}\f$, where \f$\psi(a)\f$ is the digamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta'(a)\f$.
 *
 * Returns \f$\frac{1}{2} - \gamma \approx -0.07721566490153286\f$ at \f$a = 1\f$, where \f$\gamma\f$ is Euler's constant; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 12.7 ulp (gcc) and 13.4 ulp (CUDA); test bound 19.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d2()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d1(double a) { return detail_stirling::value<1>(a); }
/**
 * \ingroup stirling
 * \brief Second derivative of Stirling's error, \f$\delta''(a) = \psi'(a) - \frac{1}{a} - \frac{1}{2a^2}\f$.
 *
 * Computes \f$\delta''(a) = \frac{\mathrm{d}^2}{\mathrm{d}a^2} \delta(a) = \psi'(a) - \frac{1}{a} - \frac{1}{2a^2}\f$, where \f$\psi'(a)\f$ is the trigamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta''(a)\f$.
 *
 * Defined on \f$(0, \infty)\f$; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 10.8 ulp (gcc) and 10.8 ulp (CUDA); test bound 16.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d1(), stirling_error_d3()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d2(double a) { return detail_stirling::value<2>(a); }
/**
 * \ingroup stirling
 * \brief Third derivative of Stirling's error, \f$\delta^{(3)}(a) = \psi''(a) + \frac{1}{a^2} + \frac{1}{a^3}\f$.
 *
 * Computes \f$\delta^{(3)}(a) = \frac{\mathrm{d}^3}{\mathrm{d}a^3} \delta(a) = \psi''(a) + \frac{1}{a^2} + \frac{1}{a^3}\f$, where \f$\psi''(a)\f$ is the second-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(3)}(a)\f$.
 *
 * Defined on \f$(0, \infty)\f$; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 13.4 ulp (gcc) and 13.4 ulp (CUDA); test bound 20.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d2(), stirling_error_d4()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d3(double a) { return detail_stirling::value<3>(a); }
/**
 * \ingroup stirling
 * \brief Fourth derivative of Stirling's error, \f$\delta^{(4)}(a) = \psi^{(3)}(a) - \frac{2}{a^3} - \frac{3}{a^4}\f$.
 *
 * Computes \f$\delta^{(4)}(a) = \frac{\mathrm{d}^4}{\mathrm{d}a^4} \delta(a) = \psi^{(3)}(a) - \frac{2}{a^3} - \frac{3}{a^4}\f$, where \f$\psi^{(3)}(a)\f$ is the third-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(4)}(a)\f$.
 *
 * Defined on \f$(0, \infty)\f$; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 11.7 ulp (gcc) and 11.7 ulp (CUDA); test bound 17.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d3(), stirling_error_d5()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d4(double a) { return detail_stirling::value<4>(a); }
/**
 * \ingroup stirling
 * \brief Fifth derivative of Stirling's error, \f$\delta^{(5)}(a) = \psi^{(4)}(a) + \frac{6}{a^4} + \frac{12}{a^5}\f$.
 *
 * Computes \f$\delta^{(5)}(a) = \frac{\mathrm{d}^5}{\mathrm{d}a^5} \delta(a) = \psi^{(4)}(a) + \frac{6}{a^4} + \frac{12}{a^5}\f$, where \f$\psi^{(4)}(a)\f$ is the fourth-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(5)}(a)\f$.
 *
 * Defined on \f$(0, \infty)\f$; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 14.8 ulp (gcc) and 14.8 ulp (CUDA); test bound 22.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d4(), stirling_error_d6()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d5(double a) { return detail_stirling::value<5>(a); }
/**
 * \ingroup stirling
 * \brief Sixth derivative of Stirling's error, \f$\delta^{(6)}(a) = \psi^{(5)}(a) - \frac{24}{a^5} - \frac{60}{a^6}\f$.
 *
 * Computes \f$\delta^{(6)}(a) = \frac{\mathrm{d}^6}{\mathrm{d}a^6} \delta(a) = \psi^{(5)}(a) - \frac{24}{a^5} - \frac{60}{a^6}\f$, where \f$\psi^{(5)}(a)\f$ is the fifth-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(6)}(a)\f$.
 *
 * Defined on \f$(0, \infty)\f$; see \ref stirling "Stirling's error" for shared boundaries.
 *
 * \par Accuracy
 * Maximum error 14.2 ulp (gcc) and 16.1 ulp (CUDA); test bound 21.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d5()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d6(double a) { return detail_stirling::value<6>(a); }

// R(a) = log Gamma(a + 1/2) - log Gamma(a) and its derivatives R^(k)(a), k = 1..6
/**
 * \ingroup stirling
 * \brief The half-step log-gamma ratio
 * \f$R(a) = \log\Gamma(a + \tfrac{1}{2}) - \log\Gamma(a)\f$.
 *
 * Computes the logarithm of \f$\Gamma(a + \tfrac{1}{2})/\Gamma(a)\f$, with
 * \f$R(a) \sim \tfrac{1}{2}\log a\f$ as \f$a \to \infty\f$ and
 * \f$R(a) \sim -\log a\f$ as \f$a \to 0^+\f$. It is written through Stirling's error as
 * \f[
 * R(a) = \tfrac{1}{2}\log a + S(a) + \delta(a + \tfrac{1}{2}) - \delta(a), \qquad
 * S(a) = a\log\!\left(1 + \frac{1}{2a}\right) - \frac{1}{2},
 * \f]
 * with \f$S\f$ summed as the series of \f$\log(1 + u) - u\f$ at \f$u = 1/(2a)\f$ and one
 * step of the recurrence \f$R(a) = R(a + 1) - \log\frac{a + 1/2}{a}\f$ below \f$a = 1\f$.
 * The direct difference of two log-gammas instead loses a digit per decade of \f$a\f$,
 * each of them being of size \f$a\log a\f$ where \f$R\f$ is only \f$(\log a)/2\f$.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R(a)\f$.
 *
 * Returns quiet NaN for \f$a \le 0\f$ or NaN, and \f$+\infty\f$ at \f$a = +\infty\f$
 * (shared by all functions of this family).
 *
 * \par Accuracy
 * \f$R\f$ vanishes at \f$a = 1.21163\ldots\f$, where \f$\Gamma(a + \tfrac{1}{2}) =
 * \Gamma(a)\f$ astride the minimum of \f$\Gamma\f$, so the value is measured in ulp1,
 * the absolute error in units of the ulp of \f$\max(|R|, 1)\f$: maximum 1.42 ulp1 (test
 * bound 3), which is 198 ulp of \f$R\f$ itself on the reference row next to that zero.
 * The derivatives have no zero in \f$a > 0\f$ and reach 2.16 to 3.96 ulp (test bounds 4
 * to 6). The CUDA build measures 1.97 ulp1 for the value and the same figures for the
 * derivatives.
 *
 * \see stirling, stirling_error(), lgamma_half_ratio_d1()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio(double a) { return detail_stirling::half_value<0>(a); }
/**
 * \ingroup stirling
 * \brief First derivative \f$R'(a) = \psi(a + \tfrac{1}{2}) - \psi(a)\f$ of the
 * half-step log-gamma ratio.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R'(a)\f$.
 * \see stirling, lgamma_half_ratio(), lgamma_half_ratio_d2()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d1(double a) { return detail_stirling::half_value<1>(a); }
/**
 * \ingroup stirling
 * \brief Second derivative \f$R''(a) = \psi'(a + \tfrac{1}{2}) - \psi'(a)\f$ of the
 * half-step log-gamma ratio.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R''(a)\f$.
 * \see stirling, lgamma_half_ratio_d1(), lgamma_half_ratio_d3()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d2(double a) { return detail_stirling::half_value<2>(a); }
/**
 * \ingroup stirling
 * \brief Third derivative \f$R^{(3)}(a) = \psi''(a + \tfrac{1}{2}) - \psi''(a)\f$ of the
 * half-step log-gamma ratio.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R^{(3)}(a)\f$.
 * \see stirling, lgamma_half_ratio_d2(), lgamma_half_ratio_d4()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d3(double a) { return detail_stirling::half_value<3>(a); }
/**
 * \ingroup stirling
 * \brief Fourth derivative of the half-step log-gamma ratio,
 * \f$R^{(4)}(a) = \psi^{(3)}(a + \tfrac{1}{2}) - \psi^{(3)}(a)\f$.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R^{(4)}(a)\f$.
 * \see stirling, lgamma_half_ratio_d3(), lgamma_half_ratio_d5()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d4(double a) { return detail_stirling::half_value<4>(a); }
/**
 * \ingroup stirling
 * \brief Fifth derivative of the half-step log-gamma ratio,
 * \f$R^{(5)}(a) = \psi^{(4)}(a + \tfrac{1}{2}) - \psi^{(4)}(a)\f$.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R^{(5)}(a)\f$.
 * \see stirling, lgamma_half_ratio_d4(), lgamma_half_ratio_d6()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d5(double a) { return detail_stirling::half_value<5>(a); }
/**
 * \ingroup stirling
 * \brief Sixth derivative of the half-step log-gamma ratio,
 * \f$R^{(6)}(a) = \psi^{(5)}(a + \tfrac{1}{2}) - \psi^{(5)}(a)\f$.
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$R^{(6)}(a)\f$.
 * \see stirling, lgamma_half_ratio_d5()
 */
LAPLACE_HD_FORCEINLINE double lgamma_half_ratio_d6(double a) { return detail_stirling::half_value<6>(a); }

}  // namespace laplace::math
