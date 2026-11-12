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

}  // namespace detail_stirling

// Stirling's error delta(a) = log Gamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)
/**
 * \ingroup stirling
 * \brief Stirling's error \f$\delta(a) = \log \Gamma(a) - (a - \tfrac{1}{2})\log a + a - \tfrac{1}{2}\log(2\pi)\f$.
 *
 * Computes Stirling's error, the remainder of Stirling's asymptotic series for \f$\log \Gamma\f$:
 * \f[
 * \delta(a) = \log \Gamma(a) - \left(a - \frac{1}{2}\right) \log a + a - \frac{1}{2} \log(2\pi)
 * \f]
 * where \f$a \in (0, \infty)\f$ is the positive real argument.
 *
 * As \f$a \to \infty\f$, \f$\delta(a) \sim \frac{1}{12a}\f$; as \f$a \to 0\f$, \f$\delta(a) \sim -\frac{1}{2}\log a\f$.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of Stirling's error \f$\delta(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a = 1\f$: returns \f$1 - \frac{1}{2}\log(2\pi) \approx 0.08106146679532726\f$.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 13.2 ulp (gcc) / 12.0 ulp (CUDA); test bound 20.0 ulp.
 *
 * \see stirling, stirling_error_d1()
 */
LAPLACE_HD_FORCEINLINE double stirling_error(double a) { return detail_stirling::value<0>(a); }
// its derivatives delta^(k)(a), k = 1..6
/**
 * \ingroup stirling
 * \brief First derivative of Stirling's error, \f$\delta'(a) = \psi(a) - \log a + \frac{1}{2a}\f$.
 *
 * Computes the first derivative:
 * \f[
 * \delta'(a) = \frac{\mathrm{d}}{\mathrm{d}a} \delta(a) = \psi(a) - \log a + \frac{1}{2a}
 * \f]
 * with \f$\delta'(1) = \frac{1}{2} - \gamma\f$ where \f$\gamma\f$ is Euler's constant.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta'(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a = 1\f$: returns \f$\frac{1}{2} - \gamma \approx -0.07721566490153286\f$.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 12.7 ulp (gcc) / 13.4 ulp (CUDA); test bound 19.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d2()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d1(double a) { return detail_stirling::value<1>(a); }
/**
 * \ingroup stirling
 * \brief Second derivative of Stirling's error, \f$\delta''(a) = \psi'(a) - \frac{1}{a} - \frac{1}{2a^2}\f$.
 *
 * Computes the second derivative:
 * \f[
 * \delta''(a) = \frac{\mathrm{d}^2}{\mathrm{d}a^2} \delta(a) = \psi'(a) - \frac{1}{a} - \frac{1}{2a^2}
 * \f]
 * where \f$\psi'(a)\f$ is the trigamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta''(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 10.8 ulp (gcc) / 10.8 ulp (CUDA); test bound 16.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d1(), stirling_error_d3()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d2(double a) { return detail_stirling::value<2>(a); }
/**
 * \ingroup stirling
 * \brief Third derivative of Stirling's error, \f$\delta^{(3)}(a) = \psi''(a) + \frac{1}{a^2} + \frac{1}{a^3}\f$.
 *
 * Computes the third derivative:
 * \f[
 * \delta^{(3)}(a) = \frac{\mathrm{d}^3}{\mathrm{d}a^3} \delta(a) = \psi''(a) + \frac{1}{a^2} + \frac{1}{a^3}
 * \f]
 * where \f$\psi''(a)\f$ is the second-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(3)}(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 13.4 ulp (gcc) / 13.4 ulp (CUDA); test bound 20.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d2(), stirling_error_d4()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d3(double a) { return detail_stirling::value<3>(a); }
/**
 * \ingroup stirling
 * \brief Fourth derivative of Stirling's error, \f$\delta^{(4)}(a) = \psi^{(3)}(a) - \frac{2}{a^3} - \frac{3}{a^4}\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * \delta^{(4)}(a) = \frac{\mathrm{d}^4}{\mathrm{d}a^4} \delta(a) = \psi^{(3)}(a) - \frac{2}{a^3} - \frac{3}{a^4}
 * \f]
 * where \f$\psi^{(3)}(a)\f$ is the third-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(4)}(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 11.7 ulp (gcc) / 11.7 ulp (CUDA); test bound 17.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d3(), stirling_error_d5()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d4(double a) { return detail_stirling::value<4>(a); }
/**
 * \ingroup stirling
 * \brief Fifth derivative of Stirling's error, \f$\delta^{(5)}(a) = \psi^{(4)}(a) + \frac{6}{a^4} + \frac{12}{a^5}\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * \delta^{(5)}(a) = \frac{\mathrm{d}^5}{\mathrm{d}a^5} \delta(a) = \psi^{(4)}(a) + \frac{6}{a^4} + \frac{12}{a^5}
 * \f]
 * where \f$\psi^{(4)}(a)\f$ is the fourth-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(5)}(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 14.8 ulp (gcc) / 14.8 ulp (CUDA); test bound 22.0 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d4(), stirling_error_d6()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d5(double a) { return detail_stirling::value<5>(a); }
/**
 * \ingroup stirling
 * \brief Sixth derivative of Stirling's error, \f$\delta^{(6)}(a) = \psi^{(5)}(a) - \frac{24}{a^5} - \frac{60}{a^6}\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * \delta^{(6)}(a) = \frac{\mathrm{d}^6}{\mathrm{d}a^6} \delta(a) = \psi^{(5)}(a) - \frac{24}{a^5} - \frac{60}{a^6}
 * \f]
 * where \f$\psi^{(5)}(a)\f$ is the fifth-order polygamma function.
 *
 * \param a Positive argument \f$a \in (0, \infty)\f$.
 * \return The value of \f$\delta^{(6)}(a)\f$.
 *
 * \par Domain and special values
 * - \f$a \in (0, \infty)\f$: standard domain.
 * - \f$a \le 0\f$ or \f$a\f$ is NaN: returns quiet NaN.
 * - \f$a = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 14.2 ulp (gcc) / 16.1 ulp (CUDA); test bound 21.5 ulp.
 *
 * \see stirling, stirling_error(), stirling_error_d5()
 */
LAPLACE_HD_FORCEINLINE double stirling_error_d6(double a) { return detail_stirling::value<6>(a); }

}  // namespace laplace::math
