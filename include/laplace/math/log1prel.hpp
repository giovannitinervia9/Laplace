// log1prel(x) = log(1 + x) / x and its derivatives psi^(k)(x), k = 1..6 (laplace::math):
//
//   log1prel(x)      psi(x) = log(1 + x) / x, psi(0) = 1
//   log1prel_dk(x)   psi^(k)(x), k = 1..6
//
// Notation: x is a real argument with x >= -1; psi^(k) is the k-th derivative of psi with respect
// to x; the Gauss-Legendre nodes t_i and weights w_i (i = 0..31) are those of the 32-point rule on
// [0, 1]. Equivalently psi(x) = int_0^1 dt / (1 + x t) and
//   psi^(k)(x) = (-1)^k k! int_0^1 t^k / (1 + x t)^(k+1) dt,
// whose integrand is positive for every x > -1, so psi^(k) has the sign (-1)^k for every x > -1.
// The quadrature sums terms of one sign and does not cancel; the series and the recursion do, and
// the note measures how much.
//
// Five regions, each with its own method (the switches are measured in docs/numerics.md):
//   |x| < 1/4          power series psi^(k)(x) = sum_(m >= 0) t_m, t_0 = (-1)^k k!/(k+1),
//                      t_(m+1) = -x (m+k+1)^2 / ((m+1)(m+k+2)) t_m, summed until |t_m| <= 4 eps |sum|
//                      (iteration cap 300); exact at x = 0.
//   -4/5 < x <= 6      Gauss-Legendre quadrature of the integral above with 32 nodes
//                      (the integrand has a pole at t = -1/x, which the rule resolves for these x).
//   6 < x <= 20        the same quadrature split at t = 1/4, 32 nodes on each part (the
//                      recursion loses up to 17 ulp at orders 5-6 near x = 8-9; measured
//                      2026-11-12, benchmarks/math_rel.cpp: at most 5.5 ulp, 65 ns against
//                      26 ns per call on one gcc thread).
//   x <= -4/5, x > 20  forward recursion from psi (below), which is stable there.
//   x = -1, x = +inf   limits (see the special values).
//
// The recursion follows from x psi(x) = log1p(x), differentiated k times:
//   x psi^(k)(x) + k psi^(k-1)(x) = (-1)^(k-1) (k-1)! / (1 + x)^k,
// so psi^(k) = (T_k - k psi^(k-1)) / x with T_k = (-1)^(k-1) (k-1)! u^k and u = 1 / (1 + x),
// started from psi^(0) = log1p(x) / x. The power u^k is a product of k factors of u, and 1/(1 + x)
// is formed in double: for x near -1 it is large (1 + x is at least 2^-53 for a double x, so u is at
// most 2^53), and the term of order one is then the only one that matters; for large x the powers
// underflow, and the psi^(k) they feed are then below the normal range, which is outside the
// promised accuracy.
//
// The Gauss-Legendre table is written as literal constants (values rounded to double from the
// nodes computed to 60 digits by Newton's method on the Legendre polynomial of degree 32); on the
// device it is a __constant__ copy, as in normal.hpp.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::math {

namespace detail_log1prel {

constexpr double eps = 2.220446049250313e-16;  // 2^-52, the spacing of doubles at 1

// 32-point Gauss-Legendre rule on [0, 1]: nodes and weights (their sum is 1)
#define LAPLACE_LOG1PREL_GL_T                                                                              \
  0.0013680690752592183, 0.007194244227365833, 0.017618872206246784, 0.03254696203113015,                  \
      0.05183942211697394, 0.07531619313371501, 0.1027581020160288, 0.13390894062985517,                   \
      0.1684778665348924, 0.20614212137961885, 0.2465500455338853, 0.2893243619346823,                     \
      0.33406569885893617, 0.38035631887393145, 0.42776401920860174, 0.4758461671561308,                   \
      0.5241538328438692, 0.5722359807913983, 0.6196436811260685, 0.6659343011410638,                      \
      0.7106756380653176, 0.7534499544661147, 0.7938578786203812, 0.8315221334651076,                      \
      0.8660910593701449, 0.8972418979839712, 0.9246838068662849, 0.9481605778830261,                      \
      0.9674530379688698, 0.9823811277937532, 0.9928057557726342, 0.9986319309247408

#define LAPLACE_LOG1PREL_GL_W                                                                              \
  0.003509305004735048, 0.008137197365452835, 0.01269603265463103, 0.017136931456510716,                  \
      0.02141794901111334, 0.025499029631188087, 0.029342046739267772, 0.032911111388180925,               \
      0.03617289705442425, 0.039096947893535156, 0.041655962113473374, 0.043826046502201906,               \
      0.045586939347881945, 0.04692219954040228, 0.04781936003963743, 0.0482700442573639,                  \
      0.0482700442573639, 0.04781936003963743, 0.04692219954040228, 0.045586939347881945,                  \
      0.043826046502201906, 0.041655962113473374, 0.039096947893535156, 0.03617289705442425,               \
      0.032911111388180925, 0.029342046739267772, 0.025499029631188087, 0.02141794901111334,               \
      0.017136931456510716, 0.01269603265463103, 0.008137197365452835, 0.003509305004735048

// clang-format off
#if defined(__CUDACC__)
static __constant__ double gl_t_dev[32] = {LAPLACE_LOG1PREL_GL_T};
static __constant__ double gl_w_dev[32] = {LAPLACE_LOG1PREL_GL_W};
#endif
inline constexpr double gl_t_host[32] = {LAPLACE_LOG1PREL_GL_T};
inline constexpr double gl_w_host[32] = {LAPLACE_LOG1PREL_GL_W};
// clang-format on

LAPLACE_HD_FORCEINLINE double gl_t(int i) {
  KOKKOS_IF_ON_DEVICE((return gl_t_dev[i];))
  KOKKOS_IF_ON_HOST((return gl_t_host[i];))
}
LAPLACE_HD_FORCEINLINE double gl_w(int i) {
  KOKKOS_IF_ON_DEVICE((return gl_w_dev[i];))
  KOKKOS_IF_ON_HOST((return gl_w_host[i];))
}

// k! for k = 0..6, and (-1)^k
LAPLACE_HD_FORCEINLINE double factorial_small(int k) {
  constexpr double f[7] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0};
  return f[k];
}
LAPLACE_HD_FORCEINLINE double sign_of(int k) { return (k & 1) ? -1.0 : 1.0; }

// psi^(K)(x) for |x| < 1/4 by the power series, the factor -x (m+K+1)^2/((m+1)(m+K+2))
// between consecutive terms; the terms decrease geometrically there
template <int K>
LAPLACE_HD double series(double x) {
  double t = sign_of(K) * factorial_small(K) / (K + 1.0);
  double s = t;
  for (int m = 0; m < 300; ++m) {
    const double a = m + K + 1.0;
    t *= -x * a * a / ((m + 1.0) * (m + K + 2.0));
    s += t;
    if (Kokkos::fabs(t) <= 4.0 * eps * Kokkos::fabs(s)) break;
  }
  return s;
}

// psi^(K)(x) for -4/5 < x <= 6 by the 32-point rule on the integral; every term is positive, the
// integrand being t^K (1 + x t)^(-(K+1)) with 1 + x t >= 1/5 on the range
template <int K>
LAPLACE_HD double quadrature(double x) {
  double s = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double t = gl_t(i);
    const double d = 1.0 + x * t;
    const double r = t / d;  // t^K / (1 + x t)^K = r^K
    double p = r;
    for (int j = 1; j < K; ++j) p *= r;
    s += gl_w(i) * p / d;
  }
  return sign_of(K) * factorial_small(K) * s;
}

// psi^(K)(x) by the same integral split at t = 1/4, with the 32-point rule on each of [0, 1/4]
// and [1/4, 1] (64 nodes): for 6 < x <= 20 the integrand t^K (1 + x t)^(-(K+1)) peaks near
// t = K/x, which one rule on [0, 1] does not resolve
template <int K>
LAPLACE_HD double quadrature_split(double x) {
  double s = 0.0;
  for (int panel = 0; panel < 2; ++panel) {
    const double a = panel == 0 ? 0.0 : 0.25, w = panel == 0 ? 0.25 : 0.75;
    for (int i = 0; i < 32; ++i) {
      const double t = a + w * gl_t(i);
      const double d = 1.0 + x * t;
      const double r = t / d;
      double p = r;
      for (int j = 1; j < K; ++j) p *= r;
      s += w * gl_w(i) * p / d;
    }
  }
  return sign_of(K) * factorial_small(K) * s;
}

// psi^(K)(x) by the forward recursion from psi, for x <= -4/5 and x > 20
template <int K>
LAPLACE_HD double recursion(double x) {
  double p = Kokkos::log1p(x) / x;
  const double u = 1.0 / (1.0 + x);
  for (int j = 1; j <= K; ++j) {
    double pw = u;
    for (int m = 1; m < j; ++m) pw *= u;
    const double t = -sign_of(j) * factorial_small(j - 1) * pw;  // (-1)^(j-1) (j-1)! u^j
    p = (t - j * p) / x;
  }
  return p;
}

// the region of x for the derivative of order K; NaN for x < -1 and for NaN
template <int K>
LAPLACE_HD double deriv(double x) {
  if (!(x >= -1.0)) return detail::quiet_nan();
  if (x == -1.0) return sign_of(K) * detail::pos_inf();  // (-1)^K inf: the sign of psi^(K)
  if (x == detail::pos_inf()) return 0.0;
  if (Kokkos::fabs(x) < 0.25) return series<K>(x);
  if (x > -0.8 && x <= 6.0) return quadrature<K>(x);
  if (x > 6.0 && x <= 20.0) return quadrature_split<K>(x);
  return recursion<K>(x);
}

// psi^(k)(x) for every k = 1..n at once (n <= 6), in out[1..n]; out[0] is not written. The same
// methods and regions as deriv<K>, with the work shared: one pass over the quadrature nodes and
// one recursion give every order (the series stays one loop per order). In the quadrature each
// node forms r = t/d and r/d, d = 1 + x t, and the orders by products with r: two divisions per
// node instead of n + 1, and the same number of roundings as quadrature<K>. The results are then
// not bit-identical to log1prel_dk there, and are within its measured errors (docs/numerics.md,
// section on the refinements of the Box-Cox, Yeo-Johnson and softplus bijectors). For callers
// that need several orders at the same point (Box-Cox, Yeo-Johnson).
LAPLACE_HD void derivs_upto(int n, double x, double* out) {
  if (!(x >= -1.0) || x == -1.0 || x == detail::pos_inf() || Kokkos::fabs(x) < 0.25) {
    for (int k = 1; k <= n; ++k) {
      switch (k) {
        case 1: out[1] = deriv<1>(x); break;
        case 2: out[2] = deriv<2>(x); break;
        case 3: out[3] = deriv<3>(x); break;
        case 4: out[4] = deriv<4>(x); break;
        case 5: out[5] = deriv<5>(x); break;
        default: out[6] = deriv<6>(x); break;
      }
    }
    return;
  }
  if (x > -0.8 && x <= 20.0) {
    double s[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int panels = x <= 6.0 ? 1 : 2;
    for (int panel = 0; panel < panels; ++panel) {
      const double a = panels == 1 ? 0.0 : (panel == 0 ? 0.0 : 0.25);
      const double w = panels == 1 ? 1.0 : (panel == 0 ? 0.25 : 0.75);
      for (int i = 0; i < 32; ++i) {
        const double t = panels == 1 ? gl_t(i) : a + w * gl_t(i);
        const double d = 1.0 + x * t;
        const double r = t / d;
        const double wi = panels == 1 ? gl_w(i) : w * gl_w(i);
        double p = r / d;  // r^k / d, k = 1..n: two divisions per node instead of n + 1
        for (int k = 1; k <= n; ++k) {
          if (k > 1) p *= r;
          s[k] += wi * p;
        }
      }
    }
    for (int k = 1; k <= n; ++k) out[k] = sign_of(k) * factorial_small(k) * s[k];
    return;
  }
  double p = Kokkos::log1p(x) / x;
  const double u = 1.0 / (1.0 + x);
  for (int j = 1; j <= n; ++j) {
    double pw = u;
    for (int m = 1; m < j; ++m) pw *= u;
    const double t = -sign_of(j) * factorial_small(j - 1) * pw;
    p = (t - j * p) / x;
    out[j] = p;
  }
}

}  // namespace detail_log1prel

/**
 * \ingroup log1prel
 * \brief Relative logarithm function, \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the relative logarithm function:
 * \f[
 * \psi(x) = \frac{\log(1 + x)}{x} = \int_0^1 \frac{\mathrm{d}t}{1 + x t}
 * \f]
 * with \f$\psi(0) = 1\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$1.0\f$.
 * - \f$x = -1\f$: returns \f$+\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 1.28 ulp (gcc) / 1.28 ulp (CUDA); test bound 2.0 ulp.
 *
 * \see log1prel, log1prel_d1()
 */
LAPLACE_HD double log1prel(double x) {
  if (!(x >= -1.0)) return detail::quiet_nan();
  if (x == -1.0) return detail::pos_inf();
  if (x == detail::pos_inf()) return 0.0;
  if (x == 0.0) return 1.0;
  return Kokkos::log1p(x) / x;
}

/**
 * \ingroup log1prel
 * \brief First derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the first derivative:
 * \f[
 * \psi'(x) = \frac{\mathrm{d}}{\mathrm{d}x} \psi(x) = -\int_0^1 \frac{t}{(1 + x t)^2} \, \mathrm{d}t = \frac{\frac{1}{1 + x} - \psi(x)}{x}
 * \f]
 * with \f$\psi'(0) = -1/2\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi'(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$-0.5\f$.
 * - \f$x = -1\f$: returns \f$-\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 3.94 ulp (gcc) / 3.94 ulp (CUDA); test bound 6.0 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d2()
 */
LAPLACE_HD double log1prel_d1(double x) { return detail_log1prel::deriv<1>(x); }

/**
 * \ingroup log1prel
 * \brief Second derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the second derivative:
 * \f[
 * \psi''(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \psi(x) = 2 \int_0^1 \frac{t^2}{(1 + x t)^3} \, \mathrm{d}t = \frac{-\frac{1}{(1 + x)^2} - 2\psi'(x)}{x}
 * \f]
 * with \f$\psi''(0) = 2/3\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi''(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$2/3\f$.
 * - \f$x = -1\f$: returns \f$+\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.03 ulp (gcc) / 5.03 ulp (CUDA); test bound 7.6 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d1(), log1prel_d3()
 */
LAPLACE_HD double log1prel_d2(double x) { return detail_log1prel::deriv<2>(x); }

/**
 * \ingroup log1prel
 * \brief Third derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the third derivative:
 * \f[
 * \psi^{(3)}(x) = \frac{\mathrm{d}^3}{\mathrm{d}x^3} \psi(x) = -6 \int_0^1 \frac{t^3}{(1 + x t)^4} \, \mathrm{d}t = \frac{\frac{2}{(1 + x)^3} - 3\psi''(x)}{x}
 * \f]
 * with \f$\psi^{(3)}(0) = -3/2\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi^{(3)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$-1.5\f$.
 * - \f$x = -1\f$: returns \f$-\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 6.39 ulp (gcc) / 6.39 ulp (CUDA); test bound 9.6 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d2(), log1prel_d4()
 */
LAPLACE_HD double log1prel_d3(double x) { return detail_log1prel::deriv<3>(x); }

/**
 * \ingroup log1prel
 * \brief Fourth derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * \psi^{(4)}(x) = \frac{\mathrm{d}^4}{\mathrm{d}x^4} \psi(x) = 24 \int_0^1 \frac{t^4}{(1 + x t)^5} \, \mathrm{d}t = \frac{-\frac{6}{(1 + x)^4} - 4\psi^{(3)}(x)}{x}
 * \f]
 * with \f$\psi^{(4)}(0) = 24/5 = 4.8\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi^{(4)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$4.8\f$.
 * - \f$x = -1\f$: returns \f$+\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 7.06 ulp (gcc) / 7.06 ulp (CUDA); test bound 10.6 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d3(), log1prel_d5()
 */
LAPLACE_HD double log1prel_d4(double x) { return detail_log1prel::deriv<4>(x); }

/**
 * \ingroup log1prel
 * \brief Fifth derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * \psi^{(5)}(x) = \frac{\mathrm{d}^5}{\mathrm{d}x^5} \psi(x) = -120 \int_0^1 \frac{t^5}{(1 + x t)^6} \, \mathrm{d}t = \frac{\frac{24}{(1 + x)^5} - 5\psi^{(4)}(x)}{x}
 * \f]
 * with \f$\psi^{(5)}(0) = -20.0\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi^{(5)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$-20.0\f$.
 * - \f$x = -1\f$: returns \f$-\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 8.27 ulp (gcc) / 8.27 ulp (CUDA); test bound 12.4 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d4(), log1prel_d6()
 */
LAPLACE_HD double log1prel_d5(double x) { return detail_log1prel::deriv<5>(x); }

/**
 * \ingroup log1prel
 * \brief Sixth derivative in \f$x\f$ of \f$\psi(x) = \log(1 + x)/x\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * \psi^{(6)}(x) = \frac{\mathrm{d}^6}{\mathrm{d}x^6} \psi(x) = 720 \int_0^1 \frac{t^6}{(1 + x t)^7} \, \mathrm{d}t = \frac{-\frac{120}{(1 + x)^6} - 6\psi^{(5)}(x)}{x}
 * \f]
 * with \f$\psi^{(6)}(0) = 720/7\f$.
 *
 * \param x Real argument \f$x > -1\f$.
 * \return The value of \f$\psi^{(6)}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > -1\f$: standard domain.
 * - \f$x = 0\f$: returns \f$720/7 \approx 102.857\f$.
 * - \f$x = -1\f$: returns \f$+\infty\f$.
 * - \f$x < -1\f$, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 9.60 ulp (gcc) / 9.60 ulp (CUDA); test bound 14.4 ulp.
 *
 * \see log1prel, log1prel(), log1prel_d5()
 */
LAPLACE_HD double log1prel_d6(double x) { return detail_log1prel::deriv<6>(x); }

}  // namespace laplace::math
