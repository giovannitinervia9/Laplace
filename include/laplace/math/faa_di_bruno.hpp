// Faa di Bruno's formula and the partial Bell polynomials, to order 7
// (laplace::math):
//
//   bell_partial(n, k, g)   B_(n,k)(g_1, ..., g_(n-k+1)), 1 <= k <= n <= 7
//   chain_rule_d1 ... _d7(f, g)
//                           d^N/dx^N f(g(x)) = sum_(k=1..N) f^(k)(g(x)) B_(N,k)(g'(x), ..., g^(N-k+1)(x))
//
// The arrays are indexed by the derivative order: f[k] = f^(k) at g(x) and
// g[j] = g^(j) at x, k, j = 1..N (element 0, the value, is not read). The
// distributions use this to carry their derivatives in a parameter theta to
// the linear predictor eta of a link, theta = h(eta) (f the log-density in
// theta, g = h); mixed derivatives in several parameters, each with its own
// link, are sums of products of the B_(n,k) of each parameter.
//
// The polynomials are written out (from sympy's bell(n, k, ...)), each a sum
// of monomials with positive integer coefficients; chain_rule_dN sums
// f[k] B_(N,k) from k = N down to 1.
#pragma once

#include <laplace/core/config.hpp>

namespace laplace::math {

// B_(n,k)(g_1, ..., g_(n-k+1)); g[j] = g_j (g[0] is not read); 0 outside 1 <= k <= n <= 7
/**
 * \ingroup faa_di_bruno
 * \brief Partial Bell polynomial \f$B_{n,k}(g_1, \dots, g_{n-k+1})\f$ for \f$1 \le k \le n \le 7\f$.
 *
 * Evaluates the explicit sum of monomials with positive integer coefficients for \f$B_{n,k}(g_1, \dots, g_{n-k+1})\f$ when \f$1 \le k \le n \le 7\f$.
 *
 * \param n Degree of the polynomial, evaluated for \f$1 \le n \le 7\f$.
 * \param k Order of the polynomial, evaluated for \f$1 \le k \le n\f$.
 * \param g Pointer to at least \f$n - k + 2\f$ doubles when \f$1 \le k \le n \le 7\f$, storing \f$g_m\f$ in `g[m]` for \f$m = 1, \dots, n - k + 1\f$. Element `g[0]` is not read.
 * \return Value of \f$B_{n,k}(g_1, \dots, g_{n-k+1})\f$ for \f$1 \le k \le n \le 7\f$, or \f$0.0\f$ outside that range.
 *
 * Outside \f$1 \le k \le n \le 7\f$, the function returns \f$0.0\f$ without reading `g`. If any required element of `g` is NaN, it returns quiet NaN. Automated tests verify exact equality against the Stirling and binomial identities documented on \ref faa_di_bruno "the module page".
 *
 * \see faa_di_bruno, chain_rule_d1(), chain_rule_d2(), chain_rule_d3(), chain_rule_d4(), chain_rule_d5(), chain_rule_d6(), chain_rule_d7()
 */
LAPLACE_HD double bell_partial(int n, int k, const double* g) {
  // outside 1 <= k <= n <= 7 the key n * 8 + k could alias a valid case
  // (n = 1, k = 9 gives the key of n = 2, k = 1)
  if (n < 1 || n > 7 || k < 1 || k > n) return 0.0;
  const double g1 = g[1];
  switch (n * 8 + k) {
    case 1 * 8 + 1: return g1;
    case 2 * 8 + 1: return g[2];
    case 2 * 8 + 2: return g1 * g1;
    case 3 * 8 + 1: return g[3];
    case 3 * 8 + 2: return 3.0 * g1 * g[2];
    case 3 * 8 + 3: return g1 * g1 * g1;
    case 4 * 8 + 1: return g[4];
    case 4 * 8 + 2: return 4.0 * g1 * g[3] + 3.0 * g[2] * g[2];
    case 4 * 8 + 3: return 6.0 * g1 * g1 * g[2];
    case 4 * 8 + 4: return (g1 * g1) * (g1 * g1);
    case 5 * 8 + 1: return g[5];
    case 5 * 8 + 2: return 5.0 * g1 * g[4] + 10.0 * g[2] * g[3];
    case 5 * 8 + 3: return 10.0 * g1 * g1 * g[3] + 15.0 * g1 * g[2] * g[2];
    case 5 * 8 + 4: return 10.0 * (g1 * g1 * g1) * g[2];
    case 5 * 8 + 5: return (g1 * g1) * (g1 * g1) * g1;
    case 6 * 8 + 1: return g[6];
    case 6 * 8 + 2: return 6.0 * g1 * g[5] + 15.0 * g[2] * g[4] + 10.0 * g[3] * g[3];
    case 6 * 8 + 3: return 15.0 * g1 * g1 * g[4] + 60.0 * g1 * g[2] * g[3] + 15.0 * g[2] * g[2] * g[2];
    case 6 * 8 + 4: return 20.0 * (g1 * g1 * g1) * g[3] + 45.0 * (g1 * g1) * (g[2] * g[2]);
    case 6 * 8 + 5: return 15.0 * ((g1 * g1) * (g1 * g1)) * g[2];
    case 6 * 8 + 6: return (g1 * g1 * g1) * (g1 * g1 * g1);
    case 7 * 8 + 1: return g[7];
    case 7 * 8 + 2: return 7.0 * g1 * g[6] + 21.0 * g[2] * g[5] + 35.0 * g[3] * g[4];
    case 7 * 8 + 3:
      return 21.0 * g1 * g1 * g[5] + 105.0 * g1 * g[2] * g[4] + 70.0 * g1 * g[3] * g[3] + 105.0 * g[2] * g[2] * g[3];
    case 7 * 8 + 4:
      return 35.0 * (g1 * g1 * g1) * g[4] + 210.0 * (g1 * g1) * g[2] * g[3] + 105.0 * g1 * (g[2] * g[2] * g[2]);
    case 7 * 8 + 5: return 35.0 * ((g1 * g1) * (g1 * g1)) * g[3] + 105.0 * (g1 * g1 * g1) * (g[2] * g[2]);
    case 7 * 8 + 6: return 21.0 * ((g1 * g1) * (g1 * g1) * g1) * g[2];
    case 7 * 8 + 7: return (g1 * g1 * g1) * (g1 * g1 * g1) * g1;
    default: return 0.0;
  }
}

namespace detail_faa {
template <int N>
LAPLACE_HD_FORCEINLINE double compose(const double* f, const double* g) {
  double s = 0.0;
  for (int k = N; k >= 1; --k) s += f[k] * bell_partial(N, k, g);
  return s;
}
}  // namespace detail_faa

/**
 * \ingroup faa_di_bruno
 * \brief First derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the first derivative \f$\frac{\mathrm{d}}{\mathrm{d}x} f(g(x)) = f_1 g_1\f$, where \f$f_1 = f'(g(x))\f$ and \f$g_1 = g'(x)\f$.
 *
 * \param f Pointer to at least 2 doubles storing \f$f'(g(x))\f$ in `f[1]`. Element `f[0]` is not read.
 * \param g Pointer to at least 2 doubles storing \f$g'(x)\f$ in `g[1]`. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}}{\mathrm{d}x} f(g(x))\f$, or quiet NaN if `f[1]` or `g[1]` is NaN.
 *
 * Measured maximum error is 0.50 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 1.0 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d2()
 */
LAPLACE_HD double chain_rule_d1(const double* f, const double* g) { return f[1] * g[1]; }

/**
 * \ingroup faa_di_bruno
 * \brief Second derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the second derivative by Faà di Bruno's formula, summed from \f$k = 2\f$ down to \f$1\f$:
 * \f[
 * \frac{\mathrm{d}^2}{\mathrm{d}x^2} f(g(x)) = f_1 g_2 + f_2 g_1^2,
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 3 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, 2\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 3 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, 2\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^2}{\mathrm{d}x^2} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 1.73 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 3.0 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d1(), chain_rule_d3()
 */
LAPLACE_HD double chain_rule_d2(const double* f, const double* g) { return detail_faa::compose<2>(f, g); }

/**
 * \ingroup faa_di_bruno
 * \brief Third derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the third derivative by Faà di Bruno's formula, summed from \f$k = 3\f$ down to \f$1\f$:
 * \f[
 * \frac{\mathrm{d}^3}{\mathrm{d}x^3} f(g(x)) = f_1 g_3 + 3 f_2 g_1 g_2 + f_3 g_1^3,
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 4 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, 2, 3\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 4 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, 2, 3\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^3}{\mathrm{d}x^3} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 2.21 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 3.5 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d2(), chain_rule_d4()
 */
LAPLACE_HD double chain_rule_d3(const double* f, const double* g) { return detail_faa::compose<3>(f, g); }

/**
 * \ingroup faa_di_bruno
 * \brief Fourth derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the fourth derivative by Faà di Bruno's formula, summed from \f$k = 4\f$ down to \f$1\f$:
 * \f[
 * \frac{\mathrm{d}^4}{\mathrm{d}x^4} f(g(x)) = f_1 g_4 + f_2 (4 g_1 g_3 + 3 g_2^2) + 6 f_3 g_1^2 g_2 + f_4 g_1^4,
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 5 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, \dots, 4\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 5 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, \dots, 4\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^4}{\mathrm{d}x^4} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 3.08 ulp of \f$S\f$ on gcc and 3.03 ulp of \f$S\f$ on CUDA, with a test bound of 4.5 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d3(), chain_rule_d5()
 */
LAPLACE_HD double chain_rule_d4(const double* f, const double* g) { return detail_faa::compose<4>(f, g); }

/**
 * \ingroup faa_di_bruno
 * \brief Fifth derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the fifth derivative by Faà di Bruno's formula, summed from \f$k = 5\f$ down to \f$1\f$:
 * \f[
 * \frac{\mathrm{d}^5}{\mathrm{d}x^5} f(g(x)) = f_1 g_5 + f_2 (5 g_1 g_4 + 10 g_2 g_3) + f_3 (10 g_1^2 g_3 + 15 g_1 g_2^2) + 10 f_4 g_1^3 g_2 + f_5 g_1^5,
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 6 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, \dots, 5\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 6 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, \dots, 5\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^5}{\mathrm{d}x^5} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 3.05 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 4.5 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d4(), chain_rule_d6()
 */
LAPLACE_HD double chain_rule_d5(const double* f, const double* g) { return detail_faa::compose<5>(f, g); }

/**
 * \ingroup faa_di_bruno
 * \brief Sixth derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the sixth derivative by Faà di Bruno's formula, summed from \f$k = 6\f$ down to \f$1\f$:
 * \f[
 * \begin{aligned}
 * \frac{\mathrm{d}^6}{\mathrm{d}x^6} f(g(x)) = {} & f_1 g_6 + f_2 (6 g_1 g_5 + 15 g_2 g_4 + 10 g_3^2) \\
 * & + f_3 (15 g_1^2 g_4 + 60 g_1 g_2 g_3 + 15 g_2^3) \\
 * & + f_4 (20 g_1^3 g_3 + 45 g_1^2 g_2^2) + 15 f_5 g_1^4 g_2 + f_6 g_1^6,
 * \end{aligned}
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 7 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, \dots, 6\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 7 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, \dots, 6\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^6}{\mathrm{d}x^6} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 3.96 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 6.0 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d5(), chain_rule_d7()
 */
LAPLACE_HD double chain_rule_d6(const double* f, const double* g) { return detail_faa::compose<6>(f, g); }

/**
 * \ingroup faa_di_bruno
 * \brief Seventh derivative of a composite function \f$f(g(x))\f$.
 *
 * Computes the seventh derivative by Faà di Bruno's formula, summed from \f$k = 7\f$ down to \f$1\f$:
 * \f[
 * \begin{aligned}
 * \frac{\mathrm{d}^7}{\mathrm{d}x^7} f(g(x)) = {} & f_1 g_7 + f_2 (7 g_1 g_6 + 21 g_2 g_5 + 35 g_3 g_4) \\
 * & + f_3 (21 g_1^2 g_5 + 105 g_1 g_2 g_4 + 70 g_1 g_3^2 + 105 g_2^2 g_3) \\
 * & + f_4 (35 g_1^3 g_4 + 210 g_1^2 g_2 g_3 + 105 g_1 g_2^3) \\
 * & + f_5 (35 g_1^4 g_3 + 105 g_1^3 g_2^2) + 21 f_6 g_1^5 g_2 + f_7 g_1^7,
 * \end{aligned}
 * \f]
 * where \f$f_k = f^{(k)}(g(x))\f$ and \f$g_m = g^{(m)}(x)\f$.
 *
 * \param f Pointer to at least 8 doubles storing \f$f^{(k)}(g(x))\f$ in `f[k]` for \f$k = 1, \dots, 7\f$. Element `f[0]` is not read.
 * \param g Pointer to at least 8 doubles storing \f$g^{(m)}(x)\f$ in `g[m]` for \f$m = 1, \dots, 7\f$. Element `g[0]` is not read.
 * \return Value of \f$\frac{\mathrm{d}^7}{\mathrm{d}x^7} f(g(x))\f$, or quiet NaN if any required element of `f` or `g` is NaN.
 *
 * Measured maximum error is 5.03 ulp of \f$S\f$ on gcc and CUDA, with a test bound of 7.5 ulp of \f$S\f$ (see \ref faa_di_bruno "the module page").
 *
 * \see faa_di_bruno, bell_partial(), chain_rule_d6()
 */
LAPLACE_HD double chain_rule_d7(const double* f, const double* g) { return detail_faa::compose<7>(f, g); }

}  // namespace laplace::math
