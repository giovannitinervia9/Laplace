// The orthonormal Legendre basis (laplace::basis); design of 2026-11-14.
//
// On [lower, upper] of width w = upper - lower, with t = 2 (x - lower)/w - 1,
// the basis of maximum degree n has K = n + 1 columns
//   b_m(x) = sqrt((2 m + 1)/w) P_m(t),   m = 0..n,
// where P_m is the Legendre polynomial; int_lower^upper b_i b_j dx = delta_ij.
// The basis is complete (no column dropped) and dense: row_width() = K and
// every row starts at column 0.
//
// Rows. The derivative of order k of P_m is
//   P_m^(k)(t) = (2 k - 1)!! C^(k + 1/2)_(m - k)(t)    (zero for m < k),
// where the Gegenbauer polynomials C^(lambda)_j come from their three-term
// recurrence in j,
//   C_0 = 1,  C_1 = 2 lambda t,
//   (j + 1) C_(j+1) = 2 (j + lambda) t C_j - (j + 2 lambda - 1) C_(j-1);
// order 0 (lambda = 1/2) is Bonnet's recurrence. The row of order k is this
// value times (2/w)^k and the normalization sqrt((2 m + 1)/w). One function
// per order, no scratch; an expansion sum_m beta_m b_m^(k)(x) is accumulated
// during the recurrence, without a buffer of K values. Orders above n are
// exact zeros.
//
// The integral from lower. For m >= 1,
//   int_{-1}^t P_m = (P_(m+1)(t) - P_(m-1)(t))/(2 m + 1) = (t^2 - 1) P_m'(t)/(m (m + 1)),
// and the second form is used: with t^2 - 1 = -4 (upper - x)(x - lower)/w^2 and
// P_m' = C^(3/2)_(m-1), column m of the integral is
//   -(2/w) (upper - x)(x - lower) sqrt((2 m + 1)/w) C^(3/2)_(m-1)(t)/(m (m + 1)),
// a product that is accurate relative to its own size near both ends, where
// the difference form loses everything (see docs/numerics.md); column 0 is
// (x - lower)/sqrt(w).
//
// Outside [lower, upper] the basis follows its outside_domain_rule (core.hpp):
// polynomial evaluates the polynomials at x; nan gives NaN beyond 1e-8 of the
// width and the polynomial inside that tolerance; linear gives f(e) + f'(e)
// (x - e) at the nearer end e (order 1: f'(e); above: 0; the integral
// I(e) + f(e) (x - e) + f'(e) (x - e)^2/2).
//
// Host functions: cross_gram(k, l, out), the K x K matrix of
// int b^(k) b^(l)' dx, by Gauss-Legendre in t (exact: the integrand is a
// polynomial of degree 2 n - k - l; the 64-node rule is exact to degree 127,
// and a rule with degree/2 + 1 nodes, computed on the host, is used above).
#pragma once

#include <laplace/basis/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace laplace::basis {

/**
 * \ingroup basis_legendre
 * \brief Orthonormal Legendre polynomial basis on a 1D interval.
 *
 * Implements the orthonormal Legendre basis of maximum degree \f$n \ge 0\f$
 * and dimension \f$K = n + 1\f$ on \f$[lower, upper]\f$:
 * \f[
 * b_m(x) = \sqrt{\frac{2m + 1}{w}} P_m(t), \quad m = 0, \dots, n,
 * \f]
 * where \f$w = upper - lower\f$, \f$t = 2(x - lower)/w - 1\f$, and \f$P_m\f$ is
 * the Legendre polynomial of degree \f$m\f$. Construct with make_legendre_basis()
 * and pass by value into Kokkos kernels.
 *
 * \see basis_legendre, make_legendre_basis()
 */
struct legendre_basis {
  /**
   * \ingroup basis_legendre
   * \var legendre_basis::lower
   * \brief Lower bound of the domain, \f$lower\f$.
   */
  /**
   * \ingroup basis_legendre
   * \var legendre_basis::upper
   * \brief Upper bound of the domain, \f$upper\f$.
   */
  double lower = -1.0, upper = 1.0;
  /**
   * \ingroup basis_legendre
   * \brief Maximum polynomial degree \f$n \ge 0\f$.
   */
  int max_degree = 5;  // n
  /**
   * \ingroup basis_legendre
   * \brief Rule for evaluating the basis outside \f$[lower, upper]\f$ (\ref outside_domain_rule).
   */
  outside_domain_rule outside = outside_domain_rule::linear;

  /**
   * \ingroup basis_legendre
   * \brief Total number of basis functions \f$K = n + 1\f$.
   * \return Basis dimension \f$K\f$ (non-negative integer for \f$n \ge 0\f$).
   * \see basis_legendre, row_width()
   */
  LAPLACE_HD int dimension() const { return max_degree + 1; }
  /**
   * \ingroup basis_legendre
   * \brief Number of columns in each evaluated row, \f$K = n + 1\f$.
   * \return Row width \f$K\f$, equal to dimension() (dense row).
   * \see basis_legendre, dimension(), row()
   */
  LAPLACE_HD int row_width() const { return max_degree + 1; }
  /**
   * \ingroup basis_legendre
   * \brief Scratch buffer size needed for evaluation.
   * \return Zero (no scratch memory required).
   * \see basis_legendre
   */
  LAPLACE_HD int scratch_size() const { return 0; }
  /**
   * \ingroup basis_legendre
   * \brief Semi-bandwidth of cross Gram matrices.
   * \return Maximum degree \f$n\f$ (dense matrix).
   * \see basis_legendre, cross_gram()
   */
  LAPLACE_HD int gram_bandwidth() const { return max_degree; }  // dense

  /**
   * \ingroup basis_legendre
   * \brief Evaluates the basis functions at \f$x\f$ (derivative order 0).
   *
   * Writes \f$b_m(x) = \sqrt{(2m + 1)/w} P_m(t)\f$ for \f$m = 0, \dots, n\f$ into \c out,
   * where \f$w = upper - lower\f$, \f$t\f$ is computed from the nearer endpoint, and \f$P_m(t)\f$
   * uses Bonnet's recurrence.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Outside \f$[lower, upper]\f$, follows \ref outside. Writes NaN to all entries if \f$x\f$ is NaN.
   * \par Measured accuracy
   * Max error 115 ulp (gcc) / 88 ulp (CUDA) for \f$n \le 20\f$ (bound 175 ulp), and 551 ulp (gcc) / 474 ulp (CUDA) at \f$n = 50\f$ (bound 830 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d1(), expansion()
   */
  // rows: values of b_m^(k)(x), m = 0..n; return the first column, 0
  template <class T> LAPLACE_HD int row(T x, T* out) const { return row_order<0>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates first derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(1)}(x) = (2/w) \sqrt{(2m + 1)/w} C_{m-1}^{(3/2)}(t)\f$ for \f$m \ge 1\f$
   * and \f$0\f$ for \f$m = 0\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 1\f$.
   * \par Measured accuracy
   * Max error 38 ulp (gcc) / 34 ulp (CUDA) for \f$n \le 20\f$ (bound 60 ulp), and 292 ulp (gcc) / 258 ulp (CUDA) at \f$n = 50\f$ (bound 440 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row(), row_d2(), expansion_d1()
   */
  template <class T> LAPLACE_HD int row_d1(T x, T* out) const { return row_order<1>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates second derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(2)}(x) = 3 (2/w)^2 \sqrt{(2m + 1)/w} C_{m-2}^{(5/2)}(t)\f$ for \f$m \ge 2\f$
   * and \f$0\f$ for \f$m < 2\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 2\f$.
   * \par Measured accuracy
   * Max error 32 ulp (gcc) / 39 ulp (CUDA) for \f$n \le 20\f$ (bound 50 ulp), and 194 ulp (gcc and CUDA) at \f$n = 50\f$ (bound 300 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d1(), row_d3(), expansion_d2()
   */
  template <class T> LAPLACE_HD int row_d2(T x, T* out) const { return row_order<2>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates third derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(3)}(x) = 15 (2/w)^3 \sqrt{(2m + 1)/w} C_{m-3}^{(7/2)}(t)\f$ for \f$m \ge 3\f$
   * and \f$0\f$ for \f$m < 3\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 3\f$.
   * \par Measured accuracy
   * Max error 28 ulp (gcc) / 23 ulp (CUDA) for \f$n \le 20\f$ (bound 45 ulp), and 116 ulp (gcc) / 111 ulp (CUDA) at \f$n = 50\f$ (bound 175 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d2(), row_d4(), expansion_d3()
   */
  template <class T> LAPLACE_HD int row_d3(T x, T* out) const { return row_order<3>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates fourth derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(4)}(x) = 105 (2/w)^4 \sqrt{(2m + 1)/w} C_{m-4}^{(9/2)}(t)\f$ for \f$m \ge 4\f$
   * and \f$0\f$ for \f$m < 4\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 4\f$.
   * \par Measured accuracy
   * Max error 34 ulp (gcc) / 25 ulp (CUDA) for \f$n \le 20\f$ (bound 52 ulp), and 145 ulp (gcc) / 149 ulp (CUDA) at \f$n = 50\f$ (bound 220 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d3(), row_d5(), expansion_d4()
   */
  template <class T> LAPLACE_HD int row_d4(T x, T* out) const { return row_order<4>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates fifth derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(5)}(x) = 945 (2/w)^5 \sqrt{(2m + 1)/w} C_{m-5}^{(11/2)}(t)\f$ for \f$m \ge 5\f$
   * and \f$0\f$ for \f$m < 5\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 5\f$.
   * \par Measured accuracy
   * Max error 19 ulp (gcc) / 20 ulp (CUDA) for \f$n \le 20\f$ (bound 30 ulp), and 136 ulp (gcc) / 104 ulp (CUDA) at \f$n = 50\f$ (bound 205 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d4(), row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD int row_d5(T x, T* out) const { return row_order<5>(x, out); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates sixth derivatives at \f$x\f$.
   *
   * Writes \f$b_m^{(6)}(x) = 10395 (2/w)^6 \sqrt{(2m + 1)/w} C_{m-6}^{(13/2)}(t)\f$ for \f$m \ge 6\f$
   * and \f$0\f$ for \f$m < 6\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Same \ref outside and NaN rules as row(); writes exact zeros for \f$n < 6\f$.
   * \par Measured accuracy
   * Max error 24 ulp (gcc) / 26 ulp (CUDA) for \f$n \le 20\f$ (bound 36 ulp), and 104 ulp (gcc) / 116 ulp (CUDA) at \f$n = 50\f$ (bound 160 ulp); within 20 times the conditioning scale.
   * \see basis_legendre, row_d5(), expansion_d6()
   */
  template <class T> LAPLACE_HD int row_d6(T x, T* out) const { return row_order<6>(x, out); }

  /**
   * \ingroup basis_legendre
   * \brief Evaluates the linear expansion \f$\sum_{m=0}^n \beta_m b_m(x)\f$ at \f$x\f$.
   *
   * Accumulates the weighted sum in a single pass during Bonnet's recurrence without allocating
   * an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Expansion value at \f$x\f$.
   * \par Domain and special values
   * Outside \f$[lower, upper]\f$, follows \ref outside. Returns NaN if \f$x\f$ or any \f$\beta_m\f$ is NaN.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m(x)|\f$.
   * \see basis_legendre, row(), expansion_d1()
   */
  // sum_m beta_m b_m^(k)(x)
  template <class T> LAPLACE_HD T expansion(T x, const T* beta) const { return expansion_order<0>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the first derivative \f$\sum_{m=0}^n \beta_m b_m^{(1)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return First derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 1\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d1() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(1)}(x)|\f$.
   * \see basis_legendre, row_d1(), expansion(), expansion_d2()
   */
  template <class T> LAPLACE_HD T expansion_d1(T x, const T* beta) const { return expansion_order<1>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the second derivative \f$\sum_{m=0}^n \beta_m b_m^{(2)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Second derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 2\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d2() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(2)}(x)|\f$.
   * \see basis_legendre, row_d2(), expansion_d1(), expansion_d3()
   */
  template <class T> LAPLACE_HD T expansion_d2(T x, const T* beta) const { return expansion_order<2>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the third derivative \f$\sum_{m=0}^n \beta_m b_m^{(3)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Third derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 3\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d3() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(3)}(x)|\f$.
   * \see basis_legendre, row_d3(), expansion_d2(), expansion_d4()
   */
  template <class T> LAPLACE_HD T expansion_d3(T x, const T* beta) const { return expansion_order<3>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the fourth derivative \f$\sum_{m=0}^n \beta_m b_m^{(4)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Fourth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 4\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d4() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(4)}(x)|\f$.
   * \see basis_legendre, row_d4(), expansion_d3(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d4(T x, const T* beta) const { return expansion_order<4>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the fifth derivative \f$\sum_{m=0}^n \beta_m b_m^{(5)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Fifth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 5\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d5() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(5)}(x)|\f$.
   * \see basis_legendre, row_d5(), expansion_d4(), expansion_d6()
   */
  template <class T> LAPLACE_HD T expansion_d5(T x, const T* beta) const { return expansion_order<5>(x, beta); }
  /**
   * \ingroup basis_legendre
   * \brief Evaluates the sixth derivative \f$\sum_{m=0}^n \beta_m b_m^{(6)}(x)\f$ at \f$x\f$.
   *
   * Single-pass accumulation during the Gegenbauer recurrence without an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Sixth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns exact 0 for \f$n < 6\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and row_d6() within 64 ulp of \f$\sum_{m=0}^n |\beta_m b_m^{(6)}(x)|\f$.
   * \see basis_legendre, row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d6(T x, const T* beta) const { return expansion_order<6>(x, beta); }

  /**
   * \ingroup basis_legendre
   * \brief Evaluates column integrals from the lower boundary to \f$x\f$.
   *
   * Evaluates \f$\int_{lower}^x b_0(\tau) \, \mathrm{d}\tau = (x - lower) / \sqrt{w}\f$, and for \f$m \ge 1\f$:
   * \f[
   * \int_{lower}^x b_m(\tau) \, \mathrm{d}\tau = -\frac{2}{w} (upper - x)(x - lower) \sqrt{\frac{2m + 1}{w}} \frac{C_{m-1}^{(3/2)}(t)}{m (m + 1)}
   * \f]
   * in a product form that avoids cancellation near both boundaries.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit.
   * \param out Output buffer of size at least row_width().
   * \return First non-zero column index (0, dense row).
   * \par Domain and special values
   * Outside \f$[lower, upper]\f$, follows \ref outside. At \f$x = lower\f$, returns all zeros;
   * at \f$x = upper\f$, returns \f$\sqrt{w}\f$ for column 0 and exact zeros for \f$m \ge 1\f$.
   * Writes NaN to all entries if \f$x\f$ is NaN.
   * \par Measured accuracy
   * Max error 35 ulp (gcc) / 41 ulp (CUDA) for \f$n \le 20\f$ (bound 53 ulp), and 197 ulp (gcc) / 193 ulp (CUDA) at \f$n = 50\f$ (bound 300 ulp).
   * \see basis_legendre, integral_expansion(), row()
   */
  // int_lower^x b_m, m = 0..n (dense, first column 0)
  template <class T>
  LAPLACE_HD int integral_row(T x, T* out) const {
    if (!(x == x)) return fill_nan(out);
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      if (use_end_piece(x, e)) {
        integral_core(x, out);
        return 0;
      }
      if (outside == outside_domain_rule::nan) return fill_nan(out);
      // linear: I(e) + b(e) dx + b'(e) dx^2/2
      const T te = left ? T(-1) : T(1), dx = x - e;
      integral_core(e, out);
      row_core<0, true>(te, dx, out);
      row_core<1, true>(te, dx * dx / T(2), out);
      return 0;
    }
    integral_core(x, out);
    return 0;
  }

  /**
   * \ingroup basis_legendre
   * \brief Evaluates the integrated basis expansion \f$\int_{lower}^x \left(\sum_{m=0}^n \beta_m b_m(\tau)\right) \mathrm{d}\tau\f$.
   *
   * Single-pass accumulation without allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit.
   * \param beta Array of \f$K = n + 1\f$ basis coefficients.
   * \return Integrated expansion value.
   * \par Domain and special values
   * Same \ref outside and NaN rules as expansion(); returns 0 at \f$x = lower\f$.
   * \par Measured accuracy
   * Agrees with the inner product of \c beta and integral_row() within 64 ulp of \f$\sum_{m=0}^n |\beta_m \int_{lower}^x b_m(\tau) \, \mathrm{d}\tau|\f$.
   * \see basis_legendre, integral_row(), expansion()
   */
  template <class T>
  LAPLACE_HD T integral_expansion(T x, const T* beta) const {
    if (!(x == x)) return T(laplace::math::detail::quiet_nan());
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      if (use_end_piece(x, e)) return integral_expansion_core(x, beta);
      if (outside == outside_domain_rule::nan) return T(laplace::math::detail::quiet_nan());
      const T te = left ? T(-1) : T(1), dx = x - e;
      return integral_expansion_core(e, beta) + expansion_core<0>(te, beta) * dx +
             expansion_core<1>(te, beta) * (dx * dx / T(2));
    }
    return integral_expansion_core(x, beta);
  }

  // ---- host functions

  /**
   * \ingroup basis_legendre
   * \brief Evaluates the cross Gram matrix \f$G_{kl} = \int_{lower}^{upper} b^{(k)}(x) (b^{(l)}(x))^\top \mathrm{d}x\f$.
   *
   * Host-only. Writes the dense \f$K \times K\f$ matrix in column-major order (\c out[i + j * K]).
   * For \f$k = l = 0\f$, writes the exact identity \f$I\f$. Otherwise integrates on \f$t \in [-1, 1]\f$
   * (polynomial integrand of degree \f$2n - k - l\f$) using the 64-node rule for \f$2n - k - l \le 127\f$
   * and \f$\lfloor(2n - k - l)/2\rfloor + 1\f$ nodes from gauss_legendre_nodes() above.
   * \param k Left derivative order (\f$k \ge 0\f$).
   * \param l Right derivative order (\f$l \ge 0\f$).
   * \param out Output buffer of size at least \f$K \times K\f$.
   * \par Domain and special values
   * Returns an all-zero matrix if \f$k > n\f$ or \f$l > n\f$.
   * \par Measured accuracy
   * Exact identity for \f$G_{00}\f$. Max error 84 ulp for \f$n \le 20\f$ (bound 126 ulp) and 384 ulp at \f$n = 50\f$ (bound 580 ulp) on gcc; at \f$n = 100\f$, \f$G_{10} + G_{01}\f$ satisfies integration by parts within \f$5 \times 10^{-13}\f$ relative error and \f$G_{11}\f$ is symmetric within \f$10^{-13}\f$.
   * \see basis_legendre, gram_bandwidth(), gauss_legendre_nodes()
   */
  // G_kl = int_lower^upper b^(k) b^(l)' dx, written dense K x K, column-major
  // (out[i + j K] = int b_i^(k) b_j^(l)); G_00 is the identity, written exactly
  void cross_gram(int k, int l, double* out) const {
    const int K = dimension(), n = max_degree;
    for (size_t i = 0; i < static_cast<size_t>(K) * K; ++i) out[i] = 0.0;
    if (k == 0 && l == 0) {
      for (int i = 0; i < K; ++i) out[i + static_cast<size_t>(i) * K] = 1.0;
      return;
    }
    if (k > n || l > n) return;
    std::vector<double> rk(K), rl(K);
    const double hw = 0.5 * (upper - lower);
    const int degree = 2 * n - k - l;  // of the integrand in t
    // up to degree 127 the 64-node rule of the quadrature module is exact;
    // above it, a rule with N = degree/2 + 1 nodes, found by Newton's method
    std::vector<double> nodes, weights;
    int count = 64;
    if (degree > 127) {
      count = degree / 2 + 1;
      gauss_legendre_nodes(count, nodes, weights);
    }
    for (int i = 0; i < count; ++i) {
      const double t = degree > 127 ? nodes[i] : laplace::math::quad::detail::legendre64(i);
      const double wq = hw * (degree > 127 ? weights[i] : laplace::math::quad::detail::legendre64(64 + i));
      row_core_of_order(k, t, rk.data());
      row_core_of_order(l, t, rl.data());
      for (int j = l; j < K; ++j)
        for (int r = k; r < K; ++r) out[r + static_cast<size_t>(j) * K] += wq * rk[r] * rl[j];
    }
  }

  /**
   * \ingroup basis_legendre
   * \brief Computes \f$N\f$ Gauss-Legendre quadrature nodes and weights on \f$[-1, 1]\f$.
   *
   * Host-only. Finds the roots of \f$P_N(t)\f$ via Newton's method from initial guesses
   * \f$\cos(\pi (i + 3/4) / (N + 1/2))\f$ and computes weights \f$2 / ((1 - t^2) (P_N'(t))^2)\f$.
   * Used by cross_gram() for integrand degrees above 127.
   * \param N Number of quadrature nodes (\f$N \ge 1\f$).
   * \param nodes Output vector of \f$N\f$ quadrature nodes.
   * \param weights Output vector of \f$N\f$ quadrature weights.
   * \par Stopping rule
   * Each Newton iteration stops when its step is at most \f$4 \varepsilon |t|\f$, or after 100 steps.
   * \see basis_legendre, cross_gram()
   */
  // The N nodes and weights of Gauss-Legendre on [-1, 1]: roots of P_N by
  // Newton's method from the initial guesses cos(pi (i + 3/4)/(N + 1/2)),
  // weights 2/((1 - t^2) P_N'(t)^2). Used by cross_gram for degrees above 127.
  static void gauss_legendre_nodes(int N, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.assign(N, 0.0);
    weights.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
      double t = std::cos(3.14159265358979323846 * (i + 0.75) / (N + 0.5)), dp = 1.0;
      for (int it = 0; it < 100; ++it) {
        double p0 = 1.0, p1 = t;
        for (int j = 1; j < N; ++j) {
          const double p2 = ((2 * j + 1) * t * p1 - j * p0) / (j + 1);
          p0 = p1;
          p1 = p2;
        }
        dp = N * (t * p1 - p0) / (t * t - 1.0);
        const double dt = p1 / dp;
        t -= dt;
        if (std::fabs(dt) <= 4 * std::numeric_limits<double>::epsilon() * std::fabs(t)) break;
      }
      // the derivative at the final t
      double p0 = 1.0, p1 = t;
      for (int j = 1; j < N; ++j) {
        const double p2 = ((2 * j + 1) * t * p1 - j * p0) / (j + 1);
        p0 = p1;
        p1 = p2;
      }
      dp = N * (t * p1 - p0) / (t * t - 1.0);
      nodes[i] = t;
      weights[i] = 2.0 / ((1.0 - t * t) * dp * dp);
    }
  }

  // ---- internals

  template <class T>
  LAPLACE_HD int fill_nan(T* out) const {
    const T nan = T(laplace::math::detail::quiet_nan());
    for (int m = 0; m <= max_degree; ++m) out[m] = nan;
    return 0;
  }

  template <class T>
  LAPLACE_HD bool use_end_piece(T x, T e) const {
    if (outside == outside_domain_rule::polynomial) return true;
    if (outside == outside_domain_rule::nan) return Kokkos::fabs(x - e) <= T(1e-8) * T(upper - lower);
    return false;
  }

  // t = 2 (x - lower)/w - 1, written from the nearer end (1 - 2 (upper - x)/w
  // in the upper half) so that near t = +-1, where the polynomials are most
  // sensitive to t, the rounding of t is a half ulp of a number below 1
  template <class T>
  LAPLACE_HD T point_t(T x) const {
    const T w = T(upper - lower), u = x - T(lower), v = T(upper) - x;
    return u <= v ? T(2) * u / w - T(1) : T(1) - T(2) * v / w;
  }

  // (2 k - 1)!! (2/w)^k
  template <int K, class T>
  LAPLACE_HD T order_factor() const {
    constexpr double df = K <= 1 ? 1.0 : K == 2 ? 3.0 : K == 3 ? 15.0 : K == 4 ? 105.0 : K == 5 ? 945.0 : 10395.0;
    const T s = T(2) / T(upper - lower);
    T f = T(df);
    for (int i = 0; i < K; ++i) f *= s;
    return f;
  }

  // out[m] (+)= s b_m^(K) at t, for m = 0..n; the polynomial in t, whatever t is
  template <int K, bool Add, class T>
  LAPLACE_HD void row_core(T t, T s, T* out) const {
    const int n = max_degree;
    const T w = T(upper - lower), f = s * order_factor<K, T>();
    for (int m = 0; m < K && m <= n; ++m) {
      if (Add) out[m] += T(0);
      else out[m] = T(0);
    }
    T cp = T(0), c = T(1);
    for (int m = K, j = 0; m <= n; ++m, ++j) {
      const T v = f * Kokkos::sqrt(T(2 * m + 1) / w) * c;
      if (Add) out[m] += v;
      else out[m] = v;
      const T cn = (T(2 * j + 2 * K + 1) * t * c - T(j + 2 * K) * cp) / T(j + 1);
      cp = c;
      c = cn;
    }
  }

  template <class T>
  LAPLACE_HD void row_core_of_order(int k, T t, T* out) const {
    switch (k) {
      case 0: row_core<0, false>(t, T(1), out); break;
      case 1: row_core<1, false>(t, T(1), out); break;
      case 2: row_core<2, false>(t, T(1), out); break;
      case 3: row_core<3, false>(t, T(1), out); break;
      case 4: row_core<4, false>(t, T(1), out); break;
      case 5: row_core<5, false>(t, T(1), out); break;
      default: row_core<6, false>(t, T(1), out); break;
    }
  }

  // sum_m beta_m b_m^(K) at t
  template <int K, class T>
  LAPLACE_HD T expansion_core(T t, const T* beta) const {
    const int n = max_degree;
    const T w = T(upper - lower), f = order_factor<K, T>();
    T cp = T(0), c = T(1), s = T(0);
    for (int m = K, j = 0; m <= n; ++m, ++j) {
      s += beta[m] * (Kokkos::sqrt(T(2 * m + 1) / w) * c);
      const T cn = (T(2 * j + 2 * K + 1) * t * c - T(j + 2 * K) * cp) / T(j + 1);
      cp = c;
      c = cn;
    }
    return s * f;
  }

  template <int K, class T>
  LAPLACE_HD int row_order(T x, T* out) const {
    if (!(x == x)) return fill_nan(out);
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      if (use_end_piece(x, e)) {
        row_core<K, false>(point_t(x), T(1), out);
        return 0;
      }
      if (outside == outside_domain_rule::nan) return fill_nan(out);
      const T te = left ? T(-1) : T(1);
      if constexpr (K == 0) {
        row_core<0, false>(te, T(1), out);
        row_core<1, true>(te, x - e, out);
      } else if constexpr (K == 1) {
        row_core<1, false>(te, T(1), out);
      } else {
        for (int m = 0; m <= max_degree; ++m) out[m] = T(0);
      }
      return 0;
    }
    row_core<K, false>(point_t(x), T(1), out);
    return 0;
  }

  template <int K, class T>
  LAPLACE_HD T expansion_order(T x, const T* beta) const {
    if (!(x == x)) return T(laplace::math::detail::quiet_nan());
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      if (use_end_piece(x, e)) return expansion_core<K>(point_t(x), beta);
      if (outside == outside_domain_rule::nan) return T(laplace::math::detail::quiet_nan());
      const T te = left ? T(-1) : T(1);
      if constexpr (K == 0) return expansion_core<0>(te, beta) + expansion_core<1>(te, beta) * (x - e);
      else if constexpr (K == 1) return expansion_core<1>(te, beta);
      else return T(0);
    }
    return expansion_core<K>(point_t(x), beta);
  }

  // int_lower^x b_m for the polynomial continued to any x
  template <class T>
  LAPLACE_HD void integral_core(T x, T* out) const {
    const int n = max_degree;
    const T w = T(upper - lower), u = x - T(lower);
    out[0] = u / Kokkos::sqrt(w);
    if (n == 0) return;
    const T t = point_t(x), fac = -(T(2) / w) * ((T(upper) - x) * u);
    T cp = T(0), c = T(1);  // C^(3/2)_(m-1), C^(3/2)_(m-2)
    for (int m = 1; m <= n; ++m) {
      out[m] = fac * Kokkos::sqrt(T(2 * m + 1) / w) / T(m * (m + 1)) * c;
      const int j = m - 1;
      const T cn = (T(2 * j + 3) * t * c - T(j + 2) * cp) / T(j + 1);
      cp = c;
      c = cn;
    }
  }

  template <class T>
  LAPLACE_HD T integral_expansion_core(T x, const T* beta) const {
    const int n = max_degree;
    const T w = T(upper - lower), u = x - T(lower);
    T s = beta[0] * (u / Kokkos::sqrt(w));
    if (n == 0) return s;
    const T t = point_t(x), fac = -(T(2) / w) * ((T(upper) - x) * u);
    T cp = T(0), c = T(1), acc = T(0);
    for (int m = 1; m <= n; ++m) {
      acc += beta[m] * (Kokkos::sqrt(T(2 * m + 1) / w) / T(m * (m + 1)) * c);
      const int j = m - 1;
      const T cn = (T(2 * j + 3) * t * c - T(j + 2) * cp) / T(j + 1);
      cp = c;
      c = cn;
    }
    return s + fac * acc;
  }
};

/**
 * \ingroup basis_legendre
 * \brief Constructs an orthonormal Legendre polynomial basis on \f$[lower, upper]\f$.
 *
 * \param lower Lower bound of the domain.
 * \param upper Upper bound of the domain (\f$upper > lower\f$).
 * \param max_degree Maximum polynomial degree \f$n \ge 0\f$.
 * \param outside Evaluation rule outside \f$[lower, upper]\f$ (defaults to \c outside_domain_rule::linear).
 * \return Configured \ref legendre_basis structure.
 * \par Domain and special values
 * Throws \c std::invalid_argument if \c lower or \c upper is non-finite, \c lower >= upper,
 * or \c max_degree < 0.
 * \see basis_legendre, legendre_basis
 */
// Checks its arguments (finite, lower < upper, max_degree >= 0) and throws
// std::invalid_argument.
inline legendre_basis make_legendre_basis(double lower, double upper, int max_degree,
                                          outside_domain_rule outside = outside_domain_rule::linear) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper))
    throw std::invalid_argument("legendre: need finite lower < upper");
  if (max_degree < 0) throw std::invalid_argument("legendre: need max_degree >= 0");
  legendre_basis b;
  b.lower = lower;
  b.upper = upper;
  b.max_degree = max_degree;
  b.outside = outside;
  return b;
}

}  // namespace laplace::basis
