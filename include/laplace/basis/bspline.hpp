// B-spline bases (laplace::basis); design decisions D1-D10 of 2026-11-13/14
// (diario.md), measurements in the session's benches (to be moved to
// docs/numerics.md).
//
// A B-spline basis of degree d (0 <= d <= 10) and dimension K is defined by a
// non-decreasing knot vector t_0 <= ... <= t_{K+d}; its domain is
// [lower, upper] = [t_d, t_K], where the K functions B_0..B_{K-1} sum to 1.
// A value may appear at most d + 1 times; at an interior knot of multiplicity
// mu the spline is C^(d - mu) (mu = d + 1: a jump).
//
// Evaluation (in kernels). The interval s, t_s <= x < t_{s+1} (x = upper
// belongs to the last interval: the basis is right-continuous, with left
// limits at upper), comes from a binary search on the knots or, when the
// constructor made them equally spaced with step h, from s = d +
// floor((x - lower)/h), corrected against t_s and t_{s+1} (rounding sends
// points on a knot or one ulp from it to the neighbouring interval). The d + 1
// nonzero values B_{s-d..s}(x) come from de Boor's BSPLVB; the derivative of
// order k from the values of degree d - k and k steps of
//   D B_{i,p} = p (B_{i,p-1}/(t_{i+p} - t_i) - B_{i+1,p-1}/(t_{i+p+1} - t_{i+1}))
// (BSPLVD; a term with an empty support is zero). Orders above d are zero.
// Against exact rational references: within 6 ulp of the row's largest value
// for d <= 5 and 15 for d = 10, 2.3 and 6.8 times the conditioning.
//
// The integral from lower: for x in interval s, columns j < s - d have
// finished and hold their total integral_column_total(j); columns j > s have
// not started and hold 0; integral_row gives the window j = s - d..s, from
//   int_{t_j}^x B_{j,d} = (t_{j+d+1} - t_j)/(d + 1) sum_{i=j}^{s} B_{i,d+1}(x)
// minus the part before lower (nonzero only when knots lie below lower).
//
// Storage: bspline_basis holds a pointer to its knots; bspline_basis_owner
// keeps them on the host and in the default memory space and hands out the
// struct for either (host() for the Gram matrices and host evaluation,
// device() for kernels).
#pragma once

#include <laplace/basis/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laplace::basis {

/**
 * \ingroup basis_bspline
 * \brief Maximum supported polynomial degree for B-splines (\f$d_{\max} = 10\f$).
 */
inline constexpr int bspline_max_degree = 10;

/**
 * \ingroup basis_bspline
 * \brief Point of reduced continuity \f$C^c\f$ across the B-spline domain.
 *
 * Records a point \f$x\f$ where the basis is \f$C^c\f$ and no smoother, with \f$c = \text{continuity}\f$.
 * A continuity of \f$-1\f$ indicates a step discontinuity (jump).
 * \see basis_bspline, bspline_basis::regularity_points()
 */
// A point where the basis is C^continuity and no smoother (continuity -1: the
// functions jump; derivatives of order above continuity are one-sided there).
struct bspline_regularity_point {
  /**
   * \ingroup basis_bspline
   * \brief Abscissa of the regularity point.
   */
  double x;
  /**
   * \ingroup basis_bspline
   * \brief Continuity order \f$c\f$ (\f$c \ge 0\f$ for \f$C^c\f$, \f$-1\f$ for jump).
   */
  int continuity;
};

/**
 * \ingroup basis_bspline
 * \brief Lightweight view struct for B-spline basis evaluation on host and device.
 *
 * Holds a pointer to \f$K + d + 1\f$ knots along with scalar parameters of the basis.
 * Evaluates sparse rows of width \f$w = d + 1\f$, derivatives of orders 0 to 6,
 * linear expansions, and column integrals from the lower boundary.
 *
 * Designed to be copied by value into Kokkos device kernels via bspline_basis_owner::device().
 * \see basis_bspline, bspline_basis_owner
 */
struct bspline_basis {
  /**
   * \ingroup basis_bspline
   * \brief Pointer to the \f$K + d + 1\f$ non-decreasing knot values in the caller's memory space.
   */
  const double* knots = nullptr;  // K + d + 1 values, in the memory space of the caller
  /**
   * \ingroup basis_bspline
   * \brief Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   */
  int degree = 3;
  /**
   * \ingroup basis_bspline
   * \brief Total number of basis functions \f$K \ge d + 1\f$.
   */
  int dim = 0;                    // K
  /**
   * \ingroup basis_bspline
   * \brief Lower domain boundary \f$t_d\f$.
   */
  /**
   * \ingroup basis_bspline
   * \brief Upper domain boundary \f$t_K\f$.
   */
  double lower = 0.0, upper = 1.0;
  /**
   * \ingroup basis_bspline
   * \brief Step size \f$h > 0\f$ for equally spaced knots, or \f$0.0\f$ for general knots.
   */
  double step = 0.0;              // h > 0 when the knots are equally spaced (arithmetic interval)
  /**
   * \ingroup basis_bspline
   * \brief Rule for evaluating the basis outside \f$[\text{lower}, \text{upper}]\f$.
   */
  outside_domain_rule outside = outside_domain_rule::linear;

  /**
   * \ingroup basis_bspline
   * \brief Total number of basis functions \f$K\f$.
   * \return Basis dimension \f$K\f$.
   * \par Domain and special values
   * Positive integer satisfying \f$K \ge d + 1\f$.
   * \par Measured accuracy
   * Exact integer.
   * \see basis_bspline, row_width()
   */
  LAPLACE_HD int dimension() const { return dim; }
  /**
   * \ingroup basis_bspline
   * \brief Number of non-zero basis functions in an evaluated row, \f$w = d + 1\f$.
   * \return Row width \f$d + 1\f$.
   * \par Domain and special values
   * Positive integer in \f$\{1, \dots, 11\}\f$ for \f$d \in \{0, \dots, 10\}\f$.
   * \par Measured accuracy
   * Exact integer.
   * \see basis_bspline, dimension(), row()
   */
  LAPLACE_HD int row_width() const { return degree + 1; }
  /**
   * \ingroup basis_bspline
   * \brief Scratch buffer size needed for evaluation.
   * \return Zero (the B-spline basis requires no caller scratch buffer).
   * \par Measured accuracy
   * Exact integer.
   * \see basis_bspline
   */
  LAPLACE_HD int scratch_size() const { return 0; }
  /**
   * \ingroup basis_bspline
   * \brief Semi-bandwidth of cross Gram matrices \f$G_{kl}\f$.
   * \return Semi-bandwidth equal to the polynomial degree \f$d\f$.
   * \par Domain and special values
   * Integer in \f$\{0, \dots, 10\}\f$.
   * \par Measured accuracy
   * Exact integer.
   * \see basis_bspline, cross_gram()
   */
  LAPLACE_HD int gram_bandwidth() const { return degree; }

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the basis functions at point \f$x\f$ (derivative order 0).
   *
   * Writes the \f$d + 1\f$ non-zero values \f$B_{f, d}(x), \dots, B_{f+d, d}(x)\f$ into \c out,
   * where \f$f = s - d\f$ and \f$s\f$ is the knot interval index such that \f$t_s \le x < t_{s+1}\f$.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * Right-continuous on \f$[t_d, t_K]\f$ with left limits at \f$t_K\f$. If \f$x\f$ is NaN, writes NaN
   * to all \f$d + 1\f$ elements and returns 0. Outside \f$[lower, upper]\f$, follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d1(), expansion()
   */
  // rows: values of B^(k)(x) for the columns first..first + d; return first
  template <class T> LAPLACE_HD int row(T x, T* out) const { return row_order<0>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the first derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(1)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 1\f$ and one step of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d = 0\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row(), row_d2(), expansion_d1()
   */
  template <class T> LAPLACE_HD int row_d1(T x, T* out) const { return row_order<1>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the second derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(2)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 2\f$ and two steps of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d < 2\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d1(), row_d3(), expansion_d2()
   */
  template <class T> LAPLACE_HD int row_d2(T x, T* out) const { return row_order<2>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the third derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(3)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 3\f$ and three steps of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d < 3\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d2(), row_d4(), expansion_d3()
   */
  template <class T> LAPLACE_HD int row_d3(T x, T* out) const { return row_order<3>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the fourth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(4)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 4\f$ and four steps of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d < 4\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d3(), row_d5(), expansion_d4()
   */
  template <class T> LAPLACE_HD int row_d4(T x, T* out) const { return row_order<4>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the fifth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(5)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 5\f$ and five steps of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d < 5\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d4(), row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD int row_d5(T x, T* out) const { return row_order<5>(x, out); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the sixth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes the \f$d + 1\f$ values \f$B_{f+q, d}^{(6)}(x)\f$ for \f$q = 0, \dots, d\f$ into \c out,
   * using values of degree \f$d - 6\f$ and six steps of the derivative recursion.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. If \f$d < 6\f$, returns zeros. Outside follows \ref outside.
   * \par Measured accuracy
   * Maximum error against exact rational references is within 7.0 ulp of the row's largest value
   * for \f$d \le 5\f$ and 19.0 ulp for \f$d = 10\f$; tested to bound 12.0 ulp (\f$d \le 5\f$)
   * and 32.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, row_d5(), expansion_d6()
   */
  template <class T> LAPLACE_HD int row_d6(T x, T* out) const { return row_order<6>(x, out); }

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the basis expansion \f$\sum_{j=0}^{K-1} \beta_j B_j(x)\f$ at point \f$x\f$ (derivative order 0).
   *
   * Accumulates the linear combination across the \f$d + 1\f$ active basis functions in interval \f$s\f$.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Value of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of the row values and beta to within summation rounding.
   * \see basis_bspline, row(), expansion_d1()
   */
  // sum_j beta_j B_j^(k)(x)
  template <class T> LAPLACE_HD T expansion(T x, const T* beta) const { return expansion_order<0>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the first derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(1)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return First derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d = 0\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d1() and beta to within summation rounding.
   * \see basis_bspline, row_d1(), expansion(), expansion_d2()
   */
  template <class T> LAPLACE_HD T expansion_d1(T x, const T* beta) const { return expansion_order<1>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the second derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(2)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Second derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d < 2\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d2() and beta to within summation rounding.
   * \see basis_bspline, row_d2(), expansion_d1(), expansion_d3()
   */
  template <class T> LAPLACE_HD T expansion_d2(T x, const T* beta) const { return expansion_order<2>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the third derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(3)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Third derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d < 3\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d3() and beta to within summation rounding.
   * \see basis_bspline, row_d3(), expansion_d2(), expansion_d4()
   */
  template <class T> LAPLACE_HD T expansion_d3(T x, const T* beta) const { return expansion_order<3>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the fourth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(4)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Fourth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d < 4\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d4() and beta to within summation rounding.
   * \see basis_bspline, row_d4(), expansion_d3(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d4(T x, const T* beta) const { return expansion_order<4>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the fifth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(5)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Fifth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d < 5\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d5() and beta to within summation rounding.
   * \see basis_bspline, row_d5(), expansion_d4(), expansion_d6()
   */
  template <class T> LAPLACE_HD T expansion_d5(T x, const T* beta) const { return expansion_order<5>(x, beta); }
  /**
   * \ingroup basis_bspline
   * \brief Evaluates the sixth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{j=0}^{K-1} \beta_j B_j^{(6)}(x)\f$ across the \f$d + 1\f$ active functions.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Sixth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. If \f$d < 6\f$, returns 0. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the scalar product of row_d6() and beta to within summation rounding.
   * \see basis_bspline, row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d6(T x, const T* beta) const { return expansion_order<6>(x, beta); }

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the integrals of basis functions from \f$\text{lower}\f$ to \f$x\f$ for the active window.
   *
   * Writes the \f$d + 1\f$ values \f$\int_{\text{lower}}^x B_j(t) \, \mathrm{d}t\f$ for columns \f$j = s - d, \dots, s\f$
   * into \c out using B-splines of degree \f$d + 1\f$.
   * Preceding columns \f$j < s - d\f$ equal \ref integral_column_total(); succeeding columns \f$j > s\f$ are 0.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit.
   * \param out Output buffer of size at least row_width() (\f$d + 1\f$ elements).
   * \return First column index \f$f = s - d\f$.
   * \par Domain and special values
   * If \f$x\f$ is NaN, writes NaN and returns 0. Outside \f$[lower, upper]\f$, follows \ref outside.
   * \par Measured accuracy
   * Within 3.0 ulp of the largest column total for \f$d \le 5\f$ and 4.7 ulp for \f$d = 10\f$;
   * tested to bound 8.0 ulp (\f$d \le 5\f$) and 12.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, integral_column_total(), integral_expansion()
   */
  // int_lower^x B_j for the window j = first..first + d (see the top of the file)
  template <class T>
  LAPLACE_HD int integral_row(T x, T* out) const {
    const int d = degree;
    if (!(x == x)) return fill_nan(out, 0);
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      const int s = left ? d : dim - 1;
      if (use_end_piece(x, e)) return integral_piece(x, s, out);
      if (outside == outside_domain_rule::nan) return fill_nan(out, s - d);
      // linear: I(e) + B(e) dx + B'(e) dx^2 / 2
      T v[bspline_max_degree + 1], g[bspline_max_degree + 1];
      integral_piece(e, s, out);
      piece_row<0>(e, s, v);
      piece_row<1>(e, s, g);
      const T dx = x - e;
      for (int q = 0; q <= d; ++q) out[q] += v[q] * dx + g[q] * (dx * dx / 2);
      return s - d;
    }
    return integral_piece(x, interval(x), out);
  }

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the total integral of basis function \f$j\f$ over the domain \f$[\text{lower}, \text{upper}]\f$.
   *
   * Computes \f$\int_{\text{lower}}^{\text{upper}} B_j(t) \, \mathrm{d}t\f$, matching the value in
   * \ref integral_row() for columns preceding the active window.
   * \tparam T Floating-point scalar type (defaults to \c double).
   * \param j Basis function column index \f$j \in \{0, \dots, K - 1\}\f$.
   * \return Total integral of column \f$j\f$ over the domain.
   * \par Domain and special values
   * Valid for \f$j \in \{0, \dots, K - 1\}\f$.
   * \par Measured accuracy
   * Within 3.0 ulp of the largest column total for \f$d \le 5\f$ and 4.7 ulp for \f$d = 10\f$;
   * tested to bound 8.0 ulp (\f$d \le 5\f$) and 12.0 ulp (\f$d = 10\f$).
   * \see basis_bspline, integral_row()
   */
  // the integral of column j over [lower, upper] (= its value in
  // integral_row's convention for the columns before the window)
  template <class T = double>
  LAPLACE_HD T integral_column_total(int j) const {
    const int d = degree;
    if (j >= dim - 1 - d) {  // the last window: the integral up to upper
      T w[bspline_max_degree + 1];
      integral_piece(T(upper), dim - 1, w);
      return w[j - (dim - 1 - d)];
    }
    T c = T(knots[j + d + 1] - knots[j]) / T(d + 1);
    if (j <= d && knots[0] < lower) {
      T w[bspline_max_degree + 1];
      lower_anchor(w);
      c -= w[j];
    }
    return c;
  }

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the integrated linear expansion \f$\sum_{j=0}^{K-1} \beta_j \int_{\text{lower}}^x B_j(t) \, \mathrm{d}t\f$.
   *
   * Accumulates completed columns \f$j < f\f$ via \ref integral_column_total() and active window columns
   * \f$j \in [f, f + d]\f$ via \ref integral_row().
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Integrated expansion value at \f$x\f$.
   * \par Domain and special values
   * If \f$x\f$ or any active coefficient is NaN, returns NaN. Outside follows \ref outside.
   * \par Measured accuracy
   * Agrees with the complete sum of column integrals to within summation rounding.
   * \see basis_bspline, integral_row(), integral_column_total()
   */
  template <class T>
  LAPLACE_HD T integral_expansion(T x, const T* beta) const {
    T w[bspline_max_degree + 1];
    const int f = integral_row(x, w);
    T s = 0;
    for (int q = 0; q <= degree; ++q) s += beta[f + q] * w[q];
    if (f > 0) {
      T a[bspline_max_degree + 1];
      const bool anchored = knots[0] < lower;
      if (anchored) lower_anchor(a);
      for (int j = 0; j < f; ++j) {
        T c = T(knots[j + degree + 1] - knots[j]) / T(degree + 1);
        if (anchored && j <= degree) c -= a[j];
        s += beta[j] * c;
      }
    }
    return s;
  }

  /**
   * \ingroup basis_bspline
   * \brief Finds the knot interval index \f$s \in [d, K - 1]\f$ such that \f$t_s \le x < t_{s+1}\f$.
   *
   * Uses \f$O(1)\f$ arithmetic index for equally spaced knots (corrected against neighbouring knots)
   * or binary search for non-uniform knots. The right boundary \f$x = \text{upper}\f$ maps to \f$s = K - 1\f$.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in [\text{lower}, \text{upper}]\f$.
   * \return Knot interval index \f$s \in [d, K - 1]\f$.
   * \par Measured accuracy
   * Exact integer matching binary search.
   * \see basis_bspline
   */
  // the interval s in [d, K - 1] of a point of [lower, upper]
  template <class T>
  LAPLACE_HD int interval(T x) const {
    const int d = degree;
    if (step > 0.0) {
      T u = (x - T(lower)) / T(step);
      int s = d + (u > T(0) ? static_cast<int>(u) : 0);
      if (s > dim - 1) s = dim - 1;
      while (s > d && x < T(knots[s])) --s;
      while (s < dim - 1 && x >= T(knots[s + 1])) ++s;
      return s;
    }
    int lo = d, hi = dim;  // largest lo with t_lo <= x
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (T(knots[mid]) <= x) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  // ---- host functions (call them on bspline_basis_owner::host())

  /**
   * \ingroup basis_bspline
   * \brief Evaluates the cross Gram matrix \f$G_{kl}\f$ of derivative orders \f$k\f$ and \f$l\f$.
   *
   * Computes \f$(G_{kl})_{ij} = \int_{\text{lower}}^{\text{upper}} B_i^{(k)}(x) B_j^{(l)}(x) \, \mathrm{d}x\f$
   * using 16-node Gauss-Legendre quadrature on each knot interval \f$[t_s, t_{s+1}]\f$ in local coordinates
   * \f$u = x - t_s\f$. Output is written dense \f$K \times K\f$ in column-major order (\c out[i + j * K]).
   * \param k Derivative order of the first factor, \f$k \ge 0\f$.
   * \param l Derivative order of the second factor, \f$l \ge 0\f$.
   * \param out Output buffer of size at least \f$K \times K\f$.
   * \par Domain and special values
   * Zero matrix if \f$k > d\f$ or \f$l > d\f$. Semi-bandwidth is \f$d\f$.
   * \par Measured accuracy
   * Within 6.0 ulp of the largest matrix entry for all \f$d \le 10\f$; tested to bound 12.0 ulp.
   * \see basis_bspline, gram_bandwidth()
   */
  // G_kl = int_lower^upper B^(k) B^(l)' dx, written dense K x K, column-major
  // (out[i + j K] = int B_i^(k) B_j^(l)); exact: Gauss-Legendre with 16 nodes
  // on each knot interval integrates polynomials of degree up to 31 >= 2d.
  // The nodes are taken in the interval's own coordinate u = x - t_s, so that
  // a domain far from 0 or a narrow interval loses nothing to rounding x.
  void cross_gram(int k, int l, double* out) const {
    const int K = dim, d = degree;
    std::fill(out, out + static_cast<size_t>(K) * K, 0.0);
    if (k > d || l > d) return;
    double rk[bspline_max_degree + 1], rl[bspline_max_degree + 1];
    for (int s = d; s < K; ++s) {
      const double a = knots[s], b = knots[s + 1];
      if (!(b > a)) continue;
      const double hl = 0.5 * (b - a);
      for (int i = 0; i < 16; ++i) {
        const double u = hl + hl * laplace::math::quad::detail::legendre16(i);
        const double w = hl * laplace::math::quad::detail::legendre16(16 + i);
        piece_row_of_order(k, u, s, rk, a);
        piece_row_of_order(l, u, s, rl, a);
        const int f = s - d;
        for (int bb = 0; bb <= d; ++bb)
          for (int aa = 0; aa <= d; ++aa) out[(f + aa) + static_cast<size_t>(f + bb) * K] += w * rk[aa] * rl[bb];
      }
    }
  }

  /**
   * \ingroup basis_bspline
   * \brief Computes the Greville abscissae for all \f$K\f$ basis functions.
   *
   * Writes \f$\xi_j = \frac{1}{d} \sum_{i=1}^d t_{j+i}\f$ for \f$d \ge 1\f$, or
   * \f$\xi_j = (t_j + t_{j+1})/2\f$ for \f$d = 0\f$, for \f$j = 0, \dots, K - 1\f$ into \c out.
   * \param out Output buffer of size at least \f$K\f$.
   * \par Domain and special values
   * Satisfies linear precision \f$\sum_{j=0}^{K-1} \xi_j B_j(x) = x\f$.
   * \par Measured accuracy
   * Within floating-point arithmetic rounding of the mean.
   * \see basis_bspline
   */
  // Greville abscissae xi_j = (t_{j+1} + ... + t_{j+d})/d (d = 0: the midpoint
  // of the support), K values
  void greville(double* out) const {
    const int d = degree;
    for (int j = 0; j < dim; ++j) {
      if (d == 0) { out[j] = 0.5 * (knots[j] + knots[j + 1]); continue; }
      double s = 0.0;
      for (int i = 1; i <= d; ++i) s += knots[j + i];
      out[j] = s / d;
    }
  }

  /**
   * \ingroup basis_bspline
   * \brief Returns the points of reduced continuity \f$C^c\f$ across the domain.
   *
   * Identifies every distinct interior knot with its continuity \f$d - \mu\f$ (where \f$\mu\f$
   * is knot multiplicity). Under \c outside_domain_rule::linear and \f$d \ge 2\f$, adds the endpoints
   * \f$\text{lower}\f$ and \f$\text{upper}\f$ with continuity 1.
   * \return Vector of \ref bspline_regularity_point entries.
   * \par Measured accuracy
   * Exact identification of knot multiplicities.
   * \see basis_bspline, bspline_regularity_point
   */
  // every distinct interior knot (C^(d - mu)) and, with the linear rule and
  // d >= 2, the two ends (C^1: the second derivative jumps to zero)
  std::vector<bspline_regularity_point> regularity_points() const {
    std::vector<bspline_regularity_point> r;
    const int d = degree, n = dim + d + 1;
    if (outside == outside_domain_rule::linear && d >= 2) r.push_back({lower, 1});
    for (int j = 0; j < n;) {
      int e = j;
      while (e + 1 < n && knots[e + 1] == knots[j]) ++e;
      if (knots[j] > lower && knots[j] < upper) r.push_back({knots[j], d - (e - j + 1)});
      j = e + 1;
    }
    if (outside == outside_domain_rule::linear && d >= 2) r.push_back({upper, 1});
    return r;
  }

  // ---- internals

  template <class T>
  LAPLACE_HD int fill_nan(T* out, int first) const {
    const T nan = T(laplace::math::detail::quiet_nan());
    for (int q = 0; q <= degree; ++q) out[q] = nan;
    return first;
  }

  // the polynomial piece at the end is used: polynomial rule, or nan rule
  // within 1e-8 of the width
  template <class T>
  LAPLACE_HD bool use_end_piece(T x, T e) const {
    if (outside == outside_domain_rule::polynomial) return true;
    if (outside == outside_domain_rule::nan) return Kokkos::fabs(x - e) <= T(1e-8) * T(upper - lower);
    return false;
  }

  // BSPLVB: b[q] = B_{s-p+q, p}(x), q = 0..p, with knots t_{s+1-p}..t_{s+p};
  // x is given as an offset from origin (0: x itself)
  template <class T>
  LAPLACE_HD void bsplvb(int s, int p, T x, T* b, double origin = 0.0) const {
    T left[bspline_max_degree + 2], right[bspline_max_degree + 2];
    b[0] = T(1);
    for (int j = 1; j <= p; ++j) {
      left[j] = x - T(knots[s + 1 - j] - origin);
      right[j] = T(knots[s + j] - origin) - x;
      T saved = 0;
      for (int r = 0; r < j; ++r) {
        const T temp = b[r] / (right[r + 1] + left[j - r]);
        b[r] = saved + right[r + 1] * temp;
        saved = left[j - r] * temp;
      }
      b[j] = saved;
    }
  }

  // B^(K)_{s-d..s}(x) on interval s (x may lie outside it: the piece continued)
  template <int K, class T>
  LAPLACE_HD int piece_row(T x, int s, T* out, double origin = 0.0) const {
    const int d = degree;
    if (K > d) {
      for (int q = 0; q <= d; ++q) out[q] = T(0);
      return s - d;
    }
    bsplvb(s, d - K, x, out, origin);  // out[q] = B_{s-(d-K)+q, d-K}
    for (int p = d - K + 1; p <= d; ++p) {
      // degree p - 1 (columns s-p+1..s in out[0..p-1]) to degree p (s-p..s)
      T prev = T(0);  // out[q - 1] of degree p - 1, before it is overwritten
      for (int q = 0; q <= p; ++q) {
        const int i = s - p + q;
        const T cur = q <= p - 1 ? out[q] : T(0);
        T acc = T(0);
        if (q >= 1) {
          const double den = knots[i + p] - knots[i];
          if (den > 0.0) acc += T(p) * prev / T(den);
        }
        if (q <= p - 1) {
          const double den = knots[i + p + 1] - knots[i + 1];
          if (den > 0.0) acc -= T(p) * cur / T(den);
        }
        prev = cur;
        out[q] = acc;
      }
    }
    return s - d;
  }

  template <class T>
  LAPLACE_HD int piece_row_of_order(int k, T x, int s, T* out, double origin = 0.0) const {
    switch (k) {
      case 0: return piece_row<0>(x, s, out, origin);
      case 1: return piece_row<1>(x, s, out, origin);
      case 2: return piece_row<2>(x, s, out, origin);
      case 3: return piece_row<3>(x, s, out, origin);
      case 4: return piece_row<4>(x, s, out, origin);
      case 5: return piece_row<5>(x, s, out, origin);
      default: return piece_row<6>(x, s, out, origin);
    }
  }

  template <int K, class T>
  LAPLACE_HD int row_order(T x, T* out) const {
    const int d = degree;
    if (!(x == x)) return fill_nan(out, 0);
    if (x < lower || x > upper) {
      const bool left = x < lower;
      const T e = left ? T(lower) : T(upper);
      const int s = left ? d : dim - 1;
      if (use_end_piece(x, e)) return piece_row<K>(x, s, out);
      if (outside == outside_domain_rule::nan) return fill_nan(out, s - d);
      // linear: f(e) + f'(e) (x - e)
      if constexpr (K == 0) {
        T g[bspline_max_degree + 1];
        piece_row<0>(e, s, out);
        piece_row<1>(e, s, g);
        const T dx = x - e;
        for (int q = 0; q <= d; ++q) out[q] += g[q] * dx;
      } else if constexpr (K == 1) {
        piece_row<1>(e, s, out);
      } else {
        for (int q = 0; q <= d; ++q) out[q] = T(0);
      }
      return s - d;
    }
    return piece_row<K>(x, interval(x), out);
  }

  template <int K, class T>
  LAPLACE_HD T expansion_order(T x, const T* beta) const {
    T r[bspline_max_degree + 1];
    const int f = row_order<K>(x, r);
    T s = 0;
    for (int q = 0; q <= degree; ++q) s += beta[f + q] * r[q];
    return s;
  }

  // int_{t_j}^x B_j for the window j = s-d..s (x in interval s, or its piece
  // continued), then minus the part below lower
  template <class T>
  LAPLACE_HD int integral_piece(T x, int s, T* out) const {
    const int d = degree;
    T n[bspline_max_degree + 2];
    bsplvb(s, d + 1, x, n);  // n[q] = B_{s-d-1+q, d+1}(x)
    T tail = 0;
    for (int q = d; q >= 0; --q) {  // column j = s - d + q: sum of n[q + 1..d + 1]
      tail += n[q + 1];
      const int j = s - d + q;
      out[q] = T(knots[j + d + 1] - knots[j]) / T(d + 1) * tail;
    }
    if (s - d <= d && knots[0] < lower) {
      T a[bspline_max_degree + 1];
      lower_anchor(a);
      for (int q = 0; q <= d; ++q) {
        const int j = s - d + q;
        if (j <= d) out[q] -= a[j];
      }
    }
    return s - d;
  }

  // a[j] = int_{t_j}^lower B_j, j = 0..d (zero for clamped knots)
  template <class T>
  LAPLACE_HD void lower_anchor(T* a) const {
    const int d = degree;
    T n[bspline_max_degree + 2];
    bsplvb(d, d + 1, T(lower), n);
    T tail = 0;
    for (int q = d; q >= 0; --q) {
      tail += n[q + 1];
      a[q] = T(knots[q + d + 1] - knots[q]) / T(d + 1) * tail;
    }
  }
};

/**
 * \ingroup basis_bspline
 * \brief Owns knot storage for a B-spline basis on host and device.
 *
 * Manages allocation and lifetime of the \f$K + d + 1\f$ knot values in both host memory
 * (\c std::vector<double>) and default device memory space (\c Kokkos::View<double*, device_memory>).
 * Hands out lightweight \ref bspline_basis views via \ref host() and \ref device().
 *
 * Constructors check argument validity and throw \c std::invalid_argument on error.
 * Neither \ref host() nor \ref device() can be called on a temporary owner object.
 * \see basis_bspline, bspline_basis
 */
// Owns the knots of a bspline_basis, on the host and in the default memory
// space. The constructors check their arguments and throw
// std::invalid_argument.
class bspline_basis_owner {
 public:
  using device_memory = Kokkos::DefaultExecutionSpace::memory_space;

  /**
   * \ingroup basis_bspline
   * \brief Constructs a B-spline owner with uniform clamped knots on \f$[\text{lower}, \text{upper}]\f$.
   *
   * The interior knots are equally spaced with step \f$h = (\text{upper} - \text{lower}) / (K - d)\f$,
   * and the boundary endpoints are repeated \f$d + 1\f$ times:
   * \f$t_0 = \dots = t_d = \text{lower}\f$, \f$t_{d+q} = \text{lower} + q h\f$ for \f$q = 1, \dots, K - d - 1\f$,
   * and \f$t_K = \dots = t_{K+d} = \text{upper}\f$.
   * \param lower Lower bound of the domain.
   * \param upper Upper bound of the domain.
   * \param dim Number of basis functions \f$K \ge d + 1\f$.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \param outside Rule for evaluation outside the domain (defaults to linear).
   * \return Initialized \ref bspline_basis_owner.
   * \par Domain and special values
   * Throws \c std::invalid_argument if \c degree < 0 or \c degree > 10, if \c lower >= upper or either is not finite,
   * or if \c dim < degree + 1.
   * \par Measured accuracy
   * Exact setup of knot coordinates.
   * \see basis_bspline, uniform_extended(), clamped()
   */
  // K functions on [lower, upper], interior knots equally spaced
  // (h = (upper - lower)/(K - d)), the ends repeated d + 1 times
  static bspline_basis_owner uniform_clamped(double lower, double upper, int dim, int degree,
                                             outside_domain_rule outside = outside_domain_rule::linear) {
    check_degree(degree);
    if (!(lower < upper) || !std::isfinite(lower) || !std::isfinite(upper))
      throw std::invalid_argument("bspline: need finite lower < upper");
    if (dim < degree + 1) throw std::invalid_argument("bspline: need dim >= degree + 1");
    const int nint = dim - degree;
    const double h = (upper - lower) / nint;
    std::vector<double> t;
    for (int j = 0; j <= degree; ++j) t.push_back(lower);
    for (int q = 1; q < nint; ++q) t.push_back(lower + q * h);
    for (int j = 0; j <= degree; ++j) t.push_back(upper);
    return bspline_basis_owner(std::move(t), degree, h, outside);
  }

  /**
   * \ingroup basis_bspline
   * \brief Constructs a B-spline owner with uniform extended knots on \f$[\text{lower}, \text{upper}]\f$.
   *
   * All knots are equally spaced with step \f$h = (\text{upper} - \text{lower}) / (K - d)\f$, extending
   * \f$d\f$ steps beyond each end (the Eilers and Marx parametrization):
   * \f$t_j = \text{lower} + (j - d) h\f$ for \f$j < K\f$ and \f$t_j = \text{upper} + (j - K) h\f$ for \f$j \ge K\f$.
   * Every basis function is a translate of a single cardinal B-spline.
   * \param lower Lower bound of the domain.
   * \param upper Upper bound of the domain.
   * \param dim Number of basis functions \f$K \ge d + 1\f$.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \param outside Rule for evaluation outside the domain (defaults to linear).
   * \return Initialized \ref bspline_basis_owner.
   * \par Domain and special values
   * Throws \c std::invalid_argument if \c degree < 0 or \c degree > 10, if \c lower >= upper or either is not finite,
   * or if \c dim < degree + 1.
   * \par Measured accuracy
   * Exact setup of knot coordinates.
   * \see basis_bspline, uniform_clamped()
   */
  // K functions on [lower, upper], all knots equally spaced with step
  // h = (upper - lower)/(K - d), d of them beyond each end (Eilers and Marx):
  // every function is a translate of one cardinal B-spline
  static bspline_basis_owner uniform_extended(double lower, double upper, int dim, int degree,
                                              outside_domain_rule outside = outside_domain_rule::linear) {
    check_degree(degree);
    if (!(lower < upper) || !std::isfinite(lower) || !std::isfinite(upper))
      throw std::invalid_argument("bspline: need finite lower < upper");
    if (dim < degree + 1) throw std::invalid_argument("bspline: need dim >= degree + 1");
    const int nint = dim - degree;
    const double h = (upper - lower) / nint;
    std::vector<double> t(dim + degree + 1);
    for (int j = 0; j <= dim + degree; ++j) {
      if (j < dim) t[j] = lower + (j - degree) * h;
      else t[j] = upper + (j - dim) * h;
    }
    t[degree] = lower;
    return bspline_basis_owner(std::move(t), degree, h, outside);
  }

  /**
   * \ingroup basis_bspline
   * \brief Constructs a B-spline owner from user-specified interior knots, clamped at the boundaries.
   *
   * Repeats \c lower and \c upper each \f$d + 1\f$ times at the boundaries around the given interior knots.
   * \param lower Lower bound of the domain.
   * \param upper Upper bound of the domain.
   * \param interior Non-decreasing vector of interior knots strictly inside \f$(\text{lower}, \text{upper})\f$.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \param outside Rule for evaluation outside the domain (defaults to linear).
   * \return Initialized \ref bspline_basis_owner.
   * \par Domain and special values
   * Throws \c std::invalid_argument if \c degree < 0 or \c degree > 10, if any interior knot does not lie
   * strictly inside \f$(\text{lower}, \text{upper})\f$, if knots are decreasing, or if any knot value appears
   * more than \f$d + 1\f$ times.
   * \par Measured accuracy
   * Exact setup of knot coordinates.
   * \see basis_bspline, uniform_clamped(), from_knots()
   */
  // interior knots (non-decreasing, inside (lower, upper), repeats allowed up
  // to d + 1), the ends repeated d + 1 times
  static bspline_basis_owner clamped(double lower, double upper, const std::vector<double>& interior, int degree,
                                     outside_domain_rule outside = outside_domain_rule::linear) {
    check_degree(degree);
    std::vector<double> t;
    for (int j = 0; j <= degree; ++j) t.push_back(lower);
    for (double v : interior) {
      if (!(v > lower && v < upper)) throw std::invalid_argument("bspline: interior knots must lie inside (lower, upper)");
      t.push_back(v);
    }
    for (int j = 0; j <= degree; ++j) t.push_back(upper);
    return bspline_basis_owner(std::move(t), degree, 0.0, outside);
  }

  /**
   * \ingroup basis_bspline
   * \brief Constructs a B-spline owner from a complete knot vector.
   *
   * Accepts the full vector \f$t_0, \dots, t_{K+d}\f$. The domain is \f$[t_d, t_K]\f$.
   * \param knots Complete non-decreasing knot vector of at least \f$2(d + 1)\f$ finite knots.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \param outside Rule for evaluation outside the domain (defaults to linear).
   * \return Initialized \ref bspline_basis_owner.
   * \par Domain and special values
   * Throws \c std::invalid_argument if \c degree < 0 or \c degree > 10, if \c knots contains non-finite values,
   * if knots are decreasing, if any knot appears more than \f$d + 1\f$ times, or if \f$t_d \ge t_K\f$.
   * \par Measured accuracy
   * Exact setup of knot coordinates.
   * \see basis_bspline, clamped()
   */
  // the full knot vector t_0..t_{K+d}
  static bspline_basis_owner from_knots(const std::vector<double>& knots, int degree,
                                        outside_domain_rule outside = outside_domain_rule::linear) {
    check_degree(degree);
    return bspline_basis_owner(knots, degree, 0.0, outside);
  }

  /**
   * \ingroup basis_bspline
   * \brief Returns an evaluation view pointing to host knot storage.
   *
   * Can be used for host evaluation, Gram matrix generation, Greville abscissae, and regularity points.
   * \return \ref bspline_basis view pointing into this owner's host knot storage.
   * \par Domain and special values
   * Cannot be called on temporary rvalue owner (\c const&& overload is deleted).
   * \see basis_bspline, device()
   */
  // the struct points into this owner's storage: not available on a
  // temporary owner, whose knots would be gone before the struct is used
  bspline_basis host() const& {
    bspline_basis b = proto_;
    b.knots = knots_.data();
    return b;
  }
  /**
   * \ingroup basis_bspline
   * \brief Returns an evaluation view pointing to device knot storage.
   *
   * Can be passed by value into Kokkos device kernels.
   * \return \ref bspline_basis view pointing into this owner's device knot storage.
   * \par Domain and special values
   * Cannot be called on temporary rvalue owner (\c const&& overload is deleted).
   * \see basis_bspline, host()
   */
  bspline_basis device() const& {
    bspline_basis b = proto_;
    b.knots = device_knots_.data();
    return b;
  }
  bspline_basis host() const&& = delete;
  bspline_basis device() const&& = delete;
  /**
   * \ingroup basis_bspline
   * \brief Returns a const reference to the host knot vector.
   * \return Reference to \c std::vector<double> containing the \f$K + d + 1\f$ knots.
   * \see basis_bspline
   */
  const std::vector<double>& knots() const { return knots_; }

 private:
  bspline_basis_owner(std::vector<double> t, int degree, double step, outside_domain_rule outside)
      : knots_(std::move(t)) {
    const int n = static_cast<int>(knots_.size());
    const int dim = n - degree - 1;
    if (dim < degree + 1) throw std::invalid_argument("bspline: need at least 2 (d + 1) knots");
    for (double v : knots_)
      if (!std::isfinite(v)) throw std::invalid_argument("bspline: knots must be finite");
    for (int j = 0; j + 1 < n; ++j)
      if (knots_[j + 1] < knots_[j]) throw std::invalid_argument("bspline: knots must be non-decreasing");
    for (int j = 0; j < n;) {
      int e = j;
      while (e + 1 < n && knots_[e + 1] == knots_[j]) ++e;
      if (e - j + 1 > degree + 1)
        throw std::invalid_argument("bspline: a knot value appears more than degree + 1 times");
      j = e + 1;
    }
    proto_.degree = degree;
    proto_.dim = dim;
    proto_.lower = knots_[degree];
    proto_.upper = knots_[dim];
    proto_.step = step;
    proto_.outside = outside;
    if (!(proto_.lower < proto_.upper)) throw std::invalid_argument("bspline: empty domain");
    device_knots_ = Kokkos::View<double*, device_memory>("laplace::basis::bspline_knots", n);
    auto m = Kokkos::create_mirror_view(device_knots_);
    for (int j = 0; j < n; ++j) m(j) = knots_[j];
    Kokkos::deep_copy(device_knots_, m);
  }

  static void check_degree(int degree) {
    if (degree < 0 || degree > bspline_max_degree)
      throw std::invalid_argument("bspline: degree must be in 0.." + std::to_string(bspline_max_degree));
  }

  std::vector<double> knots_;
  Kokkos::View<double*, device_memory> device_knots_;
  bspline_basis proto_;
};

/**
 * \ingroup basis_bspline
 * \brief Computes \f$m\f$ interior knots at sample quantiles of \f$x\f$ using R's type 7.
 *
 * Places knots at quantiles \f$j / (m + 1)\f$ for \f$j = 1, \dots, m\f$ via linear interpolation
 * between order statistics of \f$x\f$.
 * \param x Sample data vector.
 * \param m Number of interior knots, \f$m \ge 0\f$.
 * \return Vector of \f$m\f$ strictly increasing interior knots.
 * \par Domain and special values
 * Throws \c std::invalid_argument if \f$m < 0\f$, if \f$x\f$ is empty or contains non-finite values,
 * if two knots coincide, or if any knot falls on \f$\min(x)\f$ or \f$\max(x)\f$ due to sample ties.
 * \par Measured accuracy
 * Exact floating-point linear interpolation between sorted data points.
 * \see basis_bspline, bspline_basis_owner::clamped()
 */
// m interior knots at the quantiles j/(m + 1), j = 1..m, of x (R's type 7:
// linear interpolation between order statistics). Throws when two knots
// coincide or a knot falls on min(x) or max(x) (ties in x): choose fewer
// knots, or place them by hand.
inline std::vector<double> quantile_knots(std::vector<double> x, int m) {
  if (m < 0) throw std::invalid_argument("quantile_knots: m must be >= 0");
  if (x.empty()) throw std::invalid_argument("quantile_knots: x is empty");
  for (double v : x)
    if (!std::isfinite(v)) throw std::invalid_argument("quantile_knots: x must be finite");
  std::sort(x.begin(), x.end());
  const int n = static_cast<int>(x.size());
  std::vector<double> k;
  for (int j = 1; j <= m; ++j) {
    const double h = (n - 1) * static_cast<double>(j) / (m + 1);
    const int lo = static_cast<int>(std::floor(h));
    const int hi = std::min(lo + 1, n - 1);
    const double v = x[lo] + (h - lo) * (x[hi] - x[lo]);
    if (!(v > x.front() && v < x.back()))
      throw std::invalid_argument("quantile_knots: a knot falls on min(x) or max(x)");
    if (!k.empty() && !(v > k.back())) throw std::invalid_argument("quantile_knots: two knots coincide");
    k.push_back(v);
  }
  return k;
}

}  // namespace laplace::basis
