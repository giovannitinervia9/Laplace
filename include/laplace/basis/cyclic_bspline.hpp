// Cyclic (periodic) B-spline bases (laplace::basis); design decisions C1-C10
// of 2026-11-14 (diario.md).
//
// Knots and functions. A cyclic B-spline basis of degree d and dimension K
// has K knots t_0 <= t_1 <= ... <= t_{K-1} in one period [lower, upper),
// lower = t_0, upper = lower + omega, omega > 0 the period, extended
// periodically, t_{j+mK} = t_j + m omega for every integer m. Column
// j = 0..K-1 is the periodic function
//
//   b_j(x) = sum_m B_{j+mK}(x),
//
// where B_i is the B-spline of degree d on the knots t_i..t_{i+d+1} of the
// extended sequence. So b_j is supported on [t_j, t_{j+d+1}] modulo omega,
// the K functions sum to 1, and their span is the space of splines of degree
// d on these knots that are periodic with their first d - 1 derivatives
// (at a knot of multiplicity mu, C^(d - mu), lower included). A value may
// appear at most d + 1 times in one period, and K >= d + 1, so that the d + 1
// columns of a row are distinct.
//
// Construction (C1). The basis holds an ordinary bspline_basis `ext` on the
// K + 2d + 1 knots t_{-d}..t_{K+d}: its K + d columns live on its domain
// [t_0, t_K] = [lower, upper], exactly one period, and ext column e is the
// piece of b_c on that period with c = (e - d) mod K. Columns e < d and
// e + K are the two pieces of the same cyclic function c = K - d + e, the one
// that straddles lower.
//
// Evaluation (in kernels). An x in [lower, upper) is used as it is; any other
// x is reduced to one period (C5): with
// u = x - lower, r = fmod(u, omega) (exact in IEEE arithmetic), r < 0 moved
// up by omega, and y = lower + r, a y that rounds to upper becomes lower. The
// only roundings are those of x - lower and lower + r. Rows are ext's rows
// at y (BSPLVB, BSPLVD; orders above d are zero); ext's first column f_e
// becomes f = (f_e - d) mod K, and value i belongs to column (f + i) mod K
// (C2). There is no outside_domain_rule: every real x is in the domain.
//
// The integral from lower (C6). With x - lower = n omega + r (n whole),
//   int_lower^x b_j = n c_j + int_lower^(lower + r) b_j,
//   c_j = (t_{j+d+1} - t_j)/(d + 1) = int over one period of b_j
// (integral_column_total). The second term is the sum of the integrals of
// the ext pieces of b_j from lower to y, each taken from ext's integral
// window or, for a piece that has finished before y, its total over
// [lower, upper] (piece_totals, a table of K + d values made once by the
// owner). integral_row writes the d + 1 columns of the row's window;
// integral_column(j, x) gives any column, integral_expansion all of them.
//
// Gram matrices (host, C7): G_kl of the cyclic basis over one period is
// ext's exact Gram over [lower, upper] folded, G[c(a), c(b)] += G_ext[a, b];
// for K > 2d + 1 it is banded with bandwidth d plus the wrapped corners, and
// gram_bandwidth() = K - 1 (the corners). With equally spaced knots it is
// circulant.
#pragma once

#include <laplace/basis/bspline.hpp>
#include <laplace/basis/core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace laplace::basis {

/**
 * \ingroup basis_cyclic_bspline
 * \brief Lightweight view struct for cyclic B-spline evaluation on host and device.
 *
 * Holds the degree, dimension, period boundaries, and view pointers managed by
 * cyclic_bspline_basis_owner. Copied by value into Kokkos kernels via
 * cyclic_bspline_basis_owner::device().
 * \see basis_cyclic_bspline, cyclic_bspline_basis_owner
 */
struct cyclic_bspline_basis {
  bspline_basis ext;  // on t_{-d}..t_{K+d}: dimension K + d, domain [lower, upper]
  const double* piece_totals = nullptr;  // int_lower^upper of ext column e, e = 0..K + d - 1
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   */
  int degree = 3;
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Total number of basis functions \f$K \ge d + 1\f$.
   */
  int dim = 0;  // K
  /**
   * \ingroup basis_cyclic_bspline
   * \var cyclic_bspline_basis::lower
   * \brief Lower boundary of one period, \f$x_{\mathrm{lo}} = t_0\f$.
   */
  /**
   * \ingroup basis_cyclic_bspline
   * \var cyclic_bspline_basis::upper
   * \brief Upper boundary of one period, \f$x_{\mathrm{hi}} = x_{\mathrm{lo}} + \omega\f$.
   */
  /**
   * \ingroup basis_cyclic_bspline
   * \var cyclic_bspline_basis::period
   * \brief Fundamental period \f$\omega > 0\f$.
   */
  double lower = 0.0, upper = 1.0, period = 1.0;

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Total number of cyclic basis functions \f$K \ge d + 1\f$.
   * \return Basis dimension \f$K\f$.
   * \see basis_cyclic_bspline, row_width()
   */
  LAPLACE_HD int dimension() const { return dim; }
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Number of active basis functions in an evaluated row, \f$w = d + 1\f$.
   * \return Row width \f$d + 1\f$ in \f$\{1, \dots, 11\}\f$.
   * \see basis_cyclic_bspline, dimension(), row()
   */
  LAPLACE_HD int row_width() const { return degree + 1; }
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Scratch buffer size needed for evaluation (always 0).
   * \return Zero.
   * \see basis_cyclic_bspline
   */
  LAPLACE_HD int scratch_size() const { return 0; }
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Semi-bandwidth of cross Gram matrices \f$G_{kl}\f$.
   * \return Semi-bandwidth equal to \f$K - 1\f$ due to wrapped corner entries.
   * \see basis_cyclic_bspline, cross_gram()
   */
  int gram_bandwidth() const { return dim - 1; }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates the \f$d + 1\f$ active cyclic basis functions \f$b_{(f+i) \bmod K}(x)\f$ for \f$i = 0, \dots, d\f$.
   *
   * Reduces \f$x\f$ to \f$[x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$ via reduce() and writes \f$d + 1\f$ values into `out`.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point in \f$\mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return First active wrapped column index \f$f \in \{0, \dots, K - 1\}\f$.
   * \par Domain and special values
   * Defined on all of \f$\mathbb{R}\f$. Writes quiet NaN and returns 0 if \f$x\f$ is NaN or infinite.
   * \par Measured accuracy
   * Within 8 ulp of the row's largest value for \f$d \le 5\f$ and 11 ulp for \f$d = 10\f$ (test bounds 12 and 17 ulp).
   * \see basis_cyclic_bspline, row_d1(), expansion(), reduce()
   */
  // rows: values of b^(k) for the columns (first + i) mod K, i = 0..d; return first
  template <class T> LAPLACE_HD int row(T x, T* out) const { return row_order<0>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates first-derivative row \f$b_{(f+i) \bmod K}^{(1)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <1\f$.
  /// \see basis_cyclic_bspline, row(), expansion_d1()
  template <class T> LAPLACE_HD int row_d1(T x, T* out) const { return row_order<1>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates second-derivative row \f$b_{(f+i) \bmod K}^{(2)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <2\f$.
  /// \see basis_cyclic_bspline, row_d1(), expansion_d2()
  template <class T> LAPLACE_HD int row_d2(T x, T* out) const { return row_order<2>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates third-derivative row \f$b_{(f+i) \bmod K}^{(3)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <3\f$.
  /// \see basis_cyclic_bspline, row_d2(), expansion_d3()
  template <class T> LAPLACE_HD int row_d3(T x, T* out) const { return row_order<3>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates fourth-derivative row \f$b_{(f+i) \bmod K}^{(4)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <4\f$.
  /// \see basis_cyclic_bspline, row_d3(), expansion_d4()
  template <class T> LAPLACE_HD int row_d4(T x, T* out) const { return row_order<4>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates fifth-derivative row \f$b_{(f+i) \bmod K}^{(5)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <5\f$.
  /// \see basis_cyclic_bspline, row_d4(), expansion_d5()
  template <class T> LAPLACE_HD int row_d5(T x, T* out) const { return row_order<5>(x, out); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates sixth-derivative row \f$b_{(f+i) \bmod K}^{(6)}(x)\f$ (\f$i = 0, \dots, d\f$); the values are 0 if \f$d <6\f$.
  /// \see basis_cyclic_bspline, row_d5(), expansion_d6()
  template <class T> LAPLACE_HD int row_d6(T x, T* out) const { return row_order<6>(x, out); }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates the cyclic basis expansion \f$\sum_{j=0}^{K-1} \beta_j b_j(x)\f$ at \f$x\f$.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point in \f$\mathbb{R}\f$.
   * \param beta Array of \f$K\f$ coefficients.
   * \return Value of the expansion at \f$x\f$.
   * \par Domain and special values
   * Returns quiet NaN if \f$x\f$ is NaN or infinite.
   * \par Measured accuracy
   * Checked in the tests against the sum of row() times the wrapped `beta`.
   * \see basis_cyclic_bspline, row(), expansion_d1()
   */
  // sum_j beta_j b_j^(k)(x)
  template <class T> LAPLACE_HD T expansion(T x, const T* beta) const { return expansion_order<0>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the first-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(1)}(x)\f$ (0 if \f$d < 1\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d1(), expansion()
  template <class T> LAPLACE_HD T expansion_d1(T x, const T* beta) const { return expansion_order<1>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the second-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(2)}(x)\f$ (0 if \f$d < 2\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d2(), expansion_d1()
  template <class T> LAPLACE_HD T expansion_d2(T x, const T* beta) const { return expansion_order<2>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the third-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(3)}(x)\f$ (0 if \f$d < 3\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d3(), expansion_d2()
  template <class T> LAPLACE_HD T expansion_d3(T x, const T* beta) const { return expansion_order<3>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the fourth-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(4)}(x)\f$ (0 if \f$d < 4\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d4(), expansion_d3()
  template <class T> LAPLACE_HD T expansion_d4(T x, const T* beta) const { return expansion_order<4>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the fifth-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(5)}(x)\f$ (0 if \f$d < 5\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d5(), expansion_d4()
  template <class T> LAPLACE_HD T expansion_d5(T x, const T* beta) const { return expansion_order<5>(x, beta); }
  /// \ingroup basis_cyclic_bspline
  /// \brief Evaluates the sixth-derivative expansion \f$\sum_{j=0}^{K-1} \beta_j b_j^{(6)}(x)\f$ (0 if \f$d < 6\f$; quiet NaN for non-finite \f$x\f$).
  /// \see basis_cyclic_bspline, row_d6(), expansion_d5()
  template <class T> LAPLACE_HD T expansion_d6(T x, const T* beta) const { return expansion_order<6>(x, beta); }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates integrals \f$\int_{x_{\mathrm{lo}}}^x b_j(t)\,\mathrm{d}t\f$ for the active window \f$j = (f + i) \bmod K\f$, \f$i = 0, \dots, d\f$.
   *
   * Uses \f$\int_{x_{\mathrm{lo}}}^x b_j(t)\,\mathrm{d}t = n\,c_j + \int_{x_{\mathrm{lo}}}^{x_{\mathrm{lo}} + r} b_j(t)\,\mathrm{d}t\f$ with \f$x - x_{\mathrm{lo}} = n\omega + r\f$.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit in \f$\mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return First active wrapped column index \f$f \in \{0, \dots, K - 1\}\f$.
   * \par Domain and special values
   * Writes quiet NaN and returns 0 if \f$x\f$ is NaN or infinite.
   * \par Measured accuracy
   * Within 6 ulp of the largest column total for all \f$d \le 10\f$ (test bound 9 ulp); agrees bit for bit with integral_column().
   * \see basis_cyclic_bspline, integral_column(), integral_column_total(), integral_expansion()
   */
  // int_lower^x b_j for the columns (first + i) mod K, i = 0..d; return first
  template <class T>
  LAPLACE_HD int integral_row(T x, T* out) const {
    if (!Kokkos::isfinite(x)) return ext.fill_nan(out, 0);
    T n;
    const T y = reduce(x, n);
    T w[bspline_max_degree + 1];
    const int fe = ext.integral_row(y, w);
    const int f = cyclic_column(fe);
    for (int q = 0; q <= degree; ++q) {
      int c = f + q;
      if (c >= dim) c -= dim;
      out[q] = column_integral(c, n, fe, w);
    }
    return f;
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates the integral \f$\int_{x_{\mathrm{lo}}}^x b_j(t)\,\mathrm{d}t\f$ for any column \f$j\f$.
   *
   * Evaluates \f$n\,c_j + \int_{x_{\mathrm{lo}}}^{x_{\mathrm{lo}} + r} b_j(t)\,\mathrm{d}t\f$ where \f$x - x_{\mathrm{lo}} = n\omega + r\f$.
   * \tparam T Floating-point scalar type.
   * \param j Basis function column index \f$j \in \{0, \dots, K - 1\}\f$.
   * \param x Upper integration limit in \f$\mathbb{R}\f$.
   * \return Value of \f$\int_{x_{\mathrm{lo}}}^x b_j(t)\,\mathrm{d}t\f$.
   * \par Domain and special values
   * Returns quiet NaN if \f$x\f$ is NaN or infinite.
   * \par Measured accuracy
   * Within 6 ulp of the largest column total for all \f$d \le 10\f$ (test bound 9 ulp).
   * \see basis_cyclic_bspline, integral_row(), integral_column_total()
   */
  // int_lower^x b_j for any column j
  template <class T>
  LAPLACE_HD T integral_column(int j, T x) const {
    if (!Kokkos::isfinite(x)) return T(laplace::math::detail::quiet_nan());
    T n;
    const T y = reduce(x, n);
    T w[bspline_max_degree + 1];
    const int fe = ext.integral_row(y, w);
    return column_integral(j, n, fe, w);
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates the integrated expansion \f$\sum_{j=0}^{K-1} \beta_j \int_{x_{\mathrm{lo}}}^x b_j(t)\,\mathrm{d}t\f$.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit in \f$\mathbb{R}\f$.
   * \param beta Array of \f$K\f$ coefficients.
   * \return Integrated expansion value at \f$x\f$.
   * \par Domain and special values
   * Returns quiet NaN if \f$x\f$ is NaN or infinite.
   * \par Measured accuracy
   * Checked in the tests against the weighted sum of integral_column() over the \f$K\f$ columns.
   * \see basis_cyclic_bspline, integral_column(), integral_row()
   */
  template <class T>
  LAPLACE_HD T integral_expansion(T x, const T* beta) const {
    if (!Kokkos::isfinite(x)) return T(laplace::math::detail::quiet_nan());
    T n;
    const T y = reduce(x, n);
    T w[bspline_max_degree + 1];
    const int fe = ext.integral_row(y, w);
    T s = 0;
    for (int j = 0; j < dim; ++j) s += beta[j] * column_integral(j, n, fe, w);
    return s;
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Returns the integral \f$c_j = \frac{t_{j+d+1} - t_j}{d + 1}\f$ of \f$b_j\f$ over one period.
   * \tparam T Floating-point scalar type (defaults to `double`).
   * \param j Basis function column index \f$j \in \{0, \dots, K - 1\}\f$.
   * \return Period integral \f$c_j = \int_{x_{\mathrm{lo}}}^{x_{\mathrm{hi}}} b_j(t)\,\mathrm{d}t\f$.
   * \par Measured accuracy
   * Within 3 ulp of the largest column total for all \f$d \le 10\f$ (test bound 9 ulp).
   * \see basis_cyclic_bspline, integral_column(), integral_row()
   */
  // c_j = (t_{j+d+1} - t_j)/(d + 1), the integral of b_j over one period
  template <class T = double>
  LAPLACE_HD T integral_column_total(int j) const {
    return T(ext.knots[j + 2 * degree + 1] - ext.knots[j + degree]) / T(degree + 1);
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Reduces \f$x\f$ modulo \f$\omega\f$ into \f$[x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$, returning \f$y = x_{\mathrm{lo}} + r\f$ with \f$x - x_{\mathrm{lo}} = n\omega + r\f$.
   *
   * Leaves \f$x \in [x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$ unchanged with \f$n = 0\f$; otherwise computes exact \f$r = \operatorname{fmod}(x - x_{\mathrm{lo}}, \omega)\f$, adjusts negative \f$r\f$ by \f$\omega\f$, and wraps \f$y\f$ that rounds to \f$x_{\mathrm{hi}}\f$ back to \f$x_{\mathrm{lo}}\f$.
   * \tparam T Floating-point scalar type.
   * \param x Input coordinate in \f$\mathbb{R}\f$.
   * \param n Output integer-valued period count (\f$n < 0\f$ for \f$x < x_{\mathrm{lo}}\f$).
   * \return Reduced coordinate \f$y \in [x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$.
   * \par Rounding
   * Only \f$x - x_{\mathrm{lo}}\f$ and \f$x_{\mathrm{lo}} + r\f$ round; exact when \f$x \in [x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$.
   * \see basis_cyclic_bspline, row(), integral_row()
   */
  // x reduced to [lower, upper): returns y = lower + r, with x - lower = n omega + r
  template <class T>
  LAPLACE_HD T reduce(T x, T& n) const {
    if (x >= T(lower) && x < T(upper)) {  // already in the period: nothing to round
      n = T(0);
      return x;
    }
    const T w = T(period);
    const T u = x - T(lower);
    T r = Kokkos::fmod(u, w);
    n = Kokkos::round((u - r) / w);
    if (r < T(0)) {
      r += w;
      n -= T(1);
    }
    T y = T(lower) + r;
    if (y >= T(upper)) {
      y = T(lower);
      n += T(1);
    }
    return y;
  }

  // ---- host functions (call them on cyclic_bspline_basis_owner::host())

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Evaluates the cross Gram matrix \f$(G_{kl})_{ij} = \int_{x_{\mathrm{lo}}}^{x_{\mathrm{hi}}} b_i^{(k)}(x)\,b_j^{(l)}(x)\,\mathrm{d}x\f$ on the host.
   *
   * Folds the extended B-spline cross Gram matrix into dense \f$K \times K\f$ column-major storage (`out[i + j * K]`), with semi-bandwidth \f$d\f$ plus wrapped corners for \f$K > 2d + 1\f$.
   * \param k Derivative order of the first factor, \f$k \ge 0\f$.
   * \param l Derivative order of the second factor, \f$l \ge 0\f$.
   * \param out Output buffer of size at least \f$K^2\f$.
   * \par Domain and special values
   * Writes zeros if \f$k > d\f$ or \f$l > d\f$.
   * \par Measured accuracy
   * Within 8 ulp (\f$d \le 5\f$) and 9 ulp (\f$d = 10\f$) of the largest matrix entry (test bounds 12 and 14 ulp).
   * \see basis_cyclic_bspline, gram_bandwidth()
   */
  // G_kl = int_lower^upper b^(k) b^(l)' dx, written dense K x K, column-major
  // (out[i + j K] = int b_i^(k) b_j^(l)); ext's exact Gram, folded
  void cross_gram(int k, int l, double* out) const {
    const int K = dim, E = ext.dim;
    std::vector<double> g(static_cast<size_t>(E) * E);
    ext.cross_gram(k, l, g.data());
    std::fill(out, out + static_cast<size_t>(K) * K, 0.0);
    for (int b = 0; b < E; ++b)
      for (int a = 0; a < E; ++a)
        out[cyclic_column(a) + static_cast<size_t>(cyclic_column(b)) * K] += g[a + static_cast<size_t>(b) * E];
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Computes the Greville abscissae \f$\xi_0, \dots, \xi_{K-1}\f$ reduced to \f$[x_{\mathrm{lo}}, x_{\mathrm{hi}})\f$ on the host.
   *
   * Evaluates \f$\xi_j = \frac{1}{d} \sum_{i=1}^d t_{j+i}\f$ for \f$d \ge 1\f$ and \f$\xi_j = (t_j + t_{j+1}) / 2\f$ for \f$d = 0\f$, then reduces each via reduce().
   * \param out Output buffer of size at least \f$K\f$.
   * \see basis_cyclic_bspline, reduce()
   */
  // Greville abscissae xi_j = (t_{j+1} + ... + t_{j+d})/d (d = 0: the midpoint
  // of the support), reduced to [lower, upper), K values
  void greville(double* out) const {
    const int d = degree;
    for (int j = 0; j < dim; ++j) {
      const double* t = ext.knots + j + d;  // t[i] = t_{j+i}
      double s;
      if (d == 0) s = 0.5 * (t[0] + t[1]);
      else {
        s = 0.0;
        for (int i = 1; i <= d; ++i) s += t[i];
        s /= d;
      }
      double n;
      out[j] = reduce(s, n);
    }
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Returns the points of reduced continuity \f$C^{d - \mu}\f$ over one period on the host.
   *
   * Lists every distinct knot of one period, including `lower`, with continuity \f$d - \mu\f$ where \f$\mu\f$ is the knot multiplicity.
   * \return Vector of bspline_regularity_point entries.
   * \see basis_cyclic_bspline, bspline_regularity_point
   */
  // every distinct knot of one period, lower included, with its continuity
  // C^(d - mu)
  std::vector<bspline_regularity_point> regularity_points() const {
    std::vector<bspline_regularity_point> r;
    const double* t = ext.knots + degree;  // t_0..t_{K-1}
    for (int j = 0; j < dim;) {
      int e = j;
      while (e + 1 < dim && t[e + 1] == t[j]) ++e;
      r.push_back({t[j], degree - (e - j + 1)});
      j = e + 1;
    }
    return r;
  }

  // ---- internals

  // the cyclic column of ext column e
  LAPLACE_HD int cyclic_column(int e) const { return e >= degree ? e - degree : e - degree + dim; }

  // int_lower^y of ext column e, given ext's integral window (first fe) at y
  template <class T>
  LAPLACE_HD T piece_integral(int e, int fe, const T* w) const {
    if (e < fe) return T(piece_totals[e]);  // finished before y
    if (e <= fe + degree) return w[e - fe];
    return T(0);  // not started
  }

  // n c_j + int_lower^y b_j: the ext pieces of b_j are e = j + d and, for
  // j >= K - d, e = j + d - K
  template <class T>
  LAPLACE_HD T column_integral(int j, T n, int fe, const T* w) const {
    T v = n * integral_column_total<T>(j) + piece_integral(j + degree, fe, w);
    if (j >= dim - degree) v += piece_integral(j + degree - dim, fe, w);
    return v;
  }

  template <int K, class T>
  LAPLACE_HD int row_order(T x, T* out) const {
    if (!Kokkos::isfinite(x)) return ext.fill_nan(out, 0);
    T n;
    const T y = reduce(x, n);
    return cyclic_column(ext.template row_order<K>(y, out));
  }

  template <int K, class T>
  LAPLACE_HD T expansion_order(T x, const T* beta) const {
    T r[bspline_max_degree + 1];
    const int f = row_order<K>(x, r);
    T s = 0;
    for (int q = 0; q <= degree; ++q) {
      int c = f + q;
      if (c >= dim) c -= dim;
      s += beta[c] * r[q];
    }
    return s;
  }
};

/**
 * \ingroup basis_cyclic_bspline
 * \brief Owns host and device storage for a cyclic_bspline_basis.
 *
 * Manages the extended knot vector and precomputed full-period piece integrals on host and default Kokkos device memory spaces, handing out lightweight cyclic_bspline_basis views via host() and device().
 * \see basis_cyclic_bspline, cyclic_bspline_basis
 */
// Owns the knots of a cyclic_bspline_basis (through the bspline_basis_owner
// of its extended knots), on the host and in the default memory space. The
// constructors check their arguments and throw std::invalid_argument.
class cyclic_bspline_basis_owner {
 public:
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Constructs a cyclic B-spline owner with \f$K\f$ equally spaced knots on \f$[x_{\mathrm{lo}}, x_{\mathrm{lo}} + \omega)\f$.
   *
   * Places knots at step \f$h = \omega / K\f$ so every basis function is a translate of one cardinal B-spline.
   * \param lower Lower boundary of one period, \f$x_{\mathrm{lo}} = t_0\f$.
   * \param period Fundamental period \f$\omega > 0\f$.
   * \param dim Number of basis functions \f$K \ge d + 1\f$.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \return Initialized cyclic_bspline_basis_owner.
   * \par Exceptions
   * Throws `std::invalid_argument` if `degree` is outside \f$\{0, \dots, 10\}\f$, `lower` or `period` is non-finite, \f$\omega \le 0\f$, `lower + period` rounds to `lower` or overflows, or \f$K < d + 1\f$.
   * \see basis_cyclic_bspline, from_knots()
   */
  // K functions on [lower, lower + period), knots equally spaced with step
  // h = period/K: every function is a translate of one cardinal B-spline
  static cyclic_bspline_basis_owner uniform(double lower, double period, int dim, int degree) {
    check(lower, period, dim, degree);
    return cyclic_bspline_basis_owner(
        bspline_basis_owner::uniform_extended(lower, lower + period, dim + degree, degree, outside_domain_rule::polynomial),
        period, dim, degree);
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Constructs a cyclic B-spline owner from the \f$K\f$ knots of one period.
   *
   * Sets \f$x_{\mathrm{lo}} = t_0\f$ (`knots[0]`) and extends the non-decreasing knots periodically with period \f$\omega\f$.
   * \param knots Non-decreasing finite knots \f$t_0, \dots, t_{K-1}\f$ in \f$[t_0, t_0 + \omega)\f$ with multiplicity at most \f$d + 1\f$ across the wrap.
   * \param period Fundamental period \f$\omega > 0\f$.
   * \param degree Polynomial degree \f$d \in \{0, \dots, 10\}\f$.
   * \return Initialized cyclic_bspline_basis_owner.
   * \par Exceptions
   * Throws `std::invalid_argument` if `knots` is empty, `degree` is outside \f$\{0, \dots, 10\}\f$, `period` or any knot is non-finite, \f$\omega \le 0\f$, `knots[0] + period` rounds to `knots[0]` or overflows, \f$K < d + 1\f$, `knots` is decreasing, `knots.back() >= knots[0] + period`, or a knot repeats more than \f$d + 1\f$ times across the wrap.
   * \see basis_cyclic_bspline, uniform(), knots()
   */
  // the K knots of one period, non-decreasing, knots[K - 1] < knots[0] +
  // period (a value repeated at most d + 1 times, counted across the wrap);
  // lower = knots[0]
  static cyclic_bspline_basis_owner from_knots(const std::vector<double>& knots, double period, int degree) {
    const int K = static_cast<int>(knots.size());
    if (K == 0) throw std::invalid_argument("cyclic_bspline: need at least d + 1 knots");
    check(knots[0], period, K, degree);
    for (double v : knots)
      if (!std::isfinite(v)) throw std::invalid_argument("cyclic_bspline: knots must be finite");
    for (int j = 0; j + 1 < K; ++j)
      if (knots[j + 1] < knots[j]) throw std::invalid_argument("cyclic_bspline: knots must be non-decreasing");
    if (!(knots[K - 1] < knots[0] + period))
      throw std::invalid_argument("cyclic_bspline: the knots must lie in [knots[0], knots[0] + period)");
    std::vector<double> t(K + 2 * degree + 1);
    for (int i = 0; i < K + 2 * degree + 1; ++i) {
      const int e = i - degree;  // t_e
      const int m = e >= 0 ? e / K : -((-e + K - 1) / K);
      t[i] = m == 0 ? knots[e] : knots[e - m * K] + m * period;
    }
    return cyclic_bspline_basis_owner(bspline_basis_owner::from_knots(t, degree, outside_domain_rule::polynomial),
                                      period, K, degree);
  }

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Returns an evaluation view pointing to host storage.
   * \return cyclic_bspline_basis view pointing into this owner's host storage (deleted on rvalue temporaries).
   * \see basis_cyclic_bspline, device()
   */
  // the struct points into this owner's storage: not available on a
  // temporary owner
  cyclic_bspline_basis host() const& { return make(ext_.host(), piece_totals_.data()); }
  /**
   * \ingroup basis_cyclic_bspline
   * \brief Returns an evaluation view pointing to device storage for Kokkos kernels.
   * \return cyclic_bspline_basis view pointing into this owner's device storage (deleted on rvalue temporaries).
   * \see basis_cyclic_bspline, host()
   */
  cyclic_bspline_basis device() const& { return make(ext_.device(), device_piece_totals_.data()); }
  cyclic_bspline_basis host() const&& = delete;
  cyclic_bspline_basis device() const&& = delete;

  /**
   * \ingroup basis_cyclic_bspline
   * \brief Returns the \f$K\f$ knots \f$t_0, \dots, t_{K-1}\f$ of one period.
   * \return Vector of \f$K\f$ knot values on the host.
   * \see basis_cyclic_bspline, from_knots()
   */
  // the K knots of one period, t_0..t_{K-1}
  std::vector<double> knots() const {
    const auto& t = ext_.knots();
    return std::vector<double>(t.begin() + degree_, t.begin() + degree_ + dim_);
  }

 private:
  cyclic_bspline_basis_owner(bspline_basis_owner ext, double period, int dim, int degree)
      : ext_(std::move(ext)), period_(period), dim_(dim), degree_(degree) {
    const auto e = ext_.host();
    piece_totals_.resize(e.dim);
    for (int j = 0; j < e.dim; ++j) piece_totals_[j] = e.integral_column_total(j);
    device_piece_totals_ = Kokkos::View<double*, bspline_basis_owner::device_memory>("laplace::basis::cyclic_piece_totals", e.dim);
    auto m = Kokkos::create_mirror_view(device_piece_totals_);
    for (int j = 0; j < e.dim; ++j) m(j) = piece_totals_[j];
    Kokkos::deep_copy(device_piece_totals_, m);
  }

  static void check(double lower, double period, int dim, int degree) {
    if (degree < 0 || degree > bspline_max_degree)
      throw std::invalid_argument("cyclic_bspline: degree must be in 0.." + std::to_string(bspline_max_degree));
    if (!std::isfinite(lower) || !std::isfinite(period) || !(period > 0.0) || !std::isfinite(lower + period))
      throw std::invalid_argument("cyclic_bspline: need finite lower and a finite period > 0");
    if (!(lower + period > lower)) throw std::invalid_argument("cyclic_bspline: the period is below the resolution of lower");
    if (dim < degree + 1) throw std::invalid_argument("cyclic_bspline: need dim >= degree + 1");
  }

  cyclic_bspline_basis make(bspline_basis e, const double* totals) const {
    cyclic_bspline_basis b;
    b.ext = e;
    b.piece_totals = totals;
    b.degree = degree_;
    b.dim = dim_;
    b.lower = e.lower;
    b.upper = e.upper;
    b.period = period_;
    return b;
  }

  bspline_basis_owner ext_;
  double period_;
  int dim_, degree_;
  std::vector<double> piece_totals_;
  Kokkos::View<double*, bspline_basis_owner::device_memory> device_piece_totals_;
};

}  // namespace laplace::basis
