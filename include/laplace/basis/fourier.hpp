// Fourier basis (laplace::basis); design decided with Giovanni on 2026-11-14
// (diario.md), measurements in docs/numerics.md.
//
// A Fourier basis on [lower, upper] with period omega and H >= 0 harmonics has
// K = 2 H + 1 columns, in this order,
//
//   b_0 = 1,  b_{2j-1}(x) = sin(j nu x'),  b_{2j}(x) = cos(j nu x'),  j = 1..H,
//
// with x' = x - lower and nu = 2 pi / omega. The basis is complete (no column
// is dropped) and periodic: it is defined for every real x and has no outside
// domain rule. The default upper end is lower + omega (a full period); a
// shorter or a longer domain is allowed.
//
// Rows are dense: row_width() = K and the first column is 0. The derivative of
// order k follows the shift identity
//
//   d^k/dx^k sin(j nu x') = (j nu)^k sin(j nu x' + k pi/2),
//
// i.e. the pair (sin, cos) is turned by k quarter turns (k mod 4) and scaled by
// (j nu)^k; the constant column is 1 at order 0 and 0 above. Expansions are
// accumulated while the harmonics are generated, with no buffer of size K.
//
// Evaluation of (sin(j theta), cos(j theta)) for theta = nu x'. With
// u = x'/omega, the exact route for harmonic j reduces j u modulo 1 with an
// exact product (Kokkos::fma on the device, Dekker's split product on the
// host, whose gcc build has no hardware fma), r = j u - round(j u) (the
// rounding by the constant 1.5 2^52), and returns Kokkos::sin and cos of 2 pi r.
// The other harmonics come from the rotation
//   (c_{j+1}, s_{j+1}) = (c_j c_1 - s_j s_1, s_j c_1 + c_j s_1),
// re-anchored by the exact route every 8 harmonics (j = 1, 9, 17, ...), which
// keeps the error at a few times the conditioning 2 pi j |x'|/omega eps.
//
// The integral from lower: x' for the constant column, and for harmonic j
//   int_0^x' sin(j nu t) dt = (1 - cos(j nu x'))/(j nu) = 2 sin^2(j nu x'/2)/(j nu),
//   int_0^x' cos(j nu t) dt = sin(j nu x')/(j nu) = 2 sin(j nu x'/2) cos(j nu x'/2)/(j nu).
// One chain over the half angle gives both, and 1 - cos is never formed.
//
// Gram matrices (host): G_kl = int_lower^upper b^(k) (b^(l))' dx, in closed
// form. A column derivative is A sin(a t + m pi/2) (t = x - lower, a = j nu,
// A = a^k, m an integer), and
//   sin P sin Q = (cos(P - Q) - cos(P + Q))/2,
// whose integrals over [0, L] (L = upper - lower) are sin(n nu L)/(n nu),
// 2 sin^2(n nu L/2)/(n nu) and L for n = 0. On a full period (L = omega) the
// first two are exactly 0 and the matrix is block diagonal.
#pragma once

#include <laplace/basis/core.hpp>
#include <laplace/math/detail/double_double.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace laplace::basis {

namespace detail_fourier {

inline constexpr double two_pi = 6.283185307179586476925286766559;
inline constexpr double rounding_constant = 6755399441055744.0;  // 1.5 * 2^52

// the integer nearest to p (ties to even; exact for |p| < 2^51)
LAPLACE_HD_FORCEINLINE double nearest_integer(double p) {
  return laplace::math::detail::ddm::sub(laplace::math::detail::ddm::add(p, rounding_constant), rounding_constant);
}

// r = j u - round(j u) and rh = j u/2 - round(j u/2), both from the exact
// product j u = hi + lo
LAPLACE_HD_FORCEINLINE void reduce_turns(double j, double u, double& r, double& rh) {
  namespace d = laplace::math::detail::ddm;
  const d::dd p = d::two_prod(j, u);
  r = d::add(d::sub(p.hi, nearest_integer(p.hi)), p.lo);
  const double h = 0.5 * p.hi;
  rh = d::add(d::sub(h, nearest_integer(h)), 0.5 * p.lo);
}

// The values (sin(j phi), cos(j phi)) for j = 1, 2, ..., where phi = 2 pi u
// (half = false) or pi u (half = true).
template <bool half>
struct chain {
  double u = 0.0, s = 0.0, c = 1.0, s1 = 0.0, c1 = 1.0;

  LAPLACE_HD void start(double u_) { u = u_; }

  // advance to harmonic j (called with j = 1, 2, ...)
  LAPLACE_HD void next(int j) {
    if (((j - 1) & 7) == 0) {
      double r, rh;
      reduce_turns(static_cast<double>(j), u, r, rh);
      const double angle = two_pi * (half ? rh : r);
      s = Kokkos::sin(angle);
      c = Kokkos::cos(angle);
      if (j == 1) {
        s1 = s;
        c1 = c;
      }
    } else {
      const double ns = s * c1 + c * s1;
      const double nc = c * c1 - s * s1;
      s = ns;
      c = nc;
    }
  }
};

// the derivative of order K of the pair (sin, cos): (S, C) from (s, c), by K quarter turns
template <int K>
LAPLACE_HD_FORCEINLINE void quarter_turns(double s, double c, double& S, double& C) {
  if constexpr (K % 4 == 0) { S = s; C = c; }
  else if constexpr (K % 4 == 1) { S = c; C = -s; }
  else if constexpr (K % 4 == 2) { S = -s; C = -c; }
  else { S = -c; C = s; }
}

template <int K>
LAPLACE_HD_FORCEINLINE double power(double w) {
  double p = 1.0;
  for (int i = 0; i < K; ++i) p *= w;
  return p;
}

}  // namespace detail_fourier

/**
 * \ingroup basis_fourier
 * \brief Complete periodic Fourier trigonometric basis on a 1D interval.
 *
 * Implements the trigonometric basis of dimension \f$K = 2H + 1\f$ with columns
 * \f$b_0(x) = 1\f$, \f$b_{2j-1}(x) = \sin(j \nu x')\f$, \f$b_{2j}(x) = \cos(j \nu x')\f$
 * for \f$j = 1, \dots, H\f$, where \f$x' = x - \text{lower}\f$ and \f$\nu = 2\pi / \omega\f$.
 *
 * \see basis_fourier, make_fourier_basis()
 */
struct fourier_basis {
  /**
   * \ingroup basis_fourier
   * \var fourier_basis::lower
   * \brief Lower bound of the domain, \f$\text{lower}\f$.
   */
  /**
   * \ingroup basis_fourier
   * \var fourier_basis::upper
   * \brief Upper bound of the domain, \f$\text{upper}\f$.
   */
  /**
   * \ingroup basis_fourier
   * \var fourier_basis::period
   * \brief Fundamental period of the basis, \f$\omega > 0\f$.
   */
  double lower = 0.0, upper = 1.0, period = 1.0;
  /**
   * \ingroup basis_fourier
   * \brief Number of harmonics \f$H \ge 0\f$.
   */
  int harmonics = 0;
  /**
   * \ingroup basis_fourier
   * \brief Whether the domain is one full period (\f$\text{upper} - \text{lower} = \omega\f$, to rounding).
   */
  bool full_period = true;  // the domain is one full period (upper - lower = period, to rounding)

  /**
   * \ingroup basis_fourier
   * \brief Total number of basis functions \f$K = 2H + 1\f$.
   * \return Basis dimension \f$K\f$.
   * \par Domain and special values
   * Non-negative integer for \f$H \ge 0\f$.
   * \par Measured accuracy
   * Exact integer.
   * \see basis_fourier, row_width()
   */
  LAPLACE_HD int dimension() const { return 2 * harmonics + 1; }
  /**
   * \ingroup basis_fourier
   * \brief Number of columns in each evaluated row, \f$K = 2H + 1\f$.
   * \return Row width \f$K\f$ (dense row).
   * \par Domain and special values
   * Equals dimension().
   * \par Measured accuracy
   * Exact integer.
   * \see basis_fourier, dimension(), row()
   */
  LAPLACE_HD int row_width() const { return 2 * harmonics + 1; }
  /**
   * \ingroup basis_fourier
   * \brief Scratch buffer size needed for evaluation.
   * \return Zero (the Fourier basis requires no scratch memory).
   * \par Measured accuracy
   * Exact integer.
   * \see basis_fourier
   */
  LAPLACE_HD int scratch_size() const { return 0; }
  /**
   * \ingroup basis_fourier
   * \brief Semi-bandwidth of cross Gram matrices.
   *
   * On a full period, cross Gram matrices are block diagonal with \f$2 \times 2\f$ blocks
   * (semi-bandwidth \f$\min(1, 2H)\f$). Otherwise, they are dense (semi-bandwidth \f$2H\f$).
   * \return Semi-bandwidth as an integer.
   * \par Domain and special values
   * Returns 0 for \f$H = 0\f$, 1 for \f$H \ge 1\f$ on a full period, and \f$2H\f$ on non-full periods.
   * \par Measured accuracy
   * Exact integer.
   * \see basis_fourier, cross_gram()
   */
  // the Gram matrices are block diagonal (2 x 2 blocks) on a full period
  int gram_bandwidth() const { return full_period ? std::min(1, 2 * harmonics) : 2 * harmonics; }

  /**
   * \ingroup basis_fourier
   * \brief Evaluates the basis functions at point \f$x\f$ (derivative order 0).
   *
   * Writes \f$b_0(x) = 1\f$, \f$b_{2j-1}(x) = \sin(j \nu x')\f$, and \f$b_{2j}(x) = \cos(j \nu x')\f$
   * for \f$j = 1, \dots, H\f$ into \c out, with \f$x' = x - \text{lower}\f$ and \f$\nu = 2\pi / \omega\f$.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 1.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega)\f$.
   * \see basis_fourier, row_d1(), expansion()
   */
  template <class T> LAPLACE_HD int row(T x, T* out) const { return row_order<0>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the first derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(1)}(x) = 0\f$, \f$b_{2j-1}^{(1)}(x) = j \nu \cos(j \nu x')\f$, and
   * \f$b_{2j}^{(1)}(x) = -j \nu \sin(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 1\f$.
   * \see basis_fourier, row(), row_d2(), expansion_d1()
   */
  template <class T> LAPLACE_HD int row_d1(T x, T* out) const { return row_order<1>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the second derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(2)}(x) = 0\f$, \f$b_{2j-1}^{(2)}(x) = -(j \nu)^2 \sin(j \nu x')\f$, and
   * \f$b_{2j}^{(2)}(x) = -(j \nu)^2 \cos(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 2\f$.
   * \see basis_fourier, row_d1(), row_d3(), expansion_d2()
   */
  template <class T> LAPLACE_HD int row_d2(T x, T* out) const { return row_order<2>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the third derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(3)}(x) = 0\f$, \f$b_{2j-1}^{(3)}(x) = -(j \nu)^3 \cos(j \nu x')\f$, and
   * \f$b_{2j}^{(3)}(x) = (j \nu)^3 \sin(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 3\f$.
   * \see basis_fourier, row_d2(), row_d4(), expansion_d3()
   */
  template <class T> LAPLACE_HD int row_d3(T x, T* out) const { return row_order<3>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the fourth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(4)}(x) = 0\f$, \f$b_{2j-1}^{(4)}(x) = (j \nu)^4 \sin(j \nu x')\f$, and
   * \f$b_{2j}^{(4)}(x) = (j \nu)^4 \cos(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 4\f$.
   * \see basis_fourier, row_d3(), row_d5(), expansion_d4()
   */
  template <class T> LAPLACE_HD int row_d4(T x, T* out) const { return row_order<4>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the fifth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(5)}(x) = 0\f$, \f$b_{2j-1}^{(5)}(x) = (j \nu)^5 \cos(j \nu x')\f$, and
   * \f$b_{2j}^{(5)}(x) = -(j \nu)^5 \sin(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 5\f$.
   * \see basis_fourier, row_d4(), row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD int row_d5(T x, T* out) const { return row_order<5>(x, out); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the sixth derivatives of the basis functions at point \f$x\f$.
   *
   * Writes \f$b_0^{(6)}(x) = 0\f$, \f$b_{2j-1}^{(6)}(x) = -(j \nu)^6 \sin(j \nu x')\f$, and
   * \f$b_{2j}^{(6)}(x) = -(j \nu)^6 \cos(j \nu x')\f$ for \f$j = 1, \dots, H\f$ into \c out.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ is NaN, \c out[0] is 0.0 and subsequent entries are NaN.
   * \par Measured accuracy
   * Bounded by 37.0 ulp (\f$H \le 1\f$), 140.0 ulp (\f$H \le 4\f$), 660.0 ulp (\f$H \le 20\f$),
   * and 3300.0 ulp (\f$H \le 100\f$), within 2.5 times the angle conditioning \f$\max(1, 2\pi j |x'|/\omega) + 6\f$.
   * \see basis_fourier, row_d5(), expansion_d6()
   */
  template <class T> LAPLACE_HD int row_d6(T x, T* out) const { return row_order<6>(x, out); }

  /**
   * \ingroup basis_fourier
   * \brief Evaluates the basis expansion \f$\sum_{i=0}^{K-1} \beta_i b_i(x)\f$ at point \f$x\f$ (derivative order 0).
   *
   * Accumulates the linear combination of basis functions in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Expansion value at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row() to within rounding of the sum.
   * \see basis_fourier, row(), expansion_d1()
   */
  // sum_j beta_j b_j^(k)(x)
  template <class T> LAPLACE_HD T expansion(T x, const T* beta) const { return expansion_order<0>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the first derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(1)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return First derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d1() to within rounding of the sum.
   * \see basis_fourier, row_d1(), expansion(), expansion_d2()
   */
  template <class T> LAPLACE_HD T expansion_d1(T x, const T* beta) const { return expansion_order<1>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the second derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(2)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Second derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d2() to within rounding of the sum.
   * \see basis_fourier, row_d2(), expansion_d1(), expansion_d3()
   */
  template <class T> LAPLACE_HD T expansion_d2(T x, const T* beta) const { return expansion_order<2>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the third derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(3)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Third derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d3() to within rounding of the sum.
   * \see basis_fourier, row_d3(), expansion_d2(), expansion_d4()
   */
  template <class T> LAPLACE_HD T expansion_d3(T x, const T* beta) const { return expansion_order<3>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the fourth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(4)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Fourth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d4() to within rounding of the sum.
   * \see basis_fourier, row_d4(), expansion_d3(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d4(T x, const T* beta) const { return expansion_order<4>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the fifth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(5)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Fifth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d5() to within rounding of the sum.
   * \see basis_fourier, row_d5(), expansion_d4(), expansion_d6()
   */
  template <class T> LAPLACE_HD T expansion_d5(T x, const T* beta) const { return expansion_order<5>(x, beta); }
  /**
   * \ingroup basis_fourier
   * \brief Evaluates the sixth derivative of the basis expansion at point \f$x\f$.
   *
   * Accumulates \f$\sum_{i=0}^{K-1} \beta_i b_i^{(6)}(x)\f$ in a single pass without
   * allocating an intermediate row buffer.
   * \tparam T Floating-point scalar type.
   * \param x Evaluation point \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Sixth derivative of the expansion at \f$x\f$.
   * \par Domain and special values
   * Periodic on \f$\mathbb{R}\f$. If \f$x\f$ or any \f$\beta_i\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and row_d6() to within rounding of the sum.
   * \see basis_fourier, row_d6(), expansion_d5()
   */
  template <class T> LAPLACE_HD T expansion_d6(T x, const T* beta) const { return expansion_order<6>(x, beta); }

  /**
   * \ingroup basis_fourier
   * \brief Evaluates column integrals from the lower boundary to \f$x\f$.
   *
   * Evaluates \f$\int_{\text{lower}}^x b_0(t) \, \mathrm{d}t = x - \text{lower}\f$, and for \f$j = 1, \dots, H\f$ with \f$w = j \nu\f$:
   * \f[
   * \int_{\text{lower}}^x \sin(j \nu (t - \text{lower})) \, \mathrm{d}t = \frac{2 \sin^2(j \nu x' / 2)}{j \nu}
   * \f]
   * \f[
   * \int_{\text{lower}}^x \cos(j \nu (t - \text{lower})) \, \mathrm{d}t = \frac{2 \sin(j \nu x' / 2) \cos(j \nu x' / 2)}{j \nu}
   * \f]
   * evaluated in a single pass along the recurrence chain over the half angle without forming \f$1 - \cos\f$.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit \f$x \in \mathbb{R}\f$.
   * \param out Output buffer of size at least row_width().
   * \return Index of the first non-zero column (always 0, dense row).
   * \par Domain and special values
   * Defined for all \f$x \in \mathbb{R}\f$. Returns 0 at \f$x = \text{lower}\f$.
   * If \f$x\f$ is NaN, \c out[0] is NaN.
   * \par Measured accuracy
   * Maximum error 2.0 ulp on gcc and CUDA; test bound 3.0 ulp.
   * \see basis_fourier, integral_expansion(), row()
   */
  // int_lower^x b_j, dense (the first column is 0)
  template <class T>
  LAPLACE_HD int integral_row(T x, T* out) const {
    const double xp = static_cast<double>(x) - lower;
    out[0] = static_cast<T>(xp);
    if (harmonics == 0) return 0;
    const double nu = detail_fourier::two_pi / period;
    detail_fourier::chain<true> ch;
    ch.start(xp / period);
    for (int j = 1; j <= harmonics; ++j) {
      ch.next(j);
      const double w = static_cast<double>(j) * nu;
      out[2 * j - 1] = static_cast<T>((2.0 * ch.s * ch.s) / w);
      out[2 * j] = static_cast<T>((2.0 * ch.s * ch.c) / w);
    }
    return 0;
  }

  /**
   * \ingroup basis_fourier
   * \brief Evaluates the integrated basis expansion \f$\int_{\text{lower}}^x \left(\sum_{i=0}^{K-1} \beta_i b_i(t)\right) \mathrm{d}t\f$.
   *
   * Accumulates the integrated basis functions scaled by \c beta in a single pass over the half angle.
   * \tparam T Floating-point scalar type.
   * \param x Upper integration limit \f$x \in \mathbb{R}\f$.
   * \param beta Array of \f$K\f$ basis coefficients.
   * \return Integrated expansion value.
   * \par Domain and special values
   * Defined for all \f$x \in \mathbb{R}\f$. Returns 0 at \f$x = \text{lower}\f$. If \f$x\f$ is NaN, returns NaN.
   * \par Measured accuracy
   * Agrees with the scalar product of beta and integral_row() to within rounding of the sum.
   * \see basis_fourier, integral_row(), expansion()
   */
  template <class T>
  LAPLACE_HD T integral_expansion(T x, const T* beta) const {
    const double xp = static_cast<double>(x) - lower;
    double sum = static_cast<double>(beta[0]) * xp;
    if (harmonics > 0) {
      const double nu = detail_fourier::two_pi / period;
      detail_fourier::chain<true> ch;
      ch.start(xp / period);
      for (int j = 1; j <= harmonics; ++j) {
        ch.next(j);
        const double w = static_cast<double>(j) * nu;
        sum += static_cast<double>(beta[2 * j - 1]) * ((2.0 * ch.s * ch.s) / w) +
               static_cast<double>(beta[2 * j]) * ((2.0 * ch.s * ch.c) / w);
      }
    }
    return static_cast<T>(sum);
  }

  // ---- host function

  /**
   * \ingroup basis_fourier
   * \brief Evaluates the cross Gram matrix \f$G_{kl} = \int_{\text{lower}}^{\text{upper}} b^{(k)}(x) (b^{(l)}(x))^\top \mathrm{d}x\f$ in closed form.
   *
   * Evaluates the \f$K \times K\f$ cross Gram matrix in closed form using product-to-sum trigonometric
   * identities. The result is written into \c out in column-major order: \c out[i + j * K] holds
   * \f$\int_{\text{lower}}^{\text{upper}} b_i^{(k)}(x) b_j^{(l)}(x) \, \mathrm{d}x\f$.
   * On a full period, \f$G_{kl}\f$ is block diagonal with \f$2 \times 2\f$ blocks.
   * \param k Derivative order of the left factor (\f$0 \le k \le 6\f$).
   * \param l Derivative order of the right factor (\f$0 \le l \le 6\f$).
   * \param out Output array of size at least \f$K \times K\f$.
   * \par Domain and special values
   * \f$k, l \in \{0, 1, 2, 3, 4, 5, 6\}\f$. Throws \c std::invalid_argument if \f$k < 0\f$, \f$k > 6\f$,
   * \f$l < 0\f$, or \f$l > 6\f$.
   * \par Measured accuracy
   * Maximum error 18 ulp against mpmath references; test bound 27.0 ulp.
   * \see basis_fourier, gram_bandwidth()
   */
  // G_kl = int_lower^upper b^(k) (b^(l))' dx, dense K x K, column-major
  // (out[i + j K] = int b_i^(k) b_j^(l)), 0 <= k, l <= 6; see the top of the file.
  void cross_gram(int k, int l, double* out) const {
    if (k < 0 || k > 6 || l < 0 || l > 6) throw std::invalid_argument("fourier: derivative orders must be in 0..6");
    namespace df = detail_fourier;
    const int H = harmonics, K = 2 * H + 1;
    const double L = full_period ? period : upper - lower;
    const double nu = df::two_pi / period;
    // Ic[n] = int_0^L cos(n nu t) dt, Is[n] = int_0^L sin(n nu t) dt, n = 0..2H
    std::vector<double> Ic(2 * H + 1), Is(2 * H + 1);
    Ic[0] = L;
    Is[0] = 0.0;
    const double uL = full_period ? 1.0 : L / period;
    for (int n = 1; n <= 2 * H; ++n) {
      if (full_period) {
        Ic[n] = 0.0;
        Is[n] = 0.0;
        continue;
      }
      double r, rh;
      df::reduce_turns(static_cast<double>(n), uL, r, rh);
      const double a = static_cast<double>(n) * nu;
      const double sh = Kokkos::sin(df::two_pi * rh);
      Ic[n] = Kokkos::sin(df::two_pi * r) / a;
      Is[n] = (2.0 * sh * sh) / a;
    }
    // int_0^L cos(n nu t + m pi/2) dt for any integer n and m
    auto cosine_integral = [&](int n, int m) {
      if (n < 0) { n = -n; m = -m; }
      switch (((m % 4) + 4) % 4) {
        case 0: return Ic[n];
        case 1: return -Is[n];
        case 2: return -Ic[n];
        default: return Is[n];
      }
    };
    // column i of order q: A sin(j nu t + m pi/2)
    struct term { int j, m; double amplitude; };
    auto column = [&](int i, int q) {
      if (i == 0) return term{0, 1, q == 0 ? 1.0 : 0.0};  // 1 = sin(pi/2)
      const int j = (i + 1) / 2;
      double A = 1.0;
      const double w = static_cast<double>(j) * nu;
      for (int e = 0; e < q; ++e) A *= w;
      return term{j, (i % 2 == 1 ? q : q + 1) % 4, A};
    };
    for (int jj = 0; jj < K; ++jj) {
      const term tb = column(jj, l);
      for (int ii = 0; ii < K; ++ii) {
        const term ta = column(ii, k);
        double v = 0.0;
        if (ta.amplitude != 0.0 && tb.amplitude != 0.0)
          v = 0.5 * ta.amplitude * tb.amplitude *
              (cosine_integral(ta.j - tb.j, ta.m - tb.m) - cosine_integral(ta.j + tb.j, ta.m + tb.m));
        out[ii + static_cast<size_t>(jj) * K] = v;
      }
    }
  }

  // ---- internals

  template <int K, class T>
  LAPLACE_HD int row_order(T x, T* out) const {
    namespace df = detail_fourier;
    out[0] = static_cast<T>(K == 0 ? 1.0 : 0.0);
    if (harmonics == 0) return 0;
    const double xp = static_cast<double>(x) - lower;
    const double nu = df::two_pi / period;
    df::chain<false> ch;
    ch.start(xp / period);
    for (int j = 1; j <= harmonics; ++j) {
      ch.next(j);
      double S, C;
      df::quarter_turns<K>(ch.s, ch.c, S, C);
      if constexpr (K > 0) {
        const double sc = df::power<K>(static_cast<double>(j) * nu);
        S *= sc;
        C *= sc;
      }
      out[2 * j - 1] = static_cast<T>(S);
      out[2 * j] = static_cast<T>(C);
    }
    return 0;
  }

  template <int K, class T>
  LAPLACE_HD T expansion_order(T x, const T* beta) const {
    namespace df = detail_fourier;
    double sum = K == 0 ? static_cast<double>(beta[0]) : 0.0;
    if (harmonics > 0) {
      const double xp = static_cast<double>(x) - lower;
      const double nu = df::two_pi / period;
      df::chain<false> ch;
      ch.start(xp / period);
      for (int j = 1; j <= harmonics; ++j) {
        ch.next(j);
        double S, C;
        df::quarter_turns<K>(ch.s, ch.c, S, C);
        double term = static_cast<double>(beta[2 * j - 1]) * S + static_cast<double>(beta[2 * j]) * C;
        if constexpr (K > 0) term *= df::power<K>(static_cast<double>(j) * nu);
        sum += term;
      }
    }
    return static_cast<T>(sum);
  }
};

/**
 * \ingroup basis_fourier
 * \brief Constructs a Fourier basis on a specified interval \f$[\text{lower}, \text{upper}]\f$.
 *
 * Sets \c full_period to \c true if \f$\text{upper} - \text{lower} = \text{period}\f$ (or \f$\text{upper} = \text{lower} + \text{period}\f$),
 * and to \c false otherwise.
 * \param lower Lower bound of the domain.
 * \param period Fundamental period \f$\omega > 0\f$.
 * \param harmonics Number of harmonics \f$H \ge 0\f$.
 * \param upper Upper bound of the domain (\f$\text{upper} > \text{lower}\f$).
 * \return Configured \ref fourier_basis structure.
 * \par Domain and special values
 * Throws \c std::invalid_argument if \c lower, \c upper, or \c period is not finite, if \c period <= 0.0,
 * if \c harmonics < 0, or if \c lower >= upper.
 * \par Measured accuracy
 * Exact arithmetic setup.
 * \see basis_fourier, fourier_basis
 */
// Checks its arguments (finite; period > 0; harmonics >= 0; lower < upper) and
// throws std::invalid_argument.
inline fourier_basis make_fourier_basis(double lower, double period, int harmonics, double upper) {
  if (!std::isfinite(lower) || !std::isfinite(upper) || !std::isfinite(period))
    throw std::invalid_argument("fourier: lower, upper and period must be finite");
  if (!(period > 0.0)) throw std::invalid_argument("fourier: the period must be positive");
  if (harmonics < 0) throw std::invalid_argument("fourier: harmonics must be >= 0");
  if (!(lower < upper)) throw std::invalid_argument("fourier: need lower < upper");
  fourier_basis b;
  b.lower = lower;
  b.upper = upper;
  b.period = period;
  b.harmonics = harmonics;
  b.full_period = (upper - lower == period) || (upper == lower + period);
  return b;
}

/**
 * \ingroup basis_fourier
 * \brief Constructs a Fourier basis on one full period, \f$[\text{lower}, \text{lower} + \text{period}]\f$.
 *
 * Sets \c upper = \c lower + \c period and \c full_period = \c true.
 * \param lower Lower bound of the domain.
 * \param period Fundamental period \f$\omega > 0\f$.
 * \param harmonics Number of harmonics \f$H \ge 0\f$.
 * \return Configured \ref fourier_basis structure with \c full_period = \c true.
 * \par Domain and special values
 * Throws \c std::invalid_argument if \c lower or \c period is not finite, if \c period <= 0.0,
 * or if \c harmonics < 0.
 * \par Measured accuracy
 * Exact arithmetic setup.
 * \see basis_fourier, fourier_basis
 */
// the domain is one full period: upper = lower + period
inline fourier_basis make_fourier_basis(double lower, double period, int harmonics) {
  if (!std::isfinite(lower) || !std::isfinite(period))
    throw std::invalid_argument("fourier: lower and period must be finite");
  return make_fourier_basis(lower, period, harmonics, lower + period);
}

}  // namespace laplace::basis
