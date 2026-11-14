// The pieces shared by the univariate bases (laplace::basis); the design was
// decided with Giovanni on 2026-11-13/14 (diario.md, decisions D1-D10).
//
// A univariate basis is a vector of K functions b(x) = (b_0(x), ..., b_{K-1}(x))
// on a domain [lower, upper]. A basis is complete (no column is dropped for
// identifiability: that is a constraint of the smoother or the term) and knows
// nothing about data.
//
// A basis is a small struct copied by value into kernels. In a kernel it gives
// a row of the design at one point x: the first column f and the values of
// the columns f, ..., f + w - 1, where the width w = row_width() is fixed for
// the basis (d + 1 for a B-spline of degree d, K for the dense bases), the
// same for every derivative order and outside the domain:
//
//   row(x, out), row_d1(x, out) .. row_d6(x, out)
//                      the values of b^(k)(x), one function per order k;
//                      return f
//   expansion(x, beta), expansion_d1 .. expansion_d6
//                      sum_j beta_j b_j^(k)(x)
//   integral_row(x, out), integral_expansion(x, beta)
//                      the integral from lower to x (a basis may give a
//                      window and a convention for the other columns; see
//                      bspline_basis)
//   row_width(), scratch_size(), dimension()
//
// out holds row_width() values; beta holds dimension() values. Outside
// [lower, upper] the basis follows its outside_domain_rule.
#pragma once

#include <laplace/core/config.hpp>

#include <Kokkos_Core.hpp>

#include <concepts>
#include <type_traits>

namespace laplace::basis {

/**
 * \ingroup basis
 * \brief Evaluation rules for points outside the basis domain \f$[lower, upper]\f$.
 *
 * Defines the behavior of basis functions, their derivatives and integrals when
 * evaluated at \f$x \notin [lower, upper]\f$:
 * - `nan`: returns NaN beyond a relative tolerance of \f$10^{-8}\f$ of the domain
 *   width \f$upper - lower\f$; inside the tolerance, evaluates the polynomial
 *   piece at the nearer endpoint.
 * - `linear`: evaluates the linear extrapolation from the nearer endpoint \f$e\f$
 *   (where \f$e = lower\f$ for \f$x < lower\f$, and \f$e = upper\f$ for \f$x > upper\f$):
 *   \f[
 *   f(x) = f(e) + f'(e)(x - e)
 *   \f]
 *   The first derivative is \f$f'(x) = f'(e)\f$, higher derivatives are zero
 *   \f$f^{(k)}(x) = 0\f$ (\f$k \ge 2\f$), and the integral from \f$lower\f$ is:
 *   \f[
 *   I(x) = I(e) + f(e)(x - e) + \frac{1}{2} f'(e)(x - e)^2
 *   \f]
 *   For a B-spline of degree \f$d \ge 2\f$, the linear rule gives \f$C^1\f$ regularity
 *   at the ends, where the second derivative jumps to zero. This is the default.
 * - `polynomial`: continues the polynomial piece at the nearer endpoint.
 *
 * The Fourier basis is periodic on \f$\mathbb{R}\f$ and has no outside domain rule.
 *
 * \see univariate_basis, basis_row_of_order()
 */
// What a basis gives outside [lower, upper] (decision D5).
//   nan         NaN beyond a tolerance of 1e-8 of the width; inside the
//               tolerance, the polynomial piece at the end
//   linear      value and first derivative at the end: f(e) + f'(e) (x - e)
//               (default; continuous with its first derivative at the end)
//   polynomial  the polynomial piece at the end, continued
enum class outside_domain_rule : int { nan = 0, linear = 1, polynomial = 2 };

/**
 * \ingroup basis
 * \brief Concept defining the common interface of a univariate basis.
 *
 * A univariate basis represents \f$K\f$ functions \f$b(x) = (b_0(x), \dots, b_{K-1}(x))\f$
 * on a domain \f$[lower, upper]\f$. The basis is complete (no column is dropped)
 * and knows nothing about data.
 *
 * A type \c B models \c univariate_basis if it provides:
 * - `dimension()`: returns the total number of basis functions \f$K\f$.
 * - `row_width()`: returns the fixed width \f$w\f$ of an evaluated row.
 * - `scratch_size()`: returns the number of doubles of scratch buffer required by member functions.
 * - `row(x, out)`: evaluates basis functions \f$b_j(x)\f$ at order 0 into \c out, returning first column index \f$f\f$.
 * - `row_d1(x, out)` through `row_d6(x, out)`: evaluates derivative orders 1 to 6 into \c out, returning first column index \f$f\f$.
 * - `expansion(x, beta)`: evaluates the linear combination \f$\sum_{j=0}^{K-1} \beta_j b_j(x)\f$.
 * - `expansion_d1(x, beta)` through `expansion_d6(x, beta)`: evaluates derivative orders 1 to 6 of the linear combination:
 *   \f[
 *   \eta^{(k)}(x) = \sum_{j=0}^{K-1} \beta_j b_j^{(k)}(x)
 *   \f]
 * - `integral_row(x, out)`: evaluates the integral \f$\int_{lower}^x b_j(t) \, \mathrm{d}t\f$ into \c out, returning first column index \f$f\f$.
 * - `integral_expansion(x, beta)`: evaluates the integral of the linear combination from \f$lower\f$ to \f$x\f$:
 *   \f[
 *   \sum_{j=0}^{K-1} \beta_j \int_{lower}^x b_j(t) \, \mathrm{d}t
 *   \f]
 *
 * \tparam B Basis type modeling the concept.
 *
 * \see outside_domain_rule, basis_row_of_order(), fill_basis_rows()
 */
template <class B>
concept univariate_basis = requires(const B& b, double x, double* out, const double* beta) {
  { b.dimension() } -> std::convertible_to<int>;
  { b.row_width() } -> std::convertible_to<int>;
  { b.scratch_size() } -> std::convertible_to<int>;
  { b.row(x, out) } -> std::convertible_to<int>;
  { b.row_d1(x, out) } -> std::convertible_to<int>;
  { b.row_d2(x, out) } -> std::convertible_to<int>;
  { b.row_d3(x, out) } -> std::convertible_to<int>;
  { b.row_d4(x, out) } -> std::convertible_to<int>;
  { b.row_d5(x, out) } -> std::convertible_to<int>;
  { b.row_d6(x, out) } -> std::convertible_to<int>;
  { b.expansion(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d1(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d2(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d3(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d4(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d5(x, beta) } -> std::convertible_to<double>;
  { b.expansion_d6(x, beta) } -> std::convertible_to<double>;
  { b.integral_row(x, out) } -> std::convertible_to<int>;
  { b.integral_expansion(x, beta) } -> std::convertible_to<double>;
};

/**
 * \ingroup basis
 * \brief Evaluates a row of a univariate basis of runtime derivative order \f$k \in \{0, \dots, 6\}\f$.
 *
 * Dispatches row evaluation for order \f$k\f$ at point \f$x\f$ through the basis's
 * per-order member functions row(), row_d1(), ..., row_d6(). Writes the \f$w\f$ values
 * of \f$b_j^{(k)}(x)\f$ for columns \f$j \in \{f, \dots, f + w - 1\}\f$, where \f$w\f$
 * is returned by row_width(), to the caller buffer \c out.
 *
 * \tparam B Basis type modeling \ref univariate_basis.
 * \tparam T Scalar floating-point type.
 * \param b Univariate basis instance.
 * \param k Derivative order \f$k \in \{0, \dots, 6\}\f$ (orders \f$k \ge 6\f$ dispatch to row_d6()).
 * \param x Evaluation point.
 * \param out Output buffer of at least `b.row_width()` elements.
 * \return 0-based column index \f$f\f$ of the first value written to \c out.
 *
 * If \f$x\f$ is NaN, writes NaN to all entries of \c out. Outside \f$[lower, upper]\f$,
 * evaluation follows the basis's outside domain rule. Dispatching introduces no
 * additional floating-point operations; measured accuracy is that of the underlying
 * basis's member functions.
 *
 * \see fill_basis_rows(), univariate_basis, outside_domain_rule
 */
// The row of order k (0..6) at x, by the per-order functions; returns the
// first column. A dispatch for callers whose order is a runtime value (the
// Gram matrices, the helper below, the tests).
template <univariate_basis B, class T>
LAPLACE_HD int basis_row_of_order(const B& b, int k, T x, T* out) {
  switch (k) {
    case 0: return b.row(x, out);
    case 1: return b.row_d1(x, out);
    case 2: return b.row_d2(x, out);
    case 3: return b.row_d3(x, out);
    case 4: return b.row_d4(x, out);
    case 5: return b.row_d5(x, out);
    default: return b.row_d6(x, out);
  }
}

/**
 * \ingroup basis
 * \brief Evaluates basis rows of order \f$k\f$ for every point in a View inside a parallel kernel.
 *
 * Launches a `Kokkos::parallel_for` kernel on the execution space of \c XView over all
 * indices \f$i \in \{0, \dots, N-1\}\f$, where \f$N\f$ is the extent of \c x. For each
 * point \f$x(i)\f$, calls basis_row_of_order() to evaluate the basis functions or
 * derivatives of order \f$k\f$, writing the first column index to `first(i)` and the
 * \f$w\f$ values directly in place into `&values(i, 0)`.
 *
 * The \c ValuesView must have layout `Kokkos::LayoutRight` so that each row is stored
 * contiguously in memory. The basis \c b must hold pointers accessible in the execution
 * space's memory space (e.g. `bspline_basis_owner::device()`).
 *
 * \tparam B Basis type modeling \ref univariate_basis.
 * \tparam XView 1D Kokkos View of evaluation points \f$x(i)\f$.
 * \tparam FirstView 1D Kokkos View of integers for first column indices.
 * \tparam ValuesView 2D Kokkos View with `Kokkos::LayoutRight` for row values.
 * \param b Univariate basis instance with pointers accessible in the execution space.
 * \param k Derivative order \f$k \in \{0, \dots, 6\}\f$.
 * \param x Input 1D View of evaluation points \f$x(i)\f$.
 * \param first Output 1D View of size \f$N\f$ for first column indices.
 * \param values Output 2D View of size \f$N \times w\f$ with `Kokkos::LayoutRight` for row values.
 *
 * Requires \c values to have layout `Kokkos::LayoutRight` (enforced at compile time).
 * Special values and domain handling at each point \f$x(i)\f$ follow basis_row_of_order().
 * Measured accuracy is identical bit for bit to calling basis_row_of_order() at each point.
 *
 * \see basis_row_of_order(), univariate_basis
 */
// The rows of order k at every point of x, in a kernel on the execution space
// of the Views: first(i) and values(i, 0..row_width() - 1), values a
// LayoutRight View of doubles (each row written in place). b must hold
// pointers into that memory space (bspline_basis_owner::device()).
template <univariate_basis B, class XView, class FirstView, class ValuesView>
void fill_basis_rows(const B& b, int k, const XView& x, const FirstView& first, const ValuesView& values) {
  static_assert(std::is_same_v<typename ValuesView::array_layout, Kokkos::LayoutRight>,
                "fill_basis_rows writes each row in place: values must be LayoutRight");
  using exec = typename XView::execution_space;
  Kokkos::parallel_for(
      "laplace::basis::fill_basis_rows", Kokkos::RangePolicy<exec>(0, x.extent(0)),
      KOKKOS_LAMBDA(int i) { first(i) = basis_row_of_order(b, k, x(i), &values(i, 0)); });
}

}  // namespace laplace::basis
