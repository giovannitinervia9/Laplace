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
// same for every derivative order and outside the domain. Value i of a row
// belongs to column (f + i) mod K, K = dimension(): only a cyclic basis wraps
// (cyclic_bspline_basis); for the others f + i < K and the rule changes
// nothing (decision C2):
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
 * Selects how a non-periodic basis evaluates functions, derivatives and integrals
 * when \f$x \notin [lower, upper]\f$: `nan` returns NaN beyond a tolerance of
 * \f$10^{-8}\f$ of \f$upper - lower\f$, `linear` (the default) applies linear
 * extrapolation from the nearer endpoint, and `polynomial` continues the endpoint
 * polynomial piece. See \ref basis for exact formulas and Fourier periodicity.
 *
 * \see \ref basis, univariate_basis, basis_row_of_order()
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
 * \brief Concept defining the kernel evaluation interface of a univariate basis.
 *
 * Requires `dimension()`, `row_width()`, `scratch_size()`, row evaluations
 * `row()` and `row_d1()` through `row_d6()`, linear expansions `expansion()`
 * and `expansion_d1()` through `expansion_d6()`, and integrals `integral_row()`
 * and `integral_expansion()` from \f$lower\f$ to \f$x\f$. See \ref basis for
 * the row representation and conventions.
 *
 * \tparam B Basis type modeling the concept.
 *
 * \see \ref basis, outside_domain_rule, basis_row_of_order(), fill_basis_rows()
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
 * \brief Evaluates a row of a univariate basis at runtime derivative order \f$k \in \{0, \dots, 6\}\f$.
 *
 * Dispatches to `b.row(x, out)` for \f$k = 0\f$ and `b.row_d1(x, out)` through
 * `b.row_d6(x, out)` for \f$k = 1, \dots, 6\f$, writing \f$b_j^{(k)}(x)\f$ for
 * \f$j \in \{f, \dots, f + w - 1\}\f$, where \f$w\f$ is `b.row_width()`.
 *
 * \tparam B Basis type modeling \ref univariate_basis.
 * \tparam T Scalar floating-point type.
 * \param b Univariate basis instance.
 * \param k Derivative order \f$k \in \{0, \dots, 6\}\f$; values outside \f$\{0, \dots, 5\}\f$ dispatch to `b.row_d6()`.
 * \param x Evaluation point.
 * \param out Output buffer of at least `b.row_width()` elements.
 * \return 0-based column index \f$f\f$ of the first value written to \c out.
 *
 * Special values, outside-domain behavior and accuracy are those of the underlying
 * basis member functions, with no additional floating-point operations.
 *
 * \see \ref basis, fill_basis_rows(), univariate_basis, outside_domain_rule
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
 * \brief Evaluates basis rows of order \f$k\f$ across a 1D View in a parallel kernel.
 *
 * Runs a `Kokkos::parallel_for` over \f$i \in \{0, \dots, N-1\}\f$ on the execution
 * space of \c XView, calling basis_row_of_order() at each \f$x(i)\f$ to store the first
 * column index in `first(i)` and the \f$w\f$ values in place at `&values(i, 0)`.
 * Requires `Kokkos::LayoutRight` for \c ValuesView (enforced by `static_assert`) and a
 * basis \c b whose pointers are valid in the execution space's memory space.
 *
 * \tparam B Basis type modeling \ref univariate_basis.
 * \tparam XView 1D Kokkos View of evaluation points.
 * \tparam FirstView 1D Kokkos View of integers for first column indices.
 * \tparam ValuesView 2D Kokkos View with `Kokkos::LayoutRight`.
 * \param b Univariate basis instance accessible in the execution space.
 * \param k Derivative order \f$k \in \{0, \dots, 6\}\f$.
 * \param x Input 1D View of \f$N\f$ evaluation points.
 * \param first Output 1D View of size \f$N\f$ for first column indices.
 * \param values Output 2D `Kokkos::LayoutRight` View of size \f$N \times w\f$ for row values.
 *
 * Accuracy and domain behavior at each \f$x(i)\f$ match basis_row_of_order() bit for bit.
 *
 * \see \ref basis, basis_row_of_order(), univariate_basis
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
