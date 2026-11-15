#pragma once

// What an optimizer minimizes (note/optimizers.md, decision 1): a ladder of
// capabilities declared by the objective and checked at compile time.
//
//   value_objective       f(x)                                     C0
//   gradient_objective    also f and grad f together               C1
//   curvature_objective   also f, grad f and a symmetric H(x)      C2
//   composite_objective   s(x) + sum_b r_b(x_b), s in C1, each r_b with its
//                         proximal operator (wave 2: declared, not used yet)
//
// Every member evaluates everything it returns in one pass (one Kokkos
// reduction when the objective sums over observations), and derivatives are
// computed only when asked: a line search needs the value alone at its trial
// points. x is read-only; g (n entries) and H (n x n) are written by the
// objective. A non-finite value means the objective cannot be evaluated at x:
// the optimizer treats x as a refused trial point (decision 3), never as an
// error. Whether H is the observed or the expected information is the
// choice of whoever builds the objective (Fisher scoring is Newton on the
// expected one). x lives in free R^n; constraints are removed by the
// objective with bijectors.

#include <Kokkos_Core.hpp>

#include <concepts>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Read-only host view of an optimization point \f$x \in \mathbb{R}^n\f$.
 *
 * \see \ref optimizers, optimizer_vector
 */
// The point x, read-only, and a writable vector (gradient, direction, trial
// point), both on the host.
using optimizer_point = Kokkos::View<const double*, Kokkos::HostSpace>;
/**
 * \ingroup optimizers
 * \brief Writable host view of an \f$n\f$-entry vector such as a gradient or search direction.
 *
 * \see \ref optimizers, optimizer_point
 */
using optimizer_vector = Kokkos::View<double*, Kokkos::HostSpace>;

/**
 * \ingroup optimizers
 * \brief Column-major \f$n \times n\f$ host view holding a symmetric curvature matrix \f$H(x)\f$.
 *
 * Matches the `Kokkos::LayoutLeft` convention of \ref dense.
 *
 * \see \ref optimizers, curvature_objective, dense_curvature_solver
 */
// The dense curvature matrix (column-major as laplace::dense); an objective
// declares its own type of H as curvature_type, and the curvature solver is
// chosen from it (dense_curvature_solver.hpp).
using dense_curvature_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

/**
 * \ingroup optimizers
 * \brief Concept for an objective providing function values \f$f(x)\f$.
 *
 * Requires `double value(x)` returning \f$f(x)\f$ for an unconstrained point \f$x \in \mathbb{R}^n\f$.
 * A non-finite return value marks \f$x\f$ as a refused trial point rather than an error.
 *
 * \tparam O Objective type.
 * \see \ref optimizers, gradient_objective, curvature_objective
 */
// C0: double value(x).
template <class O>
concept value_objective = requires(O& objective, const optimizer_point& x) {
  { objective.value(x) } -> std::convertible_to<double>;
};

/**
 * \ingroup optimizers
 * \brief Concept for an objective providing values and gradients.
 *
 * Refines value_objective with `double value_gradient(x, g)`, which returns \f$f(x)\f$ and writes the gradient (or subgradient) \f$\nabla f(x)\f$ into `g` in one evaluation.
 *
 * \tparam O Objective type.
 * \see \ref optimizers, value_objective, curvature_objective, composite_objective
 */
// C1: double value_gradient(x, g), returning f(x) and writing grad f(x) (a
// subgradient where f is not differentiable) to g.
template <class O>
concept gradient_objective =
    value_objective<O> && requires(O& objective, const optimizer_point& x, const optimizer_vector& g) {
      { objective.value_gradient(x, g) } -> std::convertible_to<double>;
    };

/**
 * \ingroup optimizers
 * \brief Concept for an objective providing values, gradients, and curvature matrices.
 *
 * Refines gradient_objective with member type `curvature_type` and `double value_gradient_curvature(x, g, H)`, which returns \f$f(x)\f$ and writes \f$g\f$ and the symmetric curvature matrix \f$H(x)\f$ in one evaluation.
 * Whether \f$H\f$ is the observed Hessian or an expected information matrix is the objective's choice.
 *
 * \tparam O Objective type.
 * \see \ref optimizers, gradient_objective, dense_curvature_matrix, curvature_solver_for
 */
// C2: double value_gradient_curvature(x, g, H), also writing the symmetric
// H(x) to H; the type of H is O::curvature_type.
template <class O>
concept curvature_objective =
    gradient_objective<O> && requires { typename O::curvature_type; } &&
    requires(O& objective, const optimizer_point& x, const optimizer_vector& g, typename O::curvature_type& H) {
      { objective.value_gradient_curvature(x, g, H) } -> std::convertible_to<double>;
    };

/**
 * \ingroup optimizers
 * \brief Concept for a separable composite objective \f$f = f_{\mathrm{s}} + r\f$.
 *
 * Refines gradient_objective where `value()` and `value_gradient()` evaluate the smooth part \f$f_{\mathrm{s}}\f$ alone and \f$r(x) = \sum_b r_b(x_b)\f$ is block-separable.
 * Requires `nonsmooth_value(x)` returning \f$r(x)\f$, `nonsmooth_prox(v, t, z)` writing \f$z = \operatorname{prox}_{t r}(v) = \arg\min_z \{r(z) + \|z - v\|^2 / (2t)\}\f$, `nonsmooth_block_count()`, and `weak_convexity_constant(b)` returning \f$\rho_b \ge 0\f$ such that \f$r_b + \rho_b \|x_b\|^2 / 2\f$ is convex; the proximal map is exact only when \f$t \rho_b < 1\f$.
 * Only the concept is provided here.
 *
 * \tparam O Objective type.
 * \see \ref optimizers, gradient_objective
 */
// Composite: f = s + r with s smooth (value, value_gradient as above, of s
// alone) and r(x) = sum_b r_b(x_b), separable by blocks:
//   nonsmooth_value(x)               r(x)
//   nonsmooth_prox(v, t, z)          z = prox_{t r}(v) = argmin_z r(z) + |z - v|^2 / (2 t),
//                                    block by block
//   nonsmooth_block_count()          the number of blocks b
//   weak_convexity_constant(b)       rho_b >= 0 such that r_b + rho_b |x_b|^2 / 2 is
//                                    convex (0 when r_b is convex); the prox is
//                                    exact only when t rho_b < 1
template <class O>
concept composite_objective =
    gradient_objective<O> &&
    requires(O& objective, const optimizer_point& x, const optimizer_vector& z, double t, int b) {
      { objective.nonsmooth_value(x) } -> std::convertible_to<double>;
      objective.nonsmooth_prox(x, t, z);
      { objective.nonsmooth_block_count() } -> std::convertible_to<int>;
      { objective.weak_convexity_constant(b) } -> std::convertible_to<double>;
    };

}  // namespace laplace::optimizers
