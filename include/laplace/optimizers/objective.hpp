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

#include <algorithm>
#include <cmath>

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

namespace detail {

// max_i |v_i|, shared by the optimizers. NaN entries are skipped by max;
// callers test the finiteness of the whole vector.
inline double max_abs(const optimizer_point& v) {
  double m = 0.0;
  for (std::size_t i = 0; i < v.extent(0); ++i) m = std::max(m, std::abs(v(i)));
  return m;
}

// The inverse curvature that an accepted secant pair (s, y) contributes to the
// floor gamma_max of the first-order stationarity measure
// (1/2) max(g'Bg, gamma_max g'g): s's / s'y, the inverse Rayleigh quotient of
// the average Hessian along s (at most 1 / lambda_min on a quadratic). Shared by
// gradient descent, Barzilai-Borwein and conjugate gradient. BFGS's s'y / y'y,
// which first-order steps keep near 1 / lambda_max, let these methods declare
// convergence with f - f* up to 4800 times the measure (docs/numerics.md,
// wave 3).
inline double secant_floor(double sy, double ss, double /*yy*/) { return ss / sy; }

}  // namespace detail

/**
 * \ingroup optimizers
 * \brief Concept for an objective providing function values \f$f(x)\f$.
 *
 * Requires `double value(x)` returning \f$f(x)\f$ for an unconstrained point
 * \f$x \in \mathbb{R}^n\f$ passed as a read-only optimizer_point. There is no
 * base class; any type exposing the required signature models the concept.
 *
 * A non-finite return value means the objective cannot be evaluated at \f$x\f$:
 * optimizers treat \f$x\f$ as a refused trial point rather than an error, and a
 * non-finite value at the starting point ends the run with
 * `optimizer_status::start_not_admissible`. Required by brent_optimizer,
 * block_coordinate_optimizer, nelder_mead_optimizer, and compass_optimizer.
 *
 * \tparam O Objective type.
 * \see \ref optimizers, gradient_objective, curvature_objective, composite_objective
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
 * Refines value_objective with `double value_gradient(x, g)`, returning
 * \f$f(x)\f$ and writing the gradient \f$\nabla f(x)\f$ (or a subgradient
 * where \f$f\f$ is not differentiable) into the \f$n\f$-entry optimizer_vector
 * `g` in one pass. Derivatives are computed only when asked: a line search
 * calls `value()` alone at its trial points. A non-finite value or gradient at
 * the starting point gives `optimizer_status::start_not_admissible`. Required
 * by bfgs_optimizer, lbfgs_optimizer, and bundle_optimizer.
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
 * Refines gradient_objective with a member type `curvature_type` and
 * `double value_gradient_curvature(x, g, H)`, returning \f$f(x)\f$ and writing
 * the \f$n\f$-entry gradient `g` and the symmetric \f$n \times n\f$ curvature
 * matrix `H` in one pass. Whether \f$H(x)\f$ is the observed Hessian or an
 * expected information matrix is the objective's choice.
 *
 * Required by newton_optimizer, and with `curvature_type` equal to
 * dense_curvature_matrix by trust_region_optimizer and
 * proximal_newton_optimizer.
 *
 * ### Example
 *
 * Two-variable quadratic \f$f(x) = \frac{1}{2}(x_1^2 + 10 x_2^2)\f$ with
 * gradient \f$g = (x_1, 10 x_2)\f$ and curvature
 * \f$H = \mathrm{diag}(1, 10)\f$:
 *
 * \code
 * #include <laplace/optimizers/objective.hpp>
 *
 * struct quadratic_curvature_objective {
 *   using curvature_type = laplace::optimizers::dense_curvature_matrix;
 *
 *   double value(const laplace::optimizers::optimizer_point& x) const {
 *     return 0.5 * (x(0) * x(0) + 10.0 * x(1) * x(1));
 *   }
 *
 *   double value_gradient(const laplace::optimizers::optimizer_point& x,
 *                         const laplace::optimizers::optimizer_vector& g) const {
 *     g(0) = x(0);
 *     g(1) = 10.0 * x(1);
 *     return value(x);
 *   }
 *
 *   double value_gradient_curvature(const laplace::optimizers::optimizer_point& x,
 *                                   const laplace::optimizers::optimizer_vector& g,
 *                                   laplace::optimizers::dense_curvature_matrix& H) const {
 *     H(0, 0) = 1.0;
 *     H(1, 0) = 0.0;
 *     H(0, 1) = 0.0;
 *     H(1, 1) = 10.0;
 *     return value_gradient(x, g);
 *   }
 * };
 * \endcode
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
 * Refines gradient_objective where `value()` and `value_gradient()` evaluate
 * the smooth part \f$f_{\mathrm{s}}\f$ alone, and
 * \f$r(x) = \sum_b r_b(x_b)\f$ is separable over blocks \f$b\f$. Required by
 * proximal_gradient_optimizer and, together with curvature_objective, by
 * proximal_newton_optimizer; see \ref optimizers_proximal "Proximal methods"
 * for those methods and for penalized_objective.
 *
 * ### Required members
 *
 * | Member | Meaning |
 * |---|---|
 * | `double nonsmooth_value(x)` | Returns \f$r(x)\f$ |
 * | `void nonsmooth_prox(v, t, z)` | Writes, block by block, \f$z = \operatorname{prox}_{t r}(v) = \arg\min_z \{ r(z) + \lVert z - v \rVert^2 / (2t) \}\f$ for \f$t > 0\f$ |
 * | `int nonsmooth_block_count()` | Returns the number of blocks |
 * | `double weak_convexity_constant(b)` | Returns \f$\rho_b \ge 0\f$ such that \f$r_b + \rho_b \lVert x_b \rVert^2 / 2\f$ is convex (\f$0\f$ when \f$r_b\f$ is convex); exact prox requires \f$t \rho_b < 1\f$ |
 *
 * \tparam O Objective type.
 * \see \ref optimizers, \ref optimizers_proximal, gradient_objective
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
