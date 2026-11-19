#pragma once

// minimize(optimizer, objective, x0) (note/optimizers.md, decision 9): the one
// call of every method. An optimizer is an object carrying its settings
// (budgets, stopping rule, line search) with a member
//   optimizer_report run(Objective& objective, optimizer_point x0) const;
// constrained to the capability it needs (objective.hpp); an objective that
// lacks it, or a stopping rule that needs a quantity the method does not
// provide, is a compile error. A maximum is found by minimizing -f.

#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"

#include <concepts>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Concept checking that `Optimizer` can minimize `Objective` via `run(objective, x0)`.
 *
 * An optimizer carries its settings (budgets, stopping rule, line search, trace)
 * as public members and provides a `const` member
 * `optimizer_report run(Objective&, const optimizer_point&) const` constrained
 * to the objective concept it needs. This concept holds when that call is
 * valid and returns `optimizer_report` (exactly that type).
 *
 * \tparam Optimizer Optimizer type.
 * \tparam Objective Objective type.
 * \see \ref optimizers, minimize(), optimizer_report
 */
template <class Optimizer, class Objective>
concept optimizer_for = requires(const Optimizer& optimizer, Objective& objective, const optimizer_point& x0) {
  { optimizer.run(objective, x0) } -> std::same_as<optimizer_report>;
};

/**
 * \ingroup optimizers
 * \brief Runs an optimizer on an objective starting from `x0`.
 *
 * Forwards directly to `optimizer.run(objective, x0)` and returns its
 * `optimizer_report`; it adds nothing else. The optimizer is passed by `const`
 * reference and the objective by non-`const` reference so it may keep state.
 * The starting point `x0` is not modified; the final point is returned in
 * `optimizer_report::point`. A maximum of \f$f\f$ is found by minimizing
 * \f$-f\f$.
 *
 * Calling `minimize()` with an objective that lacks the capability required by
 * `optimizer`, or with a stopping rule that asks for a quantity `optimizer`
 * does not provide, is a compile-time error. The table matching each
 * optimizer to its objective concept is on the page
 * \ref optimizers "Optimizers".
 *
 * ### Example
 *
 * Minimizes the two-variable `quadratic_curvature_objective` from
 * `curvature_objective` on the host after `Kokkos::initialize`:
 *
 * \code
 * #include <laplace/optimizers/newton.hpp>
 *
 * laplace::optimizers::optimizer_vector x0("x0", 2);
 * x0(0) = 3.0;
 * x0(1) = -1.0;
 * quadratic_curvature_objective objective;   // the objective of the curvature_objective example
 * const laplace::optimizers::newton_optimizer<> opt;
 * const laplace::optimizers::optimizer_report report = laplace::optimizers::minimize(opt, objective, x0);
 * \endcode
 *
 * \tparam Optimizer Optimizer type satisfying `optimizer_for<Optimizer, Objective>`.
 * \tparam Objective Objective type accepted by `Optimizer`.
 * \param optimizer Optimizer instance holding the run settings.
 * \param objective Objective to minimize.
 * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
 * \return An `optimizer_report` describing where and why the run stopped.
 *
 * \see \ref optimizers, optimizer_for, optimizer_report, curvature_objective
 */
template <class Optimizer, class Objective>
  requires optimizer_for<Optimizer, Objective>
optimizer_report minimize(const Optimizer& optimizer, Objective& objective, const optimizer_point& x0) {
  return optimizer.run(objective, x0);
}

}  // namespace laplace::optimizers
