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
 * Requires `optimizer.run(objective, x0)` to return an `optimizer_report`.
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
 * Calls `optimizer.run(objective, x0)` and returns its `optimizer_report`.
 * A maximum of \f$f\f$ is found by minimizing \f$-f\f$.
 *
 * \tparam Optimizer Optimizer type satisfying `optimizer_for<Optimizer, Objective>`.
 * \tparam Objective Objective type accepted by `Optimizer`.
 * \param optimizer Optimizer instance holding the run settings.
 * \param objective Objective to minimize.
 * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
 * \return An optimizer_report describing where and why the run stopped.
 *
 * \see \ref optimizers, optimizer_for, optimizer_report
 */
template <class Optimizer, class Objective>
  requires optimizer_for<Optimizer, Objective>
optimizer_report minimize(const Optimizer& optimizer, Objective& objective, const optimizer_point& x0) {
  return optimizer.run(objective, x0);
}

}  // namespace laplace::optimizers
