#pragma once

// What a run of an optimizer returns (note/optimizers.md, decisions 5 and
// 5b): why it stopped, where, and what it cost. Success is declared only on a
// stationarity measure; a stop on a change in the point or the value has a
// status of its own, and the decrement is reported in every case so that
// the caller can judge.

#include <Kokkos_Core.hpp>

#include <functional>
#include <limits>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Termination status returned in an optimizer_report.
 *
 * Only `converged` and `converged_at_resolution` are considered successes by optimizer_succeeded().
 *
 * \see \ref optimizers, optimizer_succeeded(), optimizer_report
 */
// Why the run stopped. Only the first two are successes.
enum class optimizer_status {
  converged,                // a stationarity rule held at its tolerance
  ///< A stationarity rule held at its tolerance.
  converged_at_resolution,  // the stationarity measure fell below the resolution of the
                            // objective (about eps |f|): no smaller one can be verified
  ///< The stationarity measure fell below the resolution of the objective (about \f$\varepsilon |f|\f$): no smaller one can be verified.
  stopped_on_change,        // a rule on the change of the point or of the value held
  ///< A rule on the change of the point or of the value held.
  stalled,                  // the step no longer moves the point (below rounding)
  ///< The step no longer moves the point (below rounding).
  iteration_budget,         // the iteration budget ran out
  ///< The iteration budget ran out.
  evaluation_budget,        // the budget of objective evaluations ran out
  ///< The budget of objective evaluations ran out.
  line_search_failed,       // no acceptable step along the direction
  ///< No acceptable step along the direction.
  start_not_admissible      // the objective is not finite at the start
  ///< The objective is not finite at the start.
};

/**
 * \ingroup optimizers
 * \brief Returns whether an optimizer_status represents convergence.
 *
 * \param status Termination status from an optimizer_report.
 * \return `true` for `optimizer_status::converged` and `optimizer_status::converged_at_resolution`, `false` otherwise.
 *
 * \see \ref optimizers, optimizer_status
 */
// true for converged and converged_at_resolution.
constexpr bool optimizer_succeeded(optimizer_status status) {
  return status == optimizer_status::converged || status == optimizer_status::converged_at_resolution;
}

/**
 * \ingroup optimizers
 * \brief Outcome of a curvature matrix factorization.
 *
 * \see \ref optimizers, dense_curvature_solver, optimizer_report
 */
// The outcome of the last factorization of the curvature, for the methods
// that factorize it (Newton, trust region); not_evaluated otherwise.
enum class curvature_factorization {
  not_evaluated,
  ///< No curvature factorization was performed.
  positive_definite,
  ///< The matrix was positive definite and factored successfully.
  not_positive_definite,
  ///< The matrix was not positive definite.
  non_finite
  ///< A non-finite entry appeared in the matrix.
};

/**
 * \ingroup optimizers
 * \brief Summary of an optimization run returned by minimize().
 *
 * \see \ref optimizers, optimizer_status, curvature_factorization, minimize()
 */
struct optimizer_report {
  /**
   * \ingroup optimizers
   * \brief Reason the run stopped.
   */
  optimizer_status status = optimizer_status::start_not_admissible;
  /**
   * \ingroup optimizers
   * \brief Last accepted point \f$x\f$.
   */
  Kokkos::View<double*, Kokkos::HostSpace> point;     // the last accepted point
  /**
   * \ingroup optimizers
   * \brief Objective value \f$f(x)\f$ at `point`.
   */
  double value = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Gradient at `point`; empty when not evaluated by the method.
   */
  Kokkos::View<double*, Kokkos::HostSpace> gradient;  // at point; empty for C0 methods
  /**
   * \ingroup optimizers
   * \brief Decrease predicted by the next step in the objective's units.
   *
   * Equals \f$\frac{1}{2} g^\top H^{-1} g\f$ with curvature, or \f$\frac{1}{2} g^\top B g\f$ without it where \f$B\f$ is the method's model of the inverse curvature; quiet NaN when neither is available.
   */
  // The stationarity measure in the objective's units, the decrease predicted
  // by the next step: (1/2) g' H^-1 g with curvature, (1/2) g' B g with B the
  // method's inverse-curvature model without (stop_rule_input); NaN when the
  // method has neither.
  double decrement = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Size of the region still searched by a derivative-free method at the end, in the units of \f$x\f$, or quiet NaN.
   */
  // Brent's final bracket half-width, a simplex's diameter, a poll step; NaN
  // for the methods with derivatives.
  double search_size = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Outcome of the final curvature factorization, or `curvature_factorization::not_evaluated`.
   */
  curvature_factorization final_factorization = curvature_factorization::not_evaluated;
  /**
   * \ingroup optimizers
   * \brief Number of iterations completed.
   */
  int iterations = 0;
  /**
   * \ingroup optimizers
   * \brief Total objective value evaluations (`value_gradient` counts one value and one gradient; `value_gradient_curvature` counts one of each).
   */
  // Evaluations of f, grad f and H: value_gradient counts one f and one g,
  // value_gradient_curvature one of each.
  long value_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Total gradient evaluations.
   */
  long gradient_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Total curvature matrix evaluations.
   */
  long curvature_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Number of coordinate updates whose diagonal curvature was damped by proximal Newton (penalized coordinates with \f$H_{jj} \le \rho_j\f$ or unpenalized coordinates with \f$H_{jj} \le 0\f$); \f$0\f$ for the other methods.
   */
  int damped_coordinates = 0;
  /**
   * \ingroup optimizers
   * \brief Whether `decrement` is proximal Newton's fallback measure.
   *
   * True when proximal Newton falls back to the proximal gradient mapping measure because \f$H\f$ is finite and its restriction to the free coordinates has no Cholesky factor; `false` by default and for every other method.
   */
  bool decrement_from_prox_mapping = false;  // proximal Newton's fallback measure (note/optimizers.md section 12, point 16)
  /**
   * \ingroup optimizers
   * \brief Wall-clock duration of the run in seconds.
   */
  double elapsed_seconds = 0.0;
};

/**
 * \ingroup optimizers
 * \brief Data passed to an optimizer_trace callback on each accepted iteration.
 *
 * The Views in this struct reference internal optimizer buffers and are valid during the callback invocation only.
 *
 * \see \ref optimizers, optimizer_trace, optimizer_report
 */
// One accepted iteration, passed to the optional trace callback of an
// optimizer. The Views are the optimizer's own: valid during the call only.
struct optimizer_trace_point {
  /**
   * \ingroup optimizers
   * \brief Iteration index.
   */
  int iteration;
  /**
   * \ingroup optimizers
   * \brief Accepted point \f$x\f$ (valid during the callback only).
   */
  Kokkos::View<const double*, Kokkos::HostSpace> point;
  /**
   * \ingroup optimizers
   * \brief Objective value \f$f(x)\f$ at `point`.
   */
  double value;
  /**
   * \ingroup optimizers
   * \brief Gradient at `point` (valid during the callback only; empty when not evaluated).
   */
  Kokkos::View<const double*, Kokkos::HostSpace> gradient;  // empty for C0 methods
  /**
   * \ingroup optimizers
   * \brief Predicted decrease at `point`, or quiet NaN when unavailable.
   */
  double decrement;
  /**
   * \ingroup optimizers
   * \brief Accepted step length along the search direction.
   */
  double step;  // the accepted step length along the direction
};

/**
 * \ingroup optimizers
 * \brief Optional callback invoked once per accepted iteration with an optimizer_trace_point.
 *
 * \see \ref optimizers, optimizer_trace_point
 */
// The trace: a callback that an optimizer calls once per accepted iteration
// when it is set (an empty std::function costs one test per iteration).
using optimizer_trace = std::function<void(const optimizer_trace_point&)>;

}  // namespace laplace::optimizers
