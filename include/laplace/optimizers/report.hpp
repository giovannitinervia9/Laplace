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
 * \brief Termination status returned in an `optimizer_report`.
 *
 * Only `converged` and `converged_at_resolution` are successes according to
 * optimizer_succeeded(). In `converged_at_resolution`, the stationarity measure
 * fell below the objective's floating-point resolution, about
 * \f$\varepsilon |f|\f$ with \f$\varepsilon = 2^{-52}\f$, so that no smaller
 * decrease can be verified. `stopped_on_change` is returned when a change rule
 * on the point or objective value held and is never a success. `stalled` means
 * the step fell below floating-point rounding and no longer moves the point.
 * `start_not_admissible` means the objective (or gradient, for derivative
 * methods) is not finite at the start, and is also the default value before a
 * run populates `optimizer_report`. Which method returns which status, and
 * when, is documented on each method page (\ref optimizers_second_order,
 * \ref optimizers_quasi_newton, \ref optimizers_multistart,
 * \ref optimizers_block_coordinate, \ref optimizers_derivative_free,
 * \ref optimizers_proximal, \ref optimizers_bundle, \ref optimizers_primal_dual).
 *
 * \see \ref optimizers, optimizer_succeeded(), optimizer_report
 */
// Why the run stopped. Only the first two are successes.
enum class optimizer_status {
  converged,                // a stationarity rule held at its tolerance
  ///< A stationarity rule held at its tolerance (success).
  converged_at_resolution,  // the stationarity measure fell below the resolution of the
                            // objective (about eps |f|): no smaller one can be verified
  ///< The stationarity measure fell below the resolution of the objective (about \f$\varepsilon |f|\f$ with \f$\varepsilon = 2^{-52}\f$), so no smaller one can be verified (success).
  stopped_on_change,        // a rule on the change of the point or of the value held
  ///< A rule on the change of the point or of the value held (never a success).
  stalled,                  // the step no longer moves the point (below rounding)
  ///< The step no longer moves the point (below rounding).
  iteration_budget,         // the iteration budget ran out
  ///< The iteration budget ran out.
  evaluation_budget,        // the budget of objective evaluations ran out
  ///< The budget of objective evaluations ran out.
  line_search_failed,       // no acceptable step along the direction
  ///< No acceptable step along the direction.
  start_not_admissible      // the objective is not finite at the start
  ///< The objective is not finite at the start (also the default before a run fills the report).
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
 * Reported in `optimizer_report::final_factorization` for methods that
 * factorize the curvature (`newton_optimizer`, `trust_region_optimizer`, and
 * `proximal_newton_optimizer`); remains `not_evaluated` for all other methods.
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
 * \brief Summary of an optimization run returned by `minimize()`.
 *
 * Records why the run stopped, the final accepted point and value, the
 * stationarity measure, evaluation counts, and elapsed wall-clock time. All
 * fields have defaults set before the run populates them.
 *
 * ### Fields
 *
 * | Field | Type | Default | Meaning |
 * |---|---|---|---|
 * | `status` | `optimizer_status` | `start_not_admissible` | Why the run stopped |
 * | `point` | `Kokkos::View<double*, Kokkos::HostSpace>` | empty | Last accepted point \f$x\f$ |
 * | `value` | `double` | quiet NaN | Objective value \f$f(x)\f$ at `point` |
 * | `gradient` | `Kokkos::View<double*, Kokkos::HostSpace>` | empty | Gradient at `point`; empty for value-only methods |
 * | `decrement` | `double` | quiet NaN | Stationarity measure in the objective's units |
 * | `search_size` | `double` | quiet NaN | Size of the region still searched by a derivative-free method, in the units of \f$x\f$ |
 * | `final_factorization` | `curvature_factorization` | `not_evaluated` | Outcome of the last curvature factorization |
 * | `iterations` | `int` | \f$0\f$ | Iterations completed |
 * | `value_evaluations` | `long` | \f$0\f$ | Evaluations of \f$f\f$ |
 * | `gradient_evaluations` | `long` | \f$0\f$ | Evaluations of \f$\nabla f\f$ |
 * | `curvature_evaluations` | `long` | \f$0\f$ | Evaluations of \f$H\f$ |
 * | `damped_coordinates` | `int` | \f$0\f$ | Coordinate updates damped by proximal Newton |
 * | `eigen_modified_iterations` | `int` | \f$0\f$ | Iterations of proximal Newton whose indefinite curvature was eigenvalue-corrected |
 * | `decrement_from_prox_mapping` | `bool` | `false` | Whether proximal Newton reported its fallback measure in `decrement` |
 * | `elapsed_seconds` | `double` | \f$0\f$ | Wall-clock time of the run from `std::chrono::steady_clock` |
 *
 * ### Measures and counts
 *
 * One call to `value()` adds \f$1\f$ to `value_evaluations`; one call to
 * `value_gradient()` adds \f$1\f$ to `value_evaluations` and \f$1\f$ to
 * `gradient_evaluations`; one call to `value_gradient_curvature()` adds \f$1\f$
 * to each of the three counters.
 *
 * `decrement` is the decrease of \f$f\f$ predicted by the next step in the
 * objective's units: \f$\frac{1}{2} g^\top H^{-1} g\f$ with curvature, and
 * \f$\frac{1}{2} g^\top B g\f$ without curvature for a method whose next
 * direction is \f$d = -B g\f$ with inverse-curvature model \f$B\f$ (such as
 * the BFGS inverse or the L-BFGS two-loop product). It is never
 * \f$|g^\top \Delta x|\f$ for the step \f$\Delta x\f$ just taken, which is
 * near zero after any near-exact line search regardless of location. Methods
 * that measure stationarity differently store their own measure in this
 * field, and methods with no such measure leave it quiet NaN; see
 * \ref optimizers "the optimizers page" for the per-method table.
 *
 * `search_size` holds Brent's final bracket half-width, the Nelder-Mead
 * simplex diameter from its best vertex
 * \f$\max_{1 \le j \le n} \lVert v_j - v_0 \rVert_2\f$, or the compass search
 * poll step, and is quiet NaN for methods with derivatives.
 *
 * `damped_coordinates` counts the coordinate updates in proximal Newton's
 * inner coordinate descent whose diagonal curvature \f$H_{jj}\f$ was replaced
 * (penalized coordinates with \f$H_{jj} \le \rho_j\f$, where \f$\rho_j\f$ is
 * the block's weak-convexity constant, or unpenalized coordinates with
 * \f$H_{jj} \le 0\f$); it is \f$0\f$ for all other methods.
 * `eigen_modified_iterations` counts the iterations of proximal Newton where
 * \f$H\f$ had no Cholesky factor and its smallest eigenvalue lay below
 * \f$-\eta \max_i |w_i|\f$ (with eigenvalues \f$w_i\f$ and relative floor
 * \f$\eta\f$ given by `eigen_floor`), so \f$H\f$ was replaced by its
 * eigenvalue-corrected form and `decrement` was set to quiet NaN; positive
 * semidefinite singular matrices are not counted, and it is \f$0\f$ for all
 * other methods.
 * `decrement_from_prox_mapping` is `true` when proximal Newton reports its
 * fallback proximal gradient mapping measure in `decrement` because \f$H\f$ is
 * finite and its restriction to the free coordinates has no Cholesky factor,
 * and `false` otherwise and for every other method; see \ref optimizers_proximal "Proximal methods".
 * `multistart_optimizer` (\ref optimizers_multistart) returns the kept run's
 * report with `iterations`, the three evaluation counts, and `elapsed_seconds`
 * summed over all starts.
 *
 * ### Example
 *
 * Here `objective` and `x0` are an objective and a starting point defined
 * elsewhere:
 *
 * \code
 * laplace::optimizers::bfgs_optimizer<> opt;
 * const laplace::optimizers::optimizer_report report = laplace::optimizers::minimize(opt, objective, x0);
 * if (!laplace::optimizers::optimizer_succeeded(report.status)) {
 *   // report.status says why; report.decrement says how far from stationarity
 * }
 * \endcode
 *
 * \see \ref optimizers, optimizer_status, curvature_factorization, minimize()
 */
struct optimizer_report {
  /**
   * \ingroup optimizers
   * \brief Reason the run stopped (default `optimizer_status::start_not_admissible`).
   */
  optimizer_status status = optimizer_status::start_not_admissible;
  /**
   * \ingroup optimizers
   * \brief Last accepted point \f$x\f$ (default empty).
   */
  Kokkos::View<double*, Kokkos::HostSpace> point;     // the last accepted point
  /**
   * \ingroup optimizers
   * \brief Objective value \f$f(x)\f$ at `point` (default quiet NaN).
   */
  double value = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Gradient at `point`; empty for value-only methods (default empty).
   */
  Kokkos::View<double*, Kokkos::HostSpace> gradient;  // at point; empty for C0 methods
  /**
   * \ingroup optimizers
   * \brief Stationarity measure in the objective's units, or quiet NaN when unavailable (default quiet NaN).
   *
   * Predicted decrease \f$\frac{1}{2} g^\top H^{-1} g\f$ with curvature, or
   * \f$\frac{1}{2} g^\top B g\f$ without it where \f$B\f$ is the method's
   * inverse-curvature model. The proximal methods, the bundle method and the
   * primal-dual method store their own stationarity measure, in the
   * objective's units, here.
   */
  // The stationarity measure in the objective's units, the decrease predicted
  // by the next step: (1/2) g' H^-1 g with curvature, (1/2) g' B g with B the
  // method's inverse-curvature model without (stop_rule_input); NaN when the
  // method has neither.
  double decrement = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Size of the region still searched by a derivative-free method in the units of \f$x\f$, or quiet NaN (default quiet NaN).
   */
  // Brent's final bracket half-width, a simplex's diameter, a poll step; NaN
  // for the methods with derivatives.
  double search_size = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Outcome of the final curvature factorization (default `curvature_factorization::not_evaluated`).
   */
  curvature_factorization final_factorization = curvature_factorization::not_evaluated;
  /**
   * \ingroup optimizers
   * \brief Number of iterations completed (default \f$0\f$).
   */
  int iterations = 0;
  /**
   * \ingroup optimizers
   * \brief Total objective value evaluations (default \f$0\f$; `value_gradient` adds one value and one gradient, `value_gradient_curvature` adds one of each).
   */
  // Evaluations of f, grad f and H: value_gradient counts one f and one g,
  // value_gradient_curvature one of each.
  long value_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Total gradient evaluations (default \f$0\f$).
   */
  long gradient_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Total curvature matrix evaluations (default \f$0\f$).
   */
  long curvature_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Coordinate updates damped by proximal Newton (penalized with \f$H_{jj} \le \rho_j\f$ or unpenalized with \f$H_{jj} \le 0\f$); \f$0\f$ for all other methods (default \f$0\f$).
   */
  int damped_coordinates = 0;
  /**
   * \ingroup optimizers
   * \brief Iterations of proximal Newton with an eigenvalue-corrected indefinite curvature; \f$0\f$ for all other methods (default \f$0\f$).
   *
   * Counts iterations where \f$H\f$ had no Cholesky factor and its smallest
   * eigenvalue lay below \f$-\eta \max_i |w_i|\f$ (with eigenvalues \f$w_i\f$
   * and relative floor \f$\eta\f$ given by `eigen_floor`), triggering
   * eigenvalue correction and a quiet NaN `decrement`. Positive semidefinite
   * singular matrices are not counted. See \ref optimizers_proximal "Proximal methods".
   */
  int eigen_modified_iterations = 0;
  /**
   * \ingroup optimizers
   * \brief Whether `decrement` is proximal Newton's fallback measure (default `false`).
   *
   * True when proximal Newton reports the proximal gradient mapping measure in
   * `decrement` because \f$H\f$ is finite and its restriction to the free
   * coordinates has no Cholesky factor; `false` otherwise and for every other
   * method. See \ref optimizers_proximal "Proximal methods".
   */
  bool decrement_from_prox_mapping = false;  // proximal Newton's fallback measure (note/optimizers.md section 12, point 16)
  /**
   * \ingroup optimizers
   * \brief Wall-clock duration of the run in seconds from `std::chrono::steady_clock` (default \f$0\f$).
   */
  double elapsed_seconds = 0.0;
};

/**
 * \ingroup optimizers
 * \brief Data passed to an `optimizer_trace` callback on each accepted iteration.
 *
 * The Views `point` and `gradient` reference the optimizer's own internal
 * buffers and are valid during the callback invocation only.
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
   * \brief Gradient at `point` (valid during the callback only; empty for value-only methods).
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
 * \brief Optional callback invoked once per accepted iteration with an `optimizer_trace_point`.
 *
 * Exposed as the optional `trace` member of every optimizer. When set, it is
 * called once per accepted iteration (once per sweep in
 * `block_coordinate_optimizer`); an empty callback costs one test per
 * iteration.
 *
 * \see \ref optimizers, optimizer_trace_point
 */
// The trace: a callback that an optimizer calls once per accepted iteration
// when it is set (an empty std::function costs one test per iteration).
using optimizer_trace = std::function<void(const optimizer_trace_point&)>;

}  // namespace laplace::optimizers
