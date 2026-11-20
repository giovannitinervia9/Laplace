#pragma once

// Barzilai-Borwein (note/optimizers.md, section 13, point 3; Barzilai and
// Borwein 1988, Raydan 1997): d = -g with the step alpha_k offered to a
// nonmonotone line search, on the loop of gradient_descent.hpp.
//
// - alpha_0 = min(1, 1 / |g|_inf). After an accepted step with
//   s'y > curvature_tolerance |s| |y|: BB1 = s's / s'y (long) or
//   BB2 = s'y / y'y (short), or the two alternating (BB1 at odd iterations),
//   clipped to [alpha_min, alpha_max]. Otherwise (s'y <= 0: negative or no
//   curvature seen along s) alpha = min(1, 1 / |g|_inf) again. Keeping the
//   previous alpha is an absorbing state: a short step samples the curvature
//   over a short interval, on a valley floor that curvature is not positive,
//   the pair is refused again (measured on Rosenbrock with BB2: alpha frozen at
//   9.3e-4 for 650 iterations, 759 in all, 51 with the reset).
// - Stationarity: B = alpha_{k+1} I, decrement (1/2) max(alpha_{k+1}, gamma_max) g'g.

#include "laplace/optimizers/gradient_descent.hpp"

#include <algorithm>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_first_order
 * \brief Step-size formula used by barzilai_borwein_optimizer.
 *
 * Selects how the next trial scale \f$\alpha_{k+1}\f$ is computed from an
 * accepted secant pair \f$(s, y) = (x_{k+1} - x_k,\, g_{k+1} - g_k)\f$
 * satisfying \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$
 * (Barzilai and Borwein, 1988):
 * - `long_step`: \f$\alpha_{k+1} = s^\top s / s^\top y\f$.
 * - `short_step`: \f$\alpha_{k+1} = s^\top y / y^\top y\f$.
 * - `alternating`: uses `long_step` at odd iterations and `short_step` at even iterations (default).
 *
 * \see \ref optimizers_first_order, barzilai_borwein_optimizer
 */
// Which Barzilai-Borwein step: s's / s'y, s'y / y'y, or the two alternating.
enum class barzilai_borwein_step { long_step, short_step, alternating };

namespace detail {

struct barzilai_borwein_steps {
  barzilai_borwein_step variant;
  double alpha_min, alpha_max;
  double alpha = 1.0;
  int iteration = 0;
  void start(const optimizer_point& g) { alpha = std::clamp(scaled_unit_step(g), alpha_min, alpha_max); }
  double trial_step() const { return alpha; }
  double scale() const { return alpha; }
  void accepted(const first_order_step& st, const optimizer_point& g) {
    ++iteration;
    if (!st.pair) {  // a displacement of order one, never the previous alpha
      alpha = std::clamp(scaled_unit_step(g), alpha_min, alpha_max);
      return;
    }
    const bool use_long = variant == barzilai_borwein_step::long_step ||
                          (variant == barzilai_borwein_step::alternating && iteration % 2 == 1);
    const double next = use_long ? st.ss / st.sy : st.sy / st.yy;
    if (std::isfinite(next)) alpha = std::clamp(next, alpha_min, alpha_max);
  }
};

}  // namespace detail

/**
 * \ingroup optimizers_first_order
 * \brief Barzilai-Borwein gradient optimizer with nonmonotone line search.
 *
 * `barzilai_borwein_optimizer<Rule, LineSearch>` minimizes an unconstrained
 * objective modeling gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is
 * the parameter dimension, on the host only; nothing is marked `LAPLACE_HD`. A
 * run starts with `minimize(barzilai_borwein_optimizer<>{}, f, x0)`.
 *
 * ### Algorithm
 *
 * Shares the iteration loop and scaled direction \f$d_k = -\alpha_k g_k\f$
 * (searched from step \f$1\f$) with gradient_descent_optimizer. The initial
 * scale is \f$\alpha_0 = \min\bigl(1,\, 1 / \lVert g_0 \rVert_\infty\bigr)\f$,
 * clamped to `[alpha_min, alpha_max]`. After an accepted step
 * \f$s = x_{k+1} - x_k\f$ with \f$y = g_{k+1} - g_k\f$ satisfying
 * \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (\f$\tau\f$ equal to
 * `curvature_tolerance`), the next scale \f$\alpha_{k+1}\f$ is chosen
 * according to `step` (Barzilai and Borwein, 1988):
 * `barzilai_borwein_step::long_step` uses \f$s^\top s / s^\top y\f$,
 * `barzilai_borwein_step::short_step` uses \f$s^\top y / y^\top y\f$, and
 * `barzilai_borwein_step::alternating` uses the long step at odd iterations
 * and the short step at even ones. Any finite result is clamped to
 * `[alpha_min, alpha_max]`. When the curvature pair is refused,
 * \f$\alpha_{k+1}\f$ resets to
 * \f$\min\bigl(1,\, 1 / \lVert g_{k+1} \rVert_\infty\bigr)\f$ clamped to
 * `[alpha_min, alpha_max]`.
 *
 * `LineSearch` defaults to nonmonotone_line_search (Raydan, 1997), and also
 * accepts armijo_line_search and wolfe_line_search. With `wolfe_line_search`,
 * the gradient at the accepted point comes from the search; otherwise
 * `value_gradient()` is evaluated once more at the accepted point.
 *
 * ### Stationarity and confirmation
 *
 * At every iteration \f$k\f$, including \f$k = 0\f$, the stationarity
 * decrement passed to `stop_rule` and stored in `optimizer_report::decrement`
 * is
 * \f[
 * \delta_k = \frac{1}{2} \max(\alpha_k, \gamma_{\max})\, g_k^\top g_k,
 * \f]
 * where \f$\alpha_k\f$ is the next trial scale and \f$\gamma_{\max}\f$ is the
 * largest \f$s^\top s / s^\top y\f$ over accepted pairs satisfying the curvature
 * condition (\f$0\f$ before any pair). Two-iteration confirmation and pending
 * verdicts on failed confirming searches follow gradient_descent_optimizer and
 * bfgs_optimizer.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration; tolerance \f$10^{-6}\f$ in the objective's units, accepting any rule matching first_order_provides |
 * | `line_search` | `nonmonotone_line_search{}` | Line search along \f$d_k\f$ from step \f$1\f$ (`nonmonotone_line_search`, `armijo_line_search`, or `wolfe_line_search`) |
 * | `step` | `barzilai_borwein_step::alternating` | Barzilai-Borwein step formula (`long_step`, `short_step`, or `alternating`) |
 * | `alpha_min` | \f$10^{-10}\f$ | Lower clamp on the trial step scale \f$\alpha_k\f$ |
 * | `alpha_max` | \f$10^{10}\f$ | Upper clamp on the trial step scale \f$\alpha_k\f$ |
 * | `max_iterations` | \f$10000\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^6\f$ | Value-evaluation budget checked before each line search |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold \f$\tau\f$ for accepting secant pairs |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration |
 *
 * ### Statuses
 *
 * Termination statuses follow gradient_descent_optimizer (`converged`,
 * `converged_at_resolution`, `stopped_on_change`, `stalled` without reset,
 * `line_search_failed`, `iteration_budget`, `evaluation_budget`, and
 * `start_not_admissible`). In optimizer_report, `curvature_evaluations` is
 * always \f$0\f$ and `final_factorization` is
 * `curvature_factorization::not_evaluated`.
 *
 * ### Example
 *
 * \code
 * barzilai_borwein_optimizer<> opt;
 * opt.step = barzilai_borwein_step::long_step;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, first_order_provides>`.
 * \tparam LineSearch Line search type such as `nonmonotone_line_search`, `armijo_line_search`, or `wolfe_line_search`.
 * \see \ref optimizers_first_order, barzilai_borwein_step, nonmonotone_line_search, gradient_descent_optimizer, first_order_provides, minimize(), optimizer_report
 */
// Barzilai-Borwein with Zhang and Hager's nonmonotone search by default.
// minimize(barzilai_borwein_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = nonmonotone_line_search>
  requires stop_rule_for<Rule, first_order_provides>
struct barzilai_borwein_optimizer {
  /**
   * \ingroup optimizers_first_order
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_first_order
   * \brief Line search instance used along \f$d_k = -\alpha_k g_k\f$ from step \f$1\f$ (default `nonmonotone_line_search{}`).
   */
  LineSearch line_search{};
  /**
   * \ingroup optimizers_first_order
   * \brief Barzilai-Borwein step formula (default `barzilai_borwein_step::alternating`).
   */
  barzilai_borwein_step step = barzilai_borwein_step::alternating;
  /**
   * \ingroup optimizers_first_order
   * \brief Lower clamp on the trial step scale \f$\alpha_k\f$ (default \f$10^{-10}\f$).
   */
  double alpha_min = 1e-10;
  /**
   * \ingroup optimizers_first_order
   * \brief Upper clamp on the trial step scale \f$\alpha_k\f$ (default \f$10^{10}\f$).
   */
  double alpha_max = 1e10;
  /**
   * \ingroup optimizers_first_order
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default \f$10000\f$).
   */
  int max_iterations = 10000;
  /**
   * \ingroup optimizers_first_order
   * \brief Value-evaluation budget checked before each line search (default \f$10^6\f$).
   */
  long max_evaluations = 1000000;      // checked before each line search
  /**
   * \ingroup optimizers_first_order
   * \brief Minimum cosine threshold \f$\tau\f$ for accepting secant pairs when \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;  // the pair is used when s'y > tol |s| |y|
  /**
   * \ingroup optimizers_first_order
   * \brief Optional trace callback invoked once per accepted iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_first_order
   * \brief Minimizes `objective` starting from `x0` using Barzilai-Borwein steps.
   *
   * See barzilai_borwein_optimizer for the step formulas, pair reset, stationarity confirmation, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_first_order, barzilai_borwein_optimizer, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    return detail::first_order_run(detail::barzilai_borwein_steps{step, alpha_min, alpha_max}, stop_rule,
                                   line_search, max_iterations, max_evaluations, curvature_tolerance, trace,
                                   objective, x0);
  }
};

}  // namespace laplace::optimizers
