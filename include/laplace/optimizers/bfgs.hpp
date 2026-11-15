#pragma once

// BFGS with a dense inverse-Hessian approximation B (note/optimizers.md,
// decisions 4 and 5; Nocedal and Wright 2006, Algorithm 6.1). The loop below
// is shared with L-BFGS (lbfgs.hpp), which differs only in how it stores B.
//
// - Direction d = -B g, step 1 first along it, strong Wolfe line search by
//   default (Armijo accepted: the gradient is then evaluated at the accepted
//   point).
// - Before any curvature pair has been accepted (first iteration, after a
//   reset) B = min(1, 1 / |g|_inf) I: a first trial displacement of order one
//   in the parameters, never longer than -g (an unscaled -g with a gradient
//   of 12.3 on a log-scale parameter puts the first trial at exp(-12.3) times
//   the start).
// - The pair s = x_new - x, y = g_new - g is used only when
//   s'y > curvature_tolerance |s| |y|; otherwise the update is skipped (a small
//   s'y makes rho = 1 / s'y explode and one such update destroys B), and after
//   max_skipped_updates consecutive skips B is reset.
// - The first accepted pair scales the identity by s'y / y'y before the
//   update (Shanno and Phua 1978): without it the first quasi-Newton step has
//   the magnitude of a unit Hessian.
// - A direction that is not finite or not a descent direction resets B.
// - Stationarity (decision 5, amended; the floor and the confirmation decided
//   2026-11-15): the decrement (1/2) max(g' B g, gamma_max g' g) with
//   g' B g = |g' d|, d = -B g the NEXT direction, and gamma_max the largest
//   s'y / y'y of the accepted pairs (0 before any pair). B is accurate only
//   along the directions its pairs have sampled; elsewhere it keeps the scale
//   of the stiff ones, and (1/2) g' B g alone measured 6.6e3 to 6.9e8 times
//   below f - f* on quadratics with kappa 1e4 to 1e10 (a plateau that the
//   method left a few iterations later). The floor gamma_max g' g gives every
//   direction at least the largest inverse curvature seen.
// - A convergence verdict of the stopping rule is confirmed on the next
//   iteration: the rule must hold twice in a row. One secant pair taken
//   across a region of varying curvature can still mislead the floor
//   (f = x - log x from 0.01: one step to 1.01, s'y / y'y = 0.0101 against
//   the inverse curvature 1 there, "converged" with f - f* = 5e-5). When the
//   confirming line search finds no acceptable step (below rounding, below
//   its resolution, no descent direction left, or its evaluation budget
//   spent), the pending verdict stands: the confirmation exists to unmask a
//   false success, which shows as an acceptable descent step being found. A
//   start at the optimum with f rounded at about 1e3 eps |f| (a sum reduced
//   in another order on a GPU) spent the 30 evaluations of the Wolfe search.

#include "laplace/optimizers/line_search.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_quasi_newton
 * \brief Bitmask of stopping-rule inputs provided by quasi-Newton methods.
 *
 * Combines `stop_needs_gradient`, `stop_needs_decrement`, and `stop_needs_previous`, so any `stop_rule` needing at most the gradient, the decrement, and the previous point and value satisfies `stop_rule_for`.
 *
 * \see \ref optimizers_quasi_newton, bfgs_optimizer, lbfgs_optimizer, stop_rule_for
 */
// The quantities a quasi-Newton method gives its stopping rule.
inline constexpr unsigned quasi_newton_provides = stop_needs_gradient | stop_needs_decrement | stop_needs_previous;

namespace detail {

// d = -min(1, 1 / |g|_inf) g, the direction without curvature.
inline void scaled_steepest_descent(const optimizer_point& g, const optimizer_vector& d) {
  const double gmax = max_abs(g);
  const double scale = std::isfinite(gmax) && gmax > 1.0 ? 1.0 / gmax : 1.0;
  for (std::size_t i = 0; i < g.extent(0); ++i) d(i) = -scale * g(i);
}

// The dense inverse-Hessian model of BFGS: B, column-major n x n.
struct bfgs_inverse_model {
  int n = 0;
  std::vector<double> b;
  bool has_pair = false;
  std::vector<double> by;  // workspace: B y

  explicit bfgs_inverse_model(int size) : n(size), b(size * size, 0.0), by(size) {}
  void reset() { has_pair = false; }
  void direction(const optimizer_point& g, const optimizer_vector& d) const {
    if (!has_pair) return scaled_steepest_descent(g, d);
    for (int i = 0; i < n; ++i) d(i) = 0.0;
    for (int j = 0; j < n; ++j) {
      const double gj = g(j);
      for (int i = 0; i < n; ++i) d(i) -= b[i + j * n] * gj;
    }
  }
  // B+ = (I - rho s y') B (I - rho y s') + rho s s'
  //    = B - rho (s (By)' + (By) s') + (rho^2 y'By + rho) s s', rho = 1 / s'y.
  void update(const std::vector<double>& s, const std::vector<double>& y, double sy) {
    if (!has_pair) {
      double yy = 0.0;
      for (int i = 0; i < n; ++i) yy += y[i] * y[i];
      std::fill(b.begin(), b.end(), 0.0);
      for (int i = 0; i < n; ++i) b[i + i * n] = sy / yy;
      has_pair = true;
    }
    const double rho = 1.0 / sy;
    double yby = 0.0;
    for (int i = 0; i < n; ++i) {
      double t = 0.0;
      for (int j = 0; j < n; ++j) t += b[i + j * n] * y[j];
      by[i] = t;
      yby += y[i] * t;
    }
    const double c = rho * rho * yby + rho;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) b[i + j * n] += -rho * (s[i] * by[j] + by[i] * s[j]) + c * s[i] * s[j];
  }
};

// The loop of BFGS and L-BFGS. Model: reset(), direction(g, d) writing -B g,
// update(s, y, s'y) for an accepted pair.
template <class Model, class Rule, class LineSearch, gradient_objective O>
optimizer_report quasi_newton_run(Model& model, const Rule& stop_rule, const LineSearch& line_search,
                                  int max_iterations, long max_evaluations, double curvature_tolerance,
                                  int max_skipped_updates, const optimizer_trace& trace, O& objective,
                                  const optimizer_point& x0) {
  const auto start = std::chrono::steady_clock::now();
  const int n = static_cast<int>(x0.extent(0));
  optimizer_report report;
  optimizer_vector x("x", n), g("g", n), d("d", n), x_trial("x_trial", n), g_trial("g_trial", n),
      x_previous("x_previous", n);
  std::vector<double> s(n), y(n);
  Kokkos::deep_copy(x, x0);
  double f = objective.value_gradient(x, g);
  ++report.value_evaluations, ++report.gradient_evaluations;
  auto finish = [&](optimizer_status status) {
    report.status = status;
    report.point = x, report.value = f, report.gradient = g;
    report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return report;
  };
  if (!std::isfinite(f) || !std::isfinite(max_abs(g))) return finish(optimizer_status::start_not_admissible);

  // the next direction and its decrement; a direction that is not finite or
  // not a descent direction resets the model
  double gamma_max = 0.0;
  auto next_direction = [&] {
    model.direction(g, d);
    double gd = detail::dot(g, d);
    if (!std::isfinite(gd) || gd >= 0.0) {
      model.reset();
      model.direction(g, d);
      gd = detail::dot(g, d);
    }
    report.decrement = 0.5 * std::max(std::abs(gd), gamma_max * dot(g, g));
  };
  auto verdict_status = [](stop_verdict v) {
    return v == stop_verdict::converged                 ? optimizer_status::converged
           : v == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                        : optimizer_status::stopped_on_change;
  };

  next_direction();
  stop_verdict pending = stop_verdict::none;  // a pass awaiting confirmation
  // the rule's verdict, a convergence only confirmed on its second consecutive pass
  auto confirmed = [&](stop_verdict v) {
    if (v == stop_verdict::converged || v == stop_verdict::converged_at_resolution) {
      if (pending == stop_verdict::none) {
        pending = v;
        return stop_verdict::none;
      }
      return v;
    }
    pending = stop_verdict::none;
    return v;
  };
  {
    stop_rule_input in;
    in.iteration = 0, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
    const stop_verdict v = confirmed(stop_rule.verdict(in));
    if (v != stop_verdict::none) return finish(verdict_status(v));
  }
  int skipped = 0;
  for (int k = 1; k <= max_iterations; ++k) {
    if (report.value_evaluations >= max_evaluations) return finish(optimizer_status::evaluation_budget);
    line_search_result r;
    double f_new;
    if constexpr (requires { line_search.search(objective, x, f, g, d, 1.0, x_trial, g_trial); }) {
      r = line_search.search(objective, x, f, g, d, 1.0, x_trial, g_trial);
      f_new = r.value;
    } else {
      r = line_search.search(objective, x, f, g, d, 1.0, x_trial);
      f_new = r.accepted() ? objective.value_gradient(x_trial, g_trial) : r.value;
      if (r.accepted()) ++report.value_evaluations, ++report.gradient_evaluations;
    }
    report.value_evaluations += r.value_evaluations;
    report.gradient_evaluations += r.gradient_evaluations;
    // a pass whose confirming search finds no acceptable step keeps its verdict
    if (!r.accepted() && pending != stop_verdict::none &&
        (r.outcome == line_search_outcome::step_below_rounding || r.outcome == line_search_outcome::below_resolution ||
         r.outcome == line_search_outcome::not_descent_direction ||
         r.outcome == line_search_outcome::evaluation_budget))
      return finish(verdict_status(pending));
    if (!r.accepted())
      return finish(r.outcome == line_search_outcome::step_below_rounding ||
                            r.outcome == line_search_outcome::below_resolution
                        ? optimizer_status::stalled
                        : optimizer_status::line_search_failed);

    double sy = 0.0, ss = 0.0, yy = 0.0;
    for (int i = 0; i < n; ++i) {
      s[i] = x_trial(i) - x(i);
      y[i] = g_trial(i) - g(i);
      sy += s[i] * y[i], ss += s[i] * s[i], yy += y[i] * y[i];
    }
    Kokkos::deep_copy(x_previous, x);
    Kokkos::deep_copy(x, x_trial);
    Kokkos::deep_copy(g, g_trial);
    const double f_previous = f;
    f = f_new;
    report.iterations = k;

    if (std::isfinite(sy) && sy > curvature_tolerance * std::sqrt(ss) * std::sqrt(yy)) {
      model.update(s, y, sy);
      gamma_max = std::max(gamma_max, sy / yy);
      skipped = 0;
    } else if (++skipped >= max_skipped_updates) {
      model.reset();
      skipped = 0;
    }

    next_direction();
    if (trace) trace({k, x, f, g, report.decrement, r.step});
    stop_rule_input in;
    in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
    in.has_previous = true, in.previous_value = f_previous, in.previous_point = x_previous;
    const stop_verdict v = confirmed(stop_rule.verdict(in));
    if (v != stop_verdict::none) return finish(verdict_status(v));
  }
  return finish(optimizer_status::iteration_budget);
}

}  // namespace detail

/**
 * \ingroup optimizers_quasi_newton
 * \brief BFGS quasi-Newton optimizer with a dense \f$n \times n\f$ inverse-Hessian approximation.
 *
 * `bfgs_optimizer<Rule, LineSearch>` minimizes an unconstrained objective
 * modeling gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is the
 * parameter dimension, using a dense \f$n \times n\f$ symmetric positive
 * definite model of the inverse Hessian. Everything runs on the host only;
 * nothing is marked `LAPLACE_HD`. A run is started with
 * `minimize(bfgs_optimizer<>{}, f, x0)`.
 *
 * ### Algorithm
 *
 * Let \f$x_k \in \mathbb{R}^n\f$ be the iterate at step \f$k\f$,
 * \f$g_k = \nabla f(x_k)\f$ its gradient, \f$B_k\f$ the \f$n \times n\f$ model
 * of the inverse Hessian, \f$d_k = -B_k g_k\f$ the search direction, and
 * \f$t_k\f$ the step length accepted by the line search started at \f$t = 1\f$.
 * Set \f$s_k = x_{k+1} - x_k = t_k d_k\f$, \f$y_k = g_{k+1} - g_k\f$, and
 * \f$\rho_k = 1 / (s_k^\top y_k)\f$. The BFGS update (Nocedal and Wright, 2006,
 * Algorithm 6.1) is
 * \f[
 * B_{k+1} = (I - \rho_k s_k y_k^\top) B_k (I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top,
 * \f]
 * where \f$I\f$ is the \f$n \times n\f$ identity matrix, evaluated in
 * \f$O(n^2)\f$ operations as
 * \f[
 * B_{k+1} = B_k - \rho_k \bigl(s_k (B_k y_k)^\top + (B_k y_k) s_k^\top\bigr) + \bigl(\rho_k^2\, y_k^\top B_k y_k + \rho_k\bigr)\, s_k s_k^\top.
 * \f]
 *
 * Before any curvature pair has been accepted, and after a reset, the model
 * starts at \f$B = \min\bigl(1,\, 1 / \lVert g\rVert_\infty\bigr) I\f$, giving
 * the steepest-descent direction \f$-g\f$ scaled so that its largest component
 * magnitude is at most \f$1\f$. When the first pair \f$(s, y)\f$ is accepted,
 * the model first sets \f$B = \bigl((s^\top y) / (y^\top y)\bigr) I\f$
 * (Shanno and Phua, 1978) and then applies the BFGS update.
 *
 * A curvature pair \f$(s, y)\f$ is accepted only when \f$s^\top y\f$ is finite
 * and \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$, where
 * \f$\tau\f$ is `curvature_tolerance` and \f$\lVert \cdot\rVert_2\f$ is the
 * Euclidean norm; otherwise the update is skipped. After `max_skipped_updates`
 * consecutive skips, \f$B\f$ is reset. Whenever a computed direction \f$d\f$
 * is not finite (its directional slope \f$g^\top d\f$ is not finite) or
 * satisfies \f$g^\top d \ge 0\f$, \f$B\f$ is reset and \f$d\f$ is recomputed
 * from the reset model.
 *
 * Each iteration searches along \f$d_k\f$ from \f$t = 1\f$ using `LineSearch`,
 * which defaults to wolfe_line_search (see \ref optimizers "the optimizers page").
 * When armijo_line_search (Dennis and Schnabel, 1996) is used instead, the
 * optimizer evaluates the gradient once more at each accepted point.
 *
 * ### Stationarity and confirmation
 *
 * At every iteration \f$k\f$, including the start \f$k = 0\f$, the optimizer
 * computes the next direction \f$d_k\f$ and sets the stationarity decrement
 * to
 * \f[
 * \delta_k = \frac{1}{2} \max\bigl(|g_k^\top d_k|,\ \gamma_{\max}\, g_k^\top g_k\bigr),
 * \f]
 * where \f$\gamma_{\max}\f$ is the largest \f$(s^\top y) / (y^\top y)\f$ over
 * all accepted pairs (\f$0\f$ before any pair). This \f$\delta_k\f$ is passed to
 * `stop_rule` as `decrement` and stored in `optimizer_report::decrement`.
 * The floor \f$\gamma_{\max} g_k^\top g_k\f$ is needed because \f$B_k\f$ is
 * accurate only along directions its pairs have sampled.
 *
 * A `stop_verdict::converged` or `stop_verdict::converged_at_resolution`
 * verdict from `stop_rule` must hold at two consecutive iterations before the
 * run ends. If the confirming line search finds no acceptable step, with
 * outcome `line_search_outcome::step_below_rounding`,
 * `line_search_outcome::below_resolution`,
 * `line_search_outcome::not_descent_direction` or
 * `line_search_outcome::evaluation_budget`, the pending verdict stands: the
 * confirmation unmasks a false success, which shows as an acceptable descent
 * step, and near the optimum the rounding of \f$f\f$ can leave none. A
 * `stop_verdict::stopped_on_change` verdict is not confirmed and terminates at
 * once. Starting at an exact stationary point with \f$g = 0\f$ ends with
 * status `optimizer_status::converged` after \f$0\f$ iterations under the
 * default rule.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration; tolerance \f$10^{-6}\f$ in the objective's units, accepting any rule needing at most gradient, decrement, and previous point and value (checked against quasi_newton_provides) |
 * | `line_search` | `wolfe_line_search{}` | Line search along each direction from step \f$1\f$, with \f$c_1 = 10^{-4}\f$ and \f$c_2 = 0.9\f$ |
 * | `max_iterations` | \f$1000\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^5\f$ | Value-evaluation budget checked before each line search, so a run can exceed it by one search |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold \f$\tau\f$ for accepting pairs; larger values skip more updates |
 * | `max_skipped_updates` | \f$5\f$ | Consecutive skipped updates before resetting \f$B\f$ |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration with iteration, point, value, gradient, decrement, and step |
 *
 * ### Statuses
 *
 * - `optimizer_status::converged` and `optimizer_status::converged_at_resolution` are returned when the stopping rule's convergence verdict is confirmed.
 * - `optimizer_status::stopped_on_change` is returned immediately when a change rule issues `stop_verdict::stopped_on_change`.
 * - `optimizer_status::stalled` is returned when the line search fails with `line_search_outcome::step_below_rounding` or `line_search_outcome::below_resolution` outside confirmation.
 * - `optimizer_status::line_search_failed` is returned on any other line search failure, such as its evaluation budget or a non-descent direction after a reset.
 * - `optimizer_status::iteration_budget` is returned when `max_iterations` iterations complete without stopping.
 * - `optimizer_status::evaluation_budget` is returned when `value_evaluations` reaches `max_evaluations` before a line search.
 * - `optimizer_status::start_not_admissible` is returned when the objective value or gradient is non-finite at \f$x_0\f$.
 *
 * In optimizer_report, `curvature_evaluations` is always \f$0\f$ and
 * `final_factorization` is `curvature_factorization::not_evaluated`.
 *
 * ### Cost and measured behaviour
 *
 * Each iteration performs \f$O(n^2)\f$ work and stores \f$n^2\f$ `double`
 * entries for \f$B\f$; each Wolfe trial evaluates the objective value and
 * gradient once. Measured with gcc on one thread, Rosenbrock with \f$n = 100\f$
 * takes 356 microseconds (36 iterations, 42 value evaluations).
 *
 * In the same gcc measurements with the default settings, BFGS converges on
 * all standard test problems (with \f$f^*\f$ shifted by \f$0\f$, \f$1\f$,
 * \f$10^3\f$, and \f$10^6\f$) with \f$f - f^* \le 6.3 \times 10^{-8}\f$, and on
 * \f$10\f$-dimensional quadratics up to condition number \f$\kappa = 10^8\f$
 * (with \f$f - f^* \le 4.7 \times 10^{-9}\f$); at \f$\kappa = 10^{10}\f$ and
 * \f$10^{12}\f$ it ends `optimizer_status::stalled`. It does not converge on a
 * non-smooth objective such as \f$\sum_i |x_i|\f$ (tested for \f$n = 2, 4, 10\f$
 * under both line searches).
 *
 * ### Example
 *
 * \code
 * bfgs_optimizer<> opt;
 * opt.stop_rule.tolerance = 1e-8;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, quasi_newton_provides>`.
 * \tparam LineSearch Line search type such as `wolfe_line_search` or `armijo_line_search`.
 * \see \ref optimizers_quasi_newton, lbfgs_optimizer, quasi_newton_provides, minimize(), optimizer_report
 */
// BFGS with a dense n x n inverse-Hessian approximation: O(n^2) memory and
// work per iteration. Settings are members; minimize(bfgs_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = wolfe_line_search>
  requires stop_rule_for<Rule, quasi_newton_provides>
struct bfgs_optimizer {
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Line search instance used along each search direction from step \f$1\f$ (default `wolfe_line_search{}`).
   */
  LineSearch line_search{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default \f$1000\f$).
   */
  int max_iterations = 1000;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Value-evaluation budget checked before each line search (default \f$10^5\f$; a run may exceed it by one search).
   */
  long max_evaluations = 100000;       // checked before each line search
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimum cosine threshold \f$\tau\f$ for accepting a curvature pair when \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;  // the pair is used when s'y > tol |s| |y|
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Number of consecutive skipped updates before resetting the inverse-Hessian approximation (default \f$5\f$).
   */
  int max_skipped_updates = 5;         // consecutive skips before B is reset
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Optional trace callback invoked once per accepted iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimizes `objective` starting from `x0` using BFGS.
   *
   * See bfgs_optimizer for the update, stationarity confirmation, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_quasi_newton, bfgs_optimizer, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    detail::bfgs_inverse_model model(static_cast<int>(x0.extent(0)));
    return detail::quasi_newton_run(model, stop_rule, line_search, max_iterations, max_evaluations,
                                    curvature_tolerance, max_skipped_updates, trace, objective, x0);
  }
};

}  // namespace laplace::optimizers
