#pragma once

// Gradient descent (note/optimizers.md, decisions 4, 5 and section 13). The
// loop below is shared with Barzilai-Borwein (barzilai_borwein.hpp); the two
// differ only in the step offered to the line search along d = -g.
//
// - Direction d = -alpha_k g, searched from step 1 (the accepted step along
//   -g is t_k = alpha_k times the search's step; scaling d rather than the
//   initial step keeps the slope g'd finite where |g|^2 overflows, as at an
//   exploded start). For gradient descent alpha_0 = min(1, 1 / |g|_inf) (a first
//   displacement of order one, as for BFGS), then
//   alpha_{k+1} = t_k g_k'g_k / g_{k+1}'g_{k+1} (Nocedal and Wright 2006,
//   section 3.5: the first-order change predicted by the last accepted step
//   t_k is offered again).
// - Line search: armijo_line_search by default; nonmonotone_line_search
//   (its reference C_k is kept here) and wolfe_line_search are accepted.
// - Stationarity (decision 5): B = b I, with b the last accepted step t_k
//   for gradient descent (alpha_0 before any step), the next step alpha_{k+1}
//   for Barzilai-Borwein, and the decrement
//   (1/2) max(b, gamma_max) g'g, gamma_max the largest s's / s'y over the
//   accepted pairs with s'y > curvature_tolerance |s| |y| (0 before any).
//   (detail::secant_floor; BFGS's s'y / y'y declared false successes here,
//   f - f* up to 4800 times the decrement).
//   A convergence verdict must hold on two consecutive iterations; a pending
//   verdict stands when the confirming search finds no acceptable step
//   (below rounding or resolution, no descent direction, budget spent), as
//   for BFGS.

#include "laplace/optimizers/line_search.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_first_order
 * \brief Bitmask of stopping-rule inputs provided by gradient descent, Barzilai-Borwein, and Adam.
 *
 * Combines `stop_needs_gradient`, `stop_needs_decrement`, and
 * `stop_needs_previous`, so any `stop_rule` needing at most the gradient,
 * the decrement, and the previous point and value satisfies `stop_rule_for`.
 *
 * \see \ref optimizers_first_order, gradient_descent_optimizer, barzilai_borwein_optimizer, adam_optimizer, stop_rule_for
 */
// The quantities gradient descent, Barzilai-Borwein and Adam give their
// stopping rule.
inline constexpr unsigned first_order_provides = stop_needs_gradient | stop_needs_decrement | stop_needs_previous;

namespace detail {

// min(1, 1 / |g|_inf): a first displacement of order one, never longer than g.
inline double scaled_unit_step(const optimizer_point& g) {
  const double gmax = max_abs(g);
  return std::isfinite(gmax) && gmax > 1.0 ? 1.0 / gmax : 1.0;
}

// What the loop needs from the method after an accepted step t along -g.
struct first_order_step {
  double step;            // t
  double sy, ss, yy;      // of s = x_new - x, y = g_new - g
  double gg_old, gg_new;  // g'g before and after
  bool pair;              // s'y > curvature_tolerance |s| |y|
};

// Method: start(g), trial_step() (the alpha_k offered to the search),
// scale() (b of B = b I) and accepted(first_order_step, g at the new point).
template <class Method, class Rule, class LineSearch, gradient_objective O>
optimizer_report first_order_run(Method method, const Rule& stop_rule, const LineSearch& line_search,
                                 int max_iterations, long max_evaluations, double curvature_tolerance,
                                 const optimizer_trace& trace, O& objective, const optimizer_point& x0) {
  const auto start = std::chrono::steady_clock::now();
  const int n = static_cast<int>(x0.extent(0));
  optimizer_report report;
  optimizer_vector x("x", n), g("g", n), d("d", n), x_trial("x_trial", n), g_trial("g_trial", n),
      x_previous("x_previous", n);
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

  constexpr bool nonmonotone = requires(nonmonotone_reference c) {
    line_search.search(objective, x, f, c, g, d, 1.0, x_trial);
  };
  nonmonotone_reference reference;
  reference.start(f);
  method.start(g);
  double gamma_max = 0.0, gg = detail::dot(g, g), alpha = 1.0;
  auto set_direction = [&] {
    alpha = method.trial_step();
    for (int i = 0; i < n; ++i) d(i) = -alpha * g(i);
    report.decrement = 0.5 * std::max(method.scale(), gamma_max) * gg;
  };
  auto verdict_status = [](stop_verdict v) {
    return v == stop_verdict::converged                 ? optimizer_status::converged
           : v == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                        : optimizer_status::stopped_on_change;
  };
  stop_verdict pending = stop_verdict::none;
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

  set_direction();
  {
    stop_rule_input in;
    in.iteration = 0, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
    const stop_verdict v = confirmed(stop_rule.verdict(in));
    if (v != stop_verdict::none) return finish(verdict_status(v));
  }
  for (int k = 1; k <= max_iterations; ++k) {
    if (report.value_evaluations >= max_evaluations) return finish(optimizer_status::evaluation_budget);
    constexpr double s0 = 1.0;  // along d = -alpha g
    line_search_result r;
    double f_new;
    if constexpr (nonmonotone) {
      r = line_search.search(objective, x, f, reference, g, d, s0, x_trial);
    } else if constexpr (requires { line_search.search(objective, x, f, g, d, s0, x_trial, g_trial); }) {
      r = line_search.search(objective, x, f, g, d, s0, x_trial, g_trial);
    } else {
      r = line_search.search(objective, x, f, g, d, s0, x_trial);
    }
    if constexpr (requires { line_search.search(objective, x, f, g, d, s0, x_trial, g_trial); }) {
      f_new = r.value;
    } else {
      f_new = r.accepted() ? objective.value_gradient(x_trial, g_trial) : r.value;
      if (r.accepted()) ++report.value_evaluations, ++report.gradient_evaluations;
    }
    report.value_evaluations += r.value_evaluations;
    report.gradient_evaluations += r.gradient_evaluations;
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
    // the gradient at a point accepted on its value alone may still be non-finite
    if (!std::isfinite(max_abs(g_trial))) return finish(optimizer_status::line_search_failed);

    first_order_step st{r.step * alpha, 0.0, 0.0, 0.0, gg, 0.0, false};
    for (int i = 0; i < n; ++i) {
      const double si = x_trial(i) - x(i), yi = g_trial(i) - g(i);
      st.sy += si * yi, st.ss += si * si, st.yy += yi * yi;
      st.gg_new += g_trial(i) * g_trial(i);
    }
    st.pair = std::isfinite(st.sy) && st.sy > curvature_tolerance * std::sqrt(st.ss) * std::sqrt(st.yy);
    if (st.pair) gamma_max = std::max(gamma_max, secant_floor(st.sy, st.ss, st.yy));
    Kokkos::deep_copy(x_previous, x);
    Kokkos::deep_copy(x, x_trial);
    Kokkos::deep_copy(g, g_trial);
    const double f_previous = f;
    f = f_new;
    gg = st.gg_new;
    if constexpr (nonmonotone) reference.accept(f, line_search.averaging);
    method.accepted(st, g);
    report.iterations = k;

    set_direction();
    if (trace) trace({k, x, f, g, report.decrement, st.step});
    stop_rule_input in;
    in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
    in.has_previous = true, in.previous_value = f_previous, in.previous_point = x_previous;
    const stop_verdict v = confirmed(stop_rule.verdict(in));
    if (v != stop_verdict::none) return finish(verdict_status(v));
  }
  return finish(optimizer_status::iteration_budget);
}

// Gradient descent's steps: alpha_0 = min(1, 1 / |g|_inf), then
// alpha_{k+1} = t_k g_k'g_k / g_{k+1}'g_{k+1}; b = t_k (alpha_0 before any).
struct gradient_descent_steps {
  double alpha = 1.0, last = 1.0;
  void start(const optimizer_point& g) { alpha = last = scaled_unit_step(g); }
  double trial_step() const { return alpha; }
  double scale() const { return last; }
  void accepted(const first_order_step& st, const optimizer_point&) {
    last = st.step;
    const double next = st.step * st.gg_old / st.gg_new;
    alpha = std::isfinite(next) && next > 0.0 ? next : st.step;  // g_new = 0: keep t_k
  }
};

}  // namespace detail

/**
 * \ingroup optimizers_first_order
 * \brief Steepest-descent optimizer with line search and first-order step scaling.
 *
 * `gradient_descent_optimizer<Rule, LineSearch>` minimizes an unconstrained
 * objective modeling gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is
 * the parameter dimension, on the host only; nothing is marked `LAPLACE_HD`. A
 * run starts with `minimize(gradient_descent_optimizer<>{}, f, x0)`.
 *
 * ### Algorithm
 *
 * At iteration \f$k\f$, the optimizer searches from step \f$1\f$ along the
 * scaled steepest-descent direction \f$d_k = -\alpha_k g_k\f$, where
 * \f$g_k = \nabla f(x_k)\f$ and \f$\alpha_k > 0\f$ is the trial scale. Scaling
 * \f$d_k\f$ rather than the initial step keeps \f$g_k^\top d_k\f$ finite when
 * \f$\lVert g_k\rVert_2^2\f$ overflows. The accepted step along \f$-g_k\f$ is
 * \f$t_k = \alpha_k \sigma_k\f$, where \f$\sigma_k\f$ is the step returned by
 * `LineSearch`.
 *
 * The initial scale is \f$\alpha_0 = \min\bigl(1,\, 1 / \lVert g_0 \rVert_\infty\bigr)\f$,
 * and after an accepted step \f$t_k\f$ the next trial scale predicts the
 * previous first-order decrease (Nocedal and Wright, 2006, section 3.5):
 * \f[
 * \alpha_{k+1} = t_k \frac{g_k^\top g_k}{g_{k+1}^\top g_{k+1}},
 * \f]
 * keeping \f$t_k\f$ when that ratio is not finite or not positive.
 *
 * `LineSearch` defaults to armijo_line_search, and also accepts
 * nonmonotone_line_search and wolfe_line_search. With `wolfe_line_search`,
 * the gradient at the accepted point comes from the search; otherwise the
 * optimizer evaluates `value_gradient()` once more at the accepted point.
 *
 * ### Stationarity and confirmation
 *
 * At every iteration \f$k\f$, including \f$k = 0\f$, the stationarity
 * decrement passed to `stop_rule` and stored in `optimizer_report::decrement`
 * is
 * \f[
 * \delta_k = \frac{1}{2} \max(b_k, \gamma_{\max})\, g_k^\top g_k,
 * \f]
 * where \f$b_k\f$ is the last accepted step \f$t_{k-1}\f$ (\f$\alpha_0\f$
 * before any step) and \f$\gamma_{\max}\f$ is the largest
 * \f$s^\top s / s^\top y\f$ over accepted pairs \f$(s, y)\f$ satisfying
 * \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (\f$0\f$ before any
 * pair), with \f$\tau\f$ equal to `curvature_tolerance`.
 *
 * A `stop_verdict::converged` or `stop_verdict::converged_at_resolution`
 * verdict from `stop_rule` must hold at two consecutive iterations; a pending
 * verdict stands if the confirming search returns
 * `line_search_outcome::step_below_rounding`,
 * `line_search_outcome::below_resolution`,
 * `line_search_outcome::not_descent_direction`, or
 * `line_search_outcome::evaluation_budget`, following the same rule as
 * bfgs_optimizer.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration; tolerance \f$10^{-6}\f$ in the objective's units, accepting any rule matching first_order_provides |
 * | `line_search` | `armijo_line_search{}` | Line search along \f$d_k\f$ from step \f$1\f$ (`armijo_line_search`, `nonmonotone_line_search`, or `wolfe_line_search`) |
 * | `max_iterations` | \f$10000\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^6\f$ | Value-evaluation budget checked before each line search |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold \f$\tau\f$ for updating \f$\gamma_{\max}\f$ |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration |
 *
 * ### Statuses
 *
 * - `optimizer_status::converged` and `optimizer_status::converged_at_resolution` when convergence is confirmed.
 * - `optimizer_status::stopped_on_change` immediately on `stop_verdict::stopped_on_change`.
 * - `optimizer_status::stalled` outside confirmation when the search returns `line_search_outcome::step_below_rounding` or `line_search_outcome::below_resolution` (no reset on a stall).
 * - `optimizer_status::line_search_failed` on any other line-search failure or when the gradient at an accepted point is not finite.
 * - `optimizer_status::iteration_budget` when `max_iterations` iterations complete without stopping.
 * - `optimizer_status::evaluation_budget` when `value_evaluations` reaches `max_evaluations` before a search.
 * - `optimizer_status::start_not_admissible` when the objective value or gradient is non-finite at \f$x_0\f$.
 *
 * In optimizer_report, `curvature_evaluations` is always \f$0\f$ and
 * `final_factorization` is `curvature_factorization::not_evaluated`.
 *
 * ### Example
 *
 * \code
 * gradient_descent_optimizer<> opt;
 * opt.max_iterations = 20000;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, first_order_provides>`.
 * \tparam LineSearch Line search type such as `armijo_line_search`, `nonmonotone_line_search`, or `wolfe_line_search`.
 * \see \ref optimizers_first_order, barzilai_borwein_optimizer, adam_optimizer, first_order_provides, minimize(), optimizer_report
 */
// Gradient descent: d = -g, the line search started at the step of
// detail::gradient_descent_steps. minimize(gradient_descent_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = armijo_line_search>
  requires stop_rule_for<Rule, first_order_provides>
struct gradient_descent_optimizer {
  /**
   * \ingroup optimizers_first_order
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_first_order
   * \brief Line search instance used along \f$d_k = -\alpha_k g_k\f$ from step \f$1\f$ (default `armijo_line_search{}`).
   */
  LineSearch line_search{};
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
   * \brief Minimum cosine threshold \f$\tau\f$ for updating \f$\gamma_{\max}\f$ when \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;  // a pair enters gamma_max when s'y > tol |s| |y|
  /**
   * \ingroup optimizers_first_order
   * \brief Optional trace callback invoked once per accepted iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_first_order
   * \brief Minimizes `objective` starting from `x0` using gradient descent.
   *
   * See gradient_descent_optimizer for the step scaling, stationarity confirmation, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_first_order, gradient_descent_optimizer, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    return detail::first_order_run(detail::gradient_descent_steps{}, stop_rule, line_search, max_iterations,
                                   max_evaluations, curvature_tolerance, trace, objective, x0);
  }
};

}  // namespace laplace::optimizers
