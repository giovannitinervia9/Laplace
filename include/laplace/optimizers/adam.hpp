#pragma once

// Adam (Kingma and Ba 2015), deterministic, on the full gradient
// (note/optimizers.md, section 13, point 1). The minibatches come with the
// engine of phase 7.
//
// - At x_k with gradient g_k: m_k = beta1 m_{k-1} + (1 - beta1) g_k,
//   v_k = beta2 v_{k-1} + (1 - beta2) g_k^2 (elementwise), m_0 = v_0 = 0;
//   with bias correction mh = m_k / (1 - beta1^k), vh = v_k / (1 - beta2^k);
//   the step is u = -B mh with B = diag(lr / (sqrt(vh) + epsilon)).
// - No line search: the trial point x + u is accepted whenever f and g are
//   finite there, whether or not f decreased. A non-finite value or gradient
//   is a refused point (decision 3): the step is halved and tried again, up
//   to max_refusals times for this step (the moments are not updated again,
//   and the next step starts from the full u); then the run ends
//   line_search_failed. A trial point equal to x ends it stalled.
// - Stationarity (decision 5): (1/2) max(g_k'B g_k, gamma_max g_k'g_k), with B
//   the diagonal of the step about to be taken and gamma_max the largest
//   s'y / y'y over the steps with s'y > curvature_tolerance |s| |y|. A
//   convergence verdict must hold at two consecutive points; a pending verdict
//   stands when the confirming step cannot move x.

#include "laplace/optimizers/gradient_descent.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_first_order
 * \brief Deterministic Adam optimizer on the full gradient.
 *
 * `adam_optimizer<Rule>` minimizes an unconstrained objective modeling
 * gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is the parameter
 * dimension, on the host only; nothing is marked `LAPLACE_HD`. A run starts
 * with `minimize(adam_optimizer<>{}, f, x0)`.
 *
 * ### Algorithm
 *
 * Starting from \f$m_0 = v_0 = 0\f$ in \f$\mathbb{R}^n\f$, at iterate
 * \f$x_k\f$ with gradient \f$g_k = \nabla f(x_k)\f$ Adam updates the first and
 * second moment vectors elementwise (Kingma and Ba, 2015):
 * \f[
 * m_k = \beta_1 m_{k-1} + (1 - \beta_1) g_k, \qquad
 * v_k = \beta_2 v_{k-1} + (1 - \beta_2) g_k^2.
 * \f]
 * When `bias_correction` is `true`, it forms
 * \f$\hat m = m_k / (1 - \beta_1^k)\f$ and \f$\hat v = v_k / (1 - \beta_2^k)\f$
 * (otherwise \f$\hat m = m_k\f$ and \f$\hat v = v_k\f$), and proposes the step
 * \f$u = -B \hat m\f$ with diagonal scaling
 * \f$B = \operatorname{diag}\bigl(\ell / (\sqrt{\hat v} + \epsilon)\bigr)\f$,
 * where \f$\ell\f$ is `learning_rate`.
 *
 * There is no line search: \f$x_k + u\f$ is accepted whenever both \f$f\f$ and
 * \f$g\f$ are finite there, whether or not \f$f\f$ decreased. A non-finite
 * value or gradient marks a refused point (never clamped): the step is halved
 * and retried up to `max_refusals` times for that step without updating the
 * moments again; if every attempt is refused, the run ends with
 * `optimizer_status::line_search_failed`. A trial point equal to \f$x_k\f$ in
 * every coordinate ends the run with `optimizer_status::stalled`, or confirms
 * a pending convergence verdict.
 *
 * ### Stationarity and confirmation
 *
 * At every iteration, the stationarity decrement is
 * \f[
 * \delta = \frac{1}{2} \max\bigl(g^\top B g,\ \gamma_{\max}\, g^\top g\bigr),
 * \f]
 * where \f$B\f$ is the diagonal of the step about to be taken and
 * \f$\gamma_{\max}\f$ is the largest \f$s^\top y / y^\top y\f$ over accepted
 * steps \f$(s, y)\f$ satisfying
 * \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (\f$0\f$ before any
 * pair), with \f$\tau\f$ equal to `curvature_tolerance`. When \f$g^\top g\f$
 * overflows, \f$\delta\f$ is set to \f$+\infty\f$ so that overflowing
 * \f$g^2\f$ (giving \f$B = 0\f$) cannot report a zero decrement. Every
 * converged run measured satisfies \f$f - f^* \le 1.52\,\delta\f$.
 *
 * A `stop_verdict::converged` or `stop_verdict::converged_at_resolution`
 * verdict from `stop_rule` must hold at two consecutive iterations; a pending
 * verdict stands if the confirming trial point does not move \f$x\f$,
 * following the same rule as bfgs_optimizer.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration; tolerance \f$10^{-6}\f$ in the objective's units, accepting any rule matching first_order_provides |
 * | `learning_rate` | \f$10^{-3}\f$ | Step scale \f$\ell\f$ in \f$B = \operatorname{diag}(\ell / (\sqrt{\hat v} + \epsilon))\f$ |
 * | `beta1` | \f$0.9\f$ | Exponential decay rate \f$\beta_1\f$ for the first moment |
 * | `beta2` | \f$0.999\f$ | Exponential decay rate \f$\beta_2\f$ for the second moment |
 * | `epsilon` | \f$10^{-8}\f$ | Additive denominator term \f$\epsilon\f$ stabilizing \f$\sqrt{\hat v} + \epsilon\f$ |
 * | `bias_correction` | `true` | Whether to divide \f$m_k\f$ and \f$v_k\f$ by \f$1 - \beta_1^k\f$ and \f$1 - \beta_2^k\f$ |
 * | `max_iterations` | \f$10^5\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^6\f$ | Value-evaluation budget checked before each step |
 * | `max_refusals` | \f$60\f$ | Maximum number of step halvings on non-finite trial points before returning `optimizer_status::line_search_failed` |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold \f$\tau\f$ for updating \f$\gamma_{\max}\f$ |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration |
 *
 * ### Statuses
 *
 * - `optimizer_status::converged` and `optimizer_status::converged_at_resolution` when the stopping rule's convergence verdict holds at two consecutive iterations (or is pending when the confirming step does not move \f$x\f$).
 * - `optimizer_status::stopped_on_change` immediately on `stop_verdict::stopped_on_change`.
 * - `optimizer_status::stalled` when a trial point equals \f$x\f$ outside confirmation.
 * - `optimizer_status::line_search_failed` after `max_refusals` halvings of a step all hit non-finite trial points.
 * - `optimizer_status::iteration_budget` when `max_iterations` iterations complete without stopping.
 * - `optimizer_status::evaluation_budget` when `value_evaluations` reaches `max_evaluations` before a step.
 * - `optimizer_status::start_not_admissible` when the objective value or gradient is non-finite at \f$x_0\f$.
 *
 * In optimizer_report, `curvature_evaluations` is always \f$0\f$ and
 * `final_factorization` is `curvature_factorization::not_evaluated`.
 *
 * ### Example
 *
 * \code
 * adam_optimizer<> opt;
 * opt.learning_rate = 1e-2;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, first_order_provides>`.
 * \see \ref optimizers_first_order, gradient_descent_optimizer, barzilai_borwein_optimizer, first_order_provides, minimize(), optimizer_report
 */
// Adam on the full gradient. minimize(adam_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule>
  requires stop_rule_for<Rule, first_order_provides>
struct adam_optimizer {
  /**
   * \ingroup optimizers_first_order
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_first_order
   * \brief Step scale \f$\ell\f$ (default \f$10^{-3}\f$).
   */
  double learning_rate = 1e-3;
  /**
   * \ingroup optimizers_first_order
   * \brief Exponential decay rate \f$\beta_1\f$ for the first moment (default \f$0.9\f$).
   */
  double beta1 = 0.9;
  /**
   * \ingroup optimizers_first_order
   * \brief Exponential decay rate \f$\beta_2\f$ for the second moment (default \f$0.999\f$).
   */
  double beta2 = 0.999;
  /**
   * \ingroup optimizers_first_order
   * \brief Additive denominator stabilizer \f$\epsilon\f$ (default \f$10^{-8}\f$).
   */
  double epsilon = 1e-8;
  /**
   * \ingroup optimizers_first_order
   * \brief Whether to divide \f$m_k\f$ and \f$v_k\f$ by \f$1 - \beta_1^k\f$ and \f$1 - \beta_2^k\f$ (default `true`).
   */
  bool bias_correction = true;
  /**
   * \ingroup optimizers_first_order
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default \f$10^5\f$).
   */
  int max_iterations = 100000;
  /**
   * \ingroup optimizers_first_order
   * \brief Value-evaluation budget checked before each step (default \f$10^6\f$).
   */
  long max_evaluations = 1000000;
  /**
   * \ingroup optimizers_first_order
   * \brief Maximum number of step halvings on non-finite trial points before `optimizer_status::line_search_failed` (default \f$60\f$).
   */
  int max_refusals = 60;               // halvings of one step on non-finite trial points
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
   * \brief Minimizes `objective` starting from `x0` using deterministic Adam.
   *
   * See adam_optimizer for the moment updates, step halving, stationarity confirmation, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_first_order, adam_optimizer, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    optimizer_report report;
    optimizer_vector x("x", n), g("g", n), x_trial("x_trial", n), g_trial("g_trial", n), x_previous("x_previous", n);
    std::vector<double> m(n, 0.0), v(n, 0.0), u(n);
    Kokkos::deep_copy(x, x0);
    double f = objective.value_gradient(x, g);
    ++report.value_evaluations, ++report.gradient_evaluations;
    auto finish = [&](optimizer_status status) {
      report.status = status;
      report.point = x, report.value = f, report.gradient = g;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    if (!std::isfinite(f) || !std::isfinite(detail::max_abs(g))) return finish(optimizer_status::start_not_admissible);
    auto verdict_status = [](stop_verdict s) {
      return s == stop_verdict::converged                 ? optimizer_status::converged
             : s == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                          : optimizer_status::stopped_on_change;
    };
    stop_verdict pending = stop_verdict::none;
    double gamma_max = 0.0, f_previous = f, power1 = 1.0, power2 = 1.0;
    for (int k = 0;; ++k) {
      // the moments with g_k, the step u and its decrement
      power1 *= beta1, power2 *= beta2;
      const double c1 = bias_correction ? 1.0 - power1 : 1.0, c2 = bias_correction ? 1.0 - power2 : 1.0;
      double gbg = 0.0, gg = 0.0;
      for (int i = 0; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g(i);
        v[i] = beta2 * v[i] + (1.0 - beta2) * g(i) * g(i);
        const double b = learning_rate / (std::sqrt(v[i] / c2) + epsilon);
        u[i] = -b * m[i] / c1;
        gbg += b * g(i) * g(i), gg += g(i) * g(i);
      }
      // g_i^2 beyond the largest double makes v infinite and B zero there: the
      // measure is then unknown, never 0 (at cosh(700) it read 0 and the run
      // ended "converged" at the start)
      report.decrement = std::isfinite(gg) ? 0.5 * std::max(gbg, gamma_max * gg)
                                           : std::numeric_limits<double>::infinity();
      if (k > 0 && trace) trace({k, x, f, g, report.decrement, 1.0});
      stop_rule_input in;
      in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
      if (k > 0) in.has_previous = true, in.previous_value = f_previous, in.previous_point = x_previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict == stop_verdict::converged || verdict == stop_verdict::converged_at_resolution) {
        if (pending != stop_verdict::none) return finish(verdict_status(verdict));
        pending = verdict;
      } else {
        pending = stop_verdict::none;
        if (verdict != stop_verdict::none) return finish(verdict_status(verdict));
      }
      if (k == max_iterations) return finish(optimizer_status::iteration_budget);
      if (report.value_evaluations >= max_evaluations) return finish(optimizer_status::evaluation_budget);

      // the trial point, halved while refused
      double t = 1.0, f_new = 0.0;
      for (int refusals = 0;; ++refusals) {
        bool moved = false;
        for (int i = 0; i < n; ++i) {
          x_trial(i) = x(i) + t * u[i];
          moved = moved || x_trial(i) != x(i);
        }
        if (!moved) return finish(pending != stop_verdict::none ? verdict_status(pending) : optimizer_status::stalled);
        f_new = objective.value_gradient(optimizer_point(x_trial), g_trial);
        ++report.value_evaluations, ++report.gradient_evaluations;
        if (std::isfinite(f_new) && std::isfinite(detail::max_abs(g_trial))) break;
        if (refusals == max_refusals) return finish(optimizer_status::line_search_failed);
        t *= 0.5;
      }
      double sy = 0.0, ss = 0.0, yy = 0.0;
      for (int i = 0; i < n; ++i) {
        const double si = x_trial(i) - x(i), yi = g_trial(i) - g(i);
        sy += si * yi, ss += si * si, yy += yi * yi;
      }
      if (std::isfinite(sy) && sy > curvature_tolerance * std::sqrt(ss) * std::sqrt(yy))
        gamma_max = std::max(gamma_max, sy / yy);
      Kokkos::deep_copy(x_previous, x);
      Kokkos::deep_copy(x, x_trial);
      Kokkos::deep_copy(g, g_trial);
      f_previous = f;
      f = f_new;
      report.iterations = k + 1;
    }
  }
};

}  // namespace laplace::optimizers
