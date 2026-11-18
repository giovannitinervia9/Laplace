#pragma once

// The proximal gradient method, accelerated (note/optimizers.md, section 12,
// decisions 2, 4 and 5), on a composite objective f = s + r
// (composite_objective): s smooth with its gradient, r = sum_b r_b with the
// proximal operator prox_{t r}.
//
// One step from a point y with the step t = 1/L:
//
//   z = prox_{t r}(y - t grad s(y)),
//
// accepted when the quadratic model at y dominates s at z,
//
//   s(z) - s(y) - grad s(y)'(z - y) <= |z - y|^2 / (2 t) + c eps max(1, |s(y)|),
//
// and otherwise t is halved (backtracking on the Lipschitz constant L; t never
// grows). A non-finite s(z) is a refused point and halves t too. The slack c eps
// max(1, |s(y)|), c = rounding_slack, keeps the test from firing on the
// rounding of s near the solution, where |z - y| is tiny. That slack is smaller
// than the rounding of a long sum (n = 1e5 squares summed serially: t fell from
// 2.2e-7 to 1.1e-13 in 4000 iterations, and the run crawled), so an excess
// below sqrt(eps) max(1, |s(y)|) is judged on gradients instead (as TFOCS does):
// the step is accepted when (z - y)'(grad s(z) - grad s(y)) <= |z - y|^2 / (2t),
// which for a convex s implies the inequality above. The first t is
// 1/L_0, L_0 = |grad s(x_1) - grad s(x_0)| / |x_1 - x_0| from one probe along
// -grad s(x_0) (a Rayleigh quotient, so at most the largest curvature: the
// backtracking corrects it), unless initial_step is given.
//
// Acceleration: with every rho_b = 0, FISTA (Beck and Teboulle 2009) with the
// momentum theta_{k+1} = (1 + sqrt(1 + 4 theta_k^2)) / 2,
// y = z + (theta_k - 1)/theta_{k+1} (z - x), and the adaptive restart of
// O'Donoghue and Candes (2015): when f(z) > f(x) + c eps max(1, |f(x)|) the
// momentum is reset (y = x, theta = 1) and the step is retried within the same
// iteration. The slack is the lesson of optimizers7: a bare f(z) > f(x) fires on
// the rounding of f near the solution and resets the momentum at nearly every
// iteration (21646 iterations against 782 on one problem). With some rho_b > 0
// (a weakly convex penalty), monotone FISTA (Beck and Teboulle's MFISTA): x
// becomes the better of z and x, y = x + theta_k/theta_{k+1} (z - x) +
// (theta_k - 1)/theta_{k+1} (x - x_previous), and t <= rho_safety / rho with
// rho = max_b rho_b, also inside the backtracking, so that each prox is exact
// (t rho < 1).
//
// Stationarity (decision 6): the proximal gradient mapping G_t(x) =
// (x - prox_{t r}(x - t grad s(x))) / t, read as (1/2) gamma |G_t|^2 in the
// objective's units, given to the stop rule as prox_mapping and reported as the
// decrement, with gamma = max(t, gamma_max) and gamma_max the largest secant
// ratio s's / s'y seen over the points where grad s was evaluated (Giovanni,
// 2026-11-18). With gamma = t = 1/L the measure is smaller than f - f* by up
// to L / mu: on equicorrelated columns (0.9) FISTA declared convergence at 1e-8
// with f - f* = 4.5e-6. gamma_max estimates 1 / mu from below, from the
// directions sampled, as BFGS's gamma_max does (decision 5).
// With momentum the step is taken from y, not x, so
// the measure there is at y; when a rule accepts it, the momentum is reset and
// the step retried from x, and convergence is declared only on a measure read at
// the point reported.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace laplace::optimizers {

namespace detail {

// What the proximal methods give a stop rule. The gradient is not given: grad s
// is not 0 at a stationary point of s + r, so gradient_stop_rule is rejected.
inline constexpr unsigned proximal_gradient_provides = stop_needs_prox_mapping | stop_needs_previous;

inline double proximal_seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

inline optimizer_status proximal_status_from_verdict(stop_verdict v) {
  return v == stop_verdict::converged                 ? optimizer_status::converged
         : v == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                      : optimizer_status::stopped_on_change;
}

// the largest rho_b of a composite objective
template <composite_objective O>
double largest_weak_convexity(O& objective) {
  double rho = 0.0;
  for (int b = 0; b < objective.nonsmooth_block_count(); ++b) rho = std::max(rho, objective.weak_convexity_constant(b));
  return rho;
}

}  // namespace detail

/**
 * \ingroup optimizers_proximal
 * \brief Accelerated proximal gradient optimizer (FISTA with adaptive restart, or monotone FISTA for weakly convex penalties).
 *
 * Minimizes \f$f(x) = s(x) + r(x)\f$ over a `composite_objective` on the host. Each step computes
 * \f$z = \operatorname{prox}_{t r}(y - t \nabla s(y))\f$ from momentum point \f$y\f$ with step \f$t > 0\f$
 * (initialized from `initial_step` if positive, or from one gradient secant along \f$-\nabla s(x_0)\f$).
 * The step is accepted when \f$s(z) - s(y) - \nabla s(y)^\top(z - y) \le \lVert z - y\rVert^2 / (2t) + c\,\varepsilon \max(1, |s(y)|)\f$
 * with \f$c =\f$ `rounding_slack` (default \f$8\f$), or when the excess is within \f$\sqrt{\varepsilon}\max(1, |s(y)|)\f$
 * and \f$(z - y)^\top(\nabla s(z) - \nabla s(y)) \le \lVert z - y\rVert^2 / (2t)\f$ (Becker, Candès and Grant, 2011).
 * Otherwise \f$t\f$ is multiplied by `shrink` (default \f$0.5\f$), and \f$t\f$ never grows; non-finite \f$s(z)\f$ also shrinks \f$t\f$.
 *
 * When every weak-convexity constant \f$\rho_b = 0\f$, it uses FISTA momentum
 * \f$\omega_{k+1} = (1 + \sqrt{1 + 4\omega_k^2}) / 2\f$ and \f$y = z + ((\omega_k - 1)/\omega_{k+1})(z - x_k)\f$
 * (Beck and Teboulle, 2009a), with adaptive restart resetting \f$y = x_k\f$, \f$\omega = 1\f$ when
 * \f$f(z) > f(x_k) + c\,\varepsilon \max(1, |f(x_k)|)\f$ (O'Donoghue and Candès, 2015).
 * When some \f$\rho_b > 0\f$, it runs monotone FISTA (Beck and Teboulle, 2009b) with \f$x_{k+1}\f$ set to the better
 * of \f$z\f$ and \f$x_k\f$ and \f$t \le\f$ `rho_safety` \f$/ \max_b \rho_b\f$ (default \f$0.99\f$) even inside backtracking.
 *
 * Stationarity is \f$\frac{1}{2}\gamma\lVert G_t(x)\rVert^2\f$ where
 * \f$G_t(x) = (x - \operatorname{prox}_{t r}(x - t \nabla s(x))) / t\f$ and \f$\gamma = \max(t, \gamma_{\max})\f$,
 * with \f$\gamma_{\max}\f$ the largest secant ratio \f$\delta y^\top \delta y / (\delta y^\top \delta g)\f$ seen
 * (Barzilai and Borwein, 1988). The measure passes to `stop_rule` as `prox_mapping` and is stored in
 * `optimizer_report::decrement`; when accepted at \f$y \ne x_k\f$, momentum resets and the step is retried from \f$x_k\f$.
 * Possible `optimizer_report::status` values are `converged`, `converged_at_resolution`, `stopped_on_change`,
 * `stalled` (a step from \f$x_k\f$ does not move it), `line_search_failed` (more than `max_backtracks` shrinkings),
 * `iteration_budget`, `evaluation_budget`, and `start_not_admissible`. In the returned `optimizer_report`,
 * `gradient` is \f$\nabla s\f$ at `point`. See \ref optimizers_proximal for measured accuracy.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_prox_mapping` and `stop_needs_previous` (`gradient_stop_rule` is rejected at compile time).
 * \see \ref optimizers_proximal, prox_mapping_stop_rule, penalized_objective, proximal_newton_optimizer
 */
template <class StopRule = prox_mapping_stop_rule>
  requires stop_rule_for<StopRule, detail::proximal_gradient_provides>
struct proximal_gradient_optimizer {
  /// \brief Stopping rule (default `prox_mapping_stop_rule{}`).
  StopRule stop_rule{};
  /// \brief Maximum iterations (default \f$100000\f$).
  int max_iterations = 100000;
  /// \brief Maximum evaluations of \f$s\f$ (default \f$10000000\f$).
  long max_value_evaluations = 10000000;
  /// \brief Initial proximal step \f$t\f$; \f$0\f$ estimates it from one gradient secant along \f$-\nabla s(x_0)\f$ (default \f$0\f$).
  double initial_step = 0.0;
  /// \brief Backtracking multiplier applied to \f$t\f$ (default \f$0.5\f$).
  double shrink = 0.5;
  /// \brief Rounding allowance \f$c\f$ in units of \f$\varepsilon\max(1, |f|)\f$ in backtracking and restart tests (default \f$8\f$).
  double rounding_slack = 8.0;
  /// \brief Upper bound on \f$t \max_b \rho_b\f$ when any penalty is weakly convex (default \f$0.99\f$).
  double rho_safety = 0.99;
  /// \brief Maximum step shrinkings per iteration before failing with `line_search_failed` (default \f$100\f$).
  int max_backtracks = 100;
  /// \brief Optional per-iteration trace callback (default empty).
  optimizer_trace trace;

  /**
   * \brief Runs accelerated proximal gradient from `x0`.
   *
   * \tparam O Composite objective type satisfying `composite_objective`.
   * \param objective Composite objective \f$f = s + r\f$.
   * \param x0 Starting point \f$x_0\f$; \f$f(x_0)\f$ must be finite.
   * \return An `optimizer_report` with `decrement` equal to \f$\frac{1}{2}\gamma\lVert G_t\rVert^2\f$ and `gradient` equal to \f$\nabla s\f$ at `point`.
   *
   * Measured accuracy is documented on \ref optimizers_proximal.
   */
  template <composite_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    constexpr double eps = std::numeric_limits<double>::epsilon();
    optimizer_report report;
    optimizer_vector x("x", n), x_previous("x_previous", n), y("y", n), gy("gy", n), v("v", n), z("z", n),
        x_new("x_new", n), y_seen("y_seen", n), gy_seen("gy_seen", n), gz("gz", n);
    Kokkos::deep_copy(x, x0);
    Kokkos::deep_copy(y, x0);

    const double rho = detail::largest_weak_convexity(objective);
    const bool monotone = rho > 0.0;
    const double t_max = monotone ? rho_safety / rho : std::numeric_limits<double>::infinity();

    // gamma_max, the largest secant ratio s's / s'y over the pairs of points where
    // the gradient was evaluated (decision 1 of the G1 round): a long
    // Barzilai-Borwein step, an estimate from below of 1 / mu
    double gamma_max = 0.0;
    bool seen = false;
    auto record_secant = [&]() {
      if (seen) {
        double ss = 0.0, sy = 0.0;
        for (int i = 0; i < n; ++i) {
          const double si = y(i) - y_seen(i);
          ss += si * si, sy += si * (gy(i) - gy_seen(i));
        }
        if (sy > 0.0 && std::isfinite(ss / sy)) gamma_max = std::max(gamma_max, ss / sy);
      }
      Kokkos::deep_copy(y_seen, y);
      Kokkos::deep_copy(gy_seen, gy);
      seen = true;
    };
    double s_y = objective.value_gradient(y, gy);
    ++report.value_evaluations, ++report.gradient_evaluations;
    record_secant();
    double f_x = s_y + objective.nonsmooth_value(x);
    auto finish = [&](optimizer_status status, double measure, bool gradient_at_x) {
      report.status = status;
      report.point = x, report.value = f_x, report.decrement = measure;
      if (!gradient_at_x) {
        objective.value_gradient(x, gy);
        ++report.value_evaluations, ++report.gradient_evaluations;
      }
      report.gradient = gy;
      report.elapsed_seconds = detail::proximal_seconds_since(start);
      return report;
    };
    if (!std::isfinite(f_x)) return finish(optimizer_status::start_not_admissible, std::numeric_limits<double>::quiet_NaN(), true);

    // the first step: 1/L_0 from a secant along -grad s(x_0)
    double t = initial_step;
    if (!(t > 0.0)) {
      t = 1.0;
      const double gmax = detail::max_abs(gy);
      if (gmax > 0.0 && std::isfinite(gmax)) {
        const double h = 1e-4 * std::max(1.0, detail::max_abs(x)) / gmax;
        for (int i = 0; i < n; ++i) v(i) = x(i) - h * gy(i);
        const double sp = objective.value_gradient(v, z);
        ++report.value_evaluations, ++report.gradient_evaluations;
        double dg = 0.0, dx = 0.0;
        for (int i = 0; i < n; ++i) dg += (z(i) - gy(i)) * (z(i) - gy(i)), dx += (v(i) - x(i)) * (v(i) - x(i));
        if (std::isfinite(sp) && dg > 0.0 && dx > 0.0) t = std::sqrt(dx / dg);
        double sy = 0.0;
        for (int i = 0; i < n; ++i) sy += (v(i) - x(i)) * (z(i) - gy(i));
        if (std::isfinite(sp) && sy > 0.0) gamma_max = std::max(gamma_max, dx / sy);
      }
    }
    t = std::min(t, t_max);

    double theta = 1.0, f_previous = f_x, last_measure = std::numeric_limits<double>::quiet_NaN();
    bool y_is_x = true;  // no momentum: the step is taken from x itself

    for (int k = 0;;) {
      // backtracking at y
      double s_z = 0.0;
      int backtracks = 0;
      for (;; ++backtracks) {
        if (backtracks > max_backtracks) return finish(optimizer_status::line_search_failed, last_measure, false);
        for (int i = 0; i < n; ++i) v(i) = y(i) - t * gy(i);
        objective.nonsmooth_prox(v, t, z);
        s_z = objective.value(z);
        ++report.value_evaluations;
        double slope = 0.0, step2 = 0.0;
        for (int i = 0; i < n; ++i) {
          const double d = z(i) - y(i);
          slope += gy(i) * d, step2 += d * d;
        }
        const double excess = s_z - s_y - slope - step2 / (2.0 * t);
        if (std::isfinite(s_z) && excess <= rounding_slack * eps * std::max(1.0, std::abs(s_y))) break;
        // An excess within sqrt(eps) |s| may be the rounding of s itself: decide on
        // gradients, (z - y)'(grad s(z) - grad s(y)) <= |z - y|^2 / (2t), which for a
        // convex s implies the test above and has no rounding at the scale of |s|
        if (std::isfinite(s_z) && excess <= std::sqrt(eps) * std::max(1.0, std::abs(s_y))) {
          objective.value_gradient(z, gz);
          ++report.value_evaluations, ++report.gradient_evaluations;
          double curvature = 0.0;
          for (int i = 0; i < n; ++i) curvature += (z(i) - y(i)) * (gz(i) - gy(i));
          if (curvature <= step2 / (2.0 * t)) break;
        }
        t *= shrink;
      }
      const double f_z = s_z + objective.nonsmooth_value(z);
      double measure = 0.0;
      bool moved = false;
      for (int i = 0; i < n; ++i) {
        measure += (z(i) - y(i)) * (z(i) - y(i));
        moved = moved || z(i) != y(i);
      }
      // (1/2) max(t, gamma_max) |G_t|^2: with t = 1/L alone the measure is smaller
      // than f - f* by up to L / mu (measured 450 on equicorrelated columns)
      measure *= std::max(t, gamma_max) / (2.0 * t * t);

      // the stop rule, with the measure at y; convergence only when y is x
      stop_rule_input in;
      in.iteration = k, in.value = f_x, in.point = x, in.prox_mapping = measure;
      in.has_previous = k > 0, in.previous_value = f_previous, in.previous_point = x_previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict == stop_verdict::stopped_on_change) return finish(optimizer_status::stopped_on_change, measure, y_is_x);
      if (verdict != stop_verdict::none) {
        if (y_is_x) return finish(detail::proximal_status_from_verdict(verdict), measure, true);
        // verify at x: reset the momentum and retry the step from x
        Kokkos::deep_copy(y, x);
        theta = 1.0, y_is_x = true;
        s_y = objective.value_gradient(y, gy);
        ++report.value_evaluations, ++report.gradient_evaluations;
        record_secant();
        continue;
      }
      if (y_is_x && !moved) return finish(optimizer_status::stalled, measure, true);
      if (k >= max_iterations) return finish(optimizer_status::iteration_budget, measure, y_is_x);
      if (report.value_evaluations >= max_value_evaluations)
        return finish(optimizer_status::evaluation_budget, measure, y_is_x);

      // adaptive restart (convex case): an increase beyond rounding resets the momentum
      if (!monotone && !y_is_x && f_z - f_x > rounding_slack * eps * std::max(1.0, std::abs(f_x))) {
        Kokkos::deep_copy(y, x);
        theta = 1.0, y_is_x = true;
        s_y = objective.value_gradient(y, gy);
        ++report.value_evaluations, ++report.gradient_evaluations;
        record_secant();
        continue;
      }

      // momentum
      const double theta_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * theta * theta));
      Kokkos::deep_copy(x_previous, x);
      f_previous = f_x;
      if (!monotone) {
        for (int i = 0; i < n; ++i) y(i) = z(i) + (theta - 1.0) / theta_next * (z(i) - x(i));
        Kokkos::deep_copy(x, z);
        f_x = f_z;
      } else {
        const bool take_z = f_z <= f_x;
        for (int i = 0; i < n; ++i) x_new(i) = take_z ? z(i) : x(i);
        for (int i = 0; i < n; ++i)
          y(i) = x_new(i) + theta / theta_next * (z(i) - x_new(i)) + (theta - 1.0) / theta_next * (x_new(i) - x(i));
        Kokkos::deep_copy(x, x_new);
        if (take_z) f_x = f_z;
      }
      theta = theta_next;
      y_is_x = true;
      for (int i = 0; i < n && y_is_x; ++i) y_is_x = y(i) == x(i);
      s_y = objective.value_gradient(y, gy);
      ++report.value_evaluations, ++report.gradient_evaluations;
      record_secant();
      last_measure = measure;
      report.iterations = ++k;
      if (trace) trace(optimizer_trace_point{k, x, f_x, {}, measure, t});
    }
  }
};

}  // namespace laplace::optimizers
