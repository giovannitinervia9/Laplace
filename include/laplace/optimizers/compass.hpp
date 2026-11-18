#pragma once

// Compass search, a pattern search on a poll (note/optimizers.md, decisions
// 4, 5 and 12.2): value only (C0), for a function of n >= 1 variables,
// smooth or not.
//
// A poll evaluates x +- delta d_j over the n directions d_j of an orthonormal
// basis (a positive spanning set of 2n directions): the coordinate basis, or
// a fresh random one at every poll, the Householder reflection
// I - 2 w w'/w'w of a w drawn uniformly in [-1, 1]^n from a 64-bit LCG
// seeded with `seed` (the same polls on every platform and compiler). With
// a fresh basis per poll the directions used over a run are dense in the
// sphere, which the convergence theory for a merely Lipschitz f needs (the
// idea of MADS, Audet and Dennis 2006); a fixed basis can stop at a kink
// whose descent directions lie between the axes.
//
// On a success (a trial lower than f(x); with first_improvement the poll stops
// at the first one) x moves there and delta is multiplied by `expansion`,
// capped at max_step_multiple times its start; on a failure delta is
// multiplied by `contraction`. A non-finite value is +infinity, a refused
// point (decision 3). The start delta is initial_step max(1, max_i |x0_i|).
//
// The stationarity measure is the step of the poll that just failed: no
// decrease was found at that distance in any of the 2n directions. It is
// given to the stop rule as search_size after each failed poll (NaN after a
// successful one, since x has moved). A failed poll whose trial points all
// equal x in floating point ends the run as converged at resolution.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/nelder_mead.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_derivative_free
 * \brief Orthonormal basis used by `compass_optimizer` to form the \f$2n\f$ poll directions.
 *
 * \see \ref optimizers_derivative_free, compass_optimizer
 */
// The directions of a compass poll.
enum class compass_poll_basis {
  coordinate,         // +- e_j: deterministic, the theory needs f continuously differentiable
  ///< Deterministic coordinate directions \f$\pm e_j\f$; convergence theory requires continuously differentiable \f$f\f$.
  random_orthonormal  // +- the columns of a random Householder reflection, a new one per poll
  ///< Columns \f$\pm d_j\f$ of a fresh random Householder reflection at every poll.
};

/**
 * \ingroup optimizers_derivative_free
 * \brief Derivative-free minimization in several variables by compass search.
 *
 * Minimizes an objective modeling `value_objective` on the host from a starting
 * point `x0` of length \f$n \ge 1\f$ (`std::invalid_argument` is thrown if
 * `x0` has length \f$0\f$), using objective values only. Any non-finite
 * objective value counts as \f$+\infty\f$. Runs are invoked through minimize()
 * or `run()`. The template parameter `StopRule` defaults to
 * `search_size_stop_rule`, and `stop_rule_for` admits any stopping rule
 * requiring at most `stop_needs_search_size` and `stop_needs_previous`.
 *
 * ### Polling and step update
 *
 * Starting from step \f$\delta_0\f$ equal to `initial_step` times
 * \f$\max(1, \max_i |x_{0,i}|)\f$, each poll evaluates trial points
 * \f$x \pm \delta d_j\f$ along the \f$n\f$ columns \f$d_j\f$ of an orthonormal
 * basis (Kolda, Lewis and Torczon, 2003), skipping any trial point equal to
 * \f$x\f$ in floating point. With `poll_basis` set to
 * `compass_poll_basis::random_orthonormal`, each poll draws a fresh
 * Householder reflection \f$I - 2 w w^\top / (w^\top w)\f$ from a vector
 * \f$w\f$ sampled uniformly in \f$[-1, 1]^n\f$ by a 64-bit linear congruential
 * generator seeded with `seed` (yielding identical polls across platforms and
 * compilers), so the directions become dense on the sphere over a run as the
 * convergence theory for Lipschitz functions requires (Audet and Dennis,
 * 2006), though the basis is not uniformly distributed over the orthogonal
 * group. With `compass_poll_basis::coordinate`, the directions are \f$\pm e_j\f$.
 *
 * When `first_improvement` is `true`, the poll stops at the first trial point
 * with a value strictly below \f$f(x)\f$; otherwise it evaluates all \f$2n\f$
 * directions and moves to the lowest. After a successful poll, \f$x\f$ moves
 * and \f$\delta\f$ is updated to
 * \f$\min(\alpha_+ \delta,\, M \delta_0)\f$, where \f$\alpha_+\f$ is
 * `expansion` and \f$M\f$ is `max_step_multiple`. After a failed poll, \f$x\f$
 * stays fixed and \f$\delta\f$ is multiplied by \f$\alpha_-\f$ (`contraction`).
 *
 * ### Stationarity
 *
 * After a failed poll, `search_size` is set to the step \f$\delta\f$ of that
 * poll in the units of \f$x\f$; after a successful poll, `search_size` is quiet
 * NaN because \f$x\f$ has moved. A failed poll whose trial points all equal
 * \f$x\f$ in floating point ends the run with
 * `optimizer_status::converged_at_resolution`. Otherwise `stop_rule` is
 * queried after every poll. A convergence verdict from `stop_rule` (`converged`
 * or `converged_at_resolution`) stands only when the spread
 * \f$\max |f(x \pm \delta d_j) - f(x)|\f$ over the evaluated trial points of the
 * final failed poll (or \f$0\f$ when every trial point rounds to \f$x\f$) is at
 * most `value_spread_tolerance` \f$+ 64\,\varepsilon\,|f(x)|\f$, where
 * \f$\varepsilon\f$ is machine epsilon; otherwise the run ends with
 * `optimizer_status::stalled`. Because \f$x\f$ does not move on a failed poll, a
 * rule testing point change (such as `parameter_change_stop_rule`) sees zero
 * change there and ends the run with `optimizer_status::stopped_on_change`.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `search_size_stop_rule{}` | Stopping rule queried on `search_size` and `previous_*` after every poll |
 * | `poll_basis` | `compass_poll_basis::random_orthonormal` | Basis used to form the \f$2n\f$ poll directions |
 * | `seed` | \f$1\f$ | Seed of the 64-bit linear congruential generator for random bases |
 * | `first_improvement` | `true` | Stops a poll at the first lower value instead of evaluating all \f$2n\f$ points |
 * | `initial_step` | \f$0.1\f$ | Multiplier for initial poll step \f$\delta_0\f$, set to `initial_step` times \f$\max(1, \max_i |x_{0,i}|)\f$ |
 * | `expansion` | \f$2\f$ | Step multiplier after a successful poll |
 * | `contraction` | \f$0.5\f$ | Step multiplier after a failed poll |
 * | `max_step_multiple` | \f$10^4\f$ | Upper bound on \f$\delta / \delta_0\f$ |
 * | `max_iterations` | \f$10000\f$ | Maximum number of polls |
 * | `max_value_evaluations` | \f$10^5\f$ | Budget on objective evaluations, checked before each poll |
 * | `value_spread_tolerance` | \f$10^{-6}\f$ | Maximum spread over the final failed poll (beyond \f$64\,\varepsilon\,|f(x)|\f$) for a convergence verdict; otherwise ends as `stalled` |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked after every poll |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from
 * \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(x_0)\f$ is not finite (`search_size` is quiet NaN).
 * - `optimizer_status::converged`, `optimizer_status::converged_at_resolution`, or `optimizer_status::stopped_on_change` when returned by `stop_rule` (convergence verdicts are subject to the spread check).
 * - `optimizer_status::converged_at_resolution` when all trial points of a failed poll equal \f$x\f$ in floating point.
 * - `optimizer_status::stalled` when a convergence verdict from `stop_rule` has \f$\max |f(x \pm \delta d_j) - f(x)|\f$ over the evaluated trial points of the final failed poll above `value_spread_tolerance` \f$+ 64\,\varepsilon\,|f(x)|\f$.
 * - `optimizer_status::iteration_budget` after `max_iterations` polls.
 * - `optimizer_status::evaluation_budget` when `max_value_evaluations` is reached before a poll (evaluations may exceed the budget by one poll).
 *
 * In `optimizer_report`, `point` and `value` hold the best point found,
 * `search_size` holds the step of the final poll when it failed (quiet NaN
 * when it succeeded, or on an inadmissible start), `iterations` counts polls,
 * `value_evaluations` counts objective calls, `gradient` is empty,
 * `gradient_evaluations` and `curvature_evaluations` are \f$0\f$, and
 * `decrement` is quiet NaN. The optional `trace` callback receives the best
 * point, its value, and as `step` the step of a successful poll or \f$0\f$
 * after a failed one.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_search_size` and `stop_needs_previous`; `search_size_stop_rule` by default.
 * \see \ref optimizers_derivative_free, \ref optimizers, compass_poll_basis, nelder_mead_optimizer, minimize(), search_size_stop_rule
 */
template <class StopRule = search_size_stop_rule>
  requires stop_rule_for<StopRule, detail::derivative_free_provides>
struct compass_optimizer {
  /**
   * \ingroup optimizers_derivative_free
   * \brief Stopping rule instance queried after every poll (default `search_size_stop_rule{}`).
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_derivative_free
   * \brief Orthonormal basis choice for poll directions (default `compass_poll_basis::random_orthonormal`).
   */
  compass_poll_basis poll_basis = compass_poll_basis::random_orthonormal;
  /**
   * \ingroup optimizers_derivative_free
   * \brief Seed for the 64-bit linear congruential generator used by `random_orthonormal` (default \f$1\f$).
   */
  std::uint64_t seed = 1;          // of the random bases
  /**
   * \ingroup optimizers_derivative_free
   * \brief Whether to stop a poll at its first improving trial point instead of evaluating all \f$2n\f$ points (default `true`).
   */
  bool first_improvement = true;   // stop a poll at its first success
  /**
   * \ingroup optimizers_derivative_free
   * \brief Multiplier on \f$\max(1, \max_i |x_{0,i}|)\f$ setting the initial poll step \f$\delta_0\f$ (default \f$0.1\f$).
   */
  double initial_step = 0.1;       // delta_0 = initial_step max(1, max_i |x0_i|)
  /**
   * \ingroup optimizers_derivative_free
   * \brief Step multiplier applied after a successful poll (default \f$2\f$).
   */
  double expansion = 2.0;          // delta after a success
  /**
   * \ingroup optimizers_derivative_free
   * \brief Step multiplier applied after a failed poll (default \f$0.5\f$).
   */
  double contraction = 0.5;        // delta after a failure
  /**
   * \ingroup optimizers_derivative_free
   * \brief Maximum poll step expressed as a multiple of \f$\delta_0\f$ (default \f$10^4\f$).
   */
  double max_step_multiple = 1e4;  // delta <= max_step_multiple delta_0
  /**
   * \ingroup optimizers_derivative_free
   * \brief Maximum number of polls (default \f$10000\f$).
   */
  int max_iterations = 10000;      // polls
  /**
   * \ingroup optimizers_derivative_free
   * \brief Budget on objective value evaluations, checked before each poll (default \f$10^5\f$).
   */
  long max_value_evaluations = 100000;
  /**
   * \ingroup optimizers_derivative_free
   * \brief Largest spread \f$\max |f(x \pm \delta d_j) - f(x)|\f$ over the final failed poll, beyond \f$64\,\varepsilon\,|f(x)|\f$, for which a convergence verdict stands; a larger one ends the run as `optimizer_status::stalled` (default \f$10^{-6}\f$, in the units of \f$f\f$).
   */
  double value_spread_tolerance = 1e-6;  // a convergence verdict needs max |f(trial) - f(x)| of the failed poll at most this
  /**
   * \ingroup optimizers_derivative_free
   * \brief Optional trace callback invoked once per poll (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_derivative_free
   * \brief Minimizes `objective` starting from `x0` by compass search.
   *
   * Throws `std::invalid_argument` if `x0.extent(0) < 1`.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point of length \f$n \ge 1\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_derivative_free, compass_optimizer, minimize()
   */
  template <value_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    if (n < 1) throw std::invalid_argument("compass_optimizer: the point needs at least one coordinate");
    optimizer_report report;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    optimizer_vector x("x", n), trial("x_trial", n), best("x_best", n), previous("x_previous", n);
    Kokkos::deep_copy(x, x0);
    const auto f = [&](const optimizer_vector& point) {
      ++report.value_evaluations;
      const double v = objective.value(optimizer_point(point));
      return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };
    double fx = f(x), search_size = nan;
    const auto finish = [&](optimizer_status status) {
      report.status = status;
      report.point = x;
      report.value = fx;
      report.search_size = search_size;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    if (!std::isfinite(fx)) return finish(optimizer_status::start_not_admissible);

    const double delta0 = initial_step * std::max(1.0, detail::max_abs(x0));
    double delta = delta0;
    std::uint64_t state = seed;
    const auto uniform = [&] {  // Knuth's MMIX LCG, top 53 bits, as multistart
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(state >> 11) * 0x1.0p-53;
    };
    std::vector<double> w(n);
    double ww = 1.0;
    // column j of the poll basis, coordinate i
    const auto direction = [&](int i, int j) {
      if (poll_basis == compass_poll_basis::coordinate) return i == j ? 1.0 : 0.0;
      return (i == j ? 1.0 : 0.0) - 2.0 * w[i] * w[j] / ww;
    };

    for (int it = 1;; ++it) {
      if (it > max_iterations || report.value_evaluations >= max_value_evaluations) {
        report.iterations = it - 1;
        return finish(it > max_iterations ? optimizer_status::iteration_budget : optimizer_status::evaluation_budget);
      }
      if (poll_basis == compass_poll_basis::random_orthonormal) {
        // ponytail: a Householder reflection of w uniform in the cube, not a Haar-random basis;
        // its columns still reach every direction, which is what the poll needs
        do {
          ww = 0.0;
          for (double& t : w) t = 2.0 * uniform() - 1.0, ww += t * t;
        } while (!(ww > 1e-12));
      }
      Kokkos::deep_copy(previous, x);
      const double f_previous = fx;
      double f_best = fx;
      bool improved = false, any_distinct = false;
      double spread = 0.0;  // max |f(trial) - f(x)| over the poll's trial points
      for (int j = 0; j < n && !(improved && first_improvement); ++j)
        for (int s = 0; s < 2 && !(improved && first_improvement); ++s) {
          bool distinct = false;
          for (int i = 0; i < n; ++i) {
            trial(i) = x(i) + (s == 0 ? delta : -delta) * direction(i, j);
            distinct = distinct || trial(i) != x(i);
          }
          if (!distinct) continue;
          any_distinct = true;
          const double ft = f(trial);
          spread = std::max(spread, std::abs(ft - fx));
          if (ft < f_best) f_best = ft, Kokkos::deep_copy(best, trial), improved = true;
        }
      const double step = delta;
      if (improved) {
        Kokkos::deep_copy(x, best);
        fx = f_best;
        delta = std::min(delta * expansion, max_step_multiple * delta0);
        search_size = nan;
      } else {
        delta *= contraction;
        search_size = step;
      }
      if (trace) trace(optimizer_trace_point{it, x, fx, {}, nan, improved ? step : 0.0});
      report.iterations = it;
      // A failed poll whose values still differ from f(x) by more than value_spread_tolerance (plus
      // the rounding of f, 64 eps |f|) is not at a minimum of f but on noise below the step: the
      // run is stalled, not converged.
      const auto converged_unless_spread = [&](optimizer_status status) {
        const double resolution = 64.0 * std::numeric_limits<double>::epsilon() * std::abs(fx);
        return spread <= value_spread_tolerance + resolution ? status : optimizer_status::stalled;
      };
      if (!any_distinct) return finish(optimizer_status::converged_at_resolution);
      stop_rule_input in;
      in.iteration = it, in.value = fx, in.point = x, in.search_size = search_size;
      in.has_previous = true, in.previous_value = f_previous, in.previous_point = previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none)
        return finish(verdict == stop_verdict::converged                 ? converged_unless_spread(optimizer_status::converged)
                      : verdict == stop_verdict::converged_at_resolution ? converged_unless_spread(optimizer_status::converged_at_resolution)
                                                                         : optimizer_status::stopped_on_change);
    }
  }
};

}  // namespace laplace::optimizers
