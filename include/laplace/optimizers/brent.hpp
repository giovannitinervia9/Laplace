#pragma once

// Brent's method in one dimension (note/optimizers.md, decisions 4 and 12.2):
// value only (C0), for a function of one variable, smooth or not.
//
// Bracketing: from x0 and x0 + initial_step (turned round when the second is
// higher), golden expansion b + 1.618 (b - a) until the value no longer falls,
// so that f(b) <= f(a) and f(b) <= f(c) with b between a and c. A non-finite
// value is +infinity, a refused point (decision 3). If the value still falls
// after max_expansions expansions or the evaluation budget, there is no
// bracket and the run stops at the lowest point seen with a budget status.
//
// Reduction (Brent 1973, chapter 5): the vertex of the parabola through the
// three best points when it falls inside the bracket and moves less than half
// the step before last, a golden-section step into the larger part otherwise.
// The stationarity measure is the half-width of the bracket, in the units of
// x, given to the stop rule as search_size (search_size_stop_rule). Brent's
// own test, the bracket within sqrt(eps) |x| + absolute_resolution of the best
// point, ends the run as converged at resolution: no further point can be told
// apart from x there.
//
// A convex function with a kink (the Laplace mu at fixed sigma, a sum of
// |y_i - mu|) is reduced as well: the bracket holds a minimizer at every
// step, the parabolic steps are refused where they do not shorten it and the
// golden steps shrink it by 0.618. On a flat set of minimizers (the median
// interval at even n) the run ends at some point of the set.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace laplace::optimizers {

namespace detail {

// What Brent gives a stop rule: the bracket half-width and the previous point.
inline constexpr unsigned brent_provides = stop_needs_search_size | stop_needs_previous;

}  // namespace detail

/**
 * \ingroup optimizers_block_coordinate
 * \brief Derivative-free minimization in one variable by Brent's method.
 *
 * Minimizes an objective modeling `value_objective` whose starting point has
 * length \f$1\f$ (`x0.extent(0) == 1`; any other size throws
 * `std::invalid_argument`) on the host, using objective values only. Because
 * it needs no derivatives, it applies to functions with kinks as well as
 * smooth ones. Runs are invoked through minimize() or `run()`. The template
 * parameter `StopRule` defaults to `search_size_stop_rule`, and
 * `stop_rule_for` admits every stopping rule requiring at most
 * `stop_needs_search_size` and `stop_needs_previous`.
 *
 * ### Bracketing
 *
 * Starting from \f$a = x_0\f$ and \f$b = x_0 + h\f$ with \f$h\f$ equal to
 * `initial_step`, the two points are swapped if \f$f(b) > f(a)\f$. The search
 * then steps in the descending direction by golden expansion,
 * \f$c = b + \varphi (b - a)\f$ with \f$\varphi \approx 1.618034\f$, until
 * \f$f(c) \ge f(b)\f$. A non-finite value is treated as \f$+\infty\f$, so a
 * point where the objective cannot be evaluated ends expansion on its side.
 * Once expansion stops, \f$b\f$ lies strictly between \f$a\f$ and \f$c\f$ with
 * \f$f(b) \le f(a)\f$ and \f$f(b) \le f(c)\f$, bracketing a minimizer.
 *
 * ### Reduction and stationarity
 *
 * The reduction keeps a bracket \f$[\mathrm{lo}, \mathrm{hi}]\f$ that holds a
 * minimizer, starting from the ends of the one found above. Each reduction
 * iteration first checks `stop_rule` with `search_size` set to the bracket
 * half-width \f$h_{\mathrm{b}} = \tfrac{1}{2}(\mathrm{hi} - \mathrm{lo})\f$
 * in the units of \f$x\f$, and then checks Brent's own resolution test
 * (Brent, 1973, chapter 5):
 * \f[
 *   |x - m| \le 2\,\mathrm{tol}_1 - h_{\mathrm{b}},
 *   \qquad m = \tfrac{1}{2}(\mathrm{lo} + \mathrm{hi}),
 *   \qquad \mathrm{tol}_1 = \sqrt{\varepsilon}\,|x| + r_{\mathrm{abs}},
 * \f]
 * where \f$x\f$ is the best point, \f$\varepsilon\f$ is double-precision
 * machine epsilon, and \f$r_{\mathrm{abs}}\f$ is `absolute_resolution`. When
 * neither stops the run, the iteration tries the vertex of the parabola
 * through the three best points if it falls strictly inside
 * \f$(\mathrm{lo}, \mathrm{hi})\f$ and moves less than half the step before
 * last; otherwise it takes a golden-section step of fraction \f$0.381966\f$
 * into the larger part of the bracket. Parabolic steps converge quickly near a
 * smooth minimum, while golden-section steps bound the work at a kink.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `search_size_stop_rule{}` | Stopping rule queried on `search_size` and `previous_*` at each reduction step |
 * | `initial_step` | \f$1\f$ | Trial step length \f$h\f$ starting the bracketing phase |
 * | `max_expansions` | \f$50\f$ | Maximum golden expansions allowed while bracketing |
 * | `max_iterations` | \f$200\f$ | Maximum number of reduction steps |
 * | `max_value_evaluations` | \f$10^5\f$ | Budget on objective value evaluations |
 * | `absolute_resolution` | \f$10^{-12}\f$ | Absolute term \f$r_{\mathrm{abs}}\f$ in Brent's resolution test |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked once per reduction step after iteration \f$0\f$ |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from
 * \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(x_0)\f$ is not finite
 *   (`search_size` is quiet NaN).
 * - `optimizer_status::converged`, `optimizer_status::converged_at_resolution`,
 *   or `optimizer_status::stopped_on_change` when returned by `stop_rule`.
 * - `optimizer_status::converged_at_resolution` when Brent's resolution test
 *   holds because no further point can be told apart from \f$x\f$.
 * - `optimizer_status::iteration_budget` when no bracket is found after
 *   `max_expansions` expansions (returning the lowest point seen with quiet
 *   NaN `search_size`), or when `max_iterations` reduction steps are reached.
 * - `optimizer_status::evaluation_budget` when `max_value_evaluations` runs
 *   out during bracketing (with quiet NaN `search_size`) or during reduction.
 *
 * In `optimizer_report`, `point` and `value` hold the best point found,
 * `search_size` holds the final bracket half-width (quiet NaN when no bracket
 * was found), `iterations` counts reduction steps, `value_evaluations` counts
 * objective evaluations, `gradient` is empty, `gradient_evaluations` and
 * `curvature_evaluations` are \f$0\f$, and `decrement` is quiet NaN.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_search_size`
 *   and `stop_needs_previous`; `search_size_stop_rule` by default.
 * \see \ref optimizers_block_coordinate, \ref optimizers, minimize(), search_size_stop_rule
 */
template <class StopRule = search_size_stop_rule>
  requires stop_rule_for<StopRule, detail::brent_provides>
struct brent_optimizer {
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Stopping rule instance queried at each reduction step (default `search_size_stop_rule{}`).
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Length of the initial bracketing step from `x0` (default \f$1\f$).
   */
  double initial_step = 1.0;          // length of the first trial step of the bracketing
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Maximum number of golden expansions allowed while bracketing (default \f$50\f$).
   */
  int max_expansions = 50;            // golden expansions allowed while bracketing
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Maximum number of reduction steps after a bracket is found (default \f$200\f$).
   */
  int max_iterations = 200;           // reduction steps
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Budget on objective value evaluations across bracketing and reduction (default \f$10^5\f$).
   */
  long max_value_evaluations = 100000;
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Absolute resolution added to \f$\sqrt{\varepsilon}\,|x|\f$ in Brent's resolution test (default \f$10^{-12}\f$).
   */
  double absolute_resolution = 1e-12; // added to sqrt(eps) |x| in Brent's own test
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Optional trace callback invoked once per reduction step after iteration \f$0\f$ (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Minimizes the one-dimensional `objective` starting from `x0` by Brent's method.
   *
   * Throws `std::invalid_argument` if `x0.extent(0) != 1`.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point of length \f$1\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_block_coordinate, brent_optimizer, minimize()
   */
  template <value_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    if (x0.extent(0) != 1) throw std::invalid_argument("brent_optimizer: the point must have exactly one coordinate");
    const auto start = std::chrono::steady_clock::now();
    optimizer_report report;
    optimizer_vector point("x", 1), previous("x_previous", 1);
    const auto f = [&](double t) {
      point(0) = t;
      ++report.value_evaluations;
      const double v = objective.value(optimizer_point(point));
      return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };
    const auto finish = [&](optimizer_status status, double x, double fx, double half) {
      report.status = status;
      report.point = optimizer_vector("x", 1);
      report.point(0) = x;
      report.value = fx;
      report.search_size = half;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    const auto out_of_budget = [&] { return report.value_evaluations >= max_value_evaluations; };
    constexpr double gold = 1.6180339887498949, cgold = 0.3819660112501051;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // bracketing
    double a = x0(0), fa = f(a);
    if (!std::isfinite(fa)) return finish(optimizer_status::start_not_admissible, a, fa, nan);
    double b = a + initial_step, fb = f(b);
    if (fb > fa) std::swap(a, b), std::swap(fa, fb);
    double c = b + gold * (b - a), fc = f(c);
    for (int e = 1; fc < fb && e < max_expansions && !out_of_budget(); ++e) {
      a = b, fa = fb, b = c, fb = fc;
      c = b + gold * (b - a), fc = f(c);
    }
    if (fc < fb)
      return finish(out_of_budget() ? optimizer_status::evaluation_budget : optimizer_status::iteration_budget, c, fc,
                    nan);

    // reduction: x the best point, w the second best, v the previous w; (lo, hi) the bracket
    double lo = std::min(a, c), hi = std::max(a, c);
    double x = b, w = b, v = b, fx = fb, fw = fb, fv = fb, d = 0.0, e = 0.0;
    double f_previous = fx;
    previous(0) = x;
    for (int it = 0;; ++it) {
      const double mid = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
      const double tol1 = std::sqrt(std::numeric_limits<double>::epsilon()) * std::abs(x) + absolute_resolution;
      const double tol2 = 2.0 * tol1;
      point(0) = x;
      if (it > 0 && trace) trace(optimizer_trace_point{it, point, fx, {}, nan, std::abs(x - previous(0))});
      stop_rule_input in;
      in.iteration = it, in.value = fx, in.point = point, in.search_size = half;
      in.has_previous = it > 0, in.previous_value = f_previous, in.previous_point = previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none) {
        report.iterations = it;
        return finish(verdict == stop_verdict::converged                 ? optimizer_status::converged
                      : verdict == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                                         : optimizer_status::stopped_on_change,
                      x, fx, half);
      }
      if (std::abs(x - mid) <= tol2 - half) {
        report.iterations = it;
        return finish(optimizer_status::converged_at_resolution, x, fx, half);
      }
      if (it >= max_iterations || out_of_budget()) {
        report.iterations = it;
        return finish(it >= max_iterations ? optimizer_status::iteration_budget : optimizer_status::evaluation_budget, x,
                      fx, half);
      }
      bool parabolic = false;
      if (std::abs(e) > tol1) {
        const double r = (x - w) * (fx - fv);
        double q = (x - v) * (fx - fw);
        double p = (x - v) * q - (x - w) * r;
        q = 2.0 * (q - r);
        if (q > 0.0) p = -p;
        q = std::abs(q);
        const double e_before_last = e;
        e = d;
        if (std::abs(p) < std::abs(0.5 * q * e_before_last) && p > q * (lo - x) && p < q * (hi - x)) {
          d = p / q;
          const double u = x + d;
          if (u - lo < tol2 || hi - u < tol2) d = mid >= x ? tol1 : -tol1;
          parabolic = true;
        }
      }
      if (!parabolic) {
        e = x >= mid ? lo - x : hi - x;
        d = cgold * e;
      }
      const double u = std::abs(d) >= tol1 ? x + d : x + (d >= 0.0 ? tol1 : -tol1);
      const double fu = f(u);
      f_previous = fx;
      previous(0) = x;
      if (fu <= fx) {
        (u >= x ? lo : hi) = x;
        v = w, fv = fw, w = x, fw = fx, x = u, fx = fu;
      } else {
        (u < x ? lo : hi) = u;
        if (fu <= fw || w == x) {
          v = w, fv = fw, w = u, fw = fu;
        } else if (fu <= fv || v == x || v == w) {
          v = u, fv = fu;
        }
      }
    }
  }
};

}  // namespace laplace::optimizers
