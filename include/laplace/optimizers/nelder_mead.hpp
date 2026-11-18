#pragma once

// Nelder-Mead (note/optimizers.md, decisions 4, 5 and 12.2): value only (C0),
// for a function of n >= 1 variables, smooth or not.
//
// The simplex: x0 and the n points x0 + h_j e_j with h_j = initial_step
// max(1, |x0_j|), or the caller's initial_simplex (n + 1 points). A
// non-finite value is +infinity, a refused point (decision 3).
//
// One iteration sorts the vertices (best first), reflects the worst through
// the centroid of the others and then expands, contracts outside or inside,
// or shrinks every vertex toward the best (Lagarias, Reeds, Wright and
// Wright 1998). The coefficients are Gao and Han's (2012), 1, 1 + 2/n,
// 3/4 - 1/(2n), 1 - 1/n, which at n = 2 are the classical 1, 2, 1/2, 1/2;
// at n = 1 the classical ones are used (Gao and Han's shrink would be 0).
//
// Degenerate simplices (McKinnon 1998): Nelder-Mead can collapse onto a line
// through a point that is not stationary while the values behave as at a
// minimum. The shape is watched: the conditioning |det E| / prod |E_j|, E the
// edges from the best vertex (1 for a right-angled simplex, 0 for a flat one,
// scale-free), and when it falls below degenerate_tolerance the simplex is
// rebuilt right-angled at the best vertex with edges equal to the diameter
// reached, at most max_restarts times.
//
// The stationarity measure is the diameter of the simplex from the best
// vertex, max_j |x_j - x_0|_2, in the units of x, given to the stop rule as
// search_size (search_size_stop_rule). A shrink that leaves every vertex
// where it was (the simplex has collapsed to rounding) ends the run as
// converged at resolution.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace laplace::optimizers {

namespace detail {

// What Nelder-Mead and compass give a stop rule: the search size and the previous point.
inline constexpr unsigned derivative_free_provides = stop_needs_search_size | stop_needs_previous;

// |det E| / prod_j |E_j| for the edges E_j = v_{j+1} - v_0 (vertices v, n + 1 of
// them, each of length n): the product of the diagonal of R in E = QR after
// normalizing the columns, by modified Gram-Schmidt. 0 when an edge is 0.
inline double simplex_conditioning(const std::vector<std::vector<double>>& v) {
  const std::size_t n = v.size() - 1;
  std::vector<std::vector<double>> e(n, std::vector<double>(n));
  for (std::size_t j = 0; j < n; ++j) {
    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) e[j][i] = v[j + 1][i] - v[0][i], norm += e[j][i] * e[j][i];
    norm = std::sqrt(norm);
    if (!(norm > 0.0)) return 0.0;
    for (double& t : e[j]) t /= norm;
  }
  double product = 1.0;
  for (std::size_t j = 0; j < n; ++j) {
    double r = 0.0;
    for (double t : e[j]) r += t * t;
    r = std::sqrt(r);
    product *= r;
    if (!(r > 0.0)) return 0.0;
    for (double& t : e[j]) t /= r;
    for (std::size_t k = j + 1; k < n; ++k) {
      double d = 0.0;
      for (std::size_t i = 0; i < n; ++i) d += e[j][i] * e[k][i];
      for (std::size_t i = 0; i < n; ++i) e[k][i] -= d * e[j][i];
    }
  }
  return product;
}

}  // namespace detail

/**
 * \ingroup optimizers_derivative_free
 * \brief Derivative-free minimization in several variables by the Nelder-Mead simplex method.
 *
 * Minimizes an objective modeling `value_objective` on the host from a starting
 * point `x0` of length \f$n \ge 1\f$, using objective values only. A length of
 * \f$0\f$ or an `initial_simplex` whose dimensions do not match \f$n + 1\f$
 * vertices of \f$n\f$ coordinates throws `std::invalid_argument`. Any
 * non-finite objective value counts as \f$+\infty\f$. Runs are invoked through
 * minimize() or `run()`. The template parameter `StopRule` defaults to
 * `search_size_stop_rule`, and `stop_rule_for` admits any stopping rule
 * requiring at most `stop_needs_search_size` and `stop_needs_previous`.
 *
 * ### Simplex and iteration
 *
 * When `initial_simplex` is empty, the starting simplex consists of \f$x_0\f$
 * and the \f$n\f$ vertices \f$x_0 + h_j e_j\f$ with \f$h_j\f$ equal to
 * `initial_step` times \f$\max(1, |x_{0,j}|)\f$, where \f$e_j\f$ is the
 * \f$j\f$-th coordinate unit vector; otherwise `initial_simplex` is used and
 * `x0` is ignored. Each iteration (Nelder and Mead, 1965; in the order of
 * Lagarias, Reeds, Wright and Wright, 1998) sorts the vertices \f$v_0, \dots,
 * v_n\f$ by objective value, best first, and reflects the worst vertex
 * \f$v_n\f$ through the centroid \f$c\f$ of the remaining \f$n\f$ vertices as
 * \f$v_r = 2c - v_n\f$. If \f$f(v_r) < f(v_0)\f$, it evaluates the expansion
 * \f$c + \chi (v_r - c)\f$ and keeps the lower of the two; else if
 * \f$f(v_r) < f(v_{n-1})\f$, it keeps \f$v_r\f$; else it contracts outside,
 * \f$c + \gamma (v_r - c)\f$, when \f$f(v_r) < f(v_n)\f$ (accepted when at most
 * \f$f(v_r)\f$), or inside, \f$c - \gamma (c - v_n)\f$ (accepted when strictly
 * below \f$f(v_n)\f$); if the contraction is refused, every vertex is shrunk
 * toward the best as \f$v_0 + \sigma (v_j - v_0)\f$.
 *
 * When `adaptive_coefficients` is `true` and \f$n \ge 2\f$, the coefficients
 * take Gao and Han's (2012) values \f$\chi = 1 + 2/n\f$,
 * \f$\gamma = 3/4 - 1/(2n)\f$, and \f$\sigma = 1 - 1/n\f$ (reflection \f$1\f$),
 * which coincide with the classical \f$2\f$, \f$1/2\f$, and \f$1/2\f$ at
 * \f$n = 2\f$. At \f$n = 1\f$ or when `adaptive_coefficients` is `false`, the
 * classical coefficients are used.
 *
 * ### Degenerate-simplex safeguard and stationarity
 *
 * When \f$n > 1\f$, restarts remain below `max_restarts`, and the simplex
 * diameter is positive, each iteration computes the scale-free conditioning
 * \f$|\det E| / \prod_{j=1}^n \|E_j\|_2\f$ of the edge matrix \f$E\f$ with
 * columns \f$E_j = v_j - v_0\f$. If the conditioning falls below
 * `degenerate_tolerance`, the simplex is rebuilt right-angled at \f$v_0\f$ with
 * edges along the coordinate axes equal to the current diameter, costing
 * \f$n\f$ objective evaluations (guarding against collapse onto a
 * non-stationary point as in McKinnon, 1998), and the vertices are re-sorted.
 *
 * Stationarity is measured by the diameter from the best vertex,
 * \f$\max_{1 \le j \le n} \|v_j - v_0\|_2\f$ in the units of \f$x\f$, and
 * passed to `stop_rule` as `search_size` once per iteration after sorting and
 * the safeguard. A shrink that leaves every vertex unchanged in floating point
 * ends the run with `optimizer_status::converged_at_resolution`. Any
 * convergence verdict (`converged` or `converged_at_resolution`) stands only
 * when the simplex value spread \f$\max_j f(v_j) - \min_j f(v_j)\f$ is at most
 * `value_spread_tolerance` \f$+ 64\,\varepsilon\,|f(v_0)|\f$, where
 * \f$\varepsilon\f$ is machine epsilon; otherwise the run ends with
 * `optimizer_status::stalled`.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `search_size_stop_rule{}` | Stopping rule queried on `search_size` and `previous_*` once per iteration |
 * | `initial_step` | \f$0.1\f$ | Multiplier for initial edges \f$h_j\f$ when `initial_simplex` is empty |
 * | `initial_simplex` | empty | Optional \f$n + 1\f$ starting vertices of length \f$n\f$; ignores `x0` when non-empty |
 * | `adaptive_coefficients` | `true` | Uses Gao and Han's dimension-dependent coefficients for \f$n \ge 2\f$; classical when `false` or \f$n = 1\f$ |
 * | `max_restarts` | \f$3\f$ | Maximum number of degenerate-simplex rebuilds per run (`0` disables) |
 * | `degenerate_tolerance` | \f$10^{-6}\f$ | Conditioning threshold below which the simplex is rebuilt |
 * | `max_iterations` | \f$10000\f$ | Maximum number of simplex iterations |
 * | `max_value_evaluations` | \f$10^5\f$ | Budget on objective evaluations, checked once per iteration |
 * | `value_spread_tolerance` | \f$10^{-6}\f$ | Maximum simplex value spread (beyond \f$64\,\varepsilon\,|f(v_0)|\f$) for a convergence verdict; otherwise ends as `stalled` |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked once per iteration after iteration \f$0\f$ |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from
 * \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(v_0)\f$ is not finite (`search_size` is quiet NaN).
 * - `optimizer_status::converged`, `optimizer_status::converged_at_resolution`, or `optimizer_status::stopped_on_change` when returned by `stop_rule` (convergence verdicts are subject to the spread check).
 * - `optimizer_status::converged_at_resolution` when a shrink leaves every vertex unchanged in floating point (subject to the spread check).
 * - `optimizer_status::stalled` when a convergence verdict from `stop_rule` or from an unchanged shrink has \f$\max_j f(v_j) - \min_j f(v_j)\f$ above `value_spread_tolerance` \f$+ 64\,\varepsilon\,|f(v_0)|\f$.
 * - `optimizer_status::iteration_budget` after `max_iterations` iterations.
 * - `optimizer_status::evaluation_budget` when `max_value_evaluations` is reached at the iteration check (evaluations may exceed the budget by one iteration).
 *
 * In `optimizer_report`, `point` and `value` hold the best vertex and its
 * value, `search_size` holds the final simplex diameter (quiet NaN on an
 * inadmissible start), `iterations` counts simplex iterations,
 * `value_evaluations` counts objective calls, `gradient` is empty,
 * `gradient_evaluations` and `curvature_evaluations` are \f$0\f$, and
 * `decrement` is quiet NaN. The optional `trace` callback receives the best
 * point, its value, and `search_size` as `step`.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_search_size` and `stop_needs_previous`; `search_size_stop_rule` by default.
 * \see \ref optimizers_derivative_free, \ref optimizers, compass_optimizer, minimize(), search_size_stop_rule
 */
template <class StopRule = search_size_stop_rule>
  requires stop_rule_for<StopRule, detail::derivative_free_provides>
struct nelder_mead_optimizer {
  /**
   * \ingroup optimizers_derivative_free
   * \brief Stopping rule instance queried once per iteration (default `search_size_stop_rule{}`).
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_derivative_free
   * \brief Multiplier on \f$\max(1, |x_{0,j}|)\f$ for the initial simplex edges when `initial_simplex` is empty (default \f$0.1\f$).
   */
  double initial_step = 0.1;                         // edge h_j = initial_step max(1, |x0_j|)
  /**
   * \ingroup optimizers_derivative_free
   * \brief Optional starting simplex of \f$n + 1\f$ vertices of \f$n\f$ coordinates; when non-empty `x0` is ignored (default empty).
   */
  std::vector<std::vector<double>> initial_simplex;  // n + 1 vertices; when not empty x0 is ignored
  /**
   * \ingroup optimizers_derivative_free
   * \brief Whether to use Gao and Han's dimension-dependent coefficients for \f$n \ge 2\f$ instead of the classical ones (default `true`).
   */
  bool adaptive_coefficients = true;                 // Gao and Han's; classical when false
  /**
   * \ingroup optimizers_derivative_free
   * \brief Maximum number of right-angled simplex rebuilds on degeneracy per run (default \f$3\f$).
   */
  int max_restarts = 3;                              // rebuilds of a degenerate simplex
  /**
   * \ingroup optimizers_derivative_free
   * \brief Conditioning threshold below which the simplex is rebuilt (default \f$10^{-6}\f$).
   */
  double degenerate_tolerance = 1e-6;                // conditioning below which it is rebuilt
  /**
   * \ingroup optimizers_derivative_free
   * \brief Maximum number of simplex iterations (default \f$10000\f$).
   */
  int max_iterations = 10000;
  /**
   * \ingroup optimizers_derivative_free
   * \brief Budget on objective value evaluations, checked once per iteration (default \f$10^5\f$).
   */
  long max_value_evaluations = 100000;
  /**
   * \ingroup optimizers_derivative_free
   * \brief Largest spread \f$\max_j f(v_j) - \min_j f(v_j)\f$ over the final simplex, beyond \f$64\,\varepsilon\,|f(v_0)|\f$, for which a convergence verdict stands; a larger one ends the run as `optimizer_status::stalled` (default \f$10^{-6}\f$, in the units of \f$f\f$).
   */
  double value_spread_tolerance = 1e-6;  // a convergence verdict needs max f - min f over the simplex at most this
  /**
   * \ingroup optimizers_derivative_free
   * \brief Optional trace callback invoked once per iteration after iteration \f$0\f$ (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_derivative_free
   * \brief Minimizes `objective` starting from `x0` (or `initial_simplex` when non-empty) by Nelder-Mead.
   *
   * Throws `std::invalid_argument` if the dimension \f$n < 1\f$ or if any
   * vertex in `initial_simplex` does not have length \f$n\f$.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point of length \f$n\f$ (ignored when `initial_simplex` is non-empty).
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_derivative_free, nelder_mead_optimizer, minimize()
   */
  template <value_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    optimizer_report report;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int n = initial_simplex.empty() ? static_cast<int>(x0.extent(0)) : static_cast<int>(initial_simplex.size()) - 1;
    if (n < 1) throw std::invalid_argument("nelder_mead_optimizer: the point needs at least one coordinate");
    for (const auto& vertex : initial_simplex)
      if (static_cast<int>(vertex.size()) != n)
        throw std::invalid_argument("nelder_mead_optimizer: initial_simplex needs n + 1 vertices of n coordinates");

    optimizer_vector trial("x", n), best("x_best", n), previous("x_previous", n);
    const auto f = [&](const std::vector<double>& x) {
      for (int i = 0; i < n; ++i) trial(i) = x[i];
      ++report.value_evaluations;
      const double v = objective.value(optimizer_point(trial));
      return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };

    // the coefficients of reflection, expansion, contraction and shrink
    const double m = std::max(2, n);
    const bool gao_han = adaptive_coefficients && n >= 2;
    const double c_expand = gao_han ? 1.0 + 2.0 / m : 2.0;
    const double c_contract = gao_han ? 0.75 - 0.5 / m : 0.5;
    const double c_shrink = gao_han ? 1.0 - 1.0 / m : 0.5;

    std::vector<std::vector<double>> v(n + 1, std::vector<double>(n));
    if (initial_simplex.empty()) {
      for (int j = 0; j <= n; ++j)
        for (int i = 0; i < n; ++i) v[j][i] = x0(i);
      for (int j = 0; j < n; ++j) v[j + 1][j] += initial_step * std::max(1.0, std::abs(x0(j)));
    } else {
      v = initial_simplex;
    }
    std::vector<double> fv(n + 1);
    std::vector<int> order(n + 1);
    const auto sort_vertices = [&] {
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return fv[a] < fv[b]; });
      std::vector<std::vector<double>> vs(n + 1);
      std::vector<double> fs(n + 1);
      for (int j = 0; j <= n; ++j) vs[j] = std::move(v[order[j]]), fs[j] = fv[order[j]];
      v = std::move(vs), fv = std::move(fs);
    };
    const auto evaluate_all = [&](int from) {
      for (int j = from; j <= n; ++j) fv[j] = f(v[j]);
    };
    const auto diameter = [&] {
      double d = 0.0;
      for (int j = 1; j <= n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += (v[j][i] - v[0][i]) * (v[j][i] - v[0][i]);
        d = std::max(d, std::sqrt(s));
      }
      return d;
    };
    const auto finish = [&](optimizer_status status, double size) {
      report.status = status;
      report.point = optimizer_vector("x", n);
      for (int i = 0; i < n; ++i) report.point(i) = v[0][i];
      report.value = fv[0];
      report.search_size = size;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    const auto out_of_budget = [&] { return report.value_evaluations >= max_value_evaluations; };

    // A small simplex whose values still differ by more than value_spread_tolerance (plus the
    // rounding of f, 64 eps |f|) is not at a minimum of f but on noise below the search size:
    // the run is stalled, not converged.
    const auto converged_unless_spread = [&](optimizer_status status) {
      const double spread = *std::max_element(fv.begin(), fv.end()) - *std::min_element(fv.begin(), fv.end());
      const double resolution = 64.0 * std::numeric_limits<double>::epsilon() * std::abs(fv[0]);
      return spread <= value_spread_tolerance + resolution ? status : optimizer_status::stalled;
    };

    fv[0] = f(v[0]);
    if (!std::isfinite(fv[0])) return finish(optimizer_status::start_not_admissible, nan);
    evaluate_all(1);

    int restarts = 0;
    double f_previous = nan;
    std::vector<double> centroid(n), xr(n), xe(n), xc(n);
    for (int it = 0;; ++it) {
      sort_vertices();
      double size = diameter();
      // McKinnon's safeguard: rebuild a flat simplex right-angled at the best vertex
      if (n > 1 && restarts < max_restarts && size > 0.0 && detail::simplex_conditioning(v) < degenerate_tolerance) {
        for (int j = 1; j <= n; ++j) {
          v[j] = v[0];
          v[j][j - 1] += size;
        }
        evaluate_all(1);
        ++restarts;
        sort_vertices();
        size = diameter();
      }
      for (int i = 0; i < n; ++i) best(i) = v[0][i];
      if (it > 0 && trace) trace(optimizer_trace_point{it, best, fv[0], {}, nan, size});
      stop_rule_input in;
      in.iteration = it, in.value = fv[0], in.point = best, in.search_size = size;
      in.has_previous = it > 0, in.previous_value = f_previous, in.previous_point = previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none) {
        report.iterations = it;
        return finish(verdict == stop_verdict::converged                 ? converged_unless_spread(optimizer_status::converged)
                      : verdict == stop_verdict::converged_at_resolution ? converged_unless_spread(optimizer_status::converged_at_resolution)
                                                                         : optimizer_status::stopped_on_change,
                      size);
      }
      if (it >= max_iterations || out_of_budget()) {
        report.iterations = it;
        return finish(it >= max_iterations ? optimizer_status::iteration_budget : optimizer_status::evaluation_budget,
                      size);
      }
      f_previous = fv[0];
      Kokkos::deep_copy(previous, best);

      std::fill(centroid.begin(), centroid.end(), 0.0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) centroid[i] += v[j][i] / n;
      const std::vector<double>& worst = v[n];
      for (int i = 0; i < n; ++i) xr[i] = 2.0 * centroid[i] - worst[i];
      const double fr = f(xr);
      if (fr < fv[0]) {
        for (int i = 0; i < n; ++i) xe[i] = centroid[i] + c_expand * (xr[i] - centroid[i]);
        const double fe = f(xe);
        if (fe < fr)
          v[n] = xe, fv[n] = fe;
        else
          v[n] = xr, fv[n] = fr;
        continue;
      }
      if (fr < fv[n - 1]) {
        v[n] = xr, fv[n] = fr;
        continue;
      }
      if (fr < fv[n]) {  // outside contraction
        for (int i = 0; i < n; ++i) xc[i] = centroid[i] + c_contract * (xr[i] - centroid[i]);
        const double fc = f(xc);
        if (fc <= fr) {
          v[n] = xc, fv[n] = fc;
          continue;
        }
      } else {  // inside contraction
        for (int i = 0; i < n; ++i) xc[i] = centroid[i] - c_contract * (centroid[i] - worst[i]);
        const double fc = f(xc);
        if (fc < fv[n]) {
          v[n] = xc, fv[n] = fc;
          continue;
        }
      }
      // shrink toward the best vertex; a shrink that moves nothing is the rounding floor
      bool moved = false;
      for (int j = 1; j <= n; ++j)
        for (int i = 0; i < n; ++i) {
          const double t = v[0][i] + c_shrink * (v[j][i] - v[0][i]);
          moved = moved || t != v[j][i];
          v[j][i] = t;
        }
      if (!moved) {
        report.iterations = it + 1;
        return finish(converged_unless_spread(optimizer_status::converged_at_resolution), size);
      }
      evaluate_all(1);
    }
  }
};

}  // namespace laplace::optimizers
