#pragma once

// Multistart (note/optimizers.md, decision 4): any optimizer run from several
// starts, the best result kept. Sequential.
//
// Starts: the caller's (`starts`, used verbatim; x0 is then ignored), or x0
// followed by start_count - 1 points of a Latin hypercube around it: on
// coordinate j the interval x0_j +- 3 spread max(1, |x0_j|) is cut into
// start_count - 1 strata, each used once, in an order and at a position
// drawn from a 64-bit LCG seeded with `seed` (the same starts on every
// platform and compiler). x is in free R^n (decision 3), so no start can fall
// outside the domain of a constrained parameter.
//
// Selection (values first): a run with a non-finite value is never kept; a
// later run replaces the kept one when its value is lower by more than
// distinct_tolerance, whatever the two statuses, or when it is within
// distinct_tolerance, succeeded, and the kept one did not. A converged run on
// a boundary or in a worse basin therefore never displaces a lower run that
// stalled at a kink. The kept run is the best, never the last.
//
// The report is the kept run's, with the counts of evaluations and
// iterations and the time summed over every start. run_multistart() adds the
// number of starts, of converged starts, of distinct optima (converged values
// sorted, a new optimum after each gap larger than distinct_tolerance) and of
// converged starts within distinct_tolerance of the kept value, and every
// start's own report.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_multistart
 * \brief Full summary returned by `multistart_optimizer::run_multistart()`.
 *
 * Collects the aggregated `optimizer_report` of the kept run in `best`, every start's individual report in `runs` in execution order with its own counts and elapsed time, and summary counts across all starts. `start_count` records how many starting points were executed, and `converged_count` counts the runs for which `optimizer_succeeded()` is `true` and `value` is finite.
 *
 * To compute `distinct_optimum_count`, the finite objective values of the converged runs are sorted in non-decreasing order; the first value starts an optimum, and each subsequent value starts a new optimum only when it exceeds the previous value by more than `distinct_tolerance` (so a chain of values each within `distinct_tolerance` of the next counts as one optimum), yielding \f$0\f$ when no run converged with a finite value. A `distinct_optimum_count` above \f$1\f$ shows that the objective has several local minima. `best_found_count` counts the converged runs whose finite value lies within `distinct_tolerance` of `best.value` when `best` converged, or \f$0\f$ otherwise.
 *
 * \see \ref optimizers_multistart, multistart_optimizer, optimizer_report, optimizer_succeeded()
 */
struct multistart_report {
  /**
   * \ingroup optimizers_multistart
   * \brief Kept run's `optimizer_report`, with `iterations`, `value_evaluations`, `gradient_evaluations`, `curvature_evaluations`, and `elapsed_seconds` summed over all starts.
   */
  optimizer_report best;  // counts and time summed over all starts
  /**
   * \ingroup optimizers_multistart
   * \brief Number of starting points executed.
   */
  int start_count = 0;
  /**
   * \ingroup optimizers_multistart
   * \brief Number of starts that succeeded with a finite objective value.
   */
  int converged_count = 0;
  /**
   * \ingroup optimizers_multistart
   * \brief Number of distinct optima among finite converged values separated by gaps larger than `distinct_tolerance`.
   */
  int distinct_optimum_count = 0;
  /**
   * \ingroup optimizers_multistart
   * \brief Number of converged runs within `distinct_tolerance` of `best.value` when `best` converged, or \f$0\f$ otherwise.
   */
  int best_found_count = 0;
  /**
   * \ingroup optimizers_multistart
   * \brief Individual `optimizer_report` for each start, in execution order, with each run's own counts.
   */
  std::vector<optimizer_report> runs;  // one per start, in the order run
};

/**
 * \ingroup optimizers_multistart
 * \brief Multistart wrapper running an inner optimizer sequentially from several starts and keeping the best run.
 *
 * `multistart_optimizer<Inner>` runs any optimizer `Inner` (stored in `optimizer`) from multiple starting points one after the other on the host and keeps one run. It satisfies `optimizer_for` for every objective that `Inner` accepts, so `minimize()` accepts it without depending on the inner method (any optimizer of the module, or another `multistart_optimizer`).
 *
 * When `starts` is non-empty, its points are used as given in that order and `x0` is ignored. Otherwise `generated_starts()` produces \f$x_0\f$ followed by \f$m\f$ points of a Latin hypercube around it (McKay, Beckman, and Conover, 1979), where \f$m\f$ equals `start_count` minus \f$1\f$ (or \f$0\f$ when `start_count` is at most \f$1\f$, which yields \f$x_0\f$ alone). For \f$m \ge 1\f$ on coordinate \f$j\f$, the interval \f$[x_{0,j} - h_j,\, x_{0,j} + h_j]\f$ with half-width \f$h_j = 3 \sigma \max(1, |x_{0,j}|)\f$ and \f$\sigma\f$ equal to `spread` is partitioned into \f$m\f$ equal strata of width \f$2 h_j / m\f$. Generated start \f$i \in \{0, \dots, m - 1\}\f$ takes coordinate \f$j\f$ equal to
 * \f[ x_{0,j} - h_j + \frac{2 h_j (\pi_j(i) + u_{ij})}{m}, \f]
 * where \f$\pi_j\f$ is a random permutation of \f$0, \dots, m - 1\f$ (by Fisher-Yates shuffle) and \f$u_{ij}\f$ is uniform in \f$[0, 1)\f$, so every stratum of every coordinate is used once. Both \f$\pi_j\f$ and \f$u_{ij}\f$ come from a 64-bit linear congruential generator seeded with `seed` (taking the top 53 bits of the state for each uniform), producing identical starts across platforms and compilers. Because points lie in the free parametrization of the objective, no start can fall outside a constrained parameter's domain when the objective maps constraints away.
 *
 * Runs are compared in execution order to select the kept run. A run whose value is not finite is never kept while a finite one exists; if every run has a non-finite value, the first run is kept. A later run replaces the kept run when its value is lower by more than `distinct_tolerance`, whatever the statuses of the two runs. It also replaces the kept run when its value is within `distinct_tolerance` of the kept value, it succeeded (`optimizer_succeeded()` returns `true`), and the kept run did not. So success only breaks a tie within the tolerance; the kept run is the best under this rule, never merely the last.
 *
 * `run()` returns the kept run's `optimizer_report` (preserving its `status`, `point`, `value`, `gradient`, `decrement`, and `final_factorization`), with `iterations`, `value_evaluations`, `gradient_evaluations`, `curvature_evaluations`, and `elapsed_seconds` replaced by their sums across all starts. `run_multistart()` returns a `multistart_report` containing that aggregated report as `best`, `start_count`, `converged_count`, `distinct_optimum_count`, `best_found_count`, and `runs`.
 *
 * Settings and defaults: `optimizer` (default `Inner{}`) sets the inner optimizer run from each start; `start_count` (default \f$10\f$) sets the number of generated starts when `starts` is empty (more starts explore more basins at a cost linear in the count); `starts` (default empty) supplies explicit starting points; `spread` (default \f$1\f$) scales the half-width of the sampled box; `seed` (default \f$1\f$) seeds the start generator (a different seed gives different starts); `distinct_tolerance` (default \f$10^{-5}\f$) sets the replacement and optimum-counting threshold and should not be tighter than the accuracy of the inner optimizer's stopping rule, since runs reaching the same minimum differ in value by about that accuracy.
 *
 * The runtime cost is the sum of the inner runs plus start generation, which is negligible; memory holds the starting points and one `optimizer_report` per start, each storing a point and gradient vector of dimension \f$n\f$ (\f$O(n)\f$ per start). Limits: starts execute sequentially on one thread; generated starts cover only the box around `x0`, so minima outside it are found only if an inner run walks there; selection and counts compare objective values, so distinct minima within `distinct_tolerance` count as one optimum; whether the kept run is a minimum is only as reliable as the inner optimizer's verdict.
 *
 * \code
 * multistart_optimizer<bfgs_optimizer<>> ms;
 * ms.start_count = 10;
 * const optimizer_report best = minimize(ms, f, x0);
 * const multistart_report full = ms.run_multistart(f, x0);
 * \endcode
 *
 * \tparam Inner Inner optimizer type.
 * \see \ref optimizers_multistart, multistart_report, minimize(), optimizer_for
 */
template <class Inner>
struct multistart_optimizer {
  /**
   * \ingroup optimizers_multistart
   * \brief Inner optimizer instance executed from each start (default `Inner{}`).
   */
  Inner optimizer{};
  /**
   * \ingroup optimizers_multistart
   * \brief Number of generated starts when `starts` is empty, including `x0` (default \f$10\f$).
   */
  int start_count = 10;                     // generated starts, x0 included
  /**
   * \ingroup optimizers_multistart
   * \brief Optional caller-supplied starting points used in order when non-empty, ignoring `x0` (default empty).
   */
  std::vector<std::vector<double>> starts;  // the caller's starts; when not empty x0 is ignored
  /**
   * \ingroup optimizers_multistart
   * \brief Half-width multiplier \f$\sigma\f$ for the Latin-hypercube box around `x0` (default \f$1\f$).
   */
  double spread = 1.0;                      // half-width 3 spread max(1, |x0_j|)
  /**
   * \ingroup optimizers_multistart
   * \brief Seed for the 64-bit linear congruential generator of the Latin-hypercube starts (default \f$1\f$).
   */
  std::uint64_t seed = 1;
  /**
   * \ingroup optimizers_multistart
   * \brief Value threshold for run replacement and optimum counting (default \f$10^{-5}\f$).
   */
  double distinct_tolerance = 1e-5;         // values closer than this are one optimum

  /**
   * \ingroup optimizers_multistart
   * \brief Generates `x0` followed by `start_count - 1` Latin-hypercube points around `x0`.
   *
   * \param x0 Center starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Vector of generated starting points, identical across platforms and compilers for a given `seed`.
   * \see \ref optimizers_multistart, multistart_optimizer
   */
  // x0 and start_count - 1 Latin-hypercube points around it.
  std::vector<std::vector<double>> generated_starts(const optimizer_point& x0) const {
    const int n = static_cast<int>(x0.extent(0));
    const int m = std::max(start_count, 1) - 1;
    std::vector<std::vector<double>> out(m + 1, std::vector<double>(n));
    for (int j = 0; j < n; ++j) out[0][j] = x0(j);
    std::uint64_t state = seed;
    auto uniform = [&] {  // Knuth's MMIX LCG, top 53 bits
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      return static_cast<double>(state >> 11) * 0x1.0p-53;
    };
    std::vector<int> stratum(m);
    for (int j = 0; j < n; ++j) {
      std::iota(stratum.begin(), stratum.end(), 0);
      for (int i = m - 1; i > 0; --i)  // Fisher-Yates
        std::swap(stratum[i], stratum[std::min(i, static_cast<int>(uniform() * (i + 1)))]);
      const double half = 3.0 * spread * std::max(1.0, std::abs(x0(j)));
      for (int i = 0; i < m; ++i) out[i + 1][j] = x0(j) - half + 2.0 * half * (stratum[i] + uniform()) / m;
    }
    return out;
  }

  /**
   * \ingroup optimizers_multistart
   * \brief Runs the inner optimizer sequentially from all starts and returns a full `multistart_report`.
   *
   * \tparam O Objective type satisfying `optimizer_for<Inner, O>`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$ (ignored when `starts` is non-empty).
   * \return A `multistart_report` containing the kept run with aggregated counts in `best`, summary counts, and all individual run reports.
   * \see \ref optimizers_multistart, multistart_report, run()
   */
  template <class O>
    requires optimizer_for<Inner, O>
  multistart_report run_multistart(O& objective, const optimizer_point& x0) const {
    const std::vector<std::vector<double>> points = starts.empty() ? generated_starts(x0) : starts;
    multistart_report out;
    int kept = -1;
    long f_count = 0, g_count = 0, h_count = 0;
    int iterations = 0;
    double seconds = 0.0;
    for (const auto& p : points) {
      optimizer_vector start("start", p.size());
      for (std::size_t i = 0; i < p.size(); ++i) start(i) = p[i];
      optimizer_report r = minimize(optimizer, objective, optimizer_point(start));
      f_count += r.value_evaluations, g_count += r.gradient_evaluations, h_count += r.curvature_evaluations;
      iterations += r.iterations, seconds += r.elapsed_seconds;
      out.runs.push_back(std::move(r));
      const optimizer_report& c = out.runs.back();
      if (!std::isfinite(c.value)) continue;
      const int index = static_cast<int>(out.runs.size()) - 1;
      if (kept < 0) {
        kept = index;
        continue;
      }
      const optimizer_report& b = out.runs[kept];
      const bool c_ok = optimizer_succeeded(c.status), b_ok = optimizer_succeeded(b.status);
      // values first: a lower value by more than distinct_tolerance wins whatever the statuses;
      // within it, a succeeded run beats one that did not succeed
      if (c.value < b.value - distinct_tolerance || (c.value <= b.value + distinct_tolerance && c_ok && !b_ok))
        kept = index;
    }
    out.start_count = static_cast<int>(points.size());
    out.best = out.runs[kept < 0 ? 0 : kept];
    out.best.value_evaluations = f_count, out.best.gradient_evaluations = g_count;
    out.best.curvature_evaluations = h_count, out.best.iterations = iterations, out.best.elapsed_seconds = seconds;

    std::vector<double> converged;
    for (const auto& r : out.runs)
      if (optimizer_succeeded(r.status) && std::isfinite(r.value)) converged.push_back(r.value);
    std::sort(converged.begin(), converged.end());
    out.converged_count = static_cast<int>(converged.size());
    for (std::size_t i = 0; i < converged.size(); ++i) {
      if (i == 0 || converged[i] - converged[i - 1] > distinct_tolerance) ++out.distinct_optimum_count;
      if (optimizer_succeeded(out.best.status) && std::abs(converged[i] - out.best.value) <= distinct_tolerance)
        ++out.best_found_count;
    }
    return out;
  }

  /**
   * \ingroup optimizers_multistart
   * \brief Runs the inner optimizer sequentially from all starts and returns the kept run's `optimizer_report`.
   *
   * Sums `iterations`, evaluation counts, and `elapsed_seconds` over all starts.
   *
   * \tparam O Objective type satisfying `optimizer_for<Inner, O>`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$ (ignored when `starts` is non-empty).
   * \return Kept `optimizer_report` with aggregated cost fields.
   * \see \ref optimizers_multistart, run_multistart(), minimize()
   */
  template <class O>
    requires optimizer_for<Inner, O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    return run_multistart(objective, x0).best;
  }
};

}  // namespace laplace::optimizers
