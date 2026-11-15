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
// Selection: a run with a non-finite value
// is never kept; a converged run beats a non-converged one; between two runs
// of the same kind a later one replaces the kept one only when its value is
// lower by more than distinct_tolerance. The kept run is the best, never the
// last.
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
 * \ingroup optimizers_quasi_newton
 * \brief Full summary returned by `multistart_optimizer::run_multistart()`.
 *
 * \see \ref optimizers_quasi_newton, multistart_optimizer, optimizer_report
 */
struct multistart_report {
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Kept run's `optimizer_report`, with `iterations`, evaluation counts, and `elapsed_seconds` summed over all starts.
   */
  optimizer_report best;  // counts and time summed over all starts
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Number of starting points executed.
   */
  int start_count = 0;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Number of starts that converged with a finite objective value.
   */
  int converged_count = 0;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Number of distinct optima among finite converged values separated by gaps larger than `distinct_tolerance`.
   */
  int distinct_optimum_count = 0;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Number of converged runs within `distinct_tolerance` of `best.value` when `best` converged, or \f$0\f$ otherwise.
   */
  int best_found_count = 0;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Individual `optimizer_report` for each start, in the order executed.
   */
  std::vector<optimizer_report> runs;  // one per start, in the order run
};

/**
 * \ingroup optimizers_quasi_newton
 * \brief Multistart wrapper running an inner optimizer sequentially from multiple starts.
 *
 * Models `optimizer_for` for every objective accepted by `Inner`, keeping the best run under the selection rule rather than the last.
 *
 * \tparam Inner Inner optimizer type.
 * \see \ref optimizers_quasi_newton, multistart_report, minimize()
 */
template <class Inner>
struct multistart_optimizer {
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Inner optimizer instance executed from each start.
   */
  Inner optimizer{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Total number of generated starts when `starts` is empty, including `x0` (default 10).
   */
  int start_count = 10;                     // generated starts, x0 included
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Optional caller-supplied starting points; when non-empty, `x0` is ignored.
   */
  std::vector<std::vector<double>> starts;  // the caller's starts; when not empty x0 is ignored
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Half-width multiplier \f$\sigma\f$ for the Latin-hypercube interval \f$x_{0,j} \pm 3 \sigma \max(1, |x_{0,j}|)\f$ (default 1).
   */
  double spread = 1.0;                      // half-width 3 spread max(1, |x0_j|)
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Seed for the 64-bit linear congruential generator used to draw Latin-hypercube starts (default 1).
   */
  std::uint64_t seed = 1;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimum value difference required for a later run of the same convergence kind to replace the kept run and to separate distinct optima (default \f$10^{-5}\f$).
   */
  double distinct_tolerance = 1e-5;         // values closer than this are one optimum

  /**
   * \ingroup optimizers_quasi_newton
   * \brief Generates `x0` followed by `start_count - 1` Latin-hypercube points around `x0`.
   *
   * \param x0 Center starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Vector of generated starting points, deterministic for a given `seed`.
   * \see \ref optimizers_quasi_newton, multistart_optimizer
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
   * \ingroup optimizers_quasi_newton
   * \brief Runs the inner optimizer from all starts and returns a full `multistart_report`.
   *
   * \tparam O Objective type satisfying `optimizer_for<Inner, O>`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$ (ignored when `starts` is non-empty).
   * \return A `multistart_report` containing the aggregated kept run, summary counts, and all individual run reports.
   * \see \ref optimizers_quasi_newton, multistart_report, run()
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
      if (c_ok != b_ok ? c_ok : c.value < b.value - distinct_tolerance) kept = index;
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
   * \ingroup optimizers_quasi_newton
   * \brief Runs the inner optimizer from all starts and returns the kept run's `optimizer_report`.
   *
   * Sums `iterations`, evaluation counts, and `elapsed_seconds` over all starts.
   *
   * \tparam O Objective type satisfying `optimizer_for<Inner, O>`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$ (ignored when `starts` is non-empty).
   * \return Kept `optimizer_report` with aggregated cost fields.
   * \see \ref optimizers_quasi_newton, run_multistart(), minimize()
   */
  template <class O>
    requires optimizer_for<Inner, O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    return run_multistart(objective, x0).best;
  }
};

}  // namespace laplace::optimizers
