#pragma once

// Newton's method with a line search (note/optimizers.md, decision 4).
//
// At each point x the objective gives f, g = grad f and the symmetric
// curvature H. When H is positive definite (its Cholesky factor exists) the
// direction is the Newton step d = -H^-1 g, and the decrement
// (1/2) g' H^-1 g = -(1/2) g' d is the stationarity measure read by the stop
// rule (decision 5). Otherwise the decrement is NaN, so that no point with
// an indefinite or singular H is declared converged by the decrement, and
// the direction comes from the eigenvalue decomposition H = V diag(w) V':
//
//   d = -sum_i (v_i' g) / max(|w_i|, floor) v_i,    floor = eigen_floor max_i |w_i|,
//
// plus, when the smallest eigenvalue w_1 is below -floor, a unit step along
// v_1 with the sign that makes g' v_1 <= 0 (a direction of negative
// curvature, which leaves a saddle where g' v_1 = 0). The length of this
// direction is set by the floor, not by the curvature of the objective, so
// it is scaled down to a largest component of max_modified_step (default 1;
// without such a cap a line search spent 15 to 23 backtracks per iteration
// on an indefinite criterion, and on the test battery the cap of 1 saves
// iterations on Beale, helical valley, Wood and Rosenbrock from (0, 1) but
// costs 184 iterations against 5 on Brown badly scaled). Where g' d
// is not negative (g = 0 at a saddle) the step along the negative curvature
// is accepted on f(x + s d) - f(x) <= c1 s^2 d' H d / 2, halving s.
//
// Every direction is then scaled down to a largest component of max_step
// (Dennis and Schnabel's maxstep, mgcv's maxNstep): a quadratic model can
// propose a step far beyond where it holds. A step is never lengthened.
//
// The line search (Armijo by default) starts at s = 1. A failed search ends
// the run: "stalled" when the step fell below the rounding of x, "line
// search failed" otherwise. Success comes only from the stop rule.

#include "laplace/dense/dense.hpp"
#include "laplace/optimizers/dense_curvature_solver.hpp"
#include "laplace/optimizers/line_search.hpp"
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

// What the second-order methods give a stop rule: everything.
inline constexpr unsigned second_order_provides = stop_needs_gradient | stop_needs_decrement | stop_needs_previous;

inline optimizer_status status_from_verdict(stop_verdict v) {
  return v == stop_verdict::converged                 ? optimizer_status::converged
         : v == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                      : optimizer_status::stopped_on_change;
}

inline double max_abs(const optimizer_point& v) {
  double m = 0.0;
  for (std::size_t i = 0; i < v.extent(0); ++i) m = std::max(m, std::abs(v(i)));
  return m;  // NaN entries are skipped by max; callers test finiteness of the whole
}

// d <- d * limit / max|d_i| when max|d_i| > limit; never lengthens d.
inline void cap_largest_component(const optimizer_vector& d, double limit) {
  const double m = max_abs(d);
  if (std::isfinite(m) && m > limit)
    for (std::size_t i = 0; i < d.extent(0); ++i) d(i) *= limit / m;
}

// d' H d from the lower triangle of H.
inline double curvature_along(const dense_curvature_matrix& H, const optimizer_point& d) {
  const int n = static_cast<int>(d.extent(0));
  double s = 0.0;
  for (int j = 0; j < n; ++j) {
    s += H(j, j) * d(j) * d(j);
    for (int i = j + 1; i < n; ++i) s += 2.0 * H(i, j) * d(i) * d(j);
  }
  return s;
}

// The direction of the eigen route (header comment); false when H has no
// finite eigendecomposition.
class eigen_modified_direction {
 public:
  explicit eigen_modified_direction(int n)
      : eigenvalues_("laplace::optimizers::eigen_modified_direction::w", n),
        eigenvectors_("laplace::optimizers::eigen_modified_direction::V", n, n) {}

  bool compute(const dense_curvature_matrix& H, const optimizer_point& g, double eigen_floor, double cap,
               const optimizer_vector& d) {
    const int n = static_cast<int>(g.extent(0));
    if (!laplace::dense::symmetric_eigen_decomposition(H, eigenvalues_, eigenvectors_).success) return false;
    const double floor = std::max(eigen_floor * max_abs(eigenvalues_), std::numeric_limits<double>::min());
    for (int i = 0; i < n; ++i) d(i) = 0.0;
    for (int k = 0; k < n; ++k) {
      double gk = 0.0;
      for (int i = 0; i < n; ++i) gk += eigenvectors_(i, k) * g(i);
      const double c = -gk / std::max(std::abs(eigenvalues_(k)), floor);
      for (int i = 0; i < n; ++i) d(i) += c * eigenvectors_(i, k);
    }
    if (eigenvalues_(0) < -floor) {
      double g1 = 0.0;
      for (int i = 0; i < n; ++i) g1 += eigenvectors_(i, 0) * g(i);
      const double sign = g1 > 0.0 ? -1.0 : 1.0;
      for (int i = 0; i < n; ++i) d(i) += sign * eigenvectors_(i, 0);
    }
    cap_largest_component(d, cap);
    return max_abs(d) > 0.0 || n == 0;
  }

 private:
  Kokkos::View<double*, Kokkos::HostSpace> eigenvalues_;
  dense_curvature_matrix eigenvectors_;
};

// Backtracking on the second-order decrease f(x + s d) - f <= c1 s^2 (d'Hd)/2
// for a direction of negative curvature along which g' d = 0.
template <value_objective O>
line_search_result negative_curvature_search(O& objective, const optimizer_point& x, double value,
                                              const optimizer_point& d, double curvature, double c1,
                                              int max_evaluations, const optimizer_vector& x_trial) {
  line_search_result r;
  double s = 1.0;
  while (r.value_evaluations < max_evaluations) {
    r.step = s;
    if (!trial_point(x, d, s, x_trial)) {
      r.outcome = line_search_outcome::step_below_rounding;
      return r;
    }
    const double ft = objective.value(optimizer_point(x_trial));
    ++r.value_evaluations;
    if (std::isfinite(ft) && ft - value <= 0.5 * c1 * s * s * curvature) {
      r.outcome = line_search_outcome::accepted;
      r.value = ft;
      return r;
    }
    if (!std::isfinite(ft)) ++r.refused_evaluations;
    s *= 0.5;
  }
  r.outcome = line_search_outcome::evaluation_budget;
  return r;
}

inline double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace detail

/**
 * \ingroup optimizers_second_order
 * \brief Newton's method with an eigenvalue floor, a maximum step and a line search.
 *
 * Minimizes an objective modeling `curvature_objective` on the host using the Cholesky Newton step
 * \f$d = -H^{-1} g\f$ where \f$H\f$ is positive definite, and an eigenvalue-modified direction
 * with a negative-curvature component otherwise.
 *
 * \tparam StopRule Stopping rule accepting the gradient, decrement, and previous point and value;
 *   `decrement_stop_rule` by default.
 * \tparam LineSearch Line search with the signature of `armijo_line_search` (`armijo_line_search` by default).
 * \see \ref optimizers_second_order, \ref optimizers, trust_region_optimizer, minimize()
 */
template <class StopRule = decrement_stop_rule, class LineSearch = armijo_line_search>
  requires stop_rule_for<StopRule, detail::second_order_provides>
struct newton_optimizer {
  /**
   * \ingroup optimizers_second_order
   * \brief Stopping rule instance queried at the start and at each new point.
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_second_order
   * \brief Line search instance invoked from step \f$1\f$ along each direction.
   */
  LineSearch line_search{};
  /**
   * \ingroup optimizers_second_order
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default 200).
   */
  int max_iterations = 200;
  /**
   * \ingroup optimizers_second_order
   * \brief Value-evaluation budget checked before each step, default \f$10^5\f$ (a run may exceed it by one search).
   */
  long max_value_evaluations = 100000;
  /**
   * \ingroup optimizers_second_order
   * \brief Maximum component magnitude allowed for any search direction (default \f$+\infty\f$, no bound).
   */
  // largest component of a step (max norm, in the units of x); infinity: no bound
  double max_step = std::numeric_limits<double>::infinity();
  /**
   * \ingroup optimizers_second_order
   * \brief Relative eigenvalue floor \f$\eta\f$ used when \f$H\f$ has no Cholesky factor (default \f$10^{-8}\f$).
   */
  // eigenvalues are floored at eigen_floor times the largest |eigenvalue|
  double eigen_floor = 1e-8;
  /**
   * \ingroup optimizers_second_order
   * \brief Maximum component magnitude allowed for an eigenvalue-modified direction (default \f$1\f$).
   */
  // largest component of a direction from the eigen route, whose length is set
  // by the floor and not by the curvature
  double max_modified_step = 1.0;
  /**
   * \ingroup optimizers_second_order
   * \brief Optional trace callback invoked at every new point with value, gradient, decrement, and accepted step length.
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_second_order
   * \brief Minimizes `objective` starting from `x0` by Newton's method.
   *
   * Returns `optimizer_status::start_not_admissible` if \f$f(x_0)\f$ is not finite, `stalled` if the
   * line search fails with `step_below_rounding`, `line_search_failed` on any other search failure,
   * `iteration_budget` or `evaluation_budget` on budget exhaustion, or the status from `stop_rule`.
   * Sets `decrement` to quiet NaN wherever \f$H\f$ has no Cholesky factor and records the last
   * factorization outcome in `final_factorization`.
   *
   * \tparam O Objective type satisfying `curvature_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_second_order, minimize(), optimizer_report
   */
  template <curvature_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    optimizer_report report;
    optimizer_vector x("x", n), g("g", n), d("d", n), x_trial("x_trial", n), x_previous("x_previous", n);
    typename O::curvature_type H("H", n, n);
    typename curvature_solver_for<typename O::curvature_type>::type solver(n);
    detail::eigen_modified_direction eigen(n);
    Kokkos::deep_copy(x, x0);

    double f = objective.value_gradient_curvature(x, g, H);
    ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
    double f_previous = f, last_step = 0.0;
    if (!std::isfinite(f)) {
      report.status = optimizer_status::start_not_admissible;
      report.point = x, report.value = f, report.gradient = g;
      report.elapsed_seconds = detail::seconds_since(start);
      return report;
    }

    for (int k = 0;; ++k) {
      // the Newton step and the decrement at x
      report.final_factorization = solver.factorize(H);
      bool newton_step = report.final_factorization == curvature_factorization::positive_definite &&
                         solver.newton_direction(g, d);
      double slope = newton_step ? detail::dot(g, d) : 0.0;
      newton_step = newton_step && slope <= 0.0;  // 0 only at g = 0
      report.decrement = newton_step ? 0.5 * std::abs(slope) : std::numeric_limits<double>::quiet_NaN();
      if (k > 0 && trace) trace(optimizer_trace_point{k, x, f, g, report.decrement, last_step});

      stop_rule_input in;
      in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
      in.has_previous = k > 0, in.previous_value = f_previous, in.previous_point = x_previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none) {
        report.status = detail::status_from_verdict(verdict);
        break;
      }
      if (k >= max_iterations) {
        report.status = optimizer_status::iteration_budget;
        break;
      }
      if (report.value_evaluations >= max_value_evaluations) {
        report.status = optimizer_status::evaluation_budget;
        break;
      }

      if (!newton_step && !eigen.compute(H, g, eigen_floor, max_modified_step, d)) {
        // no usable curvature: steepest descent with a largest component of 1
        for (int i = 0; i < n; ++i) d(i) = -g(i);
        detail::cap_largest_component(d, 1.0);
      }
      detail::cap_largest_component(d, max_step);

      line_search_result s = line_search.search(objective, x, f, g, d, 1.0, x_trial);
      if (s.outcome == line_search_outcome::not_descent_direction && !newton_step) {
        const double curvature = detail::curvature_along(H, d);
        if (curvature < 0.0) {
          report.value_evaluations += s.value_evaluations;
          s = detail::negative_curvature_search(objective, x, f, d, curvature, line_search.sufficient_decrease,
                                                line_search.max_evaluations, x_trial);
        }
      }
      report.value_evaluations += s.value_evaluations;
      report.gradient_evaluations += s.gradient_evaluations;
      if (!s.accepted()) {
        report.status = s.outcome == line_search_outcome::step_below_rounding ? optimizer_status::stalled
                                                                               : optimizer_status::line_search_failed;
        break;
      }

      Kokkos::deep_copy(x_previous, x);
      Kokkos::deep_copy(x, x_trial);
      f_previous = f;
      f = objective.value_gradient_curvature(x, g, H);
      ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
      report.iterations = k + 1;
      last_step = s.step;
    }

    report.point = x, report.value = f, report.gradient = g;
    report.elapsed_seconds = detail::seconds_since(start);
    return report;
  }
};

}  // namespace laplace::optimizers
