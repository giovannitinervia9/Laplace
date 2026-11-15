#pragma once

// L-BFGS (Nocedal and Wright 2006, Algorithms 7.4 and 7.5): the BFGS inverse
// Hessian never formed, only the last `memory` curvature pairs kept, and
// d = -B g computed by the two-loop recursion with the initial matrix
// gamma I, gamma = s'y / y'y of the most recent pair. O(memory n) memory and
// work per iteration. The loop, the skip rule, the stationarity measure (with
// d the next two-loop direction) and its confirmation are those of BFGS
// (bfgs.hpp). L-BFGS has no reset after skips: a skipped pair leaves the
// stored ones untouched, and the oldest pair leaves when a new one enters.
//
// Limits measured on the test battery: with 10 pairs, on quadratics of
// n = 10 with kappa 1e8 and 1e10 the run is declared converged about 1e-2
// above f* (the remaining error lies along weak directions that no stored or
// past pair has sampled, so neither B nor gamma_max shows it); on Wood's function it stops
// near the saddle at f = 7.877, where a positive definite B cannot see the
// negative curvature. BFGS reaches the minimum in both cases.

#include "laplace/optimizers/bfgs.hpp"

#include <vector>

namespace laplace::optimizers {

namespace detail {

// The last `memory` pairs in a ring: slot (first + i) % memory, i = 0 oldest.
struct lbfgs_inverse_model {
  int n = 0, memory = 0, count = 0, first = 0;
  std::vector<double> s, y, rho;  // memory x n each, and memory
  mutable std::vector<double> alpha;

  lbfgs_inverse_model(int size, int pairs)
      : n(size), memory(pairs), s(size * pairs), y(size * pairs), rho(pairs), alpha(pairs) {}
  void reset() { count = 0, first = 0; }
  void direction(const optimizer_point& g, const optimizer_vector& d) const {
    if (count == 0) return scaled_steepest_descent(g, d);
    for (int i = 0; i < n; ++i) d(i) = g(i);  // q
    auto slot = [&](int i) { return (first + i) % memory; };
    for (int i = count - 1; i >= 0; --i) {
      const double* si = &s[slot(i) * n];
      const double* yi = &y[slot(i) * n];
      double a = 0.0;
      for (int j = 0; j < n; ++j) a += si[j] * d(j);
      a *= rho[slot(i)];
      alpha[i] = a;
      for (int j = 0; j < n; ++j) d(j) -= a * yi[j];
    }
    {
      const double* yl = &y[slot(count - 1) * n];
      double yy = 0.0;
      for (int j = 0; j < n; ++j) yy += yl[j] * yl[j];
      const double gamma = 1.0 / (rho[slot(count - 1)] * yy);  // s'y / y'y
      for (int j = 0; j < n; ++j) d(j) *= gamma;  // r
    }
    for (int i = 0; i < count; ++i) {
      const double* si = &s[slot(i) * n];
      const double* yi = &y[slot(i) * n];
      double beta = 0.0;
      for (int j = 0; j < n; ++j) beta += yi[j] * d(j);
      beta *= rho[slot(i)];
      for (int j = 0; j < n; ++j) d(j) += si[j] * (alpha[i] - beta);
    }
    for (int i = 0; i < n; ++i) d(i) = -d(i);
  }
  void update(const std::vector<double>& sv, const std::vector<double>& yv, double sy) {
    int k;
    if (count < memory) {
      k = (first + count) % memory;
      ++count;
    } else {
      k = first;
      first = (first + 1) % memory;
    }
    std::copy(sv.begin(), sv.end(), s.begin() + k * n);
    std::copy(yv.begin(), yv.end(), y.begin() + k * n);
    rho[k] = 1.0 / sy;
  }
};

}  // namespace detail

/**
 * \ingroup optimizers_quasi_newton
 * \brief Limited-memory BFGS optimizer keeping the last `memory` curvature pairs.
 *
 * Minimizes an objective modeling `gradient_objective` using \f$O(m n)\f$ memory and work per iteration via the two-loop recursion with \f$m\f$ stored pairs, with `wolfe_line_search` and `decrement_stop_rule` by default.
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, quasi_newton_provides>`.
 * \tparam LineSearch Line search type such as `wolfe_line_search` or `armijo_line_search`.
 * \see \ref optimizers_quasi_newton, bfgs_optimizer, quasi_newton_provides, minimize()
 */
// L-BFGS with `memory` pairs. Settings are members; minimize(lbfgs_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = wolfe_line_search>
  requires stop_rule_for<Rule, quasi_newton_provides>
struct lbfgs_optimizer {
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Stopping rule instance queried at each iteration.
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Line search instance used along each search direction from step \f$1\f$.
   */
  LineSearch line_search{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Maximum number of recent curvature pairs \f$(s, y)\f$ retained by the two-loop recursion (default 10).
   */
  int memory = 10;  // pairs kept
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default 1000).
   */
  int max_iterations = 1000;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Value-evaluation budget checked before each line search, default \f$10^5\f$ (a run may exceed it by one search).
   */
  long max_evaluations = 100000;       // checked before each line search
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimum cosine threshold \f$\tau\f$ for accepting a curvature pair when \f$s^\top y > \tau \lVert s \rVert \lVert y \rVert\f$ (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;  // the pair is used when s'y > tol |s| |y|
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Optional trace callback invoked once per accepted iteration.
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimizes `objective` starting from `x0` using L-BFGS.
   *
   * Returns `optimizer_status::start_not_admissible` if the initial value or gradient is non-finite, and sets `curvature_evaluations` to \f$0\f$.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_quasi_newton, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    detail::lbfgs_inverse_model model(static_cast<int>(x0.extent(0)), memory);
    // no reset after skips: max_skipped_updates beyond the budget
    return detail::quasi_newton_run(model, stop_rule, line_search, max_iterations, max_evaluations,
                                    curvature_tolerance, max_iterations + 1, trace, objective, x0);
  }
};

}  // namespace laplace::optimizers
