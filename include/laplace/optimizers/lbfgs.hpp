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
 * \brief Limited-memory BFGS quasi-Newton optimizer keeping the last `memory` curvature pairs.
 *
 * `lbfgs_optimizer<Rule, LineSearch>` minimizes an unconstrained objective
 * modeling gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is the
 * parameter dimension, without forming the \f$n \times n\f$ inverse-Hessian
 * matrix \f$B\f$. Instead, it retains the last \f$m\f$ = `memory` curvature
 * pairs and computes the search direction in \f$O(m n)\f$ work and memory.
 * Everything runs on the host only; nothing is marked `LAPLACE_HD`. A run is
 * started with `minimize(lbfgs_optimizer<>{}, f, x0)`.
 *
 * ### Algorithm
 *
 * Let \f$x_k \in \mathbb{R}^n\f$ be the iterate at step \f$k\f$,
 * \f$g_k = \nabla f(x_k)\f$ its gradient, \f$d_k\f$ the search direction, and
 * \f$t_k\f$ the step length accepted by the line search started at \f$t = 1\f$.
 * Set \f$s_k = x_{k+1} - x_k = t_k d_k\f$, \f$y_k = g_{k+1} - g_k\f$, and
 * \f$\rho_i = 1 / (s_i^\top y_i)\f$. Given the \f$\ell \le m\f$ stored pairs
 * \f$(s_i, y_i)\f$ for \f$i = 1, \dots, \ell\f$ ordered from oldest to newest,
 * the two-loop recursion (Nocedal and Wright, 2006, Algorithms 7.4 and 7.5)
 * computes \f$d = -B g\f$ from \f$q \leftarrow g\f$ as
 * \f[
 * \alpha_i = \rho_i\, s_i^\top q, \qquad q \leftarrow q - \alpha_i y_i \qquad (i = \ell, \dots, 1),
 * \f]
 * followed by the initial scaling
 * \f[
 * r \leftarrow \gamma q, \qquad \gamma = \frac{s_\ell^\top y_\ell}{y_\ell^\top y_\ell},
 * \f]
 * and the forward pass
 * \f[
 * \beta = \rho_i\, y_i^\top r, \qquad r \leftarrow r + s_i (\alpha_i - \beta) \qquad (i = 1, \dots, \ell),
 * \f]
 * returning \f$d = -r\f$. The accepted pairs are stored in a ring of
 * \f$m\f$ = `memory` slots, so once the ring is full each new pair replaces the
 * oldest stored pair.
 *
 * Before any pair has been accepted, and after a reset, the model uses the
 * same scaled steepest-descent direction \f$d = -\min\bigl(1,\, 1 / \lVert g\rVert_\infty\bigr) g\f$
 * as bfgs_optimizer. Pair acceptance follows the same curvature condition as
 * bfgs_optimizer, but `lbfgs_optimizer` never resets on consecutive skips: a
 * skipped pair leaves the stored pairs untouched. Like bfgs_optimizer, it
 * resets by clearing all stored pairs whenever the computed direction is not
 * finite or \f$g^\top d \ge 0\f$, and recomputes \f$d\f$ from the reset model.
 *
 * Each iteration searches along \f$d_k\f$ from \f$t = 1\f$ using `LineSearch`,
 * which defaults to wolfe_line_search (see \ref optimizers "the optimizers page").
 * With armijo_line_search (Dennis and Schnabel, 1996), the gradient is
 * evaluated once more at each accepted point.
 *
 * ### Stationarity and statuses
 *
 * The stationarity decrement \f$\delta_k\f$ (using the next two-loop direction
 * \f$d_k\f$ and the largest \f$s^\top y / y^\top y\f$ over all accepted pairs),
 * its one-step confirmation, the zero-gradient start behaviour, the
 * termination statuses, and the `optimizer_report` fields
 * (`curvature_evaluations` equal to \f$0\f$ and `final_factorization` equal to
 * `curvature_factorization::not_evaluated`) are those of bfgs_optimizer.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration; tolerance \f$10^{-6}\f$ in the objective's units, accepting any rule matching quasi_newton_provides |
 * | `line_search` | `wolfe_line_search{}` | Line search along each direction from step \f$1\f$, with \f$c_1 = 10^{-4}\f$ and \f$c_2 = 0.9\f$ |
 * | `memory` | \f$10\f$ | Number \f$m\f$ of curvature pairs kept in the ring; more pairs generally take fewer iterations on ill-conditioned problems at higher cost per iteration |
 * | `max_iterations` | \f$1000\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^5\f$ | Value-evaluation budget checked before each line search, so a run can exceed it by one search |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold \f$\tau\f$ for accepting pairs; larger values skip more updates |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration |
 *
 * On the \f$10\f$-dimensional quadratic with condition number
 * \f$\kappa = 10^6\f$ (measured with gcc on one thread and default tolerance
 * \f$10^{-6}\f$), `memory` values of \f$5\f$, \f$10\f$, and \f$20\f$ take the
 * \f$1000\f$-iteration budget, \f$249\f$ iterations, and \f$90\f$ iterations,
 * respectively.
 *
 * ### Cost and measured behaviour
 *
 * Each iteration performs \f$O(m n)\f$ work and stores \f$2 m n\f$ `double`
 * values for the \f$m\f$ pairs; each Wolfe trial evaluates the objective value
 * and gradient once. Measured with gcc on one thread, Rosenbrock with
 * \f$n = 100\f$ takes 94 microseconds (36 iterations, 43 value evaluations),
 * compared with 356 microseconds for bfgs_optimizer.
 *
 * With the default `memory` equal to \f$10\f$, L-BFGS converges on the
 * standard test problems (with \f$f^*\f$ shifted by \f$0\f$, \f$1\f$,
 * \f$10^3\f$, and \f$10^6\f$, except Wood's function) with
 * \f$f - f^* \le 3.4 \times 10^{-9}\f$, and on \f$10\f$-dimensional quadratics
 * up to \f$\kappa = 10^6\f$ (in 30 to 249 iterations with
 * \f$f - f^* \le 2.3 \times 10^{-7}\f$). At \f$\kappa = 10^8\f$ and
 * \f$10^{10}\f$ it is declared converged about \f$10^{-2}\f$ above \f$f^*\f$
 * because the remaining error lies along directions that no stored or past
 * pair has sampled. On Wood's function it stops near the saddle point at
 * \f$f = 7.877\f$, where a positive definite model cannot see negative
 * curvature. It does not converge on a non-smooth objective such as
 * \f$\sum_i |x_i|\f$ (tested for \f$n = 2, 4, 10\f$ under both line searches).
 *
 * ### Example
 *
 * \code
 * lbfgs_optimizer<> opt;
 * opt.memory = 20;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, quasi_newton_provides>`.
 * \tparam LineSearch Line search type such as `wolfe_line_search` or `armijo_line_search`.
 * \see \ref optimizers_quasi_newton, bfgs_optimizer, quasi_newton_provides, minimize(), optimizer_report
 */
// L-BFGS with `memory` pairs. Settings are members; minimize(lbfgs_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = wolfe_line_search>
  requires stop_rule_for<Rule, quasi_newton_provides>
struct lbfgs_optimizer {
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Line search instance used along each search direction from step \f$1\f$ (default `wolfe_line_search{}`).
   */
  LineSearch line_search{};
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Maximum number of recent curvature pairs \f$(s, y)\f$ retained by the two-loop recursion (default \f$10\f$).
   */
  int memory = 10;  // pairs kept
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default \f$1000\f$).
   */
  int max_iterations = 1000;
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Value-evaluation budget checked before each line search (default \f$10^5\f$; a run may exceed it by one search).
   */
  long max_evaluations = 100000;       // checked before each line search
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimum cosine threshold \f$\tau\f$ for accepting a curvature pair when \f$s^\top y > \tau \lVert s\rVert_2 \lVert y\rVert_2\f$ (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;  // the pair is used when s'y > tol |s| |y|
  /**
   * \ingroup optimizers_quasi_newton
   * \brief Optional trace callback invoked once per accepted iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_quasi_newton
   * \brief Minimizes `objective` starting from `x0` using L-BFGS.
   *
   * See lbfgs_optimizer for the two-loop recursion, stationarity confirmation, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_quasi_newton, lbfgs_optimizer, minimize(), optimizer_report
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
