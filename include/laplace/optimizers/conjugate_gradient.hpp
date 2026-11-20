#pragma once

// Nonlinear conjugate gradient (note/optimizers.md, section 13, decision 2;
// Nocedal and Wright 2006, section 5.2; Hager and Zhang 2005). The loop,
// the stationarity measure with its two-iteration confirmation, the statuses
// and the reset on a stalled search are those of BFGS (bfgs.hpp,
// detail::quasi_newton_run); CG enters it as a model of the direction.
//
// - p_k = -g_k + beta_k p_{k-1}, with beta_k chosen by a member (FR, PR+, HS,
//   DY, and Hager-Zhang with the lower bound eta_k as the default). y =
//   g_k - g_{k-1}.
// - Restarts (beta = 0): Powell's test |g_k'g_{k-1}| >= 0.2 g_k'g_k, every
//   restart_interval directions (0: n), and whenever beta is not finite or
//   p_{k-1}'y <= 0. A direction that is not a descent direction resets the
//   model through the loop, which then searches along the scaled steepest
//   descent.
// - The loop searches from step 1, so the model returns d_k = a_k p_k scaled by
//   the initial step of Nocedal and Wright (3.60): a_k = tau_{k-1}
//   g_{k-1}'p_{k-1} / g_k'p_k, tau_{k-1} the step accepted along p_{k-1}
//   (s = tau p). The first-order decrease predicted for the step is then that
//   of the previous step.
// - Stationarity: (1/2) max(|g'd|, gamma g'g)
//   with d the next direction and gamma the largest s's/s'y over the accepted
//   steps (floor_from_step), not BFGS's s'y/y'y. On the quadratics with
//   kappa 1e6 (n = 100) and 1e8 (n = 10) BFGS's floor declared every beta
//   converged up to 8.8e-3 above f*; this one gives no false success on the
//   battery (docs/numerics.md, the section on the conjugate gradient).
// - A secant pair that fails the curvature test (s'y > tol |s| |y|, which the
//   strong Wolfe search guarantees along a descent direction) restarts the
//   model: max_skipped_updates = 1.

#include "laplace/optimizers/bfgs.hpp"

#include <cmath>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_first_order
 * \brief The formula for \f$\beta_k\f$ in conjugate_gradient_optimizer.
 *
 * Selects which update rule combines the current gradient and gradient
 * difference with the previous search direction in nonlinear conjugate
 * gradient optimization.
 *
 * \see \ref optimizers_first_order, conjugate_gradient_optimizer
 */
// Which beta combines the new gradient with the previous direction.
enum class conjugate_gradient_beta {
  fletcher_reeves,     ///< \f$\beta = g_k^\top g_k / g_{k-1}^\top g_{k-1}\f$.
  polak_ribiere_plus,  ///< \f$\beta = \max(0,\, g_k^\top y / g_{k-1}^\top g_{k-1})\f$.
  hestenes_stiefel,    ///< \f$\beta = g_k^\top y / p_{k-1}^\top y\f$.
  dai_yuan,            ///< \f$\beta = g_k^\top g_k / p_{k-1}^\top y\f$.
  hager_zhang          ///< CG_DESCENT's \f$\beta\f$ with the lower bound \f$\eta_k\f$ (default).
};

namespace detail {

// The direction model of CG for quasi_newton_run: it keeps the previous
// gradient, the previous pure direction p and the step tau taken along it.
struct conjugate_gradient_model {
  // the floor of the stationarity measure reads max s's/s'y (quasi_newton_run)
  static constexpr bool floor_from_step = true;
  int n = 0;
  conjugate_gradient_beta formula;
  int restart_interval;  // directions between periodic restarts
  double powell_threshold;
  bool has_previous = false;  // g_prev, p_prev and g_prev'p_prev hold the last direction
  bool has_step = false;      // tau holds the step accepted along p_prev
  int since_restart = 0;
  double tau = 0.0, gp_previous = 0.0;
  std::vector<double> g_previous, p_previous, p;
  int restarts = 0;  // counted for the tests and the report of the measurements

  conjugate_gradient_model(int size, conjugate_gradient_beta beta, int interval, double powell)
      : n(size), formula(beta), restart_interval(interval > 0 ? interval : size), powell_threshold(powell),
        g_previous(size), p_previous(size), p(size) {}

  void reset() { has_previous = false, has_step = false; }
  bool has_pairs() const { return has_previous; }

  // beta from g (new), g_previous, p_previous; NaN asks for a restart
  double beta(const optimizer_point& g) const {
    double gg = 0.0, gy = 0.0, gpgp = 0.0, py = 0.0, yy = 0.0, pg = 0.0, pp = 0.0, ggp = 0.0;
    for (int i = 0; i < n; ++i) {
      const double y = g(i) - g_previous[i];
      gg += g(i) * g(i), gy += g(i) * y, gpgp += g_previous[i] * g_previous[i];
      py += p_previous[i] * y, yy += y * y, pg += p_previous[i] * g(i), pp += p_previous[i] * p_previous[i];
      ggp += g(i) * g_previous[i];
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (std::abs(ggp) >= powell_threshold * gg) return nan;  // Powell's restart
    switch (formula) {
      case conjugate_gradient_beta::fletcher_reeves: return gg / gpgp;
      case conjugate_gradient_beta::polak_ribiere_plus: return std::max(0.0, gy / gpgp);
      case conjugate_gradient_beta::hestenes_stiefel: return py > 0.0 ? gy / py : nan;
      case conjugate_gradient_beta::dai_yuan: return py > 0.0 ? gg / py : nan;
      case conjugate_gradient_beta::hager_zhang: {
        if (!(py > 0.0)) return nan;
        // Hager and Zhang (2005): beta_N and the lower bound eta_k, eta = 0.01
        const double b = (gy - 2.0 * yy * pg / py) / py;
        const double eta = -1.0 / (std::sqrt(pp) * std::min(0.01, std::sqrt(gpgp)));
        return std::max(b, eta);
      }
    }
    return nan;
  }

  void direction(const optimizer_point& g, const optimizer_vector& d) {
    if (!has_previous || !has_step) {
      scaled_steepest_descent(g, d);
      for (int i = 0; i < n; ++i) p[i] = -g(i);  // p_0 = -g: FR and PR+ are not invariant to its scale
      since_restart = 0;
    } else {
      double b = ++since_restart >= restart_interval ? std::numeric_limits<double>::quiet_NaN() : beta(g);
      if (!std::isfinite(b)) b = 0.0, since_restart = 0, ++restarts;
      double gp = 0.0;
      for (int i = 0; i < n; ++i) {
        p[i] = -g(i) + b * p_previous[i];
        gp += g(i) * p[i];
      }
      // Nocedal and Wright (3.60); a non-descent p gives a non-descent d,
      // which the loop answers with a reset
      double a = tau * gp_previous / gp;
      if (!std::isfinite(a) || a <= 0.0) a = 1.0;
      for (int i = 0; i < n; ++i) d(i) = a * p[i];
    }
    for (int i = 0; i < n; ++i) g_previous[i] = g(i), p_previous[i] = p[i];
    gp_previous = 0.0;
    for (int i = 0; i < n; ++i) gp_previous += g(i) * p[i];
    has_previous = true, has_step = false;
  }

  // an accepted pair: the step along p_previous, s = tau p_previous
  void update(const std::vector<double>& s, const std::vector<double>&, double) {
    double sp = 0.0, pp = 0.0;
    for (int i = 0; i < n; ++i) sp += s[i] * p_previous[i], pp += p_previous[i] * p_previous[i];
    tau = sp / pp;
    has_step = std::isfinite(tau) && tau > 0.0;
  }
};

}  // namespace detail

/**
 * \ingroup optimizers_first_order
 * \brief Nonlinear conjugate gradient optimizer with a choice of five \f$\beta\f$ formulas.
 *
 * `conjugate_gradient_optimizer<Rule, LineSearch>` minimizes an unconstrained
 * objective modeling gradient_objective in \f$\mathbb{R}^n\f$, where \f$n\f$ is
 * the parameter dimension. Everything runs on the host only; nothing is marked
 * `LAPLACE_HD`. A run is started with
 * `minimize(conjugate_gradient_optimizer<>{}, f, x0)`. The template parameter
 * `Rule` defaults to `decrement_stop_rule` and accepts any rule satisfying
 * `stop_rule_for<Rule, quasi_newton_provides>`; `LineSearch` defaults to
 * `wolfe_line_search`.
 *
 * ### Direction and restarts
 *
 * Let \f$x_k \in \mathbb{R}^n\f$ be the iterate at step \f$k\f$,
 * \f$g_k = \nabla f(x_k)\f$ its gradient, \f$y = g_k - g_{k-1}\f$ the gradient
 * difference, and \f$\lVert \cdot \rVert\f$ the Euclidean norm. Each iteration
 * forms \f$p_k = -g_k + \beta_k p_{k-1}\f$ (Nocedal and Wright, 2006, section
 * 5.2), with \f$\beta_k\f$ chosen by `beta` from conjugate_gradient_beta:
 * - `conjugate_gradient_beta::fletcher_reeves`: \f$\beta = g_k^\top g_k / g_{k-1}^\top g_{k-1}\f$ (Fletcher and Reeves, 1964).
 * - `conjugate_gradient_beta::polak_ribiere_plus`: \f$\beta = \max(0,\, g_k^\top y / g_{k-1}^\top g_{k-1})\f$ (Polak and Ribière, 1969; clamped at zero as in Nocedal and Wright, 2006, eq. (5.45)).
 * - `conjugate_gradient_beta::hestenes_stiefel`: \f$\beta = g_k^\top y / p_{k-1}^\top y\f$ (Hestenes and Stiefel, 1952).
 * - `conjugate_gradient_beta::dai_yuan`: \f$\beta = g_k^\top g_k / p_{k-1}^\top y\f$ (Dai and Yuan, 1999).
 * - `conjugate_gradient_beta::hager_zhang` (the default): \f$\beta = \max(\beta^N, \eta_k)\f$, where
 *   \f[
 *   \beta^N = \frac{\bigl(y - 2 p_{k-1} \lVert y\rVert^2 / p_{k-1}^\top y\bigr)^\top g_k}{p_{k-1}^\top y}, \qquad
 *   \eta_k = -\frac{1}{\lVert p_{k-1}\rVert \min(0.01,\, \lVert g_{k-1}\rVert)}
 *   \f]
 *   (Hager and Zhang, 2005).
 *
 * A restart sets \f$\beta = 0\f$, so \f$p_k = -g_k\f$. Restarts occur on
 * Powell's test \f$|g_k^\top g_{k-1}| \ge c\, g_k^\top g_k\f$ with \f$c\f$
 * equal to `powell_restart` (default \f$0.2\f$; \f$+\infty\f$ disables it)
 * (Powell, 1977), every `restart_interval` directions (default \f$0\f$, meaning
 * \f$n\f$), whenever \f$\beta\f$ is not finite, and for `hestenes_stiefel`,
 * `dai_yuan`, and `hager_zhang` whenever \f$p_{k-1}^\top y \le 0\f$. A direction that is
 * not finite or not a descent direction (\f$g_k^\top d_k \ge 0\f$), or a step
 * \f$s = x_k - x_{k-1}\f$ whose pair fails the curvature test
 * \f$s^\top y > \mathrm{tol} \lVert s\rVert \lVert y\rVert\f$ (where
 * \f$\mathrm{tol}\f$ is `curvature_tolerance`, \f$10^{-10}\f$; a strong Wolfe
 * step along a descent direction always passes it), restarts the method along
 * the scaled steepest-descent direction
 * \f$d = -\min\bigl(1,\, 1/\lVert g\rVert_\infty\bigr) g\f$ of bfgs_optimizer
 * (and on a stall while holding a previous pair, the loop resets once and
 * repeats the search along that direction). After a restart, \f$p = -g\f$ stays
 * unscaled because `fletcher_reeves` and `polak_ribiere_plus` are not
 * invariant to the scale of \f$p_{k-1}\f$.
 *
 * ### Initial step and line search
 *
 * The line search starts at step \f$1\f$ along \f$d_k = a_k p_k\f$, with
 * \f[
 * a_k = \tau_{k-1} \frac{g_{k-1}^\top p_{k-1}}{g_k^\top p_k}
 * \f]
 * (Nocedal and Wright, 2006, equation (3.60)), where \f$\tau_{k-1}\f$ is the
 * step accepted along \f$p_{k-1}\f$ (so \f$s = \tau_{k-1} p_{k-1}\f$), so the
 * first trial predicts the first-order decrease of the previous step. The
 * default line search is wolfe_line_search (Nocedal and Wright, 2006,
 * Algorithms 3.5 and 3.6) with \f$c_2 = 0.1\f$, set on `curvature` by
 * default_line_search(). With another `LineSearch` type nothing is set, and
 * with armijo_line_search the gradient is evaluated once more at each accepted
 * point. Over the test battery across all five formulas, total value
 * evaluations were \f$160833\f$ at \f$c_2 = 0.1\f$, \f$211886\f$ at \f$0.4\f$,
 * and \f$756169\f$ at \f$0.9\f$.
 *
 * ### Stationarity and statuses
 *
 * The stationarity decrement is
 * \f$\delta_k = \frac{1}{2} \max\bigl(|g_k^\top d_k|,\, \gamma_{\max}\, g_k^\top g_k\bigr)\f$
 * as in bfgs_optimizer, with \f$d_k\f$ the next direction (scaled as above)
 * and \f$\gamma_{\max}\f$ the largest \f$s^\top s / s^\top y\f$ over the
 * accepted steps instead of BFGS's \f$s^\top y / y^\top y\f$. Both are
 * Barzilai and Borwein's inverse curvatures (Barzilai and Borwein, 1988), and
 * the first is the larger. With BFGS's floor, every
 * `beta` formula was declared converged up to \f$8.8 \times 10^{-3}\f$ above
 * \f$f^*\f$ on quadratics with \f$\kappa = 10^6\f$ (\f$n = 100\f$) and
 * \f$10^8\f$ (\f$n = 10\f$); with this floor no run was, and the largest
 * \f$f - f^*\f$ of a converged run was \f$7.2 \times 10^{-7}\f$. The one-step
 * confirmation, zero-gradient start behaviour, termination statuses, and
 * `optimizer_report` fields are those of bfgs_optimizer.
 *
 * ### Settings
 *
 * The static helper default_line_search() builds the default `line_search`
 * member.
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried at each iteration, accepting any rule matching `quasi_newton_provides` |
 * | `line_search` | `default_line_search()` | Line search along each direction from step \f$1\f$, defaulting to `wolfe_line_search` with \f$c_2 = 0.1\f$ |
 * | `beta` | `conjugate_gradient_beta::hager_zhang` | Update formula for \f$\beta_k\f$ |
 * | `restart_interval` | \f$0\f$ | Number of directions between periodic restarts; \f$0\f$ restarts every \f$n\f$ directions |
 * | `powell_restart` | \f$0.2\f$ | Powell restart threshold \f$c\f$; \f$+\infty\f$ disables Powell restarts |
 * | `max_iterations` | \f$10000\f$ | Maximum number of iterations before returning `optimizer_status::iteration_budget` |
 * | `max_evaluations` | \f$10^5\f$ | Value-evaluation budget checked before each line search, so a run may exceed it by one search |
 * | `curvature_tolerance` | \f$10^{-10}\f$ | Cosine threshold for accepting a step pair; failure restarts the model |
 * | `trace` | empty | Optional optimizer_trace callback invoked once per accepted iteration |
 *
 * ### Cost and measured behaviour
 *
 * Each iteration performs \f$O(n)\f$ work and stores three vectors of \f$n\f$
 * `double` entries of state; each Wolfe trial evaluates value and gradient
 * once. Measured with gcc on one thread, extended Rosenbrock with \f$n = 100\f$
 * takes \f$47.7\f$ microseconds (\f$31\f$ iterations, \f$93\f$ evaluations) for
 * `hager_zhang`, compared with \f$82.3\f$ microseconds (\f$36\f$ iterations,
 * \f$43\f$ evaluations) for lbfgs_optimizer: conjugate gradient uses about three
 * evaluations per iteration against \f$1.2\f$ for L-BFGS.
 *
 * With default settings, the optimizer converges on the
 * Moré-Garbow-Hillstrom problems with \f$f^*\f$ shifted by \f$0\f$, \f$1\f$,
 * \f$10^3\f$, and \f$10^6\f$, except Brown's badly scaled function with
 * \f$f^* \ne 0\f$, where most formulas end `optimizer_status::stalled` or
 * `optimizer_status::line_search_failed` within \f$2 \times 10^{-5}\f$ of
 * \f$x^*\f$: its condition number \f$\kappa\f$ is near \f$10^{12}\f$, and the
 * decrement, whose floor carries the inverse of the smallest curvature, cannot
 * fall below about \f$\kappa\, \varepsilon\, |f| > 10^{-6}\f$ there
 * (\f$\varepsilon = 2^{-52}\f$). It converges on quadratics up to
 * \f$\kappa = 10^8\f$ (\f$n = 10\f$) and \f$10^6\f$ (\f$n = 100\f$), and ends
 * `optimizer_status::stalled` or `optimizer_status::line_search_failed` at
 * \f$\kappa = 10^{10}\f$ and \f$10^{12}\f$. On the saddle problem from
 * \f$(1, 0)\f$, where a coordinate of the gradient is \f$0\f$ along the whole
 * line through the saddle, it lands on the saddle and reports
 * `optimizer_status::converged` (a stationary point); from \f$(1, 10^{-3})\f$
 * it reaches a minimum. It never converged on \f$\sum_i |x_i|\f$ or the noisy
 * objective.
 *
 * With exact line searches on a quadratic and no restarts, every formula is
 * linear conjugate gradient (Hestenes and Stiefel, 1952): the tests check the
 * values against a textbook linear CG and termination after \f$n\f$ steps for
 * \f$n = 5\f$, \f$\kappa = 10^3\f$ and \f$n = 10\f$, \f$\kappa = 10\f$; at
 * \f$n = 10\f$, \f$\kappa = 10^3\f$ rounding delays the end by two steps for
 * both routes.
 *
 * ### Example
 *
 * \code
 * conjugate_gradient_optimizer<> opt;
 * opt.beta = conjugate_gradient_beta::polak_ribiere_plus;
 * const optimizer_report report = minimize(opt, f, x0);
 * const bool ok = optimizer_succeeded(report.status);
 * \endcode
 *
 * \tparam Rule Stopping rule type satisfying `stop_rule_for<Rule, quasi_newton_provides>`.
 * \tparam LineSearch Line search type such as `wolfe_line_search` or `armijo_line_search`.
 * \see \ref optimizers_first_order, conjugate_gradient_beta, bfgs_optimizer, lbfgs_optimizer, quasi_newton_provides, minimize(), optimizer_report
 */
// Nonlinear CG. Settings are members; minimize(conjugate_gradient_optimizer<>{}, f, x0).
template <class Rule = decrement_stop_rule, class LineSearch = wolfe_line_search>
  requires stop_rule_for<Rule, quasi_newton_provides>
struct conjugate_gradient_optimizer {
  /**
   * \ingroup optimizers_first_order
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  Rule stop_rule{};
  /**
   * \ingroup optimizers_first_order
   * \brief Line search instance used along each direction from step \f$1\f$ (default `default_line_search()`).
   */
  LineSearch line_search = default_line_search();
  /**
   * \ingroup optimizers_first_order
   * \brief Formula selecting \f$\beta_k\f$ (default `conjugate_gradient_beta::hager_zhang`).
   */
  conjugate_gradient_beta beta = conjugate_gradient_beta::hager_zhang;
  /**
   * \ingroup optimizers_first_order
   * \brief Number of directions between periodic restarts; \f$0\f$ restarts every \f$n\f$ directions (default \f$0\f$).
   */
  int restart_interval = 0;             // a restart every this many directions; 0: n
  /**
   * \ingroup optimizers_first_order
   * \brief Powell restart threshold \f$c\f$ in \f$|g_k^\top g_{k-1}| \ge c\, g_k^\top g_k\f$; infinity disables it (default \f$0.2\f$).
   */
  double powell_restart = 0.2;          // restart when |g_k'g_{k-1}| >= this g_k'g_k; infinity disables
  /**
   * \ingroup optimizers_first_order
   * \brief Maximum number of iterations before returning `optimizer_status::iteration_budget` (default \f$10000\f$).
   */
  int max_iterations = 10000;
  /**
   * \ingroup optimizers_first_order
   * \brief Value-evaluation budget checked before each line search (default \f$10^5\f$).
   */
  long max_evaluations = 100000;        // checked before each line search
  /**
   * \ingroup optimizers_first_order
   * \brief Curvature cosine threshold below which a step pair restarts the model (default \f$10^{-10}\f$).
   */
  double curvature_tolerance = 1e-10;   // a pair with s'y <= tol |s| |y| restarts the model
  /**
   * \ingroup optimizers_first_order
   * \brief Optional trace callback invoked once per accepted iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_first_order
   * \brief Constructs the default line search, setting `curvature` to \f$0.1\f$ when `LineSearch` provides it.
   *
   * \return Default-constructed `LineSearch` with `curvature = 0.1` if present.
   * \see \ref optimizers_first_order, conjugate_gradient_optimizer
   */
  static LineSearch default_line_search() {
    LineSearch s{};
    if constexpr (requires { s.curvature; }) s.curvature = 0.1;
    return s;
  }

  /**
   * \ingroup optimizers_first_order
   * \brief Minimizes `objective` starting from `x0` using nonlinear conjugate gradient.
   *
   * See conjugate_gradient_optimizer for the direction formulas, restarts, stationarity floor, and termination statuses.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary optimizer_report of the run.
   * \see \ref optimizers_first_order, conjugate_gradient_optimizer, minimize(), optimizer_report
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    detail::conjugate_gradient_model model(static_cast<int>(x0.extent(0)), beta, restart_interval, powell_restart);
    return detail::quasi_newton_run(model, stop_rule, line_search, max_iterations, max_evaluations,
                                    curvature_tolerance, 1, trace, objective, x0, true);
  }
};

}  // namespace laplace::optimizers
