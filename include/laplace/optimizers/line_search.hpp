#pragma once

// Line searches (note/optimizers.md, decision 5): given the point x, its
// value f(x), its gradient g and a direction d, a step s > 0 along d such that
// x + s d is acceptable. phi(s) = f(x + s d) and phi'(0) = g' d, the slope,
// which must be negative.
//
// - armijo_line_search: sufficient decrease phi(s) <= phi(0) + c1 s phi'(0),
//   by backtracking with quadratic, then cubic interpolation (Dennis and
//   Schnabel 1996, Algorithm A6.3.1); value evaluations only.
// - wolfe_line_search: sufficient decrease and the strong curvature condition
//   |phi'(s)| <= c2 |phi'(0)|, by bracketing and zoom with safeguarded cubic
//   and quadratic interpolation (Nocedal and Wright 2006, Algorithms 3.5 and
//   3.6; the extrapolation bounds and the interpolation of case 1 of More and
//   Thuente 1994); for BFGS and L-BFGS.
//
// A non-finite value (or slope) at a trial point is a refused point (decision
// 3): the search shortens the step and never interpolates through it. A trial
// point equal to x in every coordinate (the step below rounding) is a
// failure, never an acceptance. The optional resolution test
// is asked once, at the initial step, against the predicted decrease
// s0 |phi'(0)|, never inside the backtracking.
//
// Sufficient decrease is tested as phi(s) - phi(0) <= c1 s phi'(0), never as
// phi(s) <= phi(0) + c1 s phi'(0): once c1 s |phi'(0)| is below the rounding of
// phi(0) the sum rounds to phi(0) and the second form accepts a step that
// decreases nothing (measured on a constant objective: accepted at s = 1e-12).

#include "laplace/optimizers/objective.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Termination status of a line search.
 *
 * `not_descent_direction` and `below_resolution` return before any objective
 * evaluation. `step_below_rounding` is always a failure and never an
 * acceptance. Newton, BFGS and L-BFGS map `step_below_rounding` (BFGS and
 * L-BFGS also `below_resolution`) to `optimizer_status::stalled`, and other
 * failures to `optimizer_status::line_search_failed`; the method pages give
 * the exact mapping.
 *
 * \see \ref optimizers, line_search_result, armijo_line_search, wolfe_line_search
 */
enum class line_search_outcome {
  accepted,
  ///< A trial step satisfying the search conditions was found.
  not_descent_direction,  // phi'(0) >= 0 or not finite
  ///< Initial directional derivative \f$\phi'(0) = g^\top d\f$ satisfies \f$\phi'(0) \ge 0\f$ or is not finite; no evaluation is made.
  below_resolution,       // s0 |phi'(0)| below the resolution given to the search
  ///< Search `resolution` is positive and \f$s_0 |\phi'(0)| <\f$ `resolution` at the initial step \f$s_0\f$; checked once before any evaluation.
  step_below_rounding,    // x + s d == x, or the Wolfe bracket narrower than the rounding of s
  ///< Trial point equals `x` in every coordinate, or the Wolfe bracket is narrower than \f$4\varepsilon\f$ times its larger endpoint; never an acceptance.
  evaluation_budget       // max_evaluations reached without an acceptable step
  ///< Evaluation budget `max_evaluations` was reached without accepting a step.
};

/**
 * \ingroup optimizers
 * \brief Result returned by `armijo_line_search::search()` and `wolfe_line_search::search()`.
 *
 * ### Fields
 *
 * | Field | Type | Default | Meaning |
 * |---|---|---|---|
 * | `outcome` | `line_search_outcome` | `evaluation_budget` | Why the search ended |
 * | `step` | `double` | \f$0\f$ | Accepted step length \f$s\f$, or the last one tried (\f$0\f$ when no trial was made) |
 * | `value` | `double` | quiet NaN | \f$\phi(s)\f$ when accepted, quiet NaN otherwise |
 * | `slope` | `double` | quiet NaN | \f$\phi'(s)\f$ when accepted by `wolfe_line_search`, quiet NaN otherwise |
 * | `value_evaluations` | `int` | \f$0\f$ | Objective value evaluations performed |
 * | `gradient_evaluations` | `int` | \f$0\f$ | Gradient evaluations performed (`0` for Armijo; one per trial for Wolfe) |
 * | `refused_evaluations` | `int` | \f$0\f$ | Trial points refused due to a non-finite value or slope |
 *
 * \see \ref optimizers, line_search_outcome, armijo_line_search, wolfe_line_search
 */
struct line_search_result {
  /**
   * \ingroup optimizers
   * \brief Outcome of the line search (default `line_search_outcome::evaluation_budget`).
   */
  line_search_outcome outcome = line_search_outcome::evaluation_budget;
  /**
   * \ingroup optimizers
   * \brief Accepted step length \f$s\f$, or the last step tried; \f$0\f$ when no trial was made (default \f$0\f$).
   */
  double step = 0.0;   // the accepted step (the last one tried otherwise)
  /**
   * \ingroup optimizers
   * \brief Objective value \f$\phi(s) = f(x + s d)\f$ when accepted, or quiet NaN otherwise (default quiet NaN).
   */
  double value = std::numeric_limits<double>::quiet_NaN();  // phi(step) when accepted
  /**
   * \ingroup optimizers
   * \brief Directional derivative \f$\phi'(s) = \nabla f(x + s d)^\top d\f$ when accepted by `wolfe_line_search`, or quiet NaN otherwise (default quiet NaN).
   */
  double slope = std::numeric_limits<double>::quiet_NaN();  // phi'(step), Wolfe only
  /**
   * \ingroup optimizers
   * \brief Number of objective value evaluations performed (default \f$0\f$).
   */
  int value_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Number of gradient evaluations performed; \f$0\f$ for `armijo_line_search`, one per trial for `wolfe_line_search` (default \f$0\f$).
   */
  int gradient_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Number of trial points refused due to a non-finite value or directional slope (default \f$0\f$).
   */
  int refused_evaluations = 0;  // trial points with a non-finite value or slope
  /**
   * \ingroup optimizers
   * \brief Returns whether a step was accepted (`outcome == line_search_outcome::accepted`).
   *
   * \return `true` if `outcome == line_search_outcome::accepted`, `false` otherwise.
   */
  bool accepted() const { return outcome == line_search_outcome::accepted; }
};

namespace detail {

inline double dot(const optimizer_point& a, const optimizer_point& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.extent(0); ++i) s += a(i) * b(i);
  return s;
}

// x_trial = x + s d; false when x_trial equals x in every coordinate.
inline bool trial_point(const optimizer_point& x, const optimizer_point& d, double s, const optimizer_vector& x_trial) {
  bool moved = false;
  for (std::size_t i = 0; i < x.extent(0); ++i) {
    x_trial(i) = x(i) + s * d(i);
    moved = moved || x_trial(i) != x(i);
  }
  return moved;
}

// The minimizer of the cubic through (a, fa, da) and (b, fb, db) (Nocedal
// and Wright 2006, eq. 3.59); NaN when the cubic has no local minimizer.
inline double cubic_minimizer(double a, double fa, double da, double b, double fb, double db) {
  const double d1 = da + db - 3.0 * (fa - fb) / (a - b);
  const double disc = d1 * d1 - da * db;
  if (!(disc >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
  const double d2 = std::copysign(std::sqrt(disc), b - a);
  return b - (b - a) * (db + d2 - d1) / (db - da + 2.0 * d2);
}

// The minimizer of the quadratic through (a, fa) with slope da and (b, fb).
inline double quadratic_minimizer(double a, double fa, double da, double b, double fb) {
  const double w = b - a;
  return a - da * w * w / (2.0 * (fb - fa - da * w));
}

// The backtracking of armijo_line_search and nonmonotone_line_search: a step
// is accepted when phi(s) - reference <= c1 s phi'(0), with reference = phi(0)
// for Armijo and the moving average C_k for the nonmonotone search; the
// interpolation is always through phi(0) = value.
template <value_objective O>
line_search_result backtracking_search(O& objective, const optimizer_point& x, double value, double reference,
                                       const optimizer_point& g, const optimizer_point& d, double initial_step,
                                       const optimizer_vector& x_trial, double sufficient_decrease, double min_shrink,
                                       double max_shrink, int max_evaluations, double resolution) {
  line_search_result r;
  const double slope = detail::dot(g, d);
  if (!(slope < 0.0) || !std::isfinite(slope)) {
    r.outcome = line_search_outcome::not_descent_direction;
    return r;
  }
  if (resolution > 0.0 && initial_step * std::abs(slope) < resolution) {
    r.outcome = line_search_outcome::below_resolution;
    return r;
  }
  double s = initial_step, s_prev = 0.0, f_prev = 0.0;
  bool have_prev = false;
  while (r.value_evaluations < max_evaluations) {
    r.step = s;
    if (!detail::trial_point(x, d, s, x_trial)) {
      r.outcome = line_search_outcome::step_below_rounding;
      return r;
    }
    const double ft = objective.value(optimizer_point(x_trial));
    ++r.value_evaluations;
    if (std::isfinite(ft) && ft - reference <= sufficient_decrease * s * slope) {
      r.outcome = line_search_outcome::accepted;
      r.value = ft;
      return r;
    }
    double s_new;
    if (!std::isfinite(ft)) {
      ++r.refused_evaluations;
      s_new = max_shrink * s;
      have_prev = false;  // nothing to interpolate through
    } else {
      const double r1 = ft - value - slope * s;  // > 0: the curvature seen at s
      if (!have_prev) {
        s_new = -slope * s * s / (2.0 * r1);
      } else {
        const double r2 = f_prev - value - slope * s_prev;
        const double a = (r1 / (s * s) - r2 / (s_prev * s_prev)) / (s - s_prev);
        const double b = (-s_prev * r1 / (s * s) + s * r2 / (s_prev * s_prev)) / (s - s_prev);
        const double disc = b * b - 3.0 * a * slope;
        if (a == 0.0)
          s_new = -slope / (2.0 * b);
        else if (disc < 0.0)
          s_new = max_shrink * s;
        else if (b <= 0.0)
          s_new = (-b + std::sqrt(disc)) / (3.0 * a);
        else
          s_new = -slope / (b + std::sqrt(disc));  // the same root, without cancellation
      }
      s_prev = s;
      f_prev = ft;
      have_prev = true;
    }
    s = std::isnan(s_new) ? max_shrink * s : std::clamp(s_new, min_shrink * s, max_shrink * s);
  }
  r.outcome = line_search_outcome::evaluation_budget;
  return r;
}

}  // namespace detail

/**
 * \ingroup optimizers
 * \brief Backtracking line search enforcing the Armijo sufficient-decrease condition.
 *
 * Searches along a descent direction \f$d\f$ from \f$x\f$ for a step
 * \f$s > 0\f$ satisfying the sufficient-decrease condition
 * \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$, where
 * \f$\phi(s) = f(x + s d)\f$, \f$\phi'(0) = g^\top d < 0\f$, and \f$c_1\f$ is
 * `sufficient_decrease`. Sufficient decrease is evaluated as the difference
 * \f$\phi(s) - \phi(0)\f$ rather than as \f$\phi(s) \le \phi(0) + c_1 s \phi'(0)\f$:
 * once \f$c_1 s |\phi'(0)|\f$ falls below the rounding of \f$\phi(0)\f$, the
 * sum rounds to \f$\phi(0)\f$ and accepts steps (such as \f$s = 10^{-12}\f$ on
 * a constant objective) that decrease nothing.
 *
 * `armijo_line_search` evaluates only `value()` at trial points, is `const`,
 * and keeps no state between calls. It is the default line search of
 * `newton_optimizer` and is also accepted by `bfgs_optimizer` and
 * `lbfgs_optimizer` (which evaluate the gradient once more at the accepted
 * point).
 *
 * ### Algorithm
 *
 * Backtracking uses safeguarded polynomial interpolation (Dennis and Schnabel,
 * 1996, Algorithm A6.3.1). Let \f$\beta_{\min}\f$ denote `min_shrink`,
 * \f$\beta_{\max}\f$ denote `max_shrink`, \f$s\f$ be the trial step just
 * refused with \f$r_1 = \phi(s) - \phi(0) - \phi'(0) s\f$, and
 * \f$s_{\mathrm{p}}\f$ be the preceding refused step with
 * \f$r_2 = \phi(s_{\mathrm{p}}) - \phi(0) - \phi'(0) s_{\mathrm{p}}\f$.
 *
 * - On the first backtrack, the trial is the minimizer of the quadratic
 *   through \f$\phi(0)\f$, \f$\phi'(0)\f$, and \f$\phi(s)\f$:
 *   \f[ s_{\mathrm{new}} = -\frac{\phi'(0)\, s^2}{2 r_1}. \f]
 * - On later backtracks, the cubic \f$\phi(0) + \phi'(0) u + b u^2 + a u^3\f$
 *   through the last two trial values has coefficients
 *   \f[ a = \frac{r_1 / s^2 - r_2 / s_{\mathrm{p}}^2}{s - s_{\mathrm{p}}}, \qquad
 *       b = \frac{-s_{\mathrm{p}} r_1 / s^2 + s\, r_2 / s_{\mathrm{p}}^2}{s - s_{\mathrm{p}}}. \f]
 *   Its minimizer is \f$s_{\mathrm{new}} = (-b + \sqrt{b^2 - 3 a \phi'(0)}) / (3a)\f$
 *   when \f$b \le 0\f$ and, to avoid cancellation,
 *   \f$s_{\mathrm{new}} = -\phi'(0) / (b + \sqrt{b^2 - 3 a \phi'(0)})\f$ when
 *   \f$b > 0\f$; when \f$a = 0\f$ it is \f$-\phi'(0) / (2b)\f$, and when
 *   \f$b^2 - 3 a \phi'(0) < 0\f$ it falls back to \f$\beta_{\max} s\f$.
 * - If the interpolant is NaN, the next step is \f$\beta_{\max} s\f$;
 *   otherwise it is clamped to \f$[\beta_{\min} s,\, \beta_{\max} s]\f$.
 * - After a refused trial point (non-finite \f$\phi(s)\f$), `refused_evaluations`
 *   is incremented, the step is set to \f$\beta_{\max} s\f$, and interpolation
 *   restarts from the quadratic because there is nothing finite to interpolate
 *   through.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `sufficient_decrease` | \f$10^{-4}\f$ | Armijo constant \f$c_1\f$ in \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$ |
 * | `min_shrink` | \f$0.1\f$ | Lower safeguard factor \f$\beta_{\min}\f$ on each backtrack |
 * | `max_shrink` | \f$0.5\f$ | Upper safeguard factor \f$\beta_{\max}\f$, and shrink factor after a refused point |
 * | `max_evaluations` | \f$60\f$ | Maximum number of trial `value()` evaluations |
 * | `resolution` | \f$0\f$ | Optional threshold; when positive, fails with `below_resolution` if \f$s_0 \lvert \phi'(0)\rvert <\f$ `resolution` |
 *
 * ### Outcomes
 *
 * - `line_search_outcome::accepted`: a trial step satisfied sufficient decrease;
 *   `x_trial` holds \f$x + s d\f$, `result.step` is \f$s\f$, and `result.value`
 *   is \f$\phi(s)\f$.
 * - `line_search_outcome::not_descent_direction`: \f$\phi'(0) \ge 0\f$ or not
 *   finite; no evaluation is made.
 * - `line_search_outcome::below_resolution`: `resolution` \f$> 0\f$ and
 *   \f$s_0 |\phi'(0)| <\f$ `resolution`; checked once before any evaluation.
 * - `line_search_outcome::step_below_rounding`: a trial point `x + s d` equals
 *   `x` in every coordinate.
 * - `line_search_outcome::evaluation_budget`: `max_evaluations` was reached
 *   without accepting a step.
 *
 * ### Example
 *
 * Evaluates \f$f(x) = \frac{1}{2} \lVert x \rVert^2\f$ and searches along
 * \f$-g\f$ on the host after `Kokkos::initialize`:
 *
 * \code
 * #include <laplace/optimizers/line_search.hpp>
 *
 * struct quadratic_objective {
 *   double value(const laplace::optimizers::optimizer_point& x) const {
 *     return 0.5 * (x(0) * x(0) + x(1) * x(1));
 *   }
 *   double value_gradient(const laplace::optimizers::optimizer_point& x,
 *                         const laplace::optimizers::optimizer_vector& g) const {
 *     g(0) = x(0);
 *     g(1) = x(1);
 *     return value(x);
 *   }
 * };
 *
 * bool armijo_example() {
 *   constexpr int n = 2;
 *   laplace::optimizers::optimizer_vector x("x", n), g("g", n), d("d", n), x_trial("x_trial", n);
 *   x(0) = 3.0;
 *   x(1) = -4.0;
 *   quadratic_objective objective;
 *   const double fx = objective.value_gradient(x, g);
 *   for (int i = 0; i < n; ++i) d(i) = -g(i);
 *   const laplace::optimizers::armijo_line_search search;
 *   const laplace::optimizers::line_search_result r = search.search(objective, x, fx, g, d, 1.0, x_trial);
 *   return r.accepted();
 * }
 * \endcode
 *
 * \see \ref optimizers, wolfe_line_search, line_search_result, line_search_outcome
 */
struct armijo_line_search {
  /**
   * \ingroup optimizers
   * \brief Sufficient-decrease constant \f$c_1\f$ (default \f$10^{-4}\f$).
   */
  double sufficient_decrease = 1e-4;  // c1
  /**
   * \ingroup optimizers
   * \brief Lower safeguard factor \f$\beta_{\min}\f$ on each backtracking shrink (default \f$0.1\f$).
   */
  double min_shrink = 0.1;            // the next step lies in [min_shrink s, max_shrink s]
  /**
   * \ingroup optimizers
   * \brief Upper safeguard factor \f$\beta_{\max}\f$, also used after a refused trial point (default \f$0.5\f$).
   */
  double max_shrink = 0.5;            // also the factor after a refused point
  /**
   * \ingroup optimizers
   * \brief Maximum number of trial value evaluations (default \f$60\f$).
   */
  int max_evaluations = 60;
  /**
   * \ingroup optimizers
   * \brief Optional resolution threshold; when positive, fails with `line_search_outcome::below_resolution` before evaluating if \f$s_0 |\phi'(0)|\f$ is less than `resolution` (default \f$0\f$).
   */
  double resolution = 0.0;            // 0: no resolution test

  /**
   * \ingroup optimizers
   * \brief Searches from `x` along `d` starting at `initial_step`.
   *
   * On acceptance, `x_trial` holds \f$x + s d\f$ for the accepted step \f$s\f$
   * and `result.value` holds \f$f(x + s d)\f$. `gradient_evaluations` is
   * always \f$0\f$.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to evaluate at trial points.
   * \param x Current point \f$x\f$ (\f$n\f$ entries).
   * \param value Objective value \f$f(x)\f$ at `x`.
   * \param g Gradient \f$g\f$ at `x` (\f$n\f$ entries).
   * \param d Search direction \f$d\f$ (\f$n\f$ entries).
   * \param initial_step Initial trial step length \f$s_0 > 0\f$.
   * \param x_trial Output buffer of \f$n\f$ entries receiving trial points \f$x + s d\f$.
   * \return A `line_search_result` recording the outcome, step, value, and evaluation counts.
   */
  // On acceptance x_trial holds x + step d.
  template <value_objective O>
  line_search_result search(O& objective, const optimizer_point& x, double value, const optimizer_point& g,
                            const optimizer_point& d, double initial_step, const optimizer_vector& x_trial) const {
    return detail::backtracking_search(objective, x, value, value, g, d, initial_step, x_trial, sufficient_decrease,
                                       min_shrink, max_shrink, max_evaluations, resolution);
  }
};

/**
 * \ingroup optimizers
 * \brief Bracketing and zoom line search enforcing the strong Wolfe conditions.
 *
 * Searches along a descent direction \f$d\f$ from \f$x\f$ for a step
 * \f$s > 0\f$ satisfying both sufficient decrease,
 * \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$, and strong curvature,
 * \f$|\phi'(s)| \le c_2 |\phi'(0)|\f$, where \f$\phi(s) = f(x + s d)\f$,
 * \f$\phi'(s) = \nabla f(x + s d)^\top d\f$, \f$c_1\f$ is
 * `sufficient_decrease`, and \f$c_2\f$ is `curvature`. Sufficient decrease is
 * tested in the difference form so that a constant objective cannot accept a
 * tiny step due to rounding of \f$\phi(0)\f$.
 *
 * Every trial point evaluates both `value` and `gradient` via
 * `value_gradient()`. The search object is `const` and keeps no state between
 * calls. It is the default line search of `bfgs_optimizer` and
 * `lbfgs_optimizer` (\ref optimizers_quasi_newton).
 *
 * ### Algorithm
 *
 * Uses two-phase bracketing and zoom (Nocedal and Wright, 2006, Algorithms
 * 3.5 and 3.6) with safeguarded cubic and quadratic interpolation; the
 * extrapolation bounds and the interpolation when the value rose follow Moré
 * and Thuente (1994). Let \f$s_{\mathrm{lo}}\f$ be the last trial with sufficient decrease
 * and \f$\phi'(s_{\mathrm{lo}}) < 0\f$ (initially \f$0\f$).
 *
 * In the bracketing phase at trial \f$s\f$:
 * - If sufficient decrease fails, or \f$s_{\mathrm{lo}} > 0\f$ and
 *   \f$\phi(s) \ge \phi(s_{\mathrm{lo}})\f$, zoom on \f$[s_{\mathrm{lo}}, s]\f$.
 * - Else if strong curvature holds, accept \f$s\f$.
 * - Else if \f$\phi'(s) \ge 0\f$, zoom on \f$[s, s_{\mathrm{lo}}]\f$ (with
 *   \f$s\f$ as the low end).
 * - Else extrapolate by the cubic minimizer through
 *   \f$(s_{\mathrm{lo}}, \phi, \phi')\f$ and \f$(s, \phi, \phi')\f$ clamped to
 *   \f$[s + 1.1(s - s_{\mathrm{lo}}),\, s + 4(s - s_{\mathrm{lo}})]\f$ (or the
 *   upper end when the cubic has no minimizer), kept below any step already
 *   refused (halfway to that limit). A refused trial during bracketing halves
 *   the distance from \f$s_{\mathrm{lo}}\f$ and becomes that upper limit.
 *
 * In the zoom phase on a bracket with ends \f$s_{\mathrm{lo}}\f$ (sufficient
 * decrease and the lowest value seen) and \f$s_{\mathrm{hi}}\f$:
 * - For two steps \f$p\f$ and \f$q\f$ with \f$w = q - p\f$, the minimizer of
 *   the quadratic through \f$\phi(p)\f$, \f$\phi'(p)\f$, and \f$\phi(q)\f$ is
 *   \f[ p - \frac{\phi'(p)\, w^2}{2(\phi(q) - \phi(p) - \phi'(p) w)}, \f]
 *   and the minimizer of the cubic through \f$(p, \phi(p), \phi'(p))\f$ and
 *   \f$(q, \phi(q), \phi'(q))\f$ is (Nocedal and Wright, 2006, eq. 3.59)
 *   \f[ q - w\,\frac{\phi'(q) + \eta_2 - \eta_1}{\phi'(q) - \phi'(p) + 2 \eta_2}, \quad
 *       \eta_1 = \phi'(p) + \phi'(q) - 3\frac{\phi(p) - \phi(q)}{p - q}, \quad
 *       \eta_2 = \operatorname{sign}(w)\sqrt{\eta_1^2 - \phi'(p)\phi'(q)}, \f]
 *   with no minimizer when \f$\eta_1^2 - \phi'(p)\phi'(q) < 0\f$. The same
 *   cubic, through \f$s_{\mathrm{lo}}\f$ and \f$s\f$, is the one of the
 *   bracketing phase.
 * - With \f$p = s_{\mathrm{lo}}\f$ and \f$q = s_{\mathrm{hi}}\f$, the zoom
 *   trial is the cubic minimizer, except that when
 *   \f$\phi(s_{\mathrm{hi}}) > \phi(s_{\mathrm{lo}})\f$ it is the cubic
 *   minimizer if it is closer to \f$s_{\mathrm{lo}}\f$ than the quadratic
 *   minimizer and their mean otherwise (the quadratic alone when the cubic
 *   has none), and the midpoint when \f$s_{\mathrm{hi}}\f$ was refused or no
 *   interpolant exists; the trial is then kept at least a tenth of the bracket
 *   width from each end.
 * - A refused zoom trial becomes \f$s_{\mathrm{hi}}\f$; one failing sufficient
 *   decrease or not lowering \f$\phi(s_{\mathrm{lo}})\f$ becomes
 *   \f$s_{\mathrm{hi}}\f$; otherwise it is accepted if strong curvature holds,
 *   and else becomes \f$s_{\mathrm{lo}}\f$ (moving the old
 *   \f$s_{\mathrm{lo}}\f$ to \f$s_{\mathrm{hi}}\f$ when
 *   \f$\phi'(s)(s_{\mathrm{hi}} - s_{\mathrm{lo}}) \ge 0\f$).
 * - A bracket narrower than \f$4\varepsilon \max(s_{\mathrm{lo}}, s_{\mathrm{hi}})\f$
 *   ends the search with `line_search_outcome::step_below_rounding`.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `sufficient_decrease` | \f$10^{-4}\f$ | Armijo constant \f$c_1\f$ in \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$ |
 * | `curvature` | \f$0.9\f$ | Strong curvature constant \f$c_2\f$ in \f$\lvert \phi'(s)\rvert \le c_2 \lvert \phi'(0)\rvert \f$ |
 * | `max_evaluations` | \f$30\f$ | Maximum number of trial `value_gradient()` evaluations |
 * | `resolution` | \f$0\f$ | Optional threshold; when positive, fails with `below_resolution` if \f$s_0 \lvert \phi'(0)\rvert <\f$ `resolution` |
 *
 * ### Outcomes
 *
 * - `line_search_outcome::accepted`: a trial step satisfied both conditions;
 *   `x_trial` holds \f$x + s d\f$, `g_trial` holds its gradient, `result.step`
 *   is \f$s\f$, `result.value` is \f$\phi(s)\f$, and `result.slope` is
 *   \f$\phi'(s)\f$.
 * - `line_search_outcome::not_descent_direction`: \f$\phi'(0) \ge 0\f$ or not
 *   finite; no evaluation is made.
 * - `line_search_outcome::below_resolution`: `resolution` \f$> 0\f$ and
 *   \f$s_0 |\phi'(0)| <\f$ `resolution`; checked once before any evaluation.
 * - `line_search_outcome::step_below_rounding`: a trial point `x + s d` equals
 *   `x` in every coordinate, or the zoom bracket became narrower than
 *   \f$4\varepsilon \max(s_{\mathrm{lo}}, s_{\mathrm{hi}})\f$.
 * - `line_search_outcome::evaluation_budget`: `max_evaluations` was reached
 *   without accepting a step.
 *
 * ### Example
 *
 * With `objective`, `x`, `fx`, `g`, `d`, `x_trial` defined as in the
 * `armijo_line_search` example and `g_trial` an `optimizer_vector` of length
 * \f$n\f$:
 *
 * \code
 * const laplace::optimizers::wolfe_line_search search;
 * const laplace::optimizers::line_search_result r =
 *     search.search(objective, x, fx, g, d, 1.0, x_trial, g_trial);
 * \endcode
 *
 * \see \ref optimizers, armijo_line_search, line_search_result, line_search_outcome
 */
struct wolfe_line_search {
  /**
   * \ingroup optimizers
   * \brief Sufficient-decrease constant \f$c_1\f$ (default \f$10^{-4}\f$).
   */
  double sufficient_decrease = 1e-4;  // c1
  /**
   * \ingroup optimizers
   * \brief Strong curvature constant \f$c_2\f$ (default \f$0.9\f$).
   */
  double curvature = 0.9;             // c2 (0.9 for quasi-Newton methods)
  /**
   * \ingroup optimizers
   * \brief Maximum number of trial value-and-gradient evaluations (default \f$30\f$).
   */
  int max_evaluations = 30;
  /**
   * \ingroup optimizers
   * \brief Optional resolution threshold; when positive, fails with `line_search_outcome::below_resolution` before evaluating if \f$s_0 |\phi'(0)|\f$ is less than `resolution` (default \f$0\f$).
   */
  double resolution = 0.0;            // 0: no resolution test

  /**
   * \ingroup optimizers
   * \brief Searches from `x` along `d` starting at `initial_step`.
   *
   * On acceptance, `x_trial` holds \f$x + s d\f$, `g_trial` holds its gradient,
   * `result.value` holds \f$\phi(s)\f$, and `result.slope` holds
   * \f$\phi'(s)\f$.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to evaluate at trial points.
   * \param x Current point \f$x\f$ (\f$n\f$ entries).
   * \param value Objective value \f$f(x)\f$ at `x`.
   * \param g Gradient \f$g\f$ at `x` (\f$n\f$ entries).
   * \param d Search direction \f$d\f$ (\f$n\f$ entries).
   * \param initial_step Initial trial step length \f$s_0 > 0\f$.
   * \param x_trial Output buffer of \f$n\f$ entries receiving trial points \f$x + s d\f$.
   * \param g_trial Output buffer of \f$n\f$ entries receiving the gradient at each trial point.
   * \return A `line_search_result` recording the outcome, step, value, slope, and evaluation counts.
   */
  // On acceptance x_trial holds x + step d and g_trial the gradient there.
  template <gradient_objective O>
  line_search_result search(O& objective, const optimizer_point& x, double value, const optimizer_point& g,
                            const optimizer_point& d, double initial_step, const optimizer_vector& x_trial,
                            const optimizer_vector& g_trial) const {
    line_search_result r;
    const double slope = detail::dot(g, d);
    if (!(slope < 0.0) || !std::isfinite(slope)) {
      r.outcome = line_search_outcome::not_descent_direction;
      return r;
    }
    if (resolution > 0.0 && initial_step * std::abs(slope) < resolution) {
      r.outcome = line_search_outcome::below_resolution;
      return r;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    // one trial: false when x + s d == x
    double fa = 0.0, da = 0.0;
    auto evaluate = [&](double s) {
      r.step = s;
      if (!detail::trial_point(x, d, s, x_trial)) return false;
      fa = objective.value_gradient(optimizer_point(x_trial), g_trial);
      ++r.value_evaluations;
      ++r.gradient_evaluations;
      da = detail::dot(g_trial, d);
      if (!std::isfinite(fa) || !std::isfinite(da)) ++r.refused_evaluations;
      return true;
    };
    auto armijo = [&](double s, double f) { return f - value <= sufficient_decrease * s * slope; };
    auto strong_curvature = [&](double ds) { return std::abs(ds) <= -curvature * slope; };
    auto accept = [&] {
      r.outcome = line_search_outcome::accepted;
      r.value = fa;
      r.slope = da;
      return r;
    };
    auto fail = [&](line_search_outcome why) {
      r.outcome = why;
      return r;
    };

    // zoom on the bracket [lo, hi] (either order): lo satisfies sufficient
    // decrease with the lowest value seen; hi_known is false when hi was refused.
    auto zoom = [&](double lo, double f_lo, double d_lo, double hi, double f_hi, double d_hi, bool hi_known) {
      while (r.value_evaluations < max_evaluations) {
        const double width = std::abs(hi - lo);
        if (width <= 4.0 * eps * std::max(lo, hi)) return fail(line_search_outcome::step_below_rounding);
        // When f rose at hi (case 1 of More and Thuente): the cubic if it is
        // closer to lo than the quadratic, otherwise their mean; the quadratic
        // alone when the cubic has no minimizer.
        double s = 0.5 * (lo + hi);
        if (hi_known) {
          const double cubic = detail::cubic_minimizer(lo, f_lo, d_lo, hi, f_hi, d_hi);
          if (f_hi > f_lo) {
            const double quadratic = detail::quadratic_minimizer(lo, f_lo, d_lo, hi, f_hi);
            s = std::isnan(cubic) ? quadratic
                : std::abs(cubic - lo) < std::abs(quadratic - lo) ? cubic
                                                                  : 0.5 * (cubic + quadratic);
          } else if (!std::isnan(cubic)) {
            s = cubic;
          }
        }
        const double left = std::min(lo, hi) + 0.1 * width, right = std::max(lo, hi) - 0.1 * width;
        s = std::isnan(s) ? 0.5 * (lo + hi) : std::clamp(s, left, right);  // at least 1/10 of the bracket from each end
        if (!evaluate(s)) return fail(line_search_outcome::step_below_rounding);
        if (!std::isfinite(fa) || !std::isfinite(da)) {
          hi = s;
          hi_known = false;
        } else if (!armijo(s, fa) || fa >= f_lo) {
          hi = s, f_hi = fa, d_hi = da, hi_known = true;
        } else {
          if (strong_curvature(da)) return accept();
          if (da * (hi - lo) >= 0.0) hi = lo, f_hi = f_lo, d_hi = d_lo, hi_known = true;
          lo = s, f_lo = fa, d_lo = da;
        }
      }
      return fail(line_search_outcome::evaluation_budget);
    };

    // bracketing: lo is the last trial with sufficient decrease and phi' < 0
    double lo = 0.0, f_lo = value, d_lo = slope;
    double s = initial_step, limit = std::numeric_limits<double>::infinity();
    while (r.value_evaluations < max_evaluations) {
      if (!evaluate(s)) return fail(line_search_outcome::step_below_rounding);
      if (!std::isfinite(fa) || !std::isfinite(da)) {  // refused: shorten towards lo
        limit = s;
        s = lo + 0.5 * (s - lo);
        continue;
      }
      if (!armijo(s, fa) || (lo > 0.0 && fa >= f_lo)) return zoom(lo, f_lo, d_lo, s, fa, da, true);
      if (strong_curvature(da)) return accept();
      if (da >= 0.0) return zoom(s, fa, da, lo, f_lo, d_lo, true);
      // phi still decreasing: extrapolate within [s + 1.1 (s - lo), s + 4 (s - lo)]
      const double low = s + 1.1 * (s - lo), high = s + 4.0 * (s - lo);
      double s_new = detail::cubic_minimizer(lo, f_lo, d_lo, s, fa, da);
      s_new = std::isnan(s_new) ? high : std::clamp(s_new, low, high);
      if (s_new >= limit) s_new = 0.5 * (s + limit);
      lo = s, f_lo = fa, d_lo = da;
      s = s_new;
    }
    return fail(line_search_outcome::evaluation_budget);
  }
};

/**
 * \ingroup optimizers_first_order
 * \brief Moving-average reference value maintained by callers of `nonmonotone_line_search`.
 *
 * Stores the weighted average \f$C_k\f$ and cumulative weight \f$Q_k\f$ of
 * Zhang and Hager's nonmonotone line search (Zhang and Hager, 2004). Calling
 * `start(f)` at \f$x_0\f$ sets \f$C_0 = f_0\f$ and \f$Q_0 = 1\f$; after an
 * accepted point with value \f$f_{k+1}\f$, `accept(f, averaging)` applies
 * \f$Q_{k+1} = \eta Q_k + 1\f$ and
 * \f$C_{k+1} = (\eta Q_k C_k + f_{k+1}) / Q_{k+1}\f$, where \f$\eta\f$ is
 * `averaging`. \f$C_k\f$ lies between \f$f_k\f$ and \f$C_{k-1}\f$, so
 * \f$C_k \ge f_k\f$.
 *
 * \see \ref optimizers_first_order, nonmonotone_line_search
 */
// The reference value of nonmonotone_line_search (Zhang and Hager 2004): the
// weighted average C_k of the values f_0, ..., f_k accepted so far, with
// weight Q_k. start(f_0) gives C_0 = f_0, Q_0 = 1; after an accepted f_{k+1},
// Q_{k+1} = eta Q_k + 1 and C_{k+1} = (eta Q_k C_k + f_{k+1}) / Q_{k+1}. C_k
// lies between f_k and C_{k-1}, so C_k >= f_k.
struct nonmonotone_reference {
  /**
   * \ingroup optimizers_first_order
   * \brief Current moving-average reference value \f$C_k\f$ (default quiet NaN).
   */
  double value = std::numeric_limits<double>::quiet_NaN();  // C_k
  /**
   * \ingroup optimizers_first_order
   * \brief Cumulative weight \f$Q_k\f$ (default \f$1\f$).
   */
  double weight = 1.0;                                      // Q_k
  /**
   * \ingroup optimizers_first_order
   * \brief Initializes the reference at \f$x_0\f$ with \f$C_0 = f\f$ and \f$Q_0 = 1\f$.
   *
   * \param f Initial objective value \f$f(x_0)\f$.
   */
  void start(double f) { value = f, weight = 1.0; }
  /**
   * \ingroup optimizers_first_order
   * \brief Updates \f$Q_k\f$ and \f$C_k\f$ after accepting a step with value `f`.
   *
   * \param f Accepted objective value \f$f_{k+1}\f$.
   * \param averaging Averaging factor \f$\eta \in [0, 1]\f$ (`0` gives \f$C_{k+1} = f_{k+1}\f$).
   */
  void accept(double f, double averaging) {
    const double q = averaging * weight + 1.0;
    value = (averaging * weight * value + f) / q;
    weight = q;
  }
};

/**
 * \ingroup optimizers_first_order
 * \brief Zhang and Hager's nonmonotone backtracking line search.
 *
 * Searches along a descent direction \f$d\f$ from \f$x\f$ for a step
 * \f$s > 0\f$ satisfying \f$\phi(s) - R_k \le c_1 s \phi'(0)\f$, where
 * \f$\phi(s) = f(x + s d)\f$, \f$\phi'(0) = g^\top d < 0\f$, \f$c_1\f$ is
 * `sufficient_decrease`, and \f$R_k = \max(f(x), C_k)\f$ uses the moving
 * average \f$C_k\f$ in `nonmonotone_reference` (Zhang and Hager, 2004). A step
 * that raises \f$f\f$ is accepted while it stays below \f$C_k\f$.
 *
 * The backtracking algorithm (quadratic then cubic interpolation through
 * \f$\phi(0) = f(x)\f$, refused points, the `step_below_rounding` check, the
 * optional `resolution` test, and all `line_search_outcome` values) is
 * identical to `armijo_line_search`. Setting `averaging` to \f$0\f$ gives
 * \f$C_k = f_k\f$, reducing to `armijo_line_search`; setting `averaging` to
 * \f$1\f$ gives the unweighted mean of all accepted values.
 *
 * The caller maintains the `nonmonotone_reference` (`start()` at \f$x_0\f$ and
 * `accept()` at each new point); `gradient_descent_optimizer` and
 * `barzilai_borwein_optimizer` do this automatically, with
 * `nonmonotone_line_search` as the default of `barzilai_borwein_optimizer`.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `sufficient_decrease` | \f$10^{-4}\f$ | Sufficient-decrease constant \f$c_1\f$ in \f$\phi(s) - R_k \le c_1 s \phi'(0)\f$ |
 * | `averaging` | \f$0.85\f$ | Averaging weight \f$\eta \in [0, 1]\f$ passed to `nonmonotone_reference::accept()` (Zhang and Hager's value) |
 * | `min_shrink` | \f$0.1\f$ | Lower safeguard factor \f$\beta_{\min}\f$ on each backtrack |
 * | `max_shrink` | \f$0.5\f$ | Upper safeguard factor \f$\beta_{\max}\f$, and shrink factor after a refused point |
 * | `max_evaluations` | \f$60\f$ | Maximum number of trial `value()` evaluations |
 * | `resolution` | \f$0\f$ | Optional threshold; when positive, fails with `below_resolution` if \f$s_0 \lvert \phi'(0)\rvert <\f$ `resolution` |
 *
 * ### Example
 *
 * With `objective`, `x`, `fx`, `g`, `d`, `x_trial` as in the
 * `armijo_line_search` example:
 *
 * \code
 * laplace::optimizers::nonmonotone_reference ref;
 * ref.start(fx);
 * const laplace::optimizers::nonmonotone_line_search search;
 * const laplace::optimizers::line_search_result r =
 *     search.search(objective, x, fx, ref, g, d, 1.0, x_trial);
 * if (r.accepted()) ref.accept(r.value, search.averaging);
 * \endcode
 *
 * \see \ref optimizers_first_order, nonmonotone_reference, armijo_line_search, line_search_result, line_search_outcome
 */
// Zhang and Hager's (2004) nonmonotone line search: a step s is accepted when
// phi(s) - C_k <= c1 s phi'(0), C_k the moving average of nonmonotone_reference;
// the backtracking (interpolation through phi(0), refused points, rounding) is
// that of armijo_line_search. averaging = eta in [0, 1]: eta = 0 gives
// C_k = f_k, the monotone Armijo search; eta = 1 the plain mean of all values.
// The caller keeps the reference: start() at x_0, accept() at each new point.
struct nonmonotone_line_search {
  /**
   * \ingroup optimizers_first_order
   * \brief Sufficient-decrease constant \f$c_1\f$ (default \f$10^{-4}\f$).
   */
  double sufficient_decrease = 1e-4;  // c1
  /**
   * \ingroup optimizers_first_order
   * \brief Moving-average decay factor \f$\eta \in [0, 1]\f$ (default \f$0.85\f$).
   */
  double averaging = 0.85;            // eta (Zhang and Hager's value)
  /**
   * \ingroup optimizers_first_order
   * \brief Lower safeguard factor \f$\beta_{\min}\f$ on each backtracking shrink (default \f$0.1\f$).
   */
  double min_shrink = 0.1;
  /**
   * \ingroup optimizers_first_order
   * \brief Upper safeguard factor \f$\beta_{\max}\f$, also used after a refused trial point (default \f$0.5\f$).
   */
  double max_shrink = 0.5;
  /**
   * \ingroup optimizers_first_order
   * \brief Maximum number of trial value evaluations (default \f$60\f$).
   */
  int max_evaluations = 60;
  /**
   * \ingroup optimizers_first_order
   * \brief Optional resolution threshold; when positive, fails with `line_search_outcome::below_resolution` before evaluating if \f$s_0 |\phi'(0)|\f$ is less than `resolution` (default \f$0\f$).
   */
  double resolution = 0.0;  // 0: no resolution test

  /**
   * \ingroup optimizers_first_order
   * \brief Searches from `x` along `d` starting at `initial_step` against `reference`.
   *
   * On acceptance, `x_trial` holds \f$x + s d\f$ for the accepted step \f$s\f$
   * and `result.value` holds \f$f(x + s d)\f$; the caller updates `reference`
   * with `reference.accept(result.value, averaging)`.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to evaluate at trial points.
   * \param x Current point \f$x\f$ (\f$n\f$ entries).
   * \param value Objective value \f$f(x)\f$ at `x`.
   * \param reference Moving-average reference state \f$(C_k, Q_k)\f$.
   * \param g Gradient \f$g\f$ at `x` (\f$n\f$ entries).
   * \param d Search direction \f$d\f$ (\f$n\f$ entries).
   * \param initial_step Initial trial step length \f$s_0 > 0\f$.
   * \param x_trial Output buffer of \f$n\f$ entries receiving trial points \f$x + s d\f$.
   * \return A `line_search_result` recording the outcome, step, value, and evaluation counts.
   */
  // On acceptance x_trial holds x + step d.
  template <value_objective O>
  line_search_result search(O& objective, const optimizer_point& x, double value, const nonmonotone_reference& reference,
                            const optimizer_point& g, const optimizer_point& d, double initial_step,
                            const optimizer_vector& x_trial) const {
    return detail::backtracking_search(objective, x, value, std::max(value, reference.value), g, d, initial_step,
                                       x_trial, sufficient_decrease, min_shrink, max_shrink, max_evaluations,
                                       resolution);
  }
};

}  // namespace laplace::optimizers
