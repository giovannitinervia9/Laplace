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
 * \see \ref optimizers, line_search_result, armijo_line_search, wolfe_line_search
 */
enum class line_search_outcome {
  accepted,
  ///< A trial step satisfying the search conditions was found.
  not_descent_direction,  // phi'(0) >= 0 or not finite
  ///< Initial directional derivative \f$\phi'(0) = g^\top d\f$ is non-negative or not finite.
  below_resolution,       // s0 |phi'(0)| below the resolution given to the search
  ///< Predicted decrease \f$s_0 |\phi'(0)|\f$ at the initial step is below the search resolution.
  step_below_rounding,    // x + s d == x, or the Wolfe bracket narrower than the rounding of s
  ///< Trial point equals `x` in every coordinate, or the Wolfe bracket is narrower than \f$4\varepsilon\f$ times its larger end.
  evaluation_budget       // max_evaluations reached without an acceptable step
  ///< Evaluation budget `max_evaluations` was reached without accepting a step.
};

/**
 * \ingroup optimizers
 * \brief Result returned by armijo_line_search::search() and wolfe_line_search::search().
 *
 * \see \ref optimizers, line_search_outcome, armijo_line_search, wolfe_line_search
 */
struct line_search_result {
  /**
   * \ingroup optimizers
   * \brief Outcome of the line search.
   */
  line_search_outcome outcome = line_search_outcome::evaluation_budget;
  /**
   * \ingroup optimizers
   * \brief Accepted step length \f$s\f$; on failure the last step tried, 0 when no trial was made.
   */
  double step = 0.0;   // the accepted step (the last one tried otherwise)
  /**
   * \ingroup optimizers
   * \brief Objective value \f$\phi(s) = f(x + s d)\f$ when accepted, or quiet NaN otherwise.
   */
  double value = std::numeric_limits<double>::quiet_NaN();  // phi(step) when accepted
  /**
   * \ingroup optimizers
   * \brief Directional derivative \f$\phi'(s) = \nabla f(x + s d)^\top d\f$ when accepted by wolfe_line_search, or quiet NaN otherwise.
   */
  double slope = std::numeric_limits<double>::quiet_NaN();  // phi'(step), Wolfe only
  /**
   * \ingroup optimizers
   * \brief Number of objective value evaluations performed.
   */
  int value_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Number of gradient evaluations performed.
   */
  int gradient_evaluations = 0;
  /**
   * \ingroup optimizers
   * \brief Number of trial points refused due to non-finite value or directional derivative.
   */
  int refused_evaluations = 0;  // trial points with a non-finite value or slope
  /**
   * \ingroup optimizers
   * \brief Returns whether a step was accepted.
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

}  // namespace detail

/**
 * \ingroup optimizers
 * \brief Backtracking line search enforcing the Armijo sufficient-decrease condition.
 *
 * Searches along a descent direction \f$d\f$ for a step \f$s > 0\f$ satisfying
 * \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$, where \f$\phi(s) = f(x + s d)\f$ and \f$\phi'(0) = g^\top d < 0\f$.
 * Uses quadratic interpolation on the first backtrack and cubic interpolation afterwards, safeguarded between `min_shrink` and `max_shrink` times \f$s\f$, evaluating only objective values.
 *
 * \see \ref optimizers, wolfe_line_search, line_search_result, line_search_outcome
 */
struct armijo_line_search {
  /**
   * \ingroup optimizers
   * \brief Sufficient-decrease constant \f$c_1\f$.
   */
  double sufficient_decrease = 1e-4;  // c1
  /**
   * \ingroup optimizers
   * \brief Lower safeguard factor on each backtracking shrink.
   */
  double min_shrink = 0.1;            // the next step lies in [min_shrink s, max_shrink s]
  /**
   * \ingroup optimizers
   * \brief Upper safeguard factor on each backtracking shrink, also used after a refused trial point.
   */
  double max_shrink = 0.5;            // also the factor after a refused point
  /**
   * \ingroup optimizers
   * \brief Maximum number of trial value evaluations.
   */
  int max_evaluations = 60;
  /**
   * \ingroup optimizers
   * \brief Optional resolution threshold; when positive, fails with `line_search_outcome::below_resolution` before evaluating if \f$s_0 |\phi'(0)|\f$ is less than `resolution`.
   */
  double resolution = 0.0;            // 0: no resolution test

  /**
   * \ingroup optimizers
   * \brief Searches from `x` along `d` starting at `initial_step`.
   *
   * On acceptance, `x_trial` holds \f$x + s d\f$ for the accepted step \f$s\f$.
   * Non-finite trial values are counted in `refused_evaluations` and shrink the step by `max_shrink` without interpolation.
   *
   * \tparam O Objective type satisfying `value_objective`.
   * \param objective Objective to evaluate at trial points.
   * \param x Current point \f$x\f$.
   * \param value Objective value \f$f(x)\f$ at `x`.
   * \param g Gradient \f$g\f$ at `x`.
   * \param d Search direction \f$d\f$.
   * \param initial_step Initial step length \f$s_0 > 0\f$.
   * \param x_trial Output buffer receiving trial points \f$x + s d\f$.
   * \return A line_search_result recording the outcome, accepted step and value, and evaluation counts.
   */
  // On acceptance x_trial holds x + step d.
  template <value_objective O>
  line_search_result search(O& objective, const optimizer_point& x, double value, const optimizer_point& g,
                            const optimizer_point& d, double initial_step, const optimizer_vector& x_trial) const {
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
      if (std::isfinite(ft) && ft - value <= sufficient_decrease * s * slope) {
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
};

/**
 * \ingroup optimizers
 * \brief Bracketing and zoom line search enforcing the strong Wolfe conditions.
 *
 * Searches along a descent direction \f$d\f$ for a step \f$s > 0\f$ satisfying sufficient decrease
 * \f$\phi(s) - \phi(0) \le c_1 s \phi'(0)\f$ and strong curvature \f$|\phi'(s)| \le c_2 |\phi'(0)|\f$,
 * evaluating value and gradient at every trial point.
 *
 * \see \ref optimizers, armijo_line_search, line_search_result, line_search_outcome
 */
struct wolfe_line_search {
  /**
   * \ingroup optimizers
   * \brief Sufficient-decrease constant \f$c_1\f$.
   */
  double sufficient_decrease = 1e-4;  // c1
  /**
   * \ingroup optimizers
   * \brief Strong curvature constant \f$c_2\f$.
   */
  double curvature = 0.9;             // c2 (0.9 for quasi-Newton methods)
  /**
   * \ingroup optimizers
   * \brief Maximum number of trial value-and-gradient evaluations.
   */
  int max_evaluations = 30;
  /**
   * \ingroup optimizers
   * \brief Optional resolution threshold; when positive, fails with `line_search_outcome::below_resolution` before evaluating if \f$s_0 |\phi'(0)|\f$ is less than `resolution`.
   */
  double resolution = 0.0;            // 0: no resolution test

  /**
   * \ingroup optimizers
   * \brief Searches from `x` along `d` starting at `initial_step`.
   *
   * On acceptance, `x_trial` holds \f$x + s d\f$, `g_trial` holds its gradient, and `result.slope` holds \f$\phi'(s)\f$.
   * Non-finite trial values or slopes are counted in `refused_evaluations` and handled by halving or bisecting without interpolation.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to evaluate at trial points.
   * \param x Current point \f$x\f$.
   * \param value Objective value \f$f(x)\f$ at `x`.
   * \param g Gradient \f$g\f$ at `x`.
   * \param d Search direction \f$d\f$.
   * \param initial_step Initial step length \f$s_0 > 0\f$.
   * \param x_trial Output buffer receiving trial points \f$x + s d\f$.
   * \param g_trial Output buffer receiving the gradient at each trial point.
   * \return A line_search_result recording the outcome, accepted step, value, slope, and evaluation counts.
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

}  // namespace laplace::optimizers
