#pragma once

// The proximal bundle method (note/optimizers.md, decisions 4 and 12.7) for a
// convex objective that is not differentiable everywhere, from its value and
// one subgradient per point (gradient_objective). The case it was written for
// is least absolute deviations, sum_i |y_i - x_i'beta| / sigma, the negative
// log-likelihood of the Laplace distribution in its location with covariates:
// it has no simple proximal operator, and a smooth method stalls on its kinks.
//
// The model. Each point y_j visited gives a cut, the affine minorant
// l_j(z) = f(y_j) + g_j'(z - y_j) with g_j a subgradient at y_j; the cuts are
// kept as the subgradient and the linearization error at the stability centre
// x (the best point so far), alpha_j = f(x) - l_j(x) >= 0, which is updated on
// every move of the centre by alpha_j += f(x_new) - f(x) - g_j'(x_new - x)
// rather than recomputed from an intercept (no cancellation between f and
// g'x at large n). The model is f_hat(x + d) = f(x) + max_j (g_j'd - alpha_j).
//
// The step. d = argmin_d f_hat(x + d) + |d|^2 / (2 t), whose dual is the
// quadratic program over the simplex
//
//   min_lambda  (t/2) |G lambda|^2 + alpha'lambda,   lambda >= 0, sum lambda = 1,
//
// with d = -t p, p = G lambda the aggregate subgradient and
// alpha_p = alpha'lambda the aggregate linearization error. The model
// predicts the decrease v = t |p|^2 + alpha_p. The trial point x + d is a
// serious step (the centre moves) when f(x + d) <= f(x) - m_L v, a null step
// otherwise (the centre stays and the cut at x + d enriches the model).
//
// The dual is solved exactly by an active-set method of the kind of Wolfe's
// minimum-norm-point algorithm (1976): the working set holds cuts whose
// (g_j, 1) are linearly independent, so that the equality-constrained problem
// on it has one solution (two Cholesky solves of t G_F'G_F + s^2 1 1', with
// s^2 the mean of t |g_j|^2 to keep the two parts on one scale); a cut whose
// (g_j, 1) depends on the working set enters by replacing one of them along a
// direction on which the dual falls linearly. An exact dual is what lets the
// method reach the vertex of a polyhedral function (LAD) exactly. FISTA on the
// dual with the exact projection on the simplex was measured and set aside: on
// the LAD fits of the tests it stopped 5e-9 to 1e-7 from the vertex and took 4
// to 24 times as long (docs/numerics.md).
//
// The proximity parameter t (Kiwiel 1990, "Proximity control in bundle
// methods for convex nondifferentiable minimization", Math. Programming 46).
// With u = 1/t, and u_int = 2 u (1 - (f(x + d) - f(x)) / (-v)) the minimizer
// of the quadratic interpolating f along d:
//   serious step: u_int when the decrease reached m_R v and the previous step
//     was serious too, u/2 after more than 3 serious steps in a row; then
//     max(that, u/10, u_min); eps_v = max(eps_v, 2 v);
//   null step: eps_v = min(eps_v, the stationarity measure below); u_int when
//     the new cut's linearization error at x exceeds max(eps_v, 10 v) and more
//     than 3 null steps came in a row; then min(that, 10 u).
// Kiwiel's eps_v reads |p| + alpha_p, which mixes the units of a gradient and
// of the objective; here it reads the stationarity measure, in the
// objective's units. The first t is initial_step / |g_0|: a step of length
// initial_step in x (optimizers7 multiplied the subgradient by a raw t0, and a
// first step 233 units long on Rosenbrock left t unable to recover). u_min is
// 1e-10 u_0. A trial point where f is not finite is refused: no cut, t / 10.
//
// Compression. When the bundle holds more than max_bundle_size cuts, the
// cuts with lambda_j = 0 are dropped first (the newest is always kept); if
// that is not enough, the aggregate cut (p, alpha_p), formed at the current
// centre before the step, replaces all but the newest max_bundle_size - 1.
// The aggregate is a minorant of f (a convex combination of minorants), it
// keeps the dual's solution feasible, and its linearization error is
// alpha_p at the centre where it was formed (optimizers7 once stored it at the
// new centre after a serious step, where it can lie above f).
//
// Stationarity, in the objective's units (decision 5). The model is a
// minorant of f, so f(x) - f* <= f(x) - min_z f_hat(z), the cutting-plane gap,
// which is min{alpha'lambda : G lambda = 0, lambda in the simplex} by duality
// (infinite while 0 is not in the hull of the cuts' subgradients). The measure
// is the model's predicted decrease at a large proximity Gamma,
//
//   v(Gamma) = Gamma |p_Gamma|^2 + alpha_Gamma = f(x) - f_hat(x + d_Gamma),
//
// from a second solve of the dual at Gamma, warm-started from the first; it
// increases with Gamma towards the gap, and falls short of it by at most
// |d*|^2 / (2 Gamma), d* the step to the model's minimizer. Gamma is
// certificate_ratio times max(gamma, t), gamma the largest of t_0 and every t
// of a serious step: the longest proximity at which the model was confirmed.
// The measure is given to the stop rule as the decrement. The secant ratios
// s's / s'y between consecutive centres were tried as a third term of gamma
// (as BFGS's gamma_max) and dropped: on a polyhedral f they are 0 or infinite,
// and on the CUDA build one pair with s'y at rounding level set gamma to 1.2e6
// on d4, so that Gamma |p|^2 held the measure at 2.2e-6 on the vertex itself
// (f - f* = 9e-12) for 100000 null steps.
// Two measures were measured and set aside. (1/2) gamma |p_t|^2 + alpha_t from
// the step's own dual undershot f - f* by 5 to 54 times on the LAD fits of the
// tests (d3, d4, d5): near a vertex f - f* behaves like |p| times the distance
// to it, not like a quadratic decrement, and the converged point was 2.4e-4
// away from the vertex. A measure carrying the current t (optimizers7's -v)
// is driven under any tolerance by null steps that shrink t, and declared
// convergence on an objective that was finite only at its start; gamma never
// falls on a null step.
//
// Statuses: converged (or at resolution) on the stop rule; stalled when the
// trial point equals the centre in every coordinate (t has collapsed below
// the rounding of x), so that no step can change anything, or when a null
// step's predicted decrease v is below 64 eps |f| with t at its cap (such a
// null step otherwise multiplies t by 10, since the model sees nothing
// resolvable at that proximity); line_search_failed
// when a subgradient or the dual is not finite (the model diverged);
// iteration_budget and evaluation_budget. iterations counts trial points
// (serious and null steps); the gradient reported is the subgradient at the
// centre; the trace is called once per serious step.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace laplace::optimizers {

namespace detail {

// What the bundle method gives a stop rule. The gradient is not given: a
// subgradient at a kink is not small at the minimum.
inline constexpr unsigned bundle_provides = stop_needs_decrement | stop_needs_previous;

// In-place Cholesky of the m x m row-major matrix a (lower triangle used);
// false when a pivot is not positive or falls below relative_floor times its
// diagonal entry (a dependent column).
inline bool bundle_cholesky(std::vector<double>& a, int m, double relative_floor) {
  for (int j = 0; j < m; ++j) {
    const double diagonal = a[std::size_t(j) * m + j];
    double s = diagonal;
    for (int k = 0; k < j; ++k) s -= a[std::size_t(j) * m + k] * a[std::size_t(j) * m + k];
    if (!(s > relative_floor * diagonal)) return false;
    const double l = std::sqrt(s);
    a[std::size_t(j) * m + j] = l;
    for (int i = j + 1; i < m; ++i) {
      double r = a[std::size_t(i) * m + j];
      for (int k = 0; k < j; ++k) r -= a[std::size_t(i) * m + k] * a[std::size_t(j) * m + k];
      a[std::size_t(i) * m + j] = r / l;
    }
  }
  return true;
}

// solves L L' x = b in place, L from bundle_cholesky
inline void bundle_cholesky_solve(const std::vector<double>& l, int m, std::vector<double>& b) {
  for (int i = 0; i < m; ++i) {
    double r = b[std::size_t(i)];
    for (int k = 0; k < i; ++k) r -= l[std::size_t(i) * m + k] * b[std::size_t(k)];
    b[std::size_t(i)] = r / l[std::size_t(i) * m + i];
  }
  for (int i = m - 1; i >= 0; --i) {
    double r = b[std::size_t(i)];
    for (int k = i + 1; k < m; ++k) r -= l[std::size_t(k) * m + i] * b[std::size_t(k)];
    b[std::size_t(i)] = r / l[std::size_t(i) * m + i];
  }
}

// The dual of the bundle subproblem, min (t/2) lambda'Q lambda + alpha'lambda over
// the simplex, Q = G'G (k x k, row-major), by an active-set method in the manner
// of Wolfe's minimum-norm-point algorithm. lambda holds a feasible warm start (its
// support must have independent (g_j, 1), as the previous solution's has) or all
// zeros; on return the solution. false when the solve fails (non-finite data, or
// the cycle cap).
inline bool bundle_dual_active_set(const std::vector<double>& Q, const std::vector<double>& alpha, double t,
                                   std::vector<double>& lambda, int max_cycles = 10000) {
  const int k = static_cast<int>(alpha.size());
  constexpr double dependence_floor = 1e-12;  // relative pivot of a dependent (g_j, 1)
  double s2 = 0.0;                            // the scale of the column of ones
  for (int j = 0; j < k; ++j) s2 += t * Q[std::size_t(j) * k + j];
  s2 = s2 > 0.0 ? s2 / k : 1.0;
  if (!std::isfinite(s2)) return false;

  std::vector<int> F;
  for (int j = 0; j < k; ++j)
    if (lambda[std::size_t(j)] > 0.0) F.push_back(j);
  if (F.empty()) {
    int best = 0;
    for (int j = 1; j < k; ++j)
      if (0.5 * t * Q[std::size_t(j) * k + j] + alpha[std::size_t(j)] <
          0.5 * t * Q[std::size_t(best) * k + best] + alpha[std::size_t(best)])
        best = j;
    std::fill(lambda.begin(), lambda.end(), 0.0);
    lambda[std::size_t(best)] = 1.0;
    F.push_back(best);
  }
  std::vector<double> B, x1, x2, mu, score(static_cast<std::size_t>(k));
  auto factor = [&](const std::vector<int>& set) {
    const int m = static_cast<int>(set.size());
    B.assign(std::size_t(m) * m, 0.0);
    for (int a = 0; a < m; ++a)
      for (int b = 0; b <= a; ++b)
        B[std::size_t(a) * m + b] = t * Q[std::size_t(set[std::size_t(a)]) * k + set[std::size_t(b)]] + s2;
    return bundle_cholesky(B, m, dependence_floor);
  };
  int last_added = -1;
  for (int cycle = 0; cycle < max_cycles; ++cycle) {
    // minor cycles: the minimizer on the affine hull of F, kept inside the simplex
    for (;;) {
      const int m = static_cast<int>(F.size());
      if (!factor(F)) return false;  // cannot happen while F stays independent
      x1.assign(std::size_t(m), 1.0);
      x2.resize(std::size_t(m));
      for (int a = 0; a < m; ++a) x2[std::size_t(a)] = alpha[std::size_t(F[std::size_t(a)])];
      bundle_cholesky_solve(B, m, x1);
      bundle_cholesky_solve(B, m, x2);
      double s1 = 0.0, sa = 0.0;
      for (int a = 0; a < m; ++a) s1 += x1[std::size_t(a)], sa += x2[std::size_t(a)];
      const double scale = (1.0 + sa) / s1;
      mu.resize(std::size_t(m));
      bool inside = true;
      for (int a = 0; a < m; ++a) {
        mu[std::size_t(a)] = scale * x1[std::size_t(a)] - x2[std::size_t(a)];
        if (!std::isfinite(mu[std::size_t(a)])) return false;
        inside = inside && mu[std::size_t(a)] > 0.0;
      }
      if (inside) {
        for (int a = 0; a < m; ++a) lambda[std::size_t(F[std::size_t(a)])] = mu[std::size_t(a)];
        break;
      }
      // move towards mu until the first coordinate reaches 0, and drop it
      double theta = 1.0;
      int drop = -1;
      for (int a = 0; a < m; ++a) {
        const double la = lambda[std::size_t(F[std::size_t(a)])], ma = mu[std::size_t(a)];
        if (ma <= 0.0 && la - ma > 0.0 && la / (la - ma) < theta) theta = la / (la - ma), drop = a;
      }
      if (drop < 0) {  // mu <= 0 only where lambda is 0 already: drop those
        for (int a = 0; a < m; ++a)
          if (mu[std::size_t(a)] <= 0.0) drop = a;
        theta = 0.0;
      }
      for (int a = 0; a < m; ++a) {
        double& la = lambda[std::size_t(F[std::size_t(a)])];
        la += theta * (mu[std::size_t(a)] - la);
      }
      // a cut that entered and left at once: its gain was rounding
      if (F[std::size_t(drop)] == last_added && theta == 0.0) {
        lambda[std::size_t(F[std::size_t(drop)])] = 0.0;
        return true;
      }
      lambda[std::size_t(F[std::size_t(drop)])] = 0.0;
      F.erase(F.begin() + drop);
      double sum = 0.0;
      for (int j : F) sum += lambda[std::size_t(j)];
      for (int j : F) lambda[std::size_t(j)] /= sum;
    }
    // major cycle: the cut with the lowest dual gradient (t Q lambda + alpha)_j
    // A score's rounding is about eps (t max_j Q_jj + max_j alpha_j): t Q lambda is a
    // sum of terms up to t max Q_jj that nearly cancel near the solution
    double w = 0.0, scale = 0.0;
    for (int j = 0; j < k; ++j) {
      double r = 0.0;
      for (int i : F) r += Q[std::size_t(j) * k + i] * lambda[std::size_t(i)];
      score[std::size_t(j)] = t * r + alpha[std::size_t(j)];
      scale = std::max({scale, t * Q[std::size_t(j) * k + j], std::abs(alpha[std::size_t(j)])});
    }
    for (int i : F) w += lambda[std::size_t(i)] * score[std::size_t(i)];
    int enter = -1;
    for (int j = 0; j < k; ++j)
      if (lambda[std::size_t(j)] == 0.0 && (enter < 0 || score[std::size_t(j)] < score[std::size_t(enter)])) enter = j;
    if (enter < 0 || score[std::size_t(enter)] >= w - 1e-13 * scale) return true;  // KKT within rounding
    std::vector<int> G = F;
    G.push_back(enter);
    if (!factor(G)) {
      // (g_enter, 1) = sum_F c (g_j, 1): enter along e_enter - c, on which the dual
      // falls linearly, until a cut of F leaves
      const int m = static_cast<int>(F.size());
      if (!factor(F)) return false;
      std::vector<double> c(static_cast<std::size_t>(m));
      for (int a = 0; a < m; ++a) c[std::size_t(a)] = t * Q[std::size_t(F[std::size_t(a)]) * k + enter] + s2;
      bundle_cholesky_solve(B, m, c);
      double theta = std::numeric_limits<double>::infinity();
      int drop = -1;
      for (int a = 0; a < m; ++a)
        if (c[std::size_t(a)] > 0.0 && lambda[std::size_t(F[std::size_t(a)])] / c[std::size_t(a)] < theta)
          theta = lambda[std::size_t(F[std::size_t(a)])] / c[std::size_t(a)], drop = a;
      if (drop < 0) return false;
      for (int a = 0; a < m; ++a) lambda[std::size_t(F[std::size_t(a)])] -= theta * c[std::size_t(a)];
      lambda[std::size_t(F[std::size_t(drop)])] = 0.0;
      lambda[std::size_t(enter)] = theta;
      F[std::size_t(drop)] = enter;
      for (int a = 0; a < m; ++a) lambda[std::size_t(F[std::size_t(a)])] = std::max(lambda[std::size_t(F[std::size_t(a)])], 0.0);
      double sum = 0.0;
      for (int j : F) sum += lambda[std::size_t(j)];
      for (int j : F) lambda[std::size_t(j)] /= sum;
      // a coordinate that reached 0 on the way stays in F; the minor cycle drops it
      if (theta == 0.0) {
        F.erase(std::remove(F.begin(), F.end(), enter), F.end());
        lambda[std::size_t(enter)] = 0.0;
        return true;
      }
    } else {
      F.push_back(enter);
    }
    last_added = enter;
  }
  return false;
}

}  // namespace detail

/**
 * \ingroup optimizers_bundle
 * \brief Proximal bundle minimization of a convex, possibly non-smooth objective.
 *
 * Minimizes a convex objective modeling `gradient_objective` on the host from
 * its function value and one subgradient per point, without requiring
 * differentiability everywhere or a proximal operator. Runs are invoked through
 * minimize() or `run()`. The template parameter `StopRule` defaults to
 * `decrement_stop_rule`, and `stop_rule_for` admits any stopping rule
 * requiring at most `stop_needs_decrement` and `stop_needs_previous`; rules
 * that read `gradient` are rejected at compile time because a subgradient at a
 * kink need not vanish at a minimum.
 *
 * ### Cutting-plane model and dual step
 *
 * Each visited point \f$w_j\f$ contributes an affine minorant
 * \f$f(w_j) + g_j^\top(z - w_j)\f$, stored at the stability centre \f$x\f$ as
 * the subgradient \f$g_j\f$ and linearization error
 * \f$\alpha_j = f(x) - f(w_j) - g_j^\top(x - w_j) \ge 0\f$ updated
 * incrementally on each move of \f$x\f$. The cutting-plane model (Kelley, 1960)
 * is \f$\hat{f}(x + d) = f(x) + \max_j (g_j^\top d - \alpha_j)\f$. At
 * proximity \f$t > 0\f$, the step is \f$d = -t p\f$ with \f$p = G\lambda\f$,
 * where \f$G\f$ has columns \f$g_j\f$ and \f$\lambda\f$ minimizes
 * \f$\frac{t}{2}\lVert G\lambda \rVert^2 + \alpha^\top \lambda\f$ over the
 * unit simplex. The dual is solved by an active-set method in the manner of
 * Wolfe's minimum-norm-point algorithm (Wolfe, 1976), maintaining independent
 * extended columns \f$(g_j, 1)\f$ via two Cholesky solves of
 * \f$t G_F^\top G_F + s^2 \mathbf{1}\mathbf{1}^\top\f$.
 *
 * The model predicts a decrease \f$v = t\lVert p \rVert^2 + \alpha^\top \lambda\f$.
 * A trial point \f$x + d\f$ is accepted as a serious step (moving the centre
 * and invoking `trace`) when \f$\Delta = f(x) - f(x + d) \ge m_L v\f$, where
 * \f$m_L\f$ is `serious_fraction`; otherwise it is a null step that keeps
 * \f$x\f$ and adds the new cut. Non-finite trial evaluations are refused
 * without adding a cut and divide \f$t\f$ by \f$10\f$. Starting from
 * \f$t_0 = h_0 / \lVert g_0 \rVert\f$ where \f$h_0\f$ is `initial_step` (or
 * \f$t_0 = h_0\f$ when \f$g_0 = 0\f$), \f$t\f$ is updated by Kiwiel's rule
 * (Kiwiel, 1990)
 * using the stationarity measure below as the variation estimate. When the
 * bundle exceeds `max_bundle_size` cuts (default \f$n + 2\f$ when \f$0\f$),
 * inactive cuts with \f$\lambda_j = 0\f$ are dropped first (keeping the newest
 * cut); if still over budget, the aggregate cut \f$(p, \alpha^\top \lambda)\f$
 * replaces all cuts except the newest `max_bundle_size` \f$- 1\f$.
 *
 * ### Stationarity
 *
 * Stationarity is measured by the model's predicted decrease at a large
 * proximity \f$\Gamma = c \max(\gamma, t)\f$, where \f$c\f$ is
 * `certificate_ratio` and \f$\gamma\f$ is the largest of \f$t_0\f$ and every
 * \f$t\f$ of a serious step:
 * \f[
 *   v(\Gamma) = \Gamma \lVert p_\Gamma \rVert^2 + \alpha^\top \lambda_\Gamma,
 * \f]
 * obtained from a second warm-started solve of the dual at \f$\Gamma\f$.
 * Because \f$\gamma\f$ never shrinks on null steps, null steps cannot drive
 * \f$v(\Gamma)\f$ below tolerance by shrinking \f$t\f$. The value
 * \f$v(\Gamma)\f$ is passed to `stop_rule` as `decrement`.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `decrement_stop_rule{}` | Stopping rule queried on `decrement` and `previous_*` at each iteration |
 * | `max_iterations` | \f$10^5\f$ | Maximum number of trial points (serious and null steps combined) |
 * | `max_value_evaluations` | \f$10^7\f$ | Budget on objective `value_gradient()` calls |
 * | `initial_step` | \f$1\f$ | Euclidean length in \f$x\f$ of the first trial step |
 * | `max_bundle_size` | \f$0\f$ | Maximum cuts retained; \f$0\f$ selects \f$n + 2\f$, positive values clamp to at least \f$2\f$ |
 * | `serious_fraction` | \f$0.1\f$ | Sufficient decrease fraction \f$m_L\f$ required for a serious step |
 * | `interpolation_fraction` | \f$0.5\f$ | Decrease fraction \f$m_R\f$ permitting \f$t\f$ growth via interpolation |
 * | `certificate_ratio` | \f$100\f$ | Multiplier \f$c\f$ in \f$\Gamma = c \max(\gamma, t)\f$ for the stationarity dual |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked once per serious step |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from
 * \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(x_0)\f$ or any coordinate
 *   of its subgradient is not finite.
 * - `optimizer_status::converged`, `optimizer_status::converged_at_resolution`,
 *   or `optimizer_status::stopped_on_change` when returned by `stop_rule`.
 * - `optimizer_status::stalled` when the trial point equals the centre in
 *   every coordinate, or when a null step has \f$v \le 64\,\varepsilon\,|f(x)|\f$
 *   with \f$t \ge 10^{10} t_0\f$.
 * - `optimizer_status::line_search_failed` when a dual solve fails or \f$v\f$
 *   or \f$v(\Gamma)\f$ is not finite.
 * - `optimizer_status::iteration_budget` when `max_iterations` trial points
 *   are reached.
 * - `optimizer_status::evaluation_budget` when `max_value_evaluations` runs out.
 *
 * In `optimizer_report`, `point`, `value`, and `gradient` hold the stability
 * centre, its objective value, and its subgradient, `decrement` holds
 * \f$v(\Gamma)\f$, `iterations` counts trial points, `value_evaluations` and
 * `gradient_evaluations` each count `value_gradient()` calls,
 * `curvature_evaluations` is \f$0\f$, and `search_size` is quiet NaN.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_decrement` and
 *   `stop_needs_previous`; `decrement_stop_rule` by default.
 * \see \ref optimizers_bundle, \ref optimizers, minimize(), decrement_stop_rule
 */
template <class StopRule = decrement_stop_rule>
  requires stop_rule_for<StopRule, detail::bundle_provides>
struct bundle_optimizer {
  /**
   * \ingroup optimizers_bundle
   * \brief Stopping rule instance queried at each iteration (default `decrement_stop_rule{}`).
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_bundle
   * \brief Maximum number of trial points, counting both serious and null steps (default \f$10^5\f$).
   */
  int max_iterations = 100000;           // trial points, serious and null
  /**
   * \ingroup optimizers_bundle
   * \brief Budget on objective value and subgradient evaluations (default \f$10^7\f$).
   */
  long max_value_evaluations = 10000000;
  /**
   * \ingroup optimizers_bundle
   * \brief Euclidean length \f$h_0\f$ in \f$x\f$ of the first trial step, setting \f$t_0 = h_0 / \lVert g_0 \rVert\f$ (default \f$1\f$).
   */
  double initial_step = 1.0;             // the first step's length in x: t_0 = initial_step / |g_0|
  /**
   * \ingroup optimizers_bundle
   * \brief Maximum number of cuts kept in the bundle; \f$0\f$ selects \f$n + 2\f$ (default \f$0\f$).
   */
  int max_bundle_size = 0;               // cuts kept; 0 means n + 2 (enough for a vertex in R^n)
  /**
   * \ingroup optimizers_bundle
   * \brief Serious-step sufficient decrease fraction \f$m_L\f$ (default \f$0.1\f$).
   */
  double serious_fraction = 0.1;         // m_L: a serious step falls by at least m_L v
  /**
   * \ingroup optimizers_bundle
   * \brief Decrease fraction \f$m_R\f$ above which consecutive serious steps may enlarge \f$t\f$ by quadratic interpolation (default \f$0.5\f$).
   */
  double interpolation_fraction = 0.5;   // m_R: a serious step falling by m_R v may grow t
  /**
   * \ingroup optimizers_bundle
   * \brief Multiplier \f$c\f$ in \f$\Gamma = c \max(\gamma, t)\f$ for the stationarity certificate dual (default \f$100\f$).
   */
  double certificate_ratio = 100.0;      // Gamma / max(gamma, t) of the certificate's dual
  /**
   * \ingroup optimizers_bundle
   * \brief Optional trace callback invoked once per serious step (default empty).
   */
  optimizer_trace trace;                 // once per serious step

  /**
   * \ingroup optimizers_bundle
   * \brief Minimizes the convex `objective` starting from `x0` by the proximal bundle method.
   *
   * \tparam O Objective type satisfying `gradient_objective`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_bundle, bundle_optimizer, minimize()
   */
  template <gradient_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    const int capacity = max_bundle_size > 0 ? std::max(max_bundle_size, 2) : n + 2;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    optimizer_report report;
    optimizer_vector x("x", n), x_previous("x_previous", n), g_x("g_x", n), trial("trial", n), g_trial("g_trial", n),
        p("p", n);
    Kokkos::deep_copy(x, x0);

    double f_x = objective.value_gradient(x, g_x);
    ++report.value_evaluations, ++report.gradient_evaluations;
    double measure = nan;
    const auto finish = [&](optimizer_status status) {
      report.status = status;
      report.point = x, report.value = f_x, report.gradient = g_x, report.decrement = measure;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    double g_norm = 0.0;
    for (int i = 0; i < n; ++i) g_norm += g_x(i) * g_x(i);
    g_norm = std::sqrt(g_norm);
    if (!std::isfinite(f_x) || !std::isfinite(g_norm)) return finish(optimizer_status::start_not_admissible);

    // the bundle: subgradients (row j of cuts), linearization errors at x, Q = G'G
    std::vector<std::vector<double>> cuts;
    std::vector<double> alpha, lambda;
    std::vector<std::vector<double>> Q;
    auto add_cut = [&](const optimizer_vector& g, double error) {
      cuts.emplace_back(g.data(), g.data() + n);
      alpha.push_back(std::max(error, 0.0));  // >= 0 by convexity; rounding only
      lambda.push_back(0.0);
      const std::size_t k = cuts.size();
      for (auto& row : Q) row.push_back(0.0);
      Q.emplace_back(k, 0.0);
      for (std::size_t j = 0; j < k; ++j) {
        double d = 0.0;
        for (int i = 0; i < n; ++i) d += cuts[j][std::size_t(i)] * cuts[k - 1][std::size_t(i)];
        Q[j][k - 1] = Q[k - 1][j] = d;
      }
    };
    auto keep_only = [&](const std::vector<int>& kept) {
      std::vector<std::vector<double>> c2, q2;
      std::vector<double> a2, l2;
      for (int a : kept) {
        c2.push_back(cuts[std::size_t(a)]), a2.push_back(alpha[std::size_t(a)]), l2.push_back(lambda[std::size_t(a)]);
        std::vector<double> row;
        for (int b : kept) row.push_back(Q[std::size_t(a)][std::size_t(b)]);
        q2.push_back(row);
      }
      cuts.swap(c2), alpha.swap(a2), lambda.swap(l2), Q.swap(q2);
    };
    add_cut(g_x, 0.0);

    const double t0 = g_norm > 0.0 ? initial_step / g_norm : initial_step;
    const double t_max = 1e10 * t0;  // Kiwiel's u_min = 1e-10 u_0
    double t = t0, gamma = t0, eps_v = std::numeric_limits<double>::infinity(), f_previous = nan;
    int i_t = 0;  // Kiwiel's i_u: > 0 serious steps in a row, < 0 null steps in a row
    bool has_previous = false;
    std::vector<double> Qflat, lambda_certificate;

    for (int it = 0;; ++it) {
      // the dual and the aggregate at the centre x
      const std::size_t k = cuts.size();
      Qflat.resize(k * k);
      for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b) Qflat[a * k + b] = Q[a][b];
      if (!detail::bundle_dual_active_set(Qflat, alpha, t, lambda)) return finish(optimizer_status::line_search_failed);
      double alpha_p = 0.0, p2 = 0.0;
      for (int i = 0; i < n; ++i) p(i) = 0.0;
      for (std::size_t j = 0; j < k; ++j) {
        if (lambda[j] == 0.0) continue;
        alpha_p += lambda[j] * alpha[j];
        for (int i = 0; i < n; ++i) p(i) += lambda[j] * cuts[j][std::size_t(i)];
      }
      for (int i = 0; i < n; ++i) p2 += p(i) * p(i);
      const double v = t * p2 + alpha_p;  // the decrease the model predicts
      // the certificate: the model's predicted decrease at the proximity Gamma = c max(gamma, t)
      const double Gamma = certificate_ratio * std::max(gamma, t);
      lambda_certificate = lambda;
      if (!detail::bundle_dual_active_set(Qflat, alpha, Gamma, lambda_certificate))
        return finish(optimizer_status::line_search_failed);
      {
        double a = 0.0, q = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
          a += lambda_certificate[j] * alpha[j];
          if (lambda_certificate[j] == 0.0) continue;
          for (std::size_t l = 0; l < k; ++l) q += lambda_certificate[j] * Q[j][l] * lambda_certificate[l];
        }
        measure = Gamma * std::max(q, 0.0) + a;
      }
      if (!std::isfinite(v) || !std::isfinite(measure)) return finish(optimizer_status::line_search_failed);

      stop_rule_input in;
      in.iteration = it, in.value = f_x, in.point = x, in.decrement = measure;
      in.has_previous = has_previous, in.previous_value = f_previous, in.previous_point = x_previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none)
        return finish(verdict == stop_verdict::converged                 ? optimizer_status::converged
                      : verdict == stop_verdict::converged_at_resolution ? optimizer_status::converged_at_resolution
                                                                         : optimizer_status::stopped_on_change);
      if (it >= max_iterations) return finish(optimizer_status::iteration_budget);
      if (report.value_evaluations >= max_value_evaluations) return finish(optimizer_status::evaluation_budget);

      // the trial point
      bool moves = false;
      for (int i = 0; i < n; ++i) {
        trial(i) = x(i) - t * p(i);
        moves = moves || trial(i) != x(i);
      }
      if (!moves) return finish(optimizer_status::stalled);
      const double f_trial = objective.value_gradient(trial, g_trial);
      ++report.value_evaluations, ++report.gradient_evaluations;
      report.iterations = it + 1;
      bool finite = std::isfinite(f_trial);
      for (int i = 0; i < n; ++i) finite = finite && std::isfinite(g_trial(i));
      if (!finite) {  // refused: nothing learned
        t *= 0.1, i_t = std::min(i_t - 1, -1);
        continue;
      }
      // the aggregate cut, formed at this centre (kept by the compression below)
      const std::vector<double> p_cut(p.data(), p.data() + n);
      const double alpha_p_cut = alpha_p;

      const double fall = f_x - f_trial;  // > 0 when the trial point is lower
      const double u = 1.0 / t;
      const double u_int = 2.0 * u * (1.0 - fall / v);  // the minimizer of the interpolating quadratic, in u
      double u_new = u;
      const bool serious = fall >= serious_fraction * v;
      if (serious) {
        // serious step: the centre moves, every linearization error with it
        for (std::size_t j = 0; j < k; ++j) {
          double gs = 0.0;
          for (int i = 0; i < n; ++i) gs += cuts[j][std::size_t(i)] * (trial(i) - x(i));
          alpha[j] = std::max(alpha[j] - fall - gs, 0.0);  // f(x_new) - f(x) = -fall
        }
        Kokkos::deep_copy(x_previous, x);
        f_previous = f_x, has_previous = true;
        Kokkos::deep_copy(x, trial);
        Kokkos::deep_copy(g_x, g_trial);
        f_x = f_trial;
        gamma = std::max(gamma, t);
        if (fall >= interpolation_fraction * v && i_t > 0) u_new = u_int;
        else if (i_t > 3) u_new = u / 2.0;
        u_new = std::max({u_new, u / 10.0, 1.0 / t_max});
        eps_v = std::max(eps_v, 2.0 * v);
        i_t = std::max(i_t + 1, 1);
        if (u_new != u) i_t = 1;
        add_cut(g_trial, 0.0);
        if (trace) trace(optimizer_trace_point{report.iterations, x, f_x, g_x, measure, t});
      } else {
        // null step: the cut at the trial point, its error at the centre
        double gs = 0.0;
        for (int i = 0; i < n; ++i) gs += g_trial(i) * (x(i) - trial(i));
        const double error = f_x - f_trial - gs;
        eps_v = std::min(eps_v, measure);
        if (error > std::max(eps_v, 10.0 * v) && i_t < -3) u_new = u_int;
        u_new = std::min(u_new, 10.0 * u);
        i_t = std::min(i_t - 1, -1);
        if (u_new != u) i_t = -1;
        // a decrease the objective cannot resolve: a longer step, or the end
        if (v <= 64.0 * std::numeric_limits<double>::epsilon() * std::abs(f_x)) {
          if (t >= t_max) return finish(optimizer_status::stalled);
          u_new = std::max(u / 10.0, 1.0 / t_max), i_t = -1;
        }
        add_cut(g_trial, error);
      }
      t = 1.0 / u_new;

      // compression: drop the inactive cuts, then fold into the aggregate
      if (static_cast<int>(cuts.size()) > capacity) {
        const int newest = static_cast<int>(cuts.size()) - 1;
        std::vector<int> kept;
        for (int j = 0; j < newest; ++j)
          if (lambda[std::size_t(j)] > 0.0) kept.push_back(j);
        kept.push_back(newest);
        keep_only(kept);
      }
      if (static_cast<int>(cuts.size()) > capacity) {
        // the aggregate's error at the current centre: alpha_p at the old centre,
        // moved like every other cut when the step was serious
        double error = alpha_p_cut;
        if (serious) {  // the centre moved in this iteration: x_previous is the old one
          double gs = 0.0;
          for (int i = 0; i < n; ++i) gs += p_cut[std::size_t(i)] * (x(i) - x_previous(i));
          error = alpha_p_cut - (f_previous - f_x) - gs;
        }
        const std::vector<std::vector<double>> newest_cuts(cuts.end() - (capacity - 1), cuts.end());
        const std::vector<double> newest_alpha(alpha.end() - (capacity - 1), alpha.end());
        cuts.clear(), alpha.clear(), lambda.clear(), Q.clear();
        optimizer_vector agg("aggregate", n);
        for (int i = 0; i < n; ++i) agg(i) = p_cut[std::size_t(i)];
        add_cut(agg, error);
        lambda[0] = 1.0;  // the aggregate alone reproduces the dual's solution
        optimizer_vector g("cut", n);
        for (int j = 0; j < capacity - 1; ++j) {
          for (int i = 0; i < n; ++i) g(i) = newest_cuts[std::size_t(j)][std::size_t(i)];
          add_cut(g, newest_alpha[std::size_t(j)]);
        }
      }
    }
  }
};

}  // namespace laplace::optimizers
