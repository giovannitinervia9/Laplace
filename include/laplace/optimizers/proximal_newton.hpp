#pragma once

// Proximal Newton (note/optimizers.md, section 12, decisions 2-5), on a
// composite objective f = s + r whose smooth part s also gives its curvature H.
//
// At x, with g = grad s(x) and H = H(x), the step d solves the subproblem
//
//   min_d  q(d) = g'd + d'Hd/2 + r(x + d),
//
// inexactly, from d = 0. The inner solver is chosen from the kink geometry of
// the penalties (decision 3):
//
// - coordinate descent on the model when every block is separable with its
//   kinks on the coordinates (coordinate_composite_objective): coordinate j
//   moves to prox_{r_j / a_j}(z_j - (g_j + [H(z - x)]_j) / a_j) with a_j = H_jj,
//   keeping u = H(z - x) up to date, O(n) per coordinate. A coordinate whose
//   H_jj <= rho_j (a weakly convex penalty with too little curvature to make
//   its scalar problem convex) is damped to a_j = rho_j / rho_safety, Levenberg
//   on the diagonal (decision 5), and an unpenalized coordinate with
//   H_jj <= 0 to a small positive value; each damped update is counted in the
//   report (damped_coordinates);
// - otherwise FISTA on the model, with the fixed step 1/L, L the Gershgorin
//   bound max_i sum_j |H_ij| on the largest eigenvalue of H (and
//   L >= rho / rho_safety), and the gradient restart of O'Donoghue and Candes.
//
// The inner solve stops when its measure at the iterate z, (1/2) G'H^-1 G with
// G the model's proximal gradient mapping at z (step 1/L), is at most
// inner_fraction (default 0.01) times the current proximal Newton decrement
// (1/2) d'Hd, or after max_inner_sweeps sweeps or iterations; in that case the
// stop rule reads max(decrement, inner measure / inner_fraction). Either way a
// convergence verdict needs the inner measure below inner_fraction times the
// outer tolerance (decision 4).
//
// Stationarity (decision 4): the proximal Newton decrement (1/2) d'Hd, in the
// objective's units; it is the Newton decrement when r = 0 and is 0 exactly at
// the stationary points of s + r, kinks included. It is NaN when H restricted
// to the free coordinates (the unpenalized ones and the penalized ones not held
// at a kink at the subproblem's solution; all of them without coordinate
// access) has no Cholesky factor; then, when H is finite, it falls back to the
// measure of proximal_gradient_optimizer at x, (1/2) max(t, gamma_max) |G_t|^2
// (t = 1/L as below, gamma_max the largest secant ratio s's/s'y over the
// accepted steps), and the report's decrement_from_prox_mapping says that the
// decrement it gives was this measure (Giovanni, 2026-11-18). NaN only when H
// is not finite.
//
// The step x + s d is accepted on Armijo's condition on the full objective
// (Lee, Sun and Saunders 2014),
//
//   f(x + s d) - f(x) <= c1 s Delta,   Delta = g'd + r(x + d) - r(x) < 0,
//
// halving s from 1. A failed search ends the run: "stalled" when the step fell
// below the rounding of x, "line search failed" otherwise.

#include "laplace/optimizers/dense_curvature_solver.hpp"
#include "laplace/optimizers/line_search.hpp"
#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/proximal_gradient.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_proximal
 * \brief Composite objective exposing block coordinate ranges, separability, and single-coordinate proximal maps.
 *
 * Refines `composite_objective` with `nonsmooth_block_range(b)` (returning `(offset, length)` of block \f$b\f$),
 * `nonsmooth_block_separable(b)` (reporting whether coordinate proximal operators decouple), and
 * `nonsmooth_coordinate_prox(b, j, vj, t)` (evaluating coordinate \f$j\f$ of block \f$b\f$'s proximal operator with step \f$t\f$).
 * Used by `proximal_newton_optimizer` when solving subproblems by coordinate descent.
 *
 * \tparam O Objective type.
 * \see \ref optimizers_proximal, penalized_objective, proximal_newton_optimizer
 */
template <class O>
concept coordinate_composite_objective =
    composite_objective<O> && requires(O& objective, int b, int j, double v, double t) {
      { objective.nonsmooth_block_range(b) } -> std::convertible_to<std::pair<int, int>>;
      { objective.nonsmooth_block_separable(b) } -> std::convertible_to<bool>;
      { objective.nonsmooth_coordinate_prox(b, j, v, t) } -> std::convertible_to<double>;
    };

/**
 * \ingroup optimizers_proximal
 * \brief Choice of inner solver for the proximal Newton quadratic subproblem.
 *
 * Values:
 * - `automatic`: uses coordinate descent when the objective models `coordinate_composite_objective` and every block is separable, and FISTA on the quadratic model otherwise.
 * - `coordinate_descent`: forces cyclic coordinate descent on the quadratic model (requires `coordinate_composite_objective` with separable blocks).
 * - `proximal_gradient`: forces FISTA on the quadratic model.
 *
 * \see \ref optimizers_proximal, proximal_newton_optimizer, coordinate_composite_objective
 */
enum class proximal_newton_inner_solver { automatic, coordinate_descent, proximal_gradient };

namespace detail {

// What proximal Newton gives a stop rule.
inline constexpr unsigned proximal_newton_provides = stop_needs_decrement | stop_needs_previous;

// d'Hd / 2 for d = z - x, from u = H(z - x)
inline double half_curvature(const optimizer_point& z, const optimizer_point& x, const optimizer_point& u) {
  double s = 0.0;
  for (std::size_t i = 0; i < x.extent(0); ++i) s += (z(i) - x(i)) * u(i);
  return 0.5 * s;
}

}  // namespace detail

/**
 * \ingroup optimizers_proximal
 * \brief Proximal Newton optimizer with an inexact quadratic subproblem solve and Armijo backtracking on the full objective.
 *
 * Minimizes \f$f(x) = s(x) + r(x)\f$ for an objective satisfying `composite_objective` and `curvature_objective`
 * with `dense_curvature_matrix`. At \f$x\f$ with \f$g = \nabla s(x)\f$ and \f$H = H(x)\f$, the step \f$d\f$
 * inexactly solves \f$\min_d g^\top d + \frac{1}{2} d^\top H d + r(x + d)\f$ from \f$d = 0\f$ (Lee, Sun and Saunders, 2014).
 *
 * Depending on `inner_solver` (default `automatic`), the subproblem is solved by cyclic coordinate descent
 * (updating coordinate \f$j\f$ in \f$O(n)\f$ operations, damping penalized coordinates with \f$H_{jj} \le \rho_j\f$ to
 * \f$\rho_j / \f$`rho_safety` and unpenalized ones with \f$H_{jj} \le 0\f$ to a small positive floor, counted in
 * `optimizer_report::damped_coordinates`) or by FISTA on the model with step \f$1/L\f$ (Gershgorin bound
 * \f$L = \max_i \sum_j |H_{ij}|\f$, at least \f$\rho / \f$`rho_safety`) and gradient restart. The inner solve stops when
 * \f$\frac{1}{2} G^\top H^{-1} G\f$ (using the Cholesky factor of \f$H\f$, or of \f$H\f$ restricted to the free coordinates,
 * or \f$\frac{1}{2} t \lVert G\rVert^2\f$ when neither exists) is at most `inner_fraction` (default \f$0.01\f$)
 * times \f$\frac{1}{2} d^\top H d\f$, or after `max_inner_sweeps` (default \f$1000\f$) sweeps or iterations.
 *
 * Stationarity is the proximal Newton decrement \f$\delta = \frac{1}{2} d^\top H d\f$ in the objective's units.
 * When \f$H\f$ restricted to the free coordinates (those not held at a kink) has no Cholesky factor and \f$H\f$ is finite,
 * \f$\delta\f$ falls back to the `proximal_gradient_optimizer` measure at the outer point \f$x\f$,
 * \f$\frac{1}{2}\max(t, \gamma_{\max})\lVert G_t\rVert^2\f$, where \f$G_t = (x - \operatorname{prox}_{t r}(x - t\nabla s(x)))/t\f$,
 * \f$t = 1/L\f$ is the step of the inner model (Gershgorin bound, at least \f$\rho / \f$`rho_safety`), and
 * \f$\gamma_{\max}\f$ is the largest secant ratio \f$\delta_x^\top \delta_x / (\delta_x^\top \delta_g)\f$ (with
 * \f$\delta_x^\top \delta_g > 0\f$) over accepted steps for consecutive point and gradient differences \f$\delta_x\f$ and
 * \f$\delta_g\f$. Then `optimizer_report::decrement_from_prox_mapping` is `true`; \f$\delta\f$ is quiet NaN only when \f$H\f$
 * is not finite.
 * If the inner solve misses its target, `stop_rule` receives \f$\max(\delta, \delta_{\mathrm{in}} / \kappa_{\mathrm{in}})\f$
 * where \f$\delta_{\mathrm{in}}\f$ is the last inner measure and \f$\kappa_{\mathrm{in}}\f$ is `inner_fraction`.
 * The step \f$x + \alpha d\f$ uses Armijo backtracking \f$f(x + \alpha d) - f(x) \le c_1 \alpha (g^\top d + r(x + d) - r(x))\f$
 * with \f$c_1 =\f$ `sufficient_decrease` (default \f$10^{-4}\f$), halving \f$\alpha\f$ from \f$1\f$ up to `max_backtracks`
 * (default \f$60\f$) times.
 *
 * Possible `optimizer_report::status` values are `converged`, `converged_at_resolution`, `stopped_on_change`,
 * `stalled` (step fell below rounding in \f$x\f$), `line_search_failed`, `iteration_budget` (`max_iterations`, default \f$200\f$),
 * `evaluation_budget` (`max_value_evaluations`, default \f$10^5\f$), and `start_not_admissible`. In `optimizer_report`,
 * `gradient` is \f$\nabla s\f$ at `point`, `final_factorization` is the last factorization of \f$H\f$ or its free restriction,
 * `damped_coordinates` counts damped coordinate updates, and `decrement_from_prox_mapping` marks the fallback measure.
 * See \ref optimizers_proximal for measured accuracy.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_decrement` and `stop_needs_previous` (`gradient_stop_rule` is rejected at compile time).
 * \see \ref optimizers_proximal, coordinate_composite_objective, proximal_newton_inner_solver, penalized_objective, proximal_gradient_optimizer
 */
template <class StopRule = decrement_stop_rule>
  requires stop_rule_for<StopRule, detail::proximal_newton_provides>
struct proximal_newton_optimizer {
  /// \brief Stopping rule (default `decrement_stop_rule{}`).
  StopRule stop_rule{};
  /// \brief Inner subproblem solver (default `proximal_newton_inner_solver::automatic`).
  proximal_newton_inner_solver inner_solver = proximal_newton_inner_solver::automatic;
  /// \brief Target ratio of the inner measure to \f$\frac{1}{2} d^\top H d\f$ (default \f$0.01\f$).
  double inner_fraction = 0.01;
  /// \brief Maximum sweeps of coordinate descent or iterations of FISTA per subproblem (default \f$1000\f$).
  int max_inner_sweeps = 1000;
  /// \brief Maximum outer iterations (default \f$200\f$).
  int max_iterations = 200;
  /// \brief Maximum evaluations of \f$s\f$ (default \f$100000\f$).
  long max_value_evaluations = 100000;
  /// \brief Armijo sufficient-decrease constant \f$c_1\f$ (default \f$10^{-4}\f$).
  double sufficient_decrease = 1e-4;
  /// \brief Maximum step halvings per Armijo line search (default \f$60\f$).
  int max_backtracks = 60;
  /// \brief Damping factor giving \f$a_j = \rho_j / \f$`rho_safety` and \f$L \ge \rho / \f$`rho_safety` (default \f$0.99\f$).
  double rho_safety = 0.99;
  /// \brief Optional per-iteration trace callback (default empty).
  optimizer_trace trace;

  /**
   * \brief Runs proximal Newton from `x0`.
   *
   * \tparam O Composite objective type satisfying `composite_objective`, `curvature_objective`, and `dense_curvature_matrix` curvature.
   * \param objective Composite objective \f$f = s + r\f$.
   * \param x0 Starting point \f$x_0\f$; \f$f(x_0)\f$ must be finite.
   * \return An `optimizer_report` with `decrement` equal to \f$\frac{1}{2} d^\top H d\f$ (falling back to the `proximal_gradient_optimizer` measure at `point` when \f$H\f$ is finite and the free curvature block lacks a Cholesky factor, and quiet NaN only when \f$H\f$ is not finite), `decrement_from_prox_mapping`, `gradient` equal to \f$\nabla s\f$ at `point`, `final_factorization`, and `damped_coordinates`.
   *
   * Measured accuracy is documented on \ref optimizers_proximal.
   */
  template <class O>
    requires composite_objective<O> && curvature_objective<O> &&
             std::same_as<typename O::curvature_type, dense_curvature_matrix>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    optimizer_report report;
    optimizer_vector x("x", n), g("g", n), z("z", n), u("u", n), x_trial("x_trial", n), x_previous("x_previous", n),
        w("w", n), w_old("w_old", n), v("v", n), h("h", n), g_previous("g_previous", n);
    dense_curvature_matrix H("H", n, n);
    typename curvature_solver_for<dense_curvature_matrix>::type solver(n);
    Kokkos::deep_copy(x, x0);

    // block of each coordinate (-1: unpenalized), its offset in the block, its rho
    std::vector<int> block(static_cast<std::size_t>(n), -1), local(static_cast<std::size_t>(n), 0);
    std::vector<double> rho_of(static_cast<std::size_t>(n), 0.0), a_of(static_cast<std::size_t>(n), 1.0);
    bool separable = false;
    if constexpr (coordinate_composite_objective<O>) {
      separable = true;
      for (int b = 0; b < objective.nonsmooth_block_count(); ++b) {
        const auto [offset, length] = objective.nonsmooth_block_range(b);
        const double rho_b = objective.weak_convexity_constant(b);
        separable = separable && objective.nonsmooth_block_separable(b);
        for (int j = offset; j < offset + length; ++j)
          block[std::size_t(j)] = b, local[std::size_t(j)] = j - offset, rho_of[std::size_t(j)] = rho_b;
      }
    }
    const bool use_cd = inner_solver == proximal_newton_inner_solver::coordinate_descent ||
                        (inner_solver == proximal_newton_inner_solver::automatic && separable);
    const double rho = detail::largest_weak_convexity(objective);

    double s_x = objective.value_gradient_curvature(x, g, H);
    ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
    double r_x = objective.nonsmooth_value(x);
    double f = s_x + r_x, f_previous = f, last_step = 0.0;
    // the largest secant ratio s's / s'y over the accepted steps, for the fallback measure
    double gamma_max = 0.0;
    auto finish = [&](optimizer_status status) {
      report.status = status;
      report.point = x, report.value = f, report.gradient = g;
      report.elapsed_seconds = detail::proximal_seconds_since(start);
      return report;
    };
    if (!std::isfinite(f)) return finish(optimizer_status::start_not_admissible);

    for (int k = 0;; ++k) {
      report.final_factorization = solver.factorize(H);
      const bool positive_definite = report.final_factorization == curvature_factorization::positive_definite;

      // the inner solve, from z = x; u = H(z - x) is kept up to date, so the
      // model's gradient at z is g + u
      Kokkos::deep_copy(z, x);
      for (int i = 0; i < n; ++i) u(i) = 0.0;
      bool inner_met = false;
      double last_inner_measure = std::numeric_limits<double>::infinity();
      // the step 1/L of the model's gradient mapping, L the Gershgorin bound
      // max_i sum_j |H_ij| on the largest eigenvalue of H, and L >= rho / rho_safety
      double L = 0.0;
      for (int i = 0; i < n; ++i) {
        double row = 0.0;
        for (int j = 0; j < n; ++j) row += std::abs(H(i, j));
        L = std::max(L, row);
      }
      if (rho > 0.0) L = std::max(L, rho / rho_safety);
      const double t = L > 0.0 ? 1.0 / L : 1.0;
      // Without a Cholesky factor of H, the factor of H restricted to the free
      // coordinates at z: the unpenalized ones and the penalized ones not held at
      // a kink (Giovanni, 2026-11-18). Coordinate j is held when its scalar prox
      // returns z_j for targets on both sides of z_j - (g_j + u_j) / a_j,
      // perturbed by sqrt(eps) of their size: it lies inside the set that the prox
      // maps to the kink. A lasso with p > n has a singular X'X but a positive
      // definite block on its support. Needs coordinate access and a sweep of
      // coordinate descent (a_j); otherwise every coordinate is free.
      std::vector<int> free;
      std::optional<typename curvature_solver_for<dense_curvature_matrix>::type> free_solver;
      auto factorize_free = [&]() {
        free.clear();
        for (int j = 0; j < n; ++j) {
          bool held = false;
          if constexpr (coordinate_composite_objective<O>) {
            const int b = block[std::size_t(j)];
            if (use_cd && b >= 0) {
              const double a = a_of[std::size_t(j)], target = z(j) - (g(j) + u(j)) / a;
              const double delta = std::sqrt(std::numeric_limits<double>::epsilon()) * (std::abs(target) + std::abs(z(j))) +
                                   std::numeric_limits<double>::min();
              const int jl = local[std::size_t(j)];
              held = objective.nonsmooth_coordinate_prox(b, jl, target + delta, 1.0 / a) == z(j) &&
                     objective.nonsmooth_coordinate_prox(b, jl, target - delta, 1.0 / a) == z(j);
            }
          }
          if (!held) free.push_back(j);
        }
        const int m = static_cast<int>(free.size());
        dense_curvature_matrix H_free("H_free", m, m);
        for (int c = 0; c < m; ++c)
          for (int r = 0; r < m; ++r) H_free(r, c) = H(free[std::size_t(r)], free[std::size_t(c)]);
        free_solver.emplace(m);
        return free_solver->factorize(H_free);
      };
      // The inner measure at z: with G = (z - prox_{t r}(z - t (g + u))) / t the
      // model's gradient mapping, (1/2) G'H^-1 G when H has a Cholesky factor (the
      // suboptimality of z in the model, to first order); without one, the same
      // on the free coordinates with the restricted factor, plus (1/2) t G_j^2 on
      // the held ones (0 when they are held); (1/2) t |G|^2 when neither factor
      // exists. Measured: the cheaper (1/2) t |G|^2, or the last sweep's decrease,
      // is smaller by up to the condition number of H, and stopped the inner solve
      // so early that proximal Newton ran 200 outer iterations without converging
      // (columns equicorrelated at 0.99; and glmnet's d2, p > n, at the smallest lambda).
      auto inner_measure = [&]() {
        for (int i = 0; i < n; ++i) v(i) = z(i) - t * (g(i) + u(i));
        objective.nonsmooth_prox(v, t, h);
        for (int i = 0; i < n; ++i) v(i) = (z(i) - h(i)) / t;
        if (positive_definite && solver.newton_direction(v, h)) return -0.5 * detail::dot(v, h);
        if (!positive_definite && use_cd && factorize_free() == curvature_factorization::positive_definite) {
          const int m = static_cast<int>(free.size());
          optimizer_vector gf("gf", m), hf("hf", m);
          double rest = 0.5 * t * detail::dot(v, v);
          for (int r = 0; r < m; ++r) {
            gf(r) = v(free[std::size_t(r)]);
            rest -= 0.5 * t * gf(r) * gf(r);
          }
          if (free_solver->newton_direction(gf, hf)) return -0.5 * detail::dot(gf, hf) + rest;
        }
        return 0.5 * t * detail::dot(v, v);
      };
      auto inner_done = [&]() {
        last_inner_measure = inner_measure();
        return last_inner_measure <= inner_fraction * std::abs(detail::half_curvature(z, x, u));
      };
      if constexpr (coordinate_composite_objective<O>) {
        if (use_cd) {
          double diag_max = 0.0;
          for (int j = 0; j < n; ++j) diag_max = std::max(diag_max, std::abs(H(j, j)));
          const double floor = std::max(1e-8 * diag_max, std::numeric_limits<double>::min());
          for (int sweep = 0; sweep < max_inner_sweeps && !inner_met; ++sweep) {
            for (int j = 0; j < n; ++j) {
              const int b = block[std::size_t(j)];
              double a = H(j, j);
              if (b >= 0 && a <= rho_of[std::size_t(j)]) {
                a = std::max(rho_of[std::size_t(j)] / rho_safety, floor);
                ++report.damped_coordinates;
              } else if (b < 0 && !(a > 0.0)) {
                a = floor;
                ++report.damped_coordinates;
              }
              a_of[std::size_t(j)] = a;
              const double target = z(j) - (g(j) + u(j)) / a;
              const double zj = b >= 0 ? objective.nonsmooth_coordinate_prox(b, local[std::size_t(j)], target, 1.0 / a)
                                       : target;
              const double delta = zj - z(j);
              if (delta != 0.0) {
                z(j) = zj;
                for (int i = 0; i < n; ++i) u(i) += H(i, j) * delta;
              }
            }
            inner_met = inner_done();
          }
        }
      }
      if (!use_cd) {
        // FISTA on the model q(w) = g'(w - x) + (w - x)'H(w - x)/2 + r(w), fixed
        // step t, gradient restart; w is the momentum point, z the iterate
        Kokkos::deep_copy(w, x);
        double theta = 1.0;
        for (int it = 0; it < max_inner_sweeps && !inner_met; ++it) {
          for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += H(i, j) * (w(j) - x(j));
            v(i) = w(i) - t * (g(i) + s);
          }
          Kokkos::deep_copy(w_old, z);
          objective.nonsmooth_prox(v, t, z);
          double restart = 0.0;
          for (int i = 0; i < n; ++i) restart += (w(i) - z(i)) * (z(i) - w_old(i));
          for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += H(i, j) * (z(j) - x(j));
            u(i) = s;
          }
          inner_met = inner_done();
          const double theta_next = restart > 0.0 ? 1.0 : 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * theta * theta));
          for (int i = 0; i < n; ++i)
            w(i) = restart > 0.0 ? z(i) : z(i) + (theta - 1.0) / theta_next * (z(i) - w_old(i));
          theta = theta_next;
        }
      }

      // the factor that decides whether the decrement is defined
      bool factor_ok = positive_definite;
      if (!positive_definite && report.final_factorization != curvature_factorization::non_finite) {
        report.final_factorization = factorize_free();
        factor_ok = report.final_factorization == curvature_factorization::positive_definite;
      }

      // the decrement and the predicted decrease
      const double half_dHd = detail::half_curvature(z, x, u);
      report.decrement = factor_ok ? half_dHd : nan;
      report.decrement_from_prox_mapping = false;
      // Fallback (Giovanni, 2026-11-18): without a factor of the free block but with
      // a finite H, the measure of proximal_gradient_optimizer at x (decision 12),
      // (1/2) max(t, gamma_max) |G_t|^2, G_t = (x - prox_{t r}(x - t g)) / t with the
      // step t = 1/L above, flagged by decrement_from_prox_mapping. A stationary point whose
      // free coordinates outnumber the rows (nonzeros in the flat tails of SCAD or MCP
      // with p > n) stalled at NaN before.
      if (!factor_ok && report.final_factorization != curvature_factorization::non_finite) {
        for (int i = 0; i < n; ++i) v(i) = x(i) - t * g(i);
        objective.nonsmooth_prox(v, t, h);
        double gg = 0.0;
        for (int i = 0; i < n; ++i) gg += (x(i) - h(i)) * (x(i) - h(i));
        report.decrement = 0.5 * std::max(t, gamma_max) * gg / (t * t);
        report.decrement_from_prox_mapping = true;
      }
      const double r_z = objective.nonsmooth_value(z);
      double delta_pred = r_z - r_x;
      for (int i = 0; i < n; ++i) delta_pred += g(i) * (z(i) - x(i));

      if (k > 0 && trace) trace(optimizer_trace_point{k, x, f, g, report.decrement, last_step});
      stop_rule_input in;
      in.iteration = k, in.value = f, in.point = x, in.decrement = report.decrement;
      in.has_previous = k > 0, in.previous_value = f_previous, in.previous_point = x_previous;
      // An inner solve that missed its target inner_fraction (1/2) d'Hd gives the
      // rule max(decrement, inner measure / inner_fraction): a convergence verdict
      // then still needs the inner measure at most inner_fraction times the outer
      // tolerance (decision 4). Discarding the verdict instead never converged on
      // glmnet's d2 at 0.01 of lambda_max: the decrement fell to 5.7e-17, while the
      // inner measure, read through the factor of an ill-conditioned block (90
      // coordinates, n = 100), kept a rounding floor above 0.01 times it.
      if (!inner_met) in.decrement = std::max(report.decrement, last_inner_measure / inner_fraction);
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict != stop_verdict::none) return finish(detail::proximal_status_from_verdict(verdict));
      if (k >= max_iterations) return finish(optimizer_status::iteration_budget);
      if (report.value_evaluations >= max_value_evaluations) return finish(optimizer_status::evaluation_budget);

      // Armijo on the full objective along d = z - x
      double step = 1.0, f_trial = nan, s_trial = nan, r_trial = nan;
      bool accepted = false, moved = true;
      for (int bt = 0; bt <= max_backtracks; ++bt, step *= 0.5) {
        moved = false;
        for (int i = 0; i < n; ++i) {
          x_trial(i) = x(i) + step * (z(i) - x(i));
          moved = moved || x_trial(i) != x(i);
        }
        if (!moved) break;
        s_trial = objective.value(x_trial);
        ++report.value_evaluations;
        r_trial = step == 1.0 ? r_z : objective.nonsmooth_value(x_trial);
        f_trial = s_trial + r_trial;
        if (std::isfinite(f_trial) && delta_pred < 0.0 && f_trial - f <= sufficient_decrease * step * delta_pred) {
          accepted = true;
          break;
        }
      }
      if (!accepted) return finish(moved ? optimizer_status::line_search_failed : optimizer_status::stalled);

      Kokkos::deep_copy(x_previous, x);
      Kokkos::deep_copy(x, x_trial);
      f_previous = f;
      Kokkos::deep_copy(g_previous, g);
      s_x = objective.value_gradient_curvature(x, g, H);
      ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
      r_x = objective.nonsmooth_value(x);
      f = s_x + r_x;
      {
        double ss = 0.0, sy = 0.0;
        for (int i = 0; i < n; ++i) {
          const double si = x(i) - x_previous(i);
          ss += si * si, sy += si * (g(i) - g_previous(i));
        }
        if (sy > 0.0 && std::isfinite(ss / sy)) gamma_max = std::max(gamma_max, ss / sy);
      }
      report.iterations = k + 1;
      last_step = step;
    }
  }
};

}  // namespace laplace::optimizers
