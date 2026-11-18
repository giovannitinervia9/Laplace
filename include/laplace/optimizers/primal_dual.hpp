#pragma once

// The primal-dual method of Condat and Vu for min_x s(x) + g(D x)
// (note/optimizers.md, section 12): s smooth with a gradient, g a convex
// penalty whose prox exists in u = D x, D a linear map given by its action and
// that of its transpose. It serves the penalties on a map whose prox in beta
// does not exist (trend filtering, difference_map<K> with K >= 2; the elastic
// net on differences), through linear_map_penalty's inner() and map().
//
// Iteration (tau, sigma > 0, B >= ||D||_2^2, L the Lipschitz constant of
// grad s, tau = 0.99 / (L/2 + sigma B), so that 1/tau - sigma ||D||^2 > L/2):
//   x+ = x - tau (grad s(x) + D' y)
//   w  = y + sigma D (2 x+ - x),  u' = prox_{g/sigma}(w / sigma),  y+ = w - sigma u'
// (the dual step by Moreau's identity, y+ in dg(u')). L is estimated from a
// secant at the start and doubled whenever the descent lemma fails along the
// iterates; it never decreases. sigma is balanced between the two parts of
// the stationarity measure, by factors 1 + a_k with a_k decaying, so the steps
// settle.
//
// Stationarity in the objective's units: with r_p = grad s(x) + D' y (the next
// primal direction) and e_d = g(D x) - g(u') - y'(D x - u') >= 0 (the dual
// linearization error), convexity gives f(x) - f* <= e_d + |r_p| |x - x*|; the
// measure is V = e_d + max(gamma_max, tau) |r_p|^2, gamma_max the largest
// secant ratio s's / s'y of grad s seen. It is passed to the stop rule as
// primal_dual (stop_rule_input), read by primal_dual_stop_rule, the default rule.

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/penalized_objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"
#include "laplace/penalty/linear_map_penalty.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_primal_dual
 * \brief Concept for a composite objective \f$f(x) = s(x) + g(Dx)\f$ split through a linear map \f$D\f$.
 *
 * Refines `gradient_objective`, where `value()` and `value_gradient()` evaluate the smooth term \f$s(x)\f$ alone.
 * The `map_*` members expose the \f$m \times n\f$ linear map \f$D\f$, an upper bound \f$B \ge \lVert D\rVert_2^2\f$,
 * and the value, proximal operator, and weak-convexity constant of \f$g\f$ in \f$u = Dx\f$:
 * - `map_rows()` returns the row count \f$m\f$ of \f$D\f$.
 * - `map_apply(x, u)` writes \f$u = Dx\f$ for the full vector \f$x \in \mathbb{R}^n\f$.
 * - `map_apply_transpose(y, g)` writes all \f$n\f$ entries of \f$D^\top y\f$ into `g`.
 * - `map_norm_squared_bound()` returns \f$B \ge \lVert D\rVert_2^2\f$.
 * - `map_value(u)` returns \f$g(u)\f$, normalizing constants included.
 * - `map_prox(v, t, out)` writes \f$\operatorname{prox}_{t g}(v) = \arg\min_u \{g(u) + \lVert u - v\rVert^2 / (2t)\}\f$ into `out`.
 * - `map_weak_convexity()` returns the weak-convexity constant \f$\rho \ge 0\f$ of \f$g\f$.
 *
 * \tparam O Objective type.
 * \see \ref optimizers_primal_dual, \ref optimizers, gradient_objective, linear_map_penalized_objective, primal_dual_optimizer
 */
// An objective s(x) + g(D x): value() and value_gradient() evaluate s alone;
// map_* expose D (m rows, applied to the whole x, transpose writing all n
// entries), a bound B >= ||D||_2^2, g's value in u (normalizing constants
// included), its prox in u and its weak-convexity constant.
template <class O>
concept linear_map_composite_objective =
    gradient_objective<O> &&
    requires(O& objective, const optimizer_point& a, const optimizer_vector& b, double t) {
      { objective.map_rows() } -> std::convertible_to<int>;
      objective.map_apply(a, b);
      objective.map_apply_transpose(a, b);
      { objective.map_norm_squared_bound() } -> std::convertible_to<double>;
      { objective.map_value(a) } -> std::convertible_to<double>;
      objective.map_prox(a, t, b);
      { objective.map_weak_convexity() } -> std::convertible_to<double>;
    };

/**
 * \ingroup optimizers_primal_dual
 * \brief Composite objective \f$f(x) = s(x) + \mathrm{pen}(x_{[o, o+p)}; \theta)\f$ built from a smooth objective and one `linear_map_penalty`.
 *
 * Satisfies `linear_map_composite_objective`. Here `value()` and `value_gradient()` forward to `smooth` alone,
 * \f$g\f$ is the penalty's `inner()` \f$P\f$ (whose `prox_beta()` gives the proximal operator in \f$u\f$) plus the map constant \f$c(D)\f$
 * (see \ref penalty_linear_map "Linear maps"), and \f$D\f$ is the penalty's `map()` on the coordinate range \f$[o, o + p)\f$.
 * A penalty whose inner penalty \f$P\f$ lacks `prox_beta()` does not compile.
 *
 * \tparam Smooth Smooth objective type satisfying `gradient_objective`, held by reference.
 * \tparam P Separable inner penalty type providing `prox_beta()`.
 * \tparam Map Linear map type (`diagonal_map` or `difference_map`).
 * \see \ref optimizers_primal_dual, \ref penalty_linear_map, linear_map_composite_objective, attach_penalty(), primal_dual_optimizer
 */
// s(x) + pen(x_[o, o + p)) with pen a linear_map_penalty<P, Map> attached at
// offset o with fixed theta (attach_penalty). g is P, the penalty's inner(),
// whose prox_beta is the prox in u; D is the penalty's map() on the range.
template <gradient_objective Smooth, class P, class Map>
  requires detail::penalty_with_prox<P>
class linear_map_penalized_objective {
 public:
  /**
   * \ingroup optimizers_primal_dual
   * \brief Type of the attached `laplace::penalty::linear_map_penalty`.
   */
  using penalty_type = laplace::penalty::linear_map_penalty<P, Map>;

  /**
   * \ingroup optimizers_primal_dual
   * \brief Builds the split composite objective over dimension \f$n\f$.
   *
   * Throws `std::invalid_argument` if the attached coordinate range \f$[o, o + p)\f$ leaves \f$[0, n)\f$
   * or if `attached.theta.size()` differs from `attached.penalty.theta_size()`.
   *
   * \param smooth Smooth objective \f$s\f$ (held by reference, not copied).
   * \param n Dimension \f$n\f$ of \f$x\f$.
   * \param attached Linear-map penalty attached to \f$[o, o + p)\f$ with fixed hyperparameters \f$\theta\f$.
   */
  linear_map_penalized_objective(Smooth& smooth, int n, attached_penalty<penalty_type> attached)
      : smooth_(smooth), n_(n), attached_(std::move(attached)) {
    const penalty_type& pen = attached_.penalty;
    if (attached_.offset < 0 || attached_.offset + pen.dim() > n)
      throw std::invalid_argument("laplace::optimizers::linear_map_penalized_objective: the penalty's range leaves [0, n)");
    if (static_cast<int>(attached_.theta.size()) != pen.theta_size())
      throw std::invalid_argument("laplace::optimizers::linear_map_penalized_objective: theta has " +
                                  std::to_string(attached_.theta.size()) + " entries, the penalty takes " +
                                  std::to_string(pen.theta_size()));
    work_.assign(static_cast<std::size_t>(pen.state_size() + pen.scratch_size()), 0.0);
  }

  /**
   * \ingroup optimizers_primal_dual
   * \brief Evaluates the smooth part \f$s(x)\f$.
   * \param x Point of length \f$n\f$.
   * \return \f$s(x)\f$.
   */
  double value(const optimizer_point& x) { return smooth_.value(x); }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Evaluates \f$s(x)\f$ and writes \f$\nabla s(x)\f$ to `g`.
   * \param x Point of length \f$n\f$.
   * \param g Output gradient vector of length \f$n\f$.
   * \return \f$s(x)\f$.
   */
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) { return smooth_.value_gradient(x, g); }

  /**
   * \ingroup optimizers_primal_dual
   * \brief Returns the number of rows \f$m\f$ of the linear map \f$D\f$.
   * \return `map().m()`.
   */
  int map_rows() const { return attached_.penalty.map().m(); }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Applies the linear map to the attached slice, writing \f$u = D x_{[o, o+p)}\f$.
   * \param x Input point of length \f$n\f$.
   * \param u Output mapped vector of length \f$m\f$.
   */
  void map_apply(const optimizer_point& x, const optimizer_vector& u) const {
    attached_.penalty.map().apply(x.data() + attached_.offset, u.data());
  }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Applies the transpose map across all \f$n\f$ coordinates, writing \f$D^\top y\f$ on \f$[o, o+p)\f$ and \f$0\f$ elsewhere.
   * \param y Dual vector of length \f$m\f$.
   * \param g Output vector of length \f$n\f$.
   */
  void map_apply_transpose(const optimizer_point& y, const optimizer_vector& g) const {
    for (int i = 0; i < n_; ++i) g(i) = 0.0;
    attached_.penalty.map().apply_transpose(y.data(), g.data() + attached_.offset);
  }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Returns the upper bound \f$B \ge \lVert D\rVert_2^2\f$ from `map().norm_squared_bound()`.
   * \return Bound \f$B\f$.
   */
  double map_norm_squared_bound() const { return attached_.penalty.map().norm_squared_bound(); }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Evaluates \f$g(u) = P(u; \theta) + c(D)\f$, with normalizing constants included when \f$P\f$ is normalizable.
   * \param u Mapped point of length \f$m\f$.
   * \return \f$g(u)\f$.
   */
  // P(u) plus the constant c(D) of the penalty on the map
  double map_value(const optimizer_point& u) {
    const auto st = state();
    const double c = penalty_type::normalization == laplace::penalty::penalty_normalization::unnormalizable
                         ? 0.0
                         : attached_.penalty.map().log_normalizer();
    return attached_.penalty.inner().value(st.inner, u.data(), scratch()) + c;
  }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Computes \f$\operatorname{prox}_{t g}(v)\f$ in \f$u\f$ using `inner().prox_beta()`.
   * \param v Input vector of length \f$m\f$.
   * \param t Proximal step \f$t > 0\f$.
   * \param out Output minimizer of length \f$m\f$.
   */
  void map_prox(const optimizer_point& v, double t, const optimizer_vector& out) {
    const auto st = state();
    attached_.penalty.inner().prox_beta(st.inner, v.data(), t, out.data(), scratch());
  }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Returns the weak-convexity constant of the inner penalty \f$P\f$ at \f$\theta\f$.
   * \return `inner().weak_convexity(theta)`.
   */
  double map_weak_convexity() const { return attached_.penalty.inner().weak_convexity(attached_.theta.data()); }
  /**
   * \ingroup optimizers_primal_dual
   * \brief Returns the dimension \f$n\f$ of \f$x\f$.
   * \return \f$n\f$.
   */
  int dimension() const { return n_; }

 private:
  typename penalty_type::state_type state() { return attached_.penalty.theta_state(attached_.theta.data(), work_.data()); }
  double* scratch() { return work_.data() + attached_.penalty.state_size(); }

  Smooth& smooth_;
  int n_;
  attached_penalty<penalty_type> attached_;
  std::vector<double> work_;
};

namespace detail {
inline constexpr unsigned primal_dual_provides = stop_needs_primal_dual | stop_needs_previous;
}  // namespace detail

/**
 * \ingroup optimizers_primal_dual
 * \brief Primal-dual minimization of \f$s(x) + g(Dx)\f$ for smooth \f$s\f$ and convex \f$g\f$ through a linear map \f$D\f$.
 *
 * Minimizes an objective modeling `linear_map_composite_objective` on the host, invoked through
 * minimize() (`minimize(optimizer, objective, x0)`) or `run()`. The template parameter `StopRule`
 * defaults to `primal_dual_stop_rule`, and `stop_rule_for` admits any stopping rule requiring at most
 * `stop_needs_primal_dual` and `stop_needs_previous`; rules reading `gradient` or `prox_mapping`
 * fail to compile. Calling `run()` throws `std::invalid_argument` when `objective.map_weak_convexity()`
 * is positive because the method requires \f$g\f$ convex.
 *
 * The iteration, the rules for the steps \f$\tau\f$ and \f$\sigma\f$, the estimate of \f$L\f$ and the
 * stationarity measure \f$V\f$, passed to `stop_rule` as `primal_dual` and returned as `decrement`, are
 * described on the page \ref optimizers_primal_dual "Primal-dual method".
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `stop_rule` | `primal_dual_stop_rule{}` | Stopping rule queried on `primal_dual` and `previous_*` at each iteration |
 * | `max_iterations` | \f$10^5\f$ | Maximum number of iterations |
 * | `max_value_evaluations` | \f$10^7\f$ | Budget on smooth `value_gradient()` calls |
 * | `initial_lipschitz` | \f$0\f$ | Starting Lipschitz constant \f$L\f$; \f$0\f$ estimates \f$L\f$ from one secant |
 * | `initial_dual_ratio` | \f$50\f$ | Multiplier setting \f$\sigma_0\f$ to `initial_dual_ratio` \f$\cdot\, L / B\f$ |
 * | `balance_threshold` | \f$1.5\f$ | Ratio threshold between primal and dual parts of \f$V\f$ triggering rebalancing |
 * | `balance_rate` | \f$0.5\f$ | Initial adjustment factor \f$a_0\f$ for \f$\sigma\f$ |
 * | `balance_decay` | \f$0.95\f$ | Factor multiplying \f$a_k\f$ after each rebalancing of \f$\sigma\f$ |
 * | `rounding_slack` | \f$8\f$ | Multiple of \f$\varepsilon \max(1, |s(x)|)\f$ allowed in the descent lemma check |
 * | `max_backtracks` | \f$100\f$ | Maximum doublings of \f$L\f$ within one iteration |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked once per iteration |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(x_0) = s(x_0) + g(Dx_0)\f$ is not finite.
 * - `optimizer_status::converged`, `optimizer_status::converged_at_resolution`, or `optimizer_status::stopped_on_change` when returned by `stop_rule`.
 * - `optimizer_status::stalled` when neither \f$x\f$ nor \f$y\f$ moves in an iteration.
 * - `optimizer_status::line_search_failed` after more than `max_backtracks` doublings of \f$L\f$ in one step.
 * - `optimizer_status::iteration_budget` when `max_iterations` iterations are reached.
 * - `optimizer_status::evaluation_budget` when `max_value_evaluations` runs out.
 *
 * In `optimizer_report`, `point` holds \f$x\f$ (not \f$u'\f$, so exact zeros in \f$Dx\f$ are not attained),
 * `value` holds \f$s(x) + g(Dx)\f$ with normalizing constants included, `gradient` holds \f$\nabla s(x)\f$ alone,
 * `decrement` holds \f$V\f$, `iterations` counts iterations, `value_evaluations` and `gradient_evaluations`
 * each count `value_gradient()` calls (including backtracking trials), `curvature_evaluations` is \f$0\f$, and
 * `search_size` is quiet NaN.
 *
 * \tparam StopRule Stopping rule requiring at most `stop_needs_primal_dual` and `stop_needs_previous`;
 *   `primal_dual_stop_rule` by default.
 * \see \ref optimizers_primal_dual, \ref optimizers, minimize(), primal_dual_stop_rule
 */
template <class StopRule = primal_dual_stop_rule>
  requires stop_rule_for<StopRule, detail::primal_dual_provides>
struct primal_dual_optimizer {
  /**
   * \ingroup optimizers_primal_dual
   * \brief Stopping rule instance queried at each iteration (default `primal_dual_stop_rule{}`).
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_primal_dual
   * \brief Maximum number of primal-dual iterations (default \f$10^5\f$).
   */
  int max_iterations = 100000;
  /**
   * \ingroup optimizers_primal_dual
   * \brief Budget on smooth `value_gradient()` evaluations (default \f$10^7\f$).
   */
  long max_value_evaluations = 10000000;
  /**
   * \ingroup optimizers_primal_dual
   * \brief Initial Lipschitz bound \f$L\f$ for \f$\nabla s\f$; \f$0\f$ estimates \f$L\f$ from a gradient secant (default \f$0\f$).
   */
  double initial_lipschitz = 0.0;     // L at the start; 0 estimates it from a secant
  /**
   * \ingroup optimizers_primal_dual
   * \brief Multiplier setting the initial dual step \f$\sigma_0\f$ to `initial_dual_ratio` \f$\cdot\, L / B\f$ (default \f$50\f$).
   */
  double initial_dual_ratio = 50.0;   // sigma_0 = initial_dual_ratio L / B
  /**
   * \ingroup optimizers_primal_dual
   * \brief Ratio threshold between the primal and dual parts of \f$V\f$ that triggers rebalancing of \f$\sigma\f$ (default \f$1.5\f$).
   */
  double balance_threshold = 1.5;     // rebalance when one part of V exceeds this times the other
  /**
   * \ingroup optimizers_primal_dual
   * \brief Initial adjustment rate \f$a_0\f$ scaling \f$\sigma\f$ by \f$1 + a_k\f$ (default \f$0.5\f$).
   */
  double balance_rate = 0.5;          // a_0
  /**
   * \ingroup optimizers_primal_dual
   * \brief Decay factor multiplying \f$a_k\f$ after each rebalancing of \f$\sigma\f$ (default \f$0.95\f$).
   */
  double balance_decay = 0.95;        // a_{k+1} = balance_decay a_k after each rebalance
  /**
   * \ingroup optimizers_primal_dual
   * \brief Multiple of \f$\varepsilon \max(1, |s(x)|)\f$ allowed when checking the descent lemma (default \f$8\f$).
   */
  double rounding_slack = 8.0;        // descent lemma checked up to rounding_slack eps |s|
  /**
   * \ingroup optimizers_primal_dual
   * \brief Maximum number of doublings of \f$L\f$ allowed per iteration (default \f$100\f$).
   */
  int max_backtracks = 100;
  /**
   * \ingroup optimizers_primal_dual
   * \brief Optional trace callback invoked once per iteration (default empty).
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_primal_dual
   * \brief Minimizes `objective` starting from `x0` by the Condat–Vũ primal-dual method.
   *
   * Throws `std::invalid_argument` when `objective.map_weak_convexity()` is positive.
   *
   * \tparam O Objective type satisfying `linear_map_composite_objective`.
   * \param objective Split composite objective \f$s(x) + g(Dx)\f$ to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_primal_dual, primal_dual_optimizer, minimize()
   */
  template <linear_map_composite_objective O>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    const int m = objective.map_rows();
    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    if (objective.map_weak_convexity() > 0.0)
      throw std::invalid_argument("laplace::optimizers::primal_dual_optimizer: the penalty in D x is not convex (rho > 0)");

    optimizer_report report;
    optimizer_vector x("x", n), x_new("x_new", n), x_previous("x_previous", n), g("g", n), g_new("g_new", n),
        r("r", n), dty("dty", n), y("y", m), y_new("y_new", m), w("w", m), u("u", m), u_prox("u_prox", m);
    Kokkos::deep_copy(x, x0);

    double s_x = objective.value_gradient(x, g);
    ++report.value_evaluations, ++report.gradient_evaluations;
    objective.map_apply(x, u);
    double f_x = s_x + objective.map_value(u);
    auto finish = [&](optimizer_status status, double measure) {
      report.status = status;
      report.point = x, report.value = f_x, report.gradient = g, report.decrement = measure;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    if (!std::isfinite(f_x)) return finish(optimizer_status::start_not_admissible, nan);

    // L from a secant of grad s along a short gradient step, as FISTA's t
    double L = initial_lipschitz;
    double gamma_max = 0.0;
    if (!(L > 0.0)) {
      L = 1.0;
      const double gmax = detail::max_abs(g);
      if (gmax > 0.0 && std::isfinite(gmax)) {
        const double h = 1e-4 * std::max(1.0, detail::max_abs(x)) / gmax;
        for (int i = 0; i < n; ++i) x_new(i) = x(i) - h * g(i);
        const double sv = objective.value_gradient(x_new, g_new);
        ++report.value_evaluations, ++report.gradient_evaluations;
        double dg = 0.0, dx = 0.0, sy = 0.0;
        for (int i = 0; i < n; ++i) {
          const double a = x_new(i) - x(i), b = g_new(i) - g(i);
          dg += b * b, dx += a * a, sy += a * b;
        }
        if (std::isfinite(sv) && dg > 0.0 && dx > 0.0) L = std::sqrt(dg / dx);
        if (std::isfinite(sv) && sy > 0.0) gamma_max = dx / sy;
      }
    }
    const double B = objective.map_norm_squared_bound();
    double sigma = initial_dual_ratio * L / B;
    auto primal_step = [&]() { return 0.99 / (0.5 * L + sigma * B); };
    double tau = primal_step();
    double a_k = balance_rate, f_previous = f_x, last_measure = nan;
    Kokkos::deep_copy(r, g);  // y = 0

    for (int k = 0;;) {
      // primal step, L doubled until the descent lemma holds along it
      double s_new = 0.0;
      for (int backtracks = 0;; ++backtracks) {
        if (backtracks > max_backtracks) return finish(optimizer_status::line_search_failed, last_measure);
        for (int i = 0; i < n; ++i) x_new(i) = x(i) - tau * r(i);
        s_new = objective.value_gradient(x_new, g_new);
        ++report.value_evaluations, ++report.gradient_evaluations;
        double slope = 0.0, step2 = 0.0;
        for (int i = 0; i < n; ++i) {
          const double d = x_new(i) - x(i);
          slope += g(i) * d, step2 += d * d;
        }
        if (std::isfinite(s_new) &&
            s_new - s_x - slope - 0.5 * L * step2 <= rounding_slack * eps * std::max(1.0, std::abs(s_x)))
          break;
        L *= 2.0;
        tau = primal_step();
      }
      double ss = 0.0, sy = 0.0;
      for (int i = 0; i < n; ++i) {
        const double a = x_new(i) - x(i);
        ss += a * a, sy += a * (g_new(i) - g(i));
      }
      if (sy > 0.0 && std::isfinite(ss / sy)) gamma_max = std::max(gamma_max, ss / sy);

      // dual step by Moreau's identity
      for (int i = 0; i < n; ++i) dty(i) = 2.0 * x_new(i) - x(i);
      objective.map_apply(dty, w);
      for (int i = 0; i < m; ++i) {
        w(i) = y(i) + sigma * w(i);
        u_prox(i) = w(i) / sigma;
      }
      objective.map_prox(u_prox, 1.0 / sigma, u);  // u holds u' for a moment
      Kokkos::deep_copy(u_prox, u);
      for (int i = 0; i < m; ++i) y_new(i) = w(i) - sigma * u_prox(i);

      // the measure at (x_new, y_new)
      objective.map_apply(x_new, u);
      const double g_u = objective.map_value(u), g_prox = objective.map_value(u_prox);
      double cross = 0.0;
      for (int i = 0; i < m; ++i) cross += y_new(i) * (u(i) - u_prox(i));
      const double e_d = std::max(0.0, g_u - g_prox - cross);
      objective.map_apply_transpose(y_new, dty);
      double rr = 0.0;
      for (int i = 0; i < n; ++i) {
        r(i) = g_new(i) + dty(i);
        rr += r(i) * r(i);
      }
      const double primal_part = std::max(gamma_max, tau) * rr;
      const double measure = e_d + primal_part;

      bool moved = false;
      for (int i = 0; i < n && !moved; ++i) moved = x_new(i) != x(i);
      for (int i = 0; i < m && !moved; ++i) moved = y_new(i) != y(i);
      Kokkos::deep_copy(x_previous, x);
      f_previous = f_x;
      Kokkos::deep_copy(x, x_new);
      Kokkos::deep_copy(g, g_new);
      Kokkos::deep_copy(y, y_new);
      s_x = s_new;
      f_x = s_new + g_u;
      last_measure = measure;
      report.iterations = ++k;
      if (trace) trace(optimizer_trace_point{k, x, f_x, {}, measure, tau});

      stop_rule_input in;
      in.iteration = k, in.value = f_x, in.point = x, in.primal_dual = measure;
      in.has_previous = true, in.previous_value = f_previous, in.previous_point = x_previous;
      const stop_verdict verdict = stop_rule.verdict(in);
      if (verdict == stop_verdict::converged) return finish(optimizer_status::converged, measure);
      if (verdict == stop_verdict::converged_at_resolution)
        return finish(optimizer_status::converged_at_resolution, measure);
      if (verdict == stop_verdict::stopped_on_change) return finish(optimizer_status::stopped_on_change, measure);
      if (!moved) return finish(optimizer_status::stalled, measure);
      if (k >= max_iterations) return finish(optimizer_status::iteration_budget, measure);
      if (report.value_evaluations >= max_value_evaluations)
        return finish(optimizer_status::evaluation_budget, measure);

      // balance sigma between the two parts of the measure
      if (primal_part > balance_threshold * e_d) {
        sigma /= 1.0 + a_k;
        a_k *= balance_decay;
      } else if (e_d > balance_threshold * primal_part) {
        sigma *= 1.0 + a_k;
        a_k *= balance_decay;
      }
      tau = primal_step();
    }
  }
};

}  // namespace laplace::optimizers
