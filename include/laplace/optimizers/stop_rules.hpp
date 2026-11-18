#pragma once

// Stopping rules (note/optimizers.md, decision 5b): small objects chosen by
// the user, composable with any_of and all_of. A rule reads the quantities of
// the current iteration (stop_rule_input) and returns a stop_verdict.
//
// - Stationarity rules may declare convergence: decrement_stop_rule (the
//   default with curvature), gradient_stop_rule and prox_mapping_stop_rule
//   (the default of the proximal gradient method), and search_size_stop_rule
//   for the derivative-free methods (Brent's bracket, in the units of x).
// - Change rules, objective_change_stop_rule and parameter_change_stop_rule,
//   stop the run as "stopped on change", never as converged.
//
// Each rule declares in `needs` the quantities it reads (stop_needs_*); an
// optimizer declares those it provides, and a rule needing more is rejected
// at compile time by the concept stop_rule_for. There is no tolerance beside
// a rule: the rule is the tolerance.

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <tuple>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `stop_rule_input::gradient`.
 *
 * \see \ref optimizers, stop_rule, stop_rule_for
 */
// The quantities a rule may need, as bits of `needs` and of what an
// optimizer provides. The value and the iteration are always provided.
inline constexpr unsigned stop_needs_gradient = 1u;   // grad f at the point
/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `stop_rule_input::decrement`.
 *
 * \see \ref optimizers, stop_rule, stop_rule_for
 */
inline constexpr unsigned stop_needs_decrement = 2u;  // the stationarity measure in f's units
/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `previous_value` and `previous_point`.
 *
 * \see \ref optimizers, stop_rule, stop_rule_for
 */
inline constexpr unsigned stop_needs_previous = 4u;   // the previous point and value
/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `stop_rule_input::prox_mapping`.
 *
 * \see \ref optimizers, \ref optimizers_proximal, stop_rule, stop_rule_for, prox_mapping_stop_rule
 */
inline constexpr unsigned stop_needs_prox_mapping = 8u;  // (1/2) gamma |G_t|^2 of a composite objective
/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `stop_rule_input::search_size`.
 *
 * \see \ref optimizers, stop_rule, stop_rule_for, search_size_stop_rule
 */
inline constexpr unsigned stop_needs_search_size = 16u;  // the size of a derivative-free search, in x's units
/**
 * \ingroup optimizers
 * \brief Capability bit indicating that a stopping rule reads `stop_rule_input::primal_dual`.
 *
 * \see \ref optimizers, \ref optimizers_primal_dual, stop_rule, stop_rule_for, primal_dual_stop_rule
 */
inline constexpr unsigned stop_needs_primal_dual = 32u;  // e_d + gamma |r_p|^2 of the primal-dual method, in f's units

/**
 * \ingroup optimizers
 * \brief Verdict returned by a stopping rule, ordered from weakest to strongest.
 *
 * Values in increasing strength: `none` (continue), `stopped_on_change` (a change rule held),
 * `converged_at_resolution` (stationarity reached objective floating-point resolution), and
 * `converged` (a stationarity rule held at its tolerance).
 *
 * \see \ref optimizers, stop_rule, any_of(), all_of()
 */
// What a rule decides, ordered from weakest to strongest.
enum class stop_verdict { none, stopped_on_change, converged_at_resolution, converged };

/**
 * \ingroup optimizers
 * \brief Iteration quantities passed to `stop_rule::verdict()`.
 *
 * The fields `previous_value` and `previous_point` are populated only when `has_previous` is `true`.
 * When the calling method does not supply gradients or a predicted decrease, `gradient` is empty and `decrement` is quiet NaN.
 * The predicted decrease `decrement` is \f$\frac{1}{2} g^\top H^{-1} g\f$ with curvature, or \f$\frac{1}{2} g^\top B g\f$ for a method whose next direction is \f$d = -B g\f$ with inverse-curvature model \f$B\f$.
 *
 * \see \ref optimizers, stop_rule, optimizer_report
 */
// The quantities of one iteration. previous_* are meaningful only when
// has_previous; gradient is empty and decrement NaN when the method does not
// provide them. decrement is the decrease predicted by the next step
// (decision 5): (1/2) g' H^-1 g with curvature, (1/2) g' B g = (1/2) |g' d|
// without, d = -B g the next direction and B the method's model of the
// inverse curvature (the BFGS inverse, the L-BFGS two-loop product, alpha I for
// Barzilai-Borwein; s I with s the line-search step when the method has none).
// Never |g' dx| with dx the step just taken: it is near zero after any
// near-exact line search, wherever the point.
struct stop_rule_input {
  /**
   * \ingroup optimizers
   * \brief Current iteration index.
   */
  int iteration = 0;
  /**
   * \ingroup optimizers
   * \brief Current objective value \f$f(x)\f$.
   */
  double value = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Current point \f$x\f$.
   */
  Kokkos::View<const double*, Kokkos::HostSpace> point;
  /**
   * \ingroup optimizers
   * \brief Current gradient \f$\nabla f(x)\f$, or empty when not provided.
   */
  Kokkos::View<const double*, Kokkos::HostSpace> gradient;
  /**
   * \ingroup optimizers
   * \brief Predicted decrease in the objective's units, or quiet NaN when not provided.
   */
  double decrement = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Proximal gradient mapping measure \f$\frac{1}{2} \gamma \lVert G_t\rVert^2\f$ in the objective's units, or quiet NaN when not provided.
   *
   * Here \f$G_t(x) = (x - \operatorname{prox}_{t r}(x - t \nabla s(x))) / t\f$ is the proximal gradient mapping and \f$\gamma\f$ is a scale chosen by the method (`proximal_gradient_optimizer` uses \f$\gamma = \max(t, \gamma_{\max})\f$).
   */
  // (1/2) gamma |G_t(x)|^2 with G_t(x) = (x - prox_{t r}(x - t grad s(x))) / t
  // and gamma the method's scale (FISTA: max(t, gamma_max)), in the objective's
  // units (decision 6); NaN when the method does not provide it.
  double prox_mapping = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Size of the region still searched by a derivative-free method, in the units of \f$x\f$, or quiet NaN when not provided.
   */
  // The half-width of Brent's bracket, the diameter of a simplex, the step of
  // a poll; in the units of x, not of f. NaN when the method does not provide it.
  double search_size = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Primal-dual stationarity measure \f$V = e_d + \max(\gamma_{\max}, \tau) \lVert r_p\rVert^2\f$ in the objective's units, or quiet NaN when not provided.
   *
   * Here \f$r_p = \nabla s(x) + D^\top y\f$, \f$e_d = g(Dx) - g(u') - y^\top(Dx - u')\f$, and \f$\gamma_{\max}\f$ is the largest secant ratio of \f$\nabla s\f$; see \ref optimizers_primal_dual "Primal-dual method".
   */
  // The primal-dual measure e_d + max(gamma_max, tau) |r_p|^2 of s(x) + g(D x),
  // in the objective's units: r_p = grad s(x) + D' y, e_d the dual
  // linearization error g(D x) - g(u') - y'(D x - u') >= 0 (primal_dual.hpp).
  // NaN when the method does not provide it.
  double primal_dual = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Whether `previous_value` and `previous_point` hold data from the preceding iteration.
   */
  bool has_previous = false;
  /**
   * \ingroup optimizers
   * \brief Objective value at the previous iteration (valid when `has_previous` is `true`).
   */
  double previous_value = std::numeric_limits<double>::quiet_NaN();
  /**
   * \ingroup optimizers
   * \brief Point from the previous iteration (valid when `has_previous` is `true`).
   */
  Kokkos::View<const double*, Kokkos::HostSpace> previous_point;
};

/**
 * \ingroup optimizers
 * \brief Concept for a stopping rule.
 *
 * Requires a static `unsigned` bitmask `needs` and `stop_verdict verdict(const stop_rule_input&) const`.
 *
 * \tparam R Stopping rule type.
 * \see \ref optimizers, stop_rule_for, stop_rule_input, stop_verdict
 */
template <class R>
concept stop_rule = requires(const R& rule, const stop_rule_input& input) {
  { R::needs } -> std::convertible_to<unsigned>;
  { rule.verdict(input) } -> std::same_as<stop_verdict>;
};

/**
 * \ingroup optimizers
 * \brief Concept checking that stopping rule `R` requires no quantities outside `provided`.
 *
 * \tparam R Stopping rule type.
 * \tparam provided Bitmask of `stop_needs_*` quantities provided by the caller.
 * \see \ref optimizers, stop_rule
 */
// A rule that an optimizer providing the quantities `provided` can ask.
template <class R, unsigned provided>
concept stop_rule_for = stop_rule<R> && ((R::needs & ~provided) == 0u);

/**
 * \ingroup optimizers
 * \brief Stationarity rule testing the predicted decrease `decrement`.
 *
 * Returns `stop_verdict::converged` when `decrement` is at most `tolerance`,
 * `stop_verdict::converged_at_resolution` when `decrement` is at most \f$m \varepsilon |f|\f$ (where \f$m\f$ is `resolution_multiple` and \f$\varepsilon\f$ is double-precision machine epsilon),
 * and `stop_verdict::none` otherwise (including when `decrement` is NaN).
 *
 * \see \ref optimizers, gradient_stop_rule, stop_verdict
 */
// Converged when the decrement is at most `tolerance` (in the objective's
// units, independent of n); converged at resolution when it is at most
// resolution_multiple eps |f|, the smallest decrease the objective can show.
struct decrement_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_decrement`).
   */
  static constexpr unsigned needs = stop_needs_decrement;
  /**
   * \ingroup optimizers
   * \brief Convergence threshold in the objective's units.
   */
  double tolerance = 1e-6;
  /**
   * \ingroup optimizers
   * \brief Multiplier on \f$\varepsilon |f|\f$ for the resolution check.
   */
  double resolution_multiple = 1.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::converged`, `stop_verdict::converged_at_resolution`, or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    if (in.decrement <= tolerance) return stop_verdict::converged;
    const double resolution = resolution_multiple * std::numeric_limits<double>::epsilon() * std::abs(in.value);
    if (in.decrement <= resolution) return stop_verdict::converged_at_resolution;
    return stop_verdict::none;  // NaN compares false above
  }
};

/**
 * \ingroup optimizers
 * \brief Stationarity rule testing the proximal gradient mapping of a composite objective.
 *
 * Returns `stop_verdict::converged` when `prox_mapping` (\f$\frac{1}{2} \gamma \lVert G_t\rVert^2\f$, in the objective's units) is at most `tolerance`,
 * `stop_verdict::converged_at_resolution` when it is at most \f$m \varepsilon |f|\f$ (where \f$m\f$ is `resolution_multiple` and \f$\varepsilon\f$ is double-precision machine epsilon), and `stop_verdict::none` otherwise (including when `prox_mapping` is NaN).
 *
 * \see \ref optimizers, \ref optimizers_proximal, decrement_stop_rule, stop_rule_input::prox_mapping, stop_verdict
 */
// The decrement rule read on the proximal gradient mapping: G_t is 0 exactly
// at the stationary points of s + r, kinks included, and (1/2) t |G_t|^2 is in
// the objective's units (decision 6).
struct prox_mapping_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_prox_mapping`).
   */
  static constexpr unsigned needs = stop_needs_prox_mapping;
  /**
   * \ingroup optimizers
   * \brief Convergence threshold in the objective's units.
   */
  double tolerance = 1e-6;
  /**
   * \ingroup optimizers
   * \brief Multiplier on \f$\varepsilon |f|\f$ for the resolution check.
   */
  double resolution_multiple = 1.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::converged`, `stop_verdict::converged_at_resolution`, or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    if (in.prox_mapping <= tolerance) return stop_verdict::converged;
    if (in.prox_mapping <= resolution_multiple * std::numeric_limits<double>::epsilon() * std::abs(in.value))
      return stop_verdict::converged_at_resolution;
    return stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Stationarity rule of the derivative-free methods, testing the size of the region still searched.
 *
 * Returns `stop_verdict::converged` when `search_size` is at most \f$\tau_{\mathrm{abs}} + \tau_{\mathrm{rel}} \max_i |x_i|\f$
 * (where \f$\tau_{\mathrm{abs}}\f$ is `absolute_tolerance` and \f$\tau_{\mathrm{rel}}\f$ is `relative_tolerance`),
 * and `stop_verdict::none` otherwise (including NaN).
 *
 * \see \ref optimizers, decrement_stop_rule, stop_verdict
 */
// Converged when the search size (Brent's bracket half-width, a simplex's
// diameter, a poll step) is at most absolute_tolerance + relative_tolerance
// max_i |x_i|. In the units of x: a derivative-free method has no measure in
// the objective's units.
struct search_size_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_search_size`).
   */
  static constexpr unsigned needs = stop_needs_search_size;
  /**
   * \ingroup optimizers
   * \brief Absolute tolerance on the search size.
   */
  double absolute_tolerance = 1e-8;
  /**
   * \ingroup optimizers
   * \brief Relative tolerance multiplying \f$\max_i |x_i|\f$.
   */
  double relative_tolerance = 0.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::converged` or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    double size = 0.0;
    for (std::size_t i = 0; i < in.point.extent(0); ++i) size = std::max(size, std::abs(in.point(i)));
    return in.search_size <= absolute_tolerance + relative_tolerance * size ? stop_verdict::converged
                                                                            : stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Stationarity rule testing the primal-dual measure of a smooth plus linear-map composite objective.
 *
 * Returns `stop_verdict::converged` when `primal_dual` (\f$e_d + \max(\gamma_{\max}, \tau) \lVert r_p\rVert^2\f$, in the objective's units) is at most `tolerance`,
 * `stop_verdict::converged_at_resolution` when it is at most \f$m \varepsilon |f|\f$ (where \f$m\f$ is `resolution_multiple` and \f$\varepsilon\f$ is double-precision machine epsilon), and `stop_verdict::none` otherwise (including when `primal_dual` is NaN).
 *
 * \see \ref optimizers, \ref optimizers_primal_dual, decrement_stop_rule, stop_rule_input::primal_dual, stop_verdict
 */
// The decrement rule read on the primal-dual measure (primal_dual.hpp), an
// upper bound on f(x) - f* for convex s and g up to the distance to x*.
struct primal_dual_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_primal_dual`).
   */
  static constexpr unsigned needs = stop_needs_primal_dual;
  /**
   * \ingroup optimizers
   * \brief Convergence threshold in the objective's units.
   */
  double tolerance = 1e-6;
  /**
   * \ingroup optimizers
   * \brief Multiplier on \f$\varepsilon |f|\f$ for the resolution check.
   */
  double resolution_multiple = 1.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::converged`, `stop_verdict::converged_at_resolution`, or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    if (in.primal_dual <= tolerance) return stop_verdict::converged;
    if (in.primal_dual <= resolution_multiple * std::numeric_limits<double>::epsilon() * std::abs(in.value))
      return stop_verdict::converged_at_resolution;
    return stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Stationarity rule testing the scaled supremum norm of the gradient.
 *
 * Returns `stop_verdict::converged` when \f$\sigma \max_i |g_i| \le \tau_{\mathrm{abs}} + \tau_{\mathrm{rel}} |f|\f$
 * (where \f$\sigma\f$ is `scale`, \f$\tau_{\mathrm{abs}}\f$ is `absolute_tolerance`, and \f$\tau_{\mathrm{rel}}\f$ is `relative_tolerance`),
 * or `stop_verdict::none` if the gradient is empty, contains NaN, or exceeds the threshold.
 *
 * \see \ref optimizers, decrement_stop_rule, stop_verdict
 */
// Converged when scale max_i |g_i| <= absolute_tolerance + relative_tolerance |f|.
// The scale is the caller's: 1 / n reads the score per observation.
struct gradient_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_gradient`).
   */
  static constexpr unsigned needs = stop_needs_gradient;
  /**
   * \ingroup optimizers
   * \brief Absolute gradient tolerance.
   */
  double absolute_tolerance = 1e-6;
  /**
   * \ingroup optimizers
   * \brief Relative tolerance multiplying \f$|f|\f$.
   */
  double relative_tolerance = 0.0;
  /**
   * \ingroup optimizers
   * \brief User-chosen multiplier on \f$\max_i |g_i|\f$ (for example \f$1 / n\f$ for a per-observation score).
   */
  double scale = 1.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::converged` or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    const auto& g = in.gradient;
    if (g.extent(0) == 0) return stop_verdict::none;
    double norm = 0.0;
    for (std::size_t i = 0; i < g.extent(0); ++i) {
      if (std::isnan(g(i))) return stop_verdict::none;
      norm = std::max(norm, std::abs(g(i)));
    }
    return scale * norm <= absolute_tolerance + relative_tolerance * std::abs(in.value) ? stop_verdict::converged
                                                                                         : stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Change rule testing successive objective values.
 *
 * Returns `stop_verdict::stopped_on_change` (never `converged`) when `in.has_previous` is `true` and
 * \f$|f - f_{\mathrm{prev}}| \le \tau_{\mathrm{abs}} + \tau_{\mathrm{rel}} |f_{\mathrm{prev}}|\f$
 * (where \f$\tau_{\mathrm{abs}}\f$ is `absolute_tolerance` and \f$\tau_{\mathrm{rel}}\f$ is `relative_tolerance`).
 *
 * \see \ref optimizers, parameter_change_stop_rule, stop_verdict
 */
// Stopped on change when |f - f_previous| <= absolute_tolerance + relative_tolerance |f_previous|.
struct objective_change_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_previous`).
   */
  static constexpr unsigned needs = stop_needs_previous;
  /**
   * \ingroup optimizers
   * \brief Absolute objective change tolerance.
   */
  double absolute_tolerance = 1e-12;
  /**
   * \ingroup optimizers
   * \brief Relative tolerance multiplying \f$|f_{\mathrm{prev}}|\f$.
   */
  double relative_tolerance = 0.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::stopped_on_change` or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    if (!in.has_previous) return stop_verdict::none;
    return std::abs(in.value - in.previous_value) <= absolute_tolerance + relative_tolerance * std::abs(in.previous_value)
               ? stop_verdict::stopped_on_change
               : stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Change rule testing successive points in supremum norm.
 *
 * Returns `stop_verdict::stopped_on_change` (never `converged`) when `in.has_previous` is `true`, no coordinate difference is NaN, and
 * \f$\max_i |x_i - x_{\mathrm{prev}, i}| \le \tau_{\mathrm{abs}} + \tau_{\mathrm{rel}} \max_i |x_{\mathrm{prev}, i}|\f$
 * (where \f$\tau_{\mathrm{abs}}\f$ is `absolute_tolerance` and \f$\tau_{\mathrm{rel}}\f$ is `relative_tolerance`).
 *
 * \see \ref optimizers, objective_change_stop_rule, stop_verdict
 */
// Stopped on change when max_i |x_i - x_previous,i| <= absolute_tolerance +
// relative_tolerance max_i |x_previous,i|.
struct parameter_change_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Required quantities bitmask (`stop_needs_previous`).
   */
  static constexpr unsigned needs = stop_needs_previous;
  /**
   * \ingroup optimizers
   * \brief Absolute parameter change tolerance.
   */
  double absolute_tolerance = 1e-10;
  /**
   * \ingroup optimizers
   * \brief Relative tolerance multiplying \f$\max_i |x_{\mathrm{prev}, i}|\f$.
   */
  double relative_tolerance = 0.0;
  /**
   * \ingroup optimizers
   * \brief Evaluates the stopping rule on `in`.
   *
   * \param in Current iteration quantities.
   * \return `stop_verdict::stopped_on_change` or `stop_verdict::none`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    if (!in.has_previous) return stop_verdict::none;
    const auto& x = in.point;
    const auto& xp = in.previous_point;
    double change = 0.0, size = 0.0;
    for (std::size_t i = 0; i < x.extent(0); ++i) {
      const double d = std::abs(x(i) - xp(i));
      if (std::isnan(d)) return stop_verdict::none;
      change = std::max(change, d);
      size = std::max(size, std::abs(xp(i)));
    }
    return change <= absolute_tolerance + relative_tolerance * size ? stop_verdict::stopped_on_change
                                                                    : stop_verdict::none;
  }
};

/**
 * \ingroup optimizers
 * \brief Disjunctive combinator returning the strongest `stop_verdict` among its component rules.
 *
 * With zero component rules, always returns `stop_verdict::none`.
 *
 * \tparam R Component stopping rule types satisfying `stop_rule`.
 * \see \ref optimizers, any_of(), all_of_stop_rule
 */
// The strongest verdict of its rules (any one stopping stops the run).
template <stop_rule... R>
struct any_of_stop_rule {
  /**
   * \ingroup optimizers
   * \brief Bitwise union of `R::needs` across all component rules.
   */
  static constexpr unsigned needs = (0u | ... | R::needs);
  /**
   * \ingroup optimizers
   * \brief Tuple of component stopping rules.
   */
  std::tuple<R...> rules;
  /**
   * \ingroup optimizers
   * \brief Evaluates all component rules and returns the strongest verdict.
   *
   * \param in Current iteration quantities.
   * \return Maximum `stop_verdict` across `rules` (or `stop_verdict::none` if empty).
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    return std::apply([&](const auto&... r) { return std::max({stop_verdict::none, r.verdict(in)...}); }, rules);
  }
};

/**
 * \ingroup optimizers
 * \brief Conjunctive combinator returning the weakest `stop_verdict` among its component rules.
 *
 * Requires at least one component rule; combining a convergence rule with a change rule yields `stop_verdict::stopped_on_change` when both hold.
 *
 * \tparam R Component stopping rule types satisfying `stop_rule`.
 * \see \ref optimizers, all_of(), any_of_stop_rule
 */
// The weakest verdict of its rules (the run stops only when all of them stop
// it; a convergence combined with a change rule stops on change).
template <stop_rule... R>
struct all_of_stop_rule {
  static_assert(sizeof...(R) > 0, "laplace::optimizers::all_of needs at least one rule");
  /**
   * \ingroup optimizers
   * \brief Bitwise union of `R::needs` across all component rules.
   */
  static constexpr unsigned needs = (0u | ... | R::needs);
  /**
   * \ingroup optimizers
   * \brief Tuple of component stopping rules.
   */
  std::tuple<R...> rules;
  /**
   * \ingroup optimizers
   * \brief Evaluates all component rules and returns the weakest verdict.
   *
   * \param in Current iteration quantities.
   * \return Minimum `stop_verdict` across `rules`.
   */
  stop_verdict verdict(const stop_rule_input& in) const {
    return std::apply([&](const auto&... r) { return std::min({r.verdict(in)...}); }, rules);
  }
};

/**
 * \ingroup optimizers
 * \brief Constructs an `any_of_stop_rule` from zero or more stopping rules.
 *
 * \tparam R Component stopping rule types.
 * \param rules Stopping rules to combine by strongest verdict.
 * \return An `any_of_stop_rule` holding copies of `rules`.
 *
 * \see \ref optimizers, any_of_stop_rule, all_of()
 */
template <stop_rule... R>
any_of_stop_rule<R...> any_of(const R&... rules) {
  return {std::tuple<R...>(rules...)};
}

/**
 * \ingroup optimizers
 * \brief Constructs an `all_of_stop_rule` from one or more stopping rules.
 *
 * \tparam R Component stopping rule types.
 * \param rules Stopping rules to combine by weakest verdict.
 * \return An `all_of_stop_rule` holding copies of `rules`.
 *
 * \see \ref optimizers, all_of_stop_rule, any_of()
 */
template <stop_rule... R>
all_of_stop_rule<R...> all_of(const R&... rules) {
  return {std::tuple<R...>(rules...)};
}

}  // namespace laplace::optimizers
