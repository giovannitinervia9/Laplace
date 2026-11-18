#pragma once

// Block alternation (note/optimizers.md, decisions 6 and 12.6): x is split
// into contiguous blocks, each minimized in turn by its own sub-optimizer on
// a view of the objective in which the other coordinates are held (cyclic
// Gauss-Seidel, warm starts). The view evaluates the full objective and takes
// the slice of its gradient and the diagonal block of its curvature, so a
// block-aware objective is not needed.
//
// Inexact block solves: from the second sweep, a sub-optimizer whose stop rule
// has a tolerance in the objective's units (a rule reading the decrement or
// the proximal gradient mapping) runs
// at max(final, inexact_fraction * D), with "final" its configured tolerance
// and D the decrease of f over the previous sweep, which includes the coupling
// between the blocks. Other rules (Brent's bracket, combinations) run as
// configured.
//
// Convergence is declared on a full sweep in which every block ran at its
// final tolerance and declared its own stationarity at its start without
// moving beyond resolution: its sub-run succeeded and either left the block
// where it was, or moved it within its own search size (a Brent block whose
// start lies in its final bracket), or moved it along a set where f does not
// fall beyond value_resolution_multiple eps |f| (a flat set of minimizers,
// the median interval of the Laplace at even n). Never on the sub-runs'
// flags alone, and never on a flag per sweep (statmodels7's alternation flag
// gave false verdicts). A sweep at the final tolerances that moves no
// coordinate without converging ends the run as stalled; otherwise at most
// max_sweeps sweeps.
//
// Known limit: a kink that couples coordinates of different blocks can hold
// every block stationary at a point that is not stationary jointly. A block
// whose sub-optimizer assumes smoothness sees the kink (its line search
// fails) and the run ends stalled; a block solved by a method for kinks
// (Brent) does not, and the run is declared converged there.
//
// A composite objective (s + r, as penalized_objective) must give the range of
// each non-smooth block. f is then s + r; a block that holds a penalty is
// solved by a proximal sub-optimizer on a composite view, a penalty may not
// straddle two blocks, and a sub-optimizer that ignores the penalty of its
// block is refused (std::invalid_argument).
//
// The report: iterations are sweeps, the counts are the sub-runs' summed,
// decrement is the sum of the blocks' decrements of the last sweep where the
// sub-optimizers give one (NaN when none does), search_size the largest of the
// blocks' search sizes (NaN when none gives one).

#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace laplace::optimizers {

namespace detail {

// A composite objective that also gives the range of each non-smooth block and
// the per-coordinate members of proximal Newton's coordinate descent (as
// penalized_objective does); a named concept, not an inline requires-expression
// in a member's constraint (CLAUDE.md, nvcc trap a).
template <class O>
concept ranged_composite_objective =
    composite_objective<O> && requires(O& objective, int b, int j, double v, double t) {
      { objective.nonsmooth_block_range(b) } -> std::convertible_to<std::pair<int, int>>;
      { objective.nonsmooth_block_separable(b) } -> std::convertible_to<bool>;
      { objective.nonsmooth_coordinate_prox(b, j, v, t) } -> std::convertible_to<double>;
    };

}  // namespace detail

/**
 * \ingroup optimizers_block_coordinate
 * \brief View of an objective over one contiguous block of coordinates with the rest held fixed.
 *
 * Holds a working copy of the full coordinate vector initialized from `held`,
 * writes the block coordinates `[begin, begin + size)`
 * into that copy, evaluates the wrapped `objective`, and slices the gradient
 * and diagonal curvature block when provided. A user does not normally build
 * a `block_objective_view` directly; `block_coordinate_optimizer` creates one
 * for each block solve.
 *
 * Its member `curvature_type` is `dense_curvature_matrix`. The member
 * `value_gradient()` exists only when `O` satisfies `gradient_objective`, and
 * `value_gradient_curvature()` exists only when `O` satisfies
 * `curvature_objective` with `O::curvature_type` equal to
 * `dense_curvature_matrix`.
 *
 * When `O` is a `composite_objective` that also gives each non-smooth block's
 * range and per-coordinate proximal members (such as `penalized_objective`),
 * the view is composite too: `value()`, `value_gradient()`, and
 * `value_gradient_curvature()` evaluate the smooth part \f$s\f$ and its
 * derivatives, while `nonsmooth_value()` evaluates \f$r\f$ at the full point
 * (penalties outside the block are constant there). Its non-smooth blocks are
 * the penalties whose ranges lie inside `[begin, begin + size)`, with the
 * per-coordinate members used by `proximal_newton_optimizer` forwarded and
 * ranges shifted by `-begin`; a penalty straddling a boundary of the block
 * throws `std::invalid_argument` at construction. All non-smooth members
 * except `nonsmooth_calls()` and `local_nonsmooth_blocks()` exist only for
 * such composite objectives.
 *
 * \tparam O Full objective type.
 * \see \ref optimizers_block_coordinate, block_coordinate_optimizer, \ref optimizers
 */
// The objective in the coordinates [begin, begin + size) of x, the others held
// at their values in `held`. Each member evaluates the full objective.
//
// Over a composite objective with block ranges (ranged_composite_objective),
// the view is composite too: value and its derivatives are those of s, and
// nonsmooth_value is r of the full point (the penalties outside the block are
// constants there), so that a proximal sub-optimizer's value s + r is f of the
// full point. Its non-smooth blocks are the penalties whose range lies inside
// the view's block; a penalty that straddles the block's boundary is refused
// at construction (a kink coupling two blocks). nonsmooth_calls() counts the
// calls of nonsmooth_value and nonsmooth_prox, so that the alternation can
// tell a sub-optimizer that ignored a penalty in its block.
template <class O>
class block_objective_view {
 public:
  using curvature_type = dense_curvature_matrix;

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Constructs a view of `objective` on coordinates `[begin, begin + size)`, holding the others at `held`.
   *
   * \param objective Full objective to evaluate.
   * \param held Current full point \f$x\f$, copied into internal working storage.
   * \param begin Starting index of the active block in `x`.
   * \param size Number of coordinates in the active block.
   */
  block_objective_view(O& objective, const optimizer_point& held, int begin, int size)
      : objective_(objective),
        begin_(begin),
        size_(size),
        x_("laplace::optimizers::block_objective_view::x", held.extent(0)) {
    Kokkos::deep_copy(x_, held);
    if constexpr (detail::ranged_composite_objective<O>) {
      for (int b = 0; b < objective_.nonsmooth_block_count(); ++b) {
        const auto [offset, length] = std::pair<int, int>(objective_.nonsmooth_block_range(b));
        const bool inside = offset >= begin_ && offset + length <= begin_ + size_;
        const bool outside = offset + length <= begin_ || offset >= begin_ + size_;
        if (!inside && !outside)
          throw std::invalid_argument(
              "block_objective_view: a non-smooth block straddles the boundary of an alternation block");
        if (inside) local_blocks_.push_back(b);
      }
    }
  }

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Evaluates the non-smooth part \f$r(x)\f$ at the full point with the active block set to `xb`.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param xb Block point of length `size`.
   * \return Non-smooth value \f$r(x)\f$ of the full point.
   */
  // The non-smooth part over the block (composite objectives with block ranges only).
  double nonsmooth_value(const optimizer_point& xb)
    requires detail::ranged_composite_objective<O>
  {
    ++nonsmooth_calls_;
    place(xb);
    return objective_.nonsmooth_value(optimizer_point(x_));
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Applies the non-smooth proximal operator on the active block from `vb` with step `t` into `zb`.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param vb Block input point of length `size`.
   * \param t Proximal step \f$t > 0\f$.
   * \param zb Writable vector receiving the block slice of length `size`.
   */
  void nonsmooth_prox(const optimizer_point& vb, double t, const optimizer_vector& zb)
    requires detail::ranged_composite_objective<O>
  {
    ++nonsmooth_calls_;
    if (v_.extent(0) == 0) {
      v_ = optimizer_vector("laplace::optimizers::block_objective_view::v", n());
      z_ = optimizer_vector("laplace::optimizers::block_objective_view::z", n());
    }
    Kokkos::deep_copy(v_, x_);
    for (int i = 0; i < size_; ++i) v_(begin_ + i) = vb(i);
    objective_.nonsmooth_prox(optimizer_point(v_), t, z_);
    for (int i = 0; i < size_; ++i) zb(i) = z_(begin_ + i);
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns the number of non-smooth blocks lying inside this view's block.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \return Number of enclosed non-smooth blocks.
   */
  int nonsmooth_block_count() const
    requires detail::ranged_composite_objective<O>
  {
    return static_cast<int>(local_blocks_.size());
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns the weak-convexity constant \f$\rho_b\f$ of enclosed non-smooth block `b`.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param b Index of the enclosed non-smooth block.
   * \return Weak-convexity constant \f$\rho_b\f$.
   */
  double weak_convexity_constant(int b)
    requires detail::ranged_composite_objective<O>
  {
    return objective_.weak_convexity_constant(local_blocks_[static_cast<std::size_t>(b)]);
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns the `(offset, length)` range of enclosed non-smooth block `b` relative to this view's block.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param b Index of the enclosed non-smooth block.
   * \return Pair `(offset - begin, length)` within `[0, size)`.
   */
  std::pair<int, int> nonsmooth_block_range(int b)
    requires detail::ranged_composite_objective<O>
  {
    const std::pair<int, int> r = objective_.nonsmooth_block_range(local_blocks_[static_cast<std::size_t>(b)]);
    return {r.first - begin_, r.second};
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns whether enclosed non-smooth block `b` is separable by coordinate with coordinate-aligned kinks.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param b Index of the enclosed non-smooth block.
   * \return `true` when the block is separable by coordinate, `false` otherwise.
   */
  bool nonsmooth_block_separable(int b)
    requires detail::ranged_composite_objective<O>
  {
    return objective_.nonsmooth_block_separable(local_blocks_[static_cast<std::size_t>(b)]);
  }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Evaluates the scalar proximal step for coordinate `j` of enclosed non-smooth block `b`.
   *
   * Exists only for a composite objective that gives its non-smooth block ranges.
   *
   * \param b Index of the enclosed non-smooth block.
   * \param j Coordinate index within block `b`.
   * \param v Scalar input value.
   * \param t Proximal step \f$t > 0\f$.
   * \return Scalar proximal result for coordinate `j`.
   */
  double nonsmooth_coordinate_prox(int b, int j, double v, double t)
    requires detail::ranged_composite_objective<O>
  {
    ++nonsmooth_calls_;
    return objective_.nonsmooth_coordinate_prox(local_blocks_[static_cast<std::size_t>(b)], j, v, t);
  }

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns the number of calls made to non-smooth members of this view so far.
   *
   * Always exists, whether or not `O` is composite.
   *
   * \return Count of calls to `nonsmooth_value()`, `nonsmooth_prox()`, and `nonsmooth_coordinate_prox()`.
   */
  // Calls of the non-smooth members so far, and the number of non-smooth blocks inside the view.
  long nonsmooth_calls() const { return nonsmooth_calls_; }
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Returns the number of non-smooth blocks of the objective lying inside this view's block.
   *
   * Always exists, returning \f$0\f$ when `O` is not composite.
   *
   * \return Number of enclosed non-smooth blocks.
   */
  int local_nonsmooth_blocks() const { return static_cast<int>(local_blocks_.size()); }

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Evaluates the full objective with the active block set to `xb`.
   *
   * \param xb Block point of length `size`.
   * \return Objective value \f$f(x)\f$.
   */
  double value(const optimizer_point& xb) {
    place(xb);
    return objective_.value(optimizer_point(x_));
  }

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Evaluates the full objective and gradient with the active block set to `xb`, writing the block slice into `gb`.
   *
   * Participates in overload resolution only when `O` satisfies `gradient_objective`.
   *
   * \param xb Block point of length `size`.
   * \param gb Writable vector receiving the block gradient slice of length `size`.
   * \return Objective value \f$f(x)\f$.
   */
  double value_gradient(const optimizer_point& xb, const optimizer_vector& gb)
    requires gradient_objective<O>
  {
    place(xb);
    ensure_gradient();
    const double f = objective_.value_gradient(optimizer_point(x_), g_);
    for (int i = 0; i < size_; ++i) gb(i) = g_(begin_ + i);
    return f;
  }

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Evaluates the full objective, gradient, and curvature with the active block set to `xb`, writing the block slice into `gb` and the diagonal block into `Hb`.
   *
   * Participates in overload resolution only when `O` satisfies
   * `curvature_objective` and `O::curvature_type` is `dense_curvature_matrix`.
   *
   * \param xb Block point of length `size`.
   * \param gb Writable vector receiving the block gradient slice of length `size`.
   * \param Hb Writable matrix receiving the `size` \f$\times\f$ `size` diagonal curvature block.
   * \return Objective value \f$f(x)\f$.
   */
  double value_gradient_curvature(const optimizer_point& xb, const optimizer_vector& gb, curvature_type& Hb)
    requires curvature_objective<O> && std::same_as<typename O::curvature_type, dense_curvature_matrix>
  {
    place(xb);
    ensure_gradient();
    if (H_.extent(0) == 0) H_ = dense_curvature_matrix("laplace::optimizers::block_objective_view::H", n(), n());
    const double f = objective_.value_gradient_curvature(optimizer_point(x_), g_, H_);
    for (int j = 0; j < size_; ++j) {
      gb(j) = g_(begin_ + j);
      for (int i = 0; i < size_; ++i) Hb(i, j) = H_(begin_ + i, begin_ + j);
    }
    return f;
  }

 private:
  int n() const { return static_cast<int>(x_.extent(0)); }
  void place(const optimizer_point& xb) {
    for (int i = 0; i < size_; ++i) x_(begin_ + i) = xb(i);
  }
  void ensure_gradient() {
    if (g_.extent(0) == 0) g_ = optimizer_vector("laplace::optimizers::block_objective_view::g", n());
  }

  O& objective_;
  int begin_, size_;
  optimizer_vector x_, g_, v_, z_;
  dense_curvature_matrix H_;
  std::vector<int> local_blocks_;
  long nonsmooth_calls_ = 0;
};

namespace detail {

template <class Optimizer>
using stop_rule_of = std::remove_cvref_t<decltype(std::declval<Optimizer&>().stop_rule)>;

// A sub-optimizer whose stop rule has a tolerance in the objective's units:
// the inexact block solve may loosen it.
template <class Optimizer>
concept objective_unit_tolerance = requires(Optimizer& o) {
  { o.stop_rule.tolerance } -> std::convertible_to<double>;
} && ((stop_rule_of<Optimizer>::needs & (stop_needs_decrement | stop_needs_prox_mapping)) != 0u);

}  // namespace detail

/**
 * \ingroup optimizers_block_coordinate
 * \brief Cyclic block-coordinate minimization over sub-optimizers on the host.
 *
 * Splits the coordinate vector \f$x\f$ into `block_count` contiguous blocks
 * with positive sizes given in `block_sizes` summing to `x0.extent(0)` (any
 * other sizes throw `std::invalid_argument`) and minimizes each block in turn
 * using its sub-optimizer from `sub_optimizers` on a `block_objective_view`
 * while holding the other coordinates fixed (cyclic Gauss-Seidel with warm
 * starts). Because each sub-optimizer sees only `block_objective_view<O>`, any
 * objective accepted by all sub-optimizers on their blocks is accepted, and
 * value-only, gradient-based, and curvature-based sub-optimizers can be mixed
 * in one run. Runs are invoked through minimize() or `run()`.
 *
 * When `O` is a `composite_objective` (smooth \f$s\f$ plus block-separable
 * non-smooth \f$r\f$, such as `penalized_objective`), it must also give each
 * non-smooth block's range and per-coordinate proximal members (compile-time
 * error otherwise). The alternation then minimizes \f$f = s + r\f$ over
 * composite views whose value and derivatives are those of \f$s\f$, whose
 * non-smooth part is \f$r\f$ at the full point, and whose non-smooth blocks are
 * the penalties inside the block. A block holding a penalty is solved by a
 * proximal sub-optimizer (`proximal_gradient_optimizer` or
 * `proximal_newton_optimizer`), and a block without penalty by any
 * sub-optimizer (which reports \f$s\f$, and \f$r\f$ is added at its point). Two
 * cases throw `std::invalid_argument`: a penalty straddling the boundary
 * between two blocks, and a sub-optimizer that never calls any non-smooth
 * member on a block holding a penalty.
 *
 * ### Sweeps and inexact solves
 *
 * Each sweep visits the \f$B\f$ blocks (`block_count`) in order from the current
 * point \f$x\f$. A block's result is accepted when its value is finite and not
 * above the current value, and it moves at least one coordinate. On the first
 * sweep, every sub-optimizer runs at its configured tolerances. From the second
 * sweep onward, any sub-optimizer whose `stop_rule` reads
 * `stop_needs_decrement` or `stop_needs_prox_mapping` and has a `tolerance`
 * member convertible to `double` runs with tolerance
 * \f$\max(\tau, \kappa D)\f$, where \f$\tau\f$ is its configured tolerance,
 * \f$\kappa\f$ is `inexact_fraction`, and \f$D\f$ is the decrease of \f$f\f$
 * over the previous sweep. Other rules (such as `search_size_stop_rule` or
 * rule combinators) run as configured.
 *
 * ### Convergence and limits
 *
 * Convergence is never inferred from the sub-runs' statuses alone. It is
 * declared only on a sweep executed entirely at the configured tolerances in
 * which every block declared stationarity at its start without moving beyond
 * resolution. For one block, that requires `optimizer_succeeded()` on its
 * sub-run together with one of three conditions: the block did not move, it
 * moved by at most twice its final `search_size` (for example, a Brent block
 * whose start lies inside its final bracket), or \f$f\f$ fell by at most
 * \f$m\,\varepsilon\,|f|\f$, where \f$m\f$ is `value_resolution_multiple` and
 * \f$\varepsilon\f$ is double-precision machine epsilon.
 *
 * Cyclic alternation reaches stationary points when the non-smooth part is
 * separable across blocks, under the conditions given by Tseng (2001). A kink coupling coordinates in
 * different blocks can hold every block stationary at a point that is not
 * jointly stationary, so any kink must stay inside one block.
 *
 * ### Settings
 *
 * | Member | Default | Effect |
 * |---|---|---|
 * | `sub_optimizers` | default-constructed tuple | One sub-optimizer per block, in block order |
 * | `block_sizes` | zero-initialized array | Positive sizes of contiguous blocks summing to `x0.extent(0)` |
 * | `max_sweeps` | \f$100\f$ | Maximum number of full cyclic sweeps |
 * | `inexact_fraction` | \f$0.1\f$ | Multiplier \f$\kappa\f$ on the previous sweep decrease \f$D\f$ for inexact block tolerances |
 * | `value_resolution_multiple` | \f$64\f$ | Multiplier \f$m\f$ on \f$\varepsilon\,|f|\f$ treating tiny decreases as no move |
 * | `trace` | empty | Optional `optimizer_trace` callback invoked once per sweep with point, value, and summed decrement |
 *
 * ### Report and statuses
 *
 * A run ends with one of the following statuses from
 * \ref optimizers "the optimizers page":
 * - `optimizer_status::start_not_admissible` when \f$f(x_0)\f$ is not finite.
 * - `optimizer_status::converged` when the convergence test passes and no block
 *   ended with `converged_at_resolution`.
 * - `optimizer_status::converged_at_resolution` when the convergence test
 *   passes and at least one block ended with `converged_at_resolution`.
 * - `optimizer_status::stalled` when a sweep at the configured tolerances
 *   moves no coordinate without converging.
 * - `optimizer_status::iteration_budget` when `max_sweeps` sweeps complete
 *   without stopping earlier.
 *
 * In `optimizer_report`, `iterations` counts sweeps, `value_evaluations`,
 * `gradient_evaluations`, and `curvature_evaluations` sum the sub-runs' counts
 * (plus the initial evaluation of \f$f(x_0)\f$), `decrement` is the sum of the
 * blocks' decrements on the last sweep where available (quiet NaN when none
 * gives one), `search_size` is the maximum of the blocks' search sizes on the
 * last sweep where available (quiet NaN when none gives one), and `gradient`
 * is empty.
 *
 * \tparam SubOptimizer Sub-optimizer types, one per block in order.
 * \see \ref optimizers_block_coordinate, block_objective_view, \ref optimizers, minimize()
 */
template <class... SubOptimizer>
struct block_coordinate_optimizer {
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Number of blocks (`sizeof...(SubOptimizer)`).
   */
  static constexpr int block_count = static_cast<int>(sizeof...(SubOptimizer));
  static_assert(block_count > 0, "block_coordinate_optimizer needs at least one block");

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Tuple of sub-optimizers, one per block in block order (default-constructed).
   */
  std::tuple<SubOptimizer...> sub_optimizers;  // one per block, in the order of the blocks
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Contiguous block sizes summing to the dimension of `x0` (default zero-initialized).
   */
  std::array<int, sizeof...(SubOptimizer)> block_sizes{};  // contiguous, summing to the size of x
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Maximum number of cyclic sweeps (default \f$100\f$).
   */
  int max_sweeps = 100;
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Fraction \f$\kappa\f$ of the previous sweep's decrease used for inexact block tolerances (default \f$0.1\f$).
   */
  double inexact_fraction = 0.1;           // block tolerance max(final, fraction * last sweep's decrease)
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Multiplier \f$m\f$ on \f$\varepsilon\,|f|\f$ treating negligible objective falls as no move (default \f$64\f$).
   */
  double value_resolution_multiple = 64.0;  // a fall of f below this times eps |f| is no move
  /**
   * \ingroup optimizers_block_coordinate
   * \brief Optional trace callback invoked once per sweep with point, value, and summed decrement (default empty).
   */
  optimizer_trace trace;                   // once per sweep: point, value, decrement

  /**
   * \ingroup optimizers_block_coordinate
   * \brief Minimizes `objective` starting from `x0` by cyclic block alternation.
   *
   * Throws `std::invalid_argument` if any entry of `block_sizes` is not
   * positive, if their sum differs from `x0.extent(0)`, if a non-smooth block
   * of a composite objective straddles the boundary between two alternation
   * blocks, or if a sub-optimizer never calls any non-smooth member on a block
   * that holds a penalty.
   *
   * \tparam O Objective type accepted on every block through `block_objective_view<O>`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_block_coordinate, block_coordinate_optimizer, block_objective_view, minimize()
   */
  template <value_objective O>
    requires(optimizer_for<SubOptimizer, block_objective_view<O>> && ...)
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    int total = 0;
    for (int s : block_sizes) {
      if (s <= 0) throw std::invalid_argument("block_coordinate_optimizer: every block needs a positive size");
      total += s;
    }
    if (total != n) throw std::invalid_argument("block_coordinate_optimizer: the block sizes must sum to the size of x");

    optimizer_report report;
    optimizer_vector x("x", n);
    Kokkos::deep_copy(x, x0);
    static_assert(!composite_objective<O> || detail::ranged_composite_objective<O>,
                  "block_coordinate_optimizer: a composite objective must give the range of each non-smooth block");
    // f = s + r for a composite objective (its value is s alone), the value otherwise
    double f = objective.value(optimizer_point(x));
    if constexpr (composite_objective<O>) f += objective.nonsmooth_value(optimizer_point(x));
    ++report.value_evaluations;
    const auto finish = [&](optimizer_status status) {
      report.status = status;
      report.point = x;
      report.value = f;
      report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return report;
    };
    if (!std::isfinite(f)) return finish(optimizer_status::start_not_admissible);

    double previous_decrease = std::numeric_limits<double>::quiet_NaN();
    for (int sweep = 1; sweep <= max_sweeps; ++sweep) {
      const double f_sweep = f;
      sweep_state state;
      [&]<std::size_t... B>(std::index_sequence<B...>) {
        int begin = 0;
        ((solve_block(std::get<B>(sub_optimizers), objective, x, f, begin, block_sizes[B], sweep, previous_decrease,
                      state, report),
          begin += block_sizes[B]),
         ...);
      }(std::index_sequence_for<SubOptimizer...>{});
      report.iterations = sweep;
      report.decrement = state.measure_found ? state.measure : std::numeric_limits<double>::quiet_NaN();
      report.search_size = state.search_size_found ? state.search_size : std::numeric_limits<double>::quiet_NaN();
      if (trace) trace(optimizer_trace_point{sweep, x, f, {}, report.decrement, 0.0});
      if (state.all_final && state.all_stationary)
        return finish(state.any_at_resolution ? optimizer_status::converged_at_resolution : optimizer_status::converged);
      // no move at loosened tolerances only says the tolerances were loose: the next
      // sweep runs at the final ones (the decrease is 0)
      if (!state.any_moved && state.all_final) return finish(optimizer_status::stalled);
      previous_decrease = f_sweep - f;
    }
    return finish(optimizer_status::iteration_budget);
  }

 private:
  struct sweep_state {
    bool all_final = true, all_stationary = true, any_moved = false, any_at_resolution = false;
    bool measure_found = false, search_size_found = false;
    double measure = 0.0, search_size = 0.0;
  };

  template <class Sub, class O>
  void solve_block(const Sub& configured, O& objective, const optimizer_vector& x, double& f, int begin, int size,
                   int sweep, double previous_decrease, sweep_state& state, optimizer_report& report) const {
    Sub sub = configured;
    if constexpr (detail::objective_unit_tolerance<Sub>) {
      const double final_tolerance = configured.stop_rule.tolerance;
      if (sweep > 1 && std::isfinite(previous_decrease)) {
        const double t = std::max(final_tolerance, inexact_fraction * previous_decrease);
        sub.stop_rule.tolerance = t;
        if (t > final_tolerance) state.all_final = false;
      }
    }
    block_objective_view<O> view(objective, x, begin, size);
    optimizer_vector xb("xb", size);
    for (int i = 0; i < size; ++i) xb(i) = x(begin + i);
    const optimizer_report r = minimize(sub, view, optimizer_point(xb));
    report.value_evaluations += r.value_evaluations;
    report.gradient_evaluations += r.gradient_evaluations;
    report.curvature_evaluations += r.curvature_evaluations;

    // The sub-run's value in the units of f: a proximal sub-optimizer reports s + r; one
    // that never asked for the non-smooth part reports s, and r is added at its point. Such
    // a sub-optimizer on a block that carries a penalty would have ignored the penalty.
    double value = r.value;
    if constexpr (detail::ranged_composite_objective<O>) {
      if (view.nonsmooth_calls() == 0) {
        if (view.local_nonsmooth_blocks() > 0)
          throw std::invalid_argument(
              "block_coordinate_optimizer: a block carries a non-smooth part that its sub-optimizer ignores");
        if (r.point.extent(0) == static_cast<std::size_t>(size)) {
          optimizer_vector full("full", x.extent(0));
          Kokkos::deep_copy(full, x);
          for (int i = 0; i < size; ++i) full(begin + i) = r.point(i);
          value += objective.nonsmooth_value(optimizer_point(full));
        }
      }
    }

    double move = 0.0;
    const bool usable = r.point.extent(0) == static_cast<std::size_t>(size) && std::isfinite(value) && value <= f;
    if (usable)
      for (int i = 0; i < size; ++i) move = std::max(move, std::abs(r.point(i) - xb(i)));
    const bool moved = usable && move > 0.0;
    const double fall = usable ? f - value : 0.0;
    const bool within_search_size = std::isfinite(r.search_size) && move <= 2.0 * r.search_size;
    const bool flat = fall <= value_resolution_multiple * std::numeric_limits<double>::epsilon() * std::abs(f);
    const bool stationary = optimizer_succeeded(r.status) && (!moved || within_search_size || flat);
    if (moved) {
      for (int i = 0; i < size; ++i) x(begin + i) = r.point(i);
      f = value;
      state.any_moved = true;
    }
    state.all_stationary = state.all_stationary && stationary;
    state.any_at_resolution = state.any_at_resolution || r.status == optimizer_status::converged_at_resolution;
    if (!std::isnan(r.decrement)) state.measure += r.decrement, state.measure_found = true;
    if (!std::isnan(r.search_size))
      state.search_size = std::max(state.search_size, r.search_size), state.search_size_found = true;
  }
};

}  // namespace laplace::optimizers
