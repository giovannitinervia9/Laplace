#pragma once

// penalized_objective<Smooth, Penalty...> (note/optimizers.md, section 12,
// decision 1): the composite objective
//
//   f(x) = s(x) + sum_b pen_b(x[o_b, o_b + d_b); theta_b),
//
// s a smooth objective (Smooth, C1 or C2) and each penalty pen_b of
// laplace::penalty attached to the contiguous range of d_b = pen_b.dim()
// coordinates starting at o_b, with its hyperparameters theta_b held fixed.
// The ranges must not overlap; coordinates outside every range carry no
// penalty. value, value_gradient and value_gradient_curvature are those of s
// alone (forwarded), and the members of composite_objective forward to the
// penalties: nonsmooth_value to value, nonsmooth_prox to prox_beta block by
// block (the identity outside the ranges), weak_convexity_constant to
// weak_convexity. A penalty without prox_beta (a smooth one, which belongs in
// s; note/penalties.md section 3) does not compile here.
//
// The members used by proximal Newton's coordinate descent
// (coordinate_composite_objective in proximal_newton.hpp) are also given:
// the range of block b, whether it is separable with its kinks on the
// coordinates, and the prox of one coordinate of it.
//
// Each call recomputes the penalty's state from theta_b (theta_state), which
// costs a gather of theta_b, so that the object can be copied and moved
// without pointers into its own buffers.

#include "laplace/optimizers/objective.hpp"
#include "laplace/penalty/core.hpp"

#include <Kokkos_Core.hpp>

#include <concepts>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_proximal
 * \brief Penalty attached to a contiguous coordinate range \f$[o, o + d)\f$ of \f$x\f$ with fixed hyperparameters \f$\theta\f$.
 *
 * Constructed via attach_penalty() and passed to penalized_objective. The penalty must satisfy
 * `laplace::penalty::penalty` and provide `prox_beta()`.
 *
 * \tparam P Penalty type satisfying `laplace::penalty::penalty` with `prox_beta()`.
 * \see \ref optimizers_proximal, attach_penalty(), penalized_objective
 */
template <class P>
struct attached_penalty {
  /// \brief Penalty object.
  P penalty;
  /// \brief First coordinate \f$o\f$ of the range \f$[o, o + d)\f$ with \f$d =\f$ `penalty.dim()` (default \f$0\f$).
  int offset = 0;
  /// \brief Fixed hyperparameter vector \f$\theta\f$ of length `penalty.theta_size()`.
  std::vector<double> theta;
};

/**
 * \ingroup optimizers_proximal
 * \brief Constructs an attached_penalty binding `penalty` to coordinates \f$[o, o + d)\f$ with fixed hyperparameters `theta`.
 *
 * \tparam P Penalty type.
 * \param penalty Penalty object.
 * \param offset First penalized coordinate \f$o\f$ of \f$x\f$.
 * \param theta Hyperparameters \f$\theta\f$, of length `penalty.theta_size()`.
 * \return The constructed attached_penalty.
 * \see \ref optimizers_proximal, attached_penalty, penalized_objective
 */
template <class P>
attached_penalty<P> attach_penalty(const P& penalty, int offset, std::vector<double> theta) {
  return attached_penalty<P>{penalty, offset, std::move(theta)};
}

namespace detail {

// the penalty has the prox the composite form needs
template <class P>
concept penalty_with_prox = laplace::penalty::penalty<P> && requires(const P p, const typename P::state_type& st, const double* v,
                                                            double* out) { p.prox_beta(st, v, 1.0, out, out); };

// curvature_type of Smooth when it has one
template <class Smooth>
struct penalized_curvature_base {};
template <curvature_objective Smooth>
struct penalized_curvature_base<Smooth> {
  using curvature_type = typename Smooth::curvature_type;
};

// one penalty with its scratch: the state and the workspace of prox_beta and
// value, and two vectors of the block's length for the per-coordinate prox
template <class P>
struct penalty_slot {
  attached_penalty<P> attached;
  std::vector<double> work, v, out;
  explicit penalty_slot(attached_penalty<P> a) : attached(std::move(a)) {
    const P& p = attached.penalty;
    if (static_cast<int>(attached.theta.size()) != p.theta_size())
      throw std::invalid_argument("laplace::optimizers::penalized_objective: theta has " +
                                  std::to_string(attached.theta.size()) + " entries, the penalty takes " +
                                  std::to_string(p.theta_size()));
    work.assign(static_cast<std::size_t>(p.state_size() + p.scratch_size()), 0.0);
    v.assign(static_cast<std::size_t>(p.dim()), 0.0);
    out.assign(static_cast<std::size_t>(p.dim()), 0.0);
  }
  int dim() const { return attached.penalty.dim(); }
  typename P::state_type state() { return attached.penalty.theta_state(attached.theta.data(), work.data()); }
  double* scratch() { return work.data() + attached.penalty.state_size(); }
};

}  // namespace detail

/**
 * \ingroup optimizers_proximal
 * \brief Composite objective \f$f(x) = s(x) + \sum_b \mathrm{pen}_b(x_b; \theta_b)\f$ built from a smooth objective and attached penalties.
 *
 * Satisfies `composite_objective` and `coordinate_composite_objective`, and satisfies `curvature_objective`
 * when `Smooth` does. `value()`, `value_gradient()`, and `value_gradient_curvature()` evaluate \f$s\f$ alone;
 * `nonsmooth_value()` sums the penalties' values (minus log densities, normalizing constants included);
 * `nonsmooth_prox()` applies each penalty's `prox_beta()` on its range and leaves uncovered coordinates unchanged;
 * `weak_convexity_constant()` returns each penalty's `weak_convexity()`.
 * Each call recomputes the penalty state from \f$\theta_b\f$ without internal self-pointers, so the object can be copied.
 * A penalty without `prox_beta()` does not compile.
 *
 * \tparam Smooth Smooth objective type satisfying `gradient_objective`, held by reference.
 * \tparam P Penalty types satisfying `laplace::penalty::penalty` with `prox_beta()`.
 * \see \ref optimizers_proximal, attached_penalty, attach_penalty(), proximal_gradient_optimizer, proximal_newton_optimizer
 */
template <gradient_objective Smooth, detail::penalty_with_prox... P>
class penalized_objective : public detail::penalized_curvature_base<Smooth> {
 public:
  /**
   * \brief Builds the composite objective over dimension \f$n\f$.
   *
   * Throws `std::invalid_argument` if any range leaves \f$[0, n)\f$, two ranges overlap,
   * or any `theta` does not have `penalty.theta_size()` entries.
   *
   * \param smooth Smooth objective \f$s\f$ (held by reference, not copied).
   * \param n Dimension of \f$x\f$.
   * \param penalties Attached penalties, one per block \f$b\f$ in argument order.
   */
  penalized_objective(Smooth& smooth, int n, attached_penalty<P>... penalties)
      : smooth_(smooth), n_(n), slots_(detail::penalty_slot<P>(std::move(penalties))...) {
    std::vector<std::pair<int, int>> ranges;
    std::apply([&](auto&... s) { (ranges.emplace_back(s.attached.offset, s.attached.offset + s.dim()), ...); }, slots_);
    for (std::size_t a = 0; a < ranges.size(); ++a) {
      if (ranges[a].first < 0 || ranges[a].second > n)
        throw std::invalid_argument("laplace::optimizers::penalized_objective: a penalty's range leaves [0, n)");
      for (std::size_t b = 0; b < a; ++b)
        if (ranges[a].first < ranges[b].second && ranges[b].first < ranges[a].second)
          throw std::invalid_argument("laplace::optimizers::penalized_objective: two penalties' ranges overlap");
    }
  }

  /**
   * \brief Evaluates the smooth part \f$s(x)\f$.
   * \param x Point of length \f$n\f$.
   * \return \f$s(x)\f$.
   */
  double value(const optimizer_point& x) { return smooth_.value(x); }
  /**
   * \brief Evaluates \f$s(x)\f$ and writes \f$\nabla s(x)\f$ to `g`.
   * \param x Point of length \f$n\f$.
   * \param g Output gradient vector of length \f$n\f$.
   * \return \f$s(x)\f$.
   */
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) { return smooth_.value_gradient(x, g); }
  /**
   * \brief Evaluates \f$s(x)\f$, writes \f$\nabla s(x)\f$ to `g`, and writes the curvature of \f$s\f$ to `curvature`.
   *
   * Available when `Smooth` satisfies `curvature_objective`.
   * \tparam H Curvature matrix type matching `Smooth::curvature_type`.
   * \param x Point of length \f$n\f$.
   * \param g Output gradient vector of length \f$n\f$.
   * \param curvature Output curvature matrix.
   * \return \f$s(x)\f$.
   */
  template <class H>
    requires curvature_objective<Smooth> && std::same_as<H, typename Smooth::curvature_type>
  double value_gradient_curvature(const optimizer_point& x, const optimizer_vector& g, H& curvature) {
    return smooth_.value_gradient_curvature(x, g, curvature);
  }

  /**
   * \brief Evaluates the nonsmooth sum \f$r(x) = \sum_b \mathrm{pen}_b(x_b; \theta_b)\f$, normalizing constants included.
   * \param x Point of length \f$n\f$.
   * \return \f$r(x)\f$.
   */
  double nonsmooth_value(const optimizer_point& x) {
    double r = 0.0;
    std::apply([&](auto&... s) { ((r += block_value(s, x.data())), ...); }, slots_);
    return r;
  }

  /**
   * \brief Computes \f$z = \operatorname{prox}_{t r}(v)\f$, applying each penalty's `prox_beta()` on its range and the identity elsewhere.
   * \param v Input point of length \f$n\f$.
   * \param t Proximal step \f$t > 0\f$ (exact when \f$t \rho_b < 1\f$ for every block \f$b\f$).
   * \param z Output point of length \f$n\f$ (must not alias `v`).
   */
  void nonsmooth_prox(const optimizer_point& v, double t, const optimizer_vector& z) {
    for (int i = 0; i < n_; ++i) z(i) = v(i);
    std::apply([&](auto&... s) { (block_prox(s, v.data(), t, z.data()), ...); }, slots_);
  }

  /**
   * \brief Returns the number of penalized blocks.
   * \return Number of attached penalties.
   */
  int nonsmooth_block_count() const { return static_cast<int>(sizeof...(P)); }

  /**
   * \brief Returns the weak-convexity constant \f$\rho_b\f$ of block `b` at its \f$\theta_b\f$.
   * \param b Zero-based block index.
   * \return \f$\rho_b \ge 0\f$ (\f$0\f$ for a convex penalty).
   */
  double weak_convexity_constant(int b) {
    double rho = 0.0;
    visit(b, [&](auto& s) { rho = s.attached.penalty.weak_convexity(s.attached.theta.data()); });
    return rho;
  }

  /**
   * \brief Returns the coordinate range \f$[o_b, o_b + d_b)\f$ of block `b`.
   * \param b Zero-based block index.
   * \return Pair \f$(o_b, d_b)\f$ giving the starting offset and length.
   */
  std::pair<int, int> nonsmooth_block_range(int b) {
    std::pair<int, int> r{0, 0};
    visit(b, [&](auto& s) { r = {s.attached.offset, s.dim()}; });
    return r;
  }

  /**
   * \brief Reports whether block `b` is separable with kinks at most per coordinate.
   * \param b Zero-based block index.
   * \return `true` when the penalty's `structure` is `separable` and its `kinks` are `per_coordinate` or `none`.
   */
  bool nonsmooth_block_separable(int b) {
    bool sep = false;
    visit(b, [&](auto& s) {
      using Q = std::remove_cvref_t<decltype(s.attached.penalty)>;
      sep = Q::structure == laplace::penalty::penalty_structure::separable &&
            (Q::kinks == laplace::penalty::penalty_kink_geometry::per_coordinate || Q::kinks == laplace::penalty::penalty_kink_geometry::none);
    });
    return sep;
  }

  /**
   * \brief Computes coordinate \f$j\f$ of block `b`'s proximal operator at target `vj` with step `t`.
   *
   * Evaluates one `prox_beta()` call of block `b` on a scratch copy in \f$O(d_b)\f$ operations.
   * \param b Zero-based block index.
   * \param j Coordinate index within block `b`, in \f$[0, d_b)\f$.
   * \param vj Target value \f$v_j\f$.
   * \param t Proximal step \f$t > 0\f$.
   * \return \f$[\operatorname{prox}_{t\,\mathrm{pen}_b}(v)]_j\f$ for any \f$v\f$ with coordinate \f$j\f$ equal to `vj`.
   */
  // ponytail: one prox_beta of the whole block per coordinate, O(d_b); the same
  // order as a coordinate-descent update with a dense H. A per-coordinate member
  // of the penalty (note/penalties.md 5c.3) when a sparse H makes it the cost.
  double nonsmooth_coordinate_prox(int b, int j, double vj, double t) {
    double out = vj;
    visit(b, [&](auto& s) {
      s.v[static_cast<std::size_t>(j)] = vj;
      const auto st = s.state();
      s.attached.penalty.prox_beta(st, s.v.data(), t, s.out.data(), s.scratch());
      out = s.out[static_cast<std::size_t>(j)];
    });
    return out;
  }

  /**
   * \brief Returns the dimension \f$n\f$ of \f$x\f$.
   * \return \f$n\f$.
   */
  int dimension() const { return n_; }

 private:
  template <class S>
  static double block_value(S& s, const double* x) {
    const auto st = s.state();
    return s.attached.penalty.value(st, x + s.attached.offset, s.scratch());
  }
  template <class S>
  static void block_prox(S& s, const double* v, double t, double* z) {
    const auto st = s.state();
    s.attached.penalty.prox_beta(st, v + s.attached.offset, t, z + s.attached.offset, s.scratch());
  }
  template <class F>
  void visit(int b, F&& f) {
    int k = 0;
    std::apply([&](auto&... s) { ((k++ == b ? f(s) : void()), ...); }, slots_);
  }

  Smooth& smooth_;
  int n_;
  std::tuple<detail::penalty_slot<P>...> slots_;
};

}  // namespace laplace::optimizers
