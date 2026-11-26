#pragma once

// The penalties of the engine (laplace::engine; plan, section 7.12; interface
// 3.5). A penalty is any penalty of laplace::penalty: minus the log of a prior
// with its normalizing constant (proper or intrinsic), or an unnormalizable one
// (SCAD, MCP), smooth or with kinks. A penalty term applies one penalty to
// several replicates, each a gathered list of model coordinates in any order
// (the group effects of one node or of several, the columns of a smooth): the
// replicates share the hyperparameters, so their state is computed once.
//
// Each hyperparameter theta_j of the penalty is either fixed (a natural value)
// or read from an outer coordinate a: theta_j = phi_a(u_a), with phi_a the
// inverse of the scalar bijector of that coordinate (penalty_hyperparameter).
// Two penalties share a hyperparameter by reading the same coordinate.
//
// The penalty is evaluated on the host, through penalty_evaluator, a type-erased
// wrapper of the template penalty (penalty_evaluator_of<P>). Its derivatives are
// those of P::value_dbeta_dk<J, K> in the block layout of P: block after block,
// the packed symmetric tensor of order J over the coordinates of the block.
// for_each_penalty_entry() walks such a tensor and hands every entry, at every
// distinct permutation of its tuple, with the coordinates in the penalty's own
// numbering.

#include <laplace/bijector/scalar.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/penalty/core.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace laplace::engine {

namespace detail_penalty_term {
// named concepts: nvcc evaluates an inline requires-expression in a member differently from gcc
template <class P>
concept has_prox_beta = requires(const P q, const typename P::state_type& s, const double* v, double* o) {
  q.prox_beta(s, v, 1.0, o, o);
};
template <class P>
concept has_distribution = requires(const P q) { q.distribution; };
template <class P>
concept has_linear_map = requires(const P q) { q.linear_map; };
// one coordinate per block: a diagonal map, or a separable penalty without one (a function template,
// so that the branch not taken is not instantiated: MSVC instantiates both in a non-template lambda)
template <class P>
constexpr bool separable_layout() {
  if constexpr (has_linear_map<P>) return std::decay_t<decltype(std::declval<P>().linear_map)>::is_diagonal;
  else return P::structure == penalty::penalty_structure::separable;
}
}  // namespace detail_penalty_term

/**
 * \ingroup engine
 * \brief Highest derivative orders of a penalty that the engine asks for: 4 in \f$\beta\f$ and 2 in \f$\theta\f$.
 */
inline constexpr int penalty_max_beta_order = 4;
/**
 * \ingroup engine
 * \brief Highest derivative order in \f$\theta\f$ that the engine asks of a penalty.
 */
inline constexpr int penalty_max_theta_order = 2;

/**
 * \ingroup engine
 * \brief Host evaluator of one penalty at fixed hyperparameters, behind a virtual interface.
 *
 * Holds a copy of the penalty, its natural hyperparameters \f$\theta\f$ and the state computed
 * from them. Built by make_penalty_term() through penalty_evaluator_of.
 *
 * \see penalty_evaluator_of, penalty_term, \ref engine
 */
class penalty_evaluator {
 public:
  /**
   * \ingroup engine
   * \brief Virtual destructor.
   */
  virtual ~penalty_evaluator() = default;
  /**
   * \ingroup engine
   * \brief Dimension \f$d\f$ of the coefficient vector \f$\beta\f$.
   * \return Coefficient count \f$d\f$.
   */
  virtual int dim() const = 0;
  /**
   * \ingroup engine
   * \brief Number of natural hyperparameters \f$\theta\f$ of the penalty.
   * \return Hyperparameter count.
   */
  virtual int theta_size() const = 0;
  /**
   * \ingroup engine
   * \brief Normalization class of the penalty (`proper`, `intrinsic`, or `unnormalizable`).
   * \return Normalization class.
   */
  virtual penalty::penalty_normalization normalization() const = 0;
  /**
   * \ingroup engine
   * \brief Geometry of kinks in \f$\beta\f$.
   * \return Kink geometry.
   */
  virtual penalty::penalty_kink_geometry kinks() const = 0;
  /**
   * \ingroup engine
   * \brief Regularity in \f$\beta\f$ (`distrib::infinitely_smooth` or the continuous derivative order \f$r\f$).
   * \return Regularity order in \f$\beta\f$.
   */
  virtual int beta_regularity() const = 0;
  /**
   * \ingroup engine
   * \brief Regularity in \f$\theta\f$ (`distrib::infinitely_smooth` or the continuous derivative order \f$r\f$).
   * \return Regularity order in \f$\theta\f$.
   */
  virtual int theta_regularity() const = 0;
  /**
   * \ingroup engine
   * \brief Whether the underlying penalty provides a single-step proximal operator in \f$\beta\f$.
   * \return `true` if `prox()` is supported.
   */
  virtual bool has_prox() const = 0;
  /**
   * \ingroup engine
   * \brief Coupling structure across coordinates (`separable`, `grouped`, or `coupled`).
   * \return Penalty structure.
   */
  virtual penalty::penalty_structure structure() const = 0;
  /**
   * \ingroup engine
   * \brief Weak-convexity constant \f$\rho = \sup(-\mathrm{pen}'')\f$ at the current \f$\theta\f$.
   * \return Non-negative weak-convexity constant (0 for a convex penalty).
   */
  virtual double weak_convexity() const = 0;
  /**
   * \ingroup engine
   * \brief Number of blocks in the penalty's derivative layout.
   * \return Block count.
   */
  virtual int n_blocks() const = 0;
  /**
   * \ingroup engine
   * \brief First coordinate of block `g` in the penalty's numbering (`dim()` when `g == n_blocks()`).
   * \param g Block index in \f$[0,\f$ `n_blocks()`\f$]\f$.
   * \return Starting coordinate index.
   */
  virtual int block_start(int g) const = 0;
  /**
   * \ingroup engine
   * \brief Sets the natural hyperparameters and computes the state.
   * \param theta `theta_size()` values, copied.
   */
  virtual void set_theta(const double* theta) = 0;
  /**
   * \ingroup engine
   * \brief Evaluates the penalty at \f$\beta\f$ and the current \f$\theta\f$.
   * \param beta `dim()` coefficients.
   * \return The penalty value, its normalizing constant included.
   */
  virtual double value(const double* beta) = 0;
  /**
   * \ingroup engine
   * \brief Derivatives `J` in \f$\beta\f$ and `K` in \f$\theta\f$ along the sorted tuple `idx`, in the block layout.
   * \param J Order in \f$\beta\f$, at most `penalty_max_beta_order`.
   * \param K Order in \f$\theta\f$, at most `penalty_max_theta_order` and at most `theta_regularity()`.
   * \param idx Sorted tuple of `K` indices of \f$\theta\f$.
   * \param beta `dim()` coefficients.
   * \param out `tensor_size(J)` doubles.
   * \return `false`, `out` untouched, when the block is zero by structure.
   */
  virtual bool derivative(int J, int K, const int* idx, const double* beta, double* out) = 0;
  /**
   * \ingroup engine
   * \brief Proximal operator \f$\arg\min_b \{\mathrm{pen}(b) + \lVert b - v\rVert^2/(2t)\}\f$ at the current \f$\theta\f$.
   * \param v `dim()` values.
   * \param t Step.
   * \param out `dim()` values.
   */
  virtual void prox(const double* v, double t, double* out) = 0;
  /**
   * \ingroup engine
   * \brief Clones the evaluator, preserving its current \f$\theta\f$ and state.
   * \return Owning pointer to a copy.
   */
  virtual std::unique_ptr<penalty_evaluator> clone() const = 0;

  /**
   * \ingroup engine
   * \brief Number of doubles in a block-layout derivative tensor of order `J` in \f$\beta\f$.
   * \param J Derivative order in \f$\beta\f$ (`0` returns `1`).
   * \return Total scalar entries summed across all blocks.
   */
  int tensor_size(int J) const {
    if (J == 0) return 1;
    int n = 0;
    for (int g = 0; g < n_blocks(); ++g) n += distrib::sym_tensor_size(block_start(g + 1) - block_start(g), J);
    return n;
  }
};

/**
 * \ingroup engine
 * \brief The penalty_evaluator of a penalty type `P` of laplace::penalty.
 *
 * The block layout is that of `P`: the blocks of its distribution for a `distrib_penalty`, one
 * coordinate per block for a separable penalty without a distribution (SCAD, MCP, a diagonal map),
 * one block otherwise.
 *
 * \tparam P Type satisfying `penalty::penalty`, with `theta_state`, `value` and `value_dbeta_dk`.
 * \see penalty_evaluator, make_penalty_term(), \ref engine
 */
template <penalty::penalty P>
class penalty_evaluator_of final : public penalty_evaluator {
 public:
  /**
   * \brief Wraps a copy of `p`, with no state until set_theta().
   * \param p The penalty.
   */
  explicit penalty_evaluator_of(const P& p)
      : p_(p), theta_(p.theta_size(), 0.0), w_(static_cast<std::size_t>(p.state_size() + p.scratch_size()) + 1, 0.0) {}
  penalty_evaluator_of(const penalty_evaluator_of& o) : p_(o.p_), theta_(o.theta_), w_(o.w_.size(), 0.0) {
    if (o.have_) set_theta(theta_.data());
  }

  int dim() const override { return p_.dim(); }
  int theta_size() const override { return p_.theta_size(); }
  penalty::penalty_normalization normalization() const override { return P::normalization; }
  penalty::penalty_kink_geometry kinks() const override { return P::kinks; }
  int beta_regularity() const override { return P::beta_regularity; }
  int theta_regularity() const override { return P::theta_regularity; }
  bool has_prox() const override { return detail_penalty_term::has_prox_beta<P>; }
  penalty::penalty_structure structure() const override { return P::structure; }
  double weak_convexity() const override { return p_.weak_convexity(theta_.data()); }
  int n_blocks() const override {
    if constexpr (detail_penalty_term::has_distribution<P>) return distrib::mv_n_blocks(p_.distribution);
    else if constexpr (separable_layout) return p_.dim();
    else return 1;
  }
  int block_start(int g) const override {
    if constexpr (detail_penalty_term::has_distribution<P>) return distrib::mv_block_start(p_.distribution, distrib::y_slot, g);
    else if constexpr (separable_layout) return g;
    else return g == 0 ? 0 : p_.dim();
  }
  void set_theta(const double* theta) override {
    std::copy(theta, theta + p_.theta_size(), theta_.begin());
    st_ = p_.theta_state(theta_.data(), w_.data());
    have_ = true;
  }
  double value(const double* beta) override { return p_.value(st_, beta, scratch()); }
  bool derivative(int J, int K, const int* idx, const double* beta, double* out) override {
    if (K > penalty_max_theta_order || J > penalty_max_beta_order || K > P::theta_regularity)
      throw std::invalid_argument("penalty_evaluator: a derivative order the engine does not ask for");
    return dispatch_j(J, K, idx, beta, out, std::make_integer_sequence<int, penalty_max_beta_order + 1>{});
  }
  void prox(const double* v, double t, double* out) override {
    if constexpr (detail_penalty_term::has_prox_beta<P>)
      p_.prox_beta(st_, v, t, out, scratch());
    else
      throw std::invalid_argument("penalty_evaluator: the penalty has no proximal operator");
  }
  std::unique_ptr<penalty_evaluator> clone() const override { return std::make_unique<penalty_evaluator_of>(*this); }

 private:
  static constexpr bool separable_layout = detail_penalty_term::separable_layout<P>();
  P p_;
  std::vector<double> theta_, w_;
  typename P::state_type st_{};
  bool have_ = false;

  double* scratch() { return w_.data() + p_.state_size(); }

  template <int J, int K>
  bool call(const int* idx, const double* beta, double* out) {
    if constexpr (K > P::theta_regularity) return false;  // refused in derivative(); never reached
    else return p_.template value_dbeta_dk<J, K>(st_, beta, idx, out, scratch());
  }
  template <int J, int... Ks>
  bool dispatch_k(int K, const int* idx, const double* beta, double* out, std::integer_sequence<int, Ks...>) {
    bool r = false;
    ((K == Ks ? (r = call<J, Ks>(idx, beta, out), true) : false) || ...);
    return r;
  }
  template <int... Js>
  bool dispatch_j(int J, int K, const int* idx, const double* beta, double* out, std::integer_sequence<int, Js...>) {
    bool r = false;
    ((J == Js ? (r = dispatch_k<Js>(K, idx, beta, out, std::make_integer_sequence<int, penalty_max_theta_order + 1>{}), true)
              : false) ||
     ...);
    return r;
  }
};

/**
 * \ingroup engine
 * \brief An outer coordinate that carries a hyperparameter: the bijector from its natural value to the free scale.
 *
 * The natural value is \f$\theta = \phi(u)\f$, \f$\phi\f$ the inverse of `map`: `scalar_bijector::lower_bounded(0)`
 * (the logarithm) for a scale or a rate, the identity for a parameter of a `pdmat` structure, which
 * is already free.
 *
 * \see penalty_term, \ref engine
 */
struct penalty_hyperparameter {
  bijector::scalar_bijector map{};  ///< natural value to free coordinate
};

/**
 * \ingroup engine
 * \brief One penalty applied to several gathered lists of model coordinates, sharing its hyperparameters.
 *
 * \see make_penalty_term(), penalty_evaluator, \ref engine
 */
struct penalty_term {
  std::shared_ptr<const penalty_evaluator> penalty;  ///< the penalty (cloned by whoever evaluates it)
  std::vector<std::vector<int>> replicates;          ///< each a list of `penalty->dim()` model coordinates
  std::vector<int> theta_source;                     ///< per hyperparameter: its outer coordinate, or \f$-1\f$ when fixed
  std::vector<double> theta_value;                   ///< per hyperparameter: its natural value where fixed
};

/**
 * \ingroup engine
 * \brief Builds a penalty_term from a penalty of laplace::penalty.
 * \tparam P Penalty type.
 * \param p The penalty.
 * \param replicates The gathered coordinate lists, each of length `p.dim()`.
 * \param theta_source Per hyperparameter, the index of the outer hyperparameter it reads, or \f$-1\f$ for fixed.
 * \param theta_value Per hyperparameter, its natural value (used where fixed; it may be empty when none is fixed).
 * \return The term.
 */
template <penalty::penalty P>
penalty_term make_penalty_term(const P& p, std::vector<std::vector<int>> replicates, std::vector<int> theta_source,
                               std::vector<double> theta_value = {}) {
  const int h = p.theta_size();
  if (static_cast<int>(theta_source.size()) != h)
    throw std::invalid_argument("make_penalty_term: one source per hyperparameter");
  if (theta_value.empty()) theta_value.assign(h, 0.0);
  if (static_cast<int>(theta_value.size()) != h) throw std::invalid_argument("make_penalty_term: one value per hyperparameter");
  for (int j = 0; j < h; ++j)
    if (theta_source[j] < 0 && theta_source[j] != -1) throw std::invalid_argument("make_penalty_term: a source below -1");
  for (const auto& r : replicates)
    if (static_cast<int>(r.size()) != p.dim()) throw std::invalid_argument("make_penalty_term: a replicate of the wrong length");
  return penalty_term{std::make_shared<penalty_evaluator_of<P>>(p), std::move(replicates), std::move(theta_source),
                      std::move(theta_value)};
}

/**
 * \ingroup engine
 * \brief Calls `f(c, value)` for every entry of a derivative of order `J` in \f$\beta\f$, at every distinct permutation of its tuple.
 *
 * `c` holds `J` coordinates in the penalty's numbering; the sum of `f` over the calls is the full
 * contraction of the symmetric tensor.
 *
 * \param e The evaluator whose block layout the tensor has.
 * \param J Order, \f$1 \le J \le\f$ penalty_max_beta_order.
 * \param t The tensor, `e.tensor_size(J)` doubles.
 * \param f Callable `(const int* c, double value)`.
 */
template <class F>
void for_each_penalty_entry(const penalty_evaluator& e, int J, const double* t, F&& f) {
  int q = 0;
  for (int g = 0; g < e.n_blocks(); ++g) {
    const int s = e.block_start(g), n = e.block_start(g + 1) - s;
    int tu[penalty_max_beta_order] = {}, c[penalty_max_beta_order];
    do {
      const double v = t[q++];
      if (v != 0.0) {
        for (int r = 0; r < J; ++r) c[r] = s + tu[r];
        std::sort(c, c + J);
        do f(static_cast<const int*>(c), v);
        while (std::next_permutation(c, c + J));
      }
    } while (distrib::detail_mv::next_block_tuple(n, J, tu));
  }
}

}  // namespace laplace::engine
