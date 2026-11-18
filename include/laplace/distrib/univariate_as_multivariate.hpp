// univariate_as_multivariate<D>: a univariate distrib D read as a multivariate
// distrib of dimension 1 (note/penalties.md section 5a). D's parameter 0, its
// location (location first), becomes the dense parameter mu of size 1; D's
// parameters 1, ..., P - 1 become scalar parameters, their coordinates the
// array theta of length P - 1 in D's order. A tuple idx over theta is the tuple
// idx + 1 over D's parameters, so every derivative is one component of D's own
// tensors: logpdf_dy_dk<J, K, A>(y) is the entry (0^A, idx + 1) of
// D::logpdf_dy_dk<J, K + A>(y). The state only points at theta.
//
// Any univariate law (Giovanni, 2026-11-17). The adapter keeps a prototype of
// D, which carries what is not a parameter (truncation bounds, a map, atom
// points), and rebuilds the law at each call by copying the prototype and
// writing (x[0], theta) over its first P doubles: in every univariate law the
// P parameters are the first P doubles of the object, in params order (an
// aggregate is only its parameters; a wrapper holds its parent first, then its
// own parameters, then the rest). A discrete D gives a discrete law (logpmf,
// logpmf_dk, no derivatives in y); the atoms of D are the coordinate atoms of
// coordinate 0; mu keeps D's bounds (a positive mean stays positive: the link
// is the engine's).
#pragma once

#include <laplace/distrib/core.hpp>
#include <laplace/distrib/multivariate_core.hpp>

#include <cstring>
#include <type_traits>
#include <utility>

namespace laplace::distrib {

/**
 * \ingroup distrib_independent_product
 * \brief State of `univariate_as_multivariate<D>`: a pointer to \f$\theta\f$.
 */
struct univariate_as_multivariate_state {
  /// The scalar parameters of `D` after its location, length `theta_size()`.
  const double* theta;
};

namespace detail_uam {
template <class D>
struct param_table {
  distrib_mv_param_info v[D::n_params];
};
template <class D>
constexpr param_table<D> make_param_table() {
  param_table<D> t{};
  for (int j = 0; j < D::n_params; ++j)
    t.v[j] = {D::params[j].name, j == 0 ? distrib_mv_param_kind::vector : distrib_mv_param_kind::scalar,
              D::params[j].lower, D::params[j].upper, D::params[j].regularity};
  return t;
}
template <class D>
inline constexpr param_table<D> params_of = make_param_table<D>();
// whether D has derivatives in y; a named concept, since nvcc rejects the same
// requires-expression written inline in a member's constraint
template <class D>
concept has_logpdf_dy_dk = requires(const D d, double v) { d.template logpdf_dy_dk<1, 0>(v); };
// whether D has the optional prox_y(v, t) (note/penalties.md section 5c); a named concept,
// since nvcc rejects the requires-expression written inline in a member's constraint
template <class D>
concept has_prox_y = requires(const D d, double a) { d.prox_y(a, a); };
}  // namespace detail_uam

/**
 * \ingroup distrib_independent_product
 * \brief A univariate distribution `D` as a multivariate distribution of dimension 1.
 *
 * The location of `D` (its parameter 0) is the dense parameter \f$\mu\f$ of
 * size 1, and the other parameters of `D`, in their order, are the scalar
 * parameters whose coordinates form \f$\theta\f$. Satisfies `multivariate_distrib`.
 *
 * `D` may be continuous or discrete, with or without atoms (see \ref distrib_atoms "the atoms of a distribution"). The kind, the atoms and the bounds of \f$\mu\f$ are those of `D`; the `prototype` member keeps what is not a parameter.
 *
 * \tparam D Univariate distribution whose \f$P\f$ parameters are its first \f$P\f$ doubles, in `params` order.
 */
template <class D>
struct univariate_as_multivariate {
  static_assert(std::is_trivially_copyable_v<D> && sizeof(D) >= D::n_params * sizeof(double),
                "univariate_as_multivariate: D must be trivially copyable and hold its parameters as its first doubles");

  /// The law that the adapter copies, its parameters overwritten at each call: what is not a parameter
  /// (bounds, a map, atom points) is read from it.
  D prototype{};

  univariate_as_multivariate() = default;
  /**
   * \brief Construct from a prototype law.
   * \param p The law whose non-parameter members (bounds, map, atom points) the adapter keeps.
   */
  LAPLACE_HD explicit univariate_as_multivariate(const D& p) : prototype(p) {}

  /// Normalization: `D` is a proper law.
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// Kind of the law, that of `D`.
  static constexpr distrib_kind kind = D::kind;
  /// Number of typed parameters, those of `D`.
  static constexpr int n_params = D::n_params;
  /// Parameter descriptors: \f$\mu\f$ (vector of size 1), then the scalars of `D`.
  static constexpr const distrib_mv_param_info (&params)[n_params] = detail_uam::params_of<D>.v;
  /// State type.
  using state_type = univariate_as_multivariate_state;

  /// \return The dimension, 1.
  LAPLACE_HD int dim() const { return 1; }
  /// Number of \f$y\f$ coordinates of the derivatives in \f$y\f$ (`dim()`: the unit is not constrained).
  LAPLACE_HD int y_coord_size() const { return 1; }
  /// \param j Parameter index. \return 1, every parameter has one coordinate.
  LAPLACE_HD int param_size(int j) const { return (void)j, 1; }
  /// \param j Parameter index. \return 0 for \f$\mu\f$ (in \f$x\f$), \f$j - 1\f$ in \f$\theta\f$ otherwise.
  LAPLACE_HD int param_offset(int j) const { return j == 0 ? 0 : j - 1; }
  /// \return The length of \f$\theta\f$, \f$P - 1\f$.
  LAPLACE_HD int theta_size() const { return n_params - 1; }
  /// \return 0: the state holds no doubles.
  LAPLACE_HD int state_size() const { return 0; }
  /// \return 0: no scratch is used.
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \brief Return the state, a pointer to \f$\theta\f$.
   * \param theta Scalar parameters of `D` after the location, length `theta_size()`.
   * \param w Unused.
   * \return The state.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double* w) const { return (void)w, state_type{theta}; }

  /**
   * \brief The log-density of `D` at \f$y_0\f$ with location \f$x_0\f$.
   * \param st State from `theta_state()`.
   * \param y Observation, length 1.
   * \param x Location, length 1.
   * \return \f$\log f_D(y_0 \mid x_0, \theta)\f$. At an atom of `D` it is the log mass.
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* x) const
    requires(D::kind == distrib_kind::continuous)
  {
    return law(x[0], st.theta).logpdf(y[0]);
  }

  /**
   * \brief The log mass of a discrete `D` at \f$y_0\f$ with location \f$x_0\f$.
   * \param st State from `theta_state()`.
   * \param y Observation, length 1.
   * \param x Location, length 1.
   * \return \f$\log P_D(Y = y_0 \mid x_0, \theta)\f$.
   */
  LAPLACE_HD double logpmf(const state_type& st, const double* y, const double* x) const
    requires(D::kind == distrib_kind::discrete)
  {
    return law(x[0], st.theta).logpmf(y[0]);
  }

  /// \param i Coordinate of \f$y\f$ (only 0). \return The number of atoms of `D`.
  LAPLACE_HD int n_coord_atoms(int i) const
    requires(distrib_n_atoms<D>() > 0)
  {
    return (void)i, distrib_n_atoms<D>();
  }
  /// \param i Coordinate of \f$y\f$ (only 0). \param a Atom index. \return Where atom `a` of `D` sits.
  LAPLACE_HD double coord_atom_point(int i, int a) const
    requires(distrib_n_atoms<D>() > 0)
  {
    return (void)i, distrib_atom_point(prototype, a);
  }

  /**
   * \brief Derivative of order `J` in \f$y\f$, `K` in \f$\theta\f$ along `idx` and `A` in \f$\mu\f$, continuous `D`.
   *
   * Writes one double: the component \f$(0^A, \mathrm{idx} + 1)\f$ of
   * `D::logpdf_dy_dk<J, K + A>(y)` (of `D::logpdf_dk<K + A>(y)` when \f$J = 0\f$,
   * so that a law without derivatives in \f$y\f$ still gives its parameter
   * derivatives). Always returns `true`. At an atom of `D` the value with
   * \f$J = 0\f$ is that of the log mass; \f$J \ge 1\f$ has no meaning there.
   *
   * \tparam J Order in \f$y\f$.
   * \tparam K Order in \f$\theta\f$ (\f$J + K + A \le 6\f$).
   * \tparam A Order in \f$\mu\f$ (omitted means 0).
   * \param st State from `theta_state()`.
   * \param y Observation, length 1.
   * \param x Location, length 1.
   * \param idx Sorted tuple of `K` indices into \f$\theta\f$ (unused when `K == 0`).
   * \param out Output, one double.
   * \param w Unused.
   * \return `true`.
   */
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                               double* w) const
    requires(D::kind == distrib_kind::continuous && (J == 0 || detail_uam::has_logpdf_dy_dk<D>))
  {
    (void)w;
    constexpr int a = detail_mv::dense_order<A...>();
    const D d = law(x[0], st.theta);
    if constexpr (J == 0 && K + a == 0) out[0] = d.logpdf(y[0]);
    else if constexpr (J == 0) out[0] = component<K, a>(d.template logpdf_dk<K + a>(y[0]), idx);
    else out[0] = component<K, a>(d.template logpdf_dy_dk<J, K + a>(y[0]), idx);
    return true;
  }

  /**
   * \brief Derivative of order `K` in \f$\theta\f$ along `idx` and `A` in \f$\mu\f$ of the log mass, discrete `D`.
   *
   * Writes one double: the component \f$(0^A, \mathrm{idx} + 1)\f$ of
   * `D::logpmf_dk<K + A>(y)`. Always returns `true`.
   *
   * \tparam K Order in \f$\theta\f$.
   * \tparam A Order in \f$\mu\f$ (\f$K + A \le 6\f$; omitted means 0).
   * \param st State from `theta_state()`.
   * \param y Observation, length 1.
   * \param x Location, length 1.
   * \param idx Sorted tuple of `K` indices into \f$\theta\f$ (unused when `K == 0`).
   * \param out Output, one double.
   * \param w Unused.
   * \return `true`.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpmf_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                            double* w) const
    requires(D::kind == distrib_kind::discrete)
  {
    (void)w;
    constexpr int a = detail_mv::dense_order<A...>();
    const D d = law(x[0], st.theta);
    if constexpr (K + a == 0) out[0] = d.logpmf(y[0]);
    else out[0] = component<K, a>(d.template logpmf_dk<K + a>(y[0]), idx);
    return true;
  }

  /**
   * \brief Derivative of order `K` in \f$\theta\f$ along `idx` and `A` in \f$\mu\f$: `logpdf_dy_dk<0, K, A...>`.
   * \tparam K Order in \f$\theta\f$.
   * \tparam A Order in \f$\mu\f$ (\f$K + A \le 6\f$; omitted means 0).
   * \param st State from `theta_state()`.
   * \param y Observation, length 1.
   * \param x Location, length 1.
   * \param idx Sorted tuple of `K` indices into \f$\theta\f$ (unused when `K == 0`).
   * \param out Output, one double.
   * \param w Unused.
   * \return `true`.
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                            double* w) const
    requires(D::kind == distrib_kind::continuous)
  {
    return logpdf_dy_dk<0, K, A...>(st, y, x, idx, out, w);
  }

  /**
   * \brief Proximal operator in \f$y\f$, `D::prox_y` at location \f$x_0\f$; present when `D` has it.
   * \param st State from `theta_state()`.
   * \param v Point, length 1.
   * \param x Location, length 1.
   * \param t Step \f$t > 0\f$.
   * \param out Minimizer, length 1.
   * \param w Unused.
   */
  LAPLACE_HD void prox_y(const state_type& st, const double* v, const double* x, double t, double* out,
                         double* w) const
    requires detail_uam::has_prox_y<D>
  {
    (void)w;
    out[0] = law(x[0], st.theta).prox_y(v[0], t);
  }

  /**
   * \brief Proximal operator in \f$y\f$ with a step per coordinate, `D::prox_y` at step `t[0]`; present when `D` has it.
   * \param st State from `theta_state()`.
   * \param v Point, length 1.
   * \param x Location, length 1.
   * \param t Step per coordinate, length 1, \f$t_0 > 0\f$.
   * \param out Minimizer, length 1.
   * \param w Unused.
   */
  LAPLACE_HD void prox_y(const state_type& st, const double* v, const double* x, const double* t, double* out,
                         double* w) const
    requires detail_uam::has_prox_y<D>
  {
    (void)w;
    out[0] = law(x[0], st.theta).prox_y(v[0], t[0]);
  }

 private:
  // the prototype with (mu, theta) written over its first P doubles
  LAPLACE_HD D law(double mu, const double* theta) const {
    double p[n_params];
    p[0] = mu;
    for (int j = 1; j < n_params; ++j) p[j] = theta[j - 1];
    D d = prototype;
    std::memcpy(static_cast<void*>(&d), p, sizeof p);
    return d;
  }
  // the component (0^A, idx + 1) of a tensor over D's parameters
  template <int K, int A, class T>
  LAPLACE_HD static double component(const T& t, const int* idx) {
    if constexpr (K + A == 0) return (void)idx, t[0];
    else {
      int tuple[K + A];
      for (int i = 0; i < A; ++i) tuple[i] = 0;
      for (int i = 0; i < K; ++i) tuple[A + i] = idx[i] + 1;
      return t[sym_tensor_position<n_params, K + A>(tuple)];
    }
  }
};

}  // namespace laplace::distrib
