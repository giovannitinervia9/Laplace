// A distribution built from another one by putting mass on points of its own
// choosing (laplace::distrib). The plan is note/piano_wrapper.md section 5bis.
//
// Read it as a draw over K = N + 1 cells, the N atoms and the parent:
//
//   1. pick a cell with probability pi_0, ..., pi_{N-1}, pi_N on the open
//      simplex, the last cell being the parent;
//   2. in cell j < N the observation IS point[j]; in cell N it is drawn from
//      the parent, renormalized away from the adjusted atoms only.
//
// Adjusted and inflated differ in one thing and nothing else: whether the
// parent is renormalized off that point. An adjusted atom takes the parent's
// mass there away, an inflated one leaves it, so the parent's own mass adds to
// the cell's. With R = 1 - sum over the adjusted points of p(c_k; theta),
//
//   adjusted:  q_j = pi_j
//   inflated:  q_j = pi_j + pi_N p(c_j; theta) / R
//   elsewhere: the density is pi_N f(y; theta) / R,
//
// so the mode is one bit per atom and it lives in the type, being part of the
// model and never changing during a fit. R is identically one for a continuous
// parent, which has no mass to remove.
//
// The masses are a point of a simplex and not of a box, so they are not
// declared as parameters: the parameters are the free coordinates of the
// additive log-ratio map (bijector/alr.hpp), the parent's cell being the
// reference,
//
//   atom_alr[j] = log( P(Y = point[j]) / P(Y is not an atom) ),
//
// each on the whole line with an identity link, every point of R^N giving a
// valid distribution. They follow the parent's own parameters in theta.
//
// What this file carries today is the adjusted case over a continuous parent,
// which is the one whose derivative tensor is block diagonal: away from the
// atoms l(y) = log pi_N(u) + log f(y; theta) is a sum of two terms, each in one
// block, so every mixed component vanishes identically, and at an atom
// l = log pi_j(u) alone, where no component with a theta index survives. The
// inflated case and the discrete parent, where R depends on theta and an
// inflated atom is a two-term mixture, are step 4 of the plan.
#pragma once

#include <laplace/bijector/alr.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/math/elementary.hpp>

#include <array>
#include <vector>

namespace laplace::distrib {

/**
 * \ingroup distrib_atom_mixture
 * \brief Whether an atom replaces the parent's mass at its point or adds to it.
 */
enum class atom_mode : int {
  adjusted,  ///< The parent is renormalized off the point, so the atom's mass replaces the parent's.
  inflated   ///< The parent keeps its mass at the point, so the atom's mass adds to it.
};

/// \ingroup distrib_atom_mixture
/// \brief Shorthand for atom_mode::adjusted in a template argument list.
inline constexpr atom_mode adjusted = atom_mode::adjusted;
/// \ingroup distrib_atom_mixture
/// \brief Shorthand for atom_mode::inflated in a template argument list.
inline constexpr atom_mode inflated = atom_mode::inflated;

namespace detail_atom_mixture {

// The scratch `alr_bijector::inverse_dk` asks for, as a compile-time bound:
// its own scratch_size(), which is (2 + max_order) * max_masks doubles.
inline constexpr int alr_scratch = (2 + bijector::detail_coupled::max_order) * bijector::detail_coupled::max_masks;

// The names of the free coordinates, one per atom. A descriptor's name is a
// const char*, so they are a table and not a formatted string.
inline constexpr const char* alr_names[6] = {"atom_alr1", "atom_alr2", "atom_alr3",
                                             "atom_alr4", "atom_alr5", "atom_alr6"};

// The descriptors of the whole parameter vector: the parent's first, in their
// order, then one per atom. Appending keeps the parent's tensor layout as a
// leading block.
template <class D, int N>
constexpr std::array<distrib_param_info, D::n_params + N> params_of() {
  std::array<distrib_param_info, D::n_params + N> p{};
  for (int j = 0; j < D::n_params; ++j) p[j] = D::params[j];
  for (int j = 0; j < N; ++j)
    p[D::n_params + j] = {alr_names[j],      -distrib_infinity, distrib_infinity, false,
                          infinitely_smooth, false,             infinitely_smooth};
  return p;
}

// d^K log pi_c / du^K along the sorted tuple idx.
//
// With l = log(1 + sum_b e^{u_b}) the log-sum-exp of the free coordinates,
// log pi_c = u_c - l for an atom and log pi_N = -l for the parent's cell, so
// every derivative of order two and above is -d^K l, the same for every cell,
// and only the first order carries the Kronecker delta. Since dl/du_a = pi_a,
//
//   d^K l / du_{a_1} ... du_{a_K} = d^{K-1} pi_{a_1} / du_{a_2} ... du_{a_K},
//
// which is `alr_bijector::inverse_dk<K-1>` read at component a_1. The scratch
// is the bijector's own (512 doubles); on a device that is 4 KB of thread
// stack, which is measured rather than assumed before this runs in a kernel.
template <int K>
LAPLACE_HD double log_cell_dk(const bijector::alr_bijector& b, const double* u, const double* pi, int c,
                              const int (&idx)[K], double* w) {
  if constexpr (K == 1) {
    return (c == idx[0] ? 1.0 : 0.0) - pi[idx[0]];
  } else {
    int rest[K - 1];
    for (int j = 1; j < K; ++j) rest[j - 1] = idx[j];
    double out[8];
    b.template inverse_dk<K - 1>(u, rest, out, w);
    return -out[idx[0]];
  }
}

// d^S pi_c / du^S along the sorted tuple idx, S = 0 giving pi_c itself. The
// order is a run-time number here because the split between the two blocks of
// a component is, so it dispatches over the orders the bijector carries.
LAPLACE_HD double pi_dk(const bijector::alr_bijector& b, const double* u, const double* pi, int c, int s,
                               const int* idx, double* w) {
  if (s == 0) return pi[c];
  double out[8];
  switch (s) {
    case 1: { const int i[1] = {idx[0]}; b.inverse_dk<1>(u, i, out, w); break; }
    case 2: { const int i[2] = {idx[0], idx[1]}; b.inverse_dk<2>(u, i, out, w); break; }
    case 3: { const int i[3] = {idx[0], idx[1], idx[2]}; b.inverse_dk<3>(u, i, out, w); break; }
    case 4: { const int i[4] = {idx[0], idx[1], idx[2], idx[3]}; b.inverse_dk<4>(u, i, out, w); break; }
    case 5: { const int i[5] = {idx[0], idx[1], idx[2], idx[3], idx[4]}; b.inverse_dk<5>(u, i, out, w); break; }
    default: { const int i[6] = {idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]}; b.inverse_dk<6>(u, i, out, w); break; }
  }
  return out[c];
}

// d^S of the categorical information I_ab = pi_a (delta_ab - pi_b) in the free
// coordinates. Since I_ab is the second derivative of the log-sum-exp,
// d^S I_ab = d^{S+1} pi_a / du_b du_{idx}.
LAPLACE_HD double categorical_info_dk(const bijector::alr_bijector& b, const double* u, const double* pi,
                                             int a, int bb, int s, const int* idx, double* w) {
  int t[7];
  int n = 0;
  t[n++] = bb;
  for (int j = 0; j < s; ++j) t[n++] = idx[j];
  for (int i = 1; i < n; ++i) {  // insertion sort, n at most seven
    const int v = t[i];
    int k = i - 1;
    while (k >= 0 && t[k] > v) { t[k + 1] = t[k]; --k; }
    t[k + 1] = v;
  }
  return pi_dk(b, u, pi, a, n, t, w);
}

// The parent's log F (upper = false) or log(1 - F) derivative tensors for every
// order one to K, laid out one after the other, so that a sub-multiset of a
// component's parent indices can be read without recomputing a tensor.
template <int Q, int K>
struct flat_tensors {
  LAPLACE_HD static constexpr int offset(int t) {
    int o = 0;
    for (int j = 1; j < t; ++j) o += sym_tensor_size(Q, j);
    return o;
  }
  static constexpr int total = offset(K) + sym_tensor_size(Q, K);
  double v[total];

  // the same components written from outside, for a quantity this file builds
  LAPLACE_HD void set(int t, int pos, double value) { v[offset(t) + pos] = value; }

  // the parent's own log value (log density or log mass) at a point, every order
  template <class D, int T>
  LAPLACE_HD void fill_value_order(const D& p, double y) {
    const auto tt = [&] {
      if constexpr (D::kind == distrib_kind::continuous) {
        if constexpr (T == 1) return p.logpdf_d1(y);
        else if constexpr (T == 2) return p.logpdf_d2(y);
        else if constexpr (T == 3) return p.logpdf_d3(y);
        else if constexpr (T == 4) return p.logpdf_d4(y);
        else if constexpr (T == 5) return p.logpdf_d5(y);
        else return p.logpdf_d6(y);
      } else {
        if constexpr (T == 1) return p.logpmf_d1(y);
        else if constexpr (T == 2) return p.logpmf_d2(y);
        else if constexpr (T == 3) return p.logpmf_d3(y);
        else if constexpr (T == 4) return p.logpmf_d4(y);
        else if constexpr (T == 5) return p.logpmf_d5(y);
        else return p.logpmf_d6(y);
      }
    }();
    for (int i = 0; i < tt.size; ++i) v[offset(T) + i] = tt[i];
    if constexpr (T < K) fill_value_order<D, T + 1>(p, y);
  }

  template <class D>
  LAPLACE_HD void fill_value(const D& p, double y) {
    fill_value_order<D, 1>(p, y);
  }

  template <class D, int T>
  LAPLACE_HD void fill_order(const D& p, double y, bool upper) {
    const auto tt = [&] {
      if constexpr (T == 1) return upper ? p.log_ccdf_d1(y) : p.log_cdf_d1(y);
      else if constexpr (T == 2) return upper ? p.log_ccdf_d2(y) : p.log_cdf_d2(y);
      else if constexpr (T == 3) return upper ? p.log_ccdf_d3(y) : p.log_cdf_d3(y);
      else if constexpr (T == 4) return upper ? p.log_ccdf_d4(y) : p.log_cdf_d4(y);
      else if constexpr (T == 5) return upper ? p.log_ccdf_d5(y) : p.log_cdf_d5(y);
      else return upper ? p.log_ccdf_d6(y) : p.log_cdf_d6(y);
    }();
    for (int i = 0; i < tt.size; ++i) v[offset(T) + i] = tt[i];
    if constexpr (T < K) fill_order<D, T + 1>(p, y, upper);
  }

  template <class D>
  LAPLACE_HD void fill(const D& p, double y, bool upper) {
    fill_order<D, 1>(p, y, upper);
  }

  // the component of order t at the sorted multi-index idx
  LAPLACE_HD double at(int t, const int* idx) const {
    int pos = 0, prev = 0;
    for (int j = 0; j < t; ++j) {
      for (int w = prev; w < idx[j]; ++w) pos += binomial_coef(Q - w + t - j - 2, t - j - 1);
      prev = idx[j];
    }
    return v[offset(t) + pos];
  }
};

// y_regularity belongs to a continuous distribution alone, so the wrapper
// inherits it instead of declaring it: a discrete parent has none, and nvcc
// instantiates every static member of a class eagerly, where gcc waits for a
// use, so declaring it unconditionally is an error on one compiler and not the
// other.
template <class D, distrib_kind K>
struct observation_regularity {};

template <class D>
struct observation_regularity<D, distrib_kind::continuous> {
  /// Differentiability order of \f$\log f\f$ in the observation, away from the atoms.
  static constexpr int y_regularity = D::y_regularity;
};

// log of a sum of n terms given on the log scale, the largest factored out
LAPLACE_HD double log_sum_exp(const double* l, int n) {
  double m = -distrib_infinity;
  for (int i = 0; i < n; ++i) m = l[i] > m ? l[i] : m;
  if (!(m > -distrib_infinity)) return -distrib_infinity;
  if (m == distrib_infinity) return distrib_infinity;
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += math::exp(l[i] - m);
  return m + math::log(s);
}

}  // namespace detail_atom_mixture

/**
 * \ingroup distrib_atom_mixture
 * \brief A parent distribution with mass added on points of its own choosing.
 *
 * \tparam D Parent distribution type.
 * \tparam M One atom_mode per atom; their number is the number of atoms.
 */
template <class D, atom_mode... M>
struct atom_mixture_distrib : detail_atom_mixture::observation_regularity<D, D::kind> {
  /// Number of atoms the wrapper places.
  static constexpr int n_atom_points = static_cast<int>(sizeof...(M));
  static_assert(n_atom_points >= 1 && n_atom_points <= 6,
                "atom_mixture_distrib carries one to six atoms");
  /// Whether any atom adds to the parent's mass instead of replacing it.
  static constexpr bool any_inflated = ((M == atom_mode::inflated) || ...);
  // A declared atom has to be a panel boundary, or a quadrature node could land
  // on it and read a mass as if it were a density, and the only way to say so
  // is y_kinks(), which carries two points. So a continuous parent takes two
  // atoms at most; a discrete one declares none and may take six.
  static_assert(D::kind == distrib_kind::discrete || n_atom_points <= 2,
                "a continuous parent takes at most two atoms, which is what y_kinks() can report so that a "
                "quadrature panel splits at each of them");
  static_assert(D::kind == distrib_kind::discrete || !any_inflated,
                "a continuous parent has no mass at a point to inflate, and inflating one is the same model as "
                "adjusting it: use atom_mode::adjusted, or the factory zero_adjusted");
  // The parent is renormalized off the adjusted points only, so the constant R
  // below is one whenever the parent puts no mass on them, which is every
  // continuous parent. Where it is one the two blocks of the parameters never
  // meet and the closed routes of this file apply; where it is not, the
  // expected information and the cumulative functions are left to the generic
  // routes, which read the derivatives this file supplies.
  /// Whether the parent's cell needs no renormalization, which is what keeps the two blocks apart.
  static constexpr bool unit_renormalization = D::kind == distrib_kind::continuous;
  /// Whether the closed cumulative, information and quantile routes of this file apply.
  static constexpr bool closed_routes = unit_renormalization && !any_inflated;

  /// The wrapped distribution.
  D parent;
  /// Free coordinates of the additive log-ratio map, one per atom, on the whole line.
  double atom_alr[n_atom_points];
  /// Where each atom sits; fixed, and independent of the parameters.
  double point[n_atom_points];

  /// Distribution kind, the parent's.
  static constexpr distrib_kind kind = D::kind;
  /// Number of natural parameters: the parent's, then one free coordinate per atom.
  static constexpr int n_params = D::n_params + n_atom_points;
  /// Number of atoms carrying mass beyond what \c kind implies: none when the parent is discrete, its
  /// support points already being mass.
  static constexpr int n_atoms = D::kind == distrib_kind::continuous ? n_atom_points : 0;

 private:
  static constexpr std::array<distrib_param_info, n_params> params_ =
      detail_atom_mixture::params_of<D, n_atom_points>();

 public:
  /// Descriptors of the parameters, the parent's followed by the free coordinates.
  static constexpr const distrib_param_info* params = params_.data();


  /**
   * \brief Returns the additive log-ratio map over the atoms and the parent's cell.
   * \return An \c alr_bijector of \c n_atom_points + 1 cells.
   */
  LAPLACE_HD bijector::alr_bijector cell_map() const { return bijector::alr_bijector(n_atom_points + 1); }

  /**
   * \brief Writes the log masses of the atoms and of the parent's cell.
   * \param lpi Output buffer of \c n_atom_points + 1 doubles, the parent's cell last.
   */
  LAPLACE_HD void log_cells(double* lpi) const {
    double w[1];
    cell_map().inverse_log(atom_alr, lpi, w);
  }

  /**
   * \brief Position of atom \f$i\f$.
   * \param i Atom index.
   * \return \c point[i].
   */
  LAPLACE_HD double atom_point(int i) const { return point[i]; }

  /**
   * \brief Log mass the law puts on atom \f$i\f$.
   * \param i Atom index.
   * \return \f$\log \pi_i\f$, the parent having no mass at the point.
   */
  LAPLACE_HD double atom_log_mass(int i) const {
    double lpi[7];
    log_cells(lpi);
    return lpi[i];
  }

  /**
   * \brief Index of the atom at \f$y\f$, or \f$-1\f$ when \f$y\f$ is not one.
   * \param y Observation point.
   * \return The atom's index, or \f$-1\f$.
   */
  LAPLACE_HD int atom_at(double y) const {
    for (int i = 0; i < n_atom_points; ++i)
      if (y == point[i]) return i;
    return -1;
  }

  /**
   * \brief The parent's own log density or log mass, by its kind.
   * \param y Observation point.
   * \return \c parent.logpdf(y) for a continuous parent, \c parent.logpmf(y) for a discrete one.
   */
  LAPLACE_HD double parent_log_value(double y) const {
    if constexpr (D::kind == distrib_kind::continuous) return parent.logpdf(y);
    else return parent.logpmf(y);
  }

  /**
   * \brief Evaluates \f$\log R\f$, the mass the parent keeps once the adjusted atoms are taken from it.
   * \return Zero whenever the parent puts no mass on an adjusted point, which is every continuous parent.
   */
  LAPLACE_HD double log_renorm() const {
    if constexpr (unit_renormalization) {
      return 0.0;
    } else {
      double s = 0.0;
      constexpr bool md[] = {(M == atom_mode::inflated)...};
      for (int i = 0; i < n_atom_points; ++i)
        if (!md[i]) s += math::exp(parent_log_value(point[i]));
      return Kokkos::log1p(-s);
    }
  }

  /**
   * \brief Evaluates the log-density, or the log mass where the law has one.
   * \param y Observation point.
   * \return The log mass at an atom and \f$\log \pi_N + \log f(y) - \log R\f$ elsewhere.
   */
  LAPLACE_HD double log_value(double y) const {
    double lpi[7];
    log_cells(lpi);
    const int i = atom_at(y);
    const double lr = log_renorm();
    if (i >= 0) {
      if constexpr (!any_inflated) {
        return lpi[i];
      } else {
        constexpr bool md[] = {(M == atom_mode::inflated)...};
        if (!md[i]) return lpi[i];
        const double a = lpi[i], b = lpi[n_atom_points] + parent_log_value(point[i]) - lr;
        const double term[2] = {a, b};
        return detail_atom_mixture::log_sum_exp(term, 2);
      }
    }
    return lpi[n_atom_points] + parent_log_value(y) - lr;
  }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of the log-density in the parameters.
   *
   * Away from the atoms the log-density is a sum of one term in the parent's parameters and one in
   * the free coordinates, so every mixed component is zero; at an atom only the free coordinates
   * appear.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_value_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    constexpr int P = n_params, Q = D::n_params;
    sym_tensor<P, K> t;
    const int at = atom_at(y);
    constexpr bool md[] = {(M == atom_mode::inflated)...};
    if constexpr (any_inflated) {
      // an inflated atom is the one case that is a sum rather than a sum of
      // logarithms: the cell's own mass and the parent's share of the point
      if (at >= 0 && md[at]) {
        bool in_sum[6] = {};
        in_sum[at] = true;
        detail_atom_mixture::flat_tensors<Q, K> gt;
        gt.fill_value(parent, point[at]);
        if constexpr (!unit_renormalization) {
          detail_atom_mixture::flat_tensors<Q, K> rt;
          fill_log_renorm<K>(rt);
          for (int tt = 1; tt <= K; ++tt)
            for (int i = 0; i < sym_tensor_size(Q, tt); ++i)
              gt.set(tt, i, gt.v[gt.offset(tt) + i] - rt.v[rt.offset(tt) + i]);
        }
        return log_two_part_dk<K>(in_sum, parent_log_value(point[at]) - log_renorm(), gt);
      }
    }
    const auto b = cell_map();
    double pi[7];
    {
      double w[1];
      b.inverse(atom_alr, pi, w);
    }
    const int cell = at >= 0 ? at : n_atom_points;
    double w[detail_atom_mixture::alr_scratch];
    // the parent's tensor, once: away from the atoms it is the whole block in
    // the parent's own parameters, less the renormalization, and at an atom no
    // such component survives
    sym_tensor<Q, K> pt;
    if (at < 0) {
      if constexpr (D::kind == distrib_kind::continuous) pt = parent.template logpdf_dk<K>(y);
      else pt = parent.template logpmf_dk<K>(y);
      if constexpr (!unit_renormalization) {
        detail_atom_mixture::flat_tensors<Q, K> rt;
        fill_log_renorm<K>(rt);
        for (int i = 0; i < sym_tensor<Q, K>::size; ++i) pt[i] -= rt.v[rt.offset(K) + i];
      }
    }
    int idx[K];
    for (int j = 0; j < K; ++j) idx[j] = 0;
    for (;;) {
      if (idx[0] >= Q) {  // every index a free coordinate
        int u[K];
        for (int j = 0; j < K; ++j) u[j] = idx[j] - Q;
        t[sym_tensor_position<P, K>(idx)] = detail_atom_mixture::log_cell_dk<K>(b, atom_alr, pi, cell, u, w);
      } else if (at < 0 && idx[K - 1] < Q) {  // every index a parent parameter
        t[sym_tensor_position<P, K>(idx)] = pt[sym_tensor_position<Q, K>(idx)];
      }  // mixed, or a parent index at an atom: zero
      int j = K - 1;
      while (j >= 0 && idx[j] == P - 1) --j;
      if (j < 0) break;
      const int v = idx[j] + 1;
      for (int m = j; m < K; ++m) idx[m] = v;
    }
    return t;
  }

  /// \brief Evaluates the first derivatives of the log-density. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> logpdf_d1(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<1>(y); }
  /// \brief Derivatives of order 1 of the log-mass. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> logpmf_d1(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<1>(y); }
  /// \brief Evaluates the second derivatives of the log-density. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> logpdf_d2(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<2>(y); }
  /// \brief Derivatives of order 2 of the log-mass. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> logpmf_d2(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<2>(y); }
  /// \brief Evaluates the third derivatives of the log-density. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> logpdf_d3(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<3>(y); }
  /// \brief Derivatives of order 3 of the log-mass. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> logpmf_d3(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<3>(y); }
  /// \brief Evaluates the fourth derivatives of the log-density. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> logpdf_d4(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<4>(y); }
  /// \brief Derivatives of order 4 of the log-mass. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> logpmf_d4(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<4>(y); }
  /// \brief Evaluates the fifth derivatives of the log-density. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> logpdf_d5(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<5>(y); }
  /// \brief Derivatives of order 5 of the log-mass. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> logpmf_d5(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<5>(y); }
  /// \brief Evaluates the sixth derivatives of the log-density. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> logpdf_d6(double y) const requires(D::kind == distrib_kind::continuous) { return log_value_dk<6>(y); }
  /// \brief Derivatives of order 6 of the log-mass. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> logpmf_d6(double y) const requires(D::kind == distrib_kind::discrete) { return log_value_dk<6>(y); }

  /**
   * \brief Evaluates the expected Fisher information.
   *
   * The two blocks separate: the information in the parent's parameters is the parent's own scaled by
   * the probability of its cell, since its score is zero at an atom; the information in the free
   * coordinates is the categorical one, \f$\mathrm{diag}(\pi) - \pi\pi^\top\f$, the same whatever
   * cell the observation falls in; and the block between them is zero, the log-density being a sum.
   * \return The packed symmetric matrix of order 2.
   */
  LAPLACE_HD sym_tensor<n_params, 2> expected_info() const requires(closed_routes) {
    constexpr int P = n_params, Q = D::n_params;
    sym_tensor<P, 2> t;
    double pi[7];
    {
      double w[1];
      cell_map().inverse(atom_alr, pi, w);
    }
    const auto ip = parent.expected_info();
    for (int a = 0; a < Q; ++a)
      for (int b = a; b < Q; ++b) {
        const int idx[2] = {a, b}, pidx[2] = {a, b};
        t[sym_tensor_position<P, 2>(idx)] = pi[n_atom_points] * ip[sym_tensor_position<Q, 2>(pidx)];
      }
    for (int a = 0; a < n_atom_points; ++a)
      for (int b = a; b < n_atom_points; ++b) {
        const int idx[2] = {Q + a, Q + b};
        t[sym_tensor_position<P, 2>(idx)] = pi[a] * ((a == b ? 1.0 : 0.0) - pi[b]);
      }
    return t;
  }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of the expected Fisher information.
   *
   * Each parameter appears in one factor only, so a component with some indices in each block is the
   * plain product of the two factors' derivatives, with no Leibniz sum.
   * \tparam K Derivative order, \f$1 \le K \le 4\f$.
   * \return The derivatives, one symmetric tensor per parameter pair.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<n_params, K> expected_info_dk() const requires(closed_routes) {
    static_assert(K >= 1 && K <= 4, "orders one to four");
    constexpr int P = n_params, Q = D::n_params;
    expected_info_tensor<P, K> t;
    const auto b = cell_map();
    double pi[7];
    {
      double w[1];
      b.inverse(atom_alr, pi, w);
    }
    double w[detail_atom_mixture::alr_scratch];
    // A component's indices split into the parent's and the free ones, and
    // each variable sits in one factor only, so the two parts are enumerated
    // apart and multiplied: no Leibniz sum, and no component with indices in
    // both blocks of the information, that block being identically zero.
    fill_info<K, 0>(t, b, pi, w);
    return t;
  }

  /// \brief Evaluates the first derivatives of the expected information. \return Order-1 derivatives.
  LAPLACE_HD expected_info_tensor<n_params, 1> expected_info_d1() const requires(closed_routes) { return expected_info_dk<1>(); }
  /// \brief Evaluates the second derivatives of the expected information. \return Order-2 derivatives.
  LAPLACE_HD expected_info_tensor<n_params, 2> expected_info_d2() const requires(closed_routes) { return expected_info_dk<2>(); }
  /// \brief Evaluates the third derivatives of the expected information. \return Order-3 derivatives.
  LAPLACE_HD expected_info_tensor<n_params, 3> expected_info_d3() const requires(closed_routes) { return expected_info_dk<3>(); }
  /// \brief Evaluates the fourth derivatives of the expected information. \return Order-4 derivatives.
  LAPLACE_HD expected_info_tensor<n_params, 4> expected_info_d4() const requires(closed_routes) { return expected_info_dk<4>(); }

  /**
   * \brief Evaluates \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$.
   *
   * The side asked for is the mass of the atoms that fall on it plus the parent's cell times the
   * parent's own side, summed on the log scale.
   * \param y Observation point.
   * \param upper \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The value.
   */
  LAPLACE_HD double log_cdf_tail(double y, bool upper) const requires(closed_routes) {
    double lpi[7], term[8];
    log_cells(lpi);
    int n = 0;
    for (int i = 0; i < n_atom_points; ++i)
      if (upper ? point[i] > y : point[i] <= y) term[n++] = lpi[i];
    term[n++] = lpi[n_atom_points] + (upper ? parent.log_ccdf(y) : parent.log_cdf(y));
    return detail_atom_mixture::log_sum_exp(term, n);
  }

  /// \brief Evaluates \f$\log F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double log_cdf(double y) const requires(closed_routes) { return log_cdf_tail(y, false); }
  /// \brief Evaluates \f$\log(1 - F(y))\f$. \param y Observation point. \return The value.
  LAPLACE_HD double log_ccdf(double y) const requires(closed_routes) { return log_cdf_tail(y, true); }
  /// \brief Evaluates \f$F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double cdf(double y) const requires(closed_routes) { return math::exp(log_cdf(y)); }
  /// \brief Evaluates \f$1 - F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double ccdf(double y) const requires(closed_routes) { return math::exp(log_ccdf(y)); }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of \f$\log F\f$ or of \f$\log(1 - F)\f$.
   *
   * The side asked for is \f$A(u) + \pi_N(u) F_p(y; \theta)\f$ with \f$A\f$ the mass of the atoms on
   * it, so each variable sits in one factor and the derivative of the sum over a sub-multiset is a
   * plain product. The components of \f$\log\f$ of that sum follow by the Bell relation of
   * \ref distrib_generic, read in the ratios \f$\partial_T F / F\f$.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \param upper \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_cdf_tail_dk(double y, bool upper) const requires(closed_routes) {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    static_assert(unit_renormalization && !any_inflated,
                  "the closed cumulative route is for a parent that keeps its whole mass; otherwise the "
                  "generic routes of generic_cdf.hpp read the derivatives this file supplies");
    bool on_side[6];
    for (int i = 0; i < n_atom_points; ++i) on_side[i] = upper ? point[i] > y : point[i] <= y;
    detail_atom_mixture::flat_tensors<D::n_params, K> pct;
    pct.fill(parent, y, upper);
    return log_two_part_dk<K>(on_side, upper ? parent.log_ccdf(y) : parent.log_cdf(y), pct);
  }

  /**
   * \brief Evaluates the derivatives of \f$\log(A + \pi_N g)\f$, with \f$A\f$ a sum of cell masses.
   *
   * Both parts of the sum are a product of a function of the free coordinates and one of the parent's
   * parameters, so the derivative over a sub-multiset is a plain product, and the components of the
   * logarithm follow from the Bell relation read in the ratios \f$\partial_T / \cdot\f$.
   * \tparam K Derivative order.
   * \param in_sum Which atoms contribute their mass to \f$A\f$.
   * \param lg \f$\log g\f$, the factor in the parent's parameters.
   * \param gt The derivatives of \f$\log g\f$, orders one to \f$K\f$.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_two_part_dk(const bool* in_sum, double lg,
                                                     const detail_atom_mixture::flat_tensors<D::n_params, K>& gt)
      const {
    constexpr int P = n_params, Q = D::n_params, full = (1 << K) - 1;
    sym_tensor<P, K> t;
    const auto b = cell_map();
    double pi[7], lpi[7];
    {
      double w0[1];
      b.inverse(atom_alr, pi, w0);
      b.inverse_log(atom_alr, lpi, w0);
    }
    double term[8];
    int nt = 0;
    for (int i = 0; i < n_atom_points; ++i)
      if (in_sum[i]) term[nt++] = lpi[i];
    term[nt++] = lpi[n_atom_points] + lg;
    const double lF = detail_atom_mixture::log_sum_exp(term, nt);
    if (!(lF > -distrib_infinity)) {  // the quantity is zero
      for (int i = 0; i < sym_tensor<P, K>::size; ++i) t[i] = math::detail::quiet_nan();
      return t;
    }
    if (lF == 0.0) return t;  // one, every derivative zero
    const double g = math::exp(lpi[n_atom_points] + lg - lF);
    const auto& pct = gt;
    const bool* on_side = in_sum;
    double w[detail_atom_mixture::alr_scratch];
    const double F = math::exp(lF);
    int c[K];
    for (int j = 0; j < K; ++j) c[j] = 0;
    for (int comp = 0; comp < sym_tensor<P, K>::size; ++comp) {
      // the parent's ratios over the masks that pick only its own slots
      double up[1 << K], rp[1 << K], r[1 << K], uu[1 << K];
      for (int mask = 0; mask <= full; ++mask) {
        int idx[K];
        int nn = 0;
        bool only_parent = true;
        for (int j = 0; j < K; ++j)
          if (mask >> j & 1) {
            if (c[j] >= Q) only_parent = false;
            else idx[nn++] = c[j];
          }
        up[mask] = (mask != 0 && only_parent) ? pct.at(nn, idx) : 0.0;
      }
      detail_generic::ratios_from_log_tail_dk<K>(up, rp);
      for (int mask = 0; mask <= full; ++mask) {
        int uidx[K];
        int nu = 0;
        int pmask = 0;
        for (int j = 0; j < K; ++j)
          if (mask >> j & 1) {
            if (c[j] >= Q) uidx[nu++] = c[j] - Q;
            else pmask |= 1 << j;
          }
        // d_U pi_N / pi_N times the parent's ratio, plus the atoms when no
        // parent index is taken
        const double dpiN = detail_atom_mixture::pi_dk(b, atom_alr, pi, n_atom_points, nu, uidx, w);
        double v = (dpiN / pi[n_atom_points]) * g * rp[pmask];
        if (pmask == 0) {
          double a = 0.0;
          for (int i = 0; i < n_atom_points; ++i)
            if (on_side[i]) a += detail_atom_mixture::pi_dk(b, atom_alr, pi, i, nu, uidx, w);
          v += a / F;
        }
        r[mask] = v;
      }
      detail_generic::log_tail_dk_from_ratios<K>(r, uu);
      t[comp] = uu[full];
      // the next sorted multi-index
      int j = K - 1;
      while (j >= 0 && c[j] == P - 1) --j;
      if (j < 0) break;
      const int nv = c[j] + 1;
      for (int m = j; m < K; ++m) c[m] = nv;
    }
    return t;
  }

  /// \brief First derivatives of \f$\log F\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_cdf_d1(double y) const requires(closed_routes) { return log_cdf_tail_dk<1>(y, false); }
  /// \brief Second derivatives of \f$\log F\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_cdf_d2(double y) const requires(closed_routes) { return log_cdf_tail_dk<2>(y, false); }
  /// \brief Third derivatives of \f$\log F\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_cdf_d3(double y) const requires(closed_routes) { return log_cdf_tail_dk<3>(y, false); }
  /// \brief Fourth derivatives of \f$\log F\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_cdf_d4(double y) const requires(closed_routes) { return log_cdf_tail_dk<4>(y, false); }
  /// \brief Fifth derivatives of \f$\log F\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_cdf_d5(double y) const requires(closed_routes) { return log_cdf_tail_dk<5>(y, false); }
  /// \brief Sixth derivatives of \f$\log F\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_cdf_d6(double y) const requires(closed_routes) { return log_cdf_tail_dk<6>(y, false); }
  /// \brief First derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_ccdf_d1(double y) const requires(closed_routes) { return log_cdf_tail_dk<1>(y, true); }
  /// \brief Second derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_ccdf_d2(double y) const requires(closed_routes) { return log_cdf_tail_dk<2>(y, true); }
  /// \brief Third derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_ccdf_d3(double y) const requires(closed_routes) { return log_cdf_tail_dk<3>(y, true); }
  /// \brief Fourth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_ccdf_d4(double y) const requires(closed_routes) { return log_cdf_tail_dk<4>(y, true); }
  /// \brief Fifth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_ccdf_d5(double y) const requires(closed_routes) { return log_cdf_tail_dk<5>(y, true); }
  /// \brief Sixth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_ccdf_d6(double y) const requires(closed_routes) { return log_cdf_tail_dk<6>(y, true); }

  /**
   * \brief Evaluates the quantile \f$F^{-1}(p)\f$, which is an atom where \f$F\f$ jumps over \f$p\f$.
   *
   * Walks the atoms in the order of their points: where \f$p\f$ falls below the next jump the answer
   * comes from the parent's own quantile at the probability left to its cell, and where it falls
   * inside a jump the answer is the atom.
   * \param p Probability in \f$(0, 1)\f$.
   * \return The quantile.
   */
  LAPLACE_HD double quantile(double p) const requires(closed_routes) {
    if (!(p > 0.0)) return p == 0.0 ? support_lower() : math::detail::quiet_nan();
    if (!(p < 1.0)) return p == 1.0 ? support_upper() : math::detail::quiet_nan();
    double pi[7];
    {
      double w[1];
      cell_map().inverse(atom_alr, pi, w);
    }
    int ord[6];
    for (int i = 0; i < n_atom_points; ++i) ord[i] = i;
    for (int i = 1; i < n_atom_points; ++i) {  // insertion sort by position, six at most
      const int v = ord[i];
      int k = i - 1;
      while (k >= 0 && point[ord[k]] > point[v]) { ord[k + 1] = ord[k]; --k; }
      ord[k + 1] = v;
    }
    double acc = 0.0;  // the mass of the atoms already passed
    for (int k = 0; k < n_atom_points; ++k) {
      const int i = ord[k];
      const double below = acc + pi[n_atom_points] * parent.cdf(point[i]);  // F just below the atom
      if (p <= below) return distrib_quantile(parent, (p - acc) / pi[n_atom_points]);
      if (p <= below + pi[i]) return point[i];
      acc += pi[i];
    }
    return distrib_quantile(parent, (p - acc) / pi[n_atom_points]);
  }

  /**
   * \brief Returns a starting value read from the sample itself.
   *
   * The cells start at the sample frequencies, an empty one at half an observation so that it starts
   * inside its domain, and the parent starts from the observations that are not at an atom. The
   * statistics are the atoms' counts, which no other distribution asks for, so this is the overload
   * of \ref distrib_start that takes the sample.
   * \param y The sample.
   * \return A distribution of the same shape with its parameters set.
   */
  atom_mixture_distrib start_value(const std::vector<double>& y) const {
    int cnt[6] = {};
    std::vector<double> rest;
    rest.reserve(y.size());
    for (double v : y) {
      const int i = atom_at(v);
      if (i >= 0) ++cnt[i];
      else rest.push_back(v);
    }
    const double n = y.empty() ? 1.0 : static_cast<double>(y.size());
    const double floor_p = 0.5 / n;  // half an observation, so that an empty cell starts inside
    atom_mixture_distrib out = *this;
    out.parent = distrib_start_value(parent, rest);
    double pc = static_cast<double>(rest.size()) / n;
    if (pc < floor_p) pc = floor_p;
    for (int i = 0; i < n_atom_points; ++i) {
      double q = static_cast<double>(cnt[i]) / n;
      if (q < floor_p) q = floor_p;
      out.atom_alr[i] = math::log(q / pc);
    }
    return out;
  }

  /// \brief Lower end of the support. \return The lesser of the parent's lower end and the atoms.
  LAPLACE_HD double support_lower() const {
    double lo = parent.support_lower();
    for (int i = 0; i < n_atom_points; ++i) lo = point[i] < lo ? point[i] : lo;
    return lo;
  }
  /// \brief Upper end of the support. \return The greater of the parent's upper end and the atoms.
  LAPLACE_HD double support_upper() const {
    double hi = parent.support_upper();
    for (int i = 0; i < n_atom_points; ++i) hi = point[i] > hi ? point[i] : hi;
    return hi;
  }
  /// \brief Rough location, for placing a quadrature. \return The parent's.
  LAPLACE_HD double center() const { return parent.center(); }
  /// \brief Rough spread, for placing a quadrature. \return The parent's.
  LAPLACE_HD double scale() const { return parent.scale(); }

  /**
   * \brief Non-smooth points of \f$\log f\f$ in \f$y\f$: the atoms, so that a quadrature panel splits there.
   * \return Up to two atom positions, nothing continuous in them.
   */
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    k.count = n_atom_points < 2 ? n_atom_points : 2;
    for (int i = 0; i < k.count; ++i) k.point[i] = point[i];
    k.continuous_order = 0;
    return k;
  }

  /**
   * \brief Non-smooth points of a parameter placed by the observation.
   * \param j Parameter index.
   * \param y Observation point.
   * \return The parent's for one of its parameters, none for a free coordinate.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    if (j < D::n_params) return parent.param_kinks(j, y);
    return distrib_kinks{};
  }

  /// \brief Evaluates the log-density. \param y Observation point. \return The value.
  LAPLACE_HD double logpdf(double y) const requires(D::kind == distrib_kind::continuous) { return log_value(y); }
  /// \brief Evaluates the log-mass. \param y Observation point. \return The value.
  LAPLACE_HD double logpmf(double y) const requires(D::kind == distrib_kind::discrete) { return log_value(y); }
  /// \brief Derivatives of order \f$K\f$ of the log-density. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dk(double y) const requires(D::kind == distrib_kind::continuous) {
    return log_value_dk<K>(y);
  }
  /// \brief Derivatives of order \f$K\f$ of the log-mass. \tparam K Order. \param y Point. \return The tensor.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpmf_dk(double y) const requires(D::kind == distrib_kind::discrete) {
    return log_value_dk<K>(y);
  }

 private:
  // The derivatives of log R to every order one to K, where R is the mass the
  // parent keeps once the adjusted points are taken from it. R = 1 - sum_k p_k
  // is a sum, so each component goes through the ratios: d_T R / R is minus the
  // sum of p_k times the parent's own ratios at that point, and the Bell
  // relation turns those into the derivatives of the logarithm.
  template <int K>
  LAPLACE_HD void fill_log_renorm(detail_atom_mixture::flat_tensors<D::n_params, K>& out) const {
    constexpr int Q = D::n_params;
    constexpr bool md[] = {(M == atom_mode::inflated)...};
    detail_atom_mixture::flat_tensors<Q, K> pv[6];
    double mass[6];
    int nadj = 0;
    int which[6];
    for (int i = 0; i < n_atom_points; ++i)
      if (!md[i]) {
        pv[nadj].fill_value(parent, point[i]);
        mass[nadj] = math::exp(parent_log_value(point[i]));
        which[nadj] = i;
        ++nadj;
      }
    double rsum = 0.0;
    for (int i = 0; i < nadj; ++i) rsum += mass[i];
    const double R = 1.0 - rsum;
    fill_log_renorm_order<K, 1>(out, pv, mass, nadj, R);
    (void)which;
  }

  template <int K, int T>
  LAPLACE_HD void fill_log_renorm_order(detail_atom_mixture::flat_tensors<D::n_params, K>& out,
                                        const detail_atom_mixture::flat_tensors<D::n_params, K>* pv,
                                        const double* mass, int nadj, double R) const {
    constexpr int Q = D::n_params, full = (1 << T) - 1;
    int c[T];
    for (int j = 0; j < T; ++j) c[j] = 0;
    for (int comp = 0; comp < sym_tensor_size(Q, T); ++comp) {
      double rr[1 << T], uu[1 << T], uk[1 << T], rk[1 << T];
      for (int mask = 0; mask <= full; ++mask) rr[mask] = 0.0;
      rr[0] = 1.0;
      for (int k = 0; k < nadj; ++k) {
        for (int mask = 0; mask <= full; ++mask) {
          int idx[T];
          int nn = 0;
          for (int j = 0; j < T; ++j)
            if (mask >> j & 1) idx[nn++] = c[j];
          uk[mask] = mask == 0 ? 0.0 : pv[k].at(nn, idx);
        }
        detail_generic::ratios_from_log_tail_dk<T>(uk, rk);
        for (int mask = 1; mask <= full; ++mask) rr[mask] -= mass[k] * rk[mask] / R;
      }
      detail_generic::log_tail_dk_from_ratios<T>(rr, uu);
      out.set(T, comp, uu[full]);
      int j = T - 1;
      while (j >= 0 && c[j] == Q - 1) --j;
      if (j < 0) break;
      const int v = c[j] + 1;
      for (int m = j; m < T; ++m) c[m] = v;
    }
    if constexpr (T < K) fill_log_renorm_order<K, T + 1>(out, pv, mass, nadj, R);
  }

  // The components of d^K I with exactly S indices among the free coordinates,
  // then the same for S + 1. S is a template parameter because the parent's
  // tensor of order K - S has a type of its own.
  template <int K, int S>
  LAPLACE_HD void fill_info(expected_info_tensor<n_params, K>& t, const bijector::alr_bijector& b, const double* pi,
                            double* w) const {
    constexpr int P = n_params, Q = D::n_params, KT = K - S;
    int uidx[K == 0 ? 1 : K], pidx[K == 0 ? 1 : K], idx[K == 0 ? 1 : K];
    for (int j = 0; j < K; ++j) { uidx[j] = 0; pidx[j] = 0; idx[j] = 0; }
    // the parent's factor at this order, I itself when no index is left to it
    const auto ip0 = parent.expected_info();
    [[maybe_unused]] auto ipk = [&] {
      if constexpr (KT >= 1) return parent.template expected_info_dk<KT>();
      else return 0;
    }();
    for (;;) {  // the free part
      const double dpi = detail_atom_mixture::pi_dk(b, atom_alr, pi, n_atom_points, S, uidx, w);
      for (int j = 0; j < S; ++j) idx[KT + j] = Q + uidx[j];
      for (int j = 0; j < KT; ++j) pidx[j] = 0;
      for (;;) {  // the parent's part
        for (int j = 0; j < KT; ++j) idx[j] = pidx[j];
        const int pos = sym_tensor_position<P, K>(idx);
        // the same indices in an array of the parent's own order, which the
        // packing helper takes by reference to a fixed size
        int pk[KT == 0 ? 1 : KT];
        for (int j = 0; j < KT; ++j) pk[j] = pidx[j];
        for (int a = 0; a < Q; ++a)
          for (int bb = a; bb < Q; ++bb) {
            const int pr[2] = {a, bb};
            double v;
            if constexpr (KT == 0) {
              v = dpi * ip0[sym_tensor_position<Q, 2>(pr)];
            } else {
              v = dpi * ipk.pair[sym_tensor_position<Q, 2>(pr)][sym_tensor_position<Q, KT>(pk)];
            }
            t.pair[sym_tensor_position<P, 2>(pr)][pos] = v;
          }
        if constexpr (KT == 0) {
          // the categorical block, which no parent parameter touches
          for (int a = 0; a < n_atom_points; ++a)
            for (int bb = a; bb < n_atom_points; ++bb) {
              const int pr[2] = {Q + a, Q + bb};
              t.pair[sym_tensor_position<P, 2>(pr)][pos] =
                  detail_atom_mixture::categorical_info_dk(b, atom_alr, pi, a, bb, S, uidx, w);
            }
          break;
        } else {
          int j = KT - 1;
          while (j >= 0 && pidx[j] == Q - 1) --j;
          if (j < 0) break;
          const int v = pidx[j] + 1;
          for (int m = j; m < KT; ++m) pidx[m] = v;
        }
      }
      if constexpr (S == 0) break;
      else {
        int j = S - 1;
        while (j >= 0 && uidx[j] == n_atom_points - 1) --j;
        if (j < 0) break;
        const int v = uidx[j] + 1;
        for (int m = j; m < S; ++m) uidx[m] = v;
      }
    }
    if constexpr (S < K) fill_info<K, S + 1>(t, b, pi, w);
  }
};

/**
 * \ingroup distrib_atom_mixture
 * \brief Builds a distribution with one adjusted atom at zero.
 *
 * The familiar zero-adjusted model: the mass at zero is free, and the parent carries the rest.
 * \tparam D Parent distribution type.
 * \param parent The wrapped distribution.
 * \param atom_alr Log ratio of the mass at zero to the parent's cell.
 * \return The wrapper.
 */
template <class D>
LAPLACE_HD atom_mixture_distrib<D, adjusted> zero_adjusted(const D& parent, double atom_alr) {
  return {.parent = parent, .atom_alr = {atom_alr}, .point = {0.0}};
}

/**
 * \ingroup distrib_atom_mixture
 * \brief Builds a distribution with adjusted atoms at zero and at one.
 *
 * The model the literature writes for a proportion that can be exactly zero or exactly one.
 * \tparam D Parent distribution type.
 * \param parent The wrapped distribution.
 * \param alr_zero Log ratio of the mass at zero to the parent's cell.
 * \param alr_one Log ratio of the mass at one to the parent's cell.
 * \return The wrapper.
 */
template <class D>
LAPLACE_HD atom_mixture_distrib<D, adjusted, adjusted> zero_one_adjusted(const D& parent, double alr_zero,
                                                                         double alr_one) {
  return {.parent = parent, .atom_alr = {alr_zero, alr_one}, .point = {0.0, 1.0}};
}

}  // namespace laplace::distrib
