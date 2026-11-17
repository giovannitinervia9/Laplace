// A distribution restricted to a fixed window and renormalized there
// (laplace::distrib). The plan is note/piano_wrapper.md section 4, and the
// composition rule its section 5ter.
//
// With fixed bounds lower <= upper, held as values and independent of the
// parameters theta,
//
//   l(y; theta) = log f(y; theta) - log Z(theta),   Z = F(upper) - F(lower-),
//
// on the window and minus infinity outside it. F(lower-) is the parent's
// cumulative function just below the lower bound, so the bound itself belongs
// to the window: a point mass the parent puts there is kept. The wrapper adds no
// parameter, and since the bounds do not move with theta the normalization is
// differentiated under the integral sign and no parameter moves the support.
//
// Z is taken from whichever tail of the parent holds the smaller numbers, so
// that neither cumulative value is the complement of a small one:
//
//   lower tail:  Z = F(upper) - F(lower-)
//   upper tail:  Z = S(lower-) - S(upper),        S = 1 - F,
//
// the first when F(upper) <= S(lower-). Either is the difference A - B of two
// numbers on the log scale, log Z = log A + log(1 - e^(log B - log A)), and its
// derivatives to order six come from the parent's own derivatives of log F or
// log S through the Bell relation of generic_cdf.hpp read in the ratios
// d_T G / G:
//
//   d_T (A - B) / (A - B) = (A / Z) d_T A / A - (B / Z) d_T B / B.
//
// The left limit reads the mass at the point from the object and never from its
// kind (note/piano_wrapper.md section 3.7): for a continuous parent it is the
// parent's declared atom at lower, if any, through distrib_atom_point and
// distrib_atom_log_mass; for a discrete parent, whose support is the integers,
// F(lower-) is F(lower - 1) itself, which is the same number without the
// subtraction F(lower) - p(lower).
#pragma once

#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace laplace::distrib {

template <class D>
struct truncated_distrib;

namespace detail_truncated {

template <class D>
struct is_truncated : std::false_type {};
template <class E>
struct is_truncated<truncated_distrib<E>> : std::true_type {};

// The cells of a law that carry a parameter of their own, as points: the atoms
// it declares, or, for an atom mixture over a discrete parent, which declares
// none because its points are support points, the points it weighs.
template <class D>
constexpr int cell_points() {
  if constexpr (requires { D::n_atom_points; }) return D::n_atom_points;
  else return distrib_n_atoms<D>();
}

// A quantity on the log scale with its derivatives in theta, every order one to
// K laid out one after the other. A zero quantity has zero derivatives, which
// is what lets a wrapper over this one multiply them by its zero weight.
template <int Q, int K>
struct log_term {
  double lv = 0.0;
  detail_atom_mixture::flat_tensors<Q, K> t;
  LAPLACE_HD void zero() {
    for (int i = 0; i < t.total; ++i) t.v[i] = 0.0;
  }
};

// log F(y) (upper = false) or log(1 - F(y)) of the parent, an infinite end read
// as its limit
template <class D>
LAPLACE_HD double tail_value(const D& p, double y, bool upper) {
  if (y == distrib_infinity) return upper ? -distrib_infinity : 0.0;
  if (y == -distrib_infinity) return upper ? 0.0 : -distrib_infinity;
  return upper ? distrib_log_ccdf(p, y) : distrib_log_cdf(p, y);
}

template <int T, int K, class D>
LAPLACE_HD void fill_tail_order(detail_atom_mixture::flat_tensors<D::n_params, K>& out, const D& p, double y,
                                bool upper) {
  const auto tt = upper ? distrib_log_ccdf_dk<T>(p, y) : distrib_log_cdf_dk<T>(p, y);
  for (int i = 0; i < tt.size; ++i) out.set(T, i, tt[i]);
  if constexpr (T < K) fill_tail_order<T + 1, K>(out, p, y, upper);
}

// the same with its derivatives; where the value is zero or one they are zero
template <int K, class D>
LAPLACE_HD log_term<D::n_params, K> tail_term(const D& p, double y, bool upper) {
  log_term<D::n_params, K> r;
  r.lv = tail_value(p, y, upper);
  if (r.lv > -distrib_infinity && r.lv < 0.0) fill_tail_order<1, K>(r.t, p, y, upper);
  else r.zero();
  return r;
}

// the parent's own log density or log mass at y, with its derivatives
template <int K, class D>
LAPLACE_HD log_term<D::n_params, K> value_term(const D& p, double y) {
  log_term<D::n_params, K> r;
  r.lv = detail_generic::log_density(p, y);
  r.t.fill_value(p, y);
  return r;
}

// log(X - Y) (subtract) or log(X + Y) from log X and log Y; minus infinity where
// the difference is not positive
LAPLACE_HD double log_combine_value(double lx, double ly, bool subtract) {
  if (!(ly > -distrib_infinity)) return lx;
  if (subtract) {
    if (!(ly < lx)) return -distrib_infinity;
    return lx + detail_generic::log1mexp(ly - lx);
  }
  if (!(lx > -distrib_infinity)) return ly;
  const double hi = lx > ly ? lx : ly, lo = lx > ly ? ly : lx;
  return hi + Kokkos::log1p(math::exp(lo - hi));
}

// The order-T components of log(wx X + wy Y) / ..., from the ratios: with the
// weights wx = X / G and wy = +-Y / G of the two terms in G = X +- Y,
// d_T G / G = wx d_T X / X + wy d_T Y / Y, and the derivatives of log G follow
// from the Bell relation solved for them.
template <int Q, int K, int T>
LAPLACE_HD void combine_order(const detail_atom_mixture::flat_tensors<Q, K>& x, double wx,
                              const detail_atom_mixture::flat_tensors<Q, K>& y, double wy,
                              detail_atom_mixture::flat_tensors<Q, K>& out) {
  constexpr int full = (1 << T) - 1;
  int c[T];
  for (int j = 0; j < T; ++j) c[j] = 0;
  for (int comp = 0; comp < sym_tensor_size(Q, T); ++comp) {
    double ux[1 << T], uy[1 << T], rx[1 << T], ry[1 << T], r[1 << T], u[1 << T];
    for (int mask = 0; mask <= full; ++mask) {
      int idx[T];
      int nn = 0;
      for (int j = 0; j < T; ++j)
        if (mask >> j & 1) idx[nn++] = c[j];
      ux[mask] = mask == 0 ? 0.0 : x.at(nn, idx);
      uy[mask] = mask == 0 ? 0.0 : y.at(nn, idx);
    }
    detail_generic::ratios_from_log_tail_dk<T>(ux, rx);
    detail_generic::ratios_from_log_tail_dk<T>(uy, ry);
    for (int mask = 0; mask <= full; ++mask) r[mask] = wx * rx[mask] + wy * ry[mask];
    detail_generic::log_tail_dk_from_ratios<T>(r, u);
    out.set(T, comp, u[full]);
    int j = T - 1;
    while (j >= 0 && c[j] == Q - 1) --j;
    if (j < 0) break;
    const int v = c[j] + 1;
    for (int m = j; m < T; ++m) c[m] = v;
  }
  if constexpr (T < K) combine_order<Q, K, T + 1>(x, wx, y, wy, out);
}

// log(X - Y) or log(X + Y) with every derivative to order K
template <int Q, int K>
LAPLACE_HD log_term<Q, K> log_combine(const log_term<Q, K>& x, const log_term<Q, K>& y, bool subtract) {
  if (!(y.lv > -distrib_infinity)) return x;
  if (!subtract && !(x.lv > -distrib_infinity)) return y;
  log_term<Q, K> g;
  g.lv = log_combine_value(x.lv, y.lv, subtract);
  if (!(g.lv > -distrib_infinity)) {
    g.zero();
    return g;
  }
  const double wx = math::exp(x.lv - g.lv), wy = (subtract ? -1.0 : 1.0) * math::exp(y.lv - g.lv);
  combine_order<Q, K, 1>(x.t, wx, y.t, wy, g.t);
  return g;
}

// The refusal of a window that kills a cell names the model that is meant,
// which always exists (note/piano_wrapper.md section 5ter).
inline std::string killed_cell_message(double c, double lower, double upper) {
  return "truncated: the window [" + std::to_string(lower) + ", " + std::to_string(upper) +
         "] leaves out the atom at " + std::to_string(c) +
         ", so the coordinate that weighs it would have an identically zero score; the same model is the atom "
         "mixture over the atoms inside the window, truncated to it (or the parent alone, truncated, when no atom "
         "is left)";
}

}  // namespace detail_truncated

/**
 * \ingroup distrib_truncated
 * \brief A parent distribution restricted to the window \f$[\mathrm{lower}, \mathrm{upper}]\f$ and renormalized.
 *
 * Build it with truncated(), which checks the window and collapses a truncation of a truncation; the
 * aggregate itself checks nothing, so that it can be built inside a kernel.
 * \tparam D Parent distribution type.
 */
template <class D>
struct truncated_distrib : detail_atom_mixture::observation_regularity<D, D::kind> {
  static_assert(!detail_truncated::is_truncated<D>::value,
                "a truncation of a truncation is the truncation at the intersection of the two windows: build it "
                "with truncated(), which collapses the two");

  /// The wrapped distribution.
  D parent;
  /// Lower bound of the window, which belongs to it; fixed, and independent of the parameters.
  double lower;
  /// Upper bound of the window, which belongs to it; fixed, and independent of the parameters.
  double upper;

  /// Distribution kind, the parent's.
  static constexpr distrib_kind kind = D::kind;
  /// Number of natural parameters, the parent's: truncation adds none.
  static constexpr int n_params = D::n_params;
  /// Descriptors of the parameters, the parent's.
  static constexpr const distrib_param_info* params = D::params;
  /// Number of atoms, the parent's, every one of which lies inside the window.
  static constexpr int n_atoms = distrib_n_atoms<D>();

  /// \brief First point of the window that the law can take. \return \c lower, rounded up for a discrete parent.
  LAPLACE_HD double lower_end() const {
    if constexpr (D::kind == distrib_kind::discrete) return Kokkos::ceil(lower);
    else return lower;
  }
  /// \brief Last point of the window that the law can take. \return \c upper, rounded down for a discrete parent.
  LAPLACE_HD double upper_end() const {
    if constexpr (D::kind == distrib_kind::discrete) return Kokkos::floor(upper);
    else return upper;
  }

  /**
   * \brief Evaluates \f$\log F(\mathrm{lower}-)\f$ or \f$\log(1 - F(\mathrm{lower}-))\f$ of the parent.
   *
   * The left limit leaves out the mass at the bound itself: for a continuous parent it is the mass of an
   * atom the parent declares there, read through distrib_atom_log_mass(); for a discrete one it is the
   * cumulative function at the integer below.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The value.
   */
  LAPLACE_HD double left_limit_value(bool upper_tail) const {
    if constexpr (D::kind == distrib_kind::discrete) {
      return detail_truncated::tail_value(parent, lower_end() - 1.0, upper_tail);
    } else {
      const double t = detail_truncated::tail_value(parent, lower, upper_tail);
      const int i = atom_at_lower();
      if (i < 0) return t;
      return detail_truncated::log_combine_value(t, distrib_atom_log_mass(parent, i), !upper_tail);
    }
  }

  /// \brief Whether \f$Z\f$ is taken from the lower tail. \return \f$F(\mathrm{upper}) \le 1 - F(\mathrm{lower}-)\f$.
  LAPLACE_HD bool from_lower_tail() const {
    return detail_truncated::tail_value(parent, upper_end(), false) <= left_limit_value(true);
  }

  /**
   * \brief Evaluates \f$\log Z\f$, the parent's mass on the window.
   * \return The value, from the tail of the parent that holds the smaller numbers.
   */
  LAPLACE_HD double log_norm() const {
    using detail_truncated::tail_value;
    if (from_lower_tail())
      return detail_truncated::log_combine_value(tail_value(parent, upper_end(), false), left_limit_value(false),
                                                 true);
    return detail_truncated::log_combine_value(left_limit_value(true), tail_value(parent, upper_end(), true), true);
  }

  /**
   * \brief Evaluates the log-density, or the log mass where the law has one.
   * \param y Observation point.
   * \return \f$\log f(y) - \log Z\f$ on the window, \f$-\infty\f$ outside it.
   */
  LAPLACE_HD double log_value(double y) const {
    if (!(y >= lower_end() && y <= upper_end())) return -distrib_infinity;
    return detail_generic::log_density(parent, y) - log_norm();
  }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of the log-density in the parameters.
   *
   * The parent's own tensor less that of \f$\log Z\f$; zero outside the window.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_value_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    sym_tensor<n_params, K> t;
    if (!(y >= lower_end() && y <= upper_end())) return t;
    t = detail_generic::log_density_dk<K>(parent, y);
    const auto z = log_norm_term<K>();
    for (int i = 0; i < t.size; ++i) t[i] -= z.t.v[z.t.offset(K) + i];
    return t;
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
   * \brief Evaluates \f$J\f$ derivatives in \f$y\f$ and the tensor of order \f$K\f$ in the parameters.
   *
   * \f$\log Z\f$ does not depend on \f$y\f$, so for \f$J \ge 1\f$ the result is the parent's own. A
   * parent without this member makes the call an error.
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in the parameters, \f$J + K \le 6\f$.
   * \param y Observation point inside the window.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int J, int K>
  // ponytail: constrained on the kind alone; nvcc evaluates a nested requires-expression on the parent's
  // member template as false where gcc accepts it, and no route probes for this member
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dy_dk(double y) const requires(D::kind == distrib_kind::continuous) {
    if constexpr (J >= 1) {
      return parent.template logpdf_dy_dk<J, K>(y);
    } else if constexpr (K == 0) {
      sym_tensor<n_params, 0> t;
      t[0] = log_value(y);
      return t;
    } else {
      return log_value_dk<K>(y);
    }
  }

  /**
   * \brief Evaluates \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$ of the truncated law.
   *
   * Inside the window each side is a difference of two of the parent's cumulative values over \f$Z\f$,
   * taken on the tail that \f$Z\f$ is taken on.
   * \param y Observation point.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The value.
   */
  LAPLACE_HD double log_cdf_tail(double y, bool upper_tail) const {
    if (y < lower_end()) return upper_tail ? 0.0 : -distrib_infinity;
    if (!(y < upper_end())) return upper_tail ? -distrib_infinity : 0.0;
    using detail_truncated::tail_value;
    double a, b;
    if (from_lower_tail()) {
      const double fy = tail_value(parent, y, false);
      a = upper_tail ? tail_value(parent, upper_end(), false) : fy;
      b = upper_tail ? fy : left_limit_value(false);
    } else {
      const double sy = tail_value(parent, y, true);
      a = upper_tail ? sy : left_limit_value(true);
      b = upper_tail ? tail_value(parent, upper_end(), true) : sy;
    }
    return detail_truncated::log_combine_value(a, b, true) - log_norm();
  }

  /// \brief Evaluates \f$\log F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double log_cdf(double y) const { return log_cdf_tail(y, false); }
  /// \brief Evaluates \f$\log(1 - F(y))\f$. \param y Observation point. \return The value.
  LAPLACE_HD double log_ccdf(double y) const { return log_cdf_tail(y, true); }
  /// \brief Evaluates \f$F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double cdf(double y) const { return math::exp(log_cdf(y)); }
  /// \brief Evaluates \f$1 - F(y)\f$. \param y Observation point. \return The value.
  LAPLACE_HD double ccdf(double y) const { return math::exp(log_ccdf(y)); }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of \f$\log F\f$ or of \f$\log(1 - F)\f$.
   *
   * The same differences as log_cdf_tail(), with their derivatives from the parent's through the ratios,
   * less those of \f$\log Z\f$. Outside the window, where the value is zero or one, they are zero.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_cdf_tail_dk(double y, bool upper_tail) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    sym_tensor<n_params, K> t;
    if (y < lower_end() || !(y < upper_end())) return t;
    using detail_truncated::tail_term;
    detail_truncated::log_term<n_params, K> a, b;
    if (from_lower_tail()) {
      const auto fy = tail_term<K>(parent, y, false);
      a = upper_tail ? tail_term<K>(parent, upper_end(), false) : fy;
      b = upper_tail ? fy : left_limit_term<K>(false);
    } else {
      const auto sy = tail_term<K>(parent, y, true);
      a = upper_tail ? sy : left_limit_term<K>(true);
      b = upper_tail ? tail_term<K>(parent, upper_end(), true) : sy;
    }
    const auto g = detail_truncated::log_combine(a, b, true);
    if (!(g.lv > -distrib_infinity)) return t;
    const auto z = log_norm_term<K>();
    for (int i = 0; i < t.size; ++i) t[i] = g.t.v[g.t.offset(K) + i] - z.t.v[z.t.offset(K) + i];
    return t;
  }

  /// \brief First derivatives of \f$\log F\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_cdf_d1(double y) const { return log_cdf_tail_dk<1>(y, false); }
  /// \brief Second derivatives of \f$\log F\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_cdf_d2(double y) const { return log_cdf_tail_dk<2>(y, false); }
  /// \brief Third derivatives of \f$\log F\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_cdf_d3(double y) const { return log_cdf_tail_dk<3>(y, false); }
  /// \brief Fourth derivatives of \f$\log F\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_cdf_d4(double y) const { return log_cdf_tail_dk<4>(y, false); }
  /// \brief Fifth derivatives of \f$\log F\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_cdf_d5(double y) const { return log_cdf_tail_dk<5>(y, false); }
  /// \brief Sixth derivatives of \f$\log F\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_cdf_d6(double y) const { return log_cdf_tail_dk<6>(y, false); }
  /// \brief First derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> log_ccdf_d1(double y) const { return log_cdf_tail_dk<1>(y, true); }
  /// \brief Second derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> log_ccdf_d2(double y) const { return log_cdf_tail_dk<2>(y, true); }
  /// \brief Third derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> log_ccdf_d3(double y) const { return log_cdf_tail_dk<3>(y, true); }
  /// \brief Fourth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> log_ccdf_d4(double y) const { return log_cdf_tail_dk<4>(y, true); }
  /// \brief Fifth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> log_ccdf_d5(double y) const { return log_cdf_tail_dk<5>(y, true); }
  /// \brief Sixth derivatives of \f$\log(1 - F)\f$. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> log_ccdf_d6(double y) const { return log_cdf_tail_dk<6>(y, true); }

  /**
   * \brief Evaluates the quantile \f$F^{-1}(p)\f$.
   *
   * On the lower tail it is the parent's quantile at \f$F(\mathrm{lower}-) + pZ\f$; on the upper tail the
   * parent's upper quantile at \f$\log(1 - F(\mathrm{upper}) + (1 - p)Z)\f$, read from the log scale where
   * the parent has no atom; the result is kept inside the window.
   * \param p Probability in \f$[0, 1]\f$.
   * \return The quantile.
   */
  LAPLACE_HD double quantile(double p) const {
    if (!(p > 0.0)) return p == 0.0 ? support_lower() : math::detail::quiet_nan();
    if (!(p < 1.0)) return p == 1.0 ? support_upper() : math::detail::quiet_nan();
    const double lz = log_norm();
    double y;
    if (from_lower_tail()) {
      const double t = detail_truncated::log_combine_value(left_limit_value(false), math::log(p) + lz, false);
      y = distrib_quantile(parent, math::exp(t));
    } else {
      const double t = detail_truncated::log_combine_value(detail_truncated::tail_value(parent, upper_end(), true),
                                                           Kokkos::log1p(-p) + lz, false);
      if constexpr (distrib_n_atoms<D>() == 0) y = distrib_quantile_from_log_upper(parent, t);
      else y = distrib_quantile(parent, -Kokkos::expm1(t));
    }
    const double lo = support_lower(), hi = support_upper();
    return y < lo ? lo : (y > hi ? hi : y);
  }

  /**
   * \brief Position of atom \f$i\f$, the parent's.
   * \param i Atom index.
   * \return The parent's atom position.
   */
  LAPLACE_HD double atom_point(int i) const requires(distrib_n_atoms<D>() > 0) { return distrib_atom_point(parent, i); }
  /**
   * \brief Log mass the law puts on atom \f$i\f$.
   * \param i Atom index.
   * \return The parent's log mass there less \f$\log Z\f$.
   */
  LAPLACE_HD double atom_log_mass(int i) const requires(distrib_n_atoms<D>() > 0) {
    return distrib_atom_log_mass(parent, i) - log_norm();
  }

  /**
   * \brief Returns a starting value: the parent's own, from the whole sample, with the same window.
   * \param y The sample, every point inside the window.
   * \return A distribution of the same shape with its parameters set.
   */
  truncated_distrib start_value(const std::vector<double>& y) const {
    truncated_distrib out = *this;
    out.parent = distrib_start_value(parent, y);
    return out;
  }

  /// \brief Lower end of the support. \return The greater of the parent's lower end and the window's.
  LAPLACE_HD double support_lower() const {
    const double a = parent.support_lower(), b = lower_end();
    return a > b ? a : b;
  }
  /// \brief Upper end of the support. \return The lesser of the parent's upper end and the window's.
  LAPLACE_HD double support_upper() const {
    const double a = parent.support_upper(), b = upper_end();
    return a < b ? a : b;
  }
  /// \brief Rough location, for placing a quadrature. \return The parent's, moved into the window.
  LAPLACE_HD double center() const {
    const double c = parent.center(), lo = support_lower(), hi = support_upper();
    return c < lo ? lo : (c > hi ? hi : c);
  }
  /// \brief Rough spread, for placing a quadrature. \return The parent's, or the window's width when smaller.
  LAPLACE_HD double scale() const {
    const double s = parent.scale(), w = support_upper() - support_lower();
    return w < s ? w : s;
  }
  /// \brief Non-smooth points of \f$\log f\f$ in \f$y\f$. \return The parent's.
  LAPLACE_HD distrib_kinks y_kinks() const requires(D::kind == distrib_kind::continuous) { return parent.y_kinks(); }
  /**
   * \brief Non-smooth points of a parameter placed by the observation.
   * \param j Parameter index.
   * \param y Observation point.
   * \return The parent's.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const { return parent.param_kinks(j, y); }

  /**
   * \brief Evaluates \f$\log Z\f$ with its derivatives to order \f$K\f$.
   * \tparam K Highest derivative order.
   * \return The value and the derivatives of every order one to \f$K\f$.
   */
  template <int K>
  LAPLACE_HD detail_truncated::log_term<n_params, K> log_norm_term() const {
    using detail_truncated::tail_term;
    if (from_lower_tail())
      return detail_truncated::log_combine(tail_term<K>(parent, upper_end(), false), left_limit_term<K>(false), true);
    return detail_truncated::log_combine(left_limit_term<K>(true), tail_term<K>(parent, upper_end(), true), true);
  }

 private:
  // the index of the parent's atom sitting on the lower bound, or -1
  LAPLACE_HD int atom_at_lower() const {
    for (int i = 0; i < distrib_n_atoms<D>(); ++i)
      if (distrib_atom_point(parent, i) == lower) return i;
    return -1;
  }

  // left_limit_value with its derivatives: the atom's mass derivatives are the
  // parent's log value derivatives at its point (core.hpp)
  template <int K>
  LAPLACE_HD detail_truncated::log_term<n_params, K> left_limit_term(bool upper_tail) const {
    if constexpr (D::kind == distrib_kind::discrete) {
      return detail_truncated::tail_term<K>(parent, lower_end() - 1.0, upper_tail);
    } else {
      const auto t = detail_truncated::tail_term<K>(parent, lower, upper_tail);
      const int i = atom_at_lower();
      if (i < 0) return t;
      return detail_truncated::log_combine(t, detail_truncated::value_term<K>(parent, lower), !upper_tail);
    }
  }
};

/**
 * \ingroup distrib_truncated
 * \brief Builds the truncation of a distribution to \f$[\mathrm{lower}, \mathrm{upper}]\f$, checking the window.
 *
 * The window is first cut to the parent's support. A truncation of a truncation is built as one truncation at
 * the intersection of the two windows. A window that leaves out a cell the parent weighs with a parameter of
 * its own (an atom) is refused, since that parameter would have an identically zero score, with a message that
 * names the equivalent model; so is a window that holds no mass of the parent, or a single point.
 * \tparam D Parent distribution type.
 * \param parent The wrapped distribution.
 * \param lower Lower bound, which belongs to the window.
 * \param upper Upper bound, which belongs to the window.
 * \return The wrapper, a \c truncated_distrib over the innermost parent.
 * \exception std::invalid_argument When the window is refused.
 */
template <class D>
auto truncated(const D& parent, double lower, double upper) {
  if constexpr (detail_truncated::is_truncated<D>::value) {
    return truncated(parent.parent, std::fmax(lower, parent.lower), std::fmin(upper, parent.upper));
  } else {
    lower = std::fmax(lower, parent.support_lower());
    upper = std::fmin(upper, parent.support_upper());
    const truncated_distrib<D> t{.parent = parent, .lower = lower, .upper = upper};
    if (!(t.lower_end() < t.upper_end()))
      throw std::invalid_argument("truncated: the window [" + std::to_string(lower) + ", " + std::to_string(upper) +
                                  "] within the support holds a single point or none, where no parameter is "
                                  "identified");
    for (int i = 0; i < detail_truncated::cell_points<D>(); ++i) {
      const double c = distrib_atom_point(parent, i);
      if (c < lower || c > upper) throw std::invalid_argument(detail_truncated::killed_cell_message(c, lower, upper));
    }
    if (!(t.log_norm() > -distrib_infinity))
      throw std::invalid_argument("truncated: the window [" + std::to_string(lower) + ", " + std::to_string(upper) +
                                  "] holds no mass of the parent");
    return t;
  }
}

}  // namespace laplace::distrib
