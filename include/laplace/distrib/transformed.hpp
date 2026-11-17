// A distribution carried through a fixed monotone map (laplace::distrib): the
// law of Y = g(X), with X following a parent distribution and g a scalar
// bijector (bijector/scalar.hpp). The plan is note/piano_wrapper.md section
// 5quater, and the composition rule its section 5ter.
//
// With h = g^-1 the inverse map,
//
//   l(y; theta) = log f_X(h(y); theta) + log |h'(y)|.
//
// The map's own parameters (affine's location and scale, power's, Box-Cox's and
// Yeo-Johnson's exponent, softplus' sharpness, bounded's ends) are fixed values
// held in the bijector: they are not coordinates of theta, and n_params and the
// descriptors are the parent's. Only bijections are carried; a many-to-one map
// such as |x| is outside this wrapper.
//
// Since the Jacobian term carries no theta, everything in theta is the
// parent's, read at x = h(y):
//
//   d^K l / dtheta^K (y) = d^K l_X / dtheta^K (h(y)),    K = 1..6.
//
// The expected information and its derivatives are the parent's as well. For
// any function a of the observation, E_Y[a(h(Y))] = E_X[a(X)], because Y = g(X)
// and h(g(X)) = X; the observed derivatives in theta of l_Y at y are those of
// l_X at h(y), so -E_Y[d^2 l_Y] = -E_X[d^2 l_X], and the same holds for every
// expectation the derivatives of the information are made of. A parent's atom
// keeps its mass under a bijection, so the argument covers a law with atoms too.
//
// Only the direction in y needs the chain rule. With x = h(y) and h^(m) the
// derivatives of the inverse map, Faa di Bruno's formula gives, for J >= 1,
//
//   d^J_y d^K_theta l(y) = sum over k = 1..J of d^k_x d^K_theta l_X(x) B_(J,k)(h'(y), ..., h^(J-k+1)(y))
//                          + [K = 0] d^J_y log |h'(y)|,
//
// with B_(J,k) the partial Bell polynomials of math/faa_di_bruno.hpp.
//
// The cumulative functions pass through with no difference formed. For an
// increasing map F_Y(y) = F_X(h(y)) and 1 - F_Y(y) = 1 - F_X(h(y)). For a
// decreasing map the two sides swap, F_Y(y) = P(X >= h(y)) and
// 1 - F_Y(y) = P(X < h(y)), which differ from the parent's tails only by the
// mass of an atom the parent may carry at h(y). The direction is read once, by
// the builder, from the sign of g' at the parent's center, never from the
// map's name. The quantile is g of the parent's quantile at p (increasing) or of
// its upper quantile at p (decreasing).
//
// The builder transformed() refuses what is not a law, following the two other
// wrappers: a parent whose support is not inside the domain of g, an atom whose
// image is not finite, and a union of kinks in y longer than the two points
// distrib_kinks holds. The aggregate itself checks nothing, so that it can be
// built inside a kernel.
#pragma once

#include <laplace/bijector/scalar.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/truncated.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace laplace::distrib {

namespace detail_transformed {

// h^(k)(y) and d^k log |h'(y)| / dy^k, k = 1..6, through the bijector's one
// function per order
LAPLACE_HD double inverse_dk(const bijector::scalar_bijector& b, int k, double y) {
  switch (k) {
    case 1: return b.inverse_d1(y);
    case 2: return b.inverse_d2(y);
    case 3: return b.inverse_d3(y);
    case 4: return b.inverse_d4(y);
    case 5: return b.inverse_d5(y);
    default: return b.inverse_d6(y);
  }
}
LAPLACE_HD double inverse_log_jacobian_dk(const bijector::scalar_bijector& b, int k, double y) {
  switch (k) {
    case 1: return b.inverse_log_jacobian_d1(y);
    case 2: return b.inverse_log_jacobian_d2(y);
    case 3: return b.inverse_log_jacobian_d3(y);
    case 4: return b.inverse_log_jacobian_d4(y);
    case 5: return b.inverse_log_jacobian_d5(y);
    default: return b.inverse_log_jacobian_d6(y);
  }
}

// t += sum over k = k0..J of d^k_x d^K_theta l_X(x) B_(J,k)(hd)
template <int J, int K, int k, class D>
LAPLACE_HD void add_faa_di_bruno(sym_tensor<D::n_params, K>& t, const D& p, double x, const double* hd) {
  const double b = math::bell_partial(J, k, hd);
  const auto o = p.template logpdf_dy_dk<k, K>(x);
  for (int i = 0; i < t.size; ++i) t[i] += o[i] * b;
  if constexpr (k < J) add_faa_di_bruno<J, K, k + 1>(t, p, x, hd);
}

}  // namespace detail_transformed

/**
 * \ingroup distrib_transformed
 * \brief The law of \f$Y = g(X)\f$, with \f$X\f$ following a parent distribution and \f$g\f$ a fixed scalar bijector.
 *
 * Build it with transformed(), which checks that the parent's support lies in the domain of \f$g\f$ and reads the
 * direction of the map; the aggregate itself checks nothing, so that it can be built inside a kernel.
 * \tparam D Parent distribution type, continuous.
 */
template <class D>
struct transformed_distrib {
  static_assert(D::kind == distrib_kind::continuous,
                "transformed_distrib carries a continuous parent: the image of a discrete law under a map is not "
                "supported on the integers, which is what the discrete routes sum over");

  /// The wrapped distribution.
  D parent;
  /// The map \f$g\f$; its parameters are fixed values, independent of \f$\theta\f$.
  bijector::scalar_bijector map;
  /// Whether \f$g\f$ decreases, read by transformed() from the sign of \f$g'\f$.
  bool decreasing = false;

  /// Distribution kind, the parent's.
  static constexpr distrib_kind kind = D::kind;
  /// Number of natural parameters, the parent's: the map adds none.
  static constexpr int n_params = D::n_params;
  /// Descriptors of the parameters, the parent's.
  static constexpr const distrib_param_info* params = D::params;
  /// Number of atoms, the parent's, each carried to its image.
  static constexpr int n_atoms = distrib_n_atoms<D>();
  /// Differentiability order of \f$\log f\f$ in \f$y\f$ away from the kinks, the parent's; the order at a kink
  /// of the map (Yeo-Johnson's, 2) is reported by y_kinks().
  static constexpr int y_regularity = D::y_regularity;

  /**
   * \brief The point of the parent's scale that \f$y\f$ comes from.
   * \param y Observation point.
   * \return The parent's atom when \f$y\f$ is the image of one, so that no rounding of \f$h(g(c))\f$ moves it off
   *         the atom; \f$h(y)\f$ otherwise.
   */
  LAPLACE_HD double preimage(double y) const {
    for (int i = 0; i < n_atoms; ++i) {
      const double c = distrib_atom_point(parent, i);
      if (y == map.forward(c)) return c;
    }
    return map.inverse(y);
  }
  /// \brief Whether \f$y\f$ is the image of one of the parent's atoms. \param y Observation point. \return The answer.
  LAPLACE_HD bool at_atom(double y) const {
    for (int i = 0; i < n_atoms; ++i)
      if (y == map.forward(distrib_atom_point(parent, i))) return true;
    return false;
  }
  /// \brief Whether \f$y\f$ lies in the closed support. \param y Observation point. \return The answer.
  LAPLACE_HD bool inside(double y) const { return y >= support_lower() && y <= support_upper(); }

  /**
   * \brief Evaluates the log-density, or the log mass at the image of an atom.
   * \param y Observation point.
   * \return \f$\log f_X(h(y)) + \log|h'(y)|\f$; the parent's log mass at an atom's image; \f$-\infty\f$ outside
   *         the support.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!inside(y)) return -distrib_infinity;
    if (at_atom(y)) return parent.logpdf(preimage(y));
    return parent.logpdf(map.inverse(y)) + map.inverse_log_jacobian(y);
  }

  /**
   * \brief Evaluates the derivatives of order \f$K\f$ of the log-density in the parameters.
   *
   * The parent's own, at \f$h(y)\f$: the Jacobian term carries no parameter. Zero outside the support.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    if (!inside(y)) return sym_tensor<n_params, K>{};
    return detail_generic::log_density_dk<K>(parent, preimage(y));
  }

  /// \brief First derivatives of the log-density. \param y Observation point. \return Order-1 tensor.
  LAPLACE_HD sym_tensor<n_params, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// \brief Second derivatives of the log-density. \param y Observation point. \return Order-2 tensor.
  LAPLACE_HD sym_tensor<n_params, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// \brief Third derivatives of the log-density. \param y Observation point. \return Order-3 tensor.
  LAPLACE_HD sym_tensor<n_params, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// \brief Fourth derivatives of the log-density. \param y Observation point. \return Order-4 tensor.
  LAPLACE_HD sym_tensor<n_params, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// \brief Fifth derivatives of the log-density. \param y Observation point. \return Order-5 tensor.
  LAPLACE_HD sym_tensor<n_params, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// \brief Sixth derivatives of the log-density. \param y Observation point. \return Order-6 tensor.
  LAPLACE_HD sym_tensor<n_params, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  /**
   * \brief Evaluates \f$J\f$ derivatives in \f$y\f$ and the tensor of order \f$K\f$ in the parameters.
   *
   * For \f$J \ge 1\f$ Faa di Bruno's formula over the parent's \f$\partial_x^k \partial_\theta^K l_X\f$ at
   * \f$x = h(y)\f$ and the derivatives of \f$h\f$, plus the derivative of \f$\log|h'|\f$ when \f$K = 0\f$. It
   * has no meaning at the image of an atom. A parent without this member makes the call an error.
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in the parameters, \f$J + K \le 6\f$.
   * \param y Observation point inside the support.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dy_dk(double y) const {
    static_assert(J >= 0 && K >= 0 && J + K <= 6, "total order at most six");
    sym_tensor<n_params, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
    } else {
      double hd[J + 1];
      hd[0] = 0.0;
      for (int m = 1; m <= J; ++m) hd[m] = detail_transformed::inverse_dk(map, m, y);
      detail_transformed::add_faa_di_bruno<J, K, 1>(t, parent, map.inverse(y), hd);
      if constexpr (K == 0) t[0] += detail_transformed::inverse_log_jacobian_dk(map, J, y);
    }
    return t;
  }

  /**
   * \brief Evaluates \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$.
   *
   * The parent's own tail at \f$h(y)\f$, on the same side for an increasing map and on the other side for a
   * decreasing one, where the mass of an atom at \f$h(y)\f$ moves to the side that holds it.
   * \param y Observation point.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The value.
   */
  LAPLACE_HD double log_cdf_tail(double y, bool upper_tail) const {
    if (y < support_lower()) return upper_tail ? 0.0 : -distrib_infinity;
    if (!(y < support_upper())) return upper_tail ? -distrib_infinity : 0.0;
    const double x = preimage(y);
    if (!decreasing) return detail_truncated::tail_value(parent, x, upper_tail);
    // P(X >= x) = S(x) + p(x) for log F, P(X < x) = F(x) - p(x) for log(1 - F)
    const double t = detail_truncated::tail_value(parent, x, !upper_tail);
    if (!at_atom(y)) return t;
    return detail_truncated::log_combine_value(t, parent.logpdf(x), upper_tail);
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
   * The parent's own, on the side log_cdf_tail() reads; at an atom's image under a decreasing map the sum or
   * difference with the atom's mass, through the Bell relation of \ref distrib_generic. Zero outside the support.
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observation point.
   * \param upper_tail \c true for \f$\log(1 - F)\f$, \c false for \f$\log F\f$.
   * \return The packed symmetric tensor of order \f$K\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> log_cdf_tail_dk(double y, bool upper_tail) const {
    static_assert(K >= 1 && K <= 6, "orders one to six");
    sym_tensor<n_params, K> t;
    if (y < support_lower() || !(y < support_upper())) return t;
    const double x = preimage(y);
    const bool side = decreasing ? !upper_tail : upper_tail;
    if (!decreasing || !at_atom(y)) return side ? distrib_log_ccdf_dk<K>(parent, x) : distrib_log_cdf_dk<K>(parent, x);
    const auto g = detail_truncated::log_combine(detail_truncated::tail_term<K>(parent, x, side),
                                                 detail_truncated::value_term<K>(parent, x), upper_tail);
    if (!(g.lv > -distrib_infinity)) return t;
    for (int i = 0; i < t.size; ++i) t[i] = g.t.v[g.t.offset(K) + i];
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
   * \f$g\f$ of the parent's quantile at \f$p\f$ for an increasing map; for a decreasing one \f$g\f$ of the
   * parent's upper quantile at \f$\log p\f$, so that a small \f$p\f$ is not lost in \f$1 - p\f$ (at
   * \f$1 - p\f$ itself where the parent has atoms).
   * \param p Probability in \f$[0, 1]\f$.
   * \return The quantile.
   */
  LAPLACE_HD double quantile(double p) const {
    if (!(p > 0.0)) return p == 0.0 ? support_lower() : math::detail::quiet_nan();
    if (!(p < 1.0)) return p == 1.0 ? support_upper() : math::detail::quiet_nan();
    if (!decreasing) return map.forward(distrib_quantile(parent, p));
    if constexpr (n_atoms == 0) return map.forward(distrib_quantile_from_log_upper(parent, math::log(p)));
    else return map.forward(distrib_quantile(parent, 1.0 - p));
  }

  /**
   * \brief The expected information, the parent's: an expectation over \f$Y\f$ of a function of \f$h(Y)\f$ is
   * the same expectation over \f$X\f$.
   * \return The parent's member, or the generic route over the parent.
   */
  LAPLACE_HD sym_tensor<n_params, 2> expected_info() const { return distrib_expected_info<0>(parent); }
  /// \brief Derivatives of order 1 of the expected information, the parent's. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 1> expected_info_d1() const { return distrib_expected_info<1>(parent); }
  /// \brief Derivatives of order 2 of the expected information, the parent's. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 2> expected_info_d2() const { return distrib_expected_info<2>(parent); }
  /// \brief Derivatives of order 3 of the expected information, the parent's. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 3> expected_info_d3() const { return distrib_expected_info<3>(parent); }
  /// \brief Derivatives of order 4 of the expected information, the parent's. \return The tensor.
  LAPLACE_HD expected_info_tensor<n_params, 4> expected_info_d4() const { return distrib_expected_info<4>(parent); }

  /**
   * \brief Position of atom \f$i\f$, the image of the parent's.
   * \param i Atom index.
   * \return \f$g\f$ of the parent's atom position.
   */
  LAPLACE_HD double atom_point(int i) const requires(distrib_n_atoms<D>() > 0) {
    return map.forward(distrib_atom_point(parent, i));
  }
  /**
   * \brief Log mass on atom \f$i\f$, the parent's: a bijection carries the mass unchanged.
   * \param i Atom index.
   * \return The parent's log mass.
   */
  LAPLACE_HD double atom_log_mass(int i) const requires(distrib_n_atoms<D>() > 0) {
    return distrib_atom_log_mass(parent, i);
  }

  /**
   * \brief Returns a starting value: the parent's own, from the sample carried back through \f$h\f$.
   *
   * Each observation is mapped back before anything is summarized, so the parent reads the sample on its own
   * scale; no moment of \f$Y\f$ has to be pushed through \f$h\f$, which would not give the moment of \f$X\f$.
   * \param y The sample, every point inside the support.
   * \return A distribution of the same shape with the parent's parameters set.
   */
  transformed_distrib start_value(const std::vector<double>& y) const {
    std::vector<double> x(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) x[i] = preimage(y[i]);
    transformed_distrib out = *this;
    out.parent = distrib_start_value(parent, x);
    return out;
  }

  /**
   * \brief The image under \f$g\f$ of an end of the parent's support.
   *
   * An end of the map's domain goes to the matching end of its image, read from the bijector's bounds rather
   * than by evaluating \f$g\f$ there, where a map may have no value (an infinite end); any other point is
   * \f$g(x)\f$.
   * \param x An end of the parent's support.
   * \return Its image.
   */
  LAPLACE_HD double end_image(double x) const {
    if (x == map.forward_domain_lower()) return decreasing ? map.inverse_domain_upper() : map.inverse_domain_lower();
    if (x == map.forward_domain_upper()) return decreasing ? map.inverse_domain_lower() : map.inverse_domain_upper();
    return map.forward(x);
  }
  /// \brief Lower end of the support. \return The image of the parent's lower end, or of its upper end when \f$g\f$ decreases.
  LAPLACE_HD double support_lower() const {
    return end_image(decreasing ? parent.support_upper() : parent.support_lower());
  }
  /// \brief Upper end of the support. \return The image of the parent's upper end, or of its lower end when \f$g\f$ decreases.
  LAPLACE_HD double support_upper() const {
    return end_image(decreasing ? parent.support_lower() : parent.support_upper());
  }
  /// \brief Rough location, for placing a quadrature. \return \f$g\f$ of the parent's.
  LAPLACE_HD double center() const { return map.forward(parent.center()); }
  /// \brief Rough spread, for placing a quadrature. \return The parent's, pushed to first order: times \f$|g'|\f$ at its center.
  LAPLACE_HD double scale() const { return parent.scale() * Kokkos::fabs(map.forward_d1(parent.center())); }

  /**
   * \brief Writes every non-smooth point of \f$\log f\f$ in \f$y\f$: the images of the parent's, then the map's own.
   * \param point Output buffer of six doubles, sorted on return, no point repeated.
   * \param order The least order of continuity among them.
   * \return The number of points.
   */
  LAPLACE_HD int all_y_kinks(double* point, int& order) const {
    int n = 0;
    order = infinitely_smooth;
    auto add = [&](double v, int o) {
      for (int i = 0; i < n; ++i)
        if (point[i] == v) {
          order = o < order ? o : order;
          return;
        }
      point[n++] = v;
      order = o < order ? o : order;
    };
    const distrib_kinks pk = parent.y_kinks();
    for (int i = 0; i < pk.count; ++i) add(map.forward(pk.point[i]), pk.continuous_order);
    const bijector::bijector_kinks mk = map.inverse_kinks();
    for (int i = 0; i < mk.count; ++i) add(mk.point[i], mk.continuous_order);
    for (int i = 1; i < n; ++i)
      for (int j = i; j > 0 && point[j] < point[j - 1]; --j) {
        const double s = point[j];
        point[j] = point[j - 1];
        point[j - 1] = s;
      }
    return n;
  }

  /// \brief Non-smooth points of \f$\log f\f$ in \f$y\f$. \return At most two, checked by transformed().
  LAPLACE_HD distrib_kinks y_kinks() const {
    double p[6];
    int order;
    const int n = all_y_kinks(p, order);
    distrib_kinks k;
    k.count = n < 2 ? n : 2;
    for (int i = 0; i < k.count; ++i) k.point[i] = p[i];
    k.continuous_order = n > 0 ? order : 0;
    return k;
  }
  /**
   * \brief Non-smooth points of a parameter placed by the observation.
   * \param j Parameter index.
   * \param y Observation point.
   * \return The parent's, at \f$h(y)\f$.
   */
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const { return parent.param_kinks(j, preimage(y)); }
};

/**
 * \ingroup distrib_transformed
 * \brief Builds the law of \f$g(X)\f$, checking that it is one.
 *
 * Refuses a parent whose support is not inside the closed domain of \f$g\f$, a map that is not strictly monotone
 * at the parent's center, an atom whose image is not finite, and more than two non-smooth points in \f$y\f$
 * (those of the parent carried through \f$g\f$ together with the map's own), which is all that distrib_kinks
 * holds.
 * \tparam D Parent distribution type, continuous.
 * \param parent The wrapped distribution.
 * \param map The map \f$g\f$, with its parameters fixed.
 * \return The wrapper.
 * \exception std::invalid_argument When the law is refused.
 */
template <class D>
transformed_distrib<D> transformed(const D& parent, const bijector::scalar_bijector& map) {
  const double lo = parent.support_lower(), hi = parent.support_upper();
  const double dlo = map.forward_domain_lower(), dhi = map.forward_domain_upper();
  if (!(lo >= dlo && hi <= dhi))
    throw std::invalid_argument("transformed: the parent's support [" + std::to_string(lo) + ", " +
                                std::to_string(hi) + "] is not inside the domain [" + std::to_string(dlo) + ", " +
                                std::to_string(dhi) + "] of the map");
  const double slope = map.forward_d1(parent.center());
  if (!(slope > 0.0 || slope < 0.0))
    throw std::invalid_argument("transformed: the map's derivative at the parent's center is " + std::to_string(slope) +
                                ", so its direction cannot be read there");
  const transformed_distrib<D> t{.parent = parent, .map = map, .decreasing = slope < 0.0};
  for (int i = 0; i < distrib_n_atoms<D>(); ++i)
    if (!std::isfinite(map.forward(distrib_atom_point(parent, i))))
      throw std::invalid_argument("transformed: the parent's atom at " + std::to_string(distrib_atom_point(parent, i)) +
                                  " has no finite image under the map");
  double p[6];
  int order;
  const int n = t.all_y_kinks(p, order);
  if (n > 2)
    throw std::invalid_argument("transformed: the parent's non-smooth points in y carried through the map, together "
                                "with the map's own, are " + std::to_string(n) +
                                ", and a distribution declares at most two");
  return t;
}

}  // namespace laplace::distrib
