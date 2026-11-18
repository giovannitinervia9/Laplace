// The penalty reading of univariate laws (through univariate_as_multivariate)
// and of their independent products (note/penalties.md sections 5a and 9, step
// 2): the specializations of distrib_penalty_traits that let distrib_penalty
// take them. Lasso is laplace1 or laplace2, the elastic net enet, the Student penalty
// student_t1 or student_t2, the pseudo-Huber penalty pseudohuber, the ridge
// gaussian1 or gaussian2; repeated over the coefficients by
// independent_product_distrib.
//
// The descriptors of univariate_as_multivariate<D> follow from D: the
// regularity in beta is D::y_regularity, a kink (y_regularity finite) sits on
// each coordinate, the regularity in theta is the least of D's scalar
// parameters. What D's declarations do not say, the scale-mixture
// representation and the weak-convexity constant rho = sup(-pen''), is written
// per law in univariate_penalty_reading<D>; a law without one does not compile.
#pragma once

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/penalty/distrib_penalty.hpp>

#include <initializer_list>
#include <type_traits>

namespace laplace::penalty {

/**
 * \ingroup distrib_independent_product
 * \brief What the declarations of a univariate law `D` do not say about its penalty: scale mixture and \f$\rho\f$.
 *
 * Undefined for a general `D`; specialized for laplace1, enet, student_t1,
 * student_t2, pseudohuber, gaussian1 and gaussian2. `weak_convexity(theta)`
 * reads the parameters of `D` after the location, in their order.
 *
 * \tparam D Univariate distribution.
 */
template <class D>
struct univariate_penalty_reading;

/// \ingroup distrib_independent_product
/// \brief Lasso, \f$|\beta - \mu| / \sigma\f$: convex, an exponential mixture of normal variances.
template <>
struct univariate_penalty_reading<distrib::laplace1_distrib> {
  /// Exponential law of the variance.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::exponential_variance;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
  /// \return The rate \f$1/\sigma\f$ of the kink term \f$|\beta - \mu|/\sigma\f$.
  LAPLACE_HD static double l1_rate(const double* theta) { return 1.0 / theta[0]; }
};

/// \ingroup distrib_independent_product
/// \brief Lasso with a rate, \f$\lambda |\beta - \mu|\f$: convex, an exponential mixture of normal variances.
template <>
struct univariate_penalty_reading<distrib::laplace2_distrib> {
  /// Exponential law of the variance.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::exponential_variance;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
  /// \return The rate \f$\lambda\f$ of the kink term \f$\lambda |\beta - \mu|\f$.
  LAPLACE_HD static double l1_rate(const double* theta) { return theta[0]; }
};

/// \ingroup distrib_independent_product
/// \brief Elastic net: convex, no scale mixture declared.
template <>
struct univariate_penalty_reading<distrib::enet_distrib> {
  /// None declared.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
};

/// \ingroup distrib_independent_product
/// \brief Pseudo-Huber: convex, no scale mixture declared.
template <>
struct univariate_penalty_reading<distrib::pseudohuber_distrib> {
  /// None declared.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
};

/// \ingroup distrib_independent_product
/// \brief Ridge with standard deviation \f$\sigma\f$: convex.
template <>
struct univariate_penalty_reading<distrib::gaussian1_distrib> {
  /// None declared (a normal is its own mixture).
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
};

/// \ingroup distrib_independent_product
/// \brief Ridge with variance \f$v\f$: convex.
template <>
struct univariate_penalty_reading<distrib::gaussian2_distrib> {
  /// None declared (a normal is its own mixture).
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// \return 0, convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
};

/**
 * \ingroup distrib_independent_product
 * \brief Student penalty with scale \f$\sigma\f$: \f$\rho = (\nu + 1)/(8 \nu \sigma^2)\f$.
 *
 * With \f$u = (\beta - \mu)^2 / (\nu \sigma^2)\f$ the second derivative is
 * \f$\mathrm{pen}'' = \frac{\nu + 1}{\nu \sigma^2} \frac{1 - u}{(1 + u)^2}\f$,
 * whose least value, at \f$u = 3\f$, is \f$-(\nu + 1)/(8 \nu \sigma^2)\f$.
 */
template <>
struct univariate_penalty_reading<distrib::student_t1_distrib> {
  /// Inverse-gamma law of the variance.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::inverse_gamma_variance;
  /// \param theta \f$(\sigma, \nu)\f$. \return \f$(\nu + 1)/(8 \nu \sigma^2)\f$.
  LAPLACE_HD static double weak_convexity(const double* theta) {
    return (theta[1] + 1.0) / (8.0 * theta[1] * theta[0] * theta[0]);
  }
};

/**
 * \ingroup distrib_independent_product
 * \brief Student penalty with standard deviation \f$\sigma\f$: \f$\rho = (\nu + 1)/(8 (\nu - 2) \sigma^2)\f$.
 *
 * The scale is \f$s = \sigma\sqrt{(\nu - 2)/\nu}\f$, and \f$\rho = (\nu + 1)/(8 \nu s^2)\f$
 * as for `student_t1_distrib`.
 */
template <>
struct univariate_penalty_reading<distrib::student_t2_distrib> {
  /// Inverse-gamma law of the variance.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::inverse_gamma_variance;
  /// \param theta \f$(\sigma, \nu)\f$. \return \f$(\nu + 1)/(8 (\nu - 2) \sigma^2)\f$.
  LAPLACE_HD static double weak_convexity(const double* theta) {
    return (theta[1] + 1.0) / (8.0 * (theta[1] - 2.0) * theta[0] * theta[0]);
  }
};

namespace detail_penalty {
template <class D>
constexpr int least_theta_regularity() {
  int r = distrib::infinitely_smooth;
  for (int j = 1; j < D::n_params; ++j)
    if (D::params[j].regularity < r) r = D::params[j].regularity;
  return r;
}
}  // namespace detail_penalty

/**
 * \ingroup distrib_independent_product
 * \brief Descriptors of the penalty of a univariate law `D` read as a multivariate law of dimension 1.
 * \tparam D Univariate distribution with a `univariate_penalty_reading`.
 */
template <class D>
struct distrib_penalty_traits<distrib::univariate_as_multivariate<D>> {
  /// Regularity in \f$\beta\f$: that of \f$\log f_D\f$ in \f$y\f$.
  static constexpr int beta_regularity = D::y_regularity;
  /// Kinks: on the coordinate when \f$\log f_D\f$ is not smooth in \f$y\f$.
  static constexpr penalty_kink_geometry kinks =
      D::y_regularity == distrib::infinitely_smooth ? penalty_kink_geometry::none : penalty_kink_geometry::per_coordinate;
  /// Structure: one coordinate.
  static constexpr penalty_structure structure = penalty_structure::separable;
  /// Regularity in \f$\theta\f$: the least over the scalar parameters of `D`.
  static constexpr int theta_regularity = detail_penalty::least_theta_regularity<D>();
  /// Not a quadratic penalty with a linear precision.
  static constexpr bool q_linear_in_theta = false;
  /// Scale mixture, from `univariate_penalty_reading<D>`.
  static constexpr penalty_scale_mixture scale_mixture = univariate_penalty_reading<D>::scale_mixture;
  /// \param theta Parameters of `D` after the location. \return \f$\rho\f$ from `univariate_penalty_reading<D>`.
  LAPLACE_HD static double weak_convexity(const double* theta) { return univariate_penalty_reading<D>::weak_convexity(theta); }
};

namespace detail_penalty {
template <class E>
constexpr E largest(std::initializer_list<E> l) {
  E m = *l.begin();
  for (E e : l) m = int(e) > int(m) ? e : m;
  return m;
}
constexpr int least(std::initializer_list<int> l) {
  int m = *l.begin();
  for (int e : l) m = e < m ? e : m;
  return m;
}
}  // namespace detail_penalty

/**
 * \ingroup distrib_independent_product
 * \brief Descriptors of the penalty of an independent product, from those of its components.
 *
 * Regularities are the least over the components; the kink geometry is the
 * most general one present (none, per coordinate, radial, linear map); the
 * structure is separable when every component is, grouped otherwise; \f$Q\f$
 * is linear in the weights when it is in every component; the scale mixture is
 * that of the component when there is one component type, none otherwise;
 * \f$\rho\f$ is the largest over the parameter sets of all components.
 *
 * \tparam M Component distributions, each with its `distrib_penalty_traits`.
 */
template <class... M>
struct distrib_penalty_traits<distrib::independent_product_distrib<M...>> {
  /// Regularity in \f$\beta\f$: the least over the components.
  static constexpr int beta_regularity = detail_penalty::least({distrib_penalty_traits<M>::beta_regularity...});
  /// Kinks: the most general geometry among the components.
  static constexpr penalty_kink_geometry kinks = detail_penalty::largest({distrib_penalty_traits<M>::kinks...});
  /// Structure: separable when every component is, grouped by copies otherwise.
  static constexpr penalty_structure structure =
      ((distrib_penalty_traits<M>::structure == penalty_structure::separable) && ...) ? penalty_structure::separable
                                                                                     : penalty_structure::grouped;
  /// Regularity in \f$\theta\f$: the least over the components.
  static constexpr int theta_regularity = detail_penalty::least({distrib_penalty_traits<M>::theta_regularity...});
  /// \f$Q\f$ linear in the weights when it is in every component.
  static constexpr bool q_linear_in_theta = (distrib_penalty_traits<M>::q_linear_in_theta && ...);
  /// Scale mixture: that of the single component type, none for several.
  static constexpr penalty_scale_mixture scale_mixture =
      sizeof...(M) == 1 ? detail_penalty::largest({distrib_penalty_traits<M>::scale_mixture...})
                        : penalty_scale_mixture::none;

  /**
   * \brief \f$\rho\f$, the largest over the parameter sets of every component.
   *
   * Each set's \f$\theta\f$ is gathered into a buffer of 32 doubles; a
   * component with a longer \f$\theta\f$ gives \f$+\infty\f$ (pen'' not bounded),
   * a safe value.
   *
   * \param m The product.
   * \param theta Its parameters, param-major, length `m.theta_size()`.
   * \return \f$\rho\f$.
   */
  LAPLACE_HD static double weak_convexity(const distrib::independent_product_distrib<M...>& m, const double* theta) {
    double rho = 0.0;
    const double* tk = theta;
    m.each([&](const auto& c, auto) {
      using Mk = std::remove_cvref_t<decltype(c.distribution)>;
      const int q = c.distribution.theta_size();
      // ponytail: fixed 32-double buffer; a component with more theta reports +infinity, a buffer in w if one needs it
      double buf[32];
      for (int h = 0; h < c.n_sets; ++h) {
        double r = distrib::distrib_infinity;
        if (q <= 32) {
          distrib::independent_product_distrib<M...>::gather_set_theta(c, tk, h, buf);
          r = distrib_penalty_traits<Mk>::weak_convexity(buf);
        }
        rho = r > rho ? r : rho;
      }
      tk += c.n_sets * q;
    });
    return rho;
  }
};

}  // namespace laplace::penalty
