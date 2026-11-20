// The weak-convexity constant of the continuous univariate laws (laplace::distrib; note/penalties.md
// section 5h): rho = max(0, sup_y l''(y)), l = log f, at the parameters held by the distribution.
// It is the constant that makes -log f + rho y^2/2 convex, the one the generic proximal operator
// (generic_prox.hpp) and the penalties read.
//
// rho = 0, log-concave:   gaussian1/2/3, laplace1/2, enet, logistic, gumbel, skewnormal (Phi is
//                         log-concave), pseudohuber (-l = sqrt(nu + R z^2) + const), exponential;
//                         gamma1/2, chisq, weibull and beta when every shape is >= 1;
//                         powerexp for p >= 1.
// rho = infinity:         the same laws with a shape below 1, where l'' -> +infinity at an end of the
//                         support (gamma: l'' = (1 - k)/y^2; beta: (1 - a)/y^2 + (1 - b)/(1 - y)^2;
//                         powerexp: p(1 - p)|z|^(p - 2)/s^2 at the location).
// finite:
//   student_t1    l'' = -(nu + 1)(nu s^2 - x^2)/(nu s^2 + x^2)^2, x = y - mu, largest at x^2 = 3 nu s^2:
//                 rho = (nu + 1)/(8 nu sigma^2)
//   student_t2    the same with the scale s = sigma sqrt((nu - 2)/nu): (nu + 1)/(8 (nu - 2) sigma^2)
//   cauchy        student_t1 at nu = 1: 1/(4 sigma^2)
//   lognormal1    l'' = (1 + (w - mu - 1)/v)/y^2, w = log y, largest at w = mu + 3/2 - v:
//                 rho = exp(2 v - 2 mu - 3)/(2 v)
//   lognormal2    the same through v1 = log(1 + v/m^2), mu1 = log m - v1/2: exp(3 v1 - 3)/(2 m^2 v1)
//   invgauss      l'' = 3/(2 y^2) - lambda/y^3 with lambda = 1/phi, largest at y = lambda:
//                 rho = 1/(2 lambda^2) = phi^2/2
//   mcp, scad     the inner piece gives 1/gamma (MCP) and 1/(a - 1) (SCAD); the Student tail adds
//                 (nu + 1)/(8 nu s^2) (its Psi = log(1 + z^2) has Psi'' >= -1/4), the Gaussian and
//                 logistic tails nothing: rho is the larger.
// fixed_distrib and truncated_distrib read their parent (for truncated, the parent's constant
// bounds the supremum over [lower, upper] from above).
//
// Not declared: skewt (no closed form for sup l''), vonmises (circular), and transformed_distrib
// (its constant depends on the bijector).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_prox.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

// Forward declarations: the readings below read members of the laws only when they are
// instantiated (value is a template whose argument defaults to the law), so this header
// includes no distribution and costs nothing to the headers that include it.
/// \cond LAPLACE_WEAK_CONVEXITY_FORWARD
struct beta_distrib;
struct cauchy_distrib;
struct chisq_distrib;
struct enet_distrib;
struct exponential_distrib;
struct gamma1_distrib;
struct gamma2_distrib;
struct gaussian1_distrib;
struct gaussian2_distrib;
struct gaussian3_distrib;
struct gumbel_distrib;
struct invgauss_distrib;
struct laplace1_distrib;
struct laplace2_distrib;
struct logistic_distrib;
struct lognormal1_distrib;
struct lognormal2_distrib;
struct powerexp_distrib;
struct pseudohuber_distrib;
struct skewnormal_distrib;
struct student_t1_distrib;
struct student_t2_distrib;
struct weibull_distrib;
template <class Tail>
struct mcp_distrib;
template <class Tail>
struct scad_distrib;

template <class D, int... C>
struct fixed_distrib;
template <class D>
struct truncated_distrib;
/// \endcond

namespace detail_weak_convexity {
/// 0 when every shape is at least 1, +infinity otherwise.
LAPLACE_HD double shapes_at_least_one(bool all) { return all ? 0.0 : distrib_infinity; }
/// What a properizing tail adds: the Student tail (nu + 1)/(8 nu s^2), the others nothing.
template <class Tail>
LAPLACE_HD double tail_rho(const Tail& tail) {
  if constexpr (requires { tail.tail_df; })
    return (tail.tail_df + 1.0) / (8.0 * tail.tail_df * tail.tail_scale * tail.tail_scale);
  else return (void)tail, 0.0;
}
}  // namespace detail_weak_convexity

/// \cond LAPLACE_WEAK_CONVEXITY_SPECIALIZATIONS
#define LAPLACE_LOG_CONCAVE_READING(D)                                            \
  template <>                                                                     \
  struct distrib_weak_convexity_reading<D> {                                      \
    template <class T = D>                                                        \
    LAPLACE_HD static double value(const T&) { return 0.0; }                      \
  };
LAPLACE_LOG_CONCAVE_READING(gaussian1_distrib)
LAPLACE_LOG_CONCAVE_READING(gaussian2_distrib)
LAPLACE_LOG_CONCAVE_READING(gaussian3_distrib)
LAPLACE_LOG_CONCAVE_READING(laplace1_distrib)
LAPLACE_LOG_CONCAVE_READING(laplace2_distrib)
LAPLACE_LOG_CONCAVE_READING(enet_distrib)
LAPLACE_LOG_CONCAVE_READING(logistic_distrib)
LAPLACE_LOG_CONCAVE_READING(gumbel_distrib)
LAPLACE_LOG_CONCAVE_READING(skewnormal_distrib)
LAPLACE_LOG_CONCAVE_READING(pseudohuber_distrib)
LAPLACE_LOG_CONCAVE_READING(exponential_distrib)
#undef LAPLACE_LOG_CONCAVE_READING

template <>
struct distrib_weak_convexity_reading<gamma1_distrib> {  // shape 1/phi
  template <class T = gamma1_distrib>
  LAPLACE_HD static double value(const T& d) { return detail_weak_convexity::shapes_at_least_one(d.phi <= 1.0); }
};
template <>
struct distrib_weak_convexity_reading<gamma2_distrib> {  // shape mu^2/sigma2
  template <class T = gamma2_distrib>
  LAPLACE_HD static double value(const T& d) {
    return detail_weak_convexity::shapes_at_least_one(d.mu * d.mu >= d.sigma2);
  }
};
template <>
struct distrib_weak_convexity_reading<chisq_distrib> {  // shape mu/2
  template <class T = chisq_distrib>
  LAPLACE_HD static double value(const T& d) { return detail_weak_convexity::shapes_at_least_one(d.mu >= 2.0); }
};
template <>
struct distrib_weak_convexity_reading<weibull_distrib> {  // shape sigma
  template <class T = weibull_distrib>
  LAPLACE_HD static double value(const T& d) { return detail_weak_convexity::shapes_at_least_one(d.sigma >= 1.0); }
};
template <>
struct distrib_weak_convexity_reading<beta_distrib> {  // shapes mu phi and (1 - mu) phi
  template <class T = beta_distrib>
  LAPLACE_HD static double value(const T& d) {
    return detail_weak_convexity::shapes_at_least_one(d.mu * d.phi >= 1.0 && (1.0 - d.mu) * d.phi >= 1.0);
  }
};
template <>
struct distrib_weak_convexity_reading<powerexp_distrib> {
  template <class T = powerexp_distrib>
  LAPLACE_HD static double value(const T& d) { return detail_weak_convexity::shapes_at_least_one(d.p >= 1.0); }
};
template <>
struct distrib_weak_convexity_reading<student_t1_distrib> {
  template <class T = student_t1_distrib>
  LAPLACE_HD static double value(const T& d) { return (d.nu + 1.0) / (8.0 * d.nu * d.sigma * d.sigma); }
};
template <>
struct distrib_weak_convexity_reading<student_t2_distrib> {
  template <class T = student_t2_distrib>
  LAPLACE_HD static double value(const T& d) {
    return (d.nu + 1.0) / (8.0 * (d.nu - 2.0) * d.sigma * d.sigma);
  }
};
template <>
struct distrib_weak_convexity_reading<cauchy_distrib> {
  template <class T = cauchy_distrib>
  LAPLACE_HD static double value(const T& d) { return 0.25 / (d.sigma * d.sigma); }
};
template <>
struct distrib_weak_convexity_reading<lognormal1_distrib> {
  template <class T = lognormal1_distrib>
  LAPLACE_HD static double value(const T& d) {
    return math::exp(2.0 * d.v - 2.0 * d.mu - 3.0) / (2.0 * d.v);
  }
};
template <>
struct distrib_weak_convexity_reading<lognormal2_distrib> {
  template <class T = lognormal2_distrib>
  LAPLACE_HD static double value(const T& d) {
    const double v1 = Kokkos::log1p(d.v / (d.m * d.m));
    return math::exp(3.0 * v1 - 3.0) / (2.0 * d.m * d.m * v1);
  }
};
template <>
struct distrib_weak_convexity_reading<invgauss_distrib> {
  template <class T = invgauss_distrib>
  LAPLACE_HD static double value(const T& d) { return 0.5 * d.phi * d.phi; }
};
template <class Tail>
struct distrib_weak_convexity_reading<mcp_distrib<Tail>> {
  template <class T = mcp_distrib<Tail>>
  LAPLACE_HD static double value(const T& d) {
    return Kokkos::fmax(1.0 / d.gamma, detail_weak_convexity::tail_rho(d.tail));
  }
};
template <class Tail>
struct distrib_weak_convexity_reading<scad_distrib<Tail>> {
  template <class T = scad_distrib<Tail>>
  LAPLACE_HD static double value(const T& d) {
    return Kokkos::fmax(1.0 / (d.a - 1.0), detail_weak_convexity::tail_rho(d.tail));
  }
};
template <class D, int... C>
  requires has_weak_convexity_reading<D>
struct distrib_weak_convexity_reading<fixed_distrib<D, C...>> {
  template <class T = fixed_distrib<D, C...>>
  LAPLACE_HD static double value(const T& d) { return distrib_weak_convexity_reading<D>::value(d.parent); }
};
template <class D>
  requires has_weak_convexity_reading<D>
struct distrib_weak_convexity_reading<truncated_distrib<D>> {
  template <class T = truncated_distrib<D>>
  LAPLACE_HD static double value(const T& d) { return distrib_weak_convexity_reading<D>::value(d.parent); }
};
/// \endcond

}  // namespace laplace::distrib
