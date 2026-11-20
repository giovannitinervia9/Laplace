// The proximal operators of penalties step 3 (note/penalties.md section 5c):
//  - prox_y of laplace1, laplace2 and enet, inside a Kokkos kernel, on a grid of
//    points v (v = mu, v on the threshold mu +- t k, large |v|) and steps t
//    (1e-8 to 1e8), enet with alpha = 1e-9, 1/2, 1 - 1e-9 (and the ends 0 and 1,
//    outside the open domain of alpha, for (a) only): (a) the optimality condition
//    0 in d pen(y*) + (y* - v)/t, the residual t pen'(y*) + y* - v through
//    logpdf_dy_dk<1, 0>, divided by its conditioning 1 + t pen'', when y* != mu,
//    |v - mu| <= t k when y* = mu; (b) a
//    golden-section minimization of -logpdf(y) + (y - v)^2/(2t) on [mu, v];
//  - NaN in gives NaN out, an infinite v an infinite y*;
//  - the penalty: the product's prox equals the component's prox_y coordinate by
//    coordinate (theta shared, by group, by block, and a heterogeneous product),
//    distrib_penalty::prox applies the prior location mu_0, and the descriptor
//    is closed_form exactly when every component has prox_y;
//  - rho: pen'' >= 0 off the kink for the convex laws, max(-pen'') = rho for
//    student_t2 (student_t1 is in test_independent_product.cpp).
#include <doctest/doctest.h>

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/distrib/skewt.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace prox_test {

namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
template <class... M>
using product = ld::independent_product_distrib<M...>;

using lasso_pen = lpen::distrib_penalty<product<uam<ld::laplace1_distrib>>>;
using mixed_pen = lpen::distrib_penalty<product<uam<ld::enet_distrib>, uam<ld::laplace2_distrib>>>;
static_assert(lpen::distrib_penalty<product<uam<ld::laplace2_distrib>>>::scale_mixture ==
              lpen::penalty_scale_mixture::exponential_variance);
static_assert(lasso_pen::prox == lpen::penalty_prox_kind::closed_form &&
              mixed_pen::prox == lpen::penalty_prox_kind::closed_form);
// the smooth laws through the generic route of distrib/generic_prox.hpp (note/penalties.md 5h); a law
// without a declared weak-convexity constant has none
static_assert(lpen::distrib_penalty<uam<ld::gaussian1_distrib>>::prox == lpen::penalty_prox_kind::closed_form &&
              lpen::distrib_penalty<product<uam<ld::pseudohuber_distrib>>>::prox == lpen::penalty_prox_kind::closed_form &&
              lpen::distrib_penalty<product<uam<ld::laplace1_distrib>, uam<ld::student_t1_distrib>>>::prox ==
                  lpen::penalty_prox_kind::closed_form &&
              lpen::distrib_penalty<uam<ld::skewt_distrib>>::prox == lpen::penalty_prox_kind::none);

// the soft-threshold weight k of each law: the kink term of -log f is k |y - mu|
KOKKOS_INLINE_FUNCTION double kink_weight(const ld::laplace1_distrib& d) { return 1.0 / d.sigma; }
KOKKOS_INLINE_FUNCTION double kink_weight(const ld::laplace2_distrib& d) { return d.lambda; }
KOKKOS_INLINE_FUNCTION double kink_weight(const ld::enet_distrib& d) { return d.lambda * d.alpha; }

template <class D>
KOKKOS_INLINE_FUNCTION double objective(const D& d, double y, double v, double t) {
  return -d.logpdf(y) + (y - v) * (y - v) / (2.0 * t);
}

// golden-section minimum of the objective on the segment between mu and v
template <class D>
KOKKOS_INLINE_FUNCTION double golden(const D& d, double v, double t) {
  const double g = 0.38196601125010515;
  double a = Kokkos::fmin(d.mu, v), b = Kokkos::fmax(d.mu, v);
  double c = a + g * (b - a), e = b - g * (b - a), fc = objective(d, c, v, t), fe = objective(d, e, v, t);
  for (int i = 0; i < 200 && b - a > 0.0; ++i) {
    if (fc <= fe) {
      b = e;
      e = c;
      fe = fc;
      c = a + g * (b - a);
      fc = objective(d, c, v, t);
    } else {
      a = c;
      c = e;
      fc = fe;
      e = b - g * (b - a);
      fe = objective(d, e, v, t);
    }
  }
  return fc <= fe ? c : e;
}

// res[0]: the optimality condition, relative to max(1, |v|, t k); off the kink the
// residual t pen'(y) + y - v moves by (1 + t pen'') per unit of y, so the rounding of y
// is amplified by that factor, by which it is divided. res[1]: the distance to the
// golden-section minimum, relative to max(1, |v|), when golden is asked (it needs a
// finite logpdf, which the enet has only for alpha in the open interval (0, 1))
template <class D>
KOKKOS_INLINE_FUNCTION void check_point(const D& d, double v, double t, bool with_golden, double* res) {
  const double y = d.prox_y(v, t), k = t * kink_weight(d);
  // at y = mu a point beyond the threshold by e has its minimizer e / (1 + t pen'') from mu,
  // which rounds to mu when that is below an ulp: the same factor divides both branches
  const double d2pen = -d.template logpdf_dy_dk<2, 0>(y)[0];
  const double scale = Kokkos::fmax(1.0, Kokkos::fmax(Kokkos::fabs(v), k)) * (1.0 + t * d2pen);
  if (y == d.mu) {
    res[0] = Kokkos::fmax(0.0, Kokkos::fabs(v - d.mu) - k) / scale;
  } else {
    const double dpen = -d.template logpdf_dy_dk<1, 0>(y)[0];
    res[0] = Kokkos::fabs(t * dpen + y - v) / scale;
  }
  res[1] = with_golden ? Kokkos::fabs(y - golden(d, v, t)) / Kokkos::fmax(1.0, Kokkos::fabs(v)) : 0.0;
}

constexpr int n_t = 7, n_u = 9;
KOKKOS_INLINE_FUNCTION double step(int i) {
  const double ts[n_t] = {1e-8, 1e-3, 0.1, 1.0, 3.0, 1e3, 1e8};
  return ts[i];
}
// v - mu on the grid: zero, the threshold k on both sides (and just across it), large
KOKKOS_INLINE_FUNCTION double offset(int i, double k) {
  const double us[n_u] = {0.0, k, -k, k * (1.0 + 1e-12), -k * (1.0 + 1e-12), 0.5 * k, 0.37, -2.5, 1e8};
  return us[i];
}

template <class D>
KOKKOS_INLINE_FUNCTION void check_law(const D& d, double* worst, bool with_golden = true) {
  double r[2];
  for (int i = 0; i < n_t; ++i)
    for (int j = 0; j < n_u; ++j) {
      const double t = step(i);
      check_point(d, d.mu + offset(j, t * kink_weight(d)), t, with_golden, r);
      worst[0] = Kokkos::fmax(worst[0], r[0]);
      worst[1] = Kokkos::fmax(worst[1], r[1]);
    }
}

// the lasso penalty of a product with theta by block, mu_0 != 0, against laplace1's prox_y
KOKKOS_INLINE_FUNCTION double check_penalty(double t) {
  const lasso_pen pen(product<uam<ld::laplace1_distrib>>(
      ld::independent_product_component<uam<ld::laplace1_distrib>>{{}, 3, 3, nullptr}));
  const double theta[3] = {0.5, 1.0, 2.0}, mu0[3] = {0.1, -0.2, 0.3}, v[3] = {0.9, -0.25, -1.5};
  double w[16], out[3];
  const lasso_pen p2(pen.distribution, mu0);
  const auto st = p2.theta_state(theta, w);
  p2.prox_beta(st, v, t, out, w + p2.state_size());
  double e = 0.0;
  for (int g = 0; g < 3; ++g) e = Kokkos::fmax(e, Kokkos::fabs(out[g] - ld::laplace1_distrib{mu0[g], theta[g]}.prox_y(v[g], t)));
  return e;
}

// worst[0..1] over laplace1, [2..3] laplace2, [4..5] enet, [6] the penalty, [7] NaN and inf handling
void run_kernel(double* host) {
  Kokkos::View<double[8]> res("prox_res");
  Kokkos::parallel_for(
      "prox_test", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        double w[6] = {0, 0, 0, 0, 0, 0};
        const double scales[3] = {0.3, 1.0, 7.0};
        for (int s = 0; s < 3; ++s) {
          check_law(ld::laplace1_distrib{-0.4, scales[s]}, w);
          check_law(ld::laplace2_distrib{0.25, scales[s]}, w + 2);
          // the ridge and lasso ends alpha = 0 and 1 lie outside the open domain of alpha,
          // where logpdf is not finite: the formula is checked there by optimality only
          check_law(ld::enet_distrib{0.6, scales[s], 0.0}, w + 4, false);
          check_law(ld::enet_distrib{0.6, scales[s], 1.0}, w + 4, false);
          check_law(ld::enet_distrib{0.6, scales[s], 1e-9}, w + 4);
          check_law(ld::enet_distrib{0.6, scales[s], 0.5}, w + 4);
          check_law(ld::enet_distrib{0.6, scales[s], 1.0 - 1e-9}, w + 4);
        }
        for (int i = 0; i < 6; ++i) res(i) = w[i];
        res(6) = Kokkos::fmax(check_penalty(0.3), check_penalty(5.0));
        const double nan = Kokkos::Experimental::quiet_NaN_v<double>, inf = Kokkos::Experimental::infinity_v<double>;
        const ld::laplace1_distrib l1{0.0, 1.0};
        const ld::enet_distrib en{0.0, 1.0, 0.5};
        const bool ok = Kokkos::isnan(l1.prox_y(nan, 1.0)) && Kokkos::isnan(en.prox_y(nan, 1.0)) &&
                        l1.prox_y(inf, 1.0) == inf && l1.prox_y(-inf, 1.0) == -inf && en.prox_y(inf, 1.0) == inf;
        res(7) = ok ? 0.0 : 1.0;
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
  for (int i = 0; i < 8; ++i) host[i] = h(i);
}

}  // namespace prox_test

TEST_CASE("prox_y of laplace1, laplace2, enet: optimality and direct minimization, in a kernel") {
  using namespace prox_test;
  double r[8];
  run_kernel(r);
  MESSAGE("optimality ", r[0], " ", r[2], " ", r[4], "; golden ", r[1], " ", r[3], " ", r[5], "; penalty ", r[6]);
  CHECK(r[0] < 4e-16);
  CHECK(r[2] < 4e-16);
  CHECK(r[4] < 4e-16);
  CHECK(r[1] < 1e-7);
  CHECK(r[3] < 1e-7);
  CHECK(r[5] < 1e-7);
  CHECK(r[6] == 0.0);
  CHECK(r[7] == 0.0);
}

TEST_CASE("penalty prox: the product composes its components' prox_y, with mu_0") {
  using namespace prox_test;
  // a heterogeneous product: enet on 3 coefficients in 2 groups, laplace2 on 2 with a shared rate
  const int groups[3] = {0, 1, 0};
  const product<uam<ld::enet_distrib>, uam<ld::laplace2_distrib>> m(
      ld::independent_product_component<uam<ld::enet_distrib>>{{}, 3, 2, groups},
      ld::independent_product_component<uam<ld::laplace2_distrib>>{{}, 2, 1, nullptr});
  // theta param-major: lambda_1, lambda_2, alpha_1, alpha_2, then the laplace2 rate
  const double theta[5] = {2.0, 0.7, 0.25, 0.9, 1.5};
  const double mu0[5] = {0.1, -0.3, 0.0, 0.4, -0.2}, v[5] = {1.3, -0.35, -2.0, 0.0, 3.0};
  for (const bool located : {false, true}) {
    const mixed_pen pen(m, located ? mu0 : nullptr);
    std::vector<double> w(pen.state_size() + pen.scratch_size());
    const auto st = pen.theta_state(theta, w.data());
    for (const double t : {0.05, 0.5, 4.0}) {
      double out[5];
      pen.prox_beta(st, v, t, out, w.data() + pen.state_size());
      for (int g = 0; g < 5; ++g) {
        const double m0 = located ? mu0[g] : 0.0;
        const double e = g < 3 ? ld::enet_distrib{m0, theta[groups[g]], theta[2 + groups[g]]}.prox_y(v[g], t)
                               : ld::laplace2_distrib{m0, theta[4]}.prox_y(v[g], t);
        CHECK(out[g] == e);
      }
    }
  }
}

TEST_CASE("weak convexity: pen'' on a grid, convex laws and student_t2") {
  namespace ld = laplace::distrib;
  // pen'' = -d2/dy2 log f, off the kink at y = mu
  auto least = [](const auto& d, double width) {
    double m = std::numeric_limits<double>::infinity();
    for (double z = 1e-3 * width; z < 20.0 * width; z += 1e-4 * width)
      for (const double s : {-1.0, 1.0}) m = std::fmin(m, -d.template logpdf_dy_dk<2, 0>(d.mu + s * z)[0]);
    return m;
  };
  CHECK(least(ld::laplace1_distrib{0.2, 1.5}, 1.5) >= 0.0);
  CHECK(least(ld::laplace2_distrib{0.2, 1.5}, 1.0) >= 0.0);
  CHECK(least(ld::enet_distrib{0.2, 1.5, 0.3}, 1.0) >= 0.0);
  CHECK(least(ld::pseudohuber_distrib{0.2, 1.5, 0.7}, 1.5) >= 0.0);
  CHECK(least(ld::gaussian1_distrib{0.2, 1.5}, 1.5) >= 0.0);
  CHECK(least(ld::gaussian2_distrib{0.2, 2.25}, 1.5) >= 0.0);
  // student_t2: rho = (nu + 1) / (8 (nu - 2) sigma^2), at |y - mu| = sqrt(3 (nu - 2)) sigma
  const prox_test::uam<ld::student_t2_distrib> one{};
  const double sigma = 1.5, nu = 5.0, th[2] = {sigma, nu};
  const double rho = laplace::penalty::distrib_penalty<prox_test::uam<ld::student_t2_distrib>>(one).weak_convexity(th);
  const ld::student_t2_distrib d{0.0, sigma, nu};
  const double m = least(d, sigma);
  MESSAGE("student_t2: rho ", rho, ", max(-pen'') ", -m);
  CHECK(std::fabs(-m - rho) < 1e-7 * rho);
  const double z = std::sqrt(3.0 * (nu - 2.0)) * sigma;
  CHECK(std::fabs(d.logpdf_dy_dk<2, 0>(z)[0] - rho) < 1e-14 * rho);
}
