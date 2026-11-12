// yeo_johnson (include/laplace/bijector/detail/scalar_yeo_johnson.hpp) against the
// mpmath references (tools/reference/bijector_yeo_johnson.py, through param_common.hpp),
// and exact checks: the special values, the kink at 0, the image for each lambda,
// NaN for an invalid lambda, inversion, the distances, the constructor and the kinks.
// Bounds: 1.5 times the errors measured on the gcc/OpenMP build on 2026-11-12 (the table
// with LAPLACE_PRINT_ERRORS=1).
#include <doctest/doctest.h>

#include "param_common.hpp"

#include <laplace/bijector/scalar.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

namespace lb = laplace::bijector;
using bmap = lb::scalar_bijector_map;

const double inf = std::numeric_limits<double>::infinity();
const double nan_v = std::numeric_limits<double>::quiet_NaN();

lb::scalar_bijector yj(double lambda) { return lb::scalar_bijector::yeo_johnson(lambda); }

// |a - b| <= rel max(1, |a|, |b|)
bool close(double a, double b, double rel = 1e-14) {
  return std::fabs(a - b) <= rel * std::max({1.0, std::fabs(a), std::fabs(b)});
}

}  // namespace

TEST_CASE("bijector yeo_johnson: accuracy against mpmath") {
  // 1.5 times the measured maxima of ulpC (LAPLACE_PRINT_ERRORS=1), gcc/OpenMP, 2026-11-12
  static const double bound[6][7] = {
      {31, 39, 26, 33, 29, 28, 36},  // forward
      {3, 4, 6, 9, 30, 60, 22},      // inverse (order 5: 39.9 ulpC on CUDA)
      {2, 2, 2, 4, 4, 4, 5},         // forward log-Jacobian
      {3, 2, 7, 10, 9, 18, 70},      // inverse log-Jacobian
      {-1, -1, -1, -1, -1, -1, -1},  // distances: +inf, no rows
      {-1, -1, -1, -1, -1, -1, -1},
  };
  bijector_param_test::check_map("yeo_johnson", bmap::yeo_johnson, bound);
}

TEST_CASE("bijector yeo_johnson: special values and the kink at 0") {
  for (double lambda : {-0.5, 0.0, 0.4, 1.0, 1.6, 2.0, 2.5}) {
    const auto b = yj(lambda);
    CHECK(b.forward(0.0) == 0.0);
    CHECK(b.forward_d1(0.0) == 1.0);
    CHECK(close(b.forward_d2(0.0), lambda - 1.0));
    // the right-hand third derivative at 0 is (lambda - 1)(lambda - 2)
    CHECK(close(b.forward_d3(0.0), (lambda - 1.0) * (lambda - 2.0)));
    CHECK(b.inverse(0.0) == 0.0);
    CHECK(b.inverse_d1(0.0) == 1.0);
    // the left-hand third derivative (v < 0) is lambda (lambda - 1): they differ unless lambda = 1
    CHECK(close(b.forward_d3(-1e-300), lambda * (lambda - 1.0)));
    CHECK(b.forward_log_jacobian(0.0) == 0.0);
  }
  // lambda = 1 is the identity
  const auto id = yj(1.0);
  for (double v : {-3.0, -0.2, 0.0, 0.7, 40.0}) CHECK(close(id.forward(v), v, 1e-13));
  CHECK(close(id.forward_log_jacobian(0.7), 0.0, 1e-15));
  CHECK(id.forward_log_jacobian_d1(0.7) == 0.0);
  CHECK(id.forward_d2(0.7) == 0.0);
  CHECK(id.forward_d3(-0.7) == 0.0);
  // lambda = 0: g(v) = log(1 + v) on v >= 0, g(v) = v - v^2/2 on v < 0; h(u) = e^u - 1 on u >= 0
  const auto b0 = yj(0.0);
  for (double v : {1e-12, 0.3, 5.0}) CHECK(close(b0.forward(v), std::log1p(v), 1e-14));
  for (double v : {-1e-12, -0.3, -0.9}) CHECK(close(b0.forward(v), v - 0.5 * v * v, 1e-14));
  for (double u : {1e-12, 0.3, 4.0}) CHECK(close(b0.inverse(u), std::expm1(u), 1e-14));
  // lambda = 2: g(v) = -log(1 - v) on v < 0, h(u) = 1 - e^(-u) on u < 0
  const auto b2 = yj(2.0);
  for (double v : {-1e-12, -0.3, -0.9}) CHECK(close(b2.forward(v), -std::log1p(-v), 1e-14));
  for (double u : {-1e-12, -0.3, -3.0}) CHECK(close(b2.inverse(u), -std::expm1(-u), 1e-14));
  // kinks: one at 0, continuous to the second derivative, none at lambda = 1
  for (double lambda : {-0.5, 0.0, 0.4, 1.6, 2.5}) {
    const auto k = yj(lambda).forward_kinks();
    CHECK(k.count == 1);
    CHECK(k.point[0] == 0.0);
    CHECK(k.continuous_order == 2);
    const auto ki = yj(lambda).inverse_kinks();
    CHECK(ki.count == 1);
    CHECK(ki.point[0] == 0.0);
    CHECK(ki.continuous_order == 2);
  }
  CHECK(id.forward_kinks().count == 0);
  CHECK(id.inverse_kinks().count == 0);
}

TEST_CASE("bijector yeo_johnson: image, domain, invalid parameters") {
  // the forward domain is R for every lambda; the image (domain of h) depends on lambda
  for (double lambda : {-2.0, -0.5, 0.0, 1.0, 2.0, 2.5, 4.0}) {
    CHECK(yj(lambda).forward_domain_lower() == -inf);
    CHECK(yj(lambda).forward_domain_upper() == inf);
  }
  CHECK(yj(1.0).inverse_domain_lower() == -inf);
  CHECK(yj(1.0).inverse_domain_upper() == inf);
  CHECK(yj(0.4).inverse_domain_lower() == -inf);
  CHECK(yj(0.4).inverse_domain_upper() == inf);
  // lambda < 0: the positive branch is bounded above by -1/lambda
  CHECK(yj(-0.5).inverse_domain_upper() == 2.0);
  CHECK(yj(-0.5).inverse_domain_lower() == -inf);
  CHECK(std::isnan(yj(-0.5).inverse(2.5)));
  CHECK(std::isfinite(yj(-0.5).inverse(1.99)));
  // lambda > 2: the negative branch is bounded below by 1/(2 - lambda)
  CHECK(yj(2.5).inverse_domain_lower() == -2.0);
  CHECK(yj(2.5).inverse_domain_upper() == inf);
  CHECK(std::isnan(yj(2.5).inverse(-2.5)));
  CHECK(std::isfinite(yj(2.5).inverse(-1.99)));
  // the limits of the image are approached: g(v) -> -1/lambda for lambda < 0, and at large v
  CHECK(std::fabs(yj(-0.5).forward(1e30) - 2.0) < 1e-12);
  CHECK(std::fabs(yj(2.5).forward(-1e30) + 2.0) < 1e-12);
  // invalid lambda: NaN everywhere
  const auto bad = yj(nan_v);
  CHECK(std::isnan(bad.forward(0.3)));
  CHECK(std::isnan(bad.forward_d2(-0.3)));
  CHECK(std::isnan(bad.forward_log_jacobian(0.3)));
  CHECK(std::isnan(bad.inverse(0.3)));
  CHECK(std::isnan(bad.inverse_d3(-0.3)));
  CHECK(std::isnan(bad.inverse_log_jacobian_d1(0.3)));
  CHECK(std::isnan(bad.forward_domain_upper()));
  CHECK(std::isnan(bad.inverse_domain_lower()));
  CHECK(std::isnan(bad.forward_partials_d2(0.3).c[0]));
  CHECK(std::isnan(bad.inverse_partials_d2(0.3).c[1]));
  CHECK(std::isnan(bad.inverse_distance_upper(0.3)));
}

TEST_CASE("bijector yeo_johnson: inversion, distances, direction, constructor") {
  for (double lambda : {-0.5, 0.0, 0.4, 1.6, 2.5}) {
    const auto b = yj(lambda);
    for (double v : {-4.0, -0.3, 0.0, 0.2, 3.0}) {
      const double u = b.forward(v);
      CHECK(close(b.inverse(u), v, 1e-12));
    }
    // an inverted bijector swaps the two directions
    lb::scalar_bijector r = b;
    r.inverted = true;
    CHECK(r.forward(0.7) == b.inverse(0.7));
    CHECK(r.inverse(-0.6) == b.forward(-0.6));
    CHECK(r.forward_d2(-0.6) == b.inverse_d2(-0.6));
    CHECK(r.inverse_log_jacobian(0.7) == b.forward_log_jacobian(0.7));
    // the range of h is R: the distances to its bounds are +inf
    CHECK(b.inverse_distance_upper(0.3) == inf);
    CHECK(b.inverse_distance_lower(0.3) == inf);
    CHECK(!b.is_decreasing());
  }
  CHECK(!yj(-0.5).is_decreasing());
  CHECK(!yj(2.5).is_decreasing());
  // the constructor and the parameter
  const auto c = lb::scalar_bijector::yeo_johnson(0.5);
  CHECK(c.map == bmap::yeo_johnson);
  CHECK(c.exponent == 0.5);
  CHECK(c.parameter_count() == 1);
  CHECK(!c.inverted);
}
