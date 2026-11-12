// lower_bounded and upper_bounded (include/laplace/bijector/detail/scalar_shifted.hpp):
// g(v) = log(v - lower), h(u) = lower + e^u on (lower, inf), and g(v) = -log(upper - v),
// h(u) = upper - e^(-u) on (-inf, upper). Checked against the mpmath references
// (shared check in param_common.hpp), then exact values, the behaviour next to the bound,
// the partials in the bound, the distances, invalid bounds, inversion and the constructors.
#include <doctest/doctest.h>

#include "param_common.hpp"

#include <cmath>
#include <limits>

namespace lb = laplace::bijector;
using bijector_param_test::map_t;

namespace {

const double inf = std::numeric_limits<double>::infinity();
const double nan_value = std::numeric_limits<double>::quiet_NaN();

}  // namespace

TEST_CASE("bijector lower_bounded: accuracy against mpmath") {
  // bound[which][order]: 1.5 times the measured maximum ulpC (-1: no rows), see the report
  // measured 2026-11-12 (CPU, OpenMP build): max ulpC 0.68 forward (k = 4), 0.73 inverse
  // (k = 0), 0.27 inverse k >= 1, 0.27 distance; the inverse log-Jacobian is exact.
  // Bounds: 1.5 times the maximum, at least 0.5.
  static const double bound[6][7] = {
      {0.7, 0.5, 0.7, 0.55, 1.05, 0.65, 1.0},  // forward
      {1.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},     // inverse
      {0.7, 0.5, 0.7, 0.55, 1.05, 0.65, 1.0},  // forward log-Jacobian
      {0.5, -1, -1, -1, -1, -1, -1},           // inverse log-Jacobian (only order 0 has rows)
      {-1, -1, -1, -1, -1, -1, -1},            // distance to upper bound: +inf, no rows
      {0.5, -1, -1, -1, -1, -1, -1},           // distance to lower bound: e^u (order 0)
  };
  bijector_param_test::check_map("lower_bounded", map_t::lower_bounded, bound);
}

TEST_CASE("bijector upper_bounded: accuracy against mpmath") {
  // measured 2026-11-12 (CPU, OpenMP build): max ulpC 0.68 forward (k = 4), 0.47 inverse
  // (k = 0), 0.19 inverse k >= 1, 0.19 distance; the inverse log-Jacobian is exact.
  // Bounds: 1.5 times the maximum, at least 0.5.
  static const double bound[6][7] = {
      {0.7, 0.5, 0.7, 0.55, 1.05, 0.65, 1.0},  // forward
      {0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},     // inverse
      {0.7, 0.5, 0.7, 0.55, 1.05, 0.65, 1.0},  // forward log-Jacobian
      {0.5, -1, -1, -1, -1, -1, -1},           // inverse log-Jacobian (only order 0 has rows)
      {0.5, -1, -1, -1, -1, -1, -1},           // distance to upper bound: e^(-u) (order 0)
      {-1, -1, -1, -1, -1, -1, -1},            // distance to lower bound: +inf, no rows
  };
  bijector_param_test::check_map("upper_bounded", map_t::upper_bounded, bound);
}

TEST_CASE("bijector lower_bounded: exact values and the bound") {
  // lower = 1: g(3) = log 2, g' = 1/2, g'' = -1/4, g''' = 2/8
  const auto b = lb::scalar_bijector::lower_bounded(1.0);
  CHECK(b.map == lb::scalar_bijector_map::lower_bounded);
  CHECK(b.lower == 1.0);
  CHECK(b.forward(3.0) == doctest::Approx(std::log(2.0)).epsilon(1e-15));
  CHECK(b.forward_d1(3.0) == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(b.forward_d2(3.0) == doctest::Approx(-0.25).epsilon(1e-15));
  CHECK(b.forward_d3(3.0) == doctest::Approx(0.25).epsilon(1e-15));
  // h(0) = 1 + e^0 = 2, h' = h'' = 1
  CHECK(b.inverse(0.0) == 2.0);
  CHECK(b.inverse_d1(0.0) == 1.0);
  CHECK(b.inverse_d6(0.0) == 1.0);
  CHECK(b.inverse_log_jacobian(-2.0) == -2.0);     // log h'(u) = u
  CHECK(b.inverse_log_jacobian_d1(-2.0) == 1.0);
  CHECK(b.inverse_log_jacobian_d2(-2.0) == 0.0);
  CHECK(b.forward_log_jacobian(3.0) == doctest::Approx(-std::log(2.0)).epsilon(1e-15));  // -log(v - 1)
  CHECK(b.forward_log_jacobian_d1(3.0) == doctest::Approx(-0.5).epsilon(1e-15));
  // the bound itself gives the limits of the closed domain
  CHECK(b.forward(1.0) == -inf);
  CHECK(b.forward_log_jacobian(1.0) == inf);
  CHECK_FALSE(b.is_decreasing());
  CHECK(b.forward_domain_lower() == 1.0);
  CHECK(b.forward_domain_upper() == inf);
  CHECK(b.inverse_domain_lower() == -inf);
  CHECK(b.inverse_domain_upper() == inf);
}

TEST_CASE("bijector lower_bounded: exact next to a large bound") {
  // lower = 1e8 and v = 1e8 + 2^-20 are both doubles, so v - lower = 2^-20 exactly
  // and g(v) = -20 log 2; a formula that formed log(v) - log(lower) would lose it
  const double lower = 1e8, t = std::ldexp(1.0, -20);
  const auto b = lb::scalar_bijector::lower_bounded(lower);
  CHECK(b.forward(lower + t) == doctest::Approx(-20.0 * std::log(2.0)).epsilon(1e-15));
  CHECK(b.forward_d1(lower + t) == doctest::Approx(std::ldexp(1.0, 20)).epsilon(1e-15));
  // the distance to the lower bound is e^u exactly, for any u
  for (double u : {-700.0, -40.0, -1.0, 0.0, 3.5}) {
    CHECK(b.inverse_distance_lower(u) == doctest::Approx(std::exp(u)).epsilon(1e-14));
    CHECK(b.inverse_distance_upper(u) == inf);
  }
}

TEST_CASE("bijector lower_bounded: partials in the bound") {
  // g = log(v - p1) at v = 3, p1 = 1: d/dp1 = -1/2, d2/(dv dp1) = 1/4, d2/dp1^2 = -1/4
  const auto b = lb::scalar_bijector::lower_bounded(1.0);
  const auto f1 = b.forward_partials_d1(3.0);
  CHECK(f1.c[0] == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(f1.c[1] == doctest::Approx(-0.5).epsilon(1e-15));
  const auto f2 = b.forward_partials_d2(3.0);
  CHECK(f2.c[0] == doctest::Approx(-0.25).epsilon(1e-15));
  CHECK(f2.c[1] == doctest::Approx(0.25).epsilon(1e-15));
  CHECK(f2.c[2] == doctest::Approx(-0.25).epsilon(1e-15));
  // h = p1 + e^u: d/dp1 = 1 (order 1), nothing else at order 2
  const auto i1 = b.inverse_partials_d1(0.0);
  CHECK(i1.c[0] == 1.0);
  CHECK(i1.c[1] == 1.0);
  const auto i2 = b.inverse_partials_d2(0.0);
  CHECK(i2.c[0] == 1.0);
  CHECK(i2.c[1] == 0.0);
  CHECK(i2.c[2] == 0.0);
}

TEST_CASE("bijector lower_bounded: invalid bound gives NaN") {
  for (const auto& b : {lb::scalar_bijector::lower_bounded(nan_value), lb::scalar_bijector::lower_bounded(inf),
                        lb::scalar_bijector::lower_bounded(-inf)}) {
    CHECK(std::isnan(b.forward(2.0)));
    CHECK(std::isnan(b.inverse(0.0)));
    CHECK(std::isnan(b.forward_d1(2.0)));
    CHECK(std::isnan(b.inverse_log_jacobian(0.0)));
    CHECK(std::isnan(b.inverse_distance_lower(0.0)));
  }
}

TEST_CASE("bijector upper_bounded: exact values and the bound") {
  // upper = 1: g(-1) = -log 2, g' = 1/2, g'' = 1/4, increasing
  const auto b = lb::scalar_bijector::upper_bounded(1.0);
  CHECK(b.map == lb::scalar_bijector_map::upper_bounded);
  CHECK(b.upper == 1.0);
  CHECK(b.forward(-1.0) == doctest::Approx(-std::log(2.0)).epsilon(1e-15));
  CHECK(b.forward_d1(-1.0) == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(b.forward_d2(-1.0) == doctest::Approx(0.25).epsilon(1e-15));
  CHECK(b.forward_d3(-1.0) == doctest::Approx(0.25).epsilon(1e-15));  // 2!/s^3 with s = 2
  CHECK(b.is_decreasing() == false);
  // h(0) = 1 - e^0 = 0, h' = e^0 = 1, h'' = -1, h''' = 1
  CHECK(b.inverse(0.0) == 0.0);
  CHECK(b.inverse_d1(0.0) == 1.0);
  CHECK(b.inverse_d2(0.0) == -1.0);
  CHECK(b.inverse_d3(0.0) == 1.0);
  CHECK(b.inverse_log_jacobian(2.0) == -2.0);  // log h'(u) = -u
  CHECK(b.inverse_log_jacobian_d1(2.0) == -1.0);
  CHECK(b.inverse_log_jacobian_d2(2.0) == 0.0);
  // the bound itself: g(1) = +inf, h(u) -> 1 as u -> inf
  CHECK(b.forward(1.0) == inf);
  CHECK(b.forward_domain_lower() == -inf);
  CHECK(b.forward_domain_upper() == 1.0);
  CHECK(b.inverse_domain_lower() == -inf);
  CHECK(b.inverse_domain_upper() == inf);
}

TEST_CASE("bijector upper_bounded: exact next to a large bound") {
  // upper = -1e8 and v = -1e8 - 2^-20: p1 - v = 2^-20 exactly
  const double upper = -1e8, t = std::ldexp(1.0, -20);
  const auto b = lb::scalar_bijector::upper_bounded(upper);
  CHECK(b.forward(upper - t) == doctest::Approx(20.0 * std::log(2.0)).epsilon(1e-15));
  CHECK(b.forward_d1(upper - t) == doctest::Approx(std::ldexp(1.0, 20)).epsilon(1e-15));
  // the distance to the upper bound is e^(-u) exactly, for any u
  for (double u : {-3.5, -1.0, 0.0, 40.0, 700.0}) {
    CHECK(b.inverse_distance_upper(u) == doctest::Approx(std::exp(-u)).epsilon(1e-14));
    CHECK(b.inverse_distance_lower(u) == inf);
  }
}

TEST_CASE("bijector upper_bounded: partials in the bound") {
  // g = -log(p1 - v) at v = -1, p1 = 1: s = 2, d/dp1 = -1/s = -1/2, d/dv = 1/2,
  // d2/(dv dp1) = -1/s^2 = -1/4, d2/dp1^2 = 1/s^2 = 1/4
  const auto b = lb::scalar_bijector::upper_bounded(1.0);
  const auto f1 = b.forward_partials_d1(-1.0);
  CHECK(f1.c[0] == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(f1.c[1] == doctest::Approx(-0.5).epsilon(1e-15));
  const auto f2 = b.forward_partials_d2(-1.0);
  CHECK(f2.c[0] == doctest::Approx(0.25).epsilon(1e-15));
  CHECK(f2.c[1] == doctest::Approx(-0.25).epsilon(1e-15));
  CHECK(f2.c[2] == doctest::Approx(0.25).epsilon(1e-15));
  // h = p1 - e^(-u): d/dp1 = 1 at order 1, nothing else at order 2
  const auto i1 = b.inverse_partials_d1(0.0);
  CHECK(i1.c[0] == 1.0);
  CHECK(i1.c[1] == 1.0);
  const auto i2 = b.inverse_partials_d2(0.0);
  CHECK(i2.c[0] == -1.0);
  CHECK(i2.c[1] == 0.0);
  CHECK(i2.c[2] == 0.0);
}

TEST_CASE("bijector upper_bounded: invalid bound gives NaN") {
  for (const auto& b : {lb::scalar_bijector::upper_bounded(nan_value), lb::scalar_bijector::upper_bounded(inf),
                        lb::scalar_bijector::upper_bounded(-inf)}) {
    CHECK(std::isnan(b.forward(-2.0)));
    CHECK(std::isnan(b.inverse(0.0)));
    CHECK(std::isnan(b.forward_d1(-2.0)));
    CHECK(std::isnan(b.inverse_log_jacobian(0.0)));
    CHECK(std::isnan(b.inverse_distance_upper(0.0)));
  }
}

TEST_CASE("bijector shifted: inversion swaps the two directions") {
  // the arguments of b0.forward must lie in its domain (NaN is not equal to itself)
  const auto lo = lb::scalar_bijector::lower_bounded(-0.5);
  const auto up = lb::scalar_bijector::upper_bounded(2.0);
  auto lo_inv = lo;
  lo_inv.inverted = true;
  auto up_inv = up;
  up_inv.inverted = true;
  for (double u : {-2.0, 0.0, 0.75, 3.0}) {  // any u is in the domain of the inverse
    CHECK(lo_inv.forward(u) == lo.inverse(u));
    CHECK(lo_inv.forward_d2(u) == lo.inverse_d2(u));
    CHECK(lo_inv.forward_log_jacobian(u) == lo.inverse_log_jacobian(u));
    CHECK(up_inv.forward(u) == up.inverse(u));
    CHECK(up_inv.forward_d2(u) == up.inverse_d2(u));
    CHECK(up_inv.forward_log_jacobian(u) == up.inverse_log_jacobian(u));
  }
  for (double v : {0.0, 0.75, 3.0}) {  // in (-0.5, inf)
    CHECK(lo_inv.inverse(v) == lo.forward(v));
    CHECK(lo_inv.inverse_log_jacobian(v) == lo.forward_log_jacobian(v));
  }
  for (double v : {-2.0, 0.0, 1.5}) {  // in (-inf, 2)
    CHECK(up_inv.inverse(v) == up.forward(v));
    CHECK(up_inv.inverse_log_jacobian(v) == up.forward_log_jacobian(v));
  }
}
