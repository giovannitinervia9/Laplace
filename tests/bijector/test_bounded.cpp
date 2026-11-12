// bounded (scalar_bounded.hpp): g(v) = log(v - a) - log(b - v) on (a, b), h(u) = a + (b - a) expit(u)
// against the mpmath references (tools/reference/bijector_bounded.py), through the shared check of
// param_common.hpp, and exact checks of the special values, the domains, the distances, inversion,
// the NaN of invalid parameters and the static constructor.
#include <doctest/doctest.h>

#include "param_common.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace lb = laplace::bijector;
using bmap = lb::scalar_bijector_map;

// bound[which][order]: the bound on ulpC of quantity `which` at total order `order` (-1 where the
// reference has no rows): about 1.5 times the maxima measured on the gcc/OpenMP build on 2026-11-12.
static const double bounded_bound[6][7] = {
    {0.75, 0.9, 1.35, 1.25, 1.25, 1.7, 1.25},    // forward g (measured 0.49 0.59 0.89 0.82 0.82 1.10 0.83)
    {1.25, 1.5, 1.25, 3.75, 5.3, 7.0, 10.7},     // inverse h (measured 0.83 0.98 0.81 2.5 3.51 4.69 7.09)
    {0.85, 0.9, 1.15, 1.25, 1.25, 1.7, 1.25},    // forward log|g'| (measured 0.55 0.58 0.76 0.82 0.82 1.10 0.83)
    {1.1, 1.1, 1.25, 2.8, 5.3, 5.1, 10.7},       // inverse log|h'| (measured 0.71 0.72 0.81 1.87 3.51 3.41 7.09)
    {1.1, -1, -1, -1, -1, -1, -1},               // to upper (order 0 only; measured 0.72)
    {0.9, -1, -1, -1, -1, -1, -1},               // to lower (order 0 only; measured 0.58)
};

TEST_CASE("bounded: the references (values, derivatives and partials to order 6)") {
  bijector_param_test::check_map("bounded", bmap::bounded, bounded_bound);
}

TEST_CASE("bounded: special values, domains and distances") {
  const auto b = lb::scalar_bijector::bounded(-1.0, 3.0);
  // the midpoint v = 1: g = 0, g' = (b - a)/((v - a)(b - v)) = 4/4 = 1, h(0) = 1
  CHECK(b.forward(1.0) == 0.0);
  CHECK(b.forward_d1(1.0) == doctest::Approx(1.0));
  CHECK(b.inverse(0.0) == 1.0);
  CHECK(b.inverse_d1(0.0) == doctest::Approx(1.0));  // (b - a) expit'(0) = 4 * 1/4
  CHECK(b.forward_log_jacobian(1.0) == doctest::Approx(0.0).epsilon(1e-15));
  CHECK(b.inverse_log_jacobian(0.0) == doctest::Approx(std::log(4.0 * 0.25)).epsilon(1e-15));
  // the bounds of the domains
  CHECK(b.forward_domain_lower() == -1.0);
  CHECK(b.forward_domain_upper() == 3.0);
  CHECK(b.inverse_domain_lower() == -std::numeric_limits<double>::infinity());
  CHECK(b.inverse_domain_upper() == std::numeric_limits<double>::infinity());
  // g at the bounds: -inf at a, +inf at b; outside the closed domain NaN
  CHECK(b.forward(-1.0) == -std::numeric_limits<double>::infinity());
  CHECK(b.forward(3.0) == std::numeric_limits<double>::infinity());
  CHECK(std::isnan(b.forward(3.5)));
  CHECK(std::isnan(b.forward(-1.5)));
  // the distances: (b - a) expit(-u) and (b - a) expit(u); they add up to b - a
  for (double u : {-40.0, -3.0, -0.5, 0.0, 0.5, 3.0, 40.0}) {
    const double up = b.inverse_distance_upper(u), lo = b.inverse_distance_lower(u);
    CHECK(up + lo == doctest::Approx(4.0).epsilon(1e-15));
    CHECK(up > 0.0);
    CHECK(lo > 0.0);
    CHECK(b.inverse_distance_upper(u) == doctest::Approx(4.0 / (1.0 + std::exp(u))).epsilon(1e-14));
  }
  CHECK(b.inverse_distance_upper(0.0) == 2.0);
  CHECK(b.inverse_distance_lower(0.0) == 2.0);
  // near the bound: 1 - h(u) of the unit interval is expit(-u), exact where 1 - expit(u) rounds to 0
  const auto unit = lb::scalar_bijector::bounded(0.0, 1.0);
  CHECK(unit.inverse_distance_upper(40.0) == doctest::Approx(std::exp(-40.0)).epsilon(1e-14));
  CHECK(unit.inverse(40.0) == 1.0);  // h saturates to the bound
  CHECK(unit.inverse_distance_lower(-40.0) == doctest::Approx(std::exp(-40.0)).epsilon(1e-14));
}

TEST_CASE("bounded: partial derivatives at a point, by hand") {
  // (a, b) = (0, 1), v = 1/4: p = 1/4, q = 3/4, A = 4, B = 4/3
  const auto b = lb::scalar_bijector::bounded(0.0, 1.0);
  const auto g1 = b.forward_partials_d1(0.25);
  CHECK(g1.c[0] == doctest::Approx(4.0 + 4.0 / 3.0).epsilon(1e-15));  // A + B
  CHECK(g1.c[lb::bijector_partials<1>::index(1, 0)] == doctest::Approx(-4.0).epsilon(1e-15));  // -A = d_a g
  CHECK(g1.c[lb::bijector_partials<1>::index(0, 1)] == doctest::Approx(-4.0 / 3.0).epsilon(1e-15));  // -B = d_b g
  // the inverse at u = 0: d_u h = (b - a) expit'(0) = 1/4, d_a h = 1 - expit(0) = 1/2, d_b h = 1/2
  const auto h1 = b.inverse_partials_d1(0.0);
  CHECK(h1.c[0] == doctest::Approx(0.25).epsilon(1e-15));
  CHECK(h1.c[lb::bijector_partials<1>::index(1, 0)] == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(h1.c[lb::bijector_partials<1>::index(0, 1)] == doctest::Approx(0.5).epsilon(1e-15));
  // second partials of h in (a, b) alone are zero
  const auto h2 = b.inverse_partials_d2(0.7);
  CHECK(h2.c[lb::bijector_partials<2>::index(2, 0)] == 0.0);
  CHECK(h2.c[lb::bijector_partials<2>::index(1, 1)] == 0.0);
  CHECK(h2.c[lb::bijector_partials<2>::index(0, 2)] == 0.0);
  // the log-Jacobian of g does not depend on the sign of the derivative: d_b log|g'| = 1/s - B
  const auto lj1 = b.forward_log_jacobian_partials_d1(0.25);
  CHECK(lj1.c[lb::bijector_partials<1>::index(0, 1)] == doctest::Approx(1.0 - 4.0 / 3.0).epsilon(1e-15));
}

TEST_CASE("bounded: inversion, monotonicity and the static constructor") {
  const auto b = lb::scalar_bijector::bounded(2.0, 5.0);
  CHECK(b.map == bmap::bounded);
  CHECK(b.lower == 2.0);
  CHECK(b.upper == 5.0);
  CHECK_FALSE(b.is_decreasing());
  // the inverted bijector swaps the two directions
  auto ib = b;
  ib.inverted = true;
  for (double v : {2.1, 2.5, 3.9, 4.99}) {
    CHECK(ib.inverse(v) == doctest::Approx(b.forward(v)).epsilon(1e-14));
    CHECK(ib.forward(v) == doctest::Approx(b.inverse(v)).epsilon(1e-14));
    CHECK(ib.inverse_d2(v) == doctest::Approx(b.forward_d2(v)).epsilon(1e-13));
    CHECK(ib.inverse_log_jacobian(v) == doctest::Approx(b.forward_log_jacobian(v)).epsilon(1e-13));
  }
  // round trip
  for (double v : {2.001, 2.5, 3.0, 4.5, 4.999}) {
    CHECK(b.inverse(b.forward(v)) == doctest::Approx(v).epsilon(1e-13));
  }
  // the range: h maps the whole line onto (a, b)
  CHECK(b.inverse(-800.0) == 2.0);
  CHECK(b.inverse(800.0) == 5.0);
  CHECK(b.inverse(0.0) == 3.5);
}

TEST_CASE("bounded: invalid parameters give NaN") {
  for (auto [a, c] : {std::pair{3.0, 1.0}, std::pair{1.0, 1.0}, std::pair{std::numeric_limits<double>::quiet_NaN(), 1.0},
                      std::pair{0.0, std::numeric_limits<double>::infinity()}}) {
    const auto b = lb::scalar_bijector::bounded(a, c);
    CHECK(std::isnan(b.forward(0.5)));
    CHECK(std::isnan(b.forward_d1(0.5)));
    CHECK(std::isnan(b.forward_log_jacobian(0.5)));
    CHECK(std::isnan(b.inverse(0.5)));
    CHECK(std::isnan(b.inverse_d3(0.5)));
    CHECK(std::isnan(b.inverse_log_jacobian(0.5)));
    CHECK(std::isnan(b.inverse_distance_upper(0.5)));
    CHECK(std::isnan(b.inverse_distance_lower(0.5)));
    const auto p = b.forward_partials_d2(0.5);
    CHECK(std::isnan(p.c[0]));
  }
}
