// affine (include/laplace/bijector/detail/scalar_affine.hpp): g(v) = (v - location)/scale,
// h(u) = location + scale u, checked against the mpmath references (shared check in
// param_common.hpp), then exact special values, the sign of the scale, invalid parameters,
// the partials, the domain and the distances, inversion and the static constructor.
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

TEST_CASE("bijector affine: accuracy against mpmath") {
  // bound[which][order]: 1.5 times the measured maximum ulpC (-1: no rows), see the report
  // measured 2026-11-12 (CPU, OpenMP build): max ulpC 0.33 forward, 0.50 inverse (k = 0) and 0.50
  // inverse k = 1, 0.50 forward log-Jacobian; the inverse has rows only to order 1, the
  // distances none (infinite). Bounds: 1.5 times the maximum, at least 0.5.
  static const double bound[6][7] = {
      {0.5, 0.5, 0.5, 0.55, 0.55, 0.5, 0.55},   // forward
      {0.75, 0.5, -1, -1, -1, -1, -1},          // inverse
      {0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},     // forward log-Jacobian
      {0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},     // inverse log-Jacobian
      {-1, -1, -1, -1, -1, -1, -1},             // distance to upper bound: +inf, no rows
      {-1, -1, -1, -1, -1, -1, -1},             // distance to lower bound: +inf, no rows
  };
  bijector_param_test::check_map("affine", map_t::affine, bound);
}

TEST_CASE("bijector affine: special values, sign of the scale, derivatives") {
  // location 1.5, scale 3: g(4.5) = 1, h(1) = 4.5
  const auto b = lb::scalar_bijector::affine(1.5, 3.0);
  CHECK(b.map == lb::scalar_bijector_map::affine);
  CHECK(b.location == 1.5);
  CHECK(b.scale == 3.0);
  CHECK(b.forward(4.5) == 1.0);
  CHECK(b.forward_d1(4.5) == doctest::Approx(1.0 / 3.0).epsilon(1e-15));
  CHECK(b.forward_d6(4.5) == 0.0);
  CHECK(b.inverse(1.0) == 4.5);
  CHECK(b.inverse_d1(1.0) == 3.0);
  CHECK(b.inverse_d2(1.0) == 0.0);
  CHECK(b.forward_log_jacobian(4.5) == doctest::Approx(-std::log(3.0)).epsilon(1e-15));
  CHECK(b.inverse_log_jacobian(1.0) == doctest::Approx(std::log(3.0)).epsilon(1e-15));
  CHECK(b.forward_log_jacobian_d1(4.5) == 0.0);  // d/dv of a constant
  CHECK(b.inverse_log_jacobian_d6(1.0) == 0.0);
  CHECK_FALSE(b.is_decreasing());
  CHECK(b.forward_domain_lower() == -inf);
  CHECK(b.forward_domain_upper() == inf);
  CHECK(b.inverse_domain_lower() == -inf);
  CHECK(b.inverse_domain_upper() == inf);
}

TEST_CASE("bijector affine: decreasing for a negative scale") {
  const auto b = lb::scalar_bijector::affine(0.0, -2.0);
  CHECK(b.is_decreasing());
  CHECK(b.forward(2.0) == -1.0);
  CHECK(b.forward_d1(2.0) == -0.5);
  CHECK(b.forward_log_jacobian(2.0) == doctest::Approx(-std::log(2.0)).epsilon(1e-15));
  CHECK(b.inverse_log_jacobian(-1.0) == doctest::Approx(std::log(2.0)).epsilon(1e-15));
  CHECK(b.inverse(3.0) == -6.0);
  CHECK(b.inverse_d1(3.0) == -2.0);
}

TEST_CASE("bijector affine: invalid parameters give NaN") {
  for (const auto& b : {lb::scalar_bijector::affine(0.0, 0.0), lb::scalar_bijector::affine(nan_value, 1.0),
                        lb::scalar_bijector::affine(0.0, nan_value), lb::scalar_bijector::affine(inf, 1.0)}) {
    CHECK(std::isnan(b.forward(1.0)));
    CHECK(std::isnan(b.inverse(1.0)));
    CHECK(std::isnan(b.forward_d1(1.0)));
    CHECK(std::isnan(b.inverse_d2(1.0)));
    CHECK(std::isnan(b.forward_log_jacobian(1.0)));
    CHECK(std::isnan(b.inverse_log_jacobian(1.0)));
    CHECK(std::isnan(b.inverse_distance_upper(1.0)));
    CHECK(std::isnan(b.inverse_distance_lower(1.0)));
    CHECK(std::isnan(b.forward_domain_lower()));
  }
}

TEST_CASE("bijector affine: partial derivatives in (location, scale)") {
  // g = (v - p1)/p2 at v = 2, p1 = 1, p2 = 0.5: d/dv = 2, d2/(dv dp2) = -4, d2/(dp1 dp2) = 4,
  // d2/dp2^2 = 16. Components of bijector_partials<2>: index(n1, n2) = 3 n2 - n2(n2-1)/2 + n1.
  const auto b = lb::scalar_bijector::affine(1.0, 0.5);
  const auto f2 = b.forward_partials_d2(2.0);
  CHECK(f2.c[0] == 0.0);  // d2/dv2
  CHECK(f2.c[1] == 0.0);  // d2/(dv dp1)
  CHECK(f2.c[2] == 0.0);  // d2/dp1^2
  CHECK(f2.c[3] == doctest::Approx(-4.0).epsilon(1e-15));  // (n1, n2) = (0, 1)
  CHECK(f2.c[4] == doctest::Approx(4.0).epsilon(1e-15));   // (1, 1)
  CHECK(f2.c[5] == doctest::Approx(16.0).epsilon(1e-15));  // (0, 2)
  // first order: d/dv g = 1/p2 = 2, d/dp1 g = -1/p2 = -2, d/dp2 g = -(v - p1)/p2^2 = -4
  const auto f1 = b.forward_partials_d1(2.0);
  CHECK(f1.c[0] == doctest::Approx(2.0).epsilon(1e-15));
  CHECK(f1.c[1] == doctest::Approx(-2.0).epsilon(1e-15));  // (n1, n2) = (1, 0)
  CHECK(f1.c[2] == doctest::Approx(-4.0).epsilon(1e-15));  // (0, 1)
  // h = p1 + p2 u at u = 2: d/du = p2 = 0.5, d/dp2 = u = 2, d2/(du dp2) = 1
  const auto i1 = b.inverse_partials_d1(2.0);
  CHECK(i1.c[0] == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(i1.c[1] == doctest::Approx(1.0).epsilon(1e-15));  // d/dp1 = 1
  CHECK(i1.c[2] == doctest::Approx(2.0).epsilon(1e-15));  // d/dp2 = u
  const auto i2 = b.inverse_partials_d2(2.0);
  CHECK(i2.c[0] == 0.0);
  CHECK(i2.c[1] == 0.0);
  CHECK(i2.c[2] == 0.0);
  CHECK(i2.c[3] == doctest::Approx(1.0).epsilon(1e-15));  // d2/(du dp2)
  CHECK(i2.c[4] == 0.0);
  CHECK(i2.c[5] == 0.0);
  // log|p2|: d/dp2 = 1/p2 = 2 for the inverse, -1/p2 = -2 for the forward
  const auto il = b.inverse_log_jacobian_partials_d1(2.0);
  CHECK(il.c[2] == doctest::Approx(2.0).epsilon(1e-15));
  CHECK(il.c[0] == 0.0);
  CHECK(il.c[1] == 0.0);
  const auto fl = b.forward_log_jacobian_partials_d1(2.0);
  CHECK(fl.c[2] == doctest::Approx(-2.0).epsilon(1e-15));
  CHECK(fl.c[0] == 0.0);
  CHECK(fl.c[1] == 0.0);
}

TEST_CASE("bijector affine: distances to the bounds are infinite") {
  const auto b = lb::scalar_bijector::affine(1.0, -4.0);
  for (double u : {-1e10, -1.0, 0.0, 0.5, 1e10}) {
    CHECK(b.inverse_distance_upper(u) == inf);
    CHECK(b.inverse_distance_lower(u) == inf);
  }
}

TEST_CASE("bijector affine: inversion and the static constructor") {
  const auto b = lb::scalar_bijector::affine(-2.0, 0.25);
  auto inv = b;
  inv.inverted = true;
  for (double x : {-3.0, 0.0, 7.5}) {
    CHECK(inv.forward(x) == b.inverse(x));
    CHECK(inv.inverse(x) == b.forward(x));
    CHECK(inv.forward_d1(x) == b.inverse_d1(x));
    CHECK(inv.forward_log_jacobian(x) == b.inverse_log_jacobian(x));
    CHECK(inv.inverse_log_jacobian(x) == b.forward_log_jacobian(x));
  }
  // the field-by-field object is the same as the constructor's
  lb::scalar_bijector c{lb::scalar_bijector_map::affine};
  c.location = -2.0;
  c.scale = 0.25;
  CHECK(c.forward(1.0) == b.forward(1.0));
  CHECK(c.inverse(1.0) == b.inverse(1.0));
}
