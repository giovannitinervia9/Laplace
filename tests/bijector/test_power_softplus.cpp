// power and softplus (include/laplace/bijector/detail/scalar_power.hpp, scalar_softplus.hpp):
// checked against the mpmath references (shared check in param_common.hpp), then exact
// special values, the invalid parameters, the normalization of the power, the distances,
// inversion, the sign and the static constructors.
#include <doctest/doctest.h>

#include "param_common.hpp"

#include <cmath>
#include <limits>

namespace lb = laplace::bijector;
using bijector_param_test::map_t;

namespace {

const double inf = std::numeric_limits<double>::infinity();
const double nan_value = std::numeric_limits<double>::quiet_NaN();

// got within rel of want (relative), or within abs of zero when want is zero
bool close(double got, double want, double rel) {
  return std::fabs(got - want) <= rel * std::fabs(want) + 1e-300;
}

}  // namespace

TEST_CASE("bijector power: accuracy against mpmath") {
  // bound[which][order]: 1.5 times the larger of the measured maximum ulpC on gcc (where powers
  // go through pow_pos, exp(y log x) on the host) and on CUDA (-1: no rows), measured 2026-11-12;
  // see docs/numerics.md
  static const double bound[6][7] = {
      {2.3, 7.2, 6.3, 5.3, 5.0, 7.1, 6.6},  // forward
      {3.3, 4.2, 7.4, 3.8, 6.3, 6.2, 8.4},  // inverse
      {3.6, 2.4, 3.3, 4.4, 5.2, 8.9, 6.4},  // forward log-Jacobian
      {3.2, 3.0, 3.5, 4.7, 6.0, 8.9, 6.7},  // inverse log-Jacobian
      {-1, -1, -1, -1, -1, -1, -1},         // distance to the upper bound: infinite, no rows
      {-1, -1, -1, -1, -1, -1, -1},         // distance to the lower bound: no rows in the reference
  };
  bijector_param_test::check_map("power", map_t::power, bound);
}

TEST_CASE("bijector power: special values, parameters and distances") {
  const auto p2 = lb::scalar_bijector::power(2.0);  // exponent 2: a genuine power map
  CHECK(p2.map == lb::scalar_bijector_map::power);
  CHECK(p2.forward(3.0) == 9.0);
  CHECK(p2.forward_d1(3.0) == 6.0);
  CHECK(p2.forward_d2(3.0) == 2.0);
  CHECK(p2.forward_d3(3.0) == 0.0);  // the falling factorial 2 * 1 * 0 is exactly 0
  CHECK(p2.forward(0.0) == 0.0);
  CHECK(close(p2.inverse(9.0), 3.0, 1e-15));
  CHECK(close(p2.inverse_d1(9.0), 1.0 / 6.0, 1e-15));  // h(u) = sqrt u, h'(u) = 1/(2 sqrt u)
  CHECK(close(p2.forward_log_jacobian(3.0), std::log(6.0), 1e-15));
  CHECK(close(p2.inverse_log_jacobian(9.0), -std::log(6.0), 1e-15));
  CHECK(p2.inverse_distance_upper(9.0) == inf);
  CHECK(p2.inverse_distance_lower(9.0) == p2.inverse(9.0));
  CHECK(p2.forward_domain_lower() == 0.0);
  CHECK(p2.forward_domain_upper() == inf);
  CHECK(p2.inverse_domain_lower() == 0.0);
  CHECK(p2.inverse_domain_upper() == inf);
  CHECK(!p2.is_decreasing());

  // the named maps: the normalization of decision 9
  CHECK(lb::scalar_bijector::power(1.0).map == lb::scalar_bijector_map::identity);
  CHECK(lb::scalar_bijector::power(0.5).map == lb::scalar_bijector_map::sqrt);
  CHECK(lb::scalar_bijector::power(-1.0).map == lb::scalar_bijector_map::inverse);
  CHECK(lb::scalar_bijector::power(-2.0).map == lb::scalar_bijector_map::inverse_sq);
  CHECK(lb::scalar_bijector::power(0.0).map == lb::scalar_bijector_map::log);
  CHECK(lb::scalar_bijector::power(-0.4).map == lb::scalar_bijector_map::power);
  CHECK(lb::scalar_bijector::power(-0.4).exponent == -0.4);
  CHECK(lb::scalar_bijector::power(-0.4).is_decreasing());
  CHECK(!lb::scalar_bijector::power(0.3).is_decreasing());

  // a decreasing map: g(v) = v^(-1.5) on (0, inf), h(u) = u^(-2/3)
  const auto dec = lb::scalar_bijector::power(-1.5);
  CHECK(dec.is_decreasing());
  CHECK(close(dec.forward(4.0), 0.125, 1e-15));
  CHECK(close(dec.inverse(0.125), 4.0, 1e-14));
  CHECK(close(dec.forward_d1(4.0), -1.5 * std::pow(4.0, -2.5), 1e-14));
  CHECK(close(dec.forward_log_jacobian(4.0), std::log(1.5) - 2.5 * std::log(4.0), 1e-14));

  // inversion swaps the directions
  auto inv = lb::scalar_bijector::power(1.7);
  inv.inverted = true;
  const auto direct = lb::scalar_bijector::power(1.7);
  CHECK(inv.forward(2.0) == direct.inverse(2.0));
  CHECK(inv.inverse(2.0) == direct.forward(2.0));
  CHECK(inv.forward_d2(2.0) == direct.inverse_d2(2.0));
  CHECK(inv.forward_log_jacobian_d1(2.0) == direct.inverse_log_jacobian_d1(2.0));
  CHECK(close(direct.inverse(direct.forward(3.0)), 3.0, 1e-15));

  // invalid exponents: every function returns NaN
  for (double bad : {0.0, nan_value, inf, -inf}) {
    const auto b = bijector_param_test::make(map_t::power, bad, 0.0);
    CHECK(std::isnan(b.forward(2.0)));
    CHECK(std::isnan(b.inverse(2.0)));
    CHECK(std::isnan(b.forward_d1(2.0)));
    CHECK(std::isnan(b.inverse_d3(2.0)));
    CHECK(std::isnan(b.forward_log_jacobian(2.0)));
    CHECK(std::isnan(b.inverse_log_jacobian_d2(2.0)));
    CHECK(std::isnan(b.inverse_distance_lower(2.0)));
  }

  // the partials: component 0 is the pure derivative; there are no derivatives in the exponent
  const auto pp = lb::scalar_bijector::power(1.7);
  const auto c2 = pp.forward_partials_d2(3.0);
  CHECK(c2.c[0] == pp.forward_d2(3.0));
  CHECK(std::isnan(c2.c[1]));
  CHECK(std::isnan(c2.c[2]));
  const auto i3 = pp.inverse_partials_d3(5.0);
  CHECK(i3.c[0] == pp.inverse_d3(5.0));
  CHECK(std::isnan(i3.c[1]));
}

TEST_CASE("bijector power: large exponents without overflow") {
  // g^(6)(v) = F_6 v^(lambda - 6) with lambda = 7: 5040 v at v = 1e300, where v^7 overflows
  const auto b = lb::scalar_bijector::power(7.0);
  CHECK(close(b.forward_d6(1e300), 5040.0 * 1e300, 1e-14));
  CHECK(close(b.forward_d6(3e-300), 5040.0 * 3e-300, 1e-14));
}

TEST_CASE("bijector softplus: accuracy against mpmath") {
  // bound[which][order]: 1.5 times the measured maximum ulpC on gcc and CUDA (-1: no rows), measured 2026-11-12
  // on the gcc and CUDA builds, see docs/numerics.md. The maxima at orders 0..6 are
  // 0.71, 0.77, 1.05, 1.71, 2.61, 3.69, 7.67 (forward); 0.76, 0.74, 1.63, 1.30, 2.54, 3.18, 4.38
  // (inverse); 0.55, 1.19, 1.73, 1.87, 1.76, 2.82, 18.4 (forward log-Jacobian); 0.84, 1.19, 3.47,
  // 1.49, 1.95, 2.60, 3.48 (inverse log-Jacobian)
  static const double bound[6][7] = {
      {1.1, 1.2, 1.6, 2.6, 3.9, 5.6, 11.5},  // forward
      {1.2, 1.1, 2.5, 2.0, 3.8, 4.8, 6.6},   // inverse
      {1.5, 1.8, 2.6, 2.8, 2.7, 4.3, 28.0},  // forward log-Jacobian (order 0: 0.98 ulpC on CUDA)
      {2.3, 1.8, 5.2, 2.3, 2.9, 3.9, 5.2},   // inverse log-Jacobian (order 0: 1.49 ulpC on CUDA)
      {-1, -1, -1, -1, -1, -1, -1},          // distance to the upper bound: infinite, no rows
      {-1, -1, -1, -1, -1, -1, -1},          // distance to the lower bound: no rows in the reference
  };
  bijector_param_test::check_map("softplus", map_t::softplus, bound);
}

TEST_CASE("bijector softplus: special values, parameters and distances") {
  const auto b = lb::scalar_bijector::softplus(1.0);
  CHECK(b.map == lb::scalar_bijector_map::softplus);
  CHECK(b.sharpness == 1.0);
  CHECK(!b.is_decreasing());
  CHECK(close(b.inverse(0.0), std::log(2.0), 1e-15));  // h(0) = log 2
  CHECK(b.inverse(1000.0) == 1000.0);                  // h(u) = u + L/a, L = log(1 + e^-1000) = 0
  CHECK(b.inverse(-1000.0) == 0.0);                    // log(1 + e^-1000) underflows to 0
  CHECK(std::fabs(b.forward(std::log(2.0))) < 1e-15);  // g(log 2) = log(e^(log 2) - 1) = 0
  CHECK(close(b.forward_d1(std::log(2.0)), 2.0, 1e-15));           // g'(v) = 1/(1 - e^-v) = 2
  CHECK(close(b.forward_log_jacobian(std::log(2.0)), std::log(2.0), 1e-15));
  CHECK(close(b.inverse_d1(0.0), 0.5, 1e-15));  // h'(0) = expit(0)
  CHECK(close(b.inverse_d2(0.0), 0.25, 1e-15));  // h''(0) = a sigma'(0) = 1/4 at a = 1
  CHECK(close(b.inverse_log_jacobian(0.0), -std::log(2.0), 1e-15));

  // the distances: infinite above the range (0, inf), h(u) itself below
  CHECK(b.inverse_distance_upper(0.3) == inf);
  CHECK(b.inverse_distance_lower(0.3) == b.inverse(0.3));
  CHECK(b.inverse_distance_lower(-40.0) == b.inverse(-40.0));
  CHECK(b.inverse_distance_lower(-40.0) > 0.0);
  CHECK(b.forward_domain_lower() == 0.0);
  CHECK(b.forward_domain_upper() == inf);
  CHECK(b.inverse_domain_lower() == -inf);
  CHECK(b.inverse_domain_upper() == inf);

  // the mixed derivative in the sharpness, written without cancellation for x = a u > 0 as
  // d_a h = -(x e^-x / (1 + e^-x) + log(1 + e^-x)) / a^2
  const double a = 2.5, u = 0.9, x = a * u;
  const auto s = lb::scalar_bijector::softplus(a);
  const double e = std::exp(-x);
  const double da_h = -(x * e / (1.0 + e) + std::log1p(e)) / (a * a);
  CHECK(close(s.inverse_partials_d1(u).c[1], da_h, 1e-14));
  // for x < 0 the same derivative is (x expit(x) - log(1 + e^x)) / a^2
  const double u2 = -0.7, x2 = a * u2;
  const double da_h2 = (x2 / (1.0 + std::exp(-x2)) - std::log1p(std::exp(x2))) / (a * a);
  CHECK(close(s.inverse_partials_d1(u2).c[1], da_h2, 1e-14));
  // component 0 is the pure derivative, and the components with a parameter 2 are 0
  const auto p2 = s.inverse_partials_d2(u);
  CHECK(p2.c[0] == s.inverse_d2(u));
  CHECK(p2.c[3] == 0.0);  // index(0, 1): softplus has one parameter

  // invalid sharpness: every function returns NaN
  for (double bad : {0.0, -1.0, nan_value, inf}) {
    const auto bb = bijector_param_test::make(map_t::softplus, bad, 0.0);
    CHECK(std::isnan(bb.forward(1.0)));
    CHECK(std::isnan(bb.inverse(1.0)));
    CHECK(std::isnan(bb.forward_d3(1.0)));
    CHECK(std::isnan(bb.inverse_log_jacobian_d1(1.0)));
    CHECK(std::isnan(bb.inverse_distance_lower(1.0)));
  }

  // inversion swaps the directions
  auto inv = lb::scalar_bijector::softplus(0.5);
  inv.inverted = true;
  const auto direct = lb::scalar_bijector::softplus(0.5);
  CHECK(inv.forward(1.2) == direct.inverse(1.2));
  CHECK(inv.inverse(1.2) == direct.forward(1.2));
  CHECK(inv.forward_domain_lower() == -inf);
  CHECK(inv.forward_domain_upper() == inf);

  // the static constructor keeps the sharpness
  CHECK(lb::scalar_bijector::softplus(7.0).sharpness == 7.0);
}
