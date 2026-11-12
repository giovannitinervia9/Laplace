// Scalar bijectors without parameters (include/laplace/bijector/scalar.hpp)
// against the mpmath references (tools/reference/bijector_scalar.py),
// evaluated in a Kokkos kernel on the default execution space. For each map
// and each of the four functions (forward, inverse, forward and inverse
// log-Jacobians) at each order k = 0..6, two errors:
//   ulp   relative, in ulps of max(|f_k|, 2^-1022);
//   ulpC  in ulps of |f_k(x)| + |x f_(k+1)(x)|, the error that a relative
//         perturbation of one ulp in x would cause: the relative error
//         divided by 1 + the condition number. Near a zero of f_k (the
//         derivatives of high order cross zero) ulp grows without bound while
//         ulpC stays of the order of the rounding of the intermediate
//         quantities; the reference files carry order 7 for the scale of
//         order 6.
// The bounds are on ulpC. Set LAPLACE_PRINT_ERRORS=1 to print the table.
// Bounds: the errors measured on 2026-11-12 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/bijector/scalar.hpp>

#include <cstdlib>
#include <map>
#include <tuple>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lb = laplace::bijector;
using bmap = lb::scalar_bijector_map;

KOKKOS_INLINE_FUNCTION double call(lb::scalar_bijector b, int which, int k, double x) {
  switch (which * 8 + k) {
    case 0: return b.forward(x);
    case 1: return b.forward_d1(x);
    case 2: return b.forward_d2(x);
    case 3: return b.forward_d3(x);
    case 4: return b.forward_d4(x);
    case 5: return b.forward_d5(x);
    case 6: return b.forward_d6(x);
    case 8: return b.inverse(x);
    case 9: return b.inverse_d1(x);
    case 10: return b.inverse_d2(x);
    case 11: return b.inverse_d3(x);
    case 12: return b.inverse_d4(x);
    case 13: return b.inverse_d5(x);
    case 14: return b.inverse_d6(x);
    case 16: return b.forward_log_jacobian(x);
    case 17: return b.forward_log_jacobian_d1(x);
    case 18: return b.forward_log_jacobian_d2(x);
    case 19: return b.forward_log_jacobian_d3(x);
    case 20: return b.forward_log_jacobian_d4(x);
    case 21: return b.forward_log_jacobian_d5(x);
    case 22: return b.forward_log_jacobian_d6(x);
    case 24: return b.inverse_log_jacobian(x);
    case 25: return b.inverse_log_jacobian_d1(x);
    case 26: return b.inverse_log_jacobian_d2(x);
    case 27: return b.inverse_log_jacobian_d3(x);
    case 28: return b.inverse_log_jacobian_d4(x);
    case 29: return b.inverse_log_jacobian_d5(x);
    case 30: return b.inverse_log_jacobian_d6(x);
    case 32: return b.inverse_distance_upper(x);
    case 40: return b.inverse_distance_lower(x);
  }
  return laplace::math::detail::quiet_nan();
}

struct Eval {
  lb::scalar_bijector b;
  KOKKOS_INLINE_FUNCTION double operator()(double x, double which, double k) const {
    return call(b, static_cast<int>(which), static_cast<int>(k), x);
  }
};

const char* which_name[] = {"forward", "inverse", "fwd log-jac", "inv log-jac", "to upper", "to lower"};

// max ulp and max ulpC per (which, k), and where
struct Err {
  double ulp = 0, ulpc = 0, x = 0;
  std::size_t n = 0, bad = 0;
};

std::map<int, Err> measure(const char* name, bmap m) {
  const auto r = bench::read_ref(refdir + "/bijector_" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, Eval{lb::scalar_bijector{m, false}});
  std::map<std::tuple<int, int, double>, double> ref;  // (which, k, x) -> f
  for (std::size_t i = 0; i < y.size(); ++i)
    ref[{static_cast<int>(r.x[i][1]), static_cast<int>(r.x[i][2]), r.x[i][0]}] = r.hi[i];
  std::map<int, Err> out;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const int w = static_cast<int>(r.x[i][1]), k = static_cast<int>(r.x[i][2]);
    if (k > 6) continue;
    const double x = r.x[i][0];
    auto& e = out[w * 8 + k];
    ++e.n;
    if (!std::isfinite(y[i])) {
      ++e.bad;
      e.x = x;
      continue;
    }
    // the distances (which 4, 5) have derivative -+h'(x), the row (1, 1, x)
    const auto next = w >= 4 ? ref.find({1, 1, x}) : ref.find({w, k + 1, x});
    const double fn = next == ref.end() ? 0.0 : next->second;  // missing: constant or under/overflowed
    const double d = std::fabs((y[i] - r.hi[i]) - r.lo[i]);
    const double u = d / bench::ulp_of(std::max(std::fabs(r.hi[i]), 2.2250738585072014e-308));
    const double uc = d / bench::ulp_of(std::max(std::fabs(r.hi[i]) + std::fabs(x * fn), 2.2250738585072014e-308));
    e.ulp = std::max(e.ulp, u);
    if (uc > e.ulpc) e.ulpc = uc, e.x = x;
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (const auto& [key, e] : out)
      std::printf("%-10s %-11s k=%d  n=%4zu  max %9.3g ulp  %8.3g ulpC  at x=%.6g%s\n", name, which_name[key / 8],
                  key % 8, e.n, e.ulp, e.ulpc, e.x, e.bad ? "  NOT FINITE" : "");
  }
  return out;
}

void check(const char* name, bmap m, double bound) {
  const auto errs = measure(name, m);
  for (const auto& [key, e] : errs) {
    INFO(name, " ", which_name[key / 8], " order ", key % 8, ": max ", e.ulpc, " ulpC at x = ", e.x, ", ", e.bad,
         " not finite");
    CHECK(e.bad == 0);
    CHECK(e.ulpc <= bound);
  }
}

}  // namespace

TEST_CASE("bijector scalar: accuracy against mpmath") {
  check("log", bmap::log, 6);
  check("logit", bmap::logit, 16);
  check("probit", bmap::probit, 10);
  check("cauchit", bmap::cauchit, 13);
  check("cloglog", bmap::cloglog, 15);
  check("loglog", bmap::loglog, 17);
  check("inverse", bmap::inverse, 6);
  check("inverse_sq", bmap::inverse_sq, 11);
  check("sqrt", bmap::sqrt, 17);
  check("atanh", bmap::atanh, 8);
  check("asinh", bmap::asinh, 15);
}

TEST_CASE("bijector scalar: constant derivatives, identity, inversion, domain") {
  const lb::scalar_bijector id{bmap::identity}, lg{bmap::log}, ex{bmap::log, true};
  CHECK(id.forward(2.5) == 2.5);
  CHECK(id.forward_d1(2.5) == 1.0);
  CHECK(id.forward_d2(2.5) == 0.0);
  CHECK(id.forward_log_jacobian(2.5) == 0.0);
  CHECK(id.inverse_log_jacobian_d3(2.5) == 0.0);
  // the derivatives that are constant (no reference rows)
  CHECK(lg.inverse_log_jacobian_d1(3.0) == 1.0);
  CHECK(lg.inverse_log_jacobian_d2(3.0) == 0.0);
  CHECK((lb::scalar_bijector{bmap::probit}).inverse_log_jacobian_d2(0.3) == -1.0);
  CHECK((lb::scalar_bijector{bmap::probit}).inverse_log_jacobian_d4(0.3) == 0.0);
  CHECK((lb::scalar_bijector{bmap::sqrt}).inverse_d2(0.3) == 2.0);
  CHECK((lb::scalar_bijector{bmap::sqrt}).inverse_d5(0.3) == 0.0);
  // inversion swaps the two directions: exp is log inverted
  CHECK(ex.forward(1.0) == lg.inverse(1.0));
  CHECK(ex.forward_d3(0.5) == lg.inverse_d3(0.5));
  CHECK(ex.inverse(2.0) == lg.forward(2.0));
  CHECK(ex.forward_log_jacobian(0.5) == lg.inverse_log_jacobian(0.5));
  CHECK(ex.inverse_log_jacobian_d2(2.0) == lg.forward_log_jacobian_d2(2.0));
  CHECK(ex.forward_domain_lower() == -ex.forward_domain_upper());
  CHECK(ex.inverse_domain_lower() == 0.0);
  // outside the domain: NaN; on a bound: the limit
  const lb::scalar_bijector lt{bmap::logit};
  CHECK(std::isnan(lt.forward(1.5)));
  CHECK(std::isnan(lt.forward_d2(-0.1)));
  CHECK(std::isnan(lg.forward(-1.0)));
  CHECK(std::isnan((lb::scalar_bijector{bmap::sqrt}).inverse(-1.0)));
  CHECK(lt.forward(0.0) == -std::numeric_limits<double>::infinity());
  CHECK(lt.forward(1.0) == std::numeric_limits<double>::infinity());
  CHECK((lb::scalar_bijector{bmap::inverse}).is_decreasing());
  CHECK((lb::scalar_bijector{bmap::inverse_sq, true}).is_decreasing());
  CHECK(!lt.is_decreasing());
  // distances to the bounds: exact past the saturation of h, generic elsewhere
  CHECK(lt.inverse(40.0) == 1.0);
  CHECK(lt.inverse_distance_upper(40.0) > 0.0);
  CHECK(lg.inverse_distance_lower(2.0) == lg.inverse(2.0));
  CHECK(lg.inverse_distance_upper(2.0) == std::numeric_limits<double>::infinity());
  CHECK(ex.inverse_distance_upper(2.0) == std::numeric_limits<double>::infinity());
  CHECK(ex.inverse_distance_lower(2.0) == lg.forward(2.0) + std::numeric_limits<double>::infinity());
  CHECK(std::isnan((lb::scalar_bijector{bmap::sqrt}).inverse_distance_upper(-1.0)));
}
