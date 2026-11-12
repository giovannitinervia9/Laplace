// Mixed partial derivatives of log P(a, x) and log Q(a, x) in a and x, total
// orders 1 to 6 (include/laplace/math/incomplete_gamma_mixed.hpp), against
// the mpmath references (tools/reference/incomplete_mixed.py), evaluated in a
// Kokkos kernel on the default execution space.
//
// The figure per point is the smaller of ulpL = ulp / max(1, |log|f||) and
// ulp1 = the error in ulps of max(|f|, 1), as in test_gamma_deriv.cpp; the
// bound is on its maximum over all components j = 0..K of one order.
// Bounds: the maxima measured on 2026-11-12 on the gcc/OpenMP build (and on
// the host with the series limits of the device, the same within 10 per
// cent), with a margin of about half again: P orders 1-6: 5.37, 8.03, 12.9,
// 33.4, 176, 651; Q orders 1-6: 31, 40.4, 113, 8150, 2.6e4, 1.05e5. The large
// figures of Q at orders 4-6 are the components with one or two derivatives
// in x at a < 1 (docs/numerics.md, "Mixed partial derivatives of log P and
// log Q"); for a >= 1 the maxima of Q are 31.2, 63.4 and 235 at orders 4-6.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_gamma_mixed.hpp>

#include <limits>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

struct Mixed {
  int k;
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double jd) const {
    namespace lm = laplace::math;  // the wrappers
    const int j = static_cast<int>(jd);
    if (p) {
      switch (k) {
        case 1: return lm::log_gamma_p_mixed_d1(a, x).v[j];
        case 2: return lm::log_gamma_p_mixed_d2(a, x).v[j];
        case 3: return lm::log_gamma_p_mixed_d3(a, x).v[j];
        case 4: return lm::log_gamma_p_mixed_d4(a, x).v[j];
        case 5: return lm::log_gamma_p_mixed_d5(a, x).v[j];
        default: return lm::log_gamma_p_mixed_d6(a, x).v[j];
      }
    }
    switch (k) {
      case 1: return lm::log_gamma_q_mixed_d1(a, x).v[j];
      case 2: return lm::log_gamma_q_mixed_d2(a, x).v[j];
      case 3: return lm::log_gamma_q_mixed_d3(a, x).v[j];
      case 4: return lm::log_gamma_q_mixed_d4(a, x).v[j];
      case 5: return lm::log_gamma_q_mixed_d5(a, x).v[j];
      default: return lm::log_gamma_q_mixed_d6(a, x).v[j];
    }
  }
};

void check(int k, bool p, double bound) {
  const std::string name = std::string(p ? "log_gamma_p_mixed_d" : "log_gamma_q_mixed_d") + std::to_string(k);
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, Mixed{k, p});
  double worst = 0;
  bench::Args at{};
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double hi = r.hi[i];
    const double e = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 2.2250738585072014e-308)));
    if (!std::isfinite(e)) {
      ++bad;
      continue;
    }
    const double eL = e / std::max(1.0, std::fabs(std::log(std::fabs(hi))));
    const double e1 = std::fabs(bench::error_in(y[i], hi, r.lo[i], std::max(std::fabs(hi), 1.0)));
    const double u = std::min(eL, e1);
    if (u > worst) worst = u, at = r.x[i];
  }
  INFO(name, ": max min(ulpL, ulp1) ", worst, " at (", at[0], ", ", at[1], ", j = ", at[2], ")");
  CHECK(bad == 0);
  CHECK(worst <= bound);
}

// the limits at x = 0 and x = infinity, evaluated in a kernel
struct Limit {
  bool p;
  KOKKOS_INLINE_FUNCTION double operator()(double a, double x, double jd) const {
    return p ? laplace::math::log_gamma_p_mixed_d4(a, x).v[static_cast<int>(jd)]
             : laplace::math::log_gamma_q_mixed_d4(a, x).v[static_cast<int>(jd)];
  }
};

}  // namespace

TEST_CASE("mixed partials of log P(a, x) in a and x against mpmath") {
  check(1, true, 8.0);
  check(2, true, 12.0);
  check(3, true, 20.0);
  check(4, true, 50.0);
  check(5, true, 265.0);
  check(6, true, 980.0);
}

TEST_CASE("mixed partials of log Q(a, x) in a and x against mpmath") {
  check(1, false, 47.0);
  check(2, false, 61.0);
  check(3, false, 170.0);
  check(4, false, 12300.0);
  check(5, false, 39000.0);
  check(6, false, 158000.0);
}

TEST_CASE("mixed partials of log P and log Q at x = 0 and x = infinity") {
  const double inf = std::numeric_limits<double>::infinity();
  // order 4 at a = 2.5: P at x = 0: v[1], v[2] = d^4/da^3 dx, d^4/da^2 dx^2 are
  // finite (the cumulants of I_L at x = 0), v[3], v[4] infinite with the sign
  // of a (-1)^(j-1) (j-1)!/x^j; Q at x = 0: 0 for j < a, infinite beyond
  std::vector<bench::Args> args;
  for (int j = 0; j <= 4; ++j) args.push_back({2.5, 0.0, double(j)});
  for (int j = 0; j <= 4; ++j) args.push_back({2.5, inf, double(j)});
  const auto yp = bench::eval_kernel(args, Limit{true});
  const auto yq = bench::eval_kernel(args, Limit{false});
  // d/dx log I_L at x = 0 is 1/(a + 1): d^3/da^3 of it is -6/(a + 1)^4
  CHECK(yp[1] == doctest::Approx(-6.0 / std::pow(3.5, 4)).epsilon(1e-14));
  CHECK(std::isfinite(yp[2]));
  CHECK(yp[3] == inf);
  CHECK(yp[4] == -inf);
  CHECK(yq[0] == 0.0);
  CHECK(yq[1] == 0.0);
  CHECK(yq[2] == 0.0);
  // -(a)_j (log x)^i x^(a-j) / Gamma(a + 1): (a)_3 = 2.5 1.5 0.5 > 0 with
  // log x < 0 (i = 1); (a)_4 = (a)_3 (-0.5) < 0 (i = 0)
  CHECK(yq[3] == inf);
  CHECK(yq[4] == inf);
  for (int j = 0; j <= 4; ++j) CHECK(yp[5 + j] == 0.0);
  for (int j = 1; j <= 4; ++j) CHECK(yq[5 + j] == 0.0);
}
