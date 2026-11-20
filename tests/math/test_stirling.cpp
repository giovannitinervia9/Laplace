// Stirling's error delta(a) and its derivatives delta^(k)(a), k = 1..6, and the half-step
// log-gamma ratio R(a) = log Gamma(a + 1/2) - log Gamma(a) with its own derivatives
// (include/laplace/math/stirling.hpp), against the mpmath references
// (tools/reference/stirling.py, tools/reference/lgamma_half_ratio.py), evaluated in a
// Kokkos kernel on the default execution space. The bounds are the errors measured on the
// gcc/OpenMP build on 2026-11-12 and 2026-11-20 (docs/numerics.md) with a margin of about
// half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/stirling.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;

struct HalfRatio {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double a) const {
    switch (k) {
      case 0: return lm::lgamma_half_ratio(a);
      case 1: return lm::lgamma_half_ratio_d1(a);
      case 2: return lm::lgamma_half_ratio_d2(a);
      case 3: return lm::lgamma_half_ratio_d3(a);
      case 4: return lm::lgamma_half_ratio_d4(a);
      case 5: return lm::lgamma_half_ratio_d5(a);
      default: return lm::lgamma_half_ratio_d6(a);
    }
  }
};

struct Stirling {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double a) const {
    switch (k) {
      case 0: return lm::stirling_error(a);
      case 1: return lm::stirling_error_d1(a);
      case 2: return lm::stirling_error_d2(a);
      case 3: return lm::stirling_error_d3(a);
      case 4: return lm::stirling_error_d4(a);
      case 5: return lm::stirling_error_d5(a);
      default: return lm::stirling_error_d6(a);
    }
  }
};

}  // namespace

TEST_CASE("Stirling's error and its derivatives against mpmath") {
  // measured maxima (2026-11-12, gcc/OpenMP, 1384 rows per order), see docs/numerics.md
  // 13.2, 12.7, 10.8, 13.4, 11.7, 14.8, 14.2 ulp, all on the direct route (a < 1); the bounds
  // are about 1.5 times each
  const double bound[7] = {20, 19, 16.5, 20, 17.5, 22, 21.5};
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "stirling_error" : "stirling_error_d" + std::to_string(k);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, Stirling{k});
    const auto a = bench::accuracy(r, y);
    INFO(name, ": max ", a.max_ulp, " ulp at a = ", a.worst[0]);
    if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("%s: max %.3g ulp at a = %.9g\n", name.c_str(), a.max_ulp, a.worst[0]);
    CHECK(a.n_bad == 0);
    CHECK(a.max_ulp <= bound[k]);
  }
}

TEST_CASE("the half-step log-gamma ratio and its derivatives against mpmath") {
  // R vanishes at a = 1.21163..., where Gamma(a + 1/2) = Gamma(a) astride the minimum of
  // Gamma, so the value is bounded in ulp1, the absolute error in units of the ulp of
  // max(|R|, 1) (\ref accuracy); its derivatives have no zero and are bounded in ulp.
  // Measured 2026-11-20: 1.42 ulp1 for the value on gcc and 1.97 on CUDA (198 ulp of R
  // itself on the row next to its zero, on either build) and 2.16, 2.75, 3.25, 3.65,
  // 3.96, 3.41 ulp for the derivatives, the same on the two builds; the bounds are those
  // with a margin of about half again.
  const double bound[7] = {3, 4, 4.5, 5, 5.5, 6, 5.5};
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "lgamma_half_ratio" : "lgamma_half_ratio_d" + std::to_string(k);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, HalfRatio{k});
    const auto a = bench::accuracy(r, y);
    const double e = k == 0 ? a.max_ulp1 : a.max_ulp;
    INFO(name, ": max ", e, (k == 0 ? " ulp1 at a = " : " ulp at a = "), a.worst[0]);
    std::printf("%s: max %.3g ulp (%.3g ulp1) at a = %.9g\n", name.c_str(), a.max_ulp, a.max_ulp1, a.worst[0]);
    CHECK(a.n_bad == 0);
    CHECK(e <= bound[k]);
  }
}

TEST_CASE("the half-step log-gamma ratio: special arguments and the recurrence step") {
  CHECK(std::isnan(lm::lgamma_half_ratio(0.0)));
  CHECK(std::isnan(lm::lgamma_half_ratio(-1.0)));
  CHECK(std::isnan(lm::lgamma_half_ratio_d3(std::nan(""))));
  // R(1/2) = log Gamma(1) - log Gamma(1/2) = -log sqrt(pi)
  CHECK(lm::lgamma_half_ratio(0.5) == doctest::Approx(-0.5 * std::log(3.14159265358979323846)).epsilon(1e-15));
  // R(1) = log Gamma(3/2) - log Gamma(1) = log(sqrt(pi)/2)
  CHECK(lm::lgamma_half_ratio(1.0) ==
        doctest::Approx(std::log(0.5 * std::sqrt(3.14159265358979323846))).epsilon(1e-15));
  // the series and the recurrence meet at a = 2
  for (int k = 0; k <= 6; ++k) {
    const HalfRatio f{k};
    const double lo = f(std::nextafter(2.0, 0.0)), hi = f(2.0);
    INFO("order ", k, ": below ", lo, ", at ", hi);
    CHECK(std::fabs(lo - hi) <= 1e-13 * std::fmax(std::fabs(hi), 1.0));
  }
}

TEST_CASE("Stirling's error: special arguments") {
  CHECK(std::isnan(lm::stirling_error(0.0)));
  CHECK(std::isnan(lm::stirling_error(-1.0)));
  CHECK(std::isnan(lm::stirling_error_d3(std::nan(""))));
  CHECK(lm::stirling_error(INFINITY) == 0.0);
  CHECK(lm::stirling_error_d6(INFINITY) == 0.0);
  // delta(1) = 1 - log sqrt(2 pi); delta'(1) = -gamma_E + 1/2
  CHECK(lm::stirling_error(1.0) == doctest::Approx(1.0 - 0.91893853320467274178).epsilon(1e-15));
  CHECK(lm::stirling_error_d1(1.0) == doctest::Approx(0.5 - 0.57721566490153286061).epsilon(1e-15));
  // the routes meet at a = 1 and a = 10
  for (double a : {1.0, 10.0})
    CHECK(lm::stirling_error_d4(std::nextafter(a, 0.0)) == doctest::Approx(lm::stirling_error_d4(a)).epsilon(1e-13));
}
