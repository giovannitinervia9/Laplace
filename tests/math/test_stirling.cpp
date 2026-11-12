// Stirling's error delta(a) and its derivatives delta^(k)(a), k = 1..6
// (include/laplace/math/stirling.hpp), against the mpmath references
// (tools/reference/stirling.py), evaluated in a Kokkos kernel on the default
// execution space. The bounds are the errors measured on the gcc/OpenMP build on
// 2026-11-12 (docs/numerics.md) with a margin of about half again.
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
