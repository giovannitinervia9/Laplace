// exprel(x) and its derivatives phi^(k)(x), k = 1..6 (include/laplace/math/exprel.hpp),
// against the mpmath references (tools/reference/exprel.py), evaluated in a
// Kokkos kernel on the default execution space. The bounds are the errors
// measured on the gcc/OpenMP and CUDA builds on 2026-11-12 (docs/numerics.md) with a
// margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/exprel.hpp>

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

struct Exprel {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (k) {
      case 0: return lm::exprel(x);
      case 1: return lm::exprel_d1(x);
      case 2: return lm::exprel_d2(x);
      case 3: return lm::exprel_d3(x);
      case 4: return lm::exprel_d4(x);
      case 5: return lm::exprel_d5(x);
      default: return lm::exprel_d6(x);
    }
  }
};

}  // namespace

TEST_CASE("exprel and its derivatives against mpmath") {
  // the measured maxima (2026-11-12, gcc, 1721 rows per order) are 1.81, 2.71,
  // 4.11, 5.90, 4.25, 5.85 and 5.92 ulp; the bounds are about 1.5 times each
  const double bound[7] = {2.8, 4.1, 6.2, 8.9, 6.4, 8.8, 8.9};
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "exprel" : "exprel_d" + std::to_string(k);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, Exprel{k});
    const auto a = bench::accuracy(r, y);
    INFO(name, ": max ", a.max_ulp, " ulp at x = ", a.worst[0]);
    if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("%s: max %.3g ulp at x = %.9g\n", name.c_str(), a.max_ulp, a.worst[0]);
    CHECK(a.n_bad == 0);
    CHECK(a.max_ulp <= bound[k]);
  }
}

TEST_CASE("exprel special arguments") {
  CHECK(lm::exprel(0.0) == 1.0);
  CHECK(std::isnan(lm::exprel(std::nan(""))));
  CHECK(lm::exprel(-0.0) == 1.0);
  const double d[7] = {1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 1.0 / 7.0};
  CHECK(lm::exprel_d1(0.0) == d[1]);
  CHECK(lm::exprel_d2(0.0) == d[2]);
  CHECK(lm::exprel_d3(0.0) == d[3]);
  CHECK(lm::exprel_d4(0.0) == d[4]);
  CHECK(lm::exprel_d5(0.0) == d[5]);
  CHECK(lm::exprel_d6(0.0) == d[6]);
  CHECK(std::isnan(lm::exprel_d1(std::nan(""))));
  CHECK(std::isnan(lm::exprel_d6(std::nan(""))));
  // limits at the infinities
  CHECK(lm::exprel(INFINITY) == INFINITY);
  CHECK(lm::exprel(-INFINITY) == 0.0);
  CHECK(lm::exprel_d3(INFINITY) == INFINITY);
  CHECK(lm::exprel_d6(-INFINITY) == 0.0);
  // beyond the overflow, the results are infinite
  CHECK(lm::exprel(800.0) == INFINITY);
  CHECK(lm::exprel_d6(745.0) == INFINITY);
}
