// log1prel(x) = log(1 + x)/x and its derivatives to order 6 (include/laplace/math/log1prel.hpp)
// against the mpmath references in tests/data/reference/log1prel*.csv, evaluated in a Kokkos
// kernel on the default execution space; plus the exact values at 0, the limits at -1 and +inf,
// and the NaN results outside the domain. Bounds: the errors measured on 2026-11-12 on the gcc build
// and CUDA builds with a margin of about half again (docs/numerics.md).
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/log1prel.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;

// order k of log1prel, k = 0 for the value
struct Psi {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (k) {
      case 0: return lm::log1prel(x);
      case 1: return lm::log1prel_d1(x);
      case 2: return lm::log1prel_d2(x);
      case 3: return lm::log1prel_d3(x);
      case 4: return lm::log1prel_d4(x);
      case 5: return lm::log1prel_d5(x);
      default: return lm::log1prel_d6(x);
    }
  }
};

// the bound on the maximum error in ulp for the value and for each order 1..6
const double bound[7] = {2.0, 6.0, 7.6, 9.6, 10.6, 12.4, 14.4};

}  // namespace

TEST_CASE("log1prel and its derivatives to order 6 against mpmath") {
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "log1prel" : "log1prel_d" + std::to_string(k);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, Psi{k});
    const auto a = bench::accuracy(r, y);
    INFO(name, ": max ", a.max_ulp, " ulp at x = ", a.worst[0], ", ", a.n_bad, " non-finite results");
    if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("%s: max %.3g ulp at x = %.9g\n", name.c_str(), a.max_ulp, a.worst[0]);
    CHECK(a.n_bad == 0);
    CHECK(a.max_ulp <= bound[k]);
  }
}

TEST_CASE("exact values at zero") {
  CHECK(lm::log1prel(0.0) == 1.0);
  double fact = 1.0;  // k!
  for (int k = 1; k <= 6; ++k) {
    fact *= k;
    const double expected = (k % 2 ? -1.0 : 1.0) * fact / (k + 1.0);
    INFO("order ", k);
    CHECK(Psi{k}(0.0) == expected);
  }
}

TEST_CASE("limits at -1 and at +inf, NaN outside the domain") {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int k = 0; k <= 6; ++k) {
    INFO("order ", k);
    CHECK(std::isnan(Psi{k}(-1.5)));
    CHECK(std::isnan(Psi{k}(-inf)));
    CHECK(std::isnan(Psi{k}(nan)));
    // the limit at -1: psi goes to +inf, psi^(k) to (-1)^k inf
    CHECK(Psi{k}(-1.0) == (k % 2 ? -inf : inf));
    CHECK(Psi{k}(inf) == 0.0);
  }
}
