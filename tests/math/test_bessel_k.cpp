// log K_nu(x) and its derivatives in x to order 7
// (include/laplace/math/bessel_k.hpp) against the mpmath references,
// evaluated in a Kokkos kernel on the default execution space. Bounds: the
// errors measured on 2026-11-11 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/bessel_k.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;

struct LogK {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    switch (n) {
      case 0: return lm::log_bessel_k(nu, x);
      case 1: return lm::log_bessel_k_dx1(nu, x);
      case 2: return lm::log_bessel_k_dx2(nu, x);
      case 3: return lm::log_bessel_k_dx3(nu, x);
      case 4: return lm::log_bessel_k_dx4(nu, x);
      case 5: return lm::log_bessel_k_dx5(nu, x);
      case 6: return lm::log_bessel_k_dx6(nu, x);
      default: return lm::log_bessel_k_dx7(nu, x);
    }
  }
};

}  // namespace

// The value in units of the last place of max(|log K|, |x g|, 1), g the first
// derivative: one ulp of x moves log K by |x g| ulps of 1, and near the
// turning point x ~ 0.66 nu, where log K crosses zero, |x g| ~ 1.2 nu.
TEST_CASE("log K_nu(x) against mpmath, relative to its conditioning") {
  const auto r = bench::read_ref(refdir + "/log_bessel_k.csv");
  const auto g = bench::read_ref(refdir + "/log_bessel_k_dx1.csv");
  const auto y = bench::eval_kernel(r.x, LogK{0});
  double worst = 0, wn = 0, wx = 0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) {
      ++bad;
      continue;
    }
    const double scale = std::max({std::fabs(r.hi[i]), std::fabs(r.x[i][1] * g.hi[i]), 1.0});
    const double e = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / bench::ulp_of(scale);
    if (e > worst) worst = e, wn = r.x[i][0], wx = r.x[i][1];
  }
  INFO("log_bessel_k: max ", worst, " ulpC at nu = ", wn, ", x = ", wx);
  CHECK(bad == 0);
  CHECK(worst <= 7.5);
}

TEST_CASE("derivatives of log K_nu(x) in x against mpmath") {
  // orders 2-4: 44.4, 44.0, 24.1 ulp measured on 2026-11-11 at nu = 1/4, x near 1,
  // where Temme's first coefficient cancels (note/numerics.md); the same with the
  // platform's exp and log
  const double bound[8] = {0, 9.0, 66.0, 66.0, 36.0, 17.0, 33.0, 48.0};
  for (int n = 1; n <= 7; ++n) {
    const std::string name = "log_bessel_k_dx" + std::to_string(n);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, LogK{n});
    const auto a = bench::accuracy(r, y);
    INFO(name, ": max ", a.max_ulp, " ulp at nu = ", a.worst[0], ", x = ", a.worst[1]);
    CHECK(a.n_bad == 0);
    CHECK(a.max_ulp <= bound[n]);
  }
}

TEST_CASE("special arguments of log K") {
  CHECK(std::isnan(lm::log_bessel_k(1.0, -1.0)));
  CHECK(std::isnan(lm::log_bessel_k_dx3(1.0, -1.0)));
  CHECK(std::isinf(lm::log_bessel_k(1.0, 0.0)));
  CHECK(lm::log_bessel_k(2.0, INFINITY) == -INFINITY);
  CHECK(lm::log_bessel_k_dx1(2.0, INFINITY) == -1.0);
  CHECK(lm::log_bessel_k_dx2(2.0, INFINITY) == 0.0);
  // K_-nu = K_nu
  CHECK(lm::log_bessel_k(-2.5, 3.0) == lm::log_bessel_k(2.5, 3.0));
  // K_(1/2)(x) = sqrt(pi / (2x)) e^-x: log K = log(pi/2)/2 - log(x)/2 - x, d1 = -1/(2x) - 1
  for (double x : {0.01, 0.7, 3.0, 50.0, 1e4}) {
    const double v = 0.5 * std::log(1.5707963267948966192) - 0.5 * std::log(x) - x;
    INFO("x = ", x);
    CHECK(std::fabs(lm::log_bessel_k(0.5, x) - v) <= 4e-16 * std::max(std::fabs(v), 1.0));
    CHECK(std::fabs(lm::log_bessel_k_dx1(0.5, x) + 0.5 / x + 1.0) <= 4e-16 * (0.5 / x + 1.0));
  }
}
