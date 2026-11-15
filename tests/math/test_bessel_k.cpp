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

namespace {

// the joint routine: out[0] = log K, out[n] = the derivative of order n, from one evaluation of the shared work
struct LogKJoint {
  int n;
  int top;  // highest order requested of the joint routine (3 or 7)
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    double out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (top == 3) lm::log_bessel_k_dx_upto<3>(nu, x, out);
    else lm::log_bessel_k_dx_upto<7>(nu, x, out);
    return out[n];
  }
};

}  // namespace

// log_bessel_k_dx_upto<N> against the single-order functions: the same double on the CPU builds, where the two
// repeat the same arithmetic in every regime (checked bit for bit: 0 of 146792 entries differ on gcc). On the
// CUDA build nvcc contracts and schedules the two functions differently and 20744 of the 146792 entries
// differ in the last bits, by at most 1.2e-15 relative (bound 1.8e-15). The points are the
// reference grids of the seven orders and the boundaries between the three regimes (nu = 20 of the Debye
// expansion, x = 200 of the asymptotic one, x = 1 of Steed's fraction in double-double).
TEST_CASE("log_bessel_k_dx_upto against the single-order functions, every regime and boundary") {
  std::vector<bench::Args> pts;
  for (int n = 1; n <= 7; ++n) {
    const auto r = bench::read_ref(refdir + "/log_bessel_k_dx" + std::to_string(n) + ".csv");
    pts.insert(pts.end(), r.x.begin(), r.x.end());
  }
  for (double nu : {0.0, 0.1, 0.5, 1.0, 2.0, 19.99, 20.0, 20.01, 50.0, 300.0})
    for (double x : {0.5, 0.99, 1.0, 1.01, 2.0, 5.0, 199.9, 200.0, 200.1, 1000.0}) pts.push_back({nu, x, 0.0});
#ifdef KOKKOS_ENABLE_CUDA
  const double tol = 1.8e-15;
#else
  const double tol = 0.0;
#endif
  double worst = 0.0;
  std::size_t differing = 0;
  for (int n = 0; n <= 7; ++n) {
    const auto single = bench::eval_kernel(pts, LogK{n});
    const auto joint7 = bench::eval_kernel(pts, LogKJoint{n, 7});
    for (std::size_t i = 0; i < pts.size(); ++i)
      if (!(std::fabs(single[i] - joint7[i]) <= tol * std::fabs(single[i]))) {
        ++differing;
        worst = std::max(worst, std::fabs(single[i] - joint7[i]) / std::max(std::fabs(single[i]), 1e-300));
      }
    if (n <= 3) {
      const auto joint3 = bench::eval_kernel(pts, LogKJoint{n, 3});
      for (std::size_t i = 0; i < pts.size(); ++i)
        if (!(std::fabs(single[i] - joint3[i]) <= tol * std::fabs(single[i]))) {
          ++differing;
          worst = std::max(worst, std::fabs(single[i] - joint3[i]) / std::max(std::fabs(single[i]), 1e-300));
        }
    }
  }
  INFO("entries that differ: ", differing, " of ", 8 * pts.size(), ", worst relative difference ", worst);
  CHECK(differing == 0);  // entries outside the tolerance
  // the special arguments
  double o[8];
  lm::log_bessel_k_dx_upto<7>(2.0, -1.0, o);
  for (int n = 1; n <= 7; ++n) CHECK(std::isnan(o[n]));
  lm::log_bessel_k_dx_upto<7>(2.0, INFINITY, o);
  CHECK(o[0] == -INFINITY);
  CHECK(o[1] == -1.0);
  for (int n = 2; n <= 7; ++n) CHECK(o[n] == 0.0);
}
