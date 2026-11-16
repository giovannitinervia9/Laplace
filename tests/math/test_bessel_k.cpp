// log K_nu(x) and its derivatives in x to order 7
// (include/laplace/math/bessel_k.hpp) against the mpmath references,
// evaluated in a Kokkos kernel on the default execution space. Bounds: the
// errors measured on 2026-11-11 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

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
struct LogKScaled {
  int n;
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const {
    double out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    lm::log_bessel_k_scaled_dx_upto<7>(nu, x, out);
    return out[n];
  }
};

struct LogKScaledValue {
  KOKKOS_INLINE_FUNCTION double operator()(double nu, double x) const { return lm::log_bessel_k_scaled(nu, x); }
};

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

// The scaled pair log K + x and (log K)' + 1. The reference is the pair (hi, lo) of the plain
// quantity plus the shift, summed without error in the test (one two_sum, the shift being exact as a
// double), so that what the case measures is the error of the scaled routine alone and not of a
// difference formed here. The bounds are the errors measured on 2026-11-16 on the gcc/OpenMP and
// CUDA builds with a margin of about half again. The orders two and up are the plain ones and must
// agree to the bit.
//
// Where the value stands, worst by regime: 0.488 ulp for x <= 2 (Temme's series with x added, within
// its eight-eps floor), 35.2 ulp at nu = 4, x = 9.5 for the continued fraction, and 101 ulp at
// nu = 248, x = 7635 for the Debye expansion. Above x = 2 the scaled value is a sum of terms of the
// expansion that can be tens of times larger than the result it leaves, so it is accurate to about
// 1e-13 absolute rather than to its own ulp; the orders that the distributions ask for (nu = 1 for
// the pseudo-Huber) sit far below that, as their own reference grids show. The first derivative is at
// or below 1.17 ulp in every regime.
namespace {

double two_sum_of(double a, double b, double& e) {
  const double s = a + b;
  e = (std::fabs(a) >= std::fabs(b)) ? ((a - s) + b) : ((b - s) + a);
  return s;
}

// Worst error of the scaled entry, in units of one ulp of the entry itself plus the irreducible error
// of whatever cancellation the route could not avoid:
//   - the first derivative, (log K)' + 1, has a zero near x = 1, where the sum cancels whatever route
//     forms it, so the floor is four eps of the plain derivative;
//   - the value for x <= 2 is Temme's log K_mu with x added, the two being of the same size there, so
//     its floor is eight eps of log K, which is the accuracy the plain value itself is measured to
//     (7.5 ulp of max(|log K|, |x g|, 1)); above x = 2 no x is ever added, the floor is zero and the
//     case is strict, which is where the shift earns its keep.
// `shift` is x for the value and 1 for the first derivative.
double scaled_worst(const bench::Ref& r, const std::vector<double>& y, bool shift_is_x, double& wn, double& wx) {
  const double eps = std::numeric_limits<double>::epsilon();
  double worst = 0.0;
  double by_regime[3] = {0, 0, 0}, nu_regime[3] = {0, 0, 0}, x_regime[3] = {0, 0, 0};
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double shift = shift_is_x ? r.x[i][1] : 1.0;
    double e = 0.0;
    const double s = two_sum_of(r.hi[i], shift, e);
    const double want = (s + e) + r.lo[i];
    const double err = std::fabs(((y[i] - s) - e) - r.lo[i]);
    const double mag = std::max(1.0, std::fabs(r.hi[i]));
    const double floor_of_the_cancellation =
        shift_is_x ? (r.x[i][1] <= 2.0 ? 8.0 * eps * mag : 0.0) : 4.0 * eps * mag;
    const double u = bench::ulp_of(std::max(std::fabs(want), 1e-300)) + floor_of_the_cancellation;
    if (err / u > worst) worst = err / u, wn = r.x[i][0], wx = r.x[i][1];
    const int reg = r.x[i][1] <= 2.0 ? 0 : (r.x[i][0] >= 20.0 ? 2 : 1);
    if (err / u > by_regime[reg]) by_regime[reg] = err / u, nu_regime[reg] = r.x[i][0], x_regime[reg] = r.x[i][1];
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("  scaled, worst by regime (x <= 2 | continued fraction | Debye): %.3g at nu %.6g x %.6g | %.3g at nu %.6g x %.6g | %.3g at nu %.6g x %.6g\n",
                by_regime[0], nu_regime[0], x_regime[0], by_regime[1], nu_regime[1], x_regime[1], by_regime[2],
                nu_regime[2], x_regime[2]);
  return worst;
}

}  // namespace

TEST_CASE("the scaled pair of log K against mpmath, and the higher orders unshifted") {
  // the entries the shift does not touch are the same numbers as the plain ones, exactly on the CPU
  // build; on CUDA nvcc contracts and schedules the two instantiations differently, as it already does
  // between log_bessel_k_dx_upto and the single-order functions
#ifdef KOKKOS_ENABLE_CUDA
  const double same_tol = 1.8e-15;
#else
  const double same_tol = 0.0;
#endif
  double wn = 0, wx = 0;
  {
    const auto r = bench::read_ref(refdir + "/log_bessel_k.csv");
    const auto y = bench::eval_kernel(r.x, LogKScaled{0});
    const auto v = bench::eval_kernel(r.x, LogKScaledValue{});
    std::size_t differing = 0;
    double worst_pair = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i)
      if (!(std::fabs(y[i] - v[i]) <= same_tol * std::fabs(y[i]))) {
        ++differing;
        worst_pair = std::max(worst_pair, std::fabs(y[i] - v[i]) / std::max(std::fabs(y[i]), 1e-300));
      }
    INFO("the joint routine against log_bessel_k_scaled: ", differing, " of ", y.size(), " differ, worst ", worst_pair);
    CHECK(differing == 0);
    const double worst = scaled_worst(r, y, true, wn, wx);
    INFO("log_bessel_k_scaled: max ", worst, " ulp at nu = ", wn, ", x = ", wx);
    CHECK(worst <= 160.0);
  }
  {
    const auto r = bench::read_ref(refdir + "/log_bessel_k_dx1.csv");
    const auto y = bench::eval_kernel(r.x, LogKScaled{1});
    const double worst = scaled_worst(r, y, false, wn, wx);
    INFO("the first derivative scaled: max ", worst, " ulp at nu = ", wn, ", x = ", wx);
    CHECK(worst <= 2.0);
  }
  // the orders two and up are the plain ones, to the bit, on every reference grid and regime boundary
  std::vector<bench::Args> pts;
  for (int n = 2; n <= 7; ++n) {
    const auto r = bench::read_ref(refdir + "/log_bessel_k_dx" + std::to_string(n) + ".csv");
    pts.insert(pts.end(), r.x.begin(), r.x.end());
  }
  for (double nu : {0.0, 0.1, 1.0, 19.99, 20.0, 20.01, 300.0})
    for (double x : {0.5, 1.0, 2.0, 199.9, 200.0, 200.1, 1000.0}) pts.push_back({nu, x, 0.0});
  std::size_t differing = 0;
  double worst_order = 0.0;
  for (int n = 2; n <= 7; ++n) {
    const auto plain = bench::eval_kernel(pts, LogKJoint{n, 7});
    const auto scaled = bench::eval_kernel(pts, LogKScaled{n});
    for (std::size_t i = 0; i < pts.size(); ++i)
      if (!(std::fabs(plain[i] - scaled[i]) <= same_tol * std::fabs(plain[i]))) {
        ++differing;
        worst_order = std::max(worst_order, std::fabs(plain[i] - scaled[i]) / std::max(std::fabs(plain[i]), 1e-300));
      }
  }
  INFO("orders two to seven that differ from the plain ones: ", differing, " of ", 6 * pts.size(), ", worst ", worst_order);
  CHECK(differing == 0);
  // the special arguments
  double o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  lm::log_bessel_k_scaled_dx_upto<7>(2.0, INFINITY, o);
  CHECK(o[0] == 0.0);
  CHECK(o[1] == 0.0);
  for (int n = 2; n <= 7; ++n) CHECK(o[n] == 0.0);
  CHECK(lm::log_bessel_k_scaled(2.0, INFINITY) == 0.0);
  CHECK(lm::log_bessel_k_scaled(2.0, 0.0) == INFINITY);
  CHECK(std::isnan(lm::log_bessel_k_scaled(2.0, -1.0)));
  lm::log_bessel_k_scaled_dx_upto<7>(2.0, 0.0, o);
  CHECK(o[0] == INFINITY);
  for (int n = 1; n <= 7; ++n) CHECK(std::isnan(o[n]));
}
