// Dawson's integral F(x) = e^(-x^2) int_0^x e^(s^2) ds and its derivatives
// F^(k)(x), k = 1..6 (include/laplace/math/dawson.hpp), against the mpmath
// references (tools/reference/dawson.py), evaluated in a Kokkos kernel on the
// default execution space. The bounds are the errors measured on the
// gcc/OpenMP and CUDA builds on 2026-11-18 (docs/numerics.md) with a margin of
// about half again.
//
// The derivative of order k has ceil(k/2) simple zeros on the positive axis,
// each with its mirror image, and the grid holds all twelve of them, so no
// relative accuracy exists there: the error is read in ulps of 1 (ulp1, the
// absolute error, which is the measure that matters for a derivative near a
// zero) wherever the scaled series is used, |x| < 8, and in ulps of the value
// itself wherever the asymptotic series is used, |x| >= 8, which is past every
// zero. The value F has no zero other than x = 0 and is checked relatively
// everywhere.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/dawson.hpp>

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

struct Dawson {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double x) const {
    switch (k) {
      case 0: return lm::dawson(x);
      case 1: return lm::dawson_d1(x);
      case 2: return lm::dawson_d2(x);
      case 3: return lm::dawson_d3(x);
      case 4: return lm::dawson_d4(x);
      case 5: return lm::dawson_d5(x);
      default: return lm::dawson_d6(x);
    }
  }
};

double eval(int k, double x) { return Dawson{k}(x); }

// the rows whose argument lies in [lo, hi) in absolute value
bench::Ref restrict_ref(const bench::Ref& r, double lo, double hi) {
  bench::Ref o;
  o.nargs = r.nargs;
  for (std::size_t i = 0; i < r.x.size(); ++i) {
    const double a = std::fabs(r.x[i][0]);
    if (a >= lo && a < hi) {
      o.x.push_back(r.x[i]);
      o.hi.push_back(r.hi[i]);
      o.lo.push_back(r.lo[i]);
    }
  }
  return o;
}

}  // namespace

TEST_CASE("Dawson's integral and its derivatives against mpmath") {
  // measured maxima (2026-11-18, gcc/OpenMP and CUDA, 2176 rows per order), see docs/numerics.md.
  // Scaled series, |x| < 8: ulp1 1.24, 1.07, 2.54, 4.63, 8.00, 13.4, 89.5.
  // Asymptotic series, |x| >= 8: ulp 6.50, 5.37, 6.14, 6.01, 5.78, 8.27, 7.95.
  const double bound_series_ulp1[7] = {1.9, 1.7, 3.8, 7.0, 12.0, 19.5, 134.0};
  const double bound_asym_ulp[7] = {10.0, 8.1, 9.3, 9.1, 8.7, 12.5, 12.0};
  for (int k = 0; k <= 6; ++k) {
    const std::string name = k == 0 ? "dawson" : "dawson_d" + std::to_string(k);
    const auto r = bench::read_ref(refdir + "/" + name + ".csv");
    const auto y = bench::eval_kernel(r.x, Dawson{k});
    const auto all = bench::accuracy(r, y);
    CHECK(all.n_bad == 0);

    const auto rs = restrict_ref(r, 0.0, 8.0);
    const auto as = bench::accuracy(rs, bench::eval_kernel(rs.x, Dawson{k}));
    INFO(name, ", scaled series: max ", as.max_ulp1, " ulp1 at x = ", as.worst[0]);
    CHECK(as.n_bad == 0);
    CHECK(as.max_ulp1 <= bound_series_ulp1[k]);

    const auto ra = restrict_ref(r, 8.0, INFINITY);
    const auto aa = bench::accuracy(ra, bench::eval_kernel(ra.x, Dawson{k}));
    INFO(name, ", asymptotic series: max ", aa.max_ulp, " ulp at x = ", aa.worst[0]);
    CHECK(aa.n_bad == 0);
    CHECK(aa.max_ulp <= bound_asym_ulp[k]);

    if (std::getenv("LAPLACE_PRINT_ERRORS"))
      std::printf("%s: series max %.4g ulp1 (%.4g ulp) at x = %.17g; asymptotic max %.4g ulp at x = %.17g\n",
                  name.c_str(), as.max_ulp1, as.max_ulp, as.worst[0], aa.max_ulp, aa.worst[0]);
  }
  // F itself has no zero away from the origin, so its relative error is bounded everywhere
  {
    const auto r = bench::read_ref(refdir + "/dawson.csv");
    const auto a = bench::accuracy(r, bench::eval_kernel(r.x, Dawson{0}));
    INFO("dawson: max ", a.max_ulp, " ulp at x = ", a.worst[0]);
    CHECK(a.max_ulp <= 16.0);  // measured 9.68 on gcc, 10.35 on CUDA
  }
}

TEST_CASE("Dawson's integral: special arguments") {
  // F^(k)(0) = 0, 1, 0, -4, 0, 32, 0 (tools/reference/dawson_forms.py), exactly
  const double at0[7] = {0.0, 1.0, 0.0, -4.0, 0.0, 32.0, 0.0};
  for (int k = 0; k <= 6; ++k) CHECK(eval(k, 0.0) == at0[k]);
  for (int k = 0; k <= 6; ++k) {
    CHECK(std::isnan(eval(k, std::nan(""))));
    CHECK(eval(k, INFINITY) == 0.0);
    CHECK(eval(k, -INFINITY) == 0.0);
  }
  // the maximum of F: F = 0.5410442246351816 at x = 0.9241388730045917, where F' = 0
  CHECK(lm::dawson(0.9241388730045917) == doctest::Approx(0.5410442246351816).epsilon(1e-15));
  CHECK(std::fabs(lm::dawson_d1(0.9241388730045917)) <= 1e-15);
  // F(x) ~ x near 0 and F(x) ~ 1/(2x) far out
  CHECK(lm::dawson(0x1p-60) == doctest::Approx(0x1p-60).epsilon(1e-15));
  CHECK(lm::dawson(1e120) == doctest::Approx(0.5e-120).epsilon(1e-15));
  CHECK(lm::dawson_d1(1e120) == doctest::Approx(-0.5e-240).epsilon(1e-14));
}

TEST_CASE("Dawson's integral: parity and the change of route") {
  // F is odd, so F^(k) is odd for k even and even for k odd; both series are
  // built on x^2 and on one factor x, so the symmetry holds bit for bit
  for (double x : {0x1p-40, 0.3, 0.9241388730045917, 2.5, 7.99, 8.0, 13.0, 1e5}) {
    for (int k = 0; k <= 6; ++k) {
      const double p = eval(k, x), m = eval(k, -x);
      if (k % 2 == 0) CHECK(m == -p);
      else CHECK(m == p);
    }
  }
  // The two routes meet at |x| = 8. The difference is read absolutely: at that
  // argument the sixth derivative is already 1.7e-4, and the error of the
  // series is bounded in ulps of 1, not relatively.
  for (int k = 0; k <= 6; ++k) {
    const double below = eval(k, std::nextafter(8.0, 0.0)), above = eval(k, 8.0);
    INFO("order ", k, ": ", below, " against ", above);
    CHECK(std::fabs(below - above) <= 1e-13);
  }
  // F'(x) = 1 - 2 x F(x), the identity the two series must share
  for (double x : {0.25, 1.0, 3.0, 6.5, 9.0, 40.0}) {
    const double lhs = lm::dawson_d1(x), rhs = 1.0 - 2.0 * x * lm::dawson(x);
    INFO("x = ", x, ": F' = ", lhs, ", 1 - 2 x F = ", rhs);
    CHECK(std::fabs(lhs - rhs) <= 1e-14 * std::max(1.0, std::fabs(lhs)));
  }
}
