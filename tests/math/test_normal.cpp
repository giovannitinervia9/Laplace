// The normal block of laplace::math (include/laplace/math/normal.hpp) against
// the mpmath references, evaluated in a Kokkos kernel on the default execution
// space. Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA
// builds (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/normal.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

struct Erfc {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::erfc(x); }
};
struct Erfcx {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::erfcx(x); }
};
struct Phi {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::Phi(x); }
};
struct LogPhi {
  KOKKOS_INLINE_FUNCTION double operator()(double x) const { return laplace::math::log_Phi(x); }
};
struct LogPhiD {
  int k;
  KOKKOS_INLINE_FUNCTION double operator()(double t) const {
    switch (k) {
      case 1: return laplace::math::log_Phi_d1(t);
      case 2: return laplace::math::log_Phi_d2(t);
      case 3: return laplace::math::log_Phi_d3(t);
      case 4: return laplace::math::log_Phi_d4(t);
      case 5: return laplace::math::log_Phi_d5(t);
      default: return laplace::math::log_Phi_d6(t);
    }
  }
};
struct PhiInv {
  KOKKOS_INLINE_FUNCTION double operator()(double p) const { return laplace::math::Phi_inv(p); }
};
struct PhiInvLogp {
  KOKKOS_INLINE_FUNCTION double operator()(double l) const { return laplace::math::Phi_inv_logp(l); }
};

// ulp of |a|, with subnormal values counted as 2^-1022 (their relative error
// is not the function's)
double ulp_normal(double a) { return bench::ulp_of(std::max(std::fabs(a), 2.2250738585072014e-308)); }

void check(const char* name, auto f, double max_ulp, double max_ulp1) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  const auto a = bench::accuracy(r, y);
  INFO(name, ": max ", a.max_ulp, " ulp, max ", a.max_ulp1, " ulp1, worst x ", a.worst[0]);
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= max_ulp);
  CHECK(a.max_ulp1 <= max_ulp1);
}

// Derivatives of log Phi: the relative error below t = -1, where they have no
// zeros; above, where they cross zero, the error in ulps of max(|g|, 1).
// The rows of the main file and of the dense file log_Phi_d{k}_mid.csv (t in [-1, 10]).
void check_deriv(int k, double max_ulp_tail, double max_ulp1_centre, double max_ulp_band = 1e300,
                 double max_ulp_rel = 1e300) {
  const std::string name = "log_Phi_d" + std::to_string(k);
  double tail = 0, centre = 0, worst_tail = 0, worst_centre = 0;
  std::size_t bad = 0;
  // Two figures on [-1, 4), where the ulp1 measure above cannot see a relative error:
  // there the components of the high orders are of order 1e-2 to 1e-3, so a hundred ulp
  // of the component is half an ulp of max(|g|, 1). `band` is the error in ulp of the
  // largest |g| of that order on the band and `rel` the error relative to the component
  // itself at the worst point, reported with the value and with its ratio to the largest
  // |g|. Both are bounded: since 2026-11-14 the pieces of each order are placed so that
  // every zero of the component is the midpoint of its piece and is factored out of the
  // expansion there, which makes the route relatively accurate at the zeros as well.
  std::vector<std::array<double, 3>> mid;  // t, exact value, absolute error
  for (const char* suffix : {"", "_mid"}) {
    const auto r = bench::read_ref(refdir + "/" + name + suffix + ".csv");
    const auto y = bench::eval_kernel(r.x, LogPhiD{k});
    for (std::size_t i = 0; i < y.size(); ++i) {
      if (!std::isfinite(y[i])) {
        ++bad;
        continue;
      }
      const double e = std::fabs((y[i] - r.hi[i]) - r.lo[i]);
      const double t = r.x[i][0];
      if (t < -1.0) {
        const double u = e / ulp_normal(r.hi[i]);
        if (u > tail) tail = u, worst_tail = t;
      } else {
        const double u = e / bench::ulp_of(std::max(std::fabs(r.hi[i]), 1.0));
        if (u > centre) centre = u, worst_centre = t;
        if (t < 4.0) mid.push_back({t, r.hi[i], e});
      }
    }
  }
  double scale = 0;
  for (const auto& m : mid) scale = std::fmax(scale, std::fabs(m[1]));
  double band = 0, band_t = 0, rel = 0, rel_t = 0, rel_v = 0;
  for (const auto& m : mid) {
    const double b = m[2] / bench::ulp_of(scale);
    if (b > band) band = b, band_t = m[0];
    if (m[1] == 0.0) continue;
    const double u = m[2] / ulp_normal(m[1]);
    if (u > rel) rel = u, rel_t = m[0], rel_v = m[1];
  }
  INFO(name, ": t < -1 max ", tail, " ulp at ", worst_tail, "; t >= -1 max ", centre, " ulp1 at ", worst_centre,
       "; [-1, 4) max ", band, " ulp of the largest |g| of the order (", scale, ") at ", band_t,
       "; relative to the component itself ", rel, " ulp at ", rel_t, " (value ", rel_v, ", ", rel_v / scale,
       " of the largest)");
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("%s: tail %.3g ulp | centre %.3g ulp1 | [-1, 4) %.3g ulp of max |g| = %.3g at t = %.4g | relative to "
                "the component %.3g ulp at t = %.4g (value %.3g, ratio to the largest %.2e)\n",
                name.c_str(), tail, centre, band, scale, band_t, rel, rel_t, rel_v, rel_v / scale);
  CHECK(bad == 0);
  CHECK(tail <= max_ulp_tail);
  CHECK(centre <= max_ulp1_centre);
  CHECK(band <= max_ulp_band);
  CHECK(rel <= max_ulp_rel);
}

// Quantiles: the error in units of max(ulp(x*), dx), dx the change of the exact
// x* caused by one ulp of the input s (p or log p); ds/dx = phi(x*) for p and
// R(x*) = phi(x*)/Phi(x*) for log p.
void check_quantile(const char* name, auto f, bool from_log, double max_ulpc) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  double worst = 0, worst_s = 0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) {
      ++bad;
      continue;
    }
    const double s = r.x[i][0], x = r.hi[i];
    const double slope = from_log ? laplace::math::log_Phi_d1(x)
                                  : std::exp(-0.5 * x * x) * 0.39894228040143267794;
    const double dx = bench::ulp_of(s) / slope;
    const double u = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / std::max(bench::ulp_of(x), dx);
    if (u > worst) worst = u, worst_s = s;
  }
  INFO(name, ": max ", worst, " ulpC at ", worst_s);
  CHECK(bad == 0);
  CHECK(worst <= max_ulpc);
}

}  // namespace

TEST_CASE("erfc, erfcx, Phi, log Phi against mpmath") {
  check("erfc", Erfc{}, 5.5, 2.5);
  check("erfcx", Erfcx{}, 3.0, 3.0);
  check("Phi", Phi{}, 6.0, 1.5);
  check("log_Phi", LogPhi{}, 6.0, 3.0);
}

TEST_CASE("derivatives of log Phi against mpmath") {
  // the third bound is the relative error on [-1, 4) away from the zeros of the component,
  // measured on 2026-11-13 (gcc): 3.57, 4.88, 6.63, 43.2, 65.3, 36.8 ulp, each with half again
  // on [-1, 4): the third bound in ulp of the largest |g| of the order, the fourth
  // relative to the component itself, each about one and a half times the larger of the
  // two builds' measured maxima
  check_deriv(1, 2.0, 2.0, 1.8, 3.9);
  check_deriv(2, 2.5, 1.0, 2.0, 3.4);
  check_deriv(3, 22.0, 1.2, 2.1, 3.4);
  check_deriv(4, 50.0, 2.0, 2.2, 3.7);
  check_deriv(5, 120.0, 3.0, 1.4, 4.3);
  check_deriv(6, 1150.0, 1.0, 2.6, 4.1);
}

TEST_CASE("normal quantile from p and from log p against mpmath") {
  check_quantile("qnorm", PhiInv{}, false, 3.5);
  check_quantile("Phi_inv_logp", PhiInvLogp{}, true, 4.5);
}

TEST_CASE("normal block at the ends of the range") {
  namespace lm = laplace::math;
  const double inf = std::numeric_limits<double>::infinity();
  // erfcx overflows to +inf below -26.6288 (not inf * 0 = NaN)
  CHECK(std::isfinite(lm::erfcx(-26.628)));
  for (double x : {-26.629, -27.0, -32.0, -1e200, -inf}) CHECK(lm::erfcx(x) == inf);
  CHECK(lm::erfcx(inf) == 0.0);
  // erfc and Phi saturate; the exact square would overflow from 1.3e154
  for (double x : {27.3, 1e154, 1.4e154, 1e200, inf}) {
    CHECK(lm::erfc(x) == 0.0);
    CHECK(lm::erfc(-x) == 2.0);
  }
  for (double x : {40.5, 1e200, inf}) {
    CHECK(lm::Phi(-x) == 0.0);
    CHECK(lm::Phi(x) == 1.0);
    CHECK(lm::log_Phi(x) == 0.0);
  }
  // log Phi is finite down to -sqrt(2 DBL_MAX) = -1.8961e154, then -inf
  CHECK(std::isfinite(lm::log_Phi(-1.896e154)));
  for (double x : {-1.8962e154, -1e200, -inf}) CHECK(lm::log_Phi(x) == -inf);
  CHECK(lm::Phi_inv_logp(-inf) == -inf);
  CHECK(lm::Phi_inv_logp(0.0) == inf);
  CHECK(std::isnan(lm::erfc(std::nan(""))));
  CHECK(std::isnan(lm::log_Phi(std::nan(""))));
}
