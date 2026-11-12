// The normal block of laplace::math (include/laplace/math/normal.hpp) against
// the mpmath references, evaluated in a Kokkos kernel on the default execution
// space. Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA
// builds (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/normal.hpp>

#include <limits>

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
void check_deriv(int k, double max_ulp_tail, double max_ulp1_centre) {
  const std::string name = "log_Phi_d" + std::to_string(k);
  double tail = 0, centre = 0, worst_tail = 0, worst_centre = 0;
  std::size_t bad = 0;
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
      }
    }
  }
  INFO(name, ": t < -1 max ", tail, " ulp at ", worst_tail, "; t >= -1 max ", centre, " ulp1 at ", worst_centre);
  CHECK(bad == 0);
  CHECK(tail <= max_ulp_tail);
  CHECK(centre <= max_ulp1_centre);
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
  check_deriv(1, 2.0, 2.0);
  check_deriv(2, 2.5, 1.0);
  check_deriv(3, 22.0, 1.2);
  check_deriv(4, 50.0, 2.0);
  check_deriv(5, 120.0, 3.0);
  check_deriv(6, 1150.0, 1.0);
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
