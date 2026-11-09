// The incomplete beta function in the regions where it is hard to compute
// (tests/data/reference/ibeta_hard.csv, ibetac_hard.csv, written by
// tools/reference/ibeta_hard.py): one shape small and the other large, both
// moderate or both large near the mean, both below 1, and the worst arguments
// of the first own version. Evaluated in a Kokkos kernel on the default
// execution space. The bounds are the errors measured on 2026-11-09
// (docs/numerics.md) with a margin of about half again, on CPU and GPU.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/detail/ibeta_expansions.hpp>
#include <laplace/math/incomplete_beta.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;

// component k of laplace::math: 0 ibeta, 1 ibetac, 2 log_ibeta, 3 log_ibetac
template <int K>
struct Fn {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    if constexpr (K == 0) return laplace::math::ibeta(a, b, x);
    else if constexpr (K == 1) return laplace::math::ibetac(a, b, x);
    else if constexpr (K == 2) return laplace::math::log_ibeta(a, b, x);
    else return laplace::math::log_ibetac(a, b, x);
  }
};

// the method chosen by detail_ibx::direct_side
struct Method {
  KOKKOS_INLINE_FUNCTION double operator()(double a, double b, double x) const {
    return laplace::math::own::detail_ibx::direct_side(a, b, x).method;
  }
};

// ulp1: absolute error in ulps of max(|f|, 1); ulpL = ulp / max(1, |log f|)
template <class F>
void check_value(const char* name, F f, double max_ulp1, double max_ulpL) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto a = bench::accuracy(r, bench::eval_kernel(r.x, f));
  INFO(name, ": max ", a.max_ulp1, " ulp1, ", a.max_ulpL, " ulpL, worst (", bench::show(a.worst, 3), ")");
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp1 <= max_ulp1);
  CHECK(a.max_ulpL <= max_ulpL);
}

// the logarithmic outputs against log(hi + lo), in ulps of max(|log f|, 1)
// (references below 1e-300 are skipped: their logarithm is not accurate)
template <class F>
void check_log(const char* name, F f, double max_err) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  double worst = 0.0;
  std::size_t bad = 0, at = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!(r.hi[i] > 1e-300)) continue;
    const double ref = std::log(r.hi[i]) + std::log1p(r.lo[i] / r.hi[i]);
    if (!std::isfinite(y[i])) {
      ++bad;
      continue;
    }
    const double e = std::fabs(y[i] - ref) / bench::ulp_of(std::max(std::fabs(ref), 1.0));
    if (e > worst) {
      worst = e;
      at = i;
    }
  }
  INFO(name, " (log): max ", worst, " at (", bench::show(r.x[at], 3), ")");
  CHECK(bad == 0);
  CHECK(worst <= max_err);
}

}  // namespace

// Measured 2026-11-09, gcc/OpenMP and CUDA: ulp1 3.65 on both; ulpL 18.8 (CPU)
// and 19.7 (CUDA); logarithms at most 22 ulps of max(|log f|, 1) on both.
TEST_CASE("incomplete beta in the hard regions against mpmath") {
  check_value("ibeta_hard", Fn<0>{}, 5.5, 29.5);
  check_value("ibetac_hard", Fn<1>{}, 5.5, 29.5);
  check_log("ibeta_hard", Fn<2>{}, 33.0);
  check_log("ibetac_hard", Fn<3>{}, 33.0);
  check_log("ibeta", Fn<2>{}, 33.0);
  check_log("ibetac", Fn<3>{}, 33.0);
}

// Every piece of the expansions is exercised by the hard references: a change
// of the choice that left one out would leave it untested.
TEST_CASE("incomplete beta: every method is used by the references") {
  const auto r = bench::read_ref(refdir + "/ibeta_hard.csv");
  const auto m = bench::eval_kernel(r.x, Method{});
  int count[9] = {0};
  for (double v : m) ++count[static_cast<int>(v)];
  for (int k : {1, 2, 3, 4, 6, 7, 8}) {
    INFO("method ", k);
    CHECK(count[k] > 20);
  }
}

// Special values and the relation between the four outputs.
TEST_CASE("incomplete beta: special values") {
  namespace lm = laplace::math;
  using lm::own::beta_inc;
  const double inf = lm::detail::pos_inf();
  CHECK(beta_inc(2.0, 3.0, 0.0).i == 0.0);
  CHECK(beta_inc(2.0, 3.0, 0.0).log_i == -inf);
  CHECK(beta_inc(2.0, 3.0, 1.0).ic == 0.0);
  CHECK(beta_inc(2.0, 3.0, 1.0).log_ic == -inf);
  CHECK(std::isnan(beta_inc(-1.0, 3.0, 0.5).i));
  CHECK(std::isnan(beta_inc(2.0, 3.0, 1.5).i));
  // I_x(1, 1) = x; I_x(a, 1) = x^a; I_{1/2}(a, a) = 1/2
  CHECK(std::fabs(beta_inc(1.0, 1.0, 0.3).i - 0.3) <= 1e-16);
  CHECK(std::fabs(beta_inc(2.5, 1.0, 0.4).i - std::pow(0.4, 2.5)) <= 2e-16 * std::pow(0.4, 2.5));
  for (double a : {0.01, 0.7, 3.0, 45.0, 700.0, 5000.0}) {
    const auto r = beta_inc(a, a, 0.5);
    INFO("a = ", a);
    CHECK(std::fabs(r.i - 0.5) <= 4.5e-16);
    CHECK(std::fabs(r.ic - 0.5) <= 4.5e-16);
  }
  // the logarithm of the far tail where the value underflows (mpmath: -2623.3038335091295110)
  const auto t = beta_inc(4000.0, 30.0, 0.5);
  CHECK(t.i == 0.0);
  CHECK(std::fabs(t.log_i + 2623.3038335091295110) <= 30.0 * 4.55e-13);
  CHECK(t.log_ic == 0.0);
}
