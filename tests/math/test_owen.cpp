// Owen's T and the log distribution function of the standard skew normal with
// its derivatives in (z, alpha) to order five (include/laplace/math/owen.hpp,
// laplace::math::owens_t, log_sn_cdf, log_sn_cdf_d1..d5) against the
// mpmath references, evaluated in a Kokkos kernel on the default execution
// space. Errors as in benchmarks/math_owen.cpp:
//   ulp   relative, in ulps of max(|f|, 2^-1022);
//   ulpN  for the derivatives, in ulps of max(|f|, N_ij), N_ij = |f_K0|^(i/K) |f_0K|^(j/K)
//         the geometric mean of the pure components of the order;
//   ulpL  ulpN / max(1, |log |f||).
// The order-5 bound is large because near their zeros the derivatives of order
// 3 to 5 move by up to 1.4e4 of their ulps per ulp of F (owen.hpp).
// Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/owen.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace own = laplace::math;  // the wrappers
constexpr double tiny = 2.2250738585072014e-308;

struct OwensT {
  KOKKOS_INLINE_FUNCTION double operator()(double h, double a) const { return own::owens_t(h, a); }
};
struct LogSnCdf {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double alpha) const { return own::log_sn_cdf(z, alpha); }
};

void check_value(const char* name, auto f, double max_ulp, double max_ulpL) {
  const auto r = bench::read_ref(refdir + "/" + name + ".csv");
  const auto y = bench::eval_kernel(r.x, f);
  double eu = 0, eL = 0;
  std::size_t bad = 0, wu = 0, wL = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) {
      ++bad;
      continue;
    }
    const double u = std::fabs((y[i] - r.hi[i]) - r.lo[i]) / bench::ulp_of(std::max(std::fabs(r.hi[i]), tiny));
    const double uL = r.hi[i] != 0.0 ? u / std::max(1.0, std::fabs(std::log(std::fabs(r.hi[i])))) : u;
    if (u > eu) eu = u, wu = i;
    if (uL > eL) eL = uL, wL = i;
  }
  INFO(name, ": max ", eu, " ulp at ", bench::show(r.x[wu], 2), ", max ", eL, " ulpL at ", bench::show(r.x[wL], 2));
  CHECK(bad == 0);
  CHECK(eu <= max_ulp);
  CHECK(eL <= max_ulpL);
}

template <int K>
struct Deriv {
  KOKKOS_INLINE_FUNCTION void operator()(double z, double alpha, double* o) const {
    laplace::math::shape_deriv<K> r;
    if constexpr (K == 1) r = own::log_sn_cdf_d1(z, alpha);
    if constexpr (K == 2) r = own::log_sn_cdf_d2(z, alpha);
    if constexpr (K == 3) r = own::log_sn_cdf_d3(z, alpha);
    if constexpr (K == 4) r = own::log_sn_cdf_d4(z, alpha);
    if constexpr (K == 5) r = own::log_sn_cdf_d5(z, alpha);
    for (int j = 0; j <= K; ++j) o[j] = r.c[j];
  }
};

template <int K>
void check_deriv(double max_ulpN, double max_ulpL) {
  bench::Ref ref[K + 1];
  for (int j = 0; j <= K; ++j)
    ref[j] = bench::read_ref(refdir + "/log_sn_cdf_d" + std::to_string(K - j) + std::to_string(j) + ".csv");
  const auto& x = ref[0].x;
  for (int j = 1; j <= K; ++j) REQUIRE(ref[j].x == x);
  const std::size_t n = x.size();
  const auto hx = bench::to_host_args(x);
  bench::DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double* [6], Kokkos::LayoutRight, bench::Exec::memory_space> dy("y", n);
  const Deriv<K> f{};
  Kokkos::parallel_for(
      "sn_deriv", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t r) {
        double o[K + 1];
        f(dx(r, 0), dx(r, 1), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  const auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  double eN = 0, eL = 0;
  std::size_t bad = 0, wN = 0, wL = 0;
  int jN = 0, jL = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const double fK0 = ref[0].hi[r], f0K = ref[K].hi[r];
    for (int j = 0; j <= K; ++j) {
      const double v = hy(r, j), hi = ref[j].hi[r], lo = ref[j].lo[r];
      if (!std::isfinite(v)) {
        ++bad;
        continue;
      }
      const double N = std::pow(std::fabs(fK0), double(K - j) / K) * std::pow(std::fabs(f0K), double(j) / K);
      const double u = std::fabs((v - hi) - lo) / bench::ulp_of(std::max({std::fabs(hi), N, tiny}));
      const double uL = hi != 0.0 ? u / std::max(1.0, std::fabs(std::log(std::fabs(hi)))) : u;
      if (u > eN) eN = u, wN = r, jN = j;
      if (uL > eL) eL = uL, wL = r, jL = j;
    }
  }
  INFO("log_sn_cdf_d", K, ": max ", eN, " ulpN (d", K - jN, jN, " at ", bench::show(x[wN], 2), "), max ", eL, " ulpL (d",
       K - jL, jL, " at ", bench::show(x[wL], 2), ")");
  CHECK(bad == 0);
  CHECK(eN <= max_ulpN);
  CHECK(eL <= max_ulpL);
}

}  // namespace

TEST_CASE("Owen's T and log F of the skew normal against mpmath") {
  check_value("owens_t", OwensT{}, 6.7, 3.4);
  check_value("log_sn_cdf", LogSnCdf{}, 7.5, 5.5);
}

TEST_CASE("derivatives of log F of the skew normal in (z, alpha) against mpmath") {
  check_deriv<1>(7.7, 6.5);
  check_deriv<2>(310.0, 37.0);
  check_deriv<3>(3400.0, 760.0);
  check_deriv<4>(2300.0, 1200.0);
  check_deriv<5>(1.8e5, 4.2e4);
}
