// log T_nu(z), the Student t distribution function with real degrees of
// freedom, and its derivatives in (z, nu) of orders 1 to 5
// (include/laplace/math/student.hpp, laplace::math::log_student_cdf and
// log_student_cdf_d1..d5) against the mpmath references, evaluated in a Kokkos
// kernel on the default execution space; every order and component. Errors
// as in benchmarks/math_student.cpp:
//   ulpN  in ulps of max(|f|, N_ij), N_ij = |f_K0|^(i/K) |f_0K|^(j/K), the
//         geometric mean of the pure components of the order (the relative
//         error for the pure components and wherever |f| >= N_ij);
//   ulp1  in ulps of max(|f|, 1), for the pure z-components of orders 3 and 5
//         with |z| < 1 in place of ulpN and ulpL: near z = 0 these are small
//         against the terms they are composed of (at large nu, d^5 log T / dz^5
//         at z = 0 is about 2000 times smaller than its terms; d^3 has a zero
//         at nu near 6), and only their absolute error is small;
//   ulpL  ulpN / max(1, |log |f||), beyond the error inherent in a value
//         formed as exp(log) (the upper tail, where log T = log(1 - S) and S
//         comes from its logarithm).
// Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/student.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lo = laplace::math;  // the wrappers

template <int K>
struct Deriv {
  KOKKOS_INLINE_FUNCTION void operator()(double z, double nu, double* o) const {
    lo::mixed_partials<K> r;
    if constexpr (K == 1) r = lo::log_student_cdf_d1(z, nu);
    if constexpr (K == 2) r = lo::log_student_cdf_d2(z, nu);
    if constexpr (K == 3) r = lo::log_student_cdf_d3(z, nu);
    if constexpr (K == 4) r = lo::log_student_cdf_d4(z, nu);
    if constexpr (K == 5) r = lo::log_student_cdf_d5(z, nu);
    for (int j = 0; j <= K; ++j) o[j] = r.c[j];
  }
};

}  // namespace

// The kernel below sits in a function with external linkage, outside the anonymous
// namespace: an extended device lambda takes the address of its enclosing function as a
// tag, and a function with internal linkage can give the same tag in two translation
// units (note/brief_agenti.md, found on 2026-11-13).
template <int K>
void check_student_cdf_deriv(double max_ulpN, double max_ulp1, double max_ulpL) {
  bench::Ref ref[K + 1];
  for (int j = 0; j <= K; ++j)
    ref[j] = bench::read_ref(refdir + "/log_student_cdf_d" + std::to_string(K - j) + std::to_string(j) + ".csv");
  const auto& x = ref[0].x;
  for (int j = 1; j <= K; ++j) REQUIRE(ref[j].x == x);
  const std::size_t n = x.size();
  const auto hx = bench::to_host_args(x);
  bench::DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double* [6], Kokkos::LayoutRight, bench::Exec::memory_space> dy("y", n);
  const Deriv<K> f{};
  Kokkos::parallel_for(
      "student_deriv", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t r) {
        double o[K + 1];
        f(dx(r, 0), dx(r, 1), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  const auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  constexpr double tiny = 2.2250738585072014e-308;
  double eN = 0, e1 = 0, eL = 0;
  std::size_t bad = 0, wN = 0, w1 = 0, wL = 0;
  int jN = 0, j1 = 0, jL = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const double fK0 = ref[0].hi[r], f0K = ref[K].hi[r];
    for (int j = 0; j <= K; ++j) {
      const double v = hy(r, j), hi = ref[j].hi[r], lo_ = ref[j].lo[r];
      if (!std::isfinite(v)) {
        ++bad;
        continue;
      }
      const double N = std::pow(std::fabs(fK0), double(K - j) / K) * std::pow(std::fabs(f0K), double(j) / K);
      const double d = std::fabs((v - hi) - lo_);
      const double u = d / bench::ulp_of(std::max({std::fabs(hi), N, tiny}));
      const double u1 = d / bench::ulp_of(std::max(std::fabs(hi), 1.0));
      const double uL = hi != 0.0 ? u / std::max(1.0, std::fabs(std::log(std::fabs(hi)))) : u;
      // the pure z-components of odd order near z = 0 are bounded by ulp1 only
      const bool zero_z = j == 0 && (K == 3 || K == 5) && std::fabs(x[r][0]) < 1.0;
      if (!zero_z && u > eN) eN = u, wN = r, jN = j;
      if (zero_z && u1 > e1) e1 = u1, w1 = r, j1 = j;
      if (!zero_z && uL > eL) eL = uL, wL = r, jL = j;
    }
  }
  INFO("order ", K, ": max ", eN, " ulpN (d", K - jN, jN, " at ", bench::show(x[wN], 2), "), max ", e1, " ulp1 (d", K - j1, j1,
       " at ", bench::show(x[w1], 2), "), max ", eL, " ulpL (d", K - jL, jL, " at ", bench::show(x[wL], 2), ")");
  CHECK(bad == 0);
  CHECK(eN <= max_ulpN);
  CHECK(e1 <= max_ulp1);
  CHECK(eL <= max_ulpL);
}

struct Value {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double nu) const { return lo::log_student_cdf(z, nu); }
};

TEST_CASE("log T_nu(z) against mpmath") {
  const auto r = bench::read_ref(refdir + "/log_student_cdf.csv");
  const auto y = bench::eval_kernel(r.x, Value{});
  const auto a = bench::accuracy(r, y);
  INFO("max ", a.max_ulp, " ulp at ", bench::show(a.worst, 2), ", ", a.max_ulp1, " ulp1");
  CHECK(a.n_bad == 0);
  CHECK(a.max_ulp <= 285.0);
  CHECK(a.max_ulp1 <= 17.0);
}

TEST_CASE("derivatives of log T_nu(z) in (z, nu) against mpmath") {
  check_student_cdf_deriv<1>(650.0, 1.0, 12.0);
  check_student_cdf_deriv<2>(800.0, 1.0, 35.0);
  check_student_cdf_deriv<3>(510.0, 18.0, 50.0);
  check_student_cdf_deriv<4>(2200.0, 1.0, 420.0);
  check_student_cdf_deriv<5>(710.0, 145.0, 145.0);
}
