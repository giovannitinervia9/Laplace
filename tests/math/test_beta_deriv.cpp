// The derivatives of log I_x(a, b) and log(1 - I_x(a, b)) with respect to the
// shapes (include/laplace/math/incomplete_beta_deriv.hpp, laplace::math::log_ibeta_d1..d5
// and laplace::math::log_ibetac_d1..d5) against the mpmath references, evaluated in a
// Kokkos kernel on the default execution space; every order, side and
// component. Errors as in benchmarks/math_beta_deriv.cpp:
//   ulpN  in ulps of max(|f|, N_ij), N_ij = |f_K0|^(i/K) |f_0K|^(j/K), the
//         geometric mean of the pure components of the order (the relative
//         error for the pure components and wherever |f| >= N_ij);
//   ulpL  ulpN / max(1, |log |f||), beyond the error inherent in a value
//         formed as exp(log) (the side composed from in a far tail).
// Bounds: the errors measured on 2026-11-09 on the gcc/OpenMP and CUDA builds
// (docs/numerics.md) with a margin of about half again.
#include <doctest/doctest.h>

#include "measure.hpp"

#include <laplace/math/incomplete_beta_deriv.hpp>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace {

const std::string refdir = LAPLACE_REFERENCE_DIR;
namespace lm = laplace::math;  // the wrappers

template <int K>
struct Deriv {
  bool upper;
  KOKKOS_INLINE_FUNCTION void operator()(double a, double b, double x, double* o) const {
    lm::mixed_partials<K> r;
    if constexpr (K == 1) r = upper ? lm::log_ibetac_d1(a, b, x) : lm::log_ibeta_d1(a, b, x);
    if constexpr (K == 2) r = upper ? lm::log_ibetac_d2(a, b, x) : lm::log_ibeta_d2(a, b, x);
    if constexpr (K == 3) r = upper ? lm::log_ibetac_d3(a, b, x) : lm::log_ibeta_d3(a, b, x);
    if constexpr (K == 4) r = upper ? lm::log_ibetac_d4(a, b, x) : lm::log_ibeta_d4(a, b, x);
    if constexpr (K == 5) r = upper ? lm::log_ibetac_d5(a, b, x) : lm::log_ibeta_d5(a, b, x);
    if constexpr (K == 6) r = upper ? lm::log_ibetac_d6(a, b, x) : lm::log_ibeta_d6(a, b, x);
    for (int j = 0; j <= K; ++j) o[j] = r.c[j];
  }
};

// psi^(n)(z) - psi^(n)(z + h), n = 0..5, from the batched route and from the
// scalar one (which computes its asymptotic coefficients instead of reading the
// table, so the two are independent at order 5)
struct PsiDiffAll {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double h, double n) const {
    double out[6], dD;
    lm::own::detail_ibd::dpsi_all(z, h, 5, out, dD);
    return out[static_cast<int>(n)];
  }
};

struct PsiDiffOne {
  KOKKOS_INLINE_FUNCTION double operator()(double z, double h, double n) const {
    const int k = static_cast<int>(n);
    return k == 0 ? lm::own::detail_ibd::dpsi0(z, h) : lm::own::detail_ibd::dpsi(k, z, h);
  }
};

}  // namespace

// The kernel below sits in a function with external linkage, outside the anonymous
// namespace: an extended device lambda takes the address of its enclosing function as a
// tag, and a function with internal linkage can give the same tag in two translation
// units (note/brief_agenti.md, found on 2026-11-13).
template <int K>
void check_beta_shape_deriv(bool upper, double max_ulpN, double max_ulpL) {
  const std::string base = std::string(upper ? "log_ibetac_d" : "log_ibeta_d");
  bench::Ref ref[K + 1];
  for (int j = 0; j <= K; ++j) ref[j] = bench::read_ref(refdir + "/" + base + std::to_string(K - j) + std::to_string(j) + ".csv");
  const auto& x = ref[0].x;
  for (int j = 1; j <= K; ++j) REQUIRE(ref[j].x == x);
  const std::size_t n = x.size();
  const auto hx = bench::to_host_args(x);
  bench::DArgs dx("x", n);
  Kokkos::deep_copy(dx, hx);
  Kokkos::View<double* [7], Kokkos::LayoutRight, bench::Exec::memory_space> dy("y", n);
  const Deriv<K> f{upper};
  Kokkos::parallel_for(
      "beta_deriv", Kokkos::RangePolicy<bench::Exec>(0, n), KOKKOS_LAMBDA(const std::size_t r) {
        double o[K + 1];
        f(dx(r, 0), dx(r, 1), dx(r, 2), o);
        for (int j = 0; j <= K; ++j) dy(r, j) = o[j];
      });
  const auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dy);
  constexpr double tiny = 2.2250738585072014e-308;
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
  INFO(base, K, ": max ", eN, " ulpN (d", K - jN, jN, " at ", bench::show(x[wN], 3), "), max ", eL, " ulpL (d", K - jL, jL,
       " at ", bench::show(x[wL], 3), ")");
  CHECK(bad == 0);
  CHECK(eN <= max_ulpN);
  CHECK(eL <= max_ulpL);
}

// Kernels in a function with external linkage, as above.
void check_psi_diff(double max_ulp_all, double max_ulp_one) {
  const bench::Ref ref = bench::read_ref(refdir + "/psi_diff.csv");
  const auto a = bench::accuracy(ref, bench::eval_kernel(ref.x, PsiDiffAll{}));
  const auto b = bench::accuracy(ref, bench::eval_kernel(ref.x, PsiDiffOne{}));
  std::printf("psi differences: dpsi_all max %.3g ulp at %s, dpsi max %.3g ulp at %s\n", a.max_ulp,
              bench::show(a.worst, 3).c_str(), b.max_ulp, bench::show(b.worst, 3).c_str());
  INFO("dpsi_all: max ", a.max_ulp, " ulp at ", bench::show(a.worst, 3), "; dpsi: max ", b.max_ulp, " ulp at ",
       bench::show(b.worst, 3));
  CHECK(a.n_bad == 0);
  CHECK(b.n_bad == 0);
  CHECK(a.max_ulp <= max_ulp_all);
  CHECK(b.max_ulp <= max_ulp_one);
}

// measured on 2026-11-15: dpsi_all 7.69 ulp on gcc and 5.69 on CUDA, the scalar dpsi 6.69 and
// 5.86, worst at order 5 where the difference cancels (z = 40, h = 4e-7; z = 1e5, h = 1e-7);
// bound half again above the worse build
TEST_CASE("polygamma differences at a shift, orders 0 to 5, against mpmath") { check_psi_diff(12.0, 12.0); }

// The bounds below are 1.5 times the worst measured on the gcc and the CUDA builds, per order and
// per measure. They were re-measured on 2026-11-16, when the route of derivs started computing the
// requested side directly instead of composing it from the other one wherever that is the more
// accurate of the two (incomplete_beta_deriv.hpp says what the condition is and what it measured):
// every order of both functions improved, log I_x by a factor of two to four and log(1 - I_x) by up
// to two, so the bounds here are tighter than they were and not looser.
TEST_CASE("derivatives of log I_x(a, b) in the shapes against mpmath") {
  check_beta_shape_deriv<1>(false, 1130.0, 54.0);
  check_beta_shape_deriv<2>(false, 1255.0, 241.0);
  check_beta_shape_deriv<3>(false, 1155.0, 81.0);
  check_beta_shape_deriv<4>(false, 1235.0, 71.0);
  check_beta_shape_deriv<5>(false, 2155.0, 450.0);
  check_beta_shape_deriv<6>(false, 1950.0, 649.0);
}

TEST_CASE("derivatives of log(1 - I_x(a, b)) in the shapes against mpmath") {
  check_beta_shape_deriv<1>(true, 1810.0, 178.0);
  check_beta_shape_deriv<2>(true, 2170.0, 154.0);
  check_beta_shape_deriv<3>(true, 2425.0, 71.0);
  check_beta_shape_deriv<4>(true, 2400.0, 62.0);
  check_beta_shape_deriv<5>(true, 2460.0, 513.0);
  check_beta_shape_deriv<6>(true, 8040.0, 265.0);
}
