// gaussian3_distrib (mean mu, precision tau = 1/sigma^2; include/laplace/distrib/gaussian3.hpp): the
// descriptors, every derivative order against a 5-point difference of the order below, the mixed
// (y, theta) derivatives, the expected information against the variance of the score and its
// derivatives against differences, the derivatives of log F and log(1 - F) against differences and
// against mpmath (tools/reference/distrib_gaussian23.py), the cdf family, and a kernel launch of the
// highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian3.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("distrib gaussian3: descriptors") {
  CHECK(ld::gaussian3_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::gaussian3_distrib::n_params == 2);
  CHECK(ld::gaussian3_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::gaussian3_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::gaussian3_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::gaussian3_distrib::params[j].moves_support == false);
  }
  const ld::gaussian3_distrib d{0.4, 0.6};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.scale() == doctest::Approx(1.0 / std::sqrt(0.6)).epsilon(1e-15));
}

TEST_CASE("distrib gaussian3: derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::gaussian3_distrib>({0.4, 0.6}, y, 1e-8);
  check_derivs<ld::gaussian3_distrib>({-20.0, 0.35}, -18.5, 1e-8);  // large mean difference
  for (double y : {-1.0, 0.4, 0.9}) check_derivs<ld::gaussian3_distrib>({0.4, 16.0}, y, 1e-8);  // large tau
  for (double y : {-30.0, 0.4, 45.0}) check_derivs<ld::gaussian3_distrib>({0.4, 0.01}, y, 1e-8);  // small tau
}

TEST_CASE("distrib gaussian3: derivatives in y and mixed (y, theta) against differences") {
  const ld::gaussian3_distrib d{0.4, 0.6};
  CHECK(d.logpdf_dy_dk<0, 0>(1.3)[0] == doctest::Approx(d.logpdf(1.3)).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<0, 3>(1.3)[2] == d.logpdf_d3(1.3)[2]);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::gaussian3_distrib>({0.4, 0.6}, y, 1e-3, 1e-8);
  for (double y : {-1.0, 0.9}) check_y_derivs<ld::gaussian3_distrib>({0.4, 4.0}, y, 1e-3, 1e-8);
}

TEST_CASE("distrib gaussian3: expected information against the variance of the score") {
  for (double tau : {0.04, 0.6, 25.0}) {
    const ld::gaussian3_distrib d{0.4, tau};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("gaussian3, tau = " << tau);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-11 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-12 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-11 * e[2]);
  }
}

TEST_CASE("distrib gaussian3: derivatives of the expected information against differences") {
  check_info_derivs<ld::gaussian3_distrib>({0.4, 0.6}, 1e-8);
  check_info_derivs<ld::gaussian3_distrib>({0.4, 16.0}, 1e-8);
  check_info_derivs<ld::gaussian3_distrib>({-7.0, 0.04}, 1e-8);
}

TEST_CASE("distrib gaussian3: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    const double s = 1.0 / std::sqrt(0.6);
    for (double y : {-12.0, -3.0, 0.2, 1.7, 9.0, 20.0})
      check_cdf_derivs<ld::gaussian3_distrib>({0.4, 0.6}, y, upper, 1e-7, std::fabs(y - 0.4) / s);
    const double s2 = 1.0 / std::sqrt(16.0);
    for (double z : {-6.0, -4.5, -2.0, 0.0, 3.0, 5.0})
      check_cdf_derivs<ld::gaussian3_distrib>({0.4, 16.0}, 0.4 + s2 * z, upper, 1e-7, std::fabs(z));
  }
}

TEST_CASE("distrib gaussian3: log F of tau against the first derivative in sigma of gaussian1") {
  // d/dtau = (dsigma/dtau) d/dsigma with sigma = tau^(-1/2): dsigma/dtau = -sigma/(2 tau); the derivative in
  // mu is the same
  const double tau = 0.6, sigma = 1.0 / std::sqrt(tau);
  for (double y : {-6.0, -1.0, 0.3, 2.0, 7.0}) {
    const ld::gaussian3_distrib g3{0.4, tau};
    const ld::gaussian1_distrib g1{0.4, sigma};
    for (bool upper : {false, true}) {
      const auto a = upper ? g3.log_ccdf_d1(y) : g3.log_cdf_d1(y);
      const auto b = upper ? g1.log_ccdf_d1(y) : g1.log_cdf_d1(y);
      INFO("y = " << y << ", upper " << upper);
      CHECK(a[0] == doctest::Approx(b[0]).epsilon(1e-12).scale(1.0));
      CHECK(a[1] == doctest::Approx(-sigma / (2.0 * tau) * b[1]).epsilon(1e-12).scale(1.0));
    }
  }
}

TEST_CASE("distrib gaussian3: cdf, ccdf, logarithms, quantile") {
  const ld::gaussian3_distrib g{0.4, 0.6};
  const double sigma = 1.0 / std::sqrt(0.6);
  for (double y : {-30.0, -2.0, 0.4, 1.0, 5.0, 30.0}) {
    CHECK(g.cdf(y) == doctest::Approx(0.5 * std::erfc(-(y - 0.4) / (sigma * std::sqrt(2.0)))).epsilon(1e-14));
    CHECK(g.cdf(y) + g.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-14));
    if (std::fabs(y) < 10.0) {
      CHECK(g.log_cdf(y) == doctest::Approx(std::log(g.cdf(y))).epsilon(1e-14));
      CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE. doctest expands TEST_CASE to "static void DOCTEST_ANON_FUNC_<n>()", which has
// internal linkage and whose number restarts in every translation unit, while nvcc tags each
// extended __device__ lambda with the address of its enclosing function. Two units that both
// reach the same number therefore produce the same mangled tag for two different closure types,
// the linker merges them, and the function-local static std::map of Kokkos's
// configure_max_dynamic_shmem (Cuda/Kokkos_Cuda_KernelLaunch.hpp) is destroyed twice at exit:
// every assertion passes and the process then dies, with an access violation for one case and
// heap corruption for a whole run. nvcc's manual states the restriction: an extended lambda must
// not be defined inside a function with internal linkage. Measured on 2026-11-13; the fault is
// present with and without relocatable device code, and /OPT:NOICF does not avoid it.
void gaussian3_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_gaussian3", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::gaussian3_distrib g{0.4, 0.6};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.expected_info_d4().pair[0][1] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y - 2.0)[0] + g.log_cdf(y) +
                 g.log_ccdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::gaussian3_distrib g{0.4, 0.6};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.expected_info_d4().pair[0][1] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y - 2.0)[0] + g.log_cdf(y) +
                     g.log_ccdf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("distrib gaussian3: every order inside a kernel") { gaussian3_kernel_case(); }

TEST_CASE("distrib gaussian3: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_gaussian23.py (mean 0.4, tau = 0.6 and 16), 2026-11-13. Errors as in
  // test_gaussian2.cpp: "scaled" in ulp of the largest component of the order at that y, "own" in ulp of
  // the component itself.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_gaussian23.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo, mu, w; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64], ws[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b, his,
                    los, mus, ws) != 9)
      continue;
    if (r.dist != 1) continue;  // gaussian2 rows are in test_gaussian2.cpp
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.w = std::strtod(ws, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 500);
  std::map<std::tuple<int, double, int, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.w)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {}, worst_y[2][7] = {}, worst_w[2][7] = {};
  for (const auto& r : rows) {
    const ld::gaussian3_distrib g{r.mu, r.w};
    const auto v = cderiv(g, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.y, r.K, r.w)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    if (ulp_scaled > worst_scaled[r.upper][r.K]) worst_y[r.upper][r.K] = r.y, worst_w[r.upper][r.K] = r.w;
    worst_scaled[r.upper][r.K] = std::fmax(worst_scaled[r.upper][r.K], ulp_scaled);
    worst_own[r.upper][r.K] = std::fmax(worst_own[r.upper][r.K], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " w " << r.w << " ulp " << ulp_scaled
                  << " own " << ulp_own);
    // bounds: measured maxima (scaled, per side) times about 1.5; set after the first measurement
    // measured 2026-11-13 (gcc, Serial + OpenMP build): scaled maxima 301 (log F) and 156 (log(1 - F)), at order 6;
    // bounds half again
    const double bound_scaled[2] = {460.0, 240.0};
    CHECK(ulp_scaled <= bound_scaled[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("gaussian3 upper %d K %d: max %.3g ulp of the order's scale (y %.6g, w %.6g), %.3g ulp of the component\n", u, K,
                    worst_scaled[u][K], worst_y[u][K], worst_w[u][K], worst_own[u][K]);
}

TEST_CASE("distrib gaussian3: quantile inverts the cdf in both tails") {
  const ld::gaussian3_distrib g{0.4, 0.04};
  for (double z : {-6.0, -4.0, 2.0, 5.0}) {
    const double y = 0.4 + 5.0 * z;
    CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-10));
  }
}
