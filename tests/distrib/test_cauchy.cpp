// The tests of cauchy_distrib (include/laplace/distrib/cauchy.hpp): packing and descriptors,
// every derivative order against differences, the mixed (y, theta) derivatives, the expected
// information against the variance of the score (quadrature), the cdf family against
// identities, the derivatives of log F and log(1 - F) against mpmath (tools/reference/
// distrib_cauchy_logistic.py, both tails out to |z| = 1e8), and a kernel launch of the highest orders.
#include <doctest/doctest.h>

// Worst relative difference between the host and the device sum of the kernel case, measured
// 2026-11-13 on the CUDA build (RTX 4070 SUPER): 2.43e-14, and half again as the bound. The
// cause is FMA contraction, see the comment at the check itself.
#define CAUCHY_KERNEL_TOL 3.7e-14

#include <laplace/distrib/cauchy.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// one row of tests/data/reference/distrib_cauchy_logistic.csv
struct reference_row {
  int dist, upper, K, b;
  double y, hi, lo, mu, sigma;
};

std::vector<reference_row> read_reference(int dist) {
  std::vector<reference_row> rows;
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_cauchy_logistic.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    reference_row r;
    char ys[64], his[64], los[64], mus[64], sigs[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b, his, los,
                    mus, sigs) != 9)
      continue;
    if (r.dist != dist) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.sigma = std::strtod(sigs, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

}  // namespace

TEST_CASE("cauchy: descriptors and kinks") {
  CHECK(ld::cauchy_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::cauchy_distrib::n_params == 2);
  CHECK(ld::cauchy_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::cauchy_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::cauchy_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::cauchy_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::cauchy_distrib::params[1].moves_support == false);
  const ld::cauchy_distrib d{0.4, 1.3};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  // the logpdf at the centre: -log(pi sigma)
  CHECK(d.logpdf(0.4) == doctest::Approx(-std::log(3.14159265358979323846 * 1.3)).epsilon(1e-15));
}

TEST_CASE("cauchy: derivatives against differences, centre and tails") {
  for (double s : {0.25, 1.3, 4.0})
    for (double y : {-40.0, -3.0, 0.2, 1.7, 9.0, 300.0}) check_derivs<ld::cauchy_distrib>({0.4, s}, y, 1e-8);
}

TEST_CASE("cauchy: mixed (y, theta) derivatives against differences") {
  for (double s : {1.3, 4.0})
    for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::cauchy_distrib>({0.4, s}, y, 1e-3 * s, 1e-8);
  // the J = 1 route equals the minus of the mu-derivative of the same order
  const ld::cauchy_distrib d{0.4, 1.3};
  const auto a = d.logpdf_dy_dk<1, 0>(2.0);
  const auto b = d.logpdf_dk<1>(2.0);
  CHECK(a[0] == doctest::Approx(-b[0]).epsilon(1e-14));
}

TEST_CASE("cauchy: expected information against the variance of the score") {
  const ld::cauchy_distrib d{0.4, 1.3};
  const auto q = score_variance_continuous(d);
  const auto e = d.expected_info();
  CHECK(std::fabs(q[0] - e[0]) <= 1e-10 * e[0]);
  CHECK(std::fabs(q[1]) <= 1e-10 * e[0]);
  CHECK(std::fabs(q[2] - e[2]) <= 1e-10 * e[2]);
  const ld::cauchy_distrib d2{-2.0, 0.25};
  const auto q2 = score_variance_continuous(d2);
  const auto e2 = d2.expected_info();
  CHECK(std::fabs(q2[0] - e2[0]) <= 1e-10 * e2[0]);
  CHECK(std::fabs(q2[2] - e2[2]) <= 1e-10 * e2[2]);
}

TEST_CASE("cauchy: derivatives of the expected information against differences") {
  check_info_derivs<ld::cauchy_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::cauchy_distrib>({-2.0, 0.25}, 1e-8);
}

TEST_CASE("cauchy: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true})
    for (double y : {-40.0, -3.0, 0.2, 1.7, 9.0, 60.0})
      check_cdf_derivs<ld::cauchy_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
}

TEST_CASE("cauchy: cdf, ccdf, logarithms, quantile") {
  const ld::cauchy_distrib d{0.4, 1.3};
  for (double y : {-1e6, -40.0, -3.0, 0.4, 1.7, 9.0, 60.0, 1e6}) {
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    // the logarithm is compared with the log of the function on the side where that is not near 1
    if (d.cdf(y) < 0.5) CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
    if (d.ccdf(y) < 0.5) CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
  }
  // closed form: F(y) = 1/2 + atan(z)/pi at z = 1
  CHECK(d.cdf(0.4 + 1.3) == doctest::Approx(0.75).epsilon(1e-15));
  for (double p : {1e-12, 1e-6, 0.3, 0.5, 0.7, 1.0 - 1e-6}) {
    const double q = d.quantile(p);
    if (p <= 0.5) CHECK(d.cdf(q) == doctest::Approx(p).epsilon(1e-11));
    else CHECK(d.ccdf(q) == doctest::Approx(1.0 - p).epsilon(1e-11));
  }
}

TEST_CASE("cauchy: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_cauchy_logistic.py: mpmath.diff of the definition at 300 and 400 bits.
  // Each component is compared in ulp of the largest component of its order at that point (the
  // components change sign, so a plain relative error is undefined at their zeros), and in ulp of
  // the component itself; the error is that of the difference of the library value and the
  // reference (hi + lo) on the library's side.
  const auto rows = read_reference(0);
  CHECK(rows.size() > 2000);
  std::map<std::tuple<int, double, int, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.mu, r.sigma)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double w_scaled[2][7] = {}, w_own[2][7] = {};
  for (const auto& r : rows) {
    const ld::cauchy_distrib d{r.mu, r.sigma};
    const auto v = cderiv(d, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.y, r.K, r.mu, r.sigma)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    const double ulp_own = err / std::ldexp(1.0, std::ilogb(r.hi) - 52);
    w_scaled[r.upper][r.K] = std::fmax(w_scaled[r.upper][r.K], ulp_scaled);
    w_own[r.upper][r.K] = std::fmax(w_own[r.upper][r.K], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " mu " << r.mu << " sigma " << r.sigma
                  << " ulp " << ulp_scaled);
    // bound per side: measured on the gcc build on 2026-11-13 (scaled ulp, maximum over orders 1 to 6):
    // log F 59 (order 6), log(1 - F) 41 (order 6); bounds half again
    const double bound[2] = {90.0, 62.0};
    CHECK(ulp_scaled <= bound[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("cauchy log_%s derivs, order %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n",
                    u ? "(1-F)" : "F", K, w_scaled[u][K], w_own[u][K]);
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void cauchy_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "cauchy_distrib", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::cauchy_distrib c{0.4, 1.3};
        out(i) = c.logpdf_d6(y)[3] + c.logpdf_dy_dk<2, 3>(y)[1] + c.expected_info_d4().pair[2][4] +
                 c.log_cdf_d6(y)[6] + c.log_ccdf_d6(y + 1.0)[2] + c.logpdf(y) + c.quantile(0.25 + 0.005 * i);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::cauchy_distrib c{0.4, 1.3};
    const double r = c.logpdf_d6(y)[3] + c.logpdf_dy_dk<2, 3>(y)[1] + c.expected_info_d4().pair[2][4] +
                     c.log_cdf_d6(y)[6] + c.log_ccdf_d6(y + 1.0)[2] + c.logpdf(y) + c.quantile(0.25 + 0.005 * i);
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("cauchy kernel: worst relative host-device difference %.3g\n", worst);
  // The far tail sums about a hundred alternating coefficients, and nvcc contracts the
  // multiply-adds of that sum into FMA while the host compiler does not, so the two paths
  // round differently; no other distribution of the batch runs a series of that length. With
  // LAPLACE_CUDA_FMAD=OFF the two agree to 1e-14. Bound: the measured difference and half
  // again (measured 2026-11-13 on the CUDA build).
  CHECK(worst <= CAUCHY_KERNEL_TOL);
}

TEST_CASE("cauchy: every order inside a kernel") { cauchy_kernel_case(); }
