// invgauss_distrib (mean mu, dispersion phi; include/laplace/distrib/invgauss.hpp): packing and
// descriptors, every derivative order against a 5-point difference of the order below, the mixed
// (y, theta) derivatives, the expected information against the variance of the score and its
// derivatives against differences, the derivatives of log F and log(1 - F) against differences and
// against mpmath (tools/reference/distrib_invgauss.py, from the definition), the values of the cdf,
// the generic quantile against the cdf in both tails, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/invgauss.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("distrib invgauss: descriptors") {
  CHECK(ld::invgauss_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::invgauss_distrib::n_params == 2);
  CHECK(ld::invgauss_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::invgauss_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::invgauss_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::invgauss_distrib::params[j].moves_support == false);
  }
  const ld::invgauss_distrib d{0.4, 1.69};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.center() == doctest::Approx(0.4));
  CHECK(d.scale() == doctest::Approx(std::sqrt(1.69 * 0.4 * 0.4 * 0.4)).epsilon(1e-15));
}

TEST_CASE("distrib invgauss: derivatives against differences") {
  for (double y : {0.3, 1.0, 2.5}) check_derivs<ld::invgauss_distrib>({1.0, 0.5}, y, 1e-7);
  for (double y : {0.1, 0.4, 1.7}) check_derivs<ld::invgauss_distrib>({0.4, 0.05}, y, 1e-7);
  for (double y : {0.19, 0.25}) check_derivs<ld::invgauss_distrib>({0.2, 0.002}, y, 1e-7);
  for (double y : {0.5, 4.0, 30.0}) check_derivs<ld::invgauss_distrib>({2.0, 3.0}, y, 1e-7);
}

TEST_CASE("distrib invgauss: derivatives in y and mixed (y, theta) against differences") {
  const ld::invgauss_distrib d{0.4, 1.69};
  CHECK(d.logpdf_dy_dk<0, 0>(1.3)[0] == doctest::Approx(d.logpdf(1.3)).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<0, 3>(1.3)[2] == d.logpdf_d3(1.3)[2]);
  for (double y : {0.3, 1.3, 4.0}) check_y_derivs<ld::invgauss_distrib>({0.4, 1.69}, y, 1e-3 * y, 1e-7);
  for (double y : {0.15, 0.4}) check_y_derivs<ld::invgauss_distrib>({0.2, 0.002}, y, 1e-3 * y, 1e-7);
}

TEST_CASE("distrib invgauss: expected information against the variance of the score") {
  for (auto th : {std::pair<double, double>{0.4, 1.69}, {1.0, 0.5}, {2.0, 0.1}}) {
    const ld::invgauss_distrib d{th.first, th.second};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("invgauss, mu = " << th.first << ", phi = " << th.second);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-7 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-7 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-7 * e[2]);
  }
}

TEST_CASE("distrib invgauss: derivatives of the expected information against differences") {
  check_info_derivs<ld::invgauss_distrib>({0.4, 1.69}, 1e-7);
  check_info_derivs<ld::invgauss_distrib>({2.0, 0.1}, 1e-7);
  check_info_derivs<ld::invgauss_distrib>({0.2, 0.002}, 1e-7);
}

// the larger of |z1| and |z2| at (y, mu, phi): the scale on which log F varies, for the step of the differences
double z_scale(double y, double mu, double phi) {
  const double s = std::sqrt(phi * y);
  return std::fmax(std::fabs(y / mu - 1.0), std::fabs(y / mu + 1.0)) / s;
}

TEST_CASE("distrib invgauss: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {0.1, 0.4, 1.0, 2.5, 9.0})
      check_cdf_derivs<ld::invgauss_distrib>({0.4, 1.69}, y, upper, 1e-6, z_scale(y, 0.4, 1.69));
    for (double y : {0.19, 0.2, 0.22})
      check_cdf_derivs<ld::invgauss_distrib>({0.2, 0.002}, y, upper, 1e-6, z_scale(y, 0.2, 0.002));
    for (double y : {0.5, 1.0, 9.0})
      check_cdf_derivs<ld::invgauss_distrib>({2.0, 3.0}, y, upper, 1e-6, z_scale(y, 2.0, 3.0));
  }
}

TEST_CASE("distrib invgauss: cdf, ccdf, logarithms, quantile") {
  const ld::invgauss_distrib g{0.4, 1.69};
  for (double y : {0.01, 0.2, 0.4, 1.0, 5.0, 50.0}) {
    CHECK(g.cdf(y) + g.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-13));
    if (g.cdf(y) > 1e-300) CHECK(g.log_cdf(y) == doctest::Approx(std::log(g.cdf(y))).epsilon(1e-13));
  }
  // quantile by the generic route: inverts the cdf in the lower tail, the body and the upper tail
  for (const auto& th : {std::pair<double, double>{0.4, 1.69}, {0.2, 0.002}, {2.0, 0.5}}) {
    const ld::invgauss_distrib d{th.first, th.second};
    for (double p : {1e-12, 1e-4, 0.2, 0.5, 0.9, 1.0 - 1e-6}) {
      const double y = ld::distrib_quantile(d, p);
      INFO("p = " << p << ", mu = " << th.first << ", phi = " << th.second);
      CHECK(d.cdf(y) == doctest::Approx(p).epsilon(1e-9).scale(1.0));
    }
    for (double lq : {-40.0, -1e-6}) {
      const double y = ld::distrib_quantile_from_log_upper(d, lq);
      CHECK(d.log_ccdf(y) == doctest::Approx(lq).epsilon(1e-8).scale(1.0));
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian2_kernel_case in test_gaussian2.cpp.
void invgauss_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_invgauss", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.2 + 0.05 * i;
        const ld::invgauss_distrib g{0.4, 1.69};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y)[0] + g.log_cdf(y) +
                 g.log_ccdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.2 + 0.05 * i;
    const ld::invgauss_distrib g{0.4, 1.69};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y)[0] + g.log_cdf(y) +
                     g.log_ccdf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("distrib invgauss: every order inside a kernel") { invgauss_kernel_case(); }

TEST_CASE("distrib invgauss: log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_invgauss.py: mpmath.diff of the definition at 300 and 400 bits. The error of each
  // component in ulp of the largest component of its order at that point ("scaled") and in ulp of the component
  // itself ("own"), against hi + lo; regions: lower tail y/mu < 0.5, upper tail y/mu > 2, small phi mu < 0.01,
  // centre the rest.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_invgauss.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int upper, K, b; double y, hi, lo, mu, phi; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64], phis[64];
    if (std::sscanf(line, "%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.upper, ys, &r.K, &r.b, his, los, mus,
                    phis) != 8)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.phi = std::strtod(phis, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 1000);
  // the largest |component| of each (upper, y, K, mu, phi) group, the scale of the ulp
  std::map<std::tuple<int, double, int, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.mu, r.phi)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  // regions: 0 centre, 1 lower tail, 2 upper tail, 3 small phi mu
  const char* region_name[4] = {"centre", "lower tail", "upper tail", "small phi mu"};
  double worst_scaled[2][7][4] = {}, worst_own[2][7][4] = {};
  for (const auto& r : rows) {
    const ld::invgauss_distrib g{r.mu, r.phi};
    const auto v = cderiv(g, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.y, r.K, r.mu, r.phi)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    const double ratio = r.y / r.mu;
    int reg = 0;
    if (r.phi * r.mu < 0.01) reg = 3;
    else if (ratio < 0.5) reg = 1;
    else if (ratio > 2.0) reg = 2;
    worst_scaled[r.upper][r.K][reg] = std::fmax(worst_scaled[r.upper][r.K][reg], ulp_scaled);
    worst_own[r.upper][r.K][reg] = std::fmax(worst_own[r.upper][r.K][reg], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " mu " << r.mu << " phi " << r.phi
                  << " ulp " << ulp_scaled << " own " << ulp_own);
    // bounds: the measured maxima over orders 0 to 6 and all regions, times about 1.5, measured 2026-11-13 on the
    // final route (H identity, depth-200 continued fraction from a = 1.25, delta series at threshold 1, X from the
    // (a, d) chain): log F 580 (order 3, far tail), log(1 - F) 116 (order 6, small phi mu); bounds 870 and 175.
    const double bound_scaled[2] = {870.0, 175.0};
    CHECK(ulp_scaled <= bound_scaled[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 0; K <= 6; ++K)
        for (int reg = 0; reg < 4; ++reg)
          std::printf("invgauss upper %d K %d %-12s: max %.3g ulp of the order's scale, %.3g ulp of the component\n", u, K,
                      region_name[reg], worst_scaled[u][K][reg], worst_own[u][K][reg]);
}

TEST_CASE("distrib invgauss: log-density derivatives in y and in (mu, phi) against mpmath") {
  // tools/reference/distrib_invgauss.py (logpdf): mpmath.diff of the definition at 300 and 400 bits, every (J, K)
  // with J + K <= 6. Errors in ulp of the largest component of the same (J, K) at that point ("scaled").
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_invgauss_logpdf.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int J, K, b; double y, hi, lo, mu, phi; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64], phis[64];
    if (std::sscanf(line, "%d,%d,%d,%63[^,],%63[^,],%63[^,],%63[^,],%63s", &r.J, &r.K, &r.b, ys, his, los, mus, phis) != 8)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.phi = std::strtod(phis, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 4000);
  std::map<std::tuple<int, int, double, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.J, r.K, r.y, r.mu, r.phi)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[7][7] = {};
  for (const auto& r : rows) {
    const ld::invgauss_distrib g{r.mu, r.phi};
    const auto v = yderiv(g, r.y, r.J, r.K);
    const double sc = scale[std::make_tuple(r.J, r.K, r.y, r.mu, r.phi)];
    const double ulp_scaled = std::fabs((v[r.b] - r.hi) - r.lo) / (std::ldexp(1.0, std::ilogb(sc) - 52));
    worst[r.J][r.K] = std::fmax(worst[r.J][r.K], ulp_scaled);
    INFO("J " << r.J << " K " << r.K << " b " << r.b << " y " << r.y << " mu " << r.mu << " phi " << r.phi);
    CHECK(ulp_scaled <= 43.0);  // measured maximum 28.3 ulp (order 0, value), times about 1.5
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int J = 0; J <= 6; ++J)
      for (int K = 0; J + K <= 6; ++K)
        std::printf("invgauss logpdf J %d K %d: max %.4g ulp of the scale\n", J, K, worst[J][K]);
}
