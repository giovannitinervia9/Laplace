// gaussian2_distrib (mean mu, variance v; include/laplace/distrib/gaussian2.hpp): packing and
// descriptors, every derivative order against a 5-point difference of the order below, the
// mixed (y, theta) derivatives, the expected information against the variance of the score and
// its derivatives against differences, the derivatives of log F and log(1 - F) against
// differences and against mpmath (tools/reference/distrib_gaussian23.py), the cdf family, and a
// kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("distrib gaussian2: descriptors") {
  CHECK(ld::gaussian2_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::gaussian2_distrib::n_params == 2);
  CHECK(ld::gaussian2_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::gaussian2_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::gaussian2_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::gaussian2_distrib::params[j].moves_support == false);
  }
  const ld::gaussian2_distrib d{0.4, 1.69};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.scale() == doctest::Approx(1.3).epsilon(1e-15));
}

TEST_CASE("distrib gaussian2: derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::gaussian2_distrib>({0.4, 1.69}, y, 1e-8);
  check_derivs<ld::gaussian2_distrib>({-20.0, 0.49}, -18.5, 1e-8);  // large mean difference
  for (double y : {-1.0, 0.4, 0.9}) check_derivs<ld::gaussian2_distrib>({0.4, 0.01}, y, 1e-8);   // small v
  for (double y : {-30.0, 0.4, 45.0}) check_derivs<ld::gaussian2_distrib>({0.4, 400.0}, y, 1e-8);  // large v
}

TEST_CASE("distrib gaussian2: derivatives in y and mixed (y, theta) against differences") {
  const ld::gaussian2_distrib d{0.4, 1.69};
  CHECK(d.logpdf_dy_dk<0, 0>(1.3)[0] == doctest::Approx(d.logpdf(1.3)).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<0, 3>(1.3)[2] == d.logpdf_d3(1.3)[2]);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::gaussian2_distrib>({0.4, 1.69}, y, 1e-3, 1e-8);
  for (double y : {-1.0, 0.9}) check_y_derivs<ld::gaussian2_distrib>({0.4, 0.04}, y, 1e-3, 1e-8);
}

TEST_CASE("distrib gaussian2: expected information against the variance of the score") {
  for (double v : {0.04, 1.69, 25.0}) {
    const ld::gaussian2_distrib d{0.4, v};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("gaussian2, v = " << v);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-11 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-12 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-11 * e[2]);
  }
}

TEST_CASE("distrib gaussian2: derivatives of the expected information against differences") {
  check_info_derivs<ld::gaussian2_distrib>({0.4, 1.69}, 1e-8);
  check_info_derivs<ld::gaussian2_distrib>({0.4, 0.04}, 1e-8);
  check_info_derivs<ld::gaussian2_distrib>({-7.0, 25.0}, 1e-8);
}

TEST_CASE("distrib gaussian2: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    const double s = std::sqrt(1.69);
    for (double y : {-12.0, -3.0, 0.2, 1.7, 9.0, 20.0})
      check_cdf_derivs<ld::gaussian2_distrib>({0.4, 1.69}, y, upper, 1e-7, std::fabs(y - 0.4) / s);
    const double s2 = std::sqrt(0.04);
    for (double z : {-6.0, -4.5, -2.0, 0.0, 3.0, 5.0})
      check_cdf_derivs<ld::gaussian2_distrib>({0.4, 0.04}, 0.4 + s2 * z, upper, 1e-7, std::fabs(z));
  }
}

TEST_CASE("distrib gaussian2: log F of v against the first derivative in sigma of gaussian1") {
  // d/dv = (1/(2 sigma)) d/dsigma, with sigma = sqrt(v); the derivative in mu is the same
  for (double y : {-6.0, -1.0, 0.3, 2.0, 7.0}) {
    const ld::gaussian2_distrib g2{0.4, 1.69};
    const ld::gaussian1_distrib g1{0.4, 1.3};
    for (bool upper : {false, true}) {
      const auto a = upper ? g2.log_ccdf_d1(y) : g2.log_cdf_d1(y);
      const auto b = upper ? g1.log_ccdf_d1(y) : g1.log_cdf_d1(y);
      INFO("y = " << y << ", upper " << upper);
      CHECK(a[0] == doctest::Approx(b[0]).epsilon(1e-12).scale(1.0));
      CHECK(a[1] == doctest::Approx(b[1] / (2.0 * 1.3)).epsilon(1e-12).scale(1.0));
    }
  }
}

TEST_CASE("distrib gaussian2: cdf, ccdf, logarithms, quantile") {
  const ld::gaussian2_distrib g{0.4, 1.69};
  const double sigma = std::sqrt(1.69);
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
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void gaussian2_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_gaussian2", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::gaussian2_distrib g{0.4, 1.69};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y - 2.0)[0] + g.log_cdf(y) +
                 g.log_ccdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::gaussian2_distrib g{0.4, 1.69};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y - 2.0)[0] + g.log_cdf(y) +
                     g.log_ccdf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("distrib gaussian2: every order inside a kernel") { gaussian2_kernel_case(); }

TEST_CASE("distrib gaussian2: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_gaussian23.py (mean 0.4, v = 1.69 and 0.09), 2026-11-13. The error of each
  // component in ulp of the largest component of its order at that y ("scaled") and in ulp of the
  // component itself ("own"); the errors are the difference of the library value and the mpmath value
  // minus the mpmath value's low part.
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
    if (r.dist != 0) continue;  // gaussian3 rows are in test_gaussian3.cpp
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.w = std::strtod(ws, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 500);
  // the largest |component| of each (upper, y, K, w) group, the scale of the ulp
  std::map<std::tuple<int, double, int, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.w)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {}, worst_y[2][7] = {}, worst_w[2][7] = {};
  for (const auto& r : rows) {
    const ld::gaussian2_distrib g{r.mu, r.w};
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
    // measured 2026-11-13 (gcc, Serial + OpenMP build): scaled maxima 483 (log F) and 441 (log(1 - F)), at order 6;
    // bounds half again
    const double bound_scaled[2] = {730.0, 670.0};
    CHECK(ulp_scaled <= bound_scaled[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("gaussian2 upper %d K %d: max %.3g ulp of the order's scale (y %.6g, w %.6g), %.3g ulp of the component\n", u, K,
                    worst_scaled[u][K], worst_y[u][K], worst_w[u][K], worst_own[u][K]);
}

TEST_CASE("distrib gaussian2: quantile inverts the cdf in both tails") {
  const ld::gaussian2_distrib g{0.4, 0.04};
  for (double z : {-6.0, -4.0, 2.0, 5.0}) {
    const double y = 0.4 + 0.2 * z;
    CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-10));
  }
}
