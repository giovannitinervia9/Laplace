// The tests of logistic_distrib (include/laplace/distrib/logistic.hpp): descriptors, every
// derivative order against differences (including |z| = 40), the mixed (y, theta) derivatives,
// the expected information against the variance of the score (quadrature) and the closed form
// (pi^2 + 3)/(9 sigma^2) of its scale entry, the cdf family against identities, the derivatives of
// log F and log(1 - F) against mpmath (tools/reference/distrib_cauchy_logistic.py, |z| up to 700),
// and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/logistic.hpp>
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

TEST_CASE("logistic: descriptors and kinks") {
  CHECK(ld::logistic_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::logistic_distrib::n_params == 2);
  CHECK(ld::logistic_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::logistic_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::logistic_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::logistic_distrib::params[1].cdf_regularity == ld::infinitely_smooth);
  const ld::logistic_distrib d{0.4, 1.3};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  // the density at the centre is 1/(4 sigma)
  CHECK(d.logpdf(0.4) == doctest::Approx(-std::log(4.0 * 1.3)).epsilon(1e-15));
}

TEST_CASE("logistic: derivatives against differences, centre and tails") {
  for (double s : {0.25, 1.3, 4.0})
    for (double y : {-40.0, -3.0, 0.2, 1.7, 9.0, 40.0}) check_derivs<ld::logistic_distrib>({0.4, s}, y, 1e-8);
  check_derivs<ld::logistic_distrib>({-20.0, 0.7}, -18.5, 1e-8);
}

TEST_CASE("logistic: mixed (y, theta) derivatives against differences") {
  for (double s : {1.3, 4.0})
    for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::logistic_distrib>({0.4, s}, y, 1e-3 * s, 1e-8);
  const ld::logistic_distrib d{0.4, 1.3};
  const auto a = d.logpdf_dy_dk<1, 0>(2.0);
  const auto b = d.logpdf_dk<1>(2.0);
  CHECK(a[0] == doctest::Approx(-b[0]).epsilon(1e-14));
}

TEST_CASE("logistic: expected information against the variance of the score") {
  for (const auto& th : {std::vector<double>{0.4, 1.3}, std::vector<double>{-2.0, 0.25}}) {
    const ld::logistic_distrib d{th[0], th[1]};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("sigma " << th[1]);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-10 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-10 * e[0]);
    CHECK(std::fabs(q[2] - e[2]) <= 1e-10 * e[2]);
  }
  // the closed form of the scale entry: (pi^2 + 3)/(9 sigma^2), and 1/(3 sigma^2) for the location
  const ld::logistic_distrib d{0.4, 1.3};
  const double pi = 3.14159265358979323846;
  CHECK(d.expected_info()[2] == doctest::Approx((pi * pi + 3.0) / (9.0 * 1.3 * 1.3)).epsilon(1e-15));
  CHECK(d.expected_info()[0] == doctest::Approx(1.0 / (3.0 * 1.3 * 1.3)).epsilon(1e-15));
}

TEST_CASE("logistic: derivatives of the expected information against differences") {
  check_info_derivs<ld::logistic_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::logistic_distrib>({-2.0, 0.25}, 1e-8);
}

TEST_CASE("logistic: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true})
    for (double y : {-40.0, -3.0, 0.2, 1.7, 9.0, 40.0})
      check_cdf_derivs<ld::logistic_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
}

TEST_CASE("logistic: cdf, ccdf, logarithms, quantile") {
  const ld::logistic_distrib d{0.4, 1.3};
  for (double y : {-700.0, -40.0, -3.0, 0.4, 1.7, 9.0, 40.0, 700.0}) {
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    if (d.cdf(y) < 0.5) CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
    if (d.ccdf(y) < 0.5) CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
  }
  // the tails at z = +-700 are finite and equal to their asymptotes -|z| (log F for z < 0 and log(1 - F) for z > 0)
  CHECK(d.log_cdf(0.4 - 1.3 * 700.0) == doctest::Approx(-700.0).epsilon(1e-12));
  CHECK(d.log_ccdf(0.4 + 1.3 * 700.0) == doctest::Approx(-700.0).epsilon(1e-12));
  for (double p : {1e-12, 1e-6, 0.3, 0.5, 0.7, 1.0 - 1e-6}) {
    const double q = d.quantile(p);
    if (p <= 0.5) CHECK(d.cdf(q) == doctest::Approx(p).epsilon(1e-11));
    else CHECK(d.ccdf(q) == doctest::Approx(1.0 - p).epsilon(1e-11));
  }
}

TEST_CASE("logistic: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_cauchy_logistic.py: mpmath.diff of the definition at 300 and 400 bits.
  // Each component is compared in ulp of the largest component of its order at that point and in
  // ulp of the component itself (the components change sign, so a plain relative error is
  // undefined at their zeros); the error is |(library - hi) - lo|.
  const auto rows = read_reference(1);
  CHECK(rows.size() > 2000);
  std::map<std::tuple<int, double, int, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.mu, r.sigma)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double w_scaled[2][7] = {}, w_own[2][7] = {};
  for (const auto& r : rows) {
    const ld::logistic_distrib d{r.mu, r.sigma};
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
    // log F 108 (order 5), log(1 - F) 44 (order 5); bounds half again
    const double bound[2] = {162.0, 67.0};
    CHECK(ulp_scaled <= bound[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("logistic log_%s derivs, order %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n",
                    u ? "(1-F)" : "F", K, w_scaled[u][K], w_own[u][K]);
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void logistic_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "logistic_distrib", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::logistic_distrib g{0.4, 1.3};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.log_cdf_d6(y)[6] + g.log_ccdf_d6(y + 1.0)[2] + g.logpdf(y) + g.quantile(0.25 + 0.005 * i);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::logistic_distrib g{0.4, 1.3};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.log_cdf_d6(y)[6] + g.log_ccdf_d6(y + 1.0)[2] + g.logpdf(y) + g.quantile(0.25 + 0.005 * i);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("logistic: every order inside a kernel") { logistic_kernel_case(); }
