// gumbel_distrib (include/laplace/distrib/gumbel.hpp): every derivative order against
// differences, the mixed (y, theta) derivatives, the expected information against the variance
// of the score and against its closed form (gamma and pi^2/6), the derivatives of log F and
// log(1 - F) against differences and against mpmath (tools/reference/distrib_gumbel_laplace2.py),
// the identities of the cdf family, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/gumbel.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {
constexpr double kEulerGamma = 0.57721566490153286061;
constexpr double kPi2Over6 = 1.6449340668482264365;
}  // namespace

TEST_CASE("gumbel: descriptors and kinks") {
  CHECK(ld::gumbel_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::gumbel_distrib::n_params == 2);
  CHECK(ld::gumbel_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::gumbel_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::gumbel_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::gumbel_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::gumbel_distrib::params[1].cdf_regularity == ld::infinitely_smooth);
  const ld::gumbel_distrib d{0.4, 1.3};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 1.3);
}

TEST_CASE("gumbel: derivatives against differences") {
  for (double y : {-8.0, -3.0, 0.2, 1.7, 9.0, 25.0}) check_derivs<ld::gumbel_distrib>({0.4, 1.3}, y, 1e-8);
  check_derivs<ld::gumbel_distrib>({-2.0, 0.6}, -1.0, 1e-8);
}

TEST_CASE("gumbel: mixed (y, theta) derivatives against differences") {
  for (double y : {-8.0, -3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::gumbel_distrib>({0.4, 1.3}, y, 1e-3, 1e-8);
  // the log-density depends on y and mu through y - mu only: d/dy = -d/dmu, exactly
  const ld::gumbel_distrib d{0.4, 1.3};
  for (double y : {-3.0, 2.2}) {
    const auto dy = d.logpdf_dy_dk<1, 0>(y)[0];
    const auto dmu = d.logpdf_d1(y)[0];
    CHECK(std::fabs(dy + dmu) <= 1e-14 * std::fmax(1.0, std::fabs(dmu)));
  }
}

TEST_CASE("gumbel: expected information against the variance of the score and its closed form") {
  for (double sigma : {0.4, 1.3, 6.0}) {
    const ld::gumbel_distrib d{-0.7, sigma};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("sigma = " << sigma);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-10 * e[2]);
    // the closed form, I = sigma^-2 [[1, gamma - 1], [gamma - 1, (1 - gamma)^2 + pi^2/6]]
    const double s2 = 1.0 / (sigma * sigma);
    CHECK(e[0] == doctest::Approx(s2).epsilon(1e-15));
    CHECK(e[1] == doctest::Approx((kEulerGamma - 1.0) * s2).epsilon(1e-15));
    CHECK(e[2] == doctest::Approx(((1.0 - kEulerGamma) * (1.0 - kEulerGamma) + kPi2Over6) * s2).epsilon(1e-15));
  }
  check_info_derivs<ld::gumbel_distrib>({0.4, 1.3}, 1e-8);
}

TEST_CASE("gumbel: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {-8.0, -3.0, 0.2, 1.7, 9.0, 20.0})
      check_cdf_derivs<ld::gumbel_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
  }
}

TEST_CASE("gumbel: cdf, ccdf, logarithms, quantile") {
  const ld::gumbel_distrib d{0.4, 1.3};
  for (double y : {-6.0, -2.0, 0.4, 1.0, 5.0, 12.0}) {
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
    CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
    CHECK(d.logpdf(y) == doctest::Approx(std::log(std::exp(d.logpdf(y)))).epsilon(1e-15));
    if (std::fabs(y) < 10.0) {
      CHECK(d.quantile(d.cdf(y)) == doctest::Approx(y).epsilon(1e-11));
    }
  }
  // the median and the mode: F(mu - sigma log(log 2)) = 1/2, and the density is maximal at mu
  CHECK(d.cdf(0.4 - 1.3 * std::log(std::log(2.0))) == doctest::Approx(0.5).epsilon(1e-15));
  CHECK(d.logpdf_d1(0.4)[0] == 0.0);
  CHECK(d.quantile(0.5) == doctest::Approx(0.4 - 1.3 * std::log(std::log(2.0))).epsilon(1e-15));
  // the quantile near 0 and 1 stays accurate: the inverse of the cdf at p = 1 - 1e-12 is y = mu - sigma log(-log p)
  const double p1 = 1.0 - 1e-12;
  CHECK(d.quantile(p1) == doctest::Approx(0.4 - 1.3 * std::log(-std::log1p(p1 - 1.0))).epsilon(1e-14));
  CHECK(d.quantile(1e-12) == doctest::Approx(0.4 - 1.3 * std::log(-std::log(1e-12))).epsilon(1e-14));
  // mu is the mode: the density at the mode exceeds the density at mu +- 0.3
  CHECK(d.logpdf(0.4) > d.logpdf(0.4 + 0.3));
  CHECK(d.logpdf(0.4) > d.logpdf(0.4 - 0.3));
}

// the reference values of tools/reference/distrib_gumbel_laplace2.py; the errors of each component
// in ulp of the largest component of its order at that y, and in ulp of the component itself
TEST_CASE("gumbel: derivatives of log F and log(1 - F) against mpmath") {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_gumbel_laplace2.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo, mu, second; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64], seconds[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b, his, los,
                    mus, seconds) != 9)
      continue;
    if (r.dist != 4) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.second = std::strtod(seconds, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 500);
  double worst_scaled[2] = {}, worst_own[2] = {};
  for (const auto& r : rows) {
    const auto v = cderiv(ld::gumbel_distrib{r.mu, r.second}, r.y, r.K, r.upper);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.upper == r.upper && q.y == r.y && q.K == r.K) scale = std::fmax(scale, std::fabs(q.hi));
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(scale) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.upper] = std::fmax(worst_scaled[r.upper], ulp_scaled);
    worst_own[r.upper] = std::fmax(worst_own[r.upper], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " ulp " << ulp_scaled);
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_scaled > 4.0)
      std::printf("gumbel upper %d y %g K %d b %d: %.3g ulp scaled, %.3g ulp own, value %.3g\n", r.upper, r.y, r.K, r.b,
                  ulp_scaled, ulp_own, r.hi);
    // measured (see the numerics entry of 2026-11-13): bounds of about half again the measured maxima
    const double bound[2] = {30.0, 600.0};  // measured 2026-11-13: 19.1 (log F), 394 (log(1 - F)) ulp of the order's scale
    CHECK(ulp_scaled <= bound[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      std::printf("gumbel log_cdf derivs upper %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n", u,
                  worst_scaled[u], worst_own[u]);
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void gumbel_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_gumbel", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::gumbel_distrib g{0.4, 1.3};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.log_cdf_d6(y)[6] + g.log_ccdf_d6(y)[2] + g.log_ccdf(y) + g.quantile(0.3) + g.expected_info_d4().pair[1][4];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::gumbel_distrib g{0.4, 1.3};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.log_cdf_d6(y)[6] + g.log_ccdf_d6(y)[2] + g.log_ccdf(y) + g.quantile(0.3) + g.expected_info_d4().pair[1][4];
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("gumbel: every order inside a kernel") { gumbel_kernel_case(); }
