// laplace2_distrib (include/laplace/distrib/laplace2.hpp): the Laplace law in location and rate.
// Every derivative order against differences away from the kink, the sign(0) = 0 convention at
// y = mu, the mixed (y, theta) derivatives, the expected information against the variance of the
// score, log F and log(1 - F) against differences and against mpmath
// (tools/reference/distrib_gumbel_laplace2.py), the twin laplace1_distrib{mu, 1/lambda} (a
// reparametrization: values and y-derivatives agree, the derivatives in lambda follow by the chain
// rule), the identities of the cdf family, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("laplace2: descriptors and kinks") {
  CHECK(ld::laplace2_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::laplace2_distrib::y_regularity == 0);
  CHECK(ld::laplace2_distrib::params[0].regularity == 0);
  CHECK(ld::laplace2_distrib::params[0].cdf_regularity == 1);
  CHECK(ld::laplace2_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::laplace2_distrib::params[1].cdf_regularity == ld::infinitely_smooth);
  const ld::laplace2_distrib d{0.5, 2.0};
  CHECK(d.param_kinks(0, 1.25).count == 1);
  CHECK(d.param_kinks(0, 1.25).point[0] == 1.25);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 1);
  CHECK(d.y_kinks().point[0] == 0.5);
  CHECK(d.scale() == 0.5);
}

TEST_CASE("laplace2: derivatives against differences, away from the kink") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::laplace2_distrib>({0.4, 0.8}, y, 1e-8);
  check_derivs<ld::laplace2_distrib>({-1.0, 3.5}, -2.0, 1e-8);
}

TEST_CASE("laplace2: sign(0) = 0 at the kink y = mu") {
  // the mean of the one-sided derivatives: the mu-derivative of l is lambda sign(r), 0 at r = 0;
  // the lambda-derivative 1/lambda - |r| is continuous there
  const ld::laplace2_distrib d{0.4, 0.8};
  CHECK(d.logpdf_d1(0.4)[0] == 0.0);
  CHECK(d.logpdf_d1(0.4)[1] == doctest::Approx(1.0 / 0.8).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<1, 0>(0.4)[0] == 0.0);
  CHECK(d.logpdf_dy_dk<1, 1>(0.4)[1] == 0.0);
  // one-sided values on each side of the kink
  CHECK(d.logpdf_d1(0.4 + 1e-9)[0] == doctest::Approx(0.8).epsilon(1e-15));
  CHECK(d.logpdf_d1(0.4 - 1e-9)[0] == doctest::Approx(-0.8).epsilon(1e-15));
}

TEST_CASE("laplace2: mixed (y, theta) derivatives against differences, away from the kink") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::laplace2_distrib>({0.4, 0.8}, y, 1e-2, 1e-8);
}

TEST_CASE("laplace2: expected information against the variance of the score and its derivatives") {
  for (double lambda : {0.3, 0.8, 4.0}) {
    const ld::laplace2_distrib d{-0.7, lambda};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("lambda = " << lambda);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-12 * e[2]);
    CHECK(e[0] == doctest::Approx(lambda * lambda).epsilon(1e-15));
    CHECK(e[2] == doctest::Approx(1.0 / (lambda * lambda)).epsilon(1e-15));
    CHECK(e[1] == 0.0);
  }
  check_info_derivs<ld::laplace2_distrib>({0.4, 0.8}, 1e-8);
}

TEST_CASE("laplace2: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {-6.0, -0.5, 0.3, 1.7, 9.0})
      check_cdf_derivs<ld::laplace2_distrib>({0.4, 0.8}, y, upper, 1e-7, std::fabs(0.8 * (y - 0.4)));
  }
}

TEST_CASE("laplace2: the free twin laplace1_distrib{mu, 1/lambda}") {
  // laplace2 at (mu, lambda) is laplace1 at (mu, sigma = 1/lambda): values and y-derivatives agree;
  // the parameter derivatives in lambda follow from those in sigma by d/dlambda = -sigma^2 d/dsigma
  // (and for the information I_lambda,lambda = sigma^4 I_sigma,sigma).
  for (double lambda : {0.35, 0.8, 3.0}) {
    const double mu = 0.4, sigma = 1.0 / lambda;
    const ld::laplace2_distrib d2{mu, lambda};
    const ld::laplace1_distrib d1{mu, sigma};
    for (double y : {-6.0, -0.9, -0.1, 0.6, 1.7, 8.0}) {
      CHECK(d2.logpdf(y) == doctest::Approx(d1.logpdf(y)).epsilon(1e-14));
      CHECK(d2.cdf(y) == doctest::Approx(d1.cdf(y)).epsilon(1e-14));
      CHECK(d2.ccdf(y) == doctest::Approx(d1.ccdf(y)).epsilon(1e-14));
      CHECK(d2.log_cdf(y) == doctest::Approx(d1.log_cdf(y)).epsilon(1e-14));
      CHECK(d2.log_ccdf(y) == doctest::Approx(d1.log_ccdf(y)).epsilon(1e-14));
      CHECK(d2.logpdf_dy_dk<1, 0>(y)[0] == doctest::Approx(d1.logpdf_dy_dk<1, 0>(y)[0]).epsilon(1e-14));
      CHECK(d2.logpdf_dy_dk<2, 0>(y)[0] == doctest::Approx(d1.logpdf_dy_dk<2, 0>(y)[0]).epsilon(1e-14));
      // first derivatives: mu unchanged, lambda = -sigma^2 sigma-derivative
      const auto g2 = d2.logpdf_d1(y), g1 = d1.logpdf_d1(y);
      CHECK(g2[0] == doctest::Approx(g1[0]).epsilon(1e-13));
      CHECK(g2[1] == doctest::Approx(-sigma * sigma * g1[1]).epsilon(1e-13));
      const auto c2 = d2.log_cdf_d1(y), c1 = d1.log_cdf_d1(y);
      CHECK(c2[0] == doctest::Approx(c1[0]).epsilon(1e-12));
      CHECK(c2[1] == doctest::Approx(-sigma * sigma * c1[1]).epsilon(1e-12));
      if (std::fabs(y) < 6.0) CHECK(d2.quantile(d2.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
    }
    const auto i2 = d2.expected_info(), i1 = d1.expected_info();
    CHECK(i2[0] == doctest::Approx(i1[0]).epsilon(1e-14));
    CHECK(i2[2] == doctest::Approx(sigma * sigma * sigma * sigma * i1[2]).epsilon(1e-14));
    for (double p : {1e-9, 0.2, 0.5, 0.8, 1.0 - 1e-9})
      CHECK(d2.quantile(p) == doctest::Approx(d1.quantile(p)).epsilon(1e-13));
  }
}

// the reference values of tools/reference/distrib_gumbel_laplace2.py, dist 5; the errors of each
// component in ulp of the largest component of its order at that y, and in ulp of the component
TEST_CASE("laplace2: derivatives of log F and log(1 - F) against mpmath") {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_gumbel_laplace2.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int upper, K, b; double y, hi, lo, mu, second; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    int dist = 0;
    char ys[64], his[64], los[64], mus[64], seconds[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &dist, &r.upper, ys, &r.K, &r.b, his, los,
                    mus, seconds) != 9)
      continue;
    if (dist != 5) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.second = std::strtod(seconds, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 500);
  double worst_scaled[2] = {}, worst_own[2] = {};
  for (const auto& r : rows) {
    const auto v = cderiv(ld::laplace2_distrib{r.mu, r.second}, r.y, r.K, r.upper);
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
      std::printf("laplace2 upper %d y %g K %d b %d: %.3g ulp scaled, %.3g ulp own, value %.3g\n", r.upper, r.y, r.K, r.b,
                  ulp_scaled, ulp_own, r.hi);
    // measured (see the numerics entry of 2026-11-13): bounds of about half again the measured maxima
    const double bound[2] = {8.5, 37.0};  // measured 2026-11-13: 5.64 (log F), 24.8 (log(1 - F)) ulp of the order's scale,
    // after the Laguerre identity for a negative order in detail_laplace2 (180 and 176 before it)
    CHECK(ulp_scaled <= bound[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      std::printf("laplace2 log_cdf derivs upper %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n", u,
                  worst_scaled[u], worst_own[u]);
}

TEST_CASE("laplace2: cdf, ccdf, logarithms, quantile") {
  const ld::laplace2_distrib d{0.4, 0.8};
  for (double y : {-30.0, -2.0, 0.4, 1.0, 5.0, 30.0}) {
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
    CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
    if (std::fabs(y) < 10.0) CHECK(d.quantile(d.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
  }
  CHECK(d.cdf(0.4) == doctest::Approx(0.5).epsilon(1e-15));
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void laplace2_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_laplace2", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::laplace2_distrib l{0.4, 0.8};
        out(i) = l.logpdf_d6(y + 0.031)[2] + l.logpdf_dy_dk<2, 3>(y + 0.031)[1] + l.expected_info_d4().pair[2][4] +
                 l.expected_info_d4().pair[0][1] + l.log_cdf_d6(y)[6] + l.log_ccdf_d6(y)[3] + l.log_ccdf(y) +
                 l.quantile(0.3);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::laplace2_distrib l{0.4, 0.8};
    const double r = l.logpdf_d6(y + 0.031)[2] + l.logpdf_dy_dk<2, 3>(y + 0.031)[1] + l.expected_info_d4().pair[2][4] +
                     l.expected_info_d4().pair[0][1] + l.log_cdf_d6(y)[6] + l.log_ccdf_d6(y)[3] + l.log_ccdf(y) +
                     l.quantile(0.3);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("laplace2: every order inside a kernel") { laplace2_kernel_case(); }
