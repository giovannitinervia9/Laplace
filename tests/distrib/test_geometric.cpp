// geometric_distrib (mean mu; include/laplace/distrib/geometric.hpp): packing and descriptors, the mass
// against the closed form and its normalization, every derivative order against a 5-point difference of
// the order below, the expected information against the variance of the score (a sum over the support)
// and its derivatives against differences, the derivatives of log F and log(1 - F) against differences
// and against the generic summation route, the cdf identities, the closed quantile against the generic
// search, a kernel launch of the highest orders, and every order against mpmath
// (tools/reference/distrib_geometric.py: mu from 1e-6 to 1e6).
#include <doctest/doctest.h>

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/geometric.hpp>
#include <Kokkos_Core.hpp>

#include "check_common.hpp"
#include "generic_hidden.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;

TEST_CASE("distrib geometric: descriptors") {
  CHECK(ld::geometric_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::geometric_distrib::n_params == 1);
  CHECK(ld::geometric_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::geometric_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::geometric_distrib::params[0].moves_support == false);
  const ld::geometric_distrib d{3.0};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 3.0);
  CHECK(d.scale() == doctest::Approx(std::sqrt(12.0)).epsilon(1e-15));
}

TEST_CASE("distrib geometric: mass against the closed form and its normalization") {
  for (double mu : {0.05, 1.0, 3.5, 40.0}) {
    const ld::geometric_distrib d{mu};
    double total = 0.0;
    for (int y = 0; y < 20000; ++y) {
      const double ref = std::log(1.0 / (1.0 + mu)) + y * std::log(mu / (1.0 + mu));
      if (y < 30) CHECK(d.logpmf(y) == doctest::Approx(ref).epsilon(1e-13).scale(1.0));
      total += std::exp(d.logpmf(y));
    }
    CHECK(total == doctest::Approx(1.0).epsilon(1e-12));
  }
}

TEST_CASE("distrib geometric: derivatives against differences") {
  for (double y : {0.0, 1.0, 4.0, 30.0}) check_derivs<ld::geometric_distrib>({3.5}, y, 1e-8);
  for (double y : {0.0, 1.0, 2.0}) check_derivs<ld::geometric_distrib>({1e-3}, y, 1e-8);
  for (double y : {0.0, 5.0, 400.0}) check_derivs<ld::geometric_distrib>({40.0}, y, 1e-8);
  for (double y : {10.0, 1500.0, 3.0e5}) check_derivs<ld::geometric_distrib>({1e3}, y, 1e-8);
}

TEST_CASE("distrib geometric: expected information against the variance of the score") {
  // E[l'^2] = sum_y p(y) l'(y)^2 over the support (the tail beyond 40 mu is below 1e-17)
  for (double mu : {0.05, 1.0, 3.5, 40.0}) {
    const ld::geometric_distrib d{mu};
    double s = 0.0;
    const int n = int(60.0 * (mu + 1.0)) + 100;
    for (int y = 0; y < n; ++y) {
      const double l1 = d.logpmf_d1(y)[0];
      s += std::exp(d.logpmf(y)) * l1 * l1;
    }
    const auto e = d.expected_info();
    INFO("geometric, mu = " << mu);
    CHECK(std::fabs(s - e[0]) <= 1e-12 * e[0]);
  }
}

TEST_CASE("distrib geometric: derivatives of the expected information against differences") {
  check_info_derivs<ld::geometric_distrib>({3.0}, 1e-8);
  check_info_derivs<ld::geometric_distrib>({0.05}, 1e-8);
  check_info_derivs<ld::geometric_distrib>({500.0}, 1e-8);
}

TEST_CASE("distrib geometric: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {0.0, 1.0, 5.0, 40.0}) check_cdf_derivs<ld::geometric_distrib>({3.0}, y, upper, 1e-7);
    check_cdf_derivs<ld::geometric_distrib>({0.01}, 0.0, upper, 1e-7);
    check_cdf_derivs<ld::geometric_distrib>({0.01}, 3.0, upper, 1e-7);
    check_cdf_derivs<ld::geometric_distrib>({1e3}, 0.0, upper, 1e-7);
    check_cdf_derivs<ld::geometric_distrib>({1e3}, 2000.0, upper, 1e-7);
  }
}

namespace {
// worst |closed - sum| over the six orders, relative to the largest magnitude of the order
double worst_vs_sum(const ld::geometric_distrib& d, double y, bool upper) {
  const auto h = hide(d);
  double worst = 0.0;
  auto one = [&](auto kc) {
    constexpr int K = decltype(kc)::value;
    const double a = upper ? d.template log_ccdf_dk<K>(y)[0] : d.template log_cdf_dk<K>(y)[0];
    const double b = upper ? ld::log_ccdf_dk_by_sum<K>(h, y)[0] : ld::log_cdf_dk_by_sum<K>(h, y)[0];
    worst = std::fmax(worst, std::fabs(a - b) / std::fmax(std::fabs(a), 1e-300));
  };
  one(std::integral_constant<int, 1>{}), one(std::integral_constant<int, 2>{}), one(std::integral_constant<int, 3>{});
  one(std::integral_constant<int, 4>{}), one(std::integral_constant<int, 5>{}), one(std::integral_constant<int, 6>{});
  return worst;
}
}  // namespace

TEST_CASE("distrib geometric: closed derivatives of log F against the generic summation") {
  double worst = 0.0;
  for (double mu : {0.5, 3.5, 40.0})
    for (double y : {0.0, 1.0, 5.0, std::floor(mu), 3.0 * mu})
      for (bool upper : {false, true}) {
        const double w = worst_vs_sum(ld::geometric_distrib{mu}, y, upper);
        INFO("mu = " << mu << ", y = " << y << ", upper " << upper);
        CHECK(w <= 1e-6);
        worst = std::fmax(worst, w);
      }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("geometric closed vs sum: worst relative %.3g\n", worst);
}

TEST_CASE("distrib geometric: cdf identities") {
  for (double mu : {1e-3, 0.5, 3.0, 800.0}) {
    const ld::geometric_distrib c{mu};
    for (double y : {0.0, 1.0, 2.0, 10.0, 100.0}) {
      CHECK(c.cdf(y) + c.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
      CHECK(c.log_ccdf(y) == doctest::Approx(std::log(c.ccdf(y))).epsilon(1e-12));
      if (c.cdf(y) > 1e-300) CHECK(c.log_cdf(y) == doctest::Approx(std::log(c.cdf(y))).epsilon(1e-12));
      CHECK(c.cdf(y + 0.5) == c.cdf(y));
    }
    CHECK(c.cdf(-1.0) == 0.0);
    CHECK(c.ccdf(-1.0) == 1.0);
    CHECK(c.log_cdf(-1.0) == -ld::distrib_infinity);
    CHECK(c.log_ccdf(-1.0) == 0.0);
    CHECK(std::isnan(c.log_cdf_d3(-1.0)[0]));
    CHECK(c.log_ccdf_d3(-1.0)[0] == 0.0);
  }
  // mu = 1: F(k) = 1 - 2^-(k + 1)
  const ld::geometric_distrib one{1.0};
  CHECK(one.cdf(3.0) == doctest::Approx(1.0 - 1.0 / 16.0).epsilon(1e-15));
  // the sum of the masses is F
  const ld::geometric_distrib d{2.5};
  double s = 0.0;
  for (int k = 0; k < 12; ++k) {
    s += std::exp(d.logpmf(k));
    CHECK(d.cdf(k) == doctest::Approx(s).epsilon(1e-14));
  }
}

TEST_CASE("distrib geometric: closed quantile equals the generic search") {
  int tested = 0;
  for (double mu : {1e-6, 1e-3, 0.5, 1.0, 3.5, 40.0, 1e3, 1e6}) {
    const ld::geometric_distrib d{mu};
    std::vector<double> ps{1e-300, 1e-100, 1e-16, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-6, 1.0 - 1e-12,
                           1.0 - 0x1p-53};
    for (double k : {0.0, 1.0, 2.0, 5.0, 17.0, 400.0}) {
      const double p = d.cdf(k);
      if (p > 0.0 && p < 1.0) {
        ps.push_back(p);
        ps.push_back(std::nextafter(p, 0.0));
        ps.push_back(std::nextafter(p, 1.0));
      }
    }
    for (double p : ps) {
      const double k = ld::quantile_by_search(d, p);
      INFO("mu = " << mu << ", p = " << p);
      CHECK(d.quantile(p) == k);
      CHECK(d.quantile_from_log(std::log(p)) == ld::quantile_by_search_from_log(d, std::log(p)));
      if (p < 1.0 - 1e-15) {
        const double lq = std::log1p(-p);
        CHECK(d.quantile_from_log_upper(lq) == ld::quantile_by_search_from_log_upper(d, lq));
      }
      ++tested;
    }
    CHECK(d.quantile(0.0) == 0.0);
    CHECK(d.quantile(1.0) == ld::distrib_infinity);
    CHECK(std::isnan(d.quantile(-0.5)));
    CHECK(std::isnan(d.quantile(1.5)));
    // quantile(F(k)) is k, or k + 1 when the rounding of log F(k) against log p goes the wrong way: the
    // predicates are the generic search's own, which has the same property
    for (double k : {0.0, 1.0, 2.0, 5.0}) {
      const double p = d.cdf(k);
      if (p > 1e-12 && p < 1.0 - 1e-12 && (k == 0.0 || d.cdf(k - 1.0) < p * (1.0 - 1e-6))) {
        CHECK(d.quantile(p) >= k);
        CHECK(d.quantile(p) <= k + 1.0);
      }
    }
  }
  CHECK(tested > 150);
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
double geometric_kernel_value(double y) {
  const ld::geometric_distrib c{3.7};
  return c.logpmf_d6(y)[0] + c.expected_info_d4().pair[0][0] + c.log_cdf_d6(y)[0] + c.log_ccdf_d6(y)[0] +
         c.log_cdf(y) + c.log_ccdf(y) + c.quantile(0.3) + c.quantile_from_log(-0.2) + c.quantile_from_log_upper(-2.0);
}

void geometric_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_geometric", n, KOKKOS_LAMBDA(const int i) {
        const double y = double(i % 20);
        const ld::geometric_distrib c{3.7};
        out(i) = c.logpmf_d6(y)[0] + c.expected_info_d4().pair[0][0] + c.log_cdf_d6(y)[0] + c.log_ccdf_d6(y)[0] +
                 c.log_cdf(y) + c.log_ccdf(y) + c.quantile(0.3) + c.quantile_from_log(-0.2) +
                 c.quantile_from_log_upper(-2.0);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double r = geometric_kernel_value(double(i % 20));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-12));
  }
}

TEST_CASE("distrib geometric: every order inside a kernel") { geometric_kernel_case(); }

TEST_CASE("distrib geometric: every order against mpmath") {
  // tools/reference/distrib_geometric.py: kind 0 log p, 1 expected information, 2 log F, 3 log(1 - F); orders K
  // in mu. The error is the library value minus the mpmath value (low part removed), in ulp of:
  //  - kind 0, K >= 1: the two terms of the derivative, (K-1)! [y / mu^K + (y + 1) / (1 + mu)^K], because the
  //    derivative has a genuine zero between them and the error there is measured against what was subtracted;
  //  - kind 2, K >= 1: the largest value of that order at the same mu (the Bell sum is alternating and each
  //    derivative of log F has zeros in y, where an error is measured against the order's scale, as for chisq);
  //  - every other row: its own value (nothing cancels in kinds 1 and 3).
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_geometric.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int kind, K; double y, hi, lo, mu; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64];
    if (std::sscanf(line, "%d,%63[^,],%d,%63[^,],%63[^,],%63s", &r.kind, ys, &r.K, his, los, mus) != 6) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 2500);
  std::map<std::tuple<int, int, double>, double> scale_of;  // largest |value| of an order of log F at one mu
  for (const auto& r : rows) {
    auto& s = scale_of[std::make_tuple(r.kind, r.K, r.mu)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[4][7] = {}, worst_y[4][7] = {}, worst_mu[4][7] = {};
  // bounds: 1.5 times the worst measured on the gcc (Serial + OpenMP) and the CUDA build (2026-11-15,
  // LAPLACE_PRINT_ERRORS); kind 2 K = 0 is conditioned by |u| = (k + 1) t, up to 350 t = 100 here
  const double bound[4][7] = {{2.0, 2.7, 5.9, 6.2, 8.8, 10.7, 15.3},
                             {2.6, 4.9, 5.9, 5.9, 6.6, 0.0, 0.0},
                             {124.0, 3.9, 4.9, 12.5, 22.0, 65.0, 91.0},
                             {1.7, 2.6, 4.9, 5.9, 8.8, 10.7, 8.6}};
  for (const auto& r : rows) {
    const ld::geometric_distrib d{r.mu};
    double v;
    if (r.kind == 0) v = r.K == 0 ? d.logpmf(r.y) : lderiv(d, r.y, r.K)[0];
    else if (r.kind == 1) v = ideriv(d, r.K)[0];
    else v = cderiv(d, r.y, r.K, r.kind == 3)[0];
    double sc = std::fabs(r.hi);
    if (r.kind == 2 && r.K >= 1) sc = scale_of[std::make_tuple(r.kind, r.K, r.mu)];
    if (r.kind == 0 && r.K >= 1) {
      double fact = 1.0;
      for (int k = 2; k < r.K; ++k) fact *= k;
      sc = fact * (r.y / std::pow(r.mu, r.K) + (r.y + 1.0) / std::pow(1.0 + r.mu, r.K));
    }
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    if (ulp > worst[r.kind][r.K]) worst_y[r.kind][r.K] = r.y, worst_mu[r.kind][r.K] = r.mu;
    worst[r.kind][r.K] = std::fmax(worst[r.kind][r.K], ulp);
    INFO("kind " << r.kind << " y " << r.y << " K " << r.K << " mu " << r.mu << " ulp " << ulp);
    CHECK(ulp <= bound[r.kind][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int k = 0; k < 4; ++k)
      for (int K = 0; K <= (k == 1 ? 4 : 6); ++K)
        std::printf("geometric kind %d K %d: max %.3g ulp (y %.6g, mu %.6g)\n", k, K, worst[k][K], worst_y[k][K],
                    worst_mu[k][K]);
}
