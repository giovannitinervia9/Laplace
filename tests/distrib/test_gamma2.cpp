// gamma2_distrib (mean mu, variance sigma2; include/laplace/distrib/gamma2.hpp): descriptors, every
// derivative order against a 5-point difference of the order below (in the parameters, in y and mixed),
// the expected information against the variance of the score and the generic quadrature route, its
// derivatives against differences, the derivatives of log F and log(1 - F) against differences and
// against mpmath (tools/reference/distrib_gamma2.py, the definitions), the cdf family and the quantile
// in both tails, the relation to gamma1 at the same law, the SymPy direct forms of orders 1 and 2, and a
// kernel launch of the highest orders.
//
// The points: shape s = mu^2/sigma2 = 1/phi with phi = sigma2/mu^2. The 5-point differences of
// check_common step by 1e-3 max(1e-3, |theta|), so a parameter near 1e-9 would be stepped over by an absolute
// 1e-6; the extreme shapes are therefore taken at mu = 1e4 (shape 1e8 with sigma2 = 1) and sigma2 = 1600
// (shape 1e-4 with mu = 0.4), where every step is relative.
#include <doctest/doctest.h>

#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;

// a point (mu, sigma2) of the tests
struct point {
  double mu, sigma2;
};

// the generic route returns the closed-form container for K = 0 and the derivative container for K >= 1
template <int K, class T>
std::vector<double> generic_info_vec(const T& t) {
  if constexpr (K == 0) return as_vec<2, 2>(t);
  else return info_vec<2, K>(t);
}

TEST_CASE("distrib gamma2: descriptors") {
  CHECK(ld::gamma2_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::gamma2_distrib::n_params == 2);
  CHECK(ld::gamma2_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::gamma2_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::gamma2_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::gamma2_distrib::params[j].moves_support == false);
  }
  const ld::gamma2_distrib d{0.4, 0.16};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == doctest::Approx(0.4).epsilon(1e-15));  // sqrt(0.16)
}

TEST_CASE("distrib gamma2: derivatives against differences") {
  for (double y : {0.02, 0.3, 0.4, 0.55, 1.2, 3.0}) check_derivs<ld::gamma2_distrib>({0.4, 0.16}, y, 1e-8);
  for (double y : {0.05, 1.5, 6.0}) check_derivs<ld::gamma2_distrib>({2.5, 0.125}, y, 1e-8);
  for (double y : {0.05, 0.4, 2.0}) check_derivs<ld::gamma2_distrib>({0.4, 1600.0}, y, 1e-8);
}

TEST_CASE("distrib gamma2: derivatives in y and mixed (y, theta) against differences") {
  const ld::gamma2_distrib d{0.4, 0.16};
  CHECK(d.logpdf_dy_dk<0, 0>(0.7)[0] == doctest::Approx(d.logpdf(0.7)).epsilon(1e-15));
  CHECK(d.logpdf_dy_dk<0, 3>(0.7)[2] == d.logpdf_d3(0.7)[2]);
  for (double y : {0.05, 0.4, 0.9, 2.0}) check_y_derivs<ld::gamma2_distrib>({0.4, 0.16}, y, 1e-3 * y, 1e-8);
  for (double y : {1.0, 2.5, 9.0}) check_y_derivs<ld::gamma2_distrib>({2.5, 0.125}, y, 1e-3 * y, 1e-8);
}

TEST_CASE("distrib gamma2: expected information against the variance of the score") {
  // the variance of the score is the expected information; the off-diagonal is not zero here.
  // Shape 1e-4 (mu = 0.4, sigma2 = 1600) is not among the points: at a small shape a the cdf
  // behaves as (y/scale)^a/Gamma(a + 1), so the median is about scale 2^(-1/a) = 4000 * 2^(-1e4),
  // near 1e-3007, and half the mass lies below the smallest positive double. No quadrature in y
  // can see that distribution, and the generic route misses the first component by 94 per cent
  // there. The closed form is right at that point: it agrees with the exact transform of the
  // gamma information, computed with mpmath at 300 bits, to 1 ulp in every component
  // (24.998125411173419, -0.0031246875513966776, 3.905859439245847e-07), checked 2026-11-14.
  for (point p : {point{0.4, 0.16}, point{2.5, 0.125}, point{1.0e4, 1.0}}) {
    const ld::gamma2_distrib d{p.mu, p.sigma2};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    if (std::getenv("LAPLACE_PRINT_ERRORS"))
      std::printf("gamma2 info mu %.17g sigma2 %.17g: closed %.17g %.17g %.17g, quadrature %.17g %.17g %.17g\n", p.mu,
                  p.sigma2, e[0], e[1], e[2], q[0], q[1], q[2]);
    INFO("gamma2, mu = " << p.mu << ", sigma2 = " << p.sigma2);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-9 * e[0]);
    CHECK(std::fabs(q[1] - e[1]) <= 1e-9 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-9 * e[2]);
  }
}

TEST_CASE("distrib gamma2: expected information by the generic quadrature route") {
  // the closed form against the generic route on the hidden distribution (score form, every order)
  for (point p : {point{0.4, 0.16}, point{2.5, 0.125}}) {
    const ld::gamma2_distrib d{p.mu, p.sigma2};
    const auto h = hide(d);
    const auto gen0 = generic_info_vec<0>(ld::expected_info_by_integral<0, ld::info_form::score>(h));
    const auto ex0 = as_vec<2, 2>(d.expected_info());
    double s0 = std::fmax(std::fabs(ex0[0]), std::fabs(ex0[2]));
    INFO("gamma2 generic, mu = " << p.mu << ", sigma2 = " << p.sigma2);
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(gen0[i] - ex0[i]) <= 1e-9 * s0);
    const auto gen1 = generic_info_vec<1>(ld::expected_info_by_integral<1, ld::info_form::score>(h));
    const auto ex1 = info_vec<2, 1>(d.expected_info_d1());
    double s1 = 0.0;
    for (double v : ex1) s1 = std::fmax(s1, std::fabs(v));
    for (int i = 0; i < (int)ex1.size(); ++i) CHECK(std::fabs(gen1[i] - ex1[i]) <= 1e-8 * s1);
  }
}

TEST_CASE("distrib gamma2: derivatives of the expected information against differences") {
  check_info_derivs<ld::gamma2_distrib>({0.4, 0.16}, 1e-8);
  check_info_derivs<ld::gamma2_distrib>({2.5, 0.125}, 1e-8);
  check_info_derivs<ld::gamma2_distrib>({0.4, 1600.0}, 1e-8);
}

TEST_CASE("distrib gamma2: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double c : {0.05, 0.6, 1.0, 1.4, 4.0}) check_cdf_derivs<ld::gamma2_distrib>({0.4, 0.16}, 0.4 * c, upper, 1e-7);
    for (double c : {0.2, 1.0, 2.5}) check_cdf_derivs<ld::gamma2_distrib>({2.5, 0.125}, 2.5 * c, upper, 1e-7);
    for (double c : {0.1, 1.0, 3.0}) check_cdf_derivs<ld::gamma2_distrib>({0.4, 1600.0}, 0.4 * c, upper, 1e-7);
  }
}

TEST_CASE("distrib gamma2: the same law as gamma1 at sigma2 = phi mu^2") {
  // gamma1 (mu, phi) and gamma2 (mu, sigma2) with sigma2 = phi mu^2 are one law: the first derivatives
  // transform by the Jacobian d(mu, phi)/d(mu, sigma2), d_mu|sigma2 = d_mu|phi - (2 phi/mu) d_phi and
  // d_sigma2 = d_phi / mu^2
  for (point p : {point{0.4, 0.16}, point{2.5, 0.125}, point{0.4, 1600.0}}) {
    const double mu = p.mu, phi = p.sigma2 / (mu * mu);
    const ld::gamma2_distrib g2{mu, p.sigma2};
    const ld::gamma1_distrib g1{mu, phi};
    for (double c : {0.3, 1.0, 2.2}) {
      const double y = mu * c;
      INFO("mu = " << mu << ", phi = " << phi << ", y = " << y);
      const auto l1 = g1.logpdf_d1(y), l2 = g2.logpdf_d1(y);
      CHECK(std::fabs(l2[0] - (l1[0] - 2.0 * phi / mu * l1[1])) <= 1e-10 * (std::fabs(l1[0]) + std::fabs(l1[1]) + 1.0));
      CHECK(std::fabs(l2[1] - l1[1] / (mu * mu)) <= 1e-10 * (std::fabs(l1[1]) / (mu * mu) + 1.0));
      const auto c1 = g1.log_cdf_d1(y), c2 = g2.log_cdf_d1(y);
      CHECK(std::fabs(c2[0] - (c1[0] - 2.0 * phi / mu * c1[1])) <= 1e-9 * (std::fabs(c1[0]) + std::fabs(c1[1]) + 1.0));
      CHECK(std::fabs(c2[1] - c1[1] / (mu * mu)) <= 1e-9 * (std::fabs(c1[1]) / (mu * mu) + 1.0));
      CHECK(g2.log_cdf(y) == doctest::Approx(g1.log_cdf(y)).epsilon(1e-12));
      CHECK(g2.log_ccdf(y) == doctest::Approx(g1.log_ccdf(y)).epsilon(1e-12));
    }
  }
}

TEST_CASE("distrib gamma2: cdf, ccdf, logarithms, quantile") {
  for (point p : {point{0.4, 0.16}, point{2.5, 0.125}, point{1.0e4, 1.0}, point{0.4, 1600.0}}) {
    const ld::gamma2_distrib g{p.mu, p.sigma2};
    for (double c : {0.001, 0.05, 0.5, 1.0, 2.0, 20.0}) {
      const double y = p.mu * c;
      CHECK(g.cdf(y) + g.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
      if (g.cdf(y) > 1e-300 && g.ccdf(y) > 1e-300) {
        CHECK(g.log_cdf(y) == doctest::Approx(std::log(g.cdf(y))).epsilon(1e-10));
        CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-10));
      }
      // the quantile inverts the cdf in the lower and in the upper tail
      if (g.cdf(y) > 1e-8 && g.cdf(y) < 1.0 - 1e-8) CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-9));
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for why (doctest gives each case an internal-linkage
// function whose number restarts per translation unit, and nvcc tags every extended lambda with that address).
void gamma2_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_gamma2", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.1 + 0.05 * i;
        const ld::gamma2_distrib g{0.4, 0.16};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<3, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                 g.expected_info_d4().pair[1][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y)[0] + g.log_cdf(y) +
                 g.log_ccdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.1 + 0.05 * i;
    const ld::gamma2_distrib g{0.4, 0.16};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<3, 3>(y)[1] + g.expected_info_d4().pair[2][4] +
                     g.expected_info_d4().pair[1][4] + g.log_cdf_d6(y)[2] + g.log_ccdf_d6(y)[0] + g.log_cdf(y) +
                     g.log_ccdf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("distrib gamma2: every order inside a kernel") { gamma2_kernel_case(); }

TEST_CASE("distrib gamma2: derivatives against mpmath (log f, log F, log(1 - F), the SymPy direct forms)") {
  // tools/reference/distrib_gamma2.py (2026-11-13): the mpmath values of the definitions at 200 and 260 bits,
  // kept where they agree to 2^-80. The error of a component is the difference of the library value and the
  // mpmath value (minus its low part), in ulp of the largest component of the same (kind, upper, y, mu,
  // sigma2, K) group ("scaled") and in ulp of the component ("own"). kind 2 rows are the SymPy closed forms
  // of orders 1 and 2 of the log-density in (mu, sigma2), evaluated at 250 bits.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_gamma2.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row {
    int kind, upper, K, b;
    double y, mu, s2, hi, lo;
  };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], mus[64], s2s[64], his[64], los[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", &r.kind, &r.upper, ys, mus, s2s, &r.K,
                    &r.b, his, los) != 9)
      continue;
    r.y = std::strtod(ys, nullptr), r.mu = std::strtod(mus, nullptr), r.s2 = std::strtod(s2s, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 1000);
  using key = std::tuple<int, int, double, double, double, int>;
  std::map<key, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[key(r.kind, r.upper, r.y, r.mu, r.s2, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  // bounds in ulp of the scaled error, per kind (0 log f, 1 log F or log(1 - F), 2 the SymPy direct form)
  double worst_scaled[3][2][7] = {}, worst_own[3][2][7] = {};
  for (const auto& r : rows) {
    const ld::gamma2_distrib g{r.mu, r.s2};
    std::vector<double> v;
    if (r.kind == 1) v = cderiv(g, r.y, r.K, r.upper == 1);
    else v = lderiv(g, r.y, r.K);
    const double sc = scale[key(r.kind, r.upper, r.y, r.mu, r.s2, r.K)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.kind][r.upper][r.K] = std::fmax(worst_scaled[r.kind][r.upper][r.K], ulp_scaled);
    worst_own[r.kind][r.upper][r.K] = std::fmax(worst_own[r.kind][r.upper][r.K], ulp_own);
    INFO("kind " << r.kind << " upper " << r.upper << " y " << r.y << " mu " << r.mu << " sigma2 " << r.s2 << " K " << r.K
                 << " b " << r.b << " ulp " << ulp_scaled);
    // Bounds at about 1.5 times the measured maxima in ulp of the order's scale, measured
    // 2026-11-14 on the gcc build over the 4510 rows of the table (shapes 1e-4 to 1e4,
    // c = y/mu from 0.01 to 10): log f 186 at order 6, log F and log(1 - F) 4.6e3 at order 6
    // (1.2e3 for log(1 - F)), and 5.41 for the direct forms of orders 1 and 2, which the
    // composed route reproduces to the same error as the log-density rows.
    const double bound[3] = {280.0, 6900.0, 8.2};
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_scaled > 500.0)
      std::printf("gamma2 row: kind %d upper %d y %a mu %a sigma2 %a K %d b %d shape %.3g c %.3g ulp %.4g\n", r.kind,
                  r.upper, r.y, r.mu, r.s2, r.K, r.b, r.mu * r.mu / r.s2, r.y / r.mu, ulp_scaled);
    CHECK(ulp_scaled <= bound[r.kind]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int kd = 0; kd < 3; ++kd)
      for (int u = 0; u < 2; ++u)
        for (int K = 1; K <= 6; ++K)
          std::printf("gamma2 kind %d upper %d K %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n", kd,
                      u, K, worst_scaled[kd][u][K], worst_own[kd][u][K]);
}

TEST_CASE("distrib gamma2: quantile inverts the cdf in both tails") {
  // shape 1e8 (mu = 1e4, sigma = 1): the points within a few standard deviations of the mean
  const ld::gamma2_distrib g{1.0e4, 1.0};
  for (double z : {-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0}) {
    const double y = 1.0e4 + z;
    CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-10));
  }
  const ld::gamma2_distrib h{0.4, 0.16};
  for (double p : {1e-12, 1e-6, 0.5, 1.0 - 1e-6}) CHECK(h.cdf(h.quantile(p)) == doctest::Approx(p).epsilon(1e-9));
}
