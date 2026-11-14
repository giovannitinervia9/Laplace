// lognormal1_distrib (mean mu and variance v of log Y; include/laplace/distrib/lognormal1.hpp): descriptors,
// every derivative order against differences, the y-derivatives (Stirling combination) against
// differences, the identity with gaussian2 at x = log y, the expected information against the variance of
// the score and its derivatives against differences, the derivatives of log F and log(1 - F) against
// differences and against mpmath (tools/reference/distrib_lognormal.py, dist 1), the cdf family and the
// quantile, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("distrib lognormal1: descriptors") {
  CHECK(ld::lognormal1_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::lognormal1_distrib::n_params == 2);
  CHECK(ld::lognormal1_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::lognormal1_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::lognormal1_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::lognormal1_distrib::params[j].moves_support == false);
  }
  const ld::lognormal1_distrib d{0.3, 0.25};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.center() == doctest::Approx(std::exp(0.3)).epsilon(1e-15));
  CHECK(d.scale() == doctest::Approx(std::exp(0.3) * 0.5).epsilon(1e-15));
}

// The identity the implementation rests on: each parameter derivative of the lognormal1 log-density is
// gaussian2's at x = log y, bit for bit (the forwarding), and the value differs by -log y only.
TEST_CASE("distrib lognormal1: parameter derivatives are gaussian2's at log y") {
  for (double y : {0.05, 0.7, 2.5, 60.0}) {
    const ld::lognormal1_distrib l{0.3, 0.25};
    const ld::gaussian2_distrib g{0.3, 0.25};
    const double x = std::log(y);
    CHECK(l.logpdf(y) == doctest::Approx(g.logpdf(x) - x).epsilon(1e-15));
    for (int b = 0; b <= 3; ++b) CHECK(l.logpdf_d3(y)[b] == g.logpdf_d3(x)[b]);
    for (int b = 0; b <= 2; ++b) CHECK(l.log_cdf_d2(y)[b] == g.log_cdf_d2(x)[b]);
    for (int b = 0; b <= 2; ++b) CHECK(l.log_ccdf_d2(y)[b] == g.log_ccdf_d2(x)[b]);
    CHECK(l.log_cdf(y) == g.log_cdf(x));
    CHECK(l.cdf(y) == g.cdf(x));
    CHECK(l.expected_info()[2] == g.expected_info()[2]);
    CHECK(l.expected_info_d2().pair[0][2] == g.expected_info_d2().pair[0][2]);
  }
}

TEST_CASE("distrib lognormal1: derivatives against differences") {
  for (double y : {0.05, 0.7, 2.5, 60.0}) check_derivs<ld::lognormal1_distrib>({0.3, 0.25}, y, 1e-8);
  check_derivs<ld::lognormal1_distrib>({-1.2, 0.01}, 0.3, 1e-8);  // small variance of log Y
  check_derivs<ld::lognormal1_distrib>({1.0, 25.0}, 5.0, 1e-8);   // large variance of log Y
}

TEST_CASE("distrib lognormal1: y-derivatives against differences") {
  for (double y : {0.4, 2.5, 9.0}) {
    check_y_derivs<ld::lognormal1_distrib>({0.3, 0.25}, y, 1e-3 * y, 1e-8);
  }
  check_y_derivs<ld::lognormal1_distrib>({-1.2, 0.04}, 0.3, 1e-4, 1e-8);
}

TEST_CASE("distrib lognormal1: expected information against the variance of the score") {
  double worst = 0.0, worst_off = 0.0;
  for (double v : {0.04, 0.25, 4.0}) {
    const ld::lognormal1_distrib d{0.3, v};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("lognormal1, v = " << v);
    worst = std::fmax(worst, std::fmax(std::fabs(q[0] - e[0]) / e[0], std::fabs(q[2] - e[2]) / e[2]));
    worst_off = std::fmax(worst_off, std::fabs(q[1]) / std::sqrt(e[0] * e[2]));
    // measured 2026-11-13 (gcc build): max relative error 2.22e-16 (diagonal), 1.03e-16 (off-diagonal); bounds 1.5 times, rounded up
    CHECK(std::fabs(q[0] - e[0]) <= 3.5e-16 * e[0]);
    CHECK(std::fabs(q[1]) <= 1.6e-16 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 3.5e-16 * e[2]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("lognormal1 score variance: max relative error diagonal %.3g, off-diagonal %.3g\n", worst, worst_off);
}

TEST_CASE("distrib lognormal1: derivatives of the expected information against differences") {
  check_info_derivs<ld::lognormal1_distrib>({0.3, 0.25}, 1e-8);
  check_info_derivs<ld::lognormal1_distrib>({-1.2, 0.04}, 1e-8);
}

TEST_CASE("distrib lognormal1: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    const double mu = 0.3, s = 0.5;
    for (double z : {-6.0, -3.0, 0.0, 2.0, 5.0}) {
      const double y = std::exp(mu + s * z);
      check_cdf_derivs<ld::lognormal1_distrib>({mu, s * s}, y, upper, 1e-7, std::fabs(z));
    }
    const double mu2 = -1.2, s2 = 0.1;
    for (double z : {-5.0, 0.0, 4.0}) {
      const double y = std::exp(mu2 + s2 * z);
      check_cdf_derivs<ld::lognormal1_distrib>({mu2, s2 * s2}, y, upper, 1e-7, std::fabs(z));
    }
  }
}

TEST_CASE("distrib lognormal1: cdf, ccdf, logarithms, quantile") {
  const ld::lognormal1_distrib l{0.3, 0.25};
  for (double y : {0.001, 0.2, 1.0, 4.0, 1000.0}) {
    CHECK(l.cdf(y) + l.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(l.log_ccdf(y) == doctest::Approx(std::log(l.ccdf(y))).epsilon(1e-14));
    if (y < 10.0) {
      CHECK(l.log_cdf(y) == doctest::Approx(std::log(l.cdf(y))).epsilon(1e-14));
      CHECK(l.quantile(l.cdf(y)) == doctest::Approx(y).epsilon(1e-10));
    }
  }
  CHECK(l.cdf(0.0) == 0.0);
  CHECK(l.ccdf(-1.0) == 1.0);
  CHECK(l.quantile(0.5) == doctest::Approx(std::exp(0.3)).epsilon(1e-14));
  // the quantile inverts the cdf in both tails
  const ld::lognormal1_distrib t{-0.4, 0.04};
  for (double p : {1e-12, 1e-5, 1.0 - 1e-5}) CHECK(t.cdf(t.quantile(p)) == doctest::Approx(p).epsilon(1e-9));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE (see the
// comment on gaussian2_kernel_case in test_gaussian2.cpp).
void lognormal1_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_lognormal1", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.05 + 0.1 * i;
        const ld::lognormal1_distrib l{0.3, 0.25};
        out(i) = l.logpdf_d6(y)[3] + l.logpdf_dy_dk<2, 4>(y)[1] + l.logpdf_dy_dk<6, 0>(y)[0] +
                 l.expected_info_d4().pair[2][4] + l.expected_info_d4().pair[0][4] + l.log_cdf_d6(y)[2] +
                 l.log_ccdf_d6(y)[0] + l.log_cdf(y) + l.log_ccdf(y) + l.quantile(0.3);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.05 + 0.1 * i;
    const ld::lognormal1_distrib l{0.3, 0.25};
    const double r = l.logpdf_d6(y)[3] + l.logpdf_dy_dk<2, 4>(y)[1] + l.logpdf_dy_dk<6, 0>(y)[0] +
                     l.expected_info_d4().pair[2][4] + l.expected_info_d4().pair[0][4] + l.log_cdf_d6(y)[2] +
                     l.log_ccdf_d6(y)[0] + l.log_cdf(y) + l.log_ccdf(y) + l.quantile(0.3);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-13));
  }
}

TEST_CASE("distrib lognormal1: every order inside a kernel") { lognormal1_kernel_case(); }

// mpmath references (tools/reference/distrib_lognormal.py, dist 1): the derivatives of log F and log(1 - F) in
// (mu, v), from the definition, at 300 and 400 bits. The error of each component is measured in ulp of the largest
// component of its order at that point ("scaled"), as in test_gaussian2.cpp.
TEST_CASE("distrib lognormal1: derivatives of log F and log(1 - F) against mpmath") {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_lognormal.csv", "r");
  REQUIRE(f != nullptr);
  char line[1024];
  std::fgets(line, sizeof line, f);
  struct row { int upper, K, b; double y, t0, t1, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    int dist = 0;
    char ys[64], t0s[64], t1s[64], his[64], los[64];
    row r;
    if (std::sscanf(line, "%d,%d,%d,%d,%63[^,],%63[^,],%63[^,],%63[^,],%63s", &dist, &r.upper, &r.K, &r.b, ys, t0s,
                    t1s, his, los) != 9)
      continue;
    if (dist != 1) continue;
    r.y = std::strtod(ys, nullptr), r.t0 = std::strtod(t0s, nullptr), r.t1 = std::strtod(t1s, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 300);
  // the largest |component| of each (upper, y, K, parameters) group: the scale of the ulp
  std::map<std::tuple<int, double, int, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.t0, r.t1)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {};
  for (const auto& r : rows) {
    // The twin of lognormal2's corner (40, 0.001), s = 6.25e-7, is excluded from the plain-ulp check: there the residual
    // log y - mu (about 1e-17) lies below the rounding of log y (about 4e-16), so the error is set by the conditioning of
    // the input. Measured 2026-11-13 (gcc build), by one-ulp perturbations of y and of mu and mpmath at 300 bits: the
    // library error is at most 0.165 of max(one ulp of the result, change per input ulp) over the rows with plain errors
    // up to 5.4e6 ulp (orders 1 to 6, both tails).
    if (r.t1 < 1e-5) continue;
    const ld::lognormal1_distrib l{r.t0, r.t1};
    const auto v = cderiv(l, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.y, r.K, r.t0, r.t1)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.upper][r.K] = std::fmax(worst_scaled[r.upper][r.K], ulp_scaled);
    worst_own[r.upper][r.K] = std::fmax(worst_own[r.upper][r.K], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " mu " << r.t0 << " v " << r.t1
                  << " ulp " << ulp_scaled);
    // bounds: 1.5 times the measured maxima, rounded up. Measured 2026-11-13 on the gcc build against
    // mpmath at 300 and 400 bits, over every row except the excluded tiny-variance twin: 322.2 ulp of
    // the order's scale for log F (order 6) and 648.2 for log(1 - F) (order 6).
    const double bound_scaled[2] = {483.0, 973.0};
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_scaled > 1000.0)
      std::printf("lognormal1 row: y %a mu %a v %a K %d b %d upper %d lib %.17g ref %.17g lo %.17g ulp_scaled %.6g\n",
                  r.y, r.t0, r.t1, r.K, r.b, r.upper, v[r.b], r.hi, r.lo, ulp_scaled);
    CHECK(ulp_scaled <= bound_scaled[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("lognormal1 upper %d K %d: max %.4g ulp of the order's scale, %.4g ulp of the component\n", u, K,
                    worst_scaled[u][K], worst_own[u][K]);
}
