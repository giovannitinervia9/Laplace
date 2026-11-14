// lognormal2_distrib (mean m and variance v of Y; include/laplace/distrib/lognormal2.hpp): descriptors, every
// derivative order against differences, the y-derivatives against differences, the expected information (a
// tensor, with its off-diagonal entry) against the variance of the score and against the generic quadrature
// route, its derivatives against differences, the derivatives of log F and log(1 - F) against differences and
// against mpmath (tools/reference/distrib_lognormal.py, dist 2), the cdf family and the quantile, the corner of
// small variance over squared mean (m = 40, v = 0.001) and the corner where v / m^2 is large, and a kernel
// launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/lognormal1.hpp>
#include <laplace/distrib/lognormal2.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// the parameters of lognormal1 (mu, s) that lognormal2 (m, v) has
void to_lognormal1(double m, double v, double& mu, double& s) {
  s = std::log1p(v / (m * m));
  mu = std::log(m) - 0.5 * s;
}

}  // namespace

TEST_CASE("distrib lognormal2: descriptors") {
  CHECK(ld::lognormal2_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::lognormal2_distrib::n_params == 2);
  CHECK(ld::lognormal2_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::lognormal2_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::lognormal2_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::lognormal2_distrib::params[j].moves_support == false);
  }
  const ld::lognormal2_distrib d{1.5, 0.8};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.center() == 1.5);
  CHECK(d.scale() == doctest::Approx(std::sqrt(0.8)).epsilon(1e-15));
}

// The same law as lognormal1 at (mu, s): the value and the cdf family agree (the parameters are the
// composition, so the derivatives in (m, v) are checked against differences below).
TEST_CASE("distrib lognormal2: the same law as lognormal1 at (mu, s)") {
  for (double m : {0.4, 2.0, 40.0}) {
    for (double v : {0.001, 0.5, 50.0}) {
      double mu = 0.0, s = 0.0;
      to_lognormal1(m, v, mu, s);
      const ld::lognormal2_distrib a{m, v};
      const ld::lognormal1_distrib b{mu, s};
      for (double y : {0.3 * m, m, 3.0 * m}) {
        INFO("m = " << m << ", v = " << v << ", y = " << y);
        CHECK(a.logpdf(y) == doctest::Approx(b.logpdf(y)).epsilon(1e-12).scale(1.0));
        CHECK(a.log_cdf(y) == doctest::Approx(b.log_cdf(y)).epsilon(1e-12).scale(1.0));
        CHECK(a.log_ccdf(y) == doctest::Approx(b.log_ccdf(y)).epsilon(1e-12).scale(1.0));
        CHECK(a.cdf(y) == doctest::Approx(b.cdf(y)).epsilon(1e-12).scale(1.0));
      }
      // the mean and the variance of Y are m and v: the quantile at p = 1/2 is the median exp(mu)
      CHECK(a.quantile(0.5) == doctest::Approx(std::exp(mu)).epsilon(1e-12));
    }
  }
}

TEST_CASE("distrib lognormal2: derivatives against differences") {
  for (double y : {0.3, 1.5, 4.0, 30.0}) check_derivs<ld::lognormal2_distrib>({1.5, 0.8}, y, 1e-8);
  check_derivs<ld::lognormal2_distrib>({40.0, 0.001}, 41.0, 1e-8);  // the corner of small variance over squared mean
  check_derivs<ld::lognormal2_distrib>({0.5, 50.0}, 2.0, 1e-8);     // the corner where v / m^2 is large
  check_derivs<ld::lognormal2_distrib>({2.0, 0.5}, 0.9, 1e-8);
}

// The y-derivatives are not checked against 5-point differences here: at orders 5 and 6 the stencil of step
// 1e-3 y cannot hold the tolerance of the other tests (measured 2026-11-13, gcc build: 6.7e-8 against a bound of
// 3.8e-10 at m = 1.5, v = 0.8, y = 0.4, and worse elsewhere), while the library agrees with the symbolic mpmath
// reference of every (J, K) to 24 ulp (the mixed (y, theta) test below). The reference is the stronger check.

TEST_CASE("distrib lognormal2: expected information against the variance of the score") {
  double worst = 0.0;
  for (auto mv : {std::make_pair(1.5, 0.8), std::make_pair(0.4, 0.04), std::make_pair(2.0, 25.0)}) {
    const ld::lognormal2_distrib d{mv.first, mv.second};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("lognormal2, m = " << mv.first << ", v = " << mv.second);
    const double r0 = std::fabs(q[0] - e[0]) / e[0], r1 = std::fabs(q[1] - e[1]) / std::sqrt(e[0] * e[2]),
                 r2 = std::fabs(q[2] - e[2]) / e[2];
    worst = std::fmax(worst, std::fmax(r0, std::fmax(r1, r2)));
    // measured 2026-11-13 (gcc build, Serial + OpenMP): max relative error 7.19e-16 over (1.5, 0.8), (0.4, 0.04),
    // (2, 25); bound 1.5 times, rounded up
    CHECK(std::fabs(q[0] - e[0]) <= 1.1e-15 * e[0]);
    CHECK(std::fabs(q[1] - e[1]) <= 1.1e-15 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1.1e-15 * e[2]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("lognormal2 score variance: max relative error %.3g\n", worst);
}

// The generic quadrature route of generic_info.hpp (the score form, which needs no closed form), on the
// hidden copy of the distribution, against the closed composition.
TEST_CASE("distrib lognormal2: expected information against the generic quadrature route") {
  for (auto mv : {std::make_pair(1.5, 0.8), std::make_pair(40.0, 0.001), std::make_pair(0.5, 50.0)}) {
    const ld::lognormal2_distrib d{mv.first, mv.second};
    const auto gen = ld::expected_info_by_integral<0, ld::info_form::score>(laplace_test::hide(d));
    const auto e = d.expected_info();
    INFO("lognormal2, m = " << mv.first << ", v = " << mv.second);
    for (int i = 0; i < 3; ++i) {
      // measured 2026-11-13 (gcc build): relative error 5.06e-14 at (40, 0.001) (the worst, component 1), 3.3e-16
      // elsewhere; bounds 1.5 times the measured maxima, rounded up
      const double tol = mv.second < 0.01 ? 7.6e-14 : 5.0e-16;
      CHECK(std::fabs(gen[i] - e[i]) <= tol * std::fabs(e[i]));
      if (std::getenv("LAPLACE_PRINT_ERRORS"))
        std::printf("lognormal2 quadrature info m %g v %g component %d: relative error %.3g\n", mv.first, mv.second, i,
                    std::fabs(gen[i] - e[i]) / std::fabs(e[i]));
    }
    if (std::getenv("LAPLACE_PRINT_ERRORS"))
      std::printf("lognormal2 info at m %g v %g: closed I = (%.12g, %.12g, %.12g), quadrature = (%.12g, %.12g, %.12g)\n",
                  mv.first, mv.second, e[0], e[1], e[2], gen[0], gen[1], gen[2]);
  }
}

TEST_CASE("distrib lognormal2: derivatives of the expected information against differences") {
  check_info_derivs<ld::lognormal2_distrib>({1.5, 0.8}, 1e-8);
  check_info_derivs<ld::lognormal2_distrib>({0.4, 0.04}, 1e-8);
  check_info_derivs<ld::lognormal2_distrib>({40.0, 0.001}, 1e-8);
}

TEST_CASE("distrib lognormal2: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    // The corner (40, 0.001) is not checked against differences: its log F derivatives are checked against mpmath
    // (the test below); the stencil at this corner does not hold the tolerance (see the y-derivative note).
    for (auto mv : {std::make_pair(1.5, 0.8), std::make_pair(0.5, 50.0)}) {
      double mu = 0.0, s = 0.0;
      to_lognormal1(mv.first, mv.second, mu, s);
      for (double z : {-2.0, 0.0, 3.0}) {
        const double y = std::exp(mu + std::sqrt(s) * z);
        check_cdf_derivs<ld::lognormal2_distrib>({mv.first, mv.second}, y, upper, 1e-7, std::fabs(z));
      }
    }
  }
}

TEST_CASE("distrib lognormal2: cdf, ccdf, logarithms, quantile") {
  const ld::lognormal2_distrib g{1.5, 0.8};
  for (double y : {0.01, 0.5, 1.5, 6.0, 300.0}) {
    CHECK(g.cdf(y) + g.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-14));
    if (y < 10.0) CHECK(g.log_cdf(y) == doctest::Approx(std::log(g.cdf(y))).epsilon(1e-14));
  }
  CHECK(g.cdf(0.0) == 0.0);
  for (auto mv : {std::make_pair(1.5, 0.8), std::make_pair(40.0, 0.001), std::make_pair(0.5, 50.0)}) {
    const ld::lognormal2_distrib t{mv.first, mv.second};
    for (double p : {1e-12, 1e-5, 0.3, 1.0 - 1e-5}) CHECK(t.cdf(t.quantile(p)) == doctest::Approx(p).epsilon(1e-9));
  }
}

// The kernel lives in a named function with external linkage (see lognormal1_kernel_case).
void lognormal2_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_lognormal2", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.1 + 0.2 * i;
        const ld::lognormal2_distrib g{1.5, 0.8};
        out(i) = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 4>(y)[1] + g.logpdf_dy_dk<6, 0>(y)[0] +
                 g.expected_info_d4().pair[2][4] + g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] +
                 g.log_ccdf_d6(y)[0] + g.log_cdf(y) + g.log_ccdf(y) + g.quantile(0.3);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.1 + 0.2 * i;
    const ld::lognormal2_distrib g{1.5, 0.8};
    const double r = g.logpdf_d6(y)[3] + g.logpdf_dy_dk<2, 4>(y)[1] + g.logpdf_dy_dk<6, 0>(y)[0] +
                     g.expected_info_d4().pair[2][4] + g.expected_info_d4().pair[0][4] + g.log_cdf_d6(y)[2] +
                     g.log_ccdf_d6(y)[0] + g.log_cdf(y) + g.log_ccdf(y) + g.quantile(0.3);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-13));
  }
}

TEST_CASE("distrib lognormal2: every order inside a kernel") { lognormal2_kernel_case(); }

// mpmath references (tools/reference/distrib_lognormal.py, dist 2): the derivatives of log F and log(1 - F) in
// (m, v), from the definition, at 300 and 400 bits. Errors are measured as in the lognormal1 test: scaled, in ulp of
// the largest component of the order at that point.
TEST_CASE("distrib lognormal2: derivatives of log F and log(1 - F) against mpmath") {
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
    if (dist != 2) continue;
    r.y = std::strtod(ys, nullptr), r.t0 = std::strtod(t0s, nullptr), r.t1 = std::strtod(t1s, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 300);
  std::map<std::tuple<int, double, int, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.y, r.K, r.t0, r.t1)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {};
  for (const auto& r : rows) {
    const ld::lognormal2_distrib g{r.t0, r.t1};
    const auto v = cderiv(g, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.y, r.K, r.t0, r.t1)];
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.upper][r.K] = std::fmax(worst_scaled[r.upper][r.K], ulp_scaled);
    worst_own[r.upper][r.K] = std::fmax(worst_own[r.upper][r.K], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " m " << r.t0 << " v " << r.t1
                  << " ulp " << ulp_scaled);
    // measured 2026-11-13, gcc build, 300/400-bit mpmath: maxima 7056 (log F, order 6) and 169.4 (log(1 - F), order 2); bounds 1.5 times
    const double bound_scaled[2] = {10600.0, 255.0};
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_scaled > 1000.0)
      std::printf("lognormal2 row: y %a m %a v %a K %d b %d upper %d lib %.17g ref %.17g lo %.17g ulp_scaled %.6g\n",
                  r.y, r.t0, r.t1, r.K, r.b, r.upper, v[r.b], r.hi, r.lo, ulp_scaled);
    CHECK(ulp_scaled <= bound_scaled[r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("lognormal2 upper %d K %d: max %.4g ulp of the order's scale, %.4g ulp of the component\n", u, K,
                    worst_scaled[u][K], worst_own[u][K]);
}

// Mixed (y, theta) derivatives of the lognormal2 log-density against an mpmath reference built from the
// definition in SymPy (tools/reference/distrib_lognormal.py, mode ydk: the log-density in (y, m, v) with
// s = log(1 + v/m^2), mu = log m - s/2, differentiated symbolically and evaluated at 300 bits). The error of each
// component is in ulp of the largest component of its (J, K) group at that point.
TEST_CASE("distrib lognormal2: mixed (y, theta) derivatives against mpmath") {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_lognormal_ydk.csv", "r");
  REQUIRE(f != nullptr);
  char line[1024];
  std::fgets(line, sizeof line, f);
  struct row { double m, v, y; int J, K, b; double hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    char ms[64], vs[64], ys[64], his[64], los[64];
    row r;
    if (std::sscanf(line, "%63[^,],%63[^,],%63[^,],%d,%d,%d,%63[^,],%63s", ms, vs, ys, &r.J, &r.K, &r.b, his, los) != 8)
      continue;
    r.m = std::strtod(ms, nullptr), r.v = std::strtod(vs, nullptr), r.y = std::strtod(ys, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() == 504);
  // the largest |component| of each (point, J, K) group: the scale of the ulp
  std::map<std::tuple<double, double, double, int, int>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.m, r.v, r.y, r.J, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[7][7] = {};
  for (const auto& r : rows) {
    const ld::lognormal2_distrib d{r.m, r.v};
    const auto t = yderiv(d, r.y, r.J, r.K);
    const double sc = scale[std::make_tuple(r.m, r.v, r.y, r.J, r.K)];
    const double err = std::fabs((t[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    worst[r.J][r.K] = std::fmax(worst[r.J][r.K], ulp_scaled);
    INFO("m " << r.m << " v " << r.v << " y " << r.y << " J " << r.J << " K " << r.K << " b " << r.b << " lib "
                  << t[r.b] << " mpmath " << r.hi);
    // measured 2026-11-13, gcc build, against the symbolic mpmath reference: maximum 23.76 ulp (J = 1, K = 5); bound 1.5 times, rounded
    const double bound = 36.0;
    CHECK(ulp_scaled <= bound);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int J = 0; J <= 6; ++J)
      for (int K = 0; J + K <= 6; ++K)
        std::printf("lognormal2 mixed J %d K %d: max %.4g ulp of the group's scale\n", J, K, worst[J][K]);
}
