// chisq_distrib (degrees of freedom mu; include/laplace/distrib/chisq.hpp): packing and descriptors,
// every derivative order against a 5-point difference of the order below, the mixed (y, mu)
// derivatives, the expected information against the variance of the score and its derivatives
// against differences, the derivatives of log F and log(1 - F) against differences and against
// mpmath (tools/reference/distrib_expchi.py, mu from 1e-3 to 1e4), the cdf identities and the quantile
// in both tails, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/chisq.hpp>
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

// d^(J+K) l / dy^J dmu^K of the one-parameter chi-square (check_common::yderiv is two-parameter only)
#define CHISQ_YK(J, K) \
  case (J) * 10 + (K): return as_vec<1, K>(d.template logpdf_dy_dk<J, K>(y));
template <class D>
std::vector<double> yk(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    CHISQ_YK(0, 0) CHISQ_YK(0, 1) CHISQ_YK(0, 2) CHISQ_YK(0, 3) CHISQ_YK(0, 4) CHISQ_YK(0, 5) CHISQ_YK(0, 6)
    CHISQ_YK(1, 0) CHISQ_YK(1, 1) CHISQ_YK(1, 2) CHISQ_YK(1, 3) CHISQ_YK(1, 4) CHISQ_YK(1, 5)
    CHISQ_YK(2, 0) CHISQ_YK(2, 1) CHISQ_YK(2, 2) CHISQ_YK(2, 3) CHISQ_YK(2, 4)
    CHISQ_YK(3, 0) CHISQ_YK(3, 1) CHISQ_YK(3, 2) CHISQ_YK(3, 3)
    CHISQ_YK(4, 0) CHISQ_YK(4, 1) CHISQ_YK(4, 2)
    CHISQ_YK(5, 0) CHISQ_YK(5, 1)
    CHISQ_YK(6, 0)
    default: return {};
  }
}
#undef CHISQ_YK

// every (J, K), J + K <= 6: against the difference in y of (J - 1, K) and in mu of (J, K - 1)
template <class D>
void check_chisq_y_derivs(double mu, double y, double rel) {
  const D d{mu};
  const double hy = 1e-3 * y;
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yk(d, y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yk(d, y + s * hy, J - 1, K));
      // the difference of a term that does not depend on y (psi'(mu/2) at order K in mu) carries the
      // roundoff of its stencil values, so the bound scales with the largest stencil value
      for (const auto& w : v)
        for (double e : w) scale = std::fmax(scale, std::fabs(e));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", mu = " << mu << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12);
      }
      if (K == 0) continue;
      const std::vector<double> th{mu};
      const auto fd = diff5([&](const double* t) { return yk(make<D>(t), y, J, K - 1); }, th, 0,
                            1e-3 * std::fmax(1e-3, std::fabs(mu)));
      INFO("dmu: J = " << J << ", K = " << K << ", mu = " << mu << ", y = " << y);
      CHECK(std::fabs(exact[0] - fd[0]) <= rel * scale + 1e-12);
    }
}

}  // namespace

TEST_CASE("distrib chisq: descriptors") {
  CHECK(ld::chisq_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::chisq_distrib::n_params == 1);
  CHECK(ld::chisq_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::chisq_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::chisq_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::chisq_distrib::params[0].moves_support == false);
  const ld::chisq_distrib d{3.0};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.center() == 3.0);
  CHECK(d.scale() == doctest::Approx(std::sqrt(6.0)).epsilon(1e-15));
}

TEST_CASE("distrib chisq: logpdf value against the lgamma form") {
  // moderate mu, where the closed form with lgamma is accurate to about 1e-14 relative
  for (double mu : {0.5, 3.0, 12.0}) {
    const ld::chisq_distrib d{mu};
    for (double y : {0.2, 1.0, mu, 3.0 * mu}) {
      const double ref = (mu / 2 - 1) * std::log(y) - y / 2 - std::lgamma(mu / 2) - (mu / 2) * std::log(2.0);
      INFO("mu = " << mu << ", y = " << y);
      CHECK(d.logpdf(y) == doctest::Approx(ref).epsilon(1e-13).scale(1.0));
    }
    CHECK(d.logpdf(-1.0) == -ld::distrib_infinity);
  }
}

TEST_CASE("distrib chisq: derivatives against differences") {
  for (double y : {0.3, 2.5, 11.0}) check_derivs<ld::chisq_distrib>({3.0}, y, 1e-8);
  for (double y : {1e-5, 0.05}) check_derivs<ld::chisq_distrib>({1e-3}, y, 1e-8);
  for (double y : {0.4, 40.0}) check_derivs<ld::chisq_distrib>({0.05}, y, 1e-8);
  for (double y : {2.0, 150.0}) check_derivs<ld::chisq_distrib>({150.0}, y, 1e-8);
  for (double y : {9000.0, 1.0e4}) check_derivs<ld::chisq_distrib>({1e4}, y, 1e-8);
}

TEST_CASE("distrib chisq: derivatives in y and mixed (y, mu) against differences") {
  check_chisq_y_derivs<ld::chisq_distrib>(3.0, 2.5, 1e-8);
  check_chisq_y_derivs<ld::chisq_distrib>(0.05, 0.4, 1e-8);
  check_chisq_y_derivs<ld::chisq_distrib>(1e3, 980.0, 1e-8);
}

TEST_CASE("distrib chisq: expected information against the variance of the score") {
  // mu = 0.05 is left out: the score variance has Var(log Y) = psi'(mu/2) in its integrand, and for
  // mu/2 = 0.025 the mass of log Y below -200 is about 8 per cent, beyond the reach of the panel
  // quadrature (measured 2026-11-13: 222 against 400 for the closed form, verified by mpmath
  // integration in log y, which needs a far wider range than the panels).
  for (double mu : {0.5, 3.0, 40.0}) {
    const ld::chisq_distrib d{mu};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("chisq, mu = " << mu);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-9 * e[0]);
  }
}

TEST_CASE("distrib chisq: derivatives of the expected information against differences") {
  check_info_derivs<ld::chisq_distrib>({3.0}, 1e-8);
  check_info_derivs<ld::chisq_distrib>({0.05}, 1e-8);
}

TEST_CASE("distrib chisq: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    check_cdf_derivs<ld::chisq_distrib>({3.0}, 0.5, upper, 1e-7);
    check_cdf_derivs<ld::chisq_distrib>({3.0}, 12.0, upper, 1e-7);
    check_cdf_derivs<ld::chisq_distrib>({0.5}, 1e-4, upper, 1e-7);
    check_cdf_derivs<ld::chisq_distrib>({40.0}, 25.0, upper, 1e-7);
  }
}

TEST_CASE("distrib chisq: cdf identities and quantile in both tails") {
  const ld::chisq_distrib c{4.5};
  for (double y : {1e-6, 0.5, 4.5, 20.0, 60.0}) {
    CHECK(c.cdf(y) + c.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
    CHECK(c.log_ccdf(y) == doctest::Approx(std::log(c.ccdf(y))).epsilon(1e-12));
    if (y < 30.0) {
      CHECK(c.log_cdf(y) == doctest::Approx(std::log(c.cdf(y))).epsilon(1e-12));
      CHECK(c.quantile(c.cdf(y)) == doctest::Approx(y).epsilon(1e-9));
    }
  }
  // df = 2: F(y) = 1 - exp(-y/2) exactly
  const ld::chisq_distrib two{2.0};
  CHECK(two.cdf(3.0) == doctest::Approx(-std::expm1(-1.5)).epsilon(1e-14));
  CHECK(two.quantile(0.25) == doctest::Approx(-2.0 * std::log(0.75)).epsilon(1e-13));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
void chisq_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_chisq", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.2 + 0.7 * i;
        const ld::chisq_distrib c{5.3};
        out(i) = c.logpdf_d6(y)[0] + c.logpdf_dy_dk<2, 4>(y)[0] + c.expected_info_d4().pair[0][0] +
                 c.log_cdf_d6(y)[0] + c.log_ccdf_d6(y)[0] + c.log_cdf(y) + c.log_ccdf(y) + c.quantile(0.3);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.2 + 0.7 * i;
    const ld::chisq_distrib c{5.3};
    const double r = c.logpdf_d6(y)[0] + c.logpdf_dy_dk<2, 4>(y)[0] + c.expected_info_d4().pair[0][0] +
                     c.log_cdf_d6(y)[0] + c.log_ccdf_d6(y)[0] + c.log_cdf(y) + c.log_ccdf(y) + c.quantile(0.3);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-13));
  }
}

TEST_CASE("distrib chisq: every order inside a kernel") { chisq_kernel_case(); }

TEST_CASE("distrib chisq: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_expchi.py (mu = 1e-3 to 1e4, y = mu r with r from 1e-4 to 20), 2026-11-13.
  // The error of each derivative in ulp of its own value ("own") and in ulp of the largest value of its
  // order at the same mu ("scaled"); errors are the library value minus the mpmath value, its low part removed.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_chisq.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int upper, K; double y, hi, lo, mu; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64], mus[64];
    if (std::sscanf(line, "%d,%63[^,],%d,%63[^,],%63[^,],%63s", &r.upper, ys, &r.K, his, los, mus) != 6) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 600);
  std::map<std::tuple<int, int, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.K, r.mu)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {}, worst_y[2][7] = {}, worst_mu[2][7] = {};
  for (const auto& r : rows) {
    const ld::chisq_distrib d{r.mu};
    const auto v = cderiv(d, r.y, r.K, r.upper);
    const double sc = scale[std::make_tuple(r.upper, r.K, r.mu)];
    const double err = std::fabs((v[0] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(sc) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    if (ulp_scaled > worst_scaled[r.upper][r.K]) worst_y[r.upper][r.K] = r.y, worst_mu[r.upper][r.K] = r.mu;
    worst_scaled[r.upper][r.K] = std::fmax(worst_scaled[r.upper][r.K], ulp_scaled);
    worst_own[r.upper][r.K] = std::fmax(worst_own[r.upper][r.K], ulp_own);
    INFO("upper " << r.upper << " y " << r.y << " K " << r.K << " mu " << r.mu << " ulp " << ulp_scaled << " own "
                  << ulp_own);
    // bounds: measured maxima (2026-11-13, gcc Serial + OpenMP build, LAPLACE_PRINT_ERRORS) times about 1.5:
    // log F: 1.41, 18, 59.5, 111, 251, 294 ulp; log(1 - F): 0.83, 7.6, 23.6, 41.5, 292, 525 ulp
    const double bound[2][7] = {{0.0, 2.2, 27.0, 90.0, 167.0, 377.0, 441.0}, {0.0, 1.3, 11.4, 36.0, 63.0, 438.0, 788.0}};
    CHECK(ulp_scaled <= bound[r.upper][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("chisq upper %d K %d: max %.3g ulp of the order's scale (y %.6g, mu %.6g), %.3g ulp of the value\n",
                    u, K, worst_scaled[u][K], worst_y[u][K], worst_mu[u][K], worst_own[u][K]);
}
