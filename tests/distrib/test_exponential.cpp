// exponential_distrib (mean mu; include/laplace/distrib/exponential.hpp): packing and descriptors,
// every derivative order against a 5-point difference of the order below, the mixed (y, mu)
// derivatives, the expected information against the variance of the score and its derivatives
// against differences, the derivatives of log F and log(1 - F) against differences and against
// mpmath (tools/reference/distrib_expchi.py), the cdf identities and the quantile in both tails,
// and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/exponential.hpp>
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

// d^(J+K) l / dy^J dmu^K of the one-parameter exponential (check_common::yderiv is two-parameter only)
#define EXP_YK(J, K) \
  case (J) * 10 + (K): return as_vec<1, K>(d.template logpdf_dy_dk<J, K>(y));
template <class D>
std::vector<double> yk(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    EXP_YK(0, 0) EXP_YK(0, 1) EXP_YK(0, 2) EXP_YK(0, 3) EXP_YK(0, 4) EXP_YK(0, 5) EXP_YK(0, 6)
    EXP_YK(1, 0) EXP_YK(1, 1) EXP_YK(1, 2) EXP_YK(1, 3) EXP_YK(1, 4) EXP_YK(1, 5)
    EXP_YK(2, 0) EXP_YK(2, 1) EXP_YK(2, 2) EXP_YK(2, 3) EXP_YK(2, 4)
    EXP_YK(3, 0) EXP_YK(3, 1) EXP_YK(3, 2) EXP_YK(3, 3)
    EXP_YK(4, 0) EXP_YK(4, 1) EXP_YK(4, 2)
    EXP_YK(5, 0) EXP_YK(5, 1)
    EXP_YK(6, 0)
    default: return {};
  }
}
#undef EXP_YK

// every (J, K), J + K <= 6: against the difference in y of (J - 1, K) and in mu of (J, K - 1)
template <class D>
void check_exp_y_derivs(double mu, double y, double rel) {
  const D d{mu};
  const double hy = 1e-3 * y;
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yk(d, y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yk(d, y + s * hy, J - 1, K));
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

TEST_CASE("distrib exponential: descriptors") {
  CHECK(ld::exponential_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::exponential_distrib::n_params == 1);
  CHECK(ld::exponential_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::exponential_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::exponential_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::exponential_distrib::params[0].moves_support == false);
  const ld::exponential_distrib d{2.5};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 2.5);
  CHECK(d.scale() == 2.5);
}

TEST_CASE("distrib exponential: logpdf value against the definition") {
  for (double mu : {1e-3, 0.7, 40.0}) {
    const ld::exponential_distrib d{mu};
    for (double y : {1e-6, 0.3, 2.0, 5.0 * mu}) {
      const double ref = -std::log(mu) - y / mu;
      CHECK(d.logpdf(y) == doctest::Approx(ref).epsilon(1e-14).scale(1.0));
    }
    CHECK(d.logpdf(-1.0) == -ld::distrib_infinity);
  }
}

TEST_CASE("distrib exponential: derivatives against differences") {
  for (double y : {0.05, 1.7, 9.0}) check_derivs<ld::exponential_distrib>({2.0}, y, 1e-8);
  for (double y : {1e-4, 3.0}) check_derivs<ld::exponential_distrib>({1e-3}, y, 1e-8);
  for (double y : {2.0e4, 1.3e4}) check_derivs<ld::exponential_distrib>({1e4}, y, 1e-8);
}

TEST_CASE("distrib exponential: derivatives in y and mixed (y, mu) against differences") {
  check_exp_y_derivs<ld::exponential_distrib>(2.0, 1.3, 1e-8);
  check_exp_y_derivs<ld::exponential_distrib>(1e-3, 0.4, 1e-8);
  check_exp_y_derivs<ld::exponential_distrib>(1e3, 2500.0, 1e-8);
}

TEST_CASE("distrib exponential: expected information against the variance of the score") {
  for (double mu : {1e-3, 0.4, 2.0, 1e4}) {
    const ld::exponential_distrib d{mu};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("exponential, mu = " << mu);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-10 * e[0]);
  }
}

TEST_CASE("distrib exponential: derivatives of the expected information against differences") {
  check_info_derivs<ld::exponential_distrib>({2.0}, 1e-8);
  check_info_derivs<ld::exponential_distrib>({1e-3}, 1e-8);
}

TEST_CASE("distrib exponential: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double u : {1e-6, 0.3, 0.9, 1.0, 1.5, 7.0, 40.0}) check_cdf_derivs<ld::exponential_distrib>({2.0}, 2.0 * u, upper, 1e-7);
  }
}

TEST_CASE("distrib exponential: cdf identities and quantile in both tails") {
  const ld::exponential_distrib e{1.5};
  for (double y : {1e-8, 0.2, 1.5, 6.0, 40.0}) {
    CHECK(e.cdf(y) + e.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(e.log_ccdf(y) == doctest::Approx(std::log(e.ccdf(y))).epsilon(1e-14));
    CHECK(e.log_cdf(y) == doctest::Approx(std::log(e.cdf(y))).epsilon(1e-13));
    if (y < 30.0) CHECK(e.quantile(e.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
  }
  // lower tail: p tiny, the quantile is about mu p
  CHECK(e.quantile(1e-12) == doctest::Approx(1.5e-12).epsilon(1e-10));
  // upper tail: 1 - p = 2^-40 (exact in double), the quantile is -mu log(1 - p) = 40 mu log 2
  CHECK(e.quantile(1.0 - 0x1p-40) == doctest::Approx(1.5 * 40.0 * std::log(2.0)).epsilon(1e-13));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp (doctest gives each case an
// internal-linkage function whose number restarts per translation unit, and nvcc tags every
// extended lambda with that address).
void exponential_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_exponential", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.1 + 0.3 * i;
        const ld::exponential_distrib e{1.7};
        out(i) = e.logpdf_d6(y)[0] + e.logpdf_dy_dk<1, 5>(y)[0] + e.expected_info_d4().pair[0][0] +
                 e.log_cdf_d6(y)[0] + e.log_ccdf_d6(y)[0] + e.log_cdf(y) + e.log_ccdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.1 + 0.3 * i;
    const ld::exponential_distrib e{1.7};
    const double r = e.logpdf_d6(y)[0] + e.logpdf_dy_dk<1, 5>(y)[0] + e.expected_info_d4().pair[0][0] +
                     e.log_cdf_d6(y)[0] + e.log_ccdf_d6(y)[0] + e.log_cdf(y) + e.log_ccdf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-13));
  }
}

TEST_CASE("distrib exponential: every order inside a kernel") { exponential_kernel_case(); }

TEST_CASE("distrib exponential: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_expchi.py (grid u = y/mu from 1e-8 to 300, mu = 1e-3, 1, 1e4). The error of
  // each derivative in ulp of its own value ("own") and in ulp of the largest value of its order at the
  // same mu ("scaled"); errors are the library value minus the mpmath value, its low part removed.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_exponential.csv", "r");
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
  // the largest |value| of each (upper, K, mu) group, the scale of the "scaled" ulp
  std::map<std::tuple<int, int, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.upper, r.K, r.mu)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst_scaled[2][7] = {}, worst_own[2][7] = {}, worst_y[2][7] = {}, worst_mu[2][7] = {};
  for (const auto& r : rows) {
    const ld::exponential_distrib d{r.mu};
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
    // log F: 1.7, 1.74, 6.39, 7.77, 11.9, 22.6 ulp; log(1 - F): 0.68, 0.501, 0.887, 1.02, 1.42, 1.68 ulp
    const double bound[2][7] = {{0.0, 3.0, 3.0, 10.0, 12.0, 18.0, 34.0}, {0.0, 1.1, 1.0, 1.5, 1.6, 2.2, 2.6}};
    CHECK(ulp_scaled <= bound[r.upper][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int u = 0; u < 2; ++u)
      for (int K = 1; K <= 6; ++K)
        std::printf("exponential upper %d K %d: max %.3g ulp of the order's scale (y %.6g, mu %.6g), %.3g ulp of the value\n",
                    u, K, worst_scaled[u][K], worst_y[u][K], worst_mu[u][K], worst_own[u][K]);
}
