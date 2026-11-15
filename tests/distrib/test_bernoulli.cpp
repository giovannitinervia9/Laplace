// bernoulli_distrib (mean mu; include/laplace/distrib/bernoulli.hpp) against its mpmath references
// (tools/reference/distrib_bernoulli.py, mu from 1e-15 to 1 - 1e-15). Checks: packing and descriptors,
// closed values, every derivative order of the log-mass against a 5-point difference of the order below,
// the same for the expected information and for log F and log(1 - F), the expected information against
// the variance of the score, the generic route of log F (generic_cdf.hpp) against the closed form, the
// cdf pieces and the quantile, the mpmath references in ulp, and a kernel launch of the highest orders.
//
// The finite differences step in mu by 1e-3 min(mu, 1 - mu), so that the stencil stays in (0, 1); the
// references at mu near 0 and near 1 are checked against mpmath only, since a difference in mu cannot
// resolve a step of 1e-18 next to 1 in double.
#include <doctest/doctest.h>

#include <laplace/distrib/bernoulli.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

using bern = ld::bernoulli_distrib;

// the difference of g in mu, step 1e-3 min(mu, 1 - mu)
template <class G>
std::vector<double> fd_mu(const G& g, double mu) {
  return diff5(g, std::vector<double>{mu}, 0, 1e-3 * std::fmin(mu, 1.0 - mu));
}

// d^K l / dmu^K at y in {0, 1} against the difference of d^(K-1) l; the unit is the value itself
void check_logpmf_derivs(double mu, double rel) {
  for (double y : {0.0, 1.0})
    for (int K = 1; K <= 6; ++K) {
      const auto exact = lderiv(bern{mu}, y, K);
      const auto fd = fd_mu([&](const double* t) { return lderiv(make<bern>(t), y, K - 1); }, mu);
      INFO("logpmf: K = " << K << ", y = " << y << ", mu = " << mu);
      CHECK(std::fabs(exact[0] - fd[0]) <= rel * std::fabs(exact[0]));
    }
}

// d^K I / dmu^K against the difference of d^(K-1) I; the unit is the sum of the magnitudes of the two
// terms (-1)^K K! / mu^(K+1) and K! / (1 - mu)^(K+1), since the odd orders vanish at mu = 1/2
void check_info_derivs(double mu, double rel) {
  for (int K = 1; K <= 4; ++K) {
    const auto exact = ideriv(bern{mu}, K);
    const double unit = std::tgamma(K + 1.0) * (std::pow(mu, -(K + 1)) + std::pow(1.0 - mu, -(K + 1)));
    const auto fd = fd_mu([&](const double* t) { return ideriv(make<bern>(t), K - 1); }, mu);
    INFO("info: K = " << K << ", mu = " << mu);
    CHECK(std::fabs(exact[0] - fd[0]) <= rel * unit);
  }
}

// d^K log F (upper = false) or log(1 - F) (upper = true) at y in [0, 1), against differences in mu
void check_cdf_derivs(double mu, double y, bool upper, double rel) {
  for (int K = 1; K <= 6; ++K) {
    const auto exact = cderiv(bern{mu}, y, K, upper);
    const auto fd = fd_mu([&](const double* t) { return cderiv(make<bern>(t), y, K - 1, upper); }, mu);
    INFO("cdf: K = " << K << ", y = " << y << ", upper " << upper << ", mu = " << mu);
    CHECK(std::fabs(exact[0] - fd[0]) <= rel * std::fabs(exact[0]) + 1e-12);
  }
}

// the closed form of d^K log F against the generic route that sums the mass
template <int K>
void check_generic_order(double mu) {
  const bern d{mu};
  const double y = 0.5;
  const double lib = cderiv(d, y, K, false)[0], gen = ld::log_cdf_dk_by_sum<K>(d, y)[0];
  const double libu = cderiv(d, y, K, true)[0], genu = ld::log_ccdf_dk_by_sum<K>(d, y)[0];
  INFO("generic: K = " << K << ", mu = " << mu);
  CHECK(std::fabs(lib - gen) <= 1e-10 * std::fabs(lib));
  // the upper tail sums the mass above k and loses accuracy at mu near 1: measured 7e-7 relative at K = 6,
  // mu = 0.99 (2026-11-15, gcc build), so its bound is 1e-5
  CHECK(std::fabs(libu - genu) <= 1e-5 * std::fabs(libu));
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
void bernoulli_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_bernoulli", n, KOKKOS_LAMBDA(const int i) {
        const double mu = 0.01 + 0.0126 * i;
        const double y = i % 2;
        const ld::bernoulli_distrib b{mu};
        out(i) = b.logpmf_d6(y)[0] + b.expected_info_d4().pair[0][0] + b.log_cdf_d6(0.5)[0] +
                 b.log_ccdf_d6(0.5)[0] + b.log_cdf(0.5) + b.log_ccdf(0.5) + b.quantile(0.3) + b.logpmf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double mu = 0.01 + 0.0126 * i;
    const double y = i % 2;
    const bern b{mu};
    const double r = b.logpmf_d6(y)[0] + b.expected_info_d4().pair[0][0] + b.log_cdf_d6(0.5)[0] +
                     b.log_ccdf_d6(0.5)[0] + b.log_cdf(0.5) + b.log_ccdf(0.5) + b.quantile(0.3) + b.logpmf(y);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-13));
  }
}

// the grid of the derivative checks: the middle of the interval and the points 0.01 from each end
const double kMu[] = {0.01, 0.3, 0.5, 0.7, 0.99};

}  // namespace

TEST_CASE("distrib bernoulli: descriptors") {
  CHECK(bern::kind == ld::distrib_kind::discrete);
  CHECK(bern::n_params == 1);
  CHECK(bern::params[0].regularity == ld::infinitely_smooth);
  CHECK(bern::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(bern::params[0].moves_support == false);
  const bern d{0.3};
  CHECK(d.param_kinks(0, 1.0).count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == 1.0);
  CHECK(d.center() == 0.3);
  CHECK(d.scale() == doctest::Approx(std::sqrt(0.21)).epsilon(1e-15));
}

TEST_CASE("distrib bernoulli: closed values of the mass and the information") {
  const bern d{0.3};
  CHECK(d.logpmf(1.0) == doctest::Approx(std::log(0.3)).epsilon(1e-15));
  CHECK(d.logpmf(0.0) == doctest::Approx(std::log(0.7)).epsilon(1e-15));
  CHECK(d.logpmf(2.0) == -ld::distrib_infinity);
  CHECK(bern{0.5}.logpmf(0.0) == doctest::Approx(std::log(0.5)).epsilon(1e-15));
  CHECK(d.expected_info()[0] == doctest::Approx(1.0 / 0.21).epsilon(1e-15));
  // mu = 1e-15: log1p keeps log(1 - mu) at -mu to full precision
  CHECK(bern{1e-15}.logpmf(0.0) == doctest::Approx(-1e-15).epsilon(1e-14));
  CHECK(bern{1.0 - 1e-15}.logpmf(1.0) == doctest::Approx(-1e-15).epsilon(1e-14));
}

TEST_CASE("distrib bernoulli: log-mass derivatives against differences") {
  for (double mu : kMu) check_logpmf_derivs(mu, 1e-8);
  // closed values at mu = 1/2 (the differences above check every order at the other mu)
  CHECK(bern{0.5}.logpmf_d1(1.0)[0] == doctest::Approx(2.0).epsilon(1e-15));
  CHECK(bern{0.5}.logpmf_d1(0.0)[0] == doctest::Approx(-2.0).epsilon(1e-15));
  CHECK(bern{0.5}.logpmf_d2(0.0)[0] == doctest::Approx(-4.0).epsilon(1e-15));
  CHECK(bern{0.5}.logpmf_d2(1.0)[0] == doctest::Approx(-4.0).epsilon(1e-15));
}

TEST_CASE("distrib bernoulli: derivatives of the expected information against differences") {
  for (double mu : kMu) check_info_derivs(mu, 1e-8);
  // the odd orders vanish exactly at mu = 1/2
  const auto d1 = bern{0.5}.expected_info_d1();
  const auto d3 = bern{0.5}.expected_info_d3();
  CHECK(d1.pair[0][0] == 0.0);
  CHECK(d3.pair[0][0] == 0.0);
}

TEST_CASE("distrib bernoulli: expected information against the variance of the score") {
  for (double mu : {0.01, 0.3, 0.5, 0.7, 0.99}) {
    const bern d{mu};
    double s = 0.0;
    for (double y : {0.0, 1.0}) {
      const double g = d.logpmf_d1(y)[0];
      s += g * g * std::exp(d.logpmf(y));
    }
    INFO("mu = " << mu);
    CHECK(std::fabs(s - d.expected_info()[0]) <= 1e-13 * d.expected_info()[0]);
  }
}

TEST_CASE("distrib bernoulli: derivatives of log F and log(1 - F) against differences") {
  for (double mu : kMu)
    for (bool upper : {false, true}) {
      check_cdf_derivs(mu, 0.0, upper, 1e-8);
      check_cdf_derivs(mu, 0.5, upper, 1e-8);
    }
}

TEST_CASE("distrib bernoulli: generic route for log F against the closed form") {
  for (double mu : {0.01, 0.3, 0.7, 0.99}) {
    check_generic_order<1>(mu);
    check_generic_order<2>(mu);
    check_generic_order<3>(mu);
    check_generic_order<4>(mu);
    check_generic_order<5>(mu);
    check_generic_order<6>(mu);
  }
}

TEST_CASE("distrib bernoulli: cdf pieces, complements and log values") {
  const bern d{0.3};
  CHECK(d.cdf(-1.0) == 0.0);
  CHECK(d.cdf(0.0) == doctest::Approx(0.7).epsilon(1e-15));
  CHECK(d.cdf(0.5) == doctest::Approx(0.7).epsilon(1e-15));
  CHECK(d.cdf(1.0) == 1.0);
  CHECK(d.cdf(3.0) == 1.0);
  CHECK(d.ccdf(-1.0) == 1.0);
  CHECK(d.ccdf(0.5) == doctest::Approx(0.3).epsilon(1e-15));
  CHECK(d.ccdf(1.0) == 0.0);
  for (double y : {-1.0, 0.0, 0.5, 1.0, 2.0}) CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
  CHECK(d.log_cdf(-1.0) == -ld::distrib_infinity);
  CHECK(d.log_cdf(0.5) == doctest::Approx(std::log(0.7)).epsilon(1e-15));
  CHECK(d.log_cdf(1.0) == 0.0);
  CHECK(d.log_ccdf(-1.0) == 0.0);
  CHECK(d.log_ccdf(0.5) == doctest::Approx(std::log(0.3)).epsilon(1e-15));
  CHECK(d.log_ccdf(1.0) == -ld::distrib_infinity);
  // the derivatives off the open piece: zeros where log F or log(1 - F) is constant, NaN where it is -inf
  CHECK(d.log_cdf_d3(1.0)[0] == 0.0);
  CHECK(std::isnan(d.log_cdf_d3(-1.0)[0]));
  CHECK(d.log_ccdf_d2(-1.0)[0] == 0.0);
  CHECK(std::isnan(d.log_ccdf_d2(1.0)[0]));
  // the derivatives of log F at y in [0, 1) are -(K-1)! / (1 - mu)^K
  CHECK(d.log_cdf_d1(0.5)[0] == doctest::Approx(-1.0 / 0.7).epsilon(1e-15));
  CHECK(d.log_cdf_d2(0.5)[0] == doctest::Approx(-1.0 / 0.49).epsilon(1e-15));
  CHECK(d.log_ccdf_d2(0.5)[0] == doctest::Approx(-1.0 / 0.09).epsilon(1e-15));
}

TEST_CASE("distrib bernoulli: quantile pieces") {
  const bern d{0.3};
  CHECK(d.quantile(0.6) == 0.0);
  CHECK(d.quantile(0.8) == 1.0);
  // F(0) = 1 - mu is the threshold: its own cdf value maps back to 0
  CHECK(d.quantile(d.cdf(0.0)) == 0.0);
  CHECK(d.quantile_from_log(std::log(0.6)) == 0.0);
  CHECK(d.quantile_from_log(std::log(0.8)) == 1.0);
  CHECK(d.quantile_from_log_upper(std::log(0.4)) == 0.0);
  CHECK(d.quantile_from_log_upper(std::log(0.2)) == 1.0);
}

TEST_CASE("distrib bernoulli: every order inside a kernel") { bernoulli_kernel_case(); }

// The library value of one row, inside a kernel: quantity q (0 log-mass, 1 information, 2 log F,
// 3 log(1 - F)) at order K, at observation y. The kernel runs on the device on the CUDA build.
template <int K>
LAPLACE_HD double bernoulli_value_at_order(const ld::bernoulli_distrib& b, int q, double y) {
  if (q == 0) return b.logpmf_dk<K>(y)[0];
  if (q == 1) return b.expected_info_dk<K>().pair[0][0];
  if (q == 2) return b.log_cdf_dk<K>(y)[0];
  return b.log_ccdf_dk<K>(y)[0];
}

LAPLACE_HD double bernoulli_value(const ld::bernoulli_distrib& b, int q, int K, double y) {
  switch (K) {
    case 1: return bernoulli_value_at_order<1>(b, q, y);
    case 2: return bernoulli_value_at_order<2>(b, q, y);
    case 3: return bernoulli_value_at_order<3>(b, q, y);
    case 4: return bernoulli_value_at_order<4>(b, q, y);
    case 5: return bernoulli_value_at_order<5>(b, q, y);
    default: return bernoulli_value_at_order<6>(b, q, y);
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
void bernoulli_reference_values(const std::vector<int>& q, const std::vector<int>& K, const std::vector<double>& y,
                                const std::vector<double>& mu, std::vector<double>& out) {
  const int n = static_cast<int>(q.size());
  Kokkos::View<int*> qv("q", n), kv("K", n);
  Kokkos::View<double*> yv("y", n), mv("mu", n), ov("out", n);
  auto hq = Kokkos::create_mirror_view(qv);
  auto hk = Kokkos::create_mirror_view(kv);
  auto hy = Kokkos::create_mirror_view(yv);
  auto hm = Kokkos::create_mirror_view(mv);
  for (int i = 0; i < n; ++i) {
    hq(i) = q[i];
    hk(i) = K[i];
    hy(i) = y[i];
    hm(i) = mu[i];
  }
  Kokkos::deep_copy(qv, hq);
  Kokkos::deep_copy(kv, hk);
  Kokkos::deep_copy(yv, hy);
  Kokkos::deep_copy(mv, hm);
  Kokkos::parallel_for(
      "distrib_bernoulli_reference", n, KOKKOS_LAMBDA(const int i) {
        ov(i) = bernoulli_value(ld::bernoulli_distrib{mv(i)}, qv(i), kv(i), yv(i));
      });
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ov);
  out.assign(ho.data(), ho.data() + n);
}

TEST_CASE("distrib bernoulli: every quantity against mpmath") {
  // tools/reference/distrib_bernoulli.py (mu from 1e-15 to 1 - 1e-15). The error of each value is the
  // library value minus the mpmath value (its low part removed), in ulp of the unit of the value: the
  // magnitude of its term(s), so that the zeros of the odd orders of info at mu = 1/2 are scored on the
  // two terms. The library values are computed inside one kernel (on the device on the CUDA build).
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_bernoulli.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  std::vector<int> qs, Ks;
  std::vector<double> ys, his, los, mus;
  while (std::fgets(line, sizeof line, f)) {
    char what[32], yc[64], hc[64], lc[64], mc[64];
    int K;
    if (std::sscanf(line, "%31[^,],%63[^,],%d,%63[^,],%63[^,],%63s", what, yc, &K, hc, lc, mc) != 6) continue;
    const std::string w = what;
    qs.push_back(w == "logpmf" ? 0 : w == "info" ? 1 : w == "logcdf" ? 2 : 3);
    Ks.push_back(K);
    ys.push_back(std::strtod(yc, nullptr));
    his.push_back(std::strtod(hc, nullptr));
    los.push_back(std::strtod(lc, nullptr));
    mus.push_back(std::strtod(mc, nullptr));
  }
  std::fclose(f);
  CHECK(qs.size() > 300);
  std::vector<double> lib;
  bernoulli_reference_values(qs, Ks, ys, mus, lib);
  // bounds: 1.5 times the largest error measured on the gcc build and on the CUDA build (2026-11-15)
  // logpmf: 0.5, 1, 1.5, 2.85, 2.68, 4.87 ulp; info: 1.13, 2.25, 1.28, 1.59;
  // log F: 0.496, 0.993, 1.49, 2.85, 2.68, 4.87; log(1 - F): 0.5, 1, 1.5, 1, 1.25, 2.63
  const double bound[4][7] = {{0.0, 0.75, 1.5, 2.25, 4.3, 4.0, 7.3},
                              {0.0, 1.7, 3.4, 1.9, 2.4, 0.0, 0.0},
                              {0.0, 0.75, 1.5, 2.25, 4.3, 4.0, 7.3},
                              {0.0, 0.75, 1.5, 2.25, 1.5, 1.9, 3.9}};
  double worst[4][7] = {}, worst_mu[4][7] = {};
  for (std::size_t i = 0; i < qs.size(); ++i) {
    const int q = qs[i], K = Ks[i];
    const double mu = mus[i];
    const double unit = q == 1 ? std::tgamma(K + 1.0) * (std::pow(mu, -(K + 1)) + std::pow(1.0 - mu, -(K + 1)))
                               : std::fabs(his[i]);
    const double err = std::fabs((lib[i] - his[i]) - los[i]);
    const double ulps = err / std::ldexp(1.0, std::ilogb(unit) - 52);
    INFO("q " << q << " K " << K << " y " << ys[i] << " mu " << mu << " ulp " << ulps);
    if (ulps > worst[q][K]) worst_mu[q][K] = mu;
    worst[q][K] = std::fmax(worst[q][K], ulps);
    CHECK(ulps <= bound[q][K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int q = 0; q < 4; ++q)
      for (int K = 1; K <= 6; ++K)
        if (worst[q][K] > 0.0)
          std::printf("bernoulli q %d K %d: max %.3g ulp of the unit (mu %.6g)\n", q, K, worst[q][K], worst_mu[q][K]);
}
