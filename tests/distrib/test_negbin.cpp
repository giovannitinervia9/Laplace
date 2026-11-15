// negbin1_distrib and negbin2_distrib (include/laplace/distrib/negbin1.hpp, negbin2.hpp): packing
// and descriptors, the log-mass against the direct lgamma form and against its normalization, the
// two families against each other and against the geometric at theta = 1, every derivative order
// against a 5-point difference of the order below, the expected information of the generic sum
// against the variance of the score (negbin2: against the closed I_mu_mu = 1/(mu + mu^2/theta) and
// I_mu_theta = 0) and its derivatives against differences, the cdf identities, the derivatives of
// log F and log(1 - F) of the generic summation route against differences, a kernel launch of the
// highest orders, and every order against mpmath (tools/reference/distrib_negbin.py).
#include <doctest/doctest.h>

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/geometric.hpp>
#include <laplace/distrib/negbin1.hpp>
#include <laplace/distrib/negbin2.hpp>
#include <Kokkos_Core.hpp>

#include "check_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// the log-mass written directly, as a difference of two lgamma: the twin of the Stirling form
double logpmf_direct(double r, double m, double y) {
  return std::lgamma(y + r) - std::lgamma(r) - std::lgamma(y + 1.0) + r * std::log(r / (r + m)) +
         y * std::log(m / (r + m));
}

// the expected information and its derivatives by the generic route (neither family has a member)
template <class D>
std::vector<double> ideriv_sum(const D& d, int K) {
  switch (K) {
    case 0: return as_vec<2, 2>(ld::distrib_expected_info<0>(d));
    case 1: return info_vec<2, 1>(ld::distrib_expected_info<1>(d));
    case 2: return info_vec<2, 2>(ld::distrib_expected_info<2>(d));
    case 3: return info_vec<2, 3>(ld::distrib_expected_info<3>(d));
    default: return info_vec<2, 4>(ld::distrib_expected_info<4>(d));
  }
}

// d^K I of the generic route against the difference of d^(K-1) I, every component
template <class D>
void check_info_derivs_sum(const std::vector<double>& th, int kmax, double rel) {
  const int npair = ld::sym_tensor_size(2, 2);
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = ideriv_sum(make<D>(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (int p = 0; p < npair; ++p)
      for (const auto& idx : sorted_indices(2, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ideriv_sum(make<D>(t), K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(2, K - 1);
        const double e = exact[p * ld::sym_tensor_size(2, K) + position_of(2, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + position_of(2, lower)];
        if (std::isnan(e) || std::isnan(a)) continue;
        INFO("K = " << K << ", pair " << p << ", mu " << th[0] << ", theta " << th[1]);
        CHECK(std::fabs(e - a) <= rel * scale);
      }
  }
}

// log F (upper: log(1 - F)) and its derivatives by the generic summation route
template <class D>
std::vector<double> cderiv_sum(const D& d, double y, int K, bool upper) {
  switch (K) {
    case 0: return {upper ? d.log_ccdf(y) : d.log_cdf(y)};
    case 1: return as_vec<2, 1>(upper ? ld::distrib_log_ccdf_dk<1>(d, y) : ld::distrib_log_cdf_dk<1>(d, y));
    case 2: return as_vec<2, 2>(upper ? ld::distrib_log_ccdf_dk<2>(d, y) : ld::distrib_log_cdf_dk<2>(d, y));
    case 3: return as_vec<2, 3>(upper ? ld::distrib_log_ccdf_dk<3>(d, y) : ld::distrib_log_cdf_dk<3>(d, y));
    case 4: return as_vec<2, 4>(upper ? ld::distrib_log_ccdf_dk<4>(d, y) : ld::distrib_log_cdf_dk<4>(d, y));
    case 5: return as_vec<2, 5>(upper ? ld::distrib_log_ccdf_dk<5>(d, y) : ld::distrib_log_cdf_dk<5>(d, y));
    default: return as_vec<2, 6>(upper ? ld::distrib_log_ccdf_dk<6>(d, y) : ld::distrib_log_cdf_dk<6>(d, y));
  }
}

template <class D>
void check_cdf_derivs_sum(const std::vector<double>& th, double y, bool upper, int kmax, double rel) {
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = cderiv_sum(make<D>(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(2, K)) {
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const int j = idx.back();
      const auto fd = diff5([&](const double* t) { return cderiv_sum(make<D>(t), y, K - 1, upper); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      INFO("K = " << K << ", component " << position_of(2, idx) << ", y = " << y << ", upper " << upper
                  << ", mu " << th[0] << ", theta " << th[1]);
      CHECK(std::fabs(exact[position_of(2, idx)] - fd[position_of(2, lower)]) <= rel * scale + 1e-12);
    }
  }
}

// E[l_a l_b] by summing over the support, pair-major
template <class D>
std::vector<double> score_variance(const D& d, int n) {
  std::vector<double> r(3, 0.0);
  for (int y = 0; y <= n; ++y) {
    const auto s = d.logpmf_d1(y);
    const double p = std::exp(d.logpmf(y));
    r[0] += p * s[0] * s[0];
    r[1] += p * s[0] * s[1];
    r[2] += p * s[1] * s[1];
  }
  return r;
}

}  // namespace

TEST_CASE("distrib negbin: descriptors") {
  CHECK(ld::negbin1_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::negbin2_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::negbin1_distrib::n_params == 2);
  CHECK(ld::negbin2_distrib::n_params == 2);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::negbin1_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::negbin2_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::negbin1_distrib::params[j].moves_support == false);
  }
  const ld::negbin2_distrib d2{4.0, 2.0};
  CHECK(d2.param_kinks(0, 1.25).count == 0);
  CHECK(d2.support_lower() == 0.0);
  CHECK(d2.support_upper() == ld::distrib_infinity);
  CHECK(d2.center() == 4.0);
  CHECK(d2.scale() == doctest::Approx(std::sqrt(4.0 + 8.0)).epsilon(1e-15));
  const ld::negbin1_distrib d1{4.0, 2.0};
  CHECK(d1.center() == 4.0);
  CHECK(d1.scale() == doctest::Approx(std::sqrt(12.0)).epsilon(1e-15));
  CHECK(d1.shape() == 2.0);
  CHECK(d1.success_prob() == doctest::Approx(1.0 / 3.0).epsilon(1e-15));
}

TEST_CASE("distrib negbin: log-mass against the direct form and its normalization") {
  struct pt { double mu, theta; };
  for (pt p : {pt{0.5, 0.5}, pt{3.5, 2.0}, pt{40.0, 0.01}, pt{40.0, 1e3}, pt{2.0, 1e-3}}) {
    const ld::negbin2_distrib d2{p.mu, p.theta};
    const ld::negbin1_distrib d1{p.mu, p.theta};
    double t2 = 0.0, t1 = 0.0;
    const int n = 40000;
    for (int y = 0; y < n; ++y) {
      if (y < 40) {
        // the direct form is the inaccurate side of this comparison: it subtracts two lgamma of
        // the size of y log y, so the bound is loose and the mpmath references are the real check
        INFO("mu " << p.mu << ", theta " << p.theta << ", y " << y);
        CHECK(d2.logpmf(y) == doctest::Approx(logpmf_direct(p.theta, p.mu, y)).epsilon(1e-10).scale(1.0));
        CHECK(d1.logpmf(y) == doctest::Approx(logpmf_direct(p.mu / p.theta, p.mu, y)).epsilon(1e-10).scale(1.0));
      }
      t2 += std::exp(d2.logpmf(y));
      t1 += std::exp(d1.logpmf(y));
    }
    // the mass left beyond the cut: the terms decay by the ratio mu / (theta + mu) for negbin2
    // and theta / (1 + theta) for negbin1, which at theta = 0.01 leaves 4e-5 of the mass at
    // y = 40000
    const double tail2 = std::exp(n * std::log(p.mu / (p.theta + p.mu)));
    const double tail1 = std::exp(n * std::log(p.theta / (1.0 + p.theta)));
    INFO("mu " << p.mu << ", theta " << p.theta);
    CHECK(t2 == doctest::Approx(1.0).epsilon(std::fmax(1e-9, 100.0 * tail2)));
    CHECK(t1 == doctest::Approx(1.0).epsilon(std::fmax(1e-9, 100.0 * tail1)));
  }
}

TEST_CASE("distrib negbin: the two parametrizations and the geometric agree where they must") {
  // the same mass: negbin2(mu, theta) is negbin1(mu, mu / theta) (both have shape theta and
  // success probability theta / (theta + mu)), and negbin2(mu, 1) is the geometric of mean mu
  for (double mu : {0.25, 3.5, 40.0, 1e4}) {
    for (double theta : {1e-3, 0.5, 2.0, 1e3, 1e7}) {
      const ld::negbin2_distrib a{mu, theta};
      const ld::negbin1_distrib b{mu, mu / theta};
      for (double y : {0.0, 1.0, 3.0, std::floor(mu), 4.0 * std::floor(mu) + 1.0}) {
        INFO("mu " << mu << ", theta " << theta << ", y " << y);
        CHECK(a.logpmf(y) == doctest::Approx(b.logpmf(y)).epsilon(1e-12).scale(1.0));
        // the two cumulative functions call the incomplete beta at the same shape but at a
        // success probability formed differently (theta / (theta + mu) against 1 / (1 + mu/theta)),
        // and I_p(theta, k + 1) varies by theta per unit of p: one ulp of p is 2e-9 of F at
        // theta = 1e7, so this side is conditioned by theta, not by either family's accuracy
        CHECK(a.cdf(y) == doctest::Approx(b.cdf(y)).epsilon(std::fmax(1e-12, 1e-15 * theta)).scale(1.0));
      }
    }
    const ld::negbin2_distrib g2{mu, 1.0};
    const ld::geometric_distrib g{mu};
    for (double y : {0.0, 1.0, 7.0, std::floor(2.0 * mu)}) {
      INFO("geometric, mu " << mu << ", y " << y);
      CHECK(g2.logpmf(y) == doctest::Approx(g.logpmf(y)).epsilon(1e-12).scale(1.0));
      CHECK(g2.logpmf_d1(y)[0] == doctest::Approx(g.logpmf_d1(y)[0]).epsilon(1e-10).scale(1e-12));
      CHECK(g2.logpmf_d4(y)[0] == doctest::Approx(g.logpmf_d4(y)[0]).epsilon(1e-9).scale(1e-12));
      CHECK(g2.log_cdf(y) == doctest::Approx(g.log_cdf(y)).epsilon(1e-11).scale(1.0));
      CHECK(g2.log_ccdf(y) == doctest::Approx(g.log_ccdf(y)).epsilon(1e-12).scale(1.0));
    }
  }
}

TEST_CASE("distrib negbin2: derivatives against differences") {
  for (double y : {0.0, 1.0, 4.0, 30.0}) check_derivs<ld::negbin2_distrib>({3.5, 2.0}, y, 1e-7);
  for (double y : {0.0, 2.0, 50.0}) check_derivs<ld::negbin2_distrib>({40.0, 0.01}, y, 1e-7);
  for (double y : {0.0, 40.0, 120.0}) check_derivs<ld::negbin2_distrib>({40.0, 1e3}, y, 1e-7);
  for (double y : {0.0, 2.0, 7.0}) check_derivs<ld::negbin2_distrib>({2.0, 1e-3}, y, 1e-7);
  // y = mu at the Poisson limit is left out: the first derivative in mu is then exactly 0 and so
  // is the whole order-1 tensor to 1e-12, while the 5-point difference's own truncation error
  // (h^4/30) d^5 l / dmu^5 is 8e-13, so the comparison would test the difference, not the tensor
  for (double y : {900.0, 1010.0, 1100.0}) check_derivs<ld::negbin2_distrib>({1e3, 1e6}, y, 1e-7);
}

TEST_CASE("distrib negbin1: derivatives against differences") {
  for (double y : {0.0, 1.0, 4.0, 30.0}) check_derivs<ld::negbin1_distrib>({3.5, 2.0}, y, 1e-7);
  for (double y : {0.0, 2.0, 50.0}) check_derivs<ld::negbin1_distrib>({40.0, 0.01}, y, 1e-7);
  for (double y : {0.0, 40.0, 400.0}) check_derivs<ld::negbin1_distrib>({40.0, 1e3}, y, 1e-7);
  // theta = 1e-6 is left to the mpmath references: the shared difference check steps by
  // 1e-3 max(1e-3, theta), which at theta = 1e-6 reaches a negative theta
  for (double y : {900.0, 1010.0, 1100.0}) check_derivs<ld::negbin1_distrib>({1e3, 1e-3}, y, 1e-7);
  for (double y : {2200.0, 2242.0, 2300.0}) check_derivs<ld::negbin1_distrib>({2242.0, 0.028}, y, 1e-7);
}

TEST_CASE("distrib negbin2: expected information, closed blocks and the variance of the score") {
  for (double mu : {0.5, 3.5, 40.0})
    for (double theta : {0.5, 2.0, 50.0}) {
      const ld::negbin2_distrib d{mu, theta};
      const auto I = ld::distrib_expected_info<0>(d);
      const double v = mu + mu * mu / theta;
      INFO("mu " << mu << ", theta " << theta);
      CHECK(I[0] == doctest::Approx(1.0 / v).epsilon(1e-10));  // I_mu_mu is closed
      CHECK(std::fabs(I[1]) <= 1e-10 * I[0]);                  // I_mu_theta is exactly 0
      const auto s = score_variance(d, int(60.0 * (mu + v)) + 2000);
      CHECK(I[2] == doctest::Approx(s[2]).epsilon(1e-8));
      CHECK(I[0] == doctest::Approx(s[0]).epsilon(1e-8));
    }
}

TEST_CASE("distrib negbin1: expected information against the variance of the score") {
  for (double mu : {0.5, 3.5, 40.0})
    for (double theta : {0.1, 1.0, 5.0}) {
      const ld::negbin1_distrib d{mu, theta};
      const auto I = ld::distrib_expected_info<0>(d);
      const auto s = score_variance(d, int(60.0 * mu * (1.0 + theta)) + 2000);
      INFO("mu " << mu << ", theta " << theta);
      for (int p = 0; p < 3; ++p) CHECK(I[p] == doctest::Approx(s[p]).epsilon(1e-8));
      // nothing is orthogonal here: the off-diagonal block is not zero
      CHECK(std::fabs(I[1]) > 1e-6 * std::sqrt(I[0] * I[2]));
    }
}

TEST_CASE("distrib negbin: derivatives of the expected information against differences") {
  check_info_derivs_sum<ld::negbin2_distrib>({3.5, 2.0}, 3, 1e-6);
  check_info_derivs_sum<ld::negbin2_distrib>({0.5, 0.5}, 3, 1e-6);
  check_info_derivs_sum<ld::negbin1_distrib>({3.5, 2.0}, 3, 1e-6);
  check_info_derivs_sum<ld::negbin1_distrib>({0.5, 0.5}, 3, 1e-6);
}

TEST_CASE("distrib negbin: cdf identities") {
  for (double theta : {1e-2, 0.5, 3.0, 1e4}) {
    const ld::negbin2_distrib d2{3.5, theta};
    const ld::negbin1_distrib d1{3.5, theta};
    double s2 = 0.0, s1 = 0.0;
    for (int k = 0; k < 25; ++k) {
      s2 += std::exp(d2.logpmf(k));
      s1 += std::exp(d1.logpmf(k));
      INFO("theta " << theta << ", k " << k);
      CHECK(d2.cdf(k) == doctest::Approx(s2).epsilon(1e-12));
      CHECK(d1.cdf(k) == doctest::Approx(s1).epsilon(1e-12));
      CHECK(d2.cdf(k) + d2.ccdf(k) == doctest::Approx(1.0).epsilon(1e-14));
      CHECK(d1.cdf(k) + d1.ccdf(k) == doctest::Approx(1.0).epsilon(1e-14));
      CHECK(d2.log_cdf(k) == doctest::Approx(std::log(d2.cdf(k))).epsilon(1e-12));
      CHECK(d2.log_ccdf(k) == doctest::Approx(std::log(d2.ccdf(k))).epsilon(1e-12));
      CHECK(d2.cdf(k + 0.5) == d2.cdf(double(k)));
    }
    CHECK(d2.cdf(-1.0) == 0.0);
    CHECK(d2.ccdf(-1.0) == 1.0);
    CHECK(d2.log_cdf(-1.0) == -ld::distrib_infinity);
    CHECK(d2.log_ccdf(-1.0) == 0.0);
    CHECK(d1.log_cdf(-0.5) == -ld::distrib_infinity);
  }
}

TEST_CASE("distrib negbin: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {0.0, 2.0, 6.0, 20.0}) check_cdf_derivs_sum<ld::negbin2_distrib>({3.5, 2.0}, y, upper, 3, 1e-5);
    for (double y : {0.0, 2.0, 6.0, 20.0}) check_cdf_derivs_sum<ld::negbin1_distrib>({3.5, 2.0}, y, upper, 3, 1e-5);
    check_cdf_derivs_sum<ld::negbin2_distrib>({40.0, 1e3}, 35.0, upper, 2, 1e-5);
    check_cdf_derivs_sum<ld::negbin1_distrib>({40.0, 0.01}, 35.0, upper, 2, 1e-5);
  }
}

// The kernels live in named functions with external linkage, not in the body of a TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
double negbin_kernel_value(double y) {
  const ld::negbin2_distrib a{3.7, 2.5};
  const ld::negbin1_distrib b{3.7, 2.5};
  return a.logpmf(y) + a.logpmf_d6(y)[3] + a.log_cdf(y) + a.log_ccdf(y) +
         ld::distrib_log_cdf_dk<2>(a, y)[1] + b.logpmf(y) + b.logpmf_d6(y)[3] + b.log_cdf(y) +
         ld::distrib_expected_info<0>(b)[1];
}

void negbin_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_negbin", n, KOKKOS_LAMBDA(const int i) {
        const double y = double(i % 20);
        const ld::negbin2_distrib a{3.7, 2.5};
        const ld::negbin1_distrib b{3.7, 2.5};
        out(i) = a.logpmf(y) + a.logpmf_d6(y)[3] + a.log_cdf(y) + a.log_ccdf(y) +
                 ld::distrib_log_cdf_dk<2>(a, y)[1] + b.logpmf(y) + b.logpmf_d6(y)[3] + b.log_cdf(y) +
                 ld::distrib_expected_info<0>(b)[1];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double r = negbin_kernel_value(double(i % 20));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-11));
  }
}

TEST_CASE("distrib negbin: every order inside a kernel") { negbin_kernel_case(); }

TEST_CASE("distrib negbin: every order against mpmath") {
  // tools/reference/distrib_negbin.py: fam 1 negbin1, 2 negbin2; kind 0 log p, 1 log F, 2 log(1 - F);
  // b of the K derivatives taken in theta, K - b in mu. The error is the library value minus the
  // mpmath value (low part removed), in ulp of the largest component of the same
  // (fam, kind, y, mu, theta, K) group: every tensor of a given order has genuine zeros in y and in
  // the parameters (the score in the shape vanishes near (y - mu)^2 = y at the Poisson limit), so an
  // error is measured against the scale of the order, as for chisq and the geometric.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_negbin.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int fam, kind, K, b; double y, mu, theta, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], mus[64], ths[64], his[64], los[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", &r.fam, &r.kind, ys, mus, ths, &r.K,
                    &r.b, his, los) != 9)
      continue;
    r.y = std::strtod(ys, nullptr), r.mu = std::strtod(mus, nullptr), r.theta = std::strtod(ths, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 4500);
  std::map<std::tuple<int, int, double, double, double, int>, double> scale_of;
  for (const auto& r : rows) {
    auto& s = scale_of[std::make_tuple(r.fam, r.kind, r.y, r.mu, r.theta, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[3][3][7] = {};
  double worst_y[3][3][7] = {}, worst_mu[3][3][7] = {}, worst_th[3][3][7] = {};
  int worst_b[3][3][7] = {};
  // bounds: 1.5 times the worst measured on the gcc (Serial + OpenMP) and the CUDA build
  // (2026-11-15, LAPLACE_PRINT_ERRORS), per family and kind
  const double bound[3][3][7] = {{{}, {}, {}},
                                 {{62.0, 430.0, 37.0, 60.0, 4600.0, 660.0, 2000.0},
                                  {8.1e5, 62.0, 59.0, 2.5e4, 4.0e5, 1.8e7, 1.4e9},
                                  {410.0, 43.0, 43.0, 38.0, 38.0, 82.0, 164.0}},
                                 {{28.0, 90.0, 2340.0, 23.0, 9.1, 7.8, 13.5},
                                  {450.0, 26.0, 5640.0, 800.0, 1460.0, 8700.0, 1.2e4},
                                  {310.0, 69.0, 108.0, 254.0, 342.0, 780.0, 548.0}}};
  for (const auto& r : rows) {
    double v;
    if (r.fam == 2) {
      const ld::negbin2_distrib d{r.mu, r.theta};
      v = r.kind == 0 ? (r.K == 0 ? d.logpmf(r.y) : lderiv(d, r.y, r.K)[r.b])
                      : cderiv_sum(d, r.y, r.K, r.kind == 2)[r.b];
    } else {
      const ld::negbin1_distrib d{r.mu, r.theta};
      v = r.kind == 0 ? (r.K == 0 ? d.logpmf(r.y) : lderiv(d, r.y, r.K)[r.b])
                      : cderiv_sum(d, r.y, r.K, r.kind == 2)[r.b];
    }
    const double sc = scale_of[std::make_tuple(r.fam, r.kind, r.y, r.mu, r.theta, r.K)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    if (ulp > worst[r.fam][r.kind][r.K])
      worst_y[r.fam][r.kind][r.K] = r.y, worst_mu[r.fam][r.kind][r.K] = r.mu,
      worst_th[r.fam][r.kind][r.K] = r.theta, worst_b[r.fam][r.kind][r.K] = r.b;
    worst[r.fam][r.kind][r.K] = std::fmax(worst[r.fam][r.kind][r.K], ulp);
    INFO("fam " << r.fam << " kind " << r.kind << " y " << r.y << " mu " << r.mu << " theta " << r.theta << " K "
                << r.K << " b " << r.b << " ulp " << ulp);
    CHECK(ulp <= bound[r.fam][r.kind][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int fam = 1; fam <= 2; ++fam)
      for (int kd = 0; kd < 3; ++kd)
        for (int K = 0; K <= 6; ++K)
          std::printf("negbin%d kind %d K %d: max %.3g ulp (y %.6g, mu %.6g, theta %.6g, b %d)\n", fam, kd, K,
                      worst[fam][kd][K], worst_y[fam][kd][K], worst_mu[fam][kd][K], worst_th[fam][kd][K],
                      worst_b[fam][kd][K]);
}
