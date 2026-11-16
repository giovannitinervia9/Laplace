// betabinom_distrib (mean proportion mu, dispersion sigma, known size n;
// include/laplace/distrib/betabinom.hpp): descriptors and the trailing size, the mass against the
// direct difference of log-betas and against its normalization, the binomial limit sigma -> 0, every
// derivative order against a 5-point difference of the order below, the expected information of the
// generic support sum against the variance of the score and its derivatives against differences, the
// cdf of the generic sum against the partial sums of the mass, the derivatives of log F and
// log(1 - F) against differences, a kernel launch of the highest orders, and every order against
// mpmath (tools/reference/distrib_betabinom.py).
#include <doctest/doctest.h>

#include <laplace/distrib/betabinom.hpp>
#include <laplace/distrib/binomial.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <Kokkos_Core.hpp>

#include "check_common.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

// check_common builds a two-parameter distribution as D{a, b}; the beta-binomial also needs its size
namespace laplace_distrib_check {
inline double betabinom_test_size = 10.0;
template <>
inline ld::betabinom_distrib make<ld::betabinom_distrib>(const double* t) {
  return ld::betabinom_distrib{t[0], t[1], betabinom_test_size};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// the log-mass written directly as a difference of two log-betas: the twin of the shift sums
double logpmf_direct(double mu, double sigma, double n, double y) {
  const double a = mu / sigma, b = (1.0 - mu) / sigma;
  const auto lbeta = [](double u, double v) { return std::lgamma(u) + std::lgamma(v) - std::lgamma(u + v); };
  return std::lgamma(n + 1.0) - std::lgamma(y + 1.0) - std::lgamma(n - y + 1.0) + lbeta(y + a, n - y + b) -
         lbeta(a, b);
}

// the expected information and its derivatives by the generic route (the family has no member)
std::vector<double> ideriv_sum(const ld::betabinom_distrib& d, int K) {
  switch (K) {
    case 0: return as_vec<2, 2>(ld::distrib_expected_info<0>(d));
    case 1: return info_vec<2, 1>(ld::distrib_expected_info<1>(d));
    case 2: return info_vec<2, 2>(ld::distrib_expected_info<2>(d));
    case 3: return info_vec<2, 3>(ld::distrib_expected_info<3>(d));
    default: return info_vec<2, 4>(ld::distrib_expected_info<4>(d));
  }
}

void check_info_derivs_sum(const std::vector<double>& th, int kmax, double rel) {
  const int npair = ld::sym_tensor_size(2, 2);
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = ideriv_sum(make<ld::betabinom_distrib>(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (int p = 0; p < npair; ++p)
      for (const auto& idx : sorted_indices(2, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ideriv_sum(make<ld::betabinom_distrib>(t), K - 1); }, th,
                              j, 1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(2, K - 1);
        const double e = exact[p * ld::sym_tensor_size(2, K) + position_of(2, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + position_of(2, lower)];
        if (std::isnan(e) || std::isnan(a)) continue;
        INFO("K = " << K << ", pair " << p << ", mu " << th[0] << ", sigma " << th[1]);
        CHECK(std::fabs(e - a) <= rel * scale);
      }
  }
}

// log F (upper: log(1 - F)) and its derivatives by the generic summation route
std::vector<double> cderiv_sum(const ld::betabinom_distrib& d, double y, int K, bool upper) {
  switch (K) {
    case 0: return {upper ? ld::distrib_log_ccdf(d, y) : ld::distrib_log_cdf(d, y)};
    case 1: return as_vec<2, 1>(upper ? ld::distrib_log_ccdf_dk<1>(d, y) : ld::distrib_log_cdf_dk<1>(d, y));
    case 2: return as_vec<2, 2>(upper ? ld::distrib_log_ccdf_dk<2>(d, y) : ld::distrib_log_cdf_dk<2>(d, y));
    case 3: return as_vec<2, 3>(upper ? ld::distrib_log_ccdf_dk<3>(d, y) : ld::distrib_log_cdf_dk<3>(d, y));
    case 4: return as_vec<2, 4>(upper ? ld::distrib_log_ccdf_dk<4>(d, y) : ld::distrib_log_cdf_dk<4>(d, y));
    case 5: return as_vec<2, 5>(upper ? ld::distrib_log_ccdf_dk<5>(d, y) : ld::distrib_log_cdf_dk<5>(d, y));
    default: return as_vec<2, 6>(upper ? ld::distrib_log_ccdf_dk<6>(d, y) : ld::distrib_log_cdf_dk<6>(d, y));
  }
}

void check_cdf_derivs_sum(const std::vector<double>& th, double y, bool upper, int kmax, double rel) {
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = cderiv_sum(make<ld::betabinom_distrib>(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(2, K)) {
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const int j = idx.back();
      const auto fd = diff5([&](const double* t) { return cderiv_sum(make<ld::betabinom_distrib>(t), y, K - 1, upper); },
                            th, j, 1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      INFO("K = " << K << ", component " << position_of(2, idx) << ", y = " << y << ", upper " << upper << ", mu "
                  << th[0] << ", sigma " << th[1]);
      CHECK(std::fabs(exact[position_of(2, idx)] - fd[position_of(2, lower)]) <= rel * scale + 1e-12);
    }
  }
}

// E[l_a l_b] by summing over the finite support, pair-major
std::vector<double> score_variance(const ld::betabinom_distrib& d) {
  std::vector<double> r(3, 0.0);
  for (int y = 0; y <= (int)d.size; ++y) {
    const auto s = d.logpmf_d1(y);
    const double p = std::exp(d.logpmf(y));
    r[0] += p * s[0] * s[0];
    r[1] += p * s[0] * s[1];
    r[2] += p * s[1] * s[1];
  }
  return r;
}

}  // namespace

TEST_CASE("distrib betabinom: descriptors, size and support") {
  CHECK(ld::betabinom_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::betabinom_distrib::n_params == 2);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::betabinom_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::betabinom_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::betabinom_distrib::params[j].moves_support == false);
  }
  CHECK(std::string(ld::betabinom_distrib::params[0].name) == "mu");
  CHECK(std::string(ld::betabinom_distrib::params[1].name) == "sigma");
  const ld::betabinom_distrib d{0.25, 0.5, 20.0};
  CHECK(d.param_kinks(0, 3.0).count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == 20.0);
  CHECK(d.center() == 5.0);
  CHECK(d.scale() == doctest::Approx(std::sqrt(20.0 * 0.25 * 0.75 * (1.0 + 19.0 * 0.5 / 1.5))).epsilon(1e-15));
  // outside the support, and at a non-integer
  CHECK(d.logpmf(-1.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(21.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(3.5) == -ld::distrib_infinity);
}

TEST_CASE("distrib betabinom: the mass against the direct log-betas and its normalization") {
  struct pt { double mu, sigma, n; };
  for (pt p : {pt{0.5, 0.5, 10.0}, pt{0.3, 2.0, 5.0}, pt{0.02, 0.25, 50.0}, pt{0.98, 0.25, 50.0},
               pt{0.5, 1e3, 7.0}, pt{0.4, 0.05, 200.0}, pt{0.7, 1.0, 1.0}}) {
    const ld::betabinom_distrib d{p.mu, p.sigma, p.n};
    double total = 0.0;
    for (int y = 0; y <= (int)p.n; ++y) {
      INFO("mu " << p.mu << ", sigma " << p.sigma << ", n " << p.n << ", y " << y);
      // the direct form subtracts two log-betas of the size of (a + b) log(a + b), so it is the
      // inaccurate side of this comparison; the mpmath references are the real check
      CHECK(d.logpmf(y) == doctest::Approx(logpmf_direct(p.mu, p.sigma, p.n, y)).epsilon(1e-9).scale(1.0));
      total += std::exp(d.logpmf(y));
    }
    INFO("mu " << p.mu << ", sigma " << p.sigma << ", n " << p.n);
    CHECK(total == doctest::Approx(1.0).epsilon(1e-12));
  }
}

TEST_CASE("distrib betabinom: the binomial limit sigma -> 0") {
  // at sigma = 0 the three sums vanish and the mass is exactly the binomial's. The approach is
  // linear in sigma and its coefficient grows with the size and with 1/min(mu, 1 - mu): the sums
  // depart by about sigma sum_(i<y) i / mu, so the bounds below are 20 sigma n^2 / q for the value
  // and one power of 1/q more for the first derivative, with q = min(mu, 1 - mu). What is being
  // checked is that the family reaches the binomial at all, without a limit or a series.
  for (double mu : {0.1, 0.5, 0.9})
    for (double n : {1.0, 10.0, 200.0}) {
      const ld::binomial_distrib b{mu, n};
      const double q = std::fmin(mu, 1.0 - mu);
      for (double sigma : {1e-12, 1e-8}) {
        const ld::betabinom_distrib d{mu, sigma, n};
        const double tol = 20.0 * sigma * n * n / q;
        for (double y : {0.0, 1.0, std::floor(n * mu), n}) {
          INFO("mu " << mu << ", n " << n << ", sigma " << sigma << ", y " << y);
          CHECK(std::fabs(d.logpmf(y) - b.logpmf(y)) <= tol + 1e-13);
          // the derivatives in mu tend to the binomial's as well
          CHECK(std::fabs(d.logpmf_d1(y)[0] - b.logpmf_d1(y)[0]) <= tol / q + 1e-11);
        }
      }
    }
}

TEST_CASE("distrib betabinom: the two routes agree where both hold") {
  // The sums over the integer shifts and the form of bounded cost through Stirling's error are one
  // identity, so where both are accurate they must agree to a few ulp. Below n = 50 the sums are
  // the accurate side (1.7 to 63 ulp against the references) and above it they are not, so the
  // comparison is read as the Stirling route's own check at the small sizes and as the sums' loss
  // at the large ones.
  namespace db = ld::detail_betabinom;
  struct pt { double mu, sigma, n, tol; };
  for (pt p : {pt{0.5, 0.001, 40.0, 1e-13}, pt{0.4, 0.002, 200.0, 1e-12}, pt{0.5, 1e-6, 12.0, 1e-14},
               pt{0.02, 0.001, 500.0, 1e-11}, pt{0.5, 0.01, 1000.0, 1e-10}}) {
    REQUIRE(10.0 * p.sigma <= std::fmin(p.mu, 1.0 - p.mu));
    for (double f : {0.0, 0.1, 0.5, 0.9, 1.0}) {
      const double y = std::floor(f * p.n);
      const double a = db::logpmf_sums(p.mu, p.sigma, p.n, y);
      const double b = db::logpmf_stirling(p.mu, p.sigma, p.n, y);
      INFO("mu " << p.mu << ", sigma " << p.sigma << ", n " << p.n << ", y " << y);
      CHECK(std::fabs(a - b) <= p.tol * std::fmax(1.0, std::fabs(a)));
      for (int K = 1; K <= 6; ++K) {
        double s[7], t[7];
        switch (K) {
          case 1: db::logpmf_dk_sums<1>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<1>(p.mu, p.sigma, p.n, y, t); break;
          case 2: db::logpmf_dk_sums<2>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<2>(p.mu, p.sigma, p.n, y, t); break;
          case 3: db::logpmf_dk_sums<3>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<3>(p.mu, p.sigma, p.n, y, t); break;
          case 4: db::logpmf_dk_sums<4>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<4>(p.mu, p.sigma, p.n, y, t); break;
          case 5: db::logpmf_dk_sums<5>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<5>(p.mu, p.sigma, p.n, y, t); break;
          default: db::logpmf_dk_sums<6>(p.mu, p.sigma, p.n, y, s), db::logpmf_dk_stirling<6>(p.mu, p.sigma, p.n, y, t); break;
        }
        double scale = 0.0;
        for (int b2 = 0; b2 <= K; ++b2) scale = std::fmax(scale, std::fabs(s[b2]));
        for (int b2 = 0; b2 <= K; ++b2) {
          INFO("K = " << K << ", component " << b2 << ", y " << y << ", n " << p.n);
          CHECK(std::fabs(s[b2] - t[b2]) <= p.tol * std::fmax(scale, 1.0));
        }
      }
    }
  }
  // outside its region the Stirling route must not be chosen: its series diverges there
  CHECK(db::use_stirling(0.5, 1.0, 1000.0, 0) == false);
  CHECK(db::use_stirling(0.02, 0.25, 1000.0, 3) == false);
  CHECK(db::use_stirling(0.5, 0.01, 1000.0, 0) == true);
  CHECK(db::use_stirling(0.5, 0.01, 1000.0, 6) == true);
  // and below the cut the sums are taken, per quantity
  CHECK(db::use_stirling(0.5, 0.001, 40.0, 0) == true);
  CHECK(db::use_stirling(0.5, 0.001, 40.0, 1) == false);
}

TEST_CASE("distrib betabinom: derivatives against differences") {
  betabinom_test_size = 10.0;
  for (double y : {0.0, 1.0, 5.0, 10.0}) check_derivs<ld::betabinom_distrib>({0.5, 0.5}, y, 1e-7);
  for (double y : {0.0, 2.0, 9.0}) check_derivs<ld::betabinom_distrib>({0.3, 2.0}, y, 1e-7);
  for (double y : {0.0, 3.0, 10.0}) check_derivs<ld::betabinom_distrib>({0.9, 0.05}, y, 1e-7);
  betabinom_test_size = 50.0;
  for (double y : {0.0, 1.0, 25.0, 50.0}) check_derivs<ld::betabinom_distrib>({0.5, 0.25}, y, 1e-7);
  for (double y : {0.0, 1.0, 10.0}) check_derivs<ld::betabinom_distrib>({0.02, 0.25}, y, 1e-7);
  betabinom_test_size = 10.0;
}

TEST_CASE("distrib betabinom: expected information and its derivatives") {
  betabinom_test_size = 12.0;
  for (double mu : {0.2, 0.5, 0.8})
    for (double sigma : {0.1, 0.5, 2.0}) {
      const ld::betabinom_distrib d{mu, sigma, 12.0};
      const auto I = ld::distrib_expected_info<0>(d);
      const auto s = score_variance(d);
      INFO("mu " << mu << ", sigma " << sigma);
      for (int p = 0; p < 3; ++p) CHECK(I[p] == doctest::Approx(s[p]).epsilon(1e-9));
      // I_mu_sigma vanishes at mu = 1/2 and only there: the mass is then symmetric under
      // y -> n - y together with mu -> 1 - mu, so the score in mu is odd about the centre while
      // the score in sigma is even, and their product has mean zero
      if (mu == 0.5) CHECK(std::fabs(I[1]) <= 1e-12 * std::sqrt(I[0] * I[2]));
      else CHECK(std::fabs(I[1]) > 1e-8 * std::sqrt(I[0] * I[2]));
    }
  check_info_derivs_sum({0.5, 0.5}, 3, 1e-6);
  check_info_derivs_sum({0.3, 1.0}, 3, 1e-6);
  betabinom_test_size = 10.0;
}

TEST_CASE("distrib betabinom: cdf identities") {
  for (double sigma : {0.05, 0.5, 5.0}) {
    const ld::betabinom_distrib d{0.35, sigma, 20.0};
    double s = 0.0;
    for (int k = 0; k <= 20; ++k) {
      s += std::exp(d.logpmf(k));
      INFO("sigma " << sigma << ", k " << k);
      CHECK(ld::distrib_cdf(d, k) == doctest::Approx(s).epsilon(1e-11));
      CHECK(ld::distrib_cdf(d, k) + ld::distrib_ccdf(d, k) == doctest::Approx(1.0).epsilon(1e-13));
      CHECK(ld::distrib_log_cdf(d, k) == doctest::Approx(std::log(ld::distrib_cdf(d, k))).epsilon(1e-11));
      CHECK(ld::distrib_cdf(d, k + 0.5) == ld::distrib_cdf(d, double(k)));
    }
    CHECK(ld::distrib_cdf(d, -1.0) == 0.0);
    CHECK(ld::distrib_ccdf(d, -1.0) == 1.0);
    CHECK(ld::distrib_log_cdf(d, -1.0) == -ld::distrib_infinity);
    CHECK(ld::distrib_log_ccdf(d, 20.0) == -ld::distrib_infinity);
    CHECK(ld::distrib_log_cdf(d, 20.0) == 0.0);
  }
}

// The four cumulative quantities against the summation twin of check_common.hpp (the twin is
// described there). The support is finite here, so the twin sums all of it exactly as written;
// the points include the two where the mass sits at one end, which are the hard ones for the
// route under test. The tolerance is the worst measured on gcc and CUDA times about 4.
TEST_CASE("distrib betabinom: the cumulative quantities against the summation twin") {
  for (const auto& th : {std::array<double, 3>{0.35, 0.5, 20.0}, std::array<double, 3>{0.02, 0.25, 20.0},
                         std::array<double, 3>{0.98, 0.25, 20.0}, std::array<double, 3>{0.5, 0.05, 50.0},
                         std::array<double, 3>{0.4, 5.0, 30.0}})
    check_discrete_cdf_twin(ld::betabinom_distrib{th[0], th[1], th[2]}, 1.3e-14);
}

TEST_CASE("distrib betabinom: derivatives of log F and log(1 - F) against differences") {
  betabinom_test_size = 12.0;
  for (bool upper : {false, true}) {
    for (double y : {0.0, 2.0, 6.0, 10.0}) check_cdf_derivs_sum({0.4, 0.5}, y, upper, 3, 1e-5);
    check_cdf_derivs_sum({0.6, 2.0}, 5.0, upper, 2, 1e-5);
  }
  betabinom_test_size = 10.0;
}

TEST_CASE("distrib betabinom: starting values from a sample summary") {
  // the moment estimates invert exactly: a summary whose mean and variance are the family's own
  // returns the parameters it was built from
  for (double n : {10.0, 50.0})
    for (double mu : {0.2, 0.5, 0.75})
      for (double sigma : {0.02, 0.5, 4.0}) {
        ld::distrib_sample_summary s;
        s.count = 1000.0;
        s.mean = n * mu;
        s.variance = n * mu * (1.0 - mu) * (1.0 + (n - 1.0) * sigma / (1.0 + sigma));
        const auto a = ld::betabinom_distrib{0.5, 1.0, n}.start_value(s);
        INFO("n " << n << ", mu " << mu << ", sigma " << sigma);
        CHECK(a.mu == doctest::Approx(mu).epsilon(1e-13));
        CHECK(a.sigma == doctest::Approx(sigma).epsilon(1e-11));
        CHECK(a.size == n);
      }
  // a sample with no overdispersion starts near the binomial limit, strictly inside the domain
  ld::distrib_sample_summary u;
  u.count = 1000.0;
  u.mean = 4.0;
  u.variance = 1.0;  // below the binomial's n mu (1 - mu) = 2.4
  const auto b = ld::betabinom_distrib{0.5, 1.0, 10.0}.start_value(u);
  CHECK(b.mu == doctest::Approx(0.4).epsilon(1e-14));
  CHECK(b.sigma > 0.0);
  CHECK(b.sigma < 1e-3);
  CHECK(std::isfinite(b.logpmf(4.0)));
  CHECK(std::isfinite(b.logpmf_d1(4.0)[1]));
  // a size of 1, where sigma is not identified
  const auto c = ld::betabinom_distrib{0.5, 1.0, 1.0}.start_value(u);
  CHECK(c.sigma > 0.0);
  CHECK(std::isfinite(c.logpmf(1.0)));
}

// The kernels live in named functions with external linkage, not in the body of a TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
double betabinom_kernel_value(double y) {
  const ld::betabinom_distrib d{0.35, 0.8, 12.0};
  return d.logpmf(y) + d.logpmf_d6(y)[3] + ld::distrib_log_cdf(d, y) + ld::distrib_log_ccdf(d, y) +
         ld::distrib_log_cdf_dk<2>(d, y)[1] + ld::distrib_expected_info<0>(d)[1];
}

void betabinom_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_betabinom", n, KOKKOS_LAMBDA(const int i) {
        const double y = double(i % 12);
        const ld::betabinom_distrib d{0.35, 0.8, 12.0};
        out(i) = d.logpmf(y) + d.logpmf_d6(y)[3] + ld::distrib_log_cdf(d, y) + ld::distrib_log_ccdf(d, y) +
                 ld::distrib_log_cdf_dk<2>(d, y)[1] + ld::distrib_expected_info<0>(d)[1];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double r = betabinom_kernel_value(double(i % 12));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-11));
  }
}

TEST_CASE("distrib betabinom: every order inside a kernel") { betabinom_kernel_case(); }

TEST_CASE("distrib betabinom: every order against mpmath") {
  // tools/reference/distrib_betabinom.py: kind 0 log p, 1 log F, 2 log(1 - F); b of the K
  // derivatives taken in sigma, K - b in mu. The error is the library value minus the mpmath value
  // (low part removed), in ulp of the largest component of the same (kind, y, mu, sigma, size, K)
  // group: every tensor of a given order has genuine zeros in y and in the parameters, so an error
  // is measured against the scale of the order.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_betabinom.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int kind, K, b; double y, mu, sigma, size, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], mus[64], sgs[64], ns[64], his[64], los[64];
    if (std::sscanf(line, "%d,%63[^,],%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", &r.kind, ys, mus, sgs, ns, &r.K,
                    &r.b, his, los) != 9)
      continue;
    r.y = std::strtod(ys, nullptr), r.mu = std::strtod(mus, nullptr), r.sigma = std::strtod(sgs, nullptr);
    r.size = std::strtod(ns, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 4000);
  std::map<std::tuple<int, double, double, double, double, int>, double> scale_of;
  for (const auto& r : rows) {
    auto& s = scale_of[std::make_tuple(r.kind, r.y, r.mu, r.sigma, r.size, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[3][7] = {};
  double worst_y[3][7] = {}, worst_mu[3][7] = {}, worst_sg[3][7] = {}, worst_n[3][7] = {};
  int worst_b[3][7] = {};
  // Bounds: 1.5 times the worst measured on the gcc (Serial + OpenMP) and the CUDA build
  // (2026-11-15, LAPLACE_PRINT_ERRORS), per kind and order, and for the mass per route as well,
  // since the two routes fail differently.
  //
  // Where the sums over the integer shifts are taken, the error is the n + 1 roundings of their one
  // pass and it follows (n/50)^(3/2) closely over the reference sizes 1 to 1e4: the worst ulp
  // divided by that factor stays between 28 and 163 over every order and every size, so that bound
  // carries the factor and stays tight at the sizes the family is normally used with.
  //
  // Where the form of bounded cost is taken the error does not follow the size, so the bound is
  // flat; it is wider at the value and the first two orders because that form splits the log-mass
  // into the binomial's and a correction, and in the tail of a large size the two are each of the
  // size of n while the log-mass is not: at n = 1e5, sigma = 0.01 that is 520 ulp on the value and
  // 1.8e3 at order 1, against 24 ulp at the orders 3 to 6 and 64.6 on everything at n = 1e4.
  //
  // The two cumulative kinds are the generic summation route's own accuracy, measured in its own
  // section, and their references stop at n = 50, so their bounds are flat.
  const double bound_sums[7] = {95.0, 150.0, 57.0, 73.0, 57.0, 43.0, 245.0};
  const double bound_stirling[7] = {780.0, 2750.0, 235.0, 36.0, 41.0, 35.0, 37.0};
  const double bound_cdf[2][7] = {{85.0, 72.0, 51.0, 3400.0, 3.3e4, 4.1e5, 6.1e6},
                                  {70.0, 74.0, 110.0, 1500.0, 2.2e4, 1.9e5, 3.7e6}};
  // the route is the library's own choice, so the test and the code cannot drift apart
  const auto bound_at = [&](int kind, int K, double mu, double sigma, double n) {
    if (kind != 0) return bound_cdf[kind - 1][K];
    if (ld::detail_betabinom::use_stirling(mu, sigma, n, K)) return bound_stirling[K];
    const double r = n > 50.0 ? n / 50.0 : 1.0;
    return bound_sums[K] * r * std::sqrt(r);
  };
  for (const auto& r : rows) {
    const ld::betabinom_distrib d{r.mu, r.sigma, r.size};
    const double v = r.kind == 0 ? (r.K == 0 ? d.logpmf(r.y) : lderiv(d, r.y, r.K)[r.b])
                                 : cderiv_sum(d, r.y, r.K, r.kind == 2)[r.b];
    const double sc = scale_of[std::make_tuple(r.kind, r.y, r.mu, r.sigma, r.size, r.K)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    if (ulp > worst[r.kind][r.K])
      worst_y[r.kind][r.K] = r.y, worst_mu[r.kind][r.K] = r.mu, worst_sg[r.kind][r.K] = r.sigma,
      worst_n[r.kind][r.K] = r.size, worst_b[r.kind][r.K] = r.b;
    worst[r.kind][r.K] = std::fmax(worst[r.kind][r.K], ulp);
    INFO("kind " << r.kind << " y " << r.y << " mu " << r.mu << " sigma " << r.sigma << " n " << r.size << " K " << r.K
                 << " b " << r.b << " ulp " << ulp);
    CHECK(ulp <= bound_at(r.kind, r.K, r.mu, r.sigma, r.size));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int kd = 0; kd < 3; ++kd)
      for (int K = 0; K <= 6; ++K)
        std::printf("betabinom kind %d K %d: max %.3g ulp (y %.6g, mu %.6g, sigma %.6g, n %.6g, b %d)\n", kd, K,
                    worst[kd][K], worst_y[kd][K], worst_mu[kd][K], worst_sg[kd][K], worst_n[kd][K], worst_b[kd][K]);
}
