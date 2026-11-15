// binomial_distrib (mean proportion mu, known size n; include/laplace/distrib/binomial.hpp):
// descriptors and the trailing size, the support and its ends, the mass against its exact
// binomial form and its normalization, every derivative order against a 5-point difference of the
// order below, the expected information against the variance of the score (a sum over the
// support) and its derivatives against differences, the cdf identities, the derivatives of
// log F and log(1 - F) (the generic route by summing the mass, there being no member) against
// central differences of the library's own log_cdf and log_ccdf, and every quantity against
// mpmath (tools/reference/distrib_binomial.py: n from 1 to 1e6, mu from 1e-12 to 1 - 1e-12)
// evaluated in a Kokkos kernel on the default execution space.
#include <doctest/doctest.h>

#include <laplace/distrib/binomial.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check_common.hpp"

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

// check_common builds a one-parameter distribution as D{theta}; the binomial also needs its size
namespace laplace_distrib_check {
inline double binomial_test_size = 10.0;
template <>
inline ld::binomial_distrib make<ld::binomial_distrib>(const double* t) {
  return ld::binomial_distrib{t[0], binomial_test_size};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included

// One quantity of the reference file, by its code (tools/reference/distrib_binomial.py): 0 the
// log-mass, 1..6 its derivatives, 10..14 the expected information and its derivatives, 20..23 the
// cdf family. External linkage and a name of its own: it is called from an extended device lambda.
LAPLACE_HD double binomial_test_value(const ld::binomial_distrib& d, int what, double y) {
  switch (what) {
    case 0: return d.logpmf(y);
    case 1: return d.logpmf_d1(y)[0];
    case 2: return d.logpmf_d2(y)[0];
    case 3: return d.logpmf_d3(y)[0];
    case 4: return d.logpmf_d4(y)[0];
    case 5: return d.logpmf_d5(y)[0];
    case 6: return d.logpmf_d6(y)[0];
    case 10: return d.expected_info()[0];
    case 11: return d.expected_info_d1().pair[0][0];
    case 12: return d.expected_info_d2().pair[0][0];
    case 13: return d.expected_info_d3().pair[0][0];
    case 14: return d.expected_info_d4().pair[0][0];
    case 20: return d.cdf(y);
    case 21: return d.ccdf(y);
    case 22: return d.log_cdf(y);
    default: return d.log_ccdf(y);
  }
}

// The kernel lives in a named function with external linkage, not in the body of a TEST_CASE (see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason).
std::vector<double> binomial_test_eval(const std::vector<int>& what, const std::vector<double>& n,
                                       const std::vector<double>& mu, const std::vector<double>& y) {
  const int m = (int)what.size();
  Kokkos::View<int*> w("w", m);
  Kokkos::View<double*> vn("n", m), vmu("mu", m), vy("y", m), out("out", m);
  auto hw = Kokkos::create_mirror_view(w);
  auto hn = Kokkos::create_mirror_view(vn), hmu = Kokkos::create_mirror_view(vmu), hy = Kokkos::create_mirror_view(vy);
  for (int i = 0; i < m; ++i) hw(i) = what[i], hn(i) = n[i], hmu(i) = mu[i], hy(i) = y[i];
  Kokkos::deep_copy(w, hw), Kokkos::deep_copy(vn, hn), Kokkos::deep_copy(vmu, hmu), Kokkos::deep_copy(vy, hy);
  Kokkos::parallel_for(
      "distrib_binomial", m, KOKKOS_LAMBDA(const int i) {
        const ld::binomial_distrib d{vmu(i), vn(i)};
        out(i) = binomial_test_value(d, w(i), vy(i));
      });
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  return std::vector<double>(ho.data(), ho.data() + m);
}

// the highest orders and the generic route for the derivatives of log F, in one launch
std::vector<double> binomial_test_kernel(int n) {
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_binomial_orders", n, KOKKOS_LAMBDA(const int i) {
        const double y = double(i % 31);
        const ld::binomial_distrib d{0.37, 30.0};
        out(i) = d.logpmf(y) + d.logpmf_d6(y)[0] + d.expected_info_d4().pair[0][0] + d.log_cdf(y) + d.log_ccdf(y) +
                 ld::distrib_log_cdf_dk<3>(d, y)[0] + ld::distrib_log_ccdf_dk<2>(d, y)[0];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  return std::vector<double>(h.data(), h.data() + n);
}

namespace {

// d^K log F (upper: log(1 - F)) of the binomial by the generic route, K = 1..3
std::vector<double> bin_cderiv(const ld::binomial_distrib& d, double y, int K, bool upper) {
  switch (K) {
    case 0: return {upper ? d.log_ccdf(y) : d.log_cdf(y)};
    case 1: return {(upper ? ld::distrib_log_ccdf_dk<1>(d, y) : ld::distrib_log_cdf_dk<1>(d, y))[0]};
    case 2: return {(upper ? ld::distrib_log_ccdf_dk<2>(d, y) : ld::distrib_log_cdf_dk<2>(d, y))[0]};
    default: return {(upper ? ld::distrib_log_ccdf_dk<3>(d, y) : ld::distrib_log_cdf_dk<3>(d, y))[0]};
  }
}

double worst_cdf_derivs = 0.0;  // the largest |generic - difference| / (scale), over the cases

// the generic derivative of order K against a 5-point difference in mu of the one below (of log F itself at K = 1)
void check_binom_cdf_derivs(double n, double mu, double y, bool upper, double rel) {
  for (int K = 1; K <= 3; ++K) {
    const double exact = bin_cderiv({mu, n}, y, K, upper)[0];
    const double h = 1e-3 * std::fmin(mu, 1.0 - mu);
    double f[4], scale = std::fabs(exact);
    int j = 0;
    for (double s : {-2.0, -1.0, 1.0, 2.0}) {
      f[j] = bin_cderiv({mu + s * h, n}, y, K - 1, upper)[0];
      scale = std::fmax(scale, std::fabs(f[j]));
      ++j;
    }
    const double fd = (f[0] - 8.0 * f[1] + 8.0 * f[2] - f[3]) / (12.0 * h);
    const double err = std::fabs(exact - fd) / (scale > 0 ? scale : 1.0);
    if (scale > 1e-6) worst_cdf_derivs = std::fmax(worst_cdf_derivs, err);
    INFO("n = " << n << ", mu = " << mu << ", y = " << y << ", upper " << upper << ", K = " << K);
    CHECK(std::fabs(exact - fd) <= rel * scale + 1e-12);
  }
}

// every order K = 1..6 against a 5-point difference in mu of the order below (of the log-mass itself at
// K = 1), step 1e-3 min(mu, 1 - mu), the error taken in units of the larger of the two terms of the
// derivative, (K-1)! (y / mu^K + (n - y) / (1 - mu)^K): they cancel near y = n mu, where the derivative
// is genuinely small and a bound relative to its own value would test the cancellation of the difference
void check_binom_derivs(double n, double mu, double y, double rel) {
  const ld::binomial_distrib d{mu, n};
  const double h = 1e-3 * std::fmin(mu, 1.0 - mu);
  for (int K = 1; K <= 6; ++K) {
    const auto val = [&](double m, int order) { return lderiv(ld::binomial_distrib{m, n}, y, order)[0]; };
    const double q = 1.0 - mu;
    double a = y, b = n - y, f = 1.0;
    for (int k = 1; k <= K; ++k) a /= mu, b /= q, f *= (k > 1 ? k - 1 : 1);
    const double scale = f * (a + b);
    const double fd = (val(mu - 2 * h, K - 1) - 8.0 * val(mu - h, K - 1) + 8.0 * val(mu + h, K - 1) - val(mu + 2 * h, K - 1)) / (12.0 * h);
    INFO("n = " << n << ", mu = " << mu << ", y = " << y << ", K = " << K);
    CHECK(std::fabs(lderiv(d, y, K)[0] - fd) <= rel * scale);
  }
}

// E[score^2] by summing the mass over the support
double score_variance_binomial(const ld::binomial_distrib& d) {
  double s = 0.0;
  for (int y = 0; y <= (int)d.size; ++y) {
    const double g = d.logpmf_d1(y)[0];
    s += g * g * std::exp(d.logpmf(y));
  }
  return s;
}

}  // namespace

TEST_CASE("distrib binomial: descriptors, size and support") {
  CHECK(ld::binomial_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::binomial_distrib::n_params == 1);
  CHECK(ld::binomial_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::binomial_distrib::params[0].cdf_regularity == ld::infinitely_smooth);
  CHECK(ld::binomial_distrib::params[0].moves_support == false);
  const ld::binomial_distrib d{0.3, 20.0};
  CHECK(d.param_kinks(0, 1.25).count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == 20.0);
  CHECK(d.center() == doctest::Approx(6.0).epsilon(1e-15));
  CHECK(d.scale() == doctest::Approx(std::sqrt(4.2)).epsilon(1e-15));
  // outside the support or at a non-integer y: no mass
  CHECK(d.logpmf(-1.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(21.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(2.5) == -ld::distrib_infinity);
  CHECK(std::isfinite(d.logpmf(0.0)));
  CHECK(std::isfinite(d.logpmf(20.0)));
}

TEST_CASE("distrib binomial: mass against the exact binomial form and its normalization") {
  for (double n : {1.0, 2.0, 7.0, 40.0, 200.0})
    for (double mu : {0.02, 0.3, 0.5, 0.85}) {
      const ld::binomial_distrib d{mu, n};
      double total = 0.0;
      for (int y = 0; y <= (int)n; ++y) {
        const double ref = std::lgamma(n + 1) - std::lgamma(y + 1.0) - std::lgamma(n - y + 1.0) + y * std::log(mu) +
                           (n - y) * std::log1p(-mu);
        INFO("n = " << n << ", mu = " << mu << ", y = " << y);
        CHECK(d.logpmf(y) == doctest::Approx(ref).epsilon(1e-12).scale(1.0));
        total += std::exp(d.logpmf(y));
      }
      CHECK(total == doctest::Approx(1.0).epsilon(1e-13));
    }
  // size 1 is the Bernoulli: log mu at y = 1, log(1 - mu) at y = 0, F(0) = 1 - mu
  for (double mu : {1e-9, 0.2, 0.5, 0.97, 1.0 - 1e-9}) {
    const ld::binomial_distrib b{mu, 1.0};
    CHECK(b.logpmf(1.0) == doctest::Approx(std::log(mu)).epsilon(1e-15));
    CHECK(b.logpmf(0.0) == doctest::Approx(std::log1p(-mu)).epsilon(1e-15));
    CHECK(b.cdf(0.0) == doctest::Approx(1.0 - mu).epsilon(1e-15));
    CHECK(b.ccdf(0.0) == doctest::Approx(mu).epsilon(1e-15));
    CHECK(b.expected_info()[0] == doctest::Approx(1.0 / (mu * (1.0 - mu))).epsilon(1e-15));
  }
}

TEST_CASE("distrib binomial: derivatives against differences") {
  for (double n : {10.0, 1000.0}) {
    for (double mu : {0.05, 0.3, 0.5, 0.8, 0.97}) {
      const std::vector<double> ys{0.0, 1.0, std::round(n * mu), std::round(0.5 * n), n - 1.0, n};
      for (double y : ys) check_binom_derivs(n, mu, y, 1e-8);
    }
  }
  for (double y : {0.0, 1.0}) check_binom_derivs(1.0, 0.4, y, 1e-8);
}

TEST_CASE("distrib binomial: expected information against the variance of the score") {
  for (double n : {1.0, 10.0, 100.0, 2000.0})
    for (double mu : {0.01, 0.3, 0.5, 0.9}) {
      const ld::binomial_distrib d{mu, n};
      INFO("n = " << n << ", mu = " << mu);
      CHECK(score_variance_binomial(d) == doctest::Approx(d.expected_info()[0]).epsilon(1e-11));
    }
  binomial_test_size = 25.0;
  check_info_derivs<ld::binomial_distrib>({0.3}, 1e-8);
  check_info_derivs<ld::binomial_distrib>({0.02}, 1e-8);
  check_info_derivs<ld::binomial_distrib>({0.6}, 1e-8);
}

TEST_CASE("distrib binomial: cdf identities") {
  for (double n : {1.0, 5.0, 30.0}) {
    const ld::binomial_distrib d{0.35, n};
    double F = 0.0;
    for (int k = 0; k <= (int)n; ++k) {
      F += std::exp(d.logpmf(k));
      INFO("n = " << n << ", k = " << k);
      CHECK(d.cdf(k) == doctest::Approx(F).epsilon(1e-13));
      CHECK(d.cdf(k + 0.5) == d.cdf(k));
      CHECK(d.cdf(k) + d.ccdf(k) == doctest::Approx(1.0).epsilon(1e-15));
      if (k < (int)n) {
        CHECK(d.log_cdf(k) == doctest::Approx(std::log(d.cdf(k))).epsilon(1e-13));
        CHECK(d.log_ccdf(k) == doctest::Approx(std::log(d.ccdf(k))).epsilon(1e-13));
      }
    }
    // the ends
    CHECK(d.cdf(-1.0) == 0.0);
    CHECK(d.ccdf(-1.0) == 1.0);
    CHECK(d.log_cdf(-1.0) == -ld::distrib_infinity);
    CHECK(d.log_ccdf(-1.0) == 0.0);
    CHECK(d.cdf(n) == 1.0);
    CHECK(d.ccdf(n) == 0.0);
    CHECK(d.cdf(n + 3.0) == 1.0);
    CHECK(d.log_cdf(n) == 0.0);
    CHECK(d.log_ccdf(n) == -ld::distrib_infinity);
  }
  // far tails keep their logarithm: n = 1e6, mu = 0.5, F(10) is about 1e-301000
  const ld::binomial_distrib t{0.5, 1e6};
  CHECK(t.cdf(10.0) == 0.0);
  CHECK(t.log_cdf(10.0) < -6.9e5);
  CHECK(t.log_cdf(10.0) > -7.0e5);
  CHECK(t.log_ccdf(10.0) == doctest::Approx(0.0).scale(1.0).epsilon(1e-300));
}

TEST_CASE("distrib binomial: derivatives of log F and log(1 - F), generic route, against differences of log_cdf") {
  // orders 1 to 3, no member: distrib_log_cdf_dk takes the sum over the mass; the worst relative
  // difference over these cases is printed with LAPLACE_PRINT_ERRORS
  struct pt { double n, mu; };
  const pt pts[] = {{1, 0.3}, {2, 0.5}, {10, 0.05}, {10, 0.5}, {10, 0.8}, {100, 0.3}, {100, 0.9}, {1000, 0.02}, {1000, 0.5}};
  for (bool upper : {false, true})
    for (const auto& p : pts) {
      const double m = std::round(p.n * p.mu);
      for (double y : {0.0, 1.0, m, m + 1.0, std::round(0.5 * p.n), p.n - 2.0, p.n - 1.0}) {
        if (y < 0.0 || y >= p.n) continue;
        check_binom_cdf_derivs(p.n, p.mu, y, upper, 1e-6);
      }
    }
  // measured 2026-11-15: 4.0e-8, the central differences' own error (step 1e-3 min(mu, 1 - mu))
  CHECK(worst_cdf_derivs <= 1e-6);
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("binomial generic log F derivatives vs central differences: worst relative %.3g\n", worst_cdf_derivs);
}

TEST_CASE("distrib binomial: every order inside a kernel") {
  const int n = 62;
  const auto h = binomial_test_kernel(n);
  for (int i = 0; i < n; ++i) {
    const double y = double(i % 31);
    const ld::binomial_distrib d{0.37, 30.0};
    const double r = d.logpmf(y) + d.logpmf_d6(y)[0] + d.expected_info_d4().pair[0][0] + d.log_cdf(y) + d.log_ccdf(y) +
                     ld::distrib_log_cdf_dk<3>(d, y)[0] + ld::distrib_log_ccdf_dk<2>(d, y)[0];
    if (std::isnan(r)) continue;  // y = 30 = n: log(1 - F) = -inf, its derivative NaN
    CHECK(h[i] == doctest::Approx(r).epsilon(1e-11).scale(1.0));
  }
}

TEST_CASE("distrib binomial: every quantity against mpmath") {
  // tools/reference/distrib_binomial.py, 2026-11-15. The error of each quantity is the library value
  // (evaluated in a Kokkos kernel) minus the mpmath value, its low part removed, in ulp of the value
  // ("own"); for the derivatives of the log-mass (codes 1..6) in ulp of the larger of the two terms
  // (-1)^(K-1) y / mu^K and (n - y) / (1 - mu)^K times (K-1)!, because the two cancel near the mode
  // y = n mu, and the derivative there is genuinely small compared with them.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_binomial.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  std::vector<int> what;
  std::vector<double> n, mu, y, hi, lo;
  while (std::fgets(line, sizeof line, f)) {
    int w;
    char ns[64], mus[64], ys[64], his[64], los[64];
    if (std::sscanf(line, "%d,%63[^,],%63[^,],%63[^,],%63[^,],%63s", &w, ns, mus, ys, his, los) != 6) continue;
    what.push_back(w), n.push_back(std::strtod(ns, nullptr)), mu.push_back(std::strtod(mus, nullptr));
    y.push_back(std::strtod(ys, nullptr)), hi.push_back(std::strtod(his, nullptr)), lo.push_back(std::strtod(los, nullptr));
  }
  std::fclose(f);
  REQUIRE(what.size() > 7000);
  const auto v = binomial_test_eval(what, n, mu, y);
  double worst[28] = {}, worst_own[28] = {}, wn[28] = {}, wm[28] = {}, wy[28] = {};
  for (size_t i = 0; i < what.size(); ++i) {
    // the logarithms of a probability within 1/e of 1 are classed apart (codes 24, 25): their value is near 0
    const int w0 = what[i], w = (w0 >= 22 && std::fabs(hi[i]) < 1.0) ? w0 + 2 : w0;
    const double err = std::fabs((v[i] - hi[i]) - lo[i]);
    double unit = hi[i];
    if (w0 >= 1 && w0 <= 6) {
      const double q = 1.0 - mu[i];
      double a = y[i], b = n[i] - y[i], fct = 1.0;
      for (int k = 1; k <= w0; ++k) a /= mu[i], b /= q, fct *= (k > 1 ? k - 1 : 1);
      unit = fct * (a + b);
    }
    const double ulp = err / std::ldexp(1.0, std::ilogb(unit) - 52);
    const double own = err / std::ldexp(1.0, std::ilogb(hi[i]) - 52);
    if (ulp > worst[w]) worst[w] = ulp, wn[w] = n[i], wm[w] = mu[i], wy[w] = y[i];
    worst_own[w] = std::fmax(worst_own[w], own);
    INFO("what " << w << ", n " << n[i] << ", mu " << mu[i] << ", y " << y[i] << ", error " << ulp << " ulp");
    // bounds: 1.5 times the maxima measured on the gcc and the CUDA builds (2026-11-15), in the order of the
    // codes: log-mass 7.66, derivatives 1.63 2.0 2.4 3.4 3.8 5.6 (in ulp of the larger term), information 0.96
    // 2.1 1.4 2.1 2.0, F 79, 1 - F 461, log F 4.7, log(1 - F) 4.3; and for the logarithms within 1/e of 0
    // (codes 24, 25: log F, log(1 - F) of a probability near 1, where the value is about the other tail's
    // probability, as small as 1e-80 here) 9.0e4 and 2.5e4: the incomplete beta gives the tail as
    // exp(log w) with an absolute error of a few ulp in log w, of size |log w|, so the relative error of
    // the tiny value is about |log w| times larger than one ulp of it, while its absolute error is below 1e-90
    const double bound[28] = {12, 2.5, 3, 3.6, 5.1, 5.7, 8.5, 0, 0, 0, 1.5, 3.2, 2.1, 3.2, 3.1, 0, 0, 0, 0, 0,
                              119, 692, 7.1, 6.6, 1.36e5, 3.7e4, 0, 0};
    CHECK(ulp <= bound[w]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int w = 0; w < 28; ++w)
      if (worst[w] > 0.0 || worst_own[w] > 0.0)
        std::printf("binomial what %2d: max %.3g ulp (n %.6g, mu %.6g, y %.6g), %.3g ulp of the value\n", w, worst[w], wn[w],
                    wm[w], wy[w], worst_own[w]);
}
