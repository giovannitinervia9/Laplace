// pig_distrib (mean mu, shape alpha; include/laplace/distrib/pig.hpp): descriptors, the mass against
// the direct assembly with a Bessel function and against its normalization, the mean and the
// variance from the mass, every derivative order against a 5-point difference of the order below,
// the closed blocks of the expected information against the variance of the score and its
// derivatives against differences, the cdf of the generic sum against the partial sums of the mass,
// the derivatives of log F and log(1 - F) against differences, a kernel launch of the highest
// orders, I_alpha_alpha against the reference that mpmath sums from the definition by its own
// route, and every order against mpmath (tools/reference/distrib_pig.py).
#include <doctest/doctest.h>

#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/pig.hpp>
#include <laplace/math/bessel_k.hpp>
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

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// the log-mass written directly from its definition, with the library's own Bessel function at the
// half-integer order: the twin of the half-integer polynomial route
double logpmf_direct(double mu, double alpha, double y) {
  const double s = std::sqrt(mu * mu + alpha * alpha);
  return 0.5 * std::log(2.0 * alpha / 3.14159265358979323846) + (s - mu) +
         y * std::log(mu * alpha / (mu + s)) + laplace::math::log_bessel_k(y - 0.5, alpha) -
         std::lgamma(y + 1.0);
}

// the expected information and its derivatives, which the family supplies as a member
std::vector<double> ideriv_member(const ld::pig_distrib& d, int K) { return ideriv(d, K); }

void check_info_derivs(const std::vector<double>& th, int kmax, double rel) {
  const int npair = ld::sym_tensor_size(2, 2);
  for (int K = 1; K <= kmax; ++K) {
    const auto exact = ideriv_member(make<ld::pig_distrib>(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (int p = 0; p < npair; ++p)
      for (const auto& idx : sorted_indices(2, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ideriv_member(make<ld::pig_distrib>(t), K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(2, K - 1);
        const double e = exact[p * ld::sym_tensor_size(2, K) + position_of(2, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + position_of(2, lower)];
        if (std::isnan(e) || std::isnan(a)) continue;
        INFO("K = " << K << ", pair " << p << ", mu " << th[0] << ", alpha " << th[1]);
        CHECK(std::fabs(e - a) <= rel * scale);
      }
  }
}

// log F (upper: log(1 - F)) and its derivatives by the generic summation route
std::vector<double> cderiv_sum(const ld::pig_distrib& d, double y, int K, bool upper) {
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
    const auto exact = cderiv_sum(make<ld::pig_distrib>(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(2, K)) {
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const int j = idx.back();
      const auto fd = diff5([&](const double* t) { return cderiv_sum(make<ld::pig_distrib>(t), y, K - 1, upper); }, th,
                            j, 1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      INFO("K = " << K << ", component " << position_of(2, idx) << ", y = " << y << ", upper " << upper << ", mu "
                  << th[0] << ", alpha " << th[1]);
      CHECK(std::fabs(exact[position_of(2, idx)] - fd[position_of(2, lower)]) <= rel * scale + 1e-12);
    }
  }
}

// E[l_a l_b] by summing over the support up to n, pair-major
std::vector<double> score_variance(const ld::pig_distrib& d, int n) {
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

TEST_CASE("distrib pig: descriptors and support") {
  CHECK(ld::pig_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::pig_distrib::n_params == 2);
  for (int j = 0; j < 2; ++j) {
    CHECK(ld::pig_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::pig_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::pig_distrib::params[j].moves_support == false);
  }
  CHECK(std::string(ld::pig_distrib::params[0].name) == "mu");
  CHECK(std::string(ld::pig_distrib::params[1].name) == "alpha");
  const ld::pig_distrib d{2.0, 3.0};
  const double s = std::sqrt(13.0);
  CHECK(d.param_kinks(1, 0.5).count == 0);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 2.0);
  CHECK(d.radius() == doctest::Approx(s).epsilon(1e-15));
  CHECK(d.excess() == doctest::Approx(s - 2.0).epsilon(1e-14));
  CHECK(d.scale() == doctest::Approx(std::sqrt(2.0 + 4.0 / (s - 2.0))).epsilon(1e-13));
  CHECK(d.logpmf(-1.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(2.5) == -ld::distrib_infinity);
  // y = 0 is exactly s - mu - alpha
  CHECK(d.logpmf(0.0) == doctest::Approx(s - 2.0 - 3.0).epsilon(1e-15));
}

TEST_CASE("distrib pig: the mass against the direct Bessel assembly, its normalization and moments") {
  struct pt { double mu, alpha; };
  for (pt p : {pt{2.0, 3.0}, pt{0.5, 0.1}, pt{10.0, 1.0}, pt{40.0, 1e-2}, pt{1.0, 1e3}, pt{100.0, 5.0}}) {
    const ld::pig_distrib d{p.mu, p.alpha};
    const double s = std::sqrt(p.mu * p.mu + p.alpha * p.alpha);
    double total = 0.0, mean = 0.0, sq = 0.0;
    const long n = (long)(p.mu + 400.0 * d.scale()) + 200;
    for (long y = 0; y <= n && y < 20000000L; ++y) {
      const double l = d.logpmf((double)y);
      if (y < 30) {
        INFO("mu " << p.mu << ", alpha " << p.alpha << ", y " << y);
        CHECK(l == doctest::Approx(logpmf_direct(p.mu, p.alpha, (double)y)).epsilon(1e-11).scale(1.0));
      }
      const double q = std::exp(l);
      total += q;
      mean += q * (double)y;
      sq += q * (double)y * (double)y;
    }
    INFO("mu " << p.mu << ", alpha " << p.alpha);
    // the moments are checked only where the tail is light enough for a cut sum to hold them: the
    // mass decays as e^(-y log((mu + s) / (2 mu))), whose rate is about alpha^2 / (4 mu^2) when
    // alpha is much smaller than mu, so at (40, 0.01) the first moment needs counts beyond 1e7
    if (d.scale() < 1e3) {
      CHECK(total == doctest::Approx(1.0).epsilon(1e-8));
      CHECK(mean == doctest::Approx(p.mu).epsilon(1e-6));
      CHECK(sq - mean * mean == doctest::Approx(p.mu + p.mu * p.mu / (s - p.mu)).epsilon(1e-3));
    } else {
      // only that the cut sum is below one and rising towards it
      CHECK(total > 0.9);
      CHECK(total <= 1.0 + 1e-12);
    }
  }
}

TEST_CASE("distrib pig: derivatives against differences") {
  for (double y : {0.0, 1.0, 2.0, 5.0}) check_derivs<ld::pig_distrib>({2.0, 3.0}, y, 1e-7);
  for (double y : {0.0, 1.0, 10.0}) check_derivs<ld::pig_distrib>({10.0, 1.0}, y, 1e-7);
  for (double y : {0.0, 3.0, 40.0}) check_derivs<ld::pig_distrib>({40.0, 1e-2}, y, 1e-6);
  // at the Poisson limit two points are left to the mpmath references instead of the differences:
  // y = mu, where the score in mu is exactly 0 and so is most of each tensor, and y = 0, where the
  // tensors are of the size of (mu / alpha)^K, below the 5-point difference's own truncation error
  for (double y : {2.0, 4.0}) check_derivs<ld::pig_distrib>({1.0, 1e3}, y, 1e-6);
  for (double y : {0.0, 2.0, 100.0, 200.0}) check_derivs<ld::pig_distrib>({100.0, 5.0}, y, 1e-6);
}

TEST_CASE("distrib pig: the expected information is diagonal") {
  for (double mu : {0.5, 2.0, 10.0})
    for (double alpha : {0.3, 1.0, 5.0}) {
      const ld::pig_distrib d{mu, alpha};
      const auto I = d.expected_info();
      const double s = std::sqrt(mu * mu + alpha * alpha);
      INFO("mu " << mu << ", alpha " << alpha);
      CHECK(I[0] == doctest::Approx(1.0 / mu - 1.0 / s).epsilon(1e-14));  // closed
      CHECK(I[1] == 0.0);                                                 // exactly zero
      const auto v = score_variance(d, (int)(mu + 2000.0 * d.scale()) + 2000);
      CHECK(I[0] == doctest::Approx(v[0]).epsilon(1e-6));
      CHECK(I[2] == doctest::Approx(v[2]).epsilon(1e-6));
      CHECK(std::fabs(v[1]) <= 1e-6 * std::sqrt(v[0] * v[2]));
    }
}

TEST_CASE("distrib pig: derivatives of the expected information against differences") {
  check_info_derivs({2.0, 3.0}, 3, 1e-5);
  check_info_derivs({10.0, 1.0}, 2, 1e-5);
}

TEST_CASE("distrib pig: cdf identities") {
  for (double alpha : {0.3, 1.0, 10.0}) {
    const ld::pig_distrib d{3.0, alpha};
    double s = 0.0;
    for (int k = 0; k < 20; ++k) {
      s += std::exp(d.logpmf(k));
      INFO("alpha " << alpha << ", k " << k);
      CHECK(ld::distrib_cdf(d, k) == doctest::Approx(s).epsilon(1e-10));
      CHECK(ld::distrib_cdf(d, k) + ld::distrib_ccdf(d, k) == doctest::Approx(1.0).epsilon(1e-12));
      CHECK(ld::distrib_log_cdf(d, k) == doctest::Approx(std::log(ld::distrib_cdf(d, k))).epsilon(1e-10));
      CHECK(ld::distrib_cdf(d, k + 0.5) == ld::distrib_cdf(d, double(k)));
    }
    CHECK(ld::distrib_cdf(d, -1.0) == 0.0);
    CHECK(ld::distrib_ccdf(d, -1.0) == 1.0);
    CHECK(ld::distrib_log_cdf(d, -1.0) == -ld::distrib_infinity);
    CHECK(ld::distrib_log_ccdf(d, -1.0) == 0.0);
  }
}

// The four cumulative quantities against the summation twin of check_common.hpp, which shares
// nothing with the route under test: it walks the whole support in plain probabilities and sums
// each side from its own end, while the library sums the cheaper tail in units of its first mass
// and takes the other side by log1p(-exp). The points cover both sides of the switch at the mode
// and the long tail that a small alpha gives (spread 0.0025 at (3, 0.3), where the twin needs
// 1.6e4 terms). The tolerance is the worst measured on gcc and CUDA times about 4.
TEST_CASE("distrib pig: the cumulative quantities against the summation twin") {
  for (const auto& th : {std::array<double, 2>{3.0, 0.3}, std::array<double, 2>{3.0, 10.0},
                         std::array<double, 2>{1.0, 1.0}, std::array<double, 2>{10.0, 0.5},
                         std::array<double, 2>{0.5, 2.0}})
    check_discrete_cdf_twin(ld::pig_distrib{th[0], th[1]}, 8e-9);
}

TEST_CASE("distrib pig: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {0.0, 2.0, 6.0}) check_cdf_derivs_sum({3.0, 1.0}, y, upper, 3, 1e-5);
    check_cdf_derivs_sum({2.0, 3.0}, 4.0, upper, 2, 1e-5);
  }
}

TEST_CASE("distrib pig: starting values from a sample summary") {
  // the moment estimates invert exactly: a summary whose mean and variance are the family's own
  // returns the parameters it was built from
  for (double mu : {0.5, 3.5, 40.0})
    for (double alpha : {0.1, 1.0, 20.0}) {
      const double s = std::sqrt(mu * mu + alpha * alpha);
      ld::distrib_sample_summary sm;
      sm.count = 1000.0;
      sm.mean = mu;
      sm.variance = mu + mu * mu / (s - mu);
      const auto a = ld::pig_distrib{1.0, 1.0}.start_value(sm);
      INFO("mu " << mu << ", alpha " << alpha);
      CHECK(a.mu == doctest::Approx(mu).epsilon(1e-13));
      CHECK(a.alpha == doctest::Approx(alpha).epsilon(1e-10));
    }
  // a sample with no overdispersion starts near the Poisson limit, strictly inside the domain
  ld::distrib_sample_summary u;
  u.count = 1000.0;
  u.mean = 7.0;
  u.variance = 5.0;
  const auto b = ld::pig_distrib{1.0, 1.0}.start_value(u);
  CHECK(b.mu == 7.0);
  CHECK(b.alpha > 100.0);
  CHECK(std::isfinite(b.logpmf(7.0)));
  CHECK(std::isfinite(b.logpmf_d1(7.0)[1]));
}

// The kernels live in named functions with external linkage, not in the body of a TEST_CASE: see
// the comment on gaussian3_kernel_case in test_gaussian3.cpp for the reason.
double pig_kernel_value(double y) {
  const ld::pig_distrib d{3.0, 2.5};
  return d.logpmf(y) + d.logpmf_d6(y)[3] + ld::distrib_log_cdf(d, y) + ld::distrib_log_ccdf(d, y) +
         ld::distrib_log_cdf_dk<2>(d, y)[1] + d.expected_info()[2];
}

void pig_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_pig", n, KOKKOS_LAMBDA(const int i) {
        const double y = double(i % 12);
        const ld::pig_distrib d{3.0, 2.5};
        out(i) = d.logpmf(y) + d.logpmf_d6(y)[3] + ld::distrib_log_cdf(d, y) + ld::distrib_log_ccdf(d, y) +
                 ld::distrib_log_cdf_dk<2>(d, y)[1] + d.expected_info()[2];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double r = pig_kernel_value(double(i % 12));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-10));
  }
}

TEST_CASE("distrib pig: every order inside a kernel") { pig_kernel_case(); }

// I_alpha_alpha against tests/data/reference/pig_info.csv, which
// tools/reference/distrib_pig.py --info writes: mpmath sums the same definition at 40 digits with
// its own Bessel function of real order and its own Euler-Maclaurin acceleration, so neither the
// half-integer polynomial of the mass nor the quadrature of the library's tail enters it. The rows
// run from the Poisson limit to alpha / mu = 1e-4 and cover both of the member's routes, the
// generic sweep below a spread of 4 and the cut sum with its closed tail above it, which is also
// the only one of the two that is defined at the heavy-tailed end.
void check_pig_info_reference() {
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/pig_info.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  // the bound is 1.5 times the worst measured on the gcc and the CUDA builds (2026-11-16)
  constexpr double bound = 2.1e-11;
  double worst = 0.0, worst_mu = 0.0, worst_al = 0.0;
  int rows = 0, cut_rows = 0;
  while (std::fgets(line, sizeof line, f)) {
    char mus[64], als[64], his[64], los[64];
    if (std::sscanf(line, "%63[^,],%63[^,],%63[^,],%63s", mus, als, his, los) != 4) continue;
    const double mu = std::strtod(mus, nullptr), alpha = std::strtod(als, nullptr);
    const double hi = std::strtod(his, nullptr), lo = std::strtod(los, nullptr);
    const ld::pig_distrib d{mu, alpha};
    const double v = d.expected_info()[2];
    const double rel = std::fabs((v - hi) - lo) / std::fabs(hi);
    INFO("mu " << mu << ", alpha " << alpha << ", spread " << d.scale() << ", value " << v << ", reference "
               << hi << ", relative " << rel);
    CHECK(rel <= bound);
    if (rel > worst) worst = rel, worst_mu = mu, worst_al = alpha;
    ++rows;
    if (d.scale() > ld::detail_pig::info_cut_scale) ++cut_rows;
  }
  std::fclose(f);
  CHECK(rows >= 15);
  CHECK(cut_rows >= 8);      // both routes are exercised
  CHECK(rows - cut_rows >= 4);
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pig I_alpha_alpha: max %.3g relative (mu %.6g, alpha %.6g), %d rows, %d of them cut\n", worst,
                worst_mu, worst_al, rows, cut_rows);
}

// The derivatives of I_alpha_alpha on the cut route, against the generic sweep where the sweep
// still converges. Both points have a spread above 4, so the member takes the cut route, and the
// sweep is reached here on purpose to compare the two. The tolerances are the worst measured on the
// gcc and the CUDA builds times about 4: at (3.5, 2) the two agree to 1.9e-15 of the scale of the
// order, and at (1, 0.1), where the sweep needs 1.6e5 terms and the tail of these components weighs
// far more than the value's, to 2.5e-8.
template <int K>
void check_pig_info_dk_against_sweep(double mu, double alpha, double tol) {
  const ld::pig_distrib d{mu, alpha};
  REQUIRE(d.scale() > ld::detail_pig::info_cut_scale);
  const auto mine = d.expected_info_dk<K>();
  const auto sweep = ld::expected_info_by_integral<K, ld::info_form::score>(d);
  double scale = 0.0;
  for (int b = 0; b <= K; ++b) scale = std::fmax(scale, std::fabs(mine.pair[2][b]));
  for (int b = 0; b <= K; ++b) {
    if (!(std::fabs(sweep.pair[2][b]) < 1e308)) continue;  // the sweep gives up at the heavy end
    INFO("K = " << K << ", component " << b << ", mu " << mu << ", alpha " << alpha << ", cut "
                << mine.pair[2][b] << ", sweep " << sweep.pair[2][b]);
    CHECK(std::fabs(mine.pair[2][b] - sweep.pair[2][b]) <= tol * scale);
  }
  // the mean blocks stay closed whichever route the shape block takes
  const double s = std::sqrt(mu * mu + alpha * alpha);
  CHECK(d.expected_info()[0] == doctest::Approx(1.0 / mu - 1.0 / s).epsilon(1e-14));
  for (int b = 0; b <= K; ++b) CHECK(mine.pair[1][b] == 0.0);
}

TEST_CASE("distrib pig: the derivatives of I_alpha_alpha, cut route against the sweep") {
  check_pig_info_dk_against_sweep<1>(3.5, 2.0, 1e-13);
  check_pig_info_dk_against_sweep<2>(3.5, 2.0, 1e-13);
  check_pig_info_dk_against_sweep<3>(3.5, 2.0, 1e-13);
  check_pig_info_dk_against_sweep<4>(3.5, 2.0, 1e-13);
  check_pig_info_dk_against_sweep<1>(1.0, 0.1, 1e-7);
  check_pig_info_dk_against_sweep<2>(1.0, 0.1, 1e-7);
  check_pig_info_dk_against_sweep<3>(1.0, 0.1, 1e-7);
  check_pig_info_dk_against_sweep<4>(1.0, 0.1, 1e-7);
}

// At the heavy-tailed end the sweep reports no convergence, so there is nothing to compare against
// but the family itself: every component must be finite, and the derivatives must agree with the
// differences of the order below, which run on the same route.
TEST_CASE("distrib pig: the information and its derivatives at the heavy-tailed end") {
  for (const auto& th : {std::vector<double>{10.0, 1e-3}, std::vector<double>{1.0, 1e-2}}) {
    const ld::pig_distrib d{th[0], th[1]};
    const auto I = d.expected_info();
    INFO("mu " << th[0] << ", alpha " << th[1] << ", spread " << d.scale());
    CHECK(std::isfinite(I[2]));
    CHECK(I[2] > 0.0);
    const auto g = d.expected_info_d1();
    for (int b = 0; b <= 1; ++b) CHECK(std::isfinite(g.pair[2][b]));
    check_info_derivs(th, 2, 2e-4);
  }
}

TEST_CASE("distrib pig: I_alpha_alpha against the mpmath reference") { check_pig_info_reference(); }

// The two routes of I_alpha_alpha where both are defined: the generic sweep is forced by asking it
// directly, and must agree with the member wherever it converges honestly.
TEST_CASE("distrib pig: the two routes of I_alpha_alpha agree" * doctest::test_suite("slow")) {
  for (double mu : {0.5, 1.0, 10.0, 100.0})
    for (double alpha : {0.1, 0.3, 1.0, 3.0}) {
      const ld::pig_distrib d{mu, alpha};
      const auto sweep = ld::expected_info_by_integral<0, ld::info_form::score>(d);
      if (!(sweep[2] > 0.0)) continue;  // the sweep gives up at the heavy-tailed end
      const double v = d.expected_info()[2];
      INFO("mu " << mu << ", alpha " << alpha << ", spread " << d.scale());
      CHECK(v == doctest::Approx(sweep[2]).epsilon(1e-9));
    }
}

TEST_CASE("distrib pig: every order against mpmath") {
  // tools/reference/distrib_pig.py: kind 0 log p, 1 log F, 2 log(1 - F); b of the K derivatives
  // taken in alpha, K - b in mu. The error is the library value minus the mpmath value (low part
  // removed), in ulp of the largest component of the same (kind, y, mu, alpha, K) group: every
  // tensor of a given order has genuine zeros in y and in the parameters (the score in mu is
  // exactly zero at y = mu), so an error is measured against the scale of the order.
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_pig.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  struct row { int kind, K, b; double y, mu, alpha, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], mus[64], als[64], his[64], los[64];
    if (std::sscanf(line, "%d,%63[^,],%63[^,],%63[^,],%d,%d,%63[^,],%63s", &r.kind, ys, mus, als, &r.K, &r.b, his,
                    los) != 8)
      continue;
    r.y = std::strtod(ys, nullptr), r.mu = std::strtod(mus, nullptr), r.alpha = std::strtod(als, nullptr);
    r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 1000);
  std::map<std::tuple<int, double, double, double, int>, double> scale_of;
  for (const auto& r : rows) {
    auto& s = scale_of[std::make_tuple(r.kind, r.y, r.mu, r.alpha, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double worst[3][7] = {};
  double worst_y[3][7] = {}, worst_mu[3][7] = {}, worst_al[3][7] = {};
  int worst_b[3][7] = {};
  // Bounds: 1.5 times the worst measured on the gcc (Serial + OpenMP) and the CUDA build
  // (2026-11-15, LAPLACE_PRINT_ERRORS), per order. The reference file holds the mass only (kind 0),
  // since the cumulative functions of this family have no closed form and every reference value
  // would be a numerical derivative of a sum of Bessel functions; the generic summation route that
  // computes them here is measured in its own section, and against differences above.
  // The order-1 bound is the one wide one, and now only below the split: above alpha = mu the
  // derivative in alpha is taken as -mu^2 / (s (s + alpha)) + (kappa_1 - y) / alpha + y mu /
  // (alpha s), where the cancellation of the first cumulant against y alpha / (s (mu + s)) is gone
  // (2.6 ulp where that form had 1577); below it the same form divides kappa_1 - y, of the size of
  // y there, by a small alpha, so the sum's own form is kept and its worst is 74.4 ulp at y = 949,
  // mu = 100, alpha = 5.
  const double bound[3][7] = {{10.0, 112.0, 10.0, 16.0, 30.0, 150.0, 310.0},
                              {40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0},
                              {40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0}};
  for (const auto& r : rows) {
    const ld::pig_distrib d{r.mu, r.alpha};
    const double v = r.kind == 0 ? (r.K == 0 ? d.logpmf(r.y) : lderiv(d, r.y, r.K)[r.b])
                                 : cderiv_sum(d, r.y, r.K, r.kind == 2)[r.b];
    const double sc = scale_of[std::make_tuple(r.kind, r.y, r.mu, r.alpha, r.K)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    if (ulp > worst[r.kind][r.K])
      worst_y[r.kind][r.K] = r.y, worst_mu[r.kind][r.K] = r.mu, worst_al[r.kind][r.K] = r.alpha,
      worst_b[r.kind][r.K] = r.b;
    worst[r.kind][r.K] = std::fmax(worst[r.kind][r.K], ulp);
    INFO("kind " << r.kind << " y " << r.y << " mu " << r.mu << " alpha " << r.alpha << " K " << r.K << " b " << r.b
                 << " ulp " << ulp);
    CHECK(ulp <= bound[r.kind][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int kd = 0; kd < 3; ++kd)
      for (int K = 0; K <= 6; ++K)
        std::printf("pig kind %d K %d: max %.3g ulp (y %.6g, mu %.6g, alpha %.6g, b %d)\n", kd, K, worst[kd][K],
                    worst_y[kd][K], worst_mu[kd][K], worst_al[kd][K], worst_b[kd][K]);
}
