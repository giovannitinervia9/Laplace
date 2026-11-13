// The distrib pilot (gaussian1, laplace1, poisson, gamma1; note/distrib.md): packing of the
// symmetric tensors, every derivative order against a 5-point difference of the order
// below, the expected information against the variance of the score (quadrature with
// a knot at the kink, or a sum over the support) and its derivatives against
// differences, the cdf family against identities, and a kernel launch of every order.
#include <doctest/doctest.h>

#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/distrib/poisson.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

TEST_CASE("distrib: packing of symmetric tensors") {
  for (int P = 1; P <= 4; ++P)
    for (int K = 1; K <= 6; ++K) {
      const auto all = sorted_indices(P, K);
      CHECK((int)all.size() == ld::sym_tensor_size(P, K));
    }
  // positions of every sorted index of (P, K) = (3, 3) and (4, 2) agree with the enumeration
  const auto a33 = sorted_indices(3, 3);
  for (int i = 0; i < (int)a33.size(); ++i) {
    const int idx[3] = {a33[i][0], a33[i][1], a33[i][2]};
    CHECK(ld::sym_tensor_position<3, 3>(idx) == i);
  }
  const auto a42 = sorted_indices(4, 2);
  for (int i = 0; i < (int)a42.size(); ++i) {
    const int idx[2] = {a42[i][0], a42[i][1]};
    CHECK(ld::sym_tensor_position<4, 2>(idx) == i);
  }
  // two parameters: position = number of indices equal to 1
  const int i2[4] = {0, 1, 1, 1};
  CHECK(ld::sym_tensor_position<2, 4>(i2) == 3);
}

TEST_CASE("distrib: descriptors") {
  CHECK(ld::gaussian1_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::poisson_distrib::kind == ld::distrib_kind::discrete);
  CHECK(ld::gaussian1_distrib::params[0].regularity == ld::infinitely_smooth);
  CHECK(ld::laplace1_distrib::params[0].regularity == 0);
  CHECK(ld::laplace1_distrib::params[1].regularity == ld::infinitely_smooth);
  CHECK(ld::laplace1_distrib::params[0].cdf_regularity == 1);
  CHECK(ld::gaussian1_distrib::params[1].cdf_regularity == ld::infinitely_smooth);
  const ld::laplace1_distrib d{0.5, 2.0};
  CHECK(d.param_kinks(0, 1.25).count == 1);
  CHECK(d.param_kinks(0, 1.25).point[0] == 1.25);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.y_kinks().point[0] == 0.5);
}

TEST_CASE("distrib gaussian1: derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, 1e-8);
  check_derivs<ld::gaussian1_distrib>({-20.0, 0.7}, -18.5, 1e-8);
}

TEST_CASE("distrib laplace: derivatives against differences, away from the kink") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_derivs<ld::laplace1_distrib>({0.4, 1.3}, y, 1e-8);
  // at the kink: the mean of the one-sided derivatives in mu
  const ld::laplace1_distrib d{0.4, 1.3};
  CHECK(d.logpdf_d1(0.4)[0] == 0.0);
}

TEST_CASE("distrib poisson: derivatives against differences") {
  for (double y : {0.0, 1.0, 4.0, 30.0}) check_derivs<ld::poisson_distrib>({3.5}, y, 1e-8);
  check_derivs<ld::poisson_distrib>({0.2}, 2.0, 1e-8);
}

TEST_CASE("distrib gamma1: derivatives against differences") {
  for (double y : {0.05, 0.9, 2.0, 2.1, 7.0}) check_derivs<ld::gamma1_distrib>({2.0, 0.3}, y, 1e-8);
  check_derivs<ld::gamma1_distrib>({1.5, 0.02}, 1.4, 1e-8);  // shape 50: asymptotic series
  check_derivs<ld::gamma1_distrib>({1.5, 4.0}, 0.3, 1e-8);   // shape 1/4: direct route
}

TEST_CASE("distrib: expected information against the variance of the score") {
  for (double phi : {0.02, 0.3, 4.0}) {
    const ld::gamma1_distrib d{2.0, phi};
    // tanh-sinh on [0, mu] (the integrable singularity y^(s-1) at 0), exp-sinh beyond
    std::vector<double> q;
    for (int a = 0; a < 2; ++a)
      for (int b = a; b < 2; ++b) {
        auto f = [&](double y) {
          const auto sc = d.logpdf_d1(y);
          return sc[a] * sc[b] * std::exp(d.logpdf(y));
        };
        q.push_back(laplace::math::quad::tanh_sinh_quad(f, 0.0, 2.0).value +
                    laplace::math::quad::exp_sinh_quad(f, 2.0, 2.0 * (1.0 + phi)).value);
      }
    const auto e = d.expected_info();
    INFO("gamma1, phi = " << phi);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-12 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-12 * std::sqrt(e[0] * e[2]));
    CHECK(std::fabs(q[2] - e[2]) <= 1e-10 * e[2]);
  }
  {
    const ld::gaussian1_distrib d{0.4, 1.3};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-12 * e[2]);
  }
  {
    // the knot at the kink y = mu; -E[l_mumu] would be 0 here
    const ld::laplace1_distrib d{0.4, 1.3};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    for (int i = 0; i < 3; ++i) CHECK(std::fabs(q[i] - e[i]) <= 1e-12 * e[2]);
  }
  for (double mu : {0.05, 3.5, 40.0}) {
    const ld::poisson_distrib d{mu};
    double s = 0.0;
    for (int y = 0; y < 400; ++y) {
      const double sc = d.logpmf_d1(y)[0];
      s += sc * sc * std::exp(d.logpmf(y));
    }
    CHECK(std::fabs(s - d.expected_info()[0]) <= 1e-13 * d.expected_info()[0]);
  }
}

TEST_CASE("distrib: derivatives of the expected information against differences") {
  check_info_derivs<ld::gaussian1_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::laplace1_distrib>({0.4, 1.3}, 1e-8);
  check_info_derivs<ld::poisson_distrib>({3.5}, 1e-8);
  for (double phi : {0.02, 0.3, 4.0}) check_info_derivs<ld::gamma1_distrib>({2.0, phi}, 1e-8);
}

TEST_CASE("distrib: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {-12.0, -3.0, 0.2, 1.7, 9.0, 20.0})
      check_cdf_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
    for (double y : {-6.0, -0.5, 0.3, 1.7, 9.0})
      check_cdf_derivs<ld::laplace1_distrib>({0.4, 1.3}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
    for (double y : {0.0, 2.0, 9.0, 10.0, 11.0, 30.0}) check_cdf_derivs<ld::poisson_distrib>({10.0}, y, upper, 1e-7, 1.0);
    for (double phi : {0.05, 0.3, 4.0})
      for (double y : {0.05, 0.8, 2.0, 9.0}) check_cdf_derivs<ld::gamma1_distrib>({2.0, phi}, y, upper, 1e-7, 1.0);
  }
}

TEST_CASE("distrib: derivatives in y and mixed (y, theta) against differences") {
  // J = 0 is logpdf_dk, the order K = 0 the value
  const ld::gamma1_distrib g{2.0, 0.3};
  CHECK(g.logpdf_dy_dk<0, 0>(1.3)[0] == g.logpdf(1.3));
  CHECK(g.logpdf_dy_dk<0, 3>(1.3)[2] == g.logpdf_d3(1.3)[2]);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::gaussian1_distrib>({0.4, 1.3}, y, 1e-3, 1e-8);
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs<ld::laplace1_distrib>({0.4, 1.3}, y, 1e-3, 1e-8);
  for (double phi : {0.02, 0.3, 4.0})
    for (double y : {0.3, 1.4, 2.0, 5.0}) check_y_derivs<ld::gamma1_distrib>({2.0, phi}, y, 1e-4 * y, 1e-7);
  CHECK(ld::gaussian1_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::laplace1_distrib::y_regularity == 0);
}

TEST_CASE("distrib: cdf, ccdf, logarithms, quantile") {
  const ld::gaussian1_distrib g{0.4, 1.3};
  const ld::laplace1_distrib l{0.4, 1.3};
  for (double y : {-30.0, -2.0, 0.4, 1.0, 5.0, 30.0}) {
    CHECK(g.cdf(y) == doctest::Approx(0.5 * std::erfc(-(y - 0.4) / (1.3 * std::sqrt(2.0)))).epsilon(1e-14));
    CHECK(g.log_ccdf(y) == doctest::Approx(std::log(g.ccdf(y))).epsilon(1e-14));
    CHECK(l.cdf(y) + l.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(l.log_cdf(y) == doctest::Approx(std::log(l.cdf(y))).epsilon(1e-14));
    CHECK(l.log_ccdf(y) == doctest::Approx(std::log(l.ccdf(y))).epsilon(1e-14));
    if (std::fabs(y) < 10.0) {
      CHECK(g.quantile(g.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
      CHECK(l.quantile(l.cdf(y)) == doctest::Approx(y).epsilon(1e-12));
    }
  }
  const ld::gamma1_distrib ga{2.0, 0.3};
  for (double y : {0.01, 0.5, 2.0, 6.0, 20.0}) {
    CHECK(ga.cdf(y) + ga.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(ga.log_cdf(y) == doctest::Approx(std::log(ga.cdf(y))).epsilon(1e-13));
    if (y < 10.0) CHECK(ga.quantile(ga.cdf(y)) == doctest::Approx(y).epsilon(1e-11));
  }
  // logpdf matches the textbook form, also where (y - mu)/mu rounds to -1
  const ld::gamma1_distrib gs{2.0, 4.0};
  for (double y : {1e-30, 1e-5, 9.0})
    CHECK(gs.logpdf(y) == doctest::Approx(-std::lgamma(0.25) - 0.25 * std::log(8.0) - 0.75 * std::log(y) - y / 8.0).epsilon(1e-14));
  CHECK(ga.logpdf(1.3) == doctest::Approx(-std::lgamma(1 / 0.3) - (1 / 0.3) * std::log(2.0 * 0.3) +
                                           (1 / 0.3 - 1) * std::log(1.3) - 1.3 / (2.0 * 0.3)).epsilon(1e-14));
  // the saddle-point log-mass against the textbook form, where the latter does not cancel
  for (double y : {0.0, 1.0, 3.0, 12.0})
    CHECK(ld::poisson_distrib{3.5}.logpmf(y) ==
          doctest::Approx(y * std::log(3.5) - 3.5 - std::lgamma(y + 1.0)).epsilon(1e-14));
  // mu = y = 1000 (mpmath, 200 bits: -4.3728995060262968242), where the textbook form cancels
  CHECK(ld::poisson_distrib{1000.0}.logpmf(1000.0) == doctest::Approx(-4.3728995060262968242).epsilon(4e-16));
  const ld::poisson_distrib p{3.5};
  double F = 0.0;
  for (int k = 0; k < 20; ++k) {
    F += std::exp(p.logpmf(k));
    CHECK(p.cdf(k) == doctest::Approx(F).epsilon(1e-13));
    CHECK(p.cdf(k + 0.5) == p.cdf(k));
    CHECK(p.cdf(k) + p.ccdf(k) == doctest::Approx(1.0).epsilon(1e-15));
    CHECK(p.log_cdf(k) == doctest::Approx(std::log(p.cdf(k))).epsilon(1e-13));
  }
  CHECK(p.cdf(-1.0) == 0.0);
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why
// (doctest gives each case an internal-linkage function whose number restarts per
// translation unit, and nvcc tags every extended lambda with that address).
void pilot_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_pilot", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::gaussian1_distrib g{0.4, 1.3};
        const ld::laplace1_distrib l{0.4, 1.3};
        const ld::poisson_distrib p{3.5};
        const ld::gamma1_distrib ga{2.0, 0.3};
        out(i) = ga.logpdf_d6(0.5 + 0.1 * i)[2] + ga.logpdf_dy_dk<2, 3>(0.5 + 0.1 * i)[1] + g.logpdf_dy_dk<1, 5>(y)[2] + l.logpdf_dy_dk<1, 2>(y)[1] + ga.expected_info_d4().pair[2][4] + g.logpdf_d6(y)[3] + l.logpdf_d6(y)[6] + p.logpmf_d6(double(i % 7))[0] + g.expected_info_d4().pair[2][4] +
                 l.expected_info_d4().pair[0][4] + p.expected_info_d4().pair[0][0] + g.log_cdf(y) + l.log_ccdf(y) +
                 p.log_cdf(double(i % 7));
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::gaussian1_distrib g{0.4, 1.3};
    const ld::laplace1_distrib l{0.4, 1.3};
    const ld::poisson_distrib p{3.5};
    const ld::gamma1_distrib ga{2.0, 0.3};
    const double r = ga.logpdf_d6(0.5 + 0.1 * i)[2] + ga.logpdf_dy_dk<2, 3>(0.5 + 0.1 * i)[1] + g.logpdf_dy_dk<1, 5>(y)[2] + l.logpdf_dy_dk<1, 2>(y)[1] + ga.expected_info_d4().pair[2][4] + g.logpdf_d6(y)[3] + l.logpdf_d6(y)[6] + p.logpmf_d6(double(i % 7))[0] + g.expected_info_d4().pair[2][4] +
                     l.expected_info_d4().pair[0][4] + p.expected_info_d4().pair[0][0] + g.log_cdf(y) + l.log_ccdf(y) +
                     p.log_cdf(double(i % 7));
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-14));
  }
}

TEST_CASE("distrib: every order inside a kernel") { pilot_kernel_case(); }


TEST_CASE("distrib: derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_log_cdf.py; the error of each component in ulp of the largest
  // component of its order at that y (the components change sign, so a plain relative error
  // is undefined at their zeros), and in ulp of the component itself
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_log_cdf.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo, mu, phi; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64];
    char mus[64], phis[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],%63[^,],%63s", &r.dist, &r.upper, ys, &r.K, &r.b, his, los, mus,
                    phis) != 9)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.phi = std::strtod(phis, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 900);
  double worst_scaled[4][2] = {}, worst_own[4][2] = {};
  for (const auto& r : rows) {
    std::vector<double> v;
    if (r.dist == 0) v = cderiv(ld::gaussian1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else if (r.dist == 1) v = cderiv(ld::laplace1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else if (r.dist == 2) v = cderiv(ld::poisson_distrib{r.mu}, r.y, r.K, r.upper);
    else v = cderiv(ld::gamma1_distrib{r.mu, r.phi}, r.y, r.K, r.upper);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.dist == r.dist && q.upper == r.upper && q.y == r.y && q.K == r.K && q.mu == r.mu && q.phi == r.phi) scale = std::fmax(scale, std::fabs(q.hi));
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(scale) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    worst_scaled[r.dist][r.upper] = std::fmax(worst_scaled[r.dist][r.upper], ulp_scaled);
    worst_own[r.dist][r.upper] = std::fmax(worst_own[r.dist][r.upper], ulp_own);
    INFO("dist " << r.dist << " upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " ulp " << ulp_scaled);
    if (std::getenv("LAPLACE_PRINT_ERRORS") && ulp_own > 200)
      std::printf("dist %d upper %d y %g K %d b %d: %.3g ulp own, %.3g scaled, value %.3g\n", r.dist, r.upper, r.y, r.K, r.b, ulp_own, ulp_scaled, r.hi);
    // measured 2026-11-12 (gcc): 30.9, 29.7 (gaussian1), 11.2, 9.86 (laplace; 57.2, 391 before
    // the series route for log(1 - e^(-|z|)/2)); bounds half again;
    // poisson (2026-11-12, evening): 3.84e3 (k = 39 at mu = 40, order 6, value 4.8e-7 next to
    // 2e-6), 867 (log(1 - F), after Temme's repair near c = 1, 2026-11-12 night); gamma1, composed
    // from the partials at fixed ratio (incomplete_gamma_ratio.hpp): 379, 356
    const double bound[4][2] = {{46.4, 44.6}, {16.8, 14.8}, {5.8e3, 1.3e3}, {570.0, 540.0}};
    CHECK(ulp_scaled <= bound[r.dist][r.upper]);
  }
  for (int d = 0; d < 4; ++d)
    for (int u = 0; u < 2; ++u)
      if (std::getenv("LAPLACE_PRINT_ERRORS"))
        std::printf("log_cdf derivs dist %d upper %d: max %.3g ulp of the order's scale, %.3g ulp of the component\n", d, u,
                    worst_scaled[d][u], worst_own[d][u]);
}

TEST_CASE("distrib: derivatives of log F and log(1 - F) in the tails against mpmath") {
  // tools/reference/distrib_log_cdf_tails.py: gaussian1 where log Phi is close to -z^2/2,
  // laplace where log(1 - F) or log F is log(1 - e^(-|z|)/2); errors as in the case above,
  // in ulp of the largest component of the order at that y and in ulp of the component
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_log_cdf_tails.csv", "r");
  REQUIRE(f != nullptr);
  char line[256];
  std::fgets(line, sizeof line, f);
  struct row { int dist, upper, K, b; double y, hi, lo; };
  std::vector<row> rows;
  while (std::fgets(line, sizeof line, f)) {
    row r;
    char ys[64], his[64], los[64];
    if (std::sscanf(line, "%d,%d,%63[^,],%d,%d,%63[^,],%63[^,],", &r.dist, &r.upper, ys, &r.K, &r.b, his, los) != 7) continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  CHECK(rows.size() > 1500);
  // worst errors per distribution, side, order and component (b), and the y of the worst own error
  double w_scaled[2][2][7][7] = {}, w_own[2][2][7][7] = {}, w_y[2][2][7][7] = {};
  for (const auto& r : rows) {
    std::vector<double> v;
    if (r.dist == 0) v = cderiv(ld::gaussian1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    else v = cderiv(ld::laplace1_distrib{0.4, 1.3}, r.y, r.K, r.upper);
    double scale = 0.0;
    for (const auto& q : rows)
      if (q.dist == r.dist && q.upper == r.upper && q.y == r.y && q.K == r.K) scale = std::fmax(scale, std::fabs(q.hi));
    const double err = std::fabs((v[r.b] - r.hi) - r.lo);
    const double ulp_scaled = err / (std::ldexp(1.0, std::ilogb(scale) - 52));
    const double ulp_own = err / (std::ldexp(1.0, std::ilogb(r.hi) - 52));
    w_scaled[r.dist][r.upper][r.K][r.b] = std::fmax(w_scaled[r.dist][r.upper][r.K][r.b], ulp_scaled);
    if (ulp_own > w_own[r.dist][r.upper][r.K][r.b]) w_own[r.dist][r.upper][r.K][r.b] = ulp_own, w_y[r.dist][r.upper][r.K][r.b] = r.y;
    INFO("dist " << r.dist << " upper " << r.upper << " y " << r.y << " K " << r.K << " b " << r.b << " ulp " << ulp_scaled << " own " << ulp_own);
    // measured 2026-11-12 (gcc), scaled: 4.81, 4.38 (gaussian1), 16.7, 6.05 (laplace); own: 1.46e3,
    // 1.48e3 (gaussian1, order 6 at z = -7 and 7, where the component has a zero next to it at
    // |z| = 7.3), 53.9, 35.2 (laplace); before the tail routes the own errors were 1.57e5, 1.57e5,
    // 986, 921; bounds half again
    const double bound_scaled[2][2] = {{7.3, 6.6}, {25.0, 9.1}};
    const double bound_own[2][2] = {{2.2e3, 2.2e3}, {81.0, 53.0}};
    CHECK(ulp_scaled <= bound_scaled[r.dist][r.upper]);
    CHECK(ulp_own <= bound_own[r.dist][r.upper]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int d = 0; d < 2; ++d)
      for (int u = 0; u < 2; ++u)
        for (int K = 1; K <= 6; ++K) {
          double ms = 0.0, mo = 0.0;
          std::printf("tails dist %d upper %d K %d, own ulp (scaled ulp) per b:", d, u, K);
          for (int b = 0; b <= K; ++b) {
            std::printf(" %.3g (%.3g) [y %.4g]", w_own[d][u][K][b], w_scaled[d][u][K][b], w_y[d][u][K][b]);
            ms = std::fmax(ms, w_scaled[d][u][K][b]), mo = std::fmax(mo, w_own[d][u][K][b]);
          }
          std::printf("; order: %.3g own, %.3g scaled\n", mo, ms);
        }
}

TEST_CASE("distrib: the power-scale chain against the sigma chain") {
  // mixed_power_dk with sigma = w^1, w = sigma is mixed_dk: the same derivatives
  // reached by applying the two rules in the other order (note the comment in
  // detail/location_scale.hpp).
  namespace dls = laplace::distrib::detail_location_scale;
  double L[7];
  for (int j = 1; j <= 6; ++j) L[j] = 0.37 * j - 0.11 * j * j;
  for (double z : {-2.5, -0.3, 0.8, 4.0})
    for (double sigma : {0.25, 1.3, 7.0}) {
      const auto a = dls::mixed_dk<6>(L, z, sigma);
      const auto b = dls::mixed_power_dk<6>(L, z, sigma, sigma, 1.0);
      double scale = 0.0;
      for (int i = 0; i <= 6; ++i) scale = std::fmax(scale, std::fabs(a[i]));
      for (int i = 0; i <= 6; ++i) {
        INFO("z = " << z << ", sigma = " << sigma << ", component " << i);
        CHECK(std::fabs(a[i] - b[i]) <= 1e-12 * scale);
      }
    }
}
