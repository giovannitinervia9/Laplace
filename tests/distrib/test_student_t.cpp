// The tests of student_t1_distrib and student_t2_distrib (include/laplace/distrib/student_t1.hpp,
// student_t2.hpp): packing and descriptors, every derivative order against differences, the mixed
// (y, theta) derivatives, the expected information against the variance of the score and against
// the generic quadrature route, the derivatives of the information against differences, the
// derivatives of log F and log(1 - F) against differences, the twins (student_t1 at nu = 1 against
// cauchy_distrib, at nu = 1e8 and 1e12 against gaussian1_distrib, student_t2 against student_t1 at the
// mapped scale), the generic quantile, the references of tools/reference/distrib_student_t.py (the
// log-density, the information and log F in (mu, sigma, nu)), and a kernel launch.
#include <doctest/doctest.h>

// Worst relative difference between the host and the device sum of the kernel case, measured on
// the CUDA build (RTX 4070 SUPER) on 2026-11-15: 8.7e-15; the bound is half again.
#define STUDENT_KERNEL_TOL 1.3e-14

#include <laplace/distrib/cauchy.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/distrib/student_t2.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"

namespace laplace_distrib_check {
// the shared checks build a distribution from a parameter vector, two parameters at most by default
template <>
ld::student_t1_distrib make<ld::student_t1_distrib>(const double* t) {
  return ld::student_t1_distrib{t[0], t[1], t[2]};
}
template <>
ld::student_t2_distrib make<ld::student_t2_distrib>(const double* t) {
  return ld::student_t2_distrib{t[0], t[1], t[2]};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included
using laplace_test::hide;
using ld::detail_student_t1::pos3;

namespace {

using t1 = ld::student_t1_distrib;
using t2 = ld::student_t2_distrib;

// ---------------------------------------------------------------- the three-parameter y-derivatives

#define Y3(J, K) \
  case J * 10 + K: return as_vec<3, K>(d.template logpdf_dy_dk<J, K>(y))
template <class D>
std::vector<double> yderiv3(const D& d, double y, int J, int K) {
  switch (J * 10 + K) {
    Y3(0, 0); Y3(0, 1); Y3(0, 2); Y3(0, 3); Y3(0, 4); Y3(0, 5); Y3(0, 6);
    Y3(1, 0); Y3(1, 1); Y3(1, 2); Y3(1, 3); Y3(1, 4); Y3(1, 5);
    Y3(2, 0); Y3(2, 1); Y3(2, 2); Y3(2, 3); Y3(2, 4);
    Y3(3, 0); Y3(3, 1); Y3(3, 2); Y3(3, 3);
    Y3(4, 0); Y3(4, 1); Y3(4, 2);
    Y3(5, 0); Y3(5, 1);
    Y3(6, 0);
    default: return {};
  }
}
#undef Y3

// every (J, K): against the difference in y of (J - 1, K) and in each parameter of (J, K - 1)
template <class D>
void check_y_derivs3(const std::vector<double>& th, double y, double hy, double rel) {
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv3(make<D>(th.data()), y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yderiv3(make<D>(th.data()), y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12);
      }
      if (K == 0) continue;
      for (const auto& idx : sorted_indices(3, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const auto fd = diff5([&](const double* t) { return yderiv3(make<D>(t), y, J, K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(3, idx) << ", y = " << y);
        CHECK(std::fabs(exact[position_of(3, idx)] - fd[position_of(3, lower)]) <= rel * scale + 1e-12);
      }
    }
}

// ---------------------------------------------------------------- the generic information route

template <int P, int K, class T>
std::vector<double> info_components(const T& t) {
  std::vector<double> r;
  if constexpr (K == 0) {
    for (int i = 0; i < ld::sym_tensor_size(P, 2); ++i) r.push_back(t[i]);
  } else {
    for (int p = 0; p < ld::sym_tensor_size(P, 2); ++p)
      for (int i = 0; i < ld::sym_tensor_size(P, K); ++i) r.push_back(t.pair[p][i]);
  }
  return r;
}

// the generic quadrature route (score form) against the closed form: the largest difference relative to
// the largest closed-form component of that order
template <int K, class D>
double info_by_integral_error(const D& d) {
  std::vector<double> ex;
  if constexpr (K == 0) ex = info_components<3, 0>(d.expected_info());
  else ex = info_components<3, K>(d.template expected_info_dk<K>());
  const auto gen = info_components<3, K>(ld::expected_info_by_integral<K, ld::info_form::score>(hide(d)));
  double scale = 0.0;
  for (double e : ex) scale = std::fmax(scale, std::fabs(e));
  double worst = 0.0;
  for (size_t i = 0; i < ex.size(); ++i) {
    if (std::isnan(gen[i])) return INFINITY;
    worst = std::fmax(worst, std::fabs(gen[i] - ex[i]) / scale);
  }
  return worst;
}

// ---------------------------------------------------------------- the references

struct ref_row {
  int fam = 0, upper = 0, K = 0, a = 0, b = 0, c = 0, pair = 0;
  double y = 0, hi = 0, lo = 0, mu = 0, sigma = 0, nu = 0;
};

std::vector<std::string> split_csv(const char* line) {
  std::vector<std::string> f;
  std::string cur;
  for (const char* p = line; *p && *p != '\n' && *p != '\r'; ++p) {
    if (*p == ',') {
      f.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(*p);
    }
  }
  f.push_back(cur);
  return f;
}

// kind 0: fam,y,K,a,b,c,hi,lo,mu,sigma,nu; 1: fam,upper,y,K,a,b,c,hi,lo,mu,sigma,nu; 2: fam,K,pair,b,c,hi,lo,mu,sigma,nu
std::vector<ref_row> read_reference(const char* name, int kind) {
  std::vector<ref_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    ref_row r;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    auto in = [&](int i) { return std::atoi(s[i].c_str()); };
    if (kind == 0 && s.size() == 11) {
      r.fam = in(0), r.y = num(1), r.K = in(2), r.a = in(3), r.b = in(4), r.c = in(5), r.hi = num(6), r.lo = num(7);
      r.mu = num(8), r.sigma = num(9), r.nu = num(10);
    } else if (kind == 1 && s.size() == 12) {
      r.fam = in(0), r.upper = in(1), r.y = num(2), r.K = in(3), r.a = in(4), r.b = in(5), r.c = in(6);
      r.hi = num(7), r.lo = num(8), r.mu = num(9), r.sigma = num(10), r.nu = num(11);
    } else if (kind == 2 && s.size() == 10) {
      r.fam = in(0), r.K = in(1), r.pair = in(2), r.b = in(3), r.c = in(4), r.hi = num(5), r.lo = num(6);
      r.mu = num(7), r.sigma = num(8), r.nu = num(9);
    } else {
      continue;
    }
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// the tensor of order K of the log-density (cdf: of log F or log(1 - F)) of either family
std::vector<double> lp_tensor(const ref_row& r, int K) {
  return r.fam == 1 ? lderiv(t1{r.mu, r.sigma, r.nu}, r.y, K) : lderiv(t2{r.mu, r.sigma, r.nu}, r.y, K);
}
std::vector<double> cdf_tensor(const ref_row& r, int K) {
  return r.fam == 1 ? cderiv(t1{r.mu, r.sigma, r.nu}, r.y, K, r.upper) : cderiv(t2{r.mu, r.sigma, r.nu}, r.y, K, r.upper);
}
std::vector<double> info_tensor(const ref_row& r, int K) {
  return r.fam == 1 ? ideriv(t1{r.mu, r.sigma, r.nu}, K) : ideriv(t2{r.mu, r.sigma, r.nu}, K);
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

// the tolerances below are the worst scaled errors measured on 2026-11-15 on the gcc build and on the CUDA
// build (RTX 4070 SUPER), times about one and a half; the first index is the family minus one (and, for log F, the
// second is 0 for log F and 1 for log(1 - F))
// the log-density bounds were re-measured on 2026-11-16, when the (mu, sigma) components with no nu
// derivative were given the closed form in Re((y - mu + i sigma sqrt(nu))^-K) for |y - mu| >= sigma sqrt(nu):
// t1 fell from 13.5, 31.7, 100, 260, 940 and 2.4e3 to 3.68, 2.35, 3.66, 5.1, 20.1 and 107, and t2, which
// composes t1, from 5.81, 9.28, 12.8, 55.1, 178 and 632 to 5.81, 6.87, 9.65, 103, 9.1 and 31.2. The one
// entry that grew is t2's order four, a pure nu component at z = 4, nu = 10: the chain rule of the
// scale s = sigma sqrt((nu - 2)/nu) mixes components whose last bits have changed.
const double BOUND_LOGPDF[2][7] = {{0, 6.0, 4.0, 6.0, 8.0, 31.0, 161.0}, {0, 9.0, 11.0, 15.0, 155.0, 14.0, 47.0}};
// The log F bounds grew on 2026-11-16, when the references gained 20 (point, side) pairs at nu >= 1e4
// that mpmath.diff had never agreed on (tools/reference/distrib_student_t.py, log_cdf_def). They are the
// first look at that region beyond |z| <= 4, and what they show is the cdf route's own accuracy there:
// over that region the orders one to six read 1.2e3, 962, 786, 1.23e3, 977 and 582 ulp of the order's
// scale for t1 (the order six of log F stays at 6450, from nu = 3.5) and 903, 725, 580, 925, 733 and 582
// for t2; every one of them is at z = 30, nu = 1e4 but the order three of t1, which is at nu = 1e8 at the
// same z and stays below that figure at nu = 1e4. Nothing moved at the points that were already
// covered: run against the previous file every bucket is the same number as before (nu = 1e4 read 66 for
// t1 and 101 for t2 because only |z| <= 4 was there), so the figures below are measured, not inherited.
const double BOUND_LOGCDF[2][2][7] = {{{0, 1800.0, 1450.0, 1180.0, 1850.0, 1470.0, 9700.0}, {0, 1800.0, 1450.0, 1180.0, 1850.0, 1470.0, 1210.0}},
                                      {{0, 1360.0, 1090.0, 870.0, 1390.0, 1100.0, 880.0}, {0, 1360.0, 1090.0, 870.0, 1390.0, 1100.0, 880.0}}};
const double BOUND_LOGCDF_NEAR2[7] = {0, 27.0, 2.2e5, 8.8e6, 2.4e7, 6.9e7, 1.95e8};
const double BOUND_BLOCK[2][7] = {{0, 1800.0, 1450.0, 1180.0, 1850.0, 1470.0, 9700.0}, {0, 1360.0, 1090.0, 870.0, 1390.0, 2280.0, 6360.0}};
const double BOUND_INFO[2][5] = {{240.0, 85.0, 76.0, 53.0, 26.0}, {480.0, 173.0, 150.0, 82.0, 52.0}};

}  // namespace

TEST_CASE("student_t: descriptors and kinks") {
  CHECK(ld::student_t1_distrib::kind == ld::distrib_kind::continuous);
  CHECK(ld::student_t1_distrib::n_params == 3);
  CHECK(ld::student_t2_distrib::n_params == 3);
  CHECK(ld::student_t1_distrib::y_regularity == ld::infinitely_smooth);
  CHECK(ld::student_t2_distrib::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 3; ++j) {
    CHECK(ld::student_t1_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::student_t1_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::student_t1_distrib::params[j].moves_support == false);
    CHECK(ld::student_t2_distrib::params[j].regularity == ld::infinitely_smooth);
    CHECK(ld::student_t2_distrib::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ld::student_t2_distrib::params[j].moves_support == false);
  }
  CHECK(ld::student_t1_distrib::params[1].lower == 0.0);
  CHECK(ld::student_t1_distrib::params[2].lower == 0.0);
  CHECK(ld::student_t2_distrib::params[2].lower == 2.0);
  const t1 d{0.4, 1.3, 3.5};
  const t2 e{0.4, 1.3, 3.5};
  for (int j = 0; j < 3; ++j) {
    CHECK(d.param_kinks(j, 1.25).count == 0);
    CHECK(e.param_kinks(j, 1.25).count == 0);
  }
  CHECK(d.y_kinks().count == 0);
  CHECK(e.y_kinks().count == 0);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 1.3);
  CHECK(e.center() == 0.4);
  CHECK(e.scale() == 1.3);
  // the density at the centre: c(nu) - log sigma, c(1) = -log(pi)
  const t1 c1{0.4, 1.3, 1.0};
  CHECK(c1.logpdf(0.4) == doctest::Approx(-std::log(3.14159265358979323846 * 1.3)).epsilon(1e-14));
  // the standard deviation of student_t2 is sigma: its density is student_t1's at the scale sigma sqrt((nu - 2)/nu)
  CHECK(e.t1_scale() == doctest::Approx(1.3 * std::sqrt(1.5 / 3.5)).epsilon(1e-15));
  CHECK(e.logpdf(1.1) == e.as_t1().logpdf(1.1));
  CHECK(e.cdf(1.1) == e.as_t1().cdf(1.1));
}

TEST_CASE("student_t: the density against the closed form") {
  // l = c(nu) - log sigma - (nu + 1)/2 log(1 + z^2/nu), c from lgamma
  for (double nu : {0.3, 1.0, 4.5, 60.0}) {
    const t1 d{0.4, 1.3, nu};
    for (double y : {-9.0, 0.4, 2.0, 30.0}) {
      const double z = (y - 0.4) / 1.3;
      const double c = std::lgamma(0.5 * (nu + 1.0)) - std::lgamma(0.5 * nu) - 0.5 * std::log(nu * 3.14159265358979323846);
      const double ref = c - std::log(1.3) - 0.5 * (nu + 1.0) * std::log1p(z * z / nu);
      CHECK(d.logpdf(y) == doctest::Approx(ref).epsilon(1e-12));
    }
  }
}

TEST_CASE("student_t: derivatives against differences, centre and tails") {
  for (double nu : {0.5, 4.5, 40.0})
    for (double y : {-40.0, -3.0, 0.2, 0.4, 1.7, 9.0, 300.0}) check_derivs<t1>({0.4, 1.3, nu}, y, 1e-7);
  for (double nu : {2.5, 5.0, 40.0})
    for (double y : {-40.0, -3.0, 0.2, 0.4, 1.7, 9.0, 300.0}) check_derivs<t2>({0.4, 1.3, nu}, y, 1e-7);
}

TEST_CASE("student_t: mixed (y, theta) derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) {
    check_y_derivs3<t1>({0.4, 1.3, 4.5}, y, 1e-3, 1e-6);
    check_y_derivs3<t2>({0.4, 1.3, 4.5}, y, 1e-3, 1e-6);
  }
  // the J = 1 route equals the minus of the mu-derivative of the same order
  const t1 d{0.4, 1.3, 4.5};
  const auto a = d.logpdf_dy_dk<1, 0>(2.0);
  const auto b = d.logpdf_dk<1>(2.0);
  CHECK(a[0] == doctest::Approx(-b[0]).epsilon(1e-14));
  const auto a2 = d.logpdf_dy_dk<2, 2>(2.0);
  const auto b4 = d.logpdf_dk<4>(2.0);
  CHECK(a2[pos3(0, 1, 1)] == b4[pos3(2, 1, 1)]);
}

TEST_CASE("student_t: expected information against the variance of the score") {
  // measured 2026-11-15 on gcc: the largest relative difference of the entries is 7.1e-15 (the quadrature's floor)
  double worst_var = 0.0;
  for (const std::array<double, 3>& p : {std::array<double, 3>{0.4, 1.3, 4.5}, std::array<double, 3>{-2.0, 0.25, 12.0},
                                         std::array<double, 3>{1.0, 4.0, 2.7}}) {
    const t1 d{p[0], p[1], p[2]};
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("student_t1, nu = " << p[2]);
    const double s = std::fmax(e[0], std::fmax(e[3], std::fabs(e[4])));
    CHECK(std::fabs(q[0] - e[0]) <= 1e-12 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-12 * s);
    CHECK(std::fabs(q[2]) <= 1e-12 * s);
    CHECK(std::fabs(q[3] - e[3]) <= 1e-12 * e[3]);
    CHECK(std::fabs(q[4] - e[4]) <= 1e-12 * std::sqrt(e[3] * e[5]) * 10.0);
    CHECK(std::fabs(q[5] - e[5]) <= 1e-12 * e[5]);
    CHECK(e[1] == 0.0);
    CHECK(e[2] == 0.0);
    const t2 g{p[0], p[1], p[2]};
    const auto q2 = score_variance_continuous(g);
    const auto e2 = g.expected_info();
    INFO("student_t2, nu = " << p[2]);
    CHECK(std::fabs(q2[0] - e2[0]) <= 1e-12 * e2[0]);
    CHECK(std::fabs(q2[3] - e2[3]) <= 1e-12 * e2[3]);
    CHECK(std::fabs(q2[4] - e2[4]) <= 1e-12 * std::sqrt(e2[3] * e2[5]) * 10.0);
    CHECK(std::fabs(q2[5] - e2[5]) <= 1e-12 * e2[5]);
    CHECK(e2[1] == 0.0);
    CHECK(e2[2] == 0.0);
    worst_var = std::fmax(worst_var, std::fmax(std::fmax(std::fabs(q[0] - e[0]) / e[0], std::fabs(q[3] - e[3]) / e[3]),
                                               std::fmax(std::fabs(q[5] - e[5]) / e[5], std::fabs(q2[5] - e2[5]) / e2[5])));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("student_t information against the variance of the score: worst relative difference %.3g%s", worst_var, "\n");
}

TEST_CASE("student_t: expected information by the generic quadrature route") {
  // the closed forms and their derivatives against the hidden distribution (the score form, every order 0 to 4):
  // measured 2026-11-15 on gcc at nu = 3.5, 9 and 30, both families: at most 1.1e-15 of the largest component of
  // the order; the bound is ten times that, since the quadrature (not the closed form) sets the floor
  double worst = 0.0;
  for (double nu : {3.5, 9.0, 30.0}) {
    const t1 d{0.4, 1.3, nu};
    const t2 e{0.4, 1.3, nu};
    INFO("nu = " << nu);
    const double w[10] = {info_by_integral_error<0>(d), info_by_integral_error<1>(d), info_by_integral_error<2>(d),
                          info_by_integral_error<3>(d), info_by_integral_error<4>(d), info_by_integral_error<0>(e),
                          info_by_integral_error<1>(e), info_by_integral_error<2>(e), info_by_integral_error<3>(e),
                          info_by_integral_error<4>(e)};
    for (double v : w) {
      worst = std::fmax(worst, v);
      CHECK(v <= 1e-14);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("student_t generic information route: worst %.3g of the largest component%s", worst, "\n");
}

TEST_CASE("student_t: derivatives of the expected information against differences") {
  check_info_derivs<t1>({0.4, 1.3, 4.5}, 1e-7);
  check_info_derivs<t1>({-2.0, 0.25, 0.7}, 1e-7);
  check_info_derivs<t1>({1.0, 4.0, 40.0}, 1e-7);
  check_info_derivs<t2>({0.4, 1.3, 4.5}, 1e-7);
  check_info_derivs<t2>({-2.0, 0.25, 2.7}, 1e-7);
  check_info_derivs<t2>({1.0, 4.0, 40.0}, 1e-7);
}

TEST_CASE("student_t: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    for (double y : {-40.0, -3.0, 0.2, 1.7, 9.0, 60.0}) {
      check_cdf_derivs<t1>({0.4, 1.3, 4.5}, y, upper, 1e-6, std::fabs(y - 0.4) / 1.3);
      check_cdf_derivs<t2>({0.4, 1.3, 4.5}, y, upper, 1e-6, std::fabs(y - 0.4) / 1.3);
    }
    check_cdf_derivs<t1>({0.4, 1.3, 0.8}, -3.0, upper, 1e-6, 2.6);
    check_cdf_derivs<t2>({0.4, 1.3, 2.6}, 2.0, upper, 1e-6, 1.2);
  }
}

TEST_CASE("student_t: cdf, ccdf, logarithms and the generic quantile") {
  for (double nu : {0.8, 3.5, 40.0}) {
    const t1 d{0.4, 1.3, nu};
    const t2 e{0.4, 1.3, nu + 2.0};
    for (double y : {-1e6, -40.0, -3.0, 0.4, 1.7, 9.0, 60.0, 1e6}) {
      CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
      CHECK(e.cdf(y) + e.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
      // log(1 - F) is log F at the reflected point
      CHECK(d.log_ccdf(y) == doctest::Approx(d.log_cdf(0.8 - y)).epsilon(1e-12));
      CHECK(e.log_ccdf(y) == doctest::Approx(e.log_cdf(0.8 - y)).epsilon(1e-12));
      if (d.cdf(y) < 0.5) CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
      if (d.ccdf(y) < 0.5) CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
    }
    CHECK(d.cdf(0.4) == doctest::Approx(0.5).epsilon(1e-15));
    for (double p : {1e-8, 1e-3, 0.3, 0.5, 0.8, 1.0 - 1e-8}) {
      const double q = ld::quantile_by_root(d, p);
      if (p <= 0.5) CHECK(d.cdf(q) == doctest::Approx(p).epsilon(1e-10));
      else CHECK(d.ccdf(q) == doctest::Approx(1.0 - p).epsilon(1e-10));
      const double q2 = ld::quantile_by_root(e, p);
      if (p <= 0.5) CHECK(e.cdf(q2) == doctest::Approx(p).epsilon(1e-10));
      else CHECK(e.ccdf(q2) == doctest::Approx(1.0 - p).epsilon(1e-10));
    }
  }
  // the symmetry of the density and of F
  const t1 d{0.4, 1.3, 3.5};
  CHECK(d.logpdf(0.4 + 2.0) == doctest::Approx(d.logpdf(0.4 - 2.0)).epsilon(1e-15));
  CHECK(d.cdf(0.4 + 2.0) + d.cdf(0.4 - 2.0) == doctest::Approx(1.0).epsilon(1e-15));
}

TEST_CASE("student_t: the t distribution function at known values") {
  // nu = 1 is the Cauchy, nu = 2 has F(z) = 1/2 + z/(2 sqrt(2 + z^2))
  const t1 c{0.0, 1.0, 1.0};
  CHECK(c.cdf(1.0) == doctest::Approx(0.75).epsilon(1e-14));
  const t1 b{0.0, 1.0, 2.0};
  for (double z : {-5.0, -0.7, 0.3, 2.0})
    CHECK(b.cdf(z) == doctest::Approx(0.5 + z / (2.0 * std::sqrt(2.0 + z * z))).epsilon(1e-13));
}

namespace {

// the components of a tensor of order K in (mu, sigma, nu) with no derivative in nu, against the Cauchy
// tensor of order K in (mu, sigma): the same law at nu = 1
template <int K>
double cauchy_twin_density(double mu, double sg, double y) {
  const t1 d{mu, sg, 1.0};
  const ld::cauchy_distrib c{mu, sg};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = c.logpdf_dk<K>(y);
  double scale = 0.0, worst = 0.0;
  for (int i = 0; i <= K; ++i) scale = std::fmax(scale, std::fabs(b[i]));
  for (int i = 0; i <= K; ++i) worst = std::fmax(worst, std::fabs(a[pos3(K - i, i, 0)] - b[i]) / scale);
  return worst;
}
template <int K>
double cauchy_twin_cdf(double mu, double sg, double y, bool upper) {
  const t1 d{mu, sg, 1.0};
  const ld::cauchy_distrib c{mu, sg};
  const auto a = upper ? d.log_ccdf_dk<K>(y) : d.log_cdf_dk<K>(y);
  const auto b = upper ? c.log_ccdf_dk<K>(y) : c.log_cdf_dk<K>(y);
  double scale = 0.0, worst = 0.0;
  for (int i = 0; i <= K; ++i) scale = std::fmax(scale, std::fabs(b[i]));
  for (int i = 0; i <= K; ++i) worst = std::fmax(worst, std::fabs(a[pos3(K - i, i, 0)] - b[i]) / scale);
  return worst;
}

}  // namespace

TEST_CASE("student_t1 at nu = 1 against cauchy_distrib, every order") {
  double w_dens = 0.0, w_cdf = 0.0, w_far = 0.0;
  for (double y : {-60.0, -4.0, -0.3, 0.4, 1.9, 7.0, 90.0}) {
    w_dens = std::fmax(w_dens, cauchy_twin_density<1>(0.4, 1.3, y));
    w_dens = std::fmax(w_dens, cauchy_twin_density<2>(0.4, 1.3, y));
    w_dens = std::fmax(w_dens, cauchy_twin_density<3>(0.4, 1.3, y));
    w_dens = std::fmax(w_dens, cauchy_twin_density<4>(0.4, 1.3, y));
    w_dens = std::fmax(w_dens, cauchy_twin_density<5>(0.4, 1.3, y));
    w_dens = std::fmax(w_dens, cauchy_twin_density<6>(0.4, 1.3, y));
    for (bool upper : {false, true}) {
      double w = 0.0;
      w = std::fmax(w, cauchy_twin_cdf<1>(0.4, 1.3, y, upper));
      w = std::fmax(w, cauchy_twin_cdf<2>(0.4, 1.3, y, upper));
      w = std::fmax(w, cauchy_twin_cdf<3>(0.4, 1.3, y, upper));
      w = std::fmax(w, cauchy_twin_cdf<4>(0.4, 1.3, y, upper));
      w = std::fmax(w, cauchy_twin_cdf<5>(0.4, 1.3, y, upper));
      w = std::fmax(w, cauchy_twin_cdf<6>(0.4, 1.3, y, upper));
      (std::fabs(y - 0.4) / 1.3 > 20.0 ? w_far : w_cdf) = std::fmax(std::fabs(y - 0.4) / 1.3 > 20.0 ? w_far : w_cdf, w);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("student_t1 vs cauchy: density worst %.3g, log F and log(1 - F) worst %.3g for |z| < 20 and %.3g at |z| = 46 and 69 (relative to the largest component)\n",
                w_dens, w_cdf, w_far);
  CHECK(w_dens <= 1e-12);
  CHECK(w_cdf <= 1e-9);
  // Far in the tail where the function tends to 0 the Cauchy law is a power law, S = sigma/(pi (y - mu)) (1 + O(z^-2)),
  // whose sixth derivative in sigma vanishes: the Student route forms it from products of the derivatives of
  // log S and keeps an absolute error of about 2e-16 there, which the series in 1/z^2 of the Cauchy does not have
  // (the order-six component b = 6 of size 2.6e-11 is then right to 2e-6 relative to the largest component).
  CHECK(w_far <= 5e-6);
  // the information (mu, sigma) block and the value
  const t1 d{0.4, 1.3, 1.0};
  const ld::cauchy_distrib c{0.4, 1.3};
  CHECK(d.expected_info()[0] == doctest::Approx(c.expected_info()[0]).epsilon(1e-14));
  CHECK(d.expected_info()[3] == doctest::Approx(c.expected_info()[2]).epsilon(1e-14));
  CHECK(d.logpdf(2.0) == doctest::Approx(c.logpdf(2.0)).epsilon(1e-14));
  CHECK(d.cdf(2.0) == doctest::Approx(c.cdf(2.0)).epsilon(1e-13));
  CHECK(d.log_cdf(-30.0) == doctest::Approx(c.log_cdf(-30.0)).epsilon(1e-13));
  CHECK(d.log_ccdf(-30.0) == doctest::Approx(c.log_ccdf(-30.0)).epsilon(1e-13));
}

namespace {

// The difference between l_t and l_normal is (a polynomial in z of degree 4)/nu to first order in 1/nu, so a
// derivative of order K in (mu, sigma) differs by at most C (1 + z^2)^((4 + K)/2) / (sigma^K nu), and a
// derivative with c >= 1 indices in nu is at most C (1 + z^2)^((4 + K)/2) / (sigma^(K - c) nu^(c + 1)), both with
// the factor K! that the derivatives of a polynomial of degree 4 + K carry. The
// functions return the largest of the two ratios to those scales over the components of order K.
template <int K>
std::array<double, 2> gaussian_twin(double nu, double mu, double sg, double y) {
  const t1 d{mu, sg, nu};
  const ld::gaussian1_distrib g{mu, sg};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = g.logpdf_dk<K>(y);
  double kfact = 1.0;
  for (int i = 2; i <= K; ++i) kfact *= i;
  const double z = (y - mu) / sg, poly = kfact * std::pow(1.0 + z * z, 0.5 * (4.0 + K));
  double w_ms = 0.0, w_nu = 0.0;
  for (int i = 0; i <= K; ++i) w_ms = std::fmax(w_ms, std::fabs(a[pos3(K - i, i, 0)] - b[i]) * nu * std::pow(sg, K) / poly);
  for (int c = 1; c <= K; ++c)
    for (int b_ = 0; b_ + c <= K; ++b_)
      w_nu = std::fmax(w_nu, std::fabs(a[pos3(K - b_ - c, b_, c)]) * std::pow(nu, c + 1) * std::pow(sg, K - c) / poly);
  return {w_ms, w_nu};
}

}  // namespace

TEST_CASE("student_t1 at nu = 1e8 and 1e12 against gaussian1_distrib") {
  // The limit allows the differences stated above, with a constant C: the largest measured ratio is printed
  // with LAPLACE_PRINT_ERRORS, and the bound below is C = 1 (the measured ratios are the same at nu = 1e8
  // and nu = 1e12, which shows that no digit is lost to the cancellation of the two parts of Lambda as nu grows).
  double w_ms = 0.0, w_nu = 0.0, w_val = 0.0;
  for (double nu : {1e8, 1e12})
    for (double y : {-6.0, -1.5, 0.4, 1.0, 2.9, 8.0}) {
      const double z = (y - 0.4) / 1.3;
      auto upd = [&](const std::array<double, 2>& w) {
        w_ms = std::fmax(w_ms, w[0]);
        w_nu = std::fmax(w_nu, w[1]);
      };
      upd(gaussian_twin<1>(nu, 0.4, 1.3, y));
      upd(gaussian_twin<2>(nu, 0.4, 1.3, y));
      upd(gaussian_twin<3>(nu, 0.4, 1.3, y));
      upd(gaussian_twin<4>(nu, 0.4, 1.3, y));
      upd(gaussian_twin<5>(nu, 0.4, 1.3, y));
      upd(gaussian_twin<6>(nu, 0.4, 1.3, y));
      w_val = std::fmax(w_val, std::fabs(t1{0.4, 1.3, nu}.logpdf(y) - ld::gaussian1_distrib{0.4, 1.3}.logpdf(y)) * nu /
                                   std::pow(1.0 + z * z, 2.0));
    }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("student_t1 vs gaussian1: (mu, sigma) components %.3g, nu components %.3g, value %.3g (units of the limit's scale)\n",
                w_ms, w_nu, w_val);
  CHECK(w_ms <= 1.0);
  CHECK(w_nu <= 1.0);
  CHECK(w_val <= 1.0);
  // the information: I_mumu -> 1/sigma^2, I_sigmasigma -> 2/sigma^2, the rest -> 0
  for (double nu : {1e8, 1e12}) {
    const t1 d{0.4, 1.3, nu};
    const auto e = d.expected_info();
    CHECK(e[0] == doctest::Approx(1.0 / (1.3 * 1.3)).epsilon(4.0 / nu));
    CHECK(e[3] == doctest::Approx(2.0 / (1.3 * 1.3)).epsilon(8.0 / nu));
    CHECK(std::fabs(e[4]) <= 2.0 / (1.3 * nu * nu) * 1.01);
    CHECK(e[5] == doctest::Approx(3.5 / (nu * nu * nu * nu)).epsilon(20.0 / nu));
  }
}

TEST_CASE("student_t2 against student_t1 at the mapped scale") {
  for (double nu : {2.05, 3.0, 7.5, 200.0, 1e7})
    for (double y : {-5.0, -0.4, 0.4, 1.3, 6.0}) {
      const t2 e{0.4, 1.3, nu};
      const t1 d = e.as_t1();
      const double s = d.sigma, r = std::sqrt((nu - 2.0) / nu);
      INFO("nu = " << nu << ", y = " << y);
      CHECK(e.logpdf(y) == d.logpdf(y));
      CHECK(s == doctest::Approx(1.3 * r).epsilon(1e-15));
      // the chain rule at orders one and two by hand: r' = 1/(nu^2 r), r'' = -2/(nu^3 r) - 1/(nu^4 r^3)
      const double rp = 1.0 / (nu * nu * r), rpp = -2.0 / (nu * nu * nu * r) - 1.0 / (nu * nu * nu * nu * r * r * r);
      const auto a1 = e.logpdf_d1(y), b1 = d.logpdf_d1(y);
      const auto a2 = e.logpdf_d2(y), b2 = d.logpdf_d2(y);
      auto close = [&](double x, double yv, double scale) { return std::fabs(x - yv) <= 1e-11 * scale; };
      const double sc1 = std::fabs(b1[0]) + std::fabs(b1[1]) * r + std::fabs(b1[2]) + std::fabs(b1[1]) * 1.3 * rp;
      CHECK(close(a1[0], b1[0], sc1));
      CHECK(close(a1[1], r * b1[1], sc1));
      CHECK(close(a1[2], b1[2] + 1.3 * rp * b1[1], sc1));
      double sc2 = 0.0;
      for (int i = 0; i < 6; ++i) sc2 += std::fabs(b2[i]);
      sc2 *= (1.0 + 1.3 * rp + 1.3 * 1.3 * rp * rp + 1.3 * std::fabs(rpp));
      CHECK(close(a2[0], b2[0], sc2));                                                        // mu mu
      CHECK(close(a2[1], r * b2[1], sc2));                                                    // mu sigma
      CHECK(close(a2[2], b2[2] + 1.3 * rp * b2[1], sc2));                                     // mu nu
      CHECK(close(a2[3], r * r * b2[3], sc2));                                                // sigma sigma
      CHECK(close(a2[4], rp * b1[1] + r * (b2[4] + 1.3 * rp * b2[3]), sc2));                  // sigma nu
      CHECK(close(a2[5], b2[5] + 2.0 * 1.3 * rp * b2[4] + 1.3 * 1.3 * rp * rp * b2[3] + 1.3 * rpp * b1[1], sc2));  // nu nu
      // the pure mu derivatives of every order are student_t1's
      CHECK(e.logpdf_d6(y)[pos3(6, 0, 0)] == d.logpdf_d6(y)[pos3(6, 0, 0)]);
      CHECK(e.log_cdf_d4(y)[pos3(4, 0, 0)] == d.log_cdf_d4(y)[pos3(4, 0, 0)]);
      CHECK(e.cdf(y) == d.cdf(y));
      CHECK(e.log_ccdf(y) == d.log_ccdf(y));
    }
  // the variance of student_t2 is sigma^2 for every nu: the second moment by quadrature
  for (double nu : {2.6, 5.0, 30.0}) {
    const t2 e{0.4, 1.3, nu};
    const double var = laplace::math::quad::panels_quad([&](double y) { return (y - 0.4) * (y - 0.4) * std::exp(e.logpdf(y)); },
                                                       -ld::distrib_infinity, ld::distrib_infinity, e.center(), e.scale(),
                                                       nullptr, 0).value;
    if (nu >= 5.0) CHECK(var == doctest::Approx(1.3 * 1.3).epsilon(1e-6));
  }
}

TEST_CASE("student_t: start values") {
  std::vector<double> y;
  for (int i = 0; i < 101; ++i) y.push_back(2.0 + 0.1 * (i - 50));
  const auto s = ld::summarize_sample(y);
  const t1 a = ld::distrib_start_value(t1{0.0, 1.0, 1.0}, s);
  CHECK(a.mu == doctest::Approx(2.0).epsilon(1e-12));
  CHECK(a.sigma == doctest::Approx(0.5 * (s.upper_quartile - s.lower_quartile)).epsilon(1e-12));
  CHECK(a.nu == 5.0);
  const t2 b = ld::distrib_start_value(t2{0.0, 1.0, 3.0}, s);
  CHECK(b.mu == doctest::Approx(s.mean).epsilon(1e-12));
  CHECK(b.sigma == doctest::Approx(std::sqrt(s.variance)).epsilon(1e-12));
  CHECK(b.nu == 5.0);
  // a constant sample: sigma is moved strictly inside the domain
  const auto z = ld::summarize_sample(std::vector<double>(10, 1.0));
  CHECK(ld::distrib_start_value(t1{0.0, 1.0, 1.0}, z).sigma > 0.0);
  CHECK(ld::distrib_start_value(t2{0.0, 1.0, 3.0}, z).sigma > 0.0);
}

TEST_CASE("student_t: the log-density derivatives against mpmath") {
  // tools/reference/distrib_student_t.py: SymPy's derivatives of the definition, evaluated by mpmath at
  // 400 and 600 bits. Each component is compared in ulp of the largest component of its order at that point
  // (the components change sign), the error being that of the library value against hi + lo.
  const auto rows = read_reference("distrib_student_t_logpdf.csv", 0);
  CHECK(rows.size() > 10000);
  std::map<std::tuple<int, double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.fam, r.y, r.mu, r.sigma, r.nu, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  std::map<std::tuple<int, double, double, double, double, int>, std::vector<double>> cache;
  double worst[2][7] = {};
  std::map<double, double> by_nu[2];
  std::tuple<double, double, double, double, int, int> where[2][7];
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.fam, r.y, r.mu, r.sigma, r.nu, r.K);
    auto it = cache.find(key);
    if (it == cache.end()) {
      if (cache.size() > 64) cache.clear();
      it = cache.emplace(key, lp_tensor(r, r.K)).first;
    }
    const double v = it->second[pos3(r.a, r.b, r.c)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp_scaled = err / ulp_of(scale[key]);
    by_nu[r.fam - 1][r.nu] = std::fmax(by_nu[r.fam - 1][r.nu], ulp_scaled);
    if (ulp_scaled > worst[r.fam - 1][r.K]) {
      worst[r.fam - 1][r.K] = ulp_scaled;
      where[r.fam - 1][r.K] = std::make_tuple(r.y, r.mu, r.sigma, r.nu, r.b, r.c);
    }
    INFO("fam " << r.fam << " y " << r.y << " K " << r.K << " (a,b,c) " << r.a << r.b << r.c << " mu " << r.mu << " sigma "
                << r.sigma << " nu " << r.nu << " ulp " << ulp_scaled);
    CHECK(ulp_scaled <= BOUND_LOGPDF[r.fam - 1][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int f = 0; f < 2; ++f)
      for (int K = 1; K <= 6; ++K)
        std::printf("student_t%d logpdf, order %d: max %.3g ulp of the order's scale (y %.6g mu %.4g sigma %.4g nu %.8g, b %d c %d)\n",
                    f + 1, K, worst[f][K], std::get<0>(where[f][K]), std::get<1>(where[f][K]), std::get<2>(where[f][K]),
                    std::get<3>(where[f][K]), std::get<4>(where[f][K]), std::get<5>(where[f][K]));
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int f = 0; f < 2; ++f) {
      std::printf("student_t%d logpdf, worst scaled ulp by nu (all orders):", f + 1);
      for (const auto& kv : by_nu[f]) std::printf(" %.8g:%.3g", kv.first, kv.second);
      std::printf("%s", "\n");
    }
}

TEST_CASE("student_t: the expected information and its derivatives against mpmath") {
  // the closed forms evaluated and differentiated by mpmath at 400 and 600 bits (they are checked against the
  // definition, E[l_a l_b], in tools/reference/distrib_student_t.py forms); error in ulp of the component itself
  const auto rows = read_reference("distrib_student_t_info.csv", 2);
  CHECK(rows.size() > 2000);
  double worst[2][5] = {};
  std::tuple<double, double, int, int, int> where[2][5];
  for (const auto& r : rows) {
    const auto t = info_tensor(r, r.K);
    const double v = r.K == 0 ? t[r.pair] : t[r.pair * ld::sym_tensor_size(3, r.K) + pos3(0, r.b, r.c)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp = err / ulp_of(r.hi);
    if (ulp > worst[r.fam - 1][r.K]) {
      worst[r.fam - 1][r.K] = ulp;
      where[r.fam - 1][r.K] = std::make_tuple(r.sigma, r.nu, r.pair, r.b, r.c);
    }
    INFO("fam " << r.fam << " K " << r.K << " pair " << r.pair << " b " << r.b << " c " << r.c << " sigma " << r.sigma << " nu "
                << r.nu << " ulp " << ulp);
    CHECK(ulp <= BOUND_INFO[r.fam - 1][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int f = 0; f < 2; ++f)
      for (int K = 0; K <= 4; ++K)
        std::printf("student_t%d information, order %d: max %.3g ulp (sigma %.4g nu %.8g pair %d b %d c %d)\n", f + 1, K,
                    worst[f][K], std::get<0>(where[f][K]), std::get<1>(where[f][K]), std::get<2>(where[f][K]),
                    std::get<3>(where[f][K]), std::get<4>(where[f][K]));
}

TEST_CASE("student_t: the derivatives of log F and log(1 - F) against mpmath") {
  // tools/reference/distrib_student_t.py: the (z, nu) table of mpmath.diff of the definition at 280 and 360 bits,
  // composed exactly with z(mu, sigma, nu). Scaled ulp as for the log-density.
  const auto rows = read_reference("distrib_student_t_logcdf.csv", 1);
  CHECK(rows.size() > 10000);
  std::map<std::tuple<int, int, double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.fam, r.upper, r.y, r.mu, r.sigma, r.nu, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  // the same error measured against the largest component of the (mu, sigma) block alone (no derivative in nu):
  // at an integer nu the tail function is a power of sigma / (y - mu), whose sigma-derivatives vanish at the leading
  // order, and the components of that block are small against the ones in nu
  std::map<std::tuple<int, int, double, double, double, double, int>, double> scale_block;
  for (const auto& r : rows)
    if (r.c == 0) {
      auto& s = scale_block[std::make_tuple(r.fam, r.upper, r.y, r.mu, r.sigma, r.nu, r.K)];
      s = std::fmax(s, std::fabs(r.hi));
    }
  double worst_block[2][2][7] = {}, worst_block_integer = 0.0, worst_block_integer_z = 0.0;
  std::map<double, double> block_by_nu[2];
  std::map<std::tuple<int, int, double, double, double, double, int>, std::vector<double>> cache;
  double worst[2][2][7] = {}, worst_near2[7] = {};
  std::map<double, double> by_nu[2];
  std::map<double, double> by_z[2];  // worst scaled ulp by |z| = |y - mu|/sigma rounded to 3 digits (student_t1 only)
  std::tuple<double, double, double, double, int, int> where[2][2][7];
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.fam, r.upper, r.y, r.mu, r.sigma, r.nu, r.K);
    auto it = cache.find(key);
    if (it == cache.end()) {
      if (cache.size() > 64) cache.clear();
      it = cache.emplace(key, cdf_tensor(r, r.K)).first;
    }
    const double v = it->second[pos3(r.a, r.b, r.c)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp_scaled = err / ulp_of(scale[key]);
    by_nu[r.fam - 1][r.nu] = std::fmax(by_nu[r.fam - 1][r.nu], ulp_scaled);
    {
      const double zz = (r.y - r.mu) / r.sigma;
      const double zr = std::copysign(std::pow(10.0, std::round(std::log10(std::fmax(std::fabs(zz), 1e-3)) * 2.0) / 2.0), zz);
      by_z[r.upper][zr] = std::fmax(by_z[r.upper][zr], ulp_scaled);
    }
    // student_t2 within 0.001 of nu = 2 (the grid has nu = 2 + 1e-6): the chain through sigma sqrt((nu - 2)/nu)
    // keeps about eight digits at order six there, which has its own bound
    const bool near2 = r.fam == 2 && r.nu < 2.001;
    if (r.c == 0 && !near2) {
      const double ulp_block = err / ulp_of(scale_block[key]);
      // student_t1 at nu = 1, 2: the tail is a power of sigma/(y - mu) with an integer exponent and its sigma-derivatives
      // above that order vanish, so the tiny components of the block are the difference of terms of the size of the
      // tail probability: no bound on them (they are below the rounding of the other components), only the value
      const bool integer_nu = r.fam == 1 && r.nu == std::floor(r.nu) && r.nu <= 6.0;
      if (integer_nu) {
        worst_block_integer = std::fmax(worst_block_integer, ulp_block);
        worst_block_integer_z = std::fmax(worst_block_integer_z, std::fabs(r.y - r.mu) / r.sigma <= 31.0 ? ulp_block : 0.0);
      } else {
        worst_block[r.fam - 1][r.upper][r.K] = std::fmax(worst_block[r.fam - 1][r.upper][r.K], ulp_block);
        block_by_nu[r.fam - 1][r.nu] = std::fmax(block_by_nu[r.fam - 1][r.nu], ulp_block);
        CHECK(ulp_block <= BOUND_BLOCK[r.fam - 1][r.K]);
      }
    }
    if (near2) worst_near2[r.K] = std::fmax(worst_near2[r.K], ulp_scaled);
    if (!near2 && ulp_scaled > worst[r.fam - 1][r.upper][r.K]) {
      worst[r.fam - 1][r.upper][r.K] = ulp_scaled;
      where[r.fam - 1][r.upper][r.K] = std::make_tuple(r.y, r.mu, r.sigma, r.nu, r.b, r.c);
    }
    INFO("fam " << r.fam << " upper " << r.upper << " y " << r.y << " K " << r.K << " (a,b,c) " << r.a << r.b << r.c << " mu "
                << r.mu << " sigma " << r.sigma << " nu " << r.nu << " ulp " << ulp_scaled);
    CHECK(ulp_scaled <= (near2 ? BOUND_LOGCDF_NEAR2[r.K] : BOUND_LOGCDF[r.fam - 1][r.upper][r.K]));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int f = 0; f < 2; ++f)
      for (int u = 0; u < 2; ++u)
        for (int K = 1; K <= 6; ++K)
          std::printf("student_t%d log_%s derivs, order %d: max %.3g ulp of the order's scale (y %.6g mu %.4g sigma %.4g nu %.8g, b %d c %d)\n",
                      f + 1, u ? "(1-F)" : "F", K, worst[f][u][K], std::get<0>(where[f][u][K]), std::get<1>(where[f][u][K]),
                      std::get<2>(where[f][u][K]), std::get<3>(where[f][u][K]), std::get<4>(where[f][u][K]),
                      std::get<5>(where[f][u][K]));
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int f = 0; f < 2; ++f) {
      std::printf("student_t%d logcdf, worst scaled ulp by nu (all orders, both sides):", f + 1);
      for (const auto& kv : by_nu[f]) std::printf(" %.8g:%.3g", kv.first, kv.second);
      std::printf("%s", "\n");
    }
    for (int f = 0; f < 2; ++f) {
      std::printf("student_t%d (mu, sigma) block alone, worst scaled ulp by order (both sides):", f + 1);
      for (int K = 1; K <= 6; ++K) std::printf(" %.3g", std::fmax(worst_block[f][0][K], worst_block[f][1][K]));
      std::printf("%s", "\n");
      std::printf("student_t%d (mu, sigma) block alone, worst scaled ulp by nu:", f + 1);
      for (const auto& kv : block_by_nu[f]) std::printf(" %.8g:%.3g", kv.first, kv.second);
      std::printf("%s", "\n");
    }
    std::printf("student_t1 at nu = 1, 2: (mu, sigma) block alone, worst scaled ulp %.3g (%.3g for |z| <= 31)%s", worst_block_integer,
                worst_block_integer_z, "\n");
    std::printf("student_t2 within 0.001 of nu = 2, worst scaled ulp by order:");
    for (int K = 1; K <= 6; ++K) std::printf(" %.3g", worst_near2[K]);
    std::printf("%s", "\n");
    for (int u = 0; u < 2; ++u) {
      std::printf("log_%s, worst scaled ulp by z (both families, all orders):", u ? "(1-F)" : "F");
      for (const auto& kv : by_z[u]) std::printf(" %.3g:%.3g", kv.first, kv.second);
      std::printf("%s", "\n");
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp (doctest gives each case an internal-linkage function
// whose number restarts per translation unit, and nvcc tags every extended lambda with that address).
void student_kernel_case() {
  const int n = 48;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "student_t", n, KOKKOS_LAMBDA(const int i) {
        const double y = -3.0 + 0.2 * i;
        const ld::student_t1_distrib a{0.4, 1.3, 3.5};
        const ld::student_t2_distrib b{0.4, 1.3, 3.5};
        out(i) = a.logpdf_d6(y)[pos3(2, 1, 3)] + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] +
                 a.expected_info_d4().pair[3][pos3(0, 2, 2)] + a.expected_info_d4().pair[5][pos3(0, 0, 4)] +
                 a.log_cdf_d6(y)[pos3(1, 2, 3)] + a.log_ccdf_d6(y + 1.0)[pos3(0, 3, 3)] + a.logpdf(y) + a.cdf(y) +
                 b.logpdf_d6(y)[pos3(2, 1, 3)] + b.expected_info_d4().pair[4][pos3(0, 1, 3)] +
                 b.log_cdf_d6(y)[pos3(1, 2, 3)] + b.log_ccdf_d4(y)[pos3(0, 2, 2)] + b.logpdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -3.0 + 0.2 * i;
    const ld::student_t1_distrib a{0.4, 1.3, 3.5};
    const ld::student_t2_distrib b{0.4, 1.3, 3.5};
    const double r = a.logpdf_d6(y)[pos3(2, 1, 3)] + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] +
                     a.expected_info_d4().pair[3][pos3(0, 2, 2)] + a.expected_info_d4().pair[5][pos3(0, 0, 4)] +
                     a.log_cdf_d6(y)[pos3(1, 2, 3)] + a.log_ccdf_d6(y + 1.0)[pos3(0, 3, 3)] + a.logpdf(y) + a.cdf(y) +
                     b.logpdf_d6(y)[pos3(2, 1, 3)] + b.expected_info_d4().pair[4][pos3(0, 1, 3)] +
                     b.log_cdf_d6(y)[pos3(1, 2, 3)] + b.log_ccdf_d4(y)[pos3(0, 2, 2)] + b.logpdf(y);
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("student_t kernel: worst relative host-device difference %.3g\n", worst);
  CHECK(worst <= STUDENT_KERNEL_TOL);
}

TEST_CASE("student_t: every order inside a kernel") { student_kernel_case(); }
