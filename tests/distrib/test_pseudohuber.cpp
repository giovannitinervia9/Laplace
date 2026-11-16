// The test of pseudohuber_distrib (include/laplace/distrib/pseudohuber.hpp): packing and descriptors, the
// density against the Bessel definition, every derivative order against differences, the mixed (y, theta)
// derivatives, the Laplace limit (nu -> 0) against laplace1_distrib and the Gaussian limit (nu -> infinity)
// against gaussian1_distrib, the normalization and the variance sigma^2 by quadrature, the expected
// information by the generic route in both forms against the variance of the score and against the limits, the
// cdf of the generic quadrature route (through the dispatchers) against differences, the generic quantile,
// the references of tools/reference/distrib_pseudohuber.py (the log-density in (mu, sigma, nu)), and a kernel launch.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/math/quadrature.hpp>
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

namespace laplace_distrib_check {
// the shared checks build a distribution from a parameter vector, two parameters at most by default
template <>
ld::pseudohuber_distrib make<ld::pseudohuber_distrib>(const double* t) {
  return ld::pseudohuber_distrib{t[0], t[1], t[2]};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included
using ld::detail_pseudohuber::pos3;

namespace {

using ph = ld::pseudohuber_distrib;


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

// ---------------------------------------------------------------- the generic cdf through the dispatchers

template <int K>
std::vector<double> gcderiv_k(const ph& d, double y, bool upper) {
  const auto t = upper ? ld::distrib_log_ccdf_dk<K>(d, y) : ld::distrib_log_cdf_dk<K>(d, y);
  return as_vec<3, K>(t);
}
std::vector<double> gcderiv(const ph& d, double y, int K, bool upper) {
  switch (K) {
    case 0: return {upper ? ld::distrib_log_ccdf(d, y) : ld::distrib_log_cdf(d, y)};
    case 1: return gcderiv_k<1>(d, y, upper);
    case 2: return gcderiv_k<2>(d, y, upper);
    case 3: return gcderiv_k<3>(d, y, upper);
    case 4: return gcderiv_k<4>(d, y, upper);
    case 5: return gcderiv_k<5>(d, y, upper);
    default: return gcderiv_k<6>(d, y, upper);
  }
}

// order K of the generic log F (upper: log(1 - F)) against the 5-point difference of order K - 1, every component;
// returns the largest difference relative to the largest component of the order
double check_generic_cdf(const std::vector<double>& th, double y, bool upper, int K) {
  double worst = 0.0;
  const auto exact = gcderiv(make<ph>(th.data()), y, K, upper);
  double scale = 0.0;
  for (double e : exact) scale = std::fmax(scale, std::fabs(e));
  for (const auto& idx : sorted_indices(3, K)) {
    const std::vector<int> lower(idx.begin(), idx.end() - 1);
    const int j = idx.back();
    const auto fd = diff5([&](const double* t) { return gcderiv(make<ph>(t), y, K - 1, upper); }, th, j,
                          1e-3 * std::fmax(1e-3, std::fabs(th[j])));
    const double err = std::fabs(exact[position_of(3, idx)] - fd[position_of(3, lower)]) / scale;
    worst = std::fmax(worst, err);
  }
  return worst;
}

// ---------------------------------------------------------------- the references

struct ref_row {
  int K = 0, a = 0, b = 0, c = 0;
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

// y,K,a,b,c,hi,lo,mu,sigma,nu
std::vector<ref_row> read_reference(const char* name) {
  std::vector<ref_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/" + name;
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if (s.size() != 10) continue;
    ref_row r;
    auto num = [&](int i) { return std::strtod(s[i].c_str(), nullptr); };
    auto in = [&](int i) { return std::atoi(s[i].c_str()); };
    r.y = num(0), r.K = in(1), r.a = in(2), r.b = in(3), r.c = in(4), r.hi = num(5), r.lo = num(6);
    r.mu = num(7), r.sigma = num(8), r.nu = num(9);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

// The bound is the worst scaled error measured on 2026-11-16 on the gcc build and on the CUDA build
// (RTX 4070 SUPER), which agree here to the digit, times about one and a half, in ulp of the largest component
// of the order at that point. The row of K = 0 is the value itself, in ulp of the value.
// Three regions of nu, where the errors have different causes: nu < 1 (the series in nu for R and h, which does
// not cancel: tens of ulp at most; before it, the Bessel route lost about eps/nu and reached 1.7e7 ulp at
// nu = 1e-6), 1 <= nu < 1e4 and nu >= 1e4 (above one the Bessel derivatives of the double-double route set the
// scale; the cancellation of Lambda = -x against h = +x, which used to lose a factor of nu at every order and
// reached 1.6e4 ulp on the value at nu = 1e8, is gone since the two are carried as Lambda + x and h - x).
const double BOUND_LOGPDF[3][7] = {{2.5, 45.0, 7.5, 8.5, 13.0, 21.0, 14.0},
                                   {6.8, 50.0, 30.0, 81.0, 130.0, 120.0, 420.0},
                                   {11.0, 10.0, 130.0, 110.0, 200.0, 56.0, 37.0}};
inline int nu_region(double nu) { return nu < 1.0 ? 0 : (nu >= 1e4 ? 2 : 1); }

}  // namespace

TEST_CASE("pseudohuber: descriptors and kinks") {
  CHECK(ph::kind == ld::distrib_kind::continuous);
  CHECK(ph::n_params == 3);
  CHECK(ph::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 3; ++j) {
    CHECK(ph::params[j].regularity == ld::infinitely_smooth);
    CHECK(ph::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(ph::params[j].moves_support == false);
  }
  CHECK(ph::params[1].lower == 0.0);
  CHECK(ph::params[2].lower == 0.0);
  const ph d{0.4, 1.3, 3.5};
  for (int j = 0; j < 3; ++j) CHECK(d.param_kinks(j, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 1.3);
  // the density is symmetric about mu
  CHECK(d.logpdf(0.4 + 2.0) == doctest::Approx(d.logpdf(0.4 - 2.0)).epsilon(1e-15));
}

TEST_CASE("pseudohuber: the density against the Bessel definition") {
  // l = -sqrt(nu + R z^2) + (1/2) log R - (1/2) log nu - log K_1(x) - log 2 - log sigma, R = x K_2(x)/K_1(x),
  // with the K_n of the C++ standard library (cyl_bessel_k)
  for (double nu : {0.01, 0.3, 1.0, 4.5, 60.0, 300.0}) {
    const double x = std::sqrt(nu);
    const double k1 = std::cyl_bessel_k(1.0, x), k2 = std::cyl_bessel_k(2.0, x);
    const double R = x * k2 / k1;
    const ph d{0.4, 1.3, nu};
    CHECK(d.bessel_ratio() == doctest::Approx(R).epsilon(1e-12));
    for (double y : {-9.0, 0.4, 2.0, 30.0}) {
      const double z = (y - 0.4) / 1.3;
      const double ref = -std::sqrt(nu + R * z * z) + 0.5 * std::log(R) - 0.5 * std::log(nu) - std::log(k1) - std::log(2.0) - std::log(1.3);
      CHECK(d.logpdf(y) == doctest::Approx(ref).epsilon(1e-11));
    }
  }
  // the limits of R: 2 as nu -> 0, sqrt(nu) + 3/2 as nu -> infinity
  CHECK(ph{0.0, 1.0, 1e-12}.bessel_ratio() == doctest::Approx(2.0).epsilon(1e-10));
  CHECK(ph{0.0, 1.0, 1e8}.bessel_ratio() == doctest::Approx(1e4 + 1.5).epsilon(1e-6));
}

TEST_CASE("pseudohuber: derivatives against differences, centre and tails") {
  for (double nu : {0.05, 1.0, 7.0, 150.0})
    for (double y : {-40.0, -3.0, 0.2, 0.4, 1.7, 9.0, 300.0}) check_derivs<ph>({0.4, 1.3, nu}, y, 1e-6);
}

TEST_CASE("pseudohuber: mixed (y, theta) derivatives against differences") {
  for (double y : {-3.0, 0.2, 1.7, 9.0}) check_y_derivs3<ph>({0.4, 1.3, 4.5}, y, 1e-3, 1e-6);
  // the J = 1 route equals the minus of the mu-derivative of the same order
  const ph d{0.4, 1.3, 4.5};
  const auto a = d.logpdf_dy_dk<1, 0>(2.0);
  const auto b = d.logpdf_dk<1>(2.0);
  CHECK(a[0] == doctest::Approx(-b[0]).epsilon(1e-14));
  const auto a2 = d.logpdf_dy_dk<2, 2>(2.0);
  const auto b4 = d.logpdf_dk<4>(2.0);
  CHECK(a2[pos3(0, 1, 1)] == b4[pos3(2, 1, 1)]);
}

namespace {

// ---------------------------------------------------------------- the limits

// The Laplace limit: at small nu the density is the Laplace of scale b with variance 2 b^2 = sigma^2, so
// b = sigma / sqrt(2), and since b is linear in sigma the component (a, b_, 0) of order K is the Laplace's
// (a, b_) times 2^(-b_/2). Returns the largest difference relative to the largest component of the order in
// the (mu, sigma) block.
template <int K>
double laplace_twin(double nu, double mu, double sg, double y) {
  const ph d{mu, sg, nu};
  const ld::laplace1_distrib l{mu, sg / std::sqrt(2.0)};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = l.logpdf_dk<K>(y);
  double scale = 0.0, worst = 0.0;
  for (int i = 0; i <= K; ++i) scale = std::fmax(scale, std::fabs(b[i] * std::pow(2.0, -0.5 * i)));
  for (int i = 0; i <= K; ++i) worst = std::fmax(worst, std::fabs(a[pos3(K - i, i, 0)] - b[i] * std::pow(2.0, -0.5 * i)) / scale);
  return worst;
}

// The Gaussian limit: with R = sqrt(nu) + 3/2 + O(nu^-1/2), l - l_normal is a polynomial in z of degree 4 over
// sqrt(nu) to first order, so a derivative of order K in (mu, sigma) differs by at most C (1 + z^2)^((4 + K)/2) K! /
// (sigma^K sqrt(nu)), and a derivative with c >= 1 indices in nu is at most C (1 + z^2)^((4 + K)/2) K! /
// (sigma^(K - c) nu^(c + 1/2)). The functions return the largest of the two ratios to those scales.
template <int K>
std::array<double, 2> gaussian_twin(double nu, double mu, double sg, double y) {
  const ph d{mu, sg, nu};
  const ld::gaussian1_distrib g{mu, sg};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = g.logpdf_dk<K>(y);
  double kfact = 1.0;
  for (int i = 2; i <= K; ++i) kfact *= i;
  const double z = (y - mu) / sg, poly = kfact * std::pow(1.0 + z * z, 0.5 * (4.0 + K));
  double w_ms = 0.0, w_nu = 0.0;
  for (int i = 0; i <= K; ++i) w_ms = std::fmax(w_ms, std::fabs(a[pos3(K - i, i, 0)] - b[i]) * std::sqrt(nu) * std::pow(sg, K) / poly);
  for (int c = 1; c <= K; ++c)
    for (int b_ = 0; b_ + c <= K; ++b_)
      w_nu = std::fmax(w_nu, std::fabs(a[pos3(K - b_ - c, b_, c)]) * std::pow(nu, c + 0.5) * std::pow(sg, K - c) / poly);
  return {w_ms, w_nu};
}

}  // namespace

TEST_CASE("pseudohuber at nu = 1e-8 and 1e-6 against laplace1_distrib, every order") {
  // l - l_Laplace is O(nu log(1/nu)) in the value, and its z-derivatives of order K are of size nu K! / |z|^(K + 1)
  // (the term nu / (2 sqrt(R) |z|) of -sqrt(nu + R z^2)); the points have |z| >= 1.5. The largest measured
  // difference, relative to the largest component of the order, is printed with LAPLACE_PRINT_ERRORS.
  double worst[2] = {0.0, 0.0};
  int i = 0;
  for (double nu : {1e-8, 1e-6}) {
    for (double z : {-12.0, -4.0, -1.5, 1.5, 4.0, 12.0}) {
      const double y = 0.4 + 1.3 * z;
      double w = 0.0;
      w = std::fmax(w, laplace_twin<1>(nu, 0.4, 1.3, y));
      w = std::fmax(w, laplace_twin<2>(nu, 0.4, 1.3, y));
      w = std::fmax(w, laplace_twin<3>(nu, 0.4, 1.3, y));
      w = std::fmax(w, laplace_twin<4>(nu, 0.4, 1.3, y));
      w = std::fmax(w, laplace_twin<5>(nu, 0.4, 1.3, y));
      w = std::fmax(w, laplace_twin<6>(nu, 0.4, 1.3, y));
      worst[i] = std::fmax(worst[i], w);
    }
    ++i;
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber vs laplace1: worst relative difference %.3g at nu = 1e-8, %.3g at nu = 1e-6\n", worst[0], worst[1]);
  CHECK(worst[0] <= 5e-8);
  CHECK(worst[1] <= 3.7e-6);
}

TEST_CASE("pseudohuber at nu = 1e8 and 1e10 against gaussian1_distrib") {
  double w_ms = 0.0, w_nu = 0.0, w_val = 0.0;
  for (double nu : {1e8, 1e10})
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
      w_val = std::fmax(w_val, std::fabs(ph{0.4, 1.3, nu}.logpdf(y) - ld::gaussian1_distrib{0.4, 1.3}.logpdf(y)) * std::sqrt(nu) /
                                   std::pow(1.0 + z * z, 2.0));
    }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber vs gaussian1: (mu, sigma) components %.3g, nu components %.3g, value %.3g (units of the limit's scale)\n",
                w_ms, w_nu, w_val);
  CHECK(w_ms <= 1.1);
  CHECK(w_nu <= 0.28);
  CHECK(w_val <= 0.56);
}

TEST_CASE("pseudohuber: the density integrates to one and its variance is sigma^2") {
  double worst_mass = 0.0, worst_var = 0.0;
  for (const std::array<double, 3>& p : {std::array<double, 3>{0.4, 1.3, 1e-4}, std::array<double, 3>{-2.0, 0.25, 0.2},
                                         std::array<double, 3>{1.0, 4.0, 1.0}, std::array<double, 3>{0.0, 1.0, 30.0},
                                         std::array<double, 3>{3.0, 0.5, 5e5}}) {
    const ph d{p[0], p[1], p[2]};
    const double mass = laplace::math::quad::panels_quad([&](double y) { return std::exp(d.logpdf(y)); }, -ld::distrib_infinity,
                                                         ld::distrib_infinity, d.center(), d.scale(), nullptr, 0).value;
    const double var = laplace::math::quad::panels_quad([&](double y) { return (y - p[0]) * (y - p[0]) * std::exp(d.logpdf(y)); },
                                                        -ld::distrib_infinity, ld::distrib_infinity, d.center(), d.scale(), nullptr, 0).value;
    INFO("nu = " << p[2]);
    worst_mass = std::fmax(worst_mass, std::fabs(mass - 1.0));
    worst_var = std::fmax(worst_var, std::fabs(var / (p[1] * p[1]) - 1.0));
    CHECK(std::fabs(mass - 1.0) <= 7e-14);
    CHECK(std::fabs(var / (p[1] * p[1]) - 1.0) <= 7e-14);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber: |mass - 1| worst %.3g, |variance / sigma^2 - 1| worst %.3g\n", worst_mass, worst_var);
}

TEST_CASE("pseudohuber: expected information in both forms against the variance of the score") {
  double worst_forms = 0.0, worst_var = 0.0, worst_zero = 0.0;
  for (const std::array<double, 3>& p : {std::array<double, 3>{0.4, 1.3, 0.05}, std::array<double, 3>{-2.0, 0.25, 1.0},
                                         std::array<double, 3>{1.0, 4.0, 12.0}, std::array<double, 3>{0.0, 1.0, 400.0}}) {
    const ph d{p[0], p[1], p[2]};
    const auto s = ld::expected_info_by_integral<0, ld::info_form::score>(d);
    const auto h = ld::expected_info_by_integral<0, ld::info_form::hessian>(d);
    const auto q = score_variance_continuous(d);
    INFO("nu = " << p[2]);
    const double mx = std::fmax(s[0], std::fmax(s[3], s[5]));
    // sym_tensor<3, 2>: 0 mumu, 1 musigma, 2 munu, 3 sigmasigma, 4 sigmanu, 5 nunu
    for (int i : {0, 3, 4, 5}) {
      const double sc = i == 4 ? std::sqrt(s[3] * s[5]) : s[i];
      worst_forms = std::fmax(worst_forms, std::fabs(s[i] - h[i]) / sc);
      worst_var = std::fmax(worst_var, std::fabs(s[i] - q[i]) / sc);
      CHECK(std::fabs(s[i] - h[i]) <= 1.1e-12 * sc);
      CHECK(std::fabs(s[i] - q[i]) <= 5.3e-16 * sc);
    }
    for (int i : {1, 2}) {
      worst_zero = std::fmax(worst_zero, std::fmax(std::fabs(s[i]), std::fabs(h[i])) / mx);
      CHECK(std::fabs(s[i]) <= 1e-15 * mx);
      CHECK(std::fabs(h[i]) <= 1e-15 * mx);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber information: score vs hessian form %.3g, vs variance of the score %.3g (relative), zero entries %.3g (of the largest)\n",
                worst_forms, worst_var, worst_zero);
}

namespace {

// the components of the expected information (K = 0) or of its derivative of order K by the generic route, in
// the given form, pair-major then the sorted K-index
template <int K, ld::info_form F>
std::vector<double> info_by_route(const ph& d) {
  const auto r = ld::expected_info_by_integral<K, F>(d);
  std::vector<double> v;
  if constexpr (K == 0) {
    for (int i = 0; i < ld::sym_tensor_size(3, 2); ++i) v.push_back(r[i]);
  } else {
    for (int p = 0; p < ld::sym_tensor_size(3, 2); ++p)
      for (int i = 0; i < ld::sym_tensor_size(3, K); ++i) v.push_back(r.pair[p][i]);
  }
  return v;
}
template <ld::info_form F>
std::vector<double> info_order(const ph& d, int K) {
  switch (K) {
    case 0: return info_by_route<0, F>(d);
    case 1: return info_by_route<1, F>(d);
    case 2: return info_by_route<2, F>(d);
    case 3: return info_by_route<3, F>(d);
    default: return info_by_route<4, F>(d);
  }
}

// the derivatives of order K of the information against the 5-point difference of order K - 1, every component;
// the largest difference relative to the largest component of the order
template <ld::info_form F>
double check_generic_info_derivs(const std::vector<double>& th, int K) {
  const int npair = ld::sym_tensor_size(3, 2);
  const auto exact = info_order<F>(make<ph>(th.data()), K);
  double scale = 0.0, worst = 0.0;
  for (double e : exact) scale = std::fmax(scale, std::fabs(e));
  for (int p = 0; p < npair; ++p)
    for (const auto& idx : sorted_indices(3, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return info_order<F>(make<ph>(t), K - 1); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(3, K - 1);
      const double e = exact[p * ld::sym_tensor_size(3, K) + position_of(3, idx)];
      const double a = fd[K == 1 ? p : p * n_lower + position_of(3, lower)];
      worst = std::fmax(worst, std::fabs(e - a) / scale);
    }
  return worst;
}

}  // namespace

TEST_CASE("pseudohuber: derivatives of the expected information (generic route, both forms) against differences") {
  double worst[2][5] = {};
  for (const std::vector<double>& th : {std::vector<double>{0.4, 1.3, 2.5}, std::vector<double>{-2.0, 0.25, 0.3}})
    for (int K = 1; K <= (th[2] > 1.0 ? 4 : 3); ++K) {
      worst[0][K] = std::fmax(worst[0][K], check_generic_info_derivs<ld::info_form::score>(th, K));
      worst[1][K] = std::fmax(worst[1][K], check_generic_info_derivs<ld::info_form::hessian>(th, K));
    }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int f = 0; f < 2; ++f) {
      std::printf("pseudohuber information derivatives against differences, %s form, worst relative by order:", f ? "hessian" : "score");
      for (int K = 1; K <= 4; ++K) std::printf(" %.3g", worst[f][K]);
      std::printf("%s", "\n");
    }
  const double bound[2][5] = {{0, 1.8e-11, 4.3e-11, 8.5e-11, 1.5e-10}, {0, 1.8e-11, 4.3e-11, 8.5e-11, 1.5e-10}};
  for (int f = 0; f < 2; ++f)
    for (int K = 1; K <= 4; ++K) CHECK(worst[f][K] <= bound[f][K]);
}

TEST_CASE("pseudohuber: expected information against the limits") {
  // Laplace of scale b = sigma/sqrt(2): I_mumu = 1/b^2 = 2/sigma^2, I_sigmasigma = 1/(2 b^2) = 1/sigma^2;
  // Gaussian: I_mumu = 1/sigma^2, I_sigmasigma = 2/sigma^2
  double w_lap = 0.0, w_gau = 0.0;
  {
    const ph d{0.4, 1.3, 1e-8};
    const ld::laplace1_distrib l{0.4, 1.3 / std::sqrt(2.0)};
    const auto e = ld::expected_info_by_integral<0, ld::info_form::score>(d);
    const auto el = l.expected_info();
    w_lap = std::fmax(std::fabs(e[0] - el[0]) / el[0], std::fabs(e[3] - el[2] * 0.5) / (el[2] * 0.5));
  }
  {
    const ph d{0.4, 1.3, 1e8};
    const ld::gaussian1_distrib g{0.4, 1.3};
    const auto e = ld::expected_info_by_integral<0, ld::info_form::score>(d);
    const auto eg = g.expected_info();
    w_gau = std::fmax(std::fabs(e[0] - eg[0]) / eg[0], std::fabs(e[3] - eg[2]) / eg[2]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber information vs laplace1 at nu = 1e-8: %.3g, vs gaussian1 at nu = 1e8: %.3g (relative)\n", w_lap, w_gau);
  CHECK(w_lap <= 2.4e-4);  // the corrections are O(sqrt(nu)) and O(1 / sqrt(nu))
  CHECK(w_gau <= 2.3e-4);
}

TEST_CASE("pseudohuber: the generic cdf through the dispatchers") {
  // the order K of log F and log(1 - F) against the 5-point difference of the order below, on a handful of points
  // (the generic route costs 34 ms at order 6)
  double worst[7] = {};
  for (bool upper : {false, true})
    for (double y : {-2.5, 0.9}) {
      for (int K = 1; K <= 4; ++K) worst[K] = std::fmax(worst[K], check_generic_cdf({0.4, 1.3, 2.5}, y, upper, K));
      worst[1] = std::fmax(worst[1], check_generic_cdf({0.4, 1.3, 0.2}, y, upper, 1));
      worst[2] = std::fmax(worst[2], check_generic_cdf({0.4, 1.3, 40.0}, y, upper, 2));
    }
  worst[5] = check_generic_cdf({0.4, 1.3, 2.5}, 3.0, false, 5);
  worst[6] = check_generic_cdf({0.4, 1.3, 2.5}, -2.0, true, 6);
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    std::printf("pseudohuber generic cdf against differences, worst relative by order:");
    for (int K = 1; K <= 6; ++K) std::printf(" %.3g", worst[K]);
    std::printf("%s", "\n");
  }
  const double bound_cdf[7] = {0, 6.4e-12, 2.1e-11, 5e-11, 9.5e-11, 8.6e-11, 1.5e-10};
  for (int K = 1; K <= 6; ++K) CHECK(worst[K] <= bound_cdf[K]);
  // the values: F + (1 - F) = 1, the symmetry of F about mu, and F(mu) = 1/2
  const ph d{0.4, 1.3, 2.5};
  for (double y : {-30.0, -3.0, 0.1, 1.7, 9.0}) {
    CHECK(ld::distrib_cdf(d, y) + ld::distrib_ccdf(d, y) == doctest::Approx(1.0).epsilon(1e-13));
    CHECK(ld::distrib_log_ccdf(d, y) == doctest::Approx(ld::distrib_log_cdf(d, 0.8 - y)).epsilon(1e-12));
  }
  CHECK(ld::distrib_cdf(d, 0.4) == doctest::Approx(0.5).epsilon(1e-13));
}

TEST_CASE("pseudohuber: the generic quantile") {
  // the root finder of the generic route, on a symmetric density (it does not know the symmetry)
  double worst = 0.0, worst_sym = 0.0;
  for (double nu : {0.01, 1.0, 50.0}) {
    const ph d{0.4, 1.3, nu};
    for (double p : {1e-8, 1e-3, 0.3, 0.5, 0.8, 1.0 - 1e-8, 0x1p-26, 0x1p-10, 0.3125, 0.75}) {
      const double q = ld::quantile_by_root(d, p);
      const double back = p <= 0.5 ? ld::distrib_cdf(d, q) : ld::distrib_ccdf(d, q);
      const double target = p <= 0.5 ? p : 1.0 - p;
      worst = std::fmax(worst, std::fabs(back - target) / target);
      if (std::ldexp(p, 26) == std::floor(std::ldexp(p, 26))) {  // 1 - p is exact only for a dyadic p
        const double q2 = ld::quantile_by_root(d, 1.0 - p);
        worst_sym = std::fmax(worst_sym, std::fabs(q + q2 - 0.8) / (1.0 + std::fabs(q)));
      }
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("pseudohuber quantile: cdf(quantile(p)) vs p worst relative %.3g, symmetry q(p) + q(1 - p) = 2 mu worst %.3g\n", worst,
                worst_sym);
  CHECK(worst <= 3.5e-15);
  CHECK(worst_sym <= 1.9e-15);  // 1.27e-15 on both builds: where the root of the quantile stops
}

TEST_CASE("pseudohuber: start values") {
  std::vector<double> y;
  for (int i = 0; i < 101; ++i) y.push_back(2.0 + 0.1 * (i - 50));
  const auto s = ld::summarize_sample(y);
  const ph a = ld::distrib_start_value(ph{0.0, 1.0, 3.0}, s);
  CHECK(a.mu == doctest::Approx(s.mean).epsilon(1e-12));
  CHECK(a.sigma == doctest::Approx(std::sqrt(s.variance)).epsilon(1e-12));
  CHECK(a.nu == 1.0);
  const auto z = ld::summarize_sample(std::vector<double>(10, 1.0));
  CHECK(ld::distrib_start_value(ph{0.0, 1.0, 3.0}, z).sigma > 0.0);
}

TEST_CASE("pseudohuber: the log-density derivatives against mpmath") {
  // tools/reference/distrib_pseudohuber.py: SymPy's derivatives of the definition, evaluated by mpmath at 400 and
  // 600 bits. Each component is compared in ulp of the largest component of its order at that point (the
  // components change sign), the error being that of the library value against hi + lo.
  const auto rows = read_reference("distrib_pseudohuber.csv");
  CHECK(rows.size() > 10000);
  std::map<std::tuple<double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.y, r.mu, r.sigma, r.nu, r.K)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  std::map<std::tuple<double, double, double, double, int>, std::vector<double>> cache;
  double worst[7] = {};
  std::map<double, double> by_nu;
  std::map<double, double> by_nu_order[7];
  std::tuple<double, double, double, double, int, int> where[7];
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.y, r.mu, r.sigma, r.nu, r.K);
    auto it = cache.find(key);
    if (it == cache.end()) {
      if (cache.size() > 64) cache.clear();
      it = cache.emplace(key, lderiv(ph{r.mu, r.sigma, r.nu}, r.y, r.K)).first;
    }
    const double v = it->second[r.K == 0 ? 0 : pos3(r.a, r.b, r.c)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp_scaled = err / ulp_of(scale[key]);
    by_nu[r.nu] = std::fmax(by_nu[r.nu], ulp_scaled);
    by_nu_order[r.K][r.nu] = std::fmax(by_nu_order[r.K][r.nu], ulp_scaled);
    if (ulp_scaled > worst[r.K]) {
      worst[r.K] = ulp_scaled;
      where[r.K] = std::make_tuple(r.y, r.mu, r.sigma, r.nu, r.b, r.c);
    }
    INFO("y " << r.y << " K " << r.K << " (a,b,c) " << r.a << r.b << r.c << " mu " << r.mu << " sigma " << r.sigma << " nu " << r.nu
              << " ulp " << ulp_scaled);
    CHECK(ulp_scaled <= BOUND_LOGPDF[nu_region(r.nu)][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int K = 0; K <= 6; ++K)
      std::printf("pseudohuber logpdf, order %d: max %.3g ulp of the order's scale (y %.6g mu %.4g sigma %.4g nu %.8g, b %d c %d)\n", K,
                  worst[K], std::get<0>(where[K]), std::get<1>(where[K]), std::get<2>(where[K]), std::get<3>(where[K]),
                  std::get<4>(where[K]), std::get<5>(where[K]));
    std::printf("pseudohuber logpdf, worst scaled ulp by nu (all orders):");
    for (const auto& kv : by_nu) std::printf(" %.8g:%.3g", kv.first, kv.second);
    std::printf("%s", "\n");
    for (int K = 0; K <= 6; ++K) {
      std::printf("  order %d by nu:", K);
      for (const auto& kv : by_nu_order[K]) std::printf(" %.8g:%.3g", kv.first, kv.second);
      std::printf("%s", "\n");
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp (doctest gives each case an internal-linkage function
// whose number restarts per translation unit, and nvcc tags every extended lambda with that address).
// One kernel per derivative order (the stack frame of a kernel is that of its largest call, so a kernel that
// holds every order has the frame of the sixth): the sum and the sum of the absolute values of the components of
// the tensor of order K, on the device and on the host.
template <int K>
void pseudohuber_kernel_order(double nu, double bound) {
  const int n = 40;
  Kokkos::View<double*> out("out", n), mag("mag", n);
  Kokkos::parallel_for(
      "pseudohuber_order", n, KOKKOS_LAMBDA(const int i) {
        const ld::pseudohuber_distrib a{0.4, 1.3, nu};
        const auto t = a.logpdf_dk<K>(-3.0 + 0.2 * i);
        double s = 0.0, m = 0.0;
        for (int j = 0; j < ld::sym_tensor<3, K>::size; ++j) {
          s += t[j];
          m += Kokkos::fabs(t[j]);
        }
        out(i) = s;
        mag(i) = m;
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const ld::pseudohuber_distrib a{0.4, 1.3, nu};
    const auto t = a.logpdf_dk<K>(-3.0 + 0.2 * i);
    double s = 0.0, m = 0.0;
    for (int j = 0; j < ld::sym_tensor<3, K>::size; ++j) {
      s += t[j];
      m += std::fabs(t[j]);
    }
    worst = std::fmax(worst, std::fabs(h(i) - s) / m);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("pseudohuber kernel order %d, nu %g: worst host-device difference %.3g of the sum of |components|%s", K, nu, worst, "\n");
  CHECK(worst <= bound);
}

void pseudohuber_kernel_other(double nu) {
  const int n = 12;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "pseudohuber_other", n, KOKKOS_LAMBDA(const int i) {
        const double y = -3.0 + 0.5 * i;
        const ld::pseudohuber_distrib a{0.4, 1.3, nu};
        out(i) = a.logpdf(y) + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] + ld::distrib_log_cdf(a, y) + ld::distrib_log_cdf_dk<1>(a, y)[2];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -3.0 + 0.5 * i;
    const ld::pseudohuber_distrib a{0.4, 1.3, nu};
    const double r = a.logpdf(y) + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] + ld::distrib_log_cdf(a, y) + ld::distrib_log_cdf_dk<1>(a, y)[2];
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("pseudohuber kernel (value, dy_dk<2,3>, generic cdf), nu %g: worst relative difference %.3g%s", nu, worst, "\n");
  CHECK(worst <= 1.15e-14);
}

void pseudohuber_kernel_case() {
  // worst host-device differences measured on 2026-11-15 (RTX 4070 SUPER) times about one and a half
  pseudohuber_kernel_order<1>(3.5, 9.3e-16);
  pseudohuber_kernel_order<2>(3.5, 7.9e-16);
  pseudohuber_kernel_order<3>(3.5, 9.5e-16);
  pseudohuber_kernel_order<4>(3.5, 1.3e-15);
  pseudohuber_kernel_order<5>(3.5, 8.6e-16);
  pseudohuber_kernel_order<6>(3.5, 7.7e-16);
  pseudohuber_kernel_order<1>(0.05, 9.2e-16);
  pseudohuber_kernel_order<2>(0.05, 1.1e-15);
  pseudohuber_kernel_order<3>(0.05, 2.3e-15);
  pseudohuber_kernel_order<4>(0.05, 1.3e-14);
  pseudohuber_kernel_order<5>(0.05, 2.4e-14);
  pseudohuber_kernel_order<6>(0.05, 2.8e-15);
  // The generic cdf route of a three-parameter family keeps the derivative tensors of the integrand on the stack:
  // with the default stack of a CUDA thread (1024 bytes) the kernel ends in an illegal memory access (measured:
  // 3072 bytes fail, 4096 work), so the limit is raised for this launch and restored.
#if defined(__CUDACC__)
  size_t stack_before = 0;
  cudaDeviceGetLimit(&stack_before, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, 4096);
#endif
  pseudohuber_kernel_other(3.5);
  pseudohuber_kernel_other(0.05);
#if defined(__CUDACC__)
  cudaDeviceSetLimit(cudaLimitStackSize, stack_before);
#endif
}

TEST_CASE("pseudohuber: every order inside a kernel") { pseudohuber_kernel_case(); }
