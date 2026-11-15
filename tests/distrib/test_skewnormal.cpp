// The tests of skewnormal_distrib (include/laplace/distrib/skewnormal.hpp), the skew normal with mean mu,
// standard deviation sigma and skewness gamma: descriptors and the constant g, every derivative order
// against a 5-point difference of the order below, the mixed (y, theta) derivatives, the Gaussian limit
// at gamma = 0 against gaussian1_distrib, the three moments by quadrature, the reflection gamma -> -gamma,
// the cdf family, the expected information by the generic quadrature route (both forms, the variance of the
// score, the Gaussian limit and the derivatives against differences), the derivatives of the log density,
// log F and log(1 - F) against mpmath (tools/reference/distrib_skewnormal.py, gamma from 1e-12 to
// g - 1e-10) in classes of |gamma|, and a kernel launch of the highest orders.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/skewnormal.hpp>
#include <laplace/math/quadrature.hpp>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "check_common.hpp"

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace laplace_distrib_check {
// check_common.hpp builds a distribution of at most two parameters; this one has three
template <>
inline ld::skewnormal_distrib make<ld::skewnormal_distrib>(const double* t) {
  return {t[0], t[1], t[2]};
}
}  // namespace laplace_distrib_check

namespace {

using SN = ld::skewnormal_distrib;

// Bounds of the reference comparison in ulp of the (order, c) scale (the largest reference component of the same
// order and the same number c of gamma indices at that point), by class of |gamma| (rows, the grid values 1e-12,
// 1e-8, 1e-4, 1e-2, 0.5, 0.9, g - 1e-6, g - 1e-10) and order (columns, 0 unused): 1.5 times the worst of the gcc
// and the CUDA build over the three kinds (log density, log F, log(1 - F)), measured 2026-11-15, rounded up.
// A 0 is a corner that does not hold: at |gamma| = 1e-12 the orders 3 to 6 have no correct digit (the worst
// error is 1e17 to 1e19 ulp of the scale), because the derivatives in gamma carry cc^(n) ~ gamma^(1/3 - n)
// times a table whose accuracy is 1e-16 absolute while its true value is O(cc^2): see the header comment and
// the numerics entry. Those rows are not compared.
constexpr double SKEWNORMAL_BOUND[8][7] = {
    {0, 2.9e+09, 1.6e+14, 0, 0, 0, 0},                           // 1e-12
    {0, 2.2e+06, 1.1e+10, 2.6e+12, 1.3e+13, 2.6e+13, 3.5e+13},  // 1e-8
    {0, 2.5e+05, 1.4e+06, 4.9e+06, 8e+06, 9e+06, 1.3e+07},      // 1e-4
    {0, 200, 1.3e+04, 3.5e+04, 7.4e+04, 8.4e+04, 2.2e+05},      // 1e-2
    {0, 410, 1.1e+03, 770, 720, 4.2e+03, 2.6e+04},              // 0.5
    {0, 1.9e+03, 2.4e+03, 2.1e+03, 2.6e+03, 2.8e+03, 9.7e+03},  // 0.9
    {0, 82, 500, 500, 8.4e+04, 1.4e+03, 2.1e+03},               // g - 1e-6
    {0, 150, 110, 130, 2.6e+04, 1.1e+03, 320},                  // g - 1e-10
};
// worst relative difference between the host and the device sum of the kernel case on the CUDA build (RTX 4070
// SUPER, measured 2026-11-15): 6.35e-13, and half again as the bound; the source of the difference is the
// root finding of the quantile and FMA contraction in the cancelling compositions
#define SKEWNORMAL_KERNEL_TOL 1e-12

constexpr double pi_ = 3.14159265358979323846;

template <int J, int K>
std::vector<double> ydk(const SN& d, double y) {
  const auto t = d.logpdf_dy_dk<J, K>(y);
  return std::vector<double>(t.v, t.v + ld::sym_tensor<3, K>::size);
}

// d^(J+K) l / dy^J dtheta^K against the 5-point difference in y of the (J - 1, K) tensor
template <int J, int K>
void y_pair(const SN& d, double y, double hy, double rel) {
  const auto exact = ydk<J, K>(d, y);
  double scale = 0.0;
  for (double e : exact) scale = std::fmax(scale, std::fabs(e));
  std::vector<std::vector<double>> v;
  for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(ydk<J - 1, K>(d, y + s * hy));
  for (int i = 0; i < (int)exact.size(); ++i) {
    const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
    INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y << ", gamma = " << d.gamma);
    CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12);
  }
}
template <int J, int K>
void y_all(const SN& d, double y, double hy, double rel) {
  y_pair<J, K>(d, y, hy, rel);
  if constexpr (J + K + 1 <= 6) y_all<J, K + 1>(d, y, hy, rel);
  else if constexpr (J + 1 <= 6) y_all<J + 1, 0>(d, y, hy, rel);
}

// E[f(Y)] by quadrature
template <class F>
double expect(const SN& d, F f) {
  const auto g = [&](double y) { return f(y) * std::exp(d.logpdf(y)); };
  return laplace::math::quad::panels_quad(g, d.support_lower(), d.support_upper(), d.center(), d.scale(), nullptr, 0)
      .value;
}

// the K-th derivative of the information, K = 0..2, by the generic route in the score form
template <int K>
std::vector<double> info_score(const SN& d) {
  const auto r = ld::expected_info_by_integral<K, ld::info_form::score>(d);
  std::vector<double> v;
  if constexpr (K == 0) {
    for (int i = 0; i < 6; ++i) v.push_back(r[i]);
  } else {
    for (int p = 0; p < 6; ++p)
      for (int i = 0; i < ld::sym_tensor<3, K>::size; ++i) v.push_back(r.pair[p][i]);
  }
  return v;
}
template <int K>
std::vector<double> info_hessian(const SN& d) {
  const auto r = ld::expected_info_by_integral<K, ld::info_form::hessian>(d);
  std::vector<double> v;
  if constexpr (K == 0) {
    for (int i = 0; i < 6; ++i) v.push_back(r[i]);
  } else {
    for (int p = 0; p < 6; ++p)
      for (int i = 0; i < ld::sym_tensor<3, K>::size; ++i) v.push_back(r.pair[p][i]);
  }
  return v;
}

// the class of |gamma| of the reference grid
int gamma_class(double g) {
  const double a = std::fabs(g);
  const double G = ld::detail_skewnormal::g_max + ld::detail_skewnormal::g_max_lo;
  if (a < 1e-10) return 0;
  if (a < 1e-6) return 1;
  if (a < 1e-3) return 2;
  if (a < 0.1) return 3;
  if (a < 0.7) return 4;
  if (a < 0.95) return 5;
  return G - a > 1e-8 ? 6 : 7;
}
const char* const class_name[8] = {"1e-12", "1e-8", "1e-4", "1e-2", "0.5", "0.9", "g-1e-6", "g-1e-10"};

// one row of tests/data/reference/distrib_skewnormal.csv
struct reference_row {
  int kind, K, b, c;
  double y, hi, lo, mu, sigma, gamma;
};

std::vector<reference_row> read_reference() {
  std::vector<reference_row> rows;
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_skewnormal.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    reference_row r;
    char ys[64], his[64], los[64], mus[64], sigs[64], gs[64];
    if (std::sscanf(line, "%d,%63[^,],%d,%d,%d,%63[^,],%63[^,],%63[^,],%63[^,],%63s", &r.kind, ys, &r.K, &r.b, &r.c, his,
                    los, mus, sigs, gs) != 10)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.sigma = std::strtod(sigs, nullptr), r.gamma = std::strtod(gs, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// the position of (a derivatives in mu, b in sigma, c in gamma) in sym_tensor<3, K>, K = a + b + c
int component_position(int a, int b, int c) {
  std::vector<int> idx;
  for (int i = 0; i < a; ++i) idx.push_back(0);
  for (int i = 0; i < b; ++i) idx.push_back(1);
  for (int i = 0; i < c; ++i) idx.push_back(2);
  return position_of(3, idx);
}

// the K-th tensor of the log density (kind 0), log F (1) or log(1 - F) (2)
std::vector<double> kind_tensor(const SN& d, double y, int K, int kind) {
  return kind == 0 ? lderiv(d, y, K) : cderiv(d, y, K, kind == 2);
}

}  // namespace

TEST_CASE("distrib skewnormal: descriptors, the constant g and the kinks") {
  CHECK(SN::kind == ld::distrib_kind::continuous);
  CHECK(SN::n_params == 3);
  CHECK(SN::y_regularity == ld::infinitely_smooth);
  for (int j = 0; j < 3; ++j) {
    CHECK(SN::params[j].regularity == ld::infinitely_smooth);
    CHECK(SN::params[j].cdf_regularity == ld::infinitely_smooth);
    CHECK(SN::params[j].moves_support == false);
  }
  CHECK(SN::params[2].upper == ld::detail_skewnormal::g_max);
  CHECK(SN::params[2].lower == -ld::detail_skewnormal::g_max);
  const SN d{0.4, 1.3, 0.3};
  for (int j = 0; j < 3; ++j) CHECK(d.param_kinks(j, 1.25).count == 0);
  CHECK(d.y_kinks().count == 0);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 1.3);
  // g = ((4 - pi)/2) (b / sqrt(1 - b^2))^3, b = sqrt(2/pi) = 0.99527174643115604244... (mpmath at 300 bits)
  const double b = std::sqrt(2.0 / pi_);
  CHECK(ld::detail_skewnormal::g_max == doctest::Approx((4.0 - pi_) / 2.0 * std::pow(b / std::sqrt(1.0 - b * b), 3)).epsilon(1e-14));
  CHECK(ld::detail_skewnormal::g_max == doctest::Approx(0.99527174643115604244).epsilon(1e-14));
  // the double-double: g_max + g_max_lo is the true g to 1e-19 (the double 0.99527174643115604244 is g to 3e-17)
  CHECK(std::fabs((ld::detail_skewnormal::g_max - 0.99527174643115604244) + ld::detail_skewnormal::g_max_lo -
                  2.9952924930273875e-17) < 1e-19);
  // the value at the centre of the Gaussian case: -log(sigma sqrt(2 pi))
  const SN n{0.4, 1.3, 0.0};
  CHECK(n.logpdf(0.4) == doctest::Approx(-std::log(1.3 * std::sqrt(2.0 * pi_))).epsilon(1e-15));
}

TEST_CASE("distrib skewnormal: derivatives against differences") {
  // the 5-point difference has an error of order h^4 f^(5): the bounds are those of the difference itself
  // (measured 2026-11-15 on gcc: largest relative difference 8.2e-10 here, 4.1e-10 at gamma = 0.01)
  for (double g : {0.3, -0.6})
    for (double y : {-2.0, 0.5, 2.5}) check_derivs<SN>({0.4, 1.3, g}, y, 1.5e-9);
  for (double y : {-2.0, 0.5}) check_derivs<SN>({-1.0, 0.4, 0.01}, y, 1e-9);
  // gamma = 0.9: u = alpha t is of size 5 t, the differences converge slowly (1.3e-6 measured)
  for (double y : {-2.0, 0.5}) check_derivs<SN>({0.4, 1.3, 0.9}, y, 2e-6);
}

TEST_CASE("distrib skewnormal: derivatives in y and mixed (y, theta) against differences") {
  const SN d{0.4, 1.3, 0.3};
  CHECK(d.logpdf_dy_dk<0, 0>(1.3)[0] == d.logpdf(1.3));
  CHECK(d.logpdf_dy_dk<0, 3>(1.3)[2] == d.logpdf_d3(1.3)[2]);
  // d/dy = -d/dmu: the (J = 1, K = 0) value is minus the mu-derivative
  CHECK(d.logpdf_dy_dk<1, 0>(1.3)[0] == -d.logpdf_d1(1.3)[0]);
  for (double y : {-2.0, 0.5, 2.5}) y_all<1, 0>(d, y, 1e-3, 1e-8);
  y_all<1, 0>(SN{-1.0, 0.7, -0.6}, 0.2, 1e-3, 1e-8);
}

TEST_CASE("distrib skewnormal: gamma = 0 is the Gaussian") {
  // every component with no gamma index is that of gaussian1_distrib, which is the exact limit of the
  // density; the components with one gamma index are He_3(z)/6 and its derivatives, those with two or more
  // are infinite and are NaN
  for (double s : {0.35, 1.3, 40.0})
    for (double zz : {-3.0, -0.7, 0.0, 1.2, 4.0}) {
      const double mu = 0.4, y = mu + s * zz;
      const SN d{mu, s, 0.0};
      const ld::gaussian1_distrib g{mu, s};
      CHECK(d.logpdf(y) == doctest::Approx(g.logpdf(y)).epsilon(1e-15));
      const auto a1 = d.logpdf_d1(y);
      const auto b1 = g.logpdf_d1(y);
      CHECK(std::fabs(a1[0] - b1[0]) <= 1e-14 * (std::fabs(b1[0]) + 1.0 / s));
      CHECK(std::fabs(a1[1] - b1[1]) <= 1e-14 * (std::fabs(b1[1]) + 1.0 / s));
      // He_3(z)/6 / (the whole 1/sigma scaling in z): l_gamma = (z^3 - 3z)/6
      CHECK(a1[2] == doctest::Approx((zz * zz * zz - 3.0 * zz) / 6.0).epsilon(1e-13));
      const auto a2 = d.logpdf_d2(y);
      const auto b2 = g.logpdf_d2(y);
      for (int i : {0, 1, 3}) CHECK(std::fabs(a2[i] - b2[i == 3 ? 2 : i]) <= 1e-13 * (1.0 + std::fabs(b2[i == 3 ? 2 : i])));
      CHECK(std::isnan(a2[5]));
      const auto a3 = d.logpdf_d3(y);
      CHECK(std::isnan(a3[9]));
    }
  // the limit of the component (gamma) from gamma -> 0: error of order gamma^(1/3), the next term of the density
  {
    const SN d0{0.4, 1.3, 0.0}, d1{0.4, 1.3, 1e-9}, d2{0.4, 1.3, -1e-9};
    for (double y : {-2.0, 0.4, 2.5}) {
      CHECK(std::fabs(d1.logpdf_d1(y)[2] - d0.logpdf_d1(y)[2]) <= 2e-2 * (1.0 + std::fabs(d0.logpdf_d1(y)[2])));
      CHECK(std::fabs(d2.logpdf_d1(y)[2] - d0.logpdf_d1(y)[2]) <= 2e-2 * (1.0 + std::fabs(d0.logpdf_d1(y)[2])));
      // d/dgamma log F at 0 against the central difference (the even kurtosis term gamma^(4/3) cancels)
      for (bool upper : {false, true}) {
        const double h = 1e-9;
        const SN p{0.4, 1.3, h}, m{0.4, 1.3, -h};
        const double fd = ((upper ? p.log_ccdf(y) : p.log_cdf(y)) - (upper ? m.log_ccdf(y) : m.log_cdf(y))) / (2.0 * h);
        const double ex = (upper ? d0.log_ccdf_d1(y) : d0.log_cdf_d1(y))[2];
        INFO("y = " << y << ", upper " << upper);
        CHECK(std::fabs(fd - ex) <= 1e-3 * (1.0 + std::fabs(ex)));
        CHECK(std::isnan((upper ? d0.log_ccdf_d2(y) : d0.log_cdf_d2(y))[5]));
      }
    }
  }
  // log F and log(1 - F) at gamma = 0 are those of the normal in (mu, sigma)
  const SN d{0.4, 1.3, 0.0};
  const ld::gaussian1_distrib g{0.4, 1.3};
  for (double y : {-3.0, 0.4, 2.0}) {
    CHECK(d.log_cdf(y) == doctest::Approx(g.log_cdf(y)).epsilon(1e-14));
    CHECK(d.log_ccdf(y) == doctest::Approx(g.log_ccdf(y)).epsilon(1e-14));
    const auto a = d.log_cdf_d3(y);
    const auto b = g.log_cdf_d3(y);
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(a[component_position(3 - i, i, 0)] - b[i]) <= 1e-12 * (1.0 + std::fabs(b[i])));
  }
}

TEST_CASE("distrib skewnormal: the three parameters are the mean, the standard deviation and the skewness") {
  for (double g : {-0.99, -0.8, -0.3, 0.05, 0.6, 0.95, 0.99})
    for (double s : {0.3, 2.0}) {
      const SN d{1.5, s, g};
      const double m1 = expect(d, [](double y) { return y; });
      const double v = expect(d, [&](double y) { return (y - m1) * (y - m1); });
      const double k3 = expect(d, [&](double y) { return (y - m1) * (y - m1) * (y - m1); });
      INFO("gamma = " << g << ", sigma = " << s);
      CHECK(std::fabs(m1 - 1.5) <= 2e-10 * s);
      CHECK(std::fabs(v - s * s) <= 2e-10 * s * s);
      CHECK(std::fabs(k3 / (v * std::sqrt(v)) - g) <= 2e-9);
    }
}

TEST_CASE("distrib skewnormal: reflection gamma -> -gamma") {
  for (double g : {1e-8, 0.01, 0.4, 0.9, 0.995})
    for (double zz : {-4.0, -1.0, 0.3, 2.0, 6.0}) {
      const double mu = 0.7, s = 1.9, y = mu + s * zz;
      const SN a{mu, s, g}, b{mu, s, -g};
      INFO("gamma = " << g << ", z = " << zz);
      // the density of -Y: l(mu, sigma, -g; 2 mu - y) = l(mu, sigma, g; y), and F -> 1 - F
      CHECK(std::fabs(b.logpdf(2.0 * mu - y) - a.logpdf(y)) <= 2e-13 * (1.0 + std::fabs(a.logpdf(y))));
      CHECK(std::fabs(b.log_cdf(2.0 * mu - y) - a.log_ccdf(y)) <= 2e-13 * (1.0 + std::fabs(a.log_ccdf(y))));
      CHECK(std::fabs(b.log_ccdf(2.0 * mu - y) - a.log_cdf(y)) <= 2e-13 * (1.0 + std::fabs(a.log_cdf(y))));
      // l(2 mu0 - mu, sigma, -g; 2 mu0 - y) = l(mu, sigma, g; y) for every mu: the derivatives in mu and in gamma
      // change sign, those in sigma do not (positions (mu mu), (mu sigma), (mu g), (sigma sigma), (sigma g), (g g))
      if (g >= 0.01) {
        const auto a2 = a.logpdf_d2(y), b2 = b.logpdf_d2(2.0 * mu - y);
        const double sg[6] = {1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
        for (int i = 0; i < 6; ++i) CHECK(std::fabs(b2[i] - sg[i] * a2[i]) <= 1e-9 * (1.0 + std::fabs(a2[i])));
      }
    }
}

TEST_CASE("distrib skewnormal: the cdf family") {
  for (double g : {-0.9, -0.3, 0.2, 0.7}) {
    const SN d{0.4, 1.3, g};
    for (double y : {-8.0, -3.0, -0.5, 0.4, 1.7, 5.0}) {
      CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
      if (d.cdf(y) < 0.5) CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
      if (d.ccdf(y) < 0.5) CHECK(d.log_ccdf(y) == doctest::Approx(std::log(d.ccdf(y))).epsilon(1e-13));
    }
    // the cdf is the integral of the density
    for (double y : {-1.0, 0.4, 2.0}) {
      const auto f = [&](double x) { return std::exp(d.logpdf(x)); };
      const double F = laplace::math::quad::panels_quad(f, -ld::distrib_infinity, y, d.center(), d.scale(), nullptr, 0).value;
      INFO("gamma = " << g << ", y = " << y);
      CHECK(std::fabs(F - d.cdf(y)) <= 1e-9 * d.cdf(y));
    }
    // the quantile (generic route)
    for (double p : {1e-8, 0.01, 0.3, 0.5, 0.99, 1.0 - 1e-8}) {
      const double q = ld::distrib_quantile(d, p);
      INFO("gamma = " << g << ", p = " << p);
      if (p <= 0.5) CHECK(d.cdf(q) == doctest::Approx(p).epsilon(1e-10));
      else CHECK(d.ccdf(q) == doctest::Approx(1.0 - p).epsilon(1e-9));
    }
  }
}

TEST_CASE("distrib skewnormal: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true})
    for (double g : {0.3, -0.6})
      for (double y : {-8.0, -3.0, 0.2, 1.7, 6.0}) check_cdf_derivs<SN>({0.4, 1.3, g}, y, upper, 1e-7, std::fabs(y - 0.4) / 1.3);
}

TEST_CASE("distrib skewnormal: expected information, generic route") {
  // the two forms against each other, at three points
  for (auto th : {std::vector<double>{0.4, 1.3, 0.3}, std::vector<double>{-1.0, 0.4, -0.6}, std::vector<double>{2.0, 3.0, 0.9}}) {
    const SN d{th[0], th[1], th[2]};
    const auto s = info_score<0>(d), h = info_hessian<0>(d);
    const auto q = score_variance_continuous(d);
    double scale = 0.0;
    for (double e : s) scale = std::fmax(scale, std::fabs(e));
    for (int i = 0; i < 6; ++i) {
      INFO("gamma = " << th[2] << ", component " << i);
      CHECK(std::fabs(s[i] - h[i]) <= 1e-9 * scale);
      CHECK(std::fabs(s[i] - q[i]) <= 1e-9 * scale);
    }
    // the matrix is positive definite
    CHECK(s[0] > 0.0);
    CHECK(s[0] * s[3] - s[1] * s[1] > 0.0);
  }
  // gamma = 0: I = diag(1/sigma^2, 2/sigma^2, 1/6) (E[z He_3(z)] = E[(z^2 - 1) He_3(z)] = 0, E[He_3^2] = 6)
  for (double s : {0.5, 2.0}) {
    const SN d{0.3, s, 0.0};
    const auto i0 = info_score<0>(d);
    const ld::gaussian1_distrib g{0.3, s};
    const auto ig = g.expected_info();
    CHECK(std::fabs(i0[0] - ig[0]) <= 1e-10 * ig[0]);
    CHECK(std::fabs(i0[1]) <= 1e-10 * ig[0]);
    CHECK(std::fabs(i0[2]) <= 1e-10 * ig[0] / s);
    CHECK(std::fabs(i0[3] - ig[2]) <= 1e-10 * ig[2]);
    CHECK(std::fabs(i0[4]) <= 1e-10 * ig[0] / s);
    CHECK(std::fabs(i0[5] - 1.0 / 6.0) <= 1e-10);
  }
  // derivatives of the information against differences of the order below (generic route, score form)
  {
    const SN d{0.4, 1.3, 0.3};
    const double th0[3] = {d.mu, d.sigma, d.gamma};
    const auto e1 = info_score<1>(d);
    double scale = 0.0;
    for (double e : e1) scale = std::fmax(scale, std::fabs(e));
    for (int j = 0; j < 3; ++j) {
      const auto fd = diff5([&](const double* t) { return info_score<0>(SN{t[0], t[1], t[2]}); },
                            std::vector<double>(th0, th0 + 3), j, 1e-3 * std::fmax(1e-3, std::fabs(th0[j])));
      for (int p = 0; p < 6; ++p) {
        INFO("K = 1, pair " << p << ", parameter " << j);
        CHECK(std::fabs(e1[p * 3 + j] - fd[p]) <= 1e-7 * scale);
      }
    }
    // order 2: the component (a, b) of d^2 I against the difference of the order-1 tensor in the last index
    const auto e2 = info_score<2>(d);
    scale = 0.0;
    for (double e : e2) scale = std::fmax(scale, std::fabs(e));
    for (int j = 0; j < 3; ++j) {
      const auto fd = diff5([&](const double* t) { return info_score<1>(SN{t[0], t[1], t[2]}); },
                            std::vector<double>(th0, th0 + 3), j, 1e-3 * std::fmax(1e-3, std::fabs(th0[j])));
      for (int p = 0; p < 6; ++p)
        for (int i = 0; i <= j; ++i) {  // the sorted pairs (i, j) with i <= j
          const int pos = position_of(3, {i, j}), lower = i;  // position in sym_tensor<3, 2>, and the index in the order-1 tensor
          INFO("K = 2, pair " << p << ", (" << i << ", " << j << ")");
          CHECK(std::fabs(e2[p * 6 + pos] - fd[p * 3 + lower]) <= 1e-6 * scale);
        }
    }
  }
}

TEST_CASE("distrib skewnormal: derivatives against mpmath") {
  // tools/reference/distrib_skewnormal.py: truncated Taylor arithmetic of the definition at 300 and 400 bits.
  // Each component is compared in ulp of the largest component of its order and of its number c of gamma
  // indices at that point (the components change sign, so a plain relative error is undefined at their
  // zeros); the error is that of the library value against hi + lo.
  const auto rows = read_reference();
  CHECK(rows.size() > 40000);
  using key = std::tuple<int, double, int, double, double, double, int>;  // kind, y, K, mu, sigma, gamma, c
  std::map<key, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[key{r.kind, r.y, r.K, r.mu, r.sigma, r.gamma, r.c}];
    s = std::fmax(s, std::fabs(r.hi));
  }
  // the tensors, evaluated once per (kind, y, K, mu, sigma, gamma)
  using tkey = std::tuple<int, double, int, double, double, double>;
  std::map<tkey, std::vector<double>> tensors;
  double worst[3][8][7] = {};
  std::string where[3][8][7];
  for (const auto& r : rows) {
    const tkey tk{r.kind, r.y, r.K, r.mu, r.sigma, r.gamma};
    auto it = tensors.find(tk);
    if (it == tensors.end()) it = tensors.emplace(tk, kind_tensor(SN{r.mu, r.sigma, r.gamma}, r.y, r.K, r.kind)).first;
    const double v = it->second[component_position(r.K - r.b - r.c, r.b, r.c)];
    const double sc = scale[key{r.kind, r.y, r.K, r.mu, r.sigma, r.gamma, r.c}];
    const double err = std::isfinite(v) ? std::fabs((v - r.hi) - r.lo) : 1e300;
    const double ulp_scaled = err / std::ldexp(1.0, std::ilogb(sc) - 52);
    const int cl = gamma_class(r.gamma);
    char buf[256];
    if (ulp_scaled > worst[r.kind][cl][r.K]) {
      worst[r.kind][cl][r.K] = ulp_scaled;
      std::snprintf(buf, sizeof buf, "kind %d y %.17g mu %g sigma %g gamma %.17g b %d c %d", r.kind, r.y, r.mu, r.sigma,
                    r.gamma, r.b, r.c);
      where[r.kind][cl][r.K] = buf;
    }
    INFO("kind " << r.kind << " y " << r.y << " K " << r.K << " b " << r.b << " c " << r.c << " mu " << r.mu << " sigma "
                 << r.sigma << " gamma " << r.gamma << " ulp " << ulp_scaled);
    if (SKEWNORMAL_BOUND[cl][r.K] > 0.0) CHECK(ulp_scaled <= SKEWNORMAL_BOUND[cl][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    for (int kind = 0; kind < 3; ++kind)
      for (int cl = 0; cl < 8; ++cl)
        for (int K = 1; K <= 6; ++K)
          std::printf("skewnormal kind %d gamma class %s, order %d: max %.3g ulp of the (order, c) scale at %s\n", kind,
                      class_name[cl], K, worst[kind][cl][K], where[kind][cl][K].c_str());
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp.
void skewnormal_kernel_case() {
  const int n = 32;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "skewnormal_distrib", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.2 * i;
        const ld::skewnormal_distrib c{0.4, 1.3, 0.3 + 0.01 * i};
        out(i) = c.logpdf_d6(y)[10] + c.logpdf_dy_dk<2, 3>(y)[1] + c.log_cdf_d6(y)[20] + c.log_ccdf_d6(y + 1.0)[4] +
                 c.logpdf(y) + c.log_cdf(y) + c.log_ccdf(y) + ld::distrib_quantile(c, 0.25 + 0.01 * i);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.2 * i;
    const ld::skewnormal_distrib c{0.4, 1.3, 0.3 + 0.01 * i};
    const double r = c.logpdf_d6(y)[10] + c.logpdf_dy_dk<2, 3>(y)[1] + c.log_cdf_d6(y)[20] + c.log_ccdf_d6(y + 1.0)[4] +
                     c.logpdf(y) + c.log_cdf(y) + c.log_ccdf(y) + ld::distrib_quantile(c, 0.25 + 0.01 * i);
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("skewnormal kernel: worst relative host-device difference %.3g\n", worst);
  CHECK(worst <= SKEWNORMAL_KERNEL_TOL);
}

TEST_CASE("distrib skewnormal: every order inside a kernel") { skewnormal_kernel_case(); }

TEST_CASE("distrib skewnormal: starting value") {
  // the three parameters are the three moments, so the summary's own mean,
  // variance and skewness are the starting value, each moved strictly inside
  // its domain
  ld::distrib_sample_summary s;
  s.count = 1000.0, s.mean = 2.0, s.variance = 4.0, s.median = 1.8, s.skewness = 0.6;
  const SN d = SN{0.0, 1.0, 0.0}.start_value(s);
  CHECK(d.mu == 2.0);
  CHECK(d.sigma == 2.0);
  CHECK(d.gamma == doctest::Approx(0.6).epsilon(1e-14));
  s.skewness = 3.0;  // a sample skewness beyond g is moved strictly inside (-g, g)
  const SN e = SN{0.0, 1.0, 0.0}.start_value(s);
  CHECK(e.gamma < ld::detail_skewnormal::g_max);
  CHECK(e.gamma > 0.99);
  s.skewness = -3.0;
  const SN f = SN{0.0, 1.0, 0.0}.start_value(s);
  CHECK(f.gamma > -ld::detail_skewnormal::g_max);
  CHECK(f.gamma < -0.99);
}
