// The tests of enet_distrib (include/laplace/distrib/enet.hpp), the elastic-net distribution in
// (mu, lambda, alpha): descriptors and the kink in mu, every derivative order against a 5-point
// difference of the order below (in the parameters, in y and mixed), the kink itself (the one-sided
// limits of dl/dy differ by 2a), the density integrating to one, the expected information against
// the variance of the score by quadrature and by the generic score-form route (never the hessian
// form: the kink) and its derivatives against differences, the Laplace limit at alpha -> 1
// (laplace2_distrib) and the Gaussian limit at alpha -> 0 (gaussian3_distrib, gaussian1_distrib),
// the cdf family (identities, quantile round trips, the derivatives of log F and log(1 - F)
// against differences), every quantity against mpmath (tools/reference/distrib_enet.py), and a
// kernel launch of the highest orders. check_common.hpp is written for one or two parameters
// (make<D>, the y-derivative dispatcher), so the checks of three parameters are repeated here.
#include <doctest/doctest.h>

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian3.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/math/quadrature.hpp>
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

using E = ld::enet_distrib;

E mk(const double* t) { return E{t[0], t[1], t[2]}; }

bool printing() { return std::getenv("LAPLACE_PRINT_ERRORS") != nullptr; }

// step of a parameter in the difference checks: the parameters stay far from their bounds here
double step_of(int j, const std::vector<double>& th) {
  if (j == 2) return 1e-3 * std::fmin(th[2], 1.0 - th[2]);
  return 1e-3 * std::fmax(1e-3, std::fabs(th[j]));
}

void check_derivs3(const std::vector<double>& th, double y, double rel) {
  constexpr int P = 3;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = lderiv(mk(th.data()), y, K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return lderiv(mk(t), y, K - 1); }, th, j, step_of(j, th));
      const double e = exact[position_of(P, idx)], a = fd[position_of(P, lower)];
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y << ", theta = " << th[0] << " " << th[1] << " "
                  << th[2]);
      CHECK(std::fabs(e - a) <= rel * scale + 1e-12);
    }
  }
}

#define ENET_YD(J, K) \
  case J * 10 + K: return as_vec<3, K>(d.template logpdf_dy_dk<J, K>(y));
std::vector<double> yderiv3(const E& d, double y, int J, int K) {
  switch (J * 10 + K) {
    ENET_YD(0, 0) ENET_YD(0, 1) ENET_YD(0, 2) ENET_YD(0, 3) ENET_YD(0, 4) ENET_YD(0, 5) ENET_YD(0, 6)
    ENET_YD(1, 0) ENET_YD(1, 1) ENET_YD(1, 2) ENET_YD(1, 3) ENET_YD(1, 4) ENET_YD(1, 5)
    ENET_YD(2, 0) ENET_YD(2, 1) ENET_YD(2, 2) ENET_YD(2, 3) ENET_YD(2, 4)
    ENET_YD(3, 0) ENET_YD(3, 1) ENET_YD(3, 2) ENET_YD(3, 3)
    ENET_YD(4, 0) ENET_YD(4, 1) ENET_YD(4, 2)
    ENET_YD(5, 0) ENET_YD(5, 1)
    ENET_YD(6, 0)
    default: return {};
  }
}
#undef ENET_YD

// every (J, K): against the difference in y of (J - 1, K) and in theta of (J, K - 1)
void check_y_derivs3(const std::vector<double>& th, double y, double hy, double rel) {
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv3(mk(th.data()), y, J, K);
      double scale = 0.0;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yderiv3(mk(th.data()), y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        INFO("dy: J = " << J << ", K = " << K << ", component " << i << ", y = " << y);
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-9);
      }
      if (K == 0) continue;
      for (const auto& idx : sorted_indices(3, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const auto fd = diff5([&](const double* t) { return yderiv3(mk(t), y, J, K - 1); }, th, j, step_of(j, th));
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(3, idx) << ", y = " << y);
        CHECK(std::fabs(exact[position_of(3, idx)] - fd[position_of(3, lower)]) <= rel * scale + 1e-9);
      }
    }
}

void check_info_derivs3(const std::vector<double>& th, double rel) {
  constexpr int P = 3;
  const int npair = ld::sym_tensor_size(P, 2);
  for (int K = 1; K <= 4; ++K) {
    const auto exact = ideriv(mk(th.data()), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (int p = 0; p < npair; ++p)
      for (const auto& idx : sorted_indices(P, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ideriv(mk(t), K - 1); }, th, j, step_of(j, th));
        const int n_lower = K == 1 ? 1 : ld::sym_tensor_size(P, K - 1);
        const double e = exact[p * ld::sym_tensor_size(P, K) + position_of(P, idx)];
        const double a = fd[K == 1 ? p : p * n_lower + position_of(P, lower)];
        INFO("K = " << K << ", pair " << p << ", component " << position_of(P, idx));
        CHECK(std::fabs(e - a) <= rel * scale + 1e-14);
      }
  }
}

void check_cdf_derivs3(const std::vector<double>& th, double y, bool upper, double rel, double zscale) {
  constexpr int P = 3;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = cderiv(mk(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(P, K)) {
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const int j = idx.back();
      const auto fd =
          diff5([&](const double* t) { return cderiv(mk(t), y, K - 1, upper); }, th, j, step_of(j, th) / (1.0 + zscale));
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y << ", upper " << upper);
      CHECK(std::fabs(exact[position_of(P, idx)] - fd[position_of(P, lower)]) <= rel * scale + 1e-12);
    }
  }
}

// the reference rows of tests/data/reference/distrib_enet.csv
struct reference_row {
  int kind, K, p, pos;
  double y, hi, lo, mu, lam, alp;
};

std::vector<reference_row> read_reference() {
  std::vector<reference_row> rows;
  std::FILE* f = std::fopen(LAPLACE_REFERENCE_DIR "/distrib_enet.csv", "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    reference_row r;
    char ys[64], his[64], los[64], mus[64], lams[64], alps[64];
    if (std::sscanf(line, "%d,%d,%d,%d,%63[^,],%63[^,],%63[^,],%63[^,],%63[^,],%63s", &r.kind, &r.K, &r.p, &r.pos, ys, his,
                    los, mus, lams, alps) != 10)
      continue;
    r.y = std::strtod(ys, nullptr), r.hi = std::strtod(his, nullptr), r.lo = std::strtod(los, nullptr);
    r.mu = std::strtod(mus, nullptr), r.lam = std::strtod(lams, nullptr), r.alp = std::strtod(alps, nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// the library value of a reference row
double library_value(const reference_row& r) {
  const E d{r.mu, r.lam, r.alp};
  switch (r.kind) {
    case 0: return lderiv(d, r.y, r.K)[r.pos];
    case 1: return cderiv(d, r.y, r.K, false)[r.pos];
    case 2: return cderiv(d, r.y, r.K, true)[r.pos];
    case 3: {
      const auto v = ideriv(d, r.K);
      return v[r.p * (r.K == 0 ? 1 : ld::sym_tensor_size(3, r.K)) + r.pos];
    }
    default: return d.quantile(r.y);
  }
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

// |a - b| <= tol |b| (doctest::Approx adds an absolute scale of 1)
bool rel(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fabs(b); }

}  // namespace

TEST_CASE("enet: descriptors and kinks") {
  CHECK(E::kind == ld::distrib_kind::continuous);
  CHECK(E::n_params == 3);
  CHECK(E::y_regularity == 0);
  CHECK(E::params[0].regularity == 0);
  CHECK(E::params[0].cdf_regularity == 1);
  CHECK(E::params[1].regularity == ld::infinitely_smooth);
  CHECK(E::params[1].cdf_regularity == ld::infinitely_smooth);
  CHECK(E::params[2].regularity == ld::infinitely_smooth);
  CHECK(E::params[2].lower == 0.0);
  CHECK(E::params[2].upper == 1.0);
  for (int j = 0; j < 3; ++j) CHECK(E::params[j].moves_support == false);
  const E d{0.5, 2.0, 0.3};
  CHECK(d.param_kinks(0, 1.25).count == 1);
  CHECK(d.param_kinks(0, 1.25).point[0] == 1.25);
  CHECK(d.param_kinks(0, 1.25).continuous_order == 0);
  CHECK(d.param_kinks(1, 1.25).count == 0);
  CHECK(d.param_kinks(2, 1.25).count == 0);
  CHECK(d.y_kinks().count == 1);
  CHECK(d.y_kinks().point[0] == 0.5);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.5);
}

TEST_CASE("enet: the density integrates to one, the variance and the mean by quadrature") {
  const std::vector<std::vector<double>> pts = {{0.4, 1.7, 0.4}, {-1.0, 0.3, 0.9}, {2.0, 25.0, 0.05}, {0.0, 3.0, 0.999}, {0.0, 1e-3, 0.5}};
  for (const auto& p : pts) {
    const E d = mk(p.data());
    const auto yk = d.y_kinks();
    auto f = [&](double y) { return std::exp(d.logpdf(y)); };
    auto f1 = [&](double y) { return (y - d.mu) * std::exp(d.logpdf(y)); };
    auto f2 = [&](double y) { return (y - d.mu) * (y - d.mu) * std::exp(d.logpdf(y)); };
    const double s = d.scale();
    const double one = laplace::math::quad::panels_quad(f, d.support_lower(), d.support_upper(), d.center(), s, yk.point, yk.count).value;
    const double m1 = laplace::math::quad::panels_quad(f1, d.support_lower(), d.support_upper(), d.center(), s, yk.point, yk.count).value;
    const double m2 = laplace::math::quad::panels_quad(f2, d.support_lower(), d.support_upper(), d.center(), s, yk.point, yk.count).value;
    INFO("theta " << p[0] << " " << p[1] << " " << p[2]);
    CHECK(std::fabs(one - 1.0) <= 1e-11);
    CHECK(std::fabs(m1) <= 1e-11 * s);
    CHECK(std::fabs(m2 / (s * s) - 1.0) <= 1e-10);
  }
}

TEST_CASE("enet: derivatives against differences, away from the kink") {
  for (double y : {-6.0, -0.9, 0.2, 1.7, 5.0}) check_derivs3({0.4, 1.7, 0.4}, y, 1e-7);
  for (double y : {-3.0, 0.6, 1.0}) check_derivs3({0.4, 0.3, 0.9}, y, 1e-7);
  for (double y : {-8.0, 0.0, 3.0}) check_derivs3({-0.5, 5.0, 0.1}, y, 1e-7);
  for (double y : {-1.0, 2.5}) check_derivs3({0.5, 1.0, 0.6180339887498949}, y, 1e-7);  // x = 1
  for (double y : {-1.0, 2.5}) check_derivs3({0.5, 1.0, 0.8284271247461903}, y, 1e-7);  // x = 2
}

TEST_CASE("enet: the structure of the tensor away from the kink") {
  // l is bilinear in the rates and quadratic in z: components with three or more mu indices are 0
  const E d{0.4, 1.7, 0.4};
  const double y = 2.3;
  const auto t3 = d.logpdf_d3(y);
  const auto t2 = d.logpdf_d2(y);
  CHECK(t3[0] == 0.0);                  // mu mu mu
  CHECK(t3[1] == -(1.0 - 0.4));         // mu mu lambda: c_lambda v_2
  CHECK(t3[2] == 1.7);                  // mu mu alpha: c_alpha v_2 = lambda
  CHECK(t2[0] == -1.7 * (1.0 - 0.4));   // -c
  CHECK(t2[1] == doctest::Approx(0.4 + 0.6 * (y - 0.4)).epsilon(1e-15));  // alpha sign(z) + (1 - alpha) z
  CHECK(t2[2] == doctest::Approx(1.7 - 1.7 * (y - 0.4)).epsilon(1e-15));  // lambda sign(z) - lambda z
}

TEST_CASE("enet: mixed (y, theta) derivatives against differences") {
  for (double y : {-3.0, 0.1, 0.9, 5.0}) check_y_derivs3({0.4, 1.7, 0.4}, y, 1e-3, 1e-7);
  for (double y : {-1.0, 1.4}) check_y_derivs3({0.4, 0.3, 0.9}, y, 1e-3, 1e-7);
  const E d{0.4, 1.7, 0.4};
  CHECK(d.logpdf_dy_dk<0, 0>(1.3)[0] == d.logpdf(1.3));
  CHECK(d.logpdf_dy_dk<0, 3>(1.3)[2] == d.logpdf_d3(1.3)[2]);
  // J = 1: minus the mu-derivative of the same order
  const auto a = d.logpdf_dy_dk<1, 1>(2.0);
  const auto b = d.logpdf_dk<2>(2.0);
  CHECK(a[0] == -b[0]);  // (y, mu) = -(mu, mu)
  CHECK(a[1] == -b[1]);  // (y, lambda) = -(mu, lambda)
  CHECK(a[2] == -b[2]);
}

TEST_CASE("enet: the kink in mu and in y") {
  const E d{0.4, 1.7, 0.4};
  const double a = 1.7 * 0.4, c = 1.7 * 0.6;
  // at y = mu: sign(0) = 0 and z = 0, so the first derivative in mu is 0 and the value is -log Z
  CHECK(d.logpdf_d1(0.4)[0] == 0.0);
  CHECK(d.logpdf_dy_dk<1, 0>(0.4)[0] == 0.0);
  const double eps = 1e-9;
  // dl/dy = -a sign(z) - c z: the one-sided limits differ by 2 a
  const double left = d.logpdf_dy_dk<1, 0>(0.4 - eps)[0], right = d.logpdf_dy_dk<1, 0>(0.4 + eps)[0];
  CHECK(left - right == doctest::Approx(2.0 * a).epsilon(1e-8));
  CHECK(left == doctest::Approx(a + c * eps).epsilon(1e-8));
  // dl/dmu = -dl/dy: the jump of the first derivative in mu, from the left of the location to the right, is +2 a
  CHECK(d.logpdf_d1(0.4 + eps)[0] - d.logpdf_d1(0.4 - eps)[0] == doctest::Approx(2.0 * a).epsilon(1e-8));
  // both tails are 1/2 at the location
  CHECK(d.cdf(0.4) == 0.5);
  CHECK(d.ccdf(0.4) == 0.5);
  CHECK(d.log_cdf(0.4) == doctest::Approx(-std::log(2.0)).epsilon(1e-15));
  CHECK(d.quantile(0.5) == 0.4);
}

TEST_CASE("enet: log F is C^1 in mu across the location, its second derivative jumps by 4 a f(mu)") {
  // F depends on y - mu, so d^2 log F / dmu^2 = (f' F - f^2) / F^2 in y, where f'/f = -a sign(z) - c z jumps from +a to -a
  // across z = 0, f is continuous and F(mu) = 1/2: the jump of the second derivative, from the left to the right of
  // the location, is -2 a f(mu) / F(mu) = -4 a f(mu) (for the Laplace density f(mu) = a/2: -2 a^2, laplace1's 2/sigma^2)
  for (const std::vector<double> p : {std::vector<double>{0.4, 1.7, 0.4}, std::vector<double>{0.4, 0.3, 0.9}, std::vector<double>{0.4, 5.0, 0.1}}) {
    const E d = mk(p.data());
    const double a = p[1] * p[2], eps = 1e-8;
    const double f0 = std::exp(d.logpdf(d.mu));
    const double d1l = d.log_cdf_d1(d.mu - eps)[0], d1r = d.log_cdf_d1(d.mu + eps)[0];
    CHECK(std::fabs(d1l - d1r) <= 1e-6 * std::fabs(d1l));  // C^1
    CHECK(d1r == doctest::Approx(-2.0 * f0).epsilon(1e-6));  // (log F)_mu = -f/F
    const double j = d.log_cdf_d2(d.mu + eps)[0] - d.log_cdf_d2(d.mu - eps)[0];
    INFO("alpha " << p[2]);
    CHECK(j == doctest::Approx(-4.0 * a * f0).epsilon(1e-5));
    const double jc = d.log_ccdf_d2(d.mu + eps)[0] - d.log_ccdf_d2(d.mu - eps)[0];
    CHECK(jc == doctest::Approx(4.0 * a * f0).epsilon(1e-5));  // 1 - F is the mirror
  }
}

TEST_CASE("enet: expected information against the variance of the score (quadrature) and the generic score form") {
  const std::vector<std::vector<double>> pts = {{0.4, 1.7, 0.4}, {-1.0, 0.3, 0.9}, {2.0, 8.0, 0.05}, {0.0, 3.0, 0.97}, {0.3, 50.0, 0.5}};
  for (const auto& p : pts) {
    const E d = mk(p.data());
    const auto q = score_variance_continuous(d);
    const auto e = d.expected_info();
    INFO("theta " << p[0] << " " << p[1] << " " << p[2]);
    const double sc = std::fmax(std::fmax(e[0], e[3]), e[5]);
    CHECK(std::fabs(q[0] - e[0]) <= 1e-9 * e[0]);
    CHECK(std::fabs(q[1]) <= 1e-9 * sc);  // I_mu,lambda = 0 by symmetry
    CHECK(std::fabs(q[2]) <= 1e-9 * sc);  // I_mu,alpha = 0
    CHECK(e[1] == 0.0);
    CHECK(e[2] == 0.0);
    CHECK(std::fabs(q[3] - e[3]) <= 1e-9 * e[3]);
    CHECK(std::fabs(q[4] - e[4]) <= 1e-9 * std::fmax(std::fabs(e[4]), 1e-3 * std::sqrt(e[3] * e[5])));
    CHECK(std::fabs(q[5] - e[5]) <= 1e-9 * e[5]);
    // I_mumu = c + 2 a f(0), never the hessian form -E[l_mumu] = c
    const double c = p[1] * (1.0 - p[2]), a = p[1] * p[2];
    CHECK(e[0] == doctest::Approx(c + 2.0 * a * std::exp(d.logpdf(d.mu))).epsilon(1e-13));
  }
  // the generic route, score form, for I and its first derivatives (the hessian form is wrong at the kink)
  const E d{0.4, 1.7, 0.4};
  const auto g0 = ld::expected_info_by_integral<0, ld::info_form::score>(d);
  const auto e0 = d.expected_info();
  for (int i = 0; i < 6; ++i) CHECK(std::fabs(g0[i] - e0[i]) <= 1e-9 * std::fmax(std::fabs(e0[i]), 1e-2));
  const auto g1 = ld::expected_info_by_integral<1, ld::info_form::score>(d);
  const auto e1 = d.expected_info_d1();
  for (int p = 0; p < 6; ++p)
    for (int i = 0; i < 3; ++i)
      if (std::isfinite(g1.pair[p][i]))  // NaN where the kink makes the integrand non-differentiable (an index mu in C)
        CHECK(std::fabs(g1.pair[p][i] - e1.pair[p][i]) <= 1e-8 * std::fmax(std::fabs(e1.pair[p][i]), 1e-2));
  const auto g2 = ld::expected_info_by_integral<2, ld::info_form::score>(d);
  const auto e2 = d.expected_info_d2();
  int compared = 0;
  for (int p = 0; p < 6; ++p)
    for (int i = 0; i < 6; ++i)
      if (std::isfinite(g2.pair[p][i])) {
        ++compared;
        CHECK(std::fabs(g2.pair[p][i] - e2.pair[p][i]) <= 1e-7 * std::fmax(std::fabs(e2.pair[p][i]), 1e-2));
      }
  CHECK(compared > 0);
}

TEST_CASE("enet: derivatives of the expected information against differences") {
  check_info_derivs3({0.4, 1.7, 0.4}, 1e-7);
  check_info_derivs3({0.4, 0.3, 0.9}, 1e-7);
  check_info_derivs3({-7.0, 20.0, 0.1}, 1e-7);
  check_info_derivs3({0.0, 1.0, 0.6180339887498949}, 1e-7);
  // I does not depend on mu: every component with an index mu in the derivative is 0
  const E d{0.4, 1.7, 0.4};
  const auto i2 = d.expected_info_d2();
  for (int p = 0; p < 6; ++p)
    for (int pos = 0; pos < 3; ++pos) CHECK(i2.pair[p][pos] == 0.0);  // (mu mu), (mu lambda), (mu alpha) come first
}

TEST_CASE("enet: the Laplace limit alpha -> 1 against laplace2_distrib") {
  // enet at alpha = 1 - 1e-10 is laplace2 with rate lambda up to O(c) = 1e-10 lambda relative changes of
  // the order-one quantities (c z^2/2 in the log-density, and in the other quantities through a
  // normalization shifted by about c/a^2); the (mu, lambda) block of every order is compared with the tolerance
  // below (the derivatives in alpha have no counterpart)
  const double al = 1.0 - 1e-10;
  for (double lam : {0.7, 3.0}) {
    const E d{0.4, lam, al};
    const ld::laplace2_distrib l{0.4, lam};
    const double sc = 1.0 / lam;
    for (double y : {0.4 - 5 * sc, 0.4 - 0.5 * sc, 0.4 + 0.3 * sc, 0.4 + 4 * sc}) {
      INFO("lambda " << lam << " y " << y);
      CHECK(std::fabs(d.logpdf(y) - l.logpdf(y)) <= 1e-8);
      for (int K = 1; K <= 6; ++K) {
        const auto t = lderiv(d, y, K);
        const auto u = lderiv(l, y, K);
        double scale = 0.0;
        for (double v : u) scale = std::fmax(scale, std::fabs(v));
        for (int i = 0; i <= K; ++i) {
          const int j = K - i;  // i derivatives in mu, j in lambda, none in alpha
          INFO("K " << K << " mu^" << i << " lambda^" << j);
          CHECK(std::fabs(t[ld::detail_enet::pos3(K, i, 0)] - u[j]) <= 1e-8 * scale + 1e-9);
        }
      }
      CHECK(std::fabs(d.cdf(y) - l.cdf(y)) <= 1e-8);
      CHECK(std::fabs(d.ccdf(y) - l.ccdf(y)) <= 1e-8);
      CHECK(std::fabs(d.log_cdf(y) - l.log_cdf(y)) <= 1e-8 * std::fmax(1.0, std::fabs(l.log_cdf(y))));
      CHECK(std::fabs(d.log_ccdf(y) - l.log_ccdf(y)) <= 1e-8 * std::fmax(1.0, std::fabs(l.log_ccdf(y))));
      for (int K = 1; K <= 6; ++K)
        for (bool up : {false, true}) {
          const auto t = cderiv(d, y, K, up);
          const auto u = cderiv(l, y, K, up);
          double scale = 0.0;
          for (double v : u) scale = std::fmax(scale, std::fabs(v));
          for (int i = 0; i <= K; ++i) {
            INFO("log " << (up ? "(1-F)" : "F") << " K " << K << " mu^" << i << " lambda^" << K - i);
            CHECK(std::fabs(t[ld::detail_enet::pos3(K, i, 0)] - u[K - i]) <= 1e-8 * scale + 1e-9);
          }
        }
    }
    const auto ie = d.expected_info();
    const auto il = l.expected_info();
    CHECK(ie[0] == doctest::Approx(il[0]).epsilon(1e-8));
    CHECK(ie[3] == doctest::Approx(il[2]).epsilon(1e-8));
    CHECK(d.quantile(0.2) == doctest::Approx(l.quantile(0.2)).epsilon(1e-8));
    CHECK(d.quantile(0.9) == doctest::Approx(l.quantile(0.9)).epsilon(1e-8));
    CHECK(d.scale() == doctest::Approx(std::sqrt(2.0) / lam).epsilon(1e-8));
  }
}

TEST_CASE("enet: the Gaussian limit alpha -> 0 against gaussian3_distrib and gaussian1_distrib") {
  // enet at alpha = 1e-10 is the normal law with precision lambda (a = 1e-10 lambda): relative
  // changes of order a |z| / (c z^2) ~ 1e-10 in the quantities, compared at 1e-8 (1e-6 for the derivatives of log F and
  // log(1 - F), where the orders 5 and 6 amplify the O(a) change to 1e-7 of the scale of the order)
  const double al = 1e-10;
  for (double lam : {0.6, 5.0}) {
    const E d{0.4, lam, al};
    const ld::gaussian3_distrib g{0.4, lam};
    const ld::gaussian1_distrib g1{0.4, 1.0 / std::sqrt(lam)};
    const double sc = 1.0 / std::sqrt(lam);
    for (double y : {0.4 - 3 * sc, 0.4 - 0.5 * sc, 0.4 + 0.3 * sc, 0.4 + 2 * sc}) {
      INFO("lambda " << lam << " y " << y);
      CHECK(std::fabs(d.logpdf(y) - g.logpdf(y)) <= 1e-8);
      CHECK(std::fabs(d.logpdf(y) - g1.logpdf(y)) <= 1e-8);
      for (int K = 1; K <= 6; ++K) {
        const auto t = lderiv(d, y, K);
        const auto u = lderiv(g, y, K);
        double scale = 0.0;
        for (double v : u) scale = std::fmax(scale, std::fabs(v));
        for (int i = 0; i <= K; ++i) {
          INFO("K " << K << " mu^" << i << " lambda^" << K - i);
          CHECK(std::fabs(t[ld::detail_enet::pos3(K, i, 0)] - u[K - i]) <= 1e-8 * scale + 1e-9);
        }
      }
      CHECK(std::fabs(d.cdf(y) - g1.cdf(y)) <= 1e-8);
      CHECK(std::fabs(d.ccdf(y) - g1.ccdf(y)) <= 1e-8);
      CHECK(std::fabs(d.log_cdf(y) - g1.log_cdf(y)) <= 1e-8 * std::fmax(1.0, std::fabs(g1.log_cdf(y))));
      CHECK(std::fabs(d.log_ccdf(y) - g1.log_ccdf(y)) <= 1e-8 * std::fmax(1.0, std::fabs(g1.log_ccdf(y))));
      for (int K = 1; K <= 6; ++K)
        for (bool up : {false, true}) {
          const auto t = cderiv(d, y, K, up);
          const auto u = cderiv(g, y, K, up);
          double scale = 0.0;
          for (double v : u) scale = std::fmax(scale, std::fabs(v));
          for (int i = 0; i <= K; ++i) {
            INFO("log " << (up ? "(1-F)" : "F") << " K " << K << " mu^" << i << " lambda^" << K - i);
            CHECK(std::fabs(t[ld::detail_enet::pos3(K, i, 0)] - u[K - i]) <= 1e-6 * scale + 1e-9);
          }
        }
    }
    const auto ie = d.expected_info();
    const auto ig = g.expected_info();
    CHECK(ie[0] == doctest::Approx(ig[0]).epsilon(1e-8));
    CHECK(ie[3] == doctest::Approx(ig[2]).epsilon(1e-8));
    CHECK(d.quantile(0.2) == doctest::Approx(g1.quantile(0.2)).epsilon(1e-8));
    CHECK(d.quantile(0.9) == doctest::Approx(g1.quantile(0.9)).epsilon(1e-8));
    CHECK(d.scale() == doctest::Approx(g1.sigma).epsilon(1e-8));
  }
}

TEST_CASE("enet: derivatives of log F and log(1 - F) against differences") {
  for (bool upper : {false, true}) {
    {
      const double sc = 1.0 / (1.7 * 0.4 + std::sqrt(1.7 * 0.6));
      for (double z : {-30.0, -6.0, -1.0, -0.1, 0.1, 1.0, 6.0, 30.0})
        check_cdf_derivs3({0.4, 1.7, 0.4}, 0.4 + z * sc, upper, 1e-6, std::fabs(z));
    }
    {
      const double sc = 1.0 / (0.3 * 0.9 + std::sqrt(0.3 * 0.1));
      for (double z : {-6.0, -0.3, 0.3, 6.0}) check_cdf_derivs3({0.4, 0.3, 0.9}, 0.4 + z * sc, upper, 1e-6, std::fabs(z));
    }
    {
      const double sc = 1.0 / (5.0 * 0.1 + std::sqrt(5.0 * 0.9));
      for (double z : {-6.0, -0.3, 0.3, 6.0}) check_cdf_derivs3({0.4, 5.0, 0.1}, 0.4 + z * sc, upper, 1e-6, std::fabs(z));
    }
    for (double z : {-2.0, 0.5}) check_cdf_derivs3({0.4, 1.0, 0.6180339887498949}, 0.4 + z, upper, 1e-6, std::fabs(z));
    for (double z : {-2.0, 0.5}) check_cdf_derivs3({0.4, 1.0, 0.8284271247461903}, 0.4 + z, upper, 1e-6, std::fabs(z));
  }
}

TEST_CASE("enet: cdf, ccdf, logarithms, quantile round trips") {
  const std::vector<std::vector<double>> pts = {
      {0.4, 1.7, 0.4}, {-1.0, 0.3, 0.9}, {2.0, 8.0, 0.05}, {0.0, 3.0, 1.0 - 1e-9}, {0.3, 40.0, 1e-9}, {0.0, 1e5, 0.5}, {0.0, 1e-5, 0.5}};
  for (const auto& p : pts) {
    const E d = mk(p.data());
    const double sc = 1.0 / (p[1] * p[2] + std::sqrt(p[1] * (1.0 - p[2])));
    INFO("theta " << p[0] << " " << p[1] << " " << p[2]);
    for (double z : {-300.0, -30.0, -6.0, -1.0, -0.1, -1e-3, 0.0, 1e-3, 0.1, 1.0, 6.0, 30.0, 300.0}) {
      const double y = d.mu + z * sc;
      CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(2e-16));
      if (d.cdf(y) < 0.5 && d.cdf(y) > 1e-300) CHECK(rel(d.log_cdf(y), std::log(d.cdf(y)), 1e-13));
      if (d.ccdf(y) < 0.5 && d.ccdf(y) > 1e-300) CHECK(rel(d.log_ccdf(y), std::log(d.ccdf(y)), 1e-13));
      CHECK(d.log_cdf(y) <= 0.0);
      CHECK(d.log_ccdf(y) <= 0.0);
      // symmetry about mu
      const double ym = d.mu - z * sc;
      CHECK(d.cdf(y) == doctest::Approx(d.ccdf(ym)).epsilon(1e-9));
      // round trip through the quantile, on the side where the tail is the small one (relative in the tail)
      if (z != 0.0 && std::fabs(z) <= 30.0) {
        const double p1 = d.cdf(y);
        if (p1 > 0.0 && p1 < 0.99) {
          const double q = d.quantile(p1);
          INFO("z " << z << " cdf " << p1 << " q " << q << " y " << y);
          CHECK(std::fabs(q - y) <= 1e-9 * sc * (1.0 + std::fabs(z)) + 4e-16 * std::fabs(y));
        }
      }
    }
    for (double pr : {1e-300, 1e-100, 1e-12, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6, 1.0 - 1e-12}) {
      const double q = d.quantile(pr);
      INFO("p " << pr << " q " << q);
      if (pr <= 0.5) CHECK(rel(d.cdf(q), pr, 2e-12));
      else CHECK(rel(d.ccdf(q), 1.0 - pr, 2e-12));
    }
    CHECK(d.quantile(0.5) == d.mu);
    CHECK(d.quantile(0.0) == -ld::distrib_infinity);
    CHECK(d.quantile(1.0) == ld::distrib_infinity);
    CHECK(std::isnan(d.quantile(-0.1)));
  }
}

TEST_CASE("enet: the two representations of the normalizing constant agree where both are used") {
  // x_switch = 1.5: the table of A in (a, c) from x (L(x)) and from t = 1/x^2 (F(t)) at x in [1, 4]
  for (double x : {1.0, 1.5, 2.0, 3.0, 4.0}) {
    const double c = 0.7, a = x * std::sqrt(c);
    double Gx[28], Gt[28];
    ld::detail_enet::ac_table_x<6>(a, c, Gx);
    ld::detail_enet::ac_table_t<6>(a, c, Gt);
    double worst = 0.0;
    for (int n = 1; n <= 6; ++n) {
      double scale = 0.0;
      for (int b = 0; b <= n; ++b) scale = std::fmax(scale, std::fabs(Gx[ld::detail_enet::pn(n, b)]));
      for (int b = 0; b <= n; ++b)
        worst = std::fmax(worst, std::fabs(Gx[ld::detail_enet::pn(n, b)] - Gt[ld::detail_enet::pn(n, b)]) / scale);
    }
    INFO("x " << x << " worst scaled difference " << worst);
    if (printing()) std::printf("enet representations at x = %g: %.3g\n", x, worst);
    CHECK(worst <= 9e-12);  // measured 9.3e-14, 2.9e-14, 5.5e-14, 1.8e-13, 6.0e-12 at x = 1, 1.5, 2, 3, 4
  }
}

TEST_CASE("enet: start value from a sample summary") {
  ld::distrib_sample_summary s;
  s.median = 1.25;
  s.variance = 4.0;
  const E v = E{0.0, 1.0, 0.5}.start_value(s);
  CHECK(v.mu == 1.25);
  CHECK(v.alpha == 0.5);
  const E w{v.mu, v.lambda, 0.5};
  CHECK(w.scale() * w.scale() == doctest::Approx(4.0).epsilon(1e-10));
  s.variance = 0.0;
  const E z = E{0.0, 1.0, 0.5}.start_value(s);
  CHECK(z.lambda > 0.0);
  CHECK(std::isfinite(z.lambda));
}

TEST_CASE("enet: every quantity against mpmath") {
  // tools/reference/distrib_enet.py: mpmath.diff of the definitions at 300 and 400 bits. Each component is
  // compared in ulp of the largest component of its order at that point (the components change sign, so a
  // plain relative error is undefined at their zeros), and in ulp of the component itself.
  const auto rows = read_reference();
  CHECK(rows.size() > 5000);
  std::map<std::tuple<int, int, int, double, double, double, double>, double> scale;
  for (const auto& r : rows) {
    auto& s = scale[std::make_tuple(r.kind, r.K, r.kind == 3 ? 0 : r.p, r.y, r.mu, r.lam, r.alp)];
    s = std::fmax(s, std::fabs(r.hi));
  }
  double w_scaled[5][7] = {}, w_own[5][7] = {};
  reference_row worst_row[5][7] = {};
  for (const auto& r : rows) {
    const double v = library_value(r);
    const double sc = scale[std::make_tuple(r.kind, r.K, r.kind == 3 ? 0 : r.p, r.y, r.mu, r.lam, r.alp)];
    const double err = std::fabs((v - r.hi) - r.lo);
    double ulp_scaled, ulp_own;
    if (r.kind == 4) {  // the quantile: ulp of the larger of the quantile and mu
      const double ref = std::fmax(std::fabs(r.hi), std::fabs(r.mu));
      ulp_scaled = ulp_own = err / ulp_of(ref);
    } else {
      ulp_scaled = err / ulp_of(sc);
      ulp_own = err / ulp_of(r.hi);
    }
    if (ulp_scaled > w_scaled[r.kind][r.K]) worst_row[r.kind][r.K] = r;
    w_scaled[r.kind][r.K] = std::fmax(w_scaled[r.kind][r.K], ulp_scaled);
    w_own[r.kind][r.K] = std::fmax(w_own[r.kind][r.K], ulp_own);
    INFO("kind " << r.kind << " K " << r.K << " p " << r.p << " pos " << r.pos << " y " << r.y << " mu " << r.mu << " lam " << r.lam
                 << " alp " << r.alp << " ulp " << ulp_scaled);
    // bounds: the worst scaled error measured on 2026-11-15 on the gcc and the CUDA builds (identical to the digits
    // kept), times 1.5; per kind (log f, log F, log(1 - F), information, quantile) and order 0..6
    static const double bound[5][7] = {{240, 5, 9, 40, 175, 2500, 3100},
                                       {720, 740, 2700, 7400, 4200, 14400, 24800},
                                       {950, 1450, 2700, 7400, 4200, 14400, 24800},
                                       {30, 48, 175, 620, 1.18e6, 0, 0},
                                       {7.3, 0, 0, 0, 0, 0, 0}};
    CHECK(ulp_scaled <= bound[r.kind][r.K]);
  }
  if (printing())
    for (int k = 0; k < 5; ++k)
      for (int K = 0; K <= 6; ++K)
        if (w_scaled[k][K] > 0.0) {
          const auto& w = worst_row[k][K];
          std::printf(
              "enet reference kind %d order %d: max %.3g ulp of the order's scale, %.3g ulp of the component; worst at p %d pos "
              "%d y %.17g mu %.17g lam %.17g alp %.17g\n",
              k, K, w_scaled[k][K], w_own[k][K], w.p, w.pos, w.y, w.mu, w.lam, w.alp);
        }
}

// The kernel lives in a named function with external linkage, not in the body of the
// TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp for why.
void enet_kernel_case() {
  const int n = 64;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "enet_distrib", n, KOKKOS_LAMBDA(const int i) {
        const double y = -2.0 + 0.1 * i;
        const ld::enet_distrib e{0.4, 1.7, 0.4};
        out(i) = e.logpdf_d6(y)[3] + e.logpdf_dy_dk<2, 3>(y)[1] + e.expected_info_d4().pair[5][14] + e.expected_info().v[4] +
                 e.log_cdf_d6(y)[7] + e.log_ccdf_d6(y + 1.0)[2] + e.logpdf(y) + e.quantile(0.25 + 0.005 * i) + e.cdf(y) +
                 e.scale();
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -2.0 + 0.1 * i;
    const ld::enet_distrib e{0.4, 1.7, 0.4};
    const double r = e.logpdf_d6(y)[3] + e.logpdf_dy_dk<2, 3>(y)[1] + e.expected_info_d4().pair[5][14] + e.expected_info().v[4] +
                     e.log_cdf_d6(y)[7] + e.log_ccdf_d6(y + 1.0)[2] + e.logpdf(y) + e.quantile(0.25 + 0.005 * i) + e.cdf(y) +
                     e.scale();
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (printing()) std::printf("enet kernel: worst relative host-device difference %.3g\n", worst);
  // measured 1.88e-12 on the CUDA build (FMA contraction in the long sums), 0 on the gcc build; bound 1.5 times
  CHECK(worst <= 2.9e-12);
}

TEST_CASE("enet: every order inside a kernel") { enet_kernel_case(); }
