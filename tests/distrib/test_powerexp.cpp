// The test of powerexp_distrib (include/laplace/distrib/powerexp.hpp): the descriptors and the
// runtime order of the kink, the density against the definition, every derivative order against
// differences, the mixed (y, theta) derivatives, the limits at the location for each regime of p,
// the exact reductions to gaussian1_distrib at p = 2 and to laplace1_distrib at p = 1, the
// normalization and the variance sigma^2 by quadrature, the closed expected information against the
// variance of the score and against the generic quadrature, its divergence at p <= 1/2, the closed
// cdf against the generic quadrature route and against its own quantile, the references of
// tools/reference/distrib_powerexp.py, and a kernel launch per order.
#include <doctest/doctest.h>

#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/powerexp.hpp>
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
#include "generic_hidden.hpp"

namespace laplace_distrib_check {
// the shared checks build a distribution from a parameter vector, two parameters at most by default
template <>
ld::powerexp_distrib make<ld::powerexp_distrib>(const double* t) {
  return ld::powerexp_distrib{t[0], t[1], t[2]};
}
}  // namespace laplace_distrib_check

using namespace laplace_distrib_check;  // the shared checks, ld included
using ld::detail_powerexp::pos3;
using laplace_test::hide;

namespace {

using pe = ld::powerexp_distrib;

constexpr double INF = std::numeric_limits<double>::infinity();

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
        CHECK(std::fabs(exact[i] - fd) <= rel * scale + 1e-12 + stencil_floor(v, i, hy));
      }
      if (K == 0) continue;
      for (const auto& idx : sorted_indices(3, K)) {
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const int j = idx.back();
        const double h = 1e-3 * std::fmax(1e-3, std::fabs(th[j]));
        const auto fd = diff5([&](const double* t) { return yderiv3(make<D>(t), y, J, K - 1); }, th, j, h);
        std::vector<std::vector<double>> w;
        for (double s : {-2.0, -1.0, 1.0, 2.0}) {
          std::vector<double> t2 = th;
          t2[j] = th[j] + s * h;
          w.push_back(yderiv3(make<D>(t2.data()), y, J, K - 1));
        }
        const int pl = position_of(3, lower);
        INFO("dtheta: J = " << J << ", K = " << K << ", component " << position_of(3, idx) << ", y = " << y);
        CHECK(std::fabs(exact[position_of(3, idx)] - fd[pl]) <= rel * scale + 1e-12 + stencil_floor(w, pl, h));
      }
    }
}

// ---------------------------------------------------------------- the definition

// l from the definition, with the lgamma and pow of the C++ standard library and nothing of the
// library's own route: l = log p - log 2 - log s - lgamma(1/p) - |z|^p, s = sigma c(p)
double logpdf_definition(double mu, double sg, double p, double y) {
  const double c = std::sqrt(std::exp(std::lgamma(1.0 / p) - std::lgamma(3.0 / p)));
  const double s = sg * c, z = (y - mu) / s;
  return std::log(p) - std::log(2.0) - std::log(s) - std::lgamma(1.0 / p) - std::pow(std::fabs(z), p);
}

// ---------------------------------------------------------------- the references

struct ref_row {
  int K = 0, a = 0, b = 0, c = 0;
  double y = 0, hi = 0, lo = 0, mu = 0, sigma = 0, p = 0;
};

std::vector<std::string> split_csv(const char* line) {
  std::vector<std::string> f;
  std::string cur;
  for (const char* q = line; *q && *q != '\n' && *q != '\r'; ++q) {
    if (*q == ',') {
      f.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(*q);
    }
  }
  f.push_back(cur);
  return f;
}

// y,K,a,b,c,hi,lo,mu,sigma,p
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
    r.mu = num(7), r.sigma = num(8), r.p = num(9);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

// i,j,hi,lo,sigma,p
struct info_row {
  int i = 0, j = 0;
  double hi = 0, lo = 0, sigma = 0, p = 0;
};
std::vector<info_row> read_info_reference() {
  std::vector<info_row> rows;
  const std::string path = std::string(LAPLACE_REFERENCE_DIR) + "/distrib_powerexp_info.csv";
  std::FILE* f = std::fopen(path.c_str(), "r");
  REQUIRE(f != nullptr);
  char line[512];
  std::fgets(line, sizeof line, f);
  while (std::fgets(line, sizeof line, f)) {
    const auto s = split_csv(line);
    if (s.size() != 6) continue;
    info_row r;
    r.i = std::atoi(s[0].c_str());
    r.j = std::atoi(s[1].c_str());
    r.hi = std::strtod(s[2].c_str(), nullptr);
    r.lo = std::strtod(s[3].c_str(), nullptr);
    r.sigma = std::strtod(s[4].c_str(), nullptr);
    r.p = std::strtod(s[5].c_str(), nullptr);
    rows.push_back(r);
  }
  std::fclose(f);
  return rows;
}

double ulp_of(double v) { return std::ldexp(1.0, std::ilogb(v) - 52); }

// The bound is the worst scaled error measured on the gcc build and on the CUDA build (RTX 4070
// SUPER), the larger of the two, times about one and a half, in ulp of the largest component of the
// order at that point; the row of K = 0 is the value itself, in ulp of the value. Two regions of p
// only because the worst points fall on the two sides of 2.5; the cause of the loss is the same
// throughout, and it is the cancellation between the derivatives in p of Lambda and those of g,
// which both carry the derivatives of log c(p): a factor up to 677 over the grid of the reference
// file (order 4 at p = 0.8). The worst row of p >= 2.5, order 5 at p = 4, has a cancellation of 234
// there; with every shape derivative exact the same row would still lose 314 ulp. Measured on
// 2026-11-19, after the shape derivatives for p >= 2 moved to the chi route of shape_derivs: both
// builds read the same values to the digit in every entry.
const double BOUND_LOGPDF[2][7] = {{75.0, 60.0, 95.0, 660.0, 155.0, 1050.0, 410.0},
                                   {50.0, 50.0, 50.0, 45.0, 50.0, 535.0, 55.0}};
inline int p_region(double p) { return p < 2.5 ? 0 : 1; }

}  // namespace

TEST_CASE("powerexp: descriptors and the runtime order of the kink") {
  CHECK(pe::kind == ld::distrib_kind::continuous);
  CHECK(pe::n_params == 3);
  // the static field is the worst case over p; the true order is in y_kinks()
  CHECK(pe::y_regularity == 0);
  CHECK(pe::params[0].regularity == 0);
  CHECK(pe::params[0].cdf_regularity == 1);
  CHECK(pe::params[1].regularity == ld::infinitely_smooth);
  CHECK(pe::params[2].regularity == ld::infinitely_smooth);
  for (int j = 0; j < 3; ++j) CHECK(pe::params[j].moves_support == false);
  CHECK(pe::params[1].lower == 0.0);
  CHECK(pe::params[2].lower == 0.0);
  const pe d{0.4, 1.3, 1.7};
  CHECK(d.support_lower() == -ld::distrib_infinity);
  CHECK(d.support_upper() == ld::distrib_infinity);
  CHECK(d.center() == 0.4);
  CHECK(d.scale() == 1.3);
  // the kink sits at the location, in y and in mu, and sigma and p have none
  CHECK(d.y_kinks().count == 1);
  CHECK(d.y_kinks().point[0] == 0.4);
  CHECK(d.param_kinks(0, 2.25).count == 1);
  CHECK(d.param_kinks(0, 2.25).point[0] == 2.25);
  CHECK(d.param_kinks(1, 2.25).count == 0);
  CHECK(d.param_kinks(2, 2.25).count == 0);
  // ceil(p) - 1 everywhere, infinitely_smooth at p = 2 exactly: 0 at the cusp and at the Laplace,
  // 1 at p = 1.5, 2 at p = 2.5 and at p = 3 (where the third derivative jumps), 3 at p = 4
  const double ps[7] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0};
  const int want[7] = {0, 0, 1, ld::infinitely_smooth, 2, 2, 3};
  for (int i = 0; i < 7; ++i) {
    const pe e{0.4, 1.3, ps[i]};
    INFO("p = " << ps[i]);
    CHECK(e.kink_order() == want[i]);
    CHECK(e.y_kinks().continuous_order == want[i]);
    CHECK(e.param_kinks(0, 2.25).continuous_order == want[i]);
  }
  // the density is symmetric about mu
  for (double p : {0.6, 1.0, 2.0, 3.3}) {
    const pe e{0.4, 1.3, p};
    CHECK(e.logpdf(0.4 + 2.0) == doctest::Approx(e.logpdf(0.4 - 2.0)).epsilon(1e-15));
  }
}

TEST_CASE("powerexp: the density against the definition") {
  double worst = 0.0;
  for (double p : {0.4, 0.75, 1.0, 1.5, 2.0, 2.5, 4.0, 12.0})
    for (double y : {-9.0, -1.0, 0.4, 0.9, 2.0, 7.0}) {
      const pe d{0.4, 1.3, p};
      const double ref = logpdf_definition(0.4, 1.3, p, y);
      const double e = std::fabs(d.logpdf(y) - ref) / std::fmax(std::fabs(ref), 1.0);
      worst = std::fmax(worst, e);
      INFO("p = " << p << ", y = " << y);
      CHECK(e <= 1e-14);
    }
  // c(p): sqrt(2) at p = 2, 1/sqrt(2) at p = 1, and sqrt(3) in the limit
  CHECK(pe{0.0, 1.0, 2.0}.natural_scale() == doctest::Approx(std::sqrt(2.0)).epsilon(1e-15));
  CHECK(pe{0.0, 1.0, 1.0}.natural_scale() == doctest::Approx(1.0 / std::sqrt(2.0)).epsilon(1e-15));
  CHECK(pe{0.0, 1.0, 1e7}.natural_scale() == doctest::Approx(std::sqrt(3.0)).epsilon(1e-6));
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp value against the definition: worst relative %.3g\n", worst);
}

TEST_CASE("powerexp: normalization and variance sigma^2 by quadrature") {
  for (double p : {0.6, 1.0, 1.5, 2.0, 3.0, 6.0}) {
    const pe d{0.4, 1.3, p};
    const double cuts[1] = {0.4};
    auto mass = [&](double y) { return std::exp(d.logpdf(y)); };
    auto var = [&](double y) { return (y - 0.4) * (y - 0.4) * std::exp(d.logpdf(y)); };
    const double m = laplace::math::quad::panels_quad(mass, -INF, INF, 0.4, 1.3, cuts, 1).value;
    const double v = laplace::math::quad::panels_quad(var, -INF, INF, 0.4, 1.3, cuts, 1).value;
    INFO("p = " << p);
    CHECK(m == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(v == doctest::Approx(1.3 * 1.3).epsilon(1e-10));
  }
}

TEST_CASE("powerexp: derivatives against differences, away from the location") {
  for (double p : {0.45, 0.9, 1.3, 2.0, 2.7, 5.0})
    for (double y : {-6.0, -1.0, 1.1, 2.4, 6.0}) check_derivs<pe>({0.4, 1.3, p}, y, 2e-6);
}

TEST_CASE("powerexp: mixed (y, theta) derivatives against differences") {
  for (double p : {0.8, 1.6, 2.0, 3.4})
    for (double y : {-2.0, 1.1, 3.0}) check_y_derivs3<pe>({0.4, 1.3, p}, y, 1e-3, 3e-6);
  // the J = 1 route is minus the mu-derivative of the same order, and (2, 2) is the (4, ...) block
  const pe d{0.4, 1.3, 1.6};
  CHECK(d.logpdf_dy_dk<1, 0>(2.0)[0] == doctest::Approx(-d.logpdf_dk<1>(2.0)[0]).epsilon(1e-14));
  CHECK(d.logpdf_dy_dk<2, 2>(2.0)[pos3(0, 1, 1)] == d.logpdf_dk<4>(2.0)[pos3(2, 1, 1)]);
}

namespace {

// the components of the tensor of order K with a derivatives in mu, b in sigma and c in p
template <int K>
double comp(const pe& d, double y, int a, int b, int c) {
  return d.logpdf_dk<K>(y)[pos3(a, b, c)];
}

}  // namespace

TEST_CASE("powerexp: the limits at the location, one regime of p at a time") {
  const double mu = 0.4, sg = 1.3;
  // Order j in y (and in mu) is finite and zero while j < p, infinite once j > p, and at an integer
  // j = p it is finite for an even p and zero by sgn(0) = 0 for an odd one. The odd orders are zero
  // throughout, the mean of the two one-sided derivatives.
  SUBCASE("p = 0.5, the cusp: the second derivative in mu is +infinity") {
    const pe d{mu, sg, 0.5};
    CHECK(comp<1>(d, mu, 1, 0, 0) == 0.0);                   // sgn(0) = 0
    CHECK(comp<2>(d, mu, 2, 0, 0) == INF);                   // -|z|^(1/2) is convex off 0
    CHECK(std::isfinite(comp<1>(d, mu, 0, 1, 0)));
    CHECK(std::isfinite(comp<1>(d, mu, 0, 0, 1)));
    CHECK(d.logpdf(mu) == doctest::Approx(logpdf_definition(mu, sg, 0.5, mu)).epsilon(1e-14));
  }
  SUBCASE("p = 1.5: the second derivative in mu is -infinity") {
    const pe d{mu, sg, 1.5};
    CHECK(comp<1>(d, mu, 1, 0, 0) == 0.0);
    CHECK(comp<2>(d, mu, 2, 0, 0) == -INF);                  // -0.75 |z|^(-1/2)
  }
  SUBCASE("p = 2: the Gaussian, finite in (mu, sigma) and infinite in the mixed (mu, mu, p)") {
    const pe d{mu, sg, 2.0};
    CHECK(comp<2>(d, mu, 2, 0, 0) == doctest::Approx(-1.0 / (sg * sg)).epsilon(1e-15));
    CHECK(comp<4>(d, mu, 4, 0, 0) == 0.0);                   // the fourth derivative of a parabola
    // d^2 l / dzeta^2 at the location is 0 above p = 2, -2 exp(A) at p = 2 and infinite below it,
    // so its derivative in p is infinite there, although l is a polynomial in y at p = 2 exactly
    CHECK(comp<3>(d, mu, 2, 0, 1) == INF);
  }
  SUBCASE("p = 2.5: the second order is zero, the third zero by convention, the fourth infinite") {
    const pe d{mu, sg, 2.5};
    CHECK(comp<2>(d, mu, 2, 0, 0) == 0.0);
    CHECK(comp<3>(d, mu, 3, 0, 0) == 0.0);
    // fall(2.5, 4) = 2.5 * 1.5 * 0.5 * (-0.5) < 0, so -fall |z|^(p-4) exp(A) is +infinity
    CHECK(comp<4>(d, mu, 4, 0, 0) == INF);
  }
  SUBCASE("p = 3: an odd integer, so every order above it vanishes away from the location") {
    const pe d{mu, sg, 3.0};
    CHECK(comp<2>(d, mu, 2, 0, 0) == 0.0);
    CHECK(comp<3>(d, mu, 3, 0, 0) == 0.0);  // the jump of 6 sgn(z), the mean of the two sides
    // |z|^3 is a cubic on each side, so its fourth derivative is zero away from the location and
    // fall(3, 4) = 3 * 2 * 1 * 0 is zero with it; the derivative in p of that order is not
    CHECK(comp<4>(d, mu, 4, 0, 0) == 0.0);
    CHECK(comp<5>(d, mu, 4, 0, 1) == -INF);
  }
  SUBCASE("p = 4: the fourth derivative is finite, -4! exp(A), and the sixth vanishes") {
    const pe d{mu, sg, 4.0};
    const double eA = std::exp(-4.0 * pe{0.0, 1.0, 4.0}.log_scale_ratio());
    CHECK(comp<2>(d, mu, 2, 0, 0) == 0.0);
    CHECK(comp<4>(d, mu, 4, 0, 0) == doctest::Approx(-24.0 * eA / std::pow(sg, 4.0)).epsilon(1e-14));
    CHECK(comp<6>(d, mu, 6, 0, 0) == 0.0);  // fall(4, 6) = 0: |z|^4 is a quartic
  }
  // nothing at the location is NaN
  for (double p : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0}) {
    const pe d{mu, sg, p};
    for (int K = 1; K <= 6; ++K) {
      const auto v = lderiv(d, mu, K);
      for (int i = 0; i < (int)v.size(); ++i) {
        INFO("p = " << p << ", K = " << K << ", component " << i);
        CHECK(!std::isnan(v[i]));
      }
    }
    CHECK(!std::isnan(d.logpdf(mu)));
  }
}

namespace {

// At p = 2 the law is the Gaussian of standard deviation sigma: the (mu, sigma) block of the
// derivative tensor is gaussian1_distrib's, component by component. Returns the largest difference
// relative to the largest component of the Gaussian's order.
template <int K>
double gaussian_twin(double mu, double sg, double y) {
  const pe d{mu, sg, 2.0};
  const ld::gaussian1_distrib g{mu, sg};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = g.logpdf_dk<K>(y);
  double scale = 0.0, worst = 0.0;
  for (int i = 0; i <= K; ++i) scale = std::fmax(scale, std::fabs(b[i]));
  scale = std::fmax(scale, 1e-300);
  for (int i = 0; i <= K; ++i) worst = std::fmax(worst, std::fabs(a[pos3(K - i, i, 0)] - b[i]) / scale);
  return worst;
}

// At p = 1 it is the Laplace of scale b = sigma / sqrt(2), whose variance 2 b^2 is sigma^2. b is
// linear in sigma, so the component with i derivatives in sigma is the Laplace's times
// (db/dsigma)^i = 2^(-i/2).
template <int K>
double laplace_twin(double mu, double sg, double y) {
  const pe d{mu, sg, 1.0};
  const ld::laplace1_distrib l{mu, sg / std::sqrt(2.0)};
  const auto a = d.logpdf_dk<K>(y);
  const auto b = l.logpdf_dk<K>(y);
  double scale = 0.0, worst = 0.0;
  for (int i = 0; i <= K; ++i) scale = std::fmax(scale, std::fabs(b[i] * std::pow(2.0, -0.5 * i)));
  scale = std::fmax(scale, 1e-300);
  for (int i = 0; i <= K; ++i)
    worst = std::fmax(worst, std::fabs(a[pos3(K - i, i, 0)] - b[i] * std::pow(2.0, -0.5 * i)) / scale);
  return worst;
}

}  // namespace

TEST_CASE("powerexp at p = 2 is gaussian1_distrib and at p = 1 is laplace1_distrib") {
  double wg = 0.0, wl = 0.0;
  for (double y : {-5.0, -1.0, 0.9, 2.4, 8.0}) {
    const pe g{0.4, 1.3, 2.0};
    const pe l{0.4, 1.3, 1.0};
    wg = std::fmax(wg, std::fabs(g.logpdf(y) - ld::gaussian1_distrib{0.4, 1.3}.logpdf(y)) /
                           std::fabs(ld::gaussian1_distrib{0.4, 1.3}.logpdf(y)));
    wl = std::fmax(wl, std::fabs(l.logpdf(y) - ld::laplace1_distrib{0.4, 1.3 / std::sqrt(2.0)}.logpdf(y)) /
                           std::fabs(ld::laplace1_distrib{0.4, 1.3 / std::sqrt(2.0)}.logpdf(y)));
    wg = std::fmax(wg, gaussian_twin<1>(0.4, 1.3, y));
    wg = std::fmax(wg, gaussian_twin<2>(0.4, 1.3, y));
    wg = std::fmax(wg, gaussian_twin<3>(0.4, 1.3, y));
    wg = std::fmax(wg, gaussian_twin<4>(0.4, 1.3, y));
    wg = std::fmax(wg, gaussian_twin<5>(0.4, 1.3, y));
    wg = std::fmax(wg, gaussian_twin<6>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<1>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<2>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<3>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<4>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<5>(0.4, 1.3, y));
    wl = std::fmax(wl, laplace_twin<6>(0.4, 1.3, y));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp reductions: worst relative difference %.3g against gaussian1, %.3g against laplace1\n", wg, wl);
  CHECK(wg <= 1e-13);
  CHECK(wl <= 1e-13);
}

TEST_CASE("powerexp: the closed expected information") {
  // against the references of tools/reference/distrib_powerexp.py (the closed form at 60 digits,
  // itself checked there against the quadrature of the score products)
  const auto rows = read_info_reference();
  REQUIRE(rows.size() > 100);
  double worst = 0.0;
  double worst_p = 0.0;
  int worst_ij = 0;
  for (const auto& r : rows) {
    const pe d{0.0, r.sigma, r.p};
    const auto I = d.expected_info();
    const double v = I[ld::sym_tensor_position<3, 2>({r.i, r.j})];
    const double scale = std::fmax(std::fabs(r.hi), ulp_of(1.0));
    const double e = std::fabs((v - r.hi) - r.lo) / scale;
    if (e > worst) {
      worst = e;
      worst_p = r.p;
      worst_ij = 10 * r.i + r.j;
    }
    INFO("sigma = " << r.sigma << ", p = " << r.p << ", (i, j) = " << r.i << r.j);
    CHECK(e <= 1e-13);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp expected information: worst relative %.3g at p %.6g, pair %d\n", worst, worst_p, worst_ij);
  // I_mu mu is the integral of |z|^(2p-2) and diverges at p <= 1/2: the law says so exactly
  for (double p : {0.1, 0.25, 0.4, 0.5}) {
    const pe d{0.4, 1.3, p};
    INFO("p = " << p);
    CHECK(d.expected_info()[0] == INF);
    CHECK(std::isfinite(d.expected_info()[3]));
    CHECK(std::isfinite(d.expected_info()[4]));
    CHECK(std::isfinite(d.expected_info()[5]));
  }
  CHECK(std::isfinite(pe{0.4, 1.3, 0.5000001}.expected_info()[0]));
  // the off-diagonal entries that vanish by symmetry, and the orthogonality of sigma and p at p = 2
  for (double p : {0.7, 1.0, 2.0, 3.0}) {
    const pe d{0.4, 1.3, p};
    CHECK(d.expected_info()[1] == 0.0);
    CHECK(d.expected_info()[2] == 0.0);
  }
  CHECK(pe{0.4, 1.3, 2.0}.expected_info()[4] == 0.0);
  // the Gaussian: diag(1/sigma^2, 2/sigma^2) in the (mu, sigma) block
  const auto g = pe{0.4, 1.3, 2.0}.expected_info();
  CHECK(g[0] == doctest::Approx(1.0 / (1.3 * 1.3)).epsilon(1e-14));
  CHECK(g[3] == doctest::Approx(2.0 / (1.3 * 1.3)).epsilon(1e-14));
  // the Laplace: diag(1/b^2, 1/b^2) in (mu, b) with b = sigma/sqrt 2, which is (2/sigma^2, 1/sigma^2)
  // in (mu, sigma) because db/dsigma = 1/sqrt 2
  const auto l = pe{0.4, 1.3, 1.0}.expected_info();
  CHECK(l[0] == doctest::Approx(2.0 / (1.3 * 1.3)).epsilon(1e-14));
  CHECK(l[3] == doctest::Approx(1.0 / (1.3 * 1.3)).epsilon(1e-14));
}

TEST_CASE("powerexp: the closed expected information against the variance of the score") {
  // E[l_a l_b] by quadrature with a cut at the location; the mu-mu entry converges only well above
  // p = 1/2, where |z|^(2p-2) is integrable with room to spare
  double worst = 0.0;
  for (double p : {1.2, 1.5, 2.0, 3.0, 5.0}) {
    const pe d{0.4, 1.3, p};
    const auto num = score_variance_continuous(d);
    const auto I = d.expected_info();
    double scale = 0.0;
    for (int i = 0; i < 6; ++i) scale = std::fmax(scale, std::fabs(I[i]));
    for (int i = 0; i < 6; ++i) {
      const double e = std::fabs(num[i] - I[i]) / scale;
      worst = std::fmax(worst, e);
      INFO("p = " << p << ", pair " << i);
      CHECK(e <= 2e-8);
    }
  }
  // the same against the generic score-form route, which differentiates nothing of the closed form
  for (double p : {1.5, 2.0, 3.0}) {
    const pe d{0.4, 1.3, p};
    const auto gen = ld::expected_info_by_integral<0, ld::info_form::score>(hide(d));
    const auto I = d.expected_info();
    double scale = 0.0;
    for (int i = 0; i < 6; ++i) scale = std::fmax(scale, std::fabs(I[i]));
    for (int i = 0; i < 6; ++i) {
      const double e = std::fabs(gen[i] - I[i]) / scale;
      worst = std::fmax(worst, e);
      INFO("generic: p = " << p << ", pair " << i);
      CHECK(e <= 2e-8);
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp information against the quadratures: worst relative %.3g\n", worst);
}

TEST_CASE("powerexp: the closed cdf against the generic quadrature and against the quantile") {
  double worst_c = 0.0, worst_q = 0.0;
  for (double p : {0.6, 1.0, 1.5, 2.0, 3.0, 6.0}) {
    const pe d{0.4, 1.3, p};
    for (double z : {-9.0, -3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 9.0}) {
      const double y = 0.4 + 1.3 * z;
      // the two sides against the quadrature of the density (generic_cdf.hpp)
      const double lc = d.log_cdf(y), lu = d.log_ccdf(y);
      const double gc = ld::log_cdf_by_integral(d, y), gu = ld::log_ccdf_by_integral(d, y);
      const double ec = std::fabs(lc - gc) / std::fmax(1.0, std::fabs(lc));
      const double eu = std::fabs(lu - gu) / std::fmax(1.0, std::fabs(lu));
      worst_c = std::fmax(worst_c, std::fmax(ec, eu));
      INFO("p = " << p << ", z = " << z);
      CHECK(ec <= 1e-11);
      CHECK(eu <= 1e-11);
      // F and 1 - F add to one, and the logarithms match their exponentials
      CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-15));
      CHECK(std::exp(lc) == doctest::Approx(d.cdf(y)).epsilon(1e-14));
      CHECK(std::exp(lu) == doctest::Approx(d.ccdf(y)).epsilon(1e-14));
    }
    // the far tails are half an upper regularized gamma and keep their relative accuracy. The
    // point is placed where |z|^p is about 100, so that the tail is about e^-100 whatever p is; at
    // a fixed |z| = 40 the tail of p = 6 is e^(-2.4e8) and underflows to zero.
    const double zfar = std::exp(d.log_scale_ratio()) * std::pow(100.0, 1.0 / p);
    for (double z : {-zfar, zfar}) {
      const double y = 0.4 + 1.3 * z;
      const double side = z < 0 ? d.cdf(y) : d.ccdf(y);
      INFO("tail: p = " << p << ", z = " << z);
      CHECK(side > 0.0);
      CHECK(std::log(side) == doctest::Approx(z < 0 ? d.log_cdf(y) : d.log_ccdf(y)).epsilon(1e-13));
    }
    // the quantile inverts the cdf, and the generic root-finding route agrees with it
    for (double u : {1e-8, 1e-3, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0 - 1e-3}) {
      const double y = d.quantile(u);
      const double e = std::fabs(d.cdf(y) - u) / u;
      worst_q = std::fmax(worst_q, e);
      INFO("quantile: p = " << p << ", u = " << u);
      CHECK(e <= 1e-11);
      CHECK(ld::distrib_quantile(hide(d), u) == doctest::Approx(y).epsilon(1e-8));
    }
    CHECK(d.quantile(0.5) == doctest::Approx(0.4).epsilon(1e-15));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp cdf against the quadrature: worst relative %.3g; quantile: worst %.3g\n", worst_c, worst_q);
}

TEST_CASE("powerexp: the starting value from a sample") {
  ld::distrib_sample_summary s;
  s.count = 50.0;
  s.mean = 2.5;
  s.variance = 4.0;
  const pe d = pe{0.0, 1.0, 1.0}.start_value(s);
  CHECK(d.mu == doctest::Approx(2.5));
  CHECK(d.sigma == doctest::Approx(2.0));
  CHECK(d.p == doctest::Approx(2.0));
}

TEST_CASE("powerexp: the log-density against the mpmath references") {
  const auto rows = read_reference("distrib_powerexp.csv");
  REQUIRE(rows.size() > 1000);
  std::map<std::tuple<double, double, double, double, int>, double> scale;
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.y, r.mu, r.sigma, r.p, r.K);
    double& s = scale[key];
    s = std::fmax(s, std::fabs(r.hi));
  }
  std::map<std::tuple<double, double, double, double, int>, std::vector<double>> cache;
  double worst[7] = {};
  std::tuple<double, double, double, int, int> where[7];
  std::map<double, double> by_p;
  double by_region[2][7] = {};
  for (const auto& r : rows) {
    const auto key = std::make_tuple(r.y, r.mu, r.sigma, r.p, r.K);
    auto it = cache.find(key);
    if (it == cache.end()) {
      if (cache.size() > 64) cache.clear();
      it = cache.emplace(key, lderiv(pe{r.mu, r.sigma, r.p}, r.y, r.K)).first;
    }
    const double v = it->second[r.K == 0 ? 0 : pos3(r.a, r.b, r.c)];
    const double err = std::fabs((v - r.hi) - r.lo);
    const double ulp_scaled = err / ulp_of(std::fmax(scale[key], 1e-300));
    by_p[r.p] = std::fmax(by_p[r.p], ulp_scaled);
    by_region[p_region(r.p)][r.K] = std::fmax(by_region[p_region(r.p)][r.K], ulp_scaled);
    if (ulp_scaled > worst[r.K]) {
      worst[r.K] = ulp_scaled;
      where[r.K] = std::make_tuple(r.y, r.sigma, r.p, r.b, r.c);
    }
    INFO("y " << r.y << " K " << r.K << " (a,b,c) " << r.a << r.b << r.c << " mu " << r.mu
              << " sigma " << r.sigma << " p " << r.p << " ulp " << ulp_scaled);
    CHECK(ulp_scaled <= BOUND_LOGPDF[p_region(r.p)][r.K]);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    for (int K = 0; K <= 6; ++K)
      std::printf("powerexp logpdf, order %d: max %.3g ulp of the order's scale (y %.6g sigma %.4g p %.6g, b %d c %d)\n",
                  K, worst[K], std::get<0>(where[K]), std::get<1>(where[K]), std::get<2>(where[K]),
                  std::get<3>(where[K]), std::get<4>(where[K]));
    std::printf("%s", "powerexp logpdf, worst scaled ulp by region and order:\n");
    for (int reg = 0; reg < 2; ++reg) {
      std::printf("  region %d (%s):", reg, reg == 0 ? "p < 2.5" : "p >= 2.5");
      for (int K = 0; K <= 6; ++K) std::printf(" %.3g", by_region[reg][K]);
      std::printf("%s", "\n");
    }
    std::printf("powerexp logpdf, worst scaled ulp by p (all orders):");
    for (const auto& kv : by_p) std::printf(" %.6g:%.3g", kv.first, kv.second);
    std::printf("%s", "\n");
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE:
// doctest gives each case a static function whose counter restarts per translation unit, and nvcc
// tags every extended device lambda with that address, so two files that reach the same counter get
// the same mangled tag (CLAUDE.md). One kernel per derivative order: the stack frame of a kernel is
// that of its largest call.
//
// The points are y = -3 + i/4, exactly representable, so that host and device evaluate the law at the
// same y, and none of them is the location 0.4 or near it. The first version used -3 + 0.2 i, which
// rounds: at i = 17 it lands a few ulp from the location on the host, and the fused multiply-add of
// the device puts it a different number of ulp away. There the derivatives grow like |zeta|^(p-a), so
// a factor 2 in zeta became a factor 2^(a-p) in the component, and the comparison read 0.32 at order 2
// and 41 at order 6 for p = 0.6 on CUDA (exactly 2^0.4 - 1 and 2^5.4 - 1, give or take the other
// components), while gcc, where host and "device" run the same code, read 0.
template <int K>
void powerexp_kernel_order(double p, double bound) {
  const int n = 40;
  Kokkos::View<double*> out("out", n), mag("mag", n);
  Kokkos::parallel_for(
      "powerexp_order", n, KOKKOS_LAMBDA(const int i) {
        const ld::powerexp_distrib a{0.4, 1.3, p};
        const auto t = a.logpdf_dk<K>(-3.0 + 0.25 * i);
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
    const ld::powerexp_distrib a{0.4, 1.3, p};
    const auto t = a.logpdf_dk<K>(-3.0 + 0.25 * i);
    double s = 0.0, m = 0.0;
    for (int j = 0; j < ld::sym_tensor<3, K>::size; ++j) {
      s += t[j];
      m += std::fabs(t[j]);
    }
    worst = std::fmax(worst, std::fabs(h(i) - s) / m);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp kernel order %d, p %g: worst host-device difference %.3g of the sum of |components|\n", K, p, worst);
  CHECK(worst <= bound);
}

void powerexp_kernel_other(double p) {
  const int n = 12;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "powerexp_other", n, KOKKOS_LAMBDA(const int i) {
        const double y = -3.0 + 0.5 * i;
        const ld::powerexp_distrib a{0.4, 1.3, p};
        out(i) = a.logpdf(y) + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] + a.log_cdf(y) + a.log_ccdf(y) +
                 a.expected_info()[5] + a.quantile(0.3) + double(a.kink_order());
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  double worst = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = -3.0 + 0.5 * i;
    const ld::powerexp_distrib a{0.4, 1.3, p};
    const double r = a.logpdf(y) + a.logpdf_dy_dk<2, 3>(y)[pos3(1, 1, 1)] + a.log_cdf(y) + a.log_ccdf(y) +
                     a.expected_info()[5] + a.quantile(0.3) + double(a.kink_order());
    worst = std::fmax(worst, std::fabs(h(i) - r) / std::fmax(std::fabs(r), 1e-300));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("powerexp kernel (value, dy_dk<2,3>, cdf, information, quantile), p %g: worst relative %.3g\n", p, worst);
  CHECK(worst <= 1e-14);  // measured 6.4e-15 on CUDA (p = 1.6), 0 on gcc
}

// The infinities at the location must come from the explicit guard and not from a cancellation,
// which fused multiply-add defeats on the device (CLAUDE.md, yeo_johnson): the comparison is exact.
void powerexp_kernel_location() {
  const int n = 4;
  const double ps[4] = {0.5, 1.5, 2.5, 4.0};
  Kokkos::View<double*> out("out", n);
  Kokkos::View<double*, Kokkos::LayoutRight> pv("pv", n);
  auto hp = Kokkos::create_mirror_view(pv);
  for (int i = 0; i < n; ++i) hp(i) = ps[i];
  Kokkos::deep_copy(pv, hp);
  Kokkos::parallel_for(
      "powerexp_location", n, KOKKOS_LAMBDA(const int i) {
        const ld::powerexp_distrib a{0.4, 1.3, pv(i)};
        out(i) = a.logpdf_dk<2>(0.4)[pos3(2, 0, 0)];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  CHECK(h(0) == INF);   // p = 0.5
  CHECK(h(1) == -INF);  // p = 1.5
  CHECK(h(2) == 0.0);   // p = 2.5, the second derivative is continuous and zero
  CHECK(h(3) == 0.0);   // p = 4
}

void powerexp_kernel_case() {
  // worst host-device differences measured on 2026-11-18 on the CUDA build (RTX 4070 SUPER), at the
  // exact points y = -3 + i/4, times about one and a half; on gcc host and "device" run the same code
  // and the difference is 0
  powerexp_kernel_order<1>(1.6, 4e-16);
  powerexp_kernel_order<2>(1.6, 7.5e-16);
  powerexp_kernel_order<3>(1.6, 8e-16);
  powerexp_kernel_order<4>(1.6, 2.9e-15);
  powerexp_kernel_order<5>(1.6, 1.15e-14);
  powerexp_kernel_order<6>(1.6, 9.7e-15);
  powerexp_kernel_order<1>(0.6, 1.1e-15);
  powerexp_kernel_order<3>(0.6, 1.9e-15);
  powerexp_kernel_order<6>(0.6, 9.7e-15);
  powerexp_kernel_order<6>(4.0, 1.1e-15);
  powerexp_kernel_other(1.6);
  powerexp_kernel_other(0.6);
  powerexp_kernel_location();
}

TEST_CASE("powerexp: every order inside a kernel") { powerexp_kernel_case(); }
