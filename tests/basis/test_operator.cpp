// Linear differential operators and their penalties
// (include/laplace/basis/operator.hpp): coefficients from the roots, products
// with merged multiplicities, the null space, and operator_penalty on the
// three bases. The penalties are checked against the cross Gram matrices
// (D^m gives G_mm exactly), against the closed form on a full Fourier period
// (the block of harmonic j is |P(i j nu)|^2 omega/2 times the identity), and
// by the null space: a null-space function that the basis represents exactly
// has zero penalty, to rounding.
#include <doctest/doctest.h>

#include <laplace/basis/bspline.hpp>
#include <laplace/basis/fourier.hpp>
#include <laplace/basis/legendre.hpp>
#include <laplace/basis/operator.hpp>

#include <cmath>
#include <vector>

namespace basis_operator_test {

namespace lb = laplace::basis;
constexpr double two_pi = 6.283185307179586476925286766559;

double max_abs(const std::vector<double>& v) {
  double m = 0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// beta' S beta, S column-major K x K
double quad_form(const std::vector<double>& S, const std::vector<double>& beta) {
  const size_t K = beta.size();
  double s = 0;
  for (size_t i = 0; i < K; ++i)
    for (size_t j = 0; j < K; ++j) s += beta[i] * S[i + j * K] * beta[j];
  return s;
}

TEST_CASE("operator: coefficients, products, null space") {
  const double T = 365.25, nu = two_pi / T;
  auto check = [](const lb::differential_operator& op, std::vector<double> want) {
    const auto c = op.coefficients();
    REQUIRE(c.size() == want.size());
    for (size_t k = 0; k < c.size(); ++k) CHECK(c[k] == doctest::Approx(want[k]).epsilon(1e-15).scale(1e-300));
    CHECK(op.order() == static_cast<int>(want.size()) - 1);
  };
  check(lb::derivative_operator(2), {0, 0, 1});
  check(lb::oscillator_operator(T), {nu * nu, 0, 1});
  check(lb::harmonic_operator(T), {0, nu * nu, 0, 1});
  check(lb::derivative_operator(2) * lb::oscillator_operator(T), {0, 0, nu * nu, 0, 1});
  // (D^2 + nu^2)(D^2 + 4 nu^2) = D^4 + 5 nu^2 D^2 + 4 nu^4
  check(lb::oscillator_operator(T, 2), {4 * std::pow(nu, 4), 0, 5 * nu * nu, 0, 1});
  // a D + D^2: roots 0 and -a
  const double a = std::log(2.0) / 3.0;
  check(lb::differential_operator_from_roots({{0, 0, 1}, {-a, 0, 1}}), {0, a, 1});
  // multiplicities merge: D D^2 = D^3
  const auto d3 = lb::derivative_operator(1) * lb::derivative_operator(2);
  REQUIRE(d3.roots().size() == 1);
  CHECK(d3.roots()[0].multiplicity == 3);

  // null space values: 1, x', x'^2 for D^3; 1, cos, sin for the harmonic one
  const double lower = 2.0, x[3] = {2.0, 2.5, 4.0};
  std::vector<double> v(9);
  d3.null_space_values(lower, x, 3, v.data());
  for (int i = 0; i < 3; ++i) {
    const double t = x[i] - lower;
    CHECK(v[i] == 1.0);
    CHECK(v[i + 3] == doctest::Approx(t));
    CHECK(v[i + 6] == doctest::Approx(t * t));
  }
  const auto h = lb::harmonic_operator(T).null_space();
  REQUIRE(h.size() == 3);
  CHECK(h[0].frequency == 0.0);
  CHECK(h[1].frequency == doctest::Approx(nu));
  CHECK(!h[1].sine);
  CHECK(h[2].sine);
  const auto e = lb::differential_operator_from_roots({{0, 0, 1}, {-a, 0, 1}});
  e.null_space_values(lower, x, 3, v.data());
  for (int i = 0; i < 3; ++i) CHECK(v[i + 3] == doctest::Approx(std::exp(-a * (x[i] - lower))));

  CHECK_THROWS_AS(lb::derivative_operator(0), std::invalid_argument);
  CHECK_THROWS_AS(lb::oscillator_operator(-1.0), std::invalid_argument);
  CHECK_THROWS_AS(lb::differential_operator_from_roots({{0, -1, 1}}), std::invalid_argument);
  CHECK_THROWS_AS(lb::differential_operator_from_roots({{0, 0, 0}}), std::invalid_argument);
}

TEST_CASE("operator: from weights") {
  auto roots_of = [](const lb::differential_operator& op) { return op.roots(); };
  auto same_coefficients = [](const lb::differential_operator& op, const std::vector<double>& w) {
    const auto c = op.coefficients();
    REQUIRE(c.size() == w.size() + 1);
    double wmax = 0;
    for (double v : w) wmax = std::max(wmax, std::fabs(v));
    for (size_t j = 0; j < w.size(); ++j) CHECK(std::fabs(c[j] - w[j]) <= 1e-12 * std::max(wmax, 1.0));
  };
  // a D + D^2 (the book's linear_operator(w = c(0, a))): roots 0 and -a, 0 exact
  const double a = std::log(2.0) / 3.0;
  auto op = lb::differential_operator_from_weights({0.0, a});
  same_coefficients(op, {0.0, a});
  auto r = roots_of(op);
  REQUIRE(r.size() == 2);
  bool zero = false, minus_a = false;
  for (const auto& q : r) {
    if (q.re == 0.0 && q.im == 0.0) zero = true;
    if (std::fabs(q.re + a) <= 1e-14 && q.im == 0.0) minus_a = true;
  }
  CHECK(zero);
  CHECK(minus_a);
  // D^2 (D^2 + nu^2) = D^4 + nu^2 D^2: 0 double, +- i nu
  const double nu = two_pi / 365.25;
  op = lb::differential_operator_from_weights({0.0, 0.0, nu * nu, 0.0});
  r = roots_of(op);
  REQUIRE(r.size() == 2);
  for (const auto& q : r) {
    if (q.im == 0.0) { CHECK(q.re == 0.0); CHECK(q.multiplicity == 2); }
    else { CHECK(q.re == 0.0); CHECK(q.im == doctest::Approx(nu).epsilon(1e-10)); CHECK(q.multiplicity == 1); }
  }
  // (D + 1)^3: a triple root, found to about eps^(1/3) and grouped
  op = lb::differential_operator_from_weights({1.0, 3.0, 3.0});
  r = roots_of(op);
  REQUIRE(r.size() == 1);
  CHECK(r[0].multiplicity == 3);
  CHECK(r[0].re == doctest::Approx(-1.0).epsilon(1e-5));
  same_coefficients(lb::differential_operator_from_weights({2.0, -3.0}), {2.0, -3.0});  // roots 1, 2
  // complex pair off the axis: (D + 0.5)^2 + 4 = D^2 + D + 4.25
  op = lb::differential_operator_from_weights({4.25, 1.0});
  r = roots_of(op);
  REQUIRE(r.size() == 1);
  CHECK(r[0].re == doctest::Approx(-0.5).epsilon(1e-13));
  CHECK(r[0].im == doctest::Approx(2.0).epsilon(1e-13));
  // all weights zero: D^m
  CHECK(lb::differential_operator_from_weights({0.0, 0.0}).roots()[0].multiplicity == 2);
  CHECK_THROWS_AS(lb::differential_operator_from_weights({}), std::invalid_argument);
}

TEST_CASE("operator: D^m gives the Gram matrix G_mm") {
  const auto owner = lb::bspline_basis_owner::uniform_clamped(0.0, 1.0, 12, 3);
  const auto b = owner.host();
  const int K = b.dimension();
  std::vector<double> S(K * K), G(K * K);
  for (int m = 1; m <= 3; ++m) {
    lb::operator_penalty(b, lb::derivative_operator(m), S.data());
    b.cross_gram(m, m, G.data());
    for (int i = 0; i < K * K; ++i) CHECK(S[i] == G[i]);
  }
  CHECK_THROWS_AS(lb::operator_penalty(b, lb::derivative_operator(4), S.data()), std::invalid_argument);
  const auto leg = lb::make_legendre_basis(0.0, 1.0, 3);
  std::vector<double> L(16);
  CHECK_THROWS_AS(lb::operator_penalty(leg, lb::derivative_operator(4), L.data()), std::invalid_argument);
  const auto fb = lb::make_fourier_basis(0.0, 1.0, 3);
  std::vector<double> F(49);
  CHECK_THROWS_AS(lb::operator_penalty(fb, lb::derivative_operator(7), F.data()), std::invalid_argument);
}

TEST_CASE("operator: Fourier on a full period, closed form and null space") {
  const double T = 366.0, nu = two_pi / T;
  for (int harmonics : {1, 2}) {
    const auto fb = lb::make_fourier_basis(0.0, T, 6);
    const int K = fb.dimension();
    const auto op = lb::harmonic_operator(T, harmonics);  // P(r) = r prod_h (r^2 + (h nu)^2)
    std::vector<double> S(K * K);
    lb::operator_penalty(fb, op, S.data());
    const double smax = max_abs(S);
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < K; ++j) {
        double want = 0.0;
        if (i == j && i > 0) {
          const double w = ((i + 1) / 2) * nu;  // harmonic of column i, times nu
          double p2 = w * w;                    // |i w|^2
          for (int h = 1; h <= harmonics; ++h) p2 *= std::pow(h * h * nu * nu - w * w, 2);
          want = p2 * T / 2;
        }
        INFO("H=" << harmonics << " i=" << i << " j=" << j);
        CHECK(std::fabs(S[i + j * K] - want) <= 64 * 2.2e-16 * smax);
      }
  }
}

TEST_CASE("operator: null-space functions of a basis have zero penalty") {
  // Legendre, D^2: the polynomials of degree < 2 (columns 0 and 1)
  const auto leg = lb::make_legendre_basis(-1.0, 3.0, 8);
  std::vector<double> S(81);
  lb::operator_penalty(leg, lb::derivative_operator(2), S.data());
  const double smax = max_abs(S);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 9; ++j) CHECK(std::fabs(S[i + j * 9]) <= 1e-13 * smax);
  // B-spline, D^3 at degree 3: the quadratics, represented exactly through
  // the coefficients of 1, x, x^2 (Marsden's identity: x = sum xi_j B_j,
  // x^2 = sum (t_{j+1} t_{j+2} + t_{j+1} t_{j+3} + t_{j+2} t_{j+3})/3 B_j)
  const auto owner = lb::bspline_basis_owner::clamped(0.0, 2.0, {0.3, 0.7, 1.1, 1.6}, 3);
  const auto b = owner.host();
  const int K = b.dimension();
  const auto& t = owner.knots();
  std::vector<double> Sb(K * K), one(K, 1.0), lin(K), quad(K), rnd(K);
  lb::operator_penalty(b, lb::derivative_operator(3), Sb.data());
  b.greville(lin.data());
  for (int j = 0; j < K; ++j) {
    quad[j] = (t[j + 1] * t[j + 2] + t[j + 1] * t[j + 3] + t[j + 2] * t[j + 3]) / 3.0;
    rnd[j] = std::sin(1.3 * j + 0.2);
  }
  const double ref = quad_form(Sb, rnd);
  REQUIRE(ref > 0);
  CHECK(std::fabs(quad_form(Sb, one)) <= 1e-13 * ref);
  CHECK(std::fabs(quad_form(Sb, lin)) <= 1e-13 * ref);
  CHECK(std::fabs(quad_form(Sb, quad)) <= 1e-12 * ref);
}

}  // namespace basis_operator_test
