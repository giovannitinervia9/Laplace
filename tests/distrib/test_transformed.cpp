// transformed_distrib: the law of g(X) for a fixed scalar bijector g
// (note/piano_wrapper.md section 5quater).
//
// Two exact oracles, both laws already in the library:
//   * gaussian2 carried through exp (log inverted) is lognormal1 with the same
//     two parameters, so every quantity must agree component by component;
//   * the exponential with mean mu carried through x^(1/k) is the Weibull with
//     shape k and mean m = mu^(1/k) Gamma(1 + 1/k): the value, the cumulative
//     functions, the quantile and the derivatives in y agree directly, the
//     derivatives in mu after Faa di Bruno through m(mu), and the information
//     as I_mm (dm/dmu)^2;
// a decreasing affine map over gaussian2 is gaussian2 again; then the
// derivatives in y against central differences, the Yeo-Johnson, Box-Cox and
// bounded cases integrated through the generic expectation route, the atoms of
// a zero-adjusted gamma and of an atom on a gaussian under increasing and
// decreasing maps, the starting value and the refusals.
#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/exponential.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/generic_cdf.hpp>
#include <laplace/distrib/generic_expectation.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <laplace/distrib/transformed.hpp>
#include <laplace/distrib/weibull.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "check_common.hpp"
#include <doctest/doctest.h>

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace lb = laplace::bijector;

using tgau = ld::transformed_distrib<ld::gaussian2_distrib>;
using texp = ld::transformed_distrib<ld::exponential_distrib>;
using tgam = ld::transformed_distrib<ld::gamma2_distrib>;
using zag = ld::atom_mixture_distrib<ld::gamma2_distrib, ld::adjusted>;
using zan1 = ld::atom_mixture_distrib<ld::gaussian1_distrib, ld::adjusted>;
using zan2 = ld::atom_mixture_distrib<ld::gaussian1_distrib, ld::adjusted, ld::adjusted>;

namespace {

lb::scalar_bijector exp_map() {
  lb::scalar_bijector b{lb::scalar_bijector_map::log};
  b.inverted = true;
  return b;
}
lb::scalar_bijector inverted(lb::scalar_bijector b) {
  b.inverted = !b.inverted;
  return b;
}

// max |a - b| over max(|b|, floor)
double rel_err(const std::vector<double>& a, const std::vector<double>& b, double floor = 1e-300) {
  double s = floor, e = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    s = std::fmax(s, std::fabs(b[i]));
    e = std::fmax(e, std::fabs(a[i] - b[i]));
  }
  return e / s;
}

// d^J_y l with no derivative in theta, any number of parameters
template <class D>
double ydiff0(const D& d, double y, int J) {
  switch (J) {
    case 0: return d.template logpdf_dy_dk<0, 0>(y)[0];
    case 1: return d.template logpdf_dy_dk<1, 0>(y)[0];
    case 2: return d.template logpdf_dy_dk<2, 0>(y)[0];
    case 3: return d.template logpdf_dy_dk<3, 0>(y)[0];
    case 4: return d.template logpdf_dy_dk<4, 0>(y)[0];
    case 5: return d.template logpdf_dy_dk<5, 0>(y)[0];
    default: return d.template logpdf_dy_dk<6, 0>(y)[0];
  }
}

// every (J, K) of a two-parameter law against the 5-point difference in y of (J - 1, K); the worst relative
// error over the components, the stencil's rounding floor taken off
template <class D>
double y_derivs_against_differences(const D& d, double y, double hy) {
  double worst = 0.0;
  for (int J = 1; J <= 6; ++J)
    for (int K = 0; J + K <= 6; ++K) {
      const auto exact = yderiv(d, y, J, K);
      double scale = 1e-12;
      for (double e : exact) scale = std::fmax(scale, std::fabs(e));
      std::vector<std::vector<double>> v;
      for (double s : {-2.0, -1.0, 1.0, 2.0}) v.push_back(yderiv(d, y + s * hy, J - 1, K));
      for (int i = 0; i < (int)exact.size(); ++i) {
        const double fd = (v[0][i] - 8.0 * v[1][i] + 8.0 * v[2][i] - v[3][i]) / (12.0 * hy);
        const double err = std::fmax(0.0, std::fabs(exact[i] - fd) - stencil_floor(v, i, hy)) / scale;
        worst = std::fmax(worst, err);
      }
    }
  return worst;
}

// d^K log F (or log(1 - F)) of a law built from theta against the difference of order K - 1
template <class B>
double tail_derivs_against_differences(const B& build, const std::vector<double>& th, double y, bool upper) {
  using D = std::decay_t<decltype(build(th.data()))>;
  constexpr int P = D::n_params;
  double worst = 0.0;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = cderiv(build(th.data()), y, K, upper);
    double scale = 1e-12;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return cderiv(build(t), y, K - 1, upper); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      worst = std::fmax(worst, std::fabs(exact[position_of(P, idx)] - fd[position_of(P, lower)]) / scale);
    }
  }
  return worst;
}

// the integral of the density over the support, cut at the kinks, plus the atoms' masses
template <class D>
double total_mass(const D& d) {
  const auto q = ld::expectation_by_integral(d, [](double) { return 1.0; });
  REQUIRE(q.converged);
  double s = q.value;
  for (int i = 0; i < ld::distrib_n_atoms<D>(); ++i) s += std::exp(ld::distrib_atom_log_mass(d, i));
  return s;
}

}  // namespace

TEST_CASE("transformed: the descriptors are the parent's") {
  const tgau d = ld::transformed(ld::gaussian2_distrib{0.3, 0.64}, exp_map());
  CHECK(tgau::n_params == 2);
  CHECK(tgau::params == ld::gaussian2_distrib::params);
  CHECK(tgau::kind == ld::distrib_kind::continuous);
  CHECK(!d.decreasing);
  CHECK(d.support_lower() == 0.0);
  CHECK(d.support_upper() == ld::distrib_infinity);
}

TEST_CASE("transformed: gaussian2 through exp is lognormal1") {
  const double mu = 0.3, v = 0.64;
  const tgau d = ld::transformed(ld::gaussian2_distrib{mu, v}, exp_map());
  const ld::lognormal1_distrib o{mu, v};
  double e_val = 0, e_dk = 0, e_cdf = 0, e_cdk = 0, e_q = 0, e_info = 0, e_dy = 0;
  for (double y : {0.02, 0.3, 1.0, 1.35, 2.5, 8.0, 40.0}) {
    e_val = std::fmax(e_val, rel_err({d.logpdf(y)}, {o.logpdf(y)}, 1.0));
    for (int K = 1; K <= 6; ++K) e_dk = std::fmax(e_dk, rel_err(lderiv(d, y, K), lderiv(o, y, K)));
    for (bool up : {false, true}) {
      e_cdf = std::fmax(e_cdf, rel_err(cderiv(d, y, 0, up), cderiv(o, y, 0, up)));
      for (int K = 1; K <= 6; ++K) e_cdk = std::fmax(e_cdk, rel_err(cderiv(d, y, K, up), cderiv(o, y, K, up)));
    }
    for (int J = 0; J <= 6; ++J)
      for (int K = 0; J + K <= 6; ++K) e_dy = std::fmax(e_dy, rel_err(yderiv(d, y, J, K), yderiv(o, y, J, K), 1.0));
  }
  for (double p : {1e-12, 1e-3, 0.2, 0.5, 0.8, 0.999, 1.0 - 1e-12})
    e_q = std::fmax(e_q, rel_err({d.quantile(p)}, {o.quantile(p)}));
  for (int K = 0; K <= 4; ++K) e_info = std::fmax(e_info, rel_err(ideriv(d, K), ideriv(o, K)));
  std::printf("transformed lognormal oracle: value %.2e  d1-d6 %.2e  log F %.2e  its d1-d6 %.2e  quantile %.2e  "
              "info %.2e  dy_dk %.2e\n", e_val, e_dk, e_cdf, e_cdk, e_q, e_info, e_dy);
  CHECK(e_val <= 4e-16);
  CHECK(e_dk <= 4e-16);
  CHECK(e_cdf <= 4e-16);
  CHECK(e_cdk <= 4e-16);
  CHECK(e_q <= 4e-16);
  CHECK(e_info <= 4e-16);
  CHECK(e_dy <= 1.5e-15);
}

TEST_CASE("transformed: the exponential through a power is the Weibull") {
  const double mu = 1.8, k = 1.7, a = 1.0 / k, c = std::exp(std::lgamma(1.0 + a));
  const texp d = ld::transformed(ld::exponential_distrib{mu}, lb::scalar_bijector::power(a));
  const ld::weibull_distrib o{c * std::pow(mu, a), k};
  // d^j m / dmu^j of m(mu) = c mu^a
  double mj[7] = {0.0};
  {
    double f = c;
    for (int j = 1; j <= 6; ++j) {
      f *= a - (j - 1);
      mj[j] = f * std::pow(mu, a - j);
    }
  }
  double e_val = 0, e_cdf = 0, e_q = 0, e_dk = 0, e_dy = 0;
  for (double y : {0.01, 0.4, 1.0, 1.9, 3.5, 7.0}) {
    e_val = std::fmax(e_val, rel_err({d.logpdf(y)}, {o.logpdf(y)}, 1.0));
    for (bool up : {false, true}) e_cdf = std::fmax(e_cdf, rel_err(cderiv(d, y, 0, up), cderiv(o, y, 0, up)));
    for (int J = 1; J <= 6; ++J) e_dy = std::fmax(e_dy, rel_err({ydiff0(d, y, J)}, {ydiff0(o, y, J)}));
    // d^K l / dmu^K = sum_k d^k l / dm^k B_(K,k)(m', ...): the pure-m component is the first of each tensor
    double f[7] = {0.0};
    for (int K = 1; K <= 6; ++K) {
      f[K] = lderiv(o, y, K)[0];
      double chain = 0.0;
      for (int kk = 1; kk <= K; ++kk) chain += f[kk] * laplace::math::bell_partial(K, kk, mj);
      e_dk = std::fmax(e_dk, rel_err(lderiv(d, y, K), {chain}));
    }
  }
  for (double p : {1e-10, 0.01, 0.5, 0.9, 1.0 - 1e-10}) e_q = std::fmax(e_q, rel_err({d.quantile(p)}, {o.quantile(p)}));
  const double e_info = rel_err({d.expected_info()[0]}, {o.expected_info()[0] * mj[1] * mj[1]});
  std::printf("transformed Weibull oracle: value %.2e  log F %.2e  quantile %.2e  d1-d6 in mu %.2e  dy %.2e  "
              "info %.2e\n", e_val, e_cdf, e_q, e_dk, e_dy, e_info);
  CHECK(e_val <= 1e-15);
  CHECK(e_cdf <= 1.5e-14);
  CHECK(e_q <= 2e-15);
  CHECK(e_dk <= 1e-14);
  CHECK(e_dy <= 2e-14);
  CHECK(e_info <= 1e-15);
}

TEST_CASE("transformed: a decreasing affine map over gaussian2 is gaussian2") {
  const double mu = 0.3, v = 0.64, loc = 0.5, sc = -2.0;
  const tgau d = ld::transformed(ld::gaussian2_distrib{mu, v}, lb::scalar_bijector::affine(loc, sc));
  const ld::gaussian2_distrib o{(mu - loc) / sc, v / (sc * sc)};
  REQUIRE(d.decreasing);
  CHECK(d.support_lower() == -ld::distrib_infinity);
  double e_val = 0, e_cdf = 0, e_q = 0, e_dy = 0;
  for (double y : {-6.0, -1.0, -0.1, 0.2, 1.4, 5.0}) {
    e_val = std::fmax(e_val, rel_err({d.logpdf(y)}, {o.logpdf(y)}, 1.0));
    for (bool up : {false, true}) e_cdf = std::fmax(e_cdf, rel_err(cderiv(d, y, 0, up), cderiv(o, y, 0, up)));
    for (int J = 1; J <= 6; ++J) e_dy = std::fmax(e_dy, rel_err({ydiff0(d, y, J)}, {ydiff0(o, y, J)}, 1.0));
  }
  for (double p : {1e-200, 1e-10, 0.3, 0.5, 0.9, 1.0 - 1e-10})
    e_q = std::fmax(e_q, rel_err({d.quantile(p)}, {o.quantile(p)}));
  std::printf("transformed decreasing affine: value %.2e  log F %.2e  quantile %.2e  dy %.2e\n", e_val, e_cdf, e_q,
              e_dy);
  CHECK(e_val <= 4e-16);
  CHECK(e_cdf <= 4e-16);
  CHECK(e_q <= 2e-15);
  CHECK(e_dy <= 4e-16);
}

TEST_CASE("transformed: Yeo-Johnson over gaussian2") {
  const tgau d = ld::transformed(ld::gaussian2_distrib{0.3, 0.64}, inverted(lb::scalar_bijector::yeo_johnson(0.6)));
  const auto k = d.y_kinks();
  CHECK(k.count == 1);
  CHECK(k.point[0] == 0.0);
  CHECK(k.continuous_order == 2);
  const double mass = total_mass(d);
  // the reasoning of the header: the information over Y is the parent's over X
  const auto info_y = ld::expected_info_by_integral<0, ld::info_form::hessian>(d);
  const double e_info = rel_err(as_vec<2, 2>(info_y), as_vec<2, 2>(d.parent.expected_info()));
  double e_cdf = 0.0, e_q = 0.0, e_dy = 0.0;
  for (double y : {-2.0, -0.4, 0.5, 1.7}) {
    e_cdf = std::fmax(e_cdf, rel_err({d.cdf(y)}, {ld::cdf_by_integral(d, y)}));
    e_dy = std::fmax(e_dy, y_derivs_against_differences(d, y, 1e-3));
  }
  for (double p : {0.01, 0.3, 0.7, 0.99}) e_q = std::fmax(e_q, std::fabs(d.cdf(d.quantile(p)) - p) / p);
  std::printf("transformed Yeo-Johnson: mass - 1 %.2e  info by quadrature %.2e  F by quadrature %.2e  "
              "F(Q(p)) %.2e  dy against differences %.2e\n", mass - 1.0, e_info, e_cdf, e_q, e_dy);
  CHECK(std::fabs(mass - 1.0) <= 1e-15);
  CHECK(e_info <= 1e-15);
  CHECK(e_cdf <= 6e-15);
  CHECK(e_q <= 2e-15);
  CHECK(e_dy <= 1e-10);
}

TEST_CASE("transformed: Box-Cox over gamma2") {
  const tgam d = ld::transformed(ld::gamma2_distrib{2.0, 0.5}, lb::scalar_bijector::box_cox(0.5));
  CHECK(d.support_lower() == -2.0);
  CHECK(d.y_kinks().count == 0);
  const double mass = total_mass(d);
  double e_cdf = 0.0, e_dy = 0.0;
  for (double y : {-1.5, 0.0, 1.0, 3.0}) {
    e_cdf = std::fmax(e_cdf, rel_err({d.cdf(y)}, {ld::cdf_by_integral(d, y)}));
    e_dy = std::fmax(e_dy, y_derivs_against_differences(d, y, 1e-3));
  }
  std::printf("transformed Box-Cox: mass - 1 %.2e  F by quadrature %.2e  dy against differences %.2e\n", mass - 1.0,
              e_cdf, e_dy);
  CHECK(std::fabs(mass - 1.0) <= 1e-15);
  CHECK(e_cdf <= 1e-14);
  CHECK(e_dy <= 5e-9);
}

TEST_CASE("transformed: bounded gives a law on a finite interval") {
  const tgau d = ld::transformed(ld::gaussian2_distrib{0.3, 1.5}, inverted(lb::scalar_bijector::bounded(-1.0, 3.0)));
  CHECK(d.support_lower() == -1.0);
  CHECK(d.support_upper() == 3.0);
  const double mass = total_mass(d);
  double e_q = 0.0, e_dy = 0.0;
  for (double p : {1e-6, 0.2, 0.5, 0.95}) e_q = std::fmax(e_q, std::fabs(d.cdf(d.quantile(p)) - p) / p);
  for (double y : {-0.9, 0.4, 2.8}) e_dy = std::fmax(e_dy, y_derivs_against_differences(d, y, 1e-4));
  std::printf("transformed bounded: mass - 1 %.2e  F(Q(p)) %.2e  dy against differences %.2e\n", mass - 1.0, e_q,
              e_dy);
  CHECK(std::fabs(mass - 1.0) <= 1e-15);
  CHECK(e_q <= 2e-15);
  CHECK(e_dy <= 5e-10);
}

TEST_CASE("transformed: the atom of a zero-adjusted gamma comes out at g(0) with its mass") {
  const zag z{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}};
  const auto d = ld::transformed(z, lb::scalar_bijector::affine(-1.0, 0.5));  // g(v) = 2 v + 2
  using T = std::decay_t<decltype(d)>;
  CHECK(ld::distrib_n_atoms<T>() == 1);
  CHECK(ld::distrib_atom_point(d, 0) == 2.0);
  CHECK(ld::distrib_atom_log_mass(d, 0) == ld::distrib_atom_log_mass(z, 0));
  CHECK(d.logpdf(2.0) == z.logpdf(0.0));
  CHECK(d.y_kinks().count == 1);
  CHECK(d.y_kinks().point[0] == 2.0);
  const auto a = d.logpdf_d3(2.0), b = z.logpdf_d3(0.0);
  for (int i = 0; i < a.size; ++i) CHECK(a[i] == b[i]);
  const double mass = total_mass(d);
  std::printf("transformed zero-adjusted gamma: mass - 1 %.2e\n", mass - 1.0);
  CHECK(std::fabs(mass - 1.0) <= 1e-15);
  CHECK(d.cdf(2.0) == doctest::Approx(std::exp(z.logpdf(0.0))).epsilon(1e-15));
  // and the starting value is the parent's from the sample carried back, the zeros found exactly
  const std::vector<double> y{2.0, 2.0, 3.1, 4.5, 2.2, 6.0, 2.0};
  std::vector<double> x;
  for (double t : y) x.push_back(d.preimage(t));
  const auto s = d.start_value(y).parent, r = ld::distrib_start_value(z, x);
  CHECK(s.atom_alr[0] == r.atom_alr[0]);
  CHECK(s.parent.mu == r.parent.mu);
  CHECK(s.parent.sigma2 == r.parent.sigma2);
}

TEST_CASE("transformed: an atom inside a gaussian under a decreasing map") {
  const std::vector<double> th{0.4, 1.3, -1.0};
  const auto build = [](const double* t) {
    return ld::transformed(zan1{.parent = ld::gaussian1_distrib{t[0], t[1]}, .atom_alr = {t[2]}, .point = {0.5}},
                           lb::scalar_bijector::affine(0.0, -1.0));  // g(v) = -v
  };
  const auto d = build(th.data());
  const auto& z = d.parent;
  REQUIRE(d.decreasing);
  CHECK(ld::distrib_atom_point(d, 0) == -0.5);
  // F_Y(-0.5) = P(X >= 0.5) = P(X > 0.5) + p, 1 - F_Y(-0.5) = P(X < 0.5)
  const double p0 = std::exp(z.logpdf(0.5));
  CHECK(d.cdf(-0.5) == doctest::Approx(z.ccdf(0.5) + p0).epsilon(1e-14));
  CHECK(d.ccdf(-0.5) == doctest::Approx(z.cdf(0.5) - p0).epsilon(1e-13));
  CHECK(d.cdf(-0.5) + d.ccdf(-0.5) == doctest::Approx(1.0).epsilon(1e-15));
  CHECK(d.cdf(-1.2) == doctest::Approx(z.ccdf(1.2)).epsilon(1e-15));
  const double mass = total_mass(d);
  double e_t = 0.0;
  for (double y : {-0.5, -1.2, 0.3})
    for (bool up : {false, true}) e_t = std::fmax(e_t, tail_derivs_against_differences(build, th, y, up));
  std::printf("transformed atom under a decreasing map: mass - 1 %.2e  log F derivatives against differences %.2e\n",
              mass - 1.0, e_t);
  CHECK(std::fabs(mass - 1.0) <= 1e-15);
  CHECK(e_t <= 1e-9);
}

TEST_CASE("transformed: the refusals") {
  // the normal's support is not inside the domain of log
  CHECK_THROWS_AS(ld::transformed(ld::gaussian2_distrib{0.3, 0.64}, lb::scalar_bijector{lb::scalar_bijector_map::log}),
                  std::invalid_argument);
  // an atom at 0 has no finite image under log
  const zag z{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}};
  CHECK_THROWS_AS(ld::transformed(z, lb::scalar_bijector{lb::scalar_bijector_map::log}), std::invalid_argument);
  // two atoms and Yeo-Johnson's own kink make three non-smooth points
  const zan2 w{.parent = ld::gaussian1_distrib{0.0, 1.0}, .atom_alr = {-1.0, -1.0}, .point = {-1.0, 1.0}};
  CHECK_THROWS_AS(ld::transformed(w, inverted(lb::scalar_bijector::yeo_johnson(0.6))), std::invalid_argument);
  // the same two atoms under a smooth map are a law
  CHECK_NOTHROW(ld::transformed(w, inverted(lb::scalar_bijector::bounded(-1.0, 3.0))));
}

TEST_CASE("transformed: the starting value is the parent's on the parent's scale") {
  std::vector<double> x{-0.4, 0.1, 0.35, 0.9, 1.6, -1.1, 0.2}, y;
  for (double t : x) y.push_back(std::exp(t));
  const tgau d = ld::transformed(ld::gaussian2_distrib{0.0, 1.0}, exp_map());
  const auto s = d.start_value(y).parent;
  std::vector<double> xb;
  for (double t : y) xb.push_back(std::log(t));
  const auto r = ld::distrib_start_value(d.parent, xb);
  CHECK(s.mu == r.mu);
  CHECK(s.v == r.v);
}

// One value per index, the same function on the host and in the kernel; orders one and two only, the higher
// ones being checked on the host (test_truncated.cpp explains the cost of inlining order six into a kernel).
LAPLACE_HD double transformed_kernel_value(int i) {
  lb::scalar_bijector b{lb::scalar_bijector_map::log};
  b.inverted = true;
  const tgau d{.parent = ld::gaussian2_distrib{0.3, 0.64}, .map = b, .decreasing = false};
  const double y = 0.05 + 0.2 * i;
  return d.logpdf(y) + d.logpdf_d2(y)[1] + d.log_cdf_d1(y)[0] + d.log_ccdf(y) + d.quantile(0.01 + 0.03 * i) +
         d.logpdf_dy_dk<1, 1>(y)[1];
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for why.
void transformed_kernel_case() {
  const int n = 32;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for("distrib_transformed", n, KOKKOS_LAMBDA(const int i) { out(i) = transformed_kernel_value(i); });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    INFO("i = " << i);
    CHECK(h(i) == doctest::Approx(transformed_kernel_value(i)).epsilon(1e-11));
  }
}

TEST_CASE("transformed: a lognormal inside a kernel") { transformed_kernel_case(); }
