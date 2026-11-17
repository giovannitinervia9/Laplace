// truncated_distrib: a parent restricted to a fixed window and renormalized
// (note/piano_wrapper.md sections 4 and 5ter).
//
// No reference file: every quantity has a twin inside the test.
//   * the density integrates to one over the window, and the masses of a
//     discrete parent sum to one;
//   * the value against the parent's own less log Z computed here;
//   * every component of orders one to six against five-point differences of
//     the order below, in the middle of the window, near both bounds, in a
//     narrow window and in windows deep in either tail;
//   * the cumulative functions and their derivatives on both tails, and
//     F + (1 - F) = 1;
//   * the quantile against the cumulative function it inverts;
//   * truncated(zero_adjusted(gamma2)) with the atom kept integrates to one,
//     which is what proves that the atom is read from the object;
//   * the two compositions that are models: an atom on a truncated parent, and
//     a truncated atom mixture whose window keeps every atom;
//   * the refusals, the collapse of a nested truncation, and the generic
//     expected information over the window.
#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/distrib/poisson.hpp>
#include <laplace/distrib/truncated.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "check_common.hpp"
#include <doctest/doctest.h>

using namespace laplace_distrib_check;  // the shared checks, ld included

using tgau = ld::truncated_distrib<ld::gaussian1_distrib>;
using tpois = ld::truncated_distrib<ld::poisson_distrib>;
using zag = ld::atom_mixture_distrib<ld::gamma2_distrib, ld::adjusted>;
using tzag = ld::truncated_distrib<zag>;
using zan = ld::atom_mixture_distrib<ld::gaussian1_distrib, ld::adjusted, ld::adjusted>;
using tzan = ld::truncated_distrib<zan>;
using atg = ld::atom_mixture_distrib<tgau, ld::adjusted>;

namespace {

// a gaussian truncated to [a, b], built from theta
struct gau_window {
  double a, b;
  tgau operator()(const double* t) const {
    return tgau{.parent = ld::gaussian1_distrib{t[0], t[1]}, .lower = a, .upper = b};
  }
};

// log Phi(x), and log(Phi(b) - Phi(a)) from whichever side does not cancel, written apart from the header
double log_gauss_window(double mu, double sigma, double a, double b) {
  const double za = (a - mu) / sigma, zb = (b - mu) / sigma, r = std::sqrt(0.5);
  if (zb <= -za) return std::log(0.5 * (std::erfc(-zb * r) - std::erfc(-za * r)));
  return std::log(0.5 * (std::erfc(za * r) - std::erfc(zb * r)));
}

// every order of d^K l against the difference of d^(K-1) l in its last index
template <class B>
void check_value_derivs(const B& build, const std::vector<double>& th, double y, double rel) {
  using D = std::decay_t<decltype(build(th.data()))>;
  constexpr int P = D::n_params;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = lderiv(build(th.data()), y, K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    if (scale == 0.0) scale = 1.0;
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return lderiv(build(t), y, K - 1); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      const double e = exact[position_of(P, idx)], a = fd[position_of(P, lower)];
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y);
      CHECK(std::fabs(e - a) <= rel * scale);
    }
  }
}

// the same for log F and log(1 - F)
template <class B>
void check_tail_derivs(const B& build, const std::vector<double>& th, double y, bool upper, double rel) {
  using D = std::decay_t<decltype(build(th.data()))>;
  constexpr int P = D::n_params;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = cderiv(build(th.data()), y, K, upper);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    if (scale == 0.0) scale = 1.0;
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return cderiv(build(t), y, K - 1, upper); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      const double e = exact[position_of(P, idx)], a = fd[position_of(P, lower)];
      INFO("upper = " << upper << ", K = " << K << ", component " << position_of(P, idx) << ", y = " << y);
      CHECK(std::fabs(e - a) <= rel * scale);
    }
  }
}

// the integral of the density over [a, b], cut at the kinks, plus the masses of the atoms in it
template <class D>
double total_mass(const D& d, double a, double b, const std::vector<double>& atoms) {
  const auto f = [&](double t) { return std::exp(d.logpdf(t)); };
  double kinks[2];
  int nk = 0;
  for (double c : atoms) kinks[nk++] = c;
  const auto q = laplace::math::quad::panels_quad(f, a, b, 0.5 * (a + b), 0.25 * (b - a), kinks, nk);
  REQUIRE(q.converged);
  double s = q.value;
  for (double c : atoms) s += std::exp(d.logpdf(c));
  return s;
}

}  // namespace

TEST_CASE("truncated: the descriptors are the parent's") {
  CHECK(tgau::kind == ld::distrib_kind::continuous);
  CHECK(tgau::n_params == 2);
  CHECK(tgau::params == ld::gaussian1_distrib::params);
  CHECK(tgau::n_atoms == 0);
  CHECK(tgau::y_regularity == ld::gaussian1_distrib::y_regularity);
  CHECK(tpois::kind == ld::distrib_kind::discrete);
  CHECK(tzag::n_params == 3);
  CHECK(tzag::n_atoms == 1);
  CHECK(std::string(tzag::params[2].name) == "atom_alr1");
  for (int j = 0; j < tgau::n_params; ++j) CHECK(tgau::params[j].moves_support == false);
}

TEST_CASE("truncated: the density integrates to one and the value is the parent's less log Z") {
  const std::vector<double> th{0.4, 1.3};
  for (auto [a, b] : {std::pair{-0.5, 2.0}, std::pair{1.0, 1.01}, std::pair{10.0, 12.0}, std::pair{-12.0, -10.0},
                      std::pair{-ld::distrib_infinity, 0.0}, std::pair{1.5, ld::distrib_infinity}}) {
    const tgau d = gau_window{a, b}(th.data());
    const double lz = log_gauss_window(th[0], th[1], a, b);
    INFO("window [" << a << ", " << b << "]");
    CHECK(d.log_norm() == doctest::Approx(lz).epsilon(1e-12));
    const double lo = std::isfinite(a) ? a : b - 40.0, hi = std::isfinite(b) ? b : a + 40.0;
    for (double y : {lo, 0.5 * (lo + hi), hi})
      CHECK(d.logpdf(y) == doctest::Approx(d.parent.logpdf(y) - lz).epsilon(1e-12));
    if (std::isfinite(a)) CHECK(d.logpdf(a - 1e-9) == -ld::distrib_infinity);
    if (std::isfinite(b)) CHECK(d.logpdf(b + 1e-9) == -ld::distrib_infinity);
    const auto f = [&](double t) { return std::exp(d.logpdf(t)); };
    const auto q = laplace::math::quad::panels_quad(f, d.support_lower(), d.support_upper(), d.center(), d.scale());
    REQUIRE(q.converged);
    CHECK(q.value == doctest::Approx(1.0).epsilon(1e-10));
  }
}

TEST_CASE("truncated: a discrete parent sums to one and keeps both bounds") {
  const tpois d = ld::truncated(ld::poisson_distrib{3.5}, 2.0, 9.0);
  double s = 0.0, z = 0.0;
  for (int k = 2; k <= 9; ++k) z += std::exp(d.parent.logpmf(k));
  for (int k = 0; k <= 30; ++k) s += std::exp(d.logpmf(k));
  CHECK(s == doctest::Approx(1.0).epsilon(1e-14));
  CHECK(d.log_norm() == doctest::Approx(std::log(z)).epsilon(1e-14));
  CHECK(d.logpmf(1.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(10.0) == -ld::distrib_infinity);
  CHECK(d.logpmf(2.0) == doctest::Approx(d.parent.logpmf(2.0) - std::log(z)).epsilon(1e-14));
  // the zero-truncated Poisson in closed form
  const tpois zt = ld::truncated(ld::poisson_distrib{0.3}, 1.0, ld::distrib_infinity);
  CHECK(zt.logpmf(1.0) == doctest::Approx(std::log(0.3 * std::exp(-0.3) / (1.0 - std::exp(-0.3)))).epsilon(1e-14));
  // a bound that is not an integer is the integer inside it
  const tpois h = ld::truncated(ld::poisson_distrib{3.5}, 1.5, 9.7);
  CHECK(h.log_norm() == doctest::Approx(d.log_norm()).epsilon(1e-15));
  // and the cumulative functions, by summation
  for (double y : {2.0, 4.0, 6.5, 8.0}) {
    double c = 0.0;
    for (int k = 2; k <= static_cast<int>(y); ++k) c += std::exp(d.logpmf(k));
    INFO("y = " << y);
    CHECK(d.cdf(y) == doctest::Approx(c).epsilon(1e-13));
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-14));
  }
}

TEST_CASE("truncated: the derivatives against central differences") {
  const std::vector<double> th{0.4, 1.3};
  for (double y : {0.7, -0.49, 1.99}) check_value_derivs(gau_window{-0.5, 2.0}, th, y, 2e-5);
  // a narrow window, where Z is small and F(b) - F(a) cancels. Not at its middle, where the score in mu
  // is 2e-6 and the noise of the differences, eps / width over the step, is of 5e-11
  check_value_derivs(gau_window{1.0, 1.01}, th, 1.0095, 2e-5);
  // deep in each tail, where one side of the parent rounds to one
  check_value_derivs(gau_window{10.0, 12.0}, th, 10.3, 2e-5);
  check_value_derivs(gau_window{-12.0, -10.0}, th, -10.3, 2e-5);
  // a half line
  check_value_derivs(gau_window{1.5, ld::distrib_infinity}, th, 2.0, 2e-5);
  // a discrete parent, at both bounds and inside
  const auto pb = [](const double* t) { return tpois{.parent = ld::poisson_distrib{t[0]}, .lower = 2.0, .upper = 9.0}; };
  for (double y : {2.0, 5.0, 9.0}) check_value_derivs(pb, std::vector<double>{3.5}, y, 2e-5);
}

TEST_CASE("truncated: the cumulative functions and their derivatives on both tails") {
  const std::vector<double> th{0.4, 1.3};
  for (auto [a, b, y] : {std::tuple{-0.5, 2.0, 0.7}, std::tuple{1.0, 1.01, 1.004}, std::tuple{10.0, 12.0, 10.4},
                         std::tuple{-12.0, -10.0, -10.6}}) {
    const tgau d = gau_window{a, b}(th.data());
    const auto f = [&](double t) { return std::exp(d.logpdf(t)); };
    const auto q = laplace::math::quad::panels_quad(f, a, y, 0.5 * (a + y), 0.5 * (y - a));
    REQUIRE(q.converged);
    INFO("window [" << a << ", " << b << "], y = " << y);
    CHECK(d.cdf(y) == doctest::Approx(q.value).epsilon(1e-10));
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-13));
    CHECK(d.cdf(a - 1.0) == 0.0);
    CHECK(d.ccdf(b) == 0.0);
    for (bool upper : {false, true}) check_tail_derivs(gau_window{a, b}, th, y, upper, 3e-5);
  }
  const auto pb = [](const double* t) { return tpois{.parent = ld::poisson_distrib{t[0]}, .lower = 2.0, .upper = 9.0}; };
  for (double y : {2.0, 4.0, 8.0})
    for (bool upper : {false, true}) check_tail_derivs(pb, std::vector<double>{3.5}, y, upper, 3e-5);
}

TEST_CASE("truncated: the quantile inverts the cumulative function") {
  for (auto [a, b] : {std::pair{-0.5, 2.0}, std::pair{10.0, 12.0}, std::pair{-12.0, -10.0}}) {
    const tgau d = ld::truncated(ld::gaussian1_distrib{0.4, 1.3}, a, b);
    for (double p : {1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6}) {
      const double y = d.quantile(p);
      INFO("window [" << a << ", " << b << "], p = " << p << ", y = " << y);
      CHECK(y >= a);
      CHECK(y <= b);
      CHECK(d.cdf(y) == doctest::Approx(p).epsilon(1e-9));
    }
  }
  const tpois e = ld::truncated(ld::poisson_distrib{3.5}, 2.0, 9.0);
  for (double p : {0.01, 0.3, 0.7, 0.999}) {
    const double k = e.quantile(p);
    INFO("p = " << p << ", k = " << k);
    CHECK(e.cdf(k) >= p);
    CHECK(e.cdf(k - 1.0) < p);
  }
}

// the regression test of the defect note/piano_wrapper.md section 3.7 names: the left limit at a bound
// that holds an atom must take the atom's mass away, which only the object can say
TEST_CASE("truncated: a zero-adjusted gamma truncated above its atom integrates to one") {
  const std::vector<double> th{2.0, 0.5, -1.0};
  const auto build = [](const double* t) {
    return ld::truncated(zag{.parent = ld::gamma2_distrib{t[0], t[1]}, .atom_alr = {t[2]}, .point = {0.0}}, 0.0,
                         3.0);
  };
  const tzag d = build(th.data());
  CHECK(ld::distrib_n_atoms<tzag>() == 1);
  CHECK(ld::distrib_atom_point(d, 0) == 0.0);
  CHECK(ld::distrib_atom_log_mass(d, 0) == doctest::Approx(d.logpdf(0.0)).epsilon(1e-15));
  // Z = F(3) here, the atom being inside: had the bound been read as continuous it would be F(3) - pi_0
  CHECK(d.log_norm() == doctest::Approx(d.parent.log_cdf(3.0)).epsilon(1e-15));
  CHECK(total_mass(d, 0.0, 3.0, {0.0}) == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(d.cdf(0.0) == doctest::Approx(std::exp(d.logpdf(0.0))).epsilon(1e-14));
  for (double y : {0.0, 1.0, 2.9}) check_value_derivs(build, th, y, 2e-5);
  for (bool upper : {false, true}) check_tail_derivs(build, th, 1.5, upper, 3e-5);
  for (double p : {0.1, 0.5, 0.9}) {
    const double y = d.quantile(p);
    INFO("p = " << p << ", y = " << y);
    if (y == 0.0) CHECK(d.cdf(0.0) >= p);
    else CHECK(d.cdf(y) == doctest::Approx(p).epsilon(1e-9));
  }
}

TEST_CASE("truncated: an atom mixture keeping every atom, with one on the lower bound") {
  const std::vector<double> th{0.4, 1.3, -1.0, -2.0};
  const auto build = [](const double* t) {
    return ld::truncated(zan{.parent = ld::gaussian1_distrib{t[0], t[1]}, .atom_alr = {t[2], t[3]}, .point = {0.0, 1.0}},
                         0.0, 2.0);
  };
  const tzan d = build(th.data());
  // the parent has mass below the bound, so F(0-) = pi_N Phi(0) is neither zero nor F(0)
  CHECK(total_mass(d, 0.0, 2.0, {0.0, 1.0}) == doctest::Approx(1.0).epsilon(1e-10));
  for (double y : {0.0, 0.5, 1.0, 1.7}) check_value_derivs(build, th, y, 2e-5);
  for (bool upper : {false, true}) check_tail_derivs(build, th, 0.5, upper, 3e-5);
  CHECK(d.cdf(1.2) + d.ccdf(1.2) == doctest::Approx(1.0).epsilon(1e-13));
}

TEST_CASE("truncated: an atom on a truncated parent, the atom outside the window") {
  const std::vector<double> th{0.4, 1.3, -1.0};
  const auto build = [](const double* t) {
    return atg{.parent = tgau{.parent = ld::gaussian1_distrib{t[0], t[1]}, .lower = 0.5, .upper = 3.0},
               .atom_alr = {t[2]}, .point = {0.0}};
  };
  const atg d = build(th.data());
  CHECK(total_mass(d, 0.5, 3.0, {}) + std::exp(d.logpdf(0.0)) == doctest::Approx(1.0).epsilon(1e-10));
  for (double y : {0.0, 1.0, 2.9}) check_value_derivs(build, th, y, 2e-5);
  for (bool upper : {false, true}) check_tail_derivs(build, th, 1.0, upper, 3e-5);
  CHECK(d.cdf(1.0) + d.ccdf(1.0) == doctest::Approx(1.0).epsilon(1e-13));
}

TEST_CASE("truncated: the refusals and the collapse") {
  const zag z{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}};
  // the window kills the atom's cell
  CHECK_THROWS_AS(ld::truncated(z, 0.5, 3.0), std::invalid_argument);
  // and over a discrete parent, whose atoms are support points it does not declare as atoms
  using zip = ld::atom_mixture_distrib<ld::poisson_distrib, ld::inflated>;
  const zip p{.parent = ld::poisson_distrib{3.5}, .atom_alr = {-1.0}, .point = {0.0}};
  CHECK_THROWS_AS(ld::truncated(p, 1.0, 20.0), std::invalid_argument);
  CHECK_NOTHROW(ld::truncated(p, 0.0, 20.0));
  // a window with nothing in it, or a single point
  CHECK_THROWS_AS(ld::truncated(ld::gaussian1_distrib{0.0, 1.0}, 2.0, 1.0), std::invalid_argument);
  CHECK_THROWS_AS(ld::truncated(ld::poisson_distrib{3.5}, 2.2, 2.8), std::invalid_argument);
  CHECK_THROWS_AS(ld::truncated(ld::poisson_distrib{3.5}, 4.0, 4.0), std::invalid_argument);
  CHECK_THROWS_AS(ld::truncated(ld::gamma2_distrib{2.0, 0.5}, -3.0, -1.0), std::invalid_argument);
  // a truncation of a truncation is one truncation at the intersection
  const auto t = ld::truncated(ld::truncated(ld::gaussian1_distrib{0.4, 1.3}, -1.0, 2.0), 0.0, 5.0);
  static_assert(std::is_same_v<std::decay_t<decltype(t)>, tgau>);
  CHECK(t.lower == 0.0);
  CHECK(t.upper == 2.0);
  // the window is cut to the parent's support
  const auto g = ld::truncated(ld::gamma2_distrib{2.0, 0.5}, -5.0, 3.0);
  CHECK(g.lower == 0.0);
}

TEST_CASE("truncated: the generic expected information over the window") {
  const tgau d = ld::truncated(ld::gaussian1_distrib{0.4, 1.3}, -0.5, 2.0);
  constexpr int P = tgau::n_params;
  const auto info = ld::distrib_expected_info<0>(d);
  for (const auto& idx : sorted_indices(P, 2)) {
    const int pos = position_of(P, idx);
    const auto g = [&](double t) { return -d.logpdf_dk<2>(t)[pos] * std::exp(d.logpdf(t)); };
    const auto q = laplace::math::quad::panels_quad(g, -0.5, 2.0, 0.75, 0.5);
    REQUIRE(q.converged);
    INFO("component " << pos);
    CHECK(info[pos] == doctest::Approx(q.value).epsilon(1e-9));
  }
  // and its first derivative against differences of it
  const std::vector<double> th{0.4, 1.3};
  const auto ivec = [&](const double* t, int K) {
    const tgau e = gau_window{-0.5, 2.0}(t);
    if (K == 0) return as_vec<P, 2>(ld::distrib_expected_info<0>(e));
    return info_vec<P, 1>(ld::distrib_expected_info<1>(e));
  };
  const auto exact = ivec(th.data(), 1);
  double scale = 0.0;
  for (double v : exact) scale = std::fmax(scale, std::fabs(v));
  for (int pr = 0; pr < ld::sym_tensor_size(P, 2); ++pr)
    for (int j = 0; j < P; ++j) {
      const auto fd = diff5([&](const double* t) { return ivec(t, 0); }, th, j, 1e-3 * std::fabs(th[j]));
      INFO("pair " << pr << ", j = " << j);
      CHECK(std::fabs(exact[pr * P + j] - fd[pr]) <= 1e-6 * scale);
    }
}

TEST_CASE("truncated: the derivatives in y are the parent's") {
  const tgau d = ld::truncated(ld::gaussian1_distrib{0.4, 1.3}, -0.5, 2.0);
  const auto a = d.logpdf_dy_dk<1, 2>(0.7), b = d.parent.logpdf_dy_dk<1, 2>(0.7);
  for (int i = 0; i < a.size; ++i) CHECK(a[i] == b[i]);
  const auto c = d.logpdf_dy_dk<0, 3>(0.7), e = d.logpdf_dk<3>(0.7);
  for (int i = 0; i < c.size; ++i) CHECK(c[i] == e[i]);
}

// One value per index, the same function on the host and in the kernel. The orders three to six are checked
// on the host; the kernel uses orders one and two only, since the order-six routes of a wrapper inlined into one
// kernel kept ptxas running for over an hour on about 20 GB. One wrapper per kernel for the same reason.
template <int W>
LAPLACE_HD double truncated_kernel_value(int i) {
  const double y = 0.05 + 0.09 * i;
  if constexpr (W == 0) {
    const tgau d{.parent = ld::gaussian1_distrib{0.4, 1.3}, .lower = -0.5, .upper = 3.0};
    return d.logpdf(y) + d.logpdf_d2(y)[1] + d.log_cdf_d1(y)[0] + d.log_ccdf_d1(y)[1] + d.quantile(0.01 + 0.03 * i);
  } else if constexpr (W == 1) {
    const tzag z{.parent = zag{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}},
                 .lower = 0.0, .upper = 3.0};
    return z.logpdf(y) + z.logpdf_d2(y)[2] + z.log_cdf_d1(y)[1] + z.logpdf(0.0);
  } else {
    const tpois p{.parent = ld::poisson_distrib{3.5}, .lower = 2.0, .upper = 9.0};
    return p.logpmf(static_cast<double>(2 + i % 8)) + p.logpmf_d2(5.0)[0] + p.log_ccdf_d1(4.0)[0];
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for why.
template <int W>
void truncated_kernel_case() {
  const int n = 32;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_truncated", n, KOKKOS_LAMBDA(const int i) { out(i) = truncated_kernel_value<W>(i); });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    INFO("wrapper " << W << ", i = " << i);
    CHECK(h(i) == doctest::Approx(truncated_kernel_value<W>(i)).epsilon(1e-11));
  }
}

TEST_CASE("truncated: a truncated Gaussian inside a kernel") { truncated_kernel_case<0>(); }
TEST_CASE("truncated: a truncated zero-adjusted gamma inside a kernel") { truncated_kernel_case<1>(); }
TEST_CASE("truncated: a truncated Poisson inside a kernel") { truncated_kernel_case<2>(); }
