// The generic routes of laplace::distrib on a law that carries atoms: owning a
// closed cdf, quantile or starting value is an optimization, so the generic
// route has to give the same answers. Two objects are measured against two
// independent oracles.
//
//   * fixed_atom_distrib, a gamma with a point mass of FIXED size at zero: the
//     mass carries no parameter, so F(y) = pi + (1 - pi) F_parent(y) is a closed
//     form written here and not anywhere in the library. The generic cdf, its
//     derivatives and the generic quantile are checked against it.
//   * atom_hidden_distrib over a zero-adjusted gamma (tests/distrib/generic_hidden.hpp),
//     whose atom mass IS a parameter: the generic routes are checked against
//     atom_mixture_distrib's own closed cdf, log-cdf derivatives to order six and
//     quantile, which are themselves measured against mpmath and frozen
//     statmodels7 references in tests/distrib/test_atom_mixture.cpp. This is the
//     case that exercises the derivatives of the continuous mass
//     C = 1 - sum_i pi_i, which a constant mass leaves out.
//   * truncated_distrib over a zero-adjusted gamma, generic against its own
//     closed cdf, and one frozen figure computed by hand.
//
// Also checked: the quadrature never reads a mass as a density (the density of a
// law whose whole support is one atom integrates to nothing, and the total mass
// still comes out as one), the quantile lands exactly on the atom point when the
// probability falls inside its jump, and the generic starting value drops the
// observations sitting on an atom before the summaries.
#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/laplace1.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/distrib/truncated.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "check_common.hpp"
#include "generic_hidden.hpp"
#include <doctest/doctest.h>

using namespace laplace_distrib_check;  // the shared checks, ld included

namespace {

// A gamma with a point mass of fixed size at zero. The mass is not a parameter,
// so the law has the parent's two and no closed cumulative route of its own: the
// generic ones take over, and F(y) = pi + (1 - pi) F_parent(y) is the oracle.
struct fixed_atom_gamma {
  ld::gamma1_distrib parent;
  double log_pi;  // log of the mass at zero

  static constexpr ld::distrib_kind kind = ld::distrib_kind::continuous;
  static constexpr int n_params = 2;
  static constexpr int y_regularity = ld::infinitely_smooth;
  static constexpr const ld::distrib_param_info* params = ld::gamma1_distrib::params;
  static constexpr int n_atoms = 1;

  LAPLACE_HD double atom_point(int) const { return 0.0; }
  LAPLACE_HD double atom_log_mass(int) const { return log_pi; }

  // the log mass at the atom, the log density elsewhere: the rule of core.hpp
  LAPLACE_HD double logpdf(double y) const {
    if (y == 0.0) return log_pi;
    return Kokkos::log1p(-laplace::math::exp(log_pi)) + parent.logpdf(y);
  }
  // pi and 1 - pi are constants, so the derivatives in theta are the parent's
  // everywhere, the atom point included, where they are all zero
  template <int K>
  LAPLACE_HD ld::sym_tensor<2, K> logpdf_dk(double y) const {
    if (y == 0.0) return ld::sym_tensor<2, K>{};
    return parent.template logpdf_dk<K>(y);
  }
  LAPLACE_HD auto logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  LAPLACE_HD auto logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  LAPLACE_HD auto logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  LAPLACE_HD auto logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  LAPLACE_HD auto logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  LAPLACE_HD auto logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  LAPLACE_HD ld::distrib_kinks y_kinks() const {
    ld::distrib_kinks k;
    k.count = 1;
    k.point[0] = 0.0;
    k.continuous_order = 0;
    return k;
  }
  LAPLACE_HD ld::distrib_kinks param_kinks(int, double) const { return ld::distrib_kinks{}; }

  LAPLACE_HD double support_lower() const { return 0.0; }
  LAPLACE_HD double support_upper() const { return ld::distrib_infinity; }
  LAPLACE_HD double center() const { return parent.center(); }
  LAPLACE_HD double scale() const { return parent.scale(); }

  // the mass is fixed, so the starting value is the parent's over whatever
  // sample the generic route hands down
  fixed_atom_gamma start_value(const ld::distrib_sample_summary& s) const {
    return {.parent = parent.start_value(s), .log_pi = log_pi};
  }

  // the oracle: F(y) = pi + (1 - pi) F_parent(y), and its complement
  double exact_cdf(double y) const {
    if (y < 0.0) return 0.0;
    const double pi = std::exp(log_pi);
    return pi + (1.0 - pi) * ld::distrib_cdf(parent, y);
  }
  double exact_ccdf(double y) const { return (1.0 - std::exp(log_pi)) * ld::distrib_ccdf(parent, y); }
  // its inverse, the quantile of a law with one jump at zero
  double exact_quantile(double p) const {
    const double pi = std::exp(log_pi);
    if (p <= pi) return 0.0;
    return ld::distrib_quantile(parent, (p - pi) / (1.0 - pi));
  }
};

template <class T>
constexpr bool has_closed_cdf = requires(const T& t) { t.log_cdf(1.0); };
template <class T>
constexpr bool has_closed_quantile = requires(const T& t) { t.quantile(0.5); };

using zag = ld::atom_mixture_distrib<ld::gamma1_distrib, ld::adjusted>;
using tzag = ld::truncated_distrib<zag>;
// more than two atoms over a continuous parent, which the limit forbade until
// 2026-11-17 and nothing had ever exercised
using zag3 = ld::atom_mixture_distrib<ld::gamma1_distrib, ld::adjusted, ld::adjusted, ld::adjusted>;
using zag6 = ld::atom_mixture_distrib<ld::gamma1_distrib, ld::adjusted, ld::adjusted, ld::adjusted,
                                      ld::adjusted, ld::adjusted, ld::adjusted>;
// a parent with a real kink in y, at mu, which the mixture has to pass on
using zal = ld::atom_mixture_distrib<ld::laplace1_distrib, ld::adjusted>;

// a zero-adjusted gamma with shape 2 and rate 1 (mean 2, dispersion 1/2) and
// mass pi at zero
zag make_zag(double pi) {
  return zag{.parent = ld::gamma1_distrib{2.0, 0.5}, .atom_alr = {std::log(pi / (1.0 - pi))}, .point = {0.0}};
}

// the error of a against b in units of the ulp of b, and the absolute one
void worst_against(double a, double b, double& ulps, double& abs_err) {
  const double e = std::fabs(a - b);
  if (e > abs_err) abs_err = e;
  // the ulp measure says nothing where the reference is near zero (log F in the
  // far upper tail is zero to the last bit), the ulp of such a value being a
  // denormal; there the absolute error is the figure that means something
  if (std::fabs(b) < 1e-8) return;
  const double u = std::fabs(std::nextafter(b, ld::distrib_infinity) - b);
  const double r = u > 0.0 ? e / u : 0.0;
  if (r > ulps) ulps = r;
}

// the points the comparisons run over: the atom itself, its neighbours on both
// sides, the bulk of the window, and both tails
std::vector<double> test_points() {
  std::vector<double> y{0.0, std::nextafter(0.0, 1.0), 1e-300, 1e-12, 1e-6};
  for (double v = 0.05; v < 12.0; v += 0.25) y.push_back(v);
  for (double v : {20.0, 40.0, 80.0, 160.0, 320.0}) y.push_back(v);
  return y;
}

}  // namespace

TEST_CASE("generic atoms: the cdf of a fixed point mass against its closed form") {
  const fixed_atom_gamma d{.parent = ld::gamma1_distrib{2.0, 0.5}, .log_pi = std::log(0.3)};
  double u_cdf = 0.0, a_cdf = 0.0, u_ccdf = 0.0, a_ccdf = 0.0, u_log = 0.0, a_log = 0.0;
  (void)u_log;
  for (double y : test_points()) {
    const double got = ld::distrib_cdf(d, y), want = d.exact_cdf(y);
    const double gotc = ld::distrib_ccdf(d, y), wantc = d.exact_ccdf(y);
    INFO("y = " << y);
    worst_against(got, want, u_cdf, a_cdf);
    worst_against(gotc, wantc, u_ccdf, a_ccdf);
    worst_against(ld::distrib_log_cdf(d, y), std::log(want), u_log, a_log);
    CHECK(std::fabs(got - want) <= 1e-12 * want);
    CHECK(std::fabs(gotc - wantc) <= 1e-12 * wantc);
    CHECK(got + gotc == doctest::Approx(1.0).epsilon(1e-13));
  }
  // the atom alone: F(0) is its mass and not zero, which is what an atom-blind
  // route returns there
  CHECK(ld::distrib_cdf(d, 0.0) == doctest::Approx(0.3).epsilon(1e-13));
  CHECK(ld::distrib_ccdf(d, 0.0) == doctest::Approx(0.7).epsilon(1e-13));
  // just above the atom the cdf has left the jump but has barely begun to rise
  CHECK(ld::distrib_cdf(d, std::nextafter(0.0, 1.0)) == doctest::Approx(0.3).epsilon(1e-13));
  // log F is reported in absolute terms only: it passes through zero in the far
  // upper tail, where an ulp of it says nothing
  MESSAGE("fixed mass: cdf " << u_cdf << " ulp (" << a_cdf << " abs), ccdf " << u_ccdf << " ulp (" << a_ccdf
                             << " abs), log cdf " << a_log << " abs");
  CHECK(u_cdf <= 64.0);     // measured 2 (gcc)
  CHECK(u_ccdf <= 4096.0);  // measured 333 (gcc)
  CHECK(a_log <= 1e-13);    // measured 3.9e-16 (gcc)
}

TEST_CASE("generic atoms: the quantile of a fixed point mass inverts its closed form") {
  const fixed_atom_gamma d{.parent = ld::gamma1_distrib{2.0, 0.5}, .log_pi = std::log(0.3)};
  // inside the jump at zero the quantile is the atom point exactly
  for (double p : {1e-300, 1e-30, 1e-3, 0.1, 0.29, 0.3}) {
    INFO("p = " << p);
    CHECK(ld::distrib_quantile(d, p) == 0.0);
  }
  // above it the root of the continuous part, which the oracle inverts in closed form
  double worst = 0.0;
  for (double p : {0.3000001, 0.31, 0.4, 0.5, 0.7, 0.9, 0.99, 0.999999}) {
    const double got = ld::distrib_quantile(d, p), want = d.exact_quantile(p);
    INFO("p = " << p << ", got " << got << ", want " << want);
    CHECK(got == doctest::Approx(want).epsilon(1e-9));
    worst = std::fmax(worst, std::fabs(got - want) / std::fabs(want));
  }
  MESSAGE("fixed mass: quantile worst relative error " << worst);
  CHECK(worst <= 1e-8);  // measured 1.3e-11 (gcc)
  // and the round trip through the cdf
  for (double p : {0.35, 0.6, 0.95}) CHECK(ld::distrib_cdf(d, ld::distrib_quantile(d, p)) == doctest::Approx(p).epsilon(1e-9));
}

TEST_CASE("generic atoms: the derivatives of the log cdf of a fixed point mass against differences") {
  // the mass is a constant, so the only parameters are the parent's and the
  // shared check applies as it does to any two-parameter family
  const std::vector<double> th{2.0, 0.5};
  const auto build = [](const double* t) {
    return fixed_atom_gamma{.parent = ld::gamma1_distrib{t[0], t[1]}, .log_pi = std::log(0.3)};
  };
  for (double y : {0.3, 1.0, 2.0, 5.0}) {
    for (int K = 1; K <= 4; ++K) {
      const auto exact = [&](const double* t, int k) {
        const auto d = build(t);
        switch (k) {
          case 0: return std::vector<double>{ld::distrib_log_cdf(d, y)};
          case 1: return as_vec<2, 1>(ld::distrib_log_cdf_dk<1>(d, y));
          case 2: return as_vec<2, 2>(ld::distrib_log_cdf_dk<2>(d, y));
          case 3: return as_vec<2, 3>(ld::distrib_log_cdf_dk<3>(d, y));
          default: return as_vec<2, 4>(ld::distrib_log_cdf_dk<4>(d, y));
        }
      };
      const auto e = exact(th.data(), K);
      double scale = 0.0;
      for (double v : e) scale = std::fmax(scale, std::fabs(v));
      if (scale == 0.0) scale = 1.0;
      for (const auto& idx : sorted_indices(2, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return exact(t, K - 1); }, th, j, 1e-3 * th[j]);
        INFO("y = " << y << ", K = " << K << ", component " << position_of(2, idx));
        CHECK(std::fabs(e[position_of(2, idx)] - fd[position_of(2, lower)]) <= 2e-5 * scale);
      }
    }
  }
}

TEST_CASE("generic atoms: the generic cdf of an atom mixture equals its closed one") {
  const zag z = make_zag(0.3);
  const auto h = laplace_test::hide_atom_routes(z);
  static_assert(ld::distrib_n_atoms<decltype(h)>() == 1, "the wrapper keeps the atom");
  static_assert(has_closed_cdf<zag> && has_closed_quantile<zag>, "the mixture owns its closed routes");
  static_assert(!has_closed_cdf<decltype(h)> && !has_closed_quantile<decltype(h)>,
                "the wrapper hides them, so the generic routes are what is measured");
  static_assert(!has_closed_cdf<fixed_atom_gamma> && !has_closed_quantile<fixed_atom_gamma>,
                "the fixed-mass law owns no cumulative route at all");
  double u_f = 0.0, a_f = 0.0, u_g = 0.0, a_g = 0.0, u_lf = 0.0, a_lf = 0.0, u_lg = 0.0, a_lg = 0.0;
  (void)u_lf, (void)u_lg;
  for (double y : test_points()) {
    INFO("y = " << y);
    worst_against(ld::distrib_cdf(h, y), z.cdf(y), u_f, a_f);
    worst_against(ld::distrib_ccdf(h, y), z.ccdf(y), u_g, a_g);
    worst_against(ld::distrib_log_cdf(h, y), z.log_cdf(y), u_lf, a_lf);
    worst_against(ld::distrib_log_ccdf(h, y), z.log_ccdf(y), u_lg, a_lg);
    CHECK(ld::distrib_cdf(h, y) == doctest::Approx(z.cdf(y)).epsilon(1e-11));
    CHECK(ld::distrib_ccdf(h, y) == doctest::Approx(z.ccdf(y)).epsilon(1e-11));
  }
  MESSAGE("atom mixture: generic against closed, cdf " << u_f << " ulp (" << a_f << " abs), ccdf " << u_g << " ulp ("
                                                       << a_g << " abs), log cdf " << a_lf << " abs, log ccdf " << a_lg
                                                       << " abs");
  CHECK(u_f <= 64.0);     // measured 3 (gcc)
  CHECK(u_g <= 4096.0);   // measured 284 (gcc)
  CHECK(a_lf <= 1e-13);   // measured 4.4e-16 (gcc)
  CHECK(a_lg <= 1e-11);   // measured 5.7e-14 (gcc)
  // the atom point itself, where an atom-blind quadrature returns zero
  CHECK(ld::distrib_cdf(h, 0.0) == doctest::Approx(0.3).epsilon(1e-12));
  CHECK(ld::distrib_log_cdf(h, 0.0) == doctest::Approx(std::log(0.3)).epsilon(1e-12));
  // and a mass far below the double range, where only the log scale carries it
  const zag tiny = make_zag(1e-300);
  const auto ht = laplace_test::hide_atom_routes(tiny);
  CHECK(ld::distrib_log_cdf(ht, 0.0) == doctest::Approx(std::log(1e-300)).epsilon(1e-10));
}

TEST_CASE("generic atoms: the generic log-cdf derivatives of an atom mixture equal its closed ones") {
  const zag z = make_zag(0.3);
  const auto h = laplace_test::hide_atom_routes(z);
  double worst[7] = {};
  for (double y : {0.05, 0.5, 1.0, 2.0, 4.0, 9.0}) {
    const std::vector<double> c1 = as_vec<3, 1>(ld::distrib_log_cdf_dk<1>(h, y));
    const std::vector<double> w1 = as_vec<3, 1>(z.log_cdf_d1(y));
    const std::vector<double> c2 = as_vec<3, 2>(ld::distrib_log_cdf_dk<2>(h, y));
    const std::vector<double> w2 = as_vec<3, 2>(z.log_cdf_d2(y));
    const std::vector<double> c3 = as_vec<3, 3>(ld::distrib_log_cdf_dk<3>(h, y));
    const std::vector<double> w3 = as_vec<3, 3>(z.log_cdf_d3(y));
    const std::vector<double> g1 = as_vec<3, 1>(ld::distrib_log_ccdf_dk<1>(h, y));
    const std::vector<double> v1 = as_vec<3, 1>(z.log_ccdf_d1(y));
    const std::vector<double> g2 = as_vec<3, 2>(ld::distrib_log_ccdf_dk<2>(h, y));
    const std::vector<double> v2 = as_vec<3, 2>(z.log_ccdf_d2(y));
    const auto cmp = [&](const std::vector<double>& got, const std::vector<double>& want, int K, double rel) {
      double scale = 0.0;
      for (double v : want) scale = std::fmax(scale, std::fabs(v));
      if (scale == 0.0) scale = 1.0;
      for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("y = " << y << ", K = " << K << ", component " << i << ", got " << got[i] << ", want " << want[i]);
        CHECK(std::fabs(got[i] - want[i]) <= rel * scale);
        worst[K] = std::fmax(worst[K], std::fabs(got[i] - want[i]) / scale);
      }
    };
    // measured against the closed forms: 1.2e-15, 1.4e-15, 2.7e-15, 8.9e-16,
    // 5.7e-16 relative to the largest component of the tensor (gcc)
    cmp(c1, w1, 1, 1e-11);
    cmp(c2, w2, 2, 1e-11);
    cmp(c3, w3, 3, 1e-10);
    cmp(g1, v1, 4, 1e-11);
    cmp(g2, v2, 5, 1e-11);
  }
  MESSAGE("atom mixture: log cdf d1 " << worst[1] << ", d2 " << worst[2] << ", d3 " << worst[3] << " relative; log ccdf d1 "
                                      << worst[4] << ", d2 " << worst[5]);
}

TEST_CASE("generic atoms: the generic quantile of an atom mixture equals its closed one") {
  const zag z = make_zag(0.3);
  const auto h = laplace_test::hide_atom_routes(z);
  // inside the jump both return the atom point exactly
  for (double p : {1e-250, 1e-8, 0.1, 0.2999, 0.3}) {
    INFO("p = " << p);
    CHECK(ld::distrib_quantile(h, p) == 0.0);
    CHECK(z.quantile(p) == 0.0);
  }
  double worst = 0.0;
  for (double p : {0.3001, 0.35, 0.5, 0.75, 0.9, 0.99, 0.9999}) {
    const double got = ld::distrib_quantile(h, p), want = z.quantile(p);
    INFO("p = " << p << ", got " << got << ", want " << want);
    CHECK(got == doctest::Approx(want).epsilon(1e-8));
    worst = std::fmax(worst, std::fabs(got - want) / std::fabs(want));
  }
  MESSAGE("atom mixture: quantile worst relative difference " << worst);
  CHECK(worst <= 1e-10);  // measured 1.1e-13 (gcc)
}

TEST_CASE("generic atoms: a truncated zero-adjusted gamma, generic against closed and against a frozen figure") {
  // gamma with shape 2 and rate 1 (gamma1 mean 2, dispersion 1/2) carrying mass
  // 0.3 at zero, truncated to [0, 3]. F_gamma(3) = 1 - e^-3 (1 + 3), so the
  // window holds Z = 0.3 + 0.7 F_gamma(3) and the cdf at zero is 0.3 / Z.
  const tzag t{.parent = make_zag(0.3), .lower = 0.0, .upper = 3.0};
  const double f3 = 1.0 - std::exp(-3.0) * 4.0;
  const double Z = 0.3 + 0.7 * f3;
  // mpmath at 40 digits: F_gamma(3) = 0.80085172652854422808,
  // Z = 0.86059620856998095966, F(0) = 0.34859553994375394646
  CHECK(f3 == doctest::Approx(0.80085172652854423).epsilon(1e-15));
  CHECK(Z == doctest::Approx(0.86059620856998096).epsilon(1e-15));
  // the closed route of the wrapper, and the generic one on the same object with
  // that route hidden: an atom-blind route returns zero at the lower bound
  const auto h = laplace_test::hide_atom_routes(t);
  CHECK(t.cdf(0.0) == doctest::Approx(0.3 / Z).epsilon(1e-12));
  CHECK(ld::distrib_cdf(h, 0.0) == doctest::Approx(0.3 / Z).epsilon(1e-10));
  CHECK(ld::distrib_cdf(h, 0.0) == doctest::Approx(0.34859553994375395).epsilon(1e-9));
  double u_f = 0.0, a_f = 0.0, u_g = 0.0, a_g = 0.0;
  for (double y : {0.0, 1e-12, 0.01, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 2.999, 3.0}) {
    INFO("y = " << y);
    worst_against(ld::distrib_cdf(h, y), t.cdf(y), u_f, a_f);
    worst_against(ld::distrib_ccdf(h, y), t.ccdf(y), u_g, a_g);
    CHECK(ld::distrib_cdf(h, y) == doctest::Approx(t.cdf(y)).epsilon(1e-10));
    CHECK(ld::distrib_ccdf(h, y) == doctest::Approx(t.ccdf(y)).epsilon(1e-10));
  }
  MESSAGE("truncated: generic against closed, cdf " << u_f << " ulp (" << a_f << " abs), ccdf " << u_g << " ulp (" << a_g
                                                    << " abs)");
  CHECK(u_f <= 64.0);   // measured 1 (gcc)
  CHECK(u_g <= 64.0);   // measured 7 (gcc)
  // the whole mass, atoms included, is one at the upper bound
  CHECK(ld::distrib_cdf(h, 3.0) == doctest::Approx(1.0).epsilon(1e-10));
  CHECK(ld::distrib_ccdf(h, 0.0) == doctest::Approx(1.0 - 0.3 / Z).epsilon(1e-10));
}

TEST_CASE("generic atoms: the mixture passes on its parent's kink in y") {
  // laplace1 declares a kink of log f at mu, and the continuous part of the
  // mixture is its density times one constant, so the mixture has that kink and
  // must report it. Reporting the atom positions there instead, as it did until
  // the integrands learned to skip an atom themselves, dropped it: the quadrature
  // of the density then ran a panel straight across the kink at mu.
  const zal z{.parent = ld::laplace1_distrib{1.0, 0.7}, .atom_alr = {std::log(0.25 / 0.75)}, .point = {0.0}};
  // This is the check that fails on the old code. The accuracy below does not:
  // panels_quad always cuts at center(), and laplace1 and enet, the only
  // continuous families that declare a kink today, put it at mu and center() at
  // mu too, so the dropped kink was cut anyway. Measured with the centre moved
  // off the kink, which center() being a rough location makes legitimate, the
  // old behaviour returns NaN at two of the ten points below, the quadrature
  // not converging across an uncut kink, where the repair gives 1.2e-16.
  const auto k = z.y_kinks();
  CHECK(k.count == 1);
  CHECK(k.point[0] == 1.0);  // mu, not the atom at 0
  const auto h = laplace_test::hide_atom_routes(z);
  // the generic quadrature against the mixture's own closed cdf, on both sides
  // of the kink and of the atom
  double worst = 0.0;
  for (double y : {-3.0, -0.5, 0.0, 0.25, 0.9, 1.0, 1.1, 2.0, 4.0, 8.0}) {
    const double got = ld::distrib_cdf(h, y), want = z.cdf(y);
    INFO("y = " << y << ", got " << got << ", want " << want);
    CHECK(got == doctest::Approx(want).epsilon(1e-11));
    worst = std::fmax(worst, std::fabs(got - want) / std::fmax(want, 1e-300));
  }
  MESSAGE("laplace1 parent: generic cdf against closed, worst relative " << worst);
  CHECK(worst <= 1e-12);  // measured 2.9e-16
}

TEST_CASE("generic atoms: more than two atoms over a continuous parent") {
  // Three atoms, five parameters. The masses are equal at 0.1 each, so the
  // parent's cell is 0.7 and every alr coordinate is log(0.1 / 0.7).
  const double a = std::log(0.1 / 0.7);
  const zag3 z3{.parent = ld::gamma1_distrib{2.0, 0.5}, .atom_alr = {a, a, a}, .point = {0.0, 1.0, 2.5}};
  static_assert(zag3::n_params == 5 && zag3::n_atoms == 3, "the parent's two plus one per atom");
  static_assert(zag3::closed_routes, "a continuous parent with adjusted atoms keeps its closed routes");
  for (int i = 0; i < 3; ++i) {
    INFO("atom " << i);
    CHECK(std::exp(z3.atom_log_mass(i)) == doctest::Approx(0.1).epsilon(1e-13));
  }
  // the three masses plus the quadrature of the density make one
  const double integral = ld::expectation_by_integral(z3, [](double) { return 1.0; }).value;
  CHECK(integral == doctest::Approx(1.0).epsilon(1e-10));
  // the cuts a quadrature of its density is given: gamma1 declares no kink in y,
  // so the three atom points and nothing else
  double cut[8];
  CHECK(ld::detail_generic::density_cuts(z3, cut) == 3);
  // the generic cdf against the wrapper's own closed one, at and around every atom
  const auto h3 = laplace_test::hide_atom_routes(z3);
  double worst = 0.0;
  for (double y : {-1.0, 0.0, 1e-9, 0.5, 1.0, 1.0000001, 1.7, 2.5, 2.6, 5.0, 20.0}) {
    const double got = ld::distrib_cdf(h3, y), want = z3.cdf(y);
    INFO("y = " << y << ", got " << got << ", want " << want);
    CHECK(got == doctest::Approx(want).epsilon(1e-10));
    if (want > 0.0) worst = std::fmax(worst, std::fabs(got - want) / want);
  }
  // and the first derivatives of its log cdf, all five components
  for (double y : {0.5, 1.5, 3.0}) {
    const auto g = as_vec<5, 1>(ld::distrib_log_cdf_dk<1>(h3, y));
    const auto w = as_vec<5, 1>(z3.log_cdf_d1(y));
    double scale = 0.0;
    for (double v : w) scale = std::fmax(scale, std::fabs(v));
    for (std::size_t i = 0; i < w.size(); ++i) {
      INFO("y = " << y << ", component " << i << ", got " << g[i] << ", want " << w[i]);
      CHECK(std::fabs(g[i] - w[i]) <= 1e-10 * scale);
    }
  }
  MESSAGE("three atoms: generic cdf against closed, worst relative " << worst);
  CHECK(worst <= 1e-12);  // measured 4.0e-16 (gcc)

  // Six atoms, eight parameters: the most a continuous parent now takes, and the
  // case the cut budget was measured on (six atom points plus at most two kinks
  // is exactly the eight cuts panels_quad keeps). Checked on the host and at low
  // order only: the generic order-6 route would hold six copies of the
  // eight-parameter partials, about 144 kB, far past the 8192-byte device stack
  // the tests set, and a continuous parent reaches its closed route anyway.
  const double b = std::log(0.05 / 0.7);
  const zag6 z6{.parent = ld::gamma1_distrib{2.0, 0.5},
                .atom_alr = {b, b, b, b, b, b},
                .point = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0}};
  static_assert(zag6::n_params == 8 && zag6::n_atoms == 6, "six atoms are allowed over a continuous parent");
  CHECK(ld::detail_generic::density_cuts(z6, cut) == 6);
  CHECK(ld::expectation_by_integral(z6, [](double) { return 1.0; }).value == doctest::Approx(1.0).epsilon(1e-10));
  const auto h6 = laplace_test::hide_atom_routes(z6);
  for (double y : {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 10.0}) {
    INFO("six atoms, y = " << y);
    CHECK(ld::distrib_cdf(h6, y) == doctest::Approx(z6.cdf(y)).epsilon(1e-10));
  }
  // the descriptors of the six free coordinates carry the six names of alr_names
  CHECK(std::string(zag6::params[2].name) == "atom_alr1");
  CHECK(std::string(zag6::params[7].name) == "atom_alr6");
}

TEST_CASE("generic atoms: the quadrature never reads a mass as a density") {
  // The expectation of 1 over the law is one, which it can only be if the
  // quadrature of the density leaves the atom to the explicit sum: counting the
  // mass twice, or reading it as a density at a node, breaks the total.
  const fixed_atom_gamma d{.parent = ld::gamma1_distrib{2.0, 0.5}, .log_pi = std::log(0.3)};
  const auto one = ld::expectation_by_integral(d, [](double) { return 1.0; });
  CHECK(one.value == doctest::Approx(1.0).epsilon(1e-11));
  // a mass that dominates the law entirely
  const fixed_atom_gamma heavy{.parent = ld::gamma1_distrib{2.0, 0.5}, .log_pi = std::log(1.0 - 1e-14)};
  CHECK(ld::expectation_by_integral(heavy, [](double) { return 1.0; }).value == doctest::Approx(1.0).epsilon(1e-10));
  // and the atom at the lower end of the support is where a panel begins, which
  // is exactly where a tanh-sinh node can round onto the end
  CHECK(ld::distrib_cdf(d, 1e-8) == doctest::Approx(0.3 + 0.7 * ld::distrib_cdf(d.parent, 1e-8)).epsilon(1e-10));
}

TEST_CASE("generic atoms: the generic starting value drops the observations on an atom") {
  std::vector<double> y;
  for (int i = 0; i < 30; ++i) y.push_back(0.0);  // the atom
  for (int i = 1; i <= 70; ++i) y.push_back(0.05 * static_cast<double>(i));
  const std::vector<double> rest(y.begin() + 30, y.end());
  const fixed_atom_gamma proto{.parent = ld::gamma1_distrib{1.0, 1.0}, .log_pi = std::log(0.3)};
  const auto s = ld::distrib_start_value(proto, y);
  const auto want = ld::distrib_start_value(ld::gamma1_distrib{1.0, 1.0}, rest);
  CHECK(s.parent.mu == doctest::Approx(want.mu).epsilon(1e-14));
  CHECK(s.parent.phi == doctest::Approx(want.phi).epsilon(1e-14));
  // the zeros would have pulled the mean down by 30 per cent
  CHECK(s.parent.mu != doctest::Approx(ld::distrib_start_value(ld::gamma1_distrib{1.0, 1.0}, y).mu).epsilon(1e-6));
  // a sample lying entirely on the atom is summarized as it is, not as nothing
  const std::vector<double> all_zero(10, 0.0);
  CHECK(ld::detail_start::sample_off_atoms(proto, all_zero).size() == 10u);
  CHECK(ld::detail_start::sample_off_atoms(proto, y).size() == 70u);
}

namespace {

// One value per index, the same function on the host and in the kernel. Orders
// one and two only: the order-six generic routes of a wrapper inlined into one
// kernel keep ptxas running for a very long time (the note on
// truncated_kernel_value in test_truncated.cpp).
LAPLACE_HD double generic_atoms_kernel_value(int i) {
  const double y = 0.1 + 0.3 * i;
  const fixed_atom_gamma d{.parent = ld::gamma1_distrib{2.0, 0.5}, .log_pi = -1.2039728043259361};
  return ld::distrib_log_cdf(d, y) + ld::distrib_log_ccdf(d, y) + ld::distrib_log_cdf_dk<1>(d, y)[0] +
         ld::distrib_log_cdf(d, 0.0) + ld::distrib_quantile(d, 0.02 + 0.03 * i);
}

}  // namespace

// The kernel lives in a named function with external linkage, not in the body of
// the TEST_CASE: see the comment on gaussian3_kernel_case in test_gaussian3.cpp.
void generic_atoms_kernel_case() {
  const int n = 24;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_generic_atoms", n, KOKKOS_LAMBDA(const int i) { out(i) = generic_atoms_kernel_value(i); });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    INFO("i = " << i);
    CHECK(h(i) == doctest::Approx(generic_atoms_kernel_value(i)).epsilon(1e-10));
  }
}

TEST_CASE("generic atoms: the generic routes of a law with an atom inside a kernel") { generic_atoms_kernel_case(); }
