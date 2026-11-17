// atom_mixture_distrib over a continuous parent, every atom adjusted: the
// block that note/piano_wrapper.md section 5bis calls the easy one.
//
// What is checked here, all on the host, no reference file being needed because
// every quantity has a twin inside the test:
//   * the descriptors and the atoms the wrapper declares;
//   * the masses: the atoms' plus the quadrature of the density between them
//     is one;
//   * the value against log pi + the parent's own log-density;
//   * every component of the orders one to six against five-point central
//     differences of the order below;
//   * the block structure the design predicts: a component with indices in
//     both blocks is exactly zero, and at an atom so is every component with a
//     parent index.
#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/gamma2.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/poisson.hpp>
#include <laplace/math/quadrature.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "check_common.hpp"
#include <doctest/doctest.h>

using namespace laplace_distrib_check;  // the shared checks, ld included

using zag = ld::atom_mixture_distrib<ld::gamma2_distrib, ld::adjusted>;
using zan = ld::atom_mixture_distrib<ld::gaussian1_distrib, ld::adjusted, ld::adjusted>;

namespace {

zag make_zag(const double* t) {
  return zag{.parent = ld::gamma2_distrib{t[0], t[1]}, .atom_alr = {t[2]}, .point = {0.0}};
}

zan make_zan(const double* t) {
  return zan{.parent = ld::gaussian1_distrib{t[0], t[1]}, .atom_alr = {t[2], t[3]}, .point = {0.0, 1.0}};
}

// the components of d^K l of a wrapper built from theta by `build`
template <class D, class B>
std::vector<double> wderiv(const B& build, const double* t, double y, int K) {
  constexpr int P = D::n_params;
  const D d = build(t);
  switch (K) {
    case 0: return {d.logpdf(y)};
    case 1: return as_vec<P, 1>(d.template logpdf_dk<1>(y));
    case 2: return as_vec<P, 2>(d.template logpdf_dk<2>(y));
    case 3: return as_vec<P, 3>(d.template logpdf_dk<3>(y));
    case 4: return as_vec<P, 4>(d.template logpdf_dk<4>(y));
    case 5: return as_vec<P, 5>(d.template logpdf_dk<5>(y));
    default: return as_vec<P, 6>(d.template logpdf_dk<6>(y));
  }
}

template <class D, class B>
void check_wrapper_derivs(const B& build, const std::vector<double>& th, double y, double rel) {
  constexpr int P = D::n_params;
  for (int K = 1; K <= 6; ++K) {
    const auto exact = wderiv<D>(build, th.data(), y, K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    if (scale == 0.0) scale = 1.0;
    for (const auto& idx : sorted_indices(P, K)) {
      const int j = idx.back();
      const std::vector<int> lower(idx.begin(), idx.end() - 1);
      const auto fd = diff5([&](const double* t) { return wderiv<D>(build, t, y, K - 1); }, th, j,
                            1e-3 * std::fmax(1e-3, std::fabs(th[j])));
      const double e = exact[position_of(P, idx)], a = fd[position_of(P, lower)];
      INFO("K = " << K << ", component " << position_of(P, idx) << ", y = " << y);
      CHECK(std::fabs(e - a) <= rel * scale);
    }
  }
}

}  // namespace

TEST_CASE("atom_mixture: the descriptors and the atoms") {
  CHECK(zag::kind == ld::distrib_kind::continuous);
  CHECK(zag::n_params == 3);
  CHECK(zag::n_atoms == 1);
  CHECK(zan::n_params == 4);
  CHECK(zan::n_atoms == 2);
  CHECK(std::string(zag::params[0].name) == "mu");
  CHECK(std::string(zag::params[1].name) == "sigma2");
  CHECK(std::string(zag::params[2].name) == "atom_alr1");
  CHECK(std::string(zan::params[3].name) == "atom_alr2");
  // a free coordinate lives on the whole line, is smooth and moves no support end
  CHECK(zag::params[2].lower == -ld::distrib_infinity);
  CHECK(zag::params[2].upper == ld::distrib_infinity);
  CHECK(zag::params[2].regularity == ld::infinitely_smooth);
  CHECK(zag::params[2].moves_support == false);

  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  CHECK(ld::distrib_n_atoms<zag>() == 1);
  CHECK(ld::distrib_atom_point(d, 0) == 0.0);
  CHECK(ld::distrib_atom_log_mass(d, 0) == d.logpdf(0.0));
  // The atom is NOT reported as a kink in y: a quadrature of the density is kept
  // off it by its integrand, which returns zero at an atom point, and the panel
  // routines are given the atom points separately (2026-11-17; docs/numerics.md,
  // "The generic cumulative routes of a law that carries atoms"). What y_kinks()
  // carries is the parent's, gamma2 declaring none.
  CHECK(d.y_kinks().count == 0);
  CHECK(ld::detail_generic::density_cuts(d, std::vector<double>(8).data()) == 1);
  // a family that declares none answers for itself
  CHECK(ld::distrib_n_atoms<ld::gamma2_distrib>() == 0);
}

TEST_CASE("atom_mixture: the masses and the density add to one") {
  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  double lpi[2];
  d.log_cells(lpi);
  CHECK(std::exp(lpi[0]) + std::exp(lpi[1]) == doctest::Approx(1.0).epsilon(1e-15));
  // the atom's mass is pi_0 and the density away from it integrates to pi_1
  CHECK(d.logpdf(0.0) == lpi[0]);
  const auto f = [&](double t) { return std::exp(d.logpdf(t)); };
  const auto q = laplace::math::quad::panels_quad(f, 1e-12, d.support_upper(), d.center(), d.scale());
  REQUIRE(q.converged);
  CHECK(q.value == doctest::Approx(std::exp(lpi[1])).epsilon(1e-10));
  CHECK(std::exp(d.logpdf(0.0)) + q.value == doctest::Approx(1.0).epsilon(1e-10));
}

TEST_CASE("atom_mixture: the value is the cell's log mass plus the parent's") {
  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  double lpi[2];
  d.log_cells(lpi);
  for (double y : {0.3, 1.5, 4.0, 20.0})
    CHECK(d.logpdf(y) == doctest::Approx(lpi[1] + d.parent.logpdf(y)).epsilon(1e-15));
  CHECK(d.logpdf(0.0) == lpi[0]);

  const zan e = make_zan(std::vector<double>{0.4, 1.3, -1.0, -2.0}.data());
  double lpe[3];
  e.log_cells(lpe);
  CHECK(e.logpdf(0.0) == lpe[0]);
  CHECK(e.logpdf(1.0) == lpe[1]);
  CHECK(e.logpdf(0.5) == doctest::Approx(lpe[2] + e.parent.logpdf(0.5)).epsilon(1e-15));
  CHECK(std::exp(lpe[0]) + std::exp(lpe[1]) + std::exp(lpe[2]) == doctest::Approx(1.0).epsilon(1e-15));
}

TEST_CASE("atom_mixture: the derivatives against central differences") {
  for (double y : {0.3, 1.5, 4.0})
    check_wrapper_derivs<zag>(make_zag, std::vector<double>{2.0, 0.5, -1.0}, y, 2e-5);
  // at the atom the tensor is the cell's alone, and the differences see that too
  check_wrapper_derivs<zag>(make_zag, std::vector<double>{2.0, 0.5, -1.0}, 0.0, 2e-5);
  for (double y : {-0.6, 0.5, 2.2})
    check_wrapper_derivs<zan>(make_zan, std::vector<double>{0.4, 1.3, -1.0, -2.0}, y, 2e-5);
  check_wrapper_derivs<zan>(make_zan, std::vector<double>{0.4, 1.3, -1.0, -2.0}, 1.0, 2e-5);
}

TEST_CASE("atom_mixture: the expected information against its own definition") {
  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  constexpr int P = zag::n_params;
  // -E[l_ab] over the law itself: the atoms carry their mass, the rest a quadrature
  const auto exact = d.expected_info();
  double scale = 0.0;
  for (int i = 0; i < exact.size; ++i) scale = std::fmax(scale, std::fabs(exact[i]));
  for (const auto& idx : sorted_indices(P, 2)) {
    const int pos = position_of(P, idx);
    const auto g = [&](double t) { return -d.logpdf_dk<2>(t)[pos] * std::exp(d.logpdf(t)); };
    const auto q = laplace::math::quad::panels_quad(g, 1e-12, d.support_upper(), d.center(), d.scale());
    REQUIRE(q.converged);
    double v = q.value;
    for (int i = 0; i < zag::n_atoms; ++i)
      v += -d.logpdf_dk<2>(d.point[i])[pos] * std::exp(d.logpdf(d.point[i]));
    INFO("component " << pos);
    CHECK(std::fabs(exact[pos] - v) <= 1e-8 * scale);
  }
}

TEST_CASE("atom_mixture: the derivatives of the expected information against differences") {
  const std::vector<double> th{2.0, 0.5, -1.0};
  constexpr int P = zag::n_params;
  const auto ivec = [&](const double* t, int K) { return ideriv(make_zag(t), K); };
  for (int K = 1; K <= 4; ++K) {
    const auto exact = ivec(th.data(), K);
    double scale = 0.0;
    for (double e : exact) scale = std::fmax(scale, std::fabs(e));
    const int npairs = ld::sym_tensor_size(P, 2);
    for (int p = 0; p < npairs; ++p)
      for (const auto& idx : sorted_indices(P, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return ivec(t, K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const int sz = ld::sym_tensor_size(P, K), szl = ld::sym_tensor_size(P, K - 1);
        const double e = exact[p * sz + position_of(P, idx)], a = fd[p * szl + position_of(P, lower)];
        INFO("K = " << K << ", pair " << p << ", component " << position_of(P, idx));
        CHECK(std::fabs(e - a) <= 2e-5 * scale);
      }
  }
}

TEST_CASE("atom_mixture: the cumulative functions against the masses and a quadrature") {
  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  double lpi[2];
  d.log_cells(lpi);
  // below the atom there is nothing
  CHECK(d.cdf(-1.0) == 0.0);
  CHECK(d.ccdf(-1.0) == doctest::Approx(1.0).epsilon(1e-15));
  // at the atom the whole of its mass is already in
  CHECK(d.cdf(0.0) == doctest::Approx(std::exp(lpi[0])).epsilon(1e-15));
  for (double y : {0.4, 1.5, 4.0}) {
    const auto f = [&](double t) { return std::exp(d.logpdf(t)); };
    const auto q = laplace::math::quad::panels_quad(f, 1e-12, y, d.center(), d.scale());
    REQUIRE(q.converged);
    INFO("y = " << y);
    CHECK(d.cdf(y) == doctest::Approx(std::exp(lpi[0]) + q.value).epsilon(1e-9));
    CHECK(d.cdf(y) + d.ccdf(y) == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(d.log_cdf(y) == doctest::Approx(std::log(d.cdf(y))).epsilon(1e-13));
  }

  const zan e = make_zan(std::vector<double>{0.4, 1.3, -1.0, -2.0}.data());
  double lpe[3];
  e.log_cells(lpe);
  CHECK(e.cdf(0.0) == doctest::Approx(std::exp(lpe[0]) + std::exp(lpe[2]) * e.parent.cdf(0.0)).epsilon(1e-14));
  CHECK(e.cdf(1.0) ==
        doctest::Approx(std::exp(lpe[0]) + std::exp(lpe[1]) + std::exp(lpe[2]) * e.parent.cdf(1.0)).epsilon(1e-14));
  for (double y : {-0.6, 0.5, 2.2}) CHECK(e.cdf(y) + e.ccdf(y) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("atom_mixture: the derivatives of the cumulative functions against differences") {
  const std::vector<double> th{2.0, 0.5, -1.0};
  constexpr int P = zag::n_params;
  for (bool upper : {false, true})
    for (double y : {0.4, 1.5, 4.0}) {
      for (int K = 1; K <= 6; ++K) {
        const auto exact = cderiv(make_zag(th.data()), y, K, upper);
        double scale = 0.0;
        for (double v : exact) scale = std::fmax(scale, std::fabs(v));
        if (scale == 0.0) scale = 1.0;
        for (const auto& idx : sorted_indices(P, K)) {
          const int j = idx.back();
          const std::vector<int> lower(idx.begin(), idx.end() - 1);
          const auto fd = diff5([&](const double* t) { return cderiv(make_zag(t), y, K - 1, upper); }, th, j,
                                1e-3 * std::fmax(1e-3, std::fabs(th[j])));
          const double a = exact[position_of(P, idx)], b = fd[position_of(P, lower)];
          INFO("upper = " << upper << ", y = " << y << ", K = " << K << ", component " << position_of(P, idx));
          CHECK(std::fabs(a - b) <= 3e-5 * scale);
        }
      }
    }
}

TEST_CASE("atom_mixture: the quantile inverts the cumulative, atoms included") {
  const zag d = make_zag(std::vector<double>{2.0, 0.5, -1.0}.data());
  double pi[2];
  {
    double w[1];
    d.cell_map().inverse(d.atom_alr, pi, w);
  }
  // inside the jump at the atom every probability returns the atom itself
  for (double p : {1e-6, 0.1, pi[0] * 0.5, pi[0]}) CHECK(d.quantile(p) == 0.0);
  // above it the quantile inverts F
  for (double p : {0.3, 0.5, 0.9, 0.999}) {
    const double q = d.quantile(p);
    INFO("p = " << p << " q = " << q);
    CHECK(q > 0.0);
    CHECK(d.cdf(q) == doctest::Approx(p).epsilon(1e-9));
  }

  const zan e = make_zan(std::vector<double>{0.4, 1.3, -1.0, -2.0}.data());
  double pe[3];
  {
    double w[1];
    e.cell_map().inverse(e.atom_alr, pe, w);
  }
  for (double p : {0.02, 0.2, 0.45, 0.6, 0.8, 0.98}) {
    const double q = e.quantile(p);
    INFO("p = " << p << " q = " << q);
    if (q != 0.0 && q != 1.0) CHECK(e.cdf(q) == doctest::Approx(p).epsilon(1e-9));
    else CHECK(e.cdf(q) >= p);  // an atom swallows an interval of probabilities
  }
}

TEST_CASE("atom_mixture: the starting value reads the atoms' frequencies") {
  const zag proto = make_zag(std::vector<double>{1.0, 1.0, 0.0}.data());
  std::vector<double> y;
  for (int i = 0; i < 30; ++i) y.push_back(0.0);          // the atom
  for (int i = 1; i <= 70; ++i) y.push_back(0.1 * i);     // the parent's cell
  const zag s = proto.start_value(y);
  double pi[2];
  {
    double w[1];
    s.cell_map().inverse(s.atom_alr, pi, w);
  }
  CHECK(pi[0] == doctest::Approx(0.3).epsilon(1e-12));
  CHECK(pi[1] == doctest::Approx(0.7).epsilon(1e-12));
  // the parent starts from the observations that are not at the atom
  double sum = 0.0;
  for (int i = 1; i <= 70; ++i) sum += 0.1 * i;
  CHECK(s.parent.mu == doctest::Approx(sum / 70.0).epsilon(1e-12));
  // a sample with no observation at the atom still starts strictly inside
  std::vector<double> z(y.begin() + 30, y.end());
  const zag s0 = proto.start_value(z);
  double pi0[2];
  {
    double w[1];
    s0.cell_map().inverse(s0.atom_alr, pi0, w);
  }
  CHECK(pi0[0] > 0.0);
  CHECK(pi0[0] < 0.02);
  // and the dispatch of generic_start finds the overload that takes the sample
  const zag sd = ld::distrib_start_value(proto, y);
  CHECK(sd.atom_alr[0] == s.atom_alr[0]);
}

TEST_CASE("atom_mixture: the factories of the familiar cases") {
  const auto d = ld::zero_adjusted(ld::gamma2_distrib{2.0, 0.5}, -1.0);
  CHECK(d.n_atoms == 1);
  CHECK(d.point[0] == 0.0);
  CHECK(d.logpdf(1.5) == make_zag(std::vector<double>{2.0, 0.5, -1.0}.data()).logpdf(1.5));
  const auto e = ld::zero_one_adjusted(ld::gaussian1_distrib{0.4, 1.3}, -1.0, -2.0);
  CHECK(e.n_atoms == 2);
  CHECK(e.point[0] == 0.0);
  CHECK(e.point[1] == 1.0);
  CHECK(e.cdf(0.5) + e.ccdf(0.5) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("atom_mixture: a Poisson with zero inflated and five adjusted") {
  using zip = ld::atom_mixture_distrib<ld::poisson_distrib, ld::inflated, ld::adjusted>;
  const auto build = [](const double* t) {
    return zip{.parent = ld::poisson_distrib{t[0]}, .atom_alr = {t[1], t[2]}, .point = {0.0, 5.0}};
  };
  const std::vector<double> th{3.5, -1.3217558400, -2.7080502011};
  const zip d = build(th.data());
  CHECK(zip::kind == ld::distrib_kind::discrete);
  CHECK(zip::n_params == 3);
  CHECK(zip::n_atoms == 0);  // a discrete parent already carries its support as mass
  CHECK(ld::distrib_n_atoms<zip>() == 0);

  double pi[3];
  {
    double w[1];
    d.cell_map().inverse(d.atom_alr, pi, w);
  }
  CHECK(pi[0] == doctest::Approx(0.20).epsilon(1e-9));
  CHECK(pi[1] == doctest::Approx(0.05).epsilon(1e-9));
  CHECK(pi[2] == doctest::Approx(0.75).epsilon(1e-9));

  const double p0 = std::exp(d.parent.logpmf(0.0)), p5 = std::exp(d.parent.logpmf(5.0));
  const double R = 1.0 - p5;  // only the adjusted point is taken from the parent
  CHECK(std::exp(d.log_renorm()) == doctest::Approx(R).epsilon(1e-14));
  // the inflated point keeps the parent's share on top of its own cell
  CHECK(std::exp(d.logpmf(0.0)) == doctest::Approx(pi[0] + pi[2] * p0 / R).epsilon(1e-13));
  // the adjusted point replaces it, and here with less than the Poisson gave
  CHECK(std::exp(d.logpmf(5.0)) == doctest::Approx(pi[1]).epsilon(1e-13));
  CHECK(pi[1] < p5);
  for (double y : {1.0, 2.0, 3.0, 7.0, 12.0})
    CHECK(std::exp(d.logpmf(y)) == doctest::Approx(pi[2] * std::exp(d.parent.logpmf(y)) / R).epsilon(1e-13));
  // and the whole thing is a probability distribution
  double total = 0.0;
  for (int k = 0; k <= 200; ++k) total += std::exp(d.logpmf(static_cast<double>(k)));
  CHECK(total == doctest::Approx(1.0).epsilon(1e-12));

  // the closed cumulative and information routes are not declared here, the
  // renormalization tying the two blocks together, so the generic ones answer
  for (double y : {0.0, 2.0, 5.0, 9.0}) {
    double s = 0.0;
    for (int k = 0; k <= static_cast<int>(y); ++k) s += std::exp(d.logpmf(static_cast<double>(k)));
    INFO("y = " << y);
    CHECK(ld::distrib_cdf(d, y) == doctest::Approx(s).epsilon(1e-11));
    CHECK(ld::distrib_cdf(d, y) + ld::distrib_ccdf(d, y) == doctest::Approx(1.0).epsilon(1e-11));
  }
  {  // and the generic expected information, in its score form
    const auto info = ld::distrib_expected_info<0>(d);
    CHECK(info[0] > 0.0);
  }

  // every order against five-point differences, at the two atoms and away from them
  constexpr int P = zip::n_params;
  for (double y : {0.0, 5.0, 2.0, 9.0}) {
    const auto at = [&](const double* t, int K) {
      const zip e = build(t);
      switch (K) {
        case 0: return std::vector<double>{e.logpmf(y)};
        case 1: return as_vec<P, 1>(e.logpmf_dk<1>(y));
        case 2: return as_vec<P, 2>(e.logpmf_dk<2>(y));
        case 3: return as_vec<P, 3>(e.logpmf_dk<3>(y));
        case 4: return as_vec<P, 4>(e.logpmf_dk<4>(y));
        case 5: return as_vec<P, 5>(e.logpmf_dk<5>(y));
        default: return as_vec<P, 6>(e.logpmf_dk<6>(y));
      }
    };
    for (int K = 1; K <= 6; ++K) {
      const auto exact = at(th.data(), K);
      double scale = 0.0;
      for (double v : exact) scale = std::fmax(scale, std::fabs(v));
      if (scale == 0.0) scale = 1.0;
      for (const auto& idx : sorted_indices(P, K)) {
        const int j = idx.back();
        const std::vector<int> lower(idx.begin(), idx.end() - 1);
        const auto fd = diff5([&](const double* t) { return at(t, K - 1); }, th, j,
                              1e-3 * std::fmax(1e-3, std::fabs(th[j])));
        const double a = exact[position_of(P, idx)], b = fd[position_of(P, lower)];
        INFO("y = " << y << ", K = " << K << ", component " << position_of(P, idx));
        CHECK(std::fabs(a - b) <= 3e-5 * scale);
      }
    }
  }
}

TEST_CASE("atom_mixture: the tensor is block diagonal, and exactly so") {

  const zan e = make_zan(std::vector<double>{0.4, 1.3, -1.0, -2.0}.data());
  constexpr int P = zan::n_params, Q = ld::gaussian1_distrib::n_params;
  for (double y : {-0.6, 0.5, 0.0, 1.0}) {
    const bool at_atom = (y == 0.0 || y == 1.0);
    for (int K = 1; K <= 6; ++K) {
      const auto t = wderiv<zan>(make_zan, std::vector<double>{0.4, 1.3, -1.0, -2.0}.data(), y, K);
      for (const auto& idx : sorted_indices(P, K)) {
        const bool has_parent = idx.front() < Q, has_free = idx.back() >= Q;
        const double v = t[position_of(P, idx)];
        INFO("y = " << y << ", K = " << K << ", component " << position_of(P, idx));
        if (has_parent && has_free) CHECK(v == 0.0);        // mixed
        else if (at_atom && has_parent) CHECK(v == 0.0);    // the parent does not appear at an atom
      }
    }
  }
}

// The kernel lives in a named function with external linkage, not in the body of the TEST_CASE: see the
// comment on gaussian3_kernel_case in test_gaussian3.cpp for why.
void atom_mixture_kernel_case() {
  const int n = 32;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_atom_mixture", n, KOKKOS_LAMBDA(const int i) {
        const double y = 0.2 + 0.1 * i;
        const zag d{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}};
        out(i) = d.logpdf(y) + d.logpdf_d6(y)[4] + d.log_cdf_d6(y)[2] + d.log_ccdf_d6(y)[0] +
                 d.expected_info_d4().pair[2][3] + d.logpdf(0.0) + d.log_cdf(y);
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = 0.2 + 0.1 * i;
    const zag d{.parent = ld::gamma2_distrib{2.0, 0.5}, .atom_alr = {-1.0}, .point = {0.0}};
    const double r = d.logpdf(y) + d.logpdf_d6(y)[4] + d.log_cdf_d6(y)[2] + d.log_ccdf_d6(y)[0] +
                     d.expected_info_d4().pair[2][3] + d.logpdf(0.0) + d.log_cdf(y);
    INFO("i = " << i);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-12));
  }
}

// the discrete side, with one atom inflated and one adjusted, which is the route that carries the
// renormalization and the two-term mixture
void atom_mixture_discrete_kernel_case() {
  using zip = ld::atom_mixture_distrib<ld::poisson_distrib, ld::inflated, ld::adjusted>;
  const int n = 16;
  Kokkos::View<double*> out("out", n);
  Kokkos::parallel_for(
      "distrib_atom_mixture_discrete", n, KOKKOS_LAMBDA(const int i) {
        const double y = static_cast<double>(i);
        const zip d{.parent = ld::poisson_distrib{3.5}, .atom_alr = {-1.3217558400, -2.7080502011},
                    .point = {0.0, 5.0}};
        out(i) = d.logpmf(y) + d.logpmf_d6(y)[3] + d.log_renorm();
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  for (int i = 0; i < n; ++i) {
    const double y = static_cast<double>(i);
    const zip d{.parent = ld::poisson_distrib{3.5}, .atom_alr = {-1.3217558400, -2.7080502011},
                .point = {0.0, 5.0}};
    const double r = d.logpmf(y) + d.logpmf_d6(y)[3] + d.log_renorm();
    INFO("i = " << i);
    CHECK(h(i) == doctest::Approx(r).epsilon(1e-12));
  }
}

TEST_CASE("atom_mixture: a continuous parent inside a kernel") { atom_mixture_kernel_case(); }

TEST_CASE("atom_mixture: a discrete parent with an inflated atom inside a kernel") {
  atom_mixture_discrete_kernel_case();
}
