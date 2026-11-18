// univariate_as_multivariate<D> over any univariate law, and the independent
// product of such laws (Giovanni, 2026-11-17): a law with an atom (zero-adjusted
// gamma1 and gaussian2, atom_mixture_distrib, a wrapper that is not an
// aggregate of its parameters), discrete laws (poisson, negbin2), and means
// with a bounded domain (gamma1, poisson, negbin2: mu > 0).
//  - the adapter equals the law rebuilt by hand, value and every parameter
//    derivative to order 3, on the atom (the log mass) and off it;
//  - parameter derivatives against central differences of the value;
//  - the descriptors: kind, the coordinate atoms, the bounds of mu and theta;
//  - the product: the sum of the copies, its parameter derivatives against
//    central differences (theta by copy and shared), the coordinate atoms and
//    bounds mapped copy by copy;
//  - the total mass is 1: the atom's mass plus the integral of the continuous
//    part (trapezoid rule), the sum over the integers for a discrete law, and
//    the four pieces of a 2-d product with an atom on each coordinate;
//  - the adapter over the zero-adjusted gamma and over poisson inside a Kokkos
//    kernel (a named function with external linkage), equal to the host.
#include <doctest/doctest.h>

#include <laplace/distrib/atom_mixture.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/negbin2.hpp>
#include <laplace/distrib/poisson.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <vector>

namespace uam_test {

namespace ld = laplace::distrib;
using zgamma = decltype(ld::zero_adjusted(ld::gamma1_distrib{1.0, 1.0}, 0.0));
using zgauss = decltype(ld::zero_adjusted(ld::gaussian2_distrib{0.0, 1.0}, 0.0));
template <class D>
using uam = ld::univariate_as_multivariate<D>;

static_assert(uam<zgamma>::kind == ld::distrib_kind::continuous && uam<ld::poisson_distrib>::kind == ld::distrib_kind::discrete);
static_assert(ld::multivariate_distrib<uam<ld::poisson_distrib>> && ld::multivariate_distrib<uam<ld::negbin2_distrib>> &&
              ld::multivariate_distrib<uam<zgamma>> && ld::multivariate_distrib<uam<zgauss>>);
static_assert(uam<zgamma>::n_params == 3 && uam<zgamma>::params[0].lower == 0.0 &&
              uam<zgamma>::params[2].lower == -ld::distrib_infinity);

// the law the adapter must rebuild, by hand
KOKKOS_INLINE_FUNCTION zgamma zg(double mu, double phi, double alr) { return ld::zero_adjusted(ld::gamma1_distrib{mu, phi}, alr); }
KOKKOS_INLINE_FUNCTION zgauss zn(double mu, double v, double alr) { return ld::zero_adjusted(ld::gaussian2_distrib{mu, v}, alr); }

// value of a uam law at (y, x, theta)
template <class M>
KOKKOS_INLINE_FUNCTION double value(const M& m, double y, double x, const double* th) {
  const auto st = m.theta_state(th, nullptr);
  if constexpr (M::kind == ld::distrib_kind::continuous) return m.logpdf(st, &y, &x);
  else return m.logpmf(st, &y, &x);
}
// the (K, A) parameter derivative of a uam law along idx
template <class M, int K, int A>
KOKKOS_INLINE_FUNCTION double deriv(const M& m, double y, double x, const double* th, const int* idx) {
  const auto st = m.theta_state(th, nullptr);
  double o = 0.0;
  if constexpr (M::kind == ld::distrib_kind::continuous) m.template logpdf_dk<K, A>(st, &y, &x, idx, &o, nullptr);
  else m.template logpmf_dk<K, A>(st, &y, &x, idx, &o, nullptr);
  return o;
}

// largest relative error of the first-order parameter derivatives (each theta, and mu)
// and of the mixed (theta_0, mu) and (theta_0, theta_last) ones, against central differences
template <class M>
KOKKOS_INLINE_FUNCTION double fd_error(const M& m, double y, double x, const double* th) {
  const int q = m.theta_size();
  double t2[4], err = 0.0;
  auto rel = [](double a, double b) { return Kokkos::fabs(a - b) / Kokkos::fmax(1.0, Kokkos::fabs(a)); };
  for (int j = 0; j < q; ++j) {
    for (int i = 0; i < q; ++i) t2[i] = th[i];
    const double h = 1e-6 * Kokkos::fmax(1.0, Kokkos::fabs(th[j]));
    t2[j] = th[j] + h;
    const double p = value(m, y, x, t2);
    t2[j] = th[j] - h;
    const double n = value(m, y, x, t2);
    const int idx[1] = {j};
    err = Kokkos::fmax(err, rel(deriv<M, 1, 0>(m, y, x, th, idx), (p - n) / (2 * h)));
    // d/d theta_j of the mu derivative
    t2[j] = th[j] + h;
    const double pa = deriv<M, 0, 1>(m, y, x, t2, nullptr);
    t2[j] = th[j] - h;
    const double na = deriv<M, 0, 1>(m, y, x, t2, nullptr);
    err = Kokkos::fmax(err, rel(deriv<M, 1, 1>(m, y, x, th, idx), (pa - na) / (2 * h)));
    // d/d theta_last of the theta_j derivative
    const int l = q - 1;
    const int pair[2] = {j, l};
    for (int i = 0; i < q; ++i) t2[i] = th[i];
    const double hl = 1e-6 * Kokkos::fmax(1.0, Kokkos::fabs(th[l]));
    t2[l] = th[l] + hl;
    const double pp = deriv<M, 1, 0>(m, y, x, t2, idx);
    t2[l] = th[l] - hl;
    const double nn = deriv<M, 1, 0>(m, y, x, t2, idx);
    err = Kokkos::fmax(err, rel(deriv<M, 2, 0>(m, y, x, th, pair), (pp - nn) / (2 * hl)));
  }
  const double h = 1e-6 * Kokkos::fmax(1.0, Kokkos::fabs(x));
  err = Kokkos::fmax(err, rel(deriv<M, 0, 1>(m, y, x, th, nullptr),
                              (value(m, y, x + h, th) - value(m, y, x - h, th)) / (2 * h)));
  return err;
}

constexpr double zg_theta[2] = {0.25, -0.7};  // phi (shape 4), alr of the atom at 0
constexpr double nb_theta[1] = {2.5};         // negbin2 theta

// the adapter inside a kernel: values and finite-difference errors
void run_kernel(Kokkos::View<double*> res) {
  Kokkos::parallel_for(
      "uam_any_law", 1, KOKKOS_LAMBDA(int) {
        const uam<zgamma> g(zg(1.0, 1.0, 0.0));
        const uam<ld::poisson_distrib> p;
        const uam<ld::negbin2_distrib> nb;
        const double th[2] = {0.25, -0.7}, tn[1] = {2.5};  // zg_theta and nb_theta (globals are not device code)
        res(0) = value(g, 0.0, 1.5, th);
        res(1) = value(g, 0.8, 1.5, th);
        res(2) = fd_error(g, 0.0, 1.5, th);
        res(3) = fd_error(g, 0.8, 1.5, th);
        res(4) = value(p, 3.0, 2.2, nullptr);
        res(5) = fd_error(p, 3.0, 2.2, nullptr);
        res(6) = value(nb, 4.0, 2.2, tn);
        res(7) = fd_error(nb, 4.0, 2.2, tn);
        // a derivative in y off the atom: the (y, phi) component, through atom_mixture's logpdf_dy_dk
        const double y = 0.8, x = 1.5;
        const int i0[1] = {0};
        g.logpdf_dy_dk<1, 1>(g.theta_state(th, nullptr), &y, &x, i0, &res(8), nullptr);
      });
}

}  // namespace uam_test

TEST_CASE("univariate_as_multivariate over a law with an atom: values, derivatives, descriptors") {
  using namespace uam_test;
  const uam<zgamma> m(zg(1.0, 1.0, 0.0));
  const double x = 1.5;
  const zgamma d = zg(x, zg_theta[0], zg_theta[1]);
  CHECK(m.n_coord_atoms(0) == 1);
  CHECK(m.coord_atom_point(0, 0) == 0.0);
  CHECK(ld::mv_n_coord_atoms(m, 0) == 1);
  CHECK(ld::mv_coord_lower(m, 0, 0) == 0.0);
  CHECK(ld::mv_coord_upper(m, 0, 0) == ld::distrib_infinity);
  for (double y : {0.0, 0.8, 3.1}) {
    CAPTURE(y);
    CHECK(value(m, y, x, zg_theta) == d.logpdf(y));
    // every parameter derivative to order 3 is the component of the law's own tensor
    const auto t1 = d.logpdf_dk<1>(y);
    const auto t3 = d.logpdf_dk<3>(y);
    const int i0[1] = {0}, i1[1] = {1}, i01[2] = {0, 1};
    CHECK(deriv<uam<zgamma>, 0, 1>(m, y, x, zg_theta, nullptr) == t1[0]);
    CHECK(deriv<uam<zgamma>, 1, 0>(m, y, x, zg_theta, i1) == t1[2]);
    double o = 0.0;
    const auto st = m.theta_state(zg_theta, nullptr);
    REQUIRE(m.logpdf_dk<2, 1>(st, &y, &x, i01, &o, nullptr));
    CHECK(o == t3[ld::sym_tensor_position<3, 3>({0, 1, 2})]);
    REQUIRE(m.logpdf_dk<1, 2>(st, &y, &x, i0, &o, nullptr));
    CHECK(o == t3[ld::sym_tensor_position<3, 3>({0, 0, 1})]);
    CHECK(fd_error(m, y, x, zg_theta) < 1e-7);
  }
  // a zero-adjusted gaussian2: real mean, atom at 0
  const uam<zgauss> n(zn(0.0, 1.0, 0.0));
  const double tn[2] = {0.6, 0.4};
  CHECK(ld::mv_coord_lower(n, 0, 0) == -ld::distrib_infinity);
  for (double y : {0.0, -0.4, 1.3}) {
    CAPTURE(y);
    CHECK(value(n, y, 0.2, tn) == zn(0.2, 0.6, 0.4).logpdf(y));
    CHECK(fd_error(n, y, 0.2, tn) < 1e-7);
  }
}

TEST_CASE("univariate_as_multivariate over discrete laws: values, derivatives, descriptors") {
  using namespace uam_test;
  const uam<ld::poisson_distrib> p;
  const uam<ld::negbin2_distrib> nb;
  CHECK(p.theta_size() == 0);
  CHECK(ld::mv_coord_lower(p, 0, 0) == 0.0);
  CHECK(ld::mv_n_coord_atoms(p, 0) == 0);
  for (double y : {0.0, 1.0, 7.0}) {
    CAPTURE(y);
    CHECK(value(p, y, 2.2, nullptr) == ld::poisson_distrib{2.2}.logpmf(y));
    CHECK(value(nb, y, 2.2, nb_theta) == ld::negbin2_distrib{2.2, nb_theta[0]}.logpmf(y));
    CHECK(deriv<uam<ld::negbin2_distrib>, 0, 2>(nb, y, 2.2, nb_theta, nullptr) ==
          ld::negbin2_distrib{2.2, nb_theta[0]}.logpmf_dk<2>(y)[0]);
    CHECK(fd_error(p, y, 2.2, nullptr) < 1e-7);
    CHECK(fd_error(nb, y, 2.2, nb_theta) < 1e-7);
  }
}

TEST_CASE("univariate_as_multivariate: the total mass is one") {
  using namespace uam_test;
  {  // zero-adjusted gamma: the atom's mass plus the integral of the continuous part
    const uam<zgamma> m(zg(1.0, 1.0, 0.0));
    const double x = 1.5, h = 1e-3;
    double s = 0.0;
    for (double y = h; y < 40.0; y += h) s += std::exp(value(m, y, x, zg_theta));
    CHECK(std::fabs(std::exp(value(m, 0.0, x, zg_theta)) + s * h - 1.0) < 1e-9);
  }
  {  // zero-adjusted gaussian2: the grid avoids the atom
    const uam<zgauss> n(zn(0.0, 1.0, 0.0));
    const double tn[2] = {0.6, 0.4}, h = 1e-2;
    double s = 0.0;
    for (double y = -12.0 + h / 2; y < 12.0; y += h) s += std::exp(value(n, y, 0.2, tn));
    CHECK(std::fabs(std::exp(value(n, 0.0, 0.2, tn)) + s * h - 1.0) < 1e-12);
  }
  {  // poisson and negbin2: the sum over the integers
    const uam<ld::poisson_distrib> p;
    const uam<ld::negbin2_distrib> nb;
    double sp = 0.0, sn = 0.0;
    for (int y = 0; y < 200; ++y) {
      sp += std::exp(value(p, y, 2.2, nullptr));
      sn += std::exp(value(nb, y, 2.2, nb_theta));
    }
    CHECK(std::fabs(sp - 1.0) < 1e-13);
    CHECK(std::fabs(sn - 1.0) < 1e-13);
  }
}

TEST_CASE("independent_product of laws with atoms and of discrete laws") {
  using namespace uam_test;
  using cont = ld::independent_product_distrib<uam<zgamma>, uam<zgauss>>;
  using disc = ld::independent_product_distrib<uam<ld::poisson_distrib>, uam<ld::negbin2_distrib>>;
  static_assert(cont::kind == ld::distrib_kind::continuous && disc::kind == ld::distrib_kind::discrete);
  static_assert(ld::multivariate_distrib<disc>);
  // two zero-adjusted gammas with theta by copy, one zero-adjusted gaussian2
  const cont c(ld::independent_product_component<uam<zgamma>>{uam<zgamma>(zg(1.0, 1.0, 0.0)), 2, 2, nullptr},
               ld::independent_product_component<uam<zgauss>>{uam<zgauss>(zn(0.0, 1.0, 0.0)), 1, 1, nullptr});
  // theta, param-major: phi_0, phi_1, alr_0, alr_1 (gammas), v, alr (gaussian2)
  const double th[6] = {0.25, 0.4, -0.7, 0.3, 0.6, 0.4};
  const double x[3] = {1.5, 0.9, 0.2};
  CHECK(c.n_coord_atoms(0) == 1);
  CHECK(c.n_coord_atoms(2) == 1);
  CHECK(c.coord_atom_point(1, 0) == 0.0);
  CHECK(ld::mv_coord_lower(c, 0, 1) == 0.0);
  CHECK(ld::mv_coord_lower(c, 0, 2) == -ld::distrib_infinity);
  CHECK(ld::mv_coord_lower(c, 1, 1) == 0.0);                     // phi of the second gamma
  CHECK(ld::mv_coord_lower(c, 2, 0) == -ld::distrib_infinity);   // alr of the first gamma
  CHECK(ld::mv_coord_lower(c, 3, 0) == 0.0);                     // v of the gaussian2
  std::vector<double> w(c.state_size() + c.scratch_size() + 256);
  const auto st = c.theta_state(th, w.data());
  double* ws = w.data() + c.state_size();
  for (const auto& y : {std::vector<double>{0.0, 0.7, 0.0}, std::vector<double>{1.1, 0.0, -0.3}}) {
    const double twin = zg(x[0], th[0], th[2]).logpdf(y[0]) + zg(x[1], th[1], th[3]).logpdf(y[1]) +
                        zn(x[2], th[4], th[5]).logpdf(y[2]);
    CHECK(std::fabs(c.logpdf(st, y.data(), x) - twin) < 1e-14);
    // the theta gradient and the mu gradient against central differences
    for (int j = 0; j < 6; ++j) {
      CAPTURE(j);
      double g = 0.0, t2[6];
      const int idx[1] = {j};
      REQUIRE(c.logpdf_dk<1>(st, y.data(), x, idx, &g, ws));
      for (int i = 0; i < 6; ++i) t2[i] = th[i];
      const double h = 1e-6;
      std::vector<double> w2(w.size());
      t2[j] = th[j] + h;
      const double p = c.logpdf(c.theta_state(t2, w2.data()), y.data(), x);
      t2[j] = th[j] - h;
      const double n = c.logpdf(c.theta_state(t2, w2.data()), y.data(), x);
      CHECK(std::fabs(g - (p - n) / (2 * h)) < 1e-7 * std::fmax(1.0, std::fabs(g)));
    }
    double gm[3];
    REQUIRE(c.logpdf_dk<0, 1>(st, y.data(), x, nullptr, gm, ws));
    for (int i = 0; i < 3; ++i) {
      double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
      xp[i] += 1e-6;
      xm[i] -= 1e-6;
      CHECK(std::fabs(gm[i] - (c.logpdf(st, y.data(), xp) - c.logpdf(st, y.data(), xm)) / 2e-6) < 1e-7);
    }
  }
  // the discrete product: poisson x 2 (no theta), negbin2 x 1
  const disc dp(ld::independent_product_component<uam<ld::poisson_distrib>>{{}, 2, 1, nullptr},
                ld::independent_product_component<uam<ld::negbin2_distrib>>{{}, 1, 1, nullptr});
  CHECK(ld::mv_n_coord_atoms(dp, 0) == 0);
  const double tn[1] = {2.5}, xd[3] = {2.2, 0.7, 3.0}, yd[3] = {3.0, 0.0, 5.0};
  std::vector<double> wd(dp.state_size() + dp.scratch_size() + 64);
  const auto sd = dp.theta_state(tn, wd.data());
  const double twin = ld::poisson_distrib{2.2}.logpmf(3.0) + ld::poisson_distrib{0.7}.logpmf(0.0) +
                      ld::negbin2_distrib{3.0, 2.5}.logpmf(5.0);
  CHECK(std::fabs(dp.logpmf(sd, yd, xd) - twin) < 1e-14);
  double gt = 0.0;
  const int i0[1] = {0};
  REQUIRE(dp.logpmf_dk<1>(sd, yd, xd, i0, &gt, wd.data() + dp.state_size()));
  CHECK(gt == ld::negbin2_distrib{3.0, 2.5}.logpmf_dk<1>(5.0)[1]);
  // total mass of the 2-d product of the zero-adjusted gamma and gaussian2: four pieces
  using two = ld::independent_product_distrib<uam<zgamma>, uam<zgauss>>;
  const two t(ld::independent_product_component<uam<zgamma>>{uam<zgamma>(zg(1.0, 1.0, 0.0)), 1, 1, nullptr},
              ld::independent_product_component<uam<zgauss>>{uam<zgauss>(zn(0.0, 1.0, 0.0)), 1, 1, nullptr});
  const double tt[4] = {0.25, -0.7, 0.6, 0.4}, xt[2] = {1.5, 0.2};
  std::vector<double> wt(t.state_size() + t.scratch_size());
  const auto s2 = t.theta_state(tt, wt.data());
  auto f = [&](double a, double b) {
    const double yy[2] = {a, b};
    return std::exp(t.logpdf(s2, yy, xt));
  };
  const double ha = 2e-3, hb = 2e-2;
  double both = f(0.0, 0.0), only_a = 0.0, only_b = 0.0, none = 0.0;
  for (double a = ha; a < 40.0; a += ha) only_b += f(a, 0.0) * ha;
  for (double b = -12.0 + hb / 2; b < 12.0; b += hb) only_a += f(0.0, b) * hb;
  for (double a = ha; a < 40.0; a += ha)
    for (double b = -12.0 + hb / 2; b < 12.0; b += hb) none += f(a, b) * ha * hb;
  CHECK(std::fabs(both + only_a + only_b + none - 1.0) < 1e-8);
}

TEST_CASE("univariate_as_multivariate in a kernel equals the host") {
  using namespace uam_test;
  Kokkos::View<double*> res("res", 9);
  run_kernel(res);
  auto h = Kokkos::create_mirror_view(res);
  Kokkos::deep_copy(h, res);
  const uam<zgamma> g(zg(1.0, 1.0, 0.0));
  const uam<ld::poisson_distrib> p;
  const uam<ld::negbin2_distrib> nb;
  CHECK(std::fabs(h(0) - value(g, 0.0, 1.5, zg_theta)) < 1e-13);
  CHECK(std::fabs(h(1) - value(g, 0.8, 1.5, zg_theta)) < 1e-13);
  CHECK(h(2) < 1e-7);
  CHECK(h(3) < 1e-7);
  CHECK(std::fabs(h(4) - value(p, 3.0, 2.2, nullptr)) < 1e-13);
  CHECK(h(5) < 1e-7);
  CHECK(std::fabs(h(6) - value(nb, 4.0, 2.2, nb_theta)) < 1e-13);
  CHECK(h(7) < 1e-7);
  CHECK(h(8) == ld::gamma1_distrib{1.5, zg_theta[0]}.logpdf_dy_dk<1, 1>(0.8)[1]);
}
