// The inner fit of the engine (engine/inner_fit.hpp) on the models of
// engine_models.hpp: Newton through the library's optimizer with the dense
// and the sparse curvature, a Gaussian prior on the group effects with a
// fixed variance, held coordinates, the cache of the last assembly, and the
// stationarity of the fitted point checked on the gradient of the objective.
// Host code: the kernels are launched by the library's functions.

#include <doctest/doctest.h>

#include <laplace/engine/inner_fit.hpp>

#include "engine_models.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
using namespace engine_test;

lo::optimizer_vector point(const std::vector<double>& v) {
  lo::optimizer_vector x("x0", v.size());
  for (std::size_t j = 0; j < v.size(); ++j) x(j) = v[j];
  return x;
}

// the group effects of a model start at n_global: a prior N(0, tau^2) on each
std::vector<le::quadratic_penalty_term> group_prior(const le::row_model_spec& s, double tau) {
  return {le::quadratic_penalty_term{s.n_global, s.n_groups * s.group_width, 1, {1.0 / (tau * tau)}}};
}

// Newton with a tight stop: the default decrement tolerance (1e-6, absolute) leaves gradients of 1e-3
// on these sizes, which is converged by its rule but too loose to compare two fits
lo::newton_optimizer<> tight() {
  lo::newton_optimizer<> nt;
  nt.stop_rule.tolerance = 1e-14;
  return nt;
}

template <class Obj>
double gradient_norm(Obj& obj, const lo::optimizer_point& x) {
  lo::optimizer_vector g("g", x.extent(0));
  obj.value_gradient(x, g);
  double m = 0.0;
  for (std::size_t j = 0; j < g.extent(0); ++j) m = std::max(m, std::abs(g(j)));
  return m;
}

}  // namespace

TEST_CASE("engine inner fit: Gaussian with a group prior, dense and sparse curvature agree") {
  const auto s = gauss_spec(2000, 40, 21);
  const auto m = le::build_row_model(s, 2);
  const int N = m.n_coordinates();
  std::vector<double> x0(N, 0.0);
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> dense(m, group_prior(s, 0.5));
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> sparse(m, group_prior(s, 0.5));
  const auto rd = le::fit_inner(dense, point(x0), tight());
  const auto rs = le::fit_inner(sparse, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(rd.optimizer.status));
  REQUIRE(lo::optimizer_succeeded(rs.optimizer.status));
  for (int j = 0; j < N; ++j) CHECK(std::abs(rd.optimizer.point(j) - rs.optimizer.point(j)) <= 1e-7);
  CHECK(gradient_norm(dense, rd.optimizer.point) <= 1e-4);
  CHECK(rd.optimizer.point(1) == doctest::Approx(0.5).epsilon(0.1));   // the slope of x
  CHECK(rd.optimizer.point(3) == doctest::Approx(0.8).epsilon(0.25));  // the slope of w in log sigma
  // every curvature call of the sparse fit ran one kernel of order two, or two with a fallback
  CHECK(rs.kernels.calls[2] == rs.optimizer.curvature_evaluations);
  CHECK(rs.kernels.calls[3] == rs.kernels.expected_fallbacks);

  // the cache: asking again at the fitted point runs no kernel
  sparse.reset_counts();
  lo::optimizer_vector g("g", N);
  lo::sparse_curvature_matrix H("H", N, N);
  sparse.value_gradient_curvature(rs.optimizer.point, g, H);
  sparse.value(rs.optimizer.point);
  CHECK(sparse.counts().calls[2] + sparse.counts().calls[3] == 0);
  CHECK(sparse.counts().reused == 2);
}

TEST_CASE("engine inner fit: held coordinates keep their value") {
  const auto s = gauss_spec(1000, 20, 22);
  const auto m = le::build_row_model(s, 2);
  std::vector<double> x0(m.n_coordinates(), 0.0);
  x0[2] = -0.25, x0[3] = 0.5;
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> obj(
      m, group_prior(s, 0.5), le::row_curvature::observed_else_expected, {2, 3});
  const auto r = le::fit_inner(obj, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  CHECK(r.optimizer.point(2) == -0.25);
  CHECK(r.optimizer.point(3) == 0.5);
  CHECK(gradient_norm(obj, r.optimizer.point) <= 1e-4);
}

TEST_CASE("engine inner fit: gamma with a logistic curve and a group prior, sparse curvature") {
  const auto s = gamma_spec(3000, 30, 23);
  const auto m = le::build_row_model(s, 2);
  std::vector<double> x0(m.n_coordinates(), 0.0);
  x0[0] = 2.5, x0[1] = 4.0, x0[2] = 0.0, x0[3] = -1.0;
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> obj(m, group_prior(s, 0.3));
  const auto r = le::fit_inner(obj, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  CHECK(gradient_norm(obj, r.optimizer.point) <= 1e-4);
  CHECK(r.optimizer.point(0) == doctest::Approx(3.0).epsilon(0.1));
  CHECK(r.optimizer.point(1) == doctest::Approx(5.0).epsilon(0.1));
  CHECK(std::exp(r.optimizer.point(2)) == doctest::Approx(1.5).epsilon(0.15));
  CHECK(r.optimizer.point(3) == doctest::Approx(-2.0).epsilon(0.1));
}

// The model of prototipi/kokkos_fit/lk.cpp (kokkos_fit_model in engine_models.hpp) on the first 2000
// rows of its data (20 groups). The reference is lk.cpp's inner Newton fit at the hyperparameters and
// the coefficients of sigma of its REML fit, these held, from its start (the mean of y, zeros).
TEST_CASE("engine inner fit: the Gaussian model of the kokkos_fit prototype, against its inner fit") {
  const auto pf = kokkos_fit_model(LAPLACE_REFERENCE_DIR "/../engine/kokkos_fit_2000.csv", -0.91756492601312334,
                                   -4.8811529834528038, -0.98602388660648632, 0.27445795123482364);
  REQUIRE(pf.spec.y.size() == 2000);
  const std::vector<double> ref = {
      1.0729694752594221,    0.51480724758191043,  -0.012087307878737557, 0.38105911958652383,  0.94854753854541107,
      1.1183729016736204,    0.61916808546726221,  -0.44928891864238563,  -1.0618468420269549,  -0.7936525008667431,
      -0.16851228918507982,  -0.98602388660648632, 0.27445795123482364,   -0.072305545896541201, -0.33750084871849761,
      0.45155109890756262,   -0.35501893883035096, 0.23511437028018189,   0.07574402174651787,  0.13083832695868422,
      0.089627103052441121,  -0.1565435842375752,  0.21461069926805948,   0.73996807930533237,  0.27989948288967131,
      0.017267779000870877,  -0.2461451274439444,  -1.2395960344720096,   0.0041662967691373839, -0.22910786401393357,
      0.084581719592005056,  -0.032412682735832241, 0.34526164857802366};
  const auto m = le::build_row_model(pf.spec, 2);
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> obj(
      m, pf.penalties, le::row_curvature::observed_else_expected, pf.held);
  const auto r = le::fit_inner(obj, point(pf.x0), tight());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  for (std::size_t j = 0; j < ref.size(); ++j) CHECK(std::abs(r.optimizer.point(j) - ref[j]) <= 1e-9);
}

// Model C of prototipi/node_dispatch/hy4_main.cpp (lk3_gamma_model in engine_models.hpp): 2000 rows,
// 20 groups, the response simulated by that driver. The reference is its inner Newton fit at the
// hyperparameter and the coefficient c0 of its REML fit, c0 held, from its start.
TEST_CASE("engine inner fit: the gamma model with a logistic curve of the lk3 prototype, against its inner fit") {
  const auto pf =
      lk3_gamma_model(LAPLACE_REFERENCE_DIR "/../engine/lk3_gamma_2000.csv", 3.6755854596987816, -2.947691401829033);
  REQUIRE(pf.spec.y.size() == 2000);
  const std::vector<double> ref = {
      205.66146954486436,  685.93202540802849,  347.66548474223237,  -2.947691401829033,  -58.807877891149673,
      -34.750703740434972, -11.442377661187981, 10.022696049638867,  100.76602212269947,  45.105349474972101,
      26.839902123318517,  -39.848897262965664, -21.395730126180343, 11.20247382368659,   20.283768992763559,
      -39.761651224268086, 57.499013329736059,  -42.550935671286268, -20.888863073175425, -25.716553264025567,
      -10.887191812090675, 15.043697945548853,  5.7888364757847892,  13.499021388615727};
  const auto m = le::build_row_model(pf.spec, 2);
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> obj(
      m, pf.penalties, le::row_curvature::observed_else_expected, pf.held);
  const auto r = le::fit_inner(obj, point(pf.x0), tight());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  for (std::size_t j = 0; j < ref.size(); ++j)
    CHECK(std::abs(r.optimizer.point(j) - ref[j]) <= 1e-9 * (1.0 + std::abs(ref[j])));
}

TEST_CASE("engine inner fit: Student t, dense curvature, observed and expected information reach one point") {
  const auto s = student_spec(3000, 24);
  const auto m = le::build_row_model(s, 3);
  const std::vector<double> x0 = {0.0, 0.0, 0.0, 0.0, std::log(10.0)};
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> obs(m, {}, le::row_curvature::observed);
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> exp(m, {}, le::row_curvature::expected);
  const auto ro = le::fit_inner(obs, point(x0), tight());
  const auto re = le::fit_inner(exp, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(ro.optimizer.status));
  REQUIRE(lo::optimizer_succeeded(re.optimizer.status));
  for (int j = 0; j < 5; ++j) CHECK(std::abs(ro.optimizer.point(j) - re.optimizer.point(j)) <= 1e-5);
  CHECK(gradient_norm(obs, ro.optimizer.point) <= 1e-4);
  CHECK(std::exp(ro.optimizer.point(4)) == doctest::Approx(5.0).epsilon(0.4));
}
