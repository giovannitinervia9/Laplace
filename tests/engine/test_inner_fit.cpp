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
std::vector<le::penalty_term> group_prior(const le::row_model_spec& s, double tau) {
  return {group_gaussian_prior(s, 0, -1, tau)};
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

// A penalty whose replicate crosses the groups: a random walk over the group effects with a small
// ridge (the Gaussian prior with precision lambda (D'D + I/10), D the first differences), so the sparse
// pattern holds entries outside the arrowhead; dense and sparse curvatures agree, and the fit lands on
// the same point.
TEST_CASE("engine inner fit: a penalty across the groups, outside the arrowhead") {
  const auto s = gauss_spec(1500, 15, 24);
  const auto m = le::build_row_model(s, 2);
  const int N = m.n_coordinates(), G = s.n_groups;
  std::vector<double> S(G * G, 0.0);
  for (int g = 0; g + 1 < G; ++g) {
    S[g + G * g] += 1.0, S[(g + 1) + G * (g + 1)] += 1.0;
    S[g + G * (g + 1)] -= 1.0, S[(g + 1) + G * g] -= 1.0;
  }
  // a small ridge: without it a common shift of the effects is the intercept's, and no minimum is unique
  for (int g = 0; g < G; ++g) S[g + G * g] += 0.1;
  const std::vector<le::penalty_term> pen = {smooth_prior(s.n_global, G, S, -1, std::log(3.0))};
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> dense(m, pen);
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> sparse(m, pen);
  // the curvature at a point, entry by entry
  std::vector<double> x(N);
  for (int j = 0; j < N; ++j) x[j] = 0.1 * std::sin(1.0 + j);
  lo::optimizer_vector gd("gd", N), gs("gs", N);
  lo::dense_curvature_matrix Hd("Hd", N, N);
  lo::sparse_curvature_matrix Hs;
  const double fd = dense.observed_value_gradient_curvature(point(x), gd, Hd);
  const double fs = sparse.observed_value_gradient_curvature(point(x), gs, Hs);
  // two assemblies: on the GPU the order of the atomic sums moves the last bits, so the bounds are relative
  CHECK(std::abs(fd - fs) <= 1e-14 * std::abs(fd));
  double eg = 0.0, eh = 0.0, sg = 0.0, sh = 0.0;
  for (int j = 0; j < N; ++j) eg = std::max(eg, std::abs(gd(j) - gs(j))), sg = std::max(sg, std::abs(gd(j)));
  for (int c = 0; c < N; ++c)
    for (int r = c; r < N; ++r) {
      int q = -1;
      for (int k = Hs.col_ptr[c]; k < Hs.col_ptr[c + 1]; ++k)
        if (Hs.row_idx[k] == r) q = k;
      eh = std::max(eh, std::abs(Hd(r, c) - (q >= 0 ? Hs.values[q] : 0.0))), sh = std::max(sh, std::abs(Hd(r, c)));
    }
  CHECK(eg <= 1e-13 * sg);
  CHECK(eh <= 1e-13 * sh);
  const std::vector<double> x0(N, 0.0);
  const auto rd = le::fit_inner(dense, point(x0), tight());
  const auto rs = le::fit_inner(sparse, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(rd.optimizer.status));
  REQUIRE(lo::optimizer_succeeded(rs.optimizer.status));
  for (int j = 0; j < N; ++j) CHECK(std::abs(rd.optimizer.point(j) - rs.optimizer.point(j)) <= 1e-7);
}

TEST_CASE("engine inner fit: held coordinates keep their value") {
  const auto s = gauss_spec(1000, 20, 22);
  const auto m = le::build_row_model(s, 2);
  std::vector<double> x0(m.n_coordinates(), 0.0);
  x0[2] = -0.25, x0[3] = 0.5;
  le::row_model_objective<engine_distribs, lo::sparse_curvature_matrix> obj(
      m, group_prior(s, 0.5), {}, le::row_curvature::observed_else_expected, {2, 3});
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
      m, pf.penalties, {}, le::row_curvature::observed_else_expected, pf.held);
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
      m, pf.penalties, {}, le::row_curvature::observed_else_expected, pf.held);
  const auto r = le::fit_inner(obj, point(pf.x0), tight());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  for (std::size_t j = 0; j < ref.size(); ++j)
    CHECK(std::abs(r.optimizer.point(j) - ref[j]) <= 1e-9 * (1.0 + std::abs(ref[j])));
}

TEST_CASE("engine inner fit: Student t, dense curvature, observed and expected information reach one point") {
  const auto s = student_spec(3000, 24);
  const auto m = le::build_row_model(s, 3);
  const std::vector<double> x0 = {0.0, 0.0, 0.0, 0.0, std::log(10.0)};
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> obs(m, {}, {}, le::row_curvature::observed);
  le::row_model_objective<engine_distribs, lo::dense_curvature_matrix> exp(m, {}, {}, le::row_curvature::expected);
  const auto ro = le::fit_inner(obs, point(x0), tight());
  const auto re = le::fit_inner(exp, point(x0), tight());
  REQUIRE(lo::optimizer_succeeded(ro.optimizer.status));
  REQUIRE(lo::optimizer_succeeded(re.optimizer.status));
  for (int j = 0; j < 5; ++j) CHECK(std::abs(ro.optimizer.point(j) - re.optimizer.point(j)) <= 1e-5);
  CHECK(gradient_norm(obs, ro.optimizer.point) <= 1e-4);
  CHECK(std::exp(ro.optimizer.point(4)) == doctest::Approx(5.0).epsilon(0.4));
}

namespace {

lo::proximal_newton_optimizer<> tight_proximal() {
  lo::proximal_newton_optimizer<> pn;
  pn.stop_rule.tolerance = 1e-14;
  return pn;
}

}  // namespace

// A lasso on the six coefficients of the location of a Gaussian whose scale is modelled: laplace1 of scale
// s on each (one replicate per coefficient, separable, so the subproblem runs coordinate descent). The
// optimality conditions at the fit: g_j + sign(b_j) / s = 0 where b_j != 0, |g_j| <= 1 / s where b_j = 0,
// g the gradient of the smooth part; g = 0 on the unpenalized coordinates.
TEST_CASE("engine inner fit: a lasso on the location, proximal Newton, against the optimality conditions") {
  const auto s = lasso_spec(2000, 41);
  const auto m = le::build_row_model(s, 2);
  const double scale = 1.0 / 60.0;
  const std::vector<le::penalty_term> pen = {univariate_prior(laplace::distrib::laplace1_distrib{}, 1, 6, 1, {-1}, {scale})};
  le::row_model_composite_objective<engine_distribs> obj(m, pen);
  CHECK(obj.nonsmooth_block_count() == 6);
  const std::vector<double> x0(9, 0.0);
  const auto r = le::fit_inner(obj, point(x0), tight_proximal());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  lo::optimizer_vector g("g", 9);
  obj.value_gradient(r.optimizer.point, g);
  int zeros = 0;
  for (int j = 0; j < 9; ++j) {
    const double b = r.optimizer.point(j);
    if (j < 1 || j > 6) CHECK(std::abs(g(j)) <= 1e-6);
    else if (b != 0.0) CHECK(std::abs(g(j) + (b > 0.0 ? 1.0 : -1.0) / scale) <= 1e-7 / scale);
    else {
      CHECK(std::abs(g(j)) <= (1.0 + 1e-7) / scale);
      ++zeros;
    }
  }
  MESSAGE("lasso: ", zeros, " coefficients at zero, b1 ", r.optimizer.point(1), ", b4 ", r.optimizer.point(4),
          ", eigenvalue-corrected iterations ", r.optimizer.eigen_modified_iterations);
  CHECK(zeros >= 1);  // at least one coefficient on its kink, where the subgradient condition applies
  CHECK(r.optimizer.point(1) != 0.0);
  CHECK(r.optimizer.point(4) != 0.0);
}

// A group lasso on the same six coefficients: mvpowerexp1 over sigma^2 I with shape 1/2, the penalty
// ||b|| / (2 sigma) plus its constant (not separable: one block, FISTA on the subproblem). At the fit,
// with b != 0, g + b / (2 sigma ||b||) = 0 on the group.
TEST_CASE("engine inner fit: a group lasso on the location, proximal Newton, against the optimality conditions") {
  const auto s = lasso_spec(2000, 42);
  const auto m = le::build_row_model(s, 2);
  using pe = laplace::distrib::mvpowerexp1_distrib<laplace::pdmat::scaled_identity_pdmat>;
  laplace::pdmat::scaled_identity_pdmat I;
  I.dim = 6;
  const double log_sigma = std::log(0.01);
  const auto pen = le::make_penalty_term(laplace::penalty::distrib_penalty<pe>{pe(I)}, {{1, 2, 3, 4, 5, 6}}, {-1, -1},
                                         {log_sigma, 0.5});
  le::row_model_composite_objective<engine_distribs> obj(m, {pen});
  CHECK(obj.nonsmooth_block_count() == 1);
  CHECK(!obj.nonsmooth_block_separable(0));
  const std::vector<double> x0(9, 0.0);
  const auto r = le::fit_inner(obj, point(x0), tight_proximal());
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  lo::optimizer_vector g("g", 9);
  obj.value_gradient(r.optimizer.point, g);
  double nb = 0.0;
  for (int j = 1; j <= 6; ++j) nb += r.optimizer.point(j) * r.optimizer.point(j);
  nb = std::sqrt(nb);
  REQUIRE(nb > 0.0);
  const double c = 1.0 / (2.0 * std::exp(log_sigma) * nb);
  double e = 0.0;
  for (int j = 0; j < 9; ++j) e = std::max(e, std::abs(g(j) + (j >= 1 && j <= 6 ? c * r.optimizer.point(j) : 0.0)));
  MESSAGE("group lasso: ||b|| ", nb, ", stationarity ", e, " in ", r.optimizer.iterations, " iterations");
  CHECK(e <= 1e-6 / (2.0 * std::exp(log_sigma)));  // relative to the size of the subgradient, 1/(2 sigma)
}
