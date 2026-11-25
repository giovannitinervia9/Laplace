// The Laplace marginal criterion of the engine (engine/marginal_criterion.hpp)
// on the models of engine_models.hpp, away from its optimum:
//   - its exact gradient against central differences of its value, and its
//     exact Hessian against central differences of the exact gradient, over
//     log hyperparameters of penalties and coordinates estimated on the
//     criterion (each evaluation runs a tight inner fit from the last one);
//   - so the passes of order three and four (engine/outer_rows.hpp) and the
//     chain rule to order four through a link, a logistic curve fed by a
//     link and a distrib of three parameters are checked through it.
// Host code: the kernels are launched by the library's functions.

#include <doctest/doctest.h>

#include <laplace/engine/marginal_criterion.hpp>

#include "engine_models.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;
using namespace engine_test;

lo::newton_optimizer<> tight() {
  lo::newton_optimizer<> nt;
  nt.stop_rule.tolerance = 1e-15;
  return nt;
}

lo::optimizer_vector vec(const std::vector<double>& v) {
  lo::optimizer_vector x("u", v.size());
  for (std::size_t j = 0; j < v.size(); ++j) x(j) = v[j];
  return x;
}

struct criterion_check {
  double grad_err = 0.0, grad_scale = 0.0, hess_err = 0.0, hess_scale = 0.0, value = 0.0;
};

criterion_check check_criterion(const le::row_model_spec& s, int P, std::vector<le::quadratic_penalty_term> pen,
                                std::vector<int> psi, std::vector<double> x0, const std::vector<double>& u0,
                                double h) {
  const auto m = le::build_row_model(s, P, 7);
  le::laplace_marginal_criterion<engine_distribs> crit(m, pen, psi, {}, x0, tight());
  const int A = crit.n_outer();
  REQUIRE(A == static_cast<int>(u0.size()));
  lo::optimizer_vector g("g", A);
  lo::dense_curvature_matrix H("H", A, A);
  criterion_check c;
  c.value = crit.value_gradient_curvature(vec(u0), g, H);
  REQUIRE(std::isfinite(c.value));
  std::vector<double> g0(g.data(), g.data() + A), H0(A * A);
  for (int a = 0; a < A; ++a)
    for (int b = 0; b < A; ++b) H0[a + A * b] = H(a, b);
  const std::vector<double> xhat0 = crit.inner_point();
  // central differences at steps h and h / 2, extrapolated (Richardson): error O(h^4)
  for (int a = 0; a < A; ++a) {
    double dv[2];
    std::vector<double> dg[2];
    for (int k = 0; k < 2; ++k) {
      const double hk = k == 0 ? h : 0.5 * h;
      std::vector<double> up = u0, um = u0;
      up[a] += hk, um[a] -= hk;
      // every evaluation starts its inner fit at the inner minimum of u0: from the last point, the
      // inner residual (about the square of the distance) is not a smooth function of u
      le::laplace_marginal_criterion<engine_distribs> cp(m, pen, psi, {}, xhat0, tight()), cm(m, pen, psi, {}, xhat0, tight());
      const double vp = cp.value_gradient(vec(up), g);
      std::vector<double> gp(g.data(), g.data() + A);
      const double vm = cm.value_gradient(vec(um), g);
      dv[k] = (vp - vm) / (2.0 * hk);
      dg[k].resize(A);
      for (int b = 0; b < A; ++b) dg[k][b] = (gp[b] - g(b)) / (2.0 * hk);
    }
    c.grad_err = std::max(c.grad_err, std::abs((4.0 * dv[1] - dv[0]) / 3.0 - g0[a]));
    c.grad_scale = std::max(c.grad_scale, std::abs(g0[a]));
    for (int b = 0; b < A; ++b) {
      c.hess_err = std::max(c.hess_err, std::abs((4.0 * dg[1][b] - dg[0][b]) / 3.0 - H0[b + A * a]));
      c.hess_scale = std::max(c.hess_scale, std::abs(H0[b + A * a]));
    }
  }
  MESSAGE("V ", c.value, ": gradient error ", c.grad_err, " of ", c.grad_scale, ", Hessian error ", c.hess_err, " of ",
          c.hess_scale);
  return c;
}

}  // namespace

TEST_CASE("engine marginal criterion: Gaussian with a group prior, log lambda and the scale's coefficients") {
  const auto s = gauss_spec(400, 10, 31);
  std::vector<le::quadratic_penalty_term> pen = {{s.n_global, s.n_groups, 1, {1.0}, 0}};
  std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  const auto c = check_criterion(s, 2, pen, {2, 3}, x0, {1.0, -0.2, 0.6}, 1e-3);
  CHECK(c.grad_err <= 1e-8 * (1.0 + c.grad_scale));
  CHECK(c.hess_err <= 1e-8 * (1.0 + c.hess_scale));
}

TEST_CASE("engine marginal criterion: gamma with a logistic curve fed by a link, log lambda and log phi") {
  const auto s = gamma_spec(400, 10, 32);
  std::vector<le::quadratic_penalty_term> pen = {{s.n_global, s.n_groups, 1, {1.0}, 0}};
  std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  x0[0] = 2.7, x0[1] = 4.6, x0[2] = std::log(1.2);
  const auto c = check_criterion(s, 2, pen, {3}, x0, {2.0, -1.8}, 1e-3);
  CHECK(c.grad_err <= 1e-8 * (1.0 + c.grad_scale));
  CHECK(c.hess_err <= 1e-8 * (1.0 + c.hess_scale));
}

TEST_CASE("engine marginal criterion: Student t, separable, log nu on the criterion") {
  const auto s = student_spec(300, 33);
  std::vector<double> x0 = {0.9, 0.4, -0.2, 0.6, 0.0};
  const auto c = check_criterion(s, 3, {}, {4}, x0, {std::log(4.0)}, 1e-3);
  CHECK(c.grad_err <= 1e-8 * (1.0 + c.grad_scale));
  CHECK(c.hess_err <= 1e-8 * (1.0 + c.hess_scale));
}

namespace {

struct reml_reference {
  std::vector<double> u;  // the prototype's optimum, in the engine's outer vector
  double criterion;       // the prototype's REML criterion there, maximized
  double constant;        // what the prototype leaves out: the engine's V is constant - criterion
};

// the criterion at the prototype's optimum, then a complete fit from a start against that optimum
void check_reml_fit(const engine_test::prototype_fit& pf, int P, const reml_reference& ref,
                    const std::vector<double>& u0) {
  const auto m = le::build_row_model(pf.spec, P);
  const int A = static_cast<int>(ref.u.size());
  const double V_ref = ref.constant - ref.criterion;
  {
    le::laplace_marginal_criterion<engine_distribs> crit(m, pf.penalties, pf.held, {}, pf.x0, tight());
    lo::optimizer_vector g("g", A);
    const double V = crit.value_gradient(vec(ref.u), g);
    double gmax = 0.0;
    for (int a = 0; a < A; ++a) gmax = std::max(gmax, std::abs(g(a)));
    MESSAGE("at the prototype's optimum: V - V_ref ", V - V_ref, ", max |gradient| ", gmax);
    CHECK(std::abs(V - V_ref) <= 1e-10 * std::abs(V_ref));
    CHECK(gmax <= 1e-5);
  }
  le::laplace_marginal_criterion<engine_distribs> crit(m, pf.penalties, pf.held, {}, pf.x0, tight());
  lo::newton_optimizer<> outer;
  outer.stop_rule.tolerance = 1e-12;
  const auto r = le::fit_marginal(crit, vec(u0), outer);
  REQUIRE(lo::optimizer_succeeded(r.optimizer.status));
  double du = 0.0;
  for (int a = 0; a < A; ++a) du = std::max(du, std::abs(r.optimizer.point(a) - ref.u[a]));
  MESSAGE("complete fit: V - V_ref ", r.optimizer.value - V_ref, " in ", r.optimizer.iterations,
          " outer iterations, ", r.counts.inner_fits, " inner fits, max |u - u_ref| ", du);
  CHECK(r.optimizer.value <= V_ref + 1e-10 * std::abs(V_ref));
  CHECK(du <= 1e-5);
}

}  // namespace

// The Gaussian model of prototipi/kokkos_fit/lk.cpp (kokkos_fit_model in engine_models.hpp) on its first
// 2000 rows; the reference is lk.cpp's REML fit (outer vector log tau, log lambda, the two coefficients
// of log sigma; the engine's log lambda of the group prior is -2 log tau). lk.cpp's log-density leaves
// out -log(2 pi) / 2 per row.
TEST_CASE("engine marginal criterion: the Gaussian model of the kokkos_fit prototype, against its REML fit") {
  auto pf = kokkos_fit_model(LAPLACE_REFERENCE_DIR "/../engine/kokkos_fit_2000.csv", 0.0, 0.0, 0.0, 0.0);
  pf.penalties[0].hyperparameter = 0;  // the matrices at rho = 0 are the bare ones
  pf.penalties[1].hyperparameter = 1;
  const reml_reference ref{{2.0 * 0.91756492601312334, -4.8811529834528038, -0.98602388660648632, 0.27445795123482364},
                           883.02825304724297, 1000.0 * std::log(2.0 * 3.141592653589793)};
  check_reml_fit(pf, 2, ref, {0.0, 0.0, 0.0, 0.0});
}

// Model C of prototipi/node_dispatch/hy4_main.cpp (lk3_gamma_model): the reference is its REML fit by the
// engine of lk3.cpp (outer vector log tau and the coefficient of log phi), whose outer gradient there is
// 2e-6 and 4e-6; the start is that fit's, log tau = log 15, c0 = -2.
TEST_CASE("engine marginal criterion: the gamma model with a logistic curve of the lk3 prototype, against its REML fit") {
  auto pf = lk3_gamma_model(LAPLACE_REFERENCE_DIR "/../engine/lk3_gamma_2000.csv", 0.0, -2.0);
  pf.penalties[0].hyperparameter = 0;
  const reml_reference ref{{-2.0 * 3.6755854596987816, -2.947691401829033}, -9180.7283168838385, 0.0};
  check_reml_fit(pf, 2, ref, {-2.0 * std::log(15.0), -2.0});
}

