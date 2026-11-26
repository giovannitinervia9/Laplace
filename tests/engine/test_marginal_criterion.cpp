// The Laplace marginal criterion of the engine (engine/marginal_criterion.hpp)
// on the models of engine_models.hpp, away from its optimum:
//   - its exact gradient against central differences of its value, and its
//     exact Hessian against central differences of the exact gradient, over
//     hyperparameters of penalties (through their bijectors) and coordinates
//     estimated on the criterion (each evaluation runs a tight inner fit from
//     the inner minimum at the centre);
//   - so the passes of order three and four (engine/outer_rows.hpp) and the
//     chain rule to order four through a link, a logistic curve fed by a
//     link and a distrib of three parameters are checked through it;
//   - and the penalties' own share: priors that are not Gaussian (Student t,
//     pseudo-Huber), whose third and fourth derivatives in beta and mixed
//     derivatives in (beta, theta) enter, a hyperparameter shared by two
//     penalties, a covariance over the effects of two predictors, and the
//     covariance and precision forms of one prior;
//   - on every model of check_criterion, the selected inverse (latent_inverse_route::selected)
//     against the dense one, and the automatic route (selected for a gradient, dense for a
//     Hessian, reported as such): the same value, gradient and Hessian up to rounding.
// Host code: the kernels are launched by the library's functions.

#include <doctest/doctest.h>

#include <laplace/bijector/log_cholesky.hpp>
#include <laplace/distrib/pseudohuber.hpp>
#include <laplace/engine/marginal_criterion.hpp>
#include <laplace/pdmat/unstructured.hpp>

#include "engine_models.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
namespace lpen = laplace::penalty;
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

le::penalty_hyperparameter log_scale() { return {log_link()}; }
le::penalty_hyperparameter free_scale() { return {}; }

struct criterion_check {
  double grad_err = 0.0, grad_scale = 0.0, hess_err = 0.0, hess_scale = 0.0, value = 0.0;
  double route_value_err = 0.0, route_grad_err = 0.0, route_hess_err = 0.0;  // selected against dense
  std::vector<double> gradient;
};

criterion_check check_criterion(const le::row_model_spec& s, int P, const std::vector<le::penalty_term>& pen,
                                const std::vector<le::penalty_hyperparameter>& hyper, std::vector<int> psi,
                                std::vector<double> x0, const std::vector<double>& u0, double h) {
  const auto m = le::build_row_model(s, P, 7);
  le::laplace_marginal_criterion<engine_distribs> crit(m, pen, hyper, psi, {}, x0, tight());
  const int A = crit.n_outer();
  REQUIRE(A == static_cast<int>(u0.size()));
  lo::optimizer_vector g("g", A);
  lo::dense_curvature_matrix H("H", A, A);
  criterion_check c;
  c.value = crit.value_gradient_curvature(vec(u0), g, H);
  REQUIRE(std::isfinite(c.value));
  std::vector<double> g0(g.data(), g.data() + A), H0(A * A);
  c.gradient = g0;
  for (int a = 0; a < A; ++a)
    for (int b = 0; b < A; ++b) H0[a + A * b] = H(a, b);
  const std::vector<double> xhat0 = crit.inner_point();
  // the other routes: the same inner fits, so the derivatives differ by rounding only. Selected once with
  // the Hessian (the last reader of the dense inverse) and once with the gradient only; automatic with
  // the gradient, then the Hessian; dense explicitly
  {
    le::laplace_marginal_criterion<engine_distribs> cs(m, pen, hyper, psi, {}, x0, tight()),
        cg(m, pen, hyper, psi, {}, x0, tight());
    cs.set_latent_inverse_route(le::latent_inverse_route::selected);
    cg.set_latent_inverse_route(le::latent_inverse_route::selected);
    lo::dense_curvature_matrix Hs("Hs", A, A);
    const double vs = cs.value_gradient_curvature(vec(u0), g, Hs);
    CHECK(crit.counts().latent_inverse_used == le::latent_inverse_route::dense);
    CHECK(cs.counts().latent_inverse_used == le::latent_inverse_route::selected);
    c.route_value_err = std::abs(vs - c.value);
    for (int a = 0; a < A; ++a) {
      c.route_grad_err = std::max(c.route_grad_err, std::abs(g(a) - g0[a]));
      for (int b = 0; b < A; ++b) c.route_hess_err = std::max(c.route_hess_err, std::abs(Hs(a, b) - H0[a + A * b]));
    }
    (void)cg.value_gradient(vec(u0), g);
    CHECK(cg.counts().latent_inverse_used == le::latent_inverse_route::selected);
    for (int a = 0; a < A; ++a) c.route_grad_err = std::max(c.route_grad_err, std::abs(g(a) - g0[a]));
    // automatic (crit above, the default): selected for a gradient, then dense for the Hessian at the
    // same point; and the explicit dense route against crit
    le::laplace_marginal_criterion<engine_distribs> ca(m, pen, hyper, psi, {}, x0, tight()),
        cd(m, pen, hyper, psi, {}, x0, tight());
    cd.set_latent_inverse_route(le::latent_inverse_route::dense);
    (void)ca.value_gradient(vec(u0), g);
    CHECK(ca.counts().latent_inverse_used == le::latent_inverse_route::selected);
    for (int a = 0; a < A; ++a) c.route_grad_err = std::max(c.route_grad_err, std::abs(g(a) - g0[a]));
    for (auto* cr : {&ca, &cd}) {
      const double va = cr->value_gradient_curvature(vec(u0), g, Hs);
      CHECK(cr->counts().latent_inverse_used == le::latent_inverse_route::dense);
      c.route_value_err = std::max(c.route_value_err, std::abs(va - c.value));
      for (int a = 0; a < A; ++a) {
        c.route_grad_err = std::max(c.route_grad_err, std::abs(g(a) - g0[a]));
        for (int b = 0; b < A; ++b)
          c.route_hess_err = std::max(c.route_hess_err, std::abs(Hs(a, b) - H0[a + A * b]));
      }
    }
  }
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
      le::laplace_marginal_criterion<engine_distribs> cp(m, pen, hyper, psi, {}, xhat0, tight()),
          cm(m, pen, hyper, psi, {}, xhat0, tight());
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
          c.hess_scale, "; selected route against dense: value ", c.route_value_err, ", gradient ", c.route_grad_err,
          ", Hessian ", c.route_hess_err);
  return c;
}

void check_bounds(const criterion_check& c) {
  CHECK(c.grad_err <= 1e-8 * (1.0 + c.grad_scale));
  CHECK(c.hess_err <= 1e-8 * (1.0 + c.hess_scale));
  CHECK(c.route_value_err <= 1e-12 * (1.0 + std::abs(c.value)));
  CHECK(c.route_grad_err <= 1e-10 * (1.0 + c.grad_scale));
  CHECK(c.route_hess_err <= 1e-10 * (1.0 + c.hess_scale));
}

}  // namespace

TEST_CASE("engine marginal criterion: Gaussian with a group prior, log tau and the scale's coefficients") {
  const auto s = gauss_spec(400, 10, 31);
  const std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  check_bounds(check_criterion(s, 2, {group_gaussian_prior(s, 0, 0, 1.0)}, {log_scale()}, {2, 3}, x0, {-0.5, -0.2, 0.6}, 1e-3));
}

TEST_CASE("engine marginal criterion: gamma with a logistic curve fed by a link, log tau and log phi") {
  const auto s = gamma_spec(400, 10, 32);
  std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  x0[0] = 2.7, x0[1] = 4.6, x0[2] = std::log(1.2);
  check_bounds(check_criterion(s, 2, {group_gaussian_prior(s, 0, 0, 1.0)}, {log_scale()}, {3}, x0, {-1.0, -1.8}, 1e-3));
}

TEST_CASE("engine marginal criterion: Student t, separable, log nu on the criterion") {
  const auto s = student_spec(300, 33);
  std::vector<double> x0 = {0.9, 0.4, -0.2, 0.6, 0.0};
  check_bounds(check_criterion(s, 3, {}, {}, {4}, x0, {std::log(4.0)}, 1e-3));
}

// Priors that are not Gaussian on the group effects: their derivatives of order three and four in
// the effects enter tau, dH and Q, and the mixed ones in (effect, theta) enter s, E and tau^(a).
TEST_CASE("engine marginal criterion: a Student t prior on the group effects, log sigma and log nu") {
  const auto s = gauss_spec(400, 10, 34);
  const std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  const auto pen = univariate_prior(ld::student_t1_distrib{}, s.n_global, s.n_groups, 1, {0, 1}, {1.0, 4.0});
  check_bounds(check_criterion(s, 2, {pen}, {log_scale(), log_scale()}, {2}, x0, {std::log(0.4), std::log(3.0), -0.2}, 1e-3));
}

TEST_CASE("engine marginal criterion: a pseudo-Huber prior on the group effects, log sigma and log nu") {
  const auto s = gauss_spec(400, 10, 35);
  const std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  const auto pen = univariate_prior(ld::pseudohuber_distrib{}, s.n_global, s.n_groups, 1, {0, 1}, {1.0, 1.0});
  check_bounds(check_criterion(s, 2, {pen}, {log_scale(), log_scale()}, {}, x0, {std::log(0.5), std::log(0.7)}, 1e-3));
}

// One hyperparameter read by two penalties: the intercepts of mu and of log sigma share tau.
TEST_CASE("engine marginal criterion: a hyperparameter shared by two penalties") {
  const auto s = gauss2_spec(500, 12, 36);
  const std::vector<double> x0(s.n_global + 2 * s.n_groups, 0.0);
  const std::vector<le::penalty_term> pen = {group_gaussian_prior(s, 0, 0, 1.0), group_gaussian_prior(s, 1, 0, 1.0)};
  check_bounds(check_criterion(s, 2, pen, {log_scale()}, {2}, x0, {std::log(0.3), -0.2}, 1e-3));
}

// A covariance label: (u_g, v_g), the group effects of two predictors, ~ N(0, Sigma), Sigma unstructured
// through its log-Cholesky parameters (three outer coordinates on the free scale).
TEST_CASE("engine marginal criterion: a covariance over the effects of two predictors") {
  const auto s = gauss2_spec(500, 12, 37);
  const std::vector<double> x0(s.n_global + 2 * s.n_groups, 0.0);
  using chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
  chol S;
  S.dim = 2, S.b = laplace::bijector::log_cholesky_bijector{2};
  using mvg = ld::mvgaussian1_distrib<chol>;
  std::vector<std::vector<int>> reps;
  for (int g = 0; g < s.n_groups; ++g) reps.push_back({s.n_global + 2 * g, s.n_global + 2 * g + 1});
  const auto pen = le::make_penalty_term(lpen::distrib_penalty<mvg>{mvg(S)}, reps, {0, 1, 2});
  check_bounds(check_criterion(s, 2, {pen}, {free_scale(), free_scale(), free_scale()}, {}, x0, {-0.7, 0.1, -1.5}, 1e-3));
}

// Two crossed groupings: the effects of grouping A (u_g, v_g) and of grouping B (a_h) share units, so
// the information holds a block for every pair (g, h) met, outside the arrowhead; the passes of order
// three and four and the inner pattern run on that layout. Priors N(0, tau_u^2), N(0, tau_v^2) and
// N(0, tau_a^2), three hyperparameters, and the coefficient c0 on the criterion.
TEST_CASE("engine marginal criterion: two crossed groupings") {
  const auto s = crossed_spec(600, 8, 6, 39);
  const int G = s.n_groups, H = s.crossed[0].n_groups;
  const std::vector<double> x0(s.n_global + 2 * G + H, 0.0);
  const int a0 = s.n_global + 2 * G;
  const std::vector<le::penalty_term> pen = {
      univariate_prior(ld::gaussian1_distrib{}, s.n_global, G, 2, {0}, {1.0}),
      univariate_prior(ld::gaussian1_distrib{}, s.n_global + 1, G, 2, {1}, {1.0}),
      univariate_prior(ld::gaussian1_distrib{}, a0, H, 1, {2}, {1.0})};
  check_bounds(check_criterion(s, 2, pen, {log_scale(), log_scale(), log_scale()}, {2}, x0,
                               {std::log(0.5), std::log(0.25), std::log(0.4), -0.2}, 1e-3));
}

// The covariance form (mvgaussian1, Sigma = e^(2 theta) I) and the precision form (mvgaussian2,
// Q = e^(2 theta) I) of one prior: the same criterion at opposite theta.
TEST_CASE("engine marginal criterion: covariance and precision forms of a group prior") {
  const auto s = gauss_spec(400, 10, 38);
  const std::vector<double> x0(s.n_global + s.n_groups, 0.0);
  lp::scaled_identity_pdmat I;
  I.dim = 1;
  std::vector<std::vector<int>> reps;
  for (int g = 0; g < s.n_groups; ++g) reps.push_back({s.n_global + g});
  using mv1 = ld::mvgaussian1_distrib<lp::scaled_identity_pdmat>;
  using mv2 = ld::mvgaussian2_distrib<lp::scaled_identity_pdmat>;
  const auto p1 = le::make_penalty_term(lpen::distrib_penalty<mv1>{mv1(I)}, reps, {0});
  const auto p2 = le::make_penalty_term(lpen::distrib_penalty<mv2>{mv2(I)}, reps, {0});
  const auto c1 = check_criterion(s, 2, {p1}, {free_scale()}, {2}, x0, {-0.6, -0.2}, 1e-3);
  const auto c2 = check_criterion(s, 2, {p2}, {free_scale()}, {2}, x0, {0.6, -0.2}, 1e-3);
  check_bounds(c1);
  check_bounds(c2);
  CHECK(std::abs(c1.value - c2.value) <= 1e-10 * std::abs(c1.value));
  CHECK(std::abs(c1.gradient[0] + c2.gradient[0]) <= 1e-8 * (1.0 + std::abs(c1.gradient[0])));
  CHECK(std::abs(c1.gradient[1] - c2.gradient[1]) <= 1e-8 * (1.0 + std::abs(c1.gradient[1])));
}

namespace {

struct reml_reference {
  std::vector<double> u;  // the prototype's optimum, in the engine's outer vector
  double criterion;       // the prototype's REML criterion there, maximized
  double constant;        // what the prototype leaves out: the engine's V is constant - criterion
};

// the criterion at the prototype's optimum, then a complete fit from a start against that optimum
void check_reml_fit(const engine_test::prototype_fit& pf, const std::vector<le::penalty_hyperparameter>& hyper, int P,
                    const reml_reference& ref, const std::vector<double>& u0) {
  const auto m = le::build_row_model(pf.spec, P);
  const int A = static_cast<int>(ref.u.size());
  const double V_ref = ref.constant - ref.criterion;
  {
    le::laplace_marginal_criterion<engine_distribs> crit(m, pf.penalties, hyper, pf.held, {}, pf.x0, tight());
    lo::optimizer_vector g("g", A);
    const double V = crit.value_gradient(vec(ref.u), g);
    double gmax = 0.0;
    for (int a = 0; a < A; ++a) gmax = std::max(gmax, std::abs(g(a)));
    MESSAGE("at the prototype's optimum: V - V_ref ", V - V_ref, ", max |gradient| ", gmax);
    CHECK(std::abs(V - V_ref) <= 1e-10 * std::abs(V_ref));
    CHECK(gmax <= 1e-5);
  }
  le::laplace_marginal_criterion<engine_distribs> crit(m, pf.penalties, hyper, pf.held, {}, pf.x0, tight());
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
// of log sigma). The constant: lk.cpp's log-density leaves out -log(2 pi) / 2 per row (2000 rows); its
// criterion leaves out the normalizing constants of the priors, (G + 8)/2 log(2 pi) (8 the rank of the
// smooth's penalty: with the last column dropped, a straight line is the one direction it leaves
// unpenalized), and the engine's V has -(q/2) log(2 pi) with q = 11 + G integrated coordinates.
TEST_CASE("engine marginal criterion: the Gaussian model of the kokkos_fit prototype, against its REML fit") {
  auto pf = kokkos_fit_model(LAPLACE_REFERENCE_DIR "/../engine/kokkos_fit_2000.csv", 0.0, 0.0, 0.0, 0.0);
  pf.penalties[0].theta_source = {0}, pf.penalties[1].theta_source = {1};
  const double l2pi = std::log(2.0 * std::numbers::pi);
  const reml_reference ref{{-0.91756492601312334, -4.8811529834528038, -0.98602388660648632, 0.27445795123482364},
                           883.02825304724297, 1000.0 * l2pi - 1.5 * l2pi};
  check_reml_fit(pf, {log_scale(), free_scale()}, 2, ref, {0.0, 0.0, 0.0, 0.0});
}

// Model C of prototipi/node_dispatch/hy4_main.cpp (lk3_gamma_model): the reference is its REML fit by the
// engine of lk3.cpp (outer vector log tau and the coefficient of log phi), whose outer gradient there is
// 2e-6 and 4e-6; the start is that fit's, log tau = log 15, c0 = -2. The constant: the prior's
// normalizing constant G/2 log(2 pi), which lk3.cpp leaves out, against -(q/2) log(2 pi), q = 3 + G.
TEST_CASE("engine marginal criterion: the gamma model with a logistic curve of the lk3 prototype, against its REML fit") {
  auto pf = lk3_gamma_model(LAPLACE_REFERENCE_DIR "/../engine/lk3_gamma_2000.csv", 0.0, -2.0);
  pf.penalties[0].theta_source = {0};
  const reml_reference ref{{3.6755854596987816, -2.947691401829033}, -9180.7283168838385,
                           -1.5 * std::log(2.0 * std::numbers::pi)};
  check_reml_fit(pf, {log_scale()}, 2, ref, {std::log(15.0), -2.0});
}
