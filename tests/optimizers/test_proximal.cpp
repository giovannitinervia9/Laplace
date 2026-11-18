// The proximal methods (note/optimizers.md, section 12): penalized_objective,
// proximal_gradient_optimizer (FISTA and MFISTA), proximal_newton_optimizer
// and prox_mapping_stop_rule, on the regression test objective
// (regression_objective.hpp), whose sums run in a Kokkos kernel on the CPU or
// the GPU build.
//
// - the lasso (laplace2 rate on the slopes, intercept unpenalized) and the
//   elastic net (enet): FISTA, proximal Newton with coordinate descent and
//   proximal Newton with FISTA inside agree on the point and on the exact
//   support, and the point satisfies the optimality conditions read through
//   the prox and coordinate by coordinate (KKT);
// - MCP, a weakly convex penalty written here (rho = 1/gamma): with rho
//   declared, MFISTA and proximal Newton reach a stationary point, and proximal
//   Newton damps the coordinates whose curvature is below rho and says so;
//   with rho wrongly declared 0 no run claims convergence at a point that is not
//   stationary;
// - stop rules: gradient_stop_rule is rejected by both methods at compile time.

#include "regression_objective.hpp"
#include "wave2_reference.hpp"

#include <doctest/doctest.h>

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace proximal_test {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
using regression_test::make_regression_problem;
using regression_test::regression_loss;
using regression_test::regression_problem;
using objective_type = regression_test::regression_test_objective<>;

template <class D>
using uam = ld::univariate_as_multivariate<D>;
using lasso_penalty = lpen::distrib_penalty<ld::independent_product_distrib<uam<ld::laplace2_distrib>>>;
using enet_penalty = lpen::distrib_penalty<ld::independent_product_distrib<uam<ld::enet_distrib>>>;

static_assert(!lo::stop_rule_for<lo::gradient_stop_rule, lo::detail::proximal_gradient_provides>);
static_assert(!lo::stop_rule_for<lo::gradient_stop_rule, lo::detail::proximal_newton_provides>);
static_assert(lo::stop_rule_for<lo::prox_mapping_stop_rule, lo::detail::proximal_gradient_provides>);

lasso_penalty make_lasso(int d) {
  return lasso_penalty(ld::independent_product_distrib<uam<ld::laplace2_distrib>>(
      ld::independent_product_component<uam<ld::laplace2_distrib>>{{}, d, 1, nullptr}));
}
enet_penalty make_enet(int d) {
  return enet_penalty(ld::independent_product_distrib<uam<ld::enet_distrib>>(
      ld::independent_product_component<uam<ld::enet_distrib>>{{}, d, 1, nullptr}));
}

std::vector<double> to_vector(const lo::optimizer_point& x) {
  std::vector<double> v(x.extent(0));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = x(i);
  return v;
}

// max_i |G_t(x)_i| = max_i |x_i - prox_{t r}(x - t grad s(x))_i| / t, for a composite objective
template <class O>
double prox_mapping_sup(O& objective, const lo::optimizer_point& x, double t) {
  const int n = static_cast<int>(x.extent(0));
  lo::optimizer_vector g("g", n), v("v", n), z("z", n);
  objective.value_gradient(x, g);
  for (int i = 0; i < n; ++i) v(i) = x(i) - t * g(i);
  objective.nonsmooth_prox(v, t, z);
  double m = 0.0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(x(i) - z(i)) / t);
  return m;
}

// KKT of s(beta) + sum_{j >= 1} (a |beta_j| + c beta_j^2 / 2), intercept free: the
// largest violation, in gradient units
double kkt_violation(objective_type& smooth, const lo::optimizer_point& x, double a, double c) {
  const int n = static_cast<int>(x.extent(0));
  lo::optimizer_vector g("g", n);
  smooth.value_gradient(x, g);
  double worst = std::abs(g(0));
  for (int j = 1; j < n; ++j) {
    const double gj = g(j) + c * x(j);
    worst = std::max(worst, x(j) == 0.0 ? std::max(0.0, std::abs(gj) - a) : std::abs(gj + a * (x(j) > 0 ? 1.0 : -1.0)));
  }
  return worst;
}

int support_size(const std::vector<double>& b) {
  int k = 0;
  for (std::size_t j = 1; j < b.size(); ++j) k += b[j] != 0.0;
  return k;
}

// one problem, three runs, the checks; a and c are the kink weight and the ridge weight of the penalty
template <class P>
void check_three_routes(const regression_problem& pr, P penalty, std::vector<double> theta, double a, double c,
                        double fista_point_tolerance = 1e-6) {
  objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
  lo::penalized_objective<objective_type, P> objective(smooth, pr.p, lo::attach_penalty(penalty, 1, std::move(theta)));
  lo::optimizer_vector x0("x0", pr.p);  // zeros

  lo::proximal_gradient_optimizer<> fista;
  fista.stop_rule.tolerance = 1e-14;
  lo::proximal_newton_optimizer<> newton_cd;
  newton_cd.stop_rule.tolerance = 1e-14;
  auto newton_fista = newton_cd;
  newton_fista.inner_solver = lo::proximal_newton_inner_solver::proximal_gradient;

  const auto r1 = lo::minimize(fista, objective, x0);
  const auto r2 = lo::minimize(newton_cd, objective, x0);
  const auto r3 = lo::minimize(newton_fista, objective, x0);
  MESSAGE("FISTA ", r1.iterations, " iterations ", r1.value_evaluations, " values; proximal Newton (CD) ", r2.iterations,
          " iterations; (FISTA inside) ", r3.iterations, " iterations");
  CHECK(lo::optimizer_succeeded(r1.status));
  CHECK(lo::optimizer_succeeded(r2.status));
  CHECK(lo::optimizer_succeeded(r3.status));
  CHECK(r2.damped_coordinates == 0);

  const auto b1 = to_vector(r1.point), b2 = to_vector(r2.point), b3 = to_vector(r3.point);
  double scale = 0.0;
  for (double b : b2) scale = std::max(scale, std::abs(b));
  for (std::size_t j = 0; j < b1.size(); ++j) {
    CHECK(std::abs(b1[j] - b2[j]) <= fista_point_tolerance * scale);
    CHECK(std::abs(b3[j] - b2[j]) <= 1e-6 * scale);
    CHECK((b1[j] == 0.0) == (b2[j] == 0.0));
    CHECK((b3[j] == 0.0) == (b2[j] == 0.0));
  }
  const int k = support_size(b2);
  MESSAGE("support ", k, " of ", pr.p - 1, " slopes");
  CHECK(k > 0);
  CHECK(k < pr.p - 1);
  // optimality: KKT in gradient units (the gradient is of order n), and through the prox
  const double tol = 1e-5 * a;
  for (const auto* r : {&r1, &r2, &r3}) {
    CHECK(kkt_violation(smooth, r->point, a, c) <= tol);
    CHECK(prox_mapping_sup(objective, r->point, 1e-3) <= tol);
  }
}

// MCP with weight lambda and concavity gamma on every coordinate but the
// intercept, rho = 1/gamma, with its firm-threshold prox (exact for t < gamma)
struct mcp_objective {
  using curvature_type = lo::dense_curvature_matrix;
  objective_type& smooth;
  double lambda, gamma, declared_rho;

  double value(const lo::optimizer_point& x) { return smooth.value(x); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) { return smooth.value_gradient(x, g); }
  double value_gradient_curvature(const lo::optimizer_point& x, const lo::optimizer_vector& g, curvature_type& H) {
    return smooth.value_gradient_curvature(x, g, H);
  }
  double pen(double b) const {
    const double u = std::abs(b);
    return u <= gamma * lambda ? lambda * u - u * u / (2.0 * gamma) : 0.5 * gamma * lambda * lambda;
  }
  double prox1(double v, double t) const {
    const double u = std::abs(v);
    if (u <= t * lambda) return 0.0;
    if (u <= gamma * lambda) return std::copysign((u - t * lambda) / (1.0 - t / gamma), v);
    return v;
  }
  double nonsmooth_value(const lo::optimizer_point& x) {
    double r = 0.0;
    for (std::size_t j = 1; j < x.extent(0); ++j) r += pen(x(j));
    return r;
  }
  void nonsmooth_prox(const lo::optimizer_point& v, double t, const lo::optimizer_vector& z) {
    z(0) = v(0);
    for (std::size_t j = 1; j < v.extent(0); ++j) z(j) = prox1(v(j), t);
  }
  int nonsmooth_block_count() const { return 1; }
  double weak_convexity_constant(int) const { return declared_rho; }
  std::pair<int, int> nonsmooth_block_range(int) const { return {1, smooth.columns() - 1}; }
  bool nonsmooth_block_separable(int) const { return true; }
  double nonsmooth_coordinate_prox(int, int, double v, double t) const { return prox1(v, t); }
};
static_assert(lo::coordinate_composite_objective<mcp_objective>);

// stationarity of MCP: 0 in the Clarke subdifferential, coordinate by coordinate
double mcp_violation(objective_type& smooth, const lo::optimizer_point& x, double lambda, double gamma) {
  const int n = static_cast<int>(x.extent(0));
  lo::optimizer_vector g("g", n);
  smooth.value_gradient(x, g);
  double worst = std::abs(g(0));
  for (int j = 1; j < n; ++j) {
    const double u = std::abs(x(j));
    double e;
    if (u == 0.0)
      e = std::max(0.0, std::abs(g(j)) - lambda);
    else if (u <= gamma * lambda)
      e = std::abs(g(j) + std::copysign(lambda - u / gamma, x(j)));
    else
      e = std::abs(g(j));
    worst = std::max(worst, e);
  }
  return worst;
}

}  // namespace proximal_test

TEST_CASE("proximal methods: the lasso by FISTA and proximal Newton, point, support, optimality") {
  using namespace proximal_test;
  const auto pr = make_regression_problem(500, 21, 6, 1.0, 20261118);
  // lambda |beta_j| with lambda = 60 (glmnet's lambda = 0.12 at n = 500)
  check_three_routes(pr, make_lasso(pr.p - 1), {60.0}, 60.0, 0.0);
}

TEST_CASE("proximal methods: the lasso on equicorrelated columns (correlation 0.99)") {
  using namespace proximal_test;
  const auto pr = make_regression_problem(300, 41, 5, 1.0, 11, false, 0.99);
  // with (1/2) t |G_t|^2, t = 1/L, FISTA's point was 6.5e-6 from proximal Newton's
  // (the measure was smaller than f - f* by up to L / mu); with gamma_max, 9.2e-8
  check_three_routes(pr, make_lasso(pr.p - 1), {10.0}, 10.0, 0.0);
}

TEST_CASE("proximal methods: the elastic net by FISTA and proximal Newton") {
  using namespace proximal_test;
  const auto pr = make_regression_problem(400, 31, 8, 1.0, 7);
  // enet with lambda = 80, alpha = 0.6: a = 48 |beta|, c = 32 beta^2 / 2
  check_three_routes(pr, make_enet(pr.p - 1), {80.0, 0.6}, 48.0, 32.0);
}

TEST_CASE("proximal methods: the penalized objective forwards value, prox and rho") {
  using namespace proximal_test;
  const auto pr = make_regression_problem(50, 5, 2, 1.0, 3);
  objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
  lo::penalized_objective<objective_type, lasso_penalty> objective(smooth, pr.p,
                                                                   lo::attach_penalty(make_lasso(3), 2, {1.5}));
  static_assert(lo::coordinate_composite_objective<decltype(objective)>);
  static_assert(lo::curvature_objective<decltype(objective)>);
  lo::optimizer_vector x("x", 5), z("z", 5);
  const double xs[5] = {0.3, -2.0, 0.5, -0.1, 4.0};
  for (int i = 0; i < 5; ++i) x(i) = xs[i];
  // value: sum over coordinates 2..4 of 1.5 |b| - log(1.5 / 2)
  CHECK(objective.nonsmooth_value(x) == doctest::Approx(1.5 * (0.5 + 0.1 + 4.0) - 3.0 * std::log(0.75)).epsilon(1e-14));
  objective.nonsmooth_prox(x, 0.2, z);  // threshold 0.3 on coordinates 2..4, identity elsewhere
  const double expected[5] = {0.3, -2.0, 0.2, 0.0, 3.7};
  for (int i = 0; i < 5; ++i) CHECK(z(i) == doctest::Approx(expected[i]).epsilon(1e-15));
  CHECK(objective.nonsmooth_coordinate_prox(0, 2, 4.0, 0.2) == doctest::Approx(3.7).epsilon(1e-15));
  CHECK(objective.weak_convexity_constant(0) == 0.0);
  CHECK(objective.nonsmooth_block_separable(0));
  CHECK(objective.nonsmooth_block_range(0) == std::pair<int, int>{2, 3});
  CHECK_THROWS(lo::penalized_objective<objective_type, lasso_penalty>(smooth, pr.p, lo::attach_penalty(make_lasso(3), 3, {1.5})));
  CHECK_THROWS(lo::penalized_objective<objective_type, lasso_penalty>(smooth, pr.p, lo::attach_penalty(make_lasso(3), 2, {1.5, 2.0})));
  // the laplace loss of the test objective: sum |y - X beta|
  objective_type lad(pr.x, pr.y, pr.n, pr.p, regression_loss::laplace);
  double s = 0.0;
  for (int i = 0; i < pr.n; ++i) {
    double eta = 0.0;
    for (int j = 0; j < pr.p; ++j) eta += pr.x[std::size_t(i + pr.n * j)] * xs[j];
    s += std::abs(pr.y[std::size_t(i)] - eta);
  }
  CHECK(lad.value(x) == doctest::Approx(s).epsilon(1e-13));
}

TEST_CASE("proximal methods: MCP, rho declared and wrongly declared") {
  using namespace proximal_test;
  // sigma = 20 at n = 200: H_jj = sum x_ij^2 / 400, about 0.5, below rho = 1/gamma = 2/3 on many slopes
  const auto pr = make_regression_problem(200, 11, 4, 20.0, 99);
  objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian, 20.0);
  const double lambda = 0.15, gamma = 1.5;
  lo::optimizer_vector x0("x0", pr.p);

  mcp_objective right{smooth, lambda, gamma, 1.0 / gamma};
  lo::proximal_gradient_optimizer<> mfista;
  mfista.stop_rule.tolerance = 1e-14;
  const auto r1 = lo::minimize(mfista, right, x0);
  lo::proximal_newton_optimizer<> newton;
  newton.stop_rule.tolerance = 1e-14;
  const auto r2 = lo::minimize(newton, right, x0);
  MESSAGE("MCP: MFISTA ", r1.iterations, " iterations, status ", int(r1.status), "; proximal Newton ", r2.iterations,
          " iterations, status ", int(r2.status), ", ", r2.damped_coordinates, " damped updates");
  CHECK(lo::optimizer_succeeded(r1.status));
  CHECK(mcp_violation(smooth, r1.point, lambda, gamma) <= 1e-5);
  CHECK(r2.damped_coordinates > 0);
  if (lo::optimizer_succeeded(r2.status)) CHECK(mcp_violation(smooth, r2.point, lambda, gamma) <= 1e-5);
  // H has no Cholesky factor problem here (X'X is positive definite): proximal Newton must converge
  CHECK(lo::optimizer_succeeded(r2.status));

  // rho declared 0: the steps are no longer bounded by gamma, the firm threshold is
  // not the prox; a run may fail, but never claims a non-stationary point
  mcp_objective wrong{smooth, lambda, gamma, 0.0};
  const auto w1 = lo::minimize(mfista, wrong, x0);
  const auto w2 = lo::minimize(newton, wrong, x0);
  MESSAGE("MCP with rho declared 0: FISTA status ", int(w1.status), " after ", w1.iterations, "; proximal Newton status ",
          int(w2.status), " after ", w2.iterations);
  for (const auto* r : {&w1, &w2})
    if (lo::optimizer_succeeded(r->status)) CHECK(mcp_violation(smooth, r->point, lambda, gamma) <= 1e-5);
}

namespace proximal_test {

// one glmnet reference fit by one method; returns the report, writes the largest
// coefficient error relative to max |b| of the reference and whether the support matches
template <class P, class Optimizer>
lo::optimizer_report fit_against_glmnet(const wave2_reference::dataset& d, const wave2_reference::glmnet_fit& ref,
                                        objective_type& smooth, const P& penalty, std::vector<double> theta,
                                        const Optimizer& optimizer, double& coefficient_error, bool& same_support,
                                        double& objective_error) {
  const int p = d.p + 1;
  lo::penalized_objective<objective_type, P> objective(smooth, p, lo::attach_penalty(penalty, 1, std::move(theta)));
  lo::optimizer_vector x0("x0", p);
  const auto r = lo::minimize(optimizer, objective, x0);
  double scale = 0.0, err = 0.0;
  for (int j = 0; j < p; ++j) scale = std::max(scale, std::abs(ref.coefficients[std::size_t(j)]));
  for (int j = 0; j < p; ++j) err = std::max(err, std::abs(r.point(j) - ref.coefficients[std::size_t(j)]));
  coefficient_error = err / scale;
  std::vector<int> support;
  for (int j = 1; j < p; ++j)
    if (r.point(j) != 0.0) support.push_back(j);
  same_support = support == ref.support;
  const double reference_value = ref.objective_laplace + d.p * ref.penalty_constant_per_coef;
  objective_error = r.value - reference_value;  // f - f*, in the objective's units
  return r;
}

}  // namespace proximal_test

TEST_CASE("proximal methods: lasso and elastic net against the frozen glmnet references") {
  using namespace proximal_test;
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  const auto fits = wave2_reference::read_glmnet_fits(dir);
  REQUIRE(fits.size() == 30);
  std::string loaded;
  wave2_reference::dataset d;
  for (const auto& ref : fits) {
    if (ref.dataset != loaded) d = wave2_reference::load_dataset(dir, ref.dataset), loaded = ref.dataset;
    REQUIRE(ref.coefficients.size() == std::size_t(d.p) + 1);
    // the design with an intercept column first
    std::vector<double> x(std::size_t(d.n) * std::size_t(d.p + 1), 1.0);
    std::copy(d.x.begin(), d.x.end(), x.begin() + d.n);
    objective_type smooth(x, d.y_gaussian, d.n, d.p + 1, regression_loss::gaussian);
    lo::proximal_newton_optimizer<> newton;
    newton.stop_rule.tolerance = 1e-10;  // 1e-12 is about 30 eps |f| on d2: Armijo stalls on rounding first
    lo::proximal_gradient_optimizer<> fista;
    // within a few eps |f| of the resolution of f FISTA crawls (d5, n = 1e5, f = 5.8e4:
    // 86148 iterations at 1e-10), so its tolerance keeps clear of it
    const double f_reference = ref.objective_laplace + d.p * ref.penalty_constant_per_coef;
    fista.stop_rule.tolerance = std::max(1e-10, 1e3 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    double e1, e2, o1, o2;
    bool s1, s2;
    lo::optimizer_report r1, r2;
    if (ref.alpha == 1.0) {
      const auto pen = make_lasso(d.p);
      r1 = fit_against_glmnet(d, ref, smooth, pen, {ref.a}, newton, e1, s1, o1);
      r2 = fit_against_glmnet(d, ref, smooth, pen, {ref.a}, fista, e2, s2, o2);
    } else {
      const auto pen = make_enet(d.p);
      r1 = fit_against_glmnet(d, ref, smooth, pen, {ref.lambda_laplace, ref.alpha}, newton, e1, s1, o1);
      r2 = fit_against_glmnet(d, ref, smooth, pen, {ref.lambda_laplace, ref.alpha}, fista, e2, s2, o2);
    }
    MESSAGE(ref.dataset, " alpha ", ref.alpha, " fraction ", ref.lambda_fraction, " support ", ref.support.size(),
            " | proximal Newton status ", int(r1.status), " it ", r1.iterations, " err ", e1, " f-f* ", o1, " supp ", s1,
            " | FISTA status ", int(r2.status), " it ", r2.iterations, " err ", e2, " f-f* ", o2, " measure ", r2.decrement,
            " supp ", s2);
    CHECK(s1);
    CHECK(s2);
    // d2 has p = 200 > n = 100: X'X has no Cholesky factor, and the decrement
    // comes from the factor of its block on the free coordinates (the support and
    // the intercept). It exists unless the support has n or more coordinates (the
    // enet at the smallest lambda, 113): there proximal Newton cannot declare
    // convergence, since the enet's ridge curvature is in r, not in H
    const bool free_singular = ref.support.size() + 1 > std::size_t(d.n);
    CHECK(lo::optimizer_succeeded(r1.status) == !free_singular);
    CHECK(lo::optimizer_succeeded(r2.status));
    // at the decrement tolerance 1e-10: f - f* at most 1.8e-10 and coefficients
    // within 2.4e-6 on d2 (measured), within 2e-7 elsewhere
    CHECK(e1 <= (d.p >= d.n ? 5e-6 : 5e-7));
    if (!free_singular) CHECK(std::abs(o1) <= 4e-10);
    // FISTA's measure (1/2) max(t, gamma_max) |G_t|^2 against f - f*: at most 1.11
    // times the measure over the 30 fits (measured), checked with a margin
    CHECK(e2 <= 2e-5);
    CHECK(o2 <= 1.5 * r2.decrement + 64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
  }
}
