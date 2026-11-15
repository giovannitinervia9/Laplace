// The foundations of laplace::optimizers (include/laplace/optimizers/):
// concepts, stopping rules, line searches, the dense curvature solver with
// the SPD solve and the condition estimate of laplace::dense, minimize(), and
// the derivatives of the battery of test problems (test_problems.hpp)
// against central differences. Host code only: no Kokkos kernel.

#include <doctest/doctest.h>

#include <laplace/dense/dense.hpp>
#include <laplace/optimizers/dense_curvature_solver.hpp>
#include <laplace/optimizers/line_search.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/objective.hpp>
#include <laplace/optimizers/report.hpp>
#include <laplace/optimizers/stop_rules.hpp>

#include "test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace ld = laplace::dense;
namespace tp = optimizer_test_problems;
using lo::optimizer_point;
using lo::optimizer_vector;
using matrix = lo::dense_curvature_matrix;

optimizer_vector make_vector(const std::vector<double>& v) {
  optimizer_vector x("x", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

double max_abs(const optimizer_point& v) {
  double m = 0.0;
  for (std::size_t i = 0; i < v.extent(0); ++i) m = std::max(m, std::abs(v(i)));
  return m;
}

// ---- objectives of one variable for the line searches ----

// f = 50 x^2
struct quadratic_1d {
  double value(const optimizer_point& x) { return 50.0 * x(0) * x(0); }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) {
    g(0) = 100.0 * x(0);
    return value(x);
  }
};

// f = x - log x, defined for x > 0 only (NaN or inf elsewhere), minimum at 1
struct x_minus_log_x {
  double value(const optimizer_point& x) { return x(0) - std::log(x(0)); }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) {
    g(0) = 1.0 - 1.0 / x(0);
    return value(x);
  }
};

// f = 1 everywhere, with a false gradient of 1: no step can decrease it
struct flat_1d {
  double value(const optimizer_point&) { return 1.0; }
};

// a value-only objective: not a gradient_objective
struct value_only {
  double value(const optimizer_point& x) { return x(0) * x(0); }
};

static_assert(lo::value_objective<value_only>);
static_assert(!lo::gradient_objective<value_only>);
static_assert(lo::gradient_objective<quadratic_1d>);
static_assert(!lo::curvature_objective<quadratic_1d>);
static_assert(lo::curvature_objective<tp::extended_rosenbrock_problem>);
static_assert(lo::curvature_objective<tp::wood_problem>);
static_assert(lo::curvature_objective<tp::ill_conditioned_quadratic_problem>);
static_assert(!lo::composite_objective<tp::wood_problem>);
static_assert(std::is_same_v<lo::curvature_solver_for<matrix>::type, lo::dense_curvature_solver>);

// a composite objective (wave 2), only to check the concept: s = |x|^2 / 2, r = |x|_1
struct lasso_like {
  double value(const optimizer_point& x) { return 0.5 * x(0) * x(0); }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) {
    g(0) = x(0);
    return value(x);
  }
  double nonsmooth_value(const optimizer_point& x) { return std::abs(x(0)); }
  void nonsmooth_prox(const optimizer_point& v, double t, const optimizer_vector& z) {
    z(0) = std::copysign(std::max(std::abs(v(0)) - t, 0.0), v(0));
  }
  int nonsmooth_block_count() { return 1; }
  double weak_convexity_constant(int) { return 0.0; }
};
static_assert(lo::composite_objective<lasso_like>);

// stopping rules: needs and compile-time compatibility
constexpr unsigned gradient_method = lo::stop_needs_gradient | lo::stop_needs_decrement | lo::stop_needs_previous;
constexpr unsigned value_method = lo::stop_needs_previous;
static_assert(lo::stop_rule_for<lo::decrement_stop_rule, gradient_method>);
static_assert(lo::stop_rule_for<lo::parameter_change_stop_rule, value_method>);
static_assert(!lo::stop_rule_for<lo::gradient_stop_rule, value_method>);
static_assert(!lo::stop_rule_for<lo::decrement_stop_rule, value_method>);
static_assert(!lo::stop_rule_for<lo::any_of_stop_rule<lo::parameter_change_stop_rule, lo::gradient_stop_rule>,
                                 value_method>);
static_assert(lo::any_of_stop_rule<lo::objective_change_stop_rule, lo::gradient_stop_rule>::needs ==
              (lo::stop_needs_previous | lo::stop_needs_gradient));

// ---- a minimal optimizer, only to exercise minimize(), the report and the rules ----

// Steepest descent with Armijo; decrement (1/2) s g' g, the decrease predicted
// by the next direction with B = s I, s the last accepted step. ((1/2) |g' dx|
// with dx the step just taken is near zero after any near-exact line search,
// wherever the point: it declared this run converged 2.04 away from x_star.)
template <class Rule>
  requires lo::stop_rule_for<Rule, gradient_method>
struct test_steepest_descent_optimizer {
  Rule stop_rule;
  lo::armijo_line_search line_search;
  int max_iterations = 10000;

  template <lo::gradient_objective O>
  lo::optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    lo::optimizer_report report;
    optimizer_vector x("x", n), g("g", n), d("d", n), x_trial("x_trial", n), x_previous("x_previous", n);
    Kokkos::deep_copy(x, x0);
    double f = objective.value_gradient(x, g);
    ++report.value_evaluations, ++report.gradient_evaluations;
    if (!std::isfinite(f)) return report;
    report.status = lo::optimizer_status::iteration_budget;
    for (int k = 1; k <= max_iterations; ++k) {
      for (int i = 0; i < n; ++i) d(i) = -g(i);
      const lo::line_search_result s = line_search.search(objective, x, f, g, d, 1.0, x_trial);
      report.value_evaluations += s.value_evaluations;
      if (!s.accepted()) {
        report.status = s.outcome == lo::line_search_outcome::step_below_rounding ? lo::optimizer_status::stalled
                                                                                   : lo::optimizer_status::line_search_failed;
        break;
      }
      Kokkos::deep_copy(x_previous, x);
      Kokkos::deep_copy(x, x_trial);
      const double f_previous = f;
      f = objective.value_gradient(x, g);
      ++report.value_evaluations, ++report.gradient_evaluations;
      double gg = 0.0;
      for (int i = 0; i < n; ++i) gg += g(i) * g(i);
      report.iterations = k;
      report.decrement = 0.5 * s.step * gg;
      lo::stop_rule_input in;
      in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
      in.has_previous = true, in.previous_value = f_previous, in.previous_point = x_previous;
      const lo::stop_verdict v = stop_rule.verdict(in);
      if (v != lo::stop_verdict::none) {
        report.status = v == lo::stop_verdict::converged                 ? lo::optimizer_status::converged
                        : v == lo::stop_verdict::converged_at_resolution ? lo::optimizer_status::converged_at_resolution
                                                                         : lo::optimizer_status::stopped_on_change;
        break;
      }
    }
    report.point = x, report.value = f, report.gradient = g;
    report.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return report;
  }
};

static_assert(lo::optimizer_for<test_steepest_descent_optimizer<lo::decrement_stop_rule>, quadratic_1d>);
static_assert(!lo::optimizer_for<test_steepest_descent_optimizer<lo::decrement_stop_rule>, value_only>);

// ---- central differences (5 points) ----

// max_i |g_i - dg_i| / (max_j |g_j| + |f| 1e-3 + 1) of the gradient against
// differences of the value at x
template <class P>
double gradient_difference_error(P& p, const optimizer_vector& x) {
  const int n = static_cast<int>(x.extent(0));
  optimizer_vector g("g", n), xh("xh", n);
  const double f = p.value_gradient(x, g);
  double err = 0.0;
  for (int i = 0; i < n; ++i) {
    const double h = 1e-3 * std::max(1.0, std::abs(x(i)));
    auto at = [&](double t) {
      Kokkos::deep_copy(xh, x);
      xh(i) += t;
      return p.value(xh);
    };
    const double dg = (-at(2 * h) + 8.0 * at(h) - 8.0 * at(-h) + at(-2 * h)) / (12.0 * h);
    err = std::max(err, std::abs(dg - g(i)));
  }
  return err / (max_abs(g) + 1e-3 * std::abs(f) + 1.0);
}

// the same for the Hessian against differences of the gradient
template <class P>
double hessian_difference_error(P& p, const optimizer_vector& x) {
  const int n = static_cast<int>(x.extent(0));
  optimizer_vector g("g", n), gh("gh", n), xh("xh", n);
  matrix H("H", n, n);
  p.value_gradient_curvature(x, g, H);
  double err = 0.0, scale = max_abs(g) + 1.0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) scale = std::max(scale, std::abs(H(i, j)));
  for (int j = 0; j < n; ++j) {
    const double h = 1e-3 * std::max(1.0, std::abs(x(j)));
    std::vector<double> dh(n, 0.0);
    const double w[4] = {-1.0, 8.0, -8.0, 1.0}, t[4] = {2 * h, h, -h, -2 * h};
    for (int s = 0; s < 4; ++s) {
      Kokkos::deep_copy(xh, x);
      xh(j) += t[s];
      p.value_gradient(xh, gh);
      for (int i = 0; i < n; ++i) dh[i] += w[s] * gh(i) / (12.0 * h);
    }
    for (int i = 0; i < n; ++i) err = std::max(err, std::abs(dh[i] - H(i, j)));
    for (int i = 0; i < n; ++i) err = std::max(err, std::abs(H(i, j) - H(j, i)));  // symmetric
  }
  return err / scale;
}

// x0, x0 moved, x_star moved: the points at which the derivatives are checked
std::vector<std::vector<double>> check_points(const std::vector<double>& x0, const std::vector<double>& x_star) {
  std::vector<double> a = x0, b = x_star;
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] += 0.3 + 0.1 * i;
    b[i] += 0.05 * (i % 2 == 0 ? 1.0 : -1.0) * std::max(1.0, std::abs(b[i]));
  }
  return {x0, a, b};
}

template <class P>
void check_problem_derivatives(P& p) {
  for (const auto& v : check_points(p.x0, p.x_star)) {
    const optimizer_vector x = make_vector(v);
    INFO(p.name << " at x_1 = " << v[0]);
    CHECK(gradient_difference_error(p, x) <= 1e-8);
    CHECK(hessian_difference_error(p, x) <= 1e-8);
  }
  // the minimizer and the offset
  const optimizer_vector xs = make_vector(p.x_star);
  CHECK(std::abs(p.value(xs) - p.f_star) <= 1e-12 * std::max(1.0, std::abs(p.f_star)));
}

// ---- the dense solver on SPD matrices with known eigenvalues ----

// A = Q diag(lambda) Q^T with Q orthogonal (the eigenvectors of a
// pseudo-random symmetric matrix) and lambda log-spaced from 1 to kappa; also
// the exact inverse Q diag(1 / lambda) Q^T.
struct spd_case {
  matrix a, inverse;
  double kappa2;
};

spd_case make_spd_case(int n, double kappa, std::uint64_t seed) {
  tp::test_uniform u{seed};
  matrix s("s", n, n), q("q", n, n);
  Kokkos::View<double*, Kokkos::HostSpace> w("w", n);
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) s(i, j) = s(j, i) = u() - 0.5;
  REQUIRE(ld::symmetric_eigen_decomposition(s, w, q).success);
  spd_case c{matrix("a", n, n), matrix("inverse", n, n), kappa};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double sa = 0.0, si = 0.0;
      for (int k = 0; k < n; ++k) {
        const double lambda = n > 1 ? std::pow(kappa, double(k) / (n - 1)) : 1.0;
        sa += q(i, k) * lambda * q(j, k);
        si += q(i, k) / lambda * q(j, k);
      }
      c.a(i, j) = sa, c.inverse(i, j) = si;
    }
  return c;
}

double one_norm(const matrix& a) {
  double m = 0.0;
  for (std::size_t j = 0; j < a.extent(1); ++j) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.extent(0); ++i) s += std::abs(a(i, j));
    m = std::max(m, s);
  }
  return m;
}

}  // namespace

// ============================== stopping rules ==============================

TEST_CASE("optimizers: stopping rules") {
  const optimizer_vector x = make_vector({1.0, 2.0}), xp = make_vector({1.0, 2.0 + 1e-11}),
                         g = make_vector({1e-7, -3e-7});
  lo::stop_rule_input in;
  in.iteration = 3, in.value = 1e6, in.point = x, in.gradient = g;

  SUBCASE("decrement: tolerance, resolution, NaN") {
    CHECK(lo::decrement_stop_rule{}.tolerance == 1e-6);  // the default
    lo::decrement_stop_rule rule{1e-14};
    in.decrement = 1e-15;
    CHECK(rule.verdict(in) == lo::stop_verdict::converged);
    in.decrement = 1e-12;  // above 1e-14, below eps |f| = 2.2e-10
    CHECK(rule.verdict(in) == lo::stop_verdict::converged_at_resolution);
    in.decrement = 1e-9;
    CHECK(rule.verdict(in) == lo::stop_verdict::none);
    in.decrement = std::numeric_limits<double>::quiet_NaN();
    CHECK(rule.verdict(in) == lo::stop_verdict::none);
    in.value = 0.0, in.decrement = 1e-12;  // no resolution at f = 0
    CHECK(rule.verdict(in) == lo::stop_verdict::none);
  }
  SUBCASE("gradient: absolute, relative, scale, missing") {
    CHECK(lo::gradient_stop_rule{3e-7}.verdict(in) == lo::stop_verdict::converged);
    CHECK(lo::gradient_stop_rule{2e-7}.verdict(in) == lo::stop_verdict::none);
    CHECK(lo::gradient_stop_rule{0.0, 1e-12}.verdict(in) == lo::stop_verdict::converged);  // 1e-12 |f| = 1e-6
    CHECK(lo::gradient_stop_rule{0.0, 1e-14}.verdict(in) == lo::stop_verdict::none);       // 1e-8
    CHECK(lo::gradient_stop_rule{1e-9, 0.0, 1e-3}.verdict(in) == lo::stop_verdict::converged);  // per 1000 observations
    CHECK(lo::gradient_stop_rule{1e-10, 0.0, 1e-3}.verdict(in) == lo::stop_verdict::none);
    in.gradient = {};
    CHECK(lo::gradient_stop_rule{1.0}.verdict(in) == lo::stop_verdict::none);
  }
  SUBCASE("changes: only with a previous iterate, never converged") {
    lo::parameter_change_stop_rule par{1e-10};
    lo::objective_change_stop_rule obj{1e-3};
    CHECK(par.verdict(in) == lo::stop_verdict::none);
    CHECK(obj.verdict(in) == lo::stop_verdict::none);
    in.has_previous = true, in.previous_point = xp, in.previous_value = 1e6 + 1e-4;
    CHECK(par.verdict(in) == lo::stop_verdict::stopped_on_change);
    CHECK(obj.verdict(in) == lo::stop_verdict::stopped_on_change);
    CHECK(lo::parameter_change_stop_rule{1e-12}.verdict(in) == lo::stop_verdict::none);
    CHECK(lo::parameter_change_stop_rule{0.0, 1e-11}.verdict(in) == lo::stop_verdict::stopped_on_change);
    CHECK(lo::objective_change_stop_rule{1e-5}.verdict(in) == lo::stop_verdict::none);
    CHECK(lo::objective_change_stop_rule{0.0, 1e-11}.verdict(in) == lo::stop_verdict::none);  // 1e-5
    CHECK(lo::objective_change_stop_rule{0.0, 2e-10}.verdict(in) == lo::stop_verdict::stopped_on_change);  // 2e-4
  }
  SUBCASE("any_of and all_of") {
    in.has_previous = true, in.previous_point = xp, in.previous_value = 1e6;
    in.decrement = 1.0;
    const auto change_or_decrement = lo::any_of(lo::parameter_change_stop_rule{1e-10}, lo::decrement_stop_rule{1e-8});
    CHECK(change_or_decrement.verdict(in) == lo::stop_verdict::stopped_on_change);
    in.decrement = 1e-9;
    CHECK(change_or_decrement.verdict(in) == lo::stop_verdict::converged);  // the strongest
    const auto both = lo::all_of(lo::parameter_change_stop_rule{1e-10}, lo::decrement_stop_rule{1e-8});
    CHECK(both.verdict(in) == lo::stop_verdict::stopped_on_change);  // the weakest
    in.decrement = 1.0;
    CHECK(both.verdict(in) == lo::stop_verdict::none);
    CHECK(lo::any_of().verdict(in) == lo::stop_verdict::none);
  }
}

// ============================== line searches ==============================

TEST_CASE("optimizers: Armijo line search") {
  optimizer_vector x_trial("x_trial", 1);
  SUBCASE("quadratic interpolation finds the minimizer of a quadratic in one backtrack") {
    quadratic_1d f;
    const optimizer_vector x = make_vector({1.0}), g = make_vector({100.0}), d = make_vector({-10.0});
    const auto r = lo::armijo_line_search{}.search(f, x, 50.0, g, d, 1.0, x_trial);
    CHECK(r.accepted());
    CHECK(r.step == doctest::Approx(0.1).epsilon(1e-14));
    CHECK(r.value_evaluations == 2);
    CHECK(std::abs(x_trial(0)) <= 1e-14);
  }
  SUBCASE("non-finite trial points are refused") {
    x_minus_log_x f;
    const optimizer_vector x = make_vector({3.0}), d = make_vector({-10.0}), g("g", 1);
    const double fx = f.value_gradient(x, g);
    const auto r = lo::armijo_line_search{}.search(f, x, fx, g, d, 1.0, x_trial);
    CHECK(r.accepted());
    CHECK(r.refused_evaluations == 2);  // x = -7 and x = -2
    CHECK(r.value_evaluations == 3);
    CHECK(r.step == 0.25);
    CHECK(r.value <= fx + 1e-4 * r.step * (g(0) * d(0)));
  }
  SUBCASE("a step below rounding is a failure") {
    flat_1d f;
    const optimizer_vector x = make_vector({1.0}), g = make_vector({1.0}), d = make_vector({-1.0});
    lo::armijo_line_search ls;
    ls.max_evaluations = 1000;
    const auto r = ls.search(f, x, 1.0, g, d, 1.0, x_trial);
    CHECK(r.outcome == lo::line_search_outcome::step_below_rounding);
    CHECK(r.value_evaluations < 60);
    ls.max_evaluations = 10;
    CHECK(ls.search(f, x, 1.0, g, d, 1.0, x_trial).outcome == lo::line_search_outcome::evaluation_budget);
  }
  SUBCASE("not a descent direction; the resolution test") {
    quadratic_1d f;
    const optimizer_vector x = make_vector({1.0}), g = make_vector({100.0});
    CHECK(lo::armijo_line_search{}.search(f, x, 50.0, g, make_vector({1.0}), 1.0, x_trial).outcome ==
          lo::line_search_outcome::not_descent_direction);
    lo::armijo_line_search ls;
    ls.resolution = 1.0;
    const auto r = ls.search(f, x, 50.0, g, make_vector({-1e-3}), 1.0, x_trial);  // predicts 0.1 < 1
    CHECK(r.outcome == lo::line_search_outcome::below_resolution);
    CHECK(r.value_evaluations == 0);
  }
  SUBCASE("sufficient decrease on the battery along steepest descent and Newton directions") {
    tp::extended_rosenbrock_problem rosenbrock(2);
    tp::wood_problem wood;
    tp::ill_conditioned_quadratic_problem quadratic(10, 1e8);
    auto check = [&](auto& p) {
      const int n = p.size();
      const optimizer_vector x = make_vector(p.x0);
      optimizer_vector g("g", n), d("d", n), xt("xt", n);
      matrix H("H", n, n);
      const double f = p.value_gradient_curvature(x, g, H);
      for (int i = 0; i < n; ++i) d(i) = -g(i);
      lo::armijo_line_search ls;
      for (int pass = 0; pass < 2; ++pass) {
        const auto r = ls.search(p, x, f, g, d, 1.0, xt);
        INFO(p.name << std::string(pass == 0 ? ", steepest descent" : ", Newton") << ": step " << r.step << ", "
                    << r.value_evaluations << " evaluations");
        CHECK(r.accepted());
        CHECK(p.value(xt) == r.value);
        CHECK(r.value <= f + ls.sufficient_decrease * r.step * lo::detail::dot(g, d));
        CHECK(r.value_evaluations <= 20);
        lo::dense_curvature_solver solver(n);  // the second pass along the Newton direction
        if (solver.factorize(H) != lo::curvature_factorization::positive_definite) break;
        solver.newton_direction(g, d);
      }
    };
    check(rosenbrock);
    check(wood);
    check(quadratic);
  }
}

TEST_CASE("optimizers: strong Wolfe line search") {
  lo::wolfe_line_search ls;
  auto wolfe_holds = [&](const lo::line_search_result& r, double f, double slope) {
    return r.accepted() && r.value <= f + ls.sufficient_decrease * r.step * slope &&
           std::abs(r.slope) <= -ls.curvature * slope;
  };
  SUBCASE("expansion from a short initial step") {
    quadratic_1d f;
    optimizer_vector xt("xt", 1), gt("gt", 1);
    const optimizer_vector x = make_vector({1.0}), g = make_vector({100.0}), d = make_vector({-1.0});
    const auto r = ls.search(f, x, 50.0, g, d, 1e-4, xt, gt);
    INFO("step " << r.step << ", " << r.value_evaluations << " evaluations");
    CHECK(wolfe_holds(r, 50.0, -100.0));
    CHECK(gt(0) == doctest::Approx(100.0 * xt(0)));
    CHECK(r.value_evaluations <= 10);
  }
  SUBCASE("refused points shorten the step") {
    x_minus_log_x f;
    optimizer_vector xt("xt", 1), gt("gt", 1), g("g", 1);
    const optimizer_vector x = make_vector({3.0}), d = make_vector({-10.0});
    const double fx = f.value_gradient(x, g);
    const auto r = ls.search(f, x, fx, g, d, 1.0, xt, gt);
    INFO("step " << r.step << ", " << r.value_evaluations << " evaluations, " << r.refused_evaluations << " refused");
    CHECK(wolfe_holds(r, fx, g(0) * d(0)));
    CHECK(r.refused_evaluations >= 1);
  }
  SUBCASE("a step below rounding is a failure, not an acceptance") {
    quadratic_1d f;
    optimizer_vector xt("xt", 1), gt("gt", 1);
    const optimizer_vector x = make_vector({1.0}), g = make_vector({100.0}), d = make_vector({-1e-300});
    const auto r = ls.search(f, x, 50.0, g, d, 1.0, xt, gt);
    CHECK(r.outcome == lo::line_search_outcome::step_below_rounding);
  }
  SUBCASE("strong Wolfe on the battery along steepest descent") {
    tp::extended_rosenbrock_problem rosenbrock(2), rosenbrock_large(100);
    tp::beale_problem beale;
    tp::helical_valley_problem helical;
    tp::ill_conditioned_quadratic_problem quadratic(10, 1e8);
    auto check = [&](auto& p, double step0) {
      const int n = p.size();
      const optimizer_vector x = make_vector(p.x0);
      optimizer_vector g("g", n), d("d", n), xt("xt", n), gt("gt", n);
      const double f = p.value_gradient(x, g);
      for (int i = 0; i < n; ++i) d(i) = -g(i);
      const auto r = ls.search(p, x, f, g, d, step0, xt, gt);
      INFO(p.name << ": step " << r.step << ", " << r.value_evaluations << " evaluations");
      CHECK(wolfe_holds(r, f, lo::detail::dot(g, d)));
      CHECK(r.value_evaluations <= 20);
      optimizer_vector g2("g2", n);
      CHECK(p.value_gradient(xt, g2) == r.value);  // the trial point and gradient are those accepted
      for (int i = 0; i < n; ++i) CHECK(g2(i) == gt(i));
    };
    check(rosenbrock, 1.0);
    check(rosenbrock_large, 1.0);
    check(beale, 1.0);
    check(helical, 1.0);
    check(quadratic, 1.0);
    check(quadratic, 1e-12);
  }
}

// ============================== dense solver ==============================

TEST_CASE("optimizers: SPD solve and condition estimate of laplace::dense") {
  double worst_solve = 0.0, low = 1e300, high = 0.0;
  for (int n : {1, 2, 5, 10, 50, 200})
    for (double kappa : {1e2, 1e6, 1e12}) {
      const spd_case c = make_spd_case(n, kappa, 1000u + n);
      matrix L("L", n, n), b("b", n, 1);
      std::vector<double> x(n);
      for (int i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * i;
      for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += c.a(i, j) * x[j];
        b(i, 0) = s;
      }
      REQUIRE(ld::spd_solve(c.a, L, b).success);
      // forward error against kappa_2 eps
      double err = 0.0, xn = 0.0;
      for (int i = 0; i < n; ++i) err = std::max(err, std::abs(b(i, 0) - x[i])), xn = std::max(xn, std::abs(x[i]));
      const double scaled = err / (xn * (n > 1 ? kappa : 1.0) * std::numeric_limits<double>::epsilon());
      worst_solve = std::max(worst_solve, scaled);
      // the estimate of kappa_1 against the exact one and against kappa_2
      const double exact = one_norm(c.a) * one_norm(c.inverse);
      const double estimate = ld::cholesky_condition_estimate(c.a, L);
      const double k2 = n > 1 ? kappa : 1.0;
      INFO("n = " << n << ", kappa = " << kappa << ": estimate / exact kappa_1 = " << estimate / exact);
      CHECK(estimate <= exact * (1.0 + 1e3 * k2 * std::numeric_limits<double>::epsilon()));
      CHECK(estimate >= exact / 3.0);
      CHECK(estimate >= k2 / (3.0 * n));
      CHECK(estimate <= n * k2 * (1.0 + 1e-6));
      low = std::min(low, estimate / exact), high = std::max(high, estimate / exact);
    }
  MESSAGE("SPD solve: worst forward error " << worst_solve << " kappa_2 eps; condition estimate / exact kappa_1 in ["
                                            << low << ", " << high << "]");
  CHECK(worst_solve <= 0.5);  // measured 0.336 (gcc and CUDA)

  matrix A("A", 2, 2), L("L", 2, 2), b("b", 2, 1);
  A(0, 0) = 1.0, A(1, 0) = A(0, 1) = 2.0, A(1, 1) = 1.0;  // eigenvalues 3, -1
  CHECK(ld::spd_solve(A, L, b).reason == ld::dense_failure::not_positive_definite);
  A(1, 0) = std::numeric_limits<double>::quiet_NaN();
  CHECK(std::isnan(ld::cholesky_condition_estimate(A, L)));
  L(1, 1) = 0.0, A(1, 0) = 0.5;
  CHECK(std::isinf(ld::cholesky_condition_estimate(A, L)));
  CHECK(ld::cholesky_solve(L, b).reason == ld::dense_failure::singular);
}

TEST_CASE("optimizers: dense curvature solver") {
  const int n = 3;
  lo::dense_curvature_solver solver(n);
  matrix H("H", n, n);
  const optimizer_vector g = make_vector({1.0, -2.0, 0.5});
  optimizer_vector d("d", n);
  CHECK(solver.last_factorization() == lo::curvature_factorization::not_evaluated);
  CHECK(!solver.newton_direction(g, d));
  // positive definite: H d = -g
  H(0, 0) = 4.0, H(1, 1) = 3.0, H(2, 2) = 2.0, H(1, 0) = H(0, 1) = 1.0, H(2, 1) = H(1, 2) = 0.5;
  CHECK(solver.factorize(H) == lo::curvature_factorization::positive_definite);
  REQUIRE(solver.newton_direction(g, d));
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) s += H(i, j) * d(j);
    CHECK(s == doctest::Approx(-g(i)).epsilon(1e-14));
  }
  const double k = solver.condition_estimate();
  CHECK(k >= 1.0);
  CHECK(k <= 10.0);
  // indefinite: refused; shifted by lambda > 1 it factorizes, (H + lambda I) d = -g
  H(2, 2) = -1.0;
  CHECK(solver.factorize(H) == lo::curvature_factorization::not_positive_definite);
  CHECK(!solver.newton_direction(g, d));
  CHECK(std::isinf(solver.condition_estimate()));
  const double lambda = 2.0;
  CHECK(solver.factorize_shifted(H, lambda) == lo::curvature_factorization::positive_definite);
  REQUIRE(solver.newton_direction(g, d));
  for (int i = 0; i < n; ++i) {
    double s = lambda * d(i);
    for (int j = 0; j < n; ++j) s += H(i, j) * d(j);
    CHECK(s == doctest::Approx(-g(i)).epsilon(1e-14));
  }
  CHECK(H(2, 2) == -1.0);  // H itself untouched
  H(1, 0) = std::numeric_limits<double>::infinity();
  CHECK(solver.factorize(H) == lo::curvature_factorization::non_finite);
}

// ============================== problems and minimize ==============================

TEST_CASE("optimizers: derivatives of the test problems against central differences") {
  for (double offset : {0.0, 1.0, 1e3, 1e6}) {
    tp::extended_rosenbrock_problem rosenbrock(2, offset), rosenbrock_large(10, offset);
    tp::powell_singular_problem powell(offset);
    tp::beale_problem beale(offset);
    tp::brown_badly_scaled_problem brown(offset);
    tp::helical_valley_problem helical(offset);
    tp::wood_problem wood(offset);
    tp::saddle_problem saddle(offset);
    tp::exploded_cosh_problem cosh_problem(3, offset);
    check_problem_derivatives(rosenbrock);
    check_problem_derivatives(rosenbrock_large);
    check_problem_derivatives(powell);
    check_problem_derivatives(beale);
    check_problem_derivatives(brown);
    check_problem_derivatives(helical);
    check_problem_derivatives(wood);
    check_problem_derivatives(saddle);
    for (double kappa : {1e2, 1e6, 1e12}) {
      tp::ill_conditioned_quadratic_problem quadratic(6, kappa, offset);
      check_problem_derivatives(quadratic);
    }
    // the exploded start: finite value, gradient and curvature; derivatives near 0
    const optimizer_vector x0 = make_vector(cosh_problem.x0);
    CHECK(std::isfinite(cosh_problem.value(x0)));
    CHECK(gradient_difference_error(cosh_problem, make_vector({0.3, -0.2, 0.1})) <= 1e-8);
    CHECK(hessian_difference_error(cosh_problem, make_vector({0.3, -0.2, 0.1})) <= 1e-8);
  }
  // the large Rosenbrock evaluates at n = 1000 (dense H)
  tp::extended_rosenbrock_problem large(1000);
  optimizer_vector g("g", 1000);
  matrix H("H", 1000, 1000);
  CHECK(large.value_gradient_curvature(make_vector(large.x0), g, H) == doctest::Approx(500 * 24.2));
  // the noisy objective: the value at the start is noise of amplitude 100
  tp::noisy_problem noisy(2);
  const double f0 = noisy.value(make_vector(noisy.x0));
  CHECK(std::abs(f0 - 2.0) <= 100.0);
}

TEST_CASE("optimizers: minimize with a minimal steepest descent") {
  tp::ill_conditioned_quadratic_problem quadratic(4, 10.0);
  const optimizer_vector x0 = make_vector(quadratic.x0);
  SUBCASE("converged on the decrement") {
    test_steepest_descent_optimizer<lo::decrement_stop_rule> sd{{1e-12}};
    const lo::optimizer_report r = lo::minimize(sd, quadratic, x0);
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(r.decrement <= 1e-12);
    CHECK(r.iterations > 0);
    CHECK(r.value_evaluations > r.gradient_evaluations);
    for (int i = 0; i < quadratic.size(); ++i) CHECK(std::abs(r.point(i) - quadratic.x_star[i]) <= 1e-4);
  }
  SUBCASE("a change rule stops on change, never converged") {
    test_steepest_descent_optimizer<lo::objective_change_stop_rule> sd{{1e-1}};
    const lo::optimizer_report r = lo::minimize(sd, quadratic, x0);
    CHECK(r.status == lo::optimizer_status::stopped_on_change);
    CHECK(!lo::optimizer_succeeded(r.status));
    CHECK(r.decrement > 0.0);  // reported for the caller to judge
  }
  SUBCASE("start not admissible") {
    x_minus_log_x f;
    test_steepest_descent_optimizer<lo::decrement_stop_rule> sd;
    const lo::optimizer_report r = lo::minimize(sd, f, make_vector({-1.0}));
    CHECK(r.status == lo::optimizer_status::start_not_admissible);
  }
}
