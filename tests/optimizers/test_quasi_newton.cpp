// BFGS and L-BFGS (include/laplace/optimizers/bfgs.hpp, lbfgs.hpp) on the
// battery of test_problems.hpp and extra_problems.hpp. Host code only.
//
// Bounds: the default stopping rule is the decrement at 1e-6, so f - f* is
// checked against 1e-5 and x - x* against the distances measured with a
// margin (benchmarks/optimizers_quasi_newton.cpp), never against anything
// tighter than what the rule guarantees.

#include <doctest/doctest.h>

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/lbfgs.hpp>
#include <laplace/optimizers/minimize.hpp>

#include "extra_problems.hpp"
#include "test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;

lo::optimizer_vector start_of(const std::vector<double>& v) {
  lo::optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

double distance(const lo::optimizer_report& r, const std::vector<double>& x_star) {
  double d = 0.0;
  for (std::size_t i = 0; i < x_star.size(); ++i) d = std::max(d, std::abs(r.point(i) - x_star[i]));
  return d;
}

static_assert(lo::optimizer_for<lo::bfgs_optimizer<>, tp::wood_problem>);
static_assert(lo::optimizer_for<lo::lbfgs_optimizer<>, tp::abs_sum_problem>);
static_assert(lo::optimizer_for<lo::bfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search>, tp::beale_problem>);
static_assert(lo::optimizer_for<lo::lbfgs_optimizer<lo::gradient_stop_rule>, tp::beale_problem>);

// a minimum reached: succeeded, f - f* <= 1e-5, max |x - x*| <= dx
template <class Optimizer, class Problem>
void check_minimum(const Optimizer& optimizer, Problem p, double dx) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
  INFO(p.name, ", offset ", p.offset, ": status ", int(r.status), ", iterations ", r.iterations, ", f - f* ",
       r.value - p.f_star, ", decrement ", r.decrement);
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK(r.value - p.f_star <= 1e-5);
  CHECK(distance(r, p.x_star) <= dx);
  CHECK(r.decrement <= 1e-6);
  CHECK(r.value_evaluations >= r.iterations + 1);
  CHECK(r.gradient_evaluations >= r.iterations + 1);
  CHECK(r.curvature_evaluations == 0);
}

// a run that may not converge, but must not claim a false minimum
template <class Optimizer, class Problem>
void check_no_false_success(const Optimizer& optimizer, Problem p) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
  INFO(p.name, ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
  if (lo::optimizer_succeeded(r.status)) CHECK(r.value - p.f_star <= 1e-5);
}

template <class Optimizer>
void more_garbow_hillstrom(const Optimizer& optimizer, bool wood) {
  for (double offset : {0.0, 1.0, 1e3, 1e6}) {
    check_minimum(optimizer, tp::extended_rosenbrock_problem(2, offset), 2e-3);
    check_minimum(optimizer, tp::extended_rosenbrock_problem(100, offset), 2e-3);
    check_minimum(optimizer, tp::powell_singular_problem(offset), 5e-2);  // singular at x*: |x - x*| ~ (f - f*)^(1/4)
    check_minimum(optimizer, tp::beale_problem(offset), 2e-3);
    check_minimum(optimizer, tp::brown_badly_scaled_problem(offset), 2e-3);
    check_minimum(optimizer, tp::helical_valley_problem(offset), 2e-3);
    if (wood) check_minimum(optimizer, tp::wood_problem(offset), 2e-3);
  }
}

// none of these may end converged
template <class Optimizer>
void negative_cases(Optimizer optimizer) {
  {
    tp::noisy_problem p(4);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    CHECK_FALSE(lo::optimizer_succeeded(r.status));
  }
  for (int n : {2, 4, 10}) {
    tp::abs_sum_problem p(n);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    INFO("abs_sum n = ", n, ": status ", int(r.status), ", f ", r.value);
    CHECK_FALSE(lo::optimizer_succeeded(r.status));
  }
  optimizer.max_iterations = 100;
  {
    tp::exploded_cosh_problem p(4);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::iteration_budget);
    CHECK(r.iterations == 100);
  }
}

// f = x - log x for x > 0, NaN for x < 0 (refused trial points), minimum at 1
struct x_minus_log_x {
  double value(const lo::optimizer_point& x) { return x(0) > 0.0 ? x(0) - std::log(x(0)) : std::nan(""); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    g(0) = 1.0 - 1.0 / x(0);
    return value(x);
  }
};

// f = 1000 + |x|^2 / 2 with a relative rounding noise of 1024 eps, a
// deterministic function of the bits of x (as a sum reduced in another order
// on a GPU); the gradient x is exact. Near the optimum no trial point of a
// line search shows an acceptable decrease.
struct rounding_noise_at_optimum {
  static double noise(const lo::optimizer_point& x) {  // in [-1, 1), splitmix64 of the bits
    std::uint64_t h = 1;
    for (std::size_t i = 0; i < x.extent(0); ++i) {
      h ^= std::bit_cast<std::uint64_t>(x(i)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h += 0x9e3779b97f4a7c15ULL;
      h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
      h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
      h ^= h >> 31;
    }
    return static_cast<double>(h >> 11) * 0x1.0p-52 - 1.0;
  }
  double value(const lo::optimizer_point& x) {
    double f = 1000.0;
    for (std::size_t i = 0; i < x.extent(0); ++i) f += 0.5 * x(i) * x(i);
    return f * (1.0 + 1024.0 * std::numeric_limits<double>::epsilon() * noise(x));
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    for (std::size_t i = 0; i < x.extent(0); ++i) g(i) = x(i);
    return value(x);
  }
};

}  // namespace

TEST_CASE("optimizers: BFGS on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  more_garbow_hillstrom(lo::bfgs_optimizer<>{}, true);
  more_garbow_hillstrom(lo::bfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search>{}, true);
}

TEST_CASE("optimizers: L-BFGS on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  more_garbow_hillstrom(lo::lbfgs_optimizer<>{}, false);
  // Wood: L-BFGS stops near the saddle at f = 7.877 (a positive definite B
  // cannot see the negative curvature there); either end is a stationary point.
  tp::wood_problem p;
  const lo::optimizer_report r = lo::minimize(lo::lbfgs_optimizer<>{}, p, start_of(p.x0));
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK((r.value - p.f_star <= 1e-5 || std::abs(r.value - 7.877) < 1e-2));
}

TEST_CASE("optimizers: BFGS and L-BFGS on ill-conditioned quadratics") {
  for (double kappa : {1e2, 1e4, 1e6, 1e8}) {
    tp::ill_conditioned_quadratic_problem p(10, kappa);
    check_minimum(lo::bfgs_optimizer<>{}, p, 5e-3);
    if (kappa <= 1e6) check_minimum(lo::lbfgs_optimizer<>{}, p, 5e-3);
  }
  check_minimum(lo::bfgs_optimizer<>{}, tp::ill_conditioned_quadratic_problem(100, 1e6), 5e-3);
  // kappa 1e10 and 1e12: BFGS stalls; it must not claim the minimum. (L-BFGS
  // with 10 pairs at kappa >= 1e8 never sees curvature below about 1e4 and
  // stops 1e-2 above f*: a limit of the method, not tested.)
  for (double kappa : {1e10, 1e12}) check_no_false_success(lo::bfgs_optimizer<>{}, tp::ill_conditioned_quadratic_problem(10, kappa));
}

TEST_CASE("optimizers: BFGS and L-BFGS must not converge on the negative cases") {
  negative_cases(lo::bfgs_optimizer<>{});
  negative_cases(lo::lbfgs_optimizer<>{});
  negative_cases(lo::bfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search>{});
  negative_cases(lo::lbfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search>{});
}

TEST_CASE("optimizers: quasi-Newton start, refused points, budgets, trace") {
  SUBCASE("start at the minimum: converged at iteration 0") {
    tp::ill_conditioned_quadratic_problem p(5, 1e3);
    const lo::optimizer_report r = lo::minimize(lo::bfgs_optimizer<>{}, p, start_of(p.x_star));
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(r.iterations == 0);
    CHECK(r.decrement == 0.0);
    CHECK(r.value_evaluations == 1);
  }
  SUBCASE("start not admissible") {
    x_minus_log_x f;
    const lo::optimizer_report r = lo::minimize(lo::lbfgs_optimizer<>{}, f, start_of({-1.0}));
    CHECK(r.status == lo::optimizer_status::start_not_admissible);
  }
  SUBCASE("refused trial points across the border of the domain") {
    x_minus_log_x f;
    for (double x0 : {0.01, 30.0}) {
      const lo::optimizer_report b = lo::minimize(lo::bfgs_optimizer<>{}, f, start_of({x0}));
      const lo::optimizer_report l = lo::minimize(lo::lbfgs_optimizer<>{}, f, start_of({x0}));
      CHECK(lo::optimizer_succeeded(b.status));
      CHECK(lo::optimizer_succeeded(l.status));
      CHECK(std::abs(b.point(0) - 1.0) < 2e-3);  // f - f* ~ (x - 1)^2 / 2 <= 1e-6
      CHECK(std::abs(l.point(0) - 1.0) < 2e-3);
    }
  }
  SUBCASE("a confirming line search that spends its budget on rounding noise keeps the verdict") {
    rounding_noise_at_optimum f;
    const lo::optimizer_vector x0 = start_of({1e-6, 1e-6, 1e-6});  // decrement 1.5e-12
    const lo::optimizer_report b = lo::minimize(lo::bfgs_optimizer<>{}, f, x0);
    const lo::optimizer_report l = lo::minimize(lo::lbfgs_optimizer<>{}, f, x0);
    INFO("BFGS status ", int(b.status), " after ", b.value_evaluations, " evaluations; L-BFGS ", int(l.status));
    CHECK(b.status == lo::optimizer_status::converged);
    CHECK(l.status == lo::optimizer_status::converged);
    CHECK(b.iterations == 0);  // the confirming search found no acceptable step
    CHECK(b.value_evaluations == 31);  // the start and the 30 of the Wolfe search
    CHECK(distance(b, {0.0, 0.0, 0.0}) <= 1e-6);
  }
  SUBCASE("evaluation budget") {
    tp::extended_rosenbrock_problem p(2);
    lo::bfgs_optimizer<> b;
    b.max_evaluations = 10;
    const lo::optimizer_report r = lo::minimize(b, p, start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::evaluation_budget);
    CHECK(r.value_evaluations >= 10);
  }
  SUBCASE("trace: one call per accepted iteration") {
    tp::beale_problem p;
    lo::lbfgs_optimizer<> l;
    int calls = 0, last = 0;
    l.trace = [&](const lo::optimizer_trace_point& t) { ++calls, last = t.iteration; };
    const lo::optimizer_report r = lo::minimize(l, p, start_of(p.x0));
    CHECK(calls == r.iterations);
    CHECK(last == r.iterations);
  }
  SUBCASE("a tolerance below the resolution of f = 1e6 is never met as converged") {
    tp::extended_rosenbrock_problem p(2, 1e6);
    lo::bfgs_optimizer<> b;
    b.stop_rule.tolerance = 1e-14;  // eps |f| = 2.2e-10
    const lo::optimizer_report r = lo::minimize(b, p, start_of(p.x0));
    INFO("status ", int(r.status), ", decrement ", r.decrement);
    CHECK(r.status != lo::optimizer_status::converged);
    CHECK(r.value - p.f_star <= 1e-5);
  }
  SUBCASE("a change rule stops on change, never converged") {
    tp::extended_rosenbrock_problem p(2);
    lo::bfgs_optimizer<lo::objective_change_stop_rule> b{{1e-2}};
    const lo::optimizer_report r = lo::minimize(b, p, start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::stopped_on_change);
  }
}
