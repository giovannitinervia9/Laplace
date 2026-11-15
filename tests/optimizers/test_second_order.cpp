// Newton (newton.hpp) and the trust region (trust_region.hpp) on the battery
// of test problems (test_problems.hpp): the Moré-Garbow-Hillstrom problems
// with f* = 0, 1, 1e3 and 1e6, ill-conditioned quadratics, saddles and
// indefinite starts, refused trial points, and the negative tests that must
// not end as converged. A converged run is checked no tighter than the rule
// that stopped it: with the decrement below tol, f - f* is about the
// decrement (3/2 of it on a quartic), so f - f* <= 2 tol plus the rounding of
// f*. Host code only: no Kokkos kernel.

#include <doctest/doctest.h>

#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/stop_rules.hpp>
#include <laplace/optimizers/trust_region.hpp>

#include "test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;
using lo::optimizer_point;
using lo::optimizer_vector;
using matrix = lo::dense_curvature_matrix;

constexpr double eps = std::numeric_limits<double>::epsilon();

optimizer_vector start_of(const std::vector<double>& v) {
  optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

std::string status_name(lo::optimizer_status s) {
  switch (s) {
    case lo::optimizer_status::converged: return "converged";
    case lo::optimizer_status::converged_at_resolution: return "converged at resolution";
    case lo::optimizer_status::stopped_on_change: return "stopped on change";
    case lo::optimizer_status::stalled: return "stalled";
    case lo::optimizer_status::iteration_budget: return "iteration budget";
    case lo::optimizer_status::evaluation_budget: return "evaluation budget";
    case lo::optimizer_status::line_search_failed: return "line search failed";
    default: return "start not admissible";
  }
}

// a converged run at the minimum f_star, within what the decrement rule allows
template <class P>
void check_minimum(const lo::optimizer_report& r, const P& p, const std::string& method, double tol = 1e-6) {
  const double rounding = 8.0 * eps * std::abs(p.f_star);
  INFO(method << " on " << p.name << " + " << p.offset << ": " << status_name(r.status) << " after " << r.iterations
              << " iterations, f - f* = " << r.value - p.f_star << ", decrement " << r.decrement);
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK(r.value - p.f_star <= 2.0 * tol + rounding);
  CHECK(r.value - p.f_star >= -rounding);
  CHECK(r.final_factorization == lo::curvature_factorization::positive_definite);
  CHECK(r.curvature_evaluations == r.gradient_evaluations);
}

// f = sum |x_i|, with its subgradient and H = 0: not smooth
struct abs_sum_problem {
  using curvature_type = matrix;
  double value(const optimizer_point& x) {
    double f = 0.0;
    for (std::size_t i = 0; i < x.extent(0); ++i) f += std::abs(x(i));
    return f;
  }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) {
    for (std::size_t i = 0; i < x.extent(0); ++i) g(i) = x(i) > 0.0 ? 1.0 : x(i) < 0.0 ? -1.0 : 0.0;
    return value(x);
  }
  double value_gradient_curvature(const optimizer_point& x, const optimizer_vector& g, matrix& H) {
    for (std::size_t j = 0; j < x.extent(0); ++j)
      for (std::size_t i = 0; i < x.extent(0); ++i) H(i, j) = 0.0;
    return value_gradient(x, g);
  }
};

// f = x - log x, NaN for x <= 0, minimum 1 at x = 1. From x = 10 the Newton
// step goes to -80: a refused point (decision 3).
struct x_minus_log_x_problem {
  using curvature_type = matrix;
  double value(const optimizer_point& x) { return x(0) > 0.0 ? x(0) - std::log(x(0)) : std::nan(""); }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) {
    g(0) = 1.0 - 1.0 / x(0);
    return value(x);
  }
  double value_gradient_curvature(const optimizer_point& x, const optimizer_vector& g, matrix& H) {
    H(0, 0) = 1.0 / (x(0) * x(0));
    return value_gradient(x, g);
  }
};

static_assert(lo::optimizer_for<lo::newton_optimizer<>, tp::wood_problem>);
static_assert(lo::optimizer_for<lo::trust_region_optimizer<>, tp::wood_problem>);
static_assert(lo::optimizer_for<lo::newton_optimizer<lo::any_of_stop_rule<lo::decrement_stop_rule,
                                                                         lo::parameter_change_stop_rule>>,
                                tp::wood_problem>);

// every problem of the MGH battery with one offset, through both methods
template <class Check>
void for_each_mgh_problem(double offset, Check&& check) {
  {
    tp::extended_rosenbrock_problem p(2, offset);
    check(p);
  }
  {
    tp::extended_rosenbrock_problem p(10, offset);
    check(p);
  }
  {
    tp::powell_singular_problem p(offset);
    check(p);
  }
  {
    tp::beale_problem p(offset);
    check(p);
  }
  {
    tp::brown_badly_scaled_problem p(offset);
    check(p);
  }
  {
    tp::helical_valley_problem p(offset);
    check(p);
  }
  {
    tp::wood_problem p(offset);
    check(p);
  }
}

}  // namespace

TEST_CASE("optimizers: Newton and trust region on the More-Garbow-Hillstrom problems") {
  for (double offset : {0.0, 1.0, 1e3, 1e6}) {
    for_each_mgh_problem(offset, [](auto& p) {
      const optimizer_vector x0 = start_of(p.x0);
      check_minimum(lo::minimize(lo::newton_optimizer<>{}, p, x0), p, "Newton");
      check_minimum(lo::minimize(lo::trust_region_optimizer<>{}, p, x0), p, "trust region");
    });
  }
}

TEST_CASE("optimizers: Newton and trust region on ill-conditioned quadratics") {
  for (double kappa : {1e2, 1e4, 1e6, 1e8, 1e10, 1e12}) {
    for (double offset : {0.0, 1e6}) {
      tp::ill_conditioned_quadratic_problem p(10, kappa, offset);
      const optimizer_vector x0 = start_of(p.x0);
      const lo::optimizer_report newton = lo::minimize(lo::newton_optimizer<>{}, p, x0);
      const lo::optimizer_report region = lo::minimize(lo::trust_region_optimizer<>{}, p, x0);
      check_minimum(newton, p, "Newton");
      check_minimum(region, p, "trust region");
      // f - f* <= 2 tol bounds |x - x*| by sqrt(4 tol / lambda_min), lambda_min = 1
      for (int i = 0; i < 10; ++i) {
        CHECK(std::abs(newton.point(i) - p.x_star[i]) <= std::sqrt(4e-6) + 1e-9 * kappa);
        CHECK(std::abs(region.point(i) - p.x_star[i]) <= std::sqrt(4e-6) + 1e-9 * kappa);
      }
      CHECK(newton.iterations <= 2);  // one Newton step, a second only past rounding
    }
  }
}

TEST_CASE("optimizers: saddles and indefinite starts") {
  SUBCASE("the saddle problem from (1, 0), where H = diag(2, -1) and g_2 = 0") {
    for (double offset : {0.0, 1e3}) {
      tp::saddle_problem p(offset);
      const optimizer_vector x0 = start_of(p.x0);
      const lo::optimizer_report newton = lo::minimize(lo::newton_optimizer<>{}, p, x0);
      const lo::optimizer_report region = lo::minimize(lo::trust_region_optimizer<>{}, p, x0);
      check_minimum(newton, p, "Newton");
      check_minimum(region, p, "trust region");
      CHECK(std::abs(std::abs(newton.point(1)) - 1.0) <= 1e-2);
      CHECK(std::abs(std::abs(region.point(1)) - 1.0) <= 1e-2);
    }
  }
  SUBCASE("started exactly at the saddle, g = 0") {
    tp::saddle_problem p;
    const optimizer_vector x0 = start_of({0.0, 0.0});
    check_minimum(lo::minimize(lo::newton_optimizer<>{}, p, x0), p, "Newton");
    check_minimum(lo::minimize(lo::trust_region_optimizer<>{}, p, x0), p, "trust region");
  }
  SUBCASE("Rosenbrock from (0, 1), where H is indefinite") {
    tp::extended_rosenbrock_problem p(2);
    const optimizer_vector x0 = start_of({0.0, 1.0});
    check_minimum(lo::minimize(lo::newton_optimizer<>{}, p, x0), p, "Newton");
    check_minimum(lo::minimize(lo::trust_region_optimizer<>{}, p, x0), p, "trust region");
  }
}

TEST_CASE("optimizers: Newton and trust region refuse non-finite trial points") {
  x_minus_log_x_problem p;
  const optimizer_vector x0 = start_of({10.0});
  const lo::optimizer_report newton = lo::minimize(lo::newton_optimizer<>{}, p, x0);
  lo::trust_region_optimizer<> wide;
  wide.initial_radius = 100.0;  // the first trial is the Newton step to -80
  const lo::optimizer_report region = lo::minimize(wide, p, x0);
  for (const auto& r : {newton, region}) {
    INFO(status_name(r.status) << ", x = " << r.point(0));
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(std::abs(r.point(0) - 1.0) <= 1e-2);
  }

  const optimizer_vector bad = start_of({-1.0});
  CHECK(lo::minimize(lo::newton_optimizer<>{}, p, bad).status == lo::optimizer_status::start_not_admissible);
  CHECK(lo::minimize(lo::trust_region_optimizer<>{}, p, bad).status == lo::optimizer_status::start_not_admissible);
}

TEST_CASE("optimizers: Newton and trust region do not report success where they should not") {
  SUBCASE("noisy objective") {
    tp::noisy_problem p(2);
    const optimizer_vector x0 = start_of(p.x0);
    const lo::optimizer_report newton = lo::minimize(lo::newton_optimizer<>{}, p, x0);
    const lo::optimizer_report region = lo::minimize(lo::trust_region_optimizer<>{}, p, x0);
    INFO("Newton " << status_name(newton.status) << ", trust region " << status_name(region.status));
    CHECK_FALSE(lo::optimizer_succeeded(newton.status));
    CHECK_FALSE(lo::optimizer_succeeded(region.status));
  }
  SUBCASE("exploded start, modest budget") {
    tp::exploded_cosh_problem p(2);
    const optimizer_vector x0 = start_of(p.x0);
    lo::newton_optimizer<> newton;
    newton.max_iterations = 100;
    lo::trust_region_optimizer<> region;
    region.max_iterations = 100;
    const lo::optimizer_report a = lo::minimize(newton, p, x0), b = lo::minimize(region, p, x0);
    INFO("Newton " << status_name(a.status) << ", trust region " << status_name(b.status));
    CHECK(a.status == lo::optimizer_status::iteration_budget);
    CHECK(b.status == lo::optimizer_status::iteration_budget);
  }
  SUBCASE("smooth methods on sum |x_i|") {
    abs_sum_problem p;
    const optimizer_vector x0 = start_of({0.3, -0.7, 1.1});
    const lo::optimizer_report newton = lo::minimize(lo::newton_optimizer<>{}, p, x0);
    const lo::optimizer_report region = lo::minimize(lo::trust_region_optimizer<>{}, p, x0);
    INFO("Newton " << status_name(newton.status) << ", trust region " << status_name(region.status));
    CHECK_FALSE(lo::optimizer_succeeded(newton.status));
    CHECK_FALSE(lo::optimizer_succeeded(region.status));
    CHECK(std::isnan(newton.decrement));
    CHECK(std::isnan(region.decrement));
  }
  SUBCASE("a change rule stops as stopped on change, never as converged") {
    tp::extended_rosenbrock_problem p(2);
    const optimizer_vector x0 = start_of(p.x0);
    lo::newton_optimizer<lo::objective_change_stop_rule> newton{{1.0}};
    lo::trust_region_optimizer<lo::objective_change_stop_rule> region{{1.0}};
    CHECK(lo::minimize(newton, p, x0).status == lo::optimizer_status::stopped_on_change);
    CHECK(lo::minimize(region, p, x0).status == lo::optimizer_status::stopped_on_change);
  }
}

TEST_CASE("optimizers: Newton's maximum step and the trace") {
  tp::extended_rosenbrock_problem p(2);
  const optimizer_vector x0 = start_of(p.x0);
  std::vector<double> steps, previous{p.x0};
  double largest = 0.0;
  int calls = 0;
  lo::newton_optimizer<> newton;
  newton.max_step = 0.25;
  newton.trace = [&](const lo::optimizer_trace_point& t) {
    ++calls;
    for (int i = 0; i < 2; ++i) largest = std::max(largest, std::abs(t.point(i) - previous[i])), previous[i] = t.point(i);
  };
  const lo::optimizer_report r = lo::minimize(newton, p, x0);
  check_minimum(r, p, "Newton, max_step 0.25");
  CHECK(calls == r.iterations);
  CHECK(largest <= 0.25 * (1.0 + 1e-12));

  int region_calls = 0;
  lo::trust_region_optimizer<> region;
  region.trace = [&](const lo::optimizer_trace_point&) { ++region_calls; };
  const lo::optimizer_report q = lo::minimize(region, p, x0);
  CHECK(region_calls == q.gradient_evaluations - 1);  // one per accepted step
}
