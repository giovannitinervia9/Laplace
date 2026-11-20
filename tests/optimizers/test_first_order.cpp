// Gradient descent, Barzilai-Borwein, Adam and the nonmonotone line search
// (include/laplace/optimizers/gradient_descent.hpp, barzilai_borwein.hpp,
// adam.hpp, line_search.hpp) on the battery of test_problems.hpp and
// extra_problems.hpp, and against optimizers7's gd() and bb() frozen in
// tests/data/reference/optimizers/optimizers7_gd_bb_counts.csv. Host code only.
//
// Bounds: the default stopping rule is the decrement at 1e-6, so f - f* is
// checked against 1e-5 and x - x* against the distances measured with a
// margin, never against anything tighter than what the rule guarantees.

#include <doctest/doctest.h>

#include <laplace/optimizers/adam.hpp>
#include <laplace/optimizers/barzilai_borwein.hpp>
#include <laplace/optimizers/gradient_descent.hpp>
#include <laplace/optimizers/line_search.hpp>
#include <laplace/optimizers/minimize.hpp>

#include "extra_problems.hpp"
#include "test_problems.hpp"
#include "wave2_reference.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;

using gd_armijo = lo::gradient_descent_optimizer<>;
using gd_nonmonotone = lo::gradient_descent_optimizer<lo::decrement_stop_rule, lo::nonmonotone_line_search>;

static_assert(lo::optimizer_for<gd_armijo, tp::wood_problem>);
static_assert(lo::optimizer_for<gd_nonmonotone, tp::wood_problem>);
static_assert(lo::optimizer_for<lo::gradient_descent_optimizer<lo::decrement_stop_rule, lo::wolfe_line_search>, tp::beale_problem>);
static_assert(lo::optimizer_for<lo::barzilai_borwein_optimizer<>, tp::abs_sum_problem>);
static_assert(lo::optimizer_for<lo::barzilai_borwein_optimizer<lo::gradient_stop_rule, lo::armijo_line_search>, tp::beale_problem>);
static_assert(lo::optimizer_for<lo::adam_optimizer<>, tp::beale_problem>);

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

lo::barzilai_borwein_optimizer<> bb_with(lo::barzilai_borwein_step step) {
  lo::barzilai_borwein_optimizer<> b;
  b.step = step;
  return b;
}

// a minimum reached: succeeded, f - f* <= 1e-5, max |x - x*| <= dx; with
// stall_ok a run may also end stalled there, its measure above the tolerance
// but its steps below the rounding of f
template <class Optimizer, class Problem>
void check_minimum(const Optimizer& optimizer, Problem p, double dx, bool stall_ok = false) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
  INFO(p.name, ", offset ", p.offset, ": status ", int(r.status), ", iterations ", r.iterations, ", f - f* ",
       r.value - p.f_star, ", decrement ", r.decrement);
  const bool stalled = stall_ok && r.status == lo::optimizer_status::stalled;
  CHECK((lo::optimizer_succeeded(r.status) || stalled));
  CHECK(r.value - p.f_star <= 1e-5);
  CHECK(distance(r, p.x_star) <= dx);
  if (!stalled) CHECK(r.decrement <= 1e-6);
  CHECK(r.value_evaluations >= r.iterations + 1);
  CHECK(r.gradient_evaluations >= r.iterations + 1);
  CHECK(r.curvature_evaluations == 0);
}

// a run that may not converge, but must not claim a false minimum
template <class Optimizer, class Problem>
void check_no_false_success(const Optimizer& optimizer, Problem p) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
  INFO(p.name, ": status ", int(r.status), ", f - f* ", r.value - p.f_star, ", decrement ", r.decrement);
  if (lo::optimizer_succeeded(r.status)) CHECK(r.value - p.f_star <= 1e-5);
}

// Wood: a minimum, or the stationary point at f = 7.877 near the saddle,
// where a model without negative curvature stops (as L-BFGS does)
template <class Optimizer>
void check_wood(const Optimizer& optimizer, double offset) {
  tp::wood_problem p(offset);
  const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
  INFO("Wood, offset ", offset, ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK((r.value - p.f_star <= 1e-5 || std::abs(r.value - p.f_star - 7.877) < 1e-2));
}

// Brown badly scaled is not tested: every first-order method here ends at its
// iteration budget there (its scales differ by 1e6), as optimizers7's do.
template <class Optimizer>
void more_garbow_hillstrom(const Optimizer& optimizer) {
  for (double offset : {0.0, 1.0, 1e3, 1e6}) {
    // at f* = 1e6 gradient descent ends stalled on Rosenbrock: decrement 3e-6,
    // f - f* 6e-8, its predicted decreases below the rounding of f (2.2e-10)
    check_minimum(optimizer, tp::extended_rosenbrock_problem(2, offset), 5e-3, offset == 1e6);
    check_minimum(optimizer, tp::extended_rosenbrock_problem(100, offset), 5e-3, offset == 1e6);
    check_minimum(optimizer, tp::powell_singular_problem(offset), 5e-2);  // singular at x*: |x - x*| ~ (f - f*)^(1/4)
    check_minimum(optimizer, tp::beale_problem(offset), 5e-3);
    check_minimum(optimizer, tp::helical_valley_problem(offset), 5e-3);
    check_wood(optimizer, offset);
    check_no_false_success(optimizer, tp::brown_badly_scaled_problem(offset));
  }
}

// none of these may end converged at a false point
template <class Optimizer>
void negative_cases(Optimizer optimizer) {
  {
    tp::noisy_problem p(4);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    INFO("noisy: status ", int(r.status));
    CHECK_FALSE(lo::optimizer_succeeded(r.status));
  }
  // abs_sum: from integer starts a step of 1 / |g|_inf can land exactly on the
  // kink at 0, where the subgradient returned is 0; that is the minimum
  for (int n : {2, 4, 10}) {
    tp::abs_sum_problem p(n);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    INFO("abs_sum n = ", n, ": status ", int(r.status), ", f ", r.value);
    if (lo::optimizer_succeeded(r.status)) CHECK(r.value <= 1e-5);
  }
  // the saddle problem from (1, 0), where g_2 = 0 exactly: every method that
  // moves along -g reaches the saddle (0, 0), a stationary point
  {
    tp::saddle_problem p;
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    // a success must be at a stationary point: the saddle or a minimum (0, +-1)
    const double near = std::min({std::max(std::abs(r.point(0)), std::abs(r.point(1))),
                                  std::max(std::abs(r.point(0)), std::abs(r.point(1) - 1.0)),
                                  std::max(std::abs(r.point(0)), std::abs(r.point(1) + 1.0))});
    INFO("saddle: status ", int(r.status), " at (", r.point(0), ", ", r.point(1), ")");
    if (lo::optimizer_succeeded(r.status)) CHECK(near <= 5e-3);
  }
  for (double kappa : {1e8, 1e10}) check_no_false_success(optimizer, tp::ill_conditioned_quadratic_problem(10, kappa));
  optimizer.max_iterations = 100;
  {
    tp::exploded_cosh_problem p(4);
    const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
    INFO("exploded cosh: status ", int(r.status), ", iterations ", r.iterations);
    CHECK_FALSE(lo::optimizer_succeeded(r.status));
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

// finite at the start only
struct finite_at_start_only {
  double value(const lo::optimizer_point& x) { return x(0) == 2.0 ? 1.0 : std::nan(""); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    g(0) = 1.0;
    return value(x);
  }
};

}  // namespace

TEST_CASE("optimizers: nonmonotone line search") {
  tp::extended_rosenbrock_problem p(2);
  lo::optimizer_vector x = start_of(p.x0), g("g", 2), d("d", 2), xt("xt", 2), xa("xa", 2);
  const double f = p.value_gradient(x, g);
  for (int i = 0; i < 2; ++i) d(i) = -g(i);
  SUBCASE("averaging 0 is Armijo's search") {
    lo::nonmonotone_line_search nm;
    nm.averaging = 0.0;
    lo::nonmonotone_reference c;
    c.start(f);
    for (double s0 : {1.0, 1e-2, 1e-3}) {
      const lo::line_search_result a = lo::armijo_line_search{}.search(p, x, f, g, d, s0, xa);
      const lo::line_search_result n = nm.search(p, x, f, c, g, d, s0, xt);
      CHECK(a.outcome == n.outcome);
      CHECK(a.step == n.step);
      CHECK(a.value == n.value);
      CHECK(a.value_evaluations == n.value_evaluations);
    }
  }
  SUBCASE("the reference: C_k, Q_k and C_k >= f_k") {
    lo::nonmonotone_reference c;
    c.start(10.0);
    c.accept(4.0, 0.85);  // Q = 1.85, C = (0.85 * 10 + 4) / 1.85
    CHECK(c.weight == doctest::Approx(1.85));
    CHECK(c.value == doctest::Approx((0.85 * 10.0 + 4.0) / 1.85));
    c.accept(1.0, 0.85);
    CHECK(c.weight == doctest::Approx(0.85 * 1.85 + 1.0));
    CHECK(c.value >= 1.0);
    lo::nonmonotone_reference m;
    m.start(10.0);
    m.accept(4.0, 0.0);
    CHECK(m.value == 4.0);
  }
  SUBCASE("a step that raises f is accepted below the reference") {
    // f = 4.0 at x, reference 100: the full step to f(x - g) is far above f
    // and still accepted (Armijo refuses it)
    lo::nonmonotone_reference c;
    c.start(100.0);
    lo::nonmonotone_line_search nm;
    const lo::line_search_result n = nm.search(p, x, f, c, g, d, 1e-2, xt);
    const lo::line_search_result a = lo::armijo_line_search{}.search(p, x, f, g, d, 1e-2, xa);
    CHECK(n.accepted());
    CHECK(n.step == 1e-2);
    CHECK(n.value > f);
    CHECK(a.step < 1e-2);
  }
}

TEST_CASE("optimizers: gradient descent on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  more_garbow_hillstrom(gd_armijo{});
  more_garbow_hillstrom(gd_nonmonotone{});
}

TEST_CASE("optimizers: Barzilai-Borwein on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  for (auto step : {lo::barzilai_borwein_step::alternating, lo::barzilai_borwein_step::long_step,
                    lo::barzilai_borwein_step::short_step})
    more_garbow_hillstrom(bb_with(step));
}

TEST_CASE("optimizers: Adam on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  lo::adam_optimizer<> a;
  a.learning_rate = 1e-2;
  more_garbow_hillstrom(a);
}

TEST_CASE("optimizers: first-order methods on ill-conditioned quadratics") {
  for (double kappa : {1e2, 1e4, 1e6}) {
    tp::ill_conditioned_quadratic_problem p(10, kappa);
    if (kappa <= 1e2) {
      check_minimum(gd_armijo{}, p, 5e-3);
      check_minimum(gd_nonmonotone{}, p, 5e-3);
    } else {
      check_no_false_success(gd_armijo{}, p);  // the iteration budget at kappa >= 1e4
    }
    check_minimum(bb_with(lo::barzilai_borwein_step::alternating), p, 5e-3);
    check_minimum(bb_with(lo::barzilai_borwein_step::short_step), p, 5e-3);
    if (kappa <= 1e4) check_minimum(bb_with(lo::barzilai_borwein_step::long_step), p, 5e-3);
    else check_no_false_success(bb_with(lo::barzilai_borwein_step::long_step), p);
    check_minimum(lo::adam_optimizer<>{}, p, 5e-3);
  }
  for (int n : {20, 50, 100})
    for (double kappa : {1e2, 1e4, 1e6}) {
      tp::ill_conditioned_quadratic_problem p(n, kappa);
      check_no_false_success(gd_armijo{}, p);
      check_no_false_success(bb_with(lo::barzilai_borwein_step::alternating), p);
      check_no_false_success(bb_with(lo::barzilai_borwein_step::short_step), p);
    }
}

TEST_CASE("optimizers: first-order methods must not converge on the negative cases") {
  negative_cases(gd_armijo{});
  negative_cases(gd_nonmonotone{});
  negative_cases(bb_with(lo::barzilai_borwein_step::alternating));
  negative_cases(bb_with(lo::barzilai_borwein_step::long_step));
  negative_cases(bb_with(lo::barzilai_borwein_step::short_step));
  lo::adam_optimizer<> a;
  a.max_iterations = 20000;
  negative_cases(a);
}

TEST_CASE("optimizers: first-order start, refused points, budgets, trace") {
  SUBCASE("start at the minimum: converged at iteration 0") {
    tp::ill_conditioned_quadratic_problem p(5, 1e3);
    for (const lo::optimizer_report& r : {lo::minimize(gd_armijo{}, p, start_of(p.x_star)),
                                          lo::minimize(lo::barzilai_borwein_optimizer<>{}, p, start_of(p.x_star))}) {
      CHECK(r.status == lo::optimizer_status::converged);
      CHECK(r.iterations == 0);
      CHECK(r.decrement == 0.0);
      CHECK(r.value_evaluations == 1);
    }
    const lo::optimizer_report a = lo::minimize(lo::adam_optimizer<>{}, p, start_of(p.x_star));
    CHECK(a.status == lo::optimizer_status::converged);
    CHECK(a.iterations == 0);  // the confirming step does not move x
  }
  SUBCASE("start not admissible") {
    x_minus_log_x f;
    CHECK(lo::minimize(gd_armijo{}, f, start_of({-1.0})).status == lo::optimizer_status::start_not_admissible);
    CHECK(lo::minimize(lo::barzilai_borwein_optimizer<>{}, f, start_of({-1.0})).status ==
          lo::optimizer_status::start_not_admissible);
    CHECK(lo::minimize(lo::adam_optimizer<>{}, f, start_of({-1.0})).status ==
          lo::optimizer_status::start_not_admissible);
  }
  SUBCASE("refused trial points across the border of the domain") {
    x_minus_log_x f;
    for (double x0 : {0.01, 30.0}) {
      const lo::optimizer_report g = lo::minimize(gd_armijo{}, f, start_of({x0}));
      const lo::optimizer_report b = lo::minimize(lo::barzilai_borwein_optimizer<>{}, f, start_of({x0}));
      INFO("x0 ", x0, ": GD ", int(g.status), " at ", g.point(0), ", BB ", int(b.status), " at ", b.point(0));
      CHECK(lo::optimizer_succeeded(g.status));
      CHECK(lo::optimizer_succeeded(b.status));
      CHECK(std::abs(g.point(0) - 1.0) < 3e-3);  // f - f* ~ (x - 1)^2 / 2 <= 2e-6
      CHECK(std::abs(b.point(0) - 1.0) < 3e-3);
    }
  }
  SUBCASE("Adam halves a step to a refused point") {
    // from 2 with learning rate 5 the first step is -5 (bias-corrected
    // m / sqrt(v) = 1): -3 and -0.5 are refused, 0.75 accepted
    x_minus_log_x f;
    lo::adam_optimizer<> a;
    a.learning_rate = 5.0;
    a.max_iterations = 1;
    const lo::optimizer_report r = lo::minimize(a, f, start_of({2.0}));
    CHECK(r.status == lo::optimizer_status::iteration_budget);
    CHECK(r.point(0) == doctest::Approx(0.75).epsilon(1e-7));
    CHECK(r.value_evaluations == 4);
  }
  SUBCASE("Adam: every trial point refused") {
    finite_at_start_only f;
    lo::adam_optimizer<> a;
    a.max_refusals = 10;
    const lo::optimizer_report r = lo::minimize(a, f, start_of({2.0}));
    CHECK(r.status == lo::optimizer_status::line_search_failed);
    CHECK(r.value_evaluations == 12);
    CHECK(r.point(0) == 2.0);
  }
  SUBCASE("budgets") {
    tp::extended_rosenbrock_problem p(2);
    gd_armijo g;
    g.max_evaluations = 10;
    CHECK(lo::minimize(g, p, start_of(p.x0)).status == lo::optimizer_status::evaluation_budget);
    lo::barzilai_borwein_optimizer<> b;
    b.max_iterations = 5;
    const lo::optimizer_report rb = lo::minimize(b, p, start_of(p.x0));
    CHECK(rb.status == lo::optimizer_status::iteration_budget);
    CHECK(rb.iterations == 5);
    lo::adam_optimizer<> a;
    a.max_iterations = 5;
    const lo::optimizer_report ra = lo::minimize(a, p, start_of(p.x0));
    CHECK(ra.status == lo::optimizer_status::iteration_budget);
    CHECK(ra.iterations == 5);
    CHECK(ra.value_evaluations == 6);
    a.max_iterations = 1000;
    a.max_evaluations = 20;
    CHECK(lo::minimize(a, p, start_of(p.x0)).status == lo::optimizer_status::evaluation_budget);
  }
  SUBCASE("trace: one call per accepted iteration") {
    tp::beale_problem p;
    int calls = 0, last = 0;
    lo::barzilai_borwein_optimizer<> b;
    b.trace = [&](const lo::optimizer_trace_point& t) { ++calls, last = t.iteration; };
    const lo::optimizer_report r = lo::minimize(b, p, start_of(p.x0));
    CHECK(calls == r.iterations);
    CHECK(last == r.iterations);
    lo::adam_optimizer<> a;
    a.learning_rate = 1e-2;
    calls = 0;
    a.trace = [&](const lo::optimizer_trace_point& t) { ++calls, last = t.iteration; };
    const lo::optimizer_report ra = lo::minimize(a, p, start_of(p.x0));
    CHECK(calls == ra.iterations);
    CHECK(last == ra.iterations);
  }
  SUBCASE("a change rule stops on change, never converged") {
    tp::extended_rosenbrock_problem p(2);
    lo::gradient_descent_optimizer<lo::objective_change_stop_rule> g{{1e-2}};
    CHECK(lo::minimize(g, p, start_of(p.x0)).status == lo::optimizer_status::stopped_on_change);
    lo::adam_optimizer<lo::objective_change_stop_rule> a{{1e-2}};
    CHECK(lo::minimize(a, p, start_of(p.x0)).status == lo::optimizer_status::stopped_on_change);
  }
}

namespace {

// read_rows splits at every comma; the problem names hold commas inside
// quotes, so the fields are joined back and split again honouring the quotes
// (read_rows has already dropped the quote characters, which are restored
// from the original line's shape: the first field is the only one with commas)
std::vector<std::string> csv_fields(const std::vector<std::string>& split) {
  const std::size_t columns = 15, extra = split.size() - columns;
  std::vector<std::string> f;
  std::string name = split[0];
  for (std::size_t i = 1; i <= extra; ++i) name += "," + split[i];
  f.push_back(name);
  for (std::size_t i = extra + 1; i < split.size(); ++i) f.push_back(split[i]);
  return f;
}

// a battery problem from its name in the optimizers7 references, or false
template <class F>
bool with_problem(const std::string& name, int n, double offset, F&& f) {
  if (name.rfind("extended Rosenbrock", 0) == 0) return f(tp::extended_rosenbrock_problem(n, offset)), true;
  if (name == "Powell singular") return f(tp::powell_singular_problem(offset)), true;
  if (name == "Beale") return f(tp::beale_problem(offset)), true;
  if (name == "helical valley") return f(tp::helical_valley_problem(offset)), true;
  if (name == "Wood") return f(tp::wood_problem(offset)), true;
  if (name == "quadratic n = 10, kappa = 100") return f(tp::ill_conditioned_quadratic_problem(10, 1e2, offset)), true;
  return false;  // Brown, the quadratics at 1e6 and 1e12, the saddle, cosh, noisy: not converged in R
}

}  // namespace

// optimizers7's convergence is not a target (most of its successes are change
// rules): wherever its gd() or bb() reached the known minimum, Laplace's
// method of the same kind must succeed with f - f* <= 1e-5, except Wood for
// alternating and short steps (the stationary point at 7.877, see check_wood).
TEST_CASE("optimizers: gradient descent and Barzilai-Borwein against optimizers7's gd() and bb()") {
  auto rows = wave2_reference::read_rows(LAPLACE_REFERENCE_DIR "/optimizers/optimizers7_gd_bb_counts.csv");
  for (auto& row : rows) row = csv_fields(row);
  REQUIRE(rows.size() == 212);
  int compared = 0;
  for (const auto& row : rows) {
    // problem, n, offset, method, iterations, f_count, g_count, h_count, value, value_minus_f_star,
    // gradient_max_norm, distance_max, converged, rule_fired, point
    const std::string& name = row[0];
    const int n = std::stoi(row[1]);
    const double offset = wave2_reference::to_double(row[2]);
    const std::string& method = row[3];
    if (row[12] != "TRUE" || wave2_reference::to_double(row[9]) > 1e-5) continue;
    auto check = [&](auto p) {
      auto run = [&](const auto& optimizer) {
        const lo::optimizer_report r = lo::minimize(optimizer, p, start_of(p.x0));
        INFO(name, ", offset ", offset, ", ", method, ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
        CHECK(lo::optimizer_succeeded(r.status));
        if (name == "Wood" && method != "gd" && method != "bb_bb1")
          CHECK((r.value - p.f_star <= 1e-5 || std::abs(r.value - p.f_star - 7.877) < 1e-2));
        else
          CHECK(r.value - p.f_star <= 1e-5);
        ++compared;
      };
      if (method == "gd") run(gd_armijo{});
      else if (method == "bb_alternate") run(bb_with(lo::barzilai_borwein_step::alternating));
      else if (method == "bb_bb1") run(bb_with(lo::barzilai_borwein_step::long_step));
      else if (method == "bb_bb2") run(bb_with(lo::barzilai_borwein_step::short_step));
    };
    with_problem(name, n, offset, check);
  }
  CHECK(compared >= 100);
}
