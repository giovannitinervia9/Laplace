// multistart_optimizer (include/laplace/optimizers/multistart.hpp): the
// generated starts, the selection rule on scripted runs, and BFGS from many
// starts on Styblinski-Tang (extra_problems.hpp). Host code only.

#include <doctest/doctest.h>

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/multistart.hpp>

#include "extra_problems.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;

lo::optimizer_vector start_of(const std::vector<double>& v) {
  lo::optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

struct scripted_run {
  lo::optimizer_status status;
  double value;
};

// returns the scripted reports in order, one per start, with the start as point
struct scripted_optimizer {
  std::vector<scripted_run> script;
  mutable std::size_t next = 0;
  template <class O>
  lo::optimizer_report run(O& objective, const lo::optimizer_point& x0) const {
    (void)objective.value(x0);
    lo::optimizer_report r;
    r.status = script[next].status, r.value = script[next].value;
    r.point = lo::optimizer_vector("point", x0.extent(0));
    Kokkos::deep_copy(r.point, x0);
    r.iterations = 1, r.value_evaluations = 2, r.gradient_evaluations = 3, r.curvature_evaluations = 4;
    ++next;
    return r;
  }
};

struct any_objective {
  double value(const lo::optimizer_point&) { return 0.0; }
};

// the index of the kept run: its point is the start (k, k)
int kept_index(const std::vector<scripted_run>& script, double tolerance = 1e-6) {
  lo::multistart_optimizer<scripted_optimizer> ms{{script}};
  ms.distinct_tolerance = tolerance;
  for (std::size_t k = 0; k < script.size(); ++k) ms.starts.push_back({double(k), double(k)});
  any_objective f;
  const lo::multistart_report r = ms.run_multistart(f, start_of({0.0, 0.0}));
  CHECK(r.start_count == int(script.size()));
  CHECK(r.best.iterations == int(script.size()));
  CHECK(r.best.value_evaluations == 2 * long(script.size()));
  CHECK(r.best.gradient_evaluations == 3 * long(script.size()));
  CHECK(r.best.curvature_evaluations == 4 * long(script.size()));
  return static_cast<int>(r.best.point(0));
}

constexpr auto converged = lo::optimizer_status::converged;
constexpr auto budget = lo::optimizer_status::iteration_budget;
constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();

}  // namespace

TEST_CASE("optimizers: multistart selection rule") {
  // a converged run beats a lower non-converged one, in either order
  CHECK(kept_index({{budget, -10.0}, {converged, -5.0}}) == 1);
  CHECK(kept_index({{converged, -5.0}, {budget, -10.0}}) == 0);
  // among converged runs the lower value; a later one only if lower by more than the tolerance
  CHECK(kept_index({{converged, -5.0}, {converged, -6.0}, {converged, -5.5}}) == 1);
  CHECK(kept_index({{converged, -5.0}, {converged, -5.0 - 1e-7}}) == 0);
  CHECK(kept_index({{converged, -5.0}, {converged, -5.0 - 1e-5}}) == 1);
  CHECK(kept_index({{converged, -5.0}, {converged, -5.0 - 1e-5}}, 1e-4) == 0);
  // among non-converged runs, the lower value
  CHECK(kept_index({{budget, 3.0}, {budget, 1.0}, {budget, 2.0}}) == 1);
  // a non-finite value is never kept
  CHECK(kept_index({{lo::optimizer_status::start_not_admissible, not_a_number}, {budget, 1.0}}) == 1);
  CHECK(kept_index({{converged, 1.0}, {converged, not_a_number}, {converged, -std::numeric_limits<double>::infinity()}}) == 0);
  // the best, never the last
  CHECK(kept_index({{converged, -9.0}, {converged, -1.0}, {converged, -2.0}}) == 0);
}

TEST_CASE("optimizers: multistart counts of optima") {
  lo::multistart_optimizer<scripted_optimizer> ms{
      {{{converged, -5.0}, {converged, -5.0 + 5e-7}, {budget, -50.0}, {converged, -3.0}, {converged, -5.0 - 5e-7}}}};
  for (int k = 0; k < 5; ++k) ms.starts.push_back({double(k)});
  any_objective f;
  const lo::multistart_report r = ms.run_multistart(f, start_of({0.0}));
  CHECK(r.converged_count == 4);
  CHECK(r.distinct_optimum_count == 2);  // {-5 - 5e-7, -5, -5 + 5e-7} and -3
  CHECK(r.best_found_count == 3);
  CHECK(r.best.value == -5.0);  // the later -5 - 5e-7 is not lower by more than the default 1e-5
  CHECK(ms.distinct_tolerance == 1e-5);
  CHECK(r.runs.size() == 5);
}

TEST_CASE("optimizers: multistart generated starts (Latin hypercube around x0)") {
  lo::multistart_optimizer<lo::bfgs_optimizer<>> ms;
  ms.start_count = 9;
  ms.spread = 0.5;
  const auto x0 = start_of({0.2, -4.0, 100.0});
  const auto s = ms.generated_starts(x0);
  REQUIRE(s.size() == 9);
  for (int j = 0; j < 3; ++j) {
    CHECK(s[0][j] == x0(j));
    const double half = 3.0 * 0.5 * std::max(1.0, std::abs(x0(j)));
    std::set<int> strata;
    for (int i = 1; i < 9; ++i) {
      const double u = (s[i][j] - (x0(j) - half)) / (2.0 * half);  // in [0, 1)
      CHECK(u >= 0.0);
      CHECK(u < 1.0);
      strata.insert(static_cast<int>(u * 8));
    }
    CHECK(strata.size() == 8);  // each stratum once
  }
  CHECK(ms.generated_starts(x0) == s);  // deterministic
  ms.seed = 2;
  CHECK(ms.generated_starts(x0) != s);
  ms.start_count = 1;
  CHECK(ms.generated_starts(x0).size() == 1);
}

TEST_CASE("optimizers: multistart BFGS finds the global minimum of Styblinski-Tang") {
  for (int n : {2, 4}) {
    tp::styblinski_tang_problem p(n);
    const auto x0 = start_of(p.x0);
    // one run from x0 ends at the worst minimum
    const lo::optimizer_report single = lo::minimize(lo::bfgs_optimizer<>{}, p, x0);
    CHECK(lo::optimizer_succeeded(single.status));
    CHECK(std::abs(single.value - p.minimum_value(n)) < 1e-5);

    lo::multistart_optimizer<lo::bfgs_optimizer<>> ms;
    ms.start_count = 40;
    const lo::multistart_report r = ms.run_multistart(p, x0);
    INFO("n = ", n, ": converged ", r.converged_count, ", distinct ", r.distinct_optimum_count);
    CHECK(lo::optimizer_succeeded(r.best.status));
    CHECK(std::abs(r.best.value - p.minimum_value(0)) < 1e-5);
    for (int i = 0; i < n; ++i) CHECK(std::abs(r.best.point(i) - p.a) < 5e-3);
    CHECK(r.distinct_optimum_count >= 2);
    CHECK(r.distinct_optimum_count <= n + 1);  // n + 1 distinct minimum values exist
    CHECK(r.best_found_count >= 1);
    long f_count = 0;
    for (const auto& run : r.runs) f_count += run.value_evaluations;
    CHECK(r.best.value_evaluations == f_count);
    // minimize() on the multistart optimizer gives the same kept run
    CHECK(lo::minimize(ms, p, x0).value == r.best.value);
  }
  SUBCASE("caller's starts: the best kept, not the last") {
    tp::styblinski_tang_problem p(2);
    lo::multistart_optimizer<lo::bfgs_optimizer<>> ms;
    ms.starts = {{-2.5, -2.5}, {2.5, -2.5}, {2.5, 2.5}};
    const lo::multistart_report r = ms.run_multistart(p, start_of({0.0, 0.0}));
    CHECK(r.start_count == 3);
    CHECK(std::abs(r.best.value - p.minimum_value(0)) < 1e-5);
    CHECK(std::abs(r.runs.back().value - p.minimum_value(2)) < 1e-5);
    CHECK(r.distinct_optimum_count == 3);
    CHECK(r.best_found_count == 1);
  }
}
