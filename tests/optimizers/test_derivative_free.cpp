// Nelder-Mead (nelder_mead.hpp) and compass search (compass.hpp), value only:
// F1's battery (More-Garbow-Hillstrom problems and ill-conditioned
// quadratics, each with f* = 0, 1, 1e3 and 1e6), the non-smooth abs_sum and
// the Laplace mu at fixed sigma (a sum of |y_i - mu|, minimum at the median),
// McKinnon's function (Nelder-Mead must not end converged at its
// non-stationary point (0, 0)), the noisy objective, and each method once as
// a block sub-optimizer of block_coordinate_optimizer (the Laplace with mu by
// the method and the log scale by Newton). Tolerances are no tighter than the
// rule that stopped the run. Host only: no kernels here but the fit_distrib
// reduction, which lives in named functions (CLAUDE.md trap).

#include <doctest/doctest.h>

#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/compass.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/nelder_mead.hpp>
#include <laplace/optimizers/newton.hpp>

#include "extra_problems.hpp"
#include "fit_distrib_data.hpp"
#include "test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace fb = fit_distrib_bench;
namespace tp = optimizer_test_problems;
using lo::optimizer_point;
using lo::optimizer_vector;

const bool verbose = std::getenv("LAPLACE_DERIVATIVE_FREE_VERBOSE") != nullptr;

optimizer_vector point_of(const std::vector<double>& v) {
  optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

double distance_inf(const optimizer_point& x, const std::vector<double>& y) {
  double d = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) d = std::max(d, std::abs(x(i) - y[i]));
  return d;
}

void print_report(const std::string& what, const lo::optimizer_report& r, double dx, double df) {
  if (!verbose) return;
  std::printf("%-44s status %d it %6d f %7ld search %9.3g |x-x*| %9.3g f-f* %9.3g %8.3f ms\n", what.c_str(),
              static_cast<int>(r.status), r.iterations, r.value_evaluations, r.search_size, dx, df,
              1e3 * r.elapsed_seconds);
}

// ---------------------------------------------------------------- the battery

// One problem by one method. A converged run must lie within the bound of its
// problem: a multiple of its search size, or the distance at which f can no
// longer tell points apart, which is C (eps max(1, |f*|))^(1/2) at a minimum
// with a positive definite Hessian (C absorbs sqrt(2 / lambda_min)) and
// C (eps max(1, |f*|))^(1/4) at Powell's, where f grows as |x - x*|^4. A run
// that does not converge must say so with a budget status; `must_converge`
// marks the runs measured to converge with room to spare.
template <class Optimizer, class P>
void battery_case(const char* method, const Optimizer& optimizer, P problem, double resolution_scale,
                  double resolution_power, bool must_converge) {
  const auto r = lo::minimize(optimizer, problem, point_of(problem.x0));
  const double dx = distance_inf(r.point, problem.x_star), df = r.value - problem.f_star;
  print_report(std::string(method) + " " + problem.name + " +" + std::to_string(problem.offset), r, dx, df);
  if (must_converge) CHECK_MESSAGE(lo::optimizer_succeeded(r.status), method, " ", problem.name, " +", problem.offset);
  if (lo::optimizer_succeeded(r.status)) {
    const double floor = resolution_scale * std::pow(std::numeric_limits<double>::epsilon() * std::max(1.0, problem.offset),
                                                     resolution_power);
    CHECK_MESSAGE(dx <= std::max(1e3 * r.search_size, floor), method, " ", problem.name, " +", problem.offset);
  } else {
    CHECK_MESSAGE((r.status == lo::optimizer_status::iteration_budget ||
                   r.status == lo::optimizer_status::evaluation_budget),
                  method, " ", problem.name, " +", problem.offset);
  }
}

// `robust`: the method is expected to converge on every problem but Powell's
// singular minimum, where compass search crawls (measured: 10000 polls and
// |x - x*| about 1e-2 from f* = 0 to 1e3).
template <class Optimizer>
void battery(const char* method, const Optimizer& optimizer, bool robust, bool powell) {
  for (double offset : {0.0, 1.0, 1e3, 1e6}) {
    battery_case(method, optimizer, tp::extended_rosenbrock_problem(2, offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::extended_rosenbrock_problem(4, offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::powell_singular_problem(offset), 10.0, 0.25, powell);
    battery_case(method, optimizer, tp::beale_problem(offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::helical_valley_problem(offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::wood_problem(offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::ill_conditioned_quadratic_problem(4, 1e2, offset), 100.0, 0.5, robust);
    battery_case(method, optimizer, tp::ill_conditioned_quadratic_problem(4, 1e4, offset), 1e3, 0.5, robust);
  }
}

// The largest |f(x +- h e_j) - f(x)| over the coordinates at the final point and
// h its search size: about h |grad f| + h^2 |H| / 2 where f is smooth, the
// amplitude of the noise where it is not (measured for the report only).
template <class O>
double final_poll_spread(O& objective, const lo::optimizer_report& r) {
  optimizer_vector t("t", r.point.extent(0));
  double spread = 0.0;
  for (std::size_t j = 0; j < t.extent(0); ++j)
    for (double s : {-1.0, 1.0}) {
      Kokkos::deep_copy(t, r.point);
      t(j) += s * r.search_size;
      spread = std::max(spread, std::abs(objective.value(optimizer_point(t)) - r.value));
    }
  return spread;
}

// ---------------------------------------------------------------- non-smooth

// sum_i |y_i - mu|: minimum on the median interval
struct absolute_deviation_objective {
  std::vector<double> y;
  double value(const optimizer_point& x) const {
    double f = 0.0;
    for (double t : y) f += std::abs(t - x(0));
    return f;
  }
};

template <class Optimizer>
void nonsmooth_cases(const char* method, const Optimizer& optimizer) {
  for (int n : {2, 4}) {
    tp::abs_sum_problem problem(n);
    const auto r = lo::minimize(optimizer, problem, point_of(problem.x0));
    const double dx = distance_inf(r.point, problem.x_star);
    print_report(std::string(method) + " abs_sum n = " + std::to_string(n), r, dx, r.value);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(dx <= 2.0 * r.search_size);
  }
  // odd and even sample sizes: a unique median, and an interval of minimizers
  for (int count : {7, 8}) {
    absolute_deviation_objective objective;
    tp::test_uniform u{static_cast<std::uint64_t>(count)};
    for (int i = 0; i < count; ++i) objective.y.push_back(10.0 * u() - 3.0);
    std::vector<double> ys = objective.y;
    std::sort(ys.begin(), ys.end());
    const double lower = ys[(count - 1) / 2], upper = ys[count / 2];
    const auto r = lo::minimize(optimizer, objective, point_of({-20.0}));
    const double outside = std::max({0.0, lower - r.point(0), r.point(0) - upper});
    print_report(std::string(method) + " median n = " + std::to_string(count), r, outside, 0.0);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(outside <= 2.0 * r.search_size);
  }
}

// McKinnon (1998), tau = 2, theta = 6, phi = 60: strictly convex, C1, minimum
// -1/4 at (0, -1/2); from his simplex classical Nelder-Mead performs inside
// contractions for ever toward (0, 0), where the gradient is (0, 1).
struct mckinnon_objective {
  double value(const optimizer_point& p) const {
    const double x = p(0), y = p(1);
    return (x <= 0.0 ? 360.0 * x * x : 6.0 * x * x) + y + y * y;
  }
};

void mckinnon_cases() {
  mckinnon_objective objective;
  const double s = std::sqrt(33.0);
  lo::nelder_mead_optimizer<> nm;
  nm.initial_simplex = {{1.0, 1.0}, {(1.0 + s) / 8.0, (1.0 - s) / 8.0}, {0.0, 0.0}};
  lo::nelder_mead_optimizer<> unguarded = nm;
  unguarded.max_restarts = 0;
  const auto x0 = point_of({0.0, 0.0});  // ignored: the simplex is given

  const auto ru = lo::minimize(unguarded, objective, x0);
  print_report("nelder-mead McKinnon, no safeguard", ru, distance_inf(ru.point, {0.0, -0.5}), ru.value + 0.25);
  // McKinnon's failure reproduced without the safeguard: converged at (0, 0), f = 0 against -1/4
  CHECK(lo::optimizer_succeeded(ru.status));
  CHECK(distance_inf(ru.point, {0.0, 0.0}) <= 1e-6);
  const auto r = lo::minimize(nm, objective, x0);
  print_report("nelder-mead McKinnon", r, distance_inf(r.point, {0.0, -0.5}), r.value + 0.25);
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK(distance_inf(r.point, {0.0, -0.5}) <= 1e3 * r.search_size);
  // never converged at the non-stationary point
  CHECK_FALSE((lo::optimizer_succeeded(r.status) && distance_inf(r.point, {0.0, 0.0}) < 0.1));
}

// The noisy objective sum p_i^2 + 100 sin(1e20 p_1). The search size certifies
// only that no lower value was found at that distance, and the noise has such
// points at every scale; what tells this apart from a minimum is the spread of
// f over the final simplex or poll (measured 157 and 159 here, at most 1.02e-8
// on every converged run of this file), so both methods must end stalled. With
// the spread check off (value_spread_tolerance infinite) the same runs end
// converged near the start (f about -96, |x| about 1): the check is what
// decides.
template <class Optimizer>
void noisy_case(const char* method, Optimizer optimizer) {
  tp::noisy_problem problem(4);
  const auto r = lo::minimize(optimizer, problem, point_of(problem.x0));
  print_report(std::string(method) + " noisy", r, distance_inf(r.point, problem.x_star), r.value);
  CHECK(r.status == lo::optimizer_status::stalled);
  optimizer.value_spread_tolerance = std::numeric_limits<double>::infinity();
  const auto unchecked = lo::minimize(optimizer, problem, point_of(problem.x0));
  print_report(std::string(method) + " noisy, spread check off", unchecked, distance_inf(unchecked.point, problem.x_star),
               unchecked.value);
  CHECK(lo::optimizer_succeeded(unchecked.status));
  CHECK(distance_inf(unchecked.point, problem.x_star) > 0.5);  // nowhere near the minimum of sum p_i^2
  tp::extended_rosenbrock_problem smooth(2);
  const auto rs = lo::minimize(optimizer, smooth, point_of(smooth.x0));
  if (verbose)
    std::printf("  final poll spread: noisy %.3g, Rosenbrock %.3g\n", final_poll_spread(problem, r),
                final_poll_spread(smooth, rs));
}

// ---------------------------------------------------------------- as a block

// laplace1 (mu, sigma) in free coordinates (mu, log sigma): mu by the method, log
// sigma by Newton on the expected curvature, against the closed form
template <class MuOptimizer>
void laplace_block_case(const char* method, const MuOptimizer& mu_optimizer, long n, std::uint64_t seed) {
  const ld::laplace1_distrib truth{1.5, 2.0};
  const auto y = fb::simulate(truth, n, seed);
  fb::fit_distrib_objective<ld::laplace1_distrib> objective(y, fb::fit_curvature::expected);
  std::vector<double> ys = fb::to_host_vector(y);
  std::sort(ys.begin(), ys.end());
  const long m = n / 2;
  const double lower = n % 2 ? ys[m] : ys[m - 1], upper = ys[m];
  double mad = 0.0;
  for (double v : ys) mad += std::abs(v - lower);
  mad /= static_cast<double>(n);

  const auto t0 = fb::library_start(ld::laplace1_distrib{}, ys);
  optimizer_vector u0("u0", 2);
  objective.free_of(t0.data(), u0);
  u0(0) += 1.0, u0(1) += 1.0;

  lo::block_coordinate_optimizer<MuOptimizer, lo::newton_optimizer<>> alternation{{mu_optimizer, {}}, {1, 1}};
  const auto r = lo::minimize(alternation, objective, u0);
  double theta[2];
  REQUIRE(objective.theta_of(r.point, theta));
  const double outside = std::max({0.0, lower - theta[0], theta[0] - upper});
  print_report(std::string(method) + " laplace block n = " + std::to_string(n), r, outside, theta[1] / mad - 1.0);
  REQUIRE(lo::optimizer_succeeded(r.status));
  CHECK(outside <= 2.0 * r.search_size + 1e-12 * std::max(1.0, std::abs(lower)));
  CHECK(std::abs(theta[1] / mad - 1.0) <= 2.0 * std::sqrt(2.0 * 1e-6 / static_cast<double>(n)) + 1e-10);
}

void block_cases() {
  for (long n : {1001L, 1000L}) {
    laplace_block_case("nelder-mead", lo::nelder_mead_optimizer<>{}, n, 20261120u + static_cast<std::uint64_t>(n));
    laplace_block_case("compass", lo::compass_optimizer<>{}, n, 20261121u + static_cast<std::uint64_t>(n));
  }
}

// ---------------------------------------------------------------- statuses

template <class Optimizer>
void status_cases(const char* method, Optimizer optimizer) {
  struct refused {
    double value(const optimizer_point& x) const { return x(0) < 0.0 ? std::numeric_limits<double>::quiet_NaN() : 1.0; }
  } bad;
  CHECK(lo::minimize(optimizer, bad, point_of({-1.0, 0.0})).status == lo::optimizer_status::start_not_admissible);
  tp::extended_rosenbrock_problem problem(2);
  optimizer.max_value_evaluations = 30;
  CHECK(lo::minimize(optimizer, problem, point_of(problem.x0)).status == lo::optimizer_status::evaluation_budget);
  optimizer.max_value_evaluations = 100000;
  optimizer.max_iterations = 5;
  CHECK(lo::minimize(optimizer, problem, point_of(problem.x0)).status == lo::optimizer_status::iteration_budget);
  (void)method;
}

}  // namespace

void derivative_free_battery() {
  battery("nelder-mead", lo::nelder_mead_optimizer<>{}, true, true);
  battery("compass", lo::compass_optimizer<>{}, true, false);
  // the coordinate poll fails on Rosenbrock n = 4, Wood and the kappa 1e4 quadratic (measured):
  // only honest statuses are asked of it
  lo::compass_optimizer<> coordinate;
  coordinate.poll_basis = lo::compass_poll_basis::coordinate;
  battery("compass coordinate", coordinate, false, false);
}
void derivative_free_nonsmooth() {
  nonsmooth_cases("nelder-mead", lo::nelder_mead_optimizer<>{});
  nonsmooth_cases("compass", lo::compass_optimizer<>{});
  mckinnon_cases();
  noisy_case("nelder-mead", lo::nelder_mead_optimizer<>{});
  noisy_case("compass", lo::compass_optimizer<>{});
}
void derivative_free_blocks() { block_cases(); }
void derivative_free_statuses() {
  status_cases("nelder-mead", lo::nelder_mead_optimizer<>{});
  status_cases("compass", lo::compass_optimizer<>{});
}

TEST_CASE("nelder_mead_optimizer and compass_optimizer: the battery with f* = 0, 1, 1e3, 1e6") {
  derivative_free_battery();
}
TEST_CASE("nelder_mead_optimizer and compass_optimizer: abs_sum, the median, McKinnon, the noisy objective") {
  derivative_free_nonsmooth();
}
TEST_CASE("nelder_mead_optimizer and compass_optimizer: mu of the Laplace as a block of block_coordinate_optimizer") {
  derivative_free_blocks();
}
TEST_CASE("nelder_mead_optimizer and compass_optimizer: refused start and budgets") { derivative_free_statuses(); }
