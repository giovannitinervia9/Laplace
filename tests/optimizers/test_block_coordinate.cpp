// Brent's method (brent.hpp) and block alternation (block_coordinate.hpp):
// Brent on smooth, kinked, refused and unbounded functions of one variable;
// the Laplace (laplace1, laplace2) by alternation, mu by Brent and the scale
// by Newton on the expected curvature, against the closed form (the median,
// the mean absolute deviation from it; at even n any point of the median
// interval) at n about 1e3 and 1e5; a smooth separable and a smooth coupled
// problem against Newton on the whole of x; and the negative case, a kink
// coupling two blocks, which must not end converged. The Laplace kernels run
// in fit_distrib_bench's named reduction (CLAUDE.md trap).

#include <doctest/doctest.h>

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>

#include "fit_distrib_data.hpp"
#include "regression_objective.hpp"

#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace fb = fit_distrib_bench;
using lo::optimizer_point;
using lo::optimizer_vector;

const bool verbose = std::getenv("LAPLACE_BLOCK_COORDINATE_VERBOSE") != nullptr;

// the lasso as a penalty: laplace2's rate on each coordinate of a block
using lasso_penalty = laplace::penalty::distrib_penalty<
    ld::independent_product_distrib<ld::univariate_as_multivariate<ld::laplace2_distrib>>>;
lasso_penalty make_lasso(int d) {
  return lasso_penalty(ld::independent_product_distrib<ld::univariate_as_multivariate<ld::laplace2_distrib>>(
      ld::independent_product_component<ld::univariate_as_multivariate<ld::laplace2_distrib>>{{}, d, 1, nullptr}));
}

optimizer_vector point_of(const std::vector<double>& v) {
  optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

// a function of one variable given as a callable, value only
template <class F>
struct scalar_objective {
  F f;
  long calls = 0;
  double value(const optimizer_point& x) {
    ++calls;
    return f(x(0));
  }
};
template <class F>
scalar_objective<F> scalar(F f) {
  return {f};
}

// f(x) = 0.5 x'Ax - b'x + c sum_i log(1 + exp(x_i)), A symmetric positive definite: smooth,
// convex, not quadratic; separable when A is diagonal
struct smooth_test_objective {
  using curvature_type = lo::dense_curvature_matrix;
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  double c = 1.0;

  int n() const { return static_cast<int>(b.size()); }
  double value(const optimizer_point& x) const {
    double f = 0.0;
    for (int i = 0; i < n(); ++i) {
      double ax = 0.0;
      for (int j = 0; j < n(); ++j) ax += A[i][j] * x(j);
      f += 0.5 * x(i) * ax - b[i] * x(i) + c * std::log1p(std::exp(x(i)));
    }
    return f;
  }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) const {
    for (int i = 0; i < n(); ++i) {
      double ax = 0.0;
      for (int j = 0; j < n(); ++j) ax += A[i][j] * x(j);
      g(i) = ax - b[i] + c / (1.0 + std::exp(-x(i)));
    }
    return value(x);
  }
  double value_gradient_curvature(const optimizer_point& x, const optimizer_vector& g, curvature_type& H) const {
    for (int i = 0; i < n(); ++i)
      for (int j = 0; j < n(); ++j) {
        const double p = 1.0 / (1.0 + std::exp(-x(i)));
        H(i, j) = A[i][j] + (i == j ? c * p * (1.0 - p) : 0.0);
      }
    return value_gradient(x, g);
  }
};

// f(x1, x2) = |x1 - x2| + 0.1 (x1 + x2 - 2)^2: minimum 0 on (1, 1); from (0, 0) each
// coordinate alone is at its minimum (one-sided slopes -1.4 and 0.6), so a kink coupling
// the two blocks holds the alternation there. The gradient is a subgradient (sign 0 at
// the kink), and the curvature that of the smooth part.
struct coupled_kink_objective {
  using curvature_type = lo::dense_curvature_matrix;
  static double sign(double t) { return t > 0.0 ? 1.0 : t < 0.0 ? -1.0 : 0.0; }
  double value(const optimizer_point& x) const {
    const double s = x(0) + x(1) - 2.0;
    return std::abs(x(0) - x(1)) + 0.1 * s * s;
  }
  double value_gradient(const optimizer_point& x, const optimizer_vector& g) const {
    const double s = x(0) + x(1) - 2.0, k = sign(x(0) - x(1));
    g(0) = k + 0.2 * s;
    g(1) = -k + 0.2 * s;
    return value(x);
  }
  double value_gradient_curvature(const optimizer_point& x, const optimizer_vector& g, curvature_type& H) const {
    H(0, 0) = H(1, 1) = H(0, 1) = H(1, 0) = 0.2;
    return value_gradient(x, g);
  }
};

void print_report(const char* what, const lo::optimizer_report& r) {
  if (!verbose) return;
  std::printf("%-40s status %d sweeps/iterations %d f %ld g %ld H %ld decrement %.3g search %.3g  %.3f ms\n", what,
              static_cast<int>(r.status), r.iterations, r.value_evaluations, r.gradient_evaluations,
              r.curvature_evaluations, r.decrement, r.search_size, 1e3 * r.elapsed_seconds);
}

// ---------------------------------------------------------------- Brent

void brent_cases() {
  const lo::brent_optimizer<> brent;
  {  // smooth: (x - 2)^2 + exp(-x), stationary where 2 (x - 2) = exp(-x)
    auto obj = scalar([](double x) { return (x - 2.0) * (x - 2.0) + std::exp(-x); });
    double xs = 2.0;
    for (int k = 0; k < 50; ++k) xs -= (2.0 * (xs - 2.0) - std::exp(-xs)) / (2.0 + std::exp(-xs));
    const auto r = lo::minimize(brent, obj, point_of({0.0}));
    print_report("brent smooth", r);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(std::abs(r.point(0) - xs) <= 2.0 * r.search_size);
    CHECK(r.value_evaluations == obj.calls);
    CHECK(r.gradient.extent(0) == 0);
    CHECK(std::isnan(r.decrement));
  }
  {  // kinked: sum |y_i - x|, minimum at the median 3 of the y below
    const std::vector<double> y = {7.0, -1.0, 3.0, 2.5, 10.0, 3.2, 0.0};
    auto obj = scalar([&](double x) {
      double s = 0.0;
      for (double v : y) s += std::abs(v - x);
      return s;
    });
    const auto r = lo::minimize(brent, obj, point_of({-20.0}));
    print_report("brent median", r);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(std::abs(r.point(0) - 3.0) <= 2.0 * r.search_size);
    CHECK(r.search_size <= 1e-7);
  }
  {  // refused points: x - log x is NaN for x <= 0 and the expansion crosses 0; minimum at 1
    auto obj = scalar([](double x) { return x > 0.0 ? x - std::log(x) : std::numeric_limits<double>::quiet_NaN(); });
    const auto r = lo::minimize(brent, obj, point_of({3.0}));
    print_report("brent refused", r);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(std::abs(r.point(0) - 1.0) <= 2.0 * r.search_size);
  }
  {  // no minimum: the expansion runs out, never converged
    auto obj = scalar([](double x) { return -x; });
    const auto r = lo::minimize(brent, obj, point_of({0.0}));
    CHECK(r.status == lo::optimizer_status::iteration_budget);
    CHECK(std::isnan(r.search_size));
    lo::brent_optimizer<> few;
    few.max_value_evaluations = 10;
    CHECK(lo::minimize(few, obj, point_of({0.0})).status == lo::optimizer_status::evaluation_budget);
  }
  {  // the rule: converged on the bracket half-width, in the units of x
    lo::search_size_stop_rule rule{.absolute_tolerance = 1e-3, .relative_tolerance = 1e-3};
    lo::optimizer_vector x = point_of({100.0});
    lo::stop_rule_input in;
    in.point = x;
    CHECK(rule.verdict(in) == lo::stop_verdict::none);  // NaN size
    in.search_size = 0.101;
    CHECK(rule.verdict(in) == lo::stop_verdict::converged);
    in.search_size = 0.102;
    CHECK(rule.verdict(in) == lo::stop_verdict::none);
    lo::brent_optimizer<> loose;
    loose.stop_rule = rule;
    auto obj = scalar([](double t) { return (t - 100.0) * (t - 100.0); });
    const auto r = lo::minimize(loose, obj, point_of({0.0}));
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(r.search_size <= 1e-3 + 1e-3 * std::abs(r.point(0)));
    CHECK(std::abs(r.point(0) - 100.0) <= 2.0 * r.search_size);
  }
  {  // a start that cannot be evaluated, and a point of the wrong size
    auto obj = scalar([](double x) { return std::log(x); });
    CHECK(lo::minimize(brent, obj, point_of({-1.0})).status == lo::optimizer_status::start_not_admissible);
    CHECK_THROWS_AS(lo::minimize(brent, obj, point_of({1.0, 2.0})), std::invalid_argument);
  }
}

// ---------------------------------------------------------------- the Laplace

// laplace1 (mu, sigma) and laplace2 (mu, lambda = 1 / sigma), free coordinates (mu, log scale)
template <class D>
void laplace_case(long n, std::uint64_t seed) {
  const D truth{1.5, 2.0};
  const auto y = fb::simulate(truth, n, seed);
  fb::fit_distrib_objective<D> objective(y, fb::fit_curvature::expected);

  std::vector<double> ys = fb::to_host_vector(y);
  std::sort(ys.begin(), ys.end());
  const long m = n / 2;
  const double lower = n % 2 ? ys[m] : ys[m - 1], upper = ys[m];  // the median interval
  double mad = 0.0;
  for (double v : ys) mad += std::abs(v - lower);
  mad /= static_cast<double>(n);  // the same at every point of the interval

  // a poor start: +1 on every free coordinate from the library's start
  const auto t0 = fb::library_start(D{}, ys);
  optimizer_vector u0("u0", 2);
  objective.free_of(t0.data(), u0);
  u0(0) += 1.0, u0(1) += 1.0;

  lo::block_coordinate_optimizer<lo::brent_optimizer<>, lo::newton_optimizer<>> alternation;
  alternation.block_sizes = {1, 1};
  const auto r = lo::minimize(alternation, objective, u0);
  print_report(n % 2 ? "laplace alternation, odd n" : "laplace alternation, even n", r);

  REQUIRE(lo::optimizer_succeeded(r.status));
  double theta[2];
  REQUIRE(objective.theta_of(r.point, theta));
  const double sigma = std::is_same_v<D, ld::laplace2_distrib> ? 1.0 / theta[1] : theta[1];
  const double slack = 2.0 * r.search_size + 1e-12 * std::max(1.0, std::abs(lower));
  CHECK(theta[0] >= lower - slack);
  CHECK(theta[0] <= upper + slack);
  // the decrement tol bounds the relative error of the scale by about sqrt(2 tol / n)
  CHECK(std::abs(sigma / mad - 1.0) <= 2.0 * std::sqrt(2.0 * 1e-6 / static_cast<double>(n)) + 1e-10);
  CHECK(r.decrement <= 1e-6);
  if (verbose)
    std::printf("  n %ld mu %.17g interval [%.17g, %.17g] sigma %.17g mad %.17g\n", n, theta[0], lower, upper, sigma,
                mad);
}

void laplace_cases() {
  for (long n : {1001L, 1000L, 100001L, 100000L}) {
    laplace_case<ld::laplace1_distrib>(n, 20261118u + static_cast<std::uint64_t>(n));
    laplace_case<ld::laplace2_distrib>(n, 20261119u + static_cast<std::uint64_t>(n));
  }
}

// ---------------------------------------------------------------- smooth problems

void compare_with_newton(smooth_test_objective objective, const std::vector<int>& sizes, const char* what) {
  const auto x0 = point_of(std::vector<double>(objective.b.size(), 3.0));
  const auto joint = lo::minimize(lo::newton_optimizer<>{}, objective, x0);
  REQUIRE(lo::optimizer_succeeded(joint.status));

  lo::block_coordinate_optimizer<lo::newton_optimizer<>, lo::bfgs_optimizer<>> alternation;
  alternation.block_sizes = {sizes[0], sizes[1]};
  const auto r = lo::minimize(alternation, objective, x0);
  print_report(what, r);
  REQUIRE(lo::optimizer_succeeded(r.status));
  // both stop on decrements of at most 1e-6 per block, so f - f* is a few 1e-6
  CHECK(r.value - joint.value <= 1e-5);
  CHECK(joint.value - r.value <= 1e-5);
  for (std::size_t i = 0; i < objective.b.size(); ++i) CHECK(std::abs(r.point(i) - joint.point(i)) <= 1e-2);
}

void smooth_cases() {
  smooth_test_objective separable{{{2, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 4, 0}, {0, 0, 0, 0.5}}, {1, -2, 3, 0.5}};
  compare_with_newton(separable, {2, 2}, "separable, blocks 2 + 2");
  smooth_test_objective coupled{{{4, 1.5, 1, 0.5}, {1.5, 3, 0.8, 1}, {1, 0.8, 2, 0.6}, {0.5, 1, 0.6, 1.5}},
                                {1, -2, 3, 0.5}};
  compare_with_newton(coupled, {1, 3}, "coupled, blocks 1 + 3");
  compare_with_newton(coupled, {3, 1}, "coupled, blocks 3 + 1");
}

// ---------------------------------------------------------------- negative case

void coupled_kink_case() {
  coupled_kink_objective objective;
  lo::block_coordinate_optimizer<lo::newton_optimizer<>, lo::newton_optimizer<>> alternation;
  alternation.block_sizes = {1, 1};
  const auto r = lo::minimize(alternation, objective, point_of({0.0, 0.0}));
  print_report("coupled kink, Newton blocks", r);
  CHECK_FALSE(lo::optimizer_succeeded(r.status));
  CHECK(r.status == lo::optimizer_status::stalled);
  CHECK(r.value > 0.39);  // held at (0, 0), f = 0.4, where the minimum is 0

  // The known limit, pinned: with Brent on both blocks, each coordinate alone is at its
  // minimum at (0, 0), and the run is declared converged there, f = 0.4 against 0 at (1, 1).
  lo::block_coordinate_optimizer<lo::brent_optimizer<>, lo::brent_optimizer<>> brent_blocks;
  brent_blocks.block_sizes = {1, 1};
  const auto rb = lo::minimize(brent_blocks, objective, point_of({0.0, 0.0}));
  print_report("coupled kink, Brent blocks (known limit)", rb);
  CHECK(lo::optimizer_succeeded(rb.status));
  CHECK(rb.iterations == 1);
  CHECK(std::abs(rb.point(0)) <= 1e-7);
  CHECK(std::abs(rb.point(1)) <= 1e-7);
  CHECK(std::abs(rb.value - 0.4) <= 1e-7);

  lo::block_coordinate_optimizer<lo::newton_optimizer<>> one_block;
  one_block.block_sizes = {3};
  CHECK_THROWS_AS(lo::minimize(one_block, objective, point_of({0.0, 0.0})), std::invalid_argument);
}

// ---------------------------------------------------------------- a composite block

// The lasso on the slopes of a gaussian regression (intercept unpenalized, column 0 of X
// the constant), by alternation: the intercept by Newton, the slopes by FISTA or by
// proximal Newton, against proximal Newton on the whole of x.
template <class SlopeOptimizer>
void lasso_block_case(const char* what, SlopeOptimizer slopes) {
  const auto pr = regression_test::make_regression_problem(500, 21, 6, 1.0, 20261118);
  regression_test::regression_test_objective<> smooth(pr.x, pr.y, pr.n, pr.p, regression_test::regression_loss::gaussian);
  lo::penalized_objective<regression_test::regression_test_objective<>, lasso_penalty> objective(
      smooth, pr.p, lo::attach_penalty(make_lasso(pr.p - 1), 1, {60.0}));
  optimizer_vector x0("x0", pr.p);  // zeros

  lo::proximal_newton_optimizer<> joint;
  joint.stop_rule.tolerance = 1e-14;
  const auto rj = lo::minimize(joint, objective, x0);
  REQUIRE(lo::optimizer_succeeded(rj.status));

  lo::newton_optimizer<> intercept;
  intercept.stop_rule.tolerance = 1e-12;
  slopes.stop_rule.tolerance = 1e-12;
  lo::block_coordinate_optimizer<lo::newton_optimizer<>, SlopeOptimizer> alternation{{intercept, slopes}, {1, pr.p - 1}};
  const auto r = lo::minimize(alternation, objective, x0);
  print_report(what, r);
  REQUIRE(lo::optimizer_succeeded(r.status));

  double scale = 0.0;
  for (int j = 0; j < pr.p; ++j) scale = std::max(scale, std::abs(rj.point(j)));
  int support = 0;
  for (int j = 0; j < pr.p; ++j) {
    CHECK(std::abs(r.point(j) - rj.point(j)) <= 1e-5 * scale);
    if (j > 0) {
      CHECK((r.point(j) == 0.0) == (rj.point(j) == 0.0));
      support += rj.point(j) != 0.0;
    }
  }
  CHECK(support > 0);
  CHECK(support < pr.p - 1);
  CHECK(std::abs(r.value - rj.value) <= 1e-8 * std::abs(rj.value));
  if (verbose) std::printf("  support %d of %d, f %.17g joint %.17g\n", support, pr.p - 1, r.value, rj.value);
}

void composite_cases() {
  lasso_block_case("lasso block by FISTA", lo::proximal_gradient_optimizer<>{});
  lasso_block_case("lasso block by proximal Newton", lo::proximal_newton_optimizer<>{});

  // refusals: a penalty that straddles two blocks, and a block whose sub-optimizer ignores its penalty
  const auto pr = regression_test::make_regression_problem(50, 5, 2, 1.0, 3);
  regression_test::regression_test_objective<> smooth(pr.x, pr.y, pr.n, pr.p, regression_test::regression_loss::gaussian);
  lo::penalized_objective<regression_test::regression_test_objective<>, lasso_penalty> objective(
      smooth, pr.p, lo::attach_penalty(make_lasso(pr.p - 1), 1, {1.5}));
  optimizer_vector x0("x0", pr.p);
  lo::block_coordinate_optimizer<lo::newton_optimizer<>, lo::proximal_newton_optimizer<>> straddling;
  straddling.block_sizes = {2, 3};
  CHECK_THROWS_AS(lo::minimize(straddling, objective, x0), std::invalid_argument);
  lo::block_coordinate_optimizer<lo::newton_optimizer<>, lo::newton_optimizer<>> ignoring;
  ignoring.block_sizes = {1, 4};
  CHECK_THROWS_AS(lo::minimize(ignoring, objective, x0), std::invalid_argument);
}

}  // namespace

// named functions with external linkage that run the cases (CLAUDE.md trap)
void block_coordinate_composite_cases() { composite_cases(); }
void block_coordinate_brent_cases() { brent_cases(); }
void block_coordinate_laplace_cases() { laplace_cases(); }
void block_coordinate_smooth_cases() { smooth_cases(); }
void block_coordinate_kink_case() { coupled_kink_case(); }

TEST_CASE("brent_optimizer: smooth, kinked, refused and unbounded functions of one variable") {
  block_coordinate_brent_cases();
}
TEST_CASE("block_coordinate_optimizer: the Laplace by alternation against its closed form") {
  block_coordinate_laplace_cases();
}
TEST_CASE("block_coordinate_optimizer: smooth separable and coupled problems against Newton") {
  block_coordinate_smooth_cases();
}
TEST_CASE("block_coordinate_optimizer: a kink coupling two blocks, Newton blocks stall, Brent blocks hit the known limit") {
  block_coordinate_kink_case();
}
TEST_CASE("block_coordinate_optimizer: a lasso block by FISTA and by proximal Newton against the joint fit") {
  block_coordinate_composite_cases();
}
