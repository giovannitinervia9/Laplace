// Short bench of the proximal methods (note/optimizers.md, section 12) on the
// regression test objective: the lasso and the elastic net with a gaussian
// likelihood, n rows and p columns (an unpenalized intercept and p - 1
// penalized slopes), FISTA against proximal Newton with coordinate descent and
// with FISTA inside; also the lasso on equicorrelated columns (correlation
// 0.9, label lasso9). Prints status, iterations, evaluations of s, the size of
// the support, the inner sweeps of proximal Newton's coordinate descent, the
// time, the declared stationarity measure and f - f* (f* the least value of the
// four runs). The sums over rows run on the default
// execution space (OpenMP threads or CUDA).
//
//   optimizers_proximal [n ...]     (default 1000 10000 100000)

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include "../tests/optimizers/regression_objective.hpp"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
template <class D>
using product_penalty = lpen::distrib_penalty<ld::independent_product_distrib<uam<D>>>;
using objective_type = regression_test::regression_test_objective<>;

template <class D>
product_penalty<D> make_penalty(int d) {
  return product_penalty<D>(ld::independent_product_distrib<uam<D>>(ld::independent_product_component<uam<D>>{{}, d, 1, nullptr}));
}

// forwards everything to O and counts the coordinate proxes (inner sweeps of
// proximal Newton = calls / penalized coordinates)
template <class O>
struct counting_objective {
  using curvature_type = typename O::curvature_type;
  O& o;
  long coordinate_proxes = 0;
  double value(const lo::optimizer_point& x) { return o.value(x); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) { return o.value_gradient(x, g); }
  double value_gradient_curvature(const lo::optimizer_point& x, const lo::optimizer_vector& g, curvature_type& H) {
    return o.value_gradient_curvature(x, g, H);
  }
  double nonsmooth_value(const lo::optimizer_point& x) { return o.nonsmooth_value(x); }
  void nonsmooth_prox(const lo::optimizer_point& v, double t, const lo::optimizer_vector& z) { o.nonsmooth_prox(v, t, z); }
  int nonsmooth_block_count() const { return o.nonsmooth_block_count(); }
  double weak_convexity_constant(int b) { return o.weak_convexity_constant(b); }
  std::pair<int, int> nonsmooth_block_range(int b) { return o.nonsmooth_block_range(b); }
  bool nonsmooth_block_separable(int b) { return o.nonsmooth_block_separable(b); }
  double nonsmooth_coordinate_prox(int b, int j, double v, double t) {
    ++coordinate_proxes;
    return o.nonsmooth_coordinate_prox(b, j, v, t);
  }
};

struct bench_row {
  const char* method;
  lo::optimizer_report report;
  double sweeps;
};

template <class D>
void run(const char* label, int n, int p, std::vector<double> theta, double correlation = 0.0) {
  const auto pr = regression_test::make_regression_problem(n, p, 10, 1.0, 42, false, correlation);
  objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_test::regression_loss::gaussian);
  using penalized = lo::penalized_objective<objective_type, product_penalty<D>>;
  penalized objective(smooth, p, lo::attach_penalty(make_penalty<D>(p - 1), 1, std::move(theta)));
  lo::optimizer_vector x0("x0", p);
  lo::proximal_gradient_optimizer<> fista;
  fista.stop_rule.tolerance = 1e-8;
  lo::proximal_newton_optimizer<> newton;
  newton.stop_rule.tolerance = 1e-8;
  auto newton_loose = newton;
  newton_loose.inner_fraction = 0.1;
  auto newton_fista = newton;
  newton_fista.inner_solver = lo::proximal_newton_inner_solver::proximal_gradient;
  std::vector<bench_row> rows;
  auto counted = [&](const char* method, const auto& optimizer) {
    counting_objective<penalized> c{objective};
    const auto r = lo::minimize(optimizer, c, x0);
    rows.push_back({method, r, double(c.coordinate_proxes) / double(p - 1)});
  };
  counted("FISTA", fista);
  counted("proximal Newton (CD .1)", newton_loose);
  counted("proximal Newton (CD .01)", newton);
  counted("proximal Newton (FISTA)", newton_fista);
  double best = rows[0].report.value;
  for (const auto& r : rows) best = std::min(best, r.report.value);
  for (const auto& row : rows) {
    const auto& r = row.report;
    int support = 0;
    for (std::size_t j = 1; j < r.point.extent(0); ++j) support += r.point(j) != 0.0;
    std::printf("%-6s %7d %-24s %-4s %6d %7ld %7.0f %4d %9.3f %10.2e %10.2e\n", label, n, row.method,
                lo::optimizer_succeeded(r.status) ? "ok" : "FAIL", r.iterations, r.value_evaluations, row.sweeps, support,
                r.elapsed_seconds * 1e3, r.decrement, r.value - best);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::vector<int> sizes;
  for (int a = 1; a < argc; ++a)
    if (argv[a][0] != '-') sizes.push_back(std::atoi(argv[a]));
  if (sizes.empty()) sizes = {1000, 10000, 100000};
  const int p = 51;
  std::printf("backend %s, concurrency %d, p = %d, decrement / prox mapping tolerance 1e-8\n",
              Kokkos::DefaultExecutionSpace::name(), Kokkos::DefaultExecutionSpace().concurrency(), p);
  std::printf("%-6s %7s %-24s %-4s %6s %7s %7s %4s %9s %10s %10s\n", "pen", "n", "method", "st", "it", "s ev", "sweeps",
              "supp", "ms", "measure", "f - f*");
  for (const int n : sizes) {
    // lambda = 0.1 n (glmnet's lambda = 0.1); enet alpha = 0.5
    run<ld::laplace2_distrib>("lasso", n, p, {0.1 * n});
    run<ld::enet_distrib>("enet", n, p, {0.1 * n, 0.5});
    run<ld::laplace2_distrib>("lasso9", n, p, {0.02 * n}, 0.9);  // equicorrelated columns, correlation 0.9
  }
  return 0;
}
