// Short bench of primal_dual_optimizer (Condat-Vu, note/optimizers.md section
// 12) on s(x) + g(D x):
// - signal denoising, s = |y - x|^2 / 2 on the host, D the differences of
//   order K = 1, 2, 3 (fused lasso and trend filtering), p points; for K = 1
//   also FISTA with Condat's exact prox in beta;
// - the regression test objective (gaussian, n rows, 41 columns, intercept
//   unpenalized; sums on the default execution space), fused lasso against
//   FISTA and trend filtering of order 2.
// Prints status, iterations, evaluations of s, time, the declared measure and
// f - f_FISTA where FISTA runs.
//
//   optimizers_primal_dual [n ...]     (default 1000 10000 100000)

#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/primal_dual.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/penalty/linear_map_penalty.hpp>

#include "../tests/optimizers/regression_objective.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
template <class D>
using uam = ld::univariate_as_multivariate<D>;
using lasso_penalty = lpen::distrib_penalty<ld::independent_product_distrib<uam<ld::laplace2_distrib>>>;

lasso_penalty make_lasso(int d) {
  return lasso_penalty(ld::independent_product_distrib<uam<ld::laplace2_distrib>>(
      ld::independent_product_component<uam<ld::laplace2_distrib>>{{}, d, 1, nullptr}));
}

struct denoise_objective {
  std::vector<double> y;
  double value(const lo::optimizer_point& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) s += 0.5 * (y[i] - x(i)) * (y[i] - x(i));
    return s;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    for (std::size_t i = 0; i < y.size(); ++i) g(i) = x(i) - y[i];
    return value(x);
  }
};

void print(const char* label, int size, const lo::optimizer_report& r, double reference) {
  std::printf("%-22s %7d  status %d  it %6d  evals %7ld  %9.3f ms  V %.2e  f - f_fista %s", label, size, int(r.status),
              r.iterations, r.value_evaluations, 1e3 * r.elapsed_seconds, r.decrement, std::isnan(reference) ? "   -\n" : "");
  if (!std::isnan(reference)) std::printf("%.2e\n", r.value - reference);
}

template <int K, class O>
lo::optimizer_report run_pd(O& smooth, int n, int offset, double lambda) {
  const int p = n - offset;
  const lpen::linear_map_penalty<lasso_penalty, lpen::difference_map<K>> pen(make_lasso(p - K), lpen::difference_map<K>(p));
  lo::linear_map_penalized_objective<O, lasso_penalty, lpen::difference_map<K>> split(smooth, n,
                                                                                    lo::attach_penalty(pen, offset, {lambda}));
  lo::optimizer_vector x0("x0", n);
  return lo::minimize(lo::primal_dual_optimizer<>{}, split, x0);
}

template <class O>
double run_fista(O& smooth, int n, int offset, double lambda, const char* label, int size) {
  const int p = n - offset;
  using pen_type = lpen::linear_map_penalty<lasso_penalty, lpen::difference_map<1>>;
  const pen_type pen(make_lasso(p - 1), lpen::difference_map<1>(p));
  lo::penalized_objective<O, pen_type> composite(smooth, n, lo::attach_penalty(pen, offset, {lambda}));
  lo::optimizer_vector x0("x0", n);
  auto r = lo::minimize(lo::proximal_gradient_optimizer<>{}, composite, x0);
  print(label, size, r, std::nan(""));
  return r.value;  // FISTA reports s + r
}

void signal_cases() {
  for (int p : {200, 1000}) {
    regression_test::regression_rng rng{7u};
    for (int K = 1; K <= 3; ++K) {
      denoise_objective s{std::vector<double>(std::size_t(p))};
      for (int i = 0; i < p; ++i) {
        const double t = double(i) / (p - 1);
        s.y[std::size_t(i)] = (K == 1 ? (t < 0.3 ? 0.0 : t < 0.6 ? 2.0 : -1.0) : 3.0 * std::sin(6.0 * t)) + rng.normal();
      }
      const double lambda = K == 1 ? 5.0 : 20.0;
      double ref = std::nan("");
      if (K == 1) ref = run_fista(s, p, 0, lambda, "denoise K1 fista", p);
      if (K == 1) print("denoise K1 primal-dual", p, run_pd<1>(s, p, 0, lambda), ref);
      if (K == 2) print("denoise K2 primal-dual", p, run_pd<2>(s, p, 0, lambda), ref);
      if (K == 3) print("denoise K3 primal-dual", p, run_pd<3>(s, p, 0, lambda), ref);
    }
  }
}

void regression_cases(int n) {
  const auto pr = regression_test::make_regression_problem(n, 41, 10, 1.0, 21u, false, 0.5);
  regression_test::regression_test_objective<> smooth(pr.x, pr.y, pr.n, pr.p, regression_test::regression_loss::gaussian);
  const double lambda = 0.1 * n;
  const double ref = run_fista(smooth, pr.p, 1, lambda, "regr fused fista", n);
  print("regr fused primal-dual", n, run_pd<1>(smooth, pr.p, 1, lambda), ref);
  print("regr trend2 primal-dual", n, run_pd<2>(smooth, pr.p, 1, lambda), std::nan(""));
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    signal_cases();
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i)
      if (argv[i][0] != '-') sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1000, 10000, 100000};
    for (int n : sizes) regression_cases(n);
  }
  Kokkos::finalize();
}
