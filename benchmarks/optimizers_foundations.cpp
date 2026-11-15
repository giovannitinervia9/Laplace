// The foundations of laplace::optimizers, measured (host code, one thread).
// 1. laplace::dense::cholesky_condition_estimate: the ratio of the estimate to
//    the exact kappa_1 on A = Q diag(lambda) Q^T with known eigenvalues
//    (lambda log-spaced from 1 to kappa, Q the eigenvectors of a pseudo-random
//    symmetric matrix; exact kappa_1 from Q diag(1 / lambda) Q^T), and its time
//    against the Cholesky factorization and against the inverse by n solves.
// 2. The evaluations of the Armijo and strong Wolfe line searches on the
//    battery of tests/optimizers/test_problems.hpp, from x0 along steepest
//    descent and Newton directions, initial step 1.
//
// Usage: optimizers_foundations

#include <laplace/dense/dense.hpp>
#include <laplace/optimizers/dense_curvature_solver.hpp>
#include <laplace/optimizers/line_search.hpp>

#include "../tests/optimizers/test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

namespace ld = laplace::dense;
namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;
using matrix = lo::dense_curvature_matrix;

// ms per call: repeats until 0.2 s have passed (at least 3 calls)
template <class F>
double time_ms(F&& f) {
  using clock = std::chrono::steady_clock;
  int calls = 0;
  const auto start = clock::now();
  double elapsed = 0.0;
  do {
    f();
    ++calls;
    elapsed = std::chrono::duration<double, std::milli>(clock::now() - start).count();
  } while (elapsed < 200.0 || calls < 3);
  return elapsed / calls;
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

void spd_with_inverse(int n, double kappa, std::uint64_t seed, matrix& a, matrix& inverse) {
  tp::test_uniform u{seed};
  matrix s("s", n, n), q("q", n, n);
  Kokkos::View<double*, Kokkos::HostSpace> w("w", n);
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) s(i, j) = s(j, i) = u() - 0.5;
  ld::symmetric_eigen_decomposition(s, w, q);
  std::vector<double> lambda(n);
  for (int k = 0; k < n; ++k) lambda[k] = n > 1 ? std::pow(kappa, double(k) / (n - 1)) : 1.0;
  a = matrix("a", n, n), inverse = matrix("inverse", n, n);
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) {
      double sa = 0.0, si = 0.0;
      for (int k = 0; k < n; ++k) sa += q(i, k) * lambda[k] * q(j, k), si += q(i, k) / lambda[k] * q(j, k);
      a(i, j) = a(j, i) = sa, inverse(i, j) = inverse(j, i) = si;
    }
}

void condition_estimate() {
  std::printf("condition estimate / exact kappa_1 (min, max over 5 seeds)\n%6s", "n");
  const double kappas[] = {1e2, 1e6, 1e10, 1e13};
  for (double k : kappas) std::printf("  %16.0e", k);
  std::printf("\n");
  double low = 1e300, high = 0.0;
  for (int n : {5, 20, 100, 300}) {
    std::printf("%6d", n);
    for (double kappa : kappas) {
      double lo_ = 1e300, hi_ = 0.0;
      for (int seed = 0; seed < 5; ++seed) {
        matrix a, inverse, L("L", n, n);
        spd_with_inverse(n, kappa, 7919u * seed + n, a, inverse);
        if (!ld::cholesky_decomposition(a, L).success) continue;
        const double r = ld::cholesky_condition_estimate(a, L) / (one_norm(a) * one_norm(inverse));
        lo_ = std::min(lo_, r), hi_ = std::max(hi_, r);
      }
      std::printf("  %7.4f %7.4f ", lo_, hi_);
      low = std::min(low, lo_), high = std::max(high, hi_);
    }
    std::printf("\n");
  }
  std::printf("all: [%.4f, %.4f]\n\n", low, high);

  std::printf("%6s %14s %14s %14s\n", "n", "cholesky ms", "estimate ms", "inverse ms");
  for (int n : {10, 100, 500, 1000}) {
    matrix a, inverse, L("L", n, n), id("id", n, n);
    spd_with_inverse(n, 1e6, 17u + n, a, inverse);
    const double t_chol = time_ms([&] { ld::cholesky_decomposition(a, L); });
    volatile double sink = 0.0;
    const double t_est = time_ms([&] { sink = sink + ld::cholesky_condition_estimate(a, L); });
    const double t_inv = time_ms([&] {
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) id(i, j) = i == j;
      ld::cholesky_solve(L, id);
    });
    std::printf("%6d %14.4g %14.4g %14.4g\n", n, t_chol, t_est, t_inv);
  }
  std::printf("\n");
}

template <class P>
void line_search_counts(P p) {
  const int n = p.size();
  lo::optimizer_vector x("x", n), g("g", n), d("d", n), xt("xt", n), gt("gt", n);
  matrix H("H", n, n);
  for (int i = 0; i < n; ++i) x(i) = p.x0[i];
  const double f = p.value_gradient_curvature(x, g, H);
  for (int direction = 0; direction < 2; ++direction) {
    if (direction == 0) {
      for (int i = 0; i < n; ++i) d(i) = -g(i);
    } else {
      lo::dense_curvature_solver solver(n);
      if (solver.factorize(H) != lo::curvature_factorization::positive_definite) {
        std::printf("%-34s %-8s  (H not positive definite at x0)\n", p.name.c_str(), "Newton");
        continue;
      }
      solver.newton_direction(g, d);
    }
    const auto a = lo::armijo_line_search{}.search(p, x, f, g, d, 1.0, xt);
    const auto w = lo::wolfe_line_search{}.search(p, x, f, g, d, 1.0, xt, gt);
    std::printf("%-34s %-8s  Armijo %2d evals step %-10.3g  Wolfe %2d evals step %-10.3g %s\n", p.name.c_str(),
                direction == 0 ? "-g" : "Newton", a.value_evaluations, a.step, w.value_evaluations, w.step,
                a.accepted() && w.accepted() ? "" : "(failed)");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    condition_estimate();
    std::printf("line search evaluations from x0, initial step 1\n");
    line_search_counts(tp::extended_rosenbrock_problem(2));
    line_search_counts(tp::extended_rosenbrock_problem(100));
    line_search_counts(tp::powell_singular_problem());
    line_search_counts(tp::beale_problem());
    line_search_counts(tp::brown_badly_scaled_problem());
    line_search_counts(tp::helical_valley_problem());
    line_search_counts(tp::wood_problem());
    for (double kappa : {1e2, 1e6, 1e12}) line_search_counts(tp::ill_conditioned_quadratic_problem(10, kappa));
  }
  Kokkos::finalize();
}
