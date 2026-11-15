// BFGS, L-BFGS and multistart on the battery of tests/optimizers (host code,
// one thread): status, iterations, evaluations of f and g, f - f*, the
// distance max_i |x_i - x*_i|, the final decrement and the time of a run.
//
// Usage: optimizers_quasi_newton

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/lbfgs.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/multistart.hpp>

#include "../tests/optimizers/extra_problems.hpp"
#include "../tests/optimizers/test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;

const char* status_name(lo::optimizer_status s) {
  switch (s) {
    case lo::optimizer_status::converged: return "converged";
    case lo::optimizer_status::converged_at_resolution: return "at_resolution";
    case lo::optimizer_status::stopped_on_change: return "on_change";
    case lo::optimizer_status::stalled: return "stalled";
    case lo::optimizer_status::iteration_budget: return "iter_budget";
    case lo::optimizer_status::evaluation_budget: return "eval_budget";
    case lo::optimizer_status::line_search_failed: return "ls_failed";
    default: return "not_admissible";
  }
}

// microseconds per run: repeats until 0.1 s have passed (at least 3 runs)
template <class Optimizer, class Problem>
lo::optimizer_report measure(const Optimizer& opt, Problem p, double* us) {
  lo::optimizer_vector x0("x0", p.x0.size());
  for (std::size_t i = 0; i < p.x0.size(); ++i) x0(i) = p.x0[i];
  lo::optimizer_report r = lo::minimize(opt, p, x0);
  int runs = 0;
  const auto start = std::chrono::steady_clock::now();
  double elapsed = 0.0;
  do {
    (void)lo::minimize(opt, p, x0);
    ++runs;
    elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
  } while (elapsed < 1e5 || runs < 3);
  *us = elapsed / runs;
  return r;
}

template <class Problem>
void row(const char* method, const lo::optimizer_report& r, const Problem& p, double us) {
  double dist = 0.0;
  for (std::size_t i = 0; i < p.x_star.size(); ++i) dist = std::max(dist, std::abs(r.point(i) - p.x_star[i]));
  std::printf("%-28s %9.0e %-10s %-14s %5d %6ld %6ld %10.2e %10.2e %10.2e %10.1f\n", p.name.c_str(), p.offset,
              method, status_name(r.status), r.iterations, r.value_evaluations, r.gradient_evaluations,
              r.value - p.f_star, dist, r.decrement, us);
}

template <class Problem>
void all_methods(const Problem& p) {
  double us;
  lo::bfgs_optimizer<> bfgs;
  lo::bfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search> bfgs_armijo;
  lo::lbfgs_optimizer<> lbfgs5, lbfgs10, lbfgs20;
  lbfgs5.memory = 5, lbfgs20.memory = 20;
  lo::optimizer_report r;  // measured before row() reads us
  r = measure(bfgs, p, &us), row("bfgs", r, p, us);
  r = measure(bfgs_armijo, p, &us), row("bfgs armijo", r, p, us);
  r = measure(lbfgs5, p, &us), row("lbfgs m=5", r, p, us);
  r = measure(lbfgs10, p, &us), row("lbfgs m=10", r, p, us);
  r = measure(lbfgs20, p, &us), row("lbfgs m=20", r, p, us);
}

void header() {
  std::printf("%-28s %9s %-10s %-14s %5s %6s %6s %10s %10s %10s %10s\n", "problem", "offset", "method", "status",
              "iter", "f", "g", "f-f*", "|x-x*|", "decrement", "us");
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    header();
    for (double offset : {0.0, 1.0, 1e3, 1e6}) {
      all_methods(tp::extended_rosenbrock_problem(2, offset));
      all_methods(tp::extended_rosenbrock_problem(100, offset));
      all_methods(tp::powell_singular_problem(offset));
      all_methods(tp::beale_problem(offset));
      all_methods(tp::brown_badly_scaled_problem(offset));
      all_methods(tp::helical_valley_problem(offset));
      all_methods(tp::wood_problem(offset));
    }
    for (double kappa : {1e2, 1e4, 1e6, 1e8, 1e10, 1e12}) all_methods(tp::ill_conditioned_quadratic_problem(10, kappa));
    all_methods(tp::ill_conditioned_quadratic_problem(100, 1e6));

    std::printf("\nnegative cases (none may converge)\n");
    {
      lo::bfgs_optimizer<> bfgs;
      lo::lbfgs_optimizer<> lbfgs;
      auto show = [&](const char* name, const lo::optimizer_report& r) {
        std::printf("%-28s %-14s %5d %6ld %10.3e %10.2e\n", name, status_name(r.status), r.iterations,
                    r.value_evaluations, r.value, r.decrement);
      };
      for (int which = 0; which < 2; ++which) {
        tp::noisy_problem noisy(4);
        lo::optimizer_vector x0("x0", 4);
        Kokkos::deep_copy(x0, 1.0);
        show(which ? "noisy lbfgs" : "noisy bfgs", which ? lo::minimize(lbfgs, noisy, x0) : lo::minimize(bfgs, noisy, x0));
        tp::exploded_cosh_problem cosh(4);
        lo::optimizer_vector y0("y0", 4);
        Kokkos::deep_copy(y0, 700.0);
        auto b = bfgs;
        auto l = lbfgs;
        b.max_iterations = l.max_iterations = 100;
        show(which ? "cosh 700 lbfgs (100 it)" : "cosh 700 bfgs (100 it)",
             which ? lo::minimize(l, cosh, y0) : lo::minimize(b, cosh, y0));
        b.max_iterations = l.max_iterations = 1000;
        show(which ? "cosh 700 lbfgs (1000 it)" : "cosh 700 bfgs (1000 it)",
             which ? lo::minimize(l, cosh, y0) : lo::minimize(b, cosh, y0));
        tp::abs_sum_problem abs(4);
        lo::optimizer_vector z0("z0", 4);
        for (int i = 0; i < 4; ++i) z0(i) = abs.x0[i];
        show(which ? "abs_sum lbfgs" : "abs_sum bfgs", which ? lo::minimize(lbfgs, abs, z0) : lo::minimize(bfgs, abs, z0));
        lo::bfgs_optimizer<lo::decrement_stop_rule, lo::armijo_line_search> ba;
        if (!which) show("abs_sum bfgs armijo", lo::minimize(ba, abs, z0));
      }
    }

    std::printf("\nmultistart over bfgs on Styblinski-Tang (minima: n + 1 distinct values)\n");
    for (int n : {2, 4}) {
      tp::styblinski_tang_problem st(n);
      lo::optimizer_vector x0("x0", n);
      for (int i = 0; i < n; ++i) x0(i) = st.x0[i];
      for (int count : {1, 5, 10, 20, 40}) {
        lo::multistart_optimizer<lo::bfgs_optimizer<>> ms;
        ms.start_count = count;
        const auto t0 = std::chrono::steady_clock::now();
        const lo::multistart_report r = ms.run_multistart(st, x0);
        const double ms_time = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("n = %d starts %3d: converged %3d, distinct %2d, best found %3d, best - global %9.2e, f %6ld, g %6ld, %8.3f ms\n",
                    n, r.start_count, r.converged_count, r.distinct_optimum_count, r.best_found_count,
                    r.best.value - st.minimum_value(0), r.best.value_evaluations, r.best.gradient_evaluations, ms_time);
      }
    }
  }
  Kokkos::finalize();
  return 0;
}
