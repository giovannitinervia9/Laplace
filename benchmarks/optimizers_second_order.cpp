// Newton and the trust region, measured on the battery of
// tests/optimizers/test_problems.hpp (host code, one thread): status,
// iterations, evaluations of f, g and H, f - f*, the final decrement and the
// time per run. Newton with three maximum steps (none, 5, and 1e3 max(1,
// max|x0_i|) as Dennis and Schnabel's default maxstep in the max norm); the
// trust region with initial radius 1 and 10.
//
// Usage: optimizers_second_order

#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>
#include <laplace/optimizers/trust_region.hpp>

#include "../tests/optimizers/test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;

const char* status_label(lo::optimizer_status s) {
  switch (s) {
    case lo::optimizer_status::converged: return "conv";
    case lo::optimizer_status::converged_at_resolution: return "conv@res";
    case lo::optimizer_status::stopped_on_change: return "change";
    case lo::optimizer_status::stalled: return "stalled";
    case lo::optimizer_status::iteration_budget: return "iter";
    case lo::optimizer_status::evaluation_budget: return "evals";
    case lo::optimizer_status::line_search_failed: return "ls-fail";
    default: return "start";
  }
}

// one row: the run, then its time per call (repeats until 0.1 s, at least 3)
template <class Optimizer, class P>
void row(const char* method, const Optimizer& optimizer, P& p) {
  lo::optimizer_vector x0("x0", p.x0.size());
  for (std::size_t i = 0; i < p.x0.size(); ++i) x0(i) = p.x0[i];
  const lo::optimizer_report r = lo::minimize(optimizer, p, x0);
  using clock = std::chrono::steady_clock;
  int calls = 0;
  const auto start = clock::now();
  double elapsed = 0.0;
  do {
    (void)lo::minimize(optimizer, p, x0);
    ++calls;
    elapsed = std::chrono::duration<double, std::micro>(clock::now() - start).count();
  } while (elapsed < 1e5 || calls < 3);
  std::printf("%-34s %-14s %-9s %5d %6ld %5ld %5ld %10.2e %10.2e %10.1f\n", (p.name + " + " + std::to_string(int(p.offset))).c_str(),
              method, status_label(r.status), r.iterations, r.value_evaluations, r.gradient_evaluations,
              r.curvature_evaluations, r.value - p.f_star, r.decrement, elapsed / calls);
}

template <class P>
void all_methods(P& p) {
  double x0max = 0.0;
  for (double v : p.x0) x0max = std::max(x0max, std::abs(v));
  lo::newton_optimizer<> newton;
  row("Newton", newton, p);
  newton.max_step = 5.0;
  row("Newton ms 5", newton, p);
  newton.max_step = 1e3 * std::max(1.0, x0max);
  row("Newton ms DS", newton, p);
  lo::trust_region_optimizer<> region;
  row("TR r0 1", region, p);
  region.initial_radius = 10.0;
  row("TR r0 10", region, p);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  std::printf("%-34s %-14s %-9s %5s %6s %5s %5s %10s %10s %10s\n", "problem", "method", "status", "iter", "f", "g",
              "H", "f - f*", "decrement", "us/run");
  for (double offset : {0.0, 1e6}) {
    tp::extended_rosenbrock_problem a(2, offset), b(100, offset);
    tp::powell_singular_problem c(offset);
    tp::beale_problem d(offset);
    tp::brown_badly_scaled_problem e(offset);
    tp::helical_valley_problem f(offset);
    tp::wood_problem g(offset);
    all_methods(a), all_methods(b), all_methods(c), all_methods(d), all_methods(e), all_methods(f), all_methods(g);
  }
  for (double kappa : {1e2, 1e6, 1e12}) {
    tp::ill_conditioned_quadratic_problem q(10, kappa);
    all_methods(q);
  }
  tp::saddle_problem s;
  all_methods(s);
  tp::extended_rosenbrock_problem r(2);
  r.name = "Rosenbrock from (0, 1)";
  r.x0 = {0.0, 1.0};
  all_methods(r);
  return 0;
}
