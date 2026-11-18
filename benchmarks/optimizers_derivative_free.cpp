// Nelder-Mead (nelder_mead.hpp) and compass search (compass.hpp, random and
// coordinate polls) on the battery of tests/optimizers/test_problems.hpp at
// f* = 0, and the Laplace (laplace1) of the fit_distrib bench by alternation,
// mu by Brent, Nelder-Mead or compass and log sigma by Newton on the expected
// curvature, at n = 1e3 and 1e5. Reported: status, iterations, evaluations of
// f, time per run (fastest of repeats) and the distance from the minimizer
// (for the Laplace, of mu from the median interval). Backend of the build
// (OpenMP threads from OMP_NUM_THREADS; CUDA) for the Laplace's reductions.
//
// Usage: optimizers_derivative_free

#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/compass.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/nelder_mead.hpp>
#include <laplace/optimizers/newton.hpp>

#include "../tests/optimizers/fit_distrib_data.hpp"
#include "../tests/optimizers/test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

namespace fb = fit_distrib_bench;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace tp = optimizer_test_problems;
using clock_type = std::chrono::steady_clock;

template <class F>
double fastest(F&& f) {
  double best = 1e300, total = 0.0;
  for (int r = 0; r < 200 && (r < 3 || total < 0.2); ++r) {
    const auto t0 = clock_type::now();
    f();
    const double s = std::chrono::duration<double>(clock_type::now() - t0).count();
    best = std::min(best, s);
    total += s;
  }
  return best;
}

lo::optimizer_vector point_of(const std::vector<double>& v) {
  lo::optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

template <class Optimizer, class P>
void battery_row(const char* method, const Optimizer& optimizer, P problem) {
  const auto x0 = point_of(problem.x0);
  lo::optimizer_report r;
  const double s = fastest([&] { r = lo::minimize(optimizer, problem, x0); });
  double dx = 0.0;
  for (std::size_t i = 0; i < problem.x_star.size(); ++i) dx = std::max(dx, std::abs(r.point(i) - problem.x_star[i]));
  std::printf("%-20s %-28s %2d %6d %7ld %10.3f %10.2e\n", method, problem.name.c_str(), static_cast<int>(r.status),
              r.iterations, r.value_evaluations, 1e3 * s, dx);
}

template <class Optimizer>
void battery(const char* method, const Optimizer& optimizer) {
  battery_row(method, optimizer, tp::extended_rosenbrock_problem(2));
  battery_row(method, optimizer, tp::extended_rosenbrock_problem(4));
  battery_row(method, optimizer, tp::powell_singular_problem());
  battery_row(method, optimizer, tp::beale_problem());
  battery_row(method, optimizer, tp::helical_valley_problem());
  battery_row(method, optimizer, tp::wood_problem());
  battery_row(method, optimizer, tp::ill_conditioned_quadratic_problem(4, 1e2));
  battery_row(method, optimizer, tp::ill_conditioned_quadratic_problem(4, 1e4));
  battery_row(method, optimizer, tp::ill_conditioned_quadratic_problem(10, 1e2));
}

template <class MuOptimizer>
void laplace_row(const char* method, const MuOptimizer& mu, long n) {
  const auto y = fb::simulate(ld::laplace1_distrib{1.5, 2.0}, n, 7u + static_cast<std::uint64_t>(n));
  std::vector<double> ys = fb::to_host_vector(y);
  std::sort(ys.begin(), ys.end());
  const double lower = n % 2 ? ys[n / 2] : ys[n / 2 - 1], upper = ys[n / 2];
  fb::fit_distrib_objective<ld::laplace1_distrib> objective(y, fb::fit_curvature::expected);
  const auto t0 = fb::library_start(ld::laplace1_distrib{}, ys);
  lo::optimizer_vector u0("u0", 2);
  objective.free_of(t0.data(), u0);
  u0(0) += 1.0, u0(1) += 1.0;
  lo::block_coordinate_optimizer<MuOptimizer, lo::newton_optimizer<>> alternation{{mu, {}}, {1, 1}};
  lo::optimizer_report r;
  const double s = fastest([&] { r = lo::minimize(alternation, objective, u0); });
  const double outside = std::max({0.0, lower - r.point(0), r.point(0) - upper});
  std::printf("%-20s laplace1 n = %-15ld %2d %6d %7ld %10.3f %10.2e\n", method, n, static_cast<int>(r.status),
              r.iterations, r.value_evaluations, 1e3 * s, outside);
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("backend %s\n", Kokkos::DefaultExecutionSpace::name());
    std::printf("%-20s %-28s %2s %6s %7s %10s %10s\n", "method", "problem", "st", "iter", "f", "ms", "|x-x*|");
    battery("nelder-mead", lo::nelder_mead_optimizer<>{});
    battery("compass random", lo::compass_optimizer<>{});
    lo::compass_optimizer<> coordinate;
    coordinate.poll_basis = lo::compass_poll_basis::coordinate;
    battery("compass coordinate", coordinate);
    for (long n : {1000L, 100000L}) {
      laplace_row("brent", lo::brent_optimizer<>{}, n);
      laplace_row("nelder-mead", lo::nelder_mead_optimizer<>{}, n);
      laplace_row("compass random", lo::compass_optimizer<>{}, n);
    }
  }
  Kokkos::finalize();
  return 0;
}
