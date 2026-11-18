// Block alternation (block_coordinate.hpp) against joint Fisher scoring on the
// Laplace (laplace1, laplace2) of the fit_distrib bench: mu by Brent and the
// scale by Newton on the expected curvature, against Newton on the expected
// curvature of both, from the library start plus 1 on each free coordinate,
// at n = 1e3, 1e5, 1e6 drawn in C++. Reported: status, sweeps or iterations,
// reductions (passes over the data), time per fit (fastest of repeats), and
// the distance of mu from the median interval and of the scale from its
// closed form. Backend of the build (OpenMP threads from OMP_NUM_THREADS; CUDA).
//
// Usage: optimizers_block_coordinate

#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>

#include "../tests/optimizers/fit_distrib_data.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace {

namespace fb = fit_distrib_bench;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
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

template <class D>
void bench(const char* name, long n) {
  const auto y = fb::simulate(D{1.5, 2.0}, n, 7u + static_cast<std::uint64_t>(n));
  std::vector<double> ys = fb::to_host_vector(y);
  std::sort(ys.begin(), ys.end());
  const double lower = n % 2 ? ys[n / 2] : ys[n / 2 - 1], upper = ys[n / 2];
  double mad = 0.0;
  for (double v : ys) mad += std::abs(v - lower);
  mad /= static_cast<double>(n);

  fb::fit_distrib_objective<D> objective(y, fb::fit_curvature::expected);
  const auto t0 = fb::library_start(D{}, ys);
  lo::optimizer_vector u0("u0", 2);
  objective.free_of(t0.data(), u0);
  u0(0) += 1.0, u0(1) += 1.0;

  const auto row = [&](const char* method, const auto& optimizer) {
    objective.passes = 0;
    const auto r = lo::minimize(optimizer, objective, u0);
    const long passes = objective.passes;
    const double seconds = fastest([&] { lo::minimize(optimizer, objective, u0); });
    double t[2];
    objective.theta_of(r.point, t);
    const double scale = std::is_same_v<D, ld::laplace2_distrib> ? 1.0 / t[1] : t[1];
    const double off = t[0] < lower ? lower - t[0] : t[0] > upper ? t[0] - upper : 0.0;
    std::printf("%-9s %8ld  %-22s status %d  it %3d  passes %4ld  %9.3f ms  mu off interval %.2e  scale rel %.2e\n",
                name, n, method, static_cast<int>(r.status), r.iterations, passes, 1e3 * seconds, off,
                std::abs(scale / mad - 1.0));
  };
  lo::block_coordinate_optimizer<lo::brent_optimizer<>, lo::newton_optimizer<>> alternation;
  alternation.block_sizes = {1, 1};
  row("Brent + Newton", alternation);
  row("Newton (expected)", lo::newton_optimizer<>{});
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::printf("execution space %s\n", Kokkos::DefaultExecutionSpace::name());
  for (long n : {1000L, 100000L, 1000000L}) {
    bench<ld::laplace1_distrib>("laplace1", n);
    bench<ld::laplace2_distrib>("laplace2", n);
  }
  return 0;
}
