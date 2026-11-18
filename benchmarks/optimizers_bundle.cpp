// Short bench of the proximal bundle method (note/optimizers.md, section 12,
// decision 7) on least absolute deviations over the four wave-2 datasets with
// p < n (tests/data/reference/optimizers/wave2): bundle sizes n + 2 (the
// default), 2 n + 2 and 10, and the certificate ratios 10, 100 (the default),
// 1e3 and 1e5. Prints
// status, trial points (iterations), the time of the run (best of `reps`), the
// final stationarity measure and the distance to quantreg's vertex. The sums
// over rows run on the default execution space (OpenMP threads or CUDA).
//
//   optimizers_bundle [reps]     (default 3)

#include <laplace/optimizers/bundle.hpp>

#include "../tests/optimizers/regression_objective.hpp"
#include "../tests/optimizers/wave2_reference.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;

void run(int reps) {
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  std::printf("%-16s %-22s %6s %6s %9s %10s %10s %10s\n", "dataset", "variant", "status", "it", "seconds", "measure",
              "|b-vertex|", "f-f*");
  for (const auto& row : wave2_reference::read_rows(dir + "/rq_fits.csv")) {
    const auto d = wave2_reference::load_dataset(dir, row[0]);
    const double objective_reference = wave2_reference::to_double(row[3]);
    std::vector<double> vertex;
    for (const auto& c : wave2_reference::read_rows(dir + "/rq.csv"))
      if (c[0] == row[0]) vertex.push_back(wave2_reference::to_double(c[2]));
    const int p = d.p + 1;
    std::vector<double> x(std::size_t(d.n) * std::size_t(p), 1.0);
    std::copy(d.x.begin(), d.x.end(), x.begin() + d.n);
    regression_test::regression_test_objective<> lad(x, d.y_lad, d.n, p, regression_test::regression_loss::laplace);
    struct variant {
      const char* label;
      int size;
      double ratio;
    };
    for (const variant v : {variant{"bundle n+2", 0, 100.0}, variant{"bundle 2n+2", 2 * p + 2, 100.0},
                            variant{"bundle 10", 10, 100.0}, variant{"certificate ratio 10", 0, 10.0},
                            variant{"certificate ratio 1e3", 0, 1e3}, variant{"certificate ratio 1e5", 0, 1e5}}) {
      lo::bundle_optimizer<> bundle;
      bundle.max_bundle_size = v.size;
      bundle.certificate_ratio = v.ratio;
      bundle.max_iterations = 20000;
      lo::optimizer_report best;
      for (int r = 0; r < reps; ++r) {
        lo::optimizer_vector x0("x0", p);
        const auto report = lo::minimize(bundle, lad, x0);
        if (r == 0 || report.elapsed_seconds < best.elapsed_seconds) best = report;
      }
      double distance = 0.0;
      for (int j = 0; j < p; ++j) distance = std::max(distance, std::abs(best.point(j) - vertex[std::size_t(j)]));
      std::printf("%-16s %-22s %6d %6d %9.4f %10.3g %10.3g %10.3g\n", d.id.c_str(), v.label, int(best.status),
                  best.iterations, best.elapsed_seconds, best.decrement, distance, best.value - objective_reference);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::printf("Kokkos execution space: %s\n", Kokkos::DefaultExecutionSpace::name());
  run(argc > 1 ? std::atoi(argv[1]) : 3);
  return 0;
}
