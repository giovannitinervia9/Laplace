// The time of a complete inner fit of the engine (include/laplace/engine/inner_fit.hpp) on the
// models of two prototypes (tests/engine/engine_models.hpp), at hyperparameters and held
// coefficients given on the command line, from the prototypes' starts:
//
//   engine_inner_fit lk  <data.csv> <rho0> <rho1> <c0> <c1> [reps] [rows per block]
//   engine_inner_fit lk3 <data.csv> <rho> <c0> [reps] [rows per block]
//
// lk is the Gaussian model of prototipi/kokkos_fit/lk.cpp (columns id, x, t, w, y), lk3 the gamma
// with a logistic curve of model C of prototipi/node_dispatch/hy4_main.cpp (columns id, t, y).
// Prints the best of the repetitions, the optimizer's status and counts, the kernels run and the
// time spent in them, and the first coordinates of the fitted point.
#include <laplace/engine/inner_fit.hpp>

#include "../tests/engine/engine_models.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;

int run(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: engine_inner_fit lk|lk3 data.csv hyperparameters... [reps] [rows per block]\n");
    return 1;
  }
  const std::string model = argv[1];
  const bool lk = model == "lk";
  const int first_opt = lk ? 7 : 5;
  if (argc < first_opt) return 1;
  const auto pf = lk ? engine_test::kokkos_fit_model(argv[2], std::atof(argv[3]), std::atof(argv[4]),
                                                     std::atof(argv[5]), std::atof(argv[6]))
                     : engine_test::lk3_gamma_model(argv[2], std::atof(argv[3]), std::atof(argv[4]));
  const int reps = argc > first_opt ? std::atoi(argv[first_opt]) : 5;
  const int block = argc > first_opt + 1 ? std::atoi(argv[first_opt + 1]) : 16;
  const auto t_build = std::chrono::steady_clock::now();
  const auto m = le::build_row_model(pf.spec, 2, block);
  const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_build).count();
  lo::optimizer_vector x0("x0", pf.x0.size());
  for (std::size_t j = 0; j < pf.x0.size(); ++j) x0(j) = pf.x0[j];
  lo::newton_optimizer<> newton;
  newton.stop_rule.tolerance = 1e-14;
  double best = 1e300;
  le::inner_fit_report r;
  for (int k = 0; k < reps; ++k) {
    le::row_model_objective<engine_test::engine_distribs, lo::sparse_curvature_matrix> obj(
        m, pf.penalties, {}, le::row_curvature::observed_else_expected, pf.held);
    const auto t0 = std::chrono::steady_clock::now();
    r = le::fit_inner(obj, x0, newton);
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::printf("exec %s, model %s, n %d, coordinates %d, rows per block %d, build %.3f s\n",
              Kokkos::DefaultExecutionSpace::name(), model.c_str(), m.view.n, m.n_coordinates(), block, build_s);
  std::printf("FIT best %.6f s of %d, status %d, iterations %d, value %.17g\n", best, reps,
              static_cast<int>(r.optimizer.status), r.optimizer.iterations, r.optimizer.value);
  std::printf("KERNELS value %ld gradient %ld observed %ld expected %ld (fallbacks %ld, reused %ld), %.6f s\n",
              r.kernels.calls[0], r.kernels.calls[1], r.kernels.calls[2], r.kernels.calls[3],
              r.kernels.expected_fallbacks, r.kernels.reused, r.kernels.seconds);
  for (int j = 0; j < std::min(4, m.n_coordinates()); ++j) std::printf("COEF %d %.17g\n", j, r.optimizer.point(j));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    rc = run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    rc = 1;
  }
  Kokkos::finalize();
  return rc;
}
