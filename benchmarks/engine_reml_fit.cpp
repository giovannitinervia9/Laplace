// The time of a complete REML fit of the engine (include/laplace/engine/marginal_criterion.hpp) on the
// models of two prototypes (tests/engine/engine_models.hpp), from the prototypes' starts:
//
//   engine_reml_fit lk  <data.csv> [reps] [rows per block]
//   engine_reml_fit lk3 <data.csv> [reps] [rows per block]
//
// lk is the Gaussian model of prototipi/kokkos_fit/lk.cpp (columns id, x, t, w, y; outer vector the log
// lambda of the group prior, the log lambda of the spline, the two coefficients of log sigma; start 0, 0,
// log sd(y), 0), lk3 the gamma with a logistic curve of model C of prototipi/node_dispatch/hy4_main.cpp
// (columns id, t, y; outer vector the log lambda of the group prior and the coefficient of log phi; start
// -2 log 15, -2). Prints the best of the repetitions, the outer optimizer's status and counts, where the
// time went, and the fitted outer vector.
#include <laplace/engine/marginal_criterion.hpp>

#include "../tests/engine/engine_models.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;

int run(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: engine_reml_fit lk|lk3 data.csv [reps] [rows per block]\n");
    return 1;
  }
  const std::string model = argv[1];
  const bool lk = model == "lk";
  auto pf = lk ? engine_test::kokkos_fit_model(argv[2], 0.0, 0.0, 0.0, 0.0) : engine_test::lk3_gamma_model(argv[2], 0.0, -2.0);
  std::vector<double> u0;
  if (lk) {
    pf.penalties[0].hyperparameter = 0, pf.penalties[1].hyperparameter = 1;
    double my = 0.0, vy = 0.0;
    for (double v : pf.spec.y) my += v;
    my /= pf.spec.y.size();
    for (double v : pf.spec.y) vy += (v - my) * (v - my);
    vy /= pf.spec.y.size() - 1;
    u0 = {0.0, 0.0, 0.5 * std::log(vy), 0.0};
  } else {
    pf.penalties[0].hyperparameter = 0;
    u0 = {-2.0 * std::log(0.1 * 150.0 + 1e-8), -2.0};
  }
  const int reps = argc > 3 ? std::atoi(argv[3]) : 3;
  const int block = argc > 4 ? std::atoi(argv[4]) : 16;
  const auto m = le::build_row_model(pf.spec, 2, block);
  lo::optimizer_vector uv("u0", u0.size());
  for (std::size_t a = 0; a < u0.size(); ++a) uv(a) = u0[a];
  lo::newton_optimizer<> inner;
  inner.stop_rule.tolerance = 1e-15;
  lo::newton_optimizer<> outer;
  outer.stop_rule.tolerance = 1e-12;
  double best = 1e300;
  le::marginal_fit_report r;
  using criterion = le::laplace_marginal_criterion<engine_test::engine_distribs>;
  // ENGINE_REML_TRACE: print every evaluation of the criterion (one run, not timed)
  struct traced {
    criterion* c;
    using curvature_type = lo::dense_curvature_matrix;
    void show(const char* what, const lo::optimizer_point& u, double v) const {
      std::printf("TRACE %s V %.17g at", what, v);
      for (std::size_t a = 0; a < u.extent(0); ++a) std::printf(" %.10g", u(a));
      std::printf(" (inner failures %ld, last status %d)\n", c->counts().inner_failures,
                  static_cast<int>(c->counts().last_inner_failure));
    }
    double value(const lo::optimizer_point& u) {
      const double v = c->value(u);
      show("value", u, v);
      return v;
    }
    double value_gradient(const lo::optimizer_point& u, const lo::optimizer_vector& g) { return c->value_gradient(u, g); }
    double value_gradient_curvature(const lo::optimizer_point& u, const lo::optimizer_vector& g, curvature_type& H) {
      const double v = c->value_gradient_curvature(u, g, H);
      show("curvature", u, v);
      return v;
    }
  };
  if (std::getenv("ENGINE_REML_TRACE")) {
    criterion crit(m, pf.penalties, pf.held, {}, pf.x0, inner);
    traced t{&crit};
    (void)lo::minimize(outer, t, uv);
  }
  for (int k = 0; k < reps; ++k) {
    criterion crit(m, pf.penalties, pf.held, {}, pf.x0, inner);
    const auto t0 = std::chrono::steady_clock::now();
    r = le::fit_marginal(crit, uv, outer);
    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  const auto& c = r.counts;
  std::printf("exec %s, model %s, n %d, coordinates %d, rows per block %d\n", Kokkos::DefaultExecutionSpace::name(),
              model.c_str(), m.view.n, m.n_coordinates(), block);
  std::printf("FIT best %.6f s of %d, status %d, outer iterations %d, criterion %.17g\n", best, reps,
              static_cast<int>(r.optimizer.status), r.optimizer.iterations, r.optimizer.value);
  std::printf("COUNTS evaluations %ld %ld %ld, inner fits %ld (failures %ld, last status %d, stalls %ld, not pd %ld), outer passes %ld %ld; "
              "inner kernels value %ld gradient %ld observed %ld expected %ld (fallbacks %ld, reused %ld)\n",
              c.evaluations[0], c.evaluations[1], c.evaluations[2], c.inner_fits, c.inner_failures, static_cast<int>(c.last_inner_failure),
              c.inner_stalls,
              c.not_positive_definite, c.outer_passes[0], c.outer_passes[1], c.inner_kernels.calls[0],
              c.inner_kernels.calls[1], c.inner_kernels.calls[2], c.inner_kernels.calls[3],
              c.inner_kernels.expected_fallbacks, c.inner_kernels.reused);
  std::printf("TIME (last run) inner %.6f s (kernels %.6f), host algebra %.6f s, outer passes %.6f s\n", c.seconds_inner,
              c.inner_kernels.seconds, c.seconds_algebra, c.seconds_passes);
  for (int a = 0; a < static_cast<int>(u0.size()); ++a) std::printf("OUTER %d %.17g\n", a, r.optimizer.point(a));
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
