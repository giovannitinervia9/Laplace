// The fit_distrib bench (tests/optimizers/fit_distrib_bench.hpp) measured: for
// the 16 parametrizations of the frozen R references at n = 1e3 and 1e5, and on
// n = 1e6 drawn in C++ at the references' true values, Newton with the observed
// curvature (laplace1/laplace2: none, r_j = 0), Newton with the expected one and
// BFGS, from the library start (generic_start.hpp): status, iterations, counts of f, g and H, time per
// fit (fastest of repeated fits), and the time of one pass (one reduction) of
// order 0 (value), 1 (value and gradient) and 2 (with the observed second
// derivatives). The backend is that of the build (Serial/OpenMP: threads from
// OMP_NUM_THREADS; CUDA). Six laws added after the first bench (powerexp at
// p = 1.5, student_t1, weibull, beta, negbin2, mcp) on the R data of
// fit_distrib_laws.csv at n = 1e3 and 1e5 only.
//
// Usage: optimizers_fit_distrib [distrib ...]   (default: all 22)

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>

#include "../tests/optimizers/fit_distrib_data.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fb = fit_distrib_bench;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
using clock_type = std::chrono::steady_clock;

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

// repeat f for at least ~0.2 s (at most 1000 times) and return the fastest call in seconds
template <class F>
double fastest(F&& f) {
  double best = 1e300, total = 0.0;
  for (int r = 0; r < 1000 && (r < 3 || total < 0.2); ++r) {
    const auto t0 = clock_type::now();
    f();
    const double s = std::chrono::duration<double>(clock_type::now() - t0).count();
    best = std::min(best, s);
    total += s;
  }
  return best;
}

template <class O, class Opt>
void time_fit(const char* label, const std::string& data, const char* method, const Opt& opt, O& obj,
              const lo::optimizer_vector& u0) {
  lo::optimizer_report r;
  const double s = fastest([&] { r = lo::minimize(opt, obj, u0); });
  std::printf("%-12s %-10s %-16s %-8s %3d %4ld %4ld %4ld %12.3f  %.12g\n", label, data.c_str(), method,
              status_label(r.status), r.iterations, r.value_evaluations, r.gradient_evaluations,
              r.curvature_evaluations, s * 1e3, -r.value);
}

template <class D>
void bench_data(const char* name, const std::string& data, const Kokkos::View<const double*>& y,
                const std::vector<double>& theta0) {
  constexpr int P = D::n_params;
  using obj_t = fb::fit_distrib_objective<D>;
  obj_t expected(y, fb::fit_curvature::expected);
  lo::optimizer_vector u0("u0", P), g("g", P);
  lo::dense_curvature_matrix H("H", P, P);
  expected.free_of(theta0.data(), u0);
  bool smooth = true;
  for (int j = 0; j < P; ++j) smooth = smooth && D::params[j].regularity >= 2;
  if (smooth) {
    obj_t observed(y, fb::fit_curvature::observed);
    time_fit(name, data, "Newton observed", lo::newton_optimizer<>{}, observed, u0);
    const double p0 = fastest([&] { observed.value(u0); });
    const double p1 = fastest([&] { observed.value_gradient(u0, g); });
    const double p2 = fastest([&] { observed.value_gradient_curvature(u0, g, H); });
    std::printf("%-12s %-10s %-16s pass us: order 0 %.1f, order 1 %.1f, order 2 %.1f\n", name, data.c_str(), "", p0 * 1e6,
                p1 * 1e6, p2 * 1e6);
  }
  time_fit(name, data, "Newton expected", fb::fisher_scoring_optimizer(), expected, u0);
  time_fit(name, data, "BFGS", lo::bfgs_optimizer<>{}, expected, u0);
}

template <class D>
void bench_distrib(const char* name) {
  constexpr int P = D::n_params;
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + std::string(name);
  for (long n : {1000L, 100000L}) {
    const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
    bench_data<D>(name, n == 1000 ? "1e3" : "1e5", fb::to_device(yh), fb::library_start(D{}, yh));
  }
  double truth[P];
  int j = 0;
  for (const auto& r : fb::read_reference_rows(LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/fit_distrib.csv"))
    if (r.distrib == name && r.n == 1000) truth[j++] = r.true_value;
  const auto y = fb::simulate(fb::aggregate_builder<D>{}(truth), 1000000L, 20261115u);
  bench_data<D>(name, "1e6 sim", y, fb::library_start(D{}, fb::to_host_vector(y)));
}

// a law added after the first bench (fit_distrib_laws.csv): the R data at n = 1e3 and 1e5 only
template <class D>
void bench_law(const char* name) {
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + std::string(name);
  for (long n : {1000L, 100000L}) {
    const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
    bench_data<D>(name, n == 1000 ? "1e3" : "1e5", fb::to_device(yh), fb::library_start(D{}, yh));
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::set<std::string> want(argv + 1, argv + argc);
  auto on = [&](const char* n) { return want.empty() || want.count(n) > 0; };
  std::printf("backend %s, concurrency %d\n", Kokkos::DefaultExecutionSpace::name(),
              Kokkos::DefaultExecutionSpace().concurrency());
  std::printf("%-12s %-10s %-16s %-8s %3s %4s %4s %4s %12s  %s\n", "distrib", "data", "method", "status", "it", "f",
              "g", "H", "ms/fit", "loglik");
  if (on("cauchy")) bench_distrib<ld::cauchy_distrib>("cauchy");
  if (on("chisq")) bench_distrib<ld::chisq_distrib>("chisq");
  if (on("exponential")) bench_distrib<ld::exponential_distrib>("exponential");
  if (on("gamma1")) bench_distrib<ld::gamma1_distrib>("gamma1");
  if (on("gamma2")) bench_distrib<ld::gamma2_distrib>("gamma2");
  if (on("gaussian1")) bench_distrib<ld::gaussian1_distrib>("gaussian1");
  if (on("gaussian2")) bench_distrib<ld::gaussian2_distrib>("gaussian2");
  if (on("gaussian3")) bench_distrib<ld::gaussian3_distrib>("gaussian3");
  if (on("gumbel")) bench_distrib<ld::gumbel_distrib>("gumbel");
  if (on("invgauss")) bench_distrib<ld::invgauss_distrib>("invgauss");
  if (on("laplace1")) bench_distrib<ld::laplace1_distrib>("laplace1");
  if (on("laplace2")) bench_distrib<ld::laplace2_distrib>("laplace2");
  if (on("logistic")) bench_distrib<ld::logistic_distrib>("logistic");
  if (on("lognormal1")) bench_distrib<ld::lognormal1_distrib>("lognormal1");
  if (on("lognormal2")) bench_distrib<ld::lognormal2_distrib>("lognormal2");
  if (on("poisson")) bench_distrib<ld::poisson_distrib>("poisson");
  if (on("powerexp_p1.5")) bench_law<ld::powerexp_distrib>("powerexp_p1.5");
  if (on("student_t1")) bench_law<ld::student_t1_distrib>("student_t1");
  if (on("weibull")) bench_law<ld::weibull_distrib>("weibull");
  if (on("beta")) bench_law<ld::beta_distrib>("beta");
  if (on("negbin2")) bench_law<ld::negbin2_distrib>("negbin2");
  if (on("mcp")) bench_law<ld::mcp_distrib>("mcp");
  return 0;
}
