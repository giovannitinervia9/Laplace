// The scale of the engine (include/laplace/engine/marginal_criterion.hpp) with the number of groups:
//
//   engine_scale one|crossed <G> [parts|gradient|full|auto|inner] [automatic|dense|selected] [rows per block]
//
// one is gauss_spec of tests/engine/engine_models.hpp (a Gaussian with a random intercept u_g in the
// location and the scale modelled; outer vector log tau, c0, c1; N = 4 + G), crossed is crossed_spec with
// H = G (u_g, v_g in location and log scale, a_h in the location; outer vector the three log tau and c0;
// N = 4 + 3G). n = 20 G rows. Always runs the parts: the inner fit at the start, the observed H at its
// minimum, its factorization (analysis and numeric), the fill of the factor, a sample of the solves that
// form H^-1 (extrapolated to N), the selected inverse and one pass of the penalties on the host. With
// gradient it then evaluates the criterion and its gradient once at the start (the route given; the
// dense route only when its N x N inverse fits in the memory budget); with full (or auto when the dense
// matrices of the outer Hessian fit) a complete REML fit with fit_marginal, on the route given (the
// selected one forms nothing dense, its outer Hessian included). With inner it runs only the inner fit from the start, through an
// objective that times every call of the objective and, after each curvature, one extra numeric
// factorization of the matrix returned (not counted in the fit): the time of Newton outside the objective
// (its own factorizations and solves) is the rest. Prints the peak working set of the process; run one case
// per process.
#include <laplace/engine/marginal_criterion.hpp>

#include "../tests/engine/engine_models.hpp"

#include <Kokkos_Core.hpp>

#ifdef _WIN32
#define PSAPI_VERSION 2
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;
using clock_type = std::chrono::steady_clock;

double since(clock_type::time_point t0) { return std::chrono::duration<double>(clock_type::now() - t0).count(); }

// the inner objective with every call timed; after each curvature, one numeric factorization of the matrix
// returned, timed apart (its first call also analyzes the pattern)
template <class O>
struct timed_objective {
  using curvature_type = typename O::curvature_type;
  O& o;
  double seconds[3] = {0.0, 0.0, 0.0}, probe_first = 0.0, probe_rest = 0.0;
  long calls[3] = {0, 0, 0}, handed = 0;
  laplace::sparse::sparse_ldlt probe;
  // the objective's factor of its checked curvature, which Newton then uses instead of its own
  auto* factorized_curvature() {
    auto* f = o.factorized_curvature();
    handed += f != nullptr;
    return f;
  }
  double value(const lo::optimizer_point& x) {
    const auto t0 = clock_type::now();
    const double v = o.value(x);
    seconds[0] += since(t0), ++calls[0];
    return v;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    const auto t0 = clock_type::now();
    const double v = o.value_gradient(x, g);
    seconds[1] += since(t0), ++calls[1];
    return v;
  }
  double value_gradient_curvature(const lo::optimizer_point& x, const lo::optimizer_vector& g, curvature_type& H) {
    auto t0 = clock_type::now();
    const double v = o.value_gradient_curvature(x, g, H);
    seconds[2] += since(t0), ++calls[2];
    t0 = clock_type::now();
    (void)probe.factorize(H.n, H.col_ptr.data(), H.row_idx.data(), H.values.data());
    (calls[2] == 1 ? probe_first : probe_rest) += since(t0);
    return v;
  }
};

double peak_gb() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS c;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c))) return c.PeakWorkingSetSize / 1e9;
#endif
  return -1.0;
}

int run(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: engine_scale one|crossed G [parts|gradient|full|auto] [automatic|dense|selected] [rows per block]\n");
    return 1;
  }
  const std::string model = argv[1];
  const bool crossed = model == "crossed";
  const int G = std::atoi(argv[2]);
  const std::string mode = argc > 3 ? argv[3] : "auto";
  const std::string route_name = argc > 4 ? argv[4] : "automatic";
  const auto route = route_name == "selected" ? le::latent_inverse_route::selected
                     : route_name == "dense"  ? le::latent_inverse_route::dense
                                              : le::latent_inverse_route::automatic;
  const int block = argc > 5 ? std::atoi(argv[5]) : 16;
  const int n = 20 * G;
  auto t0 = clock_type::now();
  const auto s = crossed ? engine_test::crossed_spec(n, G, G, 41) : engine_test::gauss_spec(n, G, 40);
  std::vector<le::penalty_term> pen;
  std::vector<le::penalty_hyperparameter> outer_hyper;
  std::vector<int> psi;
  std::vector<double> u0;
  if (crossed) {
    namespace ld = laplace::distrib;
    pen = {engine_test::univariate_prior(ld::gaussian1_distrib{}, s.n_global, G, 2, {0}, {1.0}),
           engine_test::univariate_prior(ld::gaussian1_distrib{}, s.n_global + 1, G, 2, {1}, {1.0}),
           engine_test::univariate_prior(ld::gaussian1_distrib{}, s.n_global + 2 * G, G, 1, {2}, {1.0})};
    outer_hyper = {{engine_test::log_link()}, {engine_test::log_link()}, {engine_test::log_link()}};
    psi = {2};
    u0 = {std::log(0.5), std::log(0.25), std::log(0.4), -0.2};
  } else {
    pen = {engine_test::group_gaussian_prior(s, 0, 0, 1.0)};
    outer_hyper = {{engine_test::log_link()}};
    psi = {2, 3};
    u0 = {-0.5, -0.2, 0.6};
  }
  const auto m = le::build_row_model(s, 2, block);
  const int N = m.n_coordinates(), K = static_cast<int>(outer_hyper.size()), A = K + static_cast<int>(psi.size());
  std::printf("exec %s, model %s, G %d, n %d, N %d, A %d, rows per block %d, build %.3f s\n",
              Kokkos::DefaultExecutionSpace::name(), model.c_str(), G, m.view.n, N, A, block, since(t0));
  std::vector<double> x0(N, 0.0);
  for (std::size_t k = 0; k < psi.size(); ++k) x0[psi[k]] = u0[K + k];
  std::vector<double> theta(K);
  for (int a = 0; a < K; ++a) theta[a] = outer_hyper[a].map.inverse(u0[a]);
  lo::newton_optimizer<> inner;
  inner.stop_rule.tolerance = 1e-15;

  if (mode == "inner") {
    le::row_model_objective<engine_test::engine_distribs, lo::sparse_curvature_matrix> obj(
        m, pen, theta, le::row_curvature::observed_else_expected, psi);
    timed_objective<decltype(obj)> to{obj};
    lo::optimizer_vector xv("x0", N);
    for (int j = 0; j < N; ++j) xv(j) = x0[j];
    obj.reset_counts();
    t0 = clock_type::now();
    const auto rep = lo::minimize(inner, to, xv);
    const double total = since(t0) - to.probe_first - to.probe_rest;
    const auto& k = obj.counts();
    const double in_obj = to.seconds[0] + to.seconds[1] + to.seconds[2];
    std::printf("INNER %.6f s (the probe factorizations excluded), status %d, iterations %d\n", total,
                static_cast<int>(rep.status), rep.iterations);
    std::printf("INNER calls value %ld (%.6f s), value+gradient %ld (%.6f s), value+gradient+curvature %ld (%.6f s); "
                "kernels %.6f s (calls value %ld gradient %ld observed %ld expected %ld, expected fallbacks %ld)\n",
                to.calls[0], to.seconds[0], to.calls[1], to.seconds[1], to.calls[2], to.seconds[2], k.seconds, k.calls[0],
                k.calls[1], k.calls[2], k.calls[3], k.expected_fallbacks);
    std::printf("INNER in the objective %.6f s, of which kernels %.6f s and the rest %.6f s; outside the objective "
                "(Newton) %.6f s\n",
                in_obj, k.seconds, in_obj - k.seconds, total - in_obj);
    std::printf("INNER probe: one factorization after each curvature, the first with analysis %.6f s, the other %ld "
                "numeric only %.6f s (%.6f s each); nnz(L) %ld\n",
                to.probe_first, to.calls[2] - 1, to.probe_rest, to.calls[2] > 1 ? to.probe_rest / (to.calls[2] - 1) : 0.0,
                to.probe.factor_nonzeros());
    std::printf("INNER factors handed to Newton by the objective %ld of %ld curvatures (Newton factorized the "
                "other %ld)\n",
                to.handed, to.calls[2], to.calls[2] - to.handed);
    std::printf("PEAK working set %.3f GB\n", peak_gb());
    return 0;
  }

  // ---- the parts
  {
    le::row_model_objective<engine_test::engine_distribs, lo::sparse_curvature_matrix> obj(
        m, pen, theta, le::row_curvature::observed_else_expected, psi);
    lo::optimizer_vector xv("x0", N);
    for (int j = 0; j < N; ++j) xv(j) = x0[j];
    t0 = clock_type::now();
    const auto r = le::fit_inner(obj, xv, inner);
    const double t_inner = since(t0);
    std::printf("PART inner fit %.6f s (kernels %.6f s; calls value %ld gradient %ld observed %ld expected %ld), status %d, "
                "iterations %d\n",
                t_inner, r.kernels.seconds, r.kernels.calls[0], r.kernels.calls[1], r.kernels.calls[2], r.kernels.calls[3],
                static_cast<int>(r.optimizer.status), r.optimizer.iterations);
    lo::optimizer_vector g("g", N);
    lo::sparse_curvature_matrix Hs;
    t0 = clock_type::now();
    (void)obj.observed_value_gradient_curvature(r.optimizer.point, g, Hs);
    const double t_obs = since(t0);
    laplace::sparse::sparse_ldlt ldlt;
    t0 = clock_type::now();
    const auto out = ldlt.factorize(N, Hs.col_ptr.data(), Hs.row_idx.data(), Hs.values.data());
    const double t_fact1 = since(t0);
    t0 = clock_type::now();
    (void)ldlt.factorize(N, Hs.col_ptr.data(), Hs.row_idx.data(), Hs.values.data());
    const double t_fact2 = since(t0);
    const long nnzH = Hs.col_ptr[N], nnzL = ldlt.factor_nonzeros();
    std::printf("PART observed H %.6f s, nnz(lower H) %ld; factorization with analysis %.6f s, numeric only %.6f s, "
                "outcome %d; nnz(L) %ld = %.3g of N^2/2\n",
                t_obs, nnzH, t_fact1, t_fact2, static_cast<int>(out), nnzL, nnzL / (0.5 * double(N) * N));
    const int S = std::min(N, 200);
    std::vector<double> b(N);
    t0 = clock_type::now();
    for (int k = 0; k < S; ++k) {
      std::fill(b.begin(), b.end(), 0.0);
      b[(static_cast<long>(k) * N) / S] = 1.0;
      ldlt.solve(b.data());
    }
    const double t_solve = since(t0) / S;
    std::printf("PART solve %.3e s each; the N solves of H^-1 %.3f s (extrapolated)\n", t_solve, t_solve * N);
    t0 = clock_type::now();
    (void)ldlt.compute_selected_inverse();
    std::printf("PART selected inverse %.6f s\n", since(t0));
    // its derivative along one direction on the pattern of H (H itself), and one contraction: the work of
    // one outer coordinate in the outer Hessian by the derivative of Takahashi's recursion
    for (int rep = 0; rep < 2; ++rep) {  // the first call also builds the structure of the derivative
      t0 = clock_type::now();
      (void)ldlt.compute_selected_inverse_derivative(Hs.values.data());
      const double t_der = since(t0);
      t0 = clock_type::now();
      const double dot = ldlt.selected_inverse_derivative_dot(Hs.values.data());
      std::printf("PART selected inverse derivative %.6f s%s, contraction %.6f s (%g)\n", t_der,
                  rep == 0 ? " (with its structure)" : "", since(t0), dot);
    }
    const auto& ps = obj.penalties();
    std::vector<double> xh(r.optimizer.point.data(), r.optimizer.point.data() + N);
    auto xf = [&](int j) { return xh[j]; };
    t0 = clock_type::now();
    double sink = ps.value(xf);
    ps.add_gradient(xf, [&](int, double v) { sink += v; });
    ps.for_each_hessian_entry(xf, [&](int, int, double v) { sink += v; });
    std::printf("PART penalties on the host (value, gradient, Hessian) %.6f s (%g)\n", since(t0), sink);
  }

  // ---- the dense route of the criterion: H^-1 and H_full on the host, H^-1 and the A products on the device
  const double N2 = 8.0 * double(N) * N;
  const bool device_is_host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, le::row_model_view::mem>::accessible;
  const double host_gb = (device_is_host ? (2.0 + 1.0 + A) : 2.0) * N2 / 1e9, dev_gb = device_is_host ? 0.0 : (1.0 + A) * N2 / 1e9;
  const double host_budget = 16.0, dev_budget = 10.0;
  const bool fits = host_gb <= host_budget && dev_gb <= dev_budget;
  std::printf("DENSE host %.3f GB, device %.3f GB (budget %.0f, %.0f): %s\n", host_gb, dev_gb, host_budget, dev_budget,
              fits ? "fits" : "does not fit");
  if (mode == "parts" || (mode == "auto" && !fits)) {
    std::printf("PEAK working set %.3f GB\n", peak_gb());
    return 0;
  }
  if (mode == "gradient") {
    if (route != le::latent_inverse_route::selected && N2 / 1e9 > host_budget) {
      std::printf("GRADIENT dense route: the inverse needs %.3f GB, not run\nPEAK working set %.3f GB\n", N2 / 1e9, peak_gb());
      return 0;
    }
    lo::optimizer_vector uv("u0", A), g("g", A);
    for (int a = 0; a < A; ++a) uv(a) = u0[a];
    le::laplace_marginal_criterion<engine_test::engine_distribs> crit(m, pen, outer_hyper, psi, {}, x0, inner);
    crit.set_latent_inverse_route(route);
    t0 = clock_type::now();
    const double v = crit.value_gradient(uv, g);
    const double t_eval = since(t0);
    const auto& c = crit.counts();
    std::printf("GRADIENT route %s, %.6f s, criterion %.17g\n",
                c.latent_inverse_used == le::latent_inverse_route::selected ? "selected" : "dense", t_eval, v);
    std::printf("TIME inner %.6f, observed H %.6f, algebra %.6f [factorization %.6f, inverse %.6f, rest %.6f], passes %.6f "
                "[H^-1 to device %.6f, kernels %.6f], penalties %.6f, outside %.6f s\n",
                c.seconds_inner, c.seconds_observed, c.seconds_algebra, c.seconds_factorization, c.seconds_inverse,
                c.seconds_algebra - c.seconds_factorization - c.seconds_inverse, c.seconds_passes, c.seconds_to_device,
                c.seconds_passes - c.seconds_to_device, c.seconds_penalties,
                t_eval - c.seconds_inner - c.seconds_observed - c.seconds_algebra - c.seconds_passes - c.seconds_penalties);
    for (int a = 0; a < A; ++a) std::printf("GRAD %d %.17g\n", a, g(a));
    std::printf("PEAK working set %.3f GB\n", peak_gb());
    return 0;
  }
  lo::optimizer_vector uv("u0", A);
  for (int a = 0; a < A; ++a) uv(a) = u0[a];
  lo::newton_optimizer<> outer;
  outer.stop_rule.tolerance = 1e-12;
  le::laplace_marginal_criterion<engine_test::engine_distribs> crit(m, pen, outer_hyper, psi, {}, x0, inner);
  crit.set_latent_inverse_route(route);
  t0 = clock_type::now();
  const auto r = le::fit_marginal(crit, uv, outer);
  const double t_fit = since(t0);
  const auto& c = r.counts;
  std::printf("FIT %.6f s, status %d, outer iterations %d, criterion %.17g\n", t_fit, static_cast<int>(r.optimizer.status),
              r.optimizer.iterations, r.optimizer.value);
  std::printf("COUNTS evaluations %ld %ld %ld, inner fits %ld (failures %ld, stalls %ld, not pd %ld), outer passes %ld %ld, "
              "polishing steps %ld, nnz(L) %ld, dense traces %ld\n",
              c.evaluations[0], c.evaluations[1], c.evaluations[2], c.inner_fits, c.inner_failures, c.inner_stalls,
              c.not_positive_definite, c.outer_passes[0], c.outer_passes[1], c.polishing_steps, c.factor_nonzeros, c.dense_trace_evaluations);
  std::printf("TIME inner %.6f (kernels %.6f), observed H %.6f, algebra %.6f [factorization %.6f, H^-1 solves %.6f, "
              "trace products %.6f, rest %.6f], passes %.6f [H^-1 to device %.6f, kernels %.6f], penalties %.6f, "
              "outside %.6f s\n",
              c.seconds_inner, c.inner_kernels.seconds, c.seconds_observed, c.seconds_algebra, c.seconds_factorization,
              c.seconds_inverse, c.seconds_trace_products,
              c.seconds_algebra - c.seconds_factorization - c.seconds_inverse - c.seconds_trace_products, c.seconds_passes,
              c.seconds_to_device, c.seconds_passes - c.seconds_to_device, c.seconds_penalties,
              t_fit - c.seconds_inner - c.seconds_observed - c.seconds_algebra - c.seconds_passes - c.seconds_penalties);
  for (int a = 0; a < A; ++a) std::printf("OUTER %d %.17g\n", a, r.optimizer.point(a));
  std::printf("PEAK working set %.3f GB\n", peak_gb());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // a run stopped by a timeout keeps what it printed
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
