// Two diagnostics of the fit_distrib bench, skipped by default: the paths of the methods
// from the +1 start, and Fisher scoring with max_step infinite, 2 and 1 on every data set.
// They measured the choices recorded in test_fit_distrib_laws.cpp and fit_distrib_bench.hpp
// (H3, 2026-11-19); run them with --no-skip and the test-case filter given below.

#include <doctest/doctest.h>

#include "fit_distrib_runs.hpp"

#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/trust_region.hpp>

namespace {

using namespace fit_distrib_runs;

const std::vector<fb::laws_reference_row>& laws_reference_rows() {
  static const auto rows =
      fb::read_laws_reference_rows(LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/fit_distrib_laws.csv");
  return rows;
}

}  // namespace

// Diagnostic, skipped by default (run with -tc="*poor-start trace*" --no-skip; C++ only, 131 s on 8
// OpenMP threads, gcc): the path of
// Newton with the expected and the observed curvature and of BFGS from the +1 start, with
// theta, -l and the decrement at each accepted point, and the expected and observed
// curvature in u at the start (H3, 2026-11-19).
template <class D>
void poor_start_trace(const std::string& name, long n) {
  constexpr int P = D::n_params;
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + name;
  const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
  const auto y = fb::to_device(yh);
  using obj_t = fb::fit_distrib_objective<D>;
  obj_t ex(y, fb::fit_curvature::expected);
  const auto t0 = fb::library_start(D{}, yh);
  lo::optimizer_vector u0("u0", P), g("g", P);
  ex.free_of(t0.data(), u0);
  for (int j = 0; j < P; ++j) u0(j) += 1.0;
  auto show = [&](const char* what, auto& obj) {
    lo::dense_curvature_matrix H("H", P, P);
    const double f = obj.value_gradient_curvature(u0, g, H);
    std::printf("%s %s at the +1 start: -l %.10g, g", name.c_str(), what, f);
    for (int j = 0; j < P; ++j) std::printf(" %.4g", g(j));
    std::printf("\n");
    for (int j = 0; j < P; ++j) {
      std::printf("   H");
      for (int k = 0; k < P; ++k) std::printf(" %12.5g", H(j, k));
      std::printf("\n");
    }
  };
  show("expected", ex);
  bool smooth = true;
  for (int j = 0; j < P; ++j) smooth = smooth && D::params[j].regularity >= 2;
  auto trace = [&](const lo::optimizer_trace_point& p) {
    double t[P];
    lo::optimizer_vector u("u", P);
    for (int j = 0; j < P; ++j) u(j) = p.point(j);
    ex.theta_of(u, t);
    std::printf("  it %3d -l %.10g dec %9.3g step %.3g theta", p.iteration, p.value, p.decrement, p.step);
    for (int j = 0; j < P; ++j) std::printf(" %.6g", t[j]);
    std::printf("\n");
  };
  lo::newton_optimizer<> newton;
  newton.trace = trace;
  std::printf("%s n = %ld, Newton expected from the +1 start\n", name.c_str(), n);
  lo::minimize(newton, ex, u0);
  if (smooth) {
    obj_t ob(y, fb::fit_curvature::observed);
    show("observed", ob);
    std::printf("%s n = %ld, Newton observed from the +1 start\n", name.c_str(), n);
    lo::minimize(newton, ob, u0);
  }
  lo::bfgs_optimizer<> bfgs;
  bfgs.trace = trace;
  std::printf("%s n = %ld, BFGS from the +1 start\n", name.c_str(), n);
  lo::minimize(bfgs, ex, u0);
  // remedies, summarized: Newton expected with a step cap, the trust region, the alternation
  auto summary = [&](const char* what, const lo::optimizer_report& r) {
    double t[P];
    ex.theta_of(r.point, t);
    std::printf("  %-36s %-22s it %3d -l %.10g theta", what, status_name(r.status), r.iterations, r.value);
    for (int j = 0; j < P; ++j) std::printf(" %.6g", t[j]);
    std::printf("\n");
  };
  std::printf("%s n = %ld, remedies from the +1 start\n", name.c_str(), n);
  for (double cap : {2.0, 1.0, 0.5}) {
    lo::newton_optimizer<> capped;
    capped.max_step = cap;
    char what[64];
    std::snprintf(what, sizeof what, "Newton expected, max_step %g", cap);
    summary(what, lo::minimize(capped, ex, u0));
  }
  summary("trust region expected", lo::minimize(lo::trust_region_optimizer<>{}, ex, u0));
  if constexpr (P > 1) {
    lo::block_coordinate_optimizer<lo::brent_optimizer<>, lo::newton_optimizer<>> alt;
    alt.block_sizes = {1, P - 1};
    summary("alternation", lo::minimize(alt, ex, u0));
    std::get<1>(alt.sub_optimizers).max_step = 1.0;
    summary("alternation, Newton max_step 1", lo::minimize(alt, ex, u0));
  }
}

void fit_distrib_poor_start_traces() {
  poor_start_trace<ld::pseudohuber_distrib>("pseudohuber", 1000);
  poor_start_trace<ld::enet_distrib>("enet", 1000);
  poor_start_trace<ld::mcp_distrib<ld::gaussian_tail>>("mcp", 1000);
  poor_start_trace<ld::scad_distrib<ld::gaussian_tail>>("scad", 1000);
  poor_start_trace<ld::pseudohuber_distrib>("pseudohuber", 100000);
  poor_start_trace<ld::enet_distrib>("enet", 100000);
  poor_start_trace<ld::mcp_distrib<ld::gaussian_tail>>("mcp", 100000);
  poor_start_trace<ld::scad_distrib<ld::gaussian_tail>>("scad", 100000);
}

TEST_CASE("fit_distrib bench: poor-start trace (diagnostic)" * doctest::skip()) { fit_distrib_poor_start_traces(); }

// Diagnostic, skipped by default (run with -tc="*max_step*" --no-skip; C++ only, 37 s on 8
// OpenMP threads, gcc): Fisher scoring (Newton
// on the expected curvature) with max_step infinite, 2 and 1 in the free coordinates, from the
// library start and the +1 start, on every data set of the bench. For each run: status,
// iterations, value and gradient evaluations, and the distance of -l from the lowest -l
// reached by any of the six runs or by R's reference (H3, 2026-11-19).
template <class D, class Builder>
void fisher_cap_rows(const std::string& label, const std::vector<double>& yh, const D& prototype, Builder b,
                     double reference_negll) {
  constexpr int P = D::n_params;
  const auto y = fb::to_device(yh);
  fb::fit_distrib_objective<D, Builder> ex(y, fb::fit_curvature::expected, b);
  const auto t0 = fb::library_start(prototype, yh);
  lo::optimizer_vector u("u", P);
  ex.free_of(t0.data(), u);
  const double caps[3] = {std::numeric_limits<double>::infinity(), 2.0, 1.0};
  lo::optimizer_report rep[2][3];
  double best = std::isfinite(reference_negll) ? reference_negll : std::numeric_limits<double>::infinity();
  for (int s = 0; s < 2; ++s)
    for (int c = 0; c < 3; ++c) {
      lo::optimizer_vector x0("x0", P);
      for (int j = 0; j < P; ++j) x0(j) = u(j) + (s == 1 ? 1.0 : 0.0);
      lo::newton_optimizer<> newton;
      newton.max_step = caps[c];
      rep[s][c] = lo::minimize(newton, ex, x0);
      if (std::isfinite(rep[s][c].value)) best = std::min(best, rep[s][c].value);
    }
  for (int s = 0; s < 2; ++s) {
    std::printf("%-22s %-7s", label.c_str(), s == 0 ? "library" : "poor");
    for (int c = 0; c < 3; ++c) {
      const auto& r = rep[s][c];
      std::printf(" | %-9.9s %3d %4ld %4ld %9.2e", status_name(r.status), r.iterations, r.value_evaluations,
                  r.gradient_evaluations, r.value - best);
    }
    std::printf("\n");
  }
}

template <class D>
void fisher_cap_reference(const std::string& name, long n) {
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + name;
  const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
  double ref = std::numeric_limits<double>::quiet_NaN();
  for (const auto& r : reference_rows())
    if (r.distrib == name && r.n == n) ref = -r.loglik;
  fisher_cap_rows(name + " " + std::to_string(n), yh, D{}, fb::aggregate_builder<D>{}, ref);
}

template <class D, class Builder>
void fisher_cap_law(const std::string& name, const D& prototype, Builder b, long n) {
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + name;
  const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
  double ref = std::numeric_limits<double>::quiet_NaN();
  for (const auto& r : laws_reference_rows())
    if (r.name == name && r.n == n) ref = -r.loglik_mle;
  fisher_cap_rows(name + " " + std::to_string(n), yh, prototype, b, ref);
}

template <class D, class Builder>
void fisher_cap_simulated(const std::string& name, const D& truth, Builder b, long n) {
  const auto yh = fb::to_host_vector(fb::simulate(truth, n, 20261115u + static_cast<std::uint64_t>(n)));
  fisher_cap_rows(name + " " + std::to_string(n), yh, truth, b, std::numeric_limits<double>::quiet_NaN());
}

void fit_distrib_fisher_caps() {
  using fb::aggregate_builder;
  std::printf("%-22s %-7s | %-35s | %-35s | %-35s\n", "data", "start", "max_step inf: status it f g d(-l)",
              "max_step 2", "max_step 1");
  for (long n : {1000L, 100000L}) {
    fisher_cap_reference<ld::cauchy_distrib>("cauchy", n);
    fisher_cap_reference<ld::chisq_distrib>("chisq", n);
    fisher_cap_reference<ld::exponential_distrib>("exponential", n);
    fisher_cap_reference<ld::gamma1_distrib>("gamma1", n);
    fisher_cap_reference<ld::gamma2_distrib>("gamma2", n);
    fisher_cap_reference<ld::gaussian1_distrib>("gaussian1", n);
    fisher_cap_reference<ld::gaussian2_distrib>("gaussian2", n);
    fisher_cap_reference<ld::gaussian3_distrib>("gaussian3", n);
    fisher_cap_reference<ld::gumbel_distrib>("gumbel", n);
    fisher_cap_reference<ld::invgauss_distrib>("invgauss", n);
    fisher_cap_reference<ld::laplace1_distrib>("laplace1", n);
    fisher_cap_reference<ld::laplace2_distrib>("laplace2", n);
    fisher_cap_reference<ld::logistic_distrib>("logistic", n);
    fisher_cap_reference<ld::lognormal1_distrib>("lognormal1", n);
    fisher_cap_reference<ld::lognormal2_distrib>("lognormal2", n);
    fisher_cap_reference<ld::poisson_distrib>("poisson", n);
    fisher_cap_simulated("bernoulli", ld::bernoulli_distrib{0.3}, aggregate_builder<ld::bernoulli_distrib>{}, n);
    fisher_cap_simulated("binomial", ld::binomial_distrib{0.3, 10.0}, fb::binomial_builder{10.0}, n);
    fisher_cap_simulated("geometric", ld::geometric_distrib{2.5}, aggregate_builder<ld::geometric_distrib>{}, n);
    fisher_cap_law("student_t1", ld::student_t1_distrib{}, aggregate_builder<ld::student_t1_distrib>{}, n);
    fisher_cap_law("student_t2", ld::student_t2_distrib{}, aggregate_builder<ld::student_t2_distrib>{}, n);
    fisher_cap_law("skewnormal", ld::skewnormal_distrib{}, aggregate_builder<ld::skewnormal_distrib>{}, n);
    fisher_cap_law("skewt", ld::skewt_distrib{}, aggregate_builder<ld::skewt_distrib>{}, n);
    fisher_cap_law("pseudohuber", ld::pseudohuber_distrib{}, aggregate_builder<ld::pseudohuber_distrib>{}, n);
    fisher_cap_law("weibull", ld::weibull_distrib{}, aggregate_builder<ld::weibull_distrib>{}, n);
    fisher_cap_law("vonmises", ld::vonmises_distrib{}, aggregate_builder<ld::vonmises_distrib>{}, n);
    fisher_cap_law("beta", ld::beta_distrib{}, aggregate_builder<ld::beta_distrib>{}, n);
    const ld::betabinom_distrib bb{0.5, 1.0, 10.0};
    fisher_cap_law("betabinom", bb, fb::prototype_builder<ld::betabinom_distrib>{bb}, n);
    fisher_cap_law("negbin1", ld::negbin1_distrib{}, aggregate_builder<ld::negbin1_distrib>{}, n);
    fisher_cap_law("negbin2", ld::negbin2_distrib{}, aggregate_builder<ld::negbin2_distrib>{}, n);
    fisher_cap_law("pig", ld::pig_distrib{}, aggregate_builder<ld::pig_distrib>{}, n);
    fisher_cap_law("enet", ld::enet_distrib{}, aggregate_builder<ld::enet_distrib>{}, n);
    fisher_cap_law("powerexp_p1.5", ld::powerexp_distrib{}, aggregate_builder<ld::powerexp_distrib>{}, n);
    fisher_cap_law("powerexp_p3", ld::powerexp_distrib{}, aggregate_builder<ld::powerexp_distrib>{}, n);
    fisher_cap_law("mcp", ld::mcp_distrib<ld::gaussian_tail>{}, aggregate_builder<ld::mcp_distrib<ld::gaussian_tail>>{}, n);
    fisher_cap_law("scad", ld::scad_distrib<ld::gaussian_tail>{}, aggregate_builder<ld::scad_distrib<ld::gaussian_tail>>{}, n);
  }
}

TEST_CASE("fit_distrib bench: Fisher scoring max_step (diagnostic)" * doctest::skip()) { fit_distrib_fisher_caps(); }
