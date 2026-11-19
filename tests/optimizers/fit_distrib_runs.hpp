// The runs of the fit_distrib bench shared by its test files (test_fit_distrib.cpp,
// test_fit_distrib_laws.cpp, test_fit_distrib_diagnostics.cpp): the decrement-rule bound,
// one run of an optimizer with its estimate, bound and boundary flag, the three standard
// runs (Newton observed where every r_j >= 2, Fisher scoring, BFGS), and the printout.
// Split from test_fit_distrib.cpp, whose host pass under nvcc exceeded MSVC's section limit
// (C1128) once the laws added after the first bench were fitted in the same unit.
#pragma once

#include "fit_distrib_data.hpp"

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fit_distrib_runs {

namespace fb = fit_distrib_bench;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;

inline constexpr double eps = std::numeric_limits<double>::epsilon();
inline constexpr double stop_tol = 1e-6;  // decrement_stop_rule's default
inline const bool verbose = std::getenv("LAPLACE_FIT_DISTRIB_VERBOSE") != nullptr;
inline const char* status_name(lo::optimizer_status s) {
  switch (s) {
    case lo::optimizer_status::converged: return "converged";
    case lo::optimizer_status::converged_at_resolution: return "converged at resolution";
    case lo::optimizer_status::stopped_on_change: return "stopped on change";
    case lo::optimizer_status::stalled: return "stalled";
    case lo::optimizer_status::iteration_budget: return "iteration budget";
    case lo::optimizer_status::evaluation_budget: return "evaluation budget";
    case lo::optimizer_status::line_search_failed: return "line search failed";
    default: return "start not admissible";
  }
}

struct fit_run {
  std::string method;
  lo::optimizer_report report;
  std::vector<double> theta, bound;  // estimate and the decrement-rule bound per parameter
  bool boundary = false;             // the estimate on the boundary of the domain (on_boundary)
  double se_free = 0.0;              // largest standard error in u over the parameters with a finite end
};

// the decrement-rule bound on |theta_j - theta_j*|, from the expected curvature in u at the
// estimate: (H^-1)_jj is the square of the objective's free standard error, which holds for
// any number of parameters (the first version inverted a 2 x 2 by hand)
template <class O>
std::vector<double> decrement_bound(O& obj, const lo::optimizer_point& u) {
  constexpr int P = O::P;
  const auto se = obj.free_standard_errors(u);
  std::vector<double> b(P);
  double t[P];
  obj.theta_of(u, t);
  for (int j = 0; j < P; ++j)
    b[j] = std::abs(obj.h[j].inverse_d1(u(j))) * std::sqrt(2.0 * 2.0 * stop_tol) * se[j] + 8.0 * eps * std::abs(t[j]);
  return b;
}

template <class O, class Opt>
fit_run run_one(const std::string& method, const Opt& opt, O& obj, const std::vector<double>& u0) {
  constexpr int P = O::P;
  lo::optimizer_vector x0("x0", P);
  for (int j = 0; j < P; ++j) x0(j) = u0[j];
  fit_run r{method, lo::minimize(opt, obj, x0), {}, {}};
  double t[P];
  obj.theta_of(r.report.point, t);
  r.theta.assign(t, t + P);
  r.bound = decrement_bound(obj, r.report.point);
  r.boundary = obj.on_boundary(r.report.point);
  const auto se = obj.free_standard_errors(r.report.point);
  for (int j = 0; j < P; ++j)
    if (std::isfinite(O::distrib_type::params[j].lower) || std::isfinite(O::distrib_type::params[j].upper))
      r.se_free = std::max(r.se_free, se[j]);
  return r;
}

// Newton (observed unless refused, expected) and BFGS from the given start in u
template <class D, class Builder>
std::vector<fit_run> fit_all(const Kokkos::View<const double*>& y, const std::vector<double>& theta0, Builder b,
                             const std::string& start_name) {
  constexpr int P = D::n_params;
  using obj_t = fb::fit_distrib_objective<D, Builder>;
  obj_t expected(y, fb::fit_curvature::expected, b);
  lo::optimizer_vector u("u", P);
  expected.free_of(theta0.data(), u);
  std::vector<double> u0(u.data(), u.data() + P);
  if (start_name == "poor")
    for (double& x : u0) x += 1.0;
  std::vector<fit_run> runs;
  bool smooth = true;
  for (int j = 0; j < P; ++j) smooth = smooth && D::params[j].regularity >= 2;
  if (smooth) {
    obj_t observed(y, fb::fit_curvature::observed, b);
    runs.push_back(run_one("Newton observed, " + start_name, lo::newton_optimizer<>{}, observed, u0));
  }
  runs.push_back(run_one("Newton expected, " + start_name, fb::fisher_scoring_optimizer(), expected, u0));
  runs.push_back(run_one("BFGS, " + start_name, lo::bfgs_optimizer<>{}, expected, u0));
  return runs;
}

inline void print_run(const std::string& label, const fit_run& r) {
  if (!verbose) return;
  std::printf("%-28s %-26s %-24s it %3d f %3ld g %3ld H %3ld  -l %.17g  dec %.2e  theta", label.c_str(),
              r.method.c_str(), status_name(r.report.status), r.report.iterations, r.report.value_evaluations,
              r.report.gradient_evaluations, r.report.curvature_evaluations, r.report.value, r.report.decrement);
  for (double t : r.theta) std::printf(" %.17g", t);
  std::printf("  se_u %.3g\n", r.se_free);
}

inline const std::vector<fb::reference_row>& reference_rows() {
  static const auto rows =
      fb::read_reference_rows(LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/fit_distrib.csv");
  return rows;
}


}  // namespace fit_distrib_runs
