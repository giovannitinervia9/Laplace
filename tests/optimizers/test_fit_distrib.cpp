// The fit_distrib bench (fit_distrib_bench.hpp) on the 16 parametrizations of
// the frozen R references (n = 1e3 and 1e5) and on bernoulli, binomial and
// geometric data simulated in C++. Each data set is fitted by Newton with the
// observed curvature (not for laplace1/laplace2, whose mu has r_j = 0), Newton
// with the expected one (Fisher scoring) and BFGS, from the library start
// (generic_start.hpp) and
// from a poor start (every free coordinate moved by +1).
//
// Checks, for every run that must converge:
// - the closed-form MLE where it exists, within what the decrement rule allows:
//   the rule stops at (1/2) g'H^-1 g <= tol = 1e-6, so f - f* is about tol
//   (2 tol allowed) and |u_j - u_j*| <= sqrt(2 (2 tol) (H^-1)_jj) with H the
//   expected curvature in u at the estimate; carried to theta by h_j';
// - statmodels7's estimate within that bound plus statmodels7's own distance
//   from the optimum (its rule: max |score per observation| < 1e-6);
// - the log-likelihood at least statmodels7's minus 2 tol and the rounding.
// The kernels are in the functors of fit_distrib_bench.hpp (external linkage).

#include <doctest/doctest.h>

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

namespace {

namespace fb = fit_distrib_bench;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;

constexpr double eps = std::numeric_limits<double>::epsilon();
constexpr double stop_tol = 1e-6;  // decrement_stop_rule's default
const bool verbose = std::getenv("LAPLACE_FIT_DISTRIB_VERBOSE") != nullptr;

const char* status_name(lo::optimizer_status s) {
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

// the decrement-rule bound on |theta_j - theta_j*|, from the expected curvature in u at the estimate
template <class O>
std::vector<double> decrement_bound(O& obj, const lo::optimizer_point& u) {
  constexpr int P = O::P;
  O e = obj;
  e.curvature = fb::fit_curvature::expected;
  lo::optimizer_vector g("g", P);
  lo::dense_curvature_matrix H("H", P, P);
  e.value_gradient_curvature(u, g, H);
  double Hi[2][2];
  if constexpr (P == 1) {
    Hi[0][0] = 1.0 / H(0, 0);
  } else {
    const double det = H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0);
    Hi[0][0] = H(1, 1) / det;
    Hi[1][1] = H(0, 0) / det;
  }
  std::vector<double> b(P);
  double t[P];
  obj.theta_of(u, t);
  for (int j = 0; j < P; ++j)
    b[j] = std::abs(obj.h[j].inverse_d1(u(j))) * std::sqrt(2.0 * 2.0 * stop_tol * Hi[j][j]) + 8.0 * eps * std::abs(t[j]);
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
  runs.push_back(run_one("Newton expected, " + start_name, lo::newton_optimizer<>{}, expected, u0));
  runs.push_back(run_one("BFGS, " + start_name, lo::bfgs_optimizer<>{}, expected, u0));
  return runs;
}

void print_run(const std::string& label, const fit_run& r) {
  if (!verbose) return;
  std::printf("%-28s %-26s %-24s it %3d f %3ld g %3ld H %3ld  -l %.17g  dec %.2e  theta", label.c_str(),
              r.method.c_str(), status_name(r.report.status), r.report.iterations, r.report.value_evaluations,
              r.report.gradient_evaluations, r.report.curvature_evaluations, r.report.value, r.report.decrement);
  for (double t : r.theta) std::printf(" %.17g", t);
  std::printf("  se_u %.3g\n", r.se_free);
}

const std::vector<fb::reference_row>& reference_rows() {
  static const auto rows =
      fb::read_reference_rows(LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/fit_distrib.csv");
  return rows;
}

// fit one reference data set and check it
template <class D>
void reference_case(const std::string& name, long n) {
  constexpr int P = D::n_params;
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + name;
  const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
  REQUIRE(static_cast<long>(yh.size()) == n);
  std::vector<fb::reference_row> ref;
  for (const auto& r : reference_rows())
    if (r.distrib == name && r.n == n) ref.push_back(r);
  REQUIRE(static_cast<int>(ref.size()) == P);
  // rows in the order of D::params (statmodels7 names some differently: sigma2 for v, mean/var for m/v)

  auto sorted = yh;
  std::sort(sorted.begin(), sorted.end());
  const std::pair<double, double> middle{sorted[(n - 1) / 2], sorted[n / 2]};
  const auto y = fb::to_device(yh);
  const auto theta0 = fb::library_start(D{}, yh);
  for (const std::string start : {"library", "poor"}) {
    for (const auto& r : fit_all<D>(y, theta0, fb::aggregate_builder<D>{}, start)) {
      const std::string label = name + " n = " + std::to_string(n);
      print_run(label, r);
      INFO(label << ", " << r.method << ": " << status_name(r.report.status) << " after " << r.report.iterations
                 << " iterations, decrement " << r.report.decrement);
      CHECK(r.report.status == lo::optimizer_status::converged);
      CHECK_FALSE(r.boundary);
      const double loglik = -r.report.value;
      CHECK(loglik >= ref[0].loglik - 2.0 * stop_tol - 16.0 * eps * std::abs(ref[0].loglik));
      for (int j = 0; j < P; ++j) {
        INFO(D::params[j].name << " = " << r.theta[j] << ", closed form " << ref[j].closed_form_mle
                               << ", statmodels7 " << ref[j].estimate << ", bound " << r.bound[j]);
        if (D::params[j].regularity < 2) {
          // l is piecewise linear in mu with kinks at the data: at even n every mu between the
          // two middle order statistics is an MLE (the reference holds their midpoint)
          CHECK(r.theta[j] >= middle.first - r.bound[j]);
          CHECK(r.theta[j] <= middle.second + r.bound[j]);
          continue;
        }
        if (ref[j].has_closed_form) CHECK(std::abs(r.theta[j] - ref[j].closed_form_mle) <= r.bound[j]);
        // statmodels7's own distance from the optimum is bounded the same way (its stop at a
        // score of 1e-6 per observation is below a decrement of 1e-6 at these n)
        CHECK(std::abs(r.theta[j] - ref[j].estimate) <= 2.0 * r.bound[j]);
      }
    }
  }
}

// a discrete distrib on data simulated in C++, against its closed-form MLE (the mean)
template <class D, class Builder>
void simulated_case(const std::string& name, const D& truth, Builder b, double mle_divisor) {
  for (long n : {1000L, 100000L}) {
    const auto y = fb::simulate(truth, n, 20261115u + static_cast<std::uint64_t>(n));
    const auto yh = fb::to_host_vector(y);
    double mean = 0;
    for (double v : yh) mean += v;
    mean /= static_cast<double>(n);
    const double mle = mean / mle_divisor;
    const auto theta0 = fb::library_start(truth, yh);
    for (const std::string start : {"library", "poor"})
      for (const auto& r : fit_all<D>(y, theta0, b, start)) {
        const std::string label = name + " n = " + std::to_string(n);
        print_run(label, r);
        INFO(label << ", " << r.method << ": " << status_name(r.report.status) << " after " << r.report.iterations
                   << " iterations, mu = " << r.theta[0] << ", MLE " << mle << ", bound " << r.bound[0]);
        CHECK(r.report.status == lo::optimizer_status::converged);
        CHECK_FALSE(r.boundary);
        CHECK(std::abs(r.theta[0] - mle) <= r.bound[0]);
      }
  }
}

// a sample whose MLE lies on the boundary of the domain: the start is strictly inside (a
// finite free coordinate); every run converges, correctly in the objective's units (-l
// within tol of its infimum 0, reached only at u = -inf or +inf), and the fit is flagged as
// on the boundary (Giovanni, 2026-11-15: the flag belongs to the fit, not to the optimizer)
template <class D, class Builder>
void boundary_case(const std::string& name, const D& prototype, Builder b, double value, long n) {
  const std::vector<double> yh(n, value);
  const auto theta0 = fb::library_start(prototype, yh);
  INFO(name << " on " << n << " values " << value << ", start " << theta0[0]);
  CHECK(theta0[0] > D::params[0].lower);
  CHECK(theta0[0] < D::params[0].upper);
  CHECK(std::isfinite(fb::default_bijector(D::params[0]).forward(theta0[0])));
  for (const auto& r : fit_all<D>(fb::to_device(yh), theta0, b, "library")) {
    print_run(name + " boundary", r);
    INFO(r.method << ": " << status_name(r.report.status) << " after " << r.report.iterations << " iterations, theta "
                  << r.theta[0] << ", -l " << r.report.value << ", decrement " << r.report.decrement);
    CHECK(r.report.status == lo::optimizer_status::converged);
    CHECK(r.report.value <= 2.0 * stop_tol);
    CHECK(r.boundary);
  }
}

// small samples and data far from the unit scale: genuine estimates, never flagged as on
// the boundary, except a sample that came out all zeros (bernoulli 0.01 at small n), which is
// boundary data
template <class D, class Builder>
void scale_case(const std::string& name, const D& truth, Builder b, long n, std::uint64_t seed) {
  const auto y = fb::simulate(truth, n, seed);
  const auto yh = fb::to_host_vector(y);
  bool all_zero = true;
  for (double v : yh) all_zero = all_zero && v == 0.0;
  for (const auto& r : fit_all<D>(y, fb::library_start(truth, yh), b, "library")) {
    const std::string label = name + " n = " + std::to_string(n);
    print_run(label, r);
    INFO(label << ", " << r.method << ": " << status_name(r.report.status) << ", largest free standard error "
               << r.se_free);
    CHECK(r.report.status == lo::optimizer_status::converged);
    CHECK(r.boundary == all_zero);
  }
}

}  // namespace

void fit_distrib_scale_cases() {
  for (long n : {2L, 3L, 5L, 10L, 1000L}) {
    scale_case("gaussian3 tau 1e-6", ld::gaussian3_distrib{1000.0, 1e-6}, fb::aggregate_builder<ld::gaussian3_distrib>{}, n, 11);
    scale_case("gaussian1 sigma 1e-4", ld::gaussian1_distrib{1e-3, 1e-4}, fb::aggregate_builder<ld::gaussian1_distrib>{}, n, 12);
    scale_case("gamma1 mu 1e3", ld::gamma1_distrib{1000.0, 0.5}, fb::aggregate_builder<ld::gamma1_distrib>{}, n, 13);
    scale_case("exponential mu 1e-5", ld::exponential_distrib{1e-5}, fb::aggregate_builder<ld::exponential_distrib>{}, n, 14);
    scale_case("bernoulli 0.5", ld::bernoulli_distrib{0.5}, fb::aggregate_builder<ld::bernoulli_distrib>{}, n, 15);
    scale_case("bernoulli 0.01", ld::bernoulli_distrib{0.01}, fb::aggregate_builder<ld::bernoulli_distrib>{}, n, 16);
    scale_case("geometric 0.05", ld::geometric_distrib{0.05}, fb::aggregate_builder<ld::geometric_distrib>{}, n, 17);
  }
}

TEST_CASE("fit_distrib bench: small samples and far scales are not on the boundary") { fit_distrib_scale_cases(); }

void fit_distrib_boundary_cases() {
  for (long n : {1000L, 100000L}) {
    boundary_case("bernoulli all 0", ld::bernoulli_distrib{}, fb::aggregate_builder<ld::bernoulli_distrib>{}, 0.0, n);
    boundary_case("bernoulli all 1", ld::bernoulli_distrib{}, fb::aggregate_builder<ld::bernoulli_distrib>{}, 1.0, n);
    boundary_case("geometric all 0", ld::geometric_distrib{}, fb::aggregate_builder<ld::geometric_distrib>{}, 0.0, n);
  }
}

// named functions with external linkage that run the cases (CLAUDE.md trap)
void fit_distrib_reference_cases() {
  for (long n : {1000L, 100000L}) {
    reference_case<ld::cauchy_distrib>("cauchy", n);
    reference_case<ld::chisq_distrib>("chisq", n);
    reference_case<ld::exponential_distrib>("exponential", n);
    reference_case<ld::gamma1_distrib>("gamma1", n);
    reference_case<ld::gamma2_distrib>("gamma2", n);
    reference_case<ld::gaussian1_distrib>("gaussian1", n);
    reference_case<ld::gaussian2_distrib>("gaussian2", n);
    reference_case<ld::gaussian3_distrib>("gaussian3", n);
    reference_case<ld::gumbel_distrib>("gumbel", n);
    reference_case<ld::invgauss_distrib>("invgauss", n);
    reference_case<ld::laplace1_distrib>("laplace1", n);
    reference_case<ld::laplace2_distrib>("laplace2", n);
    reference_case<ld::logistic_distrib>("logistic", n);
    reference_case<ld::lognormal1_distrib>("lognormal1", n);
    reference_case<ld::lognormal2_distrib>("lognormal2", n);
    reference_case<ld::poisson_distrib>("poisson", n);
  }
}

void fit_distrib_simulated_cases() {
  simulated_case("bernoulli", ld::bernoulli_distrib{0.3}, fb::aggregate_builder<ld::bernoulli_distrib>{}, 1.0);
  simulated_case("binomial", ld::binomial_distrib{0.3, 10.0}, fb::binomial_builder{10.0}, 10.0);
  simulated_case("geometric", ld::geometric_distrib{2.5}, fb::aggregate_builder<ld::geometric_distrib>{}, 1.0);
}

TEST_CASE("fit_distrib bench: 16 parametrizations against the closed forms and statmodels7") {
  fit_distrib_reference_cases();
}

TEST_CASE("fit_distrib bench: bernoulli, binomial and geometric on data simulated in C++") {
  fit_distrib_simulated_cases();
}

TEST_CASE("fit_distrib bench: an MLE on the boundary converges and is flagged") { fit_distrib_boundary_cases(); }

TEST_CASE("fit_distrib bench: the observed curvature is refused for a parameter with r_j < 2") {
  const auto y = fb::to_device(std::vector<double>{0.0, 1.0, 2.0});
  CHECK_THROWS_AS((fb::fit_distrib_objective<ld::laplace1_distrib>(y, fb::fit_curvature::observed)),
                  std::invalid_argument);
}
