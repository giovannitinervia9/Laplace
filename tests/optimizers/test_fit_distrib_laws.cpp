// The fit_distrib bench on the seventeen cases of the laws added after the first bench
// (tools/reference/optimizers_fit_distrib_laws.R, fit_distrib_laws.csv): R's MLE of a
// log-density independent of Laplace's code (distributions7, gamlss.dist's dPE, the
// definition of mcp and scad), and statmodels7's fit_distrib() where it has the law. The
// runs and the checks are those of test_fit_distrib.cpp (fit_distrib_runs.hpp), with the
// options below for the laws whose MLE of mu sits on a kink and for the +1 start.

#include <doctest/doctest.h>

#include "fit_distrib_runs.hpp"

#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/brent.hpp>
#include <laplace/optimizers/multistart.hpp>
#include <laplace/optimizers/trust_region.hpp>

namespace {

using namespace fit_distrib_runs;

// the relative difference between the objective's log-likelihood and R's at R's MLE: measured at
// most 1.2e-13 (pig, n = 1e5, 2026-11-19), times 1.5
constexpr double laws_parametrization_tol = 1.8e-13;

const std::vector<fb::laws_reference_row>& laws_reference_rows() {
  static const auto rows =
      fb::read_laws_reference_rows(LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/fit_distrib_laws.csv");
  return rows;
}

// What a law needs beyond the default runs (H3, decided with Giovanni on 2026-11-19).
//
// kink_at_mle: the MLE of mu lies on a data point (enet, mcp, scad: the nearest observation
// within 2.4e-9 of R's estimate), the kink of l in mu. The smooth methods reach the maximum
// there but their stopping rule cannot certify it, so they are asserted on the value only,
// within 6.3e-5 of R's log-likelihood at n = 1e3 and 2.7e-6 at n = 1e5 (measured at most
// 4.2e-5, enet BFGS, and 1.75e-6, scad Newton expected, times 1.5), with the status converged,
// stalled or line search failed. The certified fit is the alternation of the Laplace case in
// test_block_coordinate.cpp: Brent on mu, Newton with the expected curvature on the rest, its
// step capped at 1 in the free coordinates (alternation_cap), asserted converged (or converged
// at the resolution of Brent's search) within R's bound.
//
// The +1 start (every free coordinate moved by 1) sends some methods the wrong way; the
// maximum is interior in every case, so these are failures of the method from that start:
// - pseudohuber: uncapped Fisher scoring takes the full step on an expected curvature of 2.33
//   in log nu against a gradient of 30.6, so nu goes to 1.5e-4, then 1e28, then 1.3e-159, where
//   l is flat and it stalls (-l 2111.4 against 2091.98 at n = 1e3). The bench's Fisher scoring
//   caps the step at 1 (fit_distrib_bench.hpp, fisher_max_step) and converges; the trust region
//   is run as well (poor_trust_region).
// - enet: Fisher scoring takes logit alpha to about 8 in one step (alpha -> 1, the Laplace end)
//   and stays there with any step cap; the trust region and the capped alternation reach the MLE.
// - scad: the capped Fisher scoring and the capped alternation reach the MLE; BFGS does not
//   (the second basin below).
// - mcp: two basins. The profile likelihood in gamma has its maximum at gamma 3.5 (-2106.99 at
//   n = 1e3) and a supremum at gamma -> infinity, where the tail variance goes to 0 and the law
//   becomes the Laplace of rate lambda (-2166.82), separated by a valley (-2185.33 at gamma 30).
//   From the +1 start (gamma 8.2) every method tried ends in the far basin; what is asserted
//   is that each of those fits is flagged on the boundary (poor_boundary_only). scad has the
//   same two basins (maximum -1985.93 at a = 4, Laplace limit -2027.60, valley -2038.45 at a 10).
//   For both, a multistart over the capped Fisher scoring from the +1 start (10 starts, spread
//   0.5) has its best run, and the run it keeps, at R's maximum; both are asserted
//   (poor_multistart; decided with Giovanni on 2026-11-19).
// Runs named in poor_unasserted are printed, not asserted.
struct laws_options {
  bool kink_at_mle = false;
  std::vector<std::string> poor_unasserted;
  bool poor_trust_region = false;
  double alternation_cap = 1.0;
  bool poor_boundary_only = false;
  bool poor_multistart = false;
};

// one data set of the laws added after the first bench (optimizers_fit_distrib_laws.R):
// R's log-likelihood at R's MLE reproduced by the objective (the parametrizations agree),
// every run converged and not on the boundary, the log-likelihood at least R's, the
// estimate within the decrement bound of R's MLE, and statmodels7's estimate, where it is at
// R's maximum, within twice that bound (as for the first sixteen)
template <class D, class Builder>
void laws_case(const std::string& name, const D& prototype, Builder b, long n, const laws_options& opt = {}) {
  constexpr int P = D::n_params;
  const std::string base = LAPLACE_REFERENCE_DIR "/optimizers/fit_distrib/" + name;
  const auto yh = n == 1000 ? fb::read_hex_csv(base + "_n1000.csv") : fb::read_f64(base + "_n100000.f64");
  REQUIRE(static_cast<long>(yh.size()) == n);
  std::vector<fb::laws_reference_row> ref;
  for (const auto& r : laws_reference_rows())
    if (r.name == name && r.n == n) ref.push_back(r);
  REQUIRE(static_cast<int>(ref.size()) == P);
  const auto y = fb::to_device(yh);
  const std::string label = name + " n = " + std::to_string(n);
  using obj_t = fb::fit_distrib_objective<D, Builder>;
  obj_t expected(y, fb::fit_curvature::expected, b);
  {
    double t[P];
    for (int j = 0; j < P; ++j) t[j] = ref[j].mle;
    lo::optimizer_vector u("u", P);
    expected.free_of(t, u);
    const double ll = -expected.value(u), rel = std::abs(ll - ref[0].loglik_mle) / std::abs(ref[0].loglik_mle);
    if (verbose) std::printf("%-28s parametrization: log-likelihood at R's MLE, relative difference %.3g\n", label.c_str(), rel);
    INFO(label << ": Laplace " << ll << ", " << ref[0].reference << " " << ref[0].loglik_mle);
    CHECK(rel <= laws_parametrization_tol);
  }
  // R's MLE on the boundary of the domain (a parameter within 1e-6 of a finite end), where
  // the fit must be flagged rather than not; no reference data set has one
  bool mle_on_boundary = false;
  for (int j = 0; j < P; ++j) {
    const double lo_end = D::params[j].lower, hi_end = D::params[j].upper;
    if (std::isfinite(lo_end) && ref[j].mle - lo_end <= 1e-6 * std::fmax(1.0, std::abs(lo_end))) mle_on_boundary = true;
    if (std::isfinite(hi_end) && hi_end - ref[j].mle <= 1e-6 * std::fmax(1.0, std::abs(hi_end))) mle_on_boundary = true;
  }
  // statmodels7's estimate is compared only where its log-likelihood is R's maximum's; where
  // it stopped lower (enet: by 10.1 at n = 1e3, by 2056 at n = 1e5) the fit must reach above it
  const double ll_scale = 2.0 * stop_tol + 16.0 * eps * std::abs(ref[0].loglik_mle);
  const bool statmodels7_at_mle = ref[0].has_statmodels7 && ref[0].statmodels7_converged &&
                                  std::abs(ref[0].loglik_statmodels7 - ref[0].loglik_mle) <= ll_scale;
  const double kink_ll_tol = n == 1000 ? 6.3e-5 : 2.7e-6;
  const auto theta0 = fb::library_start(prototype, yh);
  // smooth: a smooth method (value-only assertion at a kink); alternation: Brent's search
  // may also stop at its resolution
  auto check_run = [&](const fit_run& r, bool smooth_method, bool poor) {
    print_run(label, r);
    INFO(label << ", " << r.method << ": " << status_name(r.report.status) << " after " << r.report.iterations
               << " iterations, decrement " << r.report.decrement);
    if (poor && opt.poor_boundary_only) {
      CHECK(r.boundary);
      return;
    }
    const double loglik = -r.report.value;
    if (opt.kink_at_mle && smooth_method) {
      CHECK((r.report.status == lo::optimizer_status::converged || r.report.status == lo::optimizer_status::stalled ||
             r.report.status == lo::optimizer_status::line_search_failed));
      CHECK(loglik >= ref[0].loglik_mle - kink_ll_tol);
      return;
    }
    if (smooth_method) CHECK(r.report.status == lo::optimizer_status::converged);
    else CHECK((r.report.status == lo::optimizer_status::converged ||
                r.report.status == lo::optimizer_status::converged_at_resolution));
    CHECK(r.boundary == mle_on_boundary);
    CHECK(loglik >= ref[0].loglik_mle - ll_scale);
    if (ref[0].has_statmodels7) CHECK(loglik >= ref[0].loglik_statmodels7 - ll_scale);
    for (int j = 0; j < P; ++j) {
      INFO(D::params[j].name << " = " << r.theta[j] << ", R's MLE " << ref[j].mle << ", statmodels7 "
                             << ref[j].statmodels7 << ", bound " << r.bound[j]);
      // mu of the alternation is checked through the value only: l is piecewise in mu with a
      // kink at every observation, the decrement bound (a quadratic model) does not hold there,
      // and a neighbouring observation can be within the value bound (scad at n = 1e5 from the
      // +1 start: mu 3.3e-5 from R's, -l 7.8e-7 above it, Brent's search size 1.9e-8)
      if (!smooth_method && D::params[j].regularity < 2) continue;
      if (!mle_on_boundary) CHECK(std::abs(r.theta[j] - ref[j].mle) <= r.bound[j]);
      if (statmodels7_at_mle && !mle_on_boundary) CHECK(std::abs(r.theta[j] - ref[j].statmodels7) <= 2.0 * r.bound[j]);
    }
  };
  auto unasserted = [&](const fit_run& r) {
    for (const auto& m : opt.poor_unasserted)
      if (r.method.rfind(m, 0) == 0 && r.method.find("poor") != std::string::npos) return true;
    return false;
  };
  for (const std::string start : {"library", "poor"}) {
    const bool poor = start == "poor";
    lo::optimizer_vector u("u", P);
    expected.free_of(theta0.data(), u);
    std::vector<double> u0(u.data(), u.data() + P);
    if (poor)
      for (double& x : u0) x += 1.0;
    std::vector<std::pair<fit_run, bool>> runs;  // (run, smooth method)
    for (auto& r : fit_all<D>(y, theta0, b, start)) runs.emplace_back(std::move(r), true);
    if (poor && opt.poor_trust_region)
      runs.emplace_back(run_one("trust region expected, poor", lo::trust_region_optimizer<>{}, expected, u0), true);
    if (opt.kink_at_mle) {
      lo::block_coordinate_optimizer<lo::brent_optimizer<>, lo::newton_optimizer<>> alternation;
      alternation.block_sizes = {1, P - 1};
      std::get<1>(alternation.sub_optimizers).max_step = opt.alternation_cap;
      runs.emplace_back(run_one("alternation, " + start, alternation, expected, u0), false);
    }
    for (const auto& [r, smooth_method] : runs) {
      if (unasserted(r)) print_run(label + " (unasserted)", r);
      else check_run(r, smooth_method, poor);
    }
    // two basins: from the +1 start the bench's Fisher scoring (step capped at 1) is restarted
    // from 10 Latin-hypercube starts within 1.5 max(1, |u0_j|) of the start, and the best of its
    // runs (lowest -l) must be at R's maximum and off the boundary (measured: -l at most 9.7e-6
    // above R's at n = 1e3, 1.1e-6 at n = 1e5, against the kink bounds 6.3e-5 and 2.7e-6; 3 of
    // the 10 runs get there at n = 1e3, each stalled on the kink in mu). The run the multistart
    // keeps must be there too: a boundary run can end converged under Fisher scoring (one does
    // at n = 1e3 for both laws, -l 2186.84 for mcp and 2033.30 for scad), and the selection
    // compares values first, so that converged run does not displace the lower stalled ones.
    if (poor && opt.poor_multistart) {
      lo::multistart_optimizer<lo::newton_optimizer<>> ms{fb::fisher_scoring_optimizer()};
      ms.start_count = 10, ms.spread = 0.5;
      lo::optimizer_vector x0("x0", P);
      for (int j = 0; j < P; ++j) x0(j) = u0[j];
      const auto m = ms.run_multistart(expected, x0);
      const lo::optimizer_report* best = nullptr;
      for (const auto& r : m.runs)
        if (std::isfinite(r.value) && (!best || r.value < best->value)) best = &r;
      REQUIRE(best != nullptr);
      MESSAGE(label << ", multistart Fisher scoring, poor: kept -l " << m.best.value << " (status "
                    << int(m.best.status) << ", boundary " << expected.on_boundary(m.best.point)
                    << "), best run -l " << best->value << ", R's " << -ref[0].loglik_mle);
      CHECK(!expected.on_boundary(best->point));
      CHECK(-best->value >= ref[0].loglik_mle - kink_ll_tol);
      CHECK(!expected.on_boundary(m.best.point));
      CHECK(-m.best.value >= ref[0].loglik_mle - kink_ll_tol);
    }
  }
}
}  // namespace

void fit_distrib_laws_cases() {
  using fb::aggregate_builder;
  for (long n : {1000L, 100000L}) {
    laws_case("student_t1", ld::student_t1_distrib{}, aggregate_builder<ld::student_t1_distrib>{}, n);
    laws_case("student_t2", ld::student_t2_distrib{}, aggregate_builder<ld::student_t2_distrib>{}, n);
    laws_case("skewnormal", ld::skewnormal_distrib{}, aggregate_builder<ld::skewnormal_distrib>{}, n);
    laws_case("skewt", ld::skewt_distrib{}, aggregate_builder<ld::skewt_distrib>{}, n);
    laws_case("pseudohuber", ld::pseudohuber_distrib{}, aggregate_builder<ld::pseudohuber_distrib>{}, n,
              {.poor_trust_region = true});
    laws_case("weibull", ld::weibull_distrib{}, aggregate_builder<ld::weibull_distrib>{}, n);
    laws_case("vonmises", ld::vonmises_distrib{}, aggregate_builder<ld::vonmises_distrib>{}, n);
    laws_case("beta", ld::beta_distrib{}, aggregate_builder<ld::beta_distrib>{}, n);
    const ld::betabinom_distrib bb{0.5, 1.0, 10.0};
    laws_case("betabinom", bb, fb::prototype_builder<ld::betabinom_distrib>{bb}, n);
    laws_case("negbin1", ld::negbin1_distrib{}, aggregate_builder<ld::negbin1_distrib>{}, n);
    laws_case("negbin2", ld::negbin2_distrib{}, aggregate_builder<ld::negbin2_distrib>{}, n);
    laws_case("pig", ld::pig_distrib{}, aggregate_builder<ld::pig_distrib>{}, n);
    laws_case("enet", ld::enet_distrib{}, aggregate_builder<ld::enet_distrib>{}, n,
              {.kink_at_mle = true, .poor_unasserted = {"Newton expected,"}, .poor_trust_region = true});
    laws_case("powerexp_p1.5", ld::powerexp_distrib{}, aggregate_builder<ld::powerexp_distrib>{}, n);
    laws_case("powerexp_p3", ld::powerexp_distrib{}, aggregate_builder<ld::powerexp_distrib>{}, n);
    laws_case("mcp", ld::mcp_distrib<ld::gaussian_tail>{}, aggregate_builder<ld::mcp_distrib<ld::gaussian_tail>>{}, n,
              {.kink_at_mle = true, .poor_boundary_only = true, .poor_multistart = true});
    laws_case("scad", ld::scad_distrib<ld::gaussian_tail>{}, aggregate_builder<ld::scad_distrib<ld::gaussian_tail>>{}, n,
              {.kink_at_mle = true, .poor_unasserted = {"BFGS"}, .poor_multistart = true});
  }
}

TEST_CASE("fit_distrib bench: 17 laws added after the first bench against R's MLE and statmodels7") {
  fit_distrib_laws_cases();
}
