// The proximal methods on the non-convex penalties (note/optimizers.md,
// section 12, point 16): MCP (gamma = 3) and SCAD (a = 3.7) on the gaussian
// datasets of wave 2, against the fits of ncvreg frozen by
// tools/reference/optimizers_wave2_ncvreg.R (tests/data/reference/optimizers/
// wave2, README there).
//
// ncvreg minimizes RSS/(2n) + sum_j q(s_j |beta_j|; lambda, shape) on columns
// standardized by s_j; n times that is RSS/2 + sum_j q(w_j |beta_j|; sqrt(n)
// lambda, shape), w_j = sqrt(n) s_j, which is the gaussian test objective plus
// linear_map_penalty<mcp_penalty or scad_penalty, diagonal_map> (the
// conversion is derived in the script). Each fit is solved by
// proximal_newton_optimizer (coordinate descent inside, damped where H_jj <= rho_j),
// by proximal_gradient_optimizer (MFISTA, since rho > 0) and by
// multistart_optimizer over proximal Newton. Where the objective is convex
// (globally_convex in the reference) the minimum is unique and every method must
// reach ncvreg's point and support; elsewhere a method may stop at another local
// minimum, so each run must be stationary and the multistart must find ncvreg's
// value or a lower one.

#include "regression_objective.hpp"
#include "wave2_reference.hpp"

#include <doctest/doctest.h>

#include <laplace/optimizers/multistart.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/penalty/linear_map_penalty.hpp>
#include <laplace/penalty/nonconvex_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace nonconvex_fit_test {

namespace lo = laplace::optimizers;
namespace lpen = laplace::penalty;
using regression_test::regression_loss;
using objective_type = regression_test::regression_test_objective<>;
using wave2_reference::read_indices;
using wave2_reference::read_rows;
using wave2_reference::to_double;

using weighted_mcp = lpen::linear_map_penalty<lpen::mcp_penalty, lpen::diagonal_map>;
using weighted_scad = lpen::linear_map_penalty<lpen::scad_penalty, lpen::diagonal_map>;

// one row of ncvreg_fits.csv with its polished coefficients
struct ncvreg_fit {
  std::string dataset, penalty;
  double shape = 0, lambda_fraction = 0, lambda_laplace = 0, objective_laplace = 0, sum_log_weights = 0;
  std::vector<int> support;  // 1-based indices of beta
  bool globally_convex = false;
  std::vector<double> coefficients;  // b0, beta1, ..., betap
};

std::vector<ncvreg_fit> read_ncvreg_fits(const std::string& dir) {
  std::vector<ncvreg_fit> fits;
  for (const auto& r : read_rows(dir + "/ncvreg_fits.csv")) {
    ncvreg_fit f;
    f.dataset = r[0], f.penalty = r[1], f.shape = to_double(r[2]), f.lambda_fraction = to_double(r[3]);
    f.lambda_laplace = to_double(r[5]);
    f.support = read_indices(r[7]);
    f.objective_laplace = to_double(r[9]), f.sum_log_weights = to_double(r[12]);
    f.globally_convex = r[19] == "TRUE";
    fits.push_back(f);
  }
  for (const auto& r : read_rows(dir + "/ncvreg.csv"))
    for (auto& f : fits)
      if (f.dataset == r[0] && f.penalty == r[1] && f.lambda_fraction == to_double(r[2]))
        f.coefficients.push_back(to_double(r[4]));
  return fits;
}

std::vector<double> read_ncvreg_weights(const std::string& dir, const std::string& dataset) {
  std::vector<double> w;
  for (const auto& r : read_rows(dir + "/ncvreg_weights.csv"))
    if (r[0] == dataset) w.push_back(to_double(r[2]));
  return w;
}

// the largest violation of 0 in the Clarke subdifferential of RSS/2 + sum_j q(w_j |beta_j|),
// coordinate by coordinate (the intercept free), divided by lambda max_j w_j, the size of the kink
template <class Shape>
double stationarity_violation(objective_type& smooth, const lo::optimizer_point& x, const std::vector<double>& w,
                              double lambda, double shape) {
  const int n = static_cast<int>(x.extent(0));
  lo::optimizer_vector g("g", n);
  smooth.value_gradient(x, g);
  double worst = std::abs(g(0)), w_max = 0.0;
  for (int j = 1; j < n; ++j) {
    const double wj = w[std::size_t(j - 1)], b = x(j);
    w_max = std::max(w_max, wj);
    const double e = b == 0.0 ? std::max(0.0, std::abs(g(j)) - lambda * wj)
                              : std::abs(g(j) + std::copysign(wj * Shape::q1(wj * std::abs(b), lambda, shape), b));
    worst = std::max(worst, e);
  }
  return worst / (lambda * w_max);
}

struct fit_outcome {
  lo::optimizer_report report;
  double coefficient_error = 0, objective_error = 0, violation = 0;
  bool same_support = false;
};

template <class Shape>
fit_outcome assess(const lo::optimizer_report& r, const ncvreg_fit& ref, double f_reference, objective_type& smooth,
                   const std::vector<double>& w) {
  fit_outcome c;
  c.report = r;
  const std::size_t p = ref.coefficients.size();
  double scale = 0.0, err = 0.0;
  for (std::size_t j = 0; j < p; ++j) scale = std::max(scale, std::abs(ref.coefficients[j]));
  for (std::size_t j = 0; j < p; ++j) err = std::max(err, std::abs(r.point(j) - ref.coefficients[j]));
  c.coefficient_error = err / scale;
  c.objective_error = r.value - f_reference;
  std::vector<int> support;
  for (std::size_t j = 1; j < p; ++j)
    if (r.point(j) != 0.0) support.push_back(int(j));
  c.same_support = support == ref.support;
  c.violation = stationarity_violation<Shape>(smooth, r.point, w, ref.lambda_laplace, ref.shape);
  return c;
}

template <class Shape, class P>
void check_fit(const wave2_reference::dataset& d, const ncvreg_fit& ref, const std::vector<double>& w, const P& penalty) {
  const int p = d.p + 1;
  std::vector<double> x(std::size_t(d.n) * std::size_t(p), 1.0);  // intercept column first
  std::copy(d.x.begin(), d.x.end(), x.begin() + d.n);
  objective_type smooth(x, d.y_gaussian, d.n, p, regression_loss::gaussian);
  lo::penalized_objective<objective_type, P> objective(smooth, p, lo::attach_penalty(penalty, 1, {ref.lambda_laplace, ref.shape}));
  static_assert(lo::coordinate_composite_objective<decltype(objective)>);
  // an unnormalizable P keeps its class through the map: the value adds no c(W)
  const double f_reference = ref.objective_laplace;
  lo::optimizer_vector x0("x0", p);  // zeros

  lo::proximal_newton_optimizer<> newton;
  newton.stop_rule.tolerance = 1e-10;
  lo::proximal_gradient_optimizer<> mfista;
  mfista.stop_rule.tolerance = std::max(1e-10, 1e3 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
  lo::multistart_optimizer<lo::proximal_gradient_optimizer<>> multistart;  // 10 starts, spread 1
  multistart.optimizer = mfista;
  multistart.distinct_tolerance = 1e-6 * std::max(1.0, std::abs(f_reference));

  const auto c1 = assess<Shape>(lo::minimize(newton, objective, x0), ref, f_reference, smooth, w);
  const auto c2 = assess<Shape>(lo::minimize(mfista, objective, x0), ref, f_reference, smooth, w);
  const auto ms = multistart.run_multistart(objective, x0);
  const auto c3 = assess<Shape>(ms.best, ref, f_reference, smooth, w);
  double worst_start = 0.0;  // the largest stationarity violation over the converged starts
  for (const auto& r : ms.runs)
    if (lo::optimizer_succeeded(r.status))
      worst_start = std::max(worst_start, stationarity_violation<Shape>(smooth, r.point, w, ref.lambda_laplace, ref.shape));
  MESSAGE(ref.dataset, " ", ref.penalty, " fraction ", ref.lambda_fraction, " support ", ref.support.size(),
          std::string(ref.globally_convex ? " convex" : " non-convex"), " | proximal Newton status ",
          int(c1.report.status), " it ", c1.report.iterations, " damped ", c1.report.damped_coordinates, " factor ",
          int(c1.report.final_factorization), " decrement ", c1.report.decrement, " err ", c1.coefficient_error,
          " f-f* ", c1.objective_error, " viol ", c1.violation, " supp ", c1.same_support, " | MFISTA status ",
          int(c2.report.status), " it ", c2.report.iterations, " err ", c2.coefficient_error, " f-f* ",
          c2.objective_error, " measure ", c2.report.decrement, " viol ", c2.violation, " supp ", c2.same_support,
          " | multistart converged ", ms.converged_count, "/", ms.start_count, " optima ", ms.distinct_optimum_count,
          " f-f* ", c3.objective_error, " measure ", c3.report.decrement, " err ", c3.coefficient_error, " supp ",
          c3.same_support, " worst start viol ", worst_start);
  const double rounding = 64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference);
  CHECK(lo::optimizer_succeeded(c1.report.status));
  CHECK(lo::optimizer_succeeded(c2.report.status));
  CHECK(lo::optimizer_succeeded(c3.report.status));
  // proximal Newton's decrement is the fallback measure exactly on d2 at 0.01, where the
  // free coordinates (many in the flat tails) outnumber the rows
  CHECK(c1.report.decrement_from_prox_mapping == (ref.dataset == "d2_n100_p200" && ref.lambda_fraction == 0.01));
  CHECK(!c2.report.decrement_from_prox_mapping);
  // X'X is positive semidefinite: the eigenvalue correction never acts, so the runs are those before it
  CHECK(c1.report.eigen_modified_iterations == 0);
  if (ref.dataset != "d2_n100_p200" || ref.lambda_fraction == 0.5) {
    // ncvreg's minimum is the one every method reaches from 0 (24 fits, measured on gcc and
    // CUDA): proximal Newton within 2.3e-6 and 1.5e-10, MFISTA within 2.8e-6 and 2.95 times
    // its measure, the exact support for all
    CHECK(c1.same_support);
    CHECK(c2.same_support);
    CHECK(c3.same_support);
    CHECK(c1.coefficient_error <= 5e-6);
    CHECK(std::abs(c1.objective_error) <= 4e-10);
    CHECK(c2.coefficient_error <= 1.5e-5);
    CHECK(c2.objective_error <= 4 * c2.report.decrement + rounding);
    CHECK(c3.objective_error <= 4 * c3.report.decrement + rounding);
  } else {
    // d2, p > n, below 0.5 of lambda_max: other local minima. Every converged run is
    // stationary; at 0.1 the multistart reaches ncvreg's value or a lower one, at 0.01
    // no route from 0 or random starts does (the gap is reported)
    CHECK(c1.violation <= 1e-5);
    CHECK(c2.violation <= 1e-5);
    CHECK(worst_start <= 1e-5);
    if (ref.lambda_fraction == 0.1) CHECK(c3.objective_error <= 4 * c3.report.decrement + rounding);
  }
}

}  // namespace nonconvex_fit_test

TEST_CASE("nonconvex fits: MCP and SCAD against the frozen ncvreg references" * doctest::test_suite("slow")) {
  using namespace nonconvex_fit_test;
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  const auto fits = read_ncvreg_fits(dir);
  REQUIRE(fits.size() == 30);
  std::string loaded;
  wave2_reference::dataset d;
  std::vector<double> w;
  for (const auto& ref : fits) {
    if (ref.dataset != loaded) {
      d = wave2_reference::load_dataset(dir, ref.dataset), loaded = ref.dataset;
      w = read_ncvreg_weights(dir, ref.dataset);
    }
    REQUIRE(w.size() == std::size_t(d.p));
    REQUIRE(ref.coefficients.size() == std::size_t(d.p) + 1);
    const lpen::diagonal_map map(w.data(), d.p);
    if (ref.penalty == "MCP")
      check_fit<lpen::detail_nonconvex::mcp_shape>(d, ref, w, weighted_mcp(lpen::mcp_penalty{d.p}, map));
    else
      check_fit<lpen::detail_nonconvex::scad_shape>(d, ref, w, weighted_scad(lpen::scad_penalty{d.p}, map));
  }
}
