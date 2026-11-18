// Penalties through a linear map fitted by the proximal methods, against the
// references frozen from R by tools/reference/optimizers_wave2_linear_map.R
// (tests/data/reference/optimizers/wave2, README there), on the regression test
// objective (regression_objective.hpp), whose sums run in a Kokkos kernel on the
// CPU or the GPU build.
//
// - the adaptive lasso, RSS/2 + r sum_j w_j |beta_j| with an unpenalized
//   intercept: laplace2 at rate r on u = W beta (linear_map_penalty over
//   diagonal_map), by proximal Newton (coordinate descent inside) and FISTA,
//   against glmnet with penalty.factor = w (r = n lambda_glmnet p / sum w);
// - the fused lasso, RSS/2 + lambda |D beta|_1 with D the first differences and
//   no intercept: laplace2 at rate lambda on u = D beta (difference_map<1>,
//   Condat's prox), by FISTA, against genlasso's fusedlasso1d. The other rows of
//   difference_fits.csv (trend filtering, the elastic net on differences) have no
//   prox and are the primal-dual method's.

#include "regression_objective.hpp"
#include "wave2_reference.hpp"

#include <doctest/doctest.h>

#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/optimizers/proximal_newton.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>
#include <laplace/penalty/linear_map_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace linear_map_fit_test {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
using regression_test::regression_loss;
using objective_type = regression_test::regression_test_objective<>;
using wave2_reference::read_rows;
using wave2_reference::to_double;

using lasso2_product = ld::independent_product_distrib<ld::univariate_as_multivariate<ld::laplace2_distrib>>;
using lasso2_penalty = lpen::distrib_penalty<lasso2_product>;
using adaptive_lasso_penalty = lpen::linear_map_penalty<lasso2_penalty, lpen::diagonal_map>;
using fused_lasso_penalty = lpen::linear_map_penalty<lasso2_penalty, lpen::difference_map<1>>;

lasso2_penalty make_lasso2(int d) {
  return lasso2_penalty(
      lasso2_product(ld::independent_product_component<ld::univariate_as_multivariate<ld::laplace2_distrib>>{{}, d, 1, nullptr}));
}

using wave2_reference::read_indices;

// one adaptive-lasso fit of adaptive_lasso_fits.csv with its coefficients
struct adaptive_lasso_fit {
  std::string dataset;
  double lambda_fraction = 0, rate = 0, objective_laplace = 0, penalty_constant = 0;
  std::vector<int> support;
  std::vector<double> coefficients;  // b0, beta1, ..., betap
};

std::vector<adaptive_lasso_fit> read_adaptive_lasso_fits(const std::string& dir) {
  std::vector<adaptive_lasso_fit> fits;
  for (const auto& r : read_rows(dir + "/adaptive_lasso_fits.csv")) {
    adaptive_lasso_fit f;
    f.dataset = r[0], f.lambda_fraction = to_double(r[1]), f.rate = to_double(r[3]);
    f.support = read_indices(r[6]);
    f.objective_laplace = to_double(r[8]), f.penalty_constant = to_double(r[9]);
    fits.push_back(f);
  }
  for (const auto& r : read_rows(dir + "/adaptive_lasso.csv"))
    for (auto& f : fits)
      if (f.dataset == r[0] && f.lambda_fraction == to_double(r[1])) f.coefficients.push_back(to_double(r[3]));
  return fits;
}

std::vector<double> read_adaptive_lasso_weights(const std::string& dir, const std::string& dataset) {
  std::vector<double> w;
  for (const auto& r : read_rows(dir + "/adaptive_lasso_weights.csv"))
    if (r[0] == dataset) w.push_back(to_double(r[2]));
  return w;
}

using wave2_reference::difference_dataset;
using wave2_reference::difference_fit;
using wave2_reference::load_difference_dataset;
using wave2_reference::read_difference_fits;

struct fit_comparison {
  lo::optimizer_report report;
  double coefficient_error = 0;  // max |b - b_ref| / max |b_ref|
  double objective_error = 0;    // f - f*
  bool same_pattern = false;     // the support (adaptive) or the nonzero differences (fused), exactly
};

template <class Objective, class Optimizer>
fit_comparison compare(Objective& objective, int p, const Optimizer& optimizer, const std::vector<double>& reference,
                       double reference_value, int first_penalized, int order, const std::vector<int>& pattern) {
  fit_comparison out;
  lo::optimizer_vector x0("x0", p);
  out.report = lo::minimize(optimizer, objective, x0);
  const auto& b = out.report.point;
  double scale = 0.0, err = 0.0;
  for (int j = 0; j < p; ++j) scale = std::max(scale, std::abs(reference[std::size_t(j)]));
  for (int j = 0; j < p; ++j) err = std::max(err, std::abs(b(j) - reference[std::size_t(j)]));
  out.coefficient_error = err / scale;
  out.objective_error = out.report.value - reference_value;
  std::vector<int> found;
  if (order == 0) {
    for (int j = first_penalized; j < p; ++j)
      if (b(j) != 0.0) found.push_back(j);
  } else {
    for (int i = 0; i + 1 < p; ++i)
      if (b(i + 1) - b(i) != 0.0) found.push_back(i + 1);
  }
  out.same_pattern = found == pattern;
  return out;
}

}  // namespace linear_map_fit_test

TEST_CASE("linear map fits: the adaptive lasso against glmnet with penalty.factor") {
  using namespace linear_map_fit_test;
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  const auto fits = read_adaptive_lasso_fits(dir);
  REQUIRE(fits.size() == 15);
  std::string loaded;
  wave2_reference::dataset d;
  std::vector<double> w;
  for (const auto& ref : fits) {
    if (ref.dataset != loaded) {
      d = wave2_reference::load_dataset(dir, ref.dataset), loaded = ref.dataset;
      w = read_adaptive_lasso_weights(dir, ref.dataset);
    }
    REQUIRE(w.size() == std::size_t(d.p));
    REQUIRE(ref.coefficients.size() == std::size_t(d.p) + 1);
    std::vector<double> x(std::size_t(d.n) * std::size_t(d.p + 1), 1.0);  // intercept column first
    std::copy(d.x.begin(), d.x.end(), x.begin() + d.n);
    objective_type smooth(x, d.y_gaussian, d.n, d.p + 1, regression_loss::gaussian);
    const adaptive_lasso_penalty penalty(make_lasso2(d.p), lpen::diagonal_map(w.data(), d.p));
    lo::penalized_objective<objective_type, adaptive_lasso_penalty> objective(
        smooth, d.p + 1, lo::attach_penalty(penalty, 1, {ref.rate}));
    static_assert(lo::coordinate_composite_objective<decltype(objective)>);
    CHECK(objective.nonsmooth_block_separable(0));

    const double f_reference = ref.objective_laplace + ref.penalty_constant;
    lo::proximal_newton_optimizer<> newton;
    newton.stop_rule.tolerance = 1e-10;
    lo::proximal_gradient_optimizer<> fista;
    fista.stop_rule.tolerance = std::max(1e-10, 1e3 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    const auto c1 = compare(objective, d.p + 1, newton, ref.coefficients, f_reference, 1, 0, ref.support);
    const auto c2 = compare(objective, d.p + 1, fista, ref.coefficients, f_reference, 1, 0, ref.support);
    MESSAGE(ref.dataset, " fraction ", ref.lambda_fraction, " support ", ref.support.size(), " | proximal Newton status ",
            int(c1.report.status), " it ", c1.report.iterations, " err ", c1.coefficient_error, " f-f* ",
            c1.objective_error, " supp ", c1.same_pattern, " | FISTA status ", int(c2.report.status), " it ",
            c2.report.iterations, " err ", c2.coefficient_error, " f-f* ", c2.objective_error, " measure ",
            c2.report.decrement, " supp ", c2.same_pattern);
    CHECK(c1.same_pattern);
    CHECK(c2.same_pattern);
    CHECK(lo::optimizer_succeeded(c1.report.status));
    CHECK(lo::optimizer_succeeded(c2.report.status));
    // measured on gcc and CUDA: proximal Newton's coefficients within 1.1e-6 and |f - f*| at most
    // 8.4e-11; FISTA's within 3.0e-6, f - f* at most 1.01 times its measure
    CHECK(c1.coefficient_error <= 5e-6);
    CHECK(c2.coefficient_error <= 1.5e-5);
    CHECK(std::abs(c1.objective_error) <= 4e-10);
    CHECK(c2.objective_error <= 1.5 * c2.report.decrement + 64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
  }
}

TEST_CASE("linear map fits: the fused lasso by FISTA with Condat's prox against genlasso") {
  using namespace linear_map_fit_test;
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  const auto fits = read_difference_fits(dir);
  REQUIRE(fits.size() == 18);
  std::string loaded;
  difference_dataset d;
  int tested = 0;
  for (const auto& ref : fits) {
    if (ref.penalty != "fused") continue;
    ++tested;
    if (ref.dataset != loaded) d = load_difference_dataset(dir, ref.dataset), loaded = ref.dataset;
    REQUIRE(ref.coefficients.size() == std::size_t(d.p));
    objective_type smooth(d.x, d.y_fused, d.n, d.p, regression_loss::gaussian);
    const fused_lasso_penalty penalty(make_lasso2(d.p - 1), lpen::difference_map<1>(d.p));
    lo::penalized_objective<objective_type, fused_lasso_penalty> objective(smooth, d.p,
                                                                           lo::attach_penalty(penalty, 0, {ref.lambda}));
    CHECK(!objective.nonsmooth_block_separable(0));
    // the penalty's value is lambda |D beta|_1 - (p - 1) log(lambda / 2) - log det(D D^T) / 2
    const double f_reference =
        ref.objective_laplace + (d.p - 1) * ref.penalty_constant_per_row - 0.5 * ref.log_det_ddt;
    CHECK(penalty.map().log_normalizer() == doctest::Approx(-0.5 * ref.log_det_ddt).epsilon(1e-14));
    lo::proximal_gradient_optimizer<> fista;
    fista.stop_rule.tolerance = std::max(1e-10, 1e3 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    const auto c = compare(objective, d.p, fista, ref.coefficients, f_reference, 0, 1, ref.nonzero_rows);
    MESSAGE(ref.dataset, " fraction ", ref.lambda_fraction, " nonzero differences ", ref.nonzero_rows.size(),
            " | FISTA status ", int(c.report.status), " it ", c.report.iterations, " values ", c.report.value_evaluations,
            " err ", c.coefficient_error, " f-f* ", c.objective_error, " measure ", c.report.decrement, " pattern ",
            c.same_pattern);
    CHECK(c.same_pattern);
    CHECK(lo::optimizer_succeeded(c.report.status));
    // measured on gcc and CUDA: coefficients within 6.1e-6; f - f* at most 0.99 times FISTA's
    // measure except on f2 at fraction 0.5, where f - f* (8.1e-10 on gcc, 9.3e-10 on CUDA,
    // against a measure of 5.6e-10) is 14 and 16 ulps of f = 3.4e5, inside the rounding term
    CHECK(c.coefficient_error <= 3e-5);
    CHECK(c.objective_error <= 1.5 * c.report.decrement + 64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
  }
  CHECK(tested == 6);
}
