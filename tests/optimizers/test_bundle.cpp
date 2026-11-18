// Tests of the proximal bundle method (note/optimizers.md, section 12, decision
// 7): least absolute deviations on the four wave-2 datasets with p < n against
// the exact vertices frozen from quantreg::rq (tests/data/reference/
// optimizers/wave2), a weighted sum of absolute values and a maximum of affine
// functions with known minima, the Laplace distribution with covariates (beta
// by the bundle method, log sigma by Newton, in block alternation) against the
// closed form sigma_hat = sum |r| / n at the vertex, an objective finite only at
// its start (an honest status, never converged), and the measurement asked by
// decision 7: Newton with the expected information X'X and BFGS on the same LAD
// problems, printed, not checked. LAPLACE_TEST_VERBOSE=1 prints the runs.

#include "regression_objective.hpp"
#include "wave2_reference.hpp"

#include <doctest/doctest.h>

#include <laplace/optimizers/bfgs.hpp>
#include <laplace/optimizers/block_coordinate.hpp>
#include <laplace/optimizers/bundle.hpp>
#include <laplace/optimizers/minimize.hpp>
#include <laplace/optimizers/newton.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace bundle_test {

namespace lo = laplace::optimizers;
using regression_test::regression_loss;
using objective_type = regression_test::regression_test_objective<>;
constexpr double eps = std::numeric_limits<double>::epsilon();

inline bool verbose() {
  const char* v = std::getenv("LAPLACE_TEST_VERBOSE");
  return v != nullptr && v[0] == '1';
}

inline lo::optimizer_vector filled(int n, double value) {
  lo::optimizer_vector x("x", n);
  for (int i = 0; i < n; ++i) x(i) = value;
  return x;
}

// f(x) = c0 + sum_i w_i |x_i - c_i|, minimum c0 at c
struct weighted_abs_sum {
  std::vector<double> w, c;
  double c0;
  double value(const lo::optimizer_point& x) {
    double f = c0;
    for (std::size_t i = 0; i < w.size(); ++i) f += w[i] * std::abs(x(i) - c[i]);
    return f;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    for (std::size_t i = 0; i < w.size(); ++i) g(i) = x(i) > c[i] ? w[i] : (x(i) < c[i] ? -w[i] : 0.0);
    return value(x);
  }
};

// f(x) = b + max_j a_j'(x - c) with the a_j in pairs +-a, spanning R^n: the
// minimum b at c, unique, every piece active there
struct max_of_affine {
  int n;
  std::vector<std::vector<double>> a;
  std::vector<double> c;
  double b;
  double value(const lo::optimizer_point& x) {
    lo::optimizer_vector g("g", n);
    return value_gradient(x, g);
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    double best = -std::numeric_limits<double>::infinity();
    std::size_t arg = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      double v = 0.0;
      for (int i = 0; i < n; ++i) v += a[j][std::size_t(i)] * (x(i) - c[std::size_t(i)]);
      if (v > best) best = v, arg = j;
    }
    for (int i = 0; i < n; ++i) g(i) = a[arg][std::size_t(i)];
    return b + best;
  }
};

// finite at its start only (optimizers7's false convergence)
struct finite_at_start_only {
  double value(const lo::optimizer_point& x) { return x(0) == 3.0 && x(1) == -2.0 ? 5.0 : std::numeric_limits<double>::infinity(); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    g(0) = 1.0, g(1) = -1.0;
    return value(x);
  }
};

// The laplace1 negative log-likelihood with covariates, in (beta, u = log sigma):
// f = e^-u S(beta) + n u + n log 2, S = sum |y_i - x_i'beta|. The beta block of
// the curvature is the expected information e^-2u X'X, the u entry the observed
// e^-u S (positive), the cross terms -e^-u g_S.
struct laplace_regression_objective {
  using curvature_type = lo::dense_curvature_matrix;
  objective_type& lad;  // sigma = 1
  int p;
  lo::optimizer_vector beta, g_beta;
  curvature_type XtX;
  laplace_regression_objective(objective_type& o, int p_)
      : lad(o), p(p_), beta("beta", p_), g_beta("g_beta", p_), XtX("XtX", p_, p_) {}
  double split(const lo::optimizer_point& x) {
    for (int j = 0; j < p; ++j) beta(j) = x(j);
    return x(p);
  }
  double value(const lo::optimizer_point& x) {
    const double u = split(x);
    return std::exp(-u) * lad.value(beta) + lad.rows() * (u + std::log(2.0));
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    const double u = split(x), e = std::exp(-u);
    const double S = lad.value_gradient(beta, g_beta);
    for (int j = 0; j < p; ++j) g(j) = e * g_beta(j);
    g(p) = -e * S + lad.rows();
    return e * S + lad.rows() * (u + std::log(2.0));
  }
  double value_gradient_curvature(const lo::optimizer_point& x, const lo::optimizer_vector& g, curvature_type& H) {
    const double f = value_gradient(x, g);
    const double u = x(p), e = std::exp(-u);
    lad.value_gradient_curvature(beta, g_beta, XtX);
    const double S = std::exp(u) * (f - lad.rows() * (u + std::log(2.0)));
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b < p; ++b) H(a, b) = e * e * XtX(a, b);
      H(a, p) = H(p, a) = -g(a);
    }
    H(p, p) = e * S;
    return f;
  }
};

// a frozen LAD fit: the data with an intercept column first, the vertex, S*
struct lad_reference {
  wave2_reference::dataset data;
  std::vector<double> x;  // n x (p + 1), column-major, ones first
  std::vector<double> vertex;
  double objective = 0, sigma_hat = 0, negloglik = 0;
};

inline std::vector<lad_reference> load_lad_references() {
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  std::vector<lad_reference> refs;
  for (const auto& r : wave2_reference::read_rows(dir + "/rq_fits.csv")) {
    lad_reference ref;
    ref.data = wave2_reference::load_dataset(dir, r[0]);
    ref.objective = wave2_reference::to_double(r[3]);
    ref.sigma_hat = wave2_reference::to_double(r[5]);
    ref.negloglik = wave2_reference::to_double(r[6]);
    const auto& d = ref.data;
    ref.x.assign(std::size_t(d.n) * std::size_t(d.p + 1), 1.0);
    std::copy(d.x.begin(), d.x.end(), ref.x.begin() + d.n);
    for (const auto& c : wave2_reference::read_rows(dir + "/rq.csv"))
      if (c[0] == r[0]) ref.vertex.push_back(wave2_reference::to_double(c[2]));  // b0, beta1, ...
    refs.push_back(std::move(ref));
  }
  return refs;
}

inline double distance(const lo::optimizer_point& x, const std::vector<double>& ref) {
  double d = 0.0;
  for (std::size_t j = 0; j < ref.size(); ++j) d = std::max(d, std::abs(x(j) - ref[j]));
  return d;
}

inline void print_report(const char* label, const lo::optimizer_report& r, double distance_to_reference,
                         double excess) {
  if (verbose())
    std::printf("  %-34s status %d it %6d f %6ld g %6ld H %4ld t %.3fs measure %.3g |x-x*| %.3g f-f* %.3g\n", label,
                int(r.status), r.iterations, r.value_evaluations, r.gradient_evaluations, r.curvature_evaluations,
                r.elapsed_seconds, r.decrement, distance_to_reference, excess);
}

// ------------------------------------------------------------- the cases

void synthetic_cases() {
  {  // weighted sum of absolute values, n = 20, from 0
    const int n = 20;
    weighted_abs_sum f{std::vector<double>(n), std::vector<double>(n), 7.0};
    for (int i = 0; i < n; ++i) f.w[std::size_t(i)] = 1.0 + i, f.c[std::size_t(i)] = 0.3 * i - 2.0;
    lo::bundle_optimizer<> bundle;
    const auto r = lo::minimize(bundle, f, filled(n, 0.0));
    print_report("abs_sum n 20", r, distance(r.point, f.c), r.value - f.c0);
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(r.value - f.c0 >= 0.0);
    CHECK(r.value - f.c0 <= 2.0 * r.decrement + 64 * eps * f.c0);
    CHECK(r.decrement <= 1e-6);
  }
  {  // a maximum of 2 n + 4 affine functions, n = 10, minimum 2.5 at c
    const int n = 10;
    regression_test::regression_rng rng{20261118u};
    max_of_affine f{n, {}, std::vector<double>(n), 2.5};
    for (int i = 0; i < n; ++i) f.c[std::size_t(i)] = rng.normal();
    for (int j = 0; j < n + 2; ++j) {
      std::vector<double> a(static_cast<std::size_t>(n));
      for (double& v : a) v = rng.normal();
      f.a.push_back(a);
      for (double& v : a) v = -v;
      f.a.push_back(a);
    }
    lo::bundle_optimizer<> bundle;
    const auto r = lo::minimize(bundle, f, filled(n, 1.0));
    double dist = 0.0;
    for (int i = 0; i < n; ++i) dist = std::max(dist, std::abs(r.point(i) - f.c[std::size_t(i)]));
    print_report("max of affine n 10", r, dist, r.value - f.b);
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(r.value - f.b >= 0.0);
    CHECK(r.value - f.b <= 2.0 * r.decrement + 64 * eps * f.b);
  }
  {  // finite only at the start: every trial refused, t collapses, never converged
    finite_at_start_only f;
    lo::optimizer_vector x0("x0", 2);
    x0(0) = 3.0, x0(1) = -2.0;
    lo::bundle_optimizer<> bundle;
    const auto r = lo::minimize(bundle, f, x0);
    print_report("finite at the start only", r, 0.0, 0.0);
    CHECK(r.status == lo::optimizer_status::stalled);
    CHECK(r.value == 5.0);
    CHECK(r.decrement > 0.5);  // a model of one cut certifies nothing (measured 141)
  }
  {  // a start where f is not finite
    finite_at_start_only f;
    lo::bundle_optimizer<> bundle;
    CHECK(lo::minimize(bundle, f, filled(2, 0.0)).status == lo::optimizer_status::start_not_admissible);
  }
}

void lad_cases() {
  for (auto& ref : load_lad_references()) {
    const auto& d = ref.data;
    const int p = d.p + 1;
    REQUIRE(ref.vertex.size() == std::size_t(p));
    objective_type lad(ref.x, d.y_lad, d.n, p, regression_loss::laplace);
    // the default rule (decrement 1e-6) checks f; the measure bounds f - f* (the
    // margin 2 is not needed by any run so far). The run may stop before the vertex:
    // 3.5e-6 from it on d4, f - f* = 6e-8 under a measure of 7.8e-7 (measured)
    lo::bundle_optimizer<> bundle;
    const auto r = lo::minimize(bundle, lad, filled(p, 0.0));
    const double excess = r.value - ref.objective;
    print_report((d.id + " LAD bundle").c_str(), r, distance(r.point, ref.vertex), excess);
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(excess >= -64 * eps * ref.objective);
    CHECK(excess <= 2.0 * r.decrement + 64 * eps * ref.objective);
    CHECK(distance(r.point, ref.vertex) <= 1e-4);
    // at a tight decrement the method lands on the vertex itself; the tolerance keeps
    // clear of the resolution of f (at 1e-10, f = 9.8e3 on d4 and 1.0e5 on d5, two of
    // eight runs over thread counts ended stalled: 64 eps |f| is 1.4e-10 and 1.4e-9)
    lo::bundle_optimizer<> tight;
    tight.stop_rule.tolerance = std::max(1e-10, 1e3 * eps * ref.objective);
    const auto rt = lo::minimize(tight, lad, filled(p, 0.0));
    print_report((d.id + " LAD bundle, tol 1e-10").c_str(), rt, distance(rt.point, ref.vertex), rt.value - ref.objective);
    CHECK(lo::optimizer_succeeded(rt.status));
    CHECK(rt.value - ref.objective <= 2.0 * rt.decrement + 64 * eps * ref.objective);
    CHECK(distance(rt.point, ref.vertex) <= 1e-8);
  }
}

void laplace_alternation_cases() {
  for (auto& ref : load_lad_references()) {
    const auto& d = ref.data;
    const int p = d.p + 1;
    objective_type lad(ref.x, d.y_lad, d.n, p, regression_loss::laplace);
    laplace_regression_objective f(lad, p);
    lo::block_coordinate_optimizer<lo::bundle_optimizer<>, lo::newton_optimizer<>> alternation;
    alternation.block_sizes = {p, 1};
    const auto r = lo::minimize(alternation, f, filled(p + 1, 0.0));
    const double sigma = std::exp(r.point(p));
    print_report((d.id + " laplace1 alternation").c_str(), r, distance(r.point, ref.vertex), r.value - ref.negloglik);
    if (verbose()) std::printf("    sigma %.17g sigma_hat %.17g\n", sigma, ref.sigma_hat);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(distance(r.point, ref.vertex) <= 1e-4);  // as LAD at the default rule
    // the decrement tolerance bounds the relative error of sigma by about sqrt(2 tol / n)
    CHECK(std::abs(sigma / ref.sigma_hat - 1.0) <= 2.0 * std::sqrt(2.0 * 1e-6 / d.n) + 1e-12);
    CHECK(r.value - ref.negloglik >= -64 * eps * ref.negloglik);
    CHECK(r.value - ref.negloglik <= 2e-6 + 64 * eps * ref.negloglik);
  }
}

// Decision 7's measurement, not a check: Newton with the expected information
// X'X and BFGS on LAD, statuses, distance to the vertex, evaluations.
void smooth_methods_on_lad() {
  for (auto& ref : load_lad_references()) {
    const auto& d = ref.data;
    const int p = d.p + 1;
    objective_type lad(ref.x, d.y_lad, d.n, p, regression_loss::laplace);
    lo::newton_optimizer<> newton;
    const auto rn = lo::minimize(newton, lad, filled(p, 0.0));
    print_report((d.id + " LAD Newton, X'X").c_str(), rn, distance(rn.point, ref.vertex), rn.value - ref.objective);
    lo::bfgs_optimizer<> bfgs;
    bfgs.max_iterations = 5000;
    const auto rb = lo::minimize(bfgs, lad, filled(p, 0.0));
    print_report((d.id + " LAD BFGS").c_str(), rb, distance(rb.point, ref.vertex), rb.value - ref.objective);
    CHECK(rn.value >= ref.objective * (1.0 - 64 * eps));  // nothing below the LP's optimum
    CHECK(rb.value >= ref.objective * (1.0 - 64 * eps));
  }
}

}  // namespace bundle_test

TEST_CASE("bundle_optimizer: abs_sum, a maximum of affine functions, refused and inadmissible starts") {
  bundle_test::synthetic_cases();
}
TEST_CASE("bundle_optimizer: LAD against the exact vertices of quantreg::rq") { bundle_test::lad_cases(); }
TEST_CASE("bundle_optimizer: the Laplace with covariates, beta by bundle and sigma by Newton in alternation") {
  bundle_test::laplace_alternation_cases();
}
TEST_CASE("bundle_optimizer: Newton with X'X and BFGS on LAD (a measurement)") { bundle_test::smooth_methods_on_lad(); }
