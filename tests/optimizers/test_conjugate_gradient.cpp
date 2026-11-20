// Nonlinear conjugate gradient (include/laplace/optimizers/conjugate_gradient.hpp)
// on the battery of test_problems.hpp and extra_problems.hpp, against the
// frozen runs of optimizers7 0.22.0's cg()
// (tests/data/reference/optimizers/optimizers7_cg_counts.csv), and linear
// CG's finite termination under exact line searches. Host code only.
//
// Bounds: the default stopping rule is the decrement at 1e-6, so f - f* is
// checked against 1e-5 and x - x* against the distances measured with a
// margin, never against anything tighter than what the rule guarantees.

#include <doctest/doctest.h>

#include <laplace/optimizers/conjugate_gradient.hpp>
#include <laplace/optimizers/minimize.hpp>

#include "extra_problems.hpp"
#include "test_problems.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace lo = laplace::optimizers;
namespace tp = optimizer_test_problems;
using cg_beta = lo::conjugate_gradient_beta;

constexpr cg_beta all_betas[] = {cg_beta::fletcher_reeves, cg_beta::polak_ribiere_plus, cg_beta::hestenes_stiefel,
                                 cg_beta::dai_yuan, cg_beta::hager_zhang};

lo::optimizer_vector cg_start_of(const std::vector<double>& v) {
  lo::optimizer_vector x("x0", v.size());
  for (std::size_t i = 0; i < v.size(); ++i) x(i) = v[i];
  return x;
}

double cg_distance(const lo::optimizer_report& r, const std::vector<double>& x_star) {
  double d = 0.0;
  for (std::size_t i = 0; i < x_star.size(); ++i) d = std::max(d, std::abs(r.point(i) - x_star[i]));
  return d;
}

lo::conjugate_gradient_optimizer<> cg_with(cg_beta beta) {
  lo::conjugate_gradient_optimizer<> o;
  o.beta = beta;
  return o;
}

static_assert(lo::optimizer_for<lo::conjugate_gradient_optimizer<>, tp::wood_problem>);
static_assert(lo::optimizer_for<lo::conjugate_gradient_optimizer<lo::gradient_stop_rule, lo::armijo_line_search>,
                                tp::beale_problem>);

// a minimum reached: succeeded, f - f* <= 1e-5, max |x - x*| <= dx
template <class Problem>
lo::optimizer_report cg_check_minimum(const lo::conjugate_gradient_optimizer<>& optimizer, Problem p, double dx) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, cg_start_of(p.x0));
  INFO(p.name, ", offset ", p.offset, ", beta ", int(optimizer.beta), ": status ", int(r.status), ", iterations ",
       r.iterations, ", f - f* ", r.value - p.f_star, ", decrement ", r.decrement, ", distance ",
       cg_distance(r, p.x_star));
  CHECK(lo::optimizer_succeeded(r.status));
  CHECK(r.value - p.f_star <= 1e-5);
  CHECK(cg_distance(r, p.x_star) <= dx);
  CHECK(r.decrement <= 1e-6);
  CHECK(r.value_evaluations >= r.iterations + 1);
  CHECK(r.curvature_evaluations == 0);
  return r;
}

// a run that may not converge, but must not claim a false minimum
template <class Problem>
void cg_check_no_false_success(const lo::conjugate_gradient_optimizer<>& optimizer, Problem p) {
  const lo::optimizer_report r = lo::minimize(optimizer, p, cg_start_of(p.x0));
  INFO(p.name, ", beta ", int(optimizer.beta), ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
  if (lo::optimizer_succeeded(r.status)) CHECK(r.value - p.f_star <= 1e-5);
}

// Exact line search on a quadratic: one gradient at x + d gives d'Ad =
// (g(x + d) - g(x))'d, then the minimizer t = -g'd / d'Ad along d.
struct exact_quadratic_line_search {
  template <lo::gradient_objective O>
  lo::line_search_result search(O& objective, const lo::optimizer_point& x, double, const lo::optimizer_point& g,
                                const lo::optimizer_point& d, double, const lo::optimizer_vector& x_trial,
                                const lo::optimizer_vector& g_trial) const {
    lo::line_search_result r;
    const int n = static_cast<int>(x.extent(0));
    double gd = 0.0, dad = 0.0;
    for (int i = 0; i < n; ++i) gd += g(i) * d(i), x_trial(i) = x(i) + d(i);
    if (!(gd < 0.0)) {
      r.outcome = lo::line_search_outcome::not_descent_direction;
      return r;
    }
    objective.value_gradient(lo::optimizer_point(x_trial), g_trial);
    for (int i = 0; i < n; ++i) dad += (g_trial(i) - g(i)) * d(i);
    const double t = -gd / dad;
    bool moved = false;
    for (int i = 0; i < n; ++i) x_trial(i) = x(i) + t * d(i), moved = moved || x_trial(i) != x(i);
    if (!moved || !(dad > 0.0)) {
      r.outcome = lo::line_search_outcome::step_below_rounding;
      return r;
    }
    r.value = objective.value_gradient(lo::optimizer_point(x_trial), g_trial);
    r.value_evaluations = r.gradient_evaluations = 2;
    r.step = t;
    r.outcome = lo::line_search_outcome::accepted;
    return r;
  }
};

// one row of optimizers7_cg_counts.csv
struct optimizers7_cg_row {
  std::string problem, method;
  double offset = 0.0;
  long f_count = 0;
  double value_minus_f_star = 0.0, distance_max = 0.0;
  bool converged = false;
};

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  bool quoted = false;
  for (char c : line) {
    if (c == '"') quoted = !quoted;
    else if (c == ',' && !quoted) out.push_back(cell), cell.clear();
    else cell += c;
  }
  out.push_back(cell);
  return out;
}

std::vector<optimizers7_cg_row> read_optimizers7_cg() {
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/optimizers/optimizers7_cg_counts.csv");
  std::vector<optimizers7_cg_row> rows;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (header) {
      header = false;
      continue;
    }
    const std::vector<std::string> c = split_csv(line);
    if (c.size() < 15 || c[5] == "NA") continue;
    optimizers7_cg_row r;
    r.problem = c[0], r.offset = std::stod(c[2]), r.method = c[3], r.f_count = std::stol(c[5]);
    r.value_minus_f_star = std::stod(c[9]), r.distance_max = std::stod(c[11]), r.converged = c[12] == "TRUE";
    rows.push_back(r);
  }
  return rows;
}

// the battery's problem named as in the CSV, run with `beta`
lo::optimizer_report run_named(const std::string& name, double offset, cg_beta beta, double& f_star,
                               std::vector<double>& x_star) {
  auto go = [&](auto p) {
    f_star = p.f_star, x_star = p.x_star;
    return lo::minimize(cg_with(beta), p, cg_start_of(p.x0));
  };
  if (name == "extended Rosenbrock n = 2") return go(tp::extended_rosenbrock_problem(2, offset));
  if (name == "extended Rosenbrock n = 10") return go(tp::extended_rosenbrock_problem(10, offset));
  if (name == "extended Rosenbrock n = 100") return go(tp::extended_rosenbrock_problem(100, offset));
  if (name == "Powell singular") return go(tp::powell_singular_problem(offset));
  if (name == "Beale") return go(tp::beale_problem(offset));
  if (name == "Brown badly scaled") return go(tp::brown_badly_scaled_problem(offset));
  if (name == "helical valley") return go(tp::helical_valley_problem(offset));
  if (name == "Wood") return go(tp::wood_problem(offset));
  if (name == "quadratic n = 10, kappa = 100") return go(tp::ill_conditioned_quadratic_problem(10, 1e2));
  if (name == "quadratic n = 10, kappa = 1e+06") return go(tp::ill_conditioned_quadratic_problem(10, 1e6));
  f_star = std::numeric_limits<double>::quiet_NaN();
  return {};
}

// f = x - log x for x > 0, NaN for x < 0 (refused trial points), minimum at 1
struct cg_x_minus_log_x {
  double value(const lo::optimizer_point& x) { return x(0) > 0.0 ? x(0) - std::log(x(0)) : std::nan(""); }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    g(0) = 1.0 - 1.0 / x(0);
    return value(x);
  }
};

}  // namespace

TEST_CASE("optimizers: conjugate gradient on the More-Garbow-Hillstrom problems, f* = 0, 1, 1e3, 1e6") {
  for (cg_beta beta : all_betas) {
    const auto o = cg_with(beta);
    for (double offset : {0.0, 1.0, 1e3, 1e6}) {
      cg_check_minimum(o, tp::extended_rosenbrock_problem(2, offset), 2e-3);
      cg_check_minimum(o, tp::extended_rosenbrock_problem(100, offset), 2e-3);
      cg_check_minimum(o, tp::powell_singular_problem(offset), 5e-2);  // |x - x*| ~ (f - f*)^(1/4)
      cg_check_minimum(o, tp::beale_problem(offset), 2e-3);
      if (offset == 0.0) {
        cg_check_minimum(o, tp::brown_badly_scaled_problem(offset), 2e-3);
      } else {
        // kappa about 1e12: at x* the attainable gradient is about
        // sqrt(2 lambda_max eps |f|), so the measure, whose floor carries
        // 1/lambda_min, stays near kappa eps |f| > 1e-6. Measured: the runs end
        // stalled or line_search_failed within 2e-5 of x* (FR at f* = 1, DY at 1,
        // HZ at 1 and 1e3 converge).
        tp::brown_badly_scaled_problem p(offset);
        const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
        INFO("Brown, offset ", offset, ", beta ", int(beta), ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
        CHECK(r.value - p.f_star <= 1e-5);
        CHECK(cg_distance(r, p.x_star) <= 2e-3);
        CHECK((lo::optimizer_succeeded(r.status) || r.status == lo::optimizer_status::stalled ||
               r.status == lo::optimizer_status::line_search_failed));
      }
      cg_check_minimum(o, tp::helical_valley_problem(offset), 2e-3);
      cg_check_minimum(o, tp::wood_problem(offset), 2e-3);
    }
  }
}

TEST_CASE("optimizers: conjugate gradient on ill-conditioned quadratics") {
  for (cg_beta beta : all_betas) {
    auto o = cg_with(beta);
    o.max_iterations = 30000;  // n = 100, kappa 1e6: 7705 to 15265 iterations measured
    for (double kappa : {1e2, 1e4, 1e6, 1e8}) cg_check_minimum(o, tp::ill_conditioned_quadratic_problem(10, kappa), 5e-3);
    cg_check_minimum(o, tp::ill_conditioned_quadratic_problem(100, 1e6), 5e-3);
    // kappa 1e10 and 1e12: no minimum claimed (measured: stalled or line search failed)
    for (double kappa : {1e10, 1e12}) cg_check_no_false_success(o, tp::ill_conditioned_quadratic_problem(10, kappa));
  }
}

TEST_CASE("optimizers: conjugate gradient with exact line searches ends in at most n steps on a quadratic") {
  // Every beta reduces to linear CG when the searches are exact (g_k'p_{k-1} = 0)
  // and no restart interrupts the recursion: the values along the run are
  // those of textbook linear CG (Hestenes and Stiefel's r'r ratio, written out
  // here), and f - f* is 0 after n steps up to rounding, and not before. In
  // floating point the n-step bound holds only for a mild spectrum: at n = 10 and
  // kappa 1e3 both routes stall near f - f* = 1.6e-3 at steps 9 and 10 and end
  // two steps later.
  struct quadratic_case {
    int n;
    double kappa;
  };
  for (cg_beta beta : all_betas) {
    lo::conjugate_gradient_optimizer<lo::decrement_stop_rule, exact_quadratic_line_search> o;
    o.beta = beta;
    o.powell_restart = std::numeric_limits<double>::infinity();
    o.restart_interval = 1000;
    for (quadratic_case c : {quadratic_case{5, 1e3}, quadratic_case{10, 10.0}}) {
      tp::ill_conditioned_quadratic_problem p(c.n, c.kappa);
      // textbook linear CG from the same start, A d = g(x + d) - g(x)
      std::vector<double> linear(c.n + 1);
      {
        const int n = c.n;
        lo::optimizer_vector x = cg_start_of(p.x0), g("g", n), q("q", n), gq("gq", n);
        std::vector<double> d(n);
        p.value_gradient(x, g);
        double rr = 0.0;
        for (int i = 0; i < n; ++i) d[i] = -g(i), rr += g(i) * g(i);
        for (int k = 1; k <= n; ++k) {
          for (int i = 0; i < n; ++i) q(i) = x(i) + d[i];
          p.value_gradient(q, gq);
          double dad = 0.0;
          for (int i = 0; i < n; ++i) dad += (gq(i) - g(i)) * d[i];
          for (int i = 0; i < n; ++i) x(i) += rr / dad * d[i];
          linear[k] = p.value_gradient(x, g) - p.f_star;
          double rn = 0.0;
          for (int i = 0; i < n; ++i) rn += g(i) * g(i);
          for (int i = 0; i < n; ++i) d[i] = -g(i) + rn / rr * d[i];
          rr = rn;
        }
      }
      std::vector<double> nonlinear(c.n + 1, -1.0);
      o.trace = [&](const lo::optimizer_trace_point& t) {
        if (t.iteration <= c.n) nonlinear[t.iteration] = t.value - p.f_star;
      };
      const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
      const double f0 = p.value(cg_start_of(p.x0)) - p.f_star;
      INFO("beta ", int(beta), ", n = ", c.n, ", kappa ", c.kappa, ": status ", int(r.status), ", f - f* at n - 1 ",
           nonlinear[c.n - 1], ", at n ", nonlinear[c.n], ", at the start ", f0);
      CHECK(lo::optimizer_succeeded(r.status));
      for (int k = 1; k < c.n; ++k) CHECK(std::abs(nonlinear[k] - linear[k]) <= 1e-9 * linear[k]);
      CHECK(nonlinear[c.n] <= 1e-13 * f0);
      CHECK(nonlinear[c.n - 1] > 1e-11 * f0);  // measured 1.5e-9 f0 (kappa 10) and above
    }
  }
}

TEST_CASE("optimizers: conjugate gradient against optimizers7 0.22.0's cg()") {
  // optimizers7 has FR, PR+, HS and DY, no Powell restart and step 1 first;
  // where it converged, the same beta here reaches the minimum too.
  const std::vector<optimizers7_cg_row> rows = read_optimizers7_cg();
  REQUIRE(rows.size() == 212);
  int compared = 0;
  for (const optimizers7_cg_row& row : rows) {
    if (!row.converged || row.value_minus_f_star > 1e-5) continue;
    const cg_beta beta = row.method == "cg_fr"   ? cg_beta::fletcher_reeves
                         : row.method == "cg_pr" ? cg_beta::polak_ribiere_plus
                         : row.method == "cg_hs" ? cg_beta::hestenes_stiefel
                                                 : cg_beta::dai_yuan;
    double f_star;
    std::vector<double> x_star;
    const lo::optimizer_report r = run_named(row.problem, row.offset, beta, f_star, x_star);
    if (std::isnan(f_star)) continue;
    ++compared;
    INFO(row.problem, ", offset ", row.offset, ", ", row.method, ": status ", int(r.status), ", f - f* ",
         r.value - f_star, ", evaluations ", r.value_evaluations, " against ", row.f_count);
    CHECK(lo::optimizer_succeeded(r.status));
    CHECK(r.value - f_star <= 1e-5);
    CHECK(cg_distance(r, x_star) <= std::max(5e-2, 2.0 * row.distance_max));
  }
  CHECK(compared >= 80);
}

TEST_CASE("optimizers: conjugate gradient must not converge on the negative cases") {
  for (cg_beta beta : all_betas) {
    auto o = cg_with(beta);
    {
      tp::noisy_problem p(4);
      const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
      CHECK_FALSE(lo::optimizer_succeeded(r.status));
    }
    for (int n : {2, 4, 10}) {
      tp::abs_sum_problem p(n);
      const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
      INFO("abs_sum n = ", n, ", beta ", int(beta), ": status ", int(r.status), ", f ", r.value);
      CHECK_FALSE(lo::optimizer_succeeded(r.status));
    }
    {
      tp::exploded_cosh_problem p(4);
      const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
      INFO("exploded cosh, beta ", int(beta), ": status ", int(r.status), ", iterations ", r.iterations);
      CHECK_FALSE(lo::optimizer_succeeded(r.status));
    }
    o.max_iterations = 5;
    tp::extended_rosenbrock_problem p(100);
    const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::iteration_budget);
    CHECK(r.iterations == 5);
  }
}

TEST_CASE("optimizers: conjugate gradient near a saddle, start, refused points, budgets, trace") {
  SUBCASE("a start off the saddle's stable line reaches a minimum") {
    // from (1, 0) itself g_2 = 0 on the whole line x_2 = 0, so any gradient
    // method lands on the saddle (0, 0), a stationary point; off it the run must
    // leave along x_2
    for (cg_beta beta : all_betas) {
      tp::saddle_problem p;
      const lo::optimizer_report r = lo::minimize(cg_with(beta), p, cg_start_of({1.0, 1e-3}));
      INFO("beta ", int(beta), ": status ", int(r.status), ", f - f* ", r.value - p.f_star);
      CHECK(lo::optimizer_succeeded(r.status));
      CHECK(r.value - p.f_star <= 1e-5);
    }
  }
  SUBCASE("start at the minimum: converged at iteration 0") {
    tp::ill_conditioned_quadratic_problem p(5, 1e3);
    const lo::optimizer_report r = lo::minimize(lo::conjugate_gradient_optimizer<>{}, p, cg_start_of(p.x_star));
    CHECK(r.status == lo::optimizer_status::converged);
    CHECK(r.iterations == 0);
    CHECK(r.decrement == 0.0);
  }
  SUBCASE("start not admissible") {
    cg_x_minus_log_x f;
    const lo::optimizer_report r = lo::minimize(lo::conjugate_gradient_optimizer<>{}, f, cg_start_of({-1.0}));
    CHECK(r.status == lo::optimizer_status::start_not_admissible);
  }
  SUBCASE("refused trial points across the border of the domain") {
    cg_x_minus_log_x f;
    for (double x0 : {0.01, 30.0}) {
      const lo::optimizer_report r = lo::minimize(lo::conjugate_gradient_optimizer<>{}, f, cg_start_of({x0}));
      CHECK(lo::optimizer_succeeded(r.status));
      CHECK(std::abs(r.point(0) - 1.0) < 2e-3);
    }
  }
  SUBCASE("evaluation budget") {
    tp::extended_rosenbrock_problem p(2);
    lo::conjugate_gradient_optimizer<> o;
    o.max_evaluations = 10;
    const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::evaluation_budget);
    CHECK(r.value_evaluations >= 10);
  }
  SUBCASE("trace: one call per accepted iteration") {
    tp::beale_problem p;
    lo::conjugate_gradient_optimizer<> o;
    int calls = 0, last = 0;
    o.trace = [&](const lo::optimizer_trace_point& t) { ++calls, last = t.iteration; };
    const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
    CHECK(calls == r.iterations);
    CHECK(last == r.iterations);
  }
  SUBCASE("the default line search is strong Wolfe with c2 = 0.1") {
    CHECK(lo::conjugate_gradient_optimizer<>{}.line_search.curvature == 0.1);
  }
  SUBCASE("a change rule stops on change, never converged") {
    tp::extended_rosenbrock_problem p(2);
    lo::conjugate_gradient_optimizer<lo::objective_change_stop_rule> o{{1e-2}};
    const lo::optimizer_report r = lo::minimize(o, p, cg_start_of(p.x0));
    CHECK(r.status == lo::optimizer_status::stopped_on_change);
  }
}
