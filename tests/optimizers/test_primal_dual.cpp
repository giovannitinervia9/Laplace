// primal_dual_optimizer (Condat-Vu) on s(x) + g(D x) through
// linear_map_penalized_objective (note/optimizers.md, section 12), with s the
// regression test objective (regression_objective.hpp) or a Laplace
// log-scale likelihood, both summed in Kokkos kernels on the CPU or GPU build.
//
// - fused lasso (difference_map<1>) and adaptive lasso (diagonal_map): the
//   same point and objective as FISTA with the exact prox in beta (Condat's
//   TV prox, the weighted soft threshold);
// - trend filtering (difference_map<2>, <3>) and the elastic net on
//   differences, which have no prox in beta: the optimality conditions, read
//   through a dual vector z solving D' z = -grad s(x) in least squares
//   (z in dg(D x) coordinate by coordinate, D' z = -grad s(x));
// - statuses: an unbounded objective never converges, a small budget ends as
//   iteration_budget, a non-finite start is refused; gradient_stop_rule is
//   rejected at compile time;
// - the descent-lemma test at the rounding of s: the same fused regression
//   converges with its sum taken sequentially and on the default space;
// - the 18 genlasso fits frozen in tests/data/reference/optimizers/wave2
//   (fused lasso, trend filtering of order 2, elastic net on first
//   differences): coefficients and objective against the polished reference.

#include "regression_objective.hpp"
#include "wave2_reference.hpp"

#include <doctest/doctest.h>

#include <laplace/distrib/enet.hpp>
#include <laplace/distrib/independent_product.hpp>
#include <laplace/distrib/laplace2.hpp>
#include <laplace/distrib/univariate_as_multivariate.hpp>
#include <laplace/optimizers/penalized_objective.hpp>
#include <laplace/optimizers/primal_dual.hpp>
#include <laplace/optimizers/proximal_gradient.hpp>
#include <laplace/penalty/linear_map_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <limits>
#include <vector>

namespace primal_dual_test {

namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
namespace lpen = laplace::penalty;
using regression_test::make_regression_problem;
using regression_test::regression_loss;
using regression_test::regression_problem;
using objective_type = regression_test::regression_test_objective<>;

template <class D>
using uam = ld::univariate_as_multivariate<D>;
template <class D>
using product_penalty = lpen::distrib_penalty<ld::independent_product_distrib<uam<D>>>;
using lasso_penalty = product_penalty<ld::laplace2_distrib>;
using enet_penalty = product_penalty<ld::enet_distrib>;
template <class P, int K>
using difference_penalty = lpen::linear_map_penalty<P, lpen::difference_map<K>>;
template <class P>
using weighted_penalty = lpen::linear_map_penalty<P, lpen::diagonal_map>;

static_assert(!lo::stop_rule_for<lo::gradient_stop_rule, lo::detail::primal_dual_provides>);
static_assert(!lo::stop_rule_for<lo::prox_mapping_stop_rule, lo::detail::primal_dual_provides>);
static_assert(lo::stop_rule_for<lo::primal_dual_stop_rule, lo::detail::primal_dual_provides>);

template <class D>
product_penalty<D> make_product(int d) {
  return product_penalty<D>(ld::independent_product_distrib<uam<D>>(ld::independent_product_component<uam<D>>{{}, d, 1, nullptr}));
}

// ---- s(x) = sum_i x_i + a_i exp(-x_i): -log of Laplace(0, exp(x_i)) at y_i = +-a_i, up to n log 2

struct laplace_scale_value {
  Kokkos::View<const double*> a, x;
  KOKKOS_INLINE_FUNCTION void operator()(int i, double& s) const { s += x(i) + a(i) * Kokkos::exp(-x(i)); }
};
struct laplace_scale_gradient {
  Kokkos::View<const double*> a, x;
  Kokkos::View<double*> g;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { g(i) = 1.0 - a(i) * Kokkos::exp(-x(i)); }
};

class laplace_scale_objective {
 public:
  explicit laplace_scale_objective(const std::vector<double>& a_host)
      : n_(static_cast<int>(a_host.size())), a_("a", n_), x_("x", n_), g_("g", n_), x_host_("x_host", n_) {
    auto ah = Kokkos::create_mirror_view(a_);
    for (int i = 0; i < n_; ++i) ah(i) = a_host[std::size_t(i)];
    Kokkos::deep_copy(a_, ah);
  }
  double value(const lo::optimizer_point& x) {
    load(x);
    double s = 0.0;
    Kokkos::parallel_reduce("laplace_scale_value", n_, laplace_scale_value{a_, x_}, s);
    return s;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    const double s = value(x);
    Kokkos::parallel_for("laplace_scale_gradient", n_, laplace_scale_gradient{a_, x_, g_});
    auto gh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g_);
    for (int i = 0; i < n_; ++i) g(i) = gh(i);
    return s;
  }

 private:
  void load(const lo::optimizer_point& x) {
    for (int i = 0; i < n_; ++i) x_host_(i) = x(i);
    Kokkos::deep_copy(x_, x_host_);
  }
  int n_;
  Kokkos::View<double*> a_, x_, g_;
  Kokkos::View<double*, Kokkos::HostSpace> x_host_;
};

// ---- s(x) = c'x, unbounded below along the constants when sum c != 0

struct linear_objective {
  std::vector<double> c;
  double value(const lo::optimizer_point& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) s += c[i] * x(i);
    return s;
  }
  double value_gradient(const lo::optimizer_point& x, const lo::optimizer_vector& g) {
    for (std::size_t i = 0; i < c.size(); ++i) g(i) = c[i];
    return value(x);
  }
};

// ---- helpers

std::vector<double> identity_design(int n) {
  std::vector<double> x(std::size_t(n) * std::size_t(n), 0.0);
  for (int i = 0; i < n; ++i) x[std::size_t(i) * std::size_t(n + 1)] = 1.0;
  return x;
}

// a noisy piecewise-constant (K = 1) or smooth (K >= 2) signal
std::vector<double> signal(int n, int K, std::uint64_t seed, double noise) {
  regression_test::regression_rng rng{seed};
  std::vector<double> y(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double t = double(i) / (n - 1);
    const double truth = K == 1 ? (t < 0.3 ? 0.0 : t < 0.6 ? 2.0 : -1.0) : 3.0 * std::sin(6.0 * t);
    y[std::size_t(i)] = truth + noise * rng.normal();
  }
  return y;
}

std::vector<double> to_vector(const lo::optimizer_point& x) {
  std::vector<double> v(x.extent(0));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = x(i);
  return v;
}

double max_abs_difference(const std::vector<double>& a, const lo::optimizer_point& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b(i)));
  return d;
}

// The optimality conditions of s(x) + sum_i pen(u_i), u = D x_[o, o + p), D
// the differences of order K, pen(u) = a|u| + c u^2 / 2: z = (D D')^-1 D r
// with r = -grad s(x) on the range; returns max |D'z - r| over the whole x
// (0 on the unpenalized coordinates too), max_i dist(z_i - c u_i, a d|u_i|)
// with the sign required where |u_i| > kink_scale, and the count of such i.
struct kkt_result {
  double stationarity = 0.0, subgradient = 0.0;
  int nonzero = 0;
};

template <class O>
kkt_result difference_kkt(O& smooth, const lo::optimizer_point& x, int offset, int p, int K, double a, double c,
                          double kink_scale) {
  const int n = static_cast<int>(x.extent(0));
  const int m = p - K;
  lo::optimizer_vector g("g", n);
  smooth.value_gradient(x, g);
  std::vector<double> D(std::size_t(m) * std::size_t(p), 0.0);  // row-major, row i = (-1)^(K - j) binom(K, j) at i + j
  for (int i = 0; i < m; ++i) {
    double binom = 1.0;
    for (int j = 0; j <= K; ++j) {
      D[std::size_t(i) * std::size_t(p) + std::size_t(i + j)] = ((K - j) % 2 == 0 ? 1.0 : -1.0) * binom;
      binom = binom * (K - j) / (j + 1);
    }
  }
  std::vector<double> r(static_cast<std::size_t>(p)), b(static_cast<std::size_t>(m), 0.0), A(static_cast<std::size_t>(m * m), 0.0);
  for (int j = 0; j < p; ++j) r[std::size_t(j)] = -g(offset + j);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < p; ++j) b[std::size_t(i)] += D[std::size_t(i) * std::size_t(p) + std::size_t(j)] * r[std::size_t(j)];
  for (int i = 0; i < m; ++i)
    for (int l = 0; l <= i; ++l) {
      double s = 0.0;
      for (int j = 0; j < p; ++j)
        s += D[std::size_t(i) * std::size_t(p) + std::size_t(j)] * D[std::size_t(l) * std::size_t(p) + std::size_t(j)];
      A[std::size_t(i) * std::size_t(m) + std::size_t(l)] = s;
    }
  for (int j = 0; j < m; ++j) {  // Cholesky, lower triangle in place
    double d = A[std::size_t(j) * std::size_t(m) + std::size_t(j)];
    for (int k = 0; k < j; ++k) d -= A[std::size_t(j) * std::size_t(m) + std::size_t(k)] * A[std::size_t(j) * std::size_t(m) + std::size_t(k)];
    d = std::sqrt(d);
    A[std::size_t(j) * std::size_t(m) + std::size_t(j)] = d;
    for (int i = j + 1; i < m; ++i) {
      double s = A[std::size_t(i) * std::size_t(m) + std::size_t(j)];
      for (int k = 0; k < j; ++k) s -= A[std::size_t(i) * std::size_t(m) + std::size_t(k)] * A[std::size_t(j) * std::size_t(m) + std::size_t(k)];
      A[std::size_t(i) * std::size_t(m) + std::size_t(j)] = s / d;
    }
  }
  std::vector<double> z = b;
  for (int i = 0; i < m; ++i) {
    for (int k = 0; k < i; ++k) z[std::size_t(i)] -= A[std::size_t(i) * std::size_t(m) + std::size_t(k)] * z[std::size_t(k)];
    z[std::size_t(i)] /= A[std::size_t(i) * std::size_t(m) + std::size_t(i)];
  }
  for (int i = m - 1; i >= 0; --i) {
    for (int k = i + 1; k < m; ++k) z[std::size_t(i)] -= A[std::size_t(k) * std::size_t(m) + std::size_t(i)] * z[std::size_t(k)];
    z[std::size_t(i)] /= A[std::size_t(i) * std::size_t(m) + std::size_t(i)];
  }
  kkt_result out;
  for (int i = 0; i < n; ++i) {
    double dz = 0.0;
    if (i >= offset && i < offset + p) {
      const int j = i - offset;
      for (int l = 0; l < m; ++l) dz += D[std::size_t(l) * std::size_t(p) + std::size_t(j)] * z[std::size_t(l)];
      dz -= r[std::size_t(j)];
    } else {
      dz = g(i);
    }
    out.stationarity = std::max(out.stationarity, std::abs(dz));
  }
  for (int i = 0; i < m; ++i) {
    double u = 0.0;
    for (int j = 0; j < p; ++j) u += D[std::size_t(i) * std::size_t(p) + std::size_t(j)] * x(offset + j);
    const double v = z[std::size_t(i)] - c * u;
    double e;
    if (std::abs(u) > kink_scale) {
      e = std::abs(v - a * (u > 0.0 ? 1.0 : -1.0));
      ++out.nonzero;
    } else {
      e = std::max(0.0, std::abs(v) - a);
    }
    out.subgradient = std::max(out.subgradient, e);
  }
  return out;
}

// ---- cases (named functions; no Kokkos lambdas in a TEST_CASE)

template <class P>
void compare_with_fista(objective_type& smooth, int n, const lpen::linear_map_penalty<P, lpen::difference_map<1>>& pen,
                        int offset, double lambda, double point_bound, const char* label) {
  using pen_type = lpen::linear_map_penalty<P, lpen::difference_map<1>>;
  lo::linear_map_penalized_objective<objective_type, P, lpen::difference_map<1>> split(smooth, n,
                                                                                         lo::attach_penalty(pen, offset, {lambda}));
  lo::penalized_objective<objective_type, pen_type> composite(smooth, n, lo::attach_penalty(pen, offset, {lambda}));
  lo::optimizer_vector x0("x0", n);
  lo::primal_dual_optimizer<> pd;
  pd.stop_rule.tolerance = 1e-9;
  lo::proximal_gradient_optimizer<> fista;
  fista.stop_rule.tolerance = 1e-12;
  const auto a = lo::minimize(pd, split, x0);
  const auto b = lo::minimize(fista, composite, x0);
  const double fb = b.value;
  const double dx = max_abs_difference(to_vector(a.point), b.point);
  std::printf("primal_dual %s: status %d it %d evals %ld V %.2e | f - f_fista %.2e | max|dx| %.2e | fista it %d\n", label,
              int(a.status), a.iterations, a.value_evaluations, a.decrement, a.value - fb, dx, b.iterations);
  CHECK(a.status == lo::optimizer_status::converged);
  CHECK(lo::optimizer_succeeded(b.status));
  CHECK(std::abs(a.value - fb) <= 2e-9);
  CHECK(a.value - fb >= -1e-10 * std::max(1.0, std::abs(fb)));  // FISTA's point is the more exact one
  CHECK(dx <= point_bound);
}

void fused_lasso_cases() {
  // identity design: s = |y - x|^2 / 2, the exact solution is Condat's prox at y
  {
    const int n = 200;
    const auto y = signal(n, 1, 11u, 1.0);
    objective_type smooth(identity_design(n), y, n, n, regression_loss::gaussian);
    const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(n - 1), lpen::difference_map<1>(n));
    compare_with_fista(smooth, n, pen, 0, 2.0, 1e-4, "fused identity lambda 2");
    compare_with_fista(smooth, n, pen, 0, 20.0, 1e-4, "fused identity lambda 20");
  }
  // regression, intercept unpenalized, two correlations
  for (double rho : {0.0, 0.9}) {
    const regression_problem pr = make_regression_problem(300, 41, 10, 1.0, 21u, false, rho);
    objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
    const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(pr.p - 2),
                                                   lpen::difference_map<1>(pr.p - 1));
    compare_with_fista(smooth, pr.p, pen, 1, 30.0, 1e-3, rho == 0.0 ? "fused regression rho 0" : "fused regression rho 0.9");
  }
}

void adaptive_lasso_case() {
  const regression_problem pr = make_regression_problem(300, 41, 10, 1.0, 31u, false, 0.5);
  objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
  const int d = pr.p - 1;
  std::vector<double> w(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) w[std::size_t(j)] = 0.3 + 0.1 * (j % 7);
  using pen_type = weighted_penalty<lasso_penalty>;
  const pen_type pen(make_product<ld::laplace2_distrib>(d), lpen::diagonal_map(w.data(), d));
  lo::linear_map_penalized_objective<objective_type, lasso_penalty, lpen::diagonal_map> split(smooth, pr.p,
                                                                                             lo::attach_penalty(pen, 1, {40.0}));
  lo::penalized_objective<objective_type, pen_type> composite(smooth, pr.p, lo::attach_penalty(pen, 1, {40.0}));
  lo::optimizer_vector x0("x0", pr.p);
  lo::primal_dual_optimizer<> pd;
  pd.stop_rule.tolerance = 1e-9;
  lo::proximal_gradient_optimizer<> fista;
  fista.stop_rule.tolerance = 1e-12;
  const auto a = lo::minimize(pd, split, x0);
  const auto b = lo::minimize(fista, composite, x0);
  const double fb = b.value;
  const double dx = max_abs_difference(to_vector(a.point), b.point);
  std::printf("primal_dual adaptive lasso: status %d it %d V %.2e | f - f_fista %.2e | max|dx| %.2e\n", int(a.status),
              a.iterations, a.decrement, a.value - fb, dx);
  CHECK(a.status == lo::optimizer_status::converged);
  CHECK(lo::optimizer_succeeded(b.status));
  CHECK(std::abs(a.value - fb) <= 2e-9);
  CHECK(dx <= 1e-3);
}

template <int K, class P, class O>
void trend_filter_case(O& smooth, int n, int offset, const P& inner, std::vector<double> theta, double a, double c,
                       const char* label, double tolerance, double stationarity_bound, double subgradient_bound) {
  const int p = n - offset;
  const difference_penalty<P, K> pen(inner, lpen::difference_map<K>(p));
  lo::linear_map_penalized_objective<O, P, lpen::difference_map<K>> split(smooth, n, lo::attach_penalty(pen, offset, std::move(theta)));
  lo::optimizer_vector x0("x0", n);
  lo::primal_dual_optimizer<> pd;
  pd.stop_rule.tolerance = tolerance;
  const auto rep = lo::minimize(pd, split, x0);
  const kkt_result k = difference_kkt(smooth, rep.point, offset, p, K, a, c, 1e-4);
  std::printf("primal_dual %s: status %d it %d evals %ld V %.2e | KKT stationarity %.2e subgradient %.2e (%d kinks left)\n",
              label, int(rep.status), rep.iterations, rep.value_evaluations, rep.decrement, k.stationarity, k.subgradient,
              p - K - k.nonzero);
  CHECK(rep.status == lo::optimizer_status::converged);
  CHECK(k.stationarity <= stationarity_bound);
  CHECK(k.subgradient <= subgradient_bound);
}

void trend_filtering_cases() {
  {
    const int n = 100;
    const auto y = signal(n, 2, 41u, 1.0);
    objective_type smooth(identity_design(n), y, n, n, regression_loss::gaussian);
    trend_filter_case<2>(smooth, n, 0, make_product<ld::laplace2_distrib>(n - 2), {5.0}, 5.0, 0.0, "trend k2 identity",
                         1e-8, 1e-6, 1e-2);
    trend_filter_case<2>(smooth, n, 0, make_product<ld::enet_distrib>(n - 2), {8.0, 0.6}, 4.8, 3.2, "enet k2 identity",
                         1e-8, 1e-6, 1e-2);
  }
  {
    const int n = 50;
    const auto y = signal(n, 3, 43u, 0.5);
    objective_type smooth(identity_design(n), y, n, n, regression_loss::gaussian);
    trend_filter_case<3>(smooth, n, 0, make_product<ld::laplace2_distrib>(n - 3), {2.0}, 2.0, 0.0, "trend k3 identity",
                         1e-8, 1e-6, 1e-2);
  }
  {
    const regression_problem pr = make_regression_problem(300, 41, 10, 1.0, 51u, false, 0.5);
    objective_type smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
    trend_filter_case<2>(smooth, pr.p, 1, make_product<ld::laplace2_distrib>(pr.p - 3), {20.0}, 20.0, 0.0,
                         "trend k2 regression", 1e-8, 1e-6, 1e-2);
  }
}

std::vector<double> laplace_scale_data(int n, int K, std::uint64_t seed) {
  regression_test::regression_rng rng{seed};
  std::vector<double> a(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double t = double(i) / (n - 1);
    const double log_scale = K == 1 ? (t < 0.5 ? 0.0 : 1.5) : 1.5 * std::sin(5.0 * t);
    a[std::size_t(i)] = std::exp(log_scale) * std::abs(std::log(rng.uniform() / rng.uniform()));
  }
  return a;
}

void laplace_scale_cases() {
  const int n = 200;
  {  // order 1 against FISTA with Condat's prox, same objective
    laplace_scale_objective smooth(laplace_scale_data(n, 1, 61u));
    using P = lasso_penalty;
    using pen_type = difference_penalty<P, 1>;
    const pen_type pen(make_product<ld::laplace2_distrib>(n - 1), lpen::difference_map<1>(n));
    lo::linear_map_penalized_objective<laplace_scale_objective, P, lpen::difference_map<1>> split(smooth, n,
                                                                                                  lo::attach_penalty(pen, 0, {5.0}));
    lo::penalized_objective<laplace_scale_objective, pen_type> composite(smooth, n, lo::attach_penalty(pen, 0, {5.0}));
    lo::optimizer_vector x0("x0", n);
    lo::primal_dual_optimizer<> pd;
    pd.stop_rule.tolerance = 1e-9;
    lo::proximal_gradient_optimizer<> fista;
    fista.stop_rule.tolerance = 1e-12;
    const auto a = lo::minimize(pd, split, x0);
    const auto b = lo::minimize(fista, composite, x0);
    const double fb = b.value;
    const double dx = max_abs_difference(to_vector(a.point), b.point);
    std::printf("primal_dual laplace scale k1: status %d it %d V %.2e | f - f_fista %.2e | max|dx| %.2e | fista it %d\n",
                int(a.status), a.iterations, a.decrement, a.value - fb, dx, b.iterations);
    CHECK(a.status == lo::optimizer_status::converged);
    CHECK(lo::optimizer_succeeded(b.status));
    CHECK(std::abs(a.value - fb) <= 2e-9);
    CHECK(dx <= 1e-3);
  }
  {  // order 2 by the optimality conditions
    laplace_scale_objective smooth(laplace_scale_data(n, 2, 63u));
    trend_filter_case<2>(smooth, n, 0, make_product<ld::laplace2_distrib>(n - 2), {5.0}, 5.0, 0.0, "laplace scale k2",
                         1e-6, 1e-6, 1e-2);
  }
}

// The descent-lemma test near the solution: the bench's fused regression at
// n = 1000, its objective summed sequentially (Kokkos::Serial) and on the
// default space. Without the gradient fallback the sequential sum's rounding
// (about +-2e-12 against the slack 1.5e-12) failed the test at iteration 4723,
// doubled L 21 times and stalled the run at V = 6e-5 until the 1e5 budget, while
// the default space converged in 6372 iterations.
template <class ExecSpace>
lo::optimizer_report rounding_fit(const regression_problem& pr) {
  regression_test::regression_test_objective<ExecSpace> smooth(pr.x, pr.y, pr.n, pr.p, regression_loss::gaussian);
  const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(pr.p - 2),
                                                 lpen::difference_map<1>(pr.p - 1));
  lo::linear_map_penalized_objective<regression_test::regression_test_objective<ExecSpace>, lasso_penalty,
                                     lpen::difference_map<1>>
      split(smooth, pr.p, lo::attach_penalty(pen, 1, {0.1 * pr.n}));
  lo::optimizer_vector x0("x0", pr.p);
  return lo::minimize(lo::primal_dual_optimizer<>{}, split, x0);
}

void rounding_case() {
  const regression_problem pr = make_regression_problem(1000, 41, 10, 1.0, 21u, false, 0.5);
  const auto a = rounding_fit<Kokkos::Serial>(pr);
  const auto b = rounding_fit<Kokkos::DefaultExecutionSpace>(pr);
  std::printf("primal_dual rounding fused n 1000: serial status %d it %d V %.2e | %s status %d it %d V %.2e\n",
              int(a.status), a.iterations, a.decrement, Kokkos::DefaultExecutionSpace::name(), int(b.status), b.iterations,
              b.decrement);
  CHECK(a.status == lo::optimizer_status::converged);
  CHECK(b.status == lo::optimizer_status::converged);
  CHECK(a.iterations >= 6000);
  CHECK(a.iterations <= 6800);
  CHECK(b.iterations >= 6000);
  CHECK(b.iterations <= 6800);
}

void status_cases() {
  const int n = 30;
  {  // unbounded below: s = c'x with sum c != 0 is free along the constants
    linear_objective smooth{std::vector<double>(std::size_t(n), 1.0)};
    smooth.c[3] = -2.0;
    const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(n - 1), lpen::difference_map<1>(n));
    lo::linear_map_penalized_objective<linear_objective, lasso_penalty, lpen::difference_map<1>> split(
        smooth, n, lo::attach_penalty(pen, 0, {1.0}));
    lo::optimizer_vector x0("x0", n);
    lo::primal_dual_optimizer<> pd;
    pd.max_iterations = 20000;
    const auto rep = lo::minimize(pd, split, x0);
    std::printf("primal_dual unbounded: status %d it %d V %.2e f %.3e\n", int(rep.status), rep.iterations, rep.decrement,
                rep.value);
    CHECK(!lo::optimizer_succeeded(rep.status));
    CHECK(rep.status == lo::optimizer_status::iteration_budget);
  }
  {  // a budget too small for the problem
    const auto y = signal(100, 2, 71u, 1.0);
    objective_type smooth(identity_design(100), y, 100, 100, regression_loss::gaussian);
    const difference_penalty<lasso_penalty, 2> pen(make_product<ld::laplace2_distrib>(98), lpen::difference_map<2>(100));
    lo::linear_map_penalized_objective<objective_type, lasso_penalty, lpen::difference_map<2>> split(
        smooth, 100, lo::attach_penalty(pen, 0, {5.0}));
    lo::optimizer_vector x0("x0", 100);
    lo::primal_dual_optimizer<> pd;
    pd.max_iterations = 10;
    const auto rep = lo::minimize(pd, split, x0);
    CHECK(rep.status == lo::optimizer_status::iteration_budget);
    CHECK(rep.iterations == 10);
    CHECK(rep.decrement > 1e-6);
  }
  {  // a non-finite start
    laplace_scale_objective smooth(laplace_scale_data(n, 1, 73u));
    const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(n - 1), lpen::difference_map<1>(n));
    lo::linear_map_penalized_objective<laplace_scale_objective, lasso_penalty, lpen::difference_map<1>> split(
        smooth, n, lo::attach_penalty(pen, 0, {1.0}));
    lo::optimizer_vector x0("x0", n);
    x0(0) = std::numeric_limits<double>::quiet_NaN();
    const auto rep = lo::minimize(lo::primal_dual_optimizer<>{}, split, x0);
    CHECK(rep.status == lo::optimizer_status::start_not_admissible);
  }
  {  // a range outside x, a wrong theta
    linear_objective smooth{std::vector<double>(std::size_t(n), 1.0)};
    const difference_penalty<lasso_penalty, 1> pen(make_product<ld::laplace2_distrib>(n - 1), lpen::difference_map<1>(n));
    using split_type = lo::linear_map_penalized_objective<linear_objective, lasso_penalty, lpen::difference_map<1>>;
    CHECK_THROWS(split_type(smooth, n, lo::attach_penalty(pen, 1, {1.0})));
    CHECK_THROWS(split_type(smooth, n, lo::attach_penalty(pen, 0, {1.0, 2.0})));
  }
}

// ---- genlasso references (tests/data/reference/optimizers/wave2, README there)

using wave2_reference::difference_dataset;
using wave2_reference::difference_fit;
using wave2_reference::load_difference_dataset;
using wave2_reference::read_difference_fits;

template <int K, class P>
lo::optimizer_report genlasso_fit(objective_type& smooth, int p, const P& inner, std::vector<double> theta, double tolerance,
                                  int max_iterations) {
  const difference_penalty<P, K> pen(inner, lpen::difference_map<K>(p));
  lo::linear_map_penalized_objective<objective_type, P, lpen::difference_map<K>> split(smooth, p,
                                                                                    lo::attach_penalty(pen, 0, std::move(theta)));
  lo::optimizer_vector x0("x0", p);
  lo::primal_dual_optimizer<> pd;
  pd.stop_rule.tolerance = tolerance;
  pd.max_iterations = max_iterations;
  return lo::minimize(pd, split, x0);
}

void genlasso_cases() {
  const std::string dir = LAPLACE_REFERENCE_DIR "/optimizers/wave2";
  const auto fits = read_difference_fits(dir);
  REQUIRE(fits.size() == 18);
  std::string loaded;
  difference_dataset d;
  for (const auto& ref : fits) {
    if (ref.dataset != loaded) d = load_difference_dataset(dir, ref.dataset), loaded = ref.dataset;
    REQUIRE(ref.coefficients.size() == std::size_t(d.p));
    const bool trend = ref.penalty == "trend";
    objective_type smooth(d.x, trend ? d.y_trend : d.y_fused, d.n, d.p, regression_loss::gaussian);
    const double f_reference =
        ref.objective_laplace + (d.p - ref.order) * ref.penalty_constant_per_row - 0.5 * ref.log_det_ddt;
    const double tolerance = std::max(1e-8, 1e3 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    // f2 (n = 1000, p = 200): trend filtering at 0.1 and 0.01 needs 164753 and
    // 830586 iterations (measured), so those runs are capped at 1e4 and only the
    // honesty of the status is checked
    const bool small = ref.dataset == "f1_n200_p50";
    // f2's trend filtering at 0.5 of lambda_max: V stops falling at 2.8e-8 to 4.1e-8,
    // its rounding floor (0.69 eps |y|_inf sqrt(B) |x|_1, lambda = 1.08e6), above the
    // tolerance 1e-8, so the run ends at the resolution of V: measured 108466
    // iterations, coef err 1.5e-12 (it reached the 1e6 budget before the floor)
    const bool trend_floor = !small && trend && ref.lambda_fraction == 0.5;
    const int budget = small ? 100000 : trend_floor ? 200000 : 10000;
    lo::optimizer_report rep;
    if (ref.penalty == "fused")
      rep = genlasso_fit<1>(smooth, d.p, make_product<ld::laplace2_distrib>(d.p - 1), {ref.lambda}, tolerance, budget);
    else if (trend)
      rep = genlasso_fit<2>(smooth, d.p, make_product<ld::laplace2_distrib>(d.p - 2), {ref.lambda}, tolerance, budget);
    else
      rep = genlasso_fit<1>(smooth, d.p, make_product<ld::enet_distrib>(d.p - 1), {ref.lambda, ref.alpha}, tolerance, budget);
    double err = 0.0, scale = 0.0;
    for (int j = 0; j < d.p; ++j) {
      err = std::max(err, std::abs(rep.point(j) - ref.coefficients[std::size_t(j)]));
      scale = std::max(scale, std::abs(ref.coefficients[std::size_t(j)]));
    }
    const double gap = rep.value - f_reference;
    std::printf("primal_dual genlasso %s %s %.2f: status %d it %d V %.2e | f - f* %.2e | coef err %.2e\n", ref.dataset.c_str(),
                ref.penalty.c_str(), ref.lambda_fraction, int(rep.status), rep.iterations, rep.decrement, gap, err / scale);
    // measured on gcc: f - f* at most 1.01 V over the 18 fits, converged or not
    CHECK(gap <= 1.5 * rep.decrement + 64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    CHECK(gap >= -64 * std::numeric_limits<double>::epsilon() * std::abs(f_reference));
    if (small) {
      CHECK(rep.status == lo::optimizer_status::converged);
      CHECK(err <= 1e-6 * scale);  // measured 3.7e-8
    } else if (trend_floor) {
      CHECK(rep.status == lo::optimizer_status::converged_at_resolution);
      CHECK(err <= 1e-10 * scale);
    } else if (ref.lambda_fraction == 0.01 && !trend) {
      // fused and fused_enet at 0.01 converge since the descent-lemma test is
      // decided on gradients near the rounding of s (were the budget before):
      // measured 2472 and 3359 iterations, coef err 6.9e-13 and 2.2e-12 (gcc),
      // 2473, 3360, 6.6e-13, 2.2e-12 (CUDA)
      CHECK(rep.status == lo::optimizer_status::converged);
      CHECK(err <= 1e-10 * scale);
    } else {
      CHECK((rep.status == lo::optimizer_status::converged || rep.status == lo::optimizer_status::iteration_budget));
      if (rep.status == lo::optimizer_status::converged) CHECK(err <= 1e-6 * scale);
    }
  }
}

}  // namespace primal_dual_test

TEST_CASE("primal_dual: fused lasso against FISTA with Condat's prox") { primal_dual_test::fused_lasso_cases(); }
TEST_CASE("primal_dual: adaptive lasso against FISTA") { primal_dual_test::adaptive_lasso_case(); }
TEST_CASE("primal_dual: trend filtering and enet on differences, optimality conditions") {
  primal_dual_test::trend_filtering_cases();
}
TEST_CASE("primal_dual: Laplace log-scale likelihood") { primal_dual_test::laplace_scale_cases(); }
TEST_CASE("primal_dual: statuses") { primal_dual_test::status_cases(); }
TEST_CASE("primal_dual: the same status with a sequential and a parallel sum") { primal_dual_test::rounding_case(); }
TEST_CASE("primal_dual: genlasso references (fused, trend filtering, enet on differences)") {
  primal_dual_test::genlasso_cases();
}
