// scad_penalty and mcp_penalty (include/laplace/penalty/nonconvex_penalty.hpp;
// note/penalties.md section 5e, tests decided in 5e.6). Evaluated inside Kokkos
// kernels (named functions with external linkage), checked on the host:
//  - the value against Fan and Li's original form of SCAD and Zhang's MCP;
//  - J = 1, 2 and the theta derivatives (K = 1) against central differences of
//    the order below, away from the knots; sign(0) = 0 at beta = 0 and the inner
//    piece at the knots by exact equality; J = 3 structurally zero; continuity of
//    the value and of J = 1 across the knots, and of d/dtheta across the moving
//    knot (C1 in theta);
//  - the prox: optimality (0 in the subdifferential), golden section on the
//    objective, t rho = 1 accepted (also as 1/rho in floating point), t rho > 1
//    NaN; the scalar and the per-coordinate overloads agree;
//  - rho as the maximum of -pen'' on a grid;
//  - through linear_map_penalty: diagonal_map against q(w_j |beta_j|) and the
//    scaled prox, difference_map<1> against q(|beta_{j+1} - beta_j|) with its
//    first derivative by central differences, and the descriptors;
//  - penalties7 (statmodels7) values, gradients, second derivatives and proxes
//    frozen by tools/reference/statmodels7_nonconvex_penalty.R.
#include <doctest/doctest.h>

#include <laplace/penalty/linear_map_penalty.hpp>
#include <laplace/penalty/nonconvex_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace nonconvex_test {

namespace lpen = laplace::penalty;
using host_view = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>;

static_assert(lpen::mcp_penalty::normalization == lpen::penalty_normalization::unnormalizable &&
              lpen::scad_penalty::kinks == lpen::penalty_kink_geometry::per_coordinate &&
              lpen::scad_penalty::prox == lpen::penalty_prox_kind::closed_form &&
              lpen::mcp_penalty::structure == lpen::penalty_structure::separable &&
              lpen::mcp_penalty::beta_regularity == 0 && lpen::scad_penalty::theta_regularity == 1 &&
              !lpen::scad_penalty::q_linear_in_theta &&
              lpen::mcp_penalty::scale_mixture == lpen::penalty_scale_mixture::none);
using diag_mcp = lpen::linear_map_penalty<lpen::mcp_penalty, lpen::diagonal_map>;
using diff_scad = lpen::linear_map_penalty<lpen::scad_penalty, lpen::difference_map<1>>;
static_assert(diag_mcp::prox == lpen::penalty_prox_kind::closed_form &&
              diag_mcp::kinks == lpen::penalty_kink_geometry::per_coordinate &&
              diag_mcp::normalization == lpen::penalty_normalization::unnormalizable);
static_assert(diff_scad::prox == lpen::penalty_prox_kind::none &&
              diff_scad::kinks == lpen::penalty_kink_geometry::linear_map &&
              diff_scad::structure == lpen::penalty_structure::coupled &&
              diff_scad::normalization == lpen::penalty_normalization::unnormalizable);

// The host oracles, in the original forms: Zhang's MCP and Fan and Li's SCAD.
double q_host(int law, double u, double l, double p) {
  if (law == 0) return u <= p * l ? l * u - u * u / (2.0 * p) : p * l * l / 2.0;
  if (u <= l) return l * u;
  if (u <= p * l) return (2.0 * p * l * u - u * u - l * l) / (2.0 * (p - 1.0));
  return (p + 1.0) * l * l / 2.0;
}
double rho_host(int law, double p) { return law == 0 ? 1.0 / p : 1.0 / (p - 1.0); }

// Per point x_j (a penalty of dimension 1 each): value, J = 1, J = 2, then
// J = 0, 1, 2 in lambda, then in the shape, then whether J = 3 was written.
template <class P>
host_view eval_points(const std::vector<double>& x, double l, double p) {
  const int n = static_cast<int>(x.size());
  Kokkos::View<double*> dx("x", n);
  Kokkos::View<double**, Kokkos::LayoutRight> out("out", n, 10);
  Kokkos::deep_copy(dx, Kokkos::View<const double*, Kokkos::HostSpace>(x.data(), n));
  Kokkos::parallel_for(
      "nonconvex_eval", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int j) {
        const P pen{1};
        const double th[2] = {l, p};
        const auto st = pen.theta_state(th, nullptr);
        const double* b = &dx(j);
        double r[1];
        const int il = 0, ip = 1;
        out(j, 0) = pen.value(st, b, nullptr);
        pen.template value_dbeta_dk<1, 0>(st, b, nullptr, r, nullptr);
        out(j, 1) = r[0];
        pen.template value_dbeta_dk<2, 0>(st, b, nullptr, r, nullptr);
        out(j, 2) = r[0];
        for (int k = 0; k < 2; ++k) {
          const int* idx = k == 0 ? &il : &ip;
          pen.template value_dbeta_dk<0, 1>(st, b, idx, r, nullptr);
          out(j, 3 + 3 * k) = r[0];
          pen.template value_dbeta_dk<1, 1>(st, b, idx, r, nullptr);
          out(j, 4 + 3 * k) = r[0];
          pen.template value_dbeta_dk<2, 1>(st, b, idx, r, nullptr);
          out(j, 5 + 3 * k) = r[0];
        }
        out(j, 9) = pen.template value_dbeta_dk<3, 0>(st, b, nullptr, r, nullptr) ? 1.0 : 0.0;
      });
  host_view h("h", n, 10);
  Kokkos::deep_copy(h, out);
  return h;
}

// The prox of each x_j with step t_j: column 0 by the scalar overload on a
// penalty of dimension 1, column 1 by the per-coordinate overload on one of
// dimension n.
template <class P>
host_view prox_points(const std::vector<double>& x, const std::vector<double>& t, double l, double p) {
  const int n = static_cast<int>(x.size());
  Kokkos::View<double*> dx("x", n), dt("t", n), vec("vec", n);
  Kokkos::View<double**, Kokkos::LayoutRight> out("out", n, 2);
  Kokkos::deep_copy(dx, Kokkos::View<const double*, Kokkos::HostSpace>(x.data(), n));
  Kokkos::deep_copy(dt, Kokkos::View<const double*, Kokkos::HostSpace>(t.data(), n));
  Kokkos::parallel_for(
      "nonconvex_prox", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int j) {
        const P pen{1};
        const double th[2] = {l, p};
        const auto st = pen.theta_state(th, nullptr);
        pen.prox_beta(st, &dx(j), dt(j), &out(j, 0), nullptr);
      });
  Kokkos::parallel_for(
      "nonconvex_prox_steps", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        const P pen{n};
        const double th[2] = {l, p};
        const auto st = pen.theta_state(th, nullptr);
        pen.prox_beta(st, dx.data(), dt.data(), vec.data(), nullptr);
      });
  Kokkos::parallel_for(
      "nonconvex_prox_copy", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(int j) { out(j, 1) = vec(j); });
  host_view h("h", n, 2);
  Kokkos::deep_copy(h, out);
  return h;
}

// Through a map over p coordinates: row 0 the value (in [0][0]), row 1 J = 1,
// row 2 J = 2 (diagonal: p values; differences: J = 2 is not read), row 3 the
// prox at step t (diagonal only).
template <class P, bool diagonal>
host_view mapped(const std::vector<double>& beta, const std::vector<double>& w, double l, double sh, double t) {
  const int p = static_cast<int>(beta.size());
  Kokkos::View<double*> db("b", p), dw("w", p), scratch("scratch", 4 * p + 8);
  Kokkos::View<double**, Kokkos::LayoutRight> out("out", 4, p);
  Kokkos::deep_copy(db, Kokkos::View<const double*, Kokkos::HostSpace>(beta.data(), p));
  Kokkos::deep_copy(dw, Kokkos::View<const double*, Kokkos::HostSpace>(w.data(), p));
  Kokkos::parallel_for(
      "nonconvex_mapped", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(int) {
        // every capture first used outside the constexpr-if (nvcc refuses a first capture inside one)
        const double th[2] = {l, sh};
        const double tt = t;
        const int pp = p;
        double* s = scratch.data();
        const double* b = db.data();
        const double* wt = dw.data();
        double* o1 = &out(1, 0);
        double* o2 = &out(2, 0);
        double* o3 = &out(3, 0);
        double* o0 = &out(0, 0);
        if constexpr (diagonal) {
          const lpen::linear_map_penalty<P, lpen::diagonal_map> pen(P{pp}, lpen::diagonal_map(wt, pp));
          const auto st = pen.theta_state(th, s);
          *o0 = pen.value(st, b, s);
          pen.template value_dbeta_dk<1, 0>(st, b, nullptr, o1, s);
          pen.template value_dbeta_dk<2, 0>(st, b, nullptr, o2, s);
          pen.prox_beta(st, b, tt, o3, s);
        } else {
          const lpen::linear_map_penalty<P, lpen::difference_map<1>> pen(P{pp - 1}, lpen::difference_map<1>(pp));
          const auto st = pen.theta_state(th, s);
          *o0 = pen.value(st, b, s);
          pen.template value_dbeta_dk<1, 0>(st, b, nullptr, o1, s);
          (void)wt;
          (void)o2;
          (void)o3;
          (void)tt;
        }
      });
  host_view h("h", 4, p);
  Kokkos::deep_copy(h, out);
  return h;
}

struct law_case {
  int law;
  double lambda, shape;
};
const law_case cases[] = {{0, 0.7, 1.5}, {0, 2.0, 3.0}, {0, 1.0, 0.4}, {1, 0.7, 2.2}, {1, 2.0, 3.7}, {1, 1.0, 1.3}};

template <class F>
void for_each_case(F f) {
  for (const auto& c : cases) f(c);
}

// points u lambda with u on a grid avoiding the knots (u = 1, shape) by at least 1e-3
std::vector<double> smooth_points(double l, double p) {
  std::vector<double> x;
  for (int i = 1; i <= 400; ++i) {
    const double u = 0.0137 * i;
    if (std::abs(u - 1.0) < 1e-3 || std::abs(u - p) < 1e-3) continue;
    x.push_back(u * l);
    x.push_back(-u * l * 0.999);
  }
  return x;
}

template <class P>
void check_derivatives(const law_case& c, double& worst_value, double& worst_fd, double& worst_theta) {
  const double l = c.lambda, p = c.shape, h = 1e-6;
  const auto x = smooth_points(l, p);
  const int n = static_cast<int>(x.size());
  std::vector<double> xp(x), xm(x);
  for (int j = 0; j < n; ++j) {
    xp[j] += h;
    xm[j] -= h;
  }
  const auto e = eval_points<P>(x, l, p), ep = eval_points<P>(xp, l, p), em = eval_points<P>(xm, l, p);
  const auto lp = eval_points<P>(x, l + h, p), lm = eval_points<P>(x, l - h, p);
  const auto pp = eval_points<P>(x, l, p + h), pm = eval_points<P>(x, l, p - h);
  for (int j = 0; j < n; ++j) {
    const double u = std::abs(x[j]);
    worst_value = std::max(worst_value, std::abs(e(j, 0) - q_host(c.law, u, l, p)) / std::max(1.0, e(j, 0)));
    // in beta: J = 1 from the value, J = 2 from J = 1 (no knot within h)
    worst_fd = std::max(worst_fd, std::abs(e(j, 1) - (ep(j, 0) - em(j, 0)) / (2 * h)));
    worst_fd = std::max(worst_fd, std::abs(e(j, 2) - (ep(j, 1) - em(j, 1)) / (2 * h)));
    // in theta: each order J against the difference of the same order in lambda and in the shape
    // (a knot moves by h l or h p, so points within 1e-3 of a knot are excluded above)
    for (int J = 0; J < 3; ++J) {
      worst_theta = std::max(worst_theta, std::abs(e(j, 3 + J) - (lp(j, J) - lm(j, J)) / (2 * h)));
      worst_theta = std::max(worst_theta, std::abs(e(j, 6 + J) - (pp(j, J) - pm(j, J)) / (2 * h)));
    }
    CHECK(e(j, 9) == 0.0);
  }
}

}  // namespace nonconvex_test

using namespace nonconvex_test;

TEST_CASE("nonconvex penalties: value against the original forms, derivatives against central differences") {
  double wv = 0.0, wf = 0.0, wt = 0.0;
  for_each_case([&](const law_case& c) {
    if (c.law == 0)
      check_derivatives<lpen::mcp_penalty>(c, wv, wf, wt);
    else
      check_derivatives<lpen::scad_penalty>(c, wv, wf, wt);
  });
  MESSAGE("value ", wv, ", beta differences ", wf, ", theta differences ", wt);
  CHECK(wv < 2e-15);
  CHECK(wf < 5e-9);
  CHECK(wt < 5e-9);
}

template <class P>
void check_knots(const law_case& c) {
  const double l = c.lambda, p = c.shape, d = 1e-12;
  std::vector<double> knots = {p * l};
  if (c.law == 1) knots.push_back(l);
  std::vector<double> x = {0.0};
  for (const double k : knots) {
    x.push_back(k);
    x.push_back(k * (1 - d));
    x.push_back(k * (1 + d));
  }
  const auto e = eval_points<P>(x, l, p);
  // sign(0) = 0 for the odd order; the inner piece at u = 0 for the even ones
  CHECK(e(0, 0) == 0.0);
  CHECK(e(0, 1) == 0.0);
  CHECK(e(0, 4) == 0.0);
  CHECK(e(0, 7) == 0.0);
  CHECK(e(0, 2) == (c.law == 0 ? -1.0 / p : 0.0));
  for (std::size_t k = 0; k < knots.size(); ++k) {
    const int a = 1 + 3 * static_cast<int>(k);
    // the inner piece at the knot: equal to the point just inside for every order
    for (const int col : {2, 5, 8}) CHECK(e(a, col) == e(a + 1, col));
    // C1: value, J = 1 and d/dtheta continuous across the knot
    for (const int col : {0, 1, 3, 6}) CHECK(std::abs(e(a + 2, col) - e(a + 1, col)) < 1e-10);
  }
}

TEST_CASE("nonconvex penalties: conventions at zero and at the knots, continuity across them") {
  for_each_case([&](const law_case& c) {
    if (c.law == 0)
      check_knots<lpen::mcp_penalty>(c);
    else
      check_knots<lpen::scad_penalty>(c);
  });
}

template <class P>
void check_prox(const law_case& c, double& worst_kkt, double& worst_golden, double& worst_rho) {
  const double l = c.lambda, p = c.shape, rho = rho_host(c.law, p);
  const double bound = 1.0 / (1.0 / p);  // t = 1/rho as an optimizer computes it (MCP)
  std::vector<double> ts = {1e-8, 0.1 / rho, 0.5 / rho, 0.99 / rho, 1.0 / rho, c.law == 0 ? bound : 1.0 / rho_host(1, p)};
  std::vector<double> x, t;
  for (const double tt : ts)
    for (const double u : {0.0, 0.3, 1.0, 1.0 + tt, 1.5, 0.5 * (1.0 + tt + p), p, 1.5 * p, 50.0})
      for (const double s : {-1.0, 1.0}) {
        x.push_back(s * u * l);
        t.push_back(tt);
        x.push_back(s * u * l * (1 + 1e-9));
        t.push_back(tt);
      }
  const auto r = prox_points<P>(x, t, l, p);
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double v = x[j], tt = t[j], b = r(j, 0);
    REQUIRE(std::isfinite(b));
    CHECK(b == r(j, 1));
    // optimality: q'(|b|) sign(b) + (b - v)/t = 0 when b != 0, |v| <= t lambda when b = 0
    if (b == 0.0) {
      CHECK(std::abs(v) <= tt * l * (1 + 1e-15));
    } else {
      const double g = (c.law == 0 ? (std::abs(b) <= p * l ? l - std::abs(b) / p : 0.0)
                                   : (std::abs(b) <= l ? l : (std::abs(b) <= p * l ? (p * l - std::abs(b)) / (p - 1) : 0.0)));
      worst_kkt = std::max(worst_kkt, std::abs(tt * g * (b > 0 ? 1 : -1) + b - v) / std::max(1.0, std::abs(v)));
    }
    // golden section on [min(0, v), max(0, v)] (the objective is convex for t rho <= 1)
    auto f = [&](double y) { return q_host(c.law, std::abs(y), l, p) + (y - v) * (y - v) / (2 * tt); };
    double a = std::min(0.0, v), bb = std::max(0.0, v);
    const double g = 0.38196601125010515;
    for (int i = 0; i < 300; ++i) {
      const double y1 = a + g * (bb - a), y2 = bb - g * (bb - a);
      if (f(y1) <= f(y2)) bb = y2; else a = y1;
    }
    const double fy = std::min({f(0.5 * (a + bb)), f(0.0), f(v)});
    worst_golden = std::max(worst_golden, (f(b) - fy) / std::max(1.0, std::abs(fy)));
  }
  // t rho > 1 is refused
  const auto n = prox_points<P>({0.5 * l, 3.0 * l}, {1.01 / rho, 10.0 / rho}, l, p);
  CHECK(std::isnan(n(0, 0)));
  CHECK(std::isnan(n(1, 1)));
  // rho = max(-pen'') on a grid
  std::vector<double> grid;
  for (int i = 0; i <= 2000; ++i) grid.push_back(i * 0.003 * p * l);
  const auto e = eval_points<P>(grid, l, p);
  double m = -std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < grid.size(); ++j) m = std::max(m, -e(j, 2));
  const double th[2] = {l, p};
  worst_rho = std::max(worst_rho, std::abs(m - rho) + std::abs(P{1}.weak_convexity(th) - rho));
}

TEST_CASE("nonconvex penalties: prox optimality, golden section, t rho bound, rho on a grid") {
  double wk = 0.0, wg = 0.0, wr = 0.0;
  for_each_case([&](const law_case& c) {
    if (c.law == 0)
      check_prox<lpen::mcp_penalty>(c, wk, wg, wr);
    else
      check_prox<lpen::scad_penalty>(c, wk, wg, wr);
  });
  MESSAGE("prox optimality ", wk, ", above the golden-section minimum ", wg, ", rho ", wr);
  CHECK(wk < 1e-15);
  CHECK(wg < 2e-15);
  CHECK(wr == 0.0);
}

template <class P>
void check_maps(const law_case& c, double& worst_diag, double& worst_diff) {
  const double l = c.lambda, p = c.shape, rho = rho_host(c.law, p);
  const int n = 9;
  std::vector<double> beta(n), w(n);
  for (int j = 0; j < n; ++j) {
    beta[j] = (j % 3 == 0 ? -1.0 : 1.0) * 0.37 * (j + 1) * l / 3.0;
    w[j] = 0.5 + 0.25 * j;
  }
  // diagonal: pen = sum q(w_j |beta_j|); prox_j = prox_{t w_j^2}(w_j v_j)/w_j, t w_j^2 rho <= 1
  const double t = 0.9 / (rho * w[n - 1] * w[n - 1]);
  const auto d = mapped<P, true>(beta, w, l, p, t);
  double val = 0.0;
  for (int j = 0; j < n; ++j) val += q_host(c.law, w[j] * std::abs(beta[j]), l, p);
  worst_diag = std::max(worst_diag, std::abs(d(0, 0) - val) / val);
  std::vector<double> wv(n), tw(n);
  for (int j = 0; j < n; ++j) {
    wv[j] = w[j] * beta[j];
    tw[j] = t * w[j] * w[j];
  }
  const auto one = eval_points<P>(wv, l, p);
  const auto px = prox_points<P>(wv, tw, l, p);
  for (int j = 0; j < n; ++j) {
    worst_diag = std::max(worst_diag, std::abs(d(1, j) - w[j] * one(j, 1)));
    worst_diag = std::max(worst_diag, std::abs(d(2, j) - w[j] * w[j] * one(j, 2)));
    worst_diag = std::max(worst_diag, std::abs(d(3, j) - px(j, 0) / w[j]));
  }
  // differences of order 1: pen = sum q(|beta_{j+1} - beta_j|), J = 1 by central differences
  const auto e = mapped<P, false>(beta, w, l, p, 0.0);
  double dv = 0.0;
  for (int j = 0; j + 1 < n; ++j) dv += q_host(c.law, std::abs(beta[j + 1] - beta[j]), l, p);
  worst_diff = std::max(worst_diff, std::abs(e(0, 0) - dv) / dv);
  const double h = 1e-6;
  for (int j = 0; j < n; ++j) {
    auto bp = beta, bm = beta;
    bp[j] += h;
    bm[j] -= h;
    const double fd = (mapped<P, false>(bp, w, l, p, 0.0)(0, 0) - mapped<P, false>(bm, w, l, p, 0.0)(0, 0)) / (2 * h);
    worst_diff = std::max(worst_diff, std::abs(e(1, j) - fd));
  }
}

TEST_CASE("nonconvex penalties: through linear_map_penalty, diagonal map and first differences") {
  double wd = 0.0, wf = 0.0;
  for_each_case([&](const law_case& c) {
    if (c.law == 0)
      check_maps<lpen::mcp_penalty>(c, wd, wf);
    else
      check_maps<lpen::scad_penalty>(c, wd, wf);
  });
  MESSAGE("diagonal map ", wd, ", differences ", wf);
  CHECK(wd < 1e-16);
  CHECK(wf < 5e-9);
}

TEST_CASE("nonconvex penalties: penalties7 (statmodels7) references") {
  std::ifstream in(std::string(LAPLACE_REFERENCE_DIR) + "/penalty_nonconvex_s7.csv");
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  struct row {
    int law, kind;
    double l, p, t, x, ref;
  };
  std::vector<row> rows;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string f[8];
    for (auto& s : f) std::getline(ss, s, ',');
    rows.push_back({std::atoi(f[0].c_str()), std::atoi(f[1].c_str()), std::strtod(f[2].c_str(), nullptr),
                    std::strtod(f[3].c_str(), nullptr), std::strtod(f[4].c_str(), nullptr),
                    std::strtod(f[5].c_str(), nullptr), std::strtod(f[6].c_str(), nullptr)});
  }
  REQUIRE(rows.size() == 744);
  double worst = 0.0;
  for (const auto& r : rows) {
    const std::vector<double> x = {r.x};
    double mine;
    if (r.kind == 3) {
      mine = r.law == 0 ? prox_points<lpen::mcp_penalty>(x, {r.t}, r.l, r.p)(0, 0)
                        : prox_points<lpen::scad_penalty>(x, {r.t}, r.l, r.p)(0, 0);
    } else {
      const auto e = r.law == 0 ? eval_points<lpen::mcp_penalty>(x, r.l, r.p) : eval_points<lpen::scad_penalty>(x, r.l, r.p);
      mine = e(0, r.kind);
    }
    worst = std::max(worst, std::abs(mine - r.ref) / std::max(1.0, std::abs(r.ref)));
  }
  MESSAGE("penalties7: ", rows.size(), " rows, worst relative difference ", worst);
  CHECK(worst < 5e-16);
}
