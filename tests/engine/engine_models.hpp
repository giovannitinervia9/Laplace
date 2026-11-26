// Simulated models for the tests of the engine (test_row_kernels.cpp,
// test_inner_fit.cpp): data drawn with a generator of our own (std's
// distributions differ between standard libraries), and the specifications
// that row_model reads.
//
//   gauss  y ~ N(mu, sigma), mu = eta0 = b0 + b1 x + u_g, sigma = exp(eta1), eta1 = c0 + c1 w;
//          G groups of a random intercept u_g (the arrowhead)
//   gamma  y ~ gamma1(mu, phi), mu = A / (1 + exp((xm - t) / s)) (logistic_curve),
//          A = eta0 = a0 + u_g, xm = eta1, s = exp(eta2), phi = exp(eta3): a link node feeding
//          the curve, so the general chain rule runs through a nesting
//   student y ~ student_t1(mu, sigma, nu), mu = eta0 = b0 + b1 x, sigma = exp(eta1),
//          nu = exp(eta2): separable, no grouping
#pragma once

#include <laplace/basis/bspline.hpp>
#include <laplace/bijector/scalar.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/distrib/gaussian1.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/engine/inner_fit.hpp>
#include <laplace/engine/row_model.hpp>
#include <laplace/pdmat/scaled_identity.hpp>
#include <laplace/pdmat/weighted_sum.hpp>
#include <laplace/penalty/distrib_penalty.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine_test {

namespace le = laplace::engine;
using engine_distribs = le::distrib_list<laplace::distrib::gaussian1_distrib, laplace::distrib::gamma1_distrib,
                                         laplace::distrib::student_t1_distrib>;
enum : int { GAUSS = 0, GAMMA = 1, STUDENT = 2 };

struct rng {
  std::uint64_t s;
  explicit rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {  // splitmix64
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double unif() { return (double)((next() >> 11) + 1) * (1.0 / 9007199254740994.0); }
  double norm() { return std::sqrt(-2.0 * std::log(unif())) * std::cos(6.283185307179586 * unif()); }
  double gamma(double k) {  // Marsaglia and Tsang, k >= 1
    const double d = k - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
      const double z = norm(), v = (1.0 + c * z) * (1.0 + c * z) * (1.0 + c * z), u = unif();
      if (v > 0.0 && std::log(u) < 0.5 * z * z + d - d * v + d * std::log(v)) return d * v;
    }
  }
};

inline laplace::bijector::scalar_bijector log_link() {
  return laplace::bijector::scalar_bijector{laplace::bijector::scalar_bijector_map::log};
}

// ------------------------------------------------- penalties
//
// A univariate law D (of laplace::distrib, through univariate_as_multivariate) as the prior of each
// coordinate first + g * stride + j, g = 0 .. G - 1: one replicate per group.
template <class D>
le::penalty_term univariate_prior(D law, int first, int G, int stride, std::vector<int> source, std::vector<double> value) {
  using uam = laplace::distrib::univariate_as_multivariate<D>;
  const laplace::penalty::distrib_penalty<uam> p{uam(law)};
  std::vector<std::vector<int>> reps;
  for (int g = 0; g < G; ++g) reps.push_back({first + g * stride});
  return le::make_penalty_term(p, reps, std::move(source), std::move(value));
}

// u_g ~ N(0, tau^2) on the group effects (gaussian1, theta = tau)
inline le::penalty_term group_gaussian_prior(const le::row_model_spec& s, int j, int source, double tau) {
  return univariate_prior(laplace::distrib::gaussian1_distrib{}, s.n_global + j, s.n_groups, s.group_width, {source}, {tau});
}

// the improper Gaussian prior lambda S on the coordinates [first, first + p), S p x p column-major
// (mvgaussian2 over weighted_sum_psdmat, theta = log lambda)
inline le::penalty_term smooth_prior(int first, int p, const std::vector<double>& S, int source, double log_lambda) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> m("S", p, p);
  for (int j = 0; j < p; ++j)
    for (int k = 0; k < p; ++k) m(j, k) = S[j + p * k];
  using mvg = laplace::distrib::mvgaussian2_distrib<laplace::pdmat::weighted_sum_psdmat>;
  const laplace::penalty::distrib_penalty<mvg> pen{mvg(laplace::pdmat::weighted_sum_psdmat(std::vector<decltype(m)>{m}))};
  std::vector<int> c(p);
  for (int j = 0; j < p; ++j) c[j] = first + j;
  return le::make_penalty_term(pen, {c}, {source}, {log_lambda});
}

inline le::pointwise_node link_node(int input) {
  le::pointwise_node nd;
  nd.kind = le::pointwise_node_kind::link;
  nd.input[0] = input;
  nd.link = log_link();
  return nd;
}

// coordinates: b0, b1, c0, c1, then u_0 .. u_{G-1}
inline le::row_model_spec gauss_spec(int n, int G, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 2;
  m.distrib = GAUSS;
  m.n_global = 4;
  m.n_groups = G, m.group_width = 1;
  std::vector<double> x(n), w(n), one(n, 1.0), u(G);
  for (auto& v : u) v = 0.5 * r.norm();
  m.group.resize(n);
  m.y.resize(n);
  for (int i = 0; i < n; ++i) {
    x[i] = r.norm(), w[i] = r.unif() - 0.5, m.group[i] = static_cast<int>(r.next() % G);
    const double mu = 1.0 + 0.5 * x[i] + u[m.group[i]], sigma = std::exp(-0.3 + 0.8 * w[i]);
    m.y[i] = mu + sigma * r.norm();
  }
  m.design = {le::design_column(0, one, 0), le::design_column(0, x, 1), le::design_column(1, one, 2),
              le::design_column(1, w, 3)};
  m.design.push_back(le::design_group_effect(m, 0, one, 0));
  m.nodes = {link_node(1)};
  m.param_value = {0, 2};
  return m;
}

// gauss with a random intercept in each predictor: mu = b0 + b1 x + u_g, log sigma = c0 + c1 w + v_g,
// (u_g, v_g) correlated. Coordinates: b0, b1, c0, c1, then (u_g, v_g) for g = 0 .. G - 1
inline le::row_model_spec gauss2_spec(int n, int G, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 2;
  m.distrib = GAUSS;
  m.n_global = 4;
  m.n_groups = G, m.group_width = 2;
  std::vector<double> x(n), w(n), one(n, 1.0), u(G), v(G);
  for (int g = 0; g < G; ++g) {
    const double a = r.norm(), b = r.norm();
    u[g] = 0.5 * a, v[g] = 0.2 * (0.6 * a + 0.8 * b);
  }
  m.group.resize(n);
  m.y.resize(n);
  for (int i = 0; i < n; ++i) {
    x[i] = r.norm(), w[i] = r.unif() - 0.5, m.group[i] = static_cast<int>(r.next() % G);
    const double mu = 1.0 + 0.5 * x[i] + u[m.group[i]], sigma = std::exp(-0.3 + 0.8 * w[i] + v[m.group[i]]);
    m.y[i] = mu + sigma * r.norm();
  }
  m.design = {le::design_column(0, one, 0), le::design_column(0, x, 1), le::design_column(1, one, 2),
              le::design_column(1, w, 3)};
  m.design.push_back(le::design_group_effect(m, 0, one, 0));
  m.design.push_back(le::design_group_effect(m, 1, one, 1));
  m.nodes = {link_node(1)};
  m.param_value = {0, 2};
  return m;
}

// two crossed groupings: mu = b0 + b1 x + u_g + a_h, log sigma = c0 + c1 w + v_g, g of grouping A
// (width 2: u_g, v_g) and h of grouping B (width 1), drawn independently of g. Coordinates: b0, b1, c0,
// c1, then (u_g, v_g) for g = 0 .. G - 1, then a_h for h = 0 .. H - 1
inline le::row_model_spec crossed_spec(int n, int G, int H, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 2;
  m.distrib = GAUSS;
  m.n_global = 4;
  m.n_groups = G, m.group_width = 2;
  m.crossed.resize(1);
  auto& B = m.crossed[0];
  B.n_groups = H, B.width = 1;
  std::vector<double> x(n), w(n), one(n, 1.0), u(G), v(G), a(H);
  for (int g = 0; g < G; ++g) u[g] = 0.5 * r.norm(), v[g] = 0.2 * r.norm();
  for (int h = 0; h < H; ++h) a[h] = 0.4 * r.norm();
  m.group.resize(n), B.group.resize(n);
  m.y.resize(n);
  for (int i = 0; i < n; ++i) {
    x[i] = r.norm(), w[i] = r.unif() - 0.5;
    m.group[i] = static_cast<int>(r.next() % G), B.group[i] = static_cast<int>(r.next() % H);
    const double mu = 1.0 + 0.5 * x[i] + u[m.group[i]] + a[B.group[i]];
    const double sigma = std::exp(-0.3 + 0.8 * w[i] + v[m.group[i]]);
    m.y[i] = mu + sigma * r.norm();
  }
  m.design = {le::design_column(0, one, 0), le::design_column(0, x, 1), le::design_column(1, one, 2),
              le::design_column(1, w, 3)};
  m.design.push_back(le::design_group_effect(m, 0, one, 0));
  m.design.push_back(le::design_group_effect(m, 1, one, 1));
  m.design.push_back(le::design_crossed_effect(m, 1, 0, one, 0));
  m.nodes = {link_node(1)};
  m.param_value = {0, 2};
  return m;
}

// a sparse location: mu = b0 + sum_k b_k x_k (k = 1 .. 6, b = 0.8, 0, 0, -0.5, 0, 0), log sigma = c0 + c1 w.
// Coordinates: b0, b1 .. b6, c0, c1
inline le::row_model_spec lasso_spec(int n, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 2;
  m.distrib = GAUSS;
  m.n_global = 9;
  const double b[7] = {1.0, 0.8, 0.0, 0.0, -0.5, 0.0, 0.0};
  std::vector<std::vector<double>> x(7, std::vector<double>(n, 1.0));
  std::vector<double> w(n), one(n, 1.0);
  m.y.resize(n);
  for (int i = 0; i < n; ++i) {
    double mu = b[0];
    for (int k = 1; k < 7; ++k) x[k][i] = r.norm(), mu += b[k] * x[k][i];
    w[i] = r.unif() - 0.5;
    m.y[i] = mu + std::exp(-0.3 + 0.8 * w[i]) * r.norm();
  }
  for (int k = 0; k < 7; ++k) m.design.push_back(le::design_column(0, x[k], k));
  m.design.push_back(le::design_column(1, one, 7));
  m.design.push_back(le::design_column(1, w, 8));
  m.nodes = {link_node(1)};
  m.param_value = {0, 2};
  return m;
}

// coordinates: a0, xm, log s, log phi, then u_0 .. u_{G-1}
inline le::row_model_spec gamma_spec(int n, int G, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 4;
  m.distrib = GAMMA;
  m.n_global = 4;
  m.n_groups = G, m.group_width = 1;
  std::vector<double> t(n), one(n, 1.0), u(G);
  for (auto& v : u) v = 0.3 * r.norm();
  m.group.resize(n);
  m.y.resize(n);
  const double phi = std::exp(-2.0);
  for (int i = 0; i < n; ++i) {
    t[i] = 10.0 * r.unif(), m.group[i] = static_cast<int>(r.next() % G);
    const double A = 3.0 + u[m.group[i]], mu = A / (1.0 + std::exp((5.0 - t[i]) / 1.5));
    m.y[i] = r.gamma(1.0 / phi) * mu * phi;
  }
  m.design = {le::design_column(0, one, 0), le::design_column(1, one, 1), le::design_column(2, one, 2),
              le::design_column(3, one, 3)};
  m.design.push_back(le::design_group_effect(m, 0, one, 0));
  m.data_column = {t};
  le::pointwise_node curve;
  curve.kind = le::pointwise_node_kind::logistic_curve;
  curve.input[0] = 0, curve.input[1] = 1, curve.input[2] = 4;  // value 4: exp(eta2), the first node
  curve.data_column = 0;
  m.nodes = {link_node(2), curve, link_node(3)};  // values 4, 5, 6
  m.param_value = {5, 6};
  return m;
}

// coordinates: b0, b1, c0, c1, log nu
inline le::row_model_spec student_spec(int n, std::uint64_t seed) {
  rng r(seed);
  le::row_model_spec m;
  m.n_predictors = 3;
  m.distrib = STUDENT;
  m.n_global = 5;
  std::vector<double> x(n), w(n), one(n, 1.0);
  m.y.resize(n);
  const double nu = 5.0;
  for (int i = 0; i < n; ++i) {
    x[i] = r.norm(), w[i] = r.unif() - 0.5;
    const double mu = 1.0 + 0.5 * x[i], sigma = std::exp(-0.3 + 0.8 * w[i]);
    m.y[i] = mu + sigma * r.norm() / std::sqrt(2.0 * r.gamma(nu / 2.0) / nu);
  }
  m.design = {le::design_column(0, one, 0), le::design_column(0, x, 1), le::design_column(1, one, 2),
              le::design_column(1, w, 3), le::design_column(2, one, 4)};
  m.nodes = {link_node(1), link_node(2)};  // values 3, 4
  m.param_value = {0, 3, 4};
  return m;
}

// ------------------------------------------------- the models of the prototypes
//
// A model of a prototype with what its inner fit holds fixed: the penalties at its hyperparameters,
// the held coordinates and its start.
struct prototype_fit {
  le::row_model_spec spec;
  std::vector<le::penalty_term> penalties;  // their hyperparameters fixed (source -1)
  std::vector<int> held;
  std::vector<double> x0;
};

// reads a CSV with a header line and k numeric columns; false when the file cannot be read
inline bool read_columns(const char* path, int k, std::vector<std::vector<double>>& cols) {
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return false;
  char line[1024];
  if (std::fgets(line, sizeof line, f) == nullptr) {
    std::fclose(f);
    return false;
  }
  cols.assign(k, {});
  while (std::fgets(line, sizeof line, f) != nullptr) {
    char* p = line;
    for (int j = 0; j < k; ++j) {
      cols[j].push_back(std::strtod(p, &p));
      if (*p == ',') ++p;
    }
  }
  std::fclose(f);
  return true;
}

// The model of prototipi/kokkos_fit/lk.cpp: y ~ N(mu, sigma), mu = b0 + b1 x + s(t) + u_g,
// log sigma = c0 + c1 w, s a cubic B-spline of 10 columns (the last dropped) with the penalty
// lambda S0 on the integrated squared second derivative (log lambda = rho1), u_g ~ N(0, tau^2)
// (tau = exp(rho0)). Columns of the data: id (from 1), x, t, w, y. Coordinates: 0 b0, 1 b1,
// 2..10 the spline, 11 c0, 12 c1 (held), 13 + g the group effects. Start: the mean of y, zeros.
inline prototype_fit kokkos_fit_model(const char* path, double rho0, double rho1, double c0, double c1) {
  std::vector<std::vector<double>> col;
  if (!read_columns(path, 5, col)) throw std::runtime_error(std::string("cannot read ") + path);
  const std::vector<double>&id = col[0], &x = col[1], &t = col[2], &w = col[3], &y = col[4];
  const int n = static_cast<int>(y.size()), NG = 13;
  int G = 0;
  for (double v : id) G = std::max(G, static_cast<int>(v));
  // the basis on lk.cpp's padded range, and S0 by the three-point Gauss rule on each knot interval
  double lo = *std::min_element(t.begin(), t.end()), hi = *std::max_element(t.begin(), t.end());
  const double pad = (hi - lo) * 0.001 + 2.220446e-16;
  lo -= pad, hi += pad;
  const auto owner = laplace::basis::bspline_basis_owner::uniform_clamped(lo, hi, 10, 3);
  const auto basis = owner.host();
  std::vector<double> S0(81, 0.0);
  const double gx[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)}, gw[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  for (int s = 0; s < 7; ++s) {
    const double a = lo + (hi - lo) * s / 7.0, b = lo + (hi - lo) * (s + 1) / 7.0;
    for (int q = 0; q < 3; ++q) {
      const double xx = 0.5 * (a + b) + 0.5 * (b - a) * gx[q], wt = 0.5 * (b - a) * gw[q];
      double r[4];
      const int first = basis.row_d2(xx, r);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          if (first + i < 9 && first + j < 9) S0[(first + i) + 9 * (first + j)] += wt * r[i] * r[j];
    }
  }
  prototype_fit pf;
  le::row_model_spec& s = pf.spec;
  s.n_predictors = 2;
  s.distrib = GAUSS;
  s.y = y;
  s.n_global = NG, s.n_groups = G, s.group_width = 1;
  s.group.resize(n);
  for (int i = 0; i < n; ++i) s.group[i] = static_cast<int>(id[i]) - 1;
  const std::vector<double> one(n, 1.0);
  s.design = {le::design_column(0, one, 0), le::design_column(0, x, 1), le::design_basis(0, basis, t, 2, {9}),
              le::design_column(1, one, 11), le::design_column(1, w, 12)};
  s.design.push_back(le::design_group_effect(s, 0, one, 0));
  s.nodes = {link_node(1)};
  s.param_value = {0, 2};
  pf.penalties = {group_gaussian_prior(s, 0, -1, std::exp(rho0)), smooth_prior(2, 9, S0, -1, rho1)};
  pf.held = {11, 12};
  pf.x0.assign(NG + G, 0.0);
  double my = 0.0;
  for (double v : y) my += v;
  pf.x0[0] = my / n, pf.x0[11] = c0, pf.x0[12] = c1;
  return pf;
}

// Model C of prototipi/node_dispatch/hy4_main.cpp (the engine of lk3.cpp with a gamma):
// y ~ gamma1(mu, phi), mu = (A0 + b_g) / (1 + exp((xm - t) / s)), phi = exp(c0), b_g ~ N(0, tau^2)
// (tau = exp(rho)). Columns: id (from 1), t, y. Coordinates: 0 A0, 1 xm, 2 s, 3 c0 (held), 4 + g the
// group effects. Start: 150, 600, 300, c0, zeros.
inline prototype_fit lk3_gamma_model(const char* path, double rho, double c0) {
  std::vector<std::vector<double>> col;
  if (!read_columns(path, 3, col)) throw std::runtime_error(std::string("cannot read ") + path);
  const std::vector<double>&id = col[0], &t = col[1], &y = col[2];
  const int n = static_cast<int>(y.size()), NG = 4;
  int G = 0;
  for (double v : id) G = std::max(G, static_cast<int>(v));
  prototype_fit pf;
  le::row_model_spec& s = pf.spec;
  s.n_predictors = 4;
  s.distrib = GAMMA;
  s.y = y;
  s.n_global = NG, s.n_groups = G, s.group_width = 1;
  s.group.resize(n);
  for (int i = 0; i < n; ++i) s.group[i] = static_cast<int>(id[i]) - 1;
  const std::vector<double> one(n, 1.0);
  for (int r = 0; r < 4; ++r) s.design.push_back(le::design_column(r, one, r));
  s.design.push_back(le::design_group_effect(s, 0, one, 0));
  s.data_column = {t};
  le::pointwise_node curve;
  curve.kind = le::pointwise_node_kind::logistic_curve;
  curve.input[0] = 0, curve.input[1] = 1, curve.input[2] = 2;
  curve.data_column = 0;
  s.nodes = {curve, link_node(3)};  // values 4 and 5
  s.param_value = {4, 5};
  pf.penalties = {group_gaussian_prior(s, 0, -1, std::exp(rho))};
  pf.held = {3};
  pf.x0.assign(NG + G, 0.0);
  pf.x0[0] = 150.0, pf.x0[1] = 600.0, pf.x0[2] = 300.0, pf.x0[3] = c0;
  return pf;
}

}  // namespace engine_test
