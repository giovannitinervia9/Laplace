// The row kernels of the engine (engine/row_kernels.hpp) on the three models
// of engine_models.hpp, at a point away from the optimum:
//   - against a twin written here by hand: theta(eta) and its derivatives
//     model by model, composed with dense tensors over the original
//     (unsorted) units and the design rows of the specification;
//   - the gradient against central differences of the value, the observed
//     information against central differences of the gradient;
//   - the four kernels agree on the orders they share (the value of every
//     kernel, the gradient of the three that give it).
// Host code: the kernels are launched by the library's functions.

#include <doctest/doctest.h>

#include <laplace/engine/inner_fit.hpp>

#include "engine_models.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

namespace le = laplace::engine;
namespace lo = laplace::optimizers;
namespace ld = laplace::distrib;
using namespace engine_test;

// theta and its derivatives in eta, by hand: t[j], t1[j][a], t2[j][a][b]
struct theta_jet {
  double t[3] = {}, t1[3][4] = {}, t2[3][4][4] = {};
};

theta_jet hand_theta(int model, const double* eta, double tcov) {
  theta_jet o;
  if (model == GAUSS || model == STUDENT) {
    o.t[0] = eta[0], o.t1[0][0] = 1.0;
    for (int j = 1; j < (model == GAUSS ? 2 : 3); ++j) {
      const double e = std::exp(eta[j]);
      o.t[j] = e, o.t1[j][j] = e, o.t2[j][j][j] = e;
    }
  } else {  // gamma: mu = A / (1 + exp((xm - t) / s)), s = exp(eta2)
    const double A = eta[0], xm = eta[1], s = std::exp(eta[2]);
    const double z = (xm - tcov) / s, p = 1.0 / (1.0 + std::exp(z)), q = p * (1.0 - p), w = 1.0 - 2.0 * p;
    const double fA = p, fx = -A * q / s, fs = A * q * z / s;
    const double fAx = -q / s, fAs = q * z / s, fxx = A * q * w / (s * s), fxs = A * q * (1.0 - w * z) / (s * s),
                 fss = A * q * z * (w * z - 2.0) / (s * s);
    o.t[0] = A * p;
    o.t1[0][0] = fA, o.t1[0][1] = fx, o.t1[0][2] = fs * s;
    o.t2[0][0][1] = o.t2[0][1][0] = fAx;
    o.t2[0][0][2] = o.t2[0][2][0] = fAs * s;
    o.t2[0][1][1] = fxx;
    o.t2[0][1][2] = o.t2[0][2][1] = fxs * s;
    o.t2[0][2][2] = fss * s * s + fs * s;
    const double e = std::exp(eta[3]);
    o.t[1] = e, o.t1[1][3] = e, o.t2[1][3][3] = e;
  }
  return o;
}

template <class D>
D make(const double* t) {
  if constexpr (D::n_params == 2) return D{t[0], t[1]};
  else return D{t[0], t[1], t[2]};
}

// the twin: f, its gradient and the dense information (observed or expected) in x
template <class D>
void twin(const le::row_model_spec& s, int model, const std::vector<double>& x, bool expected, double& f,
          std::vector<double>& g, std::vector<double>& H) {
  constexpr int P = D::n_params;
  const int n = static_cast<int>(s.y.size()), R = s.n_predictors, N = static_cast<int>(x.size());
  f = 0.0;
  g.assign(N, 0.0);
  H.assign(N * N, 0.0);
  for (int i = 0; i < n; ++i) {
    std::vector<int> ec, er;
    std::vector<double> ev;
    double eta[4] = {};
    for (const auto& d : s.design)
      for (int j = 0; j < d.width; ++j) {
        const int c = d.coefficient[i * d.width + j];
        if (c < 0) continue;
        ec.push_back(c), er.push_back(d.predictor), ev.push_back(d.value[i * d.width + j]);
        eta[d.predictor] += d.value[i * d.width + j] * x[c];
      }
    const theta_jet th = hand_theta(model, eta, s.data_column.empty() ? 0.0 : s.data_column[0][i]);
    const D dist = make<D>(th.t);
    f -= dist.logpdf(s.y[i]);
    const auto l1 = dist.logpdf_d1(s.y[i]);
    const auto l2 = dist.logpdf_d2(s.y[i]);
    const auto I = dist.expected_info();
    auto pk = [](int j, int k) { return j <= k ? le::packed_pair(j, k, P) : le::packed_pair(k, j, P); };
    double le1[4] = {}, le2[4][4] = {};
    for (int a = 0; a < R; ++a) {
      for (int j = 0; j < P; ++j) le1[a] += l1[j] * th.t1[j][a];
      for (int b = 0; b < R; ++b) {
        double v = 0.0;
        for (int j = 0; j < P; ++j)
          for (int k = 0; k < P; ++k)
            v += (expected ? -I[pk(j, k)] : l2[pk(j, k)]) * th.t1[j][a] * th.t1[k][b];
        if (!expected)
          for (int j = 0; j < P; ++j) v += l1[j] * th.t2[j][a][b];
        le2[a][b] = v;
      }
    }
    for (std::size_t e = 0; e < ec.size(); ++e) {
      g[ec[e]] -= ev[e] * le1[er[e]];
      for (std::size_t q = 0; q < ec.size(); ++q) H[ec[e] + N * ec[q]] -= ev[e] * ev[q] * le2[er[e]][er[q]];
    }
  }
}

using dense_objective = le::row_model_objective<engine_distribs, lo::dense_curvature_matrix>;

struct kernel_eval {
  double f;
  std::vector<double> g, H;
};

kernel_eval run(dense_objective& obj, const std::vector<double>& x) {
  const int N = static_cast<int>(x.size());
  lo::optimizer_vector xv("x", N), g("g", N);
  for (int j = 0; j < N; ++j) xv(j) = x[j];
  lo::dense_curvature_matrix H("H", N, N);
  kernel_eval o;
  o.f = obj.value_gradient_curvature(xv, g, H);
  o.g.assign(g.data(), g.data() + N);
  o.H.resize(N * N);
  for (int c = 0; c < N; ++c)
    for (int r = 0; r < N; ++r) o.H[r + N * c] = H(r, c);
  return o;
}

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (double a : v) m = std::max(m, std::abs(a));
  return m;
}

template <class D>
void check_model(const le::row_model_spec& s, int model, const std::vector<double>& x) {
  const le::row_model m = le::build_row_model(s, D::n_params, 7);
  const int N = static_cast<int>(x.size());
  REQUIRE(m.n_coordinates() == N);
  dense_objective obs(m, {}, le::row_curvature::observed), exp(m, {}, le::row_curvature::expected);
  const kernel_eval ko = run(obs, x), ke = run(exp, x);
  double tf, tf2;
  std::vector<double> tg, tH, tg2, tHe;
  twin<D>(s, model, x, false, tf, tg, tH);
  twin<D>(s, model, x, true, tf2, tg2, tHe);

  // against the twin
  CHECK(std::abs(ko.f - tf) <= 1e-12 * std::abs(tf));
  const double gs = max_abs(tg), hs = max_abs(tH), es = max_abs(tHe);
  for (int j = 0; j < N; ++j) CHECK(std::abs(ko.g[j] - tg[j]) <= 1e-12 * gs);
  for (int k = 0; k < N * N; ++k) CHECK(std::abs(ko.H[k] - tH[k]) <= 1e-12 * hs);
  for (int k = 0; k < N * N; ++k) CHECK(std::abs(ke.H[k] - tHe[k]) <= 1e-12 * es);
  // the kernels agree on what they share
  lo::optimizer_vector xv("x", N), g("g", N);
  for (int j = 0; j < N; ++j) xv(j) = x[j];
  dense_objective plain(m, {});
  CHECK(plain.value(xv) == doctest::Approx(ko.f).epsilon(1e-13));
  const double fg = plain.value_gradient(xv, g);
  CHECK(fg == doctest::Approx(ko.f).epsilon(1e-13));
  for (int j = 0; j < N; ++j) CHECK(std::abs(g(j) - ko.g[j]) <= 1e-13 * gs);
  CHECK(ke.f == doctest::Approx(ko.f).epsilon(1e-13));

  // central differences: of the value against the gradient, of the gradient against the observed information
  for (int j = 0; j < N; ++j) {
    const double h = 1e-5 * (1.0 + std::abs(x[j]));
    for (int k = 0; k < N; ++k) xv(k) = x[k];
    xv(j) = x[j] + h;
    const double fp = plain.value_gradient(xv, g);
    std::vector<double> gp(g.data(), g.data() + N);
    xv(j) = x[j] - h;
    const double fm = plain.value_gradient(xv, g);
    CHECK(std::abs((fp - fm) / (2.0 * h) - ko.g[j]) <= 1e-7 * (gs + 1.0));
    for (int k = 0; k < N; ++k) CHECK(std::abs((gp[k] - g(k)) / (2.0 * h) - ko.H[k + N * j]) <= 1e-7 * (hs + 1.0));
  }
}

}  // namespace

TEST_CASE("engine row kernels: Gaussian with a group effect, against the twin and differences") {
  const auto s = gauss_spec(300, 10, 11);
  std::vector<double> x = {0.8, 0.3, -0.1, 0.5};
  for (int g = 0; g < 10; ++g) x.push_back(0.1 * (g % 4) - 0.15);
  check_model<ld::gaussian1_distrib>(s, GAUSS, x);
}

TEST_CASE("engine row kernels: gamma with a logistic curve fed by a link, against the twin and differences") {
  const auto s = gamma_spec(300, 10, 12);
  std::vector<double> x = {2.7, 4.6, std::log(1.2), -1.7};
  for (int g = 0; g < 10; ++g) x.push_back(0.05 * (g % 5) - 0.1);
  check_model<ld::gamma1_distrib>(s, GAMMA, x);
}

TEST_CASE("engine row kernels: Student t, separable, against the twin and differences") {
  const auto s = student_spec(300, 13);
  check_model<ld::student_t1_distrib>(s, STUDENT, {0.9, 0.4, -0.2, 0.6, std::log(4.0)});
}
