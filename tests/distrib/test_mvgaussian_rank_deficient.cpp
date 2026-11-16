// The rank-deficient classes of mvgaussian1_distrib (degenerate) and
// mvgaussian2_distrib (improper) over weighted_sum_psdmat, on the host (the
// structure is host only):
//  - against the mpmath references
//    (tools/reference/distrib_mvgaussian_rank_deficient.py): blocks (A, K) of
//    the log-density, A <= 2, A + K <= 6, and the expected information in
//    theta to order 4, with and without the theta table; error normwise in ulps
//    of max(max_entries |ref|, 1) (ulp1N);
//  - -inf off the support of the degenerate law, finite within the tolerance;
//  - the draws: no null-space component, empirical covariance near Sigma or Q^+;
//    Q^+ a pseudo-inverse (Q Q^+ Q = Q, Q^+ U0 = 0);
//  - start_value recovers theta from Sigma(theta) or Q(theta)^+.
// Set LAPLACE_PRINT_ERRORS=1 to print the tables.
#include <doctest/doctest.h>

#include <laplace/distrib/mvgaussian1.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/pdmat/reduced_rank.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace mvgaussian_rank_deficient_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using ws_t = lp::weighted_sum_psdmat;
using host_matrix = ws_t::host_matrix;
using deg_t = ld::mvgaussian1_distrib<ws_t>;
using imp_t = ld::mvgaussian2_distrib<ws_t>;

static_assert(ld::multivariate_distrib<deg_t> && ld::multivariate_distrib<imp_t>);
static_assert(deg_t::normalization == ld::distrib_normalization::degenerate);
static_assert(imp_t::normalization == ld::distrib_normalization::improper);
// reduced_rank moves its null space: neither class exists over it
static_assert(!ld::constant_null_space_psdmat<lp::reduced_rank_psdmat>);

// k-th differences, (d - k) x d
std::vector<std::vector<double>> diff_matrix(int d, int k) {
  std::vector<std::vector<double>> r(d, std::vector<double>(d, 0.0));
  for (int i = 0; i < d; ++i) r[i][i] = 1.0;
  for (int s = 0; s < k; ++s) {
    std::vector<std::vector<double>> n(r.size() - 1, std::vector<double>(d));
    for (std::size_t i = 0; i + 1 < r.size(); ++i)
      for (int j = 0; j < d; ++j) n[i][j] = r[i + 1][j] - r[i][j];
    r = n;
  }
  return r;
}
host_matrix gram(const std::vector<std::vector<double>>& D, std::vector<double> w = {}) {
  const int d = int(D[0].size());
  if (w.empty()) w.assign(D.size(), 1.0);
  host_matrix m("S", d, d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      for (std::size_t k = 0; k < D.size(); ++k) m(i, j) += w[k] * D[k][i] * D[k][j];
  return m;
}
// the structures of the reference script
ws_t make_structure(int s) {
  switch (s) {
    case 0: return ws_t(std::vector<host_matrix>{gram(diff_matrix(4, 2))});
    case 1: return ws_t(std::vector<host_matrix>{gram(diff_matrix(5, 1)), gram(diff_matrix(5, 2))});
    case 2: return ws_t(std::vector<host_matrix>{gram(diff_matrix(6, 1)), gram(diff_matrix(6, 2)), gram(diff_matrix(6, 3))});
    default: {
      const auto D2 = diff_matrix(5, 2);
      return ws_t(std::vector<host_matrix>{gram(D2), gram(D2, {1.0, 2.0, 3.0})});
    }
  }
}

std::vector<double> hex_list(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, ';')) v.push_back(std::strtod(f.c_str(), nullptr));
  return v;
}
std::vector<int> int_list(const std::string& s) {
  std::vector<int> v;
  if (s == "-") return v;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, ';')) v.push_back(std::atoi(f.c_str()));
  return v;
}

struct Case {
  int param, structure, d, A, K;
  std::vector<double> theta, y, mu;
  std::vector<int> idx;
  int ab[2] = {0, 0};
  std::vector<int> pos;
  std::vector<double> hi;
};

std::vector<Case> read_cases(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::string, int> where;
  std::vector<Case> cases;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) f.push_back(x);
    REQUIRE(f.size() == 11);
    const int A = std::atoi(f[6].c_str()), entry = std::atoi(f[8].c_str());
    const std::string key = A < 0 ? line : line.substr(0, line.size() - f[8].size() - f[9].size() - f[10].size() - 3);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(cases.size())).first;
      Case c;
      c.param = std::atoi(f[0].c_str()), c.structure = std::atoi(f[1].c_str()), c.d = std::atoi(f[2].c_str());
      c.theta = hex_list(f[3]), c.y = hex_list(f[4]), c.mu = hex_list(f[5]);
      c.A = A, c.idx = int_list(f[7]), c.K = int(c.idx.size());
      if (A < 0) {
        int a = 0;
        while ((a + 1) * (a + 2) / 2 <= entry) ++a;
        c.ab[0] = a, c.ab[1] = entry - a * (a + 1) / 2;
      }
      cases.push_back(c);
    }
    Case& c = cases[it->second];
    c.pos.push_back(A < 0 ? 0 : entry);
    c.hi.push_back(std::strtod(f[9].c_str(), nullptr));
  }
  return cases;
}

template <class D, int A, int K>
bool block_ak(const D& g, const typename D::state_type& st, const Case& c, double* out, double* w) {
  if constexpr (A < 0)
    return g.template expected_info_dk<K>(st, ld::distrib_mv_info_block::rest_rest, c.ab, c.idx.data(), out, w);
  else return g.template logpdf_dmu_dk<A, K>(st, c.y.data(), c.mu.data(), c.idx.data(), out, w);
}
template <class D, int A>
bool block_a(const D& g, const typename D::state_type& st, const Case& c, double* out, double* w) {
  switch (c.K) {
    case 0: return block_ak<D, A, 0>(g, st, c, out, w);
    case 1: return block_ak<D, A, 1>(g, st, c, out, w);
    case 2: return block_ak<D, A, 2>(g, st, c, out, w);
    case 3: return block_ak<D, A, 3>(g, st, c, out, w);
    case 4: return block_ak<D, A, 4>(g, st, c, out, w);
    case 5:
      if constexpr (A >= 0 && A <= 1) return block_ak<D, A, 5>(g, st, c, out, w);
      break;
    case 6:
      if constexpr (A == 0) return block_ak<D, A, 6>(g, st, c, out, w);
      break;
  }
  return false;
}

// the block of the case; table: through a theta table of order K
template <class D>
std::vector<double> evaluate(const D& g, const Case& c, bool table) {
  const int np = lp::packed_size(c.d);
  const int sc = std::max(g.scratch_size(), g.info_scratch_size());
  std::vector<double> w(g.state_size() + sc + g.theta_table_size(std::max(c.K, 1))), out(np, 0.0);
  typename D::state_type st;
  if constexpr (D::normalization == ld::distrib_normalization::improper) st = g.constrained_state(c.theta.data(), w.data());
  else st = g.theta_state(c.theta.data(), w.data());
  double* ws = w.data() + g.state_size();
  if (table && c.K >= 1) {
    double* tab = ws + sc;
    for (int e = 0; e < g.theta_table_entries(c.K); ++e) g.fill_theta_table_entry(st, c.K, e, tab, ws);
    st = g.with_theta_table(st, c.K, tab);
  }
  switch (c.A) {
    case -1: block_a<D, -1>(g, st, c, out.data(), ws); break;
    case 0: block_a<D, 0>(g, st, c, out.data(), ws); break;
    case 1: block_a<D, 1>(g, st, c, out.data(), ws); break;
    default: block_a<D, 2>(g, st, c, out.data(), ws); break;
  }
  return out;
}

double ulp1n_error(const Case& c, const std::vector<double>& got) {
  double scale = 1.0, err = 0.0;
  for (double h : c.hi) scale = std::max(scale, std::abs(h));
  for (std::size_t k = 0; k < c.hi.size(); ++k) err = std::max(err, std::abs(got[c.pos[k]] - c.hi[k]));
  int e;
  std::frexp(scale, &e);
  return err / std::ldexp(1.0, e - 53);
}

// worst error of the references by (param, A, K); bounds: param x A (-1..2) x K
void check_references(bool table) {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvgaussian_rank_deficient.csv");
  REQUIRE(cases.size() > 100);
  std::map<int, ws_t> structs;
  for (int s = 0; s < 4; ++s) structs.emplace(s, make_structure(s));
  // measured worst (ulp1N), gcc 14.2; bounds 1.5 times, at least 4. By A = -1
  // (information), 0, 1, 2. The degenerate blocks go through the dense
  // (Sigma + cP)^-1 and lose 24-500 times cond(Sigma on its range) eps where the
  // lambda_j are spread (accepted 2026-11-16): at the spread points (theta
  // components about 9 apart) by structure, worst 4.86e6 (1), 3.28e12 (2),
  // 0.54 (3); elsewhere 117.
  static const double bound[2][4] = {{4.0, 176.0, 176.0, 176.0}, {4.0, 6.0, 4.0, 4.0}};
  static const double spread_bound[4] = {176.0, 7.3e6, 4.9e12, 176.0};
  double worst[2][4][7] = {};
  std::map<std::string, double> by_point;
  for (const Case& c : cases) {
    const ws_t& s = structs.at(c.structure);
    REQUIRE(s.dim == c.d);
    const auto got = c.param == 1 ? evaluate(deg_t(s), c, table) : evaluate(imp_t(s), c, table);
    const double e = ulp1n_error(c, got);
    double& wv = worst[c.param - 1][c.A + 1][c.K];
    wv = std::max(wv, e);
    char key[96];
    std::snprintf(key, sizeof key, "param %d structure %d theta0 %+.2f %s", c.param, c.structure, c.theta[0],
                  c.A < 0 ? "info" : "blocks");
    by_point[key] = std::max(by_point[key], e);
    const auto [lo, hi] = std::minmax_element(c.theta.begin(), c.theta.end());
    const bool spread = *hi - *lo > 5.0;
    const double b = c.param == 1 && c.A >= 0 && spread ? spread_bound[c.structure] : bound[c.param - 1][c.A + 1];
    CHECK_MESSAGE(e <= b, "param ", c.param, " structure ", c.structure, " A ", c.A,
                  " K ", c.K, " error ", e);
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) {
    std::printf("mvgaussian rank-deficient (%s): worst ulp1N by K = 0..6\n", table ? "table" : "direct");
    const char* an[] = {"info", "A=0", "A=1", "A=2"};
    for (int p = 0; p < 2; ++p)
      for (int a = 0; a < 4; ++a) {
        std::printf("  %s %-4s", p == 0 ? "degenerate" : "improper  ", an[a]);
        for (int k = 0; k < 7; ++k) std::printf(" %9.3g", worst[p][a][k]);
        std::printf("\n");
      }
    for (const auto& [k, v] : by_point) std::printf("  %s: %.3g\n", k.c_str(), v);
  }
}

void check_support() {
  const ws_t s = make_structure(1);
  const deg_t g(s);
  const double th[2] = {0.3, -0.8};
  std::vector<double> w(g.state_size() + g.scratch_size());
  const auto st = g.theta_state(th, w.data());
  const auto& B = s.null_space_basis();  // the constants, d = 5
  double mu[5] = {0.1, -0.2, 0.3, 0.0, 0.5}, y[5], r[5] = {1.0, -2.0, 0.5, 0.25, 0.25};
  for (int i = 0; i < 5; ++i) y[i] = mu[i] + r[i];  // sum r = 0: in the range
  const double on = g.logpdf(st, y, mu);
  CHECK(std::isfinite(on));
  double y2[5], y3[5];
  for (int i = 0; i < 5; ++i) y2[i] = y[i] + 1e-6 * B(i, 0), y3[i] = y[i] + 1e-15 * B(i, 0);
  CHECK(g.logpdf(st, y2, mu) == -INFINITY);
  CHECK(std::isfinite(g.logpdf(st, y3, mu)));
  double out[1];
  CHECK(g.logpdf_dmu_dk<0, 0>(st, y2, mu, nullptr, out, w.data() + g.state_size()));
  CHECK(out[0] == -INFINITY);
}

// empirical mean and covariance of n draws against mu and C (packed): each
// entry within 6 standard errors; the null-space component below 1e-12
template <class Draw>
void check_draws(const ws_t& s, const double* mu, const std::vector<double>& C, Draw draw) {
  const int d = s.dim, np = lp::packed_size(d), n = 40000;
  const auto& B = s.null_space_basis();
  std::vector<double> m(d, 0.0), S(np, 0.0), y(d), tmp(d), ys(std::size_t(n) * d);
  double off = 0.0;
  for (int u = 0; u < n; ++u) {
    draw(std::uint64_t(u), y.data(), tmp.data());
    for (int k = 0; k < d - s.rank(); ++k) {
      double v = 0.0;
      for (int i = 0; i < d; ++i) v += B(i, k) * (y[i] - mu[i]);
      off = std::max(off, std::abs(v));
    }
    for (int i = 0; i < d; ++i) ys[std::size_t(u) * d + i] = y[i] - mu[i], m[i] += (y[i] - mu[i]) / n;
  }
  for (int u = 0; u < n; ++u)
    for (int i = 0; i < d; ++i)
      for (int j = 0; j <= i; ++j) S[lp::packed_index(i, j)] += ys[std::size_t(u) * d + i] * ys[std::size_t(u) * d + j] / n;
  CHECK(off < 1e-12);
  for (int i = 0; i < d; ++i) CHECK(std::abs(m[i]) < 6.0 * std::sqrt(C[lp::packed_index(i, i)] / n) + 1e-14);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j <= i; ++j) {
      const double cij = C[lp::packed_index(i, j)];
      const double se = std::sqrt((C[lp::packed_index(i, i)] * C[lp::packed_index(j, j)] + cij * cij) / n);
      CHECK_MESSAGE(std::abs(S[lp::packed_index(i, j)] - cij) < 6.0 * se + 1e-14, "entry ", i, ",", j);
    }
}

void check_draws_and_pinv() {
  for (int sidx : {1, 3}) {
    const ws_t s = make_structure(sidx);
    const int d = s.dim, np = lp::packed_size(d);
    std::vector<double> th(s.n_params());
    for (int j = 0; j < s.n_params(); ++j) th[j] = 0.4 - 0.7 * j;
    std::vector<double> mu(d);
    for (int i = 0; i < d; ++i) mu[i] = 0.1 * i - 0.2;
    {
      const deg_t g(s);
      std::vector<double> w(g.state_size() + g.scratch_size()), C(np);
      const auto st = g.theta_state(th.data(), w.data());
      g.covariance(th.data(), C.data(), w.data() + g.state_size());
      check_draws(s, mu.data(), C, [&](std::uint64_t u, double* y, double* tmp) { g.rng(st, mu.data(), 7, u, y, tmp); });
    }
    {
      const imp_t g(s);
      std::vector<double> w(g.state_size() + g.scratch_size()), C(np), Q(np);
      const auto st = g.constrained_state(th.data(), w.data());
      g.constrained_covariance(st, C.data());
      s.matrix(th.data(), Q.data(), nullptr);
      // Q Q^+ Q = Q and Q^+ U0 = 0
      double qmax = 0.0, e1 = 0.0, e2 = 0.0;
      for (double v : Q) qmax = std::max(qmax, std::abs(v));
      std::vector<double> T(d * d);
      for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
          double r = 0.0;
          for (int k = 0; k < d; ++k) r += ld::detail_mvgaussian::sym_at(Q.data(), i, k) * ld::detail_mvgaussian::sym_at(C.data(), k, j);
          T[i * d + j] = r;
        }
      for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) {
          double r = 0.0;
          for (int k = 0; k < d; ++k) r += T[i * d + k] * ld::detail_mvgaussian::sym_at(Q.data(), k, j);
          e1 = std::max(e1, std::abs(r - ld::detail_mvgaussian::sym_at(Q.data(), i, j)));
        }
      const auto& B = s.null_space_basis();
      double cmax = 0.0;
      for (double v : C) cmax = std::max(cmax, std::abs(v));
      for (int k = 0; k < d - s.rank(); ++k)
        for (int i = 0; i < d; ++i) {
          double r = 0.0;
          for (int j = 0; j < d; ++j) r += ld::detail_mvgaussian::sym_at(C.data(), i, j) * B(j, k);
          e2 = std::max(e2, std::abs(r));
        }
      CHECK(e1 <= 1e-12 * qmax);
      CHECK(e2 <= 1e-13 * cmax);
      check_draws(s, mu.data(), C,
                  [&](std::uint64_t u, double* y, double* tmp) { g.constrained_rng(st, mu.data(), 11, u, y, tmp); });
    }
  }
}

void check_start_value() {
  for (int sidx = 0; sidx < 4; ++sidx) {
    const ws_t s = make_structure(sidx);
    const int d = s.dim, np = lp::packed_size(d);
    std::vector<double> th(s.n_params()), got(s.n_params());
    for (int j = 0; j < s.n_params(); ++j) th[j] = 0.5 - 0.6 * j;
    ld::distrib_mv_sample_summary sm;
    sm.count = 100, sm.dim = d, sm.mean.assign(d, 0.0), sm.covariance.assign(np, 0.0);
    {
      const deg_t g(s);
      s.matrix(th.data(), sm.covariance.data(), nullptr);
      g.start_value(sm, got.data());
      for (int j = 0; j < s.n_params(); ++j) CHECK(std::abs(got[j] - th[j]) < 1e-11);
    }
    {
      const imp_t g(s);
      std::vector<double> w(g.state_size() + g.scratch_size());
      const auto st = g.constrained_state(th.data(), w.data());
      g.constrained_covariance(st, sm.covariance.data());
      g.start_value(sm, got.data());
      for (int j = 0; j < s.n_params(); ++j) CHECK(std::abs(got[j] - th[j]) < 1e-10);
    }
  }
}

}  // namespace mvgaussian_rank_deficient_test

using namespace mvgaussian_rank_deficient_test;

TEST_CASE("mvgaussian rank-deficient: blocks and information against mpmath") { check_references(false); }
TEST_CASE("mvgaussian rank-deficient: the same through the theta table") { check_references(true); }
TEST_CASE("mvgaussian rank-deficient: -inf off the support of the degenerate law") { check_support(); }
TEST_CASE("mvgaussian rank-deficient: draws and the pseudo-inverse") { check_draws_and_pinv(); }
TEST_CASE("mvgaussian rank-deficient: start_value") { check_start_value(); }
