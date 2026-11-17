// The improper (intrinsic) class of mvstudent_t2_distrib over weighted_sum_psdmat
// (host only, as the structure), note/distrib_multivariate.md section 17:
//  - against the mpmath references (tools/reference/distrib_mvstudent_t_improper.py):
//    blocks (A, K) of the log-density, A + K <= 6, the expected information in
//    the rest (theta, nu) to order 4 and its (mu, mu) block, normwise in ulps of
//    max(max_entries |ref|, 1) (ulp1N), one row per reference point;
//  - flat along the null space: value and blocks unchanged by a null-space shift;
//  - one matrix: the theta-theta information equals nu rho/(2 (nu + rho + 2)) at
//    small nu, for the improper t2 and the degenerate t1;
//  - constrained draws (zero null component, covariance nu/(nu - 2) Q^+), the
//    constrained mean and covariance (NaN outside their existence), and the
//    constrained mean of the improper mvgaussian2.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/distrib/mvstudent_t1.hpp>
#include <laplace/distrib/mvstudent_t2.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

namespace mvstudent_t_improper_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using ws_t = lp::weighted_sum_psdmat;
using host_matrix = ws_t::host_matrix;
using rows_t = std::vector<std::vector<double>>;
using t2_t = ld::mvstudent_t2_distrib<ws_t>;

static_assert(ld::multivariate_distrib<t2_t>);
static_assert(t2_t::normalization == ld::distrib_normalization::improper);

rows_t diff_matrix(int d, int k) {
  rows_t r(d, std::vector<double>(d, 0.0));
  for (int i = 0; i < d; ++i) r[i][i] = 1.0;
  for (int t = 0; t < k; ++t) {
    rows_t n(r.size() - 1, std::vector<double>(d));
    for (std::size_t i = 0; i + 1 < r.size(); ++i)
      for (int j = 0; j < d; ++j) n[i][j] = r[i + 1][j] - r[i][j];
    r = n;
  }
  return r;
}
host_matrix gram(const rows_t& D, std::vector<double> w = {}) {
  const int d = int(D[0].size());
  if (w.empty()) w.assign(D.size(), 1.0);
  host_matrix m("S", d, d);
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      for (std::size_t k = 0; k < D.size(); ++k) m(i, j) += w[k] * D[k][i] * D[k][j];
  return m;
}
// the structures 0..3 of tools/reference/distrib_mvgaussian_rank_deficient.py
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
  int structure, d, A, K, point_no;
  std::vector<double> theta, y, mu;
  std::vector<int> idx;
  int ab[2] = {0, 0};
  std::vector<int> pos;
  std::vector<double> hi, lo;
  std::string point;
};

// a block groups its entries; an information row of the rest is a case of its own
std::vector<Case> read_cases(const std::string& path, int& n_points) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::string, int> where, points;
  std::vector<Case> cases;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) f.push_back(x);
    REQUIRE(f.size() == 11);
    const int A = std::atoi(f[6].c_str()), entry = std::atoi(f[8].c_str());
    const std::string key = A == -1 ? line : line.substr(0, line.size() - f[8].size() - f[9].size() - f[10].size() - 3);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(cases.size())).first;
      Case c;
      c.point = f[0] + "," + f[1] + "," + f[2] + "," + f[3] + "," + f[4] + "," + f[5];
      c.point_no = points.emplace(c.point, int(points.size())).first->second;
      c.structure = std::atoi(f[1].c_str()), c.d = std::atoi(f[2].c_str());
      c.theta = hex_list(f[3]), c.y = hex_list(f[4]), c.mu = hex_list(f[5]);
      c.A = A, c.idx = int_list(f[7]), c.K = int(c.idx.size());
      if (A == -1) {
        int a = 0;
        while ((a + 1) * (a + 2) / 2 <= entry) ++a;
        c.ab[0] = a, c.ab[1] = entry - a * (a + 1) / 2;
      }
      cases.push_back(c);
    }
    Case& c = cases[it->second];
    c.pos.push_back(A == -1 ? 0 : entry);
    c.hi.push_back(std::strtod(f[9].c_str(), nullptr));
    c.lo.push_back(std::strtod(f[10].c_str(), nullptr));
  }
  n_points = int(points.size());
  return cases;
}

template <int A>
bool block(const t2_t& g, const t2_t::state_type& st, int K, const Case& c, const double* y, double* out, double* w) {
  const int* idx = c.idx.data();
  const double* mu = c.mu.data();
  auto one = [&](auto Kc) {
    constexpr int KK = decltype(Kc)::value;
    if constexpr (A + KK > 6 || (A < 0 && KK > 4)) {
      return false;
    } else if constexpr (A == -2) {
      return g.template expected_info_dk<KK>(st, 0, 0, c.ab, nullptr, idx, out, w);
    } else if constexpr (A == -1) {
      return g.template expected_info_dk<KK>(st, ld::theta_slot, ld::theta_slot, c.ab, nullptr, idx, out, w);
    } else {
      return g.template logpdf_dk<KK, A>(st, y, mu, idx, out, w);
    }
  };
  switch (K) {
    case 0: return one(std::integral_constant<int, 0>{});
    case 1: return one(std::integral_constant<int, 1>{});
    case 2: return one(std::integral_constant<int, 2>{});
    case 3: return one(std::integral_constant<int, 3>{});
    case 4: return one(std::integral_constant<int, 4>{});
    case 5: return one(std::integral_constant<int, 5>{});
    default: return one(std::integral_constant<int, 6>{});
  }
}
void eval(const t2_t& g, const t2_t::state_type& st, const Case& c, const double* y, double* out, double* ws) {
  switch (c.A) {
    case -2: block<-2>(g, st, c.K, c, y, out, ws); break;
    case -1: block<-1>(g, st, c.K, c, y, out, ws); break;
    case 0: block<0>(g, st, c.K, c, y, out, ws); break;
    case 1: block<1>(g, st, c.K, c, y, out, ws); break;
    case 2: block<2>(g, st, c.K, c, y, out, ws); break;
    case 3: block<3>(g, st, c.K, c, y, out, ws); break;
    case 4: block<4>(g, st, c.K, c, y, out, ws); break;
    case 5: block<5>(g, st, c.K, c, y, out, ws); break;
    default: block<6>(g, st, c.K, c, y, out, ws); break;
  }
}

constexpr int n_rows = 8;  // the reference points, in the order of the script

void mpmath_case() {
  int np = 0;
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t_improper.csv", np);
  REQUIRE(np == n_rows);
  std::map<int, ws_t> structs;
  for (int s = 0; s < 4; ++s) structs.emplace(s, make_structure(s));
  std::map<std::string, double> ts;  // the scale of a block: its largest reference entry
  for (const Case& c : cases) {
    double& v = ts[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)];
    for (double h : c.hi) v = std::fmax(v, std::fabs(h));
  }
  double worst[n_rows][9][7] = {}, shift = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (const Case& c : cases) {
    const ws_t& s = structs.at(c.structure);
    const t2_t g(s);
    std::vector<double> w(g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()));
    auto st = g.theta_state(c.theta.data(), w.data());
    double* ws = w.data() + g.state_size();
    const int n_out = laplace::distrib::binomial_coef(c.d + 5, 6) + 64;
    std::vector<double> out(n_out, 0.0), out2(n_out, 0.0);
    eval(g, st, c, c.y.data(), out.data(), ws);
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs((out[c.pos[r]] - c.hi[r]) - c.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double scale = std::fmax(std::fmax(1.0, ts[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)]),
                                   c.A == -1 ? ts[c.point + ",-1,0"] : 0.0);
    double& wv = worst[c.point_no][c.A + 2][c.K];
    wv = std::fmax(wv, err / (scale * eps));
    // flat along the null space: y + 0.75 (1, ..., 1) (the constant vector is in the
    // null space of every difference penalty) gives the same block
    if (c.A >= 0) {
      std::vector<double> y2 = c.y;
      for (double& v : y2) v += 0.75;
      eval(g, st, c, y2.data(), out2.data(), ws);
      for (int p : c.pos) shift = std::fmax(shift, std::fabs(out2[p] - out[p]) / (scale * eps));
    }
  }
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  const char* rname[n_rows] = {"m=1 d=4",       "m=2 d=5",         "m=3 d=6",       "m=2 d=5 two D2",
                               "m=1 nu=0.05",   "m=1 nu=1e6",      "spread 9",      "large residual"};
  // measured (gcc and the host code of the CUDA build), two digits up, at least 1;
  // the spread point (theta 9 apart) is the worst, as in the degenerate t
  const double bound[n_rows][9][7] = {
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1.5, 1, 1, 1, 1}, {1.1, 1.3, 1, 1, 1, 1, 0}, {2.3, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=1 d=4
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1.9, 1, 1, 1, 1}, {1.5, 2, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=2 d=5
      {{1, 1, 1, 0, 0, 0, 0}, {2, 1, 1.5, 1, 3.1, 0, 0}, {1, 1, 1.9, 1, 1, 1.6, 2.5}, {1, 3.5, 1, 1, 1, 1, 0}, {2, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=3 d=6
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1.7, 3.1, 1, 1, 1, 1}, {1, 4.3, 1, 1, 1, 1, 0}, {3.9, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=2 d=5 two D2
      {{1, 1, 1, 0, 0, 0, 0}, {1.5, 1.8, 1.7, 1.4, 2, 0, 0}, {1, 1, 1, 1.2, 1.5, 1.5, 2.2}, {1, 1, 1, 1, 1, 1, 0}, {1.2, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=1 nu=0.05
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // m=1 nu=1e6
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 2.5, 5.3, 1, 1, 1, 1}, {1, 9.2, 1.2, 1, 1.2, 1, 0}, {1.9, 2.5, 2.4, 1, 2.1, 0, 0}, {1, 1.8, 1.7, 1.6, 0, 0, 0}, {2.4, 9.7, 2.2, 0, 0, 0, 0}, {14, 24, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // spread 9
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 2.1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // large residual
  };
  for (int r = 0; r < n_rows; ++r)
    for (int a = 0; a < 9; ++a) {
      if (print) {
        std::printf("mvstudent_t2 improper %-15s A=%2d", rname[r], a - 2);
        for (int K = 0; K <= 6; ++K) std::printf(" %9.3g", worst[r][a][K]);
        std::printf("\n");
      }
      for (int K = 0; K <= 6; ++K)
        CHECK_MESSAGE(worst[r][a][K] <= bound[r][a][K],
                      std::string(rname[r]) << " A " << a - 2 << " K " << K << ": " << worst[r][a][K] << " ulp1N");
    }
  if (print) std::printf("mvstudent_t2 improper null-space shift: %.3g ulp1N\n", shift);
  // the shifted y rounds in its last bits, so r moves by about eps |y| (measured 65.3, gcc)
  CHECK_MESSAGE(shift <= 96.0, "null-space shift " << shift << " ulp1N");
}

// one matrix: I_theta,theta = nu rho/(2 (nu + rho + 2)) exactly (t_a = -+ rho)
template <class D>
double one_matrix_info(const D& g, double th, double nu) {
  const double theta[2] = {th, nu};
  std::vector<double> w(g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()));
  auto st = g.theta_state(theta, w.data());
  const int ab[2] = {0, 0};
  double out = 0.0;
  g.template expected_info_dk<0>(st, ld::theta_slot, ld::theta_slot, ab, nullptr, nullptr, &out, w.data() + g.state_size());
  return out;
}
void one_matrix_case() {
  const ws_t s = make_structure(0);  // rho = 2
  const double rho = 2.0;
  double worst = 0.0;
  for (double nu : {1e-3, 0.01, 0.05, 1.0, 30.0}) {
    const double exact = nu * rho / (2.0 * (nu + rho + 2.0));
    for (double th : {-2.0, 0.4}) {
      const double e2 = std::fabs(one_matrix_info(t2_t(s), th, nu) / exact - 1.0);
      const double e1 = std::fabs(one_matrix_info(ld::mvstudent_t1_distrib<ws_t>(s), th, nu) / exact - 1.0);
      worst = std::fmax(worst, std::fmax(e1, e2) / std::ldexp(1.0, -52));
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("one matrix theta-theta information: %.3g ulp\n", worst);
  CHECK_MESSAGE(worst <= 8.0, "one-matrix information " << worst << " ulp");
}

void constrained_case() {
  const ws_t s = make_structure(1);  // d = 5, rho = 4, null space the constant vector
  const t2_t g(s);
  const int d = 5, np = d * (d + 1) / 2;
  const double nu = 7.0;
  const std::vector<double> th = {0.3, -0.2, nu}, mu = {0.5, -1.0, 0.0, 2.0, 0.25};
  std::vector<double> w(g.state_size() + g.scratch_size());
  auto cst = g.constrained_state(th.data(), w.data());
  std::vector<double> cov(np), y(d), tmp(d), m(d, 0.0), c(np, 0.0);
  g.constrained_covariance(cst, cov.data());
  const int n = 200000;
  double nullc = 0.0;
  for (int u = 0; u < n; ++u) {
    g.constrained_rng(cst, mu.data(), 11, std::uint64_t(u), y.data(), tmp.data());
    double sum = 0.0, nr = 0.0;
    for (int k = 0; k < d; ++k) sum += y[k] - mu[k], nr += std::fabs(y[k] - mu[k]);
    nullc = std::fmax(nullc, std::fabs(sum) / (std::sqrt(double(d)) * nr));
    for (int i = 0; i < d; ++i) {
      m[i] += y[i] - mu[i];
      for (int j = 0; j <= i; ++j) c[i * (i + 1) / 2 + j] += (y[i] - mu[i]) * (y[j] - mu[j]);
    }
  }
  double em = 0.0, ec = 0.0, cmax = 0.0;
  for (int i = 0; i < d; ++i) em = std::fmax(em, std::fabs(m[i] / n) / std::sqrt(cov[i * (i + 1) / 2 + i]));
  for (int k = 0; k < np; ++k) ec = std::fmax(ec, std::fabs(c[k] / n - cov[k])), cmax = std::fmax(cmax, std::fabs(cov[k]));
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("constrained draws: null %.3g, mean %.3g sd, covariance %.3g\n", nullc, em, ec / cmax);
  CHECK_MESSAGE(nullc <= 64.0 * std::ldexp(1.0, -52), "null component " << nullc);
  CHECK_MESSAGE(em <= 0.01, "mean " << em);
  CHECK_MESSAGE(ec <= 0.03 * cmax, "covariance " << ec / cmax);
  // the covariance is nu/(nu - 2) times the improper Gaussian's Q^+
  const ld::mvgaussian2_distrib<ws_t> gg(s);
  std::vector<double> wg(gg.state_size() + gg.scratch_size()), qp(np), gm(d);
  gg.constrained_covariance(gg.constrained_state(th.data(), wg.data()), qp.data());
  for (int k = 0; k < np; ++k) CHECK(cov[k] == doctest::Approx(nu / (nu - 2.0) * qp[k]).epsilon(1e-14));
  gg.constrained_mean(mu.data(), gm.data());
  for (int k = 0; k < d; ++k) CHECK(gm[k] == mu[k]);
  // existence: mean for nu > 1, covariance for nu > 2
  const std::vector<double> th1 = {0.3, -0.2, 1.0}, th2 = {0.3, -0.2, 2.0};
  g.constrained_mean(th1.data(), mu.data(), y.data());
  CHECK(std::isnan(y[0]));
  g.constrained_mean(th2.data(), mu.data(), y.data());
  CHECK(y[3] == mu[3]);
  g.constrained_covariance(g.constrained_state(th2.data(), w.data()), cov.data());
  CHECK(std::isnan(cov[0]));
}

}  // namespace mvstudent_t_improper_test

TEST_CASE("mvstudent_t2 improper over weighted_sum_psdmat: against mpmath, flat null space") {
  mvstudent_t_improper_test::mpmath_case();
}
TEST_CASE("mvstudent_t2 improper, mvstudent_t1 degenerate: one-matrix theta-theta information") {
  mvstudent_t_improper_test::one_matrix_case();
}
TEST_CASE("mvstudent_t2 improper: constrained draws, mean and covariance; mvgaussian2 improper constrained_mean") {
  mvstudent_t_improper_test::constrained_case();
}
