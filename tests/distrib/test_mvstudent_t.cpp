// mvstudent_t1_distrib and mvstudent_t2_distrib (include/laplace/distrib/):
//  - against the mpmath references (tools/reference/distrib_mvstudent_t.py), in a
//    Kokkos kernel on the default execution space for the proper class and on the
//    host for the degenerate one (weighted_sum_psdmat is host only): blocks
//    (A, K) of the log-density, A + K <= 6, the expected information in the rest
//    (theta, nu) to order 4 and its (mu, mu) block; the error of a block is
//    normwise, in ulps of max(max_entries |ref|, 1) (ulp1N);
//  - mvstudent_t1 over the diagonal structure at theta against mvstudent_t2 at
//    -theta (the same law, nu unchanged): every block agrees up to (-1)^K_theta;
//  - the theta table gives the blocks to the bit;
//  - draws: the mean and the covariance nu/(nu - 2) Sigma (and Q^-1), and the
//    degenerate draw on mu + col(Sigma).
// Set LAPLACE_PRINT_ERRORS=1 to print the tables.
#include <doctest/doctest.h>

#include <laplace/distrib/mvstudent_t1.hpp>
#include <laplace/distrib/mvstudent_t2.hpp>
#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/diagonal.hpp>
#include <laplace/pdmat/factor_analytic.hpp>
#include <laplace/pdmat/unit_diagonal_scaled.hpp>
#include <laplace/pdmat/unstructured.hpp>
#include <laplace/pdmat/weighted_sum.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
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

// a named namespace: the kernels must sit in functions with external linkage
namespace mvstudent_t_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using scaled_ar1 = lp::unit_diagonal_scaled_pdmat<lp::ar1_pdmat>;
using ws_t = lp::weighted_sum_psdmat;

constexpr int max_d = 4, max_q = 12, max_out = 84;  // C(4 + 6 - 1, 6)
const char* sname[] = {"diagonal", "ar1", "compound_symmetry", "log_cholesky", "scaled_ar1", "factor_analytic"};

// block A (-2: the (mu, mu) information, -1: the information in the rest at ab)
template <class D, int A, int K>
KOKKOS_INLINE_FUNCTION bool block_ak(const D& g, const typename D::state_type& st, const double* y, const double* mu,
                                     const int* idx, const int* ab, double* out, double* w) {
  if constexpr (A == -2) return g.template expected_info_dk<K>(st, 0, 0, ab, nullptr, idx, out, w);
  else if constexpr (A == -1) return g.template expected_info_dk<K>(st, ld::theta_slot, ld::theta_slot, ab, nullptr, idx, out, w);
  else return g.template logpdf_dk<K, A>(st, y, mu, idx, out, w);
}
template <class D, int A>
KOKKOS_INLINE_FUNCTION bool block_a(const D& g, const typename D::state_type& st, int K, const double* y,
                                    const double* mu, const int* idx, const int* ab, double* out, double* w) {
  switch (K) {
    case 0: return block_ak<D, A, 0>(g, st, y, mu, idx, ab, out, w);
    case 1: if constexpr (A <= 5) return block_ak<D, A, 1>(g, st, y, mu, idx, ab, out, w); break;
    case 2: if constexpr (A <= 4) return block_ak<D, A, 2>(g, st, y, mu, idx, ab, out, w); break;
    case 3: if constexpr (A <= 3) return block_ak<D, A, 3>(g, st, y, mu, idx, ab, out, w); break;
    case 4: if constexpr (A <= 2) return block_ak<D, A, 4>(g, st, y, mu, idx, ab, out, w); break;
    case 5: if constexpr (A >= 0 && A <= 1) return block_ak<D, A, 5>(g, st, y, mu, idx, ab, out, w); break;
    case 6: if constexpr (A == 0) return block_ak<D, A, 6>(g, st, y, mu, idx, ab, out, w); break;
  }
  return false;
}
template <class D>
KOKKOS_INLINE_FUNCTION void eval_d(const D& g, int A, int K, const double* th, const double* y, const double* mu,
                                   const int* idx, const int* ab, double* out, double* w, bool table) {
  for (int k = 0; k < max_out; ++k) out[k] = 0.0;
  auto st = g.theta_state(th, w);
  double* ws = w + g.state_size();
  if (table) {
    // the theta table of order 3 after the scratch
    const int sc = g.scratch_size() > g.info_scratch_size() ? g.scratch_size() : g.info_scratch_size();
    double* tab = ws + sc;
    for (int e = 0; e < g.theta_table_entries(3); ++e) g.fill_theta_table_entry(st, 3, e, tab, ws);
    st = g.with_theta_table(st, 3, tab);
  }
  switch (A) {
    case -2: block_a<D, -2>(g, st, K, y, mu, idx, ab, out, ws); break;
    case -1: block_a<D, -1>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 0: block_a<D, 0>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 1: block_a<D, 1>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 2: block_a<D, 2>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 3: block_a<D, 3>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 4: block_a<D, 4>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 5: block_a<D, 5>(g, st, K, y, mu, idx, ab, out, ws); break;
    default: block_a<D, 6>(g, st, K, y, mu, idx, ab, out, ws); break;
  }
}
template <class S>
KOKKOS_INLINE_FUNCTION void eval_s(const S& s, int param, int A, int K, const double* th, const double* y,
                                   const double* mu, const int* idx, const int* ab, double* out, double* w, bool table) {
  if (param == 1) eval_d(ld::mvstudent_t1_distrib<S>(s), A, K, th, y, mu, idx, ab, out, w, table);
  else eval_d(ld::mvstudent_t2_distrib<S>(s), A, K, th, y, mu, idx, ab, out, w, table);
}
KOKKOS_INLINE_FUNCTION void eval_any(int structure, int d, int param, int A, int K, const double* th, const double* y,
                                     const double* mu, const int* idx, const int* ab, double* out, double* w,
                                     bool table = false) {
  switch (structure) {
    case 0: eval_s(lp::diagonal_pdmat(d), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    case 1: eval_s(lp::ar1_pdmat(d), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    case 2: eval_s(lp::compound_symmetry_pdmat(d), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    case 3: eval_s(log_chol(d), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    case 4: eval_s(scaled_ar1(lp::ar1_pdmat(d)), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    default: eval_s(lp::factor_analytic_pdmat(d, 1), param, A, K, th, y, mu, idx, ab, out, w, table); break;
  }
}

template <class S>
int work_of(const S& s) {
  const ld::mvstudent_t1_distrib<S> g(s);
  const ld::mvstudent_t2_distrib<S> g2(s);
  const int a = g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()) + g.theta_table_size(3);
  const int b = g2.state_size() + std::max(g2.scratch_size(), g2.info_scratch_size()) + g2.theta_table_size(3);
  return std::max(a, b);
}
int work_any(int structure, int d) {
  switch (structure) {
    case 0: return work_of(lp::diagonal_pdmat(d));
    case 1: return work_of(lp::ar1_pdmat(d));
    case 2: return work_of(lp::compound_symmetry_pdmat(d));
    case 3: return work_of(log_chol(d));
    case 4: return work_of(scaled_ar1(lp::ar1_pdmat(d)));
    default: return work_of(lp::factor_analytic_pdmat(d, 1));
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
  std::vector<double> hi, lo;
  std::string point;
};

// a block groups its entries; an information row of the rest is a case of its own
std::vector<Case> read_cases(const std::string& path, bool degenerate) {
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
    const int param = std::atoi(f[0].c_str());
    if ((param == 3) != degenerate) continue;
    const int A = std::atoi(f[6].c_str()), entry = std::atoi(f[8].c_str());
    const std::string key = A == -1 ? line : line.substr(0, line.size() - f[8].size() - f[9].size() - f[10].size() - 3);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(cases.size())).first;
      Case c;
      c.point = f[0] + "," + f[1] + "," + f[2] + "," + f[3] + "," + f[4] + "," + f[5];
      c.param = param, c.structure = std::atoi(f[1].c_str()), c.d = std::atoi(f[2].c_str());
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
  return cases;
}

struct DeviceCases {
  Kokkos::View<int* [7]> meta;
  Kokkos::View<double* [max_q]> theta;
  Kokkos::View<double* [max_d]> y, mu;
  Kokkos::View<int* [6]> idx;
  int work = 0;
};

DeviceCases upload(const std::vector<Case>& cs) {
  const int n = int(cs.size());
  DeviceCases dc{Kokkos::View<int* [7]>("meta", n), Kokkos::View<double* [max_q]>("theta", n),
                 Kokkos::View<double* [max_d]>("y", n), Kokkos::View<double* [max_d]>("mu", n),
                 Kokkos::View<int* [6]>("idx", n), 0};
  auto hm = Kokkos::create_mirror_view(dc.meta);
  auto ht = Kokkos::create_mirror_view(dc.theta);
  auto hy = Kokkos::create_mirror_view(dc.y);
  auto hu = Kokkos::create_mirror_view(dc.mu);
  auto hi = Kokkos::create_mirror_view(dc.idx);
  for (int g = 0; g < n; ++g) {
    const Case& c = cs[g];
    REQUIRE(c.d <= max_d);
    REQUIRE(int(c.theta.size()) <= max_q);
    hm(g, 0) = c.param, hm(g, 1) = c.structure, hm(g, 2) = c.d, hm(g, 3) = c.A, hm(g, 4) = c.K;
    hm(g, 5) = c.ab[0], hm(g, 6) = c.ab[1];
    for (std::size_t k = 0; k < c.theta.size(); ++k) ht(g, k) = c.theta[k];
    for (int k = 0; k < c.d; ++k) hy(g, k) = c.y[k], hu(g, k) = c.mu[k];
    for (int k = 0; k < c.K; ++k) hi(g, k) = c.idx[k];
    dc.work = std::max(dc.work, work_any(c.structure, c.d));
  }
  Kokkos::deep_copy(dc.meta, hm);
  Kokkos::deep_copy(dc.theta, ht);
  Kokkos::deep_copy(dc.y, hy);
  Kokkos::deep_copy(dc.mu, hu);
  Kokkos::deep_copy(dc.idx, hi);
  return dc;
}

// every case; theta_sign multiplies the structure's parameters (not nu), swap
// exchanges the two parametrizations
void evaluate(const DeviceCases& dc, Kokkos::View<double* [max_out]> val, double theta_sign, bool swap,
              bool table = false) {
  const int n = int(dc.meta.extent(0));
  Kokkos::View<double**, Kokkos::LayoutRight> work("work", n, dc.work);
  auto meta = dc.meta;
  auto theta = dc.theta;
  auto ys = dc.y;
  auto mus = dc.mu;
  auto idxs = dc.idx;
  Kokkos::parallel_for(
      "mvstudent_t_cases", n, KOKKOS_LAMBDA(int g) {
        const int param = swap ? 3 - meta(g, 0) : meta(g, 0), s = meta(g, 1), d = meta(g, 2), A = meta(g, 3),
                  K = meta(g, 4);
        const int ab[2] = {meta(g, 5), meta(g, 6)};
        const int q = s == 0 ? d : s == 1 || s == 2 ? 1 : s == 3 ? d * (d + 1) / 2 : s == 4 ? d + 1 : 2 * d;
        double th[max_q], y[max_d], mu[max_d], a[max_out];
        int idx[6];
        for (int k = 0; k < max_q; ++k) th[k] = k < q ? theta_sign * theta(g, k) : theta(g, k);
        for (int k = 0; k < max_d; ++k) y[k] = ys(g, k), mu[k] = mus(g, k);
        for (int k = 0; k < 6; ++k) idx[k] = idxs(g, k);
        eval_any(s, d, param, A, K, th, y, mu, idx, ab, a, &work(g, 0), table && K <= 3);
        for (int k = 0; k < max_out; ++k) val(g, k) = a[k];
      });
}

// the scale of a block: the largest |ref| over the entries and tuples of the
// same point, A and K, at least 1 (and for the information at least that of
// the information itself)
std::map<std::string, double> scales_of(const std::vector<Case>& cases) {
  std::map<std::string, double> t;
  for (const Case& c : cases) {
    double& v = t[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)];
    for (double h : c.hi) v = std::fmax(v, std::fabs(h));
  }
  return t;
}
double scale_of(std::map<std::string, double>& t, const Case& c) {
  return std::fmax(std::fmax(1.0, t[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)]),
                   c.A == -1 ? t[c.point + ",-1,0"] : 0.0);
}

void mvstudent_t_mpmath_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t.csv", false);
  const int n = int(cases.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> val("val", n);
  evaluate(dc, val, 1.0, false);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  auto ts = scales_of(cases);
  // worst ulp1N by [param - 1][structure][A + 2][K]
  double worst[2][6][9][7] = {};
  int worst_g[2][6][9][7] = {};
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < n; ++g) {
    const Case& c = cases[g];
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs((hv(g, c.pos[r]) - c.hi[r]) - c.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double u = err / (scale_of(ts, c) * eps);
    double& w = worst[c.param - 1][c.structure][c.A + 2][c.K];
    if (!(u <= w)) w = u, worst_g[c.param - 1][c.structure][c.A + 2][c.K] = g;
  }
  // bounds on ulp1N, [param - 1][structure][A + 2][K]: the larger of the errors
  // measured on gcc/OpenMP and CUDA on 2026-11-17 (with the double-double sums of
  // orders >= 3), times 1.5, rounded up to two digits; 0 where no case exists.
  // The largest are the near-singular points: log_cholesky (log L33 = -11.5)
  // and compound_symmetry (theta = 8), where the errors of d_U Sigma^-1 are
  // amplified by the far-tail sums (note/distrib_multivariate.md section 15)
  const double bound[2][6][9][7] = {
   {
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1.2, 1, 2.3, 1, 1, 1, 1}, {1, 2.3, 1, 1, 1, 1, 0}, {1.2, 1.3, 1, 1, 1.3, 0, 0}, {1.2, 1.6, 1, 1, 0, 0, 0}, {1, 1.3, 1.2, 0, 0, 0, 0}, {2.2, 2.1, 0, 0, 0, 0, 0}, {2.3, 0, 0, 0, 0, 0, 0}},  // diagonal
      {{1, 1, 1, 0, 0, 0, 0}, {1.6, 1.8, 3.3, 2.2, 1.5, 0, 0}, {1, 3, 2.3, 1.7, 2.4, 1, 4.1}, {1, 1, 1, 1, 1.9, 1, 0}, {2.5, 1, 1, 2.5, 1.1, 0, 0}, {1, 1, 1.7, 2.1, 0, 0, 0}, {2.2, 1, 1, 0, 0, 0, 0}, {1.6, 1.4, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // ar1
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1.3, 6.3, 1, 2.7, 6.1, 13}, {1.5, 7.8, 2, 6.6, 1.1, 1, 0}, {3, 3.8, 8.9, 1, 70, 0, 0}, {1, 1, 4.8, 16, 0, 0, 0}, {5.7, 7.6, 210, 0, 0, 0, 0}, {2.4, 2.8, 0, 0, 0, 0, 0}, {1.7, 0, 0, 0, 0, 0, 0}},  // compound_symmetry
      {{1.9, 1.9, 2, 0, 0, 0, 0}, {2.8, 1, 3.8, 1, 1, 0, 0}, {20, 26, 7.7, 6.5, 24, 940, 91}, {16, 9.1, 8.6, 50, 38, 30, 0}, {7.9, 13, 13, 31, 1200, 0, 0}, {17, 14, 21, 22, 0, 0, 0}, {20, 31, 21, 0, 0, 0, 0}, {19, 24, 0, 0, 0, 0, 0}, {46, 0, 0, 0, 0, 0, 0}},  // log_cholesky
      {{1, 1, 1, 0, 0, 0, 0}, {1.8, 1.8, 1, 2.2, 1.5, 0, 0}, {1, 2.5, 1.3, 2.4, 1, 2.2, 2.6}, {1.8, 1.8, 1.7, 2.5, 3.9, 5.6, 0}, {1.3, 2.7, 2.8, 4.5, 4, 0, 0}, {1.4, 3.7, 3, 2.5, 0, 0, 0}, {2.1, 5.7, 7.8, 0, 0, 0, 0}, {3, 2.1, 0, 0, 0, 0, 0}, {4.2, 0, 0, 0, 0, 0, 0}},  // scaled_ar1
      {{1.3, 1, 4, 0, 0, 0, 0}, {5.7, 1, 1, 3.5, 30, 0, 0}, {1, 1.6, 2.7, 1.2, 4.7, 7, 7.1}, {1, 4.5, 1, 2, 6.5, 1.7, 0}, {1.9, 2.8, 4.5, 4.8, 9.7, 0, 0}, {1, 3.9, 2.3, 8.8, 0, 0, 0}, {2.9, 1.7, 4.3, 0, 0, 0, 0}, {1.8, 5.3, 0, 0, 0, 0, 0}, {4.2, 0, 0, 0, 0, 0, 0}},  // factor_analytic
   },
   {
      {{1.1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1.7, 1, 1, 1, 1.5}, {1, 1, 1, 1, 1, 1, 0}, {1.8, 1, 1, 1, 1.3, 0, 0}, {1, 1.4, 1, 5, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1.6, 0, 0, 0, 0, 0, 0}},  // diagonal
      {{1, 1, 1.1, 0, 0, 0, 0}, {1.6, 1.8, 1, 3.4, 1.5, 0, 0}, {1.1, 1.3, 1.2, 2.3, 2.4, 3.3, 2.9}, {1.9, 2.2, 6, 17, 7.4, 21, 0}, {2.4, 3.1, 6.8, 7.4, 19, 0, 0}, {1.5, 3.4, 7.3, 15, 0, 0, 0}, {2.8, 5.8, 7.7, 0, 0, 0, 0}, {3, 4.7, 0, 0, 0, 0, 0}, {3.5, 0, 0, 0, 0, 0, 0}},  // ar1
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1.4, 1, 1, 1, 1, 1}, {2.4, 1, 1, 1, 1, 1, 0}, {1.5, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1.4, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1.9, 0, 0, 0, 0, 0, 0}},  // compound_symmetry
      {{2, 2, 1, 0, 0, 0, 0}, {2.8, 4.2, 1.5, 1, 3.8, 0, 0}, {1.2, 1.7, 1.4, 1, 1, 1, 1.3}, {1.4, 1.4, 3.7, 1.6, 2.5, 2.3, 0}, {2.8, 2.7, 2.1, 1, 2.1, 0, 0}, {1.8, 4, 5, 8.6, 0, 0, 0}, {4.7, 3.4, 4.4, 0, 0, 0, 0}, {4.2, 6.3, 0, 0, 0, 0, 0}, {7.3, 0, 0, 0, 0, 0, 0}},  // log_cholesky
      {{1, 1, 2.3, 0, 0, 0, 0}, {1.8, 1.8, 1, 8.4, 1.5, 0, 0}, {1.2, 1.9, 3.8, 1.9, 1.5, 2.9, 2.8}, {1, 3.9, 1.6, 2.3, 2.4, 2.7, 0}, {2.2, 3.8, 1.1, 3, 3.4, 0, 0}, {1, 1, 1, 1.3, 0, 0, 0}, {1, 1.8, 1, 0, 0, 0, 0}, {1, 1.1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // scaled_ar1
      {{1, 1, 1, 0, 0, 0, 0}, {1.3, 1, 1, 1, 13, 0, 0}, {1, 1, 1, 1, 1.4, 3.1, 3}, {1, 1.2, 1, 1, 1, 1, 0}, {1.7, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 2.1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // factor_analytic
   },
  };
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 6; ++s)
      for (int a = 0; a < 9; ++a) {
        if (print) {
          std::printf("mvstudent_t%d %-18s A=%2d", p + 1, sname[s], a - 2);
          for (int K = 0; K <= 6; ++K) std::printf(" %9.3g", worst[p][s][a][K]);
          std::printf("  worst nu by K:");
          for (int K = 0; K <= 6; ++K)
            if (worst[p][s][a][K] > 0.0) std::printf(" %g", cases[worst_g[p][s][a][K]].theta.back());
          std::printf("\n");
        }
        for (int K = 0; K <= 6; ++K)
          CHECK_MESSAGE(worst[p][s][a][K] <= bound[p][s][a][K], "mvstudent_t" << p + 1 << " " << std::string(sname[s]) << " A "
                                                                  << a - 2 << " K " << K << ": " << worst[p][s][a][K]
                                                                  << " ulp1N");
      }
}

// against statmodels7's frozen results (tools/reference/statmodels7_mvstudent_t.R):
// log_cholesky, blocks to A + K = 4 and the information at order 0; the error is
// statmodels7's as much as Laplace's, in ulp1N
void mvstudent_t_s7_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t_s7.csv", false);
  const int n = int(cases.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> val("val", n);
  evaluate(dc, val, 1.0, false);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  auto ts = scales_of(cases);
  double worst[2][9] = {};  // [param - 1][A + 2]
  for (int g = 0; g < n; ++g) {
    const Case& c = cases[g];
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) err = std::fmax(err, std::fabs(hv(g, c.pos[r]) - c.hi[r]));
    double& w = worst[c.param - 1][c.A + 2];
    w = std::fmax(w, err / (scale_of(ts, c) * std::ldexp(1.0, -52)));
  }
  for (int p = 0; p < 2; ++p) {
    if (std::getenv("LAPLACE_PRINT_ERRORS")) {
      std::printf("statmodels7 mvstudent_t%d by A = -1..4:", p + 1);
      for (int a = 1; a <= 6; ++a) std::printf(" %9.3g", worst[p][a]);
      std::printf(" ulp1N\n");
    }
    // measured on gcc and CUDA on 2026-11-17, times 1.5, rounded up (A = -1..4)
    const double sb[2][6] = {{5, 16, 18, 25, 11, 8}, {9, 5, 22, 8, 16, 9}};
    for (int a = 1; a <= 6; ++a)
      CHECK_MESSAGE(worst[p][a] <= sb[p][a - 1], "statmodels7 mvstudent_t" << p + 1 << " A " << a - 2 << ": " << worst[p][a]);
  }
}

// mvstudent_t1 at theta against mvstudent_t2 at -theta over the diagonal structure
void mvstudent_t_twin_case() {
  auto all = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t.csv", false);
  std::vector<Case> cs;
  for (const Case& c : all)
    if (c.structure == 0 && c.param == 1) cs.push_back(c);
  const int n = int(cs.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cs);
  Kokkos::View<double* [max_out]> v1("v1", n), v2("v2", n);
  evaluate(dc, v1, 1.0, false);
  evaluate(dc, v2, -1.0, true);
  auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v1);
  auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v2);
  auto ts = scales_of(cs);
  double worst = 0.0;
  for (int g = 0; g < n; ++g) {
    const Case& c = cs[g];
    int kt = 0;
    for (int i : c.idx) kt += i < c.d;  // theta indices of the tuple, and of the pair for the information
    if (c.A == -1) kt += (c.ab[0] < c.d) + (c.ab[1] < c.d);
    const double sgn = kt % 2 ? -1.0 : 1.0;
    for (int k = 0; k < max_out; ++k)
      worst = std::fmax(worst, std::fabs(h1(g, k) - sgn * h2(g, k)) / (scale_of(ts, c) * std::ldexp(1.0, -52)));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("twin t1/t2 diagonal: %.3g ulp1N\n", worst);
  CHECK(worst <= 1.0);  // measured 0 on gcc and CUDA (2026-11-17): the same law
}

// the table of order 3 gives the same blocks, to the bit
void mvstudent_t_table_case() {
  auto all = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t.csv", false);
  std::vector<Case> cs;
  for (const Case& c : all)
    if (c.K >= 1) cs.push_back(c);
  const int n = int(cs.size());
  const DeviceCases dc = upload(cs);
  Kokkos::View<double* [max_out]> v1("v1", n), v2("v2", n);
  evaluate(dc, v1, 1.0, false, false);
  evaluate(dc, v2, 1.0, false, true);
  auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v1);
  auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v2);
  int diff = 0;
  for (int g = 0; g < n; ++g)
    for (int k = 0; k < max_out; ++k) diff += !(h1(g, k) == h2(g, k) || (std::isnan(h1(g, k)) && std::isnan(h2(g, k))));
  CHECK(diff == 0);
}

// draws: mean and covariance of n units against mu and nu/(nu - 2) Sigma
template <class D>
void draws_check(const D& g, const std::vector<double>& th, const char* name) {
  const int d = g.dim(), np = lp::packed_size(d), n = 400000;
  Kokkos::View<double**, Kokkos::LayoutRight> ys("ys", n, d);
  Kokkos::View<double*> thv("th", th.size()), sbuf("sbuf", g.state_size() + g.scratch_size());
  Kokkos::deep_copy(thv, Kokkos::View<const double*, Kokkos::HostSpace>(th.data(), th.size()));
  Kokkos::View<typename D::state_type> stv("st");
  Kokkos::parallel_for(
      "mvstudent_t_state", 1, KOKKOS_LAMBDA(int) { stv() = g.theta_state(thv.data(), sbuf.data()); });
  Kokkos::parallel_for(
      "mvstudent_t_draws", n, KOKKOS_LAMBDA(int i) {
        double mu[max_d] = {0.5, -1.0, 0.25, 2.0}, y[max_d];
        g.rng(stv(), mu, 20261116, i, y);
        for (int k = 0; k < d; ++k) ys(i, k) = y[k];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ys);
  std::vector<double> y(n * d);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) y[i * d + k] = h(i, k);
  const auto sm = ld::summarize_mv_sample(y.data(), n, d);
  std::vector<double> cov(np), ws(4096), m(d);
  g.covariance(th.data(), cov.data(), ws.data());
  const double mu[max_d] = {0.5, -1.0, 0.25, 2.0};
  g.mean(th.data(), mu, m.data());
  double em = 0.0, ec = 0.0;
  for (int k = 0; k < d; ++k) em = std::fmax(em, std::fabs(sm.mean[k] - m[k]) / std::sqrt(cov[lp::packed_index(k, k)]));
  for (int k = 0; k < np; ++k) ec = std::fmax(ec, std::fabs(sm.covariance[k] - cov[k]));
  double cmax = 0.0;
  for (double v : cov) cmax = std::fmax(cmax, std::fabs(v));
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("%s draws: mean %.3g sd, covariance %.3g\n", name, em, ec / cmax);
  CHECK_MESSAGE(em <= 0.01, name << ": mean " << em);
  CHECK_MESSAGE(ec <= 0.03 * cmax, name << ": covariance " << ec / cmax);
}

// --------------------------------------------------------------- degenerate (host)
using host_matrix = ws_t::host_matrix;
using rows_t = std::vector<std::vector<double>>;
static_assert(ld::multivariate_distrib<ld::mvstudent_t1_distrib<ws_t>>);
static_assert(ld::multivariate_distrib<ld::mvstudent_t1_distrib<log_chol>>);
static_assert(ld::multivariate_distrib<ld::mvstudent_t2_distrib<log_chol>>);
static_assert(ld::mvstudent_t1_distrib<ws_t>::normalization == ld::distrib_normalization::degenerate);
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

template <int A>
bool degenerate_block(const ld::mvstudent_t1_distrib<ws_t>& g, const ld::mvstudent_t1_distrib<ws_t>::state_type& st,
                      int K, const Case& c, double* out, double* w) {
  // host only (the degenerate class is): not through the KOKKOS_INLINE_FUNCTION
  // dispatch above, which nvcc compiles for the device too
  const int* idx = c.idx.data();
  const double *y = c.y.data(), *mu = c.mu.data();
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

void mvstudent_t_degenerate_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvstudent_t.csv", true);
  REQUIRE(!cases.empty());
  std::map<int, ws_t> structs;
  for (int s = 0; s < 4; ++s) structs.emplace(s, make_structure(s));
  auto ts = scales_of(cases);
  double worst[4][9][7] = {};
  const double eps = std::ldexp(1.0, -52);
  for (const Case& c : cases) {
    const ld::mvstudent_t1_distrib<ws_t> g(structs.at(c.structure));
    std::vector<double> w(g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()));
    auto st = g.theta_state(c.theta.data(), w.data());
    double* ws = w.data() + g.state_size();
    std::vector<double> out(laplace::distrib::binomial_coef(c.d + 5, 6) + 64, 0.0);  // A <= 6
    switch (c.A) {
      case -2: degenerate_block<-2>(g, st, c.K, c, out.data(), ws); break;
      case -1: degenerate_block<-1>(g, st, c.K, c, out.data(), ws); break;
      case 0: degenerate_block<0>(g, st, c.K, c, out.data(), ws); break;
      case 1: degenerate_block<1>(g, st, c.K, c, out.data(), ws); break;
      case 2: degenerate_block<2>(g, st, c.K, c, out.data(), ws); break;
      case 3: degenerate_block<3>(g, st, c.K, c, out.data(), ws); break;
      case 4: degenerate_block<4>(g, st, c.K, c, out.data(), ws); break;
      case 5: degenerate_block<5>(g, st, c.K, c, out.data(), ws); break;
      default: degenerate_block<6>(g, st, c.K, c, out.data(), ws); break;
    }
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs((out[c.pos[r]] - c.hi[r]) - c.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    // the spread point (theta 9 apart) is the dense (Sigma + cP)^-1 of M3, measured apart
    const int row = c.theta.back() > 1e4 ? 3 : c.structure == 3 ? 2 : c.structure == 0 ? 0 : 1;
    double& wv = worst[row][c.A + 2][c.K];
    wv = std::fmax(wv, err / (scale_of(ts, c) * eps));
  }
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  const char* rname[] = {"ws d=4", "ws d=5,6", "ws d=5 two D2", "spread nu=1e5"};
  // bounds as for the proper class (gcc and the host code of the CUDA build); the
  // spread point (theta 9 apart) carries the loss of the dense (Sigma + cP)^-1 of
  // the degenerate Gaussian (note/distrib_multivariate.md section 13)
  const double dbound[4][9][7] = {
      {{1, 1, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 5.1}, {1.1, 1, 1, 1, 1.2, 2.2, 0}, {1, 1, 1, 1, 1.7, 0, 0}, {1, 1, 1, 1, 0, 0, 0}, {1, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0}},  // ws d=4
      {{8.4, 4.3, 1, 0, 0, 0, 0}, {1.4, 1, 1, 1, 1, 0, 0}, {1.3, 4.5, 8.6, 7.5, 19, 1, 12}, {8.8, 14, 1.7, 2.8, 2.6, 4.9, 0}, {16, 20, 3.9, 8.5, 6.4, 0, 0}, {20, 34, 16, 26, 0, 0, 0}, {21, 14, 19, 0, 0, 0, 0}, {42, 49, 0, 0, 0, 0, 0}, {19, 0, 0, 0, 0, 0, 0}},  // ws d=5,6
      {{12, 4.9, 1, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0}, {1.9, 2.1, 1.2, 1, 2, 4.4, 1}, {9.5, 6, 2.7, 5.5, 4.8, 3.1, 0}, {26, 8.8, 1.6, 2.6, 4.8, 0, 0}, {11, 9.8, 3.4, 3.6, 0, 0, 0}, {14, 9.8, 1.9, 0, 0, 0, 0}, {20, 28, 0, 0, 0, 0, 0}, {13, 0, 0, 0, 0, 0, 0}},  // ws d=5 two D2
      {{42000, 390000, 390000, 0, 0, 0, 0}, {1.6, 1, 1, 1, 1, 0, 0}, {35000, 84000, 170000, 1, 1, 1.4e+06, 2.7e+06}, {42000, 83000, 170000, 430000, 670000, 1.4e+06, 0}, {42000, 83000, 170000, 340000, 6.3, 0, 0}, {44000, 130000, 190000, 3.9, 0, 0, 0}, {40000, 120000, 190000, 0, 0, 0, 0}, {1200, 4700, 0, 0, 0, 0, 0}, {1100, 0, 0, 0, 0, 0, 0}},  // spread nu=1e5
  };
  for (int r = 0; r < 4; ++r)
    for (int a = 0; a < 9; ++a) {
      if (print) {
        std::printf("mvstudent_t1 degenerate %-14s A=%2d", rname[r], a - 2);
        for (int K = 0; K <= 6; ++K) std::printf(" %9.3g", worst[r][a][K]);
        std::printf("\n");
      }
      for (int K = 0; K <= 6; ++K)
        CHECK_MESSAGE(worst[r][a][K] <= dbound[r][a][K], "degenerate " << rname[r] << " A " << a - 2 << " K " << K << ": "
                                                                       << worst[r][a][K] << " ulp1N");
    }
  // -inf off the support, and the degenerate draw on mu + col(Sigma)
  const ld::mvstudent_t1_distrib<ws_t> g(structs.at(1));
  const std::vector<double> th = {0.3, -0.2, 6.0};
  std::vector<double> w(g.state_size() + g.scratch_size());
  auto st = g.theta_state(th.data(), w.data());
  std::vector<double> mu(5, 0.0), y(5, 1.0), tmp(5);
  CHECK(g.logpdf(st, y.data(), mu.data()) == -INFINITY);  // the constant vector is the null space of D1
  g.rng(st, mu.data(), 7, 3, y.data(), tmp.data());
  CHECK(std::isfinite(g.logpdf(st, y.data(), mu.data())));
}

}  // namespace mvstudent_t_test

TEST_CASE("mvstudent_t1, mvstudent_t2: blocks and expected information against mpmath") {
  mvstudent_t_test::mvstudent_t_mpmath_case();
}
TEST_CASE("mvstudent_t1, mvstudent_t2: log_cholesky against frozen statmodels7 results") {
  mvstudent_t_test::mvstudent_t_s7_case();
}
TEST_CASE("mvstudent_t1 at theta against mvstudent_t2 at -theta (diagonal)") { mvstudent_t_test::mvstudent_t_twin_case(); }
TEST_CASE("mvstudent_t1, mvstudent_t2: the theta table gives the blocks to the bit") {
  mvstudent_t_test::mvstudent_t_table_case();
}
TEST_CASE("mvstudent_t1, mvstudent_t2: draws") {
  namespace lp = laplace::pdmat;
  namespace ld = laplace::distrib;
  using lc = mvstudent_t_test::log_chol;
  mvstudent_t_test::draws_check(ld::mvstudent_t1_distrib<lc>(lc(3)), {0.2, 0.4, -0.3, 0.1, 0.5, -0.2, 8.0},
                                "mvstudent_t1 log_cholesky d=3");
  mvstudent_t_test::draws_check(ld::mvstudent_t2_distrib<lc>(lc(3)), {0.2, 0.4, -0.3, 0.1, 0.5, -0.2, 8.0},
                                "mvstudent_t2 log_cholesky d=3");
}
TEST_CASE("mvstudent_t1: the typed parameter list and the one-block fallback of the block helpers") {
  namespace lp = laplace::pdmat;
  namespace ld = laplace::distrib;
  using lc = mvstudent_t_test::log_chol;
  using T = ld::mvstudent_t1_distrib<lc>;
  static_assert(T::n_params == 3 && T::params[0].kind == ld::distrib_mv_param_kind::vector &&
                T::params[1].kind == ld::distrib_mv_param_kind::matrix &&
                T::params[2].kind == ld::distrib_mv_param_kind::scalar);
  constexpr int d = 3;
  const T g{lc(d)};
  const int q = lc(d).n_params();
  CHECK(g.param_size(0) == d);
  CHECK(g.param_size(1) == q);
  CHECK(g.param_size(2) == 1);
  CHECK(g.param_offset(0) == 0);
  CHECK(g.param_offset(1) == 0);
  CHECK(g.param_offset(2) == q);
  CHECK(g.theta_size() == q + 1);
  CHECK(ld::mv_n_blocks(g) == 1);
  CHECK(ld::mv_block_start(g, ld::y_slot, 0) == 0);
  CHECK(ld::mv_block_start(g, ld::y_slot, 1) == d);
  CHECK(ld::mv_block_start(g, 0, 1) == d);
  CHECK(ld::mv_block_tensor_size<2>(g) == lp::packed_size(d));
  CHECK(ld::mv_block_tensor_size<>(g) == 1);
  CHECK(ld::mv_block_tensor_size(g, ld::y_slot, 3) == ld::sym_tensor_size(d, 3));
  CHECK(ld::mv_block_tensor_size<1>(g, 2) == lp::packed_size(d) * d);
  const std::vector<double> th = {0.2, 0.4, -0.3, 0.1, 0.5, -0.2, 8.0}, y = {0.3, -1.1, 0.7}, mu = {0.1, 0.2, -0.4};
  std::vector<double> w(g.state_size() + g.scratch_size());
  const auto st = g.theta_state(th.data(), w.data());
  double* ws = w.data() + g.state_size();
  const int idx[2] = {1, q};
  std::vector<double> a(d), b(d);
  REQUIRE(g.logpdf_dk<2, 1>(st, y.data(), mu.data(), idx, a.data(), ws));
  REQUIRE(ld::mv_logpdf_dk_blocks<2, 1>(g, st, y.data(), mu.data(), idx, b.data(), ws));
  CHECK(a == b);
  REQUIRE(ld::mv_logpdf_dy_dk_blocks<1, 2>(g, st, y.data(), mu.data(), idx, b.data(), ws));
  for (int k = 0; k < d; ++k) CHECK(b[k] == -a[k]);
  double s0 = 0.0, s1 = 0.0;
  REQUIRE(g.logpdf_dk<1>(st, y.data(), mu.data(), idx + 1, &s0, ws));
  REQUIRE(g.logpdf_dk<1, 0>(st, y.data(), mu.data(), idx + 1, &s1, ws));
  CHECK(s0 == s1);
  // logpdf_dy_dk<J, K, A>: d_y^J d_mu^A = (-1)^J d_mu^(J + A), y factor first
  std::vector<double> wy(g.scratch_size() + ld::sym_tensor_size(d, 3)), t3(ld::sym_tensor_size(d, 3)), t2(lp::packed_size(d));
  std::vector<double> c(d * d), e(lp::packed_size(d) * d);
  REQUIRE(g.logpdf_dk<1, 2>(st, y.data(), mu.data(), idx + 1, t2.data(), ws));
  REQUIRE(g.logpdf_dy_dk<1, 1, 1>(st, y.data(), mu.data(), idx + 1, c.data(), wy.data()));
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) CHECK(c[i * d + j] == -t2[lp::packed_index(i > j ? i : j, i > j ? j : i)]);
  REQUIRE(g.logpdf_dk<0, 3>(st, y.data(), mu.data(), nullptr, t3.data(), ws));
  REQUIRE(g.logpdf_dy_dk<2, 0, 1>(st, y.data(), mu.data(), nullptr, e.data(), wy.data()));
  for (int i = 0; i < d; ++i)  // the y pair (i, j), j <= i, packed by rows; then k
    for (int j = 0; j <= i; ++j)
      for (int k = 0; k < d; ++k) {
        int s[3] = {i, j, k};
        std::sort(s, s + 3);
        CHECK(e[lp::packed_index(i, j) * d + k] == t3[ld::sym_tensor_position<d, 3>(s)]);
      }
  // against central differences in y of the order below
  const double h = 1e-5;
  std::vector<double> yp(y), ym(y), fp(d * d), fm(d * d), gp(d), gm(d);
  for (int i = 0; i < d; ++i) {
    yp = y;
    ym = y;
    yp[i] += h;
    ym[i] -= h;
    REQUIRE(g.logpdf_dy_dk<0, 1, 1>(st, yp.data(), mu.data(), idx + 1, gp.data(), wy.data()));
    REQUIRE(g.logpdf_dy_dk<0, 1, 1>(st, ym.data(), mu.data(), idx + 1, gm.data(), wy.data()));
    for (int j = 0; j < d; ++j) CHECK(c[i * d + j] == doctest::Approx((gp[j] - gm[j]) / (2 * h)).epsilon(1e-6));
    REQUIRE(g.logpdf_dy_dk<1, 0, 1>(st, yp.data(), mu.data(), nullptr, fp.data(), wy.data()));
    REQUIRE(g.logpdf_dy_dk<1, 0, 1>(st, ym.data(), mu.data(), nullptr, fm.data(), wy.data()));
    for (int j = 0; j < d; ++j)
      for (int k = 0; k < d; ++k) {
        const int o = lp::packed_index(i > j ? i : j, i > j ? j : i) * d + k;
        CHECK(e[o] == doctest::Approx((fp[j * d + k] - fm[j * d + k]) / (2 * h)).epsilon(1e-6));
      }
  }
}
TEST_CASE("mvstudent_t1 degenerate over weighted_sum_psdmat: against mpmath, support, draw") {
  mvstudent_t_test::mvstudent_t_degenerate_case();
}
