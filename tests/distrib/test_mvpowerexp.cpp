// mvpowerexp1_distrib and mvpowerexp2_distrib (include/laplace/distrib/):
//  - against the mpmath references (tools/reference/distrib_mvpowerexp.py), in a
//    Kokkos kernel on the default execution space: blocks (A, K) of the
//    log-density, A + K <= 6, over six matrix structures and the betas 0.3, 0.5,
//    0.7, 1, 2 and 6, with r at the location, at 1e-3 of it, at order one and
//    far in the tail; the error of a block is normwise, in ulps of
//    max(max_entries |ref|, 1) (ulp1N);
//  - beta = 1 against mvgaussian1 and mvgaussian2: every block, within a few
//    tenths of an ulp of its scale (log Z(d, 1) goes through lgamma);
//  - mvpowerexp1 over the diagonal structure at theta against mvpowerexp2 at
//    -theta (the same law, beta unchanged, Q = Sigma^-1): every block agrees up
//    to (-1)^K_theta;
//  - beta = 1/2 with Sigma = sigma^2 I: the kernel is ||r||/(2 sigma), the group
//    lasso;
//  - log Z against a one-dimensional quadrature of the radial form of the
//    definition, and the mass of the law itself the same way;
//  - the kink at y = mu: the value and the derivatives in beta alone, and the
//    smoothness at an integer beta;
//  - the derivatives against central differences of the order below, away from
//    the kink;
//  - the theta table gives the blocks to the bit;
//  - the typed parameter list and the descriptors;
//  - draws: the mean mu and the covariance Sigma 2^(1/beta) Gamma((d+2)/(2 beta))
//    / (d Gamma(d/(2 beta))).
// Set LAPLACE_PRINT_ERRORS=1 to print the tables.
#include <doctest/doctest.h>

#include <laplace/distrib/mvpowerexp1.hpp>
#include <laplace/distrib/mvpowerexp2.hpp>
#include <laplace/math/quadrature.hpp>
#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/diagonal.hpp>
#include <laplace/pdmat/factor_analytic.hpp>
#include <laplace/pdmat/unit_diagonal_scaled.hpp>
#include <laplace/pdmat/unstructured.hpp>

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
namespace mvpowerexp_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
namespace quad = laplace::math::quad;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using scaled_ar1 = lp::unit_diagonal_scaled_pdmat<lp::ar1_pdmat>;

constexpr int max_d = 5, max_q = 11, max_out = 210;  // C(5 + 6 - 1, 6)
const char* sname[] = {"diagonal", "ar1", "compound_symmetry", "log_cholesky", "scaled_ar1", "factor_analytic"};

template <class D, int A, int K>
KOKKOS_INLINE_FUNCTION bool block_ak(const D& g, const typename D::state_type& st, const double* y, const double* mu,
                                     const int* idx, double* out, double* w) {
  return g.template logpdf_dk<K, A>(st, y, mu, idx, out, w);
}
template <class D, int A>
KOKKOS_INLINE_FUNCTION bool block_a(const D& g, const typename D::state_type& st, int K, const double* y,
                                    const double* mu, const int* idx, double* out, double* w) {
  switch (K) {
    case 0: return block_ak<D, A, 0>(g, st, y, mu, idx, out, w);
    case 1: if constexpr (A <= 5) return block_ak<D, A, 1>(g, st, y, mu, idx, out, w); break;
    case 2: if constexpr (A <= 4) return block_ak<D, A, 2>(g, st, y, mu, idx, out, w); break;
    case 3: if constexpr (A <= 3) return block_ak<D, A, 3>(g, st, y, mu, idx, out, w); break;
    case 4: if constexpr (A <= 2) return block_ak<D, A, 4>(g, st, y, mu, idx, out, w); break;
    case 5: if constexpr (A <= 1) return block_ak<D, A, 5>(g, st, y, mu, idx, out, w); break;
    case 6: if constexpr (A == 0) return block_ak<D, A, 6>(g, st, y, mu, idx, out, w); break;
  }
  return false;
}
template <class D>
KOKKOS_INLINE_FUNCTION void eval_d(const D& g, int A, int K, const double* th, const double* y, const double* mu,
                                   const int* idx, double* out, double* w, bool table) {
  for (int k = 0; k < max_out; ++k) out[k] = 0.0;
  auto st = g.theta_state(th, w);
  double* ws = w + g.state_size();
  if (table) {
    double* tab = ws + g.scratch_size();  // the theta table of order 3 after the scratch
    for (int e = 0; e < g.theta_table_entries(3); ++e) g.fill_theta_table_entry(st, 3, e, tab, ws);
    st = g.with_theta_table(st, 3, tab);
  }
  switch (A) {
    case 0: block_a<D, 0>(g, st, K, y, mu, idx, out, ws); break;
    case 1: block_a<D, 1>(g, st, K, y, mu, idx, out, ws); break;
    case 2: block_a<D, 2>(g, st, K, y, mu, idx, out, ws); break;
    case 3: block_a<D, 3>(g, st, K, y, mu, idx, out, ws); break;
    case 4: block_a<D, 4>(g, st, K, y, mu, idx, out, ws); break;
    case 5: block_a<D, 5>(g, st, K, y, mu, idx, out, ws); break;
    default: block_a<D, 6>(g, st, K, y, mu, idx, out, ws); break;
  }
}
template <class S>
KOKKOS_INLINE_FUNCTION void eval_s(const S& s, int param, int A, int K, const double* th, const double* y,
                                   const double* mu, const int* idx, double* out, double* w, bool table) {
  if (param == 1) eval_d(ld::mvpowerexp1_distrib<S>(s), A, K, th, y, mu, idx, out, w, table);
  else eval_d(ld::mvpowerexp2_distrib<S>(s), A, K, th, y, mu, idx, out, w, table);
}
KOKKOS_INLINE_FUNCTION void eval_any(int structure, int d, int param, int A, int K, const double* th, const double* y,
                                     const double* mu, const int* idx, double* out, double* w, bool table = false) {
  switch (structure) {
    case 0: eval_s(lp::diagonal_pdmat(d), param, A, K, th, y, mu, idx, out, w, table); break;
    case 1: eval_s(lp::ar1_pdmat(d), param, A, K, th, y, mu, idx, out, w, table); break;
    case 2: eval_s(lp::compound_symmetry_pdmat(d), param, A, K, th, y, mu, idx, out, w, table); break;
    case 3: eval_s(log_chol(d), param, A, K, th, y, mu, idx, out, w, table); break;
    case 4: eval_s(scaled_ar1(lp::ar1_pdmat(d)), param, A, K, th, y, mu, idx, out, w, table); break;
    default: eval_s(lp::factor_analytic_pdmat(d, 1), param, A, K, th, y, mu, idx, out, w, table); break;
  }
}

template <class S>
int work_of(const S& s) {
  const ld::mvpowerexp1_distrib<S> g(s);
  const ld::mvpowerexp2_distrib<S> g2(s);
  const int a = g.state_size() + g.scratch_size() + g.theta_table_size(3);
  const int b = g2.state_size() + g2.scratch_size() + g2.theta_table_size(3);
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
KOKKOS_INLINE_FUNCTION int n_struct_params(int s, int d) {
  return s == 0 ? d : s == 1 || s == 2 ? 1 : s == 3 ? d * (d + 1) / 2 : s == 4 ? d + 1 : 2 * d;
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
  std::vector<int> pos;
  std::vector<double> hi, lo;
  std::string point;
};

// a block groups the entries of the same (point, A, idx)
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
    const std::string key = line.substr(0, line.size() - f[8].size() - f[9].size() - f[10].size() - 3);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(cases.size())).first;
      Case c;
      c.point = f[0] + "," + f[1] + "," + f[2] + "," + f[3] + "," + f[4] + "," + f[5];
      c.param = std::atoi(f[0].c_str()), c.structure = std::atoi(f[1].c_str()), c.d = std::atoi(f[2].c_str());
      c.theta = hex_list(f[3]), c.y = hex_list(f[4]), c.mu = hex_list(f[5]);
      c.A = std::atoi(f[6].c_str()), c.idx = int_list(f[7]), c.K = int(c.idx.size());
      cases.push_back(c);
    }
    Case& c = cases[it->second];
    c.pos.push_back(std::atoi(f[8].c_str()));
    c.hi.push_back(std::strtod(f[9].c_str(), nullptr));
    c.lo.push_back(std::strtod(f[10].c_str(), nullptr));
  }
  return cases;
}

struct DeviceCases {
  Kokkos::View<int* [5]> meta;
  Kokkos::View<double* [max_q]> theta;
  Kokkos::View<double* [max_d]> y, mu;
  Kokkos::View<int* [6]> idx;
  int work = 0;
};

DeviceCases upload(const std::vector<Case>& cs) {
  const int n = int(cs.size());
  DeviceCases dc{Kokkos::View<int* [5]>("meta", n), Kokkos::View<double* [max_q]>("theta", n),
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

// every case; theta_sign multiplies the structure's parameters (not beta), swap
// exchanges the two parametrizations. The output tensor lives in the work View,
// not on the thread stack: at d = 5 and A = 6 it is 210 doubles.
void evaluate(const DeviceCases& dc, Kokkos::View<double* [max_out]> val, double theta_sign, bool swap,
              bool table = false) {
  const int n = int(dc.meta.extent(0));
  Kokkos::View<double**, Kokkos::LayoutRight> work("work", n, dc.work + max_out);
  auto meta = dc.meta;
  auto theta = dc.theta;
  auto ys = dc.y;
  auto mus = dc.mu;
  auto idxs = dc.idx;
  Kokkos::parallel_for(
      "mvpowerexp_cases", n, KOKKOS_LAMBDA(int g) {
        const int param = swap ? 3 - meta(g, 0) : meta(g, 0), s = meta(g, 1), d = meta(g, 2), A = meta(g, 3),
                  K = meta(g, 4);
        const int q = n_struct_params(s, d);
        double th[max_q], y[max_d], mu[max_d];
        int idx[6];
        for (int k = 0; k < max_q; ++k) th[k] = k < q ? theta_sign * theta(g, k) : theta(g, k);
        for (int k = 0; k < max_d; ++k) y[k] = ys(g, k), mu[k] = mus(g, k);
        for (int k = 0; k < 6; ++k) idx[k] = idxs(g, k);
        double* out = &work(g, 0);
        eval_any(s, d, param, A, K, th, y, mu, idx, out, &work(g, max_out), table && K <= 3);
        for (int k = 0; k < max_out; ++k) val(g, k) = out[k];
      });
}

// the scale of a block: the largest |ref| over the entries and tuples of the
// same point, A and K, at least 1
std::map<std::string, double> scales_of(const std::vector<Case>& cases) {
  std::map<std::string, double> t;
  for (const Case& c : cases) {
    double& v = t[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)];
    for (double h : c.hi) v = std::fmax(v, std::fabs(h));
  }
  return t;
}
double scale_of(std::map<std::string, double>& t, const Case& c) {
  return std::fmax(1.0, t[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)]);
}

void mvpowerexp_mpmath_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvpowerexp.csv");
  const int n = int(cases.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> val("val", n);
  evaluate(dc, val, 1.0, false);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  auto ts = scales_of(cases);
  double worst[2][6][7][7] = {};  // [param - 1][structure][A][K]
  int worst_g[2][6][7][7] = {};
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < n; ++g) {
    const Case& c = cases[g];
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs((hv(g, c.pos[r]) - c.hi[r]) - c.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double u = err / (scale_of(ts, c) * eps);
    double& w = worst[c.param - 1][c.structure][c.A][c.K];
    if (!(u <= w)) w = u, worst_g[c.param - 1][c.structure][c.A][c.K] = g;
  }
  // bounds on ulp1N, [param - 1][structure][A][K]: the larger of the errors
  // measured on gcc/OpenMP and on the default CUDA build (2026-11-18), times
  // 1.5, rounded up to two digits; 0 where no case exists. The largest, up to
  // 3.96e3 at A = 4, are mvpowerexp1 over log_cholesky at the near-singular
  // point (log L33 = -7.5, cond(Sigma) = 3.5e7) at beta = 1/2, and they are
  // the error of Sigma^-1 itself: the double Sigma^-1 there is 1.0e-9 off, and
  // feeding it into an otherwise exact computation of the same blocks costs
  // 750, 1441 and 3694 ulp1N at A = 2, 3, 4 (docs/numerics.md). mvpowerexp2
  // takes Q directly and never inverts, and stays within 18 there.
  static const double bound[2][6][7][7] = {
   {
      {{1.3, 14, 14, 16, 3.9, 2.5, 3.7}, {15, 14, 17, 17, 15, 52, 0}, {14, 11, 11, 4.2, 56, 0, 0}, {3, 8.8, 3.5, 15, 0, 0, 0}, {2.4, 19, 30, 0, 0, 0, 0}, {6.8, 89, 0, 0, 0, 0, 0}, {59, 0, 0, 0, 0, 0, 0}},  // diagonal
      {{6.1, 6.6, 2, 2.1, 1.5, 2.8, 3.4}, {1, 2.2, 1.2, 1.5, 1.3, 2, 0}, {1, 1.2, 1.1, 1.5, 2, 0, 0}, {1, 1.3, 3.1, 1.5, 0, 0, 0}, {1.3, 3.4, 4.2, 0, 0, 0, 0}, {1.6, 2.8, 0, 0, 0, 0, 0}, {7, 0, 0, 0, 0, 0, 0}},  // ar1
      {{2.5, 5.9, 6.5, 3.2, 2.9, 2.6, 3.8}, {7.7, 7.3, 4.6, 6.7, 6.2, 6.5, 0}, {9.1, 5.1, 4.2, 5.2, 5.9, 0, 0}, {3.2, 3.3, 4.3, 2.5, 0, 0, 0}, {4.2, 3.7, 4.8, 0, 0, 0, 0}, {3.2, 5.1, 0, 0, 0, 0, 0}, {6.2, 0, 0, 0, 0, 0, 0}},  // compound_symmetry
      {{15, 31, 32, 12, 12, 13, 13}, {43, 43, 43, 43, 26, 110, 0}, {2200, 66, 97, 12, 38, 0, 0}, {2400, 300, 12, 17, 0, 0, 0}, {6000, 12, 13, 0, 0, 0, 0}, {12, 15, 0, 0, 0, 0, 0}, {12, 0, 0, 0, 0, 0, 0}},  // log_cholesky
      {{22, 17, 15, 14, 13, 12, 11}, {16, 12, 14, 14, 33, 25, 0}, {11, 14, 13, 12, 21, 0, 0}, {14, 12, 16, 14, 0, 0, 0}, {12, 11, 12, 0, 0, 0, 0}, {11, 9.8, 0, 0, 0, 0, 0}, {9.1, 0, 0, 0, 0, 0, 0}},  // scaled_ar1
      {{4.5, 12, 11, 3.8, 4, 5.7, 4}, {13, 11, 5.9, 4.8, 8.8, 8.1, 0}, {11, 4, 6.6, 5.5, 9.2, 0, 0}, {3.5, 4.3, 4, 7.1, 0, 0, 0}, {3.9, 4.6, 4.8, 0, 0, 0, 0}, {4.3, 4.1, 0, 0, 0, 0, 0}, {4.2, 0, 0, 0, 0, 0, 0}},  // factor_analytic
   },
   {
      {{7.1, 25, 22, 17, 8.4, 2.5, 4.1}, {25, 22, 24, 21, 5.5, 8.4, 0}, {18, 15, 14, 11, 15, 0, 0}, {6.9, 17, 11, 14, 0, 0, 0}, {4.8, 20, 37, 0, 0, 0, 0}, {6.5, 160, 0, 0, 0, 0, 0}, {75, 0, 0, 0, 0, 0, 0}},  // diagonal
      {{2.6, 7.1, 9, 2, 1.9, 3.7, 3.4}, {9.1, 7.2, 4.2, 4.5, 4, 3, 0}, {7.8, 7.1, 3.9, 3.7, 3.9, 0, 0}, {1, 1.6, 3, 2.2, 0, 0, 0}, {1.2, 3.2, 4.4, 0, 0, 0, 0}, {1.3, 4.2, 0, 0, 0, 0, 0}, {13, 0, 0, 0, 0, 0, 0}},  // ar1
      {{3.9, 6.5, 6.8, 7.6, 5.9, 2.7, 4.1}, {6.9, 6.7, 5.4, 5.9, 7.3, 8.6, 0}, {6.6, 4.6, 3.3, 4.9, 5.8, 0, 0}, {1, 1.6, 3.2, 1.5, 0, 0, 0}, {1.2, 4.2, 3.6, 0, 0, 0, 0}, {2.5, 3.4, 0, 0, 0, 0, 0}, {7.3, 0, 0, 0, 0, 0, 0}},  // compound_symmetry
      {{15, 8.7, 8.7, 7.9, 6.8, 6.5, 6.2}, {7.3, 9, 9.3, 5.2, 4.9, 5.7, 0}, {8.7, 8.6, 5, 4.9, 5.3, 0, 0}, {28, 12, 9.8, 15, 0, 0, 0}, {5.6, 15, 20, 0, 0, 0, 0}, {6.2, 8.7, 0, 0, 0, 0, 0}, {6.4, 0, 0, 0, 0, 0, 0}},  // log_cholesky
      {{12, 13, 9.6, 8.2, 7.5, 7.4, 7.8}, {13, 9.9, 4.7, 6, 7.5, 7.5, 0}, {9.5, 5.4, 5.3, 5.9, 6.6, 0, 0}, {9.8, 8.4, 6.1, 5.2, 0, 0, 0}, {4.2, 4.3, 9.8, 0, 0, 0, 0}, {3.6, 3.7, 0, 0, 0, 0, 0}, {4.1, 0, 0, 0, 0, 0, 0}},  // scaled_ar1
      {{1.1, 17, 3, 4.9, 2.5, 5.9, 3.3}, {3.4, 3.8, 4.4, 4.5, 4.3, 6.1, 0}, {6, 3.6, 7, 5.5, 3.9, 0, 0}, {1.4, 3.6, 15, 8.7, 0, 0, 0}, {1.9, 3.4, 12, 0, 0, 0, 0}, {1.2, 1.8, 0, 0, 0, 0, 0}, {1.5, 0, 0, 0, 0, 0, 0}},  // factor_analytic
   },
  };
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 6; ++s)
      for (int a = 0; a < 7; ++a) {
        if (print) {
          std::printf("mvpowerexp%d %-18s A=%d", p + 1, sname[s], a);
          for (int K = 0; K <= 6; ++K) std::printf(" %9.3g", worst[p][s][a][K]);
          std::printf("  worst (beta, min theta, delta) by K:");
          for (int K = 0; K <= 6; ++K)
            if (worst[p][s][a][K] > 0.0) {
              const Case& wc = cases[worst_g[p][s][a][K]];
              double mn = 0.0;
              for (std::size_t k = 0; k + 1 < wc.theta.size(); ++k) mn = std::fmin(mn, wc.theta[k]);
              double r2 = 0.0;
              for (int k = 0; k < wc.d; ++k) r2 += (wc.y[k] - wc.mu[k]) * (wc.y[k] - wc.mu[k]);
              std::printf(" (%g,%g,%.2g)", wc.theta.back(), mn, std::sqrt(r2));
            }
          std::printf("\n");
        }
        for (int K = 0; K <= 6; ++K)
          CHECK_MESSAGE(worst[p][s][a][K] <= bound[p][s][a][K],
                        "mvpowerexp" << p + 1 << " " << std::string(sname[s]) << " A " << a << " K " << K << ": "
                                     << worst[p][s][a][K] << " ulp1N");
      }
}

// beta = 1 is exactly the Gaussian: every block of mvpowerexp1 at beta = 1
// against mvgaussian1's, and of mvpowerexp2 against mvgaussian2's
template <class D, class G, int A, int K>
void gaussian_block(const D& g, const G& gg, const std::vector<double>& th, const std::vector<double>& y,
                    const std::vector<double>& mu, const int* idx, double& worst, int& nblocks) {
  const int d = g.dim(), m = A == 0 ? 1 : A == 1 ? d : A == 2 ? lp::packed_size(d) : ld::sym_tensor_size(d, A);
  std::vector<double> w(g.state_size() + g.scratch_size()), wg(gg.state_size() + gg.scratch_size());
  std::vector<double> a(m + 8, 0.0), b(m + 8, 0.0);
  const auto st = g.theta_state(th.data(), w.data());
  const auto sg = gg.theta_state(th.data(), wg.data());
  const bool ok = g.template logpdf_dk<K, A>(st, y.data(), mu.data(), idx, a.data(), w.data() + g.state_size());
  const bool og = gg.template logpdf_dk<K, A>(sg, y.data(), mu.data(), idx, b.data(), wg.data() + gg.state_size());
  if (!ok) {
    CHECK(!og);
    return;
  }
  if (!og) {
    // the Gaussian reports its blocks with A >= 3 as zero by structure; at
    // beta = 1 the kernel is quadratic in mu, so this law computes exact zeros
    for (int k = 0; k < m; ++k) CHECK(a[k] == 0.0);
    return;
  }
  ++nblocks;
  double sc = 1.0;
  for (int k = 0; k < m; ++k) sc = std::fmax(sc, std::fabs(b[k]));
  for (int k = 0; k < m; ++k) worst = std::fmax(worst, std::fabs(a[k] - b[k]) / (sc * std::ldexp(1.0, -52)));
}

template <class S>
void gaussian_limit(const S& s, const std::vector<double>& th_mat, const std::vector<double>& y,
                    const std::vector<double>& mu, const char* name) {
  std::vector<double> th = th_mat;
  th.push_back(1.0);  // beta = 1
  const int q = int(th_mat.size());
  double w1 = 0.0, w2 = 0.0;
  int nb = 0;
  const int i0[6] = {0, 0, 0, 0, 0, 0};
  // every block without a beta index: the Gaussian has no such parameter
  const ld::mvpowerexp1_distrib<S> p1(s);
  const ld::mvgaussian1_distrib<S> g1(s);
  const ld::mvpowerexp2_distrib<S> p2(s);
  const ld::mvgaussian2_distrib<S> g2(s);
  gaussian_block<decltype(p1), decltype(g1), 0, 0>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 1, 0>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 2, 0>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 3, 0>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 0, 1>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 1, 1>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 2, 1>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 0, 2>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 1, 2>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p1), decltype(g1), 0, 3>(p1, g1, th, y, mu, i0, w1, nb);
  gaussian_block<decltype(p2), decltype(g2), 0, 0>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 1, 0>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 2, 0>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 3, 0>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 0, 1>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 1, 1>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 2, 1>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 0, 2>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 1, 2>(p2, g2, th, y, mu, i0, w2, nb);
  gaussian_block<decltype(p2), decltype(g2), 0, 3>(p2, g2, th, y, mu, i0, w2, nb);
  (void)q;
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("beta = 1 against the Gaussian, %-16s: %.3g and %.3g ulp1N over %d blocks\n", name, w1, w2, nb);
  // the kernel -1/2 delta^1 is pow(delta, 1.0), exactly delta, and the constant
  // log Z(d, 1) = d/2 log(2 pi) through lgamma; measured at most 0.884 ulp1N on
  // gcc and CUDA (2026-11-18), times 1.5
  CHECK_MESSAGE(w1 <= 1.4, name << " mvpowerexp1 against mvgaussian1: " << w1 << " ulp1N");
  CHECK_MESSAGE(w2 <= 1.4, name << " mvpowerexp2 against mvgaussian2: " << w2 << " ulp1N");
}

// mvpowerexp1 at theta against mvpowerexp2 at -theta over the diagonal
// structure: the same law, since there Q = Sigma^-1 exactly
void mvpowerexp_twin_case() {
  auto all = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvpowerexp.csv");
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
    for (int i : c.idx) kt += i < c.d;  // the theta indices of the tuple
    const double sgn = kt % 2 ? -1.0 : 1.0;
    for (int k = 0; k < max_out; ++k) {
      const double a = h1(g, k), b = sgn * h2(g, k);
      if (!std::isfinite(a) || !std::isfinite(b)) {
        CHECK(a == b);
        continue;
      }
      worst = std::fmax(worst, std::fabs(a - b) / (scale_of(ts, c) * std::ldexp(1.0, -52)));
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("twin mvpowerexp1/2 diagonal: %.3g ulp1N\n", worst);
  CHECK(worst <= 1.0);  // the same law, computed through the same path
}

// the table of order 3 gives the same blocks, to the bit
void mvpowerexp_table_case() {
  auto all = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvpowerexp.csv");
  std::vector<Case> cs;
  for (const Case& c : all)
    if (c.K >= 1) cs.push_back(c);
  const int n = int(cs.size());
  REQUIRE(n > 0);
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

// log Z and the mass of the law against a one-dimensional quadrature of the
// radial form of the definition:
//   int exp(-delta^beta/2) dr = |Sigma|^(1/2) 2 pi^(d/2)/Gamma(d/2) int_0^inf s^(d-1) exp(-s^(2 beta)/2) ds
void mvpowerexp_constant_case() {
  const double lpi = 1.14472988584940017414342735135306;
  double worst = 0.0, worst_mass = 0.0;
  for (int d : {1, 2, 3, 5})
    for (double beta : {0.3, 0.5, 0.7, 1.0, 2.0, 6.0}) {
      // the radial integral, split where the integrand turns over (s = 1 is the
      // scale of the fall for every beta tried, and the tail is cut at the s
      // where s^(2 beta)/2 reaches 60)
      const double send = std::pow(120.0, 0.5 / beta);
      auto f = [=](double s) { return std::pow(s, d - 1) * std::exp(-std::pow(s, 2.0 * beta) / 2); };
      const auto r1 = quad::gauss_kronrod_quad(f, 0.0, 1.0, 1e-15, 256);
      const auto r2 = quad::gauss_kronrod_quad(f, 1.0, send, 1e-15, 256);
      REQUIRE(r1.converged);
      REQUIRE(r2.converged);
      const double logq = std::log(2.0) + 0.5 * d * lpi - std::lgamma(0.5 * d) + std::log(r1.value + r2.value);
      double lz[7];
      ld::detail_mvpowerexp::log_z_dk(d, beta, 0, lz);
      worst = std::fmax(worst, std::fabs(lz[0] - logq) / std::fmax(std::fabs(logq), 1.0));
    }
  // and the mass through logpdf itself, in one dimension, where the quadrature
  // runs over y and so shares no algebra with the radial form above
  for (double beta : {0.3, 0.5, 0.7, 1.0, 2.0, 6.0}) {
    const lp::diagonal_pdmat s(1);
    const ld::mvpowerexp1_distrib<lp::diagonal_pdmat> g(s);
    const std::vector<double> th = {0.6, beta}, mu = {0.4};
    std::vector<double> w(g.state_size() + g.scratch_size());
    const auto st = g.theta_state(th.data(), w.data());
    const double sigma = std::exp(0.6);  // diagonal_pdmat: theta_0 = log sigma
    // delta = (y - mu)^2/sigma^2, so delta^beta/2 reaches 60 at this half width
    const double hw = sigma * std::pow(120.0, 0.5 / beta);
    auto f = [&](double t) { return std::exp(g.logpdf(st, &t, mu.data())); };
    const auto q1 = quad::gauss_kronrod_quad(f, mu[0] - hw, mu[0], 1e-15, 512);
    const auto q2 = quad::gauss_kronrod_quad(f, mu[0], mu[0] + hw, 1e-15, 512);
    REQUIRE(q1.converged);
    REQUIRE(q2.converged);
    worst_mass = std::fmax(worst_mass, std::fabs(q1.value + q2.value - 1.0));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("log Z against the radial quadrature: %.3g relative; mass - 1: %.3g\n", worst, worst_mass);
  // measured 2.22e-16 (gcc) and 2.95e-16 (CUDA) for log Z, 2.22e-16 for the
  // mass on both (2026-11-18), times 1.5; the kink at y = mu sits on a panel end
  CHECK(worst <= 4.5e-16);
  CHECK(worst_mass <= 3.4e-16);
}

// beta = 1/2 with Sigma = sigma^2 I: l = -log Z - d log sigma - ||r||/(2 sigma)
void mvpowerexp_group_lasso_case() {
  constexpr int d = 3;
  const lp::diagonal_pdmat s(d);
  const ld::mvpowerexp1_distrib<lp::diagonal_pdmat> g(s);
  // diagonal_pdmat: theta_i = log sigma_i, so Sigma = sigma^2 I at theta = log sigma
  const double sigma = 1.7, lv = std::log(sigma);
  const std::vector<double> th = {lv, lv, lv, 0.5};
  const std::vector<double> mu = {0.2, -0.5, 1.1};
  std::vector<double> w(g.state_size() + g.scratch_size());
  const auto st = g.theta_state(th.data(), w.data());
  double lz[7];
  ld::detail_mvpowerexp::log_z_dk(d, 0.5, 0, lz);
  double worst = 0.0;
  for (double t : {0.0, 1e-8, 0.3, 1.0, 5.0, 100.0, 1e6}) {
    const std::vector<double> y = {mu[0] + t, mu[1] - 0.5 * t, mu[2] + 0.2 * t};
    const double nr = std::sqrt(t * t + 0.25 * t * t + 0.04 * t * t);
    const double want = -lz[0] - d * std::log(sigma) - nr / (2.0 * sigma);
    const double got = g.logpdf(st, y.data(), mu.data());
    worst = std::fmax(worst, std::fabs(got - want) / (std::fmax(std::fabs(want), 1.0) * std::ldexp(1.0, -52)));
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("beta = 1/2 group lasso form: %.3g ulp1\n", worst);
  CHECK(worst <= 1.2);  // measured 0.794 ulp1 on gcc and CUDA (2026-11-18), times 1.5
}

// the kink at y = mu: the value is finite, the derivatives in beta alone are
// -log Z^(c), and at an integer beta the law is smooth there
void mvpowerexp_kink_case() {
  constexpr int d = 3;
  const log_chol s(d);
  const std::vector<double> th_mat = {0.2, 0.4, -0.3, 0.1, 0.5, -0.2};
  const std::vector<double> mu = {0.3, -1.0, 0.8};
  for (double beta : {0.3, 0.5, 0.7, 1.0, 2.0, 3.0}) {
    std::vector<double> th = th_mat;
    th.push_back(beta);
    const ld::mvpowerexp1_distrib<log_chol> g(s);
    std::vector<double> w(g.state_size() + g.scratch_size());
    const auto st = g.theta_state(th.data(), w.data());
    double* ws = w.data() + g.state_size();
    const double v = g.logpdf(st, mu.data(), mu.data());
    CHECK(std::isfinite(v));
    CHECK(v == doctest::Approx(-st.lz[0] - 0.5 * st.gauss.log_det).epsilon(1e-15));
    // the pure beta blocks at the kink: the kernel contributes nothing
    const int q = int(th_mat.size());
    for (int c = 1; c <= 6; ++c) {
      int idx[6];
      for (int k = 0; k < c; ++k) idx[k] = q;
      double out = 0.0;
      bool ok = false;
      switch (c) {
        case 1: ok = g.logpdf_dk<1>(st, mu.data(), mu.data(), idx, &out, ws); break;
        case 2: ok = g.logpdf_dk<2>(st, mu.data(), mu.data(), idx, &out, ws); break;
        case 3: ok = g.logpdf_dk<3>(st, mu.data(), mu.data(), idx, &out, ws); break;
        case 4: ok = g.logpdf_dk<4>(st, mu.data(), mu.data(), idx, &out, ws); break;
        case 5: ok = g.logpdf_dk<5>(st, mu.data(), mu.data(), idx, &out, ws); break;
        default: ok = g.logpdf_dk<6>(st, mu.data(), mu.data(), idx, &out, ws); break;
      }
      REQUIRE(ok);
      CHECK(out == -st.lz[c]);
    }
    // the gradient in mu at y = mu: 0 when beta > 1/2 (the first partial is
    // homogeneous of degree 2 beta - 1), infinite at beta = 1/2 and below
    std::vector<double> gr(d, 0.0);
    REQUIRE(g.logpdf_dk<0, 1>(st, mu.data(), mu.data(), nullptr, gr.data(), ws));
    for (int k = 0; k < d; ++k) {
      if (beta > 0.5) CHECK(gr[k] == 0.0);
      else CHECK(!std::isfinite(gr[k]));
    }
    // at an integer beta the second derivative in mu is finite as well
    std::vector<double> he(lp::packed_size(d), 0.0);
    REQUIRE(g.logpdf_dk<0, 2>(st, mu.data(), mu.data(), nullptr, he.data(), ws));
    if (beta == 1.0 || beta == 2.0 || beta == 3.0)
      for (double x : he) CHECK(std::isfinite(x));
  }
}

// the derivatives against central differences of the order below, away from the kink
void mvpowerexp_differences_case() {
  constexpr int d = 3;
  const log_chol s(d);
  const ld::mvpowerexp1_distrib<log_chol> g(s);
  const int q = s.n_params();
  const std::vector<double> mu = {0.3, -1.0, 0.8}, y = {1.1, -0.2, 0.1};
  const double h = 1e-5;
  double worst = 0.0;
  for (double beta : {0.3, 0.5, 0.7, 1.0, 2.0, 6.0}) {
    std::vector<double> th(q);
    for (int k = 0; k < q; ++k) th[k] = 0.2 * (k + 1) - 0.4;
    th.push_back(beta);
    std::vector<double> w(g.state_size() + g.scratch_size());
    double* ws = w.data() + g.state_size();
    // d_beta^2 l against the difference of d_beta l, and d_mu d_beta l against
    // the difference in beta of d_mu l
    const int ib[2] = {q, q};
    double s2 = 0.0;
    {
      const auto st = g.theta_state(th.data(), w.data());
      REQUIRE(g.logpdf_dk<2>(st, y.data(), mu.data(), ib, &s2, ws));
    }
    std::vector<double> gp(d), gm(d);
    double s1p = 0.0, s1m = 0.0;
    std::vector<double> tp = th, tm = th;
    tp[q] += h, tm[q] -= h;
    std::vector<double> wp(g.state_size() + g.scratch_size()), wm(g.state_size() + g.scratch_size());
    const auto sp = g.theta_state(tp.data(), wp.data());
    const auto sm = g.theta_state(tm.data(), wm.data());
    REQUIRE(g.logpdf_dk<1>(sp, y.data(), mu.data(), ib, &s1p, wp.data() + g.state_size()));
    REQUIRE(g.logpdf_dk<1>(sm, y.data(), mu.data(), ib, &s1m, wm.data() + g.state_size()));
    const double fd = (s1p - s1m) / (2 * h);
    worst = std::fmax(worst, std::fabs(s2 - fd) / std::fmax(std::fabs(s2), 1.0));
    // d_mu d_beta against the difference in beta of d_mu
    std::vector<double> mb(d);
    {
      const auto st = g.theta_state(th.data(), w.data());
      REQUIRE(g.logpdf_dk<1, 1>(st, y.data(), mu.data(), ib, mb.data(), ws));
    }
    REQUIRE(g.logpdf_dk<0, 1>(sp, y.data(), mu.data(), nullptr, gp.data(), wp.data() + g.state_size()));
    REQUIRE(g.logpdf_dk<0, 1>(sm, y.data(), mu.data(), nullptr, gm.data(), wm.data() + g.state_size()));
    for (int k = 0; k < d; ++k) {
      const double dk = (gp[k] - gm[k]) / (2 * h);
      worst = std::fmax(worst, std::fabs(mb[k] - dk) / std::fmax(std::fabs(mb[k]), 1.0));
    }
  }
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("central differences in beta: %.3g relative\n", worst);
  // the second-order error of the difference at h = 1e-5: measured 2.66e-9 on
  // gcc and CUDA (2026-11-18), times 1.5
  CHECK(worst <= 4e-9);
}

// draws: mean and covariance of n units against mu and the covariance formula
template <class D>
void draws_check(const D& g, const std::vector<double>& th, const char* name) {
  const int d = g.dim(), np = lp::packed_size(d), n = 400000;
  Kokkos::View<double**, Kokkos::LayoutRight> ys("ys", n, d);
  Kokkos::View<double*> thv("th", th.size()), sbuf("sbuf", g.state_size() + g.scratch_size());
  Kokkos::deep_copy(thv, Kokkos::View<const double*, Kokkos::HostSpace>(th.data(), th.size()));
  Kokkos::View<typename D::state_type> stv("st");
  Kokkos::parallel_for(
      "mvpowerexp_state", 1, KOKKOS_LAMBDA(int) { stv() = g.theta_state(thv.data(), sbuf.data()); });
  Kokkos::parallel_for(
      "mvpowerexp_draws", n, KOKKOS_LAMBDA(int i) {
        double mu[max_d] = {0.5, -1.0, 0.25, 2.0, 0.0}, y[max_d];
        g.rng(stv(), mu, 20261118, i, y);
        for (int k = 0; k < d; ++k) ys(i, k) = y[k];
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ys);
  std::vector<double> y(std::size_t(n) * d);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) y[std::size_t(i) * d + k] = h(i, k);
  const auto sm = ld::summarize_mv_sample(y.data(), n, d);
  std::vector<double> cov(np), ws(4096), m(d);
  g.covariance(th.data(), cov.data(), ws.data());
  const double mu[max_d] = {0.5, -1.0, 0.25, 2.0, 0.0};
  g.mean(th.data(), mu, m.data());
  double em = 0.0, ec = 0.0, cmax = 0.0;
  for (int k = 0; k < d; ++k) em = std::fmax(em, std::fabs(sm.mean[k] - m[k]) / std::sqrt(cov[lp::packed_index(k, k)]));
  for (int k = 0; k < np; ++k) ec = std::fmax(ec, std::fabs(sm.covariance[k] - cov[k]));
  for (double v : cov) cmax = std::fmax(cmax, std::fabs(v));
  if (std::getenv("LAPLACE_PRINT_ERRORS")) std::printf("%s draws: mean %.3g sd, covariance %.3g\n", name, em, ec / cmax);
  // counter-based draws, the same on gcc and CUDA: measured at most 0.00137 sd
  // and 0.0065 relative (2026-11-18), times 1.5
  CHECK_MESSAGE(em <= 0.0021, name << ": mean " << em);
  CHECK_MESSAGE(ec <= 0.0098 * cmax, name << ": covariance " << ec / cmax);
}

}  // namespace mvpowerexp_test

TEST_CASE("mvpowerexp1, mvpowerexp2: blocks against mpmath") { mvpowerexp_test::mvpowerexp_mpmath_case(); }
TEST_CASE("mvpowerexp1, mvpowerexp2: beta = 1 is the Gaussian") {
  namespace lp = laplace::pdmat;
  using lc = mvpowerexp_test::log_chol;
  mvpowerexp_test::gaussian_limit(lc(3), {0.2, 0.4, -0.3, 0.1, 0.5, -0.2}, {0.3, -1.1, 0.7}, {0.1, 0.2, -0.4},
                                  "log_cholesky d=3");
  mvpowerexp_test::gaussian_limit(lp::diagonal_pdmat(2), {0.3, -0.5}, {0.8, -0.2}, {0.1, 0.4}, "diagonal d=2");
  mvpowerexp_test::gaussian_limit(lp::ar1_pdmat(3), {0.4}, {0.8, -0.2, 1.3}, {0.1, 0.4, -0.2}, "ar1 d=3");
}
TEST_CASE("mvpowerexp1 at theta against mvpowerexp2 at -theta (diagonal)") { mvpowerexp_test::mvpowerexp_twin_case(); }
TEST_CASE("mvpowerexp1, mvpowerexp2: the theta table gives the blocks to the bit") {
  mvpowerexp_test::mvpowerexp_table_case();
}
TEST_CASE("mvpowerexp: log Z and the mass against the radial quadrature of the definition") {
  mvpowerexp_test::mvpowerexp_constant_case();
}
TEST_CASE("mvpowerexp1: beta = 1/2 with Sigma = sigma^2 I is the group lasso") {
  mvpowerexp_test::mvpowerexp_group_lasso_case();
}
TEST_CASE("mvpowerexp1: the radial kink at y = mu") { mvpowerexp_test::mvpowerexp_kink_case(); }
TEST_CASE("mvpowerexp1: the derivatives in beta against central differences") {
  mvpowerexp_test::mvpowerexp_differences_case();
}
TEST_CASE("mvpowerexp1, mvpowerexp2: draws") {
  namespace ld = laplace::distrib;
  using lc = mvpowerexp_test::log_chol;
  for (double beta : {0.5, 1.0, 2.0}) {
    const std::vector<double> th = {0.2, 0.4, -0.3, 0.1, 0.5, -0.2, beta};
    mvpowerexp_test::draws_check(ld::mvpowerexp1_distrib<lc>(lc(3)), th, "mvpowerexp1 log_cholesky d=3");
    mvpowerexp_test::draws_check(ld::mvpowerexp2_distrib<lc>(lc(3)), th, "mvpowerexp2 log_cholesky d=3");
  }
}
TEST_CASE("mvpowerexp1: the typed parameter list and the block helpers") {
  namespace lp = laplace::pdmat;
  namespace ld = laplace::distrib;
  using lc = mvpowerexp_test::log_chol;
  using T = ld::mvpowerexp1_distrib<lc>;
  static_assert(ld::multivariate_distrib<T>);
  static_assert(ld::multivariate_distrib<ld::mvpowerexp2_distrib<lc>>);
  static_assert(T::normalization == ld::distrib_normalization::proper);
  static_assert(T::n_params == 3 && T::params[0].kind == ld::distrib_mv_param_kind::vector &&
                T::params[1].kind == ld::distrib_mv_param_kind::matrix &&
                T::params[2].kind == ld::distrib_mv_param_kind::scalar);
  // mu carries the radial kink: regularity 0, the worst case over beta > 0
  static_assert(T::params[0].regularity == 0);
  static_assert(T::params[1].regularity == ld::infinitely_smooth);
  static_assert(T::params[2].regularity == ld::infinitely_smooth);
  static_assert(T::params[2].lower == 0.0 && T::params[2].upper == ld::distrib_infinity);
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
  CHECK(g.dim() == d);
  CHECK(g.y_coord_size() == d);
  CHECK(ld::mv_n_blocks(g) == 1);
  CHECK(ld::mv_block_tensor_size<2>(g) == lp::packed_size(d));
  CHECK(ld::mv_block_tensor_size(g, ld::y_slot, 3) == ld::sym_tensor_size(d, 3));
  const std::vector<double> th = {0.2, 0.4, -0.3, 0.1, 0.5, -0.2, 0.7}, y = {0.3, -1.1, 0.7}, mu = {0.1, 0.2, -0.4};
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
  // the start value puts beta at 1, the Gaussian
  std::vector<double> start(q + 1);
  ld::distrib_mv_sample_summary sm{};
  sm.count = 100.0;
  sm.dim = d;
  sm.mean.assign(d, 0.0);
  sm.covariance.assign(lp::packed_size(d), 0.0);
  for (int k = 0; k < d; ++k) sm.covariance[lp::packed_index(k, k)] = 1.0;
  g.start_value(sm, start.data());
  CHECK(start[q] == 1.0);
  // d_y^J = (-1)^J d_mu^J
  std::vector<double> t2(lp::packed_size(d)), c2(lp::packed_size(d));
  REQUIRE(g.logpdf_dk<0, 2>(st, y.data(), mu.data(), nullptr, t2.data(), ws));
  REQUIRE(g.logpdf_dy_dk<2, 0>(st, y.data(), mu.data(), nullptr, c2.data(), ws));
  CHECK(t2 == c2);
  std::vector<double> t1(d), c1(d);
  REQUIRE(g.logpdf_dk<0, 1>(st, y.data(), mu.data(), nullptr, t1.data(), ws));
  REQUIRE(g.logpdf_dy_dk<1, 0>(st, y.data(), mu.data(), nullptr, c1.data(), ws));
  for (int k = 0; k < d; ++k) CHECK(c1[k] == -t1[k]);
  // logpdf_dy_dk<J, K, A> with A > 0 goes through location_dy_dk: d_y^J d_mu^A
  // is (-1)^J d_mu^(J + A), the y factor first in the Kronecker product
  std::vector<double> wy(g.scratch_size() + ld::sym_tensor_size(d, 3));
  std::vector<double> t3(ld::sym_tensor_size(d, 3)), e3(lp::packed_size(d) * d), c11(d * d);
  // its own buffer: t2 holds the (A = 2, K = 0) block, compared with the y
  // differences below, and must not be overwritten here
  std::vector<double> t2b(lp::packed_size(d));
  REQUIRE(g.logpdf_dk<1, 2>(st, y.data(), mu.data(), idx + 1, t2b.data(), ws));
  REQUIRE(g.logpdf_dy_dk<1, 1, 1>(st, y.data(), mu.data(), idx + 1, c11.data(), wy.data()));
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) CHECK(c11[i * d + j] == -t2b[lp::packed_index(i > j ? i : j, i > j ? j : i)]);
  REQUIRE(g.logpdf_dk<0, 3>(st, y.data(), mu.data(), nullptr, t3.data(), ws));
  REQUIRE(g.logpdf_dy_dk<2, 0, 1>(st, y.data(), mu.data(), nullptr, e3.data(), wy.data()));
  for (int i = 0; i < d; ++i)  // the y pair (i, j), j <= i, packed by rows; then k
    for (int j = 0; j <= i; ++j)
      for (int k = 0; k < d; ++k) {
        int sorted[3] = {i, j, k};
        std::sort(sorted, sorted + 3);
        CHECK(e3[lp::packed_index(i, j) * d + k] == t3[ld::sym_tensor_position<d, 3>(sorted)]);
      }
  // against central differences in y of the order below
  const double h = 1e-5;
  std::vector<double> yp(y), ym(y), fp(d), fm(d);
  // gp and gm hold the (J = 0, A = 1) block, d entries; the rest is a sentinel,
  // checked below, so that nothing past the documented block size is written
  const double sentinel = -1.25e300;
  std::vector<double> gp(d + 8, sentinel), gm(d + 8, sentinel);
  for (int i = 0; i < d; ++i) {
    // reset element-wise: gcc 14 reports a false -Wstringop-overflow on the
    // copy-assignment yp = y here (region of size 0 after inlining)
    for (int k = 0; k < d; ++k) yp[k] = ym[k] = y[k];
    yp[i] += h;
    ym[i] -= h;
    REQUIRE(g.logpdf_dy_dk<1, 0>(st, yp.data(), mu.data(), nullptr, fp.data(), ws));
    REQUIRE(g.logpdf_dy_dk<1, 0>(st, ym.data(), mu.data(), nullptr, fm.data(), ws));
    for (int j = 0; j < d; ++j) {
      const int o = lp::packed_index(i > j ? i : j, i > j ? j : i);
      CHECK(t2[o] == doctest::Approx((fp[j] - fm[j]) / (2 * h)).epsilon(1e-6));
    }
    // and the mixed (y, mu) block against the difference in y of d_mu
    REQUIRE(g.logpdf_dy_dk<0, 1, 1>(st, yp.data(), mu.data(), idx + 1, gp.data(), wy.data()));
    REQUIRE(g.logpdf_dy_dk<0, 1, 1>(st, ym.data(), mu.data(), idx + 1, gm.data(), wy.data()));
    for (int j = 0; j < d; ++j) CHECK(c11[i * d + j] == doctest::Approx((gp[j] - gm[j]) / (2 * h)).epsilon(1e-6));
    for (int j = d; j < d + 8; ++j) {
      CHECK(gp[j] == sentinel);
      CHECK(gm[j] == sentinel);
    }
  }
}
