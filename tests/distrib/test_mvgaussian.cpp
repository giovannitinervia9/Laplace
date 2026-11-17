// mvgaussian1_distrib and mvgaussian2_distrib (include/laplace/distrib/) in a
// Kokkos kernel on the default execution space:
//  - against the mpmath references (tools/reference/distrib_mvgaussian.py):
//    blocks (A, K) of the log-density, A <= 2, A + K <= 6, and the expected
//    information in theta to order 4; the error of a block is normwise, in
//    ulps of max(max_entries |ref|, 1) (ulp1N);
//  - the two parametrizations against each other: mvgaussian1 over S at theta
//    and mvgaussian2 over S at -theta are the same law when M(-theta) =
//    M(theta)^-1, which holds for diagonal_pdmat and unstructured_pdmat with
//    matrix_log; then every block in theta agrees up to (-1)^K;
//  - orders 5 and 6 (and the information of orders 1 to 4) against central
//    differences of the order below;
//  - the expected information against the Monte Carlo mean of -d^2 l over draws
//    of rng(), and the draws' mean and covariance against mu and Sigma.
// Set LAPLACE_PRINT_ERRORS=1 to print the tables.
#include <doctest/doctest.h>

#include <laplace/distrib/mvgaussian1.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
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
#include <tuple>
#include <vector>

#ifndef LAPLACE_REFERENCE_DIR
#error "LAPLACE_REFERENCE_DIR must point to tests/data/reference"
#endif

// a named namespace, not an anonymous one: the kernels below must sit in
// functions with external linkage (note/brief_agenti.md)
namespace mvgaussian_test {

namespace ld = laplace::distrib;
namespace lp = laplace::pdmat;
using log_chol = lp::unstructured_pdmat<laplace::bijector::log_cholesky_bijector>;
using matrix_log = lp::unstructured_pdmat<laplace::bijector::matrix_log_bijector>;
using scaled_ar1 = lp::unit_diagonal_scaled_pdmat<lp::ar1_pdmat>;

constexpr int max_d = 6, max_q = 16, max_out = 21;  // packed_size(6)
const char* sname[] = {"diagonal", "ar1", "compound_symmetry", "log_cholesky", "scaled_ar1", "factor_analytic"};

// the block (A, K) of D at (theta, y, mu) into out (zeroed when false); A = -1
// is the expected information in theta at the pair ab
template <class D, int A, int K>
KOKKOS_INLINE_FUNCTION bool block_ak(const D& g, const typename D::state_type& st, const double* y, const double* mu,
                                     const int* idx, const int* ab, double* out, double* w) {
  if constexpr (A < 0) return g.template expected_info_dk<K>(st, ld::theta_slot, ld::theta_slot, ab, nullptr, idx, out, w);
  else return g.template logpdf_dk<K, A>(st, y, mu, idx, out, w);
}
template <class D, int A>
KOKKOS_INLINE_FUNCTION bool block_a(const D& g, const typename D::state_type& st, int K, const double* y,
                                    const double* mu, const int* idx, const int* ab, double* out, double* w) {
  switch (K) {
    case 0: return block_ak<D, A, 0>(g, st, y, mu, idx, ab, out, w);
    case 1: return block_ak<D, A, 1>(g, st, y, mu, idx, ab, out, w);
    case 2: return block_ak<D, A, 2>(g, st, y, mu, idx, ab, out, w);
    case 3: return block_ak<D, A, 3>(g, st, y, mu, idx, ab, out, w);
    case 4: return block_ak<D, A, 4>(g, st, y, mu, idx, ab, out, w);
    case 5:
      if constexpr (A >= 0 && A <= 1) return block_ak<D, A, 5>(g, st, y, mu, idx, ab, out, w);
      break;
    case 6:
      if constexpr (A == 0) return block_ak<D, A, 6>(g, st, y, mu, idx, ab, out, w);
      break;
  }
  return false;
}
template <class D>
KOKKOS_INLINE_FUNCTION void eval_d(const D& g, int A, int K, const double* th, const double* y, const double* mu,
                                   const int* idx, const int* ab, double* out, double* w, bool table) {
  for (int k = 0; k < max_out; ++k) out[k] = 0.0;
  auto st = g.theta_state(th, w);
  double* ws = w + g.state_size();
  if (table && K >= 1) {
    // the theta table of order K after the scratch, filled entry by entry
    const int sc = g.scratch_size() > g.info_scratch_size() ? g.scratch_size() : g.info_scratch_size();
    double* tab = ws + sc;
    for (int e = 0; e < g.theta_table_entries(K); ++e) g.fill_theta_table_entry(st, K, e, tab, ws);
    st = g.with_theta_table(st, K, tab);
  }
  switch (A) {
    case -1: block_a<D, -1>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 0: block_a<D, 0>(g, st, K, y, mu, idx, ab, out, ws); break;
    case 1: block_a<D, 1>(g, st, K, y, mu, idx, ab, out, ws); break;
    default: block_a<D, 2>(g, st, K, y, mu, idx, ab, out, ws); break;
  }
}
template <class S>
KOKKOS_INLINE_FUNCTION void eval_s(const S& s, int param, int A, int K, const double* th, const double* y,
                                   const double* mu, const int* idx, const int* ab, double* out, double* w,
                                   bool table) {
  if (param == 1) eval_d(ld::mvgaussian1_distrib<S>(s), A, K, th, y, mu, idx, ab, out, w, table);
  else eval_d(ld::mvgaussian2_distrib<S>(s), A, K, th, y, mu, idx, ab, out, w, table);
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
    case 5: eval_s(lp::factor_analytic_pdmat(d, 1), param, A, K, th, y, mu, idx, ab, out, w, table); break;
    case 6: eval_s(matrix_log(d), param, A, K, th, y, mu, idx, ab, out, w, table); break;
  }
}

template <class S>
int work_of(const S& s) {
  // state, the larger of the two scratches, a theta table of order 3 (the
  // largest the table check uses)
  const ld::mvgaussian1_distrib<S> g(s);
  return g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()) + g.theta_table_size(3);
}
int work_any(int structure, int d) {
  switch (structure) {
    case 0: return work_of(lp::diagonal_pdmat(d));
    case 1: return work_of(lp::ar1_pdmat(d));
    case 2: return work_of(lp::compound_symmetry_pdmat(d));
    case 3: return work_of(log_chol(d));
    case 4: return work_of(scaled_ar1(lp::ar1_pdmat(d)));
    case 5: return work_of(lp::factor_analytic_pdmat(d, 1));
    default: return work_of(matrix_log(d));
  }
}
int q_any(int structure, int d) {
  switch (structure) {
    case 0: return d;
    case 1: case 2: return 1;
    case 3: case 6: return d * (d + 1) / 2;
    case 4: return d + 1;
    default: return 2 * d;
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

// one evaluation: a block at a point (or, for the information, one pair)
struct Case {
  int param, structure, d, A, K;
  std::vector<double> theta, y, mu;
  std::vector<int> idx;
  int ab[2] = {0, 0};
  std::vector<int> pos;
  std::vector<double> hi, lo;
  std::string point;  // param, structure, d, theta, y, mu as in the file
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
    // a block groups its entries; an information row is a case of its own
    const std::string key = A < 0 ? line : line.substr(0, line.size() - f[8].size() - f[9].size() - f[10].size() - 3);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(cases.size())).first;
      Case c;
      c.point = f[0] + "," + f[1] + "," + f[2] + "," + f[3] + "," + f[4] + "," + f[5];
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
    c.lo.push_back(std::strtod(f[10].c_str(), nullptr));
  }
  return cases;
}

// The cases on the device: meta (param, structure, d, A, K, a, b), the point,
// the tuple; out the value, and for K >= 1 the central difference of order
// K - 1 along the last index of the tuple (the information: of the pair's
// order K - 1 along the last index).
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

// value and difference of every case; sign: the theta of the call is
// sign * theta and param is swapped when swap (the twin)
void evaluate(const DeviceCases& dc, Kokkos::View<double* [max_out]> val, Kokkos::View<double* [max_out]> fd,
              double sign, bool swap, bool table = false) {
  const int n = int(dc.meta.extent(0));
  Kokkos::View<double**, Kokkos::LayoutRight> work("work", n, dc.work);
  auto meta = dc.meta;
  auto theta = dc.theta;
  auto ys = dc.y;
  auto mus = dc.mu;
  auto idxs = dc.idx;
  Kokkos::parallel_for(
      "mvgaussian_cases", n, KOKKOS_LAMBDA(int g) {
        const int param = swap ? 3 - meta(g, 0) : meta(g, 0), s = meta(g, 1), d = meta(g, 2), A = meta(g, 3),
                  K = meta(g, 4);
        const int ab[2] = {meta(g, 5), meta(g, 6)};
        double th[max_q], y[max_d], mu[max_d], a[max_out], b[max_out];
        int idx[6];
        for (int k = 0; k < max_q; ++k) th[k] = sign * theta(g, k);
        for (int k = 0; k < max_d; ++k) y[k] = ys(g, k), mu[k] = mus(g, k);
        for (int k = 0; k < 6; ++k) idx[k] = idxs(g, k);
        double* w = &work(g, 0);
        eval_any(s, d, param, A, K, th, y, mu, idx, ab, a, w, table && K <= 3);
        for (int k = 0; k < max_out; ++k) val(g, k) = a[k];
        if (K == 0 || table) return;
        const int j = idx[K - 1];
        const double x = th[j], h = 1e-5 * (x < 1.0 && x > -1.0 ? 1.0 : (x < 0 ? -x : x));
        th[j] = x + h;
        eval_any(s, d, param, A, K - 1, th, y, mu, idx, ab, a, w);
        th[j] = x - h;
        eval_any(s, d, param, A, K - 1, th, y, mu, idx, ab, b, w);
        for (int k = 0; k < max_out; ++k) fd(g, k) = (a[k] - b[k]) / (2.0 * h);
      });
}

void mvgaussian_mpmath_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvgaussian.csv");
  const int n = int(cases.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> val("val", n), fd("fd", n);
  evaluate(dc, val, fd, 1.0, false);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fd);

  // the scale of a block: the largest |ref| over every tuple and entry of the
  // same point, A and K (the sampled derivative tensor of that order, normwise),
  // and at least 1
  std::map<std::string, double> tensor_scale;
  for (const Case& c : cases) {
    double& t = tensor_scale[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)];
    for (double h : c.hi) t = std::fmax(t, std::fabs(h));
  }
  // worst ulp1N by [param - 1][structure][A + 1][K]
  double worst[2][6][4][7] = {};
  int worst_g[2][6][4][7] = {};
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < n; ++g) {
    const Case& c = cases[g];
    double err = 0.0;
    // the information's derivatives are sampled on a few pairs, which may all
    // be zero: their scale is at least that of the information itself (all
    // pairs at order 0; theta is on an unconstrained scale)
    const double scale =
        std::fmax(std::fmax(1.0, tensor_scale[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)]),
                  c.A < 0 ? tensor_scale[c.point + ",-1,0"] : 0.0);
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs((hv(g, c.pos[r]) - c.hi[r]) - c.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    double& w = worst[c.param - 1][c.structure][c.A + 1][c.K];
    if (!(err / (scale * eps) <= w)) {
      w = err / (scale * eps);
      worst_g[c.param - 1][c.structure][c.A + 1][c.K] = g;
    }
    if (c.K >= 5 || (c.A < 0 && c.K >= 1)) {
      // orders 5, 6 and the information's derivatives against differences;
      // tolerance 1e-6 of the scales of orders K and K - 1 (truncation h^2 f^(K+2)
      // and rounding eps f^(K-1)/h)
      for (std::size_t r = 0; r < c.pos.size(); ++r) {
        const double lower = tensor_scale[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K - 1)];
        // 1e-5 for factor_analytic, whose hard point has large derivatives of
        // order K + 2 (the truncation term)
        const double tol = c.structure == 5 ? 1e-5 : 1e-6;
        const double dd = std::fabs(hf(g, c.pos[r]) - hv(g, c.pos[r])) / (tol * (scale + lower));
        worst_fd = std::fmax(worst_fd, dd);
        CHECK_MESSAGE(dd <= 1.0, "param " << c.param << " " << std::string(sname[c.structure]) << " d " << c.d
                                          << " A " << c.A << " K " << c.K << ": difference check " << dd);
      }
    }
  }
  // bounds on ulp1N: the larger of the errors measured on gcc/OpenMP and CUDA
  // on 2026-11-16, times 1.5, rounded up to two digits. The information of
  // mvgaussian1 over the structures with closed_factor takes the factor form
  // (bounds of 2026-11-16, 07:20). factor_analytic was measured again at 09:50,
  // after its inverse_dk moved to the augmented matrix; the largest left is
  // mvgaussian1, A = 0, K = 5 at psi = e^-6 (118 on gcc, 78.9 on CUDA).
  // [param - 1][structure][A + 1][K], A + 1 = 0 the information in theta
  const double bound[2][6][4][7] = {
   {
      {{1, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}},  // diagonal
      {{2, 3, 2, 9, 25, 0, 0}, {51, 52, 38, 47, 38, 47, 38}, {6, 5, 6, 5, 6, 5, 0}, {1, 1, 1, 1, 1, 0, 0}},  // ar1
      {{3, 1, 1, 2, 2, 0, 0}, {2, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}},  // compound_symmetry
      {{2, 2, 2, 2, 2, 0, 0}, {3, 5, 3, 5, 3, 4, 8}, {4, 3, 2, 3, 5, 9, 0}, {3, 3, 4, 2, 8, 0, 0}},  // log_cholesky
      {{3, 4, 4, 7, 18, 0, 0}, {3, 8, 3, 2, 2, 1, 2}, {2, 5, 3, 1, 2, 1, 0}, {1, 1, 1, 1, 1, 0, 0}},  // scaled_ar1
      {{3, 6, 5, 19, 50, 0, 0}, {2, 8, 3, 7, 9, 180, 3}, {5, 2, 2, 3, 6, 24, 0}, {1, 3, 2, 5, 5, 0, 0}}},  // factor_analytic
   {
      {{2, 0, 1, 1, 1, 0, 0}, {1, 2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}},  // diagonal
      {{1, 4, 11, 34, 200, 0, 0}, {2, 5, 6, 3, 7, 12, 4}, {2, 2, 5, 1, 5, 14, 0}, {2, 2, 3, 2, 5, 0, 0}},  // ar1
      {{3, 1, 2, 2, 5, 0, 0}, {1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}},  // compound_symmetry
      {{2, 2, 1, 1, 1, 0, 0}, {2, 4, 3, 1, 1, 4, 1}, {3, 2, 2, 3, 1, 2, 0}, {3, 3, 3, 3, 1, 0, 0}},  // log_cholesky
      {{3, 2, 1, 4, 6, 0, 0}, {1, 1, 1, 3, 2, 4, 1}, {2, 1, 6, 2, 1, 4, 0}, {2, 1, 3, 2, 8, 0, 0}},  // scaled_ar1
      {{3, 1, 1, 1, 1, 0, 0}, {1, 1, 1, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 0}, {1, 1, 1, 1, 1, 0, 0}}}};  // factor_analytic
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 6; ++s)
      for (int a = 0; a < 4; ++a) {
        if (print) {
          std::printf("mvgaussian%d %-18s A=%2d", p + 1, sname[s], a - 1);
          for (int K = 0; K <= 6; ++K) std::printf(" %9.3g", worst[p][s][a][K]);
          std::printf("   worst points (d, theta_0) by K:");
          for (int K = 0; K <= 6; ++K)
            if (worst[p][s][a][K] > 0.0)
              std::printf(" (%d, %g)", cases[worst_g[p][s][a][K]].d, cases[worst_g[p][s][a][K]].theta[0]);
          std::printf("\n");
        }
        for (int K = 0; K <= 6; ++K)
          CHECK_MESSAGE(worst[p][s][a][K] <= bound[p][s][a][K], "mvgaussian" << p + 1 << " " << std::string(sname[s]) << " A "
                                                                             << a - 1 << " K " << K << ": "
                                                                             << worst[p][s][a][K] << " ulp1N");
      }
  if (print) std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
}

// mvgaussian1 over S at theta against mvgaussian2 over S at -theta, for the
// structures with M(-theta) = M(theta)^-1: the cases of the diagonal structure
// from the reference file, and the same points with matrix_log (d (d + 1)/2
// parameters, the first ones of the point when there are fewer).
void mvgaussian_twin_case() {
  auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvgaussian.csv");
  std::vector<Case> twins;
  for (const Case& c : cases) {
    if (c.structure != 0 && c.structure != 3) continue;
    Case t = c;
    if (c.structure == 3) {
      t.structure = 6;
      if (c.d > 3) continue;
    }
    twins.push_back(t);
  }
  const int n = int(twins.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(twins);
  Kokkos::View<double* [max_out]> v1("v1", n), v2("v2", n), fd("fd", n);
  evaluate(dc, v1, fd, 1.0, false);
  evaluate(dc, v2, fd, -1.0, true);
  auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v1);
  auto h2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v2);
  double worst[2][4][7] = {};  // [diagonal, matrix_log][A + 1][K]
  for (int g = 0; g < n; ++g) {
    const Case& c = twins[g];
    const double sgn = c.K % 2 ? -1.0 : 1.0;
    const int m = c.A == 1 ? c.d : c.A == 2 ? lp::packed_size(c.d) : 1;
    double scale = 1.0, err = 0.0;
    for (int k = 0; k < m; ++k) scale = std::fmax(scale, std::fabs(h1(g, k)));
    for (int k = 0; k < m; ++k) err = std::fmax(err, std::fabs(h1(g, k) - sgn * h2(g, k)));
    double& w = worst[c.structure == 6][c.A + 1][c.K];
    w = std::fmax(w, err / (scale * std::ldexp(1.0, -52)));
  }
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int s = 0; s < 2; ++s)
    for (int a = 0; a < 4; ++a) {
      if (print) {
        std::printf("twin %-10s A=%2d", s ? "matrix_log" : "diagonal", a - 1);
        for (int K = 0; K <= 6; ++K) std::printf(" %9.2f", worst[s][a][K]);
        std::printf("\n");
      }
      // both structures compute M^-1(theta) as M(-theta) (matrix_log negates
      // theta and calls its matrix map), so the twins agree to the bit; measured
      // 0 on gcc and CUDA, bound 1
      for (int K = 0; K <= 6; ++K)
        CHECK_MESSAGE(worst[s][a][K] <= 1.0, "twin " << s << " A " << a - 1 << " K " << K << ": "
                                                                    << worst[s][a][K] << " ulp1N");
    }
}


// The theta table: every block of orders 1 to 3 read from a table filled
// once per point must equal, to the bit, the block computed directly (the
// same arithmetic; the information in theta does not read the table).
void mvgaussian_table_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvgaussian.csv");
  const int n = int(cases.size());
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> v0("v0", n), v1("v1", n), fd("fd", n);
  evaluate(dc, v0, fd, 1.0, false, false);
  evaluate(dc, v1, fd, 1.0, false, true);
  auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v0);
  auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v1);
  int checked = 0, differ = 0;
  for (int g = 0; g < n; ++g) {
    if (cases[g].K < 1 || cases[g].K > 3) continue;
    ++checked;
    for (int k = 0; k < max_out; ++k)
      if (h0(g, k) != h1(g, k)) {
        ++differ;
        break;
      }
  }
  CHECK(checked > 0);
  CHECK_MESSAGE(differ == 0, differ << " of " << checked << " blocks differ with the theta table");
}

// Draws: n units from one theta, shared by all (its state computed once on the
// device). The sample mean and covariance (summarize_mv_sample) against mu and
// the covariance, and the mean over the draws of -d^2 l in (theta, theta) and
// in (mu, theta) against the expected information, within 5 standard errors
// estimated from the same draws; the (mu, mu) block against -d^2 l/dmu^2,
// which does not depend on y.
template <class D>
void mvgaussian_draws_check(const D& g, const std::vector<double>& theta_h, const char* name, bool exact_start) {
  using St = typename D::state_type;
  const int d = g.dim(), q = g.theta_size(), np = lp::packed_size(d), nq = lp::packed_size(q);
  const int n = 200000, chunk = 20000, nv = d + nq + d * q;
  Kokkos::View<double*> theta("theta", q), mu("mu", d), state("state", g.state_size() + std::max(g.scratch_size(), g.info_scratch_size()));
  Kokkos::View<St> st("st");
  auto ht = Kokkos::create_mirror_view(theta);
  auto hm = Kokkos::create_mirror_view(mu);
  for (int k = 0; k < q; ++k) ht(k) = theta_h[k];
  for (int k = 0; k < d; ++k) hm(k) = 0.5 * k - 0.3;
  Kokkos::deep_copy(theta, ht);
  Kokkos::deep_copy(mu, hm);
  // the information on the device: (theta, theta) pairs, then the (mu, mu)
  // block and -d^2 l/dmu^2 at mu
  Kokkos::View<double*> info("info", nq + 2 * np);
  Kokkos::parallel_for(
      "mv_state", 1, KOKKOS_LAMBDA(int) {
        st() = g.theta_state(theta.data(), state.data());
        double* ws = state.data() + g.state_size();
        for (int a = 0; a < q; ++a)
          for (int b = 0; b <= a; ++b) {
            const int ab[2] = {a, b};
            double r = 0.0;
            if (!g.template expected_info_dk<0>(st(), ld::theta_slot, ld::theta_slot, ab, nullptr, nullptr, &r, ws))
              r = 0.0;
            info(lp::packed_index(a, b)) = r;
          }
        g.template expected_info_dk<0>(st(), 0, 0, nullptr, nullptr, nullptr, &info(nq), ws);
        g.template logpdf_dk<0, 2>(st(), mu.data(), mu.data(), nullptr, &info(nq + np), ws);
      });
  Kokkos::View<double**, Kokkos::LayoutRight> v("v", n, nv);
  Kokkos::View<double**, Kokkos::LayoutRight> work("work", chunk, g.scratch_size() + 64);
  for (int c0 = 0; c0 < n; c0 += chunk) {
    Kokkos::parallel_for(
        "mv_draws", chunk, KOKKOS_LAMBDA(int c) {
          const int i = c0 + c;
          double* y = &v(i, 0);
          double* ws = &work(c, 0);
          g.rng(st(), mu.data(), 20261115ULL, std::uint64_t(i), y);
          for (int a = 0; a < q; ++a)
            for (int b = 0; b <= a; ++b) {
              const int ab[2] = {b, a};
              double r = 0.0;
              if (!g.template logpdf_dk<2, 0>(st(), y, mu.data(), ab, &r, ws)) r = 0.0;
              v(i, d + lp::packed_index(a, b)) = -r;
            }
          double m[max_d];
          for (int a = 0; a < q; ++a) {
            const int t[1] = {a};
            if (!g.template logpdf_dk<1, 1>(st(), y, mu.data(), t, m, ws))
              for (int k = 0; k < d; ++k) m[k] = 0.0;
            for (int k = 0; k < d; ++k) v(i, d + nq + a * d + k) = -m[k];
          }
        });
  }
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), info);

  // covariance of the law, from the structure on the host
  std::vector<double> cov(np), ws(g.scratch_size() + 64);
  g.covariance(theta_h.data(), cov.data(), ws.data());
  std::vector<double> ys(std::size_t(n) * d);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < d; ++k) ys[std::size_t(i) * d + k] = hv(i, k);
  const auto sum = ld::summarize_mv_sample(ys.data(), n, d);
  double worst = 0.0;
  for (int i = 0; i < d; ++i) {
    const double z = (sum.mean[i] - hm(i)) / std::sqrt(cov[lp::packed_index(i, i)] / n);
    if (std::getenv("LAPLACE_PRINT_ERRORS"))
      for (int j = 0; j <= i; ++j)
        std::printf("  %s cov(%d,%d) sample %.6g law %.6g\n", name, i, j, sum.covariance[lp::packed_index(i, j)],
                    cov[lp::packed_index(i, j)]);
    worst = std::fmax(worst, std::fabs(z));
    for (int j = 0; j <= i; ++j) {
      const double sij = cov[lp::packed_index(i, j)];
      const double se = std::sqrt((cov[lp::packed_index(i, i)] * cov[lp::packed_index(j, j)] + sij * sij) / n);
      const double zc = (sum.covariance[lp::packed_index(i, j)] - sij) / se;
      worst = std::fmax(worst, std::fabs(zc));
    }
  }
  // -d2 l columns: mean and standard error against the information (zero for
  // the mixed block)
  for (int c = d; c < nv; ++c) {
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i) s += hv(i, c);
    const double mean = s / n;
    for (int i = 0; i < n; ++i) s2 += (hv(i, c) - mean) * (hv(i, c) - mean);
    const double se = std::sqrt(s2 / n / n), ref = c < d + nq ? hi(c - d) : 0.0;
    // a column that does not depend on y spreads by rounding only: its
    // standard error is floored at 1e-12 (1 + |ref|)
    const double z = (mean - ref) / std::fmax(se, 1e-12 * (1.0 + std::fabs(ref)));
    if (std::fabs(z) > 5.0 && std::getenv("LAPLACE_PRINT_ERRORS"))
      std::printf("  %s column %d: mean %.6g ref %.6g se %.3g\n", name, c, mean, ref, se);
    worst = std::fmax(worst, std::fabs(z));
  }
  double mumu = 0.0;
  for (int k = 0; k < np; ++k) mumu = std::fmax(mumu, std::fabs(hi(nq + k) + hi(nq + np + k)));
  if (std::getenv("LAPLACE_PRINT_ERRORS"))
    std::printf("draws %-28s worst |z| %.2f, (mu, mu) block %.3g\n", name, worst, mumu);
  CHECK_MESSAGE(worst <= 5.0, name << ": worst |z| " << worst);
  CHECK_MESSAGE(mumu == 0.0, name << ": (mu, mu) information against -d2 l/dmu2: " << mumu);
}

void mvgaussian_draws_case() {
  const std::vector<double> lc = {0.3, 0.8, -0.2, -0.4, 0.5, -1.0};
  mvgaussian_draws_check(ld::mvgaussian1_distrib<log_chol>(log_chol(3)), lc, "mvgaussian1 log_cholesky d=3", true);
  mvgaussian_draws_check(ld::mvgaussian2_distrib<log_chol>(log_chol(3)), lc, "mvgaussian2 log_cholesky d=3", true);
  const std::vector<double> fa = {0.2, 0.9, -0.6, -0.5, 0.1, -1.0};
  mvgaussian_draws_check(ld::mvgaussian1_distrib<lp::factor_analytic_pdmat>(lp::factor_analytic_pdmat(3, 1)), fa,
                         "mvgaussian1 factor_analytic d=3", false);
  mvgaussian_draws_check(ld::mvgaussian2_distrib<lp::factor_analytic_pdmat>(lp::factor_analytic_pdmat(3, 1)), fa,
                         "mvgaussian2 factor_analytic d=3", false);
}

// Against the frozen statmodels7 results (tools/reference/statmodels7_multivariate.R):
// log_cholesky, d = 2, 3, 4 and a near-singular point (log L33 = -8), blocks
// with A + K <= 4 and the information at order 0. The values are statmodels7's
// doubles, so the error is the sum of both libraries' errors; ulp1N as above.
void mvgaussian_s7_case() {
  const auto cases = read_cases(std::string(LAPLACE_REFERENCE_DIR) + "/mvgaussian_s7.csv");
  const int n = int(cases.size());
  REQUIRE(n > 0);
  const DeviceCases dc = upload(cases);
  Kokkos::View<double* [max_out]> val("val", n), fd("fd", n);
  evaluate(dc, val, fd, 1.0, false);
  auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
  std::map<std::string, double> tensor_scale;
  for (const Case& c : cases) {
    double& t = tensor_scale[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)];
    for (double h : c.hi) t = std::fmax(t, std::fabs(h));
  }
  // worst ulp1N by [param - 1][near singular][A + 1][K]
  double worst[2][2][4][5] = {};
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < n; ++g) {
    const Case& c = cases[g];
    const double scale = std::fmax(1.0, tensor_scale[c.point + "," + std::to_string(c.A) + "," + std::to_string(c.K)]);
    double err = 0.0;
    for (std::size_t r = 0; r < c.pos.size(); ++r) {
      const double e = std::fabs(hv(g, c.pos[r]) - c.hi[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const bool near_singular = c.theta.back() < -4.0;
    // The information (A = -1) at the near-singular point is left out: statmodels7
    // is 2.2e7 (mvgaussian1) and 2.0e7 (mvgaussian2) ulp1N from mpmath there, and
    // Laplace's own error at that point is covered by the mpmath case.
    if (near_singular && c.A == -1) continue;
    double& w = worst[c.param - 1][near_singular][c.A + 1][c.K];
    w = std::fmax(w, err / (scale * eps));
  }
  // bounds: the larger of the errors measured on gcc/OpenMP and CUDA on
  // 2026-11-16, times 1.5, rounded up (at least 1); [param - 1][near
  // singular][A + 1][K]; the near-singular information is not compared (above).
  const double bound[2][2][4][5] = {
      {{{3, 0, 0, 0, 0}, {2, 6, 5, 4, 8}, {3, 4, 4, 4, 0}, {2, 3, 3, 0, 0}},  // mvgaussian1, regular
       {{0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, {2, 2, 2, 2, 0}, {2, 2, 2, 0, 0}}},  // near singular
      {{{6, 0, 0, 0, 0}, {2, 2, 2, 3, 3}, {2, 2, 2, 2, 0}, {2, 1, 1, 0, 0}},  // mvgaussian2, regular
       {{0, 0, 0, 0, 0}, {1, 1, 1, 2, 4}, {1, 1, 1, 1, 0}, {1, 1, 1, 0, 0}}}};  // near singular
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 2; ++s)
      for (int a = 0; a < 4; ++a) {
        if (print) {
          std::printf("s7 mvgaussian%d %-13s A=%2d", p + 1, s ? "near-singular" : "regular", a - 1);
          for (int K = 0; K <= 4; ++K) std::printf(" %9.3g", worst[p][s][a][K]);
          std::printf("\n");
        }
        for (int K = 0; K <= 4; ++K)
          CHECK_MESSAGE(worst[p][s][a][K] <= bound[p][s][a][K], "s7 mvgaussian" << p + 1 << (s ? " near-singular" : "")
                                                                                << " A " << a - 1 << " K " << K << ": "
                                                                                << worst[p][s][a][K] << " ulp1N");
      }
}

}  // namespace mvgaussian_test

TEST_CASE("mvgaussian1, mvgaussian2: log_cholesky blocks against frozen statmodels7 results") {
  mvgaussian_test::mvgaussian_s7_case();
}
TEST_CASE("mvgaussian1, mvgaussian2: blocks and expected information against mpmath and differences") {
  mvgaussian_test::mvgaussian_mpmath_case();
}
TEST_CASE("mvgaussian1 at theta against mvgaussian2 at -theta (diagonal, matrix_log)") {
  mvgaussian_test::mvgaussian_twin_case();
}
TEST_CASE("mvgaussian1, mvgaussian2: the theta table gives the blocks to the bit") {
  mvgaussian_test::mvgaussian_table_case();
}
TEST_CASE("mvgaussian1, mvgaussian2: draws, and the expected information against Monte Carlo") {
  mvgaussian_test::mvgaussian_draws_case();
}
