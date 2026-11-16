// reduced_rank_psdmat (include/laplace/pdmat/reduced_rank.hpp): M, Lambda and
// log|M|_+ with their derivatives to order 6 against the mpmath references
// (tools/reference/pdmat_reduced_rank.py) and against central differences of
// the derivative one order below, in a Kokkos kernel on the default execution
// space; on the host, the twins: M is factor_analytic_pdmat's M - Psi at the
// same loadings, and log|M|_+ is 2 sum_j theta_jj + log det(I + W^T W),
// W = B L1^-1 (L1 the top d x d block of Lambda, B the rest). Errors:
//   ulpN   for M and Lambda: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|,
//          or, when that derivative vanishes, in ulps of max_ij of the value
//          of the quantity at the same point;
//   ulp    for log|M|_+: relative, in ulps of max(|ref|, 1).
// A reference below 1e-25 in absolute value is read as 0. The difference
// check: along the last index v of the tuple,
// (f^(K-1)(theta + h e_v) - f^(K-1)(theta - h e_v))/(2h), h = 1e-5, within
// 1e-6 of the largest entry plus 1e-6 of that of order K-1.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/pdmat/factor_analytic.hpp>
#include <laplace/pdmat/reduced_rank.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
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

namespace {

namespace lp = laplace::pdmat;
using rr_psdmat = lp::reduced_rank_psdmat;

constexpr int max_p = 5;
constexpr int max_out = 20;  // p d at (5, 4)
constexpr int max_n = 14;    // parameters at (5, 4)

int out_size(int p, int d, int quantity) { return quantity == 0 ? lp::packed_size(p) : (quantity == 1 ? p * d : 1); }

template <int K>
KOKKOS_INLINE_FUNCTION bool eval_k(const rr_psdmat& s, int quantity, const int* tup, const double* u, double* out,
                                   double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = tup[k];
  if (quantity == 0) return s.template matrix_dk<K>(u, idx, out, w);
  if (quantity == 1) return s.template factor_dk<K>(u, idx, out, w);
  out[0] = s.template log_pdet_dk<K>(u, idx, w);
  return true;
}

// out = the quantity along the tuple of order K; zeroed if identically zero
KOKKOS_INLINE_FUNCTION void eval(const rr_psdmat& s, int quantity, int K, const int* tup, const double* u, double* out,
                                 double* w) {
  const int n = quantity == 0 ? lp::packed_size(s.dim) : (quantity == 1 ? s.dim * s.n_factors : 1);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.matrix(u, out, w);
      else if (quantity == 1) s.factor(u, out, w);
      else out[0] = s.log_pdet(u, w);
      break;
    case 1: nz = eval_k<1>(s, quantity, tup, u, out, w); break;
    case 2: nz = eval_k<2>(s, quantity, tup, u, out, w); break;
    case 3: nz = eval_k<3>(s, quantity, tup, u, out, w); break;
    case 4: nz = eval_k<4>(s, quantity, tup, u, out, w); break;
    case 5: nz = eval_k<5>(s, quantity, tup, u, out, w); break;
    default: nz = eval_k<6>(s, quantity, tup, u, out, w); break;
  }
  if (!nz)
    for (int k = 0; k < n; ++k) out[k] = 0.0;
}

struct Group {
  int p, d, quantity, order;
  std::string ukey;
  std::vector<int> tup;
  std::vector<double> u;
  std::vector<int> pos;
  std::vector<double> hi, lo;
};

std::vector<double> split_hex(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, ';')) v.push_back(std::strtod(f.c_str(), nullptr));
  return v;
}

std::vector<Group> read_groups(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::tuple<int, int, std::string, int, std::string>, int> where;
  std::vector<Group> g;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) f.push_back(x);
    const int p = std::stoi(f[0]), d = std::stoi(f[1]), q = std::stoi(f[3]), K = std::stoi(f[4]);
    const auto key = std::make_tuple(p, d, f[2], q, f[5]);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(g.size())).first;
      Group G{p, d, q, K, f[2], {}, split_hex(f[2]), {}, {}, {}};
      if (K > 0)
        for (double t : split_hex(f[5])) G.tup.push_back(int(t));
      g.push_back(G);
    }
    Group& gr = g[it->second];
    const int i = std::stoi(f[6]), j = std::stoi(f[7]);
    gr.pos.push_back(q == 0 ? lp::packed_index(i, j) : (q == 1 ? i * d + j : 0));
    gr.hi.push_back(std::strtod(f[8].c_str(), nullptr));
    gr.lo.push_back(std::strtod(f[9].c_str(), nullptr));
  }
  return g;
}

const char* qname[] = {"matrix", "factor", "log_pdet"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE (note/brief_agenti.md: nvcc tags the device lambda with its enclosing function).
void pdmat_reduced_rank_mpmath_case() {
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_reduced_rank.csv");
  const int ng = int(groups.size());
  REQUIRE(ng > 0);
  int sw = 0;
  for (int p = 1; p <= max_p; ++p)
    for (int d = 1; d <= p; ++d) sw = std::max(sw, rr_psdmat(p, d).scratch_size());

  Kokkos::View<int* [10]> meta("meta", ng);  // p, d, quantity, order, tuple[6]
  Kokkos::View<double* [max_n]> uu("u", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto hu = Kokkos::create_mirror_view(uu);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.p, hm(g, 1) = G.d, hm(g, 2) = G.quantity, hm(g, 3) = G.order;
    for (int k = 0; k < 6; ++k) hm(g, 4 + k) = k < G.order ? G.tup[k] : 0;
    for (int k = 0; k < max_n; ++k) hu(g, k) = k < int(G.u.size()) ? G.u[k] : 0.0;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(uu, hu);

  // y: the value; fd: the difference of order K - 1; lower: order K - 1 at theta
  Kokkos::View<double* [max_out]> y("y", ng), fd("fd", ng), lower("lower", ng);
  constexpr int chunk = 2048;
  Kokkos::View<double**, Kokkos::LayoutRight> scratch("scratch", chunk, sw);  // a contiguous row per group
  for (int g0 = 0; g0 < ng; g0 += chunk) {
    const int g1 = std::min(ng, g0 + chunk);
    Kokkos::parallel_for(
        "pdmat_reduced_rank", Kokkos::RangePolicy<>(g0, g1), KOKKOS_LAMBDA(int g) {
          const rr_psdmat s(meta(g, 0), meta(g, 1));
          const int q = meta(g, 2), K = meta(g, 3);
          int tup[6];
          for (int k = 0; k < 6; ++k) tup[k] = meta(g, 4 + k);
          double u[max_n], a[max_out], b[max_out];
          for (int k = 0; k < max_n; ++k) u[k] = uu(g, k);
          for (int k = 0; k < max_out; ++k) a[k] = b[k] = 0.0;
          double* w = &scratch(g - g0, 0);
          eval(s, q, K, tup, u, a, w);
          for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
          if (K == 0) return;
          eval(s, q, K - 1, tup, u, a, w);
          for (int k = 0; k < max_out; ++k) lower(g, k) = a[k];
          const int v = tup[K - 1];
          const double x = u[v], h = 1e-5;
          u[v] = x + h;
          eval(s, q, K - 1, tup, u, a, w);
          u[v] = x - h;
          eval(s, q, K - 1, tup, u, b, w);
          for (int k = 0; k < max_out; ++k) fd(g, k) = (a[k] - b[k]) / (2.0 * h);
        });
    Kokkos::fence();
  }
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hfd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fd);
  auto hlo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower);

  // max |value| of M and Lambda at each point, the scale of a vanishing derivative
  std::map<std::tuple<int, int, std::string>, double> value_scale;
  for (const Group& G : groups)
    if (G.quantity < 2 && G.order == 0) {
      double s = 0.0;
      for (double h : G.hi) s = std::fmax(s, std::fabs(h));
      value_scale[{G.d, G.quantity, G.ukey}] = s;
    }
  double worst[3][7] = {};
  int worst_g[3][7] = {};
  std::map<std::tuple<int, int, std::string>, std::array<double, 3>> by_point;  // over the orders
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int n = out_size(G.p, G.d, G.quantity);
    double scale = 0.0, err = 0.0;
    auto ref = [&](std::size_t r) { return std::fabs(G.hi[r]) < 1e-25 ? 0.0 : G.hi[r]; };
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(ref(r)));
    scale = G.quantity == 2 ? std::fmax(scale, 1.0)
                            : (scale > 0.0 ? scale : value_scale.at({G.d, G.quantity, G.ukey}));
    for (std::size_t r = 0; r < G.pos.size(); ++r) {
      const double lo = ref(r) == 0.0 ? 0.0 : G.lo[r];
      const double e = std::fabs((hy(g, G.pos[r]) - ref(r)) - lo);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double ulp = err / (scale * eps);
    double& w = worst[G.quantity][G.order];
    if (!(ulp <= w)) w = ulp, worst_g[G.quantity][G.order] = g;
    double& wp = by_point[{G.p, G.d, G.ukey}][G.quantity];
    wp = std::fmax(wp, ulp);
    if (G.order > 0) {
      double lscale = 0.0;
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double dd = std::fabs(hfd(g, k) - hy(g, k)) / (1e-6 * (scale + lscale));
        worst_fd = std::fmax(worst_fd, dd);
        if (!(dd <= 1.0))
          FAIL_CHECK(std::string(qname[G.quantity]) << " order " << G.order << " p " << G.p << " d " << G.d
                                       << ": difference check " << dd);
      }
    }
  }

  // bounds on ulpN / ulp, [quantity][order]: the larger of the errors measured
  // on 2026-11-16 on gcc/OpenMP and CUDA, times 1.5, rounded up. log|M|_+ for
  // d < p runs in double-double from Lambda^T Lambda onward; in double the
  // recursion lost up to 3850 ulp at orders 5-6 (terms of 1e4 cancelling to
  // 1e-2 at the spread point of (p, d) = (5, 4)).
  const double bound[3][7] = {{1, 1, 1, 1, 1, 1, 1},
                              {1, 1, 1, 1, 1, 1, 1},
                              {3, 1, 2, 10, 5, 4, 19}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int q = 0; q < 3; ++q)
    for (int K = 0; K <= 6; ++K) {
      const Group& G = groups[worst_g[q][K]];
      std::string tup;
      for (int t : G.tup) tup += std::to_string(t) + " ";
      if (print)
        std::printf("reduced_rank %-9s K=%d  %10.2f  (p %d d %d, theta0 %g, tuple %s)\n", qname[q], K, worst[q][K], G.p,
                    G.d, G.u.empty() ? 0.0 : G.u[0], tup.c_str());
      CHECK_MESSAGE(worst[q][K] <= bound[q][K],
                    qname[q] << " order " << K << ": " << worst[q][K] << " ulp at p " << G.p << " d " << G.d);
    }
  if (print) {
    std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
    for (const auto& [key, e] : by_point)
      std::printf("p %d d %d theta0 %-8.3g  matrix %10.2f  factor %10.2f  log_pdet %10.2f\n", std::get<0>(key),
                  std::get<1>(key), split_hex(std::get<2>(key))[0], e[0], e[1], e[2]);
  }
}

TEST_CASE("reduced_rank_psdmat: M, Lambda and log|M|_+ against mpmath and differences") {
  pdmat_reduced_rank_mpmath_case();
}

namespace {

std::vector<double> point(int n) {
  std::vector<double> u(n);
  for (int k = 0; k < n; ++k) u[k] = 0.37 * std::sin(1.3 * k + 0.4) + 0.05 * k;
  return u;
}

// 2 sum_j theta_jj + log det(I + W^T W), W = B L1^-1
double log_pdet_twin(const rr_psdmat& s, const std::vector<double>& u) {
  const int p = s.dim, d = s.n_factors;
  std::vector<double> lam(p * d), w(s.scratch_size());
  s.factor(u.data(), lam.data(), w.data());
  double r = 0.0;
  for (int j = 0; j < d; ++j) r += 2.0 * std::log(lam[j * d + j]);
  // W^T row by row: solve L1^T x = b_i^T for each row b_i of B, i.e. w_i = b_i L1^-1
  std::vector<double> C(d * d, 0.0);
  for (int a = 0; a < d; ++a) C[a * d + a] = 1.0;
  for (int i = d; i < p; ++i) {
    std::vector<double> x(d);
    for (int j = d - 1; j >= 0; --j) {  // x L1 = b: x_j = (b_j - sum_(k > j) x_k L1_kj) / L1_jj
      double t = lam[i * d + j];
      for (int k = j + 1; k < d; ++k) t -= x[k] * lam[k * d + j];
      x[j] = t / lam[j * d + j];
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) C[a * d + b] += x[a] * x[b];
  }
  lp::detail_pdmat::fa_cholesky(d, C.data());
  for (int k = 0; k < d; ++k) r += 2.0 * std::log(C[k * d + k]);
  return r;
}

}  // namespace

TEST_CASE("reduced_rank_psdmat: M is factor_analytic's M - Psi, log|M|_+ by the twin") {
  const int shapes[][2] = {{1, 1}, {3, 1}, {4, 2}, {5, 2}, {6, 3}, {4, 4}, {7, 5}};
  for (const auto& sh : shapes) {
    const int p = sh[0], d = sh[1], np = lp::packed_size(p);
    const rr_psdmat s(p, d);
    const lp::factor_analytic_pdmat f(p, d);
    REQUIRE(s.n_params() == f.n_loadings());
    CHECK(s.rank() == d);
    std::vector<double> u = point(f.n_params()), w(s.scratch_size()), wf(f.scratch_size()), x(np), y(np);
    s.matrix(u.data(), x.data(), w.data());
    f.matrix(u.data(), y.data(), wf.data());
    for (int i = 0; i < p; ++i) y[lp::packed_index(i, i)] -= std::exp(2.0 * u[s.n_params() + i]);
    for (int k = 0; k < np; ++k) CHECK(x[k] == doctest::Approx(y[k]).epsilon(1e-14).scale(1.0));
    const int i2[2] = {0, s.n_params() - 1};
    std::fill(x.begin(), x.end(), 0.0), std::fill(y.begin(), y.end(), 0.0);
    CHECK(s.matrix_d2(u.data(), i2, x.data(), w.data()) == f.matrix_d2(u.data(), i2, y.data(), wf.data()));
    CHECK(x == y);
    const std::vector<double> ul(u.begin(), u.begin() + s.n_params());
    CHECK(s.log_pdet(ul.data(), w.data()) == doctest::Approx(log_pdet_twin(s, ul)).epsilon(1e-13).scale(1.0));
  }
}

