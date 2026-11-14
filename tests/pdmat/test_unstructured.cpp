// unstructured_pdmat<B> (include/laplace/pdmat/unstructured.hpp) for the three
// coupled bijectors B: M^-1 and log det M with their derivatives to order 6
// against the mpmath references (tools/reference/pdmat_unstructured.py) and
// against central differences of the derivative one order below, in a Kokkos
// kernel on the default execution space; on the host, M, L and the
// log-Jacobian identical bit for bit to the bijector's, and log det against
// 2 sum log L_ii of a Cholesky factorization of M. Errors:
//   ulpN   for M^-1: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|, or,
//          when that derivative vanishes (an exact zero computed as rounding
//          noise), in ulps of max_ij |M^-1_ij| at the same point;
//   ulp    for log det: relative, in ulps of max(|ref|, 1).
// The reference differences at 640 bits carry noise up to about 1e-30
// (1 + max |ref|), so a reference below 1e-25 in absolute value is read as 0.
// The difference check: along the last index v of the tuple,
// (f^(K-1)(u + h e_v) - f^(K-1)(u - h e_v))/(2h), h = 1e-5, within 1e-6 of the
// largest entry plus 1e-6 of that of order K-1.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

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

namespace {

namespace lp = laplace::pdmat;
namespace lb = laplace::bijector;
using lc_pdmat = lp::unstructured_pdmat<lb::log_cholesky_bijector>;
using ml_pdmat = lp::unstructured_pdmat<lb::matrix_log_bijector>;
using pc_pdmat = lp::unstructured_corr_pdmat;

constexpr int max_p = 4;
constexpr int max_out = 10;  // packed_size(4)
constexpr int max_n = 10;    // free parameters at p = 4

template <int K, class S>
KOKKOS_INLINE_FUNCTION bool eval_k(const S& s, int quantity, const int* tup, const double* u, double* out, double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = tup[k];
  if (quantity == 0) return s.template inverse_dk<K>(u, idx, out, w);
  out[0] = s.template log_det_dk<K>(u, idx, w);
  return true;
}

// out = the quantity along the tuple of order K; zeroed if identically zero
template <class S>
KOKKOS_INLINE_FUNCTION void eval(const S& s, int quantity, int K, const int* tup, const double* u, double* out, double* w) {
  const int n = quantity == 1 ? 1 : lp::packed_size(s.dim);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.inverse(u, out, w);
      else out[0] = s.log_det(u, w);
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

KOKKOS_INLINE_FUNCTION void eval_any(int map, int p, int quantity, int K, const int* tup, const double* u, double* out,
                                     double* w) {
  if (map == 0) eval(lc_pdmat(p), quantity, K, tup, u, out, w);
  else if (map == 1) eval(ml_pdmat(p), quantity, K, tup, u, out, w);
  else eval(pc_pdmat(p), quantity, K, tup, u, out, w);
}

int scratch_of(int map, int p) {
  if (map == 0) return lc_pdmat(p).scratch_size();
  if (map == 1) return ml_pdmat(p).scratch_size();
  return pc_pdmat(p).scratch_size();
}

struct Group {
  int map, p, quantity, order;
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
    const int map = std::stoi(f[0]), p = std::stoi(f[1]), q = std::stoi(f[3]), K = std::stoi(f[4]);
    const auto key = std::make_tuple(map, p, f[2], q, f[5]);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(g.size())).first;
      Group G{map, p, q, K, f[2], {}, split_hex(f[2]), {}, {}, {}};
      if (K > 0)
        for (double t : split_hex(f[5])) G.tup.push_back(int(t));
      g.push_back(G);
    }
    Group& gr = g[it->second];
    gr.pos.push_back(q == 1 ? 0 : lp::packed_index(std::stoi(f[6]), std::stoi(f[7])));
    gr.hi.push_back(std::strtod(f[8].c_str(), nullptr));
    gr.lo.push_back(std::strtod(f[9].c_str(), nullptr));
  }
  return g;
}

const char* mname[] = {"log_cholesky", "matrix_log", "partial_corr"};
const char* qname[] = {"inverse", "log_det"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE (note/brief_agenti.md: nvcc tags the device lambda with its enclosing function).
void pdmat_unstructured_mpmath_case() {
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_unstructured.csv");
  const int ng = int(groups.size());
  REQUIRE(ng > 0);
  int sw = 0;
  for (int m = 0; m < 3; ++m) sw = std::max(sw, scratch_of(m, max_p));

  Kokkos::View<int* [10]> meta("meta", ng);  // map, p, quantity, order, tuple[6]
  Kokkos::View<double* [max_n]> uu("u", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto hu = Kokkos::create_mirror_view(uu);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.map, hm(g, 1) = G.p, hm(g, 2) = G.quantity, hm(g, 3) = G.order;
    for (int k = 0; k < 6; ++k) hm(g, 4 + k) = k < G.order ? G.tup[k] : 0;
    for (int k = 0; k < max_n; ++k) hu(g, k) = k < int(G.u.size()) ? G.u[k] : 0.0;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(uu, hu);

  // y: the value; fd: the difference of order K - 1; lower: order K - 1 at u.
  // The scratch (about 25000 doubles a thread for matrix_log at p = 4) is
  // allocated for a chunk of groups at a time.
  Kokkos::View<double* [max_out]> y("y", ng), fd("fd", ng), lower("lower", ng);
  constexpr int chunk = 512;
  Kokkos::View<double**, Kokkos::LayoutRight> scratch("scratch", chunk, sw);  // a contiguous row per group
  for (int g0 = 0; g0 < ng; g0 += chunk) {
    const int g1 = std::min(ng, g0 + chunk);
    Kokkos::parallel_for(
        "pdmat_unstructured", Kokkos::RangePolicy<>(g0, g1), KOKKOS_LAMBDA(int g) {
          const int map = meta(g, 0), p = meta(g, 1), q = meta(g, 2), K = meta(g, 3);
          int tup[6];
          for (int k = 0; k < 6; ++k) tup[k] = meta(g, 4 + k);
          double u[max_n], a[max_out], b[max_out];
          for (int k = 0; k < max_n; ++k) u[k] = uu(g, k);
          double* w = &scratch(g - g0, 0);
          eval_any(map, p, q, K, tup, u, a, w);
          for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
          if (K == 0) return;
          eval_any(map, p, q, K - 1, tup, u, a, w);
          for (int k = 0; k < max_out; ++k) lower(g, k) = a[k];
          const int v = tup[K - 1];
          const double x = u[v], h = 1e-5;
          u[v] = x + h;
          eval_any(map, p, q, K - 1, tup, u, a, w);
          u[v] = x - h;
          eval_any(map, p, q, K - 1, tup, u, b, w);
          for (int k = 0; k < max_out; ++k) fd(g, k) = (a[k] - b[k]) / (2.0 * h);
        });
    Kokkos::fence();
  }
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hfd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fd);
  auto hlo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower);

  // max |M^-1_ij| at each point, the scale of a vanishing derivative
  std::map<std::pair<int, std::string>, double> inv_scale;
  for (const Group& G : groups)
    if (G.quantity == 0 && G.order == 0) {
      double s = 0.0;
      for (double h : G.hi) s = std::fmax(s, std::fabs(h));
      inv_scale[{G.map, G.ukey}] = s;
    }
  double worst[3][2][7] = {};
  int worst_g[3][2][7] = {};
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int n = G.quantity == 1 ? 1 : lp::packed_size(G.p);
    double scale = 0.0, err = 0.0;
    auto ref = [&](std::size_t r) { return std::fabs(G.hi[r]) < 1e-25 ? 0.0 : G.hi[r]; };
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(ref(r)));
    scale = G.quantity == 1 ? std::fmax(scale, 1.0) : (scale > 0.0 ? scale : inv_scale.at({G.map, G.ukey}));
    for (std::size_t r = 0; r < G.pos.size(); ++r) {
      const double lo = ref(r) == 0.0 ? 0.0 : G.lo[r];
      const double e = std::fabs((hy(g, G.pos[r]) - ref(r)) - lo);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double ulp = err / (scale * eps);
    double& w = worst[G.map][G.quantity][G.order];
    if (!(ulp <= w)) w = ulp, worst_g[G.map][G.quantity][G.order] = g;
    if (G.order > 0) {
      double lscale = 0.0;
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double d = std::fabs(hfd(g, k) - hy(g, k)) / (1e-6 * (scale + lscale));
        worst_fd = std::fmax(worst_fd, d);
        if (!(d <= 1.0))
          FAIL_CHECK(std::string(mname[G.map]) << " " << qname[G.quantity] << " order " << G.order << " p " << G.p
                                  << ": difference check " << d);
      }
    }
  }

  // bounds on ulpN / ulp, [map][quantity][order]: the larger of the errors
  // measured on 2026-11-13 on gcc/OpenMP and CUDA, times 1.5, rounded up
  const double bound[3][2][7] = {{{6, 6, 6, 8, 43, 29, 26}, {1, 1, 1, 1, 1, 1, 1}},
                                 {{23, 37, 36, 35, 33, 32, 30}, {1, 1, 1, 1, 1, 1, 1}},
                                 {{3, 6, 12, 15, 23, 58, 90}, {5, 2, 2, 4, 6, 3, 43}}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int m = 0; m < 3; ++m)
    for (int q = 0; q < 2; ++q)
      for (int K = 0; K <= 6; ++K) {
        const Group& G = groups[worst_g[m][q][K]];
        std::string tup;
        for (int t : G.tup) tup += std::to_string(t) + " ";
        if (print)
          std::printf("%-13s %-8s K=%d  %10.2f  (p %d, u0 %g, tuple %s)\n", mname[m], qname[q], K, worst[m][q][K], G.p,
                      G.u.empty() ? 0.0 : G.u[0], tup.c_str());
        CHECK_MESSAGE(worst[m][q][K] <= bound[m][q][K],
                      std::string(mname[m]) << " " << qname[q] << " order " << K << ": " << worst[m][q][K] << " ulp at p " << G.p);
      }
  if (print) std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
}

TEST_CASE("unstructured_pdmat: M^-1 and log det against mpmath and differences") { pdmat_unstructured_mpmath_case(); }

namespace {

// a fixed point of n free coordinates
std::vector<double> point(int n) {
  std::vector<double> u(n);
  for (int k = 0; k < n; ++k) u[k] = 0.37 * std::sin(1.3 * k + 0.4) + 0.05 * k;
  return u;
}

template <class S, class B>
void same_as_bijector(int p) {
  const S s(p);
  const B b(p);
  const int n = s.n_params(), np = lp::packed_size(p);
  std::vector<double> u = point(n), w(s.scratch_size()), wb(b.scratch_size()), x(np), y(np);
  s.matrix(u.data(), x.data(), w.data());
  b.inverse(u.data(), y.data(), wb.data());
  CHECK(x == y);
  CHECK(s.log_jacobian(u.data(), w.data()) == b.inverse_log_jacobian(u.data(), wb.data()));
  const int i2[2] = {0, n - 1}, i3[3] = {0, 0, n - 1};
  std::fill(x.begin(), x.end(), 0.0), std::fill(y.begin(), y.end(), 0.0);
  CHECK(s.matrix_d2(u.data(), i2, x.data(), w.data()) == b.inverse_d2(u.data(), i2, y.data(), wb.data()));
  CHECK(x == y);
  CHECK(s.log_jacobian_d3(u.data(), i3, w.data()) == b.inverse_log_jacobian_d3(u.data(), i3, wb.data()));
  if constexpr (S::closed_factor) {
    s.cholesky(u.data(), x.data(), w.data());
    b.inverse_cholesky(u.data(), y.data(), wb.data());
    CHECK(x == y);
    std::fill(x.begin(), x.end(), 0.0), std::fill(y.begin(), y.end(), 0.0);
    CHECK(s.cholesky_d3(u.data(), i3, x.data(), w.data()) == b.inverse_cholesky_d3(u.data(), i3, y.data(), wb.data()));
    CHECK(x == y);
  }
}

// log det M against 2 sum log L_ii of the Cholesky factorization of M in double
template <class S>
void log_det_by_cholesky(int p) {
  const S s(p);
  const int np = lp::packed_size(p);
  std::vector<double> u = point(s.n_params()), w(s.scratch_size()), m(np), l(np, 0.0);
  s.matrix(u.data(), m.data(), w.data());
  double ld = 0.0;
  for (int j = 0; j < p; ++j) {
    double d = m[lp::packed_index(j, j)];
    for (int k = 0; k < j; ++k) d -= l[lp::packed_index(j, k)] * l[lp::packed_index(j, k)];
    l[lp::packed_index(j, j)] = std::sqrt(d);
    ld += std::log(d);
    for (int i = j + 1; i < p; ++i) {
      double r = m[lp::packed_index(i, j)];
      for (int k = 0; k < j; ++k) r -= l[lp::packed_index(i, k)] * l[lp::packed_index(j, k)];
      l[lp::packed_index(i, j)] = r / l[lp::packed_index(j, j)];
    }
  }
  CHECK(s.log_det(u.data(), w.data()) == doctest::Approx(ld).epsilon(1e-13).scale(1.0));
}

}  // namespace

TEST_CASE("unstructured_pdmat: M, L and the log-Jacobian are the bijector's, bit for bit") {
  for (int p = 2; p <= 4; ++p) {
    same_as_bijector<lc_pdmat, lb::log_cholesky_bijector>(p);
    same_as_bijector<ml_pdmat, lb::matrix_log_bijector>(p);
    same_as_bijector<pc_pdmat, lb::partial_corr_bijector>(p);
  }
}

TEST_CASE("unstructured_pdmat: log det against a Cholesky factorization of M") {
  for (int p = 1; p <= 4; ++p) {
    log_det_by_cholesky<lc_pdmat>(p);
    log_det_by_cholesky<ml_pdmat>(p);
    if (p > 1) log_det_by_cholesky<pc_pdmat>(p);
  }
  CHECK(pc_pdmat::unit_diagonal);
  CHECK_FALSE(lc_pdmat::unit_diagonal);
  CHECK(ml_pdmat::closed_inverse);
  CHECK_FALSE(ml_pdmat::closed_factor);
}
