// The pilot structures (include/laplace/pdmat/: diagonal_pdmat,
// compound_symmetry_pdmat, ar1_pdmat) against the mpmath references
// (tools/reference/pdmat_pilot.py) and against central differences of the
// derivative one order below, evaluated in a Kokkos kernel on the default
// execution space. For each structure, quantity (M, M^-1, L, log det) and
// order 0..6 the error is
//   ulpN   for a matrix: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|,
//          the largest entry of that derivative matrix (normwise);
//   ulp    for log det: relative, in ulps of max(|ref|, 1e-20), and in ulps
//          of max(|ref|, 1) at order 0 (log det crosses 0).
// The difference check: (f^(K-1)(theta + h) - f^(K-1)(theta - h))/(2h),
// h = 1e-5 max(1, |theta|), within 1e-6 of the largest entry plus 1e-6 of
// that of order K-1 (truncation h^2 f^(K+2)/6 and rounding eps f^(K-1)/h).
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/diagonal.hpp>

#include <Kokkos_Core.hpp>

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
constexpr int max_out = 45;  // packed_size(9)

template <int K, class S>
KOKKOS_INLINE_FUNCTION bool eval_k(const S& s, int quantity, int param, const double* th, double* out) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = param;
  switch (quantity) {
    case 0: return s.template matrix_dk<K>(th, idx, out, nullptr);
    case 1: return s.template inverse_dk<K>(th, idx, out, nullptr);
    case 2: return s.template cholesky_dk<K>(th, idx, out, nullptr);
    default: out[0] = s.template log_det_dk<K>(th, idx, nullptr); return true;
  }
}

// out = the quantity of order K; false if identically zero (out zeroed)
template <class S>
KOKKOS_INLINE_FUNCTION bool eval(const S& s, int quantity, int K, int param, const double* th, double* out) {
  const int n = quantity == 3 ? 1 : lp::packed_size(s.dim);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.matrix(th, out, nullptr);
      else if (quantity == 1) s.inverse(th, out, nullptr);
      else if (quantity == 2) s.cholesky(th, out, nullptr);
      else out[0] = s.log_det(th, nullptr);
      break;
    case 1: nz = eval_k<1>(s, quantity, param, th, out); break;
    case 2: nz = eval_k<2>(s, quantity, param, th, out); break;
    case 3: nz = eval_k<3>(s, quantity, param, th, out); break;
    case 4: nz = eval_k<4>(s, quantity, param, th, out); break;
    case 5: nz = eval_k<5>(s, quantity, param, th, out); break;
    case 6: nz = eval_k<6>(s, quantity, param, th, out); break;
  }
  if (!nz)
    for (int k = 0; k < n; ++k) out[k] = 0.0;
  return nz;
}

KOKKOS_INLINE_FUNCTION void eval_any(int structure, int p, int quantity, int K, int param, const double* th, double* out) {
  if (structure == 0) eval(lp::diagonal_pdmat(p), quantity, K, param, th, out);
  else if (structure == 1) eval(lp::compound_symmetry_pdmat(p), quantity, K, param, th, out);
  else eval(lp::ar1_pdmat(p), quantity, K, param, th, out);
}

struct Group {
  int structure, p, param, quantity, order;
  double theta;
  std::vector<int> pos;  // packed position of each reference row
  std::vector<double> hi, lo;
};

std::vector<Group> read_groups(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::tuple<int, int, int, double, int, int>, int> where;
  std::vector<Group> g;
  while (std::getline(in, line)) {
    std::vector<double> v;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ',')) v.push_back(std::strtod(f.c_str(), nullptr));
    const int s = int(v[0]), p = int(v[1]), par = int(v[2]), q = int(v[4]), K = int(v[5]), i = int(v[6]), j = int(v[7]);
    const auto key = std::make_tuple(s, p, par, v[3], q, K);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(g.size())).first;
      g.push_back(Group{s, p, par, q, K, v[3], {}, {}, {}});
    }
    Group& gr = g[it->second];
    gr.pos.push_back(q == 3 ? 0 : lp::packed_index(i, j));
    gr.hi.push_back(v[8]);
    gr.lo.push_back(v[9]);
  }
  return g;
}

const char* sname[] = {"diagonal", "compound_symmetry", "ar1"};
const char* qname[] = {"matrix", "inverse", "cholesky", "log_det"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE: doctest's case function has internal linkage, and the extended device lambda
// below takes the address of its enclosing function as a tag, which collides between
// translation units (note/brief_agenti.md, found on 2026-11-13).
void pdmat_mpmath_case() {
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_pilot.csv");
  const int ng = int(groups.size());
  REQUIRE(ng > 0);

  Kokkos::View<int* [5]> meta("meta", ng);
  Kokkos::View<double*> theta("theta", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto ht = Kokkos::create_mirror_view(theta);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.structure, hm(g, 1) = G.p, hm(g, 2) = G.param, hm(g, 3) = G.quantity, hm(g, 4) = G.order;
    ht(g) = G.theta;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(theta, ht);

  // y: the value; fd: the difference of order K - 1; lower: order K - 1 at theta
  Kokkos::View<double* [max_out]> y("y", ng), fd("fd", ng), lower("lower", ng);
  Kokkos::parallel_for(
      "pdmat_pilot", ng, KOKKOS_LAMBDA(int g) {
        const int s = meta(g, 0), p = meta(g, 1), par = meta(g, 2), q = meta(g, 3), K = meta(g, 4);
        double th[9], a[max_out], b[max_out];
        for (int k = 0; k < p; ++k) th[k] = s == 0 ? theta(g) + 0.25 * k : theta(g);
        eval_any(s, p, q, K, par, th, a);
        for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
        if (K == 0) return;
        eval_any(s, p, q, K - 1, par, th, a);
        for (int k = 0; k < max_out; ++k) lower(g, k) = a[k];
        const double x = th[par], h = 1e-5 * (x < 1.0 && x > -1.0 ? 1.0 : (x < 0 ? -x : x));
        th[par] = x + h;
        eval_any(s, p, q, K - 1, par, th, a);
        th[par] = x - h;
        eval_any(s, p, q, K - 1, par, th, b);
        for (int k = 0; k < max_out; ++k) fd(g, k) = (a[k] - b[k]) / (2.0 * h);
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hfd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fd);
  auto hlo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower);

  // worst error per (structure, quantity, order)
  double worst[3][4][7] = {};
  double worst_theta[3][4][7] = {};
  int worst_p[3][4][7] = {};
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int n = G.quantity == 3 ? 1 : lp::packed_size(G.p);
    double scale = 0.0, err = 0.0;
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(G.hi[r]));
    // the references come from differences at 640 bits and carry noise up to
    // about 1e-30 (1 + max |ref|), so a value that is exactly zero (an odd
    // function's even derivative at 0) is read as zero below 1e-20
    if (G.quantity == 3 && G.order == 0) scale = std::fmax(scale, 1.0);
    else scale = std::fmax(scale, 1e-20);
    for (std::size_t r = 0; r < G.pos.size(); ++r) {
      const double e = std::fabs((hy(g, G.pos[r]) - G.hi[r]) - G.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double u = err / (scale * eps);
    double& w = worst[G.structure][G.quantity][G.order];
    if (!(u <= w)) {
      w = u;
      worst_theta[G.structure][G.quantity][G.order] = G.theta;
      worst_p[G.structure][G.quantity][G.order] = G.p;
    }
    if (G.order > 0) {
      double lscale = 0.0;
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double d = std::fabs(hfd(g, k) - hy(g, k)) / (1e-6 * (scale + lscale));
        worst_fd = std::fmax(worst_fd, d);
        if (!(d <= 1.0))
          FAIL_CHECK(sname[G.structure] << " " << qname[G.quantity] << " order " << G.order << " p " << G.p
                                        << " theta " << G.theta << ": difference check " << d);
      }
    }
  }

  // bounds on ulpN / ulp, [structure][quantity][order]: the larger of the
  // errors measured on 2026-11-12 on gcc/OpenMP and CUDA, times 1.5, rounded up
  const double bound[3][4][7] = {
      {{1, 1, 1, 1, 1, 1, 1}, {1, 2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}},
      {{1, 3, 3, 4, 13, 8, 5}, {2, 1, 2, 1, 2, 1, 2}, {3, 9, 9, 10, 10, 11, 13}, {12, 6, 3, 3, 4, 13, 8}},
      {{5, 5, 7, 7, 13, 11, 20}, {1, 1, 1, 1, 1, 1, 1}, {5, 5, 6, 7, 13, 13, 11}, {4, 2, 2, 4, 5, 5, 8}}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int s = 0; s < 3; ++s)
    for (int q = 0; q < 4; ++q)
      for (int K = 0; K <= 6; ++K) {
        if (print)
          std::printf("%-18s %-9s K=%d  %8.2f  (p %d, theta %g)\n", sname[s], qname[q], K, worst[s][q][K],
                      worst_p[s][q][K], worst_theta[s][q][K]);
        CHECK_MESSAGE(worst[s][q][K] <= bound[s][q][K], sname[s] << " " << qname[q] << " order " << K << ": "
                                                                 << worst[s][q][K] << " ulp at p " << worst_p[s][q][K]
                                                                 << ", theta " << worst_theta[s][q][K]);
      }
  if (print) std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
}

TEST_CASE("pdmat pilot: diagonal, compound symmetry, AR(1) against mpmath and differences") { pdmat_mpmath_case(); }

TEST_CASE("pdmat pilot: structural zeros of diagonal_pdmat") {
  const lp::diagonal_pdmat d(3);
  const double th[3] = {0.1, -0.2, 0.3};
  double m[6];
  const int mixed[2] = {0, 1}, pure[2] = {1, 1};
  CHECK_FALSE(d.matrix_d2(th, mixed, m, nullptr));
  CHECK_FALSE(d.inverse_d2(th, mixed, m, nullptr));
  CHECK_FALSE(d.cholesky_d2(th, mixed, m, nullptr));
  CHECK(d.log_det_d2(th, pure, nullptr) == 0.0);
  CHECK(d.matrix_d2(th, pure, m, nullptr));
  CHECK(m[lp::packed_index(1, 1)] == doctest::Approx(4.0 * std::exp(-0.4)));
  CHECK(m[lp::packed_index(0, 0)] == 0.0);
  const lp::ar1_pdmat a(1);
  const int z[1] = {0};
  CHECK_FALSE(a.matrix_d1(th, z, m, nullptr));
  CHECK(a.log_det_d1(th, z, nullptr) == 0.0);
}
