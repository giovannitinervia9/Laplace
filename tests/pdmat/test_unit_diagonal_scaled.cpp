// unit_diagonal_scaled_pdmat<R> (include/laplace/pdmat/unit_diagonal_scaled.hpp)
// with R = compound_symmetry_pdmat, ar1_pdmat and unstructured_corr_pdmat,
// heterogeneous and common scales, against the mpmath references of
// tools/reference/pdmat_scaled.py, evaluated in a Kokkos kernel on the
// default execution space. The error is
//   ulpN   for a matrix: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|,
//          bounded on the matrix with the scaling undone (D^-1 M D^-1,
//          D M^-1 D, D^-1 L; printed as "unscaled"), also printed for M
//          itself and per entry, in ulps of each entry ("entry");
//   ulp    for log det and the log-Jacobian: in ulps of max(|ref|, 1).
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/compound_symmetry.hpp>
#include <laplace/pdmat/unit_diagonal_scaled.hpp>
#include <laplace/pdmat/unstructured.hpp>

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
using scaled_cs = lp::unit_diagonal_scaled_pdmat<lp::compound_symmetry_pdmat>;
using scaled_ar1 = lp::unit_diagonal_scaled_pdmat<lp::ar1_pdmat>;
using scaled_corr = lp::unit_diagonal_scaled_pdmat<lp::unstructured_corr_pdmat>;
static_assert(lp::pdmat_structure<scaled_cs> && lp::pdmat_structure<scaled_ar1> && lp::pdmat_structure<scaled_corr>);
static_assert(scaled_corr::has_density && !scaled_ar1::has_density);

constexpr int max_out = 6;    // packed_size(3)
constexpr int max_theta = 6;  // 3 scales + 3 partial correlations

template <int K, class S>
KOKKOS_INLINE_FUNCTION bool eval_k(const S& s, int quantity, const int* tup, const double* th, double* out, double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = tup[k];
  switch (quantity) {
    case 0: return s.template matrix_dk<K>(th, idx, out, w);
    case 1: return s.template inverse_dk<K>(th, idx, out, w);
    case 2: return s.template cholesky_dk<K>(th, idx, out, w);
    case 3: out[0] = s.template log_det_dk<K>(th, idx, w); return true;
    default:
      if constexpr (S::has_density) out[0] = s.template log_jacobian_dk<K>(th, idx, w);
      return true;
  }
}

template <class S>
KOKKOS_INLINE_FUNCTION void eval(const S& s, int quantity, int K, const int* tup, const double* th, double* out, double* w) {
  const int n = quantity >= 3 ? 1 : lp::packed_size(s.dim);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.matrix(th, out, w);
      else if (quantity == 1) s.inverse(th, out, w);
      else if (quantity == 2) s.cholesky(th, out, w);
      else if (quantity == 3) out[0] = s.log_det(th, w);
      else if constexpr (S::has_density) out[0] = s.log_jacobian(th, w);
      break;
    case 1: nz = eval_k<1>(s, quantity, tup, th, out, w); break;
    case 2: nz = eval_k<2>(s, quantity, tup, th, out, w); break;
    case 3: nz = eval_k<3>(s, quantity, tup, th, out, w); break;
    case 4: nz = eval_k<4>(s, quantity, tup, th, out, w); break;
    case 5: nz = eval_k<5>(s, quantity, tup, th, out, w); break;
    case 6: nz = eval_k<6>(s, quantity, tup, th, out, w); break;
  }
  if (!nz)
    for (int k = 0; k < n; ++k) out[k] = 0.0;
}

KOKKOS_INLINE_FUNCTION void eval_any(int structure, int p, bool common, int quantity, int K, const int* tup,
                                     const double* th, double* out, double* w) {
  if (structure == 0) eval(scaled_cs(lp::compound_symmetry_pdmat(p), common), quantity, K, tup, th, out, w);
  else if (structure == 1) eval(scaled_ar1(lp::ar1_pdmat(p), common), quantity, K, tup, th, out, w);
  else eval(scaled_corr(lp::unstructured_corr_pdmat(p), common), quantity, K, tup, th, out, w);
}

struct Group {
  int structure, p, common, quantity, order;
  std::vector<double> theta;
  std::vector<int> tup;
  std::vector<int> pos;
  std::vector<double> hi, lo;
};

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string f;
  while (std::getline(ss, f, c)) out.push_back(f);
  return out;
}

std::vector<Group> read_groups(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);
  std::map<std::tuple<int, int, int, std::string, int, std::string>, int> where;
  std::vector<Group> g;
  while (std::getline(in, line)) {
    const auto f = split(line, ',');
    const int s = std::stoi(f[0]), p = std::stoi(f[1]), c = std::stoi(f[2]), q = std::stoi(f[4]), K = std::stoi(f[5]);
    const auto key = std::make_tuple(s, p, c, f[3], q, f[6]);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(g.size())).first;
      Group G{s, p, c, q, K, {}, {}, {}, {}, {}};
      for (const auto& x : split(f[3], ';')) G.theta.push_back(std::strtod(x.c_str(), nullptr));
      if (f[6] != "-")
        for (const auto& x : split(f[6], ';')) G.tup.push_back(std::stoi(x));
      g.push_back(G);
    }
    Group& gr = g[it->second];
    gr.pos.push_back(q >= 3 ? 0 : lp::packed_index(std::stoi(f[7]), std::stoi(f[8])));
    gr.hi.push_back(std::strtod(f[9].c_str(), nullptr));
    gr.lo.push_back(std::strtod(f[10].c_str(), nullptr));
  }
  return g;
}

const char* sname[] = {"compound_symmetry", "ar1", "unstructured_corr"};
const char* qname[] = {"matrix", "inverse", "cholesky", "log_det", "log_jacobian"};

}  // namespace

// in a named function with external linkage, not a TEST_CASE body (the tag of
// the extended device lambda; note/brief_agenti.md)
void pdmat_scaled_mpmath_case() {
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_scaled.csv");
  const int ng = int(groups.size());
  REQUIRE(ng > 0);
  const int ws = scaled_corr(lp::unstructured_corr_pdmat(3)).scratch_size();

  Kokkos::View<int* [12]> meta("meta", ng);  // structure, p, common, quantity, order, tuple[6]
  Kokkos::View<double* [max_theta]> theta("theta", ng);
  Kokkos::View<double**, Kokkos::LayoutRight> scratch("scratch", ng, ws > 0 ? ws : 1);  // a contiguous row per group
  auto hm = Kokkos::create_mirror_view(meta);
  auto ht = Kokkos::create_mirror_view(theta);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.structure, hm(g, 1) = G.p, hm(g, 2) = G.common, hm(g, 3) = G.quantity, hm(g, 4) = G.order;
    for (int k = 0; k < 6; ++k) hm(g, 5 + k) = k < int(G.tup.size()) ? G.tup[k] : 0;
    for (int k = 0; k < max_theta; ++k) ht(g, k) = k < int(G.theta.size()) ? G.theta[k] : 0.0;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(theta, ht);

  Kokkos::View<double* [max_out]> y("y", ng);
  Kokkos::parallel_for(
      "pdmat_scaled", ng, KOKKOS_LAMBDA(int g) {
        double th[max_theta], a[max_out] = {};
        int tup[6];
        for (int k = 0; k < max_theta; ++k) th[k] = theta(g, k);
        for (int k = 0; k < 6; ++k) tup[k] = meta(g, 5 + k);
        eval_any(meta(g, 0), meta(g, 1), meta(g, 2) != 0, meta(g, 3), meta(g, 4), tup, th, a, &scratch(g, 0));
        for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
      });
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

  double worst[3][5][7] = {}, worst_unscaled[3][5][7] = {}, worst_entry[3][5][7] = {};
  std::string worst_at[3][5][7];
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    double scale = G.quantity >= 3 ? 1.0 : 1e-300, err = 0.0;
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(G.hi[r]));
    for (std::size_t r = 0; r < G.pos.size(); ++r) {
      const double e = std::fabs((hy(g, G.pos[r]) - G.hi[r]) - G.lo[r]);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    // a derivative that is zero by structure: the references carry noise up
    // to about 1e-30 (1 + max |ref|), so they are read as zero below 1e-20
    const double u = scale < 1e-20 ? (err < 1e-20 ? 0.0 : INFINITY) : err / (scale * eps);
    // the same with the scaling undone (errors of R's own quantity, in ulpN)
    if (G.quantity < 3) {
      double us = 0.0, es = 0.0;
      for (std::size_t r = 0; r < G.pos.size(); ++r) {
        int i = 0;
        while (lp::packed_index(i + 1, 0) <= G.pos[r]) ++i;
        const int j = G.pos[r] - lp::packed_index(i, 0);
        const double si = G.theta[G.common ? 0 : i], sj = G.theta[G.common ? 0 : j];
        const double f = G.quantity == 0 ? std::exp(-si - sj) : G.quantity == 1 ? std::exp(si + sj) : std::exp(-si);
        us = std::fmax(us, f * std::fabs(G.hi[r]));
        es = std::fmax(es, f * std::fabs((hy(g, G.pos[r]) - G.hi[r]) - G.lo[r]));
      }
      const double ud = us < 1e-20 ? 0.0 : es / (us * eps);
      double& wd = worst_unscaled[G.structure][G.quantity][G.order];
      wd = std::fmax(wd, ud);
      // elementwise: each entry in ulps of itself (entries below 1e-12 of
      // the largest are skipped: zero by structure or reference noise)
      for (std::size_t r = 0; r < G.pos.size(); ++r) {
        const double a = std::fabs(G.hi[r]);
        if (a <= 1e-12 * scale || a < 1e-25) continue;
        const double ce = std::fabs((hy(g, G.pos[r]) - G.hi[r]) - G.lo[r]) / (a * eps);
        double& wc = worst_entry[G.structure][G.quantity][G.order];
        wc = std::fmax(wc, ce);
      }
    }
    double& w = worst[G.structure][G.quantity][G.order];
    if (!(u <= w)) {
      w = u;
      std::ostringstream os;
      os << "p " << G.p << " common " << G.common << " tuple";
      for (int k : G.tup) os << " " << k;
      worst_at[G.structure][G.quantity][G.order] = os.str();
    }
  }

  // bounds, [structure][quantity][order]: the larger of the errors measured
  // on gcc/OpenMP and CUDA on 2026-11-13, times 1.5, rounded up. The matrix
  // quantities are bounded in the unscaled measure (decided with Giovanni on
  // 2026-11-14): ulpN of D^-1 M D^-1, D M^-1 D and D^-1 L, R's own scale, the
  // one that reaches x^T M^-1 x = z^T R^-1 z for x = D z. The normwise measure
  // of M itself is printed: with unstructured_corr and s = +-3 the inverse
  // reaches 1.6e4 ulpN (gcc) / 2.8e4 (CUDA) there, from entries that are zero
  // in exact arithmetic (a hidden cancellation in R^-1) computed as a residue
  // of a few ulp of max |R^-1|, multiplied by e^(-s_i - s_j) up to e^6
  const double bound[3][5][7] = {
      {{1, 2, 7, 8, 8, 36, 126}, {2, 3, 5, 5, 3, 4, 3}, {2, 3, 3, 4, 7, 8, 11}, {2, 1, 1, 1, 2, 2, 6}, {}},
      {{2, 3, 4, 4, 12, 7, 9}, {2, 2, 2, 2, 2, 2, 2}, {2, 3, 4, 4, 5, 7, 9}, {4, 1, 2, 3, 7, 2, 8}, {}},
      {{2, 3, 5, 8, 26, 23, 61},
       {3, 4, 8, 11, 19, 33, 47},
       {1, 3, 4, 4, 24, 4, 7},
       {3, 1, 2, 2, 6, 2, 15},
       {1, 1, 2, 3, 2, 2, 22}}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int s = 0; s < 3; ++s)
    for (int q = 0; q < 5; ++q)
      for (int K = 0; K <= 6; ++K) {
        if (q == 4 && s != 2) continue;
        if (print)
          std::printf("%-18s %-12s K=%d  %8.2f  unscaled %8.2f  entry %10.4g  (%s)\n", sname[s], qname[q], K,
                      worst[s][q][K], worst_unscaled[s][q][K], worst_entry[s][q][K], worst_at[s][q][K].c_str());
        const double e = q < 3 ? worst_unscaled[s][q][K] : worst[s][q][K];
        CHECK_MESSAGE(e <= bound[s][q][K], sname[s] << " " << qname[q] << " order " << K << ": " << e);
      }
}

TEST_CASE("unit_diagonal_scaled_pdmat against mpmath") { pdmat_scaled_mpmath_case(); }

TEST_CASE("unit_diagonal_scaled_pdmat: structural zeros and common scale") {
  const scaled_ar1 h(lp::ar1_pdmat(3));
  const scaled_ar1 c(lp::ar1_pdmat(3), true);
  CHECK(h.n_params() == 4);
  CHECK(c.n_params() == 2);
  const double th[4] = {0.1, -0.2, 0.3, 0.4};
  double m[6] = {7, 7, 7, 7, 7, 7};
  const int three[3] = {0, 1, 2}, two_l[2] = {0, 1}, one_r[2] = {0, 3};
  CHECK_FALSE(h.matrix_d3(th, three, m, nullptr));
  CHECK_FALSE(h.inverse_d3(th, three, m, nullptr));
  CHECK_FALSE(h.cholesky_d2(th, two_l, m, nullptr));
  CHECK(m[0] == 7.0);  // untouched
  CHECK(h.log_det_d2(th, two_l, nullptr) == 0.0);
  CHECK(h.log_det_d2(th, one_r, nullptr) == 0.0);
  const int s0[1] = {0};
  CHECK(h.log_det_d1(th, s0, nullptr) == 2.0);
  CHECK(c.log_det_d1(th, s0, nullptr) == 6.0);
  const scaled_corr hc(lp::unstructured_corr_pdmat(2)), cc(lp::unstructured_corr_pdmat(2), true);
  CHECK(hc.log_jacobian_d1(th, s0, nullptr) == 3.0);
  CHECK(std::isnan(cc.log_jacobian(th, nullptr)));
}
