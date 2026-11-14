// toeplitz_pdmat (include/laplace/pdmat/toeplitz.hpp): M, M^-1, L and log det M
// with their derivatives to order 6 against the mpmath references
// (tools/reference/pdmat_toeplitz.py, built from the definition of the partial
// autocorrelation, not from Durbin-Levinson) and against central differences of
// the derivative one order below, in a Kokkos kernel on the default execution
// space; on the host, the AR(1) twin (theta = (t, 0, ..., 0) gives ar1_pdmat(p)
// at t), M^-1 M = I and L L^T = M, and the scaling composition. Errors:
//   ulpN   for M, M^-1, L: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|,
//          or, when that derivative vanishes, in ulps of max_ij of the
//          quantity itself at the same point;
//   ulp    for log det: relative, in ulps of max(|ref|, 1).
// A reference below 1e-25 in absolute value is read as 0 (the noise of the
// differences at 640 bits). The difference check: along the last index v of
// the tuple, (f^(K-1)(theta + h e_v) - f^(K-1)(theta - h e_v))/(2h), h = 1e-5,
// within 1e-6 of the largest entry plus 1e-6 of that of order K-1.
// Set LAPLACE_PRINT_ERRORS=1 to print the table, with each error also in ulps
// of max_ij of the quantity itself at that point.
#include <doctest/doctest.h>

#include <laplace/pdmat/ar1.hpp>
#include <laplace/pdmat/toeplitz.hpp>
#include <laplace/pdmat/unit_diagonal_scaled.hpp>

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

constexpr int max_p = 6;
constexpr int max_out = 21;  // packed_size(6)
constexpr int max_n = 5;     // parameters at p = 6

template <int K>
KOKKOS_INLINE_FUNCTION bool eval_k(const lp::toeplitz_pdmat& s, int quantity, const int* tup, const double* t, double* out,
                                   double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = tup[k];
  switch (quantity) {
    case 0: return s.template matrix_dk<K>(t, idx, out, w);
    case 1: return s.template inverse_dk<K>(t, idx, out, w);
    case 2: return s.template cholesky_dk<K>(t, idx, out, w);
  }
  out[0] = s.template log_det_dk<K>(t, idx, w);
  return true;
}

// out = the quantity along the tuple of order K; zeroed if identically zero
KOKKOS_INLINE_FUNCTION void eval(int p, int quantity, int K, const int* tup, const double* t, double* out, double* w) {
  const lp::toeplitz_pdmat s(p);
  const int n = quantity == 3 ? 1 : lp::packed_size(p);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.matrix(t, out, w);
      else if (quantity == 1) s.inverse(t, out, w);
      else if (quantity == 2) s.cholesky(t, out, w);
      else out[0] = s.log_det(t, w);
      break;
    case 1: nz = eval_k<1>(s, quantity, tup, t, out, w); break;
    case 2: nz = eval_k<2>(s, quantity, tup, t, out, w); break;
    case 3: nz = eval_k<3>(s, quantity, tup, t, out, w); break;
    case 4: nz = eval_k<4>(s, quantity, tup, t, out, w); break;
    case 5: nz = eval_k<5>(s, quantity, tup, t, out, w); break;
    default: nz = eval_k<6>(s, quantity, tup, t, out, w); break;
  }
  if (!nz)
    for (int k = 0; k < n; ++k) out[k] = 0.0;
}

struct Group {
  int p, quantity, order;
  std::string tkey;
  std::vector<int> tup;
  std::vector<double> theta;
  std::vector<int> pos;
  std::vector<double> hi, lo;
};

std::vector<double> split_hex(const std::string& s) {
  std::vector<double> v;
  if (s == "-") return v;
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
  std::map<std::tuple<int, std::string, int, std::string>, int> where;
  std::vector<Group> g;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string x;
    while (std::getline(ss, x, ',')) f.push_back(x);
    const int p = std::stoi(f[0]), q = std::stoi(f[2]), K = std::stoi(f[3]);
    const auto key = std::make_tuple(p, f[1], q, f[4]);
    auto it = where.find(key);
    if (it == where.end()) {
      it = where.emplace(key, int(g.size())).first;
      Group G{p, q, K, f[1], {}, split_hex(f[1]), {}, {}, {}};
      if (K > 0)
        for (double t : split_hex(f[4])) G.tup.push_back(int(t));
      g.push_back(G);
    }
    Group& gr = g[it->second];
    gr.pos.push_back(q == 3 ? 0 : lp::packed_index(std::stoi(f[5]), std::stoi(f[6])));
    gr.hi.push_back(std::strtod(f[7].c_str(), nullptr));
    gr.lo.push_back(std::strtod(f[8].c_str(), nullptr));
  }
  return g;
}

const char* qname[] = {"matrix", "inverse", "cholesky", "log_det"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE (note/brief_agenti.md: nvcc tags the device lambda with its enclosing function).
void pdmat_toeplitz_mpmath_case() {
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_toeplitz.csv");
  const int ng = int(groups.size());
  REQUIRE(ng > 0);
  const int sw = std::max(1, lp::toeplitz_pdmat(max_p).scratch_size());

  Kokkos::View<int* [9]> meta("meta", ng);  // p, quantity, order, tuple[6]
  Kokkos::View<double* [max_n]> tt("theta", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto ht = Kokkos::create_mirror_view(tt);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.p, hm(g, 1) = G.quantity, hm(g, 2) = G.order;
    for (int k = 0; k < 6; ++k) hm(g, 3 + k) = k < G.order ? G.tup[k] : 0;
    for (int k = 0; k < max_n; ++k) ht(g, k) = k < int(G.theta.size()) ? G.theta[k] : 0.0;
  }
  Kokkos::deep_copy(meta, hm);
  Kokkos::deep_copy(tt, ht);

  // y: the value; fd: the difference of order K - 1; lower: order K - 1 at theta
  Kokkos::View<double* [max_out]> y("y", ng), fd("fd", ng), lower("lower", ng);
  // the scratch (3328 doubles a thread at p = 6) for a chunk of groups at a time
  constexpr int chunk = 512;
  Kokkos::View<double**, Kokkos::LayoutRight> scratch("scratch", chunk, sw);  // a contiguous row per group
  for (int g0 = 0; g0 < ng; g0 += chunk) {
    const int g1 = std::min(ng, g0 + chunk);
    Kokkos::parallel_for(
      "pdmat_toeplitz", Kokkos::RangePolicy<>(g0, g1), KOKKOS_LAMBDA(int g) {
        const int p = meta(g, 0), q = meta(g, 1), K = meta(g, 2);
        int tup[6];
        for (int k = 0; k < 6; ++k) tup[k] = meta(g, 3 + k);
        double t[max_n], a[max_out] = {}, b[max_out] = {};
        for (int k = 0; k < max_n; ++k) t[k] = tt(g, k);
        double* w = &scratch(g - g0, 0);
        eval(p, q, K, tup, t, a, w);
        for (int k = 0; k < max_out; ++k) y(g, k) = a[k];
        if (K == 0) return;
        eval(p, q, K - 1, tup, t, a, w);
        for (int k = 0; k < max_out; ++k) lower(g, k) = a[k];
        const int v = tup[K - 1];
        const double x = t[v], h = 1e-5;
        t[v] = x + h;
        eval(p, q, K - 1, tup, t, a, w);
        t[v] = x - h;
        eval(p, q, K - 1, tup, t, b, w);
        for (int k = 0; k < max_out; ++k) fd(g, k) = (a[k] - b[k]) / (2.0 * h);
      });
    Kokkos::fence();
  }
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hfd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fd);
  auto hlo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower);

  // max |entry| of each quantity at each point, the scale of a vanishing derivative
  std::map<std::tuple<int, std::string, int>, double> zero_scale;
  for (const Group& G : groups)
    if (G.order == 0) {
      double s = 0.0;
      for (double h : G.hi) s = std::fmax(s, std::fabs(h));
      zero_scale[{G.p, G.tkey, G.quantity}] = s;
    }
  double worst[4][7] = {}, worst_m[4][7] = {};
  int worst_g[4][7] = {};
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int n = G.quantity == 3 ? 1 : lp::packed_size(G.p);
    double scale = 0.0, err = 0.0;
    auto ref = [&](std::size_t r) { return std::fabs(G.hi[r]) < 1e-25 ? 0.0 : G.hi[r]; };
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(ref(r)));
    scale = G.quantity == 3 ? std::fmax(scale, 1.0) : (scale > 0.0 ? scale : zero_scale.at({G.p, G.tkey, G.quantity}));
    for (std::size_t r = 0; r < G.pos.size(); ++r) {
      const double lo = ref(r) == 0.0 ? 0.0 : G.lo[r];
      const double e = std::fabs((hy(g, G.pos[r]) - ref(r)) - lo);
      err = std::fmax(err, std::isfinite(e) ? e : INFINITY);
    }
    const double ulp = err / (scale * eps);
    double& w = worst[G.quantity][G.order];
    if (!(ulp <= w)) w = ulp, worst_g[G.quantity][G.order] = g;
    if (G.quantity < 3) {
      // the same error in ulps of max_ij of the quantity itself (order 0) at that point
      const double um = err / (zero_scale.at({G.p, G.tkey, G.quantity}) * eps);
      if (!(um <= worst_m[G.quantity][G.order])) worst_m[G.quantity][G.order] = um;
    }
    if (G.order > 0) {
      const double tol = 1e-6;
      double lscale = 0.0;
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double d = std::fabs(hfd(g, k) - hy(g, k)) / (tol * (scale + lscale));
        worst_fd = std::fmax(worst_fd, d);
        if (!(d <= 1.0))
          FAIL_CHECK(std::string(qname[G.quantity]) << " order " << G.order << " p " << G.p << ": difference check " << d);
      }
    }
  }

  // bounds on ulpN / ulp, [quantity][order]: the larger of the errors measured
  // on 2026-11-14 on gcc/OpenMP and CUDA, times 1.5, rounded up
  const double bound[4][7] = {{2, 12, 12, 190, 169, 228, 244},
                              {15, 14, 15, 14, 15, 14, 21},
                              {3, 7, 8, 10, 12, 17, 17},
                              {3, 2, 2, 4, 19, 6, 35}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int q = 0; q < 4; ++q)
    for (int K = 0; K <= 6; ++K) {
      const Group& G = groups[worst_g[q][K]];
      std::string tup;
      for (int t : G.tup) tup += std::to_string(t) + " ";
      if (print)
        std::printf("toeplitz %-8s K=%d  %8.2f  (p %d, theta0 %g, tuple %s)  in ulps of the quantity %8.2f\n", qname[q], K,
                    worst[q][K], G.p, G.theta.empty() ? 0.0 : G.theta[0], tup.c_str(), worst_m[q][K]);
      CHECK_MESSAGE(worst[q][K] <= bound[q][K], std::string(qname[q]) << " order " << K << ": " << worst[q][K] << " ulp at p " << G.p);
    }
  if (print) std::printf("toeplitz worst difference check: %.3g of its tolerance\n", worst_fd);
}

TEST_CASE("toeplitz_pdmat: M, M^-1, L and log det against mpmath and differences") { pdmat_toeplitz_mpmath_case(); }

namespace {

// max |a - b| over the packed entries, in units of max |b| (or 1 if b vanishes)
double rel_gap(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0, s = 0.0;
  for (std::size_t k = 0; k < b.size(); ++k) d = std::fmax(d, std::fabs(a[k] - b[k])), s = std::fmax(s, std::fabs(b[k]));
  return d / (s > 0.0 ? s : 1.0);
}

template <int K>
void ar1_twin_order(int p, double t) {
  const lp::toeplitz_pdmat s(p);
  const lp::ar1_pdmat a(p);
  const int np = lp::packed_size(p);
  std::vector<double> th(std::max(1, p - 1), 0.0), w(std::max(1, s.scratch_size())), x(np), y(np);
  th[0] = t;
  int idx[K] = {};
  CHECK(s.template matrix_dk<K>(th.data(), idx, x.data(), w.data()) == a.template matrix_dk<K>(&t, idx, y.data(), nullptr));
  CHECK(rel_gap(x, y) < 1e-13);
  s.template inverse_dk<K>(th.data(), idx, x.data(), w.data());
  a.template inverse_dk<K>(&t, idx, y.data(), nullptr);
  CHECK(rel_gap(x, y) < 1e-13);
  s.template cholesky_dk<K>(th.data(), idx, x.data(), w.data());
  a.template cholesky_dk<K>(&t, idx, y.data(), nullptr);
  CHECK(rel_gap(x, y) < 1e-13);
  CHECK(s.template log_det_dk<K>(th.data(), idx, w.data()) ==
        doctest::Approx(a.template log_det_dk<K>(&t, idx, nullptr)).epsilon(1e-13).scale(1.0));
}

}  // namespace

TEST_CASE("toeplitz_pdmat: theta = (t, 0, ..., 0) is ar1_pdmat at t") {
  for (int p = 2; p <= 6; ++p)
    for (double t : {-3.0, -0.4, 0.0, 0.8, 2.5}) {
      const lp::toeplitz_pdmat s(p);
      const lp::ar1_pdmat a(p);
      const int np = lp::packed_size(p);
      std::vector<double> th(p - 1, 0.0), w(s.scratch_size()), x(np), y(np);
      th[0] = t;
      s.matrix(th.data(), x.data(), w.data()), a.matrix(&t, y.data(), nullptr);
      CHECK(rel_gap(x, y) < 1e-15);
      s.inverse(th.data(), x.data(), w.data()), a.inverse(&t, y.data(), nullptr);
      CHECK(rel_gap(x, y) < 1e-15);
      s.cholesky(th.data(), x.data(), w.data()), a.cholesky(&t, y.data(), nullptr);
      CHECK(rel_gap(x, y) < 1e-15);
      CHECK(s.log_det(th.data(), w.data()) == doctest::Approx(a.log_det(&t, nullptr)).epsilon(1e-15).scale(1.0));
      ar1_twin_order<1>(p, t);
      ar1_twin_order<2>(p, t);
      ar1_twin_order<3>(p, t);
      ar1_twin_order<4>(p, t);
      ar1_twin_order<5>(p, t);
      ar1_twin_order<6>(p, t);
    }
}

TEST_CASE("toeplitz_pdmat: M^-1 M = I, L L^T = M, p = 1 and the scaling composition") {
  for (int p = 1; p <= 8; ++p) {
    const lp::toeplitz_pdmat s(p);
    CHECK(s.n_params() == p - 1);
    const int np = lp::packed_size(p);
    std::vector<double> th(std::max(1, p - 1)), w(std::max(1, s.scratch_size())), m(np), mi(np), l(np);
    for (int k = 0; k + 1 < p; ++k) th[k] = 0.9 * std::sin(1.7 * k + 0.3);
    s.matrix(th.data(), m.data(), w.data());
    s.inverse(th.data(), mi.data(), w.data());
    s.cholesky(th.data(), l.data(), w.data());
    auto at = [&](const std::vector<double>& a, int i, int j) { return i >= j ? a[lp::packed_index(i, j)] : a[lp::packed_index(j, i)]; };
    double e1 = 0.0, e2 = 0.0;
    for (int i = 0; i < p; ++i) {
      CHECK(m[lp::packed_index(i, i)] == 1.0);
      for (int j = 0; j < p; ++j) {
        double r1 = 0.0, r2 = 0.0;
        for (int k = 0; k < p; ++k) r1 += at(mi, i, k) * at(m, k, j);
        for (int k = 0; k <= std::min(i, j); ++k) r2 += l[lp::packed_index(i, k)] * l[lp::packed_index(j, k)];
        e1 = std::fmax(e1, std::fabs(r1 - (i == j ? 1.0 : 0.0)));
        e2 = std::fmax(e2, std::fabs(r2 - at(m, i, j)));
      }
    }
    CHECK(e1 < 1e-13);
    CHECK(e2 < 1e-15);
    if (p == 1) {
      CHECK(m[0] == 1.0);
      CHECK(s.log_det(th.data(), w.data()) == 0.0);
    }
  }
  // D R D with R Toeplitz: entries e^(s_i + s_j) m_|i-j|
  using scaled_toeplitz = lp::unit_diagonal_scaled_pdmat<lp::toeplitz_pdmat>;
  static_assert(lp::pdmat_structure<scaled_toeplitz>);
  const int p = 4;
  const scaled_toeplitz c(lp::toeplitz_pdmat{p});
  const lp::toeplitz_pdmat r(p);
  std::vector<double> th = {0.1, -0.2, 0.3, 0.05, 0.7, -0.4, 0.2}, w(c.scratch_size()), x(10), y(10);
  c.matrix(th.data(), x.data(), w.data());
  r.matrix(th.data() + p, y.data(), w.data());
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j)
      CHECK(x[lp::packed_index(i, j)] == doctest::Approx(std::exp(th[i] + th[j]) * y[lp::packed_index(i, j)]).epsilon(1e-15));
}

