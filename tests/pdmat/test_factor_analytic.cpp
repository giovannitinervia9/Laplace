// factor_analytic_pdmat (include/laplace/pdmat/factor_analytic.hpp): M, M^-1
// and log det M with their derivatives to order 6 against the mpmath
// references (tools/reference/pdmat_factor_analytic.py, and
// pdmat_factor_analytic_small_psi.py for psi from e^-2 to e^-8) and against central
// differences of the derivative one order below, in a Kokkos kernel on the
// default execution space; on the host, the twins: d = 0 is diagonal_pdmat
// bit for bit, at d = p M - Psi is the log_cholesky_bijector's M, and the
// derivatives of M^-1 and log det M agree with the recursion on M (Cholesky
// of M and X[S] = -X[0] sum M[T] X[S \ T]). Errors:
//   ulpN   for M and M^-1: max_ij |y_ij - ref_ij| in ulps of max_ij |ref_ij|,
//          or, when that derivative vanishes, in ulps of max_ij of the value
//          of the quantity at the same point;
//   ulp    for log det: relative, in ulps of max(|ref|, 1).
// A reference below 1e-25 in absolute value is read as 0 (the noise of the
// differences at 640 bits). The difference check: along the last index v of
// the tuple, (f^(K-1)(theta + h e_v) - f^(K-1)(theta - h e_v))/(2h),
// h = 1e-5, within 1e-6 of the largest entry plus 1e-6 of that of order K-1.
// Set LAPLACE_PRINT_ERRORS=1 to print the table.
#include <doctest/doctest.h>

#include <laplace/bijector/log_cholesky.hpp>
#include <laplace/pdmat/diagonal.hpp>
#include <laplace/pdmat/factor_analytic.hpp>

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
using fa_pdmat = lp::factor_analytic_pdmat;

constexpr int max_p = 5;
constexpr int max_out = 15;  // packed_size(5)
constexpr int max_n = 14;    // parameters at (p, d) = (5, 2)

template <int K>
KOKKOS_INLINE_FUNCTION bool eval_k(const fa_pdmat& s, int quantity, const int* tup, const double* u, double* out,
                                   double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = tup[k];
  if (quantity == 0) return s.template matrix_dk<K>(u, idx, out, w);
  if (quantity == 1) return s.template inverse_dk<K>(u, idx, out, w);
  out[0] = s.template log_det_dk<K>(u, idx, w);
  return true;
}

// out = the quantity along the tuple of order K; zeroed if identically zero
KOKKOS_INLINE_FUNCTION void eval(const fa_pdmat& s, int quantity, int K, const int* tup, const double* u, double* out,
                                 double* w) {
  const int n = quantity == 2 ? 1 : lp::packed_size(s.dim);
  bool nz = true;
  switch (K) {
    case 0:
      if (quantity == 0) s.matrix(u, out, w);
      else if (quantity == 1) s.inverse(u, out, w);
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
    gr.pos.push_back(q == 2 ? 0 : lp::packed_index(std::stoi(f[6]), std::stoi(f[7])));
    gr.hi.push_back(std::strtod(f[8].c_str(), nullptr));
    gr.lo.push_back(std::strtod(f[9].c_str(), nullptr));
  }
  return g;
}

const char* qname[] = {"matrix", "inverse", "log_det", "log_det at d = p"};

}  // namespace

// The body sits in a named function with external linkage, never in the body of a
// TEST_CASE (note/brief_agenti.md: nvcc tags the device lambda with its enclosing function).
void pdmat_factor_analytic_mpmath_case() {
  auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_factor_analytic.csv");
  for (Group& G : read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_factor_analytic_small_psi.csv"))
    groups.push_back(std::move(G));
  const int ng = int(groups.size());
  REQUIRE(ng > 0);
  int sw = 0;
  for (int p = 1; p <= max_p; ++p)
    for (int d = 0; d <= p; ++d) sw = std::max(sw, fa_pdmat(p, d).scratch_size());

  Kokkos::View<int* [11]> meta("meta", ng);  // p, d, quantity, order, tuple[6], unused
  Kokkos::View<double* [max_n]> uu("u", ng);
  auto hm = Kokkos::create_mirror_view(meta);
  auto hu = Kokkos::create_mirror_view(uu);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    hm(g, 0) = G.p, hm(g, 1) = G.d, hm(g, 2) = G.quantity, hm(g, 3) = G.order;
    for (int k = 0; k < 6; ++k) hm(g, 4 + k) = k < G.order ? G.tup[k] : 0;
    hm(g, 10) = 0;
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
        "pdmat_factor_analytic", Kokkos::RangePolicy<>(g0, g1), KOKKOS_LAMBDA(int g) {
          const fa_pdmat s(meta(g, 0), meta(g, 1));
          const int q = meta(g, 2), K = meta(g, 3);
          int tup[6];
          for (int k = 0; k < 6; ++k) tup[k] = meta(g, 4 + k);
          double u[max_n], a[max_out], b[max_out];
          for (int k = 0; k < max_n; ++k) u[k] = uu(g, k);
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

  // max |value| of M and M^-1 at each point, the scale of a vanishing derivative
  std::map<std::tuple<int, int, std::string>, double> value_scale;
  for (const Group& G : groups)
    if (G.quantity < 2 && G.order == 0) {
      double s = 0.0;
      for (double h : G.hi) s = std::fmax(s, std::fabs(h));
      value_scale[{G.d, G.quantity, G.ukey}] = s;
    }
  // [quantity][order]; row 3: log det at d = p (the traces on M)
  double worst[4][7] = {};
  int worst_g[4][7] = {};
  std::map<std::tuple<int, int, std::string>, std::array<double, 3>> by_point;  // over the orders
  double worst_fd = 0.0;
  const double eps = std::ldexp(1.0, -52);
  for (int g = 0; g < ng; ++g) {
    const Group& G = groups[g];
    const int n = G.quantity == 2 ? 1 : lp::packed_size(G.p);
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
    const int row = G.quantity == 2 && G.d == G.p ? 3 : G.quantity;
    double& w = worst[row][G.order];
    if (!(ulp <= w)) w = ulp, worst_g[row][G.order] = g;
    double& wp = by_point[{G.p, G.d, G.ukey}][G.quantity];
    wp = std::fmax(wp, ulp);
    if (G.order > 0) {
      double lscale = 0.0;
      for (int k = 0; k < n; ++k) lscale = std::fmax(lscale, std::fabs(hlo(g, k)));
      for (int k = 0; k < n; ++k) {
        const double tol = 1e-6;
        const double dd = std::fabs(hfd(g, k) - hy(g, k)) / (tol * (scale + lscale));
        worst_fd = std::fmax(worst_fd, dd);
        if (!(dd <= 1.0))
          FAIL_CHECK(qname[G.quantity] << " order " << G.order << " p " << G.p << " d " << G.d
                                       << ": difference check " << dd);
      }
    }
  }

  // bounds on ulpN / ulp, [quantity][order]: the larger of the errors measured
  // on 2026-11-16 on gcc/OpenMP and CUDA over both reference files, times
  // 1.5, rounded up. The orders 2, 4, 6 of M^-1 come from the spread point at
  // (p, d) = (3, 1), where a change of one ulp in one theta_k moves the exact
  // derivative by 1.6e4 ulpN (an inflection along t_2: the even orders are
  // near a zero); log det at d = p (the traces on the augmented matrix) in a
  // row of its own.
  const double bound[4][7] = {{2, 1, 1, 1, 1, 1, 1},
                              {561, 2155, 25214, 8198, 27872, 8307, 35027},
                              {6, 7, 17, 50, 133, 539, 1390},
                              {53, 1254, 692, 1020, 2197, 6211, 6137}};
  const bool print = std::getenv("LAPLACE_PRINT_ERRORS") != nullptr;
  for (int q = 0; q < 4; ++q)
    for (int K = 0; K <= 6; ++K) {
      const Group& G = groups[worst_g[q][K]];
      std::string tup;
      for (int t : G.tup) tup += std::to_string(t) + " ";
      if (print)
        std::printf("factor_analytic %-16s K=%d  %10.2f  (p %d d %d, theta0 %g, tuple %s)\n", qname[q], K, worst[q][K],
                    G.p, G.d, G.u.empty() ? 0.0 : G.u[0], tup.c_str());
      CHECK_MESSAGE(worst[q][K] <= bound[q][K],
                    qname[q] << " order " << K << ": " << worst[q][K] << " ulp at p " << G.p << " d " << G.d);
    }
  if (print) {
    std::printf("worst difference check: %.3g of its tolerance\n", worst_fd);
    for (const auto& [key, e] : by_point)
      std::printf("p %d d %d theta0 %-8.3g  matrix %10.2f  inverse %12.2f  log_det %10.2f\n", std::get<0>(key),
                  std::get<1>(key), split_hex(std::get<2>(key))[0], e[0], e[1], e[2]);
  }
}

TEST_CASE("factor_analytic_pdmat: M, M^-1 and log det against mpmath and differences") {
  pdmat_factor_analytic_mpmath_case();
}

namespace {

// a fixed point of n parameters
std::vector<double> point(int n) {
  std::vector<double> u(n);
  for (int k = 0; k < n; ++k) u[k] = 0.37 * std::sin(1.3 * k + 0.4) + 0.05 * k;
  return u;
}

// a quantity of order K = idx.size() (runtime), 0 M, 1 M^-1; false when zero
template <int K>
bool fa_dk(const fa_pdmat& s, int q, const std::vector<int>& sub, const double* u, double* out, double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = sub[k];
  return q == 0 ? s.matrix_dk<K>(u, idx, out, w) : s.inverse_dk<K>(u, idx, out, w);
}
bool fa_any(const fa_pdmat& s, int q, const std::vector<int>& sub, const double* u, double* out, double* w) {
  switch (sub.size()) {
    case 1: return fa_dk<1>(s, q, sub, u, out, w);
    case 2: return fa_dk<2>(s, q, sub, u, out, w);
    case 3: return fa_dk<3>(s, q, sub, u, out, w);
    case 4: return fa_dk<4>(s, q, sub, u, out, w);
    case 5: return fa_dk<5>(s, q, sub, u, out, w);
    default: return fa_dk<6>(s, q, sub, u, out, w);
  }
}

using Mat = std::vector<double>;  // full p x p, row-major

Mat unpack(int p, const double* m) {
  Mat a(p * p);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) a[i * p + j] = a[j * p + i] = m[lp::packed_index(i, j)];
  return a;
}
Mat mul(int p, const Mat& a, const Mat& b) {
  Mat c(p * p, 0.0);
  for (int i = 0; i < p; ++i)
    for (int k = 0; k < p; ++k)
      for (int j = 0; j < p; ++j) c[i * p + j] += a[i * p + k] * b[k * p + j];
  return c;
}
// the inverse of a symmetric positive definite matrix by Cholesky
Mat spd_inverse(int p, const Mat& a) {
  Mat l(p * p, 0.0), x(p * p, 0.0);
  for (int j = 0; j < p; ++j) {
    double s = a[j * p + j];
    for (int k = 0; k < j; ++k) s -= l[j * p + k] * l[j * p + k];
    l[j * p + j] = std::sqrt(s);
    for (int i = j + 1; i < p; ++i) {
      double t = a[i * p + j];
      for (int k = 0; k < j; ++k) t -= l[i * p + k] * l[j * p + k];
      l[i * p + j] = t / l[j * p + j];
    }
  }
  for (int c = 0; c < p; ++c) {
    std::vector<double> z(p);
    for (int i = 0; i < p; ++i) {
      double s = i == c ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) s -= l[i * p + k] * z[k];
      z[i] = s / l[i * p + i];
    }
    for (int i = p - 1; i >= 0; --i) {
      double s = z[i];
      for (int k = i + 1; k < p; ++k) s -= l[k * p + i] * x[k * p + c];
      x[i * p + c] = s / l[i * p + i];
    }
  }
  return x;
}

// The recursion on M along the tuple idx: Xm[S] = -Xm[0] sum M[T] Xm[S \ T]
// over the derivatives of M, and d^F log det M = sum tr(Xm[T] M[f + rest]).
void recursion_on_m(const fa_pdmat& s, const double* u, const std::vector<int>& idx, Mat& inv, double& ld) {
  const int p = s.dim, K = int(idx.size()), NM = 1 << K, np = lp::packed_size(p);
  std::vector<double> w(s.scratch_size()), buf(np);
  std::vector<Mat> Md(NM), X(NM);
  std::vector<bool> mz(NM), xz(NM);
  s.matrix(u, buf.data(), w.data());
  Md[0] = unpack(p, buf.data());
  for (int T = 1; T < NM; ++T) {
    std::vector<int> sub;
    for (int q = 0; q < K; ++q)
      if (T & (1 << q)) sub.push_back(idx[q]);
    mz[T] = fa_any(s, 0, sub, u, buf.data(), w.data());
    if (mz[T]) Md[T] = unpack(p, buf.data());
  }
  X[0] = spd_inverse(p, Md[0]);
  xz[0] = true;
  for (int Q = 1; Q < NM; ++Q) {
    Mat acc(p * p, 0.0);
    xz[Q] = false;
    for (int T = Q; T != 0; T = (T - 1) & Q)
      if (mz[T] && xz[Q ^ T]) {
        const Mat t = mul(p, Md[T], X[Q ^ T]);
        for (int e = 0; e < p * p; ++e) acc[e] += t[e];
        xz[Q] = true;
      }
    if (!xz[Q]) continue;
    X[Q] = mul(p, X[0], acc);
    for (double& e : X[Q]) e = -e;
  }
  inv = xz[NM - 1] ? X[NM - 1] : Mat(p * p, 0.0);
  ld = 0.0;
  const int rest = (NM - 1) ^ 1;
  for (int T = rest;; T = (T - 1) & rest) {
    const int U = 1 | (rest ^ T);
    if (xz[T] && mz[U]) {
      const Mat t = mul(p, X[T], Md[U]);
      for (int a = 0; a < p; ++a) ld += t[a * p + a];
    }
    if (T == 0) break;
  }
}

template <int K>
double fa_log_det_dk(const fa_pdmat& s, const std::vector<int>& sub, const double* u, double* w) {
  int idx[K];
  for (int k = 0; k < K; ++k) idx[k] = sub[k];
  return s.log_det_dk<K>(u, idx, w);
}
double fa_log_det_any(const fa_pdmat& s, const std::vector<int>& sub, const double* u, double* w) {
  switch (sub.size()) {
    case 1: return fa_log_det_dk<1>(s, sub, u, w);
    case 2: return fa_log_det_dk<2>(s, sub, u, w);
    case 3: return fa_log_det_dk<3>(s, sub, u, w);
    case 4: return fa_log_det_dk<4>(s, sub, u, w);
    case 5: return fa_log_det_dk<5>(s, sub, u, w);
    default: return fa_log_det_dk<6>(s, sub, u, w);
  }
}

}  // namespace

TEST_CASE("factor_analytic_pdmat: d = 0 is diagonal_pdmat, bit for bit") {
  for (int p = 1; p <= 4; ++p) {
    const fa_pdmat s(p, 0);
    const lp::diagonal_pdmat g(p);
    REQUIRE(s.n_params() == p);
    const int np = lp::packed_size(p);
    std::vector<double> u = point(p), w(s.scratch_size() + 1), x(np), y(np);
    s.matrix(u.data(), x.data(), w.data()), g.matrix(u.data(), y.data(), nullptr);
    CHECK(x == y);
    s.inverse(u.data(), x.data(), w.data()), g.inverse(u.data(), y.data(), nullptr);
    CHECK(x == y);
    CHECK(s.log_det(u.data(), w.data()) == g.log_det(u.data(), nullptr));
    const int i3[3] = {p - 1, p - 1, p - 1}, i1[1] = {0};
    CHECK(s.matrix_d3(u.data(), i3, x.data(), w.data()) == g.matrix_d3(u.data(), i3, y.data(), nullptr));
    CHECK(x == y);
    CHECK(s.inverse_d3(u.data(), i3, x.data(), w.data()) == g.inverse_d3(u.data(), i3, y.data(), nullptr));
    CHECK(x == y);
    CHECK(s.log_det_d1(u.data(), i1, w.data()) == g.log_det_d1(u.data(), i1, nullptr));
  }
}

TEST_CASE("factor_analytic_pdmat: at d = p, M - Psi is the log-Cholesky M") {
  for (int p = 1; p <= 4; ++p) {
    const fa_pdmat s(p, p);
    const laplace::bijector::log_cholesky_bijector b(p);
    const int np = lp::packed_size(p), nl = s.n_loadings();
    REQUIRE(nl == np);
    std::vector<double> u = point(s.n_params()), w(s.scratch_size()), wb(b.scratch_size() + 1), x(np), y(np);
    s.matrix(u.data(), x.data(), w.data());
    b.inverse(u.data(), y.data(), wb.data());
    for (int i = 0; i < p; ++i) x[lp::packed_index(i, i)] -= std::exp(2.0 * u[nl + i]);
    for (int k = 0; k < np; ++k) CHECK(x[k] == doctest::Approx(y[k]).epsilon(1e-14).scale(1.0));
    // a derivative along the loadings alone is the bijector's
    const int i2[2] = {0, nl - 1};
    std::fill(x.begin(), x.end(), 0.0), std::fill(y.begin(), y.end(), 0.0);
    CHECK(s.matrix_d2(u.data(), i2, x.data(), w.data()) == b.inverse_d2(u.data(), i2, y.data(), wb.data()));
    for (int k = 0; k < np; ++k) CHECK(x[k] == doctest::Approx(y[k]).epsilon(1e-14).scale(1.0));
  }
}

TEST_CASE("factor_analytic_pdmat: M^-1 and log det derivatives against the recursion on M") {
  const int shapes[][2] = {{3, 1}, {4, 2}, {5, 2}, {6, 3}, {4, 4}};
  for (const auto& sh : shapes) {
    const fa_pdmat s(sh[0], sh[1]);
    const int p = s.dim, n = s.n_params(), np = lp::packed_size(p);
    const std::vector<double> u = point(n);
    std::vector<double> w(s.scratch_size()), out(np);
    std::vector<std::vector<int>> tuples = {{0}, {n - 1}, {0, n - 1}, {1, 2, n - 2}, {0, 0, 1, n - 1},
                                            {1, 1, 2, 3, n - 3}, {0, 1, 2, 3, n - 2, n - 1}, {2, 2, 2, 2, 2, 2}};
    for (auto& t : tuples) {
      std::sort(t.begin(), t.end());
      Mat gi;
      double gld;
      recursion_on_m(s, u.data(), t, gi, gld);
      std::fill(out.begin(), out.end(), 0.0);
      fa_any(s, 1, t, u.data(), out.data(), w.data());
      double scale = 0.0;
      for (double e : gi) scale = std::fmax(scale, std::fabs(e));
      for (int i = 0; i < p; ++i)
        for (int j = 0; j <= i; ++j)
          CHECK(std::fabs(out[lp::packed_index(i, j)] - gi[i * p + j]) <= 1e-12 * std::fmax(scale, 1.0));
      CHECK(fa_log_det_any(s, t, u.data(), w.data()) == doctest::Approx(gld).epsilon(1e-12).scale(1.0));
    }
  }
}

TEST_CASE("factor_analytic_pdmat: Ledermann's bound and the flags") {
  CHECK(lp::factor_analytic_max_factors(1) == 0);
  CHECK(lp::factor_analytic_max_factors(3) == 1);
  CHECK(lp::factor_analytic_max_factors(5) == 2);
  CHECK(lp::factor_analytic_max_factors(10) == 6);
  for (int p = 1; p <= 30; ++p) {
    const int d = lp::factor_analytic_max_factors(p);
    CHECK((p - d) * (p - d) >= p + d);
    CHECK((d == p || (p - d - 1) * (p - d - 1) < p + d + 1));
  }
  CHECK(fa_pdmat(5, 2).n_params() == 14);
  CHECK_FALSE(fa_pdmat::unit_diagonal);
  CHECK(fa_pdmat::closed_inverse);
  CHECK_FALSE(fa_pdmat::closed_factor);
  CHECK_FALSE(fa_pdmat::has_density);
}

// With LAPLACE_PRINT_ERRORS set: the error of the recursion on M (Cholesky of M
// and the recursion on the derivatives of M) at the reference points, in the
// measures of the first test case, for comparison with Woodbury's.
TEST_CASE("factor_analytic_pdmat: the recursion on M against mpmath (printed only)") {
  if (std::getenv("LAPLACE_PRINT_ERRORS") == nullptr) return;
  const auto groups = read_groups(std::string(LAPLACE_REFERENCE_DIR) + "/pdmat_factor_analytic.csv");
  std::map<std::tuple<int, int, std::string>, double> value_scale;
  for (const Group& G : groups)
    if (G.quantity == 1 && G.order == 0) {
      double s = 0.0;
      for (double h : G.hi) s = std::fmax(s, std::fabs(h));
      value_scale[{G.p, G.d, G.ukey}] = s;
    }
  std::map<std::tuple<int, int, std::string>, std::array<double, 2>> by_point;
  const double eps = std::ldexp(1.0, -52);
  for (const Group& G : groups) {
    if (G.quantity == 0 || G.order == 0) continue;
    const fa_pdmat s(G.p, G.d);
    Mat gi;
    double gld;
    recursion_on_m(s, G.u.data(), G.tup, gi, gld);
    double scale = 0.0, err = 0.0;
    auto ref = [&](std::size_t r) { return std::fabs(G.hi[r]) < 1e-25 ? 0.0 : G.hi[r]; };
    for (std::size_t r = 0; r < G.pos.size(); ++r) scale = std::fmax(scale, std::fabs(ref(r)));
    if (G.quantity == 2) {
      scale = std::fmax(scale, 1.0);
      err = std::fabs((gld - ref(0)) - (ref(0) == 0.0 ? 0.0 : G.lo[0]));
    } else {
      if (scale == 0.0) scale = value_scale.at({G.p, G.d, G.ukey});
      for (int i = 0, r = 0; i < G.p; ++i)
        for (int j = 0; j <= i; ++j, ++r)
          err = std::fmax(err, std::fabs((gi[i * G.p + j] - ref(r)) - (ref(r) == 0.0 ? 0.0 : G.lo[r])));
    }
    double& w = by_point[{G.p, G.d, G.ukey}][G.quantity - 1];
    w = std::fmax(w, err / (scale * eps));
  }
  for (const auto& [key, e] : by_point)
    std::printf("recursion on M: p %d d %d theta0 %-8.3g  inverse %12.2f  log_det %10.2f\n", std::get<0>(key),
                std::get<1>(key), split_hex(std::get<2>(key))[0], e[0], e[1]);
}
