// The test matrices of laplace::dense and their references
// (tests/data/reference/dense.csv, written by tools/reference/dense.py), read
// for the test (tests/dense/test_dense.cpp) and the measurement tool
// (benchmarks/dense.cpp), with the residual norms both of them report.
#pragma once

#include <laplace/dense/dense.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace dense_cases {

using host_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using host_vector = Kokkos::View<double*, Kokkos::HostSpace>;

struct dense_case {
  int number = 0;
  std::string kind;  // symmetric, generalized, rectangular
  int m = 0, n = 0;
  host_matrix A, B;
  std::vector<double> value_hi, value_lo;  // w (increasing) or s (decreasing)
  int rank = -1;                           // rectangular only
};

inline std::vector<dense_case> read_dense_cases(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::map<int, dense_case> cases;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string f[9];
    for (auto& x : f) std::getline(ss, x, ',');
    const int number = std::stoi(f[0]);
    dense_case& c = cases[number];
    if (c.number == 0) {
      c.number = number;
      c.kind = f[1];
      c.m = std::stoi(f[2]);
      c.n = std::stoi(f[3]);
      c.A = host_matrix("A", c.m, c.n);
      if (c.kind == "generalized") c.B = host_matrix("B", c.n, c.n);
    }
    const int i = std::stoi(f[5]), j = std::stoi(f[6]);
    const double hi = std::strtod(f[7].c_str(), nullptr), lo = std::strtod(f[8].c_str(), nullptr);
    const std::string& q = f[4];
    if (q == "A") c.A(i, j) = hi;
    else if (q == "B") c.B(i, j) = hi;
    else if (q == "rank") c.rank = static_cast<int>(hi);
    else {  // w or s, in order
      c.value_hi.push_back(hi);
      c.value_lo.push_back(lo);
    }
  }
  std::vector<dense_case> out;
  for (auto& [k, c] : cases) out.push_back(c);
  return out;
}

// Frobenius norm of a matrix.
inline double frobenius(const host_matrix& a) {
  double s = 0.0;
  for (size_t j = 0; j < a.extent(1); ++j)
    for (size_t i = 0; i < a.extent(0); ++i) s += a(i, j) * a(i, j);
  return std::sqrt(s);
}

// C = op(X) op(Y), op the transpose when the flag is set (long double sums).
inline host_matrix product(const host_matrix& x, bool tx, const host_matrix& y, bool ty) {
  const size_t m = tx ? x.extent(1) : x.extent(0), k = tx ? x.extent(0) : x.extent(1);
  const size_t n = ty ? y.extent(0) : y.extent(1);
  host_matrix c("product", m, n);
  for (size_t j = 0; j < n; ++j)
    for (size_t i = 0; i < m; ++i) {
      long double s = 0.0L;
      for (size_t l = 0; l < k; ++l) s += (long double)(tx ? x(l, i) : x(i, l)) * (ty ? y(j, l) : y(l, j));
      c(i, j) = static_cast<double>(s);
    }
  return c;
}

// The full symmetric matrix from its lower triangle.
inline host_matrix symmetric_full(const host_matrix& a) {
  host_matrix s("symmetric", a.extent(0), a.extent(1));
  for (size_t j = 0; j < a.extent(1); ++j)
    for (size_t i = 0; i < a.extent(0); ++i) s(i, j) = i >= j ? a(i, j) : a(j, i);
  return s;
}

// ||X^T X - I||_F, the loss of orthonormality of the columns of X.
inline double orthonormality_error(const host_matrix& x) {
  host_matrix g = product(x, true, x, false);
  for (size_t i = 0; i < g.extent(0); ++i) g(i, i) -= 1.0;
  return frobenius(g);
}

// ||X - Y||_F
inline double difference(const host_matrix& x, const host_matrix& y) {
  double s = 0.0;
  for (size_t j = 0; j < x.extent(1); ++j)
    for (size_t i = 0; i < x.extent(0); ++i) s += (x(i, j) - y(i, j)) * (x(i, j) - y(i, j));
  return std::sqrt(s);
}

// X with its columns scaled by d.
inline host_matrix scale_columns(const host_matrix& x, const host_vector& d) {
  host_matrix y("scaled", x.extent(0), x.extent(1));
  for (size_t j = 0; j < x.extent(1); ++j)
    for (size_t i = 0; i < x.extent(0); ++i) y(i, j) = x(i, j) * d(j);
  return y;
}

// The largest |computed_i - reference_i| over the vector, in units of eps
// times the largest |reference|.
inline double values_error(const host_vector& v, const dense_case& c) {
  double scale = 0.0, worst = 0.0;
  for (size_t i = 0; i < c.value_hi.size(); ++i) scale = std::fmax(scale, std::fabs(c.value_hi[i]));
  for (size_t i = 0; i < c.value_hi.size(); ++i)
    worst = std::fmax(worst, std::fabs((v(i) - c.value_hi[i]) - c.value_lo[i]));
  return worst / (scale * 0x1p-52);
}

// The errors of laplace::dense on one case, each in units of eps: the
// values against mpmath (values_error) and the residuals of the identities,
// relative to the norms involved; the worst over the thin and full shapes.
struct dense_errors {
  double values = 0;          // w or s against the reference
  double residual = 0;        // A V - V diag(w), A V - B V diag(w), or A - U diag(s) V^T
  double orthonormality = 0;  // V^T V - I, V^T B V - I, or U^T U - I and V^T V - I
  double cholesky = -1;       // L L^T - A (symmetric positive definite cases)
  double triangular = -1;     // op(T) X - B for the four triangle and operation pairs
  double qr_residual = -1, qr_orthonormality = -1;
  double null_residual = -1, null_orthonormality = -1;  // C T, T^T T - I (m < n)
  int svd_rank = -1, qr_rank = -1, null_columns = -1;
  bool success = true;
};

inline dense_errors measure_case(const dense_case& c) {
  namespace ld = laplace::dense;
  constexpr double eps = 0x1p-52;
  dense_errors e;
  auto check = [&](const ld::dense_decomposition_status& s) { e.success = e.success && s.success; };
  const int m = c.m, n = c.n;
  if (c.kind == "symmetric") {
    host_vector w("w", n);
    host_matrix V("V", n, n);
    check(ld::symmetric_eigen_decomposition(c.A, w, V));
    e.values = values_error(w, c);
    const host_matrix a = symmetric_full(c.A);
    e.residual = difference(product(a, false, V, false), scale_columns(V, w)) / (frobenius(a) * eps);
    e.orthonormality = orthonormality_error(V) / eps;
    if (c.value_hi.front() > 64 * eps * c.value_hi.back()) {  // positive definite
      host_matrix L("L", n, n);
      check(ld::cholesky_decomposition(c.A, L));
      e.cholesky = difference(product(L, false, L, true), a) / (frobenius(a) * eps);
      host_matrix U("U", n, n);  // L^T, for the upper triangle
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) U(i, j) = L(j, i);
      e.triangular = 0;
      for (int t = 0; t < 4; ++t) {
        const bool lower = t < 2, transposed = t % 2 == 1;
        const host_matrix& T = lower ? L : U;
        host_matrix X("X", n, n);
        Kokkos::deep_copy(X, a);
        check(ld::triangular_solve(T, X, lower ? ld::dense_triangle::lower : ld::dense_triangle::upper,
                                   transposed ? ld::dense_operation::transpose : ld::dense_operation::identity));
        const double r = difference(product(T, transposed, X, false), a) / (frobenius(T) * frobenius(X) * eps);
        e.triangular = std::fmax(e.triangular, r);
      }
    }
  } else if (c.kind == "generalized") {
    host_vector w("w", n);
    host_matrix V("V", n, n);
    check(ld::generalized_symmetric_eigen_decomposition(c.A, c.B, w, V));
    const host_matrix a = symmetric_full(c.A), b = symmetric_full(c.B);
    const double na = frobenius(a), nb = frobenius(b);
    // values: each |w_i - reference| in units of eps kappa_i, with
    // kappa_i = (||A|| + |w_i| ||B||) ||v_i||^2 (v_i B-normalized) the condition
    // of the eigenvalue; the scale eps max |w| of the symmetric case would
    // charge the method with the ill-conditioning of the problem.
    for (int i = 0; i < n; ++i) {
      double v2 = 0;
      for (int k = 0; k < n; ++k) v2 += V(k, i) * V(k, i);
      const double kappa = (na + std::fabs(c.value_hi[i]) * nb) * v2;
      e.values = std::fmax(e.values, std::fabs((w(i) - c.value_hi[i]) - c.value_lo[i]) / (eps * kappa));
    }
    double wmax = 0;
    for (int i = 0; i < n; ++i) wmax = std::fmax(wmax, std::fabs(w(i)));
    const host_matrix bv = product(b, false, V, false);
    const double nv = frobenius(V);
    e.residual = difference(product(a, false, V, false), scale_columns(bv, w)) / ((na + nb * wmax) * nv * eps);
    // V^T B V - I relative to ||V||^2 ||B||, which grows with the condition of B
    host_matrix g = product(V, true, bv, false);
    for (int i = 0; i < n; ++i) g(i, i) -= 1.0;
    e.orthonormality = frobenius(g) / (nv * nv * nb * eps);
  } else {
    const int p = m < n ? m : n;
    const double norm = frobenius(c.A);
    for (int full = 0; full < 2; ++full) {
      host_matrix U("U", m, full ? m : p), V("V", n, full ? n : p);
      host_vector s("s", p);
      const auto st = ld::svd_decomposition(c.A, U, s, V);
      check(st);
      e.svd_rank = st.rank;
      e.values = values_error(s, c);
      host_matrix Uc("Uc", m, p), Vp("Vp", n, p);  // the first p columns
      for (int j = 0; j < p; ++j) {
        for (int i = 0; i < m; ++i) Uc(i, j) = U(i, j);
        for (int i = 0; i < n; ++i) Vp(i, j) = V(i, j);
      }
      e.residual = std::fmax(e.residual, difference(product(scale_columns(Uc, s), false, Vp, true), c.A) / (norm * eps));
      e.orthonormality = std::fmax(e.orthonormality, std::fmax(orthonormality_error(U), orthonormality_error(V)) / eps);

      const int qc = full ? m : p;
      host_matrix Q("Q", m, qc), R("R", qc, n);
      Kokkos::View<int*, Kokkos::HostSpace> perm("perm", n);
      const auto sq = ld::pivoted_qr_decomposition(c.A, Q, R, perm);
      check(sq);
      e.qr_rank = sq.rank;
      host_matrix ap("AP", m, n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) ap(i, j) = c.A(i, perm(j));
      e.qr_residual = std::fmax(e.qr_residual, difference(product(Q, false, R, false), ap) / (norm * eps));
      e.qr_orthonormality = std::fmax(e.qr_orthonormality, orthonormality_error(Q) / eps);
    }
    if (m < n) {
      host_matrix T("T", 1, 1);
      check(ld::null_space_basis(c.A, T));
      e.null_columns = static_cast<int>(T.extent(1));
      e.null_residual = frobenius(product(c.A, false, T, false)) / (norm * eps);
      e.null_orthonormality = orthonormality_error(T) / eps;
    }
  }
  return e;
}

}  // namespace dense_cases
