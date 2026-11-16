// Host helpers of params_from_matrix, the map from a target symmetric matrix
// M_hat (packed lower triangle by rows) to parameters theta of a structure
// (note/distrib_multivariate.md section 6, note/pdmat_structures.md section
// 11). The result always lies inside the domain: correlations are clamped to
// [-start_rho_max, start_rho_max] and eigenvalues floored at
// start_eigen_floor times the largest, so that a start value is never at the
// boundary.
#pragma once

#include <laplace/bijector/matrix_log.hpp>
#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/loadings.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace laplace::pdmat::detail_pdmat {

constexpr double start_rho_max = 0.99;
constexpr double start_eigen_floor = 1e-6;

inline double clamp_rho(double r) { return std::clamp(r, -start_rho_max, start_rho_max); }

// the diagonal of m, each entry at least start_eigen_floor times the largest
// (all ones when none is positive)
inline std::vector<double> floored_diagonal(int p, const double* m) {
  std::vector<double> d(p);
  double top = 0.0;
  for (int i = 0; i < p; ++i) top = std::max(top, d[i] = m[packed_index(i, i)]);
  for (int i = 0; i < p; ++i) d[i] = top > 0.0 ? std::max(d[i], start_eigen_floor * top) : 1.0;
  return d;
}

// the correlation matrix of m (packed), from its floored diagonal
inline std::vector<double> correlation_of(int p, const double* m) {
  const auto d = floored_diagonal(p, m);
  std::vector<double> r(packed_size(p));
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) r[packed_index(i, j)] = i == j ? 1.0 : m[packed_index(i, j)] / std::sqrt(d[i] * d[j]);
  return r;
}

// eigenvalues lam in decreasing order and the eigenvectors as the columns of
// V (p x p row-major), by the cyclic Jacobi method of the matrix-log bijector
inline void eigen_decreasing(int p, const double* m, std::vector<double>& lam, std::vector<double>& V) {
  std::vector<double> A(p * p), W(p * p), l(p);
  bijector::detail_matrix_log::unpack_matrix(p, m, A.data());
  bijector::detail_matrix_log::jacobi(p, A.data(), W.data(), l.data());
  std::vector<int> o(p);
  std::iota(o.begin(), o.end(), 0);
  std::sort(o.begin(), o.end(), [&](int a, int b) { return l[a] > l[b]; });
  lam.resize(p);
  V.resize(p * p);
  for (int k = 0; k < p; ++k) {
    lam[k] = l[o[k]];
    for (int i = 0; i < p; ++i) V[i * p + k] = W[i * p + o[k]];
  }
}

// m (packed) with its eigenvalues raised to start_eigen_floor times the
// largest (to 1 when none is positive); m is rewritten only when one is
// raised, and the function says whether it was
inline bool floor_eigenvalues(int p, double* m) {
  std::vector<double> lam, V;
  eigen_decreasing(p, m, lam, V);
  const double low = lam[0] > 0.0 ? start_eigen_floor * lam[0] : 1.0;
  if (lam[p - 1] >= low) return false;
  for (auto& x : lam) x = std::max(x, low);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int k = 0; k < p; ++k) s += V[i * p + k] * lam[k] * V[j * p + k];
      m[packed_index(i, j)] = s;
    }
  return true;
}

// The loadings of the convention of detail/loadings.hpp (lower trapezoidal,
// positive diagonal) of a p x d Lambda with Lambda Lambda^T = A A^T, A p x d
// row-major, written by rows into theta (log on the diagonal): the first d
// columns of the Cholesky factor of A A^T, with each pivot floored at
// start_eigen_floor times the largest diagonal entry of A A^T.
inline void trapezoid_params(int p, int d, const double* A, double* theta) {
  auto b = [&](int i, int j) {
    double s = 0.0;
    for (int k = 0; k < d; ++k) s += A[i * d + k] * A[j * d + k];
    return s;
  };
  double top = 0.0;
  for (int i = 0; i < p; ++i) top = std::max(top, b(i, i));
  const double low = top > 0.0 ? start_eigen_floor * top : 1.0;
  std::vector<double> L(p * d, 0.0);
  for (int j = 0; j < d; ++j) {
    double s = b(j, j);
    for (int k = 0; k < j; ++k) s -= L[j * d + k] * L[j * d + k];
    L[j * d + j] = std::sqrt(std::max(s, low));
    for (int i = j + 1; i < p; ++i) {
      double t = b(i, j);
      for (int k = 0; k < j; ++k) t -= L[i * d + k] * L[j * d + k];
      L[i * d + j] = t / L[j * d + j];
    }
  }
  const trapezoid_loadings tz{p, d};
  for (int i = 0; i < p; ++i)
    for (int c = 0; c <= std::min(i, d - 1); ++c) theta[tz.row_offset(i) + c] = i == c ? std::log(L[i * d + c]) : L[i * d + c];
}

// x solving G x = b, G (n x n row-major) symmetric positive definite
inline std::vector<double> spd_solve_small(int n, std::vector<double> G, std::vector<double> b) {
  fa_cholesky(n, G.data());
  std::vector<double> z(n);
  fa_forward(n, G.data(), b.data(), z.data());
  for (int i = n - 1; i >= 0; --i) {
    double s = z[i];
    for (int k = i + 1; k < n; ++k) s -= G[k * n + i] * b[k];
    b[i] = s / G[i * n + i];
  }
  return b;
}

}  // namespace laplace::pdmat::detail_pdmat
