// The pieces shared by the structures built on a p x d lower-trapezoidal matrix
// of loadings Lambda (factor_analytic_pdmat, reduced_rank_psdmat): Lambda_ij = 0
// for j > i, Lambda_jj = e^(theta) on its diagonal (j < d) and the other
// loadings free, stored by rows in theta[0 .. n_loadings() - 1] (row i has
// min(i + 1, d) of them, the last one the log of Lambda_ii when i < d).
//
// Also the derivatives of log det C for a d x d matrix C(theta) whose
// derivatives C[T] along the sub-tuples T of a tuple F are known: with X = C^-1,
//   X[S] = -X[0] sum_(T nonempty subset of S) C[T] X[S \ T],
//   d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)]),
// f the first position of F, skipping the C[T] that are zero by structure.
#pragma once

#include <laplace/pdmat/core.hpp>

namespace laplace::pdmat::detail_pdmat {

// in-place Cholesky of the full d x d row-major A (lower part used and
// overwritten, the strict upper part ignored)
LAPLACE_HD void fa_cholesky(int d, double* A) {
  for (int j = 0; j < d; ++j) {
    double s = A[j * d + j];
    for (int k = 0; k < j; ++k) s -= A[j * d + k] * A[j * d + k];
    const double r = Kokkos::sqrt(s);
    A[j * d + j] = r;
    for (int i = j + 1; i < d; ++i) {
      double t = A[i * d + j];
      for (int k = 0; k < j; ++k) t -= A[i * d + k] * A[j * d + k];
      A[i * d + j] = t / r;
    }
  }
}

// z = R^-1 b, R lower triangular d x d row-major
LAPLACE_HD void fa_forward(int d, const double* R, const double* b, double* z) {
  for (int i = 0; i < d; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) s -= R[i * d + k] * z[k];
    z[i] = s / R[i * d + i];
  }
}

struct trapezoid_loadings {
  int p, d;

  LAPLACE_HD int n_loadings() const { return p * d - d * (d - 1) / 2; }
  // first parameter of row i
  LAPLACE_HD int row_offset(int i) const { return i <= d ? i * (i + 1) / 2 : d * (d + 1) / 2 + (i - d) * d; }
  // row and column of the loading v < n_loadings()
  LAPLACE_HD void loading_position(int v, int& r, int& c) const {
    r = 0;
    while (row_offset(r + 1) <= v) ++r;
    c = v - row_offset(r);
  }
  // Lambda_jc, c <= min(j, d - 1)
  LAPLACE_HD double loading_value(const double* theta, int j, int c) const {
    const double x = theta[row_offset(j) + c];
    return j == c ? m::exp(x) : x;
  }
  // the derivative of order n >= 0 of the loading v in its own parameter
  LAPLACE_HD double loading_d(const double* theta, int v, int n) const {
    int r, c;
    loading_position(v, r, c);
    if (r == c) return m::exp(theta[v]);
    return n == 0 ? theta[v] : (n == 1 ? 1.0 : 0.0);
  }
  // Lambda, p x d row-major, zeros above the diagonal
  LAPLACE_HD void load_lambda(const double* theta, double* lam) const {
    for (int i = 0; i < p; ++i)
      for (int k = 0; k < d; ++k) lam[i * d + k] = k <= i ? loading_value(theta, i, k) : 0.0;
  }
  // the derivative of the row r of Lambda (lam) along the loadings sl[0..n-1]
  // of that row into out (d values); false when it is zero
  LAPLACE_HD bool lambda_row_d(const double* theta, const double* lam, int r, const int* sl, int n, double* out) const {
    if (n == 0) {
      for (int k = 0; k < d; ++k) out[k] = lam[r * d + k];
      return true;
    }
    for (int k = 1; k < n; ++k)
      if (sl[k] != sl[0]) return false;
    const double x = loading_d(theta, sl[0], n);
    if (x == 0.0) return false;
    int rr, c;
    loading_position(sl[0], rr, c);
    for (int k = 0; k < d; ++k) out[k] = k == c ? x : 0.0;
    return true;
  }

  // d^K (Lambda Lambda^T) along a sorted tuple of loadings into m (packed);
  // false, m untouched, when it is zero by structure. Loadings reach the
  // product only on at most two variables of one column.
  template <int K>
  LAPLACE_HD bool gram_dk(const double* theta, const int (&idx)[K], double* mm) const {
    const int v1 = idx[0], v2 = idx[K - 1];
    int m1 = 0;
    for (int k = 0; k < K; ++k) {
      if (idx[k] != v1 && idx[k] != v2) return false;
      m1 += idx[k] == v1;
    }
    int r1, c1, r2, c2;
    loading_position(v1, r1, c1);
    if (v1 == v2) {  // M_rr and M_rj, j > c, j != r
      const double drr = r1 == c1 ? ipow(2.0, K) * m::exp(2.0 * theta[v1]) : (K == 1 ? 2.0 * theta[v1] : (K == 2 ? 2.0 : 0.0));
      const double dk = loading_d(theta, v1, K);
      if (drr == 0.0 && dk == 0.0) return false;
      fill_zero(mm, packed_size(p));
      mm[packed_index(r1, r1)] = drr;
      for (int j = c1; j < p; ++j)
        if (j != r1) mm[r1 > j ? packed_index(r1, j) : packed_index(j, r1)] = dk * loading_value(theta, j, c1);
      return true;
    }
    loading_position(v2, r2, c2);
    if (c1 != c2) return false;
    const double a = loading_d(theta, v1, m1), b = loading_d(theta, v2, K - m1);
    if (a == 0.0 || b == 0.0) return false;
    fill_zero(mm, packed_size(p));
    mm[packed_index(r2, r1)] = a * b;  // r2 > r1: same column, later row
    return true;
  }
};

// X[T] for every mask T of a tuple of order K, from C[T] (C + T d^2, flagged
// by cnz; C[0] = C must be filled): X[0] = C^-1 = Y^T Y, Y = R^-1, C = R R^T,
// then the recursion. R and tmp are d x d buffers.
template <int K>
LAPLACE_HD void log_det_masks_inverse(int d, const double* C, const bool* cnz, double* R, double* tmp, double* X, bool* xnz) {
  constexpr int NM = 1 << K;
  const int dd = d * d;
  for (int e = 0; e < dd; ++e) R[e] = C[e];
  fa_cholesky(d, R);
  double* Y = tmp;
  for (int j = 0; j < d; ++j) {
    for (int i = 0; i < j; ++i) Y[i * d + j] = 0.0;
    Y[j * d + j] = 1.0 / R[j * d + j];
    for (int i = j + 1; i < d; ++i) {
      double t = 0.0;
      for (int k = j; k < i; ++k) t += R[i * d + k] * Y[k * d + j];
      Y[i * d + j] = -t / R[i * d + i];
    }
  }
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      double t = 0.0;
      for (int k = a > b ? a : b; k < d; ++k) t += Y[k * d + a] * Y[k * d + b];
      X[a * d + b] = t;
    }
  xnz[0] = true;
  double* acc = tmp;
  for (int Q = 1; Q < NM; ++Q) {
    xnz[Q] = false;
    for (int e = 0; e < dd; ++e) acc[e] = 0.0;
    for (int T = Q; T != 0; T = (T - 1) & Q) {
      const int Rm = Q ^ T;
      if (!cnz[T] || !xnz[Rm]) continue;
      xnz[Q] = true;
      const double* ct = C + T * dd;
      const double* xr = X + Rm * dd;
      for (int a = 0; a < d; ++a)
        for (int k = 0; k < d; ++k) {
          const double c = ct[a * d + k];
          for (int b = 0; b < d; ++b) acc[a * d + b] += c * xr[k * d + b];
        }
    }
    if (!xnz[Q]) continue;
    double* xq = X + Q * dd;
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int k = 0; k < d; ++k) t += X[a * d + k] * acc[k * d + b];
        xq[a * d + b] = -t;
      }
  }
}

// d^F log det C from C[T] and X[T] (log_det_masks_inverse)
template <int K>
LAPLACE_HD double log_det_masks_sum(int d, const double* C, const bool* cnz, const double* X, const bool* xnz) {
  const int full = (1 << K) - 1, rest = full ^ 1;
  double r = 0.0;
  for (int T = rest;; T = (T - 1) & rest) {
    const int U = 1 | (rest ^ T);
    if (xnz[T] && cnz[U]) {
      const double* x = X + T * d * d;
      const double* c = C + U * d * d;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) r += x[a * d + b] * c[b * d + a];
    }
    if (T == 0) break;
  }
  return r;
}

// C[T] for every mask T of a tuple of loadings and t's, C = sum_i w_i
// lambda_i lambda_i^T (+ I when unit): C[T] is zero unless every index of T
// belongs to one row r, and then w_r (-2)^(number of t_r) times the Leibniz
// sum over the loadings of lambda_r lambda_r^T. row_of(v) gives the row of
// parameter v (a loading, or a t when v >= n_loadings()); wt (p values) the
// row weights w_i; u, v two d-vectors.
template <int K, class RowOf>
LAPLACE_HD void loadings_gram_masks(const trapezoid_loadings& tz, const double* theta, const int (&idx)[K], const double* lam, const double* wt,
                                    bool unit, RowOf row_of, double* C, bool* cnz, double* u, double* v) {
  constexpr int NM = 1 << K;
  const int d = tz.d, dd = d * d, nl = tz.n_loadings();
  for (int a = 0; a < d; ++a)
    for (int b = 0; b < d; ++b) {
      double t = 0.0;
      for (int i = 0; i < tz.p; ++i) t += wt[i] * lam[i * d + a] * lam[i * d + b];
      C[a * d + b] = (unit && a == b ? 1.0 : 0.0) + t;
    }
  cnz[0] = true;
  for (int T = 1; T < NM; ++T) {
    int r = -1, nt = 0, nsl = 0, sl[K];
    bool one_row = true;
    for (int q = 0; q < K; ++q)
      if (T & (1 << q)) {
        const int rq = row_of(idx[q]);
        if (r < 0) r = rq;
        one_row = one_row && rq == r;
        if (idx[q] >= nl) ++nt;
        else sl[nsl++] = idx[q];
      }
    cnz[T] = false;
    if (!one_row) continue;
    const double f = ipow(-2.0, nt) * wt[r];
    double* ct = C + T * dd;
    for (int e = 0; e < dd; ++e) ct[e] = 0.0;
    for (int A = 0; A < (1 << nsl); ++A) {
      int sa[K], sb[K], na = 0, nb = 0;
      for (int k = 0; k < nsl; ++k) (A & (1 << k) ? sa[na++] : sb[nb++]) = sl[k];
      if (!tz.lambda_row_d(theta, lam, r, sa, na, u) || !tz.lambda_row_d(theta, lam, r, sb, nb, v)) continue;
      cnz[T] = true;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) ct[a * d + b] += f * u[a] * v[b];
    }
  }
}

}  // namespace laplace::pdmat::detail_pdmat
