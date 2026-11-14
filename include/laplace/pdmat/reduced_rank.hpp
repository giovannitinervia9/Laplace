// reduced_rank_psdmat: M = Lambda Lambda^T of rank d (note/pdmat_structures.md
// section 10; interface in core.hpp, psdmat_structure). Lambda is p x d with
// the convention of factor_analytic_pdmat (detail/loadings.hpp): lower
// trapezoidal, Lambda_jj = e^(theta) on its diagonal, the other loadings free,
// theta the loadings by rows, q = pd - d(d-1)/2. Precondition 1 <= d <= p, not
// checked.
//
// log|M|_+ = log det C, C = Lambda^T Lambda (d x d): the nonzero eigenvalues
// of Lambda Lambda^T are those of Lambda^T Lambda. At d = p it is
// 2 sum_j theta_jj exactly (Lambda square and triangular); otherwise the value
// from a Cholesky factorization of C and the derivatives by the recursion on
// C of factor_analytic_pdmat, with C[T] zero unless every index of T is a
// loading of one row r, and then the Leibniz sum over those loadings of
// lambda_r lambda_r^T. M and Lambda have closed sparse derivatives.
#pragma once

#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/loadings.hpp>

namespace laplace::pdmat {

struct reduced_rank_psdmat : psdmat_orders<reduced_rank_psdmat> {
  int dim = 1;        // p
  int n_factors = 1;  // d, the rank

  reduced_rank_psdmat() = default;
  LAPLACE_HD reduced_rank_psdmat(int p, int d) : dim(p), n_factors(d) {}

  LAPLACE_HD int n_params() const { return dim * n_factors - n_factors * (n_factors - 1) / 2; }
  LAPLACE_HD int rank() const { return n_factors; }
  // Lambda (p d), the row weights (p, all 1), R and a temporary (d^2 each),
  // C[T] and X[T] (64 d^2 each) and two d-vectors
  LAPLACE_HD int scratch_size() const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    return p * d + p + 2 * d * d + 2 * nm * d * d + 2 * d;
  }

  // M = Lambda Lambda^T, packed
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const {
    const int p = dim, d = n_factors;
    double* lam = w;
    tz().load_lambda(theta, lam);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) s += lam[i * d + k] * lam[j * d + k];
        m[packed_index(i, j)] = s;
      }
  }

  // Lambda, p x d row-major with zeros above the diagonal
  LAPLACE_HD void factor(const double* theta, double* lam, [[maybe_unused]] double* w) const { tz().load_lambda(theta, lam); }

  // log|M|_+ = log det(Lambda^T Lambda)
  LAPLACE_HD double log_pdet(const double* theta, double* w) const {
    const int d = n_factors;
    if (d == dim) {
      double s = 0.0;
      for (int j = 0; j < d; ++j) s += theta[j * (j + 1) / 2 + j];
      return 2.0 * s;
    }
    const work s = carve(w);
    tz().load_lambda(theta, s.lam);
    build_c(s.lam, s.R);
    detail_pdmat::fa_cholesky(d, s.R);
    double lc = 0.0;
    for (int k = 0; k < d; ++k) lc += detail_pdmat::m::log(s.R[k * d + k]);
    return 2.0 * lc;
  }

  // d^K M along a sorted tuple: closed and sparse (false, m untouched, when
  // zero by structure)
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    return tz().template gram_dk<K>(theta, idx, m);
  }

  // d^K Lambda: nonzero only along one repeated loading (1 at order 1 for a
  // free loading, e^theta at every order for a diagonal one)
  template <int K>
  LAPLACE_HD bool factor_dk(const double* theta, const int (&idx)[K], double* lam, [[maybe_unused]] double* w) const {
    for (int k = 1; k < K; ++k)
      if (idx[k] != idx[0]) return false;
    const detail_pdmat::trapezoid_loadings z = tz();
    const double x = z.loading_d(theta, idx[0], K);
    if (x == 0.0) return false;
    int r, c;
    z.loading_position(idx[0], r, c);
    detail_pdmat::fill_zero(lam, dim * n_factors);
    lam[r * n_factors + c] = x;
    return true;
  }
  LAPLACE_HD bool factor_d1(const double* t, const int (&i)[1], double* l, double* w) const { return factor_dk<1>(t, i, l, w); }
  LAPLACE_HD bool factor_d2(const double* t, const int (&i)[2], double* l, double* w) const { return factor_dk<2>(t, i, l, w); }
  LAPLACE_HD bool factor_d3(const double* t, const int (&i)[3], double* l, double* w) const { return factor_dk<3>(t, i, l, w); }
  LAPLACE_HD bool factor_d4(const double* t, const int (&i)[4], double* l, double* w) const { return factor_dk<4>(t, i, l, w); }
  LAPLACE_HD bool factor_d5(const double* t, const int (&i)[5], double* l, double* w) const { return factor_dk<5>(t, i, l, w); }
  LAPLACE_HD bool factor_d6(const double* t, const int (&i)[6], double* l, double* w) const { return factor_dk<6>(t, i, l, w); }

  // d^K log|M|_+ along a sorted tuple; 0 when identically zero
  template <int K>
  LAPLACE_HD double log_pdet_dk(const double* theta, const int (&idx)[K], double* w) const {
    const int d = n_factors;
    const detail_pdmat::trapezoid_loadings z = tz();
    if (d == dim) {  // 2 sum_j theta_jj
      if (K > 1) return 0.0;
      int r, c;
      z.loading_position(idx[0], r, c);
      return r == c ? 2.0 : 0.0;
    }
    constexpr int NM = 1 << K;
    const work s = carve(w);
    bool cnz[NM], xnz[NM];
    z.load_lambda(theta, s.lam);
    for (int i = 0; i < dim; ++i) s.wt[i] = 1.0;
    detail_pdmat::loadings_gram_masks<K>(
        z, theta, idx, s.lam, s.wt, false,
        [&](int v) {
          int r, c;
          z.loading_position(v, r, c);
          return r;
        },
        s.C, cnz, s.u, s.v);
    detail_pdmat::log_det_masks_inverse<K>(d, s.C, cnz, s.R, s.tmp, s.X, xnz);
    return detail_pdmat::log_det_masks_sum<K>(d, s.C, cnz, s.X, xnz);
  }

 private:
  struct work {
    double *lam, *wt, *R, *tmp, *C, *X, *u, *v;
  };
  LAPLACE_HD work carve(double* w) const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    work s;
    s.lam = w;
    s.wt = s.lam + p * d;
    s.R = s.wt + p;
    s.tmp = s.R + d * d;
    s.C = s.tmp + d * d;
    s.X = s.C + nm * d * d;
    s.u = s.X + nm * d * d;
    s.v = s.u + d;
    return s;
  }
  LAPLACE_HD detail_pdmat::trapezoid_loadings tz() const { return {dim, n_factors}; }

  // C = Lambda^T Lambda into c (full d x d)
  LAPLACE_HD void build_c(const double* lam, double* c) const {
    const int p = dim, d = n_factors;
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int i = 0; i < p; ++i) t += lam[i * d + a] * lam[i * d + b];
        c[a * d + b] = t;
      }
  }
};

static_assert(psdmat_structure<reduced_rank_psdmat>);

}  // namespace laplace::pdmat
