// The distribution function of the multivariate Gaussian and its derivatives:
// the optional members cdf, ccdf, log_cdf, log_ccdf, log_rect_prob and their
// derivatives of mvgaussian1_distrib and mvgaussian2_distrib.
//
// With r = y - mu, the three regions are
//   cdf     Pr(Y <= y)      = Pr(X <= r),             X ~ N(0, Sigma),
//   ccdf    Pr(Y > y)       = Pr(X > r)  (componentwise),
//   rect    Pr(a < Y <= b)  = Pr(a - mu < X <= b - mu),
// each a rectangle Pr(lo < X <= hi) for laplace::math::log_mvn_rect.
//
// Derivatives. Let F(delta) = Pr(lo + delta < X <= hi + delta), and f_alpha =
// d^alpha F / F the normalized derivatives in the shift (log_mvn_rect_derivs).
// The location moves the rectangle by -mu, so d_mu = -d_delta, and y moves it
// by +y (both ends of the cdf and ccdf regions that are finite), d_y = d_delta.
// Sigma enters through the heat equation of the Gaussian convolution,
//   dF = (1/2) sum_ij (dSigma)_ij d_i d_j F,
// so a derivative along a set T of theta indices is, by Faa di Bruno over the
// set partitions sigma of T (Sigma(theta) is the inner function),
//   d_T F = sum_sigma (1/2)^|sigma| sum (prod_{c in sigma} (d_c Sigma)_{i_c j_c}) d_{i_1 j_1 ... } F,
// with d_c Sigma the structure's derivative along the indices of the block c
// (matrix_dk for the covariance form, inverse_dk for the precision form).
// The derivative of log F along items (the A location coordinates and the K
// theta indices) is the sum over the set partitions pi of the items of
// (-1)^(|pi| - 1) (|pi| - 1)! prod_{B in pi} (d_B F) / F.
// The contractions with the matrices are done one block at a time on the
// symmetric tensors of normalized derivatives, so every location tuple of the
// output comes out of the same chain.
#pragma once

#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/math/mvnormal.hpp>

namespace laplace::distrib::detail_mvgaussian_cdf {

namespace mv = laplace::distrib::detail_mv;
namespace mn = laplace::math::detail_mvnormal;
using pdmat::packed_index;
using pdmat::packed_size;

enum class region { lower, upper, rect };

LAPLACE_HD_FORCEINLINE double inf() { return math::detail::pos_inf(); }
LAPLACE_HD_FORCEINLINE double nan() { return math::detail::quiet_nan(); }

// Sigma, packed, from the state: L L' (covariance form, L the factor of Sigma)
// or (L L')^-1 (precision form, L the factor of Q); w has packed_size(d) doubles
template <bool Prec, class St>
LAPLACE_HD void sigma_from_state(int d, const St& st, double* S, double* w) {
  const double* L = st.factor;
  if constexpr (!Prec) {
    for (int i = 0; i < d; ++i)
      for (int j = 0; j <= i; ++j) {
        double v = 0.0;
        for (int p = 0; p <= j; ++p) v += L[packed_index(i, p)] * L[packed_index(j, p)];
        S[packed_index(i, j)] = v;
      }
  } else {
    // Sigma = Q^-1 = L^-T L^-1: invert L (lower), then (i, j) = sum_p (L^-1)_pi (L^-1)_pj
    double* Linv = w;  // packed, lower: packed_size(d) doubles
    for (int c = 0; c < d; ++c)
      for (int i = c; i < d; ++i) {
        double s = i == c ? 1.0 : 0.0;
        for (int p = c; p < i; ++p) s -= L[packed_index(i, p)] * Linv[packed_index(p, c)];
        Linv[packed_index(i, c)] = s / L[packed_index(i, i)];
      }
    for (int i = 0; i < d; ++i)
      for (int j = 0; j <= i; ++j) {
        double v = 0.0;
        for (int p = i; p < d; ++p) v += Linv[packed_index(p, i)] * Linv[packed_index(p, j)];
        S[packed_index(i, j)] = v;
      }
  }
}

// the rectangle of the region, relative to mu
template <region Rg>
LAPLACE_HD void limits(int d, const double* y, const double* b, const double* mu, double* lo, double* hi) {
  for (int i = 0; i < d; ++i) {
    if constexpr (Rg == region::lower) {
      lo[i] = -inf();
      hi[i] = y[i] - mu[i];
    } else if constexpr (Rg == region::upper) {
      lo[i] = y[i] - mu[i];
      hi[i] = inf();
    } else {
      lo[i] = y[i] == -inf() ? -inf() : y[i] - mu[i];
      hi[i] = b[i] == inf() ? inf() : b[i] - mu[i];
    }
  }
}

// doubles of w of the value: Sigma, the limits, the state's inverse, the inner probability
LAPLACE_HD int value_scratch(int d) { return packed_size(d) + 2 * d + (packed_size(d) > math::mvn_rect_scratch(d) ? packed_size(d) : math::mvn_rect_scratch(d)); }

template <bool Prec, region Rg, class St, class S>
LAPLACE_HD double log_value(const S& s, const St& st, const double* y, const double* b, const double* mu, double* w) {
  const int d = s.dim;
  double* Sg = w;
  double* lo = w + packed_size(d);
  double* hi = lo + d;
  double* ws = hi + d;
  sigma_from_state<Prec>(d, st, Sg, ws);
  limits<Rg>(d, y, b, mu, lo, hi);
  return math::log_mvn_rect(d, lo, hi, Sg, ws);
}

// ---------------------------------------------------------------- derivatives

// doubles of w of the derivative of orders (A in mu, K in theta): Sigma, limits,
// the normalized shift derivatives to order A + 2K and their workspace, the
// structure's derivative matrices of the 2^K - 1 subsets of the items, the
// tensors W[T][a] of normalized derivatives, two chain buffers, and the
// structure's own workspace
template <class S>
LAPLACE_HD int deriv_scratch(const S& s, int K, int A) {
  const int d = s.dim, N = A + 2 * K, np = packed_size(d);
  const int nf = mn::tuples_upto(d, N);
  int nW = 0;
  for (int a = 0; a <= A; ++a) nW += mn::tuples(d, a);
  const int general = np + 2 * d + nf + math::mvn_rect_derivs_scratch(d, N) + (1 << K) * np + (1 << K) * nW +
                      2 * mn::tuples(d, N) + np + s.scratch_size();
  if (K > 0 || A == 0) return general;
  // K = 0: the cumulant route
  const int cum = np + 2 * d + nf + math::mvn_rect_log_derivs_scratch(d, A) + np + s.scratch_size();
  return general > cum ? general : cum;
}

// the next restricted growth string of length n (a set partition); false after the last
template <int Nmax>
LAPLACE_HD bool next_partition(int n, int* g, int* mx) {
  // g[i] <= mx[i-1] + 1, mx[i] = max(g[0..i])
  for (int i = n - 1; i > 0; --i) {
    if (g[i] <= mx[i - 1]) {
      ++g[i];
      mx[i] = g[i] > mx[i - 1] ? g[i] : mx[i - 1];
      for (int j = i + 1; j < n; ++j) g[j] = 0, mx[j] = mx[i];
      return true;
    }
  }
  return false;
}

// d^K_theta d^A_mu log F of the region, the location tensor packed as logpdf_dk
template <bool Prec, region Rg, int K, int A, class St, class S>
LAPLACE_HD bool log_dk(const S& s, const St& st, const double* y, const double* b, const double* mu, const int* idx,
                       double* out, double* w) {
  static_assert(K >= 0 && A >= 0 && K + A <= 6, "orders K + A <= 6");
  const int d = s.dim, N = A + 2 * K, np = packed_size(d);
  const int nout = sym_tensor_size(d, A);
  double* Sg = w;
  double* lo = Sg + np;
  double* hi = lo + d;
  double* f = hi + d;
  double* wd = f + mn::tuples_upto(d, N);
  double* Dm = wd + math::mvn_rect_derivs_scratch(d, N);  // (1 << K) matrices
  double* Wt = Dm + (1 << K) * np;
  int nW = 0;
  for (int a = 0; a <= A; ++a) nW += mn::tuples(d, a);
  double* buf0 = Wt + (1 << K) * nW;
  double* buf1 = buf0 + mn::tuples(d, N);
  double* ws = buf1 + mn::tuples(d, N) + np;
  if constexpr (K == 0 && A > 0) {
    // derivatives in the location alone: through the cumulants of the
    // truncated law (log_mvn_rect_log_derivs), which do not cancel in the tail
    double* ws2 = f + mn::tuples_upto(d, N);
    sigma_from_state<Prec>(d, st, Sg, ws2);
    limits<Rg>(d, y, b, mu, lo, hi);
    math::log_mvn_rect_log_derivs(d, A, lo, hi, Sg, f, ws2);
    const double sg = A % 2 ? -1.0 : 1.0;  // d_mu = -d_delta
    int u[6] = {0, 0, 0, 0, 0, 0};
    do {
      out[mv::block_position(d, A, u)] = sg * f[mn::tuples_upto(d, A - 1) + mn::tuple_rank(d, A, u)];
    } while (mn::next_tuple(d, A, u));
    return true;
  }
  sigma_from_state<Prec>(d, st, Sg, ws);
  limits<Rg>(d, y, b, mu, lo, hi);
  const double lF = math::log_mvn_rect_derivs(d, N, lo, hi, Sg, f, wd);
  if (!(lF > -inf()) || lF != lF) {
    for (int q = 0; q < nout; ++q) out[q] = nan();
    return true;
  }
  // the structure's derivative matrices of every nonempty subset of the items
  bool nz[1 << K];
  nz[0] = false;
  for (unsigned c = 1; c < (1u << K); ++c) {
    int sub[6], n = 0;
    for (int q = 0; q < K; ++q)
      if ((c >> q) & 1u) sub[n++] = idx[q];
    nz[c] = mv::structure_matrix_runtime<Prec>(s, n, st.theta, sub, Dm + c * np, ws);
  }
  // offsets of the location orders inside W[T]
  int woff[7];
  woff[0] = 0;
  for (int a = 0; a < A; ++a) woff[a + 1] = woff[a] + mn::tuples(d, a);
  // W[T][a] = d_T d^a_mu F / F, by set partitions sigma of T
  for (unsigned T = 0; T < (1u << K); ++T) {
    double* WT = Wt + T * nW;
    for (int q = 0; q < nW; ++q) WT[q] = 0.0;
    int items[6], nt = 0;
    for (int q = 0; q < K; ++q)
      if ((T >> q) & 1u) items[nt++] = q;
    int g[6] = {0, 0, 0, 0, 0, 0}, mx[6] = {0, 0, 0, 0, 0, 0};
    do {
      // the blocks of sigma as subsets of the items
      const int nb = nt == 0 ? 0 : mx[nt - 1] + 1;
      unsigned blk[6] = {0, 0, 0, 0, 0, 0};
      for (int q = 0; q < nt; ++q) blk[g[q]] |= 1u << items[q];
      bool zero = false;
      for (int q = 0; q < nb; ++q)
        if (!nz[blk[q]]) zero = true;
      if (zero) continue;
      const double coef = 1.0 / double(1 << nb);
      for (int a = 0; a <= A; ++a) {
        // contract f of order a + 2 nb with the nb matrices, one at a time
        int n = a + 2 * nb;
        const double* src = f + mn::tuples_upto(d, n - 1);
        double* dst = buf0;
        for (int q = 0; q < nb; ++q) {
          const double* M = Dm + blk[q] * np;
          const int m = n - 2;
          int t[16] = {0}, t2[16];
          int pos = 0;
          do {
            double v = 0.0;
            for (int i = 0; i < d; ++i)
              for (int j = 0; j < d; ++j) {
                // the tuple t + {i, j}, sorted
                int r = 0, u = 0;
                bool pi = false, pj = false;
                const int lo2 = i < j ? i : j, hi2 = i < j ? j : i;
                for (; u < m; ++u) {
                  if (!pi && lo2 < t[u]) t2[r++] = lo2, pi = true;
                  if (pi && !pj && hi2 < t[u]) t2[r++] = hi2, pj = true;
                  t2[r++] = t[u];
                }
                if (!pi) t2[r++] = lo2;
                if (!pj) t2[r++] = hi2;
                v += M[packed_index(hi2, lo2)] * src[mn::tuple_rank(d, n, t2)];
              }
            dst[pos++] = v;
          } while (mn::next_tuple(d, m, t));
          src = dst;
          dst = dst == buf0 ? buf1 : buf0;
          n = m;
        }
        // n == a now; src holds the order-a tensor (or f itself when nb == 0)
        const double sg = (a % 2) ? -coef : coef;
        for (int q = 0; q < mn::tuples(d, a); ++q) WT[woff[a] + q] += sg * src[q];
      }
    } while (nt > 0 && next_partition<6>(nt, g, mx));
  }
  // log F: set partitions of the A + K items, for every location tuple
  const int n = A + K;
  int u[6] = {0, 0, 0, 0, 0, 0};
  do {
    double tot = 0.0;
    if (n == 0) {
      tot = lF;
    } else {
      int g[6] = {0, 0, 0, 0, 0, 0}, mx[6] = {0, 0, 0, 0, 0, 0};
      do {
        const int nb = mx[n - 1] + 1;
        double prod = 1.0;
        for (int q = 0; q < nb && prod != 0.0; ++q) {
          unsigned T = 0;
          int m[6], a = 0;
          for (int p = 0; p < n; ++p) {
            if (g[p] != q) continue;
            if (p < A) m[a++] = u[p];
            else T |= 1u << (p - A);
          }
          prod *= Wt[T * nW + woff[a] + mn::tuple_rank(d, a, m)];
        }
        double fac = 1.0;
        for (int q = 1; q < nb; ++q) fac *= -q;
        tot += fac * prod;
      } while (next_partition<6>(n, g, mx));
    }
    out[A == 0 ? 0 : mv::block_position(d, A, u)] = tot;
  } while (A > 0 && mn::next_tuple(d, A, u));
  return true;
}

// d^J_y d^K_theta d^A_mu log F: log F depends on y - mu, so d_y = -d_mu; the
// order-(J + A) location tensor goes to w[0, m), the call's scratch after it, and
// out[(u, v)] (u over y outside, v over mu inside, each in its block layout) is
// (-1)^J times its entry at the merged tuple, as detail_mv::location_dy_dk
template <bool Prec, region Rg, int J, int K, int A, class St, class S>
LAPLACE_HD bool log_dy_dk(const S& s, const St& st, const double* y, const double* mu, const int* idx, double* out,
                          double* w) {
  static_assert(J >= 0 && K >= 0 && A >= 0 && J + K + A <= 6, "J + K + A <= 6");
  const int d = s.dim, m = sym_tensor_size(d, J + A);
  log_dk<Prec, Rg, K, J + A>(s, st, y, nullptr, mu, idx, w, w + m);
  const double sign = J % 2 == 1 ? -1.0 : 1.0;
  int u[J + 1] = {}, v[A + 1] = {}, t[J + A + 1];
  int o = 0;
  do {
    for (int c = 0; c < A; ++c) v[c] = 0;
    do {
      mv::merge_sorted(u, J, v, A, t);
      out[o++] = sign * w[J + A == 0 ? 0 : mv::block_position(d, J + A, t)];
    } while (mv::next_block_tuple(d, A, v));
  } while (mv::next_block_tuple(d, J, u));
  return true;
}

}  // namespace laplace::distrib::detail_mvgaussian_cdf
