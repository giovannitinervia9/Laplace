// The code shared by mvgaussian1_distrib (covariance form) and
// mvgaussian2_distrib (precision form); note/distrib_multivariate.md sections
// 3 and 6. Prec = false: M = Sigma, the quadratic form uses W = M^-1 and the
// log-density has -1/2 log det M; Prec = true: M = Q, W = M and +1/2 log det M.
//
//   l = -d/2 log 2 pi + c log det M - 1/2 r' W r,   r = y - mu, c = -+1/2
//   block (A, T):  A = 0: c d_T log det M - 1/2 r' d_T W r
//                  A = 1: d_T W r
//                  A = 2: -d_T W
//                  A >= 3: zero
// The expected information in theta, with V = W^-1 (the covariance),
//   I_ab = -c d_ab log det M + 1/2 tr(V d_ab W),
//   d_T I_ab = -c d_abT log det M + 1/2 sum_(U subset T) tr(d_U V d_(ab, T\U) W),
// Leibniz's rule on the trace; no sandwich V W_a V W_b is formed. For the
// covariance form over a structure with its Cholesky factor (closed_factor),
// the information takes the factor form instead, which does not cancel near a
// singular Sigma (decided 2026-11-16, note/distrib_multivariate.md section 12):
//   I_ab = 1/2 tr(B_a B_b),  B_a = G_a + G_a',  G_a = L^-1 d_a L,
//   d_T I_ab = 1/2 sum_(U subset T) tr(d_U B_a d_(T\U) B_b),
// every derivative of G through whitened lower-triangular products only:
// H[m] = L^-1 L[m] (forward substitution, L[m] the derivative of L along the
// indices at the positions of the mask m), X[S, R] = Y[S] L[R] with
// X[0, R] = H[R], X[S, R] = -sum_(0 != T subset S) H[T] X[S\T, R]
// (Y = L^-1), and d_S G_a = sum_(S1 subset S) X[S1, {a} + (S\S1)].
//
// The theta table (decided 2026-11-16): for the orders k = 1..k_max, the
// sorted k-tuples of {0..q-1} in lexicographic order, one entry each of
// np + 2 doubles: a flag (1 when d_T W is not zero by structure),
// d_T log det M, and d_T W packed. Filled once per theta, one entry per call;
// the unit functions read it when it covers the order asked for.
#pragma once

#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace laplace::distrib::detail_mvgaussian {

namespace mv = laplace::distrib::detail_mv;
using pdmat::packed_index;
using pdmat::packed_size;

template <bool Prec>
inline constexpr double log_det_coef = Prec ? 0.5 : -0.5;

// ------------------------------------------------------------- theta table
LAPLACE_HD int tuples_of_order(int q, int k) { return binomial_coef(q + k - 1, k); }
// entries of the orders 1..k-1
LAPLACE_HD int table_offset(int q, int k) {
  int o = 0;
  for (int j = 1; j < k; ++j) o += tuples_of_order(q, j);
  return o;
}
// lexicographic rank of the sorted k-tuple idx among the sorted k-tuples of {0..q-1}
LAPLACE_HD int tuple_rank(int q, int k, const int* idx) {
  int pos = 0, prev = 0;
  for (int j = 0; j < k; ++j) {
    for (int v = prev; v < idx[j]; ++v) pos += binomial_coef(q - v + k - j - 2, k - j - 1);
    prev = idx[j];
  }
  return pos;
}
// the sorted k-tuple of rank r
LAPLACE_HD void tuple_unrank(int q, int k, int r, int* idx) {
  int v = 0;
  for (int j = 0; j < k; ++j) {
    for (;;) {
      const int c = binomial_coef(q - v + k - j - 2, k - j - 1);
      if (r < c) break;
      r -= c;
      ++v;
    }
    idx[j] = v;
  }
}
template <class S>
LAPLACE_HD int table_entry_size(const S& s) { return packed_size(s.dim) + 2; }

// the entry of the table for the tuple idx of order K, or nullptr when the
// state carries no table of that order
template <int K, class St, class S>
LAPLACE_HD const double* table_entry(const S& s, const St& st, const int* idx) {
  if (st.table == nullptr || K > st.table_order) return nullptr;
  const int q = s.n_params();
  const int e = table_offset(q, K) + tuple_rank(q, K, idx);
  return st.table + std::size_t(e) * std::size_t(table_entry_size(s));
}

template <bool Prec, class St, class S>
LAPLACE_HD void fill_theta_table_entry(const S& s, const St& st, int k_max, int entry, double* table, double* w) {
  const int q = s.n_params();
  int k = 1;
  while (k < k_max && entry >= table_offset(q, k + 1)) ++k;
  int idx[6];
  tuple_unrank(q, k, entry - table_offset(q, k), idx);
  double* e = table + std::size_t(entry) * std::size_t(table_entry_size(s));
  const bool nz = mv::structure_matrix_runtime<!Prec>(s, k, st.theta, idx, e + 2, w);
  e[0] = nz ? 1.0 : 0.0;
  e[1] = mv::structure_log_det_runtime(s, k, st.theta, idx, w);
}

template <bool Prec, class St, class S>
LAPLACE_HD St theta_state(const S& s, const double* theta, double* w) {
  const int np = packed_size(s.dim);
  double* W = w;
  double* L = w + np;
  double* ws = w + 2 * np;
  mv::structure_matrix_dk<!Prec, 0>(s, theta, nullptr, W, ws);
  mv::structure_factor(s, theta, L, ws);
  return St{theta, s.log_det(theta, ws), W, L};
}

template <bool Prec, class St, class S>
LAPLACE_HD double logpdf(const S& s, const St& st, const double* y, const double* mu) {
  return -0.5 * s.dim * mv::log_two_pi + log_det_coef<Prec> * st.log_det -
         0.5 * mv::packed_quadratic(s.dim, st.quadratic, y, mu);
}

template <bool Prec, int K, int A, class St, class S>
LAPLACE_HD bool logpdf_dk(const S& s, const St& st, const double* y, const double* mu, const int* idx, double* out,
                          double* w) {
  static_assert(A >= 0 && K >= 0 && A + K <= 6, "logpdf_dk: A + K <= 6");
  const int d = s.dim, np = packed_size(d);
  if constexpr (A >= 3) {
    return false;
  } else if constexpr (K == 0) {
    if constexpr (A == 0) out[0] = logpdf<Prec>(s, st, y, mu);
    else if constexpr (A == 1) mv::packed_times_residual(d, st.quadratic, y, mu, 1.0, out);
    else
      for (int k = 0; k < np; ++k) out[k] = -st.quadratic[k];
    return true;
  } else if constexpr (A == 2) {
    if (const double* e = table_entry<K>(s, st, idx)) {
      if (e[0] == 0.0) return false;
      for (int k = 0; k < np; ++k) out[k] = -e[2 + k];
      return true;
    }
    if (!mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, out, w)) return false;
    for (int k = 0; k < np; ++k) out[k] = -out[k];
    return true;
  } else {
    const double* e = table_entry<K>(s, st, idx);
    const double* WT = e ? e + 2 : w;
    double* ws = w + np;
    const bool nz = e ? e[0] != 0.0 : mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, w, ws);
    if constexpr (A == 1) {
      if (!nz) return false;
      mv::packed_times_residual(d, WT, y, mu, 1.0, out);
      return true;
    } else {
      const double ld = e ? e[1] : mv::structure_log_det_dk<K>(s, st.theta, idx, ws);
      if (!nz && ld == 0.0) return false;
      out[0] = log_det_coef<Prec> * ld - (nz ? 0.5 * mv::packed_quadratic(d, WT, y, mu) : 0.0);
      return true;
    }
  }
}

template <bool Prec, int J, int K, class St, class S>
LAPLACE_HD bool logpdf_dy_dk(const S& s, const St& st, const double* y, const double* mu, const int* idx, double* out,
                             double* w) {
  if (!logpdf_dk<Prec, K, J>(s, st, y, mu, idx, out, w)) return false;
  if constexpr (J % 2 == 1) {
    const int n = J == 1 ? s.dim : packed_size(s.dim);
    for (int k = 0; k < n; ++k) out[k] = -out[k];
  }
  return true;
}

// ------------------------------------------- the information, factor form
// C = A B for lower-triangular A, B (full d x d, row-major, zeros above)
LAPLACE_HD void lower_product(int d, const double* A, const double* B, double* C) {
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      double r = 0.0;
      for (int k = j; k <= i; ++k) r += A[i * d + k] * B[k * d + j];
      C[i * d + j] = j <= i ? r : 0.0;
    }
}

// doubles of scratch of expected_info_factor at K = 4: H (64), d_S G for the
// two members of the pair (2 x 16), X[., R] (16) and a product, full d x d
// each; a packed derivative of L; then the structure's scratch
template <class S>
LAPLACE_HD int factor_info_scratch(const S& s) {
  return 113 * s.dim * s.dim + packed_size(s.dim) + s.scratch_size();
}

// Traceless (the multivariate t, detail/mvstudent_t_common.hpp): 1/2 tr(Btilde_a
// Btilde_b) instead, Btilde = B - tr(B)/d I, the diagonal of each d_S G centred
// before the products, so that a B proportional to the identity gives zero
// exactly.
template <int K, bool Traceless = false, class St, class S>
LAPLACE_HD bool expected_info_factor(const S& s, const St& st, const int* pair, const int* idx, double* out,
                                     double* w) {
  constexpr int NT = 1 << K, NM = 4 * NT;  // masks over the positions (a, b, T)
  const int d = s.dim, dd = d * d;
  double* H = w;
  double* dG = H + 64 * dd;
  double* X = dG + 32 * dd;
  double* tmp = X + 16 * dd;
  double* Lm = tmp + dd;
  double* ws = Lm + packed_size(d);
  const double* L = st.factor;
  bool hz[NM];
  for (int m = 1; m < NM; ++m) {
    hz[m] = false;
    if ((m & 3) == 3) continue;  // never both members of the pair
    int sub[6], k = 0;
    for (int q = 0; q < K + 2; ++q)
      if (m & (1 << q)) sub[k++] = q < 2 ? pair[q] : idx[q - 2];
    for (int i = 1; i < k; ++i)  // sort the sub-tuple
      for (int j = i; j > 0 && sub[j - 1] > sub[j]; --j) {
        const int t = sub[j];
        sub[j] = sub[j - 1], sub[j - 1] = t;
      }
    if (!pdmat::detail_pdmat::cholesky_dk_runtime(s, k, st.theta, sub, Lm, ws)) continue;
    hz[m] = true;
    double* Hm = H + m * dd;  // L Hm = L[m], column by column
    for (int j = 0; j < d; ++j)
      for (int i = 0; i < d; ++i) {
        if (i < j) {
          Hm[i * d + j] = 0.0;
          continue;
        }
        double r = Lm[packed_index(i, j)];
        for (int k2 = j; k2 < i; ++k2) r -= L[packed_index(i, k2)] * Hm[k2 * d + j];
        Hm[i * d + j] = r / L[packed_index(i, i)];
      }
  }
  // d_S G_c for c = a, b and S a mask over the positions of T
  bool gz[2][NT];
  for (int c = 0; c < 2; ++c) {
    double* G = dG + c * NT * dd;
    for (int e = 0; e < NT * dd; ++e) G[e] = 0.0;
    for (int S1 = 0; S1 < NT; ++S1) gz[c][S1] = false;
    for (int r = 0; r < NT; ++r) {
      const int R = (1 << c) | (r << 2);
      if (!hz[R]) continue;
      bool xz[NT];
      for (int S1 = 0; S1 < NT; ++S1) {
        xz[S1] = false;
        if (S1 & r) continue;
        double* XS = X + S1 * dd;
        if (S1 == 0) {
          for (int e = 0; e < dd; ++e) XS[e] = H[R * dd + e];
          xz[0] = true;
        } else {
          for (int e = 0; e < dd; ++e) XS[e] = 0.0;
          for (int T = S1; T != 0; T = (T - 1) & S1) {
            if (!hz[T << 2] || !xz[S1 ^ T]) continue;
            lower_product(d, H + (T << 2) * dd, X + (S1 ^ T) * dd, tmp);
            for (int e = 0; e < dd; ++e) XS[e] -= tmp[e];
            xz[S1] = true;
          }
        }
        if (!xz[S1]) continue;
        double* GS = G + (S1 | r) * dd;
        for (int e = 0; e < dd; ++e) GS[e] += XS[e];
        gz[c][S1 | r] = true;
      }
    }
  }
  // 1/2 tr(B1 B2) with B = G + G' and G lower: sum_i G1_ii G2_ii + sum_(j <= i) G1_ij G2_ij
  double res = 0.0;
  bool any = false;
  for (int U = 0; U < NT; ++U) {
    const int V = (NT - 1) ^ U;
    if (!gz[0][U] || !gz[1][V]) continue;
    any = true;
    const double* G1 = dG + U * dd;
    const double* G2 = dG + (NT + V) * dd;
    if constexpr (Traceless) {
      double m1 = 0.0, m2 = 0.0;
      for (int i = 0; i < d; ++i) m1 += G1[i * d + i], m2 += G2[i * d + i];
      m1 /= d, m2 /= d;
      for (int i = 0; i < d; ++i) {
        res += 2.0 * (G1[i * d + i] - m1) * (G2[i * d + i] - m2);
        for (int j = 0; j < i; ++j) res += G1[i * d + j] * G2[i * d + j];
      }
      continue;
    }
    for (int i = 0; i < d; ++i) {
      res += G1[i * d + i] * G2[i * d + i];
      for (int j = 0; j <= i; ++j) res += G1[i * d + j] * G2[i * d + j];
    }
  }
  if (!any) return false;
  out[0] = res;
  return true;
}

// doubles of scratch of expected_info_dk
template <bool Prec, class S>
LAPLACE_HD int info_scratch_size(const S& s) {
  const int base = 2 * packed_size(s.dim) + s.scratch_size();
  if constexpr (!Prec && S::closed_factor) {
    const int f = factor_info_scratch(s);
    return f > base ? f : base;
  } else {
    return base;
  }
}

template <bool Prec, int K, class St, class S>
LAPLACE_HD bool expected_info_dk(const S& s, const St& st, int slot_a, int slot_b, const int* ab, const int* idx,
                                 double* out, double* w) {
  static_assert(K >= 0 && K <= 4, "expected_info_dk: K <= 4");
  const int d = s.dim, np = packed_size(d);
  if (slot_a != slot_b) return false;
  if (slot_a != theta_slot) {
    if constexpr (K == 0) {
      for (int k = 0; k < np; ++k) out[k] = st.quadratic[k];
      return true;
    } else {
      if (const double* e = table_entry<K>(s, st, idx)) {
        if (e[0] == 0.0) return false;
        for (int k = 0; k < np; ++k) out[k] = e[2 + k];
        return true;
      }
      return mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, out, w);
    }
  }
  const int pair[2] = {ab[0] <= ab[1] ? ab[0] : ab[1], ab[0] <= ab[1] ? ab[1] : ab[0]};
  if constexpr (!Prec && S::closed_factor) return expected_info_factor<K>(s, st, pair, idx, out, w);
  double* V = w;
  double* WT = w + np;
  double* ws = w + 2 * np;
  int full[K + 2];
  mv::merge_sorted(pair, 2, idx, K, full);
  double r = -log_det_coef<Prec> * mv::structure_log_det_runtime(s, K + 2, st.theta, full, ws);
  bool any = r != 0.0;
  for (int U = 0; U < (1 << K); ++U) {
    int u[K + 1], rest[K + 1], nu = 0, nr = 0;
    for (int q = 0; q < K; ++q) {
      if (U & (1 << q)) u[nu++] = idx[q];
      else rest[nr++] = idx[q];
    }
    if (!mv::structure_matrix_runtime<Prec>(s, nu, st.theta, u, V, ws)) continue;
    int m[K + 2];
    const int nm = mv::merge_sorted(pair, 2, rest, nr, m);
    if (!mv::structure_matrix_runtime<!Prec>(s, nm, st.theta, m, WT, ws)) continue;
    r += 0.5 * mv::packed_trace_product(d, V, WT);
    any = true;
  }
  if (!any) return false;
  out[0] = r;
  return true;
}

// y = mu + L z (Prec false, Sigma = L L') or mu + L^-T z (Prec true, Q = L L'),
// z_k the k-th standard normal of counter_uniform_source{seed, unit}
template <bool Prec, class St, class S>
LAPLACE_HD void rng(const S& s, const St& st, const double* mu, std::uint64_t seed, std::uint64_t unit, double* y) {
  const int d = s.dim;
  const double* L = st.factor;
  counter_uniform_source src{seed, unit};
  for (int k = 0; k < d; ++k) y[k] = standard_normal_by_inversion(src);
  for (int i = d - 1; i >= 0; --i) {
    if constexpr (Prec) {
      double t = y[i];
      for (int k = i + 1; k < d; ++k) t -= L[packed_index(k, i)] * y[k];
      y[i] = t / L[packed_index(i, i)];
    } else {
      double t = 0.0;
      for (int k = 0; k <= i; ++k) t += L[packed_index(i, k)] * y[k];
      y[i] = t;
    }
  }
  for (int k = 0; k < d; ++k) y[k] += mu[k];
}

// (host) theta from the sample covariance (divisor n), its eigenvalues
// floored at pdmat's start floor, inverted for the precision form, through the
// structure's params_from_matrix
template <bool Prec, class S>
void start_value(const S& s, const distrib_mv_sample_summary& sm, double* theta) {
  namespace dp = pdmat::detail_pdmat;
  const int d = s.dim;
  std::vector<double> m = sm.covariance;
  dp::floor_eigenvalues(d, m.data());
  if constexpr (Prec) {
    std::vector<double> lam, V;
    dp::eigen_decreasing(d, m.data(), lam, V);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j <= i; ++j) {
        double t = 0.0;
        for (int k = 0; k < d; ++k) t += V[i * d + k] * V[j * d + k] / lam[k];
        m[packed_index(i, j)] = t;
      }
  }
  s.params_from_matrix(m.data(), theta);
}

}  // namespace laplace::distrib::detail_mvgaussian
