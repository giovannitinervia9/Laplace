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
// Leibniz's rule on the trace; no sandwich V W_a V W_b is formed.
#pragma once

#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>

#include <cstdint>
#include <vector>

namespace laplace::distrib::detail_mvgaussian {

namespace mv = laplace::distrib::detail_mv;
using pdmat::packed_index;
using pdmat::packed_size;

template <bool Prec>
inline constexpr double log_det_coef = Prec ? 0.5 : -0.5;

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

template <bool Prec, int A, int K, class St, class S>
LAPLACE_HD bool logpdf_dmu_dk(const S& s, const St& st, const double* y, const double* mu, const int* idx, double* out,
                              double* w) {
  static_assert(A >= 0 && K >= 0 && A + K <= 6, "logpdf_dmu_dk: A + K <= 6");
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
    if (!mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, out, w)) return false;
    for (int k = 0; k < np; ++k) out[k] = -out[k];
    return true;
  } else {
    double* WT = w;
    double* ws = w + np;
    const bool nz = mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, WT, ws);
    if constexpr (A == 1) {
      if (!nz) return false;
      mv::packed_times_residual(d, WT, y, mu, 1.0, out);
      return true;
    } else {
      const double ld = mv::structure_log_det_dk<K>(s, st.theta, idx, ws);
      if (!nz && ld == 0.0) return false;
      out[0] = log_det_coef<Prec> * ld - (nz ? 0.5 * mv::packed_quadratic(d, WT, y, mu) : 0.0);
      return true;
    }
  }
}

template <bool Prec, int J, int K, class St, class S>
LAPLACE_HD bool logpdf_dy_dk(const S& s, const St& st, const double* y, const double* mu, const int* idx, double* out,
                             double* w) {
  if (!logpdf_dmu_dk<Prec, J, K>(s, st, y, mu, idx, out, w)) return false;
  if constexpr (J % 2 == 1) {
    const int n = J == 1 ? s.dim : packed_size(s.dim);
    for (int k = 0; k < n; ++k) out[k] = -out[k];
  }
  return true;
}

template <bool Prec, int K, class St, class S>
LAPLACE_HD bool expected_info_dk(const S& s, const St& st, distrib_mv_info_block block, const int* ab, const int* idx,
                                 double* out, double* w) {
  static_assert(K >= 0 && K <= 4, "expected_info_dk: K <= 4");
  const int d = s.dim, np = packed_size(d);
  if (block == distrib_mv_info_block::vector_rest) return false;
  if (block == distrib_mv_info_block::vector_vector) {
    if constexpr (K == 0) {
      for (int k = 0; k < np; ++k) out[k] = st.quadratic[k];
      return true;
    } else {
      return mv::structure_matrix_dk<!Prec, K>(s, st.theta, idx, out, w);
    }
  }
  double* V = w;
  double* WT = w + np;
  double* ws = w + 2 * np;
  const int pair[2] = {ab[0] <= ab[1] ? ab[0] : ab[1], ab[0] <= ab[1] ? ab[1] : ab[0]};
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
