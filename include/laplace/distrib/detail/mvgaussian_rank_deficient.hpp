// The rank-deficient classes of mvgaussian1_distrib (degenerate: Sigma = M(theta)
// singular) and mvgaussian2_distrib (improper: Q = M(theta) singular), over a
// psdmat whose null space is constant and supplied (note/distrib_multivariate.md
// sections 4 and 13, decided with Giovanni on 2026-11-16).
//
// rho the rank, U0 the orthonormal basis of the null space, P = U0 U0', c > 0:
//   l1 = -rho/2 log 2 pi - 1/2 log|Sigma|_+ - 1/2 r' Sigma^+ r  on mu + col(Sigma),
//        -inf where ||U0' r|| > 64 d eps ||r||,
//   l2 = -rho/2 log 2 pi + 1/2 log|Q|_+  - 1/2 r' Q r,
//   M^+ = (M + cP)^-1 - P/c,   d_T M^+ = d_T (M + cP)^-1   (T not empty),
// for any fixed c > 0, so c = tr M / rho at the theta of the state keeps
// M + cP well scaled and stays fixed while differentiating. With A = (M + cP)^-1
// and A[T] its derivative along the positions of the mask T,
//   A[Q] = -A sum_(0 != T subset Q) M[T] A[Q \ T].
// The draws: M + cP = L L'; degenerate y = mu + (I - P) L z (covariance
// (I - P)(M + cP)(I - P) = Sigma), improper constrained y = mu + (I - P) L^-T z
// (covariance (I - P) A (I - P) = Q^+).
//
// The adapter rank_deficient_view presents the psdmat to the code of the
// proper classes (detail/mvgaussian_common.hpp) as a structure whose "inverse"
// is M^+ and whose "log det" is log|M|_+: the blocks, the theta table and the
// expected information are that code, unchanged.
//
// ponytail: host only (the one structure of the class, weighted_sum_psdmat, is
// host only), dense packed loops with no blocking: about d^3 per product, so
// d_T Sigma^+ at d = 2000 takes seconds; a device or blocked version when a
// structure needs it.
#pragma once

#include <laplace/distrib/detail/mvgaussian_common.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace laplace::pdmat {
struct weighted_sum_psdmat;
}

namespace laplace::distrib {


namespace detail_mvgaussian {

// entry (i, j) of a packed symmetric matrix
inline double sym_at(const double* P, int i, int j) {
  return P[i >= j ? packed_index(i, j) : packed_index(j, i)];
}

// out = P x, P = U0 U0' the projector on the null space
template <class S>
void null_projection(const S& s, const double* x, double* out) {
  const auto& B = s.null_space_basis();
  const int p = s.dim, nn = p - s.rank();
  for (int i = 0; i < p; ++i) out[i] = 0.0;
  for (int k = 0; k < nn; ++k) {
    double v = 0.0;
    for (int i = 0; i < p; ++i) v += B(i, k) * x[i];
    for (int i = 0; i < p; ++i) out[i] += B(i, k) * v;
  }
}

// m (packed) += a P
template <class S>
void add_null_projector(const S& s, double a, double* m) {
  const auto& B = s.null_space_basis();
  const int p = s.dim, nn = p - s.rank();
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double v = 0.0;
      for (int k = 0; k < nn; ++k) v += B(i, k) * B(j, k);
      m[packed_index(i, j)] += a * v;
    }
}

// the factor of m + cP (in place into l) and A = (m + cP)^-1; Y has p x p doubles
template <class S>
void shifted_factor_inverse(const S& s, const double* m, double c, double* l, double* A, double* Y) {
  const int p = s.dim, np = packed_size(p);
  for (int k = 0; k < np; ++k) l[k] = m[k];
  add_null_projector(s, c, l);
  mv::packed_cholesky_in_place(p, l);
  pdmat::detail_pdmat::lower_inverse(p, l, Y);
  for (int k = 0; k < np; ++k) A[k] = 0.0;
  pdmat::detail_pdmat::add_gram(p, Y, Y, A);
}

// doubles of scratch of shifted_inverse_dk, before the structure's
inline int shifted_inverse_scratch(int p) { return 64 * packed_size(p) + p * p + packed_size(p); }

// d_T A along the K positions of idx, A = (M + cP)^-1 (packed, A0); false when
// zero by structure
template <int K, class S>
bool shifted_inverse_dk(const S& s, const double* A0, const double* t, const int* idx, double* out, double* w) {
  constexpr int NM = 1 << K;
  const int p = s.dim, np = packed_size(p);
  double* D = w;               // D[Q] = A[Q], packed
  double* acc = D + NM * np;   // sum_T M[T] A[Q \ T], full p x p
  double* Mt = acc + p * p;    // M[T], packed
  double* ws = Mt + np;
  bool dz[NM];
  for (int k = 0; k < np; ++k) D[k] = A0[k];
  dz[0] = true;
  for (int Q = 1; Q < NM; ++Q) {
    dz[Q] = false;
    for (int e = 0; e < p * p; ++e) acc[e] = 0.0;
    for (int T = Q; T != 0; T = (T - 1) & Q) {
      const int R = Q ^ T;
      if (!dz[R]) continue;
      int sub[K], k = 0;
      for (int q = 0; q < K; ++q)
        if (T & (1 << q)) sub[k++] = idx[q];
      if (!mv::structure_matrix_runtime<false>(s, k, t, sub, Mt, ws)) continue;
      dz[Q] = true;
      const double* DR = D + R * np;
      for (int i = 0; i < p; ++i)
        for (int l = 0; l < p; ++l) {
          const double a = sym_at(Mt, i, l);
          if (a == 0.0) continue;
          for (int j = 0; j < p; ++j) acc[i * p + j] += a * sym_at(DR, l, j);
        }
    }
    if (!dz[Q]) continue;
    double* DQ = D + Q * np;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) {
        double r = 0.0;
        for (int l = 0; l < p; ++l) r += sym_at(A0, i, l) * acc[l * p + j];
        DQ[packed_index(i, j)] = -r;
      }
  }
  if (!dz[NM - 1]) return false;
  for (int k = 0; k < np; ++k) out[k] = D[(NM - 1) * np + k];
  return true;
}

// The psdmat as the code of the proper classes sees it: "inverse" M^+,
// "log det" log|M|_+ (from the structure's evaluated state when it keeps one).
// Its members are host and device functions because the shared code calls them
// from host and device functions; their bodies run on the host only (the
// structures of this class are host only), which keeps nvcc quiet.
template <class S>
struct rank_deficient_view {
  const S& s;
  const mvgaussian_rank_deficient_state& st;
  int dim;
  static constexpr bool closed_factor = false;

  rank_deficient_view(const S& s_, const mvgaussian_rank_deficient_state& st_) : s(s_), st(st_), dim(s_.dim) {}
  LAPLACE_HD int n_params() const {
    KOKKOS_IF_ON_HOST((return s.n_params();))
    KOKKOS_IF_ON_DEVICE((return 0;))
  }
  int scratch_size() const { return shifted_inverse_scratch(dim) + s.scratch_size(); }
  LAPLACE_HD void matrix(const double* t, double* out, double* w) const {
    KOKKOS_IF_ON_HOST((s.matrix(t, out, w);))
  }
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* t, const int (&i)[K], double* out, double* w) const {
    KOKKOS_IF_ON_HOST((return s.template matrix_dk<K>(t, i, out, w);))
    KOKKOS_IF_ON_DEVICE((return false;))
  }
  // M^+ = A - P/c (needs A: the degenerate state, the improper constrained state)
  LAPLACE_HD void inverse(const double*, double* out, double*) const {
    KOKKOS_IF_ON_HOST((for (int k = 0; k < packed_size(dim); ++k) out[k] = st.shifted_inverse[k];
                       add_null_projector(s, -1.0 / st.shift, out);))
  }
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* t, const int (&i)[K], double* out, double* w) const {
    KOKKOS_IF_ON_HOST((return shifted_inverse_dk<K>(*this, st.shifted_inverse, t, i, out, w);))
    KOKKOS_IF_ON_DEVICE((return false;))
  }
  LAPLACE_HD double log_det(const double*, double*) const { return st.log_det; }
  template <int K>
  LAPLACE_HD double log_det_dk(const double* t, const int (&i)[K], double* w) const {
    KOKKOS_IF_ON_HOST((if constexpr (requires { s.template log_pdet_dk_at<K>(i, w); }) {
      if (st.pdet != nullptr) return s.template log_pdet_dk_at<K>(i, st.pdet);
    } return s.template log_pdet_dk<K>(t, i, w);))
    KOKKOS_IF_ON_DEVICE((return 0.0;))
  }
};

// state layout in w: W, L, A (packed each), the structure's log_pdet state;
// then scratch (p x p and the structure's)
template <class S>
int rank_deficient_state_size(const S& s) {
  return 3 * packed_size(s.dim) + s.scratch_size();
}

// the state; with_factor: also L and A (always for the degenerate class)
template <bool Prec, class S>
mvgaussian_rank_deficient_state rank_deficient_state(const S& s, const double* theta, double* w, bool with_factor) {
  const int p = s.dim, np = packed_size(p);
  double* W = w;
  double* L = w + np;
  double* A = w + 2 * np;
  double* pdet = w + 3 * np;
  double* ws = w + rank_deficient_state_size(s);
  mvgaussian_rank_deficient_state st{theta, 0.0, W};
  s.matrix(theta, W, ws);
  double tr = 0.0;
  for (int i = 0; i < p; ++i) tr += W[packed_index(i, i)];
  st.shift = s.rank() > 0 && tr > 0.0 ? tr / s.rank() : 1.0;
  if constexpr (requires { s.log_pdet_state(theta, pdet); }) {
    st.log_det = s.log_pdet_state(theta, pdet);
    st.pdet = pdet;
  } else {
    st.log_det = s.log_pdet(theta, ws);
  }
  if (with_factor || !Prec) {
    shifted_factor_inverse(s, W, st.shift, L, A, ws);
    st.factor = L;
    st.shifted_inverse = A;
  }
  if constexpr (!Prec) {  // W = Sigma^+ = A - P/c
    for (int k = 0; k < np; ++k) W[k] = A[k];
    add_null_projector(s, -1.0 / st.shift, W);
  }
  return st;
}

// Structures with d_ab M = delta_ab d_a M (each component linear in its own
// exp(theta_j)). For them the expected information of both classes follows
// from log|M|_+ alone, with B_a similar to e^(theta_a) M^+ S_a:
//   I_ab = 1/2 tr(B_a B_b) = -1/2 d_ab log|M|_+ + 1/2 delta_ab d_a log|M|_+,
//   d_T I_ab = -1/2 d_abT log|M|_+ + 1/2 delta_ab d_aT log|M|_+.
// The trace route through M^+ cancels (measured 2026-11-16: 1.8e17 ulp1N at
// theta components 9 apart) and is not offered; the information exists only
// for these structures.
template <class S>
inline constexpr bool exp_linear_structure = false;
template <>
inline constexpr bool exp_linear_structure<pdmat::weighted_sum_psdmat> = true;

template <bool Prec, int K, class S>
bool rank_deficient_info(const S& s, const mvgaussian_rank_deficient_state& st, int slot_a, int slot_b,
                         const int* ab, const int* idx, double* out, double* w) {
  static_assert(K >= 0 && K <= 4, "expected_info_dk: K <= 4");
  const rank_deficient_view<S> v(s, st);
  if (slot_a != slot_b) return false;
  if (slot_a != theta_slot) {
    if constexpr (K == 0) {
      for (int k = 0; k < packed_size(s.dim); ++k) out[k] = st.quadratic[k];
      return true;
    } else {
      return mv::structure_matrix_dk<!Prec, K>(v, st.theta, idx, out, w);
    }
  }
  const int pair[2] = {ab[0] <= ab[1] ? ab[0] : ab[1], ab[0] <= ab[1] ? ab[1] : ab[0]};
  int full[K + 2];
  mv::merge_sorted(pair, 2, idx, K, full);
  double r = -0.5 * mv::structure_log_det_runtime(v, K + 2, st.theta, full, w);
  if (pair[0] == pair[1]) {
    mv::merge_sorted(pair, 1, idx, K, full);
    r += 0.5 * mv::structure_log_det_runtime(v, K + 1, st.theta, full, w);
  }
  out[0] = r;
  return true;
}

// scratch of the unit functions (after the state), and of expected_info_dk
template <bool Prec, class S>
int rank_deficient_scratch(const S& s) {
  const int p = s.dim, np = packed_size(p);
  const int unit = Prec ? np + s.scratch_size() : np + shifted_inverse_scratch(p) + s.scratch_size();
  const int state = p * p + s.scratch_size();
  return unit > state ? unit : state;
}
template <bool Prec, class S>
int rank_deficient_info_scratch(const S& s) {
  return (Prec ? 0 : shifted_inverse_scratch(s.dim)) + s.scratch_size();
}

// log-density: the constant of the range, -inf off the support (degenerate)
template <bool Prec, class S>
double rank_deficient_logpdf(const S& s, const mvgaussian_rank_deficient_state& st, const double* y,
                             const double* mu) {
  const int p = s.dim;
  if constexpr (!Prec) {
    const auto& B = s.null_space_basis();
    double rr = 0.0, off = 0.0;
    for (int i = 0; i < p; ++i) rr += (y[i] - mu[i]) * (y[i] - mu[i]);
    for (int k = 0; k < p - s.rank(); ++k) {
      double v = 0.0;
      for (int i = 0; i < p; ++i) v += B(i, k) * (y[i] - mu[i]);
      off += v * v;
    }
    const double tau = 64.0 * p * std::numeric_limits<double>::epsilon();
    if (off > tau * tau * rr) return -std::numeric_limits<double>::infinity();
  }
  return -0.5 * s.rank() * mv::log_two_pi + log_det_coef<Prec> * st.log_det -
         0.5 * mv::packed_quadratic(p, st.quadratic, y, mu);
}

// y = mu + (I - P) L z (degenerate) or mu + (I - P) L^-T z (improper, constrained);
// tmp has p doubles
template <bool Prec, class S>
void rank_deficient_rng(const S& s, const mvgaussian_rank_deficient_state& st, const double* mu, std::uint64_t seed,
                        std::uint64_t unit, double* y, double* tmp) {
  const int p = s.dim;
  rank_deficient_view<S> v(s, st);
  rng<Prec>(v, st, mu, seed, unit, y);
  for (int i = 0; i < p; ++i) y[i] -= mu[i];
  null_projection(s, y, tmp);
  for (int i = 0; i < p; ++i) y[i] += mu[i] - tmp[i];
}

// (host) theta from the sample covariance C: degenerate (I - P) C (I - P),
// improper its pseudo-inverse (C_p + cP)^-1 - P/c, c = tr C_p / rho
template <bool Prec, class S>
void rank_deficient_start_value(const S& s, const distrib_mv_sample_summary& sm, double* theta) {
  const int p = s.dim, np = packed_size(p);
  std::vector<double> R(p * p), C(p * p), T(p * p), Cp(np);
  for (int i = 0; i < p; ++i) {  // R = I - P
    std::vector<double> e(p, 0.0), pe(p);
    e[i] = 1.0;
    null_projection(s, e.data(), pe.data());
    for (int j = 0; j < p; ++j) R[j * p + i] = (i == j ? 1.0 : 0.0) - pe[j];
  }
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j) C[i * p + j] = sym_at(sm.covariance.data(), i, j);
  for (int i = 0; i < p; ++i)
    for (int j = 0; j < p; ++j) {
      double r = 0.0;
      for (int k = 0; k < p; ++k) r += C[i * p + k] * R[k * p + j];
      T[i * p + j] = r;
    }
  double tr = 0.0;
  for (int i = 0; i < p; ++i)
    for (int j = 0; j <= i; ++j) {
      double r = 0.0;
      for (int k = 0; k < p; ++k) r += R[i * p + k] * T[k * p + j];
      Cp[packed_index(i, j)] = r;
      if (i == j) tr += r;
    }
  if constexpr (Prec) {
    const double c = s.rank() > 0 && tr > 0.0 ? tr / s.rank() : 1.0;
    std::vector<double> L(np), A(np);
    shifted_factor_inverse(s, Cp.data(), c, L.data(), A.data(), T.data());
    add_null_projector(s, -1.0 / c, A.data());
    Cp = A;
  }
  s.params_from_matrix(Cp.data(), theta);
}

}  // namespace detail_mvgaussian
}  // namespace laplace::distrib
