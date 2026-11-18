// The code shared by mvpowerexp1_distrib (scale matrix Sigma = M(theta)) and
// mvpowerexp2_distrib (inverse scale Q = M(theta)): the multivariate power
// exponential of Gomez, Gomez-Villegas and Marin (1998). Prec = false: W = M^-1,
// s = -1/2; Prec = true: W = M, s = +1/2. With r = y - mu, d the dimension,
// delta = r' W r the squared Mahalanobis distance and beta > 0 the shape,
//
//   l = -log Z(d, beta) + s log|M| - 1/2 delta^beta,
//   Z(d, beta) = 2^(1 + d/(2 beta)) pi^(d/2) Gamma(1 + d/(2 beta)) / (d Gamma(d/2)).
//
// "The rest" is (theta_0..theta_(q-1), beta): beta is the index q, the last.
// Sigma is the SCALE matrix and is not rescaled to the covariance (Giovanni,
// 2026-11-18), so that beta = 1 reduces exactly to mvgaussian1_distrib; the
// covariance is Sigma 2^(1/beta) Gamma((d+2)/(2 beta)) / (d Gamma(d/(2 beta))),
// whose factor is 1 at beta = 1.
//
// The blocks. l depends on (mu, theta) through delta (and log|M|), and
//   d_U delta = r' W_U r,  d_mu_i d_U delta = -2 (W_U r)_i,  d_mu_i d_mu_j d_U delta = 2 (W_U)_ij,
// zero with three or more mu, W_U = d_U W (U a subset of the theta positions).
// So by Faa di Bruno's formula a block with A positions in mu, the theta tuple
// T_theta and c positions in beta is the sum over the set partitions of the
// A + K_theta positions whose parts hold at most two mu positions, of
// H(n, c) prod_(parts) d_part delta, n the number of parts, with
//
//   H(n, c) = d^n/d delta^n d^c/d beta^c (-1/2 delta^beta)
//           = -1/2 delta^(beta - n) sum_(j=0..c) binom(c, j) P_n^(j)(beta) L^(c-j),
//
// L = log delta and P_n(beta) = beta (beta - 1) ... (beta - n + 1) the falling
// factorial, whose beta-derivatives come from the recursion
// P_(n+1)^(j) = (beta - n) P_n^(j) + j P_n^(j-1), which has no cancellation since
// P_n itself is a product. The sum over j can cancel, at moderate |L|, where
// terms of different powers of L have opposite signs: at delta = 1e-3,
// beta = 0.5, n = 2, c = 4 the term in L^4 (-569) and the term in L^2 (572)
// leave about 3, and that entry of the table is 3.7e-14 relative from the
// reference (measured against mpmath over beta in [0.3, 6], delta in
// [1e-12, 1e8] and n + c <= 6; it is the worst entry). It does not reach the
// blocks: a block holding it is within 6.7 ulp of max(|ref|, 1) (measured over
// A <= 2, up to six derivatives in beta, delta from 1e-4 to 9). A = 0 with c = 0 adds
// s d_T log|M|; the pure beta blocks are -log Z^(c) + H(0, c).
//
// delta^(beta - n) is evaluated as exp((beta - n) log delta), which neither
// overflows nor underflows spuriously where delta^beta / delta^n would, and
// delta^beta itself by pow, which is exact at beta = 1, so that the kernel
// there is the Gaussian's -delta/2 exactly; log Z(d, 1) goes through lgamma,
// and the whole law is the Gaussian within 0.884 ulp1N of each block's scale
// (measured; exactly on several structures).
//
// From three Faa di Bruno positions on (n = A + K_theta >= 3), the residual
// r = y - mu, the vectors W_U r, the scalars r' W_U r, the kernel table and the
// partition sum are all carried in double-double; orders one and two stay in
// double. Two separate losses make this worth its cost, both measured
// (docs/numerics.md): the partition sum cancels, 951-fold at the worst entry of
// an order-six mu block of the scale form at delta = 9.42 and beta = 0.7, which
// in double left that block 153 ulp of its own scale from the reference against
// 9.03 with double-double; and exp((beta - n) log delta) in double carries the
// relative error eps |(beta - n) log delta| inherent in exp(log), which forming
// the product exactly in double-double removes.
//
// At delta = 0 (the radial kink at y = mu) the
// kernel is identically zero in beta, so the value and every pure beta
// derivative vanish; a delta derivative of order n is the limit 0 when
// beta > n, -1/2 n! when beta = n and c = 0, and infinite otherwise, with the
// sign of the term of highest power of L.
//
// The derivatives of log Z in beta. With u = 1/beta and
// F(u) = (d/2) u log 2 + lgamma(1 + (d/2) u),
//   log Z = log 2 + (d/2) log pi - log d - lgamma(d/2) + F(1/beta),
//   d^n/dbeta^n F(1/beta) = (-1)^n sum_(k=1..n) F^(k)(1/beta) (n!/k!) binom(n-1, k-1) beta^(-n-k),
//   F^(1)(u) = (d/2) (log 2 + psi(1 + (d/2) u)),  F^(k)(u) = (d/2)^k psi^(k-1)(1 + (d/2) u),
// through laplace::math::polygamma. Every term of the inner sum carries one
// sign for k >= 2, since psi^(k-1) does.
//
// The regularity. delta^beta = ||r||_W^(2 beta) is homogeneous of degree
// 2 beta in r, so a k-th partial is a homogeneous function of degree 2 beta - k
// times a function of the direction: continuous at r = 0 while 2 beta - k > 0
// and direction-dependent at k = 2 beta. The order up to which l stays
// continuous in y (and in mu) at y = mu is therefore infinite when beta is a
// positive integer, where delta^beta is a polynomial in r, and
// ceil(2 beta) - 1 otherwise. The multivariate interface carries no place for a
// runtime order, so the mu descriptor declares 0, the worst case over beta > 0.
//
// The draw: with t ~ Gamma(d/(2 beta), 1) and z a standard normal vector,
// y = mu + L (s z / ||z||) with s = (2 t)^(1/(2 beta)) and L z the Gaussian's
// (L of Sigma, or L^-T of Q): the radial law of ||u|| under
// exp(-||u||^(2 beta)/2) has t = ||u||^(2 beta)/2 gamma with shape d/(2 beta).
//
// All of the closed forms above are checked against SymPy's own derivatives in
// tools/reference/mvpowerexp_forms.py.
#pragma once

#include <laplace/distrib/detail/mvgaussian_common.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/mvpowerexp_state.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/polygamma.hpp>
#include <laplace/math/special.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace laplace::distrib {

namespace detail_mvpowerexp {

namespace mv = laplace::distrib::detail_mv;
namespace mg = laplace::distrib::detail_mvgaussian;
namespace ddm = laplace::math::detail::ddm;
using pdmat::packed_index;
using pdmat::packed_size;

inline constexpr double log_two = 0.69314718055994530941723212145818;
inline constexpr double log_pi = 1.14472988584940017414342735135306;

// ------------------------------------------------------------ log Z in beta
// c[k] = d^k/dbeta^k log Z(d, beta), k = 0..K (K <= 6)
LAPLACE_HD void log_z_dk(int d, double beta, int K, double* c) {
  const double a = 0.5 * d, u = 1.0 / beta, arg = 1.0 + a * u;
  double F[7];
  F[0] = a * u * log_two + math::lgamma(arg);
  double ak = 1.0;
  for (int k = 1; k <= K; ++k) {
    ak *= a;
    F[k] = ak * math::polygamma(k - 1, arg);
    if (k == 1) F[k] += a * log_two;
  }
  c[0] = log_two + a * log_pi - math::log(double(d)) - math::lgamma(a) + F[0];
  double fact[7] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0};
  for (int n = 1; n <= K; ++n) {
    double s = 0.0, pw = 1.0;  // beta^-(n + k), from k = 1
    for (int i = 1; i <= n; ++i) pw *= u;
    for (int k = 1; k <= n; ++k) {
      pw *= u;
      s += F[k] * (fact[n] / fact[k]) * binomial_coef(n - 1, k - 1) * pw;
    }
    c[n] = (n % 2 == 0 ? 1.0 : -1.0) * s;
  }
}

// ------------------------------------------------------------ the kernel table
// P[j] = P_n^(j)(beta), j = 0..c, from P_(n+1)^(j) = (beta - n) P_n^(j) + j P_n^(j-1)
LAPLACE_HD void falling_derivs(int n, int c, double beta, double* P) {
  for (int j = 1; j <= c; ++j) P[j] = 0.0;
  P[0] = 1.0;
  for (int m = 0; m < n; ++m)
    for (int j = c; j >= 0; --j) P[j] = (beta - m) * P[j] + (j > 0 ? j * P[j - 1] : 0.0);
}

// H[n] = d^n/d delta^n d^c/d beta^c (-1/2 delta^beta), n = 0..N (N + c <= 6)
LAPLACE_HD void kernel_table(double delta, double beta, int N, int c, double* H) {
  double P[7];
  if (delta == 0.0) {  // the radial kink at y = mu
    for (int n = 0; n <= N; ++n) {
      falling_derivs(n, c, beta, P);
      const double e = beta - n;
      if (e > 0.0) {
        H[n] = 0.0;
      } else if (e == 0.0 && c == 0) {
        H[n] = -0.5 * P[0];
      } else {
        // the smallest j with P_n^(j) != 0 carries the highest power of log delta
        H[n] = 0.0;
        for (int j = 0; j <= c; ++j)
          if (P[j] != 0.0) {
            const double t = -0.5 * P[j] * ((c - j) % 2 == 0 ? 1.0 : -1.0);
            H[n] = t > 0.0 ? std::numeric_limits<double>::infinity()
                           : -std::numeric_limits<double>::infinity();
            break;
          }
      }
    }
    return;
  }
  const double Lg = math::log(delta);
  for (int n = 0; n <= N; ++n) {
    falling_derivs(n, c, beta, P);
    double acc = 0.0, lp = 1.0;  // L^(c-j), from j = c
    for (int j = c; j >= 0; --j) {
      acc += binomial_coef(c, j) * P[j] * lp;
      lp *= Lg;
    }
    if (acc == 0.0) {
      H[n] = 0.0;
      continue;
    }
    const double e = beta - n;
    const double pw = n == 0 ? Kokkos::pow(delta, beta) : (e == 0.0 ? 1.0 : math::exp(e * Lg));
    H[n] = -0.5 * pw * acc;
  }
}

// H[n] = d^n/d delta^n d^c/d beta^c (-1/2 delta^beta) in double-double, from
// delta in double-double, for n = 0..N (N + c <= 6). Two things are recovered
// here that the double route loses: the product (beta - n) log delta is formed
// exactly, where exp of it in double carries the relative error
// eps |(beta - n) log delta| inherent in exp(log), and the polynomial in L,
// which can cancel at moderate |L| (above), is summed in double-double.
LAPLACE_HD void kernel_table_dd(const ddm::dd& delta, double beta, int N, int c, ddm::dd* H) {
  double P[7];
  const ddm::dd Lg = ddm::log(delta);
  for (int n = 0; n <= N; ++n) {
    falling_derivs(n, c, beta, P);
    ddm::dd acc(0.0), lp(1.0);  // L^(c-j), from j = c
    for (int j = c; j >= 0; --j) {
      acc = acc + lp * (binomial_coef(c, j) * P[j]);
      lp = lp * Lg;
    }
    if (acc.hi == 0.0) {
      H[n] = ddm::dd(0.0);
      continue;
    }
    const double e = beta - n;
    const ddm::dd pw = e == 0.0 ? ddm::dd(1.0) : ddm::exp(Lg * e);
    H[n] = pw * acc * (-0.5);
  }
}

// ------------------------------------------------------------ W_U at run time
// These two dispatchers duplicate detail_mvstudent_t's; both are about the
// matrix structure alone and belong in detail/mvgaussian_common.hpp, which is a
// shared header and so is left to a change of its own.
//
// d_U W along the k sorted indices sub (k = 0: W itself): a pointer to it (the
// state, the table, or buf, where it is computed); false when zero by structure
template <bool Prec, int K, class St, class S>
LAPLACE_HD bool w_dk_fixed(const S& s, const St& st, const int* sub, double* buf, const double*& Wp, double* ws) {
  if constexpr (K == 0) {
    Wp = st.quadratic;
    return true;
  } else {
    if (const double* e = mg::table_entry<K>(s, st, sub)) {
      Wp = e + 2;
      return e[0] != 0.0;
    }
    Wp = buf;
    return mv::structure_matrix_dk<!Prec, K>(s, st.theta, sub, buf, ws);
  }
}
template <bool Prec, class St, class S>
LAPLACE_HD bool w_dk_runtime(const S& s, const St& st, int k, const int* sub, double* buf, const double*& Wp,
                             double* ws) {
  switch (k) {
    case 0: return w_dk_fixed<Prec, 0>(s, st, sub, buf, Wp, ws);
    case 1: return w_dk_fixed<Prec, 1>(s, st, sub, buf, Wp, ws);
    case 2: return w_dk_fixed<Prec, 2>(s, st, sub, buf, Wp, ws);
    case 3: return w_dk_fixed<Prec, 3>(s, st, sub, buf, Wp, ws);
    case 4: return w_dk_fixed<Prec, 4>(s, st, sub, buf, Wp, ws);
    case 5: return w_dk_fixed<Prec, 5>(s, st, sub, buf, Wp, ws);
    default: return w_dk_fixed<Prec, 6>(s, st, sub, buf, Wp, ws);
  }
}
// d_U log|M| at run time, from the table when it covers the order
template <int K, class St, class S>
LAPLACE_HD double log_det_dk_fixed(const S& s, const St& st, const int* sub, double* ws) {
  if constexpr (K > 0) {
    if (const double* e = mg::table_entry<K>(s, st, sub)) return e[1];
  }
  return mv::structure_log_det_dk<K>(s, st.theta, sub, ws);
}
template <class St, class S>
LAPLACE_HD double log_det_dk_runtime(const S& s, const St& st, int k, const int* sub, double* ws) {
  switch (k) {
    case 0: return log_det_dk_fixed<0>(s, st, sub, ws);
    case 1: return log_det_dk_fixed<1>(s, st, sub, ws);
    case 2: return log_det_dk_fixed<2>(s, st, sub, ws);
    case 3: return log_det_dk_fixed<3>(s, st, sub, ws);
    case 4: return log_det_dk_fixed<4>(s, st, sub, ws);
    case 5: return log_det_dk_fixed<5>(s, st, sub, ws);
    default: return log_det_dk_fixed<6>(s, st, sub, ws);
  }
}

// ------------------------------------------------------------ the blocks
// doubles of scratch of the unit functions: the matrices W_U kept for the
// blocks with two mu positions (at most 16, A = 2 and K_theta <= 4), one
// computed matrix, the vectors W_U r and the scalars r' W_U r in double-double
// (64 subsets), then the structure's scratch
template <class S>
LAPLACE_HD int unit_scratch(const S& s) {
  const int d = s.dim, np = packed_size(d);
  return 17 * np + 128 * (d + 1) + s.scratch_size();
}

template <bool Prec, class St, class S>
LAPLACE_HD double logpdf_value(const S& s, const mvpowerexp_state<St>& st, const double* y, const double* mu) {
  const double delta = mv::packed_quadratic(s.dim, st.gauss.quadratic, y, mu);
  const double k = delta == 0.0 ? 0.0 : 0.5 * Kokkos::pow(delta, st.beta);
  return -st.lz[0] + mg::log_det_coef<Prec> * st.gauss.log_det - k;
}

// A derivatives in mu, the sorted tuple idx of K indices of the rest (beta = q)
template <bool Prec, int K, int A, class St, class S>
LAPLACE_HD bool logpdf_dk(const S& s, const mvpowerexp_state<St>& st, const double* y, const double* mu,
                          const int* idx, double* out, double* w) {
  static_assert(A >= 0 && K >= 0 && A + K <= 6, "logpdf_dk: A + K <= 6");
  const int d = s.dim, np = packed_size(d), q = s.n_params();
  const St& gs = st.gauss;
  int c = 0;
  for (int k = 0; k < K; ++k) c += idx[k] == q;
  const int Kt = K - c, n = A + Kt, NU = 1 << Kt;
  if constexpr (A == 0) {
    if (Kt == 0) {
      if (c == 0) {
        out[0] = logpdf_value<Prec>(s, st, y, mu);
      } else {
        double H[7];
        kernel_table(mv::packed_quadratic(d, gs.quadratic, y, mu), st.beta, 0, c, H);
        out[0] = -st.lz[c] + H[0];
      }
      return true;
    }
  }
  double* Wkeep = w;          // 16 np
  double* buf = w + 16 * np;  // np
  double* V = buf + np;       // 64 d, W_U r (high parts)
  double* Vl = V + 64 * d;    // 64 d, the low parts
  double* sv = Vl + 64 * d;   // 64, r' W_U r (high parts)
  double* svl = sv + 64;      // 64, the low parts
  double* ws = svl + 64;
  const double* Wp[64];
  bool nz[64];
  // from three Faa di Bruno positions on the partition sum cancels (measured:
  // 951-fold at the worst entry of an order-six mu block, so about 150 ulp of
  // the block's scale in double), so there the residual, W_U r, r' W_U r, the
  // kernel table and the partition sum all go in double-double; orders one and
  // two stay in double. The threshold is mvstudent_t's.
  const bool hp = n >= 3;
  for (int U = 0; U < NU; ++U) {
    int sub[6], k = 0;
    for (int p = 0; p < Kt; ++p)
      if (U & (1 << p)) sub[k++] = idx[p];
    const double* P = nullptr;
    nz[U] = w_dk_runtime<Prec>(s, gs, k, sub, A >= 2 ? Wkeep + U * np : buf, P, ws);
    Wp[U] = P;
    if (!nz[U]) continue;
    if (hp) {
      // W_U r and r' W_U r in double-double from r = y - mu exact (two_sum)
      ddm::dd a;
      for (int i = 0; i < d; ++i) {
        ddm::dd t;
        for (int j = 0; j < d; ++j)
          t = t + ddm::two_sum(y[j], -mu[j]) * P[i >= j ? packed_index(i, j) : packed_index(j, i)];
        V[U * d + i] = t.hi, Vl[U * d + i] = t.lo;
        a = a + ddm::two_sum(y[i], -mu[i]) * t;
      }
      sv[U] = a.hi, svl[U] = a.lo;
    } else {
      mv::packed_times_residual(d, P, y, mu, 1.0, V + U * d);
      double a = 0.0;
      for (int i = 0; i < d; ++i) a += (y[i] - mu[i]) * V[U * d + i];
      sv[U] = a;
    }
  }
  // the set partitions of the n positions (mu 0..A-1, theta A..n-1) as restricted
  // growth strings; per part its theta mask and its mu positions
  signed char pU[203][6], pm0[203][6], pm1[203][6], pn[203];
  int npart = 0;
  {
    int a[6] = {0, 0, 0, 0, 0, 0}, mx[6] = {0, 0, 0, 0, 0, 0};
    for (;;) {
      int nb = n == 0 ? 0 : mx[n - 1] + 1;
      int U[6] = {0, 0, 0, 0, 0, 0}, cnt[6] = {0, 0, 0, 0, 0, 0}, m0[6], m1[6];
      bool ok = true;
      for (int p = 0; p < n && ok; ++p) {
        const int bl = a[p];
        if (p < A) {
          if (cnt[bl] == 0) m0[bl] = p;
          else if (cnt[bl] == 1) m1[bl] = p;
          else ok = false;
          ++cnt[bl];
        } else {
          U[bl] |= 1 << (p - A);
        }
      }
      for (int bl = 0; bl < nb && ok; ++bl) ok = nz[U[bl]];
      if (ok) {
        pn[npart] = static_cast<signed char>(nb);
        for (int bl = 0; bl < nb; ++bl) {
          pU[npart][bl] = static_cast<signed char>(U[bl]);
          pm0[npart][bl] = static_cast<signed char>(cnt[bl] >= 1 ? m0[bl] : -1);
          pm1[npart][bl] = static_cast<signed char>(cnt[bl] >= 2 ? m1[bl] : -1);
        }
        ++npart;
      }
      // next restricted growth string
      int p = n - 1;
      while (p >= 1 && a[p] == mx[p - 1] + 1) --p;
      if (p < 1) break;
      ++a[p];
      mx[p] = a[p] > mx[p - 1] ? a[p] : mx[p - 1];
      for (int r = p + 1; r < n; ++r) a[r] = 0, mx[r] = mx[p];
    }
  }
  double ldT = 0.0;
  if (A == 0 && c == 0) ldT = log_det_dk_runtime(s, gs, Kt, idx, ws);
  // The radial kink at r = 0. There every factor that carries r vanishes
  // exactly (r' W_U r and (W_U r)_i), and every partition term is homogeneous
  // of degree 2 beta - A in ||r||: a part with no mu position contributes
  // degree 2, with one mu position degree 1, with two degree 0, and H(n, c)
  // degree 2(beta - n), which sums to 2 beta - A whatever the partition and
  // whatever K_theta. So for 2 beta > A the kernel contributes nothing and only
  // d_T log|M| is left, while the factor-by-factor product below would read
  // 0 times infinity and give NaN. For 2 beta <= A the derivative does not
  // exist at r = 0 and the product is left to return its infinities.
  if (sv[0] == 0.0 && 2.0 * st.beta > A) {
    out[0] = mg::log_det_coef<Prec> * ldT;
    if constexpr (A > 0) {
      const int nn = binomial_coef(d + A - 1, A);
      for (int k = 0; k < nn; ++k) out[k] = 0.0;
    }
    return true;
  }
  if (npart == 0 && ldT == 0.0) return false;
  // the value of one partition at the mu indices ii, with T = double or
  // double-double (the weights H of type T)
  auto run = [&](const auto* H) {
    using T = typename std::remove_cv<typename std::remove_pointer<decltype(H)>::type>::type;
    constexpr bool D2 = !std::is_same<T, double>::value;
    auto part_value = [&](int pi, const int* ii) {
      T v = H[pn[pi]];
      for (int bl = 0; bl < pn[pi]; ++bl) {
        const int U = pU[pi][bl];
        if (pm0[pi][bl] < 0) {
          if constexpr (D2) v = v * ddm::dd(sv[U], svl[U]);
          else v *= sv[U];
        } else if (pm1[pi][bl] < 0) {
          const int k = U * d + ii[pm0[pi][bl]];
          if constexpr (D2) v = v * ddm::dd(-2.0 * V[k], -2.0 * Vl[k]);
          else v *= -2.0 * V[k];
        } else {
          const int i = ii[pm0[pi][bl]], j = ii[pm1[pi][bl]];
          v = v * (2.0 * Wp[U][i >= j ? packed_index(i, j) : packed_index(j, i)]);
        }
      }
      return v;
    };
    auto rounded = [&](const T& v) {
      if constexpr (D2) return v.hi + v.lo;
      else return v;
    };
    int ii[6] = {0, 0, 0, 0, 0, 0};
    if constexpr (A == 0) {
      T v(mg::log_det_coef<Prec> * ldT);
      for (int pi = 0; pi < npart; ++pi) v = v + part_value(pi, ii);
      out[0] = rounded(v);
    } else if constexpr (A == 1) {
      for (int i = 0; i < d; ++i) {
        ii[0] = i;
        T v(0.0);
        for (int pi = 0; pi < npart; ++pi) v = v + part_value(pi, ii);
        out[i] = rounded(v);
      }
    } else if constexpr (A == 2) {
      for (int i = 0; i < d; ++i)
        for (int j = 0; j <= i; ++j) {
          ii[0] = i, ii[1] = j;
          T v(0.0);
          for (int pi = 0; pi < npart; ++pi) v = v + part_value(pi, ii);
          out[packed_index(i, j)] = rounded(v);
        }
    } else {  // sorted tuples in lexicographic order
      for (int e = 0;; ++e) {
        T v(0.0);
        for (int pi = 0; pi < npart; ++pi) v = v + part_value(pi, ii);
        out[e] = rounded(v);
        int p = A - 1;
        while (p >= 0 && ii[p] == d - 1) --p;
        if (p < 0) break;
        ++ii[p];
        for (int r = p + 1; r < A; ++r) ii[r] = ii[p];
      }
    }
  };
  if (hp) {
    ddm::dd H[7];
    kernel_table_dd(ddm::dd(sv[0], svl[0]), st.beta, n, c, H);
    run(H);
  } else {
    double H[7];
    kernel_table(sv[0], st.beta, n, c, H);
    run(H);
  }
  return true;
}

template <bool Prec, int J, int K, class St, class S>
LAPLACE_HD bool logpdf_dy_dk(const S& s, const mvpowerexp_state<St>& st, const double* y, const double* mu,
                             const int* idx, double* out, double* w) {
  if (!logpdf_dk<Prec, K, J>(s, st, y, mu, idx, out, w)) return false;
  if constexpr (J % 2 == 1) {
    const int nn = J == 1 ? s.dim : binomial_coef(s.dim + J - 1, J);
    for (int k = 0; k < nn; ++k) out[k] = -out[k];
  }
  return true;
}

// ------------------------------------------------------------ covariance, draw
// Var Y = Sigma 2^(1/beta) Gamma((d+2)/(2 beta)) / (d Gamma(d/(2 beta))),
// through the log so that the two gammas do not overflow apart
LAPLACE_HD double covariance_factor(int d, double beta) {
  const double a = 0.5 / beta;
  return math::exp(log_two / beta + math::lgamma((d + 2.0) * a) - math::log(double(d)) -
                   math::lgamma(d * a));
}

// x = L v or L^-T v, with v the radial draw times a uniform direction, so that
// the caller's y = mu + x has delta = v' v
template <bool Prec, class St, class S>
LAPLACE_HD void rng_whitened(const S& s, const mvpowerexp_state<St>& st, std::uint64_t seed, std::uint64_t unit,
                             double* x) {
  const int d = s.dim;
  const double* L = st.gauss.factor;
  counter_uniform_source src{seed, unit};
  double nrm = 0.0;
  for (int k = 0; k < d; ++k) {
    x[k] = standard_normal_by_inversion(src);
    nrm += x[k] * x[k];
  }
  nrm = Kokkos::sqrt(nrm);
  const double t = standard_gamma_by_inversion(0.5 * d / st.beta, src);
  const double rad = Kokkos::pow(2.0 * t, 0.5 / st.beta);
  for (int k = 0; k < d; ++k) x[k] *= rad / nrm;
  for (int i = d - 1; i >= 0; --i) {
    if constexpr (Prec) {
      double v = x[i];
      for (int k = i + 1; k < d; ++k) v -= L[packed_index(k, i)] * x[k];
      x[i] = v / L[packed_index(i, i)];
    } else {
      double v = 0.0;
      for (int k = 0; k <= i; ++k) v += L[packed_index(i, k)] * x[k];
      x[i] = v;
    }
  }
}

}  // namespace detail_mvpowerexp
}  // namespace laplace::distrib
