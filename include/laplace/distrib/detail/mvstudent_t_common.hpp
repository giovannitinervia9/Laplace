// The code shared by mvstudent_t1_distrib (scale matrix Sigma = M(theta)) and
// mvstudent_t2_distrib (inverse scale Q = M(theta)); note/distrib_multivariate.md
// section 15, decided with Giovanni on 2026-11-16. Prec = false: W = M^-1 (or
// M^+ in the degenerate class), s = -1/2; Prec = true: W = M, s = +1/2. rho is
// the dimension d (proper) or the rank (degenerate), r = y - mu, delta = r' W r:
//
//   l = c_rho(nu) + s log|M| + g(delta, nu),
//   c_rho(nu) = lgamma((nu + rho)/2) - lgamma(nu/2) - rho/2 log(nu pi),
//   g(delta, nu) = -(nu + rho)/2 log1p(delta/nu).
//
// "The rest" is (theta_0..theta_(q-1), nu): nu is the index q, the last.
//
// The blocks. l depends on (mu, theta) through delta (and log|M|), and
//   d_U delta = r' W_U r,  d_mu_i d_U delta = -2 (W_U r)_i,  d_mu_i d_mu_j d_U delta = 2 (W_U)_ij,
// zero with three or more mu, W_U = d_U W (U a subset of the theta positions).
// So by Faa di Bruno's formula a block with A positions in mu, the theta tuple
// T_theta and c positions in nu is the sum over the set partitions of the
// A + K_theta positions whose parts hold at most two mu positions, of
// G(n, c) prod_(parts) d_part delta, n the number of parts, with
//   G(n, c) = d^n/d delta^n d^c/d nu^c g,
//   G(n, 0) = -1/2 (-1)^(n-1) (n-1)! (nu + rho) (nu + delta)^-n,
//   G(n, c) = -1/2 (-1)^(n-1) (n-1)! f_(c-1) (nu + delta)^-(n+c)
//             [(1 - n) nu - (n + c - 1) rho + c delta],  f_k = (-1)^k n (n+1)...(n+k-1),
// in which nothing cancels as nu grows (n = 1: c (delta - rho)); A = 0 with
// c = 0 adds s d_T log|M|; the pure nu blocks are c_rho^(c) + G(0, c), G(0, c)
// student_t1's pure nu derivative of -(nu + 1)/2 log1p(delta/nu) plus
// -(rho - 1)/2 (-1)^(c-1) (c-1)! nu^-c expm1(-c log1p(delta/nu)).
// c_rho by integer shifts of student_t1's f(x) = lgamma(x + 1/2) - lgamma(x) - 1/2 log x,
// b = nu/2, m = floor(rho/2):
//   c_rho = sum_(j=1..m-1) log1p(j/b) + [rho odd] (f(b + m) + 1/2 log1p(m/b)) - rho/2 log 2 pi,
//   d^k log1p(j/b)/db^k = (-1)^(k-1) (k-1)! b^-k expm1(-k log1p(j/b))  (one sign).
// All of these are checked against SymPy in tools/reference/mvstudent_t_forms.py.
//
// The expected information (Lange, Little and Taylor 1989; the script above
// checks it against a quadrature over the law, the degenerate class included),
// kappa = (nu + rho)/(nu + rho + 2), t_a = d_a log|Sigma| = -+ d_a log|M|:
//   I_mumu = kappa W,  I_mu,rest = 0,
//   I_ab = kappa I^G_ab - t_a t_b / (2 (nu + rho + 2))                  (I^G the Gaussian's)
//        = kappa Ttilde_ab / 2 + nu t_a t_b / (2 rho (nu + rho + 2)),  (closed_factor, proper)
//   I_a,nu = -t_a / ((nu + rho)(nu + rho + 2)),
//   I_nunu = 1/4 [psi'(b) - psi'(b + rho/2)] - rho (nu + rho + 4)/(2 nu (nu + rho)(nu + rho + 2)).
// The second form of I_ab (decided 2026-11-16) avoids the cancellation of the
// first at small nu along a common-scale direction: Ttilde_ab = tr(Btilde_a
// Btilde_b), Btilde = B - tr(B)/rho I, B_a = G_a + G_a' the whitened derivative of
// the factor form (detail/mvgaussian_common.hpp), with its diagonal centred before
// the trace. I_nunu by a recurrence in rho whose terms carry one sign (h = rho/2):
//   I(nu, rho) - I(nu, rho - 2) = 4 (h + 1) / ((nu + 2h)(nu + 2h - 2)^2 (nu + 2h + 2))
//                                + 8 (h - 1)^2 / (nu (nu + 2h)(nu + 2h - 2)^2 (nu + 2h + 2)),
//   I(nu, 2) = 8 / (nu^2 (nu + 2)(nu + 4)),  I(nu, 1) student_t1's;
// the nu derivatives of the inverse products are sums of positive products
// (student_t1's inv_prod_deriv), so nothing cancels at any nu or rho.
//
// The draw: y = mu + L z / sqrt(w), w = 2 G / nu with G = P^-1(nu/2, u),
// z_k the k-th standard normal of counter_uniform_source{seed, unit}, k < d,
// and u its d-th uniform; L z is the Gaussian's (L of Sigma, or L^-T of Q).
#pragma once

#include <laplace/distrib/detail/mvgaussian_common.hpp>
#include <laplace/distrib/mvstudent_t_state.hpp>
#include <laplace/distrib/detail/mvgaussian_rank_deficient.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/student.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace laplace::distrib {

namespace detail_mvstudent_t {

namespace mv = laplace::distrib::detail_mv;
namespace mg = laplace::distrib::detail_mvgaussian;
namespace st1 = laplace::distrib::detail_student_t1;
namespace ddm = laplace::math::detail::ddm;
using pdmat::packed_index;
using pdmat::packed_size;

LAPLACE_HD_FORCEINLINE double sgn_pow(int n) { return n % 2 == 0 ? 1.0 : -1.0; }

// ------------------------------------------------------------ nu only
// c[k] = c_rho^(k)(nu), k = 0..K (K <= 6)
LAPLACE_HD void nu_constant_dk(double nu, int rho, int K, double* c) {
  const double b = 0.5 * nu;
  const int m = rho / 2;
  for (int k = 0; k <= K; ++k) c[k] = 0.0;
  // weight * log1p(j/b) and its b-derivatives
  auto add_log1p = [&](double j, double weight) {
    const double lj = Kokkos::log1p(j / b);
    c[0] += weight * lj;
    double bk = 1.0, fct = 1.0;
    for (int k = 1; k <= K; ++k) {
      bk /= b;
      if (k > 1) fct *= (k - 1);
      c[k] += weight * sgn_pow(k - 1) * fct * bk * Kokkos::expm1(-k * lj);
    }
  };
  for (int j = 1; j < m; ++j) add_log1p(j, 1.0);
  if (rho % 2 == 1) {
    double f[7];
    math::own::detail_st::fder(b + m, K, f);
    for (int k = 0; k <= K; ++k) c[k] += f[k];
    if (m > 0) add_log1p(m, 0.5);
  }
  c[0] -= 0.5 * rho * mv::log_two_pi;
  double s = 1.0;  // d/dnu = 1/2 d/db
  for (int k = 1; k <= K; ++k) {
    s *= 0.5;
    c[k] *= s;
  }
}

// f[q] = d^q/dnu^q I_nunu(nu, rho), q = 0..Q (Q <= 4)
LAPLACE_HD void info_nunu_dk(double nu, int rho, int Q, double* f) {
  double base[5];
  int h2;  // 2h of the first term of the recurrence
  if (rho % 2 == 1) {
    st1::info_nunu_derivs<4>(nu, base);
    h2 = 3;
  } else {
    const double a4[4] = {0.0, 0.0, 2.0, 4.0};
    for (int q = 0; q <= Q; ++q) base[q] = 8.0 * st1::inv_prod_deriv<4>(q, a4, nu);
    h2 = 4;
  }
  for (int q = 0; q <= Q; ++q) f[q] = base[q];
  for (; h2 <= rho; h2 += 2) {
    const double h = 0.5 * h2, x = h2;
    const double a4[4] = {x, x - 2.0, x - 2.0, x + 2.0};
    const double a5[5] = {0.0, x, x - 2.0, x - 2.0, x + 2.0};
    for (int q = 0; q <= Q; ++q)
      f[q] += 4.0 * (h + 1.0) * st1::inv_prod_deriv<4>(q, a4, nu) +
              8.0 * (h - 1.0) * (h - 1.0) * st1::inv_prod_deriv<5>(q, a5, nu);
  }
}

// d^c/dnu^c of A + B/(nu + x) for c >= 1: B (-1)^c c! (nu + x)^-(c+1)
LAPLACE_HD double inv_linear_dk(int c, double B, double nu, double x) {
  double v = B / (nu + x);
  for (int i = 1; i <= c; ++i) v *= -i / (nu + x);
  return v;
}

// kappa^(c), kappa = (nu + rho)/(nu + rho + 2) = 1 - 2/(nu + rho + 2)
LAPLACE_HD double kappa_dk(int c, double nu, int rho) {
  return c == 0 ? (nu + rho) / (nu + rho + 2.0) : inv_linear_dk(c, -2.0, nu, rho + 2.0);
}

// ------------------------------------------------------------ G(n, c)
// Gt[n * 7 + c] = d^n/d delta^n d^c/d nu^c g for 1 <= n, n + c <= N, and for
// n = 0 the value (c = 0) and the order c0 (c0 >= 1) when c0 > 0
LAPLACE_HD void g_table(double delta, double nu, int rho, int N, int c0, double* Gt) {
  const double nd = nu + delta, ind = 1.0 / nd;
  double ndn = 1.0, fn1 = 1.0;  // (nu + delta)^-n, (n - 1)!
  for (int n = 1; n <= N; ++n) {
    ndn *= ind;
    if (n > 1) fn1 *= (n - 1);
    const double pre = -0.5 * sgn_pow(n - 1) * fn1;
    Gt[n * 7] = pre * (nu + rho) * ndn;
    double fk = 1.0, pw = ndn * ind;  // f_(c-1), (nu + delta)^-(n+c)
    for (int c = 1; n + c <= N; ++c) {
      if (c > 1) fk *= -(n + c - 2);
      Gt[n * 7 + c] = pre * fk * pw * ((1.0 - n) * nu - (n + c - 1.0) * rho + c * delta);
      pw *= ind;
    }
  }
  Gt[0] = -0.5 * (nu + rho) * Kokkos::log1p(delta / nu);
  if (c0 > 0) {
    double D[49];
    const double z = Kokkos::sqrt(delta);
    switch (c0) {
      case 1: st1::lambda_table<1>(z, nu, D); break;
      case 2: st1::lambda_table<2>(z, nu, D); break;
      case 3: st1::lambda_table<3>(z, nu, D); break;
      case 4: st1::lambda_table<4>(z, nu, D); break;
      case 5: st1::lambda_table<5>(z, nu, D); break;
      default: st1::lambda_table<6>(z, nu, D); break;
    }
    double fct = 1.0, nc = 1.0;
    for (int i = 1; i <= c0; ++i) {
      if (i > 1) fct *= (i - 1);
      nc /= nu;
    }
    const double dl = sgn_pow(c0 - 1) * fct * nc * Kokkos::expm1(-c0 * Kokkos::log1p(delta / nu));
    Gt[c0] = D[c0 * (c0 + 1)] - 0.5 * (rho - 1.0) * dl;
  }
}

// G(n, c) for 1 <= n, n + c <= N, as g_table, in double-double from delta in
// double-double and nu: Gd[n * 7 + c]
LAPLACE_HD void g_table_dd(const ddm::dd& delta, double nu, int rho, int N, ddm::dd* Gd) {
  const ddm::dd nd = delta + nu, ind = ddm::dd(1.0) / nd, nr = ddm::two_sum(nu, double(rho));
  ddm::dd ndn(1.0);  // (nu + delta)^-n
  double fn1 = 1.0;  // (n - 1)!
  for (int n = 1; n <= N; ++n) {
    ndn = ndn * ind;
    if (n > 1) fn1 *= (n - 1);
    const double pre = -0.5 * sgn_pow(n - 1) * fn1;  // exact
    Gd[n * 7] = nr * ndn * pre;
    double fk = 1.0;  // f_(c-1), exact
    ddm::dd pw = ndn * ind;
    for (int c = 1; n + c <= N; ++c) {
      if (c > 1) fk *= -(n + c - 2);
      // (1 - n) nu - (n + c - 1) rho + c delta
      const ddm::dd br = ddm::two_prod(1.0 - n, nu) + delta * double(c) - (n + c - 1.0) * rho;
      Gd[n * 7 + c] = br * pw * (pre * fk);
      pw = pw * ind;
    }
  }
}

// ------------------------------------------------------------ W_U at run time
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
// (64 subsets), then
// the structure's scratch
template <class S>
LAPLACE_HD int unit_scratch(const S& s) {
  const int d = s.dim, np = packed_size(d);
  return 17 * np + 128 * (d + 1) + s.scratch_size();
}

template <bool Prec, class St, class S>
LAPLACE_HD double logpdf_value(const S& s, const mvstudent_t_state<St>& st, const double* y, const double* mu) {
  const double delta = mv::packed_quadratic(s.dim, st.gauss.quadratic, y, mu);
  return st.cnu[0] + mg::log_det_coef<Prec> * st.gauss.log_det -
         0.5 * (st.nu + st.rho) * Kokkos::log1p(delta / st.nu);
}

// A derivatives in mu, the sorted tuple idx of K indices of the rest (nu = q)
template <bool Prec, int A, int K, class St, class S>
LAPLACE_HD bool logpdf_dmu_dk(const S& s, const mvstudent_t_state<St>& st, const double* y, const double* mu,
                              const int* idx, double* out, double* w) {
  static_assert(A >= 0 && K >= 0 && A + K <= 6, "logpdf_dmu_dk: A + K <= 6");
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
        double Gt[49];
        g_table(mv::packed_quadratic(d, gs.quadratic, y, mu), st.nu, st.rho, 0, c, Gt);
        out[0] = st.cnu[c] + Gt[c];
      }
      return true;
    }
  }
  double* Wkeep = w;              // 16 np
  double* buf = w + 16 * np;      // np
  double* V = buf + np;           // 64 d, W_U r (high parts)
  double* Vl = V + 64 * d;        // 64 d, the low parts
  double* sv = Vl + 64 * d;       // 64, r' W_U r (high parts)
  double* svl = sv + 64;          // 64, the low parts
  double* ws = svl + 64;
  const double* Wp[64];
  bool nz[64];
  // double-double from three Faa di Bruno positions on: there the terms cancel
  // where delta >> nu (note at the top, decided 2026-11-17); orders 1 and 2 in double
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
      // W_U r and r' W_U r in double-double from r = y - mu exact (two_sum) and
      // W_U as computed: the far-tail sums below are ill-conditioned in them
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
  if (npart == 0 && ldT == 0.0) return false;
  // the value of one partition at the mu indices ii, with T = double or
  // double-double (the weights Gt of type T)
  auto run = [&](const auto* Gt) {
    using T = typename std::remove_cv<typename std::remove_pointer<decltype(Gt)>::type>::type;
    constexpr bool D2 = !std::is_same<T, double>::value;
    auto part_value = [&](int pi, const int* ii) {
      T v = Gt[pn[pi] * 7 + c];
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
    ddm::dd Gt[49];
    g_table_dd(ddm::dd(sv[0], svl[0]), st.nu, st.rho, n + c, Gt);
    run(Gt);
  } else {
    double Gt[49];
    g_table(sv[0], st.nu, st.rho, n + c, 0, Gt);
    run(Gt);
  }
  return true;
}

template <bool Prec, int J, int K, class St, class S>
LAPLACE_HD bool logpdf_dy_dk(const S& s, const mvstudent_t_state<St>& st, const double* y, const double* mu,
                             const int* idx, double* out, double* w) {
  if (!logpdf_dmu_dk<Prec, J, K>(s, st, y, mu, idx, out, w)) return false;
  if constexpr (J % 2 == 1) {
    const int nn = J == 1 ? s.dim : binomial_coef(s.dim + J - 1, J);
    for (int k = 0; k < nn; ++k) out[k] = -out[k];
  }
  return true;
}

// ------------------------------------------------------------ the information
// How the theta-theta part is formed: the traceless factor form (proper,
// closed_factor), the Gaussian's information of the proper class (trace route),
// or that of the degenerate class (S is then the rank_deficient_view).
enum class info_route : int { traceless, gauss_proper, gauss_degenerate };

// The theta-theta part at the theta tuple tt of length kt (pair a <= b):
// H = d_tt of Ttilde_ab/2 (traceless) or of I^G_ab, and P = d_tt (t_a t_b);
// the flags say whether each is nonzero by structure.
template <bool Prec, info_route R, int KT, class St, class S>
LAPLACE_HD void info_theta_parts(const S& s, const St& gs, const int* pair, const int* tt, double& H, bool& hz,
                                 double& P, bool& pz, double* w) {
  if constexpr (R == info_route::traceless) {
    hz = mg::expected_info_factor<KT, true>(s, gs, pair, tt, &H, w);
  } else if constexpr (R == info_route::gauss_proper) {
    hz = mg::expected_info_dk<Prec, KT>(s, gs, distrib_mv_info_block::rest_rest, pair, tt, &H, w);
  } else {
    // host only, as the degenerate class
    KOKKOS_IF_ON_HOST((hz = mg::rank_deficient_info<false, KT>(s.s, gs, distrib_mv_info_block::rest_rest, pair, tt, &H, w);))
    KOKKOS_IF_ON_DEVICE((hz = false;))
  }
  if (!hz) H = 0.0;
  P = 0.0;
  pz = false;
  for (int U = 0; U < (1 << KT); ++U) {
    int u[KT + 1], v[KT + 1], nu_ = 0, nv = 0;
    for (int p = 0; p < KT; ++p) {
      if (U & (1 << p)) u[nu_++] = tt[p];
      else v[nv++] = tt[p];
    }
    int ua[KT + 2], vb[KT + 2];
    const int na = mv::merge_sorted(pair, 1, u, nu_, ua);
    const int nb = mv::merge_sorted(pair + 1, 1, v, nv, vb);
    const double la = log_det_dk_runtime(s, gs, na, ua, w);
    if (la == 0.0) continue;
    const double lb = log_det_dk_runtime(s, gs, nb, vb, w);
    if (lb == 0.0) continue;
    P += la * lb;
    pz = true;
  }
}
template <bool Prec, info_route R, class St, class S>
LAPLACE_HD void info_theta_parts_runtime(const S& s, const St& gs, int kt, const int* pair, const int* tt, double& H,
                                         bool& hz, double& P, bool& pz, double* w) {
  switch (kt) {
    case 0: info_theta_parts<Prec, R, 0>(s, gs, pair, tt, H, hz, P, pz, w); break;
    case 1: info_theta_parts<Prec, R, 1>(s, gs, pair, tt, H, hz, P, pz, w); break;
    case 2: info_theta_parts<Prec, R, 2>(s, gs, pair, tt, H, hz, P, pz, w); break;
    case 3: info_theta_parts<Prec, R, 3>(s, gs, pair, tt, H, hz, P, pz, w); break;
    default: info_theta_parts<Prec, R, 4>(s, gs, pair, tt, H, hz, P, pz, w); break;
  }
}

// doubles of scratch of expected_info_dk
template <bool Prec, class S>
LAPLACE_HD int info_scratch(const S& s) {
  const int a = mg::info_scratch_size<Prec>(s);
  if constexpr (S::closed_factor) {
    const int f = mg::factor_info_scratch(s);
    return f > a ? f : a;
  } else {
    return a;
  }
}

template <bool Prec, info_route R, int K, class St, class S>
LAPLACE_HD bool expected_info_dk(const S& s, const mvstudent_t_state<St>& st, distrib_mv_info_block block,
                                 const int* ab, const int* idx, double* out, double* w) {
  static_assert(K >= 0 && K <= 4, "expected_info_dk: K <= 4");
  const int q = s.n_params(), np = packed_size(s.dim), rho = st.rho;
  const double nu = st.nu;
  const St& gs = st.gauss;
  int c = 0;
  for (int k = 0; k < K; ++k) c += idx[k] == q;
  const int kt = K - c;
  if (block == distrib_mv_info_block::vector_rest) return false;
  if (block == distrib_mv_info_block::vector_vector) {
    const double* P = nullptr;
    double* ws = w + np;
    if (!w_dk_runtime<Prec>(s, gs, kt, idx, w, P, ws)) return false;
    const double kc = kappa_dk(c, nu, rho);
    for (int k = 0; k < np; ++k) out[k] = kc * P[k];
    return true;
  }
  const int pair[2] = {ab[0] <= ab[1] ? ab[0] : ab[1], ab[0] <= ab[1] ? ab[1] : ab[0]};
  if (pair[0] == q) {  // nu, nu
    if (kt > 0) return false;
    double f[5];
    info_nunu_dk(nu, rho, c, f);
    out[0] = f[c];
    return true;
  }
  if (pair[1] == q) {  // theta_a, nu: -t_a / ((nu + rho)(nu + rho + 2)), t_a = -+ d_a log|M|
    int full[K + 1];
    const int nf = mv::merge_sorted(pair, 1, idx, kt, full);
    const double la = log_det_dk_runtime(s, gs, nf, full, w);
    if (la == 0.0) return false;
    const double al[2] = {double(rho), rho + 2.0};
    out[0] = (Prec ? 1.0 : -1.0) * la * st1::inv_prod_deriv<2>(c, al, nu);
    return true;
  }
  double H, P;
  bool hz, pz;
  info_theta_parts_runtime<Prec, R>(s, gs, kt, pair, idx, H, hz, P, pz, w);
  if (!hz && !pz) return false;
  double e;  // the nu coefficient of P
  if constexpr (R == info_route::traceless) {
    // nu / (2 rho (nu + rho + 2)) = 1/(2 rho) - (rho + 2)/(2 rho) / (nu + rho + 2)
    e = c == 0 ? nu / (2.0 * rho * (nu + rho + 2.0)) : inv_linear_dk(c, -(rho + 2.0) / (2.0 * rho), nu, rho + 2.0);
  } else {
    e = c == 0 ? -0.5 / (nu + rho + 2.0) : inv_linear_dk(c, -0.5, nu, rho + 2.0);
  }
  out[0] = kappa_dk(c, nu, rho) * H + e * P;
  return true;
}

// ------------------------------------------------------------ draw, start
// y = mu + (L z or L^-T z)/sqrt(w); tmp (degenerate only) projects on the range
template <bool Prec, class St, class S>
LAPLACE_HD void rng_whitened(const S& s, const mvstudent_t_state<St>& st, std::uint64_t seed, std::uint64_t unit,
                             double* x) {
  const int d = s.dim;
  const double* L = st.gauss.factor;
  counter_uniform_source src{seed, unit};
  for (int k = 0; k < d; ++k) x[k] = standard_normal_by_inversion(src);
  for (int i = d - 1; i >= 0; --i) {
    if constexpr (Prec) {
      double t = x[i];
      for (int k = i + 1; k < d; ++k) t -= L[packed_index(k, i)] * x[k];
      x[i] = t / L[packed_index(i, i)];
    } else {
      double t = 0.0;
      for (int k = 0; k <= i; ++k) t += L[packed_index(i, k)] * x[k];
      x[i] = t;
    }
  }
  const double sw = Kokkos::sqrt(2.0 * standard_gamma_by_inversion(0.5 * st.nu, src) / st.nu);
  for (int k = 0; k < d; ++k) x[k] /= sw;
}

}  // namespace detail_mvstudent_t
}  // namespace laplace::distrib
