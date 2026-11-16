// student_t1_distrib (laplace::distrib): the Student t distribution in its location mu,
// scale sigma > 0 and degrees of freedom nu > 0. With z = (y - mu)/sigma and A = nu + z^2,
//
//   l(mu, sigma, nu; y) = c(nu) - log sigma + Lambda(z, nu),
//   Lambda(z, nu) = -((nu + 1)/2) log(1 + z^2/nu),
//   c(nu) = log Gamma((nu + 1)/2) - log Gamma(nu/2) - log(nu pi)/2,
//
//   F(y) = T_nu(z), the Student t distribution function (laplace::math::own::log_student_cdf).
//
// sigma is the scale, not the standard deviation (student_t2 has the standard deviation).
// l, log F and log(1 - F) are C^inf in the three parameters and in y: no kinks.
//
// The derivatives in (mu, sigma, nu) of l: the mixed table D[c][j] = d^c/dnu^c d^j/dz^j Lambda
// goes to detail_location_scale::mixed_shape_dk, which writes the whole sym_tensor<3, K>; c(nu)
// (cpoly of student.hpp, free of cancellation) adds its K-th derivative to the component with
// every index in nu, and -log sigma adds (-1)^K (K - 1)! / sigma^K to the one with every index
// in sigma. The table is closed, and written so that nothing cancels as nu grows (the nu-derivatives
// of Lambda would otherwise lose a factor nu at each order against a part of size 1/nu):
//   c = 0, j >= 1:  Lambda_z = -(nu + 1) z/A, so D[0][j] = -(nu + 1) (z g(1, j-1) + (j-1) g(1, j-2)),
//     with g(m, n) = d^n/dz^n A^-m, which obeys g(m, n) = -2 m (z g(m + 1, n - 1) + (n - 1) g(m + 1, n - 2));
//   c >= 1, j >= 1: d/dnu Lambda_z = (nu + 1) z A^-2 - z A^-1 = z (1 - z^2) A^-2, so with P(z) = z (1 - z^2)
//     D[c][j] = -(-1)^c c! d^(j-1)/dz^(j-1) [P(z) A^-(c+1)], by Leibniz's rule over P (three derivatives);
//   j = 0 (needed only at c = K): with r = z^2/A, q = nu/A = 1 - r, log1pmx(t) = log1p(t) - t,
//     Lambda_nu = (1/2) log1pmx(-r) + r/(2 nu), and for m = c - 1 >= 1
//     D[c][0] = (1/2)(-1)^m [(m - 1)! nu^-m B_m(r) + m! nu^-(m+1) r (1 + q + ... + q^m)],
//     B_m(r) = (1 + m r) (1 - r)^m - 1 = r^2 sum_(k = 2)^(m+1) (-1)^k [C(m, k) - m C(m, k-1)] r^(k-2)
//     (the polynomial below r = 1/4, the product above).
// The closed forms were checked against SymPy's own derivative of Lambda for every (c, j) with
// c + j <= 6 (tools/reference/distrib_student_t.py).
//
// The derivatives of log F and log(1 - F): log F = f(z, nu) with f = log T_nu(z), whose mixed
// (z, nu) table comes from one call of the quadrature of laplace::math::own (all the orders up
// to K at once), and the same table feeds mixed_shape_dk; log(1 - F) = f(-z, nu), with the sign
// (-1)^j on the j-th derivative in z. On the side where the function tends to 0 (log F for
// z >= 1, log(1 - F) for z <= -1) the polynomials of mixed_shape_dk cancel (as in the Cauchy's
// far tail: the leading term of log(1 - S) = -S - ... is a power of z whose derivatives in sigma
// vanish at integer nu): there the function is log(1 - e^l) with l the log of the other tail, whose
// tensors are formed without cancellation, and the derivatives of log(1 - e^l) come from those of l
// by the complete Bell polynomials (bell_log_one_minus_exp below).
//
// The expected information is closed, with a = (nu + 1)/(nu + 3), b = 2 nu/(nu + 3),
// e = -2/((nu + 1)(nu + 3)):
//   I_mumu = a/sigma^2, I_sigmasigma = b/sigma^2, I_sigmanu = e/sigma, I_musigma = I_munu = 0,
//   I_nunu = (1/4)[psi'(nu/2) - psi'((nu + 1)/2)] - (nu + 5)/(2 nu (nu + 1)(nu + 3)).
// I_nunu is of order 3.5/nu^4 (the two parts are of order 1/(2 nu^2), with the same terms down to
// 1/nu^3): below nu = 18 it is formed from own::detail_ibd::dpsi_all (the polygamma difference
// without forming the two terms) minus the rational part written as a sum of positive products;
// from nu = 18 on from the series in 1/nu (nu^-4 to nu^-38, whose coefficients are those of the
// asymptotic series of the polygamma difference minus the series of the rational part,
// tools/reference/distrib_student_t.py), cut where it is below 2^-55 of the sum. The derivatives
// in sigma are powers of sigma, those in nu are, for a, b and e, sums of positive products of
// powers of 1/(nu + 1) and 1/(nu + 3).
//
// No closed quantile: the generic route (quantile_by_root) and the generic draw apply.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta_deriv.hpp>
#include <laplace/math/student.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_student_t1 {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// position in sym_tensor<3, a + b + c> of the sorted multi-index with a zeros, b ones and c twos
// (the derivative orders in mu, sigma and nu), for orders known only at run time
LAPLACE_HD_FORCEINLINE constexpr int pos3(int a, int b, int c) {
  const int k = a + b + c;
  int pos = 0, prev = 0;
  for (int j = 0; j < k; ++j) {
    const int v_end = j < a ? 0 : (j < a + b ? 1 : 2);
    for (int v = prev; v < v_end; ++v) pos += binomial_coef(3 - v + k - j - 2, k - j - 1);
    prev = v_end;
  }
  return pos;
}

// sum over the compositions (i_1, ..., i_NF) of q of prod_l (nu + alpha_l)^-(i_l + 1): a sum of
// positive terms; (-1)^q q! times it is the q-th derivative of 1 / prod_l (nu + alpha_l)
template <int NF>
LAPLACE_HD double inv_prod_sum(int q, const double* alpha, double nu) {
  double cur[7], nxt[7];
  const double w0 = 1.0 / (nu + alpha[0]);
  double p = w0;
  for (int i = 0; i <= q; ++i, p *= w0) cur[i] = p;
  for (int l = 1; l < NF; ++l) {
    const double w = 1.0 / (nu + alpha[l]);
    for (int s = 0; s <= q; ++s) {
      double acc = 0.0, wp = w;
      for (int i = 0; i <= s; ++i, wp *= w) acc += cur[s - i] * wp;
      nxt[s] = acc;
    }
    for (int s = 0; s <= q; ++s) cur[s] = nxt[s];
  }
  return cur[q];
}

// d^q/dnu^q [1 / prod_l (nu + alpha_l)]
template <int NF>
LAPLACE_HD double inv_prod_deriv(int q, const double* alpha, double nu) {
  return sign_of_power(q) * fact(q) * inv_prod_sum<NF>(q, alpha, nu);
}

// ---------------------------------------------------------------- the table of Lambda

// B_m(r) = (1 + m r)(1 - r)^m - 1, 1 <= m <= 5; q = 1 - r given
LAPLACE_HD_FORCEINLINE double bracket_B(int m, double r, double q) {
  if (r >= 0.25) {
    double qm = 1.0;
    for (int i = 0; i < m; ++i) qm *= q;
    return (1.0 + m * r) * qm - 1.0;
  }
  // r^2 sum_(k=2)^(m+1) (-1)^k [C(m, k) - m C(m, k-1)] r^(k-2), by Horner from the top
  double acc = 0.0;
  for (int k = m + 1; k >= 2; --k) {
    const double c = binomial_coef(m, k) - m * binomial_coef(m, k - 1);
    acc = acc * r + ((k % 2 == 0) ? c : -c);
  }
  return r * r * acc;
}

// D[c * (K + 1) + j] = d^c/dnu^c d^j/dz^j Lambda(z, nu) for 1 <= j, c + j <= K, and the pure
// nu-derivative of order K at D[K * (K + 1)]; the other entries are left as they are.
template <int K>
LAPLACE_HD void lambda_table(double z, double nu, double* D) {
  const double z2 = z * z, A = nu + z2, iA = 1.0 / A;
  // g[m][n] = d^n/dz^n A^-m, m = 1..K, m + n <= K
  double g[K + 1][K + 1] = {};
  {
    double p = 1.0;
    for (int m = 1; m <= K; ++m) {
      p *= iA;
      g[m][0] = p;
    }
    for (int n = 1; n < K; ++n)
      for (int m = 1; m + n <= K; ++m) {
        double v = z * g[m + 1][n - 1];
        if (n >= 2) v += (n - 1) * g[m + 1][n - 2];
        g[m][n] = -2.0 * m * v;
      }
  }
  const double P0 = z * Kokkos::fma(-z, z, 1.0), P1 = 1.0 - 3.0 * z2, P2 = -6.0 * z, P3 = -6.0;
  for (int c = 0; c < K; ++c)
    for (int j = 1; c + j <= K; ++j) {
      const int m = c + 1, n = j - 1;
      double h = P0 * g[m][n];
      if (n >= 1) h += n * P1 * g[m][n - 1];
      if (n >= 2) h += (0.5 * n * (n - 1)) * P2 * g[m][n - 2];
      if (n >= 3) h += (n * (n - 1) * (n - 2) / 6.0) * P3 * g[m][n - 3];
      double v;
      if (c == 0) {
        v = z * g[1][j - 1];
        if (j >= 2) v += (j - 1) * g[1][j - 2];
        v *= -(nu + 1.0);
      } else {
        v = -sign_of_power(c) * fact(c) * h;
      }
      D[c * (K + 1) + j] = v;
    }
  // the pure nu-derivative of order K
  {
    const double r = z2 * iA, q = nu * iA, inu = 1.0 / nu;
    double v;
    if constexpr (K == 1) {
      double l1pmx;  // log1p(-r) + r
      if (r <= 0.5) {
        l1pmx = math::own::detail_ibx::log1pmx(-r);
      } else {
        l1pmx = r - Kokkos::log1p(z2 * inu);
      }
      v = 0.5 * (l1pmx + r * inu);
    } else {
      constexpr int m = K - 1;
      double nm = 1.0;  // nu^-m
      for (int i = 0; i < m; ++i) nm *= inu;
      double S = 0.0, qp = 1.0;  // 1 + q + ... + q^m
      for (int i = 0; i <= m; ++i, qp *= q) S += qp;
      v = 0.5 * sign_of_power(m) *
          (fact(m - 1) * nm * bracket_B(m, r, q) + fact(m) * nm * inu * r * S);
    }
    D[K * (K + 1)] = v;
  }
}

// the series of I_nunu in 1/nu: sum_(n = 4)^(38) c_n nu^-n
constexpr int series_first = 4;
constexpr int series_terms = 35;
constexpr double series_split = 18.0;
#define LAPLACE_STUDENT_INFO_COEF \
  3.5, -13.0, 39.5, -119.0, 363.5, -1101.0, 3279.5, -9763.0, \
      29523.5, -89609.0, 265719.5, -778047.0, 2391483.5, -7639237.0, 21523359.5, -50159771.0, \
      193710243.5, -1135957185.0, 1743392199.5, 20741464265.0, 15690529803.5, -1499647110653.0, \
      141214768239.5, 95241191937261.0, 1270932914163.5, -7331625876126841.0, 11438396227479.5, \
      6.459082290353265e+17, 102945566047323.5, -6.492439067725179e+19, 926510094425919.5, \
      7.380720587361798e+21, 8338590849833284.0, -9.422577958801819e+23, 7.504731764849955e+16
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double series_dev[series_terms] = {LAPLACE_STUDENT_INFO_COEF};
#endif
inline constexpr double series_host[series_terms] = {LAPLACE_STUDENT_INFO_COEF};
LAPLACE_HD_FORCEINLINE double series_coef(int i) {
  KOKKOS_IF_ON_DEVICE((return series_dev[i];))
  KOKKOS_IF_ON_HOST((return series_host[i];))
}

// f[q] = d^q/dnu^q I_nunu(nu), q = 0..Q (Q <= 4), by the series from nu = split and by the polygamma
// difference below
template <int Q>
LAPLACE_HD void info_nunu_derivs_split(double nu, double* f, double split) {
  if (nu >= split) {
    const double w = 1.0 / nu;
    for (int q = 0; q <= Q; ++q) {
      // sum_n c_n rise(n, q) w^n, then times (-1)^q w^q; Horner from n = 38 down
      double acc = 0.0;
      for (int i = series_terms - 1; i >= 0; --i) {
        const int n = series_first + i;
        double rise = 1.0;
        for (int t = 0; t < q; ++t) rise *= (n + t);
        acc = acc * w + series_coef(i) * rise;
      }
      double pw = 1.0;
      for (int t = 0; t < series_first + q; ++t) pw *= w;
      f[q] = sign_of_power(q) * acc * pw;
    }
    return;
  }
  double dp[6], dD;
  math::own::detail_ibd::dpsi_all(0.5 * nu, 0.5, Q + 1, dp, dD);
  const double alpha3[3] = {0.0, 1.0, 3.0};
  double scale = 0.25;  // (1/4) 2^-q
  for (int q = 0; q <= Q; ++q, scale *= 0.5) {
    // R^(q) = (1/2)[(nu + 5) W^(q) + q W^(q-1)], W = 1/(nu (nu + 1)(nu + 3))
    double Rq = (nu + 5.0) * inv_prod_deriv<3>(q, alpha3, nu);
    if (q >= 1) Rq += q * inv_prod_deriv<3>(q - 1, alpha3, nu);
    f[q] = scale * dp[q + 1] - 0.5 * Rq;
  }
}

template <int Q>
LAPLACE_HD void info_nunu_derivs(double nu, double* f) {
  info_nunu_derivs_split<Q>(nu, f, series_split);
}

// the nu-derivatives, orders 0..Q, of a = (nu + 1)/(nu + 3), b = 2 nu/(nu + 3) and
// e = -2/((nu + 1)(nu + 3))
template <int Q>
LAPLACE_HD void info_ab_e_derivs(double nu, double* a, double* b, double* e) {
  const double alpha2[2] = {1.0, 3.0};
  a[0] = (nu + 1.0) / (nu + 3.0);
  b[0] = 2.0 * nu / (nu + 3.0);
  e[0] = -2.0 / ((nu + 1.0) * (nu + 3.0));
  double ip = 1.0 / (nu + 3.0);  // (nu + 3)^-(q+1)
  for (int q = 1; q <= Q; ++q) {
    ip /= (nu + 3.0);
    const double base = sign_of_power(q) * fact(q) * ip;
    a[q] = -2.0 * base;
    b[q] = -6.0 * base;
    e[q] = -2.0 * inv_prod_deriv<2>(q, alpha2, nu);
  }
}

// the derivatives in (sigma, nu) of the three entries a(nu) sigma^-2, b(nu) sigma^-2, e(nu) sigma^-1
// and f(nu) of the information, written into the pairs of an expected_info_tensor<3, K>
template <int K>
LAPLACE_HD void fill_info(const double* a, const double* b, const double* e, const double* f, double sigma,
                          expected_info_tensor<3, K>& t) {
  double is2[K + 1], is1[K + 1];  // d^p sigma^-2 and d^p sigma^-1
  {
    double s2 = 1.0 / (sigma * sigma), s1 = 1.0 / sigma;
    for (int p = 0; p <= K; ++p) {
      is2[p] = s2 * sign_of_power(p) * fact(p + 1);
      is1[p] = s1 * sign_of_power(p) * fact(p);
      s2 /= sigma;
      s1 /= sigma;
    }
  }
  for (int p = 0; p <= K; ++p) {
    const int q = K - p, pos = pos3(0, p, q);
    t.pair[0][pos] = a[q] * is2[p];
    t.pair[3][pos] = b[q] * is2[p];
    t.pair[4][pos] = e[q] * is1[p];
    if (p == 0) t.pair[5][pos] = f[q];
  }
}

// ---------------------------------------------------------------- log F and log(1 - F)

// |z| from which the side of log F (or log(1 - F)) that tends to 0 goes through the Bell recursion below
constexpr double cancel_split = 1.0;
constexpr int tensor_max = 28;

// the table D of order M of f(z, nu) = log T_nu(z), from L, the bivariate Taylor polynomial of
// laplace::math::own::detail_st::derivs; flip: the function is f(-z, nu), whose j-th derivative in z has the sign (-1)^j
template <int M>
LAPLACE_HD void cdf_table(const double* L, bool flip, double* D) {
  namespace ibd = math::own::detail_ibd;
  for (int c = 0; c <= M; ++c)
    for (int j = (c == M ? 0 : 1); c + j <= M; ++j) {
      double v = L[ibd::ix(j, c)] * ibd::factorial(j) * ibd::factorial(c);
      if (flip && (j % 2 == 1)) v = -v;
      D[c * (M + 1) + j] = v;
    }
}

template <int M>
LAPLACE_HD void store_cdf_tensor(const double* L, bool flip, double z, double sigma, double* T) {
  double D[(M + 1) * (M + 1)] = {};
  cdf_table<M>(L, flip, D);
  const sym_tensor<3, M> t = detail_location_scale::mixed_shape_dk<M>(D, z, sigma);
  for (int i = 0; i < sym_tensor<3, M>::size; ++i) T[M * tensor_max + i] = t[i];
}

// the tensors of orders 1..K of the log of the function, packed as T[k * tensor_max + position]
template <int K>
LAPLACE_HD void store_all_cdf_tensors(const double* L, bool flip, double z, double sigma, double* T) {
  store_cdf_tensor<1>(L, flip, z, sigma, T);
  if constexpr (K >= 2) store_cdf_tensor<2>(L, flip, z, sigma, T);
  if constexpr (K >= 3) store_cdf_tensor<3>(L, flip, z, sigma, T);
  if constexpr (K >= 4) store_cdf_tensor<4>(L, flip, z, sigma, T);
  if constexpr (K >= 5) store_cdf_tensor<5>(L, flip, z, sigma, T);
  if constexpr (K >= 6) store_cdf_tensor<6>(L, flip, z, sigma, T);
}

// The derivatives of order K of log(1 - e^l) for a function l of the parameters, from the tensors T of
// l (orders 1..K) and S = e^l, in the side where S is small and log(1 - S) = -S - S^2/2 - ... cannot be
// obtained from the z-derivatives of log(1 - S) without cancellation. With the derivative of e^l over a
// multiset U of the indices equal to S B_U(l) (B the complete Bell polynomial of the derivatives of l,
// B_U = sum over the block V of U holding its lowest index of l_V B_(U - V), B_empty = 1), the function
// w = 1 - e^l has D_U w / w = E_U := -(S/(1 - S)) B_U (U not empty), and log w has derivatives Lam_U with
// E_U = sum_V Lam_V E_(U - V), so Lam_U = E_U - sum_(V != U) Lam_V E_(U - V). Every term is of the size of S
// times a product of derivatives of l, which have one sign in the tail: no cancellation.
template <int K>
LAPLACE_HD sym_tensor<3, K> bell_log_one_minus_exp(const double* T, double S) {
  sym_tensor<3, K> out;
  constexpr int full = (1 << K) - 1;
  const double r = S / (1.0 - S);
  for (int b = 0; b <= K; ++b)
    for (int c = 0; b + c <= K; ++c) {
      const int a = K - b - c;
      int type[K];  // 0 mu, 1 sigma, 2 nu
      for (int i = 0; i < K; ++i) type[i] = i < a ? 0 : (i < a + b ? 1 : 2);
      double l[1 << K], B[1 << K], E[1 << K], Lam[1 << K];
      for (int mask = 1; mask <= full; ++mask) {
        int n[3] = {0, 0, 0};
        for (int i = 0; i < K; ++i)
          if (mask >> i & 1) ++n[type[i]];
        l[mask] = T[(n[0] + n[1] + n[2]) * tensor_max + pos3(n[0], n[1], n[2])];
      }
      B[0] = 1.0;
      E[0] = 1.0;
      Lam[0] = 0.0;
      for (int mask = 1; mask <= full; ++mask) {
        const int low = mask & -mask, rest = mask ^ low;
        double acc = 0.0;
        for (int sub = rest;; sub = (sub - 1) & rest) {
          const int U = sub | low;
          acc += l[U] * B[mask ^ U];
          if (sub == 0) break;
        }
        B[mask] = acc;
        E[mask] = -r * acc;
        double lam = E[mask];
        for (int sub = rest;; sub = (sub - 1) & rest) {
          const int U = sub | low;
          if (U != mask) lam -= Lam[U] * E[mask ^ U];
          if (sub == 0) break;
        }
        Lam[mask] = lam;
      }
      out[pos3(a, b, c)] = Lam[full];
    }
  return out;
}

// ---------------------------------------------------------------- the far tail of the (mu, sigma) block

// For |y - mu| >= sigma sqrt(nu) the z-polynomials of mixed_shape_dk cancel: their terms grow
// like z^(K-1) while the component they build is O(1) or smaller, and at |z| = 1e8 the order six
// loses three digits of the tensor's largest component. The components with no nu derivative are
// closed in one complex power instead. With u = y - mu and w = sigma sqrt(nu),
//
//   log f = c(nu) + ((nu + 1)/2) log nu + nu log sigma - (nu + 1) Re log(u + i w),
//
// because log(1 + z^2/nu) = log(u^2 + w^2) - log(nu sigma^2) and log(u^2 + w^2) = 2 Re log(u + i w).
// Since d/dmu = -d/du and d/dsigma = sqrt(nu) d/dw, and log(u + i w) obeys
// d^a/du^a d^b/dw^b log(u + i w) = i^b (-1)^(K-1) (K-1)! (u + i w)^-K with K = a + b >= 1,
//
//   d^a/dmu^a d^b/dsigma^b l = -(nu + 1) (-1)^a nu^(b/2) (-1)^(K-1) (K-1)! Re[i^b (u + i w)^-K]
//                              + [a == 0] nu (-1)^(b-1) (b - 1)! sigma^-b,
//
// the second term being the only one nu log sigma contributes. The power is taken from the
// reciprocal of u + i w by repeated multiplication, scaled by m = max(|u|, w) so that neither
// u^2 + w^2 nor the power overflows, and every factor of the scale is spent one at a time.
// tools/reference/distrib_student_t.py checks the two forms against SymPy.
template <int K>
LAPLACE_HD void far_mu_sigma_block(double y, double mu, double sigma, double nu, double* out) {
  const double u = y - mu, w = sigma * Kokkos::sqrt(nu);
  const double m = Kokkos::fmax(Kokkos::fabs(u), w);
  const double us = u / m, ws = w / m;
  const double r2 = Kokkos::fma(us, us, ws * ws);
  const double ir = 1.0 / r2;
  const double cr = us * ir, ci = -ws * ir;  // 1/(u/m + i w/m)
  double pr = cr, pi = ci;
  for (int k = 1; k < K; ++k) {
    const double nr = pr * cr - pi * ci, ni = pr * ci + pi * cr;
    pr = nr;
    pi = ni;
  }
  const double snu = Kokkos::sqrt(nu);
  for (int b = 0; b <= K; ++b) {
    const int a = K - b;
    double re;
    switch (b % 4) {
      case 0: re = pr; break;
      case 1: re = -pi; break;
      case 2: re = -pr; break;
      default: re = pi; break;
    }
    double v = -(nu + 1.0) * sign_of_power(a) * sign_of_power(K - 1) * fact(K - 1) * re;
    for (int i = 0; i < b; ++i) v *= snu / m;
    for (int i = 0; i < a; ++i) v /= m;
    if (a == 0) {
      double s = nu * sign_of_power(b - 1) * fact(b - 1);
      for (int i = 0; i < b; ++i) s /= sigma;
      v += s;
    }
    out[b] = v;
  }
}

}  // namespace detail_student_t1

/**
 * \ingroup distrib_student_t1
 * \brief Student t distribution parametrized by location, scale, and degrees of freedom.
 *
 * Models a real observation \f$y \in \mathbb{R}\f$ with location \f$\mu \in \mathbb{R}\f$, scale
 * \f$\sigma > 0\f$, and degrees of freedom \f$\nu > 0\f$ through the log-density
 * \f[
 * \ell(\mu, \sigma, \nu; y) = c(\nu) - \log\sigma - \frac{\nu + 1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right),
 * \qquad z = \frac{y - \mu}{\sigma},
 * \f]
 * where \f$c(\nu) = \log\Gamma((\nu + 1)/2) - \log\Gamma(\nu/2) - \frac{1}{2}\log(\nu\pi)\f$.
 * Here \f$\sigma\f$ is the scale parameter, not the standard deviation (which equals
 * \f$\sigma\sqrt{\nu/(\nu - 2)}\f$ for \f$\nu > 2\f$ and does not exist for \f$\nu \le 2\f$).
 * Setting \f$\nu = 1\f$ yields the Cauchy family, and \f$\nu \to \infty\f$ yields the Gaussian
 * family.
 *
 * \see distrib_student_t1, student_t2_distrib, cauchy_distrib, gaussian1_distrib
 */
struct student_t1_distrib {
  /// Location parameter \f$\mu \in \mathbb{R}\f$.
  double mu;
  /// Scale parameter \f$\sigma > 0\f$ (not the standard deviation).
  double sigma;
  /// Degrees of freedom \f$\nu > 0\f$.
  double nu;

  /// Distribution kind (continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of distribution parameters (\f$3\f$).
  static constexpr int n_params = 3;
  /// Regularity of the log-density with respect to \f$y\f$ (infinitely smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Parameter metadata descriptors for `mu`, `sigma`, and `nu`.
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"nu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the log-density \f$\ell(\mu, \sigma, \nu; y)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log-density value at `y` (\f$-\infty\f$ when `y` is \f$\pm\infty\f$).
   * \see logpdf_dk, log_cdf, expected_info
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    double cp[1];
    math::own::detail_st::cpoly(nu, 0, cp);
    return cp[0] - math::log(sigma) - 0.5 * (nu + 1.0) * Kokkos::log1p(z * z / nu);
  }

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log f(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see logpdf, logpdf_d1, logpdf_dy_dk
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double D[(K + 1) * (K + 1)] = {};
    detail_student_t1::lambda_table<K>(z, nu, D);
    sym_tensor<3, K> t = detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
    double cp[7];
    math::own::detail_st::cpoly(nu, K, cp);
    int idx[K];
    for (int i = 0; i < K; ++i) idx[i] = 2;
    t[sym_tensor_position<3, K>(idx)] += detail_student_t1::fact(K) * cp[K];
    for (int i = 0; i < K; ++i) idx[i] = 1;
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[sym_tensor_position<3, K>(idx)] += detail_student_t1::sign_of_power(K) * detail_student_t1::fact(K - 1) * isk;
    // the components with no nu derivative, rewritten where the z-polynomials cancel; the closed
    // form carries the whole of nu log sigma, so it replaces the two terms added above as well
    if (Kokkos::fabs(y - mu) >= sigma * Kokkos::sqrt(nu)) {
      double far[K + 1];
      detail_student_t1::far_mu_sigma_block<K>(y, mu, sigma, nu, far);
      for (int b = 0; b <= K; ++b) t[detail_student_t1::pos3(K - b, b, 0)] = far[b];
    }
    return t;
  }
  /**
   * \name Log-density parameter derivatives by order
   * \ingroup distrib_student_t1
   * \brief Shorthand accessors calling `logpdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see logpdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate \f$\partial^J / \partial y^J\f$ composed with order-\f$K\f$ parameter
   *        derivatives of \f$\log f(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam J Observation derivative order (\f$J \ge 0\f$).
   * \tparam K Parameter derivative order (\f$K \ge 0\f$, with \f$J + K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of order \f$K\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \see logpdf_dk
   */
  // l depends on y and mu only through y - mu, so d/dy = -d/dmu: the component with a derivatives
  // in mu, b in sigma and c in nu of d^J/dy^J of order K is (-1)^J times that with a + J in mu of
  // logpdf_dk<J + K>.
  template <int J, int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<3, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<3, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<3, K> r;
      const double s = detail_student_t1::sign_of_power(J);
      for (int b = 0; b <= K; ++b)
        for (int c = 0; b + c <= K; ++c) r[detail_student_t1::pos3(K - b - c, b, c)] = s * t[detail_student_t1::pos3(K - b - c + J, b, c)];
      return r;
    }
  }

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the expected Fisher information matrix in \f$(\mu, \sigma, \nu)\f$.
   * \return Symmetric \f$3 \times 3\f$ Fisher information matrix stored in `sym_tensor<3, 2>`.
   * \see expected_info_dk, logpdf_dk
   */
  LAPLACE_HD sym_tensor<3, 2> expected_info() const {
    double a[1], b[1], e[1], f[1];
    detail_student_t1::info_ab_e_derivs<0>(nu, a, b, e);
    detail_student_t1::info_nunu_derivs<0>(nu, f);
    const double is2 = 1.0 / (sigma * sigma);
    sym_tensor<3, 2> t;
    t[0] = a[0] * is2;
    t[3] = b[0] * is2;
    t[4] = e[0] / sigma;
    t[5] = f[0];
    return t;
  }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate order-\f$K\f$ parameter derivatives of the expected Fisher information in
   *        \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Expected-information tensor of order \f$K\f$.
   * \see expected_info, expected_info_d1
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<3, K> expected_info_dk() const {
    double a[K + 1], b[K + 1], e[K + 1], f[K + 1];
    detail_student_t1::info_ab_e_derivs<K>(nu, a, b, e);
    detail_student_t1::info_nunu_derivs<K>(nu, f);
    expected_info_tensor<3, K> t;
    detail_student_t1::fill_info<K>(a, b, e, f, sigma, t);
    return t;
  }
  /**
   * \name Expected-information parameter derivatives by order
   * \ingroup distrib_student_t1
   * \brief Shorthand accessors calling `expected_info_dk<K>()` for \f$K = 1, \dots, 4\f$.
   * \return Expected-information tensor of the corresponding order.
   * \see expected_info_dk
   * @{
   */
  /// First-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the cumulative distribution function \f$F(y) = T_\nu((y - \mu)/\sigma)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Lower-tail probability in \f$[0, 1]\f$ (\f$0\f$ at \f$-\infty\f$, \f$1\f$ at \f$+\infty\f$).
   * \see ccdf, log_cdf
   */
  LAPLACE_HD double cdf(double y) const { return math::exp(log_cdf(y)); }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the complementary cumulative distribution function \f$1 - F(y) = T_\nu(-(y - \mu)/\sigma)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Upper-tail probability in \f$[0, 1]\f$ (\f$1\f$ at \f$-\infty\f$, \f$0\f$ at \f$+\infty\f$).
   * \see cdf, log_ccdf
   */
  LAPLACE_HD double ccdf(double y) const { return math::exp(log_ccdf(y)); }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the log cumulative distribution function \f$\log F(y)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log lower-tail probability in \f$[-\infty, 0]\f$ (\f$-\infty\f$ at \f$-\infty\f$, \f$0\f$ at \f$+\infty\f$).
   * \see log_ccdf, log_cdf_dk, cdf
   */
  LAPLACE_HD double log_cdf(double y) const { return math::own::log_student_cdf((y - mu) / sigma, nu); }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate the log complementary cumulative distribution function \f$\log(1 - F(y))\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log upper-tail probability in \f$[-\infty, 0]\f$ (\f$0\f$ at \f$-\infty\f$, \f$-\infty\f$ at \f$+\infty\f$).
   * \see log_cdf, log_ccdf_dk, ccdf
   */
  LAPLACE_HD double log_ccdf(double y) const { return math::own::log_student_cdf(-(y - mu) / sigma, nu); }

  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$
   *        in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \param upper If `false`, differentiate \f$\log F(y)\f$; if `true`, differentiate \f$\log(1 - F(y))\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_dk, log_ccdf_dk
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_tail_dk(double y, bool upper) const {
    const double z = (y - mu) / sigma;
    double L[math::own::detail_st::NC];
    const bool cancel = upper ? (z <= -detail_student_t1::cancel_split) : (z >= detail_student_t1::cancel_split);
    if (!cancel) {
      math::own::detail_st::derivs<K>(upper ? -z : z, nu, math::own::detail_st::par{}, L);
      double D[(K + 1) * (K + 1)] = {};
      detail_student_t1::cdf_table<K>(L, upper, D);
      return detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
    }
    // the side where the function tends to 0: log(1 - e^l) of the log l of the other tail, whose
    // tensors have no cancellation
    math::own::detail_st::derivs<K>(upper ? z : -z, nu, math::own::detail_st::par{}, L);
    double T[(K + 1) * detail_student_t1::tensor_max];
    detail_student_t1::store_all_cdf_tensors<K>(L, !upper, z, sigma, T);
    const double small = math::exp(upper ? log_cdf(y) : log_ccdf(y));
    return detail_student_t1::bell_log_one_minus_exp<K>(T, small);
  }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log F(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_tail_dk, log_cdf, log_cdf_d1
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_dk(double y) const { return log_cdf_tail_dk<K>(y, false); }
  /**
   * \ingroup distrib_student_t1
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log(1 - F(y))\f$ in
   *        \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_tail_dk, log_ccdf, log_ccdf_d1
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_ccdf_dk(double y) const { return log_cdf_tail_dk<K>(y, true); }
  /**
   * \name Log-CDF parameter derivatives by order
   * \ingroup distrib_student_t1
   * \brief Shorthand accessors calling `log_cdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see log_cdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Log-CCDF parameter derivatives by order
   * \ingroup distrib_student_t1
   * \brief Shorthand accessors calling `log_ccdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see log_ccdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of the support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of the support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (scale \f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }
  /**
   * \ingroup distrib_student_t1
   * \brief Return parameter-domain non-smooth points (none for Student t).
   * \return Empty kink list.
   * \see y_kinks
   */
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// Return observation-domain non-smooth points (none for Student t).
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
  /**
   * \ingroup distrib_student_t1
   * \brief Construct a starting parameter vector inside the domain from sample summaries.
   * \param s Sample summary statistics (`median`, `lower_quartile`, `upper_quartile`).
   * \return Valid starting `student_t1_distrib` with median location, half-IQR scale, and \f$\nu = 5\f$.
   * \see distrib_start
   */
  // Starting value from a sample (generic_start.hpp): the median, half the interquartile range (robust, the
  // law may have no moments) and nu = 5; strictly inside the domain.
  student_t1_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.median, params[0]), start_inside(0.5 * (s.upper_quartile - s.lower_quartile), params[1]),
            start_inside(5.0, params[2])};
  }
};

}  // namespace laplace::distrib
