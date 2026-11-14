// invgauss_distrib (laplace::distrib): the inverse Gaussian distribution on (0, inf) in its
// mean mu > 0 and dispersion phi > 0, with variance phi mu^3 (shape lambda = 1/phi),
//
//   f(y) = (2 pi phi y^3)^(-1/2) exp(-(y - mu)^2 / (2 phi mu^2 y)),
//   l(mu, phi; y) = -(1/2) log(2 pi phi y^3) - g(mu, y) / (2 phi),
//   g(mu, y) = (y - mu)^2 / (mu^2 y) = y/mu^2 - 2/mu + 1/y.
//
// The derivatives in mu and phi (a in mu, b in phi, a + b = K, b the position in the packed
// tensor) are closed forms, checked against SymPy's own derivative in
// tools/reference/distrib_invgauss.py:
//   g^(a)(mu) = (-1)^a a! mu^-(a+2) ((a+1) y - 2 mu)  (a >= 1),  g^(0) = ((y - mu)/mu)^2 / y,
//   component (a, b) of d^K l = -(1/2) g^(a) (-1)^b b! phi^-(b+1) + [a = 0, b >= 1] (-1)^b (b-1)!/(2 phi^b).
// Expected information diag(1/(phi mu^3), 1/(2 phi^2)), with I_muphi = 0.
//
// The cdf. With z1 = (y/mu - 1)/sqrt(phi y), z2 = (y/mu + 1)/sqrt(phi y) and E = 2/(phi mu) (so that
// E = (z2^2 - z1^2)/2 exactly),
//   F(y) = Phi(z1) + exp(E) Phi(-z2),    S(y) = 1 - F(y) = Phi(-z1) - exp(E) Phi(-z2).
// With H(z) = log Phi(-z) + z^2/2 = log(erfcx(z/sqrt2)/2) (no cancellation: erfcx of laplace::math), the identity
//   E + log Phi(-z2) = H(z2) - z1^2/2
// is applied before differentiating: the z1^2/2 part is a closed-form array in (mu, phi), and only H goes through
// chain_rule_pair_dk. The survival is S = Phi(-z1) (1 - e^X) with X = H(z2) - H(z1) (the large parts cancel
// exactly); log F = A + log(1 + e^(X_F)) with A = H(-z1) - z1^2/2 and X_F = H(z2) - H(-z1). The derivatives of
// H are H' = -sqrt2 K(a), H'' = -K'(a) with a = z/sqrt2 and K the continued fraction of erfc (a >= cf_min_a), and
// (log Phi)^(n)(-z) for n >= 3 from laplace::math. Values are computed on the log scale with the erfcx ratio.

// The quantile is not a member: it is the generic route (generic_quantile.hpp).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/logistic.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_invgauss {

namespace cp = laplace::math::detail_chain_pair;

// number of partial derivatives of total order at most 6 in two variables, and their packing
inline constexpr int n_pairs = cp::pair_count(6);
LAPLACE_HD_FORCEINLINE constexpr int pidx(int i, int j) { return cp::pair_index(i, j); }

// (-1)^n
LAPLACE_HD_FORCEINLINE double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// Measurement switches. cf_route: the continued fraction of erfc gives erfcx'(a) and K(a) for a >= cf_min_a (false: the
// erfcx values, which cancel by about a^2). survival_threshold: below d (1 + |a|) the delta series replaces the direct
// difference of erfcx values (0: never).
inline constexpr bool cf_route = true;
// Below cf_min_a the erfcx-difference form is used: the continued fraction at depth 200 has K' error 1.5e-16 at a = 1.5 and
// the difference form 6e-15, at a = 1 the reverse (2.8e-16 against 1.4e-14).
inline constexpr double cf_min_a = 1.25;
inline constexpr double survival_threshold = 1.0;
inline constexpr double erfcx_c = 1.1283791670955125739;  // 2/sqrt(pi)
inline constexpr double sqrt2 = 1.4142135623730950488;

// erfcx(a) = (1/sqrt(pi)) / (a + K(a)) with K = (1/2)/(a + 1/(a + (3/2)/(a + ...))) the continued fraction of erfc.
// Evaluated backwards from depth 200 together with K'(a), from the recurrences T_k = (k/2)/(a + T_(k+1)) and
// T_k' = -T_k (1 + T_(k+1)') / (a + T_(k+1)), whose factors are all positive: no cancellation. Used for
// a >= cf_min_a; at depth 60 K' was wrong by 4e-13 at a = 1.78, which set the far tail of log(1 - F).
LAPLACE_HD void erfc_cf(double a, double& K, double& K1) {
  double t = 0.0, tp = 0.0;
  for (int k = 200; k >= 1; --k) {
    const double D = a + t;
    const double tn = (0.5 * k) / D;
    tp = -tn * (1.0 + tp) / D;
    t = tn;
  }
  K = t;
  K1 = tp;
}

// K(a) = c/(2 erfcx(a)) - a and K'(a) = -1 + 2 K(a) (a + K(a)), with a + K = c/(2 erfcx(a)) exact in the
// erfcx value: the route used where cf_route is off or a < cf_min_a (f = erfcx(a); 0 when f overflows).
LAPLACE_HD void erfc_K(double a, double f, double& K, double& K1) {
  if (cf_route && a >= cf_min_a) {
    erfc_cf(a, K, K1);
    return;
  }
  const double eps = erfcx_c / (2.0 * f);
  K = eps - a;
  K1 = -1.0 + 2.0 * K * eps;
}

// erfcx'(a) = -c K / (a + K) from the continued fraction (a >= cf_min_a), else 2a erfcx(a) - c, which cancels by about a^2
LAPLACE_HD double erfcx_d1(double a, double f) {
  if (!(cf_route && a >= cf_min_a)) return 2.0 * a * f - erfcx_c;
  double K, K1;
  erfc_cf(a, K, K1);
  return -erfcx_c * K / (a + K);
}

// The derivatives of nu(x) = log(1 - e^x) for x < 0: nd[n] = nu^(n)(x) = -R_(n-1), n = 1..K, with R_0 = r =
// e^x/(1 - e^x) and R_(k+1) = R_k + sum_i C(k, i) R_i R_(k-i), from r' = r + r^2 (all terms positive).
template <int K>
LAPLACE_HD void nu_derivs(double x, double* nd) {
  double R[K + 1];
  R[0] = Kokkos::exp(x) / (-Kokkos::expm1(x));
  for (int k = 0; k + 1 < K; ++k) {
    double acc = R[k];
    for (int i = 0; i <= k; ++i) acc += double(binomial_coef(k, i)) * R[i] * R[k - i];
    R[k + 1] = acc;
  }
  for (int n = 1; n <= K; ++n) nd[n] = -R[n - 1];
}

// The inner map z_q(mu, phi) = (y + q mu)/(mu sqrt(phi y)) with q = -1 for z1 and q = +1 for z2: the product
// g(mu) c(phi) of g(mu) = (y + q mu)/mu and c(phi) = (phi y)^(-1/2). Its partial derivative of order a in mu and b
// in phi is g^(a)(mu) c^(b)(phi), with c^(b) = c coef_b phi^-b and coef_b = prod_(k < b) (-1/2 - k).
// zp[pidx(a, b)] for a + b <= K; zp[0] is the value z_q.
template <int K>
LAPLACE_HD void z_map(double y, double mu, double phi, double q, double* zp) {
  const double c = 1.0 / Kokkos::sqrt(phi * y);
  double gm[K + 1];
  gm[0] = (y + q * mu) / mu;
  double im = 1.0 / mu, f = 1.0;  // im = mu^-(a+1) at order a (starting from a = 0), f = (-1)^a a!
  for (int a = 1; a <= K; ++a) {
    im /= mu;
    f *= -a;
    gm[a] = y * f * im;
  }
  double cd[K + 1];
  double coef = 1.0, ip = 1.0;  // coef = coef_b, ip = phi^-b
  for (int b = 0; b <= K; ++b) {
    cd[b] = c * coef * ip;
    coef *= (-0.5 - b);
    ip /= phi;
  }
  for (int n = 0; n <= K; ++n)
    for (int b = 0; b <= n; ++b) zp[pidx(n - b, b)] = gm[n - b] * cd[b];
}

// Q = z1^2/2 = g1(mu)^2 / (2 y phi), g1 = (y - mu)/mu: its partial of order a in mu and b in phi is
// (g1^2)^(a) (1/(2y)) (-1)^b b! phi^-(b+1), with (g1^2)^(a) = (-1)^a a! mu^-(a+2) ((a+1) y^2 - 2 y mu) for a >= 1.
template <int K>
LAPLACE_HD void z_square_half(double y, double mu, double phi, double* Q) {
  const double w = (y - mu) / mu;
  double gq[K + 1];
  gq[0] = w * w;
  double pw = 1.0 / (mu * mu);  // mu^-(a+2)
  double fa = 1.0;              // a!
  for (int a = 1; a <= K; ++a) {
    pw /= mu;
    fa *= a;
    gq[a] = sgn(a) * fa * pw * ((a + 1) * y * y - 2.0 * y * mu);
  }
  double ip[K + 1];  // phi^-(b+1)
  ip[0] = 1.0 / phi;
  for (int b = 1; b <= K; ++b) ip[b] = ip[b - 1] / phi;
  for (int n = 0; n <= K; ++n)
    for (int b = 0; b <= n; ++b) {
      double bf = 1.0;  // b!
      for (int i = 2; i <= b; ++i) bf *= i;
      Q[pidx(n - b, b)] = gq[n - b] * (0.5 / y) * sgn(b) * bf * ip[b];
    }
}

// H(z) = log Phi(-z) + z^2/2 = log(erfcx(z/sqrt2)/2), with H^(n)(z) = Hd[n], n = 0..K: H' = -sqrt2 K(a) and
// H'' = -K'(a) with a = z/sqrt2 (erfcx'/erfcx = -2K, d/dz = (d/da)/sqrt2), and H^(n) = (-1)^n (log Phi)^(n)(-z)
// for n >= 3 (laplace::math::log_Phi_dK at -z). Hd[0] is the value (not used in the derivative arrays).
template <int K>
LAPLACE_HD void H_derivs(double z, double* Hd) {
  const double a = 0.70710678118654752440 * z;
  const double f = math::erfcx(a);
  Hd[0] = math::log(f) - 0.69314718055994530942;
  if constexpr (K >= 1) {
    double Kc, K1;
    erfc_K(a, f, Kc, K1);
    Hd[1] = -sqrt2 * Kc;
    if constexpr (K >= 2) Hd[2] = -K1;
  }
  if constexpr (K >= 3) Hd[3] = -math::log_Phi_d3(-z);
  if constexpr (K >= 4) Hd[4] = math::log_Phi_d4(-z);
  if constexpr (K >= 5) Hd[5] = -math::log_Phi_d5(-z);
  if constexpr (K >= 6) Hd[6] = math::log_Phi_d6(-z);
}

// compose_order / compose_upto: the partial derivatives in (mu, phi) of f(z(mu, phi)) for the outer derivatives
// fd[0..N] of f at z: the chain rule of chain_rule_pair_dk with the second inner map identically zero, written into
// arr[pidx(N - b, b)] = d^N H / dmu^(N-b) dphi^b. arr[pidx(a, b)] for 1 <= a + b <= K, arr[0] = fd[0].
template <int N>
LAPLACE_HD void compose_order(const double* fd, const double* zp, double* arr) {
  double G[n_pairs] = {};
  for (int n = 0; n <= N; ++n) G[pidx(n, 0)] = fd[n];
  const double u1[n_pairs] = {};
  double out[N + 1];
  math::chain_rule_pair_dk<N>(G, zp, u1, out);
  for (int b = 0; b <= N; ++b) arr[pidx(N - b, b)] = out[b];
}

template <int K>
LAPLACE_HD void compose_upto(const double* fd, const double* zp, double* arr) {
  arr[0] = fd[0];
  if constexpr (K >= 1) compose_order<1>(fd, zp, arr);
  if constexpr (K >= 2) compose_order<2>(fd, zp, arr);
  if constexpr (K >= 3) compose_order<3>(fd, zp, arr);
  if constexpr (K >= 4) compose_order<4>(fd, zp, arr);
  if constexpr (K >= 5) compose_order<5>(fd, zp, arr);
  if constexpr (K >= 6) compose_order<6>(fd, zp, arr);
}

// The outer functions of the (a, x) form G(a, x) = a + f(x), whose partials are G[pidx(1, 0)] = 1 and
// G[pidx(0, j)] = f^(j)(x), j = 1..K, all others zero.
// For log(1 - F): f = nu, the log of 1 - e^x.
template <int K>
LAPLACE_HD void outer_survival(double x, double* G) {
  double nd[K + 1];
  nu_derivs<K>(x, nd);
  G[pidx(1, 0)] = 1.0;
  for (int j = 1; j <= K; ++j) G[pidx(0, j)] = nd[j];
}
// For log F: f = log(1 + e^x), f^(j) = sigma^(j-1) with sigma the logistic function (p_derivs).
template <int K>
LAPLACE_HD void outer_log_cdf(double x, double* G) {
  double p, q;
  detail_logistic::pq(x, p, q);
  double s[K + 1];
  detail_logistic::p_derivs<K>(p, q, s);  // s[k] = sigma^(k)(x), k < K
  G[pidx(1, 0)] = 1.0;
  for (int j = 1; j <= K; ++j) G[pidx(0, j)] = s[j - 1];
}

// L[pidx(a, b)] for 1 <= a + b <= K: the partial derivatives in (mu, phi) of G(A, X), order by order.
template <int N>
LAPLACE_HD void chain_order(const double* G, const double* A, const double* X, double* L) {
  double out[N + 1];
  math::chain_rule_pair_dk<N>(G, A, X, out);
  for (int b = 0; b <= N; ++b) L[pidx(N - b, b)] = out[b];
}
template <int K>
LAPLACE_HD void chain_upto(const double* G, const double* A, const double* X, double* L) {
  if constexpr (K >= 1) chain_order<1>(G, A, X, L);
  if constexpr (K >= 2) chain_order<2>(G, A, X, L);
  if constexpr (K >= 3) chain_order<3>(G, A, X, L);
  if constexpr (K >= 4) chain_order<4>(G, A, X, L);
  if constexpr (K >= 5) chain_order<5>(G, A, X, L);
  if constexpr (K >= 6) chain_order<6>(G, A, X, L);
}

// The inner map d(phi) = sqrt(2/(phi y)) = sqrt2 c(phi) with c as in z_map: Dd[pidx(0, b)] = sqrt2 c^(b)(phi)
template <int K>
LAPLACE_HD void delta_map(double y, double phi, double* Dd) {
  const double c = 1.0 / Kokkos::sqrt(phi * y);
  double coef = 1.0, ip = 1.0;  // coef = coef_b, ip = phi^-b
  for (int b = 0; b <= K; ++b) {
    Dd[pidx(0, b)] = sqrt2 * c * coef * ip;
    coef *= (-0.5 - b);
    ip /= phi;
  }
  Dd[0] = sqrt2 * c;
}

// The survival ratio q = 1 - erfcx(a + d)/erfcx(a) = D/erfcx(a), D = erfcx(a) - erfcx(a + d), for a = z1/sqrt2 and
// d = (z2 - z1)/sqrt2 = sqrt(2/(phi y)). Below d (1 + |a|) < survival_threshold the difference is the series
// D = - sum_(n >= 1) erfcx^(n)(a) d^n / n!, with erfcx^(n+1) = 2a erfcx^(n) + 2n erfcx^(n-1) (from erfcx' = 2a erfcx - c);
// it stops when a term is within 4 eps of the running sum. Above the threshold the direct difference is kept.
LAPLACE_HD double survival_q(double a, double d) {
  const double f = math::erfcx(a);
  if (!(d * (1.0 + Kokkos::fabs(a)) < survival_threshold)) return (f - math::erfcx(a + d)) / f;
  double Fm = f, Fc = erfcx_d1(a, f);  // erfcx^(n-1), erfcx^(n) at n = 1
  double dn = d;                        // d^n / n!
  double sum = 0.0;
  for (int n = 1; n <= 40; ++n) {
    const double term = -Fc * dn;
    sum += term;
    if (Kokkos::fabs(term) <= 4.0 * 0x1p-52 * Kokkos::fabs(sum)) break;
    const double Fn = 2.0 * a * Fc + 2.0 * n * Fm;
    Fm = Fc, Fc = Fn;
    dn *= d / (n + 1);
  }
  return sum / f;
}

// x = log(1 - q) and L0 = log(1 - F) = A' + log(q) (z1 > 0 or erfcx range), else from B - A' with log1mexp.
// S = 1 - F. Asv = log Phi(-z1), Bv = E + log Phi(-z2) (values only).
LAPLACE_HD_FORCEINLINE void survival_x_L0(double y, double phi, double z1, double z2, double Bv, double Asv, double& x,
                                          double& L0, double& S) {
  const double a1 = 0.70710678118654752440 * z1, a2 = 0.70710678118654752440 * z2;
  if (a1 > 0.0) {
    const double q = survival_q(a1, Kokkos::sqrt(2.0 / (phi * y)));
    x = Kokkos::log1p(-q);
    L0 = Asv + math::log(q);
  } else if (a1 > -20.0) {  // z1 <= 0: x from the logarithms of erfcx (absolute error about eps |log erfcx|), no E
    x = math::log(math::erfcx(a2)) - math::log(math::erfcx(a1));
    L0 = Asv + detail_generic::log1mexp(x);
  } else {
    x = Bv - Asv;
    L0 = Asv + detail_generic::log1mexp(x);
  }
  S = math::exp(L0);
}

// fd[n] = (log Phi)^(n)(t), n = 0..K (laplace::math::log_Phi_dK)
template <int K>
LAPLACE_HD void log_phi_fd(double t, double* fd) {
  fd[0] = math::log_Phi(t);
  if constexpr (K >= 1) fd[1] = math::log_Phi_d1(t);
  if constexpr (K >= 2) fd[2] = math::log_Phi_d2(t);
  if constexpr (K >= 3) fd[3] = math::log_Phi_d3(t);
  if constexpr (K >= 4) fd[4] = math::log_Phi_d4(t);
  if constexpr (K >= 5) fd[5] = math::log_Phi_d5(t);
  if constexpr (K >= 6) fd[6] = math::log_Phi_d6(t);
}

// log F (upper = false) or log(1 - F) (upper = true) as a symmetric tensor of order K in (mu, phi), for y > 0.
// The identity B = H(z2) - z1^2/2 and A' = H(z1) - z1^2/2, A = H(-z1) - z1^2/2 is applied before differentiating:
// the z1^2/2 parts are the closed-form arrays Q, and only H goes through the chain rule. The survival is
// log(1 - F) = A' + nu(X) with X = B - A' = H(z2) - H(z1), so the large parts cancel exactly in X. For log F,
// log F = A + ell(X_F) with X_F = B - A = H(z2) - H(-z1). Where S < 1/2 (F > 1/2), log F = nu(log S) composed with the
// survival arrays. Returns zeros for y <= 0 (the cdf is constant there).
template <int K>
LAPLACE_HD sym_tensor<2, K> log_cdf_dk(bool upper, double y, double mu, double phi) {
  sym_tensor<2, K> t;
  if (!(y > 0.0)) return t;
  double zp1[n_pairs] = {}, zp2[n_pairs] = {}, Q[n_pairs] = {};
  z_map<K>(y, mu, phi, -1.0, zp1);  // z1
  z_map<K>(y, mu, phi, 1.0, zp2);   // z2
  z_square_half<K>(y, mu, phi, Q);
  const double z1 = zp1[0], z2 = zp2[0];
  double Hz2[K + 1], Hz1[K + 1], Hmz1[K + 1], hm[K + 1];
  H_derivs<K>(z2, Hz2);
  H_derivs<K>(z1, Hz1);
  H_derivs<K>(-z1, Hmz1);
  for (int n = 0; n <= K; ++n) hm[n] = sgn(n) * Hmz1[n];  // d^n/du^n H(-u) at u = z1
  double Hc2[n_pairs] = {}, Hc1[n_pairs] = {}, Hcm[n_pairs] = {};
  compose_upto<K>(Hz2, zp2, Hc2);
  compose_upto<K>(Hz1, zp1, Hc1);
  compose_upto<K>(hm, zp1, Hcm);
  // the values of the survival (x: the weight of nu; L0 = log(1 - F))
  double x = 0.0, L0 = 0.0, S = 0.0;
  const double Asv = math::log_Phi(-z1), Bv = 2.0 / (mu * phi) + math::log_Phi(-z2);
  survival_x_L0(y, phi, z1, z2, Bv, Asv, x, L0, S);
  double Ap[n_pairs] = {}, Xs[n_pairs] = {}, G[n_pairs] = {}, L[n_pairs] = {};
  // A' = H(z1) - z1^2/2 where z1 > 0 (H small, z1^2/2 is the large part); for z1 <= 0 H(z1) and z1^2/2 are both of
  // size z1^2 and would cancel to eps |z1| in the arrays, so there A' = log Phi(-z1) is differentiated directly.
  if (z1 > 0.0) {
    for (int p = 1; p < n_pairs; ++p) Ap[p] = Hc1[p] - Q[p];
  } else {
    double fl[K + 1];
    log_phi_fd<K>(-z1, fl);  // fl[n] = (log Phi)^(n)(-z1)
    for (int n = 0; n <= K; ++n) fl[n] *= sgn(n);  // d^n/du^n log Phi(-u) at u = z1
    compose_upto<K>(fl, zp1, Ap);
  }
  // X = log(erfcx(a + d)/erfcx(a)) = H(z2) - H(z1) with a = z1/sqrt2, d = sqrt(2/(phi y)), as an (a, d) chain: the
  // outer G(a, d) = L(a + d) - L(a) with L = log erfcx = H(sqrt2 .) + const, so G_ij = (sqrt2)^(i+j) H^(i+j)(z2) for
  // j >= 1 (one term each), and G_i0 = (sqrt2)^i (H^(i)(z2) - H^(i)(z1)) is the only subtraction, of scalars.
  {
    double A1[n_pairs] = {}, Dd[n_pairs] = {}, Gx[n_pairs] = {};
    for (int p = 1; p < n_pairs; ++p) A1[p] = zp1[p] / sqrt2;
    delta_map<K>(y, phi, Dd);
    double s2n = 1.0;  // (sqrt2)^n
    for (int n = 1; n <= K; ++n) {
      s2n *= sqrt2;
      for (int i = 0; i <= n; ++i) {
        const int j = n - i;
        Gx[pidx(i, j)] = s2n * Hz2[n];
      }
    }
    // the j = 0 entries carry - (sqrt2)^i H^(i)(z1)
    double s2i = 1.0;
    for (int i = 0; i <= K; ++i) {
      if (i >= 1) s2i *= sqrt2;
      if (i >= 1) Gx[pidx(i, 0)] -= s2i * Hz1[i];
    }
    chain_upto<K>(Gx, A1, Dd, Xs);
  }
  Ap[0] = Asv;
  Xs[0] = x;
  if (upper || L0 < -0.69314718055994530942) {
    outer_survival<K>(x, G);
    chain_upto<K>(G, Ap, Xs, L);
    if (upper) {
      for (int b = 0; b <= K; ++b) t[b] = L[pidx(K - b, b)];
      return t;
    }
    // log F = nu(log S): the one-variable outer nu at L0 composed with the survival arrays L
    double nd[K + 1], Gn[n_pairs] = {}, zero[n_pairs] = {}, out[K + 1];
    nu_derivs<K>(L0, nd);
    for (int n = 1; n <= K; ++n) Gn[pidx(n, 0)] = nd[n];
    math::chain_rule_pair_dk<K>(Gn, L, zero, out);
    for (int b = 0; b <= K; ++b) t[b] = out[b];
    return t;
  }
  // F <= 1/2: log F = A + ell(X_F) with A = H(-z1) - z1^2/2 and X_F = H(z2) - H(-z1)
  double Af[n_pairs] = {}, Xf[n_pairs] = {}, out[K + 1];
  for (int p = 1; p < n_pairs; ++p) {
    Af[p] = Hcm[p] - Q[p];
    Xf[p] = Hc2[p] - Hcm[p];
  }
  Af[0] = math::log_Phi(z1);
  Xf[0] = Hz2[0] - Hmz1[0];
  outer_log_cdf<K>(Xf[0], G);
  math::chain_rule_pair_dk<K>(G, Af, Xf, out);
  for (int b = 0; b <= K; ++b) t[b] = out[b];
  return t;
}

// The values. log(1 - F) = A' + log1mexp(x) (x as in survival_x_L0). log F = log1p(-S) where F > 1/2,
// else M + log1p(e^(m - M)) with M = max(A, B) and m = min(A, B).
LAPLACE_HD_FORCEINLINE double log_ccdf_value(double y, double mu, double phi) {
  double zp1[1], zp2[1];
  z_map<0>(y, mu, phi, -1.0, zp1);
  z_map<0>(y, mu, phi, 1.0, zp2);
  const double Av = math::log_Phi(-zp1[0]), Bv = 2.0 / (mu * phi) + math::log_Phi(-zp2[0]);
  double x = 0.0, L0 = 0.0, S = 0.0;
  survival_x_L0(y, phi, zp1[0], zp2[0], Bv, Av, x, L0, S);
  return L0;
}
LAPLACE_HD_FORCEINLINE double log_cdf_value(double y, double mu, double phi) {
  double zp1[1], zp2[1];
  z_map<0>(y, mu, phi, -1.0, zp1);
  z_map<0>(y, mu, phi, 1.0, zp2);
  const double Av = math::log_Phi(-zp1[0]), Bv = 2.0 / (mu * phi) + math::log_Phi(-zp2[0]);
  double x = 0.0, L0 = 0.0, S = 0.0;
  survival_x_L0(y, phi, zp1[0], zp2[0], Bv, Av, x, L0, S);
  if (L0 < -0.69314718055994530942) return Kokkos::log1p(-S);
  const double A = math::log_Phi(zp1[0]);
  const double M = A > Bv ? A : Bv, m = A > Bv ? Bv : A;
  return M + Kokkos::log1p(math::exp(m - M));
}

}  // namespace detail_invgauss

/**
 * \ingroup distrib_invgauss
 * \brief Univariate inverse Gaussian distribution parameterized by mean \f$\mu > 0\f$ and dispersion \f$\phi > 0\f$.
 *
 * Implements the continuous density with variance \f$\phi\mu^3\f$ and its parameter derivatives to order 6 in closed
 * form, the expected Fisher information \f$\operatorname{diag}(1/(\phi\mu^3), 1/(2\phi^2))\f$ and its derivatives, and
 * the cumulative distribution function \f$\Phi(z_1) + e^{2/(\phi\mu)}\Phi(-z_2)\f$ on the log scale. The quantile is the
 * generic route.
 * \see distrib_invgauss, distrib_core, logistic_distrib
 */
struct invgauss_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Dispersion parameter \f$\phi \in (0, \infty)\f$ (variance \f$\phi\mu^3\f$).
  double phi;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Differentiability order of log f in y (\c infinitely_smooth on (0, inf)).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c phi: both \c C^inf, neither moves the support.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"phi", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates the log-density \f$\log f(y; \mu, \phi)\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Log-density value, \f$-\infty\f$ for \f$y \le 0\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    const double w = (y - mu) / mu;  // one subtraction, then the division
    return -0.91893853320467274178 - 0.5 * math::log(phi) - 1.5 * math::log(y) - 0.5 * w * w / (phi * y);
  }

  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \phi)\f$ in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    // gd[a] = d^a g / dmu^a with g = (y - mu)^2 / (mu^2 y)
    const double w = (y - mu) / mu;
    double gd[K + 1];
    gd[0] = w * w / y;
    double pw = 1.0 / (mu * mu);  // mu^-(a+2) at a = 0, then updated
    for (int a = 1; a <= K; ++a) {
      pw /= mu;
      gd[a] = detail_invgauss::sgn(a) * detail_logistic::fact(a) * pw * ((a + 1) * y - 2.0 * mu);
    }
    double pinv[K + 2];  // phi^-k
    pinv[0] = 1.0;
    for (int k = 1; k <= K + 1; ++k) pinv[k] = pinv[k - 1] / phi;
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      double v = -0.5 * gd[a] * detail_invgauss::sgn(b) * detail_logistic::fact(b) * pinv[b + 1];
      if (a == 0 && b >= 1) v += 0.5 * detail_invgauss::sgn(b) * detail_logistic::fact(b - 1) * pinv[b];
      t[b] = v;
    }
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \phi)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates mixed partial derivatives of \f$\log f(y; \mu, \phi)\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * The terms of \f$\ell\f$ that depend on \f$y\f$ are \f$-\frac{3}{2}\log y\f$, \f$-\frac{y}{2\phi\mu^2}\f$ and \f$-\frac{1}{2\phi y}\f$.
   * With \f$a\f$ derivatives in \f$\mu\f$, \f$b\f$ in \f$\phi\f$ and \f$a + b = K\f$, the component is
   * \f$[a = b = 0]\,(-\tfrac{3}{2})(-1)^{J-1}(J-1)!\,y^{-J}\f$, plus \f$[J = 1]\,(-\tfrac{1}{2})(-1)^a (a+1)!\,\mu^{-(a+2)}(-1)^b b!\,\phi^{-(b+1)}\f$,
   * plus \f$[a = 0]\,(-\tfrac{1}{2})(-1)^J J!\,y^{-(J+1)}(-1)^b b!\,\phi^{-(b+1)}\f$.
   * \tparam J Derivative order with respect to \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to parameters \f$(\mu, \phi)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else {
      double pinv[K + 2], imu[K + 3];  // phi^-k and mu^-k
      pinv[0] = 1.0;
      imu[0] = 1.0;
      for (int k = 1; k <= K + 2; ++k) {
        if (k <= K + 1) pinv[k] = pinv[k - 1] / phi;
        imu[k] = imu[k - 1] / mu;
      }
      double yj = 1.0;  // y^-J
      for (int k = 0; k < J; ++k) yj /= y;
      for (int b = 0; b <= K; ++b) {
        const int a = K - b;
        double v = 0.0;
        if (a == 0 && b == 0) v += -1.5 * detail_invgauss::sgn(J - 1) * detail_logistic::fact(J - 1) * yj;
        if constexpr (J == 1) {
          if (a == 0) {
            // the two a = 0 terms, -(1/2)(1/mu^2 - 1/y^2) sgn(b) b! phi^-(b+1), combined as (y - mu)(y + mu)/(mu^2 y^2)
            // so that they do not cancel for y near mu
            v += -0.5 * detail_invgauss::sgn(b) * detail_logistic::fact(b) * pinv[b + 1] * ((y - mu) * (y + mu)) /
                 (mu * mu * y * y);
          } else {
            // mu^-(a+2) = imu[a+2]
            v += -0.5 * detail_invgauss::sgn(a) * detail_logistic::fact(a + 1) * imu[a + 2] *
                 detail_invgauss::sgn(b) * detail_logistic::fact(b) * pinv[b + 1];
          }
        } else if (a == 0) {
          // y^-(J+1) = y^-J / y
          v += -0.5 * detail_invgauss::sgn(J) * detail_logistic::fact(J) * (yj / y) * detail_invgauss::sgn(b) *
               detail_logistic::fact(b) * pinv[b + 1];
        }
        t[b] = v;
      }
      return t;
    }
  }

  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \phi) = \operatorname{diag}(1/(\phi\mu^3), 1/(2\phi^2))\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{\mu\mu}\f$ at index 0 and \f$I_{\phi\phi}\f$ at index 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    sym_tensor<2, 2> t;
    t[0] = 1.0 / (phi * mu * mu * mu);
    t[2] = 0.5 / (phi * phi);
    return t;
  }
  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \phi)\f$.
   *
   * \f$I_{\mu\mu} = \phi^{-1}\mu^{-3}\f$ has the derivative \f$(-1)^K \frac{(a+2)!}{2}\, b!\, \mu^{-(3+a)}\phi^{-(1+b)}\f$ at position
   * \f$b\f$ (\f$a = K - b\f$ derivatives in \f$\mu\f$). \f$I_{\phi\phi} = \frac{1}{2}\phi^{-2}\f$ depends on \f$\phi\f$ only, with
   * the derivative \f$(-1)^K \frac{(K+1)!}{2}\phi^{-(K+2)}\f$ at position \f$K\f$. \f$I_{\mu\phi} = 0\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    expected_info_tensor<2, K> t;
    double pinv[K + 3], imu[K + 1];
    pinv[0] = 1.0;
    for (int k = 1; k <= K + 2; ++k) pinv[k] = pinv[k - 1] / phi;
    imu[0] = 1.0 / (mu * mu * mu);  // mu^-(3 + k)
    for (int k = 1; k <= K; ++k) imu[k] = imu[k - 1] / mu;
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      t.pair[0][b] = detail_invgauss::sgn(K) * 0.5 * detail_logistic::fact(a + 2) * detail_logistic::fact(b) * imu[a] *
                     pinv[b + 1];
    }
    t.pair[2][K] = detail_invgauss::sgn(K) * 0.5 * detail_logistic::fact(K + 1) * pinv[K + 2];
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \phi)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \phi)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates the cumulative distribution function \f$F(y) = \Phi(z_1) + e^{2/(\phi\mu)}\Phi(-z_2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : math::exp(log_cdf(y)); }
  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : math::exp(log_ccdf(y)); }
  /**
   * \ingroup distrib_invgauss
   * rief Evaluates the log of the cdf: the log-sum-exp of log Phi(z_1) and 2/(phi mu) + log Phi(-z_2), or
   * log(1 - e^(log(1 - F))) where F > 1/2 (then log(1 - F) is the accurate quantity).
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y <= 0.0 ? -distrib_infinity : detail_invgauss::log_cdf_value(y, mu, phi);
  }
  /**
   * \ingroup distrib_invgauss
   * rief Evaluates log(1 - F(y)) as log Phi(-z_1) + log(1 - e^(B - A')), with B - A' from the erfcx
   * ratio of the survival (no cancellation of the z^2/2 terms).
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : detail_invgauss::log_ccdf_value(y, mu, phi); }
  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$ (zero tensor for \f$y \le 0\f$).
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    return detail_invgauss::log_cdf_dk<K>(false, y, mu, phi);
  }
  /**
   * \ingroup distrib_invgauss
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$ (zero tensor for \f$y \le 0\f$).
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    return detail_invgauss::log_cdf_dk<K>(true, y, mu, phi);
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of the support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of the support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Rough location: the mean \f$\mu\f$.
  LAPLACE_HD double center() const { return mu; }
  /// Rough spread: the standard deviation \f$\sqrt{\phi\mu^3}\f$.
  LAPLACE_HD double scale() const { return Kokkos::sqrt(phi * mu * mu * mu); }
  /// No non-smooth points in either parameter.
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// No non-smooth points in y.
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
};

}  // namespace laplace::distrib
