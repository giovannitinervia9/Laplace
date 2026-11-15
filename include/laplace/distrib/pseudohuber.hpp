// pseudohuber_distrib (laplace::distrib): the pseudo-Huber distribution in its location mu, its
// standard deviation sigma > 0 and its shape nu > 0. With z = (y - mu)/sigma and x = sqrt(nu),
//
//   f(y) = sqrt(R) / (2 sigma x K_1(x)) exp(-sqrt(nu + R z^2)),   R(nu) = x K_2(x) / K_1(x),
//
// so that sigma carries the whole variance (Var Y = sigma^2 for every nu) and nu only the tails:
// the Laplace law as nu -> 0 (R -> 2) and the Gaussian as nu -> infinity (R ~ sqrt(nu) + 3/2).
// Every parameter is C^inf and none moves the support: no kinks.
//
//   l = Lambda(z, nu) + h(nu) - log 2 - log sigma,
//   Lambda = -sqrt(nu + R z^2),   h = (1/2) log R - (1/2) log nu - log K_1(x).
//
// The Bessel ratio. K_nu'(x) = -K_(nu+1)(x) + (nu/x) K_nu(x) at nu = 1 gives (log K_1)' =
// -K_2/K_1 + 1/x, that is the exact identity
//
//   R = 1 - x g'(x),   g = log K_1,   d^k R / dx^k = -(x g^(k+1)(x) + k g^(k)(x)),  k >= 1,
//
// whose two terms have the same sign (g' is about -1 - 1/(2 x)), so no digit is lost to the
// ratio K_2/K_1 = 1 + 3/(2x) + ...; g^(k) = log_bessel_k_dx<k>(1, x), orders 1 to 7. The nu-derivatives
// of R follow by one Faa di Bruno composition per order through x = nu^(1/2).
//
// The derivatives of h in nu: (1/2) log R through R's derivatives, -(1/2) log nu in closed form,
// and -log K_1(sqrt nu) through x; so one set of seven Bessel derivatives serves the whole tensor
// (log K_2 is never formed: log R - log x = log(K_2/K_1) is the same function), and it comes from one
// call of math::log_bessel_k_dx_upto, which shares Steed's continued fraction between the orders.
//
// For nu < small_nu_split = 1 the Bessel route loses about eps/nu (the derivative of order k of R in x is the
// difference of terms of size x^-k that cancels to size x^(2-k)), and R and h come instead from the
// small-argument series of K_0 and K_1 (DLMF 10.31.1) in powers of nu and log nu, with no Bessel function
// and no cancellation: see small_nu_derivs below, where the series and the choice of the split are described.
//
// The derivatives of Lambda in (z, nu). Writing R = rho^2, q = nu / R,
//
//   Lambda(z, nu) = -rho(nu) G(q(nu), z),   G(q, z) = sqrt(q + z^2),
//
// the derivatives of G are closed: d^i/dq^i G = fall(1/2, i) (q + z^2)^(1/2 - i), and
// H_p = (q + z^2)^p obeys (q + z^2) H_p^(m+1) = 2 (p - m) z H_p^(m) + m (2 p - m + 1) H_p^(m-1)
// in z (from (q + z^2) H_p' = 2 p z H_p, differentiated m times). The pure z-derivatives of G
// (i = 0, p = 1/2) cancel at large |z| in that recurrence (fall(1, m) = 0 for m >= 2); they use
// the exact form H_(1/2)'' = q H_(-3/2) instead, so H_(1/2)^(m) = q H_(-3/2)^(m-2) for m >= 2,
// whose terms do not cancel. With Faa di Bruno through q(nu) and Leibniz against rho,
//
//   d^c/dnu^c d^j/dz^j Lambda = - sum_(a=0)^c C(c, a) rho^(c-a) Gamma_a^(j),
//   Gamma_a^(j) = sum_(i=1)^a [d^i/dq^i d^j/dz^j G] B_(a,i)(q', ..., q^(a-i+1)),  Gamma_0 = G,
//
// with the nu-derivatives of rho = R^(1/2) and of q = nu / R from those of R (Faa di Bruno through
// the powers of R, which has no sign change in its terms). The table of these derivatives is handed
// to detail_location_scale::mixed_shape_dk, which carries it to (mu, sigma, nu). The derivation is
// checked against SymPy in tools/reference/distrib_pseudohuber.py.
//
// The expected information, the cdf, the quantile and the draw have no closed form here: the
// generic routes apply (generic_info.hpp, generic_cdf.hpp, generic_quantile.hpp, generic_rng.hpp).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/distrib/generic_start.hpp>
#include <laplace/math/bessel_k.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_pseudohuber {

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double sign_of_power(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// position in sym_tensor<3, a + b + c> of the sorted multi-index with a zeros, b ones and c twos
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

// d^n/dnu^n f(g(nu)) for n = 1..6 with f[1..n], g[1..n]
LAPLACE_HD_FORCEINLINE double chain(int n, const double* f, const double* g) {
  switch (n) {
    case 1: return math::chain_rule_d1(f, g);
    case 2: return math::chain_rule_d2(f, g);
    case 3: return math::chain_rule_d3(f, g);
    case 4: return math::chain_rule_d4(f, g);
    case 5: return math::chain_rule_d5(f, g);
    default: return math::chain_rule_d6(f, g);
  }
}

// the falling factorial (1/2)(1/2 - 1) ... (1/2 - i + 1)
LAPLACE_HD_FORCEINLINE double half_fall(int i) {
  double r = 1.0;
  for (int k = 0; k < i; ++k) r *= 0.5 - k;
  return r;
}

// the nu-derivatives of the Bessel ratio R(nu), orders 0..K, by the exact identity of the header
// comment and Faa di Bruno through x = sqrt(nu); xd[j] = d^j x / dnu^j, j = 0..K, is also returned,
// and gx[k] = (log K_1)^(k)(x), k = 1..K + 1, with gx[0] = log K_1(x), from one evaluation of the
// shared work of the Bessel routine
template <int K>
LAPLACE_HD void ratio_derivs(double nu, double* Rn, double* xd, double* gx) {
  const double x = Kokkos::sqrt(nu);
  math::log_bessel_k_dx_upto<K + 1>(1.0, x, gx);
  double Rx[K + 1];
  Rx[0] = 1.0 - x * gx[1];
  for (int k = 1; k <= K; ++k) Rx[k] = -(x * gx[k + 1] + k * gx[k]);
  xd[0] = x;
  for (int j = 1; j <= K; ++j) xd[j] = xd[j - 1] * (0.5 - (j - 1)) / nu;
  Rn[0] = Rx[0];
  for (int k = 1; k <= K; ++k) Rn[k] = chain(k, Rx, xd);
}

// the nu-derivatives of h(nu) = (1/2) log R - (1/2) log nu - log K_1(sqrt nu), orders 1..K, from those of R
// (Rn), of x = sqrt(nu) (xd) and of log K_1 (gx, with gx[0] = log K_1(x)) already formed by ratio_derivs.
// hd[0] is the value.
template <int K>
LAPLACE_HD void h_derivs(double nu, const double* Rn, const double* xd, const double* gx, double* hd) {
  const double R = Rn[0];
  hd[0] = 0.5 * math::log(R) - 0.5 * math::log(nu) - gx[0];
  double f1[K + 1], f2[K + 1];  // (1/2) log R and -log K_1 as functions of their arguments
  double ir = 1.0;
  for (int k = 1; k <= K; ++k) {
    ir /= R;
    f1[k] = 0.5 * sign_of_power(k - 1) * fact(k - 1) * ir;
    f2[k] = -gx[k];
  }
  for (int k = 1; k <= K; ++k) {
    double inu = 1.0;
    for (int i = 0; i < k; ++i) inu /= nu;
    hd[k] = chain(k, f1, Rn) + chain(k, f2, xd) - 0.5 * sign_of_power(k - 1) * fact(k - 1) * inu;
  }
}

// ---------------------------------------------------------------- the series for small nu
//
// With x = sqrt(nu), u = nu/4, L = log(x/2), gamma Euler's constant and H_k the harmonic numbers, the
// small-argument series of K_0 and K_1 (DLMF 10.31.1) read
//   K_0 = -(L + gamma) P0 + Hs,        x K_1 = B = 1 + (1/2) L J1 - W1,
//   P0 = sum_k u^k/(k!)^2,   Hs = sum_(k>=1) H_k u^k/(k!)^2,
//   J1 = sum_k nu^(k+1)/(4^k k! (k+1)!),   W1 = sum_k (H_k + H_(k+1) - 2 gamma) nu^(k+1)/(4^(k+1) k! (k+1)!),
// and since K_2 = K_0 + (2/x) K_1,
//   R = 2 + nu K_0/B = 2 + nu A/B,   A = K_0,   h = (1/2) log R - log B
// (log B = log K_1 + (1/2) log nu, so the pole of log K_1 and the logarithm of nu are gone: this is the form
// of h in which nothing cancels at small nu). A and B are polynomials in nu, and in L, which
// has the derivatives L' = 1/(2 nu), L^(j) = (1/2)(-1)^(j-1)(j-1)!/nu^j. The derivatives of A and B follow
// by Leibniz, those of 1/B and of the two logarithms by the Faa di Bruno functions, those of A/B by Leibniz
// again; the route needs no Bessel function and does not cancel (the terms of A^(i) and B^(i) of size
// nu^-i are the leading ones). The tables are checked against SymPy's series of K_0 and x K_1 and the whole
// derivation against the definition in tools/reference/distrib_pseudohuber.py (small).
// small_nu_terms terms hold to 1e-22 relative at nu = 1 and to 1e-15 at nu = 3.5.
constexpr int small_nu_terms = 14;
// The series is used for nu below small_nu_split = 1. The two routes were compared with the definition (the
// series jets of tools/reference/distrib_pseudohuber.py, rder) on 32 values of nu from 1e-6 to 56: the series
// is right to 2e-15 relative in R, h and every derivative to order six from 1e-6 to 1 (nothing grows with
// 1/nu), and loses digits above (4e-15 at 1.78, 3e-14 at 3.2, 1e-8 at 10: the truncation of the 14 terms),
// while the Bessel route is right to 4e-16 in R from nu = 0.3 on but loses about eps/nu below
// (at order six 3e-15 at 0.1, 1e-14 at 1e-2, 2e-12 at 1e-4, 4e-10 at 1e-6). Between 0.3 and 1 the two have about the
// same worst error (1e-15 to 2e-15 in h at order six), and the series is the cheaper (no Bessel function), so the split
// is where the series is still as good as the other route, nu = 1.
constexpr double small_nu_split = 1.0;

// the coefficient tables in nu of P0, Hs, J1 and W1
LAPLACE_HD void small_nu_coefs(double* p0, double* hs, double* j1, double* w1) {
  constexpr double gam = 0.57721566490153286061;
  double Hk = 0.0, c = 1.0, d = 1.0;  // H_k, 1/(4^k (k!)^2), 1/(4^k k! (k+1)!)
  j1[0] = 0.0;
  w1[0] = 0.0;
  for (int k = 0; k < small_nu_terms; ++k) {
    if (k > 0) {
      Hk += 1.0 / k;
      c /= 4.0 * k * k;
      d /= 4.0 * k * (k + 1.0);
    }
    p0[k] = c;
    hs[k] = Hk * c;
    if (k + 1 < small_nu_terms) {
      j1[k + 1] = d;
      w1[k + 1] = (Hk + (Hk + 1.0 / (k + 1.0)) - 2.0 * gam) * d / 4.0;
    }
  }
}

// d^i/dnu^i of sum_k c[k] nu^k
LAPLACE_HD_FORCEINLINE double small_poly(const double* c, double nu, int i) {
  double r = 0.0;
  for (int k = small_nu_terms - 1; k >= i; --k) {
    double f = 1.0;
    for (int m = 0; m < i; ++m) f *= (k - m);
    r = r * nu + c[k] * f;
  }
  return r;
}

// the nu-derivatives of R, Rn[0..K], and of h, hd[0..K], from the series
template <int K>
LAPLACE_HD void small_nu_derivs(double nu, double* Rn, double* hd) {
  constexpr double gam = 0.57721566490153286061;
  double p0[small_nu_terms], hs[small_nu_terms], j1[small_nu_terms], w1[small_nu_terms];
  small_nu_coefs(p0, hs, j1, w1);
  double L[K + 1];  // L and its derivatives
  L[0] = 0.5 * math::log(nu) - 0.6931471805599453094;
  {
    double inu = 1.0;
    for (int j = 1; j <= K; ++j) {
      inu /= nu;
      L[j] = 0.5 * sign_of_power(j - 1) * fact(j - 1) * inu;
    }
  }
  double P[K + 1], J[K + 1];  // the derivatives of P0 and J1
  for (int i = 0; i <= K; ++i) {
    P[i] = small_poly(p0, nu, i);
    J[i] = small_poly(j1, nu, i);
  }
  double A[K + 1], Bd[K + 1];
  for (int i = 0; i <= K; ++i) {
    double a = -(L[0] + gam) * P[i] + small_poly(hs, nu, i);
    double b = (i == 0 ? 1.0 : 0.0) - small_poly(w1, nu, i);
    for (int j = 1; j <= i; ++j) a -= binomial_coef(i, j) * L[j] * P[i - j];
    for (int j = 0; j <= i; ++j) b += 0.5 * binomial_coef(i, j) * L[j] * J[i - j];
    A[i] = a;
    Bd[i] = b;
  }
  const double B0 = Bd[0];
  double fW[K + 1], W[K + 1], Q[K + 1];  // 1/B, its derivatives, and A/B
  {
    double ib = 1.0 / B0;
    W[0] = ib;
    for (int j = 1; j <= K; ++j) {
      ib /= B0;
      fW[j] = sign_of_power(j) * fact(j) * ib;
    }
    for (int m = 1; m <= K; ++m) W[m] = chain(m, fW, Bd);
  }
  for (int k = 0; k <= K; ++k) {
    double q = 0.0;
    for (int i = 0; i <= k; ++i) q += binomial_coef(k, i) * A[i] * W[k - i];
    Q[k] = q;
  }
  Rn[0] = 2.0 + nu * Q[0];
  for (int k = 1; k <= K; ++k) Rn[k] = nu * Q[k] + k * Q[k - 1];
  // h = (1/2) log R - log B
  hd[0] = 0.5 * math::log(Rn[0]) - math::log(B0);
  if constexpr (K >= 1) {
    double f1[K + 1], f2[K + 1];
    double ir = 1.0, ib = 1.0;
    for (int j = 1; j <= K; ++j) {
      ir /= Rn[0];
      ib /= B0;
      f1[j] = 0.5 * sign_of_power(j - 1) * fact(j - 1) * ir;
      f2[j] = sign_of_power(j - 1) * fact(j - 1) * ib;
    }
    for (int k = 1; k <= K; ++k) hd[k] = chain(k, f1, Rn) - chain(k, f2, Bd);
  }
}

// D[c * (K + 1) + j] = d^c/dnu^c d^j/dz^j Lambda(z, nu), c + j <= K, Lambda = -sqrt(nu + R z^2)
// (the entries with c + j > K are not written, and j = 0 only at c = K, which mixed_shape_dk reads)
template <int K>
LAPLACE_HD void lambda_table(double z, double nu, const double* Rn, double* D) {
  const double R = Rn[0];
  // rho = R^(1/2) and u = 1/R, their nu-derivatives, and q = nu u
  double rho[K + 1], u[K + 1], qd[K + 1], fr[K + 1], fu[K + 1];
  double pw = 1.0;  // R^(-k)
  rho[0] = Kokkos::sqrt(R);
  u[0] = 1.0 / R;
  for (int k = 1; k <= K; ++k) {
    pw /= R;
    fr[k] = half_fall(k) * rho[0] * pw;
    fu[k] = sign_of_power(k) * fact(k) * u[0] * pw;
  }
  for (int k = 1; k <= K; ++k) {
    rho[k] = chain(k, fr, Rn);
    u[k] = chain(k, fu, Rn);
  }
  qd[0] = nu * u[0];
  for (int k = 1; k <= K; ++k) qd[k] = nu * u[k] + k * u[k - 1];
  const double q = qd[0];
  // One table G[i][j] serves three stages (the stack of a GPU thread is small, and three tables of
  // (K + 1)^2 doubles did not fit): first the raw H[i][m] = d^m/dz^m (q + z^2)^(1/2 - i), i = 1..K,
  // m = 0..K - i, in rows 1..K; row 0 gets the derivatives of G itself, d^j/dz^j G, from the raw rows
  // (j >= 2: q H[2][j - 2]); then rows i >= 1 are scaled to d^i/dq^i d^j/dz^j G = fall(1/2, i) H[i][j];
  // last, in each column j, Gamma[a][j] = d^a/dnu^a d^j/dz^j G(q(nu), z) replaces row a, a = K .. 1 (it
  // reads the rows i <= a of its own column only, and those below a are still unscaled-by-Bell).
  const double a2 = q + z * z, s = Kokkos::sqrt(a2), ps = 1.0 / s, ia2 = 1.0 / a2;
  double G[K + 1][K + 1];
  double psp = ps;  // s^-(2 i - 1)
  for (int i = 1; i <= K; ++i, psp *= ia2) {
    const double p = 0.5 - i;
    G[i][0] = psp;
    for (int m = 0; m + 1 <= K - i; ++m) {
      const double lower = m > 0 ? G[i][m - 1] : 0.0;
      G[i][m + 1] = (2.0 * (p - m) * z * G[i][m] + m * (2.0 * p - m + 1.0) * lower) * ia2;
    }
  }
  G[0][0] = s;
  G[0][1] = z * ps;
  for (int j = 2; j <= K; ++j) G[0][j] = q * G[2][j - 2];
  for (int i = 1; i <= K; ++i) {
    const double ff = half_fall(i);
    for (int j = 0; i + j <= K; ++j) G[i][j] *= ff;
  }
  for (int a = K; a >= 1; --a)
    for (int j = 0; a + j <= K; ++j) {
      double acc = 0.0;
      for (int i = 1; i <= a; ++i) acc += G[i][j] * math::bell_partial(a, i, qd);
      G[a][j] = acc;
    }
  for (int c = 0; c <= K; ++c)
    for (int j = (c == K ? 0 : 1); c + j <= K; ++j) {
      double acc = 0.0;
      for (int a = 0; a <= c; ++a) acc += binomial_coef(c, a) * rho[c - a] * G[a][j];
      D[c * (K + 1) + j] = -acc;
    }
}

}  // namespace detail_pseudohuber

/**
 * \brief The pseudo-Huber distribution in its location \f$\mu\f$, standard deviation \f$\sigma\f$ and shape \f$\nu\f$.
 */
struct pseudohuber_distrib {
  /// Location parameter mu (the mean).
  double mu;
  /// Standard deviation sigma > 0.
  double sigma;
  /// Shape parameter nu > 0 (the Laplace law as nu tends to 0, the Gaussian as it tends to infinity).
  double nu;

  static constexpr distrib_kind kind = distrib_kind::continuous;
  static constexpr int n_params = 3;
  static constexpr int y_regularity = infinitely_smooth;
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"nu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// The Bessel ratio R(nu) = sqrt(nu) K_2(sqrt nu)/K_1(sqrt nu) = 1 - x (log K_1)'(x), x = sqrt(nu); the series in nu for small nu.
  LAPLACE_HD double bessel_ratio() const {
    if (nu < detail_pseudohuber::small_nu_split) {
      double Rn[1], hd[1];
      detail_pseudohuber::small_nu_derivs<0>(nu, Rn, hd);
      return Rn[0];
    }
    const double x = Kokkos::sqrt(nu);
    return 1.0 - x * math::log_bessel_k_dx1(1.0, x);
  }

  /// log f(y) = -sqrt(nu + R z^2) + (1/2) log R - (1/2) log nu - log K_1(sqrt nu) - log 2 - log sigma.
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    double R, h;
    if (nu < detail_pseudohuber::small_nu_split) {
      double Rn[1], hd[1];
      detail_pseudohuber::small_nu_derivs<0>(nu, Rn, hd);
      R = Rn[0];
      h = hd[0];
    } else {
      R = bessel_ratio();
      h = 0.5 * math::log(R) - 0.5 * math::log(nu) - math::log_bessel_k(1.0, Kokkos::sqrt(nu));
    }
    return -Kokkos::sqrt(nu + R * z * z) + h - math::log(2.0 * sigma);
  }

  /// The derivatives of order K of log f in (mu, sigma, nu).
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    // R and h: the series below small_nu_split, one set of Bessel derivatives above it
    double Rn[K + 1], D[(K + 1) * (K + 1)], hd[K + 1];
    if (nu < detail_pseudohuber::small_nu_split) {
      detail_pseudohuber::small_nu_derivs<K>(nu, Rn, hd);
    } else {
      double xd[K + 1], gx[K + 2];
      detail_pseudohuber::ratio_derivs<K>(nu, Rn, xd, gx);
      detail_pseudohuber::h_derivs<K>(nu, Rn, xd, gx, hd);
    }
    detail_pseudohuber::lambda_table<K>(z, nu, Rn, D);
    sym_tensor<3, K> t = detail_location_scale::mixed_shape_dk<K>(D, z, sigma);
    t[detail_pseudohuber::pos3(0, 0, K)] += hd[K];
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[detail_pseudohuber::pos3(0, K, 0)] += detail_pseudohuber::sign_of_power(K) * detail_pseudohuber::fact(K - 1) * isk;
    return t;
  }
  /// First-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives of log f.
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  // l depends on y and mu only through y - mu, so d/dy = -d/dmu: the component of logpdf_dk<J + K>
  // with J more indices in mu, times (-1)^J.
  /// J derivatives in y and the tensor of order K in (mu, sigma, nu).
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
      const double s = detail_pseudohuber::sign_of_power(J);
      for (int b = 0; b <= K; ++b)
        for (int c = 0; b + c <= K; ++c)
          r[detail_pseudohuber::pos3(K - b - c, b, c)] = s * t[detail_pseudohuber::pos3(K - b - c + J, b, c)];
      return r;
    }
  }

  /// Lower bound of the support.
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of the support.
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (mu).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (sigma, the standard deviation).
  LAPLACE_HD double scale() const { return sigma; }
  /// No non-smooth points in any parameter.
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// No non-smooth points in y.
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
  // Starting value from a sample (generic_start.hpp): the sample mean and standard deviation, and nu = 1
  // (between the two limits); strictly inside the domain.
  pseudohuber_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.mean, params[0]), start_inside(Kokkos::sqrt(s.variance), params[1]),
            start_inside(1.0, params[2])};
  }
};

}  // namespace laplace::distrib
