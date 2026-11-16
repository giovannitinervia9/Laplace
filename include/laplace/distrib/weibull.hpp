// weibull_distrib (laplace::distrib): the Weibull distribution in its mean m > 0 and shape
// sigma > 0. The textbook form has shape k and scale b,
// density (k/b)(y/b)^(k-1) exp(-(y/b)^k) and mean b Gamma(1 + 1/k); here
//
//   k = sigma,   log b = log m - lgamma(1 + 1/sigma),
//
// so that E[Y] = m. The scale b is never formed as a number (at small sigma it underflows
// while log b stays finite). With x = 1/sigma and
//
//   phi(x) = lgamma(1 + x)/x = sigma LG,   LG = lgamma(1 + 1/sigma),
//   C = log(y/m),   h = sigma C + phi(x) = sigma log(y/b),   T = e^h = (y/b)^sigma,
//
// the log-density is  l = log sigma - log y + h - T,  and T is a standard exponential variable
// under the model for every (m, sigma): that is why the expected information is closed and why
// the survival function is e^(-T), log(1 - F) = -T exactly.
//
// Derivatives. For a multi-index I (a indices in m, b in sigma, j in y),
//   d^I l = [log sigma and -log y terms] + d^I h - e^h R_I,   R_I = e^-h d^I e^h,
// R_I being the complete Bell polynomial of the partial derivatives of h over the set
// partitions of I. It is computed by the recurrence (Leibniz on d e^h = (d h) e^h)
//   R_(I + e) = sum_(J <= I) C(I, J) h_(e + J) R_(I - J),
// with no table of transcribed polynomials (exp_table below). The partials of h are
//   m^a:           sigma (-1)^a (a - 1)!/m^a         (a >= 1)
//   m^a sigma:     (-1)^a (a - 1)!/m^a               (a >= 1)
//   sigma^b:       C [b = 1] + d^b phi(1/sigma)/dsigma^b
//   y^j:           sigma (-1)^(j-1) (j - 1)!/y^j     y^j sigma: (-1)^(j-1) (j - 1)!/y^j
// and zero for two or more sigma together with any m or y, and for m together with y.
//
// Derivatives of phi in sigma. phi(1/sigma) is composed by Faa di Bruno from the derivatives
// phi^(k)(x), whose route depends on x (measured against mpmath, see the tests):
//   x <= 1/4 (sigma >= 4): the series phi(x) = -gamma + sum_(j >= 1) (-1)^(j+1) zeta(j + 1) x^j/(j + 1),
//     differentiated term by term. The direct form lgamma(1 + x)/x is not used there: 1 + x
//     rounds and lgamma(1 + x) ~ -gamma x, so the relative error of phi would be eps/x;
//   1/4 < x < 2: Leibniz' rule on lgamma(1 + x) and 1/x, with polygamma(k - 1, 1 + x);
//   x >= 2 (sigma <= 1/2): lgamma(1 + x) = (x + 1/2) log x - x + (1/2) log(2 pi) + delta(x), delta
//     Stirling's error, so phi = log x - 1 + q(x)/x with q = (1/2) log(2 pi x) + delta. The log x is
//     differentiated exactly, and only q/x goes through the polygamma-free Leibniz sum, in which
//     nothing cancels (Leibniz on lgamma(1 + x) loses about log x per order there).
// The pure sigma part of the log-density is log sigma + phi(1/sigma) = -1 + q(x)/x at x >= 2, which
// has no log x left to cancel against log sigma, so it is composed separately (P below).
//
// cdf, both tails. F = -expm1(-T), 1 - F = e^-T, log(1 - F) = -T with derivatives -T R_I, the
// same Bell machinery (not derivatives of F divided by 1 - F). log F = h + log exprel(-T) for T < 1
// (T/exprel(-T) is the usual 1 - e^-T, h = log T is the exact pole), log1p(-e^-T) for T >= 1.
// Derivatives of log F: the multivariate Faa di Bruno sum S_k(I) = d^I [G^(k)(T)] of the derivatives
// of the outer function in T (copied from the exponential's two routes: Eulerian polynomials of
// e^-T for T >= 1; log exprel(-T) plus the exact derivatives of h for T < 1), by the recurrence
//   S_k(I + e) = sum_(J <= I) C(I, J) S_(k+1)(J) T_(e + I - J),   T_I = e^h R_I,
// each term of the T < 1 sum carrying a factor T so that the pole of the density does not
// cancel by subtraction.
//
// Quantile: log Q = log b + log(-log1p(-p))/sigma, never forming 1 - p; the log p and log(1 - p)
// forms are quantile_from_log and quantile_from_log_upper.
//
// Expected information (T standard exponential, E[T^j (log T)^k] = Gamma^(k)(j + 1)):
//   I_mm = sigma^2/m^2,  I_m,sigma = D/m,  I_sigma,sigma = (D^2 + pi^2/6)/sigma^2,
//   D = psi(1 + 1/sigma) - (1 - gamma),
// with D written as that single difference (near sigma = 1, where D passes through zero and the
// two parameters are orthogonal, it is the series psi(2 + d) - psi(2) = sum_k (-1)^(k+1)
// (zeta(k + 1) - 1) d^k, d = 1/sigma - 1, which keeps D to full relative accuracy).
//
// y = 0: the derivative of the log-density in y exists there only at sigma = 1 (it is -1/m, the
// exponential density at its scale m); at every other sigma it is infinite. logpdf_dy_dk
// returns +infinity for a component at y <= 0 (and the finite exponential values for the pure m
// components at sigma == 1); the derivatives in the parameters at y <= 0 are NaN.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace laplace::distrib {

namespace detail_weibull {

constexpr double euler_gamma = 0.57721566490153286061;
constexpr double one_minus_gamma = 0.42278433509846713939;
constexpr double pi2_over_6 = 1.6449340668482264365;
constexpr double log_2pi = 1.8378770664093454836;

constexpr double x_series = 0.25;     // phi by its series in x = 1/sigma up to here
constexpr double x_stirling = 2.0;    // phi through Stirling's error from here
constexpr int zeta_terms = 63;        // zeta(k + 1) - 1, k = 1..63

// zeta(k + 1) - 1 for k = 1..63, from mpmath at 200 bits (tools/reference/distrib_weibull.py checks
// the series they feed against mpmath). The difference from 1 is stored, not zeta itself, so that
// the small values keep their relative accuracy.
#define LAPLACE_WEIBULL_ZM1 \
      0.6449340668482264365, 0.2020569031595942854, 0.08232323371113819152, 0.03692775514336992633, \
      0.01734306198444913971, 0.00834927738192282684, 0.004077356197944339379, 0.002008392826082214418, \
      0.0009945751278180853371, 0.0004941886041194645587, 0.0002460865533080482986, 0.0001227133475784891468, \
      0.00006124813505870482926, 0.00003058823630702049355, 0.00001528225940865187173, 7.637197637899762274e-6, \
      3.817293264999839856e-6, 1.908212716553938926e-6, 9.539620338727961132e-7, 4.769329867878064631e-7, \
      2.3845050272773299e-7, 1.192199259653110731e-7, 5.960818905125947961e-8, 2.980350351465228019e-8, \
      1.490155482836504123e-8, 7.450711789835429492e-9, 3.725334024788457055e-9, 1.862659723513049006e-9, \
      9.313274324196681829e-10, 4.656629065033784073e-10, 2.328311833676505492e-10, 1.164155017270051978e-10, \
      5.820772087902700889e-11, 2.910385044497099687e-11, 1.455192189104198424e-11, 7.275959835057481015e-12, \
      3.63797954737865119e-12, 1.818989650307065948e-12, 9.094947840263889283e-13, 4.547473783042154027e-13, \
      2.273736845824652515e-13, 1.136868407680227849e-13, 5.684341987627585609e-14, 2.842170976889301855e-14, \
      1.421085482803160677e-14, 7.105427395210852713e-15, 3.552713691337113673e-15, 1.776356843579120327e-15, \
      8.881784210930815903e-16, 4.440892103143813364e-16, 2.220446050798041984e-16, 1.110223025141066134e-16, \
      5.551115124845481244e-17, 2.775557562136124173e-17, 1.387778780972523276e-17, 6.938893904544153697e-18, \
      3.469446952165922625e-18, 1.734723476047576572e-18, 8.673617380119933728e-19, 4.336808690020650487e-19, \
      2.168404344997219785e-19, 1.084202172494241406e-19, 5.421010862456645411e-20
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double zm1_dev[zeta_terms] = {LAPLACE_WEIBULL_ZM1};
#endif
inline constexpr double zm1_host[zeta_terms] = {LAPLACE_WEIBULL_ZM1};
LAPLACE_HD_FORCEINLINE double zm1(int k) {
  KOKKOS_IF_ON_DEVICE((return zm1_dev[k - 1];))
  KOKKOS_IF_ON_HOST((return zm1_host[k - 1];))
}

LAPLACE_HD_FORCEINLINE double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^n
LAPLACE_HD_FORCEINLINE double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// Pascal's triangle up to row 6: c[n][k] = C(n, k)
struct pascal_table {
  double c[7][7];
};
LAPLACE_HD_FORCEINLINE pascal_table make_pascal() {
  pascal_table p{};
  for (int n = 0; n < 7; ++n) {
    p.c[n][0] = 1.0;
    for (int k = 1; k <= n; ++k) p.c[n][k] = p.c[n - 1][k - 1] + (k <= n - 1 ? p.c[n - 1][k] : 0.0);
  }
  return p;
}

// d^N f(g(x)) by laplace::math::detail_faa::compose, f[1..N] and g[1..N]
template <int N>
LAPLACE_HD_FORCEINLINE double chain(const double* f, const double* g) {
  return math::detail_faa::compose<N>(f, g);
}

// delta^(k)(x), k = 0..N, into d[0..N] (Stirling's error and its derivatives)
template <int N>
LAPLACE_HD_FORCEINLINE void stirling_upto(double x, double* d) {
  namespace ds = laplace::math::detail_stirling;
  d[0] = ds::value<0>(x);
  if constexpr (N >= 1) d[1] = ds::value<1>(x);
  if constexpr (N >= 2) d[2] = ds::value<2>(x);
  if constexpr (N >= 3) d[3] = ds::value<3>(x);
  if constexpr (N >= 4) d[4] = ds::value<4>(x);
  if constexpr (N >= 5) d[5] = ds::value<5>(x);
  if constexpr (N >= 6) d[6] = ds::value<6>(x);
}

// the series coefficient c_j of phi(x) = sum_j c_j x^j: c_0 = -gamma, c_j = (-1)^(j+1) zeta(j + 1)/(j + 1)
LAPLACE_HD_FORCEINLINE double phi_coef(int j) {
  if (j == 0) return -euler_gamma;
  return ((j % 2 == 1) ? 1.0 : -1.0) * (1.0 + zm1(j)) / (j + 1);
}

// phi(x) = lgamma(1 + x)/x, value only
LAPLACE_HD_FORCEINLINE double phi_value(double x) {
  if (x > x_series) return math::lgamma(1.0 + x) / x;
  double s = 0.0, xp = 1.0;
  for (int j = 0; j <= zeta_terms; ++j) {
    const double term = phi_coef(j) * xp;
    s += term;
    if (j >= 2 && Kokkos::fabs(term) <= 4.0 * 2.220446049250313e-16 * Kokkos::fabs(s)) break;
    xp *= x;
  }
  return s;
}

// f[k] = d^k phi/dx^k and fP[k] = d^k (phi(x) - log x)/dx^k, k = 0..K
template <int K>
LAPLACE_HD void phi_derivs(double x, double* f, double* fP) {
  // dl[k] = d^k log x/dx^k; rho[r] = d^r (1/x)/dx^r = (-1)^r r!/x^(r+1)
  double dl[K + 1], rho[K + 1];
  const double ix = 1.0 / x;
  dl[0] = math::log(x);
  {
    double p = 1.0;  // x^-k
    for (int k = 1; k <= K; ++k) {
      p *= ix;
      dl[k] = sgn(k - 1) * fact(k - 1) * p;
    }
    p = ix;
    for (int r = 0; r <= K; ++r, p *= ix) rho[r] = sgn(r) * fact(r) * p;
  }
  const pascal_table pt = make_pascal();
  if (x <= x_series) {
    for (int k = 0; k <= K; ++k) {
      double ff = fact(k), xp = 1.0, s = 0.0;  // ff = j!/(j - k)!
      for (int j = k; j <= zeta_terms; ++j) {
        const double term = phi_coef(j) * ff * xp;
        s += term;
        if (j >= k + 2 && Kokkos::fabs(term) <= 4.0 * 2.220446049250313e-16 * Kokkos::fabs(s)) break;
        ff = ff * (j + 1) / (j + 1 - k);
        xp *= x;
      }
      f[k] = s;
      fP[k] = s - dl[k];
    }
  } else if (x >= x_stirling) {
    double dd[K + 1], q[K + 1];
    stirling_upto<K>(x, dd);
    q[0] = 0.5 * (log_2pi + dl[0]) + dd[0];
    for (int i = 1; i <= K; ++i) q[i] = 0.5 * dl[i] + dd[i];  // d^i (1/2) log x = (1/2) dl[i]
    for (int k = 0; k <= K; ++k) {
      double s = 0.0;
      for (int i = 0; i <= k; ++i) s += pt.c[k][i] * q[i] * rho[k - i];
      fP[k] = (k == 0) ? s - 1.0 : s;
      f[k] = fP[k] + dl[k];
    }
  } else {
    double lg[K + 1];
    lg[0] = math::lgamma(1.0 + x);
    for (int i = 1; i <= K; ++i) lg[i] = math::polygamma(i - 1, 1.0 + x);
    for (int k = 0; k <= K; ++k) {
      double s = 0.0;
      for (int i = 0; i <= k; ++i) s += pt.c[k][i] * lg[i] * rho[k - i];
      f[k] = s;
      fP[k] = s - dl[k];
    }
  }
}

// phi_d[n] = d^n phi(1/sigma)/dsigma^n and P_d[n] = d^n (log sigma + phi(1/sigma))/dsigma^n, n = 0..K
// (the values for n = 0), composing phi^(k)(x) with the derivatives g_j = (-1)^j j!/sigma^(j+1) of x = 1/sigma
template <int K>
LAPLACE_HD void sigma_derivs(double sigma, double* phi_d, double* P_d) {
  const double x = 1.0 / sigma;
  double f[K + 1], fP[K + 1], g[K + 1];
  phi_derivs<K>(x, f, fP);
  g[0] = 0.0;
  double p = x;
  for (int j = 1; j <= K; ++j) {
    p *= x;  // x^(j+1)
    g[j] = sgn(j) * fact(j) * p;
  }
  phi_d[0] = f[0];
  P_d[0] = fP[0];
  if constexpr (K >= 1) phi_d[1] = chain<1>(f, g), P_d[1] = chain<1>(fP, g);
  if constexpr (K >= 2) phi_d[2] = chain<2>(f, g), P_d[2] = chain<2>(fP, g);
  if constexpr (K >= 3) phi_d[3] = chain<3>(f, g), P_d[3] = chain<3>(fP, g);
  if constexpr (K >= 4) phi_d[4] = chain<4>(f, g), P_d[4] = chain<4>(fP, g);
  if constexpr (K >= 5) phi_d[5] = chain<5>(f, g), P_d[5] = chain<5>(fP, g);
  if constexpr (K >= 6) phi_d[6] = chain<6>(f, g), P_d[6] = chain<6>(fP, g);
}

// log(y/m): log1p of the exact (y - m)/m when y is within a factor 2 of m (Sterbenz), where the
// quotient y/m would lose the relative accuracy of the logarithm
LAPLACE_HD_FORCEINLINE double log_ratio(double y, double m) {
  if (Kokkos::fabs(y - m) < 0.5 * m) return Kokkos::log1p((y - m) / m);
  return math::log(y) - math::log(m);
}

// R[j][a][b] = e^-h d^(j + a + b) e^h/(dy^j dm^a dsigma^b) from the partials H[j][a][b] of h
template <int J, int K>
LAPLACE_HD void exp_table(double (&H)[J + 1][K + 1][K + 1], double (&R)[J + 1][K + 1][K + 1]) {
  const pascal_table pt = make_pascal();
  R[0][0][0] = 1.0;
  for (int tot = 1; tot <= J + K; ++tot)
    for (int j = 0; j <= J && j <= tot; ++j)
      for (int a = 0; a <= K && a + j <= tot; ++a) {
        const int b = tot - j - a;
        if (b < 0 || a + b > K) continue;
        int dj = 0, da = 0, db = 0;  // the index raised: sigma first, then m, then y
        if (b > 0) db = 1;
        else if (a > 0) da = 1;
        else dj = 1;
        const int pj = j - dj, pa = a - da, pb = b - db;
        double s = 0.0;
        for (int i = 0; i <= pj; ++i)
          for (int p = 0; p <= pa; ++p)
            for (int q = 0; q <= pb; ++q) {
              const double hh = H[i + dj][p + da][q + db];
              if (hh != 0.0) s += pt.c[pj][i] * pt.c[pa][p] * pt.c[pb][q] * hh * R[pj - i][pa - p][pb - q];
            }
        R[j][a][b] = s;
      }
}

// the partials of h, the table R, T, C and the pure sigma part P
template <int J, int K>
struct weibull_tables {
  double H[J + 1][K + 1][K + 1];
  double R[J + 1][K + 1][K + 1];
  double P[K + 1];  // P[b] = d^b (log sigma + phi(1/sigma))/dsigma^b
  double C, h, T;
};

template <int J, int K>
LAPLACE_HD void build_tables(double m, double sigma, double y, weibull_tables<J, K>& s) {
  for (int j = 0; j <= J; ++j)
    for (int a = 0; a <= K; ++a)
      for (int b = 0; b <= K; ++b) s.H[j][a][b] = 0.0, s.R[j][a][b] = 0.0;
  double phi_d[K + 1];
  sigma_derivs<K>(sigma, phi_d, s.P);
  s.C = log_ratio(y, m);
  s.h = sigma * s.C + phi_d[0];
  s.T = math::exp(s.h);
  if constexpr (K >= 1) {
    double ima = 1.0 / m;  // m^-a
    for (int a = 1; a <= K; ++a, ima /= m) {
      const double v = sgn(a) * fact(a - 1) * ima;
      s.H[0][a][0] = sigma * v;
      s.H[0][a][1] = v;
    }
    for (int b = 1; b <= K; ++b) s.H[0][0][b] = phi_d[b] + (b == 1 ? s.C : 0.0);
  }
  {
    double iyj = 1.0 / y;  // y^-j
    for (int j = 1; j <= J; ++j, iyj /= y) {
      const double v = sgn(j - 1) * fact(j - 1) * iyj;
      s.H[j][0][0] = sigma * v;
      if constexpr (K >= 1) s.H[j][0][1] = v;
    }
  }
  exp_table<J, K>(s.H, s.R);
}

// G[k] = d^k/dT^k log(1 - e^-T), k = 1..K, for T >= 1: (-1)^(k+1) q A_(k-1)(q)/(1 - q)^k with q = e^-T and
// A_m the Eulerian polynomials (every coefficient positive, nothing cancels)
template <int K>
LAPLACE_HD_FORCEINLINE void log_one_minus_exp_derivs_large(double T, double* G) {
  const double q = Kokkos::exp(-T);
  const double omq = -Kokkos::expm1(-T);  // 1 - q
  const double A[6] = {1.0,
                       1.0,
                       1.0 + q,
                       1.0 + q * (4.0 + q),
                       1.0 + q * (11.0 + q * (11.0 + q)),
                       1.0 + q * (26.0 + q * (66.0 + q * (26.0 + q)))};
  double denom = 1.0;
  for (int k = 1; k <= K; ++k) {
    denom /= omq;
    G[k] = -sgn(k) * q * A[k - 1] * denom;
  }
}

// L[k] = d^k/dT^k log exprel(-T), k = 1..K, T in (0, 1): the chain rule of log at z = exprel(-T)
template <int K>
LAPLACE_HD void log_exprel_derivs_small(double T, double* L) {
  const double x = -T;
  const double z = math::exprel(x);
  double e[7] = {0, 0, 0, 0, 0, 0, 0};  // e[j] = exprel^(j)(x)
  if constexpr (K >= 1) e[1] = math::exprel_d1(x);
  if constexpr (K >= 2) e[2] = math::exprel_d2(x);
  if constexpr (K >= 3) e[3] = math::exprel_d3(x);
  if constexpr (K >= 4) e[4] = math::exprel_d4(x);
  if constexpr (K >= 5) e[5] = math::exprel_d5(x);
  if constexpr (K >= 6) e[6] = math::exprel_d6(x);
  double f[K + 1], g[K + 1];
  f[0] = 0.0;
  g[0] = 0.0;
  double iz = 1.0;  // z^-k
  for (int k = 1; k <= K; ++k) {
    iz /= z;
    f[k] = sgn(k - 1) * fact(k - 1) * iz;  // d^k log z = (-1)^(k-1) (k - 1)!/z^k
    g[k] = sgn(k) * e[k];                  // d^k/dT^k exprel(-T) = (-1)^k exprel^(k)(x)
  }
  L[0] = 0.0;
  if constexpr (K >= 1) L[1] = chain<1>(f, g);
  if constexpr (K >= 2) L[2] = chain<2>(f, g);
  if constexpr (K >= 3) L[3] = chain<3>(f, g);
  if constexpr (K >= 4) L[4] = chain<4>(f, g);
  if constexpr (K >= 5) L[5] = chain<5>(f, g);
  if constexpr (K >= 6) L[6] = chain<6>(f, g);
}

// psi(1 + 1/sigma) - (1 - gamma), a single difference; near sigma = 1 the series in d = 1/sigma - 1
LAPLACE_HD_FORCEINLINE double digamma_shift(double sigma) {
  const double d = (1.0 - sigma) / sigma;  // 1/sigma - 1, (1 - sigma) exact for sigma in [1/2, 2]
  if (Kokkos::fabs(d) <= 0.25) {
    double s = 0.0, p = 1.0;
    for (int k = 1; k <= 40; ++k) {
      p *= d;
      const double t = sgn(k + 1) * zm1(k) * p;
      s += t;
      if (Kokkos::fabs(t) <= 4.0 * 2.220446049250313e-16 * Kokkos::fabs(s)) break;
    }
    return s;
  }
  return math::digamma(1.0 + 1.0 / sigma) - one_minus_gamma;
}

// Dd[n] = d^n/dsigma^n [psi(1 + 1/sigma) - (1 - gamma)], n = 0..K
template <int K>
LAPLACE_HD void digamma_shift_derivs(double sigma, double* Dd) {
  const double x = 1.0 / sigma;
  Dd[0] = digamma_shift(sigma);
  if constexpr (K >= 1) {
    double f[K + 1], g[K + 1];
    f[0] = 0.0;
    g[0] = 0.0;
    double p = x;
    for (int k = 1; k <= K; ++k) {
      f[k] = math::polygamma(k, 1.0 + x);
      p *= x;
      g[k] = sgn(k) * fact(k) * p;
    }
    Dd[1] = chain<1>(f, g);
    if constexpr (K >= 2) Dd[2] = chain<2>(f, g);
    if constexpr (K >= 3) Dd[3] = chain<3>(f, g);
    if constexpr (K >= 4) Dd[4] = chain<4>(f, g);
  }
}

// the components of an (y, theta) derivative outside the support (y <= 0), see the file comment
template <int J, int K>
LAPLACE_HD sym_tensor<2, K> outside_support(double m, double sigma) {
  sym_tensor<2, K> t;
  for (int b = 0; b <= K; ++b) {
    if constexpr (J == 0) {
      t[b] = math::detail::quiet_nan();
    } else {
      double v = distrib_infinity;
      if (b == 0 && sigma == 1.0) {  // the exponential at scale m
        double im = 1.0;
        for (int i = 0; i <= K; ++i) im /= m;  // m^-(K + 1)
        v = (J == 1) ? -sgn(K) * fact(K) * im : 0.0;
      }
      t[b] = v;
    }
  }
  return t;
}

}  // namespace detail_weibull

/**
 * \ingroup distrib_weibull
 * \brief Weibull distribution on \f$(0, \infty)\f$ parametrized by its mean \f$m > 0\f$ and shape \f$\sigma > 0\f$.
 *
 * Implements the continuous density
 * \f[
 *   f(y; m, \sigma) = \frac{\sigma}{y} \exp\bigl(h - e^h\bigr), \qquad
 *   h = \sigma \log\!\left(\frac{y}{m}\right) + \log\Gamma\!\left(1 + \frac{1}{\sigma}\right), \qquad y > 0,
 * \f]
 * where \f$b = m / \Gamma(1 + 1/\sigma)\f$ is the implied scale and \f$\mathbb{E}[Y] = m\f$.
 * Random variates use the generic quantile-inversion route on \ref distrib_generic "the generic routes".
 *
 * \see distrib_weibull, distrib_core, logpdf(), expected_info(), cdf(), quantile()
 */
struct weibull_distrib {
  /// \brief Mean parameter \f$m \in (0, \infty)\f$.
  // Mean m in (0, inf).
  double m;
  /// \brief Shape parameter \f$\sigma \in (0, \infty)\f$.
  // Shape sigma in (0, inf): the cumulative hazard is (y/b)^sigma.
  double sigma;

  /// \brief Distribution kind tag (`distrib_kind::continuous`).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// \brief Number of natural parameters (`2`).
  static constexpr int n_params = 2;
  /// \brief Smoothness order of \f$\log f\f$ in \f$y\f$ (`infinitely_smooth`).
  // regularity of log f in y: C^inf on (0, inf)
  static constexpr int y_regularity = infinitely_smooth;
  /// \brief Parameter descriptors for `m` and `sigma` on \f$(0, \infty)\f$.
  static constexpr distrib_param_info params[2] = {
      {"m", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the log-density \f$\ell(m, \sigma; y) = \log\sigma - \log y + h - e^h\f$.
   * \param y Observation point.
   * \return Log-density value, or \f$-\infty\f$ when \f$y \le 0\f$.
   * \par Accuracy
   * Measured worst error is \f$45.3\f$ scaled ulp on `gcc` (test bound \f$80\f$).
   * \see distrib_weibull, logpdf_dk(), logpdf_dy_dk()
   */
  // log f(y) = log sigma - log y + h - T, written as P(sigma) - log y + sigma C - T with
  // P = log sigma + phi(1/sigma), so that at small sigma the log sigma and the log x of phi cancel exactly;
  // -inf for y <= 0
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    double phi_d[1], P_d[1];
    detail_weibull::sigma_derivs<0>(sigma, phi_d, P_d);
    const double C = detail_weibull::log_ratio(y, m);
    const double h = sigma * C + phi_d[0];
    return P_d[0] - math::log(y) + sigma * C - math::exp(h);
  }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates order-\f$K\f$ parameter derivatives of \f$\log f\f$ in \f$(m, \sigma)\f$.
   *
   * Component \f$c\f$ holds \f$\partial^K \ell / \partial m^{K-c} \partial \sigma^c\f$.
   * Returns quiet NaN in every component when \f$y \le 0\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>`.
   * \par Accuracy
   * Measured worst error across orders \f$1\f$ to \f$6\f$ is \f$7.33\f$ to \f$12.1\f$ scaled ulp on `gcc` (bounds \f$20\f$ to \f$50\f$).
   * \see distrib_weibull, logpdf(), logpdf_dy_dk(), logpdf_d1()
   */
  // Derivatives of order K of log f in (m, sigma); the component at position b has b indices in sigma.
  // NaN for y <= 0.
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    return logpdf_dy_dk<0, K>(y);
  }
  /**
   * \name Fixed-order derivatives of the log-density
   * \brief Convenience wrappers delegating to `logpdf_dk<K>()` for \f$K = 1 \dots 6\f$.
   * @{
   */
  /// \brief First-order derivatives of \f$\log f\f$; measured \f$7.33\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `logpdf_dk<1>(y)`.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// \brief Second-order derivatives of \f$\log f\f$; measured \f$8.01\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `logpdf_dk<2>(y)`.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// \brief Third-order derivatives of \f$\log f\f$; measured \f$12.1\f$ scaled ulp on `gcc` (bound \f$20\f$). \param y Observation point. \return `logpdf_dk<3>(y)`.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// \brief Fourth-order derivatives of \f$\log f\f$; measured \f$9.78\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `logpdf_dk<4>(y)`.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// \brief Fifth-order derivatives of \f$\log f\f$; measured \f$8.79\f$ scaled ulp on `gcc` (bound \f$40\f$). \param y Observation point. \return `logpdf_dk<5>(y)`.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// \brief Sixth-order derivatives of \f$\log f\f$; measured \f$8.09\f$ scaled ulp on `gcc` (bound \f$50\f$). \param y Observation point. \return `logpdf_dk<6>(y)`.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates mixed derivatives \f$\partial_y^J \partial_\theta^K \log f(y; m, \sigma)\f$ for \f$J + K \le 6\f$.
   *
   * Returns `logpdf()` when \f$J = K = 0\f$. At \f$y \le 0\f$: returns quiet NaN when \f$J = 0, K \ge 1\f$;
   * for \f$J \ge 1\f$, returns \f$+\infty\f$ except on pure-\f$m\f$ components at \f$\sigma = 1\f$, where it returns
   * \f$-(-1)^K K!\,m^{-(K+1)}\f$ for \f$J = 1\f$ and \f$0.0\f$ for \f$J \ge 2\f$.
   * \tparam J Derivative order in \f$y\f$ (\f$J \ge 0\f$).
   * \tparam K Derivative order in \f$(m, \sigma)\f$ (\f$K \ge 0\f$, \f$J + K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>`.
   * \par Accuracy
   * Measured worst error across total orders \f$1\f$ to \f$6\f$ is \f$2.53\f$ to \f$137\f$ scaled ulp on `gcc` (bounds \f$20\f$ to \f$300\f$).
   * \see distrib_weibull, logpdf(), logpdf_dk()
   */
  // d^(J+K) log f/dy^J d(m, sigma)^K, J + K <= 6, as a symmetric tensor of order K in (m, sigma).
  // J = 0, K = 0 is the value. y <= 0: see the file comment (infinity in y, NaN in the parameters).
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    static_assert(J >= 0 && K >= 0 && J + K <= 6, "J + K must be at most 6");
    sym_tensor<2, K> t;
    if constexpr (J == 0 && K == 0) {
      t[0] = logpdf(y);
      return t;
    } else {
      using detail_weibull::fact;
      using detail_weibull::sgn;
      if (!(y > 0.0)) return detail_weibull::outside_support<J, K>(m, sigma);
      detail_weibull::weibull_tables<J, K> s;
      detail_weibull::build_tables<J, K>(m, sigma, y, s);
      for (int b = 0; b <= K; ++b) {
        const int a = K - b;
        double v = s.H[J][a][b] - s.T * s.R[J][a][b];
        if (J == 0 && a == 0 && b >= 1) {
          // the pure sigma part: log sigma + phi(1/sigma) replaces log sigma and the phi of h (the C
          // of h_sigma stays), so that nothing cancels between them
          v = s.P[b] + (b == 1 ? s.C : 0.0) - s.T * s.R[0][0][b];
        }
        if (J >= 1 && a == 0 && b == 0) {
          double iyj = 1.0;  // y^-J
          for (int i = 0; i < J; ++i) iyj /= y;
          v += sgn(J) * fact(J - 1) * iyj;  // d^J (-log y)
        }
        t[b] = v;
      }
      return t;
    }
  }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the closed expected Fisher information \f$I(m, \sigma)\f$.
   *
   * Stores \f$I_{mm} = \sigma^2/m^2\f$, \f$I_{m,\sigma} = D/m\f$, and
   * \f$I_{\sigma,\sigma} = (D^2 + \pi^2/6)/\sigma^2\f$ with \f$D = \psi(1 + 1/\sigma) - (1 - \gamma)\f$.
   * Returns \f$I_{m,\sigma} = 0.0\f$ identically at \f$\sigma = 1\f$.
   * \return Symmetric tensor `sym_tensor<2, 2>` with entries `[I_mm, I_m_sigma, I_sigma_sigma]`.
   * \par Accuracy
   * Measured within \f$1.8 \times 10^{-16}\f$, \f$1.6 \times 10^{-16}\f$, and \f$3.0 \times 10^{-16}\f$ of the natural scales on `gcc` (test bound \f$3 \times 10^{-14}\f$).
   * \see distrib_weibull, expected_info_dk()
   */
  // Expected information I(m, sigma): [0] I_mm, [1] I_m,sigma, [2] I_sigma,sigma
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double D = detail_weibull::digamma_shift(sigma);
    sym_tensor<2, 2> t;
    t[0] = sigma * sigma / (m * m);
    t[1] = D / m;
    t[2] = (D * D + detail_weibull::pi2_over_6) / (sigma * sigma);
    return t;
  }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates order-\f$K\f$ parameter derivatives of the expected Fisher information.
   *
   * Returns exact zeros for components beyond two \f$\sigma\f$-derivatives in \f$I_{mm}\f$ and
   * for any \f$m\f$-derivative in \f$I_{\sigma,\sigma}\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container `expected_info_tensor<2, K>`.
   * \par Accuracy
   * Checked against 5-point finite differences (tolerance \f$10^{-6}\f$) and generic score quadrature (\f$10^{-8}\f$ at \f$K=1\f$, \f$10^{-7}\f$ at \f$K=2\f$).
   * \see distrib_weibull, expected_info(), expected_info_d1()
   */
  // d^K I_ab in (m, sigma), K = 1..4; pair[0] = I_mm, pair[1] = I_m,sigma, pair[2] = I_sigma,sigma,
  // each a symmetric tensor of order K (position b = number of sigma indices)
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    using detail_weibull::fact;
    using detail_weibull::sgn;
    static_assert(K >= 1 && K <= 4, "K must be 1..4");
    double Dd[K + 1];
    detail_weibull::digamma_shift_derivs<K>(sigma, Dd);
    const detail_weibull::pascal_table pt = detail_weibull::make_pascal();
    expected_info_tensor<2, K> t;
    // powers of 1/m and 1/sigma
    double im[K + 3], is[K + 3];
    im[0] = 1.0;
    is[0] = 1.0;
    for (int i = 1; i < K + 3; ++i) im[i] = im[i - 1] / m, is[i] = is[i - 1] / sigma;
    const double s2[3] = {sigma * sigma, 2.0 * sigma, 2.0};
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      // I_mm = sigma^2 m^-2
      t.pair[0][b] = (b <= 2) ? sgn(a) * fact(a + 1) * im[a + 2] * s2[b] : 0.0;
      // I_m,sigma = D m^-1
      t.pair[1][b] = sgn(a) * fact(a) * im[a + 1] * Dd[b];
    }
    // I_sigma,sigma = (D^2 + pi^2/6) sigma^-2 involves no m: only the component with all indices in sigma
    double E[K + 1];  // E[n] = d^n (D^2 + pi^2/6)/dsigma^n
    for (int n = 0; n <= K; ++n) {
      double acc = 0.0;
      for (int i = 0; i <= n; ++i) acc += pt.c[n][i] * Dd[i] * Dd[n - i];
      E[n] = acc + (n == 0 ? detail_weibull::pi2_over_6 : 0.0);
    }
    for (int b = 0; b < K; ++b) t.pair[2][b] = 0.0;
    double acc = 0.0;
    for (int n = 0; n <= K; ++n) {
      const int r = K - n;  // derivatives of sigma^-2: (-1)^r (r + 1)!/sigma^(r + 2)
      acc += pt.c[K][n] * E[n] * sgn(r) * fact(r + 1) * is[r + 2];
    }
    t.pair[2][K] = acc;
    return t;
  }
  /**
   * \name Fixed-order derivatives of the expected Fisher information
   * \brief Convenience wrappers delegating to `expected_info_dk<K>()` for \f$K = 1 \dots 4\f$.
   * @{
   */
  /// \brief First-order derivatives of the expected information. \return `expected_info_dk<1>()`.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// \brief Second-order derivatives of the expected information. \return `expected_info_dk<2>()`.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// \brief Third-order derivatives of the expected information. \return `expected_info_dk<3>()`.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// \brief Fourth-order derivatives of the expected information. \return `expected_info_dk<4>()`.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the log-cumulative hazard \f$h = \sigma \log(y/m) + \phi(1/\sigma)\f$.
   * \param y Positive observation \f$y > 0\f$.
   * \return Value of \f$h = \log T\f$, or \f$-\infty\f$ at \f$y = 0\f$ and quiet NaN for \f$y < 0\f$.
   * \par Accuracy
   * Verified through `log_cdf()` when \f$T\f$ underflows within \f$10^{-12}\f$.
   * \see distrib_weibull, log_cdf(), log_ccdf()
   */
  // The cumulative hazard T = (y/b)^sigma at y > 0, with h = log T
  LAPLACE_HD double hazard_log(double y) const {
    return sigma * detail_weibull::log_ratio(y, m) + detail_weibull::phi_value(1.0 / sigma);
  }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the cumulative distribution function \f$F(y) = 1 - e^{-T}\f$.
   * \param y Observation point.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$, or `0.0` when \f$y \le 0\f$.
   * \par Accuracy
   * Checked against \f$-\operatorname{expm1}(-(y/b)^\sigma)\f$ within \f$10^{-12}\f$ and `1 - ccdf(y)` within \f$10^{-15}\f$.
   * \see distrib_weibull, ccdf(), log_cdf(), quantile()
   */
  // F(y) = -expm1(-T)
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : -Kokkos::expm1(-math::exp(hazard_log(y))); }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = e^{-T}\f$.
   * \param y Observation point.
   * \return Survival probability \f$\mathbb{P}(Y > y)\f$, or `1.0` when \f$y \le 0\f$.
   * \par Accuracy
   * Checked against `1 - cdf(y)` within \f$10^{-15}\f$ and `exp(log_ccdf(y))`.
   * \see distrib_weibull, cdf(), log_ccdf()
   */
  // 1 - F(y) = e^-T
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : math::exp(-math::exp(hazard_log(y))); }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates \f$\log F(y)\f$ without cancellation in either tail.
   *
   * Uses \f$h + \log\operatorname{exprel}(-T)\f$ for \f$T < 1\f$ and \f$\operatorname{log1p}(-e^{-T})\f$ for \f$T \ge 1\f$.
   * Returns \f$-\infty\f$ when \f$y \le 0\f$, and stays finite when \f$T\f$ underflows.
   * \param y Observation point.
   * \return Value of \f$\log F(y)\f$.
   * \par Accuracy
   * Measured worst error is \f$2.52\f$ scaled ulp on `gcc` (test bound \f$20\f$).
   * \see distrib_weibull, cdf(), log_cdf_dk(), log_ccdf()
   */
  // log F = h + log exprel(-T) for T < 1 (h = log T exactly, also where T underflows), log1p(-e^-T) for T >= 1
  LAPLACE_HD double log_cdf(double y) const {
    if (y <= 0.0) return -distrib_infinity;
    const double h = hazard_log(y), T = math::exp(h);
    if (T < 1.0) return h + math::log(math::exprel(-T));
    return Kokkos::log1p(-Kokkos::exp(-T));
  }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates \f$\log(1 - F(y)) = -T\f$ exactly.
   * \param y Observation point.
   * \return Value of \f$\log(1 - F(y))\f$, or `0.0` when \f$y \le 0\f$.
   * \par Accuracy
   * Measured worst error is \f$9.35\f$ scaled ulp on `gcc` (test bound \f$30\f$).
   * \see distrib_weibull, ccdf(), log_ccdf_dk(), log_cdf()
   */
  // log(1 - F) = -T exactly
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : -math::exp(hazard_log(y)); }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates order-\f$K\f$ parameter derivatives of \f$\log(1 - F(y)) = -T\f$.
   *
   * Returns `0.0` in every component when \f$y \le 0\f$, and stays finite where \f$1 - F\f$ underflows.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>`.
   * \par Accuracy
   * Measured worst error across orders \f$1\f$ to \f$6\f$ is \f$6.80\f$ to \f$12.1\f$ scaled ulp on `gcc` (bounds \f$20\f$ to \f$50\f$).
   * \see distrib_weibull, log_ccdf(), log_cdf_dk(), log_ccdf_d1()
   */
  // Derivatives of order K of log(1 - F) = -T in (m, sigma): -T R_I, the density's own Bell machinery
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    sym_tensor<2, K> t;
    if (!(y > 0.0)) return t;  // log(1 - F) = 0 for y <= 0
    detail_weibull::weibull_tables<0, K> s;
    detail_weibull::build_tables<0, K>(m, sigma, y, s);
    for (int b = 0; b <= K; ++b) t[b] = -s.T * s.R[0][K - b][b];
    return t;
  }

  /**
   * \ingroup distrib_weibull
   * \brief Evaluates order-\f$K\f$ parameter derivatives of \f$\log F(y)\f$.
   *
   * Returns quiet NaN in every component when \f$y \le 0\f$, and `0.0` in every component when \f$T \ge 750\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>`.
   * \par Accuracy
   * Measured worst error across orders \f$1\f$ to \f$6\f$ is \f$4.10\f$ to \f$479\f$ scaled ulp on `gcc` (bounds \f$20\f$ to \f$900\f$).
   * \see distrib_weibull, log_cdf(), log_ccdf_dk(), log_cdf_d1()
   */
  // Derivatives of order K of log F in (m, sigma); NaN for y <= 0
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    using detail_weibull::sgn;
    sym_tensor<2, K> t;
    if (!(y > 0.0)) {
      for (int b = 0; b <= K; ++b) t[b] = math::detail::quiet_nan();
      return t;
    }
    detail_weibull::weibull_tables<0, K> s;
    detail_weibull::build_tables<0, K>(m, sigma, y, s);
    const double T = s.T;
    if (T >= 750.0) return t;  // log F = log1p(-e^-T) = 0 to the last bit, with every derivative
    double Tp[K + 1][K + 1];  // the partials of T over (m^a sigma^b)
    for (int a = 0; a <= K; ++a)
      for (int b = 0; b <= K; ++b) Tp[a][b] = (a + b <= K) ? T * s.R[0][a][b] : 0.0;
    double G[K + 1];
    if (T >= 1.0) detail_weibull::log_one_minus_exp_derivs_large<K>(T, G);
    else detail_weibull::log_exprel_derivs_small<K>(T, G);
    const detail_weibull::pascal_table pt = detail_weibull::make_pascal();
    // S[k][a][b] = d^(a, b) G^(k)(T(theta)); k + a + b <= K
    double S[K + 1][K + 1][K + 1];
    for (int k = 0; k <= K; ++k)
      for (int a = 0; a <= K; ++a)
        for (int b = 0; b <= K; ++b) S[k][a][b] = 0.0;
    for (int k = 1; k <= K; ++k) S[k][0][0] = G[k];
    for (int tot = 1; tot <= K; ++tot)
      for (int a = 0; a <= tot; ++a) {
        const int b = tot - a;
        const int da = (b > 0) ? 0 : 1, db = (b > 0) ? 1 : 0;
        const int pa = a - da, pb = b - db;
        for (int k = (tot == K ? 0 : 1); k <= K - tot; ++k) {
          double acc = 0.0;
          for (int p = 0; p <= pa; ++p)
            for (int q = 0; q <= pb; ++q)
              acc += pt.c[pa][p] * pt.c[pb][q] * S[k + 1][p][q] * Tp[pa - p + da][pb - q + db];
          S[k][a][b] = acc;
        }
      }
    for (int b = 0; b <= K; ++b) {
      const int a = K - b;
      t[b] = S[0][a][b] + (T < 1.0 ? s.H[0][a][b] : 0.0);
    }
    return t;
  }

  /**
   * \name Fixed-order derivatives of the log-CDF
   * \brief Convenience wrappers delegating to `log_cdf_dk<K>()` for \f$K = 1 \dots 6\f$.
   * @{
   */
  /// \brief First-order derivatives of \f$\log F\f$; measured \f$4.10\f$ scaled ulp on `gcc` (bound \f$20\f$). \param y Observation point. \return `log_cdf_dk<1>(y)`.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// \brief Second-order derivatives of \f$\log F\f$; measured \f$13.0\f$ scaled ulp on `gcc` (bound \f$30\f$). \param y Observation point. \return `log_cdf_dk<2>(y)`.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// \brief Third-order derivatives of \f$\log F\f$; measured \f$9.75\f$ scaled ulp on `gcc` (bound \f$40\f$). \param y Observation point. \return `log_cdf_dk<3>(y)`.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// \brief Fourth-order derivatives of \f$\log F\f$; measured \f$110\f$ scaled ulp on `gcc` (bound \f$150\f$). \param y Observation point. \return `log_cdf_dk<4>(y)`.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// \brief Fifth-order derivatives of \f$\log F\f$; measured \f$402\f$ scaled ulp on `gcc` (bound \f$600\f$). \param y Observation point. \return `log_cdf_dk<5>(y)`.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// \brief Sixth-order derivatives of \f$\log F\f$; measured \f$479\f$ scaled ulp on `gcc` (bound \f$900\f$). \param y Observation point. \return `log_cdf_dk<6>(y)`.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Fixed-order derivatives of the log-CCDF
   * \brief Convenience wrappers delegating to `log_ccdf_dk<K>()` for \f$K = 1 \dots 6\f$.
   * @{
   */
  /// \brief First-order derivatives of \f$\log(1 - F)\f$; measured \f$6.80\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `log_ccdf_dk<1>(y)`.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// \brief Second-order derivatives of \f$\log(1 - F)\f$; measured \f$8.03\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `log_ccdf_dk<2>(y)`.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// \brief Third-order derivatives of \f$\log(1 - F)\f$; measured \f$12.1\f$ scaled ulp on `gcc` (bound \f$20\f$). \param y Observation point. \return `log_ccdf_dk<3>(y)`.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// \brief Fourth-order derivatives of \f$\log(1 - F)\f$; measured \f$9.78\f$ scaled ulp on `gcc` (bound \f$25\f$). \param y Observation point. \return `log_ccdf_dk<4>(y)`.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// \brief Fifth-order derivatives of \f$\log(1 - F)\f$; measured \f$8.78\f$ scaled ulp on `gcc` (bound \f$40\f$). \param y Observation point. \return `log_ccdf_dk<5>(y)`.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// \brief Sixth-order derivatives of \f$\log(1 - F)\f$; measured \f$8.27\f$ scaled ulp on `gcc` (bound \f$50\f$). \param y Observation point. \return `log_ccdf_dk<6>(y)`.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_weibull
   * \brief Returns the log-scale \f$\log b = \log m - \log\Gamma(1 + 1/\sigma)\f$.
   * \return Finite logarithm of the implied scale \f$b\f$, avoiding underflow at small \f$\sigma\f$.
   * \par Accuracy
   * Checked against \f$\log m - \operatorname{lgamma}(1 + 1/\sigma)\f$ within \f$10^{-13}\f$ at \f$\sigma = 10^{-3}\f$.
   * \see distrib_weibull, quantile(), scale()
   */
  // log b = log m - lgamma(1 + 1/sigma), on the log scale only
  LAPLACE_HD double log_scale() const {
    const double x = 1.0 / sigma;
    return math::log(m) - x * detail_weibull::phi_value(x);
  }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the quantile \f$F^{-1}(p) = \exp(\log b + \log(-\operatorname{log1p}(-p))/\sigma)\f$.
   * \param p Probability \f$p \in [0, 1]\f$.
   * \return Quantile value, `0.0` at \f$p = 0.0\f$, \f$+\infty\f$ at \f$p = 1.0\f$, or quiet NaN outside \f$[0, 1]\f$.
   * \par Accuracy
   * Round-trips through `cdf()` within \f$10^{-11}\f$ and matches generic root-finding within \f$10^{-9}\f$.
   * \see distrib_weibull, quantile_from_log(), quantile_from_log_upper(), cdf()
   */
  // Q(p) = exp(log b + log(Lambda)/sigma) with Lambda = -log(1 - p) = -log1p(-p), never forming 1 - p
  LAPLACE_HD double quantile(double p) const {
    if (!(p > 0.0)) return p == 0.0 ? 0.0 : math::detail::quiet_nan();
    if (!(p < 1.0)) return p == 1.0 ? distrib_infinity : math::detail::quiet_nan();
    return math::exp(log_scale() + math::log(-Kokkos::log1p(-p)) / sigma);
  }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the quantile at probability \f$p = e^{\log p}\f$ without forming \f$1 - p\f$.
   * \param log_p Log-probability \f$\log p \le 0\f$.
   * \return Quantile value, \f$+\infty\f$ at `log_p == 0.0`, or quiet NaN when `log_p > 0.0` or NaN.
   * \par Accuracy
   * Round-trips through `log_cdf()` within \f$10^{-10}\f$ and matches `quantile()` within \f$10^{-12}\f$.
   * \see distrib_weibull, quantile(), quantile_from_log_upper(), log_cdf()
   */
  // the quantile at p = e^log_p, log_p <= 0: Lambda = -log(1 - e^log_p) (Maechler's switch at -log 2), and
  // log Lambda = log_p once e^log_p underflows
  LAPLACE_HD double quantile_from_log(double log_p) const {
    if (!(log_p <= 0.0)) return math::detail::quiet_nan();
    if (log_p == 0.0) return distrib_infinity;
    double log_lambda;
    if (log_p < -700.0) {
      log_lambda = log_p;
    } else {
      const double lam = (log_p > -0.69314718055994530942) ? -math::log(-Kokkos::expm1(log_p))
                                                             : -Kokkos::log1p(-math::exp(log_p));
      log_lambda = math::log(lam);
    }
    return math::exp(log_scale() + log_lambda / sigma);
  }
  /**
   * \ingroup distrib_weibull
   * \brief Evaluates the quantile at upper-tail log-probability \f$\log(1 - p)\f$.
   * \param log_q Complementary log-probability \f$\log(1 - p) \le 0\f$.
   * \return Quantile value, `0.0` at `log_q == 0.0`, or quiet NaN when `log_q > 0.0` or NaN.
   * \par Accuracy
   * Round-trips through `log_ccdf()` within \f$10^{-10}\f$ and matches `quantile()` within \f$10^{-12}\f$.
   * \see distrib_weibull, quantile(), quantile_from_log(), log_ccdf()
   */
  // the quantile at p = 1 - e^log_q, log_q <= 0: Lambda = -log_q
  LAPLACE_HD double quantile_from_log_upper(double log_q) const {
    if (!(log_q <= 0.0)) return math::detail::quiet_nan();
    if (log_q == 0.0) return 0.0;
    return math::exp(log_scale() + math::log(-log_q) / sigma);
  }

  /// \ingroup distrib_weibull
  /// \brief Lower bound of the support (`0.0`). \return `0.0`.
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_weibull
  /// \brief Upper bound of the support (\f$+\infty\f$). \return `distrib_infinity`.
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /**
   * \ingroup distrib_weibull
   * \brief Returns the central location, which equals the mean \f$m\f$.
   * \return Value of `m`.
   * \see distrib_weibull, scale()
   */
  // the mean m
  LAPLACE_HD double center() const { return m; }
  /**
   * \ingroup distrib_weibull
   * \brief Returns the standard deviation \f$m\,e^{L/2}\sqrt{-\operatorname{expm1}(-L)}\f$.
   * \return Positive standard deviation of the distribution.
   * \par Accuracy
   * Checked against the closed standard deviation within \f$10^{-12}\f$ (and within \f$2 \times 10^{-4}\f$ of \f$m\pi/(\sigma\sqrt{6})\f$ at \f$\sigma = 10^4\f$).
   * \see distrib_weibull, center(), log_scale()
   */
  // the standard deviation m sqrt(Gamma(1 + 2/sigma)/Gamma(1 + 1/sigma)^2 - 1), written as
  // m e^(L/2) sqrt(-expm1(-L)) with L = lgamma(1 + 2/sigma) - 2 lgamma(1 + 1/sigma), the Gamma factors
  // never divided, L = (2/sigma)(phi(2/sigma) - phi(1/sigma))
  LAPLACE_HD double scale() const {
    const double x = 1.0 / sigma;
    const double L = 2.0 * x * (detail_weibull::phi_value(2.0 * x) - detail_weibull::phi_value(x));
    return m * math::exp(0.5 * L) * Kokkos::sqrt(-Kokkos::expm1(-L));
  }
  /// \ingroup distrib_weibull
  /// \brief Non-smooth points in the parameters (none). \return Empty `distrib_kinks`.
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// \ingroup distrib_weibull
  /// \brief Non-smooth points in the observation \f$y\f$ (none). \return Empty `distrib_kinks`.
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }

  /**
   * \ingroup distrib_weibull
   * \brief Host starting values from a sample summary.
   *
   * Sets `m` to `s.mean` and `sigma` to \f$(s_y / \bar{y})^{-1.086}\f$ floored at \f$10^{-2}\f$,
   * defaulting to \f$m = 1.0\f$ (if non-positive or non-finite) and \f$\sigma = 1.0\f$ on
   * degenerate samples, each pushed inside \f$(0, \infty)\f$ via `start_inside()`.
   * \param s Sample summary computed by `summarize_sample()`.
   * \return A `weibull_distrib` strictly inside the open parameter domain.
   * \par Accuracy
   * Exact formula in the summary statistics; checked on regular and degenerate samples in the tests.
   * \see distrib_weibull, distrib_start_value(), distrib_start
   */
  // Starting value from a sample (generic_start.hpp): the moment estimate m = the sample
  // mean and sigma = (sd/mean)^-1.086 (the approximation of the Weibull shape from the coefficient
  // of variation), at least 1e-2, each strictly inside the domain. A degenerate sample (a non-positive
  // or non-finite mean, or a zero or non-finite variance) gives m = the mean when it is positive and
  // finite, 1 otherwise, and sigma = 1 (the exponential), which a fit moves from.
  weibull_distrib start_value(const distrib_sample_summary& s) const {
    const bool mean_ok = std::isfinite(s.mean) && s.mean > 0.0;
    const double mean = mean_ok ? s.mean : 1.0;
    double shape = 1.0;
    if (mean_ok && std::isfinite(s.variance) && s.variance > 0.0) {
      const double cv = std::sqrt(s.variance) / s.mean;
      shape = std::pow(cv, -1.086);
      if (!std::isfinite(shape)) shape = 1.0;
      if (shape < 1e-2) shape = 1e-2;
    }
    return {start_inside(mean, params[0]), start_inside(shape, params[1])};
  }
};

}  // namespace laplace::distrib
