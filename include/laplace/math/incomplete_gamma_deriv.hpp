// Derivatives of log P(a, x) and log Q(a, x) with respect to the shape a,
// orders 1 to 6, Laplace's own (a > 0, x >= 0). P and Q = 1 - P are the
// regularized lower and upper incomplete gamma functions.
//
// One function per order: own::log_gamma_p_da1 .. log_gamma_p_da6 and
// own::log_gamma_q_da1 .. log_gamma_q_da6 (order 6 added on 2026-11-12). They share the private helpers
// below, which compute the derivatives up to the order asked for.
//
// Mathematics. With v = log t and d = v - log x, the integrals of the two sides
// are, relative to the factor x^a e^-x,
//
//   lower  a Gamma(a) P(a, x) / (x^a e^-x) = I_L(a) = 1 + x int_{-inf}^0 exp((a+1) d - x (e^d - 1)) dd,
//   upper    Gamma(a) Q(a, x) / (x^a e^-x) = I_U(a) =     int_0^{inf}    exp( a    d - x (e^d - 1)) dd.
//
// The lower one comes from integrating t^(a-1) e^-t by parts once
// (P(a, x) = P(a + 1, x) + x^a e^-x / Gamma(a + 1)): its measure is a point mass
// at d = 0 plus a density, and the pole 1/a of psi(a) at small a has been
// taken out analytically (Gamma(a + 1) instead of Gamma(a)). In both, the
// a-dependence is exp(a d) against a fixed measure, so d^k log I / da^k are the
// cumulants kappa_k of d under the normalized measure, and
//
//   d log P / da   = log(x/a) - (psi(a + 1) - log a) + kappa_1     (lower)
//   d log Q / da   = log(x/a) + (log a - psi(a))     + kappa_1     (upper)
//   d^k log S / da^k = kappa_k - psi^(k-1)(a + 1)  (lower) or  - psi^(k-1)(a)  (upper),  k >= 2.
//
// In the first two lines every term has the sign of the result (kappa_1 <= 0
// and log(x/a) < 0 on the lower side, both >= 0 on the upper side), so nothing
// cancels; psi(a) - log a is computed without forming psi(a) and log a apart.
// The side computed is the one with the smaller probability S (a >= 1: lower
// when x < a - 1/3, the approximate median; a < 1: upper for x >= 1, below
// from a two-term estimate of P), except that log P for a < 1 and x < 1/4 is
// always taken from the lower side (see log_gamma_da). On the side computed
// the integrand is monotone away from d = 0, or nearly so. The other side is
// log(1 - S): from the derivatives l_j of log S, the moments S^(j)/S are the
// complete Bell polynomials b_j(l_1, ..., l_j), the moments of 1 - S are
// -rho b_j with rho = S/(1 - S), and its cumulants follow by the usual
// recursion. For S = Q the Bell polynomials are formed from the derivatives
// of log(Q/a), and S^(j)/S = b_j + j b_(j-1)/a: the pole 1/a^j of psi^(j-1)(a)
// would otherwise cancel inside b_j (1e-8 relative at a = 0.009, order 5).
// In the far tails the complement inherits the relative error of
// rho = exp(log S - log(1 - S)), about |log S| ulps.
//
// Order 6 uses the same routes and helpers as orders 1 to 5: the quadrature
// carries the sixth central moment (M6 in central, only when k = 6) and
// kappa_6 = mu_6 - 15 mu_4 mu_2 - 10 mu_3^2 + 30 mu_2^3; the series, the
// fraction, the closed form at small x and Temme's expansion carry their
// Taylor polynomials to degree 6; log erfcx in Temme's route gets
// g_6 = -g_5 (t + 2R) - 4 g_4 (1 + 2 g_2) - 6 g_3^2. Measured on 2026-11-12
// (docs/numerics.md): the largest errors, at small a where kappa_6 and
// psi^(5)(a + 1) (about 113 to 121 for a < 0.02) cancel, are about 4 times
// those of order 5, the same relative error of kappa_k.
//
// Three ways of computing the cumulants kappa_k are written:
//
// 1. quad: fixed double-exponential rules, 8 nodes per unit of the rule's
//    variable. The lower side in d by the exp-exp rule of Takahasi and Mori
//    from d = 0, scale 1/sqrt(r^2 + x), r = a + 1 - x the slope of the
//    log-integrand at d = 0 and x its curvature. The upper side in
//    u = t - x = x (e^d - 1), where the integrand falls like e^-u, by the same
//    rule; for x < 1/4 (only a < 1), where the integrand is flat in d up to
//    d = log(1/x) (t = 1), tanh-sinh on [0, log(1/x)] in d (two panels) and
//    the exp-exp rule in u = t - 1 beyond. The central moments of d are
//    accumulated in one pass with the weighted update of Pebay (Sandia report
//    SAND2008-6212), so that the cumulants are not formed from raw moments
//    that cancel. About 70 nodes, each with three exponentials.
// 2. series: the lower side by the series I_L = sum_n prod_{j<=n} x/(a + j)
//    differentiated term by term (Taylor coefficients in a carried by a
//    recursion of five multiply-adds per order), centred on the mean of a
//    first pass. The cost grows like the number of terms, O(sqrt(a)) near
//    x = a.
// 3. fraction: the upper side by Legendre's continued fraction
//    I_U = 1/(x + 1 - a - 1 (1 - a)/(x + 3 - a - 2 (2 - a)/(x + 5 - a - ...)))
//    differentiated through the forward recurrences of its numerators and
//    denominators, as in Moore (Applied Statistics AS 187, 1982) for orders 1
//    and 2, here to order 5. Measured and rejected: from order 2 on the
//    cumulants of numerator and denominator cancel (1e8 to 1e12 ulp), and its
//    convergence test on the derivatives can run to the iteration cap.
// 4. the asymptotic series of the upper function, x I_U = sum_n prod_{j<=n} (a - j)/x,
//    differentiated like the lower series, for x >= 36 where it reaches the
//    tolerance before its smallest term.
// 5. for the upper side at x < 3/2, a < 2 (when log Q itself is asked for) the
//    closed form of gamma_inc's small-x branch with the pole of log a taken
//    out (upper_small_x).
// 6. Temme's uniform expansion for a >= 20, |x/a - 1| < 0.3 (a >= 300: < 0.4;
//    a >= 1000: < 0.5), carried in Taylor form in the increment of a
//    (temme_side), on both sides and both backends.
// 7. Legendre's continued fraction by the modified Lentz scheme in Taylor form
//    (fraction_taylor) for the upper side at x >= 3/2 (log Q) or x >= 3
//    (complement), and x >= a + 0.75 sqrt(a) (closer to a its high orders lose
//    accuracy, see chosen_side), on the host only.
// The chosen route (log_gamma_da_impl, chosen_side): 6 where it applies; then
// on the side of the smaller probability (log P from the lower side for a < 2,
// x < 3/2) 5, 2 or 4, 7, and the quadrature for what remains. The series
// lengths are limited per backend (lower 300, upper 500 on the host; lower 60
// for a < 2 only on the device). The quadrature's nodes are tabulated on the
// host and in the device's constant memory.
// Derived from the formulas above; no code copied (R's pgamma.c is GPL).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/log1p.hpp>

#if defined(__CUDACC__)
#define LAPLACE_GD_NOINLINE __noinline__
#else
#define LAPLACE_GD_NOINLINE
#endif

namespace laplace::math::own {

namespace detail_gamma_deriv {

constexpr double eps = 1.1102230246251565e-16;  // 2^-53
constexpr double half_pi = 1.57079632679489661923;
constexpr double ln_half = -0.69314718055994530942;

// Derivatives of log S on the side computed: S = P (lower) or Q.
// With direct false, l holds instead the derivatives of log R, R = S / a^p,
// p = 0 for P and 1 for Q (Q = a R takes the pole of psi(a) out of the
// derivatives of log Q at small a, which the complement would cancel).
struct side_result {
  bool lower, direct;
  double log_s;
  double l[7];  // l[k] = d^k log S / da^k (or d^k log R / da^k), k = 1..order
};

// e^d - 1 - d without cancellation: its Taylor series for |d| <= 1/2, where
// the direct form loses digits
LAPLACE_HD double expm1mx(double d) {
  if (Kokkos::fabs(d) <= 0.5) {
    // d^2 sum_{n>=0} d^n / (n + 2)!, n = 0..12 (the next term is below 2^-60)
    constexpr double c[13] = {1.0 / 2,          1.0 / 6,           1.0 / 24,          1.0 / 120,
                              1.0 / 720,        1.0 / 5040,        1.0 / 40320,       1.0 / 362880,
                              1.0 / 3628800,    1.0 / 39916800,    1.0 / 479001600,   1.0 / 6227020800.0,
                              1.0 / 87178291200.0};
    double s = c[12];
    for (int n = 11; n >= 0; --n) s = s * d + c[n];
    return d * d * s;
  }
  return Kokkos::expm1(d) - d;
}

// log(1 + z) - z without cancellation: for |z| <= 1, with s = z/(2 + z)
// (|s| <= 1/3), log(1 + z) = 2 atanh(s) and z - 2 s = z s give
// log(1 + z) - z = -z s + 2 s^3 sum_{k>=0} s^(2k)/(2k + 3), both terms of the
// sign of the result's leading part; 18 terms reach 2^-60 at |s| = 1/3.
LAPLACE_HD double log1pmx(double z) {
  if (Kokkos::fabs(z) <= 1.0) {
    const double s = z / (2.0 + z), s2 = s * s;
    constexpr double c[18] = {1.0 / 3, 1.0 / 5, 1.0 / 7, 1.0 / 9, 1.0 / 11, 1.0 / 13, 1.0 / 15, 1.0 / 17, 1.0 / 19, 1.0 / 21, 1.0 / 23, 1.0 / 25, 1.0 / 27, 1.0 / 29, 1.0 / 31, 1.0 / 33, 1.0 / 35, 1.0 / 37};
    double p = c[17];
    for (int k = 16; k >= 0; --k) p = p * s2 + c[k];
    return -z * s + 2.0 * s * s2 * p;
  }
  return Kokkos::log1p(z) - z;
}

// psi(a) - log(a) < 0 without forming psi(a) and log(a) apart. For a >= 10 the
// asymptotic series -1/(2a) - sum_k B_2k / (2k a^2k); below, the recurrence
// psi(a) = psi(a + 1) - 1/a with log(a) = log(a + 1) - log1p(1/a) gives
//   psi(a) - log a = psi(z) - log z + sum_{j=0}^{m-1} log1pmx(1/(a + j)),  z = a + m,
// with log1pmx(u) = log1p(u) - u < 0: every term is negative.
LAPLACE_HD double psi_minus_log(double a) {
  double z = a, s = 0.0;
  while (z < 10.0) {
    s += log1pmx(1.0 / z);
    z += 1.0;
  }
  const double w = 1.0 / (z * z);
  // B_2k / (2k), k = 1..8
  double t = -3617.0 / 8160.0;
  t = t * w + 1.0 / 12.0;
  t = t * w - 691.0 / 32760.0;
  t = t * w + 1.0 / 132.0;
  t = t * w - 1.0 / 240.0;
  t = t * w + 1.0 / 252.0;
  t = t * w - 1.0 / 120.0;
  t = t * w + 1.0 / 12.0;
  return (-0.5 / z - t * w) + s;
}

// psi(a + 1) - log(a) > 0. For a >= 1 as 1/a + (psi(a) - log a), at most a
// factor two of cancellation; for a < 1 directly, where -log(a) > 0 dominates.
LAPLACE_HD double psi1p_minus_log(double a) {
  if (a >= 1.0) return 1.0 / a + psi_minus_log(a);
  return math::digamma(a + 1.0) - math::log(a);
}

// log(x/a): through log1p near x = a, where x - a is exact
LAPLACE_HD double log_ratio(double x, double a) {
  const double q = x / a;
  if (q > 0.5 && q < 2.0) return Kokkos::log1p((x - a) / a);
  if (q > 0.0 && q < math::detail::pos_inf()) return math::log(q);
  return math::log(x) - math::log(a);
}

// The side with the smaller probability: true for P.
LAPLACE_HD bool choose_lower(double a, double x) {
  if (a >= 1.0) return x < a - 1.0 / 3.0;
  if (x >= 1.0) return false;  // the median is below a < 1
  // log P ~ log(x^a e^-x / Gamma(a + 1)) + log(1 + x/(a + 1) + x^2/((a + 1)(a + 2)))
  const double lp = detail_igamma::log_prefactor(a, x) + Kokkos::log1p(x / (a + 1.0) * (1.0 + x / (a + 2.0)));
  return lp < ln_half;
}

// C(n, i), n, i <= 5
LAPLACE_HD_FORCEINLINE double binom(int n, int i) {
  constexpr double c[6][6] = {{1, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0}, {1, 2, 1, 0, 0, 0},
                              {1, 3, 3, 1, 0, 0}, {1, 4, 6, 4, 1, 0}, {1, 5, 10, 10, 5, 1}};
  return c[n][i];
}

// cumulants c[1..k] from the raw moments m[1..k] (m[0] = 1):
// c_n = m_n - sum_{i=1}^{n-1} C(n-1, i-1) c_i m_(n-i)
LAPLACE_HD void cumulants_from_moments(int k, const double* m, double* c) {
  for (int n = 1; n <= k; ++n) {
    double s = m[n];
    for (int i = 1; i < n; ++i) s -= binom(n - 1, i - 1) * c[i] * m[n - i];
    c[n] = s;
  }
}

// raw moments m[1..k] from the cumulants c[1..k] (the complete Bell
// polynomials): m_n = sum_{i=1}^{n} C(n-1, i-1) c_i m_(n-i), m_0 = 1
LAPLACE_HD void moments_from_cumulants(int k, const double* c, double* m) {
  m[0] = 1.0;
  for (int n = 1; n <= k; ++n) {
    double s = 0.0;
    for (int i = 1; i <= n; ++i) s += binom(n - 1, i - 1) * c[i] * m[n - i];
    m[n] = s;
  }
}

// d^k log(1 - S) / da^k from the derivatives of log S.
LAPLACE_HD double complement_dk(int k, const side_result& s, double a) {
  double b[7], M[7], c[7];
  moments_from_cumulants(k, s.l, b);  // b_j = R^(j) / R
  // S^(j)/S = b_j + p j b_(j-1) / a
  if (!s.lower) {
    const double ia = 1.0 / a;
    for (int j = k; j >= 1; --j) b[j] += j * ia * b[j - 1];
  }
  const double ls = s.log_s;
  // log(1 - S), and rho = S / (1 - S)
  const double l1ms = ls < ln_half ? Kokkos::log1p(-math::exp(ls)) : math::log(-Kokkos::expm1(ls));
  const double lrho = ls - l1ms;
  M[0] = 1.0;
  if (lrho > -700.0) {
    const double rho = math::exp(lrho);
    for (int j = 1; j <= k; ++j) M[j] = -rho * b[j];
  } else {
    // rho underflows: scale through the logarithm
    for (int j = 1; j <= k; ++j) {
      M[j] = b[j] == 0.0 ? 0.0 : -Kokkos::copysign(math::exp(lrho + math::log(Kokkos::fabs(b[j]))), b[j]);
    }
  }
  cumulants_from_moments(k, M, c);
  return c[k];
}

// From the cumulants kap[1..k] of d on the side and log I (the integral of
// the side relative to x^a e^-x) to the derivatives of log S.
LAPLACE_HD side_result from_cumulants(double a, double x, int k, bool lower, bool direct, double log_i,
                                      const double* kap) {
  side_result s;
  s.lower = lower;
  s.direct = direct;
  // log Q directly, or log R = log P and log(Q/a)
  const bool q = !lower && direct;
  const double z = q ? a : a + 1.0;
  if (direct) {
    // only the order asked for: no prefactor, one polygamma
    s.log_s = 0.0;
    if (k == 1) {
      const double lr = log_ratio(x, a);
      s.l[1] = q ? (lr + kap[1]) - psi_minus_log(a) : (lr + kap[1]) - psi1p_minus_log(a);
    } else {
      s.l[k] = kap[k] - math::polygamma(k - 1, z);
    }
    return s;
  }
  s.log_s = detail_igamma::log_prefactor(a, x) + log_i + (lower ? 0.0 : math::log(a));
  const double lr = log_ratio(x, a);
  s.l[1] = q ? (lr + kap[1]) - psi_minus_log(a) : (lr + kap[1]) - psi1p_minus_log(a);
  for (int j = 2; j <= k; ++j) s.l[j] = kap[j] - math::polygamma(j - 1, z);
  return s;
}

// ---------------------------------------------------------------- candidate 1: quadrature

// Weighted central moments of order 2..6 of the points seen so far, updated
// one point at a time (Pebay 2008): W the total weight, m the mean,
// M_p = sum w_i (y_i - m)^p. M6 is carried only when six is true (order 6).
struct central {
  double W = 0, m = 0, M2 = 0, M3 = 0, M4 = 0, M5 = 0, M6 = 0;
  bool six = false;
  LAPLACE_HD void add(double w, double y) {
    const double W1 = W + w;
    const double delta = y - m;
    const double e = -delta * w / W1;  // old points: their shift about the new mean
    const double f = delta + e;        // the new point about the new mean, delta W / W1
    const double e2 = e * e, e3 = e2 * e;
    const double f2 = f * f, f3 = f2 * f;
    // M_p' = sum_{j=0}^{p} C(p, j) M_(p-j) e^j + w f^p, with M_1 = 0 and M_0 = W
    if (six) M6 += 6.0 * M5 * e + 15.0 * M4 * e2 + 20.0 * M3 * e3 + 15.0 * M2 * e2 * e2 + W * e3 * e3 + w * f3 * f3;
    M5 += 5.0 * M4 * e + 10.0 * M3 * e2 + 10.0 * M2 * e3 + W * e3 * e2 + w * f3 * f2;
    M4 += 4.0 * M3 * e + 6.0 * M2 * e2 + W * e2 * e2 + w * f2 * f2;
    M3 += 3.0 * M2 * e + W * e3 + w * f3;
    M2 += W * e2 + w * f2;
    m -= e;
    W = W1;
  }
};

// Nodes of the rules for h = 1/8, written by mpmath at 200 bits and rounded
// to double (the formulas are those of ee_node and ts_node below). The host
// and the device read them, which saves two calls of exp per node.
namespace tab {
// ee1: t = j/8, j = -32..64
constexpr int ee1_n = 97;
#define LAPLACE_GD_EE1_XI {0x1.6051bb2809c87p-85, 0x1.dc980d46e7e7dp-76, 0x1.2f5efb1ba9b1ep-67, 0x1.8d2375927d08bp-60, 0x1.21075a1a03640p-53, 0x1.f52d2ef635fc5p-48, 0x1.13117658a72f9p-42, 0x1.9360e256b52bep-38, 0x1.9e528f79b26f4p-34, 0x1.36ce524730917p-30, 0x1.615f0f995dad0p-27, 0x1.3a8c33234806dp-24, 0x1.c3347140979abp-22, 0x1.0b751ef664887p-19, 0x1.0bfd0775c84e2p-17, 0x1.cef51238ff5e7p-16, 0x1.5ec9b640c10f8p-14, 0x1.d98e847e4ec1bp-13, 0x1.20a4d6bf73db0p-11, 0x1.419326deb1576p-10, 0x1.4ae620a9f3b11p-9, 0x1.3d6fc61d16f14p-8, 0x1.1e3ff0eb23d7ep-7, 0x1.e8d17c8c3391bp-7, 0x1.8dbb6bda2ce20p-6, 0x1.3624f892edf2ep-5, 0x1.d1df850ba625ep-5, 0x1.527fa9dbc678ap-4, 0x1.ddbac780be21fp-4, 0x1.4887a17445225p-3, 0x1.b9aefa10c5c60p-3, 0x1.230001a0bf611p-2, 0x1.78b56362cef38p-2, 0x1.e01733aad073ap-2, 0x1.2dba3489b3957p-1, 0x1.76aa88668482cp-1, 0x1.cc42d84cad645p-1, 0x1.18090fd4da7d6p+0, 0x1.51ebab0113024p+0, 0x1.94c4bfc2903bap+0, 0x1.e1b04d04b85c6p+0, 0x1.1cf7dfb7bb5a2p+1, 0x1.4f77b68f2f147p+1, 0x1.892658387e511p+1, 0x1.caee5cfcec5cdp+1, 0x1.0aed0d510e2dep+2, 0x1.358bf7fe29b8cp+2, 0x1.65ff8dbe59e74p+2, 0x1.9d0a98613bf84p+2, 0x1.db8a21d942889p+2, 0x1.113c7899911adp+3, 0x1.3979be9c30e1ap+3, 0x1.671e3265c7d66p+3, 0x1.9ae0bf8de05abp+3, 0x1.d590cc1285a65p+3, 0x1.0c0cbc2799935p+4, 0x1.31c2a6df5eaefp+4, 0x1.5c8135aab17fep+4, 0x1.8cf3a01a6831ap+4, 0x1.c3dbf3d9fdc7ep+4, 0x1.010b0f0e4d28dp+5, 0x1.244dae4fe7a31p+5, 0x1.4c43120778c1bp+5, 0x1.798b45b3d1ae2p+5, 0x1.acdba70a3c073p+5, 0x1.e701bc66f7736p+5, 0x1.147335ff7114cp+6, 0x1.39c8d059dc749p+6, 0x1.64173591b20adp+6, 0x1.9407d7be2519dp+6, 0x1.ca5ab96a31e1dp+6, 0x1.03f4b74fe8a88p+7, 0x1.26d5422d897e7p+7, 0x1.4e5aae03bf90ep+7, 0x1.7b23453d72005p+7, 0x1.ade265f7aa968p+7, 0x1.e763506972beep+7, 0x1.14462a79f6d49p+8, 0x1.3931374b97c7fp+8, 0x1.6306ac1d3aa97p+8, 0x1.926e16912853dp+8, 0x1.c825538e67383p+8, 0x1.0281c3d8d1b92p+9, 0x1.24fe3e7f530a9p+9, 0x1.4c1239a6cfa02p+9, 0x1.785a395d876eep+9, 0x1.aa8798b4e7e75p+9, 0x1.e363501a6b28ap+9, 0x1.11e88d22b3fb9p+10, 0x1.36698228ddc3bp+10, 0x1.5fc6bba63c48cp+10, 0x1.8ea5e5af1899ep+10, 0x1.c3c2bb72cf978p+10, 0x1.fff1f722ffc1ep+10, 0x1.221252fc0c402p+11, 0x1.48b5f6d3e6dd5p+11, 0x1.747ea93428c59p+11}
inline constexpr double ee1_xi[97] = LAPLACE_GD_EE1_XI;
#define LAPLACE_GD_EE1_W {0x1.3211325ed716bp-82, 0x1.6e40b4d0c3cc1p-73, 0x1.9c9847ad81b80p-65, 0x1.de1d35c7d8bebp-58, 0x1.3422c34e17713p-51, 0x1.d95d9b05ebd50p-46, 0x1.cc92bc0387779p-41, 0x1.2b81d682e121dp-36, 0x1.1101b3037a9fcp-32, 0x1.6bbf91d2d8ff8p-29, 0x1.6f9076c17efe8p-26, 0x1.230be7d3431f1p-23, 0x1.73c0158202a76p-21, 0x1.88dcb160fc52ep-19, 0x1.5f52ed86901f8p-17, 0x1.0f3405bd94420p-15, 0x1.6fd8f32c05497p-14, 0x1.bd31196b84d6bp-13, 0x1.e76b31180f82fp-12, 0x1.e8aaa74098e27p-11, 0x1.c578a52823b49p-10, 0x1.893b0f000552ep-9, 0x1.4157045013acap-8, 0x1.f29f052d9434bp-8, 0x1.71b85374ab8d0p-7, 0x1.078919ccb9928p-6, 0x1.6b08192581c34p-6, 0x1.e572e0d6a53f1p-6, 0x1.3c57d8335a407p-5, 0x1.9344d87b293cdp-5, 0x1.f8687e129cbecp-5, 0x1.365f853675cd8p-4, 0x1.78b56362cef38p-4, 0x1.c3e272c8d2812p-4, 0x1.0c5b3f60d3efap-3, 0x1.3c15c80d09068p-3, 0x1.71b6382dea09ap-3, 0x1.aded7394fdc59p-3, 0x1.f18aff34a36aep-3, 0x1.1ec026e2d506cp-2, 0x1.49722e3b2ba78p-2, 0x1.797be137b7d71p-2, 0x1.af94a9b2d2057p-2, 0x1.ec8db9c7ab248p-2, 0x1.18aa8ea0a9985p-1, 0x1.3f7ca0b50f385p-1, 0x1.6b568173075cbp-1, 0x1.9ce6294733451p-1, 0x1.d4f0c20a3c117p-1, 0x1.0a2ad6dfc446cp+0, 0x1.2e08fc5736709p+0, 0x1.56a22231baed7p+0, 0x1.84989d839034fp+0, 0x1.b8a44e4d3ee0bp+0, 0x1.f3957d0b97c25p+0, 0x1.1b2c0d4bb9064p+1, 0x1.40fbb7a03cb13p+1, 0x1.6bd123e7a9ed9p+1, 0x1.9c57d88b777f3p+1, 0x1.d3522a7d57009p+1, 0x1.08ce23961cb17p+2, 0x1.2c17d3090d7e8p+2, 0x1.541377d0a7453p+2, 0x1.8161349770a37p+2, 0x1.b4b67bc67ec46p+2, 0x1.eee0e63220da5p+2, 0x1.1864b541d3cc8p+3, 0x1.3dbc0120c643ap+3, 0x1.680be587364a2p+3, 0x1.97fdda547385bp+3, 0x1.ce51e7352421ep+3, 0x1.05f0d2603264fp+4, 0x1.28d1d1fd877dap+4, 0x1.5057a4f1d42a2p+4, 0x1.7d20973d1fbb2p+4, 0x1.afe008630ee09p+4, 0x1.e96139d7fb094p+4, 0x1.15453e8adb84dp+5, 0x1.3a30670a477fbp+5, 0x1.6405f44b99cccp+5, 0x1.936d7452187e8p+5, 0x1.c924c45a80e11p+5, 0x1.030184a69dec1p+6, 0x1.257e06b87df62p+6, 0x1.4c92086c86565p+6, 0x1.78da0deb11799p+6, 0x1.ab07725c9a8bcp+6, 0x1.e3e32e42f76b8p+6, 0x1.12287e33bc773p+7, 0x1.36a974faeddb3p+7, 0x1.6006b0049b0bdp+7, 0x1.8ee5db6b3cc3bp+7, 0x1.c402b263a5f89p+7, 0x1.0018f79223ae6p+8, 0x1.22324f74e8e87p+8, 0x1.48d5f3b6dd77bp+8, 0x1.749ea674c2d99p+8}
inline constexpr double ee1_w[97] = LAPLACE_GD_EE1_W;
// ee2: t = j/8, j = -32..128
constexpr int ee2_n = 161;
#define LAPLACE_GD_EE2_XI {0x1.4569d13d31008p-82, 0x1.9d86f7b7dd7fdp-73, 0x1.ee8e900c0a642p-65, 0x1.3018962966b70p-57, 0x1.9fcfb53f467ddp-51, 0x1.52aab4a1ac4ffp-45, 0x1.5d3a58630921bp-40, 0x1.e11a4d64cd94ap-36, 0x1.d037560f91a28p-32, 0x1.4722ae6ca767bp-28, 0x1.5d67162e6e584p-25, 0x1.242beca25c078p-22, 0x1.89b70504fe531p-20, 0x1.b67acd631f135p-18, 0x1.9cbb2eef6325ap-16, 0x1.4ee7210e238d5p-14, 0x1.dcc52e4beaa56p-13, 0x1.2e513e2ef95d3p-11, 0x1.5a35da4f1eb55p-10, 0x1.6a570b10efed2p-9, 0x1.5e41abb6ffcfbp-8, 0x1.3ba627f0d8abep-7, 0x1.0b647a12ca579p-6, 0x1.acf390719d490p-6, 0x1.47dfec3eb8dfbp-5, 0x1.e05c6f2dc4e98p-5, 0x1.52ebb2d935ee3p-4, 0x1.ceac3faafb4a9p-4, 0x1.32b55edb1b76fp-3, 0x1.8c4857e9e702cp-3, 0x1.f47e33d4fab97p-3, 0x1.35c49602d5525p-2, 0x1.78b56362cef38p-2, 0x1.c300df8db74c1p-2, 0x1.0a4601c1a29a7p-1, 0x1.369bf0c88594bp-1, 0x1.6673a39b6fe30p-1, 0x1.99c1ad173b228p-1, 0x1.d07f95e1f0309p-1, 0x1.0556957d10a69p+0, 0x1.2428b0f1a544dp+0, 0x1.44bd73162e5f7p+0, 0x1.6720322371213p+0, 0x1.8b60520ffe4dbp+0, 0x1.b19141bcff86ep+0, 0x1.d9ca6e9d7ce13p+0, 0x1.02139a08d9f3dp+1, 0x1.1863653837283p+1, 0x1.2fe61c5388dbbp+1, 0x1.48af253d16dd8p+1, 0x1.62d3c660ba8cdp+1, 0x1.7e6b283aff109p+1, 0x1.9b8e5a84ddb6ap+1, 0x1.ba585d1e467c9p+1, 0x1.dae62cbbe4d94p+1, 0x1.fd56d34f9e344p+1, 0x1.10e5be0e6e896p+2, 0x1.2433c5b403d91p+2, 0x1.38a85cd90bd0fp+2, 0x1.4e57993a8b568p+2, 0x1.6556db2dc0675p+2, 0x1.7dbce0795078fp+2, 0x1.97a1d8ef44a48p+2, 0x1.b31f7cce8206dp+2, 0x1.d05124f5cdd2dp+2, 0x1.ef53e4f5c4587p+2, 0x1.08235389396c7p+3, 0x1.19a525243e090p+3, 0x1.2c40e1366c501p+3, 0x1.40091d40b1315p+3, 0x1.55119cadb8e69p+3, 0x1.6b6f645e75926p+3, 0x1.8338cf819afbep+3, 0x1.9c85a5c9605cbp+3, 0x1.b76f331475b30p+3, 0x1.d41060a0c221dp+3, 0x1.f285cfe03792bp+3, 0x1.0976fb84e7e69p+4, 0x1.1ab49fc151b29p+4, 0x1.2d0d131e7ef79p+4, 0x1.4092af16afc7ep+4, 0x1.5558fa7bc27a0p+4, 0x1.6b74bcfbeb5fbp+4, 0x1.82fc13e805a60p+4, 0x1.9c0688502a032p+4, 0x1.b6ad268c92d4ep+4, 0x1.d30a974a45923p+4, 0x1.f13b3a34808d7p+4, 0x1.08aea12ac5c97p+5, 0x1.19c86a25a1d7bp+5, 0x1.2bfc1336c811fp+5, 0x1.3f5bd1837113ep+5, 0x1.53fb06607a4b2p+5, 0x1.69ee52b51b121p+5, 0x1.814bab9d513b3p+5, 0x1.9a2a7060a76a1p+5, 0x1.b4a381d34c595p+5, 0x1.d0d15b38dcdf1p+5, 0x1.eed02cc1c4862p+5, 0x1.075efbded9bddp+6, 0x1.185d564faef60p+6, 0x1.2a742577c53d3p+6, 0x1.3db581a7b2880p+6, 0x1.5234add60d626p+6, 0x1.68062ae3fa96ap+6, 0x1.7f3fcc1fc08c6p+6, 0x1.97f8cd19e48b5p+6, 0x1.b249e8e2a8b94p+6, 0x1.ce4d72c72a2bfp+6, 0x1.ec1f70a6de2dap+6, 0x1.05eedb7de32a1p+7, 0x1.16d40358b2f50p+7, 0x1.28d0167429096p+7, 0x1.3bf51263116b2p+7, 0x1.50561db9cbe94p+7, 0x1.66079b3678d7cp+7, 0x1.7d1f3e2569ef0p+7, 0x1.95b420163d724p+7, 0x1.afded7f75a861p+7, 0x1.cbb992aeec1eep+7, 0x1.e9602d49f58dfp+7, 0x1.0478286dd8f80p+8, 0x1.1544c8149016cp+8, 0x1.2726c39f34c94p+8, 0x1.3a2ffe86dd829p+8, 0x1.4e73839c99406p+8, 0x1.64059815d6dfcp+8, 0x1.7afbcfd34b062p+8, 0x1.936d22f69d351p+8, 0x1.ad7204dc7382fp+8, 0x1.c9247c91d8ed4p+8, 0x1.e6a03edd75b5cp+8, 0x1.030164fb5296ep+9, 0x1.13b5c183112a3p+9, 0x1.257deac6e7355p+9, 0x1.386baa6b7df07p+9, 0x1.4c91efc457598p+9, 0x1.6204e2c4b5b50p+9, 0x1.78d9f8293cdd4p+9, 0x1.912806ee78afcp+9, 0x1.ab075f29c0e8fp+9, 0x1.c691e25b55394p+9, 0x1.e3e31d5205b79p+9, 0x1.018c31dd26c03p+10, 0x1.122876ba3874dp+10, 0x1.23d6fae77bed1p+10, 0x1.36a96e6260ad7p+10, 0x1.4ab2a52ff89c8p+10, 0x1.6006aa328ecd3p+10, 0x1.76bad336356b6p+10, 0x1.8ee5d64859030p+10, 0x1.a89fe06fb27e3p+10, 0x1.c402addb51b01p+10, 0x1.e129a3a0f17a5p+10, 0x1.0018f5922b044p+11, 0x1.109d47a18f633p+11, 0x1.22324db11d294p+11, 0x1.34e99e3e063bbp+11, 0x1.48d5f2282b897p+11, 0x1.5e0b376c8c612p+11, 0x1.749ea514eca8fp+11}
inline constexpr double ee2_xi[161] = LAPLACE_GD_EE2_XI;
#define LAPLACE_GD_EE2_W {0x1.1826b1b091a94p-79, 0x1.3a8e7ba836daap-70, 0x1.4c717d42858f4p-62, 0x1.6959579f0ce3cp-55, 0x1.b4cd7e83a4192p-49, 0x1.3a95076d8726ap-43, 0x1.1eea905ed7eacp-38, 0x1.5db331d9f5e7fp-34, 0x1.2aa11c5791f4cp-30, 0x1.74a2ec949fe89p-27, 0x1.6084764911b73p-24, 0x1.05363d9455a7cp-21, 0x1.381497f59d64ap-19, 0x1.3455741baa6b0p-17, 0x1.01a40030b15bep-15, 0x1.7371e060eaf75p-14, 0x1.d6288c80fe2a0p-13, 0x1.09507eb4c5162p-11, 0x1.0ead107535affp-10, 0x1.f9523401c9e90p-10, 0x1.b4379f7d2cdb2p-9, 0x1.5f8f67fbdebc5p-8, 0x1.0abf36f87b3b9p-7, 0x1.7fef4d9931c44p-7, 0x1.07cc5d2215b79p-6, 0x1.5c20701a5a595p-6, 0x1.bb7a4ce54ea4bp-6, 0x1.11ee473e32ea4p-5, 0x1.4983f8dadb80ap-5, 0x1.835d51f89aa98p-5, 0x1.be72336c77048p-5, 0x1.f9e5a11b3a88bp-5, 0x1.1a880a8a1b36ap-4, 0x1.37c162842d8bbp-4, 0x1.5482be32237c7p-4, 0x1.70c86c78e22c0p-4, 0x1.8ca347b4aa36bp-4, 0x1.a834849e92115p-4, 0x1.c3a9a60103c4ap-4, 0x1.df38e06004c88p-4, 0x1.fb1e13fbe8284p-4, 0x1.0bcc344cd4f86p-3, 0x1.1a744250b0170p-3, 0x1.29a7a76325e53p-3, 0x1.39869251a0c57p-3, 0x1.4a30b7dce77f9p-3, 0x1.5bc541ceb1cd6p-3, 0x1.6e62d9575976ep-3, 0x1.8227c4f2cae67p-3, 0x1.973213a318564p-3, 0x1.ad9fd0bd690efp-3, 0x1.c58f3cc285ba3p-3, 0x1.df1f08c1e7b73p-3, 0x1.fa6e9296ea1cap-3, 0x1.0bcf10789a335p-2, 0x1.1b678f4972dd3p-2, 0x1.2c122a43ea736p-2, 0x1.3de111ac21f0cp-2, 0x1.50e75db254915p-2, 0x1.65392a5a553fap-2, 0x1.7aebb34a8c82bp-2, 0x1.92156fb569a2ap-2, 0x1.aace2e92ab2dbp-2, 0x1.c52f335ea325ap-2, 0x1.e153539554a7fp-2, 0x1.ff57151e76ac9p-2, 0x1.0fac66ef2ffedp-1, 0x1.20bc624f6f7f9p-1, 0x1.32eca9c115c9bp-1, 0x1.464f8618c9a1dp-1, 0x1.5af86efca3603p-1, 0x1.70fc1f0169d91p-1, 0x1.8870a8f02d137p-1, 0x1.a16d8e4db917ap-1, 0x1.bc0bd73dfdb23p-1, 0x1.d8662bde598f4p-1, 0x1.f698ef32b67ffp-1, 0x1.0b612de14885cp+0, 0x1.1c8151022ee8cp+0, 0x1.2ebe045ba198bp+0, 0x1.422987a666776p+0, 0x1.56d749430d1e3p+0, 0x1.6cdbf9b50696dp+0, 0x1.844da05e1aec0p+0, 0x1.9d43b18f4a432p+0, 0x1.b7d726056f5c9p+0, 0x1.d42293e95a0c8p+0, 0x1.f242496c927b0p+0, 0x1.092a348ec57c3p+1, 0x1.1a3c8407db9fep+1, 0x1.2c6926fae7502p+1, 0x1.3fc24ba736284p+1, 0x1.545b4ce2a7fbbp+1, 0x1.6a48c576d90f4p+1, 0x1.81a0a4be3b762p+1, 0x1.9a7a4495c3b64p+1, 0x1.b4ee80b9209f9p+1, 0x1.d117cfa0e02a1p+1, 0x1.ef125cfb649f4p+1, 0x1.077e12ed93de9p+2, 0x1.187a8b5b3f44dp+2, 0x1.2a8f95ab70eb3p+2, 0x1.3dcf486bd76c2p+2, 0x1.524ce4eb314ddp+2, 0x1.681cea7c4248dp+2, 0x1.7f552af6e515bp+2, 0x1.980ce08bbcdf7p+2, 0x1.b25cc5005c275p+2, 0x1.ce5f2a6b1f6bcp+2, 0x1.ec30158779afdp+2, 0x1.05f6acde84c4bp+3, 0x1.16db5b7ac0226p+3, 0x1.28d6fcaf99ce9p+3, 0x1.3bfb8d9e0fac8p+3, 0x1.505c346f94585p+3, 0x1.660d537dd1e98p+3, 0x1.7d249db6b7263p+3, 0x1.95b92c5134edep+3, 0x1.afe395e8626b7p+3, 0x1.cbbe071425255p+3, 0x1.e9645c97f7a1ep+3, 0x1.047a1fa0ffcccp+4, 0x1.1546a0cb2a905p+4, 0x1.27287fb21445dp+4, 0x1.3a319fb22f6b2p+4, 0x1.4e750b81a2fcdp+4, 0x1.6407083c96f09p+4, 0x1.7afd29ac00e28p+4, 0x1.936e67db39736p+4, 0x1.ad733611f1b95p+4, 0x1.c9259b4984a37p+4, 0x1.e6a14c361bab5p+4, 0x1.0301e37ed9129p+5, 0x1.13b6385c58409p+5, 0x1.257e5a6cd147ap+5, 0x1.386c134db98c8p+5, 0x1.4c92524bceeb7p+5, 0x1.62053f53f8757p+5, 0x1.78da4f1ce164bp+5, 0x1.9128589d79917p+5, 0x1.ab07abe5d3b1cp+5, 0x1.c6922a713c29fp+5, 0x1.e3e36109dc8b3p+5, 0x1.018c51abe8c65p+6, 0x1.1228949ba2701p+6, 0x1.23d716f971b80p+6, 0x1.36a988c0f689ep+6, 0x1.4ab2bdf58f49fp+6, 0x1.6006c177edeb7p+6, 0x1.76bae912a443dp+6, 0x1.8ee5ead1b5d1dp+6, 0x1.a89ff3ba88523p+6, 0x1.c402bffaecf71p+6, 0x1.e129b4a7734f7p+6, 0x1.0018fd91632b4p+7, 0x1.109d4f24bea85p+7, 0x1.223254bfc75b4p+7, 0x1.34e9a4df3a9abp+7, 0x1.48d5f8628bd6bp+7, 0x1.5e0b3d4653814p+7, 0x1.749eaa93f4dfap+7}
inline constexpr double ee2_w[161] = LAPLACE_GD_EE2_W;
// tanh-sinh: t = j/8, j = 0..28: e/(1 + e) and h pi/2 cosh(t) 2e/(1 + e)^2, e = exp(-pi sinh t)
constexpr int ts_n = 29;
#define LAPLACE_GD_TS_NEAR {0x1.0000000000000p-1, 0x1.9c7d3d25f111dp-2, 0x1.3ede5d7db49bbp-2, 0x1.d7e9ed12733b7p-3, 0x1.4d8bc620abe11p-3, 0x1.c150e3146c370p-4, 0x1.1f9a41e22fe6cp-4, 0x1.5ca78fc7d75bep-5, 0x1.8e64c57b0deaep-6, 0x1.aa86de6d91c83p-7, 0x1.a8a7a865cc9f6p-8, 0x1.85940c9c02bb1p-9, 0x1.45bb947bf2ce4p-10, 0x1.ea1b353465bd9p-12, 0x1.46e0fcd9f277ap-13, 0x1.7c117155fe2f5p-15, 0x1.79deb8c754ecap-17, 0x1.3a55cc9176340p-19, 0x1.aaf41943aeea2p-22, 0x1.cc7fdd25c1e85p-25, 0x1.7e3aafbf01aacp-28, 0x1.d74d883823f4ap-32, 0x1.9eadd759ab50cp-36, 0x1.f199ea24b739fp-41, 0x1.82c89ccd8b24cp-46, 0x1.6f7c9a07a6f7dp-52, 0x1.8f92aac33e360p-59, 0x1.cd7532b66177cp-67, 0x1.0416803c41fa4p-75}
inline constexpr double ts_near[29] = LAPLACE_GD_TS_NEAR;
#define LAPLACE_GD_TS_W {0x1.921fb54442d18p-4, 0x1.85f5ebfa10b92p-4, 0x1.63bdcaf901815p-4, 0x1.3186caf901e08p-4, 0x1.ee947b728e394p-5, 0x1.799174231745ep-5, 0x1.0fe97de128bdbp-5, 0x1.711ccc4f2eb22p-6, 0x1.d715fb2e4f1bcp-7, 0x1.1970c313b0d53p-7, 0x1.38e041e84722ap-8, 0x1.40ebf0ed4ce33p-9, 0x1.2c88d194ea674p-10, 0x1.fb6b4a3163556p-12, 0x1.7c70559b49937p-13, 0x1.f20bb3a8a0696p-15, 0x1.1721a8030e921p-16, 0x1.0612231ac7148p-18, 0x1.921b35fc363d2p-21, 0x1.ea40955e09cf9p-24, 0x1.cc3b4b048a3b1p-27, 0x1.410b895d92343p-30, 0x1.3fb70d9c5c84bp-34, 0x1.b256959c5ec40p-39, 0x1.7e4b11db90a01p-44, 0x1.9b5b831ff30b3p-50, 0x1.fa9caab4cca6dp-57, 0x1.4b60d436e990cp-64, 0x1.a72bf17fb33bep-73}
inline constexpr double ts_w[29] = LAPLACE_GD_TS_W;
#if defined(__CUDACC__)
// the same tables in the device's constant memory (one copy per translation
// unit; every thread of a warp reads the same node at the same time)
static __constant__ double d_ee1_xi[97] = LAPLACE_GD_EE1_XI;
static __constant__ double d_ee1_w[97] = LAPLACE_GD_EE1_W;
static __constant__ double d_ee2_xi[161] = LAPLACE_GD_EE2_XI;
static __constant__ double d_ee2_w[161] = LAPLACE_GD_EE2_W;
static __constant__ double d_ts_near[29] = LAPLACE_GD_TS_NEAR;
static __constant__ double d_ts_w[29] = LAPLACE_GD_TS_W;
#endif
}  // namespace tab

// log of the lower integrand: r d - x (e^d - 1 - d)
struct log_integrand {
  double r, x;
  LAPLACE_HD_FORCEINLINE double operator()(double d) const { return r * d - x * expm1mx(d); }
};

// exp inside the quadrature: E = 0 laplace::math::exp (the default), E = 1
// laplace::math::own::exp (a variant for the measurement: 4-6 ns against
// about 28 ns for the Rtools libm, error up to 1 ulp; the choice of the
// elementary functions is open)
template <int E>
LAPLACE_HD_FORCEINLINE double qexp(double v) {
  if constexpr (E == 1) {
    return own::exp(v);
  } else {
    return math::exp(v);
  }
}

// node j of the exp-exp rule: xi = exp(ip t - e^-t), weight h (ip + e^-t) xi,
// t = j h, ip = 1 (gauss false) or 1/2 (gauss true)
template <int NH, int E>
LAPLACE_HD_FORCEINLINE void ee_node(int j, bool gauss, double& xi, double& wq) {
  if constexpr (NH == 8) {
    const int i = j + 32;
    KOKKOS_IF_ON_HOST((if (!gauss) {
      if (i < tab::ee1_n) {
        xi = tab::ee1_xi[i];
        wq = tab::ee1_w[i];
        return;
      }
    } else if (i < tab::ee2_n) {
      xi = tab::ee2_xi[i];
      wq = tab::ee2_w[i];
      return;
    }))
    KOKKOS_IF_ON_DEVICE((if (!gauss) {
      if (i < tab::ee1_n) {
        xi = tab::d_ee1_xi[i];
        wq = tab::d_ee1_w[i];
        return;
      }
    } else if (i < tab::ee2_n) {
      xi = tab::d_ee2_xi[i];
      wq = tab::d_ee2_w[i];
      return;
    }))
  }
  constexpr double h = 1.0 / NH;
  const double ip = gauss ? 0.5 : 1.0;
  const double t = j * h;
  const double iE = qexp<E>(-t);
  xi = qexp<E>(ip * t - iE);
  wq = h * (ip + iE) * xi;
}

// node j of tanh-sinh on [0, 1]: the distance to the nearer end e/(1 + e),
// e = exp(-pi sinh|t|), and the weight h pi/2 cosh(t) 2e/(1 + e)^2
template <int NH, int E>
LAPLACE_HD_FORCEINLINE void ts_node(int j, double& near, double& wq) {
  const int aj = j < 0 ? -j : j;
  if constexpr (NH == 8) {
    KOKKOS_IF_ON_HOST((if (aj < tab::ts_n) {
      near = tab::ts_near[aj];
      wq = tab::ts_w[aj];
      return;
    }))
    KOKKOS_IF_ON_DEVICE((if (aj < tab::ts_n) {
      near = tab::d_ts_near[aj];
      wq = tab::d_ts_w[aj];
      return;
    }))
  }
  constexpr double h = 1.0 / NH;
  const double t = aj * h;
  const double E1 = qexp<E>(t), iE = 1.0 / E1;
  const double e = qexp<E>(-2.0 * half_pi * 0.5 * (E1 - iE));
  near = e / (1.0 + e);
  wq = h * half_pi * 0.5 * (E1 + iE) * 2.0 * e / ((1.0 + e) * (1.0 + e));
}

// Where a half-line rule starts and when it stops. The first node is
// t = -29/8: below it xi (1 + e^-t) h < 1e-17 (2^-56). The rule stops at the
// first node past t = 0 whose contribution w (1 + xi)^5 to the moments of
// order up to five (w (1 + xi)^6 for order six) is below cut times the running weight (cut = 1e-25 before
// 2026-11-09 evening; the measured choice is below).
constexpr int ee_first = -29;
constexpr double ee_cut = 1e-20;

// The exp-exp rule of Takahasi and Mori, for integrands that decay
// exponentially: nodes xi_j = exp(t - e^-t), weights h (1 + e^-t) xi_j,
// and its variant xi_j = exp(t/2 - e^-t) where the integrand falls like a
// Gaussian (curvature larger than the squared slope at the end point: with
// exp(t - e^-t) the strip where exp(-xi^2) stays bounded is half as wide, and
// h = 1/8 gave 1e-11 relative on the fifth cumulant at a = 474, x = 468),
// t = j h, h = 1/NH. (Exp-sinh, xi = exp(pi/2 sinh t), was tried first: on an
// exponentially decaying integrand its strip of analyticity narrows and it
// converged as exp(-1.9/h), 1e-10 relative at h = 1/8.)
//
// The lower side in d, from 0 towards -infinity, scale sc; the integrand is
// x exp(r d - x (e^d - 1 - d)), r = a + 1 - x.
template <int NH, int E, int S = 1, class C = central>
LAPLACE_HD void lower_rule(C& c, double r, double x) {
  const double sc = 1.0 / Kokkos::sqrt(r * r + x);
  const bool gauss = r * r < x;
  const double f = sc * x * S;
  for (int j = (ee_first * NH / 8) / S * S; j <= 16 * NH; j += S) {
    double xi, wq;
    ee_node<NH, E>(j, gauss, xi, wq);
    const double d = -(sc * xi);
    const double w = (wq * f) * qexp<E>(r * d - x * expm1mx(d));
    if (w > 0.0) c.add(w, d);
    if (j > 0) {
      const double z = 1.0 + xi, z2 = z * z;
      const double wz = c.six ? w * z2 * z2 * z2 : w * z2 * z2 * z;
      if (!(wz > ee_cut * c.W)) break;
    }
  }
}

// The upper side beyond t0 (t0 = x, or 1 when x < 1/4) in u = t - t0, where
// t = x e^d is the original variable: there the integrand decays like e^-u,
// while in d it decays double-exponentially and the exp-exp rule in d converged
// slowly (1e-7 relative at h = 1/8 for a = 1.6, x = 1.4). With z = u/t0,
// d = d0 + log1p(z) and the integrand exp(a d - (t - x)) dd = exp(a d - (t0 - x) - u) du / t.
// For t0 = x the exponent, with the factor 1/t = exp(-log1p(z))/x folded in, is
// (a - 1 - x) z + (a - 1) log1pmx(z), without the cancellation of a d against u
// near x = a.
template <int NH, int E, int S = 1, class C = central>
LAPLACE_HD void upper_rule(C& c, double a, double x, double t0, double d0) {
  const double slope = (a - 1.0) / t0 - 1.0;
  const double curv = a > 1.0 ? (a - 1.0) / (t0 * t0) : 0.0;
  const double sc = 1.0 / Kokkos::sqrt(slope * slope + curv);
  const bool gauss = slope * slope < curv;
  const bool at_x = t0 == x;
  const double it0 = 1.0 / t0;
  const double f = sc * it0 * S;
  const double base = a * d0 - (t0 - x);  // the exponent at u = 0 when t0 = 1 (log t0 = 0)
  const double r = a - 1.0 - x;
  for (int j = (ee_first * NH / 8) / S * S; j <= 16 * NH; j += S) {
    double xi, wq;
    ee_node<NH, E>(j, gauss, xi, wq);
    const double u = sc * xi;
    const double z = u * it0;
    double d, ex;
    if (z <= 1.0) {
      const double lm = log1pmx(z);
      d = z + lm;
      ex = at_x ? r * z + (a - 1.0) * lm : base + (a - 1.0) * d - u;
    } else {
      d = Kokkos::log1p(z);
      ex = at_x ? (a - 1.0) * d - u : base + (a - 1.0) * d - u;
    }
    d += d0;
    const double w = (wq * f) * qexp<E>(ex);
    if (w > 0.0) c.add(w, d);
    if (j > 0) {
      const double zz = 1.0 + xi, z2 = zz * zz;
      const double wz = c.six ? w * z2 * z2 * z2 : w * z2 * z2 * zz;
      if (!(wz > ee_cut * c.W)) break;
    }
  }
}

// Tanh-sinh on [d0, d0 + L] in d: d = d0 + L/2 (1 + tanh(pi/2 sinh t)), the
// distance to the nearer end computed directly, t in [-3.5, 3.5]; integrand
// exp(r d - x (e^d - 1 - d)).
template <int NH, int E, int S = 1, class C = central>
LAPLACE_HD void tanh_sinh_quad(C& c, double d0, double L, const log_integrand& g) {
  const int J = (7 * NH) / 2 / S * S;
  for (int j = -J; j <= J; j += S) {
    double fr, wq;
    ts_node<NH, E>(j, fr, wq);
    const double near = L * fr;
    const double d = d0 + (j < 0 ? near : L - near);
    const double w = (wq * S) * L * qexp<E>(g(d));
    if (w > 0.0) c.add(w, d);
  }
}

// The cumulants kappa_k of d on the side and log I by the quadrature; kappa_6
// only for k = 6 (kap has 7 elements).
template <int NH, int E, int SL = 1, int SU = 1, int ST = 1, int ST2 = 1, int SPLIT = 3>
LAPLACE_HD void quad_moments(double a, double x, bool lower, double* kap, double& log_i, int k = 5) {
  central c;
  c.six = k >= 6;
  if (lower) {
    c.W = 1.0;  // the point mass at d = 0
    const double r = (a - x) + 1.0;
    lower_rule<NH, E, SL>(c, r, x);
  } else if (x < 0.25) {
    // flat in d up to d = log(1/x), where t = 1; then the fall like e^-t.
    // The last 3 units of d before L, where e^-t starts to fall, get a panel of
    // their own: on one panel [0, L] the image of the tanh-sinh strip reaches
    // where exp(-x e^d) is huge, and h = 1/8 gave 4e-12 relative at L = 26
    // (1e-15 with the split).
    const double L = -math::log(x);
    const log_integrand g{a - x, x};
    if (L > SPLIT + 1.0) {
      tanh_sinh_quad<NH, E, ST>(c, 0.0, L - SPLIT, g);
      tanh_sinh_quad<NH, E, ST2>(c, L - SPLIT, SPLIT, g);
    } else {
      tanh_sinh_quad<NH, E, ST2>(c, 0.0, L, g);
    }
    upper_rule<NH, E, SU>(c, a, x, 1.0, L);
  } else {
    upper_rule<NH, E, SU>(c, a, x, x, 0.0);
  }
  const double W = c.W;
  kap[1] = c.m;
  kap[2] = c.M2 / W;
  kap[3] = c.M3 / W;
  kap[4] = c.M4 / W - 3.0 * kap[2] * kap[2];
  kap[5] = c.M5 / W - 10.0 * kap[3] * kap[2];
  // kappa_6 = mu_6 - 15 mu_4 mu_2 - 10 mu_3^2 + 30 mu_2^3 (mu_p the central moments)
  if (c.six) kap[6] = c.M6 / W - 15.0 * (c.M4 / W) * kap[2] - 10.0 * kap[3] * kap[3] + 30.0 * kap[2] * kap[2] * kap[2];
  log_i = math::log(W);
}

template <int SL, int SU, int ST, int ST2, int SPLIT = 3>
LAPLACE_HD side_result quad_side_s(double a, double x, int k, bool lower, bool direct) {
  double kap[7], log_i;
  quad_moments<8, 0, SL, SU, ST, ST2, SPLIT>(a, x, lower, kap, log_i, k);
  return from_cumulants(a, x, k, lower, direct, log_i, kap);
}

template <int NH, int E>
LAPLACE_HD side_result quad_side(double a, double x, int k, bool lower, bool direct) {
  double kap[7], log_i;
  quad_moments<NH, E>(a, x, lower, kap, log_i, k);
  return from_cumulants(a, x, k, lower, direct, log_i, kap);
}

// ---------------------------------------------------------------- the series

// The two series share one form: I x^s = sum_n tau_n, tau_0 = 1,
// tau_n = tau_(n-1) q_n(a), and the Taylor coefficients in a of
// tau_n e^(-mu (a' - a)) (mu the mean of a first pass, so that the first
// moment is near zero and the cumulants do not cancel).
//
// Lower: I_L = sum_n prod_{j<=n} x/(a + j); q = x/(a + n), z = -1/(a + n),
// q^(m)/m! = q z^m, hence T_k(n) = q U_k, U_k = T_k(n-1) + z U_(k-1).
// Upper, the asymptotic series of the upper incomplete gamma function,
// x I_U = sum_n prod_{j<=n} (a - j)/x; q = (a - n)/x is linear in a, hence
// T_k(n) = q T_k(n-1) + T_(k-1)(n-1)/x. For x > a its terms decrease to the
// smallest one, of order e^-x relative, near n = a + x; it is used only where
// it reaches 2^-73 relative within nmax terms (and x >= 36).
//
// Returns false when the first pass has not converged within nmax terms; the
// caller then uses the quadrature.
LAPLACE_HD bool series_moments(double a, double x, bool lower, int k, int nmax, double* kap, double& log_i) {
  constexpr double tol = 1e-22;
  const double ix = 1.0 / x;
  // first pass: S0 = sum tau_n, S1 = sum of the first Taylor coefficients
  double t0 = 1.0, t1 = 0.0, s0 = 1.0, s1 = 0.0;
  int n_end = -1;
  if (lower) {
    for (int n = 1; n <= nmax; ++n) {
      const double z = -1.0 / (a + n), q = -x * z;
      t1 = q * (t1 + z * t0);
      t0 = q * t0;
      s0 += t0;
      s1 += t1;
      if (t0 <= tol * s0) {
        n_end = n;
        break;
      }
    }
  } else {
    for (int n = 1; n <= nmax; ++n) {
      const double q = (a - n) * ix;
      if (q <= -1.0) break;  // past the smallest term: the series diverges
      t1 = q * t1 + t0 * ix;
      t0 = q * t0;
      s0 += t0;
      s1 += t1;
      if (Kokkos::fabs(t0) <= tol * s0 && Kokkos::fabs(t1) <= tol * (Kokkos::fabs(s1) + s0)) {
        n_end = n;
        break;
      }
    }
  }
  if (n_end < 0) return false;
  const double mu = s1 / s0;
  double T[7], S[7];
  T[0] = 1.0;
  for (int i = 1; i <= k; ++i) T[i] = T[i - 1] * (-mu) / i;
  for (int i = 0; i <= k; ++i) S[i] = T[i];
  if (lower) {
    for (int n = 1; n <= n_end; ++n) {
      const double z = -1.0 / (a + n), q = -x * z;
      double U = T[0];
      T[0] = q * U;
      for (int i = 1; i <= k; ++i) {
        U = T[i] + z * U;
        T[i] = q * U;
      }
      for (int i = 0; i <= k; ++i) S[i] += T[i];
    }
  } else {
    for (int n = 1; n <= n_end; ++n) {
      const double q = (a - n) * ix;
      for (int i = k; i >= 1; --i) T[i] = q * T[i] + T[i - 1] * ix;
      T[0] = q * T[0];
      for (int i = 0; i <= k; ++i) S[i] += T[i];
    }
  }
  double m[7];
  m[0] = 1.0;
  double fact = 1.0;
  for (int i = 1; i <= k; ++i) {
    fact *= i;
    m[i] = fact * S[i] / S[0];
  }
  cumulants_from_moments(k, m, kap);
  kap[1] += mu;
  // S[0] is the sum at a (the factor e^(-mu (a' - a)) is 1 there)
  log_i = math::log(S[0]) - (lower ? 0.0 : math::log(x));
  return true;
}

LAPLACE_HD side_result series_lower(double a, double x, int k, bool direct) {
  double kap[7], log_i;
  series_moments(a, x, true, k, 1000000, kap, log_i);
  return from_cumulants(a, x, k, true, direct, log_i, kap);
}

// ---------------------------------------------------------------- candidate: the continued fraction

// I_U = 1 / K, K = b_0 + a_1/(b_1 + a_2/(b_2 + ...)), b_n = x + 2n + 1 - a,
// a_n = n (a - n). Convergents K_n = A_n / B_n with A_n = b_n A_(n-1) + a_n A_(n-2);
// with b_n' = -1 and a_n' = n, the Taylor coefficients in a obey
// A_n[k] = b_n A_(n-1)[k] - A_(n-1)[k-1] + a_n A_(n-2)[k] + n A_(n-2)[k-1].
// The four arrays are rescaled by A_n[0] at every step.
LAPLACE_HD side_result fraction_upper(double a, double x, int k, bool direct) {
  double Am[7] = {1, 0, 0, 0, 0, 0}, Bm[7] = {0, 0, 0, 0, 0, 0, 0};
  double A[7] = {x + 1.0 - a, -1.0, 0, 0, 0, 0}, B[7] = {1, 0, 0, 0, 0, 0};
  double g_prev[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    const double s = 1.0 / A[0];
    for (int i = 0; i <= k; ++i) A[i] *= s, B[i] *= s, Am[i] *= s;
  }
  for (int n = 1; n < 100000; ++n) {
    const double bn = x + (2.0 * n + 1.0) - a;
    const double an = n * (a - n);
    const double dn = n;
    double An[7], Bn[7];
    An[0] = bn * A[0] + an * Am[0];
    Bn[0] = bn * B[0] + an * Bm[0];
    for (int i = 1; i <= k; ++i) {
      An[i] = bn * A[i] - A[i - 1] + an * Am[i] + dn * Am[i - 1];
      Bn[i] = bn * B[i] - B[i - 1] + an * Bm[i] + dn * Bm[i - 1];
    }
    const double s = 1.0 / An[0];
    bool done = true;
    double g[7];
    for (int i = 0; i <= k; ++i) {
      Am[i] = A[i] * s;
      Bm[i] = B[i] * s;
      A[i] = An[i] * s;
      B[i] = Bn[i] * s;
    }
    const double iB = 1.0 / B[0];
    g[0] = B[0];
    if (!(Kokkos::fabs(g[0] - g_prev[0]) <= 16.0 * eps * Kokkos::fabs(g[0]))) done = false;
    for (int i = 1; i <= k; ++i) {
      const double p = B[i] * iB, q = A[i];
      g[i] = p - q;
      if (!(Kokkos::fabs(g[i] - g_prev[i]) <= 16.0 * eps * (Kokkos::fabs(p) + Kokkos::fabs(q)))) done = false;
    }
    for (int i = 0; i <= k; ++i) g_prev[i] = g[i];
    if (done && n > 2) break;
  }
  double mA[7], mB[7], kA[7], kB[7], kap[7];
  mA[0] = mB[0] = 1.0;
  double fact = 1.0;
  for (int i = 1; i <= k; ++i) {
    fact *= i;
    mA[i] = fact * A[i] / A[0];
    mB[i] = fact * B[i] / B[0];
  }
  cumulants_from_moments(k, mA, kA);
  cumulants_from_moments(k, mB, kB);
  for (int i = 1; i <= k; ++i) kap[i] = kB[i] - kA[i];
  return from_cumulants(a, x, k, false, direct, math::log(B[0] / A[0]), kap);
}

LAPLACE_HD side_result series_side(double a, double x, int k, bool lower, bool direct) {
  if (lower) return series_lower(a, x, k, direct);
  if (x >= 1.0) return fraction_upper(a, x, k, direct);
  return quad_side<8, 0>(a, x, k, false, direct);
}

// ---------------------------------------------------------------- the upper side for small x

// For x < 3/2 and a < 2 the upper side in closed form, from the series that
// gamma_inc uses for a < 1, x < 1.5:
//   Q = -expm1(L) - g J,  L = a log x - log Gamma(1 + a),  g = e^L,
//   J = a K,  K = sum_{n>=1} (-x)^n / (n! (a + n)).
// With R = Q / a (the pole 1/a^j of the derivatives of log Q taken out) and
// -expm1(L)/a = (-L/a) e^(L/2) sinhc(L/2), sinhc(z) = sinh(z)/z,
//   R = e^(L/2) F,  F = (-L/a) sinhc(L/2) - e^(L/2) K,
// so log R = L/2 + log F. The large first derivative of L (log x - psi(a + 1),
// up to about -30) enters only through L/2 and through sinhc, which is even and
// so carries no drift; F has Taylor coefficients in a of moderate size, and the
// cumulants of log F from its moments do not cancel. Here
//   L/a = log x - lambda(a), lambda(a) = log Gamma(1 + a)/a
//       = -gamma + sum_{k>=2} (-1)^k zeta(k) a^(k-1)/k   (a < 1/2),
// and for a >= 1/2 lambda's Taylor coefficients follow from those of
// log Gamma(1 + a) by the division by a + h (an error growth of at most 2^j).
// It replaces the quadrature of the upper side for x < 1/4 (two tanh-sinh
// panels and an exp-exp rule, about 160 nodes) and for 1/4 <= x < 3/2.
namespace tab {
// c_1 = -gamma, c_k = (-1)^k zeta(k)/k, k = 2..91 (mpmath at 200 bits)
#define LAPLACE_GD_LGAM_C {-0x1.2788cfc6fb619p-1, 0x1.a51a6625307d3p-1, -0x1.9a4d55beab2d7p-2, 0x1.151322ac7d848p-2, -0x1.a8b9c17aa6149p-3, 0x1.5b40cb100c306p-3, -0x1.2703a1dcea3aep-3, 0x1.010b36af86397p-3, -0x1.c806706d57db4p-4, 0x1.9a01e385d5f8fp-4, -0x1.748c33114c6d6p-4, 0x1.556ad63243bc4p-4, -0x1.3b1d971fc5985p-4, 0x1.2496df8320c5fp-4, -0x1.11133476e7fe0p-4, 0x1.00010064cdeb2p-4, -0x1.e1e2d311e8abdp-5, 0x1.c71ce3a20b419p-5, -0x1.af28a1b5688a0p-5, 0x1.9999b3352d5bap-5, -0x1.86186db77bfbfp-5, 0x1.745d1d1778df9p-5, -0x1.642c88591b66dp-5, 0x1.555556aaafdcdp-5, -0x1.47ae151eb9fb7p-5, 0x1.3b13b189d925ep-5, -0x1.2f684c00002bcp-5, 0x1.24924936db7bcp-5, -0x1.1a7b961a7b9aap-5, 0x1.111111155556dp-5, -0x1.08421086318cep-5, 0x1.0000000100002p-5, -0x1.f07c1f08ba2eap-6, 0x1.e1e1e1e25a5a6p-6, -0x1.d41d41d457c58p-6, 0x1.c71c71c738e39p-6, -0x1.bacf914c29837p-6, 0x1.af286bca21af3p-6, -0x1.a41a41a41d89ep-6, 0x1.999999999b333p-6, -0x1.8f9c18f9c2577p-6, 0x1.8618618618c31p-6, -0x1.7d05f417d08eep-6, 0x1.745d1745d18bap-6, -0x1.6c16c16c16ccdp-6, 0x1.642c8590b21bdp-6, -0x1.5c9882b931083p-6, 0x1.555555555556bp-6, -0x1.4e5e0a72f0544p-6, 0x1.47ae147ae1480p-6, -0x1.4141414141417p-6, 0x1.3b13b13b13b15p-6, -0x1.3521cfb2b78c2p-6, 0x1.2f684bda12f69p-6, -0x1.29e4129e4129ep-6, 0x1.2492492492492p-6, -0x1.1f7047dc11f70p-6, 0x1.1a7b9611a7b96p-6, -0x1.15b1e5f75270dp-6, 0x1.1111111111111p-6, -0x1.0c9714fbcda3bp-6, 0x1.0842108421084p-6, -0x1.0410410410410p-6, 0x1.0000000000000p-6, -0x1.f81f81f81f820p-7, 0x1.f07c1f07c1f08p-7, -0x1.e9131abf0b767p-7, 0x1.e1e1e1e1e1e1ep-7, -0x1.dae6076b981dbp-7, 0x1.d41d41d41d41dp-7, -0x1.cd85689039b0bp-7, 0x1.c71c71c71c71cp-7, -0x1.c0e070381c0e0p-7, 0x1.bacf914c1bad0p-7, -0x1.b4e81b4e81b4fp-7, 0x1.af286bca1af28p-7, -0x1.a98ef606a63bep-7, 0x1.a41a41a41a41ap-7, -0x1.9ec8e951033d9p-7, 0x1.999999999999ap-7, -0x1.948b0fcd6e9e0p-7, 0x1.8f9c18f9c18fap-7, -0x1.8acb90f6bf3aap-7, 0x1.8618618618618p-7, -0x1.8181818181818p-7, 0x1.7d05f417d05f4p-7, -0x1.78a4c8178a4c8p-7, 0x1.745d1745d1746p-7, -0x1.702e05c0b8170p-7, 0x1.6c16c16c16c17p-7, -0x1.6816816816817p-7}
inline constexpr double lgam_c[91] = LAPLACE_GD_LGAM_C;
#if defined(__CUDACC__)
static __constant__ double d_lgam_c[91] = LAPLACE_GD_LGAM_C;
#endif
}  // namespace tab

LAPLACE_HD bool upper_small_x(double a, double x, int k, bool direct, side_result& s) {
  const double lx = math::log(x);
  const double lg1p = Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
  // Taylor coefficients (in h, at a) of L(a + h) and of lambda(a + h)
  double Lc[7], lam[7];
  Lc[0] = a * lx - lg1p;
  if (!(Kokkos::fabs(Lc[0]) <= 2.0)) return false;
  Lc[1] = lx - math::digamma(a + 1.0);
  double fact = 1.0;
  for (int j = 2; j <= k; ++j) {
    fact *= j;
    Lc[j] = -math::polygamma(j - 1, a + 1.0) / fact;
  }
  if (a < 0.5) {
    const double* cz = nullptr;
    KOKKOS_IF_ON_HOST((cz = tab::lgam_c;))
    KOKKOS_IF_ON_DEVICE((cz = tab::d_lgam_c;))
    for (int j = 0; j <= k; ++j) lam[j] = 0.0;
    // lambda_j = sum_{m>=j} c_(m+1) C(m, j) a^(m-j), summed from the small end;
    // terms to m = M with a^M M^5 / 5! below 2^-60 (order 6: four more terms,
    // for the factor C(M, 6) / a^6 of the last term of lambda_6)
    int M = a < 1e-3 ? 12 + (k > 5 ? 4 : 0) : static_cast<int>(-53.4 / math::log(a)) + (k > 5 ? 12 : 8);
    if (M > 90) M = 90;
    for (int m = M; m >= 0; --m) {
      // Horner in a for each j: lam[j] = lam[j] * a + c_(m+1) C(m, j)
      double bin = 1.0;  // C(m, j)
      for (int j = 0; j <= k && j <= m; ++j) {
        lam[j] = lam[j] * a + cz[m] * bin;
        bin = bin * (m - j) / (j + 1);
      }
    }
  } else {
    // lambda (a + h) (a + h) = log Gamma(1 + a + h): lambda_j = (G_j - lambda_(j-1)) / a,
    // G_0 = log Gamma(1 + a), G_j = psi^(j-1)(a + 1)/j!
    const double ia = 1.0 / a;
    lam[0] = lg1p * ia;
    double f = 1.0;
    for (int j = 1; j <= k; ++j) {
      f *= j;
      const double G = (j == 1 ? math::digamma(a + 1.0) : math::polygamma(j - 1, a + 1.0)) / f;
      lam[j] = (G - lam[j - 1]) * ia;
    }
  }
  // c(h) = -L(a + h)/(a + h) = lambda(a + h) - log x
  double cc[7];
  cc[0] = lam[0] - lx;
  for (int j = 1; j <= k; ++j) cc[j] = lam[j];
  // sinhc(z + dz(h)), z = L/2, dz = sum_{j>=1} L_j/2 h^j: sum_m sinhc^(m)(z)/m! dz^m
  const double z = 0.5 * Lc[0];
  // sinhc^(m)(z)/m! = sum_n C(2n, m) z^(2n-m)/(2n+1)!, |z| <= 1, n <= 12
  double sig[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    double zpow[25];
    zpow[0] = 1.0;
    for (int i = 1; i < 25; ++i) zpow[i] = zpow[i - 1] * z;
    double coef = 1.0;  // 1/(2n+1)!
    for (int n = 0; n <= 12; ++n) {
      if (n > 0) coef /= (2.0 * n) * (2.0 * n + 1.0);
      double bin = coef;  // C(2n, m)/(2n+1)!
      for (int m = 0; m <= k && m <= 2 * n; ++m) {
        sig[m] += bin * zpow[2 * n - m];
        bin = bin * (2 * n - m) / (m + 1);
      }
    }
  }
  double dz[7], pw[7], S[7];
  dz[0] = 0.0;
  for (int j = 1; j <= k; ++j) dz[j] = 0.5 * Lc[j];
  // S = sum_m sig[m] dz^m, the powers of dz as truncated series
  for (int j = 0; j <= k; ++j) S[j] = 0.0, pw[j] = 0.0;
  pw[0] = 1.0;
  S[0] = sig[0];
  for (int m = 1; m <= k; ++m) {
    double nw[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i <= k; ++i)
      for (int j = 1; i + j <= k; ++j) nw[i + j] += pw[i] * dz[j];
    for (int j = 0; j <= k; ++j) {
      pw[j] = nw[j];
      S[j] += sig[m] * pw[j];
    }
  }
  // e^(L(a + h)/2) = e^(L/2) exp(dz(h)): e_n = (1/n) sum_j j dz_j e_(n-j)
  double E[7];
  E[0] = 1.0;
  for (int n = 1; n <= k; ++n) {
    double acc = 0.0;
    for (int j = 1; j <= n; ++j) acc += j * dz[j] * E[n - j];
    E[n] = acc / n;
  }
  const double eh = math::exp(z);
  // K(a + h) = sum_{n>=1} (-x)^n/n! 1/(a + n + h): K_j = sum_n (-x)^n/n! (-1)^j/(a + n)^(j+1)
  double Kc[7] = {0, 0, 0, 0, 0, 0, 0};
  double term = 1.0;
  for (int n = 1; n < 60; ++n) {
    term *= -x / n;
    const double u = 1.0 / (a + n);
    double p = term * u;
    for (int j = 0; j <= k; ++j) {
      Kc[j] += p;
      p *= -u;
    }
    if (Kokkos::fabs(term) <= 1e-20 * Kokkos::fabs(Kc[0])) break;
  }
  // F = c S - eh E K
  double F[7];
  for (int j = 0; j <= k; ++j) {
    double acc = 0.0, ek = 0.0;
    for (int i = 0; i <= j; ++i) {
      acc += cc[i] * S[j - i];
      ek += E[i] * Kc[j - i];
    }
    F[j] = acc - eh * ek;
  }
  if (!(F[0] > 0.0)) return false;
  double m[7], kap[7];
  m[0] = 1.0;
  fact = 1.0;
  for (int j = 1; j <= k; ++j) {
    fact *= j;
    m[j] = fact * F[j] / F[0];
  }
  cumulants_from_moments(k, m, kap);
  // log R derivatives: j! L_j/2 + kappa_j(F); log Q adds the pole of log a
  s.lower = false;
  s.direct = direct;
  s.log_s = math::log(a) + z + math::log(F[0]);
  fact = 1.0;
  double ia_pow = 1.0, fact1 = 1.0;  // (j-1)! / a^j with its sign
  for (int j = 1; j <= k; ++j) {
    fact *= j;
    if (j > 1) fact1 *= (j - 1);
    ia_pow /= a;
    const double lr = fact * dz[j] + kap[j];
    s.l[j] = direct ? lr + ((j % 2 == 1) ? 1.0 : -1.0) * fact1 * ia_pow : lr;
  }
  return true;
}

// ---------------------------------------------------------------- Temme's expansion for large a

// The uniform asymptotic expansion of Temme (SIAM J. Math. Anal. 10, 1979,
// 757), as in DiDonato and Morris (ACM TOMS 12, 1986, 377):
//   Q = erfc(eta sqrt(a/2))/2 + R,  P = erfc(-eta sqrt(a/2))/2 - R,
//   R = exp(-a eta^2/2) / sqrt(2 pi a) sum_k C_k(eta) a^-k,
// eta^2/2 = lambda - 1 - log lambda, lambda = x/a, sign(eta) = sign(lambda - 1).
// The coefficients C_0..C_9, polynomials in eta, are those of Boost.Math's
// igamma_temme_large for 53 bits (detail/igamma_large.hpp, Copyright John
// Maddock 2006, Boost Software License 1.0), adapted.
// With erfc(z) = exp(-z^2) erfcx(z), the smaller side S (Q for x >= a, P below)
// is S = exp(-a eta^2/2) / sqrt(2 pi a) G,
//   G = sqrt(pi a / 2) erfcx(zeta) + s sum_k C_k(eta) a^-k,
// zeta = |eta| sqrt(a/2), s = +1 for Q and -1 for P. Then
//   log S = -a phi - log(2 pi a)/2 + log G,  a phi = a (lambda - 1 - log lambda),
// whose first two terms have exact derivatives in a (log lambda, then
// -(-1)^k (k-2)!/a^(k-1); and -(-1)^(k-1) (k-1)!/(2 a^k)), and G is carried as
// a Taylor polynomial of degree k <= 6 in the increment h of a: lambda(a + h)
// = x/(a + h) gives mu = lambda - 1 exactly, eta(mu) comes from its power
// series (eta = mu - mu^2/3 + 7 mu^3/36 - ..., 91 terms written by mpmath),
// the C_k by their derivatives in eta composed with the series of eta, and
// erfcx(zeta(a + h)) = erfcx(zeta) exp(sum_m L_m dzeta^m) with L_m the
// derivatives of log erfcx divided by m!: for zeta <= 1/sqrt(2) from the
// derivatives of log Phi at t = -sqrt(2) zeta (include/laplace/math/normal.hpp,
// its piecewise Chebyshev expansions; until 2026-11-12, night, the recursion
// in R = phi(t)/Phi(t) up to zeta = 1, which lost 5e4 ulp at order 6); above
// by its downward recursion for the cumulants of exp(-x s - s^2/2) on s > 0
// (x = -t), which gives L_m m! = (-sqrt 2)^m kappa_m without the cancellation of
// 2 zeta against R.
namespace tab {
#define LAPLACE_GD_ETA_SERIES {0x0.0p+0, 0x1.0000000000000p+0, -0x1.5555555555555p-2, 0x1.8e38e38e38e39p-3, -0x1.14dbf86a314dcp-3, 0x1.a4a96708379ffp-4, -0x1.51413a00a985ep-4, 0x1.185beff04cc45p-4, -0x1.de6d51a8397cap-5, 0x1.a04f527141348p-5, -0x1.6fdb43150e5efp-5, 0x1.49114db1b9168p-5, -0x1.295ae9ba40582p-5, 0x1.0ef9098745d5ap-5, -0x1.f166e7fedcf88p-6, 0x1.cb53717f15ef8p-6, -0x1.aa6d0051e828fp-6, 0x1.8dbac4b7495d4p-6, -0x1.747e105264989p-6, 0x1.5e2260c1c9a25p-6, -0x1.4a3261d6acb2bp-6, 0x1.3850316dba606p-6, -0x1.282fd2f8f9137p-6, 0x1.199322e36d8dep-6, -0x1.0c46d6491aa8fp-6, 0x1.0020398519c94p-6, -0x1.e9f6f1391abd1p-7, 0x1.d574995f877b2p-7, -0x1.c285e8d7bce90p-7, 0x1.b0fec9ae909aep-7, -0x1.a0b949dd67f93p-7, 0x1.9194943db2fe4p-7, -0x1.83741c6a01488p-7, 0x1.763ef26f383cfp-7, -0x1.69df35df686a9p-7, 0x1.5e41a1d1de9a1p-7, -0x1.53552cd4a8037p-7, 0x1.490ab8ee9c880p-7, -0x1.3f54d0a7514dcp-7, 0x1.36276eae0d519p-7, -0x1.2d77ce36e33aep-7, 0x1.253c4286d2020p-7, -0x1.1d6c14742188ep-7, 0x1.15ff64dc2e947p-7, -0x1.0eef133f4c973p-7, 0x1.0834a7d923a01p-7, -0x1.01ca40aa245d6p-7, 0x1.f75501fdedcbap-8, -0x1.eba1062cd8cb7p-8, 0x1.e06f972cdfcb4p-8, -0x1.d5b87d1aef142p-8, 0x1.cb742b9148248p-8, -0x1.c19bb09f0b973p-8, 0x1.b828a5bb8323ep-8, -0x1.af152271aed71p-8, 0x1.a65bb08aa3053p-8, -0x1.9df74185a92c7p-8, 0x1.95e32534205b5p-8, -0x1.8e1b015506a01p-8, 0x1.869aca1115ce4p-8, -0x1.7f5ebb3c9aa13p-8, 0x1.78635247c6b03p-8, -0x1.71a548c94e9dfp-8, 0x1.6b218f91c47c9p-8, -0x1.64d54a385cb84p-8, 0x1.5ebdcb13b91b3p-8, -0x1.58d88f92fe1cfp-8, 0x1.53233cece5279p-8, -0x1.4d9b9d1bbb79cp-8, 0x1.483f9c1e50239p-8, -0x1.430d4576c267cp-8, 0x1.3e02c1e0f1bb0p-8, -0x1.391e553b060bbp-8, 0x1.345e5c9b259cep-8, -0x1.2fc14c8df8b63p-8, 0x1.2b45af7a15639p-8, -0x1.26ea2424d8bacp-8, 0x1.22ad5c5591980p-8, -0x1.1e8e1b94341bfp-8, 0x1.1a8b360116e5ap-8, -0x1.16a38f437c3d3p-8, 0x1.12d6198ce33ebp-8, -0x1.0f21d4af5073fp-8, 0x1.0b85cd44eb012p-8, -0x1.08011be7743e0p-8, 0x1.0492e47643e6fp-8, -0x1.013a55699386cp-8, 0x1.fbed4e64017cfp-9, -0x1.f58e374690275p-9, 0x1.ef55fad2a283fp-9, -0x1.e9433f0d75fc1p-9}
#define LAPLACE_GD_TEMME_C {-0.33333333333333333, 0.083333333333333333, -0.014814814814814815, 0.0011574074074074074, 0.0003527336860670194, -0.00017875514403292181, 0.39192631785224378e-4, -0.21854485106799922e-5, -0.185406221071516e-5, 0.8296711340953086e-6, -0.17665952736826079e-6, 0.67078535434014986e-8, 0.10261809784240308e-7, -0.43820360184533532e-8, 0.91476995822367902e-9, -0.0018518518518518519, -0.0034722222222222222, 0.0026455026455026455, -0.00099022633744855967, 0.00020576131687242798, -0.40187757201646091e-6, -0.18098550334489978e-4, 0.76491609160811101e-5, -0.16120900894563446e-5, 0.46471278028074343e-8, 0.1378633446915721e-6, -0.5752545603517705e-7, 0.11951628599778147e-7, 0.0041335978835978836, -0.0026813271604938272, 0.00077160493827160494, 0.20093878600823045e-5, -0.00010736653226365161, 0.52923448829120125e-4, -0.12760635188618728e-4, 0.34235787340961381e-7, 0.13721957309062933e-5, -0.6298992138380055e-6, 0.14280614206064242e-6, 0.00064943415637860082, 0.00022947209362139918, -0.00046918949439525571, 0.00026772063206283885, -0.75618016718839764e-4, -0.23965051138672967e-6, 0.11082654115347302e-4, -0.56749528269915966e-5, 0.14230900732435884e-5, -0.0008618882909167117, 0.00078403922172006663, -0.00029907248030319018, -0.14638452578843418e-5, 0.66414982154651222e-4, -0.39683650471794347e-4, 0.11375726970678419e-4, -0.00033679855336635815, -0.69728137583658578e-4, 0.00027727532449593921, -0.00019932570516188848, 0.67977804779372078e-4, 0.1419062920643967e-6, -0.13594048189768693e-4, 0.80184702563342015e-5, -0.22914811765080952e-5, 0.00053130793646399222, -0.00059216643735369388, 0.00027087820967180448, 0.79023532326603279e-6, -0.81539693675619688e-4, 0.56116827531062497e-4, -0.18329116582843376e-4, 0.00034436760689237767, 0.51717909082605922e-4, -0.00033493161081142236, 0.0002812695154763237, -0.00010976582244684731, -0.00065262391859530942, 0.00083949872067208728, -0.00043829709854172101, -0.00059676129019274625}
inline constexpr double eta_d[91] = LAPLACE_GD_ETA_SERIES;
inline constexpr double temme_c[80] = LAPLACE_GD_TEMME_C;
#if defined(__CUDACC__)
static __constant__ double d_eta_d[91] = LAPLACE_GD_ETA_SERIES;
static __constant__ double d_temme_c[80] = LAPLACE_GD_TEMME_C;
#endif
}  // namespace tab

// r = p q, truncated to degree k (r may not alias p or q)
LAPLACE_HD_FORCEINLINE void ser_mul(const double* p, const double* q, double* r, int k) {
  for (int j = 0; j <= k; ++j) {
    double acc = 0.0;
    for (int i = 0; i <= j; ++i) acc += p[i] * q[j - i];
    r[j] = acc;
  }
}

// Taylor coefficients q_m = p^(m)(x0)/m!, m = 0..k, of p(x) = sum_{n<N} c_n x^n,
// by repeated synthetic division (no divisions; N <= 91)
LAPLACE_HD_FORCEINLINE void taylor_shift(const double* c, int N, double x0, int k, double* q) {
  double b[91];
  for (int n = 0; n < N; ++n) b[n] = c[n];
  for (int m = 0; m <= k; ++m) {
    for (int n = N - 2; n >= m; --n) b[n] += x0 * b[n + 1];
    q[m] = m < N ? b[m] : 0.0;
  }
}

// derivatives of log erfcx at z >= 0, divided by m!: Lm[m], m = 1..k
LAPLACE_HD void log_erfcx_taylor(double z, int k, double* Lm) {
  constexpr double sqrt2 = 1.41421356237309504880;
  double L[7] = {0, 0, 0, 0, 0, 0, 0};
  const double t = -sqrt2 * z;
  // Up to z = 1/sqrt(2) (t >= -1) from the derivatives g_m of log Phi at t,
  // L_m = (-sqrt 2)^m g_m plus the terms of z^2, with g_m from normal.hpp's
  // piecewise Chebyshev expansions; above, the downward recursion below.
  // Until 2026-11-12 (night) the g_m came here from the recursion in
  // R = phi(t)/Phi(t) up to z = 1, whose polynomials in t and R cancel: the
  // Taylor coefficient L_6 lost up to 5.3e4 ulp of its value for z in
  // [0.3, 1] (2.2e3 for L_5), against 80 for the downward recursion at
  // z = 1.05, and that loss went whole into the pure derivatives of order 5
  // and 6 in x of Temme's expansion at large a (docs/numerics.md).
  if (z <= 0.70710678118654752440) {
    L[1] = 2.0 * z - sqrt2 * math::log_Phi_d1(t);
    if (k >= 2) L[2] = 2.0 + 2.0 * math::log_Phi_d2(t);
    if (k >= 3) L[3] = -2.0 * sqrt2 * math::log_Phi_d3(t);
    if (k >= 4) L[4] = 4.0 * math::log_Phi_d4(t);
    if (k >= 5) L[5] = -4.0 * sqrt2 * math::log_Phi_d5(t);
    if (k >= 6) L[6] = 8.0 * math::log_Phi_d6(t);
  } else {
    // as in normal.hpp's log_Phi_dk: the scaled ratios rho_j = x r_j, downward
    const double x = -t;
    const double ix = 1.0 / x, w = ix * ix;
    const int K = k + 20 + static_cast<int>(Kokkos::ceil(300.0 * w));
    const double K1 = K + 1.0;
    double rho = 2.0 * K1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * K1 * w));
    double r[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int j = K; j >= 1; --j) {
      rho = j / (1.0 + w * rho);
      if (j <= k) r[j] = rho;
    }
    double mu[7], kap[7];
    mu[0] = 1.0;
    for (int j = 1; j <= k; ++j) mu[j] = mu[j - 1] * r[j];
    cumulants_from_moments(k, mu, kap);
    // kappa_j (unscaled) = kap_j / x^j; L_j = (-sqrt 2)^j kappa_j
    double f = 1.0;
    for (int j = 1; j <= k; ++j) {
      f *= -sqrt2 * ix;
      L[j] = f * kap[j];
    }
  }
  double fact = 1.0;
  for (int m = 1; m <= k; ++m) {
    fact *= m;
    Lm[m] = L[m] / fact;
  }
}

// Temme's route for the side S (P for x < a, Q for x >= a); s.lower tells which.
LAPLACE_HD bool temme_side(double a, double x, int k, bool want_p, side_result& s) {
  constexpr int off[11] = {0, 15, 28, 39, 48, 55, 64, 71, 76, 79, 80};
  const double sigma = (x - a) / a;
  const bool lower = x < a;
  const bool direct = lower == want_p;
  const double lam = x / a, ia = 1.0 / a;
  const double* dd = nullptr;
  const double* cc = nullptr;
  KOKKOS_IF_ON_HOST((dd = tab::eta_d; cc = tab::temme_c;))
  KOKKOS_IF_ON_DEVICE((dd = tab::d_eta_d; cc = tab::d_temme_c;))
  // eta^(m)(sigma)/m!, m = 0..k, from eta = sum_n d_n mu^n
  double em[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    const double as = Kokkos::fabs(sigma);
    int M = as < 1e-3 ? 12 : static_cast<int>(-53.4 / math::log(as)) + 8;
    if (M > 90) M = 90;
    taylor_shift(dd, M + 1, sigma, k, em);
  }
  const double eta = em[0];
  // dmu^m has the coefficient lam^m C(j-1, m-1) (-1/a)^j at h^j; de = eta(h) - eta
  double de[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    double ph = 1.0;  // (-1/a)^j
    for (int j = 1; j <= k; ++j) {
      ph *= -ia;
      double acc = 0.0, lm = 1.0, bin = 1.0;  // lam^m, C(j-1, m-1)
      for (int m = 1; m <= j; ++m) {
        lm *= lam;
        acc += em[m] * lm * bin;
        bin = bin * (j - m) / m;
      }
      de[j] = acc * ph;
    }
  }
  // powers of de (no constant term): P[m] = de^m
  double P[7][7];
  for (int j = 0; j <= 6; ++j) P[0][j] = 0.0, P[1][j] = de[j];
  P[0][0] = 1.0;
  for (int m = 2; m <= k; ++m) ser_mul(P[m - 1], de, P[m], k);
  // S(h) = sum_i C_i(eta(h)) (a + h)^-i by Horner in w = 1/(a + h)
  double w[7], S[7] = {0, 0, 0, 0, 0, 0, 0};
  {
    double f = ia;
    for (int j = 0; j <= k; ++j) {
      w[j] = f;
      f *= -ia;
    }
  }
  for (int i = 9; i >= 0; --i) {
    // C_i^(m)(eta)/m!
    const int b = off[i], e = off[i + 1];
    double cm[7] = {0, 0, 0, 0, 0, 0, 0};
    taylor_shift(cc + b, e - b, eta, k, cm);
    double T[7];
    ser_mul(S, w, T, k);
    for (int j = 0; j <= k; ++j) {
      double acc = 0.0;
      for (int m = 0; m <= j; ++m) acc += cm[m] * P[m][j];
      S[j] = T[j] + acc;
    }
  }
  // sqrt((a + h)/a) = (1 + h/a)^(1/2)
  double sq[7];
  {
    double c = 1.0, f = 1.0;
    for (int j = 0; j <= k; ++j) {
      sq[j] = c * f;
      c = c * (0.5 - j) / (j + 1);
      f *= ia;
    }
  }
  const double sgn = lower ? -1.0 : 1.0;
  const double rt = Kokkos::sqrt(0.5 * a);
  // zeta(h) = sgn eta(h) sqrt((a + h)/2)
  double ez[7], zh[7];
  ez[0] = sgn * eta;
  for (int j = 1; j <= k; ++j) ez[j] = sgn * de[j];
  ser_mul(ez, sq, zh, k);
  const double zeta = rt * zh[0];
  if (!(zeta >= 0.0)) return false;
  double dz[7];
  dz[0] = 0.0;
  for (int j = 1; j <= k; ++j) dz[j] = rt * zh[j];
  double Lm[7];
  log_erfcx_taylor(zeta, k, Lm);
  // Lz = sum_m Lm dz^m, then E = exp(Lz)
  double Lz[7] = {0, 0, 0, 0, 0, 0, 0}, pw[7], tmp[7];
  for (int j = 0; j <= k; ++j) pw[j] = dz[j];
  for (int m = 1; m <= k; ++m) {
    for (int j = 0; j <= k; ++j) Lz[j] += Lm[m] * pw[j];
    if (m < k) {
      ser_mul(pw, dz, tmp, k);
      for (int j = 0; j <= k; ++j) pw[j] = tmp[j];
    }
  }
  double E[7];
  E[0] = 1.0;
  for (int n = 1; n <= k; ++n) {
    double acc = 0.0;
    for (int j = 1; j <= n; ++j) acc += j * Lz[j] * E[n - j];
    E[n] = acc / n;
  }
  // G = sqrt(pi a/2) erfcx(zeta) sq E + sgn S
  constexpr double sqrt_pi_half = 1.25331413731550025121;
  const double A0 = sqrt_pi_half * Kokkos::sqrt(a) * math::erfcx(zeta);
  double AE[7], G[7];
  ser_mul(sq, E, AE, k);
  for (int j = 0; j <= k; ++j) G[j] = A0 * AE[j] + sgn * S[j];
  if (!(G[0] > 0.0)) return false;
  double mom[7], kap[7];
  mom[0] = 1.0;
  double fact = 1.0;
  for (int j = 1; j <= k; ++j) {
    fact *= j;
    mom[j] = fact * G[j] / G[0];
  }
  cumulants_from_moments(k, mom, kap);
  s.lower = lower;
  s.direct = direct;
  const double aphi = -a * log1pmx(sigma);
  constexpr double half_log_2pi = 0.91893853320467274178;
  s.log_s = -aphi - 0.5 * math::log(a) - half_log_2pi + math::log(G[0]);
  double f1 = 1.0, f2 = 1.0, ap = 1.0;  // (j-1)!, (j-2)!, a^(j-1)
  for (int j = 1; j <= k; ++j) {
    if (j >= 2) f1 *= (j - 1);
    if (j >= 3) f2 *= (j - 2);
    const double aj = ap * a;  // a^j
    double v = kap[j];
    if (j == 1) {
      v += Kokkos::log1p(sigma);
    } else {
      v -= ((j % 2 == 0) ? 1.0 : -1.0) * f2 / ap;
    }
    v -= 0.5 * ((j % 2 == 1) ? 1.0 : -1.0) * f1 / aj;
    // the complement of Q works with log(Q/a)
    if (!direct && !lower) v -= ((j % 2 == 1) ? 1.0 : -1.0) * f1 / aj;
    s.l[j] = v;
    ap = aj;
  }
  return true;
}

// ---------------------------------------------------------------- the upper side by the fraction, in Taylor form

// Legendre's continued fraction K = b_0 + a_1/(b_1 + a_2/(b_2 + ...)),
// I_U = 1/K, b_n = x + 2n + 1 - a, a_n = n (a - n), evaluated by the modified
// Lentz scheme with every quantity a Taylor polynomial of degree k in the
// increment h of a (b_n - h, a_n + n h): C_n = b_n + a_n / C_(n-1),
// D_n = 1 / (b_n + a_n D_(n-1)), Delta_n = C_n D_n, and
// log K = log b_0 + sum_n log Delta_n. Unlike the forward recurrences of the
// numerators and denominators (rejected in the first round: their cumulants
// cancel), the terms log Delta_n tend to 0 and their sum does not cancel.
// Returns false if a C_n or D_n comes near zero or the fraction has not
// converged within nmax steps.
LAPLACE_HD_FORCEINLINE void ser_recip(const double* p, double* r, int k) {
  const double i0 = 1.0 / p[0];
  r[0] = i0;
  for (int j = 1; j <= k; ++j) {
    double acc = 0.0;
    for (int i = 1; i <= j; ++i) acc += p[i] * r[j - i];
    r[j] = -acc * i0;
  }
}

LAPLACE_HD bool fraction_taylor(double a, double x, int k, bool direct, int nmax, side_result& s) {
  double C[7], D[7], L[7], Labs[7], t[7], u[7];
  for (int j = 0; j <= 6; ++j) C[j] = D[j] = L[j] = Labs[j] = 0.0;
  const double b0 = x + 1.0 - a;
  if (!(b0 > 0.0)) return false;
  C[0] = b0;
  C[1] = -1.0;
  // log b_0(h) = log b0 + log(1 - h/b0)
  double logK = math::log(b0);
  {
    const double ib = 1.0 / b0;
    double p = 1.0;
    for (int j = 1; j <= k; ++j) {
      p *= ib;
      L[j] = -p / j;
    }
  }
  double prod = 1.0;  // product of Delta_n(0), folded into logK now and then
  bool conv = false;
  for (int n = 1; n <= nmax; ++n) {
    const double bn = x + (2.0 * n + 1.0) - a;
    const double an0 = n * (a - n), an1 = n;
    // D_n = 1/(b_n + a_n D_(n-1))
    for (int j = 0; j <= k; ++j) t[j] = an0 * D[j] + (j > 0 ? an1 * D[j - 1] : 0.0);
    t[0] += bn;
    if (k >= 1) t[1] -= 1.0;
    if (!(Kokkos::fabs(t[0]) > 1e-200)) return false;
    ser_recip(t, D, k);
    // C_n = b_n + a_n / C_(n-1)
    if (!(Kokkos::fabs(C[0]) > 1e-200)) return false;
    ser_recip(C, u, k);
    for (int j = 0; j <= k; ++j) C[j] = an0 * u[j] + (j > 0 ? an1 * u[j - 1] : 0.0);
    C[0] += bn;
    if (k >= 1) C[1] -= 1.0;
    // Delta = C D and its log: l_0 = log Delta_0, l_j from l' = Delta'/Delta
    double Dl[7], l[7];
    ser_mul(C, D, Dl, k);
    const double i0 = 1.0 / Dl[0];
    for (int j = 1; j <= k; ++j) {
      double acc = j * Dl[j];
      for (int i = 1; i < j; ++i) acc -= i * l[i] * Dl[j - i];
      l[j] = acc * i0 / j;
    }
    prod *= Dl[0];
    if (n % 16 == 0) {
      logK += math::log(prod);
      prod = 1.0;
    }
    bool small = Kokkos::fabs(Dl[0] - 1.0) <= 4.0 * eps;
    for (int j = 1; j <= k; ++j) {
      L[j] += l[j];
      Labs[j] += Kokkos::fabs(l[j]);
      if (!(Kokkos::fabs(l[j]) <= 2.0 * eps * Labs[j])) small = false;
    }
    if (small && n > 2) {
      conv = true;
      break;
    }
  }
  if (!conv) return false;
  logK += math::log(prod);
  // log I_U = -log K: cumulants kappa_j = -j! L_j
  double kap[7];
  double fact = 1.0;
  for (int j = 1; j <= k; ++j) {
    fact *= j;
    kap[j] = -fact * L[j];
  }
  s = from_cumulants(a, x, k, false, direct, -logK, kap);
  return true;
}

// ---------------------------------------------------------------- dispatch

// The chosen route: the closed form for the upper side at small x (when log Q
// itself is asked for), a series where it converges within nmax terms (the
// lower series on the lower side, the asymptotic series on the upper side for
// x >= 36), the quadrature elsewhere. The limits on the number of terms are
// where the series stops being cheaper than the quadrature, measured per
// backend: on one CPU thread a term costs 5-6 ns on the lower side (a division
// per pass) against about 25-50 ns per node of the quadrature (about 65
// nodes). On the GPU long series are slow (the series of a = 50-1e4 near
// x = a took 500-950 ns) and a warp that mixes series and quadrature runs
// both: with no series at all the route took 112-117 ns (P) and 126-161 ns
// (Q) on the mixed timing set, against 122-136 and 137-182 ns with series of
// up to 60 terms on both sides; but without the lower series log Q at
// a = 0.47, x = 0.097 (from the complement of the lower side) reached
// 9.5 ulp against the bound of 7.5. The device keeps the lower series up to
// 60 terms and drops the upper one.
constexpr int nmax_lower_host = 300;
constexpr int nmax_upper_host = 500;
constexpr int nmax_lower_device = 60;
constexpr int nmax_upper_device = 0;
// The fraction in Taylor form: on one CPU thread 1.0-2.6 us against 3.7-4.0 us
// for the quadrature (x from 3/2 to 36); on the GPU its long, uneven loops on
// arrays of Taylor coefficients took 420-1060 ns where the quadrature takes
// 54-110 ns, so the device does not use it.
constexpr int nmax_fraction_host = 400;
constexpr int nmax_fraction_device = 0;

template <int E, bool SMALLX = true>
LAPLACE_HD side_result chosen_side(double a, double x, int k, bool lower, bool direct, int nmax_lower,
                                   int nmax_upper, int nmax_fraction = 0) {
  double kap[7], log_i;
  if (SMALLX && direct && !lower && x < 1.5 && a < 2.0) {
    side_result s;
    if (upper_small_x(a, x, k, direct, s)) return s;
  }
  if ((lower || x >= 36.0) && series_moments(a, x, lower, k, lower ? nmax_lower : nmax_upper, kap, log_i))
    return from_cumulants(a, x, k, lower, direct, log_i, kap);
  // the fraction in Taylor form: for log Q from x = 3/2 (at most 0.25 of the
  // test bounds), for the complement (log P) from x = 3 (0.11; between 3/2 and 3
  // it reached 0.91 of the bound of order 1). Not within 0.75 sqrt(a) above
  // x = a (measured on 2026-11-12 against the quadrature, log Q, a = 1.6..30):
  // for z = (x - a)/sqrt(a) in [-0.25, 0) the fraction differed by 2-7 ulp1
  // at order 2, 560-1500 at order 5 and 5e3-1.4e4 at order 6; for z >= 0.75
  // by at most 1.9 ulp1 at order 5 and 4 at order 6.
  if (!lower && x >= (direct ? 1.5 : 3.0) && x >= a + 0.75 * Kokkos::sqrt(a) && nmax_fraction > 0) {
    side_result s;
    if (fraction_taylor(a, x, k, direct, nmax_fraction, s)) return s;
  }
  quad_moments<8, E>(a, x, lower, kap, log_i, k);
  return from_cumulants(a, x, k, lower, direct, log_i, kap);
}

template <int E>
LAPLACE_HD side_result chosen_side_backend(double a, double x, int k, bool lower, bool direct) {
  int nl = nmax_lower_host, nu = nmax_upper_host, nf = nmax_fraction_host;
  // on the device the lower series only for a < 2 (where the bound of log Q
  // needs it); elsewhere one warp running series and quadrature side by side
  // was slower than the quadrature alone (far lower tail: 70-160 ns against 57-75)
  KOKKOS_IF_ON_DEVICE((nl = a < 2.0 ? nmax_lower_device : 0; nu = nmax_upper_device; nf = nmax_fraction_device;))
  return chosen_side<E>(a, x, k, lower, direct, nl, nu, nf);
}

// Candidates for the measurement: 0 the chosen route; 1 series and fraction
// (the quadrature for the upper side with x < 1); 2, 3, 5 the quadrature with
// h = 1/6, 1/12, 1/8; 4 the series on every lower side, the quadrature (1/8)
// on the upper side (the route chosen in the first round); 6 the chosen route
// with own::exp in the quadrature; 7 the quadrature (1/8) with own::exp;
// 10..13: the chosen route with (lower, upper) limits on the terms (60, 60),
// (150, 300), (300, 500), (1000, 1000) on every backend; 20..27: coarser steps
// of the quadrature's panels (rejected); 30: without the small-x closed form.
LAPLACE_HD side_result side_any(int method, double a, double x, int k, bool lower, bool direct) {
  switch (method) {
    case 1: return series_side(a, x, k, lower, direct);
    case 2: return quad_side<6, 0>(a, x, k, lower, direct);
    case 3: return quad_side<12, 0>(a, x, k, lower, direct);
    case 4:
      if (lower && (x < 800.0 || x < 0.5 * a)) return series_lower(a, x, k, direct);
      return quad_side<8, 0>(a, x, k, lower, direct);
    case 5: return quad_side<8, 0>(a, x, k, lower, direct);
    case 6: return chosen_side_backend<1>(a, x, k, lower, direct);
    case 7: return quad_side<8, 1>(a, x, k, lower, direct);
    case 10: return chosen_side<0>(a, x, k, lower, direct, 60, 60);
    case 11: return chosen_side<0>(a, x, k, lower, direct, 150, 300);
    case 12: return chosen_side<0>(a, x, k, lower, direct, 300, 500);
    case 13: return chosen_side<0>(a, x, k, lower, direct, 1000, 1000);
    case 20: return quad_side_s<2, 1, 1, 1>(a, x, k, lower, direct);
    case 21: return quad_side_s<1, 2, 1, 1>(a, x, k, lower, direct);
    case 22: return quad_side_s<1, 1, 2, 1>(a, x, k, lower, direct);
    case 23: return quad_side_s<1, 1, 1, 2>(a, x, k, lower, direct);
    case 24: return quad_side_s<1, 1, 4, 1>(a, x, k, lower, direct);
    case 25: return quad_side_s<1, 1, 2, 1, 6>(a, x, k, lower, direct);
    case 26: return quad_side_s<1, 1, 2, 1, 9>(a, x, k, lower, direct);
    case 27: return quad_side_s<1, 1, 1, 1, 6>(a, x, k, lower, direct);
    case 30: return chosen_side<0, false>(a, x, k, lower, direct, 1000, 1000);
    case 31: return chosen_side_backend<0>(a, x, k, lower, direct);  // without Temme
    case 32: {  // the fraction in Taylor form on the upper side, else the chosen route
      side_result t;
      if (!lower && fraction_taylor(a, x, k, direct, 2000, t)) return t;
      return chosen_side_backend<0>(a, x, k, lower, direct);
    }
    default: return chosen_side_backend<0>(a, x, k, lower, direct);
  }
}

LAPLACE_HD bool side_lower(double a, double x, bool want_p) {
  // For a < 1 and x < 1/4 the lower side gives log P without cancellation
  // even where P is near 1 (every term of d log P/da is negative for x < a),
  // while the complement of the upper side cancels there: 2.5e6 ulpL against
  // 57 on order 5 for x < a. The threshold 1/4 was the best of a, 2a, 4a,
  // 1/4, 0.35, 1/2, 1 (order 5: 2.8e3, 1.6e3, 1e3, 115, 286, 286, 344 ulp).
  // Extended on 2026-11-09 (evening) to a < 2, x < 3/2, where the lower series
  // is short and the complement of the upper side needed the quadrature:
  // max errors for P orders 1-5 became 4.0, 4.4, 8.8, 23.5, 197 (min of ulpL
  // and ulp1; before 3.9, 13.8, 36.5, 51.7, 159).
  return (want_p && a < 2.0 && x < 1.5) || choose_lower(a, x);
}

// Fixed true: the chosen route (what the public functions call); false: the
// candidate `method` of side_any (for the bench).
template <bool Fixed>
LAPLACE_GD_NOINLINE LAPLACE_HD double log_gamma_da_impl(int k, double a, double x, bool want_p, int method) {
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(x >= 0.0) || a == inf) return nan;
  // limits: P -> 0 at x = 0 (d log P/da -> -inf, higher orders -> -psi^(k-1)(a + 1));
  // Q -> 0 as x -> inf (d log Q/da -> +inf, higher orders -> -psi^(k-1)(a));
  // the other side is 1 and its derivatives 0
  if (x == 0.0) return want_p ? (k == 1 ? -inf : -math::polygamma(k - 1, a + 1.0)) : 0.0;
  if (x == inf) return want_p ? 0.0 : (k == 1 ? inf : -math::polygamma(k - 1, a));
  // Temme's expansion where it keeps the relative accuracy of the other routes
  // (measured against the references: within 0.13 of the test bounds already
  // for a >= 20, |x/a - 1| < 0.4, but there the relative error of orders 4-5
  // reached 1.7e4 ulpL at a = 30.8, x = 42.6 and 466 at a = 122, x = 170; in
  // the region below at most 288 ulpL, against 130-245 for the quadrature)
  if (Fixed || method == 0 || method == 6) {
    const double sg = Kokkos::fabs(x - a) / a;
    if ((a >= 20.0 && sg < 0.3) || (a >= 300.0 && sg < 0.4) || (a >= 1000.0 && sg < 0.5)) {
      side_result t;
      if (temme_side(a, x, k, want_p, t)) return t.direct ? t.l[k] : complement_dk(k, t, a);
    }
  }
  const bool lower = side_lower(a, x, want_p);
  const bool direct = lower == want_p;
  side_result s;
  if constexpr (Fixed) {
    s = chosen_side_backend<0>(a, x, k, lower, direct);
  } else {
    s = side_any(method, a, x, k, lower, direct);
  }
  if (direct) return s.l[k];
  return complement_dk(k, s, a);
}

LAPLACE_HD double log_gamma_da(int k, double a, double x, bool want_p, int method) {
  return log_gamma_da_impl<false>(k, a, x, want_p, method);
}

}  // namespace detail_gamma_deriv

// d^k log P(a, x) / da^k and d^k log Q(a, x) / da^k, k = 1..6
/**
 * \ingroup incomplete_gamma
 * \brief First derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the first shape derivative:
 * \f[
 * \frac{\partial}{\partial a} \log P(a, x) = \log(x/a) - (\psi(a + 1) - \log a) + \kappa_1
 * \f]
 * where \f$\kappa_1\f$ is the first cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial}{\partial a}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\infty\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 4.0 ulp (gcc) / 4.3 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 6.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da2(), log_gamma_q_da1()
 */
LAPLACE_HD double log_gamma_p_da1(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(1, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Second derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the second shape derivative:
 * \f[
 * \frac{\partial^2}{\partial a^2} \log P(a, x) = \kappa_2 - \psi'(a + 1)
 * \f]
 * where \f$\psi'(z) = \frac{\mathrm{d}^2}{\mathrm{d}z^2}\log \Gamma(z)\f$ is the trigamma function, and \f$\kappa_2\f$ is the
 * second cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^2}{\partial a^2}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\psi'(a + 1)\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 4.4 ulp (gcc) / 4.4 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 7.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da1(), log_gamma_p_da3(), log_gamma_q_da2()
 */
LAPLACE_HD double log_gamma_p_da2(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(2, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Third derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the third shape derivative:
 * \f[
 * \frac{\partial^3}{\partial a^3} \log P(a, x) = \kappa_3 - \psi''(a + 1)
 * \f]
 * where \f$\psi''(z) = \frac{\mathrm{d}^3}{\mathrm{d}z^3}\log \Gamma(z)\f$ is the polygamma function of order 2, and \f$\kappa_3\f$ is the
 * third cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^3}{\partial a^3}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\psi''(a + 1)\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 8.8 ulp (gcc) / 10.8 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 16.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da2(), log_gamma_p_da4(), log_gamma_q_da3()
 */
LAPLACE_HD double log_gamma_p_da3(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(3, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Fourth derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the fourth shape derivative:
 * \f[
 * \frac{\partial^4}{\partial a^4} \log P(a, x) = \kappa_4 - \psi'''(a + 1)
 * \f]
 * where \f$\psi'''(z) = \frac{\mathrm{d}^4}{\mathrm{d}z^4}\log \Gamma(z)\f$ is the polygamma function of order 3, and \f$\kappa_4\f$ is the
 * fourth cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^4}{\partial a^4}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\psi'''(a + 1)\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 23.5 ulp (gcc) / 23.0 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 35.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da3(), log_gamma_p_da5(), log_gamma_q_da4()
 */
LAPLACE_HD double log_gamma_p_da4(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(4, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Fifth derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the fifth shape derivative:
 * \f[
 * \frac{\partial^5}{\partial a^5} \log P(a, x) = \kappa_5 - \psi^{(4)}(a + 1)
 * \f]
 * where \f$\psi^{(4)}(z) = \frac{\mathrm{d}^5}{\mathrm{d}z^5}\log \Gamma(z)\f$ is the polygamma function of order 4, and \f$\kappa_5\f$ is the
 * fifth cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^5}{\partial a^5}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\psi^{(4)}(a + 1)\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 197.0 ulp (gcc) / 171.0 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 245.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da4(), log_gamma_p_da6(), log_gamma_q_da5()
 */
LAPLACE_HD double log_gamma_p_da5(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(5, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Sixth derivative of \f$\log P(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the sixth shape derivative:
 * \f[
 * \frac{\partial^6}{\partial a^6} \log P(a, x) = \kappa_6 - \psi^{(5)}(a + 1)
 * \f]
 * where \f$\psi^{(5)}(z) = \frac{\mathrm{d}^6}{\mathrm{d}z^6}\log \Gamma(z)\f$ is the polygamma function of order 5, and \f$\kappa_6\f$ is the
 * sixth cumulant of \f$d = \log(t/x)\f$ under the normalized lower incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^6}{\partial a^6}\log P(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\psi^{(5)}(a + 1)\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 771.1 ulp (gcc) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 1160.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_p(), log_gamma_p_da5(), log_gamma_q_da6()
 */
LAPLACE_HD double log_gamma_p_da6(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(6, a, x, true, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief First derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the first shape derivative:
 * \f[
 * \frac{\partial}{\partial a} \log Q(a, x) = \log(x/a) + (\log a - \psi(a)) + \kappa_1
 * \f]
 * where \f$\kappa_1\f$ is the first cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial}{\partial a}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 4.8 ulp (gcc) / 4.8 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 7.5 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da2(), log_gamma_p_da1()
 */
LAPLACE_HD double log_gamma_q_da1(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(1, a, x, false, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Second derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the second shape derivative:
 * \f[
 * \frac{\partial^2}{\partial a^2} \log Q(a, x) = \kappa_2 - \psi'(a)
 * \f]
 * where \f$\psi'(z) = \frac{\mathrm{d}^2}{\mathrm{d}z^2}\log \Gamma(z)\f$ is the trigamma function, and \f$\kappa_2\f$ is the
 * second cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^2}{\partial a^2}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\psi'(a)\f$.
 *
 * \par Measured accuracy
 * Maximum error 5.5 ulp (gcc) / 7.3 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 11.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da1(), log_gamma_q_da3(), log_gamma_p_da2()
 */
LAPLACE_HD double log_gamma_q_da2(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(2, a, x, false, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Third derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the third shape derivative:
 * \f[
 * \frac{\partial^3}{\partial a^3} \log Q(a, x) = \kappa_3 - \psi''(a)
 * \f]
 * where \f$\psi''(z) = \frac{\mathrm{d}^3}{\mathrm{d}z^3}\log \Gamma(z)\f$ is the polygamma function of order 2, and \f$\kappa_3\f$ is the
 * third cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^3}{\partial a^3}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\psi''(a)\f$.
 *
 * \par Measured accuracy
 * Maximum error 13.9 ulp (gcc) / 13.9 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 21.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da2(), log_gamma_q_da4(), log_gamma_p_da3()
 */
LAPLACE_HD double log_gamma_q_da3(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(3, a, x, false, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Fourth derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the fourth shape derivative:
 * \f[
 * \frac{\partial^4}{\partial a^4} \log Q(a, x) = \kappa_4 - \psi'''(a)
 * \f]
 * where \f$\psi'''(z) = \frac{\mathrm{d}^4}{\mathrm{d}z^4}\log \Gamma(z)\f$ is the polygamma function of order 3, and \f$\kappa_4\f$ is the
 * fourth cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^4}{\partial a^4}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\psi'''(a)\f$.
 *
 * \par Measured accuracy
 * Maximum error 22.1 ulp (gcc) / 23.0 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 35.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da3(), log_gamma_q_da5(), log_gamma_p_da4()
 */
LAPLACE_HD double log_gamma_q_da4(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(4, a, x, false, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Fifth derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the fifth shape derivative:
 * \f[
 * \frac{\partial^5}{\partial a^5} \log Q(a, x) = \kappa_5 - \psi^{(4)}(a)
 * \f]
 * where \f$\psi^{(4)}(z) = \frac{\mathrm{d}^5}{\mathrm{d}z^5}\log \Gamma(z)\f$ is the polygamma function of order 4, and \f$\kappa_5\f$ is the
 * fifth cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^5}{\partial a^5}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\psi^{(4)}(a)\f$.
 *
 * \par Measured accuracy
 * Maximum error 62.9 ulp (gcc) / 67.9 ulp (CUDA) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 102.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da4(), log_gamma_q_da6(), log_gamma_p_da5()
 */
LAPLACE_HD double log_gamma_q_da5(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(5, a, x, false, 0); }

/**
 * \ingroup incomplete_gamma
 * \brief Sixth derivative of \f$\log Q(a, x)\f$ with respect to the shape \f$a\f$.
 *
 * Computes the sixth shape derivative:
 * \f[
 * \frac{\partial^6}{\partial a^6} \log Q(a, x) = \kappa_6 - \psi^{(5)}(a)
 * \f]
 * where \f$\psi^{(5)}(z) = \frac{\mathrm{d}^6}{\mathrm{d}z^6}\log \Gamma(z)\f$ is the polygamma function of order 5, and \f$\kappa_6\f$ is the
 * sixth cumulant of \f$d = \log(t/x)\f$ under the normalized upper incomplete gamma measure
 * (defined on \ref incomplete_gamma).
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\frac{\partial^6}{\partial a^6}\log Q(a, x)\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, \f$a = +\infty\f$, \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\psi^{(5)}(a)\f$.
 *
 * \par Measured accuracy
 * Maximum error 96.3 ulp (gcc) in \f$\min(\text{ulpL}, \text{ulp1})\f$; test bound 145.0 ulp.
 *
 * \see incomplete_gamma, log_gamma_q(), log_gamma_q_da5(), log_gamma_p_da6()
 */
LAPLACE_HD double log_gamma_q_da6(double a, double x) { return detail_gamma_deriv::log_gamma_da_impl<true>(6, a, x, false, 0); }

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls (docs/numerics.md, 2026-11-09). The routes
// differ by backend: the continued fraction and the upper asymptotic series run
// on the host only (on CUDA the quadrature is faster there), decided 2026-11-09.
LAPLACE_HD_FORCEINLINE double log_gamma_p_da1(double a, double x) { return own::log_gamma_p_da1(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_p_da2(double a, double x) { return own::log_gamma_p_da2(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_p_da3(double a, double x) { return own::log_gamma_p_da3(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_p_da4(double a, double x) { return own::log_gamma_p_da4(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_p_da5(double a, double x) { return own::log_gamma_p_da5(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_p_da6(double a, double x) { return own::log_gamma_p_da6(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da1(double a, double x) { return own::log_gamma_q_da1(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da2(double a, double x) { return own::log_gamma_q_da2(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da3(double a, double x) { return own::log_gamma_q_da3(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da4(double a, double x) { return own::log_gamma_q_da4(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da5(double a, double x) { return own::log_gamma_q_da5(a, x); }
LAPLACE_HD_FORCEINLINE double log_gamma_q_da6(double a, double x) { return own::log_gamma_q_da6(a, x); }

}  // namespace laplace::math
