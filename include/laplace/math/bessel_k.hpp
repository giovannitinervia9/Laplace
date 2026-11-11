// The modified Bessel function of the second kind of real order, in log form,
// with its derivatives in the argument (laplace::math):
//
//   log_bessel_k(nu, x)        log K_nu(x), x > 0 (K_-nu = K_nu)
//   log_bessel_k_dx1 ... _dx7  d^n/dx^n log K_nu(x), n = 1..7
//
// The pseudo-Huber distributions use K_0 ... K_4 at x = sqrt(nu_ph), the
// Poisson-inverse Gaussian log K_(y - 1/2)(alpha) with derivatives in alpha;
// the seventh derivative is what the fifth derivative of the latter's expected
// information needs. x <= 0 or NaN gives NaN; x = +inf gives log K = -inf,
// d1 = -1 and zero for the higher orders.
//
// The value. With n = round(nu) and mu = nu - n in [-1/2, 1/2], K_mu and
// K_(mu+1) come from Temme's series for x <= 2 and from Steed's continued
// fraction (scaled by e^x, no overflow) for x > 2 (both adapted from
// Boost.Math, boost/math/special_functions/detail/bessel_ik.hpp, Boost
// Software License 1.0), then the forward recurrence on the ratio
// r_k = K_(mu+k+1)/K_(mu+k), r_(k+1) = 2 (mu + k + 1)/x + 1/r_k, which is
// stable for K, gives log K_nu = log K_mu + sum_k log r_k (the product kept in
// one double, rescaled). For nu >= nu_debye the uniform (Debye) expansion
// (DLMF 10.41.4), with z = x/nu, t = (1 + z^2)^(-1/2),
// eta = sqrt(1 + z^2) + log(z / (1 + sqrt(1 + z^2))):
//   log K_nu(x) = log(pi / (2 nu))/2 - nu eta - log(1 + z^2)/4 + log P(t),
//   P(t) = sum_k (-1)^k u_k(t) / nu^k,
// the u_k exact rationals rounded once (tools/reference/bessel_k.py coefs).
//
// The derivatives in x, three regimes:
//   nu >= nu_debye: the Debye form differentiated: d/dx = (1/nu) d/dz; the
//     derivatives of eta' = h = sqrt(1 + z^2)/z from h h' = -z^-3 by
//     Leibniz's rule, those of log(1 + z^2) from (1 + z^2) q' = 2z, and those of
//     log P(t(z)) from P's Taylor coefficients in t (a polynomial), their
//     logarithm as a power series, composed with t(z)'s Taylor series
//     ((1 + z^2) t' = -z t).
//   nu < nu_debye, x >= x_hankel: the asymptotic expansion
//     K_nu(x) ~ sqrt(pi/(2x)) e^-x S(x), S = sum_k a_k x^-k,
//     a_k = prod_(j<=k) (4 nu^2 - (2j - 1)^2) / (k! 8^k) (DLMF 10.40.2): the
//     derivatives of S term by term, then those of log S (S is close to 1, so
//     nothing cancels).
//   otherwise: g = d log K/dx = nu/x - K_(nu+1)/K_nu in double-double, and the
//     Riccati identity that follows from Bessel's equation,
//       g' = 1 + nu^2/x^2 - g/x - g^2,
//     differentiated n times, in double-double (it cancels: about a factor
//     max(nu, x) per order). The ratio K_(mu+1)/K_mu in double-double comes
//     from Steed's continued fraction for x >= x_cf, from Temme's series in
//     double below (the forward recurrence in double-double then damps its
//     error by about x^2 / (4 (mu + k)^2) per step).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::math {

namespace own {

namespace detail_bessel_k {

namespace ddm = laplace::math::detail::ddm;
using ddm::dd;

constexpr double eps = 2.220446049250313e-16;
constexpr double pi = 3.14159265358979323846;
constexpr double euler = 0.57721566490153286061;
constexpr double log_half_pi = 0.45158270528945486473;  // log(pi/2)

// clang-format off
#define LAPLACE_BESSEL_K_DEBYE \
    0x1.0000000000000p+0, 0x1.0000000000000p-3, -0x1.aaaaaaaaaaaabp-3, 0x1.2000000000000p-4, \
    -0x1.9aaaaaaaaaaabp-2, 0x1.5638e38e38e39p-2, 0x1.2c00000000000p-4, -0x1.c84cccccccccdp-1, \
    0x1.d8b1c71c71c72p+0, -0x1.069ba781948b1p+0, 0x1.cb60000000000p-4, -0x1.2e9a666666666p+1, \
    0x1.1940800000000p+3, -0x1.669fc3f35ba78p+3, 0x1.2ada78a021b64p+2, 0x1.d11e000000000p-3, \
    -0x1.d79a53a83a83bp+2, 0x1.5447ad6c16c17p+5, -0x1.6f45e11c71c72p+6, 0x1.528b7ca566307p+6, \
    -0x1.c364a631dd95fp+4, 0x1.251ee80000000p-1, -0x1.a7dce636db6dbp+4, 0x1.b4618ac15dc91p+7, \
    -0x1.5dca313ad82d8p+9, 0x1.08ff639300000p+10, -0x1.7ea050e044d42p+9, 0x1.a923e815a1cf4p+7, \
    0x1.ba4c598000000p+0, -0x1.b05d1a13b6db7p+6, 0x1.2c39c95483d71p+10, -0x1.4b9a5a063f1c7p+12, \
    0x1.6c3b258dcc4bep+13, -0x1.a8946669c5f9bp+13, 0x1.f7db8e0e6ff83p+12, -0x1.dfdd4a56e48aep+10, \
    0x1.84bd1aa980000p+2, -0x1.edea5169e2492p+8, 0x1.bc583a953f412p+12, -0x1.41d14f581555cp+15, \
    0x1.dd58770920853p+16, -0x1.8d4416b11fe98p+17, 0x1.7811802863395p+17, -0x1.7ad4992fff6c7p+16, \
    0x1.3bb12a52aa2fbp+14, 0x1.8616a64f6c000p+4, -0x1.387a934e97623p+11, 0x1.614589b7ecd85p+15, \
    -0x1.43df4b09fcb1fp+18, 0x1.35a8d45f867f0p+20, -0x1.5773d9d00c99dp+21, 0x1.cb623a6199ae4p+21, \
    -0x1.6df7ff592a81cp+21, 0x1.404139d5a8d89p+20, -0x1.da73980d20117p+17, 0x1.b8118d37ff700p+6, \
    -0x1.b1f0b7d0cbfb1p+13, 0x1.2cf699e52c822p+18, -0x1.540a91065230fp+21, 0x1.958a7e55353d9p+23, \
    -0x1.1e9d645493e4cp+25, 0x1.fa2b20232a522p+25, -0x1.1ab04f0d89c04p+26, 0x1.84bccd3f0fa29p+25, \
    -0x1.2cb3c31e51931p+24, 0x1.90efaed3176ecp+21, 0x1.13aafea4e5774p+9, -0x1.48256f009b97ep+16, \
    0x1.11e5c16c629afp+21, -0x1.7571ceb9ca037p+24, 0x1.0ef6a77985642p+27, -0x1.d8ead78466863p+28, \
    0x1.07e453034ac45p+30, -0x1.827ee7a06eeffp+30, 0x1.7268078e48462p+30, -0x1.bff876bd73df6p+29, \
    0x1.367d9d22f8e58p+28, -0x1.785a32d50ea99p+25, 0x1.7bc2e57729724p+11, -0x1.0c7a4a7b78e16p+19, \
    0x1.096da38dd1835p+24, -0x1.ad5adfbc76170p+27, 0x1.73c2e3e3845c1p+30, -0x1.8733ea609e897p+32, \
    0x1.0b89e3d8c9f56p+34, -0x1.ec227ad1733f1p+34, 0x1.338fb49d78209p+35, -0x1.0207616f8514bp+35, \
    0x1.1679daa552eedp+34, -0x1.5dab67540d45ep+32, 0x1.84858f40f24dap+29, 0x1.1d47059b0d98ap+14, \
    -0x1.d8a2cb8a63829p+21, 0x1.110d4e9701237p+27, -0x1.0258a06e72954p+31, 0x1.06c7289bb5702p+34, \
    -0x1.475491eb205f5p+36, 0x1.0c20dd26c89a3p+38, -0x1.2c857cd0fac0cp+39, 0x1.d575cfc9e2c3ap+39, \
    -0x1.ff7885a2e2725p+39, 0x1.7d7a79bfd9279p+39, -0x1.73aff169e1ddbp+38, 0x1.aa8a07524069bp+36, \
    -0x1.b579e01fd909fp+33, 0x1.d0366d1f2a1fcp+16, -0x1.bd61241f49dd0p+24, 0x1.294f68360d03fp+30, \
    -0x1.452fdce361dedp+34, 0x1.7f8bafc20347bp+37, -0x1.16981c00442e8p+40, 0x1.0c7ef86c5c487p+42, \
    -0x1.6692d03f4fc93p+43, 0x1.53c36f1e925d7p+44, -0x1.ccd1190f05129p+44, 0x1.bcc7f72fa3f52p+44, \
    -0x1.2a8f68053f670p+44, 0x1.091de1749afccp+43, -0x1.18214409fe408p+41, 0x1.0aca592e16b75p+38, \
    0x1.96ab69ba805e8p+19, -0x1.bf624170b648dp+27, 0x1.55b4c489b95fdp+33, -0x1.abb30c9d41f8cp+37, \
    0x1.214acc7d910afp+41, -0x1.e4231a32338f6p+43, 0x1.0e915e49881c9p+46, -0x1.a71b10ac0f97ep+47, \
    0x1.dbb73479600cap+48, -0x1.85ed9a23ccb78p+49, 0x1.d30151d4d5254p+49, -0x1.947f1290b1214p+49, \
    0x1.ed9e6cefe60bcp+48, -0x1.92a1f52c0b7a6p+47, 0x1.8a317a4459106p+45, -0x1.5e64de75a4806p+42, \
    0x1.7da65df946f8bp+22, -0x1.dd3ad540c4610p+30, 0x1.9d7f5fdd32dd4p+36, -0x1.258c4c70506d6p+41, \
    0x1.c337b3c824076p+44, -0x1.ae77ec0689828p+47, 0x1.13bb05a2903aep+50, -0x1.f1df281e71adfp+51, \
    0x1.4678cb08b74acp+53, -0x1.3c7354c38a71ap+54, 0x1.c8b65d444fac0p+54, -0x1.e986473674e26p+54, \
    0x1.80787763a0616p+54, -0x1.ae03f99bb5229p+53, 0x1.44560eec11e37p+52, -0x1.27d0e84f1180ep+50, \
    0x1.ed06d883c7d6cp+46
// clang-format on
#if defined(__CUDACC__)
static __constant__ double debye_dev[153] = {LAPLACE_BESSEL_K_DEBYE};
#endif
inline constexpr double debye_host[153] = {LAPLACE_BESSEL_K_DEBYE};
// c_(k,i), u_k(t) = t^k sum_i c_(k,i) t^(2i), k = 0..16
LAPLACE_HD_FORCEINLINE double debye(int k, int i) {
  KOKKOS_IF_ON_DEVICE((return debye_dev[k * (k + 1) / 2 + i];))
  KOKKOS_IF_ON_HOST((return debye_host[k * (k + 1) / 2 + i];))
}
constexpr int debye_terms = 16;

// nu_enough(k - 1), k = 1..16: from this order on, the Debye terms 0..k-1
// suffice for the value and the first seven derivatives (the first omitted
// term is below 2^-60 of P at every t in [0, 1]; tools/reference/bessel_k.py coefs)
LAPLACE_HD_FORCEINLINE double nu_enough(int k) {
  constexpr double v[16] = {7.205759403792794e+17, 2420058081.000543, 4722780.547157874, 229918.22226045706,
                            37588.60581807104,     10435.461591054232, 4027.9748767097703, 1932.5269253457743,
                            1078.0705545289732,    670.4927152825686,  452.1805174246621,  324.4339238723952,
                            244.32632955726294,    191.23532265498693, 154.42936574595012, 127.94615677318323};
  return v[k];
}

struct params {
  double nu_debye = 20.0;   // Debye expansion from this order
  double x_hankel = 200.0;  // below nu_debye: asymptotic expansion in 1/x from here
  double x_cf = 1.0;        // the ratio in double-double by Steed's fraction from here
  bool wallis = true;       // that fraction by its convergents (else in Steed's form)
};

LAPLACE_HD_FORCEINLINE double nan() { return laplace::math::detail::quiet_nan(); }

// ---------------------------------------------------------------- Temme and Steed

// K_v(x) and K_(v+1)(x) by Temme's series, |v| <= 1/2, 0 < x <= 2
LAPLACE_HD void temme(double v, double x, double& K, double& K1) {
  const double gp = boost::math::tgamma1pm1(v, laplace::math::detail::boost_policy());
  const double gm = boost::math::tgamma1pm1(-v, laplace::math::detail::boost_policy());
  const double a = laplace::math::log(0.5 * x);
  const double b = laplace::math::exp(v * a);
  const double sigma = -a * v;
  const double c = Kokkos::fabs(v) < eps ? 1.0 : Kokkos::sin(pi * v) / (v * pi);
  const double d = Kokkos::fabs(sigma) < eps ? 1.0 : Kokkos::sinh(sigma) / sigma;
  const double gamma1 = Kokkos::fabs(v) < eps ? -euler : (0.5 / v) * (gp - gm) * c;
  const double gamma2 = (2.0 + gp + gm) * c / 2.0;
  double p = (gp + 1.0) / (2.0 * b);
  double q = (1.0 + gm) * b / 2.0;
  double f = (Kokkos::cosh(sigma) * gamma1 + d * (-a) * gamma2) / c;
  double coef = 1.0, s = f, s1 = p;
  for (int k = 1; k < 200; ++k) {
    f = (k * f + p + q) / (double(k) * k - v * v);
    p /= k - v;
    q /= k + v;
    const double h = p - k * f;
    coef *= x * x / (4.0 * k);
    s += coef * f;
    s1 += coef * h;
    if (Kokkos::fabs(coef * f) <= eps * Kokkos::fabs(s)) break;
  }
  K = s;
  K1 = 2.0 * s1 / x;
}

// log K_v(x) and K_(v+1)/K_v by Steed's algorithm, |v| <= 1/2, x > 2 (in the
// scaled form: no e^-x is formed)
LAPLACE_HD void steed(double v, double x, double& logK, double& ratio) {
  double a = v * v - 0.25, b = 2.0 * (x + 1.0);
  double D = 1.0 / b, f = D, delta = D;
  double prev = 0.0, cur = 1.0;
  double C = -a, Q = -a;
  double S = 1.0 + Q * delta;
  for (int k = 2; k < 1000; ++k) {
    a -= 2.0 * (k - 1);
    b += 2.0;
    D = 1.0 / (b + a * D);
    delta *= b * D - 1.0;
    f += delta;
    const double q = (prev - (b - 2.0) * cur) / a;
    prev = cur, cur = q;
    C *= -a / k;
    Q += C * q;
    S += Q * delta;
    if (q < eps) {  // rescale against underflow (as Boost does)
      C *= q, prev /= q, cur /= q;
    }
    if (Kokkos::fabs(Q * delta) <= eps * Kokkos::fabs(S)) break;
  }
  logK = 0.5 * (log_half_pi - laplace::math::log(x)) - x - laplace::math::log(S);
  ratio = (0.5 + v + x + (v * v - 0.25) * f) / x;
}

// K_(v+1)/K_v in double-double from the continued fraction of Steed's
// algorithm alone, f = 1/(b_1 + a_2/(b_2 + a_3/(b_3 + ...))) with
// b_k = 2 (x + k), a_k = v^2 - (k - 1/2)^2, evaluated forwards by its
// convergents P_k/Q_k (P_k = b_k P_(k-1) + a_k P_(k-2), likewise Q_k): only
// products, one division at the end (Steed's own form divides at every step,
// three times the cost in double-double). The number of steps is twice what
// the same fraction needs in double, plus four (measured at 106 bits with
// mpmath: the convergents then agree with the limit to about 1e-30); P and Q
// are rescaled by 2^-500 when they grow past 2^500.
LAPLACE_HD dd steed_ratio_wallis(double v, double x) {
  int n = 2;
  {
    double a = v * v - 0.25, b = 2.0 * (x + 1.0), D = 1.0 / b, f = D, delta = D;
    for (int k = 2; k < 3000; ++k) {
      a -= 2.0 * (k - 1);
      b += 2.0;
      D = 1.0 / (b + a * D);
      delta *= b * D - 1.0;
      f += delta;
      n = k;
      if (Kokkos::fabs(delta) <= eps * Kokkos::fabs(f)) break;
    }
  }
  n = 2 * n + 4;
  const dd vv = ddm::two_prod(v, v);
  const dd a0 = vv - 0.25;
  dd a = a0, b = (dd(x) + 1.0) * 2.0;
  dd P0(0.0), P1(1.0), Q0(1.0), Q1 = b;
  for (int k = 2; k <= n; ++k) {
    a = a - 2.0 * (k - 1);
    b = b + 2.0;
    const dd P2 = P1 * b + P0 * a, Q2 = Q1 * b + Q0 * a;
    P0 = P1, P1 = P2, Q0 = Q1, Q1 = Q2;
    if (Kokkos::fabs(Q1.hi) > 0x1p500) {
      P0 = P0 * 0x1p-500, P1 = P1 * 0x1p-500, Q0 = Q0 * 0x1p-500, Q1 = Q1 * 0x1p-500;
    }
  }
  const dd f = P1 / Q1;
  return ((a0 * f + (dd(x) + 0.5)) + v) / x;
}

// K_(v+1)/K_v in double-double from the continued fraction f of Steed's
// algorithm alone (it converges faster than the sum S), |v| <= 1/2
LAPLACE_HD dd steed_ratio_dd(double v, double x) {
  const dd vv = ddm::two_prod(v, v);
  dd a = vv - 0.25;
  const dd a0 = a;
  dd b(2.0 * (x + 1.0));  // exact up to the rounding of x + 1: carried in dd
  b = (dd(x) + 1.0) * 2.0;
  dd D = dd(1.0) / b, f = D, delta = D;
  for (int k = 2; k < 2000; ++k) {
    a = a - 2.0 * (k - 1);
    b = b + 2.0;
    D = dd(1.0) / (b + a * D);
    delta = delta * (b * D - 1.0);
    f = f + delta;
    if (Kokkos::fabs(delta.hi) <= 0x1p-108 * Kokkos::fabs(f.hi)) break;
  }
  return ((a0 * f + (dd(x) + 0.5)) + v) / x;
}

// ---------------------------------------------------------------- the value

// log K_nu(x) by Temme or Steed and the forward recurrence; r = K_(nu+1)/K_nu
LAPLACE_HD double log_k_rec(double nu, double x, double& r) {
  const int n = int(Kokkos::floor(nu + 0.5));
  const double mu = nu - n;
  double lk;
  if (x <= 2.0) {
    double k0, k1;
    temme(mu, x, k0, k1);
    lk = laplace::math::log(k0);
    r = k1 / k0;
  } else {
    steed(mu, x, lk, r);
  }
  double prod = 1.0, acc = 0.0;
  for (int k = 1; k <= n; ++k) {
    prod *= r;
    if (prod > 1e280) {
      acc += laplace::math::log(prod);
      prod = 1.0;
    }
    r = 2.0 * (mu + k) / x + 1.0 / r;
  }
  return lk + (acc + laplace::math::log(prod));
}

// the pieces of the Debye form at z = x/nu
LAPLACE_HD double log_k_debye(double nu, double x) {
  const double z = x / nu;
  const double s = Kokkos::sqrt(1.0 + z * z), t = 1.0 / s, t2 = t * t;
  const double eta = s + laplace::math::log(z / (1.0 + s));
  // P - 1 = sum_(k >= 1) (-1)^k u_k(t) / nu^k
  double pm1 = 0.0, w = 1.0, tk = 1.0;
  for (int k = 1; k <= debye_terms; ++k) {
    w *= -1.0 / nu;
    tk *= t;
    double u = debye(k, k);
    for (int i = k - 1; i >= 0; --i) u = u * t2 + debye(k, i);
    const double term = w * tk * u;
    pm1 += term;
    if (Kokkos::fabs(term) <= 0x1p-60) break;
  }
  return 0.5 * (log_half_pi - laplace::math::log(nu)) - nu * eta - 0.25 * Kokkos::log1p(z * z) + Kokkos::log1p(pm1);
}

LAPLACE_HD double log_k(double nu, double x, const params& p) {
  if (!(x >= 0.0) || nu != nu) return nan();
  nu = Kokkos::fabs(nu);
  if (x == 0.0) return laplace::math::detail::pos_inf();
  if (!(x < 1e308)) return -x;
  if (nu >= p.nu_debye) return log_k_debye(nu, x);
  double r;
  return log_k_rec(nu, x, r);
}

// ---------------------------------------------------------------- derivatives

// d^m/dx^m log S with S^(0..N) given (Faa di Bruno for log, by the recursion
// L^(n) = (S^(n) - sum_(k=1)^(n-1) C(n-1, k-1) L^(k) S^(n-k)) / S)
template <int N>
LAPLACE_HD_FORCEINLINE double log_derivative(const double (&S)[N + 1]) {
  double L[N + 1];
  L[0] = 0.0;
  for (int n = 1; n <= N; ++n) {
    double s = S[n], c = 1.0;  // C(n-1, k-1)
    for (int k = 1; k < n; ++k) {
      s -= c * L[k] * S[n - k];
      c = c * double(n - k) / double(k);
    }
    L[n] = s / S[0];
  }
  return L[N];
}

// Taylor coefficients of log a from those of a (a[0] > 0), orders 1..N
template <int N>
LAPLACE_HD_FORCEINLINE void series_log(const double (&a)[N + 1], double (&l)[N + 1]) {
  l[0] = 0.0;
  for (int m = 1; m <= N; ++m) {
    double s = 0.0;
    for (int k = 1; k < m; ++k) s += k * l[k] * a[m - k];
    l[m] = (a[m] - s / m) / a[0];
  }
}

// d^N/dx^N log K by the Riccati identity in double-double, from g
template <int N>
LAPLACE_HD double riccati(const dd& g, double nu, double x) {
  const dd ix = dd(1.0) / dd(x);
  const dd nu2 = ddm::two_prod(nu, nu);
  dd inv[N > 1 ? N - 1 : 1], inv2[N > 1 ? N - 1 : 1];  // (1/x)^(m), (1/x^2)^(m)
  {
    dd p = ix, p2 = ix * ix;
    double f = 1.0, f2 = 1.0;  // m!, (m+1)!
    for (int m = 0; m < N - 1; ++m) {
      f2 *= double(m + 1);
      inv[m] = (m % 2) ? -(p * f) : p * f;
      inv2[m] = (m % 2) ? -(p2 * f2) : p2 * f2;
      p = p * ix;
      p2 = p2 * ix;
      f *= double(m + 1);
    }
  }
  dd d[N];
  d[0] = g;
  for (int n = 0; n < N - 1; ++n) {
    dd s = nu2 * inv2[n];
    double c = 1.0;  // C(n, j)
    for (int j = 0; j <= n; ++j) {
      s = s - (d[j] * (inv[n - j] + d[n - j])) * c;
      c = c * double(n - j) / double(j + 1);
    }
    d[n + 1] = (n == 0) ? s + 1.0 : s;
  }
  return d[N - 1].hi;
}

// g = nu/x - K_(nu+1)/K_nu in double-double
LAPLACE_HD dd g_dd(double nu, double x, const params& p) {
  const int n = int(Kokkos::floor(nu + 0.5));
  const double mu = nu - n;
  dd r;
  if (x >= p.x_cf) {
    r = p.wallis ? steed_ratio_wallis(mu, x) : steed_ratio_dd(mu, x);
  } else {
    double k0, k1;
    temme(mu, x, k0, k1);
    r = dd(k1 / k0);
  }
  const dd xd(x);
  for (int k = 1; k <= n; ++k) r = (dd(mu) + double(k)) * 2.0 / xd + dd(1.0) / r;
  return dd(nu) / xd - r;
}

// d^N/dx^N log K from the asymptotic expansion in 1/x
template <int N>
LAPLACE_HD double hankel(double nu, double x) {
  const double mu4 = 4.0 * nu * nu, u = 1.0 / x;
  double S[N + 1];
  for (int m = 0; m <= N; ++m) S[m] = 0.0;
  double a = 1.0, uk = 1.0, prev = 1e300;
  for (int k = 0; k < 200; ++k) {
    if (k > 0) {
      a *= (mu4 - double(2 * k - 1) * (2 * k - 1)) / (8.0 * k);
      uk *= u;
    }
    const double t = a * uk;
    const double at = Kokkos::fabs(t);
    if (k > 2 && at > prev) break;  // the divergent tail
    // d^m x^-k = (-1)^m k (k+1) ... (k+m-1) x^-(k+m)
    double um = t, fac = 1.0;
    for (int m = 0; m <= N; ++m) {
      S[m] += (m % 2 ? -fac : fac) * um;
      fac *= double(k + m);
      um *= u;
    }
    if (at == 0.0 || at <= 0x1p-60) break;
    prev = at;
  }
  double base = 0.0;  // d^N of -log(x)/2 - x
  {
    double f = 1.0;
    for (int j = 2; j < N; ++j) f *= j;
    double xn = 1.0;
    for (int j = 0; j < N; ++j) xn *= x;
    base = (N % 2 ? -0.5 : 0.5) * f / xn;
    if (N == 1) base -= 1.0;
  }
  return base + log_derivative<N>(S);
}

// d^N/dx^N log K from the Debye form
template <int N>
LAPLACE_HD double debye_deriv(double nu, double x) {
  const double z = x / nu, z2 = z * z, opz = 1.0 + z2;
  const double s = Kokkos::sqrt(opz), t0 = 1.0 / s;
  // h^(m) = d^m/dz^m (s / z), from h h' = -z^-3: h^(n+1) = (d^n(-z^-3) - sum_(i=1)^n C(n,i) h^(i) h^(n+1-i)) / h
  double h[N];
  h[0] = s / z;
  {
    double zp = 1.0 / (z2 * z), fac = 1.0;  // d^n z^-3 = (-1)^n (n+2)!/2 z^-(3+n)
    for (int n = 0; n < N - 1; ++n) {
      fac *= double(n + 2) / (n == 0 ? 2.0 : 1.0);
      const double dz3 = (n % 2 ? -fac : fac) * zp;  // d^n z^-3
      zp /= z;
      double acc = -dz3, c = 1.0;
      for (int i = 1; i <= n; ++i) {
        c = c * double(n - i + 1) / double(i);
        acc -= c * h[i] * h[n + 1 - i];
      }
      h[n + 1] = acc / h[0];
    }
  }
  // q^(m) = d^m log(1 + z^2): (1 + z^2) q^(n+1) + 2 n z q^(n) + n (n-1) q^(n-1) = d^n(2z)
  double q[N + 1];
  q[0] = Kokkos::log1p(z2);
  q[1] = 2.0 * z / opz;
  for (int n = 1; n < N; ++n) q[n + 1] = ((n == 1 ? 2.0 : 0.0) - 2.0 * n * z * q[n] - n * (n - 1.0) * q[n - 1]) / opz;
  // P's Taylor coefficients in t at t0: P = sum_k (-1)^k nu^-k t^k sum_i c_(k,i) t^(2i);
  // the terms 0..k-1 suffice from nu = nu_enough(k - 1) on
  double Pc[N + 1];
  for (int j = 0; j <= N; ++j) Pc[j] = 0.0;
  {
    double tp[3 * debye_terms + 1];  // t0^e
    tp[0] = 1.0;
    for (int e = 1; e <= 3 * debye_terms; ++e) tp[e] = tp[e - 1] * t0;
    double w = 1.0;
    for (int k = 0; k <= debye_terms; ++k) {
      if (k > 0) {
        if (nu >= nu_enough(k - 1)) break;
        w *= -1.0 / nu;
      }
      for (int i = 0; i <= k; ++i) {
        const int e = k + 2 * i;
        const double c = w * debye(k, i);
        // j-th Taylor coefficient of c t^e at t0: c C(e, j) t0^(e-j)
        double binom = 1.0;
        for (int j = 0; j <= N && j <= e; ++j) {
          Pc[j] += c * binom * tp[e - j];
          binom = binom * double(e - j) / double(j + 1);
        }
      }
    }
  }
  double LP[N + 1];
  series_log<N>(Pc, LP);
  // t(z)'s Taylor coefficients: (1 + z^2) t^(n+1) + 2 n z t^(n) + n (n-1) t^(n-1) = -(z t^(n) + n t^(n-1))
  double T[N + 1];
  {
    double dt[N + 1];
    dt[0] = t0;
    for (int n = 0; n < N; ++n) {
      const double tn1 = n >= 1 ? dt[n - 1] : 0.0;
      dt[n + 1] = (-(z * dt[n] + n * tn1) - 2.0 * n * z * dt[n] - n * (n - 1.0) * tn1) / opz;
    }
    double f = 1.0;
    for (int n = 0; n <= N; ++n) {
      if (n > 0) f *= n;
      T[n] = dt[n] / f;
    }
  }
  // compose: log P(t(z0 + e)) = sum_j LP_j (dt)^j, dt = sum_(i>=1) T_i e^i; coefficient of e^N
  double comp = 0.0;
  {
    double pw[N + 1], nw[N + 1];
    for (int i = 0; i <= N; ++i) pw[i] = i == 0 ? 1.0 : 0.0;
    for (int j = 1; j <= N; ++j) {
      for (int i = 0; i <= N; ++i) {
        double acc = 0.0;
        for (int l = 1; l <= i; ++l) acc += T[l] * pw[i - l];
        nw[i] = acc;
      }
      for (int i = 0; i <= N; ++i) pw[i] = nw[i];
      comp += LP[j] * pw[N];
    }
  }
  double fN = 1.0;
  for (int j = 2; j <= N; ++j) fN *= j;
  double nuN = 1.0;
  for (int j = 0; j < N; ++j) nuN *= nu;
  return (-nu * h[N - 1] - 0.25 * q[N] + fN * comp) / nuN;
}

template <int N>
LAPLACE_HD double derivative(double nu, double x, const params& p) {
  if (!(x > 0.0) || nu != nu) return nan();
  nu = Kokkos::fabs(nu);
  if (!(x < 1e308)) return N == 1 ? -1.0 : 0.0;
  if (nu >= p.nu_debye) return debye_deriv<N>(nu, x);
  if (x >= p.x_hankel) return hankel<N>(nu, x);
  return riccati<N>(g_dd(nu, x, p), nu, x);
}

}  // namespace detail_bessel_k

/**
 * \ingroup bessel_k
 * \brief Natural logarithm of the modified Bessel function of the second kind, \f$\log K_\nu(x)\f$.
 *
 * Computes \f$\log K_\nu(x)\f$ for real order \f$\nu\f$ and argument \f$x > 0\f$.
 * The modified Bessel function of the second kind is defined by
 * \f[
 * K_\nu(x) = \int_0^\infty e^{-x \cosh t} \cosh(\nu t) \, \mathrm{d}t
 * \f]
 * where \f$\nu \in \mathbb{R}\f$ is the order, \f$x \in (0, \infty)\f$ is the argument,
 * and \f$t \in [0, \infty)\f$ is the integration variable.
 *
 * The function is symmetric in the order: \f$K_{-\nu}(x) = K_\nu(x)\f$.
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain.
 * - \f$\nu < 0\f$: evaluated as \f$|\nu|\f$ by symmetry \f$\log K_{-\nu}(x) = \log K_\nu(x)\f$.
 * - \f$x < 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$+\infty\f$.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$-x\f$ (\f$-\infty\f$).
 *
 * \par Measured accuracy
 * Maximum error 4.8 ulp relative to the conditioning scale \f$\max(|\log K|, |x g|, 1)\f$
 * on both CPU and GPU (test bound 7.5 ulp), where \f$g = \frac{\mathrm{d}}{\mathrm{d}x}\log K_\nu(x)\f$.
 *
 * \see bessel_k, log_bessel_k_dx1()
 */
LAPLACE_HD double log_bessel_k(double nu, double x) { return detail_bessel_k::log_k(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief First derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the first derivative:
 * \f[
 * \frac{\mathrm{d}}{\mathrm{d}x} \log K_\nu(x) = \frac{\nu}{x} - \frac{K_{\nu+1}(x)}{K_\nu(x)}
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}}{\mathrm{d}x}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$-1.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 8.9 ulp (gcc); test bound 9.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx2()
 */
LAPLACE_HD double log_bessel_k_dx1(double nu, double x) { return detail_bessel_k::derivative<1>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Second derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the second derivative:
 * \f[
 * \frac{\mathrm{d}^2}{\mathrm{d}x^2} \log K_\nu(x) = 1 + \frac{\nu^2}{x^2} - \frac{1}{x}\frac{\mathrm{d}}{\mathrm{d}x}\log K_\nu(x) - \left(\frac{\mathrm{d}}{\mathrm{d}x}\log K_\nu(x)\right)^2
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^2}{\mathrm{d}x^2}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 44.4 ulp (gcc) / 37.4 ulp (CUDA); test bound 66.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx1(), log_bessel_k_dx3()
 */
LAPLACE_HD double log_bessel_k_dx2(double nu, double x) { return detail_bessel_k::derivative<2>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Third derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the third derivative:
 * \f[
 * \frac{\mathrm{d}^3}{\mathrm{d}x^3} \log K_\nu(x)
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^3}{\mathrm{d}x^3}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 44.0 ulp (gcc) / 37.0 ulp (CUDA); test bound 66.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx2(), log_bessel_k_dx4()
 */
LAPLACE_HD double log_bessel_k_dx3(double nu, double x) { return detail_bessel_k::derivative<3>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Fourth derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * \frac{\mathrm{d}^4}{\mathrm{d}x^4} \log K_\nu(x)
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^4}{\mathrm{d}x^4}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 24.1 ulp (gcc) / 25.6 ulp (CUDA); test bound 36.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx3(), log_bessel_k_dx5()
 */
LAPLACE_HD double log_bessel_k_dx4(double nu, double x) { return detail_bessel_k::derivative<4>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Fifth derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * \frac{\mathrm{d}^5}{\mathrm{d}x^5} \log K_\nu(x)
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^5}{\mathrm{d}x^5}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 12.4 ulp (gcc); test bound 17.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx4(), log_bessel_k_dx6()
 */
LAPLACE_HD double log_bessel_k_dx5(double nu, double x) { return detail_bessel_k::derivative<5>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Sixth derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * \frac{\mathrm{d}^6}{\mathrm{d}x^6} \log K_\nu(x)
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^6}{\mathrm{d}x^6}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 21.9 ulp (gcc); test bound 33.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx5(), log_bessel_k_dx7()
 */
LAPLACE_HD double log_bessel_k_dx6(double nu, double x) { return detail_bessel_k::derivative<6>(nu, x, detail_bessel_k::params{}); }

/**
 * \ingroup bessel_k
 * \brief Seventh derivative in \f$x\f$ of \f$\log K_\nu(x)\f$.
 *
 * Computes the seventh derivative:
 * \f[
 * \frac{\mathrm{d}^7}{\mathrm{d}x^7} \log K_\nu(x)
 * \f]
 *
 * \param nu Real order \f$\nu \in \mathbb{R}\f$.
 * \param x Positive argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\frac{\mathrm{d}^7}{\mathrm{d}x^7}\log K_\nu(x)\f$.
 *
 * \par Domain and special values
 * - \f$x > 0, \nu \in \mathbb{R}\f$: standard domain, symmetric in \f$\nu\f$.
 * - \f$x \le 0\f$, or \f$x\f$ is NaN, or \f$\nu\f$ is NaN: returns quiet NaN.
 * - \f$x \ge 10^{308}\f$ (\f$x \to +\infty\f$): returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 31.5 ulp (gcc); test bound 48.0 ulp.
 *
 * \see bessel_k, log_bessel_k(), log_bessel_k_dx6()
 */
LAPLACE_HD double log_bessel_k_dx7(double nu, double x) { return detail_bessel_k::derivative<7>(nu, x, detail_bessel_k::params{}); }

}  // namespace own

using own::log_bessel_k;
using own::log_bessel_k_dx1;
using own::log_bessel_k_dx2;
using own::log_bessel_k_dx3;
using own::log_bessel_k_dx4;
using own::log_bessel_k_dx5;
using own::log_bessel_k_dx6;
using own::log_bessel_k_dx7;

}  // namespace laplace::math
