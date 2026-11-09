// The functions of the standard normal distribution as numerical functions:
// erfc, erfcx, Phi, log Phi, the derivatives of log Phi to order five, and the
// quantile Phi^-1 from p and from log p.
//
// Notation: phi is the standard normal density, Phi its distribution function,
// erfcx(x) = exp(x^2) erfc(x) the scaled complementary error function,
// R(t) = phi(t)/Phi(t) the inverse Mills ratio, which is d log Phi / dt.
//
// Everything is built on erfcx, which neither underflows nor cancels:
//   Phi(x)     = exp(-x^2/2) erfcx(-x/sqrt 2) / 2            (x < 0)
//   log Phi(x) = -x^2/2 + log(erfcx(-x/sqrt 2) / 2)          (x < 0)
//   R(t)       = sqrt(2/pi) / erfcx(-t/sqrt 2)                (t <= 0)
// with x^2/2 carried as an exact sum h + l, so that the rounding of x^2 (a
// relative error of x^2 eps in exp(-x^2/2)) does not enter. The rounding of
// x/sqrt 2 does not matter: erfcx varies slowly, its logarithmic derivative is
// about -1/z for large z.
//
// The implementations are Laplace's own (namespace own), measured against
// mpmath, the platform, Boost and CUDA in benchmarks/math_normal.cpp; the wrapper
// in laplace::math, at the end, uses them on every backend (docs/numerics.md,
// 2026-11-09).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/erf.hpp>

namespace laplace::math {

namespace own {

namespace detail_normal {

constexpr double sqrt_half = 0.70710678118654752440;       // 1/sqrt(2)
constexpr double sqrt2 = 1.41421356237309504880;
constexpr double inv_sqrt_pi = 0.56418958354775628695;     // 1/sqrt(pi)
constexpr double sqrt_2_over_pi = 0.79788456080286535588;  // sqrt(2/pi)
constexpr double inv_sqrt_2pi = 0.39894228040143267794;    // 1/sqrt(2 pi)
constexpr double log_half = -0.69314718055994530942;
constexpr double eps = 2.220446049250313e-16;

// p + e = u v exactly (the FMA on the device; Dekker's product on the host,
// where the build does not assume FMA hardware)
LAPLACE_HD_FORCEINLINE void two_prod(double u, double v, double& p, double& e) {
  p = u * v;
  KOKKOS_IF_ON_DEVICE((e = Kokkos::fma(u, v, -p);))
  KOKKOS_IF_ON_HOST((constexpr double split = 134217729.0;  // 2^27 + 1
                     const double cu = split * u, uh = cu - (cu - u), ul = u - uh;
                     const double cv = split * v, vh = cv - (cv - v), vl = v - vh;
                     e = ((uh * vh - p) + uh * vl + ul * vh) + ul * vl;))
}

// h + l = x^2 / 2 exactly (x/2 is exact for every normal x)
LAPLACE_HD_FORCEINLINE void half_square(double x, double& h, double& l) { two_prod(0.5 * x, x, h, l); }

// exp(-x^2/2) without the rounding of x^2
LAPLACE_HD_FORCEINLINE double exp_minus_half_square(double x) {
  double h, l;
  half_square(x, h, l);
  return math::exp(-h) * (1.0 - l);
}

template <int N>
LAPLACE_HD_FORCEINLINE double horner(const double (&c)[N], double z) {
  double s = c[N - 1];
  for (int i = N - 2; i >= 0; --i) s = s * z + c[i];
  return s;
}

}  // namespace detail_normal

// erfcx(x) for x >= 0. The rational approximations on [0, 0.5), [0.5, 1.5),
// [1.5, 2.5), [2.5, 4.5) and [4.5, 28) are Boost.Math's for erf and erfc in
// double precision (boost/math/special_functions/erf.hpp, erf_imp for 53 bits,
// Boost Software License), there written as erfc = exp(-z^2)/z (Y + P/Q);
// here without the factor exp(-z^2), which is what makes erfcx free of
// underflow. Above 28, the asymptotic series
//   erfcx(x) = 1/(x sqrt pi) sum_k (-1)^k (2k - 1)!! / (2 x^2)^k,
// whose ninth term is below 1e-19 there.
LAPLACE_HD double erfcx_pos(double z) {
  using namespace detail_normal;
  if (z < 0.5) {
    // erfcx = exp(z^2) (1 - erf z), erf z = z (Y + P(z^2)/Q(z^2))
    constexpr double Y = 1.044948577880859375;
    constexpr double P[] = {0.0834305892146531832907, -0.338165134459360935041, -0.0509990735146777432841,
                            -0.00772758345802133288487, -0.000322780120964605683831};
    constexpr double Q[] = {1.0, 0.455004033050794024546, 0.0875222600142252549554, 0.00858571925074406212772,
                            0.000370900071787748000569};
    const double zz = z * z;
    const double erf = z * (Y + horner(P, zz) / horner(Q, zz));
    return math::exp(zz) * (1.0 - erf);
  }
  if (z < 1.5) {
    constexpr double Y = 0.405935764312744140625;
    constexpr double P[] = {-0.098090592216281240205, 0.178114665841120341155,  0.191003695796775433986,
                            0.0888900368967884466578, 0.0195049001251218801359, 0.00180424538297014223957};
    constexpr double Q[] = {1.0,
                            1.84759070983002217845,
                            1.42628004845511324508,
                            0.578052804889902404909,
                            0.12385097467900864233,
                            0.0113385233577001411017,
                            0.337511472483094676155e-5};
    const double u = z - 0.5;
    return (Y + horner(P, u) / horner(Q, u)) / z;
  }
  if (z < 2.5) {
    constexpr double Y = 0.50672817230224609375;
    constexpr double P[] = {-0.0243500476207698441272, 0.0386540375035707201728,  0.04394818964209516296,
                            0.0175679436311802092299,  0.00323962406290842133584, 0.000235839115596880717416};
    constexpr double Q[] = {1.0, 1.53991494948552447182, 0.982403709157920235114, 0.325732924782444448493,
                            0.0563921837420478160373, 0.00410369723978904575884};
    const double u = z - 1.5;
    return (Y + horner(P, u) / horner(Q, u)) / z;
  }
  if (z < 4.5) {
    constexpr double Y = 0.5405750274658203125;
    constexpr double P[] = {0.00295276716530971662634, 0.0137384425896355332126,  0.00840807615555585383007,
                            0.00212825620914618649141, 0.000250269961544794627958, 0.113212406648847561139e-4};
    constexpr double Q[] = {1.0, 1.04217814166938418171, 0.442597659481563127003, 0.0958492726301061423444,
                            0.0105982906484876531489, 0.000479411269521714493907};
    const double u = z - 3.5;
    return (Y + horner(P, u) / horner(Q, u)) / z;
  }
  if (z < 28.0) {
    constexpr double Y = 0.5579090118408203125;
    constexpr double P[] = {0.00628057170626964891937, 0.0175389834052493308818, -0.212652252872804219852,
                            -0.687717681153649930619,  -2.5518551727311523996,   -3.22729451764143718517,
                            -2.8175401114513378771};
    constexpr double Q[] = {1.0, 2.79257750980575282228, 11.0567237927800161565, 15.930646027911794143,
                            22.9367376522880577224, 13.5064170191802889145, 5.48409182238641741584};
    const double u = 1.0 / z;
    return (Y + horner(P, u) / horner(Q, u)) / z;
  }
  // (-1)^k (2k - 1)!!, k = 0..8, in powers of y = 1/(2 z^2); y is 0 when 2 z^2 overflows
  constexpr double C[] = {1.0, -1.0, 3.0, -15.0, 105.0, -945.0, 10395.0, -135135.0, 2027025.0};
  const double y = 1.0 / (2.0 * z * z);
  return inv_sqrt_pi / z * horner(C, y);
}

// erfcx(x) = exp(x^2) erfc(x); overflows to infinity below x = -26.63
LAPLACE_HD double erfcx(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  if (x >= 0.0) return erfcx_pos(x);
  const double a = -x;
  if (a < 0.5) {
    // exp(a^2) (1 + erf a): erf a = 1 - exp(-a^2) erfcx(a), with no cancellation
    return 2.0 * math::exp(a * a) - erfcx_pos(a);
  }
  // 2 exp(a^2) - erfcx(a), exp(a^2) from the exact square
  double h, l;
  two_prod(a, a, h, l);
  const double e = math::exp(h);
  return 2.0 * (e + e * l) - erfcx_pos(a);
}

// erfc(x) = exp(-x^2) erfcx(x) for x >= 0, with x^2 carried exactly; 2 - erfc(-x) below
LAPLACE_HD double erfc(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  const double a = Kokkos::fabs(x);
  double h, l;
  two_prod(a, a, h, l);
  const double r = erfcx_pos(a) * (math::exp(-h) * (1.0 - l));
  return x >= 0.0 ? r : 2.0 - r;
}

// Phi(x)
LAPLACE_HD double Phi(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  if (x < 0.0) return 0.5 * erfcx_pos(-x * sqrt_half) * exp_minus_half_square(x);
  return 1.0 - 0.5 * erfcx_pos(x * sqrt_half) * exp_minus_half_square(x);
}

// log Phi(x), without underflow down to x = -1.9e154
LAPLACE_HD double log_Phi(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  if (x < 0.0) {
    double h, l;
    half_square(x, h, l);
    return -h + (math::log(0.5 * erfcx_pos(-x * sqrt_half)) - l);
  }
  return Kokkos::log1p(-0.5 * erfcx_pos(x * sqrt_half) * exp_minus_half_square(x));
}

// R(t) = phi(t)/Phi(t) = d log Phi / dt
LAPLACE_HD double inv_mills(double t) {
  using namespace detail_normal;
  if (t != t) return t;
  if (t <= 0.0) return sqrt_2_over_pi / erfcx_pos(-t * sqrt_half);
  return inv_sqrt_2pi * exp_minus_half_square(t) / Phi(t);
}

namespace detail_normal {

// d^k log Phi / dt^k at t, k = 1..5; the work is shared by the five public
// functions below, each of which asks for its own order only.
//
// For t >= t_switch: from R and the recursion R' = -R (t + R):
//   g1 = R, g2 = -R (t + R), g3 = -g2 (t + 2R) - R,
//   g4 = -g3 (t + 2R) - 2 g2 (1 + g2), g5 = -g4 (t + 2R) - 3 g3 (1 + 2 g2).
// Below, t + R cancels (both terms are of size |t|, their sum of size 1/|t|).
// There, with x = -t, log Phi(t) = -x^2/2 - log(2 pi)/2 + log M_0(x), where
// M_j(x) = int_0^inf s^j exp(-x s - s^2/2) ds, and the derivatives of
// log M_0 are (-1)^k kappa_k, the cumulants of the density proportional to
// exp(-x s - s^2/2) on s > 0 (statmodels7, pt_logphi.h and pt_enet_logz.h):
//   g1 = x + kappa_1, g2 = -1 + kappa_2, g_k = kappa_k for k >= 3.
// The moments come from the ratios r_j = M_j / M_(j-1), which satisfy
// r_j = j / (x + r_(j+1)) and are computed downward, where the recursion is
// stable. Here in the scaled form rho_j = x r_j = j / (1 + w rho_(j+1)),
// w = 1/x^2, so that the scaled moments x^j m_j stay of order j! for every x;
// the recursion starts at j = K + 1 from the root of rho (1 + w rho) = K + 1,
// its value for large j, and K = k + m0 + ceil(c2 w): each downward step
// damps the error of the start by about j w / (1 + w rho)^2, which is small
// for large x (m0 steps suffice) and close to 1 for x near 1, where the error
// falls like exp(-2 x sqrt K).
// kappa_k from the raw moments mu_0 = 1, mu_1, .., mu_k (k = 1..5), by the
// recursion kappa_n = mu_n - sum_(i=1)^(n-1) C(n-1, i-1) kappa_i mu_(n-i)
// (statmodels7, enet_cumulants in pt_enet_logz.h); measured more accurate here
// than the closed forms in powers of mu_1
LAPLACE_HD_FORCEINLINE double cumulant(int k, const double* mu) {
  double kap[6];
  for (int n = 1; n <= k; ++n) {
    double s = mu[n], binom = 1.0;  // C(n - 1, i - 1)
    for (int i = 1; i < n; ++i) {
      s -= binom * kap[i] * mu[n - i];
      binom = binom * (n - i) / i;
    }
    kap[n] = s;
  }
  return kap[k];
}

// g_k from kappa_k for x = -t
LAPLACE_HD_FORCEINLINE double g_from_kappa(int k, double x, double kappa) {
  if (k == 1) return x + kappa;
  if (k == 2) return -1.0 + kappa;
  return kappa;
}

LAPLACE_HD double log_Phi_dk(int k, double t, double t_switch, double c2, int m0) {
  if (t != t) return t;
  if (t >= t_switch) {
    const double R = inv_mills(t);
    if (k == 1) return R;
    const double g2 = -R * (t + R);
    if (k == 2) return g2;
    const double g3 = -g2 * (t + 2.0 * R) - R;
    if (k == 3) return g3;
    const double g4 = -g3 * (t + 2.0 * R) - 2.0 * g2 * (1.0 + g2);
    if (k == 4) return g4;
    return -g4 * (t + 2.0 * R) - 3.0 * g3 * (1.0 + 2.0 * g2);
  }
  const double x = -t;
  double mu[6] = {1.0, 0, 0, 0, 0, 0};
  const double ix = 1.0 / x, w = ix * ix;
  const double kk = c2 * w;
  const int K = k + m0 + (kk < 20000.0 ? static_cast<int>(Kokkos::ceil(kk)) : 20000);
  const double K1 = K + 1.0;
  double rho = 2.0 * K1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * K1 * w));
  double r[6] = {0, 0, 0, 0, 0, 0};
  for (int j = K; j >= 1; --j) {
    rho = j / (1.0 + w * rho);
    if (j <= k) r[j] = rho;
  }
  // scaled raw moments mu_j = x^j m_j give the scaled cumulants x^j kappa_j
  for (int j = 1; j <= k; ++j) mu[j] = mu[j - 1] * r[j];
  double scale = ix;
  for (int j = 1; j < k; ++j) scale *= ix;
  return g_from_kappa(k, x, cumulant(k, mu) * scale);
}

// Phi^-1(p) for 0 < p <= 1/2: Boost's erfc^-1, then in the tail (p < 0.1) one
// Newton step on log Phi, which brings it from 1.7-2.9 to below 1.6 ulps of
// the conditioned result; nearer the centre the step does not help (the error
// of log Phi there is as large as Boost's).
LAPLACE_HD double Phi_inv_lower(double p) {
  const double x = -sqrt2 * boost::math::erfc_inv(2.0 * p, math::detail::boost_policy());
  if (p >= 0.1) return x;
  return x - (log_Phi(x) - math::log(p)) / inv_mills(x);
}

}  // namespace detail_normal

// d^k log Phi / dt^k, one function per order: the recursion from R for
// t >= -1, the cumulants below, K = k + 20 + ceil(300 / t^2) downward steps
LAPLACE_HD double log_Phi_d1(double t) { return detail_normal::log_Phi_dk(1, t, -1.0, 300.0, 20); }
LAPLACE_HD double log_Phi_d2(double t) { return detail_normal::log_Phi_dk(2, t, -1.0, 300.0, 20); }
LAPLACE_HD double log_Phi_d3(double t) { return detail_normal::log_Phi_dk(3, t, -1.0, 300.0, 20); }
LAPLACE_HD double log_Phi_d4(double t) { return detail_normal::log_Phi_dk(4, t, -1.0, 300.0, 20); }
LAPLACE_HD double log_Phi_d5(double t) { return detail_normal::log_Phi_dk(5, t, -1.0, 300.0, 20); }

// Phi^-1(p)
LAPLACE_HD double Phi_inv(double p) {
  using namespace detail_normal;
  if (!(p > 0.0 && p < 1.0)) {
    if (p == 0.0) return -detail::pos_inf();
    if (p == 1.0) return detail::pos_inf();
    return detail::quiet_nan();
  }
  if (p <= 0.5) return Phi_inv_lower(p);
  return -Phi_inv_lower(1.0 - p);  // 1 - p is exact for p >= 1/2
}

// Phi^-1(exp(l)), for l = log p <= 0, also below the double range of p
LAPLACE_HD double Phi_inv_logp(double l) {
  using namespace detail_normal;
  if (!(l < 0.0)) {
    if (l == 0.0) return detail::pos_inf();
    return detail::quiet_nan();
  }
  if (l >= log_half) {
    // p >= 1/2: through q = 1 - p = -expm1(l), which keeps its relative precision
    return -Phi_inv_lower(-Kokkos::expm1(l));
  }
  double x;
  if (l > -700.0) {
    x = -sqrt2 * boost::math::erfc_inv(2.0 * math::exp(l), math::detail::boost_policy());
  } else {
    // log Phi(x) ~ -x^2/2 - log(-x sqrt(2 pi)), so x^2 ~ -2 l - log(-4 pi l)
    x = -Kokkos::sqrt(-2.0 * l - math::log(-12.566370614359172954 * l));
  }
  for (int it = 0; it < 8; ++it) {
    const double dx = (log_Phi(x) - l) / inv_mills(x);
    x -= dx;
    if (Kokkos::fabs(dx) <= 4.0 * eps * Kokkos::fabs(x)) break;
  }
  return x;
}

}  // namespace own

// The wrapper: Laplace's own on every backend (CUDA's erfcx, normcdf and
// normcdfinv are 3-10 times faster on the GPU, about 1 ns per call, with
// similar accuracy; the choice per backend is reopened with lgamma's).
LAPLACE_HD_FORCEINLINE double erfc(double x) { return own::erfc(x); }
LAPLACE_HD_FORCEINLINE double erfcx(double x) { return own::erfcx(x); }
LAPLACE_HD_FORCEINLINE double Phi(double x) { return own::Phi(x); }
LAPLACE_HD_FORCEINLINE double log_Phi(double x) { return own::log_Phi(x); }
// d^k log Phi / dt^k; the first is the inverse Mills ratio phi(t)/Phi(t)
LAPLACE_HD_FORCEINLINE double log_Phi_d1(double t) { return own::log_Phi_d1(t); }
LAPLACE_HD_FORCEINLINE double log_Phi_d2(double t) { return own::log_Phi_d2(t); }
LAPLACE_HD_FORCEINLINE double log_Phi_d3(double t) { return own::log_Phi_d3(t); }
LAPLACE_HD_FORCEINLINE double log_Phi_d4(double t) { return own::log_Phi_d4(t); }
LAPLACE_HD_FORCEINLINE double log_Phi_d5(double t) { return own::log_Phi_d5(t); }
LAPLACE_HD_FORCEINLINE double Phi_inv(double p) { return own::Phi_inv(p); }
LAPLACE_HD_FORCEINLINE double Phi_inv_logp(double l) { return own::Phi_inv_logp(l); }

}  // namespace laplace::math
