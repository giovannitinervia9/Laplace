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
/**
 * \ingroup normal
 * \brief Scaled complementary error function, \f$\operatorname{erfcx}(x) = e^{x^2} \operatorname{erfc}(x)\f$.
 *
 * Computes the scaled complementary error function:
 * \f[
 * \operatorname{erfcx}(x) = e^{x^2} \operatorname{erfc}(x) = \frac{2}{\sqrt{\pi}} e^{x^2} \int_x^\infty e^{-u^2} \, \mathrm{d}u
 * \f]
 * where \f$u\f$ is the real integration variable.
 *
 * For \f$x \ge 0\f$, evaluated by Boost.Math rational approximations across \f$[0, 0.5)\f$, \f$[0.5, 1.5)\f$,
 * \f$[1.5, 2.5)\f$, \f$[2.5, 4.5)\f$, and \f$[4.5, 28.0)\f$, and an asymptotic series in \f$1/(2x^2)\f$
 * for \f$x \ge 28.0\f$. For \f$x < 0\f$, evaluated via \f$2 e^{x^2} - \operatorname{erfcx}(-x)\f$ with
 * \f$x^2\f$ carried as an exact split \f$h + l\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\operatorname{erfcx}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: finite down to \f$x = -26.628\f$.
 * - Returns \f$+\infty\f$ for \f$x \le -26.629\f$ and at \f$-\infty\f$.
 * - Returns \f$0.0\f$ at \f$+\infty\f$ (\f$\operatorname{erfcx}(+\infty) = 0.0\f$).
 * - If \f$x\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 1.78 ulp on CPU; test bound 3.0 ulp (3.0 ulp1).
 *
 * \see normal, erfc(), Phi(), log_Phi()
 */
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
  // exp(a^2) overflows from a = 26.64 on (and a^2 itself from 1.3e154): +inf,
  // not inf * 0 = NaN
  if (!(e < detail::pos_inf())) return detail::pos_inf();
  return 2.0 * (e + e * l) - erfcx_pos(a);
}

// erfc(x) = exp(-x^2) erfcx(x) for x >= 0, with x^2 carried exactly; 2 - erfc(-x) below
/**
 * \ingroup normal
 * \brief Complementary error function, \f$\operatorname{erfc}(x)\f$.
 *
 * Computes the complementary error function:
 * \f[
 * \operatorname{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty e^{-u^2} \, \mathrm{d}u
 * \f]
 * where \f$u\f$ is the real integration variable.
 *
 * For \f$x \ge 0\f$, evaluated as \f$e^{-x^2} \operatorname{erfcx}(x)\f$ with \f$x^2\f$ carried as an
 * exact split \f$h + l\f$. For \f$x < 0\f$, evaluated via \f$2 - \operatorname{erfc}(-x)\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\operatorname{erfc}(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - Returns \f$0.0\f$ for \f$x > 27.3\f$ (including \f$+\infty\f$).
 * - Returns \f$2.0\f$ for \f$x < -27.3\f$ (including \f$-\infty\f$).
 * - If \f$x\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 3.46 ulp on CPU; test bound 5.5 ulp (2.5 ulp1).
 *
 * \see normal, erfcx(), Phi()
 */
LAPLACE_HD double erfc(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  const double a = Kokkos::fabs(x);
  // erfc(27.3) is below half the smallest subnormal; beyond, the exact square
  // overflows (from 1.3e154) and would give NaN
  if (a > 27.3) return x >= 0.0 ? 0.0 : 2.0;
  double h, l;
  two_prod(a, a, h, l);
  const double r = erfcx_pos(a) * (math::exp(-h) * (1.0 - l));
  return x >= 0.0 ? r : 2.0 - r;
}

// Phi(x)
/**
 * \ingroup normal
 * \brief Standard normal cumulative distribution function, \f$\Phi(x)\f$.
 *
 * Computes the cumulative distribution function of the standard normal distribution:
 * \f[
 * \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-u^2/2} \, \mathrm{d}u = \frac{1}{2} \operatorname{erfc}\left(-\frac{x}{\sqrt{2}}\right)
 * \f]
 * where \f$u\f$ is the real integration variable.
 *
 * For \f$x < 0\f$, evaluated as \f$\frac{1}{2} \operatorname{erfcx}(-x/\sqrt{2}) \exp(-x^2/2)\f$ with
 * \f$x^2/2\f$ carried as an exact split \f$h + l\f$. For \f$x \ge 0\f$, evaluated as \f$1 - \Phi(-x)\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\Phi(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - At \f$x = 0.0\f$: returns \f$0.5\f$.
 * - Underflows to \f$0.0\f$ below \f$x = -38.5\f$, returning \f$0.0\f$ for \f$x < -40.0\f$ (including \f$-\infty\f$).
 * - Returns \f$1.0\f$ for \f$x > 40.0\f$ (including \f$+\infty\f$).
 * - If \f$x\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 3.86 ulp (0.86 ulp1) on CPU; test bound 6.0 ulp (1.5 ulp1).
 *
 * \see normal, log_Phi(), Phi_inv(), erfcx()
 */
LAPLACE_HD double Phi(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  // Phi underflows below -38.5 and rounds to 1 long before 40; beyond, the
  // square would overflow (from 1.3e154) and give NaN
  if (x < -40.0) return 0.0;
  if (x > 40.0) return 1.0;
  if (x < 0.0) return 0.5 * erfcx_pos(-x * sqrt_half) * exp_minus_half_square(x);
  return 1.0 - 0.5 * erfcx_pos(x * sqrt_half) * exp_minus_half_square(x);
}

// log Phi(x), without underflow down to x = -1.9e154
/**
 * \ingroup normal
 * \brief Natural logarithm of the standard normal cumulative distribution function, \f$\log \Phi(x)\f$.
 *
 * Computes the natural logarithm of the distribution function:
 * \f[
 * \log \Phi(x) = \log\left(\frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-u^2/2} \, \mathrm{d}u\right)
 * \f]
 * where \f$u\f$ is the real integration variable.
 *
 * For \f$x < 0\f$, evaluated as \f$-h + (\log(\frac{1}{2}\operatorname{erfcx}(-x/\sqrt{2})) - l)\f$ where
 * \f$h + l = x^2/2\f$ is exact. For \f$x \ge 0\f$, evaluated as \f$\operatorname{log1p}(-\Phi(-x))\f$.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\log \Phi(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - At \f$x = 0.0\f$: returns \f$-\log 2 \approx -0.693147\f$.
 * - Finite down to \f$x = -1.8961 \times 10^{154}\f$ (`-sqrt(2 DBL_MAX)`); returns \f$-\infty\f$ below it and at \f$-\infty\f$.
 * - Returns \f$-0.0\f$ for \f$x > 40.0\f$.
 * - If \f$x\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 4.08 ulp on CPU (CUDA: 3.79 ulp); test bound 6.0 ulp (3.0 ulp1).
 *
 * \see normal, Phi(), log_Phi_d1(), Phi_inv_logp()
 */
LAPLACE_HD double log_Phi(double x) {
  using namespace detail_normal;
  if (x != x) return x;
  // -x^2/2 passes -DBL_MAX below -sqrt(2 DBL_MAX) = -1.8961e154; log Phi(40)
  // is -4e-350, which rounds to -0
  if (x < -1.8961e154) return -detail::pos_inf();
  if (x > 40.0) return -0.0;
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

// BEGIN GENERATED: tools/reference/log_phi_cheb.py
// d^k log Phi / dt^k on [cheb_lo, cheb_hi): the piecewise Chebyshev expansions, one
// set of pieces per order, generated by tools/reference/log_phi_cheb.py (do not
// edit). On a piece without a zero of g_k the table holds the coefficients of g_k
// itself; on a piece whose midpoint is a zero z of g_k it holds those of
// g_k(t) / (t - z), and cheb_has_zero marks it so that the factor is restored from
// the pair of doubles in cheb_zero. The bins of cheb_bin find the piece of t.
namespace detail_normal {

constexpr double cheb_lo = -0x1.0000000000000p+0;  // the expansions cover [cheb_lo, cheb_hi)
constexpr double cheb_hi = 0x1.0000000000000p+2;
constexpr int cheb_bins = 40;  // bins of the piece lookup, of width (hi - lo) / cheb_bins
constexpr double cheb_bin_scale = 0x1.0000000000000p+3;
constexpr int cheb_size = 928;  // coefficients in all

// clang-format off
#define LAPLACE_LOG_PHI_CHEB_EDGE \
  -0x1.0000000000000p+0, 0x0.0p+0, 0x1.0000000000000p+0, \
  0x1.8000000000000p+0, 0x1.0000000000000p+1, 0x1.4000000000000p+1, \
  0x1.8000000000000p+1, 0x1.c000000000000p+1, 0x1.0000000000000p+2, \
  -0x1.0000000000000p+0, 0x0.0p+0, 0x1.0000000000000p+0, \
  0x1.8000000000000p+0, 0x1.0000000000000p+1, 0x1.4000000000000p+1, \
  0x1.8000000000000p+1, 0x1.c000000000000p+1, 0x1.0000000000000p+2, \
  -0x1.0000000000000p+0, 0x0.0p+0, 0x1.0000000000000p+0, \
  0x1.8000000000000p+0, 0x1.0000000000000p+1, 0x1.4000000000000p+1, \
  0x1.8000000000000p+1, 0x1.c000000000000p+1, 0x1.0000000000000p+2, \
  -0x1.0000000000000p+0, -0x1.fd92e421c6fb4p-3, 0x1.01368def1c826p-1, \
  0x1.809b46f78e413p+0, 0x1.0000000000000p+1, 0x1.4000000000000p+1, \
  0x1.8000000000000p+1, 0x1.c000000000000p+1, 0x1.0000000000000p+2, \
  -0x1.0000000000000p+0, -0x1.0f465a3465ba1p-1, 0x1.e1734b97348bep-2, \
  0x1.0000000000000p+0, 0x1.923ffdecc02b4p+0, 0x1.491ffef66015ap+1, \
  0x1.8000000000000p+1, 0x1.c000000000000p+1, 0x1.0000000000000p+2, \
  -0x1.0000000000000p+0, -0x1.7c0a12a849824p-2, 0x1.70d84c8dfb3bcp-3, \
  0x1.76712f9b225f0p-1, 0x1.0000000000000p+0, 0x1.8000000000000p+0, \
  0x1.bb3897cd912f8p+0, 0x1.0000000000000p+1, 0x1.1f71195cad7e3p+1, \
  0x1.4000000000000p+1, 0x1.8000000000000p+1, 0x1.9f71195cad7e3p+1, \
  0x1.c000000000000p+1, 0x1.0000000000000p+2
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double cheb_edge_dev[59] = {LAPLACE_LOG_PHI_CHEB_EDGE};
#endif
inline constexpr double cheb_edge_host[59] = {LAPLACE_LOG_PHI_CHEB_EDGE};
LAPLACE_HD_FORCEINLINE double cheb_edge(int i) {
  KOKKOS_IF_ON_DEVICE((return cheb_edge_dev[i];))
  KOKKOS_IF_ON_HOST((return cheb_edge_host[i];))
}

// clang-format off
#define LAPLACE_LOG_PHI_CHEB_COEF \
  0x1.26ba557bb4303p+0, -0x1.74e5b2f29125ap-2, 0x1.4ec38de39af0fp-7, \
  0x1.1761689b70c27p-11, 0x1.b80cb2060ae9cp-17, -0x1.90485eb73664dp-21, \
  -0x1.d3d1b39e3df92p-24, -0x1.97f6bca64c238p-28, -0x1.47ddff7203740p-35, \
  0x1.8d19dab4293f0p-36, 0x1.2724b19e00828p-39, 0x1.695bdf25cf5bbp-44, \
  -0x1.609ccee3fa989p-49, -0x1.4a0a2ebe5a224p-51, -0x1.64cb6d75f9ad1p-55, \
  -0x1.cb847efcc83b3p-61, 0x1.0d4e4adf34d29p-1, -0x1.05b5db397ad9ap-2, \
  0x1.131a6bd6743f9p-6, 0x1.cba1f161b33a8p-12, -0x1.3307ee5f2cd38p-15, \
  -0x1.143613b9b5d6ep-18, -0x1.b847bcf8d5883p-25, 0x1.613429d10343bp-26, \
  0x1.cad2e7f901463p-30, -0x1.b87107db15bd7p-37, -0x1.88120eaa538e0p-37, \
  -0x1.85c935819d684p-41, 0x1.7dd4f95a8fb53p-46, 0x1.a1825d0bf87b5p-48, \
  0x1.3ac3e9912ed25p-52, -0x1.3a5d7a44fee4dp-56, 0x1.ab72e101c2ebep-3, \
  -0x1.309a1afb11165p-4, 0x1.25e8a646f91a1p-8, -0x1.4a88b8001b0ccp-15, \
  -0x1.52c5ed4f0b94dp-18, 0x1.2bb572dcd4683p-27, 0x1.2b5ebbea99ba1p-27, \
  0x1.0fcd93ba3fdfap-33, -0x1.e70b1a722717fp-37, -0x1.0f900ce0307b5p-41, \
  0x1.2d53a42c2c434p-46, 0x1.607e49ce59c46p-50, -0x1.1331e403d8a01p-57, \
  -0x1.66aa1e4b1d539p-59, -0x1.40ad60400f318p-65, 0x1.23bd9b5b3094bp-68, \
  0x1.7ec68f87add95p-4, -0x1.554f73d9dbe3bp-5, 0x1.d3fef68921203p-9, \
  -0x1.c1f83760bf99bp-14, -0x1.78776ca0b088ep-19, 0x1.adb3b3a9f6f84p-23, \
  0x1.48903f1b11952p-28, -0x1.a9fbfa1eb2e57p-32, -0x1.6d45d7ee72fb4p-37, \
  0x1.9db79dcd9d34bp-41, 0x1.ac6240b076fd6p-46, -0x1.8444fdfeaa56ep-50, \
  -0x1.effedd1bcb212p-55, 0x1.5f5bf31f8f2adp-59, 0x1.1737de251878ap-63, \
  -0x1.3231785a13495p-68, 0x1.18df7fef35421p-5, -0x1.322c186d49745p-6, \
  0x1.1a7c8d56cfd70p-9, -0x1.ee99e8c5396f5p-14, 0x1.74c0d399e3812p-20, \
  0x1.7746aeacf43d2p-23, -0x1.ac192d1bd01cep-28, -0x1.145cc6c666f60p-32, \
  0x1.1c0ac1a637b7cp-36, 0x1.5d63fe0710393p-42, -0x1.43ac4c78d0ff3p-45, \
  -0x1.0ee074b3ae892p-52, 0x1.5277f67d0d934p-54, -0x1.4c17608e300fbp-62, \
  -0x1.4b59a58011ea8p-63, 0x1.279168626c706p-69, 0x1.4a2cee2cfdbcfp-7, \
  -0x1.ab7ac87778b14p-8, 0x1.f68eb431f8523p-11, -0x1.43aef6d2c1c30p-14, \
  0x1.a401f0eba3fefp-19, 0x1.b48855633e0aep-34, -0x1.a8d4caf656110p-28, \
  0x1.b5f945823ccf6p-33, 0x1.cf20f29f3a8b7p-38, -0x1.3f1f87d22183dp-41, \
  0x1.c153370a917b2p-50, 0x1.2ae80c90dcfe1p-50, -0x1.fa377628d4b05p-56, \
  -0x1.75cd373778bcdp-60, 0x1.77488320b49fbp-64, 0x1.051ebd543db51p-71, \
  0x1.32cae562766f6p-9, -0x1.ca95355764685p-10, 0x1.47c152a9bbad1p-12, \
  -0x1.15399a14844bfp-15, 0x1.207f7e7c53b74p-19, -0x1.3f06965629672p-24, \
  -0x1.1ca2d2412eaa7p-32, 0x1.582cdb8de4f6ep-33, -0x1.a5b51408bbf43p-38, \
  -0x1.153a4751de233p-44, 0x1.d4a2567a5dd76p-47, -0x1.887ac61e44ac8p-52, \
  -0x1.8aa7f3a0d49fcp-57, 0x1.0e0f765e427edp-60, -0x1.d8962f3d5a7f8p-67, \
  -0x1.3db75104db5fap-70, 0x1.bfe5626493350p-12, -0x1.792c92225843fp-12, \
  0x1.3c1a8b132c92ep-14, -0x1.4803540db3b83p-17, 0x1.c67c414d13130p-21, \
  -0x1.a37fb3e1260ecp-25, 0x1.c0d63dfa786d9p-30, -0x1.8f38d3a9e468fp-39, \
  -0x1.8cd76e59b38e1p-39, 0x1.37c27e76b622bp-43, -0x1.8fe27078320aap-50, \
  -0x1.79a4f2ff94145p-53, 0x1.3f76b6c82d371p-57, -0x1.ba0e609e4a4dfp-64, \
  -0x1.49d907476dcbap-67, 0x1.13825f9233f06p-71, -0x1.73439dc951a5ap-1, \
  0x1.4f9cd635daa60p-4, 0x1.a215293f8002dp-8, 0x1.b290a47f6a1d6p-13, \
  -0x1.f9e7535241b1cp-17, -0x1.5f0361a833191p-19, -0x1.63373b4f6cf27p-23, \
  -0x1.30d78c24f1b1ep-30, 0x1.c0a9c215ac9f7p-31, 0x1.706734d11c21cp-34, \
  0x1.ecac0afe38996p-39, -0x1.0d5268c902a9fp-43, -0x1.0c91f57121298p-45, \
  -0x1.37336781addf7p-49, -0x1.a6bdd9dfb6ae9p-55, 0x1.fd308b19300efp-58, \
  0x1.01cba5a8a11e7p-60, -0x1.04627d2c77b8fp-1, 0x1.11e6c5e93235cp-3, \
  0x1.535e0d0320a6bp-8, -0x1.33a5ed4209d17p-11, -0x1.56da0189714d8p-14, \
  -0x1.3bfdc5b9fbeb4p-20, 0x1.34cb8f58ff8aap-21, 0x1.c7010014875dbp-25, \
  -0x1.0854f78d426e4p-31, -0x1.e8f3f23cf43fbp-32, -0x1.09563220633b4p-35, \
  0x1.22a017f431d4cp-40, 0x1.5211545c7e33fp-42, 0x1.10173c1183356p-46, \
  -0x1.28973d3bb040cp-50, -0x1.aa1826c2e1507p-53, -0x1.df9a9b014a41bp-58, \
  -0x1.3116024fd1cfap-2, 0x1.253fb3570e304p-4, -0x1.ef9d5302e5181p-11, \
  -0x1.51e5dfd5d3a85p-13, 0x1.7e07ea1bd86efp-22, 0x1.c01af26fd9064p-22, \
  0x1.d946a1f3bb30cp-28, -0x1.e64ee01b21bedp-31, -0x1.3090491a4b47ap-35, \
  0x1.7874ae0ab235ap-40, 0x1.e38ac3d6862dcp-44, -0x1.9ef96428f3565p-51, \
  -0x1.22e1a5354046ep-52, -0x1.17471197331c5p-58, 0x1.1126cf8f1a163p-61, \
  0x1.50a2a0da2f04fp-66, -0x1.6bc8598dd4c4dp-71, -0x1.57f0508020b76p-3, \
  0x1.d34335add9bd1p-5, -0x1.506e532269da3p-9, -0x1.7781b68ec6419p-14, \
  0x1.0bd66625d9108p-17, 0x1.eb6c23d48ea43p-23, -0x1.73d448c295536p-26, \
  -0x1.6c3ad40bbb7edp-31, 0x1.d064308e6ad19p-35, 0x1.0b03e2b77c685p-39, \
  -0x1.0a60f8e609afdp-43, -0x1.730b6d9bec198p-48, 0x1.1ceb7216d7de8p-52, \
  0x1.e77071d87eb66p-57, -0x1.1e86e598e89dfp-61, -0x1.3152e86c1cb37p-65, \
  0x1.0ef67712ed90fp-70, -0x1.37f43dd9c2727p-4, 0x1.1ad91d8fe5dfcp-5, \
  -0x1.72095b1e3f89cp-9, 0x1.7240e45822ff2p-15, 0x1.d426eb57137cfp-18, \
  -0x1.3ff7a0e041029p-22, -0x1.e2de023b9ef1ap-27, 0x1.1b40f49b13107p-30, \
  0x1.88b33f2a79cd3p-36, -0x1.939a16494e9c9p-39, -0x1.74fa75e1cc8e4p-46, \
  0x1.fa929b6d44e85p-48, -0x1.0baad579236d6p-55, -0x1.21564e4f7492bp-56, \
  0x1.14147d31cfb12p-62, 0x1.304501363507cp-65, -0x1.03d4aa75f83c3p-70, \
  -0x1.baa6f3f05fc17p-6, 0x1.f9d43bbc40d0ap-7, -0x1.e5856f1ce206ap-10, \
  0x1.a2c3c5243f3a0p-14, 0x1.031f409de76e4p-26, -0x1.3e2bc764c4e92p-22, \
  0x1.7dd3e68ccb791p-27, 0x1.cf434fee8e6c5p-32, -0x1.66364529bc603p-35, \
  0x1.12ea7a9f0754cp-43, 0x1.9a674552a830dp-44, -0x1.7a61f1e4e60c3p-49, \
  -0x1.2f97e90f59029p-53, 0x1.47a6b9b97807ep-57, 0x1.ed3cdb916cccap-65, \
  -0x1.7172064c07c3ep-66, 0x1.d19c49bc8c980p-72, -0x1.e4ab757d0f7ffp-8, \
  0x1.4c4319a095171p-8, -0x1.a164025ab179ep-11, 0x1.2071bdb65a7e1p-14, \
  -0x1.8d9b3beb07f19p-19, -0x1.b818bf2725249p-27, 0x1.2d0000abcf4d2p-27, \
  -0x1.a49078abe49c1p-32, -0x1.39fb82c857355p-38, 0x1.249b5cd7581d6p-40, \
  -0x1.0cf9461ea6d95p-45, -0x1.2864d48a24e55p-50, 0x1.b6442c50ba7eap-54, \
  -0x1.9b774615b60bap-60, -0x1.29e810e31ee12p-63, 0x1.06119ffc91f40p-67, \
  -0x1.869dd226ef577p-76, -0x1.982e6f3f2a75ep-10, 0x1.4339bb80efea7p-10, \
  -0x1.f01dd1cd231f8p-13, 0x1.c7cc1b70d5e3ep-16, -0x1.0634ee2562cc0p-19, \
  0x1.4fda23c2d0e42p-24, -0x1.476e2ac0af25fp-33, -0x1.8d157212dc113p-33, \
  0x1.5e38e73f8b5ddp-37, -0x1.f01dc94418ee9p-44, -0x1.03ce0c03126f9p-46, \
  0x1.dea1a912cf555p-51, -0x1.65279ae50cc05p-57, -0x1.20d232e8fa97ap-60, \
  0x1.0209cdc7cf844p-64, -0x1.a26453d42b9cap-71, -0x1.01af4504b46eap-74, \
  0x1.521af72906de5p-3, 0x1.a012fcb9243bbp-5, 0x1.3f1079961c281p-9, \
  -0x1.0116432de38fdp-12, -0x1.b700725cdb7c6p-15, -0x1.08a66130adbd0p-18, \
  -0x1.e1c254dce8764p-26, 0x1.c30b4ae3f8d18p-26, 0x1.9db7031df4086p-29, \
  0x1.30c467261901ep-33, -0x1.7a309a973b370p-38, -0x1.938fdc652efb6p-40, \
  -0x1.f7d2a0b5e3575p-44, -0x1.69d876fa7a400p-49, 0x1.e0e77cd733bc5p-52, \
  0x1.01b4f528b316ap-54, 0x1.c4fd37cb572afp-59, -0x1.6b07fee07d3cdp-66, \
  0x1.0e48ef61fac86p-2, 0x1.48c42725c0e7cp-5, -0x1.ceeb439bb6aa2p-8, \
  -0x1.533cbbabf7ddbp-10, -0x1.725fb8a7effeap-16, 0x1.cea2eebcb7ecdp-17, \
  0x1.89d7e808ae770p-20, -0x1.1cd0918ec6326p-26, -0x1.123e0251febe9p-26, \
  -0x1.47b9a0183c41ep-30, 0x1.966be0954a834p-35, 0x1.f90f481fe416ap-37, \
  0x1.b3efe9617faf5p-41, -0x1.055b356c9ba8dp-44, -0x1.8d760ebd61dcbp-47, \
  -0x1.d6dfd86170239p-52, 0x1.004acca8af5acp-54, 0x1.175853fb43c4ep-57, \
  0x1.234709004b184p-2, -0x1.ef3b0fdbf6e4ap-7, -0x1.f8aa56c31804ep-9, \
  0x1.890c9bb8cdb54p-17, 0x1.173c7ed2bbcafp-16, 0x1.6096339ea8c96p-22, \
  -0x1.a8b16657b1e7bp-25, -0x1.2f62ec11cd956p-29, 0x1.a73b80173e935p-34, \
  0x1.2d5d087db2433p-37, -0x1.1f0ed327a4033p-44, -0x1.b363d0c332c73p-46, \
  -0x1.c35e4b7cb7d3cp-52, 0x1.dd4e195b46311p-55, 0x1.3a988c7d9d211p-59, \
  -0x1.6ba3be4eeb901p-64, -0x1.ffd49ddde73cap-68, 0x1.24d9f749a6290p-75, \
  0x1.d11256e820366p-3, -0x1.4e58d245733f7p-5, -0x1.186f62dcc355ap-9, \
  0x1.0ac06e7b4d65ep-12, 0x1.31e60e515b8c9p-17, -0x1.15f7aa8baa99dp-20, \
  -0x1.3d88137d9a0d4p-25, 0x1.cf180c8ac96e5p-29, 0x1.2b660caa01b33p-33, \
  -0x1.4c2403a163404p-37, -0x1.fca4c8d4844cap-42, 0x1.aa66fc51b6f70p-46, \
  0x1.8aede1e056621p-50, -0x1.f45da119ad69fp-55, -0x1.1d7a9f9092198p-58, \
  0x1.0e70b1e9aa865p-63, 0x1.8634a9919d681p-67, -0x1.0b8a528615427p-72, \
  0x1.1bebb22befbd8p-3, -0x1.70369d954ae36p-5, 0x1.12949c09ddbdfp-10, \
  0x1.d2bd88f4a65cbp-13, -0x1.8e079c1e40b14p-17, -0x1.6962626d2043ep-21, \
  0x1.edecfa1091f24p-25, 0x1.883e7f2de2b01p-30, -0x1.c4b1fecf6a814p-33, \
  -0x1.d2fff25c746dep-40, 0x1.5b5a430deee2dp-41, -0x1.8dbe0469781a5p-49, \
  -0x1.d50f9a60f9dbcp-50, 0x1.e11de61e84e13p-56, 0x1.1ca4e02392a3cp-58, \
  -0x1.02fa7c534a5cfp-63, -0x1.37a1fe7e27190p-67, 0x1.b45c455bbe75ep-72, \
  0x1.01cc357da945dp-4, -0x1.e5812eeaf3647p-6, 0x1.3885e7e2375d7p-9, \
  0x1.100c7ba88b214p-20, -0x1.8cebf8f80e0bcp-17, 0x1.1cf9b6b32ed45p-21, \
  0x1.95808bd02f444p-26, -0x1.65363669c67a2p-29, 0x1.2d2efb952bc8bp-37, \
  0x1.000ebff5e60f2p-37, -0x1.0319cbb78ed6ap-42, -0x1.c72d770c3d6adp-47, \
  0x1.098a95cf51c6bp-50, 0x1.b3345640c88dfp-58, -0x1.59c22ebf7f556p-59, \
  0x1.cf8b10c4ad755p-65, 0x1.316e4f8fe4841p-68, -0x1.089c7bef9153fp-72, \
  0x1.59c74ffffcd5bp-6, -0x1.a47bb270d2372p-7, 0x1.b086cbecf7d41p-10, \
  -0x1.8bd80b105ea03p-14, -0x1.1e85247f4893fp-21, 0x1.c330daa9516d9p-22, \
  -0x1.6eb5a0da2ba38p-26, -0x1.3c99619618532p-32, 0x1.48c8bc3c650ecp-34, \
  -0x1.4eef66e08ee86p-39, -0x1.9830d7784946fp-44, 0x1.4830c5c1a7474p-47, \
  -0x1.4c6674ad1744bp-53, -0x1.04b6f5c92f69fp-56, 0x1.ea74448ca4c4ap-61, \
  -0x1.7ce04e8c9dd33p-69, -0x1.d98ed9d9c9d77p-70, 0x1.37b0734a3087ap-74, \
  0x1.58b1741986ee3p-8, -0x1.f84fb12c9ca53p-9, 0x1.577b8989703c1p-11, \
  -0x1.063bebef30b5cp-14, 0x1.a274f4cfd12a4p-19, -0x1.bf72737a701abp-28, \
  -0x1.5bb7e3b3f2e81p-27, 0x1.5d966534b4f22p-31, -0x1.147f8dc964017p-37, \
  -0x1.450415acd7508p-40, 0x1.4899b675025e2p-44, -0x1.0a1aa4011410dp-50, \
  -0x1.d5b71fb0b1f67p-54, 0x1.c3102ab57f6bdp-58, -0x1.8633d86b003b9p-64, \
  -0x1.01fad057774cap-67, 0x1.051b15f4539fep-71, -0x1.2e2d4b0b77f17p-77, \
  0x1.8b459780d1612p-4, 0x1.10104c23747f4p-6, -0x1.02bd80f1d6855p-10, \
  -0x1.2f027f295e04bp-12, -0x1.94b1af953c057p-16, -0x1.4a72b0ca138cfp-21, \
  0x1.45d124d018253p-24, 0x1.5dc845883a294p-27, 0x1.32442707a6934p-31, \
  0x1.fd67dacec0812p-39, -0x1.1e4aed4e1132cp-39, -0x1.a0c7b10122a37p-43, \
  -0x1.0938c799e435dp-47, 0x1.c23787f3a1135p-54, 0x1.404e4cd8e12e7p-55, \
  0x1.62ecb521a4f67p-59, 0x1.3f4160d07f974p-64, -0x1.c0d282662f74bp-69, \
  -0x1.13d5f3d2516bdp-71, 0x1.b3a79cb74916dp-4, -0x1.75d6373701206p-7, \
  -0x1.ab367ab67ab2dp-8, -0x1.1fcc2df0cb4eep-11, 0x1.9b52dd4f6b8b4p-17, \
  0x1.81285e029c26fp-18, 0x1.b9afe058dd2dfp-22, -0x1.e9d61a8b359e5p-29, \
  -0x1.82bc455ee847ep-29, -0x1.bd900fed9eb1bp-33, -0x1.9baf2f61ee843p-42, \
  0x1.219341bd54068p-40, 0x1.62e4d9ac07c12p-44, 0x1.d4e5afe59109bp-51, \
  -0x1.689229e7b467dp-52, -0x1.e43e1ea895d24p-56, -0x1.10b42fe431860p-61, \
  0x1.89335f2597269p-64, 0x1.27e6ec41e831dp-67, -0x1.c2b2dc2d9624dp-3, \
  -0x1.1d2b26512b086p-5, 0x1.6eae4f7a2d480p-7, 0x1.b900bc7644a46p-10, \
  -0x1.7b5b2048137d7p-14, -0x1.23bc17a0ab229p-15, -0x1.2410a651258ecp-20, \
  0x1.b57871d8b8d4cp-22, 0x1.64b4bb0180315p-25, -0x1.35881de15a539p-29, \
  -0x1.5b461264fc981p-31, -0x1.07ea8d433bf10p-36, 0x1.a0849c24ed225p-38, \
  0x1.28165243ca21fp-41, -0x1.01401bf10144ap-45, -0x1.fa0cd5b2a679ap-48, \
  -0x1.463d19f55f6d2p-53, 0x1.127db75b8b0f3p-54, 0x1.67af28e1176a2p-58, \
  -0x1.4892dfde7eef7p-3, -0x1.13e20c5a60f88p-5, 0x1.8a4bfd5a74b2dp-8, \
  0x1.2a395c8268c0dp-12, -0x1.54245c0caf248p-15, -0x1.cc3312422fe51p-20, \
  0x1.89b4d4652e272p-23, 0x1.1ece7f03e3d39p-27, -0x1.68583c122d84ap-31, \
  -0x1.2df2bd5e6e466p-35, 0x1.18b4cc7338defp-39, 0x1.16efdfcfe663ap-43, \
  -0x1.82d0f4f2f8ebcp-48, -0x1.d276c28265ea4p-52, 0x1.e00308c67a2e9p-57, \
  0x1.693c2b854b703p-60, -0x1.0c5bd5d40d34dp-65, -0x1.074e0e7cc493dp-68, \
  0x1.082c5ff4d08a5p-74, -0x1.6ac56e8b8cc33p-3, 0x1.0c68030f37b82p-6, \
  0x1.5c4bc26f880f0p-8, -0x1.8b263ea981741p-12, -0x1.c26447f4b67f0p-16, \
  0x1.70aeba5f9e9dap-19, 0x1.56b313b1d5d8fp-24, -0x1.c3012cced8172p-27, \
  -0x1.07372ca10248dp-33, 0x1.b0d200926a1c4p-35, -0x1.0e689a019608fp-42, \
  -0x1.5ed33f007f3d1p-43, 0x1.850483764493dp-49, 0x1.f0e990784f73cp-52, \
  -0x1.e3c7828763265p-57, -0x1.36f7c5f12ac63p-60, 0x1.ce2694c847e59p-65, \
  0x1.547119f8a5a7cp-69, -0x1.7b74e93277af5p-73, -0x1.e569600b04348p-4, \
  0x1.356e7079e0125p-5, 0x1.7cedfef2ffb36p-15, -0x1.8bbbb42ba5926p-12, \
  0x1.61c944692eb4ep-16, 0x1.3044cc687962ap-20, -0x1.376ffb65ea3fap-23, \
  0x1.231c62af7b48fp-31, 0x1.1f7436a36b33cp-31, -0x1.42531cb60ff73p-36, \
  -0x1.38c2a26f3a9adp-40, 0x1.8d21ef62950e2p-44, 0x1.64fb497cf4d09p-51, \
  -0x1.2df154659bf02p-52, 0x1.b061b428ee9a9p-58, 0x1.312903e6f52b1p-61, \
  -0x1.1855c7ba01b37p-65, -0x1.152ea3bd63d7ep-71, 0x1.a0f7e9116b84cp-74, \
  -0x1.adb1272a7afe6p-5, 0x1.b03ae3a4192fbp-6, -0x1.26ae973518e89p-9, \
  -0x1.2fa1237a91912p-16, 0x1.19b88b9707c91p-16, -0x1.11bfefb48fd2cp-20, \
  -0x1.17f44b2c6db12p-26, 0x1.4848ef10e7db6p-28, -0x1.770ae48c34316p-33, \
  -0x1.ff34adf4cd71bp-38, 0x1.c26f306c9418ap-41, -0x1.ef413ce3b1ed3p-47, \
  -0x1.a7befb63d2c9fp-50, 0x1.ac390ff87ce86p-54, -0x1.5abfce5bdd152p-62, \
  -0x1.d958052687593p-63, 0x1.4a4f68fedc1bfp-67, 0x1.b6a59a13f1e4ep-74, \
  -0x1.b822ffeee86c9p-76, -0x1.087320667f235p-6, 0x1.5abc5f3a19547p-7, \
  -0x1.8968fa061a174p-10, 0x1.a06ad8548c2e3p-14, -0x1.e303ea20d5a5ep-23, \
  -0x1.050e3da27e0a1p-21, 0x1.312c98e0d9edep-25, -0x1.114b6e1f7028ep-31, \
  -0x1.6dff5a88cc045p-34, 0x1.9a0fd4f9ec4e9p-38, -0x1.6b08996727178p-44, \
  -0x1.609e30ad4e3f5p-47, 0x1.6e040d2a3fcb2p-51, -0x1.5363a32319e74p-57, \
  -0x1.e45624def1c98p-61, 0x1.04dd1d4326725p-64, -0x1.3fbc75e433c8ap-70, \
  -0x1.efc589696c77cp-75, 0x1.53a9d7fba7e6ep-78, 0x1.85467680b2bd6p-5, \
  -0x1.4a820cd739510p-9, -0x1.6716eb6e60f71p-10, -0x1.d5e6b2406b270p-14, \
  -0x1.e19760b91c331p-19, 0x1.7b874e03e0b1dp-24, 0x1.0b299b1d5b084p-26, \
  0x1.b74022b5db22ep-31, 0x1.22796756a2576p-36, -0x1.454c7a1365cfcp-41, \
  -0x1.1f7d601504e6fp-44, -0x1.785605861e390p-49, -0x1.67d56707f559dp-55, \
  0x1.2b5d8d27ed49bp-59, 0x1.8d719e49a81a5p-63, 0x1.ba8aeeee9337bp-68, \
  0x1.28f1d4b04f5a4p-74, -0x1.8aaa4b1a043c6p-78, -0x1.aa22c8ad9d26ap-82, \
  -0x1.9e30d6e93528dp-87, -0x1.2b693802821cbp-3, -0x1.f6d8c10f0e157p-5, \
  -0x1.2ba47a5499d8ap-10, 0x1.4a725caa6f6cfp-10, 0x1.a943fd2695881p-13, \
  0x1.ff6d4ba3ba11fp-18, -0x1.e675e233b0d6ep-20, -0x1.53c92cdb1491cp-22, \
  -0x1.1dd6cc4c79fb2p-26, 0x1.96f5900104aacp-30, 0x1.6f69181fea43fp-32, \
  0x1.88820b10a46fep-36, -0x1.87bcb8276d9b2p-41, -0x1.332ee8955ccb2p-42, \
  -0x1.9396927a4f399p-46, -0x1.3786284b77bb8p-57, 0x1.a5e27ec365113p-53, \
  0x1.54a8b69ee94c2p-56, 0x1.d95e19cfb4ad0p-62, -0x1.e6357c0fa56f2p-64, \
  -0x1.5cfb392e3c15cp-3, -0x1.08f26968cfc42p-4, 0x1.92046aaa1d7a3p-9, \
  0x1.0b2cd3af3e2cdp-10, 0x1.3d221b9e4af60p-15, -0x1.34cb80f0a6415p-18, \
  -0x1.eec61e36ba446p-22, -0x1.4f8dd26a27eabp-30, 0x1.0fbb37330d455p-29, \
  0x1.b6968db69798cp-34, -0x1.93781b270ed53p-39, -0x1.12b40c79ee78fp-41, \
  -0x1.ad2e0ddf85732p-47, 0x1.36c07a4d7db7fp-50, 0x1.82ebf8d686565p-54, \
  0x1.2a489c0665d33p-62, -0x1.19fa17211bb3ap-62, -0x1.8ffee2729b22fp-67, \
  0x1.27ad917604a97p-72, 0x1.7bfcd58fb575fp-75, -0x1.dca44ae520194p-3, \
  0x1.d63a09e31ae70p-7, 0x1.11ddebfaafc69p-6, 0x1.c7769eddd60e9p-12, \
  -0x1.6a6edb99c2ab4p-13, -0x1.4f4d4cef2a89dp-17, 0x1.f4393d8730af4p-21, \
  0x1.9789ba5b7fb97p-24, -0x1.474bf0e9e1ebap-29, -0x1.3d8a8bd9b5e25p-31, \
  -0x1.845b61a3a685cp-38, 0x1.650c6541da787p-39, 0x1.b5f0800cbe5bfp-44, \
  -0x1.22d56391bed5fp-47, -0x1.66ccf08b85d81p-51, 0x1.17221b089c4e0p-56, \
  0x1.9edcd45d50ea8p-59, 0x1.4d48dbe7fbfa2p-66, -0x1.759975d585a40p-67, \
  -0x1.83e774730ac6ep-72, 0x1.8bca9c8421c4ep-2, -0x1.e3f383d2b35b4p-5, \
  -0x1.4aae04c4e8252p-5, 0x1.d24a9723271c4p-9, 0x1.1bd645f2cb2e8p-10, \
  -0x1.be79f2c16abf7p-14, -0x1.5cb1420a3373cp-16, 0x1.32b52ebc08d78p-19, \
  0x1.5a2354a9d2e9bp-22, -0x1.59b483f549488p-25, -0x1.267cb5ad3d091p-28, \
  0x1.5515dde0cf35fp-31, 0x1.bb19763cce2e3p-35, -0x1.31536e4bb9668p-37, \
  -0x1.2b118bafd3198p-41, 0x1.fad7978f65726p-44, 0x1.69f9cb7fec29cp-48, \
  -0x1.8ba4ea639dfb9p-50, -0x1.7ffcc9a661821p-55, 0x1.255ab7ffe0ed0p-56, \
  0x1.31070d6a8aa8bp-3, -0x1.cc8f3025ec404p-9, -0x1.97fbe1698f153p-8, \
  0x1.ddfe59ee66eddp-12, 0x1.419b10f827d7cp-16, -0x1.ac23d625d6c71p-19, \
  0x1.43b5ee63c90cbp-25, 0x1.63006ea5549c7p-27, -0x1.ebf6567d5fce7p-32, \
  -0x1.25abc69a39352p-36, 0x1.db2d28f17b94dp-40, -0x1.9c48358a83370p-48, \
  -0x1.16fe3a176b038p-48, 0x1.1383c79b5adcbp-53, 0x1.85196476c383bp-58, \
  -0x1.d4f05f0a0b82ep-62, -0x1.a2d67916d83d4p-72, 0x1.ed12c81fb088bp-71, \
  -0x1.908aaf2bed620p-76, -0x1.460f61e65152fp-80, 0x1.af41eba0b9e4dp-4, \
  -0x1.22495ec891e9dp-5, -0x1.f1f006be95c6ap-12, 0x1.194e1b21bfaebp-11, \
  -0x1.53f28c35db67cp-15, -0x1.a9c1d5206974fp-21, 0x1.1eafb5ec2fdb9p-22, \
  -0x1.74d917713ad45p-27, -0x1.2036853608bb3p-31, 0x1.18e68d7cae80ep-34, \
  -0x1.51c6b8aa56806p-40, -0x1.3de18e5c1cf8ap-43, 0x1.5b109919e95e7p-47, \
  -0x1.251d13ee12e5dp-55, -0x1.bb87bff83cb5ap-56, 0x1.49581449d1956p-60, \
  0x1.d626df117005cp-67, -0x1.eea96a150d07bp-69, 0x1.05bad70fc0a38p-73, \
  0x1.f7d75ee1c9a43p-79, 0x1.646a73d2f9dacp-5, -0x1.898008f2e5cc0p-6, \
  0x1.35c2931c10c8dp-9, -0x1.70eeccbb4bdffp-18, -0x1.46c791ac2cec1p-16, \
  0x1.c85475962717dp-20, -0x1.d712843d7dd6bp-26, -0x1.6e6fbb1fcd00cp-28, \
  0x1.cc5f1e619c33ap-32, -0x1.c1825c03f1abap-38, -0x1.e5626f1b4973cp-41, \
  0x1.1218ef3fcc726p-44, -0x1.11d859fbb94eap-50, -0x1.a86a7f8d97fd9p-54, \
  0x1.e89a90cbbd426p-58, -0x1.3e3e9508da734p-63, -0x1.080c2455a7fa1p-67, \
  0x1.7de0db59556bfp-71, -0x1.5666ab4cb4e50p-76, -0x1.f1bd0e3bbe129p-82, \
  -0x1.8f011214f1d42p-4, -0x1.0dc0c75ef7b24p-5, -0x1.c431b32624e27p-10, \
  0x1.d85b3dc673aa0p-14, 0x1.a446a05f0520fp-16, 0x1.ebe063633aa9ep-20, \
  0x1.65e9d3c719c10p-25, -0x1.49b5a3739fe49p-28, -0x1.43db39cd7e839p-31, \
  -0x1.090989e39c685p-35, -0x1.2fa00c7f8f82dp-42, 0x1.768e2ab426d44p-44, \
  0x1.0632730c35d84p-47, 0x1.48dac1f5f8b3bp-52, -0x1.330c163581d61p-60, \
  -0x1.0e324be6ef57bp-60, -0x1.0b5cc21b8105cp-3, -0x1.14ffcd9732f70p-4, \
  -0x1.4442f0484d214p-9, 0x1.56d392642234bp-11, 0x1.758c3bc8013d3p-14, \
  0x1.d95662ea69940p-19, -0x1.cdf1cb1f138b1p-23, -0x1.269a64ebeb8d1p-25, \
  -0x1.bc0ce8aa74417p-30, 0x1.75a934767709bp-36, 0x1.053ddcae004e8p-37, \
  0x1.ddf5b01803970p-42, 0x1.fb88fbbc6a0f0p-49, -0x1.3b24810cf3450p-50, \
  -0x1.6c2382f32bd43p-54, -0x1.fd737defc5989p-60, -0x1.f082fe041ee74p-3, \
  -0x1.eee0d0aa48415p-6, 0x1.d393ba74a734cp-7, 0x1.01c8676ec7492p-9, \
  0x1.b608897a4e581p-17, -0x1.037eec4bf83aap-16, -0x1.435ec0225d77ap-20, \
  0x1.4dfb8be4a6436p-27, 0x1.dcd774a0aac92p-28, 0x1.a458541b344f8p-32, \
  -0x1.0ef5acf7f83acp-37, -0x1.1be9d7d99a032p-39, -0x1.79c7707e3c583p-44, \
  0x1.97d20db74cac5p-49, 0x1.036278bdd00aep-51, 0x1.06f665eb7d552p-56, \
  -0x1.b96581ed78bbfp-3, 0x1.84a5ab2ae8f89p-5, 0x1.b197eaca3406fp-8, \
  0x1.53c317305db94p-15, -0x1.31901cdcebbd0p-16, -0x1.589dc82d685b6p-21, \
  0x1.0678cb467c5b7p-26, 0x1.83e77322f929ap-30, 0x1.f53968d34abb6p-37, \
  -0x1.94f6d03f7347ap-40, -0x1.c41f9a6c05e44p-45, 0x1.f0c1962f36f45p-52, \
  0x1.1a26bdb64882bp-54, 0x1.096091a60f0fep-60, -0x1.6a5b9f2fdf408p-65, \
  -0x1.e6a8f98d812bep-70, 0x1.9ab61f7f07077p-1, 0x1.5197666cae5c7p-4, \
  -0x1.86fe5a7412b49p-6, -0x1.f1165dada8707p-10, 0x1.0fad09ee4ab95p-13, \
  0x1.26ce187f87292p-16, -0x1.7ae879b487e25p-23, -0x1.a136f53db665dp-24, \
  -0x1.265266942949ap-29, 0x1.8b4cdd730d81fp-32, 0x1.4f1ace9aa7d65p-36, \
  -0x1.dc6f19f8e74e9p-41, -0x1.a1c1b278751cfp-44, 0x1.0e2ff11ce1c36p-51, \
  0x1.7657267eb917ep-52, 0x1.fd8dd2e17808ap-58, 0x1.42b405e2fa4a3p-2, \
  0x1.4735e30c9a345p-4, -0x1.3cfe62d99a5e0p-8, -0x1.72461d21806edp-12, \
  0x1.3bb783f15c02dp-17, 0x1.8feefe8773245p-21, -0x1.33e4dc16c611cp-27, \
  -0x1.20ba5e180f15dp-30, 0x1.bcb4e21576f37p-39, 0x1.3f24dd2b5246ep-40, \
  0x1.367315102c5ebp-48, -0x1.21f5e8818c7b1p-50, -0x1.6d793361e1ea5p-57, \
  0x1.c28300a46ae08p-61, 0x1.cf0b291a67cccp-67, -0x1.30fb191691579p-71, \
  0x1.b4d791985a1fap-2, 0x1.b067e74967167p-6, -0x1.58e300a4b1b92p-7, \
  -0x1.00b46833cc79bp-13, 0x1.34e09e1c901afp-15, 0x1.7331a5808e16bp-22, \
  -0x1.4dea0a79a501ep-24, -0x1.907102dd2f6ecp-31, 0x1.0f13670751b9dp-33, \
  0x1.58dba30094c95p-40, -0x1.6db889e81c82bp-43, -0x1.f4bcecf975115p-50, \
  0x1.b1bd031dbe09ap-53, 0x1.3ffc01b53a5d1p-59, -0x1.d3ee70f4a6798p-63, \
  -0x1.736569d694479p-69, 0x1.9d0720b1d62b3p-2, -0x1.7e58d243b8126p-5, \
  -0x1.e16914f2a29d6p-8, 0x1.447898c0a805ap-12, 0x1.3b46d55d9c967p-16, \
  -0x1.b69bdff751309p-21, -0x1.e704c727623fcp-26, 0x1.7e75016a55cf5p-30, \
  0x1.0e54d8b74aa5ep-35, -0x1.006c316125378p-39, -0x1.cd75addbb8d74p-46, \
  0x1.1fc3e1ab7e2d7p-49, 0x1.29fab52e7aa5dp-56, -0x1.1b4958f9ab21dp-59, \
  -0x1.cf0e98bc5789ap-68, 0x1.f760ac461323bp-70, 0x1.0cc2f87baafb7p-2, \
  -0x1.718c56640ffa2p-4, -0x1.4a7f73835c5ecp-9, 0x1.09951358bf899p-11, \
  -0x1.7bdd369c4bf57p-20, -0x1.37b470df76acbp-20, 0x1.37ffe666a1cf5p-26, \
  0x1.c104809194841p-30, -0x1.830b402ae0f97p-35, -0x1.bfd6fcc2387c6p-40, \
  0x1.30fbfd6e85077p-44, 0x1.2aa3d507c429dp-50, -0x1.6e72915121152p-54, \
  -0x1.10182c7d2b8a2p-62, 0x1.67a3b38f55c8cp-64, -0x1.377b86af31ef4p-71, \
  -0x1.2878e279eececp-1, 0x1.05cfc4130f966p-3, 0x1.7f1a8db948a98p-7, \
  -0x1.22a1424021c37p-9, 0x1.5bc282215ced6p-17, 0x1.f680128b00aa0p-17, \
  -0x1.6789d925fc346p-21, -0x1.b90190132bc50p-25, 0x1.53fe189317d4dp-28, \
  0x1.7a92d5223c5eap-35, -0x1.6f7da12ea15fap-36, 0x1.3089325b8e610p-41, \
  0x1.f54ae28731052p-45, -0x1.05cc3f5591d83p-48, -0x1.29986ffab00f1p-54, \
  0x1.f199a537e1ae5p-57, -0x1.12edc13899186p-3, -0x1.22c6b258ff8a2p-6, \
  0x1.0785b1251d381p-8, -0x1.091b10bbda917p-13, -0x1.94df00abfebd2p-18, \
  0x1.e58071fd76ca2p-22, -0x1.ec8c37688e73bp-29, -0x1.15046deb45d58p-31, \
  0x1.234e49e8bd46dp-36, 0x1.1c97c90eadd49p-43, -0x1.49d0976eb178bp-46, \
  0x1.560041a9dbb55p-52, 0x1.36bb1e7dff982p-57, -0x1.e501bf4108776p-62, \
  0x1.77646e6d1fc20p-69, 0x1.25b1aa7fbfac2p-72, -0x1.24418e3bb40b2p-3, \
  0x1.02bc9c0f6dd4ep-7, 0x1.30314ecee4d8dp-9, -0x1.7116eb9cbfaa5p-13, \
  0x1.b163c0a124ee8p-20, 0x1.4258a0586ecbfp-22, -0x1.b9a78eb494859p-27, \
  -0x1.1efe40fc7a457p-37, 0x1.0a86a6ba37618p-36, -0x1.c5fba5ec8b1c5p-42, \
  -0x1.fd6df65b307f1p-49, 0x1.f6f5423908d26p-52, -0x1.2be01cd9391eep-57, \
  -0x1.35c3a655128c4p-63, 0x1.5c82691f9e271p-67, -0x1.42370d2ed8be2p-73, \
  -0x1.89a1b743f0353p-4, 0x1.30a4c0931490ep-5, -0x1.0d72885349820p-14, \
  -0x1.4774a23f0dfa3p-11, 0x1.1bf3aac5a84dep-14, -0x1.5a2125c21c35bp-20, \
  -0x1.411eb83020fa0p-22, 0x1.cb2f5b008ad3bp-26, -0x1.f3d1d1366579dp-32, \
  -0x1.2fc361115ff8ap-34, 0x1.7835938299d1ap-38, -0x1.976e81484131bp-44, \
  -0x1.596aea3e9742ap-47, 0x1.ab02d8aa621f4p-51, -0x1.28c53717892cep-56, \
  -0x1.08cc0fc6f5a1ep-60
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_GLOBAL_TABLE double cheb_coef_dev[928] = {LAPLACE_LOG_PHI_CHEB_COEF};
#endif
inline constexpr double cheb_coef_host[928] = {LAPLACE_LOG_PHI_CHEB_COEF};
LAPLACE_HD_FORCEINLINE double cheb_coef(int i) {
  KOKKOS_IF_ON_DEVICE((return cheb_coef_dev[i];))
  KOKKOS_IF_ON_HOST((return cheb_coef_host[i];))
}

// clang-format off
#define LAPLACE_LOG_PHI_CHEB_ZERO \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x1.009b46f78e413p+0, 0x1.311a50a8acfb4p-55, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  -0x1.e8cb468cb7426p-6, -0x1.1e43c3f9f0b48p-61, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x1.091ffef66015ap+1, -0x1.fd78be88706b9p-53, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, -0x1.be05095424c12p-1, \
  0x1.8e26921cd71c4p-55, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x1.3b3897cd912f8p+0, -0x1.fb5861e8a0230p-55, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x1.5f71195cad7e3p+1, \
  -0x1.048babdfbd477p-54, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0, 0x0.0p+0, 0x0.0p+0, \
  0x0.0p+0
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double cheb_zero_dev[106] = {LAPLACE_LOG_PHI_CHEB_ZERO};
#endif
inline constexpr double cheb_zero_host[106] = {LAPLACE_LOG_PHI_CHEB_ZERO};
LAPLACE_HD_FORCEINLINE double cheb_zero(int i) {
  KOKKOS_IF_ON_DEVICE((return cheb_zero_dev[i];))
  KOKKOS_IF_ON_HOST((return cheb_zero_host[i];))
}

// clang-format off
#define LAPLACE_LOG_PHI_CHEB_BIN \
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, \
  3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, \
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, \
  3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, \
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, \
  3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, \
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, \
  2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, \
  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, \
  3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, \
  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, \
  5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE int cheb_bin_dev[240] = {LAPLACE_LOG_PHI_CHEB_BIN};
#endif
inline constexpr int cheb_bin_host[240] = {LAPLACE_LOG_PHI_CHEB_BIN};
LAPLACE_HD_FORCEINLINE int cheb_bin(int i) {
  KOKKOS_IF_ON_DEVICE((return cheb_bin_dev[i];))
  KOKKOS_IF_ON_HOST((return cheb_bin_host[i];))
}

// clang-format off
#define LAPLACE_LOG_PHI_CHEB_HAS_ZERO \
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, \
  1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE int cheb_has_zero_dev[53] = {LAPLACE_LOG_PHI_CHEB_HAS_ZERO};
#endif
inline constexpr int cheb_has_zero_host[53] = {LAPLACE_LOG_PHI_CHEB_HAS_ZERO};
LAPLACE_HD_FORCEINLINE int cheb_has_zero(int i) {
  KOKKOS_IF_ON_DEVICE((return cheb_has_zero_dev[i];))
  KOKKOS_IF_ON_HOST((return cheb_has_zero_host[i];))
}

// per order k = 1..6: the number of pieces, the degree of the expansions, and the
// first index of the order in cheb_edge, in cheb_coef and in cheb_has_zero
LAPLACE_HD_FORCEINLINE constexpr int cheb_pieces(int k) {
  constexpr int v[6] = {8, 8, 8, 8, 8, 13};
  return v[k - 1];
}
LAPLACE_HD_FORCEINLINE constexpr int cheb_degree(int k) {
  constexpr int v[6] = {15, 16, 17, 18, 19, 15};
  return v[k - 1];
}
LAPLACE_HD_FORCEINLINE constexpr int cheb_edge_offset(int k) {
  constexpr int v[6] = {0, 9, 18, 27, 36, 45};
  return v[k - 1];
}
LAPLACE_HD_FORCEINLINE constexpr int cheb_coef_offset(int k) {
  constexpr int v[6] = {0, 128, 264, 408, 560, 720};
  return v[k - 1];
}
LAPLACE_HD_FORCEINLINE constexpr int cheb_flag_offset(int k) {
  constexpr int v[6] = {0, 8, 16, 24, 32, 40};
  return v[k - 1];
}

}  // namespace detail_normal
// END GENERATED: tools/reference/log_phi_cheb.py

namespace detail_normal {

// d^k log Phi / dt^k at t, k = 1..6; the work is shared by the six public
// functions below, each of which asks for its own order only.
//
// For t >= t_switch: from R and the recursion R' = -R (t + R):
//   g1 = R, g2 = -R (t + R), g3 = -g2 (t + 2R) - R,
//   g4 = -g3 (t + 2R) - 2 g2 (1 + g2), g5 = -g4 (t + 2R) - 3 g3 (1 + 2 g2),
//   g6 = -g5 (t + 2R) - 4 g4 (1 + 2 g2) - 6 g3^2.
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
// kappa_k from the raw moments mu_0 = 1, mu_1, .., mu_k (k = 1..6), by the
// recursion kappa_n = mu_n - sum_(i=1)^(n-1) C(n-1, i-1) kappa_i mu_(n-i)
// (statmodels7, enet_cumulants in pt_enet_logz.h); measured more accurate here
// than the closed forms in powers of mu_1
LAPLACE_HD_FORCEINLINE double cumulant(int k, const double* mu) {
  double kap[7];
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
    const double g5 = -g4 * (t + 2.0 * R) - 3.0 * g3 * (1.0 + 2.0 * g2);
    if (k == 5) return g5;
    return -g5 * (t + 2.0 * R) - 4.0 * g4 * (1.0 + 2.0 * g2) - 6.0 * g3 * g3;
  }
  const double x = -t;
  double mu[7] = {1.0, 0, 0, 0, 0, 0, 0};
  const double ix = 1.0 / x, w = ix * ix;
  const double kk = c2 * w;
  const int K = k + m0 + (kk < 20000.0 ? static_cast<int>(Kokkos::ceil(kk)) : 20000);
  const double K1 = K + 1.0;
  double rho = 2.0 * K1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * K1 * w));
  double r[7] = {0, 0, 0, 0, 0, 0, 0};
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

// d^k log Phi(t) / dt^k for t in [cheb_lo, cheb_hi). The piece of t comes from
// its bin: the table cheb_bin gives the piece whose left edge is at or before
// the bin's left end, and one comparison corrects the index, which is enough
// because no piece is narrower than a bin. On the piece [a, b) the expansion is
// summed at x = 2 (t - a) / (b - a) - 1 in [-1, 1] by Clenshaw's recurrence
// b_n = c_n + 2 x b_(n+1) - b_(n+2) (b beyond the degree zero), the value
// c_0 + x b_1 - b_2. A piece whose midpoint is a zero z of g_k carries the
// expansion of g_k(t) / (t - z), and the factor is restored from the pair of
// doubles z_hi + z_lo: t - z_hi is exact, so the product keeps its relative
// accuracy however near t lies to the zero.
template <int K>
LAPLACE_HD double log_Phi_dk_cheb(double t) {
  const int N = cheb_degree(K);
  const int eo = cheb_edge_offset(K);
  // t - cheb_lo rounds up to cheb_hi - cheb_lo for t one ulp below cheb_hi, so the bin
  // index is clamped; without it the last bin of an order reads past its row
  int bin = static_cast<int>((t - cheb_lo) * cheb_bin_scale);
  if (bin >= cheb_bins) bin = cheb_bins - 1;
  int p = cheb_bin((K - 1) * cheb_bins + bin);
  if (p + 1 < cheb_pieces(K) && t >= cheb_edge(eo + p + 1)) ++p;
  const double a = cheb_edge(eo + p), b = cheb_edge(eo + p + 1);
  const double x = 2.0 * (t - a) / (b - a) - 1.0;
  const int base = cheb_coef_offset(K) + p * (N + 1);
  double b1 = 0.0, b2 = 0.0;
  for (int n = N; n >= 1; --n) {
    const double b0 = 2.0 * x * b1 - b2 + cheb_coef(base + n);
    b2 = b1;
    b1 = b0;
  }
  const double v = cheb_coef(base) + x * b1 - b2;
  const int f = cheb_flag_offset(K) + p;
  if (cheb_has_zero(f)) return v * ((t - cheb_zero(2 * f)) - cheb_zero(2 * f + 1));
  return v;
}

// d^k log Phi / dt^k outside [cheb_lo, cheb_hi): the cumulants below -1
// (K = k + 20 + ceil(300 / t^2) downward steps), the recursion from R above
// cheb_hi (t_switch below every t)
LAPLACE_HD double log_Phi_dk_outside(int k, double t) {
  return log_Phi_dk(k, t, t >= cheb_hi ? -math::detail::pos_inf() : -1.0, 300.0, 20);
}

}  // namespace detail_normal

// d^k log Phi / dt^k, one function per order: the piecewise Chebyshev
// expansion on [-1, cheb_hi) (generated tables, measured in
// note/numerics.md), the cumulants below -1, the recursion from R above
/**
 * \ingroup normal
 * \brief First derivative of \f$\log \Phi(t)\f$, the inverse Mills ratio \f$R(t) = \phi(t)/\Phi(t)\f$.
 *
 * Computes the first derivative:
 * \f[
 * \frac{\mathrm{d}}{\mathrm{d}t} \log \Phi(t) = R(t) = \frac{\phi(t)}{\Phi(t)}
 * \f]
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * for \f$t < -1\f$; and the ratio \f$R(t)\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}}{\mathrm{d}t}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - As \f$t \to -\infty\f$: \f$R(t) \sim -t \to +\infty\f$.
 * - As \f$t \to +\infty\f$: \f$R(t) \sim \phi(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 1.19 ulp of the order's scale, 2.54 relative to the
 * component itself; test bounds: 2.0 ulp below \f$t = -1\f$, 1.8 ulp and 3.9 relative on
 * \f$[-1, 4)\f$, 2.0 ulp1 for \f$t \ge -1\f$ as a whole.
 *
 * \see normal, log_Phi(), log_Phi_d2()
 */
LAPLACE_HD double log_Phi_d1(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<1>(t);
  return detail_normal::log_Phi_dk_outside(1, t);
}
/**
 * \ingroup normal
 * \brief Second derivative of \f$\log \Phi(t)\f$.
 *
 * Computes the second derivative:
 * \f[
 * \frac{\mathrm{d}^2}{\mathrm{d}t^2} \log \Phi(t) = g_2(t) = -R(t)(t + R(t))
 * \f]
 * where \f$R(t)\f$, the first derivative, `log_Phi_d1()`, is the inverse Mills ratio.
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * \f$g_2 = -1 + \kappa_2\f$ for \f$t < -1\f$; and \f$-R(t + R)\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}^2}{\mathrm{d}t^2}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - At \f$t = 0\f$: returns \f$-2/\pi \approx -0.636620\f$.
 * - As \f$t \to -\infty\f$: \f$g_2(t) \to -1.0\f$.
 * - As \f$t \to +\infty\f$: \f$g_2(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 1.28 ulp of the order's scale, 2.21 relative to the
 * component itself; test bounds: 2.5 ulp below \f$t = -1\f$, 2.0 ulp and 3.4 relative on
 * \f$[-1, 4)\f$, 1.0 ulp1 for \f$t \ge -1\f$ as a whole.
 *
 * \see normal, log_Phi(), log_Phi_d1(), log_Phi_d3()
 */
LAPLACE_HD double log_Phi_d2(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<2>(t);
  return detail_normal::log_Phi_dk_outside(2, t);
}
/**
 * \ingroup normal
 * \brief Third derivative of \f$\log \Phi(t)\f$.
 *
 * Computes the third derivative:
 * \f[
 * \frac{\mathrm{d}^3}{\mathrm{d}t^3} \log \Phi(t) = g_3(t) = -g_2(t)(t + 2R(t)) - R(t)
 * \f]
 * where \f$R(t)\f$, the first derivative, `log_Phi_d1()`, is the inverse Mills ratio, and \f$g_2(t)\f$, the second derivative, `log_Phi_d2()`.
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * \f$g_3 = \kappa_3\f$ for \f$t < -1\f$; and the recurrence from \f$R\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}^3}{\mathrm{d}t^3}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - As \f$t \to -\infty\f$: \f$g_3(t) \to 0.0\f$.
 * - As \f$t \to +\infty\f$: \f$g_3(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 1.33 ulp of the order's scale, 2.24 relative to the
 * component itself; test bounds: 22.0 ulp below \f$t = -1\f$, 2.1 ulp and 3.4 relative on
 * \f$[-1, 4)\f$, 1.2 ulp1 for \f$t \ge -1\f$ as a whole.
 *
 * \see normal, log_Phi(), log_Phi_d2(), log_Phi_d4()
 */
LAPLACE_HD double log_Phi_d3(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<3>(t);
  return detail_normal::log_Phi_dk_outside(3, t);
}
/**
 * \ingroup normal
 * \brief Fourth derivative of \f$\log \Phi(t)\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * \frac{\mathrm{d}^4}{\mathrm{d}t^4} \log \Phi(t) = g_4(t) = -g_3(t)(t + 2R(t)) - 2 g_2(t)(1 + g_2(t))
 * \f]
 * where \f$R(t)\f$, \f$g_2(t)\f$, and \f$g_3(t)\f$ are the lower-order derivatives (`log_Phi_d1()`, `log_Phi_d2()`, `log_Phi_d3()`).
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * \f$g_4 = \kappa_4\f$ for \f$t < -1\f$; and the recurrence from \f$R\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}^4}{\mathrm{d}t^4}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - As \f$t \to -\infty\f$: \f$g_4(t) \to 0.0\f$.
 * - As \f$t \to +\infty\f$: \f$g_4(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 1.49 ulp of the order's scale, 2.30 relative to the
 * component itself (CUDA 2.46); test bounds: 50.0 ulp below \f$t = -1\f$, 2.2 ulp and 3.7 relative on
 * \f$[-1, 4)\f$, 2.0 ulp1 for \f$t \ge -1\f$ as a whole.
 *
 * \see normal, log_Phi(), log_Phi_d3(), log_Phi_d5()
 */
LAPLACE_HD double log_Phi_d4(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<4>(t);
  return detail_normal::log_Phi_dk_outside(4, t);
}
/**
 * \ingroup normal
 * \brief Fifth derivative of \f$\log \Phi(t)\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * \frac{\mathrm{d}^5}{\mathrm{d}t^5} \log \Phi(t) = g_5(t) = -g_4(t)(t + 2R(t)) - 3 g_3(t)(1 + 2 g_2(t))
 * \f]
 * where \f$R(t)\f$, \f$g_2(t)\f$, \f$g_3(t)\f$, and \f$g_4(t)\f$ are the lower-order derivatives (`log_Phi_d1()` through `log_Phi_d4()`).
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * \f$g_5 = \kappa_5\f$ for \f$t < -1\f$; and the recurrence from \f$R\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}^5}{\mathrm{d}t^5}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - As \f$t \to -\infty\f$: \f$g_5(t) \to 0.0\f$.
 * - As \f$t \to +\infty\f$: \f$g_5(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 0.84 ulp of the order's scale, 2.85 relative to the
 * component itself; test bounds: 120.0 ulp below \f$t = -1\f$, 1.4 ulp and 4.3 relative on
 * \f$[-1, 4)\f$, 3.0 ulp1 for \f$t \ge -1\f$ as a whole.
 *
 * \see normal, log_Phi(), log_Phi_d4()
 */
LAPLACE_HD double log_Phi_d5(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<5>(t);
  return detail_normal::log_Phi_dk_outside(5, t);
}

/**
 * \ingroup normal
 * \brief Sixth derivative of \f$\log \Phi(t)\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * \frac{\mathrm{d}^6}{\mathrm{d}t^6} \log \Phi(t) = g_6(t) = -g_5(t)(t + 2R(t)) - 4 g_4(t)(1 + 2 g_2(t)) - 6 g_3(t)^2
 * \f]
 * where \f$R(t)\f$, \f$g_2(t)\f$, \f$g_3(t)\f$, \f$g_4(t)\f$, and \f$g_5(t)\f$ are the lower-order derivatives (`log_Phi_d1()` through `log_Phi_d5()`).
 *
 * Evaluated via piecewise Chebyshev expansions on \f$[-1, 4)\f$; downward cumulant recurrence
 * \f$g_6 = \kappa_6\f$ for \f$t < -1\f$; and the recurrence from \f$R\f$ for \f$t \ge 4\f$.
 *
 * \param t Real argument \f$t \in \mathbb{R}\f$.
 * \return The value of \f$\frac{\mathrm{d}^6}{\mathrm{d}t^6}\log \Phi(t)\f$.
 *
 * \par Domain and special values
 * - \f$t \in \mathbb{R}\f$: standard domain.
 * - As \f$t \to -\infty\f$: \f$g_6(t) \to 0.0\f$.
 * - As \f$t \to +\infty\f$: \f$g_6(t) \to 0.0\f$.
 * - If \f$t\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error on \f$[-1, 4)\f$: 1.34 ulp of the order's scale (CUDA 1.41), 2.70 relative to the
 * component itself; test bounds: 1150.0 ulp below \f$t = -1\f$, 2.6 ulp and 4.1 relative on
 * \f$[-1, 4)\f$, 1.0 ulp1 for \f$t \ge -1\f$ as a whole. Below \f$t = -1\f$: 767 ulp on gcc, 478 on CUDA.
 *
 * \see normal, log_Phi(), log_Phi_d5()
 */
LAPLACE_HD double log_Phi_d6(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<6>(t);
  return detail_normal::log_Phi_dk_outside(6, t);
}

// Phi^-1(p)
/**
 * \ingroup normal
 * \brief Standard normal quantile function, \f$\Phi^{-1}(p)\f$.
 *
 * Computes the quantile \f$x\f$ satisfying \f$\Phi(x) = p\f$ for \f$p \in (0, 1)\f$.
 *
 * For \f$p \le 0.5\f$, evaluated from Boost.Math's \f$-\sqrt{2}\operatorname{erfc}^{-1}(2p)\f$,
 * followed for \f$p < 0.1\f$ by one Newton step on \f$\log \Phi\f$:
 * \f[
 * x = x_0 - \frac{\log \Phi(x_0) - \log p}{R(x_0)}
 * \f]
 * where \f$R(x_0) = \phi(x_0)/\Phi(x_0)\f$. For \f$p > 0.5\f$, evaluated as \f$-\Phi^{-1}(1 - p)\f$.
 *
 * \param p Probability \f$p \in (0, 1)\f$.
 * \return The quantile \f$x = \Phi^{-1}(p)\f$.
 *
 * \par Domain and special values
 * - \f$p \in (0, 1)\f$: standard domain.
 * - At \f$p = 0.5\f$: returns \f$0.0\f$.
 * - At \f$p = 0.0\f$: returns \f$-\infty\f$.
 * - At \f$p = 1.0\f$: returns \f$+\infty\f$.
 * - If \f$p < 0.0\f$ or \f$p > 1.0\f$ or NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Error in units of conditioned ulpC: 0.82 for \f$p < 10^{-300}\f$, 0.99 to \f$p = 10^{-10}\f$,
 * 1.58 to \f$p = 0.1\f$, 2.32 at the centre; test bound 3.5 ulpC.
 *
 * \see normal, Phi(), Phi_inv_logp()
 */
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
/**
 * \ingroup normal
 * \brief Standard normal quantile function from log-probability, \f$\Phi^{-1}(e^l)\f$.
 *
 * Computes the quantile \f$x\f$ satisfying \f$\log \Phi(x) = l\f$ for \f$l = \log p \le 0\f$.
 *
 * For \f$l \ge -\log 2\f$, evaluated through \f$q = 1 - p = -\operatorname{expm1}(l)\f$ as \f$-\Phi^{-1}(q)\f$.
 * For \f$-700.0 < l < -\log 2\f$, started from Boost.Math's \f$-\sqrt{2}\operatorname{erfc}^{-1}(2e^l)\f$.
 * For \f$l \le -700.0\f$, started from the asymptotic approximation
 * \f[
 * x_0 = -\sqrt{-2 l - \log(-4\pi l)}
 * \f]
 * Followed by up to 8 Newton steps \f$x_{k+1} = x_k - (\log \Phi(x_k) - l)/R(x_k)\f$ until
 * the update satisfies \f$|\Delta x| \le 4\varepsilon |x|\f$.
 *
 * \param l Log-probability \f$l \in (-\infty, 0]\f$.
 * \return The quantile \f$x = \Phi^{-1}(e^l)\f$.
 *
 * \par Domain and special values
 * - \f$l \in (-\infty, 0]\f$: standard domain.
 * - At \f$l = 0.0\f$: returns \f$+\infty\f$.
 * - At \f$l = -\infty\f$: returns \f$-\infty\f$.
 * - If \f$l > 0.0\f$ or NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Conditioned error at most 2.76 ulpC everywhere; test bound 4.5 ulpC.
 *
 * \see normal, log_Phi(), Phi_inv()
 */
LAPLACE_HD double Phi_inv_logp(double l) {
  using namespace detail_normal;
  if (!(l < 0.0)) {
    if (l == 0.0) return detail::pos_inf();
    return detail::quiet_nan();
  }
  if (l == -detail::pos_inf()) return -detail::pos_inf();
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
LAPLACE_HD_FORCEINLINE double log_Phi_d6(double t) { return own::log_Phi_d6(t); }
LAPLACE_HD_FORCEINLINE double Phi_inv(double p) { return own::Phi_inv(p); }
LAPLACE_HD_FORCEINLINE double Phi_inv_logp(double l) { return own::Phi_inv_logp(l); }

}  // namespace laplace::math
