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

// BEGIN GENERATED: tools/reference/log_phi_cheb.py
// d^k log Phi / dt^k on [cheb_lo, cheb_hi): piecewise Chebyshev expansions of width
// cheb_width, c_0 + sum_(n >= 1) c_n T_n(x), x = 2 (t - a) / width - 1 on the piece
// [a, a + width); generated by tools/reference/log_phi_cheb.py (do not edit).
namespace detail_normal {

constexpr double cheb_lo = -0x1.0000000000000p+0;  // the expansions cover [cheb_lo, cheb_hi)
constexpr double cheb_hi = 0x1.0000000000000p+2;
constexpr double cheb_width = 0x1.0000000000000p+0;  // pieces of this length
constexpr int cheb_pieces = 5;
constexpr int cheb_size = 480;

// clang-format off
#define LAPLACE_LOG_PHI_CHEB \
  0x1.26ba557bb4303p+0, -0x1.74e5b2f29125ap-2, 0x1.4ec38de39af0fp-7, \
  0x1.1761689b70c27p-11, 0x1.b80cb2060ae9cp-17, -0x1.90485eb73664dp-21, \
  -0x1.d3d1b39e3df92p-24, -0x1.97f6bca64c238p-28, -0x1.47ddff7203740p-35, \
  0x1.8d19dab4293f0p-36, 0x1.2724b19e00828p-39, 0x1.695bdf25cf5bbp-44, \
  -0x1.609ccee3fa989p-49, -0x1.4a0a2ebe5a224p-51, -0x1.64cb6d75f9ad1p-55, \
  -0x1.cb847efcc83b3p-61, 0x1.f9d68b4046391p-64, 0x1.e565a266d9582p-67, \
  0x1.0d4e4adf34d29p-1, -0x1.05b5db397ad9ap-2, 0x1.131a6bd6743f9p-6, \
  0x1.cba1f161b33a8p-12, -0x1.3307ee5f2cd38p-15, -0x1.143613b9b5d6ep-18, \
  -0x1.b847bcf8d5883p-25, 0x1.613429d10343bp-26, 0x1.cad2e7f901463p-30, \
  -0x1.b87107db15bd7p-37, -0x1.88120eaa538e0p-37, -0x1.85c935819d684p-41, \
  0x1.7dd4f95a8fb53p-46, 0x1.a1825d0bf87b5p-48, 0x1.3ac3e9912ed25p-52, \
  -0x1.3a5d7a44fee4dp-56, -0x1.abf931559b105p-59, -0x1.cabd33b078bedp-64, \
  0x1.3dcc7c33c3bc9p-3, -0x1.d96027c1869f4p-4, 0x1.0b5499785e28bp-6, \
  -0x1.3f86964c4d29bp-11, -0x1.1ea67aae004a6p-14, 0x1.e6df09746885fp-19, \
  0x1.1a9261de66236p-21, -0x1.3906975143ec6p-26, -0x1.2b567fac1bd9fp-28, \
  0x1.85c1b35696437p-35, 0x1.2f2e0665cee88p-35, 0x1.1289cd033ba6fp-41, \
  -0x1.1c0062fb5aad8p-42, -0x1.685f399746d52p-47, 0x1.e33418f95851cp-50, \
  0x1.1113fdf15c61ep-53, -0x1.693841f04d660p-57, -0x1.52d565abcb486p-60, \
  0x1.84497cc39d3bcp-6, -0x1.9335f20890376p-6, 0x1.8ff738a3ec638p-8, \
  -0x1.a2424f9e84a0fp-11, 0x1.59625b84299d4p-15, 0x1.75793fa21bdc2p-19, \
  -0x1.0183f8f447e18p-21, 0x1.2b9dea50f0aa4p-30, 0x1.047becc1e2b76p-28, \
  -0x1.597f2f2a874afp-33, -0x1.a5567fec02ffbp-36, 0x1.21b37fb599ce5p-39, \
  0x1.fbdf4387b1536p-44, -0x1.5d9d60a797ce9p-46, -0x1.c20e6c4348db3p-53, \
  0x1.5db6caa43aba3p-53, -0x1.0f0c8449f1b7cp-58, -0x1.27dfd4f4848e9p-60, \
  0x1.97efcc421e782p-10, -0x1.0510c6b10f9bap-9, 0x1.725bc866de48bp-11, \
  -0x1.4ba2ef1577b7bp-13, 0x1.855b817dca139p-16, -0x1.1a25f7c4e392dp-19, \
  0x1.38189c7d0ff9cp-24, 0x1.1f1ceb23b1df2p-27, -0x1.68d8eb686e246p-30, \
  0x1.ce370199448e7p-35, 0x1.54a8f59b9866bp-38, -0x1.947336a7e7cccp-41, \
  0x1.87b3f8a261a2ep-46, 0x1.a2fd6c2ece965p-49, -0x1.8528c0fd24967p-52, \
  0x1.eae653711e5e8p-58, 0x1.b76d19a3d2fbbp-60, -0x1.560ef0611bd6fp-63, \
  -0x1.73439dc951a5ap-1, 0x1.4f9cd635daa60p-4, 0x1.a215293f8002dp-8, \
  0x1.b290a47f6a1d6p-13, -0x1.f9e7535241b1cp-17, -0x1.5f0361a833191p-19, \
  -0x1.63373b4f6cf27p-23, -0x1.30d78c24f1b1ep-30, 0x1.c0a9c215ac9f7p-31, \
  0x1.706734d11c21cp-34, 0x1.ecac0afe38996p-39, -0x1.0d5268c902a9fp-43, \
  -0x1.0c91f57121298p-45, -0x1.37336781addf7p-49, -0x1.a6bdd9dfb6ae9p-55, \
  0x1.fd308b19300efp-58, 0x1.01cba5a8a11e7p-60, 0x1.acffec74eaf06p-65, \
  -0x1.04627d2c77b8fp-1, 0x1.11e6c5e93235cp-3, 0x1.535e0d0320a6bp-8, \
  -0x1.33a5ed4209d17p-11, -0x1.56da0189714d8p-14, -0x1.3bfdc5b9fbeb4p-20, \
  0x1.34cb8f58ff8aap-21, 0x1.c7010014875dbp-25, -0x1.0854f78d426e4p-31, \
  -0x1.e8f3f23cf43fbp-32, -0x1.09563220633b4p-35, 0x1.22a017f431d4cp-40, \
  0x1.5211545c7e33fp-42, 0x1.10173c1183356p-46, -0x1.28973d3bb040cp-50, \
  -0x1.aa1826c2e1507p-53, -0x1.df9a9b014a41bp-58, 0x1.e10a92b9bfdf2p-61, \
  -0x1.e0ca6cb1e6b2cp-3, 0x1.091dd96adae69p-3, -0x1.da913c1804dfap-8, \
  -0x1.1b6006c1a10bbp-10, 0x1.2e29569bb7b6cp-14, 0x1.a339f62f9f55ap-17, \
  -0x1.1107a684ce78bp-21, -0x1.2867277e77dd3p-23, 0x1.bc3bc499ec3b3p-30, \
  0x1.77ac16d1fe637p-30, 0x1.7076b620cbd48p-36, -0x1.a6b896a21f995p-37, \
  -0x1.20d87874c621dp-41, 0x1.a3feeb743578fp-44, 0x1.fa732b11b5ae9p-48, \
  -0x1.675532fbe7498p-51, -0x1.649444b5e23d9p-54, 0x1.e30ef4661c8bap-59, \
  -0x1.ba31c7dab13c1p-5, 0x1.9544de4eb45afp-5, -0x1.37deae9108254p-7, \
  0x1.53696ab1fdda3p-11, 0x1.d30d25db53775p-15, -0x1.7e3c348af0c0bp-17, \
  0x1.acb28585218bfp-26, 0x1.027038dec463ep-23, -0x1.7e8752220e77cp-28, \
  -0x1.05d9f18f29bf7p-30, 0x1.89f0bb726f223p-34, 0x1.7c1e64582057cp-38, \
  -0x1.198509d1165dcp-40, -0x1.921c9b494d8e0p-47, 0x1.456a5b8a7d0a7p-47, \
  -0x1.09ff91cdb9c6cp-52, -0x1.388137bd02084p-54, 0x1.433c9f0dfc3fep-58, \
  -0x1.449e08898086fp-8, 0x1.8ace0f36fb048p-8, -0x1.fc6a0ec3875a6p-10, \
  0x1.87246d01cbbcfp-12, -0x1.5eb5046a78dabp-15, 0x1.c8eb8401a95b1p-20, \
  0x1.fa714ba39cc6ap-23, -0x1.672cd73dd372fp-25, 0x1.ff581292c018dp-30, \
  0x1.ac142a9ab16a5p-33, -0x1.14b9e735a0ed8p-35, 0x1.207bcc1974fcdp-40, \
  0x1.554e5dce6f42bp-43, -0x1.52cb98150f56cp-46, 0x1.c0ebd0ced11dep-52, \
  0x1.b810c870acedap-54, -0x1.6983ad676b577p-57, 0x1.475d99b3e3e0ap-63, \
  0x1.521af72906de5p-3, 0x1.a012fcb9243bbp-5, 0x1.3f1079961c281p-9, \
  -0x1.0116432de38fdp-12, -0x1.b700725cdb7c6p-15, -0x1.08a66130adbd0p-18, \
  -0x1.e1c254dce8764p-26, 0x1.c30b4ae3f8d18p-26, 0x1.9db7031df4086p-29, \
  0x1.30c467261901ep-33, -0x1.7a309a973b370p-38, -0x1.938fdc652efb6p-40, \
  -0x1.f7d2a0b5e3575p-44, -0x1.69d876fa7a400p-49, 0x1.e0e77cd733bc5p-52, \
  0x1.01b4f528b316ap-54, 0x1.c4fd37cb572afp-59, -0x1.6b07fee07d3cdp-66, \
  -0x1.6959b8713a3edp-66, 0x1.0e48ef61fac86p-2, 0x1.48c42725c0e7cp-5, \
  -0x1.ceeb439bb6aa2p-8, -0x1.533cbbabf7ddbp-10, -0x1.725fb8a7effeap-16, \
  0x1.cea2eebcb7ecdp-17, 0x1.89d7e808ae770p-20, -0x1.1cd0918ec6326p-26, \
  -0x1.123e0251febe9p-26, -0x1.47b9a0183c41ep-30, 0x1.966be0954a834p-35, \
  0x1.f90f481fe416ap-37, 0x1.b3efe9617faf5p-41, -0x1.055b356c9ba8dp-44, \
  -0x1.8d760ebd61dcbp-47, -0x1.d6dfd86170239p-52, 0x1.004acca8af5acp-54, \
  0x1.175853fb43c4ep-57, 0x1.7a5d6c02d8734p-63, 0x1.0299d9be8d448p-2, \
  -0x1.d1396de7370f2p-5, -0x1.a0ffeb136883cp-7, 0x1.2af9c619ba111p-10, \
  0x1.0203e1e121b9dp-12, -0x1.97c840fed2dadp-17, -0x1.0015ff2876ed2p-18, \
  0x1.c338c862da449p-25, 0x1.a221b318fa349p-25, 0x1.bf40f23b82591p-31, \
  -0x1.1ff9a4d0fed27p-31, -0x1.aa6e2daf8e12ap-36, 0x1.52a15f3b737d4p-38, \
  0x1.b5a1bfe6c8041p-42, -0x1.4ee0097bfa800p-45, -0x1.60c1722dbd2b3p-48, \
  0x1.ffd6502e54eadp-53, 0x1.e969441289343p-55, -0x1.69937e2969891p-61, \
  0x1.a4b81c30d50a8p-4, -0x1.3438226056f82p-4, 0x1.ee67bc4415f35p-8, \
  0x1.d34618589692ap-11, -0x1.d6cc78dcda812p-13, 0x1.c793ea18da93dp-22, \
  0x1.bfb234349befdp-19, -0x1.76efbc5daf7c5p-23, -0x1.248bd44eefbc2p-25, \
  0x1.e5e57117bedfcp-29, 0x1.04adb90f9da4cp-32, -0x1.a1de4dd302bdep-35, \
  -0x1.4e57d9f12f15dp-41, 0x1.1a5039a7b3afcp-41, -0x1.e826f168049d2p-47, \
  -0x1.366b38bcdcb6fp-48, 0x1.5307ff36f6f14p-52, 0x1.0aff8012a8a46p-55, \
  -0x1.1a1a71f944a4dp-58, 0x1.d4aecbb61c0bep-7, -0x1.0912b3a270644p-6, \
  0x1.2782f1fc841d9p-8, -0x1.5bb5881596e0ep-11, 0x1.13d01d95a7f0cp-15, \
  0x1.7fbe2a70fce8bp-18, -0x1.38629d6c3d051p-20, 0x1.f498db23a9e03p-25, \
  0x1.e49ee9dbff854p-28, -0x1.57e6ede2c7147p-30, 0x1.841cf6fbf6cfbp-35, \
  0x1.00b990210a38ap-37, -0x1.11a834e1417c3p-40, 0x1.7d938c6d8d2f8p-46, \
  0x1.9d36afdafa485p-48, -0x1.675948ed39549p-51, 0x1.4de7e2b0532b3p-57, \
  0x1.15323d190176ep-58, -0x1.bd7213d9de6edp-62, 0x1.99e3807a9c22dp-4, \
  0x1.31560a90e6fa8p-6, -0x1.8bdf0fa20638fp-9, -0x1.b74de0a6a5b29p-11, \
  -0x1.47b55ba61c250p-14, -0x1.35b92728d8d89p-21, 0x1.8d4eeb5e83a3fp-21, \
  0x1.9cc4c3e6ac014p-24, 0x1.5284eb84f652fp-28, -0x1.e47e6e900e3c3p-33, \
  -0x1.1602217973ce7p-34, -0x1.7835aa6086ef6p-38, -0x1.1e73e7c70335ap-43, \
  0x1.a84e27e39221cp-46, 0x1.e2fe392007983p-49, 0x1.c1cd53a26e7acp-53, \
  -0x1.a894624192035p-60, -0x1.97f214745815fp-60, -0x1.36df1b9867958p-63, \
  -0x1.6d24f4f68f4f3p-68, 0x1.2986c47d7263ap-4, -0x1.d1873bcc7d9bap-5, \
  -0x1.f3d62a84e8428p-7, -0x1.4dfc186378befp-12, 0x1.209ddfa171404p-12, \
  0x1.231d0223b9fd3p-15, -0x1.0feb290367972p-21, -0x1.113af8b23703fp-21, \
  -0x1.6b4a9a67a2b0ep-25, 0x1.03099fc7ba9b1p-29, 0x1.598e6ce8464f1p-31, \
  0x1.418cda9b0241fp-35, -0x1.ac14ada6807e8p-39, -0x1.59c51b8760644p-41, \
  -0x1.b03b6b01c61d2p-46, 0x1.0118af2a9e6e7p-48, 0x1.26cdfb3260877p-51, \
  0x1.9bc503de274a4p-57, -0x1.ffde089779c49p-59, -0x1.bc82b4a186ed0p-62, \
  -0x1.b5b0a8b5e9aecp-4, -0x1.913e18a377842p-4, 0x1.b88c5314d6062p-7, \
  0x1.f83a4dfe1ff4cp-9, -0x1.fa95847044dc4p-13, -0x1.79aeb8846fdb1p-14, \
  0x1.926667215aa04p-20, 0x1.9c918e10a227ep-20, 0x1.e52df2b6e9112p-26, \
  -0x1.64094216032c1p-26, -0x1.1fb1dcc09930ep-30, 0x1.f765f79dad806p-33, \
  0x1.5e78aa061efc6p-36, -0x1.2305cedfeee2fp-39, -0x1.46b07958e1bb8p-42, \
  0x1.fe396c8c4d11bp-47, 0x1.01386c7fe73d6p-48, -0x1.9ce3a207d91aap-55, \
  -0x1.63bfd4f8d32d2p-55, -0x1.876d06509818cp-61, -0x1.2943840012b19p-3, \
  0x1.d1a0938592f28p-5, 0x1.5e93cc0888d14p-7, -0x1.cc728be8300c8p-9, \
  0x1.f39c617e3420fp-19, 0x1.4b3d9e954e93ep-14, -0x1.3faab3dff7285p-18, \
  -0x1.220224899800ep-20, 0x1.0ce21e404c8e4p-23, 0x1.44d7e2abdd9d3p-27, \
  -0x1.1bbc5743b3e85p-29, -0x1.0144a7a770ce5p-35, 0x1.c63f1eaefd1c1p-36, \
  -0x1.a0b10ca35f7c3p-41, -0x1.20cfc061b8618p-42, 0x1.4e20d2f491aa6p-46, \
  0x1.1a6267ab45058p-49, -0x1.39cb109951b8ap-52, -0x1.4d10688e291c5p-57, \
  0x1.c957cf8dc0655p-59, -0x1.293222f26f2c7p-5, 0x1.2b981a0944638p-5, \
  -0x1.00fb7a7ff641ep-7, 0x1.054a0330117aap-11, 0x1.e655c81d73698p-14, \
  -0x1.d0c34cb2cec53p-16, 0x1.aa04c40dd1a8cp-20, 0x1.e84fb7c661328p-23, \
  -0x1.801f762c5f6d7p-25, 0x1.d866f530d69fdp-30, 0x1.622ab96044c53p-32, \
  -0x1.97a7f143bc7adp-35, 0x1.2b9332d6b74bap-40, 0x1.6a2f0712dfbd8p-42, \
  -0x1.4e95e8496d5dep-45, 0x1.3e5a66897c759p-51, 0x1.26e60a7c30b32p-52, \
  -0x1.f1af84dad6b40p-56, 0x1.82a6467c9337fp-62, 0x1.b868bd21c44fdp-63, \
  0x1.081d746cf3e28p-5, -0x1.a00f4006764e8p-6, -0x1.49c4b11f98bfep-7, \
  -0x1.4303064701592p-10, -0x1.29228a71e7cb4p-17, 0x1.2c9557c6b2f7fp-16, \
  0x1.68139a049d0dcp-19, 0x1.4d139fe81e7e7p-23, -0x1.189165397356fp-27, \
  -0x1.5c52e735f51d3p-29, -0x1.0148e10d6aa21p-32, -0x1.a07abc48b646ap-38, \
  0x1.5c0414f222862p-40, 0x1.a663ec39d13acp-43, 0x1.a24a449de558dp-47, \
  -0x1.d42e11ab539f3p-54, -0x1.b31ce5311d215p-54, -0x1.5ccd7b4e0cdeep-57, \
  -0x1.abaf757f89f81p-62, 0x1.db077acf528f9p-66, 0x1.7b1b59500d7ebp-68, \
  -0x1.d7f6af676ee00p-4, -0x1.e1da7025af2b8p-4, -0x1.9bdce6bc51164p-9, \
  0x1.1fbba5f391704p-8, 0x1.6474f7639020ap-11, -0x1.c4735bbf9fe6dp-17, \
  -0x1.dbd2d24616f59p-17, -0x1.6494f1d424211p-20, 0x1.2a3078e4a8ad6p-24, \
  0x1.ad6a24dfa3f3ap-26, 0x1.b16948fdafbb8p-30, -0x1.43f1a159f797ap-33, \
  -0x1.170c72ee6be4fp-35, -0x1.710f8e8b9c5f7p-40, 0x1.e3b36f926c88cp-43, \
  0x1.248de2be1f401p-45, 0x1.a527228379b8dp-51, -0x1.200c3a20a3a9ap-52, \
  -0x1.05a3198900619p-55, -0x1.f554b6f2b1468p-64, 0x1.2540db73dd974p-62, \
  -0x1.63c5a5257f0acp-3, 0x1.a8de4825a279ap-4, 0x1.6bc39befc3caap-5, \
  -0x1.f5c15de6718f7p-9, -0x1.cd03d1da859d0p-10, 0x1.3509a274d3336p-15, \
  0x1.62d29960c69afp-15, 0x1.cf3546f3ceca8p-21, -0x1.8b30bb71d1fe3p-21, \
  -0x1.5f8abc31a46a6p-25, 0x1.566bb9bc64d56p-27, 0x1.0272f7e36256ep-30, \
  -0x1.d5304001396c4p-34, -0x1.19e1e84d39983p-36, 0x1.dc981555623c6p-41, \
  0x1.fc40f045f5f19p-43, -0x1.bdc06e2604408p-49, -0x1.8bfa2e76224c2p-49, \
  -0x1.c3a47f6b650fbp-55, 0x1.0f684872d1a81p-55, 0x1.a59f0289f1b70p-60, \
  0x1.7e7784b95ad2ap-4, 0x1.5e5e7d43381cdp-4, -0x1.4ca43b30e07fap-5, \
  -0x1.aa762a85a3af6p-15, 0x1.9635b7a87138ap-10, -0x1.cf094601ebe82p-14, \
  -0x1.f5d3a48c400aep-16, 0x1.076c7ce06d44ep-18, 0x1.6c06d9227daafp-22, \
  -0x1.5d6857f7d255fp-24, -0x1.6c05dedba5e08p-30, 0x1.50c54733a3207p-30, \
  -0x1.48ef0aad5898fp-35, -0x1.f507e7cd5a4b2p-37, 0x1.3419f2ec9f7f2p-40, \
  0x1.18ea376a17df2p-43, -0x1.4934b62a44263p-46, -0x1.7830412d266bdp-51, \
  0x1.0cc2de28ab812p-52, -0x1.7dcb8d382bfa2p-59, -0x1.64a289bf5d84fp-59, \
  0x1.36b876b368eabp-4, -0x1.f2775f279834fp-5, 0x1.640b954490e62p-8, \
  0x1.eff2bb0a89d95p-10, -0x1.1f1b7c1c4a8e6p-11, 0x1.339e5da2cdf91p-15, \
  0x1.af49e9bb56714p-18, -0x1.7ca6acf1e8af8p-20, 0x1.010a3b7059452p-24, \
  0x1.bc649d3b5ef83p-27, -0x1.15f5cf63ee98dp-29, 0x1.af35830901bbap-35, \
  0x1.26d33d517dcc7p-36, -0x1.22749391135d7p-39, 0x1.1a0f245004f21p-45, \
  0x1.274bd7962a53cp-46, -0x1.0659bd0dfbc20p-49, 0x1.973467e682604p-56, \
  0x1.05c0c32326cf5p-56, -0x1.bc6a765a33eb6p-60, 0x1.0a4b5c68ffcb8p-66
// clang-format on
#if defined(__CUDACC__)
static __constant__ double log_phi_cheb_dev[cheb_size] = {LAPLACE_LOG_PHI_CHEB};
#endif
inline constexpr double log_phi_cheb_host[cheb_size] = {LAPLACE_LOG_PHI_CHEB};
// the coefficient i of the table
LAPLACE_HD_FORCEINLINE double cheb_coef(int i) {
  KOKKOS_IF_ON_DEVICE((return log_phi_cheb_dev[i];))
  KOKKOS_IF_ON_HOST((return log_phi_cheb_host[i];))
}
// the degree of the expansion of g_k, and the index of its first coefficient, k = 1..5
LAPLACE_HD_FORCEINLINE int cheb_degree(int k) {
  constexpr int d[5] = {17, 17, 18, 19, 20};
  return d[k - 1];
}
LAPLACE_HD_FORCEINLINE int cheb_offset(int k) {
  constexpr int o[5] = {0, 90, 180, 275, 375};
  return o[k - 1];
}

}  // namespace detail_normal
// END GENERATED: tools/reference/log_phi_cheb.py

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

// d^k log Phi(t) / dt^k for t in [cheb_lo, cheb_hi): the piece of the table
// that contains t, x = 2 (t - a) / width - 1 in [-1, 1], and Clenshaw's
// recurrence b_n = c_n + 2 x b_(n+1) - b_(n+2) (b beyond the degree zero),
// the value c_0 + x b_1 - b_2
template <int K>
LAPLACE_HD double log_Phi_dk_cheb(double t) {
  const int N = cheb_degree(K);
  int p = static_cast<int>(t - cheb_lo);  // t >= cheb_lo, so the truncation is the floor
  if (p >= cheb_pieces) p = cheb_pieces - 1;
  const double x = 2.0 * (t - (cheb_lo + p * cheb_width)) / cheb_width - 1.0;
  const int base = cheb_offset(K) + p * (N + 1);
  double b1 = 0.0, b2 = 0.0;
  for (int n = N; n >= 1; --n) {
    const double b0 = 2.0 * x * b1 - b2 + cheb_coef(base + n);
    b2 = b1;
    b1 = b0;
  }
  return cheb_coef(base) + x * b1 - b2;
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
LAPLACE_HD double log_Phi_d1(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<1>(t);
  return detail_normal::log_Phi_dk_outside(1, t);
}
LAPLACE_HD double log_Phi_d2(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<2>(t);
  return detail_normal::log_Phi_dk_outside(2, t);
}
LAPLACE_HD double log_Phi_d3(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<3>(t);
  return detail_normal::log_Phi_dk_outside(3, t);
}
LAPLACE_HD double log_Phi_d4(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<4>(t);
  return detail_normal::log_Phi_dk_outside(4, t);
}
LAPLACE_HD double log_Phi_d5(double t) {
  if (t >= -1.0 && t < detail_normal::cheb_hi) return detail_normal::log_Phi_dk_cheb<5>(t);
  return detail_normal::log_Phi_dk_outside(5, t);
}

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
