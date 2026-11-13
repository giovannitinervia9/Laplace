// Owen's T function and the distribution function of the standard skew normal
// on the log scale, with its derivatives in (z, alpha) to order five.
//
// Notation. phi and Phi are the standard normal density and distribution
// function, Q(h) = Phi(-h), erfcx(x) = exp(x^2) erfc(x). For h >= 0 and real a,
//   T(h, a) = (1/(2 pi)) int_0^a exp(-h^2 (1 + s^2)/2) / (1 + s^2) ds   (Owen's T),
//   F(z; alpha) = 2 int_(-inf)^z phi(t) Phi(alpha t) dt = Phi(z) - 2 T(z, alpha)
// is the distribution function of the standard skew normal, and
// 1 - F(z; alpha) = F(-z; -alpha). Below z = -h <= 0, eta = h^2/2, beta = alpha h.
//
// The one representation. For every real alpha,
//   F(-h; alpha) = exp(-eta) G(h, alpha) / pi,  G(h, alpha) = int_alpha^inf exp(-eta b^2) / (1 + b^2) db,
// because dF/dalpha = -exp(-(1 + alpha^2) z^2/2) / (pi (1 + alpha^2)) and
// F(z; +inf) = 0 for z < 0. G is an integral of a positive function; three
// forms of it avoid cancellation and underflow (G0 = G(h, 0) = (pi/2) erfcx(h / sqrt 2)):
//   Jbar   alpha > 0: G = exp(-eta alpha^2) Jbar(h, alpha),
//          Jbar(h, a) = int_0^inf exp(-h^2 (a u + u^2/2)) / (1 + (a + u)^2) du
//                     = (1/h) int_0^inf exp(-beta y - y^2/2) r(a + y/h) dy,  r(x) = 1/(1 + x^2),
//          so that log F = -eta (1 + alpha^2) - log pi + log Jbar: the quadratic,
//          which carries the cancellation of the far lower tail, in closed form;
//   G0 - Ttilde  |beta| small: G = G0 - Ttilde(h, alpha), Ttilde = int_0^alpha
//          exp(-eta s^2)/(1 + s^2) ds, a short integral (Gauss-Legendre, 16 points);
//   2 G0 - exp(-eta a^2) Jbar(h, a)   alpha = -a < 0, |beta| larger.
// Jbar is integrated in y by fixed Gauss rules, chosen by D = sqrt(h^2 + beta^2),
// the distance from y = 0 of the poles y = -beta +- i h of r (measured against
// mpmath, also for the weighted moments below, before this file was written):
//   D >= 2, beta >= 4   Gauss-Laguerre in t = beta y + y^2/2 (weight e^-t), 20 nodes;
//   D >= 2, beta < 4    half-range Gauss-Hermite in y (weight e^(-y^2/2) on y > 0), 28 nodes;
//   D < 2               16-point Gauss-Legendre on pieces of y graded away from
//                       the poles (length <= y + D) and limited in the increase of t (<= 20),
//                       up to t = 60.
// Regions (z = -h <= 0): h < 1: Jbar when beta >= 0.8, elsewhere the elementary
// series below; h >= 1: G0 - Ttilde for |beta| <= 0.5, Jbar for beta > 0.5,
// 2 G0 - ... for beta < -0.5. For z > 0, F = 1 - S with S = F(-z; -alpha) when
// S <= 1/2, else F = erf(z/sqrt 2) + F(-z; alpha) (both terms positive; only
// for z < 0.68 and alpha > 0); F(-h; alpha) for h < 1 and small beta from Owen's
// T: G0 - Ttilde (alpha <= 1), 2 T(a h, 1/a) - Q(a h) erf(h / sqrt 2) (alpha > 1,
// which loses at most a factor 3), Q(h) + 2 T(h, a) (alpha = -a < 0), the last
// through 1 - F = erf(h / sqrt 2) + F(-h; a) near F = 1.
//
// Derivatives of log F (one public function per order, the components of the
// order in mixed_partials<K>, c[j] = d^K / dz^(K-j) dalpha^j). Every derivative of F
// is elementary; the derivatives of log F are not, and their naive composition
// from those of F cancels like t + R(t) in the derivatives of log Phi. So:
//   Jbar: at (h + d, a + e) and the same y the integrand of Jbar is the one at
//     (h, a) times exp(-y (a d + h e + d e)) r(x + e + w(d)) / (1 + d/h),
//     x = a + y/h, w(d) = -(y/h)(d/h)/(1 + d/h), r(x + s) = sum_k c_k s^k,
//     c_k = (-1)^(k+1) Im (x + i)^-(k+1); its Taylor coefficients in (d, e)
//     are sums of the moments sum_nodes w y^m c_k (y/h)^l with node-free
//     polynomials, assembled once per call; then the logarithm of the
//     bivariate series (the recursion g dl = dg) and the closed-form quadratic.
//   G0 - Ttilde: log G0 from the cumulants of exp(-h s - s^2/2) on s > 0 (the
//     downward ratio recursion of normal.hpp), log(1 - Ttilde/G0) in h from the
//     moments of Ttilde, and the alpha direction from the equation
//     d log G / dalpha = -exp(-eta alpha^2) / ((1 + alpha^2) G), solved column by
//     column in alpha (alpha_ode).
//   2 G0 - ...: log(2 G0) + log(1 - eps), eps = exp(-eta a^2) Jbar(h, a) / (2 G0) <= 1/2.
//   elsewhere: d log F/dz = f/F = exp(log f - log F), f = 2 phi(z) Phi(alpha z),
//     solved order by order in z (the nested form of normal.hpp's recursion for
//     log Phi), and d log F/dalpha = -g/F, g = exp(-(1 + alpha^2) z^2/2) / (pi (1 + alpha^2)),
//     column by column in alpha; the factors f/F and g/F are kept apart as
//     exp(-S) m (struct scale) because they can lie below the double range
//     while the derivatives, multiplied by powers of alpha and z, do not.
//   order 1 alone: f/F and -g/F with the Gaussian factors cancelled in closed form.
// Near their zeros the derivatives of order 3 to 5 are conditioned by the value
// of F: one ulp of F moves them by up to 10^4 of their own ulps (measured with
// mpmath), which bounds what any double computation of F allows there.
//
// Owen's T (h, a >= 0; T is even in h and odd in a):
//   a <= 1, a h <= 2   T = exp(-eta)/(2 pi) Ttilde(h, a);
//   a <= 1, a h > 2    T = (Q(h) - F(-h; a)) / 2, F(-h; a) small against Q(h);
//   a > 1              T = (Q(h) erf(a h / sqrt 2) + F(-a h; 1/a)) / 2, a sum of
//                      positive terms (from Owen's identity
//                      T(h, a) + T(a h, 1/a) = (Q(h) + Q(a h))/2 - Q(h) Q(a h)
//                      and T(x, b) = (Q(x) - F(-x; b))/2).
// Everything here is Laplace's own; no code was taken from elsewhere (Boost's
// owens_t, Patefield and Tandy's algorithm, is measured against it in
// benchmarks/math_owen.cpp; normal.hpp's ratio recursion is repeated in
// log_g0_series). The Gauss rules were computed by the Golub-Welsch algorithm
// at 600 bits with mpmath.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta_deriv.hpp>  // mixed_partials
#include <laplace/math/normal.hpp>

// On the device the large helpers are subroutines, not inlined into the kernel
#if defined(__CUDACC__)
#define LAPLACE_OWEN_NOINLINE __noinline__
#else
#define LAPLACE_OWEN_NOINLINE
#endif

namespace laplace::math::own {

namespace detail_owen {

namespace dn = detail_normal;

constexpr double pi = 3.14159265358979323846;
constexpr double inv_pi = 0.31830988618379067154;
constexpr double inv_2pi = 0.15915494309189533577;
constexpr double log_pi = 1.14472988584940017414;
constexpr double log2 = 0.69314718055994530942;
constexpr double half_pi = 1.57079632679489661923;
constexpr double sqrt_half = 0.70710678118654752440;
constexpr double inv_sqrt_2pi = 0.39894228040143267794;

// Gauss rules as pairs (node, weight): Gauss-Legendre on [0, 1] (16 nodes),
// half-range Gauss-Hermite for the weight exp(-y^2/2) on y > 0 (20, 24, 28),
// Gauss-Laguerre for the weight exp(-t) on t > 0 (16, 20, 24)
// clang-format off
#define LAPLACE_OWEN_RULE_DATA \
  /* gl16 */ \
  0.005299532504175033, 0.013576229705877048, \
  0.02771248846338371, 0.031126761969323947, \
  0.06718439880608412, 0.04757925584124639, \
  0.12229779582249849, 0.06231448562776694, \
  0.19106187779867811, 0.07479799440828837, \
  0.2709916111713863, 0.08457825969750127, \
  0.35919822461037054, 0.09130170752246179, \
  0.4524937450811813, 0.09472530522753425, \
  0.5475062549188188, 0.09472530522753425, \
  0.6408017753896295, 0.09130170752246179, \
  0.7290083888286137, 0.08457825969750127, \
  0.8089381222013219, 0.07479799440828837, \
  0.8777022041775016, 0.06231448562776694, \
  0.9328156011939158, 0.04757925584124639, \
  0.9722875115366163, 0.031126761969323947, \
  0.994700467495825, 0.013576229705877048, \
  /* hh20 */ \
  0.02019427628166043, 0.051714850719782705, \
  0.10554459719217753, 0.1178287871343394, \
  0.2557768753886745, 0.1752976804057703, \
  0.4658891200442014, 0.21358666615188174, \
  0.729805676867617, 0.22120346058919488, \
  1.0412224534821048, 0.19373627661335285, \
  1.39423578060332, 0.14075561130980363, \
  1.783732032644749, 0.08277925237781589, \
  2.205578685856521, 0.03837578863269532, \
  2.6566877440359065, 0.013639713146338715, \
  3.135020373680754, 0.003608670089808719, \
  3.639588658035118, 0.0006879427698948457, \
  4.170501295205704, 9.100669589381178e-05, \
  4.729103819723427, 7.979546568706326e-06, \
  5.3182914277817686, 4.371102928953932e-07, \
  5.943151320610031, 1.3797648831834631e-08, \
  6.61230944490259, 2.2289232112128777e-10, \
  7.341043566665424, 1.5216170674134436e-12, \
  8.159991275837488, 3.068650455642283e-15, \
  9.150751427113702, 7.511203686612847e-19, \
  /* hh24 */ \
  0.01546572428284264, 0.03963177373583972, \
  0.08102467456890859, 0.09092634417200773, \
  0.197149559777437, 0.13783887055143054, \
  0.36100608687786123, 0.1747888048604866, \
  0.5689668040943318, 0.1945140755122695, \
  0.8170238621296854, 0.19105038808593336, \
  1.1011405125709275, 0.16408047185145377, \
  1.4175178831852557, 0.12132378686704731, \
  1.7627703893469242, 0.07585085352846821, \
  2.134023433270853, 0.03933583490926128, \
  2.5289558815223177, 0.016591451848373934, \
  2.945810315977092, 0.0055775589625401125, \
  3.3833908005913442, 0.0014629037523407429, \
  3.8410642919629963, 0.00029253770881227086, \
  4.318779879592194, 4.346182159496116e-05, \
  4.817121166723755, 4.6549281472489504e-06, \
  5.337413129910965, 3.4655168831888866e-07, \
  5.88191971950335, 1.7129164294995397e-08, \
  6.454202369593304, 5.291582829662128e-10, \
  7.059790264811719, 9.396968158076973e-12, \
  7.707527148649141, 8.476746787737468e-14, \
  8.412628265821523, 3.1786252203900723e-16, \
  9.205170595246297, 3.401523845411264e-19, \
  10.164717875914276, 4.065997163351631e-23, \
  /* hh28 */ \
  0.012331359648925731, 0.031612268920266307, \
  0.06469899222985864, 0.07282280540967767, \
  0.1578244224103699, 0.11158482890342654, \
  0.28998139949612806, 0.14473115655685095, \
  0.45888293852464823, 0.16778588538932115, \
  0.6618969584016275, 0.17603591018451695, \
  0.8962430623650428, 0.16669422122702812, \
  1.1591608532681899, 0.1411350545765108, \
  1.4480392683143843, 0.10551468601459593, \
  1.7605057429918674, 0.06869222006264245, \
  2.0944805511301396, 0.03838059794125329, \
  2.4482047244404392, 0.018134420916990263, \
  2.820250458006877, 0.007137795431540411, \
  3.2095220428904137, 0.0023046650779993327, \
  3.6152540858716864, 0.000600691917285774, \
  4.037012701536728, 0.00012422989553636268, \
  4.474704875919474, 2.0004315749603787e-05, \
  4.928601605719019, 2.4551078019878406e-06, \
  5.399382129122877, 2.2401785472094218e-07, \
  5.888210400012084, 1.4751893686665583e-08, \
  6.396862694282391, 6.758112522869629e-10, \
  6.9279409591954595, 2.055254812203856e-11, \
  7.4852402838604775, 3.899067513744234e-13, \
  8.074417989404077, 4.231607478556004e-15, \
  8.704320867847056, 2.3105076398282748e-17, \
  9.38998006028488, 5.1511453465445054e-20, \
  10.16090837046425, 3.166810932577928e-23, \
  11.094905187270315, 2.0112083237557898e-27, \
  /* lag16 */ \
  0.08764941047892784, 0.206151714957801, \
  0.46269632891508083, 0.3310578549508842, \
  1.141057774831227, 0.26579577764421414, \
  2.1292836450983805, 0.13629693429637754, \
  3.4370866338932067, 0.04732892869412522, \
  5.078018614549768, 0.011299900080339454, \
  7.070338535048234, 0.0018490709435263109, \
  9.438314336391938, 0.00020427191530827845, \
  12.21422336886616, 1.4844586873981299e-05, \
  15.441527368781617, 6.828319330871199e-07, \
  19.180156856753136, 1.8810248410796733e-08, \
  23.515905693991908, 2.8623502429738814e-10, \
  28.57872974288214, 2.1270790332241028e-12, \
  34.58339870228662, 6.297967002517868e-15, \
  41.94045264768833, 5.050473700035513e-18, \
  51.70116033954332, 4.161462370372855e-22, \
  /* lag20 */ \
  0.07053988969198875, 0.16874680185111388, \
  0.37212681800161146, 0.2912543620060683, \
  0.9165821024832735, 0.2666861028670013, \
  1.707306531028344, 0.16600245326950683, \
  2.749199255309432, 0.07482606466879237, \
  4.048925313850887, 0.02496441730928322, \
  5.6151749708616165, 0.006202550844572237, \
  7.459017453671064, 0.0011449623864769082, \
  9.594392869581096, 0.00015574177302781197, \
  12.038802546964316, 1.5401440865224915e-05, \
  14.81429344263074, 1.0864863665179824e-06, \
  17.948895520519375, 5.3301209095567146e-08, \
  21.47878824028501, 1.757981179050582e-09, \
  25.451702793186904, 3.7255024025123206e-11, \
  29.93255463170061, 4.767529251578191e-13, \
  35.013434240479, 3.3728442433624386e-15, \
  40.83305705672857, 1.1550143395003989e-17, \
  47.6199940473465, 1.5395221405823435e-20, \
  55.810795750063896, 5.286442725569158e-24, \
  66.52441652561575, 1.6564566124990233e-28, \
  /* lag24 */ \
  0.05901985218150798, 0.14281197333478185, \
  0.31123914619848375, 0.2587741075174239, \
  0.7660969055459367, 0.2588067072728698, \
  1.4255975908036131, 0.18332268897777804, \
  2.2925620586321904, 0.0981662726299189, \
  3.3707742642089977, 0.040732478151408645, \
  4.665083703467171, 0.013226019405120156, \
  6.1815351187367655, 0.0033693490584783036, \
  7.927539247172152, 0.0006721625640935479, \
  9.912098015077706, 0.00010446121465927518, \
  12.146102711729766, 1.2544721977993332e-05, \
  14.642732289596674, 1.15131581273728e-06, \
  17.417992646508978, 7.96081295913363e-08, \
  20.491460082616424, 4.0728589875499996e-09, \
  23.887329848169735, 1.507008226292585e-10, \
  27.635937174332717, 3.917736515058451e-12, \
  31.776041352374722, 6.894181052958085e-14, \
  36.35840580165162, 7.819800382459448e-16, \
  41.45172048487077, 5.3501888130100375e-18, \
  47.153106445156325, 2.0105174645555034e-20, \
  53.60857454469507, 3.6057658645529593e-23, \
  61.05853144721876, 2.4518188458784027e-26, \
  69.96224003510503, 4.088301593680658e-30, \
  81.49827923394889, 5.575345788328357e-35
// clang-format on
constexpr int n_rule_pairs = 148;
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double rule_dev[2 * n_rule_pairs] = {LAPLACE_OWEN_RULE_DATA};
#endif
inline constexpr double rule_host[2 * n_rule_pairs] = {LAPLACE_OWEN_RULE_DATA};
LAPLACE_HD_FORCEINLINE double rule(int i) {
  KOKKOS_IF_ON_DEVICE((return rule_dev[i];))
  KOKKOS_IF_ON_HOST((return rule_host[i];))
}
// offsets (in pairs)
constexpr int off_gl16 = 0;
LAPLACE_HD_FORCEINLINE int off_hh(int n) { return n == 20 ? 16 : n == 24 ? 36 : 60; }
LAPLACE_HD_FORCEINLINE int off_lag(int n) { return n == 16 ? 88 : n == 20 ? 104 : 124; }

// The constants of the algorithm (the bench varies them; the public functions
// use the defaults).
struct params {
  double h_n = 1.0;     // z = -h <= 0: below h_n the elementary series, unless alpha h >= beta_a (Jbar) ...
  double beta_a = 0.8;
  double beta_t = 0.5;  // ... and above it the series of G0 - Ttilde for |alpha| h <= beta_t, Jbar beyond
  double d_h = 2.0;     // Jbar: Gauss-Hermite / Gauss-Laguerre for D = sqrt(h^2 + (alpha h)^2) >= d_h,
                        // Gauss-Legendre pieces below
  double beta_l = 4.0;  // Gauss-Laguerre in t for a h >= beta_l, half-range Gauss-Hermite below
  int n_h = 28;         // nodes of the Hermite rule: 20, 24 or 28
  int n_l = 20;         // nodes of the Laguerre rule: 16, 20 or 24
  double gl_cr = 1.0;   // Gauss-Legendre pieces: length <= gl_cr (y + D) (poles at distance D from y = 0) ...
  double gl_ce = 20.0;  // ... and an increase of the exponent t <= gl_ce
  double t_cut = 60.0;  // ... up to t = t_cut
  double beta_d = 2.0;  // Owen's T with a <= 1: direct quadrature for a h <= beta_d
  // the same for the value alone (order 0), which needs fewer nodes
  double d_h0 = 2.0, beta_l0 = 4.0, gl_cr0 = 1.0, gl_ce0 = 20.0, t_cut0 = 60.0;
  int n_h0 = 28, n_l0 = 20;
};

// ---------------------------------------------------------------- bivariate Taylor series
// Arrays of Taylor coefficients c(i, j) = d^(i+j) f / dx^i dy^j / (i! j!) at a
// point, total order i + j <= K, stored at ix(i, j).

LAPLACE_HD_FORCEINLINE constexpr int ix(int i, int j) { return (i + j) * (i + j + 1) / 2 + j; }
template <int K>
inline constexpr int nc = (K + 1) * (K + 2) / 2;

// e = exp(f), by the recursion of d e = e d f; e0, when given, is exp(f(0, 0))
template <int K>
LAPLACE_HD void ser_exp(const double* f, double* e, double e0 = -1.0) {
  e[0] = e0 >= 0.0 ? e0 : math::exp(f[0]);
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double s = 0.0;
      if (i >= 1) {
        for (int p = 1; p <= i; ++p)
          for (int q = 0; q <= j; ++q) s += p * f[ix(p, q)] * e[ix(i - p, j - q)];
        e[ix(i, j)] = s / i;
      } else {
        for (int q = 1; q <= j; ++q) s += q * f[ix(0, q)] * e[ix(0, j - q)];
        e[ix(0, j)] = s / j;
      }
    }
}

// l = log g for g(0, 0) > 0, by the recursion of g d l = d g; l(0, 0) = log g(0, 0)
template <int K>
LAPLACE_HD void ser_log(const double* g, double* l) {
  const double r = 1.0 / g[0];
  l[0] = math::log(g[0]);
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      if (i >= 1) {
        double s = i * g[ix(i, j)] * r;
        for (int p = 1; p <= i; ++p)
          for (int q = 0; q <= j; ++q)
            if (p != i || q != j) s -= p * l[ix(p, q)] * g[ix(i - p, j - q)] * r;
        l[ix(i, j)] = s / i;
      } else {
        double s = j * g[ix(0, j)] * r;
        for (int q = 1; q < j; ++q) s -= q * l[ix(0, q)] * g[ix(0, j - q)] * r;
        l[ix(0, j)] = s / j;
      }
    }
}

LAPLACE_HD_FORCEINLINE double binom(int m, int r) {
  constexpr double c[6][6] = {{1, 0, 0, 0, 0, 0},  {1, 1, 0, 0, 0, 0},   {1, 2, 1, 0, 0, 0},
                              {1, 3, 3, 1, 0, 0},  {1, 4, 6, 4, 1, 0},   {1, 5, 10, 10, 5, 1}};
  return c[m][r];
}

// lambda_n(x), n = 1..K: the Taylor coefficients of log(1 + (x + e)^2) in e,
// lambda_n = ((-1)^(n-1) / n) 2 Re (x + i)^-n
template <int K>
LAPLACE_HD_FORCEINLINE void log1px2_coeffs(double x, double* lam) {
  const double r = 1.0 / (1.0 + x * x);
  const double zr = x * r, zi = -r;  // 1/(x + i)
  double pr = zr, pim = zi;
  lam[1] = 2.0 * pr;
  for (int n = 2; n <= K; ++n) {
    const double nr = pr * zr - pim * zi;
    pim = pr * zi + pim * zr;
    pr = nr;
    lam[n] = (n % 2 ? 2.0 : -2.0) * pr / n;
  }
}

// ---------------------------------------------------------------- Jbar and its derivatives

// c = a b, truncated at total order K
template <int K>
LAPLACE_HD_FORCEINLINE void ser_mul(const double* a, const double* b, double* c) {
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double s = 0.0;
      for (int p = 0; p <= i; ++p)
        for (int q = 0; q <= j; ++q) s += a[ix(p, q)] * b[ix(i - p, j - q)];
      c[ix(i, j)] = s;
    }
}

// One node of the quadrature of Jbar in the variable y = h u:
//   Jbar(h, a) = (1/h) int_0^inf exp(-beta y - y^2/2) r(a + y/h) dy,  r(x) = 1/(1 + x^2).
// At (h + d, a + e) and the same y the integrand is
//   exp(-y (a d + h e + d e)) r(x + e + w(d)) / (1 + d/h) times exp(-beta y - y^2/2) / h,
// x = a + u, u = y/h, w(d) = y/(h + d) - u = -u (d/h) / (1 + d/h), and
//   r(x + s) = sum_k c_k s^k,  c_k = (-1)^(k+1) Im (x + i)^-(k+1).
// The node adds om y^m c_k u^l to M[m][k][l] (m + k <= K, l <= k), om the weight
// of the rule times exp(-beta y - y^2/2) and the Jacobian.
template <int K>
LAPLACE_HD_FORCEINLINE void jbar_node(double y, double om, double h, double a, double (&M)[K + 1][K + 1][K + 1]) {
  const double u = y / h, x = a + u;
  const double r = 1.0 / (1.0 + x * x);
  if constexpr (K == 0) {
    M[0][0][0] += om * r;
  } else {
    const double zr = x * r, zi = -r;  // 1/(x + i)
    double c[K + 1], yp[K + 1], up[K + 1];
    double pr = zr, pim = zi;
    for (int k = 0; k <= K; ++k) {
      c[k] = (k % 2 ? pim : -pim);
      const double nr = pr * zr - pim * zi;
      pim = pr * zi + pim * zr;
      pr = nr;
    }
    yp[0] = up[0] = 1.0;
    for (int k = 1; k <= K; ++k) {
      yp[k] = yp[k - 1] * y;
      up[k] = up[k - 1] * u;
    }
    for (int m = 0; m <= K; ++m) {
      const double am = om * yp[m];
      for (int k = 0; k <= K - m; ++k) {
        const double b = am * c[k];
        for (int l = 0; l <= k; ++l) M[m][k][l] += b * up[l];
      }
    }
  }
}

// Taylor coefficients of log Jbar(h + d, a + e), total order <= K; a >= 0,
// h > 0, h^2 + (a h)^2 not small (see the regions). jv, when given, receives Jbar itself.
//   Jbar(h + d, a + e) = (1/h) sum_(m, k, l) (-1)^(m + l) / m! C(k, l) M[m][k][l]
//                        (a d + h e + d e)^m e^(k - l) (d/h)^l (1 + d/h)^-(l + 1).
template <int K>
LAPLACE_OWEN_NOINLINE LAPLACE_HD void log_jbar(double h, double a, const params& p, double* L, double* jv = nullptr) {
  double M[K + 1][K + 1][K + 1];
  for (int m = 0; m <= K; ++m)
    for (int k = 0; k <= K; ++k)
      for (int l = 0; l <= K; ++l) M[m][k][l] = 0.0;
  const double beta = a * h, D2 = h * h + beta * beta;
  const double d_h = K ? p.d_h : p.d_h0, beta_l = K ? p.beta_l : p.beta_l0, gl_cr = K ? p.gl_cr : p.gl_cr0;
  const double gl_ce = K ? p.gl_ce : p.gl_ce0, t_cut = K ? p.t_cut : p.t_cut0;
  const int n_h = K ? p.n_h : p.n_h0, n_l = K ? p.n_l : p.n_l0;
  if (D2 < d_h * d_h) {
    // 16-point Gauss-Legendre on pieces of y in [0, y_max], graded away from
    // the poles at y = -beta +- i h and limited in the increase of t = beta y + y^2/2
    const double D = Kokkos::sqrt(D2);
    const double ymax = 2.0 * t_cut / (Kokkos::sqrt(beta * beta + 2.0 * t_cut) + beta);
    double lo = 0.0;
    for (int piece = 0; piece < 64 && lo < ymax; ++piece) {
      const double g = beta + lo;
      const double d1 = gl_cr * (lo + D), d2 = 2.0 * gl_ce / (g + Kokkos::sqrt(g * g + 2.0 * gl_ce));
      const double hi = Kokkos::fmin(ymax, lo + Kokkos::fmin(d1, d2));
      const double len = hi - lo;
      for (int k = 0; k < 16; ++k) {
        const double y = lo + len * rule(2 * (off_gl16 + k));
        jbar_node<K>(y, len * rule(2 * (off_gl16 + k) + 1) * math::exp(-y * (beta + 0.5 * y)), h, a, M);
      }
      lo = hi;
    }
  } else if (beta >= beta_l) {
    // Gauss-Laguerre in t = beta y + y^2/2 (weight e^-t), dy/dt = 1/(beta + y)
    const int o = off_lag(n_l);
    for (int k = 0; k < n_l; ++k) {
      const double t = rule(2 * (o + k)), wt = rule(2 * (o + k) + 1);
      const double s = Kokkos::sqrt(beta * beta + 2.0 * t);  // beta + y
      jbar_node<K>(2.0 * t / (s + beta), wt / s, h, a, M);
    }
  } else {
    // half-range Gauss-Hermite in y (weight e^(-y^2/2))
    const int o = off_hh(n_h);
    for (int k = 0; k < n_h; ++k) {
      const double y = rule(2 * (o + k)), wt = rule(2 * (o + k) + 1);
      jbar_node<K>(y, wt * math::exp(-beta * y), h, a, M);
    }
  }
  const double ih = 1.0 / h;
  double T[nc<K>];
  if constexpr (K == 0) {
    T[0] = M[0][0][0] * ih;
  } else {
    // V_l(d) = (d/h)^l (1 + d/h)^-(l + 1): coefficient of d^i is (-1)^(i - l) C(i, l) h^-i
    double ihp[K + 1];
    ihp[0] = 1.0;
    for (int i = 1; i <= K; ++i) ihp[i] = ihp[i - 1] * ih;
    double P[nc<K>], Pn[nc<K>], R[nc<K>], RP[nc<K>];
    for (int k = 0; k < nc<K>; ++k) {
      P[k] = 0.0;
      T[k] = 0.0;
    }
    P[0] = 1.0;  // (a d + h e + d e)^m, from m = 0
    double inv_mfact = 1.0;
    for (int m = 0; m <= K; ++m) {
      if (m > 0) {
        // P <- P (a d + h e + d e)
        for (int n = K; n >= 0; --n)
          for (int j = 0; j <= n; ++j) {
            const int i = n - j;
            double v = 0.0;
            if (i >= 1) v += a * P[ix(i - 1, j)];
            if (j >= 1) v += h * P[ix(i, j - 1)];
            if (i >= 1 && j >= 1) v += P[ix(i - 1, j - 1)];
            Pn[ix(i, j)] = v;
          }
        for (int k = 0; k < nc<K>; ++k) P[k] = Pn[k];
        inv_mfact /= m;
      }
      // R_m(d, e) = sum_(k, l) (-1)^l C(k, l) M[m][k][l] e^(k - l) V_l(d), total order <= K - m
      for (int k = 0; k < nc<K>; ++k) R[k] = 0.0;
      for (int n = 0; n <= K - m; ++n)
        for (int j = 0; j <= n; ++j) {
          const int i = n - j;
          double v = 0.0;
          for (int l = 0; l <= i; ++l) {
            const double vl = ((i - l) % 2 ? -1.0 : 1.0) * binom(i, l) * ihp[i];
            v += (l % 2 ? -1.0 : 1.0) * binom(j + l, l) * M[m][j + l][l] * vl;
          }
          R[ix(i, j)] = v;
        }
      ser_mul<K>(P, R, RP);
      const double cm = (m % 2 ? -inv_mfact : inv_mfact) * ih;
      for (int k = 0; k < nc<K>; ++k) T[k] += cm * RP[k];
    }
  }
  if (jv) *jv = T[0];
  ser_log<K>(T, L);
}

// int_0^a exp(-h^2 s^2/2) / (1 + s^2) ds for 0 <= a <= 1 and a h <= 2 (16-point Gauss-Legendre)
LAPLACE_HD double t_direct(double h, double a) {
  double s = 0.0;
  for (int k = 0; k < 16; ++k) {
    const double x = a * rule(2 * (off_gl16 + k)), hx = h * x;
    s += rule(2 * (off_gl16 + k) + 1) * math::exp(-0.5 * hx * hx) / (1.0 + x * x);
  }
  return s * a;
}

// G0(h) = int_0^inf exp(-h^2 s^2/2) / (1 + s^2) ds = (pi/2) erfcx(h / sqrt 2), so that Q(h) = exp(-eta) G0 / pi
LAPLACE_HD_FORCEINLINE double g0(double h) { return half_pi * own::erfcx(h * sqrt_half); }

// s + e = (a h)^2 / 2 with the product a h carried exactly
LAPLACE_HD_FORCEINLINE void half_square_prod(double a, double h, double& s, double& e) {
  double q, qe;
  dn::two_prod(a, h, q, qe);
  dn::half_square(q, s, e);
  e += q * qe;
}

// exp(-(s1 + e1 + s2 + e2)) for an exponent given as two unevaluated sums
// (|e1| <= ulp(s1), |e2| <= ulp(s2)), without the rounding of s1 + s2
LAPLACE_HD_FORCEINLINE double exp_neg2(double s1, double e1, double s2, double e2) {
  const double S = s1 + s2, bv = S - s1;
  const double err = ((s1 - (S - bv)) + (s2 - bv)) + e1 + e2;
  return math::exp(-S) * (1.0 - err);
}

// int_0^alpha exp(-h^2 s^2/2) / (1 + s^2) ds for |alpha| <= 1 and |alpha| h <= 2, any sign of alpha
LAPLACE_HD_FORCEINLINE double t_signed(double h, double alpha) {
  return alpha < 0.0 ? -t_direct(h, -alpha) : t_direct(h, alpha);
}

// The regions of (h, alpha), z = -h <= 0, beta = alpha h:
//   0  elementary: h < h_n and (alpha <= 0 or beta < beta_a);
//   1  G = G0 - Ttilde: h >= h_n and |beta| <= beta_t;
//   2  Jbar: alpha > 0 and (beta >= beta_a when h < h_n, beta > beta_t otherwise);
//   3  2 G0 - exp(-eta a^2) Jbar(h, a): alpha < 0, h >= h_n, |beta| > beta_t.
// Here G(h, alpha) = int_alpha^inf exp(-h^2 b^2/2) / (1 + b^2) db = pi exp(eta) F(-h; alpha),
// G0 = G(h, 0) and Ttilde(h, alpha) = int_0^alpha exp(-h^2 s^2/2) / (1 + s^2) ds.
LAPLACE_HD_FORCEINLINE int region(double h, double alpha, const params& p) {
  const double beta = alpha * h;
  if (h < p.h_n) return (alpha > 0.0 && beta >= p.beta_a) ? 2 : 0;
  if (Kokkos::fabs(beta) <= p.beta_t) return 1;
  return alpha > 0.0 ? 2 : 3;
}

// F(-h; alpha) and its logarithm, h >= 0: the value with its relative accuracy
// down to the underflow, the logarithm also below it
struct val_log {
  double v, l;
};

// F(-h; alpha) for h >= 0, alpha >= 0
LAPLACE_HD val_log lower_pos(double h, double alpha, const params& p) {
  double s1, e1;
  dn::half_square(h, s1, e1);
  const int reg = region(h, alpha, p);
  if (reg == 2) {
    double L[1], jv, s2, e2;
    log_jbar<0>(h, alpha, p, L, &jv);
    half_square_prod(alpha, h, s2, e2);
    return {exp_neg2(s1, e1, s2, e2) * jv * inv_pi, -(s1 + s2) + (L[0] - log_pi - (e1 + e2))};
  }
  if (reg == 1 || alpha <= 1.0) {
    const double g = g0(h) - t_direct(h, alpha);
    return {dn::exp_minus_half_square(h) * g * inv_pi, -s1 + (math::log(g) - log_pi - e1)};
  }
  // 2 T(a h, 1/a) - Q(a h) erf(h / sqrt 2)
  const double beta = alpha * h, b = 1.0 / alpha;
  const double E = dn::exp_minus_half_square(beta) * t_direct(beta, b) * inv_pi - own::Phi(-beta) * Kokkos::erf(h * sqrt_half);
  return {E, math::log(E)};
}

// T(h, a) for h >= 0, a > 0
LAPLACE_HD double owens_t_pos(double h, double a, const params& p) {
  if (a <= 1.0) {
    if (a * h <= p.beta_d) return inv_2pi * dn::exp_minus_half_square(h) * t_direct(h, a);
    return 0.5 * (own::Phi(-h) - lower_pos(h, a, p).v);
  }
  if (!(a < detail::pos_inf())) return 0.5 * own::Phi(-h);
  const double beta = a * h;
  return 0.5 * (own::Phi(-h) * Kokkos::erf(beta * sqrt_half) + lower_pos(beta, 1.0 / a, p).v);
}

// F(-h; alpha) for h >= 0
LAPLACE_HD val_log lower(double h, double alpha, const params& p) {
  if (alpha >= 0.0) return lower_pos(h, alpha, p);
  double s1, e1;
  dn::half_square(h, s1, e1);
  const double a = -alpha;
  const int reg = region(h, alpha, p);
  if (reg == 3) {
    // exp(-eta) (2 G0 - exp(-eta a^2) Jbar(h, a)) / pi
    double L[1], jv, s2, e2;
    log_jbar<0>(h, a, p, L, &jv);
    half_square_prod(a, h, s2, e2);
    const double g = 2.0 * g0(h) - exp_neg2(s2, e2, 0.0, 0.0) * jv;
    return {dn::exp_minus_half_square(h) * g * inv_pi, -s1 + (math::log(g) - log_pi - e1)};
  }
  if (reg == 1) {
    const double g = g0(h) + t_direct(h, a);
    return {dn::exp_minus_half_square(h) * g * inv_pi, -s1 + (math::log(g) - log_pi - e1)};
  }
  // F = Q(h) + 2 T(h, a); near 1 through 1 - F = erf(h / sqrt 2) + F(-h; a)
  const double R = Kokkos::erf(h * sqrt_half) + lower_pos(h, a, p).v;
  if (R <= 0.5) return {1.0 - R, Kokkos::log1p(-R)};
  const double F = own::Phi(-h) + 2.0 * owens_t_pos(h, a, p);
  return {F, math::log(F)};
}

LAPLACE_HD double owens_t(double h, double a, const params& p) {
  if (h != h || a != a) return detail::quiet_nan();
  const double sign = a < 0.0 ? -1.0 : 1.0;
  h = Kokkos::fabs(h);
  a = Kokkos::fabs(a);
  if (a == 0.0 || !(h < detail::pos_inf())) return sign * 0.0;
  return sign * owens_t_pos(h, a, p);
}

LAPLACE_HD double log_sn_cdf(double z, double alpha, const params& p) {
  if (z != z || alpha != alpha) return detail::quiet_nan();
  if (!(z > -detail::pos_inf())) return -detail::pos_inf();
  if (!(z < detail::pos_inf())) return 0.0;
  if (z <= 0.0) return lower(-z, alpha, p).l;
  const double S = lower(z, -alpha, p).v;
  if (S <= 0.5) return Kokkos::log1p(-S);
  return math::log(Kokkos::erf(z * sqrt_half) + lower(z, alpha, p).v);
}

// ---------------------------------------------------------------- series of log F

// A positive factor exp(-S) m, kept apart so that it may lie below the double
// range while the quantities it multiplies are large (S = 0, m = 1: none).
struct scale {
  double S = 0.0, m = 1.0;
};

// v exp(-S) m, also when exp(-S) is below the double range: exp(-S) = 2^-k exp(r),
// r = k ln 2 - S with ln 2 in two parts (as in fdlibm's exp), and the power of
// two applied in two steps so that a subnormal result is rounded once
LAPLACE_HD_FORCEINLINE double apply_scale(double v, const scale& sc) {
  if (sc.S < 700.0) return v * sc.m * math::exp(-sc.S);
  if (v == 0.0 || sc.S > 1.0e5) return v * 0.0;
  constexpr double ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
  const double k = Kokkos::round(sc.S * 1.44269504088896338700);
  const double r = (k * ln2_hi - sc.S) + k * ln2_lo;
  const double y = v * sc.m * math::exp(r);
  int kk = static_cast<int>(k);
  if (kk > 2400) return y * 0.0;
  double t = y;
  while (kk > 1000) {
    t *= detail::pow2i(-1000);
    kk -= 1000;
  }
  return t * detail::pow2i(-kk);
}

// The columns j >= 1 of the Taylor coefficients of y(d, e) from the
// equation dy/de = -c0 exp(f - (y - y(0, 0))), f(0, 0) = 0, given the column
// j = 0 (as pure(i) sp, i >= 1). Q receives the columns j >= 1 divided by c0.
// Filled column by column: the coefficients of column j of the exponential
// need those of f - (y - y(0, 0)) up to column j only.
template <int K>
LAPLACE_HD void alpha_ode(const double* f, const double* pure, double sp, double c0, double* Q) {
  double e[nc<K>], g[nc<K>];
  for (int k = 0; k < nc<K>; ++k) e[k] = g[k] = 0.0;
  for (int i = 1; i <= K; ++i) g[ix(i, 0)] = f[ix(i, 0)] - sp * pure[ix(i, 0)];
  e[0] = 1.0;
  for (int j = 0; j < K; ++j) {
    for (int i = 0; i <= K - 1 - j; ++i) {
      if (i == 0 && j == 0) continue;
      double s = 0.0;
      if (i >= 1) {
        for (int p = 1; p <= i; ++p)
          for (int q = 0; q <= j; ++q) s += p * g[ix(p, q)] * e[ix(i - p, j - q)];
        e[ix(i, j)] = s / i;
      } else {
        for (int q = 1; q <= j; ++q) s += q * g[ix(0, q)] * e[ix(0, j - q)];
        e[ix(0, j)] = s / j;
      }
    }
    for (int i = 0; i <= K - 1 - j; ++i) {
      Q[ix(i, j + 1)] = -e[ix(i, j)] / (j + 1);
      g[ix(i, j + 1)] = f[ix(i, j + 1)] - c0 * Q[ix(i, j + 1)];
    }
  }
}

// Taylor coefficients of log F(z + d; alpha + e) from F and its elementary
// derivatives: F = F(z; alpha), logF = log F.
// Pure z: d log F / dz = f / F = exp(lambda - log F), lambda = log f,
// f = 2 phi(z) Phi(alpha z), whose Taylor coefficients come from those of
// log Phi at alpha z (own::log_Phi_dk). Order by order, with
// log F(z + d) - log F = s_p sum_i P_i d^i and s_p = f / F,
//   P_i = rho_(i-1) / i,  rho = exp(Lam - Lam_0),  Lam = lambda - log F(z + d),
// the nested form of normal.hpp's recursion for log Phi, which cancels less
// than the logarithm of the series of F.
// Mixed: d log F / dalpha = -g / F, g = exp(-(1 + alpha^2) z^2/2) / (pi (1 + alpha^2)),
// likewise column by column in alpha (alpha_ode).
// s_m = g / F and s_p can lie below the double range while the derivatives of
// log F, multiplied by powers of alpha and z, do not: the coefficients are
// returned divided by s_p (j = 0) and s_m (j >= 1), factors kept apart in sc_p and sc_m.
template <int K>
LAPLACE_OWEN_NOINLINE LAPLACE_HD void naive_series(double z, double alpha, double F, double logF, double* L, scale& sc_p,
                                                   scale& sc_m) {
  double s1, e1, s2, e2;
  dn::half_square(z, s1, e1);
  half_square_prod(alpha, z, s2, e2);  // (alpha z)^2 / 2
  const double pz = alpha * z;
  const double c = 1.0 + alpha * alpha;
  const double Sm = s1 + s2, bv = Sm - s1;
  const double err_m = ((s1 - (Sm - bv)) + (s2 - bv)) + e1 + e2;
  sc_m = scale{Sm, (1.0 - err_m) * inv_pi / (c * F)};  // s_m = g / F
  // s_p = f / F = sigma_p A0 / F, sigma_p = exp(-z^2/2 - (alpha z)^2/2) for alpha z < 0, exp(-z^2/2) otherwise
  if (pz < 0.0)
    sc_p = scale{Sm, (1.0 - err_m) * own::erfcx(-pz * sqrt_half) * inv_sqrt_2pi / F};
  else
    sc_p = scale{s1, (1.0 - e1) * 2.0 * inv_sqrt_2pi * own::Phi(pz) / F};
  const double sp = apply_scale(1.0, sc_p), sm = apply_scale(1.0, sc_m);  // s_p, s_m
  // lambda: lambda_1 = -z + alpha g1, lambda_2 = -1/2 + alpha^2 g2 / 2, lambda_k = alpha^k g_k / k!
  double lam[K + 1], P[K + 1], rho[K + 1], Lam[K + 1];
  {
    double ak = 1.0, fk = 1.0;
    for (int k = 1; k < K; ++k) {
      ak *= alpha;
      fk *= k;
      lam[k] = ak * dn::log_Phi_dk(k, pz, -1.0, 300.0, 20) / fk;
    }
    if (K > 1) lam[1] -= z;
    if (K > 2) lam[2] -= 0.5;
  }
  rho[0] = 1.0;
  for (int n = 1; n <= K; ++n) {
    P[n] = rho[n - 1] / n;
    if (n == K) break;
    Lam[n] = lam[n] - sp * P[n];
    double t = 0.0;
    for (int k = 1; k <= n; ++k) t += k * Lam[k] * rho[n - k];
    rho[n] = t / n;
  }
  for (int i = 1; i <= K; ++i) L[ix(i, 0)] = P[i];
  // mixed: d log F / dalpha = -g / F = -s_m exp(gam - (log F(z + d; alpha + e) - log F)),
  // s_m = g(z, alpha) / F, gam = log g(z + d, alpha + e) - log g(z, alpha)
  double gam[nc<K>], lmb[K + 1];
  for (int k = 0; k < nc<K>; ++k) gam[k] = 0.0;
  const double Pz[3] = {0.5 * z * z, z, 0.5}, Cq[3] = {c, 2.0 * alpha, 1.0};
  for (int i = 0; i <= 2; ++i)
    for (int j = 0; j <= 2; ++j)
      if ((i || j) && i + j <= K) gam[ix(i, j)] -= Pz[i] * Cq[j];
  log1px2_coeffs<K>(alpha, lmb);
  for (int n = 1; n <= K; ++n) gam[ix(0, n)] -= lmb[n];
  alpha_ode<K>(gam, L, sp, sm, L);
  L[0] = logF;
}

// Taylor coefficients of log G0(h + d), G0(h) = int_0^inf exp(-h^2 s^2/2)/(1 + s^2) ds
// = sqrt(pi/2) M_0(h), M_0(x) = int_0^inf exp(-x s - s^2/2) ds, for h >= 1, in the
// j = 0 entries of L (the others are set to 0): d^k log M_0 / dx^k = (-1)^k kappa_k,
// kappa_k the cumulants of the density proportional to exp(-x s - s^2/2) on s > 0,
// by the downward ratio recursion of normal.hpp (detail_normal::log_Phi_dk).
template <int K>
LAPLACE_HD void log_g0_series(double h, double* L) {
  for (int k = 0; k < nc<K>; ++k) L[k] = 0.0;
  L[0] = math::log(g0(h));
  if constexpr (K >= 1) {
    const double ih = 1.0 / h, w = ih * ih;
    const double kk = 300.0 * w;
    const int Kr = K + 20 + (kk < 20000.0 ? static_cast<int>(Kokkos::ceil(kk)) : 20000);
    const double K1 = Kr + 1.0;
    double rr = 2.0 * K1 / (1.0 + Kokkos::sqrt(1.0 + 4.0 * K1 * w));
    double r[K + 1], mu[6];
    for (int j = Kr; j >= 1; --j) {
      rr = j / (1.0 + w * rr);
      if (j <= K) r[j] = rr;
    }
    mu[0] = 1.0;
    for (int j = 1; j <= K; ++j) mu[j] = mu[j - 1] * r[j];
    double scl = 1.0, fk = 1.0;
    for (int k = 1; k <= K; ++k) {
      scl *= -ih;
      fk *= k;
      L[ix(k, 0)] = dn::cumulant(k, mu) * scl / fk;
    }
  }
}

// Taylor coefficients of Ttilde(h + d, alpha + e), Ttilde(h, alpha) = int_0^alpha
// exp(-h^2 s^2/2)/(1 + s^2) ds, |alpha| <= 1, |alpha| h <= 2: in d from the moments
// int_0^alpha s^(2m) exp(-h^2 s^2/2)/(1 + s^2) ds (16-point Gauss-Legendre), in e
// from d Ttilde / dalpha = exp(-h^2 alpha^2/2) / (1 + alpha^2).
template <int K>
LAPLACE_HD void ttilde_series(double h, double alpha, double* T) {
  double Mm[K + 1];
  for (int m = 0; m <= K; ++m) Mm[m] = 0.0;
  for (int k = 0; k < 16; ++k) {
    const double x = alpha * rule(2 * (off_gl16 + k)), hx = h * x, x2 = x * x;
    double v = alpha * rule(2 * (off_gl16 + k) + 1) * math::exp(-0.5 * hx * hx) / (1.0 + x2);
    for (int m = 0; m <= K; ++m) {
      Mm[m] += v;
      v *= x2;
    }
  }
  constexpr double inv_fact[6] = {1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0};
  double hp[K + 1];
  hp[0] = 1.0;
  for (int k = 1; k <= K; ++k) hp[k] = hp[k - 1] * h;
  for (int i = 0; i <= K; ++i) {
    // [d^i] exp(-(h d + d^2/2) s^2): sum over m of (-1)^m / m! C(m, i - m) h^(2m - i) 2^(m - i) s^(2m)
    double t = 0.0;
    for (int m = (i + 1) / 2; m <= i; ++m) {
      const double cf = binom(m, i - m) * hp[2 * m - i] * inv_fact[m] / static_cast<double>(1 << (i - m));
      t += (m % 2 ? -cf : cf) * Mm[m];
    }
    T[ix(i, 0)] = t;
  }
  if constexpr (K >= 1) {
    // gamma(h + d, alpha + e) = exp(-(h + d)^2 (alpha + e)^2 / 2) / (1 + (alpha + e)^2)
    double gam[nc<K>], gm[nc<K>], lmb[K + 1];
    for (int k = 0; k < nc<K>; ++k) gam[k] = 0.0;
    const double Ph[3] = {0.5 * h * h, h, 0.5}, Qa[3] = {alpha * alpha, 2.0 * alpha, 1.0};
    for (int i = 0; i <= 2; ++i)
      for (int j = 0; j <= 2; ++j)
        if ((i || j) && i + j <= K) gam[ix(i, j)] -= Ph[i] * Qa[j];
    log1px2_coeffs<K>(alpha, lmb);
    for (int n = 1; n <= K; ++n) gam[ix(0, n)] -= lmb[n];
    double s2, e2;
    half_square_prod(alpha, h, s2, e2);
    ser_exp<K>(gam, gm, exp_neg2(s2, e2, 0.0, 0.0) / (1.0 + alpha * alpha));
    for (int n = 1; n <= K; ++n)
      for (int j = 1; j <= n; ++j) T[ix(n - j, j)] = gm[ix(n - j, j - 1)] / j;
  }
}

// Taylor coefficients of log F(z + d; alpha + e) at (z, alpha), total order <= K;
// those of order >= 1 are to be multiplied by the factor sc_p (pure z) or sc_m (mixed).
// method 0: the chosen algorithm; method 1: the elementary series everywhere (a baseline).
template <int K>
LAPLACE_OWEN_NOINLINE LAPLACE_HD void sn_series(double z, double alpha, const params& p, int method, double* L, scale& sc_p,
                                                scale& sc_m) {
  sc_p = sc_m = scale{};
  if (z != z || alpha != alpha) {
    for (int k = 0; k < nc<K>; ++k) L[k] = detail::quiet_nan();
    return;
  }
  const int reg = (z <= 0.0 && method == 0) ? region(-z, alpha, p) : 0;
  if (reg != 0) {
    const double h = -z;
    const double P[3] = {0.5 * h * h, h, 0.5};  // (h + d)^2 / 2
    double s1, e1;
    dn::half_square(h, s1, e1);
    if (reg == 2) {
      // log F = -(h + d)^2 (1 + (alpha + e)^2)/2 - log pi + log Jbar(h + d, alpha + e)
      log_jbar<K>(h, alpha, p, L);
      double s2, e2;
      half_square_prod(alpha, h, s2, e2);
      L[0] = -(s1 + s2) + (L[0] - log_pi - (e1 + e2));
      const double Cq[3] = {1.0 + alpha * alpha, 2.0 * alpha, 1.0};
      for (int i = 0; i <= 2; ++i)
        for (int j = 0; j <= 2; ++j)
          if ((i || j) && i + j <= K) L[ix(i, j)] -= P[i] * Cq[j];
    } else if (reg == 1) {
      // log G = log G0(h + d) + log(1 - Ttilde(h + d, alpha) / G0(h + d)) in d, then
      // d log G / dalpha = -gamma / G, gamma = exp(-h^2 alpha^2/2) / (1 + alpha^2), in e;
      // log F = -(h + d)^2/2 - log pi + log G
      double L0[nc<K>], G0s[nc<K>], Ts[nc<K>], R[nc<K>], Lr[nc<K>], gam[nc<K>], lmb[K + 1];
      log_g0_series<K>(h, L0);
      ser_exp<K>(L0, G0s, g0(h));
      ttilde_series<K>(h, alpha, Ts);
      for (int k = 0; k < nc<K>; ++k) R[k] = gam[k] = 0.0;
      for (int i = 0; i <= K; ++i) {
        double v = Ts[ix(i, 0)];
        for (int k = 1; k <= i; ++k) v -= G0s[ix(k, 0)] * R[ix(i - k, 0)];
        R[ix(i, 0)] = v / G0s[0];
      }
      const double G = G0s[0] - Ts[0];
      R[0] = G / G0s[0];
      for (int i = 1; i <= K; ++i) R[ix(i, 0)] = -R[ix(i, 0)];
      ser_log<K>(R, Lr);
      for (int i = 1; i <= K; ++i) L[ix(i, 0)] = L0[ix(i, 0)] + Lr[ix(i, 0)];
      const double Ph[3] = {0.5 * h * h, h, 0.5}, Qa[3] = {alpha * alpha, 2.0 * alpha, 1.0};
      for (int i = 0; i <= 2; ++i)
        for (int j = 0; j <= 2; ++j)
          if ((i || j) && i + j <= K) gam[ix(i, j)] -= Ph[i] * Qa[j];
      log1px2_coeffs<K>(alpha, lmb);
      for (int n = 1; n <= K; ++n) gam[ix(0, n)] -= lmb[n];
      double s2, e2;
      half_square_prod(alpha, h, s2, e2);
      const double c0 = exp_neg2(s2, e2, 0.0, 0.0) / ((1.0 + alpha * alpha) * G);
      alpha_ode<K>(gam, L, 1.0, c0, L);
      for (int n = 1; n <= K; ++n)
        for (int j = 1; j <= n; ++j) L[ix(n - j, j)] *= c0;
      L[0] = -s1 + (math::log(G) - log_pi - e1);
      for (int i = 1; i <= 2 && i <= K; ++i) L[ix(i, 0)] -= P[i];
    } else {
      // log F = -(h + d)^2/2 - log pi + log 2 + log G0(h + d) + log(1 - eps),
      // log eps = log Jbar(h + d, a + e) - (h + d)^2 (a + e)^2/2 - log G0(h + d) - log 2, a = -alpha
      const double a = -alpha;
      double La[nc<K>], L0[nc<K>], le[nc<K>], E[nc<K>], ja;
      log_jbar<K>(h, a, p, La, &ja);
      log_g0_series<K>(h, L0);
      double s2, e2;
      half_square_prod(a, h, s2, e2);
      const double Qa[3] = {a * a, 2.0 * a, 1.0};
      le[0] = 0.0;  // eps itself is computed as a product below
      for (int n = 1; n <= K; ++n)
        for (int j = 0; j <= n; ++j) {
          const int i = n - j;
          double v = La[ix(i, j)];
          if (i <= 2 && j <= 2) v -= P[i] * Qa[j];
          if (j == 0) v -= L0[ix(i, 0)];
          le[ix(i, j)] = v;
        }
      ser_exp<K>(le, E, exp_neg2(s2, e2, 0.0, 0.0) * ja / (2.0 * g0(h)));
      E[0] = 1.0 - E[0];
      for (int k = 1; k < nc<K>; ++k) E[k] = -E[k];
      ser_log<K>(E, L);
      L[0] = -s1 + (L[0] + log2 + L0[0] - log_pi - e1);
      for (int i = 1; i <= K; ++i) L[ix(i, 0)] += L0[ix(i, 0)] - (i <= 2 ? P[i] : 0.0);
      // e = -dalpha
      for (int n = 1; n <= K; ++n)
        for (int j = 1; j <= n; j += 2) L[ix(n - j, j)] = -L[ix(n - j, j)];
    }
    // d = -dz
    for (int n = 1; n <= K; ++n)
      for (int j = 0; j <= n; ++j)
        if ((n - j) % 2) L[ix(n - j, j)] = -L[ix(n - j, j)];
    return;
  }
  double F, logF;
  if (z <= 0.0) {
    const val_log r = lower(-z, alpha, p);
    F = r.v;
    logF = r.l;
  } else {
    const double S = lower(z, -alpha, p).v;
    if (S <= 0.5) {
      F = 1.0 - S;
      logF = Kokkos::log1p(-S);
    } else {
      F = Kokkos::erf(z * sqrt_half) + lower(z, alpha, p).v;
      logF = math::log(F);
    }
  }
  naive_series<K>(z, alpha, F, logF, L, sc_p, sc_m);
}

// The first derivatives alone: d log F / dz = f / F and d log F / dalpha = -g / F,
// f = 2 phi(z) Phi(alpha z), g = exp(-(1 + alpha^2) z^2/2) / (pi (1 + alpha^2)),
// with the Gaussian factors cancelled in closed form where F carries them:
//   Jbar region   f / F = sqrt(pi/2) erfcx(alpha h / sqrt 2) / Jbar,  g / F = 1 / ((1 + alpha^2) Jbar);
//   G regions     f / F = sqrt(2 pi) Phi(alpha z) / G,  g / F = exp(-(alpha h)^2/2) / ((1 + alpha^2) G),
// G = pi exp(h^2/2) F (G0 - Ttilde, or 2 G0 - exp(-eta a^2) Jbar(h, a)).
LAPLACE_HD mixed_partials<1> sn_d1(double z, double alpha, const params& p) {
  mixed_partials<1> r;
  if (z != z || alpha != alpha) {
    r.c[0] = r.c[1] = detail::quiet_nan();
    return r;
  }
  const double c = 1.0 + alpha * alpha;
  if (z <= 0.0) {
    const double h = -z;
    const int reg = region(h, alpha, p);
    if (reg != 0) {
      double s2, e2;
      half_square_prod(alpha, h, s2, e2);
      if (reg == 2) {
        double L[1], jv;
        log_jbar<0>(h, alpha, p, L, &jv);
        r.c[0] = 1.25331413731550025121 * own::erfcx(alpha * h * sqrt_half) / jv;
        r.c[1] = -1.0 / (c * jv);
        return r;
      }
      double G;
      if (reg == 1) {
        G = g0(h) - t_signed(h, alpha);
      } else {
        double L[1], jv;
        log_jbar<0>(h, -alpha, p, L, &jv);
        G = 2.0 * g0(h) - exp_neg2(s2, e2, 0.0, 0.0) * jv;
      }
      r.c[0] = 2.50662827463100050242 * own::Phi(-alpha * h) / G;
      r.c[1] = -exp_neg2(s2, e2, 0.0, 0.0) / (c * G);
      return r;
    }
  }
  double F;
  if (z <= 0.0) {
    F = lower(-z, alpha, p).v;
  } else {
    const double S = lower(z, -alpha, p).v;
    F = S <= 0.5 ? 1.0 - S : Kokkos::erf(z * sqrt_half) + lower(z, alpha, p).v;
  }
  double s1, e1, s2, e2;
  dn::half_square(z, s1, e1);
  half_square_prod(alpha, z, s2, e2);
  const double pz = alpha * z;
  const double Sm = s1 + s2, bv = Sm - s1;
  const double err_m = ((s1 - (Sm - bv)) + (s2 - bv)) + e1 + e2;
  if (pz < 0.0)
    r.c[0] = apply_scale(1.0, scale{Sm, (1.0 - err_m) * own::erfcx(-pz * sqrt_half) * inv_sqrt_2pi / F});
  else
    r.c[0] = apply_scale(1.0, scale{s1, (1.0 - e1) * 2.0 * inv_sqrt_2pi * own::Phi(pz) / F});
  r.c[1] = -apply_scale(1.0, scale{Sm, (1.0 - err_m) * inv_pi / (c * F)});
  return r;
}

// the components of order K: c[j] = d^K log F / dz^(K-j) dalpha^j
template <int K>
LAPLACE_HD_FORCEINLINE mixed_partials<K> sn_deriv(double z, double alpha, const params& p, int method) {
  if constexpr (K == 1) {
    if (method == 0) return sn_d1(z, alpha, p);
  }
  constexpr double fact[6] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0};
  double L[nc<K>];
  scale sc_p, sc_m;
  sn_series<K>(z, alpha, p, method, L, sc_p, sc_m);
  mixed_partials<K> r;
  for (int j = 0; j <= K; ++j) r.c[j] = apply_scale(fact[K - j] * fact[j] * L[ix(K - j, j)], j ? sc_m : sc_p);
  return r;
}

}  // namespace detail_owen

// Owen's T(h, a)
/**
 * \ingroup owen
 * \brief Owen's \f$T\f$ function, \f$T(h, a)\f$.
 *
 * Computes Owen's \f$T\f$ function defined by the integral:
 * \f[
 * T(h, a) = \frac{1}{2\pi} \int_0^a \frac{e^{-h^2 (1 + s^2)/2}}{1 + s^2} \, \mathrm{d}s
 * \f]
 * where \f$h \in \mathbb{R}\f$, \f$a \in \mathbb{R}\f$, and \f$s\f$ is the real integration variable.
 *
 * \param h Real argument \f$h \in \mathbb{R}\f$.
 * \param a Real argument \f$a \in \mathbb{R}\f$.
 * \return The value of \f$T(h, a)\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$h, a \in \mathbb{R}\f$.
 * - Even in \f$h\f$: \f$T(-h, a) = T(h, a)\f$.
 * - Odd in \f$a\f$: \f$T(h, -a) = -T(h, a)\f$.
 * - At \f$a = 0\f$: returns \f$0.0\f$.
 * - At \f$h = 0\f$: returns \f$\frac{1}{2\pi}\arctan a\f$.
 * - At \f$a = 1\f$: \f$T(h, 1) = \frac{1}{2}\Phi(h)(1 - \Phi(h))\f$.
 * - At \f$h = \pm\infty\f$: returns \f$0.0\f$ (with the sign of \f$a\f$).
 * - At \f$a = +\infty\f$ with finite \f$h\f$: returns \f$\Phi(-h)/2\f$.
 * - At \f$a = -\infty\f$ with finite \f$h\f$: returns \f$-\Phi(-h)/2\f$.
 * - If \f$h\f$ is NaN or \f$a\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 4.44 ulp (2.24 ulpL) on gcc / 4.08 ulp (1.95 ulpL) on CUDA; test bound 6.7 ulp (3.4 ulpL).
 *
 * \see owen, log_sn_cdf()
 */
LAPLACE_HD double owens_t(double h, double a) { return detail_owen::owens_t(h, a, detail_owen::params{}); }

// log F(z; alpha), F the distribution function of the standard skew normal
/**
 * \ingroup owen
 * \brief Logarithm of the standard skew normal distribution function, \f$\log F(z; \alpha)\f$.
 *
 * Computes the natural logarithm of the distribution function of the standard skew normal distrib:
 * \f[
 * F(z; \alpha) = 2 \int_{-\infty}^z \phi(t) \Phi(\alpha t) \, \mathrm{d}t = \Phi(z) - 2 T(z, \alpha)
 * \f]
 * where \f$\phi(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2/2}\f$ is the standard normal density, \f$\Phi\f$ is the standard normal distribution function, \f$T\f$ is Owen's \f$T\f$ function, \f$z \in \mathbb{R}\f$ is the argument, and \f$\alpha \in \mathbb{R}\f$ is the shape parameter.
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return The value of \f$\log F(z; \alpha)\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - At \f$\alpha = 0\f$: \f$F(z; 0) = \Phi(z)\f$, \f$\log F(z; 0) = \log \Phi(z)\f$.
 * - Reflection: \f$1 - F(z; \alpha) = F(-z; -\alpha)\f$.
 * - As \f$z \to -\infty\f$: returns \f$-\infty\f$.
 * - As \f$z \to +\infty\f$: returns \f$0.0\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 5.0 ulp (3.6 ulpL) on gcc / 5.0 ulp (3.6 ulpL) on CUDA; test bound 7.5 ulp (5.5 ulpL).
 *
 * \see owen, owens_t(), log_sn_cdf_d1()
 */
LAPLACE_HD double log_sn_cdf(double z, double alpha) { return detail_owen::log_sn_cdf(z, alpha, detail_owen::params{}); }

// d^K log F / dz^(K-j) dalpha^j in c[j], j = 0..K; one function per order
/**
 * \ingroup owen
 * \brief First derivatives in \f$(z, \alpha)\f$ of \f$\log F(z; \alpha)\f$.
 *
 * Computes the first partial derivatives of \f$\log F(z; \alpha)\f$:
 * \f[
 * c_0 = \frac{\partial}{\partial z} \log F(z; \alpha) = \frac{f}{F}, \quad
 * c_1 = \frac{\partial}{\partial \alpha} \log F(z; \alpha) = -\frac{g}{F}
 * \f]
 * where \f$f = 2 \phi(z) \Phi(\alpha z)\f$, \f$g = \frac{e^{-(1+\alpha^2) z^2/2}}{\pi (1 + \alpha^2)}\f$, and Gaussian factors are cancelled in closed form where \f$F\f$ carries them.
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return A mixed_partials of order 1 containing \f$c_0 = \frac{\partial}{\partial z} \log F\f$ in `c[0]` and \f$c_1 = \frac{\partial}{\partial \alpha} \log F\f$ in `c[1]`.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 4.8 ulpN (4.3 ulpL) on gcc / 5.1 ulpN (4.3 ulpL) on CUDA; test bound 7.7 ulpN (6.5 ulpL).
 *
 * \see owen, log_sn_cdf(), log_sn_cdf_d2()
 */
LAPLACE_HD mixed_partials<1> log_sn_cdf_d1(double z, double alpha) {
  return detail_owen::sn_deriv<1>(z, alpha, detail_owen::params{}, 0);
}

/**
 * \ingroup owen
 * \brief Second derivatives in \f$(z, \alpha)\f$ of \f$\log F(z; \alpha)\f$.
 *
 * Computes all second partial derivatives of \f$\log F(z; \alpha)\f$:
 * \f[
 * c_j = \frac{\partial^2}{\partial z^{2-j} \partial \alpha^j} \log F(z; \alpha), \quad j = 0, 1, 2.
 * \f]
 * Explicitly:
 * - \f$c_0 = \frac{\partial^2}{\partial z^2} \log F(z; \alpha)\f$
 * - \f$c_1 = \frac{\partial^2}{\partial z \partial \alpha} \log F(z; \alpha)\f$
 * - \f$c_2 = \frac{\partial^2}{\partial \alpha^2} \log F(z; \alpha)\f$
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return A mixed_partials of order 2 containing \f$c_j\f$ in `c[j]` for \f$j = 0, 1, 2\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 206 ulpN (24 ulpL) on gcc / 198 ulpN (24 ulpL) on CUDA; test bound 310.0 ulpN (37.0 ulpL).
 *
 * \see owen, log_sn_cdf(), log_sn_cdf_d1(), log_sn_cdf_d3()
 */
LAPLACE_HD mixed_partials<2> log_sn_cdf_d2(double z, double alpha) {
  return detail_owen::sn_deriv<2>(z, alpha, detail_owen::params{}, 0);
}

/**
 * \ingroup owen
 * \brief Third derivatives in \f$(z, \alpha)\f$ of \f$\log F(z; \alpha)\f$.
 *
 * Computes all third partial derivatives of \f$\log F(z; \alpha)\f$:
 * \f[
 * c_j = \frac{\partial^3}{\partial z^{3-j} \partial \alpha^j} \log F(z; \alpha), \quad j = 0, \dots, 3.
 * \f]
 * Explicitly:
 * - \f$c_0 = \frac{\partial^3}{\partial z^3} \log F(z; \alpha)\f$
 * - \f$c_1 = \frac{\partial^3}{\partial z^2 \partial \alpha} \log F(z; \alpha)\f$
 * - \f$c_2 = \frac{\partial^3}{\partial z \partial \alpha^2} \log F(z; \alpha)\f$
 * - \f$c_3 = \frac{\partial^3}{\partial \alpha^3} \log F(z; \alpha)\f$
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return A mixed_partials of order 3 containing \f$c_j\f$ in `c[j]` for \f$j = 0, \dots, 3\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 2240 ulpN (501 ulpL) on gcc / 2150 ulpN (481 ulpL) on CUDA; test bound 3400.0 ulpN (760.0 ulpL).
 *
 * \see owen, log_sn_cdf(), log_sn_cdf_d2(), log_sn_cdf_d4()
 */
LAPLACE_HD mixed_partials<3> log_sn_cdf_d3(double z, double alpha) {
  return detail_owen::sn_deriv<3>(z, alpha, detail_owen::params{}, 0);
}

/**
 * \ingroup owen
 * \brief Fourth derivatives in \f$(z, \alpha)\f$ of \f$\log F(z; \alpha)\f$.
 *
 * Computes all fourth partial derivatives of \f$\log F(z; \alpha)\f$:
 * \f[
 * c_j = \frac{\partial^4}{\partial z^{4-j} \partial \alpha^j} \log F(z; \alpha), \quad j = 0, \dots, 4.
 * \f]
 * Explicitly:
 * - \f$c_0 = \frac{\partial^4}{\partial z^4} \log F(z; \alpha)\f$
 * - \f$c_1 = \frac{\partial^4}{\partial z^3 \partial \alpha} \log F(z; \alpha)\f$
 * - \f$c_2 = \frac{\partial^4}{\partial z^2 \partial \alpha^2} \log F(z; \alpha)\f$
 * - \f$c_3 = \frac{\partial^4}{\partial z \partial \alpha^3} \log F(z; \alpha)\f$
 * - \f$c_4 = \frac{\partial^4}{\partial \alpha^4} \log F(z; \alpha)\f$
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return A mixed_partials of order 4 containing \f$c_j\f$ in `c[j]` for \f$j = 0, \dots, 4\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error 1500 ulpN (777 ulpL) on gcc / 1490 ulpN (784 ulpL) on CUDA; test bound 2300.0 ulpN (1200.0 ulpL).
 *
 * \see owen, log_sn_cdf(), log_sn_cdf_d3(), log_sn_cdf_d5()
 */
LAPLACE_HD mixed_partials<4> log_sn_cdf_d4(double z, double alpha) {
  return detail_owen::sn_deriv<4>(z, alpha, detail_owen::params{}, 0);
}

/**
 * \ingroup owen
 * \brief Fifth derivatives in \f$(z, \alpha)\f$ of \f$\log F(z; \alpha)\f$.
 *
 * Computes all fifth partial derivatives of \f$\log F(z; \alpha)\f$:
 * \f[
 * c_j = \frac{\partial^5}{\partial z^{5-j} \partial \alpha^j} \log F(z; \alpha), \quad j = 0, \dots, 5.
 * \f]
 * Explicitly:
 * - \f$c_0 = \frac{\partial^5}{\partial z^5} \log F(z; \alpha)\f$
 * - \f$c_1 = \frac{\partial^5}{\partial z^4 \partial \alpha} \log F(z; \alpha)\f$
 * - \f$c_2 = \frac{\partial^5}{\partial z^3 \partial \alpha^2} \log F(z; \alpha)\f$
 * - \f$c_3 = \frac{\partial^5}{\partial z^2 \partial \alpha^3} \log F(z; \alpha)\f$
 * - \f$c_4 = \frac{\partial^5}{\partial z \partial \alpha^4} \log F(z; \alpha)\f$
 * - \f$c_5 = \frac{\partial^5}{\partial \alpha^5} \log F(z; \alpha)\f$
 *
 * \param z Real argument \f$z \in \mathbb{R}\f$.
 * \param alpha Real shape parameter \f$\alpha \in \mathbb{R}\f$.
 * \return A mixed_partials of order 5 containing \f$c_j\f$ in `c[j]` for \f$j = 0, \dots, 5\f$.
 *
 * \par Domain and special values
 * - Defined for all \f$z, \alpha \in \mathbb{R}\f$.
 * - If \f$z\f$ is NaN or \f$\alpha\f$ is NaN: returns quiet NaN in all components.
 *
 * \par Measured accuracy
 * Maximum error \f$1.16 \times 10^5\f$ ulpN (\f$2.7 \times 10^4\f$ ulpL) on gcc / \f$1.17 \times 10^5\f$ ulpN (\f$2.8 \times 10^4\f$ ulpL) on CUDA; test bound \f$1.8 \times 10^5\f$ ulpN (\f$4.2 \times 10^4\f$ ulpL). Near zeros of components of order 3 to 5, the problem is ill-conditioned: a 1-ulp variation in \f$F\f$ moves the derivative by up to \f$1.4 \times 10^4\f$ to \f$2.9 \times 10^4\f$ ulps.
 *
 * \see owen, log_sn_cdf(), log_sn_cdf_d4()
 */
LAPLACE_HD mixed_partials<5> log_sn_cdf_d5(double z, double alpha) {
  return detail_owen::sn_deriv<5>(z, alpha, detail_owen::params{}, 0);
}

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls (docs/numerics.md, 2026-11-09).
LAPLACE_HD_FORCEINLINE double owens_t(double h, double a) { return own::owens_t(h, a); }
LAPLACE_HD_FORCEINLINE double log_sn_cdf(double z, double alpha) { return own::log_sn_cdf(z, alpha); }
LAPLACE_HD_FORCEINLINE mixed_partials<1> log_sn_cdf_d1(double z, double alpha) { return own::log_sn_cdf_d1(z, alpha); }
LAPLACE_HD_FORCEINLINE mixed_partials<2> log_sn_cdf_d2(double z, double alpha) { return own::log_sn_cdf_d2(z, alpha); }
LAPLACE_HD_FORCEINLINE mixed_partials<3> log_sn_cdf_d3(double z, double alpha) { return own::log_sn_cdf_d3(z, alpha); }
LAPLACE_HD_FORCEINLINE mixed_partials<4> log_sn_cdf_d4(double z, double alpha) { return own::log_sn_cdf_d4(z, alpha); }
LAPLACE_HD_FORCEINLINE mixed_partials<5> log_sn_cdf_d5(double z, double alpha) { return own::log_sn_cdf_d5(z, alpha); }

}  // namespace laplace::math
