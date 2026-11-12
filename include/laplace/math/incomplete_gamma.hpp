// Regularized incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x),
// and their logarithms, Laplace's own (a > 0, x >= 0).
//
// Why own: in double precision (the only one on the GPU) Boost.Math loses the
// relative accuracy of the tails for large a, because its prefactor
// x^a e^-x / Gamma(a) is formed from terms of size a log a that cancel
// (docs/numerics.md). Here the prefactor is written so that nothing cancels:
//
//     x^a e^-x / Gamma(a + 1) = exp(-a phi(x/a)) / (sqrt(2 pi a) Gamma*(a)),
//     phi(lambda) = lambda - 1 - log(lambda) = -log1pmx((x - a)/a) >= 0,
//
// with Gamma*(a) = Gamma(a) / (sqrt(2 pi / a) (a/e)^a) from its Stirling series
// for a >= 10. The remaining error of the prefactor is that of exp(-a phi):
// about a phi ulps, which is |log P| or |log Q| in the tails; the logarithmic
// versions do not have it.
//
// Two routes, chosen by gamma_inc from a and the distance of x from a:
//
// (1) Temme's uniform asymptotic expansion, for large a near x = a (see the
//     comment on gamma_inc_temme). It needs O(1) operations, against the
//     O(sqrt(a)) terms of the classic route for x close to a.
//
// (2) The classic route (gamma_inc_classic), for all other arguments:
//   a < 1, x < 1.5   P = g (1 + J), Q = -expm1(log g) - g J, with
//                    g = x^a / Gamma(a + 1) and J = a sum_{n>=1} (-x)^n / (n! (a + n));
//                    Q is computed directly, without 1 - P, since it is small
//                    when a is small;
//   x < a + 1        P = prefactor * sum_{n>=0} x^n / ((a+1)...(a+n)), Q = 1 - P;
//   x >= a + 1       Q = a * prefactor * Legendre's continued fraction (modified
//                    Lentz), P = 1 - Q.
// In the last two the complement is formed only where it is not small.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>

namespace laplace::math {

namespace own {

/**
 * \ingroup incomplete_gamma
 * \brief Result of incomplete gamma evaluation functions.
 *
 * Holds the values \f$P(a, x)\f$, \f$Q(a, x)\f$, and their natural logarithms,
 * along with low-order correction terms for extended precision in tail regions.
 *
 * - `p`:
 * Regularized lower incomplete gamma function \f$P(a, x)\f$.
 *
 * - `q`:
 * Regularized upper incomplete gamma function \f$Q(a, x) = 1 - P(a, x)\f$.
 *
 * - `log_p`:
 * Natural logarithm of the regularized lower incomplete gamma function, \f$\log P(a, x)\f$.
 *
 * - `log_q`:
 * Natural logarithm of the regularized upper incomplete gamma function, \f$\log Q(a, x)\f$.
 *
 * - `log_p_lo`:
 * Low-order correction of \f$\log P(a, x)\f$ as a double-double number.
 *
 * - `log_q_lo`:
 * Low-order correction of \f$\log Q(a, x)\f$ as a double-double number.
 */
struct gamma_inc_result {
  double p, q;          // P(a, x), Q(a, x)
  double log_p, log_q;  // their logarithms
  // the low parts of the logarithms as double-doubles (log_p + log_p_lo), where
  // they are computed (the lower tail with x below a/2 or with small a); 0
  // elsewhere. The inverses use them to solve far in the lower tail.
  double log_p_lo = 0.0, log_q_lo = 0.0;
};

namespace detail_igamma {

constexpr double half_log_two_pi = 0.91893853320467274178;

// log Gamma*(a), Gamma*(a) = Gamma(a) / (sqrt(2 pi / a) (a/e)^a), a >= 1
LAPLACE_HD double log_gamma_star(double a) {
  if (a >= 10.0) {
    // sum_k B_2k / (2k (2k-1) a^(2k-1)), k = 1..8
    const double ia = 1.0 / a, w = ia * ia;
    double s = -3617.0 / 122400.0;
    s = s * w + 1.0 / 156.0;
    s = s * w - 691.0 / 360360.0;
    s = s * w + 1.0 / 1188.0;
    s = s * w - 1.0 / 1680.0;
    s = s * w + 1.0 / 1260.0;
    s = s * w - 1.0 / 360.0;
    s = s * w + 1.0 / 12.0;
    return ia * s;
  }
  return math::lgamma(a) + a - (a - 0.5) * math::log(a) - half_log_two_pi;
}

// log(x^a e^-x / Gamma(a + 1)), without cancellation
LAPLACE_HD double log_prefactor(double a, double x) {
  if (a < 1.0) {
    const double lg1p = Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
    return a * math::log(x) - x - lg1p;
  }
  // phi(lambda) = lambda - 1 - log(lambda), lambda = x / a: through
  // log1pmx(lambda - 1) near lambda = 1, where the direct form cancels; directly
  // elsewhere, where 1 + (lambda - 1) would lose the digits of a small lambda
  const double lambda = x / a;
  double phi;
  if (lambda > 0.5 && lambda < 1.5) {
    phi = -boost::math::log1pmx((x - a) / a, math::detail::boost_policy());
  } else {
    phi = lambda - 1.0 - math::log(lambda);
  }
  return -a * phi - 0.5 * math::log(a) - half_log_two_pi - log_gamma_star(a);
}

// The same as a double-double. For a >= 1 and x <= a/2 the large term
// -a phi = a - x + a log(x/a) is formed with log x and log a split by their
// exponents (ddm::log_split), so its absolute error is a few 2^-53 times a
// instead of |a phi| 2^-53; elsewhere the plain value (lo = 0).
LAPLACE_HD math::detail::ddm::dd log_prefactor_dd(double a, double x) {
  namespace ddm = math::detail::ddm;
  if (a >= 1.0 && x <= 0.5 * a) {
    const ddm::dd log_lambda = ddm::log_split(x) - ddm::log_split(a);
    return (log_lambda * a + ddm::two_sum(a, -x)) +
           (-0.5 * math::log(a) - half_log_two_pi - log_gamma_star(a));
  }
  return ddm::dd(log_prefactor(a, x));
}

// Temme's coefficients c_k(eta), k = 0..9, as polynomials in eta (ascending
// powers), concatenated: c_k has temme_len(k) coefficients from temme_offset(k).
// c_9 is a constant. Values: Boost.Math's igamma_temme_large, 53-bit version
// (boost/math/special_functions/detail/igamma_large.hpp, Boost Software
// License 1.0, copyright John Maddock 2006 and Matt Borland 2024), which it
// took from Temme (1979) and DiDonato and Morris (1986).
// clang-format off
#define LAPLACE_TEMME_COEF                                                                                         \
  -0.33333333333333333, 0.083333333333333333, -0.014814814814814815, 0.0011574074074074074,                         \
      0.0003527336860670194, -0.00017875514403292181, 0.39192631785224378e-4, -0.21854485106799922e-5,             \
      -0.185406221071516e-5, 0.8296711340953086e-6, -0.17665952736826079e-6, 0.67078535434014986e-8,               \
      0.10261809784240308e-7, -0.43820360184533532e-8, 0.91476995822367902e-9,                                      \
      -0.0018518518518518519, -0.0034722222222222222, 0.0026455026455026455, -0.00099022633744855967,              \
      0.00020576131687242798, -0.40187757201646091e-6, -0.18098550334489978e-4, 0.76491609160811101e-5,           \
      -0.16120900894563446e-5, 0.46471278028074343e-8, 0.1378633446915721e-6, -0.5752545603517705e-7,             \
      0.11951628599778147e-7,                                                                                       \
      0.0041335978835978836, -0.0026813271604938272, 0.00077160493827160494, 0.20093878600823045e-5,              \
      -0.00010736653226365161, 0.52923448829120125e-4, -0.12760635188618728e-4, 0.34235787340961381e-7,           \
      0.13721957309062933e-5, -0.6298992138380055e-6, 0.14280614206064242e-6,                                       \
      0.00064943415637860082, 0.00022947209362139918, -0.00046918949439525571, 0.00026772063206283885,            \
      -0.75618016718839764e-4, -0.23965051138672967e-6, 0.11082654115347302e-4, -0.56749528269915966e-5,          \
      0.14230900732435884e-5,                                                                                       \
      -0.0008618882909167117, 0.00078403922172006663, -0.00029907248030319018, -0.14638452578843418e-5,           \
      0.66414982154651222e-4, -0.39683650471794347e-4, 0.11375726970678419e-4,                                      \
      -0.00033679855336635815, -0.69728137583658578e-4, 0.00027727532449593921, -0.00019932570516188848,          \
      0.67977804779372078e-4, 0.1419062920643967e-6, -0.13594048189768693e-4, 0.80184702563342015e-5,             \
      -0.22914811765080952e-5,                                                                                      \
      0.00053130793646399222, -0.00059216643735369388, 0.00027087820967180448, 0.79023532326603279e-6,            \
      -0.81539693675619688e-4, 0.56116827531062497e-4, -0.18329116582843376e-4,                                     \
      0.00034436760689237767, 0.51717909082605922e-4, -0.00033493161081142236, 0.0002812695154763237,             \
      -0.00010976582244684731,                                                                                      \
      -0.00065262391859530942, 0.00083949872067208728, -0.00043829709854172101,                                     \
      -0.00059676129019274625
// clang-format on
#if defined(__CUDACC__)
static __constant__ double temme_dev[80] = {LAPLACE_TEMME_COEF};
#endif
inline constexpr double temme_host[80] = {LAPLACE_TEMME_COEF};

LAPLACE_HD_FORCEINLINE double temme_coef(int i) {
  KOKKOS_IF_ON_DEVICE((return temme_dev[i];))
  KOKKOS_IF_ON_HOST((return temme_host[i];))
}

// first index of c_k in the table, k = 0..9
LAPLACE_HD_FORCEINLINE int temme_offset(int k) {
  constexpr int offset[10] = {0, 15, 28, 39, 48, 55, 64, 71, 76, 79};
  return offset[k];
}

// number of coefficients of c_k, k = 0..9
LAPLACE_HD_FORCEINLINE int temme_length(int k) {
  constexpr int length[10] = {15, 13, 11, 9, 7, 9, 7, 5, 3, 1};
  return length[k];
}

// S(a, eta) = sum_{k=0}^{9} c_k(eta) a^-k, by Horner's rule in 1/a over k and in
// eta within each c_k
LAPLACE_HD double temme_sum(double a, double eta) {
  const double w = 1.0 / a;
  double s = 0.0;
  for (int k = 9; k >= 0; --k) {
    const int off = temme_offset(k), len = temme_length(k);
    double c = temme_coef(off + len - 1);
    for (int j = len - 2; j >= 0; --j) c = c * eta + temme_coef(off + j);
    s = s * w + c;
  }
  return s;
}

// gamma_inc takes the Temme route when a >= temme_a_min and |eta| <= temme_eta_max,
// where eta is defined in gamma_inc_temme. The coefficients c_k(eta) are
// polynomials truncated about eta = 0: measured on the gamma_*_large
// references (log P and log Q of the small tail), the route is within 6 ulp
// for |eta| < 0.5 at every a from 1e2 to 1e15, 11-38 ulp for |eta| in
// [0.5, 1), and up to 1.7e3 ulp beyond. For |eta| > 0.5 (|x/a - 1| above
// about 0.4) the classic route needs a few tens of terms.
inline constexpr double temme_a_min = 100.0;
inline constexpr double temme_eta_max = 0.5;

}  // namespace detail_igamma

// The classic route: the series, the continued fraction and the small-a series,
// as described in the header comment.
LAPLACE_HD gamma_inc_result gamma_inc_classic(double a, double x) {
  using namespace detail_igamma;
  constexpr double eps = 1.1102230246251565e-16;  // 2^-53
  constexpr int max_iter = 1000000;
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(x >= 0.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == inf) return {1.0, 0.0, 0.0, -inf};

  if (a < 1.0 && x < 1.5) {
    // J = a sum_{n>=1} (-x)^n / (n! (a + n))
    double term = 1.0, J = 0.0;
    for (int n = 1; n < 200; ++n) {
      term *= -x / n;
      const double c = term / (a + n);
      J += c;
      if (Kokkos::fabs(c) <= eps * Kokkos::fabs(J)) break;
    }
    J *= a;
    // log g = a log x - log Gamma(a + 1), with log x split by its exponent
    namespace ddm = math::detail::ddm;
    const ddm::dd log_g =
        ddm::log_split(x) * a - Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
    double g = math::exp(log_g.hi);
    g += g * log_g.lo;
    const double p = g * (1.0 + J);
    const double q = -Kokkos::expm1(log_g.hi) - g * J;
    const ddm::dd log_p = log_g + Kokkos::log1p(J);
    return {p, q, log_p.hi, math::log(q), log_p.lo, 0.0};
  }

  if (x < a + 1.0) {
    // P = prefactor * sum_{n>=0} x^n / ((a+1)...(a+n))
    double term = 1.0, s = 1.0;
    for (int n = 1; n < max_iter; ++n) {
      term *= x / (a + n);
      s += term;
      if (term <= eps * s) break;
    }
    const auto log_p = log_prefactor_dd(a, x) + math::log(s);
    double p = math::exp(log_p.hi);
    p += p * log_p.lo;
    return {p, 1.0 - p, log_p.hi, Kokkos::log1p(-p), log_p.lo, 0.0};
  }

  const double log_d = log_prefactor(a, x);

  // Q = x^a e^-x / Gamma(a) * 1 / (x + 1 - a - 1 (1 - a) / (x + 3 - a - ...))
  constexpr double tiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i < max_iter; ++i) {
    const double an = -i * (i - a);
    b += 2.0;
    d = an * d + b;
    if (Kokkos::fabs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Kokkos::fabs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    // 4 eps: with FMA contraction (CUDA) delta can settle one ulp away from 1
    if (Kokkos::fabs(delta - 1.0) <= 4.0 * eps) break;
  }
  const double log_q = log_d + math::log(a) + math::log(h);
  const double q = math::exp(log_q);
  return {1.0 - q, q, Kokkos::log1p(-q), log_q};
}

// Temme's uniform asymptotic expansion of the incomplete gamma functions
// (N. M. Temme, SIAM J. Math. Anal. 10, 1979; DiDonato and Morris, ACM TOMS 12,
// 1986), for large a with x near a. With
//
//     lambda = x / a,   eta^2 / 2 = phi = lambda - 1 - log(lambda) >= 0,
//     eta having the sign of lambda - 1,   y = a phi,   s = sqrt(y),
//
// the expansion reads
//
//     Q(a, x) = erfc(s) / 2 + R,     P(a, x) = erfc(-s)/2 - R, with
//     R = exp(-y) S(a, eta) / sqrt(2 pi a),  S = sum_k c_k(eta) a^-k,
//
// and erfc(s) = exp(-y) erfcx(s) (exact, since s^2 = y), so with E = erfcx(s)/2
// both are exp(-y) times a bracket:
//
//     x >= a:  Q = exp(-y) (E + r),   P = 1 - Q
//     x <  a:  P = exp(-y) (E - r),   Q = 1 - P
//
// with r = S / sqrt(2 pi a). Nothing underflows before exp(-y) does, and the
// logarithm of the small one is -y + log(bracket). phi comes from log1pmx, so
// it has no cancellation near x = a; c_k(eta) are polynomials, so eta = 0 needs
// no special case (c_0(0) = -1/3). The coefficients are those of Boost.Math's
// 53-bit version (see LAPLACE_TEMME_COEF), with c_9 constant; the truncation
// after k = 9 is of order a^-10 (measured in note/numerics.md).
//
// Argument: phi (not x), computed by the caller as -log1pmx((x - a) / a).
LAPLACE_HD gamma_inc_result gamma_inc_temme(double a, double x, double phi) {
  using namespace detail_igamma;
  constexpr double two_pi = 6.2831853071795864769;
  const double y = a * phi;
  const double s = Kokkos::sqrt(y);
  const double E = 0.5 * own::erfcx(s);
  const double eta = (x < a ? -1.0 : 1.0) * Kokkos::sqrt(2.0 * phi);
  const double r = temme_sum(a, eta) / Kokkos::sqrt(two_pi * a);
  const double e_minus_y = math::exp(-y);
  if (x >= a) {
    const double bracket = E + r;
    const double q = e_minus_y * bracket;
    return {1.0 - q, q, Kokkos::log1p(-q), -y + math::log(bracket)};
  }
  const double bracket = E - r;
  const double p = e_minus_y * bracket;
  return {p, 1.0 - p, -y + math::log(bracket), Kokkos::log1p(-p)};
}

// P(a, x), Q(a, x) and their logarithms. The Temme route when a is large and x
// is near a, the classic route elsewhere.
/**
 * \ingroup incomplete_gamma
 * \brief Regularized incomplete gamma functions \f$P(a, x)\f$, \f$Q(a, x)\f$, and logarithms in one call.
 *
 * Computes, in one call, everything the module's single functions return: the regularized lower and
 * upper incomplete gamma functions
 * \f[
 * P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} \, \mathrm{d}t, \quad
 * Q(a, x) = \frac{1}{\Gamma(a)} \int_x^\infty t^{a-1} e^{-t} \, \mathrm{d}t = 1 - P(a, x),
 * \f]
 * their natural logarithms \f$\log P(a, x)\f$ and \f$\log Q(a, x)\f$, and the low parts of the logarithms
 * as double-double numbers.
 *
 * The single functions gamma_p(), gamma_q(), log_gamma_p(), and log_gamma_q() each call gamma_inc() and
 * keep one field, so one call costs as much as evaluating one of them; use gamma_inc() when more than
 * one is needed.
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return A \ref gamma_inc_result struct holding \f$P(a, x)\f$, \f$Q(a, x)\f$, \f$\log P(a, x)\f$,
 * \f$\log Q(a, x)\f$, and the low parts of the logarithms.
 *
 * \par Domain and special values
 * Those of the single functions (see gamma_p(), gamma_q(), log_gamma_p(), log_gamma_q()):
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, or \f$x < 0\f$, or if \f$a\f$ or \f$x\f$ is NaN: returns quiet NaN for the four values (the low parts are 0).
 * - At \f$x = 0\f$: \f$P(a, 0) = 0.0\f$, \f$Q(a, 0) = 1.0\f$, \f$\log P(a, 0) = -\infty\f$, \f$\log Q(a, 0) = 0.0\f$.
 * - At \f$x = +\infty\f$: \f$P(a, \infty) = 1.0\f$, \f$Q(a, \infty) = 0.0\f$, \f$\log P(a, \infty) = 0.0\f$, \f$\log Q(a, \infty) = -\infty\f$.
 *
 * \par Measured accuracy
 * Those of the single functions: see gamma_p(), gamma_q(), log_gamma_p(), and log_gamma_q().
 *
 * \see incomplete_gamma, gamma_inc_result, gamma_p(), gamma_q(), log_gamma_p(), log_gamma_q()
 */
LAPLACE_HD gamma_inc_result gamma_inc(double a, double x) {
  using namespace detail_igamma;
  const double nan = math::detail::quiet_nan();
  const double inf = math::detail::pos_inf();
  if (!(a > 0.0) || !(x >= 0.0)) return {nan, nan, nan, nan};
  if (x == 0.0) return {0.0, 1.0, -inf, 0.0};
  if (x == inf) return {1.0, 0.0, 0.0, -inf};
  if (a >= temme_a_min && a < inf) {
    const double phi = -boost::math::log1pmx((x - a) / a, math::detail::boost_policy());
    if (Kokkos::sqrt(2.0 * phi) <= temme_eta_max) return gamma_inc_temme(a, x, phi);
  }
  return gamma_inc_classic(a, x);
}

/**
 * \ingroup incomplete_gamma
 * \brief Regularized lower incomplete gamma function \f$P(a, x)\f$.
 *
 * Computes the regularized lower incomplete gamma function
 * \f[
 * P(a, x) = \frac{\gamma(a, x)}{\Gamma(a)} = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} \, \mathrm{d}t
 * \f]
 * where \f$a \in (0, \infty)\f$ is the shape parameter, \f$x \in [0, \infty)\f$ is the argument,
 * and \f$t \in [0, x]\f$ is the integration variable.
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$P(a, x) \in [0, 1]\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, or \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$1.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 7.6 ulp1, 30.3 ulpL on general reference set (test bound 12.0 ulp1, 45.0 ulpL),
 * measured in the section "Own incomplete gamma" before Temme's expansion and double-double logarithms
 * (backend not stated; later measurements showed maxima stayed unchanged or lower);
 * 0.82 ulp1, 2.9–3.4 ulpL for large shapes \f$a \in [10^2, 10^{15}]\f$ (test bound 1.5 ulp1, 4.5 ulpL).
 *
 * \see incomplete_gamma, gamma_q(), log_gamma_p(), gamma_p_inv()
 */
LAPLACE_HD_FORCEINLINE double gamma_p(double a, double x) { return gamma_inc(a, x).p; }

/**
 * \ingroup incomplete_gamma
 * \brief Regularized upper incomplete gamma function \f$Q(a, x)\f$.
 *
 * Computes the regularized upper incomplete gamma function
 * \f[
 * Q(a, x) = \frac{\Gamma(a, x)}{\Gamma(a)} = \frac{1}{\Gamma(a)} \int_x^\infty t^{a-1} e^{-t} \, \mathrm{d}t = 1 - P(a, x)
 * \f]
 * where \f$a \in (0, \infty)\f$ is the shape parameter, \f$x \in [0, \infty)\f$ is the argument,
 * and \f$t \in [x, \infty)\f$ is the integration variable.
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$Q(a, x) \in [0, 1]\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, or \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$1.0\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 7.6 ulp1, 24.7 ulpL on general reference set (test bound 12.0 ulp1, 40.0 ulpL),
 * measured in the section "Own incomplete gamma" before Temme's expansion and double-double logarithms
 * (backend not stated; later measurements showed maxima stayed unchanged or lower);
 * 0.82 ulp1, 2.9–3.4 ulpL for large shapes \f$a \in [10^2, 10^{15}]\f$ (test bound 1.5 ulp1, 5.5 ulpL).
 *
 * \see incomplete_gamma, gamma_p(), log_gamma_q(), gamma_q_inv()
 */
LAPLACE_HD_FORCEINLINE double gamma_q(double a, double x) { return gamma_inc(a, x).q; }

}  // namespace own

// What the rest of Laplace calls: the own versions (docs/numerics.md).
LAPLACE_HD_FORCEINLINE double gamma_p(double a, double x) { return own::gamma_inc(a, x).p; }
LAPLACE_HD_FORCEINLINE double gamma_q(double a, double x) { return own::gamma_inc(a, x).q; }

/**
 * \ingroup incomplete_gamma
 * \brief Natural logarithm of the regularized lower incomplete gamma function, \f$\log P(a, x)\f$.
 *
 * Computes \f$\log P(a, x)\f$ directly without forming \f$P(a, x)\f$, avoiding underflow in the lower tail:
 * \f[
 * \log P(a, x) = \log \left(\frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} \, \mathrm{d}t\right)
 * \f]
 * where \f$a \in (0, \infty)\f$ is the shape parameter, \f$x \in [0, \infty)\f$ is the argument,
 * and \f$t \in [0, x]\f$ is the integration variable.
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\log P(a, x) \in (-\infty, 0]\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, or \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$-\infty\f$.
 * - \f$x = +\infty\f$: returns \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error within 5.9–6.4 ulp1 for large shapes \f$a \in [10^2, 10^{15}]\f$ (test bound 9.0 ulp1).
 *
 * \see incomplete_gamma, gamma_p(), log_gamma_q(), gamma_p_inv_logp(), log_gamma_p_da1()
 */
LAPLACE_HD_FORCEINLINE double log_gamma_p(double a, double x) { return own::gamma_inc(a, x).log_p; }

/**
 * \ingroup incomplete_gamma
 * \brief Natural logarithm of the regularized upper incomplete gamma function, \f$\log Q(a, x)\f$.
 *
 * Computes \f$\log Q(a, x)\f$ directly without forming \f$Q(a, x)\f$, avoiding underflow in the upper tail:
 * \f[
 * \log Q(a, x) = \log \left(\frac{1}{\Gamma(a)} \int_x^\infty t^{a-1} e^{-t} \, \mathrm{d}t\right)
 * \f]
 * where \f$a \in (0, \infty)\f$ is the shape parameter, \f$x \in [0, \infty)\f$ is the argument,
 * and \f$t \in [x, \infty)\f$ is the integration variable.
 *
 * \param a Positive shape parameter \f$a \in (0, \infty)\f$.
 * \param x Non-negative argument \f$x \in [0, \infty)\f$.
 * \return The value of \f$\log Q(a, x) \in (-\infty, 0]\f$.
 *
 * \par Domain and special values
 * - \f$a > 0, x \ge 0\f$: standard domain.
 * - \f$a \le 0\f$, or \f$x < 0\f$, or \f$a\f$ is NaN, or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$-\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error within 5.9–6.4 ulp1 for large shapes \f$a \in [10^2, 10^{15}]\f$ (test bound 10.0 ulp1).
 *
 * \see incomplete_gamma, gamma_q(), log_gamma_p(), gamma_q_inv_logq(), log_gamma_q_da1()
 */
LAPLACE_HD_FORCEINLINE double log_gamma_q(double a, double x) { return own::gamma_inc(a, x).log_q; }

}  // namespace laplace::math
