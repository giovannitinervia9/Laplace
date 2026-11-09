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
// Branches (the classical ones, as in Numerical Recipes and DiDonato-Morris):
//   a < 1, x < 1.5   P = g (1 + J), Q = -expm1(log g) - g J, with
//                    g = x^a / Gamma(a + 1) and J = a sum_{n>=1} (-x)^n / (n! (a + n));
//                    Q is computed directly, without 1 - P, since it is small
//                    when a is small;
//   x < a + 1        P = prefactor * sum_{n>=0} x^n / ((a+1)...(a+n)), Q = 1 - P;
//   x >= a + 1       Q = a * prefactor * Legendre's continued fraction (modified
//                    Lentz), P = 1 - Q.
// In the last two the complement is formed only where it is not small. The
// series and the fraction need O(sqrt(a)) terms near x = a: for very large a a
// uniform asymptotic expansion (Temme) would be faster; it is not written yet.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>

namespace laplace::math {

namespace own {

struct gamma_inc_result {
  double p, q;          // P(a, x), Q(a, x)
  double log_p, log_q;  // their logarithms
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

}  // namespace detail_igamma

LAPLACE_HD gamma_inc_result gamma_inc(double a, double x) {
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
    const double log_g = a * math::log(x) - Kokkos::log1p(boost::math::tgamma1pm1(a, math::detail::boost_policy()));
    const double g = math::exp(log_g);
    const double p = g * (1.0 + J);
    const double q = -Kokkos::expm1(log_g) - g * J;
    return {p, q, log_g + Kokkos::log1p(J), math::log(q)};
  }

  const double log_d = log_prefactor(a, x);
  if (x < a + 1.0) {
    // P = prefactor * sum_{n>=0} x^n / ((a+1)...(a+n))
    double term = 1.0, s = 1.0;
    for (int n = 1; n < max_iter; ++n) {
      term *= x / (a + n);
      s += term;
      if (term <= eps * s) break;
    }
    const double log_p = log_d + math::log(s);
    const double p = math::exp(log_p);
    return {p, 1.0 - p, log_p, Kokkos::log1p(-p)};
  }

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

LAPLACE_HD_FORCEINLINE double gamma_p(double a, double x) { return gamma_inc(a, x).p; }
LAPLACE_HD_FORCEINLINE double gamma_q(double a, double x) { return gamma_inc(a, x).q; }

}  // namespace own

// What the rest of Laplace calls: the own versions (docs/numerics.md).
LAPLACE_HD_FORCEINLINE double gamma_p(double a, double x) { return own::gamma_inc(a, x).p; }
LAPLACE_HD_FORCEINLINE double gamma_q(double a, double x) { return own::gamma_inc(a, x).q; }
LAPLACE_HD_FORCEINLINE double log_gamma_p(double a, double x) { return own::gamma_inc(a, x).log_p; }
LAPLACE_HD_FORCEINLINE double log_gamma_q(double a, double x) { return own::gamma_inc(a, x).log_q; }

}  // namespace laplace::math
