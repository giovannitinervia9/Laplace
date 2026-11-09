// Polygamma functions psi^(n)(x) of order n = 0..6 for x > 0, Laplace's own
// (Boost.Math has no GPU version of polygamma for n >= 2; digamma and trigamma
// are measured against Boost's in docs/numerics.md).
//
// Method. For x below a threshold X0 the argument is shifted up with the
// recurrence
//     psi^(n)(x) = psi^(n)(x + m) + (-1)^(n+1) n! sum_{j=0}^{m-1} (x + j)^-(n+1),
// and at z = x + m >= X0 the asymptotic expansion is used:
//     psi(z)       ~ log z - 1/(2z) - sum_k B_2k / (2k z^2k),
//     psi^(n)(z)   ~ (-1)^(n+1) z^-n [ (n-1)! + n!/(2z)
//                                      + sum_k B_2k (2k+n-1)!/(2k)! z^-2k ],  n >= 1,
// with B_2k the Bernoulli numbers. For n >= 1 every term has the sign of the
// result, so the sum is accumulated from the smallest term (the asymptotic
// part) to the largest (j = 0) without cancellation. For n = 0 the sum and the
// logarithm have opposite signs near the positive root x0 = 1.4616...: there
// the relative error is unbounded and the absolute error stays at a few ulps.
//
// Domain: x > 0 (what the distributions need). x <= 0 or NaN gives NaN; the
// result overflows to infinity like the function itself for tiny x and large n.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::math {

namespace own {

LAPLACE_HD double polygamma(int n, double x) {
  // B_2k for k = 1..10
  constexpr double B[10] = {1.0 / 6.0,           -1.0 / 30.0,        1.0 / 42.0,
                            -1.0 / 30.0,         5.0 / 66.0,         -691.0 / 2730.0,
                            7.0 / 6.0,           -3617.0 / 510.0,    43867.0 / 798.0,
                            -174611.0 / 330.0};
  constexpr double factorial[8] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0};
  constexpr double X0 = 15.0;
  constexpr int K = 10;

  if (!(x > 0.0) || n < 0 || n > 6) return detail::quiet_nan();

  // shift x up to z >= X0
  int m = 0;
  if (x < X0) m = static_cast<int>(X0 - x) + 1;
  const double z = x + static_cast<double>(m);
  const double iz = 1.0 / z;
  const double w = iz * iz;

  double result;
  if (n == 0) {
    // sum_k B_2k / (2k) w^k by Horner, highest k first
    double s = B[K - 1] / (2.0 * K);
    for (int k = K - 1; k >= 1; --k) s = s * w + B[k - 1] / (2.0 * k);
    s *= w;
    result = math::log(z) - 0.5 * iz - s;
    for (int j = m - 1; j >= 0; --j) result -= 1.0 / (x + static_cast<double>(j));
    return result;
  }

  // c_k = B_2k (2k+1)(2k+2)...(2k+n-1)
  auto coef = [&](int k) {
    double c = B[k - 1];
    for (int i = 1; i <= n - 1; ++i) c *= static_cast<double>(2 * k + i);
    return c;
  };
  double s = coef(K);
  for (int k = K - 1; k >= 1; --k) s = s * w + coef(k);
  s *= w;
  // magnitude |psi^(n)|; the sign (-1)^(n+1) is applied at the end. For large
  // z the factor z^-n is subnormal: the bracket is scaled up by 2^600 first so
  // that only the last product rounds in the subnormal range.
  const bool big = z > 1e30;
  result = factorial[n - 1] + 0.5 * factorial[n] * iz + s;
  if (big) result *= 4.149515568880993e+180;   // 2^600
  for (int i = 0; i < n; ++i) result *= iz;
  if (big) result *= 2.409919865102884e-181;   // 2^-600
  for (int j = m - 1; j >= 0; --j) {
    const double t = 1.0 / (x + static_cast<double>(j));
    double p = t;
    for (int i = 0; i < n; ++i) p *= t;
    result += factorial[n] * p;
  }
  return (n % 2 == 1) ? result : -result;
}

LAPLACE_HD_FORCEINLINE double digamma(double x) { return polygamma(0, x); }
LAPLACE_HD_FORCEINLINE double trigamma(double x) { return polygamma(1, x); }

}  // namespace own

}  // namespace laplace::math
