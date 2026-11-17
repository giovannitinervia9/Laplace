// Scalar bijector yeo_johnson (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
// Yeo-Johnson, with exponent lambda = p1 (any real; NaN gives NaN), p2 unused:
//   g(v) = ((1 + v)^lambda - 1)/lambda             v >= 0   (Box-Cox of 1 + v)
//   g(v) = -((1 - v)^(2 - lambda) - 1)/(2 - lambda) v < 0   (minus Box-Cox of 1 - v)
//   h = g^-1: u >= 0 gives v = (1 + lambda u)^(1/lambda) - 1, u < 0 gives
//   v = 1 - (1 - (2 - lambda) u)^(1/(2 - lambda)).
// Range of g: R for 0 <= lambda <= 2; for lambda < 0 the positive branch is bounded
// above by -1/lambda (the image is (-inf, -1/lambda)); for lambda > 2 the negative
// branch is bounded below by 1/(2 - lambda) (the image is (1/(2 - lambda), inf)).
// The domain of g is R, so the distances of h(u) to the bounds of its range are +inf.
//
// Every form is written from exprel and log1prel (laplace/math/exprel.hpp,
// laplace/math/log1prel.hpp) and their derivatives, with M = log1p(v) and
// N = log1p(-v) (Kokkos::log1p, so that v tiny is not rounded away), c = 2 - lambda,
// and the inverse on the negative branch in a = -u:
//   forward, v >= 0:  g = M exprel(lambda M); g^(n) = prod_(k=1..n-1)(lambda - k) (1+v)^(lambda - n)
//   forward, v < 0:   g = -N exprel(c N);    g^(n) = (-1)^(n-1) prod_(k=0..n-2)(1 - lambda - k) (1-v)^(c - n)
//   inverse, u >= 0:  h = expm1(u log1prel(lambda u)) (no cancellation: the value is
//                     expm1 of a product, and log1prel is the series near 0);
//                     h^(n) = prod_(k=1..n-1)(1 - k lambda) exp(u (1 - n lambda) log1prel(lambda u))
//   inverse, u < 0:   h = -expm1(a log1prel(c a)); h^(n) = prod_(k=0..n-2)(1 - lambda + k c)
//                     exp(u (1 - lambda + (n-1) c) log1prel(c a))
//   log|g'|: (lambda - 1) M (v >= 0), (1 - lambda) N (v < 0); their x-derivatives
//                     (lambda - 1)(-1)^(n-1)(n-1)! (1+v)^(-n) and -(1 - lambda)(n-1)!(1-v)^(-n)
//   log|h'|: (1 - lambda) u log1prel(lambda u) (u >= 0), (1 - lambda) u log1prel(c a) (u < 0);
//                     their u-derivatives (1 - lambda)(-1)^(n-1)(n-1)! lambda^(n-1)(1+lambda u)^(-n)
//                     and (1 - lambda)(n-1)! c^(n-1)(1 - c u)^(-n)
// The lambda-derivatives are SymPy's chain rule on these forms: the derivative of exprel
// is exprel_d1, that of log1prel is log1prel_d1, and so on to order 6; the derivatives of
// log1prel that one function needs (all at one argument per branch) come from one call of
// m::detail_log1prel::derivs_upto, which gives every order in one pass. Every form is
// regular at lambda = 0 and at lambda = 2, so no limit is taken; the function is
// a function of lambda with no branch at 0 or 2. The forms and the checks are
// in tools/reference/bijector_yeo_johnson_forms.py (SymPy 1.14.0), which writes this file.
//
// At v = 0 (u = 0) the forward and inverse take the branch v >= 0, whose
// derivatives at 0 are the right-hand ones: g''' differs from the left-hand one
// by a jump (the kink, declared by forward_kinks()), g and its first two
// derivatives are continuous. The log-Jacobians are continuous at 0 up to
// their first derivative.
//
// At an infinite argument every form above is NaN, exprel and log1prel being
// evaluated at an infinite argument themselves, while the limit exists; and at
// the finite end of the image, where there is one, the forms reach the limit
// only because 1 + lambda u cancels to exactly zero, which an FMA prevents. So
// every quantity takes its limit from a guard at both kinds of end: the value
// and the x-derivatives from yj_fwd_at_inf, yj_inv_at_inf, yj_fwd_lj_at_inf,
// yj_inv_lj_at_inf, yj_inv_at_edge and yj_inv_lj_at_edge, and the
// lambda-derivatives (the _partials functions) from the four
// yj_*_partials_at_inf and from yj_inv_partials_at_edge and
// yj_inv_lj_partials_at_edge, with the derivation and the limits below.
// g(+inf) and g(-inf) are the two ends of the image of g. Nothing is NaN at
// an end of either domain for a valid lambda.
//
// Invalid parameters (lambda NaN or infinite): every function returns NaN; the
// bounds of the domains are NaN (the dispatch then rejects every argument).
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/log1prel.hpp>

namespace laplace::bijector::detail_scalar {

LAPLACE_HD_FORCEINLINE bool yeo_johnson_valid(double p1) { return p1 - p1 == 0.0; }  // finite

// w^(a + k) for w > 0, a = lambda or -lambda and k an integer (|k| <= 6): pow_pos(w, a) multiplied
// or divided by w |k| times, so that the exponent a + k (lambda - 2 or 2 - lambda, say) is not
// rounded: its rounding error, eps |a + k| at most, would be multiplied by log w (by 41 at
// w = 1e18: 20 ulpC in g'' measured on 2026-11-12). Where the products overflow or underflow
// while the power does not, pow_pos(w, a + k).
LAPLACE_HD_FORCEINLINE double yj_pow(double w, double a, int k) {
  double r = pow_pos(w, a);
  for (int i = 0; i < k; ++i) r *= w;
  for (int i = 0; i < -k; ++i) r /= w;
  if (!(r > 0.0) || !Kokkos::isfinite(r)) return pow_pos(w, a + k);
  return r;
}

// exprel^(J)(y) at y = s log1p(X), s = a + k as in yj_pow. For 8 <= |y| < 700 and |X| >= 1, by the
// forward recursion of exprel (the method of m::exprel_dJ there) started from E = (1 + X)^s by
// yj_pow instead of exp(y): the rounding of log1p(X), of s and of the product s log1p(X) is then
// not amplified by |y| (at v = -1.15e18, lambda = 1e-9 it gave 26 ulpC in d_lambda g). For
// |X| >= 1 the rounding of 1 + X is within the perturbation of X that ulpC allows. Elsewhere
// m::exprel_dJ(y).
template <int J>
LAPLACE_HD double yj_exprel_d(double y, double X, double a, int k) {
  const double ay = Kokkos::fabs(y);
  if (ay >= 8.0 && ay < 700.0 && Kokkos::fabs(X) >= 1.0) {
    const double E = yj_pow(1.0 + X, a, k);
    double p = (E - 1.0) / y;
    for (int j = 1; j <= J; ++j) p = (E - j * p) / y;
    return p;
  }
  if constexpr (J == 0) return m::exprel(y);
  if constexpr (J == 1) return m::exprel_d1(y);
  if constexpr (J == 2) return m::exprel_d2(y);
  if constexpr (J == 3) return m::exprel_d3(y);
  if constexpr (J == 4) return m::exprel_d4(y);
  if constexpr (J == 5) return m::exprel_d5(y);
  if constexpr (J == 6) return m::exprel_d6(y);
}

// ------------------------------------------- a huge argument while lambda u stays small
//
// The rescaled route below starts at |lambda u| = 1e3 and cannot reach the other side of the
// forms' loss: where lambda is so small that lambda u stays small even at a huge u, the forms
// carry a power of the argument in front of a bracket in the psi^(j)(z) which cancels as z goes
// to zero, so the power is infinite, the bracket is zero, and the product is NaN again.
// Measured on 2026-11-17 for the log-Jacobian's partials: at lambda = 0 the entries start to
// fail from |x| = 1e103 at order six and from 1e154 at orders four and five, and at x = 1e300
// order six returns NaN where the exact values are 0 at n1 = 2 and -3.6e301 at n1 = 3, order
// five where it is 4 at n1 = 2. Most of those are finite, so returning signed infinities would
// be wrong, and the rescaled route cannot serve either, its q_j being a cancelling difference at
// small z. The band is narrow: for lambda above about 1e-200 nothing fails anywhere, lambda u
// then reaching 1e3 before a power of u can overflow.
//
// What this route uses is that the power of u in each term does not depend on the summation
// index. Writing S = u psi(lambda u) with psi = log1prel and z = lambda u, Leibniz in u gives
//
//   d^n0_u S^(j) = sum_m C(n0, m) [d^(n0-m)_u u^(j+1)] lambda^m psi^(j+m)(z)
//
// and u^(j+1-n0+m) lambda^m = u^(j+1-n0) z^m, one power for the whole sum:
//
//   d^n0_u S^(j) = u^(j+1-n0) sum_(m=m0..n0) C(n0,m) (j+1)!/(j+1-n0+m)! z^m psi^(j+m)(z)
//
// with m0 = max(0, n0 - j - 1), which keeps every factorial's argument non-negative. The sum is
// bounded at small z, every term being a small power of z times a derivative of psi, so it is
// taken first and the single power applied once at the end. A sum that is exactly zero returns
// zero without that multiplication: at z = 0 it is zero exactly when the m = 0 term is absent,
// and a power of u that has overflowed would otherwise turn an exact zero into NaN. That is the
// whole of the repair; the identity is checked against SymPy's own derivative for every (j, n0)
// with j + n0 <= 6 by check_dS_small_z in this file's script.
//
// The entries of the log-Jacobian follow from log|h'| = (1 - lambda) S:
//
//   d^n0_u d^n1_lambda log|h'| = (1 - lambda) d^n0_u S^(n1) - n1 d^n0_u S^(n1-1).
//
// The route is taken for |u| above 1e50, fifty decades below the measured onset and far above
// any argument a model produces, and below that the forms are kept: over the overlap the two
// agree to 2.3e-16 relative (measured for lambda from 0 to 1.9 and |x| from 1 to 1e140, every
// order and every entry). The map's own partials need nothing here: at lambda = 0 the inverse is
// exp(u) - 1, already an infinity for every |u| above about 710, so the band holds no finite
// value of theirs to lose (measured 0 of 7 NaN at every place in it).

LAPLACE_HD double yj_fact_small(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= double(i);
  return f;
}

// d^n0_u S^(j) by the identity above; lr[k] = psi^(k)(z) for k = 0..j+n0
LAPLACE_HD double yj_dS_small_z(int j, int n0, double z, double u, const double* lr) {
  const int m0 = (n0 - j - 1 > 0) ? (n0 - j - 1) : 0;
  double acc = 0.0;
  for (int mm = m0; mm <= n0; ++mm) {
    double zm = 1.0;
    for (int i = 0; i < mm; ++i) zm *= z;
    const double cm = yj_fact_small(n0) / (yj_fact_small(mm) * yj_fact_small(n0 - mm));
    acc += cm * (yj_fact_small(j + 1) / yj_fact_small(j + 1 - n0 + mm)) * zm * lr[j + mm];
  }
  if (acc == 0.0) return 0.0;  // exact, and a power of u that has overflowed must not reach it
  const int e = j + 1 - n0;
  double p = 1.0;
  for (int i = 0; i < (e >= 0 ? e : -e); ++i) p *= u;
  return (e >= 0) ? acc * p : acc / p;
}

// out[n1] = d^(K-n1)_u d^n1_lambda [(1 - lambda) log(1 + lambda u)/lambda] for n1 = 0..K, valid
// wherever 1 + lambda u > 0, so for the positive branch of Yeo-Johnson and the whole of Box-Cox.
// Yeo-Johnson's negative branch is this at (2 - lambda, -u) with every entry times (-1)^K, the
// sign being (-1)^(n0+n1) and n0 + n1 = K.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_small_z(double lam, double u, double* out) {
  const double z = lam * u;
  double lr[K + 1];
  lr[0] = m::log1prel(z);
  m::detail_log1prel::derivs_upto(K, z, lr);
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    double v = (1.0 - lam) * yj_dS_small_z(n1, n0, z, u, lr);
    if (n1 >= 1) v -= double(n1) * yj_dS_small_z(n1 - 1, n0, z, u, lr);
    out[n1] = v;
  }
}

// ---------------------------------------------------------------- a large finite argument
//
// The lambda-derivatives of the inverse are written below as a polynomial in the derivatives of
// log1prel multiplied by a power of the argument, and above |lambda u| of about 1e3 that shape
// loses the value: psi^(j)(z) decays like j! log(1+z)/z^(j+1) and underflows while the power
// z^(j+1) in front of it overflows. Measured on 2026-11-17: at lambda = 1/2 and u = 1e60,
// log1prel_d6 is exactly 0 and z^7 is infinite, so the form returned inf * 0 = NaN for a value
// whose true size is 7.36e135 (mpmath at 60 digits). Regrouping the polynomial does not fix it,
// because its terms genuinely span hundreds of orders of magnitude: two attempts that only
// moved the powers around left more NaN than they removed, not fewer.
//
// What works is to compute the whole derivative from its structure, with the dominant scale
// factored out and every intermediate bounded:
//
//   h^(n0) = C_n0(lambda) exp(A_n0),   A_n0 = (1 - n0 lambda) S,   S = log(1 + lambda u)/lambda
//   d^n1_lambda [C exp(A)] = exp(A) sum_i binom(n1, i) C^(i) Y_(n1-i)(A', ..., A^(n1-i))
//
// with Y the complete Bell polynomials, A^(j) = (1 - n0 lambda) S^(j) - n0 j S^(j-1), and
//
//   S^(j) = q_j/lambda^(j+1),   q_j = z^(j+1) psi^(j)(z) = (-1)^j j! [log(1+z) - sum_(k=1..j) r^k/k]
//
// where r = z/(1+z). That closed form for q_j is proved equal to the derivative by SymPy for
// j = 0 to 7 (check_scaled_log1prel in this file's script); it is bounded, growing only as a
// logarithm, so S^(j) and the Bell polynomials stay representable and the only large number is
// the exp at the end. At lambda = 1/2 and u = 1e60 the chain runs q_6 ~ 1e5, S^(6) ~ 1e7,
// Y_6 ~ 1e16, exp(A) ~ 1e119, product 7.36e135, which is the value to eight digits.
//
// Below |z| = 1e3 the forms are exact and this route is not taken: q_j there is a difference of
// two nearly equal quantities and would lose about two digits at z = 1. The two agree to 1e-15
// or better over the whole overlap (measured from u = 1 to u = 1e20).

LAPLACE_HD double yj_binom(int n, int k) {
  double b = 1.0;
  for (int i = 0; i < k; ++i) b = b * double(n - i) / double(i + 1);
  return b;
}

// q[j] = z^(j+1) psi^(j)(z) for j = 0..n, by the closed form above
LAPLACE_HD void yj_log1prel_scaled(int n, double z, double* q) {
  const double L = Kokkos::log1p(z);
  const double r = z / (1.0 + z);
  double fact = 1.0, rk = 1.0, sum = 0.0;
  for (int j = 0; j <= n; ++j) {
    if (j >= 1) {
      fact *= double(j);
      rk *= r;
      sum += rk / double(j);
    }
    q[j] = ((j % 2) ? -fact : fact) * (L - sum);
  }
}

// out[n1] = d^(K-n1)_u d^n1_lambda h for n1 = 0..K, on the POSITIVE branch (u >= 0, lambda != 0,
// which |z| > 1e3 guarantees). Every entry carries at least one derivative, K being 1 or more, so
// the -1 of h = exp(S) - 1 never appears here and the same routine serves any map whose inverse
// is exp(S) times a constant: Box-Cox calls it unchanged for its own column. The negative branch is this one reflected, h(u) = -h_pos(-u; c)
// with c = 2 - lambda, so its entries are these at (-u, c) times (-1)^(K+1), the sign being the
// same for every n1 since n0 + n1 = K.
template <int K>
LAPLACE_HD void yj_inv_partials_rescaled_pos(double lam, double u, double* out) {
  double q[K + 1];
  yj_log1prel_scaled(K, lam * u, q);
  const double S = q[0] / lam;
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    double Sd[K + 2], Ad[K + 2], Y[K + 2], Cd[K + 2];
    for (int j = 0; j <= K + 1; ++j) Sd[j] = Ad[j] = Y[j] = Cd[j] = 0.0;
    double lp = lam;
    for (int j = 1; j <= n1; ++j) {
      lp *= lam;
      Sd[j] = q[j] / lp;
    }
    for (int j = 1; j <= n1; ++j)
      Ad[j] = (1.0 - double(n0) * lam) * Sd[j] - double(n0) * double(j) * (j == 1 ? S : Sd[j - 1]);
    Y[0] = 1.0;
    for (int mm = 0; mm < n1; ++mm) {
      double acc = 0.0;
      for (int j = 0; j <= mm; ++j) acc += yj_binom(mm, j) * Ad[j + 1] * Y[mm - j];
      Y[mm + 1] = acc;
    }
    // C_n0 = prod_(k=1..n0-1)(1 - k lambda) and its lambda-derivatives, one factor at a time
    Cd[0] = 1.0;
    int d = 0;
    for (int k = 1; k <= n0 - 1; ++k) {
      const double f = 1.0 - double(k) * lam, fp = -double(k);
      for (int i = d + 1; i >= 0; --i) {
        double v = (i <= d) ? Cd[i] * f : 0.0;
        if (i >= 1 && i - 1 <= d) v += double(i) * Cd[i - 1] * fp;
        Cd[i] = v;
      }
      ++d;
    }
    double acc = 0.0;
    for (int i = 0; i <= n1; ++i) acc += yj_binom(n1, i) * Cd[i] * Y[n1 - i];
    out[n1] = acc * m::exp((1.0 - double(n0) * lam) * S);
  }
}

template <int K>
LAPLACE_HD void yj_inv_partials_rescaled(double p1, double x, double* out) {
  if (x >= 0.0) {
    yj_inv_partials_rescaled_pos<K>(p1, x, out);
    return;
  }
  yj_inv_partials_rescaled_pos<K>(2.0 - p1, -x, out);
  if ((K + 1) % 2) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = -out[n1];
  }
}

// d^K_lambda [(1 - lambda) log(1 + lambda u)/lambda] = (1 - lambda) S^(K) - K S^(K-1): the only
// entry of the log-Jacobian's partials that loses its value at a large finite argument, every
// other carrying a power of 1 + lambda u instead of S and staying bounded. It holds wherever
// 1 + lambda u > 0, so it serves the positive branch of Yeo-Johnson and the whole of Box-Cox;
// Yeo-Johnson's negative branch calls it at (2 - lambda, -u) and takes (-1)^K.
template <int K>
LAPLACE_HD double yj_inv_lj_pure_lambda_rescaled(double lam, double u) {
  double q[K + 1];
  yj_log1prel_scaled(K, lam * u, q);
  double lp = 1.0, sd[K + 2];
  for (int j = 0; j <= K + 1; ++j) sd[j] = 0.0;
  for (int j = 0; j <= K; ++j) {
    lp *= lam;
    sd[j] = q[j] / lp;  // S^(j)
  }
  // d^K_lambda [(1 - lambda) S] = (1 - lambda) S^(K) - K S^(K-1)
  double v = (1.0 - lam) * sd[K];
  if (K >= 1) v -= double(K) * sd[K - 1];
  return v;
}

// ---------------------------------------------------------------- the limits at an infinite argument
//
// Every form above is written from exprel or log1prel, whose argument is itself infinite at an
// infinite x: exprel(inf) and log1prel(inf) are inf/inf, so the form gives NaN where the limit
// exists. The four functions below return the limit instead, and each quantity calls its own at
// the top. The argument enters every closed form only through a power of 1 + v, 1 - v,
// 1 + lambda u or 1 - c u, c = 2 - lambda, so the sign of that power's exponent decides: above
// zero the power runs to +inf and the prefactor gives the sign, at zero the power is 1 and the
// prefactor is the whole limit, below zero the power dies and the limit is 0. The limits and
// their derivation are in tools/reference/bijector_yeo_johnson_forms.py (check_limits), which
// compares each of them with SymPy's limit of the definition.
//
// The exponent is never formed: it vanishes exactly where a product of lambda with a small
// integer equals 1 (or where lambda equals the integer), so the two are compared instead of
// subtracted. A subtraction is what nvcc contracts into an FMA, and the exact product
// 3 (1/3) is 1 - 2^-54, not 1, so the knife edge would be reached on one platform and not on
// the other. Where the comparison fails the limit is +-inf or 0, which is the limit for that
// lambda: only lambda exactly 1/n carries a finite limit of order n.
//
// The derivation and every limit are in tools/reference/bijector_yeo_johnson_forms.py
// (check_limits), which compares each of them with SymPy's limit of the definition.

// The limit of pref w^e as w -> +inf: above is true where e > 0, and pref is then never zero
// in any of the uses below; an exponent known to vanish returns pref before this is called.
LAPLACE_HD_FORCEINLINE double yj_limit_power(double pref, bool above) {
  if (above) return pref > 0.0 ? inf() : -inf();
  return 0.0;
}

// g and its v-derivatives at v = +inf (upper) and at v = -inf. The value is the end of the
// image of g: -1/lambda below 0 and 1/(2 - lambda) above 2, infinite otherwise. For n >= 1
//   g^(n) = prod_(k=1..n-1)(lambda - k) (1 + v)^(lambda - n)                     v > 0
//   g^(n) = (-1)^(n-1) prod_(k=1..n-1)(c - k) (1 - v)^(c - n)                    v < 0
// and the exponent lambda - n (or c - n) decides, the prefactor being (-1)^(n-1) (n-1)! where
// it vanishes.
template <int K>
LAPLACE_HD double yj_fwd_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    return upper ? (p1 < 0.0 ? -1.0 / p1 : inf()) : (p1 > 2.0 ? 1.0 / (2.0 - p1) : -inf());
  }
  const double a = upper ? p1 : 2.0 - p1;  // lambda, or c = 2 - lambda
  double pref = (upper || (K % 2) == 1) ? 1.0 : -1.0;
  for (int k = 1; k <= K - 1; ++k) pref *= a - k;
  if (a == double(K)) return pref;
  return yj_limit_power(pref, a > double(K));
}

// h and its u-derivatives at u = +inf (upper) and at u = -inf. The range of h is R, so the
// value is infinite with the sign of u; u = +inf belongs to the domain of h only for
// lambda >= 0 and u = -inf only for lambda <= 2, the dispatch rejecting the other end. For
// n >= 1
//   h^(n) = prod_(k=1..n-1)(1 - k lambda) (1 + lambda u)^(1/lambda - n)          u > 0
//   h^(n) = prod_(k=1..n-1)(k c - 1) (1 - c u)^(1/c - n)                         u < 0
// and 1 - n lambda (or 1 - n c) has the sign of the exponent, zero included; lambda = 0 above,
// where h = e^u - 1, and lambda = 2 below, where h = 1 - e^(-u), fall under its positive case.
template <int K>
LAPLACE_HD double yj_inv_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) return upper ? inf() : -inf();
  const double a = upper ? p1 : 2.0 - p1;  // lambda, or c = 2 - lambda
  double pref = 1.0;
  for (int k = 1; k <= K - 1; ++k) pref *= upper ? 1.0 - k * a : k * a - 1.0;
  const double ka = double(K) * a;
  if (ka == 1.0) return pref;
  return yj_limit_power(pref, ka < 1.0);
}

// log|g'| and its v-derivatives at an infinite v: (lambda - 1) log(1 + v) above and
// (1 - lambda) log(1 - v) below run to an infinity of the sign of lambda - 1 (of 1 - lambda)
// and are 0 at lambda = 1, where g is the identity. Every v-derivative carries (1 + v)^(-n) or
// (1 - v)^(-n) and dies.
template <int K>
LAPLACE_HD double yj_fwd_lj_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    const double s = upper ? p1 - 1.0 : 1.0 - p1;
    return s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
  }
  static_cast<void>(upper);
  return 0.0;
}

// log|h'| and its u-derivatives at an infinite u: (1/lambda - 1) log(1 + lambda u) above and
// (1/c - 1) log(1 - c u) below run to an infinity of the sign of 1 - lambda (of lambda - 1) and
// are 0 at lambda = 1, where h is the identity; lambda = 0 above, where log|h'| = u, and
// lambda = 2 below, where log|h'| = -u, fall under the positive case. Every u-derivative
// carries (1 + lambda u)^(-n) or (1 - c u)^(-n) and dies, except on those two edges, where
// log|h'| is affine in u: the first derivative is then its slope, 1 above and -1 below.
template <int K>
LAPLACE_HD double yj_inv_lj_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    const double s = upper ? 1.0 - p1 : p1 - 1.0;
    return s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
  }
  if constexpr (K == 1) return upper ? (p1 == 0.0 ? 1.0 : 0.0) : (p1 == 2.0 ? -1.0 : 0.0);
  static_cast<void>(upper);
  return 0.0;
}

// ---- the mixed derivatives at an infinite argument
//
// out[n1] is d^(K - n1)_x d^n1_lambda, n1 = 0..K: the entries of bijector_partials<K> with
// n2 = 0, which are the only ones this map writes (it has no second parameter). A
// lambda-derivative multiplies a closed form by a logarithm of the argument, which loses to a
// negative power of it and wins against a vanishing exponent, so the sign of the exponent
// decides as it does for the pure derivatives: above zero an infinity, below zero zero, and at
// zero the prefactor for n1 = 0 and an infinity above it. The entries at n1 = 0 agree with
// the pure functions above, which the reference script checks.
//
// The inverse's negative branch is its positive branch reflected: h(u) = -h_pos(-u; c) and
// log|h'|(u) = log|h'_pos|(-u; c) with c = 2 - lambda, so d^n0_u d^n1_lambda carries
// (-1)^(n0+n1+1) for h and (-1)^(n0+n1) for its log-Jacobian, with c in place of lambda.
//
// A factorial and an integer power with a run-time order: the loops run over n1, and this
// path is taken only at an infinite argument.
LAPLACE_HD_FORCEINLINE double yj_fact_at(int n) {
  double f = 1.0;
  for (int k = 2; k <= n; ++k) f *= double(k);
  return f;
}
LAPLACE_HD_FORCEINLINE double yj_ipow_at(double z, int n) {
  double p = 1.0;
  for (int k = 0; k < n; ++k) p *= z;
  return p;
}

// g: prod_(k=1..n0-1)(a - k) (1 +- v)^(a - n0), a = lambda above and c = 2 - lambda below,
// with (-1)^(n0-1) on the negative branch and (-1)^n1 from d^n1_lambda = (-1)^n1 d^n1_c there.
// At n0 = 0 the value tends to an end of the image, whose lambda-derivatives are those of
// -1/lambda (above, below lambda = 0) and of 1/(2 - lambda) (below, above lambda = 2).
template <int K>
LAPLACE_HD void yj_fwd_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    if (n0 == 0) {
      if (upper) {
        out[n1] = p1 < 0.0 ? ((n1 % 2 ? 1.0 : -1.0) * yj_fact_at(n1) / yj_ipow_at(p1, n1 + 1)) : inf();
      } else {
        out[n1] = p1 > 2.0 ? (yj_fact_at(n1) / yj_ipow_at(2.0 - p1, n1 + 1))
                           : yj_limit_power(n1 % 2 ? 1.0 : -1.0, true);
      }
      continue;
    }
    double pref = (upper || ((n0 - 1 + n1) % 2) == 0) ? 1.0 : -1.0;
    for (int k = 1; k <= n0 - 1; ++k) pref *= a - double(k);
    if (a == double(n0) && n1 == 0) {
      out[n1] = pref;
      continue;
    }
    out[n1] = yj_limit_power(pref, a >= double(n0));
  }
}

// h: prod_(k=1..n0-1)(1 - k a) (1 + a |u|)^(1/a - n0), a = lambda above and c below, the sign
// of 1 - n0 a being the sign of the exponent (lambda = 0 above and lambda = 2 below included).
template <int K>
LAPLACE_HD void yj_inv_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    double v = 1.0;
    for (int k = 1; k <= n0 - 1; ++k) v *= 1.0 - double(k) * a;
    if (n1 % 2) v = -v;                                 // (-1)^n1 from the lambda-derivatives
    if (!upper && ((n0 + n1) % 2) == 0) v = -v;         // (-1)^(n0+n1+1) from the reflection
    const double na = double(n0) * a;
    out[n1] = (na == 1.0 && n1 == 0) ? v : yj_limit_power(v, na <= 1.0);
  }
}

// log|g'| is (lambda - 1) log(1 + v) above and (1 - lambda) log(1 - v) below: linear in
// lambda, so only n1 = 0 and n1 = 1 are not identically zero, and every v-derivative carries
// a negative power of 1 +- v and dies.
template <int K>
LAPLACE_HD void yj_fwd_lj_partials_at_inf(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    if (K - n1 >= 1) {
      out[n1] = 0.0;
    } else if (n1 == 0) {
      const double s = upper ? p1 - 1.0 : 1.0 - p1;
      out[n1] = s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
    } else if (n1 == 1) {
      out[n1] = upper ? inf() : -inf();
    } else {
      out[n1] = 0.0;
    }
  }
}

// log|h'| above is Psi = (1 - lambda) log(1 + lambda u)/lambda. For lambda > 0 a u-derivative
// carries (1 + lambda u)^(-n0) and dies, while a pure lambda-derivative multiplies the
// logarithm by d^n1_lambda[(1 - lambda)/lambda], of sign (-1)^n1. At lambda = 0, Psi is the
// series (1 - lambda) sum_j (-1)^j u^(j+1) lambda^j/(j + 1), so d^n1_lambda Psi is a polynomial
// in u of degree n1 + 1 with leading coefficient (-1)^n1 n1!/(n1 + 1): its n0-th u-derivative
// runs to (-1)^n1 inf below n1 + 1, is the constant (-1)^n1 (n1!)^2 at n0 = n1 + 1, and
// vanishes above.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    double v;
    if (a == 0.0) {
      if (n0 < n1 + 1) {
        v = (n1 % 2) ? -inf() : inf();
      } else if (n0 == n1 + 1) {
        v = ((n1 % 2) ? -1.0 : 1.0) * yj_fact_at(n1) * yj_fact_at(n1);
      } else {
        v = 0.0;
      }
    } else if (n0 >= 1) {
      v = 0.0;
    } else if (n1 == 0) {
      v = a == 1.0 ? 0.0 : (a < 1.0 ? inf() : -inf());
    } else {
      v = (n1 % 2) ? -inf() : inf();
    }
    if (!upper && ((n0 + n1) % 2)) v = -v;
    out[n1] = v;
  }
}

// h and log|h'| with their u-derivatives at the FINITE end of the domain of h. There
// w = 1 + lambda u (1 - c u below) vanishes and every derivative is a negative power of it,
// hence an infinity. On the host w comes out exactly zero and the powers give that infinity by
// themselves, so these two were right without a branch; nvcc contracts 1 + lambda u into an FMA,
// which multiplies lambda by the ROUNDED -1/lambda exactly instead of rounding the product,
// leaves w at about 4e-17 and turns the infinity into a finite 1e100 (measured 7.8e100 at order
// six, 2026-11-17). The limit is therefore returned here rather than left to the arithmetic.
// h^(n) = P_n w^(1/lambda - n), with P_n > 0 at the upper end (lambda < 0, every factor above 1)
// and of sign (-1)^(n-1) at the lower one (lambda > 2).
template <int K>
LAPLACE_HD double yj_inv_at_edge(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if (upper) return inf();
  return (K % 2) ? inf() : -inf();  // (-1)^(K-1)
}

// log|h'| = (1/lambda - 1) log w and its u-derivatives: +inf at the upper end, (-1)^K inf at
// the lower one, which are the n1 = 0 entries of yj_inv_lj_partials_at_edge below.
template <int K>
LAPLACE_HD double yj_inv_lj_at_edge(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if (upper) return inf();
  return (K % 2) ? -inf() : inf();
}

// h and its mixed derivatives at the FINITE end of the domain of h. There
// h^(n0) = P_n0 w^(1/lambda - n0) with w = 1 + lambda u vanishing and the exponent negative, so
// every entry is an infinity; each lambda-derivative multiplies by a term dominated by
// (1/lambda - n0) u/w, which is negative at the upper end (u = -1/lambda > 0 with lambda < 0,
// where P_n0 > 0 as well), so the sign alternates in n1. The reflection adds (-1)^(n0+n1+1)
// below lambda = 2 and leaves (-1)^(n0+1). Without this the entries come from the forms, where
// FMA keeps w at about 4e-17 and an infinity comes out as a finite 1e100 on the device.
template <int K>
LAPLACE_HD void yj_inv_partials_at_edge(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int e = upper ? n1 : (K - n1 + 1);
    out[n1] = (e % 2) ? -inf() : inf();
  }
}

// log|h'| and its mixed derivatives at the FINITE end of the domain of h, which is -1/lambda
// below lambda = 0 and 1/(2 - lambda) above lambda = 2. There w = 1 + lambda u (1 - c u below)
// vanishes, and d^n0_u log|h'| = (1 - lambda)(-1)^(n0-1)(n0-1)! lambda^(n0-1) w^(-n0); each
// lambda-derivative raises the power of 1/w by one and brings a factor -n0 u, so the most
// singular term is C u^n1 w^(-n0-n1) and the limit is an infinity whose sign, with u at the end
// itself, is that of (1 - lambda)(-1)^(n0-1) lambda^(n0-1-n1), namely (-1)^n1 above; below,
// log|h'|(u) = log|h'_pos|(-u; c) adds (-1)^(n0+n1) and leaves (-1)^n0.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_at_edge(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int e = upper ? n1 : K - n1;
    out[n1] = (e % 2) ? -inf() : inf();
  }
}

template <int K>
LAPLACE_HD double yeo_johnson_fwd(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  // an infinite x: the forms below give NaN there, the limit is taken instead
  if (Kokkos::isinf(x)) return yj_fwd_at_inf<K>(x > 0.0, p1);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<0>((p1) * lp0, x, p1, 0);
      return (ex0 * lp0);
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<0>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      return (-1.0 * ex0 * lp0);
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      return pw0;
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      return pw0;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -2);
      return (pw0 * (-1.0 + p1));
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      return (pw0 * (-1.0 + p1));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -3);
      return (pw0 * (-1.0 + p1) * (-2.0 + p1));
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -1);
      return (-1.0 * pw0 * p1 * (1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -4);
      return (pw0 * (-1.0 + p1) * (-3.0 + p1) * (-2.0 + p1));
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -2);
      return (pw0 * p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -5);
      return (pw0 * (-1.0 + p1) * (-4.0 + p1) * (-3.0 + p1) * (-2.0 + p1));
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -3);
      return (-1.0 * pw0 * p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)) * (-2.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double pw0 = yj_pow(1.0 + x, p1, -6);
      return (pw0 * (-1.0 + p1) * (-5.0 + p1) * (-4.0 + p1) * (-3.0 + p1) * (-2.0 + p1));
    }
    {
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -4);
      return (pw0 * p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)) * (-3.0 + (-1.0 * p1)) * (-2.0 + (-1.0 * p1)));
    }
  }
  return nan();
}

template <int K>
LAPLACE_HD void yeo_johnson_fwd_partials(double x, double p1, double p2, double* out) {
  static_assert(K >= 1 && K <= 6, "order 1 to 6");
  static_cast<void>(p2);
  for (int i = 0; i < (K + 1) * (K + 2) / 2; ++i) out[i] = 0.0;
  // an infinite x: the forms below give NaN there, the limits are taken instead
  if (Kokkos::isinf(x)) {
    yj_fwd_partials_at_inf<K>(x > 0.0, p1, out);
    return;
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<1>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      out[0] = pw0;
      out[1] = (ex0 * ipow<2>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<1>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      out[0] = pw0;
      out[1] = (ex0 * ipow<2>(lp0));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<2>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      const double pw1 = yj_pow(1.0 + x, p1, -2);
      out[0] = (pw1 * (-1.0 + p1));
      out[1] = (lp0 * pw0);
      out[2] = (ex0 * ipow<3>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<2>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      const double pw1 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      out[0] = (pw0 * (-1.0 + p1));
      out[1] = (-1.0 * lp0 * pw1);
      out[2] = (-1.0 * ex0 * ipow<3>(lp0));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<3>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      const double pw1 = yj_pow(1.0 + x, p1, -2);
      const double pw2 = yj_pow(1.0 + x, p1, -3);
      const double t0 = (-1.0 + p1);
      out[0] = (pw2 * t0 * (-2.0 + p1));
      out[1] = (pw1 + (lp0 * pw1 * t0));
      out[2] = (pw0 * ipow<2>(lp0));
      out[3] = (ex0 * ipow<4>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<3>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -1);
      const double pw1 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      const double pw2 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      const double t0 = (-1.0 + p1);
      out[0] = (pw0 * t0 * p1);
      out[1] = (pw1 + (-1.0 * lp0 * pw1 * t0));
      out[2] = (pw2 * ipow<2>(lp0));
      out[3] = (ex0 * ipow<4>(lp0));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<4>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      const double pw1 = yj_pow(1.0 + x, p1, -2);
      const double pw2 = yj_pow(1.0 + x, p1, -3);
      const double pw3 = yj_pow(1.0 + x, p1, -4);
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 + p1);
      const double t2 = (pw2 * t1);
      out[0] = (pw3 * t0 * t1 * (-3.0 + p1));
      out[1] = (t2 + (pw2 * t0) + (lp0 * t0 * t2));
      out[2] = (lp0 * pw1 * (2.0 + (lp0 * t1)));
      out[3] = (pw0 * ipow<3>(lp0));
      out[4] = (ex0 * ipow<5>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<4>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -1);
      const double pw1 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -2);
      const double pw2 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      const double pw3 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (pw0 * p1);
      out[0] = (pw1 * t1 * p1 * (-1.0 + (-1.0 * p1)));
      out[1] = (t2 + (-1.0 * pw0 * t1) + (lp0 * t1 * t2));
      out[2] = (lp0 * pw2 * (-2.0 + (lp0 * t0)));
      out[3] = (-1.0 * pw3 * ipow<3>(lp0));
      out[4] = (-1.0 * ex0 * ipow<5>(lp0));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<5>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      const double pw1 = yj_pow(1.0 + x, p1, -2);
      const double pw2 = yj_pow(1.0 + x, p1, -3);
      const double pw3 = yj_pow(1.0 + x, p1, -4);
      const double pw4 = yj_pow(1.0 + x, p1, -5);
      const double t0 = (-1.0 + p1);
      const double t1 = (-3.0 + p1);
      const double t2 = (-2.0 + p1);
      const double t3 = (t1 * t2);
      const double t4 = (pw3 * t3);
      const double t5 = (pw3 * t0);
      const double t6 = (lp0 * t0);
      const double t7 = ipow<2>(lp0);
      out[0] = (pw4 * t0 * t3 * (-4.0 + p1));
      out[1] = (t4 + (t1 * t5) + (t2 * t5) + (t4 * t6));
      out[2] = (pw2 * (2.0 + (2.0 * t6) + (2.0 * lp0 * t2) + (t0 * t2 * t7)));
      out[3] = (pw1 * t7 * (3.0 + t6));
      out[4] = (pw0 * ipow<4>(lp0));
      out[5] = (ex0 * ipow<6>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<5>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -1);
      const double pw1 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -2);
      const double pw2 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -3);
      const double pw3 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      const double pw4 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-1.0 + (-1.0 * p1));
      const double t3 = (t2 * p1);
      const double t4 = (pw1 * t1);
      const double t5 = (lp0 * p1);
      const double t6 = (lp0 * t0);
      const double t7 = ipow<2>(lp0);
      out[0] = (-1.0 * pw2 * t1 * t3 * (-2.0 + (-1.0 * p1)));
      out[1] = ((-1.0 * pw1 * t3) + (-1.0 * t4 * p1) + (pw1 * t1 * t2) + (-1.0 * t2 * t4 * t5));
      out[2] = (pw0 * (2.0 + (-2.0 * t5) + (-2.0 * t6) + (t0 * t7 * p1)));
      out[3] = (pw3 * t7 * (3.0 + (-1.0 * t6)));
      out[4] = (pw4 * ipow<4>(lp0));
      out[5] = (ex0 * ipow<6>(lp0));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double lp0 = Kokkos::log1p(x);
      const double ex0 = yj_exprel_d<6>((p1) * lp0, x, p1, 0);
      const double pw0 = yj_pow(1.0 + x, p1, -1);
      const double pw1 = yj_pow(1.0 + x, p1, -2);
      const double pw2 = yj_pow(1.0 + x, p1, -3);
      const double pw3 = yj_pow(1.0 + x, p1, -4);
      const double pw4 = yj_pow(1.0 + x, p1, -5);
      const double pw5 = yj_pow(1.0 + x, p1, -6);
      const double t0 = (-4.0 + p1);
      const double t1 = (-3.0 + p1);
      const double t2 = (-1.0 + p1);
      const double t3 = (-2.0 + p1);
      const double t4 = (t2 * t3);
      const double t5 = (t1 * t4);
      const double t6 = (pw4 * t0);
      const double t7 = (t1 * t6);
      const double t8 = (t3 * t7);
      const double t9 = (lp0 * t2);
      const double t10 = (lp0 * t3);
      const double t11 = (2.0 * t9);
      const double t12 = (t4 * ipow<2>(lp0));
      out[0] = (pw5 * t0 * t5 * (-5.0 + p1));
      out[1] = (t8 + (pw4 * t5) + (t2 * t7) + (t4 * t6) + (t8 * t9));
      out[2] = (pw3 * (-12.0 + (6.0 * p1) + (t1 * t11) + (t1 * t12) + (t11 * t3) + (2.0 * t1 * t10)));
      out[3] = (lp0 * pw2 * (6.0 + t12 + (3.0 * t10) + (3.0 * t9)));
      out[4] = (pw1 * ipow<3>(lp0) * (4.0 + t9));
      out[5] = (pw0 * ipow<5>(lp0));
      out[6] = (ex0 * ipow<7>(lp0));
      return;
    }
    {
      const double lp0 = Kokkos::log1p((-1.0 * x));
      const double ex0 = yj_exprel_d<6>(((2.0 + (-1.0 * p1))) * lp0, (-1.0 * x), (-1.0 * p1), 2);
      const double pw0 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -1);
      const double pw1 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -2);
      const double pw2 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -3);
      const double pw3 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), -4);
      const double pw4 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 0);
      const double pw5 = yj_pow(1.0 + (-1.0 * x), (-1.0 * p1), 1);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-2.0 + (-1.0 * p1));
      const double t3 = (1.0 + p1);
      const double t4 = (-1.0 * t3);
      const double t5 = (pw2 * t1 * p1);
      const double t6 = (pw2 * t2 * t4);
      const double t7 = (t1 * t6);
      const double t8 = (lp0 * p1);
      const double t9 = (lp0 * t0);
      const double t10 = (2.0 * t3);
      const double t11 = ipow<2>(lp0);
      out[0] = (pw3 * t1 * t2 * t4 * p1 * (-3.0 + (-1.0 * p1)));
      out[1] = ((-1.0 * t7) + (t2 * t5) + (t4 * t5) + (t6 * p1) + (t7 * t8));
      out[2] = (pw1 * ((6.0 * p1) + (-1.0 * t10 * t8) + (-1.0 * t10 * t9) + (-2.0 * t9 * p1) + (t0 * t11 * t3 * p1)));
      out[3] = (lp0 * pw0 * (-6.0 + (3.0 * lp0 * t0) + (3.0 * lp0 * p1) + (-1.0 * t0 * t11 * p1)));
      out[4] = (pw4 * ipow<3>(lp0) * (-4.0 + t9));
      out[5] = (-1.0 * pw5 * ipow<5>(lp0));
      out[6] = (-1.0 * ex0 * ipow<7>(lp0));
      return;
    }
  }
}

template <int K>
LAPLACE_HD double yeo_johnson_inv(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  // an infinite x: the forms below give NaN there, the limit is taken instead
  if (Kokkos::isinf(x)) return yj_inv_at_inf<K>(x > 0.0, p1);
  // the finite end of the domain of h, where the forms give a large finite value
  // under FMA contraction instead of the infinity the limit has
  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1)))
    return yj_inv_at_edge<K>(p1 < 0.0, p1);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return Kokkos::expm1((lr[0] * x));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return (-1.0 * Kokkos::expm1((-1.0 * lr[0] * x)));
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return m::exp((lr[0] * x * (1.0 + (-1.0 * p1))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return m::exp((lr[0] * x * (1.0 + (-1.0 * p1))));
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * m::exp((lr[0] * x * (1.0 + (-2.0 * p1)))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * m::exp((lr[0] * x * (3.0 + (-2.0 * p1)))));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((lr[0] * x * (1.0 + (-3.0 * p1)))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * m::exp((lr[0] * x * (5.0 + (-3.0 * p1)))));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((lr[0] * x * (1.0 + (-4.0 * p1)))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * m::exp((lr[0] * x * (7.0 + (-4.0 * p1)))));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-4.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((lr[0] * x * (1.0 + (-5.0 * p1)))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * (7.0 + (-4.0 * p1)) * m::exp((lr[0] * x * (9.0 + (-5.0 * p1)))));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-5.0 * p1)) * (1.0 + (-4.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((lr[0] * x * (1.0 + (-6.0 * p1)))));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * (7.0 + (-4.0 * p1)) * (9.0 + (-5.0 * p1)) * m::exp((lr[0] * x * (11.0 + (-6.0 * p1)))));
    }
  }
  return nan();
}

template <int K>
LAPLACE_HD void yeo_johnson_inv_partials(double x, double p1, double p2, double* out) {
  static_assert(K >= 1 && K <= 6, "order 1 to 6");
  static_cast<void>(p2);
  for (int i = 0; i < (K + 1) * (K + 2) / 2; ++i) out[i] = 0.0;
  // an infinite x: the forms below give NaN there, the limits are taken instead
  if (Kokkos::isinf(x)) {
    yj_inv_partials_at_inf<K>(x > 0.0, p1, out);
    return;
  }
  // a large finite argument: the forms lose the value, the rescaled route keeps it
  if (Kokkos::fabs((x >= 0.0 ? p1 : p1 - 2.0) * x) > 1.0e3) {
    yj_inv_partials_rescaled<K>(p1, x, out);
    return;
  }
  // the finite end of the domain of h, where the forms give a large finite
  // value under FMA contraction instead of the infinity the limit has
  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1))) {
    yj_inv_partials_at_edge<K>(p1 < 0.0, p1, out);
    return;
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[2];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(1, z, lr);
      const double t0 = (lr[0] * x);
      out[0] = m::exp((t0 * (1.0 + (-1.0 * p1))));
      out[1] = (lr[1] * ipow<2>(x) * m::exp(t0));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[2];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(1, z, lr);
      const double t0 = (lr[0] * x);
      out[0] = m::exp((t0 * (1.0 + (-1.0 * p1))));
      out[1] = (lr[1] * ipow<2>(x) * m::exp((-1.0 * t0)));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[3];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(2, z, lr);
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (lr[0] * x);
      out[0] = (t0 * m::exp((t1 * (1.0 + (-2.0 * p1)))));
      out[1] = (((-1.0 * t1) + (lr[1] * t0 * ipow<2>(x))) * m::exp((t0 * t1)));
      out[2] = (ipow<3>(x) * (lr[2] + (x * ipow<2>(lr[1]))) * m::exp(t1));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[3];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(2, z, lr);
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (lr[0] * x);
      out[0] = (t0 * m::exp((t1 * (3.0 + (-2.0 * p1)))));
      out[1] = (((-1.0 * t1) + (lr[1] * t0 * ipow<2>(x))) * m::exp((t0 * t1)));
      out[2] = (-1.0 * ipow<3>(x) * ((-1.0 * lr[2]) + (x * ipow<2>(lr[1]))) * m::exp((-1.0 * t1)));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[4];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(3, z, lr);
      const double t0 = (1.0 + (-2.0 * p1));
      const double t1 = (-1.0 + p1);
      const double t2 = (-1.0 * t1);
      const double t3 = (lr[0] * x);
      const double t4 = m::exp((t0 * t3));
      const double t5 = ipow<2>(x);
      const double t6 = (t1 * x);
      out[0] = (t0 * t2 * m::exp((t3 * (1.0 + (-3.0 * p1)))));
      out[1] = ((-1.0 * t4) + (t2 * t4 * ((-2.0 * t3) + (lr[1] * t0 * t5))));
      out[2] = (t5 * (ipow<2>((lr[0] + (lr[1] * t6))) + (-2.0 * lr[1]) + (-1.0 * lr[2] * t6)) * m::exp((-1.0 * t1 * t3)));
      out[3] = (ipow<4>(x) * (lr[3] + (t5 * ipow<3>(lr[1])) + (3.0 * lr[1] * lr[2] * x)) * m::exp(t3));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[4];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(3, z, lr);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (3.0 + (-2.0 * p1));
      const double t3 = (lr[0] * x);
      const double t4 = m::exp((t2 * t3));
      const double t5 = ipow<2>(x);
      const double t6 = (t0 * x);
      out[0] = (t1 * t2 * m::exp((t3 * (5.0 + (-3.0 * p1)))));
      out[1] = ((-1.0 * t4) + (t1 * t4 * ((-2.0 * t3) + (lr[1] * t2 * t5))));
      out[2] = (t5 * (ipow<2>((lr[0] + (lr[1] * t6))) + (-2.0 * lr[1]) + (-1.0 * lr[2] * t6)) * m::exp((-1.0 * t0 * t3)));
      out[3] = (ipow<4>(x) * (lr[3] + (t5 * ipow<3>(lr[1])) + (-3.0 * lr[1] * lr[2] * x)) * m::exp((-1.0 * t3)));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[5];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(4, z, lr);
      const double t0 = (1.0 + (-3.0 * p1));
      const double t1 = (-1.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (-1.0 + p1);
      const double t4 = (-1.0 * t3);
      const double t5 = (lr[0] * x);
      const double t6 = m::exp((t0 * t5));
      const double t7 = ipow<2>(x);
      const double t8 = (2.0 * lr[1]);
      const double t9 = (t1 * x);
      const double t10 = (4.0 * lr[1]);
      const double t11 = (t3 * x);
      const double t12 = ipow<3>(x);
      const double t13 = (lr[0] + (lr[1] * t11));
      out[0] = (t0 * t2 * t4 * m::exp((t5 * (1.0 + (-4.0 * p1)))));
      out[1] = ((-1.0 * t2 * t6) + (-2.0 * t4 * t6) + (t2 * t4 * t6 * ((-3.0 * t5) + (lr[1] * t0 * t7))));
      out[2] = (x * ((4.0 * lr[0]) + (t11 * (t10 + (-1.0 * ipow<2>(((2.0 * lr[0]) + (lr[1] * t9)))) + (lr[2] * t9))) + (t8 * t9)) * m::exp((-1.0 * t1 * t5)));
      out[3] = (t12 * ((-1.0 * ipow<3>(t13)) + (-3.0 * lr[2]) + (-1.0 * lr[3] * t11) + (3.0 * t13 * (t8 + (lr[2] * t11)))) * m::exp((-1.0 * t3 * t5)));
      out[4] = (ipow<5>(x) * (lr[4] + (t12 * ipow<4>(lr[1])) + (3.0 * x * ipow<2>(lr[2])) + (lr[3] * t10 * x) + (6.0 * lr[2] * t7 * ipow<2>(lr[1]))) * m::exp(t5));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[5];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(4, z, lr);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-3.0 + (2.0 * p1));
      const double t3 = (-1.0 * t2);
      const double t4 = (5.0 + (-3.0 * p1));
      const double t5 = (lr[0] * x);
      const double t6 = m::exp((t4 * t5));
      const double t7 = ipow<2>(x);
      const double t8 = (2.0 * lr[1]);
      const double t9 = (t2 * x);
      const double t10 = (4.0 * lr[1]);
      const double t11 = (t0 * x);
      const double t12 = ipow<3>(x);
      const double t13 = (lr[0] + (lr[1] * t11));
      out[0] = (t1 * t3 * t4 * m::exp((t5 * (7.0 + (-4.0 * p1)))));
      out[1] = ((-1.0 * t3 * t6) + (-2.0 * t1 * t6) + (t1 * t3 * t6 * ((-3.0 * t5) + (lr[1] * t4 * t7))));
      out[2] = (x * ((4.0 * lr[0]) + (t11 * (t10 + (-1.0 * ipow<2>(((2.0 * lr[0]) + (lr[1] * t9)))) + (lr[2] * t9))) + (t8 * t9)) * m::exp((-1.0 * t2 * t5)));
      out[3] = (t12 * ((-1.0 * ipow<3>(t13)) + (-3.0 * lr[2]) + (-1.0 * lr[3] * t11) + (3.0 * t13 * (t8 + (lr[2] * t11)))) * m::exp((-1.0 * t0 * t5)));
      out[4] = (-1.0 * ipow<5>(x) * ((-1.0 * lr[4]) + (t12 * ipow<4>(lr[1])) + (3.0 * x * ipow<2>(lr[2])) + (lr[3] * t10 * x) + (-6.0 * lr[2] * t7 * ipow<2>(lr[1]))) * m::exp((-1.0 * t5)));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[6];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(5, z, lr);
      const double t0 = (1.0 + (-4.0 * p1));
      const double t1 = (-1.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (lr[0] * x);
      const double t4 = (-1.0 + (3.0 * p1));
      const double t5 = (-1.0 * t4);
      const double t6 = (-1.0 + p1);
      const double t7 = (-1.0 * t6);
      const double t8 = (t5 * t7);
      const double t9 = m::exp((t0 * t3));
      const double t10 = (t2 * t9);
      const double t11 = ipow<2>(x);
      const double t12 = (t4 * x);
      const double t13 = ((3.0 * lr[0]) + (lr[1] * t12));
      const double t14 = (t6 * x);
      const double t15 = (t1 * x);
      const double t16 = (3.0 * lr[2]);
      const double t17 = ((2.0 * lr[0]) + (lr[1] * t15));
      const double t18 = ipow<4>(x);
      const double t19 = (lr[0] + (lr[1] * t14));
      const double t20 = ((2.0 * lr[1]) + (lr[2] * t14));
      const double t21 = (10.0 * lr[2]);
      out[0] = (t0 * t2 * t8 * m::exp((t3 * (1.0 + (-5.0 * p1)))));
      out[1] = ((-1.0 * t10 * t5) + (-3.0 * t10 * t7) + (-2.0 * t8 * t9) + (t2 * t5 * t7 * t9 * ((-4.0 * t3) + (lr[1] * t0 * t11))));
      out[2] = ((4.0 + (-4.0 * t13 * t14) + (-2.0 * t13 * t15) + (-1.0 * t1 * t11 * t6 * ((-1.0 * ipow<2>(t13)) + (6.0 * lr[1]) + (lr[2] * t12)))) * m::exp((-1.0 * t3 * t4)));
      out[3] = (t11 * ((-3.0 * ipow<2>(t17)) + (12.0 * lr[1]) + (t14 * (ipow<3>(t17) + (6.0 * lr[2]) + (lr[3] * t15) + (-3.0 * t17 * ((4.0 * lr[1]) + (lr[2] * t15))))) + (t15 * t16)) * m::exp((-1.0 * t1 * t3)));
      out[4] = (t18 * (ipow<4>(t19) + (-4.0 * lr[3]) + (3.0 * ipow<2>(t20)) + (-1.0 * lr[4] * t14) + (-6.0 * t20 * ipow<2>(t19)) + (4.0 * t19 * (t16 + (lr[3] * t14)))) * m::exp((-1.0 * t3 * t6)));
      out[5] = (ipow<6>(x) * (lr[5] + (t18 * ipow<5>(lr[1])) + (lr[3] * t21 * x) + (t21 * ipow<3>(lr[1]) * ipow<3>(x)) + (5.0 * lr[1] * lr[4] * x) + (10.0 * lr[3] * t11 * ipow<2>(lr[1])) + (15.0 * lr[1] * t11 * ipow<2>(lr[2]))) * m::exp(t3));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[6];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(5, z, lr);
      const double t0 = (7.0 + (-4.0 * p1));
      const double t1 = (-3.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (lr[0] * x);
      const double t4 = (-5.0 + (3.0 * p1));
      const double t5 = (-1.0 * t4);
      const double t6 = (-1.0 + p1);
      const double t7 = (-1.0 * t6);
      const double t8 = (t5 * t7);
      const double t9 = m::exp((t0 * t3));
      const double t10 = (t2 * t9);
      const double t11 = ipow<2>(x);
      const double t12 = (t4 * x);
      const double t13 = ((3.0 * lr[0]) + (lr[1] * t12));
      const double t14 = (t6 * x);
      const double t15 = (t1 * x);
      const double t16 = (3.0 * lr[2]);
      const double t17 = ((2.0 * lr[0]) + (lr[1] * t15));
      const double t18 = ipow<4>(x);
      const double t19 = (lr[0] + (lr[1] * t14));
      const double t20 = ((2.0 * lr[1]) + (lr[2] * t14));
      const double t21 = (10.0 * lr[2]);
      out[0] = (t0 * t2 * t8 * m::exp((t3 * (9.0 + (-5.0 * p1)))));
      out[1] = ((-1.0 * t10 * t5) + (-3.0 * t10 * t7) + (-2.0 * t8 * t9) + (t2 * t5 * t7 * t9 * ((-4.0 * t3) + (lr[1] * t0 * t11))));
      out[2] = ((4.0 + (-4.0 * t13 * t14) + (-2.0 * t13 * t15) + (-1.0 * t1 * t11 * t6 * ((-1.0 * ipow<2>(t13)) + (6.0 * lr[1]) + (lr[2] * t12)))) * m::exp((-1.0 * t3 * t4)));
      out[3] = (t11 * ((-3.0 * ipow<2>(t17)) + (12.0 * lr[1]) + (t14 * (ipow<3>(t17) + (6.0 * lr[2]) + (lr[3] * t15) + (-3.0 * t17 * ((4.0 * lr[1]) + (lr[2] * t15))))) + (t15 * t16)) * m::exp((-1.0 * t1 * t3)));
      out[4] = (t18 * (ipow<4>(t19) + (-4.0 * lr[3]) + (3.0 * ipow<2>(t20)) + (-1.0 * lr[4] * t14) + (-6.0 * t20 * ipow<2>(t19)) + (4.0 * t19 * (t16 + (lr[3] * t14)))) * m::exp((-1.0 * t3 * t6)));
      out[5] = (ipow<6>(x) * (lr[5] + (t18 * ipow<5>(lr[1])) + (-1.0 * lr[3] * t21 * x) + (-1.0 * t21 * ipow<3>(lr[1]) * ipow<3>(x)) + (-5.0 * lr[1] * lr[4] * x) + (10.0 * lr[3] * t11 * ipow<2>(lr[1])) + (15.0 * lr[1] * t11 * ipow<2>(lr[2]))) * m::exp((-1.0 * t3)));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[7];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(6, z, lr);
      const double t0 = (1.0 + (-5.0 * p1));
      const double t1 = (-1.0 + (4.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (-1.0 + (3.0 * p1));
      const double t4 = (-1.0 * t3);
      const double t5 = (-1.0 + (2.0 * p1));
      const double t6 = (-1.0 * t5);
      const double t7 = (-1.0 + p1);
      const double t8 = (-1.0 * t7);
      const double t9 = (lr[0] * x);
      const double t10 = m::exp((t0 * t9));
      const double t11 = (t10 * t4 * t6);
      const double t12 = (t10 * t2 * t8);
      const double t13 = ipow<2>(x);
      const double t14 = (t1 * x);
      const double t15 = ((4.0 * lr[0]) + (lr[1] * t14));
      const double t16 = (t5 * x);
      const double t17 = (t3 * x);
      const double t18 = (t15 * t17);
      const double t19 = (t13 * t5 * t7);
      const double t20 = (lr[1] * t17);
      const double t21 = (t20 + (3.0 * lr[0]));
      const double t22 = (6.0 * lr[1]);
      const double t23 = (t22 + (lr[2] * t17));
      const double t24 = (t23 + (-1.0 * ipow<2>(t21)));
      const double t25 = (t7 * x);
      const double t26 = ipow<3>(x);
      const double t27 = (4.0 * lr[3]);
      const double t28 = ((2.0 * lr[0]) + (lr[1] * t16));
      const double t29 = ((4.0 * lr[1]) + (lr[2] * t16));
      const double t30 = ipow<5>(x);
      const double t31 = (lr[0] + (lr[1] * t25));
      const double t32 = ((2.0 * lr[1]) + (lr[2] * t25));
      const double t33 = ((3.0 * lr[2]) + (lr[3] * t25));
      const double t34 = (15.0 * lr[2]);
      const double t35 = (15.0 * t13);
      const double t36 = ipow<2>(lr[1]);
      out[0] = (t0 * t2 * t4 * t6 * t8 * m::exp((t9 * (1.0 + (-6.0 * p1)))));
      out[1] = ((-1.0 * t11 * t2) + (-4.0 * t11 * t8) + (-3.0 * t12 * t6) + (-2.0 * t12 * t4) + (t10 * t2 * t4 * t6 * t8 * ((-5.0 * t9) + (lr[1] * t0 * t13))));
      out[2] = ((22.0 + (-36.0 * p1) + (2.0 * t18 * t5) + (4.0 * t18 * t7) + (t19 * t3 * ((-1.0 * ipow<2>(t15)) + (8.0 * lr[1]) + (lr[2] * t14))) + (6.0 * t15 * t16 * t7)) * m::exp((-1.0 * t1 * t9)));
      out[3] = (-1.0 * x * ((12.0 * t20) + (36.0 * lr[0]) + (t19 * (ipow<3>(t21) + (9.0 * lr[2]) + (lr[3] * t17) + (-3.0 * t21 * t23))) + (3.0 * t16 * t24) + (6.0 * t24 * t25)) * m::exp((-1.0 * t3 * t9)));
      out[4] = (t26 * ((4.0 * ipow<3>(t28)) + (24.0 * lr[2]) + (t16 * t27) + (t25 * ((-1.0 * ipow<4>(t28)) + (-3.0 * ipow<2>(t29)) + (8.0 * lr[3]) + (lr[4] * t16) + (-4.0 * t28 * ((6.0 * lr[2]) + (lr[3] * t16))) + (6.0 * t29 * ipow<2>(t28)))) + (-12.0 * t28 * t29)) * m::exp((-1.0 * t5 * t9)));
      out[5] = (t30 * ((-1.0 * ipow<5>(t31)) + (-5.0 * lr[4]) + (-1.0 * lr[5] * t25) + (-15.0 * t31 * ipow<2>(t32)) + (-10.0 * t33 * ipow<2>(t31)) + (5.0 * t31 * (t27 + (lr[4] * t25))) + (10.0 * t32 * t33) + (10.0 * t32 * ipow<3>(t31))) * m::exp((-1.0 * t7 * t9)));
      out[6] = (ipow<7>(x) * (lr[6] + (t30 * ipow<6>(lr[1])) + (t35 * ipow<3>(lr[2])) + (10.0 * x * ipow<2>(lr[3])) + (lr[4] * t34 * x) + (lr[4] * t35 * t36) + (lr[5] * t22 * x) + (t34 * ipow<4>(lr[1]) * ipow<4>(x)) + (20.0 * lr[3] * t26 * ipow<3>(lr[1])) + (45.0 * t26 * t36 * ipow<2>(lr[2])) + (60.0 * lr[1] * lr[2] * lr[3] * t13)) * m::exp(t9));
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[7];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(6, z, lr);
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-3.0 + (2.0 * p1));
      const double t3 = (-1.0 * t2);
      const double t4 = (-5.0 + (3.0 * p1));
      const double t5 = (-1.0 * t4);
      const double t6 = (-7.0 + (4.0 * p1));
      const double t7 = (-1.0 * t6);
      const double t8 = (9.0 + (-5.0 * p1));
      const double t9 = (lr[0] * x);
      const double t10 = m::exp((t8 * t9));
      const double t11 = (t10 * t3 * t5);
      const double t12 = (t1 * t10 * t7);
      const double t13 = ipow<2>(x);
      const double t14 = (t6 * x);
      const double t15 = ((4.0 * lr[0]) + (lr[1] * t14));
      const double t16 = (t2 * x);
      const double t17 = (t4 * x);
      const double t18 = (t15 * t17);
      const double t19 = (t0 * t13 * t2);
      const double t20 = (lr[1] * t17);
      const double t21 = (t20 + (3.0 * lr[0]));
      const double t22 = (6.0 * lr[1]);
      const double t23 = (t22 + (lr[2] * t17));
      const double t24 = (t23 + (-1.0 * ipow<2>(t21)));
      const double t25 = (t0 * x);
      const double t26 = ipow<3>(x);
      const double t27 = (4.0 * lr[3]);
      const double t28 = ((2.0 * lr[0]) + (lr[1] * t16));
      const double t29 = ((4.0 * lr[1]) + (lr[2] * t16));
      const double t30 = ipow<5>(x);
      const double t31 = (lr[0] + (lr[1] * t25));
      const double t32 = ((2.0 * lr[1]) + (lr[2] * t25));
      const double t33 = ((3.0 * lr[2]) + (lr[3] * t25));
      const double t34 = (15.0 * lr[2]);
      const double t35 = (15.0 * t13);
      const double t36 = ipow<2>(lr[1]);
      out[0] = (t1 * t3 * t5 * t7 * t8 * m::exp((t9 * (11.0 + (-6.0 * p1)))));
      out[1] = ((-1.0 * t11 * t7) + (-4.0 * t1 * t11) + (-3.0 * t12 * t3) + (-2.0 * t12 * t5) + (t1 * t10 * t3 * t5 * t7 * ((-5.0 * t9) + (lr[1] * t13 * t8))));
      out[2] = ((50.0 + (-36.0 * p1) + (2.0 * t18 * t2) + (4.0 * t0 * t18) + (t19 * t4 * ((-1.0 * ipow<2>(t15)) + (8.0 * lr[1]) + (lr[2] * t14))) + (6.0 * t0 * t15 * t16)) * m::exp((-1.0 * t6 * t9)));
      out[3] = (-1.0 * x * ((12.0 * t20) + (36.0 * lr[0]) + (t19 * (ipow<3>(t21) + (9.0 * lr[2]) + (lr[3] * t17) + (-3.0 * t21 * t23))) + (3.0 * t16 * t24) + (6.0 * t24 * t25)) * m::exp((-1.0 * t4 * t9)));
      out[4] = (t26 * ((4.0 * ipow<3>(t28)) + (24.0 * lr[2]) + (t16 * t27) + (t25 * ((-1.0 * ipow<4>(t28)) + (-3.0 * ipow<2>(t29)) + (8.0 * lr[3]) + (lr[4] * t16) + (-4.0 * t28 * ((6.0 * lr[2]) + (lr[3] * t16))) + (6.0 * t29 * ipow<2>(t28)))) + (-12.0 * t28 * t29)) * m::exp((-1.0 * t2 * t9)));
      out[5] = (t30 * ((-1.0 * ipow<5>(t31)) + (-5.0 * lr[4]) + (-1.0 * lr[5] * t25) + (-15.0 * t31 * ipow<2>(t32)) + (-10.0 * t33 * ipow<2>(t31)) + (5.0 * t31 * (t27 + (lr[4] * t25))) + (10.0 * t32 * t33) + (10.0 * t32 * ipow<3>(t31))) * m::exp((-1.0 * t0 * t9)));
      out[6] = (-1.0 * ipow<7>(x) * ((-1.0 * lr[6]) + (t30 * ipow<6>(lr[1])) + (-1.0 * t35 * ipow<3>(lr[2])) + (10.0 * x * ipow<2>(lr[3])) + (lr[4] * t34 * x) + (lr[5] * t22 * x) + (-1.0 * lr[4] * t35 * t36) + (-1.0 * t34 * ipow<4>(lr[1]) * ipow<4>(x)) + (20.0 * lr[3] * t26 * ipow<3>(lr[1])) + (45.0 * t26 * t36 * ipow<2>(lr[2])) + (-60.0 * lr[1] * lr[2] * lr[3] * t13)) * m::exp((-1.0 * t9)));
      return;
    }
  }
}

template <int K>
LAPLACE_HD double yeo_johnson_fwd_lj(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  // an infinite x: the forms below give NaN there, the limit is taken instead
  if (Kokkos::isinf(x)) return yj_fwd_lj_at_inf<K>(x > 0.0, p1);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * Kokkos::log1p(x));
    }
    {
      return ((1.0 + (-1.0 * p1)) * Kokkos::log1p((-1.0 * x)));
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      return ((1.0 / ipow<1>((1.0 + x))) * (-1.0 + p1));
    }
    {
      return ((1.0 / ipow<1>((1.0 + (-1.0 * x)))) * (-1.0 + p1));
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      return ((1.0 / ipow<2>((1.0 + x))) * (1.0 + (-1.0 * p1)));
    }
    {
      return ((1.0 / ipow<2>((1.0 + (-1.0 * x)))) * (-1.0 + p1));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      return ((1.0 / ipow<3>((1.0 + x))) * (-2.0 + (2.0 * p1)));
    }
    {
      return ((1.0 / ipow<3>((1.0 + (-1.0 * x)))) * (-2.0 + (2.0 * p1)));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      return ((1.0 / ipow<4>((1.0 + x))) * (6.0 + (-6.0 * p1)));
    }
    {
      return ((1.0 / ipow<4>((1.0 + (-1.0 * x)))) * (-6.0 + (6.0 * p1)));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      return ((1.0 / ipow<5>((1.0 + x))) * (-24.0 + (24.0 * p1)));
    }
    {
      return ((1.0 / ipow<5>((1.0 + (-1.0 * x)))) * (-24.0 + (24.0 * p1)));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      return ((1.0 / ipow<6>((1.0 + x))) * (120.0 + (-120.0 * p1)));
    }
    {
      return ((1.0 / ipow<6>((1.0 + (-1.0 * x)))) * (-120.0 + (120.0 * p1)));
    }
  }
  return nan();
}

template <int K>
LAPLACE_HD void yeo_johnson_fwd_lj_partials(double x, double p1, double p2, double* out) {
  static_assert(K >= 1 && K <= 6, "order 1 to 6");
  static_cast<void>(p2);
  for (int i = 0; i < (K + 1) * (K + 2) / 2; ++i) out[i] = 0.0;
  // an infinite x: the forms below give NaN there, the limits are taken instead
  if (Kokkos::isinf(x)) {
    yj_fwd_lj_partials_at_inf<K>(x > 0.0, p1, out);
    return;
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      out[0] = ((1.0 / ipow<1>((1.0 + x))) * (-1.0 + p1));
      out[1] = Kokkos::log1p(x);
      return;
    }
    {
      out[0] = ((1.0 / ipow<1>((1.0 + (-1.0 * x)))) * (-1.0 + p1));
      out[1] = (-1.0 * Kokkos::log1p((-1.0 * x)));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double t0 = (1.0 + x);
      out[0] = ((1.0 / ipow<2>(t0)) * (1.0 + (-1.0 * p1)));
      out[1] = (1.0 / ipow<1>(t0));
      out[2] = 0.0;
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * x));
      out[0] = ((1.0 / ipow<2>(t0)) * (-1.0 + p1));
      out[1] = (1.0 / ipow<1>(t0));
      out[2] = 0.0;
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double t0 = (1.0 + x);
      out[0] = ((1.0 / ipow<3>(t0)) * (-2.0 + (2.0 * p1)));
      out[1] = (-1.0 * (1.0 / ipow<2>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * x));
      out[0] = ((1.0 / ipow<3>(t0)) * (-2.0 + (2.0 * p1)));
      out[1] = (1.0 / ipow<2>(t0));
      out[2] = 0.0;
      out[3] = 0.0;
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double t0 = (1.0 + x);
      out[0] = ((1.0 / ipow<4>(t0)) * (6.0 + (-6.0 * p1)));
      out[1] = (2.0 * (1.0 / ipow<3>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * x));
      out[0] = ((1.0 / ipow<4>(t0)) * (-6.0 + (6.0 * p1)));
      out[1] = (2.0 * (1.0 / ipow<3>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double t0 = (1.0 + x);
      out[0] = ((1.0 / ipow<5>(t0)) * (-24.0 + (24.0 * p1)));
      out[1] = (-6.0 * (1.0 / ipow<4>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      out[5] = 0.0;
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * x));
      out[0] = ((1.0 / ipow<5>(t0)) * (-24.0 + (24.0 * p1)));
      out[1] = (6.0 * (1.0 / ipow<4>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      out[5] = 0.0;
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double t0 = (1.0 + x);
      out[0] = ((1.0 / ipow<6>(t0)) * (120.0 + (-120.0 * p1)));
      out[1] = (24.0 * (1.0 / ipow<5>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      out[5] = 0.0;
      out[6] = 0.0;
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * x));
      out[0] = ((1.0 / ipow<6>(t0)) * (-120.0 + (120.0 * p1)));
      out[1] = (24.0 * (1.0 / ipow<5>(t0)));
      out[2] = 0.0;
      out[3] = 0.0;
      out[4] = 0.0;
      out[5] = 0.0;
      out[6] = 0.0;
      return;
    }
  }
}

// d^(K - n1)_u d^n1_lambda [(1 - lambda) log(1 + lambda u)/lambda], n1 = 0..K, in out[0..K] (the
// other entries are not written): log|h'| of Yeo-Johnson for u >= 0, and log|h'| of Box-Cox
// (scalar_box_cox.hpp) for every u with 1 + lambda u > 0.
template <int K>
LAPLACE_HD void yeo_johnson_inv_lj_pos_partials(double x, double p1, double* out) {
  static_assert(K >= 1 && K <= 6, "order 1 to 6");
  if constexpr (K == 1) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<1>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<1>(p1, x, out);
      return;
    }
    const double z = (p1 * x);
    double lr[2];
    lr[0] = m::log1prel(z);
    m::detail_log1prel::derivs_upto(1, z, lr);
    const double t0 = (1.0 + (-1.0 * p1));
    out[0] = (t0 * (1.0 / ipow<1>((1.0 + (p1 * x)))));
    out[1] = yj_big ? yj_top : (((-1.0 * lr[0] * x) + (lr[1] * t0 * ipow<2>(x))));
    return;
  }
  if constexpr (K == 2) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<2>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<2>(p1, x, out);
      return;
    }
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[3];
      m::detail_log1prel::derivs_upto(2, z, lr);
      const double f0 = (-1.0 + p1);
      out[0] = (f0 * p1 * (1.0 / ipow<2>((1.0 + (p1 * x)))));
      out[1] = (-1.0 * e * (1.0 / ipow<1>(p1)) * (1.0 + (-1.0 * e) + (e * p1)));
      out[2] = yj_big ? yj_top : ((-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (f0 * lr[2] * x))));
      return;
    }
    const double z = (p1 * x);
    double lr[3];
    m::detail_log1prel::derivs_upto(2, z, lr);
    const double t0 = (-1.0 + p1);
    const double t1 = (1.0 + (p1 * x));
    const double t2 = (1.0 / ipow<2>(t1));
    out[0] = (t0 * t2 * p1);
    out[1] = ((-1.0 * (1.0 / ipow<1>(t1))) + (t0 * t2 * x));
    out[2] = yj_big ? yj_top : ((-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (lr[2] * t0 * x))));
    return;
  }
  if constexpr (K == 3) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<3>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<3>(p1, x, out);
      return;
    }
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[4];
      m::detail_log1prel::derivs_upto(3, z, lr);
      const double f0 = ipow<2>(p1);
      const double f1 = (2.0 * p1);
      const double f2 = (2.0 * e);
      out[0] = (f0 * (1.0 / ipow<3>((1.0 + (p1 * x)))) * (2.0 + (-1.0 * f1)));
      out[1] = (ipow<2>(e) * (1.0 + (-1.0 * f2) + (e * f1)));
      out[2] = (-1.0 * f2 * (1.0 / ipow<1>(f0)) * (-1.0 + e) * (1.0 + (-1.0 * e) + (e * p1)));
      out[3] = yj_big ? yj_top : ((-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * x * (-1.0 + p1)))));
      return;
    }
    const double z = (p1 * x);
    double lr[4];
    m::detail_log1prel::derivs_upto(3, z, lr);
    const double t0 = (p1 * x);
    const double t1 = (1.0 + t0);
    const double t2 = (1.0 / ipow<3>(t1));
    const double t3 = (1.0 / ipow<2>(t1));
    const double t4 = (-1.0 + p1);
    const double t5 = (t4 * x);
    out[0] = (t2 * ipow<2>(p1) * (2.0 + (-2.0 * p1)));
    out[1] = ((t3 * t4) + (t3 * p1) + (-2.0 * t0 * t2 * t4));
    out[2] = (2.0 * t3 * x * (1.0 + (-1.0 * t5 * (1.0 / ipow<1>(t1)))));
    out[3] = yj_big ? yj_top : ((-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * t5))));
    return;
  }
  if constexpr (K == 4) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<4>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<4>(p1, x, out);
      return;
    }
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[5];
      m::detail_log1prel::derivs_upto(4, z, lr);
      const double f0 = ipow<3>(p1);
      const double f1 = (3.0 * e);
      const double f2 = ipow<2>(e);
      const double f3 = (e * p1);
      out[0] = (f0 * (1.0 / ipow<4>((1.0 + (p1 * x)))) * (-6.0 + (6.0 * p1)));
      out[1] = (-2.0 * p1 * ipow<3>(e) * (1.0 + (-1.0 * f1) + (f1 * p1)));
      out[2] = (2.0 * f2 * (1.0 / ipow<1>(p1)) * (-1.0 + (-3.0 * f2) + (-2.0 * f3) + (4.0 * e) + (3.0 * f2 * p1)));
      out[3] = (-6.0 * e * (1.0 / ipow<1>(f0)) * ipow<2>((-1.0 + e)) * (1.0 + f3 + (-1.0 * e)));
      out[4] = yj_big ? yj_top : ((-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * x * (-1.0 + p1)))));
      return;
    }
    const double z = (p1 * x);
    double lr[5];
    m::detail_log1prel::derivs_upto(4, z, lr);
    const double t0 = (p1 * x);
    const double t1 = (1.0 + t0);
    const double t2 = (1.0 / ipow<4>(t1));
    const double t3 = ipow<2>(p1);
    const double t4 = (1.0 / ipow<3>(t1));
    const double t5 = (2.0 + (-2.0 * p1));
    const double t6 = (1.0 / ipow<2>(t1));
    const double t7 = (1.0 / ipow<1>(t1));
    const double t8 = (-1.0 + p1);
    const double t9 = (t8 * x);
    const double t10 = (t7 * t9);
    const double t11 = ipow<2>(x);
    out[0] = (t2 * ipow<3>(p1) * (-6.0 + (6.0 * p1)));
    out[1] = ((-2.0 * t3 * t4) + (2.0 * t4 * t5 * p1) + (-3.0 * t2 * t3 * t5 * x));
    out[2] = (2.0 * t6 * (1.0 + (-2.0 * t10) + (-2.0 * t0 * t7) + (3.0 * t11 * t6 * t8 * p1)));
    out[3] = (6.0 * t11 * t4 * (-1.0 + t10));
    out[4] = yj_big ? yj_top : ((-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * t9))));
    return;
  }
  if constexpr (K == 5) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<5>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<5>(p1, x, out);
      return;
    }
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[6];
      m::detail_log1prel::derivs_upto(5, z, lr);
      const double f0 = ipow<4>(p1);
      const double f1 = ipow<2>(p1);
      const double f2 = (4.0 * e);
      const double f3 = (e * p1);
      const double f4 = ipow<2>(e);
      const double f5 = (6.0 * f4);
      const double f6 = (-1.0 + e);
      out[0] = (f0 * (1.0 / ipow<5>((1.0 + (p1 * x)))) * (24.0 + (-24.0 * p1)));
      out[1] = (6.0 * f1 * ipow<4>(e) * (1.0 + (-1.0 * f2) + (f2 * p1)));
      out[2] = (-4.0 * ipow<3>(e) * (-1.0 + (-1.0 * f5) + (-3.0 * f3) + (6.0 * e) + (6.0 * f4 * p1)));
      out[3] = (f5 * f6 * (1.0 / ipow<1>(f1)) * (-1.0 + (-4.0 * f4) + (-2.0 * f3) + (5.0 * e) + (4.0 * f4 * p1)));
      out[4] = (-24.0 * e * (1.0 / ipow<1>(f0)) * ipow<3>(f6) * (1.0 + f3 + (-1.0 * e)));
      out[5] = yj_big ? yj_top : ((-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * x * (-1.0 + p1)))));
      return;
    }
    const double z = (p1 * x);
    double lr[6];
    m::detail_log1prel::derivs_upto(5, z, lr);
    const double t0 = (p1 * x);
    const double t1 = (1.0 + t0);
    const double t2 = (1.0 / ipow<5>(t1));
    const double t3 = ipow<3>(p1);
    const double t4 = (1.0 / ipow<4>(t1));
    const double t5 = (-6.0 + (6.0 * p1));
    const double t6 = ipow<2>(p1);
    const double t7 = (3.0 * t6);
    const double t8 = (1.0 / ipow<3>(t1));
    const double t9 = (1.0 / ipow<1>(t1));
    const double t10 = (-1.0 + p1);
    const double t11 = (6.0 * t10);
    const double t12 = ((1.0 / ipow<2>(t1)) * ipow<2>(x));
    const double t13 = (t10 * x);
    out[0] = (t2 * ipow<4>(p1) * (24.0 + (-24.0 * p1)));
    out[1] = ((6.0 * t3 * t4) + (t4 * t5 * t7) + (-4.0 * t2 * t3 * t5 * x));
    out[2] = (4.0 * t8 * (1.0 + (-3.0 * p1) + (t0 * t11 * t9) + (t7 * t9 * x) + (-1.0 * t11 * t12 * t6)));
    out[3] = (6.0 * t8 * x * (-2.0 + (-4.0 * t10 * t12 * p1) + (3.0 * t10 * t9 * x) + (3.0 * t9 * p1 * x)));
    out[4] = (24.0 * t4 * ipow<3>(x) * (1.0 + (-1.0 * t13 * t9)));
    out[5] = yj_big ? yj_top : ((-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * t13))));
    return;
  }
  if constexpr (K == 6) {
    const bool yj_big = Kokkos::fabs(p1 * x) > 1.0e3;
    const double yj_top =
        yj_big ? yj_inv_lj_pure_lambda_rescaled<6>(p1, x) : 0.0;
    if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
      yj_inv_lj_partials_small_z<6>(p1, x, out);
      return;
    }
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[7];
      m::detail_log1prel::derivs_upto(6, z, lr);
      const double f0 = ipow<5>(p1);
      const double f1 = ipow<3>(p1);
      const double f2 = (5.0 * e);
      const double f3 = (e * p1);
      const double f4 = ipow<2>(e);
      const double f5 = (12.0 * p1);
      const double f6 = ipow<3>(e);
      const double f7 = (10.0 * f6);
      const double f8 = (-1.0 + e);
      out[0] = (f0 * (1.0 / ipow<6>((1.0 + (p1 * x)))) * (-120.0 + (120.0 * p1)));
      out[1] = (-24.0 * f1 * ipow<5>(e) * (1.0 + (-1.0 * f2) + (f2 * p1)));
      out[2] = (f5 * ipow<4>(e) * (-1.0 + (-10.0 * f4) + (-4.0 * f3) + (8.0 * e) + (10.0 * f4 * p1)));
      out[3] = (-12.0 * f6 * (1.0 / ipow<1>(p1)) * (1.0 + (-1.0 * f7) + (-9.0 * e) + (3.0 * f3) + (18.0 * f4) + (f7 * p1) + (-1.0 * f4 * f5)));
      out[4] = (24.0 * f4 * (1.0 / ipow<1>(f1)) * ipow<2>(f8) * (-1.0 + (-5.0 * f4) + (-2.0 * f3) + (6.0 * e) + (5.0 * f4 * p1)));
      out[5] = (-120.0 * e * (1.0 / ipow<1>(f0)) * ipow<4>(f8) * (1.0 + f3 + (-1.0 * e)));
      out[6] = yj_big ? yj_top : ((-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * x * (-1.0 + p1)))));
      return;
    }
    const double z = (p1 * x);
    double lr[7];
    m::detail_log1prel::derivs_upto(6, z, lr);
    const double t0 = (p1 * x);
    const double t1 = (1.0 + t0);
    const double t2 = (1.0 / ipow<6>(t1));
    const double t3 = ipow<4>(p1);
    const double t4 = (1.0 / ipow<5>(t1));
    const double t5 = (24.0 + (-24.0 * p1));
    const double t6 = (1.0 / ipow<4>(t1));
    const double t7 = ipow<2>(p1);
    const double t8 = (1.0 / ipow<1>(t1));
    const double t9 = (4.0 * t8);
    const double t10 = (-1.0 + p1);
    const double t11 = (t0 * t8);
    const double t12 = ipow<2>(x);
    const double t13 = (1.0 / ipow<2>(t1));
    const double t14 = (12.0 * p1);
    const double t15 = (1.0 / ipow<3>(t1));
    const double t16 = (t10 * x);
    const double t17 = (t16 * t8);
    const double t18 = (t12 * t13);
    const double t19 = (t10 * t18);
    out[0] = (t2 * ipow<5>(p1) * (-120.0 + (120.0 * p1)));
    out[1] = ((-24.0 * t3 * t4) + (4.0 * t4 * t5 * ipow<3>(p1)) + (-5.0 * t2 * t3 * t5 * x));
    out[2] = (t14 * t6 * (-3.0 + (6.0 * p1) + (-12.0 * t10 * t11) + (-1.0 * t7 * t9 * x) + (10.0 * t10 * t12 * t13 * t7)));
    out[3] = (12.0 * t15 * (-1.0 + (3.0 * t17) + (6.0 * t11) + (-1.0 * t14 * t19) + (-6.0 * t18 * t7) + (10.0 * t10 * t15 * t7 * ipow<3>(x))));
    out[4] = (24.0 * t12 * t6 * (3.0 + (-4.0 * t17) + (-1.0 * t0 * t9) + (5.0 * t19 * p1)));
    out[5] = (120.0 * t4 * ipow<4>(x) * (-1.0 + t17));
    out[6] = yj_big ? yj_top : ((-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * t16))));
    return;
  }
}

template <int K>
LAPLACE_HD double yeo_johnson_inv_lj(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  // an infinite x: the forms below give NaN there, the limit is taken instead
  if (Kokkos::isinf(x)) return yj_inv_lj_at_inf<K>(x > 0.0, p1);
  // the finite end of the domain of h, where the forms give a large finite value
  // under FMA contraction instead of the infinity the limit has
  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1)))
    return yj_inv_lj_at_edge<K>(p1 < 0.0, p1);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      const double z = (p1 * x);
      double lr[1];
      lr[0] = m::log1prel(z);
      return (lr[0] * x * (1.0 + (-1.0 * p1)));
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[1];
      lr[0] = m::log1prel(z);
      return (lr[0] * x * (1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      return ((1.0 / ipow<1>((1.0 + (p1 * x)))) * (1.0 + (-1.0 * p1)));
    }
    {
      return ((1.0 / ipow<1>((1.0 + (-1.0 * x * (2.0 + (-1.0 * p1)))))) * (1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      return (p1 * (1.0 / ipow<2>((1.0 + (p1 * x)))) * (-1.0 + p1));
    }
    {
      const double t0 = (2.0 + (-1.0 * p1));
      return (t0 * (1.0 / ipow<2>((1.0 + (-1.0 * t0 * x)))) * (1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      return (ipow<2>(p1) * (1.0 / ipow<3>((1.0 + (p1 * x)))) * (2.0 + (-2.0 * p1)));
    }
    {
      const double t0 = (2.0 + (-1.0 * p1));
      return (ipow<2>(t0) * (1.0 / ipow<3>((1.0 + (-1.0 * t0 * x)))) * (2.0 + (-2.0 * p1)));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      return (ipow<3>(p1) * (1.0 / ipow<4>((1.0 + (p1 * x)))) * (-6.0 + (6.0 * p1)));
    }
    {
      const double t0 = (2.0 + (-1.0 * p1));
      return (ipow<3>(t0) * (1.0 / ipow<4>((1.0 + (-1.0 * t0 * x)))) * (6.0 + (-6.0 * p1)));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      return (ipow<4>(p1) * (1.0 / ipow<5>((1.0 + (p1 * x)))) * (24.0 + (-24.0 * p1)));
    }
    {
      const double t0 = (2.0 + (-1.0 * p1));
      return (ipow<4>(t0) * (1.0 / ipow<5>((1.0 + (-1.0 * t0 * x)))) * (24.0 + (-24.0 * p1)));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      return (ipow<5>(p1) * (1.0 / ipow<6>((1.0 + (p1 * x)))) * (-120.0 + (120.0 * p1)));
    }
    {
      const double t0 = (2.0 + (-1.0 * p1));
      return (ipow<5>(t0) * (1.0 / ipow<6>((1.0 + (-1.0 * t0 * x)))) * (120.0 + (-120.0 * p1)));
    }
  }
  return nan();
}

template <int K>
LAPLACE_HD void yeo_johnson_inv_lj_partials(double x, double p1, double p2, double* out) {
  static_assert(K >= 1 && K <= 6, "order 1 to 6");
  static_cast<void>(p2);
  for (int i = 0; i < (K + 1) * (K + 2) / 2; ++i) out[i] = 0.0;
  // an infinite x: the forms below give NaN there, the limits are taken instead
  if (Kokkos::isinf(x)) {
    yj_inv_lj_partials_at_inf<K>(x > 0.0, p1, out);
    return;
  }
  // the finite end of the domain of h, where the forms give a large finite
  // value under FMA contraction instead of the infinity the limit has
  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1))) {
    yj_inv_lj_partials_at_edge<K>(p1 < 0.0, p1, out);
    return;
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<1>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? -yj_inv_lj_pure_lambda_rescaled<1>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<1>(2.0 - p1, -x, out);
        for (int n1 = 0; n1 <= 1; ++n1) out[n1] = -out[n1];
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[2];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(1, z, lr);
      const double t0 = (1.0 + (-1.0 * p1));
      out[0] = (t0 * (1.0 / ipow<1>((1.0 + (-1.0 * x * (2.0 + (-1.0 * p1)))))));
      out[1] = yj_big ? yj_top : (((-1.0 * lr[0] * x) + (lr[1] * t0 * ipow<2>(x))));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<2>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? yj_inv_lj_pure_lambda_rescaled<2>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<2>(2.0 - p1, -x, out);
        return;
      }
      const double zs = (x * (-2.0 + p1));
      if (Kokkos::fabs(zs) > 0.5) {
        const double e = 1.0 / (1.0 + zs);
        const double z = (x * (-2.0 + p1));
        double lr[3];
        m::detail_log1prel::derivs_upto(2, z, lr);
        const double f0 = (-1.0 + p1);
        const double f1 = (-2.0 + p1);
        const double f2 = (-1.0 * f1);
        out[0] = (-1.0 * f0 * f2 * (1.0 / ipow<2>((1.0 + (-1.0 * f2 * x)))));
        out[1] = (-1.0 * e * (1.0 / ipow<1>(f1)) * (-1.0 + (-1.0 * e) + (e * p1)));
        out[2] = yj_big ? yj_top : ((-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (f0 * lr[2] * x))));
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[3];
      m::detail_log1prel::derivs_upto(2, z, lr);
      const double t0 = (2.0 + (-1.0 * p1));
      const double t1 = (1.0 + (-1.0 * t0 * x));
      const double t2 = (-1.0 + p1);
      const double t3 = (-1.0 * t2 * (1.0 / ipow<2>(t1)));
      out[0] = (t0 * t3);
      out[1] = ((-1.0 * (1.0 / ipow<1>(t1))) + (-1.0 * t3 * x));
      out[2] = yj_big ? yj_top : ((-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (lr[2] * t2 * x))));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<3>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? -yj_inv_lj_pure_lambda_rescaled<3>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<3>(2.0 - p1, -x, out);
        for (int n1 = 0; n1 <= 3; ++n1) out[n1] = -out[n1];
        return;
      }
      const double zs = (x * (-2.0 + p1));
      if (Kokkos::fabs(zs) > 0.5) {
        const double e = 1.0 / (1.0 + zs);
        const double z = (x * (-2.0 + p1));
        double lr[4];
        m::detail_log1prel::derivs_upto(3, z, lr);
        const double f0 = (-2.0 + p1);
        const double f1 = (-1.0 * f0);
        const double f2 = (2.0 * e);
        out[0] = (ipow<2>(f1) * (1.0 / ipow<3>((1.0 + (-1.0 * f1 * x)))) * (2.0 + (-2.0 * p1)));
        out[1] = (ipow<2>(e) * (-1.0 + (-1.0 * f2) + (2.0 * e * p1)));
        out[2] = (-1.0 * f2 * (1.0 / ipow<2>(f0)) * (-1.0 + e) * (-1.0 + (-1.0 * e) + (e * p1)));
        out[3] = yj_big ? yj_top : ((-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * x * (-1.0 + p1)))));
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[4];
      m::detail_log1prel::derivs_upto(3, z, lr);
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (t1 * x);
      const double t3 = (1.0 + (-1.0 * t2));
      const double t4 = (1.0 / ipow<3>(t3));
      const double t5 = (-1.0 + p1);
      const double t6 = (-1.0 * t5);
      const double t7 = (1.0 / ipow<2>(t3));
      const double t8 = (1.0 + (t0 * x));
      const double t9 = (t5 * x);
      out[0] = (t4 * ipow<2>(t1) * (2.0 + (-2.0 * p1)));
      out[1] = ((-1.0 * t1 * t7) + (-1.0 * t6 * t7) + (-2.0 * t2 * t4 * t6));
      out[2] = (2.0 * x * (1.0 / ipow<2>(t8)) * (1.0 + (-1.0 * t9 * (1.0 / ipow<1>(t8)))));
      out[3] = yj_big ? yj_top : ((-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * t9))));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<4>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? yj_inv_lj_pure_lambda_rescaled<4>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<4>(2.0 - p1, -x, out);
        return;
      }
      const double zs = (x * (-2.0 + p1));
      if (Kokkos::fabs(zs) > 0.5) {
        const double e = 1.0 / (1.0 + zs);
        const double z = (x * (-2.0 + p1));
        double lr[5];
        m::detail_log1prel::derivs_upto(4, z, lr);
        const double f0 = (-2.0 + p1);
        const double f1 = (-1.0 * f0);
        const double f2 = ipow<2>(e);
        const double f3 = (3.0 * f2);
        out[0] = (ipow<3>(f1) * (1.0 / ipow<4>((1.0 + (-1.0 * f1 * x)))) * (6.0 + (-6.0 * p1)));
        out[1] = (-2.0 * f0 * ipow<3>(e) * (-1.0 + (-3.0 * e) + (3.0 * e * p1)));
        out[2] = (2.0 * f2 * (1.0 / ipow<1>(f0)) * (1.0 + (-1.0 * f3) + (f3 * p1) + (-2.0 * e * p1)));
        out[3] = (-6.0 * e * (1.0 / ipow<3>(f0)) * ipow<2>((-1.0 + e)) * (-1.0 + (-1.0 * e) + (e * p1)));
        out[4] = yj_big ? yj_top : ((-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * x * (-1.0 + p1)))));
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[5];
      m::detail_log1prel::derivs_upto(4, z, lr);
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (1.0 + (-1.0 * t1 * x));
      const double t3 = (1.0 / ipow<4>(t2));
      const double t4 = ipow<2>(t1);
      const double t5 = (1.0 / ipow<3>(t2));
      const double t6 = (2.0 * p1);
      const double t7 = (2.0 + (-1.0 * t6));
      const double t8 = (t0 * x);
      const double t9 = (1.0 + t8);
      const double t10 = (1.0 / ipow<2>(t9));
      const double t11 = (1.0 / ipow<1>(t9));
      const double t12 = (-1.0 + p1);
      const double t13 = (t12 * x);
      const double t14 = (t11 * t13);
      const double t15 = ipow<2>(x);
      out[0] = (t3 * ipow<3>(t1) * (6.0 + (-6.0 * p1)));
      out[1] = ((-2.0 * t4 * t5) + (t5 * t7 * (-4.0 + t6)) + (-3.0 * t3 * t4 * t7 * x));
      out[2] = (2.0 * t10 * (1.0 + (-2.0 * t14) + (-2.0 * t11 * t8) + (3.0 * t0 * t10 * t12 * t15)));
      out[3] = (6.0 * t15 * (1.0 / ipow<3>(t9)) * (-1.0 + t14));
      out[4] = yj_big ? yj_top : ((-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * t13))));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<5>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? -yj_inv_lj_pure_lambda_rescaled<5>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<5>(2.0 - p1, -x, out);
        for (int n1 = 0; n1 <= 5; ++n1) out[n1] = -out[n1];
        return;
      }
      const double zs = (x * (-2.0 + p1));
      if (Kokkos::fabs(zs) > 0.5) {
        const double e = 1.0 / (1.0 + zs);
        const double z = (x * (-2.0 + p1));
        double lr[6];
        m::detail_log1prel::derivs_upto(5, z, lr);
        const double f0 = (-2.0 + p1);
        const double f1 = (-1.0 * f0);
        const double f2 = ipow<2>(f0);
        const double f3 = (e * p1);
        const double f4 = ipow<2>(e);
        const double f5 = (6.0 * f4);
        const double f6 = (-1.0 + e);
        out[0] = (ipow<4>(f1) * (1.0 / ipow<5>((1.0 + (-1.0 * f1 * x)))) * (24.0 + (-24.0 * p1)));
        out[1] = (6.0 * f2 * ipow<4>(e) * (-1.0 + (-4.0 * e) + (4.0 * e * p1)));
        out[2] = (-4.0 * ipow<3>(e) * (1.0 + (-1.0 * f5) + (-3.0 * f3) + (f5 * p1)));
        out[3] = (f5 * f6 * (1.0 / ipow<1>(f2)) * ((-1.0 * f6) + (-4.0 * f4) + (-2.0 * f3) + (4.0 * f4 * p1)));
        out[4] = (-24.0 * e * (1.0 / ipow<4>(f0)) * ipow<3>(f6) * (-1.0 + (-1.0 * e) + (e * p1)));
        out[5] = yj_big ? yj_top : ((-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * x * (-1.0 + p1)))));
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[6];
      m::detail_log1prel::derivs_upto(5, z, lr);
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (1.0 + (-1.0 * t1 * x));
      const double t3 = (1.0 / ipow<5>(t2));
      const double t4 = ipow<3>(t1);
      const double t5 = (1.0 / ipow<4>(t2));
      const double t6 = (6.0 + (-6.0 * p1));
      const double t7 = (t0 * x);
      const double t8 = (1.0 + t7);
      const double t9 = (1.0 / ipow<3>(t8));
      const double t10 = ipow<2>(t0);
      const double t11 = (1.0 / ipow<1>(t8));
      const double t12 = (-1.0 + p1);
      const double t13 = (6.0 * t12);
      const double t14 = ((1.0 / ipow<2>(t8)) * ipow<2>(x));
      const double t15 = (t12 * x);
      out[0] = (t3 * ipow<4>(t1) * (24.0 + (-24.0 * p1)));
      out[1] = ((-6.0 * t4 * t5) + (-3.0 * t5 * t6 * ipow<2>(t1)) + (-4.0 * t3 * t4 * t6 * x));
      out[2] = (4.0 * t9 * (5.0 + (-3.0 * p1) + (t11 * t13 * t7) + (-1.0 * t10 * t13 * t14) + (3.0 * t10 * t11 * x)));
      out[3] = (6.0 * t9 * x * (-2.0 + (-4.0 * t0 * t12 * t14) + (3.0 * t0 * t11 * x) + (3.0 * t11 * t12 * x)));
      out[4] = (24.0 * (1.0 / ipow<4>(t8)) * ipow<3>(x) * (1.0 + (-1.0 * t11 * t15)));
      out[5] = yj_big ? yj_top : ((-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * t15))));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<6>(x, p1, out);
      return;
    }
    {
      const bool yj_big = Kokkos::fabs((p1 - 2.0) * x) > 1.0e3;
      const double yj_top =
          yj_big ? yj_inv_lj_pure_lambda_rescaled<6>(2.0 - p1, -x) : 0.0;
      if (!yj_big && Kokkos::fabs(x) > 1.0e50) {
        yj_inv_lj_partials_small_z<6>(2.0 - p1, -x, out);
        return;
      }
      const double zs = (x * (-2.0 + p1));
      if (Kokkos::fabs(zs) > 0.5) {
        const double e = 1.0 / (1.0 + zs);
        const double z = (x * (-2.0 + p1));
        double lr[7];
        m::detail_log1prel::derivs_upto(6, z, lr);
        const double f0 = (-2.0 + p1);
        const double f1 = (-1.0 * f0);
        const double f2 = ipow<3>(f0);
        const double f3 = ipow<2>(e);
        const double f4 = (10.0 * f3);
        const double f5 = ipow<3>(e);
        const double f6 = (3.0 * e);
        const double f7 = (10.0 * f5);
        const double f8 = (-1.0 + e);
        const double f9 = (2.0 * e);
        out[0] = (ipow<5>(f1) * (1.0 / ipow<6>((1.0 + (-1.0 * f1 * x)))) * (120.0 + (-120.0 * p1)));
        out[1] = (-24.0 * f2 * ipow<5>(e) * (-1.0 + (-5.0 * e) + (5.0 * e * p1)));
        out[2] = (12.0 * f0 * ipow<4>(e) * (1.0 + (-1.0 * f4) + (f4 * p1) + (-4.0 * e * p1)));
        out[3] = (-12.0 * f5 * (1.0 / ipow<1>(f0)) * (-1.0 + f6 + (-1.0 * f7) + (6.0 * f3) + (f6 * p1) + (f7 * p1) + (-12.0 * f3 * p1)));
        out[4] = (24.0 * f3 * (1.0 / ipow<1>(f2)) * ipow<2>(f8) * (1.0 + (-1.0 * f9) + (-5.0 * f3) + (-1.0 * f9 * p1) + (5.0 * f3 * p1)));
        out[5] = (-120.0 * e * (1.0 / ipow<5>(f0)) * ipow<4>(f8) * (-1.0 + (-1.0 * e) + (e * p1)));
        out[6] = yj_big ? yj_top : ((-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * x * (-1.0 + p1)))));
        return;
      }
      const double z = (x * (-2.0 + p1));
      double lr[7];
      m::detail_log1prel::derivs_upto(6, z, lr);
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (1.0 + (-1.0 * t1 * x));
      const double t3 = (1.0 / ipow<6>(t2));
      const double t4 = ipow<4>(t1);
      const double t5 = (1.0 / ipow<5>(t2));
      const double t6 = (24.0 + (-24.0 * p1));
      const double t7 = (t0 * x);
      const double t8 = (1.0 + t7);
      const double t9 = (1.0 / ipow<4>(t8));
      const double t10 = ipow<2>(t0);
      const double t11 = (1.0 / ipow<1>(t8));
      const double t12 = (4.0 * t11);
      const double t13 = (-1.0 + p1);
      const double t14 = (t11 * t7);
      const double t15 = ipow<2>(x);
      const double t16 = (1.0 / ipow<2>(t8));
      const double t17 = (12.0 * t0);
      const double t18 = (1.0 / ipow<3>(t8));
      const double t19 = (t13 * x);
      const double t20 = (t11 * t19);
      const double t21 = (t15 * t16);
      const double t22 = (t13 * t21);
      out[0] = (t3 * ipow<5>(t1) * (120.0 + (-120.0 * p1)));
      out[1] = ((-24.0 * t4 * t5) + (-4.0 * t5 * t6 * ipow<3>(t1)) + (-5.0 * t3 * t4 * t6 * x));
      out[2] = (t17 * t9 * (-9.0 + (6.0 * p1) + (-12.0 * t13 * t14) + (-1.0 * t10 * t12 * x) + (10.0 * t10 * t13 * t15 * t16)));
      out[3] = (12.0 * t18 * (-1.0 + (3.0 * t20) + (6.0 * t14) + (-1.0 * t17 * t22) + (-6.0 * t10 * t21) + (10.0 * t10 * t13 * t18 * ipow<3>(x))));
      out[4] = (24.0 * t15 * t9 * (3.0 + (-4.0 * t20) + (-1.0 * t12 * t7) + (5.0 * t0 * t22)));
      out[5] = (120.0 * (1.0 / ipow<5>(t8)) * ipow<4>(x) * (-1.0 + t20));
      out[6] = yj_big ? yj_top : ((-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * t19))));
      return;
    }
  }
}

LAPLACE_HD_FORCEINLINE double yeo_johnson_dom_lo(double p1, double p2) {
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? -inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_dom_hi(double p1, double p2) {
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_img_lo(double p1, double p2) {  // R, or (1/(2 - p1), inf) above 2
  static_cast<void>(p2);
  if (!yeo_johnson_valid(p1)) return nan();
  return p1 > 2.0 ? 1.0 / (2.0 - p1) : -inf();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_img_hi(double p1, double p2) {  // R, or (-inf, -1/p1) below 0
  static_cast<void>(p2);
  if (!yeo_johnson_valid(p1)) return nan();
  return p1 < 0.0 ? -1.0 / p1 : inf();
}
// dom_hi - h(u) and h(u) - dom_lo: the range of h is R (dom = R), so both are +inf
LAPLACE_HD_FORCEINLINE double yeo_johnson_inv_to_upper(double u, double p1, double p2) {
  static_cast<void>(u);
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_inv_to_lower(double u, double p1, double p2) {
  static_cast<void>(u);
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}

}  // namespace laplace::bijector::detail_scalar
