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

template <int K>
LAPLACE_HD double yeo_johnson_fwd(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
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
    const double z = (p1 * x);
    double lr[2];
    lr[0] = m::log1prel(z);
    m::detail_log1prel::derivs_upto(1, z, lr);
    const double t0 = (1.0 + (-1.0 * p1));
    out[0] = (t0 * (1.0 / ipow<1>((1.0 + (p1 * x)))));
    out[1] = ((-1.0 * lr[0] * x) + (lr[1] * t0 * ipow<2>(x)));
    return;
  }
  if constexpr (K == 2) {
    const double zs = (p1 * x);
    if (Kokkos::fabs(zs) > 0.5) {
      const double e = 1.0 / (1.0 + zs);
      const double z = (p1 * x);
      double lr[3];
      m::detail_log1prel::derivs_upto(2, z, lr);
      const double f0 = (-1.0 + p1);
      out[0] = (f0 * p1 * (1.0 / ipow<2>((1.0 + (p1 * x)))));
      out[1] = (-1.0 * e * (1.0 / ipow<1>(p1)) * (1.0 + (-1.0 * e) + (e * p1)));
      out[2] = (-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (f0 * lr[2] * x)));
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
    out[2] = (-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (lr[2] * t0 * x)));
    return;
  }
  if constexpr (K == 3) {
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
      out[3] = (-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * x * (-1.0 + p1))));
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
    out[3] = (-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * t5)));
    return;
  }
  if constexpr (K == 4) {
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
      out[4] = (-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * x * (-1.0 + p1))));
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
    out[4] = (-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * t9)));
    return;
  }
  if constexpr (K == 5) {
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
      out[5] = (-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * x * (-1.0 + p1))));
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
    out[5] = (-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * t13)));
    return;
  }
  if constexpr (K == 6) {
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
      out[6] = (-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * x * (-1.0 + p1))));
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
    out[6] = (-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * t16)));
    return;
  }
}

template <int K>
LAPLACE_HD double yeo_johnson_inv_lj(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
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
  if constexpr (K == 1) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<1>(x, p1, out);
      return;
    }
    {
      const double z = (x * (-2.0 + p1));
      double lr[2];
      lr[0] = m::log1prel(z);
      m::detail_log1prel::derivs_upto(1, z, lr);
      const double t0 = (1.0 + (-1.0 * p1));
      out[0] = (t0 * (1.0 / ipow<1>((1.0 + (-1.0 * x * (2.0 + (-1.0 * p1)))))));
      out[1] = ((-1.0 * lr[0] * x) + (lr[1] * t0 * ipow<2>(x)));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<2>(x, p1, out);
      return;
    }
    {
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
        out[2] = (-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (f0 * lr[2] * x)));
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
      out[2] = (-1.0 * ipow<2>(x) * ((2.0 * lr[1]) + (lr[2] * t2 * x)));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<3>(x, p1, out);
      return;
    }
    {
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
        out[3] = (-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * x * (-1.0 + p1))));
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
      out[3] = (-1.0 * ipow<3>(x) * ((3.0 * lr[2]) + (lr[3] * t9)));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<4>(x, p1, out);
      return;
    }
    {
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
        out[4] = (-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * x * (-1.0 + p1))));
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
      out[4] = (-1.0 * ipow<4>(x) * ((4.0 * lr[3]) + (lr[4] * t13)));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<5>(x, p1, out);
      return;
    }
    {
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
        out[5] = (-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * x * (-1.0 + p1))));
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
      out[5] = (-1.0 * ipow<5>(x) * ((5.0 * lr[4]) + (lr[5] * t15)));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      yeo_johnson_inv_lj_pos_partials<6>(x, p1, out);
      return;
    }
    {
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
        out[6] = (-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * x * (-1.0 + p1))));
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
      out[6] = (-1.0 * ipow<6>(x) * ((6.0 * lr[5]) + (lr[6] * t19)));
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
