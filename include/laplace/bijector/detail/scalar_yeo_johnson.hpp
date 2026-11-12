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
// is exprel_d1, that of log1prel is log1prel_d1, and so on to order 6. Every form is
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

template <int K>
LAPLACE_HD double yeo_johnson_fwd(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      const double t0 = Kokkos::log1p(x);
      return (t0 * m::exprel((t0 * p1)));
    }
    {
      const double t0 = Kokkos::log1p((-1.0 * x));
      return (-1.0 * t0 * m::exprel((t0 * (2.0 + (-1.0 * p1)))));
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      return pow_pos(1.0 + x, (-1.0 + p1));
    }
    {
      return pow_pos(1.0 + (-1.0 * x), (1.0 + (-1.0 * p1)));
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * pow_pos(1.0 + x, (-2.0 + p1)));
    }
    {
      return ((-1.0 + p1) * pow_pos(1.0 + (-1.0 * x), (-1.0 * p1)));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * (-2.0 + p1) * pow_pos(1.0 + x, (-3.0 + p1)));
    }
    {
      return (-1.0 * p1 * (1.0 + (-1.0 * p1)) * pow_pos(1.0 + (-1.0 * x), (-1.0 + (-1.0 * p1))));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * (-3.0 + p1) * (-2.0 + p1) * pow_pos(1.0 + x, (-4.0 + p1)));
    }
    {
      return (p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)) * pow_pos(1.0 + (-1.0 * x), (-2.0 + (-1.0 * p1))));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * (-4.0 + p1) * (-3.0 + p1) * (-2.0 + p1) * pow_pos(1.0 + x, (-5.0 + p1)));
    }
    {
      return (-1.0 * p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)) * (-2.0 + (-1.0 * p1)) * pow_pos(1.0 + (-1.0 * x), (-3.0 + (-1.0 * p1))));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      return ((-1.0 + p1) * (-5.0 + p1) * (-4.0 + p1) * (-3.0 + p1) * (-2.0 + p1) * pow_pos(1.0 + x, (-6.0 + p1)));
    }
    {
      return (p1 * (1.0 + (-1.0 * p1)) * (-1.0 + (-1.0 * p1)) * (-3.0 + (-1.0 * p1)) * (-2.0 + (-1.0 * p1)) * pow_pos(1.0 + (-1.0 * x), (-4.0 + (-1.0 * p1))));
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
      const double t0 = Kokkos::log1p(x);
      out[0] = pow_pos(1.0 + x, (-1.0 + p1));
      out[1] = (ipow<2>(t0) * m::exprel_d1((t0 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 * x);
      const double t1 = Kokkos::log1p(t0);
      out[0] = pow_pos(1.0 + t0, (1.0 + (-1.0 * p1)));
      out[1] = (ipow<2>(t1) * m::exprel_d1((t1 * (2.0 + (-1.0 * p1)))));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double t0 = (-1.0 + p1);
      const double t1 = Kokkos::log1p(x);
      out[0] = (t0 * pow_pos(1.0 + x, (-2.0 + p1)));
      out[1] = (t1 * pow_pos(1.0 + x, t0));
      out[2] = (ipow<3>(t1) * m::exprel_d2((t1 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * x);
      const double t2 = Kokkos::log1p(t1);
      out[0] = (t0 * pow_pos(1.0 + t1, (-1.0 * p1)));
      out[1] = (-1.0 * t2 * pow_pos(1.0 + t1, (-1.0 * t0)));
      out[2] = (-1.0 * ipow<3>(t2) * m::exprel_d2((-1.0 * t2 * (-2.0 + p1))));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 + p1);
      const double t2 = pow_pos(1.0 + x, t0);
      const double t3 = Kokkos::log1p(x);
      out[0] = (t0 * t1 * pow_pos(1.0 + x, (-3.0 + p1)));
      out[1] = (t2 + (t1 * t2 * t3));
      out[2] = (ipow<2>(t3) * pow_pos(1.0 + x, t1));
      out[3] = (ipow<4>(t3) * m::exprel_d3((t3 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-1.0 * x);
      const double t3 = pow_pos(1.0 + t2, (-1.0 * p1));
      const double t4 = Kokkos::log1p(t2);
      out[0] = (-1.0 * t1 * p1 * pow_pos(1.0 + t2, (-1.0 + (-1.0 * p1))));
      out[1] = (t3 + (-1.0 * t0 * t3 * t4));
      out[2] = (ipow<2>(t4) * pow_pos(1.0 + t2, t1));
      out[3] = (ipow<4>(t4) * m::exprel_d3((-1.0 * t4 * (-2.0 + p1))));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double t0 = (-3.0 + p1);
      const double t1 = (-2.0 + p1);
      const double t2 = (-1.0 + p1);
      const double t3 = pow_pos(1.0 + x, t0);
      const double t4 = (t2 * t3);
      const double t5 = Kokkos::log1p(x);
      out[0] = (t0 * t1 * t2 * pow_pos(1.0 + x, (-4.0 + p1)));
      out[1] = (t4 + (t1 * t3) + (t1 * t4 * t5));
      out[2] = (t5 * (2.0 + (t2 * t5)) * pow_pos(1.0 + x, t1));
      out[3] = (ipow<3>(t5) * pow_pos(1.0 + x, t2));
      out[4] = (ipow<5>(t5) * m::exprel_d4((t5 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-1.0 + (-1.0 * p1));
      const double t3 = (-1.0 * x);
      const double t4 = pow_pos(1.0 + t3, t2);
      const double t5 = (t4 * p1);
      const double t6 = Kokkos::log1p(t3);
      out[0] = (t1 * t2 * p1 * pow_pos(1.0 + t3, (-2.0 + (-1.0 * p1))));
      out[1] = (t5 + (-1.0 * t1 * t4) + (t1 * t5 * t6));
      out[2] = (t6 * (-2.0 + (t0 * t6)) * pow_pos(1.0 + t3, (-1.0 * p1)));
      out[3] = (-1.0 * ipow<3>(t6) * pow_pos(1.0 + t3, t1));
      out[4] = (-1.0 * ipow<5>(t6) * m::exprel_d4((-1.0 * t6 * (-2.0 + p1))));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double t0 = (-1.0 + p1);
      const double t1 = (-4.0 + p1);
      const double t2 = (-3.0 + p1);
      const double t3 = (-2.0 + p1);
      const double t4 = (t2 * t3);
      const double t5 = pow_pos(1.0 + x, t1);
      const double t6 = (t4 * t5);
      const double t7 = (t0 * t5);
      const double t8 = Kokkos::log1p(x);
      const double t9 = (t0 * t8);
      const double t10 = ipow<2>(t8);
      out[0] = (t0 * t1 * t4 * pow_pos(1.0 + x, (-5.0 + p1)));
      out[1] = (t6 + (t2 * t7) + (t3 * t7) + (t6 * t9));
      out[2] = ((2.0 + (2.0 * t9) + (2.0 * t3 * t8) + (t0 * t10 * t3)) * pow_pos(1.0 + x, t2));
      out[3] = (t10 * (3.0 + t9) * pow_pos(1.0 + x, t3));
      out[4] = (ipow<4>(t8) * pow_pos(1.0 + x, t0));
      out[5] = (ipow<6>(t8) * m::exprel_d5((t8 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-2.0 + (-1.0 * p1));
      const double t3 = (-1.0 * x);
      const double t4 = (1.0 + p1);
      const double t5 = (-1.0 * t4);
      const double t6 = (t5 * p1);
      const double t7 = pow_pos(1.0 + t3, t2);
      const double t8 = (t1 * t7);
      const double t9 = Kokkos::log1p(t3);
      const double t10 = (t9 * p1);
      const double t11 = (t0 * t9);
      const double t12 = ipow<2>(t9);
      out[0] = (-1.0 * t1 * t2 * t6 * pow_pos(1.0 + t3, (-3.0 + (-1.0 * p1))));
      out[1] = ((-1.0 * t6 * t7) + (-1.0 * t8 * p1) + (t1 * t5 * t7) + (-1.0 * t10 * t5 * t8));
      out[2] = ((2.0 + (-2.0 * t10) + (-2.0 * t11) + (t0 * t12 * p1)) * pow_pos(1.0 + t3, (-1.0 * t4)));
      out[3] = (t12 * (3.0 + (-1.0 * t11)) * pow_pos(1.0 + t3, (-1.0 * p1)));
      out[4] = (ipow<4>(t9) * pow_pos(1.0 + t3, t1));
      out[5] = (ipow<6>(t9) * m::exprel_d5((-1.0 * t9 * (-2.0 + p1))));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double t0 = (-4.0 + p1);
      const double t1 = (-5.0 + p1);
      const double t2 = (-3.0 + p1);
      const double t3 = (-1.0 + p1);
      const double t4 = (-2.0 + p1);
      const double t5 = (t3 * t4);
      const double t6 = (t2 * t5);
      const double t7 = pow_pos(1.0 + x, t1);
      const double t8 = (t0 * t7);
      const double t9 = (t2 * t8);
      const double t10 = (t4 * t9);
      const double t11 = Kokkos::log1p(x);
      const double t12 = (t11 * t3);
      const double t13 = (t11 * t4);
      const double t14 = (2.0 * t12);
      const double t15 = (t5 * ipow<2>(t11));
      out[0] = (t0 * t1 * t6 * pow_pos(1.0 + x, (-6.0 + p1)));
      out[1] = (t10 + (t10 * t12) + (t3 * t9) + (t5 * t8) + (t6 * t7));
      out[2] = ((-12.0 + (6.0 * p1) + (t14 * t2) + (t14 * t4) + (t15 * t2) + (2.0 * t13 * t2)) * pow_pos(1.0 + x, t0));
      out[3] = (t11 * (6.0 + t15 + (3.0 * t12) + (3.0 * t13)) * pow_pos(1.0 + x, t2));
      out[4] = (ipow<3>(t11) * (4.0 + t12) * pow_pos(1.0 + x, t4));
      out[5] = (ipow<5>(t11) * pow_pos(1.0 + x, t3));
      out[6] = (ipow<7>(t11) * m::exprel_d6((t11 * p1)));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-3.0 + (-1.0 * p1));
      const double t3 = (2.0 + p1);
      const double t4 = (-1.0 * t3);
      const double t5 = (1.0 + p1);
      const double t6 = (-1.0 * t5);
      const double t7 = (-1.0 * x);
      const double t8 = pow_pos(1.0 + t7, t2);
      const double t9 = (t1 * t8 * p1);
      const double t10 = (t4 * t6 * t8);
      const double t11 = (t1 * t10);
      const double t12 = Kokkos::log1p(t7);
      const double t13 = (t12 * p1);
      const double t14 = (t0 * t12);
      const double t15 = (2.0 * t5);
      const double t16 = ipow<2>(t12);
      out[0] = (t1 * t2 * t4 * t6 * p1 * pow_pos(1.0 + t7, (-4.0 + (-1.0 * p1))));
      out[1] = ((-1.0 * t11) + (t10 * p1) + (t11 * t13) + (t4 * t9) + (t6 * t9));
      out[2] = (((6.0 * p1) + (-1.0 * t13 * t15) + (-1.0 * t14 * t15) + (-2.0 * t14 * p1) + (t0 * t16 * t5 * p1)) * pow_pos(1.0 + t7, (-1.0 * t3)));
      out[3] = (t12 * (-6.0 + (3.0 * t0 * t12) + (3.0 * t12 * p1) + (-1.0 * t0 * t16 * p1)) * pow_pos(1.0 + t7, (-1.0 * t5)));
      out[4] = (ipow<3>(t12) * (-4.0 + t14) * pow_pos(1.0 + t7, (-1.0 * p1)));
      out[5] = (-1.0 * ipow<5>(t12) * pow_pos(1.0 + t7, t1));
      out[6] = (-1.0 * ipow<7>(t12) * m::exprel_d6((-1.0 * t12 * (-2.0 + p1))));
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
      return Kokkos::expm1((x * m::log1prel((p1 * x))));
    }
    {
      return (-1.0 * Kokkos::expm1((-1.0 * x * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
    }
  }
  if constexpr (K == 1) {
    if (x >= 0.0) {
      return m::exp((x * (1.0 + (-1.0 * p1)) * m::log1prel((p1 * x))));
    }
    {
      return m::exp((x * (1.0 + (-1.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1))))));
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      return ((1.0 + (-1.0 * p1)) * m::exp((x * (1.0 + (-2.0 * p1)) * m::log1prel((p1 * x)))));
    }
    {
      return ((1.0 + (-1.0 * p1)) * m::exp((x * (3.0 + (-2.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((x * (1.0 + (-3.0 * p1)) * m::log1prel((p1 * x)))));
    }
    {
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * m::exp((x * (5.0 + (-3.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((x * (1.0 + (-4.0 * p1)) * m::log1prel((p1 * x)))));
    }
    {
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * m::exp((x * (7.0 + (-4.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-4.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((x * (1.0 + (-5.0 * p1)) * m::log1prel((p1 * x)))));
    }
    {
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * (7.0 + (-4.0 * p1)) * m::exp((x * (9.0 + (-5.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      return ((1.0 + (-1.0 * p1)) * (1.0 + (-5.0 * p1)) * (1.0 + (-4.0 * p1)) * (1.0 + (-3.0 * p1)) * (1.0 + (-2.0 * p1)) * m::exp((x * (1.0 + (-6.0 * p1)) * m::log1prel((p1 * x)))));
    }
    {
      return ((1.0 + (-1.0 * p1)) * (3.0 + (-2.0 * p1)) * (5.0 + (-3.0 * p1)) * (7.0 + (-4.0 * p1)) * (9.0 + (-5.0 * p1)) * m::exp((x * (11.0 + (-6.0 * p1)) * m::log1prel((-1.0 * x * (2.0 + (-1.0 * p1)))))));
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
      const double t0 = (p1 * x);
      const double t1 = (x * m::log1prel(t0));
      out[0] = m::exp((t1 * (1.0 + (-1.0 * p1))));
      out[1] = (ipow<2>(x) * m::log1prel_d1(t0) * m::exp(t1));
      return;
    }
    {
      const double t0 = (-1.0 * x * (2.0 + (-1.0 * p1)));
      const double t1 = (x * m::log1prel(t0));
      out[0] = m::exp((t1 * (1.0 + (-1.0 * p1))));
      out[1] = (ipow<2>(x) * m::log1prel_d1(t0) * m::exp((-1.0 * t1)));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (p1 * x);
      const double t2 = (x * m::log1prel(t1));
      const double t3 = m::log1prel_d1(t1);
      out[0] = (t0 * m::exp((t2 * (1.0 + (-2.0 * p1)))));
      out[1] = (((-1.0 * t2) + (t0 * t3 * ipow<2>(x))) * m::exp((t0 * t2)));
      out[2] = (ipow<3>(x) * ((x * ipow<2>(t3)) + m::log1prel_d2(t1)) * m::exp(t2));
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (-2.0 + p1);
      const double t2 = (t1 * x);
      const double t3 = (x * m::log1prel(t2));
      const double t4 = (t1 * x);
      out[0] = (t0 * m::exp((t3 * (3.0 + (-2.0 * p1)))));
      out[1] = (((-1.0 * t3) + (t0 * ipow<2>(x) * m::log1prel_d1(t2))) * m::exp((t0 * t3)));
      out[2] = (-1.0 * ipow<3>(x) * ((-1.0 * m::log1prel_d2(t4)) + (x * ipow<2>(m::log1prel_d1(t4)))) * m::exp((-1.0 * x * m::log1prel(t4))));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double t0 = (1.0 + (-2.0 * p1));
      const double t1 = (-1.0 + p1);
      const double t2 = (-1.0 * t1);
      const double t3 = (p1 * x);
      const double t4 = m::log1prel(t3);
      const double t5 = (t4 * x);
      const double t6 = m::exp((t0 * t5));
      const double t7 = ipow<2>(x);
      const double t8 = m::log1prel_d1(t3);
      const double t9 = m::log1prel_d2(t3);
      const double t10 = (t1 * x);
      out[0] = (t0 * t2 * m::exp((t5 * (1.0 + (-3.0 * p1)))));
      out[1] = ((-1.0 * t6) + (t2 * t6 * ((-2.0 * t5) + (t0 * t7 * t8))));
      out[2] = (t7 * (ipow<2>((t4 + (t10 * t8))) + (-2.0 * t8) + (-1.0 * t10 * t9)) * m::exp((-1.0 * t1 * t5)));
      out[3] = (ipow<4>(x) * ((t7 * ipow<3>(t8)) + (3.0 * t8 * t9 * x) + m::log1prel_d3(t3)) * m::exp(t5));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (3.0 + (-2.0 * p1));
      const double t3 = (-2.0 + p1);
      const double t4 = (t3 * x);
      const double t5 = (x * m::log1prel(t4));
      const double t6 = m::exp((t2 * t5));
      const double t7 = ipow<2>(x);
      const double t8 = (t3 * x);
      const double t9 = m::log1prel(t8);
      const double t10 = (t9 * x);
      const double t11 = m::log1prel_d1(t8);
      const double t12 = m::log1prel_d2(t8);
      const double t13 = (t0 * x);
      out[0] = (t1 * t2 * m::exp((t5 * (5.0 + (-3.0 * p1)))));
      out[1] = ((-1.0 * t6) + (t1 * t6 * ((-2.0 * t5) + (t2 * t7 * m::log1prel_d1(t4)))));
      out[2] = (t7 * (ipow<2>((t9 + (t11 * t13))) + (-2.0 * t11) + (-1.0 * t12 * t13)) * m::exp((-1.0 * t0 * t10)));
      out[3] = (ipow<4>(x) * ((t7 * ipow<3>(t11)) + (-3.0 * t11 * t12 * x) + m::log1prel_d3(t8)) * m::exp((-1.0 * t10)));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
      const double t0 = (1.0 + (-3.0 * p1));
      const double t1 = (-1.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (-1.0 + p1);
      const double t4 = (-1.0 * t3);
      const double t5 = (p1 * x);
      const double t6 = m::log1prel(t5);
      const double t7 = (t6 * x);
      const double t8 = m::exp((t0 * t7));
      const double t9 = ipow<2>(x);
      const double t10 = m::log1prel_d1(t5);
      const double t11 = (2.0 * t10);
      const double t12 = (t1 * x);
      const double t13 = (4.0 * t10);
      const double t14 = m::log1prel_d2(t5);
      const double t15 = (t3 * x);
      const double t16 = ipow<3>(x);
      const double t17 = m::log1prel_d3(t5);
      const double t18 = (t6 + (t10 * t15));
      out[0] = (t0 * t2 * t4 * m::exp((t7 * (1.0 + (-4.0 * p1)))));
      out[1] = ((-1.0 * t2 * t8) + (-2.0 * t4 * t8) + (t2 * t4 * t8 * ((-3.0 * t7) + (t0 * t10 * t9))));
      out[2] = (x * ((4.0 * t6) + (t11 * t12) + (t15 * (t13 + (-1.0 * ipow<2>(((2.0 * t6) + (t10 * t12)))) + (t12 * t14)))) * m::exp((-1.0 * t1 * t7)));
      out[3] = (t16 * ((-1.0 * ipow<3>(t18)) + (-3.0 * t14) + (-1.0 * t15 * t17) + (3.0 * t18 * (t11 + (t14 * t15)))) * m::exp((-1.0 * t3 * t7)));
      out[4] = (ipow<5>(x) * ((t16 * ipow<4>(t10)) + (3.0 * x * ipow<2>(t14)) + (t13 * t17 * x) + (6.0 * t14 * t9 * ipow<2>(t10)) + m::log1prel_d4(t5)) * m::exp(t7));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-3.0 + (2.0 * p1));
      const double t3 = (-1.0 * t2);
      const double t4 = (5.0 + (-3.0 * p1));
      const double t5 = (-2.0 + p1);
      const double t6 = (t5 * x);
      const double t7 = (x * m::log1prel(t6));
      const double t8 = m::exp((t4 * t7));
      const double t9 = ipow<2>(x);
      const double t10 = (t5 * x);
      const double t11 = m::log1prel(t10);
      const double t12 = (t11 * x);
      const double t13 = m::log1prel_d1(t10);
      const double t14 = (2.0 * t13);
      const double t15 = (t2 * x);
      const double t16 = (4.0 * t13);
      const double t17 = m::log1prel_d2(t10);
      const double t18 = (t0 * x);
      const double t19 = ipow<3>(x);
      const double t20 = m::log1prel_d3(t10);
      const double t21 = (t11 + (t13 * t18));
      out[0] = (t1 * t3 * t4 * m::exp((t7 * (7.0 + (-4.0 * p1)))));
      out[1] = ((-1.0 * t3 * t8) + (-2.0 * t1 * t8) + (t1 * t3 * t8 * ((-3.0 * t7) + (t4 * t9 * m::log1prel_d1(t6)))));
      out[2] = (x * ((4.0 * t11) + (t14 * t15) + (t18 * (t16 + (-1.0 * ipow<2>(((2.0 * t11) + (t13 * t15)))) + (t15 * t17)))) * m::exp((-1.0 * t12 * t2)));
      out[3] = (t19 * ((-1.0 * ipow<3>(t21)) + (-3.0 * t17) + (-1.0 * t18 * t20) + (3.0 * t21 * (t14 + (t17 * t18)))) * m::exp((-1.0 * t0 * t12)));
      out[4] = (-1.0 * ipow<5>(x) * ((-1.0 * m::log1prel_d4(t10)) + (t19 * ipow<4>(t13)) + (3.0 * x * ipow<2>(t17)) + (t16 * t20 * x) + (-6.0 * t17 * t9 * ipow<2>(t13))) * m::exp((-1.0 * t12)));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
      const double t0 = (1.0 + (-4.0 * p1));
      const double t1 = (-1.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (p1 * x);
      const double t4 = m::log1prel(t3);
      const double t5 = (t4 * x);
      const double t6 = (-1.0 + (3.0 * p1));
      const double t7 = (-1.0 * t6);
      const double t8 = (-1.0 + p1);
      const double t9 = (-1.0 * t8);
      const double t10 = (t7 * t9);
      const double t11 = m::exp((t0 * t5));
      const double t12 = (t11 * t2);
      const double t13 = m::log1prel_d1(t3);
      const double t14 = ipow<2>(x);
      const double t15 = (t13 * t14);
      const double t16 = (t6 * x);
      const double t17 = ((3.0 * t4) + (t13 * t16));
      const double t18 = (t8 * x);
      const double t19 = (t1 * x);
      const double t20 = m::log1prel_d2(t3);
      const double t21 = (3.0 * t20);
      const double t22 = ((2.0 * t4) + (t13 * t19));
      const double t23 = m::log1prel_d3(t3);
      const double t24 = ipow<4>(x);
      const double t25 = m::log1prel_d4(t3);
      const double t26 = (t4 + (t13 * t18));
      const double t27 = ((2.0 * t13) + (t18 * t20));
      const double t28 = (10.0 * t23);
      out[0] = (t0 * t10 * t2 * m::exp((t5 * (1.0 + (-5.0 * p1)))));
      out[1] = ((-1.0 * t12 * t7) + (-3.0 * t12 * t9) + (-2.0 * t10 * t11) + (t11 * t2 * t7 * t9 * ((-4.0 * t5) + (t0 * t15))));
      out[2] = ((4.0 + (-4.0 * t17 * t18) + (-2.0 * t17 * t19) + (-1.0 * t1 * t14 * t8 * ((-1.0 * ipow<2>(t17)) + (6.0 * t13) + (t16 * t20)))) * m::exp((-1.0 * t5 * t6)));
      out[3] = (t14 * ((-3.0 * ipow<2>(t22)) + (12.0 * t13) + (t18 * (ipow<3>(t22) + (6.0 * t20) + (t19 * t23) + (-3.0 * t22 * ((4.0 * t13) + (t19 * t20))))) + (t19 * t21)) * m::exp((-1.0 * t1 * t5)));
      out[4] = (t24 * (ipow<4>(t26) + (-4.0 * t23) + (3.0 * ipow<2>(t27)) + (-1.0 * t18 * t25) + (-6.0 * t27 * ipow<2>(t26)) + (4.0 * t26 * (t21 + (t18 * t23)))) * m::exp((-1.0 * t5 * t8)));
      out[5] = (ipow<6>(x) * ((t24 * ipow<5>(t13)) + (15.0 * t15 * ipow<2>(t20)) + (t14 * t28 * ipow<2>(t13)) + (t20 * t28 * x) + (5.0 * t13 * t25 * x) + (10.0 * t20 * ipow<3>(t13) * ipow<3>(x)) + m::log1prel_d5(t3)) * m::exp(t5));
      return;
    }
    {
      const double t0 = (7.0 + (-4.0 * p1));
      const double t1 = (-3.0 + (2.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (-2.0 + p1);
      const double t4 = (t3 * x);
      const double t5 = (x * m::log1prel(t4));
      const double t6 = (-5.0 + (3.0 * p1));
      const double t7 = (-1.0 * t6);
      const double t8 = (-1.0 + p1);
      const double t9 = (-1.0 * t8);
      const double t10 = (t7 * t9);
      const double t11 = m::exp((t0 * t5));
      const double t12 = (t11 * t2);
      const double t13 = ipow<2>(x);
      const double t14 = (t3 * x);
      const double t15 = m::log1prel(t14);
      const double t16 = (t15 * x);
      const double t17 = m::log1prel_d1(t14);
      const double t18 = (t6 * x);
      const double t19 = ((3.0 * t15) + (t17 * t18));
      const double t20 = (t8 * x);
      const double t21 = (t1 * x);
      const double t22 = m::log1prel_d2(t14);
      const double t23 = (3.0 * t22);
      const double t24 = ((2.0 * t15) + (t17 * t21));
      const double t25 = m::log1prel_d3(t14);
      const double t26 = ipow<4>(x);
      const double t27 = m::log1prel_d4(t14);
      const double t28 = (t15 + (t17 * t20));
      const double t29 = ((2.0 * t17) + (t20 * t22));
      const double t30 = (10.0 * t22);
      out[0] = (t0 * t10 * t2 * m::exp((t5 * (9.0 + (-5.0 * p1)))));
      out[1] = ((-1.0 * t12 * t7) + (-3.0 * t12 * t9) + (-2.0 * t10 * t11) + (t11 * t2 * t7 * t9 * ((-4.0 * t5) + (t0 * t13 * m::log1prel_d1(t4)))));
      out[2] = ((4.0 + (-4.0 * t19 * t20) + (-2.0 * t19 * t21) + (-1.0 * t1 * t13 * t8 * ((-1.0 * ipow<2>(t19)) + (6.0 * t17) + (t18 * t22)))) * m::exp((-1.0 * t16 * t6)));
      out[3] = (t13 * ((-3.0 * ipow<2>(t24)) + (12.0 * t17) + (t20 * (ipow<3>(t24) + (6.0 * t22) + (t21 * t25) + (-3.0 * t24 * ((4.0 * t17) + (t21 * t22))))) + (t21 * t23)) * m::exp((-1.0 * t1 * t16)));
      out[4] = (t26 * (ipow<4>(t28) + (-4.0 * t25) + (3.0 * ipow<2>(t29)) + (-1.0 * t20 * t27) + (-6.0 * t29 * ipow<2>(t28)) + (4.0 * t28 * (t23 + (t20 * t25)))) * m::exp((-1.0 * t16 * t8)));
      out[5] = (ipow<6>(x) * ((t26 * ipow<5>(t17)) + (-1.0 * t25 * t30 * x) + (-1.0 * t30 * ipow<3>(t17) * ipow<3>(x)) + (-5.0 * t17 * t27 * x) + (10.0 * t13 * t25 * ipow<2>(t17)) + (15.0 * t13 * t17 * ipow<2>(t22)) + m::log1prel_d5(t14)) * m::exp((-1.0 * t16)));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
      const double t0 = (1.0 + (-5.0 * p1));
      const double t1 = (-1.0 + (4.0 * p1));
      const double t2 = (-1.0 * t1);
      const double t3 = (-1.0 + (3.0 * p1));
      const double t4 = (-1.0 * t3);
      const double t5 = (-1.0 + (2.0 * p1));
      const double t6 = (-1.0 * t5);
      const double t7 = (-1.0 + p1);
      const double t8 = (-1.0 * t7);
      const double t9 = (p1 * x);
      const double t10 = m::log1prel(t9);
      const double t11 = (t10 * x);
      const double t12 = m::exp((t0 * t11));
      const double t13 = (t12 * t4 * t6);
      const double t14 = (t12 * t2 * t8);
      const double t15 = m::log1prel_d1(t9);
      const double t16 = ipow<2>(x);
      const double t17 = (t15 * t16);
      const double t18 = (t1 * x);
      const double t19 = ((4.0 * t10) + (t15 * t18));
      const double t20 = (t5 * x);
      const double t21 = (t3 * x);
      const double t22 = (t19 * t21);
      const double t23 = m::log1prel_d2(t9);
      const double t24 = (t16 * t5 * t7);
      const double t25 = (t15 * t21);
      const double t26 = (t25 + (3.0 * t10));
      const double t27 = (6.0 * t15);
      const double t28 = (t27 + (t21 * t23));
      const double t29 = (t28 + (-1.0 * ipow<2>(t26)));
      const double t30 = (t7 * x);
      const double t31 = m::log1prel_d3(t9);
      const double t32 = ipow<3>(x);
      const double t33 = (4.0 * t31);
      const double t34 = ((2.0 * t10) + (t15 * t20));
      const double t35 = ((4.0 * t15) + (t20 * t23));
      const double t36 = m::log1prel_d4(t9);
      const double t37 = ipow<5>(x);
      const double t38 = m::log1prel_d5(t9);
      const double t39 = (t10 + (t15 * t30));
      const double t40 = ((2.0 * t15) + (t23 * t30));
      const double t41 = ((3.0 * t23) + (t30 * t31));
      const double t42 = (15.0 * t16);
      const double t43 = (15.0 * t23);
      const double t44 = ipow<2>(t15);
      out[0] = (t0 * t2 * t4 * t6 * t8 * m::exp((t11 * (1.0 + (-6.0 * p1)))));
      out[1] = ((-1.0 * t13 * t2) + (-4.0 * t13 * t8) + (-3.0 * t14 * t6) + (-2.0 * t14 * t4) + (t12 * t2 * t4 * t6 * t8 * ((-5.0 * t11) + (t0 * t17))));
      out[2] = ((22.0 + (-36.0 * p1) + (2.0 * t22 * t5) + (4.0 * t22 * t7) + (t24 * t3 * ((-1.0 * ipow<2>(t19)) + (8.0 * t15) + (t18 * t23))) + (6.0 * t19 * t20 * t7)) * m::exp((-1.0 * t1 * t11)));
      out[3] = (-1.0 * x * ((12.0 * t25) + (36.0 * t10) + (t24 * (ipow<3>(t26) + (9.0 * t23) + (t21 * t31) + (-3.0 * t26 * t28))) + (3.0 * t20 * t29) + (6.0 * t29 * t30)) * m::exp((-1.0 * t11 * t3)));
      out[4] = (t32 * ((4.0 * ipow<3>(t34)) + (24.0 * t23) + (t20 * t33) + (t30 * ((-1.0 * ipow<4>(t34)) + (-3.0 * ipow<2>(t35)) + (8.0 * t31) + (t20 * t36) + (-4.0 * t34 * ((6.0 * t23) + (t20 * t31))) + (6.0 * t35 * ipow<2>(t34)))) + (-12.0 * t34 * t35)) * m::exp((-1.0 * t11 * t5)));
      out[5] = (t37 * ((-1.0 * ipow<5>(t39)) + (-5.0 * t36) + (-1.0 * t30 * t38) + (-15.0 * t39 * ipow<2>(t40)) + (-10.0 * t41 * ipow<2>(t39)) + (5.0 * t39 * (t33 + (t30 * t36))) + (10.0 * t40 * t41) + (10.0 * t40 * ipow<3>(t39))) * m::exp((-1.0 * t11 * t7)));
      out[6] = (ipow<7>(x) * ((t37 * ipow<6>(t15)) + (t42 * ipow<3>(t23)) + (10.0 * x * ipow<2>(t31)) + (t27 * t38 * x) + (t36 * t42 * t44) + (t36 * t43 * x) + (t43 * ipow<4>(t15) * ipow<4>(x)) + (20.0 * t31 * t32 * ipow<3>(t15)) + (45.0 * t32 * t44 * ipow<2>(t23)) + (60.0 * t17 * t23 * t31) + m::log1prel_d6(t9)) * m::exp(t11));
      return;
    }
    {
      const double t0 = (-1.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (-3.0 + (2.0 * p1));
      const double t3 = (-1.0 * t2);
      const double t4 = (-5.0 + (3.0 * p1));
      const double t5 = (-1.0 * t4);
      const double t6 = (-7.0 + (4.0 * p1));
      const double t7 = (-1.0 * t6);
      const double t8 = (9.0 + (-5.0 * p1));
      const double t9 = (-2.0 + p1);
      const double t10 = (t9 * x);
      const double t11 = (x * m::log1prel(t10));
      const double t12 = m::exp((t11 * t8));
      const double t13 = (t12 * t3 * t5);
      const double t14 = (t1 * t12 * t7);
      const double t15 = ipow<2>(x);
      const double t16 = (t9 * x);
      const double t17 = m::log1prel(t16);
      const double t18 = (t17 * x);
      const double t19 = m::log1prel_d1(t16);
      const double t20 = (t6 * x);
      const double t21 = ((4.0 * t17) + (t19 * t20));
      const double t22 = (t2 * x);
      const double t23 = (t4 * x);
      const double t24 = (t21 * t23);
      const double t25 = m::log1prel_d2(t16);
      const double t26 = (t0 * t15 * t2);
      const double t27 = (t19 * t23);
      const double t28 = (t27 + (3.0 * t17));
      const double t29 = (6.0 * t19);
      const double t30 = (t29 + (t23 * t25));
      const double t31 = (t30 + (-1.0 * ipow<2>(t28)));
      const double t32 = (t0 * x);
      const double t33 = m::log1prel_d3(t16);
      const double t34 = ipow<3>(x);
      const double t35 = (4.0 * t33);
      const double t36 = ((2.0 * t17) + (t19 * t22));
      const double t37 = ((4.0 * t19) + (t22 * t25));
      const double t38 = m::log1prel_d4(t16);
      const double t39 = ipow<5>(x);
      const double t40 = m::log1prel_d5(t16);
      const double t41 = (t17 + (t19 * t32));
      const double t42 = ((2.0 * t19) + (t25 * t32));
      const double t43 = ((3.0 * t25) + (t32 * t33));
      const double t44 = (15.0 * t15);
      const double t45 = (15.0 * t25);
      const double t46 = ipow<2>(t19);
      out[0] = (t1 * t3 * t5 * t7 * t8 * m::exp((t11 * (11.0 + (-6.0 * p1)))));
      out[1] = ((-1.0 * t13 * t7) + (-4.0 * t1 * t13) + (-3.0 * t14 * t3) + (-2.0 * t14 * t5) + (t1 * t12 * t3 * t5 * t7 * ((-5.0 * t11) + (t15 * t8 * m::log1prel_d1(t10)))));
      out[2] = ((50.0 + (-36.0 * p1) + (2.0 * t2 * t24) + (4.0 * t0 * t24) + (t26 * t4 * ((-1.0 * ipow<2>(t21)) + (8.0 * t19) + (t20 * t25))) + (6.0 * t0 * t21 * t22)) * m::exp((-1.0 * t18 * t6)));
      out[3] = (-1.0 * x * ((12.0 * t27) + (36.0 * t17) + (t26 * (ipow<3>(t28) + (9.0 * t25) + (t23 * t33) + (-3.0 * t28 * t30))) + (3.0 * t22 * t31) + (6.0 * t31 * t32)) * m::exp((-1.0 * t18 * t4)));
      out[4] = (t34 * ((4.0 * ipow<3>(t36)) + (24.0 * t25) + (t22 * t35) + (t32 * ((-1.0 * ipow<4>(t36)) + (-3.0 * ipow<2>(t37)) + (8.0 * t33) + (t22 * t38) + (-4.0 * t36 * ((6.0 * t25) + (t22 * t33))) + (6.0 * t37 * ipow<2>(t36)))) + (-12.0 * t36 * t37)) * m::exp((-1.0 * t18 * t2)));
      out[5] = (t39 * ((-1.0 * ipow<5>(t41)) + (-5.0 * t38) + (-1.0 * t32 * t40) + (-15.0 * t41 * ipow<2>(t42)) + (-10.0 * t43 * ipow<2>(t41)) + (5.0 * t41 * (t35 + (t32 * t38))) + (10.0 * t42 * t43) + (10.0 * t42 * ipow<3>(t41))) * m::exp((-1.0 * t0 * t18)));
      out[6] = (-1.0 * ipow<7>(x) * ((-1.0 * m::log1prel_d6(t16)) + (t39 * ipow<6>(t19)) + (-1.0 * t44 * ipow<3>(t25)) + (10.0 * x * ipow<2>(t33)) + (t29 * t40 * x) + (t38 * t45 * x) + (-1.0 * t38 * t44 * t46) + (-1.0 * t45 * ipow<4>(t19) * ipow<4>(x)) + (20.0 * t33 * t34 * ipow<3>(t19)) + (45.0 * t34 * t46 * ipow<2>(t25)) + (-60.0 * t15 * t19 * t25 * t33)) * m::exp((-1.0 * t18)));
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

template <int K>
LAPLACE_HD double yeo_johnson_inv_lj(double x, double p1, double p2) {
  static_assert(K >= 0 && K <= 6, "order 0 to 6");
  static_cast<void>(p2);
  if constexpr (K == 0) {
    if (x >= 0.0) {
      return (x * (1.0 + (-1.0 * p1)) * m::log1prel((p1 * x)));
    }
    {
      return (x * (1.0 + (-1.0 * p1)) * m::log1prel((x * (-2.0 + p1))));
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
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (p1 * x);
      out[0] = (t0 * (1.0 / ipow<1>((1.0 + t1))));
      out[1] = ((-1.0 * x * m::log1prel(t1)) + (t0 * ipow<2>(x) * m::log1prel_d1(t1)));
      return;
    }
    {
      const double t0 = (1.0 + (-1.0 * p1));
      const double t1 = (-2.0 + p1);
      const double t2 = (t1 * x);
      out[0] = (t0 * (1.0 / ipow<1>((1.0 + (t1 * x)))));
      out[1] = ((-1.0 * x * m::log1prel(t2)) + (t0 * ipow<2>(x) * m::log1prel_d1(t2)));
      return;
    }
  }
  if constexpr (K == 2) {
    if (x >= 0.0) {
      const double t0 = (-1.0 + p1);
      const double t1 = (p1 * x);
      const double t2 = (1.0 + t1);
      const double t3 = (1.0 / ipow<2>(t2));
      out[0] = (t0 * t3 * p1);
      out[1] = ((-1.0 * (1.0 / ipow<1>(t2))) + (t0 * t3 * x));
      out[2] = (-1.0 * ipow<2>(x) * ((2.0 * m::log1prel_d1(t1)) + (t0 * x * m::log1prel_d2(t1))));
      return;
    }
    {
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (1.0 + (-1.0 * t1 * x));
      const double t3 = (-1.0 + p1);
      const double t4 = (-1.0 * t3 * (1.0 / ipow<2>(t2)));
      const double t5 = (t0 * x);
      out[0] = (t1 * t4);
      out[1] = ((-1.0 * (1.0 / ipow<1>(t2))) + (-1.0 * t4 * x));
      out[2] = (-1.0 * ipow<2>(x) * ((2.0 * m::log1prel_d1(t5)) + (t3 * x * m::log1prel_d2(t5))));
      return;
    }
  }
  if constexpr (K == 3) {
    if (x >= 0.0) {
      const double t0 = (p1 * x);
      const double t1 = (1.0 + t0);
      const double t2 = (1.0 / ipow<3>(t1));
      const double t3 = (1.0 / ipow<2>(t1));
      const double t4 = (-1.0 + p1);
      const double t5 = (t4 * x);
      out[0] = (t2 * ipow<2>(p1) * (2.0 + (-2.0 * p1)));
      out[1] = ((t3 * t4) + (t3 * p1) + (-2.0 * t0 * t2 * t4));
      out[2] = (2.0 * t3 * x * (1.0 + (-1.0 * t5 * (1.0 / ipow<1>(t1)))));
      out[3] = (-1.0 * ipow<3>(x) * ((3.0 * m::log1prel_d2(t0)) + (t5 * m::log1prel_d3(t0))));
      return;
    }
    {
      const double t0 = (-2.0 + p1);
      const double t1 = (-1.0 * t0);
      const double t2 = (t1 * x);
      const double t3 = (1.0 + (-1.0 * t2));
      const double t4 = (1.0 / ipow<3>(t3));
      const double t5 = (-1.0 + p1);
      const double t6 = (-1.0 * t5);
      const double t7 = (1.0 / ipow<2>(t3));
      const double t8 = (t0 * x);
      const double t9 = (1.0 + t8);
      const double t10 = (t5 * x);
      out[0] = (t4 * ipow<2>(t1) * (2.0 + (-2.0 * p1)));
      out[1] = ((-1.0 * t1 * t7) + (-1.0 * t6 * t7) + (-2.0 * t2 * t4 * t6));
      out[2] = (2.0 * x * (1.0 / ipow<2>(t9)) * (1.0 + (-1.0 * t10 * (1.0 / ipow<1>(t9)))));
      out[3] = (-1.0 * ipow<3>(x) * ((3.0 * m::log1prel_d2(t8)) + (t10 * m::log1prel_d3(t8))));
      return;
    }
  }
  if constexpr (K == 4) {
    if (x >= 0.0) {
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
      out[4] = (-1.0 * ipow<4>(x) * ((4.0 * m::log1prel_d3(t0)) + (t9 * m::log1prel_d4(t0))));
      return;
    }
    {
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
      out[4] = (-1.0 * ipow<4>(x) * ((4.0 * m::log1prel_d3(t8)) + (t13 * m::log1prel_d4(t8))));
      return;
    }
  }
  if constexpr (K == 5) {
    if (x >= 0.0) {
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
      out[5] = (-1.0 * ipow<5>(x) * ((5.0 * m::log1prel_d4(t0)) + (t13 * m::log1prel_d5(t0))));
      return;
    }
    {
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
      out[5] = (-1.0 * ipow<5>(x) * ((5.0 * m::log1prel_d4(t7)) + (t15 * m::log1prel_d5(t7))));
      return;
    }
  }
  if constexpr (K == 6) {
    if (x >= 0.0) {
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
      out[6] = (-1.0 * ipow<6>(x) * ((6.0 * m::log1prel_d5(t0)) + (t16 * m::log1prel_d6(t0))));
      return;
    }
    {
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
      out[6] = (-1.0 * ipow<6>(x) * ((6.0 * m::log1prel_d5(t7)) + (t19 * m::log1prel_d6(t7))));
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
