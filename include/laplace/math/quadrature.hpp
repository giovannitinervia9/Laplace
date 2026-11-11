// Quadrature rules callable inside Kokkos kernels (laplace::math::quad): the
// integrand is a functor f(x) -> double, nothing is allocated, nothing
// recurses, every loop has a cap. Each adaptive rule returns the estimate,
// an error estimate, the number of evaluations and whether it met its
// tolerance.
//
//   tanh_sinh(f, a, b)         [a, b] finite; integrable singularities at the ends
//                              (f may take (x, xc), xc the signed distance to the nearer end)
//   exp_sinh(f, a, s)          [a, inf), s the scale of the integrand's decay
//   sinh_sinh(f, c, s)         (-inf, inf), centre c and scale s
//   gauss_kronrod(f, a, b)     [a, b], GK 7-15, adaptive bisection (at most 64 intervals)
//   panels(f, a, b, c, s, kinks, nk)
//                              [a, b] (either end may be infinite) cut at the centre c and
//                              at the kinks, finite pieces in panels of widths s, 2s, 4s,
//                              ... from the side nearer c (tanh-sinh on each), unbounded
//                              ones beyond 63 s by exp-sinh: the rule of statmodels7's
//                              pt_trunc_rule.h, with a convergence check on every panel
//   gauss_legendre<n>(f, a, b) n = 16, 32, 64, fixed
//   gauss_hermite32(f, c, s)   int f over R as sum w_i e^(y_i^2) s f(c + s y_i), fixed
//   gauss_laguerre32(f, a, s)  int f over [a, inf) as sum w_i e^(y_i) s f(a + s y_i), fixed
//
// The double-exponential rules (Takahasi and Mori, 1974) map the interval to
// t in [-6.8, 6.8] (beyond, the nodes leave the double range) and halve the
// step from 1/2 until two successive estimates differ by less than
// `stop` times the integral of |f| (the error of such a rule is roughly the
// square of that difference once it converges); the range of t is first
// trimmed to where |f w| exceeds 2^-64 of the integral of |f|, at step 1/2.
// tanh-sinh computes each node's distance to its nearer end directly, so
// that integrable singularities at the ends are sampled to the last bit.
//
// The Gauss-Kronrod constants are those of Boost.Math (gauss_kronrod.hpp,
// Boost Software License 1.0); the Gauss tables come from the Golub-Welsch
// algorithm at 256 bits (tools/reference/quadrature.py nodes).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace laplace::math::quad {

struct result {
  double value = 0.0, error = 0.0;
  int evals = 0;
  bool converged = false;
};

struct de_options {
  double stop = 1e-9;  // successive estimates within stop * int |f|
  int max_level = 8;   // step 2^-(max_level + 1) at most
};

namespace detail {

constexpr double half_pi = 1.57079632679489661923;
constexpr double t_max = 6.8;

// f(x), or f(x, xc) for an integrand that takes the distance to the end
// (Boost's convention: xc = a - x on the left half of tanh-sinh, b - x on the
// right half; x - a for exp-sinh): an integrand singular at an end can then
// be evaluated from that distance, which x itself loses near a nonzero end
template <class F>
LAPLACE_HD_FORCEINLINE double call(const F& f, double x, double xc) {
  if constexpr (std::is_invocable_v<const F&, double, double>) {
    return f(x, xc);
  } else {
    (void)xc;
    return f(x);
  }
}

// the double-exponential sum over t at step h from the nodes not yet
// summed (all of them at level 0, the odd multiples of h above it), within
// [t_lo, t_hi]; node(t, x, w, xc) returns 0 for a node to skip, 1 for a node
// inside the interval and 2 for one whose x has rounded to an end although its
// distance xc is not zero (evaluated all the same: an integrand that takes xc
// gets the exact distance, one that does not gets the end, and a non-finite
// value there, an integrable singularity sampled at its end, is skipped)
template <class F, class Node>
LAPLACE_HD_FORCEINLINE void de_sum(const F& f, const Node& node, double h, bool odd, double t_lo, double t_hi,
                                   double& s, double& l1, int& evals) {
  const int k0 = int(Kokkos::ceil(t_lo / h)), k1 = int(Kokkos::floor(t_hi / h));
  for (int k = k0; k <= k1; ++k) {
    if (odd && (k % 2 == 0)) continue;
    double x, w, xc;
    if (node(k * h, x, w, xc) == 0) continue;
    const double v = call(f, x, xc);
    ++evals;
    if (!(Kokkos::fabs(v) < 1e308)) continue;  // NaN or infinite: an integrable singularity sampled at its end
    const double fw = v * w;
    s += fw;
    l1 += Kokkos::fabs(fw);
  }
}

template <class F, class Node>
LAPLACE_HD result de_rule(const F& f, const Node& node, const de_options& o) {
  result r;
  // level 0 at h = 1/2, and the range of t where |f w| matters
  double h = 0.5, s = 0.0, l1 = 0.0;
  double t_lo = -t_max, t_hi = t_max;
  {
    const int K = int(t_max / h);
    double fw[2 * 13 + 1];
    for (int k = -K; k <= K; ++k) {
      double x, w, xc, v = 0.0;
      if (node(k * h, x, w, xc) != 0) {
        v = call(f, x, xc);
        ++r.evals;
        if (!(Kokkos::fabs(v) < 1e308)) v = 0.0;
        v *= w;
      }
      fw[k + K] = v;
      s += v;
      l1 += Kokkos::fabs(v);
    }
    const double floor = l1 * 0x1p-64;
    int lo = -K, hi = K;
    while (lo < 0 && Kokkos::fabs(fw[lo + K]) <= floor) ++lo;
    while (hi > 0 && Kokkos::fabs(fw[hi + K]) <= floor) --hi;
    // keep one step beyond the last node that counts
    t_lo = (lo - 1) * h, t_hi = (hi + 1) * h;
    if (t_lo < -t_max) t_lo = -t_max;
    if (t_hi > t_max) t_hi = t_max;
  }
  double prev = s * h;
  for (int level = 1; level <= o.max_level; ++level) {
    h *= 0.5;
    de_sum(f, node, h, true, t_lo, t_hi, s, l1, r.evals);
    const double est = s * h;
    const double diff = Kokkos::fabs(est - prev);
    r.value = est;
    r.error = diff;
    prev = est;
    if (diff <= o.stop * l1 * h) {
      r.converged = true;
      // the error of the finer estimate is about the square of the relative difference
      const double rel = diff / (l1 * h);
      r.error = rel * rel * l1 * h;
      return r;
    }
  }
  return r;
}

// the nodes of the three rules
struct tanh_sinh_node {
  double a, b;
  LAPLACE_HD_FORCEINLINE int operator()(double t, double& x, double& w, double& xc) const {
    const double q = half_pi * Kokkos::sinh(t);
    const double e = laplace::math::exp(-2.0 * Kokkos::fabs(q));
    const double L = b - a;
    const double d = L * e / (1.0 + e);  // distance to the nearer end
    if (!(d > 0.0)) return 0;
    x = q < 0.0 ? a + d : b - d;
    xc = q < 0.0 ? -d : d;
    w = half_pi * Kokkos::cosh(t) * L * 2.0 * e / ((1.0 + e) * (1.0 + e));
    return (x > a && x < b) ? 1 : 2;
  }
};
struct exp_sinh_node {
  double a, s;
  LAPLACE_HD_FORCEINLINE int operator()(double t, double& x, double& w, double& xc) const {
    const double u = laplace::math::exp(half_pi * Kokkos::sinh(t));
    const double d = s * u;
    if (!(d > 0.0) || !(d < 1e300)) return 0;
    x = a + d;
    xc = d;
    w = s * half_pi * Kokkos::cosh(t) * u;
    return x > a ? 1 : 2;
  }
};
struct sinh_sinh_node {
  double c, s;
  LAPLACE_HD_FORCEINLINE int operator()(double t, double& x, double& w, double& xc) const {
    const double q = half_pi * Kokkos::sinh(t);
    const double sh = Kokkos::sinh(q);
    xc = 0.0;
    if (!(Kokkos::fabs(sh) < 1e300)) return 0;
    x = c + s * sh;
    w = s * half_pi * Kokkos::cosh(t) * Kokkos::cosh(q);
    return 1;
  }
};

// clang-format off
#define LAPLACE_QUAD_GK15_X \
    0.00000000000000000000000000000000000e+00, 2.07784955007898467600689403773244913e-01, \
    4.05845151377397166906606412076961463e-01, 5.86087235467691130294144838258729598e-01, \
    7.41531185599394439863864773280788407e-01, 8.64864423359769072789712788640926201e-01, \
    9.49107912342758524526189684047851262e-01, 9.91455371120812639206854697526328517e-01
#define LAPLACE_QUAD_GK15_W \
    2.09482141084727828012999174891714264e-01, 2.04432940075298892414161999234649085e-01, \
    1.90350578064785409913256402421013683e-01, 1.69004726639267902826583426598550284e-01, \
    1.40653259715525918745189590510237920e-01, 1.04790010322250183839876322541518017e-01, \
    6.30920926299785532907006631892042867e-02, 2.29353220105292249637320080589695920e-02
#define LAPLACE_QUAD_G7_W \
    4.17959183673469387755102040816326531e-01, 3.81830050505118944950369775488975134e-01, \
    2.79705391489276667901467771423779582e-01, 1.29484966168869693270611432679082018e-01
#define LAPLACE_QUAD_LEGENDRE16 \
    -0x1.fa92c264d787ep-1, -0x1.e39f56616f9b0p-1, -0x1.bb3403514e483p-1, -0x1.82c45dda4726bp-1, \
    -0x1.3c5a466d5e8b8p-1, -0x1.d50259a43a772p-2, -0x1.205cae642337cp-2, -0x1.852bd6676a9f9p-4, \
    0x1.852bd6676a9f9p-4, 0x1.205cae642337cp-2, 0x1.d50259a43a772p-2, 0x1.3c5a466d5e8b8p-1, \
    0x1.82c45dda4726bp-1, 0x1.bb3403514e483p-1, 0x1.e39f56616f9b0p-1, 0x1.fa92c264d787ep-1, \
    0x1.bcddab4b7c211p-6, 0x1.fdfb1a2c1265dp-5, 0x1.85c4ee79cc258p-4, 0x1.fe7af2bad386ap-4, \
    0x1.325f61bca3cbfp-3, 0x1.5a6ebbb5a75fcp-3, 0x1.75f8c77e0c00fp-3, 0x1.83feae80e4dfcp-3, \
    0x1.83feae80e4dfcp-3, 0x1.75f8c77e0c00fp-3, 0x1.5a6ebbb5a75fcp-3, 0x1.325f61bca3cbfp-3, \
    0x1.fe7af2bad386ap-4, 0x1.85c4ee79cc258p-4, 0x1.fdfb1a2c1265dp-5, 0x1.bcddab4b7c211p-6

#define LAPLACE_QUAD_LEGENDRE32 \
    -0x1.fe995e70409b6p-1, -0x1.f8a212714bcdcp-1, -0x1.edf5518053baap-1, -0x1.deac0259f7f42p-1, \
    -0x1.caea9b4574cb9p-1, -0x1.b2e04fd686a13p-1, -0x1.96c69481c4bc5p-1, -0x1.76e0931d693bap-1, \
    -0x1.537a89c487f8ap-1, -0x1.2ce9146962ca4p-1, -0x1.038862866b29dp-1, -0x1.af76b57c6f8f1p-2, \
    -0x1.53d55ce57bdf6p-2, -0x1.ea0f7e19c094bp-3, -0x1.27e0ea717f237p-3, -0x1.8bbc8488cc499p-5, \
    0x1.8bbc8488cc499p-5, 0x1.27e0ea717f237p-3, 0x1.ea0f7e19c094bp-3, 0x1.53d55ce57bdf6p-2, \
    0x1.af76b57c6f8f1p-2, 0x1.038862866b29dp-1, 0x1.2ce9146962ca4p-1, 0x1.537a89c487f8ap-1, \
    0x1.76e0931d693bap-1, 0x1.96c69481c4bc5p-1, 0x1.b2e04fd686a13p-1, 0x1.caea9b4574cb9p-1, \
    0x1.deac0259f7f42p-1, 0x1.edf5518053baap-1, 0x1.f8a212714bcdcp-1, 0x1.fe995e70409b6p-1, \
    0x1.cbf8bc743cc5cp-8, 0x1.0aa3c248696c9p-6, 0x1.a0060a8531ffap-6, 0x1.18c5800a355d9p-5, \
    0x1.5ee963a335495p-5, 0x1.a1c6ae961fbfap-5, 0x1.e0bd76c924981p-5, 0x1.0d9b9a62cac10p-4, \
    0x1.2854103b35e0cp-4, 0x1.40483e126fd14p-4, 0x1.553ee25ebebc6p-4, 0x1.6705e18e13ed1p-4, \
    0x1.7572bdb3f6e51p-4, 0x1.8062fc0f6fef9p-4, 0x1.87bc776f8c6d7p-4, 0x1.8b6d9eaec77adp-4, \
    0x1.8b6d9eaec77adp-4, 0x1.87bc776f8c6d7p-4, 0x1.8062fc0f6fef9p-4, 0x1.7572bdb3f6e51p-4, \
    0x1.6705e18e13ed1p-4, 0x1.553ee25ebebc6p-4, 0x1.40483e126fd14p-4, 0x1.2854103b35e0cp-4, \
    0x1.0d9b9a62cac10p-4, 0x1.e0bd76c924981p-5, 0x1.a1c6ae961fbfap-5, 0x1.5ee963a335495p-5, \
    0x1.18c5800a355d9p-5, 0x1.a0060a8531ffap-6, 0x1.0aa3c248696c9p-6, 0x1.cbf8bc743cc5cp-8

#define LAPLACE_QUAD_LEGENDRE64 \
    -0x1.ffa4e911f7533p-1, -0x1.fe204ab274ecdp-1, -0x1.fb661ac8c85a9p-1, -0x1.f777d976cfadap-1, \
    -0x1.f257e4db5aabcp-1, -0x1.ec09586b58faap-1, -0x1.e490081f2891bp-1, -0x1.dbf07d935a5afp-1, \
    -0x1.d22ff5221288ap-1, -0x1.c7545aa8c0dadp-1, -0x1.bb6445eadae2cp-1, -0x1.ae66f68eedbc6p-1, \
    -0x1.a0644fb6d8db8p-1, -0x1.9164d335425e2p-1, -0x1.81719c62ec68ep-1, -0x1.70945a96f12c4p-1, \
    -0x1.5ed74b4532f83p-1, -0x1.4c4533c68b412p-1, -0x1.38e95ace7b3c3p-1, -0x1.24cf81925487fp-1, \
    -0x1.1003dca600f34p-1, -0x1.f52619257c3a1p-2, -0x1.c9142c5898fc5p-2, -0x1.9becb55272c9dp-2, \
    -0x1.6dcb1f0620fffp-2, -0x1.3ecb6c46c76cbp-2, -0x1.0f0a26c56e49cp-2, -0x1.bd489b79ec83bp-3, \
    -0x1.5b6e88ad5c00fp-3, -0x1.f182ff48e8a27p-4, -0x1.2afad5ee95ad0p-4, -0x1.8ef487a8cbc33p-6, \
    0x1.8ef487a8cbc33p-6, 0x1.2afad5ee95ad0p-4, 0x1.f182ff48e8a27p-4, 0x1.5b6e88ad5c00fp-3, \
    0x1.bd489b79ec83bp-3, 0x1.0f0a26c56e49cp-2, 0x1.3ecb6c46c76cbp-2, 0x1.6dcb1f0620fffp-2, \
    0x1.9becb55272c9dp-2, 0x1.c9142c5898fc5p-2, 0x1.f52619257c3a1p-2, 0x1.1003dca600f34p-1, \
    0x1.24cf81925487fp-1, 0x1.38e95ace7b3c3p-1, 0x1.4c4533c68b412p-1, 0x1.5ed74b4532f83p-1, \
    0x1.70945a96f12c4p-1, 0x1.81719c62ec68ep-1, 0x1.9164d335425e2p-1, 0x1.a0644fb6d8db8p-1, \
    0x1.ae66f68eedbc6p-1, 0x1.bb6445eadae2cp-1, 0x1.c7545aa8c0dadp-1, 0x1.d22ff5221288ap-1, \
    0x1.dbf07d935a5afp-1, 0x1.e490081f2891bp-1, 0x1.ec09586b58faap-1, 0x1.f257e4db5aabcp-1, \
    0x1.f777d976cfadap-1, 0x1.fb661ac8c85a9p-1, 0x1.fe204ab274ecdp-1, 0x1.ffa4e911f7533p-1, \
    0x1.d379f1846042ep-10, 0x1.0fc7ac3ac322fp-8, 0x1.aa46b24145a02p-8, 0x1.21e400109d479p-7, \
    0x1.6df524de84e24p-7, 0x1.b9283b35dfa9cp-7, 0x1.01a7c0a5c987fp-6, 0x1.261ef40a7a2e7p-6, \
    0x1.49e391bd2145fp-6, 0x1.6cdfe10bba3c0p-6, 0x1.8efea346845b4p-6, 0x1.b02b2071c0c2dp-6, \
    0x1.d05133c3af937p-6, 0x1.ef5d57d53b4a2p-6, 0x1.069e593b92370p-5, 0x1.14ee9010d92c8p-5, \
    0x1.22969f7b5c8c9p-5, 0x1.2f8e3ca7574ddp-5, 0x1.3bcd87e50de1cp-5, 0x1.474d117092813p-5, \
    0x1.5205ddf5a36dbp-5, 0x1.5bf16accdf42ep-5, 0x1.6509b1efb8deep-5, 0x1.6d492da0c2510p-5, \
    0x1.74aadbc614fb3p-5, 0x1.7b2a40f3ccddap-5, 0x1.80c36b24bdd19p-5, 0x1.8572f41fbb52dp-5, \
    0x1.89360387fe3a9p-5, 0x1.8c0a5097676bap-5, 0x1.8dee238192cd0p-5, 0x1.8ee0567ee2e50p-5, \
    0x1.8ee0567ee2e50p-5, 0x1.8dee238192cd0p-5, 0x1.8c0a5097676bap-5, 0x1.89360387fe3a9p-5, \
    0x1.8572f41fbb52dp-5, 0x1.80c36b24bdd19p-5, 0x1.7b2a40f3ccddap-5, 0x1.74aadbc614fb3p-5, \
    0x1.6d492da0c2510p-5, 0x1.6509b1efb8deep-5, 0x1.5bf16accdf42ep-5, 0x1.5205ddf5a36dbp-5, \
    0x1.474d117092813p-5, 0x1.3bcd87e50de1cp-5, 0x1.2f8e3ca7574ddp-5, 0x1.22969f7b5c8c9p-5, \
    0x1.14ee9010d92c8p-5, 0x1.069e593b92370p-5, 0x1.ef5d57d53b4a2p-6, 0x1.d05133c3af937p-6, \
    0x1.b02b2071c0c2dp-6, 0x1.8efea346845b4p-6, 0x1.6cdfe10bba3c0p-6, 0x1.49e391bd2145fp-6, \
    0x1.261ef40a7a2e7p-6, 0x1.01a7c0a5c987fp-6, 0x1.b9283b35dfa9cp-7, 0x1.6df524de84e24p-7, \
    0x1.21e400109d479p-7, 0x1.aa46b24145a02p-8, 0x1.0fc7ac3ac322fp-8, 0x1.d379f1846042ep-10

#define LAPLACE_QUAD_HERMITE32 \
    -0x1.c80d55c906b62p+2, -0x1.9a3537b9b8afap+2, -0x1.73fb828c88899p+2, -0x1.51a2a09addd6fp+2, \
    -0x1.31bd102f89f77p+2, -0x1.138e1900c0335p+2, -0x1.ed47dc1870b2ep+1, -0x1.b565be914e170p+1, \
    -0x1.7f09f0797b8d8p+1, -0x1.49e3501718808p+1, -0x1.15b2263524badp+1, -0x1.c484facec8d21p+0, \
    -0x1.5ed0fd0c409ffp+0, -0x1.f3f7de674b058p-1, -0x1.2b825634cdb6cp-1, -0x1.8f08a9a7c06cfp-3, \
    0x1.8f08a9a7c06cfp-3, 0x1.2b825634cdb6cp-1, 0x1.f3f7de674b058p-1, 0x1.5ed0fd0c409ffp+0, \
    0x1.c484facec8d21p+0, 0x1.15b2263524badp+1, 0x1.49e3501718808p+1, 0x1.7f09f0797b8d8p+1, \
    0x1.b565be914e170p+1, 0x1.ed47dc1870b2ep+1, 0x1.138e1900c0335p+2, 0x1.31bd102f89f77p+2, \
    0x1.51a2a09addd6fp+2, 0x1.73fb828c88899p+2, 0x1.9a3537b9b8afap+2, 0x1.c80d55c906b62p+2, \
    0x1.6185ca6732db3p-74, 0x1.1079077446a10p-60, 0x1.591c6504e6094p-50, 0x1.da9165dac4be2p-42, \
    0x1.04f2ec4d99a29p-34, 0x1.19ab6b0289e1dp-28, 0x1.520cca23243e8p-23, 0x1.e9f926629737dp-19, \
    0x1.c66041cb42acep-15, 0x1.1928b8bc0d2e1p-11, 0x1.df0dc4d8ce1d3p-9, 0x1.1f986ab0e9d28p-6, \
    0x1.ef45e3e774296p-5, 0x1.35cce805ded09p-3, 0x1.1c1dfcbccb09fp-2, 0x1.803e7b925d2ffp-2, \
    0x1.803e7b925d2ffp-2, 0x1.1c1dfcbccb09fp-2, 0x1.35cce805ded09p-3, 0x1.ef45e3e774296p-5, \
    0x1.1f986ab0e9d28p-6, 0x1.df0dc4d8ce1d3p-9, 0x1.1928b8bc0d2e1p-11, 0x1.c66041cb42acep-15, \
    0x1.e9f926629737dp-19, 0x1.520cca23243e8p-23, 0x1.19ab6b0289e1dp-28, 0x1.04f2ec4d99a29p-34, \
    0x1.da9165dac4be2p-42, 0x1.591c6504e6094p-50, 0x1.1079077446a10p-60, 0x1.6185ca6732db3p-74

#define LAPLACE_QUAD_LAGUERRE32 \
    0x1.6c74f668c1d61p-5, 0x1.e04f39938f1b1p-3, 0x1.275d6c10d557dp-1, 0x1.128c006448924p+0, \
    0x1.b8efc8152c521p+0, 0x1.43a08985b004ap+1, 0x1.bf00d83324aedp+1, 0x1.277407185e468p+2, \
    0x1.79da74c517a53p+2, 0x1.d6eb8c63b6ed2p+2, 0x1.1f7440866c946p+3, 0x1.590e7d175d5edp+3, \
    0x1.987036c2a297fp+3, 0x1.ddcbe599bf31bp+3, 0x1.14ade4993c402p+4, 0x1.3db19b3dcaacbp+4, \
    0x1.6a181f13f15fbp+4, 0x1.9a0ee4a558cccp+4, 0x1.cdcb2b462ee78p+4, 0x1.02c5e58180f31p+5, \
    0x1.20cdd03880fb1p+5, 0x1.412a6f20a923dp+5, 0x1.6412dba445088p+5, 0x1.89cb8f995f040p+5, \
    0x1.b2ab761727568p+5, 0x1.df23dbd7f5f7cp+5, 0x1.07e6c94dc3c4dp+6, 0x1.22c0219404d44p+6, \
    0x1.40bff219bf7d9p+6, 0x1.62f0fd1419197p+6, 0x1.8b5173af87a38p+6, 0x1.bf016e80c3ad1p+6, \
    0x1.bf5bbb6cf1006p-4, 0x1.aefccbd2205c3p-3, 0x1.e1b7794825f53p-3, 0x1.9135c4a972541p-3, \
    0x1.0a34f078df178p-3, 0x1.21170d14cc8f7p-4, 0x1.042f761093a79p-5, 0x1.86893b789e639p-7, \
    0x1.ea0ddb6c16667p-9, 0x1.011c9880bcb73p-10, 0x1.c29ab993f1a1ap-13, 0x1.48dcb3e4afa28p-15, \
    0x1.8e42a609151cbp-18, 0x1.8e2a49d36219cp-21, 0x1.469d1baaaceedp-24, 0x1.b468f9d247f0cp-28, \
    0x1.d6be379ad5412p-32, 0x1.95a87b1ddbce8p-36, 0x1.13d3f82e99e70p-40, 0x1.23a29f8886630p-45, \
    0x1.d712d41b44499p-51, 0x1.1c793615bcb68p-56, 0x1.f48594e314114p-63, 0x1.3674e602354ccp-69, \
    0x1.048b58d82fc11p-76, 0x1.18556dd48f84bp-84, 0x1.67a5d7e2629f8p-93, 0x1.f0bd9d1a62cf6p-103, \
    0x1.3cf43efa71623p-113, 0x1.22e6dd1e558c4p-125, 0x1.dda26da4715e9p-140, 0x1.a5e5e26a8fda1p-158
// clang-format on

#if defined(__CUDACC__)
static __constant__ double gk15_x_dev[8] = {LAPLACE_QUAD_GK15_X};
static __constant__ double gk15_w_dev[8] = {LAPLACE_QUAD_GK15_W};
static __constant__ double g7_w_dev[4] = {LAPLACE_QUAD_G7_W};
static __constant__ double legendre16_dev[32] = {LAPLACE_QUAD_LEGENDRE16};
static __constant__ double legendre32_dev[64] = {LAPLACE_QUAD_LEGENDRE32};
static __constant__ double legendre64_dev[128] = {LAPLACE_QUAD_LEGENDRE64};
static __constant__ double hermite32_dev[64] = {LAPLACE_QUAD_HERMITE32};
static __constant__ double laguerre32_dev[64] = {LAPLACE_QUAD_LAGUERRE32};
#endif
inline constexpr double gk15_x_host[8] = {LAPLACE_QUAD_GK15_X};
inline constexpr double gk15_w_host[8] = {LAPLACE_QUAD_GK15_W};
inline constexpr double g7_w_host[4] = {LAPLACE_QUAD_G7_W};
inline constexpr double legendre16_host[32] = {LAPLACE_QUAD_LEGENDRE16};
inline constexpr double legendre32_host[64] = {LAPLACE_QUAD_LEGENDRE32};
inline constexpr double legendre64_host[128] = {LAPLACE_QUAD_LEGENDRE64};
inline constexpr double hermite32_host[64] = {LAPLACE_QUAD_HERMITE32};
inline constexpr double laguerre32_host[64] = {LAPLACE_QUAD_LAGUERRE32};

#if defined(__CUDACC__)
#define LAPLACE_QUAD_TABLE(name, i) \
  KOKKOS_IF_ON_DEVICE((return name##_dev[i];)) KOKKOS_IF_ON_HOST((return name##_host[i];))
#else
#define LAPLACE_QUAD_TABLE(name, i) return name##_host[i];
#endif
LAPLACE_HD_FORCEINLINE double gk15_x(int i) { LAPLACE_QUAD_TABLE(gk15_x, i) }
LAPLACE_HD_FORCEINLINE double gk15_w(int i) { LAPLACE_QUAD_TABLE(gk15_w, i) }
LAPLACE_HD_FORCEINLINE double g7_w(int i) { LAPLACE_QUAD_TABLE(g7_w, i) }
// node i, then weight i (offset n), of each fixed rule
LAPLACE_HD_FORCEINLINE double legendre16(int i) { LAPLACE_QUAD_TABLE(legendre16, i) }
LAPLACE_HD_FORCEINLINE double legendre32(int i) { LAPLACE_QUAD_TABLE(legendre32, i) }
LAPLACE_HD_FORCEINLINE double legendre64(int i) { LAPLACE_QUAD_TABLE(legendre64, i) }
LAPLACE_HD_FORCEINLINE double hermite32(int i) { LAPLACE_QUAD_TABLE(hermite32, i) }
LAPLACE_HD_FORCEINLINE double laguerre32(int i) { LAPLACE_QUAD_TABLE(laguerre32, i) }
#undef LAPLACE_QUAD_TABLE

// GK15 on [a, b]: the Kronrod estimate and |K - G|
template <class F>
LAPLACE_HD_FORCEINLINE void gk15(const F& f, double a, double b, double& K, double& err, int& evals) {
  const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
  const double fc = f(c);
  double k = fc * gk15_w(0), g = fc * g7_w(0);
  for (int j = 1; j < 8; ++j) {
    const double dx = hl * gk15_x(j);
    const double s = f(c - dx) + f(c + dx);
    k += s * gk15_w(j);
    if (j % 2 == 0) g += s * g7_w(j / 2);
  }
  evals += 15;
  K = k * hl;
  err = Kokkos::fabs((k - g) * hl);
}

}  // namespace detail

template <class F>
LAPLACE_HD result tanh_sinh(const F& f, double a, double b, const de_options& o = de_options{}) {
  return detail::de_rule(f, detail::tanh_sinh_node{a, b}, o);
}
template <class F>
LAPLACE_HD result exp_sinh(const F& f, double a, double s = 1.0, const de_options& o = de_options{}) {
  return detail::de_rule(f, detail::exp_sinh_node{a, s}, o);
}
template <class F>
LAPLACE_HD result sinh_sinh(const F& f, double c = 0.0, double s = 1.0, const de_options& o = de_options{}) {
  return detail::de_rule(f, detail::sinh_sinh_node{c, s}, o);
}

// adaptive Gauss-Kronrod 7-15: bisect the interval with the largest error
// until the sum of the errors is below tol times |value|, at most 64 intervals
template <class F>
LAPLACE_HD result gauss_kronrod(const F& f, double a, double b, double tol = 1e-14, int max_intervals = 64) {
  constexpr int cap = 64;
  if (max_intervals > cap) max_intervals = cap;
  double lo[cap], hi[cap], val[cap], err[cap];
  result r;
  int n = 1;
  lo[0] = a, hi[0] = b;
  detail::gk15(f, a, b, val[0], err[0], r.evals);
  for (;;) {
    double total = 0.0, etot = 0.0;
    int worst = 0;
    for (int i = 0; i < n; ++i) {
      total += val[i], etot += err[i];
      if (err[i] > err[worst]) worst = i;
    }
    r.value = total, r.error = etot;
    if (etot <= tol * Kokkos::fabs(total)) {
      r.converged = true;
      return r;
    }
    if (n >= max_intervals) return r;
    const double m = 0.5 * (lo[worst] + hi[worst]);
    if (!(m > lo[worst] && m < hi[worst])) return r;
    lo[n] = m, hi[n] = hi[worst];
    hi[worst] = m;
    detail::gk15(f, lo[worst], hi[worst], val[worst], err[worst], r.evals);
    detail::gk15(f, lo[n], hi[n], val[n], err[n], r.evals);
    ++n;
  }
}

// fixed Gauss rules
template <int N, class F>
LAPLACE_HD double gauss_legendre(const F& f, double a, double b) {
  static_assert(N == 16 || N == 32 || N == 64, "Gauss-Legendre with 16, 32 or 64 nodes");
  const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
  double s = 0.0;
  for (int i = 0; i < N; ++i) {
    const double x = N == 16 ? detail::legendre16(i) : N == 32 ? detail::legendre32(i) : detail::legendre64(i);
    const double w = N == 16 ? detail::legendre16(N + i) : N == 32 ? detail::legendre32(N + i) : detail::legendre64(N + i);
    s += w * f(c + hl * x);
  }
  return s * hl;
}
template <class F>
LAPLACE_HD double gauss_hermite32(const F& f, double c = 0.0, double s = 1.0) {
  double acc = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double y = detail::hermite32(i), w = detail::hermite32(32 + i);
    acc += w * laplace::math::exp(y * y) * f(c + s * y);
  }
  return acc * s;
}
template <class F>
LAPLACE_HD double gauss_laguerre32(const F& f, double a = 0.0, double s = 1.0) {
  double acc = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double y = detail::laguerre32(i), w = detail::laguerre32(32 + i);
    acc += w * laplace::math::exp(y) * f(a + s * y);
  }
  return acc * s;
}

// the panel rule: [a, b] cut at c and at the kinks (at most 8), finite pieces
// in doubling panels from the side nearer c, tanh-sinh on each; a piece
// unbounded on one side gets doubling panels over its first 63 s, then exp-sinh
template <class F>
LAPLACE_HD result panels(const F& f, double a, double b, double c, double s, const double* kinks = nullptr, int nk = 0,
                         const de_options& o = de_options{}) {
  result r;
  r.converged = true;
  if (!(b > a)) return r;
  double cut[10];
  int nc = 0;
  if (c > a && c < b) cut[nc++] = c;
  for (int i = 0; i < nk && i < 8; ++i)
    if (kinks[i] > a && kinks[i] < b) cut[nc++] = kinks[i];
  for (int i = 1; i < nc; ++i)  // sort the cuts
    for (int j = i; j > 0 && cut[j] < cut[j - 1]; --j) {
      const double t = cut[j];
      cut[j] = cut[j - 1], cut[j - 1] = t;
    }
  auto add = [&](const result& p) {
    r.value += p.value, r.error += p.error, r.evals += p.evals;
    r.converged = r.converged && p.converged;
  };
  // a finite piece [u, v] in doubling panels from the end nearer c
  auto finite = [&](double u, double v) {
    const bool from_u = Kokkos::fabs(u - c) <= Kokkos::fabs(v - c);
    double width = s, at = from_u ? u : v;
    for (int k = 0; k < 64; ++k) {
      const double next = from_u ? at + width : at - width;
      const bool last = !(width > 0.0) || next == at || (from_u ? !(next < v) : !(next > u));
      if (last) {
        add(from_u ? tanh_sinh(f, at, v, o) : tanh_sinh(f, u, at, o));
        return;
      }
      add(from_u ? tanh_sinh(f, at, next, o) : tanh_sinh(f, next, at, o));
      at = next;
      width *= 2.0;
    }
  };
  const bool inf_lo = !(a > -1e308), inf_hi = !(b < 1e308);
  double ends[12];
  int ne = 0;
  ends[ne++] = a;
  for (int i = 0; i < nc; ++i) ends[ne++] = cut[i];
  ends[ne++] = b;
  for (int i = 0; i + 1 < ne; ++i) {
    const double u = ends[i], v = ends[i + 1];
    if (i == 0 && inf_lo) {
      if (inf_hi && ne == 2) {  // the whole line without cuts: split at c
        const double m = c;
        finite(m, m + 63.0 * s);
        add(exp_sinh(f, m + 63.0 * s, s, o));
        finite(m - 63.0 * s, m);
        add(exp_sinh([&](double x) { return f(-x); }, -(m - 63.0 * s), s, o));
        return r;
      }
      finite(v - 63.0 * s, v);
      add(exp_sinh([&](double x) { return f(-x); }, -(v - 63.0 * s), s, o));
    } else if (i + 2 == ne && inf_hi) {
      finite(u, u + 63.0 * s);
      add(exp_sinh(f, u + 63.0 * s, s, o));
    } else {
      finite(u, v);
    }
  }
  return r;
}

}  // namespace laplace::math::quad
