// Quadrature rules callable inside Kokkos kernels (laplace::math::quad): the
// integrand is a functor f(x) -> double, nothing is allocated, nothing
// recurses, every loop has a cap. Each adaptive rule returns the estimate,
// an error estimate, the number of evaluations and whether it met its
// tolerance.
//
//   tanh_sinh_quad(f, a, b)         [a, b] finite; integrable singularities at the ends
//                              (f may take (x, xc), xc the signed distance to the nearer end)
//   exp_sinh_quad(f, a, s)          [a, inf), s the scale of the integrand's decay
//   sinh_sinh_quad(f, c, s)         (-inf, inf), centre c and scale s
//   gauss_kronrod_quad(f, a, b)     [a, b], GK 7-15, adaptive bisection (at most 64 intervals)
//   panels_quad(f, a, b, c, s, kinks, nk)
//                              [a, b] (either end may be infinite) cut at the centre c and
//                              at the kinks, finite pieces in panels of widths s, 2s, 4s,
//                              ... from the side nearer c (tanh-sinh on each), unbounded
//                              ones beyond 63 s by exp-sinh: the rule of statmodels7's
//                              pt_trunc_rule.h, with a convergence check on every panel
//   gauss_legendre_quad<n>(f, a, b) n = 16, 32, 64, fixed
//   gauss_hermite32_quad(f, c, s)   int f over R as sum w_i e^(y_i^2) s f(c + s y_i), fixed
//   gauss_laguerre32_quad(f, a, s)  int f over [a, inf) as sum w_i e^(y_i) s f(a + s y_i), fixed
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
// The Gauss-Kronrod constants are those of Boost.Math (gauss_kronrod_quad.hpp,
// Boost Software License 1.0); the Gauss tables come from the Golub-Welsch
// algorithm at 256 bits (tools/reference/quadrature.py nodes).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace laplace::math::quad {

/**
 * \ingroup quadrature
 * \brief Result of an adaptive quadrature rule.
 *
 * Encapsulates the computed integral estimate, an error estimate, the total
 * number of integrand evaluations, and a boolean flag indicating whether the
 * convergence criterion was satisfied. When the rule does not converge,
 * `value` holds the last estimate and `error` holds the last difference between
 * successive estimates.
 *
 * - `value`:
 * Estimated value of the definite integral \f$I = \int_a^b f(x) \, \mathrm{d}x\f$; holds the last estimate when the rule does not converge.
 *
 * - `error`:
 * Estimated absolute error of the integral estimate; holds the last difference between successive estimates when the rule does not converge.
 *
 * - `evals`:
 * Total number of integrand evaluations performed.
 *
 * - `converged`:
 * Flag indicating whether the convergence tolerance was met within the evaluation limit.
 *
 * \see quadrature, de_quad_options
 */
struct quad_result {
  double value = 0.0, error = 0.0;
  int evals = 0;
  bool converged = false;
};

/**
 * \ingroup quadrature
 * \brief Options for double-exponential and panel quadrature rules.
 *
 * Specifies the relative stopping tolerance between successive trapezoidal
 * refinement levels and the maximum refinement level for step halving.
 *
 * - `stop`:
 * Successive estimates tolerance factor: iteration terminates when two
 * successive trapezoidal estimates differ by less than \f$\mathrm{stop} \times \int |f|\f$.
 * Default is \f$10^{-9}\f$.
 *
 * - `max_level`:
 * Maximum refinement level for step halving (default 8). At level \f$L\f$,
 * the step size is \f$h = 2^{-(L+1)}\f$, down to \f$h = 2^{-9} = 1/512\f$ at level 8.
 *
 * \see quadrature, quad_result, tanh_sinh_quad(), exp_sinh_quad(), sinh_sinh_quad(), panels_quad()
 */
struct de_quad_options {
  double stop = 1e-9;  // successive estimates within stop * int |f|
  int max_level = 8;   // step 2^-(max_level + 1) at most
};

namespace detail {

// argument checks shared by the rules: an end or a centre must be finite, a
// scale positive and finite; a NaN fails both
LAPLACE_HD_FORCEINLINE bool finite(double x) { return Kokkos::fabs(x) <= 1.7976931348623157e308; }
LAPLACE_HD_FORCEINLINE bool scale_ok(double s) { return s > 0.0 && finite(s); }
// invalid arguments: NaN value and error, not converged
LAPLACE_HD_FORCEINLINE quad_result invalid() {
  quad_result r;
  r.value = r.error = laplace::math::detail::quiet_nan();
  return r;
}
// an empty interval: exactly 0, converged
LAPLACE_HD_FORCEINLINE quad_result empty() {
  quad_result r;
  r.converged = true;
  return r;
}

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
LAPLACE_HD quad_result de_rule(const F& f, const Node& node, const de_quad_options& o) {
  quad_result r;
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
LAPLACE_DEVICE_TABLE double gk15_x_dev[8] = {LAPLACE_QUAD_GK15_X};
LAPLACE_DEVICE_TABLE double gk15_w_dev[8] = {LAPLACE_QUAD_GK15_W};
LAPLACE_DEVICE_TABLE double g7_w_dev[4] = {LAPLACE_QUAD_G7_W};
LAPLACE_DEVICE_TABLE double legendre16_dev[32] = {LAPLACE_QUAD_LEGENDRE16};
LAPLACE_DEVICE_TABLE double legendre32_dev[64] = {LAPLACE_QUAD_LEGENDRE32};
LAPLACE_DEVICE_TABLE double legendre64_dev[128] = {LAPLACE_QUAD_LEGENDRE64};
LAPLACE_DEVICE_TABLE double hermite32_dev[64] = {LAPLACE_QUAD_HERMITE32};
LAPLACE_DEVICE_TABLE double laguerre32_dev[64] = {LAPLACE_QUAD_LAGUERRE32};
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

/**
 * \ingroup quadrature
 * \brief Adaptive double-exponential tanh-sinh quadrature on a finite interval.
 *
 * Computes the definite integral:
 * \f[
 * I = \int_a^b f(x) \, \mathrm{d}x
 * \f]
 * using the double-exponential tanh-sinh transformation (Takahasi and Mori, 1974):
 * \f[
 * x(t) = \frac{a+b}{2} + \frac{b-a}{2} \tanh\left(\frac{\pi}{2} \sinh t\right), \quad t \in [-6.8, 6.8].
 * \f]
 * The integrand \f$f\f$ may be a unary functor \f$f(x)\f$ or a binary functor \f$f(x, x_c)\f$,
 * where \f$x_c\f$ is the signed distance to the nearer endpoint (\f$x_c = a - x\f$ on the left
 * half, \f$x_c = b - x\f$ on the right half).
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$ or \f$f(x, x_c)\f$.
 * \param f Integrand functor.
 * \param a Finite lower integration limit \f$a \in \mathbb{R}\f$.
 * \param b Finite upper integration limit \f$b \in \mathbb{R}\f$.
 * \param o Options specifying stopping tolerance and maximum level (default `de_quad_options{}`).
 * \return A \ref quad_result containing the integral value, error estimate, evaluation count, and convergence status.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$a, b \in \mathbb{R}\f$.
 * - Reversed interval: if \f$b < a\f$, \f$\int_a^b f = -\int_b^a f\f$; the result of the reversed call has the opposite value and the same error, evaluations and convergence.
 * - Empty interval: if \f$a = b\f$, returns value 0, error 0, `converged = true`, and no evaluation.
 * - Invalid arguments: if \f$a\f$ or \f$b\f$ is NaN or infinite, returns value and error NaN and `converged = false` (evaluations 0).
 * - If a node rounds to an endpoint in double precision while \f$x_c \ne 0\f$, it is evaluated; if the integrand returns a non-finite value or \f$|v| \ge 10^{308}\f$ at an endpoint singularity, the node is skipped.
 * - When the rule does not converge, `value` holds the last estimate and `error` holds the last difference between successive estimates.
 *
 * \par Measured accuracy
 * Maximum error 6.5 ulp (gcc) / 7.8 ulp (CUDA) on the beta integral with distance to end (test bound 10.0 ulp, 88 evaluations);
 * 6.4 ulp (gcc) / 7.4 ulp (CUDA) on \f$x^p \log(x)^2\f$ (test bound 10.0 ulp, 72 evaluations).
 *
 * \see quadrature, exp_sinh_quad(), sinh_sinh_quad(), gauss_kronrod_quad(), panels_quad(), de_quad_options, quad_result
 */
template <class F>
LAPLACE_HD quad_result tanh_sinh_quad(const F& f, double a, double b, const de_quad_options& o = de_quad_options{}) {
  if (!detail::finite(a) || !detail::finite(b)) return detail::invalid();
  if (a == b) return detail::empty();
  if (b < a) {  // int_a^b = -int_b^a
    quad_result r = detail::de_rule(f, detail::tanh_sinh_node{b, a}, o);
    r.value = -r.value;
    return r;
  }
  return detail::de_rule(f, detail::tanh_sinh_node{a, b}, o);
}

/**
 * \ingroup quadrature
 * \brief Adaptive double-exponential exp-sinh quadrature on a semi-infinite interval.
 *
 * Computes the definite integral:
 * \f[
 * I = \int_a^\infty f(x) \, \mathrm{d}x
 * \f]
 * using the double-exponential exp-sinh transformation (Takahasi and Mori, 1974):
 * \f[
 * x(t) = a + s \exp\left(\frac{\pi}{2} \sinh t\right), \quad t \in [-6.8, 6.8]
 * \f]
 * where \f$s > 0\f$ is the characteristic scale of decay of the integrand.
 * The integrand \f$f\f$ may be a unary functor \f$f(x)\f$ or a binary functor \f$f(x, x_c)\f$,
 * where \f$x_c = x - a\f$ is the exact distance from the lower limit \f$a\f$.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$ or \f$f(x, x_c)\f$.
 * \param f Integrand functor.
 * \param a Finite lower integration limit \f$a \in \mathbb{R}\f$.
 * \param s Scale parameter of the integrand's decay \f$s > 0\f$ (default 1.0).
 * \param o Options specifying stopping tolerance and maximum level (default `de_quad_options{}`).
 * \return A \ref quad_result containing the integral value, error estimate, evaluation count, and convergence status.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$a \in \mathbb{R}\f$ and \f$s > 0\f$.
 * - Integrable singularities at \f$x = a\f$ that evaluate to non-finite values or \f$|v| \ge 10^{308}\f$ are skipped.
 * - Invalid arguments: if \f$a\f$ is NaN or infinite, or if scale \f$s\f$ is not positive and finite (\f$s \le 0\f$, infinite, or NaN), returns value and error NaN and `converged = false` (evaluations 0).
 * - When the rule does not converge, `value` holds the last estimate and `error` holds the last difference between successive estimates.
 *
 * \par Measured accuracy
 * Maximum error 68 ulp (gcc) / 20 ulp (CUDA) on the gamma integral (test bound 100.0 ulp, 136 evaluations);
 * 5.4 ulp on the Student t tail (test bound 8.0 ulp, 85 evaluations).
 *
 * \see quadrature, tanh_sinh_quad(), sinh_sinh_quad(), gauss_laguerre32_quad(), panels_quad(), de_quad_options, quad_result
 */
template <class F>
LAPLACE_HD quad_result exp_sinh_quad(const F& f, double a, double s = 1.0, const de_quad_options& o = de_quad_options{}) {
  if (!detail::finite(a) || !detail::scale_ok(s)) return detail::invalid();
  return detail::de_rule(f, detail::exp_sinh_node{a, s}, o);
}

/**
 * \ingroup quadrature
 * \brief Adaptive double-exponential sinh-sinh quadrature over the real line.
 *
 * Computes the definite integral:
 * \f[
 * I = \int_{-\infty}^\infty f(x) \, \mathrm{d}x
 * \f]
 * using the double-exponential sinh-sinh transformation (Takahasi and Mori, 1974):
 * \f[
 * x(t) = c + s \sinh\left(\frac{\pi}{2} \sinh t\right), \quad t \in [-6.8, 6.8]
 * \f]
 * where \f$c \in \mathbb{R}\f$ is the centre and \f$s > 0\f$ is the scale parameter.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param c Centre parameter of the integrand \f$c \in \mathbb{R}\f$ (default 0.0).
 * \param s Scale parameter of the integrand \f$s > 0\f$ (default 1.0).
 * \param o Options specifying stopping tolerance and maximum level (default `de_quad_options{}`).
 * \return A \ref quad_result containing the integral value, error estimate, evaluation count, and convergence status.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$c \in \mathbb{R}\f$ and \f$s > 0\f$.
 * - Nodes with displacement \f$|\sinh(q)| \ge 10^{300}\f$ (where \f$q = \frac{\pi}{2} \sinh t\f$) are skipped; integrand values are skipped by the common test \f$|v| < 10^{308}\f$ of every double-exponential rule (non-finite values and finite values with \f$|v| \ge 10^{308}\f$).
 * - Invalid arguments: if centre \f$c\f$ is NaN or infinite, or if scale \f$s\f$ is not positive and finite (\f$s \le 0\f$, infinite, or NaN), returns value and error NaN and `converged = false` (evaluations 0).
 * - When the rule does not converge, `value` holds the last estimate and `error` holds the last difference between successive estimates.
 *
 * \par Measured accuracy
 * Maximum error 7.3 ulp (gcc) / 6.7 ulp (CUDA) on \f$2 x^2 \phi(x) \phi(p x)^2 / \Phi(p x)\f$, with \f$\phi\f$ and \f$\Phi\f$ the standard normal density and distribution function (test bound 11.0 ulp, 159 evaluations);
 * 18 ulp on \f$\exp(-p(\sqrt{1 + x^2} - 1))\f$ (test bound 27.0 ulp, 615 evaluations);
 * 6.3 ulp on \f$\mathrm{sech}^p\f$ (test bound 10.0 ulp, 182 evaluations).
 *
 * \see quadrature, tanh_sinh_quad(), exp_sinh_quad(), gauss_hermite32_quad(), panels_quad(), de_quad_options, quad_result
 */
template <class F>
LAPLACE_HD quad_result sinh_sinh_quad(const F& f, double c = 0.0, double s = 1.0, const de_quad_options& o = de_quad_options{}) {
  if (!detail::finite(c) || !detail::scale_ok(s)) return detail::invalid();
  return detail::de_rule(f, detail::sinh_sinh_node{c, s}, o);
}

// adaptive Gauss-Kronrod 7-15: bisect the interval with the largest error
// until the sum of the errors is below tol times |value|, at most 64 intervals
/**
 * \ingroup quadrature
 * \brief Adaptive Gauss-Kronrod 7–15 quadrature on a finite interval.
 *
 * Computes the definite integral:
 * \f[
 * I = \int_a^b f(x) \, \mathrm{d}x
 * \f]
 * using the adaptive Gauss-Kronrod 7–15 rule with interval bisection.
 * On each subinterval, evaluates a 15-point Kronrod estimate \f$K\f$ and an embedded 7-point
 * Gauss estimate \f$G\f$ using constants from Boost.Math. The interval with the largest error
 * is iteratively bisected until the total error satisfies \f$\sum \mathrm{err}_i \le \mathrm{tol} \times |\sum \mathrm{val}_i|\f$
 * or the maximum number of subintervals (at most 64) is reached.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param a Finite lower integration limit \f$a \in \mathbb{R}\f$.
 * \param b Finite upper integration limit \f$b \in \mathbb{R}\f$.
 * \param tol Relative convergence tolerance (default \f$10^{-14}\f$).
 * \param max_intervals Maximum number of subintervals (default 64, capped at 64).
 * \return A \ref quad_result containing the integral value, error estimate, evaluation count, and convergence status.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$a, b \in \mathbb{R}\f$.
 * - Reversed interval: if \f$b < a\f$, \f$\int_a^b f = -\int_b^a f\f$; the result of the reversed call has the opposite value and the same error, evaluations and convergence.
 * - Empty interval: if \f$a = b\f$, returns value 0, error 0, `converged = true`, and no evaluation.
 * - Invalid arguments: if \f$a\f$ or \f$b\f$ is NaN or infinite, returns value and error NaN and `converged = false` (evaluations 0).
 * - If midpoint bisection underflows in double precision, iteration terminates.
 * - Fails to converge on integrands with endpoint singularities.
 * - When the rule does not converge, `value` holds the last accumulated estimate and `error` holds the last accumulated error estimate.
 *
 * \par Measured accuracy
 * Maximum error 72 ulp on the truncated Laplace density with a kink at 0 (test bound 110.0 ulp, 167 evaluations).
 *
 * \see quadrature, tanh_sinh_quad(), gauss_legendre_quad(), panels_quad(), quad_result
 */
template <class F>
LAPLACE_HD quad_result gauss_kronrod_quad(const F& f, double a, double b, double tol = 1e-14, int max_intervals = 64) {
  constexpr int cap = 64;
  if (!detail::finite(a) || !detail::finite(b)) return detail::invalid();
  if (a == b) return detail::empty();  // b < a: the rule itself gives -int_b^a
  if (max_intervals > cap) max_intervals = cap;
  double lo[cap], hi[cap], val[cap], err[cap];
  quad_result r;
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
/**
 * \ingroup quadrature
 * \brief Fixed Gauss-Legendre quadrature with 16, 32, or 64 nodes on a finite interval.
 *
 * Evaluates the definite integral:
 * \f[
 * \int_a^b f(x) \, \mathrm{d}x \approx \frac{b - a}{2} \sum_{i=0}^{N-1} w_i f\left(\frac{a+b}{2} + \frac{b-a}{2} x_i\right)
 * \f]
 * where \f$x_i \in (-1, 1)\f$ and \f$w_i > 0\f$ are the Gauss-Legendre nodes and weights
 * generated via the Golub-Welsch algorithm at 256 bits. Exact for polynomials of degree up to \f$2N - 1\f$.
 *
 * \tparam N Number of quadrature nodes, must be 16, 32, or 64.
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param a Finite lower integration limit \f$a \in \mathbb{R}\f$.
 * \param b Finite upper integration limit \f$b \in \mathbb{R}\f$.
 * \return The computed integral value in double precision.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$a, b \in \mathbb{R}\f$.
 * - Reversed interval: if \f$b < a\f$, \f$\int_a^b f = -\int_b^a f\f$; the result of the reversed call has the opposite value.
 * - Empty interval: if \f$a = b\f$, returns \f$0.0\f$.
 *
 * \par Tests
 * In `tests/math/test_quadrature.cpp`, tested on monomials \f$x^k\f$ on \f$[0, 1]\f$ for \f$k \in \{0, 1, 7, 31\}\f$
 * for \f$N = 16\f$ and \f$N = 64\f$ (\f$N = 32\f$ untested) against test tolerance \f$4 \times 10^{-16}\f$ (`doctest::Approx` epsilon).
 *
 * \see quadrature, gauss_kronrod_quad(), tanh_sinh_quad()
 */
template <int N, class F>
LAPLACE_HD double gauss_legendre_quad(const F& f, double a, double b) {
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

/**
 * \ingroup quadrature
 * \brief Fixed 32-node Gauss-Hermite quadrature over the real line.
 *
 * Evaluates the definite integral:
 * \f[
 * \int_{-\infty}^\infty f(x) \, \mathrm{d}x \approx s \sum_{i=0}^{31} w_i e^{y_i^2} f(c + s y_i)
 * \f]
 * where \f$y_i\f$ and \f$w_i\f$ are the 32 Gauss-Hermite nodes and weights for the weight
 * function \f$e^{-y^2}\f$, computed via the Golub-Welsch algorithm at 256 bits.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param c Centre parameter of the integrand \f$c \in \mathbb{R}\f$ (default 0.0).
 * \param s Scale parameter of the integrand \f$s > 0\f$ (default 1.0).
 * \return The computed integral value in double precision.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$c \in \mathbb{R}\f$ and \f$s > 0\f$.
 * - Invalid arguments: if centre \f$c\f$ is NaN or infinite, or if scale \f$s\f$ is not positive and finite (\f$s \le 0\f$, infinite, or NaN), returns NaN.
 *
 * \par Tests
 * In `tests/math/test_quadrature.cpp`, tested on monomials times the weight function, \f$x^{2j} e^{-x^2}\f$
 * for \f$j \in \{0, 1, 2, 3\}\f$, against test tolerance \f$10^{-14}\f$ (`doctest::Approx` epsilon).
 *
 * \see quadrature, sinh_sinh_quad(), gauss_laguerre32_quad()
 */
template <class F>
LAPLACE_HD double gauss_hermite32_quad(const F& f, double c = 0.0, double s = 1.0) {
  if (!detail::finite(c) || !detail::scale_ok(s)) return laplace::math::detail::quiet_nan();
  double acc = 0.0;
  for (int i = 0; i < 32; ++i) {
    const double y = detail::hermite32(i), w = detail::hermite32(32 + i);
    acc += w * laplace::math::exp(y * y) * f(c + s * y);
  }
  return acc * s;
}

/**
 * \ingroup quadrature
 * \brief Fixed 32-node Gauss-Laguerre quadrature on a semi-infinite interval.
 *
 * Evaluates the definite integral:
 * \f[
 * \int_a^\infty f(x) \, \mathrm{d}x \approx s \sum_{i=0}^{31} w_i e^{y_i} f(a + s y_i)
 * \f]
 * where \f$y_i\f$ and \f$w_i\f$ are the 32 Gauss-Laguerre nodes and weights for the weight
 * function \f$e^{-y}\f$, computed via the Golub-Welsch algorithm at 256 bits.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param a Finite lower integration limit \f$a \in \mathbb{R}\f$ (default 0.0).
 * \param s Scale parameter of the integrand \f$s > 0\f$ (default 1.0).
 * \return The computed integral value in double precision.
 *
 * \par Domain and special values
 * - Standard domain: finite \f$a \in \mathbb{R}\f$ and \f$s > 0\f$.
 * - Invalid arguments: if start \f$a\f$ is NaN or infinite, or if scale \f$s\f$ is not positive and finite (\f$s \le 0\f$, infinite, or NaN), returns NaN.
 *
 * \par Tests
 * In `tests/math/test_quadrature.cpp`, tested on monomials times the weight function, \f$x^k e^{-x}\f$
 * for \f$k \in \{0, \dots, 9\}\f$, against test tolerance \f$10^{-14}\f$ (`doctest::Approx` epsilon).
 *
 * \see quadrature, exp_sinh_quad(), gauss_hermite32_quad()
 */
template <class F>
LAPLACE_HD double gauss_laguerre32_quad(const F& f, double a = 0.0, double s = 1.0) {
  if (!detail::finite(a) || !detail::scale_ok(s)) return laplace::math::detail::quiet_nan();
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
/**
 * \ingroup quadrature
 * \brief Geometric panel quadrature rule on a finite or infinite interval cut at a centre and kinks.
 *
 * Cuts the interval into pieces and each piece into geometric panels:
 * - The interval \f$[a, b]\f$ is cut at the centre \f$c\f$ and at up to 8 user-supplied interior kink points.
 * - Each finite subpiece \f$[u, v]\f$ is partitioned into doubling geometric panels of widths \f$s, 2s, 4s, \dots, 2^k s\f$
 *   advancing from the end closer to \f$c\f$, with \ref tanh_sinh_quad() evaluated on each panel.
 * - Pieces extending to \f$\pm\infty\f$ use doubling panels over the first \f$63s\f$, followed by \ref exp_sinh_quad() on the unbounded tail.
 * - Convergence is checked on each panel individually; the result is marked converged only if every panel converged.
 *
 * \tparam F Integrand functor type callable as \f$f(x)\f$.
 * \param f Integrand functor.
 * \param a Lower integration boundary \f$a \in [-\infty, \infty]\f$.
 * \param b Upper integration boundary \f$b \in [-\infty, \infty]\f$.
 * \param c Centre point \f$c \in \mathbb{R}\f$.
 * \param s Base panel width and scale parameter \f$s > 0\f$.
 * \param kinks Optional array of interior kink coordinates (at most 8 used).
 * \param nk Number of elements in `kinks` array.
 * \param o Options specifying stopping tolerance and maximum level (default `de_quad_options{}`).
 * \return A \ref quad_result containing the accumulated integral value, accumulated error, total evaluation count, and combined convergence status.
 *
 * \par Domain and special values
 * - Standard domain: \f$-\infty \le a, b \le \infty\f$, finite centre \f$c \in \mathbb{R}\f$, and \f$s > 0\f$.
 * - Reversed interval: if \f$b < a\f$, \f$\int_a^b f = -\int_b^a f\f$; the result of the reversed call has the opposite value and the same error, evaluations and convergence.
 * - Empty interval: if \f$a = b\f$, returns value 0, error 0, `converged = true`, and no evaluation.
 * - Invalid arguments: if \f$a\f$ or \f$b\f$ is NaN, if centre \f$c\f$ is not finite (NaN or infinite), or if scale \f$s\f$ is not positive and finite (\f$s \le 0\f$, infinite, or NaN), returns value and error NaN and `converged = false` (evaluations 0). Integration limits \f$a\f$ and \f$b\f$ may be infinite.
 * - Boundaries with \f$a \le -10^{308}\f$ or \f$b \ge 10^{308}\f$ are treated as infinite.
 * - Takes integrands of \f$x\f$ alone; does not support distance-to-endpoint evaluation \f$f(x, x_c)\f$.
 * - When the rule does not converge, `value` holds the accumulated estimate and `error` holds the accumulated error.
 *
 * \par Measured accuracy
 * Maximum error 13 ulp (gcc) / 21 ulp (CUDA) on gamma (test bound 32.0 ulp, 839 evaluations);
 * 4.0 ulp on Student t tail (test bound 6.0 ulp, 793 evaluations);
 * 4.5 ulp on truncated Laplace (test bound 7.0 ulp, 229 evaluations);
 * 3.9 ulp on \f$x^p \log(x)^2\f$ (test bound 6.0 ulp, 274 evaluations);
 * 6.8 ulp (gcc) / 5.4 ulp (CUDA) on \f$2 x^2 \phi(x) \phi(p x)^2 / \Phi(p x)\f$, with \f$\phi\f$ and \f$\Phi\f$ the standard normal density and distribution function (test bound 10.0 ulp, 1527 evaluations);
 * 2.8 ulp on \f$\exp(-p(\sqrt{1 + x^2} - 1))\f$ (test bound 5.0 ulp, 1557 evaluations);
 * 3.7 ulp on \f$\mathrm{sech}^p\f$ (test bound 6.0 ulp, 1561 evaluations).
 *
 * \see quadrature, tanh_sinh_quad(), exp_sinh_quad(), sinh_sinh_quad(), de_quad_options, quad_result
 */
template <class F>
LAPLACE_HD quad_result panels_quad(const F& f, double a, double b, double c, double s, const double* kinks = nullptr, int nk = 0,
                         const de_quad_options& o = de_quad_options{}) {
  if (a != a || b != b || !detail::finite(c) || !detail::scale_ok(s)) return detail::invalid();
  if (a == b) return detail::empty();
  if (b < a) {  // int_a^b = -int_b^a
    quad_result r = panels_quad(f, b, a, c, s, kinks, nk, o);
    r.value = -r.value;
    return r;
  }
  quad_result r;
  r.converged = true;
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
  auto add = [&](const quad_result& p) {
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
        add(from_u ? tanh_sinh_quad(f, at, v, o) : tanh_sinh_quad(f, u, at, o));
        return;
      }
      add(from_u ? tanh_sinh_quad(f, at, next, o) : tanh_sinh_quad(f, next, at, o));
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
        add(exp_sinh_quad(f, m + 63.0 * s, s, o));
        finite(m - 63.0 * s, m);
        add(exp_sinh_quad([&](double x) { return f(-x); }, -(m - 63.0 * s), s, o));
        return r;
      }
      finite(v - 63.0 * s, v);
      add(exp_sinh_quad([&](double x) { return f(-x); }, -(v - 63.0 * s), s, o));
    } else if (i + 2 == ne && inf_hi) {
      finite(u, u + 63.0 * s);
      add(exp_sinh_quad(f, u + 63.0 * s, s, o));
    } else {
      finite(u, v);
    }
  }
  return r;
}

}  // namespace laplace::math::quad
