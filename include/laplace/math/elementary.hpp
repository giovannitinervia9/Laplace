// Elementary functions of Laplace.
//
// Two implementations of each function live side by side until the
// measurement decides:
//
//   laplace::math::platform::  the platform's own (Kokkos -> std:: on the host,
//                              CUDA's libdevice on the GPU);
//   laplace::math::own::       Laplace's own, written once for every platform.
//
// The own versions follow the algorithms of fdlibm 5.3 (Sun Microsystems,
// e_exp.c and e_log.c: "Developed at SunPro, a Sun Microsystems, Inc.
// business. Permission to use, copy, modify, and distribute this software is
// freely granted, provided that this notice is preserved."). They need no
// table, which suits the GPU, and fdlibm documents an error below one ulp.
//
// laplace::math::exp and laplace::math::log are what the rest of Laplace
// calls. Since 2026-11-11 they forward to the own versions on the host
// (CMake option LAPLACE_OWN_ELEMENTARY, on by default; OFF defines
// LAPLACE_PLATFORM_ELEMENTARY and restores the platform's) and to
// the platform versions (libdevice) on the device; see docs/numerics.md.
#pragma once

#include <Kokkos_Core.hpp>
#include <laplace/core/config.hpp>

#include <cstdint>

namespace laplace::math {

namespace detail {

LAPLACE_HD_FORCEINLINE double from_bits(std::uint64_t u) { return Kokkos::bit_cast<double>(u); }
LAPLACE_HD_FORCEINLINE std::uint64_t to_bits(double x) { return Kokkos::bit_cast<std::uint64_t>(x); }
// 2^k for -1022 <= k <= 1023, built in the exponent field.
LAPLACE_HD_FORCEINLINE double pow2i(int k) { return from_bits(std::uint64_t(k + 1023) << 52); }
LAPLACE_HD_FORCEINLINE double pos_inf() { return from_bits(0x7ff0000000000000ULL); }
LAPLACE_HD_FORCEINLINE double quiet_nan() { return from_bits(0x7ff8000000000000ULL); }

}  // namespace detail

namespace platform {

LAPLACE_HD_FORCEINLINE double exp(double x) { return Kokkos::exp(x); }
LAPLACE_HD_FORCEINLINE double log(double x) { return Kokkos::log(x); }

}  // namespace platform

namespace own {

// exp(x): x = k ln2 + r with |r| <= ln2 / 2 (ln2 split in a high part with
// trailing zeros and a low part, so that k ln2_hi is exact), then
// exp(r) = 1 + 2 r / (R(r^2) - r) with R a degree-5 minimax polynomial in r^2,
// and the result scaled by 2^k.
LAPLACE_HD double exp(double x) {
  constexpr double o_threshold = 7.09782712893383973096e+02;
  constexpr double u_threshold = -7.45133219101941108420e+02;
  constexpr double ln2_hi = 6.93147180369123816490e-01;
  constexpr double ln2_lo = 1.90821492927058770002e-10;
  constexpr double inv_ln2 = 1.44269504088896338700e+00;
  constexpr double P1 = 1.66666666666666019037e-01;
  constexpr double P2 = -2.77777777770155933842e-03;
  constexpr double P3 = 6.61375632143793436117e-05;
  constexpr double P4 = -1.65339022054652515390e-06;
  constexpr double P5 = 4.13813679705723846039e-08;

  if (!(x == x)) return x + x;                    // NaN
  if (x > o_threshold) return detail::pos_inf();  // overflow
  if (x < u_threshold) return 0.0;                // underflow
  const double ax = x < 0.0 ? -x : x;

  double hi = 0.0, lo = 0.0, r;
  int k = 0;
  if (ax > 0.5 * 6.93147180559945286227e-01) {
    if (ax < 1.5 * 6.93147180559945286227e-01) {
      k = x < 0.0 ? -1 : 1;
    } else {
      k = static_cast<int>(inv_ln2 * x + (x < 0.0 ? -0.5 : 0.5));
    }
    const double dk = static_cast<double>(k);
    hi = x - dk * ln2_hi;
    lo = dk * ln2_lo;
    r = hi - lo;
  } else if (ax < 3.7252902984619140625e-09) {    // |x| < 2^-28
    return 1.0 + x;
  } else {
    r = x;
  }

  const double t = r * r;
  const double c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
  if (k == 0) return 1.0 - ((r * c) / (c - 2.0) - r);
  const double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
  if (k >= -1021) {
    if (k == 1024) return y * 2.0 * 8.98846567431157953865e+307;  // y * 2^1024
    return y * detail::pow2i(k);
  }
  return y * detail::pow2i(k + 1000) * 9.33263618503218878990e-302;  // 2^-1000
}

// log(x): x = 2^k (1 + f) with sqrt(2)/2 < 1 + f < sqrt(2); with s = f / (2 + f),
// log(1 + f) = 2 s + s R(s^2), R a degree-7 minimax polynomial in s^2 split in
// even and odd parts; k ln2 is added in two pieces as in exp.
LAPLACE_HD double log(double x) {
  constexpr double ln2_hi = 6.93147180369123816490e-01;
  constexpr double ln2_lo = 1.90821492927058770002e-10;
  constexpr double two54 = 1.80143985094819840000e+16;
  constexpr double Lg1 = 6.666666666666735130e-01;
  constexpr double Lg2 = 3.999999999940941908e-01;
  constexpr double Lg3 = 2.857142874366239149e-01;
  constexpr double Lg4 = 2.222219843214978396e-01;
  constexpr double Lg5 = 1.818357216161805012e-01;
  constexpr double Lg6 = 1.531383769920937332e-01;
  constexpr double Lg7 = 1.479819860511658591e-01;

  std::uint64_t u = detail::to_bits(x);
  std::int32_t hx = static_cast<std::int32_t>(u >> 32);
  const std::uint32_t lx = static_cast<std::uint32_t>(u);
  int k = 0;
  if (hx < 0x00100000) {                          // x < 2^-1022, or negative
    if (((hx & 0x7fffffff) | static_cast<std::int32_t>(lx)) == 0) return -detail::pos_inf();
    if (hx < 0) return detail::quiet_nan();       // log(negative)
    k -= 54;                                      // subnormal: scale up
    x *= two54;
    u = detail::to_bits(x);
    hx = static_cast<std::int32_t>(u >> 32);
  }
  if (hx >= 0x7ff00000) return x + x;             // inf or NaN
  k += (hx >> 20) - 1023;
  hx &= 0x000fffff;
  std::int32_t i = (hx + 0x95f64) & 0x100000;
  // normalize x or x / 2 into [sqrt(2)/2, sqrt(2))
  x = detail::from_bits((std::uint64_t(static_cast<std::uint32_t>(hx | (i ^ 0x3ff00000))) << 32) |
                        (u & 0xffffffffULL));
  k += (i >> 20);
  const double f = x - 1.0;
  const double dk = static_cast<double>(k);
  if ((0x000fffff & (2 + hx)) < 3) {              // -2^-20 <= f < 2^-20
    if (f == 0.0) {
      if (k == 0) return 0.0;
      return dk * ln2_hi + dk * ln2_lo;
    }
    const double R = f * f * (0.5 - 0.33333333333333333 * f);
    if (k == 0) return f - R;
    return dk * ln2_hi - ((R - dk * ln2_lo) - f);
  }
  const double s = f / (2.0 + f);
  const double z = s * s;
  i = hx - 0x6147a;
  const double w = z * z;
  const std::int32_t j = 0x6b851 - hx;
  const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
  const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
  i |= j;
  const double R = t2 + t1;
  if (i > 0) {
    const double hfsq = 0.5 * f * f;
    if (k == 0) return f - (hfsq - s * (hfsq + R));
    return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
  }
  if (k == 0) return f - s * (f - R);
  return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}

}  // namespace own

// What the rest of Laplace calls (docs/numerics.md, 2026-11-11): the own
// versions on the host, the same algorithm on every CPU platform (the libm of
// Rtools costs about 28 ns per exp or log, the own ones 4-6 ns), and the
// platform's (CUDA's) on the device. Defining LAPLACE_PLATFORM_ELEMENTARY
// (CMake: LAPLACE_OWN_ELEMENTARY=OFF) puts the platform's on the host as well.
#if !defined(LAPLACE_PLATFORM_ELEMENTARY)
/**
 * \ingroup elementary
 * \brief Exponential function, \f$\exp(x)\f$.
 *
 * Evaluates Laplace's own table-free fdlibm-style implementation (`own::exp`) by default
 * on host execution spaces (6.4 ns/call on gcc, compared to 29.0 ns/call for the platform's
 * libm) and CUDA's libdevice (`platform::exp`) on device execution spaces (0.070 ns/call
 * on CUDA). Defining `LAPLACE_PLATFORM_ELEMENTARY` directs host calls to the platform's libm as well.
 *
 * \param x Real argument \f$x \in \mathbb{R}\f$.
 * \return The value of \f$\exp(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in \mathbb{R}\f$: standard domain.
 * - If \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0.0\f$: returns \f$1.0\f$.
 * - \f$x = +\infty\f$: returns \f$+\infty\f$.
 * - \f$x = -\infty\f$: returns \f$0.0\f$.
 * - \f$x > 709.782712893383973096\f$ (overflow threshold): returns \f$+\infty\f$.
 * - \f$x < -745.133219101941108420\f$ (underflow threshold): returns \f$0.0\f$.
 * - \f$|x| < 2^{-28}\f$: returns \f$1.0 + x\f$ for the own version on the host only (on the device the platform's is called).
 *
 * \par Measured accuracy
 * Maximum error 1.0000 ulp on CPU (gcc, MSVC) and GPU (CUDA); 0.5002 ulp for the platform
 * candidate on gcc (share of correctly rounded results 0.9270 for own, 0.9997 for platform).
 *
 * \see elementary, log()
 */
LAPLACE_HD_FORCEINLINE double exp(double x) {
  KOKKOS_IF_ON_HOST((return own::exp(x);))
  KOKKOS_IF_ON_DEVICE((return platform::exp(x);))
}

/**
 * \ingroup elementary
 * \brief Natural logarithm, \f$\log(x)\f$.
 *
 * Evaluates Laplace's own table-free fdlibm-style implementation (`own::log`) by default
 * on host execution spaces (4.1 ns/call on gcc, compared to 27.6 ns/call for the platform's
 * libm) and CUDA's libdevice (`platform::log`) on device execution spaces (0.131 ns/call
 * on CUDA). Defining `LAPLACE_PLATFORM_ELEMENTARY` directs host calls to the platform's libm as well.
 *
 * \param x Positive real argument \f$x \in (0, \infty)\f$.
 * \return The value of \f$\log(x)\f$.
 *
 * \par Domain and special values
 * - \f$x \in (0, \infty)\f$: standard domain.
 * - If \f$x < 0.0\f$ or \f$x\f$ is NaN: returns quiet NaN.
 * - \f$x = 0.0\f$ (or \f$-0.0\f$): returns \f$-\infty\f$.
 * - \f$x = 1.0\f$: returns \f$0.0\f$.
 * - \f$x = +\infty\f$: returns \f$+\infty\f$.
 * - Subnormal arguments (\f$x < 2^{-1022}\f$): supported down to \f$5 \times 10^{-324}\f$.
 *
 * \par Measured accuracy
 * Maximum error 0.7767 ulp on gcc and MSVC, 0.7774 ulp on CUDA (share of correctly rounded
 * results 0.9799 on gcc, 0.9827 on CUDA); 0.5003 ulp for the platform candidate on gcc
 * (share of correctly rounded results 0.9992).
 *
 * \see elementary, exp()
 */
LAPLACE_HD_FORCEINLINE double log(double x) {
  KOKKOS_IF_ON_HOST((return own::log(x);))
  KOKKOS_IF_ON_DEVICE((return platform::log(x);))
}
#else
LAPLACE_HD_FORCEINLINE double exp(double x) { return platform::exp(x); }
LAPLACE_HD_FORCEINLINE double log(double x) { return platform::log(x); }
#endif

}  // namespace laplace::math
