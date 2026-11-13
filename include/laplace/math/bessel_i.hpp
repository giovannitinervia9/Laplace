// The modified Bessel functions of the first kind of orders 0 and 1, in the
// forms the von Mises distribution uses (laplace::math):
//
//   log_bessel_i0(k)          log I0(k), k >= 0
//   log_bessel_i0_scaled(k)   log I0(k) - k, k >= 0 (the log of e^-k I0(k),
//                             finite for every k, for densities written as
//                             exp(k (cos d - 1)) / (2 pi e^-k I0(k)))
//   bessel_i_ratio(k)         A(k) = I1(k) / I0(k), k >= 0; A' ... A^(6) as
//                             bessel_i_ratio_d1 ... _d6 (d/dk log I0 = A, so
//                             these are also the derivatives of log I0 of
//                             orders 2 to 7)
//   bessel_i_ratio_inv(rho)   K(rho), the inverse of A: A(K(rho)) = rho,
//                             0 <= rho < 1
//   bessel_i_ratio_inv_d1 ... _d6 (kappa)
//                             K^(n)(rho) at rho = A(kappa). They take kappa,
//                             not rho: a von Mises distribution in (mu, rho)
//                             computes kappa = K(rho) once, and kappa keeps
//                             the information that rho loses near 1.
//   bessel_i_ratios(k, m, out) I_j(k) / I_0(k) for j = 1..m, the coefficients
//                             of the series of the von Mises distribution
//                             function, by Miller's backward recurrence
//                             (measured against statmodels7's continued
//                             fraction with running products, and against
//                             double-double versions, in math_bessel)
//
// Arguments below 0 or NaN give NaN (rho outside [0, 1] also); A(0) = 0,
// K(0) = 0, K(1) = +inf.
//
// The ratio and its inverse are ported from statmodels7's own code
// (numericals7/src/bessel_ratio.cpp, Giovanni Tinervia), extended from order
// four to order six. Three regimes in k:
//
//   k < small (0.25)  the power series A(k) = sum_n a_n k^(2n+1), radius 2.405
//                     (the first zero of I0 on the imaginary axis),
//                     differentiated term by term. The Riccati identity
//                     A' = 1 - A/k - A^2 cancels there (its terms are of order
//                     k^-n at derivative n against a result of order one).
//   small <= k < large (30)
//                     A by its continued fraction r_j = k / (2j + k r_(j+1)),
//                     truncated at j = floor(k) + extra (extra = 20): backwards
//                     in double for the value, forwards by the Wallis
//                     recurrences of the convergents in double-double for the
//                     derivatives, which then follow from the Riccati identity
//                     differentiated n times, in double-double:
//                       A^(n+1) = [n = 0] - sum_j C(n, j) A^(j) ((1/k)^(n-j) + A^(n-j)),
//                     (1/k)^(m) = (-1)^m m! / k^(m+1). The identity loses about
//                     log2 k bits per order, which double-double absorbs.
//   k >= large        the asymptotic series A(k) ~ sum_n q_n k^-n (the
//                     quotient of those of I1 and I0), differentiated term by
//                     term and summed until the terms fall below the last bit.
//                     The rounding of u = 1/k would be amplified M + 1 times
//                     in u^M S_M(u); it is corrected to first order.
//
// The coefficient tables are exact rationals rounded once to double
// (tools/reference/bessel.py coefs).
//
// The inverse K(rho) is Newton's method on A(k) = rho (A is increasing and
// concave, so the iterates rise to the root from the left after the first
// step); for rho >= 1/2 the residual is (1 - rho) - (1 - A(k)), where 1 - rho
// is exact and 1 - A comes without cancellation from the continued fraction
// in double-double or from the asymptotic series. Its derivatives follow the
// inverse function rule written on w = 1/A' and r_j = A^(j)/A':
//   K'  = w,  K'' = -r2 w^2,  K''' = (3 r2^2 - r3) w^3,
//   K^(4) = (-15 r2^3 + 10 r2 r3 - r4) w^4,
//   K^(5) = (105 r2^4 - 105 r2^2 r3 + 15 r2 r4 + 10 r3^2 - r5) w^5,
//   K^(6) = (-945 r2^5 + 1260 r2^3 r3 - 210 r2^2 r4 - 280 r2 r3^2 + 21 r2 r5
//            + 35 r3 r4 - r6) w^6,
// with the powers of w applied one at a time (w ~ 2 k^2 and r_j ~ k^-(j-1)
// stay normal where A'^(2n-1) would not).
//
// log I0 comes from Boost's rational approximations of I0 for double (the
// forms of Holoborodko, 2015), rewritten in log form (see log_i0 below); it
// was measured against Laplace's own series and asymptotic expansion
// (benchmarks/math_bessel.cpp) and is both more accurate and faster.
//
// The derivatives of the inverse avoid the cancellation of the polynomial in
// r_j (whose terms reach a hundred times the result at order six) where they
// can: for k < small and k >= large they are power series in k and in 1/k
// with exact rational coefficients (inverse_series, inverse_asymptotic),
// normalized by their leading term; in between, the polynomial is evaluated
// in double-double.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/math/elementary.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::math {

namespace own {

namespace detail_bessel {

namespace ddm = laplace::math::detail::ddm;
using ddm::dd;

constexpr double eps = 2.220446049250313e-16;
constexpr double tiny_rel = 0x1p-56;                // stopping test of the series (below eps/8)
constexpr double log_2pi = 1.8378770664093454836;   // log(2 pi)

// A(k) = sum_n ratio_series(n) k^(2n+1)
// clang-format off
#define LAPLACE_BESSEL_RATIO_SERIES \
    0x1.0000000000000p-1,   -0x1.0000000000000p-4,  0x1.5555555555555p-7,   -0x1.d555555555555p-10, \
    0x1.4444444444444p-12,  -0x1.c0795ceb24079p-15, 0x1.362e0d8b8362ep-17,  -0x1.ad133bde68913p-20, \
    0x1.28c61905e8dd0p-22,  -0x1.9a8880ee348fap-25, 0x1.1bf312ba0f5e4p-27,  -0x1.88cb0df4dd495p-30, \
    0x1.0fadefc84c6c2p-32,  -0x1.77d1db538374dp-35, 0x1.03f0893c4d382p-37,  -0x1.67946ca504650p-40, \
    0x1.f16a02b32cab7p-43,  -0x1.580aa4bebd258p-45, 0x1.dbeb76ebb7368p-48,  -0x1.492cc0443da0bp-50, \
    0x1.c75ab1fa6c9a2p-53,  -0x1.3af351ba4a0d9p-55, 0x1.b3ad6d8399675p-58,  -0x1.2d573dd06e1dap-60, \
    0x1.a0d9d4d55d938p-63,  -0x1.2051b7d4045c0p-65
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double ratio_series_dev[26] = {LAPLACE_BESSEL_RATIO_SERIES};
#endif
inline constexpr double ratio_series_host[26] = {LAPLACE_BESSEL_RATIO_SERIES};
LAPLACE_HD_FORCEINLINE double ratio_series(int i) {
  KOKKOS_IF_ON_DEVICE((return ratio_series_dev[i];))
  KOKKOS_IF_ON_HOST((return ratio_series_host[i];))
}

// A(k) ~ sum_n ratio_asymptotic(n) k^-n
// clang-format off
#define LAPLACE_BESSEL_RATIO_ASYMPTOTIC \
    0x1.0000000000000p+0,   -0x1.0000000000000p-1,  -0x1.0000000000000p-3,  -0x1.0000000000000p-3, \
    -0x1.9000000000000p-3,  -0x1.a000000000000p-2,  -0x1.0c40000000000p+0,  -0x1.9c00000000000p+1, \
    -0x1.6eed400000000p+3,  -0x1.73d4000000000p+5,  -0x1.a68d638000000p+7,  -0x1.0a2b680000000p+10, \
    -0x1.7040bb3f40000p+12, -0x1.1591c14000000p+15, -0x1.c4e6516704400p+17, -0x1.8da4d74080000p+20, \
    -0x1.75df774407083p+23, -0x1.76cdfb7430a00p+26, -0x1.8f17890e3d642p+29, -0x1.c1d4bb515df36p+32, \
    -0x1.0b892e868aeebp+36, -0x1.4eebb01c12c2bp+39, -0x1.b82c8e8cdf382p+42, -0x1.2efc95f6b156ap+46, \
    -0x1.b405c4954374ap+49, -0x1.475847ad71e9cp+53, -0x1.fff22131bb76fp+56, -0x1.a04e8b739de4ep+60, \
    -0x1.5f88352e27d47p+64, -0x1.33cfd57a01475p+68, -0x1.172404ca7c505p+72, -0x1.05db5f934317dp+76, \
    -0x1.fba4459db9375p+79, -0x1.fbeaa7b1fd104p+83, -0x1.060704b0c5db5p+88, -0x1.16896b2b9a4d2p+92, \
    -0x1.30c943b5fb598p+96, -0x1.570782d29f236p+100, -0x1.8cc9306f863b5p+104, -0x1.d75c6a630f9dbp+108, \
    -0x1.1f56966688222p+113, -0x1.674b64a260973p+117, -0x1.cc7e86fd66587p+121, -0x1.2e4ab69952b38p+126, \
    -0x1.9652b41cfd057p+130, -0x1.176cba66f2412p+135, -0x1.890b92b4c7dfcp+139, -0x1.1a92a3d7b93dep+144, \
    -0x1.9f211ed469146p+148, -0x1.376b53ecf0eddp+153, -0x1.dcf77a5cf585cp+157, -0x1.74b5ad14b4cc2p+162, \
    -0x1.29105484192c1p+167, -0x1.e2d2d3de2b912p+171, -0x1.8fe9ef9597647p+176, -0x1.517d1737b203dp+181, \
    -0x1.22147fe20c82ep+186, -0x1.fbb9cc68f3965p+190, -0x1.c4444eb1d1c06p+195, -0x1.99ee5f1b8bd5bp+200
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double ratio_asymptotic_dev[60] = {LAPLACE_BESSEL_RATIO_ASYMPTOTIC};
#endif
inline constexpr double ratio_asymptotic_host[60] = {LAPLACE_BESSEL_RATIO_ASYMPTOTIC};
LAPLACE_HD_FORCEINLINE double ratio_asymptotic(int i) {
  KOKKOS_IF_ON_DEVICE((return ratio_asymptotic_dev[i];))
  KOKKOS_IF_ON_HOST((return ratio_asymptotic_host[i];))
}

// K^(n)(A(k)) = n! 2^n k^(n+1) (1 + sum_(m >= 1) inverse_asymptotic[n-1][m-1] k^-m)
// clang-format off
#define LAPLACE_BESSEL_INVERSE_ASYMPTOTIC \
    -0x1.0000000000000p-1, -0x1.0000000000000p-1, -0x1.e000000000000p-1, -0x1.3800000000000p+1, -0x1.f5c0000000000p+2, -0x1.d880000000000p+4, -0x1.f987800000000p+6, -0x1.2d89000000000p+9, -0x1.8c08cd6000000p+11, -0x1.1bb6360000000p+14, -0x1.b84e08e120000p+16, -0x1.6ff7c9fa00000p+19, -0x1.4998807432980p+22, -0x1.3b1ac73bd0000p+25, -0x1.405a32fe42e14p+28, -0x1.59325b3f291c0p+31, -0x1.8912f4d10017ep+34, -0x1.d7b4ce1819ec2p+37, -0x1.298b3506d271bp+41, -0x1.89b79086c7733p+44, -0x1.10a61e45a7038p+48, -0x1.8a7cb7917dd53p+51, -0x1.299ab040091b5p+55, -0x1.d37c1cce7c047p+58, -0x1.7dad1190b468fp+62, -0x1.437880f5c94edp+66, -0x1.1c31725c7fa1cp+70, -0x1.02853f980340fp+74, -0x1.e66a8a976e115p+77, -0x1.d8bda90cbdb72p+81, -0x1.da2a45a11aa79p+85, -0x1.ea5b8b5015e8ep+89, -0x1.05300742ca6a0p+94, -0x1.1e60db6fdafb0p+98, -0x1.42ec0543d4a9ep+102, -0x1.7631519789429p+106, -0x1.bd44771b07ae5p+110, -0x1.0fdca28eeb09dp+115, -0x1.5473a9318dacbp+119, -0x1.b4f6a2f853a15p+123, -0x1.1f3b2762f606dp+128, -0x1.82929fc798e78p+132, -0x1.0a2a70b94767ep+137, -0x1.76d40984a0c0cp+141, -0x1.0dc6010ddb925p+146, \
    -0x1.8000000000000p-1, -0x1.8000000000000p-2, -0x1.6000000000000p-2, 0x0.0p+0, 0x1.89c0000000000p+1, 0x1.7f40000000000p+4, 0x1.3c3c200000000p+7, 0x1.0165800000000p+10, 0x1.aebe24b800000p+12, 0x1.7830188000000p+15, 0x1.590b64f96c000p+18, 0x1.4d26ad2900000p+21, 0x1.52c5c39d9e3b0p+24, 0x1.6a9ff9625c000p+27, 0x1.982b27a307fe3p+30, 0x1.e27a477f991d0p+33, 0x1.2b04cd118f01ep+37, 0x1.840e1e90f5fafp+40, 0x1.0739cea480a25p+44, 0x1.74bc33de45b9ap+47, 0x1.130bdcbd6aff0p+51, 0x1.a676097c7df4ap+54, 0x1.51342b782fb07p+58, 0x1.1760a54bd4823p+62, 0x1.dfeef2455f129p+65, 0x1.aadee1c399d02p+69, 0x1.88b9e00831bbap+73, 0x1.755819290efa7p+77, 0x1.6e604fe9b30ccp+81, 0x1.72cc8a513f645p+85, 0x1.82afd46439b68p+89, 0x1.9f2b881d59bc1p+93, 0x1.ca8e8eb971aa2p+97, 0x1.04513d181fad7p+102, 0x1.2f9a012854032p+106, 0x1.6b7882a87b626p+110, 0x1.be64806f512bdp+114, 0x1.1906cbc95f1a7p+119, 0x1.6a8be691593c1p+123, 0x1.def35a3d1da2cp+127, 0x1.43ca54c51cf72p+132, 0x1.bfd5ff41c36e6p+136, 0x1.3ca6add581f2fp+141, 0x1.c99e83316f616p+145, 0x1.51c7250ef7755p+150, \
    -0x1.0000000000000p+0, -0x1.8000000000000p-2, -0x1.4000000000000p-1, -0x1.e800000000000p+0, -0x1.11c0000000000p+3, -0x1.81e0000000000p+5, -0x1.3475400000000p+8, -0x1.0a9b500000000p+11, -0x1.e8ded7b000000p+13, -0x1.d7fed3c000000p+16, -0x1.de8490fc98000p+19, -0x1.fcc2001fe0000p+22, -0x1.1b62587bb9330p+26, -0x1.4a856b1fc8c00p+29, -0x1.933d796d79dbfp+32, -0x1.010af57645f9cp+36, -0x1.56119b4e9022ap+39, -0x1.daa314b7d9075p+42, -0x1.56ef2e6329eebp+46, -0x1.01bd0fab3890ap+50, -0x1.9285a5f67eab4p+53, -0x1.463378c20f148p+57, -0x1.120a408069098p+61, -0x1.dccddfe6d9a36p+64, -0x1.ad18f4fa452f4p+68, -0x1.8f152bf2e5b0cp+72, -0x1.7f38e99de00e7p+76, -0x1.7b9a015ed4e08p+80, -0x1.838b3c681cee4p+84, -0x1.9772e16d4b893p+88, -0x1.b8cd8539c3756p+92, -0x1.ea5aea25e6641p+96, -0x1.183dddf6e4222p+101, -0x1.48e88cf1ef0aap+105, -0x1.8c1e69efd58b1p+109, -0x1.e93b5793ef282p+113, -0x1.35a374e11a4a2p+118, -0x1.9179e43355a24p+122, -0x1.0a75e9f02593dp+127, -0x1.69eaacd9fddd3p+131, -0x1.f6bc5eb943fe9p+135, -0x1.64ef434cc4e34p+140, -0x1.02ee8fac1a17ap+145, -0x1.7fadf4e0a3a17p+149, -0x1.22328dd73905fp+154, \
    -0x1.4000000000000p+0, -0x1.4000000000000p-2, -0x1.4000000000000p-1, -0x1.9000000000000p+0, -0x1.cf40000000000p+1, 0x0.0p+0, 0x1.cf54800000000p+6, 0x1.7766300000000p+10, 0x1.e949e41400000p+13, 0x1.2daeb8d000000p+17, 0x1.717a420aa8000p+20, 0x1.cab8823578000p+23, 0x1.23a0ed949d8d6p+27, 0x1.7dd7c62c1cb80p+30, 0x1.02361efb7e41cp+34, 0x1.694d8f62c5e7ap+37, 0x1.05b9c1e02e96cp+41, 0x1.88b185541093cp+44, 0x1.310d15b095367p+48, 0x1.ea91bd83aeb58p+51, 0x1.98152c3d2e6e3p+55, 0x1.5ef9745e29455p+59, 0x1.37e10b8b93ec9p+63, 0x1.1e227bd2377fbp+67, 0x1.0ed6f49fb1685p+71, 0x1.084c6e00bf2e2p+75, 0x1.09b538b81c3dfp+79, 0x1.13001a5d9d33fp+83, 0x1.24ce76f705039p+87, 0x1.4084f78692471p+91, 0x1.6878af90e4e33p+95, 0x1.a040824b78960p+99, 0x1.ed39edb2d9588p+103, 0x1.2bada94294239p+108, 0x1.753f972c37c2fp+112, 0x1.dc3827f062d65p+116, 0x1.370bc35c27ff0p+121, 0x1.9fce5888891b4p+125, 0x1.1c45a89d6d3eap+130, 0x1.8d627d3fcb9c9p+134, 0x1.1bd5a7a3023c8p+139, 0x1.9e28d2a3d728dp+143, 0x1.3483bd13dc52bp+148, 0x1.d51bf71eeb99dp+152, 0x1.6bdbda110b769p+157, \
    -0x1.8000000000000p+0, -0x1.8000000000000p-3, -0x1.3000000000000p-1, -0x1.9800000000000p+0, -0x1.3c80000000000p+2, -0x1.4d80000000000p+4, -0x1.199fa00000000p+7, -0x1.508e200000000p+10, -0x1.c7be8d1800000p+13, -0x1.3d76016800000p+17, -0x1.bda651ec11000p+20, -0x1.3b9087b310000p+24, -0x1.c54415fa45ab0p+27, -0x1.4bd17c6650480p+31, -0x1.f11b861fb315ep+34, -0x1.7e0c05a2d5912p+38, -0x1.2dcc5d31c9cdfp+42, -0x1.eaa823eb13648p+45, -0x1.9ab3357d4b2b9p+49, -0x1.6217fa453e682p+53, -0x1.3a77fbd9667ddp+57, -0x1.1fa29dc25cd20p+61, -0x1.0ee4275808cc6p+65, -0x1.069729214abdbp+69, -0x1.05e264bde728ap+73, -0x1.0c960fa7d2878p+77, -0x1.1b216264b508fp+81, -0x1.329e009f7edaep+85, -0x1.54f2d33ee550bp+89, -0x1.85144cf3f994ep+93, -0x1.c76dc3259dc60p+97, -0x1.1143f31c40cd3p+102, -0x1.5006c98cbcde2p+106, -0x1.a732da855a77ap+110, -0x1.10cf2d6efb8ffp+115, -0x1.67e51db4622f2p+119, -0x1.e5965b27a02f4p+123, -0x1.4ee5e349ea7c7p+128, -0x1.d80c2e3870834p+132, -0x1.53d186852a664p+137, -0x1.f38d9046385a3p+141, -0x1.76c423e1160b6p+146, -0x1.1ed8b2e8093abp+151, -0x1.bfd652e507711p+155, -0x1.646ac39fa3af7p+160, \
    -0x1.c000000000000p+0, 0x0.0p+0, -0x1.1800000000000p-1, -0x1.9600000000000p+0, -0x1.4580000000000p+2, -0x1.2788000000000p+4, -0x1.fb16400000000p+5, 0x0.0p+0, 0x1.2401b4e800000p+12, 0x1.6125868800000p+16, 0x1.4f53488c43800p+20, 0x1.25b708530c000p+24, 0x1.f2f07542dc11cp+27, 0x1.a45aa122f7a10p+31, 0x1.63ae3f19ad158p+35, 0x1.308e450ca5bf8p+39, 0x1.092ed13ec10a0p+43, 0x1.d71bb8f039b1ap+46, 0x1.abd1ffb5df9cap+50, 0x1.8dc0140dbdf36p+54, 0x1.7aee8562ade15p+58, 0x1.722017305df9ap+62, 0x1.72c381ab61991p+66, 0x1.7cee75396fde4p+70, 0x1.9166a43bc8b7ep+74, 0x1.b1bd20f96def8p+78, 0x1.e0820f3595830p+82, 0x1.10cd1ec897223p+87, 0x1.3d6318445b6c2p+91, 0x1.7a39dfcbb9c9cp+95, 0x1.cd8173cc3e158p+99, 0x1.202ff790c81aep+104, 0x1.703fa7fdaf962p+108, 0x1.e142c535c4edfp+112, 0x1.418330147ab3ap+117, 0x1.b7064fd5e3959p+121, 0x1.3237f3db0af62p+126, 0x1.b43d78b5bce61p+130, 0x1.3d37153e7605fp+135, 0x1.d6c8e11a16c63p+139, 0x1.6462a53de52e0p+144, 0x1.131fcfe8cb1afp+149, 0x1.b10c26238f5b7p+153, 0x1.5b53d30e633e8p+158, 0x1.1bd032fb79f0dp+163
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double inverse_asymptotic_dev[270] = {LAPLACE_BESSEL_INVERSE_ASYMPTOTIC};
#endif
inline constexpr double inverse_asymptotic_host[270] = {LAPLACE_BESSEL_INVERSE_ASYMPTOTIC};
LAPLACE_HD_FORCEINLINE double inverse_asymptotic(int n, int m) {
  KOKKOS_IF_ON_DEVICE((return inverse_asymptotic_dev[n * 45 + m];))
  KOKKOS_IF_ON_HOST((return inverse_asymptotic_host[n * 45 + m];))
}

// K^(n)(A(k)) = k^p sum_m inverse_series[n-1][m] k^(2m), p = 0 for odd n, 1 for even n
// clang-format off
#define LAPLACE_BESSEL_INVERSE_SERIES \
    0x1.0000000000000p+1, 0x1.8000000000000p-1, 0x1.2aaaaaaaaaaabp-4, -0x1.5555555555555p-11, -0x1.6c16c16c16c17p-13, 0x1.097b425ed097bp-17, 0x1.3193193193193p-22, -0x1.395723ab1e402p-25, 0x1.baf15553474c8p-32, 0x1.cfcaf0cce294cp-34, -0x1.6ef582a47ef1ap-38, -0x1.5e0d1d9b71555p-43, 0x1.9ce8245724ca8p-46, -0x1.7f258fb123f0dp-52, -0x1.23f9f0767bd25p-54, 0x1.fa08cfb68874bp-59, 0x1.84cce492a3f5ap-64, -0x1.0ed8d27f23792p-66, 0x1.36c3365a3839ep-72, 0x1.6d84fe77b7a14p-75, -0x1.5aade4d88ddb8p-79, -0x1.a020b23836dbep-85, 0x1.61eb625afbfcbp-87, -0x1.e3a763bb3b7e2p-93, \
    0x1.8000000000000p+1, 0x1.b555555555555p+0, 0x1.4800000000000p-2, 0x1.ddddddddddddep-7, -0x1.a84bda12f684cp-10, -0x1.2b2cb2cb2cb2dp-14, 0x1.861a8c536fe1bp-17, 0x1.1afee1ddae9b1p-24, -0x1.1e7b85df0d289p-24, 0x1.1cfba04302682p-29, 0x1.1d2ee662c6889p-32, -0x1.61b065972f52ap-36, -0x1.ef2f50189cee3p-42, 0x1.edd9854a84ae9p-44, -0x1.54690360ca785p-49, -0x1.bf3f39789dbbep-52, 0x1.d78ae87ef7affp-56, 0x1.74e0844d34227p-61, -0x1.375d2ee80f9c1p-63, 0x1.8ad9074657499p-69, 0x1.069c3dbf75e85p-71, -0x1.0bcea26602d2fp-75, -0x1.8f44ae82d7f3ep-81, 0x1.4e76338032004p-83, \
    0x1.8000000000000p+2, 0x1.9000000000000p+3, 0x1.d100000000000p+2, 0x1.c6eeeeeeeeeefp+0, 0x1.4866666666666p-3, -0x1.c81ba1ba1ba1cp-8, -0x1.b0ec98c98c98dp-10, 0x1.0c0432289635fp-14, 0x1.c8b9393ddd716p-17, -0x1.afa63d81b6b59p-21, -0x1.60522d5bf22efp-24, 0x1.1cb764675e3c1p-27, 0x1.2e32ead8c673cp-32, -0x1.15bb868b1a39dp-34, 0x1.755e0095d8452p-41, 0x1.87896cb916f04p-42, -0x1.38c1089a58faep-46, -0x1.6352c24dac675p-50, 0x1.48501a9c0927ep-53, 0x1.a5f5a13135735p-61, -0x1.be020d9ef0a98p-61, 0x1.fef5c88722d63p-66, 0x1.8958cab8d924fp-69, -0x1.1d3ceec55c389p-72, \
    0x1.9000000000000p+5, 0x1.3380000000000p+6, 0x1.678eeeeeeeeefp+5, 0x1.954aaaaaaaaabp+3, 0x1.93bebaebaebafp+0, -0x1.4a6de233788cep-7, -0x1.517a81ca7b785p-6, -0x1.bccf679073b7fp-12, 0x1.f07c5c1d54e1ep-13, 0x1.0f4b49e73c180p-18, -0x1.46bd39bec89f5p-19, 0x1.fa07079afc872p-28, 0x1.6dace28a76d7fp-26, -0x1.684207dd396e0p-31, -0x1.43cc937ae12ddp-33, 0x1.5954952ffc73bp-37, 0x1.8fb7fdab0fb2ap-41, -0x1.a4acf5840b014p-44, -0x1.56e51408b7878p-50, 0x1.72f3e7591f586p-51, -0x1.4825b5b67bdebp-56, -0x1.d42817a1f1eb4p-59, 0x1.111bcb61cabfep-62, 0x1.4db38aecd884bp-67, \
    0x1.9000000000000p+6, 0x1.f2c0000000000p+8, 0x1.3908000000000p+9, 0x1.6aa4555555555p+8, 0x1.bc6f649249249p+6, 0x1.0b3d3228a28a3p+4, 0x1.491100417f5a5p-2, -0x1.f49b1256b954dp-3, -0x1.250009f63dfd5p-6, 0x1.8d4c437630407p-9, 0x1.3a0c0b8fa6ac0p-12, -0x1.326a6952a6e84p-15, -0x1.c780b65e26d31p-19, 0x1.d4f96afa5380ep-22, 0x1.d91cb3e859b8fp-26, -0x1.4b3a0898daadfp-28, -0x1.1f0cd3a7e1d7fp-33, 0x1.921a33232cce1p-35, -0x1.6aab8c005f766p-42, -0x1.8e69d7b7ee3e1p-42, 0x1.07e89151a2019p-46, 0x1.2d2f466b6102cp-49, -0x1.a299d194ecf31p-53, -0x1.18bb4b9c50d5ap-57, \
    0x1.f2c0000000000p+10, 0x1.67ca000000000p+12, 0x1.89ea000000000p+12, 0x1.c0f29a4924925p+11, 0x1.216371638e38ep+10, 0x1.87e6d08c02874p+7, 0x1.d297449732b1cp+2, -0x1.8d29845644a07p+1, -0x1.840c98bf56564p-2, 0x1.1f777ac37d1d7p-5, 0x1.f80622eaa2fb9p-8, -0x1.6afeabf7708f0p-12, -0x1.dc5c74a6df074p-14, 0x1.12a6b2cad6f8ep-18, 0x1.72bdc02d17d92p-20, -0x1.05e5d60418b6fp-24, -0x1.e52f2b35df2cap-27, 0x1.f12c7c9c0dcf1p-31, 0x1.03274a6fbc33ap-33, -0x1.9641d0233a1c9p-37, -0x1.9a86b4ad947f5p-41, 0x1.12887fc658685p-43, 0x1.310f8c98d43a1p-49, 0x1.7fbc921b3372bp-27
// clang-format on
#if defined(__CUDACC__)
LAPLACE_DEVICE_TABLE double inverse_series_dev[144] = {LAPLACE_BESSEL_INVERSE_SERIES};
#endif
inline constexpr double inverse_series_host[144] = {LAPLACE_BESSEL_INVERSE_SERIES};
LAPLACE_HD_FORCEINLINE double inverse_series(int n, int m) {
  KOKKOS_IF_ON_DEVICE((return inverse_series_dev[n * 24 + m];))
  KOKKOS_IF_ON_HOST((return inverse_series_host[n * 24 + m];))
}

// The switch points (the defaults are the values chosen by measurement; the
// bench varies them).
struct params {
  double small = 0.25;    // power series below
  double large = 30.0;    // asymptotic series from here
  int extra = 20;         // continued fraction truncated at floor(k) + extra
};

LAPLACE_HD_FORCEINLINE double nan() { return laplace::math::detail::quiet_nan(); }

// d^M/dk^M of the power series, only that order
template <int M>
LAPLACE_HD_FORCEINLINE double ratio_small(double k) {
  constexpr int nmin = M / 2;  // the first power 2n + 1 >= M
  const double k2 = k * k;
  double kp = (2 * nmin + 1 - M) ? k : 1.0;  // k^(2n + 1 - M)
  double s = 0.0;
  for (int n = nmin; n < 26; ++n) {
    const int p = 2 * n + 1;
    double fac = 1.0;
    for (int j = 0; j < M; ++j) fac *= double(p - j);
    const double t = ratio_series(n) * fac * kp;
    s += t;
    if (Kokkos::fabs(t) <= tiny_rel * Kokkos::fabs(s)) break;
    kp *= k2;
  }
  return s;
}

// u = 1/k rounded, and delta with 1/k = u (1 + delta) to first order: k u = p + e
// exactly, 1 - p is exact (p is within a factor 2 of 1), delta = (1 - p) - e
LAPLACE_HD_FORCEINLINE double recip_delta(double k, double& u) {
  u = 1.0 / k;
  const dd pe = ddm::two_prod(k, u);
  return (1.0 - pe.hi) - pe.lo;
}

// S_M(u) = sum_(n >= 1) q_n n (n+1) ... (n+M-1) u^n, summed while the terms
// decrease and exceed the last bit, corrected to first order for the rounding
// of u: a term in u^n of the result u^M S_M moves by (n + M) delta times itself.
// For M = 0 the first term q_0 = 1 is left out (1 - A = -S_0 without
// cancellation).
//
// Near k = 30 some fifty terms are above the last bit (they decrease by a
// factor of about n / (2k)), so the sum is compensated (Neumaier): the
// rounding error of each addition is kept in e and added at the end.
template <int M>
LAPLACE_HD_FORCEINLINE double ratio_large_s(double u, double delta) {
  double s = 0.0, e = 0.0, c = 0.0, un = 1.0, prev = 1e300;
  for (int n = 1; n < 60; ++n) {
    un *= u;
    double fac = 1.0;
    for (int j = 0; j < M; ++j) fac *= double(n + j);
    const double t = ratio_asymptotic(n) * fac * un;
    const double at = Kokkos::fabs(t);
    if (n > M + 1 && at > prev) break;  // the divergent tail
    const dd st = ddm::two_sum(s, t);
    s = st.hi, e += st.lo;
    c += double(n + M) * t;
    if (at <= tiny_rel * Kokkos::fabs(s)) break;
    prev = at;
  }
  return s + (e + delta * c);
}

template <int M>
LAPLACE_HD_FORCEINLINE double ratio_large(double k) {
  double u;
  const double delta = recip_delta(k, u);
  if constexpr (M == 0) {
    return 1.0 + ratio_large_s<0>(u, delta);
  } else {
    // u^M in double-double: its rounding would add up to M/2 ulp
    dd um(u);
    for (int j = 1; j < M; ++j) um = um * u;
    const double s = (um * ratio_large_s<M>(u, delta)).hi;
    return (M % 2) ? -s : s;
  }
}

// A by the continued fraction, backwards in double
LAPLACE_HD_FORCEINLINE double ratio_cf(double k, int extra) {
  double r = 0.0;
  for (int j = int(k) + extra; j >= 1; --j) r = k / (2.0 * j + k * r);
  return r;
}

// The same fraction forwards in double-double: with x = k/2 it reads
// x / (1 + x^2 / (2 + x^2 / (3 + ...))), so P_j = j P_(j-1) + x^2 P_(j-2) and
// likewise Q_j, with every term positive (stable), and one division at the end.
LAPLACE_HD_FORCEINLINE dd ratio_cf_dd(double k, int extra) {
  const double x = 0.5 * k;
  const dd x2 = ddm::two_prod(x, x);
  dd p0(0.0), p1(x), q0(1.0), q1(1.0);
  const int n = int(k) + extra;
  for (int j = 2; j <= n; ++j) {
    const dd p2 = p1 * double(j) + x2 * p0;
    const dd q2 = q1 * double(j) + x2 * q0;
    p0 = p1, p1 = p2, q0 = q1, q1 = q2;
  }
  return p1 / q1;
}

// A^(0..N) at k from A by the Riccati identity, in double-double
template <int N>
LAPLACE_HD_FORCEINLINE void riccati(const dd& A, double k, dd (&d)[N + 1]) {
  const dd ik = dd(1.0) / dd(k);
  dd inv[N > 0 ? N : 1];  // (1/k)^(m), m = 0 .. N-1
  {
    dd p = ik;
    double f = 1.0;
    for (int m = 0; m < N; ++m) {
      inv[m] = (m % 2) ? -(p * f) : p * f;
      p = p * ik;
      f *= double(m + 1);
    }
  }
  d[0] = A;
  for (int n = 0; n < N; ++n) {
    dd s(0.0);
    double c = 1.0;  // C(n, j)
    for (int j = 0; j <= n; ++j) {
      s = s + (d[j] * (inv[n - j] + d[n - j])) * c;
      c = c * double(n - j) / double(j + 1);
    }
    d[n + 1] = (n == 0) ? 1.0 - s : -s;
  }
}

template <int M>
LAPLACE_HD double ratio_dm(double k, const params& p) {
  if (!(k >= 0.0)) return nan();
  if (k < p.small) return ratio_small<M>(k);
  if (k < p.large) {
    if constexpr (M == 0) {
      return ratio_cf(k, p.extra);
    } else {
      dd d[M + 1];
      riccati<M>(ratio_cf_dd(k, p.extra), k, d);
      return d[M].hi;
    }
  }
  return ratio_large<M>(k);
}

// k = A^-1(rho) by Newton's method. In the continued fraction's range the
// iterates use the fraction in double (and A' from the Riccati identity in
// double) until the step falls below 1e-9 k, then the fraction in
// double-double for the last steps, which fix the last bits.
LAPLACE_HD double ratio_inv(double rho, const params& p) {
  if (!(rho >= 0.0 && rho <= 1.0)) return nan();
  if (rho == 0.0) return 0.0;
  if (rho == 1.0) return laplace::math::detail::pos_inf();
  const double r = rho;
  double g = r < 0.53 ? 2.0 * r + r * r * r + 5.0 * (r * r) * (r * r) * r / 6.0
                      : (r < 0.85 ? -0.4 + 1.39 * r + 0.43 / (1.0 - r) : 1.0 / (r * r * r - 4.0 * r * r + 3.0 * r));
  const double lo = 2.0 * r;  // A(k) < k/2 puts the root above 2 rho
  if (!(g >= lo)) g = lo;
  const bool upper = r >= 0.5;
  const double c = 1.0 - r;  // exact for r >= 1/2
  bool fine = false;
  for (int it = 0; it < 100; ++it) {
    const double kk = g;
    double res, d1;
    if (kk < p.small) {
      res = ratio_small<0>(kk) - r;
      d1 = ratio_small<1>(kk);
    } else if (kk < p.large) {
      if (fine) {
        dd d[2];
        riccati<1>(ratio_cf_dd(kk, p.extra), kk, d);
        res = upper ? (c - (1.0 - d[0])).hi : (d[0] - r).hi;
        d1 = d[1].hi;
      } else {
        const double A = ratio_cf(kk, p.extra);
        res = upper ? c - (1.0 - A) : A - r;
        d1 = 1.0 - A / kk - A * A;
      }
    } else {
      double u;
      const double delta = recip_delta(kk, u);
      res = upper ? c + ratio_large_s<0>(u, delta) : ratio_large<0>(kk) - r;
      d1 = ratio_large<1>(kk);
    }
    double kn = kk - res / d1;
    if (!(kn >= lo) || !(kn < 1e308)) kn = lo;
    g = kn;
    const double step = Kokkos::fabs(kn - kk);
    if (fine || !(kk >= p.small && kk < p.large)) {
      if (step <= 4.0 * eps * kn || (it > 0 && kn <= kk)) break;
    }
    if (step <= 1e-9 * kn) fine = true;
  }
  return g;
}

// The polynomial of the inverse function rule in r_j = A^(j)/A' (T is double
// or double-double)
template <int N, class T>
LAPLACE_HD_FORCEINLINE T inverse_poly(const T (&r)[N + 1]) {
  if constexpr (N == 1) {
    return T(1.0);
  } else if constexpr (N == 2) {
    return -r[2];
  } else if constexpr (N == 3) {
    return r[2] * r[2] * 3.0 - r[3];
  } else if constexpr (N == 4) {
    return r[2] * (r[3] * 10.0 - r[2] * r[2] * 15.0) - r[4];
  } else if constexpr (N == 5) {
    const T r2 = r[2], r22 = r2 * r2;
    return r22 * r22 * 105.0 - r22 * r[3] * 105.0 + r2 * r[4] * 15.0 + r[3] * r[3] * 10.0 - r[5];
  } else {
    const T r2 = r[2], r22 = r2 * r2, r3 = r[3];
    return r22 * r22 * r2 * (-945.0) + r22 * r2 * r3 * 1260.0 - r22 * r[4] * 210.0 - r2 * r3 * r3 * 280.0 +
           r2 * r[5] * 21.0 + r3 * r[4] * 35.0 - r[6];
  }
}

// K^(N)(A(k)) for k < small: k^p sum_m g_m k^(2m)
template <int N>
LAPLACE_HD_FORCEINLINE double inverse_small(double k) {
  const double k2 = k * k;
  double s = 0.0, km = 1.0;
  for (int m = 0; m < 24; ++m) {
    const double t = inverse_series(N - 1, m) * km;
    s += t;
    if (Kokkos::fabs(t) <= tiny_rel * Kokkos::fabs(s)) break;
    km *= k2;
  }
  return (N % 2) ? s : s * k;
}

// K^(N)(A(k)) for k >= large: N! 2^N k^(N+1) (1 + sum_m e_m u^m), with
// k^(N+1) in double-double (below 1e40, where the split of the exact product
// cannot overflow)
template <int N>
LAPLACE_HD_FORCEINLINE double inverse_large(double k) {
  const double u = 1.0 / k;
  double s = 0.0, um = 1.0, prev = 1e300;
  for (int m = 0; m < 45; ++m) {
    um *= u;
    const double t = inverse_asymptotic(N - 1, m) * um;
    const double at = Kokkos::fabs(t);
    if (at == 0.0) continue;  // the tables have exact zeros
    if (m > 4 && at > prev) break;  // the divergent tail
    s += t;
    if (at <= tiny_rel) break;
    prev = at;
  }
  double lead = 1.0;  // N! 2^N, exact
  for (int j = 1; j <= N; ++j) lead *= 2.0 * j;
  const double v = lead * (1.0 + s);
  if (k < 1e40) {
    dd kp(k);
    for (int j = 0; j < N; ++j) kp = kp * k;
    return (kp * v).hi;
  }
  double kp = k;
  for (int j = 0; j < N; ++j) kp *= k;
  return kp * v;
}

template <int N>
LAPLACE_HD double ratio_inv_dn(double k, const params& p) {
  if (!(k >= 0.0)) return nan();
  if (k < p.small) return inverse_small<N>(k);
  if (k >= p.large) return inverse_large<N>(k);
  // w = 1/A' and r_j = A^(j)/A' in double-double; the powers of w applied one
  // at a time
  dd d[N + 1];
  riccati<N>(ratio_cf_dd(k, p.extra), k, d);
  const dd w = dd(1.0) / d[1];
  dd r[N + 1];
  for (int j = 2; j <= N; ++j) r[j] = d[j] * w;
  dd q = inverse_poly<N>(r);
  for (int j = 0; j < N; ++j) q = q * w;
  return q.hi;
}

// log I0(k) - k (scaled = true) or log I0(k), from Boost's rational
// approximations of I0 for 53 bits (boost/math/special_functions/detail/
// bessel_i0.hpp, Boost Software License 1.0; the forms are Holoborodko's, the
// coefficients Boost's) rewritten in log form: below 7.75, I0 = 1 + a P(a)
// with a = k^2/4, so log I0 = log1p(a P(a)); above, I0 = e^k P(1/k) / sqrt(k),
// so log I0 - k = log P(1/k) - log(k)/2, with no exponential and no overflow.
LAPLACE_HD double log_i0(double k, bool scaled) {
  if (!(k >= 0.0)) return nan();
  if (k < 7.75) {
    constexpr double P[] = {1.00000000000000000e+00, 2.49999999999999909e-01, 2.77777777777782257e-02,
                            1.73611111111023792e-03, 6.94444444453352521e-05, 1.92901234513219920e-06,
                            3.93675991102510739e-08, 6.15118672704439289e-10, 7.59407002058973446e-12,
                            7.59389793369836367e-14, 6.27767773636292611e-16, 4.34709704153272287e-18,
                            2.63417742690109154e-20, 1.13943037744822825e-22, 9.07926920085624812e-25};
    const double a = 0.25 * k * k;
    double s = P[14];
    for (int i = 13; i >= 0; --i) s = s * a + P[i];
    const double l = Kokkos::log1p(a * s);
    return scaled ? l - k : l;
  }
  if (!(k < 1e308)) return scaled ? -k : k;  // +inf: log I0 = +inf, log I0 - k = -inf
  const double u = 1.0 / k;
  double s;
  if (k < 500.0) {
    constexpr double P[] = {3.98942280401425088e-01,  4.98677850604961985e-02,  2.80506233928312623e-02,
                            2.92211225166047873e-02,  4.44207299493659561e-02,  1.30970574605856719e-01,
                            -3.35052280231727022e+00, 2.33025711583514727e+02,  -1.13366350697172355e+04,
                            4.24057674317867331e+05,  -1.23157028595698731e+07, 2.80231938155267516e+08,
                            -5.01883999713777929e+09, 7.08029243015109113e+10,  -7.84261082124811106e+11,
                            6.76825737854096565e+12,  -4.49034849696138065e+13, 2.24155239966958995e+14,
                            -8.13426467865659318e+14, 2.02391097391687777e+15,  -3.08675715295370878e+15,
                            2.17587543863819074e+15};
    s = P[21];
    for (int i = 20; i >= 0; --i) s = s * u + P[i];
  } else {
    constexpr double P[] = {3.98942280401432905e-01, 4.98677850491434560e-02, 2.80506308916506102e-02,
                            2.92179096853915176e-02, 4.53371208762579442e-02};
    s = P[4];
    for (int i = 3; i >= 0; --i) s = s * u + P[i];
  }
  const double l = laplace::math::log(s) - 0.5 * laplace::math::log(k);
  return scaled ? l : k + l;
}


// R_j = I_j(k)/I_0(k), j = 1..m, into out[0..m-1], by Miller's backward
// recurrence y_(j-1) = (2j/k) y_j + y_(j+1) from y_(n0+1) = 0 and a tiny
// y_(n0), rescaled against overflow, with R_j = y_j / y_0. The start is
// n0 = m + max(30, ceil(sqrt(40 m))) + ceil(7 sqrt(k)): the recurrence damps
// its starting value by about (I_n0 / I_m)^2, and for large k
// I_j/I_0 ~ exp(-j^2 / (2k)), so 7 sqrt(k) above m damps it by e^-49 whatever
// m is (statmodels7 added ceil(k), which costs O(k) instead of O(sqrt k)).
LAPLACE_HD void ratios(double k, int m, double* out) {
  if (m < 1) return;
  if (!(k >= 0.0)) {
    for (int i = 0; i < m; ++i) out[i] = nan();
    return;
  }
  if (k == 0.0 || !(k < 1e308)) {  // I_j(0) = 0; I_j/I_0 -> 1 as k -> inf
    for (int i = 0; i < m; ++i) out[i] = k == 0.0 ? 0.0 : 1.0;
    return;
  }
  if (k < 1e-8) {
    // R_j = (k/2)^j / j! (1 + O(k^2)), the correction below k^2/4 < 1e-16
    // relative; here the factor 2j/k of the recurrence would overflow y in
    // one step
    double r = 0.5 * k;
    out[0] = r;
    for (int j = 2; j <= m; ++j) out[j - 1] = r = r * (0.5 * k) / j;
    return;
  }
  const int s = int(Kokkos::ceil(Kokkos::sqrt(40.0 * m)));
  const int n0 = m + (s > 30 ? s : 30) + int(Kokkos::ceil(7.0 * Kokkos::sqrt(k)));
  double y1 = 0.0, y = 1e-280;  // y_(j+1), y_j
  for (int j = n0; j >= 1; --j) {
    const double ym = (2.0 * j) / k * y + y1;  // y_(j-1)
    y1 = y, y = ym;
    if (j - 1 >= 1 && j - 1 <= m) out[j - 2] = y;
    if (y > 1e250) {
      y *= 1e-250, y1 *= 1e-250;
      for (int i = (j - 1 > 1 ? j - 1 : 1); i <= m; ++i) out[i - 1] *= 1e-250;
    }
  }
  for (int i = 0; i < m; ++i) out[i] /= y;
}

}  // namespace detail_bessel

/**
 * \ingroup bessel_i
 * \brief Logarithm of the modified Bessel function of the first kind of order 0, \f$\log I_0(k)\f$.
 *
 * Computes \f$\log I_0(k)\f$, where \f$I_0(k)\f$ is the modified Bessel function of the first kind of order 0:
 * \f[
 * I_0(k) = \sum_{m=0}^\infty \frac{1}{(m!)^2} \left(\frac{k}{2}\right)^{2m}.
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$\log I_0(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - For \f$k \ge 10^{308}\f$: returns \f$k\f$, which is finite for finite \f$k\f$; only \f$k = +\infty\f$ gives \f$+\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.31 ulp (gcc) / 2.15 ulp (CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, log_bessel_i0_scaled(), bessel_i_ratio()
 */
LAPLACE_HD double log_bessel_i0(double k) { return detail_bessel::log_i0(k, false); }

/**
 * \ingroup bessel_i
 * \brief Scaled logarithm of the modified Bessel function of the first kind of order 0, \f$\log I_0(k) - k\f$.
 *
 * Computes \f$\log(e^{-k} I_0(k)) = \log I_0(k) - k\f$. Remains finite across all finite \f$k \ge 0\f$ without intermediate overflow (returns \f$-\infty\f$ only at \f$k = +\infty\f$).
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$\log I_0(k) - k\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - For \f$k \ge 10^{308}\f$: returns \f$-k\f$, which is finite for finite \f$k\f$; only \f$k = +\infty\f$ gives \f$-\infty\f$.
 *
 * \par Measured accuracy
 * Maximum error 3.0 ulp (gcc) / 3.0 ulp (CUDA); test bound 4.5 ulp.
 *
 * \see bessel_i, log_bessel_i0(), bessel_i_ratio()
 */
LAPLACE_HD double log_bessel_i0_scaled(double k) { return detail_bessel::log_i0(k, true); }

/**
 * \ingroup bessel_i
 * \brief Ratio of modified Bessel functions of the first kind of orders 1 and 0, \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the ratio:
 * \f[
 * A(k) = \frac{I_1(k)}{I_0(k)} = \frac{\mathrm{d}}{\mathrm{d}k}\log I_0(k).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A(k) \in [0, 1)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$1.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 3.30 ulp (gcc) / 3.48 ulp (CUDA); test bound 5.0 ulp.
 *
 * \see bessel_i, log_bessel_i0(), bessel_i_ratio_d1(), bessel_i_ratio_inv()
 */
LAPLACE_HD double bessel_i_ratio(double k) { return detail_bessel::ratio_dm<0>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief First derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the first derivative:
 * \f[
 * A'(k) = \frac{\mathrm{d}}{\mathrm{d}k}A(k) = \frac{\mathrm{d}^2}{\mathrm{d}k^2}\log I_0(k) = 1 - \frac{A(k)}{k} - A(k)^2.
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A'(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.5\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.04 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio(), bessel_i_ratio_d2()
 */
LAPLACE_HD double bessel_i_ratio_d1(double k) { return detail_bessel::ratio_dm<1>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Second derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the second derivative:
 * \f[
 * A''(k) = \frac{\mathrm{d}^2}{\mathrm{d}k^2}A(k) = \frac{\mathrm{d}^3}{\mathrm{d}k^3}\log I_0(k) = \frac{A(k)}{k^2} - \frac{A'(k)}{k} - 2 A(k) A'(k).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A''(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.23 ulp (gcc and CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_d1(), bessel_i_ratio_d3()
 */
LAPLACE_HD double bessel_i_ratio_d2(double k) { return detail_bessel::ratio_dm<2>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Third derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the third derivative:
 * \f[
 * A'''(k) = \frac{\mathrm{d}^3}{\mathrm{d}k^3}A(k) = \frac{\mathrm{d}^4}{\mathrm{d}k^4}\log I_0(k) = -\frac{2A(k)}{k^3} + \frac{2A'(k)}{k^2} - \frac{A''(k)}{k} - 2 (A'(k))^2 - 2 A(k) A''(k).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A'''(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$-0.375\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.00 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio_d2(), bessel_i_ratio_d4()
 */
LAPLACE_HD double bessel_i_ratio_d3(double k) { return detail_bessel::ratio_dm<3>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Fourth derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * A^{(4)}(k) = \frac{\mathrm{d}^4}{\mathrm{d}k^4}A(k) = \frac{\mathrm{d}^5}{\mathrm{d}k^5}\log I_0(k) = \frac{6A(k)}{k^4} - \frac{6A'(k)}{k^3} + \frac{3A''(k)}{k^2} - \frac{A'''(k)}{k} - 6 A'(k) A''(k) - 2 A(k) A'''(k).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A^{(4)}(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.42 ulp (gcc and CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_d3(), bessel_i_ratio_d5()
 */
LAPLACE_HD double bessel_i_ratio_d4(double k) { return detail_bessel::ratio_dm<4>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Fifth derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * A^{(5)}(k) = \frac{\mathrm{d}^5}{\mathrm{d}k^5}A(k) = \frac{\mathrm{d}^6}{\mathrm{d}k^6}\log I_0(k) = -\sum_{j=0}^4 \binom{4}{j} A^{(j)}(k) \left( (-1)^{4-j}\frac{(4-j)!}{k^{5-j}} + A^{(4-j)}(k) \right).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A^{(5)}(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.08 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio_d4(), bessel_i_ratio_d6()
 */
LAPLACE_HD double bessel_i_ratio_d5(double k) { return detail_bessel::ratio_dm<5>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Sixth derivative of the ratio \f$A(k) = I_1(k) / I_0(k)\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * A^{(6)}(k) = \frac{\mathrm{d}^6}{\mathrm{d}k^6}A(k) = \frac{\mathrm{d}^7}{\mathrm{d}k^7}\log I_0(k) = -\sum_{j=0}^5 \binom{5}{j} A^{(j)}(k) \left( (-1)^{5-j}\frac{(5-j)!}{k^{6-j}} + A^{(5-j)}(k) \right).
 * \f]
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \return The value of \f$A^{(6)}(k)\f$.
 *
 * \par Domain and special values
 * - \f$k \ge 0\f$: standard domain.
 * - \f$k = 0\f$: returns \f$0.0\f$.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: returns quiet NaN.
 * - \f$k \to +\infty\f$: approaches \f$0.0\f$.
 *
 * \par Measured accuracy
 * Maximum error 2.42 ulp (gcc and CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_d5()
 */
LAPLACE_HD double bessel_i_ratio_d6(double k) { return detail_bessel::ratio_dm<6>(k, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Inverse of the Bessel function ratio, \f$K(\rho) = A^{-1}(\rho)\f$.
 *
 * Computes the inverse function \f$K(\rho)\f$ satisfying \f$A(K(\rho)) = \rho\f$ for \f$\rho \in [0, 1)\f$ via Newton's method.
 *
 * \param rho Ratio value \f$\rho \in [0, 1)\f$.
 * \return The value \f$k = K(\rho) \in [0, \infty)\f$.
 *
 * \par Domain and special values
 * - \f$\rho \in [0, 1)\f$: standard domain.
 * - \f$\rho = 0\f$: returns \f$0.0\f$.
 * - \f$\rho = 1\f$: returns \f$+\infty\f$.
 * - \f$\rho < 0\f$, \f$\rho > 1\f$, or \f$\rho\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 1.80 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio(), bessel_i_ratio_inv_d1()
 */
LAPLACE_HD double bessel_i_ratio_inv(double rho) { return detail_bessel::ratio_inv(rho, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief First derivative of the inverse ratio, \f$K'(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the first derivative:
 * \f[
 * K'(\rho) = \frac{\mathrm{d}K}{\mathrm{d}\rho} = w
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K'(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa = 0\f$: returns \f$2.0\f$.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 1.99 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv(), bessel_i_ratio_inv_d2()
 */
LAPLACE_HD double bessel_i_ratio_inv_d1(double kappa) { return detail_bessel::ratio_inv_dn<1>(kappa, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Second derivative of the inverse ratio, \f$K''(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the second derivative:
 * \f[
 * K''(\rho) = \frac{\mathrm{d}^2K}{\mathrm{d}\rho^2} = -r_2 w^2
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$ and \f$r_2 = \frac{A''(\kappa)}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K''(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa = 0\f$: returns \f$0.0\f$.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 2.14 ulp (gcc and CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv_d1(), bessel_i_ratio_inv_d3()
 */
LAPLACE_HD double bessel_i_ratio_inv_d2(double kappa) { return detail_bessel::ratio_inv_dn<2>(kappa, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Third derivative of the inverse ratio, \f$K'''(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the third derivative:
 * \f[
 * K'''(\rho) = \frac{\mathrm{d}^3K}{\mathrm{d}\rho^3} = (3 r_2^2 - r_3) w^3
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$ and \f$r_j = \frac{A^{(j)}(\kappa)}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K'''(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 2.31 ulp (gcc and CUDA); test bound 3.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv_d2(), bessel_i_ratio_inv_d4()
 */
LAPLACE_HD double bessel_i_ratio_inv_d3(double kappa) { return detail_bessel::ratio_inv_dn<3>(kappa, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Fourth derivative of the inverse ratio, \f$K^{(4)}(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the fourth derivative:
 * \f[
 * K^{(4)}(\rho) = \frac{\mathrm{d}^4K}{\mathrm{d}\rho^4} = (-15 r_2^3 + 10 r_2 r_3 - r_4) w^4
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$ and \f$r_j = \frac{A^{(j)}(\kappa)}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K^{(4)}(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa = 0\f$: returns \f$0.0\f$.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 1.96 ulp (gcc and CUDA); test bound 3.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv_d3(), bessel_i_ratio_inv_d5()
 */
LAPLACE_HD double bessel_i_ratio_inv_d4(double kappa) { return detail_bessel::ratio_inv_dn<4>(kappa, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Fifth derivative of the inverse ratio, \f$K^{(5)}(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the fifth derivative:
 * \f[
 * K^{(5)}(\rho) = \frac{\mathrm{d}^5K}{\mathrm{d}\rho^5} = (105 r_2^4 - 105 r_2^2 r_3 + 15 r_2 r_4 + 10 r_3^2 - r_5) w^5
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$ and \f$r_j = \frac{A^{(j)}(\kappa)}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K^{(5)}(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 2.54 ulp (gcc and CUDA); test bound 4.0 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv_d4(), bessel_i_ratio_inv_d6()
 */
LAPLACE_HD double bessel_i_ratio_inv_d5(double kappa) { return detail_bessel::ratio_inv_dn<5>(kappa, detail_bessel::params{}); }

/**
 * \ingroup bessel_i
 * \brief Sixth derivative of the inverse ratio, \f$K^{(6)}(\rho)\f$, evaluated at \f$\rho = A(\kappa)\f$.
 *
 * Computes the sixth derivative:
 * \f[
 * K^{(6)}(\rho) = \frac{\mathrm{d}^6K}{\mathrm{d}\rho^6} = (-945 r_2^5 + 1260 r_2^3 r_3 - 210 r_2^2 r_4 - 280 r_2 r_3^2 + 21 r_2 r_5 + 35 r_3 r_4 - r_6) w^6
 * \f]
 * where \f$w = \frac{1}{A'(\kappa)}\f$ and \f$r_j = \frac{A^{(j)}(\kappa)}{A'(\kappa)}\f$. Evaluated at \f$\kappa\f$ rather than \f$\rho = A(\kappa)\f$ to preserve precision near \f$\rho = 1\f$.
 *
 * \param kappa Point \f$\kappa \in [0, \infty)\f$ corresponding to \f$\rho = A(\kappa)\f$.
 * \return The value of \f$K^{(6)}(\rho)\f$ at \f$\rho = A(\kappa)\f$.
 *
 * \par Domain and special values
 * - \f$\kappa \ge 0\f$: standard domain.
 * - \f$\kappa = 0\f$: returns \f$0.0\f$.
 * - \f$\kappa < 0\f$ or \f$\kappa\f$ is NaN: returns quiet NaN.
 *
 * \par Measured accuracy
 * Maximum error 2.92 ulp (gcc and CUDA); test bound 4.5 ulp.
 *
 * \see bessel_i, bessel_i_ratio_inv_d5()
 */
LAPLACE_HD double bessel_i_ratio_inv_d6(double kappa) { return detail_bessel::ratio_inv_dn<6>(kappa, detail_bessel::params{}); }

// I_j(k)/I_0(k) for j = 1..m into out[0..m-1] (out has room for m values):
// the coefficients of the series of the von Mises distribution function
/**
 * \ingroup bessel_i
 * \brief Ratios of modified Bessel functions of the first kind of orders \f$j = 1, \dots, m\f$ to order 0, \f$R_j(k) = I_j(k) / I_0(k)\f$.
 *
 * Computes the sequence of \f$m\f$ ratios:
 * \f[
 * R_j(k) = \frac{I_j(k)}{I_0(k)}, \quad j = 1, \dots, m,
 * \f]
 * into \f$\mathrm{out}[0 \dots m-1]\f$ using Miller's backward recurrence.
 *
 * \param k Non-negative argument \f$k \in [0, \infty)\f$.
 * \param m Number of ratios to compute, \f$m \ge 1\f$.
 * \param out Output array with storage for at least \f$m\f$ double-precision values.
 *
 * \par Domain and special values
 * - \f$k \ge 0, m \ge 1\f$: standard domain.
 * - \f$m < 1\f$: returns immediately without writing.
 * - \f$k = 0\f$: writes \f$0.0\f$ to all \f$m\f$ entries.
 * - \f$k < 0\f$ or \f$k\f$ is NaN: writes quiet NaN to all \f$m\f$ entries.
 * - \f$k \ge 10^{308}\f$ (\f$k \to +\infty\f$): writes \f$1.0\f$ to all \f$m\f$ entries.
 *
 * \par Measured accuracy
 * Maximum error 13.8 ulp (4.7 ulp for \f$R_j > 10^{-3}\f$), 2.3 ulp1 on gcc and CUDA; test bounds 23.0 ulp and 3.0 ulp1.
 *
 * \see bessel_i, bessel_i_ratio()
 */
LAPLACE_HD void bessel_i_ratios(double k, int m, double* out) { detail_bessel::ratios(k, m, out); }

}  // namespace own

using own::bessel_i_ratio;
using own::bessel_i_ratio_d1;
using own::bessel_i_ratio_d2;
using own::bessel_i_ratio_d3;
using own::bessel_i_ratio_d4;
using own::bessel_i_ratio_d5;
using own::bessel_i_ratio_d6;
using own::bessel_i_ratio_inv;
using own::bessel_i_ratio_inv_d1;
using own::bessel_i_ratio_inv_d2;
using own::bessel_i_ratio_inv_d3;
using own::bessel_i_ratio_inv_d4;
using own::bessel_i_ratio_inv_d5;
using own::bessel_i_ratio_inv_d6;
using own::log_bessel_i0;
using own::bessel_i_ratios;
using own::log_bessel_i0_scaled;

}  // namespace laplace::math
