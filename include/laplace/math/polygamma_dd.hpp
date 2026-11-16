// Polygamma functions of order n = 0..13 in double-double, for x > 0, with
// their differences at a shift and their remainders after the leading term.
//
//   polygamma_dd(n, x)                       psi_n(x)
//   polygamma_difference_dd(n, x, h)         G_n(x, h) = psi_n(x + h) - psi_n(x), h > 0
//   polygamma_monomial_remainder_dd(n, x)    r_n(x) = psi_n(x) - mu_n(x),
//                                            mu_0 = log x, mu_n = (-1)^(n+1) (n-1)! x^-n
//
// The single-order functions are the general route, as everywhere in Laplace
// (one function per order). The *_orders_dd functions, which return every
// order 0..N at one argument, are an exception to that rule, accepted for this
// layer only because it was measured to pay: they share the shift loop and the
// powers, and on one thread over x in [1e-2, 1e8] orders 0..13 cost 3.2 us in
// one call against 7.8 us in fourteen for psi_n, and 4.3 us against 21 us for
// G_n (docs/numerics.md). A caller that needs every order at the same point
// uses them; any other caller uses the single-order functions.
//
// Method. Below X0 = 20 the argument is shifted up to z = x + m >= X0 with the
// recurrence psi_n(x) = psi_n(x + m) + (-1)^(n+1) n! sum_(j<m) (x + j)^-(n+1),
// and at z the asymptotic expansion is used,
//     psi(z)   ~ log z - 1/(2z) - sum_k b_(0,k) z^-2k,
//     psi_n(z) ~ (-1)^(n+1) z^-n [ (n-1)! + n!/(2z) + sum_k b_(n,k) z^-2k ],
// with the coefficients b of tools/reference/polygamma_dd_coefficients.py, the
// series stopped when a term falls below 2^-110 of the leading one. At X0 = 20
// order 13 needs 33 terms to reach 2^-107 (45 at 18; at 15 the divergent series
// never reaches it for n >= 6), and 20 is the cheapest of 20, 25, 30 and 40 at
// the same accuracy (docs/numerics.md).
//
// The difference is never formed from two values. With u = 1/z, v = 1/(z + h)
// after the common shift, every power difference is a sum of positive terms,
//     u^p - v^p = (u - v) S_p,  u - v = h u v,  S_p = sum_(i<p) u^i v^(p-1-i),
//     S_1 = 1,  S_(p+1) = u S_p + v^p,
// and the pole terms of the shift use the same identity with a = x + j and
// b = a + h. log((z + h)/z) is 2 atanh(h/(2z + h)) by its series. So the
// relative precision of G_n does not depend on h/x; the difference of two
// polygamma_dd values loses about x/h units of 2^-104 instead.
//
// The remainder leaves the leading term out of the expansion, and below X0 adds
// mu_n(z) - mu_n(x) in closed form: log1p(m/x) through the same atanh series for
// n = 0, and x^-n - z^-n = m x^-1 z^-1 S_n(1/x, 1/z) for n >= 1.
//
// Domain: x > 0, h > 0, 0 <= n <= 13. Anything else gives NaN. The result
// carries double-double precision while its magnitude is above about 1e-292
// (below, its low part is subnormal: order 13 reaches that only for x > 1e23)
// and below about 1e300 (the exact product of the host splits its operands).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/detail/double_double.hpp>

// b_(n,k), k = 1..36, n = 0..13, as (hi, lo) pairs:
// python -I tools/reference/polygamma_dd_coefficients.py
// clang-format off
#define LAPLACE_POLYGAMMA_DD_COEF \
  /* n = 0 */ 0x1.5555555555555p-4, 0x1.5555555555555p-58, -0x1.1111111111111p-7, -0x1.1111111111111p-63, 0x1.0410410410410p-8, 0x1.0410410410410p-62, -0x1.1111111111111p-8, -0x1.1111111111111p-64, 0x1.f07c1f07c1f08p-8, -0x1.f07c1f07c1f08p-63, -0x1.5995995995996p-6, 0x1.9a99a99a99a9ap-60, 0x1.5555555555555p-4, 0x1.5555555555555p-58, -0x1.c5e5e5e5e5e5ep-2, -0x1.7979797979798p-56, 0x1.86e7f9b9fe6e8p+1, -0x1.9180646019180p-57, -0x1.a74ca514ca515p+4, 0x1.ad759ad759ad7p-51, 0x1.1975cc0ed7304p+8, -0x1.28cfc4a33f129p-46, -0x1.c2f0566566566p+11, -0x1.5995995995996p-43, 0x1.ac572aaaaaaabp+15, -0x1.5555555555555p-39, -0x1.dc0b1a5cfbe16p+19, -0x1.73ef85973ef86p-35, 0x1.31fad7cbf3c00p+24, -0x1.e8188007a0620p-31, -0x1.c280563b8bcbdp+28, 0x1.0d0d0d0d0d0d1p-26, 0x1.7892edfdf5555p+33, 0x1.5555555555555p-21, -0x1.62b8b44651d09p+38, -0x1.cf0592849cf06p-17, 0x1.76024c215d22bp+43, -0x1.5555555555555p-11, -0x1.b6c0dfed2955bp+48, -0x1.09ae109ae109bp-6, 0x1.1cca39b77b027p+54, 0x1.79e958d7e43e8p-1, -0x1.97212d8cc1040p+59, -0x1.591e6c56134afp+4, 0x1.3f0cb06b17e29p+65, -0x1.cf63f8bcd29c2p+10, -0x1.1101d96823ee1p+71, -0x1.d0ace6ff3b3f0p+16, 0x1.fc474bdd53c20p+76, 0x1.ea44d3364d936p+21, -0x1.007db56db95dfp+83, 0x1.62c1f7db5e598p+28, 0x1.17c6dd28a9378p+89, 0x1.c198c1d0e0935p+35, -0x1.48df88a383ad8p+95, 0x1.c0e317b060b46p+37, 0x1.9f7b3fa37f314p+101, -0x1.3e3b3ee586b08p+45, -0x1.195c16c40d563p+108, 0x1.37ad4772a5d8ap+52, 0x1.97922eafb5d17p+114, -0x1.d1c72fe18d41ap+59, -0x1.3b0a43def5904p+121, 0x1.e3a565dd24964p+66, 0x1.035a171273534p+128, -0x1.dd86ac626ea32p+74, -0x1.c5e89f1fd242ep+134, 0x1.e1865bee143acp+79, 0x1.a575dd47b788ap+141, 0x1.e385ad0330ea4p+86, -0x1.9e84a01ae153ap+148, -0x1.eb9e165135e98p+94, \
  /* n = 1 */ 0x1.5555555555555p-3, 0x1.5555555555555p-57, -0x1.1111111111111p-5, -0x1.1111111111111p-61, 0x1.8618618618618p-6, 0x1.8618618618618p-60, -0x1.1111111111111p-5, -0x1.1111111111111p-61, 0x1.364d9364d9365p-4, -0x1.364d9364d9365p-59, -0x1.0330330330330p-2, -0x1.981981981981ap-57, 0x1.2aaaaaaaaaaabp+0, -0x1.5555555555555p-54, -0x1.c5e5e5e5e5e5ep+2, -0x1.7979797979798p-52, 0x1.b7c4f8f13e3c5p+5, -0x1.c3b070ec1c3b0p-53, -0x1.088fe72cfe72dp+9, 0x1.8d3018d3018d3p-51, 0x1.8301f89467e25p+12, 0x1.9f89467e251a0p-44, -0x1.523440cc0cc0dp+16, 0x1.f99f99f99f9a0p-39, 0x1.5c06d2aaaaaabp+20, -0x1.5555555555555p-34, -0x1.a089b7115c654p+24, 0x1.ba8e6b1ba8e6bp-30, 0x1.1edb2a4f34840p+29, -0x1.c996f807265bep-26, -0x1.c280563b8bcbdp+33, 0x1.0d0d0d0d0d0d1p-21, 0x1.901c1cddd4aabp+38, -0x1.5555555555555p-16, -0x1.8f0fcacf1c0aap+43, -0x1.8473226a98473p-11, 0x1.bc22ba679e993p+48, -0x1.5555555555555p-6, -0x1.12388bf439d59p+54, -0x1.3066530665306p-3, 0x1.75c96bc0d1733p+59, 0x1.b801224dadc90p+5, -0x1.17e6cf50c4b2cp+65, -0x1.da89d4f65a871p+9, 0x1.caa23d99f255bp+70, -0x1.8d0fdac7b7604p+16, -0x1.9982c61c35e52p+76, 0x1.46fca58127218p+21, 0x1.8d17b344e96f9p+82, 0x1.7f05c5026c9b2p+27, -0x1.a0cc46d24d38ap+88, -0x1.7f099a770d5d1p+32, 0x1.d81f95349d8dbp+94, 0x1.ed638e20f5f14p+39, -0x1.1fc3978f1337dp+101, 0x1.88c6b4ba549ddp+43, 0x1.7887b1ac2b44ap+107, 0x1.bf349dfffbe01p+50, -0x1.07c65557cc80dp+114, 0x1.090c9cbededecp+60, 0x1.8ad59d3a3822ep+120, 0x1.f31c2675fca1dp+63, -0x1.3b0a43def5904p+127, 0x1.e3a565dd24964p+72, 0x1.0b74e7cb06eddp+134, 0x1.271a3c74fbcfap+79, -0x1.e2472911cf671p+140, 0x1.7fcf60d67abf3p+86, 0x1.ccf8ea0670bd7p+147, 0x1.90da353b7d803p+92, -0x1.d255341e3d7e2p+154, 0x1.addc4dc946b2bp+99, \
  /* n = 2 */ 0x1.0000000000000p-1, 0x0.0p+0, -0x1.5555555555555p-3, -0x1.5555555555555p-57, 0x1.5555555555555p-3, 0x1.5555555555555p-57, -0x1.3333333333333p-2, -0x1.999999999999ap-57, 0x1.aaaaaaaaaaaabp-1, -0x1.5555555555555p-55, -0x1.a52e52e52e52ep+1, -0x1.4b94b94b94b95p-53, 0x1.1800000000000p+4, 0x0.0p+0, -0x1.e244444444444p+6, -0x1.1111111111111p-48, 0x1.051cf3cf3cf3dp+10, -0x1.8618618618618p-47, -0x1.5b3cdf6b0df6bp+13, -0x1.bed61bed61bedp-44, 0x1.16296aaaaaaabp+17, -0x1.5555555555555p-37, -0x1.0838d29f69f6ap+21, 0x1.2c12c12c12c13p-36, 0x1.25a5c1c000000p+25, 0x0.0p+0, -0x1.797ccde7bbbbcp+29, 0x1.1111111111111p-25, 0x1.15e450fcbadfep+34, -0x1.bb4a4046ed290p-21, -0x1.d09458ed682a3p+38, 0x1.7575757575757p-16, 0x1.b59ebf92a09abp+43, -0x1.5555555555555p-11, -0x1.cd6a427f786c5p+48, -0x1.251fcb4012520p-14, 0x1.0ea5299724a55p+54, 0x1.8000000000000p+0, -0x1.5f587350ea19ap+59, -0x1.43018d3018d30p+3, 0x1.f646a8cb1972dp+64, 0x1.bd06186186186p+8, -0x1.899c9389949b6p+70, -0x1.36a3a6f4de9bdp+13, 0x1.50cf253d0df6fp+76, -0x1.d397a4aaaaaabp+22, -0x1.39881fad99437p+82, 0x1.5d2cb75b70fadp+28, 0x1.3c6ee2daea04fp+88, -0x1.b763b38108ba3p+34, -0x1.59292aa627f2ep+94, -0x1.6f4cfcfaa5c44p+40, 0x1.95bb24393765cp+100, 0x1.a400c71229adbp+46, -0x1.004a32fb6d1dbp+107, -0x1.3a22f070a0ca3p+53, 0x1.5b1d17cab7e34p+113, 0x1.13889034ff865p+59, -0x1.f6d212af5dd58p+119, -0x1.26bff53427276p+65, 0x1.84aa46c54f425p+126, 0x1.5d69f6bb8495fp+72, -0x1.3ff66cee71668p+133, 0x1.6b33fb749928ap+78, 0x1.17fe62a88b410p+140, -0x1.f588445ac4315p+86, -0x1.03fa5c2399cd9p+147, 0x1.26e5ca339e2b1p+93, 0x1.ff64239f25123p+153, -0x1.11a6f6790062ep+99, -0x1.09f497b93f11fp+161, 0x1.d527a45cca51fp+106, \
  /* n = 3 */ 0x1.0000000000000p+1, 0x0.0p+0, -0x1.0000000000000p+0, 0x0.0p+0, 0x1.5555555555555p+0, 0x1.5555555555555p-54, -0x1.8000000000000p+1, 0x0.0p+0, 0x1.4000000000000p+3, 0x0.0p+0, -0x1.7088888888889p+5, 0x1.ddddddddddddep-49, 0x1.1800000000000p+8, 0x0.0p+0, -0x1.0f46666666666p+11, -0x1.999999999999ap-43, 0x1.466430c30c30cp+14, 0x1.8618618618618p-41, -0x1.dd73b33333333p+17, -0x1.999999999999ap-38, 0x1.a13e200000000p+21, 0x0.0p+0, -0x1.ad5c56430c30cp+25, -0x1.8618618618618p-30, 0x1.00f1098800000p+30, 0x0.0p+0, -0x1.61e5010940000p+34, 0x0.0p+0, 0x1.15e450fcbadfep+39, -0x1.bb4a4046ed290p-16, -0x1.ed9d9e7c3eacdp+43, 0x1.999999999999ap-12, 0x1.ec529784f4ae0p+48, 0x0.0p+0, -0x1.11f7177baf805p+54, 0x1.f51f51f51f51fp-4, 0x1.524e73fcedcebp+59, -0x1.2000000000000p+5, -0x1.cd24175a3341ap+64, -0x1.d3f904a7904a8p+9, 0x1.5950940ba17efp+70, 0x1.8fa1861861862p+11, -0x1.1ae88a0ae2cfbp+76, 0x1.482e980000000p+21, 0x1.f936b7db94f26p+81, -0x1.7ac6ee0000000p+26, -0x1.e9e4b17f3f795p+87, -0x1.9e6a21811f781p+33, 0x1.011a1851de240p+94, -0x1.4a0203b1ae2e9p+39, -0x1.233abbfc31b4fp+100, -0x1.d7a3d5cdef766p+44, 0x1.6303bfb210791p+106, -0x1.20fea3a037102p+51, -0x1.d0867c67b5c5dp+112, -0x1.f95f53cc236e8p+58, 0x1.456b464e0c651p+119, 0x1.280398d7c6f60p+58, -0x1.e71b8219e2e6ep+125, 0x1.e2760a757a11ep+71, 0x1.84aa46c54f425p+132, 0x1.5d69f6bb8495fp+78, -0x1.49f62055e4f1bp+139, -0x1.12e4c95f841c4p+83, 0x1.297e48d313f50p+146, 0x1.eb1f375f8f8b9p+92, -0x1.1c59d4c6f038dp+153, -0x1.d74aad78b00e3p+95, 0x1.1fa8540984da4p+160, -0x1.d9edeaa41037ap+106, -0x1.3382cf6e30eccp+167, 0x1.af3aeb05a4f76p+113, \
  /* n = 4 */ 0x1.4000000000000p+3, 0x0.0p+0, -0x1.c000000000000p+2, 0x0.0p+0, 0x1.8000000000000p+3, 0x0.0p+0, -0x1.0800000000000p+5, 0x0.0p+0, 0x1.0400000000000p+7, 0x0.0p+0, -0x1.5980000000000p+9, 0x0.0p+0, 0x1.2980000000000p+12, 0x0.0p+0, -0x1.422399999999ap+15, 0x1.999999999999ap-39, 0x1.ac63800000000p+18, 0x0.0p+0, -0x1.572b28ccccccdp+22, 0x1.999999999999ap-33, 0x1.45f8890000000p+26, 0x0.0p+0, -0x1.6a45e8c892492p+30, -0x1.2492492492492p-24, 0x1.d1b4e14680000p+34, 0x0.0p+0, -0x1.56d5d900f6000p+39, 0x0.0p+0, 0x1.1e937384a0b6ep+44, -0x1.2492492492492p-10, -0x1.0df232abf2468p+49, 0x0.0p+0, 0x1.1c9fbf98dd74ap+54, -0x1.0000000000000p+1, -0x1.4de5249ebde46p+59, -0x1.cea0ea0ea0ea1p+2, 0x1.b174849c10b0dp+64, -0x1.1100000000000p+10, -0x1.35d43fb09a702p+70, 0x1.82ca586fb5870p+16, 0x1.e59950305b1a8p+75, 0x1.c63f649249249p+19, -0x1.9f858abffd210p+81, -0x1.aefdc86000000p+27, 0x1.82cde4c41e099p+87, -0x1.880138e000000p+30, -0x1.86623d716694bp+93, -0x1.d1e49597289d9p+36, 0x1.a9d3384797ebap+99, -0x1.1149ab0f243e9p+45, -0x1.f48cf319756f0p+105, 0x1.5579a21831b20p+47, 0x1.3c2f56ba96abdp+112, 0x1.e9d36454ef5aap+53, -0x1.ac3bfaaf9b926p+118, -0x1.83c7c2806153cp+63, 0x1.362a3f0263d05p+125, 0x1.a4688db6b6a69p+70, -0x1.df7f14117b5b4p+131, 0x1.75d8649748533p+76, 0x1.8abcefe0647f6p+138, -0x1.348185a6355f0p+82, -0x1.596da9d9ebad0p+145, -0x1.57f1e0b3ff936p+91, 0x1.40bc268391843p+152, -0x1.dd04b099f27b0p+97, -0x1.3b73a80cb27f0p+159, -0x1.d0ad6d871e350p+105, 0x1.481bffdadb88fp+166, -0x1.dc935fa3227f7p+112, -0x1.685d4b1521557p+173, 0x1.b9590b6a9d51ep+119, \
  /* n = 5 */ 0x1.e000000000000p+5, 0x0.0p+0, -0x1.c000000000000p+5, 0x0.0p+0, 0x1.e000000000000p+6, 0x0.0p+0, -0x1.8c00000000000p+8, 0x0.0p+0, 0x1.c700000000000p+10, 0x0.0p+0, -0x1.5980000000000p+13, 0x0.0p+0, 0x1.4eb0000000000p+16, 0x0.0p+0, -0x1.92ac800000000p+19, 0x0.0p+0, 0x1.2684680000000p+23, 0x0.0p+0, -0x1.01605e999999ap+27, 0x1.999999999999ap-27, 0x1.08d9ef5000000p+31, 0x0.0p+0, -0x1.3cfd2baf80000p+35, 0x0.0p+0, 0x1.b499933218000p+39, 0x0.0p+0, -0x1.56d5d900f6000p+44, 0x0.0p+0, 0x1.307caabceac25p+49, -0x1.b6db6db6db6dbp-5, -0x1.2fb07901708f5p+54, 0x0.0p+0, 0x1.51fdb38586fa7p+59, 0x1.2000000000000p+5, -0x1.a15e6dc66d5d8p+64, 0x1.b7b6db6db6db7p+10, 0x1.1c7477066af40p+70, 0x1.6cd8000000000p+16, -0x1.aa03d792d45a2p+75, -0x1.d8538cccccccdp+20, 0x1.5d0631a2c17b1p+81, -0x1.5cc137db6db6ep+26, -0x1.37a4280ffdd8cp+87, -0x1.433e564800000p+33, 0x1.2e30dab937777p+93, 0x1.f9b7e17220000p+39, -0x1.3d2fd1ec2358dp+99, 0x1.0aec8cea5e000p+41, 0x1.674a377c682edp+105, -0x1.26962854c694cp+51, -0x1.b5fb54b646c12p+111, 0x1.2aca6dd52b7bcp+53, 0x1.1e8ae699188bbp+118, 0x1.2ddf3c9767c75p+64, -0x1.91783b04a1d94p+124, 0x1.9474b9a7a4c18p+69, 0x1.2c78ed0a50b1dp+131, 0x1.5d15252403c55p+74, -0x1.df7f14117b5b4p+137, 0x1.75d8649748533p+82, 0x1.9712d75f67a36p+144, -0x1.8f896474d9c28p+90, -0x1.6f0484778a67dp+151, -0x1.6d70febf3f8cap+97, 0x1.5ecdca1fe7289p+158, 0x1.b216f6bcf64bcp+100, -0x1.62e21d0e48cefp+165, 0x1.f53ce4c7fe046p+111, 0x1.7b605fd50dd65p+172, -0x1.8e14cd293fc6cp+117, -0x1.abeec92917957p+179, 0x1.98337b1d35a27p+124, \
  /* n = 6 */ 0x1.a400000000000p+8, 0x0.0p+0, -0x1.f800000000000p+8, 0x0.0p+0, 0x1.4a00000000000p+10, 0x0.0p+0, -0x1.41c0000000000p+12, 0x0.0p+0, 0x1.aa90000000000p+14, 0x0.0p+0, -0x1.6f18000000000p+17, 0x0.0p+0, 0x1.8d71000000000p+20, 0x0.0p+0, -0x1.0841340000000p+24, 0x0.0p+0, 0x1.a75e558000000p+27, 0x0.0p+0, -0x1.922693d000000p+31, 0x0.0p+0, 0x1.beefc3d700000p+35, 0x0.0p+0, -0x1.1f456f970c000p+40, 0x0.0p+0, 0x1.a6f4c69887400p+44, 0x0.0p+0, -0x1.618c87c8fdb00p+49, 0x0.0p+0, 0x1.4d085abea0c48p+54, 0x0.0p+0, -0x1.5f240be9aa25bp+59, -0x1.2000000000000p+5, 0x1.9bed32cabc814p+64, -0x1.0200000000000p+9, -0x1.0b607e531e0fep+70, -0x1.993b6db6db6dbp+14, 0x1.7e3c7ff09fb7ep+75, 0x1.ea42400000000p+21, -0x1.2b8ab3933d4f6p+81, -0x1.18357e0000000p+25, 0x1.00508c7386166p+87, -0x1.201de50524925p+32, -0x1.dd335d587cb3fp+92, 0x1.221117c380000p+37, 0x1.e19ddc9730666p+98, 0x1.c5fd0f4de3000p+44, -0x1.06ab99d78d459p+105, 0x1.fba17c9642bb0p+50, 0x1.34c3c7aee9883p+111, 0x1.b2d6f55725582p+57, -0x1.8613d77257040p+117, -0x1.ebc76c5c4a8b9p+58, 0x1.08280c9522a0dp+124, -0x1.59b634246c544p+70, -0x1.7ea698406a431p+130, 0x1.017f40f3c9086p+75, 0x1.27c70956276f1p+137, -0x1.a50be5ee11893p+83, -0x1.e6fd1061c148bp+143, 0x1.7dd7e314d2ba4p+89, 0x1.aa27b977e07f0p+150, 0x1.f6f0ab56b0219p+94, -0x1.8bb0ded0e137fp+157, -0x1.33fba54c61073p+102, 0x1.852c4c3b64710p+164, 0x1.c322f37342780p+105, -0x1.94c9e9244b0c0p+171, -0x1.a234585f1217fp+114, 0x1.bc94f04dac372p+178, 0x1.0d7f9f93a9432p+123, -0x1.016da502b82fep+186, -0x1.ba7107f06dbc5p+131, \
  /* n = 7 */ 0x1.a400000000000p+11, 0x0.0p+0, -0x1.3b00000000000p+12, 0x0.0p+0, 0x1.ef00000000000p+13, 0x0.0p+0, -0x1.1988000000000p+16, 0x0.0p+0, 0x1.aa90000000000p+18, 0x0.0p+0, -0x1.9cfb000000000p+21, 0x0.0p+0, 0x1.f0cd400000000p+24, 0x0.0p+0, -0x1.6b59a78000000p+28, 0x0.0p+0, 0x1.3d86c02000000p+32, 0x0.0p+0, -0x1.46bf581900000p+36, 0x0.0p+0, 0x1.8711cb5c20000p+40, 0x0.0p+0, -0x1.0d51189d9b400p+45, 0x0.0p+0, 0x1.a6f4c69887400p+49, 0x0.0p+0, -0x1.77a550458d8b0p+54, 0x0.0p+0, 0x1.76a9661674dd1p+59, 0x0.0p+0, -0x1.a0face257a0ccp+64, -0x1.9600000000000p+10, 0x1.01743fbeb5d0cp+70, 0x1.af60000000000p+16, -0x1.5eeea5cd1774ep+75, 0x1.f9b8800000000p+21, 0x1.06c997f56dce7p+81, -0x1.7793a00000000p+24, -0x1.ae976223a8222p+86, 0x1.1b4cc6b800000p+32, 0x1.8078d2ad49219p+92, -0x1.b02cd787b6db7p+37, -0x1.74d020ed216c9p+98, -0x1.bac54ade88000p+42, 0x1.8750433ad7533p+104, 0x1.e1bb38de90e00p+49, -0x1.bb41939bbe656p+110, 0x1.d8a0823d909b9p+55, 0x1.0e2b4eb90c573p+117, -0x1.c1f4a9dfa96f6p+56, -0x1.6181fb3f9edbap+123, -0x1.bdacba33a38e8p+64, 0x1.ef4b1797a0ed8p+129, -0x1.0835a1c44b1e0p+75, -0x1.72b1637e66f0fp+136, -0x1.63465c89ea9ffp+82, 0x1.27c70956276f1p+143, -0x1.a50be5ee11893p+89, -0x1.f634f8e4cf52fp+149, 0x1.4e35116bca809p+92, 0x1.c4ca350f5e86fp+156, 0x1.0b2fdb060d91dp+101, -0x1.b0c973b476553p+163, -0x1.21b67997143fcp+107, 0x1.b5d1d5c2d0ff2p+170, 0x1.fb8751e1aac70p+111, -0x1.d4097591f6c5ep+177, -0x1.e38c862decebbp+120, 0x1.07f86eae1e40cp+185, -0x1.dffc394053802p+130, -0x1.39bda11b507a6p+192, 0x1.64c63e52fa428p+137, \
  /* n = 8 */ 0x1.d880000000000p+14, 0x0.0p+0, -0x1.b120000000000p+15, 0x0.0p+0, 0x1.9230000000000p+17, 0x0.0p+0, -0x1.07ef800000000p+20, 0x0.0p+0, 0x1.c539000000000p+22, 0x0.0p+0, -0x1.ea6a100000000p+25, 0x0.0p+0, 0x1.4606b20000000p+29, 0x0.0p+0, -0x1.0528706400000p+33, 0x0.0p+0, 0x1.f0228c3200000p+36, 0x0.0p+0, -0x1.13b1725518000p+41, 0x0.0p+0, 0x1.6268204b7d000p+45, 0x0.0p+0, -0x1.04e68fd8ae660p+50, 0x0.0p+0, 0x1.b42c6ccd4b7a0p+54, 0x0.0p+0, -0x1.9adccfcc12d00p+59, -0x1.0000000000000p+6, 0x1.b133de09f71fap+64, -0x1.6000000000000p+10, -0x1.fc31ab3dacbf9p+69, -0x1.bb40000000000p+13, 0x1.49dcf1ac58f38p+75, -0x1.5d34000000000p+19, -0x1.d790aecb97850p+80, -0x1.30e0280000000p+25, 0x1.718b7db1226a5p+86, -0x1.c409e64000000p+31, -0x1.3c372c122f791p+92, 0x1.e018c3de40000p+37, 0x1.265c814cabfdbp+98, -0x1.5712a81f40000p+40, -0x1.2915da3cf6a28p+104, -0x1.08354dea55180p+50, 0x1.440e77acba50ep+110, 0x1.b777858c27fccp+56, -0x1.7cec5ad9d79f2p+116, 0x1.2b14f7f67422dp+62, 0x1.e13d24399dfb5p+122, -0x1.4c85ef3a59974p+67, -0x1.45e3d39ea6727p+129, -0x1.ecd6d9dd3cd64p+75, 0x1.d813927c85626p+135, -0x1.7bd31e2f17989p+81, -0x1.6ce69df06d553p+142, -0x1.b6e50c5f0bd5bp+86, 0x1.2c66257b800ccp+149, 0x1.645fea7a3630ap+95, -0x1.06dfba47c4857p+156, 0x1.adddf8e3cd7ffp+102, 0x1.e82a013491e98p+162, -0x1.7ff067dd795ecp+107, -0x1.e01f7c5c33468p+169, -0x1.40b33771cd3b6p+114, 0x1.f36357d236630p+176, 0x1.90b9965959b2cp+119, -0x1.123d8ae38697fp+184, -0x1.cdaa2b4f74691p+128, 0x1.3d96e5297c65ep+191, 0x1.3e848b1e9b89dp+136, -0x1.834612ddb7571p+198, 0x1.3c325a7736751p+144, \
  /* n = 9 */ 0x1.2750000000000p+18, 0x0.0p+0, -0x1.44d8000000000p+19, 0x0.0p+0, 0x1.5fea000000000p+21, 0x0.0p+0, -0x1.07ef800000000p+24, 0x0.0p+0, 0x1.fde0200000000p+26, 0x0.0p+0, -0x1.32824a0000000p+30, 0x0.0p+0, 0x1.c04934c000000p+33, 0x0.0p+0, -0x1.87bca89600000p+37, 0x0.0p+0, 0x1.931c11e8a0000p+41, 0x0.0p+0, -0x1.e2768814ea000p+45, 0x0.0p+0, 0x1.4c419e46c5300p+50, 0x0.0p+0, -0x1.04e68fd8ae660p+55, 0x0.0p+0, 0x1.cf6f339a2031ap+59, 0x0.0p+0, -0x1.ce3869c5952a1p+64, 0x1.c000000000000p+10, 0x1.0136cbd5eabacp+70, 0x1.de00000000000p+15, -0x1.3d9f0b068bf7cp+75, 0x1.3abe000000000p+21, 0x1.b0f1fd3234bf9p+80, 0x1.8d6af00000000p+26, -0x1.4433782bf82b7p+86, -0x1.a334370000000p+30, 0x1.099c425750bc6p+92, 0x1.bd8c714100000p+38, -0x1.da52c21b47359p+97, -0x1.2fdada32a0000p+42, 0x1.cbf08a07ccbc6p+103, 0x1.3cfe5b29e5800p+48, -0x1.e2c382a310c81p+109, -0x1.ad569e9cca470p+55, 0x1.116c34f9bd344p+116, 0x1.6599b15c837a8p+61, -0x1.4d4ecf7e9cab3p+122, -0x1.fa4da7085a618p+68, 0x1.b41f68d4372bcp+128, -0x1.dab2c199c2622p+72, -0x1.31859664bc0b5p+135, -0x1.c12d87ed211aep+76, 0x1.c952f5e8a1374p+141, 0x1.d00b7ac261243p+87, -0x1.6ce69df06d553p+148, -0x1.b6e50c5f0bd5bp+92, 0x1.35c956a75c0d3p+155, -0x1.107d1631f81ddp+101, -0x1.174db5ec40cdcp+162, 0x1.177b0e414afd0p+103, 0x1.0af6f8a8bfcbbp+169, 0x1.704439970d016p+111, -0x1.0e11b5f3dcd7bp+176, 0x1.a5cd9867fe475p+122, 0x1.20b56ec587714p+183, -0x1.c6152d4516093p+128, -0x1.45a914ee2fd47p+190, 0x1.b78bd9234b868p+133, 0x1.830fe74a8f9c3p+197, -0x1.fbce7672b2700p+142, -0x1.e4179795252cdp+204, 0x1.167de22a0824ap+149, \
  /* n = 10 */ 0x1.960e000000000p+21, 0x0.0p+0, -0x1.07ef800000000p+23, 0x0.0p+0, 0x1.49eb600000000p+25, 0x0.0p+0, -0x1.186e780000000p+28, 0x0.0p+0, 0x1.2ebd130000000p+31, 0x0.0p+0, -0x1.924b012000000p+34, 0x0.0p+0, 0x1.42349dea00000p+38, 0x0.0p+0, -0x1.320b63b530000p+42, 0x0.0p+0, 0x1.541faf1c47000p+46, 0x0.0p+0, -0x1.b53b6b52f4100p+50, 0x0.0p+0, 0x1.41df91548f068p+55, 0x0.0p+0, -0x1.0d0dc45773d93p+60, 0x0.0p+0, 0x1.fae1a07093364p+64, 0x1.c000000000000p+10, -0x1.0b389d263a3c5p+70, 0x1.6600000000000p+15, 0x1.397ac86cb613ap+75, -0x1.72e0000000000p+19, -0x1.96f3c62063556p+80, -0x1.ecbc900000000p+26, 0x1.22e2961dbb70cp+86, -0x1.c4fc26c000000p+32, -0x1.c7e860fdd4fd1p+91, -0x1.f3605b5600000p+37, 0x1.861d81703e94bp+97, 0x1.ce66465778000p+43, -0x1.6b275c9ce2850p+103, -0x1.0451c78761400p+49, 0x1.6e83adfe37262p+109, -0x1.86ca9eb53a3c0p+53, -0x1.8fc9e82f09e5bp+115, -0x1.671776b3af05ap+60, 0x1.d5f1fb0d3d31dp+121, 0x1.66a028d701fa9p+66, -0x1.28da30ccc3888p+128, 0x1.1a25ae791f025p+73, 0x1.920cf4a3a2dc5p+134, 0x1.a5319ac32066dp+79, -0x1.23335358033adp+141, 0x1.e29f04d38fe39p+87, 0x1.c22daa10feb27p+147, -0x1.7734b328a8606p+93, -0x1.729a38682f0a8p+154, -0x1.9f70282422014p+100, 0x1.444ec6b7345ddp+161, -0x1.8d42f33c4fbf4p+107, -0x1.2d1fc822b5dddp+168, -0x1.6d2af54a19b27p+113, 0x1.2829fbdb34c5fp+175, 0x1.e988bafe3926dp+121, -0x1.340c338a27e60p+182, 0x1.623d03ad3c12bp+127, 0x1.52549dcf7ab8bp+189, 0x1.6bdf2ef30a2d4p+134, -0x1.87cf6d2e918b9p+196, -0x1.6de577d831244p+142, 0x1.ddc7a1800944cp+203, 0x1.969692e535deep+149, -0x1.3256edec61866p+211, 0x1.701dd48f4c93ap+157, \
  /* n = 11 */ 0x1.308a800000000p+25, 0x0.0p+0, -0x1.cde3200000000p+26, 0x0.0p+0, 0x1.49eb600000000p+29, 0x0.0p+0, -0x1.3b7c470000000p+32, 0x0.0p+0, 0x1.7a6c57c000000p+35, 0x0.0p+0, -0x1.149390c600000p+39, 0x0.0p+0, 0x1.e34eecdf00000p+42, 0x0.0p+0, -0x1.f15282066e000p+46, 0x0.0p+0, 0x1.299bb938be200p+51, 0x0.0p+0, -0x1.99e7b49dc4cf0p+55, 0x0.0p+0, 0x1.41df91548f068p+60, 0x0.0p+0, -0x1.1ddea09ceb16cp+65, -0x1.8000000000000p+10, 0x1.1d1eea3f52ce8p+70, 0x1.fc00000000000p+16, -0x1.3d533a9d6527ap+75, 0x1.1490000000000p+21, 0x1.87d97a87e3988p+80, 0x1.8c1a000000000p+26, -0x1.0b0ffa0541301p+86, 0x1.f948830000000p+31, 0x1.8ff78e68e1bb0p+91, -0x1.bb6ad52000000p+35, -0x1.47af05b67115fp+97, 0x1.b912be5a30000p+43, 0x1.249621142eef9p+103, -0x1.a5334b3e66000p+49, -0x1.1bb6c05a90f7fp+109, 0x1.34a01c1e3c060p+55, 0x1.29cafd5e8ccf0p+115, -0x1.cf61283ccfd43p+61, -0x1.51525be7b059dp+121, 0x1.220867b0c8a69p+65, 0x1.9b33bbab958bap+127, -0x1.e319ee21ef226p+73, -0x1.0d05bc399133bp+134, -0x1.004dd9e23be5fp+79, 0x1.78ec255968ae9p+140, -0x1.ea42fdd2233f4p+84, -0x1.1a19b8bd4320fp+147, -0x1.18ebe6a619371p+92, 0x1.c22daa10feb27p+153, -0x1.7734b328a8606p+99, -0x1.7e2f0a2b7082ep+160, 0x1.5394569abceebp+106, 0x1.5893b322a7a3ap+167, 0x1.99e8dd8feb44dp+113, -0x1.495ac2e5f6eaap+174, 0x1.6132076de7c9ap+118, 0x1.4d2f3b569b5ebp+181, 0x1.a6b9d25e004bbp+127, -0x1.642e1b97be21fp+188, 0x1.99968c404d759p+133, 0x1.91c47b6661bb5p+195, 0x1.180c83e04e0aep+141, -0x1.dd84cd10c1622p+202, 0x1.884157c2106f3p+146, 0x1.2a9cc4f005cb0p+210, -0x1.01e1e430be54bp+156, -0x1.887f60d6dcf42p+217, -0x1.d0b38f50cbc5dp+162, \
  /* n = 12 */ 0x1.eee1100000000p+28, 0x0.0p+0, -0x1.b104ee0000000p+30, 0x0.0p+0, 0x1.5e8a160000000p+33, 0x0.0p+0, -0x1.76a3945000000p+36, 0x0.0p+0, 0x1.f0ae332c00000p+39, 0x0.0p+0, -0x1.8d94201ca0000p+43, 0x0.0p+0, 0x1.7995a90e38000p+47, 0x0.0p+0, -0x1.a39d9db56cd00p+51, 0x0.0p+0, 0x1.0db51fdb6c4d0p+56, 0x0.0p+0, -0x1.8d1876f8d6a88p+60, -0x1.0000000000000p+7, 0x1.4bee8ddf337ebp+65, 0x1.0000000000000p+11, -0x1.38ab7faba120ep+70, -0x1.5200000000000p+16, 0x1.49abbed937bedp+75, -0x1.6940000000000p+20, -0x1.82bd6f6fd3484p+80, -0x1.6ef0800000000p+26, 0x1.f60ea4fe1b9b7p+85, -0x1.047eb00000000p+31, -0x1.66dd77f70f989p+91, -0x1.9068ff8000000p+32, 0x1.193a1021beb78p+97, -0x1.4df1c776a0000p+43, -0x1.e1491063f6183p+102, 0x1.9f4e1e51da000p+46, 0x1.c005e2a6e7dedp+108, -0x1.64f68b378c300p+54, -0x1.c42b4290570b2p+114, 0x1.2f7cb340bea64p+58, 0x1.ed3833a49936dp+120, -0x1.fef1d549706eep+65, -0x1.21e2c6fb1b8d3p+127, 0x1.1c9f9c8df6379p+72, 0x1.6e3a1324d1305p+133, 0x1.e379cfd38e0acp+78, -0x1.f002930a23b75p+139, -0x1.88f89b91e6fe5p+80, 0x1.6741139937c66p+146, -0x1.268fb3d89330ap+89, -0x1.15b151da4e147p+153, 0x1.96ef91e8fe5b9p+97, 0x1.c93660b942ad3p+159, 0x1.e5dcf41569fc4p+104, -0x1.90193ea579c90p+166, 0x1.86fe9553fb93cp+111, 0x1.737f3d215cbc7p+173, -0x1.30878906b2ce9p+116, -0x1.6d60b0371dec4p+180, -0x1.fe0b1ff084b71p+126, 0x1.7c09dfaec9381p+187, -0x1.edd40c0cc7a9ap+133, -0x1.a1660855d2cfcp+194, -0x1.800727494a747p+138, 0x1.e36064772d956p+201, 0x1.83bc3aa777945p+145, -0x1.26b7f690575a9p+209, 0x1.32185429c624ap+153, 0x1.79ee693fc754ep+216, 0x1.b99e1b324f1cdp+162, -0x1.fd053196a68cap+223, 0x1.caae44466f770p+167, \
  /* n = 13 */ 0x1.b104ee0000000p+32, 0x0.0p+0, -0x1.b104ee0000000p+34, 0x0.0p+0, 0x1.8a5b58c000000p+37, 0x0.0p+0, -0x1.d44c796400000p+40, 0x0.0p+0, 0x1.5577c32e40000p+44, 0x0.0p+0, -0x1.2a2f181578000p+48, 0x0.0p+0, 0x1.32c9995b8d800p+52, 0x0.0p+0, -0x1.6f29e9febf360p+56, 0x0.0p+0, 0x1.f9b39bbb6b106p+60, 0x0.0p+0, -0x1.8d1876f8d6a88p+65, -0x1.0000000000000p+12, 0x1.60ad76bd26b6ap+70, -0x1.8000000000000p+13, -0x1.5fc0efa115450p+75, -0x1.f100000000000p+19, 0x1.877bf2a1f2329p+80, 0x1.d304000000000p+25, -0x1.e36ccb4bc81a5p+85, -0x1.caaca00000000p+31, 0x1.49799c46c21e0p+91, -0x1.2bcc8e0000000p+35, -0x1.ed7084f3b571cp+96, -0x1.913482fa80000p+42, 0x1.944377308227cp+102, 0x1.ff471457a0000p+44, -0x1.68f6cc4af8922p+108, -0x1.6442b4a14e400p+53, 0x1.5e04991265261p+114, -0x1.db82730d96160p+58, -0x1.6f63261546b91p+120, 0x1.bda5546926b9cp+66, 0x1.a0276b92e1464p+126, -0x1.ef1c0bf5f6dd9p+71, -0x1.fb4cdc3770371p+132, -0x1.bd15c0f223d82p+72, 0x1.4be4a1595d93dp+139, -0x1.04eccdd423a32p+85, -0x1.d10269d9817bep+145, 0x1.347cb77139bb9p+91, 0x1.5c070afc6e083p+152, -0x1.c756cd8e73a5cp+97, -0x1.15b151da4e147p+159, 0x1.96ef91e8fe5b9p+103, 0x1.d78013bf0cc2ap+165, -0x1.4af42449eab3ep+110, -0x1.a91ad28fd1659p+172, 0x1.9f6e7ea93b4d0p+117, 0x1.96532adc7d6e2p+179, -0x1.89a287bbea724p+125, -0x1.9b0cc63e01a9dp+186, -0x1.ee641f74aa6fbp+129, 0x1.b76b6aa218a8dp+193, -0x1.9afd2deec6dc3p+139, -0x1.efa929e5ea56bp+200, -0x1.72021fa9c21a9p+146, 0x1.268ebd389fc70p+208, 0x1.db11acef8337ap+154, -0x1.7065f4346d313p+215, -0x1.40b0cb65e4292p+160, 0x1.e43976d9b764cp+222, 0x1.f5d292d8755cep+168, -0x1.4e0b688add4c4p+230, -0x1.f4bf68cc71b68p+176
// clang-format on

namespace laplace::math {

namespace detail::pgdd {

using ddm::dd;

constexpr int max_order = 13;
constexpr int terms = 36;
constexpr double shift_threshold = 20.0;  // X0
constexpr double stop = 0x1p-110;

// 8064 bytes: in global memory, not in the 64 KB constant bank that the whole
// -rdc device program shares (measured on CUDA: the same ns per call in both)
#if defined(__CUDACC__)
LAPLACE_DEVICE_GLOBAL_TABLE double coef_dev[(max_order + 1) * terms * 2] = {LAPLACE_POLYGAMMA_DD_COEF};
#endif
inline constexpr double coef_host[(max_order + 1) * terms * 2] = {LAPLACE_POLYGAMMA_DD_COEF};

// b_(n,k), k = 1..terms
LAPLACE_HD_FORCEINLINE dd coef(int n, int k) {
  const int i = 2 * (n * terms + k - 1);
  KOKKOS_IF_ON_DEVICE((return dd(coef_dev[i], coef_dev[i + 1]);))
  KOKKOS_IF_ON_HOST((return dd(coef_host[i], coef_host[i + 1]);))
}

// n!, exact in double for n <= 18
LAPLACE_HD_FORCEINLINE double factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= static_cast<double>(i);
  return f;
}

LAPLACE_HD_FORCEINLINE dd recip(const dd& x) { return dd(1.0) / x; }

// x^n, n >= 0, by squaring
LAPLACE_HD_FORCEINLINE dd pow_int(dd x, int n) {
  dd r(1.0);
  while (n) {
    if (n & 1) r = r * x;
    x = x * x;
    n >>= 1;
  }
  return r;
}

// the number of recurrence steps that take x to z = x + m >= X0
LAPLACE_HD_FORCEINLINE int shift_count(const dd& x) {
  return x.hi < shift_threshold ? static_cast<int>(shift_threshold - x.hi) + 1 : 0;
}

LAPLACE_HD_FORCEINLINE dd quiet_nan_dd() { return dd(laplace::math::detail::quiet_nan()); }

// log((x + delta) / x) for x, delta > 0, with relative precision: below
// delta = x/2 as 2 atanh(w), w = delta / (2x + delta) <= 1/5, by its series
// (at most 24 terms); above, the result is at least log 1.5 and the plain
// double-double logarithm of the ratio is accurate.
LAPLACE_HD dd log_ratio(const dd& x, const dd& delta) {
  if (delta.hi < 0.5 * x.hi) {
    const dd w = delta / (x * 2.0 + delta), w2 = w * w;
    dd s = w, t = w;
    for (int i = 1; i < 40; ++i) {
      t = t * w2;
      const dd term = t / static_cast<double>(2 * i + 1);
      s = s + term;
      if (Kokkos::fabs(term.hi) <= stop * Kokkos::fabs(s.hi)) break;
    }
    return s * 2.0;
  }
  return ddm::log((x + delta) / x);
}

// psi(z) - log z = -u/2 - sum_k b_(0,k) u^2k, u = 1/z, z >= X0
LAPLACE_HD dd digamma_rest(const dd& u) {
  const dd w = u * u;
  dd s(0.0), pw(1.0);
  const double lead = 0.5 * u.hi;
  for (int k = 1; k <= terms; ++k) {
    pw = pw * w;
    const dd t = coef(0, k) * pw;
    s = s + t;
    if (Kokkos::fabs(t.hi) <= stop * lead) break;
  }
  return -(u * 0.5) - s;
}

// the bracket of psi_n(z), n >= 1: (n-1)! + n!/2 u + sum_k b_(n,k) u^2k, or
// the same without its first term (the remainder) when with_leading is false
LAPLACE_HD dd bracket(int n, const dd& u, bool with_leading) {
  const dd w = u * u;
  const double lead = factorial(n - 1);
  dd s(0.0), pw(1.0);
  for (int k = 1; k <= terms; ++k) {
    pw = pw * w;
    const dd t = coef(n, k) * pw;
    s = s + t;
    if (Kokkos::fabs(t.hi) <= stop * lead) break;
  }
  const dd second = u * (0.5 * factorial(n));
  return with_leading ? (dd(lead) + second) + s : second + s;
}

// sum_k b_(n,k) S_(n+2k), together with the first two terms (n-1)! S_n and
// n!/2 S_(n+1) for n >= 1, or 1/2 for n = 0; S is extended on demand.
struct power_differences {
  dd u, v, vp;
  dd S[max_order + 2 * terms + 2];
  int top;
  LAPLACE_HD power_differences(const dd& u_, const dd& v_) : u(u_), v(v_), vp(v_), top(1) { S[1] = dd(1.0); }
  LAPLACE_HD const dd& at(int p) {
    while (top < p) {
      S[top + 1] = u * S[top] + vp;
      vp = vp * v;
      ++top;
    }
    return S[p];
  }
  LAPLACE_HD dd series(int n) {
    dd s;
    double lead;
    if (n == 0) {
      s = dd(0.5);
      lead = 0.5;
    } else {
      s = at(n) * factorial(n - 1);
      lead = Kokkos::fabs(s.hi);
      s = s + at(n + 1) * (0.5 * factorial(n));
    }
    for (int k = 1; k <= terms; ++k) {
      const dd t = coef(n, k) * at(n + 2 * k);
      s = s + t;
      if (Kokkos::fabs(t.hi) <= stop * lead) break;
    }
    return s;
  }
};

}  // namespace detail::pgdd

// ------------------------------------------------------------ one order

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma function \f$\psi_n(x)\f$ for \f$n \in \{0, \dots, 13\}\f$.
 *
 * Evaluates \f$\psi_n(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)\f$
 * in double-double arithmetic by shifting arguments below \f$20\f$ with the
 * recurrence and summing the Bernoulli asymptotic expansion at \f$z \ge 20\f$;
 * see \ref polygamma_double_double "the module page".
 *
 * \param n Derivative order \f$n \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \return The double-double value of \f$\psi_n(x)\f$, or quiet NaN if \f$x \le 0\f$, \f$x\f$ is NaN, \f$n < 0\f$, or \f$n > 13\f$.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$, worst error is \f$0.74\f$ to \f$4.47\f$ units of \f$2^{-104}\f$
 * on both gcc/OpenMP and CUDA (test bounds \f$1.2\f$ to \f$6.8\f$); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_orders_dd(), polygamma_difference_dd(), polygamma_monomial_remainder_dd()
 */
// psi_n(x), n = 0..13
LAPLACE_HD detail::ddm::dd polygamma_dd(int n, const detail::ddm::dd& x) {
  namespace P = detail::pgdd;
  using P::dd;
  if (!(x.hi > 0.0) || n < 0 || n > P::max_order) return P::quiet_nan_dd();
  const int m = P::shift_count(x);
  dd poles(0.0);  // sum_j (x + j)^-(n+1), smallest first
  for (int j = m - 1; j >= 0; --j) {
    const dd t = P::recip(x + static_cast<double>(j));
    poles = poles + (n == 0 ? t : P::pow_int(t, n + 1));
  }
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z);
  if (n == 0) return (detail::ddm::log(z) + P::digamma_rest(u)) - poles;
  const dd mag = P::bracket(n, u, true) * P::pow_int(u, n) + poles * P::factorial(n);
  return (n % 2 == 1) ? mag : -mag;
}

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma difference \f$G_n(x, h) = \psi_n(x + h) - \psi_n(x)\f$ for \f$h > 0\f$ and \f$n \in \{0, \dots, 13\}\f$.
 *
 * Computes
 * \f[
 * G_n(x, h) = \psi_n(x + h) - \psi_n(x)
 * \f]
 * directly from sums of same-sign terms after a common shift to \f$z = x + m \ge 20\f$,
 * using positive-term power differences \f$u^p - v^p = h u v\, S_p(u, v)\f$ and
 * \f$2\operatorname{atanh}(h / (2z + h))\f$ for \f$\log((z + h)/z)\f$ when \f$n = 0\f$ and \f$h < z/2\f$;
 * see \ref polygamma_double_double "the module page".
 *
 * \param n Derivative order \f$n \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \param h Positive double-double shift \f$h \in (0, \infty)\f$.
 * \return The double-double value of \f$G_n(x, h)\f$, or quiet NaN if \f$x \le 0\f$, \f$h \le 0\f$, \f$n < 0\f$, \f$n > 13\f$, or any input is NaN.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$ with \f$x/h\f$ up to \f$10^{16}\f$, worst error is \f$0.69\f$ to \f$3.29\f$
 * units of \f$2^{-104}\f$ on both gcc/OpenMP and CUDA (test bounds \f$1.1\f$ to \f$5.0\f$); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_difference_orders_dd(), polygamma_dd(), polygamma_monomial_remainder_dd()
 */
// psi_n(x + h) - psi_n(x), n = 0..13, h > 0, as sums of terms of one sign
LAPLACE_HD detail::ddm::dd polygamma_difference_dd(int n, const detail::ddm::dd& x, const detail::ddm::dd& h) {
  namespace P = detail::pgdd;
  using P::dd;
  if (!(x.hi > 0.0) || !(h.hi > 0.0) || n < 0 || n > P::max_order) return P::quiet_nan_dd();
  const int m = P::shift_count(x);
  // a^-(n+1) - b^-(n+1) = h a^-1 b^-1 P_(n+1), P_1 = 1, P_(r+1) = b^-1 P_r + a^-r
  dd poles(0.0);
  for (int j = m - 1; j >= 0; --j) {
    const dd a = x + static_cast<double>(j);
    const dd ia = P::recip(a), ib = P::recip(a + h);
    dd Pr(1.0), iar = ia;
    for (int r = 1; r <= n; ++r) {
      Pr = ib * Pr + iar;
      iar = iar * ia;
    }
    poles = poles + ia * ib * Pr;
  }
  poles = poles * h;
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z), v = P::recip(z + h);
  P::power_differences S(u, v);
  const dd s = S.series(n);
  if (n == 0) return (P::log_ratio(z, h) + (u * v * h) * s) + poles;
  const dd mag = (u * v * h) * s + poles * P::factorial(n);
  return (n % 2 == 1) ? -mag : mag;
}

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma monomial remainder \f$r_n(x) = \psi_n(x) - \mu_n(x)\f$ for \f$n \in \{0, \dots, 13\}\f$.
 *
 * Computes
 * \f[
 * r_n(x) = \psi_n(x) - \mu_n(x), \qquad
 * \mu_0(x) = \log x, \qquad
 * \mu_n(x) = (-1)^{n+1}(n - 1)!\, x^{-n} \quad (n \ge 1),
 * \f]
 * without subtracting \f$\mu_n(x)\f$ from \f$\psi_n(x)\f$: the leading term is
 * omitted from the asymptotic expansion at \f$z = x + m \ge 20\f$, and for
 * \f$m > 0\f$ the closed-form difference \f$\mu_n(z) - \mu_n(x)\f$ is added;
 * see \ref polygamma_double_double "the module page".
 *
 * \param n Derivative order \f$n \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \return The double-double value of \f$r_n(x)\f$, or quiet NaN if \f$x \le 0\f$, \f$x\f$ is NaN, \f$n < 0\f$, or \f$n > 13\f$.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$, worst error is \f$3.44\f$ to \f$9.05\f$ units of \f$2^{-104}\f$
 * on both gcc/OpenMP and CUDA, peaking for \f$x \in [12, 15]\f$ (test bounds \f$5.2\f$ to \f$14\f$); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_monomial_remainder_orders_dd(), polygamma_dd(), polygamma_difference_dd()
 */
// psi_n(x) - mu_n(x), n = 0..13: mu_0 = log x, mu_n = (-1)^(n+1) (n-1)! x^-n
LAPLACE_HD detail::ddm::dd polygamma_monomial_remainder_dd(int n, const detail::ddm::dd& x) {
  namespace P = detail::pgdd;
  using P::dd;
  if (!(x.hi > 0.0) || n < 0 || n > P::max_order) return P::quiet_nan_dd();
  const int m = P::shift_count(x);
  dd poles(0.0);
  for (int j = m - 1; j >= 0; --j) {
    const dd t = P::recip(x + static_cast<double>(j));
    poles = poles + (n == 0 ? t : P::pow_int(t, n + 1));
  }
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z);
  if (n == 0) {
    dd r = P::digamma_rest(u);
    if (m > 0) r = r + (P::log_ratio(x, dd(static_cast<double>(m))) - poles);
    return r;
  }
  dd mag = P::bracket(n, u, false) * P::pow_int(u, n);
  if (m > 0) {
    // x^-n - z^-n = m x^-1 u S_n(x^-1, u), S_1 = 1, S_(p+1) = x^-1 S_p + u^p
    const dd ix = P::recip(x);
    dd S(1.0), up = u;
    for (int p = 1; p < n; ++p) {
      S = ix * S + up;
      up = up * u;
    }
    mag = (mag + poles * P::factorial(n)) - (ix * u * S * static_cast<double>(m)) * P::factorial(n - 1);
  }
  return (n % 2 == 1) ? mag : -mag;
}

// ------------------------------------------------------------ every order 0..N at once

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma functions \f$\psi_n(x)\f$ for every order \f$n = 0, \dots, N\f$ (\f$N \le 13\f$) in one call.
 *
 * Writes \f$\psi_n(x)\f$ into `out[n]` for \f$n = 0, \dots, N\f$, sharing the
 * shift loop and powers across orders. Leaves `out` untouched if \f$N < 0\f$
 * or \f$N > 13\f$, and sets `out[0]` through `out[N]` to quiet NaN if \f$x \le 0\f$
 * or \f$x\f$ is NaN.
 *
 * \param N Highest derivative order \f$N \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \param out Output array of at least \f$N + 1\f$ double-doubles receiving `out[n]` = \f$\psi_n(x)\f$ for \f$n = 0, \dots, N\f$.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$, worst error is \f$0.74\f$ to \f$3.95\f$ units of \f$2^{-104}\f$
 * on both gcc/OpenMP and CUDA (test bounds \f$1.2\f$ to \f$6.8\f$, shared with polygamma_dd()); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_dd(), polygamma_difference_orders_dd(), polygamma_monomial_remainder_orders_dd()
 */
// out[n] = psi_n(x), n = 0..N, N <= 13
LAPLACE_HD void polygamma_orders_dd(int N, const detail::ddm::dd& x, detail::ddm::dd* out) {
  namespace P = detail::pgdd;
  using P::dd;
  if (N < 0 || N > P::max_order) return;
  if (!(x.hi > 0.0)) {
    for (int n = 0; n <= N; ++n) out[n] = P::quiet_nan_dd();
    return;
  }
  const int m = P::shift_count(x);
  dd poles[P::max_order + 1];
  for (int n = 0; n <= N; ++n) poles[n] = dd(0.0);
  for (int j = m - 1; j >= 0; --j) {
    const dd t = P::recip(x + static_cast<double>(j));
    dd q = t;
    for (int n = 0; n <= N; ++n) {
      poles[n] = poles[n] + q;
      q = q * t;
    }
  }
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z);
  out[0] = (detail::ddm::log(z) + P::digamma_rest(u)) - poles[0];
  dd un = u;
  for (int n = 1; n <= N; ++n) {
    const dd mag = P::bracket(n, u, true) * un + poles[n] * P::factorial(n);
    out[n] = (n % 2 == 1) ? mag : -mag;
    un = un * u;
  }
}

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma differences \f$G_n(x, h) = \psi_n(x + h) - \psi_n(x)\f$ for every order \f$n = 0, \dots, N\f$ (\f$N \le 13\f$, \f$h > 0\f$) in one call.
 *
 * Writes \f$G_n(x, h) = \psi_n(x + h) - \psi_n(x)\f$ into `out[n]` for \f$n = 0, \dots, N\f$,
 * sharing the shift loop and positive-term power-difference tables across orders.
 * Leaves `out` untouched if \f$N < 0\f$ or \f$N > 13\f$, and sets `out[0]` through
 * `out[N]` to quiet NaN if \f$x \le 0\f$, \f$h \le 0\f$, or any input is NaN.
 *
 * \param N Highest derivative order \f$N \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \param h Positive double-double shift \f$h \in (0, \infty)\f$.
 * \param out Output array of at least \f$N + 1\f$ double-doubles receiving `out[n]` = \f$G_n(x, h)\f$ for \f$n = 0, \dots, N\f$.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$, worst error is \f$0.69\f$ to \f$3.29\f$ units of \f$2^{-104}\f$
 * on both gcc/OpenMP and CUDA (test bounds \f$1.1\f$ to \f$5.0\f$, shared with polygamma_difference_dd()); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_difference_dd(), polygamma_orders_dd(), polygamma_monomial_remainder_orders_dd()
 */
// out[n] = psi_n(x + h) - psi_n(x), n = 0..N, N <= 13, h > 0
LAPLACE_HD void polygamma_difference_orders_dd(int N, const detail::ddm::dd& x, const detail::ddm::dd& h,
                                               detail::ddm::dd* out) {
  namespace P = detail::pgdd;
  using P::dd;
  if (N < 0 || N > P::max_order) return;
  if (!(x.hi > 0.0) || !(h.hi > 0.0)) {
    for (int n = 0; n <= N; ++n) out[n] = P::quiet_nan_dd();
    return;
  }
  const int m = P::shift_count(x);
  dd poles[P::max_order + 1];
  for (int n = 0; n <= N; ++n) poles[n] = dd(0.0);
  for (int j = m - 1; j >= 0; --j) {
    const dd a = x + static_cast<double>(j);
    const dd ia = P::recip(a), ib = P::recip(a + h), iab = ia * ib;
    dd Pr(1.0), iar = ia;
    poles[0] = poles[0] + iab;
    for (int r = 1; r <= N; ++r) {
      Pr = ib * Pr + iar;
      iar = iar * ia;
      poles[r] = poles[r] + iab * Pr;
    }
  }
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z), v = P::recip(z + h);
  const dd uvh = u * v * h;
  P::power_differences S(u, v);
  out[0] = (P::log_ratio(z, h) + uvh * S.series(0)) + poles[0] * h;
  for (int n = 1; n <= N; ++n) {
    const dd mag = uvh * S.series(n) + (poles[n] * h) * P::factorial(n);
    out[n] = (n % 2 == 1) ? -mag : mag;
  }
}

/**
 * \ingroup polygamma_double_double
 * \brief Double-double polygamma monomial remainders \f$r_n(x) = \psi_n(x) - \mu_n(x)\f$ for every order \f$n = 0, \dots, N\f$ (\f$N \le 13\f$) in one call.
 *
 * Writes \f$r_n(x) = \psi_n(x) - \mu_n(x)\f$ into `out[n]` for \f$n = 0, \dots, N\f$,
 * sharing the shift loop, powers, and recurrence for \f$x^{-n} - z^{-n}\f$ across orders.
 * Leaves `out` untouched if \f$N < 0\f$ or \f$N > 13\f$, and sets `out[0]` through
 * `out[N]` to quiet NaN if \f$x \le 0\f$ or \f$x\f$ is NaN.
 *
 * \param N Highest derivative order \f$N \in \{0, \dots, 13\}\f$.
 * \param x Positive double-double argument \f$x \in (0, \infty)\f$.
 * \param out Output array of at least \f$N + 1\f$ double-doubles receiving `out[n]` = \f$r_n(x)\f$ for \f$n = 0, \dots, N\f$.
 *
 * \par Measured accuracy
 * Across orders \f$0\f$ to \f$13\f$, worst error is \f$3.44\f$ to \f$8.97\f$ units of \f$2^{-104}\f$
 * on both gcc/OpenMP and CUDA (test bounds \f$5.2\f$ to \f$14\f$, shared with polygamma_monomial_remainder_dd()); see \ref polygamma_double_double "the module page".
 *
 * \see polygamma_double_double, polygamma_monomial_remainder_dd(), polygamma_orders_dd(), polygamma_difference_orders_dd()
 */
// out[n] = psi_n(x) - mu_n(x), n = 0..N, N <= 13
LAPLACE_HD void polygamma_monomial_remainder_orders_dd(int N, const detail::ddm::dd& x, detail::ddm::dd* out) {
  namespace P = detail::pgdd;
  using P::dd;
  if (N < 0 || N > P::max_order) return;
  if (!(x.hi > 0.0)) {
    for (int n = 0; n <= N; ++n) out[n] = P::quiet_nan_dd();
    return;
  }
  const int m = P::shift_count(x);
  dd poles[P::max_order + 1];
  for (int n = 0; n <= N; ++n) poles[n] = dd(0.0);
  for (int j = m - 1; j >= 0; --j) {
    const dd t = P::recip(x + static_cast<double>(j));
    dd q = t;
    for (int n = 0; n <= N; ++n) {
      poles[n] = poles[n] + q;
      q = q * t;
    }
  }
  const dd z = x + static_cast<double>(m);
  const dd u = P::recip(z);
  out[0] = P::digamma_rest(u);
  if (m > 0) out[0] = out[0] + (P::log_ratio(x, dd(static_cast<double>(m))) - poles[0]);
  const dd ix = m > 0 ? P::recip(x) : dd(0.0);
  dd un = u, S(1.0), up = u;  // S_n(x^-1, u) as in polygamma_monomial_remainder_dd
  for (int n = 1; n <= N; ++n) {
    dd mag = P::bracket(n, u, false) * un;
    if (m > 0) {
      if (n > 1) {
        S = ix * S + up;
        up = up * u;
      }
      mag = (mag + poles[n] * P::factorial(n)) - (ix * u * S * static_cast<double>(m)) * P::factorial(n - 1);
    }
    out[n] = (n % 2 == 1) ? mag : -mag;
    un = un * u;
  }
}

}  // namespace laplace::math
