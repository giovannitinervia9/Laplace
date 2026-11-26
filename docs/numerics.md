# Numerics

This document records how each numerical routine of Laplace was measured and
chosen. For every function or group of functions it gives the candidates that
were compared (the platform's, Boost.Math's and Laplace's own), their accuracy
in ulps against mpmath references computed at 200 bits or more, their speed on
the CPU and on the GPU, and the choice that followed.

The rule for the choice is the following: Laplace's own implementation is used
when, measured, it is as accurate and as fast as the platform's; otherwise the
platform's is used, behind the wrapper `laplace::math`.

## Error measures

The error measures are defined in `benchmarks/measure.hpp`. A reference value
is stored as a double-double (hi, lo), and y is the computed value.

- **ulp**: ((y - hi) - lo) / ulp(hi), the relative error in units in the last
  place of the exact value.
- **ulp1**: ((y - hi) - lo) / ulp(max(|hi|, 1)), the absolute error in units of
  ulp(1) when |f| < 1, which is what counts when the value is added to
  quantities of order one.
- **ulpL**: ulp / max(1, |log f|), the error beyond the inherent error of
  computing f as exp(log f).
- **CR share**: the share of correctly rounded results (|error| <= 0.5 ulp).

Two further measures, ulpC for inverse functions and ulpN for the components of
bivariate derivatives, are defined in the sections where they are first used.

Unless stated otherwise, timings are in ns per call. The builds are the CPU
build (Kokkos Serial + OpenMP, gcc 14.2 of Rtools45), an MSVC 14.44 host build,
and the CUDA build (MSVC + nvcc, CUDA 13.1, sm_89) on an RTX 4070 SUPER.

**Reference files and measuring tools.** The reference scripts under `tools/reference/` are run in isolated mode with `python -I` (Python 3.13 and mpmath 1.3.0 as of 2026-11-11; mpmath by Johansson and others), and a reference file has the layout of, for example, `tests/data/reference/Phi.csv`. The function under test is evaluated over a reference file by `bench::eval_kernel` in `benchmarks/measure.hpp`, inside a Kokkos kernel. The same header defines `bench::ulp_of(a)`, the unit in the last place of a, and `bench::error_in(y, hi, lo, s)`, the error ((y - hi) - lo) in units of the scale s; points that cannot be scored are counted in `n_bad`, and a summary metric such as `max_ulp` or `max_ulp1` is taken over the N - n_bad points, where N - n_bad > 0. The tests also check the boundary and special values: arguments outside the domain (such as x <= 0 for functions defined on (0, inf)) and the limits (for instance log K_nu(x) tends to -inf and the derivative of log K_nu(x) in x tends to -1.0 as x tends to inf).

## exp and log: fdlibm-style own versions against the platforms

**What was compared.** `laplace::math::platform::{exp,log}` (Kokkos, which
calls the host libm on the CPU and CUDA's libdevice on the GPU) against
`laplace::math::own::{exp,log}` (`include/laplace/math/elementary.hpp`): the
fdlibm 5.3 algorithms, without tables. fdlibm's licence is a permissive notice
compatible with MIT.

**How.** `benchmarks/math_elementary.cpp` on the references written by
`tools/reference/elementary.py` (mpmath at 200 bits, stored as double-double
in `tests/data/reference/{exp,log}.csv`, about 20000 arguments each: a fixed-seed
sample over the whole domain, plus the arguments near 0 and 1, the overflow and
underflow thresholds, and subnormals). The error is in units in the last place
of the exact value; "CR" is the share of correctly rounded results
(|error| <= 0.5 ulp). Speed: 2^23 arguments (exp uniform on [-700, 700], log
log-uniform on [2^-1000, 2^1000]), the minimum of 5-7 repetitions, in ns per
call. Three builds: gcc 14.2 of Rtools45 with OpenMP (16 threads); MSVC 14.44
host with CUDA 13.1 on the RTX 4070 SUPER; the same CUDA build with
`--fmad=false`.

| function | build, space | max error (ulp) | CR share | ns/call |
|---|---|---|---|---|
| exp platform | gcc, 1 thread | 0.5002 | 0.9997 | 29.0 |
| exp own | gcc, 1 thread | 1.0000 | 0.9270 | 6.4 |
| exp platform | gcc, OpenMP 16 | same | same | 2.15 |
| exp own | gcc, OpenMP 16 | same | same | 0.65 |
| exp platform | MSVC host, 1 thread | 1.4106 | | 2.9 |
| exp own | MSVC host, 1 thread | 1.0000 | | 8.1 |
| exp platform | CUDA | 1.0000 (0.7930 on normal results) | 0.9498 | 0.070 |
| exp own | CUDA | 1.0000 | 0.9272 | 0.161 |
| exp own | CUDA, no FMA contraction | 1.0000 | 0.9270 | 0.191 |
| log platform | gcc, 1 thread | 0.5003 | 0.9992 | 27.6 |
| log own | gcc, 1 thread | 0.7767 | 0.9799 | 4.1 |
| log platform | gcc, OpenMP 16 | same | same | 2.34 |
| log own | gcc, OpenMP 16 | same | same | 0.45 |
| log platform | MSVC host, 1 thread | 0.5137 | | 3.4 |
| log own | MSVC host, 1 thread | 0.7767 | | 5.0 |
| log platform | CUDA | 0.5207 | 0.9988 | 0.131 |
| log own | CUDA | 0.7774 | 0.9827 | 0.145 |
| log own | CUDA, no FMA contraction | 0.7767 | 0.9799 | 0.180 |

**Identity across platforms** (results compared bit for bit with the gcc build,
on the reference arguments):

| function | CUDA (default) | CUDA without FMA contraction |
|---|---|---|
| exp own | 13 of 20019 differ | 0 differ |
| log own | 132 of 20017 differ | 0 differ |
| exp platform | 1004 of 20019 differ | 1004 differ |
| log platform | 40 of 20017 differ | 40 differ |

**What this says.**

1. No candidate satisfies the rule today. On the CPU the own versions are 4.5
   to 7 times faster than the libm of Rtools (whose exp and log are correctly
   rounded in practice but cost about 28 ns per call), but less accurate: 1.0
   ulp against 0.5 for exp, 0.78 against 0.50 for log. On the GPU the own
   versions are slower than CUDA's (2.3 times for exp, 1.1 times for log) with
   the same maximum error for exp and a larger one for log.
2. The platforms disagree with each other: MSVC's exp reaches 1.41 ulp, CUDA's
   1.0 ulp on subnormal results, and the platform versions differ from the gcc
   build in 1004 (exp) and 40 (log) of 20000 arguments.
3. The own versions give bit-identical results on gcc, MSVC and CUDA only when
   nvcc does not contract multiply-adds into FMA (`--fmad=false`, CMake option
   `LAPLACE_CUDA_FMAD=OFF`). That costs 19-24 per cent on these two functions;
   on whole kernels it is not measured. The other route to identity is to write
   the FMAs explicitly and build the CPU code for processors with FMA
   (x86-64-v3, from 2013 on), which makes CPU binaries unusable on older
   processors.

**Chosen.** Bit identity across CPU and GPU is not required: chasing it would
mean rewriting every elementary function and building without FMA, for
differences of one ulp (a relative 1.1e-16), far below the tolerances of the
optimizers. The wrapper `laplace::math::{exp,log}` forwards to the platform;
tests compare with relative tolerances. The own versions stay in
`laplace::math::own` as measured candidates. One point stays open and is a
matter of speed only: in the gcc build (the one R loads) the libm of Rtools
costs about 28 ns per exp or log on one thread, more than a whole Gaussian
log-density; if profiling of the CPU kernels shows exp and log dominating, the
own versions (4-6 ns, error up to 1.0 ulp) or another libm come back into
question.

**Details of the own exp and log.** In exp, ln 2 is split into a high part with trailing zeros, ln2_hi = 6.93147180369123816490e-1, and a low part, ln2_lo = 1.90821492927058770002e-10, and the reduced argument is r = (x - k ln2_hi) - k ln2_lo, so that forming x - k ln 2 does not cancel. On |r| <= ln(2)/2, with t = r^2, R(t) = 2 + t (P1 + P2 t + ... + P5 t^4) and c = r - t (P1 + t (P2 + t (P3 + t (P4 + t P5)))), the exponential is e^r = 1 + 2r/(2 - c), with P1 = 1.66666666666666019037e-1, P2 = -2.77777777770155933842e-3, P3 = 6.61375632143793436117e-5, P4 = -1.65339022054652515390e-6 and P5 = 4.13813679705723846039e-8. For k = 0, exp(r) = 1 - ((r c)/(c - 2) - r). For k != 0 the result is scaled by 2^k, built directly in the IEEE 754 exponent field: for k >= -1021 by 2^k itself (k = 1024 is handled separately), and for k < -1021 in two steps, 2^(k+1000) times 2^-1000, so that subnormal results are evaluated without premature underflow. In log, a subnormal input (x < 2^-1022) is multiplied by 2^54 and k is adjusted by -54. For -2^-20 <= f < 2^-20, log(1 + f) = f - R(f) with R(f) = f^2 (1/2 - f/3). Otherwise, with s = f/(2 + f), log(1 + f) = 2s + s R(s^2), where R(s^2) is a minimax polynomial of degree 7 in z = s^2 with coefficients Lg1, ..., Lg7, evaluated by separating the even powers (w = z^2) from the odd ones.

## lgamma, digamma, trigamma: platform against Boost.Math

**What was compared.** The platform's `lgamma` (Kokkos: libm of Rtools on the
gcc build, UCRT on the MSVC host, libdevice on CUDA) and Boost.Math 1.90
(`lgamma`, `digamma`, `trigamma`; Boost Software License, header-only, taken
from the R package BH; called with a policy that never throws and never
promotes double to long double). Boost.Math 1.90 compiles these functions for
CUDA (`BOOST_MATH_GPU_ENABLED`); it also does so for the incomplete gamma and
beta functions and their inverses, erf, Bessel I and K, log1p and expm1, but
not for `polygamma` (orders 2 and above) nor for Owen's T. There is no
platform digamma or trigamma.

**How.** `benchmarks/math_special.cpp`, references from `tools/reference/special.py`
(mpmath, about 20000 positive arguments per function, with extra points near
the zeros: lgamma at 1 and 2, digamma at 1.4616). "ulp1" is the absolute error
in units of ulp(max(|f|, 1)), which is what counts when the value is added to
quantities of order one. Speed on 2^23 arguments with exponent uniform in
[-5, 15].

| function | build | max error (ulp) | max error (ulp1) | CR share | ns/call 1 thread | ns/call parallel |
|---|---|---|---|---|---|---|
| lgamma platform | gcc (Rtools libm) | 1.9e15 near x = 1; plus wrong infinities at subnormal x | 2.57 | 0.50 | 37.6 | 8.4 (OpenMP 16) |
| lgamma Boost | gcc | 2.42 | 2.42 | 0.72 | 81.2 | 5.5 (OpenMP 16) |
| lgamma platform | MSVC host | 2.82 | | | 29.1 | |
| lgamma Boost | MSVC host | 2.42 | | | 21.9 | |
| lgamma platform | CUDA | 2.45 | 2.45 | 0.60 | | 0.63 |
| lgamma Boost | CUDA | 2.42 | 2.42 | 0.74 | | 2.86 |
| digamma Boost | gcc / CUDA | 2.14 | 2.14 | 0.82 / 0.84 | 22.7 (MSVC 10.7) | 1.70 / 0.53 |
| trigamma Boost | gcc / CUDA | 1.82 / 2.03 | 1.57 / 2.03 | 0.82 | 5.4 (MSVC 6.7) | 0.54 / 0.37 |

**What this says.**

1. Every candidate is within about 2.5 ulp where it works; none is close to
   correct rounding. The Rtools libm's `lgamma` loses all relative accuracy just
   below x = 1 (where lgamma is near zero; the absolute error stays at 2.6
   ulp1), returns wrong infinities for subnormal arguments, and scales badly
   across threads (16 threads give only 4.5 times one thread, consistent with
   the global `signgam` it writes).
2. On the gcc build everything built on exp and log is slow, because the Rtools
   libm's exp and log cost about 28 ns each (previous section): Boost's lgamma
   takes 81 ns there against 22 ns with MSVC's UCRT.
3. On CUDA the platform's lgamma is 4.5 times faster than Boost's, with the same
   accuracy.

**Questions raised.** Whether Boost.Math becomes a build dependency of Laplace
(header-only, Boost licence, GPU-capable for most of the special functions that
the distributions need), with Laplace's own code only where Boost has no GPU
version (polygamma of order 2 and above, Owen's T) or where measurement shows
it worse; whether the wrapper may choose a different implementation per backend
(for lgamma: Boost on the host, the platform on CUDA); and what to do about the
slow exp and log of the Rtools libm, which weigh on every function of the CPU
build that R loads.

**Chosen.** Boost.Math is a build dependency of Laplace for the special
functions, because it is header-only, under a permissive licence and compiles
for the GPU; Laplace writes its own only where Boost has no GPU version or
measures worse. Still open: a different implementation per backend in the
wrapper, and the slow exp and log of the Rtools libm.

## polygamma of orders 0 to 6: own against Boost.Math

**What.** `laplace::math::own::polygamma(n, x)`, n = 0..6, x > 0
(`include/laplace/math/polygamma.hpp`): upward recurrence to z >= 15, then the
asymptotic expansion with the Bernoulli numbers B_2..B_20; for n >= 1 every
term has the sign of the result, so there is no cancellation; for large z the
factor z^-n is scaled by 2^600 so that only the last product rounds in the
subnormal range. Compared with Boost.Math's `digamma`, `trigamma` (GPU) and
`polygamma` (host only), against mpmath (`tests/data/reference/polygamma{2..6}.csv`).

| function | own: max ulp (ulp1) | own ns/call: 1 thread gcc, CUDA | Boost: max ulp | Boost ns/call: 1 thread gcc, CUDA |
|---|---|---|---|---|
| digamma | unbounded at the root (8.1 ulp1) | 34.0, 0.65 | 2.14 | 23.0, 0.53 |
| trigamma | 2.89 (2.11) | 12.7, 0.83 | 1.82-2.03 | 5.3, 0.37 |
| polygamma 2 | 3.36 | 15.8, 0.91 | 72 | 110, host only |
| polygamma 3 | 5.26 | 17.8, 2.40 | 160 | 125, host only |
| polygamma 4 | 6.34 | 21.0, 2.25 | 227 | 127, host only |
| polygamma 5 | 7.90 | 23.9, 1.75 | 563 | 137, host only |
| polygamma 6 | 9.44 | 27.6, 1.57 | 830 | 147, host only |

The own errors grow with n near x = 1, where up to 15 recurrence terms, each a
power computed by n + 1 products, are summed; they stay below 2.1e-15 relative.

**Chosen** (`include/laplace/math/special.hpp`): digamma and trigamma from
Boost; polygamma of order 2 to 6 own; lgamma from Boost on every backend until
the per-backend question is settled. `tests/math/test_special.cpp` checks each
against mpmath inside a kernel, with the measured error plus a margin as bound
(passes on the gcc/OpenMP and CUDA builds).

**Constants and points.** The even Bernoulli numbers used in the asymptotic expansion are B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66, B_12 = -691/2730, B_14 = 7/6, B_16 = -3617/510, B_18 = 43867/798 and B_20 = -174611/330. The accuracy of the gamma functions in the manual is measured across log-uniform points from 2^-60 to 2^1000 and uniform values in [0, 30].

## Incomplete gamma and beta, their inverses, erfc, normal quantile

**What.** Boost.Math 1.90 `gamma_p`, `gamma_q`, `gamma_p_inv`, `ibeta`,
`ibetac`, `ibeta_inv`, `erfc`, `erfc_inv` (normal quantile as
-sqrt(2) erfc^-1(2p), through 1 - p for p >= 1/2), with the Laplace policy (no
exceptions, no promotion to long double: the only route that runs on the GPU),
and on the gcc host also with Boost's default promotion to the 80-bit long
double ("boost-ld"). The platform's erfc for comparison. References
(`tools/reference/incomplete.py`): mpmath at 200 bits, about 10000 arguments
per function, with shapes up to 2^14 (gamma) and 2^12 (beta) and arguments up
to 30 standard deviations into both tails; the upper incomplete gamma by
Legendre's continued fraction (mpmath's own failed to converge there), the
complement of the incomplete beta by the symmetry I_{1-x}(b, a) (a subtraction
1 - I lost it in the far tail). Inverses: forward error against the exact
inverse of the double p, and backward error |F(y) - p| in ulps of p
(`tools/reference/backward.py`), because the inverse of a small shape is
ill-conditioned.

| function | Boost, double (CPU and GPU) | Boost, long double (gcc host only) | ns/call: gcc 1 thread, OpenMP 12, CUDA |
|---|---|---|---|
| gamma_p | 3.8e4 ulp (8e-12 relative) in the tails of large a; 13 ulp1 | 6.7 ulp | 186, 19.4, 24.6 (ld: 538) |
| gamma_q | 2.7e4 ulp; 26 ulp1 | 9.3 ulp | 184, 18.5, 22.7 (ld: 535) |
| gamma_p_inv | wrong (1e248) for subnormal p; backward error median 0.2, 99th percentile 5.7e3 ulps of p | 5.5e8 ulp forward at p = 4e-323 | 1343 ld |
| ibeta | 7.5e15 ulp: all relative accuracy lost in the far tail (absolute 18 ulp1) | 1.8 ulp | 535, 54, 81 (ld: 1704) |
| ibetac | 7.7e13 ulp; 18 ulp1 | 2.2 ulp | 533, 52, 81 (ld: 1704) |
| ibeta_inv | NaN for subnormal p; factor-2 errors for p near 1e-44; backward 99th percentile 4.2e4 | 1.0 ulp | 2156, 205, 269 (ld: 6431) |
| erfc platform | 2.1 ulp (gcc), 2.7 ulp (CUDA) | | 38.0, 3.7, 0.25 |
| erfc Boost | 3.2 ulp | | 56.1, 5.1, 1.26 |
| qnorm Boost | 2.9 ulp (CUDA 2.7) | | 25.0, 2.3, 1.19 |

**What this says.** In double, the only precision the GPU has, Boost's
incomplete gamma and beta keep the absolute error small (13-26 ulp1) but lose
relative accuracy in the tails, up to all of it for the incomplete beta; the
inverses fail for very small p. With long double they are excellent, but long
double exists only on the gcc host (with MSVC it is double), is 3 times slower,
and cannot run on the GPU. Tail probabilities matter for censored likelihoods
(log P of a tail), quantile residuals and the cdfs of the counts: there a
relative error of 80 per cent is a wrong likelihood. erfc and the normal
quantile are fine (2-3 ulp). On the GPU the incomplete functions are slow
(23-269 ns per call, slower than the CPU with 12 threads): branchy code in FP64.

**Question raised.** Whether to write Laplace's own incomplete gamma and beta
for the tails (Temme's uniform asymptotic expansions, DiDonato and Morris, in
double with the result kept in log scale where it underflows), measured against
Boost; erfc and the normal quantile could go to the wrapper as they were. The
following sections answer it.

## Own incomplete gamma

**What.** `laplace::math::own::gamma_inc(a, x)` (`include/laplace/math/incomplete_gamma.hpp`),
returning P, Q and their logarithms: the prefactor x^a e^-x / Gamma(a + 1) as
exp(-a phi(x/a)) / (sqrt(2 pi a) Gamma*(a)) with phi from log1pmx near x = a and
directly elsewhere (through 1 + (x - a)/a a small x/a lost its digits: 1.9e11
ulp at a = 21.9, x = 1.3e-9 in a first version); the small-a series with Q
computed directly; the series for P below a + 1 and Legendre's continued
fraction for Q above, the complement formed only where it is not small. The
continued fraction stops at |delta - 1| <= 4 eps: with FMA contraction on CUDA
delta settled one ulp away from 1 and the loop ran to its limit (the first GPU
run did not end).

| | max ulp | ulp1 | ulpL | ns/call: gcc 1 thread, OpenMP 12, MSVC 1 thread, CUDA |
|---|---|---|---|---|
| gamma_p own | 4729 | 7.6 | 30.3 | 262, 22.4, 142, 17.9 |
| gamma_q own | 1884 | 7.6 | 24.7 | 261, 22.7, 141, 17.6 |
| gamma_p Boost double | 37510 | 13.0 | 312 | 191, 19.3, 116, 24.6 |
| gamma_q Boost double | 26600 | 25.6 | 243 | 197, 20.3, 113, 22.8 |
| gamma_p Boost long double (gcc only) | 6.7 | 0.25 | 0.50 | 506 |

ulpL = ulp / max(1, |log f|). The worst own case, a = 4753.75, x = 2687.18,
has P = e^-646 with an error of 5e-13 relative; the problem's own condition
number there, |d log P / d log x| = |a - x| = 2066, is of the same order, so a
one-ulp change in x moves P by more than the computed error. The own version is
8 to 14 times more accurate than Boost in double, as fast on the GPU, 1.4 times
slower on one CPU thread (the series and fraction need O(sqrt(a)) terms near
x = a; Temme's uniform expansion would remove that, not written). Chosen:
`laplace::math::gamma_p`, `gamma_q`, `log_gamma_p`, `log_gamma_q` = own;
tested on CPU and GPU (`tests/math/test_special.cpp`, ulp1 <= 12, ulpL <= 45).

## Own incomplete beta

**What.** `laplace::math::own::beta_inc(a, b, x)` (`include/laplace/math/incomplete_beta.hpp`),
returning I, 1 - I and their logarithms. Prefactor x^a (1-x)^b / B(a, b): for
a, b >= 10 through the mean p0 = a/(a+b) as exp(a L(u) + b L(v)) times
Stirling's Gamma* factors, L(t) = log(1+t) - t and a u + b v = 0, so nothing
cancels; otherwise from Boost's beta(a, b). Then Numerical Recipes' continued
fraction below x = (a+1)/(a+b+2) and the symmetry above, the small side always
computed directly.

| | max ulp | ulp1 | ulpL | ns/call: gcc 1 thread, OpenMP 12, MSVC 1 thread, CUDA |
|---|---|---|---|---|
| ibeta own | 5809 (CUDA 10740) | 216 (CUDA 331) | 982 (CUDA 1622) | 561, 43.0, 184, 13.9 |
| ibetac own | 7888 (CUDA 8303) | 216 (CUDA 331) | 1311 (CUDA 1349) | 554, 43.9, 183, 12.4 |
| ibeta Boost double | 7.5e15 | 17.8 | 1.1e13 | 734, 67.1, 255, 81.0 |
| ibetac Boost double | 7.7e13 | 18.0 | 1.1e11 | 741, 66.8, 255, 81.3 |
| ibeta Boost long double (gcc only) | 1.8 | 0.25 | 0.50 | 2339 |

**What this says.** The own version keeps the relative accuracy of the tails
(at worst 1e-12 relative, against a total loss for Boost in double) and is
faster everywhere (6 times on CUDA). Its absolute error is larger than Boost's
(216-331 ulp of 1, 2.4e-14 to 3.7e-14), at shapes where one is small, the other
large and x near the mean: (a, b, x) = (1.149, 3599.17, 0.00061), (6111.26,
17.29, 0.99689), (4902.22, 7.29, 0.9983); there x sits next to the switching
point (a+1)/(a+b+2) and the continued fraction needs many terms. The remedy is
DiDonato and Morris' expansions for one small and one large shape (BGRAT, in
terms of the incomplete gamma) and for two large shapes near the mean (BASYM);
not written in this version (see the next section). Chosen:
`laplace::math::ibeta`, `ibetac`, `log_ibeta`, `log_ibetac` = own; tested
(ulp1 <= 500, ulpL <= 2500) on CPU and GPU. At this stage the inverses
(gamma_p_inv, ibeta_inv) were still Boost's, which fail for very small p.

## Incomplete beta: DiDonato and Morris' expansions

**What.** `include/laplace/math/detail/ibeta_expansions.hpp`, wired into
`own::beta_inc` in `incomplete_beta.hpp`. The pieces of DiDonato and Morris'
BRATIO (ACM TOMS 708, 1992), adapted from Boost's `ibeta_imp` (Boost licence)
but working in logarithms and with Laplace's cancellation-free prefactor: BPSER
(power series), BUP (finite sum, summed outwards from its largest term), BGRAT
(one small and one large shape, through the incomplete gamma, with the branches
of `own::gamma_inc`), BFRAC (continued fraction), BASYM (Temme's expansion for
both shapes above 100 near the mean). lambda = a - (a + b) x is computed with
the sum and the product carried exactly (FMA on the device, Dekker's product on
the host). Measured choices that differ from BRATIO: the continued fraction
instead of BPSER and BUP + BPSER for a, b > 1, b < 40, and instead of
BUP + BGRAT for a < 500 (as accurate there with this prefactor, faster, fewer
paths per warp). On the device `direct_side` is `__noinline__`: inlined into a
Kokkos `parallel_reduce` its divergent body left a warp unconverged at the
reduction barrier and the kernel stopped with cudaErrorIllegalInstruction.
New references: `tools/reference/ibeta_hard.py` writes `ibeta_hard.csv` and
`ibetac_hard.csv`, concentrated where one shape is small, the other large and x
near the mean. Bench: `benchmarks/math_ibeta.cpp`; tests: `tests/math/test_ibeta.cpp`
and the bounds of the incomplete-beta case in `test_special.cpp`.

| reference | DM: max ulp, ulp1, ulpL; log versions (ulp of max(abs log f, 1)) | previous own (continued fraction only) | Boost double | Boost long double (gcc host) |
|---|---|---|---|---|
| ibeta | 2597, 2.82, 12.7; 17 (CUDA 2096, 3.80, 12.9; 12) | 5809, 216, 982; 861 | 7.5e15, 17.8, 1.1e13 | 1.76, 0.25, 0.50 |
| ibetac | 2076, 2.82, 16.9; 10 (CUDA 1887, 3.80, 12.2; 15) | 7888, 216, 1311; 1544 | 7.7e13, 18.0, 1.1e11 | 2.18, 0.25, 0.51 |
| ibeta_hard | 1325, 3.65, 18.8; 22 (CUDA the same) | 10950, 1059, 3745; 5086 | 4394, 72.3, 236 | 2.04, 0.25, 0.51 |
| ibetac_hard | 1701, 3.65, 18.4; 15 (CUDA 19.7; 16) | 23690, 1059, 6937; 5193 | 4203, 72.6, 290 | 1.07, 0.26, 0.51 |

| ns/call | gcc 1 thread: common, mixed, large shapes | OpenMP 12 | MSVC 1 thread | CUDA |
|---|---|---|---|---|
| DM | 336, 377, 403 | 34.8, 40.3, 44.6 | 194, 205, 315 | 48.4, 39.0, 29.4 |
| previous own | 429, 459, 555 | 39.6, 47.0, 61.4 | 179, 203, 374 | 12.6, 24.2, 23.2 |
| Boost double | 729, 727, 392 | 68.3, 68.7, 46.3 | 249, 239, 282 | 79.9, 48.8, 25.4 |

**What this says.** The absolute error falls from 216-1059 to 2.8-3.8 ulp of 1,
ulpL from up to 6937 to below 20, the log versions to 10-22 ulp of |log f|, on
CPU and GPU; the remaining relative error in the deep tails is the exp of a log
of size |log f|. Faster than the previous version and than Boost on the CPU; on
the GPU 2-4 times slower than the previous version (more paths in a warp, the
non-inlined call) and faster than Boost except for large shapes. Chosen: this
version. Tests pass on CPU and GPU (10 cases, 144 assertions).

## Own inverse incomplete gamma and beta

**What.** include/laplace/math/incomplete_inverse.hpp, namespace laplace::math::own with wrappers in laplace::math:
- gamma_p_inv(a, p), gamma_q_inv(a, q) give x with P(a, x) = p, or Q(a, x) = q.
- gamma_p_inv_logp(a, l), gamma_q_inv_logq(a, l) do the same from l = log p or log q, for probabilities below the double range.
- ibeta_inv, ibetac_inv, ibeta_inv_logp, ibetac_inv_logq are the analogues for I_x(a, b) and 1 - I_x(a, b).
- The own:: beta versions return the pair (x, 1 - x), each with its own relative precision: 1 - x = 2.5e-264 is returned even though x rounds to 1.

**How.** The side solved is always the smaller probability, from its own value. The equation is g(v) = log S(x(v)) - log s = 0, with v = log x for the gamma and logit x for the beta; in these variables log S is concave for every a, b > 0, so Newton's iterates approach the root monotonically after the first step and a bracket from the signs of g bisects the rest. Derivatives closed-form: D = x f(x) (gamma) or x (1 - x) f(x) (beta), f the density; g' = +-D/S, g'' = g' h, g''' = g' (h^2 - c - g' h), with h = a - x - g', c = x (gamma) and h = a (1 - x) - b x - g', c = (a + b) x (1 - x) (beta); log D without cancellation for shapes >= 10 (Stirling's Gamma*, log1pmx). Halley's correction when 1 + dv h / 2 > 1/2. Stopping: predicted error after the step below eps/16 (K |dv|^3 for Halley, K = |h^2/12 + (c + g' h)/6|; |h| dv^2 / 2 for Newton), which cut mean evaluations from 2.4 to 1.3 (gamma) and 4.4 to 2.5 (beta); step below half an ulp of x or 4 eps; the forward function's noise floor (|g| < 1e-6 and no further shrinking, or a bracket closed by the noise); 64 steps. Starting values: gamma, DiDonato and Morris (1986) equations 21-25 and 31-36, adapted from Boost's find_inverse_gamma and rewritten from log p and log q; beta, Abramowitz and Stegun 26.5.22 with DiDonato and Morris' equation 32 as a logit for a, b >= 1, otherwise Numerical Recipes' power-law tails in logs. Roots below 2^-1075 return 0.

**Measurement.** mpmath at 200 bits: gamma_p_inv.csv, ibeta_inv.csv (incomplete.py); from tools/reference/inverse.py gamma_q_inv, ibetac_inv, *_tail (p log-uniform in [2^-1074, 2^-1]), *_logp and *_logq (-log p log-uniform in [2^-10, 2^17]), *_c (exact 1 - x on rows with x > 1/2). Shapes: gamma a in [2^-10, 2^15), beta a, b in [2^-8, 2^13). Error ulpC = |x - x*| / max(ulp(x*), dx), dx = ulp(s) / |ds/dx| the change of x* caused by one ulp of the input s; tools/reference/backward_inverse.py gives the backward error |F(y) - s| in ulps of s, normalized the same way, and agrees to two digits. Timing on 2^16 arguments; gcc 14.2 OpenMP 8 threads; CUDA 13.1 compiled for sm_89.

| file | own ulpC CPU: 99%, max | own max CUDA | Boost double: 99%, max; non-finite | ns own: gcc 1 thread, OpenMP 8, CUDA | ns Boost: same |
|---|---|---|---|---|---|
| gamma_p_inv | 23, 52.2 | 52.2 | 4.85, 7.9e256 (1.1e248 at subnormal p); CUDA 99% 7.4e3 | 750, 102, 83 | 532, 70, 69 |
| gamma_p_inv_tail | 255, 657 | 657 | 4.5e15, 4.5e15 | 685, 88, 22 | 679, 90, 35 |
| gamma_q_inv | 7.2, 25 | 18.1 | 5.15, 12 (CUDA 6.8e3, 4.6e4) | 755, 113, 83 | 533, 71, 73 |
| gamma_q_inv_tail | 2.17, 10.1 | 15.2 | 3.83, 13.4 (CUDA 1.3e4, 5.9e4) | 696, 99, 20 | 492, 66, 41 |
| gamma_p_inv_logp | 6.7, 39 | 39 | no Boost version | 751, 107, 84 | |
| gamma_q_inv_logq | 4.9, 11.1 | 11.1 | no Boost version | 747, 104, 72 | |
| ibeta_inv (1 - x) | 49, 483 (694, 2160) | 381 (2880) | 185, 4.4e15; 1 NaN | 1938, 292, 159 | 2208, 297, 326 |
| ibeta_inv_tail (1 - x) | 298, 827 (383, 423) | 827 (513) | 4.3e67, 1.6e139; 11 NaN or throw | 1413, 206, 53 | 7089, 1030, 1648 |
| ibetac_inv (1 - x) | 291, 3250 (53, 150) | 4620 (87) | 59, 1.4e8 | 1947, 286, 162 | 2253, 335, 293 |
| ibetac_inv_tail (1 - x) | 14.4, 388 (411, 918) | 1360 (918) | 847, 1.6e14; 11 | 1487, 213, 52 | 7092, 1066, 1679 |
| ibeta_inv_logp (1 - x) | 322, 1310 (267, 646) | 1220 (667) | no Boost version | 2078, 277, 165 | |
| ibetac_inv_logq (1 - x) | 71.5, 1080 (604, 1620) | 1180 (1620) | no Boost version | 2023, 280, 145 | |

On one MSVC host thread: gamma own 330 ns against Boost 250; beta own 720 ns against Boost 890.

**What this says.** (1) The own inverses are correct over the whole range, with no non-finite or wrong result on about 28000 rows, on CPU and GPU; Boost in double returns 1.1e248 for subnormal p, and NaN, exceptions (which kill a host program despite the ignore_error policy) or errors up to 1e139 for the beta. (2) Where Boost works it is more accurate on the gamma side; the own error is that of the forward functions: gamma_inc forms P as exp(log P), so the inverse solves P to about |log p| ulps; the log-p versions are at 6.7 (99%) and 39 (max) ulps of log p; beta_inc is not smooth at the 7e-13 level where one shape is small and the other large, e.g. (a, b) = (0.005, 5104), and the inverse stops at that noise. (3) Speed: on the CPU the gamma inverse is about 1.4 times slower than Boost (1.3 evaluations of gamma_inc at 260 ns); on CUDA it is faster in the tails; the beta inverse is faster than Boost everywhere.

**Open.** Computing -a phi in double-double inside own::gamma_inc (and the analogous term in own::beta_inc) would bring the inverses to a few ulps. Shapes above 2^15 (gamma) and 2^13 (beta) only spot-checked (a = 1e6, 1e8; (a, b) = (1e6, 1e6), (1e7, 3) give plausible values). Better beta starting values for one small and one large shape (Temme's method 3) not written. In the first measurement the CUDA build targeted sm_75 only (CMake's default), so the driver recompiled the PTX for the sm_89 card at the first launch of each kernel: more than 10 minutes for math_inverse, and the cache reached 289 MB, past its 256 MB default; this is why the CUDA build sets CMAKE_CUDA_ARCHITECTURES to the card's architecture.

**Remeasured after the new incomplete beta (gcc, OpenMP 12).** The
gamma side is unchanged. On the beta side the inverse inherits the smaller
noise of the new beta_inc: max ulpC ibeta_inv 118 (was 483), its 1 - x 22.2
(was 2160); ibetac_inv 11.8 (was 3250), 1 - x 58 (was 150); ibeta_inv_logp 18.6
(was 1310), 1 - x 10.2 (was 646); ibetac_inv_logq 8.9 (was 1080), 1 - x 16.3
(was 1620). The deep tails (ibeta_inv_tail 813, ibetac_inv_tail 1 - x 1040) are
where the inverse solves P to about |log p| ulps. The CUDA build now targets
sm_89, the card's architecture. Chosen: the own inverses in
`laplace::math`.

## The normal block: erfc, erfcx, Phi, log Phi, its derivatives, the quantile

**What.** `include/laplace/math/normal.hpp`, Laplace's own, built on
erfcx(x) = exp(x^2) erfc(x), which neither underflows nor cancels:
- `erfcx`: Boost.Math's rational approximations for erfc in double (erf_imp,
  53 bits, Boost licence) without their factor exp(-z^2), on [0, 0.5), [0.5,
  1.5), [1.5, 2.5), [2.5, 4.5), [4.5, 28); above 28 the asymptotic series to
  nine terms; for x < 0, 2 exp(x^2) - erfcx(-x) with x^2 exact.
- `erfc(x)` = exp(-x^2) erfcx(x) with x^2 carried exactly (Dekker's product on
  the host, the FMA on the device); 2 - erfc(-x) below zero.
- `Phi(x)` = exp(-x^2/2) erfcx(-x/sqrt 2)/2 for x < 0, 1 minus the same above,
  with x^2/2 carried exactly: the direct erfc(-x/sqrt 2)/2 rounds the argument,
  a relative error of about x^2 eps in the tail.
- `log_Phi(x)` = -x^2/2 + log(erfcx(-x/sqrt 2)/2) for x < 0 (no underflow down
  to x = -1.9e154), log1p(-Phi(-x)) above.
- `log_Phi_d1` .. `log_Phi_d5`, one function per order. g1 = R = phi/Phi =
  sqrt(2/pi)/erfcx(-t/sqrt 2). For t >= -1 the recursion R' = -R(t + R); below,
  the route of statmodels7, an earlier R package by the same author (its files
  pt_logphi.h, pt_enet_logz.h): the cumulants kappa_k of
  the density proportional to exp(-x s - s^2/2) on s > 0, x = -t, with
  g1 = x + kappa_1, g2 = -1 + kappa_2, g_k = kappa_k, from the moment ratios by
  the downward recursion r_j = j/(x + r_(j+1)). Changed from statmodels7: the
  ratios scaled by x (rho_j = x r_j = j/(1 + rho_(j+1)/x^2), so the moments stay
  of order j! for every x), the start at the asymptotic root instead of 0, and
  K = k + 20 + ceil(300/x^2) steps instead of 2000, 300 or 80.
- `Phi_inv(p)`: Boost's erfc^-1, then for p < 0.1 one Newton step on log Phi;
  `Phi_inv_logp(l)`, l = log p: for p >= 1/2 through q = -expm1(l); otherwise a
  start from Boost (l > -700) or from x^2 = -2l - log(-4 pi l), then Newton on
  log Phi until the step is below 4 eps |x|.

**How.** References `tools/reference/normal.py` (mpmath at 200 bits): erfcx,
Phi, log_Phi (x down to -2^500), log_Phi_d1..d5 (t from -2^60 to 38; the
recursion R^(n+1) = -(t R^(n) + n R^(n-1)) - sum C(n,i) R^(i) R^(n-i) at a
precision raised by 12 bits per binary order of |t|, checked against 80 more
bits), Phi_inv_logp (-l log-uniform in [2^-60, 2^1000]); erfc.csv and qnorm.csv
from incomplete.py. Bench `benchmarks/math_normal.cpp`; gcc 14.2 with OpenMP 8
threads, CUDA 13.1 on the RTX 4070 SUPER. Quantiles in ulpC = error /
max(ulp(x), dx), dx the change of x caused by one ulp of p or log p.

| function | own: max error (CPU; CUDA) | alternatives | ns/call own: gcc 1 thread, OpenMP 8, CUDA | alternatives ns: gcc 1 thread, CUDA |
|---|---|---|---|---|
| erfc | 3.46 ulp | platform 2.15 (CUDA 2.68); Boost 3.20 | 21.8, 2.92, 0.72 | platform 39.2, 0.28; Boost 61.2, 1.35 |
| erfcx | 1.78 ulp | exp(x^2) erfc(x): 498 ulp, infinite above 26.6; CUDA `erfcx` 2.48 | 8.3, 1.11, 0.87 | CUDA 0.28 |
| Phi | 3.86 ulp (ulp1 0.86) | erfc(-x/sqrt 2)/2, platform or Boost: 1658 ulp near x = -37; CUDA `normcdf` 3.16 | 24.2, 3.20, 1.40 | platform 38.9, 0.28; Boost 53.9, 1.41; CUDA 0.32 |
| log Phi | 4.08 ulp (CUDA 3.79) | log of the direct formula: 3.6e12 ulp and -inf below x = -38.5 (also with CUDA's normcdf) | 52.6, 7.51, 1.58 | 82.7, 0.82 |
| Phi_inv | ulpC: p < 1e-300 0.82, to 1e-10 0.99, to 0.1 1.58, centre 2.32 | Boost 1.72, 2.92, 2.38, 2.32; CUDA `normcdfinv` 2.21, 4.14, 3.24, 2.96 | 37.4, 5.25, 2.82 | Boost 24.6, 1.32; CUDA 0.27 |
| Phi_inv_logp | ulpC <= 2.76 everywhere | Boost on exp(l): wrong or infinite where exp(l) underflows | 132, 19.0, 5.96 | Boost 64.5, 1.60 |

Derivatives of log Phi, max relative error (ulp) below t = -1 and max error in
ulps of max(|g|, 1) above (where g_k crosses zero), CPU (CUDA similar, within
30 per cent):

| order | t < -20 | [-20, -4) | [-4, -1) | [-1, 0), ulp1 | [0, 5), ulp1 | ns/call own: gcc 1 thread, CUDA | statmodels7 port | recursion everywhere |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.50 | 0.52 | 1.19 | 2.53 | 1.67 | 111, 8.4 | same | same |
| 2 | 0.50 | 0.54 | 1.52 | 6.14 | 2.39 | 117, 8.5 | same | 5e36 ulp at t = -2^60 |
| 3 | 7.3 | 7.1 | 14.6 | 11.8 | 4.12 | 126, 8.6 | 5.2-13 below -1 | 7e109 ulp |
| 4 | 10.7 | 10.7 | 25.3 | 29.1 | 10.2 | 134, 8.8 | 9-22 | 2e146 ulp |
| 5 | 14.1 | 18.1 | 81.3 | 92.3 | 24.7 | 144, 9.0 | 13-55; 905 ns, 81.6 ns on CUDA | 3e182 ulp |

**What this says.** (1) Every own function keeps its relative accuracy in the
tails, where the direct formulas lose it (Phi 1658 ulp, log Phi -inf, the
quantile from log p undefined). (2) Where both work, the own versions are within
1-2 ulp of the platform's, Boost's and CUDA's, better or worse by function; on
the GPU CUDA's erfcx, normcdf and normcdfinv are 3-10 times faster (about 1 ns
per call less). (3) Below t = -1 the derivatives are as accurate as
statmodels7's, 6-8 times faster on the CPU and 9 times on the GPU. Tried and
dropped: the closed forms of the cumulants in powers of mu_1 (2-3 times less
accurate than the recursion kappa_n = mu_n - sum C(n-1, i-1) kappa_i mu_(n-i));
the upward recursion from M_0 for x <= 1 (no gain) and up to x = 2 (1.5e4
ulp); fewer downward steps (K = k + 20 + 200/x^2 gives 1.25e3 ulp near
t = -2). (4) In [-1, 5) the recursion from R loses up to 92 ulps of 1 for the
fifth derivative (2e-14 absolute), as statmodels7's does; with R exact from
mpmath it loses a quarter of that, so half of the error is the recursion's own.
Above t = 30 the derivatives are below 1e-190 and R is subnormal near 38, so
their relative error is large but their absolute error nil. (5) The Newton
step in Phi_inv helps only in the tail (p < 0.1); at the centre it made the
result worse (3.2 ulpC), because there log Phi is no more accurate than Boost's
quantile.

**Chosen.** The own versions on every backend, in the
wrapper `laplace::math::{erfc, erfcx, Phi, log_Phi, log_Phi_d1..d5, Phi_inv,
Phi_inv_logp}`: one algorithm and one error profile on CPU and GPU; the choice
per backend (CUDA's erfcx, normcdf, normcdfinv) is reopened together with
lgamma's when a profile of a real kernel shows these functions weigh. erfc is
the own one (3.5 ulp, against the platform's 2.1-2.7, and faster on the CPU).
Tested in `tests/math/test_normal.cpp` (3 cases, 31 assertions) on the gcc/OpenMP
and CUDA builds.

**Open.** Piecewise Chebyshev approximations of the derivatives on [-1, 5]
would bring the central region to a few ulps. The classical Mills ratio
M(x) = (1 - Phi(x))/phi(x) = sqrt(pi/2) erfcx(x/sqrt 2) is not in the wrapper
until something needs it.

**Further details of the normal block.** The rational approximations of erfcx are used as follows. On [0, 0.5): erfcx(z) = e^(z^2) (1 - erf(z)) with erf(z) = z (Y + P(z^2)/Q(z^2)). On [0.5, 1.5), with u = z - 0.5, on [1.5, 2.5), with u = z - 1.5, and on [2.5, 4.5), with u = z - 3.5: (Y + P(u)/Q(u))/z. On [4.5, 28.0), with u = 1/z: (Y + P(u)/Q(u))/z. The ninth term of the asymptotic series is below 1e-19 for z >= 28. For x < 0, e^(x^2) is computed from the exact product a^2 = h + l (Dekker's product on the host, the FMA on the device), with a = -x, as 2 (e^h + e^h l) - erfcx(a) for a >= 0.5 and as 2 e^(a^2) - erfcx(a) for a < 0.5. In the derivatives of log Phi below t = -1, the Newton iteration starts from the asymptotic root rho = 2 (K + 1)/(1 + sqrt(1 + 4 (K + 1) w)), with K = k + 20 + min(ceil(300/x^2), 20000) steps (the cap applies to the ceil term, so K <= k + 20 + 20000; w and k are those of `include/laplace/math/normal.hpp`). The normal references agree to 2^-120 relative; the quantile references are `tests/data/reference/qnorm.csv` and `tests/data/reference/Phi_inv_logp.csv`, the latter with -l log-uniform in [2^-60, 2^1000].

## Shape derivatives of log P(a, x) and log Q(a, x), orders 1 to 6

**What.** `include/laplace/math/incomplete_gamma_deriv.hpp`;
`laplace::math::log_gamma_p_da1..5` and
`log_gamma_q_da1..5`, one function per order. With d = log(t/x) both sides are
written relative to x^a e^-x, the lower one through P(a, x) = P(a+1, x) +
x^a e^-x / Gamma(a+1) (no pole of psi(a) at small a), so a enters only through
e^(a d) and the derivatives of the log-integral are the cumulants kappa_k of d.
d log P/da = log(x/a) - (psi(a+1) - log a) + kappa_1 has terms of one sign
on the lower side, d log Q/da = log(x/a) + (log a - psi(a)) + kappa_1 on the
upper side; psi - log a without forming the two apart. The less probable side
is computed (chosen by x < a - 1/3, without calling gamma_inc); the other side
is log(1 - S), composed through Bell polynomials.

**Routes, in order.** (1) Temme's uniform expansion as a Taylor polynomial in
the increment of a (coefficients C_0..C_9 adapted from Boost's
igamma_temme_large, Boost licence; eta(mu) from a 91-term series written by
mpmath; log erfcx from normal.hpp) for a >= 20, |x/a - 1| < 0.3 (wider for
larger a). (2) The lower series sum prod x/(a+j), differentiated term by term
(300 terms on the host, 60 and a < 2 on the device). (3) A closed form for the
upper side at x < 3/2, a < 2. (4) The upper asymptotic series for x >= 36
(host). (5) Legendre's continued fraction by modified Lentz in Taylor form
(host). (6) Otherwise a double-exponential quadrature (exp-exp rule, h = 1/8,
about 65 nodes, tables on the host and in constant memory on the device).
Rejected: the continued fraction differentiated through its recurrences (AS 187
extended; cancels, up to 1e12 ulp); h = 1/4 (1e4-1e11 ulp); the fraction on the
device (420-1060 ns against 54-110 for the quadrature).

**Measurement.** References `log_gamma_{p,q}_da{1..5}.csv` (tools/reference/incomplete_deriv.py,
1500 rows; integrals in log t, Taylor series of S composed with log and
log(1 - S), at 240 and 320 bits, checked against mpmath's gammainc). Bench
`benchmarks/math_gamma_deriv.cpp`; timing set a = 2^U(-3, 10), x = a e^(0.5 Z).
Errors as max min(ulpL, ulp1), CPU / CUDA:

| order | log P | log Q | central differences of gamma_inc |
|---|---|---|---|
| 1 | 4.0 / 4.3 | 4.8 / 4.8 | 6.7e5 |
| 2 | 4.4 / 4.4 | 5.5 / 7.3 | 3e12 |
| 3 | 8.8 / 10.8 | 13.9 / 13.9 | 3e16 |
| 4 | 23.5 / 23.0 | 22.1 / 23.0 | 6e20 |
| 5 | 197 / 171 | 62.9 / 67.9 | 9e23 |

ns per call, order 1 / order 5, mixed set (CUDA: minimum of 3 runs, the GPU was
shared with other work):

| | first version | after two rounds of optimization |
|---|---|---|
| log P, gcc 1 thread | 2217 / ~2800 | 647 / 1059 |
| log Q, gcc 1 thread | 2668 / ~2800 | 534 / 1192 |
| log P, OpenMP 8 | 292 | 105 / 148 |
| log P, CUDA | 188 / ~330 | 125 / 145 |
| log Q, CUDA | 329 | 141 / 195 |

By region on CUDA (order 5): a > 50 near the centre 33-34 ns, a < 1 41-90 ns,
far tails 52-140 ns; on the mixed set a warp that mixes routes runs all of them.

**What this says.** The cancellation-free form keeps relative accuracy in both
far tails (1-5 ulpL); differences lose everything from order 2. The complement
in the far tails carries about |log S| ulps (the error of exp(log S)). In the
bulk (1 <= a <= 1000) orders 4-5 reach about 3e3 ulpL where the values pass near
zero (before Temme 70-180), with the absolute error at most 63 ulp1. Speed:
2-4 times faster than the first version on the CPU; on CUDA the mixed set stays
above 100 ns because of divergence between routes.

**Chosen.** This version, with routes that differ by
backend (fraction and upper asymptotic series on the host only). Tested in
`tests/math/test_gamma_deriv.cpp` on the gcc/OpenMP and CUDA builds.

**Open.** A cheaper quadrature or fewer route boundaries for the device;
own::exp in the quadrature gives no gain on the new routes.

**Order 6 (2026-11-12).** `log_gamma_p_da6` and `log_gamma_q_da6`, on every
route of orders 1 to 5 and with the same helpers: the quadrature carries the
sixth central moment in Pebay's update (M6 += 6 M5 e + 15 M4 e^2 + 20 M3 e^3 +
15 M2 e^4 + W e^6 + w f^6, only when k = 6, where e is the shift of the old
points about the new mean, f that of the new point, W the old weight and w the
new one) and kappa_6 = mu_6 - 15 mu_4 mu_2 - 10 mu_3^2 + 30 mu_2^3 (mu_p the
central moments); its stopping test weighs a node by (1 + xi)^6 instead of
(1 + xi)^5 at order 6. The series, the fraction in Taylor form, the closed form
at small x (four more terms of the series of lambda at order 6) and Temme's
route carry their Taylor polynomials to degree 6; log erfcx in Temme's route
gets g_6 = -g_5 (t + 2R) - 4 g_4 (1 + 2 g_2) - 6 g_3^2 (R = phi/Phi, checked
against SymPy's sixth derivative of log Phi); the complement and the 1/a pole
of S = Q are generic in the order. Orders 1 to 5 give the same values as
before except where the fraction was moved (below).

References `log_gamma_{p,q}_da6.csv` (kind `gamma6` of
tools/reference/incomplete_deriv.py): the 1500 points of orders 1 to 5 (their
orders 1 to 5, recomputed with the series carried to degree 6, agree with the
existing files to the double rounding) plus 400 points at the route boundaries
(x near 1/4, 1, 3/2, 3 and 36; a near 2, 20, 300, 1000 with |x/a - 1| near
0.3, 0.4, 0.5; x near a - 1/3; a < 2^-4; far tails), all 1900 accepted (240 and
320 bits agree to 2^-100). Order 5 on the 400 edge points is in
`log_gamma_{p,q}_da5_edges.csv`, to compare the two orders on the same points.

A defect found on the edge points, and repaired for every order: the
continued fraction in Taylor form loses accuracy just above x = a. Against the
quadrature (h = 1/8; bench mode `fracgrid`, log Q, a = 1.6 to 30, 64 points per
cell of z = (x - a)/sqrt(a)), for z in [-0.25, 0) the difference was 2-7 ulp1 at
order 2, 560-1500 at order 5 and 5e3-1.4e4 at order 6; for z in [0.5, 0.75) at
most 6 ulp1 at order 5 and 22.5 at order 6; for z >= 0.75 at most 1.9 and 4.
On the edge points the route reached 553 (P) and 629 (Q) ulp1 at order 5 and
9.4e3 at order 6, at a = 4.05, x = 3.72; the 1500 common points had missed
the band. The fraction is now used only for x >= a + 0.75 sqrt(a) (host only,
as before; the quadrature takes the band). After the repair, max
min(ulpL, ulp1) per route, gcc/OpenMP, order 5 (1500 points; edges 400 points)
and order 6 (1900 points); "+c" is the complement log(1 - S):

| route | P order 5 | P order 5 edges | P order 6 | Q order 5 | Q order 5 edges | Q order 6 |
|---|---|---|---|---|---|---|
| Temme | 0.0009 | 0.29 | 0.10 | 0.004 | 8.8 | 14.5 |
| Temme +c | 0.006 | 0.20 | 0.18 | 0.004 | 0.03 | 0.002 |
| lower series | 197 | 155 | 771 | | | |
| lower series +c | | | | 40.9 | 30.3 | 96.3 |
| closed form, small x | | | | 4.7 | 6.7 | 5.2 |
| upper asymptotic series | | | | 2.5 | 3.4 | 1.9 |
| upper asymptotic series +c | 0.0001 | 0.0004 | 0.0001 | | | |
| fraction in Taylor form | | | | 2.2 | 0.58 | 2.6 |
| fraction +c | 0.64 | 1.6 | 2.9 | | | |
| quadrature, upper side | | | | 2.3 | 2.7 | 1.5 |
| quadrature, upper side +c | 7.8 | 8.4 | 33.3 | | | |
| all | 197 | 155 | 771 | 40.9 | 30.3 | 96.3 |

Worst points of order 6: P at a = 0.0139, x = 1.045 (lower series), Q at
a = 2.04, x = 1.56 (complement of the lower series), Temme Q at a = 20.05,
x = 25.73 (14.5 ulp1, near the boundary |x/a - 1| = 0.3 at a = 20; 8.8 at
order 5). The growth from order 5 to order 6 is the cancellation between
kappa_k and psi^(k-1)(a + 1) at small a: psi^(5)(1.014) = 112.5 against
psi^(4)(1.001) = -24.7, with results of order 1-3, so the same relative error
of kappa_k (about 6 ulps at both orders) is about 4 times more ulp1 at order
6. The quadrature is no better there (quad8 1117 ulp1 on the lower-series
points of order 6). Test bounds (tests/math/test_gamma_deriv.cpp): 1160 (P) and
145 (Q), gcc only; the CUDA build, whose routes differ (no fraction, no upper
series, the lower series only for a < 2 and 60 terms), is still to be measured.

ns per call, mixed timing set, orders 1 / 5 / 6, gcc, Serial (1 thread) and
OpenMP (4 threads), taken while a CUDA build (nvcc, ptxas) occupied the
machine: log P 685 / 1044 / 1283 and 179 / 281 / 338; log Q 687 / 1178 / 1473
and 194 / 318 / 409. By region (Serial, order 5 / 6): a < 1 P 300 / 378, Q 845 /
1056; centre 1-50 P 1259 / 1565, Q 1233 / 1542; centre a > 50 P 794 / 1003,
Q 802 / 1017; far lower tail P 375 / 437, Q 697 / 898; far upper tail
P 1211 / 1584, Q 900 / 1127. Order 6 costs 20-30 per cent more than order 5.

## Shape derivatives of log I_x(a, b) and log(1 - I_x(a, b)), orders 1 to 5

**What.** `include/laplace/math/incomplete_beta_deriv.hpp`;
`laplace::math::log_ibeta_d1..5` and
`log_ibetac_d1..5`, one function per order, returning `shape_deriv<K>` with
c[j] = d^K / da^(K-j) db^j. The derivatives of the log-integral are the joint
cumulants of (log T, log(1 - T)) for T a Beta(a, b) truncated to the side; the
log-beta terms are polygamma differences computed term by term (b/a reaches
2^-20), sharing one shift for all orders; 1 - I_x(a, b) = I_(1-x)(b, a), so every
side is a lower side; the other side is composed by the moment and cumulant
recursions (from S/b, so that the pole of log S at b -> 0 does not cancel).

**Candidates.** (1) Quadrature: double-exponential rays from the end point or
the mean, in log t towards 0 and -log(1 - t) towards 1 (in logit t the
integrand has poles of order a + b at u = +-i pi, convergence e^(-2.9/h)),
Takahasi-Mori map with h = 1/5, tables; on rays dominated by their slope
(c = kappa/(2 g^2) <= 1e-2) Gauss-Laguerre with 16-32 nodes. (2) Series:
DiDonato-Morris' BPSER differentiated term by term (for x <= 1/2 with b <= 1 or
b x <= 0.7, up to x = 0.8 when b <= 1), the quadrature elsewhere. Temme's BASYM
in bivariate Taylor form was estimated at 1e5-1e6 flops per call (the quadrature
about 1e4) and not written.

**Measurement.** References `log_ibeta{,c}_d{ij}.csv` (519 rows, all
components, from integrals in logit t). Errors in ulpN = ulps of max(|f|, N_ij),
N_ij = |f_K0|^(i/K) |f_0K|^(j/K) (the Cauchy-Schwarz scale of a mixed component;
ulp for the pure ones), max over both sides and all components, gcc (CUDA
within 1.9 times):

| order | series (chosen) | quadrature alone | central differences |
|---|---|---|---|
| 1 | 1580 | 1580 | 9e9 |
| 2 | 1340 | 1.9e4 | 3e17 |
| 3 | 1620 | 1.8e6 | 4e20 |
| 4 | 1370 | 1.3e8 | 2e23 |
| 5 | 1950 | 1.7e10 | 5e25 |

That is 4e-13 relative or better outside the deep tails; there the ~1e3 ulps
are exp(log(1 - I)) inside the forward function (ulpL 36-435).

ns per call, mixed set a, b = 2^U(-2, 8), x uniform, orders 1-5:

| | first version | after two rounds |
|---|---|---|
| gcc 1 thread | 5300-7000 | 1740-2970 (own::exp/log: 1420-2430) |
| OpenMP 8 | 740-955 | 235-397 |
| CUDA, mixed | 330-668 | 250-519 (sorted by (a, b): 173 / 370 at orders 1 / 5) |
| CUDA, homogeneous regions | | 37-140 (order 1), 98-495 (order 5) |

The levers that paid: expm1mx without divisions, tabulated maps, series for
log1p and expm1 at small arguments, the composition by recursions (955 to 281
ns and smaller errors), no beta_inc, shared polygamma, Gauss-Laguerre on the
CPU (14-18 per cent; 10 per cent slower on CUDA).

**Chosen.** The series route, Gauss-Laguerre on the host
only (`quad_par::gauss_device = false`), the platform's exp and log. Tested in
`tests/math/test_beta_deriv.cpp` on both builds.

**Open.** Mixed components 1e4-1e8 times smaller than the pure ones with one
shape below 1: relative errors up to 2e-8 (ulpN at most 0.2). Both shapes below
1 with x near 0 or 1 rely on the quadrature, whose own-shape pole is not
resolved (a by-parts form would remove it). On the CPU about 41-48 nodes per
double-exponential ray remain on curvature-dominated rays; on CUDA, divergence
between series and quadrature. own::exp/log in the quadrature: 15-20 per cent
on gcc, open with the choice of elementary functions.

**Quadrature nodes of the shape derivatives.** The nodes are tabulated in constant memory on the device, with spacing h = 1/5 on the double-exponential line and 43 or 56 positive quadrature points for inner points, 15 for the right-tail split and 65 for Gauss-Laguerre when x = 1.

## Owen's T and log F of the standard skew normal, with derivatives in (z, alpha) to order 5

**What.** `include/laplace/math/owen.hpp`:
`laplace::math::owens_t(h, a)`, `log_sn_cdf(z, alpha)` and
`log_sn_cdf_d1..5(z, alpha)` (one function per order, `shape_deriv<K>` with
c[j] = d^K / dz^(K-j) dalpha^j); the survival is F(-z; -alpha). With z = -h <= 0,
eta = h^2/2, beta = alpha h, every case uses F(-h; alpha) = exp(-eta) G / pi,
G = int_alpha^inf exp(-eta b^2) / (1 + b^2) db, an integral of a positive
function, in one of three forms that neither cancel nor underflow: G =
exp(-eta alpha^2) Jbar (alpha > 0; the quadratic added in closed form); G0 - T~
with G0 = (pi/2) erfcx(h/sqrt 2) (|beta| small); 2 G0 - exp(-eta a^2) Jbar(h, a)
(alpha = -a < 0). Jbar by fixed Golub-Welsch rules chosen by the distance of the
poles (Gauss-Laguerre 20, half-range Gauss-Hermite 28, Gauss-Legendre pieces up
to 96 nodes). Derivatives from node moments as bivariate series and their
logarithm, or (|z| < 1 with small beta, and z > 0) by composing F's elementary
derivatives (dF/dz = 2 phi(z) Phi(alpha z), dF/dalpha = -exp(-(1 + alpha^2) z^2 / 2)
/ (pi (1 + alpha^2))) with the ratios kept as exp(-S) m. Owen's T: a <= 1,
a h <= 2: exp(-eta) T~ / (2 pi); a <= 1, a h > 2: (Q(h) - F(-h; a))/2; a > 1:
(Q(h) erf(a h / sqrt 2) + F(-a h; 1/a))/2, a sum of positive terms.

**Measurement.** References from tools/reference/student_owen.py: owens_t.csv
(3006 rows, by double-exponential quadrature of the definition), log_sn_cdf*.csv
(800 rows; the value as a sum of positive terms, the derivatives of F
elementary with Hermite polynomials, composed with log; at 280 and 360 bits).
gcc / CUDA:

| | own | alternatives |
|---|---|---|
| owens_t, max ulp (ulpL) | 4.44 (2.24) / 4.08 (1.95) | Boost (host): 1350 ulp at h = 35, a = -1122; plain quadrature with the usual identity: 6.6e12 |
| log_sn_cdf | 5.0 (3.6) / 5.0 (3.6) | log(Phi - 2T): 8.9e15 ulp and 78 non-finite (34 with Boost's T) |
| d1, ulpN (ulpL) | 4.8 (4.3) / 5.1 (4.3) | elementary series everywhere: 9e15, 94 non-finite |
| d2 | 206 (24) / 198 (24) | |
| d3 | 2240 (501) / 2150 (481) | |
| d4 | 1500 (777) / 1490 (784) | |
| d5 | 1.16e5 (2.7e4) / 1.17e5 (2.8e4) | |

Order 5 by region (ulpN): lower tail alpha > 0 1.4e4, alpha < 0 1.3e3, upper
tail 100, centre 1.2e5. ns/call: owens_t 740 (gcc 1 thread), 106 (OpenMP 8),
25 (CUDA); log_sn_cdf 816, 114, 121; d1-d5 876-2485, 118-434, 114-236; Boost's
owens_t 153-239 on gcc.

**What this says.** Relative accuracy everywhere, including the deep tails and
the upper tail where log F = -S with S below 1e-300. The largest derivative
errors sit at zeros of components of order 3-5, where the problem is
ill-conditioned: one ulp of F moves them by 1.4e4-2.9e4 ulps (mpmath, e.g. d50 at
z = -0.198, alpha = 1.30); the observed errors match F and f known to 2-8 ulps.
On gcc the cost is the platform's exp (about 30 ns, 16 calls in T~ alone).

**Chosen.** This version on both backends. Tested in
`tests/math/test_owen.cpp` on both builds.

**Open.** Double-double F and f would lower the order 3-5 errors at the zeros;
own::exp would cut the gcc cost (the choice of elementary functions is open);
log_sn_cdf on CUDA (121 ns against 25 for owens_t) is probably divergence
between regions, not profiled.

**Lower tail quadrature.** In the representation of the lower tail the interval length is at most y + D, the increase of t is at most 20 per step, up to t = 60, and the rule uses up to 96 nodes.

## Student t distribution function with real nu: log T_nu(z) and its derivatives in (z, nu) to order 5

**What.** `include/laplace/math/student.hpp`:
`laplace::math::log_student_cdf(z, nu)` and
`log_student_cdf_d1..5(z, nu)` (`shape_deriv<K>`, c[j] = d^K / dz^(K-j) dnu^j);
the survival is T_nu(-z). The value by own::beta_inc (T = I_x(nu/2, 1/2)/2,
x = nu/(nu + z^2), with the complement form for x > 1/2). The derivatives by
quadrature, every function carried as a truncated bivariate Taylor polynomial
in (dz, dnu): centre (T(-|z|) > 1/5), T = 1/2 + int_0^z t, nu-derivatives by
16-point Gauss-Legendre in s = sqrt(nu) sinh(tau) on pieces [0, 1], [1, 3],
[3, 9], ...; Gaussian lower tail (z^2 < 4 nu), s = z - u, log T = c + k(z) +
log int exp(Dk) with every coefficient of Dk an exact difference; power lower
tail (z^2 >= 4 nu), s = z e^v (for nu < 1 at fixed w = nu v, with the pole of
c(nu) written out); upper side log(1 - S). Tail nodes by the Takahasi-Mori map,
h = 1/4, tabulated. c(nu) from an asymptotic series for nu/2 >= 15 and a
recurrence below (1-4 ulp against mpmath).

Why not the direct routes: d^2 log T / dz^2 = (t/T)(d log t/dz - t/T) loses a
factor z^2 in the Gaussian tail and nu in the power tail; the composition from
the beta shape derivatives (a = nu/2, x(z, nu)) loses a factor nu as nu grows,
because the nu-derivatives of the normal limit are O(nu^-2) formed from O(nu^-1)
terms. A 1/nu expansion around log Phi has terms growing like (z^4/nu)^k and was
not written.

**Measurement.** References log_student_cdf*.csv (tools/reference/student_owen.py:
780 rows, mpmath.diff of the betainc form at 280 and 360 bits, again at 440 and
560 where they disagreed; 20 rows dropped, |z| >= 25 with nu in 4600-1e6).
Errors in ulpN (ulp1 for the pure z-components of orders 3 and 5 with |z| < 1):

| order | quadrature, gcc / CUDA | ulpL | beta composition | central differences |
|---|---|---|---|---|
| 1 | 432 / 432 | 7 | 2.8e7 (68 for nu < 1) | 1e16 |
| 2 | 525 / 525 | 23 | 1.7e12 | 1e23 |
| 3 | 337 / 319 | 33 | 1.3e18 | 5e28 |
| 4 | 506 / 1480 | 281 | 4.7e23 | 5e32 |
| 5 | 470 / 470 | 96 | 8.7e28 | 4e35 |

The value: own::beta_inc 188 ulp (10.8 ulp1), the quadrature 357 ulp, both in the
upper tail (the error of exp(log S), |log S| up to 430). Deep tails missing from
the references: on 2000 points the quadrature agrees with itself at h/2 to
1-50 ulpN, the value with beta_inc to 4 ulp, and at 8 points with independent
300-bit mpmath integrals to 1-3 ulp; the beta composition is off there by 1e7 to
3e29. ns/call, nu = 2^U(-2, 10), z = 3 Z: derivatives 2350 / 2810 / 3530 / 4820 /
9300 (gcc 1 thread, orders 1-5), 365-1900 (OpenMP 7), 216-727 (CUDA); value by
beta_inc 535 and 88.

**Chosen.** The value by beta_inc, the derivatives by
the quadrature, on both backends. Tested in `tests/math/test_student.cpp` on both
builds.

**Open.** Speed (about 35 nodes per call, each with about 4 transcendentals and
a 21-coefficient polynomial exp and log; the region switch calls beta_inc,
0.5 us); nu below 2^-4 is not covered by the references.

**Nodes and conditioning.** The nodes and weights of the quadrature rules are written by mpmath at 200 bits. One double-exponential rule has step h = 0.25, scale sigma = 2/(nu + 1) and alpha = 0.75 (node table written by mpmath at 200 bits); a second has h = 0.25, sigma = 1/(nu + 1) and alpha = 0.75, from the same node table. At large nu the fifth derivative of log T_nu(z) in z at z = 0 is about 2000 times smaller than its terms, and the third derivative has a zero near nu = 6.

## Von Mises Bessel functions: log I0, A = I1/I0 and its inverse, derivatives to order 6

**What.** `include/laplace/math/bessel_i.hpp`, the Bessel functions the von
Mises distributions use: `log_bessel_i0(k)`, `log_bessel_i0_scaled(k)`
(log I0(k) - k), the ratio `bessel_i_ratio(k)` = A(k) = I1(k)/I0(k) with
`bessel_i_ratio_d1..d6`, its inverse `bessel_i_ratio_inv(rho)` = K(rho) and
`bessel_i_ratio_inv_d1..d6(kappa)` = K^(n)(rho) at rho = A(kappa). Orders to
six because the fifth derivative of the expected information -A' (von Mises
in kappa) or -K' (in rho) needs them. The derivatives of the inverse take
kappa, which the distribution in (mu, rho) computes once and which keeps the
information that rho loses near 1. New shared header
`detail/double_double.hpp` (sum, product, quotient).

**How.** A and K are ported from statmodels7's own `bessel_ratio.cpp` and
extended from order four to six. A: power series for k < 0.25; continued
fraction for 0.25 <= k < 30 (backwards in double for the value, forwards in
double-double for the derivatives, which follow from the Riccati identity
A' = 1 - A/k - A^2 differentiated n times, in double-double); asymptotic
series in u = 1/k for k >= 30, summed with Neumaier compensation (about fifty
terms above the last bit near k = 30) and corrected to first order for the
rounding of u, with u^M in double-double. K: Newton's method (in double until
the step is below 1e-9 k, then double-double for the last steps). K^(n): power
series in k for k < 0.25 and normalized asymptotic series
K^(n) = n! 2^n k^(n+1) (1 + sum_m e_(n,m) k^-m) for k >= 30, both with exact
rational coefficients (`tools/reference/bessel.py coefs`); in between, the
inverse function rule as a polynomial in r_j = A^(j)/A' evaluated in
double-double. log I0: Boost's rational approximations of I0 for double
(Holoborodko's forms, Boost Software License), rewritten in log form:
log1p(a P(a)) with a = k^2/4 below 7.75, log P(1/k) - log(k)/2 + k above (no
exponential, no overflow).

**Measurement.** References from `tools/reference/bessel.py` (mpmath, every
row at two precisions that must agree to 2^-110; none dropped): 4150 points in
k from 2^-60 to 2^45, concentrated around 0.5, 1, 20, 30, 50, 100; 4842 points
in rho for the inverse, 1 - rho down to 2^-53. The derivatives of A from the
Riccati recursion at 160 + (n + 2)(|log2 k| + 4) bits; those of K by reversion
of the Taylor series of A (independent of the closed forms in the library).
Max ulp, gcc (OpenMP) / CUDA; ns/call on one gcc thread, OpenMP with 16
threads, CUDA:

| function | own | alternatives | ns/call |
|---|---|---|---|
| log I0 | 2.31 / 2.15 | own series and asymptotic expansion 4.16; log of Boost's I0 9e15 near 0, overflows from k = 713 | 65 / 5.1 / 0.72 |
| log I0 - k | 3.0 / 3.0 | series 6.0 | 64 / 5.0 / 0.72 |
| A | 3.30 / 3.48 | Boost I1/I0 6.27, overflows from k = 713 | 60 / 5.6 / 3.8 |
| A' ... A^(6) | 2.04, 2.23, 2.00, 2.42, 2.08, 2.42 (both) | first port: up to 10.5 | 290-495 / 28-47 / 18-26 |
| K | 1.80 / 1.80 | | 609 / 54 / 39 |
| K' ... K^(6) | 1.99, 2.14, 2.31, 1.96, 2.54, 2.92 (both) | first port: 8, 22, 40, 82, 155, 4171 | 292-542 / 29-50 / 16-25 |

The first port (statmodels7's algorithm extended to order six, with the
polynomial in r_j in double everywhere and the asymptotic series summed
plainly) lost accuracy in three places, each found by splitting the errors by
regime: the polynomial cancels (its terms reach a hundred times the result at
order six, near k = 30 and above); the rounding of u = 1/k is amplified M + 1
times in u^M S_M(u); the forward sum of the asymptotic series accumulates one
rounding per term near k = 30. Switch points: the large-k switch at 20 costs
up to 1.7e6 ulp at order six (the asymptotic series is not yet accurate), at
40 or 60 it gains nothing and costs 20-50 per cent; the small-k switch at 0.25
instead of 0.5 lowers the worst errors from 4.0-5.4 to 2.0-2.9 ulp at the same
cost; truncating the continued fraction at floor(k) + 30 instead of + 20
changes nothing.

On the GPU the double-double arithmetic first lost up to 1.4e4 ulp at orders
4-6 (2.4 on the CPU): nvcc contracted a product and the sum after it into an
FMA after inlining, which breaks the exact identities of two_sum and
quick_two_sum. The double-double operations now go through the intrinsics
`__dadd_rn`, `__dsub_rn`, `__dmul_rn` on the device, which nvcc never
contracts; then CUDA matches the CPU to the digit.

**What this says.** Every function is within 3.5 ulp over the whole range on
both backends, and the derivatives of the inverse, which lost hundreds to
thousands of ulp in the straightforward form, are now as accurate as A's. The
cost is the continued fraction in double-double for 0.25 <= k < 30: about
0.3-0.6 us on one gcc thread (the host product is Dekker's split), about
20 ns on CUDA.

**Chosen.** As above, on both backends. Tested in `tests/math/test_bessel.cpp`
on both builds (bounds: measured error plus about half).

**Open.** The value of A in the continued fraction range is computed in double
(3.3-3.5 ulp); in double-double it would be at the last bit for about four
times the cost. The ratios I_j/I_0 of the von Mises cdf are in the next entry. Any other
double-double code on the device must use the intrinsics of
`detail/double_double.hpp`; the `two_prod` in `normal.hpp` is still the
contractible form.

**Degrees and term counts.** log I0 is evaluated in three ranges. For k < 7.75, I0(k) = 1 + a P(a) with a = k^2/4 and P a polynomial of degree 14. For 7.75 <= k < 500.0, I0(k) = e^k P(u)/sqrt(k) with u = 1/k and P a polynomial of degree 21 in u. For 500.0 <= k < 1e308 the same asymptotic form is used with a polynomial of degree 4 in u = 1/k. The power series of the ratio for k < 0.25 is A(k) = sum_(n=0..25) a_n k^(2n+1). The compensated asymptotic series for k >= 30.0 is A(k) ~ sum_(n=0..59) q_n k^-n, with the derivative sums S_M(u) = sum_(n=1..59) q_n n (n + 1) ... (n + M - 1) u^n. For the inverse ratio, below kappa = 0.25 the power series is K^(n)(A(kappa)) = kappa^p sum_(m=0..23) g_(n,m) kappa^(2m), and from kappa = 30.0 the normalized asymptotic series is K^(n)(A(kappa)) = n! 2^n kappa^(n+1) (1 + sum_(m=1..45) e_(n,m) kappa^-m).

## The ratios I_j(k)/I_0(k) of the von Mises distribution function

**What.** `bessel_i_ratios(k, m, out)` in `include/laplace/math/bessel_i.hpp`:
R_j = I_j(k)/I_0(k) for j = 1..m, the coefficients of the series of the von
Mises distribution function,
F(x) = (x + pi)/(2 pi) + (1/pi) sum_j (R_j / j) [sin(j(x - mu)) + sin(j(pi + mu))],
truncated in statmodels7 at m = max(20, ceil(8.5 sqrt k) + 10) terms. The
caller supplies room for m values.

**How.** Miller's backward recurrence y_(j-1) = (2j/k) y_j + y_(j+1) from
y_(n0+1) = 0 and a tiny y_(n0), rescaled by 1e-250 when it passes 1e250, with
R_j = y_j / y_0, from n0 = m + max(30, ceil(sqrt(40 m))) + ceil(7 sqrt(k))
(statmodels7 added ceil(k); see the start below).
Below k = 1e-8 the factor 2j/k would overflow y in one step, so there
R_j = R_(j-1) (k/2) / j, whose neglected correction is below k^2/4 < 1e-16
relative. k = 0 gives zeros, k = +inf ones, k < 0 or NaN NaNs.

**Measurement.** References `bessel_i_ratios.csv` (tools/reference/bessel.py
ratios): 7416 rows (k, j), k log-uniform on (2^-30, 2^10) and uniform on
(0, 100), j = 1, 2, 3, m, m/2 and four random orders up to m, at 200 and 264
bits; values below 1e-300 left out. Each candidate computes all m ratios of
the cdf series (m at least j) and returns R_j; timing for the whole vector,
k uniform on (0, 50). Max ulp (max ulp for R_j > 1e-3), gcc / CUDA; ns/call
on one gcc thread, CUDA:

| candidate | max ulp | R_j > 1e-3 | ulp1 | ns/call |
|---|---|---|---|---|
| Miller (chosen) | 15.3 / 12.5 | 3.7 / 4.3 | 1.9 | 182 / 11.8 |
| Miller, start m + 5 + ceil(k) | 13.3 / 13.3 | 5.3 | 2.3 | 130 / 9.8 |
| statmodels7: continued fraction for I_j/I_(j-1) in double, running products | 14.9 / 14.9 | 6.1 | 2.7 | 472 / 9.1 |
| the same, products in double-double | 10.6 / 10.6 | 5.3 | 2.7 | 741 / 10.9 |
| fraction and products in double-double | 0.50 / 0.50 | 0.50 | 0.25 | 5690 / 96 |

**What this says.** In double every candidate loses about one rounding per
order, so the relative error grows along j: the largest errors (10-15 ulp)
sit at orders whose R_j lies between 1e-30 and 1e-3, where they move the cdf
by less than 1e-18. In units of max(R_j, 1), which is what the cdf sees, all
are within 3 ulp. The double-double version is exact to the last bit but
twelve times slower on the CPU. Miller's recurrence multiplies where the
fraction divides, hence its speed on the CPU; on CUDA the two cost the same.
A start closer to m would do for the cdf's m but not for every m, so
statmodels7's start is kept.

**The start.** The recurrence damps its starting value by about
(I_n0 / I_m)^2, and for large k I_j/I_0 ~ exp(-j^2 / (2k)), so a start
c sqrt(k) above m damps it by about e^-(c^2) whatever m is; c = 7 gives e^-49.
statmodels7 added ceil(k) instead, which makes the cost O(k). With the cdf's m
(which already holds 8.5 sqrt(k)) the gain is modest: unchanged for k below 50,
1435 to 763 ns on one gcc thread for k uniform on (500, 1000). Accuracy is the
same for c = 3, 5, 7, 9 and for statmodels7's start (max 12.5-16.3 ulp, 1.9-2.6
ulp1, on the cdf's m); the differences are rounding along different paths,
not truncation: replayed in Python against mpmath, m = 1 at k = 1e4 is off by
4.9 eps from statmodels7's start, 5.6 from 7 sqrt(k), 1.6 from 9 sqrt(k) and
3.4 from 12 sqrt(k). With the new start: 13.8 ulp, 4.7 ulp for R_j > 1e-3,
2.3 ulp1, on both backends; CUDA 11.8 ns (k in (0, 50)) and 67 ns (k in
(500, 1000)).

**Chosen.** Miller's recurrence from m + max(30, ceil(sqrt(40 m))) +
ceil(7 sqrt(k)), on both backends. Tested in `tests/math/test_bessel.cpp`
(23 ulp, 3 ulp1; and m = 1 against A up to k = 1e5) on both builds.

**Open.** The derivatives of the ratios in k, which a cdf differentiated
through its series would need, are not done: they follow from
R_j' = (R_(j-1) + R_(j+1))/2 - R_1 R_j (R_0 = 1) differentiated n times, which
cancels at large k as the Riccati identity of A does; whether the von Mises
cdf is differentiated through the series or, as in statmodels7, by
quadrature of its integral is to be decided with the cdf (phase 4).

**Checked at the same time.** The `two_prod` of `normal.hpp` and
`detail/ibeta_expansions.hpp` is still the form that nvcc may contract. In
`normal.hpp` and `owen.hpp` it only splits x^2/2 for an exponent, where a
contraction would cost at most an ulp. `lambda_of` in `ibeta_expansions.hpp`
does rely on the exact identity, but ibeta_hard on CUDA matches the CPU to the
digit (1325 ulp, ulpL 18.83 on both), and the general set has the same mean
and ulpL (12.9 against 12.7). Left as they are.

**Starting value.** The backward recurrence starts from y_(n0+1) = 0 and y_(n0) = 1e-280.

## log K_nu(x) for real order and its derivatives in x to order 7

**What.** `include/laplace/math/bessel_k.hpp`: `log_bessel_k(nu, x)` = log K_nu(x)
for real nu (K_-nu = K_nu) and x > 0, and `log_bessel_k_dx1..dx7(nu, x)`, its
derivatives in x. What the distributions need, read from statmodels7: the
pseudo-Huber families use K_0 ... K_4 at x = sqrt(nu_ph), the Poisson-inverse
Gaussian log K_(y-1/2)(alpha) with derivatives in alpha (the plan said
"derivatives in nu": that nu was the pseudo-Huber parameter, not the order).
Order seven because the fifth derivative of the PIG expected information
contains the seventh of log K.

**How.** The value: n = round(nu), mu = nu - n; K_mu and K_(mu+1) by Temme's
series (x <= 2) or Steed's continued fraction (x > 2, scaled), both adapted
from Boost.Math (`bessel_ik.hpp`, Boost Software License); then the forward
recurrence on the ratio K_(mu+k+1)/K_(mu+k). From nu = 20 the uniform (Debye)
expansion with the polynomials u_0..u_16 (exact rationals,
`tools/reference/bessel_k.py coefs`). The derivatives, three regimes:
nu >= 20, the Debye form differentiated (the derivatives of eta' = sqrt(1+z^2)/z
from h h' = -z^-3, of log(1+z^2) from (1+z^2) q' = 2z, of log P(t(z)) from P's
Taylor coefficients in t, their logarithm as a power series, composed with the
Taylor series of t(z)); nu < 20 and x >= 200, the expansion in 1/x
(DLMF 10.40.2), its series S differentiated term by term and then log S;
otherwise g = d log K/dx = nu/x - K_(nu+1)/K_nu in double-double and the Riccati
identity g' = 1 + nu^2/x^2 - g/x - g^2 differentiated n times in
double-double. The ratio in double-double comes from Steed's continued
fraction evaluated forwards by its convergents (x >= 1; twice the number of
steps that the same fraction needs in double, plus four), or from Temme's
series in double followed by the forward recurrence in double-double (x < 1;
each step damps the error of the start by about x^2/(4 (mu+k)^2)). The Debye
terms stop once nu passes the order from which the next term is below 2^-60 for
the value and the first seven derivatives in t (table from the generator).

**Measurement.** References from `tools/reference/bessel_k.py`: 2407 points (2607 later, see the entry on Temme's first coefficient)
(nu, x) with nu from 0 to 2^40 (small real orders, half-integers y - 1/2 up to
y = 2^40, large orders around x ~ nu, general) and x from 2^-30 to 2^30. mpmath's
besselk fails near nu = 1e6 (its hypergeometric series), so the references
come from K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt: log K is the log of
that integral and d^n/dx^n log K = (-1)^n kappa_n, the cumulants of cosh T
under the weight, by a double-exponential rule around the peak
t* = asinh(nu/x), at P = 256 + 8 log2(2 + nu + x + 1/x) bits and P + 96 (all rows
agree to 2^-110). Checked against log besselk and the Riccati identity at 300
bits where besselk works: agreement to 1e-75.

The value, in ulps of max(|log K|, |x g|, 1) (one ulp of x moves log K by |x g|
ulps; near the turning point x ~ 0.66 nu, where log K crosses zero, |x g| is
about 1.2 nu and plain ulps of log K reach 508 on the CPU and 1980 on CUDA):
4.8 on both backends. Boost's cyl_bessel_k, logged: 2.9e8 ulp and 1651
non-finite values (it overflows and underflows; it is O(nu) and throws above
nu = 2^31). The derivatives, max ulp gcc / CUDA:

| order | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| own | 6.0 / 5.9 | 15.8 / 21.0 | 16.1 / 20.6 | 10.9 / 10.9 | 11.1 / 11.1 | 21.9 / 21.9 | 31.5 / 29.5 |
| Riccati in double everywhere | | 1.3e20 | | 7.7e37 | | | 1.3e64 |

By regime (gcc): the double-double region within 0.5-0.66 ulp at every order;
x < 1 with nu < 3 up to 16 ulp (Temme's ratio in double, not damped by a
recurrence of 0 to 2 steps); Debye up to 31.5 (at nu = 20.5, x = 20, the edge
of its range); 1/x expansion up to 9.1. Switch points: Debye from nu = 15 costs
up to 4901 ulp at order 7; from 25 the same as from 20 (23 against 31.5 at
order 7, 15.5 against 11.1 at order 5); the 1/x expansion from 100 gives 58 ulp
at order 7, from 400 the same as from 200; Steed's fraction in double-double
from x = 0.5 lowers the x < 1 errors to 5.5 ulp at 1.6 times the cost, from 2
raises them to 550.

ns/call on one gcc thread, OpenMP 16 threads, CUDA, by regime (order 4):
continued fraction in double-double, x in (1, 16): 1222 / 128 / 53 (Steed's own
form in double-double: 4914 / 378 / 168); Temme, x < 1: 384 / 31 / 9.4; 1/x
expansion: 45 / 4.5 / 2.0; Debye: 214 / 23 / 17. The value: 266 / 20 / 19 for
small orders, 139 / 11 / 1.5 for Debye.

**What this says.** The value is accurate to the conditioning of the function
everywhere, at any order up to 2^40 at constant cost. The derivatives are
within 32 ulp everywhere, and within one ulp where the Riccati identity runs
in double-double; that region is also the expensive one (about 1.2 us on one
CPU thread, 53 ns on CUDA).

**Chosen.** As above: Debye from nu = 20, the 1/x expansion from x = 200,
Steed's fraction by convergents from x = 1. Tested in
`tests/math/test_bessel_k.cpp` (value 7.5 ulpC; derivatives 9, 24, 24, 17, 17,
33, 48 ulp; K_(1/2) in closed form) on both builds.

**Open.** The Debye derivatives use Taylor-series arithmetic for log P and its
composition with t(z) (as owen.hpp does for its series): an explicit
computation of one function's derivatives, not a general jet type. The cost of
the double-double region (Steed's fraction needs about 200/x steps) is the
price of the one-ulp derivatives at moderate x; the pseudo-Huber families sit
there (nu = 0..4, x = sqrt(nu_ph)).

**Scaling in the recurrences.** The running product of the value is kept in a single double precision accumulator and rescaled when it exceeds 1e280, to prevent overflow. In the derivatives in the argument, the recurrence of 2n + 4 steps (n being the step count, in double precision) is rescaled by 2^-500 when |Q_1| > 2^500.

**The scaled pair.** `log_bessel_k_scaled(nu, x)` is log K_nu(x) + x and
`log_bessel_k_scaled_dx_upto<N>(nu, x, out)` fills out[0] with it, out[1] with
(log K_nu)'(x) + 1 and out[n] with the plain derivative of order n for n >= 2,
which the shift leaves alone. They exist because a caller whose own quantity is
one of the two shifted entries would otherwise receive a difference of numbers
of size x, or of size one, and lose about log2(x) bits of it: the pseudo-Huber
families need log K_1(x) + x and (log K_1)'(x) + 1 to carry their normalizer and
their Bessel ratio apart from sqrt(nu). Nothing is recomputed for them. Each
regime writes the shifted entry instead of the plain one: Steed's branch omits
the -x it would subtract and Temme's branch, used only for x <= 2 where the two
terms are of a size, adds it; the asymptotic expansion in 1/x omits the -1 of its
order one; the Riccati regime adds one to g in double-double, so the order one is
exact to the double; and the Debye form uses
-nu eta + x = -nu^2/(nu s + x) - nu log(z/(1 + s)) and 1 - s/z = -1/(z(s + z)),
with z = x/nu and s = sqrt(1 + z^2). The plain functions are untouched, and at
x = +inf the scaled value and its order one are 0 where the plain ones are -inf
and -1.

The Debye branch needed one step beyond the shift. Its second term is
-nu log(z/(1 + s)), whose argument is 1 - 1/z + O(1/z^2) for large z, so nu
multiplies the rounding of a logarithm near one: at nu = 3.1e4, x = 6.2e7 that
alone put the scaled value 8693 ulp out. Written as
nu log1p(-(1 + 1/(s + z))/(1 + s)), which is exact because s - z = 1/(s + z), it
is 101. Below z = 1 the argument is nowhere near one and the plain logarithm is
the accurate form, so the branch keeps both.

Measured against the module's own reference grids, the reference pair (hi, lo)
plus the shift summed without error in the test (one two_sum, the shift being
exact as a double), so that what is measured is the scaled routine and not a
difference formed in the test: the first derivative is at or below 1.17 ulp of
its own size in every regime, in units of one ulp plus four eps of (log K)',
the irreducible error where the shifted quantity crosses zero near x = 1. The
value is 0.49 ulp for x <= 2, in units of one ulp plus eight eps of log K, which
is the accuracy the plain value itself is measured to; above x = 2 it carries no
floor and reaches 35.2 ulp at nu = 4, x = 9.5 (the continued fraction) and 101
ulp at nu = 248, x = 7635 (Debye), because there the scaled value is what remains
of a sum of terms of the expansion that can be tens of times larger than it, so
it is accurate to about 1e-13 absolute and not to its own last place. The orders
the distributions ask for are far below that: the pseudo-Huber uses nu = 1 only,
and its own grid reads 3.71 ulp on the value at nu_ph = 1e8. The orders two and
up are the plain ones to the bit, on every reference grid and regime boundary.
Tested in `tests/math/test_bessel_k.cpp` on both builds, bounds 160 and 2.

## Faa di Bruno's formula and the partial Bell polynomials to order 7

**What.** `include/laplace/math/faa_di_bruno.hpp`: `bell_partial(n, k, g)`, the
partial Bell polynomials B_(n,k)(g_1, ..., g_(n-k+1)) for 1 <= k <= n <= 7,
and `compose_d1..d7(f, g)` = d^N/dx^N f(g(x)) = sum_k f^(k)(g(x)) B_(N,k)(g', ...),
the arrays indexed by derivative order. They carry a distribution's
derivatives in a parameter theta to the linear predictor of a link
theta = h(eta) (phase 3), and mixed derivatives in several parameters, each
with its own link, are sums of products of the B_(n,k) of each parameter.

**How.** The 28 polynomials written out from sympy's `bell(n, k, ...)`, each a
sum of monomials with positive integer coefficients; compose_dN sums
f_k B_(N,k) from k = N down to 1.

**Measurement.** References `faa_di_bruno.csv` (tools/reference/faa_di_bruno.py):
1950 rows, 1500 with derivatives drawn at random (magnitudes log-uniform on
(2^-10, 2^10), random signs) and 450 from compositions met in practice (outer
log, exp, lgamma, log Phi, x^-1/2; inner exp, logistic, Phi, x^2, sqrt, log, at
random points, derivatives by mpmath.diff rounded to doubles). The expected
values are exact on the double inputs, by composition of truncated Taylor
series at 300 bits (independent of the Bell polynomials). Error in ulps of
S = sum_k |f_k| B_(N,k)(|g|), the sum of the absolute values of the terms,
gcc / CUDA:

| order | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| ulps of S | 0.5 / 0.5 | 1.73 / 1.73 | 2.21 / 2.21 | 3.08 / 3.03 | 3.05 / 3.05 | 3.96 / 3.96 | 5.03 / 5.03 |

In plain ulps the error reaches 1e16-1e18 on the practical cases whose
derivative is zero or nearly so (log(exp(x)) = x has zero derivatives from
order two on): there the result is the cancellation of the rounded inputs and
no order of summation recovers it. ns/call (the 16 inputs read from memory):
2.8-6.6 on one gcc thread, 1.7-1.9 with 16 threads, 0.31 on CUDA. The
polynomials are also checked exactly against the Stirling numbers
B_(n,k)(1, 1, ...) = S(n, k) and against B_(n,k)(1, 2, 3, ...) = C(n, k) k^(n-k).

**What this says.** The formula is backward stable: its error is a few
roundings of its largest term.

**Chosen.** This version. Tested in `tests/math/test_faa_di_bruno.cpp` on both
builds.

**Open.** The multivariate assembly (mixed derivatives through several links)
is written with the link layer, where its shape is known.

**Explicit formulas.** Writing f_k = f^(k)(g(x)) and g_m = g^(m)(x), the derivatives of f(g(x)) of orders 1 to 7 are:
- N = 1: f_1 g_1.
- N = 2: f_1 g_2 + f_2 g_1^2.
- N = 3: f_1 g_3 + 3 f_2 g_1 g_2 + f_3 g_1^3.
- N = 4: f_1 g_4 + f_2 (4 g_1 g_3 + 3 g_2^2) + 6 f_3 g_1^2 g_2 + f_4 g_1^4.
- N = 5: f_1 g_5 + f_2 (5 g_1 g_4 + 10 g_2 g_3) + f_3 (10 g_1^2 g_3 + 15 g_1 g_2^2) + 10 f_4 g_1^3 g_2 + f_5 g_1^5.
- N = 6: f_1 g_6 + f_2 (6 g_1 g_5 + 15 g_2 g_4 + 10 g_3^2) + f_3 (15 g_1^2 g_4 + 60 g_1 g_2 g_3 + 15 g_2^3) + f_4 (20 g_1^3 g_3 + 45 g_1^2 g_2^2) + 15 f_5 g_1^4 g_2 + f_6 g_1^6.
- N = 7: f_1 g_7 + f_2 (7 g_1 g_6 + 21 g_2 g_5 + 35 g_3 g_4) + f_3 (21 g_1^2 g_5 + 105 g_1 g_2 g_4 + 70 g_1 g_3^2 + 105 g_2^2 g_3) + f_4 (35 g_1^3 g_4 + 210 g_1^2 g_2 g_3 + 105 g_1 g_2^3) + f_5 (35 g_1^4 g_3 + 105 g_1^3 g_2^2) + 21 f_6 g_1^5 g_2 + f_7 g_1^7.

Outside the triangular range of valid pairs the Bell polynomial returns 0.0; this is tested on the out-of-range pairs (3, 4) with B_(3,4) = 0.0, (1, 9), (2, -7), (0, 9), (8, 1) and (0, 0). The expected values of the composition test are exact on the double-precision inputs, generated by composing the truncated Taylor series F(t) = sum_(k=1..N) f_k t^k/k! and G(e) = sum_(j=1..N) g_j e^j/j!: the coefficient of e^N in F(G(e)), multiplied by N!, is the exact composite derivative of order N, independently of the library's Bell polynomials.

## Quadrature rules for kernels: double-exponential, Gauss-Kronrod, panels, fixed Gauss

**What.** `include/laplace/math/quadrature.hpp` (namespace
`laplace::math::quad`), rules callable inside Kokkos kernels (integrand a
functor, no allocation, no recursion, every loop capped), each returning the
value, an error estimate, the evaluations and whether it converged:
tanh-sinh on [a, b], exp-sinh on [a, inf), sinh-sinh on R (double-exponential,
step halved from 1/2 until two estimates differ by less than `stop` = 1e-9 of
int |f|, at most 8 halvings, t in [-6.8, 6.8] trimmed at step 1/2 to where
|f w| exceeds 2^-64 of int |f|); Gauss-Kronrod 7-15 adaptive (at most 64
intervals; Boost's constants); `panels`, statmodels7's rule of
`pt_trunc_rule.h` (cut at the centre and at up to 8 kinks, doubling panels of
widths s, 2s, ... from the side nearer the centre, exp-sinh beyond 63 s on an
unbounded side) with a convergence check on every panel; fixed Gauss-Legendre
16, 32, 64, Gauss-Hermite 32 and Gauss-Laguerre 32 (Golub-Welsch at 256 bits).
tanh-sinh and exp-sinh also accept an integrand f(x, xc) that takes the signed
distance to the nearer end (Boost's convention), for singularities at an end
that x itself cannot resolve.

**Measurement.** References `quadrature.csv` (tools/reference/quadrature.py,
540 rows): nine integrals from the uses, each at 60 parameter values, by
closed forms (the skew normal entry by mpmath.quad at 200 bits): beta (singular
ends, exponents down to 0.05), gamma (singular end, a up to 100),
exp(-x^2/2) cos(px), the Student t tail (nu down to 0.25), the Laplace density
on [p1, p2] (kink at 0), x^p log(x)^2 (p + 1 >= 0.1), the skew normal entry
E[(d/dalpha log f)^2] (|alpha| up to 30), the pseudo-Huber normalizer
(nu from 0.01 to 100) and sech(x)^p. Max ulp, gcc (CUDA where it differs),
with the mean number of evaluations; every row converged:

| integral | tanh-sinh | exp-sinh | sinh-sinh | panels | GK15 |
|---|---|---|---|---|---|
| beta (with the distance to the end) | 6.5 (7.8), 88 | | | | fails |
| gamma | | 68 (20), 136 | | 13 (21), 839 | |
| Student tail | | 5.4, 85 | | 4.0, 793 | |
| Laplace with a kink | | | | 4.5, 229 | 72, 167 |
| x^p log(x)^2 | 6.4 (7.4), 72 | | | 3.9, 274 | fails |
| skew normal entry | | | 7.3 (6.7), 159 | 6.8 (5.4), 1527 | |
| pseudo-Huber normalizer | | | 18, 615 | 2.8, 1557 | |
| sech^p | | | 6.3, 182 | 3.7, 1561 | |

Not in the table: exp(-x^2/2) cos(px) loses relative accuracy as p grows (the
value is e^(-p^2/2) of int |f|; 1e7 ulp at p = 5.7, a cancellation no rule
avoids); GK15 on the singular ends and the fixed Gauss rules on these
integrands (1e9 to 1e15 ulp: they are for smooth integrands of Gaussian or
exponential shape, and integrate their polynomials exactly, which the test
checks); stop = 1e-11 instead of 1e-9 gains nothing and costs 10-30 per cent
more evaluations. Three defects found on the way and repaired: a node whose x
rounded to the end was dropped (it carried the whole singular mass of beta
with b = 0.06, 0.2 relative; and 1251 ulp on a smooth integrand far from 0);
exp-sinh stopped at x = 1e-152 (t <= 6.1), too early for x^(a-1) with small a
(2e7 ulp at a = 0.055); both fixed by keeping such nodes (with xc for the
two-argument form, skipping a non-finite value) and t up to 6.8. The limit
that remains is the double range: x^(a-1) needs nodes below 10^(-16/a) of the
scale, so exponents below about 0.06 cannot be integrated to the last bit.

Throughput on 69120 integrals (the 540 rows 128 times), us per integral, one
gcc thread / 16 threads / CUDA: tanh-sinh 17.3 / 1.51 / 3.44; exp-sinh 10.0 /
0.91 / 0.82; sinh-sinh 20.8 / 1.79 / 2.62; GK15 61 / 4.4 / 3.9; panels 121 /
9.5 / 8.7; the fixed rules 2.1-2.7 / 0.26-0.32 / 0.20. The cost is the
integrand (90 to 1500 evaluations, each with exp and log at about 28 ns on the
Rtools libm). 16 threads give 11-13 times one thread; the GPU matches them
(FP64 at 1/64 of FP32 on this card, and threads that stop at different
levels diverge).

**What this says.** For every integral of the uses there is a rule within a
few tens of ulps: tanh-sinh with the distance to the end for singular ends,
exp-sinh or sinh-sinh where the mass sits near the end or the centre, panels
where there are kinks or the mass is far from the end, at about ten times the
evaluations.

**Chosen.** As above, stop = 1e-9. Tested in `tests/math/test_quadrature.cpp`
on both builds.

**Open.** `panels` takes integrands of x alone, so it cannot use the distance
to a singular end (beta with a small exponent goes to tanh-sinh); the choice of
rule, centre and scale for each integral is the caller's, decided when the
distributions are written.

**Details of the rules.** Beyond |t| > 6.8 the transformed nodes leave the IEEE 754 double-precision range. The first estimate is evaluated at step h = 1/2 over k in [-K, K] with K = floor(6.8/0.5) = 13 (27 nodes), keeping one step beyond the last significant node; each halving (at most 8, so down to h = 2^-9 = 1/512) evaluates only the odd multiples of h. Gauss-Kronrod uses Boost's constants (`boost/math/quadrature/gauss_kronrod_quad.hpp`): on each subinterval [u, v] a 15-point Kronrod estimate K and an embedded 7-point Gauss-Legendre estimate G are formed with 15 function evaluations, the local error estimate is |(K - G) (v - u)/2|, the subintervals are stored in flat arrays capped at 64 intervals, and at each iteration the subinterval with the largest error is bisected at its midpoint m = (u + v)/2. The nodes of the fixed Gauss rules are written by `tools/reference/quadrature.py nodes`. In `tests/math/test_quadrature.cpp` the values 4e-16 and 1e-14 are test tolerances (`doctest::Approx` epsilon), not measured errors: Gauss-Legendre with N = 16 and 64 (N = 32 is untested) is tested on the monomials x^k on [0, 1] for k in {0, 1, 7, 31} with 4e-16, Gauss-Hermite 32 on x^(2j) e^(-x^2) for j in {0, 1, 2, 3} with 1e-14, and Gauss-Laguerre 32 on x^k e^(-x) for k in {0, ..., 9} with 1e-14.

## Piecewise Chebyshev expansions of the derivatives of log Phi on [-1, 4)

**What.** `log_Phi_d1..d5` in `include/laplace/math/normal.hpp` on t in
[-1, 4):
Chebyshev expansions on five pieces of width 1, evaluated by Clenshaw's
recurrence from a `__constant__` table (generated block between BEGIN and END
markers). Outside, t < -1 keeps the cumulants and t >= 4 the recursion from
R, both unchanged. Each order keeps its own function.

**How.** `tools/reference/log_phi_cheb.py`: g_k at 80 Chebyshev nodes per
piece at 300 bits (checked against 380), coefficients by the discrete cosine
sum, degree the smallest with the tail of |c_m| below 2^-56 max |g_k| on the
piece: 17, 17, 18, 19, 20 for k = 1..5. Denser references
`log_Phi_d{k}_mid.csv` (20011 rows each on [-1, 10], added by normal.py). The
upper end 4: above it the old route is already below 0.08 ulp1 at every order;
the zeros of Phi at about 3.08 +- 3.81i set the piece width.

**Measurement.** Max error in ulp of max(|g|, 1), old / new, gcc:

| region | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| [-1, 0) | 2.85 / 1.19 | 6.14 / 0.64 | 13.5 / 0.16 | 36.2 / 0.082 | 109 / 0.070 |
| [0, 1) | 1.67 / 0.73 | 2.39 / 0.62 | 4.26 / 0.32 | 10.2 / 0.10 | 24.7 / 0.20 |
| [1, 2) | 0.40 / 0.33 | 0.77 / 0.32 | 1.21 / 0.34 | 2.09 / 0.18 | 3.04 / 0.23 |
| [2, 3) | 0.067 / 0.051 | 0.19 / 0.12 | 0.44 / 0.16 | 0.91 / 0.14 | 1.43 / 0.18 |
| [3, 4) | 0.0065 / 0.0055 | 0.018 / 0.013 | 0.062 / 0.040 | 0.22 / 0.094 | 0.50 / 0.14 |

CUDA, overall on the main file (t < -1 included): old 2.53, 6.14, 12.2, 32.7,
90.5; new 0.99, 0.63, 0.75, 1.19, 1.85. ns per call on [-1, 4), old / new:
gcc one thread 38-43 / 15-19; CUDA 0.90-1.32 / 0.20-0.29.

**What this says.** On [-1, 4) the error falls from up to 109 ulp1 to within
2 at every order, and the evaluation is 2.5 times faster.

**Chosen.** This version. Tested in `tests/math/test_normal.cpp` (centre bounds
2.0, 1.0, 1.2, 2.0, 3.0 ulp1, were 4, 9, 18, 50, 140) on both builds.

**Open.** Order 1 at [-1, 0) reaches 1.19 ulp1, mostly Clenshaw's rounding at
values near 1.5.

## Double-double sqrt, exp, expm1 and log

**What.** `include/laplace/math/detail/double_double.hpp` gains `ddm::sqrt`,
`exp`, `expm1` and `log`. sqrt by Karp's method, scaled by an even power of two outside
[2^-900, 2^900] (Dekker's split overflows above 2^996, 1/sqrt(a.hi) for
subnormal a). exp as in the QD library: Cody-Waite reduction by a two-part
ln 2, Taylor series of e^r - 1 at r = t/2^9, nine steps s <- s (s + 2), which
keep the relative precision of e^r - 1 (`expm1_reduced`, also behind
`expm1`), scaling by 2^k. log: the argument split as m 2^e with m in
[sqrt(1/2), sqrt(2)); for e = 0 and |m - 1| < 1/4 Newton's step on
expm1(x) = m - 1 from x0 = log1p(m - 1), elsewhere on exp(x) = m with the
log(1 + u) terms to u^3. All through the add/sub/mul helpers (safe from
nvcc's contraction).

**Measurement.** References `dd_{sqrt,exp,log,expm1}.csv`
(tools/reference/double_double.py, 300 bits, inputs hi + lo with lo a random
fraction of the ulp). Error in units of 2^-104 |f|, gcc and CUDA identical:

| function | max | ns/call, 1 gcc thread / CUDA | plain double |
|---|---|---|---|
| sqrt | 1.27 | 13 / 0.22 | 2.3 / 0.05 |
| exp | 0.57 | 490 / 6.3 | 3.1-29 / 0.09 |
| log | 1.86 | 530 / 8.0 | 3.6-29 / 0.15 |
| expm1 | 2.11 | | |

The first version failed three ways, each repaired before it was kept: sqrt gave
1.9e23 units on subnormal input and NaN near the largest double (scaling
added); log near 1 was accurate only to 2^-104 absolute, 8.3e15 units
relative, because its Newton step works on m exp(-x0) ~ 1 (the expm1 form
added); expm1 was missing.

**Chosen.** As above. Tested in `tests/math/test_double_double.cpp` on both
builds (2, 1, 3, 3.5 units; exp(log x) = x within 1.2 units of max(1, |log x|)).

**Open.** exp and log cost about 500 ns on one gcc thread (Dekker's product on
the host); a table-driven reduction would cut it if a hot loop needs them.

## Temme's uniform expansion for the incomplete gamma at large shapes

**What.** `include/laplace/math/incomplete_gamma.hpp`: `own::gamma_inc` takes
`gamma_inc_temme` for a >= 100 and |eta| <= 0.5 (eta^2/2 = phi =
x/a - 1 - log(x/a), signed as x - a), `gamma_inc_classic` (the former code)
elsewhere.
Q = erfc(s)/2 + R with s = sqrt(a phi), R = e^-(a phi) S(a, eta) / sqrt(2 pi a),
S = sum_(k<=9) c_k(eta) a^-k, the c_k Boost.Math's 53-bit polynomials
(igamma_large.hpp, Boost licence); erfc as e^-y erfcx(s) so that nothing
underflows, and the logarithm of the small tail as -y + log(bracket). The
classic route needs O(sqrt(a)) terms near x = a; this one is O(1).

**Measurement.** New references `gamma_{p,q}_large.csv` and
`log_gamma_{p,q}_large.csv` (2400 rows: a log-uniform on [1e2, 1e15], x =
a (1 + d), |d| log-uniform on [2^-40, 1], both signs). Two defects in the
first references, both repaired: mpmath.quad on [0, x] without breakpoints
near x, where the mass sits (wrong by a factor 0.58 at a = 1e12, x 24 sqrt(a)
below the mode), and the integrand not scaled (values like e^-288 defeat
quad's absolute error control; scaled by its maximum on the interval, the
values agree to 30 digits across breakpoints and degrees); and the logarithm
of the complement taken as log(1 - p) at 200 bits instead of log1p(-p). The
first switch at |eta| <= 3 was wrong: Boost's c_k are polynomials truncated
about eta = 0, and the route, measured on the logarithm of the small tail, is
within 6 ulp for |eta| < 0.5 at every a, 11-38 ulp for |eta| in [0.5, 1) and
up to 1.7e3 beyond; the switch is now 0.5 (beyond, the classic route needs a
few tens of terms). Result, gcc (CUDA within the same test bounds): P and Q
within 0.82 ulp of max(|f|, 1) and 2.9-3.4 ulpL (the small tail carries
about a phi ulps from its argument: 1.0e3-1.3e3 plain ulps where a phi is
near 500); log P and log Q within 5.9-6.4 ulp of max(|f|, 1). Every call
within 5 us on one thread, against O(sqrt(a)) terms of the classic route
(1e6 at a = 1e12). The first bench ran for hours: Boost outside its own
Temme zone (a sigma^2 < 20) sums O(1/sigma) terms, about 1e6 at a = 1e15;
Boost is left out of the large-shape part of the bench.

**Chosen.** Temme for a >= 100, |eta| <= 0.5. Tested in
`tests/math/test_special.cpp` on both builds; the tests of the inverses and of
the shape derivatives, which call gamma_inc, pass unchanged.

## The large logarithms of the incomplete gamma and beta in double-double

**What.** The inverses (`incomplete_inverse.hpp`) solve g(x) = log S(x) - log s
= 0, S the forward probability and s the target. Far in a tail two terms
carried an absolute error of about |log s| 2^-53, so the inverses solved s to
about |log s| ulps (up to 700): the rounding of log s itself, and the term
a log x (or b log y) inside the prefactor of gamma_inc and beta_inc. Both errors
come from the exponent part of the logarithm. New `ddm::log_split(x)`
(`detail/double_double.hpp`): x = m 2^e with m in [sqrt(1/2), sqrt(2)) taken
from the bits, log x = e LN2_1 + (e LN2_2 + log m) as a double-double, LN2_1
being ln 2 to 42 bits (so e LN2_1 is exact) and LN2_2 the rest; one plain log,
absolute error that of log m. Carried as double-doubles: the gamma prefactor for
a >= 1, x <= a/2 (`log_prefactor_dd`, -a phi = a - x + a (log x - log a)) and the
small-a series; in the beta, lx and ly in `detail_ibx::args` (low parts
`lx_lo`, `ly_lo`), `log_power_dd` (the tail branches u, v < -1/2 with
log_split(a/s), and the branches with a shape below 1), `bpser`, `bfrac`, `bup`,
`log_add`; the results gain `log_p_lo`, `log_q_lo` (gamma) and `log_i_lo`,
`log_ic_lo` (beta), the probability is formed as e^hi (1 + lo), and the inverses
take log s from log_split and compare g = (ls - target) + (ls_lo - target_lo).
The full double-double log (`ddm::log`, about 500 ns) is not needed. The
Student t value (`log_cdf_beta`) uses the low part in log(1 - e^(log S)), where
rounding log S to a double alone loses up to |log S| / 2 ulps.

**Measurement.** Inverses in ulpC (99%, max), gcc OpenMP 8; CUDA within the same
figures:

| set | before | after |
|---|---|---|
| gamma_p_inv | 23, 52.2 | 2.01, 6.86 |
| gamma_p_inv_tail | 255, 657 | 1.74, 3.06 |
| gamma_q_inv_tail | 2.17, 10.1 | 1.92, 6.09 |
| gamma_p_inv_logp | 6.7, 39 | 6.37, 17 |
| ibeta_inv (1 - x) | 43, 118 (8.09, 22.2) | 4.09, 7.05 (7.27, 19.5) |
| ibeta_inv_tail (1 - x) | 343, 813 (9.31, 13.9) | 2.07, 5.38 (2.71, 11.3) |
| ibetac_inv (1 - x) | 5.95, 11.8 (39, 58) | 5.53, 12.3 (4.32, 7.96) |
| ibetac_inv_tail (1 - x) | 2.04, 4.39 (456, 1040) | 1.83, 4.4 (2.75, 4.39) |
| ibeta_inv_logp (1 - x) | 5.92, 18.6 (7.43, 10.2) | 4.9, 18.6 (4.4, 6.49) |
| ibetac_inv_logq (1 - x) | 5.64, 8.88 (5.85, 16.3) | 3.28, 5.82 (5.34, 16.3) |

Forward functions: mean plain-ulp error of gamma P from 29.6 to 13.5 (Q
unchanged, its path untouched), of ibeta from 25.2 to 9.8, ibetac 23.6 to 8.2,
the hard sets 6.7 to 5.4 and 5.6 to 3.9; maximum ulp1, ulpL and log errors
unchanged or lower. log_split within 0.49 units of 2^-53 absolute on the
arguments of dd_log.csv.

Cost, base and new alternated on the same day (ns per call):

| | gcc 1 thread | CUDA |
|---|---|---|
| beta_inc, common / mixed / large | 331 / 377 / 395 to 371 / 407 / 424 (+12, +8, +7%) | +4%, +3%, 0 |
| gamma_inc | 252 to 264 (+5%, at the noise) | |
| gamma inverses, bulk / tail | +2% / 673 to 798 (+18%) | 0 / -15% |
| beta inverses, bulk / tails | +7% / +10-17% | +2-4% / +7% |

On gcc most of the cost is the extra plain logs (about 28 ns each with the
Rtools libm). A first version took log a and log s separately in the tail
branch of log_power (+71 ns on bfrac); one log_split(a/s) is as accurate (the
rounding of a/s moves its logarithm by 2^-53) and cheaper. Replacing
Kokkos::frexp by reading the bits made no measurable difference.

**Chosen.** Adopted as above, always on (a version that
uses the double-doubles only for |log x| large would save most of the cost but
leave about 8-16 ulpC in the middle). Tested on both builds: the inverse tests
unchanged, `log_split` in `tests/math/test_double_double.cpp` (bound 0.75).

## Own exp and log in the gcc build: effect on whole functions

**What.** The first entry (2026-11-08) left open the libm of Rtools, whose exp and
log cost about 28 ns per call in the gcc build that R loads, against 4-6 ns for
the own fdlibm-style versions (exp within 1 ulp, log within 0.78). New macro
`LAPLACE_OWN_HOST_ELEMENTARY` (`elementary.hpp`): the wrapper
`laplace::math::{exp,log}` takes the own versions on the host and keeps the
platform's on the device. Measured with a build that defines it
(`-DCMAKE_CXX_FLAGS=-DLAPLACE_OWN_HOST_ELEMENTARY`) against the default build,
same day, gcc. Calls inside Boost.Math (lgamma, tgamma1pm1, log1p, expm1) and
direct Kokkos calls still go to the libm.

**Measurement.** ns per call on one gcc thread:

| function | platform exp/log | own exp/log | change |
|---|---|---|---|
| beta_inc, common / mixed / large | 370 / 401 / 424 | 263 / 264 / 344 | -29%, -34%, -19% |
| gamma_inc (P, Q) | 262 | 193 | -26% |
| gamma inverses | 715-740 | 433-485 | -35 to -41% |
| beta inverses | 1506-1979 | 976-1303 | -35% |
| Student t derivatives, orders 1-5 | 2315 / 2781 / 3466 / 4796 / 8014 | 1826 / 2229 / 3032 / 4356 / 7303 | -21 to -9% |
| skew normal derivatives, orders 4-5 | 2165 / 2361 | 1478 / 1670 | -32%, -29% |

Accuracy: unchanged to within noise on every reference set measured (beta_inc
max ulp, ulp1, ulpL and mean equal to three digits; gamma_inc the same; the
inverses' tails the same, ibetac_inv_tail 1 - x max plain ulp 80 to 55). The
whole test suite passes with the own versions except the Bessel K derivatives
of orders 2 and 3 at nu = 1/2, x = 0.805 (25.8 and 26.1 ulp against a bound of
24); the skew normal's order 5 in the large-alpha region goes from 1.4e4 to
2.8e4 ulp1, near the zeros where it is ill-conditioned.

**Chosen.** The own exp and log on the host in every CPU build (gcc,
and MSVC on the host side of the CUDA build): the same algorithm on every CPU
platform, and the speed of the gcc build that R loads. On Linux, where glibc's
exp and log are fast and nearly correctly rounded, the switch costs about half
an ulp for a gain not yet measured (no Linux build yet). The default is in
`elementary.hpp` itself, so builds without CMake (an R package) get it too;
`LAPLACE_PLATFORM_ELEMENTARY` (CMake: `LAPLACE_OWN_ELEMENTARY=OFF`) restores the
libm. The device keeps CUDA's. The Bessel K failure turned out to be an older
defect, which the references had missed (next entry). Open: whether the calls
inside Boost should follow.

## log K_nu derivatives near x = 1: Temme's first coefficient

**What.** With the own exp and log, the test of the derivatives of log K_nu
failed at nu = 1/2, x = 0.805 (25.8 ulp against a bound of 24). The defect is
older and independent of exp and log. For x < 1 (below x_cf) the ratio
K_(mu+1)/K_mu comes from Temme's series in double, mu = nu - round(nu). Its
first coefficient is a difference of two terms,
f_0 = ((x/2)^-mu Gamma(1 + mu) - (x/2)^mu Gamma(1 - mu)) / (2 mu),
which vanishes near x = 1-1.12 for every mu (at mu = -1/2 it is
sqrt(pi/2) (x^-1/2 - x^1/2), zero at x = 1); each term carries about one ulp
from exp, log, sinh and tgamma1pm1, so the ratio is off by 5-10 ulps there
(summing the series in double-double lowers 10.5 to 7.9 only: the error is in
the inputs). Below nu = 3/2 at most one recurrence step damps it, and the
Riccati identity amplifies it about eight times.

**Measurement.** Second derivative at nu = 1/2 against its closed form, own
exp and log (the platform's gives the same pattern, 31.6 ulp at most): x in
(0, 0.25] 0.6-1.5 ulp, (0.25, 0.5] 6.8, (0.5, 0.75] 19.3, (0.75, 1] 41.6; at
nu = 3/2 6.2 at most, at nu = 5/2 0.7. The references had missed the worst
points: `tools/reference/bessel_k.py` now adds 200 points with x in (0.5, 1)
and nu in {0, 1/4, 1/2, 1, 3/2} or uniform in (0, 3) (2607 rows; the earlier
2407 are reproduced bit for bit). On them, orders 1-7: 8.9, 44.4, 44.0, 24.1,
12.4, 21.9, 31.5 ulp, worst at nu = 1/4, x = 0.979 for orders 2-3, the same
to three digits with the platform's exp and log; on CUDA 37.4, 37.0, 25.6 for orders 2-4. Steed's fraction in
double-double from x = 0.5 for nu < 3/2 brings every order to 0.5 ulp but
costs 2679 against 282 ns (order 2) and 2960 against 626 ns (order 7) per
call on one gcc thread in that band.

**Chosen.** The bounds of orders 2-4 raised to 66, 66,
36 ulp (`tests/math/test_bessel_k.cpp`); Steed's fraction below x = 1 rejected
for its cost. The Poisson-inverse Gaussian, which uses exactly nu = y - 1/2,
gets a closed form for half-integer orders in phase 4: K_(n+1/2)(x) =
sqrt(pi/(2x)) e^-x times a polynomial in 1/x, exact and cheaper than the series.

## Edge cases found while writing the manual (2026-11-12)

**What.** Reviewing the manual's text against the code turned up three
defects at the edges of the domains; none affected the measured accuracy.

**Faa di Bruno.** `bell_partial(n, k, g)` switched on the key n * 8 + k, so a
pair outside 1 <= k <= n <= 7 could alias a valid case (n = 1, k = 9 returned
g_2). An explicit check now returns 0 outside the domain; the test adds
(1, 9), (2, -7), (0, 9), (8, 1), (0, 0).

**Normal block.** At extreme arguments the exact square (two_prod, half
square) overflows from |x| = 1.3e154, and exp(a^2) from a = 26.64, which gave
NaN where the result is a constant: erfcx(x) for x <= -26.629 (now +inf; finite
down to -26.628), erfc(x) for |x| > 27.3 (now 0 or 2), Phi beyond +-40 (now 0
or 1; Phi underflows below -38.5), log Phi below -1.8961e154 = -sqrt(2 DBL_MAX)
(now -inf) and above 40 (now -0), Phi_inv_logp(-inf) (now -inf). The checks sit
where the result is already constant, so the measured accuracy is unchanged;
`tests/math/test_normal.cpp` checks every boundary ("normal block at the ends
of the range"), on CPU and CUDA.

**Quadrature conventions (decided with Giovanni).** The rules disagreed on
reversed, empty and invalid arguments: `tanh_sinh` with b < a returned 0 with
`converged` = true and error NaN, while `gauss_kronrod` returned the negated
integral; `sinh_sinh` with s < 0 flipped the sign of an integral that does not
depend on it. Now every rule follows int_a^b = -int_b^a (tanh_sinh,
gauss_kronrod, gauss_legendre, panels); an empty interval gives exactly 0,
error 0, converged; a NaN or infinite end or centre, or a scale that is not
positive and finite, gives NaN and `converged` = false (NaN for the fixed Gauss
rules). Valid inputs give the same results as before. Tested in
`tests/math/test_quadrature.cpp` ("reversed, empty and invalid arguments"), on
CPU and CUDA.

## Scalar bijectors without parameters

**What.** `include/laplace/bijector/scalar.hpp` (namespace `laplace::bijector`):
twelve maps g: V -> U between intervals with their inverses h = g^-1, named by
the forward direction, which goes from the constrained side to the
unconstrained one: identity, log, logit, probit, cauchit, cloglog, loglog,
inverse (1/v), inverse_sq (1/v^2), sqrt, atanh (the rhobit link), asinh. For
each, the value and the derivatives of order 1-6 of both directions, and the
log-Jacobians log|g'(v)| and log|h'(u)| with their derivatives of order 1-6:
28 member functions of the type `scalar_bijector` (map chosen by the enum
`scalar_bijector_map`), one per quantity and order: `forward`, `forward_d1`
... `forward_d6`, `inverse`, `inverse_d1` ..., `forward_log_jacobian` ...,
`inverse_log_jacobian` .... A flag inverts the bijector (exp is log
inverted). Nothing is clamped (decided with the clamping study: see below).

**How.** Every quantity is written directly for its direction, from
auxiliary quantities that do not cancel: for logit p, q = 1 - p, U = p q and
D = q - p = -tanh(u/2) from one exp(-|u|), and the forward derivatives of even
order as (2p - 1) a b S_K(a, b) with a = 1/p, b = 1/q and S_K a sum of
positive terms; for atanh T = tanh u and S = 1 - T^2 from exp(-2|u|); for asinh
and the inverse of cauchit r = 1/hypot(1, x) and s = x r; for cloglog and
loglog t = log s and c = 1/(s t). The derivatives are these quantities times
polynomials derived and checked with SymPy 1.14.0
(`tools/reference/bijector_forms.py`). Five choices came out of the
measurement:

- probit: 1/phi(z) at z = Phi^-1(v) computed as m(z)/v, with the Mills-type
  ratio m(z) = Phi(z)/phi(z) = sqrt(pi/2) erfcx(-z/sqrt 2) (and 1 - v, -z
  above 1/2). Through exp(z^2/2) the rounding of z costs z^2 eps: 1050 ulp at
  v = 3.4e-118 for the second derivative, against 6.2 ulp now.
- The polynomials of tanh and of the r, s family are written in T^2 (s^2)
  below 1/2 and in S = 1 - T^2 (r^2 = 1 - s^2) above, where the T^2 forms
  cancel (terms up to 50 times the result): up to 48 ulpC before, 8.6 after.
- log cosh u below |u| = 1 as log1p(em^2/(2 (1 + em))), em = expm1(|u|), and
  -log(1 - v^2) of atanh below |v| = 1/2 as -log1p(-v^2): the forms used
  elsewhere cancel to u^2 and v^2 there (errors of 1e17 ulp at |u| = 1e-9).
- Powers multiplied with the constant first (c a^K), and the derivatives of
  sqrt and of the inverse of inverse_sq as powers of 1/sqrt: a^K went
  subnormal or overflowed before the result did (sqrt: NaN from v < 1e-102).
- The factor 1 - w = 1 - e^u in the second derivative of the inverse of
  cloglog (and loglog) as -expm1(u): 1.4e8 ulpC at u = -1.2e-9 before.

**Measurement.** References `bijector_<map>.csv`
(`tools/reference/bijector_scalar.py`): every map in SymPy by its definition,
differentiated symbolically and evaluated with mpmath (the forward probit by
Newton on Phi and series reversion), at a precision that grows with the point,
twice 120 bits apart, rows kept if the two agree to 2^-100; zero, non-finite
and subnormal values dropped. 52150 rows: points log-uniform towards 0 and
towards the bounds (down to 2^-1000 and to 1 - 2^-54), uniform in the middle,
|u| up to 2^10 (2^1000 for the inverse of cauchit and for asinh). Two errors:
ulp, relative; ulpC, in ulps of |f_k(x)| + |x f_(k+1)(x)|, the error that one
ulp of relative perturbation of x would cause, which stays meaningful near
the zeros of the derivatives of high order (there ulp grows without bound).
Maximum ulpC over the orders 0-6, gcc / CUDA:

| map | forward | inverse | forward log-Jacobian | inverse log-Jacobian |
|---|---|---|---|---|
| log | 3.84 / 3.84 | 0.574 / 0.574 | 3.84 / 3.84 | 0 / 0 |
| logit | 3.17 / 3.17 | 10.5 / 10.1 | 5.17 / 5.17 | 10.5 / 10.1 |
| probit | 6.17 / 6.17 | 2.28 / 2.28 | 6.57 / 6.57 | 0.832 / 0.832 |
| cauchit | 6.19 / 6.19 | 8.63 / 8.63 | 5.23 / 5.23 | 7.78 / 7.78 |
| cloglog | 6.73 / 3.12 | 9.98 / 7.61 | 4.16 / 3.34 | 0.969 / 0.969 |
| loglog | 4.58 / 3.23 | 11 / 8.49 | 4.84 / 3.38 | 0.883 / 0.883 |
| inverse | 3.92 / 3.92 | 3.25 / 3.25 | 3.92 / 3.92 | 3.25 / 3.25 |
| inverse_sq | 3.09 / 3.09 | 7.29 / 7.29 | 3.86 / 3.86 | 2.79 / 2.79 |
| sqrt | 10.9 / 10.9 | 0.246 / 0.246 | 2.43 / 2.43 | 2.95 / 2.95 |
| atanh | 2.78 / 2.78 | 2.67 / 1.67 | 3.53 / 5.05 | 2.67 / 1.67 |
| asinh | 8.55 / 9.49 | 0.494 / 0.622 | 5.35 / 7.85 | 1.9 / 2.27 |

(The inverse log-Jacobian of log is u itself; its other rows are constants
and are checked exactly.) In plain ulps the values and first derivatives
(orders 0-1) are within 4.1 ulp on gcc, except for cloglog (up to 147 ulp)
and loglog (up to 1430 ulp), whose values and first derivatives cross zero
(loglog at v = 1/e) or are ill-conditioned (exp(u - e^u) carries the rounding
of e^u); their ulpC there is at most 2.01. Timings not measured yet.

**Distances to the bounds** (added 2026-11-12). `b.inverse_distance_upper(u)` and
`b.inverse_distance_lower(u)` return the distance of h(u) to the upper and
lower bound of its range, computed without forming h where h saturates at a
bound other than 0 or infinity: 1 - h(u) is expit(-u) for logit, Phi(-u) for
probit, h(-u) for cauchit, exp(-e^u) for cloglog, the cloglog inverse at -u
for loglog, 2e/(1 + e) with e = exp(-2u) for 1 - tanh u; elsewhere the bound
minus h(u). References: the same script, bound - h and h - bound evaluated as
written with 1200 more bits (rows below 2^-1022 dropped), for the six maps
whose range is (0, 1) or (-1, 1). ulpC scaled with h'(u), the derivative of
the distance. Maximum over the six maps and the two distances: 2.28 ulpC on
gcc and on CUDA (probit, lower distance); in plain ulps up to 130 (cloglog,
upper) and 173-174 (loglog, lower), where exp(-e^u) carries the rounding of
e^u, condition number u e^u.

**What this says.** Every quantity is within 11 ulps of the error that the
rounding of its argument alone would cause, on both builds; the largest
errors are in the derivatives of order 5-6 near their zeros.

**Chosen.** This version. Tested in `tests/bijector/test_scalar.cpp` on both
builds, with bounds of half again the measured errors per map, and exactly
for the constant derivatives, the identity, inversion and the domain.

**Closed forms of the derivatives.** U, D, T, S, s, r, t and w in the lists below are the intermediate quantities of the corresponding function in `include/laplace/bijector/scalar.hpp`.
- Logit: h^(k)(u) = U (A_k(U) + D B_k(U)), with h'(u) = U, h''(u) = U D, h^(3)(u) = U (1 - 6U), h^(4)(u) = U D (1 - 12U), h^(5)(u) = U (1 - 30U + 120U^2) and h^(6)(u) = U D (1 - 60U + 360U^2).
- Probit: Q_1(z) = 1, Q_2(z) = z, Q_3(z) = 1 + 2z^2, Q_4(z) = z (7 + 6z^2), Q_5(z) = 7 + 46z^2 + 24z^4, Q_6(z) = z (127 + 326z^2 + 120z^4); R_1(z) = z, R_2(z) = 1 + z^2, R_3(z) = z (4 + 2z^2), R_4(z) = 4 + 18z^2 + 6z^4, R_5(z) = z (52 + 96z^2 + 24z^4), R_6(z) = 52 + 548z^2 + 600z^4 + 120z^6. The derivatives of the inverse probit are h'(u) = phi(u), h''(u) = -u phi(u), h^(3)(u) = (u^2 - 1) phi(u), h^(4)(u) = u (3 - u^2) phi(u), h^(5)(u) = (3 - 6u^2 + u^4) phi(u) and h^(6)(u) = u (-15 + 10u^2 - u^4) phi(u).
- Cauchit: T_1(z) = 1 + z^2, T_2(z) = 2z (1 + z^2), T_3(z) = 2 + 8z^2 + 6z^4, T_4(z) = z (16 + 40z^2 + 24z^4), T_5(z) = 16 + 136z^2 + 240z^4 + 120z^6, T_6(z) = z (272 + 1232z^2 + 1680z^4 + 720z^6); W_0(s) = 1, W_1(s) = -2s, W_2(s) = 8s^2 - 2 = 6 - 8r^2, W_3(s) = s (24 - 48s^2) = s (48r^2 - 24), W_4(s) = 24 - 288s^2 + 384s^4 = 120 - 480r^2 + 384r^4, W_5(s) = s (-720 + 3840s^2 - 3840s^4) = s (-720 + 3840r^2 - 3840r^4); X_0(s) = s, X_1(s) = 1 - 2s^2 = 2r^2 - 1, X_2(s) = s (8s^2 - 6) = s (2 - 8r^2), X_3(s) = -6 + 48s^2 - 48s^4 = -6 + 48r^2 - 48r^4, X_4(s) = s (120 - 480s^2 + 384s^4) = s (24 - 288r^2 + 384r^4), X_5(s) = 120 - 2160s^2 + 5760s^4 - 3840s^6 = -120 + 2160r^2 - 5760r^4 + 3840r^6.
- Inverse and inverse square: c_0 = 1, c_1 = -1/2, c_2 = 3/4, c_3 = -15/8, c_4 = 105/16, c_5 = -945/32, c_6 = 10395/64.
- Square root: d_0 = 1, d_1 = 1/2, d_2 = -1/4, d_3 = 3/8, d_4 = -15/16, d_5 = 105/32, d_6 = -945/64.
- Inverse hyperbolic tangent: P_1 = 1, P_2 = -2T, P_3 = 6T^2 - 2 = 4 - 6S, P_4 = T (16 - 24T^2) = T (24S - 8), P_5 = 16 - 120T^2 + 120T^4 = 16 - 120S + 120S^2, P_6 = T (-272 + 960T^2 - 720T^4) = T (-32 + 480S - 720S^2).
- Inverse hyperbolic sine: V_0 = 1, V_1 = -s, V_2 = 3s^2 - 1 = 2 - 3r^2, V_3 = s (9 - 15s^2) = s (15r^2 - 6), V_4 = 9 - 90s^2 + 105s^4 = 24 - 120r^2 + 105r^4, V_5 = s (-225 + 1050s^2 - 945s^4) = s (-120 + 840r^2 - 945r^4).
- Complementary log-log: C_1(t) = 1, C_2(t) = -(1 + t), C_3(t) = 2 + 3t + 2t^2, C_4(t) = -(6 + 12t + 11t^2 + 6t^3), C_5(t) = 24 + 60t + 70t^2 + 50t^3 + 24t^4, C_6(t) = -(120 + 360t + 510t^2 + 450t^3 + 274t^4 + 120t^5); E_1(t) = -(1 + t), E_2(t) = 1 + t + t^2, E_3(t) = -(2 + 3t + 2t^2 + 2t^3), E_4(t) = 6 + 12t + 11t^2 + 6t^3 + 6t^4, E_5(t) = -(24 + 60t + 70t^2 + 50t^3 + 24t^4 + 24t^5), E_6(t) = 120 + 360t + 510t^2 + 450t^3 + 274t^4 + 120t^5 + 120t^6; P_1(w) = 1, P_2(w) = 1 - w = -expm1(u) (evaluated through expm1 to avoid the cancellation near u = 0), P_3(w) = 1 - 3w + w^2, P_4(w) = 1 - 7w + 6w^2 - w^3, P_5(w) = 1 - 15w + 25w^2 - 10w^3 + w^4, P_6(w) = 1 - 31w + 90w^2 - 65w^3 + 15w^4 - w^5.

## The derivative of order 6 of log Phi

**What.** `log_Phi_d6(t)` = d^6 log Phi(t) / dt^6, for the truncation term
-log Phi(t) of the Box-Cox bijector at order 6 (decision 5 of
`note/bijectors.md`). The other orders are unchanged.

**How.** The same three routes as orders 1-5: the piecewise Chebyshev
expansion on [-1, 4) (`tools/reference/log_phi_cheb.py`, now generating
orders 1-6; g6 has degree 20 and a table error of 0.072 ulp1; the
coefficients of orders 1-5 are bit for bit the earlier ones); above 4 the
recursion from R = phi/Phi, one more step
g6 = -g5 (t + 2R) - 4 g4 (1 + 2 g2) - 6 g3^2; below -1 the cumulants of
exp(-x s - s^2/2) on s > 0 (x = -t), kappa_6 from the raw moments.
References `log_Phi_d6.csv` and `log_Phi_d6_mid.csv`
(`tools/reference/normal.py`; the extra working precision in the left tail is
16 log2|t| bits for order 6 instead of 12, since the cancellation grows with
the order: 12 failed the two-precision check at t = -4.2e15).

**Measurement.** Errors as for the other orders: relative below t = -1 (no
zeros there), in units of ulp(max(|g|, 1)) above.

| | t < -1 (ulp) | t >= -1 (ulp1) |
|---|---|---|
| gcc | 767 (at t = -1.10) | 0.43 |
| CUDA | 478 (at t = -1.01) | 0.34 |

By band on gcc: below -5 within 20 ulp (as order 5: 18); between -3 and -1,
where the cumulant recursion converges slowly (x near 1), 72-767 ulp, that is
1.1-6.0 ulp1, against 40-81 ulp (0.6-2.5 ulp1) for order 5. The relative
figure is inflated because g6 is small there (0.0134 at t = -1.1, on its way to
a zero just above -1).

**Chosen.** This version. Tested in `tests/math/test_normal.cpp` with bounds
1150 ulp (t < -1) and 1.0 ulp1 (t >= -1), on both builds.

**Open.** Extending the Chebyshev range from [-1, 4) down to -3 would remove
the slow band for every order (orders 4-6); not done, it changes orders 1-5.

## exprel and log1prel with their derivatives to order 6

**What.** `include/laplace/math/exprel.hpp`: phi(x) = (e^x - 1)/x, phi(0) = 1
(`exprel`, the name of GSL and SciPy), and its derivatives
phi^(k)(x) = int_0^1 t^k e^(x t) dt, k = 1..6 (`exprel_d1` ... `exprel_d6`).
`include/laplace/math/log1prel.hpp`: psi(x) = log(1 + x)/x for x > -1,
psi(0) = 1 (`log1prel`), and psi^(k)(x) = (-1)^k k! int_0^1 t^k (1 + x t)^(-(k+1)) dt
(`log1prel_d1` ... `log1prel_d6`). Both integrands keep one sign, so phi^(k)
is positive and psi^(k) has the sign (-1)^k everywhere: no zeros, and the
plain relative ulp is the measure.

**How, exprel.** The value as expm1(x)/x. The derivatives of order K:
- 0 < x < T+_K: the power series sum_n x^n/(n! (n + K + 1));
- -T-_K < x < 0: Kummer's form e^x sum_n (-x)^n K!/(n + K + 1)!, the same
  integral after t -> 1 - t, whose terms are all positive (the power series
  cancels there: 35-143 ulp at |x| in [2, 4) and 3e3-5e4 ulp in [4, 8) for
  orders 1, 3, 6);
- elsewhere the forward recursion phi^(K) = (e^x - K phi^(K-1))/x, from phi;
  for x >= 700 on the quantities scaled by e^-700, so that every result whose
  exact value is a finite double is finite (+inf above 740).
Measured thresholds (T+_K, T-_K): K = 1 (1, 2), 2 (2, 2), 3 (2, 4), 4 (4, 4),
5 (4, 8), 6 (4, 8).

**How, log1prel.** The value as log1p(x)/x. The derivatives, by region:
- |x| < 1/4: the power series of psi differentiated K times;
- -4/5 < x <= 6: the 32-point Gauss-Legendre rule on the integral (terms of
  one sign; the integrand has its pole at t = -1/x, outside [0, 1]);
- 6 < x <= 20: the same rule on each of [0, 1/4] and [1/4, 1], 64 nodes;
- x <= -4/5 and x > 20: the forward recursion
  psi^(K) = ((-1)^(K-1) (K-1)!/(1 + x)^K - K psi^(K-1))/x, from psi.
The recursion alone cancels in the middle range (order 6: 202 ulp on
[-0.5, -0.3), 1.6e4 on [0.3, 0.5), 547 on [1, 1.5)), and the series converges
only for |x| < 1. Between 6 and 20 the recursion lost up to 17.2 ulp at
order 6 near x = 8; one rule on [0, 1] does not resolve the peak of the
integrand near t = K/x, two do. A third candidate, the rule after
s = log(1 + x t), psi^(K) = (-1)^K K! x^-(K+1) int_0^log1p(x) (1 - e^-s)^K ds,
was within 7.7 ulp but cost 800-1000 ns per call (32 calls of expm1), and it
fails above 1e5: discarded.

**Measurement.** References `exprel*.csv` (`tools/reference/exprel.py`:
mpmath.quad of the integral) and `log1prel*.csv` (`tools/reference/log1prel.py`:
a closed form of the integral at raised precision, checked against
mpmath.quad to 1.7e-60), each row at 200 and 300 bits, kept if the two agree
to 2^-100. 1721 rows per order for exprel (|x| log-uniform from 2^-60, up to
745 on both sides, around every threshold), 1792-2132 for log1prel (1 + x
down to 2^-53, x up to 1e300). Maximum ulp, gcc / CUDA:

| order | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| exprel | 1.81 / 1.52 | 2.71 / 2.64 | 4.11 / 4.11 | 5.90 / 5.90 | 4.25 / 4.25 | 5.85 / 5.85 | 5.92 / 6.57 |
| log1prel | 1.28 / 1.28 | 3.94 / 3.94 | 5.03 / 5.03 | 6.39 / 6.39 | 7.06 / 7.06 | 8.27 / 8.27 | 9.60 / 9.60 |

ns per call on one gcc thread (`benchmarks/math_rel.cpp`, measured while a
CUDA build occupied the machine): exprel 16-44 (the value 16-31, the
derivatives 30-44); log1prel 20-60 in the series region, 20-37 with one rule,
59-68 with two, 22-50 with the recursion. On CUDA (RTX 4070 SUPER, FP64 at
1/64 of FP32): exprel 0.44 for the value, 0.5-3.3 for the derivatives;
log1prel 0.46-0.63 for the value, 2.8-3.7 for the derivatives with the series
or one rule, 6.1-7.6 with two rules, 0.54-0.79 with the recursion. The two-rule
quadrature costs about 2.5 times the recursion on one gcc thread and about 10
times on this GPU. Kept on every backend (Giovanni, 2026-11-12): the same
algorithm and accuracy on CPU and GPU.

**What this says.** Both functions are within 10 ulp at every order on both
builds, at a few tens of ns per call.

**Chosen.** This version. Tested in `tests/math/test_exprel.cpp` and
`tests/math/test_log1prel.cpp` on both builds, with bounds of half again the
measured errors and exact checks at 0 (phi^(k)(0) = 1/(k + 1),
psi^(k)(0) = (-1)^k k!/(k + 1)), at the limits and outside the domain.

**Stopping rules and reference construction.** The series of exprel are summed until a term is at most 4 eps of the partial sum, with iteration cap 200 and eps = 2^-52 = 2.22e-16; the second series of exprel uses the same criterion and cap. For x >= 700.0 the recurrence evaluates the scaled quantities phi^(j)(x) e^-700 with e^(x - 700), where x - 700 is exact by Sterbenz's lemma and e^(x - 700) does not overflow; the final multiplication by e^700 overflows only when the exact result exceeds `DBL_MAX`, from x = 716.5 (approximately) onward. The series of log1prel terminates when |tau_m| <= 4 eps |sum|, with iteration cap 300. The exprel references use the integral of t^k e^(x t) over [0, 1] evaluated by `mpmath.quad` (tanh-sinh quadrature on subintervals of [0, 1] with clustered breakpoints at both ends) at 200 and 300 bits, and agree to 2^-100 relative. The log1prel references use the closed form psi^(k)(x) = (-1)^k k! x^-(k+1) B_k(1 + x), with B_k(a) = log a + sum_(j=0..k-1) C(k, j) (-1)^(k-j) (a^(j-k) - 1)/(j - k), evaluated at P = 200 and P + 100 bits with extra precision (k + 1) log2(1/|x|) + 64 bits near x = 0 to overcome the cancellation, verified to agree to 2^-100 relative and checked against `mpmath.quad`.

## Scalar bijectors with parameters, and the speed of every scalar bijector

**What.** Eight maps added to `include/laplace/bijector/scalar.hpp`, one header
each in `include/laplace/bijector/detail/`, with the parameters as fields of
`scalar_bijector` (location, scale, exponent, sharpness, lower, upper) and
static construction functions:
- `affine`: g(v) = (v - location)/scale, scale != 0 (decreasing for a
  negative scale);
- `power`: g(v) = v^exponent on (0, inf), exponent != 0, no derivatives in the
  exponent; `scalar_bijector::power` maps the exponents 1, 1/2, -1, -2, 0 to
  identity, sqrt, inverse, inverse_sq, log;
- `softplus`: g(v) = log(e^(a v) - 1)/a, a = sharpness > 0;
- `bounded`: g(v) = log(v - a) - log(b - v) on (a, b);
- `lower_bounded`: g(v) = log(v - a); `upper_bounded`: g(v) = -log(b - v),
  increasing;
- `box_cox`: g(v) = (v^lambda - 1)/lambda, log v at lambda = 0;
- `yeo_johnson`: Box-Cox of 1 + v with lambda for v >= 0, minus Box-Cox of
  1 - v with 2 - lambda for v < 0.
For each, as for the maps without parameters: both directions and both
log-Jacobians to order 6, and the partial derivatives of total order 1-6 in
(argument, parameters), returned as `bijector_partials<K>` (components
ordered by the number of derivatives in the second parameter, then in the
first; the pure derivative is c[0]). `bijector_kinks` declares the point at
which Yeo-Johnson is only C^2 (v = 0, unless lambda = 1). Invalid parameters
give NaN everywhere.

**How.** The forms, each checked against SymPy's
derivatives of the definitions (scripts `bijector_*_forms.py` and the
reference scripts):
- affine and the shifted maps: closed forms in t = v - p1 or p1 - v, single
  subtractions, exact near the bound; the distances e^u and e^-u exactly;
- bounded: the logit structure with A = 1/(v - a), B = 1/(b - v), and in the
  forward log-Jacobian the pairs B^K - S^K, A^K - S^K (S = 1/(b - a))
  rewritten with the exact factors B - S = p S B and A - S = q S A (written
  as differences they gave 1e15 ulpC near a bound; 0.6 after);
- power: the falling factorial times v^(lambda - K), the exponent split as an
  exact two-sum, with a halved exponent where v^(lambda - K) under- or
  overflows while the result does not;
- softplus: h^(K)(u) = a^(K-1) expit^(K-1)(a u); the derivatives in a by
  Leibniz forms with the part linear in x = a u cancelled analytically; the
  forward log-Jacobian's mixed orders by a Bernoulli series below x = 3;
- box_cox: g = L exprel(lambda L) (L = log v) for |lambda L| < 1, the power
  beyond; d_lambda^k g = L^(k+1) exprel^(k)(lambda L); derivatives involving v
  through v^(lambda - m) formed as v^lambda divided by v m times; the inverse
  as exp(S), S = u log1prel(lambda u), its lambda-derivatives through the
  Bell polynomials of the derivatives of S;
- yeo_johnson: the same Box-Cox forms on each branch, with log1p for v and u
  near 0, generated with SymPy common-subexpression elimination.

Three changes made in the review, all measured:
- **Dispatch not inlined on the device.** With the map chosen at run time the
  test kernels held the code of all twenty maps and every order in one device
  function; ptxas failed ("memory allocation failure") even with three build
  jobs. The dispatch functions are `__noinline__` on the device
  (`LAPLACE_BIJ_NOINLINE`, as `student.hpp` does).
- **Box-Cox partials.** The Bell polynomials of the lambda-derivatives of S
  were rebuilt for every component (21 calls of log1prel_dk at order 6):
  1188 ns per call of `inverse_partials_d6` on one gcc thread. Formed once,
  with a new internal `log1prel` routine that returns every order at one
  point in one pass (`detail_log1prel::derivs_upto`, bit-identical to
  `log1prel_dk`): 310 ns.
- **Powers on the host.** The platform's pow of the gcc build (Rtools) costs
  about 150 ns per call: `power` took 155 ns for its value, Box-Cox and
  Yeo-Johnson 150-175 ns for their sixth derivatives. `pow_pos(x, y)` (in
  `scalar_helpers.hpp`) is exp(y log x) on the host, with log x from
  `ddm::log_split` and the exponent carried as a double-double, and the
  platform's pow on the device (under 1 ns): `power` 155 -> 42 ns, the sixth
  derivatives of Box-Cox 173 -> 47 and of Yeo-Johnson 152 -> 37, accuracy
  unchanged except for `power`, from at most 3.4 to at most 5.9 ulpC (its
  derivatives split the exponent and call pow twice; the full double-double
  log gave the same errors at 580 ns). Chosen by Giovanni, 2026-11-12. The
  integer powers of softplus are repeated products (`pow_int`).

**Measurement.** References `bijector_<map>.csv`: SymPy differentiates the
definitions in the argument and the parameters to total order 7, mpmath
evaluates them at two precisions, rows kept if they agree to 2^-100
(`tools/reference/bijector_param_common.py` and one script per map;
6239-62203 rows per map). The Box-Cox rows near lambda = 0 were checked
against an independent evaluation at 300 bits: they agree to 6e-33. The test
(`tests/bijector/param_common.hpp`) checks every derivative, mixed ones
included, in ulpC with the scale |f| + |x df/dx| + |p1 df/dp1| + |p2 df/dp2|,
and checks that c[0] of the partials equals the pure derivative. Maximum ulpC
over orders 0-6, gcc / CUDA:

| map | forward | inverse | forward log-Jacobian | inverse log-Jacobian | distances |
|---|---|---|---|---|---|
| affine | 0.362 / 0.362 | 0.5 / 0.5 | 0.495 / 0.495 | 0.495 / 0.495 | - |
| power | 4.76 / 4.37 | 5.59 / 4.93 | 5.88 / 5.88 | 5.87 / 5.87 | - |
| softplus | 7.67 / 7.04 | 4.38 / 4.25 | 18.4 / 18.4 | 3.48 / 3.47 | - |
| bounded | 1.1 / 1.1 | 7.09 / 5.82 | 1.1 / 1.1 | 7.09 / 5.82 | 0.721 / 0.721, 0.584 / 0.584 |
| lower_bounded | 0.679 / 0.679 | 0.725 / 0.725 | 0.679 / 0.679 | 0 / 0 | 0.273 / 0.273 |
| upper_bounded | 0.679 / 0.679 | 0.467 / 0.467 | 0.679 / 0.679 | 0 / 0 | 0.193 / 0.193 |
| box_cox | 7.97 / 9.71 | 42.8 / 46.3 | 1.28 / 1.28 | 49.4 / 43 | - |
| yeo_johnson | 26 / 26 | 24.9 / 39.9 | 3.06 / 3.06 | 46.3 / 27.3 | - |

The largest errors: the inverse of Box-Cox and its log-Jacobian at orders 4-6
(up to 49 ulpC, where |lambda u| is of order 1-10 and the mixed sums
cancel); Yeo-Johnson's forward map at |v| near 1e18 (26 ulpC) and its inverse
log-Jacobian at order 6 (46 ulpC at u = -3, lambda = 0.4); softplus's forward
log-Jacobian at order 6 near a v = 3 (18 ulpC). Not reduced.

Speed of every scalar bijector, `benchmarks/bijector_scalar.cpp` (2^16
points inside each domain, typical parameters), ns per call: one gcc thread /
OpenMP with 6 threads / CUDA (RTX 4070 SUPER):

| map | forward | forward_d6 | inverse | inverse_d6 | log-Jacobian fwd / inv | inverse_partials_d1 | inverse_partials_d6 | distance |
|---|---|---|---|---|---|---|---|---|
| identity | 4 / 1.1 / 0.40 | 5 / 1.2 / 0.39 | 4 / 1.0 / 0.40 | 4 / 1.5 / 0.39 | 4 / 4 / 0.38-0.39 | 6 / 1.4 / 0.42 | 15 / 2.9 / 0.63 | 5 / 1.2 / 0.42 |
| log | 7 / 1.6 / 0.50 | 5 / 1.2 / 0.43 | 10 / 3.4 / 0.46 | 10 / 2.0 / 0.45 | 7 / 4 / 0.39-0.51 | 15 / 2.9 / 0.48 | 24 / 4.7 / 0.68 | 11 / 2.3 / 0.48 |
| logit | 44 / 7.6 / 1.09 | 5 / 1.2 / 0.54 | 11 / 2.2 / 0.50 | 30 / 5.5 / 0.63 | 43 / 49 / 0.65-1.07 | 15 / 3.4 / 0.54 | 42 / 7.4 / 0.83 | 10 / 2.9 / 0.49 |
| probit | 34 / 6.1 / 2.95 | 49 / 8.7 / 3.64 | 19 / 3.4 / 1.78 | 11 / 3.2 / 0.51 | 36 / 4 / 0.40-2.95 | 14 / 2.9 / 0.51 | 24 / 4.5 / 0.72 | 18 / 5.2 / 1.78 |
| cauchit | 10 / 2.0 / 0.77 | 11 / 2.3 / 0.79 | 42 / 7.4 / 0.59 | 9 / 2.7 / 0.56 | 46 / 37 / 0.87-1.36 | 11 / 2.2 / 0.53 | 23 / 4.4 / 0.77 | 41 / 7.5 / 0.59 |
| cloglog | 42 / 7.6 / 0.73 | 36 / 6.6 / 0.68 | 27 / 4.8 / 0.55 | 26 / 4.7 / 0.56 | 44 / 10 / 0.46-0.73 | 27 / 4.9 / 0.56 | 36 / 6.6 / 0.76 | 19 / 3.4 / 0.53 |
| loglog | 19 / 3.6 / 0.64 | 11 / 3.1 / 0.58 | 19 / 3.6 / 0.54 | 26 / 4.7 / 0.56 | 19 / 10 / 0.46-0.64 | 27 / 5.0 / 0.56 | 35 / 6.3 / 0.77 | 24 / 4.5 / 0.56 |
| inverse | 4 / 1.6 / 0.40 | 5 / 1.3 / 0.44 | 4 / 1.1 / 0.41 | 4 / 1.1 / 0.44 | 8 / 7 / 0.52-0.53 | 6 / 1.5 / 0.44 | 16 / 3.2 / 0.64 | 4 / 1.1 / 0.42 |
| inverse_sq | 4 / 1.0 / 0.41 | 5 / 1.3 / 0.43 | 4 / 1.5 / 0.44 | 5 / 1.2 / 0.50 | 8 / 7 / 0.50-0.51 | 6 / 1.6 / 0.48 | 20 / 3.8 / 0.70 | 4 / 1.1 / 0.46 |
| sqrt | 4 / 1.4 / 0.42 | 5 / 1.3 / 0.48 | 4 / 1.0 / 0.39 | 4 / 1.4 / 0.40 | 8 / 7 / 0.51-0.52 | 6 / 1.3 / 0.42 | 15 / 2.7 / 0.61 | 4 / 1.0 / 0.40 |
| atanh | 55 / 9.2 / 0.84 | 5 / 1.2 / 0.54 | 29 / 5.3 / 0.56 | 30 / 5.8 / 0.64 | 44 / 53 / 0.78-0.92 | 14 / 2.8 / 0.53 | 44 / 8.1 / 0.84 | 20 / 3.8 / 0.68 |
| asinh | 31 / 5.7 / 0.65 | 10 / 2.4 / 0.53 | 10 / 1.9 / 0.56 | 10 / 2.1 / 0.55 | 37 / 51 / 0.77-0.88 | 14 / 2.7 / 0.50 | 23 / 4.3 / 0.75 | 11 / 2.2 / 0.57 |
| affine | 4 / 2.4 / 0.46 | 4 / 1.1 / 0.41 | 4 / 1.4 / 0.44 | 4 / 1.1 / 0.41 | 5 / 4 / 0.55-0.56 | 5 / 1.6 / 0.44 | 21 / 4.2 / 0.75 | 3 / 1.3 / 0.42 |
| power | 42 / 7.2 / 0.77 | 46 / 8.2 / 0.99 | 50 / 8.6 / 0.95 | 68 / 11.7 / 1.23 | 11 / 11 / 0.66-0.70 | 60 / 10.8 / 1.08 | 81 / 14.3 / 1.46 | 3 / 0.9 / 0.40 |
| softplus | 37 / 6.6 / 0.92 | 47 / 8.5 / 0.89 | 70 / 12.1 / 0.97 | 33 / 7.1 / 0.91 | 35 / 68 / 0.90-0.91 | 70 / 12.0 / 1.34 | 110 / 20.0 / 6.07 | 3 / 0.9 / 0.41 |
| bounded | 9 / 2.2 / 0.58 | 11 / 2.7 / 0.54 | 27 / 5.1 / 0.56 | 29 / 5.2 / 0.67 | 12 / 50 / 0.62-0.82 | 14 / 5.5 / 0.73 | 49 / 9.1 / 0.94 | 29 / 5.3 / 0.54 |
| lower_bounded | 7 / 1.6 / 0.52 | 4 / 1.1 / 0.44 | 10 / 2.2 / 0.47 | 10 / 2.3 / 0.46 | 8 / 4 / 0.40-0.53 | 12 / 2.2 / 0.47 | 27 / 4.9 / 0.64 | 3 / 1.0 / 0.40 |
| upper_bounded | 8 / 2.0 / 0.53 | 4 / 1.1 / 0.44 | 11 / 2.1 / 0.47 | 11 / 2.1 / 0.46 | 8 / 4 / 0.39-0.53 | 11 / 2.2 / 0.48 | 27 / 4.9 / 0.64 | 9 / 2.9 / 0.47 |
| box_cox | 35 / 6.1 / 1.06 | 47 / 8.2 / 1.00 | 39 / 6.7 / 0.78 | 65 / 11.5 / 0.83 | 7 / 28 / 0.54-0.72 | 78 / 13.4 / 7.91 | 310 / 56.3 / 46.5 | 4 / 1.0 / 0.47 |
| yeo_johnson | 46 / 8.7 / 1.16 | 37 / 6.8 / 1.31 | 47 / 9.5 / 1.19 | 36 / 6.6 / 1.26 | 32 / 30 / 0.87-1.03 | 78 / 13.5 / 14.1 | 359 / 60.9 / 99.5 | 3 / 1.0 / 0.45 |

On CUDA the slowest are the partials of order 6 of Box-Cox (47 ns) and
Yeo-Johnson (99 ns), whose divergent branches and calls of the exprel and
log1prel derivatives cost most; everything else is within 4 ns.

**What this says.** Every quantity of every map is within 50 ulpC on both
builds (within 8 ulpC for all maps except Box-Cox, Yeo-Johnson and softplus's
order-6 log-Jacobian), and within a few tens of ns per call on one gcc thread
and a few ns on the GPU, except the order-6 partials of Box-Cox and
Yeo-Johnson.

**Chosen.** This version. Tested on both builds by one test per file in
`tests/bijector/` (bounds 1.5 times the larger of the gcc and CUDA maxima),
with exact checks of the special values, the domains, invalid parameters,
inversion, the distances, the construction functions and the kinks.

**Open.** The order-6 partials of Yeo-Johnson (359 ns on gcc, 99 on CUDA) could
share the derivatives of log1prel as Box-Cox now does; the largest errors
listed above were not reduced. (Both taken up in the next section.)

**Bounded map.** Evaluating g(v) as log(p/q) rather than log p - log q avoids a cancellation of 18 eps near the midpoint.

## Refinements of the Box-Cox, Yeo-Johnson and softplus bijectors

**What.** The two open items of the previous section: the speed of the
order-6 partials of Yeo-Johnson, and the largest errors of Box-Cox,
Yeo-Johnson and softplus, each traced to its cause by measurement.

**Speed.**
- `scalar_yeo_johnson.hpp` (generated by
  `tools/reference/bijector_yeo_johnson_forms.py`): every derivative of
  log1prel that one function needs is at one argument per branch; the
  generator now takes them from one call of `detail_log1prel::derivs_upto`
  instead of one call of `log1prel_dK` per order (each a 32- or 64-node
  quadrature). inverse_partials_d6 on one gcc thread: 367 -> 235 ns.
- `derivs_upto` itself cost 110-170 ns at the points of the bench, the time
  of 7 divisions per quadrature node. It now forms r = t/d and r/d (d = 1 + x t) and the orders
  by products with r: two divisions per node and the same number of
  roundings as `log1prel_dK`. Measured against the log1prel references in
  the quadrature regions (534 rows per order), maximum ulp of derivs_upto /
  log1prel_dK: order 1 3.94 / 3.94, 2 4.20 / 4.10, 3 5.71 / 5.46, 4 4.72 /
  3.92, 5 4.09 / 4.45, 6 7.32 / 7.32; within the maxima of log1prel_dK over
  all regions (3.94-9.60), but no longer bit-identical to it. 141 -> 93 ns
  at the points of the bench. One division per node (multiplying by 1/d)
  gave 84 ns but up to 9.32 ulp at order 6: not taken. The same reciprocal
  in `log1prel_dK` itself made its quadrature 30-45 per cent faster, with
  errors up to 6.3 ulp instead of 3.8 at order 6: not taken. A version
  templated on the number of orders gave nothing on gcc and cost 4 ns more
  at order 1 on CUDA: not taken.
- Box-Cox's inverse partials: the binomial coefficients of the inner sums by
  Pascal's rule, once per call (bc_binom divided at every term): 290 -> 260
  ns. The binomial coefficients of Box-Cox and softplus are now computed in
  integers (FP64 division is slow on this GPU): softplus inverse_partials_d6
  on CUDA 4.4 -> 3.4 ns.
- The CUDA build of the tests ran out of memory in ptxas with 6 jobs
  (test_yeo_johnson.cpp); it builds with 2.
- Not changed: the order-1 partials of Box-Cox and Yeo-Johnson on CUDA (8
  and 11 ns) spend their time in the log1prel quadrature (32 nodes, two FP64
  divisions each).

**Accuracy.** Each cause measured with mpmath (scratch scripts, not kept): the ratio of the sum of the absolute values
of the terms of a formula to the ulpC scale, against the measured error.
- **Yeo-Johnson, inverse log-Jacobian** (46.3 ulpC at u = -3, lambda = 0.4,
  order (3, 3)): the mixed derivatives are rational in u and lambda, and
  SymPy's expanded form has terms 96 times the result there (46 at u = -8,
  38 at u = -30). Written in e = 1/(1 + z), z = lambda u or (lambda - 2) u,
  and factored, each is a power of e times a short polynomial in e: 13 times
  more accurate at that point. The factored form divides by lambda (or
  2 - lambda), so it is used for |z| > 1/2 only. The positive branch is the
  inverse log-Jacobian of Box-Cox, (1 - lambda) log(1 + lambda u)/lambda, for
  every u > -1/lambda: the generator writes it as its own function
  (`yeo_johnson_inv_lj_pos_partials`), which Box-Cox calls for its partials
  (its closed sums T_(K,n) cancelled in the same way: 49.4 ulpC at u = -8,
  lambda = -1.5).
- **Yeo-Johnson, forward map at |v| = 1.15e18** (26 ulpC): the hypothesis of
  the previous section (exp(lambda M) instead of the rounding of log1p) was
  half right. Two causes, both amplified by log|v| = 41: the exponents
  lambda - n and 2 - lambda - n of the pure v-derivatives are rounded (eps
  absolute, times 41: about 20 ulpC in g''), and the lambda-derivatives
  exprel^(n)(c N) take exp of c N with N = log1p(-v) and the product rounded
  (0.5 ulp of 83). Now every power (1 + z)^(k + a), a = lambda or -lambda,
  is pow_pos(1 + z, a) times or divided by 1 + z |k| times (`yj_pow`, as
  Box-Cox's bc_pow_shift), and for 8 <= |c N| < 700 and |v| >= 1 the
  exprel recursion starts from that power instead of exp(c N)
  (`yj_exprel_d`). 26 -> 5.8 ulpC (gcc), 26 -> 5.3 (CUDA). Costs 37 -> 46 ns
  for forward_d6 on one gcc thread, 1.31 -> 1.69 on CUDA.
- **Yeo-Johnson and Box-Cox, inverse map** (YJ 27.8 ulpC at u = -8,
  lambda = 0, order (4, 1); Box-Cox 42.8 at u = -8.5, lambda = 9e-13, order
  (3, 3)): not reduced. The terms of the product-rule sums are 36 (YJ) and
  75 (Box-Cox) times the ulpC scale there; at lambda = 0 the scale has no
  lambda term, and the true derivative is small against its parts. A second
  hypothesis for YJ, the exponential of a rounded product exp(c u
  log1prel(z)) replaced by the power (1 + z)^(-n + 1/s) for |z| > 1/2, was
  measured and was wrong: 40.9 ulpC instead of 27.8, and 372 ns instead of
  235 for the order-6 partials. The log of h^(K) (Bell polynomials of the derivatives of
  log P_K + (1 - K lambda) S) cancels in the same way: x_j - j K x_(j-1).
- **softplus, forward log-Jacobian at order 6** (18.4 ulpC at a v = 7.56,
  a = 0.1, order (3, 3)): not reduced. The sum
  sum_i C(3, i) 3!/(3 - i)! x^(3 - i) t^(5 - i)(x), t = 1/expm1(x), has terms
  20.3 times its value at x = 7.56 (for large x it tends to e^-x times a
  Laguerre polynomial in x). A remedy would need the roots of these
  polynomials, or the series of t in e^-mx term by term.

**Measurement.** The tests of `tests/bijector/` with LAPLACE_PRINT_ERRORS=1,
maximum ulpC over orders 0-6, gcc / CUDA (the other maps and quantities are
as in the previous section):

| map | forward | inverse | forward log-Jacobian | inverse log-Jacobian |
|---|---|---|---|---|
| box_cox | 7.97 / 9.71 | 42.8 / 46.3 | 1.28 / 1.28 | 6.61 / 5.22 |
| yeo_johnson | 5.83 / 5.33 | 27.8 / 30.5 | 3.06 / 3.06 | 6.38 / 5.88 |

The inverse of Yeo-Johnson moved from 24.9 / 39.9 to 27.8 / 30.5 with the
derivatives of log1prel from derivs_upto (orders 3-4 from 5.8 to 8.2 / 11.8
on CUDA).

Speed, `benchmarks/bijector_scalar.cpp`, ns per call, one gcc thread /
OpenMP with 6 threads / CUDA:

| map | forward | forward_d6 | inverse | inverse_d6 | inverse_partials_d1 | inverse_partials_d6 |
|---|---|---|---|---|---|---|
| softplus | unchanged | unchanged | unchanged | unchanged | 70 / 12.5 / 1.22 | 116 / 20.7 / 3.42 |
| box_cox | 35 / 6.1 / 1.06 | 50 / 9.6 / 1.10 | 40 / 7.3 / 0.78 | 68 / 12.0 / 0.83 | 78 / 14.0 / 8.34 | 268 / 45.8 / 27.8 |
| yeo_johnson | 46 / 8.2 / 1.27 | 46 / 8.4 / 1.69 | 46 / 8.2 / 1.27 | 35 / 6.2 / 1.33 | 73 / 13.1 / 11.2 | 193 / 37.8 / 27.4 |

**What this says.** The order-6 partials of Yeo-Johnson are 1.9 times as fast
on gcc (359 -> 193 ns) and 3.6 times on CUDA (99.5 -> 27.4); Box-Cox's
are 310 -> 268 and 46.5 -> 27.8. Every quantity of Yeo-Johnson except its
inverse is within 6.4 ulpC, and Box-Cox's inverse log-Jacobian is within
6.6. The remaining errors above 8 ulpC are the inverses of Box-Cox and
Yeo-Johnson near lambda = 0 (up to 46.3 and 30.5) and softplus's order-6
forward log-Jacobian (18.4), each a cancellation measured in the formula.

**Chosen.** This version. Test bounds 1.5 times the larger of the gcc and
CUDA maxima for the changed quantities.

**Open.** The inverses of Box-Cox and Yeo-Johnson near lambda = 0 and
softplus's forward log-Jacobian (above); a cheaper log1prel quadrature on
the device (one division per node) is a trade of speed against accuracy.

## Stirling's error and its derivatives to order 6

**What.** `include/laplace/math/stirling.hpp`: Stirling's error
delta(a) = log Gamma(a) - (a - 1/2) log a + a - log sqrt(2 pi), a > 0
(`stirling_error`), and its derivatives delta^(k)(a), k = 1..6
(`stirling_error_d1` ... `stirling_error_d6`). delta(a) ~ 1/(12 a) as
a -> inf and ~ -log(a)/2 as a -> 0. Needed by the gamma distribution: its
log-density written as -a D(y/mu) + log(a)/2 - log sqrt(2 pi) - delta(a) - log y,
with a = 1/phi the shape and D(t) = t - 1 - log t, has no cancellation at
large a, and its derivatives in the shape need delta^(k) to order 6
(note/distrib.md). The existing `log_gamma_star` (the value only, a >= 10 by
the series, inside the incomplete gamma) is the same function.

**How.** Four candidates, by region of a:
- direct: the definition, lgamma(a) - ... and, for k >= 1,
  psi^(k-1)(a) - d^k[(a - 1/2) log a - a] in closed form. The result is about
  1/(12 a^2) times its largest term, so the error grows like a^2: 256 ulp at
  a in [2, 4), 1e3 near 10, 1e30 at 1e15 (order 0). Good below 1 (at most
  15 ulp), where the pole of psi^(k-1) dominates;
- asymptotic: the series sum_j B_2j / (2j (2j-1) a^(2j-1)), j <= 20,
  differentiated term by term, stopped when a term falls below 2^-54 of the
  sum. Within 8 ulp at every order for a >= 10; at order 6 it loses 84 ulp
  on [8, 10) and fails below 6;
- recurrence: delta(a) = delta(a + n) + sum_(j < n) g(a + j), with n the
  number of unit steps that brings a + n to [10, 11), and
  g(b) = delta(b) - delta(b + 1) = (b + 1/2) log(1 + 1/b) - 1
  = atanh(u)/u - 1 = sum_(m >= 1) u^(2m)/(2m + 1), u = 1/(2b + 1).
  Differentiated term by term, g^(k)(b) = (-2)^k sum_m (2m)(2m+1)...(2m+k-1)
  u^(2m+k)/(2m + 1): every term of every sum has the sign (-1)^k, so nothing
  cancels. Within 10 ulp on [1, 10) at every order; below 1/4 the series in
  u converges too slowly for its cap of 80 terms;
- Binet's first formula, delta^(k)(a) = (-1)^k int_0^inf t^(k-1) b(t) e^(-a t) dt
  with b(t) = 1/(e^t - 1) - 1/t + 1/2 > 0 (Taylor series below t = 2), by the
  exp-sinh rule: 3-7 ulp almost everywhere (spikes of 70 ulp at order 3 near
  a = 1/2 and 1e4 ulp at order 6 near 1/2), but 7-11 microseconds per call on
  one gcc thread. Discarded.

Chosen routes: direct below 1, recurrence on [1, 10), asymptotic from 10.

**Measurement.** References `stirling_error*.csv`
(`tools/reference/stirling.py`): the definition with mpmath's loggamma and
polygamma, the elementary part in closed form, at 400 and 500 bits (the
difference cancels by about a^2, 2^100 at 1e15), kept if the two agree to
2^-100. 1384 rows per order: a log-uniform in [2^-40, 1e15], uniform in
[1/4, 40], around 1/2, 8 and 12. Maximum ulp of the chosen routes, gcc/OpenMP:

| a | k = 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| (0, 1) direct | 13.2 | 12.7 | 10.8 | 13.4 | 11.7 | 14.8 | 14.2 |
| [1, 10) recurrence | 3.44 | 3.92 | 3.81 | 6.07 | 6.40 | 9.31 | 9.72 |
| [10, 1e15] asymptotic | 2.51 | 3.62 | 4.20 | 5.27 | 4.76 | 6.69 | 8.19 |

ns per call on one gcc thread (`benchmarks/math_stirling.cpp`, measured while
another build occupied the machine): direct 5-37; recurrence 59-307
(the cost grows with the number of steps, n up to 9, and with the order);
asymptotic 22-44.

**What this says.** Every order is within 15 ulp everywhere; the largest
errors are those of lgamma and the polygamma functions below 1. The recurrence
is the slow route, up to 0.3 microseconds at order 6 near a = 1; if the gamma
distribution at small shapes needs it faster, start the series lower for the
low orders or shift by fewer steps.

**Chosen.** This version. Tested in `tests/math/test_stirling.cpp` (gcc
build) with bounds of half again the measured maxima, exact values at a = 1
(delta(1) = 1 - log sqrt(2 pi), delta'(1) = 1/2 - gamma_E), continuity across
a = 1 and 10, NaN for a <= 0 and 0 at infinity. On the CUDA build (RTX 4070 SUPER)
the maxima are 12.0, 13.4, 10.8, 13.4, 11.7, 14.8, 16.1 ulp (orders 0-6), all
below a = 1, within the test bounds.

**Stopping tolerance of the recurrence.** The summation in m terminates when a term is at most 2^-54 of the partial sum (iteration cap 80).

## Mixed partial derivatives of log P(a, x) and log Q(a, x), total orders 1 to 6

2026-11-12. `include/laplace/math/incomplete_gamma_mixed.hpp`: for each total
order K = 1..6, `log_gamma_p_mixed_dK(a, x)` and `log_gamma_q_mixed_dK(a, x)`
return a `gamma_inc_mixed_partials<K>` whose v[j] = d^K log S / da^(K-j) dx^j,
j = 0..K (j derivatives in x; the packing of `laplace::distrib::sym_tensor`).
The gamma distribution in (mu, phi) needs them for log F = log P(s, s y/mu),
and the Poisson cdf F(k) = Q(k + 1, mu) needs the pure x components (j = K) at
integer a.

**What cancels.** The obvious formulas use the hazard H = f/S, with f the
gamma density: d log P/dx = H, d^2 log P/da dx = H (log x - psi(a) - d log P/da),
d^2 log P/dx^2 = H ((a - 1)/x - 1 - H). Formed from the library's outputs
(candidate `naive` of the bench), they lose, on the 2971 points below, up to
6.1e4 ulp1 at order 1 (H = exp(log f - log P) in the far tails), and at order 2
up to 1.2e5 and 7.3e8 for P and 1.7e3 and 3.5e4 for Q.

**How.** Every route of the derivatives in a (incomplete_gamma_deriv.hpp)
carries log S(a + h) as a Taylor polynomial in h. Here the routes carry
log S(a + h, x (1 + e)) as a polynomial in h and in the relative increment e of
x, of total degree K, and d^(i+j) log S / da^i dx^j = i! j! B_ij / x^j. In the
variable d = log(t/x) both sides are integrals whose x-dependence is
exp-linear, like their a-dependence: with y = x - t, raising x to x (1 + e)
multiplies the integrand by exp(e y) and the prefactor x^a e^-x by
exp(a log(1 + e) - x e). So for j >= 1, B_ij is a joint cumulant of (d, y)
divided by i! j!, plus a (-1)^(j-1)/j when i = 0, (-1)^(j-1)/j when i = 1, and
nothing when i >= 2. B_01 = x d log S/dx is taken in closed form (a/I_L for P,
-1/I_U for Q). The point mass of the lower side (from its integration by
parts) does not move with x, while its density part carries the factor
x (1 + e): log I_L(h, e) = log I_L + K + log(1 + e (pi + (1 - pi)(1 - exp(-K)))),
with K the joint cumulant generating function and pi the share of the density
(checked with SymPy in `tools/reference/incomplete_mixed_forms.py`). The
bivariate logarithm and exponential follow from the Euler operator
h d/dh + e d/de; they are the two-variable versions of cumulants_from_moments
and moments_from_cumulants. Routes:
- quadrature: the same rules and nodes, stored; the joint central moments of
  (d, y) to total order K are summed in a second pass about the weighted
  means. Two repairs found by the measurement: the share of the density is
  summed from the nodes (computed as (W - 1)/W it was 0 at x = 3e-24), and the
  lower rule runs without the point mass in its running weight (with it, the
  rule stopped at the first node past t = 0 when the density is 1e-24 of the
  point mass, and the result was off by a factor 61 at x = 3e-24);
- lower series: term n gains the factor (1 + e)^n, coefficients C(n, m); the
  series runs until the tail of every sum of C(n, m) tau_n is below 1e-22 of
  it;
- upper asymptotic series: term n gains (1 + e)^-n, coefficients
  (-1)^m C(n + m - 1, m);
- closed form at small x and Temme's expansion: each of their series in
  (h, e); in Temme's, lambda = x/a becomes lambda (1 + e)/(1 + h/a);
- complement log(1 - S): from the series of log S as in complement_dk, except
  for the complement of the lower side (log Q where P is small, which includes
  the Poisson cdf below its mean). There exp of the series of log P cancels in
  e: its coefficients of e^m add up to C(a, m) and less, to 0 at a = 1 for
  m >= 2. So S(h, e)/S is formed through the density,
  S(a + h, x (1 + e)) = S(a + h, x) + x int_0^e f(a + h, x (1 + u)) du, with
  f(a + h, x (1 + u))/f(a, x) = (1 + h/a) x^h Gamma(a + 1)/Gamma(a + 1 + h)
  (1 + u)^(a - 1 + h) e^(-x u), and the binomial series of (1 + u)^(a - 1 + h)
  as products of linear factors. The same density ratio replaces the series in
  e of the closed form at small x. Measured at order 6 on the complement of
  the lower series: without it 4.1e4, 2.4e3 and 2.6e3 ulp1 at j = 1, 5, 6;
  with it 1.05e5, 38.9 and 43.9 (the column j = 1 is the small-a problem
  below either way);
- log Q at a < 2, 1/4 <= x < 3/2: the component j = 1 comes from the
  quadrature of the upper side (order 6, a >= 1: 44 ulp1 against 940
  without). Its other components are worse there (1.3e4 at j = 6), so only
  j = 1 is taken from it.

The fraction in Taylor form has no bivariate version; on the host its region
takes the quadrature.

**References.** `tools/reference/incomplete_mixed.py` (mpmath, 240 and 320
bits, again at 480 and 640 bits where these disagree), from the definition and
independent of the transcription: the moments in a of
J = int exp(a v - e^v) dv over the side (the quadrature of
incomplete_deriv.py), and the increment of x as the integral of the density
over [x, x (1 + e)], whose bivariate series in (h, e) is known in closed form.
A row is kept if the two precisions agree to 2^-100 of max(|f|, 1) (in the far
lower tail the cumulants with i >= 2 are 1e-14 next to terms of order
(a/x)^j). 2971 points: the 1500 of incomplete_deriv.py, its 400 boundary
points, 600 for the derivatives in x (x down to 1e-30 at a < 2, a near 1, x
near a, the far tails, x near 3/2 and 36, Temme's boundaries, tiny a) and 600
at integer a from 1 to 1000 with x in [0.05, 1000] (x near a, x down to a/1e4,
x far above a). 39 points were dropped (the quadrature at 480 bits did not
converge for x/a below 1e-6, or x near 1e-29). Files
`log_gamma_{p,q}_mixed_d{K}.csv`, columns a, x, j. The pure a components agree
with the files of incomplete_deriv.py to 1.1e-16 relative.

**Measurement** (`benchmarks/math_gamma_mixed.cpp`, gcc/OpenMP): maximum over
the points of min(ulpL, ulp1), by component j = 0..K.

| | K | j = 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| P, all a | 1 | 5.37 | 5.36 | | | | | |
| | 2 | 5.87 | 4.77 | 8.03 | | | | |
| | 3 | 8.27 | 7.18 | 6.94 | 12.9 | | | |
| | 4 | 33.4 | 18.7 | 19.7 | 12.4 | 23.5 | | |
| | 5 | 176 | 60.6 | 35 | 55.9 | 60.2 | 41.4 | |
| | 6 | 651 | 361 | 209 | 92 | 176 | 217 | 107 |
| Q, a >= 1 | 1 | 5.18 | 15.1 | | | | | |
| | 2 | 6.53 | 15.9 | 19.5 | | | | |
| | 3 | 16.9 | 23.6 | 38.3 | 46.3 | | | |
| | 4 | 31.2 | 16.1 | 28.1 | 31.9 | 31.7 | | |
| | 5 | 45.7 | 37.6 | 63.4 | 41.9 | 58.7 | 19.8 | |
| | 6 | 97.2 | 44 | 235 | 60.3 | 67.6 | 38.9 | 86.2 |
| Q, 0.1 <= a < 1 | 4 | 21.2 | 1.07e3 | 151 | 68.6 | 34.8 | | |
| | 5 | 16.9 | 463 | 479 | 198 | 74.8 | 67.9 | |
| | 6 | 17.7 | 1.05e5 | 2.47e3 | 651 | 194 | 98.9 | 111 |
| Q, a < 0.1 | 4 | 0.53 | 8.15e3 | 241 | 142 | 74.8 | | |
| | 5 | 0.55 | 338 | 2.6e4 | 333 | 182 | 94.5 | |
| | 6 | 0.68 | 1.6e4 | 3.71e3 | 1.04e3 | 431 | 129 | 66.6 |

At orders 1 to 3, Q at a < 1 is within 31, 40.4 and 113. The row "Q, a >= 1"
excludes a = 1 itself, which falls in the integer group: there order 6, j = 1
reaches 741 at x = 0.2 (below the quadrature's x = 1/4). For the Poisson cdf
(522 points at integer a), the pure x components j = K reach, at orders 1 to
6: Q 3.0, 6.0, 9.0, 12.8, 10.6, 15; P 3.3, 1.5, 3.3, 7.6, 25.4, 16.3. By route
(order 6, the largest component): Temme 38 (P) and 14.5 (Q), its complement
1.2; the upper series 1.9, its complement 3e-4; the upper quadrature 86 (Q),
its complement 217 (P); the lower series 651 (P, j = 0 at a < 0.1) and its
complement 1.05e5 (Q, small a); the closed form 1.6e4 (Q, small a). The route
of the device (the series limits of the CUDA build, run on the host, with the
lower quadrature for a >= 2) has the same maxima within 10 per cent.

ns per call on the timing set of math_gamma_deriv (a in [1/8, 1024],
x = a e^(z/2)), one gcc thread / OpenMP with 8 threads, measured while a
CUDA test occupied the GPU: P orders 1-6: 1197/160, 1400/182, 1581/212,
1867/264, 2246/313, 2788/391; Q: 1340/176, 2423/313, 2679/356, 3135/424,
3692/517, 4372/621 (the derivatives in a alone take 0.7 to 1.5 us). From
order 2, Q pays the extra quadrature of its j = 1 component at a < 2; the
quadrature alone costs 3.4 to 4.8 us per call.

**What this says.** log P is within the errors of the derivatives in a of the
same order everywhere, and so is log Q for a >= 1, which covers the Poisson
cdf and the gamma distribution with shape at least 1. The target is not
reached for log Q at a < 1 and x below about 3/2, in the components with one
to three derivatives in a and at least one in x, at orders 4 to 6: up to
1.05e5 ulp1 at order 6, that is 2e-11 absolute. There log Q = log a + log R,
and each route forms these components from terms that cancel by powers of
1/a: the complement of P from the series of P in h, whose terms carry the pole
of log a; the closed form from the cumulants of its factor F, which is a
difference of terms of order one; the quadrature of the upper side is accurate
there only for j = 1 and x >= 1/4. Not tried: the upper side's quadrature
carried down to x -> 0 (its tanh-sinh panels at x < 1/4 lose 1e5 to 1e7 on
these components as they are), or a series of R in which the derivatives in x
are taken analytically.

**Chosen.** As above. Tested in `tests/math/test_gamma_mixed.cpp` (gcc build;
the CUDA build not yet run) with bounds of half again the measured maxima over
all components: P 8, 12, 20, 50, 265, 980; Q 47, 61, 170, 1.23e4, 3.9e4,
1.58e5; and the limits at x = 0 and x = infinity.

## Generic routes of the distributions: quantile, discrete cdf, expected information, moments, rng

**What.** `include/laplace/distrib/generic_*.hpp` (note/distrib.md sections
4 and 5 C): algorithms that work on any distribution lacking a member,
selected at compile time with C++20 `requires` by the dispatchers
`distrib_quantile`, `distrib_quantile_from_log`, `distrib_quantile_from_log_upper`,
`distrib_cdf`, `distrib_ccdf`, `distrib_log_cdf`, `distrib_log_ccdf` and
`distrib_expected_info<K>`, which take the distribution's own member when it
has one.
- `generic_expectation.hpp`: `expectation_by_integral(d, h)` = E[h(Y)], by
  `panels_quad` over the support cut at `center()` and at `y_kinks()`, panels of
  width `scale()` (continuous), or by a sum from floor(center()) outwards
  (discrete) that stops, once more than 2 scale() from the centre and with the
  mass ratio r < 1, when (|t_y| + |t_(y-1)|)/(1 - r) falls below 2^-60 of the
  sum of |t| (valid for log-concave masses); `moment_by_integral(d, k, about)`.
- `generic_cdf.hpp`: `cdf_by_sum`, `ccdf_by_sum`, `log_cdf_by_sum`,
  `log_ccdf_by_sum` for discrete distributions: below the centre the lower tail
  is summed downwards from k, from the centre up the upper tail upwards from
  k + 1, the other one as the complement; each sum adds e^(l(i) - l(start)) and
  returns l(start) + log(sum), so that tails below the double range keep their
  logarithm.
- `generic_quantile.hpp`: `quantile_by_root` (continuous) and
  `quantile_by_search` (discrete), each from p, log p and log(1 - p). Both work
  on the tail that holds the target and on the log scale (log F(y) = log p, or
  log(1 - F(y)) = log(1 - p) above 1/2). Root: a bracket from `center()` by
  doubling steps of `scale()` towards an infinite end, by distances
  (e - y) 2^-m, m = 1, 2, 4, 8, ..., towards a finite end e; Newton's step when
  it stays in the bracket, bisection otherwise (at the geometric mean of the
  distances to a finite end when they differ by more than 4); then one Newton
  step on F(y) = p itself. Search: a bracket from floor(center()) by doubling
  steps, then integer bisection.
- `generic_info.hpp`: `expected_info_by_integral<K, form>(d)`, K = 0..4 (total
  order 6), in two forms. With C the sorted derivative indices and B_T the
  complete Bell polynomial of l over T (computed by the recursion over the
  block that holds the lowest element, at most 2^4 subsets):
  hessian form d_C I_ab = -E[sum_(T in C) l_(ab + C-T) B_T] (reads l to order
  K + 2); score form d_C I_ab = E[sum_(T in C) S_(C-T) B_T], S_U = sum_(V in U)
  l_(a+V) l_(b+U-V) (reads l to order K + 1). A component is NaN when the form
  cannot give it: for a parameter j of finite regularity r_j, hessian form when
  n_a + n_b + n_C > r_j + 1, score form when n_C > 0 and max(n_a, n_b) + n_C >
  r_j + 1 (n the multiplicities of j); beyond that the a.e. derivatives miss a
  point mass or the boundary term of an integrand that jumps at the moving kink.
  On the Laplace this leaves NaN every derivative with two indices mu in C, or
  with mu in C and in (a, b); statmodels7 had left the same components wrong
  (derivate.md, 2026-10-05).
  Amended 2026-11-20: a kink can move with several parameters at once (the
  knots y = mu +- gamma lambda of mcp, mu +- lambda and mu +- a lambda of scad,
  C^1 in y), and then the counts of j alone admit components whose integrand
  jumps after one derivative in lambda and one in the shape. The generic second
  derivative of I_lambda_lambda in gamma came out 2.32 times the closed one on
  mcp with the Student tail at (0.3, 1.3, 3, 0.9, 3.5). Every parameter that
  moves a kink of order k has r_j <= k, so the rule now sums the counts over
  M_r = {j : r_j <= r} for each finite r among the parameters and asks the same
  inequalities with r + 1. With one parameter of finite regularity (laplace1,
  laplace2, enet, powerexp) the rule is the old one, checked component by
  component to order four; with several it can return NaN where a value exists
  and never a wrong value. Measured at order two, score form, on mcp and scad
  with the Student tail: 170 components NaN by the rule, the other 280 finite
  ones equal to five-point differences of the order-one route within 2.4e-11
  relative.
  The rule is conservative when several parameters move a kink. Against the
  closed forms at the settings of test_mcp_scad_info.cpp (order two, score
  form): 27 components that the one-parameter rule had admitted with a wrong
  value are now NaN (13 on mcp, up to 2.32 of the tensor's scale; 14 on scad,
  up to 1.52), and 28 components with C = (mu, lambda) or (mu, shape) are
  refused although they are exactly 0, the information of a location law over
  the whole line not depending on mu (the one-parameter rule had them within
  2.3e-15). Decided with Giovanni: NaN is kept there, with no descriptor flag
  for a pure location parameter.
- `generic_rng.hpp`: `rng_by_inversion(d, gen)` = `distrib_quantile(d, u)` with
  u = (n + 1/2) 2^-53 from the top 53 bits of `gen.urand64()`, in
  [2^-54, 1 - 2^-54]; gen is a Kokkos pool state (XorShift64, XorShift1024) or
  `counter_uniform_source{seed, index}`, SplitMix64 (Steele, Lea and Flood,
  2014) at the counter seed + (index 2^32 + i) gamma, whose draws depend only on
  (seed, index, i).

**Measurement.** `tests/distrib/test_generic.cpp` (the twin test: the four
pilot distributions behind `tests/distrib/generic_hidden.hpp`, which hides
their quantile, expected information and, for the Poisson, the cdf family) and
`benchmarks/distrib_generic.cpp`. Errors against the hidden closed forms.

Quantile by root, 13 values of p from 1e-300 to 1 - 1e-15 and the three
inputs, on gaussian1(0.4, 1.3), gaussian1(-20, 1e-3), laplace1(0.4, 1.3),
gamma1(2, 0.3), gamma1(1.5, 4), gamma1(1.5, 0.02): worst 31.9 (gcc) in units
of the change of y caused by one ulp of y or of p, at p = 1e-300 on
gamma1(2, 0.3) (y = 1.2e-90); without the last Newton step on F(y) = p it was
67, the log cdf of a far tail having an absolute error of a few ulps of
|log F|. Quantile by search on the Poisson (mu = 0.05 to 1e4, p from 1e-300 to
1 - 1e-15): the smallest k with F(k) >= p in every case, from each input.

Discrete cdf by summing the Poisson mass, k = mu + z sqrt(mu), z in
[-30, 100], worst error of log F and log(1 - F) in ulps of max(|log|, 1):

| mu | 0.05 | 3.5 | 40 | 1000 |
|---|---|---|---|---|
| logpmf y log mu - mu - log y! | 0.03 | 4.4 | 17.0 | 1243 |
| logpmf -mu D(y/mu) - log sqrt(2 pi y) - delta(y) | 0.60 | 2.64 | 0.82 | 5.91 |

The first row was measured with the pilot's first logpmf, whose terms near
7000 cancel to about -4 at mu = 1000: an absolute error of 1e-12, a relative
error of every mass that no sum can remove. The log-density was then written
logpmf in the saddle-point form (D(z) = z log z - z + 1, delta Stirling's
error), which has no cancellation: second row, the same on gcc and on the
host side of the CUDA build.

Expected information and its derivatives (K = 0..4), worst error in eps of the
largest closed-form component of each order, gcc:

| distribution | form | K = 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| gaussian1(0.4, 1.3) | hessian | 0.3 | 1.1 | 1.9 | 0.7 | 3.9 |
| | score | 0.8 | 1.6 | 1.4 | 0.9 | 3.2 |
| gamma1(2, 0.02) | hessian | 0.0 | 1.6 | 0.9 | 1.1 | 0.6 |
| | score | 1.6 | 1.0 | 0.9 | 1.7 | 1.2 |
| gamma1(2, 0.3) | hessian | 0.7 | 2.5 | 4.0 | 0.8 | 2.3 |
| | score | 1.3 | 0.8 | 1.3 | 3.2 | 3.1 |
| gamma1(2, 4) | hessian | 0.5 | 2.5 | 0.7 | 0.2 | 2.7 |
| | score | 1.0 | 1.0 | 1.3 | 1.3 | 0.7 |
| poisson(0.05) | hessian | 1.6 | 0.6 | 1.0 | 1.6 | 0.0 |
| | score | 0.0 | 1.3 | 0.5 | 1.1 | 0.0 |
| poisson(3.5) | hessian | 0.0 | 2.3 | 3.3 | 4.7 | 0.0 |
| | score | 1.8 | 1.5 | 3.3 | 4.7 | 10.9 |
| poisson(40) | hessian | 1.2 | 3.9 | 19.5 | 54.5 | 247 |
| | score | 0.6 | 4.7 | 13.7 | 116 | 72.2 |

(Poisson with the saddle-point logpmf; with the first one poisson(40) read
427 and 1415 at K = 4.) The same numbers on the host side of the MSVC build.
At mu = 40, d3 I = -6/mu^4 and d4 I = 24/mu^5 are 1e-5 and 2e-7 of the
terms summed, which is where the 50-250 eps come from.

Laplace(0.4, 1.3): the score form gives I and the derivatives that are not NaN
within 64 eps (K = 0, 1), 256 (K = 2), 1024 (K = 4, the bound tested); the
hessian form gives I_musigma and I_sigmasigma, NaN for I_mumu (0 where the
true value is 1/sigma^2).

Moments (mean, variance, third and fourth central moments) against their
closed forms, in eps of the scale: gaussian1 and laplace1 1.8, gamma1 3.0,
Poisson 0.6 / 3.1 / 2.4 / 6.7 at mu = 0.05 / 3.5 / 40 / 1000 (1343 at 1000
with the first logpmf).

Evaluations of l per component (one quadrature or sum): gaussian1 1472,
laplace1 1460, gamma1 816 and 812 (phi = 0.3, 4), Poisson 33 (mu = 3.5) and
111 (mu = 40).

ns per call, one gcc thread / OpenMP 8 threads (`distrib_generic`):

| quantity | member | generic (hessian) | generic (score) |
|---|---|---|---|
| quantile gaussian1 | 29 / 13 | 514 / 80 | |
| quantile laplace1 | 7.7 / 9.5 | 120 / 28 | |
| quantile gamma1(2, 0.3) | 604 / 96 | 3619 / 496 | |
| quantile poisson(3.5), member cdf | | 2272 / 327 | |
| log cdf poisson(3.5), \|z\| < 3 | 199 / 53 | 648 / 96 | |
| log cdf poisson(40) | 193 / 48 | 2044 / 270 | |
| log cdf poisson(1000) | 99 / 31 | 10816 / 1439 | |
| I gaussian1 | 1.6 | 9.0e4 / 1.3e4 | 9.2e4 / 1.2e4 |
| d4 I gaussian1 | 6.6 | 4.0e6 / 5.8e5 | 3.2e6 / 5.0e5 |
| I gamma1(2, 0.3) | 109 | 5.7e5 / 8.5e4 | 3.5e5 / 4.5e4 |
| d4 I gamma1(2, 0.3) | 729 | 6.0e6 / 8.4e5 | 4.5e6 / 6.5e5 |
| I poisson(3.5) | 3.5 | 1788 / 434 | 1847 / 448 |
| d4 I poisson(3.5) | 4.3 | 8686 / 1462 | 1.1e4 / 1705 |
| I laplace1 | 1.6 | | 8.7e4 / 1.2e4 |

The expected information is computed in one pass: every component on the same
nodes of panels_quad (or the same terms of the sum), a level refined until every
component passes its own stopping test, and at each node the Bell polynomials B_T
and the derivatives of l over a + V looked up once per C for all the pairs
(`detail_generic::expectation_vec_by_integral`). With one quadrature per
component the same rows were, one gcc thread, hessian / score: I gaussian1 2.0e5 /
2.0e5, d4 I gaussian1 7.1e6 / 2.2e7, I gamma1 1.4e6 / 8.6e5, d4 I gamma1 4.7e7 /
4.6e7, I poisson 1751 / 1822, d4 I poisson 1.1e4 / 3.2e4: the one pass divides
the cost by 2.2-2.5 for I and by 1.3-10 for d4 I (7 and 10 in the score form).
The errors are those of the table above; one quadrature per component gave
the same except gaussian1 hessian K = 1 0.5, gamma1(2, 0.3) hessian K = 2 2.7,
gamma1(2, 4) score K = 1-3 2.0, 0.8, 0.7 (the stopping level now depends on
all the components).
| variance gamma1(2, 0.3) | | 1.6e5 / 2.2e4 | |
| uniform draw: counter / XorShift64 pool / XorShift1024 pool | | 1.2 / -; 0.25 / 1.31 / 2.09 (OpenMP) | |
| rng gamma1(2, 0.3), counter | | 603 / 81 | |

On the CUDA build (RTX 4070 SUPER; first column one MSVC host thread, second
the GPU, ns per call): quantile gaussian1 479 / 139 (member 22 / 14), laplace1
101 / 22 (9 / 5), gamma1(2, 0.3) 2964 / 2762 (460 / 141), gamma1(2, 4) 1941 /
604 (318 / 102); quantile poisson(3.5) by search 2682 / 563; log cdf by sum
poisson(3.5) 780 / 118, (40) 2578 / 395, (1000) 12761 / 1612 (member 99 / 33,
160 / 47, 91 / 13); I gaussian1 3.1e5 / 2.4e5 hessian, 2.9e5 / 2.2e5 score;
d4 I gaussian1 1.5e7 / 3.8e6 hessian, 2.9e7 / 9.7e6 score; d4 I gamma1 4.9e7 /
4.5e7 hessian, 4.7e7 / 3.6e7 score; d4 I poisson(3.5) 1.6e4 / 4368 hessian,
4.5e4 / 1.0e4 score (one quadrature per component; in one pass: I gaussian1
1.5e5 / 1.1e5 hessian, 1.3e5 / 1.1e5 score; d4 I gaussian1 6.6e6 / 2.7e6, 4.6e6 /
1.7e6; d4 I gamma1 8.0e6 / 6.4e6, 5.6e6 / 4.3e6; d4 I poisson 1.4e4 / 6906,
1.5e4 / 7578); uniform draw 0.03 (counter and XorShift64 pool), 0.13
(XorShift1024 pool); rng gamma1 by inversion 446 / 62. The expected
information was timed on 256 calls, which leave the GPU almost idle: one call
takes 256 times the figure, 60 ms for I of gaussian1 with one quadrature per
component, 28 ms in one pass. The generic test cases
pass on CUDA with the same errors as on the host, and a kernel of 16 threads
gives the host values to 1e-11 for every route (d4 I gamma1, score form: 3.8 s
for the 16 threads on the GPU with one quadrature per component, 0.47 s in one
pass; 0.12 s on OpenMP).

**What this says.** The routes give the closed forms of the pilot to a few eps
wherever the input functions are accurate; where they are not (the Poisson
logpmf at large mu) the error of the input passes through, amplified by the
derivatives of order 3-4 more in the score form than in the hessian form. On
the smooth pilot distributions the two forms are equally accurate except on
poisson(40), where the hessian form is better by 3-50 times at K >= 2. In one
pass the score form is the cheaper at K = 4 on gaussian1 and gamma1 (1.3 times)
and the dearer on poisson (1.3 times); with one quadrature per component the
hessian form was 3 times cheaper at K = 4 on gaussian1 and poisson. Against this stands statmodels7's pig2 (derivate.md), where -E[l_aa]
cancels and E[l_a^2] does not. The expected information by quadrature costs
0.09-6 ms a call on one thread, 10^3-10^6 times the closed forms: each node
evaluates every derivative order of l, and at K = 4 the Bell polynomials and
the products of sums dominate.

**Chosen.** The score form is the default for every distribution (Giovanni, 2026-11-12):
it is the only one valid at a kink, ties the hessian form on the smooth pilot, and
does not cancel where -E[l_aa] does (pig2).

The counter source is the default uniform source (Giovanni, 2026-11-12):
`rng_by_inversion(d, seed, index)` draws from `counter_uniform_source{seed, index}`;
a generator object, the Kokkos pools included, can still be passed.

**Open.** (2, settled) The
uniform source (Giovanni): the counter source is the fastest and the only one
whose draws do not depend on the thread or the backend. (3, done) The
vector-valued rule, every component on the same nodes (above). (4) The
derivatives of I in a parameter with a kink need the jumps of the derivatives
of l at the kink, which no distribution declares yet. (5) On CUDA,
`binomial_coef` in core.hpp was constexpr host-only, and sym_tensor_position
and the generic lookup gave wrong positions at run time on the device (the
expected information came out 0 and a sum ran for minutes); found by this
test and repaired (LAPLACE_HD).

**Stopping rules of the generic routes.** The bracket of a continuous quantile is bounded in at most 64 steps, which allows roots within 1e-300 of a boundary to converge in tens of steps. The iteration terminates when Newton's step or the bracket width falls below 4 eps |y| or when the bracket closes, with an iteration cap of 200 steps. The error of a continuous quantile is measured in units of the change of the quantile caused by one ulp of y or of p: with y the exact quantile (the distribution's closed form), q the computed one, f the density and eps = 2^-52, the measure is |q - y| / max(|y| eps, min(p, 1 - p) eps / f(y)). The summation of a discrete cdf stops when it reaches a support boundary, or when the geometric tail bound t r/(1 - r) (r < 1 the ratio of consecutive terms, t the current term) drops below 2^-60 of the partial sum; the bound holds for distributions with log-concave probability masses, and the summation ends after at most 2^26 terms per direction if convergence is not attained. In the expectations, once the sweep is more than 2 scale() + 2 away from the centre and the mass decreases (r = p(y)/p(y - 1) < 1), the summation stops when the geometric bound on the remaining terms, (|t_y| + |t_(y-1)|)/(1 - r), is at most 2^-60 of the sum of |t|; two successive terms are used so that a zero of h does not trigger a premature stop. The sum ends after at most 2^26 terms per direction.

## Partial derivatives of log P(a, a c) and log Q(a, a c) at fixed ratio, and the Poisson cdf

2026-11-12 (evening). `include/laplace/math/incomplete_gamma_ratio.hpp`: for
each total order K = 1..6, `log_gamma_p_ratio_mixed_dK(a, c)` and
`log_gamma_q_ratio_mixed_dK(a, c)` return a `gamma_inc_mixed_partials<K>` with
v[j] = d^K log S(a, a c) / da^(K-j) dc^j: the derivatives in a are taken at
fixed ratio c = x/a. The gamma distribution gamma1 (mu, phi) has
F(y) = P(a, a c) with a = 1/phi and c = y/mu, so a depends on phi alone and c
on mu alone; its log F and log(1 - F) derivatives
(`detail_gamma_cdf::gamma1_dk`) are now the product of two univariate Faa di
Bruno sums over these partials (`laplace::math::bell_partial`), each Bell
polynomial of 1/phi and of y/mu of one sign. At the same time the Poisson
cdf's derivatives in mu (the components j = K of
`log_gamma_{q,p}_mixed_dK(k + 1, mu)`) were repaired in
incomplete_gamma_mixed.hpp.

**What cancelled (gamma1).** Composed from the partials in (a, x), a
derivative at fixed c is d/da + c d/dx; at a = x = 50 (phi = 0.02, y = mu)
the two terms are large and of opposite sign, and the pure phi derivatives
of order 6 lost 1.7e7 (log F) and 4.1e7 (log(1 - F)) ulp of the order's
largest component against mpmath (`distrib_log_cdf.csv`, dist 3).

**How.** With x = a c and the increments h of a and g (relative) of c, the
point (a + h, c (1 + g)) has x' = x (1 + h/a)(1 + g). The prefactor
A = a log x - x - log Gamma(a) = (1/2) log a - a phi_c - log sqrt(2 pi) - delta(a),
with phi_c = c - 1 - log c and delta Stirling's error (stirling.hpp), is a
series in (h, g) without cancellation. In the variable d = log(t/x) of
incomplete_gamma_mixed.hpp the move multiplies the integrand by
exp(h w + g (1 + h/a) y), y = -x (e^d - 1), w = d + y/a = (1 - c) d - c (e^d - 1 - d):
w is formed at each node without the cancellation of d against y/a, vanishes
at d = 0 and stays small where the measure lies. Routes, the counterparts of
those of incomplete_gamma_mixed.hpp with the same rules and cut-offs:
quadrature (joint cumulants of (w, y), then t = g (1 + h/a); on the lower side
the point mass enters as log(1 + e (1 - (1 - pi) exp(-Kf))) for pi < 1/2 and as
log(1 + e) + log(1 - e/(1 + e) (1 - pi) exp(-Kf)) for pi >= 1/2, whose
log(1 + h/a) cancels the -log(a + h) of log P exactly); the lower series and
the upper asymptotic series, whose factor per term in h, c (a + h)/(a + h + l)
and (a + h - l)/(c (a + h)), has the Taylor coefficients x/(a + l),
c l (-1)^(m+1)/(a + l)^(m+1) and (a - l)/x, (-1/a)^(m-1) l/(a x); Temme's
expansion, where sigma = c - 1 and eta do not move with h and the factor
sqrt((a + h)/a) of the erfcx term is divided out of G; the complement from exp
of the series of log S. Below a = 2 the series in (h, e) of
incomplete_gamma_mixed.hpp is composed with e = h/a + g + h g/a (the fallback,
which keeps its closed form at small x and its density ratio).

Temme's region, by measurement: Temme's series in (h, g) lost, for
20 <= a < 300, up to 1e5-3e6 ulp of their value on the components with three
or more derivatives in c (the quadrature and the series 1e3-4e4), and for
a < 50 on every component; with no derivative in c (a >= 50) or at most two
(a >= 100) it was the better (order 6, 100 <= a < 300: j = 0 12 against 284
ulp, j = 2 1.7e3 against 3.2e4). So Temme is used whole for a >= 300, for
50 <= a < 300 only for those components, and not below.

**References.** `tools/reference/incomplete_ratio.py` (mpmath): the series in
(h, e) of incomplete_mixed.mixed_row (from the definition) composed with
e = h/a + g + h g/a at 240 and 320 bits (480 and 640 where they disagree), rows
kept at 2^-90 of the order's largest component; 1162 points (a from 1/8 to
3.3e4, c from 1e-3 to 64: near c = 1, log-uniform, at the route boundaries, at
large a). Independent check: on 12 points against mpmath.diff of
log gammainc(a, a c) at 300 bits, max difference 5.1e-77 of the order's scale.
The same script writes `log_gamma_{p,q}_mixed_poisson_d{K}.csv`: integer a
from 10 to 1000 and x within 4 sqrt(a) of a, x = a - 1 and x = a (213 points).

**Figures.** "S": the error in ulp of the order's largest component at that
point in the logarithmic variables, max_j |a^(K-j) c^j v_j| / (a^(K-j) c^j)
(up to sign the derivatives in (log phi, log mu) of the gamma distribution's
log F); "O": the error in ulp of the component itself. Bench
`benchmarks/math_gamma_ratio.cpp`, gcc/OpenMP; max S over all components of
the order:

| order | P own | P fallback | Q own | Q fallback |
|---|---|---|---|---|
| 1 | 1.5e3 | 1.5e3 | 1.7e3 | 1.1e3 |
| 2 | 2.1e3 | 2.1e3 | 1.4e3 | 817 |
| 3 | 2.1e3 | 2.8e3 | 1.2e3 | 1.2e3 |
| 4 | 1.7e3 | 1.0e4 | 1.2e3 | 3.5e3 |
| 5 | 9.0e3 | 4.5e4 | 5.8e4 | 3.3e4 |
| 6 | 7.3e4 | 1.4e6 | 6.2e4 | 5.1e4 |

The fallback is the old composition over (a, x). By component at order 6,
own / fallback, max S for j = 0..6: P 108 / 1.4e6, 134 / 6.7e5, 141 / 1.0e5,
156 / 8.0e3, 220 / 4.6e3, 3.7e3 / 2.9e3, 7.3e4 / 6.1e4; Q 1.4e3 / 5.1e4,
862 / 1.9e4, 522 / 9.4e3, 1.2e3 / 7.0e3, 631 / 6.1e3, 1.8e3 / 3.8e3,
6.2e4 / 3.8e4. The maxima at orders 1-4 are complements log(1 - S) with S near
1e-250, where exp(log S) carries |log S| eps relative (ulpL 2-3); those at
orders 5-6 are the pure derivatives in c at a >= 300 near c = 1, from Temme
and from every other route alike (route quad at order 6: 7.3e4 at j = 6), the
same loss as the pure derivatives in x of incomplete_gamma_mixed.hpp, which
its figure min(ulpL, ulp1) does not show (the components are far below 1).
By route at order 6 (max S over j): the series 0.001-64, the quadrature
2.6e3 (upper side, 2 <= a < 20, j = 6), the complements 1.2e3-1.4e3, Temme
(a >= 300) 7.3e4 at j = 6, the fallback (a < 2) 72 (P) and 260 (Q). The device
limits (devroute) give the same figures within 1 per cent except P order 5
(1.2e3 against 678 at j = 4).

gamma1 (`distrib_log_cdf.csv`, dist 3, phi = 0.02, 0.3, 4; error in ulp of the
order's largest component, as in test_pilot.cpp): log F 1.69e7 before, 379
after; log(1 - F) 4.14e7 before, 356 after (the worst rows: the pure phi
derivatives of order 6, at y = 9 for log F and y = mu for log(1 - F)).

Speed (ns per call, gcc, the timing set of math_gamma_mixed with c = e^(z/2);
one thread / OpenMP 8 threads, no other build running at the end of the run):
P orders 1-6 1711 / 223, 2019 / 313, 2322 / 381, 3341 / 497, 3890 / 595,
4476 / 787; Q 1957 / 281, 3007 / 467, 3316 / 443, 4255 / 678, 5058 / 876,
7024 / 1012; the fallback (one route of incomplete_gamma_mixed.hpp and the
composition) is within 0.7-1.1 times, faster at orders 1-2.

**The Poisson cdf.** The derivatives in mu of log F(k) = log Q(k + 1, mu) lost
up to 3.86e5 ulp (order 6, k = 40 and 41 at mu = 40), and of log(1 - F) up to
4.4e3 (k = 39 at mu = 40, order 5), while the mixed section above reported 15
ulp for integer a. The references are right: the Taylor series of
F(mu + t) = F(mu) - int_0^t pmf(k; mu + s) ds, formed from the finite sum of
the pmf at 300 and 500 bits with no gammainc and no finite differences, agree
with the rows of distrib_log_cdf.csv to the last digit of the double (for example
d^6 log F / dmu^6 = -6.9957584608560983e-7 at k = 40, mu = 40). The library
was wrong, and the mixed section's figure min(ulpL, ulp1) hid it: these
derivatives are 1e-7 to 1e-4, and 3e5 ulp of 7e-7 is 0.2 ulp of 1. The cause
was density_ratio, used for log Q from the lower side: the binomial series of
(1 + u)^(a - 1 + h) times that of e^(-x u) sums terms of alternating sign
that grow like (a x)^(m/2). For a >= 2 and x >= a/2 the factor is now the
exponential of its logarithm ((a - 1) - x + h) u + (a - 1 + h)
sum_(m>=2) (-1)^(m-1) u^m/m (below x = a/2 the products are exact at integer a,
and the exponential lost 4.7e4 at k = 2, mu = 0.5). On the new Poisson points
the same Temme problem as above appeared, so in incomplete_gamma_mixed.hpp
too Temme now starts at a = 50 and serves, for a < 300, only the components
with at most two derivatives in x. Order 6 on the Poisson points, error in ulp
of the component's largest value over the x of the same a, j = 0..6:

| | log P (j = 0..6) | log Q (j = 0..6) |
|---|---|---|
| before (products, Temme from 20) | 4.8e3, 3.1e3, 1.7e4, 4.4e4, 1.0e5, 3.5e3, 2.0e3 | 3.1e5, 4.8e4, 1.5e4, 8.9e3, 1.3e5, 1.9e6, 3.9e8 |
| now | 1.4e3, 1.1e3, 1.3e3, 2.8e3, 4.1e3, 1.5e3, 1.4e3 | 1.5e3, 588, 548, 397, 384, 345, 339 |

The pilot (dist 2, in ulp of the value): log F 3.86e5 before, 3.84e3 after
(k = 39 at mu = 40, order 6, value 4.8e-7 next to -7.0e-7 at k = 40 and
-2.1e-6 at k = 41, absolute error 2.0e-19); log(1 - F) 4.41e3 before, 3.70e3
after (k = 299 at mu = 300, order 5, value -2.2e-8 next to -1.7e-8 and -1.2e-8,
absolute error 1.2e-20). The maxima of the mixed test (min(ulpL, ulp1)) did
not move (P orders 2 and 3 fell from 8.03 and 12.9 to 7.07 and 11.6).

**What this says.** At fixed ratio the large terms in a and c are never
formed, and gamma1's derivatives fall to the level of the partials they are
built from (a few hundred ulp of the order's scale). What remains is the
loss on the high derivatives in c (and in x) at large a near the mean, on
every route, 1e3-7e4 ulp of the value at orders 5-6.

**Open.** (1) The pure derivatives of order 5-6 in c and in x at a >= 300
near c = 1: Temme's series and the cumulants of the quadrature both lose
there; the cause is not established. (2) The complements with S near 1e-250
carry the |log S| eps of exp(log S), as everywhere in the library. (3) The
names: `log_gamma_p_ratio_mixed_dK` says "at fixed ratio"; an alternative is
`log_gamma_p_shape_ratio_dK`.

**Chosen.** The routes at fixed ratio for gamma1 (no choice involved: the old
composition is the fallback below a = 2 and the candidate `fallback` of the
bench). For the Poisson cdf, the exponential form of the density ratio for
a >= 2, x >= a/2 and Temme from a = 50, by the measurements above; to confirm
on the CUDA build.

**Earlier measurements at fixed ratio.** On the points at integer a (from 10 to 1000, x within 4 sqrt(a) of a, 213 points; error in ulp of the component's largest value over the x of the same a), the order 6 error is at most 1.4e3 ulp for log P and 1.5e3 ulp for log Q; the test bounds for orders 1 to 6 were 7.1, 19, 170, 700, 2310, 6200 for P and 6.0, 21.5, 109, 480, 1080, 2180 for Q. Before the repair, the maximum errors of `log_gamma_p_ratio_mixed_d5`, `log_gamma_p_ratio_mixed_d6`, `log_gamma_q_ratio_mixed_d5` and `log_gamma_q_ratio_mixed_d6` (gcc/OpenMP; CUDA not yet measured), in max_j |a^(n-j) c^j v_j| / (a^(n-j) c^j) with n the order, were 9.0e3, 7.3e4, 5.8e4 and 6.2e4 ulp, with test bounds of 13500, 110000, 87000 and 94000 ulp. They were replaced by the post-repair figures 2050 / 6300, 2638 / 3960, 1701 / 2560 and 1892 / 2840 ulp.

## Tails of the log F derivatives of gaussian1 and laplace1

2026-11-12 (night). `include/laplace/distrib/gaussian1.hpp` and `laplace1.hpp`:
the derivatives in (mu, sigma) to order 6 of log F and log(1 - F) were all
built by `detail_location_scale::mixed_dk`, which writes each component as
sigma^-K sum c_jm z^m L_j, z = (y - mu)/sigma, with L_j the derivatives in z.
Two places cancelled inside these z-polynomials; each now has a route of its
own, and mixed_dk stays everywhere else.

**What cancelled.** gaussian1: in its tail log Phi(t) = -t^2/2 - log(-t) + ...,
and these two parts have no mixed derivatives in (mu, sigma) beyond two in mu
(the quadratic) or with derivatives in both (the logarithm); carried through
the z-polynomials with the rest, their large terms cancel to zero and leave
their rounding error on a small component: 1.57e5 ulp of the component at
order 6 (3 in mu, 3 in sigma) at z = -27. laplace: on the side where the
function is log(1 - e^(-|z|)/2), each derivative is a Laguerre-type polynomial
in z with alternating terms much larger than their sum: 391 ulp of the order's
largest component at z = -4.9, order 6 in sigma.

**How (gaussian1).** For x = -t >= 4, t = z for log F and -z for log(1 - F),
and w = 1/x^2,
log Phi(t) = -t^2/2 - log x - log sqrt(2 pi) - w + S_2(t),
S_2 = log(x M_0(x)) + w, M_0(x) = int_0^inf exp(-x s - s^2/2) ds. The first parts
are differentiated in (mu, sigma) in closed form (-(y - mu)^2/(2 sigma^2),
-log(+-(mu - y)) + log sigma, -sigma^2/(y - mu)^2), and only S_2, of size
w^2, goes through mixed_dk. Its derivatives in t are those of log M_0 (the
cumulants kappa_j of the density proportional to exp(-x s - s^2/2), computed
in math::normal.hpp by a downward recursion of moment ratios) minus their
exponential limit (j-1)!/x^j and minus the derivatives of -w. Every
deviation is formed directly, never as a difference: with rho_j = j (1 + e_j)
the scaled ratios, e_j = -w rho_(j+1)/(1 + w rho_(j+1)) and its part beyond
first order f_j = e_j + (j+1) w = (j+1) w (w rho_(j+1) - e_(j+1))/(1 + w rho_(j+1))
are sums of terms of one sign; the moment deviations eps_n and eps2_n follow
by products of these, and the cumulant deviations by the moment-cumulant
recursion with its exponential case and its part linear in w subtracted
exactly (formulas in the header). A first version split off only -t^2/2: the
errors stayed at 1.57e5 ulp (the logarithm cancels as much); a second split
off -log x too: 9.3e3 ulp (the -w part has no derivatives beyond two in sigma
and cancelled in the component with three).

**Switch (gaussian1).** On a grid x = 1 .. 15 step 0.25 and 16 .. 40 (mpmath,
scratch), largest error in ulp of the component over orders 3 to 6, for
x in [1, 2): z-polynomials 252, tail route 2.3e6; [2, 4): 1.9e3 and 8.7e3;
[4, 6): 1.7e3 and 721; [6, 8): 2.9e4 and 5.4e3; [8, 12): 8.4e3 and 490;
[12, 40]: 8.9e4 and 263. The largest values in each range sit next to zeros
of a component (order 6 with 3 in sigma at x = 3.4, order 5 with 2 in sigma
at x = 5.9, order 6 with 2 in sigma at x = 7.3), where any route loses in ulp
of the component. Below x = 4 the parts are larger than their sum and the
split loses; the switch is x = 4 (z <= -4 for log F, z >= 4 for log(1 - F)).

**How (laplace1).** With p = e^(-|z|)/2, log(1 - e^(-|z|)/2) = -sum_(n>=1) p^n/n,
and p^n = 2^-n exp(c_n/sigma), c_n = n s (y - mu) (s = 1 for log(1 - F) at
z < 0, -1 for log F at z > 0). With a = K - b derivatives in mu and b in sigma,
d^a/dmu^a d^b/dsigma^b exp(c/sigma) = (-n s)^a (-1)^b b! sigma^-(a+b) exp(c/sigma) L_b^(a-1)(n |z|),
L_b^(alpha) the generalized Laguerre polynomial (identity checked with SymPy
for a <= 3, b <= 4), computed by its forward three-term recurrence. Terms are
added until p^n n^K (1 + n |z|)^K is below 2^-60 of its value at n = 1. On a
grid |z| = 0.05 .. 30 (mpmath, scratch; largest error over all orders), in ulp
of the component, z-polynomials 3.7, 3.4, 35, 66, 42 at |z| = 0.05, 0.1, 0.2,
0.3, 0.5 and up to 2.5e4 beyond, series 7.3, 5.5, 14, 7.6, 19 and at most
1.7e3 beyond (both largest at |z| = 4.6, on components much smaller than
their order's largest, next to their zeros); in
ulp of the order's largest component, z-polynomials 3.7, 3.4, 2.7, 2.1, 2.6
and up to 1.2e3 beyond (4.5), series 4.1, 3.7, 4.0, 3.8, 4.7 and at most 44
beyond. The two are within a few ulp of each other below |z| = 0.5; the switch
is |z| = 0.25, where the series starts to win in ulp of the component.

**Measured** (gcc, CPU; `tests/distrib/test_pilot.cpp`, new case on
`distrib_log_cdf_tails.csv`, written by `tools/reference/distrib_log_cdf_tails.py`:
1674 rows, gaussian1 |z| = 1.5 .. 35 on the tail side, laplace1 |z| = 0.05 .. 30
on the series side, mu = 0.4, sigma = 1.3). Largest error per order, in ulp
of the component ("own") and of the order's largest component ("scaled"),
before -> after:

| | order | own | scaled |
|---|---|---|---|
| gaussian1 log F | 4 | 776 -> 10.4 | 1.96 -> 2.30 |
| | 5 | 3.19e3 -> 182 | 3.32 -> 3.70 |
| | 6 | 1.57e5 -> 1.46e3 | 2.08 -> 4.81 |
| gaussian1 log(1 - F) | 4 | 773 -> 11.4 | 1.87 -> 2.44 |
| | 5 | 3.20e3 -> 162 | 2.88 -> 3.88 |
| | 6 | 1.57e5 -> 1.48e3 | 2.64 -> 4.38 |
| laplace1 log F | 3 | 74.6 -> 16.6 | 37.3 -> 16.6 |
| | 5 | 504 -> 53.9 | 126 -> 13.5 |
| | 6 | 986 -> 48.9 | 202 -> 16.7 |
| laplace1 log(1 - F) | 3 | 59.3 -> 8.84 | 29.6 -> 3.42 |
| | 5 | 484 -> 24.2 | 121 -> 6.05 |
| | 6 | 921 -> 35.2 | 190 -> 5.80 |

The errors of gaussian1 in ulp of the order's scale grow by 1 to 3 ulp: the
closed-form parts are added to a component of their own size with one
rounding each. Orders 1 to 3 of gaussian1 were already within 5.6 ulp and did not change by
more than 1 ulp. Per component, order 6 of gaussian1 log F (b derivatives in
sigma, b = 0..6), own ulp before: 47.8, 576, 7.56e3, 1.57e5, 3.85, 6.28, 2.08;
after: 47.8, 51.3, 1.46e3, 606, 3.85, 6.28, 4.81 (b = 0, 1, 4, 5 are taken at
|z| < 4, where nothing changed; 1.46e3 at z = -7 is next to the zero of that
component at z = -7.3, 606 at z = -2.5 next to the zero at -3.4). laplace1 log F
order 6 per component, own ulp before: 15.8, 17.3, 218, 65.1, 986, 139, 202;
after: 15.8, 13.7, 15.5, 11.1, 48.9, 35.2, 18.7. The largest own errors left
in laplace1 log F (13 to 16 ulp at z = 20 and 30) are those of e^(-|z|) under
the rounding of z = (y - mu)/sigma, which no form in z removes. On the older
reference file (`distrib_log_cdf.csv`) the worst errors in ulp of the order's
scale are now 30.9, 29.7 (gaussian1, unchanged: they are at |z| < 4) and 11.2,
9.86 (laplace1; 57.2 and 391 before); in ulp of the component gaussian1 goes
from 2.65e4, 3.57e4 to 524, 237 (the largest now at z = 1, not in a tail).
Test bounds are these maxima plus half.

**Cost** (order 6, one thread, ns per call of the detail routes): gaussian1
on |z| in [4, 40] 2.8-2.9e3 against 3.1-3.3e3 for the z-polynomials (the
downward recursion is shorter at x >= 4 than log_Phi_dK's, which runs once
per order); laplace1 2.5e3 against 2.6e3 at |z| in [0.25, 0.5], 1.0e3 against
2.6e3 at [1, 3], 156 against 2.7e3 at [10, 30].

**Open.** The CUDA build was not run for these routes. The gaussian1 route below x = 4 is unchanged: there the order-6
components with 2 or 3 derivatives in sigma keep up to 1.9e3 ulp of the
component next to their zeros.

**Chosen.** The tail route for gaussian1 from |z| = 4 and the series route for
laplace1 from |z| = 0.25, by the measurements above.

## The pure derivatives of orders 5 and 6 in x and c near the mean at large shapes

2026-11-12 (night). The derivatives d^K log S(a, a c) / dc^K and
d^K log S(a, x) / dx^K, K = 5, 6, S = P or Q (the component j = K of
incomplete_gamma_ratio.hpp and incomplete_gamma_mixed.hpp, the derivatives in
mu of the Poisson and gamma log F), lost 1e3 to 7e4 ulp of the order's scale
for a >= 300 near c = 1 on every route (open point (1) of the section on the
partials at fixed ratio).
Here is the cause, by measurement, and the repair.

**References.** The pure derivatives in x need no quadrature: with
D = x^a e^-x / Gamma(a), S(a, x (1 + e)) = S +- D int_0^e (1 + u)^(a - 1)
e^(-x u) du, and the series in e of log S follows from mpmath's gammainc and
the series of exp((a - 1) log(1 + u) - x u)
(`tools/reference/incomplete_pure_x.py`, mpmath at 300 and 400 bits, x = a c
formed exactly). On the 157 points of the ratio files with
a >= 300 and |c - 1| < 4/sqrt(a) it agrees with log_gamma_{p,q}_ratio_mixed_d{5,6}.csv
to 2.2e-16 relative (the double of the comparison): the references were right.
Its grid: a = 300, 1000, 3000, 1e4, 3e4, 1e5 with c = 1 + z/sqrt(a), z from
-4 to 4 in steps of 1/4, then z to +-11 and |c - 1| up to 0.5.

**Not the cause: the rounding of x = a c.** The routes at fixed ratio form
x = a c in double. Measured against a second reference at x = fl(a c), the
errors are the same within a few per cent, so the rounding is not what is
lost. The condition number of v_K in c, |c v_(K+1) / v_K|, is 100 to 2e3
away from the zeros of v_K and up to 3e5 next to them: one ulp of c moves v_K
by that many ulp of its value, so near a zero only the error in units of
eps (|v_K| + |c v_(K+1)|) is meaningful.

**The causes.** Three, all in Temme's expansion, which serves this region
(the quadrature without Temme loses 2e5-3e5 ulp of the value at order 6 for
a >= 3000: the sixth cumulant of y is a small difference of the central
moments of a measure of width sqrt(a)).
1. The Taylor coefficients L_m of log erfcx at zeta (log_erfcx_taylor in
   incomplete_gamma_deriv.hpp), which enter every derivative in x through
   zeta = |eta| sqrt(a/2), d zeta/dx ~ 1/sqrt(2a). Up to zeta = 1 they came from
   the recursion in R = phi(t)/Phi(t), t = -sqrt(2) zeta, whose polynomials
   in t and R cancel: against mpmath, L_6 lost up to 1.4e4 ulp of its value
   for zeta in [0.5, 1] (5.3e4 next to its zero near zeta = 0.62) and L_5
   up to 2.2e3.
   Now, for zeta <= 1/sqrt(2) (t >= -1), L_m = (-sqrt 2)^m g_m(t) plus the
   terms of zeta^2, with g_m the derivatives of log Phi from normal.hpp's
   piecewise Chebyshev expansions; above, the downward recursion as before.
   Maxima over zeta in [0, 2.5] by order 1-6: 2.3, 3.0, 8.3, 15, 39, 170 ulp
   (the largest from the downward recursion at zeta = 0.75-1.15; the zero of
   L_6 near zeta = 0.62 excluded).
2. log G formed by blog of A0 exp(Lz) + sgn S: exp(Lz) carries the moments
   of the Gaussian part, of size dz^m with dz ~ sqrt(a/2) per derivative, and
   the logarithm takes them back to cumulants that are small differences of
   them near zeta = 0. Now log G = Lz + log(1 + sgn S exp(-Lz) / A0)
   (temme_log_g in incomplete_gamma_mixed.hpp, used by both files): the
   Gaussian part enters as it is and only the correction, of relative size
   1/sqrt(a), goes through exp and log. At a = 1e4, Q, order 6: z = 0 193 ->
   48 ulp of the value, z = 0.25 443 -> 30.
3. The complement. Below c = 1 Temme gave log Q as log(1 - P) from the series
   of log P (and log P from Q above), whose exponential and logarithm cancel
   in the same way: log Q at a = 7827, order 5, lost 880 to 3.7e3 ulp of the
   value for z in [-0.2, -0.1] and 2.2e4-2.9e4 next to a zero of v_5
   (z = -0.05 to -0.02), against 1 to 10 for z >= 0 where Q is direct. Erfcx and the derivatives of log Phi hold
   for negative arguments, so Temme's expansion now gives the side wanted
   directly also beyond c = 1 while zeta >= -1 (temme_zeta_cross); beyond it
   the complement is kept, since -a phi and log G cancel in the value and the
   orders 1-2 as zeta^2 grows.

A fourth loss appeared in the measurement of the repaired Temme: on its
direct side far from c = 1 (P for c < 1, Q for c > 1) Temme's series loses on
the derivatives with three or more in c from |c - 1| = 0.25-0.3 on, whatever
a (likely the truncation of the power series in eta of the coefficients C_k,
differentiated): order 6, P at a = 300, c = 0.51: 1.1e5 ulp of the value
against 1.0 for the series; c = 0.74: 41 against 8.5; Q at a = 300, c = 1.49:
3.3e4 against 52; at a = 1000 P from c = 0.70 and Q from c = 1.32. These
components now come from the route without Temme for |c - 1| >= 0.25
(temme_c_band, gm::temme_band), where that route is within 300 ulp.

With the repairs Temme is better than the routes without it for every
component down to a = 50 within the band: worst error in ulp of the order's
scale on the ratio references, order 6 (P, Q), Temme for every component from
a = 300 (the old dispatch) 4.3e3, 5.8e3; from a = 100 4.0e3, 2.3e3; from
a = 50 2.6e3, 1.9e3 (order 5: 9.0e3, 5.2e3 and 2.1e3, 900); from a = 20 P
lost 8.8e3 at order 6. So for 50 <= a < 300 Temme now serves every component
within |c - 1| < 0.25 (and |x/a - 1| < 0.25 in incomplete_gamma_mixed.hpp),
and outside it those with at most two derivatives (a >= 100) or none (a < 100)
as before.

**Results.** On the grid, error of the pure derivative of order K in c in
ulp of its value (max over z in [-4, 4] in steps of 1/4, the zeros of v_K
included), before / after:

| | a = 300 | 1000 | 3000 | 1e4 | 3e4 | 1e5 |
|---|---|---|---|---|---|---|
| P, K = 4 | 711 / 35 | 322 / 39 | 407 / 9.4 | 312 / 17 | 866 / 15 | 1.7e3 / 22 |
| P, K = 5 | 4.0e3 / 49 | 1.5e3 / 36 | 1.9e3 / 35 | 1.5e3 / 48 | 1.4e3 / 92 | 4.2e3 / 25 |
| P, K = 6 | 1.3e5 / 1.7e3 | 2.4e4 / 850 | 3.1e5 / 297 | 5.1e4 / 269 | 3.2e4 / 280 | 3.0e4 / 277 |
| Q, K = 4 | 3.0e3 / 60 | 405 / 15 | 356 / 8.9 | 327 / 11 | 335 / 17 | 295 / 15 |
| Q, K = 5 | 5.2e3 / 26 | 3.5e3 / 36 | 1.4e3 / 21 | 1.4e3 / 60 | 3.3e4 / 64 | 3.9e3 / 39 |
| Q, K = 6 | 2.0e5 / 257 | 1.1e4 / 146 | 1.7e5 / 351 | 1.8e4 / 5.9e3 | 2.1e4 / 354 | 2.0e4 / 251 |

Next to the zeros of v_K the error in ulp of the value is set by the
conditioning; in units of eps (|v_K| + |c v_(K+1)|) the worst over the grid
is 84 (P, order 6, a = 300; 5.9e3 ulp of the value at a = 1e4, Q, order 6, is 4.5 in these units, at a zero where the condition number is 2.9e5), and in ulp of the largest |v_K| over the z of the same a it is
27.

The references gained 102 points at a = 300, 1000, 3000, 1e4, 3e4, 1e5 with
c = 1 + t/sqrt(a), t from -4 to 4 in steps of 1/2
(log_gamma_{p,q}_ratio_mixed_d{K}.csv, now 1264 points) and 78 points at the
same a with x = a + t sqrt(a), t from -3 to 3 in steps of 1/2
(log_gamma_{p,q}_mixed_poisson_d{K}.csv, now 291 points); incomplete_ratio.py
writes them. Test figures (gcc, ulp of the order's scale), ratio files,
orders 5 and 6: P 9.0e3, 7.3e4 before, 2050, 2638 now; Q 5.8e4, 6.2e4 before,
1701, 1892 now. The worst are now below a = 50 near c = 1 (a = 19.8 and 32.1,
order 6, j = 6, the route without Temme), at a = 13097, c = 1.33 (order 5, P,
j = 5, the complement of the series, outside Temme's band) and at a = 334,
c = 0.074 (order 5, Q, j = 0, a complement with S near 1e-250). Bounds 3080, 3960 (P) and 2560, 2840 (Q).
Poisson files: maxima P 4.73, 9.04, 79.9, 370, 849, 4128 and Q 4.00, 14.2,
62.4, 174, 715, 1425 at orders 1-6, all at a <= 42; bounds kept (P order 5 at
2310 to cover 1536 on the CUDA build at a = 12, x = 12, j = 3).
The pilot (test_pilot.cpp, error in ulp of the order's largest component):
the Poisson log(1 - F) fell from 3.70e3 to 867 (bound 1.3e3); log F stays at
3.84e3 (k = 39 at mu = 40, a = 40, below Temme), gamma1 at 379 and 356.
The mixed partials of incomplete_gamma_mixed.hpp and the derivatives in a of
incomplete_gamma_deriv.hpp keep their maxima (the tables above unchanged to
three digits).

Speed (ns per call, gcc, the timing set of the bench, one thread / OpenMP 8
threads), log_gamma_{p,q}_ratio_mixed_dK, K = 1..6: P 1689 / 227, 1925 / 254,
2324 / 319, 2652 / 367, 3127 / 439, 3761 / 539; Q 1792 / 234, 2823 / 382,
3357 / 462, 3772 / 509, 4430 / 628, 5345 / 737: the same as before or faster
(Temme's expansion now replaces the quadrature for more of the components
between a = 50 and 300).

**What remains.** (1) Below a = 50 near c = 1 the pure derivatives of order
6 come from the quadrature, whose cumulants lose up to 2.6e3 ulp of the
order's scale (a = 19.8); Temme does not help there (its 10 terms in 1/a).
(2) The downward recursion of log_erfcx_taylor for zeta in (0.71, 1.2) keeps
up to 170 ulp on L_6. (3) Next to the zeros of v_K the error in ulp of the
value stays large, as the conditioning allows. (4) The CUDA build is not yet
run on these changes.

**Chosen.** The three repairs and the dispatch above, by the measurements.

## Structured matrices: diagonal, compound symmetry and AR(1)

**What.** `include/laplace/pdmat/` (2026-11-12, the pilot of phase 5,
`note/pdmat_structures.md`): `diagonal_pdmat`, `compound_symmetry_pdmat`,
`ar1_pdmat`, the first structures of the interface decided with Giovanni the
same day (`core.hpp`). A structure maps q unconstrained parameters theta to a
symmetric positive definite p x p matrix M(theta) and supplies M, log det M,
M^-1, the Cholesky factor L (M = L L^T) and the derivatives of each in theta
to order 6, one function per order, along a sorted tuple of parameter indices;
a matrix derivative returns false when it is identically zero for that tuple.
Matrices go to a caller's buffer, the lower triangle packed by rows,
(i, j) at i(i + 1)/2 + j. Each structure fixes the constraint of its natural
parameter: rho = tanh(theta) (AR(1)); rho = a + (1 - a) expit(theta),
a = -1/(p - 1) (compound symmetry, positive definite exactly on
(-1/(p - 1), 1)); theta_i = log sigma_i, M_ii = e^(2 theta_i) (diagonal).

**Why the derivatives are written in theta.** The alternative is to
differentiate in rho and compose with the bijector by Faa di Bruno. Measured
on the AR(1) log-determinant (p - 1) log(1 - rho^2) with rho = tanh(theta),
in double, against mpmath at 200 bits (1 - rho given exact): the composition
loses 1.6e-11 relative at theta = 3 (order 6), 9.1e-10 at theta = 5 and
1.2e-5 at theta = 10 (3.8e-8 already at order 2), because the singularity of
log(1 - rho^2) at rho = 1 cancels against the derivatives of tanh. The direct
form -2 (p - 1) tanh^(K-1)(theta) stays within 2.7e-15 at every point.

**How.**
- AR(1), T = tanh(theta), S = 1 - T^2 and sech(theta) from e = exp(-2|theta|):
  M_ij = T^|i-j|; log det M = -2 (p - 1) log cosh(theta); M^-1 tridiagonal,
  cosh^2(theta) at the two ends of the diagonal, cosh(2 theta) inside,
  -sinh(2 theta)/2 next to it; L_i0 = T^i, L_ij = T^(i-j) sech(theta) for
  j >= 1. The derivatives of T^k sech^h (h = 0, 1) are sums
  sum_n c_n T^(k+K-2n) S^n sech^h with integer coefficients from the
  recursion c'_(n+1) += (k + K - 2n) c_n, c'_n = -(2n + h) c_n (D T = S,
  D S^b = -2b T S^b): every term keeps a factor S, so nothing cancels as
  T -> +-1.
- Compound symmetry, s = expit(theta), q = 1 - s, E = e^theta:
  1 - rho = p q/(p - 1), 1 + (p - 1) rho = p s; log det M =
  (p - 1) log(p/(p - 1)) + log p + (p - 1) log q + log s, first derivative
  q - (p - 1) s, K-th -p expit^(K-1)(theta); M^-1 by Sherman-Morrison in E,
  diagonal [(p^2 - 2p + 2) + (p - 1)^2 E + 1/E]/p^2, off-diagonal
  -[(p - 2) + (p - 1) E - 1/E]/p^2; L from the closed LDL^T of the
  equicorrelation matrix, with 1 + j rho proportional to
  u_j/(1 + E), u_j = (p - 1 - j) + (p - 1)(1 + j) E (no cancellation, both
  coefficients >= 0): L_jj = exp(g_j), L_ij = rho exp(r_j) for i > j >= 1,
  g_j and r_j sums of log(A + B E) whose derivatives are expit^(n-1) at
  theta + log(B/A); exp(g) differentiated by the complete Bell polynomial
  (`bell_partial`), rho exp(r) by Leibniz. expit and its derivatives come from
  the logit bijector, except in L, where all orders come from one exponential
  (the same polynomials; up to 3 ulp worse on the derivatives of M and
  log det, so used only in L, where it cut the time by 1.2 at p = 4, order 0, to 3.9 at p = 16, order 6).
- Diagonal: e^(2 theta_i), 2 sum theta_i, e^(-2 theta_i), e^(theta_i); the
  derivatives are nonzero only along a tuple whose indices are all equal.

**Measurement.** References `pdmat_pilot.csv` (`tools/reference/pdmat_pilot.py`): the matrices built from their definitions in mpmath
at 640 bits (inverse by `mp.inverse`, L by `mp.cholesky`, log det by
`mp.det`), derivatives by central differences with steps 2^-60 and 2^-70,
which agree to 1.6e-35 relative; 16569 rows, AR(1) p in {1, 2, 3, 6} and
theta in {-4, -0.8, 0, 0.35, 1.7, 5, 10}, compound symmetry p in
{1, 2, 3, 5, 9} and theta in {-8, -2, -0.3, 0, 0.6, 2.5, 9}, diagonal p in
{1, 3}. Error of a matrix: ulpN, the largest absolute error over the entries
in ulps of the largest entry of that derivative matrix; of log det: relative
ulp, ulp1 at order 0 (it crosses 0). Values below 1e-20 count as zero (the
difference noise). Maximum over orders 0-6, gcc/OpenMP and CUDA (RTX 4070
SUPER):

| structure | M | M^-1 | L | log det |
|---|---|---|---|---|
| diagonal | 0.52 / 0.52 | 0.89 / 0.66 | 0.59 / 0.59 | 0.45 / 0.45 |
| compound symmetry | 8.01 (K=4) / 6.67 (K=4) | 0.81 / 0.81 | 9.34 (K=5) / 9.77 (K=6) | 8.01 (K=5) / 7.91 (K=0) |
| AR(1) | 13.0 (K=6) / 11.1 (K=6) | 0.52 / 0.58 | 8.22 (K=4) / 8.22 (K=4) | 3.92 (K=6) / 4.83 (K=6) |

The largest, AR(1) d^6 M at p = 2, theta = -0.8, is tanh^(6) near a zero of
it; the compound symmetry log det at order 0 (7.9 ulp1, p = 9, theta = -2)
is the sum of the constant terms. Central differences of order K - 1 against
order K agree within 0.0044 of their tolerance (1e-6 of the largest entries).

ns per call, one call filling the whole packed matrix, theta uniform in
[-3, 3] (`benchmarks/pdmat_pilot.cpp`), one gcc thread / 8 OpenMP threads,
orders 0 and 6:

| structure, p | M | M^-1 | L | log det |
|---|---|---|---|---|
| diagonal, 4 | 29 / 18 | 30 / 18 | 29 / 18 | 2.1 / 0.6 |
| diagonal, 16 | 139 / 100 | 148 / 99 | 138 / 99 | 4.1 / 0.7 |
| compound symmetry, 4 | 22 / 35 | 27 / 21 | 376 / 437 | 91 / 11 |
| compound symmetry, 16 | 119 / 133 | 121 / 121 | 1531 / 2086 | 90 / 11 |
| AR(1), 4 | 44 / 116 | 20 / 20 | 42 / 227 | 47 / 51 |
| AR(1), 16 | 165 / 566 | 95 / 96 | 157 / 1055 | 46 / 48 |

With 8 OpenMP threads the times fall by 4 to 7 times. CUDA timings of the same bench, in the
default space, are 0.1-190 ns per call and erratic between neighbouring
cases (the compiler drops part of the work for some kernels), so they are not
quoted per case; the most expensive, L of compound symmetry at p = 16, order
6, is 187 ns.

**What this says.** Every quantity is within 13 ulpN on both backends,
including AR(1) at theta = 10 (rho = 1 - 4e-9), where a composition through
the bijector would lose 5 digits. The Cholesky factor of compound symmetry is
the expensive part (1.5-2.1 microseconds at p = 16 on one thread), because
each column needs its own logarithms and Bell polynomials; it depends on the
column only, so its cost grows like p, not p^2. Two repairs during the work:
AR(1) M and L computed each entry separately (4.3 microseconds for L at
p = 16, order 6), now one value per diagonal band (1.06).

**Open.** Storage is dense packed even for the diagonal, where a derivative
fills p(p + 1)/2 zeros to set one entry; sparse storage is to be decided with
the psdmat structures (`note/pdmat_structures.md` section 8).

**Chosen.** This version; tested in `tests/pdmat/test_pilot.cpp` on gcc and
CUDA, bounds 1.5 times the larger measured error.

## Univariate distributions: the pilot (gaussian1, laplace1, poisson, gamma1)

**What.** `include/laplace/distrib/`: `core.hpp` (the kind, continuous or
discrete; the descriptor of a parameter, `distrib_param_info`: name, natural
domain, regularity order of log f, whether it moves the support, regularity
order of log F; `distrib_kinks`; the packed symmetric tensor `sym_tensor<P, K>`
over sorted indices in lexicographic order; `expected_info_tensor<P, K>`),
and four distributions as structs whose fields are the natural parameters:
`gaussian1_distrib` (mu, sigma), `laplace1_distrib` (mu, sigma),
`poisson_distrib` (mu), `gamma1_distrib` (mu, phi; shape 1/phi). Each has the
log-density (or log-mass) and its derivatives in the parameters to order 6,
the expected information I = -E[d^2 l] and its derivatives to order 4 (total
order 6), the cdf, its complement and their logarithms, the derivatives of
log F and log(1 - F) to order 6, the quantile (continuous), the support, rough
center and scale, and the non-smooth points. Decisions in note/distrib.md.

**How.**
- gaussian1, laplace: closed forms; log F and log(1 - F) are functions of
  z = (y - mu)/sigma, differentiated in z (log_Phi_dK; for the Laplace
  polynomials in q = 1/(2 e^z - 1) or, in the tails, the series
  -sum p^n/n) and carried to (mu, sigma) by the location-scale rules
  (`detail/location_scale.hpp`; tails: section "Tails of the log F
  derivatives of gaussian1 and laplace1").
- laplace: l is C^0 in mu at mu = y (derivative sign(0) = 0 there, the mean
  of the one-sided ones), log F is C^1 there (`cdf_regularity` 1).
- poisson: the log-mass in saddle-point form
  -mu D(y/mu) - log sqrt(2 pi y) - delta(y), D(z) = z log z - z + 1, delta
  Stirling's error: the textbook y log mu - mu - log y! loses 1e-12 absolute
  at mu = 1000 (the generic cdf by summing the mass went from 1243 to 5.9 ulp
  there). F(k) = Q(k + 1, mu); the derivatives of log F in mu are the
  x-derivatives of log Q(k + 1, x) (`incomplete_gamma_mixed.hpp`).
- gamma1: l = -s D(y/mu) + log(s)/2 - delta(s) - log y - log sqrt(2 pi),
  s = 1/phi, D(z) = z - 1 - log z (series in w = (y - mu)/mu below
  |w| = 1/4, the logarithm directly below z = 1/2, where w rounds to -1 for
  y < mu 2^-53 and the log-density was -inf before the fix). The phi-
  derivatives of -delta(1/phi) by Faa di Bruno over delta^(k): the terms
  cancel by up to 1e10 among themselves, but the result stays within 8 ulp
  of the whole shape term (log(s)/2 differentiated in closed form dominates).
  I_phiphi = s^2/2 + s^4 delta''(s). log F = log P(a, a c), a = 1/phi,
  c = y/mu, composed from the partials at fixed ratio
  (`incomplete_gamma_ratio.hpp`) by two univariate Faa di Bruno sums; the
  composition over (a, x) cancelled up to 4e7 ulp, because along phi the
  ratio x/a is fixed.

**Measurement.** Every derivative order of l, of I and of log F against a
5-point difference of the order below (relative 1e-8, 1e-7 for log F; the
step scales with the parameter and, for log F, with 1/(1 + |z|)); I against
the variance of the score (quadrature with a knot at the Laplace kink, where
-E[l_mumu] = 0; support sum for the Poisson; tanh-sinh plus exp-sinh for the
gamma at shape 1/4) to 1e-12; log F and log(1 - F) against mpmath
(`tools/reference/distrib_log_cdf.py`, mpmath.diff at 300 and 400 bits, 2217
rows; tails `distrib_log_cdf_tails.py`, 1674 rows). Maximum error of log F /
log(1 - F) derivatives, gcc, in ulp of the largest component of each order
(the Poisson has one component; near zeros of a component the per-component
error is larger, see the tails section):

| distribution | log F | log(1 - F) |
|---|---|---|
| gaussian1 | 30.9 | 29.7 |
| laplace1 | 11.2 | 9.86 |
| poisson | 3.84e3 (k = 39, mu = 40, order 6, value 4.8e-7 next to -7e-7, absolute 2e-19) | 867 |
| gamma1 | 379 | 356 |

**Open.** The Poisson and gamma figures are set by the incomplete gamma
partials below Temme's range near x = a (a < 50), see its sections. rng is
not a member (it is a generic route). Derivatives in y and cross derivatives
are not written yet.

**Chosen.** This version. Tested in `tests/distrib/test_pilot.cpp` on gcc and
CUDA.

**Regularity of the Laplace log-density.** The log-density log f(y; mu, sigma) = -log(2 sigma) - |y - mu|/sigma is C^0 in mu at mu = y; its first derivative in mu has a step discontinuity and its pointwise second derivative is zero almost everywhere. The second derivative in mu carries a Dirac point mass at the kink mu = y, so the pointwise expectation -E[d^2/dmu^2 log f] vanishes and does not equal the Fisher information, which is defined by the variance of the score: I_mumu = E[(d/dmu log f)^2] = 1/sigma^2. The distribution function log F(y) integrates the density, so it is one order smoother: it is C^1 at mu = y, and its second derivative in mu jumps by 2/sigma^2 at that point.

## Derivatives of the log-density in y and mixed (y, theta)

**What.** For the continuous distributions of the pilot (gaussian1, laplace1,
gamma1): `logpdf_dy_dk<J, K>(y)`, the derivatives with J derivatives in the
observation y and the symmetric tensor of order K in the parameters,
J + K <= 6 (J = 0 is `logpdf_dk`, J = K = 0 the value); and
`y_regularity`, the regularity of log f in y (laplace: 0, with the kink at
y = mu given by `y_kinks()`; the others infinitely smooth). Each (J, K) is
its own function, instantiated only where it is called, so a model that does
not use derivatives in y pays nothing. They are needed where the argument of
a density is itself a variable: a prior on a coefficient (the coefficient
plays the role of y), a random effect, a latent response, a transformation
of the response with parameters (decided with Giovanni on 2026-11-12,
note/distrib.md). The Poisson, discrete, has none.

**How.** gaussian1 and laplace1 depend on y - mu only, so d/dy = -d/dmu and
the (J, K) component with a derivatives in mu and b in sigma is (-1)^J times
the (a + J, b) component of the parameter derivatives, already measured
(Laplace: sign(0) = 0 at y = mu, as for mu). gamma1: the terms of l with y
are -s y/mu + (s - 1) log y (s = 1/phi), so the components are closed forms
in powers of 1/mu and 1/y times the derivatives of s in phi; for J = 1 and no
derivative in mu the two terms are combined as s^(b) (mu - y)/(mu y) - [b = 0]/y,
whose zero (the mode) is then a single subtraction.

**Measurement.** Every (J, K) with J >= 1 against a 5-point difference in y
of (J - 1, K) and, for K >= 1, in each parameter of (J, K - 1), relative
1e-8 (gamma1 1e-7) of the largest component of the order, at gaussian1 and
laplace1 (0.4, 1.3), y in {-3, 0.2, 1.7, 9}, and gamma1 (2, phi),
phi in {0.02, 0.3, 4}, y in {0.3, 1.4, 2, 5}; and inside a Kokkos kernel.
No mpmath reference: the forms are elementary and each is the exact
derivative of a measured one.

**Chosen.** This version. Tested in `tests/distrib/test_pilot.cpp`.

## Coupled bijectors: alr, log_cholesky, partial_corr

**What.** `include/laplace/bijector/` (2026-11-12, kind C of
`note/bijectors.md`, decisions 17-19): `alr_bijector` (the simplex of K
categories and R^(K-1)), `log_cholesky_bijector` (p x p positive definite
matrices and R^(p(p+1)/2)), `partial_corr_bijector` (p x p correlation
matrices and R^(p(p-1)/2), through the canonical partial correlations
z = tanh u). Each gives forward g and inverse h, the two log-Jacobians in
the reference coordinates (pi_1..pi_(K-1); the lower triangle of Sigma; the
strict lower triangle of R), their derivatives to order 6 along a sorted
tuple of indices, one function per order, returning false when the tuple is
structurally zero, and for the matrices the Cholesky factor L(u) with its
derivatives. Interface as the pdmat structures (`coupled.hpp`): runtime
dimension, caller buffers and a caller scratch buffer, packed storage.

**How.** Closed forms where they exist:
- alr forward: d^k u_a/dpi_t = [t = (a..a)] (-1)^(k-1) (k-1)!/pi_a^k +
  (k-1)!/pi_K^k, the forward log-Jacobian likewise; where the two terms have
  opposite signs, 1/pi_K^k - 1/pi_a^k is written as
  (pi_a - pi_K)/(pi_a pi_K) S_k(1/pi_a, 1/pi_K) (as the logit bijector),
  which took the forward log-Jacobian from 110 to 3 ulp.
- log_cholesky inverse: Sigma_ij = sum_(l<=j) L_il L_jl with L_ii = e^(u_ii),
  L_ij = u_ij, so a tuple is nonzero only on at most two variables of one
  column; its log-Jacobian p log 2 + sum_i (p - i + 1) u_ii is linear.
- partial_corr inverse: L_ab = tanh(u_ab) prod_(l<b) sech(u_al),
  L_aa = prod_(l<a) sech(u_al) (no 1 - z^2 is formed), so every derivative
  is a product of tanh^(k) and sech^(k) = sech P_k(T) (P_k in T^2 below
  T^2 = 1/2, in S = sech^2 beyond, from `tools/reference/bijector_coupled_forms.py`,
  SymPy 1.14.0); log-Jacobian -sum_(a>b) (p - b) log cosh u_ab, separable
  (checked against the symbolic Jacobian for p = 2, 3, 4).
Without a closed form, an explicit recursion over the subsets of the
tuple's positions (Leibniz's rule, solved for the top term):
- alr inverse: l = log pi_K, d_b l = -pi_b, d_b log pi_c = delta_bc + d_b l;
  1 - pi_c is taken as the sum of the other probabilities. Forming it as
  1 + d_b l lost everything near pi_c = 1 (1.3e11 ulp at u = 25, K = 2;
  27 ulp after).
- log_cholesky and partial_corr forward: the derivatives of the Cholesky
  factor from L_jj^2 = X_jj - sum L_jl^2 and L_ij L_jj = X_ij - sum L_il L_jl,
  then log on the diagonal (log_cholesky) or u_ab = asinh(L_ab/Q_ab),
  Q_ab^2 = sum_(l=b+1..a) L_al^2 (partial_corr; equal to atanh z_ab without
  forming 1 - z^2).
SymPy count of the structurally nonzero tuples at order 6 (checked by the
test): log_cholesky p = 4, 10 of 5005 (16 nonzero components of 50050);
partial_corr p = 4, 251 of 462; alr dense.

**Measurement.** References `bijector_coupled.csv` (71660 rows,
`tools/reference/bijector_coupled.py`): exact SymPy derivatives of the
definitions evaluated by mpmath at 300 bits for every inverse and the alr
forward (the partial correlations through the onion construction
L_ab = z_ab sqrt(1 - sum_(l<b) L_al^2), not the sech product of the code);
for the forward Cholesky maps, mixed central differences of the mpmath
definitions at 1200 bits, steps 2^-130 and 2^-140, which agree to 7e-66.
Points: alr K in {2, 3, 5} (three random, one spread to -30..20, one with
pi_K = e^-25); log_cholesky p in {1, 2, 3, 4} (three random, one with
L_ii = e^(+-4)); partial_corr p in {2, 3, 4} (three random, one with
|u| = 4, 2.5, 1.5 for p = 2, 3, 4). Tuples: all of orders 1-2, all pure and
24 random mixed ones of orders 3-6. Error: ulpN, the largest error over the
outputs in ulps of the largest reference output, with the mixed tuples
measured against the geometric mean of their pure derivatives (the manual's
ulpN of mixed partials, weights m_v/K); log-Jacobians relative. Maximum over
orders 0-6, gcc/OpenMP and CUDA (RTX 4070 SUPER):

| map | inverse | inverse log-Jacobian | L(u) | forward | forward log-Jacobian |
|---|---|---|---|---|---|
| alr | 90.5 / 55.2 | 90.5 / 93.9 | - | 3.8 / 3.8 | 3.4 / 3.4 |
| log_cholesky | 1.03 / 0.95 | 0.90 / 0.90 (exact beyond order 0) | 0.44 / 0.44 | 1984 / 1982 | 1984 / 1982 |
| partial_corr | 2839 (K=5) / 366 (K=5) | 2839 (K=6) / 1959 (K=5) | 6.2 / 4.0 | 7305 / 7304 | 7356 / 7361 |

**What this says.** The inverses, which are what a constraint evaluates,
are within 100 ulp except partial_corr at orders 5-6, where the error is
tanh^(k) near a root of its polynomial (u = -1.57: sech^2 u = 0.155, a root of
16 - 120 S + 120 S^2), as in the scalar atanh bijector. The forward
derivatives of order K carry K times the relative error of the smallest
Cholesky pivot, which is the conditioning of the data: on the forward side
of partial_corr the error at order K is about 0.3 K cond(R) ulp (cond(R) =
2.4e4 gives 243 ulp at order 0 and 7300 at order 6; a random point of
p = 4 with cond(R) = 8.3e3 reaches 2390; the random points with cond <= 64
stay within 21 ulp), while log_cholesky at its spread points
(cond(Sigma) = 8.9e6 and 3.4e9, graded) stays below 2000, the pivots of a
graded matrix keeping their relative accuracy. Another algorithm in double
cannot do better on R itself: its entries carry eps, which moves
L_aa^2 = 1 - sum L_al^2 by eps.

ns per call (`benchmarks/bijector_coupled.cpp`; one call writes every output
of the tuple (0, ..., 0); inverse at u_k = x + 0.1 k, forward at a fixed
interior point moved by x/1000, x uniform in [-3, 3]; scratch in a View of
the space's memory), one gcc thread / 8 OpenMP threads / CUDA (RTX 4070
SUPER):

| map, dim | inverse, K = 0 | inverse, K = 6 | forward, K = 0 | forward, K = 6 |
|---|---|---|---|---|
| alr, 3 | 72 / 14 / 2.0 | 616 / 104 / 31 | 19 / 5.0 / 1.4 | 6.9 / 3.5 / 1.1 |
| alr, 10 | 137 / 20 / 3.9 | 704 / 108 / 33 | 47 / 10 / 2.7 | 11 / 4.3 / 1.3 |
| log_cholesky, 3 | 38 / 8.9 / 1.4 | 18 / 5.1 / 1.1 | 114 / 25 / 1.9 | 3849 / 526 / 261 |
| log_cholesky, 6 | 80 / 13 / 3.5 | 24 / 6.2 / 1.6 | 285 / 86 / 14 | 19282 / 2888 / 1523 |
| partial_corr, 3 | 108 / 16 / 2.1 | 162 / 26 / 2.9 | 210 / 35 / 2.2 | 8454 / 1208 / 554 |
| partial_corr, 6 | 487 / 68 / 7.4 | 525 / 75 / 7.9 | 789 / 126 / 17 | 48733 / 7128 / 3446 |

The inverses, closed or nearly so, cost about as much at order 6 as at
order 0 (the alr inverse at order 6: 0.6-0.7 microseconds on one thread, the
sub-tuple recursion over 64 subsets). The forward Cholesky maps at order 6
cost 4-49 microseconds per tuple on one thread: every packed entry carries
its 64 sub-tuple derivatives and each Leibniz sum visits the subsets of
each subset (3^6 = 729 products per entry and term). The log-Jacobians cost as
their map, except where they are closed (log_cholesky inverse: linear).

A defect found by the bench on CUDA: log_cholesky_bijector and
partial_corr_bijector first derived from two empty CRTP bases; MSVC places
the second at a nonzero offset and the device compiler does not, so a
bijector built on the host and copied into a kernel was read with a wrong
dim (illegal address). They now derive from one chain
(cholesky_bijector_orders : coupled_bijector_orders), and the test asserts
that each map has the size of its int.

**Open.** The forward Cholesky recursion visits all 2^K subsets of
every entry even when a variable does not reach that entry.

**Chosen.** This version; tested in `tests/bijector/test_coupled.cpp` on gcc
and CUDA (references, differences of order K - 1 within 0.3 of their
tolerance, the structural zero counts), bounds 1.5 times the larger measured
error.

**Derivatives of the partial correlations.** The derivatives of sech u are sech^(k) u = sech(u) P_k with T = tanh u and S = 1 - T^2 = sech^2 u (evaluated in T^2 for T^2 < 1/2 and in S beyond): P_1 = -T; P_2 = 2T^2 - 1 = 1 - 2S; P_3 = T (5 - 6T^2) = T (6S - 1); P_4 = 5 - 28T^2 + 24T^4 = 1 - 20S + 24S^2; P_5 = T (-61 + 180T^2 - 120T^4) = T (-1 + 60S - 120S^2); P_6 = -61 + 662T^2 - 1320T^4 + 720T^6 = 1 - 182S + 840S^2 - 720S^3.

## The matrix logarithm bijector

**What.** `include/laplace/bijector/matrix_log.hpp` (2026-11-13,
`note/bijectors.md` decision 20): `matrix_log_bijector` between the p x p
symmetric positive definite matrices and R^(p(p+1)/2), forward S = log Sigma,
inverse Sigma = exp(S), log |det J_h(u)| = sum_(i<=j) log f[s_i, s_j] with
f[x, y] = (e^x - e^y)/(x - y), f[x, x] = e^x, s the eigenvalues of S; both
directions and both log-Jacobians to order 6, interface and reference
coordinates as log_cholesky_bijector.

**Why not the Daleckii-Krein formula.** In the eigenbasis of S, the order-K
derivative along K directions is a sum over the K! orderings of the
directions and the p^(K+1) index chains of divided differences of exp. A
scratch prototype (one gcc thread, order 6) took 3.0 ms at p = 3 and 20 ms at
p = 4 per tuple, and lost up to 1554 ulpN against mpmath (divided
differences of close eigenvalues). The best published method (Higham and
Relton 2014, SIMAX 35:1019-1037) evaluates exp on a 2^K p block matrix,
O(8^K p^3); no published method was found whose cost is polynomial in K
(survey in the project notes). At S = 0 an entry of the order-K derivative
along basis directions counts the orderings of the tuple's edges that form a
walk, Eulerian trails, whose counting is #P-complete in general (Brightwell
and Winkler 2005; the reduction is ours and was not checked further), so a
method polynomial in both K and p is not expected.

**How.** Values and order 0: a cyclic Jacobi eigendecomposition,
f[x, y] = e^max(x, y) exprel(-|x - y|). Derivatives: a truncated Taylor
arithmetic written for this map, one symbol per distinct coordinate of the
tuple truncated at its multiplicity m_v, an element one p x p matrix per
monomial (prod (m_v + 1) of them), a product sum_(U <= T) A[U] B[T - U]
(prod (m_v + 1)(m_v + 2)/2 matrix products, 3^K for K distinct coordinates,
against the 8^K of the block matrix). exp by scaling the constant part to
norm <= 1/4, a Taylor polynomial of degree 12 + K and squaring; log by K
Newton steps X += L^-1 [Sigma(e) - exp X], L^-1 the inverse derivative of exp
at log Sigma (division of entry (i, j) by f[s_i, s_j] in the eigenbasis); the
log-Jacobians as log det J(S(e)), the n x n Jacobian of exp with one more
symbol per column, written in the eigenbasis of S_0, where J(0) is diagonal,
and factorized by LU in the scalar arithmetic. Written in the packed
coordinates, the LU mixed the columns and lost up to 3.8e5 ulpN at order 6;
the eigenbasis brought it to 1.0e5 on gcc (2.9e5 on CUDA, the same tuple
class).

**Measurement.** References in `bijector_coupled.csv` (map 3, 26680 rows):
mixed central differences of mp.expm, of the logarithm through mp.eigsy and
of the eigenvalue form of the log-Jacobian (written as
(x + y)/2 + log(sinh z / z), z = (x - y)/2, which does not cancel at repeated
eigenvalues), 1200 bits, steps 2^-130 and 2^-140. Points: p in {1, 2, 3, 4},
three random S (entries N(0, 0.25)), S = 0 (Sigma = I, every eigenvalue
equal) and a spread S (diagonal +-2); all tuples of orders 1-2, the pure
ones and 12 random mixed ones of orders 3-6. Error: ulpN as for the other
coupled maps; a derivative that is zero at a point (the odd orders of the
log-Jacobian at S = 0) is measured against the largest derivative of the
nearest order that does not vanish there. Maximum over orders 0-6,
gcc/OpenMP and CUDA:

| quantity | gcc | CUDA |
|---|---|---|
| inverse, exp | 12.0 (K=6) | 12.0 (K=3) |
| forward, log | 199 (K=6) | 210 (K=6) |
| inverse log-Jacobian | 102794 (K=6) | 291832 (K=6) |
| forward log-Jacobian | 3293 (K=3) | 2364 (K=3) |

The exponential is within 12 ulpN at every order. The logarithm loses about
30 ulpN per order (K Newton steps, each dividing by the f[s_i, s_j] of the
spread point). The log-Jacobians at orders 3-6 lose most where the value is
small against its order: the worst inverse ones are tuples of the spread
point of p = 2 whose values (7.7e-4 at order 5, 1.4e-3 at order 6) are 1/24
to 1/70 of the largest derivative of their order; the absolute error is
about 1e-14. The logarithms of the LU pivots cancel to give them.

ns per call (`benchmarks/bijector_coupled.cpp`, tuple (0, ..., 0), the
cheapest shape for the arithmetic, prod (m_v + 1) = 7 at order 6; one gcc
thread / 8 OpenMP threads / CUDA):

| p | inverse, K = 0 | inverse, K = 6 | forward, K = 0 | forward, K = 6 | inverse log-Jac., K = 6 | forward log-Jac., K = 6 |
|---|---|---|---|---|---|---|
| 3 | 406 / 90 / 52 | 8029 / 1684 / 1672 | 361 / 88 / 52 | 54075 / 7898 / 13864 | 134294 / 18708 / 28488 | 225681 / 34058 / 53396 |
| 4 | 881 / 144 / 123 | 12781 / 1822 / 3576 | 886 / 176 / 128 | 92496 / 13632 / 32173 | 372422 / 55355 / 101556 | 612460 / 89873 / 187509 |

On CUDA the arithmetic's elements (up to 64 p x p matrices per element, in
the caller's scratch in global memory) make the device slower than 8 OpenMP
threads beyond order 0. Prototype timings for the other shapes of order 6
(one gcc thread, p = 4): exp 12-180 microseconds, log 0.11-1.8 ms, inverse
log-Jacobian 0.33-6.4 ms, from a pure tuple to six distinct coordinates.

**Open.** The log-Jacobian derivatives of orders 3-6 through the
log-determinant (a form with less cancellation, through the eigenvalues, is
not found); the Newton steps recompute exp of the whole element each time;
the scratch grows like p^4 (23360 doubles at p = 4).

**Chosen.** This version; tested in `tests/bijector/test_coupled.cpp` on gcc
and CUDA, bounds 1.5 times the larger measured error.

**Layout of the truncated Taylor arithmetic.** An element represents one p x p matrix per monomial prod_v e_v^(t_v) with t_v <= m_v, indexed in mixed radix, so that its total size is prod_v (m_v + 1).

## Univariate distributions, batch 1: the location-scale families in closed form

**What.** `include/laplace/distrib/`: `gaussian2.hpp` (mu, v = sigma^2),
`gaussian3.hpp` (mu, tau = 1/sigma^2), `cauchy.hpp`, `logistic.hpp`,
`gumbel.hpp` (all three in mu and sigma) and `laplace2.hpp` (mu, lambda = 1/sigma),
each with the whole interface of `core.hpp`: the parameter descriptors, the
log-density and its derivatives to order 6, the mixed (y, theta) derivatives,
the expected information and its derivatives to order 4, cdf, ccdf, log F,
log(1 - F) with their derivatives to order 6, and the quantile. All six are
continuous with a fixed support; only laplace2 has kinks, those of laplace1
(mu is C^0 in log f and C^1 in log F at mu = y). Tests in
`tests/distrib/test_<name>.cpp`, references in
`tools/reference/distrib_gaussian23.py` (2267 and 2248 rows),
`distrib_cauchy_logistic.py` (4734 rows, 612 dropped) and
`distrib_gumbel_laplace2.py` (1314 rows, 468 dropped). The suite goes from
122 cases and 14950 assertions to 181 and 35400, green on gcc and on CUDA.

**How.** Every closed form was derived and checked against SymPy's own
derivative in a script under `tools/reference/` before being transcribed;
`gaussian23_forms.py` alone checks 232 of them. The z-derivatives of log F go
to the parameters through the location-scale rules of
`detail/location_scale.hpp`. gaussian2 and gaussian3 reuse gaussian1's log Phi
derivatives and its tail route, with the new power-scale chain below; cauchy
uses a Riccati recurrence for log(1 - F) and a monomial series in 1/|z| beyond
|z| = 2, where the z-polynomials cancel (1e41 ulp at |z| = 1e8 without it);
logistic runs one recurrence in p = F(z), and its log F and log(1 - F) are
exact as written; gumbel has log F = -e^(-z) exactly; laplace2 reuses
laplace1's series.

**The power-scale chain.** `mixed_dk` carries the z-derivatives to (mu, sigma)
only. For a second parameter w with sigma = w^k (the variance k = 1/2, the
precision k = -1/2, the rate k = -1), a derivative in w leaves the power of
sigma where it is and multiplies by k/w, but that factor is itself a function
of w, so with the state after i derivatives written as sigma^-a w^-i R_i(z),

    d^a/dmu^a d^b/dw^b Lambda = sigma^-a w^-b R_b,
    R_0 = (-d/dz)^a Lambda,   R_(i+1) = -((k a + i) R_i + k z R_i').

`mixed_power_dk` implements this. Its first version dropped the w-dependence
of k/w and was wrong from b = 2 on (relative error up to 2.4 at k = 1),
caught by checking gaussian2 and gaussian3 against SymPy.
`tools/reference/location_scale_forms.py` now checks both chains against
sympy.diff of Lambda((y - mu) w^-k) with Lambda an undefined function: 162
forms, k = 1/2, -1/2, -1, 1, 1/3, orders 1 to 6, every split between the two
parameters, all agreeing. `mixed_dk` is kept as it stands because the bounds
of gaussian1 and laplace1 were measured on that code.

**The Laguerre order of laplace2.** The series for the derivatives of log F
differentiates e^(-n lambda q) term by term, which gives b! L_b^(a-b)(u). For
a < b that order is a negative integer, where the forward recurrence starts
from terms of size b and has to cancel them down to a result of size u^(b-a);
at a = 0 the exact value is (-u)^b and the recurrence lost 720 ulp of it. The
identity L_b^(-m)(u) = (-u)^m ((b - m)!/b!) L_(b-m)^(m)(u) with m = b - a
(DLMF 18.5.12) makes the degeneracy explicit and leaves a recurrence of
positive order. Applied to laplace1, where the negative part of the order is
only -1, it measured 12.2 ulp against 11.2 for log F and no change for
log(1 - F), so it was not kept there.

**Accuracy of the log F derivatives**, against mpmath at 300 and 400 bits, in
ulp of the largest component of the order at that point ("scaled"), maximum
over orders 1 to 6, gcc build, 2026-11-13:

| distribution | log F | log(1 - F) | bounds |
|---|---|---|---|
| gaussian2 | 483 | 441 | 730, 670 |
| gaussian3 | 301 | 156 | 460, 240 |
| cauchy | 59 | 41 | 90, 62 |
| logistic | 108 | 44.4 | 162, 67 |
| gumbel | 19.1 | 394 | 30, 600 |
| laplace2 | 5.64 | 24.8 | 8.5, 37 |

For comparison the pilot measured 30.9 and 29.7 (gaussian1) and 11.2 and 9.86
(laplace1) on the same scale. laplace2's figures are those after the Laguerre
identity; before it they were 180 and 176.

**FMA contraction in the cauchy kernel test.** The kernel case of the cauchy
compared a host sum and a device sum of seven quantities at a relative
1e-14 and failed on CUDA. nvcc contracts the multiply-adds of the far tail's
hundred alternating coefficients into FMA and the host compiler does not, and
no other distribution of the batch sums a series of that length. With
`LAPLACE_CUDA_FMAD=OFF` (`build_cuda.bat <dir> nofmad`) the case passes; on
the default build the worst relative difference is 2.43e-14, so the case now
measures that difference and bounds it at 3.7e-14. Noted in passing: on the
nofmad build `tests/math/test_bessel_k.cpp` fails at 10.9 ulp against a bound
of 9, those bounds having been tuned on the FMA build; nofmad is not the
default build and nothing was changed.

**An extended device lambda in a function with internal linkage.** On CUDA the
kernel cases of gaussian3 and logistic passed every assertion and the process
then died, with an access violation for a single case and heap corruption for
a whole run. The cause is not in the distributions: doctest expands TEST_CASE
to `static void DOCTEST_ANON_FUNC_<__COUNTER__>()`, which has internal linkage
and whose counter restarts in every translation unit, while nvcc tags each
extended `__device__` lambda with the address of its enclosing function
(`__nv_dl_tag<void(*)(void), &DOCTEST_ANON_FUNC_39, 1>` in the report of
AddressSanitizer). Two units that reach the same counter produce the same
mangled tag for two different closure types; the linker merges them, and with
them the function-local `static std::map` of Kokkos's
`configure_max_dynamic_shmem` (`Cuda/Kokkos_Cuda_KernelLaunch.hpp:254`), which
is then destroyed twice at exit. nvcc's manual forbids defining an extended
lambda inside a function with internal linkage. Measured: the fault is present
with and without relocatable device code, `/OPT:NOICF /OPT:NOREF` does not
avoid it, the device constant bank (65296 of 65536 bytes) is not involved
(freeing 800 bytes changed nothing) and neither is the device stack (raising
the limit from 1024 to 131072 bytes changed nothing). Moving the two kernels
into named functions with external linkage, with nothing else changed, cleared
both faults; the same shape was then given to the other four cases of the
batch and to `tests/distrib/test_pilot.cpp`. The diagnosis came from MSVC's
AddressSanitizer, built in a scratch directory with
`/D_DISABLE_STRING_ANNOTATION=1 /D_DISABLE_VECTOR_ANNOTATION=1` because the
installed Kokkos libraries carry no sanitizer.

**What is open.** The worst errors of gaussian2 and gaussian3 are on the
central branch at |z| between 13 and 22, not in the tail route; whether that is
`log_Phi_dK` at a large positive argument or the composition is not located.
The gumbel's log(1 - F) at 394 ulp is twenty times its log F at 19.1, and the
choice between the Laguerre route and the z-polynomials there was never
**Logistic expected information.** The identity used in `expected_info` of `include/laplace/distrib/logistic.hpp` is verified symbolically against finite differences in `tools/reference/cauchy_logistic_forms.py`.

## The chain rule for two variables in two variables, to order 6

**What.** `include/laplace/math/chain_pair.hpp`:
`chain_rule_pair_dk<K>(G, u0, u1, out)`, the symmetric tensor of order K in
(x_0, x_1) of H(x) = G(u_0(x), u_1(x)), from the partial derivatives of the
outer function G in its own two arguments and those of the two inner maps in
(x_0, x_1). `faa_di_bruno.hpp` composes one function of one variable, and the
distributions have so far formed a mixed derivative as a product of one Bell
sum per parameter, which is exact only when each inner variable depends on one
parameter alone (gamma1: the shape 1/phi on phi, the ratio y/mu on mu). The
second batch of distributions has two reparametrizations that are not of that
kind, gamma2's shape mu^2/sigma^2 and lognormal2's map
sigma^2 = log(1 + v/m^2), mu = log m - sigma^2/2, and the Jacobian of an
expected information transforms the same way.

**How.** Not the multivariate Bell polynomials but one numeric recursion. With
multi-indices gamma over (x_0, x_1) and k over the arguments of G, and
D[gamma][k] = d^gamma (G_k o u), one more derivative in x_m expanded by
Leibniz's rule over the product gives

    D[gamma + e_m][k] = sum_(alpha <= gamma) C(gamma, alpha)
                        sum_(r = 0, 1) D[alpha][k + e_r] d^(gamma - alpha + e_m) u_r,

C(gamma, alpha) being the product of two binomial coefficients. The table is
filled by increasing |gamma| from D[0][k] = G_k; only the entries with
|gamma| + |k| <= K are read, so the state is 210 doubles at K = 6, laid out
compactly so that the array fits a device thread. Every coefficient is a
positive binomial, so the recursion adds no cancellation of its own: whatever
cancels does so in the derivatives it is handed. A multi-index (i, j) sits at
position (i + j)(i + j + 1)/2 + j, by total order and then by the number of
derivatives in the second variable, which within one order is the packing of
`sym_tensor<2, n>`, so a caller fills its inputs order by order.

**Measurement.** Against the truncated bivariate Taylor composition of
`tools/reference/chain_pair.py` at 300 bits, which is independent of the
recursion: 96 cases, 60 with derivatives drawn at random (magnitudes
log-uniform on 2^-8 to 2^8, random signs, so the sums cancel) and 36 on the two
maps of the batch with an outer function of the shape of a log-density.
Maximum error in ulp of the largest component of the order, gcc build,
2026-11-13, the two groups apart:

| order | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| random derivatives | 0.841 | 0.898 | 2.48 | 0.994 | 2.07 | 2.8 |
| the two maps | 0.647 | 1.06 | 1.59 | 3.74 | 0.755 | 26.2 |

The random cases, whose signs are drawn so that the sums cancel, stay within
2.8 ulp at every order; the whole of the order-6 figure comes from one point of
the gamma2 map, mu = 0.3 and v = 9 (shape mu^2/v = 1.1e-3) at y = 0.05, which
is also the worst point at orders 1 to 4. The worst point at order 5 is the
lognormal2 map at m = 0.3, v = 9. So what the recursion costs is set by the
extreme corner of a real reparametrization and not by cancelling inputs, and
the distributions that compose through it at a shape of 1e-3 inherit those 26
ulp at order 6.

An earlier version of this table, written before the test printed the groups
apart, is wrong and was corrected here: the test's own accumulator was a flat
array indexed by order while the function writing into it indexed by group, so
each group's maxima were stored in the next order's slot (and at order 6 one
element past the end). The figures above come from the repaired test.

**Open.** Nothing in the recursion; what the composition costs each
distribution is measured in that distribution's own section.

**Chosen.** This version, bounded at 40 ulp in
`tests/math/test_chain_pair.cpp`, on gcc and CUDA, with a kernel case at
order 6.

measured. Every other test file of the project that writes a `KOKKOS_LAMBDA`
in a TEST_CASE body carries the latent collision above and passes only because
its counters do not collide today.

**Chosen.** The six distributions as written, with the bounds of each test at
about 1.5 times the measured maximum, on gcc and on CUDA.

**State size and reference.** For total order K the required state is sum_(n=0..K) (n + 1) C(K - n + 2, 2) doubles, which is 210 doubles at K = 6. The reference values are computed at 300 bits of precision with mpmath (`tools/reference/chain_pair.py`, written to `tests/data/reference/chain_pair.csv`). The order 6 maximum of 26.2 ulp comes from a single point of one of the two maps, at which the arguments of the outer function span a very wide range; that same point is the worst at orders 1 to 4.

## Where the log F derivatives of the gaussians and of the gumbel lose their digits

**What.** The two accuracy questions left open by batch 1 are answered, and
both are the conditioning of the quantity rather than a defect of a route, so
nothing in the headers changed. New: `tools/reference/distrib_conditioning.py`
(2473 rows) and `tests/distrib/test_conditioning.cpp`, which measure the same
derivatives at an argument that is exact in double (mu = 0, every scale 1, so
z = y) and therefore separate the error a route makes from the error it
inherits.

**How, and why the grids of batch 1 could not tell.** On the grid of
test_gaussian2.cpp the worst points of log F sit at z = +13 and of log(1 - F)
at z = -13, that is on the side where the logarithm tends to 0 from below. The
first suspicion was cancellation in the z-polynomial of the chain, and it was
wrong: at z = 13 the ratio between the largest term of the sum and the sum
itself is 1.2 at order 6, and at order 1 the derivative is a single term. The
cause is that z = (y - mu)/sigma is itself rounded. With L_1 = R(z) =
phi(z)/Phi(z) and d log L_1 / dz about -z, half an ulp on z, that is z eps/2 in
absolute terms, becomes a relative error of about z^2 eps/2 in the result: 85
ulp at z = 13 against 97.4 measured, and 239 ulp at z = 21.9 against 301
measured. The gumbel is the same story in its own variable: log(1 - F) = -e^-u
with u = e^-z, so the inherited error is about |z| u eps/2, which at the worst
point of its grid (z = -5, u = 148) is 370 ulp against 394 measured.

**What the exact argument shows** (gcc, 2026-11-13, in ulp of the largest
component of the order, maximum over orders 1 to 6):

| route | on its own grid | at an exact argument |
|---|---|---|
| gaussian1, chain in sigma (k = 1) | 30.9 / 29.7 | 38 |
| gaussian2, chain in the variance (k = 1/2) | 483 / 441 | 117 |
| gaussian3, chain in the precision (k = -1/2) | 301 / 156 | 3.08 |
| gumbel, chain in sigma | 19.1 / 394 | 3.66 / 30.5 |

**What it says.** Most of what batch 1 measured on the gaussians and on the
gumbel was inherited, not produced: no route can do better without taking y
and the parameters in extended precision. What is left is one real and
measured asymmetry between the chains. The step of `mixed_power_dk` is
-((k a + i) R_i + k z R_i'), whose two terms subtract when k > 0 and add when
k < 0, so the chain in the precision is the best conditioned of the three and
the chain in the variance the worst: 3.08 against 117 ulp, a factor of 38, and
the same ordering appears in the cancellation ratio of the polynomial at
z = 4 and order 6, 2.4 for k = -1/2 against 291 for k = 1/2.

**The alternative chain for the variance, tried and rejected.** Giovanni asked
for the attempt, so both routes were run in double from the same z-derivatives
rounded to double, and compared with the exact value at 300 bits
(`tools/reference/variance_chain.py`): the chain now in the header against the
chain in the precision (k = -1/2, where the recursion step adds) followed by the
conversion of the tau-derivatives to v-derivatives through tau = 1/v by Faa di
Bruno. The alternative is worse almost everywhere, in ulp of the largest
component of the order:

| z | order | chain in the variance | through the precision |
|---|---|---|---|
| 3 | 5 | 21 | 141 |
| 3 | 6 | 16 | 133 |
| 4 | 5 | 3 | 5 |
| 4 | 6 | 28 | 124 |
| 6 | 6 | 13 | 5 |
| 8 | 6 | 1 | 0.2 |

The conversion costs more than the better-conditioned chain saves, because the
Faa di Bruno sum over the derivatives of 1/v brings its own cancellation. The
chain stays as it is.

**Where gaussian2's remaining 117 ulp actually comes from.** The same
measurement answers that too, and it is not the chain: fed z-derivatives that
are exactly rounded, the chain in the variance costs at most 28 ulp at the
worst point, while the header measures 117 at mu = 0 and v = 1, where z is
exact and the chain is the same. The difference is the relative accuracy of
log_Phi_dK itself near z = 3 and 4, at the upper end of its Chebyshev range.
The tests of the normal block measured those derivatives for t >= -1 in ulp of
max(|g|, 1), which cannot see a relative error in a component smaller than 1,
so this was not a regression there but a quantity that had never been bounded
relatively.

It has been now, and in the form that bounds something: `check_deriv` in
`tests/math/test_normal.cpp` measures the error of each order on [-1, 4) in
ulp of the largest |g_k| of that order on the band, which is the property of
a Chebyshev route, its error being absolute and of the size of an ulp of the
coefficients of the piece. It is uniform and small, 0.84 to 1.49 ulp over the
six orders on gcc. See "The accuracy of the log Phi derivatives on [-1, 4),
absolutely and relatively".

That was what fed the chain, and it explained gaussian2's 117 without anything
being wrong in the composition: the inputs carried about 1.5 ulp of the largest
component of their order, and at the point where the header was worst the
component the chain needs is a small fraction of its neighbours, so the same
absolute error was tens of ulp of that component.

**Measured again after those pieces were repaired (2026-11-14).** Each order
now has its own pieces on [-1, 4), a piece centred on a zero of g_k expands
g_k(t)/(t - z), and the pieces above t = 1 are half as wide, so the error
relative to the component is 2.2 to 2.9 ulp everywhere on the band instead of
reaching thousands near a zero. gaussian2's figure at an exact argument falls
from 117 ulp at z = 3 and order 5 to **29 at z = 2 and order 6**, which is the
chain's own cost and agrees with the at most 28 ulp measured independently for
that chain on exactly rounded inputs. gaussian1 does not move, 38 ulp, because
its worst point z = 4 is above the band; gaussian3 stays at 3.08 and the gumbel
at 3.66 and 30.5. On the grids of `test_gaussian2.cpp` and
`test_gaussian3.cpp` **nothing moves**: 483, 441, 301 and 156 before and after.
That is the sharpest confirmation of the diagnosis in this entry, since those
grids are dominated by the inherited z^2 eps/2 and a change to the inputs
cannot reach them.

**A figure withdrawn.** An earlier version of this paragraph said that orders
4, 5 and 6 carry 43.2, 65.3 and 36.8 ulp "away from the zeros of the
component", on a threshold that kept points where |g_k| is a hundredth of the
largest component of the order. A hundredth is not away from a zero: at that
ratio an absolute error of one ulp of the largest component is already a
hundred ulp of the component, and those three points lie less than 0.013 from
a sign change. Those figures were conditioning, not the route, and the
sentence that made them the limit on the chains was wrong.

**What is open.** Whether the 1.5 ulp of the largest component of the order
can be cut, which is Giovanni's decision and a question about how the
Chebyshev pieces of that band are generated. Whatever it is, it bounds every
chain built on those derivatives alike, gaussian1, gaussian2 and gaussian3.

**Chosen.** The routes as they are, with the conditioning measured and
recorded, and a test that pins each chain's own error so that a future change
to a chain is judged on the figure it can actually move.

## Unstructured matrices: unstructured_pdmat and unstructured_corr_pdmat

**What.** `include/laplace/pdmat/unstructured.hpp` (2026-11-13, phase 5,
batch 2; interface agreed with Giovanni the same day):
`unstructured_pdmat<B>`, B one of the coupled bijectors `log_cholesky`,
`matrix_log`, `partial_corr`, and `unstructured_corr_pdmat` =
`unstructured_pdmat<partial_corr_bijector>`. theta is the bijector's free
vector u (same packing) and M(theta) its inverse map: any symmetric positive
definite M with q = p(p+1)/2 (log_cholesky, matrix_log), any correlation
matrix with q = p(p-1)/2 (partial_corr). With it, every structure (the pilot
included) gained a last argument `double* w`, a caller scratch of
`scratch_size()` doubles (0 for the pilot), the flag `has_density`, and the
log-Jacobian of theta -> M with its derivatives (`log_jacobian`,
`log_jacobian_d1..d6`, here the bijector's inverse log-Jacobian).

**How.**
- M, its derivatives, L and its derivatives, the log-Jacobian and its
  derivatives: the bijector's functions, called directly (identical bit for
  bit, checked on the host).
- log det in closed form: log_cholesky 2 sum_i theta_ii, matrix_log
  tr S = sum_i theta_ii (first derivatives 2 and 1 on the diagonal
  coordinates, nothing above); partial_corr 2 sum_a log L_aa =
  -2 sum_(a>b) log cosh theta_ab, order K -2 tanh^(K-1)(theta_ab) on a tuple
  of one variable, 0 on a mixed one (tanh^(k) from `detail_coupled::tanh_d`).
- M^-1 for matrix_log: exp(-S), the bijector's inverse at -theta, and
  d^K exp(-S) = (-1)^K times its derivative at -theta.
- M^-1 for log_cholesky and partial_corr: a generic route in core.hpp for any
  structure that has its Cholesky factor (`detail_pdmat::generic_inverse_dk`).
  Y = L^-1 by forward substitution; along a tuple of K positions, with Y[S]
  the derivative along the positions in the mask S,
  Y[S] = -Y[0] sum_(T nonempty in S) L[T] Y[S \ T] (from L Y = I), then
  M^-1[F] = sum_(T in F) Y[T]^T Y[F \ T] by Leibniz; about
  3^K p^3/6 + 2^K p^3/3 operations for K distinct positions, and a
  derivative is zero by structure when every product has a zero factor.
- Equal sub-tuples are computed once. Two masks that select the same
  multiset of indices give the same L[T] and Y[T], so `cholesky_dK` is
  called and Y[T] computed only for the canonical mask of each class (the
  lowest positions of each group of equal indices,
  `detail/tuple_classes.hpp`), and every other mask reads it: prod (m_v + 1) - 1
  sub-tuples instead of 2^K - 1, with m_v the number of times index v occurs
  in the tuple (6 instead of 63 for the pure tuple of order 6, unchanged for
  K distinct indices). The sums still run over every subset, so the result
  is rounded exactly as without the merging: the error maxima of the
  references are the same, on gcc and CUDA. Summing over the classes
  instead, with binomial weights prod_g C(c_g, a_g), was 2 to 5 times faster
  again at order 6 but moved the errors (log_cholesky order 4 from 28.2 to
  34.7 ulpN, order 5 from 19.2 to 25.6) and was not taken (decided with
  Giovanni on 2026-11-16).
  Scratch 64 p^2 + 64 p(p+1)/2 + p^2 doubles before the structure's own.
- Why L and not M. The first version ran the same recursion on M,
  X[S] = -X[0] sum M[T] X[S \ T]: its cancellation grows with kappa(M), and
  at the spread points of the references it lost up to 1.4e8 ulpN
  (log_cholesky, p = 4, kappa(M) = 8.1e4, order 4) and 2.3e6 ulpN
  (partial_corr, p = 4, kappa = 9.7e3, order 6). On L the products are
  triangular and the loss follows kappa(L) = sqrt(kappa(M)): 28 and 60 ulpN at
  the same points. It is also cheaper (triangular products, and L[T] is
  structurally zero more often than M[T]).

**Measurement.** References `pdmat_unstructured.csv`
(`tools/reference/pdmat_unstructured.py`, 11012 rows): M built from the
definition of each map in mpmath at 640 bits (`mp.expm` for matrix_log, the
onion product for partial_corr), M^-1 by `mp.inverse`, log det by
`mp.det`, none through the closed forms of the C++ code; derivatives along a
tuple by tensor products of central differences, steps 2^-60 and 2^-70,
which agree to 1.4e-35 relative. Points: p in {1, 2, 3, 4} (partial_corr
from 2), a random u (entries N(0, 0.25)) and a spread u (diagonal +-2 and
off-diagonal 0.7 for log_cholesky and matrix_log, every u_ab = +-1.5 for
partial_corr; kappa(M) at p = 4: 8.1e4, 344, 9.7e3); tuples: all of orders 1
and 2, and for orders 3-6 the pure ones and 6 random mixed ones. Errors as
for the pilot (ulpN for M^-1, relative ulp of max(|ref|, 1) for log det); a
derivative of M^-1 that vanishes at a point (all of the tuple's references
below 1e-25, an exact zero computed as rounding noise) is measured in ulps of
the largest entry of M^-1 there. Maximum over orders 0-6, gcc/OpenMP and
CUDA:

| B | M^-1 | log det |
|---|---|---|
| log_cholesky | 28.2 (K=4) / 23.9 (K=4) | 0.28 (K=0) / 0.28 (K=0) |
| matrix_log | 24.2 (K=1) / 21.7 (K=1) | 0.06 (K=0) / 0.06 (K=0) |
| partial_corr | 59.7 (K=6) / 52.2 (K=6) | 28.4 (K=6) / 7.38 (K=6) |

Every worst case of M^-1 beyond order 0 is at a spread point (p = 4, except
log_cholesky order 6 at p = 3). The partial_corr log det at order 6 on gcc
is -2 tanh^(5)(0.444), near a zero of tanh^(5) (value 2.08 against 8.1 at
order 5), the scalar function's own cancellation; on CUDA it is 7.4 ulp. The
derivatives of log det of log_cholesky and matrix_log are exact (0 ulp).
Central differences of order K - 1 against order K agree within 4e-4 of
their tolerance (1e-6 of the largest entries).

ns per call (`benchmarks/pdmat_unstructured.cpp`, u_k = 0.1 x + 0.05 k, x
uniform in [-3, 3]; one gcc thread / 8 OpenMP threads / CUDA), M^-1 at
order 0, and order 6 along the pure tuple (0, ..., 0) and along six distinct
coordinates (the order-6 columns of log_cholesky and partial_corr measured on
2026-11-16 with the merging of equal sub-tuples; before it, the pure tuple
cost 16100 / 2560 / 782 ns at p = 3 and 26582 / 3970 / 1379 ns at p = 4 for
log_cholesky):

| B, p | M^-1, K = 0 | M^-1, K = 6, pure | M^-1, K = 6, distinct | log det, K = 0 |
|---|---|---|---|---|
| log_cholesky, 3 | 46 / 15 / 2.1 | 3698 / 547 / 114 | 6558 / 1172 / 222 | 2.3 / 3.5 / 1.0 |
| log_cholesky, 4 | 105 / 27 / 2.8 | 5635 / 779 / 215 | 8580 / 1194 / 461 | 2.5 / 3.0 / 1.1 |
| partial_corr, 3 | 104 / 26 / 2.7 | 3872 / 577 / 127 | (q = 3) | 83 / 14 / 2.1 |
| partial_corr, 4 | 185 / 39 / 4.2 | 5740 / 820 / 240 | 11644 / 1646 / 608 | 171 / 27 / 3.4 |
| matrix_log, 3 | 474 / 111 / 55 | 9493 / 1832 / 1469 | 169617 / 35381 / 29282 | 4.6 / 26 / 9.0 |
| matrix_log, 4 | 1123 / 166 / 127 | 11915 / 2937 / 3308 | 296256 / 64124 / 61653 | 3.0 / 32 / 10 |

The pure tuple is the dear one for the generic route: every one of its 63
sub-tuples is a nonzero derivative of L (for log_cholesky the diagonal
coordinate (0, 0) moves the whole column 0), while six distinct coordinates
leave most L[T] structurally zero; with the merging of equal sub-tuples the
pure tuple costs about what the distinct one does (table above). For matrix_log the order of cost is
reversed (the bijector's Taylor arithmetic grows with the number of distinct
coordinates). log det costs 1-171 ns, partial_corr's being the q
logarithms of cosh.

**Open.** The generic route needs the
Cholesky factor (`closed_factor`): a structure with neither M^-1 nor L in
closed form would need the factor computed from M first.

**Chosen.** This version; tested in `tests/pdmat/test_unstructured.cpp` on
gcc and CUDA, bounds 1.5 times the larger measured error.

## Diagonal scaling: unit_diagonal_scaled_pdmat

**What.** `unit_diagonal_scaled_pdmat<R>` (`include/laplace/pdmat/unit_diagonal_scaled.hpp`):
M = D R D with D = diag(e^(s_i)), s_i = log sigma_i, and R any structure with
unit diagonal (compound_symmetry_pdmat, ar1_pdmat, unstructured_corr_pdmat).
theta = (s, r); with `common_scale` one s for every row. Decided with
Giovanni on 2026-11-13.

**How.** Every quantity is R's, rescaled in place: M_ij = e^(s_i + s_j) R_ij,
M^-1_ij = e^(-s_i - s_j) R^-1_ij, L = D L_R, log det M = 2 sum s_i + log det R,
log-Jacobian p log 2 + (p + 1) sum s_i + R's. A derivative splits its tuple
into the scale indices A and those of R, B, and multiplies d^B of R's quantity
by c_ij(A) = prod_(k in A) (delta_ik + delta_jk) (with (-1)^|A| for the
inverse, prod delta_ik for the factor). e^(s_i + s_j) is formed as a product of
two exponentials (exp(2 s_i) on the diagonal), so that the rounding of
s_i + s_j is not magnified by |s|. No scratch of its own.

**Measured** (gcc/OpenMP, 2026-11-13; references `tools/reference/pdmat_scaled.py`,
mpmath at 640 bits from the definition, mixed central differences with steps
2^-60 and 2^-70 that agree to 2.7e-35; 13906 values; p in {1, 3} for compound
symmetry and AR(1), {2, 3} for the correlation, heterogeneous and common
scales, a random point and a spread one with s = +-3 and r = +-1.5). Worst
ulpN over orders 0..6:

| R | M | M^-1 | L | log det (ulp) | log-Jacobian (ulp) |
|---|---|---|---|---|---|
| compound_symmetry | 83.4 (K = 6) | 2.8 | 5.1 | 3.4 | |
| ar1 | 7.4 | 1.1 | 5.2 | 5.3 | |
| unstructured_corr | 40.2 (K = 6) | 16250 (K = 6); 31 with the scaling undone | 16.0 | 9.4 | 14.1 |

On CUDA (RTX 4070 SUPER, same day), in the same order: compound symmetry
76.3, 2.8, 4.8, 3.1; AR(1) 6.3, 0.73, 5.8, 6.1; unstructured_corr 17.3,
27504 (28 with the scaling undone), 16.0, 7.4, 11.1.

**What it says.** The composition adds a few ulp to R's own error; the large
figures are R's. The 83 ulpN of compound symmetry at order 6 is R's matrix
derivative itself (the same with the scaling undone). The inverse with the
correlation and heterogeneous scales: the normwise measure divides by the
largest entry of M^-1, while R^-1's error, of at most 31 ulpN of R^-1's largest
entry, is multiplied by e^(-s_i - s_j), up to e^12 apart at s = +-3. This is
the conditioning of the scaled inverse in the normwise measure, not a loss in
the code.

Measured per entry on 2026-11-14 (each entry in ulps of itself): every nonzero
entry of M^-1 and its derivatives is within 1.9, 3.0, 4.7, 7.7, 18, 103, 299
ulp at orders 0 to 6. The normwise figure comes from entries that are zero in
exact arithmetic: at order 1 along r_20 the entry (1, 1) has reference 1e-169
and comes out as 2.5e-12, the residue of R^-1's generic route (a few ulp of
max |R^-1|) times e^(-2 s_1) = e^6. The zero is a hidden cancellation,
(R^-1)_11 = 1/L_11^2 + (L_21/(L_11 L_22))^2 with z_20 cancelling between L_21
and L_22, so the order of the operations cannot remove it (e^(-s_i - s_j)
multiplies the whole sum); only a route to R^-1 with exact zeros could.

**Chosen.** The composition as designed; test bounds 1.5 times the larger of
the gcc and CUDA errors, on the matrices with the scaling undone (D^-1 M D^-1,
D M^-1 D, D^-1 L), the measure that reaches x^T M^-1 x = z^T R^-1 z for
x = D z (Giovanni, 2026-11-14). No change to the route.
## The accuracy of the log Phi derivatives on [-1, 4), absolutely and relatively

**What.** `tests/math/test_normal.cpp`, `check_deriv`, measures a third figure
for each order k = 1..6 of `log_Phi_dK` on t in [-1, 4), where the piecewise
Chebyshev expansions live: the error in ulp of the largest |g_k| of that order
on the band, and, reported beside it, the error relative to the component
itself at the point where that ratio is worst, with the value there and its
size relative to the largest component. The band had been measured only in ulp
of max(|g|, 1), which cannot see a relative error in a component smaller than
1, and on this band the components of the high orders are of order 1e-2 to
1e-3. The question came from the other side, where something was plainly
unbounded: the derivatives of log F of gaussian2 at an argument exact in double
cost 117 ulp, of which its own chain, run in Python on z-derivatives rounded
from mpmath, accounts for at most 28.

**How.** The absolute figure is the one bounded, because it is the property of
the route: the error of the Chebyshev route is an absolute quantity, of the
size of an ulp of the coefficients of the piece, and the coefficients are of
the size of the largest component of the order. The relative figure is that
same error divided by |g_k(t)|, so it grows without limit as a component
approaches one of its zeros and it bounds nothing; it is reported with the
ratio |g_k(t)| / max |g_k| so that a reader sees how much of it is
conditioning rather than inheriting the number. An earlier version of this
entry bounded the relative error at the points where that ratio is at least a
hundredth and called them "away from the zeros of the component"; a hundredth
is not away, since at that ratio an absolute error of one ulp of the largest
component is already a hundred ulp of the component, so those figures (43.2,
65.3 and 36.8 ulp for orders 4, 5 and 6) were conditioning as well, and have
been withdrawn.

**Measurement.** The rows of `log_Phi_d{k}.csv` and `log_Phi_d{k}_mid.csv`
with t in [-1, 4):

| order | gcc, ulp of max \|g_k\| | CUDA, same | relative at its worst point, gcc | there: t, \|g_k\|/max \|g_k\| |
|---|---|---|---|---|
| 1 | 1.19 | 1.10 | 8.62 | 3.916, 1.2e-4 |
| 2 | 1.28 | 1.26 | 5.34 | 3.986, 7.0e-4 |
| 3 | 1.36 | 1.36 | 8.80 | 3.964, 7.7e-3 |
| 4 | 1.41 | 1.61 | 5.34e3 | 1.002, 4.6e-5 |
| 5 | 0.913 | 0.792 | 982 | 2.071, 1.6e-4 |
| 6 | 1.72 | 1.36 | 1.70e4 | -0.8712, 6.1e-6 |

**Where the error comes from.** Three stages separated against mpmath at 300
bits, one piece of width 1 at a time, in ulp of |g_k(t)|: the exact
coefficients truncated at the degree of the order and summed by Clenshaw's
recurrence at 300 bits (the truncation of the series alone), the same
coefficients rounded to double and summed at 300 bits (the rounding of the
table), and the same summed in double arithmetic (the rounding of the
recurrence, which is what the header computes).

| order | piece | sum \|c_n\| | truncation | + table rounding | + recurrence | sum \|c_n\| / \|g_k\| there |
|---|---|---|---|---|---|---|
| 1 | [-1, 0) | 1.53 | 0.00 | 0.11 | 0.92 | 1.4 |
| 2 | [2, 3) | 0.114 | 0.12 | 0.67 | 2.72 | 8.0 |
| 3 | [3, 4) | 0.0357 | 0.01 | 2.61 | 4.35 | 11.3 |
| 4 | [0, 1) | 0.145 | 0.06 | 15.4 | 26.0 | 60.1 |
| 4 | [1, 2) | 0.222 | 0.47 | 0.65 | 16.3 | 89.2 |
| 5 | [-1, 0) | 0.069 | 0.00 | 3.23 | 12.0 | 24.9 |
| 5 | [2, 3) | 0.221 | 0.12 | 2.64 | 18.8 | 62.6 |
| 6 | [1, 2) | 0.587 | 1.58 | 9.81 | 20.2 | 120.0 |
| 6 | [2, 3) | 0.500 | 1.58 | 4.42 | 10.8 | 60.9 |

The truncation of the series contributes nothing: it stays below 0.1 ulp of
|g_k| everywhere except g5 on [0, 1), 3.0, and g6 on [1, 2) and [2, 3), 1.6.
The rounding of the table and the rounding of Clenshaw's recurrence each cost
about one ulp of the coefficient scale of the piece, and both are multiplied
by sum_n |c_n| / |g_k(t)|, the condition number of the sum at that point,
which reaches 60 to 120 where the component is small. The last column of the
first table says the same thing from the other end: for every order that has a
zero in the band, the worst relative point sits next to one of them. The zeros
are at t = 1.0023693 for g4, -0.0298336 and 2.0712889 for g5, and -0.8711322,
1.2313323 and 2.7456390 for g6. Orders 1 to 3 have no zero on the band and stay
under 9 ulp at their own worst points, in the right tail where |g_k| decays.

**What this says.** The accuracy of the route is 1.7 ulp of the largest
component of the order, uniformly over the band and on both builds, and
nothing in the generation of the pieces is loose: a degree criterion relative
to |g_k| in the point rather than to the maximum of the piece, narrower or
non-uniform pieces, and an expansion in another variable all act on the
truncation, which is already invisible. A consumer that composes one component
at a time inherits that absolute error divided by the size of its own
component, which near a zero of that component is whatever the ratio says.
This is the floor under gaussian1, gaussian2, gaussian3 and the two
lognormals, and it accounts for gaussian2's 117 ulp together with a chain
costing at most 28.

**Chosen.** The absolute figure bounded, the relative one reported with its
ratio. Bounds in `tests/math/test_normal.cpp`: 1.8, 2.0, 2.1, 2.2, 1.4 and
2.6 ulp of the largest component of the order, each about one and a half times
the larger of the two builds' maxima.

**Open.** Nothing: the candidates were measured the same day and the route
changed, in the next section.

## The zeros factored out of the Chebyshev pieces of log Phi

**What.** `log_Phi_d1` to `log_Phi_d6` on [-1, 4) keep the piecewise Chebyshev
route, but each order now has its own pieces, and a piece whose midpoint is a
zero of the component carries the expansion of g_k(t) / (t - z) rather than of
g_k. The error relative to the component itself falls from 8.6, 5.3, 8.8,
5336, 982 and 17048 ulp to between 2.2 and 2.9 at every order over the whole
band, the zeros included, while the error in ulp of the largest component of
the order stays where it was. Decided with Giovanni on 2026-11-14 after the
candidates below were measured. The routes outside the band, t < -1 and
t >= 4, are untouched.

**Why these two changes and not the others.** The diagnosis of the previous
section says that the error of the route is absolute, of the size of an ulp of
the coefficients of the piece, and that the relative figure is that error
divided by |g_k(t)|, amplified by sum_n |c_n| / |g_k(t)|. Two things make that
ratio large, and each gets its own remedy:

- the zeros of g_k in the band (g4 at 1.0023693, g5 at -0.0298336 and
  2.0712889, g6 at -0.8711322, 1.2313323 and 2.7456390). The zero is made the
  midpoint of its piece and
  divided out: h(t) = g_k(t) / (t - z) has no zero there, so the coefficients
  of the piece are of the size of h and the sum is no longer a cancellation.
  The factor is restored from z written as a pair of doubles, z_hi + z_lo; the
  subtraction t - z_hi is exact wherever it matters (Sterbenz), so the product
  keeps its relative accuracy however near t lies to the zero. The quotient is
  analytic wherever g_k is, the nearest singularity being the complex zeros of
  Phi at about 3.08 +- 3.81i, so its expansion converges as fast as g_k's.
- the decay of g_k in the right tail, where g_k falls by an order of magnitude
  across a piece of width one, so that the coefficients are much larger than
  |g_k| at the right end. Above t = 1 the pieces are half as wide.

Everything else that was tried acts on the truncation of the series, which the
diagnosis had already shown to be invisible, and none of it reaches the goal.
Worst error relative to the component, over the whole band, by order (Python
prototypes against the test's references):

| variant | 1 | 2 | 3 | 4 | 5 | 6 | table (doubles) |
|---|---|---|---|---|---|---|---|
| the route as it was | 8.6 | 5.3 | 8.8 | 5336 | 982 | 17048 | 585 |
| a relative degree criterion | 7.1 | 5.7 | 8.8 | 451 | 982 | 17048 | 640 |
| coefficients and Clenshaw in double-double | 0.8 | 0.9 | 0.7 | 78 | 1187 | 493 | 1170 |
| the same with a relative degree criterion | 0.8 | 0.6 | 0.7 | 0.5 | 1187 | 1.7 | 1280 |
| half-width pieces above t = 1 | 2.5 | 2.2 | 2.2 | 744 | 557 | 17048 | 904 |
| the zeros factored out | 7.1 | 5.7 | 8.8 | 4.5 | 2.9 | 2.7 | 698 |
| **both, the version chosen** | **2.5** | **2.2** | **2.2** | **2.3** | **2.9** | **2.7** | **1093** |

Double-double arithmetic is both dearer and weaker here, which is the
informative failure: once the arithmetic is exact the limit becomes the
truncation of the series, and on a piece that contains a zero of g_k no degree
criterion can be made relative, because the smallest value at the nodes is
arbitrarily close to zero. Order 5 stays at 1187 ulp however much precision
the recurrence is given. Only dividing the zero out makes the whole route
relative, and it does so in plain double arithmetic.

**How.** `tools/reference/log_phi_cheb.py` finds the zeros of each order by a
scan for sign changes followed by Newton's method on g_k with g_(k+1) at 300
bits, places the cuts (the ends of the band, z +- 1/2 for each zero, and the
half-width cuts above t = 1 where they fall no nearer than 0.2 to another cut
or to a zero), and fits each piece at 80 Chebyshev nodes. The degree of a piece
is the smallest whose tail of |c_m| falls below 2^-56 of the smallest |value|
at its nodes, a relative criterion, with the largest |value| as the fallback;
the degree of an order is the largest of its pieces'. Pieces and degrees: 8, 8,
8, 8, 8 and 13 pieces of degree 15, 16, 17, 18, 19 and 15, against five pieces
of degree 17, 17, 18, 19, 20 and 20 before.

The piece of t is found without a scan: 40 bins of width 1/8 cover the band,
the bin of t gives the piece whose left edge is at or before the bin's left
end, and one comparison corrects the index, which is enough because no piece is
narrower than a bin (the generator checks it). The bin index is clamped: for t
one ulp below 4, t - cheb_lo rounds up to exactly 5 and the index would reach
40, reading past the row of the order. Without the clamp the symptom is
gaussian3's derivatives at y = 1.4 and tau = 16, where z = 3.9999999999999996,
wrong by factors of 1e20, and a segfault at order 6, whose row is the last and
whose overrun leaves the table altogether.

The coefficients, 928 doubles, no longer live in the constant bank. With them
there the CUDA link failed, "File uses too much global constant data (0x11388
bytes, 0x10000 max)": the bank is one 64 KB resource for the whole device
program, the thirty-odd tables of the library already filled 65.3 KB of it, and
only 260 bytes were free. They are now in ordinary device global memory,
through a new `LAPLACE_DEVICE_GLOBAL_TABLE` in `include/laplace/core/config.hpp`
beside `LAPLACE_DEVICE_TABLE`. That is also the better place for them on their
own merits: the constant bank serves one address per cycle, so a read whose
index differs from thread to thread serializes, while a few kilobytes read
through the caches do not. The edges, the zeros, the bins and the flags, 2.4 KB
in all, stay in the constant bank, which now holds 2.3 KB less than before the
change.

**Measurement.** Per order on [-1, 4), old then new; "band" is the error in ulp
of the largest |g_k| of the order and "relative" the error relative to the
component itself at its worst point:

| order | relative, gcc | band, gcc | relative, CUDA | band, CUDA |
|---|---|---|---|---|
| 1 | 8.62 -> 2.54 | 1.19 -> 1.19 | 7.40 -> 2.45 | 1.10 -> 1.10 |
| 2 | 5.34 -> 2.21 | 1.28 -> 1.28 | 4.82 -> 2.17 | 1.26 -> 1.27 |
| 3 | 8.80 -> 2.24 | 1.36 -> 1.33 | 8.74 -> 2.19 | 1.36 -> 1.33 |
| 4 | 5336 -> 2.30 | 1.41 -> 1.49 | 4312 -> 2.46 | 1.61 -> 1.44 |
| 5 | 982 -> 2.85 | 0.91 -> 0.84 | 1068 -> 2.85 | 0.79 -> 0.84 |
| 6 | 17048 -> 2.70 | 1.72 -> 1.34 | 3432 -> 2.45 | 1.36 -> 1.41 |

Time per call on arguments spread over [-1, 4), the old route then the new, on
one gcc thread: 15.5 -> 15.7, 15.5 -> 16.6, 17.3 -> 18.0, 18.7 -> 20.6,
19.7 -> 21.7 and 19.7 -> 16.2 ns; on CUDA 0.48 -> 0.50, 0.48 -> 0.52,
0.48 -> 0.54, 0.49 -> 0.55, 0.50 -> 0.56 and 0.50 -> 0.52 ns. Order 6 gains on
the host because its degree falls from 20 to 15; the others pay one to two
nanoseconds for the bin lookup and the extra multiplication. Moving the
coefficients out of the constant bank costs nothing measurable on CUDA
(`math_normal` reports 0.21 to 0.24 ns per call for orders 1 to 4 against the
0.20 to 0.29 recorded for the old route).

**What the consumers gained.** `tests/distrib/test_conditioning.cpp`, the log F
derivatives of the gaussians at an argument exact in double, before and after:
gaussian2 117 ulp of the order's scale at z = 3 and order 5 becomes 29 at z = 2
and order 6, which is the cost of its own chain in the variance, measured
separately at no more than 28; gaussian1 stays at 38, its worst point being
z = 4, where the route is the one above the band and unchanged; gaussian3 stays
at 3.08. lognormal1's figures do not move, its worst points lying in the tail;
lognormal2's bounded figures (in ulp of the order's scale) do not move either,
and its component-relative figures shift a little at the zeros of its own
components, 6034 to 8525 at order 4 and 41500 to 40410 at order 5, which is the
conditioning of its own chain and not of these inputs.

**Chosen.** This version. Tested in `tests/math/test_normal.cpp`, which now
bounds both figures on the band: 1.8, 2.0, 2.1, 2.2, 1.4 and 2.6 ulp of the
largest component of the order, and 3.9, 3.4, 3.4, 3.7, 4.3 and 4.1 relative to
the component itself, each about one and a half times the larger of the two
builds' maxima. The whole suite passes on gcc and on CUDA.

**Open.** The constant bank is the resource to watch: before this change it was
full to 260 bytes, and the next table of any size will have to go to global
memory as well, or an existing one will have to move. Nobody has looked at
whether the tables now in it are the ones that most need it.

## Univariate distributions, batch 2: exponential, chisq and the inverse Gaussian

**What.** `include/laplace/distrib/`: `exponential.hpp` (mean mu),
`chisq.hpp` (mu degrees of freedom, treated as a continuous parameter) and
`invgauss.hpp` (mean mu, dispersion phi, Var = phi mu^3), each with the whole
interface of `core.hpp`: the parameter descriptors, the log-density and its
derivatives to order 6, the mixed (y, theta) derivatives, the expected
information and its derivatives to order 4, cdf, ccdf, log F and log(1 - F)
with their derivatives to order 6, and the quantile. All three are continuous
on (0, infinity) with a fixed support, every parameter is C^inf in log f and
in log F, and none has kinks. The exponential and the chisq carry one
parameter, so each derivative tensor has one component; neither name carries a
digit, since each family is implemented in one parametrization only. The
inverse Gaussian has no closed quantile and deliberately has no member: the
generic route of `generic_quantile.hpp` supplies it, and the test checks it
against this distribution's own cdf in both tails. gamma2 and the two
lognormals of the same batch are in their own section. The mathematics of each
form was checked by hand against its definition.

**How.**
- exponential: log(1 - F) = -y/mu is exact and its derivatives are closed.
  log F = G(y/mu) with G(u) = log(1 - e^(-u)) is split at u = 1. Above it,
  G^(k)(u) = (-1)^(k+1) q A_(k-1)(q)/(1 - q)^k with q = e^(-u) and A_m the
  Eulerian polynomials, which follows from G^(k) = (-1)^(k+1) Li_(1-k)(q) and
  whose coefficients are all positive, so nothing cancels; 1 - q is formed as
  -expm1(-u). Below it the pole is taken out exactly,
  log F = log u + log exprel(-u), the first part being -log mu plus a constant
  whose derivatives are closed and the second going through the univariate
  chain rule of `faa_di_bruno.hpp` over the derivatives of `exprel`. The value
  is log(-expm1(-u)) and needs no split.
- chisq: the log-density is the saddle-point form of gamma1 at shape a = mu/2
  and mean mu, l = -a D(y/mu) + (1/2) log a - delta(a) - log y - (1/2) log 2pi,
  an exact rewriting of (a - 1) log y - y/2 - log Gamma(a) - a log 2 that does
  not cancel at large a; it reuses gamma1's `deviance_unit` and
  `stirling_error` rather than copying them. The parameter derivatives are
  (1/2)(log(y/2) - psi(a)) and -2^(-K) psi^(K-1)(a) from `polygamma`, the
  information is psi'(a)/4 with 2^(-(K+2)) psi^(K+1)(a), and F(y) = P(a, y/2)
  carries x = y/2 free of mu with a linear in mu, so the K-th derivative of
  log F is exactly 2^(-K) times the pure shape derivative of log P, the j = 0
  column of `incomplete_gamma_mixed.hpp`. Nothing can cancel in that
  composition, so the figures below are inherited from that module.
- invgauss: the log-density derivatives are closed and elementary, with 1/phi
  factored out. The cdf is the hard part. Writing
  H(z) = log Phi(-z) + z^2/2 = log(erfcx(z/sqrt 2)/2) and using the exact
  identity z2^2/2 - z1^2/2 = 2/(phi mu), the exponential factor e^(2/(phi mu))
  never appears: the term that carried it becomes H(z2) - z1^2/2, and the
  identity is applied before differentiating, the z^2/2 parts being closed
  arrays and only H going through `chain_rule_pair_dk`. The survival rests on
  X = log erfcx(a + d) - log erfcx(a), a = z1/sqrt 2 and d = sqrt(2/(phi y)),
  whose value is log1p(-S/erfcx(a)) with S the difference of the two erfcx
  values from a series in d where d is small. H' and H'' come from a backward
  continued fraction of erfc at depth 200 for a >= 1.25, whose factors are all
  positive; H^(n) for n >= 3 are the derivatives of log Phi. Every switch and
  threshold is a named constant in the header.

**Measurement.** Every derivative order of l, of I and of log F against a
5-point difference of the order below; I against the variance of the score
through the generic route; the cdf against its identities and the quantile
against the cdf in both tails; and log F and log(1 - F) against mpmath
references written from the definition
(`tools/reference/distrib_expchi.py`, `distrib_invgauss.py`, with
`distrib_expchi_forms.py` checking each closed form against SymPy's own
derivative). Maximum error in ulp of the largest component of each order,
gcc, 2026-11-13:

| distribution | log F | log(1 - F) | bounds |
|---|---|---|---|
| exponential | 22.6 (order 6, y = 1.01, mu = 1) | 1.68 | per order, 3 to 34 and 1.0 to 2.6 |
| chisq | 294 (order 6, y = mu = 17) | 525 (order 6, y = 15.3, mu = 17) | per order, 2.2 to 441 and 1.3 to 788 |
| invgauss | 580 (order 3, upper tail) | 116 | 870 and 175 |

The exponential is at the pilot's level (gaussian1 measured 30.9 and 29.7,
laplace1 11.2 and 9.86). The chisq sits where its own cdf module does: the
pure shape column of log P reaches 176 ulp at order 5 and 651 at order 6, and
log Q at a >= 1 reaches 45.7 and 97.2, measured there in min(ulpL, ulp1)
rather than in the scale of the order, so the two measures are not the same
quantity; since the composition is one power of 1/2, the chisq figures are
inherited and not introduced, and they sit below gamma1's 379 and 356. The
inverse Gaussian's own derivatives of the log-density, measured against mpmath
over J + K <= 6, stay within 28.3 ulp.

**What the switches bought, invgauss.** The first version formed 1 - F by
subtraction: 2035 ulp at the value in the far upper tail and about 1e4 in the
orders. The erfcx form removed that. The identity above then took the small
region from 1398 to 179 ulp at order 6, by removing the term
2/(phi mu) = 666.7 that was cancelling against log Phi(-z2) at a measured
factor of 8e3 (4.8e17 against -4.8006e17). The continued fraction is required:
switched off, the central region of log F runs from 1e3 to 4e3 and the small
region to 7e5. Its depth matters as much: at depth 60 K' was wrong by 4e-13 at
a = 1.78, which alone set the far tail of log(1 - F) at 580 to 630 ulp. The
series in d was swept over thresholds from 1/16 to 2; 1 was chosen, and at 2
the series breaks down.

**What is open.** The far-tail log F of the inverse Gaussian at order 3, 580
ulp, is unchanged by every switch and is not located; the bound is set on it.
The band where the series in d applies still carries one scalar difference
whose cancellation factor is about 1/d, measured at about 11 where d = 0.09.
The chisq figures will move only when the pure shape column of the incomplete
gamma moves. The derivatives of H for orders 1 and 2 are computed here by a
continued fraction of this header's own, because the shared tail route of
gaussian1 covers only arguments above about 4 while the inverse Gaussian needs
them from 1.25: the two should be reconciled when the normal tail is next
revisited.

**Chosen.** The three distributions as written, with the bounds of each test at
about 1.5 times its measured maximum, on gcc.

## Univariate distributions, batch 2: the two lognormals

**What.** `include/laplace/distrib/lognormal1.hpp` (mu and v = sigma^2, the
mean and the variance of log Y, neither of them a moment of Y) and
`lognormal2.hpp` (the mean m and the variance v of Y), each with the whole
interface of `core.hpp`. Both are continuous on (0, infinity), smooth in every
parameter and in y, with no kinks. They are the two parametrizations
`note/distrib.md` point H.2 keeps although only the second one's first
parameter is a moment of Y.

**How.** With x = log y, l(mu, v; y) = l_gaussian2(mu, v; x) - log y, and
-log y carries no parameter, so every parameter derivative of order 1 to 6,
the expected information with its four derivative orders, and log F, log(1 - F)
with their derivatives are `gaussian2_distrib`'s at x, unchanged: lognormal1
builds one and forwards to it, instead of carrying a second copy of the normal
tail route. Its own members are the value, the quantile (the exponential of
gaussian2's), the support, the hints, and the derivatives in y, where
d/dy = (1/y) d/dx gives

    d^J/dy^J f(log y) = y^(-J) sum_(k = 1..J) s(J, k) f^(k)(log y)

with s(J, k) the signed Stirling numbers of the first kind, one function per J.

lognormal2 is the same law through s = log(1 + v/m^2) and mu = log m - s/2, so
its derivatives are lognormal1's composed with a two-into-two map in which both
inner variables depend on both parameters; the composition is
`chain_rule_pair_dk` (see its own section). Its information is the tensor
transform I_new = J' I_old J of diag(1/s, 1/(2 s^2)), with a nonzero
off-diagonal entry, -3/(2 m v) to leading order at small v/m^2; its derivatives
to order 4 come from the same routine applied to the products, the product rule
being the chain rule of the product function, so there is no private Leibniz
recursion. The residual x - mu is formed as log1p((y - m)/m) + s/2 near y = m
and as log y - log m + s/2 elsewhere.

**A defect this found.** The first version formed x - mu by subtracting, which
cancels when y is near m and v/m^2 is small: 6e-11 relative at (m, v) =
(40, 0.001) and y = 40 in the mixed (y, theta) derivatives. The residual form
above removed it, and the measured maximum over every (J, K) with J + K <= 6
at six points is 23.76 ulp of the group's scale (J = 1, K = 5), bounded at 36.

**Measurement.** Every order against a 5-point difference of the order below;
the information against the variance of the score and against the generic
quadrature route; log F and log(1 - F) against mpmath at 300 and 400 bits
(`tools/reference/distrib_lognormal.py`, 7776 rows, plus 504 rows for the
mixed derivatives in y). Checked against independent mpmath values at
300 bits, 2026-11-14: lognormal2's expected information agrees with the exact
transform to 1 to 2 ulp in every component at (1.5, 0.8), (40, 0.001) and
(0.5, 50), and so do log F and its first derivatives; the off-diagonal entry
times m v is -1.4999995 at (40, 0.001), as the leading term says. The
cancellation of the expected information that was recorded for this family at
mean 40 and variance 0.001 does not occur in this formulation.

Maximum error of the log F derivatives in ulp of the largest component of each
order, gcc, 2026-11-13:

| distribution | log F | log(1 - F) | bounds |
|---|---|---|---|
| lognormal1 | 322 (order 6) | 648 (order 6) | 483 and 973 |
| lognormal2 | 7056 (order 6) | 169 (order 2) | 10600 and 255 |

**What this says.** Two different things are in those numbers. The first is
that the accuracy of the normal route both distributions inherit through
gaussian2 is absolute, set by the largest component of the order and not by the
component being read: where a high-order component falls near a zero of itself,
one ulp of the order's largest component is already many ulp of that component,
and the figures above are in ulp of the order's scale for exactly that reason.
At the worst row of lognormal2's order 6 the component is about one part in a
hundred and twenty of the largest component of its order (7.06e3 ulp of the
order's scale against 8.55e5 ulp of the component itself), so almost all of
that reading is the smallness of the component. That error is uniform across the order
rather than larger at the high orders, so a chain that works one component at a
time inherits it divided by the size of its own component, which near a zero of
that component is whatever the ratio says: conditioning, not a floor. No
number is put on the normal route's own share here. An earlier figure of 43
to 65 ulp, quoted as a relative floor, was withdrawn as being that same
absolute error divided by a small component; the route was then improved
(see its own section), and it is now accurate to a couple of ulp of the
component itself across the band, zeros of the components included, so there
is no inherited floor of any size to quote. What remains in the figures
above is the cost of the composition and the conditioning of the components
being read. Remeasured after that change: lognormal1 does not move, its
worst points lying in the tail, and lognormal2's figures in ulp of the
order's scale do not move either. The second is that the composition through
the map costs lognormal2 about three times lognormal1's error at orders 1
and 2 and
twenty to thirty times at orders 3 to 6 on the common points, the worst rows
being at v/m^2 = 200 with y far in the upper tail, where log F tends to zero
and every derivative of it is a cancelling sum. At the opposite corner the
comparison reverses: at (40, 0.001) lognormal2 is more accurate than its
lognormal1 twin, 6034 against 5.4e6 ulp at order 4, because its residual avoids
the subtraction that lognormal1 cannot avoid there.

**The conditioning of lognormal1 at a tiny variance.** At the twin of that
corner, (mu, s) = (3.689, 6.25e-7), the residual log y - mu is about 1e-17
while the rounding of log y alone is about 4e-16: the digits are gone before
the library sees them, and no rearrangement inside lognormal1 can recover them,
because mu and y are what the caller passes. Measured by perturbing y and mu by
one ulp and recomputing with mpmath at 300 bits, the library's error is at most
0.165 of max(one ulp of the result, the change caused by one input ulp), where
the plain measure reads up to 5.4e6 ulp. So the distribution is as accurate as
its arguments allow; those rows are excluded from the plain-ulp check with that
figure recorded in the test.

**What is open.** lognormal2's log F at orders 3 to 6 in the far upper tail at
large v/m^2, 7056 ulp at order 6: the region is located and the smallness of
the components is part of it, but the share that belongs to the chain itself is
not separated, and separating it waits on the log Phi table. The 5-point difference in y cannot hold the tolerance of the other
checks for lognormal2 (measured 6.7e-8 against 3.8e-10 at orders 5 and 6), so
there the mpmath reference is the only check, which is the stronger one.

**Chosen.** Both distributions as written, with every bound at about 1.5 times
its measured maximum, on gcc and on CUDA.

## Univariate distributions, batch 2: gamma2, the gamma in mean and variance

**What.** `include/laplace/distrib/gamma2.hpp`: the gamma law in its mean
mu > 0 and its variance sigma^2 > 0, so that the shape is s = mu^2/sigma^2 and
the scale sigma^2/mu, with the whole interface of `core.hpp`. Unlike gamma1's
(mu, phi), these two parameters are not orthogonal: the off-diagonal entry of
the expected information is not zero, and a derivative in mu reaches both the
shape and the ratio y/mu.

**How.** The log-density is gamma1's cancellation-free form with a different
inner map,

    l = -s D(y/mu) + (1/2) log s - delta(s) - log y - (1/2) log 2pi,
    D(z) = z - 1 - log z,   s = mu^2/sigma^2,

with delta Stirling's error; the deviance and the Stirling error are gamma1's
own functions, not copies. The outer function G(s, c) = -s D(c) + (1/2) log s
- delta(s), c = y/mu, has closed partials in (s, c) to total order 6 (the only
mixed term is -s D(c), whose derivatives in s above the first vanish), and the
inner maps s(mu, sigma^2) = mu^2/sigma^2 and c(mu) = y/mu have closed partials
too; both inner variables depend on mu, so the composition is the two-into-two
chain rule `chain_rule_pair_dk` and not a product of one Bell sum per
parameter. log F = log P(s, s c) composes the partials of log P(a, a c) at
fixed ratio the same way. The expected information is the tensor transform
I_new = J' I_old J of gamma1's, with J the Jacobian of phi = sigma^2/mu^2, and
its derivatives to order 4 follow by the same routine. The quantile is
(sigma^2/mu) P^-1(s, p).

**Measurement.** Every derivative order of l, of I and of log F against a
5-point difference of the order below, in the parameters and in y; the
information against the variance of the score and against the generic
quadrature route; the cdf against its identities and the quantile against the
cdf; and l, log F and log(1 - F) against mpmath references computed from the
definitions (`tools/reference/distrib_gamma2.py`, 4510 rows, shapes 1e-4 to
1e4 and c = y/mu from 0.01 to 10). That script also checks every closed form of
the header against SymPy's own derivative, including the entries of the
information and the Leibniz expansion of its derivatives, and it writes the
direct closed forms of the log-density at orders 1 and 2 as a second route:
the composed route reproduces them to the same error as the log-density rows,
4.41 and 5.41 ulp, which is the check that the composition is the right
formula and not only a convergent one. Maximum error in ulp of the largest
component of each order, gcc, 2026-11-14:

| quantity | order 1 | 2 | 3 | 4 | 5 | 6 | bound |
|---|---|---|---|---|---|---|---|
| log f | 4.41 | 5.41 | 39.8 | 20.1 | 38.7 | 186 | 280 |
| log F | 138 | 131 | 191 | 274 | 1.13e3 | 4.60e3 | 6900 |
| log(1 - F) | 122 | 125 | 97.3 | 197 | 482 | 1.20e3 | 6900 |

**Where the error is, and where it is not.** The worst rows are not at the
extreme shapes, where the partials of the incomplete gamma are hardest: at
shapes 1e-4 and 1e4 the log F derivatives stay within 5.6e2 and 7.0e2 ulp at
order 6. They are at shape 1, at c between 0.7 and 10, in the components with
every derivative in one parameter: 4.60e3 at order 6 with all six derivatives
in mu (c = 0.95), 2.59e3 at c = 1 and 2.42e3 with all six in sigma^2 at
c = 10. That is the signature of the composition rather than of the partials,
because a derivative in mu reaches both inner variables and the component with
all derivatives in mu is the one where the chain sums the most products. For
comparison the pilot gamma1, whose shape depends on one parameter alone and
whose ratio depends on the other, measured 379 and 356 on its own grid.

**The expected information at extreme shapes.** Checked against the exact
transform of the gamma information, computed with mpmath at 300 bits from the
textbook entries s/mu^2 and s^4 (psi'(s) - 1/s): the closed form agrees to 1
to 2 ulp in every component at (mu, sigma^2) = (0.4, 1600), (0.4, 0.16),
(2.5, 0.125), (1e4, 1) and (1, 1e-8), that is at shapes from 1e-4 to 1e8. The
cancellation of the information at a shape of 1e8 that was recorded for this
parametrization does not appear.

**What is open.** The composition's cost at shape 1, above. The component with
every derivative in mu could instead be composed over (a, x) with a = mu^2/v
and x = mu y/v, both polynomial in mu, where the objection that rules that
route out for gamma1 does not apply, since along mu the ratio x/a = y/mu is
not fixed; whether that is more accurate than the present route has not been
measured, and it is a choice between two routes rather than a repair. The
mpmath table stops at shape 1e4 because mpmath's own gammainc raises
NoConvergence at shape 1e8 for every c >= 1 and takes no maxterms argument;
that shape is covered by the SymPy checks, by the difference checks at
mu = 1e4 and sigma^2 = 1, and by the comparison of the information above.

**Chosen.** As written, with the bounds of the test at about 1.5 times its
measured maxima, on gcc and on CUDA.

## Univariate bases: the choices of route, measured before the code

**What.** The decisions behind `include/laplace/basis/` (2026-11-13/14, with
Giovanni), each taken on a small measurement written for the purpose and not
kept in the repository; the chosen routes are measured again by the tests and
the bench of the entries below.

**B-spline row: interval and degree.** Cubic, eta = sum beta B over one row,
n = 1e6 random x, K = 10/40/200, ns per row:

| | K | knots from (lower, h) | binary search | degree as template | one thread / 16 / CUDA |
|---|---|---|---|---|---|
| uniform, runtime degree | 200 | yes | | | 13.1 / 1.95 / 0.41 |
| array, runtime degree | 200 | | yes | | 37.3 / 3.14 / 0.35 |
| uniform, template degree | 200 | yes | | yes | 9.8 / 1.20 / 0.39 |
| a stored row, read | 200 | | | | 1.5 / 0.33 / 0.03 |

The binary search costs on the CPU and grows with K (random x make its
branches unpredictable); on CUDA the array is 15-20% faster than the
arithmetic interval. The degree as a template parameter is worth 10-25% on one
thread, 35-60% on 16, 4% on CUDA. Chosen: one type, knots always in an array,
the arithmetic interval when the knots are uniform, the degree at runtime with
one loop.

**Fourier: sin(j theta) and cos(j theta), j <= 100** (mpmath at 400 bits;
error in ulp(1); conditioning 2 pi j x eps, from a relative error eps in x):

| route | max error | error / conditioning | ns per row, H = 100: one thread / CUDA |
|---|---|---|---|
| sincos(j z), z = 2 pi x rounded | 533 | 1.05 | 2098 / 15.9 |
| j x mod 1 from an exact product, then sincos | 1.6 | 0.7-1.0 | 2562 / 18.7 |
| rotation by the angle of j = 1 | 157-171 | 21-22 | 236 / 3.2 |
| Chebyshev three-term recurrence | 2429 | 2300 | 231 / 1.9 |
| rotation re-anchored by the exact route every 8 harmonics | 12.5-13.5 | 3.0 | 633 / 5.3 |

The exact product uses fma on the device and Dekker's split on the host (the
gcc build has no hardware fma and std::fma is emulated); the rounding to an
integer adds and subtracts 1.5 2^52 (nearbyint is slow on mingw). Chosen: the
rotation re-anchored every 8 harmonics.

**Legendre: P_n^(k)(t), n <= 50, k <= 6** (references from the exact integer
coefficients, mpmath at 400 bits): three routes, Gegenbauer
(2k - 1)!! C^(k + 1/2)_(n-k), the Bonnet recurrence differentiated (orders
0..k carried together) and the recurrence P^(k)_(n+1) = P^(k)_(n-1) +
(2n + 1) P^(k-1)_n on arrays, are all within 1-4 times the conditioning
|t P^(k+1)(t)| eps (up to 359 ulp of |P_n^(k)(1)| near t = +-1 at n = 50).
At n = 49, 16 threads: Gegenbauer 12.6-14.8 ns, Bonnet 29-55, the arrays
15-34; on CUDA 3.4-3.7, 3.2-18.2, 2.7-6.2. Chosen: Gegenbauer (no scratch).

**B-spline: values and derivatives.** De Boor's BSPLVB for the values, then
k steps of the derivative formula (BSPLVD), against a closed form of the
cardinal B-spline for uniform extended knots (Horner in the local coordinate):
the closed form is 2-7 times faster (one thread, d = 3: 8 ns against 14-38;
CUDA 0.13-0.24 against 0.34-1.9) but loses up to 34 ulp at d = 5 to the
cancellation of the expanded pieces, and covers one arrangement of knots.
Chosen: BSPLVB/BSPLVD only. The same measurement found that the arithmetic
interval floor((x - lower)/h) picks the previous interval for points on a
knot or one ulp from it; the interval is now corrected against t_s and
t_(s+1).

## The B-spline basis

**What.** `bspline_basis` and `bspline_basis_owner`
(`include/laplace/basis/bspline.hpp`): degree d <= 10, knots t_0..t_(K+d)
with any multiplicity up to d + 1, domain [t_d, t_K]; rows of order 0..6
(BSPLVB, BSPLVD), expansions, the integral from lower (a window of d + 1
values and the column totals before it, from the degree d + 1 B-splines), the
cross Gram matrices G_kl = int B^(k) B^(l)' (16-node Gauss-Legendre per knot
interval, in the interval's own coordinate x - t_s), Greville abscissae, the
points of reduced continuity, and the outside rules (nan, linear, polynomial).

**How.** `tools/reference/basis_bspline.py`: exact rational values from the
Cox-de Boor definition and the derivative formula, integrals and Gram matrices
by exact integration of the polynomial pieces (their Taylor coefficients at
t_s), independent of the integral formula of the code; a SymPy check
(`bspline_basis_set`) of the values and derivatives agrees exactly. 42 knot
configurations, d in {0, 1, 2, 3, 5, 10}: uniform clamped and extended,
random, interior multiplicity 2..d + 1, an interval of width 1e-12, the
domain [1000, 1001]; points on the knots and one ulp away. Rows in ulps of
the row's largest value; integrals and totals in ulps of the largest total;
Gram entries in ulps of the largest entry.

| | d <= 5 | d = 10 |
|---|---|---|
| rows, orders 0..6 | 7 | 19 |
| integrals and totals | 3 | 4.7 |
| Gram matrices | 6 | 6 |

The same bounds hold on CUDA. Two defects found by these references and
repaired: the total of a column of extended knots stopped at the end of its
support instead of at upper; the Gram matrices took the quadrature nodes as
x = c + h x_i, which on [1000, 1001] lost up to 6768 ulp to the rounding of
x, and now take them in the coordinate x - t_s (4 ulp).

**Speed** (`benchmarks/basis_bspline.cpp`, cubic, uniform clamped knots on
[0, 1], ms per call; R 4.6.0 with splines2 0.5.4 and splines::splineDesign,
which return a dense n x K matrix allocated at every call; Laplace with its
sparse rows and with a dense output into a preallocated matrix):

| n = 1e6, values | K = 10 | K = 40 |
|---|---|---|
| splines2::bSpline | 107 | 412 |
| splines::splineDesign | 141 | 216 |
| Laplace dense, one thread | 22.2 | 20.7 |
| Laplace sparse, one thread | 18.8 | 15.8 |
| Laplace sparse, 16 threads | 2.7 | 2.7 |
| Laplace dense, CUDA | 0.38 | 2.0 |
| Laplace sparse, CUDA | 0.37 | 0.37 |

The second derivative and the integral cost 1.1-2.0 times the values (2.0 for the sparse integral on CUDA).
At n = 1e5, K = 200: splines2 167 ms, splineDesign 53, Laplace dense 7.1 on
one thread, sparse 1.6 on one thread and 0.044 on CUDA.

**What it says.** The sparse row makes the cost independent of K; a dense
output is limited by the writing of zeros, on CUDA above all.

**Chosen.** As written; test bounds 12 and 32 ulp for the rows, 8 and 12 for
the integrals, 12 for the Gram matrices.

## The Fourier basis

**What.** `fourier_basis` and `make_fourier_basis`
(`include/laplace/basis/fourier.hpp`): columns 1, sin(j nu x'), cos(j nu x'),
j = 1..H, x' = x - lower, nu = 2 pi / omega, on [lower, upper] (upper = lower
+ omega by default; a shorter or longer domain is allowed). The sines and
cosines by the rotation re-anchored every 8 harmonics; derivatives by the
shift identity; the integral from lower by one chain over the half angle
(2 sin^2(theta/2) and 2 sin(theta/2) cos(theta/2), never 1 - cos); Gram
matrices in closed form by product-to-sum, block diagonal on a full period.

**How.** `tools/reference/basis_fourier.py`: mpmath at 300 bits from the
definition, 27 configurations ([0, 1], [-3, 4.5] with omega = 7.5, [0, 10]
with omega = 7, [0, 9] with omega = 12, [1000, 1001], [1000, 1003] with
omega = 0.7; H in {0, 1, 4, 20, 100}), points random, outside the domain and
where j u is near an integer or a half-integer; Gram references from complex
exponentials, a route independent of the code's. Rows in ulps of
max(1, the row's largest |value|), and divided by the conditioning
max(1, 2 pi j |x'|/omega) + k:

| H | rows gcc / CUDA | ratio to conditioning | integrals gcc / CUDA | Gram |
|---|---|---|---|---|
| 1 | 24.5 / 24.5 | 1.0 / 1.0 | 2.0 / 1.1 | 11 |
| 4 | 93 / 93 | 1.2 / 1.1 | 2.0 / 2.0 | 11 |
| 20 | 437 / 437 | 2.0 / 2.5 | 2.0 / 2.0 | 18 |
| 100 | 2147 / 2147 | 2.0 / 2.0 | 1.5 / 1.2 | 11 |

**What it says.** The row errors are the rounding of x'/omega times
2 pi j |x'|/omega: the worst case is lower = 1000, omega = 0.7, x' up to 3.

**Open.** x' and x'/omega in double-double would remove that term, for one
more product per row.

**Chosen.** As written; test bounds 1.5 times the measured maxima.

**Exact reduction of the argument.** For harmonic j, the exact reduction route computes the product j u as an exact double-double split p = p_hi + p_lo, using `Kokkos::fma` on the device and Dekker's two-product split on the host. Rounding to the nearest integer is done by adding and subtracting the constant C = 1.5 x 2^52 = 6755399441055744.0 (ties to even, exact for |p| < 2^51): round(p_hi) = (p_hi + C) - C. The reduced turns r = j u - round(j u) and half-turns r_h = (j u/2) - round(j u/2) are formed as r = (p_hi - round(p_hi)) + p_lo and r_h = (0.5 p_hi - round(0.5 p_hi)) + 0.5 p_lo.

## The Legendre basis

**What.** `legendre_basis` and `make_legendre_basis`
(`include/laplace/basis/legendre.hpp`): b_m = sqrt((2m + 1)/w) P_m(t),
m = 0..n, w = upper - lower, t = 2(x - lower)/w - 1, orthonormal on the
interval. Derivatives by Gegenbauer; t computed from the nearer end
(1 - 2(upper - x)/w in the upper half), which cut the row error near t = +-1
three to five times (n = 50, k = 0: 2468 to 551 ulp). Integral columns as
(t^2 - 1) P_m'/(m(m + 1)), the same polynomial as
(P_(m+1) - P_(m-1))/(2m + 1) without its loss near the ends (in a double
emulation at n = 20, relative error 3e-5 at 1e-12 from lower, against 4e-15).
Gram matrices: G_00 = I written exactly, the others by Gauss-Legendre in t,
the 64-node table up to degree 127 and a rule of degree/2 + 1 Newton nodes
above it (panels would not lower the degree of the integrand).

**How.** `tools/reference/basis_legendre.py`: exact integer coefficients,
mpmath at 400 bits, exact polynomial integration; n in {0, 1, 2, 5, 10, 20,
50} on [0, 1], [-1, 1], [-3, 7], [1000, 1001]. Rows in ulps of the row's
largest |value|, gcc / CUDA:

| n | k = 0 | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 | integral | Gram |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 9 / 7 | 5 / 4.5 | 6 / 5 | 4 / 4 | 2 / 2 | 3 / 3 | 0 / 0 | 5 / 4 | 6 |
| 10 | 28 / 20 | 16 / 14 | 9 / 10 | 12 / 11 | 6 / 10 | 12 / 6 | 5 / 5 | 12 / 11 | 14 |
| 20 | 115 / 88 | 38 / 34 | 32 / 39 | 28 / 23 | 34 / 25 | 19 / 20 | 24 / 26 | 35 / 41 | 84 |
| 50 | 551 / 474 | 292 / 258 | 194 / 194 | 116 / 111 | 145 / 149 | 136 / 104 | 104 / 116 | 197 / 193 | 384 |

n = 0, 1, 2 within 4 ulp everywhere. Against the conditioning
eps/2 (|b^(k)| + (1 + |t|) |b^(k+1)| w/2): at most 5.6 on gcc and 13.4 on
CUDA (one entry, n = 50, k = 3).

**What it says.** The large figures at n = 50 are the conditioning of P_n in
t near the ends; CUDA differs from gcc through the contraction into FMA.

**Chosen.** As written; test bounds 1.5 times the measured maxima.

## Linear differential operators and their penalties

**What.** `differential_operator` (`include/laplace/basis/operator.hpp`):
L = D^m + c_(m-1) D^(m-1) + ... + c_0 stored by the roots of its
characteristic polynomial with their multiplicities (merged exactly in
products), the coefficients by multiplying the factors, the null space
x'^i e^(a x') cos/sin(b x'), and `operator_penalty_matrix`,
S = sum_(k,l) c_k c_l G_kl from the basis's cross Gram matrices, so that the
penalty int (L f)^2 is exact on every basis. `differential_operator_from_weights`
finds the roots of a polynomial given by its coefficients by the
Aberth-Ehrlich iteration and groups approximations within 1e-4 s
(s = 2 max_j |w_j|^(1/(m - j)), Fujiwara's bound on the roots), since a root of multiplicity mu is
found only to about eps^(1/mu) s.

**How.** `tests/basis/test_operator.cpp`: D^m gives G_mm bit for bit; on a
full Fourier period the penalty of the harmonic operators equals
|P(i j nu)|^2 omega/2 on the diagonal within 64 eps of the largest entry; the
polynomials of degree < 2 (Legendre, D^2) and the quadratics written by
Marsden's identity (cubic B-spline, D^3) have a penalty below 1e-12 of a
random vector's; from weights, the roots of a D + D^2 (0 exactly and -a),
D^2 (D^2 + nu^2), (D + 1)^3 and D^2 + D + 4.25.

**Open.** Distinct roots closer than 1e-4 s are merged by
`differential_operator_from_weights`; such operators are given by their
roots.

**Chosen.** As written.

**Details of the root finding and of the tests.** The Aberth-Ehrlich iteration finds the m roots simultaneously in complex arithmetic, initialized on a circle of radius s and updated by simultaneous Newton steps with a repulsion between the approximations, dz_k = (P(z_k)/P'(z_k)) / (1 - (P(z_k)/P'(z_k)) sum_(j != k) 1/(z_k - z_j)). It stops when the maximum step satisfies max_k |dz_k| <= 4 eps s (eps = 2.22e-16) or after 500 sweeps. A multiple root of multiplicity mu is conditioned to eps^(1/mu) s only (about 6e-6 s for mu = 3), which requires a grouping tolerance such as 1e-4 s to recover the true multiplicity. The coefficients c_j reconstructed from the grouped roots are checked against w_j: `differential_operator_from_weights` throws `std::invalid_argument` if the order does not match m (conjugate roots could not be paired) or if, for any j in {0, ..., m - 1}, |c_j - w_j| > 1e-6 max(max_k |w_k|, s^(m-j)). In the tests, D^m with m in {1, 2, 3} on a uniform clamped cubic B-spline basis gives `operator_penalty_matrix` equal bit for bit to `basis.cross_gram(m, m, ...)` (S[i] == G[i]). On a full Fourier period [0, omega] with omega = 366.0, nu = 2 pi/omega and 6 harmonics (K = 13), the penalty matrix of the harmonic operator L = D prod_(h=1..H) (D^2 + (h nu)^2), H in {1, 2}, is diagonal; for a column i > 0 of harmonic j = floor((i + 1)/2) the entry is |P(i j nu)|^2 omega/2 = (j nu)^2 prod_(h=1..H) ((h nu)^2 - (j nu)^2)^2 omega/2, and the computed matrix agrees with it within 64 eps max_(a,b) |S_ab|. On a Legendre basis of degree 8, D^2 applied to the polynomials of degree < 2 (columns 0 and 1) gives entries bounded by |S_ij| <= 1e-13 max |S|. On a clamped cubic B-spline basis, D^3 applied to the quadratics written by Marsden's identity gives quadratic penalties below 1e-13 of a random vector's penalty for the constant and linear functions, and below 1e-12 for the quadratic functions. For the root finding from weights: a D + D^2 with a = log(2)/3 has root 0 exact, root -a within 1e-14 and coefficients reproduced within 1e-12 max(|w|, 1); D^2 (D^2 + nu^2) = D^4 + nu^2 D^2 has the double root 0 exact and the complex pair +-i nu with real part zero and imaginary part within relative error 1e-10; (D + 1)^3 has the triple root -1 grouped with real part within relative error 1e-5; (D + 0.5)^2 + 4 = D^2 + D + 4.25 has the complex pair -0.5 +- 2i within relative error 1e-13.

## Dense linear algebra: laplace::dense

**What.** `include/laplace/dense/dense.hpp`: Cholesky, triangular solve,
pivoted QR, SVD, symmetric and generalized symmetric eigendecompositions and
an orthonormal basis of a null space, on host Kokkos Views
(`View<double**, LayoutLeft>`, column-major). The work is done by Eigen 3.4
(MPL2, `EIGEN_MPL2_ONLY`) in one translation unit,
`src/dense/eigen_backend.cpp`, compiled as C++ in every build (never by nvcc)
into the static library `laplace_dense`; the header does not include Eigen.
One thread per call (`EIGEN_DONT_PARALLELIZE`). Numerical outcomes come back
in `dense_decomposition_status` {success, rank, reason}; wrong dimensions
throw `std::invalid_argument`. Views in device memory are rejected at
compile time; a copy route for them is added when a caller has device data.

**Candidates, measured before the choice.**

- Kokkidio (Eigen inside Kokkos kernels): GPL-3.0, incompatible with MIT;
  not measured.
- Kokkos Kernels 5.2.2. KokkosLapack has potrf, potrs, geqrf, gegqr, gemqr,
  gesv, svd and trtri, and no code of its own: every routine needs LAPACK,
  cuSOLVER, rocSOLVER or MAGMA; there is no symmetric or generalized
  eigensolver. KokkosBatched has QR, QR with pivoting, SVD and LU for many
  small matrices, banded and tridiagonal Cholesky but no dense Cholesky, and
  a nonsymmetric eigendecomposition that is host-only with LAPACK and does
  not compile with gcc. One K x K matrix, ms, CPU (OpenBLAS's LAPACK, one
  thread) / GPU (cuSOLVER, RTX 4070 SUPER):

  | K | potrf | geqrf | svd |
  |---|---|---|---|
  | 50 | 0.004 / 0.19 | 0.017 / 1.0 | 0.27 / 5.6 |
  | 200 | 0.086 / 0.79 | 0.33 / 2.0 | 10.2 / 36.3 |
  | 1000 | 6.3 / 3.3 | 22.1 / 12.9 | 1069 / 270 |

  1e5 matrices m x m, ms for the whole batch, KokkosBatched on 16 threads /
  CUDA, with Eigen in an OpenMP loop on 16 threads in brackets: m = 4 QR
  0.88 / 0.27 (3.4), SVD 9.8 / 3.3 (11.0); m = 8 QR 2.6 / 2.1 (7.1), SVD
  32.5 / 24.7 (58.3); m = 16 QR 13.3 / 147 (18.1), SVD 142 / 774 (345);
  Eigen's symmetric eigendecomposition 8.6 / 23.6 / 79.3 at m = 4 / 8 / 16.
  The batched SVD errs by 2e-15 to 6e-15. The FP64 units of the card run at
  1/64 of FP32, so the GPU gains only from K ~ 1000 or on batches of very
  small matrices.
- Eigen 3.4, one thread, measured alone: K = 20 / 50 / 200 / 1000,
  Cholesky 0.001 / 0.007 / 0.16 / 14.7 ms, pivoted QR (factorization only)
  0.004 / 0.023 / 0.85 / 80, SVD (BDCSVD) 0.085 / 0.70 / 11.0 / 532, null
  space of 3 constraints (JacobiSVD of 3 x K) 0.002 / 0.011 / 0.105 / 1.7,
  symmetric eigendecomposition 0.015 / 0.125 / 5.7 / 600, generalized
  0.017 / 0.155 / 6.2 / 626.

Chosen: Eigen on the host behind an interface of Laplace on Kokkos Views,
so that Kokkos Kernels, cuSOLVER or code of our own can be put behind the
same functions when a measurement asks for it, without changing the callers.

**Methods.** Cholesky: `LLT` (lower triangle). Triangular solve: Eigen's
`triangularView` solves in place, after a check for a zero on the diagonal.
Pivoted QR: `ColPivHouseholderQR`, with Q formed by applying the Householder
reflections to the first columns of the identity. SVD: `BDCSVD`, which
switches to one-sided Jacobi below 16 columns. Symmetric eigenproblem:
`SelfAdjointEigenSolver` (tridiagonal reduction and QL). Generalized
problem A v = w B v: with B = L L^T, the symmetric eigenproblem
C Y = Y diag(w) of C = L^-1 A L^-T, then V = L^-T Y, so that V^T B V = I.
This is the method of Eigen's `GeneralizedSelfAdjointEigenSolver` and of
LAPACK's `dsygvd`, written out because Eigen's solver does not check the
factorization of B. Null space: `JacobiSVD` of C with the full V. Rank: the
number of s_i above max(m, n) eps s_1 (the rule of NumPy's `matrix_rank`
and MATLAB's `rank`), or of |R_kk| above the same factor times |R_11| for
the QR; the caller may give its own relative tolerance.

**Accuracy** (`benchmarks/dense.cpp` on the 17 cases of
`tools/reference/dense.py`: matrices built at 200 bits from chosen spectra
and random orthogonal factors, then rounded to double; the references are
the eigenvalues or singular values of the rounded matrix, from mpmath at 200
bits). Errors in units of eps, the larger of gcc/OpenMP and of the CUDA
build (where the backend is compiled by MSVC); every residual and loss of
orthonormality is a Frobenius norm, relative to the norms given in
`benchmarks/dense_cases.hpp`:

| kind (cases) | values | residual | orthonormality | other |
|---|---|---|---|---|
| symmetric, n = 5..60 (6) | 12.2 | 9.92 | 72.2 (n = 60) | Cholesky L L^T - A 0.62, op(T) X - B 0.44 |
| generalized, n = 5, 20, 40 (3) | 1270 (below) | 0.19 | 0.10 | |
| rectangular, 1 x 20 .. 40 x 25 (8) | 7.93 | 14.4 | 44.5 | QR residual 1.56, Q^T Q - I 14.1; null space C T 1.74, T^T T - I 13.1 |

Symmetric and SVD values: max |w_i - w_i,ref| / (eps max |w_ref|). The
ranks of the SVD and of the QR agree with the expected ones on the eight
rectangular cases (5, 5, 18, 10, 30, 1, 3, 2); three of them have exact
rank below min(m, n), and their trailing singular values after rounding are
1e-18 to 1e-17 of s_1, far below the threshold. A cluster of ten
eigenvalues within 9e-9 and a condition of 1e12 do not raise the errors.

**The generalized problem with an ill-conditioned B.** The scale
eps max |w| charges the method with the condition of the problem. In case 8
(A the second-difference penalty D^T D, of rank n - 2, cond(B) = 1e6) the
largest eigenvalue, 8.6e6, has the condition
kappa_i = (||A|| + |w_i| ||B||) ||v_i||^2 = 8.6e12 (v_i B-normalized), and
a method backward stable on (A, B) may lose eps kappa_i on it. The values
are therefore measured in units of eps kappa_i, eigenvalue by eigenvalue:
within 1.4 in cases 7 and 9 (cond(B) 1e2 and 1e3), and up to 1270 in case 8,
at the two zero eigenvalues of the penalty, whose error is 3.6e-10 against
eps kappa_i = 4e-14. The reduction with the Cholesky factor of B is not
backward stable when B is ill-conditioned: its error on an eigenvalue near
zero is of the order of eps ||A|| ||B^-1|| (here 3.5e-9). LAPACK's `dsygvd`
(through SciPy 1.17's `eigh(A, B)`), the same reduction, loses more on the
same matrices: 9057 eps kappa_i in case 8, 1.9 and 41 in cases 7 and 9, and
max |w - w_ref| / (eps max |w|) 15796 in case 8 against 5731 here. A method
without this loss (the generalized SVD of the factors of A and B, or a
reduction of B by its eigendecomposition with deflation of its small
eigenvalues) is not written; it is worth writing if a caller needs
eigenvalues near zero to better than eps ||A|| ||B^-1||.

**Speed** (`benchmarks/dense.cpp`, ms per call, one thread; A = G G^T / K + I
with G uniform on (-1, 1), a general G for the QR and the SVD, three random
rows for the null space), gcc / MSVC:

| K | Cholesky | triangular solve, K right sides | pivoted QR with Q | SVD with U, V | symmetric eigen | generalized | null space of 3 x K |
|---|---|---|---|---|---|---|---|
| 20 | 0.0008 / 0.0017 | 0.0012 / 0.0025 | 0.0063 / 0.010 | 0.055 / 0.077 | 0.011 / 0.016 | 0.016 / 0.023 | 0.0026 / 0.0046 |
| 50 | 0.0066 / 0.010 | 0.011 / 0.016 | 0.053 / 0.075 | 0.50 / 0.60 | 0.088 / 0.13 | 0.14 / 0.20 | 0.0059 / 0.011 |
| 200 | 0.16 / 0.23 | 0.52 / 0.61 | 2.18 / 2.45 | 10.4 / 14.8 | 5.97 / 6.56 | 6.53 / 9.35 | 0.070 / 0.11 |
| 1000 | 15.6 / 17.8 | 44.3 / 50.8 | 194 / 241 | 608 / 630 | 459 / 765 | 736 / 1203 | 3.3 / 4.9 |

On gcc the functions take 0.54 to 1.3 times the time of Eigen measured
alone above (a separate run), with two exceptions: the pivoted QR takes 1.6
to 2.6 times the factorization alone, because Q is formed, and the null
space at K = 1000 takes 1.9 times, because the full 1000 x 1000 V is copied
into T.

**Tests.** `tests/dense/test_dense.cpp`, on gcc and CUDA: the values and the
identities of every case with bounds 1.5 times the larger measured error of
the two builds, the ranks and the number of null-space columns exactly, the
default and an explicit rank tolerance, the empty set of constraints
(T = I), the failures reported in the status (A or B not positive definite,
a NaN in the lower triangle, a zero on a triangular diagonal; a NaN in the
upper triangle of a symmetric input is not read) and the exceptions for
wrong dimensions.

**Cases of the dense accuracy table.** For the symmetric kind, T is the Cholesky factor of a positive definite case (or its transpose for the upper triangle) and the right-hand sides B are the matrix A of that case, evaluated over all four combinations of triangle and operation. The six symmetric cases have dimensions n in {5, ..., 60}: the Cholesky factor and the triangular solve are evaluated on the four positive definite ones, and the others are an indefinite matrix, a matrix with condition number 1e12, a matrix with a cluster of ten eigenvalues within 9e-9 and a semidefinite matrix with four zero eigenvalues. The eight rectangular cases have shapes from 1 x 20 to 40 x 25 and expected ranks 5, 5, 18, 10, 30, 1, 3 and 2 (three of them below min(m, n), and the 30 x 30 case of full rank with condition 1e12), among them a single row of ones and three rows of integers whose third row is the sum of the other two.

## Toeplitz correlation matrices: toeplitz_pdmat

**What.** `toeplitz_pdmat` (`include/laplace/pdmat/toeplitz.hpp`): the p x p
Toeplitz correlation matrix M_ij = m_|i-j|, m_0 = 1, with q = p - 1
parameters, the partial autocorrelations phi_k = tanh(theta_(k-1)),
k = 1..p-1. Every real theta gives a positive definite M. Unit diagonal, so
`unit_diagonal_scaled_pdmat<toeplitz_pdmat>` is a Toeplitz correlation with
heterogeneous variances. Design decided with Giovanni on 2026-11-14: closed
log det and M^-1, L supplied, derivatives over the sub-masks of the tuple,
p = 1 allowed (q = 0).

**How.** c_k = sech(theta_(k-1)), v_k = prod_(j<=k) c_j^2 (the variance of the
prediction errors of order k).
- log det M = -2 sum_k (p - k) log cosh theta_(k-1); order K:
  -2 (p - 1 - a) tanh^(K-1)(theta_a) on a tuple of the one index a, 0 on a
  mixed tuple.
- M^-1 = A^T V^-1 A, A unit lower triangular with the Durbin-Levinson
  coefficients (A_(t,t-j) = -phi_(t,j)), V = diag(v); 1/v_t =
  prod_(a<t) cosh^2 theta_a with closed derivatives (2^(n-1) sinh or cosh of
  2 theta), never a division.
- M and L by the normalized lattice: the forward and backward prediction
  errors divided by v_k^(1/2) satisfy, per order, the orthogonal rotation
  E_(k-1)(t) = c_k E_k(t) + phi_k B_(k-1)(t-1),
  B_k(t) = c_k B_(k-1)(t-1) - phi_k E_k(t); L_ts is the coefficient of the
  standardized innovation eps_s = E_s(s) in x_t, and m_t = L_t0.
- Derivatives: every quantity keeps its derivatives along the 2^K sub-masks
  of the tuple, products by Leibniz over the subsets (3^K per product);
  masks holding an index >= t are zero at time or stage t and skipped. About
  2 p^2 products for M, 2 p^3/3 for L, p^3/6 for M^-1. Scratch
  64 (p^2 + 3p - 2) doubles (0 at p = 1).

**The route for M, measured.** The first version computed m by the
Durbin-Levinson recursion, m_k = phi_k v_(k-1) + sum_j phi_(k-1,j) m_(k-j),
and L = A^-1 V^(1/2). The derivative of m_n along phi_k is v_(k-1) times a
polynomial in phi (SymPy, p <= 6), but the recursion forms it as a
difference of terms of order one. At the steep point (theta_0 = 6,
1 - phi_1^2 = 2.5e-5, p = 6) its derivatives of M along theta_2..theta_4 lost
1.4e7 to 6.0e7 ulpN (up to 5466 ulp of max |M|), those of L up to 2.1e5
ulpN, and tr(M^-1 dM/dtheta_a) missed the closed d log det/dtheta_a by up to
1.2e-6 relative. In the lattice every small factor is a product of c's. The
first derivatives of m at that point in double, against mpmath (dual
numbers): Durbin-Levinson 7.5, 1.3e4, 2.8e5, 5.0e7, 4.1 ulpN along
theta_0..theta_4, the lattice 0.84, 0.25, 1.6, 3.3, 4.7. With the lattice in
the code the trace agrees with the closed form to 6.6e-9, 4.1e-11, 4.1e-12,
3.6e-14, 2.7e-15 (along theta_0..theta_4; the residue along theta_0 is the
trace's own conditioning, M^-1 at kappa(M) = 1.2e14). M^-1 by
A^T V^-1 A was accurate in both versions.

**Measurement.** References `pdmat_toeplitz.csv`
(`tools/reference/pdmat_toeplitz.py`, 18568 values): m from the definition
of the partial autocorrelation (the last coefficient of the solution of the
Yule-Walker system of order k, affine in m_k, two `mp.lu_solve`), not
from the recursion; M^-1, L, log det by `mp.inverse`, `mp.cholesky`,
`mp.det`; derivatives by tensor products of central differences at 640 bits,
steps 2^-60 and 2^-70, which agree to 1.3e-35 relative. Points: p in
{1, 2, 3, 4, 6}; a random theta (N(0, 0.6^2)), a spread one (+-1.5) and a
steep one (6, then -+2.5); kappa(M) at p = 2, 3, 4, 6: random 1.2, 1.3, 34,
16; spread 20, 410, 8.2e3, 3.1e6; steep 1.6e5, 2.7e7, 4.5e9, 1.2e14. Tuples:
all of orders 1 and 2, the pure ones and up to 6 mixed ones of orders 3-6.
Errors as for the other structures (ulpN; relative ulp of max(|ref|, 1) for
log det). Worst over the points, orders 0..6, gcc/OpenMP / CUDA:

| K | M | M^-1 | L | log det |
|---|---|---|---|---|
| 0 | 1.28 / 1.28 | 9.70 / 2.28 | 1.81 / 1.81 | 1.37 / 1.37 |
| 1 | 7.38 / 7.38 | 9.05 / 3.35 | 4.45 / 4.16 | 0.83 / 0.62 |
| 2 | 7.50 / 7.76 | 9.56 / 4.01 | 4.90 / 4.90 | 1.16 / 1.16 |
| 3 | 126 / 126 | 9.23 / 3.87 | 6.52 / 6.52 | 2.04 / 2.04 |
| 4 | 108 / 108 | 9.56 / 4.32 | 7.52 / 7.52 | 12.6 / 2.76 |
| 5 | 151 / 162 | 9.23 / 6.59 | 10.9 / 10.9 | 3.39 / 3.39 |
| 6 | 163 / 23.1 | 14.5 / 15.2 | 8.65 / 8.65 | 22.8 / 19.9 |

The M figures from order 3 on are at the random p = 4 point along mixed
tuples (1 1 2, 0 1 1 2, ...), where M's derivative is small next to L's
(column 0 of L carries L's absolute error, M's normwise measure divides by
column 0 only); in ulps of max |M| they are 35, 135, 284, 3418 (orders 3-6,
gcc),
the growth of the derivatives with K. The log det at orders 4 and 6 (gcc) is
-2 (p - 1 - a) tanh^(K-1)(theta_a) near a zero of the scalar function: order 4
at theta_0 = -0.68 (tanh^(3) vanishes at -0.658), order 6 at theta_3 = 0.40
(tanh^(5) vanishes at 0.421). On CUDA the same
pattern, with M^-1 lower below order 6 (2.3 to 6.6) and 15.2 at order 6, and M at order 6
23.1. Central differences of
order K - 1 agree with order K within 7.7e-4 of the tolerance (1e-6 of the
largest entries). The AR(1) twin (theta = (t, 0, ..., 0) against ar1_pdmat,
p = 2..6, t in {-3, -0.4, 0, 0.8, 2.5}) passes the test's tolerances, 1e-15 at
order 0 and 1e-13 at orders 1-6, relative to the largest entry.

ns per call on one gcc thread (`benchmarks/pdmat_toeplitz.cpp`, theta_k =
0.1 x + 0.05 k, x uniform in [-3, 3]); order 6 along the pure tuple of
theta_0 (no stage is skipped) and of the last parameter, after the merging of
equal sub-masks below:

| p | M, K = 0 | M, K = 6 (first / last) | M^-1, K = 0 | M^-1, K = 6 (first / last) | L, K = 0 | L, K = 6 (first / last) | log det, K = 0 |
|---|---|---|---|---|---|---|---|
| 3 | 73 | 1726 / 1637 | 113 | 1766 / 1764 | 82 | 2219 / 2149 | 49 |
| 6 | 175 | 3194 / 2806 | 361 | 3680 / 3378 | 255 | 8380 / 6087 | 114 |
| 10 | 375 | 6632 / 5000 | 737 | 8339 / 7492 | 824 | 32566 / 18363 | 222 |

On CUDA (RTX 4070 SUPER, one call per thread, 20000 calls; 5000 at p = 10),
ns per call at order 0 / order 6 along theta_0, with the merging of equal
sub-masks below: M 106 / 1995, 231 / 4016, 441 / 8370; M^-1 112 / 2175,
299 / 4447, 670 / 9979; L 126 / 2729, 341 / 11132, 1085 / 43989; log det
39.6, 82.2, 144 at order 0 and 49 at order 6 (p = 3, 6, 10).

L costs about 2 p^3/3 products against the p^3/6 of A^-1 V^(1/2): at p = 10,
order 6 along theta_0, 32 us on one gcc thread against about 0.06 ms for the
Durbin-Levinson L (64223 ns, measured in a run with other processes active),
the price of the accuracy above.

**Merging of equal sub-masks.** Equal indices of a tuple are interchangeable,
so a mask S has the derivative of its class (how many positions of each group
of equal indices it holds). Only the canonical mask of a class (the lowest
positions of each group) is computed, the others copy it, and Leibniz's rule
runs over the classes with the weights prod_g C(c_g, a_g) (c_g, a_g the sizes
of S and of the sub-class in group g): prod (m_v + 1) terms instead of 2^|S|,
28 instead of 729 at order 6 for a pure tuple. A tuple without repeated
indices keeps the plain sum over subsets, since the table of classes costs
more than it saves there. L at order 6, p = 10 / 20 / 50, one gcc thread
(`ns` per call; a run with no other build): pure tuple 391 us -> 32 us,
3.2 ms -> 0.25 ms, 49 ms -> 3.8 ms (12 times); two groups of three
(0 0 0 1 1 1) 386 -> 82 us, 49 -> 9.9 ms at p = 50 (4.7 times); six distinct
indices unchanged (329 -> 320 us, 48 -> 49 ms). At order 6 and p = 10 M
fell from 53 to 6 us and M^-1 from 35 to 6.4 us along the pure tuple. The errors
did not worsen beyond the table above (M^-1 at order 6 went from 10.0 to
14.5 on gcc and from 13.9 to 15.2 on CUDA; L at order 6 from 10.8 / 9.66 to
8.65 / 8.65): the sums run in another order and with integer weights.

**Chosen.** The lattice for M and L, A^T V^-1 A for M^-1; tested in
`tests/pdmat/test_toeplitz.cpp` on gcc and CUDA, bounds 1.5 times the larger
measured error.

**Test bounds.** The bounds of `tests/pdmat/test_toeplitz.cpp` are 1.5 times the larger of the measured errors on gcc/OpenMP and CUDA, rounded up; for orders 0 to 6 they are 2, 12, 12, 190, 163, 244, 244 ulpN for M; 15, 14, 15, 14, 15, 14, 23 ulpN for M^-1; 3, 7, 8, 10, 12, 17, 13 ulpN for L; and 3, 2, 2, 4, 19, 6, 35 ulp for log det M.

## Factor-analytic matrices: factor_analytic_pdmat

**What.** `factor_analytic_pdmat` (`include/laplace/pdmat/factor_analytic.hpp`):
M = Lambda Lambda^T + Psi, Lambda p x d lower trapezoidal with
Lambda_jj = e^(theta) on its diagonal and the other loadings free,
Psi = diag(e^(2 t_i)). theta = the loadings by rows, then t_0, ..., t_(p-1);
q = pd - d(d-1)/2 + p. Precondition 0 <= d <= p; at d = p Psi is not
identifiable (Ledermann's bound (p - d)^2 >= p + d, whose largest d is given
by the host function `factor_analytic_max_factors`). Decided with Giovanni on
2026-11-14.

**Checks before the code.** The convention of glmmTMB 1.1.14's `rr`
(vignette covstruct and a fit): the same zero pattern, but the diagonal
loadings are raw parameters of either sign, placed first, and the other
loadings follow by columns; `rr` has no Psi. The derivatives of log det M and
M^-1 by Woodbury, differentiated over the sub-masks of the tuple (below),
agree with mpmath's numerical derivatives of the definition to 1e-59 at 60
digits, orders 0 to 6, (p, d) in {(3, 1), (3, 3), (4, 2), (4, 3), (5, 2)}.

**How.** M and its derivatives in closed form: a tuple of t_i alone reaches
M_ii (2^K e^(2 t_i)); loadings reach M only along at most two variables of
one column; a tuple that mixes a t with a loading is zero. The values by
Woodbury with C = I + Lambda^T Psi^-1 Lambda (d x d) and C = R R^T:
log det M = 2 sum t_i + log det C, M^-1 = Psi^-1 - Z^T Z with
Z = R^-1 Lambda^T Psi^-1, O(p d^2 + d^3); at d = p M^-1 comes from a
Cholesky factorization of M instead, because Psi^-1 and Z^T Z cancel when
every row is explained by the factors (5253 ulpN at small psi, 26 by
Cholesky in a NumPy transcription). The derivatives of log det M at d < p by Woodbury: C[T] (the
derivative along the positions in the mask T) is zero unless T lies in one
row of M, X[S] = -X[0] sum_T C[T] X[S \ T] with X = C^-1, and
d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)]),
f the first position.

The derivatives of M^-1 (every d >= 1) and those of log det M at d = p come
from the recursion on the augmented (p + d) x (p + d) matrix
A = [[Psi, Lambda], [Lambda^T, -I_d]], whose inverse is
G = A^-1 = [[M^-1, U], [U^T, -C^-1]] with U = Psi^-1 Lambda C^-1 = M^-1 Lambda,
and log|det A| = log det M:
G[S] = -G[0] sum_T A[T] G[S \ T], d^F M^-1 is the top-left p x p block of
G[F], and d^F log det M = sum_(T subset of F \ f) tr(G[T] A[f + (F \ f \ T)]).
A is linear in each loading, so A[T] is nonzero only along one repeated
parameter: 2^k psi_i at (i, i) for t_i, the derivative of the loading (r, c)
at (r, p + c) and (p + c, r). Each A[T] G[R] touches one or two rows, and
G[S] is one product of G[0] with those rows. G[0] is assembled from its
blocks: M^-1 as the value above, U and C^-1 as products, so that neither
M^-1 Lambda nor C^-1 is formed by a subtraction.

Three routes to the derivatives of M^-1 were measured in double (a
transcription in Python with NumPy) on 2026-11-14 against the references
below (worst ulpN over orders 1-6; "spread" is the point with psi_i from
e^-4 to e^2 and loadings +-1.5):

| (p, d), point | Woodbury | factor of M | recursion on M |
|---|---|---|---|
| (1, 1) spread | 2.5e5 | 18 | 17 |
| (2, 1) spread | 6.1e6 | 116 | 325 |
| (3, 2) spread | 1.1e6 | 401 | 410 |
| (3, 3) spread | 7.3e7 | 137 | 522 |
| (5, 2) spread | 1.8e5 | 6714 | 12866 |
| (3, 3) random | 2292 | 11647 | 1903 |

Woodbury's derivatives of M^-1 cancel, Psi^-1 against B C^-1 B^T
(B = Psi^-1 Lambda), when psi_i is small against the loadings of its row:
both terms are of order 1/psi_i while d/dt_i M^-1 = -2 psi_i (M^-1 e_i)(M^-1 e_i)^T
is small, and each order multiplies the Psi^-1 term by -2. The factor route
(the Cholesky factor of M with its derivatives over the sub-masks) and the
recursion on M (G[S] = -G[0] sum_T M[T] G[S \ T], G[0] from a Cholesky
factorization of M) were of the same class there; the factor route cost 3 to
6 times as much, and the recursion on M was chosen (Giovanni, 2026-11-14).

**Small specific variances (2026-11-16).** The recursion on M lost up to
3.1e6 ulp1N at psi = e^-6 inside `mvgaussian1`, where the problem is well
conditioned. New references `tools/reference/pdmat_factor_analytic_small_psi.py`
(the same method; 98373 values, steps that agree to 2.5e-34): for (p, d) in
{(3, 1), (4, 1), (4, 2), (5, 2), (3, 3), (4, 4)}, the spread loadings with
every t_i = -1, -2, -3, -4 (psi from e^-2 to e^-8), the spread loadings with
t_i = -3 and 1 alternating, random loadings N(0, 0.5^2) with every t_i = -3,
and the point (4, 1), Lambda = (e^0.4, 1.5, -1.5, 1.5), t = (-3, -3, 1, -3).
Conditioning in theta (mpmath, one relative ulp on every theta_k, summed): at
most 500 ulpN for the derivatives of M^-1 at most of these points; 2.0e3 and
3.4e3 at order 6 at two (4, 2) points, and 1.6e3 to 9.1e3 at the points with
alternating t, inflections like the one below.
Worst ulpN over orders 1-6 on these 37 points (gcc), by route:

| route | small psi | the 12 earlier points |
|---|---|---|
| recursion on M | 3.0e8 (2.2e4 at order 1) | 2.4e5 |
| the same with G[0] exact (rounded from mpmath) | 7.2e8 | |
| Woodbury | 3.3e12 | 1.2e8 |
| Woodbury or the recursion on M, chosen by estimates of their losses | 7.3e5 | 3.3e4 |
| recursion on A | 1.8e4 | 2.1e4 |

The recursion on M loses in its products, not in G[0]: with M close to rank
d, G[0] has entries of order 1/psi, and G[0] M[T] with M[T] along a loading
forms M^-1 lambda_c, of order 1, as a difference of terms of order 1/psi.
Woodbury cancels where the rows with small psi are fully explained by the
factors (d = p, or d small-psi rows at d < p). A third recursion, on M with
M^-1 Lambda and Lambda^T M^-1 Lambda carried as recursions of their own, lost
up to 1e13: the cancellation moved into the products with the columns of
G[0]. The recursion on A loses neither way. With its block M^-1 by Woodbury
at d = p it lost up to 2.4e5 ulpN; with Cholesky there, 3.6e3 (Python).

**Measured** (2026-11-16, both reference files, 49 points). By order, worst
gcc / CUDA (RTX 4070 SUPER), K = 0..6; M and M^-1 in ulpN, log det in ulp:
M 1.01 / 1.01 at K = 0, then within 0.49; M^-1 178 / 374, 1436 / 747,
1.7e4 / 1.7e4, 5465 / 2651, 1.9e4 / 1.8e4, 5538 / 3697, 2.1e4 / 2.3e4;
log det for d < p 3.5 / 3.5, 4.5 / 3.5, 11 / 5.9, 23 / 33, 66 / 88,
221 / 359, 795 / 926; log det at d = p 35 / 32, 836 / 98, 461 / 88,
680 / 188, 1464 / 269, 4141 / 794, 4091 / 492. Before the recursion on A,
M^-1 reached 3.0e8 / 5.6e8 and log det at d = p 1.7e6 / 4.8e5 on these
points. The orders 2, 4, 6 of M^-1 peak at the (3, 1) spread point (below);
the other peaks of M^-1 and of log det at d = p are at (3, 3) and (4, 4) with
psi from e^-6 to e^-8, where the conditioning is below 500.

**What it says.** The 2.1e4 of M^-1 at the (3, 1) spread point (orders 2, 4,
6, along t_2) is the conditioning of that derivative. With
A2 = M - psi_2 e_2 e_2^T, u = A2^-1 e_2 and a_22 = (A2^-1)_22,
M^-1 = A2^-1 - u u^T / (e^(-2 t_2) + a_22), and at that point
e^(-2 t_2) = 54.598 is within 2.5e-4 (relative) of a_22 = 54.584: an
inflection of 1/(e^(-2 t_2) + a_22), where the even derivatives nearly
vanish. Their largest entries are 0.0069, 0.055 and 0.93 against 27, 55 and
437 at orders 1, 3 and 5. mpmath: a relative change of one ulp in one theta_k
moves the exact derivative by up to 1.6e4 ulpN (4.3e4 summed over k), and one
ulp on every entry of M by 1.9e6 ulpN, the floor of every route that forms M.
The recursion on M lost 2.4e5 there.

log det is well conditioned at every earlier reference point: a relative
change of one ulp in every theta_k moves the exact derivative by at most
96 ulp in total (mpmath). Woodbury's derivatives of log det cancel at d = p,
where C is close to Lambda^T Psi^-1 Lambda: the traces along t_r are of order
2^K and the exact derivative of order psi_r (7.3e5 ulp at the (3, 3) spread
point on gcc, 1.0e6 on CUDA). From 2026-11-15 to 2026-11-16 the derivatives
at d = p came from the same traces on M with the recursion on M, 820 ulp at
the earlier points and 1.7e6 at psi = e^-6. At d < p the traces on A were
measured too (91 ulp at small psi and 67 at the earlier points against 457 and
795 by Woodbury) and not chosen: they cost 2 to 5 times as much at order 6
for p >= 8 (Giovanni, 2026-11-16).

ns per call (`benchmarks/pdmat_factor_analytic.cpp`, gcc, theta_k =
0.1 x + 0.05 sin k, x uniform in [-3, 3]), one thread / OpenMP on 8 threads;
order 6 along K distinct parameters spread over the rows ("pure": along
theta_0 alone); "recursion on M" is the route of 2026-11-14, kept in the
bench:

| (p, d) | M^-1, K = 0 | M^-1, K = 6 | M^-1, K = 6 pure | recursion on M, K = 6 / pure | log det, K = 0 | log det, K = 6 |
|---|---|---|---|---|---|---|
| (4, 2) | 414 / 56 | 5551 / 906 | 8450 / 1547 | 6506 / 13204 | 388 / 65 | 2927 / 581 |
| (8, 2) | 806 / 125 | 12280 / 1914 | 12031 / 2243 | 22409 / 44558 | 746 / 112 | 3182 / 582 |
| (16, 2) | 1651 / 303 | 26262 / 6829 | 23222 / 6027 | 114697 / 191916 | 1587 / 310 | 4099 / 778 |
| (16, 4) | 2766 / 488 | 31762 / 6615 | 28267 / 7324 | 117467 / 200021 | 2372 / 370 | 8714 / 1632 |
| (4, 4) | 618 / 106 | 8801 / 1479 | 9999 / 1893 | 5687 / 13804 | 545 / 89 | 4991 / 1544 |
| (8, 8) | | | | | 1674 / 309 | 17472 / 3478 |

(the recursion-on-M column in ns on one thread). The recursion on A is 1.8 to
8 times faster than the recursion on M at order 6 for p >= 8, and up to 1.5
times slower at (4, 4) on mixed tuples; log det at (8, 8), order 6 along one
parameter, 14973 ns.

**Chosen.** Woodbury for the values (Cholesky for M^-1 at d = p) and, at
d < p, for the derivatives of log det; the recursion on the augmented matrix
for the derivatives of M^-1 and, through its traces, for those of log det at
d = p (Giovanni, 2026-11-16). Tested in `tests/pdmat/test_factor_analytic.cpp`
on gcc and CUDA against both reference files, bounds 1.5 times the larger
measured error, log det at d = p in a row of its own; twins: d = 0 is
`diagonal_pdmat` bit for bit, at d = p M - Psi is `log_cholesky`'s M.

**Test bounds.** The bounds of `tests/pdmat/test_factor_analytic.cpp` are 1.5 times the larger of the measured errors on gcc/OpenMP and CUDA, rounded up, for orders 0 to 6: for M, 2, 1, 1, 1, 1, 1, 1 ulpN; for M^-1, 561, 2155, 25214, 8198, 27872, 8307, 35027 ulpN; for log det M with d < p, 6, 7, 17, 50, 133, 539, 1390 ulp; for log det M at d = p, 53, 1254, 692, 1020, 2197, 6211, 6137 ulp. The difference check against the derivative of order K - 1 uses tolerance 1e-6 relative to the sum of the scales of orders K and K - 1.

## Reduced-rank matrices: reduced_rank_psdmat

**What.** `reduced_rank_psdmat` (`include/laplace/pdmat/reduced_rank.hpp`), the
first positive semidefinite structure (`psdmat_structure` in
`include/laplace/pdmat/core.hpp`): M = Lambda Lambda^T of rank d, Lambda
p x d with the convention of `factor_analytic_pdmat` (lower trapezoidal,
Lambda_jj = e^(theta) on its diagonal, the other loadings free, theta the
loadings by rows; q = pd - d(d-1)/2; 1 <= d <= p). It supplies M, the factor
Lambda and log|M|_+, the log of the product of the d nonzero eigenvalues of M,
each with its derivatives of orders 1 to 6, and the rank d. Decided with
Giovanni on 2026-11-14.

**How.** M and Lambda with closed sparse derivatives: Lambda moves only along
one repeated loading (1 at order 1 for a free loading, e^theta at every order
for a diagonal one), and M as in `factor_analytic_pdmat` without Psi. The
nonzero eigenvalues of Lambda Lambda^T are those of C = Lambda^T Lambda (d x d),
so log|M|_+ = log det C. At d = p, Lambda is square and triangular and
log|M|_+ = 2 sum_j theta_jj exactly (derivative 2 along a diagonal loading at
order 1, else 0). For d < p the value comes from a Cholesky factorization of
C, and the derivatives from the recursion of `factor_analytic_pdmat` on C:
C[T] is zero unless every index of the mask T is a loading of one row r, and
then it is the Leibniz sum over those loadings of lambda_r lambda_r^T;
X[S] = -X[0] sum_T C[T] X[S \ T] with X = C^-1, and
d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)]). The
helpers (the loadings and, for the value, the d x d Cholesky) are shared with
`factor_analytic_pdmat` in `include/laplace/pdmat/detail/loadings.hpp`.
Since 2026-11-16 the derivatives of log|M|_+ for d < p run in double-double
(`include/laplace/math/detail/double_double.hpp`) from C onward: C, C[T],
the Cholesky factor of C and C^-1, X[T] and the traces, with Lambda in double
and the result rounded to double at the end. The value (order 0) keeps the
Cholesky factorization of C in double.

**Measured** (2026-11-14; references `tools/reference/pdmat_reduced_rank.py`,
mpmath at 640 bits, log|M|_+ as the sum of the logs of the d largest
eigenvalues of M from `mp.eigsy`, independent of Lambda^T Lambda; mixed
central differences with steps 2^-60 and 2^-70 that agree to 4.3e-34; 47362
values; (p, d) in {(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (5, 2), (5, 4)}, a
random point, a spread one with loadings +-1.5 and diagonal loadings 0.4, and
two near-singular ones with free loadings +-1e-3 and +-1e-5).
M and Lambda in ulpN, log|M|_+ in ulp of max(|ref|, 1); by order, gcc/OpenMP /
CUDA (RTX 4070 SUPER), K = 0..6:

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| M | 0.63 / 0.55 | 0.43 / 0.43 | 0.49 / 0.38 | 0.38 / 0.38 | 0.38 / 0.38 | 0.38 / 0.38 | 0.38 / 0.38 |
| Lambda | 0.34 / 0.34 | 0.49 / 0.43 | 0.49 / 0.34 | 0.49 / 0.43 | 0.49 / 0.43 | 0.49 / 0.43 | 0.49 / 0.43 |
| log\|M\|_+ | 1.3 / 1.6 | 0.63 / 0.63 | 0.85 / 0.85 | 6.2 / 6.2 | 3.2 / 3.2 | 2.3 / 2.3 | 13 / 13 |
| log\|M\|_+, double (2026-11-14) | 0.97 / 1.6 | 9.6 / 21 | 131 / 146 | 742 / 219 | 161 / 1440 | 3850 / 2502 | 3682 / 2946 |

The log|M|_+ row is the double-double route of 2026-11-16 (test bounds 3, 1,
2, 10, 5, 4, 19 ulp); the row below it is the route in double that it
replaced. At d = p log|M|_+ is within 0.42 ulp at every order.

The loss of the route in double. Its peaks at orders 4 to 6 came from the
spread point of (p, d) = (5, 4) along one repeated diagonal loading
(Lambda_22). There the reference is well conditioned: one ulp of any theta_k
moves the order-5 derivative by 1.1 ulp and the order-6 one by 2.2 ulp
(mpmath, the same script). At order 6 the trace terms sum to 1.2e4 in
absolute value against a result of 0.006, and each stage after Lambda (C,
C[T], the recursion, the traces) loses digits on its own: a prototype in
mpmath with the stages after Lambda at 106 bits was within 0.004 ulp. A route
by Stirling numbers in s = e^theta was no better (9.4e3 ulp at order 6).
Repetition of a diagonal loading is not a usable criterion for switching
precision: without one the route in double lost up to 674 ulp at order 5
(CUDA).

Worst log|M|_+ over the 1697 reference rows with d < p, ulp of
max(|ref|, 1), by how much of the route runs in double-double (measurement
twin `benchmarks/pdmat_reduced_rank_dd.cpp`), gcc/OpenMP / CUDA, orders 1..6:

| route | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| double | 9.5 / 20.5 | 131 / 146 | 752 / 219 | 203 / 1440 | 3850 / 2502 | 3682 / 2946 |
| double-double, C^-1 in double | 8 / 14 | 83 / 62 | 52 / 119 | 67 / 159 | 134 / 339 | 297 / 698 |
| double-double, C^-1 in double-double | 1 / 1 | 1 / 1 | 6 / 6 | 4 / 4 | 4 / 4 | 18 / 18 |

C^-1 must be in double-double too: X[0] enters every term of the recursion.
The last route was chosen for every order (Giovanni, 2026-11-16).

A second route was measured and removed (Giovanni, 2026-11-14): for a tuple
within one row r, log det C = log det A_r + log g with
A_r = sum_(i != r) lambda_i lambda_i^T (free of row r) and
g = 1 + lambda_r^T A_r^-1 lambda_r, the derivatives of the scalar log g by the
sub-mask recursion (g y[f] = g[f]), with a fallback to the recursion on C when
A_r is near singular. Worst log|M|_+ over the points (gcc, ulp), orders 1..6:
row route 131, 131, 110, 316, 1029, 4583; recursion on C 131, 131, 742, 161,
3850, 3682. It was better at orders 3 and 5 and worse at order 6, up to twice
at several points: the cancellation of a repeated e^theta stays in the
derivatives of log g. The two near-singular reference points did not move the
worst errors of the recursion on C.

Speed (gcc, ns per call, one thread / OpenMP on 8 threads, log|M|_+ along K
distinct parameters spread over the rows; M and the double row measured on
2026-11-14, the double-double row on 2026-11-16):

| (p, d) | M | log\|M\|_+ K=0 | K=1 | K=3 | K=6 |
|---|---|---|---|---|---|
| (4, 1) | 112 / 23 | 119 / 24 | 240 / 40 | 369 / 74 | 1944 / 326 |
| (8, 2) | 553 / 74 | 480 / 66 | 877 / 131 | 1299 / 202 | 5361 / 747 |
| (16, 4) | 2308 / 314 | 1856 / 274 | 4129 / 563 | 6045 / 830 | 29242 / 3960 |
| (16, 4), double | | 2223 / 299 | 2339 / 348 | 3103 / 496 | 9686 / 2171 |

Double-double costs 1.0 to 1.7 times the route in double at (4, 1) and
(8, 2), and 2 to 3.4 times at (16, 4) (same run of the measurement twin). On
CUDA (ns per call, one thread of the Serial backend / CUDA) the order-6 call
at (16, 4) went from 9289 / 987 to 46055 / 2326, and at (8, 2) from
3404 / 155 to 5999 / 183; orders 0 to 3 and d = 1 cost 1.0 to 1.7 times.

## Weighted sums of penalties: weighted_sum_psdmat

**What.** `weighted_sum_psdmat` (`include/laplace/pdmat/weighted_sum.hpp`) is
a `psdmat_structure` that runs on the host only:
M(theta) = sum_j e^(theta_j) S_j, where the S_j are fixed p x p positive
semidefinite matrices (the penalties of smooth and tensor-product terms) and
q = m. It supplies M and its derivatives (e^(theta_j) S_j along a tuple of
one repeated index, zero otherwise), the log pseudo-determinant log|M|_+ (the
sum of the logs of the nonzero eigenvalues) with its derivatives of orders 1
to 6, the rank r and an orthonormal basis of the null space. The null space
of a sum of positive semidefinite matrices is the intersection of their null
spaces, so r and the basis do not depend on theta. Both come from
`laplace::dense::null_space_basis` applied to sum_j S_j / max|S_j| at
construction. Decided with Giovanni on 2026-11-14.

**Why not the eigenvalues of the sum.** When the lambda_j = e^(theta_j)
spread apart, an eigendecomposition of M in double resolves the eigenvalues
of the small terms only to lambda_max eps. The r-th eigenvalue then turns
negative (its log is NaN) or carries no correct digit, and the derivatives
tr(lambda_j M^+ S_j) lose one digit per order of magnitude of the spread.
statmodels7 recorded this failure on its additive penalties, with values
that were finite and wrong from a spread of about 1e13.

**How.** A graded similarity transform after Wood (2011, appendix B), with
the components grouped by lambda_j ||S_j||, where ||S|| is the largest
absolute entry. Let U be the basis of the subspace not yet resolved (at
first I_p). Each level:

- drops the components whose restriction R_j = U^T S_j U is zero
  (max|R_j| <= 8 p eps max|S_j|);
- takes as dominant the components whose theta_j + log max|R_j| is within a
  factor split_tolerance of the largest;
- splits U by the eigendecomposition of sum_dom R_j / max|R_j| into its range
  U_+ (rank by the rule of laplace::dense, lambda > u eps lambda_max) and its
  null space U_0;
- appends U U_+ to Q, with the level's scale d_l = e^(top);
- continues on U U_0, where every dominant component is exactly zero.

These structural zeros are set, never computed: a zero computed in floating
point costs lambda_dom eps, which swamps the subordinate terms. Let D be the
diagonal matrix of the scales, G_j = D^-1/2 Q^T (lambda_j S_j) Q D^-1/2 with
the zeros set, and H = sum_j G_j, whose entries are of order one. Then

    log|M|_+ = log det H + sum_l r_l log d_l,

where r_l is the number of columns of level l and log det H comes from a
Cholesky factorization. B_j = H^-1 G_j is similar to lambda_j M^+ S_j on the
range of M, so d log|M|_+ / d theta_j = tr B_j and
d B_j / d theta_k = delta_jk B_j - B_k B_j. Along a sorted tuple of K
positions, the derivative is a sum over the set partitions pi of the
positions whose blocks each repeat one index: each partition contributes
(-1)^(|pi| - 1) times the sum, over the (|pi| - 1)! cyclic orders of its
blocks, of tr(B_(j_1) ... B_(j_|pi|)). Each word of indices is reduced to its
least rotation, and its trace is computed once. sum_j tr B_j = r is an exact
identity, a check that needs no reference. Every call recomputes the
transform. statmodels7's route takes a Schur complement at each level and
builds the next level's correction from the dominant reductions; this route
subtracts nothing, because H is formed directly from the scaled G_j.

**Double-double in the derivatives.** Along a tuple of orders 2 to 6 the
partition sum has alternating signs, so it loses digits that grow with the
order (1.7e4 ulp at order 6 on te2 with the B_j in double), and the entries of
a B_j in double are already wrong by eps. Measured by steps on the 3045
references (twin `benchmarks/pdmat_ws_dd.cpp`; worst error at order 6 in ulps of
max(|ref|, 1); gcc and the host code of the CUDA build agree to the digits shown):

| route | te2 | te3 | ridge | single | lowrank |
|---|---|---|---|---|---|
| B_j in double, as before | 1.73e4 | 1.03e4 | 2070 | 2050 | 1280 |
| (a) traces and weighted sum in double-double | 601 | 616 | 711 | 1.5 | 115 |
| (b) (a) and the products of the words in double-double | 6.4 | 4 | 4.5 | 1.5 | 9.2 |
| (c) (b) and B_j = H^-1 G_j in double-double | 1.5 | 2.5 | 0.6 | 2e-15 | 10.9 |

lowrank_ill stays at 2.2e4 in every route (conditioning). Step (a) removes the
loss from the signs but not the rounding of the products, (b) leaves one eps
per entry of B_j, and (c) removes it. What is left on lowrank, about 10 ulp,
is the rounding of G_j in the transform, which is still in double. Chosen
(Giovanni, 2026-11-17): (c). `log_pdet_state` and `log_pdet_dk` build B_j from
the G_j of the transform (H = sum_j G_j, its Cholesky factor and two triangular
solves, all in double-double); the products of the words, their traces and the
weighted sum are in double-double too. `log_pdet` (order 0) and `rank_identity`
do not build them. The double-double matrices sit in the scratch buffer after
the double ones, which adds 2 (m + 4) r^2 doubles to `scratch_size()`.

The words are visited in lexicographic order, and the product of the first k
matrices of a word is kept for the next word when it shares that prefix (the
last factor only enters the trace): the results are bit for bit those without
the sharing, and the derivatives from the state at order 6 take 2.7 to 2.9
times less (te3 20.6 ms to 7.0 ms, te2 4.7 ms to 1.7 ms, B_j already built).
A memo of the traces across tuples was measured and not adopted: on te3 the 83
tuples of orders 1 to 6 hold 671 words and 225 distinct ones, but the memo
saves only another 1.15 times (77 ms against 88 ms) and would make the state
an object with a map.

**References** (`tools/reference/pdmat_weighted_sum.py`). Every S_j is exact
in double (integers, or integers times powers of two), so its null space is
exact. SymPy gives an exact rational basis Z of the range. Then
log|M|_+ = log det(Z^T M Z) - log det(Z^T Z), and the derivatives follow from
the partition formula applied to (Z^T M Z)^-1 (lambda_j Z^T S_j Z), at 256
bits, without the levels or the scaling. The partition formula agrees with
tensor central differences of log det(Z^T M Z) at 640 bits (steps 2^-60 and
2^-70) to 5.3e-61, on every sorted tuple of orders 1 to 6 at two points.
The cases:

- te2: P6 (x) I5 and I6 (x) P5, where P_k is the integrated squared second
  derivative of k cubic B-splines on integer knots, scaled to integers;
  p 30, r 26.
- te3: three second-difference penalties, 3 x 3 x 4; p 36, r 28.
- lowrank: three matrices A A^T, A with 3, 2 and 2 integer columns; p 8, r 7.
- lowrank_ill: the same, with columns scaled by 2^-4 to 2^-8, so that one
  component reaches condition 2^16.
- ridge: P10 + I10, full rank.
- single: P8 alone.

The points are theta_0 = s ln 10, theta_1 = 0, theta_2 = (s/2) ln 10, with
s in -20, -12, -6, 0, 3, 6, 9, 12, 15, 20, plus one random point. Every
sorted tuple of orders 0 to 6 is used: 3045 values in all.

**The split threshold, measured** (benchmarks/pdmat_weighted_sum.cpp, gcc, with
the B_j in double-double). The columns give the worst error over the cases in
ulps of max(|ref|, 1) over orders 0 to 6, the same on the two tensor cases,
and the worst |sum_j tr B_j - r|:

| split_tolerance | worst | worst on te2, te3 | rank identity |
|---|---|---|---|
| eps^0 (one component per level, ties grouped) | 3.0e4 | 1.4 | 1.5e-12 |
| eps^0.05 (chosen) | 2.2e4 | 2.5 | 6.1e-14 |
| eps^0.10 to eps^0.15 | 6.9e4 | 27 | 1.8e-13 |
| eps^0.20 to eps^0.25 | 2.9e5 | 141 | 1.2e-11 |
| eps^0.30 to eps^0.35 | 4.3e5 | 141 | 4.2e-11 |
| eps^0.40 | 3.4e9 | 1.1e6 | 1.2e-8 |
| eps^0.50 | 3.4e9 | 1.1e6 | 4.2e-8 |
| naive (eigenvalues of the sum) | NaN; 1e15 to 4e21 | NaN; 1e15 to 1e18 | not computed |

The best threshold is the reverse of statmodels7's (eps^0.35 there, and
eps^0.3 in mgcv's gam.reparam). A level that holds a spread leaves that
spread unscaled in H, while a deeper recursion costs nothing here, because
nothing is subtracted. With the B_j in double (measured on 2026-11-14) eps^0.05
was the best on the worst column; with the B_j in double-double eps^0 is better
on the tensor cases (1.4 against 2.5 ulp) and on lowrank (5 against 11), and
worse on lowrank_ill (3.0e4 against 2.2e4, conditioning) and on the rank
identity (1.5e-12 against 6.1e-14). Decided (Giovanni, 2026-11-17): eps^0.05,
about 0.165, is kept, because eps^0 gains at most about 6 ulp (on lowrank)
while it loses on lowrank_ill and on the rank identity. A level holds the
components within a factor of about 6 of the largest. The zero test gives
identical results from 1 to 1024 p eps; 8 p eps is kept.

**Accuracy by order** at eps^0.05, with the B_j in double-double (ulps of
max(|ref|, 1); gcc and the CUDA build, whose host code MSVC compiles, agree to
the digits shown):

| case | K=0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| te2 | 1.96 | 0 | 0.55 | 0.5 | 1 | 0.5 | 1.5 |
| te3 | 1.13 | 0 | 2 | 0.88 | 2 | 0.75 | 2.5 |
| lowrank | 0.71 | 4.5 | 10.9 | 9.2 | 10.9 | 9.2 | 10.9 |
| lowrank_ill | 1.31e4 | 1.09e4 | 2.18e4 | 2.18e4 | 2.18e4 | 2.18e4 | 2.18e4 |
| ridge | 5.95 | 0 | 0.004 | 0.004 | 0.31 | 0.25 | 0.63 |
| single | 16.2 | 0 | 2e-15 | 2e-15 | 2e-15 | 2e-15 | 2e-15 |

In single the derivatives of orders 2 to 6 are exactly 0 and the double-double
sum returns it to 2e-15. The order-0 errors of single (16.2) and ridge (5.95)
come from log det H by a Cholesky factor in double, which was not changed.
lowrank_ill loses about 1e4 at every order and every threshold, also at a
spread of 1, where the naive route loses as much (3.9e4): the loss comes
from the conditioning of the components themselves, not from the route.
By spread, on te2 (worst over orders 0 to 6, graded / naive): 1.8e-15 / NaN at
1e-20, 0.77 / 6.7e12 at 1e-12, 1.96 / 3.0e3 at 1, 0.77 / 4.3e5 at 1e6,
0.84 / 1.2e11 at 1e12, 1.05 / NaN at 1e20. The test bounds are these worst
errors times 1.5, with at least 1 ulp. Central differences of the order below
agree within 1e-6 at three points per case for every tuple.

Speed (gcc, microseconds per call, one thread, theta = (9.2, 0, 4.6); the
orders 1 to 6 include the construction of the B_j in double-double and the
derivative with the prefix sharing):

| case | p | r | m | value | K=1 | K=2 | K=6 |
|---|---|---|---|---|---|---|---|
| lowrank | 8 | 7 | 3 | 9.7 | 25.9 | 26.5 | 145 |
| ridge | 10 | 10 | 2 | 8.8 | 38.1 | 39.8 | 170 |
| single | 8 | 6 | 1 | 4.0 | 9.0 | 9.4 | 118 |
| te2 | 30 | 26 | 2 | 97 | 460 | 451 | 2014 |
| te3 | 36 | 28 | 3 | 213 | 936 | 869 | 6856 |

With the B_j in double the same calls took, on te3, 233, 235 and 718, and on
te2 112, 111 and 250. The construction of the B_j in double-double costs 0.65
ms on te3 and 0.35 ms on te2 per theta. Every `log_pdet_dk` call repeats the
transform, one restriction and one eigendecomposition per level, then
Q^T S_j Q, the Cholesky factor and the B_j, and builds the B_j in
double-double.

**One evaluated state per theta** (benchmarks/pdmat_ws_state.cpp, gcc, one
thread). `log_pdet_state(t, w)` runs the transform and the construction of the
B_j in double-double once and leaves them in the caller's scratch `w`;
`log_pdet_dk_at<K>(idx, w)` then reads any derivative from it without changing
it. No new memory beyond `scratch_size()` and the per-call functions are
unchanged. The values equal the per-call ones bit for bit (tested on every
reference point). Tables of all sorted tuples of orders 0 to K, microseconds,
for m penalties that split the second-difference penalty of p columns by rows:

| m, p | tuples (K<=6) | table K<=2, per call | from state | table K<=6, per call | from state |
|---|---|---|---|---|---|
| 2, 10 | 28 | 142 | 30 (x4.7) | 1692 | 991 (x1.7) |
| 2, 50 | 28 | 12804 | 2475 (x5.2) | 136359 | 71322 (x1.9) |
| 3, 20 | 84 | 2281 | 242 (x9.4) | 43531 | 23917 (x1.8) |
| 3, 50 | 84 | 30944 | 3343 (x9.3) | 708003 | 428503 (x1.7) |
| 4, 20 | 210 | 4563 | 324 (x14) | 163937 | 103436 (x1.6) |
| 4, 50 | 210 | 66547 | 4586 (x15) | 2815504 | 1842146 (x1.5) |

Up to order 2 the transform and the B_j are the whole cost, so a table costs
one state: the gain is the number of tuples, 5 to 15. Up to order 6 the
products of the words dominate, and the gain is 1.5 to 2. A full table of the
83 tuples of orders 1 to 6 on te3 costs 0.85 ms for the state and 88 ms for
the derivatives. Open: the rounding of G_j in the transform (about 10 ulp on
lowrank) and log det H by a Cholesky factor in double (16 ulp on single).

## Basis transforms: constraints, orthonormalization, Demmler-Reinsch

**What.** `basis_transform` (`include/laplace/basis/transform.hpp`) replaces
the K columns of a univariate basis b(x) by the q columns of b(x)' T, with T a
K x q matrix, and the coefficients by beta = T gamma. The basis keeps its
sparse rows; T acts on the coefficient side: rows and derivatives T' b^(k)(x),
Gram matrices T' G_kl T, penalties T' S T, gradients g_gamma = T' g_beta.
`make_basis_transform(K, C, G, S, r)` builds T from up to three inputs, always
in this order (the canonical order), so that every property holds at the end:

1. constraints C beta = 0 (C is m x K): T1 is an orthonormal basis of the null
   space of C, from the SVD of `laplace::dense::null_space_basis`, so C T1 = 0;
2. Gram matrix G: T2 = T1 L^-T with T1' G T1 = L L' (Cholesky), so T' G T = I;
   orthonormal columns of T1 keep this true after the constraint;
3. penalty S: T = T2 W with T2' S T2 = W diag(lambda) W' (symmetric
   eigendecomposition, increasing lambda), the Demmler-Reinsch basis when G is
   given, so T' S T = diag(lambda).

An empty View skips its step; S without G gives the eigenvectors of S in
Euclidean coordinates. `basis_transform_from_matrix(T)` takes a T built in
any other way and declares no property. T is built on the host and copied to
the default device; `expand_coefficients` and `reduce_gradient` run in the
execution space of the Views they are given, `congruence` (T' A T) on the
host. Decided with Giovanni on 2026-11-14 (T1-T7 in the diary).

**The null eigenvalues.** The caller declares r, the dimension of the null
space of S within the constraints (m for D^m without constraints, m - 1 with
the zero-integral constraint). The r smallest eigenvalues are set to exactly
0, and the build fails (status: step demmler_reinsch, ratio reported) if one
of them exceeds 1e-8 lambda_max. A threshold relative to lambda_max is not
safe: the smallest penalized eigenvalue lambda_(r+1)/lambda_max falls as
K^(-2m), from 1.1e-4 (K = 10, D^3) to 2.3e-13 = 1040 eps (K = 200, D^3),
so a threshold K eps lambda_max (200 eps there) keeps a margin of 5 only.

**Failures.** A singular G (data that leave some columns without support)
is an error, not a silent drop of directions: the status gives the failed
step (orthonormalization), the dense reason (not_positive_definite) and the
numerical rank of T1' G T1, the number of its eigenvalues above
q eps times the largest. Wrong shapes, K < 1 and r outside [0, q] throw
`std::invalid_argument`.

**The route, measured before the code** (cubic B-splines on [0, 1],
K = 10, 40, 100, 200, D^2 and D^3, G theoretical G_00 and empirical X'X/n
with n = 5K, with and without the zero-integral constraint, references with
mpmath at 200 bits). R1, the generalized eigenproblem; R2, the Cholesky of G
then the symmetric eigenproblem (the canonical order); R3, the exact null
space of S taken out first. All three are in the same class: max|T'GT - I|
up to 460 eps (K = 200, constrained), off-diagonal of T'ST within 2.3 eps
lambda_max for R1 and R2 and 14.8 for R3, eigenvalues within 370 eps
lambda_max. The null eigenvalues come within 0.53 eps lambda_max of 0 with R1
and R2 and exactly 0 with R3. R2 was chosen because the canonical order gives
it.

**How.** `tools/reference/basis_transform.py` writes
`tests/data/reference/basis_transform.csv`: for uniform clamped cubic
B-splines on [0, 1] it computes G = G_00 and S = G_mm in exact rationals
(Cox-de Boor on polynomials), the zero-integral constraint
C_j = (t_(j+4) - t_j)/4 and an exact basis Z of its null space, and the
eigenvalues of the pencil (Z'SZ, Z'GZ) with mpmath at 200 bits; values below
1e-55 lambda_max are written as 0. Eight cases: K = 10 and 40, D^2 and D^3,
with and without the constraint. `tests/basis/test_transform.cpp` checks the
eigenvalues (in eps lambda_max), max|T'GT - I| (in eps), the off-diagonal of
T'ST (in eps lambda_max) and max|C T| (in eps max|C|); the worst over the
eight cases:

| quantity | gcc/OpenMP | CUDA (RTX 4070 SUPER) | test bound |
|---|---|---|---|
| eigenvalues | 54.7 | 54.7 | 83 |
| T'GT - I | 43.0 | 45.0 | 68 |
| off-diagonal of T'ST | 5.2 | 6.7 | 10 |
| C T | 75.1 | 71.7 | 113 |

The worst case of each column is K = 40, D^3, constrained, except the
off-diagonal, whose worst is K = 40, D^3 without constraint on both builds.
The bounds are 1.5 times the larger of the two builds. With an empirical
G = X'X/n (K = 40, n = 200, D^2, constrained): T'GT - I 26.0 (gcc) and 24.0
(CUDA) eps against 39, off-diagonal 1.2 eps lambda_max against 2, C T 51.7
and 55.3 eps against 83. The test also checks the skipped steps (no input,
C only, G only, S only), the two failures (G from data on [0, 0.5] only; r = 3
declared for D^2), the coefficients and gradients on the device and on the
host, the congruence (T'ST diagonal within 64 eps lambda_max, T'GT within 68
eps of I) and a given T.

**Open.** r is declared by the caller; the exact null space (R3) can be added
on top to compute it. A basis type with rows T' b(x) can be added if a kernel
needs rows in gamma.

**Chosen.** As written.

## The cyclic B-spline basis

**What.** `cyclic_bspline_basis` and `cyclic_bspline_basis_owner`
(`include/laplace/basis/cyclic_bspline.hpp`): periodic B-splines of degree
d <= 10 with period omega and K >= d + 1 knots t_0 <= ... <= t_(K-1) in one
period [lower, upper), lower = t_0, upper = lower + omega, extended by
t_(j+mK) = t_j + m omega. Column j is b_j(x) = sum_m B_(j+mK)(x), B_i the
B-spline of degree d on t_i..t_(i+d+1). Built as an ordinary B-spline basis
on the K + 2d + 1 knots t_(-d)..t_(K+d) (dimension K + d, domain one period)
whose columns are folded modulo K: extended column e belongs to cyclic column
(e - d) mod K. A row has width d + 1 and value i belongs to column
(f + i) mod K. Any real x is in the domain: x in [lower, upper) is used as it
is, any other x is reduced by r = fmod(x - lower, omega) (exact), so the only
roundings are those of x - lower and lower + r. Integral from lower:
n c_j + int_lower^(lower + r) b_j with x - lower = n omega + r and
c_j = (t_(j+d+1) - t_j)/(d + 1); `integral_row` gives the d + 1 columns of
the row, `integral_column(j, x)` any column. The owner precomputes the K + d
totals over one period of the extended columns. Gram matrices: the extended
basis's Gram folded; `gram_bandwidth()` = K - 1. Decided with Giovanni on
2026-11-14 (C1-C10 in the diary).

**How.** `tools/reference/basis_cyclic_bspline.py`: exact rationals on the
bi-infinite knot sequence T_e = t_(e mod K) + floor(e/K) omega, Cox-de Boor
and the derivative recursion at the unreduced x, integrals and Gram matrices
by exact integration of the polynomial pieces; no reduction and no folding,
so the route is independent of the code's. 47 cases, d in {0, 1, 2, 3, 5,
10}: uniform on [0, 1), on [1000, 1001) and with period 365.25, the minimum
K = d + 1, random knots with period 365, interior multiplicity 2 and d + 1,
multiplicity 2 and d + 1 at lower. Points: 20 random in
[lower - 2 omega, lower + 3 omega], the knots of one period and one ulp on
each side, upper. Errors as for the B-spline basis (rows in ulps of the
row's largest value, integrals and totals of the largest total, Gram of the
largest entry); worst over gcc/OpenMP and CUDA (RTX 4070 SUPER):

| | d <= 5 | d = 10 | test bound |
|---|---|---|---|
| rows, orders 0..min(d, 6) | 8 | 11 | 12 and 17 |
| integrals | 6 | 6 | 9 |
| column totals | 3 | 3 | 9 |
| Gram matrices | 8 | 9 | 12 and 14 |

The test also checks: derivatives 0..d - mu continuous across lower, G_00
positive definite, `integral_row` bit for bit equal to `integral_column`,
expansions against the rows, adding one period adds the totals, NaN and Inf
give NaN, `operator_penalty_matrix` of D^2 equals G_22 and D^4 on a cubic throws.
Defect found and repaired in `bspline.hpp`: the integral anchor at lower ran
BSPLVB on interval d, empty when lower is a multiple knot with knots below
it (0/0, NaN integrals, reachable through bspline `from_knots` too); it now
uses the interval that contains lower.

**Speed** (`benchmarks/basis_cyclic_bspline.cpp`, cubic, K = 10, 40, 200,
n = 1e6, against `bspline_basis::uniform_extended` with the same knots on the
points reduced to [0, 1)), cost ratio cyclic / B-spline:

| points | one thread and OpenMP | CUDA |
|---|---|---|
| in one period, values and second derivative | 1.1-1.2 | 1.02-1.04 |
| over several periods ([-3, 4)) | 1.3-1.9 | 1.1-1.2 |
| integral rows | 1.06-1.9 | 1.1-1.3 |

Before the totals were precomputed the integral rows cost 1.1-2.1 (CPU) and
1.7-3.1 (CUDA). The first CUDA lines of the second point set (K = 10) jump to
2-7 ms for both bases from run to run: a timing artifact.

**Chosen.** As written.

## Tensor product bases

**What.** `tensor_product_basis<B_1, ..., B_D>` (`include/laplace/basis/tensor.hpp`)
is the tensor product of D >= 2 univariate bases of any types (B-spline,
cyclic B-spline, Fourier, Legendre, mixed), D fixed at compile time, with its
own concept `multivariate_basis`. Margin j has K_j columns and rows of width
w_j; the product has K = prod_j K_j columns
B_c(x) = b_(1,j_1)(x_1) ... b_(D,j_D)(x_D), c = sum_j s_j j_j, s_D = 1,
s_j = s_(j+1) K_(j+1): the last margin varies fastest, as
kron(b_1, ..., b_D). A row at x = (x_1, ..., x_D) is first[D], the first
column of each margin, and W = prod_j w_j values in the same order; value
(i_1, ..., i_D) belongs to column sum_j s_j ((f_j + i_j) mod K_j), which
`column()` computes (the modulus matters only for a cyclic margin). The
marginal rows are written to a caller's scratch of `scratch_size(k)` =
sum_j w_j(k_j) values, then their product to the output in place.
Partial derivatives and integrals take a runtime multi-index k[D]: k_j in
0..6 is the order in x_j, k_j = -1 the integral from lower_j to x_j. An
integrated margin contributes its K_j integrals (first column 0, width K_j),
because the integrals of the columns before a B-spline window are not zero;
hence row_width(k) = prod_j w_j(k_j) with w_j(-1) = K_j. The integral vector
of a B-spline margin is its window, `integral_column_total` before it and 0
after; of a cyclic margin, its column integrals; of Fourier and Legendre,
`integral_row`. `expansion_partial` contracts the marginal rows with beta
(flat in the column order), the last margin innermost. Outside its domain each
margin applies its own rule; a NaN x_j or a margin with the nan rule outside
its domain makes the whole row NaN. Host functions: `cross_gram(k, l)` =
kron_j G_j(k_j, l_j), dense K x K column-major; `kronecker_product` of dense
column-major factors; `tensor_operator_penalty_matrix(basis, terms)`, the penalty
int over the box of (L f)^2 for L = sum_a c_a d^(k^(a)), equal to
sum_(a,b) c_a c_b G_(k^(a) k^(b)) (a sum of squares of several operators,
such as thin plate, is a sum of calls; an order above a polynomial margin's
degree throws); `marginal_operator_penalty_matrix(basis, j, op)` =
G_0 kron ... kron S_j kron ... kron G_0, S_j the `operator_penalty_matrix` of op on
margin j; `kronecker_transform(T_1, ..., T_D)` (constraints, T' G_0 T = I and
Demmler-Reinsch carry over factor by factor, the result declares no
property; T is dense K x q); `compose_basis_transforms(a, b)` = T_a T_b;
`fill_tensor_basis_rows(b, k, x, first, values)`, the rows at n points in a
kernel. Decided with Giovanni on 2026-11-14 (P1-P10 in the diary).

**How.** `tools/reference/basis_tensor.py` writes
`tests/data/reference/basis_tensor.csv` (about 1 MB) from the definitions of
the margins, never from the product code: B-spline and cyclic margins in exact
rationals on the double knots the C++ constructors compute (Cox-de Boor),
Legendre from integer coefficients, Fourier with mpmath; products at 400 bits.
Four cases: B-spline(0, 1, K = 7, d = 3) x B-spline(0, 2, 5, 2), with the thin
plate penalty G_(2,0)(2,0) + 2 G_(1,1)(1,1) + G_(0,2)(0,2); B-spline(0, 1, 6, 3)
x cyclic(0, 1, 5, 2), x_2 over several periods; Legendre(0, 2, n = 3) x
Fourier(0, 1, H = 2); B-spline(0, 1, 5, 2) x cyclic(0, 1, 6, 3) x
Legendre(-1, 1, 2). 14 points per case drawn from the margins' knots, one ulp
on either side and random points; 5 to 10 multi-indices per case, integrals
included; Gram matrices for 2 to 5 pairs. Errors in eps times a scale: rows
of the row's largest reference value, expansions of sum_c |beta_c B_c|, Gram
and penalty of the largest entry (the unit is max(eps scale, denorm_min), for
the integrals one ulp above lower).

| case | rows | expansions | Gram | test bounds (rows, expansions, Gram) |
|---|---|---|---|---|
| B-spline x B-spline | 2.31 | 1.17 | 1.74 | 3.5, 1.8, 2.6 |
| B-spline x cyclic | 11.0 | 3.00 | 3.26 | 16.5, 4.5, 4.9 |
| Legendre x Fourier | 2.10 | 1.70 (CUDA 1.42) | 1.24 | 3.2, 2.6, 1.9 |
| B-spline x cyclic x Legendre | 4.01 | 1.14 (CUDA 1.35) | 2.67 | 6.0, 2.1, 4.0 |

gcc/OpenMP and CUDA (RTX 4070 SUPER) agree to the digits printed except where
noted. Thin plate penalty 0.88 eps (bound 1.3). The rows of the cyclic case
carry the cyclic margin's own error. Bounds are 1.5 times the worst of the two
builds. The test also checks: each row equal bit for bit to the product of
the margins' rows, also outside the domain; NaN propagation and the nan rule;
`expansion_partial` against the dot product with the scattered row (8 eps);
`marginal_operator_penalty_matrix` with D^2 equal to G_(2,0)(2,0) and G_(0,2)(0,2) bit
for bit; the Kronecker product of two Demmler-Reinsch transforms
(T' G T - I 7 eps, bound 256; off-diagonals of T' S_j T 2.2 eps of the
largest, bound 64); a zero-integral constraint on the product composed after
the margins' transforms (|C T| 7.1 eps max|C|, bound 256); the throws on
orders out of range and on an order above a margin's degree; and
`fill_tensor_basis_rows` on the device against the host (0 on gcc, 1.9 eps
on CUDA, bound 16).

**Speed** (`benchmarks/basis_tensor.cpp`, cubic B-spline margins on [0, 1],
n = 1e5 uniform points, ms per call, one thread / 16 OpenMP threads / CUDA;
run while other builds were using the machine):

| | D = 2, K_j = 20 | D = 3, K_j = 10 |
|---|---|---|
| sparse rows | 4.1 / 0.79 / 0.15 | 8.4 / 1.5 / 0.46 |
| the same rows scattered into a dense n x K matrix | 13.1 / 10.2 / 1.9 | not run |
| expansion on the fly | 4.5 / 0.54 / 0.093 | 10.5 / 1.1 / 0.16 |
| dot from stored sparse rows | 3.8 / 0.54 / 0.034 | 23.1 / 2.9 / 0.14 |
| dot from stored dense rows | 23.1 / 4.9 / 2.4 | not run |
| bytes per row, sparse and dense | 136 and 3200 | 524 and 8000 |

At D = 2 the margins called directly, with the order fixed at compile time,
are faster than `row_partial` by 1.08 (one thread), 1.12 (OpenMP) and 1.14
(CUDA); a compile-time multi-index was measured and gave nothing (1.00-1.02),
so the gap is the scratch at runtime offsets. At D = 3 on the CPU a dot from
stored sparse rows is slower than the expansion on the fly (the cost of
`column()` per value). The integral row with a cyclic second margin costs
1.10-1.29 times the one with a B-spline second margin.

**Open.** `kronecker_transform` builds T dense (512 MB at D = 3, K_j = 20);
applying it factored by mode products if a product that large is needed.
`fill_tensor_basis_rows` allocates an n x scratch_size(k) scratch View; team
scratch if that memory matters.

**Chosen.** As written.

## Polygamma differences at a shift, to order 5

**What.** `own::detail_ibd::dpsi_all(z, h, N, out, dDv)`
(`include/laplace/math/incomplete_beta_deriv.hpp`) returns
psi^(n)(z) - psi^(n)(z + h) for n = 0..N and D(z) - D(z + h) with
D(w) = log w - psi(w), without forming the two terms, and now takes N = 5
instead of N = 4. The sixth derivative of a log-gamma difference at an integer
shift is psi^(5)(y + r) - psi^(5)(r), so the negative binomial and the
beta-binomial masses need that order; the scalar `dpsi(n, z, h)` was already
written for any n.

**How.** The route is unchanged: the terms below z + M >= 15 are differenced
one by one and at Z = z + M the asymptotic expansion is differenced term by
term, each power Z^-p multiplied by E_p = 1 - (Z/(Z+h))^p, a sum of positive
terms. The batched version reads its asymptotic coefficients from a table, one
row per order n, whose entry k is B_2k (2k+1)(2k+2)...(2k+n-1) with B_2k the
Bernoulli numbers; the row for n = 5, B_2k (2k+1)(2k+2)(2k+3)(2k+4), was
computed with SymPy as exact rationals rounded once
(`tools/reference/psi_diff.py` documents the grid; the coefficients are in the
header). The four rows already in the file came out identical, which checks the
derivation. `zE` is one element longer (p runs to 20 + N) and `fact`, `asym`
and `rec` take the extra order.

**Measured.** `tools/reference/psi_diff.py` writes
`tests/data/reference/psi_diff.csv`, 900 rows at 200 bits from `mpmath.psi`
(the definition, independent of the route): z from 1e-3 to 1e8, h/z from 1e-12,
where the difference is about -h psi^(n+1)(z) and cancels against either term,
to 1e12, where it is the first term alone, and h also 1, 2, 5, 40, 1000, the
shift of a discrete mass. Worst error over every row and order, in ulps of the
difference:

| route | gcc | CUDA | worst argument |
|---|---|---|---|
| `dpsi_all` (batched) | 7.69 | 5.69 | z = 40, h = 4e-7, n = 5 |
| `dpsi` (scalar) | 6.69 | 5.86 | z = 40, h = 4e-7 (gcc), z = 1e5, h = 1e-7 (CUDA) |

**What this says.** The accuracy at order 5 is that of the orders already in
use, and the worst points are the cancelling ones, as expected. Nothing in the
existing callers changes: `incomplete_beta_deriv.hpp` and `student.hpp` still
ask for N <= 4 and get the same numbers.

**Chosen.** As above, bound 12 ulp in
`tests/math/test_beta_deriv.cpp` ("polygamma differences at a shift"), both
routes inside a kernel.

**Open.** Order 6 is not written (no family needs psi^(6) differences). The
functions live in a detail namespace of the incomplete beta, which three
distributions will include for them alone; a header of their own is worth it
only if that compile cost shows.

## The derivatives of log F of a discrete distribution by summing the mass

**What.** `log_cdf_dk_by_sum<K>(d, y)` and `log_ccdf_dk_by_sum<K>(d, y)`
(`include/laplace/distrib/generic_cdf.hpp`), the derivatives of order K = 1..6
in the parameters of log F(y) and log(1 - F(y)) for any discrete distribution,
with the dispatchers `distrib_log_cdf_dk<K>` and `distrib_log_ccdf_dk<K>`,
which take the distribution's own `log_cdf_dK` member when it has one. One
route serves the five discrete families whose cdf has no closed derivative
(binomial, negbin1, negbin2, betabinom, pig), in place of the mixed (a, x)
derivatives of the incomplete beta that two of them would need
(decided with Giovanni on 2026-11-15, `note/piano_distrib_discrete.md`).

**How.** With k = floor(y), S the tail summed, C a sorted multi-index of
parameters, T a sub-multiset of C and B_T the complete Bell polynomial over T
of the derivatives of a function (B_T = e^-g d_T e^g, B_empty = 1),

    d_T S = sum_i p(i) B_T(l(i)),    l = log p,

whose ratios to S are d_T S / S, from which the derivatives of log S follow by
the same relation solved for them (`log_tail_dk_from_ratios`: the term that
holds the lowest index of T leaves d_T log S alone). Summed that way the route
is unusable at high order: the ratios are powers of the score, of size
((y - mu)/mu)^K on the Poisson, while d_T log S is of the size of d_T l, so the
step from one to the other cancels, up to 4e14 ulp at order 6. The sum is
therefore centred on the first mass: with A = l(start) and m_i = l(i) - A,

    S = e^A W,  W = sum_i e^(m_i),  d_T log S = d_T A + d_T log W,

where d_T A is the mass's own derivative, exact and closed, and the
derivatives of m_i, which vanish at the start, carry only the variation of l
over the tail. The walk is that of `log_tail_by_sum` (the cheaper tail, each
term relative to the first), with the stopping bound taken on the largest of
the 2^K sums instead of the mass alone and asked for twice in a row. When the
side asked for is the other one, O = 1 - S, its ratios follow from
d_T O = -d_T S: d_T O / O = -(S/O) d_T S / S, with d_T S / S rebuilt from the
derivatives of log S (`ratios_from_log_tail_dk`) and S / O <= 1, since the side
summed is the one of smaller probability. The Bell machinery
(`log_density_partials`, `l_over`) is the information's, extended from four to
six positions in a mask.

**Measured.** Against the Poisson rows of
`tests/data/reference/distrib_log_cdf.csv` (mpmath, mu = 0.05 to 300, k from 0
to 5 mu + 10, both sides, orders 1 to 6), worst error in ulps of the component,
the same on gcc and on CUDA:

| order | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| by sum, ratios of S (first version) | 481 | 7.5e4 | 2.1e7 | 7.4e9 | 1.2e12 | 4.5e14 |
| by sum, centred | 476 | 502 | 856 | 7.58e3 | 2.41e5 | 5.34e5 |
| the Poisson's own route | 479 | 509 | 537 | 562 | 867 | 3.84e3 |

The worst points of orders 1 and 2 are at mu = 40, k = 210, where log F itself
is -1e-79 and the Poisson's own route scores the same 479 ulp; the loss at
orders 4 to 6 is at k near the median. Against the Poisson's own route at
k = mu, relative difference at order 6: 1.6e-11 at mu = 40, 8.0e-11 at 300,
9.9e-11 at 3e3, 1.1e-10 at 3e4, so the loss does not grow with the mean. Cost
on one gcc thread at order 6, mu = 300, k = median: 103 us per call against
3.9 us for the Poisson's closed route (one walk per component of the tensor,
with l's derivatives to order 6 and a 3^K Bell recursion at each point).

**What this says.** The route is exact to the last bits at orders 1 to 3 and
loses about five digits at orders 5 and 6 near the median, which is the
moment-to-cumulant cancellation of the centred variable and not something a
better summation removes; a family that needs more has to carry its own closed
route. Giovanni accepted it for the seven discrete families on 2026-11-15 and
asked that one family also get its own closed route, to measure the difference
in error and in time on the same references: the binomial, whose mean enters
its cdf only through x = 1 - mu, so that the x-derivatives of log I_x(a, b) are
elementary.

**Chosen.** As above. Tested in `tests/distrib/test_generic.cpp` against the
mpmath references with bounds half again above the measurement (715, 755,
1.3e3, 1.14e4, 3.62e5, 8.0e5 ulp by order), plus the ends of the support, the
dispatchers and both routes inside a kernel.

**The complement side, measured again on a two-point support (2026-11-15).** The
derivatives asked for on the side that is not summed are rebuilt from the other
one, and that step loses digits when the two sides differ widely in how fast
they move with the parameter. On the Poisson the loss is the 1e-10 of the table
above; on the Bernoulli at mu = 0.99 (support {0, 1}, so one term a side), the
generic route reproduces the closed derivatives of log F exactly, and those of
log(1 - F) = log mu at y = 0 to 1.3e-16 at order 1, 9.2e-13 at order 3 and
7.3e-7 at order 6. The quantity itself is the reason: with S = F = 1 - mu
summed, the derivatives of log(1 - S) are of size (K-1)!/mu^K while those of
S carry (K-1)!/(1 - mu)^K, larger by (mu/(1 - mu))^K, and in double precision
the difference cannot be recovered. The five families that will use the route
have many support points, where the two sides move alike and the loss is the
Poisson's; a family whose support has two or three points should carry its own
closed forms, as the Bernoulli does.

**Open.** One walk per component: a two-parameter family at order 6 pays seven
walks, which share the derivatives of l but not the Bell recursion, so fusing
them saves about a factor of two and is left until a profile asks for it. The
tail bound assumes a log-concave mass and |B_T| growing at most polynomially,
true of the five families that will use it.

## Optimizer foundations: condition estimate and line searches

**What.** Two pieces under the optimizers of phase 6. In `laplace::dense`,
`cholesky_solve(L, B)`, `spd_solve(A, L, B)` and
`cholesky_condition_estimate(A, L)`, which estimates the 1-norm condition
number kappa_1(A) = |A|_1 |A^-1|_1 of a symmetric positive definite A from its
Cholesky factor L. In `laplace::optimizers`, `armijo_line_search` and
`wolfe_line_search` (`include/laplace/optimizers/line_search.hpp`).

**How.** |A|_1 is computed exactly from the lower triangle. |A^-1|_1 is
Higham's estimate (Higham 1988, ACM TOMS 14, Algorithm 4.1, the iteration of
LAPACK's dlacn2): a power-like iteration on sign vectors with at most five
solves, then one solve with the alternating vector
x_i = (-1)^i (1 + i / (n - 1)), keeping the largest of the norms found. Every
norm it computes is |A^-1 x|_1 for some x with |x|_1 = 1, so the estimate
never exceeds |A^-1|_1 beyond rounding. Each solve is two triangular solves
with L, O(n^2).

Armijo: sufficient decrease phi(s) - phi(0) <= c1 s phi'(0), with
phi(s) = f(x + s d) and c1 = 1e-4, by backtracking: the minimizer of the
quadratic through phi(0), phi'(0) and phi(s) at the first backtrack, of the
cubic through the last two trial values afterwards (Dennis and Schnabel 1996,
A6.3.1), kept within [0.1 s, 0.5 s]. Strong Wolfe: also
|phi'(s)| <= c2 |phi'(0)| with c2 = 0.9, by bracketing and zoom (Nocedal and
Wright 2006, Algorithms 3.5 and 3.6). Extrapolation is the cubic minimizer
kept within [s + 1.1 (s - s_lo), s + 4 (s - s_lo)]; inside the bracket, when
the value rose at the far end, the cubic minimizer if it is closer to the
low end than the minimizer of the quadratic through phi(s_lo), phi'(s_lo)
and phi(s_hi), otherwise their mean (case 1 of More and Thuente 1994), and
the result is kept at least a tenth of the bracket from either end. A
non-finite value or slope is a refused trial point: the step is halved
towards the last finite point and never interpolated through. A trial point
equal to x in every coordinate ends the search as a failure (step below
rounding).

**Measured** (`benchmarks/optimizers_foundations.cpp`, gcc, one thread).
Estimate over exact kappa_1, A = Q diag(lambda) Q^T with lambda log-spaced
from 1 to kappa and Q the eigenvectors of a pseudo-random symmetric matrix,
exact kappa_1 from Q diag(1 / lambda) Q^T, five matrices per cell (min, max):

| n | kappa 1e2 | 1e6 | 1e10 | 1e13 |
|---|---|---|---|---|
| 5 | 0.726, 1.000 | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 |
| 20 | 0.736, 1.000 | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 |
| 100 | 0.703, 0.924 | 0.799, 1.000 | 0.672, 1.000 | 0.663, 0.977 |
| 300 | 0.806, 0.901 | 0.668, 0.946 | 0.647, 0.949 | 0.609, 0.956 |

All in [0.609, 1.0001]; the excess over 1 is rounding of the solves at
kappa 1e13. Time in ms per call (kappa 1e6):

| n | Cholesky | estimate | inverse by n solves |
|---|---|---|---|
| 10 | 0.00028 | 0.00073 | 0.00056 |
| 100 | 0.029 | 0.016 | 0.129 |
| 500 | 2.43 | 0.31 | 10.7 |
| 1000 | 16.0 | 1.21 | 82.4 |

Evaluations of the line searches from the standard starts, initial step 1
(Newton: the Cholesky direction where H(x0) is positive definite):

| problem | direction | Armijo | Wolfe |
|---|---|---|---|
| Rosenbrock (n = 2 and 100) | -g | 5 | 5 |
| Powell singular | -g | 6 | 4 |
| Beale | -g | 3 | 3 |
| Brown badly scaled | -g | 2 | 2 |
| helical valley | -g | 4 | 4 |
| Wood | -g | 10 | 6 |
| quadratic n = 10, kappa 1e2 / 1e6 / 1e12 | -g | 3 / 7 / 13 | 3 / 7 / 13 |
| every problem with H(x0) positive definite | Newton | 1 | 1 |

**What it says.** The estimate errs on the small side by at most a factor
1.65 here. It costs a thirty-fourth of the inverse at n = 500, an eighth of
the factorization at n = 500 and a thirteenth at n = 1000; at n = 10 it costs
about two and a half factorizations, 0.7 us. The step needed
on the kappa 1e12 quadratic along -g is 1e-12, and both searches reach it in
13 evaluations, the number that the 0.1 lower bound on each shrink allows.

Two defects were found and repaired while measuring. The first version of
the zoom used the cubic alone and bisected whenever the interpolated step
fell outside the safeguarded interval, instead of moving it to the nearest
end of the interval: 19 evaluations on the kappa 1e6 quadratic and a failure
after 30 on kappa 1e12 (now 7 and 13). The quadratic of case 1 took Wolfe
from 8 to 5 evaluations on Rosenbrock, 7 to 4 on Powell and 9 to 6 on Wood. And sufficient decrease written as
phi(s) <= phi(0) + c1 s phi'(0) accepted a step on a constant objective at
s = 1e-12, where c1 s phi'(0) is below the rounding of phi(0) and the sum
rounds to phi(0); the difference form refuses it, and the search ends as step
below rounding.

**Open.** The tolerances and constants (c1, c2, the shrink interval, the
budgets of 60 and 30 evaluations) are the textbook values and wait for the
methods of wave 1 to measure them.

## The three closed discrete distributions: bernoulli, binomial, geometric

**What.** `bernoulli_distrib` (mean mu in (0, 1), y in {0, 1}),
`binomial_distrib` (mean proportion mu with a known size n, y in {0, ..., n})
and `geometric_distrib` (mean mu > 0, y = 0, 1, ...), the first three of the
seven discrete families of phase 4
(`include/laplace/distrib/bernoulli.hpp`, `binomial.hpp`, `geometric.hpp`).
Each writes the log-mass and its six derivative orders, the expected
information and its four, the cdf family, and the quantile where it is closed;
the size of the binomial is a trailing field of the struct, not a parameter,
and `support_upper()` returns it. The derivatives of log F in the parameter are
the families' own except for the binomial, which takes the generic route by
summing the mass (the entry above).

**How, bernoulli.** l = y log mu + (1 - y) log1p(-mu); only the term of the
observed y is formed, so
d^K l / dmu^K = y (-1)^(K-1) (K-1)! / mu^K at y = 1 and
-(K-1)! / (1 - mu)^K at y = 0 (the second has no alternating sign: the first
reading of this formula in the plan gave the two terms one, which is right at
odd K only). I(mu) = 1/mu + 1/(1 - mu), its K-th derivative
(-1)^K K! / mu^(K+1) + K! / (1 - mu)^(K+1), the two terms summed at the end, so
the odd orders are exactly 0 at mu = 1/2. F is constant on each piece of the
support, so the derivatives of log F are those of log1p(-mu) and those of
log(1 - F) are those of log mu. The quantile is 0 at p <= 1 - mu and 1 above,
also from log p and from log(1 - p).

**How, binomial.** The log-mass in the saddle-point form (Loader 2000), as the
Poisson's:
l = (1/2) log(n / (2 pi y (n - y))) + delta(n) - delta(y) - delta(n - y)
- n mu D(y / (n mu)) - n (1 - mu) D((n - y) / (n (1 - mu))), with delta
Stirling's error (`math::stirling_error`) and D(1 + w) = (1 + w) log1p(w) - w
by its series below |w| = 1/4; the ends y = 0 and y = n are n log(1 - mu) and
n log mu. Both deviance arguments come from the one numerator y - n mu, formed
by a fused multiply-add, since the rounded product n mu moves the mass by tens
of ulp at n = 1e6 (measured: 41 ulp). The derivatives are
d^K l = (K-1)! [(-1)^(K-1) y / mu^K - (n - y) / (1 - mu)^K], with order 1
written (y - n mu) / (mu (1 - mu)) so that it stays accurate at the mode, and
I = n / (mu (1 - mu)) with its derivatives closed. With k = floor(y) and
0 <= k < n, F(k) = 1 - I_mu(k + 1, n - k) and 1 - F(k) = I_mu(k + 1, n - k):
both sides are taken at mu itself from `own::ibeta`, `ibetac`, `log_ibeta` and
`log_ibetac`, each side computed directly, and 1 - mu is never formed.

**How, geometric.** l = -y t - log1p(mu) with t = log1p(1/mu), both terms
negative. With S_K = sum_(j<K) C(K, j) mu^j and B_K = S_K / (mu (1 + mu))^K,
d^K l = (-1)^(K-1) (K-1)! [y S_K - mu^K] / (mu (1 + mu))^K, I = B_1 and
d^K I = (-1)^K K! B_(K+1): the identity (1 + mu)^K - mu^K = S_K removes the
subtraction of two nearly equal terms, and what is left is the genuine zero of
the derivative near y = mu / K. log(1 - F) = -n t with n = floor(y) + 1, whose
derivatives are (-1)^(K-1) (K-1)! n B_K. log F takes two regimes, because the
plain composition of g(u) = log(-expm1(u)) with u = -n t cancels by 1/mu^K at
small mu (measured: 1e24 ulp at mu = 1e-6, y = 0, order 5):
- F <= 1/2 (that is n t <= log 2): g^(k)(u) = -R_(k-1)(q) with q = 1/expm1(n t),
  R_0 = q and R_(j+1) = (q + q^2) R_j'(q), whose coefficients k! S(j+1, k) are
  all positive (S the Stirling numbers of the second kind), composed with the
  chain rule;
- F > 1/2: log F = log1p(-w) with w = r^n and r = mu / (1 + mu); the
  derivatives of w from the chain rule of x^n at r, where
  r^(m) = (-1)^(m-1) m! v^(m+1) with v = 1/(1 + mu), and those of log(1 - w)
  are -(k-1)! / F^k.
The quantile is closed: k + 1 >= target / t gives ceil - 1, corrected by at
most eight one-step moves on its own predicate, so it is O(1) and not a search
on the cdf.

**Measured.** References from the definitions with mpmath (bernoulli at 300 and
400 bits, kept when they agree to 2^-80 of the order's largest term, 360 rows;
binomial from SymPy's own derivatives at 300 bits with the cdf summed with
exact `math.comb` and cross-checked against mpmath's incomplete beta for
n <= 100, 7622 rows; geometric from SymPy's derivatives at 400 bits, 2640
rows), each script also checking every closed form against SymPy before writing
anything. Worst error in ulps, gcc and CUDA (one figure when they agree):

| quantity | bernoulli | binomial | geometric |
|---|---|---|---|
| log-mass | 0.5 | 7.7 / 6.7 | 1.3 |
| its derivatives, orders 1 to 6 | 1.0 to 4.9 | 1.6 to 5.6 | 1.8 to 10.2 |
| I and its derivatives to order 4 | 1.1 to 2.3 | 1.0 to 2.1 | 1.7 to 4.4 |
| log F | 0.5 to 4.9 | 4.7 | 63.5 / 82.5 (value), 2.6 to 60.3 |
| log(1 - F) | 0.5 to 2.6 | 4.3 | 1.1 to 7.1 |
| F, 1 - F | exact | 79, 124 / 461 | — |

The derivatives of the mass are scored in ulps of the larger of their two
terms, since near the mode they are small against both (the binomial's order 3
reaches 33 ulp of its own value there); the geometric's log F derivatives are
scored against the largest value of the order at the same mu, since they have
zeros in y. The binomial's F and 1 - F inherit the incomplete beta's error at
large parameters, and where log F or log(1 - F) is within 1/e of zero its
relative error reaches 9.0e4 and 2.5e4 ulp, which is |log w| ulp of a value
formed as exp(log w) (the absolute error stays below 1e-90). The geometric's
log F value is conditioned by n t, which reaches 100 on its grid.

Checked here on top of the agents' own measurements: the bernoulli reference
file regenerated byte for byte; the binomial's log-mass within 2 ulp and its
first derivative within 0.43 ulp of the exact formula at 300 bits at
(n = 1e6, mu = 0.3) near the mode, at (1e6, 1e-6) and at (1e4, 0.5); the
geometric's log F derivatives at mu = 1e-6, y = 50, where log F is -5.1e-299,
equal to a 2000-bit reference in every printed digit at orders 1, 5 and 6
(-5.0997348070279152e-299, -2.8187129482141245e-268,
-1.2966063657062807e-260); and the generic route against the bernoulli's closed
derivatives, which is the measurement of the complement side recorded above.

**What this says.** The three families that have closed forms everywhere are at
the accuracy of the elementary functions they use, a few ulp, with the three
expected soft spots documented: the odd derivatives of the bernoulli's
information at mu = 1/2 and the derivatives of the mass near the mode are
genuine zeros, scored on their terms; the binomial's cdf is as accurate as the
incomplete beta; the geometric's log F needs its two regimes, and the first
form, which looked natural, was unusable at small mu.

**Chosen.** As above. Tests `tests/distrib/test_bernoulli.cpp`,
`test_binomial.cpp` and `test_geometric.cpp`, bounds at 1.5 times the worse of
the two builds, every reference row evaluated inside a Kokkos kernel; the whole
suite is 351 cases and 227433 assertions on both builds.

**Open.** The geometric's derivatives overflow once t^-7 passes the double
range, that is above mu of about 1e44. The binomial's logarithms near zero
could be made exact inside `log_ibeta` and `log_ibetac`, which would serve
every family that takes a cdf from the incomplete beta. The manual pages of the
three families are not written, and their `\ingroup` names do not exist in
`groups.dox` yet.

## Frozen R references for the optimizers

**What.** Three sets of files produced in R once, so that the optimizers of
phase 6 and the fit_distrib bench can be compared with statmodels7 and optimizers7 without R at test
time (note/optimizers.md, decisions 8 and 10). Versions: R 4.6.0,
statmodels7 0.244.0, distributions7 0.94.2, optimizers7 0.22.0 (installed
from its source tree for these files; 0.21.0 gave the same counts and
estimates bit for bit, since 0.22 only compiles the stopping rules). Every script is run from the
repository root with `Rscript --vanilla`.

**The battery in R.** `tools/reference/optimizers_problems.R` writes the
problems of `tests/optimizers/test_problems.hpp` in R: extended Rosenbrock at
n = 2, 10, 100, Powell singular, Beale, Brown badly scaled, helical valley,
Wood, the quadratics at n = 10 and kappa = 1e2, 1e6, 1e12, the saddle and the
exploded cosh at n = 3, each with the added constants 0, 1, 1e3 and 1e6, and
the noisy objective at n = 2 (53 problems). The quadratics need the 64-bit
LCG of `test_uniform`; it is computed in 16-bit limbs held in doubles, and
its first values equal the C++ ones bit for bit (checked with `%a` against a
compiled copy). The script checks its gradients and Hessians against central
differences (relative error below 1e-4; Brown badly scaled reaches 1.3e-5
because f is near 1e12 at the check point, the others stay below 1e-9).

**optimizers7 counts.** `tools/reference/optimizers_optimizers7.R` runs
optimizers7's `newton()`, `bfgs()` and `lbfgs()` with their defaults on the
53 problems (159 runs, 2.3 s) and writes
`tests/data/reference/optimizers/optimizers7_counts.csv`: iterations, f, g, H
counts, final point and value at full precision, f - f*, max |g|,
max |x - x*|, the converged flag and the rule that fired. optimizers7
reports only the label of its composite rule (gradient max-norm < 1e-6, or
|df| < 1e-10, or |dx| < 1e-8); the script repeats each run with
maxit = iterations - 1 to get the previous iterate and evaluates the three
rules with `crit_met()`. The gradient check at the start is switched off
(`optimizers7.check_gradient = FALSE`) so that the counts are the method's.

Converged flags: newton 48 of 53, bfgs 44, lbfgs 47. Of the 139 runs
flagged converged, 42 were declared on a change rule alone, with the
gradient rule not met (newton 4, bfgs 12, lbfgs 26). Among them:

| problem | method | rule | f - f* | max \|g\| | max \|x - x*\| |
|---|---|---|---|---|---|
| quadratic kappa 1e12 (every constant) | lbfgs | \|dx\| < 1e-8 | 9374 | 2371 | 5.07 |
| quadratic kappa 1e6 | lbfgs | \|df\| < 1e-10 | 6.3e-10 | 2.1e-3 | 5.0e-6 |
| Brown badly scaled | bfgs, lbfgs | \|df\| < 1e-10 | 0 | 5.0e-4, 2.1e-3 | 3.5e-10, 1.6e-9 |
| Powell singular | bfgs, lbfgs | \|df\| < 1e-10 | 4.3e-11, 5.1e-11 | 1.7e-5, 7.8e-5 | 1.3e-3, 1.2e-3 |

The first row is a false success 5 units from the minimizer; the others
stop near it without meeting stationarity. All three methods also stop at
the saddle from (1, 0), where the gradient is zero at (0, 0) (f - f* = 0.25):
a stationary point, not a false success. The exploded cosh exhausts the
iteration budget (200 for newton, 500 for the quasi-Newton methods), and the
noisy objective ends not converged after one iteration with 31 or 32
evaluations of f. The added constant changes the counts in five runs only,
all on |df| < 1e-10 (an absolute change rule, met later or never when |f|
is larger): Powell bfgs and lbfgs 38 to 39 iterations at 1e6, Wood lbfgs
89 to 86, the kappa 1e2 quadratic lbfgs 38 to 39, and the kappa 1e6
quadratic lbfgs from converged to "stopped without converging" at 1e6.

**fit_distrib data and estimates.** `tools/reference/optimizers_fit_distrib.R`
simulates the 16 parametrizations of `include/laplace/distrib` (cauchy,
chisq, exponential, gamma1, gamma2, gaussian1-3, gumbel, invgauss, laplace1,
laplace2, logistic, lognormal1, lognormal2, poisson) at n = 1e3 and 1e5 with
seed 20261115 + 100 k + j (k the distrib's position in the script, j = 1, 2
the size; Mersenne-Twister, Inversion), fits each with statmodels7's
`fit_distrib()` at its defaults (Fisher scoring, n_start = 5, `set.seed(1)`
before the fit) and writes the files in
`tests/data/reference/optimizers/fit_distrib/` (13 MB in all; a README there
repeats the layout):

- `<distrib>_n1000.csv`: a header `y`, then 1000 values as C99 hexadecimal
  doubles (`%a`), exact in both directions;
- `<distrib>_n100000.f64`: 100000 values as raw IEEE 754 binary64,
  little-endian, 8 bytes each, no header and no count (800000 bytes), in
  the order of the draws;
- `fit_distrib.csv`: per (distrib, n, parameter) the true value, the
  estimate, standard error, log-likelihood, iterations, f/g/H counts, the
  converged flag and fit_distrib's score (max-norm of the score per
  observation on the link scale); and the
  closed-form MLE with its log-likelihood where one exists.

Each Laplace distrib was matched to the distributions7 constructor with the
same parameters (`invgauss` to `invgauss1`, `laplace1` to `laplace`, the
others by name), and the script first checks that the log-densities, written
in R from the Laplace headers, agree with distributions7's: at most 4.4e-16
at three points each. No parametrization differs. distributions7 calls the
logistic's mu a mean and the gumbel's mu a location; they are the same
parameters as Laplace's (the gumbel's mean is mu + gamma sigma in both).
Inverse Gaussian data are drawn by the method of Michael, Schucany and Haas
(1976); gumbel and Laplace data by transforming exponentials.

The closed forms: the gaussian mean and mean squared deviation (as sigma,
sigma^2 or its inverse), the exponential and Poisson mean, the lognormal from
the moments of log y (lognormal2 through the invariance m = exp(mu + s/2),
v = (e^s - 1) e^(2 mu + s), with mu and s the mean and variance of log y),
the inverse Gaussian mean and mean of 1/y - 1/ybar, the Laplace median and
mean absolute deviation from it (at even n the median is the midpoint of the
two middle values; the likelihood is flat between them and the scale is the
same anywhere there). Against them statmodels7 is within 8.3e-7 relative at
n = 1e3 and 7.9e-11 at n = 1e5, and its log-likelihood is lower by at most
1.3e-10. The difference comes from fit_distrib's stopping rule on the score
per observation (< 1e-6): on the gaussians it stops after one Fisher step
from a start with the n - 1 variance, with a relative error near
(1/(2n))^2 = 2.5e-7 in sigma. Every fit converged, in 1 to 3 iterations,
because fit_distrib starts from data-based values.

**What this says.** optimizers7's counts are a reference for the number of
evaluations and for the ways a run can end, not targets: about a third of
its successes rest on a change rule, one of them 5 units from the
minimizer, which is the class of false success that decision 5 removes.
The fit_distrib estimates are a reference to about 1e-6 relative at
n = 1e3 and 1e-10 at n = 1e5; where a closed form exists it is the sharper
oracle.

**Chosen.** Data at full precision as CSV for n = 1e3 and as binary for
n = 1e5, no n = 1e6 (decided with Giovanni); optimizers7 0.22.0.

**Open.** Laplace's side of the times and counts is measured by the
optimizer benches and by the fit_distrib bench against these files. The
bernoulli, binomial and geometric distribs, written at the same time, have
no fit_distrib reference yet.

## BFGS, L-BFGS and multistart: the stationarity of a quasi-Newton method

**What.** `bfgs_optimizer` (dense inverse-Hessian approximation B, O(n^2) per
iteration), `lbfgs_optimizer` (the last `memory` = 10 pairs, two-loop
recursion, O(memory n)) and `multistart_optimizer` (any optimizer from many
starts) in `include/laplace/optimizers/{bfgs,lbfgs,multistart}.hpp`.

**How.** Direction d = -B g, strong Wolfe line search by default (c1 1e-4,
c2 0.9), first trial step 1. Before any curvature pair, and after a reset,
B = min(1, 1 / |g|_inf) I. A pair s = x_new - x, y = g_new - g is used when
s'y > 1e-10 |s| |y|, otherwise skipped; BFGS resets B after 5 consecutive
skips, L-BFGS keeps its pairs. The first pair scales the identity by
s'y / y'y (Shanno and Phua 1978); L-BFGS starts its two-loop recursion from
gamma I with gamma = s'y / y'y of the latest pair. A direction that is not a
finite descent direction resets B.

Stationarity: the decrement
(1/2) max(g' B g, gamma_max g' g), with g' B g = |g' d| for the next
direction d and gamma_max the largest s'y / y'y of the accepted pairs (0
before any pair), and a convergence verdict of the stopping rule confirmed
on the next iteration (if the confirming line search finds no acceptable
step, below rounding, below its resolution, without a descent direction or
with its evaluation budget spent, the verdict stands). The budget case was
added after a run started at the optimum on the GPU, where f varies with the
order of the reduction: the confirming Wolfe search spent its 30 evaluations
and the run ended line search failed. Reproduced on the host with
f = 1000 + |x|^2 / 2 carrying a relative noise of 1024 eps, n = 3, from
x = 1e-6 (decrement 1.5e-12): before, line search failed after 31
evaluations; now converged.

Why both. (1/2) g' B g alone, with the default tolerance 1e-6, declared BFGS
converged far from x* on the quadratics of the battery (n = 10, x0 = 0,
x* = (1, ..., 10)):

| kappa | iterations | f - f* | decrement | (f - f*) / decrement |
|---|---|---|---|---|
| 1e2 | 38 | 4.1e-8 | 3.8e-8 | 1.07 |
| 1e4 | 60 | 3.6e-3 | 5.5e-7 | 6.6e3 |
| 1e6 | 84 | 7.4e-3 | 3.6e-7 | 2.1e4 |
| 1e8 | 107 | 9.8e-3 | 2.8e-8 | 3.6e5 |
| 1e10 | 103 | 1.4e2 | 2.0e-7 | 6.9e8 |

Without a stopping rule the same runs reach x* exactly (f - f* = 0) at
iterations 47, 79, 108 and 136 for kappa 1e2 to 1e8: the stop, not the
method, was wrong. The trace shows a plateau: B has learned the stiff
directions and still carries about 1 / lambda_max along a weak direction
that no step has sampled, where the remaining error lies; the realized
decrease of each step agrees with the predicted one (ratio 1.0 to 1.2), so
no consistency test sees it, and a stricter Wolfe curvature (c2 = 0.1) does
not help (ratio 9.1e3 at kappa 1e4). The same measure declared both methods
converged on abs_sum with n = 10 (f about 1e-6, a smooth method on a
non-smooth function) and BFGS on Wood with f - f* = 1.0e-3.

The floor gamma_max g' g gives every direction at least the largest inverse
curvature seen. With it alone one case remained: f = x - log x from
x0 = 0.01, where one step reaches 1.01 and the only pair, taken across
curvatures from 1e4 to 1, gives s'y / y'y = 0.0101 against 1 at the point:
converged with f - f* = 5e-5. The confirmation takes one more step, which
corrects B. Requiring two consecutive passes without the floor removed only
the kappa 1e4 case.

**Measured** (`benchmarks/optimizers_quasi_newton.cpp`, gcc, one thread,
decrement tolerance 1e-6). MGH problems from their standard starts; the
same iterations with f* = 0, 1, 1e3 and 1e6 up to a few runs at 1e6 (in
brackets):

| problem | BFGS it / f, g | L-BFGS it / f, g | BFGS us | L-BFGS us |
|---|---|---|---|---|
| Rosenbrock n = 2 | 35 / 41 | 36 / 43 (35 / 59) | 22 | 31 |
| Rosenbrock n = 100 | 36 / 42 | 36 / 43 | 356 | 94 |
| Powell singular | 31 / 32 | 31 / 33 | 22 | 36 |
| Beale | 14 / 15 | 14 / 15 | 13 | 14 |
| Brown badly scaled | 12 / 24 (11 / 38) | 13 / 25 | 11 | 12 |
| helical valley | 29 / 31 | 26 / 29 | 21 | 23 |
| Wood | 35 / 43 | 19 / 22, saddle | 27 | 17 |

All end converged with f - f* <= 6.3e-8 and max |x - x*| <= 6.1e-3 (Powell,
whose Hessian is singular at x*; 9.9e-5 or less elsewhere), except L-BFGS on
Wood, which stops near the saddle at f = 7.877 (x about
(-0.967, 0.945, -0.971, 0.953), one negative pivot of the Hessian,
max |g| = 0.011): a positive definite B cannot see negative curvature.

Quadratics, n = 10 (iterations, f - f* at the end):

| kappa | BFGS | L-BFGS m = 5 | m = 10 | m = 20 |
|---|---|---|---|---|
| 1e2 | 39, 3.2e-10 | 40, 6.1e-7 | 30, 6.8e-8 | 26, 4.5e-12 |
| 1e4 | 73, 1.1e-10 | 199, 3.1e-6 | 100, 1.5e-8 | 56, 7.1e-9 |
| 1e6 | 102, 1.2e-10 | budget | 249, 2.3e-7 | 90, 1.2e-8 |
| 1e8 | 130, 4.7e-9 | budget | 305, 9.8e-3 false | 119, 9.9e-3 false |
| 1e10 | 130, stalled, 1.1e-2 | budget | 457, 1.1e-2 false | 145, 1.1e-2 false |
| 1e12 | 91, stalled, 9.4e3 | budget | 106, line search failed | 62, stalled |

BFGS with Armijo instead of Wolfe converges at kappa 1e10 (158 iterations,
f - f* = 1.4e-15). n = 100, kappa 1e6: BFGS 868 iterations, f - f* 4.0e-7,
12 ms; L-BFGS reaches the 1000-iteration budget with every memory. The
false successes of L-BFGS at kappa 1e8 and 1e10 remain: its error lies along
directions that no pair has sampled, which neither B nor gamma_max can show.

Negative cases, none converged: the noisy objective (line search failed at
iteration 0), sum cosh from 700 with 100 iterations (iteration budget; with
1000 both converge at 860 iterations), abs_sum with n = 2, 4, 10 under Wolfe
and Armijo (stalled, line search failed or budget). abs_sum with n = 1
converges exactly: one step lands on 0, where the subgradient is 0.

Multistart over BFGS on Styblinski-Tang, f = (1/2) sum (x^4 - 16 x^2 + 5 x),
from x0 = (2.5, ..., 2.5) in the basin of the worst minimum, Latin hypercube
on x0 +- 3 max(1, |x0|):

| n | starts | converged | distinct optima | global found | f, g evaluations | ms |
|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 1 | no | 6 | 0.010 |
| 2 | 10 | 10 | 3 | yes | 130 | 0.086 |
| 2 | 40 | 40 | 3 | yes | 533 | 0.33 |
| 4 | 10 | 10 | 3 | no | 193 | 0.12 |
| 4 | 40 | 40 | 5 | yes | 805 | 0.49 |

Distinct optima are counted on the converged values sorted, a new one after
each gap larger than distinct_tolerance = 1e-5. Under the measure without
floor and confirmation the converged values at one minimum spread by 2.4e-6
and were counted as two optima at 1e-6.

**Chosen.** The floored decrement with confirmation, for both methods;
L-BFGS memory 10; constants as above (curvature tolerance 1e-10, 5 skips,
1000 iterations, 1e5 evaluations); multistart with 10 starts, spread 1, seed
1, distinct tolerance 1e-5, the kept run the best (a converged run before a
non-converged one, then the lower value, a later run only when lower by more
than the tolerance).

**Open.** L-BFGS's false successes on quadratics with kappa >= 1e8 and its
stop at Wood's saddle are limits of the method (tested as such, not hidden).
BFGS with Wolfe stalls at kappa 1e10 (130 iterations, f - f* = 1.1e-2) where
Armijo converges (158 iterations, f - f* = 1.4e-15): not studied.
Multistart runs its starts in sequence; threads or processes are for later.

## Newton and the trust region: the eigenvalue route, the exact subproblem

**What.** `newton_optimizer` (`include/laplace/optimizers/newton.hpp`) and
`trust_region_optimizer` (`include/laplace/optimizers/trust_region.hpp`), the
second-order methods of phase 6, both for a `curvature_objective` with a dense
H. Both declare success only through the stop rule, by default
`decrement_stop_rule` (decrement (1/2) g' H^-1 g <= 1e-6); the decrement is NaN
wherever H has no Cholesky factor, so no point with an indefinite or singular H
converges on it.

**How.** Newton: when the Cholesky factor of H exists, d = -H^-1 g. Otherwise
H = V diag(w) V' (w increasing) and d = -sum_i (v_i' g) / max(|w_i|, floor) v_i
with floor = 1e-8 max|w_i| (an eigenvalue modification, Nocedal and Wright
2006, section 3.4), plus a unit step along v_1 with g' v_1 <= 0 when
w_1 < -floor (a direction of negative curvature, More and Sorensen 1979); that
direction is scaled down to a largest component of `max_modified_step` = 1,
because its length is set by the floor and not by the curvature. Where
g' d = 0 (g = 0 at a saddle) the step is accepted on the second-order decrease
f(x + s d) - f(x) <= c1 s^2 d' H d / 2, halving s. Every direction is then
scaled down to a largest component of `max_step` (Dennis and Schnabel's
maxstep in the max norm; default unbounded), and Armijo searches from s = 1.

Trust region: the step minimizes g' p + p' H p / 2 over |p|_2 <= radius (More
and Sorensen 1983). The Newton step is taken when H is positive definite and
the step fits. Otherwise, with H = V diag(w) V' and gamma = V' g,
|p(lambda)|^2 = sum gamma_i^2 / (w_i + lambda)^2 and lambda solves
1/|p(lambda)| = 1/radius by Newton's method, safeguarded by bisection on
[max(0, -w_1), max(0, -w_1) + |g| / radius], in the eigenbasis: one
eigendecomposition per accepted point, O(n) per secular iteration, gamma
scaled by its largest entry so that the squares cannot overflow. Eigenvalue
sums w_i + lambda below delta = 8 n eps max|w| are treated as zero. When
|p(max(0, -w_1) + delta)| <= radius the step is the one at lambda = max(0, -w_1)
without the unresolved components, plus tau v_1 to the boundary when
w_1 < -delta (the hard case). Radius: |p| / 4 when rho < 1/4 or the trial value
is not finite, doubled when rho > 3/4 and p is on the boundary, the step
accepted when rho > 1e-4 (Nocedal and Wright 2006, Algorithm 4.1); initial
radius 1, no maximum.

**Measured** (`benchmarks/optimizers_second_order.cpp`, gcc, one thread):
iterations, evaluations of f, g (= H), f - f*, us per run. Every run below
ends converged; with f* shifted by 1e6 the counts are the same.

| problem | Newton | | | | trust region | | | |
|---|---|---|---|---|---|---|---|---|
| | iter | f, g | f - f* | us | iter | f, g | f - f* | us |
| Rosenbrock, n = 2 | 23 | 55, 24 | 3.3e-10 | 17.2 | 24 | 46, 22 | 7.6e-9 | 19.6 |
| extended Rosenbrock, n = 100 | 23 | 55, 24 | 1.7e-8 | 1529 | 23 | 45, 22 | 4.7e-10 | 1966 |
| Powell singular | 12 | 25, 13 | 5.7e-7 | 13.8 | 12 | 25, 13 | 6.0e-7 | 16.4 |
| Beale | 6 | 15, 7 | 3.3e-8 | 9.1 | 6 | 12, 6 | 7.8e-7 | 9.9 |
| Brown badly scaled | 184 | 375, 185 | 2.1e-15 | 155 | 31 | 63, 32 | 1.4e-19 | 34.1 |
| helical valley | 14 | 31, 15 | 2.2e-8 | 17.1 | 8 | 16, 8 | 1.6e-9 | 12.7 |
| Wood | 41 | 102, 42 | 6.8e-8 | 61.0 | 44 | 83, 39 | 2.0e-10 | 59.1 |
| quadratic n = 10, kappa 1e2 | 1 | 3, 2 | 3.2e-28 | 6.3 | 5 | 11, 6 | 1.2e-28 | 31.1 |
| quadratic, kappa 1e6 | 1 | 3, 2 | 4.7e-20 | 6.2 | 5 | 11, 6 | 5.1e-21 | 27.0 |
| quadratic, kappa 1e12 | 1 | 3, 2 | 4.5e-10 | 6.3 | 5 | 11, 6 | 9.5e-10 | 27.6 |
| saddle from (1, 0) | 1 | 3, 2 | 0 | 5.1 | 5 | 11, 6 | 6.9e-10 | 9.0 |
| Rosenbrock from (0, 1) | 2 | 5, 3 | 0 | 5.8 | 4 | 9, 5 | 4.6e-11 | 9.2 |

The trust region's iterations count every trial step; its accepted steps are
the g evaluations less one.

Newton's `max_step`: 5, and 1e3 max(1, max|x0_i|) (Dennis and Schnabel's
default in the max norm), end Brown badly scaled on the budget of 200
iterations (x* = (1e6, 2e-6)); 5 also costs Wood 58 iterations against 41;
elsewhere the counts are unchanged. Unbounded by default.

Newton's `max_modified_step` (iterations, f evaluations):

| problem | 1 | 10 | 1e3 | inf |
|---|---|---|---|---|
| Beale | 6, 15 | 14, 39 | 14, 39 | 14, 39 |
| Brown badly scaled | 184, 375 | 25, 55 | 5, 11 | 5, 11 |
| helical valley | 14, 31 | 21, 56 | 21, 56 | 21, 56 |
| Wood | 41, 102 | 57, 152 | 57, 152 | 57, 152 |
| Rosenbrock from (0, 1) | 2, 5 | 22, 65 | 22, 65 | 22, 65 |

Rosenbrock n = 2 and 100, Powell and the saddle do not depend on it. On Brown,
the first step lands where H is indefinite and the capped direction moves
x_2 by 0.0025 per iteration.

Initial radius 10 instead of 1: quadratics 2 iterations instead of 5, Brown
26 instead of 31, Rosenbrock from (0, 1) 12 instead of 4, helical valley 11
instead of 8.

**Tests** (`tests/optimizers/test_second_order.cpp`, 6 cases, 720 assertions,
gcc and CUDA). The MGH problems with f* = 0, 1, 1e3 and 1e6 and the quadratics
with kappa 1e2 to 1e12 must end converged with f - f* <= 2 tol plus 8 eps |f*|
(the decrement is f - f* on a quadratic, 2/3 of it on Powell's quartic) and a
positive definite final factorization; on the quadratics also
|x_i - x*_i| <= sqrt(4 tol) + 1e-9 kappa, and Newton in at most 2 iterations.
Both methods leave the saddle started from (1, 0) and from the saddle itself,
reaching (0, +-1). A trial point with a non-finite value is refused
(x - log x from 10, where the Newton step goes to -80) and a non-finite start
is reported as not admissible. Must not converge: the noisy objective, the
exploded start (iteration budget at 100), sum |x_i| (decrement NaN
throughout), and a change rule (stopped on change).

**Repaired while testing.** The resolution shift of the subproblem was
1e-10 (max|w| + |g| / radius): on Brown badly scaled (eigenvalues 2 and 2.7e11)
it exceeded the small eigenvalue and the step dropped that component, and
the run stalled at f = 4e11; now 8 n eps max|w|. gamma^2 overflowed at the
exploded start (gamma about 5e303); now scaled. At an exact stationary point
(g = 0) the decrement was NaN because the Newton slope was tested as < 0; now
<= 0.

**Chosen** (Giovanni, 2026-11-15): the defaults above (max_step unbounded,
max_modified_step 1, floor 1e-8, the negative-curvature step, decrement NaN
without a Cholesky factor, subproblem in the eigenbasis, initial radius 1,
max_iterations 200 for Newton and 1000 trial steps for the trust region,
1e5 value evaluations).

**Open.** The cap of 1 on the modified direction is in the units of x and
costs Brown badly scaled 37 times its iterations. The trust region takes an
eigendecomposition at every accepted point where the Newton step does not
fit; at n = 100 it is 1.3 times slower than Newton for the same iterations.

## The fit_distrib bench: starting values, Newton and BFGS on iid data

What. The optimizers of wave 1 fitted to iid samples of the 16 parametrizations
with frozen references (cauchy, chisq, exponential, gamma1, gamma2, gaussian1-3,
gumbel, invgauss, laplace1, laplace2, logistic, lognormal1, lognormal2, poisson)
and of bernoulli, binomial and geometric, with starting values from the library.
The bench is a test, not an API (`tests/optimizers/fit_distrib_bench.hpp`); the
starting values are in the library (`include/laplace/distrib/generic_start.hpp`
and a `start_value` member in each distrib).

How. The objective is f(u) = -sum_i l(theta(u); y_i), summed over the n
observations, in free coordinates u with theta_j = h_j(u_j): the identity on R,
exp on (0, inf), the logistic map on (0, 1). One Kokkos `parallel_reduce` per
call accumulates 1 sum (value), 1 + P (value and score) or 1 + P + P(P+1)/2
(also the observed second derivatives); the chain rule through h is applied on
the host. The expected curvature is n I(theta) h' h'^T, with I(theta) evaluated
once on the host, so Fisher scoring costs one order-1 pass per iteration. The
observed curvature is refused for a parameter whose log-density has regularity
r_j < 2 (mu of laplace1 and laplace2). Methods: Newton with the observed
curvature, Newton with the expected one, BFGS (Wolfe search); stop rule
decrement <= 1e-6 in log-likelihood units.

Starting values. `summarize_sample(y)` (mean and variance with divisor n,
median and quartiles of type 7, mean |y - median|, moments of log y and the
mean of 1/y over y > 0) and `distrib_start_value(prototype, summary)`, which
calls the distrib's own `start_value`; a distrib without one does not compile.
The member gives the closed-form MLE where one exists (gaussian1-3, lognormal1,
exponential, poisson, geometric, bernoulli, binomial, invgauss with phi =
mean(1/y) - 1/mean, laplace1 and laplace2 from the median and mean |y - median|)
and a moment estimate otherwise (gamma1, gamma2, lognormal2, logistic, gumbel,
chisq; cauchy from the median and half the interquartile range). Each value is
moved strictly inside its domain (`start_inside`: 1e-8 max(|v|, 1) at an end 0,
16 eps relative at another finite end), so that u is finite even when the MLE is
on the boundary.

Results (gcc and CUDA builds alike). All 16 x {n = 1e3, 1e5} x {library start,
poor start (u + 1 in every coordinate)} x {Newton observed (14 distribs), Newton
expected, BFGS}: 188 of 188 runs converged. Every estimate is within the bound
the decrement rule allows, |u_j - u_j*| <= sqrt(4 tol (H^-1)_jj) with H the
expected curvature in u, of the closed-form MLE where it exists and of the frozen
statmodels7 estimate. The largest relative distance from the closed forms at
n = 1e3 is 4.0e-5 (Newton observed, stopping at a decrement of up to 1.4e-7 after
one step); BFGS 1.1e-6. The log-likelihood is never below the reference's by more
than 2 tol. Bernoulli (mu 0.3), binomial (mu 0.3, size 10) and geometric (mu 2.5)
on 1e3 and 1e5 draws in C++ (inversion, counter-based source, fixed seed) reach
the sample mean within the same bound.

- laplace1, laplace2: at even n the log-likelihood is flat in mu between the two
  middle order statistics, and every mu there is an MLE. Newton with the expected
  curvature and BFGS end inside that interval (log-likelihood equal to the
  reference within 4e-13); the test checks the interval, not the midpoint.
- Boundary data (bernoulli all 0 or all 1, geometric all 0, n = 1e3 and 1e5):
  every run ends converged, correctly in the objective's units: -l tends to its
  infimum 0 only as u tends to infinity, and the decrement n mu / 2 is the true
  remaining decrease (bernoulli all 0: mu = 1.35e-9, -l = 1.35e-6, decrement
  6.8e-7 after 2 Newton iterations). The bench flags such a fit as on the boundary
  of the domain; the flag is about the fit, and the optimizer's verdict is about f.
  The rule: some parameter with a finite end of its domain has a standard error
  above 10 in its free coordinate, sqrt((H^-1)_jj) with H = n I(theta) h' h'^T
  the expected information in u. These coordinates are log or logit maps, so the
  measure is relative and does not depend on the scale of theta; parameters on
  all of R are not checked. Measured, Newton (both curvatures) and BFGS from the
  library start: on boundary data the standard error is 860 to 1080 (bernoulli
  and geometric all 0, n = 1e3 and 1e5), 5.3e4 and 5.3e5 (bernoulli all 1), 3.2e3
  to 7.5e3 (bernoulli 0.01 drawn at n = 2 to 10, every draw 0). On genuine fits it
  is at most 0.076 for every reference fit at n = 1e3 and 1e5, 0.045 for gaussian3
  with tau = 1e-6 on data of scale 1e3 (n = 1e3, estimate 9.5e-7), 0.022 for
  gaussian1 with sigma = 1e-4, 0.032 for exponential with mu = 1e-5, 0.042 for
  gamma1 with mu = 1e3, and at most 1.41 on samples of n = 2, 3, 5 and 10 of the
  same laws and of bernoulli 0.5 and geometric 0.05 (the largest: bernoulli 0.5 at
  n = 2, mu = 0.5). A threshold of 10 lies more than 7 times inside either side.
  Two alternatives fail: theta within 1e-6 relative of the end measured against
  the start does not flag bernoulli all 0, whose start is already 1e-8; a large
  move of u from its start does not separate it either (2 units, against 1 unit
  for every run from the poor start).
- BFGS from a start at the optimum on the GPU: the reduction sums in an order that
  depends on the backend, so f varies at a few eps |f|; the step that confirms a
  convergence (two consecutive passes) found no acceptable point and ended with
  its evaluation budget (gaussian3, lognormal1, poisson at n = 1e5: start
  decrement 1e-19, line search failed after 30 evaluations). BFGS and L-BFGS now
  keep a pending verdict also when the confirming search exhausts its budget, and
  these runs end converged.

Times (ms per fit, the fastest of repeated fits; library start; gcc 14.2 with
OpenMP on 1 and 16 threads, CUDA 13.1 on an RTX 4070 SUPER; n = 1e6 drawn in C++
at the references' true values). Iterations are those of n = 1e5 (in brackets
where n = 1e6 differs).

| distrib | method | it | 1e3: 1 / 16 / GPU | 1e5: 1 / 16 / GPU | 1e6: 1 / 16 / GPU |
|---|---|---|---|---|---|
| gaussian1 | Newton expected | 0 | 0.011 / 0.041 / 0.029 | 0.55 / 0.11 / 0.055 | 5.5 / 0.71 / 0.25 |
| lognormal1 | Newton expected | 0 | 0.020 / 0.045 / 0.032 | 1.46 / 0.20 / 0.080 | 15.2 / 1.73 / 0.48 |
| laplace1 | Newton expected | 0 | 0.012 / 0.043 / 0.028 | 0.80 / 0.11 / 0.051 | 7.8 / 0.82 / 0.21 |
| poisson | Newton expected | 0 | 0.092 / 0.056 / 0.078 | 9.7 / 1.10 / 0.42 | 98 / 10.7 / 3.7 |
| cauchy | Newton expected | 1 | 0.15 / 0.24 / 0.14 | 9.7 / 0.93 / 0.28 | 100 / 8.2 / 2.0 |
| logistic | Newton observed | 1 | 0.63 / 0.23 / 0.16 | 63 / 7.1 / 0.83 | 632 / 70 / 7.1 |
| logistic | Newton expected | 1 | 0.22 / 0.26 / 0.15 | 13.9 / 1.40 / 0.35 | 140 / 12.4 / 2.6 |
| logistic | BFGS | 7 (6) | 0.42 / 0.47 / 0.30 | 54 / 5.4 / 1.5 | 1953 / 50 / 12.4 |
| gumbel | Newton expected | 1 | 0.16 / 0.25 / 0.11 | 10.5 / 1.17 / 0.18 | 105 / 10.5 / 1.0 |
| gamma1 | Newton expected | 2 (1) | 0.98 / 0.35 / 0.59 | 98 / 11.1 / 3.9 | 626 / 68 / 20 |
| gamma2 | Newton observed | 2 (1) | 2.5 / 0.54 / 1.08 | 248 / 27 / 7.9 | 1573 / 168 / 42 |
| gamma2 | Newton expected | 2 (1) | 1.35 / 0.41 / 0.68 | 133 / 15.0 / 4.7 | 867 / 91 / 24 |
| gamma2 | BFGS | 4 (3) | 3.4 / 0.89 / 1.6 | 303 / 34 / 10.6 | 3693 / 396 / 104 |
| lognormal2 | Newton expected | 2 (1) | 0.88 / 0.35 / 0.20 | 88 / 7.3 / 0.93 | 553 / 46 / 4.5 |
| chisq | Newton expected | 1 | 0.39 / 0.20 / 0.26 | 41 / 4.4 / 1.5 | 392 / 43 / 12.6 |

Cost of one pass at n = 1e6 (us per pass, 1 thread / 16 threads / GPU), order 0
(value) and order 2 (value, score, observed second derivatives):

| distrib | order 0 | order 2 |
|---|---|---|
| gaussian1 | 4381 / 547 / 171 | 7478 / 922 / 379 |
| logistic | 40795 / 3277 / 434 | 313360 / 30911 / 3250 |
| gamma2 | 125386 / 12995 / 3861 | 748667 / 73881 / 18731 |
| poisson | 96760 / 10728 / 3588 | 97440 / 10299 / 3719 |

What it says. The start decides most of the work: with the closed-form MLE as the
start, nine of the sixteen fits take no iteration (one pass; at 1e6 0.2-0.5 ms on
the GPU, poisson 3.7 ms). An iteration of Fisher scoring needs an order-1 pass
only, so it is 2-5 times faster per fit than Newton with the observed curvature
wherever the second derivatives are expensive (logistic 4.5 times at 1e5, gamma2
1.9 times); it is slower where it needs one more iteration (gumbel and logistic
at n = 1e3, 2 iterations instead of 1). BFGS takes 2-5 times the time of Fisher
scoring (more passes, no curvature). At n = 1e5 and 1e6, 16 threads give 8-12
times one thread for the costly densities and 5-10 times for the cheap ones; at
n = 1e3 the fork-join overhead (about 37 us a pass) makes them slower than one
thread for the cheap distribs. At 1e6 the GPU gives 20-37 times one thread for
most distribs, 54 times for logistic and 102 times for gumbel.

Open. (1) The confirming step of BFGS (decision 5, V3) costs up to 30 value
evaluations when f is noisy at its rounding level just above the resolution
(logistic at 1e6 on one thread: 39 value evaluations and 1953 ms, against 11
evaluations and 50 ms on 16 threads);
a resolution of a few tens of eps |f| would remove it, at the price of stopping
earlier where the curvature is small. (2) Poisson, chisq and the gamma
parametrizations evaluate lgamma or the Stirling error at every pass although
the terms that depend only on y could be summed once; that is an engine
optimization (sufficient statistics), not done here.

## The two negative binomial families: negbin1 and negbin2

**What.** `negbin2_distrib` (mean mu > 0, dispersion theta > 0, variance
mu + mu^2/theta) and `negbin1_distrib` (the same two parameters, variance
mu (1 + theta)), the fourth and fifth of the seven discrete families of phase 4
(`include/laplace/distrib/negbin2.hpp`, `negbin1.hpp` and the shared
`negbin_core.hpp`). Each writes the log-mass with its six derivative orders and
the cumulative functions, which are closed through the regularized incomplete
beta, F(k) = I_p(r, k + 1) with r the shape and p the success probability; the
derivatives of log F and log(1 - F), the expected information with its
derivatives, the quantile and the random numbers come from the generic routes
(`generic_cdf.hpp`, `generic_info.hpp`, `generic_quantile.hpp`,
`generic_rng.hpp`). Of the information, negbin2's I_mu_mu = 1/(mu + mu^2/theta)
and I_mu_theta = 0 are known in closed form and the test checks the generic sum
against them; negbin1 has no orthogonal block.

**How, the form.** Both families are the same mass in a shape r and a mean m,
with (r, m) = (theta, mu) for negbin2 and (mu/theta, mu) for negbin1:

    p(y) = Gamma(y + r) / (Gamma(r) y!) (r/(r+m))^r (m/(r+m))^y.

Every grouping of that expression keeps terms of the size of y log y while the
log-mass is of the size of log y, so at large counts it loses about log10(y)
digits. The form used is Loader's saddle point, as in `poisson.hpp` and
`binomial.hpp`, with delta Stirling's error (`math::stirling_error`, equal to
Loader's stirlerr) and bd0(a, b) = a log(a/b) - a + b: for y >= 1,

    l = (1/2) log( r / (2 pi y (y + r)) ) + delta(y+r) - delta(y) - delta(r)
        - bd0(y, b1) - bd0(r, b2),
    b1 = m (y+r)/(r+m),  b2 = r (y+r)/(r+m),  b1 + b2 = y + r,

and both deviances are taken at the one numerator Delta = r (y - m)/(r + m),
since b1 - y = -Delta and b2 - r = Delta exactly. At y = 0 the form has log y in
it and the mass is written directly as r log(r/(r+m)) = -r log1p(m/r).

**How, the derivatives.** Each family composes that form with its own rational
functions of (mu, theta), never with the shape: with r = mu/theta the chain
rule's terms are of the size of theta^(1-K) while d^K l / dtheta^K is of the
size of one, a relative loss of theta^(K-1). negbin1 therefore reads
(y + r)/r = 1 + u with u = y theta / mu, b1 = (y theta + mu)/(1 + theta) and
w1 = (y - mu)/(y theta + mu), all regular at theta = 0. A deviance is
differentiated through psi1(z) = z - log1p(z), since bd0(a, b) = a psi1((b-a)/a):

- at the shape, where b - a = Delta always, this is a product of a and one
  composition (negbin2: theta psi1((y - mu)/(theta + mu)), with b2 never
  formed; negbin1: -b1 w1 psi(-theta w1) with psi(v) = D(1+v)/v, since
  b2 = b1/theta and w2 = -theta w1 exactly, so the 1/theta cancels);
- at the count, the composition y psi1(z), z = (b1 - y)/y, below |z| = 1/2 and
  the two closed tensors d^a b1 - y d^a log b1 above it.

The small algebra that assembles them (packed pairs of derivatives of total
order at most 6, in the layout of `math/chain_pair.hpp`) is in
`negbin_core.hpp`: constants, affine forms, their reciprocals and logarithms,
products by Leibniz's rule and the composition of a function of one variable,
which calls `chain_rule_pair_dk` once per order. Every rational is built from a
closed formula with one numerator rather than from a product of tensors:
theta/(theta+mu) and mu/(theta+mu) have
d^(p+q) = (-1)^(p+q) (p+q-1)! (p theta - q mu) / (theta+mu)^(p+q+1) up to sign,
whose pure derivatives in either parameter are single terms. negbin1's
difference of Stirling's errors delta(y+r) - delta(r) is taken as a function of
s = theta/mu above r = 10, where both arguments are in the range of the
asymptotic series of `stirling.hpp`, so that delta(r) = sum_j sigma_j s^(2j-1)
and delta(y+r) the same series at t = s/(1 + y s); below r = 10 the shape is a
regular variable and the composition is taken in it.

**What was tried and rejected.** Four routes were written and measured before
this one; each is recorded because each looked right:

1. the log-gamma difference at the integer shift y in Stirling form, with
   lgamma(y+1) left as a term: 1.4e5 ulp on the mass at y = 1e5 (both
   families), the loss of log10(y) digits described above;
2. negbin1 by composing the core's derivatives in r with r = mu/theta: 6e23 ulp
   at order 6 and theta = 1e-6, the loss of theta^(K-1);
3. a deviance as the product b D(1 + w) differentiated by Leibniz's rule:
   3.8e6 ulp on the second derivative in mu at y = 1, mu = 1e3, theta = 1e6,
   where the intermediate terms are 1.2e-2 and the result 1e-6; and the
   composition of f(b) = bd0(a, b) with b: 4.9e11 ulp at order 6 for a count
   far from the mean, where the Bell sum alternates with terms 1e7 times the
   result;
4. negbin1's bd0(r, b2) written out as
   mu (1 + kappa(theta)) - y (1 + kappa(u)) + (y - mu)/(1 + theta): terms of the
   size of mu against a result of the size of (y - mu)^2 theta / mu, so 1.3e4
   ulp on the value at (1e5, 1e-3) near the mode.

The rationals built as products of tensors cost a further mu/theta in the mu
direction (1.9e5 ulp at order 1 at (1e5, 1e-3)), and -(V(theta, 0, mu) + mu)
for the mass at y = 0 subtracts mu from a quantity of its own size (239 ulp at
mu = 40, theta = 0.01) where one log1p cancels nothing.

**Measured.** References from `tools/reference/distrib_negbin.py`, which first
checks every closed form against SymPy (Loader's identity, both
parametrizations' rationals, negbin1's deviance at the shape as a product,
kappa and psi with their series and their recursions, t(s), and V with its
series) and then writes 5397 rows: the mass from lambdified symbolic
derivatives of the definition at 400 and 500 bits, log F and log(1 - F) from
mpmath's own incomplete beta by `mpmath.diff` at 200 and 260 bits, each kept
where the two precisions agree to 2^-80. The mass rows need the high precision:
the symbolic sixth derivative in theta carries terms of the size of theta^-6
against a result of the size of mu, so at theta = 1e-6 about 31 digits are lost
and 200 bits leave too few. Worst error in ulps of the largest component of the
same order at the same point, gcc (Serial + OpenMP) and CUDA:

| quantity | negbin2 | negbin1 |
|---|---|---|
| log-mass | 18.8 | 41.2 |
| its derivatives, orders 1 to 6 | 5.2 to 1.6e3 | 24.6 to 3.1e3 |
| log F, orders 0 to 6 | 17.5 to 8.1e3 | 39.3 to 9.0e8 |
| log(1 - F), orders 0 to 6 | 45.7 to 519 | 25 to 272 |

The two builds agree on every cell: the only differences over the 5397 rows are
one ulp on negbin1's log(1 - F) at orders 0 and 6 (272 against 271 and 109
against 110), so one figure is given for both.

The outliers of each row are named points, not a general level: negbin2's
second derivative reaches 1.6e3 ulp at (mu = 1e3, theta = 1e6, y = 1), where
theta = mu^2/y makes the exact d^2 l / dmu^2 = -y/mu^2 + (y+theta)/(theta+mu)^2
a zero to three digits, and negbin1's fourth reaches 3.1e3 ulp at
(1e5, 1e-3, y = 100158), where the four pieces of the assembled mass are 140 to
700 times the result. The derivatives of log F by the generic summation route
are at the level of that route, tens to thousands of ulp, with two corners:
negbin1 at theta = 1000 (a shape of 0.04, so a tail that decays by 0.999 a
step) reaches 9.0e8 ulp at order 6, and the value log F itself reaches 5.4e5 ulp
at (1e3, 1e-6), where p = 1 - 1e-6 and log I_p is within 1e-6 of zero -- the
limit `log_ibeta` already records for the binomial.

Speed, ns per call as Serial (one host thread) / CUDA,
`benchmarks/distrib_negbin.cpp`:

| quantity | negbin2(3.5, 2) | negbin1(3.5, 2) | negbin1(2242, 0.028) |
|---|---|---|---|
| log p | 255 / 170 | 272 / 191 | 144 / 428 |
| log p, order 1 | 732 / 159 | 949 / 677 | 461 / 886 |
| log p, order 6 | 13100 / 3920 | 26900 / 7310 | 34000 / 46300 |
| log F (incomplete beta) | 281 / 129 | 332 / 166 | 511 / 2100 |
| log F, order 2 (generic sum) | 2.5e5 / 1.2e5 | 4.2e5 / 1.7e5 | 1.5e6 / 1.9e6 |
| I (generic sum) | 1.0e5 / 1.3e4 | 1.5e5 / 2.9e4 | 3.9e5 / 4.2e5 |
| I, order 2 (generic sum) | 9.3e5 / 1.0e5 | 8.9e5 / 1.6e5 | 3.9e6 / 4.6e6 |

The last column is launched on 64 arguments against 1024 for the first two, so
its CUDA figures are the launch, not the kernel; the generic sums walk a tail
whose length depends on the parameters, which is why they cost ten to a hundred
times the mass. OpenMP with 4 threads is 2 to 4 times Serial on the mass and
its derivatives.

**What this says.** The mass and its derivatives are at the accuracy of the
functions they are built from, tens of ulp, once three things hold together:
Loader's form for the value, inner maps that are regular where the family's
Poisson limit is, and a deviance differentiated as a product or a composition
of quantities of the result's own size rather than by Leibniz's rule on a
product of a large factor and a small one. The three rewrites above were each
worth about four digits, and the one that mattered most was the last: an
alternating Bell sum is unusable at order 6 when its terms are far from the
result. What remains is either a genuine zero of the quantity asked for or the
accuracy of a route these families only call.

**Chosen.** As above. Test `tests/distrib/test_negbin.cpp`, bounds at 1.5 times
the worse of the two builds, every reference row evaluated inside a Kokkos
kernel; the two families also check against each other (negbin2(mu, theta) is
negbin1(mu, mu/theta)) and against the geometric at theta = 1.

**Open.** The order-6 tensor costs 13 to 34 microseconds because every order
reassembles the whole tensor through `chain_rule_pair_dk`; orders 1 and 2, which
is what a fit reads, are under a microsecond, so this is left until a profile
asks for it. The derivatives of log F have no closed route here: writing one
through the mixed derivatives of the incomplete beta in its shape and its
argument is the same open point the binomial records. The cumulative functions
inherit the conditioning of I_p(r, k + 1) in p, which costs theta ulp per ulp of
p: the two parametrizations agree on the mass to 1e-12 and on the cumulative
function only to 1e-15 theta. The starting values are the moment estimates
inverted from the variance, mu = mean for both and theta = mu^2 / (s^2 - mu) for
negbin2 and s^2/mu - 1 for negbin1; a sample with no overdispersion gives an
estimate outside the domain, and the two families need opposite treatment there,
since negbin1's Poisson limit is the lower end of theta (where `start_inside`
already puts it) while negbin2's is the upper one, so its denominator is floored
at mu/100, which starts it at theta = 100 mu, a variance of 1.01 times the mean.
The floor is a convention, not a measurement.

## The Mills ratio on the log scale, with its derivatives to order 6

**What.** `own::log_mills(x)` and `own::log_mills_d1 .. _d6(x)`
(`include/laplace/math/normal.hpp`, with the wrapper in `laplace::math`):
log M(x) and its derivatives in x to order six, where
M(x) = Phi(-x) / phi(x) = sqrt(pi/2) erfcx(x / sqrt 2) is the Mills ratio.
`inv_mills(t)` = phi(t)/Phi(t), which is 1/M(-t), was already there. The
normalizing constant of the elastic-net density is 2 M(x) / sqrt(c) with
x = a / sqrt(c), a = lambda alpha and c = lambda (1 - alpha), so its logarithm
and the derivatives of that log-density in (lambda, alpha) read exactly these
seven functions.

**How.** The value: for x >= 0 from `erfcx_pos`, which neither cancels nor
underflows (erfcx(x / sqrt 2) falls like sqrt(2/pi) / x, so log M falls like
-log x); for x < 0, where erfcx overflows below x = -37.6, from
log M = log Phi(-x) + x^2/2 + log(2 pi)/2 with x^2/2 carried as the exact pair
h + l, as log Phi does. Below -sqrt(2 DBL_MAX) = -1.8961e154 the quadratic
passes DBL_MAX and the value saturates to +inf.

The derivatives rest on one identity. With t = -x,
log M(x) = log Phi(t) + x^2/2 + log(2 pi)/2, so

    d^k log M / dx^k = (-1)^k [d^k log Phi / dt^k](-x) + x [k = 1] + [k = 2],

which from order three on is the derivative of log Phi alone: orders 3 to 6 are
`log_Phi_d3 .. _d6` at -x with the sign (-1)^k, and nothing is added or
subtracted. At orders one and two the two terms cancel for large x, each being
of size x and of size one while the results are -1/x and 1/x^2. There the
cumulant form is used instead: d^k log M / dx^k = (-1)^k kappa_k, with kappa_k
the cumulants of the density proportional to exp(-x s - s^2/2) on s > 0, which
`log_Phi_dk` already computed inside its own branch below t = -1 and which is
now a function of its own, `detail_normal::mills_cumulant(k, x, c2, m0)` (the
downward ratio recursion with K = k + m0 + ceil(c2 / x^2) steps, then the
moment-to-cumulant recursion). `log_Phi_dk` calls it in place of its own copy,
operation for operation, so its measured numbers are untouched. The split is at
x = 1, which is where `log_Phi_dk` changes route as well.

So the only new arithmetic is the value for x >= 0 and the two cumulant calls;
the rest is the normal block's own measured code, read through the identity.
`detail_owen::log_g0_series` carries the same recursion in an all-orders form
(one downward pass fills every order's Taylor coefficient) and keeps it: it
sits in the skew normal's cdf, where one pass for five orders is the point.

**Measured.** `tools/reference/mills.py` writes `log_mills.csv` (2412 rows) and
`log_mills_d1 .. _d6.csv` (2011 rows each) at 220 bits, every row checked
against a computation at 290 bits. The route is independent of the C++ one: the
raw moments m_k = int_0^inf s^k exp(-x s - s^2/2) ds of the same density are
integrated with mpmath (positive integrands, so no quadrature cancels; the
substitution s = u/x above x = 1 keeps the mass at u of order one for every x
up to 1e9), then log M = log m_0 and the cumulants follow from the moments. The
grid runs over x from 1e-6 to 1e9, a fifth of the rows in [0.4, 2.5] around the
split, a fifth in [-5, 1], a fifth in [25, 45] where Phi(-x) underflows on the
natural scale, and, for the value alone, down to x = -1e8. Below x = -5 the
cumulants of order three and above fall like exp(-x^2/2) while the raw moments
grow like |x|^k, so no working precision recovers them by this route; the C++
route there is `log_Phi_dk` at a positive argument, which has its own
references.

Worst error over every row, relative, in ulps of the result:

| quantity | gcc | CUDA | worst x (gcc, CUDA) | the plain form, gcc |
|---|---|---|---|---|
| log M | 858 | 858 | 0.3021, 0.3021 | 6.6e16 |
| order 1 | 2.25 | 1.84 | 0.843, 36.1 | 8.7e15 |
| order 2 | 5.12 | 4.44 | 90.7, 0.991 | 8.7e15 |
| order 3 | 13.9 | 11.0 | 1.010, 1.026 | the same route |
| order 4 | 26.0 | 18.6 | 1.515, 1.423 | the same route |
| order 5 | 96.2 | 44.4 | 1.011, 1.066 | the same route |
| order 6 | 536 | 485 | 1.108, 1.084 | the same route |

The last column is the plain form, which takes every order from log Phi at -x
and adds the quadratic's terms at orders one and two, and the value from
log Phi(-x) + x^2/2 at every x. The two differ only where the split exists, and
there the plain form keeps no digit at all beyond x of a few thousand (its
worst points are x = 2.1e9 for the value, 1.4e8 and 1.9e8 for the two orders).
From order three on the two are the same code, which is why those rows agree.

The 858 ulp of the value are the conditioning of its zero: log M crosses zero
at x = 0.30246, where M(x) = 1, and no route that computes M first is
relatively accurate there. The quantity enters a log-density additively, so
what matters is its absolute error: 2.03 ulp of max(|log M|, 1) on gcc and 1.79
on CUDA over the whole grid. The same measure for the six orders is at most
3.01 (gcc, order 5) and 4.19 (gcc, order 6), 3.79 on CUDA.

Speed, one thread (Serial) and the device, over 2^20 points of x log-uniform in
[1e-3, 1e6]:

| quantity | Serial ns (gcc) | OpenMP ns | Serial ns (MSVC) | CUDA ns |
|---|---|---|---|---|
| log M | 15.6 | 1.89 | 16.6 | 0.79 |
| order 1 | 75.0 | 7.20 | 82.0 | 8.73 |
| order 2 | 81.3 | 7.83 | 90.4 | 7.97 |
| order 3 | 90.5 | 8.40 | 99.9 | 8.29 |
| order 4 | 107 | 10.1 | 129 | 8.62 |
| order 5 | 123 | 11.7 | 140 | 8.55 |
| order 6 | 130 | 11.5 | 146 | 7.97 |

**What this says.** The identity makes four of the six orders free: they are
`log_Phi_dk` with a sign, and their errors are that function's own in its
branch below t = -1 (767 ulp on gcc measured there at order 6, against the 536
seen over this grid). The two orders that do cancel are repaired by the
recursion the module already had, and the plain form shows what the repair is
worth: 8.7e15 ulp against 5 ulp. The cost is log Phi's derivatives': each order
runs its own downward recursion, so asking for all six costs about 600 ns on
one thread near the split, where the recursion takes 20 + ceil(300 / x^2)
steps.

**Chosen.** As above, bounds in `tests/math/test_normal.cpp` ("the Mills ratio
on the log scale against mpmath"): for the value 3.0 ulp of max(|log M|, 1)
with its relative error unbounded, and 3.5, 8.0, 21, 40, 145 and 805 ulp for
the six orders, each about one and a half times the larger of the two builds'
maxima. The ends of the range are checked in the same file's "normal block at
the ends of the range": the value saturates to +inf below -1.8961e154 and the
derivatives are those of the quadratic alone there (x, 1 and zero).

**Open.** All six orders in one pass would run one recursion instead of six, as
`detail_owen::log_g0_series` does for the Taylor coefficients: worth measuring
if the elastic net's profile shows these calls. The relative accuracy at the
zero of the value would need an expansion of sqrt(pi/2) erfcx(x / sqrt 2) - 1
around x = 0.30246, and nothing asks for it, since the quantity is used
additively.

## Order six everywhere: the incomplete beta, the Student cdf and the skew normal cdf

**What.** Three modules stopped at total order five and now reach six, which
is the order every other derivative family in the library already had
(Giovanni, 2026-11-15):

    own::log_ibeta_d6(a, b, x), own::log_ibetac_d6(a, b, x)   in (a, b)
    own::log_student_cdf_d6(z, nu)                            in (z, nu)
    own::log_sn_cdf_d6(z, alpha)                              in (z, alpha)

each returning `mixed_partials<6>`, whose component c[j] is the derivative with
j differentiations in the second variable. The gamma side
(`incomplete_gamma_deriv`, `incomplete_gamma_mixed`,
`incomplete_gamma_ratio`) reached order six in phase 2 and does not change; no
public derivative family of the library stops at five any more.

**How.** Nothing in the three algorithms is specific to the order. Each
carries its quantities as truncated bivariate Taylor polynomials in one array
with the multi-index (i, j) at (i + j)(i + j + 1)/2 + j, and every step —
the moment-to-cumulant recursion, the truncated product, the composition for
the complementary side, the series, the quadrature's moment accumulation, the
polygamma differences of the log-beta terms — is a loop over the total order.
So the work was the size of the arrays and the per-order pieces that are
written out:

- `detail_ibd::NC`, the number of coefficients of a bivariate polynomial of
  total order at most five, becomes 28 for order six. It is the array size
  shared by all three modules (student.hpp and owen.hpp carry their own
  polynomials in `ibd::NC`-sized arrays and index them with `ibd::ix`), so
  one constant covers them.
- the buffers that hold one value per order, which are indexed 0..K: the
  polygamma differences of `assemble` and of `series_lower`
  (psi^(n)(a) - psi^(n)(s) and its three companions, now n = 0..5, which
  `dpsi_all` already supplied since the discrete families needed psi^(5)),
  the recurrence terms of `psi_all`, the Taylor coefficients of c(nu) and of
  the anchor in student.hpp, the factorials and the binomial table of
  owen.hpp.
- `lower_beta` in student.hpp writes its pure shape coefficients out by
  order, so it gains the order-6 term `pure_a<6>(...) / 720`, which reads the
  incomplete beta at its new order.
- the quadrature of the incomplete beta chooses between two ways of placing
  the moments: storing the nodes and summing them twice about their mean
  (orders 3 and 5) or keeping a first pass at a coarser step (order 4).
  Measured for order six on the 519 reference rows, worst ulpN over the rows
  where the quadrature is used: with the stored nodes 1.30e3 for log I and
  1.16e3 for log(1 - I), without them 1.67e3 and 815. The stored nodes are
  kept, because for log I they set the worst error of the whole order and for
  log(1 - I) the worst comes from the series rows (5.36e3) either way.

**Measured.** References by the routes already in `tools/reference/`, extended
to degree six:

- `incomplete_deriv.py beta6` (new kind) writes the 14 files
  `log_ibeta_d{i}{j}.csv` and `log_ibetac_d{i}{j}.csv` with i + j = 6, 519
  rows, on the same points as the kind `beta` (same seed) so that the two sets
  describe the same grid. It also recomputes orders 1 to 5 at degree six and
  compares them with the committed files: over 10380 values the largest
  relative difference is 1.08e-16 for both sides, which is the check that the
  higher degree did not disturb the lower orders.
- `student_owen.py` now has KMAX = 6, so the kinds `student` and `sn` write
  every order from one to six on their own grids. Here the higher degree left
  the lower orders untouched to the last bit: over the rows the two sets share,
  the recomputed orders 1 to 5 are identical to the committed ones, 15600
  values for the Student and 15980 for the skew normal. One row of the skew
  normal (of 800) no longer passes the agreement of the two working precisions
  at degree six and is dropped; the Student keeps all 780 of its rows.

Worst error over every row and component, relative, in ulps of the
normalization each module uses (ulpN for the incomplete beta, as before):

| quantity | gcc | CUDA | worst component and argument (gcc) |
|---|---|---|---|
| log I_x(a, b), order 6 | 1298 ulpN, 433 ulpL | 1550 ulpN, 433 ulpL | d06 at a = 2756.91, b = 12.6704, x = 0.9978 |
| log(1 - I_x(a, b)), order 6 | 5357 ulpN, 110 ulpL | 1640 ulpN, 176 ulpL | d06 at a = 3.96153, b = 0.0688547, x = 6.4e-18 |
| log T_nu(z), order 6 | 30089 ulpN, 329 ulp1, 2433 ulpL | 19289 ulpN, 274 ulp1, 1560 ulpL | d60 at z = -7.0277, nu = 4.5762 (ulpN); d60 at z = -0.8326, nu = 19174.6 (ulp1) |
| log F(z; alpha), order 6 | 840845 ulpN, 96270 ulpL | 841118 ulpN, 96302 ulpL | d60 at z = -0.8789, alpha = -0.0124 |

For comparison, order five on the same grids measures 1440 and 1950 ulpN
(185 and 342 ulpL) for the two sides of the incomplete beta, so order six
costs a factor of about one to three, which is what each added order has cost
in that module.

The two cdfs' worst points are both the pure z-component near one of its
zeros. The derivatives of order six of log T and of log F cross zero inside
|z| < 1 in the region where the family approaches the normal (nu large,
alpha near zero), as the high derivatives of log Phi do, and there one ulp of
the value is many ulps of the component. The Student's test already measured
the pure z-components of odd order near z = 0 by their absolute error rather
than their relative one (ulps of max(|f|, 1)); that clause now covers order
six as well, which is why the Student's ulpN falls from 1.68e5 to 3.01e4 and
the point that carried it is reported as 329 ulp1 instead. The skew normal's
test has no such clause and keeps its relative measure at every point, as it
does at orders one to five, so its 8.4e5 ulpN is that conditioning and not an
error of the route: the measure, not the computation, is what grows there.

**What this says.** The step from five to six is the arrays and the few
written-out per-order terms, not the algorithms: every route was already a
loop over the total order, and the one real choice, how the quadrature places
its moments at the new order, was measured rather than assumed. The error
grows with the order as it did before, and the references confirm that the
higher degree leaves the lower orders where they were.

**Chosen.** As above. Bounds in the tests, each about one and a half times the
larger of the two builds' maxima: `tests/math/test_beta_deriv.cpp` 2330 ulpN
and 650 ulpL for log I at order 6, 8050 and 265 for log(1 - I);
`tests/math/test_student.cpp` 45200 ulpN, 495 ulp1 and 3650 ulpL at order 6;
`tests/math/test_owen.cpp` 1.27e6 ulpN and 1.45e5 ulpL. The incomplete beta's
two sides differ by a factor of three between the builds at this order (5357
against 1640 ulpN for log(1 - I)), as they already did at order five, so each
bound is set on the build that is worse for it.

**Found on the way.** The three tests and the three benches collected the
components of one row into a buffer of six doubles (`std::array<double, 6>`, a
Kokkos `View<double* [6]>`), which the seven components of order six overran.
It is the kind of defect that shows as an impossible number rather than as a
crash: the first measurements of order six reported infinities and errors of
1e28 to 1e47 ulps, while the same arguments computed in isolation were right
to the last bit. Every one of those buffers is now one longer than the highest
order.

## The beta-binomial and the Poisson-inverse Gaussian: betabinom and pig

**What.** `betabinom_distrib` (mean proportion mu in (0, 1), dispersion sigma > 0, known size n)
and `pig_distrib` (mean mu > 0, shape alpha > 0), the last two of the seven discrete families of
phase 4 (`include/laplace/distrib/betabinom.hpp`, `pig.hpp`). Each writes the log-mass with its six
derivative orders; the Poisson-inverse Gaussian also writes its expected information, which is
diagonal in this parametrization. Neither has a closed cumulative function, so the value of F, the
derivatives of log F and log(1 - F), the quantile and the random numbers come from
`generic_cdf.hpp`, `generic_quantile.hpp` and `generic_rng.hpp`, and the beta-binomial takes its
expected information from `generic_info.hpp` as well, where the finite support makes the sum exact.

### betabinom

**How, the form.** With a = mu/sigma and b = (1 - mu)/sigma,

    l = log C(n, y) + log B(y + a, n - y + b) - log B(a, b),   y = 0, ..., n.

The two log-betas are of the size of (a + b) log(a + b) while their difference is of the size of
one, so neither is ever formed. Both shifts y and n - y are integers, so the recurrence of the gamma
function turns the difference into sums of logarithms, and in those the logarithms of sigma cancel,
because y + (n - y) - n = 0:

    l = log C(n, y) + sum_(i<y) log(mu + i sigma) + sum_(i<n-y) log(1 - mu + i sigma)
                    - sum_(i<n) log(1 + i sigma).

Every term is regular at sigma = 0, where the three sums vanish and the value is the binomial's
log C(n, y) + y log mu + (n - y) log(1 - mu): the binomial limit needs no limit, no series and no
branch. log C(n, y) is taken in Loader's form, (1/2) log(n / (2 pi y (n - y))) + delta(n) - delta(y)
- delta(n - y) + n H(y/n) with delta Stirling's error and n H = -y log(y/n) - (n - y) log(1 - y/n),
as in `binomial.hpp`.

This is written in (mu, sigma) and not in (a, b): the derivatives in (a, b) composed with
a = mu/sigma carry inner factors of the size of sigma^-(K+1) against a result of the size of one, a
relative loss of sigma^-K at the binomial limit, which is the trap negbin1's shape chain has.

**How, the three sums are paired.** They cancel against each other and each is far larger than the
result once the size grows: at (mu, sigma, n) = (0.02, 0.25, 50) and y = 0 the second and the third
are of the size of 100 while the log-mass is 0.4, and at n = 1e4 the three are of the size of 4e4
against a result of the size of 10. Summed each over its own range and subtracted afterwards they
lose that ratio: measured 651 ulp at the first point and 1.75e5 at the second. They are therefore
walked in one pass over i = 0, ..., n - 1 and paired at each i, so that the error of a term is
relative to that term's own contribution. The value takes one logarithm of the product of whichever
of A = mu + i sigma and B = 1 - mu + i sigma is still in range over C = 1 + i sigma, which is also a
third of the cost.

**How, the derivatives.** Each logarithm is a logarithm of an affine function of (mu, sigma), so
with K = p + q derivatives, p in mu and q in sigma,

    d^K l = (-1)^(K-1) (K-1)! [ sum_(i<y) i^q / A^K + (-1)^p sum_(i<n-y) i^q / B^K
                                - [p = 0] sum_(i<n) i^K / C^K ],

the third sum entering the pure derivatives in sigma only, since C does not depend on mu. They are
taken in the same one pass, and two of the pairings remove a cancellation of their own:

- the pure derivatives in mu of odd order need 1/A^K - 1/B^K, whose two terms are equal at mu = 1/2
  and within (1 - 2 mu)/(i sigma) of each other at every i once sigma is large, while their
  difference is of the size of the result: measured **3.9e15 ulp** at order 5, mu = 1/2,
  sigma = 1e3, n = 7, with the two sums taken apart. Since B - A = (1 - mu) - mu is exact and does
  not depend on i,

      1/A^K - 1/B^K = (B - A) sum_(j=0)^(K-1) A^-(K-j) B^-(j+1),

  a sum of terms of one sign, exactly 0 at mu = 1/2;
- the pure derivative in sigma takes i^K (1/A^K + 1/B^K - 1/C^K) at each i, with (i/C)^K formed as
  a power of the ratio.

**The error grows with the size, by a law.** What is left after the pairing is the n + 1 roundings
of the pass itself, and over the reference sizes 1 to 1e4 the worst error follows (n/50)^(3/2)
closely: dividing the worst ulp of each order by that factor leaves between 38 and 163 over every
order and every size. The raw numbers are 1.7 to 63 ulp at n <= 50, 1.5e4 at n = 1e3 and 2.2e5 at
n = 1e4. The test's bound for the mass carries the same factor, so that it stays tight at the sizes
the family is normally used with.

**What was tried and rejected.** Accumulating the sums as products of their positive terms, with one
logarithm taken whenever the running product leaves [2^-512, 2^512]: it saves 11 per cent of the
value at n = 1e4 (342 us against 385) and costs two orders of magnitude of accuracy, because the
relative error of a product of n factors is n eps while n independent logarithms accumulate
sqrt(n) eps.

**The second route, of bounded cost (Giovanni, 2026-11-15: write it).** The sums cost n + 1 terms
and their roundings grow by the law above, so the family also carries a form that costs the same at
every size. Substituting lgamma(z) = (z - 1/2) log z - z + log sqrt(2 pi) + delta(z) in the six
log-gammas of the definition gives, with Delta = y - n mu, A = mu + y sigma,
B = 1 - mu + (n - y) sigma and C = 1 + n sigma, so that A + B = C exactly:

    l = l_binom(mu; n, y)                                 the binomial log-mass, in Loader's form
        + Delta [rho(w) - rho(v)],   rho(t) = D(1 + t)/t,
          w = Delta sigma / (mu C),  v = -Delta sigma / ((1 - mu) C)
        - (1/2) [log A - log mu + log B - log(1 - mu) - log C]
        + [delta(A/sigma) + delta(B/sigma) - delta(C/sigma)]
        - [delta(mu/sigma) + delta((1 - mu)/sigma) - delta(1/sigma)].

Nothing in it is a difference of large terms. l_binom is Loader's own form (`binomial.hpp`), so the
pair log C(n, y) and y log mu + (n - y) log(1 - mu), each of the size of n log n and cancelling to
the size of one, is never formed. The two deviances are
[bd0(A, mu C) + bd0(B, (1 - mu) C)] / sigma, taken through rho because 1 + w = A/(mu C) and
1 + v = B/((1 - mu) C) exactly, and w and v have opposite signs, so the difference of the two rho is
a sum of magnitudes. Every logarithm is a logarithm of an affine form. The two bracketed groups of
Stirling errors are each of the size of sigma and their difference of the size of sigma^2, which is
an absolute error of the size of eps and never a relative one.

The derivatives are closed except for the two rho: d^(p+q) log A =
(-1)^(p+q-1) (p+q-1)! y^q A^-(p+q) and likewise for the other four logarithms; w = (y/mu - n) g and
v = ((n-y)/(1-mu) - n) g with g = sigma/C are each a function of mu times a function of sigma, so
their partials are the product of the two factors'; and Delta is affine, so
d^(p+q)(Delta F) = Delta F_(p,q) - n p F_(p-1,q). The Stirling errors are taken as the asymptotic
series of `stirling.hpp` in the SMALL variable x = sigma/z, delta(z/sigma) = sum_j s_j x^(2j-1) with
s_j = B_2j / (2j (2j-1)), and differentiated term by term in closed form, which is why the route
holds only where that series does, that is where all six arguments are at least 10:
**10 sigma <= min(mu, 1 - mu)**. Composing delta's own derivatives with z/sigma instead would carry
inner factors of the size of sigma^-(q+1) against a result of the size of one, the same trap the
(a, b) chain has; outside the region the series diverges and the route gives 1e36 to 1e180 ulp, so
the guard is not a convenience.

**Checked before transcription.** The six log-gammas cancel between the two sides, so what is left
is an identity among logarithms; SymPy proves that one at three rational mu (it does not pull the
constant out of log(c x + c), so the check is on the ratio of the two sides). The whole identity is
then checked against the definition at 60 digits over 90 points, 6.6e-55 relative at worst, with
non-integer y and n among them, which also shows that it needs no integer shift.

**Which route each quantity takes.** The two routes' costs cross at a different size for the value
than for a tensor (ns per call on one thread, sums against Stirling): the value at n = 8
(366/361 there, 3230/253 at n = 160), the tensors at 120, 250, 420, 570, 700 and 800 for the orders
1 to 6, the Stirling side costing 214, 460, 1040, 2200, 3400 and 6600 ns at every size. The cut is
therefore n = 8 for the value and n = 160 for every derivative order. Above order 1 that is set by
accuracy and not by cost: at n = 160 the sums have already reached about 940 ulp by the law above,
where the Stirling route stays near 65 ulp at every size, and the price is at most five times the
cost of a tensor between n = 160 and n = 800. Where both routes hold they agree to 1e-14 to 1e-10
of the scale of the order (a test case, read as the Stirling route's own check below n = 50, where
the sums are the accurate side, and as the sums' loss above it).

**What it is worth.** Measured against the same references, in ulps of the largest component of the
order: at n = 1e4, sigma = 0.01 the sums give 7.9e4 to 2.2e5 ulp and this route 64.6; at n = 1000,
51.6 against 1.5e4; at n = 40 and n = 12, where the sums are already good, 4.11 and 10.2 against
44.6 and 6.98. The value's cost falls from 178 us to 230 ns at n = 1e4 and the order-6 tensor's from
63 to 6.5 us. A size of 1e5 is now in reach at all.

**Two measurements inside the route.** (i) The composition it needs is of a function of ONE
variable with one map of the two parameters, and `chain_rule_pair_dk` pays for the general case:
`chain_rule_pair_scalar_dk` (chain_pair.hpp) replaces the multi-index over the outer arguments by
one order, which takes the state from 210 to 84 doubles and one loop dimension away, and the
order-6 composition from 3.6 to 0.7 us. The two agree to 1e-14 of the scale of the order, not bit
for bit, because the scalar one takes the two binomial coefficients out of its innermost loop.
(ii) A local table of the binomial coefficients in that innermost loop was measured and dropped: 49
doubles have to be written to the stack at every call, where the product form stays in registers,
and the order-6 composition went from 6.9 to 44 us.

**What is open.** The region of the second route is set by the asymptotic branch of Stirling's
error, so above 10 sigma = min(mu, 1 - mu) only the sums are available, and there their error still
follows the law above: at n = 1000 and sigma = 1 it is about 1.5e4 ulp. The generic sums over the
support are sums of sums, so the expected information and the cumulative derivatives cost n^2 where
the sums route is taken: at n = 100 the information is 400 us on one thread.

### pig

**How, the form.** With s = sqrt(mu^2 + alpha^2),

    p(y) = sqrt(2 alpha / pi) e^(s - mu) (mu alpha / (mu + s))^y K_(y - 1/2)(alpha) / y!,

mean mu and variance mu + mu^2/(s - mu). The order of the Bessel function is always half an odd
integer, so it is elementary: with m = max(y - 1, 0),

    K_(y-1/2)(alpha) = sqrt(pi / (2 alpha)) e^-alpha sum_(k=0)^m (m+k)! / (k! (m-k)!) (2 alpha)^-k,

a sum of positive terms, checked against mpmath's own Bessel function for y = 0..24, 40 and 60 and
alpha from 1e-4 to 1e4 (worst 4.6e-60 relative, at 60 digits). Its square root of alpha cancels the
mass's exactly, and the powers of alpha of the sum join the (mu alpha)^y, which leaves one
exponential family in theta = log(2 alpha) over the shifted index j = y - k, which runs from 1 to y:

    l = (s - mu - alpha) + y log mu - y log(mu + s) - y log 2 + log Q(theta) - loggamma(y + 1),
    Q(theta) = sum_(j=1)^y a_j e^(j theta),   a_j = (2y - 1 - j)! / ((y - j)! (j - 1)!),

with Q = 1 at y = 0, where the mass is e^(s - mu - alpha) exactly. **So no Bessel routine, no
uniform (Debye) expansion and no generated table enters this family**, and the open point of the
Bessel section, a closed form for half-integer orders which the Poisson-inverse Gaussian needs, is
closed by not needing one.

The terms are walked outwards from their largest, whose index j* is the root in [1, y] of
j^2 - (2y - 1 + 2 alpha) j + 2 alpha y = 0, by the exact ratio
a_(j+1) e^theta / a_j = 2 alpha (y - j) / (j (2y - 1 - j)), and the walk stops when a term falls
below 2^-80 of the largest. j* is about alpha and the width of the weights about sqrt(alpha), so
**the cost does not follow the count**: measured flat from y = 1 to y = 1e6 (table below).

**How, the value.** Written as log Q - loggamma(y + 1) the mass keeps terms of the size of y log y
while it is itself of the size of y log((mu + s) / (2 mu)), which the heavy-tailed end makes small:
measured 2.6e5 ulp at (mu, alpha, y) = (1e3, 1e-3, 1e6). The four log-gammas of a_(j*) and of y! are
therefore taken through Stirling's error: their linear parts sum to j* + 2, the coefficients of
log y to -3/2, and the coefficient of log 2 to -(j* + 1/2) once 2y log 2 is taken out, which is
exactly what the remaining y log(2 mu / (mu + s)) needs. What is left of the four arguments'
logarithms is three log1p of arguments of the size of j*/y, and the one term of the size of y is
-y log1p(alpha^2 / (2 mu (s + mu))):

    l = (s - mu - alpha) + (j* + 2) - log(2 pi)
        + delta(2y - j*) - delta(y - j* + 1) - delta(j*) - delta(y + 1)
        - (3/2) log y - (j* + 1/2) log 2 - (j* - 1/2) log j*
        + (2y - j* - 1/2) log1p(-j*/(2y)) - (y - j* + 1/2) log1p((1 - j*)/y)
        - (y + 1/2) log1p(1/y) + j* log(2 alpha)
        - y log1p(alpha^2/(2 mu (s + mu))) + log W,

W the sum of the terms scaled by the largest. The constant s - mu - alpha is the log-mass at y = 0
and cancels at both ends, so it is taken as mu^2/(s + alpha) - mu when alpha >= mu and as
alpha^2/(s + mu) - alpha otherwise: in each the positive term is at most half the negative one. With
the whole grouping the worst error on the grid of the measurement above falls from 2.6e5 ulp to
3.8 ulp.

**How, the derivatives in the mean.** The score in the mean is exactly

    dl/dmu = c (y - mu),   c = 1/mu - 1/s,

the terms of the size of y cancelling identically, which is also why I_mu_mu = c. The first order
keeps that factored form, which is exactly 0 at y = mu. Above it the product is differentiated
expanded instead,

    d^(p+q) l / dmu^p dalpha^q = y c_(p-1,q) + m_(p-1,q),   m = mu / s,   p >= 1, p + q >= 2,

because the factored form (y - mu) c_(p-1,q) - (p-1) c_(p-2,q) splits a result of the size of
alpha^-(p-1) into terms of the size of one at the Poisson limit: measured 4.1e15 ulp on the order-6
tensor at (mu, alpha, y) = (1, 1e3, 0), where the whole tensor is d^K s / dmu^K. The two forms are
algebraically the same, since mu c_(p-1) + (p-1) c_(p-2) = -m_(p-1).

Both c and m are functions of the one ratio alpha / mu, and which of its two powers is the small one
decides the composition, because an inner map that is large makes every term of the chain large
while the result is not. With h(x) = 1 - (1 + x)^(-1/2) and g(x) = (1 + x)^(-1/2) = 1 - h(x), whose
derivatives are g^(n)(x) = (-1)^n (2n)!/(4^n n!) (1 + x)^(-1/2-n), the split is at alpha = mu:

    alpha <= mu:  u = (alpha / mu)^2 <= 1,  c = h(u) / mu,           m = 1 - h(u),
    alpha >  mu:  v = (mu / alpha)^2 < 1,   c = 1/mu - g(v) / alpha,  m = (mu / alpha) g(v).

h's value is written x / (w (w + 1)), w = sqrt(1 + x), because 1/mu - 1/s loses everything when
alpha is much smaller than mu. Taking the first branch everywhere gave 6.8e12 ulp at the same
Poisson-limit point, every term of the chain for d^5 m / dmu^5 being of the size of 1e-3 against a
result of 4.5e-14. In each branch the inner maps are products of a power of one parameter and a
power of the other, so their partials are closed, and `chain_rule_pair_dk` composes them once per
order.

**How, the derivatives in the shape.** The pure derivative in alpha is
s_(0,q) - [q = 1] - y (log(mu + s))_(0,q) plus the derivatives of log Q. Those are cumulants:
d^q log Q / dtheta^q is the q-th cumulant of j under the weights a_j e^(j theta) / Q, taken from the
central moments of the same walk (kappa_1 = <j> >= 1, so nothing cancels inside that block), and
theta = log(2 alpha) composes them into alpha. Above the first order the derivatives of s are taken
as d^q s / dalpha^q = mu^2 d^(q-2) (mu^2 + alpha^2)^(-3/2) / dalpha^(q-2), which is free of the
cancellation that 1/s - alpha^2/s^3 has when alpha is much larger than mu, and at the first order
alpha/s - 1 is taken as -mu^2 / (s (s + alpha)).

At the first order two of those pieces cancel against each other above the split, kappa_1 / alpha
against y alpha / (s (mu + s)), which both tend to y / alpha as alpha grows. Since alpha^2 =
s^2 - mu^2 and s (mu + s) - alpha^2 = mu (s + mu), the second is exactly (y / alpha) (1 - mu / s),
so the difference of the two is (kappa_1 - y) / alpha + y mu / (alpha s), where kappa_1 - y is
formed as (j* - y) + m_1 / sum and both of those are exact. That form is taken where alpha >= mu
and nowhere else: there it leaves 2.6 ulp where the first form had 1577, and below the split it
divides kappa_1 - y, which is then of the size of y, by a small alpha and carries the walk's own
error up instead (1.2e4 ulp at y = 949, mu = 100, alpha = 5, against 74.4 for the first form). The
worst of the order over the whole reference file is therefore 74.4 ulp, at that point below the
split.

Below the split m = 1 - h(u) has no second factor, so its outer function is of one variable and
`chain_rule_pair_scalar_dk` composes it instead of `chain_rule_pair_dk`, which takes about a
quarter off the order-6 tensor there: 3.55 us against 4.84 with the two headers compiled into the
same probe one after the other, every component unchanged to the digit. Above the split, where m
carries the factor mu / alpha, the pair form is the one that applies and nothing changes, which the
same probe confirms (5.62 us against 4.76, one run against the other, the difference being the
machine). The absolute figures of the speed table below were measured on a quiet machine in another
session and are not re-measured here.

**The expected information.** Measured with mpmath at 40 digits by summing the mass:
I_mu_mu = 1/mu - 1/s = c exactly and I_mu_alpha = 0 exactly (the off-diagonal block 5.2e-28 to
4.7e-44 of I_mu_mu at four (mu, alpha) with alpha >= mu, where the cut support sum holds the second
moment). The member therefore supplies both of those closed, the derivatives of I_mu_mu being the
partials of c already written, and takes only I_alpha_alpha and its derivatives from a sum over the
support.

**I_alpha_alpha, and why the generic sweep is not enough for it.** The sum of
(dl/dalpha)^2 p(y) over the support is dominated by the small counts, half of it at y <= 1 and 98
per cent at y <= 1000, but its terms then fall as a power of y until the geometric factor
e^(-lambda y), lambda = log1p(alpha^2 / (2 mu (s + mu))), takes over at counts of the size of
mu^2 / alpha^2. The generic sweep stops a direction when two neighbouring terms are below 2^-60 of
the running sum of absolute values, so it needs terms of that order: measured 444 at alpha = mu,
17.4e3 at alpha / mu = 0.1, 1.6e6 at 0.01 and 1.7e7 at 0.003, which is 0.4 ms, 6 ms, 0.5 s and 6 s
a call, and at alpha / mu = 1e-3 and below it passes its cap of 2^26 terms a direction and reports
no convergence, so the component is NaN. At (mu, alpha) = (40, 1e-2) it does worse than give up: it
stops early on an underflowed mass and reports convergence 4.2e-5 away from the value.

Above a cut the count can be treated as a real number instead. The half-integer polynomial of the
mass needs an integer, so the tail calls the Bessel function as it stands (`bessel_k`, whose
uniform branch covers the large orders) and takes the score from its derivative in the argument:

    log p(y) = log(2 alpha / pi) / 2 + (s - mu) + y log(mu alpha / (mu + s))
               + log K_(y - 1/2)(alpha) - loggamma(y + 1),
    dl/dalpha = 1/(2 alpha) + alpha / s + y (1/alpha - lambda') + d log K_(y - 1/2)(alpha) / dalpha,

with lambda' = alpha / (s (mu + s)). Against the family's own logpmf and logpmf_d1 at integer counts
up to 3000 those two agree to 3.1e-13 and 3.7e-13 of the value, which is the twin check of the
whole construction. The integrand is then smooth in a real count, so the sum over the integers above
y_cut = 1024 is the integral from y_cut + 1/2, taken by `exp_sinh_quad` with the decay length
1 / lambda, plus the first Euler-Maclaurin correction f'(y_cut + 1/2) / 24, the midpoint rule's own
term, whose derivative is taken from the tail model A y^(-3/2) e^(-lambda y) (a - lambda' y)^2 with
a = 1/alpha + alpha / s. Three digits of that model are enough for a term that is itself 1e-5 of the
tail: with it the result moves from 5.3e-10 to 1.6e-11 of the value, measured against the same sum
cut at 16384, and the quadrature's own error estimate is never above 5e-21 of the result.

The member takes this route where the spread is above 4 and the generic sweep below it. The cut
route costs 0.42 ms whatever the parameters, the sweep 0.09 ms at a spread of 1.8, 0.27 at 3.1, 0.70
at 4.9 and 6.0 at 14, so between a spread of 4 and about 6 the sweep can still be the faster of the
two by up to a factor of two (0.25 ms against 0.43 at (mu, alpha) = (3.5, 2)), and the single number
is kept because above the switch the cut route is the only one of the two that stays right.

**The derivatives of I_alpha_alpha take the same two routes.** Every partial of log p in
(mu, alpha) is available at a count that is not an integer: mu does not enter the order of the
Bessel function, so the components with an index in mu are the family's own Leibniz form, which is
an identity in y, and the pure ones in alpha are the real-count form's own, with the derivatives of
log K in alpha to order 7 from `bessel_k`. Against the family's own partials at integer counts to
order 5 the two agree to 1.9e-12 of the scale of the order. The integrands are then the generic ones
of generic_info.hpp in the score form, taken from the family's partials below the cut and from the
real-count ones above it. These components carry powers of y, so their tails weigh far more than the
value's, from 1.6 per cent to ninety times the component itself, and the integral is taken without
the Euler-Maclaurin term the value uses: against the same route cut at 16384 the difference is
2.5e-8 of the scale of the order at worst, at order 4 and (mu, alpha) = (1, 0.1), and 5e-10 on most
components, which is that missing term.
Against the generic sweep where it still converges the two agree to 1.9e-15 of the scale at a spread
of 5.2 and to 2.5e-8 at (mu, alpha) = (1, 0.1), where the sweep itself needs 1.6e5 terms. What it
costs: at (10, 0.1) the order-4 tensor is 7.3 ms against the sweep's 11.0 s, and at (1, 1e-3), where
the sweep gives up after 310 s, 12.6 ms. At a spread between 4 and about 6 the cut route is again
the slower of the two, 6.7 ms against 4.2 at (3.5, 2).

**What is open.** (i) The reference file of the mass holds
the mass only, for the reason in the next section; I_alpha_alpha has a reference file of its own,
written by the same script, and no reference exists for its derivatives.

### Forms checked before transcription

`tools/reference/distrib_betabinom.py` and `tools/reference/distrib_pig.py` check every form with
SymPy or against mpmath before it is written in C++, and exit non-zero on any disagreement:

- the shift identity and its derivatives to order 6, and Loader's form of log C(n, y) (SymPy,
  exactly; the recurrence of the gamma function has to be applied by hand, since SymPy does not
  reduce loggamma(z + k) - loggamma(z) on its own);
- the half-integer Bessel identity, against mpmath's own Bessel function (4.6e-60 relative);
- the Q form of the mass for y = 0, 1, 2 (SymPy, exactly; on the masses and not on their
  logarithms, since SymPy reduces e^alpha (cosh alpha - sinh alpha) to 1 only inside a product);
- the score in mu, dl/dmu = c (y - mu) (SymPy, exactly), and the Leibniz form of the derivatives
  with an index in mu (orders 1 and 2 symbolically, orders 3 to 6 within 1.5e-191 relative at 200
  digits on five (mu, alpha); the points keep |log(alpha/mu)| moderate, because c itself cancels
  like (mu/alpha)^2 per order, which is what the split above exists for);
- h with its derivatives, c = h(u)/mu, and the partials of the two inner maps (SymPy, exactly);
- the cumulants of Q to order 6 and their conversion from the central moments (SymPy, exactly);
- the two information identities above (mpmath at 40 digits).

### References and the measured accuracy

`distrib_betabinom.py` writes 4072 rows (kind 0 the log-mass and its six orders, 2490 rows; kinds
1 and 2 log F and log(1 - F) by summing the mass, 1582 rows; fifteen (mu, sigma, n) with sigma from
1e-6 to 1e3, mu from 0.02 to 0.98 and n from 1 to 1e5, five of them inside the region of the route
of bounded cost and on the two sides of its cut) and `distrib_pig.py` 1532 rows (the mass only; ten (mu, alpha)
with alpha from 1e-3 to 1e3 against mu from 1e-3 to 1e4, and counts up to 3000). The mass rows are
mpmath evaluations of the symbolic derivatives of the definitions, so no form of the section above
enters them. The pig has no reference for its cumulative functions: without a closed form each
value would be a numerical derivative of a sum of Bessel functions at a raised working precision,
measured at about a second a component at order 4 on the light-tailed parameters and without bound
at the heavy end; what the library computes there is the generic summation route, measured in its
own section, and the test checks it against 5-point differences to order 3. For the same reason
betabinom's cumulative rows stop at n = 50. mpmath's own Bessel function is slow where its order
and its argument are close (6.5 s for one value at (1e3, 1e3, y = 1e3) against 0.03 s at y = 1175),
which is a property of the reference route and not of the library's, whose cost does not depend on
y at all; the counts in that band are left out of the grid.

**The twin of the cumulative quantities.** What replaces the missing reference tables for these two
families is a twin inside the tests, `check_discrete_cdf_twin` in `tests/distrib/check_common.hpp`,
which shares nothing with the route it checks: it walks the whole support once in plain
probabilities and accumulates the lower side from the bottom and the upper side from the top in two
compensated passes, so that neither tail is formed as a difference of the other, where the library
sums the cheaper of the two tails in units of its own first mass, stops when the neighbouring terms
fall below 2^-60 of the running sum, and takes the side it did not sum as log1p(-exp). The
quadrature of `generic_cdf.hpp` cannot serve as the twin here: a static_assert restricts it to
continuous distributions, a mass on a lattice being no integral. All four quantities are compared,
F, 1 - F, log F and log(1 - F), at five (mu, sigma, n) for betabinom, two of them with the mass at
one end of the support, and at five (mu, alpha) for the pig, from a tail that decays by a factor of
2.6 a step to one that decays by 0.25 per cent.

Two properties of the twin had to be measured before it said anything about the library. The cost
is quadratic in the support if every lattice point is taken, because each call of the route under
test sums a tail of its own: at (mu, alpha) = (3, 0.3) the pig's walk is 1.6e4 terms and the suite
had not finished in fifteen minutes, so the points are taken in a stride that leaves about sixty of
them, the last one included. And the truncation of the walk is not negligible against a small upper
tail: stopping it where the geometric bound v r / (1 - r) on the terms not taken fell below 1e-18
showed up as a 5.4e-5 relative difference on an upper tail of 1.4e-14, and a margin of 1e8 over
that bound left 9.6e-9 on one of 5.4e-18, where the route under test agrees with the same sum
accumulated in long double to 2.4e-15. The walk now runs until the mass itself is below 1e-200, or
until its cap of 2^16 terms, and an upper tail is compared only where the bound is below the
tolerance times the tail. The bound is set to zero on a finite support, where nothing is left out.

What it measures. betabinom: 3.13e-15 relative at worst, the same on gcc and CUDA, over the 140
points of the five sizes from 20 to 50 (test bound 1.3e-14). The pig: 9.8e-10 on gcc and 2.03e-9 on
CUDA, both far out in the tail of (mu, alpha) = (10, 0.5), around y = 3.4e4, where the walk stops at
the cap and the two builds differ by a factor of two; everywhere else 2.9e-11 to 2.8e-10, with the
two builds equal to the digit (test bound 8e-9, four times the worse of the two). The derivatives of the two cumulative functions keep
the 5-point differences they had, to order 3.

The error is the library value minus the mpmath value, in ulps of the largest component of the same
order at the same point, which is how every family with genuine zeros in its tensors is measured.
Worst ulp over the whole file, per order. betabinom is read per route, because the two fail
differently: where the sums are taken the error follows the size by the law above, so the figure is
divided by (n/50)^(3/2); where the form of bounded cost is taken it does not follow the size at all,
so the figure is raw.

| order | betabinom sums, / (n/50)^(3/2) | betabinom Stirling, raw | pig |
|---|---|---|---|
| 0 (the value) | 63.3 | 520 | 6.24 |
| 1 | 98.2 | 1.83e3 | 74.4 |
| 2 | 37.9 | 156 | 5.81 |
| 3 | 48.3 | 23.6 | 10.4 |
| 4 | 37.8 | 27.0 | 19.7 |
| 5 | 28.3 | 22.7 | 99.6 |
| 6 | 163 | 24.1 | 201 |

The sums' raw worst is 63.3 ulp at n <= 50 and 1.5e4 at order 6 where the family is outside the
Stirling region at n = 1000, and the law covers both. Every one of the Stirling route's own worsts
above is at n = 1e5, sigma = 0.01, and the three that are not small are the value and its first two
orders: that route splits the log-mass into the binomial's and a correction, and in the tail of a
large size the two are each of the size of n while the log-mass is not. At n = 1e4 the whole tensor
is within 64.6 ulp, so this is a corner of n = 1e5 and not of the route. The pig's order-1 figure is the one
below the split named above; every other order is below 201 ulp at counts from 0 to 3000 and shapes
from 1e-3 to 1e3. The two builds agree everywhere within a few ulp, so the table
gives one figure for both. betabinom's cumulative rows, which measure the generic summation route
and not this family, run from 34 ulp at order 0 to 4.1e6 at order 6, the worst at mu = 0.02 and
mu = 0.98, where the mass sits at one end of the support.

**The reference of I_alpha_alpha.** `tests/data/reference/pig_info.csv` holds 18 rows,
(mu, alpha, hi, lo) as hexadecimal doubles, written by `tools/reference/distrib_pig.py --info`:
mpmath sums the same definition at 40 digits with its own Bessel function of real order, through
the identity d log K_nu / dalpha = -(K_(nu-1) + K_(nu+1)) / (2 K_nu), and accelerates the tail with
its own Euler-Maclaurin summation, so neither the half-integer polynomial nor the library's
quadrature enters the reference. The rows run from a spread of 0.58 to 4.5e4, that is from
alpha / mu = 1.5 down to 1e-4, and cover both of the member's routes, four of them the sweep and
fourteen the cut. The Poisson limit is left out for the same reason the mass rows avoid that band:
mpmath's Bessel function is slow where its order and its argument are close, and at alpha = 1e3 the
summation does not finish. The member measures 1.36e-11 of the value at worst against those rows,
at (mu, alpha) = (10, 1e-3), with every other row below 4.5e-12; the four rows of the sweep are
within 1.9e-15. The test bound is 2.1e-11.

### Speed

ns per call on one host thread (Serial) and on four OpenMP threads, from
`benchmarks/distrib_betabinom.cpp` and `benchmarks/distrib_pig.cpp` (Serial figures):

| quantity | betabinom n = 10 | n = 100 | n = 1e3 | n = 1e4 | pig(3.5, 2) | pig(1, 1e3) |
|---|---|---|---|---|---|---|
| log p | 520 | 3709 | 36070 | 359906 | 252 | 271 |
| log p, order 1 | 18.5 | 198 | 1803 | 18150 | 22.6 | 15.6 |
| log p, order 6 | 41.6 | 384 | 3695 | 37319 | 2138 | 2111 |
| log F (generic sum) | 1889 | 96287 | | | 78222 | 3710 |
| information (generic sum) | 6150 | 399100 | | | 242655 | 11296 |

The pig's cost does not follow the count, which is what the walk from the largest term buys: at
(mu, alpha) = (10, 1) the value costs 553, 320, 313, 309, 303, 304 and 307 ns at y = 1, 10, 1e2,
1e3, 1e4, 1e5 and 1e6, and the order-6 tensor 2131 to 2232 ns over the same counts. For scale,
negbin2's order-6 tensor is 16 us. betabinom's cost is proportional to n in every quantity, and the
generic sums over the support are sums of sums, so they cost n^2.

## The cdf of a continuous distribution by quadrature of its density

**What.** `cdf_by_integral(d, y)`, `ccdf_by_integral`, `log_cdf_by_integral`,
`log_ccdf_by_integral` and the derivatives in the parameters
`log_cdf_dk_by_integral<K>(d, y)`, `log_ccdf_dk_by_integral<K>` for K = 1..6
(`include/laplace/distrib/generic_cdf.hpp`), the route that
`note/distrib.md` section 4 foresaw for a continuous distribution with no
closed cdf. The dispatchers `distrib_cdf`, `distrib_ccdf`, `distrib_log_cdf`,
`distrib_log_ccdf`, `distrib_log_cdf_dk<K>` and `distrib_log_ccdf_dk<K>` now
take the distribution's own member when it has one, the sum over the mass when
it is discrete, and this quadrature when it is continuous.

**How.** The mirror of the discrete walk. The side of y that does not hold
`center()` is integrated, and the density at the end point y is divided out, so
that a tail far below the double range still gives its logarithm:
log S = l(y) + log W with W the integral of f / f(y). The other side is the
complement on the log scale. For the derivatives,

    d_T F = int f B_T(l),

with B_T the complete Bell polynomial of the derivatives of l over the
multiset T, built on the derivatives centred at y; the composition to log F
reuses `log_tail_dk_from_ratios` and `ratios_from_log_tail_dk`, the same two
relations the discrete route uses. A component whose parameter has finite
regularity is NaN: the parameter moves a kink of l in y, and differentiating
the integrand across it would leave a boundary term.

One quadrature serves every component. The integrands are the Bell polynomials
of every multi-index of order at most K, in the layout of
`log_density_partials` (84 of them for three parameters at order six), so the
density's six derivative tensors are evaluated once per node instead of once
per component; and the index arithmetic of the Bell recursion is hoisted into a
table of single bytes, built once per call, which leaves the per-node work a
sequence of multiply-adds.

**Measured.** Against the closed forms of the two families that write out log F
and its six derivatives, the cauchy and the logistic, over sigma = 0.25, 1 and
7 and y from -40 to 25 on both sides, as the error over max(|f|, 1) (the pure
derivatives in the scale vanish at y = mu, where a measure relative to the
component itself says nothing):

| quantity | cauchy | logistic |
|---|---|---|
| log F and log(1 - F) | 5.6e-16 | 4.4e-16 |
| order 1 | 1.8e-15 | |
| order 2 | 1.4e-14 | |
| order 3 | 5.7e-14 | |
| order 4 | 9.1e-13 | |
| order 5 | 2.2e-11 | |
| order 6 | 5.8e-10 | 1.5e-9 |

The value is exact to rounding into the deep tails (y = -300 at sigma = 0.25,
where F is far below the double range), and the derivatives lose about one
digit and a half per order, which is the conditioning of the Bell composition
and not of the quadrature. The figures are the host's, since the comparison is
between two closed-form routes evaluated there; the route itself is exercised
on the device by the kernel case of the same test, where it compiles and runs
on both builds.

Cost per call on one thread, and what each step of the design bought:

| order | one quadrature per component | one pass, indices computed per node | one pass, index table |
|---|---|---|---|
| 1 | 110 us | 66 us | 72 us |
| 2 | 938 us | 320 us | 424 us |
| 4 | 9.5 ms | 2.4 ms | 2.9 ms |
| 6 | 94 ms | 56 ms | 34 ms |

**What this says.** The accuracy is that of the identity, not of the
arithmetic: the same numbers came out of all three versions, to the last digit.
The cost is dominated by the density's own derivative tensors at every node,
which is why one pass over the components bought 1.7 times at order six and
hoisting the index arithmetic another 1.6: neither touches the number of nodes,
and the rule refines to its maximum level because some components never pass
their own stopping test. Those are the high pure derivatives in the scale in
the far tail, where the Bell polynomial carries powers of z that cancel; they
are accurate to 1.9e-8 there, so the value is returned rather than a NaN, which
would poison the whole tensor. A further factor needs fewer nodes, that is the
rule's options threaded through `panels_vec` and a stopping test on the
dominant components; it is not written, because at 34 ms per call the route is
a reference and a fallback, not something a fit calls per observation.

**Chosen.** As above, with the test
`tests/distrib/test_generic.cpp` ("continuous cdf family by quadrature of the
density"): the value within 8 ulps of max(|log F|, 1) against both families,
the derivatives within about three times the measured error, through the
dispatcher, and the route run inside a Kokkos kernel in the same file's case
"every route inside a kernel". The regularity of a parameter is read through
`regularities_of<D>()`, the helper the expected information already had, since
a static constexpr array member cannot be indexed in device code: nvcc refuses
`D::params[j]` there, and the gcc build does not.

**Open.** The number of nodes, above. The same one-pass shape would serve the
discrete route, whose 103 us at order six come from the same 28 walks over the
support; it is unchanged for now, since its bounds and its tests are measured
on the per-component version. At four parameters the pass needs 210 integrands,
which is a good deal of local memory on the device: to measure when the skew t
asks for it.

## enet, the elastic-net distribution

**What.** `enet_distrib{mu, lambda, alpha}`
(`include/laplace/distrib/enet.hpp`), the density whose negative logarithm is
the elastic-net penalty: a Laplace and a Gaussian at the same location,
multiplied and normalized. With z = y - mu, a = lambda alpha,
c = lambda (1 - alpha), x = a / sqrt(c) and M the Mills ratio,

    f(y) = exp(-a |z| - c z^2/2) / Z,   Z = 2 M(x) / sqrt(c),
    l = -a |z| - c z^2/2 - log 2 + log(c)/2 - log M(x),

with mu on the whole line, lambda > 0 and alpha in (0, 1); at alpha -> 1 the
family is the Laplace and at alpha -> 0 the Gaussian, both of them families of
their own. Every member is closed: the log-density and its six orders, the
derivatives in y and the mixed ones, the expected information and its four
orders, the cdf family with six orders, the quantile, the descriptors and the
starting value. Only the draw is inherited (`rng_by_inversion`).

It is the one family of this batch whose log-density is not differentiable in
the location: l is C^0 in mu with the kink at y = mu, and log F and log(1 - F)
are C^1 there, their second derivative jumping by 4 a f(mu) across the
location (d^2 log F / dmu^2 = f'(z)/F - (f/F)^2, F(mu) = 1/2 by symmetry, and
f'(z) = f (-a sgn(z) - c z) jumps by -2 a f(0)); at the Laplace limit
f(mu) = a/2 and the jump is 2 a^2, which is the 2/sigma^2 of
`note/distrib.md` point D with sigma = 1/a.

**How.** Everything rests on log Z and its derivatives in (a, c) to order six,
carried to (lambda, alpha) by the chain rule through the bilinear map. Two
forms, switched at x = 1.5 (measured at x = 0.5, 1, 1.5, 2 and 3: the second
loses digits below x = 1, the first above x = 2, and 1.5 gave the smallest
worst error at orders 5 and 6):

- x < 1.5: log Z = log 2 + log M(x) - log(c)/2, from `log_mills_d1 .. _d6`
  composed with x(a, c) by `chain_rule_pair_dk`;
- x >= 1.5: log Z = log 2 - log a + F(t) with t = c/a^2 and F = log(x M(x)),
  whose m-th derivative in t is the m-th cumulant of -U^2/2 under the same
  density as the Mills ratio, from the downward ratio recursion of
  `detail_normal::mills_cumulant` in scaled form.

The expected information is closed and must not come from -E[l_ab], which
misses the point mass at the kink: I_mumu = c + 2 a exp(-log Z), the
(lambda, alpha) block is read off the derivatives of log Z
(I_ll = A_ll, I_la = A_la - A_a/lambda, I_aa = A_aa with A = log Z), and
I_mu,lambda = I_mu,alpha = 0 by symmetry, as is every derivative with an index
in mu. The derivatives of I_mumu are written in the two forms
c (1 + x/M(x)) and a^2 (t + exp(-F(t))) rather than through exp(log Z), which
cancelled the pure powers of alpha exactly and lost 4e9 ulp at
(lambda, alpha) = (1e6, 0.5).

The cdf is closed, with the tail beyond D = |z| as exp(T),

    T = -a D - c D^2/2 - log 2 + log M(x + sD) - log M(x),   s = sqrt(c),

so that the normalizer's own table serves the cdf as well. The form through
log Phi, log Phi(-(sD + x)) - log Phi(-x) - log 2, was tried and dropped: near
alpha = 1 it subtracts two terms of size x^2/2 and keeps nothing. T <= -log 2
always, which leaves log1p(-e^T) as the only branch of log(1 - e^u) that is
needed. The quantile is Newton's method on a D + c D^2/2 - (log M(x + sD) -
log M(x)) = -log(2 p), whose left side is convex and increasing and whose
start, the root of the quadratic part, lies to the right of the root; the
closed form through `Phi_inv_logp` was dropped for the same reason as the cdf's.

**Measured.** `tools/reference/distrib_enet.py` writes 35595 rows
(`tests/data/reference/distrib_enet.csv`): 17 parameter points, 12 values of y
for the cdf and 4 for the density, the orders 0 to 6, the information to order
4 and 9 quantile probabilities, each row computed at 500 bits with step 2^-40
and at 600 bits with step 2^-48 and kept when the two agree to 2^-80. The
quantities come from the definition (mpmath's `ncdf` for Z, the closed tail,
the moments by integration by parts), checked against `mpmath.quad` at the
start of the script. Units: ulp of the largest component of that order at that
point. The gcc and CUDA tables are identical to the digits kept.

| quantity | order 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| log f | 157 | 2.9 | 5.5 | 24 | 116 | 1650 | 2030 |
| log F | 478 | 493 | 1800 | 4940 | 2790 | 9570 | 16500 |
| log(1 - F) | 628 | 965 | 1800 | 4940 | 2790 | 9570 | 16500 |
| the information and its derivatives | 19.5 | 31.7 | 116 | 412 | 7.8e5 | | |
| quantile | 4.86 | | | | | | |

Where the worst points are: the orders 5 and 6 of log f at
(lambda, alpha) = (1.7, 0.4), in the pure alpha-derivatives, which is the
accuracy of `log_mills_d5` and `_d6` (96 and 536 ulp) amplified by the
cancellation between the derivatives in a and in c; the orders 3 to 6 of the
two cumulative functions for y within about 0.01 scale units of mu at x
between 0.7 and 2, where the derivatives in (lambda, alpha) are differences of
two tables of log Z with an absolute error of about 1e-12; their orders 0 and 1
in the far tail, which is the conditioning of exp(T) at |T| = 170, about 2 ulp
per unit of the logarithm; the order-4 derivative of I_mumu in alpha at
(lambda, alpha) = (1e6, 0.5), where the component is 1.2e4 against
I_mumu = 1e12, so 6e-11 of the component and 1e-14 of the matrix entry; and
I_la itself at alpha = 1 - 1e-12, which is 3.7e-13 with an absolute error of
7e-17, where an entry-wise relative error means nothing.

The two forms of the (a, c) table agree to 9e-14, 3e-14, 5e-14, 2e-13 and
6e-12 at x = 1, 1.5, 2, 3 and 4.

The two limits hold. At alpha = 1 - 1e-10 the (mu, lambda) block of every order
of log f and of the two cumulative functions agrees with `laplace2_distrib` to
1e-8 of the order's scale, with an absolute floor of 1e-9 because the Laplace's
far-tail derivatives are exactly zero while the enet's are about 1e-15; the
information and the quantile agree to 1e-8. At alpha = 1e-10, log f and the
information agree with `gaussian1_distrib` and `gaussian3_distrib` to 1e-8, and
the cumulative derivatives to 1e-6, since the orders 5 and 6 amplify the O(a)
difference between the two families to about 1e-7.

**What this says.** The kink is handled exactly, and the expected information
agrees both with the variance of the score and with
`expected_info_by_integral<K, score>` for K = 0, 1 and 2, which is the check
that the point mass at the kink is where it belongs. What limits the high
orders is the accuracy of the Mills ratio's own fifth and sixth derivatives and
genuine cancellation between the derivatives in a and in c, not the closed
forms: the two independent forms of the table agree to 1e-13.

**Chosen.** As above, with the bounds in `tests/distrib/test_enet.cpp` (17
cases, 44647 assertions) at 1.5 times the worst of the two builds, per kind and
order: for log f 240, 5, 9, 40, 175, 2500 and 3100; for log F 720, 740, 2700,
7400, 4200, 14400 and 24800; for log(1 - F) 950, 1450 and then the same; for
the information 30, 48, 175, 620 and 1.18e6; for the quantile 7.3. The test
also checks the 2a jump of dl/dy at the kink, the 4 a f(mu) jump of the second
derivative of log F, that the density integrates to one, the quantile's round
trips, and the family inside a Kokkos kernel (host and device within 1.9e-12).

**Open.** The orders 4 to 6 of log F and log(1 - F) within about 0.01 scale
units of mu, where a Taylor or integral form in D for the components with no
derivative in mu would replace the difference of two tables. The order-4
derivative of I_mumu in alpha at lambda >= 1e5, which an accurate series for
exp(-F(t)) - 1 - t would fix. I_la near alpha = 1 is accurate absolutely and
not relatively. Better fifth and sixth derivatives of the Mills ratio would
lower the two worst rows of log f.

## skewnormal, the skew normal in its centred parametrization

**What.** `skewnormal_distrib{mu, sigma, gamma}`
(`include/laplace/distrib/skewnormal.hpp`): the mean, the standard deviation
and the skewness, all three moments of the law, which is what the centred
parametrization means. gamma lies in (-g, g) with
g = ((4 - pi)/2) (b / sqrt(1 - b^2))^3 = 0.9952717..., b = sqrt(2/pi), the
supremum of the skewness over the family, attained only in the half-normal
limit. The density is the classic skew normal at the triple the centred
parameters imply,

    cc = sign(gamma) (2 |gamma| / (4 - pi))^(1/3),
    xi = mu - sigma cc,   omega = sigma sqrt(1 + cc^2),
    alpha = cc / (b sqrt(1 - q)),   1 - q = -expm1((2/3) log(|gamma| / g)),

so that, with t = (y - xi)/omega and u = alpha t,
l = log 2 - log omega - log(2 pi)/2 - t^2/2 + log Phi(u). The log-density and
its six orders, the derivatives in y and the mixed ones, the cdf family with
six orders, the descriptors and the starting value are written; the expected
information, the quantile and the draw are the generic routes.

**How.** The composition is carried on cc and only at the last step replaced by
gamma. The table of mixed derivatives of W = -log s - t^2/2 + log Phi(u) in
(z, cc) comes first: t and u are linear in z with coefficients that depend on
cc alone, so the z-derivatives are Leibniz plus log_Phi_dj(u) C^j, and the
cc-derivative of that is Leibniz over the one-variable Faa di Bruno sum
(`bell_partial`); the cc-derivatives of the coefficients are exact recurrences
from (1 + x^2) A' = -x A and (1 - kappa x^2) E' = kappa x E, written in 1 - q so
that they stay accurate near the boundary. Only then is cc replaced by gamma,
by Faa di Bruno with cc^(n) = cc (1/3)(1/3 - 1)...(1/3 - n + 1) / gamma^n, so
that the gamma^(1/3 - n) blow-up enters once instead of in every term; the
table then goes to `detail_location_scale::mixed_shape_dk<K>`. log F is the
(t, alpha) table of `log_sn_cdf_d1 .. _d6` composed with t(z, cc) and
alpha(cc) and then carried to gamma the same way, and log(1 - F) is the same
code at (-t, -alpha).

Two things the family needed that were not foreseen. The supremum g as a
`constexpr` double is 2 ulp from its true value, which costs 1e-6 of relative
error in 1 - q at |gamma| = g - 1e-10, so the map carries g as a pair of
doubles (the low part, -1.9209167999475743e-16, from mpmath at 300 bits). And
at gamma = 0 exactly the components with one index in gamma are written in
closed form (l_gamma = He_3(z)/6 and d log F / dgamma =
-(z^2 - 1) phi / (6 Phi)), while those with two or more are NaN, for the reason
under "What this says".

**Measured.** `tools/reference/distrib_skewnormal.py` evaluates the definition
in truncated three-variable Taylor arithmetic of total degree 6 at 300 and 400
bits, keeping a row when the two agree to 2^-80, and writes 54539 rows
(`tests/data/reference/distrib_skewnormal.csv`): gamma = +-1e-12, 1e-8, 1e-4,
1e-2, 0.5, 0.9, g - 1e-6 and g - 1e-10, z from -12 to 12 (so u reaches 2e6),
(mu, sigma) = (0.4, 1.3) and (-2, 0.25), with (1e5, 1e-6) and (3, 1e6) on a
subset. Units: ulp of the largest reference component of the same order and
with the same number of indices in gamma at that point, the maximum over the
log-density, log F and log(1 - F), and over both builds, which agree within a
factor of three everywhere.

| gamma | order 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1e-12 | 1.9e9 | 1.0e14 | 5.9e17 | 2.3e18 | 5.7e18 | 9.4e18 |
| 1e-8 | 1.4e6 | 6.8e9 | 1.7e12 | 8.4e12 | 1.7e13 | 2.3e13 |
| 1e-4 | 1.6e5 | 9.1e5 | 3.3e6 | 5.3e6 | 6.0e6 | 8.5e6 |
| 1e-2 | 130 | 8.3e3 | 2.3e4 | 4.9e4 | 5.5e4 | 1.4e5 |
| 0.5 | 272 | 678 | 513 | 475 | 2.8e3 | 1.7e4 |
| 0.9 | 1.2e3 | 1.6e3 | 1.4e3 | 1.7e3 | 1.8e3 | 6.4e3 |
| g - 1e-6 | 54 | 331 | 330 | 5.6e4 | 873 | 1.4e3 |
| g - 1e-10 | 100 | 68 | 86 | 1.7e4 | 683 | 208 |

The log-density alone is far better than the worst of each row: from
|gamma| = 0.5 to g - 1e-10 it stays between 8 and 520 ulp at every order, with
u reaching 2e6. The expected information by the generic route agrees in its two
forms to 1e-9 relative and with the variance of the score, and at gamma = 0
gives diag(1/sigma^2, 2/sigma^2, 1/6), whose (mu, sigma) block is the
Gaussian's closed form; its first two derivatives agree with differences to
1e-7 and 1e-6.

**What this says.** Away from zero the family is accurate, and the pair of
doubles for g is what makes the two rows nearest the boundary work.

Near gamma = 0 the derivatives in gamma of order two and above lose most of
their digits, and the reason belongs to the law and not to the route: the
expansion in cc gives gamma He_3(z)/6 and then a term of order
cc^4 ~ gamma^(4/3), so the log-density is only C^1 in gamma at zero and every
derivative of order two or more diverges like gamma^(-2/3). The expected
information stays finite there (E[l_gamma gamma] = 1/6 at zero), which is why
the generic route still answers. Composing in cc does not remove the
cancellation inside the (z, cc) table, whose entries with at least one
cc-index are O(cc^2) formed from O(1) terms, so they carry an absolute error of
about 1e-16 and are then multiplied by cc^(n): at |gamma| = 1e-2 the loss is
about five digits at order six, at 1e-4 about ten, and at 1e-12 the first order
keeps seven digits, the second two and the rest none. The value and the orders
in (mu, sigma) alone are untouched.

**Chosen.** As above, with the bounds in
`tests/distrib/test_skewnormal.cpp` (12 cases, 53195 assertions) at 1.5 times
the worst of the two builds, per class of gamma and order; the orders 3 to 6
at |gamma| = 1e-12 are not compared, which the file states. The test also
checks the Gaussian limit at gamma = 0 against `gaussian1_distrib`, that the
three parameters are the three moments by quadrature, the reflection in the
sign of gamma, the derivatives of log F against differences of its own log F,
and the family inside a Kokkos kernel (host and device within 6.4e-13).

**Open.** A series in cc of W near zero skewness was examined and not
written, and the reason is measured rather than guessed. The expansion of W in
powers of cc, taken with SymPy and with log Phi left as an unknown function L,
has no cc^1 and no cc^2 term at all: the first vanishes because
L'(0) = b = sqrt(2/pi) is exactly the factor the quadratic part contributes,
the second because L''(0) = -2/pi, and the cc^3 coefficient is
(1 - kappa) He_3(z)/6 = gamma He_3(z)/6, which is the score the family already
writes in closed form at gamma = 0. A series would therefore have to remove two
orders of cancellation and not one, and it cannot be short: the natural
variable is u = alpha t, about cc z / b, and the Taylor series of log Phi at
zero has radius about 1.7, the nearest complex zero of Phi, so at |cc| = 0.05
and the |z| = 12 of the reference grid about forty terms are needed for the last
bit. What it would buy is small beside that: the score in gamma keeps seven
digits at |gamma| = 1e-12 and ten at 1e-8, which a fit can use, and the orders
two and above diverge like gamma^(-2/3) because the law is only C^1 in gamma at
zero, so there is nothing there to be accurate about. The non-smooth point at
gamma = 0 is now declared, in the field the descriptor gained for it: the law is
only C^1 in gamma there, and `params[2].axis_kinks` says so with one point at
zero and `continuous_order` 1, while `regularity` stays `infinitely_smooth`
because that field holds at every value of the parameter and a finite order in
it would make the generic routes drop the derivatives in gamma everywhere, not
only at the one point where they do not exist. The two kinds of non-smooth
point are different things and are declared apart: `param_kinks(j, y)` gives a
point of theta_j placed by the observation, where l has a kink in y as well
(the Laplace and the enet at mu = y), and `axis_kinks` gives the points fixed
on theta_j's own axis. Nothing reads the new field yet, the engine not being
there, so the declaration changes no result. The
starting value, now that `distrib_sample_summary` carries the sample skewness
(added for this family, the standardized third central moment with divisor n),
is the three sample moments; `start_inside` can still land it at
g (1 - 16 eps), where alpha is about 1e7, so a wider margin there is worth
considering.

## student_t1 and student_t2, the Student t in its two parametrizations

**What.** `student_t1_distrib(mu, sigma, nu)` with sigma the scale and nu > 0,
and `student_t2_distrib(mu, sigma, nu)` with sigma the standard deviation and
nu > 2, both with

    l = c(nu) - log sigma - ((nu + 1)/2) log(1 + z^2/nu),   z = (y - mu)/sigma,
    c(nu) = log Gamma((nu + 1)/2) - log Gamma(nu/2) - (1/2) log(nu pi).

Each supplies the log-density and its orders one to six in (mu, sigma, nu),
the derivatives in y and the mixed ones, the expected information in closed
form with its orders one to four, and the cdf family with the orders one to six
of log F and of log(1 - F). The quantile and the draw are the generic routes.
`student_t2_distrib` is `student_t1_distrib` at the scale
s = sigma sqrt((nu - 2)/nu), composed with `chain_rule_pair_dk`, whose inner
map needs the derivatives of r(nu) = sqrt((nu - 2)/nu) from the one-signed
recurrence

    r^(q) = [-2 (-1)^q q! nu^-(q+1) - sum_i C(q, i) r^(i) r^(q-i)] / (2 r).

**How.** The density's table D[c][j] of the mixed derivatives of
Lambda = -((nu + 1)/2) log(1 + z^2/nu) in the shape nu and in z goes to
`detail_location_scale::mixed_shape_dk`, and c(nu) and -log sigma are added to
the all-nu and the all-sigma components. The closed forms, checked against
SymPy for every c + j <= 6, are:

- c = 0: -(nu + 1) (z g(1, j - 1) + (j - 1) g(1, j - 2)) with
  g(m, n) = d^n/dz^n A^-m and A = nu + z^2;
- c >= 1 and j >= 1: -(-1)^c c! d^(j-1)/dz^(j-1) [P A^-(c+1)] with
  P = z(1 - z^2);
- j = 0: with r = z^2/A, the value (1/2) log1pmx(-r) + r/(2 nu) at c = 1, and
  for c >= 2 with m = c - 1,
  (1/2)(-1)^m [(m - 1)! nu^-m B_m(r) + m! nu^-(m+1) r (1 + q + ... + q^m)],
  B_m(r) = (1 + m r)(1 - r)^m - 1, a polynomial in r below 1/4.

Subtracting the Gaussian part from the c = 0 row as well, which the other rows
need, loses everything for z^2 much larger than nu; the c = 0 row is the direct
product.

The information is

    I_mumu = (nu + 1)/((nu + 3) sigma^2),
    I_sigmasigma = 2 nu/((nu + 3) sigma^2),
    I_sigmanu = -2/(sigma (nu + 1)(nu + 3)),
    I_nunu = (psi'(nu/2) - psi'((nu + 1)/2))/4 - (nu + 5)/(2 nu (nu + 1)(nu + 3)),

with I_munu = I_musigma = 0 by the symmetry of the density, each verified
against E[l_a l_b] by mpmath quadrature at 30 digits. I_nunu is
7/(2 nu^4) as nu grows, not O(1/nu^3): the coefficients of nu^-2 and nu^-3 both
vanish. Below nu = 18 it is formed from `dpsi_all` minus a sum of positive
products; from 18 on from its series in nu^-4 to nu^-38, whose relative
truncation error is 1e-18 at nu = 18. The derivatives in nu of the other
entries are sums of positive products of powers of 1/(nu + a). For
`student_t2_distrib` the transform J' I J simplifies, r cancelling, to
A2 = nu(nu + 1)/((nu + 3)(nu - 2)), B2 = 2 nu/(nu + 3),
E2 = 6/((nu + 3)(nu - 2)(nu + 1)) and
F2 = I_nunu + 2/((nu + 3) nu (nu - 2)^2) - 4/(nu (nu - 2)(nu + 1)(nu + 3)),
checked with SymPy.

The (mu, sigma) components of log f that carry no nu-derivative are not formed
from the (z, nu) table where |y - mu| >= sigma sqrt(nu). The carry of
`mixed_shape_dk` is a sum of monomials c z^m L_j, and in the far tail its terms
grow like z^(K-1) while the component they build is O(1) or smaller: at
|z| = 1e8 the order six used to lose three digits of the tensor's largest
component. With u = y - mu and w = sigma sqrt(nu),

    log f = c(nu) + ((nu + 1)/2) log nu + nu log sigma - (nu + 1) Re log(u + i w),

because log(1 + z^2/nu) = log(u^2 + w^2) - log(nu sigma^2) and
log(u^2 + w^2) = 2 Re log(u + i w). Since d/dmu = -d/du and
d/dsigma = sqrt(nu) d/dw, and log(u + i w) has
d^a/du^a d^b/dw^b log(u + i w) = i^b (-1)^(K-1) (K-1)! (u + i w)^-K for
K = a + b >= 1,

    d^a/dmu^a d^b/dsigma^b l = -(nu + 1) (-1)^a nu^(b/2) (-1)^(K-1) (K-1)!
                               Re[i^b (u + i w)^-K]
                               + [a = 0] nu (-1)^(b-1) (b - 1)! sigma^-b,

the last term being all that nu log sigma contributes. One complex power per
order, taken from the reciprocal of u + i w by repeated multiplication and
scaled by max(|u|, w) so that neither u^2 + w^2 nor the power overflows, and
no quantity on the right is a difference of larger ones. The split is on
|u| >= w rather than on |z| or on nu: below it the closed form is the bad one,
because Re[i^b (u + i w)^-K] is then a small difference and nu log sigma a
large term against a small result, and it reaches 1e12 ulp at nu = 1e12. The
two forms were measured against each other on the whole reference grid, by
region of r = |z|/sqrt(nu): at r < 1 the table route is at worst 33 ulp at
order six and the closed form 1.5e12, at r >= 1 the table route reaches 2.4e3
and the closed form stays at or below 4.6. `tools/reference/distrib_student_t.py`
checks the rewrite and the closed component against SymPy at two points.

The skew t carries the same block in its own far tail, on the same split. Its
table is the Student one plus the skew factor's, so the two are kept apart
rather than subtracted: `detail_skewt::lambda_table` hands back the skew
factor's own z-derivatives before the Student part is added, those are carried
by `mixed_dk` as usual, and the Student half comes from the closed form. Formed
by subtraction instead, the Student half would carry back the cancellation the
closed form removes. At alpha = 0 the skew factor is constant and the block is
student_t1's to the bit, which is what the twin test asks and what it now
measures: zero ulp at every order, where the closed form in student_t1 alone
left the two families 2, 3, 5, 15, 103 and 502 ulp apart at the orders one to
six.

For log F one call of `detail_st::derivs<K>` gives the whole (z, nu) table,
which feeds `mixed_shape_dk`. For |z| >= 1 on the side where the function tends
to zero the polynomials of `mixed_dk` cancel, as they do for the Cauchy, so
there the tensors are formed from those of the other tail by the
complete-Bell recursion for log(1 - e^l) (`bell_log_one_minus_exp`): measured
against the direct route with the Cauchy twin, 5e-6 before and 3e-12 for
|z| < 20 after.

**Measured.** Worst error in ulp of the largest component of that order at that
point, worst of the gcc and the CUDA build; references by mpmath in
`tools/reference/distrib_student_t.py`.

| quantity | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| t1 log f | 3.68 | 2.35 | 3.66 | 5.1 | 20.1 | 107 |
| t2 log f | 5.81 | 6.87 | 9.65 | 103 | 9.1 | 31.2 |
| t1 log F | 1.2e3 | 962 | 786 | 1.23e3 | 977 | 6450 |
| t1 log(1 - F) | 1.2e3 | 962 | 786 | 1.23e3 | 977 | 805 |
| t2 log F, nu >= 2.5 | 903 | 725 | 580 | 925 | 733 | 582 |
| t2 log(1 - F), nu >= 2.5 | 903 | 725 | 580 | 925 | 733 | 582 |
| t2 log F and log(1 - F), nu = 2 + 1e-6 | 17.9 | 1.5e5 | 5.9e6 | 1.6e7 | 4.6e7 | 1.3e8 |

The two rows of the density are what the closed form of the far tail bought.
Before it they read 13.5, 31.7, 100, 260, 940 and 2400 for t1 and 5.8, 9.3,
12.8, 55, 178 and 632 for t2, so t1 improves at every order, by 22 times at
order six, and t2 at every order but the fourth, which grows from 55 to 103 at
a pure nu component at z = 4, nu = 10: t2 is t1 at the scale
s = sigma sqrt((nu - 2)/nu) composed through `chain_rule_pair_dk`, and the
chain mixes components whose last bits the closed form has moved. One component
1.9 times worse against orders five and six 20 times better is the trade that
was taken. The worst points of the density are now the components with at least
one nu-derivative, which the closed form does not touch: 107 ulp at order six
at z = 1e8, nu = 2. The worst arguments of the cumulative
functions are |z| = 30 with nu = 1e4 at the orders one, two, four and five for
both families, and nu = 1e8 at the order three of t1; the order six of t1's
log F at z = 31.6 with nu = 3.5, and of its log(1 - F) at |z| = 4 with nu = 2,
while t2 stays at |z| = 30, nu = 1e4 at every order. That point, which carries
five of the six orders of both families, is the one the conditioning
measurement below is made at. The density of t2 at nu = 2 + 1e-6
stays within 39 ulp, so the chain through r(nu) holds there; its log F keeps
about eight digits at order six. The information and its orders one to four
measure 161, 56, 51, 35 and 17 ulp for t1 and 321, 115, 100, 54 and 35 for t2,
the worst point being I_nunu at nu = 17.9, just below the series cut, where
the polygamma difference cancels; from nu = 19.9 on the error is a few ulp.

Twins and limits: t1 at nu = 1 against `cauchy_distrib`, 9e-14 on the density
and 3.2e-12 on log F and log(1 - F) for |z| < 20; t1 at nu = 1e8 and 1e12
against `gaussian1_distrib`, where the differences are 0.5, 0.25 and 0.25 of
the O(1/nu) scale and identical at both values of nu, so no digit is lost as nu
grows and the nu-derivatives of order six down to 1e-82 are right; the generic
quadrature route for the information in its score form, 1.1e-15 of the largest
component at nu = 3.5, 9 and 30 for both families, and the variance of the
score 7e-15; the host and the device within 8.7e-15 inside a Kokkos kernel.

The references of log F gained 20 (point, side) pairs that had none, and it
took one line of `tools/reference/distrib_student_t.py`. Its definition took
the tail as (1 - I_y(1/2, nu/2))/2 with y = z^2/(nu + z^2) wherever
x = nu/(nu + z^2) exceeded 1/2, which is the usual choice for the incomplete
beta but is a difference against one: its nu-derivatives cancel and mpmath.diff
never reaches the 2^-100 the script demands of two precisions. Holding each
form fixed and comparing them at the same point says it plainly: at z = 30,
nu = 1e4 the table of the six orders disagrees by 3e197 between 280 and 360
bits in that form and by 3e213 between 440 and 560, while I_x(nu/2, 1/2) taken
directly agrees to 1.3e-81 at the lowest pair; at z = 100, nu = 1e4, where x is
exactly 1/2, the same. Raising the ladder would also have worked and is not
needed: 900 against 1200 bits agrees at 3.8 minutes a point and 1400 against
1800 at 12, where the direct form costs 20 seconds. With the one form the file
keeps every row it had, to the bit, and adds 1660.

What that form cannot do is z = 0, and the reason is worth writing down because
no agreement test can see it. I_x is even in z, so log F is log(t) on one side
of zero and log1p(-t) on the other: two functions meeting there, and the
stencil differentiates across the join. Against the exact f(0)/F(0) = 2 f(0),
the first z-derivative at zero comes out wrong by 2.4e-7 at nu = 3.5, 1.9e-6 at
nu = 10 and 6.8e-4 at nu = 1e4, and both precisions agree on the wrong value.
The form that is analytic across zero is F(z) = 1/2 + sign(z) I_y(1/2, nu/2)/2,
where I_y ~ c|z| makes sign(z) I_y the analytic odd part: it gives 2 f(0)
exactly and agrees to every digit with a quadrature of the density, which uses
no incomplete beta at all. It is useless in the tail, where I_y is one to the
last bit and the sum cancels (at z = 30, nu = 1e4 it returns a flat zero), so
the two forms would have to be chosen per point. That was tried and is not in:
with the central form in place the library disagrees with the new central
references by far more than their own accuracy, and which of the two is right
there has to be settled before either is written to the file. The z = 0 points
therefore still have no reference, 48 (point, side) pairs in all, of which the
far ones (|z| of 1e8 with nu = 100, and the like) can never have one, their
log(1 - F) being of the order of 1e-2000 and below the smallest normal double.

Those new pairs are the first measurement of log F at nu >= 1e4 beyond
|z| <= 4, and what they show is the conditioning of the point, not a loss of
the route. At z = 30, nu = 1e4, where the library reads 580 to 1230 ulp of the
order's scale, the components themselves move by that much when the inputs
move by one ulp of a double: recomputed in mpmath at 280 bits with one ulp
added to sigma, the orders one to six shift by 1.51e3, 1.21e3, 970, 1.54e3,
1.23e3 and 968 ulp of their own order's scale, and with one ulp added to y by
809, 649, 519, 828, 657 and 520; one ulp of mu or of nu moves them by 16 to 28.
The library's error is therefore of the size of the uncertainty the arguments
already carry, so no route in double can do better there and the bound is the
problem's. The same measurement is the reason the bound was raised rather than
recorded as an open choice.

**Chosen.** As above, with the bounds of `tests/distrib/test_student_t.cpp`
(18 cases, 78391 assertions) at 1.5 times the worst of the two builds. The
region nu < 2.001 of t2 and the block of t1 at integer nu have their own
treatment, which the file states: the first its own bound, the second none on
the block alone, only on the whole tensor. The test also checks the two twins
above, the derivatives against central differences in the centre and in the
tails, the generic quantile against its own cdf, and the family inside a
Kokkos kernel in a named function with external linkage.

**Open.** Four corners. The (mu, sigma) block of t1 at integer nu up to 6 in
the far tail: the tail is a power of sigma/(y - mu) with an integer exponent,
so the components of order above nu are differences of terms of the size of the
tail probability and lose their accuracy within the block (2e8 ulp of the
block's own largest component at |z| <= 31, that is 2.7e-8 relative at order
six and z = 30, and meaningless at z = 1e8), although they stay below 805 ulp
of the whole tensor; the cure is a series in 1/z^2, which for the Student needs
the nu-derivatives of the hypergeometric expansion of the incomplete beta and
is not written. The log F of t2 as nu approaches 2. The reference itself at
nu >= 1e4, where 78 (point, side) pairs are absent because `mpmath.diff` did
not reach agreement to 2^-100 even at 560 bits, so log F there is checked only
for |z| <= 4 and at |z| = 1e8 with nu = 1e8. And the 2400 ulp at order six of
the log-density at |z| = 1e8, which comes from the z-polynomials of `mixed_dk`
and which an assembly through Re(u + i sqrt(nu) sigma)^-K, as the Cauchy uses,
would remove.

## pseudohuber, the pseudo-Huber distribution in the standard deviation

**What.** `pseudohuber_distrib(mu, sigma, nu)` with sigma the standard deviation and nu > 0 the
shape, the one parametrization of the family Laplace carries, with density

    f(y) = sqrt(R) / (2 sigma sqrt(nu) K_1(sqrt nu)) exp(-sqrt(nu + R z^2)),
    z = (y - mu)/sigma,   R = R(nu) = sqrt(nu) K_2(sqrt nu)/K_1(sqrt nu),

so sigma carries the whole variance and nu only the tails: the Laplace as nu -> 0 and the
Gaussian as nu -> infinity, both with variance sigma^2. On the log scale

    log f = -sqrt(nu + R z^2) + h(nu) - log 2 - log sigma,
    h(nu) = (1/2) log R - (1/2) log nu - log K_1(sqrt nu).

The header writes the log-density and its orders one to six in (mu, sigma, nu), the derivatives
in y and the mixed ones, and the starting value; the expected information, the cdf family, the
quantile and the draw are the generic routes.

**How.** The Bessel ratio comes from the identity that removes the cancellation of the quotient:
from K_nu'(x) = -K_(nu+1)(x) + (nu/x) K_nu(x) at nu = 1,

    R = 1 - x (log K_1)'(x),   x = sqrt(nu),   d^k R/dx^k = -(x g^(k+1)(x) + k g^(k)(x)),

with g^(j) = log_bessel_k_dx<j>(1, x), and the derivatives in nu by one Faa di Bruno composition
with x = nu^(1/2). Both terms of the sum are of the same sign, so the quotient of two functions
each of size exp(-x) is never formed. The normalizer h(nu) takes its nu-derivatives from the same
set of Bessel derivatives, log R - log x being the same function as log K_2 - log K_1, so the
whole family needs one evaluation of log_bessel_k and of its orders one to seven a tensor.

Above the split Lambda and h are carried shifted by x = sqrt(nu), because Lambda tends to -x and h
to +x while their sum is O(1): their pure-nu derivatives are O(nu^(1/2 - c)) and the sum is
O(nu^(-1/2 - c)), so the one component that holds them both used to lose a whole factor of nu.
With Rm = R - x,

    Lambda + x = -rho z^2 / (G + sqrt q) = -(z^2/2) (1 + Theta),
    Theta = (4 Rm - z^2) / ( [(R + Rm)/rho + G] (G + sqrt q) ),
    h - x = (1/2) log R - log x - gs(x),   gs(x) = log K_1(x) + x,
    Rm = 1 - x gs'(x),   gs'(x) = (log K_1)'(x) + 1,

the second form of Lambda + x from 2 rho - sqrt q - G = (4 Rm - z^2)/[(R + Rm)/rho + G], which
multiplies the difference by its conjugate. Theta is O(nu^(-1/2)) and Rm tends to 3/2, and no
quantity on the right is a difference of larger ones. The shift is taken where R z^2 <= nu, which
keeps 1 + Theta = 2 R/(sqrt(nu + R z^2) + x) away from zero; above that Lambda is nowhere near -x
and the two pair as they stand, h being restored by adding the derivatives of x back. The two
scaled Bessel pieces are `math::log_bessel_k_scaled` and `log_bessel_k_scaled_dx_upto`, which cost
nothing beyond the plain ones: Steed's branch omits the -x it would subtract, the asymptotic
expansion omits the -1, the Riccati regime adds one to g in double-double, and the Debye form uses
-nu eta + x = -nu^2/(nu s + x) - nu log(z/(1 + s)) and 1 - s/z = -1/(z(s + z)). h - x is the same
h_derivs fed with the scaled set, and the only entry of the (nu, z) table that changes is the one
at c = K, j = 0, which is where mixed_shape_dk reads the pure-nu derivative.

Lambda = -sqrt(nu + R z^2) is written as -rho(nu) G(q(nu), z) with rho = sqrt(R), q = nu/R and
G = sqrt(q + z^2); the table D[c][j] of its mixed (nu, z) derivatives feeds
`detail_location_scale::mixed_shape_dk`. The z-derivatives of (q + z^2)^p come from a recurrence
in p, except the pure z-derivatives at p = 1/2, which use the exact identity
H_(1/2)^(m) = q H_(-3/2)^(m-2) because the plain recurrence cancels completely at large |z|.

**Measured.** Log-density against mpmath (SymPy's derivative of the definition with R and h from
mpmath Bessel values at 400 and 600 bits; the nu-derivatives of the Bessel pieces from truncated
power series of the Bessel ODE system, `mpmath.diff` needing 90 s a point for R alone), in ulp of
the largest component of the order, over 209 points with nu from 1e-6 to 1e8, |z| to 1e8 and
sigma from 2^-20 to 2^20. The three regions have different causes, so they have their own bounds.

| order | nu < 1 | 1 <= nu < 1e4 | nu >= 1e4 |
|---|---|---|---|
| value | 1.61 | 4.51 | 7.02 |
| 1 | 29.2 | 33.2 | 6.55 |
| 2 | 4.67 | 19.9 | 84.8 |
| 3 | 5.49 | 53.7 | 71.5 |
| 4 | 8.19 | 86.6 | 129 |
| 5 | 13.5 | 79.7 | 37.1 |
| 6 | 9.08 | 278 | 24.8 |

The two columns above one are what the shift of Lambda and h by x bought: before it they read 41.5,
1.5e3, 747, 1.0e3, 339, 869 and 1.4e3, and 1.6e4, 1.2e4, 1.0e4, 5.3e3, 5.4e3, 8.6e3 and 9.1e3, so
the value and the first order at nu >= 1e4 improve by 2300 and 1800 times and no order anywhere is
worse. At nu = 1e8 every order now sits between 0.61 and 3.7 ulp, and the worst point of the whole
grid is the order six at nu = 1 (278 ulp), where the shift also took away most of what used to be
1.4e3. The two builds agree to the digit on every entry of the table. The split at nu = 1 and the
series below it are unchanged, and the first column is bit for bit what it was.

The first column is the small-shape region after the series below, which replaced the Bessel
route there: before it the same column read 9.4 on the value and 1.8e4, 1.8e6, 6.8e6, 1.4e7,
1.5e7 and 1.7e7 at the orders one to six, and at nu = 1e-6 the worst was 1.7e7 ulp against the
12.7 it is now.

The limits: against `laplace1_distrib(mu, sigma/sqrt 2)`, every order of the (mu, sigma) block
agrees to 3.3e-8 at nu = 1e-8 and 2.5e-6 at nu = 1e-6, that is about 3 nu, the rate of the limit
itself; against `gaussian1_distrib` at nu = 1e8 and 1e10 the ratios to the stated scales are
0.75 for the (mu, sigma) components, 0.19 for the nu components and 0.375 for the value. The
density integrates to one and has variance sigma^2 to 4.5e-14 by quadrature. The expected
information from the generic route: the score form against the variance of the score 3.5e-16,
the hessian form against the score form 7.0e-13, the two entries that vanish by symmetry
(I_mu,sigma and I_mu,nu) to 6.9e-16 of the largest entry, and against the two limits 1.57e-4 and
1.5e-4, the corrections being O(sqrt nu) and O(1/sqrt nu); the orders one to four against
differences 1.2e-11 to 1.0e-10. The generic cdf against differences of its own log F: 4.2e-12 at
order one to 9.9e-11 at order six. The generic quantile needed no special treatment:
cdf(quantile(p)) against p is 2.3e-15 relative and q(p) + q(1 - p) - 2 mu is 7e-16 at dyadic p.
Device against host, orders one to six: at most 1.3e-15 at nu = 3.5 and from 6.1e-16 to
1.6e-14 at nu = 0.05.

Cost on one gcc thread, the whole tensor a call:

| nu | order-6 tensor | before the two changes below |
|---|---|---|
| 0.1 | 5.6 us | 8.0 us |
| 2 | 7.2 us | 19.5 us |
| 100 | 6.3 us | 12.0 us |
| 1e4 | 6.3 us | 11.0 us |

and 5.7 us at nu = 1e-4, where log f alone costs 79 ns against the 282 ns it cost at nu = 0.1
through the Bessel route. What is left is the assembly of Lambda and the packing of the
symmetric tensor, not the Bessel functions.

The generic information costs 3 ms (nu = 0.05) to 16 ms (nu = 2.5) at order zero and 60 to
160 ms at order four, the score form being 1.5 to 2 times faster than the hessian form.

**Chosen.** As above, with the bounds of `tests/distrib/test_pseudohuber.cpp` (15 cases, 21033
assertions) at about 1.5 times the worst of the two builds, per region of nu and per order; the
symmetry of the generic quantile, q(p) + q(1 - p) - 2 mu, moved from 1.1e-15 to 1.9e-15 when the
density changed in its last bits and the root stopped elsewhere (1.27e-15 on both builds). The
closed forms are checked against SymPy by
`python -I tools/reference/distrib_pseudohuber.py forms` (the R identity, the recurrence for
(q + z^2)^p, the identity at p = 1/2 and the whole table D, worst 1.4e-51 relative at five
points), which also checks that the density integrates to one with variance one from nu = 1e-8
to 1e8.

**The two changes that followed the first measurement** (both decided with Giovanni, both in).

*A series for R and h below nu = 1.* The components with every index in nu were losing about
eps/nu, because d^k R/dx^k = -(x g^(k+1)(x) + k g^(k)(x)) is a difference of terms of size x^-k
that cancels to size x^(2-k). Below the split the family does not call the Bessel functions at
all: from the small-argument series of K_0 and K_1 (DLMF 10.31.1),

    R = 2 + nu K_0/B,   B = x K_1 = 1 + (1/2) L J_1 - W_1,   L = log(x/2),
    h = (1/2) log R - log B,

where K_0 = -(L + gamma) P_0 + H_s with the series P_0, H_s, J_1 and W_1 in nu built by
recurrences in a loop, fourteen terms, no table of constants. Writing h with B = x K_1 rather
than with K_1 is what makes the pole and the log nu term cancel exactly. The derivatives follow
by Leibniz's rule, Faa di Bruno and `chain_rule_d1 .. _d6`, one function per order as everywhere
else. The split was chosen by measuring both routes against the definition at 32 values of nu
from 1e-6 to 56: the series holds to 2e-15 from 1e-6 to 1 and then loses digits (4e-15 at 1.78,
3e-14 at 3.2, 1e-8 at 10), the Bessel route holds to about 1e-15 from nu = 0.3 and loses them
downwards (3e-15 at 0.1, 1e-14 at 1e-2, 2e-12 at 1e-4, 4e-10 at 1e-6), and between 0.3 and 1 the
two are equal at 1e-15 to 2e-15, so the split is nu = 1, where the series is also the cheaper of
the two. `python -I tools/reference/distrib_pseudohuber.py small` checks the tables against
SymPy's series and the whole derivation against the definition, to 1e-20 at every order up to
six.

*Every order of log_bessel_k from one evaluation.* Each `log_bessel_k_dx<n>` used to recompute
the shared work of its regime, so a tensor paid for it eight times. `log_bessel_k_dx_upto<N>`
fills the value and the orders one to N from one evaluation in each of the three regimes (one
`g_dd` and one pass for Riccati, one series sum for Hankel, one set of Taylor coefficients for
Debye), and it is 2.9 to 5.7 times faster than the eight calls, the most at x = sqrt(2) where
this family sits. It agrees with the single-order functions bit for bit on gcc (0 of 840 cases
differ, every regime and both sides of nu = 20, x = 200 and x = 1) and to 1.2e-15 relative on
CUDA, where nvcc contracts and schedules the two differently.

**Open.** One corner, the other having been repaired by the shift of Lambda and h by x: nu = 1 at
order six, 278 ulp and the worst point of the grid, down from 1.4e3 before the shift and still not
chased. It is where the series and the Bessel route meet, and before the shift the series was not
the explanation either (using it at nu = 1 left 746 ulp of the 1.4e3). For the record, what the
shift replaced: the value and the first order at nu >= 1e4 used to reach 1.6e4 and 1.2e4 ulp, an
absolute error of about 1e-12 on the value at nu = 1e8.

## skewt, Azzalini's skew t

**What.** `skewt_distrib(mu, sigma, alpha, nu)` with sigma the scale, alpha the shape and nu > 0
the degrees of freedom, the four-parameter family

    f(y) = (2/sigma) t_nu(z) T_(nu+1)(w),   z = (y - mu)/sigma,
    w = alpha z sqrt((nu + 1)/(nu + z^2)),

with t_nu the Student density and T_m the Student cdf at m degrees of freedom. On the log scale

    log f = log 2 + c(nu) - log sigma + Lambda(z, nu) + log T_(nu+1)(w),

the Student log-density of `student_t1_distrib` plus one term. alpha = 0 is the Student,
nu -> infinity the direct skew normal and |alpha| -> infinity the half t. The header writes the
log-density and its orders one to six in the four parameters, the derivatives in y and the mixed
ones, the starting value and the draw; the expected information, the cdf family and the quantile
are the generic routes.

**How.** The assembly is the location-scale one with two smooth shape parameters,
`detail_location_scale::mixed_two_shape_dk` (added for this family: it is `mixed_shape_dk` with
one shape more and is pure index work over `mixed_dk`), fed with the table of mixed
(alpha, nu, z) derivatives of Lambda + log T_(nu+1)(w). The first term is student_t1's table.
The second is a composition in three variables, which two facts reduce to compositions in two:

- alpha enters w = alpha u(z, nu) linearly, with u free of alpha, so
  d^c/dalpha^c log T_(nu+1)(w) = u^c (d^c log T/dw^c)(w, nu + 1) exactly: no Bell sum in alpha,
  only a power of u, whose own (z, nu) derivatives come from
  `math::chain_rule_pair_scalar_dk` on a power;
- what is left is a function of the two arguments of the Student cdf composed with the two inner
  maps w = alpha u(z, nu) and m = nu + 1 in the two variables (z, nu), which is
  `math::chain_rule_pair_dk` with the outer table shifted by c in its first argument and the
  second inner map linear.

The two tables are then multiplied by Leibniz's rule. The only new closed forms are the mixed
(z, nu) derivatives of u = z sqrt((nu + 1)/(nu + z^2)), from the recurrence

    d/dz [z^m B^-s] = m z^(m-1) B^-s - 2 s z^(m+1) B^-(s+1),   B = nu + z^2,

every exponent staying a half-integer, with d^q/dnu^q B^(-1/2) = prod_(i<q) (-1/2 - i)
B^(-1/2-q) since dB/dnu = 1, and the factor z joined by
d^p/dz^p [z W] = z d^p W + p d^(p-1) W. The derivatives in y follow from d/dy = -d/dmu.

The draw is the exact representation and not the inversion of the cdf:

    Y = mu + sigma (delta |N_1| + sqrt(1 - delta^2) N_2) / sqrt(G/nu),
    delta = alpha/sqrt(1 + alpha^2),   G ~ chi^2_nu,

with the two normals and the gamma each by inverting their own cdf
(`standard_normal_by_inversion` and `standard_gamma_by_inversion`, added to `generic_rng.hpp`:
`Phi_inv` and `gamma_p_inv` of one uniform). Inversion rather than a rejection method because one
draw then consumes exactly one number of the stream, so the draw of an observation is the same on
every thread and backend, and a kernel takes no divergent branch.

**Measured.** Log-density against mpmath, in ulp of the largest component of the order, over 180
points (18 parameter quadruples times ten values of z) with nu from 0.3 to 1e6, alpha from -20 to
50, |z| to 1e8 and sigma from 2^-10 to 2^10; 37620 rows and none dropped.

| order | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| gcc | 4 | 9 | 63 | 99 | 336 | 1.3e3 |

The worst points are the small shape (nu = 0.3) at order six, the far tail with |alpha| large at
the orders four and five, and nu = 6 with |z| in the thousands at the orders two and three. By
nu the worst over every order is 1.3e3 at 0.3, 508 at 2, 1.12e3 at 4, 265 at 12, 203 at 200 and
568, 298 and 387 at 1e3, 1e4 and 1e6: the large-nu region is as healthy as the centre.

The reference deserves its own note. It is built by SymPy's own chain rule over the definition,
with log T_(nu+1) carried as an unknown function of two variables whose derivatives are
substituted, so no closed form of the header enters; those derivatives come from the integral of
the density's nu-derivatives, which is legitimate because they decay like |u|^(-M-1) times powers
of a logarithm. The first version of that integral used panels of fixed width and lost everything
but three digits in the deep tail at large nu, where the density decays over a length
(m + x^2)/((m + 1)|x|) of about 0.07: it disagreed with the library by 1e15 ulp, and a third route
(mpmath's own incomplete beta) showed the library right to 2e-16 and the reference wrong. With
geometric panels from that length outwards and the integrand scaled by its value at the endpoint
the agreement with the incomplete beta is 3.2e-55 over the whole grid, and that comparison is now
a mode of the script (`python -I tools/reference/distrib_skewt_ref.py check`).

Other measurements: the value against the definition 3.8e-16; the orders one to three against
nested central differences 2.0e-6, 1.1e-5 and 6.7e-5, which is the differences' own resolution;
the mixed (y, theta) derivatives 1.1e-5; at alpha = 0 the family is `student_t1_distrib` to the
last bit at every order in the three parameters they share, which also shows that the whole
log T branch and its derivatives vanish exactly there; at nu = 1e6 and 1e8 the value against
`skewnormal_distrib` at the parameters the direct-to-centred map implies, 7.7e-6 and 7.7e-8, the
1/nu rate of the limit; the generic cdf against the density by central differences and against
the quantile by root finding; the expected information in its two forms within 1e-7 of each other
and of student_t1's closed information at alpha = 0; the tensor of order six identical on the
host and in a Kokkos kernel.

The draw costs 529 ns a call on one gcc thread, against 5.8 ms by inversion of the generic cdf
(whose log F costs 442 us and whose quantile by root finding 8.3 ms), eleven thousand times less.
The empirical cdf of 2e5 draws agrees with the generic cdf within two standard errors at five
points, and the sample mean and variance with the closed moments.

**Chosen.** As above, with the bounds of `tests/distrib/test_skewt.cpp` (13 cases, 40373
assertions) at 1.5 times the worst of the two builds. The closed forms of u are checked against
SymPy by `python -I tools/reference/distrib_skewt.py forms`, which verifies the three identities
the recurrence encodes and a reimplementation of the whole recurrence against SymPy's own
derivative of u for every p + q <= 6 at three points.

**Open.** The generic cdf of a four-parameter family inside a CUDA kernel runs with the
per-thread stack raised to 8192 bytes, which the test executable now asks the driver for before
its first launch; the price measured on an RTX 4070 SUPER is 588 MB of device memory out of
12282. A family with more parameters will need more, and the figure has to be measured again
then.

## Start values of the structures: params_from_matrix

What: `params_from_matrix(m_hat, theta)`, a host member of every structure
that maps a target symmetric matrix (packed by rows) to parameters inside the
domain. Helpers in `include/laplace/pdmat/detail/from_matrix.hpp`, tests in
`tests/pdmat/test_params_from_matrix.cpp`.

How the parameters are found:

| structure | route |
|---|---|
| `diagonal_pdmat` | theta_i = log(M_ii)/2 |
| `ar1_pdmat` | rho = mean lag-1 correlation |
| `compound_symmetry_pdmat` | rho = mean off-diagonal correlation, (rho - a)/(1 - a) = expit(theta), a = -1/(p - 1) |
| `toeplitz_pdmat` | m_k = mean correlation on diagonal k, partial autocorrelations by Durbin-Levinson |
| `unstructured_pdmat<B>` | the bijector's forward map (on the correlation matrix for partial_corr) |
| `unit_diagonal_scaled_pdmat<R>` | s_i = log(M_ii)/2 (their mean with common_scale), then R's |
| `factor_analytic_pdmat` | 100 steps of principal axis factoring from the probabilistic PCA fit, then the trapezoid |
| `reduced_rank_psdmat` | the d leading eigenvectors scaled by the square roots of their eigenvalues, then the trapezoid |
| `weighted_sum_psdmat` | lambda_j by least squares in the Frobenius inner product |

The trapezoid (lower trapezoidal Lambda with positive diagonal, the
convention of `factor_analytic_pdmat`) is the first d columns of the Cholesky
factor of A A^T. The structures with unit diagonal read the target as a
covariance and take its correlations. Guards: correlations clamped to
[-0.99, 0.99] (`start_rho_max`; for compound symmetry the clamp is on
expit(theta), [0.005, 0.995]), eigenvalues and diagonal entries floored at
1e-6 times the largest (`start_eigen_floor`); a target is changed only when a
guard acts, so the exact routes stay exact.

Measured (gcc 14.2, OpenMP), params_from_matrix(matrix(theta)) against theta,
max |difference|:

| structure | error |
|---|---|
| diagonal (p = 4), ar1 (p = 5, theta = 0.7), unstructured log_cholesky (p = 4), unit_diagonal_scaled<ar1> common, compound_symmetry (p = 2) | 0 |
| compound_symmetry (p = 4), toeplitz (p = 5), unstructured_corr (p = 4), unit_diagonal_scaled<ar1>, <corr>, reduced_rank (5, 2) | 5.6e-17 to 1.7e-16 |
| unstructured matrix_log (p = 3), ar1 at theta = -1.9 | 5.6e-16, 8.9e-16 |
| weighted_sum (three 6 x 6 difference penalties) | 2.1e-13 |

The weighted sum loses three digits to the conditioning of the 3 x 3 Gram
matrix of the penalties. Factor analysis (p = 8,
d = 2) reaches 1.1e-3 on a matrix of its own structure and 0.17 on a generic
positive definite matrix: a start value, not an estimate. Degenerate targets
(rank one with a zero diagonal entry, all correlations equal to 1) give finite
theta inside the guards.


## The multivariate Gaussian: mvgaussian1 and mvgaussian2

What: the first multivariate distribs (design in `note/distrib_multivariate.md`,
2026-11-15): `include/laplace/distrib/multivariate_core.hpp` (the concept
`multivariate_distrib`, `distrib_normalization`, the information slots (an enum of three blocks until the typed parameter list of 2026-11-17),
`distrib_mv_sample_summary` and `summarize_mv_sample`),
`include/laplace/distrib/mvgaussian1.hpp` (covariance Sigma = M(theta)) and
`include/laplace/distrib/mvgaussian2.hpp` (precision Q = M(theta)), over any
pdmat structure; the shared code in `distrib/detail/mvgaussian_common.hpp`.
Test `tests/distrib/test_mvgaussian.cpp`, references
`tools/reference/distrib_mvgaussian.py`, bench `benchmarks/distrib_mvgaussian.cpp`.

How. `theta_state(theta, w)` writes log det M (the structure's `log_det`),
the matrix W of the quadratic form (Sigma^-1 by `inverse`, or Q by `matrix`)
and the Cholesky factor of M (the structure's `cholesky`, or a Cholesky
factorization of `matrix` when it has none). The blocks with A derivatives in
mu and K in theta along a tuple T are, with r = y - mu and c = -1/2 (Sigma)
or +1/2 (Q): A = 0, c d_T log det M - 1/2 r' d_T W r; A = 1, d_T W r;
A = 2, -d_T W; zero for A >= 3. d_T W is the structure's `inverse_dk`
(mvgaussian1) or `matrix_dk` (mvgaussian2); nothing new in the numerics. The
expected information in theta is I_ab = -c d_ab log det M + 1/2 tr(V d_ab W),
V = W^-1, and its derivatives to order 4 by Leibniz's rule on the trace,
d_T I_ab = -c d_abT log det M + 1/2 sum_(U subset T) tr(d_U V d_(ab, T\U) W).
Draws y = mu + L z (Sigma = L L') or mu + L^-T z (Q = L L'), z by inversion
from `counter_uniform_source{seed, unit}`. `start_value` floors the
eigenvalues of the sample covariance (divisor n) at 1e-6 of the largest,
inverts it for mvgaussian2 and calls the structure's `params_from_matrix`.

References: mpmath at 60 digits from the definition (log det and the solve by
mpmath's Cholesky), the blocks by `mpmath.diff` over (mu, theta); the
information as d^2 F(t', theta)/dt'_a dt'_b at t' = theta with
F = E_theta[-l(t')] up to a constant (F1 = 1/2 log det Sigma(t') +
1/2 tr(Sigma(t')^-1 Sigma(theta)), F2 likewise), its derivatives by
`mpmath.diff` in theta. Every value computed at 60 and 75 digits, agreement
within 1e-40 (1 + |value|). 5693 values: six structures (diagonal, ar1,
compound_symmetry, unstructured log_cholesky, unit_diagonal_scaled<ar1>,
factor_analytic with one factor), d = 2 to 5, a random point and a hard one
each (log L22 = -11.5, rho = tanh 3.5, compound symmetry at theta = 8, scales
e^+-2 with rho = tanh(-2.5), psi = e^-6, residuals up to 30); A = 0 to order
6, A = 1 to 5, A = 2 to 4, the information to order 4. 1017 s on 4 processes.

Error measure: ulp1N, the largest error of the block over its entries in ulps
of the largest |ref| over every tuple and entry of the same point, A and K
(and at least 1); for the information at least the largest |I_ab| at that
point. A per-entry measure is wrong here: at the log_cholesky hard point the
information has entries of 2.3e10 next to exact zeros and entries of order 1.

Results, worst ulp1N over the points, gcc/OpenMP and CUDA (RTX 4070 SUPER):

| | blocks A = 0..2, orders 0..6 | information, orders 0..4 |
|---|---|---|
| mvgaussian1: diagonal, compound_symmetry, log_cholesky, scaled_ar1 | 5.6 (gcc and CUDA) | 286 (gcc), 238 (CUDA), log_cholesky order 4; compound_symmetry 1080 (CUDA) |
| mvgaussian1: ar1 | 34 (gcc), 22 (CUDA), hard point | 8150 (CUDA, order 4) |
| mvgaussian1: factor_analytic | 3.1e6 (gcc), 1.4e6 (CUDA), hard point | 1.0e6 |
| mvgaussian2, every structure | 9.0 (gcc), 4.0 (CUDA) | 7.7e4 (factor_analytic, order 2), 1030 (ar1), else below 100 |

What this says:
- The blocks are as accurate as the structure's derivatives: a few ulp where
  those are. The large figures are all at hard points.
- factor_analytic at psi = e^-6 (mvgaussian1, so through `inverse_dk`): up to
  3.1e6 ulp1N, while one ulp of any theta_k moves the exact A = 0 blocks of
  order 3 by at most 10.5 ulp (mpmath). The loss is in
  `factor_analytic_pdmat::inverse_dk` at that point, not in the conditioning;
  that structure's own test has a bound of 5.5e5 ulpN on M^-1. mvgaussian2,
  which uses its `matrix_dk`, is within 1.3 ulp there.
- The information of mvgaussian1 near a singular Sigma loses up to about
  cond(Sigma) eps relative to the largest entry: the traces
  tr(d_U Sigma d_(ab, T\U) Sigma^-1) multiply small derivatives of Sigma by
  large ones of Sigma^-1 and cancel (ar1 at rho = tanh 3.5: 8150; compound
  symmetry at theta = 8: 1080). Normwise over the information this is small;
  per entry, an entry of order 1 next to entries of 1e10 has no correct
  digits. mvgaussian2 does not cancel at the same points.
- The two parametrizations agree to the bit where M(-theta) = M(theta)^-1
  (diagonal, unstructured matrix_log): every block and the information, with
  the sign (-1)^K (0 ulp on gcc, 0.7 on CUDA).
- Orders 5 and 6 and the information's orders 1 to 4 agree with central
  differences of the order below within 1e-6 of the scales of the two orders
  (1e-5 for factor_analytic, whose hard point has large orders K + 2).
- Draws, 2e5 units on the device: mean and covariance (by
  `summarize_mv_sample`) within 1.2 standard errors of mu and of the
  covariance, the mean of -d^2 l within 1.2 standard errors of the information
  in (theta, theta) and of zero in (mu, theta), identical on gcc and CUDA (the
  counter source). `start_value` reproduces the sample covariance to 1e-12
  relative for log_cholesky in both parametrizations.

Speed, ns per unit, n = 1e5, unstructured log_cholesky; order 0 the
log-density, order 1 the gradient (mu and every theta_j), order 2 every block
of order 2; theta shared (state computed once) or per unit (each unit computes
its state). Each cell: OpenMP 16 threads; CUDA.

| | d | shared, orders 0 / 1 / 2 | per unit, orders 0 / 1 / 2 |
|---|---|---|---|
| mvgaussian1 | 2 | 1.3 / 28 / 103; 0.3 / 5.3 / 29 | 8.9 / 34 / 113; 1.0 / 6.0 / 30 |
| | 3 | 1.6 / 71 / 463; 0.3 / 23 / 221 | 16 / 81 / 507; 2.0 / 25 / 223 |
| | 5 | 3.2 / 310 / 5316; 0.6 / 197 / 3989 | 33 / 334 / 5314; 6.1 / 211 / 4118 |
| | 10 | 7.6 / 3925 / 2.38e5; 2.0 / 3407 / 2.37e5 | 88 / 4169 / 2.40e5; 26 / 3351 / 2.38e5 |
| mvgaussian2 | 2 | 1.4 / 6.4 / 11; 0.3 / 1.0 / 1.8 | 5.6 / 10 / 15; 0.8 / 1.5 / 2.5 |
| | 3 | 2.0 / 15 / 32; 0.3 / 2.8 / 6.7 | 12 / 22 / 37; 1.4 / 3.9 / 8.0 |
| | 5 | 2.3 / 38 / 137; 0.6 / 17 / 55 | 20 / 50 / 151; 3.5 / 20 / 59 |
| | 10 | 7.1 / 248 / 2021; 2.3 / 186 / 922 | 50 / 287 / 2118; 13 / 201 / 956 |

What this says: the log-density follows the plan, O(d^2) a unit with a shared
theta and O(d^3) with theta per unit (7 to 13 times at d = 10). The
derivatives in theta do not: every block with K >= 1 recomputes d_T W, which
depends on theta only, inside the unit, so a shared theta saves almost
nothing at orders 1 and 2. For mvgaussian1 over log_cholesky that is the
generic route to d_T Sigma^-1 (about 3^K d^3 / 6 a tuple): 2.4e5 ns a unit at
d = 10, order 2, 100 to 250 times mvgaussian2. Open: how the shared-theta case
avoids it (d_T W computed once outside the units, or the sum over the units of
r r' formed first so that the theta blocks are traces against one matrix)
belongs to the fitting engine and is to be decided with Giovanni.

## Which side the derivatives of the incomplete beta compute

**What.** `log_ibeta_dK` and `log_ibetac_dK` each ask for one side of the
distribution, and the module can either compute that side or compute the other
one and compose the complement. Which it does was decided by the sign of
lambda = a - (a + b) x, the quantity whose sign says which side is monotone away
from its end point: the monotone side was computed, and when that was not the
side asked for, the answer came from the complement. The composition is the
lossy step, and the choice ignored how well the requested side's own
computation would do.

**What it cost.** At x equal to the mean a/(a + b), lambda is zero in exact
arithmetic and comes out negative by one rounding (at (a, b, x) = (1, 9, 0.1),
where 9 * 0.1 is not 0.9, lambda is -5.55e-17), so the complement was taken
there. Against mpmath at 60 digits on the pure-a derivatives:

| (a, b, x) | probability of the requested side | the complement | the side's own computation |
|---|---|---|---|
| (1, 9, 0.1) | 0.6126 | 2.7e-7, 1.4e-4 | 1.9e-16, 1.1e-13 |
| (1, 9, 0.16) | 0.7918 | 5.5e-9, 1.4e-6 | 7.1e-16, 1.8e-13 |
| (1, 30, 0.0516) | 0.7960 | 4.7e-9, 1.4e-6 | 1.7e-16, 1.4e-12 |
| (1, 2, 0.5333) | 0.7822 | 1.9e-16, 2.0e-15 | 3.7e-16, 1.8e-12 |

the pairs being the orders one and six. It is a region and not a point: every
lambda < 0 took the complement, and the loss is largest where the requested
side holds between a half and nine tenths of the mass.

**How it is chosen now.** The monotone side is still computed first, and when it
is the side asked for nothing changes. When it is not, the requested side is
computed directly instead of composing the complement, under two conditions,
both measured:

- its own shape is at least `own_min`, as in the first choice of the route.
  Below that its own logarithm carries the pole of a small shape, whose
  derivatives (n-1)!/shape^n cancel in it, and only the composition from the
  other side avoids them: at (5.47, 0.21, 0.864) the upper side taken directly
  gives 4.1e7 ulp against the complement's 2e3;
- its probability is at most `p_own = 0.7`. Above that the side holds nearly all
  the mass, its logarithm is near zero and its derivatives are differences of
  large cumulants, which is what the complement from the small side exists for.
  Sweeping x on log I_x(1, 9), the two routes cross at a probability of about
  0.99 in the relative error of a component (at 0.9899 both are at 1e-14 and
  2e-11, and at 0.998 the direct route gives 2.6e-5 against the complement's
  1.6e-17), but in the ulp measures of the module's own test the crossover is
  lower: at 0.9 and at 0.8 a point or two of the reference grid lost digits,
  and at 0.7 every order of both functions is at or below what it measured
  before.

**Measured.** The module's reference grid, worst over it, before and after, on
gcc; the bounds in `tests/math/test_beta_deriv.cpp` are 1.5 times the worst of
the two builds.

| order | log I_x ulpN before / after | log(1 - I_x) ulpN before / after |
|---|---|---|
| 1 | 1067 / 751 | 1600 / 1205 |
| 2 | 1200 / 836 | 1367 / 1444 |
| 3 | 2067 / 768 | 1633 / 1616 |
| 4 | 1200 / 822 | 1967 / 1600 |
| 5 | 2200 / 1437 | 1967 / 1640 |
| 6 | 1553 / 1298 | 5367 / 5357 |

Two cells of the bounds are set by CUDA rather than by gcc, through the
contraction of a product and a sum into one operation: `log_ibeta_d5` measures
298.6 ulpL there against 184.9 on gcc, and `log_ibetac_d6` 175.8 against 110.0.

**What it says.** A branch keyed on the sign of a quantity that vanishes
identically on a surface is wrong on one side of that surface, and here the
surface is the mean of the distribution equalling the observation, which is not
an exotic place to evaluate a cdf. The repair does not trade one region for
another: every order of both functions improved or stayed, and the region that
was losing seven digits at order one now keeps them all.

**Open.** The condition is a pair of thresholds, not a criterion derived from
the conditioning of the two routes. A criterion would compare the two directly,
which costs what it saves, so the thresholds stay until a cheap estimate of the
complement's conditioning is worth writing.

## The incomplete beta where a frame's other shape is one

**What.** Each side of the distribution is computed in a frame in which "a" is
the side's own shape, the exponent of t, and "b" the other one: the upper side
of (a, b, x) is the lower side of the frame (b, a, y). When the other shape of
the frame is exactly one, the integrand t^(a-1) (1 - t)^(b-1) loses its factor
in (1 - t) altogether, and the quadrature's anchor and scale, which read the
structure of both factors, have only one of them left to read. The side's own
computation then loses digits, and the loss reaches the shape derivatives of
`log_ibetac_dK` at a = 1, whose frame is (b, 1, y).

**Measured.** Six frames, asking for log(1 - I_x(a, b)), against mpmath at 60
digits; the pairs are the orders one and six, and the times are per call on one
gcc thread for the whole table to order six.

| (a, b, x) | the side's own quadrature | the series | the complement | us: quad / series / complement |
|---|---|---|---|---|
| (1, 9, 0.1) | 2.7e-7, 1.7e-7 | 2.3e-15, 1.8e-15 | 2.8e-17, 3.8e-15 | 2.25 / 9.81 / 8.57 |
| (1, 9, 0.15) | 9.9e-9, 1.2e-8 | 8.7e-16, 7.1e-16 | 1.7e-15, 5.9e-15 | 2.04 / 6.81 / 8.78 |
| (1, 9, 0.06) | 2.4e-7, 8.2e-8 | 2.4e-15, 5.9e-15 | 3.0e-16, 2.0e-15 | 6.45 / 15.71 / 4.24 |
| (1, 4, 0.2) | 4.4e-7, 1.6e-7 | 1.1e-15, 3.8e-18 | 1.7e-16, 1.2e-15 | 2.15 / 5.28 / 8.51 |
| (0.5, 20, 0.1) | 2.8e-17, 2.4e-17 | 1.2e-16, 1.4e-16 | 1.8e-14, 1.6e-13 | 3.70 / 10.34 / 9.65 |
| (0.8, 9, 0.1) | 2.6e-17, 3.9e-16 | 2.9e-16, 1.0e-15 | 3.3e-16, 7.8e-15 | 3.99 / 10.21 / 8.89 |

The first four rows have the frame's other shape at exactly one and lose seven
to nine digits; the last two have it at 0.5 and 0.8 and keep them all. Both of
the other routes are accurate in that corner, at three to five times the
quadrature's cost, and no cheaper accurate route was found.

**Chosen.** The series on the requested side is taken in its wide form, up to
`x_wide`, when the frame's other shape is at most one, instead of only up to
x = 1/2, and `x_wide` goes from 0.8 to 0.95. The condition is the corner
itself, since `series_ok` tests the frame's other shape against one, and the
series is the route already wired for it; the complement measures the same and
costs the same, so there is nothing to choose between them on the numbers.

The module's whole reference grid measures identically to the digit before and
after the change, so no bound of
`tests/math/test_beta_deriv.cpp` moves: the corner is not in that grid, which
is why nothing caught it from inside the module. Above the cut, at a frame x
from 0.95 to 0.98, the quadrature measures 1e-15 at the orders one and six, so
the corner is closed on both sides of `x_wide` and not moved.

**What it says.** A frame with its other shape at one is the degenerate point
of the family, where I_x(1, b) = 1 - (1 - x)^b in closed form, and a quadrature
whose machinery reads the structure of two factors should be expected to lose
something where one of them is constant. The defect was reported from outside,
by a family composing `log_ibetac_dK` with two linear shape maps, and it was
found by the same evidence pattern as the route defect above: mpmath and the
quadrature of the generic cdf, which reads only `logpdf` and `logpdf_dk` and
nothing of the incomplete beta, agreed with each other against the closed
route.

**Open.** The quadrature's own weakness in the corner is not repaired, only
avoided: if a configuration ever needs the quadrature there, with the other
shape at one and x above `x_wide`, the anchor and the scale have to read the
single factor that is left.

## mvgaussian1 and mvgaussian2: the factor form of the information and the theta table

What: two changes decided with Giovanni on 2026-11-16 (`note/distrib_multivariate.md`
section 12, items 2 and 3), in `include/laplace/distrib/detail/mvgaussian_common.hpp`
and the two headers; tests and bench as in the section above.

The factor form. The information of mvgaussian1 in theta lost up to cond(Sigma) eps
near a singular Sigma through the traces of the Leibniz route (section above). Over a
structure with its Cholesky factor (`closed_factor`) it is now
I_ab = 1/2 tr(B_a B_b), B_a = G_a + G_a', G_a = L^-1 d_a L, and
d_T I_ab = 1/2 sum_(U subset T) tr(d_U B_a d_(T\U) B_b). Every derivative of G is
formed from whitened lower-triangular matrices only: H[m] = L^-1 L[m] by forward
substitution (L[m] the structure's `cholesky_dk` along the indices at the positions
of the mask m), X[S, R] = Y[S] L[R] with X[0, R] = H[R] and
X[S, R] = -sum_(0 != T subset S) H[T] X[S\T, R] (Y = L^-1), and
d_S G_a = sum_(S1 subset S) X[S1, {a} + (S\S1)]; with G lower triangular,
1/2 tr(B1 B2) = sum_i G1_ii G2_ii + sum_(j <= i) G1_ij G2_ij. No large matrix is
formed, so nothing cancels. factor_analytic and matrix_log (no `closed_factor`) keep
the trace route, as does mvgaussian2, which does not cancel. Scratch: 113 d^2 doubles
plus a packed matrix and the structure's scratch, at order 4 (`info_scratch_size()`).

Worst ulp1N of the information over orders 0 to 4 (normwise, as above), trace route
before and factor form now, gcc / CUDA:

| structure | trace route | factor form |
|---|---|---|
| ar1 | 4560 / 8150 | 16.4 / 16.3 |
| compound_symmetry | 664 / 1080 | 1.35 / 1.35 |
| log_cholesky | 286 / 238 | 1.25 / 1.25 |
| unit_diagonal_scaled<ar1> | 67 / 68 | 11.6 / 11.6 |
| diagonal | 1 / 1 | 0 / 0 |

Measured first in a host prototype (one thread, heap allocations): 1.1 to 4.9 times the
trace route's cost per pair on log_cholesky; the information depends on theta only.

The theta table. Every block with K >= 1 recomputed d_T W inside the unit, so a shared
theta saved nothing at orders 1 and 2. The table holds, for the sorted tuples of orders
1..k_max in lexicographic order, one entry of np + 2 doubles: a flag (d_T W not zero
by structure), d_T log det M and d_T W packed. `theta_table_entries(k_max)`,
`theta_table_size(k_max)`, `fill_theta_table_entry(state, k_max, entry, table, w)`
(one entry per call, so that a kernel fills it in parallel) and
`with_theta_table(state, k_max, table)`, which attaches it to the state; the unit
functions (`logpdf_dk`, `logpdf_dy_dk`, the (mu, mu) block of `expected_info_dk`)
read it when it covers the order asked for and compute as before otherwise. Test:
every block of orders 1 to 3 of the reference file read from a table equals the block
computed directly, to the bit, on gcc and CUDA.

Speed, ns per unit, n = 1e5, unstructured log_cholesky, theta shared, without and with
the table (orders 1 / 2), and the fill of the table of order 2 (once per theta),
OpenMP 16 threads and CUDA:

| | d | OpenMP, shared | OpenMP, table | CUDA, shared | CUDA, table | fill (OpenMP / CUDA) |
|---|---|---|---|---|---|---|
| mvgaussian1 | 2 | 32 / 128 | 3.7 / 8.3 | 6.8 / 31 | 0.5 / 0.9 | 81 / 35 us |
| | 3 | 77 / 570 | 7.2 / 27 | 25 / 210 | 0.9 / 3.0 | 71 / 45 us |
| | 5 | 342 / 6217 | 27 / 188 | 209 / 4050 | 3.7 / 29 | 96 / 119 us |
| | 10 | 4526 / 2.50e5 | 158 / 4775 | 3412 / 2.40e5 | 36 / 1002 | 414 / 497 us |
| mvgaussian2 | 2 | 9.5 / 15 | 3.4 / 7.0 | 1.2 / 2.1 | 0.5 / 0.7 | 90 / 13 us |
| | 3 | 18 / 41 | 7.9 / 21 | 3.0 / 7.2 | 1.0 / 2.0 | 95 / 14 us |
| | 5 | 48 / 163 | 22 / 119 | 17 / 52 | 3.7 / 12 | 102 / 15 us |
| | 10 | 267 / 2070 | 157 / 1953 | 185 / 908 | 35 / 245 | 70 / 17 us |

What this says: with the table, a shared theta costs a unit what the reading of the
blocks costs, O(d^2) a tuple; mvgaussian1 at d = 10, order 2 drops 52 times on OpenMP
and 240 times on CUDA, and the two parametrizations come within a factor of 2.4
(OpenMP) and 4.1 (CUDA) of each other. The fill is paid once per theta: 0.4 ms
on OpenMP and 0.5 ms on CUDA at d = 10, order 2 for mvgaussian1, below 0.1 ms
elsewhere. Order 0 is unchanged (8.8 and 6.9 ns at d = 10 on OpenMP, noise). These
timings were taken on quiet cores under the build lock; the earlier table of the
section above was taken while another build ran, and its figures differ from these
by up to 25 per cent (mvgaussian1 at d = 2, order 2: 103 then, 128 now).

Open: the block A = 0, K = 6 of mvgaussian1 over log_cholesky moved from 4.6 to 10.0
ulp1N on gcc with an uncommitted change of another session to the generic route of
M^-1 in `pdmat/core.hpp` (sums over classes of equal indices); its bound (8) is left
until that change is final.

### Against frozen statmodels7 results (2026-11-16)

statmodels7's results for the multivariate part are frozen into the tests, so that
they survive its retirement. The script `tools/reference/statmodels7_multivariate.R`
(statmodels7 used as an oracle only: it calls distributions7 0.94.2, penalties7 0.34.0
and parameters7 0.27.0 and writes what they return) writes two files of hex doubles:

- `tests/data/reference/mvgaussian_s7.csv`, in the format of `mvgaussian.csv`:
  mvgaussian1 and mvgaussian2 over `unstructured_pdmat<log_cholesky_bijector>`, every
  block of the log-density with A <= 2 derivatives in mu and A + K <= 4 (the orders
  statmodels7 has, from its log-density, gradient, Hessian and third and fourth
  derivatives), and the expected information in theta at order 0; d = 2, 3, 4 at
  random points and d = 3 at a near-singular one (log L33 = -8); 8546 values.
- `tests/data/reference/penalty_s7.csv`: statmodels7's `structured_penalty` (precision
  log-Cholesky M(theta), at the same four points) and `additive_penalty`
  (sum_k lambda_k P_k: a full-rank pair at q = 4, a 3 x 3 tensor of second
  differences of rank 8 at two lambda, one with lambda_2 / lambda_1 = 1e6): the value,
  log pdet, the gradient in beta, the gradient and Hessian in theta and the mixed
  block in (beta, theta), on the parameter scale; 317 values. Laplace has no penalty
  code yet, so these are data only, with no test. Checked against an independent
  numpy evaluation (the value from the definition, the theta gradient by
  differences, the additive gradient on the fixed range of the sum): agreement to
  1e-12 relative.

The two libraries agree on the log-Cholesky factor (L_ii = exp(t_ii), L_ij = t_ij,
M = L L') and differ in the order of t: Laplace packs the lower triangle by rows
(t_ij at i (i + 1)/2 + j), statmodels7 puts the p log-diagonal entries first and then
the strict lower triangle by columns. The files are written in Laplace's order; the
map is in the script's header.

The test (`mvgaussian_s7_case` in `tests/distrib/test_mvgaussian.cpp`) evaluates
every block in a Kokkos kernel and measures the error in ulp1N as the mpmath case
does. Worst over orders, gcc / CUDA:

| | regular points | near-singular point |
|---|---|---|
| mvgaussian1, blocks | 4.75 / 2.72 | 0.94 / 0.86 |
| mvgaussian1, information | 1.99 / 1.99 | 2.18e7 / 2.18e7 |
| mvgaussian2, blocks | 1.61 / 1.61 | 2.01 / 2.01 |
| mvgaussian2, information | 3.98 / 2.98 | 1.97e7 / 1.97e7 |

What this says: the two libraries agree to a few ulp on every block, at every point.
At the near-singular point the information differs by 2e7 ulp1N (5e-9 normwise, on
entries up to 1.1e7); mpmath at 60 digits puts the whole difference on statmodels7's
side (2.18e7 and 1.97e7 ulp1N from mpmath), while Laplace is within 3 ulp1N of it
there (the mpmath case). The bounds are 1.5 times the larger of the two builds,
rounded up. Those two cells are excluded from the test: the error is on
statmodels7's side, as mpmath shows, and Laplace at that point is covered by the
mpmath case. They stay in the CSV, which records what statmodels7 gives.

## Polygamma functions to order 13 in double-double

What: `include/laplace/math/polygamma_dd.hpp` (2026-11-16), in `laplace::math`:
psi_n(x), G_n(x, h) = psi_n(x + h) - psi_n(x) for h > 0, and
r_n(x) = psi_n(x) - mu_n(x) with mu_0 = log x and mu_n = (-1)^(n+1) (n-1)! x^-n,
for n = 0..13, arguments and results in double-double. Each comes as a single-order
function (`polygamma_dd`, `polygamma_difference_dd`, `polygamma_monomial_remainder_dd`),
which is the general route, and as an all-orders function filling out[0..N]
(`polygamma_orders_dd`, `polygamma_difference_orders_dd`,
`polygamma_monomial_remainder_orders_dd`). The all-orders form is an exception to one
function per order, accepted for this layer only because it was measured to pay
(Giovanni, 2026-11-16; table below). It is the first prerequisite of the generalized
gamma in the mean, whose order-six derivatives need polygammas to order 12-13 in
double-double towards the lognormal limit d/p -> inf (`note/piano_distrib_restanti.md`
section 5).

How: below X0 = 20 the shift recurrence, at z >= 20 the asymptotic expansion with at
most 36 Bernoulli terms per order, stopped when a term falls below 2^-110 of the
leading one. The coefficients b_(0,k) = B_2k/(2k) and b_(n,k) = B_2k (2k+n-1)!/(2k)!
are exact rationals from SymPy's Bernoulli numbers, rounded to (hi, lo)
(`tools/reference/polygamma_dd_coefficients.py`). G_n is never formed as a difference:
with u = 1/z, v = 1/(z + h) after the common shift, u^p - v^p = h u v S_p with
S_1 = 1, S_(p+1) = u S_p + v^p, all terms positive, and the same identity for the pole
terms a_j^-(n+1) - b_j^-(n+1), a_j = x + j, b_j = a_j + h; log((z + h)/z) is
2 atanh(h/(2z + h)) by its series below h = z/2. r_n leaves the leading term out of
the expansion and adds mu_n(z) - mu_n(x) in closed form (log1p(m/x) by the same
series, x^-n - z^-n = m x^-1 z^-1 S_n(1/x, 1/z)). The table (8064 bytes) is in global
device memory (`LAPLACE_DEVICE_GLOBAL_TABLE`), not in the constant bank (below).

Threshold X0. Bernoulli terms needed for a truncation below 2^-107 of the bracket at
z = X0 (mpmath):

| X0 | n = 0 | n = 1 | n = 6 | n = 9 | n = 13 |
|---|---|---|---|---|---|
| 15 | 21 | 22 | never | never | never |
| 18 | 18 | 18 | 26 | 32 | 45 |
| 20 | 16 | 16 | 23 | 27 | 33 |
| 25 | 14 | 14 | 18 | 21 | 24 |
| 30 | 13 | 12 | 16 | 18 | 20 |
| 40 | 11 | 11 | 14 | 15 | 16 |

At 15 the divergent series never reaches 2^-107 for n >= 6; at 18 order 13 needs 45
terms, more than the table holds. Cost on one
thread (gcc) over x in [1e-2, 1e8], at the same accuracy within a unit: psi_13 576,
656, 778, 959 ns and G_13 2202, 2441, 2801, 3525 ns at X0 = 20, 25, 30, 40. The
recurrence step (a double-double division and a power) costs more than the series
terms it saves, so the smallest admissible X0 wins: 20.

References: `tools/reference/polygamma_dd.py`, mpmath at 300 bits from `mpmath.psi`
(Hurwitz zeta), each row recomputed at 400 bits and kept when the two agree to
2^-200; values stored as three doubles (`tests/data/reference/polygamma_dd.csv`, 8022
rows). psi_n and r_n at x from 1e-6 to 3.3e15, dense from 0.25 to 60 and on both sides
of X0, and at double-double arguments d/p; G_n at x from 1e-3 to 1e13 against h from
1e-3 to 333 (x/h from 3e-6 to 1e16) and at the generalized gamma's pairs x = d/p,
h = 1/p, d = 10..1e9, p = 0.003..30. mpmath.psi was checked once against the integral
representations psi_n(x) = (-1)^(n+1) int_0^inf t^n e^(-xt)/(1 - e^(-t)) dt and their
analogues for G_n (with -expm1(-ht)) and psi(x) - log x, by quadrature, at 55 points
up to x = 1e13 and n = 13: agreement to 1.2e-75. (mpmath.nsum of the defining series
failed at large x, and mpmath.quad stops on an absolute tolerance: the integrands,
near 1e-160, had to be scaled.)

Accuracy: worst over the file, units of 2^-104 of the value (psi_0: of max(|f|, 1)),
identical on gcc/OpenMP and CUDA (`benchmarks/math_polygamma_dd.cpp`):

| n | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| psi_n, one order | 0.96 | 0.74 | 1.04 | 1.26 | 1.27 | 1.50 | 2.07 | 2.10 | 2.63 | 3.08 | 3.15 | 3.53 | 3.93 | 4.47 |
| psi_n, all orders | 0.96 | 0.74 | 1.04 | 1.26 | 1.75 | 2.05 | 2.15 | 2.01 | 2.23 | 2.64 | 3.07 | 3.53 | 3.87 | 3.95 |
| G_n, both forms | 0.79 | 0.69 | 1.09 | 1.00 | 1.75 | 1.55 | 1.47 | 1.70 | 2.13 | 2.39 | 2.85 | 2.84 | 3.26 | 3.29 |
| r_n, one order | 8.97 | 3.44 | 6.13 | 5.00 | 5.32 | 9.05 | 8.27 | 7.66 | 7.78 | 7.29 | 6.30 | 7.62 | 8.49 | 8.38 |
| r_n, all orders | 8.97 | 3.44 | 6.13 | 5.00 | 5.24 | 7.01 | 7.07 | 7.05 | 6.28 | 6.44 | 4.99 | 6.38 | 4.76 | 5.65 |

r_n is worst at x = 12 to 15, a shift of one to eight steps, where the pole sum and
the difference of the two monomials are of one size: the digit the double
`polygamma_monomial_remainder` also loses there. Test bounds
(`tests/math/test_polygamma_dd.cpp`): the worst of the two forms plus half, per order.
On the same grid the double `polygamma(n, x)`, n <= 6, is at 2.86 ulp1 (n = 0) and
1.67, 1.95, 1.82, 3.34, 4.09, 5.03 ulp.

What the difference form buys: G_n formed as polygamma_dd(n, x + h) - polygamma_dd(n, x),
worst over all orders, by band of log10(x/h): below 1.5, 61 units (exact form 2.9);
1.5-3.5, 1540 (2.1); 3.5-5.5, 2.7e5 (2.8); 5.5-7.5, 6.4e7 (2.0); 7.5-9.5, 8.7e9 (2.2);
9.5-12.5, 3.4e12 (0.95); 12.5-17, 6.5e15 (3.3). The subtraction loses about x/h units;
at x/h = 1e7, the lognormal regime d/p = 1e7 with p = 1, that is 7e-25 relative
against 1e-31.

Speed, ns per evaluation (h = 0.37; all orders = one call for N = 13):

| | 1 thread [1e-2, 1e8] | 1 thread [20, 1e8] | OpenMP 16 [1e-2, 1e8] | OpenMP 16 [20, 1e8] | CUDA [1e-2, 1e8] | CUDA [20, 1e8] |
|---|---|---|---|---|---|---|
| psi_0 | 868 | 650 | 100 | 57 | 40-90 | 8.0 |
| psi_1 | 403 | 99 | | | 8.0 | 1.9 |
| psi_6 | 485 | 119 | 93 | 17 | 8.9 | 2.5 |
| psi_13 | 560 | 134 | 106 | 21 | 10.1 | 3.0 |
| G_0 | 957 | 414 | 159 | 57 | 18.3 | 7.2 |
| G_6 | 1249 | 305 | 233 | 43 | 24.2 | 6.0 |
| G_13 | 1936 | 407 | 388 | 58 | 37.0 | 8.4 |
| r_0 | 484 | 92 | 100 | 13 | 10.4 | 1.7 |
| r_13 | 612 | 124 | 127 | 21 | 11.2 | 2.9 |
| psi, all orders | 3072 | 1664 | 489 | 236 | 50.3 | 25.8 |
| G, all orders | 3894 | 1543 | 632 | 235 | 63.5 | 24.1 |
| r, all orders | 2667 | 917 | | | 45.9 | 18.6 |
| double polygamma(6) | 24 | 19 | 4.4 | 3.1 | 0.5 | 0.1 |

(OpenMP from the prototype, which computes the same operations.) psi_0 is dominated
by the double-double logarithm (about 600 ns on one thread); on CUDA it is the first
kernel timed and varies from 40 to 90 ns between runs. Fourteen single calls against
one all-orders call, one thread: psi_0..psi_13 7807 against 3193 ns over [1e-2, 1e8]
and 2584 against 1947 over [20, 1e8]; G_0..G_13 20991 against 4252 and 5587 against
1429. This is the measurement behind the exception to one function per order.

Table placement. With the table in `__constant__` (`LAPLACE_DEVICE_TABLE`) the -rdc
device program of laplace_tests needed 0x11918 = 71960 bytes of constant data against
the 64 KB bank, and nvlink failed for every build (measured by another session on
2026-11-16). In global memory the CUDA cost is the same to 0.1 ns on every function
above (both placements built from the same bench, two runs each), and laplace_tests
links, with 63896 bytes of the 65536 in the constant bank (.nv.constant3 of the
linked program, cuobjdump -elf): 1640 bytes left. Decided with Giovanni: tables of several kilobytes go to global memory.

Open: values below about 1e-292 in magnitude lose double-double precision (their low
part is subnormal; order 13 only, x > 1e23) and values above about 1e300 may be NaN on
the host (Dekker's split overflows); both are declared, not handled.

Update (2026-11-16, 09:50): after `factor_analytic_pdmat::inverse_dk` moved to the
augmented matrix [[Psi, Lambda], [Lambda', -I]], mvgaussian1 over factor_analytic
measures, worst ulp1N over A = 0..2 and K <= 6, 118 on gcc and 78.9 on CUDA (A = 0,
K = 5, psi = e^-6; was 3.1e6), and the information (trace route) 33.1 / 12.1 (was
1.6e6); mvgaussian2 over factor_analytic at most 1.73 everywhere, its information
included (was 1.2e5). Bounds in `tests/distrib/test_mvgaussian.cpp` reset to the
larger of the two times 1.5. The log_cholesky A = 0, K = 6 block is back at 4.58 /
5.25 (bound 8), so the open note above is closed.
Accepted with the current bounds (Giovanni, 2026-11-16): mvgaussian1 over
factor_analytic at 118 ulp1N, about 10 times what the conditioning (10.5) explains,
and the information of mvgaussian2 over AR(1) at 129 ulp1N (order 4, near rho = 1,
trace route).

## Device tables in global memory: incomplete gamma and beta derivatives, Bessel I

What: on 2026-11-16 the device tables of `math/incomplete_gamma_deriv.hpp` (nine
tables, 6688 bytes: the quadrature nodes and weights `d_ee1_*`, `d_ee2_*`, `d_ts_*`,
and the series coefficients `d_lgam_c`, `d_eta_d`, `d_temme_c`) and of
`math/incomplete_beta_deriv.hpp` (`map_dev` 5024 and `gl_dev` 1664, 6688 bytes) moved from
`LAPLACE_DEVICE_TABLE` (`__constant__`) to `LAPLACE_DEVICE_GLOBAL_TABLE`, so that the
64 KB constant bank shared by the whole -rdc device program has room (decided with
Giovanni; the count is kept in `note/cuda_constant_budget.md`).

How: one scratch bench of the public derivative functions, built once with the
tables in each place, two runs each, on CUDA (RTX 4070 SUPER), ns per call:

| function, region | global | constant |
|---|---|---|
| log P da1, all | 127.0 | 127.0 |
| log Q da5, all | 178.9 | 178.9 |
| log P da6, centre 1-50 | 139.7 | 140.0 |
| log Q da5, a < 1 | 68.3 | 68.6 |
| log P da5, upper tail | 102.5 | 102.4 |
| log I d5, all shapes | 436.4 | 436.3 |
| log I d6, small shapes | 781.6 | 781.6 |
| log I d5, large shapes | 141.8 | 141.7 |

Every function and region timed (gamma orders 1, 5, 6 of P and Q in six regions,
beta orders 1, 5, 6 of I and 1 - I in three) agrees within 0.3 per cent, including
the quadratures, whose nodes every thread of a warp reads at the same time; the
only outlier, 139 against 127 ns, was the first kernel of the first run and did not
repeat. The values are unchanged (same arithmetic). After the move the linked
laplace_tests needs 50520 bytes of constant data (.nv.constant3, cuobjdump -elf),
15016 below the ceiling.

The same day the four tables of `math/bessel_i.hpp` followed (`ratio_series_dev` 26,
`ratio_asymptotic_dev` 60, `inverse_asymptotic_dev` 270 and `inverse_series_dev` 144
doubles, 4000 bytes), measured the same way (the bench built twice, the second
run of each taken, the first having run on a cold card), ns per call on CUDA:

| function, region | global | constant |
|---|---|---|
| A, k < 1 | 1.36 | 1.36 |
| A d6, k in [16, 4096] | 23.49 | 23.48 |
| log I0, k in [1, 16] | 0.94 | 0.95 |
| A^-1 d1, k in [16, 4096] | 15.99 | 16.15 |
| A^-1 d4, k < 1 | 10.00 | 10.02 |
| A^-1, rho < 0.5 | 19.50 | 19.48 |
| A^-1, rho in [0.5, 0.95] | 25.62 | 25.49 |
| A^-1, rho > 0.95 | 33.12 | 34.01 |

Every function and region timed (A at orders 0, 3, 6, log I0, A^-1 at orders 1
and 4 in four ranges of k, A^-1 in three ranges of rho) agrees within 1 per cent,
except A^-1 for rho > 0.95, where the global table is 2.6 per cent faster and the
two runs of the constant one differ by 1.6 per cent. test_bessel passes on gcc
and CUDA (56 of 56 assertions). After the move .nv.constant3 is 46520 bytes,
19016 below the ceiling.

## kappa from the circular standard deviation s, with its six derivatives (2026-11-17)

The von Mises family of Laplace is parametrized in the mean direction mu and the
circular standard deviation s = sqrt(-2 log rho), where rho = A(kappa) = I_1(kappa)/I_0(kappa)
is the mean resultant length. Every per-observation quantity is written in kappa,
so the family needs kappa(s) and its derivatives to order six
(`include/laplace/distrib/vonmises_scale.hpp`). The map has no closed form:
it is the inverse of s(kappa) = sqrt(-2 log A(kappa)).

Three routes, chosen by s, because each of the other two loses the quantity it
needs where the third keeps it:

| region | route | why |
|---|---|---|
| s <= 0.05 | series in v = s^2/2 | kappa diverges like 1/v, and 1 - rho = -expm1(-v) is the only form that keeps v when rho is within an ulp of one |
| 0.05 < s < 4 | the inverse of A, read in rho or in its complement | the general case; the derivatives follow by Faa di Bruno over exp(-s^2/2) |
| s >= 4 | series in rho | kappa is about 2 rho, and rho = exp(-s^2/2) underflows gracefully while A's inverse would be asked for a value it cannot resolve |

The series coefficients come from `tools/reference/vonmises_kappa_series.py`
(SymPy, reverting the Taylor series of A and composing with the exact series of
exp(-s^2/2)): 8 terms below, 7 above, the first of each being 1/(2v) and 2 rho.
The derivatives of exp(-a s^2/2) are generated by the recurrence
d[n+1] = -a (s d[n] + n d[n-1]), exact in double and free of the cancellation
that a transcribed polynomial of order six would carry.

The middle region needed a new entry point in `math/bessel_i.hpp`. For large
kappa, A is within an ulp of one, so the quantity that determines kappa is the
complement 1 - A, not A: a complement formed as `1.0 - rho` from a rounded rho
has already lost half its digits. `ratio_inv` was refactored into
`ratio_inv_rc(r, c, p)`, which takes rho and its complement as a pair, and two
entry points were published: `bessel_i_ratio_inv(rho)` as before and
`bessel_i_ratio_inv_c(q)`, which reads the complement. The map calls the second
whenever rho > 0.5, where `-expm1(-s^2/2)` gives the complement exactly. At
q = 1e-17 the rho route returns infinity (rho rounds to one) while the
complement route returns kappa = 5e16 to eight digits
(`tests/distrib/test_vonmises_scale.cpp`).

Accuracy over the reference grid, s from 1e-8 to 4.5, gcc/OpenMP, worst ulp of
the order's own value (`benchmarks/distrib_vonmises_scale.cpp`):

| order | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| own (the three routes) | 5.32 | 6.12 | 9.55 | 6.79 | 20.4 | 25.7 | 25.3 |
| the inverse alone | 5.32 | 6.12 | 9.55 | 8.19 | 20.4 | 26.7 | 46.6 |
| the series in v alone | 3.5e30 | 2.7e30 | 1.9e30 | 1.3e30 | 7.9e29 | 4.4e29 | 4.4e29 |
| the series in rho alone | 8.8e15 | 8.8e15 | 8.9e15 | 8.7e15 | 9.1e15 | 9.0e15 | 8.8e15 |

The last two rows are each route evaluated over the whole grid, which is what
makes the switch points measured rather than chosen: outside its own region each
series is wrong by thirty or sixteen orders of magnitude. The worst point of
every order but the value is s = 0.2, the low end of the inverse's band; the
value's is s = 4.1, just below the switch to the series in rho. Bounds in the
test are the worst of the two builds times about one and a half:
{8, 10, 15, 20, 32, 40, 50}.

Above s = 4.5 there is no reference file and should be none: kappa is about
2 exp(-s^2/2), so d kappa / kappa = -s^2 (ds / s) and one eps of s moves kappa by
s^2 eps whatever computes it. At s = 17.8 that is 178 ulp, which is the
conditioning of the map and not an error of the code; the test checks the
asymptote there instead.

Speed, ns per call, gcc (Serial, then OpenMP on 16 threads): the value 244 / 17,
order six 1290 / 97, and the joint routine `vonmises_kappa_from_s_upto<6>`, which
returns all seven, 1302 / 99 against 4854 / 367 for the six separate calls. The
joint routine is 3.7 times cheaper and gives bit for bit what the per-order
functions give (checked against the same references), so it is an addition to the
per-order functions and not a replacement: the rule of one function per
derivative order stands.

How the references were made, and a lesson. `tools/reference/vonmises_scale.py`
builds the derivatives from the Riccati identity A' = 1 - A/kappa - A^2 and the
chain rule, at mpmath precision. Two mistakes were made and both were caught by
cross-checking, which is the only reason they are in this entry and not in the
library:

- The constant 1 was dropped from the Riccati identity, which flips the sign of
  every derivative. Caught against `mpmath.diff` of the map itself, an
  independent route, which then agreed to 1e-79.
- At s = 1e-8 the order-six derivative came out as 1.34e68 against the true
  5.04e67. The identity for A^(j) needs about (j + 1) log2(kappa) bits and had
  166 of the 371 required. The reference script now sizes its precision as
  (n + 1) bits + 80 and evaluates every row at two precisions, keeping only the
  rows that agree to 2^-140. The library's disagreement of 3.5e15 ulp is what
  exposed it, and the exact leading Laurent term settled which of the two was
  wrong.

A reference is not a reference until two routes that share no arithmetic agree on
it.

## The polygamma remainder psi^(n)(x) - mu_n(x) (2026-11-17)

`own::polygamma_monomial_remainder(n, x)` in `include/laplace/math/polygamma.hpp`
returns psi^(n)(x) minus its leading term mu_n(x), with mu_0(x) = log x and
mu_n(x) = (-1)^(n+1) (n - 1)! x^-n, for n from 0 to 6.

Why it exists. Every expected information of a family parametrized in a
precision has the shape of a weighted sum of polygammas whose leading terms
cancel exactly. The beta in the mean mu and the precision phi has

    I_phi,phi = mu^2 psi_1(alpha) + nu^2 psi_1(beta) - psi_1(phi),

with alpha = mu phi, beta = nu phi, nu = 1 - mu, and the leading terms cancel
because mu^2/alpha + nu^2/beta = 1/phi exactly. The true value is of size
1/(2 phi^2) while the terms are of size 1/phi, so the straight form keeps about
eps phi of relative error: measured over the reference grid of the beta, the pure
phi block of the order-K derivative loses 182 ulp of itself at phi = 1e2,
1.7e4 at 1e4, 3.3e6 at 1e6 and 3.9e8 at 1e8, which is eps phi to within a factor
of four at every band. Carrying only the remainders, with the bracket of
monomials summed algebraically (that is, not summed at all, since it is zero),
the same quantity is accurate to between one and seven ulp.

How it is computed, with no subtraction anywhere. For x >= 15 the asymptotic
expansion is evaluated with its first term simply left out, so the leading term
is never formed and never cancelled. Below 15 the recurrence shifts x up by whole
steps, and the monomials of the two arguments are differenced in closed form
(for n = 0 as log1p(m/x), m being the shift). The 2^600 scaling guard of
`polygamma` is kept.

This is not what `own::detail_ibd::dpsi_all` does. Regrouping the three
polygammas into differences leaves ab/s^3, ab/s^3 and -2 ab/s^3, so the
cancellation only moves one level down; the monomials have to leave the sum
entirely.

Accuracy against `tools/reference/psi_remainder.py` (mpmath at 200 bits, 75 rows
per order, x from 1e-3 to 1e12, no row dropped), gcc, worst ulp per order:
12, 12, 14.8, 12, 16, 20, 24 as the bounds in
`tests/math/test_psi_remainder.cpp`, where order two's 14.8 sits at x = 13, just
below the recurrence threshold 15: the shift is then by one, the two monomials
z^-n and x^-n are of one size, and their difference costs the digit the header
predicts. The CUDA build passes the same bounds. The references subtract at a
precision where the subtraction costs nothing, while the library never performs
it at all, so the two routes share no arithmetic.

Closed (point 4a of `note/piano_distrib_restanti.md`), the same day: the beta is
wired onto it, which is the section "The beta's pure phi components from the
remainders" below.

## weibull, beta and vonmises: the three remaining continuous families (2026-11-17)

Three families of phase 4, each with the full interface (value and six orders in
theta, the mixed derivatives in (y, theta), the expected information with four
orders, cdf, log cdf with six orders, quantile, draw, start values). The
gengamma was left out by Giovanni's decision of the same day.

### weibull (mean m, shape sigma)

The parametrization of statmodels7's weibull3: the mean is the first parameter,
so the scale is m / Gamma(1 + 1/sigma) and every derivative in sigma carries the
polygammas of that Gamma. Accuracy over the reference grid, gcc, worst scaled ulp
per order (the error of a component divided by the ulp of the largest component
of the same order):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| log f | 45.3 | 7.33 | 8.01 | 12.1 | 9.78 | 8.79 | 8.09 |
| log F | 2.52 | 4.10 | 13.0 | 9.75 | 110 | 402 | 479 |
| log(1 - F) | 9.35 | 6.80 | 8.03 | 12.1 | 9.78 | 8.78 | 8.27 |
| in y | - | 2.53 | 3.66 | 4.81 | 107 | 137 | 71.1 |

The expected information is at 1.8e-16, 1.6e-16 and 3.0e-16 of the natural scale
for the three blocks. log F is the hard route, as it is for every family on the
half line: its high orders carry the derivatives of log(-expm1(-u)) with u the
transformed argument, and the worst points are where u is of order 1e-3 and the
logarithm's own cancellation is the error.

Speed, gcc, ns per call Serial / OpenMP at (m, sigma) = (2, 1.5): log f 39 / 75,
order six 360 / 199, the (y, theta) block of total order five 341 / 187, cdf with
both logarithms 176 / 88, log F order six 788 / 351, quantile 85 / 83, the
information 7 / 62 and its order four 138 / 137. At sigma = 0.4 the density costs
three times more (123 / 110 at order zero), which is the cost of the polygammas
of 1 + 1/sigma.

Two defects of the test, not of the family, found by its own twins and worth
recording because both printed numbers that looked like bugs in the code. The
twin against the exponential divided by a reference value that is exactly zero
and printed no label, so the first failure named neither the quantity nor the
point. And the closed form checked in the underflow case (y of order 1e-28,
sigma = 0.01) was missing the factor (-1)^K: only the odd orders failed, with
magnitudes agreeing to six digits, which is what finally identified it.

### beta (mean mu, precision phi)

statmodels7's beta1, with alpha = mu phi and beta = (1 - mu) phi. Accuracy, gcc:

- orders zero and one: 1.38 eps of the sum of the terms.
- orders two to six: 2.14, 4.37, 5.72, 12.2 and 9.61 ulp of the order's largest
  component.
- expected information: 5.66 eps of the largest component; its pure phi block
  4.89 eps of (phi + 1) of itself.
- cdf: 6.2e-14 on F and 6.6e-14 on 1 - F relative, 1.9e-15 on log F and 1.4e-15
  on log(1 - F) absolute over max(1, |log|).
- the six orders of log F: 4.3e-14, 4.7e-14, 1.2e-13, 1.1e-12, 2.4e-11, 2.8e-10
  of the order's largest component, and of log(1 - F) 2.9e-15, 1.8e-14, 8.3e-14,
  1.3e-12, 2.8e-11, 1.8e-10. The two tails are of one size, which they were not
  before the side-selecting derivatives of the incomplete beta landed.

The pure phi components were the exception until point 4a was closed: they are
now built from the polygamma remainders and stay within 13.4 ulp of themselves,
and the pure phi block of the information within 7.68 eps of itself, at every phi
measured. The figures of the direct assembly and the new ones are in the section
"The beta's pure phi components from the remainders" below.

Speed at (mu, phi) = (0.3, 6), ns per call Serial / OpenMP: log f 241 / 109,
order six 221 / 131, the information 13 / 74 and its order four 343 / 151, log F
through the incomplete beta 259 / 130, its closed derivatives 5556 / 735 at order
one and 23301 / 2082 at order six. The generic quadrature route for the same
order two costs 85281 / 27844, so the closed derivatives are 14 times faster and
are what the family uses; the quadrature stays as the test's independent route.

### vonmises (mean direction mu, circular standard deviation s)

A new parametrization, not one of statmodels7's: see the map above. The density is
written so that nothing cancels, the tensor separates as W_a(z) kappa^(b)(s) with
z = y - mu (the component with a derivatives in mu is a function of z times a
derivative of the map), and the information is diagonal, with I_mu,mu = kappa A
and I_ss from the map alone.

The measure is not the per-component relative ulp. Many components of the tensor
vanish identically by symmetry, at y - mu = pi/2 and at the ends of the grid, so
dividing by a component that is exactly zero reports 1e11 or 1e16 ulp for a value
that is correct. `benchmarks/distrib_vonmises.cpp` prints those columns and they
are not the bound; the test measures the error of a component against the ulp of
the largest component of the same order plus a stencil floor, which is the
honest measure. On that measure, gcc:

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| log f, s <= 4.5 | 4.70 | 5.01 | 4.99 | 5.10 | 4.87 | 4.90 | 5.02 |
| log f, s > 4.5 | 0.10 | 0.53 | 0.49 | 0.53 | 0.55 | 0.61 | 0.81 |
| mixed (y, theta), s <= 4.5 | - | 0.52 | 0.70 | 1.15 | 1.30 | 2.50 | 3.33 |
| mixed (y, theta), s > 4.5 | - | 0.53 | 0.49 | 0.53 | 0.55 | 0.57 | 0.60 |

The CUDA build measures 4.70 to 5.16 and 0.10 to 0.81 on the same grid, identical
to gcc within a hundredth of an ulp, so one set of bounds holds for both builds.

The cdf is measured from a fixed -pi (Giovanni's choice of 2026-11-17: the
alternative, measuring from the mode, would make F depend on mu through its own
lower limit). What that convention costs is rotation invariance: F(y; mu) is
not F(y - c; mu - c), and the difference measures 4.0e-5 at mu = 0 and 1.5e-4 at
mu = 1 for a rotation of c = 0.4 (mpmath at 200 bits). It is not a defect of the
implementation but a property of measuring the arc from a fixed point, so a user
who rotates the data rotates the censoring bounds with them. The test of the
identity bounds the non-invariance from below at 1e-5, a threshold set from these
measurements (it had been 1e-3, a conjecture that the measured 2.25e-4 failed). It is computed as the mass of the arc from -pi to y, by a series in
Bessel ratios where that converges and by panelled quadrature otherwise, and the
worst error over its own tolerance is 0.24 for the series and 0.18 for the
integral.

The arc mass in closed form. The quadrature route was wrong in the fourth digit
at s = 1e-4 on CUDA (0.841804 against 0.841345). Two causes: the panels were
laid out by s rather than by the decay length of the integrand, which for large
kappa is 1/(kappa |sin((c - mu)/2)|) and not s; and the route never read
`r.converged`. The panels are now laid out by the decay length, and where the
quadrature does not converge the mass is computed in closed form through
w = 2 sin((t - mu)/2), with `log_Phi` for both tails and `log1mexp` for their
difference, g being evaluated at the dominating endpoint moved in by one decay
length. The route is selected by `!r.converged` and not by a threshold in kappa,
so no band of the parameter is assumed to be the hard one. Decided with Giovanni
on 2026-11-17 after two weaker guards had been measured and rejected: a guard on
the bare flag threw away correct values, because the integrand underflows to
exact zero on the far panels where no relative stopping test can be met, and a
guard on `r.error` did not help, because the quadrature genuinely does not
converge there.

Open: the crossover of the closed form, w_0 sqrt(kappa) of order one, where
neither the Gaussian tail nor the series in w is the dominant behaviour. There F
is accurate to 2.9e-10 and log F to 1.8e-9 against the 1e-15 of everywhere else.
The test delimits that band narrowly (|2 sin((y - mu)/2)| sqrt(kappa) between 0.2
and 5) instead of widening its global tolerance, so the rest keeps its 1e-15.
Closing it needs a second order in the expansion of g, which is work and not a
repair.

Speed, gcc, ns per call Serial / OpenMP: the tensor 622 / 47 at order zero and
2891 / 204 at order six, of which the map alone is 2746 / 200, so above order
three the family's cost is the map's; the information 821 / 67 and its order four
2276 / 186; the cdf 29613 / 1643 and log F the same; a draw by inversion
72074 / 5851. The generic derivatives of log F, which the family does not use and
the test compares against, cost 4.1e6 / 5.9e5 at order six.

## The exponential's log F where 1 - F is within an ulp of one (2026-11-17)

Found while the three families above were being tested, in a family committed by
another session: `exponential_distrib::log_cdf` returned exactly 0 for u = y/mu
above about 36. It computed log F as log(-expm1(-u)), which is correct only where
exp(-u) is resolvable against one; above 36 the argument of the logarithm rounds
to one and the result to zero, while the true log F is about -exp(-u), that is
-2e-16 and smaller. Repaired by splitting at a threshold and using
log1p(-exp(-u)) above it, which keeps the small quantity:

    if (u >= u_split) return log1p(-exp(-u));
    return log(-expm1(-u));

The lesson, agreed with the session that wrote the family, is not the line but
the gap it fell through: the family's own test never evaluated log F at a point
where 1 - F is within an ulp of one. A test grid that stops at moderate
quantiles cannot see this class of defect, and every family on the half line has
the same two-form structure in its tails.

## The beta's pure phi components from the remainders (2026-11-17)

Point 4a of `note/piano_distrib_restanti.md`, closed the day the remainder
landed: `polygamma_monomial_remainder` existed, measured, and no family called
it. The beta is the family that motivated it, so the beta is what was wired onto
it.

**What.** The pure phi component of order n + 1 of the log-density, and with it
the pure phi block of the expected information, is

    mu^(n+1) psi_n(alpha) + nu^(n+1) psi_n(beta) - psi_n(phi),   alpha = mu phi,
                                                                 beta = nu phi.

The three leading monomials cancel exactly: mu^(n+1) mu_n(mu phi) is
(-1)^(n+1) (n - 1)! mu phi^-n, the nu term is the same with nu, and their sum is
mu_n(phi) (mu + nu) = mu_n(phi), which is the third term. The remainder of the
combination is of size phi^-(n+1), so computing it from the polygamma functions
themselves forms a result of that size as a difference of terms of size phi^-n,
and about log10(phi) digits go. Built from the remainders r_n = psi_n - mu_n
instead (`detail_beta::pure_phi_combination`), nothing is formed by subtraction
and the monomials cost nothing, because they are dropped algebraically rather
than cancelled in floating point. The identity used is mu + nu = 1 with nu the
stored 1 - mu, the same value that defines alpha and beta, so no new
approximation enters.

**How much.** Pure phi components of `logpdf_dk`, relative to themselves, orders
two to six, gcc and CUDA identical:

| K | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| from the remainders, ulp | 11 | 13.4 | 8.2 | 11 | 10.3 |

The direct assembly gave 1.6e8 to 3.9e8 ulp over the same orders at phi = 1e8,
and 63 to 230 at phi <= 1e2, which is the eps phi the cancellation predicts.

The pure phi block of the expected information, relative to itself: 4.89 eps of
(phi + 1), that is 4.9e8 eps at phi = 1e8, before; 7.68 eps flat after, the worst
row now at K = 0, mu = 0.5, phi = 3. The scaled errors of the other components
are unchanged within the noise of the measurement (orders two to six 2.18, 5.09,
4.06, 12.2, 9.61 ulp of the order's largest component against 2.14, 4.37, 5.72,
12.2, 9.61 before; the information 5.66 eps of the largest component, the same).

**What it says.** The loss was never in the polygamma functions and never in the
references: it was in the subtraction, and it is gone once the quantity is never
subtracted. The normalization of the test's bound is the measurement that proves
it. The bound used to be 40 eps (phi + 1) + 40 eps of the value, and under it the
worst row sat at phi = 1e8; it is now 16 eps flat, and the worst row sits at
phi = 3, where the old divisor (phi + 1) was 4 and hid nothing. A bound that
carries a factor of the parameter is a bound that has given up on a corner, so
dropping the factor is part of the repair and not a cosmetic change. Bounds:
24 ulp for the pure phi components of the log-density (measured 13.4) and 16 eps
for the information's block (measured 7.68), both passed by gcc and CUDA.

**What is not done.** The mixed components still lose the pole terms
(-1)^(m+1) m!/alpha^(m+1) when alpha or beta is small, and the mixed components at
large phi still cancel nearly, not exactly. Neither is rearranged: that needs the
same treatment of the pole terms, and no family has asked for it yet.

## log K_nu at half-integer orders by the closed form

What: on 2026-11-16 `math/bessel_k.hpp` gained a closed form for nu = n + 1/2,
0 <= n <= 19, used by `log_bessel_k`, `log_bessel_k_dx1` ... `_dx7`,
`log_bessel_k_dx_upto` and the scaled pair. At these orders the asymptotic
series in 1/x terminates and is exact (DLMF 10.47.9, 10.49.1, 10.49.12):

    K_(n+1/2)(x) = sqrt(pi/(2x)) e^-x S(x),  S = sum_(k=0..n) a_k x^-k,
    a_k = (n + k)! / ((n - k)! k! 2^k).

Before, these orders went through Temme's series (x <= 2) at the reduced order
mu = -1/2, whose first coefficient cancels near x = 1, and the Riccati identity.
At nu = 1/2 that cost 26 ulp at orders 2 and 3; nu = +-1/2 is the order the
Poisson-inverse Gaussian reaches at y = 0 and y = 1.

How. Every a_k is positive and x^m S^(m) = (-1)^m sum_k a_k (k)_m x^-k, so the
sums themselves do not cancel. The step that turns them into the derivatives of
log S (L^(m) = (S^(m) - sum_k C(m-1, k-1) L^(k) S^(m-k)) / S) does. In double the
closed form was measured at 304 ulp at order 2 (n = 38, x = 0.05), 1.3e4 at
order 3 and 7.3e8 at order 7 (n = 40, x = 1.55), and already 94 ulp at order 7
for n <= 3. So the terms, the sums and that recursion are in double-double; the value needs
only the sum and stays in double. The terms are scaled by the term of index 0
for x >= 1 and by that of index n below, so that every scaled term lies in
(0, 1] and nothing overflows before the result does. The cost is O(n) per call,
and the cap n <= 19 keeps the closed form below nu_debye = 20, where it replaces
only the Temme/Steed and Riccati route.

References: `tools/reference/bessel_k_half.py` writes
`tests/data/reference/log_bessel_k_half*.csv`, 1528 rows at n = 0..40, 60 per
cent of them with x in [0.3, 3] and the rest log-uniform on [2^-14, 2^12], from
mpmath's besselk and diffs at 320 and 448 bits (no row dropped at 2^-110); the
closed form is not used. On six points they agree with the Riccati identity at
400 bits to 1e-97 or better.

Accuracy on the half-integer grid, ulp (order 0 in ulpC), gcc and CUDA alike
except where two figures are given (gcc / CUDA):

| order | before, n <= 19 | after, n <= 19 | n >= 20 (Debye, unchanged) |
|---|---|---|---|
| 0 | 4.13 | 1.70 | 2.79 and 3.43 (the two builds) |
| 1 | 6.76 / 6.21 | 1.07 | 3.00 |
| 2 | 26.19 (13.2 ulp1) | 1.22 | 3.63 |
| 3 | 26.34 | 1.58 | 4.40 |
| 4 | 13.05 / 13.15 | 1.55 | 8.54 / 6.71 |
| 5 | 6.5 | 2.24 | 7.92 / 6.66 |
| 6 | 8.8 | 1.83 | 10.56 / 10.25 |
| 7 | 11.9 | 2.06 | 76.66 |

Before, the worst cases of orders 1 to 4 were all at nu = 1/2, x = 0.64 to 0.9;
those of orders 6 and 7 at n = 10 to 19.
The full reference grid of `log_bessel_k*.csv` does not move: its worst cases,
44.4 ulp (gcc) and 37.4 (CUDA) at order 2, are at nu = 1/4, x = 0.98, which is
not a half-integer, so the bounds of the general test stay as they were. The
half-integer grid has a test of its own, with bounds per band (n <= 19 and
n >= 20). The joint routine agrees with the single-order functions here as
elsewhere (test "log_bessel_k_dx_upto against the single-order functions, every
regime and boundary", whose grid holds half-integer orders).

Cost, ns per call, x log-uniform on (1/4, 4), gcc on one thread / CUDA (RTX
4070 SUPER):

| n | function | before | after |
|---|---|---|---|
| 0 | value | 164 / 6.6 | 12.9 / 1.1 |
| 0 | order 1 | 899 / 74 | 70 / 4.1 |
| 0 | order 4 | 1018 / 58 | 360 / 9.6 |
| 0 | order 7 | 1295 / 63 | 731 / 18 |
| 19 | value | 236 / 6.8 | 80 / 4.0 |
| 19 | order 1 | 2265 / 196 | 1127 / 17 |
| 19 | order 4 | 2481 / 227 | 1875 / 91 |
| 19 | order 7 | 2782 / 171 | 2614 / 41 |

The closed form in double-double is cheaper than the route it replaces at every
n <= 19 on both builds. Above, it would stay as accurate (2.24 ulp at most at
n = 20..40, every order) but costs more than the Debye route it would replace:
order 7 at 3796 against 1241 ns on gcc and 188 against 42 ns on CUDA, order 1 at
1726 against 158 ns and 33 against 9.6 ns. Debye there reaches 76.66 ulp at order 7
(nu = 20.5, x = 21.7) and at most 10.6 at the lower orders. Whether to extend
the cap to trade that cost for that accuracy was put to Giovanni.

Chosen: n_max stays at 19 (Giovanni, 2026-11-16). For n = 20 to 40 the Debye
route reaches up to 76.7 ulp at order 7, which the closed form would bring down
to 2.24, but raising the cap would cost about 3 times as much on gcc (3796
against 1241 ns at order 7) and 4.5 times on CUDA (188 against 42 ns).

No device table was added (`.nv.constant3` unchanged at 46520 bytes).

## The rank-deficient multivariate Gaussian: degenerate mvgaussian1, improper mvgaussian2 (2026-11-16)

What: `mvgaussian1_distrib<S>` and `mvgaussian2_distrib<S>` over a
positive semidefinite structure whose null space is constant in theta and
supplied (`constant_null_space_psdmat`; today `weighted_sum_psdmat`), as
partial specializations of the same templates
(`include/laplace/distrib/mvgaussian1.hpp`, `mvgaussian2.hpp`, shared code
in `distrib/detail/mvgaussian_rank_deficient.hpp`). Host only, as the
structure. `reduced_rank_psdmat`, whose null space moves with theta, is
excluded from both. Test `tests/distrib/test_mvgaussian_rank_deficient.cpp`,
references `tools/reference/distrib_mvgaussian_rank_deficient.py` (5215
values), bench `benchmarks/distrib_mvgaussian_rank_deficient.cpp`.

How. rho the rank, U0 the orthonormal null-space basis, P = U0 U0', r = y - mu:

    l1 = -rho/2 log 2pi - 1/2 log|Sigma|_+ - 1/2 r' Sigma^+ r   (degenerate)
    l2 = -rho/2 log 2pi + 1/2 log|Q|_+     - 1/2 r' Q r         (improper)

- log|M|_+ and its derivatives from the structure (`log_pdet_state` once per
  theta, then `log_pdet_dk_at` for every tuple).
- Sigma^+ = (Sigma + cP)^-1 - P/c with c = tr(Sigma)/rho fixed at the theta of
  the state; any fixed c > 0 gives the identity for every theta, so
  d_T Sigma^+ = d_T (Sigma + cP)^-1, by the recursion
  A[Q] = -A sum_(0 != T subset Q) M[T] A[Q \ T] over the masks of the tuple.
- Off the support (degenerate): -inf when ||U0' r|| > 64 d eps ||r||. The
  derivative blocks are those of the projected density.
- Blocks: the code of the proper classes through an adapter whose "inverse"
  is M^+ and whose "log det" is log|M|_+.
- Expected information, both classes, for structures with
  d_ab M = delta_ab d_a M (weighted_sum):
  I_ab = 1/2 tr(B_a B_b) = -1/2 d_ab log|M|_+ + 1/2 delta_ab d_a log|M|_+,
  derivatives from log_pdet_dk. B_a is the structure's B_a, similar to
  e^(theta_a) M^+ S_a. The trace route through M^+ (the one of the proper
  classes) cancelled: 1.8e17 ulp1N (degenerate) and 9.5e11 (improper) at the
  points with theta components 9 apart; it is not offered.
- Draws: M + cP = L L'; degenerate y = mu + (I - P) L z, covariance
  (I - P)(Sigma + cP)(I - P) = Sigma; improper, the constrained law
  U0' r = 0: `constrained_state` adds the factor, y = mu + (I - P) L^-T z,
  covariance Q^+ (`constrained_covariance`). The improper `theta_state`
  computes no factor.
- start_value: degenerate (I - P) C (I - P), C the sample covariance, no
  eigenvalue floor; improper its pseudo-inverse; then `params_from_matrix`.

Accuracy against mpmath at 60 digits (the definition: the eigendecomposition
gives log|M|_+ and M^+; mpmath.diff gives the blocks; the information from
F = E[-l]). Four structures: d = 4 with D2'D2 (rank 2); d = 5 with D1'D1,
D2'D2 (rank 4); d = 6 with D1'D1, D2'D2, D3'D3 (rank 5); d = 5 with D2'D2,
D2' diag(1, 2, 3) D2 (rank 3). At each one, a random point and a spread one
(theta components 9 apart). Worst ulp1N; gcc 14.2 and the host code of the
CUDA build (MSVC) give the same numbers.

| | blocks, worst of A = 0, 1, 2 | information |
|---|---|---|
| improper, every point | 4 (A = 0), 1 (A = 1, 2) | 1.12 |
| degenerate, random points | 117 | 1.12 |
| degenerate, spread, d = 5 two penalties | 4.86e6 | 2.4e-4 |
| degenerate, spread, d = 6 three penalties | 3.28e12 (A = 0: 1.15e11, A = 1: 7.15e11) | 0.25 |

What it says: the dense (Sigma + cP)^-1 loses in proportion to the condition
number of Sigma on its range (measured with mpmath: 2.07e5 at the d = 5
spread point, 6.53e9 at the d = 6 one), here 24 and 500 times cond eps:
about 7e-4 relative at the worst point. The improper
blocks never form an inverse and keep 1-4 ulp. Accepted by Giovanni on
2026-11-16. The upgrade path is the structure's graded factorization
(Sigma = Q D^1/2 H D^1/2 Q' on the range), when a model needs a degenerate
covariance with spread lambda. Test bounds: 1.5 times the worst, at least 4,
and per spread point for the degenerate blocks.

Other checks: -inf at y + 1e-6 u0, finite at y + 1e-15 u0; draws (40000)
with null component below 1e-12 and covariance within 6 standard errors of
Sigma or Q^+; Q Q^+ Q = Q within 1e-12 max|Q|, Q^+ U0 = 0; start_value
recovers theta within 1e-10.

Speed, host, one thread, gcc 14.2, S = (D1'D1, D2'D2, D3'D3), rank d - 1,
seconds (the median of 5 calls at d = 500, one call at d = 2000):

| | improper d = 500 | degenerate d = 500 | improper d = 2000 | degenerate d = 2000 |
|---|---|---|---|---|
| construction of weighted_sum (null space) | 1.04 | | 113 | |
| log_pdet of the structure | 0.38 | 0.37 | 32.2 | 28.3 |
| theta_state | 0.39 | 0.41 | 31.7 | 35.4 |
| constrained_state | 0.42 | | 36.8 | |
| logpdf | 6.7e-5 | 6.4e-5 | 1.4e-3 | 1.3e-3 |
| A = 1, K = 0 (W r) | 1.4e-4 | 1.3e-4 | 5.9e-3 | 3.3e-3 |
| A = 0, K = 1 | 1.2e-4 | 0.040 | 4.0e-3 | 4.1 |
| A = 0, K = 2 | 1.7e-3 | 0.12 | 0.036 | 13.3 |
| theta table, orders 1-2 (9 entries) | 0.010 | 0.86 | 0.20 | 78.6 |
| a block read from the table | < 1e-6 | 6.6e-5 | 6e-7 | 1.3e-3 |

Memory (doubles): the state 7.9e6 at d = 500 and 1.26e8 (1.0 GB) at
d = 2000, most of it the structure's evaluated log_pdet state; the scratch
of the degenerate blocks 2.6e8 at d = 2000 (64 packed matrices for the
inverse recursion to order 6).

What it says: at the penalty scale the improper class costs what the
structure's log|Q|_+ costs (theta_state), and every block afterwards takes
milliseconds; the factor of `constrained_state` adds about 5 s at
d = 2000. The degenerate blocks with K >= 1 cost d^3 products per mask
(unblocked loops: 4 s per tuple at d = 2000), so the table matters there.
Open: the construction of weighted_sum at d = 2000 (113 s) is the
structure's, not this module's.

## atom_mixture_distrib: atoms on a parent distribution (2026-11-17)

What: `include/laplace/distrib/core.hpp` gains the atoms of a distribution,
`n_atoms`, `atom_point(i)` and `atom_log_mass(i)`, read through
`distrib_n_atoms<D>()`, `distrib_atom_point(d, i)` and
`distrib_atom_log_mass(d, i)`, which answer for a distribution that declares
none, so no existing header changed. An atom is a point carrying mass beyond
what `kind` implies, so a discrete distribution declares none.
`include/laplace/distrib/atom_mixture.hpp` is new:
`atom_mixture_distrib<D, M...>` with one `atom_mode` per atom in the type, the
points as values, the parent held by value, and the factories
`zero_adjusted` and `zero_one_adjusted`. The plan and the decisions are
`note/piano_wrapper.md` section 5bis.

How. The wrapper is a draw over K = N + 1 cells, the N atoms and the parent.
Adjusted and inflated differ in one thing, whether the parent is renormalized
off that point: with R = 1 - sum over the adjusted points of p(c_k; theta),
q_j = pi_j adjusted and q_j = pi_j + pi_N p(c_j; theta) / R inflated, and the
density away from the atoms is pi_N f(y; theta) / R. R is identically one for a
continuous parent. Inflating a point where the parent has no mass is the same
model as adjusting it, term by term, so a continuous parent with an inflated
atom is a static_assert.

The masses are a simplex and not a box, so the declared parameters are the free
coordinates of the additive log-ratio map with the parent's cell as the
reference, each on the whole line with an identity link, every point of R^N
giving a valid distribution. They follow the parent's parameters in theta.

What that buys, and it is the reason the alr road costs no new numerics: with
S the log-sum-exp of the free coordinates, log pi_c = u_c - S for an atom and
log pi_N = -S, so every derivative of order two and above is -d^K S, the same
for every cell, and d^K S is the order K - 1 derivative of pi, which
`alr_bijector::inverse_dk` supplies. Away from the atoms the log-density is
log pi_N(u) + log f(y; theta) - log R(theta), a sum of terms each in one block,
so for a continuous parent every mixed component is exactly zero and the
order-six tensor is block diagonal at every y; at an adjusted atom the
log-density is log pi_j(u) alone and every component with a parent index is
exactly zero. The expected information needs no Leibniz sum for the same
reason: each parameter sits in one factor, the parent block being
pi_N I_parent(theta) and the free block the categorical diag(pi) - pi pi'.
An inflated atom and the cumulative functions have the same shape,
log(A(u) + pi_N(u) g(theta)), and go through one routine that takes each part's
derivative as a plain product and turns the sum into its logarithm by the Bell
relation of `generic_cdf.hpp`, read in the ratios d_T F / F.

Where R is one and no atom is inflated the wrapper carries its own cumulative
functions to order six on both tails, expected information to order four and
quantile; otherwise those members are not declared (a `requires` clause) and
the generic routes answer, which the test checks against a direct sum of the
masses.

**Measured.** `tests/distrib/test_atom_mixture.cpp`, 15 cases and 3046
assertions, green on gcc and on CUDA with the same counts. No reference file:
every quantity has a twin inside the test. The atoms' masses plus the
quadrature of the density between them add to one to 1e-10, and a discrete
parent's masses sum to one over the support to 1e-12; the value agrees with
the cell's log mass plus the parent's to 1e-15 (continuous) and 1e-13
(discrete); every component of the orders one to six is within 2e-5
(continuous) and 3e-5 (discrete) of five-point central differences of the order
below, at the atoms and away from them; the expected information agrees with
-E[l_ab], taken as the atoms' masses plus a quadrature, to 1e-8, and its
derivatives to order four with differences to 2e-5; the cumulative functions
agree with a quadrature plus the masses to 1e-9 and F + (1 - F) = 1 to 1e-12,
and their derivatives to order six on both tails with differences to 3e-5; the
quantile inverts the cumulative function to 1e-9; the mixed and the atom-parent
components come out as exact zeros, not small ones.

The worked case, a Poisson of mean 3.5 with zero inflated and five adjusted at
pi = (0.20, 0.05, 0.75): the Poisson puts 0.030197 at zero and 0.132168 at
five, so R = 0.867832, and the law reads 0.226097 at zero, keeping the
Poisson's renormalized share on top of its own cell, and 0.05 at five, below
the 0.132168 the Poisson gave there, which is what an inflated atom can never
do.

**The thread stack on CUDA, which was a conjecture and is now a measurement.**
`alr_bijector::inverse_dk` asks for 512 doubles of scratch, 4 KB, and the
cumulative route adds four arrays of 64 doubles, so the kernels looked likely
to need the 8192-byte per-thread stack that `tests/main.cpp` sets. They do not:
the same 3046 assertions pass at 8192, 4096, 2048 and at **1024 bytes, the
CUDA default**, with no stack raised at all, the device memory price falling
from 588 MB at 8192 to 252 at 4096, 84 at 2048 and nothing at 1024 (RTX 4070
SUPER). nvcc places those arrays in local memory rather than in the call
frame. A kernel that calls the generic routes on the wrapper still needs the
raised stack, for the reason `tests/main.cpp` gives, which is the quadrature's
own tensors and not this file's.

**Chosen.** As above. A continuous parent takes at most two atoms, which is
what `y_kinks()` can report so that a quadrature panel splits at each of them,
a `distrib_kinks` carrying two points; a discrete parent declares no atom and
takes up to six. `y_regularity` is inherited from a base specialized on the
kind rather than declared, a discrete parent having none: nvcc instantiates
every static member of a class eagerly where gcc waits for a use, so declaring
it unconditionally compiles on one and not on the other.

**Open.** The atom branches of the generic routes, which no distribution of
this library needs: a continuous wrapper carries its own cumulative route and
a discrete one declares no atom, so they would serve only a distribution
written outside this file. The other wrappers of `note/piano_wrapper.md`,
truncation first.

## The multivariate Student t: mvstudent_t1 and mvstudent_t2 (2026-11-17)

What: `mvstudent_t1_distrib<S>` (scale matrix Sigma = M(theta)) and
`mvstudent_t2_distrib<S>` (inverse scale Q = M(theta)), degrees of freedom
nu > 0 as the last component of "the rest" (theta, nu); proper over a pdmat,
and for mvstudent_t1 the degenerate class over a psdmat with a constant null
space (`weighted_sum_psdmat`, host only). The improper class of mvstudent_t2 is
not built. Headers `include/laplace/distrib/mvstudent_t1.hpp`,
`mvstudent_t2.hpp`, the state in `mvstudent_t_state.hpp`, shared code
`distrib/detail/mvstudent_t_common.hpp`. Test
`tests/distrib/test_mvstudent_t.cpp`, references
`tools/reference/distrib_mvstudent_t.py` (12568 mpmath values) and
`tools/reference/statmodels7_mvstudent_t.R` (3378 frozen values), closed forms
checked by `tools/reference/mvstudent_t_forms.py` (SymPy), bench
`benchmarks/distrib_mvstudent_t.cpp`.

How. rho = d (proper) or the rank (degenerate), r = y - mu, W = Sigma^-1,
Sigma^+ or Q, delta = r' W r:

    l = c_rho(nu) -+ 1/2 log|M| - (nu + rho)/2 log1p(delta/nu),
    c_rho(nu) = lgamma((nu + rho)/2) - lgamma(nu/2) - rho/2 log(nu pi).

- The state is the Gaussian state of the same matrix (W, the factor, log|M|,
  the theta table unchanged) plus nu and c_rho^(k)(nu), k = 0..6.
- c_rho by integer shifts of the univariate t's f(x) = lgamma(x + 1/2) -
  lgamma(x) - 1/2 log x (`math::own::detail_st::fder`): with b = nu/2,
  m = floor(rho/2), c_rho = sum_(j=1..m-1) log1p(j/b) + [rho odd](f(b + m) +
  1/2 log1p(m/b)) - rho/2 log 2pi, and d^k log1p(j/b)/db^k = (-1)^(k-1) (k-1)!
  b^-k expm1(-k log1p(j/b)), one sign. Measured before the choice (mpmath at 60
  digits, f exact): c'_rho within 2.7e-16 relative for d = 2..1000 and nu = 2
  to 1e10; the direct digamma form loses 1.9e-4 at d = 3, nu = 1e6 and has the
  wrong magnitude at 1e10 (d = 2 is exactly zero).
- Blocks: Faa di Bruno through delta, whose derivatives are r' W_U r,
  -2 (W_U r)_i, 2 (W_U)_ij and zero with three or more mu: a block is the sum
  over the set partitions of its A + K_theta positions with at most two mu per
  part, weighted by G(n, c) = d^n/d delta^n d^c/d nu^c g,
  g = -(nu + rho)/2 log1p(delta/nu), in closed form without cancellation as
  nu grows (n = 1: c (delta - rho)). The pure nu blocks: student_t1's pure nu
  derivative plus -(rho - 1)/2 (-1)^(c-1) (c-1)! nu^-c expm1(-c log1p(delta/nu)).
  Nonzero up to A = 6 (`sym_tensor` over d for A >= 3).
- The far tail (decided 2026-11-17): where delta >> nu the terms of the sum are
  of size one and the block of size nu/(nu + delta) (1-D: d2 log(nu + delta
  e^-2s) = -4u^2 + 4u = 4u(1 - u), u = delta/(nu + delta)). The sum, its weights
  and its inputs r' W_U r and W_U r (r by two_sum) are in double-double for
  A + K_theta >= 3; orders 1 and 2 stay in double. Worst ulp1N before and
  after: diagonal (large residuals, delta/nu ~ 170) A = 1, K = 5, 1.0e4 to
  below 1.5; compound_symmetry near singular A = 4, K = 2, 7.7e3 to 136, A = 2,
  K = 4, 3.3e3 to 46; log_cholesky near singular A = 2, K = 4, 3.8e3 to 266 (gcc;
  784 on CUDA). A split log(nu + delta) = log delta + log1p(nu/delta) helped only
  where one component dominates exactly. Cost: 3.3 times per order-3 block at
  d = 3 and 4.7 at d = 5 (one thread, host); orders 0 to 2 unchanged.
- Expected information (Lange, Little and Taylor 1989), kappa = (nu + rho)/(nu +
  rho + 2), t_a = d_a log|Sigma|: I_mumu = kappa W, I_mu,rest = 0, I_ab = kappa
  I^G_ab - t_a t_b/(2 (nu + rho + 2)), I_a,nu = -t_a/((nu + rho)(nu + rho + 2)),
  I_nunu = 1/4 [psi'(nu/2) - psi'((nu + rho)/2)] - rho (nu + rho + 4)/(2 nu (nu +
  rho)(nu + rho + 2)). Checked against E[score score'] by quadrature over the law
  (mpmath at 20 digits): proper d = 2, 4.2e-22; degenerate d = 3, rho = 2,
  3.4e-21, which confirms rho in place of d on the range.
- Small nu (decided 2026-11-16): along a direction where Sigma^-1 d_a Sigma is a
  multiple of the identity the two terms of I_ab cancel. For closed_factor
  structures I_ab = kappa Ttilde_ab/2 + nu t_a t_b/(2 rho (nu + rho + 2)),
  Ttilde from the whitened B_a of the Gaussian's factor form with the diagonal
  centred (one flag added to `expected_info_factor`). Measured on
  unit_diagonal_scaled<ar1> with common_scale, exact I_ss = 2 d nu/(nu + d + 2):
  <= 0.81 ulp for nu down to 1e-3 and d = 3..10, against 48 to 3370 ulp for the
  direct form at nu <= 0.01. factor_analytic, matrix_log and the degenerate class
  keep the direct form (loss about (nu + rho)/nu there).
- I_nunu cancels as nu^-2 relative (the direct form with correctly rounded
  trigammas: 1.2e-6 at d = 1, nu = 1e4; 7.7e-3 at 1e5). By a recurrence in rho
  found with SymPy, I(nu, rho) - I(nu, rho - 2) = 4 (h + 1)/((nu + 2h)(nu + 2h -
  2)^2 (nu + 2h + 2)) + 8 (h - 1)^2/(nu (nu + 2h)(nu + 2h - 2)^2 (nu + 2h + 2)),
  h = rho/2, positive; I(nu, 2) = 8/(nu^2 (nu + 2)(nu + 4)) and I(nu, 1) is
  student_t1's. Its nu derivatives are sums of positive products: no series,
  no cancellation, O(rho) per state.
- Draws: y = mu + L z/sqrt(w) (L^-T z for mvstudent_t2), w = 2 G/nu,
  G = P^-1(nu/2, u), z at the indices 0..d-1 and u at index d of
  `counter_uniform_source{seed, unit}`; degenerate (I - P) L z/sqrt(w).
  400000 draws (log_cholesky, d = 3, nu = 8): mean within 0.0032 sd, covariance
  nu/(nu - 2) Sigma within 0.38 per cent of its largest entry.
- Moments NaN where they do not exist (mean nu <= 1, covariance nu <= 2);
  start_value nu = 5, Sigma from 0.6 times the sample covariance.

Accuracy against mpmath at 60 digits from the definition (Cholesky or the
eigendecomposition; the blocks by one-dimensional mpmath.diff along the
directions of the polarization identity, the information from the formula above
with the Gaussian part from F = E[-l] as in the Gaussian references). Points:
per structure a random one at nu = 4.5, and nu = 1e6 (log_cholesky), nu = 0.05
(ar1, scaled_ar1), large residuals at nu = 2.5 (diagonal), near singular at
nu = 6 (log_cholesky, log L33 = -11.5) and nu = 3 (compound_symmetry,
theta = 8). Worst ulp1N, gcc 14.2 / CUDA:

| | blocks, orders <= 2 | blocks, orders >= 3 | information |
|---|---|---|---|
| mvstudent_t1 diagonal | 1.47 / 1.47 | 1.49 / 0.92 | 0.55 / 0.55 |
| mvstudent_t1 ar1 | 1.96 / 1.63 | 2.68 / 2.68 | 2.15 / 1.16 |
| mvstudent_t1 compound_symmetry | 5.19 / 5.19 | 136 / 136 | 0.42 / 0.44 |
| mvstudent_t1 log_cholesky | 17.1 / 12.2 | 622 / 784 | 1.49 / 2.51 |
| mvstudent_t1 scaled_ar1 | 1.66 / 1.14 | 5.19 / 5.19 | 1.43 / 1.17 |
| mvstudent_t1 factor_analytic | 2.98 / 2.98 | 6.46 / 6.46 | 19.7 / 19.7 |
| mvstudent_t2 diagonal | 1.15 / 1.15 | 3.27 / 3.27 | 0.72 / 0.72 |
| mvstudent_t2 ar1 | 1.55 / 1.44 | 13.5 / 13.6 | 2.04 / 2.24 |
| mvstudent_t2 compound_symmetry | 1.59 / 1.59 | 1.25 / 1.25 | 0.41 / 0.44 |
| mvstudent_t2 log_cholesky | 1.08 / 1.84 | 2.39 / 5.7 | 2.51 / 2.79 |
| mvstudent_t2 scaled_ar1 | 2.58 / 2.58 | 2.51 / 2.51 | 1.55 / 5.57 |
| mvstudent_t2 factor_analytic | 1.11 / 1.11 | 2.02 / 1.96 | 8.13 / 7.38 |
| degenerate, ws d = 4, rank 2 | 0.72 | 3.4 | 0.22 |
| degenerate, ws d = 5, 6 | 10.2 | 32.6 | 5.58 |
| degenerate, ws d = 5, two D2 | 16.8 | 18.4 | 7.85 |
| degenerate, spread (theta 9 apart), nu = 1e5 | 1.1e5 | 1.8e6 | 2.6e5 |

(orders: A + K, nu included; the degenerate rows are host code, the same on both
builds.) What it says: the mvstudent_t2 blocks stay within 14 ulp1N; mvstudent_t1
loses at the two near-singular points, where the errors of d_U Sigma^-1 (the
structure's inverse recursion) enter sums that the far tail makes
ill-conditioned; the spread degenerate point is the dense (Sigma + cP)^-1 of the
degenerate Gaussian (section above), not the t. The bounds of the test are
these numbers times 1.5 per (structure, A, K).

Against statmodels7 (log_cholesky, d = 2, 3, nu = 0.8, 4.5, 400, orders to 4 and
the information at order 0; statmodels7's own rounding included): worst
16.5 ulp1N on mvstudent_t1 and 14.4 on mvstudent_t2.

Speed (gcc 14.2, OpenMP 16 threads, n = 1e5, log_cholesky, nu = 5), ns per
unit, theta shared / shared with the theta table / per unit:

| | order 0 | order 1 | order 2 |
|---|---|---|---|
| mvstudent_t1 d = 3 | 4.4 / 4.8 / 115 | 118 / 50 / 217 | 1144 / 283 / 1241 |
| mvstudent_t2 d = 3 | 4.2 / 3.9 / 110 | 57 / 48 / 154 | 318 / 267 / 412 |
| mvstudent_t1 d = 10 | 10.9 / 9.3 / 128 | 4493 / 834 / 4732 | 4.63e5 / 4.46e4 / 4.70e5 |
| mvstudent_t2 d = 10 | 9.5 / 9.9 / 102 | 957 / 889 / 1073 | 3.82e4 / 3.78e4 / 3.94e4 |

Order 2 at d = 10 costs about 19 times the Gaussian's (mvgaussian2, 2021 ns):
each (rest, rest) pair needs the quadratic forms of its four sub-tuples. Not
measured on CUDA yet.

Open: the improper (intrinsic) class of mvstudent_t2; the double-double sums
take the inputs d_U W in double, so the near-singular mvstudent_t1 cells keep
the structure's own error (622 to 784 ulp1N).

## transformed_distrib: a distribution carried through a fixed bijector (2026-11-17)

What: `include/laplace/distrib/transformed.hpp` is new:
`transformed_distrib<D>`, the law of Y = g(X) with X following a continuous
parent D and g a `laplace::bijector::scalar_bijector` whose own parameters are
fixed values, and the host factory `transformed(parent, map)`. The plan and the
decisions are `note/piano_wrapper.md` section 5quater: bijections only, the
map's parameter not modelled.

How. With h = g^-1, l(y; theta) = log f_X(h(y); theta) + log |h'(y)|. The
Jacobian term carries no theta, so `logpdf_dk<K>` is the parent's at h(y)
unchanged, and so are the expected information and its derivatives, since
E_Y[a(h(Y))] = E_X[a(X)] for any a. Only y needs the chain rule:
`logpdf_dy_dk<J, K>` is Faa di Bruno over the parent's d^k_x d^K_theta l_X and
the bijector's h^(m), with the bijector's d^J log|h'| added when K = 0. The
cumulative functions are the parent's own logarithms at h(y), the two tails
swapped for a decreasing map, the atom's mass moved to the side that holds it
where the parent has one at h(y); the quantile is g of the parent's quantile, or
of its upper quantile at log p for a decreasing map. The direction is read by
the factory from the sign of g' at the parent's center. The atoms keep their
mass and move to g(c); `logpdf` at an atom's image is read at c itself, so that
rounding in h(g(c)) does not move it off the atom. `start_value` maps the
sample back point by point. The ends of the support are taken from the
bijector's image bounds where the parent's end is an end of the map's domain:
`yeo_johnson` inverted returned NaN at plus and minus infinity rather than its
limit, and an infinite support end would otherwise have become NaN. The
bijector was repaired the same day (the section "The yeo_johnson bijector at
the ends of its two domains"), and the wrapper still reads the bounds, which
costs it nothing.

Measured (`tests/distrib/test_transformed.cpp`, gcc and CUDA, the same numbers on both; worst error, bound in
brackets):

| case | quantity | worst | bound |
|---|---|---|---|
| gaussian2 through exp vs lognormal1 | value, d1-d6, log F and its d1-d6, quantile, information d0-d4 | 0 | 4e-16 |
| | `logpdf_dy_dk`, J + K <= 6 | 5.6e-16 | 1.5e-15 |
| exponential through x^(1/k) vs weibull | value | 3.8e-16 | 1e-15 |
| | log F, log(1 - F) | 5.2e-15 | 1.5e-14 |
| | quantile | 6.9e-16 | 2e-15 |
| | d1-d6 in mu (weibull's in m through m(mu)) | 3.0e-15 | 1e-14 |
| | d1-d6 in y | 6.1e-15 | 2e-14 |
| | information vs I_mm (dm/dmu)^2 | 1.8e-16 | 1e-15 |
| affine, scale -2, over gaussian2 vs gaussian2 | value; log F; quantile; y derivatives | 1.0e-16; 0; 7.6e-16; 0 | 4e-16; 4e-16; 2e-15; 4e-16 |
| yeo_johnson 0.6 inverted over gaussian2 | total mass - 1 | 1.1e-16 | 1e-15 |
| | information over Y by quadrature vs parent's | 1.4e-16 | 1e-15 |
| | F vs quadrature; F(Q(p)) - p | 1.8e-15; 5.2e-16 | 6e-15; 2e-15 |
| | y derivatives vs differences | 2.9e-11 | 1e-10 |
| box_cox 0.5 over gamma2 | mass; F vs quadrature; y derivatives | 3.3e-16; 3.4e-15; 1.6e-9 | 1e-15; 1e-14; 5e-9 |
| bounded (-1, 3) inverted over gaussian2 | mass; F(Q(p)); y derivatives | 2.2e-16; 4.2e-16; 1.1e-10 | 1e-15; 2e-15; 5e-10 |
| zero-adjusted gamma2 under 2v + 2 | mass; atom point, mass, value, d3 | 2.2e-16; exact | 1e-15; == |
| atom on gaussian1 under -v | mass; log F derivatives vs differences | 2.2e-16; 2.9e-10 | 1e-15; 1e-9 |

What it says: where an oracle exists the wrapper reproduces it to the last bit
or within a few ulp, and the information computed over Y agrees with the
parent's, which checks the argument that it is the parent's.

The CUDA build gives the same worst errors as gcc in every row.

### transformed_distrib: atoms on the device and the decreasing quantile (2026-11-18)

Two defects of the atom routes, found by inspection and then measured.

The image of an atom on the device. `at_atom` and `preimage` compared y with
g(c) recomputed at each call. On the device g(c) need not round to the same
double as on the host: the platform's elementary functions differ in the last
bit, and nvcc contracts a * b + c into an FMA. This is the same class of
defect as a comparison moved by FMA contraction. An observation computed on
the host as an atom's image then missed the atom inside a kernel, and its mass
was read as a density. Measured on CUDA with 320 atoms (gaussian1 with an atom
at c = -2.3 + 0.0731 k, k = 0..63, under exp, the logistic onto (-1, 3),
Yeo-Johnson's inverse with exponent 0.6, sinh and Phi): g(c) recomputed in the
kernel differs from the host's value for 59 of them. Repair: transformed()
computes the images once on the host and stores them in the field
`atom_image`, against which both comparisons now run. After the repair 0 of
320 are missed on CUDA, and the log mass read in the kernel equals the host's
exactly. An aggregate built without the factory over a parent with atoms must
fill `atom_image` itself.

The quantile under a decreasing map with atoms. It was g(Q_X(1 - p)), and
1 - p rounds to 1 below p = 1.1e-16. Over gaussian1 (0.2, 1.3) with an atom
at 0.5, mapped by g(x) = -x, the error |log F(Q(p)) - log p| was 6.7e-9 at
p = 1e-8 and 9.9e-3 at 1e-14, and Q was -inf from p = 1e-17 down. Repair: the
parent's upper quantile at log p, as without atoms, since the generic upper
route returns the atom point when log p falls in the atom's jump. After the
repair the error is at most 2.8e-14 for p from 1e-3 down to 1e-100 (the bound
is 1e-13), and quantile(0.3), which lies inside the jump, is the atom's image
exactly. The same numbers hold on gcc and CUDA.

## truncated_distrib: a parent restricted to a fixed window (2026-11-17)

What: `include/laplace/distrib/truncated.hpp` is new: `truncated_distrib<D>`,
the parent restricted to a window [a, b] and renormalized, an aggregate with
the members `parent`, `lower` and `upper`, and the host factory
`truncated(parent, lower, upper)`. The bounds are values, independent of the
parameters theta, so the wrapper adds no parameter: `kind`, `n_params` and the
parameter descriptors are the parent's. The plan is `note/piano_wrapper.md`
sections 4, 5, 5bis and 5ter.

How. Inside the window l(y; theta) = log f(y; theta) - log Z(theta), with
Z = F(b) - F(a-), and minus infinity outside. Z is taken on the tail of the
parent that holds the smaller numbers, so that neither cumulative value is the
complement of a small one: F(b) - F(a-) when F(b) <= S(a-), with S = 1 - F,
and S(a-) - S(b) otherwise. Either is a difference A - B of two numbers held on
the log scale, log Z = log A + log1p(-exp(log B - log A)), which keeps a window
deep in a tail from underflowing. Its derivatives in theta to order six come
from the parent's own derivatives of log F or log S, read through the
dispatchers of `generic_cdf.hpp` (so a parent without closed cumulative
derivatives gets the generic ones), by the Bell relation read in the ratios:
d_T (A - B) / (A - B) = (A / Z) d_T A / A - (B / Z) d_T B / B, and the
derivatives of the logarithm follow by the same relation solved for them
(`ratios_from_log_tail_dk` and `log_tail_dk_from_ratios`). The cumulative
functions of the truncated law are differences of the same kind over Z, on the
tail Z is taken on, and go through the same routine.

The left limit reads the mass at the bound from the object, never from `kind`:
for a continuous parent it is the mass of an atom the parent declares at a
(`distrib_atom_point`, `distrib_atom_log_mass`), subtracted from F(a) on the
lower tail and added to S(a) on the upper; for a discrete parent F(a-) is
F(ceil(a) - 1), the same number without the subtraction. A zero-adjusted gamma
truncated to [0, 3] keeps its atom at zero, F(0-) = 0 and Z = F(3); reading
the bound as a continuous point gives Z = F(3) - pi_0 and a law that does not
integrate to one.

The factory cuts the window to the parent's support, collapses a truncation of
a truncation to one truncation at the intersection (the nested type itself is
a static_assert), and refuses with `std::invalid_argument` a window with no
mass, a window of one point, and a window that leaves out a cell of the parent,
whose coordinate would have an identically zero score; the message names the
equivalent model. An atom mixture over a discrete parent declares no atoms, its
points being support points, so its cells are read from `n_atom_points`. The
cells are values, so this refusal can only be made at run time.

The expected information is not declared: the information of a truncated law
is not the parent's, and the generic route of `generic_info.hpp` answers,
integrating over the window.

**Measured.** `tests/distrib/test_truncated.cpp`, 16 cases and 3169
assertions, green on gcc and on CUDA (RTX 4070 SUPER) with the same counts.
One reference file, for the narrow windows (below); every other quantity has a
twin inside the test. log Z of a Gaussian
agrees with a difference of erfc values to 1e-12, in the middle, in a narrow
window [1, 1.01], deep in each tail ([10, 12] and [-12, -10], mean 0.4,
standard deviation 1.3) and on half lines, and the density integrates to one
to 1e-10; a truncated Poisson sums to one to 1e-14, and the zero-truncated
Poisson agrees with its closed form to 1e-14; every component of the orders one
to six is within 2e-5 of five-point central differences of the order below,
in the middle, near both bounds, in the narrow window, in both tails and for
the Poisson; the cumulative functions agree with a quadrature to 1e-10,
F + (1 - F) = 1 to 1e-13, and their derivatives to order six on both tails
with differences to 3e-5; the quantile inverts the cumulative function to
1e-9; the zero-adjusted gamma on [0, 3], an atom mixture with atoms at 0 and
1 on [0, 2] (the atom on the lower bound kept) and an atom at 0 on a Gaussian
truncated to [0.5, 3] integrate to one with their atoms to 1e-10; the generic
expected information agrees with -E[l_ab] by a quadrature over the window to
1e-9, and its first derivative with differences to 1e-6; three kernels agree
with the host to 1e-11.

The narrow window costs precision that the log form cannot give back: the two
cumulative values are close, and the relative error of Z grows like the
machine epsilon over the width of the window. In [1, 1.01] at its middle the
mu score is 2e-6 while the noise of the differences was about 5e-11, so the
check point sat at y = 1.0095, away from the middle, until the repair below.

How large the loss is was measured on log Z of a Gaussian with mean 0.4 and
standard deviation 1.3 truncated to [a, a + w], for a in {1, 4, -3}, against
mpmath at 200 bits (the reference takes the double nearest to a + w as the
upper bound, so that the rounding of the bound is not counted). The table
gives the largest absolute error of log Z over the three windows, which is the
relative error of Z, for the cumulative route that Laplace used before the repair below and, for
comparison, for a `panels_quad` of the density over the window:

| w | cumulative route | quadrature of the density |
|---|---|---|
| 1e-2 | 1.4e-14 to 3.9e-14 | 0 to 9e-16 |
| 1e-4 | 2.6e-13 to 4.7e-12 | 0 to 1.8e-15 |
| 1e-6 | 2.2e-10 to 2.5e-10 | 0 |
| 1e-8 | 8.6e-9 to 4.2e-8 | 0 |
| 1e-10 | 1.6e-6 to 3.9e-6 | 0 |
| 1e-12 | 2.5e-4 to 4.3e-4 | 0 |

The cumulative route loses about 4e-16 / w, as the cancellation predicts; the
quadrature keeps the machine epsilon.

**A device defect found on the way, in `atom_mixture.hpp`.**
`detail_atom_mixture::flat_tensors::offset()` was a host-only `constexpr`
function. Called with a run-time order from device code (`set`, `at`,
`fill_value_order`), which the truncation does at every order, it is not
allowed without `--expt-relaxed-constexpr`, and nvcc said so in a warning
(#20013) rather than an error. On the RTX 4070 SUPER the derivatives of log Z
came out as zeros (the Poisson's order-two component read the parent's
-0.408163 instead of -0.295226), and the order-two kernel of the Gaussian
never returned. Marking `offset()` `LAPLACE_HD` repaired both: every term agrees
with the host to the last few ulp, and the atom mixture's own 15 cases and
3046 assertions still pass on gcc.

**The kernel, and what it costs ptxas.** One kernel that called the order-six
derivatives of three truncated laws (Gaussian, zero-adjusted gamma, Poisson)
stopped ptxas with "Memory allocation failure"; split into three, the
translation unit took about two hours and 20 GB in ptxas. The kernels now call
orders one and two only; the orders three to six are the host cases' and run
through the same code. The translation unit compiles in about ten minutes.

**Chosen.** As above. `logpdf_dy_dk<J, K>` is constrained on the parent's kind
alone: nvcc rejected the nested requires-expression that probed the parent for
the member, where gcc accepted it.

**Repaired afterwards.** `atom_mixture_distrib` now reaches its parent through
the dispatchers (`distrib_expected_info`, `distrib_log_cdf`, `distrib_log_ccdf`
and their derivatives) rather than through the parent's members, so an atom on
a truncated parent has its closed information, checked against a quadrature of
the score products plus the atom to 1e-9 and its first derivative against
differences to 1e-6. The generic expectation (`expectation_by_integral` and the
generic information built on it) now adds the atoms' masses times h at their
points to the quadrature of the density, so the generic information of a
truncated zero-adjusted gamma matches the same quadrature to 1e-9 and its total
mass is 1 to 1e-10. The other host-only `constexpr` functions that nvcc named in
#20013 warnings (`fact` and `sgn` in the bijector helpers, `distrib_n_atoms`,
`regularities_of`) are now `LAPLACE_HD`; after the change nvcc reports none of
Laplace's own (the ones left come from Boost's error handling).

**The narrow-window loss, repaired (decided with Giovanni).** When
log B - log A > log(15/16) and both ends are finite, the mass of the interval
is summed rather than taken as a difference: for a continuous parent, a
16-node Gauss-Legendre rule of the density on each piece between the parent's
`y_kinks()`, plus the atoms inside the interval; for a discrete parent, the
masses at the integers, when there are at most 64. The terms are added by
log-sum-exp, and the derivatives fold each term in through the same ratio
relation as the difference. Past 15/16 the difference loses more than four
bits, and on so short an interval 16 nodes integrate the density to the last
bit; a spike narrower than the window would need an adaptive rule. The same
rule serves Z, the mass of [a, y] for F and that of (y, b] for 1 - F.
Reference: `tests/data/reference/distrib_truncated_narrow.csv`, written by
`tools/reference/distrib_truncated_narrow.py` from mpmath.diff of the
definition at 400 and 500 bits (rows kept when the two agree to 2^-80; 270
rows, none dropped): log Z of the Gaussian above on the 18 windows and its
derivatives in (mu, sigma) to order four. Measured: log Z within 2.03e-15
(bound 4e-15), the derivatives within 2.28e-13 of the largest reference of
their order and window (bound 5e-13); a Poisson(1000) on [1000, 1001] equals
the direct sum of its two masses (error 0, bound 4e-15). The derivative check
in [1, 1.01] now also runs at its middle, y = 1.005, and the device kernel
evaluates a window of width 1e-8.

**The quantile, the starting value and an atom on the upper bound
(2026-11-17, later).**

Quantile in a narrow window. The quantile inverted the parent at the target
F(a-) + pZ (or at log(S(b) + (1 - p)Z) on the upper tail), a difference that is
never summed, so y missed by about eps F(a) / f(y), which is eps F(a) / Z of the
window's width. Measured against mpmath at 300 bits (bisection on the
definition) on the Gaussian above, windows [a, a + w] with w from 1e-2 to
1e-12, p in {1e-6, 0.1, 0.3, 0.5, 0.7, 0.9, 1 - 1e-6}, largest error in ulps of
y:

| a | before | after |
|---|---|---|
| 1, 4, -3, 0.4 | 0.1 to 12 | 0.1 to 0.5 |
| -1e-3 | 1.3e3 to 3.1e5 | 0.4 to 0.5; 1.9e3 at w = 1e-2 |
| 0 | 5.4e7 to 8.1e15 | 3.9 to 24 |

Where |y| is of the order of F(a) / f(a) the error was already near the ulp of
y, which is the best a double holds; near zero it was not, and at a = 0,
w = 1e-12 the quantile missed by 2e-4 of the window. The repair (decided with
Giovanni): where Z is summed, the result takes one Newton step on the summed
mass below it, M(a, y) / Z - p over the density, or on the mass above it for
p > 1/2. The summed mass has a relative error, so the step lands within a few
ulps; a second step measured no better. At a = 0 the remaining 4 to 24 ulps are
at most 1.6e-15 of the window, and the 1.9e3 ulps at a = -1e-3, w = 1e-2 are
1.3e-16 of the window at a y that crosses zero: the rounding of the residual
against a y much smaller than the bounds. A discrete parent's window holds at most 64
integers there, and the quantile is the first whose summed mass from the lower
bound reaches pZ. The test checks |F(y) - p| <= 1e-13 + 8e-16 |y| / w on 15
windows and five p (the unrepaired quantile fails it by 2e-4 at a = 0), and a
Poisson(1000) on [1000, 1001].

Quantile of a parent with atoms, upper tail. The upper tail went to the linear
scale through -expm1(t) whenever the parent had atoms, because the generic
root could not invert a cumulative function with jumps; since the generic
quantile handles atoms (the section on the generic cumulative routes of a law
that carries atoms), the log-scale route serves every parent. Measured on a
Gaussian(0.4, 1.3) with adjusted atoms at 0 and 1 (alr -1, -2) truncated to
[0, 8] and [0, 20], p = 1 - 10^-k for k = 1 to 16, against mpmath: the linear
route reached 8.2e-3 relative error at 1 - p = 1.1e-16 (1.3e-3 at 1e-15,
7e-7 at 1e-12); the log route stays within 1.9e-16. The test checks 1 - F(y)
against 1 - p to 1e-12 at 1 - p = 1e-5, 1e-10 and 1e-15.

Starting value. `start_value` passes the whole sample to the parent's own
start, which estimates the untruncated law. Measured on samples drawn from the
truncated law (n = 200 and 5000) of a Gaussian(0.4, 1.3) on [-0.5, 2], [1.5, inf),
[3, inf), (-inf, -1] and [0, 0.5] and of a gamma2(2, 0.5) on [3, inf) and [0, 1]:
the start's scale is about half the truncated MLE's and its location is pulled
into the window; the log-likelihood falls short of the MLE's by 10 to 30 at
n = 200 and 270 to 900 at n = 5000. A damped Newton from it reached the MLE in
9 to 21 steps in 13 of the 14 cases; on [0, 0.5] with n = 200, where the
likelihood is flat in sigma, it stopped at another point. Chosen with Giovanni:
the start stays as it is and the manual says what it is.

An atom on the upper bound. A test with a Gaussian(0.4, 1.3) carrying an
adjusted atom at 2 (alr -1), truncated to [0, 2], and a Poisson(3.5) inflated at
9 truncated to [2, 9], found nothing to repair: F(b) holds the atom's jump and
1 - F(b) does not, so both tails of Z count it (each agrees with the direct
difference to 1e-14); the mass is 1 to 1e-10 (the Poisson's 1 to 1e-14);
F(b) = 1 and 1 - F(b) = 0 exactly; 1 - F just below the bound (2 - 1e-3 and
2 - 1e-9) equals the atom plus the continuous mass in closed form to 1e-12; the
quantile returns the bound for every p inside the jump and inverts F below it
to 1e-9; the derivatives at the bound, inside, and of both tails agree with
five-point differences to 2e-5 and 3e-5; a window [2 - 1e-8, 2] gives log Z to
1e-14. The tests now have 17 cases and 4048 assertions, green on gcc and on
CUDA (RTX 4070 SUPER) with the same counts; the device kernel also evaluates
the quantile of the window [1, 1 + 1e-8]. In that CUDA build no #20013 warning
names `distrib_atom_point` (which returns `quiet_NaN_v` since the narrow-window
repair) or any function of `truncated.hpp`.

## The yeo_johnson bijector at the ends of its two domains (2026-11-17)

**What was wrong.** `scalar_bijector::yeo_johnson` returned NaN at an infinite
argument, in both directions, where the limit exists and is either finite or an
infinity of a definite sign. Every closed form of
`include/laplace/bijector/detail/scalar_yeo_johnson.hpp` is written from
`exprel` or `log1prel`, so that lambda = 0 and lambda = 2 need no branch of
their own; at an infinite argument those two primitives are themselves
evaluated at an infinite argument, where each of them is inf/inf. The defect
was found by the session that wrote `transformed_distrib`, which reads the ends
of the support of g(X) from g at the ends of R and had to read the bounds of
the image instead.

Measured at the ends of the domain of h (the image of g) and at both ends of
the domain of g (which is R), over lambda in {-3/2, -1/2, 0, 1/6, 1/5, 1/4,
1/3, 1/2, 1, 3/2, 5/3, 7/4, 9/5, 11/6, 2, 5/2, 4}:

| quantity | where | before | after |
|---|---|---|---|
| h, and h' to h^(6) | an infinite end of the domain of h | NaN | the limits below |
| log of h' | an infinite end | NaN | an infinity, or 0 at lambda = 1 |
| its 1st to 6th derivatives | u = +inf at lambda = 0, u = -inf at lambda = 2 | NaN | 1 and -1 at order 1, 0 above |
| its 1st to 6th derivatives | any other infinite end | 0 | 0 |
| g | v = +inf at lambda <= 0, v = -inf at lambda >= 2 | NaN | -1/lambda and 1/(2 - lambda), an infinity at lambda = 0 and 2 |
| g' to g^(6) | v = +-inf | the limit | unchanged |
| log of g' | v = +-inf at lambda = 1 | NaN | 0 |
| its 1st to 6th derivatives | v = +-inf | 0 | 0 |
| every quantity | the finite end of the image, where there is one | the limit | unchanged |

**The limits.** The argument enters every closed form only through a power of
1 + v, 1 - v, 1 + lambda u or 1 - c u, c = 2 - lambda, so the sign of that
power's exponent decides each limit. For the inverse, with
P_n = prod_(k=1..n-1) (1 - k lambda) and Q_n = prod_(k=1..n-1) (k c - 1),

    h^(n)(u) = P_n (1 + lambda u)^(1/lambda - n)   u > 0
    h^(n)(u) = Q_n (1 - c u)^(1/c - n)             u < 0

and 1 - n lambda has the sign of 1/lambda - n, as 1 - n c has that of
1/c - n. Above zero the power runs to +inf and the prefactor gives the sign,
being positive (P_n) or of sign (-1)^(n-1) (Q_n) and vanishing in neither case;
at zero, that is at lambda = 1/n or c = 1/n, the power is 1 and the prefactor
is the whole limit; below zero the power dies and the limit is 0. The value h
itself runs to an infinity of the sign of u, the range of h being R, at the
finite end of its domain as well as at the infinite one. For the forward map
g^(n)(v) is prod_(k=1..n-1) (lambda - k) (1 + v)^(lambda - n) above and
(-1)^(n-1) prod_(k=1..n-1) (c - k) (1 - v)^(c - n) below, with the same three
cases and the prefactor (-1)^(n-1) (n-1)! on the edge, while g(+inf) and
g(-inf) are the two ends of the image: -1/lambda for lambda < 0,
1/(2 - lambda) for lambda > 2 and an infinity otherwise. The log-Jacobians are
(lambda - 1) log(1 + v) and (1 - lambda) log(1 - v) for g, and
(1/lambda - 1) log(1 + lambda u) and (1/c - 1) log(1 - c u) for h: each runs to
an infinity of the sign of its own coefficient and is 0 at lambda = 1, where
both maps are the identity. Every argument-derivative of a log-Jacobian carries
a negative power and dies, except that log of h' is u at lambda = 0 above and
-u at lambda = 2 below, whose first derivatives are 1 and -1.

Each of the 896 limits was compared with SymPy's own limit of the definition
(`tools/reference/bijector_yeo_johnson_forms.py`, `check_limits`), the script
that also writes the header. The C++ never forms the exponent: it vanishes
exactly where a product of lambda with a small integer equals 1, so the product
is compared with 1 rather than subtracted from it. A subtraction is what nvcc
contracts into an FMA, and the exact product of 3 with the double nearest 1/3
is 1 - 2^-54 and not 1, so the edge would be taken on one platform and not on
the other. The edge is therefore reached wherever lambda is the double nearest
1/n, n = 1 to 6, and on the negative branch only where 2 - lambda is exactly
the double 1/n, which holds for 1/n dyadic (lambda = 1, 3/2, 7/4) and not for
1/3, 1/5 or 1/6; at those lambda the limit returned is the infinity or the zero
that this very double carries.

Found while writing the test: one ulp from an edge the limit is right and out
of reach. At lambda = 5/3, where c = 2 - lambda is the double below 1/3, the
exponent 1/c - 3 is 6.7e-16 while the logarithm of 1 - c u reaches 138 at the
largest double, so the power stays 1 and h''' sits on the plateau 2/9 over the
whole representable range; its limit is nonetheless +inf, and that is what the
function returns at u = -inf. The same happens at lambda = 9/5 for the fifth
derivative (plateau 0.0384, limit +inf) and at lambda = 11/6 for the sixth
(plateau -0.0154, limit 0). The test tells a plateau from an approach by the
two large arguments giving the same value, and on a plateau checks only that it
is finite and carries the sign of the limit.

**The test** is in `tests/bijector/test_yeo_johnson.cpp`, in a kernel on the
default execution space: for each lambda above, the fourteen quantities (the
map and its log-Jacobian, orders 0 to 6) at the four ends, checked first
against the function's own approach to the end, where an infinite limit grows
past 1 in its own sign between the arguments 1e30 and 1e60, a zero limit falls
below 1e-6 and a finite limit is reached to 1e-12, and then against the
written-out values of the edges, of the signs of the infinities and of the
log-Jacobians. gcc and CUDA give the same counts.

**The finite end on CUDA, found on 2026-11-17 by the first CUDA run.** At the finite end
of the domain of h the quantity w = 1 + lambda u (1 - c u on the negative branch) vanishes, and
log|h'| with its u-derivatives carries negative powers of it, so the limit is an infinity. The
host produces that infinity by itself: lambda times the rounded -1/lambda rounds back to -1, w
is exactly 0, and 0^(-n) is +inf. nvcc contracts 1 + lambda * u into an FMA, which multiplies
lambda by that rounded reciprocal exactly instead of rounding the product, so the cancellation
does not complete, w survives at about 3.7e-17, and the infinity comes out finite: measured
4.5036e16, 1.21694e33, 6.57676e49, 5.33143e66, 5.76256e83 and 7.78567e100 for the orders one to
six, each about 1/w times the one before. The value and its u-derivatives escaped only because
their own form reaches the end through a product that does round back to -1. Both now take the
limit from a guard at that end (`yj_inv_at_edge`, `yj_inv_lj_at_edge`) rather than from the
arithmetic, and gcc is unchanged, the limit being what it already produced. The general shape is
worth stating: an assertion that holds on gcc because an intermediate is exactly zero is a CUDA
failure in waiting, and it does not present as a tolerance problem but as an infinity turning
into a large finite number.

**Left open by this repair.** The parameter derivatives (the `*_partials`
functions, and so `forward_partials_dK` and the rest) were NaN at every
infinite end, and so were the mixed entries of the log-Jacobian's partials at
the finite end of the image. They are repaired in the section "The yeo_johnson
bijector: the parameter partials at the ends" below, which was written on the
same day. `inverse_distance_upper` and `inverse_distance_lower` of an inverted
bijector are NaN where the bound and the value are both infinite; those two go
through the generic difference `forward_domain_upper() - inverse(u)` of
`scalar.hpp`, which every map shares, and not through anything of
yeo_johnson's own.

**box_cox** has the same hole, and wider. It is NaN at both infinite ends of
the domain of h, and also at the finite ends of either domain wherever a
vanishing prefactor meets an infinite power: g''(0) at lambda = 1, whose limit
is 0 because g is then affine, h'(-1) at lambda = 1, whose limit is 1 for the
same reason, and h''(-2) at lambda = 1/2, whose limit is 1/2. Its header states
that a value on a bound gives its limit, which is therefore not true of it
today. Not repaired.

## The generic cumulative routes of a law that carries atoms

A law may put mass on points that its kind does not account for: a gamma with a
point mass at zero is neither continuous nor discrete in the usual reading, and
`distrib/core.hpp` calls such a point an atom. A family declares `n_atoms`,
`atom_point(i)` and `atom_log_mass(i)` and nothing else, `logpdf(y)` returning
the log mass at an atom point, so the derivatives of that mass to order six are
`logpdf_dk<K>` evaluated there.

**What the generic routes did before.** `generic_expectation.hpp` added
`sum_i pi_i h(c_i)` after its quadrature, so expectations and the expected
information built on them counted the atoms. The three other generic routes did
not mention atoms at all: `generic_cdf.hpp` integrated the density and nothing
else, so `F` at an atom point came out as the integral alone (zero, at an atom
sitting on the lower end of the support); `generic_quantile.hpp` looked for a
root of a function that has jumps, and its Newton step divided by
`exp(logpdf(y))`, which at an atom is a mass and not a density; and
`generic_start.hpp` summarized a sample with the observations on the atoms still
in it. Nothing was wrong in practice, because no type with atoms reached those
routes: `atom_mixture_distrib` over a continuous parent always has
`closed_routes`, so its cdf, its log-cdf derivatives, its quantile and its
starting value are closed forms of its own; over a discrete parent it declares
no atom at all, the mass being part of the pmf; and `truncated_distrib` and
`transformed_distrib` own their cumulative routes and read the atoms
explicitly. That is a set of coincidences and not a design, so the generic
routes were made right instead (Giovanni, 2026-11-17, who chose this over a
`static_assert` forbidding the composition).

**What they do now.** The whole addition sits behind
`if constexpr (distrib_n_atoms<D>() > 0)`, so a family without atoms compiles to
what it compiled to before.

*The cdf.* `F(y)` is the integral of the density over the support up to `y` plus
the masses at or below `y` (F is right-continuous, so an atom at `y` counts),
and the complement is the integral above `y` plus the masses strictly above it.
The sum is a log-sum-exp against the log integral, so a small probability keeps
its accuracy. The side of `y` that does not hold `center()` is the one
integrated, as before; the other is now the complement against the mass the law
spreads over Lebesgue measure, `C = 1 - sum_i pi_i`, and no longer against one.

*The derivatives.* The side asked for is `G = S + A`, the integral over it plus
the atoms on it, and the two are combined in ratio space: with `w_j` the weight
of a term in `G`, `d_T G / G = sum_j w_j d_T X_j / X_j`, and the derivatives of
`log G` follow from the Bell relation solved for them (`log_sum_ratios` in
`generic_cdf.hpp`, the arithmetic of `detail_truncated::combine_order` written
one component at a time, which is how that file is indexed). The derivatives of
an atom's log mass are `logpdf_dk<K>` at its point, as the descriptor says, and
`C` carries derivatives of its own, so the complement of the integrated side is
`C - I` and not `1 - I`.

*The quantile.* With atoms `F` has jumps and the quantile is
`inf{y : F(y) >= p}`, which is the atom point `c_i` whenever `p` lies in
`(F(c_i-), F(c_i)]`; there no root exists. Both ends of the jump come from the
cumulative function at `c_i` itself, which already counts the atoms at or below
it, so the atom points need not be sorted and nothing is asked of a family
beyond the three descriptors. Outside a jump the root-finder works as it did:
`d/dy log F(y) = f(y)/F(y)` still holds, the mass at or below `y` being locally
constant, so no mass has to be subtracted. At an atom point the density is
taken as zero, which sends the step to the bisection fallback; the final Newton
step on the linear scale is skipped there as well.

*The starting value.* The observations sitting on an atom point are dropped
before the summaries, so the moments and the quartiles describe the part of the
law spread over Lebesgue measure: a gamma with a third of its mass at zero would
otherwise start from a mean two thirds of its own. What the mass of each atom
starts at is not lifted and cannot be: an atom declares its point and its mass
and nothing that says which parameter carries it, so a family with atom
parameters supplies its own `start_value` over the sample, as
`atom_mixture_distrib` does.

**The quadrature never reads a mass as a density.** `logpdf(y)` is one log
density with respect to Lebesgue measure plus a counting measure on the atoms,
so a quadrature node landing exactly on an atom point would read a mass where a
density belongs, and an atom usually sits at the lower end of the support or at
a panel boundary. `core.hpp` states the rule that stops it: a declared atom is
also reported by `y_kinks()`, which the panel routines split on. Measured here,
the rule is not sufficient by itself. `panels_quad` cuts the interval at the
points it is given, and each panel is integrated by `tanh_sinh_quad`, whose node
function returns the code 2 for a node whose `x` has rounded onto an end
although its distance is not zero, and `de_sum` evaluates the integrand at that
`x` all the same. A cut therefore makes the atom a panel end, which is exactly
where a node can land. An atom at the lower end of the support is not even a
cut, `panels_quad` keeping only the kinks strictly inside the interval.

That test is an exact equality, and it is sound on the device for a reason worth
stating, since an exact comparison is usually a CUDA failure waiting to happen.
It compares a STORED atom point with the point being evaluated, and it is the
same comparison that decides what `logpdf` means there (`atom_at` in
`atom_mixture.hpp` tests `y == point[i]`), so the guard and the switch of meaning
agree bit for bit on any backend: nothing is required to cancel, and no
intermediate has to come out exactly zero. Where a node merely lands NEAR an
atom the guard does not fire, which is right, the density being what belongs
there. The exact comparisons elsewhere in these routes are of the same harmless
kind or are opportunistic early exits (`gx == 0.0` in the root-finder, `t == 0.0`
in a tail sum), which cost an iteration and not an answer when the value misses
zero by an ulp.

Every quadrature of the density now tests the node instead:
`detail_generic::at_an_atom` (in `generic_expectation.hpp`, the lowest of these
headers, so that all of them see it) says whether a point is an atom point, and
the integrand returns zero there. That holds for the two of `generic_cdf.hpp`,
through `detail_generic::density_off_atoms`, and for the two continuous
integrands of `generic_expectation.hpp`, where a node on an atom would have
counted the mass a second time on top of the explicit
`sum_i pi_i h(c_i)`. None of it depends on where the nodes fall.

**The cut budget.** `panels_quad` keeps the centre plus at most 8 interior cuts
in a `double cut[10]`. The cuts are now the kinks of `log f` in `y`, at most 2
(`distrib_kinks::point` holds two), followed by the atom points, at most 6
(`core.hpp`), with duplicates dropped for a family that already reports its
atoms as kinks: 8 at most, exactly the rule's budget, and 9 of the 10 slots used
with the centre. The two-atom ceiling that `atom_mixture_distrib` imposes on a
continuous parent was there because `y_kinks()` was the only way to tell the
quadrature about an atom; the generic cdf no longer depends on that, so the
ceiling is now a property of `atom_mixture` alone and not of the quadrature. The
ceiling was not changed.

**Measured, gcc, host.** The generic routes against the closed forms they are
meant to reproduce, over the atom point itself, the floating-point neighbours on
both sides, the window and both tails.

| what | object | oracle | worst |
| --- | --- | --- | --- |
| `cdf` | gamma, fixed mass 0.3 at 0 | `pi + (1 - pi) F_gamma`, closed | 2 ulp |
| `ccdf` | the same | closed | 21 ulp |
| `log_cdf` | the same | closed | 3.9e-16 absolute |
| `quantile` | the same | closed inverse | 1.3e-11 relative |
| `cdf` | zero-adjusted gamma, mass a parameter | `atom_mixture_distrib::cdf` | 3 ulp |
| `ccdf` | the same | `atom_mixture_distrib::ccdf` | 16 ulp |
| `log_cdf` | the same | closed | 4.4e-16 absolute |
| `log_ccdf` | the same | closed | 5.7e-14 absolute |
| `log_cdf_dk<1>` | the same | `log_cdf_d1` | 1.2e-15 relative |
| `log_cdf_dk<2>` | the same | `log_cdf_d2` | 1.4e-15 relative |
| `log_cdf_dk<3>` | the same | `log_cdf_d3` | 2.7e-15 relative |
| `log_ccdf_dk<1>` | the same | `log_ccdf_d1` | 8.9e-16 relative |
| `log_ccdf_dk<2>` | the same | `log_ccdf_d2` | 5.7e-16 relative |
| `quantile` | the same | `atom_mixture_distrib::quantile` | 1.1e-13 relative |
| `cdf` | zero-adjusted gamma truncated to [0, 3] | `truncated_distrib::cdf` | 1 ulp |
| `ccdf` | the same | `truncated_distrib::ccdf` | 7 ulp |

The relative figures are against the largest component of the tensor. The ulp
measure is not reported for the logarithms: `log F` passes through zero in the
far upper tail, where an ulp of it is a denormal and says nothing; the absolute
error is the figure that means something there.

One frozen figure, kept so that a dropped mass fails loudly: a gamma with shape
2 and rate 1 carrying mass 0.3 at zero, truncated to [0, 3], has
`F_gamma(3) = 1 - e^-3 (1 + 3) = 0.80085172652854423`, hence
`Z = 0.3 + 0.7 F_gamma(3) = 0.86059620856998096` and a cdf at zero of
`0.3 / Z = 0.34859553994375395` (mpmath at 40 digits). An atom-blind route
returns zero there.

**A defect the atoms uncovered, repaired.** When the integral over the
integrated side came out empty, `log_cdf_side_by_integral` returned minus
infinity whichever side had been asked for. The side integrated is empty there,
but the other one holds the whole of the continuous mass, so `log(1 - F)` just
above a finite lower end came out as minus infinity where it is `log C` (0 for a
law without atoms). A window of denormal width is where the quadrature reaches
that case: `log(1 - F)` of a gamma at the smallest positive double was minus
infinity. The empty integral is now carried into the complement instead of
short-circuiting it.

**The mixture's kinks in y, repaired.** `atom_mixture_distrib::y_kinks()` built a
fresh `distrib_kinks` from its own atom points and never consulted the parent,
so an atom mixture over `laplace1` or `enet`, each of which declares a real kink
of `log f` at `mu`, lost it. It now forwards the parent's kinks, `continuous_order`
included: the continuous part of the mixture is the parent's density times one
constant, so it has exactly the parent's kinks and the parent's regularity in
`y`, and an atom is not one of them, the integrands skipping the atom points
themselves and the panel routines being given those points separately.

One consequence outside this file, found by the full suite. `transformed()`
refuses a law whose non-smooth points in `y`, the parent's carried through the
map plus the map's own, number more than two, a `distrib_kinks` declaring at
most two. An atom mixture's atoms used to count towards that number, so a
zero-one-adjusted parent through Yeo-Johnson made three and was refused; with
the atoms no longer reported as kinks the count is one and the law is accepted,
which is right, an atom being no kink of the continuous part. The refusal still
fires on three real kinks, and the case in `tests/distrib/test_transformed.cpp`
that exercised it was rebuilt on three of those.

What the repair is worth, measured. On `laplace1` nothing: the generic cdf
against the mixture's own closed cdf is 2.9e-16 relative at worst either way,
because `panels_quad` always cuts at `center()` and both families that declare a
kink today put it at `mu` with `center()` at `mu` as well, so the dropped kink
was a cut in any case. The cost appears as soon as a parent's kink is not at its
centre, which `center()` being a rough location makes legitimate: with the
centre moved 2.3 off `mu`, the old behaviour returns NaN at 2 of 10 points
across the window, the quadrature not converging across an uncut kink, where the
repaired one gives 1.2e-16. So the defect was latent and would have appeared on
the first family or wrapper whose kink sits away from its centre.

**The cut budget, and the two-atom limit.** `density_cuts` writes the parent's
kinks first and then the atom points, at most 8 in all, which is what
`panels_quad` keeps (the centre plus 8 interior cuts, in a `double cut[10]`).
A `distrib_kinks` holds two points and no more, so the kinks can never be more
than 2, and the room left for atom points is therefore 6 whatever the parent is
— exactly the six that `core.hpp` allows an atom declaration. The
`static_assert` in `atom_mixture.hpp` that limits a continuous parent to two
atoms gave as its reason "which is what `y_kinks()` can report"; that reason is
gone, the atom points no longer travelling through `y_kinks()`. The limit and
its message were left as they are, the change being a convention and not a
repair. Should the limit be raised, 6 is the bound, and overflowing the 8 cuts
would cost accuracy and not correctness: an atom point dropped from the cuts is
still skipped by the integrand.

**The continuous-parent atom limit raised from two to six** (Giovanni,
2026-11-17), the same number a discrete parent already took, bounded now only by
the `n_atom_points >= 1 && n_atom_points <= 6` assert that both kinds share. The
reason for two had been that a declared atom must be a panel boundary and
`y_kinks()` carries two points, which no longer holds.

Measured rather than assumed, by counting what `detail_generic::density_cuts`
writes and what `panels_quad` keeps:

| what | measured |
| --- | --- |
| `distrib_kinks::point` capacity | 2, and it is a fixed array |
| `density_cuts`, 6 atoms and 2 kinks | 8 cuts, the kinks first |
| `density_cuts`, 6 atoms and no kink | 6 cuts |
| `panels_quad` with those 8 cuts and a centre inside | converged, and the integral of e^-x over (0, infinity) came to 1 within 1 ulp, 1534 evaluations |
| `panels_quad` offered 9 cuts | reads 8 of them, still converged |

So six fits exactly: `panels_quad` reads at most 8 interior cuts, and uses 9 of
its `cut[10]` and 11 of its `ends[12]` once the centre is added. A cut dropped
for want of room would cost accuracy and not correctness, the integrand skipping
the atom points whether or not they are cuts. Six is therefore the largest sound
value and the limit was set there, not capped lower.

The other ceilings, checked on the way. `detail_atom_mixture::alr_names` holds
exactly six names, `atom_alr1` to `atom_alr6`, and `params_of<D, N>` and the
`cnt[6]` of `start_value` take six. `fill_log_renorm`'s scratch
`flat_tensors<Q, K> pv[6]` is dimensioned on the PARENT's parameter count, and is
instantiated only for a discrete parent, so six atoms do not grow it. What six
atoms do cost is the parameter count, and with it the tensors: a parent with P
parameters gives P + 6, and the order-6 tensor has binomial(P + 11, 6)
components, which is 924 at P = 1, 1716 at P = 2, 3003 at P = 3 and 5005 at
P = 4. That is the user's to pay.

**The stack the generic derivative route needs, and why neither obvious repair
closes it** (measured on 2026-11-17, after an earlier entry here put the figure
at 144 kB, which was an underestimate: it counted the atoms' partials and left
out the route's own). The frame of `log_cdf_side_dk_by_integral` at order K holds
`log_density_partials<P, K>` for the density at y, the `A[]` table of integral
ratios of the same size, and one more set of partials per atom, so

    bytes = (2 + n_atoms) * sizeof(log_density_partials<P, K>)

with P the number of parameters. At order six, in bytes:

| P | no atoms | 1 atom | 6 atoms |
| --- | --- | --- | --- |
| 2 | 440 | 656 | 1736 |
| 3 | 1336 | 2000 | 5320 |
| 4 | 3352 | 5024 | 13384 |
| 5 | 7384 | 11072 | 29512 |
| 6 | 14776 | 22160 | 59080 |
| 8 | 48040 | 72056 | 192136 |
| 9 | 80072 | 120104 | 320264 |

The tests set the CUDA per-thread stack to 8192 bytes, measured for a
three-parameter family (`tests/main.cpp` records that 3072 fails and 4096 works).
So 8192 covers P up to 4 with at most one atom, and anything above needs a raised
`cudaLimitStackSize` from the host. Two things follow that are easy to get wrong.

The limit is NOT a property of the atoms. The route is already over 8192 at
P >= 6 with no atom at all, so what decides the frame is the parameter count at
order six; the atoms only move the crossing earlier, and P = 8 arises at all only
because a two-parameter parent may now carry six of them.

And neither repair that suggests itself works. Streaming the atoms, refilling one
buffer instead of keeping one per atom, takes the six-atom P = 8 case from 192 kB
to 72 kB, which is still nine times over the 8192 bytes, while replacing 36
derivative-tensor evaluations by 61776, a factor of 1716: it costs a great deal
exactly where it fixes nothing. A `static_assert` on a fixed budget would refuse
code that is perfectly sound on a host, where 192 kB is unremarkable, and would
still say nothing about the no-atom P >= 6 case. So the formula above is
documented and the code is left as it is (Giovanni, 2026-11-17).

The case is not reached by an atom mixture over a continuous parent, which has
its own closed cumulative routes; it is reached by a law that declares atoms and
owns none.

Tested: `tests/distrib/test_generic_atoms.cpp`, "more than two atoms over a
continuous parent". Three atoms over `gamma1` (five parameters): the three
masses come out at 0.1 each, the masses plus the quadrature of the density make
one within 1e-10, `density_cuts` writes the three atom points and nothing else,
the generic cdf matches the wrapper's closed one to 4.0e-16 relative at worst
over eleven points at and around every atom, and all five components of the
first derivatives of the log cdf match to 1e-10 of the tensor's scale. Six atoms
over `gamma1` (eight parameters): six cuts, total mass one, the generic cdf
against the closed one at eight points, and the six descriptor names. Host only
and at low order, for the stack reason above.

**On CUDA.** The same chained job built gcc and then CUDA from one snapshot:
`build-ga2` gives 667 cases and 1187615 assertions with nothing failing, and
`build-cuda-ga2` gives 667 cases and 1187648 assertions (the device adds a few
of its own) with 3 cases failing, all of them in `test_box_cox.cpp` and
`test_yeo_johnson.cpp` and none in the atom work. Execution space Cuda, per
thread stack 8192 bytes.

The atom figures are the same on both, to the digits that matter:

| what | gcc | CUDA |
| --- | --- | --- |
| fixed mass, `cdf` | 2 ulp | 2 ulp |
| fixed mass, `ccdf` | 21 ulp | 21 ulp |
| fixed mass, `log_cdf` | 3.9e-16 abs | 3.9e-16 abs |
| fixed mass, `quantile` | 1.28e-11 rel | 1.28e-11 rel |
| atom mixture, `cdf` | 3 ulp | 3 ulp |
| atom mixture, `ccdf` | 16 ulp | 16 ulp |
| atom mixture, `log_cdf` | 4.4e-16 abs | 4.4e-16 abs |
| atom mixture, `log_ccdf` | 5.7e-14 abs | 5.7e-14 abs |
| atom mixture, `log_cdf_dk<1>` | 1.2e-15 rel | 3.7e-15 rel |
| atom mixture, `log_cdf_dk<2>` | 1.4e-15 rel | 1.6e-15 rel |
| atom mixture, `log_cdf_dk<3>` | 2.7e-15 rel | 3.1e-15 rel |
| atom mixture, `quantile` | 1.06e-13 rel | 1.06e-13 rel |
| truncated, `cdf` | 1 ulp | 2 ulp |
| truncated, `ccdf` | 7 ulp | 7 ulp |
| laplace1 parent, `cdf` | 2.9e-16 rel | 2.9e-16 rel |
| three atoms, `cdf` | 4.0e-16 rel | 4.0e-16 rel |

The derivative rows are the only ones that move, by a factor of three at order
one and less above it, which is the FMA contraction of the device against the
host; the bounds of the tests were set from the larger of the two and hold on
both builds.

## The yeo_johnson bijector: the parameter partials at the ends (2026-11-17)

**What was wrong.** The repair above left the parameter partials alone, and they were NaN at
the ends of both domains. Measured over lambda in {-3/2, -1/2, 0, 1/2, 1, 3/2, 2, 5/2}, at
every end, for the total orders 1 to 6 and every entry c[n1] of `bijector_partials<K>`, which
is the mixed derivative in n0 = K - n1 arguments and n1 exponents:

| quantity | where | before | after |
|---|---|---|---|
| h | an infinite end of the domain of h | NaN, every K and every n1 | the limits below |
| log of h' | an infinite end | NaN at n1 = K, and every entry at lambda = 0 above | the limits below |
| log of h' | the finite end of the image, where there is one | NaN in the mixed entries | an infinity of sign (-1)^n1 above, (-1)^n0 below |
| g | v = +-inf | NaN wherever the limit is zero or finite | the limits below |
| log of g' | v = +-inf | already the limit | unchanged |
| h, log of h' at n1 = 0 or the extreme entries | the finite end | already the limit | unchanged |

**The limits.** A lambda-derivative multiplies a closed form by a logarithm of the argument,
which loses to a negative power of it and wins against a vanishing exponent, so the sign of the
exponent decides as it does for the pure derivatives: above zero an infinity, below zero zero,
and at zero the prefactor for n1 = 0 and an infinity above it. Four cases carry their own form:

- g at an infinite v with n0 = 0 reaches an end of the image, so its pure lambda-derivatives
  there are those of -1/lambda below lambda = 0 and of 1/(2 - lambda) above lambda = 2, namely
  (-1)^(n1+1) n1!/lambda^(n1+1) and n1!/(2 - lambda)^(n1+1).
- log of h' above is Psi = (1 - lambda) log(1 + lambda u)/lambda. For lambda > 0 a u-derivative
  carries (1 + lambda u)^(-n0) and dies, while a pure lambda-derivative multiplies the logarithm
  by the n1-th lambda-derivative of (1 - lambda)/lambda, of sign (-1)^n1, and runs to an
  infinity of that sign. At lambda = 0, Psi is the series (1 - lambda) sum_j (-1)^j u^(j+1)
  lambda^j/(j + 1), so its n1-th lambda-derivative is a polynomial in u of degree n1 + 1 whose
  leading coefficient is (-1)^n1 n1!/(n1 + 1): the n0-th u-derivative of that runs to
  (-1)^n1 infinity below n1 + 1, is the constant (-1)^n1 (n1!)^2 at n0 = n1 + 1, and vanishes
  above. The same holds at lambda = 2 on the negative branch, with (-1)^n0 for the sign.
- The inverse's negative branch is its positive branch reflected, h(u) = -h_pos(-u; c) and
  log of h'(u) = log of h'_pos(-u; c) with c = 2 - lambda, so the mixed derivative carries
  (-1)^(n0+n1+1) for h and (-1)^(n0+n1) for its log-Jacobian, with c in place of lambda. Every
  limit of the negative branch is that of the positive one read this way.
- At the finite end of the domain of h, u = -1/lambda below lambda = 0 and u = 1/(2 - lambda)
  above lambda = 2, the quantity w = 1 + lambda u (1 - c u below) vanishes and
  d^n0_u log|h'| = (1 - lambda)(-1)^(n0-1)(n0-1)! lambda^(n0-1) w^(-n0); each lambda-derivative
  raises the power of 1/w by one and brings a factor -n0 u, so the most singular term is
  C u^n1 w^(-n0-n1) and the limit is an infinity whose sign, with u at the end itself, is that
  of (1 - lambda)(-1)^(n0-1) lambda^(n0-1-n1), namely (-1)^n1 above and (-1)^n0 below.

Each of these limits was compared with SymPy's limit of the mixed derivative of the definition
(`tools/reference/bijector_yeo_johnson_forms.py`, `check_partial_limits` for the infinite ends
and `check_edge_limits` for the finite one), which also checks that the entry at n1 = 0 is the
pure limit the functions above already return.

**The test** is the case "the parameter partials at the ends of both domains" in
`tests/bijector/test_yeo_johnson.cpp`, in a kernel on the default execution space: for each
lambda the fifty-four entries at each of the four ends, with no NaN, the entry at n1 = 0 equal
to the pure derivative of the same order, the limit checked against the function's own approach
to it, and the written-out values of the four special forms above.

Measured: 6 cases green on both builds, 11188 assertions on gcc and 11204 on CUDA, with the
whole suite green on both. The two counts differ by design
and not by a skipped check: the consistency test branches on the value it reads -- a limit
already attained, a plateau, or an approach -- and which branch a quantity takes differs between
the platforms wherever FMA contraction moves the last bits, each branch asserting a different
number of things. Every assertion taken passes on both.

**Found on the way and not repaired.** The partials themselves return NaN at a large finite
argument: at 1e60, 28 of the 448 order-six entries measured are NaN, while 1e40 and below are
clean. That is a loss inside the forms at a finite point and not a missing limit at an end, so
the test compares a limit with the values at 1e20 and 1e40.


**The finite end of the inverse domain, and what a test there has to assert.** The map's own
parameter partials reached that end through the forms while the pure quantities and the
log-Jacobian's partials were guarded, which another session noticed from the call sites and
measured before reporting: they are not NaN there, but "not NaN" is not "the right limit", and
the unguarded route is the one FMA turns into 1e100. Their limits are infinite throughout --
h^(n0) = P_n0 w^(1/lambda - n0) with the exponent 1/lambda - K of the total order alone, which
is strictly negative in both regimes where yeo_johnson has a finite end (below lambda = 0 and,
through the reflection, above lambda = 2), so it never vanishes and no column is finite there,
the sign being (-1)^n1 above and (-1)^(n0+1) below. `yj_inv_partials_at_edge` now returns them,
and `check_edge_limits` covers the map as well as its log-Jacobian: 216 limits against SymPy,
twice what it checked before.

The test asserts those entries as exact infinities, and the reason is worth stating as a rule,
because two sessions left the two halves of it to a weaker check an hour apart. A check for
"not NaN", for "large" or for "finite or infinite" passes whether the guard fires or not, so it
cannot tell a working guard from a missing one; only the exact comparison can. And the exact
comparison can do it only on the device: remove the guard and the host's forms return the very
same infinity, the cancellation being exact there, so the assertion stays green on gcc either
way. A host run therefore says nothing at all about whether a guard at a finite end is
exercised.

## The yeo_johnson bijector: the parameter partials at a large finite argument (2026-11-17)

**What was wrong.** The two repairs above left the parameter partials of the inverse returning
NaN at a large finite argument, which is not an end of anything: the map, its derivatives in the
argument and its log-Jacobian are all finite and ordinary there. Measured at lambda = 1/2 and
x = 1e60, the sixth derivative in lambda came back NaN for a value whose size is 7.4e135 (mpmath
at 60 digits). At x = 1e60 one entry in seven failed, the one with no derivative in the argument;
at x = 1e300, six of seven.

**The cause.** Each generated form is a polynomial in the derivatives of `log1prel` multiplied by
a power of the argument, and the two factors move in opposite directions: psi^(j)(z) decays like
j! log(1 + z)/z^(j+1) and underflows, while z^(j+1) overflows. At z = 5e59 the sixth derivative
of `log1prel` is exactly zero and x^7 is infinite, so the form computes inf times 0.

**Two routes that did not work**, both of which only moved the powers around:

| route | what happened |
|---|---|
| substitute psi^(j)(z) = q_j/z^(j+1) and let SymPy cancel | NaN appeared at 1e40 and at lambda = 0, the cancellation leaving a division by z |
| the same substitution per additive monomial, two emitted paths | the cancellation succeeded and the header grew from 1960 to 3263 lines, but the NaN count rose from 151 to 171: a monomial that keeps a positive power of the argument still overflows |

**What works** is to compute the derivative from its structure, with the dominant scale factored
out, so that every intermediate is representable and the only large number is one exponential at
the end. Writing S = log(1 + lambda u)/lambda, the n0-th derivative of the inverse in its
argument is C_n0(lambda) exp(A_n0) with A_n0 = (1 - n0 lambda) S and C_n0 the product of
(1 - k lambda) for k = 1 to n0 - 1, so

    d^n1_lambda [C exp(A)] = exp(A) sum_i binom(n1, i) C^(i) Y_(n1-i)(A', ..., A^(n1-i))

with Y the complete Bell polynomials and A^(j) = (1 - n0 lambda) S^(j) - n0 j S^(j-1). The
derivatives of S are the ones that have to be found without forming a power, and they are, by

    S^(j) = q_j/lambda^(j+1),    q_j = z^(j+1) psi^(j)(z) = (-1)^j j! [log(1 + z) - sum_(k=1..j) r^k/k]

with r = z/(1 + z) and z = lambda u. That closed form is proved equal to SymPy's own derivative
for j = 0 to 7 by `check_scaled_log1prel` in
`tools/reference/bijector_yeo_johnson_forms.py`; it grows only as a logarithm. At lambda = 1/2
and u = 1e60 the chain runs q_6 about 1e5, S^(6) about 1e7, Y_6 about 1e16 and exp(A) about
1e119, for a product of 7.36e135.

The negative branch is the positive one reflected, h(u) = -h_pos(-u; 2 - lambda), whose entries
pick up (-1)^(n0+n1+1); since n0 + n1 = K the sign is (-1)^(K+1) and does not depend on the
entry. Of the log-Jacobian's partials only the pure derivative in lambda needs this route, every
entry with a derivative in the argument carrying a power of 1 + lambda u instead of S and staying
bounded, and for that one entry the whole expression is

    d^K_lambda [(1 - lambda) S] = (1 - lambda) S^(K) - K S^(K-1).

**Where it is taken.** Above |lambda u| of 1e3, in `yeo_johnson_inv_partials` for the map and in
`yeo_johnson_inv_lj_pos_partials` and the negative branch of `yeo_johnson_inv_lj_partials` for
the log-Jacobian. Below the threshold the forms are exact and this route would cancel: q_j is
there a difference of two nearly equal quantities, and at z = 1/8 it loses six digits. The two
agree over the whole overlap, measured for every total order 1 to 6, every entry, both branches
and lambda in {-2/5, 1/4, 1/2, 1, 13/10, 7/4, 13/5}:

| |z| | worst relative difference between the two routes |
|---|---|
| 1/8 to 1e8 (the handover region and beyond) | 1.5e-10, and that worst case is at z = 1/8, which the guard never sends |
| 1e-3 and below (not reachable through the guard) | up to 0.16 at the top order, the cancellation in q_j |

**Against the references.** The pure derivatives in lambda of orders 1 to 6, at four places with
|z| above the threshold, against the definition of the inverse differentiated symbolically by
SymPy and evaluated at 60 digits by mpmath
(`tools/reference/bijector_yeo_johnson_large.py`):

| quantity | worst relative difference |
|---|---|
| the inverse | 8.4e-14, at lambda = 13/10 and x = 1e300 |
| its log-Jacobian | 5.7e-16, one unit in the last place |

The inverse's error is not a property of the route. Its value is exp(S), so one unit in the last
place of S moves it by a relative |S|, which is 276 at lambda = 1/2 and x = 1e60, and 532 at lambda = 13/10 and x = 1e300; the
measured 8.4e-14 is that amplification and no route computing the value in double can be better.
The entry with no derivative in the argument is exp(S) itself, so wherever it is finite S is at
most the logarithm of the largest double, about 709, and the bound 4e-13 in the test holds
everywhere the value is a number. The log-Jacobian is S times a factor and
carries no amplification, which is why its bound is 1e-14.

Reference values no numerical differentiator can supply: the first attempt at them took
mpmath's own `diff` in the argument at u = 1e60, and it returned entries wrong by two hundred
orders of magnitude, several of them exactly zero. The references are differentiated
symbolically for that reason, and the script says so.

**What it cost.** The header grew from 1960 to 2131 lines, the two routines and their comment
being hand-written in the generator's preamble rather than generated per order; the per-monomial
route had reached 3263 lines while leaving more NaN than it removed.

**What is open.** Where the true value overflows a double the entry is an infinity, which is
correct: at lambda = 1/2 and x = 1e300 the sixth derivative is 1.1e620. The entries that carry a
derivative in the argument underflow to zero at these arguments, by a factor (1 + lambda u)^(-1)
per order, so there is nothing to check against in them and the test only requires them to be
numbers. `box_cox` has the same loss in its own map's partials, and its log-Jacobian's partials
go through `yeo_johnson_inv_lj_pos_partials` and are repaired by this change.

**A second band, on the other side of the loss, repaired by a third route.** The guard above
starts at |lambda x| = 1e3 and cannot reach the other side: where lambda is so small that
lambda x stays small even at a huge x, the forms carry a power of the argument in front of a
bracket in the psi^(j)(z) which cancels as z goes to zero, so the power is infinite, the bracket
is zero, and the product is NaN again. Found by the box_cox session measuring the repair above
against its own grid, where it appeared as a hole at lambda = 0. It is a band and not a point:
measured identically at lambda = 1e-310 and 1e-305 with x = 1e300, while at lambda = 1e-200 and
above nothing fails, lambda x reaching 1e3 there before a power of x can overflow. The first
decade of |x| at which an entry fails, by order, measured on the log-Jacobian's partials:

| lambda | K = 1 to 3 | K = 4 | K = 5 | K = 6 |
|---|---|---|---|---|
| 0, 1e-310, 1e-300, 1e-200 | none | 1e154 | 1e154 | 1e103 |
| 1e-100 and above | none | none | none | none |

What the entries lost there is mostly FINITE, which is why neither obvious repair works. At
lambda = 0 and x = 1e300, order six returned NaN where the exact value is 0 at n1 = 2 and
-3.6e301 at n1 = 3, and order five where it is 4 at n1 = 2; only two of the entries found are
genuinely infinite. So signed infinities would have been wrong, and the rescaled route above
cannot serve either, its q_j being a cancelling difference at small z.

**The route.** The power of the argument in each term does not depend on the summation index.
With S = u psi(lambda u), psi = log1prel and z = lambda u, Leibniz in u gives

    d^n0_u S^(j) = sum_m C(n0, m) [d^(n0-m)_u u^(j+1)] lambda^m psi^(j+m)(z)

and u^(j+1-n0+m) lambda^m = u^(j+1-n0) z^m, one power for the whole sum:

    d^n0_u S^(j) = u^(j+1-n0) sum_(m=m0..n0) C(n0,m) (j+1)!/(j+1-n0+m)! z^m psi^(j+m)(z)

with m0 = max(0, n0 - j - 1), which keeps every factorial's argument non-negative. The sum is
bounded at small z, each term being a small power of z times a derivative of psi, so it is taken
first and the single power applied once at the end; a sum that is exactly zero returns zero
without that multiplication, since at z = 0 it is zero exactly when the m = 0 term is absent and
a power of u that has overflowed would otherwise turn an exact zero into NaN. The identity is
checked against SymPy's own derivative for every (j, n0) with j + n0 <= 6 by `check_dS_small_z`
in `tools/reference/bijector_yeo_johnson_forms.py`. The log-Jacobian's entries follow from
log|h'| = (1 - lambda) S as (1 - lambda) d^n0_u S^(n1) - n1 d^n0_u S^(n1-1), and the negative
branch is the positive one at (2 - lambda, -u) with every entry times (-1)^K.

**Where it is taken, and what it measures.** Above |x| = 1e50 with |lambda x| at most 1e3, fifty
decades below the measured onset and far above any argument a model produces. Below that the
forms are kept, and over the overlap the two agree to 2.3e-16 relative (lambda from 0 to 1.9,
|x| from 1 to 1e140, every order, every entry; the one larger figure, 1.3e-10, is at an entry of
size 9.6e-299, where a relative measure is meaningless). Against the references, which are
SymPy's own limit of the definition as lambda goes to zero and so a third route independent both
of the forms and of the identity:

| what | result |
|---|---|
| every finite entry, orders 1 to 6, at x = 1e103, 1e150, 1e200, 1e300 | exact to 1.6e-16 |
| every entry that is zero or an infinity | exact, including the sign |
| the whole grid afterwards, lambda from 0 to 2 and |x| from 1 to 1e300, both branches | 0 NaN in the log-Jacobian's partials, 312 cells |

The map needs nothing in this band: at lambda = 0 its inverse is exp(u) - 1, already an infinity
for every |u| above about 710, so there is no finite value of its own to lose (0 of 7 NaN
measured at every place in the band). `box_cox` is repaired with it, the route being called from
`yeo_johnson_inv_lj_pos_partials`, which holds wherever 1 + lambda u > 0.

**What is open after it, measured on a dense grid.** The map's partials still return NaN where
the value itself overflows: 370 entries over 183 of 1147 cells, for lambda from 1e-30 down to
1e-310 together with |x| from 1e50 up. Inside the rescaled route the intermediates there run
past the largest double -- at lambda = 1e-30 and x = 1e60 the first derivative of the exponent
is about 1e210, so its sixth Bell polynomial would be 1e1260 -- and an inf minus inf or inf
times 0 appears. **Nothing finite is lost in it**: for each of those 370 entries the exponential
that carries the value, exp((1 - n0 lambda) S), was checked and is infinite in every one, so the
true derivative is a determinate infinity and what is returned is NaN in place of an infinity of
a known sign. A first, coarser grid put this regime at lambda of 1e-100 or below with |x| of
1e125 or above; that was an artifact of its spacing, which stepped lambda from 1e-20 straight to
1e-100 and |x| by 25 decades at a time, and it also excluded lambda = 0 on the positive branch,
which is in the inverse's domain there. The figures here replace it. With lambda = 0 included,
it turns out to lose nothing at all in this map: the forms take z = 0 cleanly and return
infinities, so the regime begins only at a lambda small enough to overflow the route's own
chain. `box_cox` shares this regime, reaching it from lambda = 1e-310 to 1e-50 by its own
measurement, and has a second one of its own at lambda = 0 exactly, where |lambda u| is zero so
the rescaled route is never taken and its forms lose entries from u = 1e40; that one is not this
defect and nothing here is involved in it. Closing this regime means the signs of those
infinities, which is a fifth piece of work and not this one.

## The box_cox bijector at the ends of its two domains (2026-11-17)

**What was wrong.** `scalar_bijector::box_cox` returned NaN at an end of either
of its domains, where the limit exists and is finite or an infinity of a
definite sign, although the header stated in its opening comment that a value
on a bound gives its limit. The hole was the one repaired the same day in
`yeo_johnson` (the section above) and wider. At an infinite argument the cause
is the one the closed forms of
`include/laplace/bijector/detail/scalar_box_cox.hpp` were written for: each of
them passes through `exprel` or `log1prel` so that lambda = 0 needs no branch of
its own, and at an infinite argument those two primitives are themselves
evaluated at an infinite argument, where each is inf/inf. At a finite end the
cause is different and the point is finite: a prefactor that vanishes meets a
power that is infinite, which is 0 times inf. The defect was found by the
session that repaired `yeo_johnson`, which measured box_cox alongside it.

**The ends.** g has domain (0, inf), so they are v = 0 and v = +inf whatever
lambda is. The domain of h is the image of g: (-1/lambda, inf) for lambda > 0,
so they are u = -1/lambda and u = +inf; (-inf, -1/lambda) for lambda < 0, so
u = -inf and u = -1/lambda; the whole line for lambda = 0, which has no finite
end. Measured on the host, gcc, over lambda in {-3/2, -1, -1/2, -1/3, -1/5, 0,
1/6, 1/5, 1/4, 1/3, 1/2, 2/3, 7/10, 1, 3/2, 2, 5/2, 3, 4, 5, 6}, which covers
every regime (below 0, 0, in (0, 1), 1, above 1) and both families of knife
edges, lambda = 1/n and lambda = n for n = 1 to 6:

| quantity | where | before | after |
|---|---|---|---|
| h, and h' to h^(6) | u = +inf, an end for lambda >= 0 | NaN | the limits below |
| h, and h' to h^(6) | u = -inf, an end for lambda <= 0 | NaN | 0 |
| log of h' | an infinite end | NaN | an infinity, or 0 at lambda = 1 |
| its 1st to 6th derivatives | an infinite end at lambda = 0 | NaN | 1 at order 1, 0 above |
| its 1st to 6th derivatives | any other infinite end | 0 | 0 |
| h^(n) | u = -1/lambda at lambda = 1/n | NaN | (n-1)!/n^(n-1) |
| h^(n) | u = -1/lambda at lambda = 1/r, r < n | NaN | 0 |
| log of h', with every derivative | u = -1/lambda at lambda = 1 | NaN | 0 |
| h and the rest | u = -1/lambda otherwise | the limit | unchanged |
| g | v = 0 and v = +inf | the limit | unchanged |
| g^(m) | v = 0, lambda an integer below m | NaN | 0 |
| 1st to 6th derivatives of log of g' | v = 0 at lambda = 1 | NaN | 0 |
| g^(m), m >= 1 | v = 0 or v = +inf otherwise | the limit | unchanged |
| every `*_partials` component | every end | NaN | unchanged |

**The limits.** The argument enters every closed form only through a power of v
or of w = 1 + lambda u, so the sign of that power's exponent decides each limit:
where the base runs to +inf the power runs to +inf for a positive exponent and
dies for a negative one, where the base goes to 0 from above the two cases
exchange, and on the knife edge where the exponent vanishes the power is 1 and
the prefactor is the whole limit.

For the forward map, g^(m) = F_m(lambda) v^(lambda - m) with
F_m = prod_(k=1..m-1)(lambda - k), and the exponent vanishes exactly at the
integer lambda = m, where F_m is (m-1)!. Where F_m itself vanishes, that is at
an integer lambda below m, g is a polynomial of degree lambda and g^(m) is
identically 0. The value g(0) and g(+inf) are the two ends of the image of g,
-1/lambda where that end is finite and an infinity otherwise, which is what a
caller reads the support of a transformed law from. The log-Jacobian is
(lambda - 1) log v, whose m-th derivative is
(lambda - 1)(-1)^(m-1)(m-1)! v^(-m): the value runs to an infinity of its own
sign, each derivative to one at v = 0 and to 0 at v = +inf, and at lambda = 1
the map is the identity and the whole log-Jacobian is 0.

For the inverse, h^(K) = P_K(lambda) w^(1/lambda - K) with
P_K = prod_(r=1..K-1)(1 - r lambda), and 1 - K lambda carries the sign of the
exponent for lambda >= 0. At u = +inf, an end only for lambda >= 0, w runs to
+inf, so the order is +inf below the edge, P_K on it and 0 above it; lambda = 0,
where h = e^u, falls under the first case with K lambda = 0 and P_K = 1. At
u = -inf, an end only for lambda <= 0, the exponent is negative and h dies with
every order of it. At the finite end u = -1/lambda, w goes to 0 from above and
the same exponent decides the other way round: above 0 the order is 0 below the
edge, P_K on it, and an infinity of the sign of P_K above it, 0 where P_K itself
vanishes, since h is then a polynomial of degree 1/lambda; below 0 the exponent
1/lambda - K is negative whatever K is and P_K is a product of factors above 1,
so every order runs to +inf, h included, the range of h being (0, inf). On the
edge lambda = 1/n the limit of h^(n) is P_n = (n-1)!/n^(n-1), which is 1, 1/2,
2/9, 3/32, 24/625 and 120/7776 for n = 1 to 6. The log-Jacobian is
(1 - lambda) S with S = log(1 + lambda u)/lambda, which is u itself at
lambda = 0, and its K-th derivative is
(1 - lambda)(-1)^(K-1)(K-1)! lambda^(K-1) w^(-K): S runs to +inf at u = +inf and
to -inf at u = -inf, and at the finite end to -inf above 0 and to +inf below;
every derivative carries w^(-K) and dies at an infinite u, except at lambda = 0,
where the log-Jacobian is u and its first derivative is the slope 1, and at a
finite end each runs to an infinity of the sign of its own coefficient. At
lambda = 1 the map is the identity and every order is 0.

Each of the 1232 limits was compared with SymPy's own limit of the definition
(`tools/reference/bijector_box_cox_forms.py`, `check_limits`, the script that
also checks every closed form of the header against SymPy). The C++ never forms
the exponent: it vanishes where lambda is a small integer for the forward map
and where a product of lambda with the order equals 1 for the inverse, so the
two are compared rather than subtracted. A subtraction is what nvcc contracts
into an FMA, and the exact product of 3 with the double nearest 1/3 is
1 - 2^-54, which rounds to 1 in a plain multiplication and stays 1 - 2^-54 in an
FMA, so the edge would be taken on one platform and not on the other. Measured
here: the edge is reached at the double nearest 1/n for every n from 1 to 6,
box_cox comparing lambda itself and not a 2 - lambda as yeo_johnson's negative
branch does.

**Three NaN inside the domain, of the same kind, repaired with the ends.** The
test's own approach to each end found them; each is a vanishing prefactor
meeting a power that has overflowed, and in each case the quantity vanishes
identically where the prefactor does, so the power is now not formed at all.
`bc_fwd_any`: g^(m) at an integer lambda below m, where v^(lambda - m)
overflows for a small enough v (at lambda = 1 and m = 3, below v = 1e-103).
`bc_fwd_lj_any`: a v-derivative of log|g'| at lambda = 1, where v^(-m)
overflows (at m = 6, below v = 1e-52); the derivative in lambda is log v and
does not vanish there, so the guard is on the pure v-derivatives alone.
`bc_inv_mixed`: h^(K) where P_K vanishes, that is at lambda = 1/r with r below
K, the exponential exp((1 - K lambda) S) overflowing for a w small enough.

**A fourth defect found and repaired, not about the ends.** `bc_pow_shift`
computes v^(lambda - m) as v^lambda divided by v m times, so that the exponent
is not rounded (lambda - m is not a double, and log v magnifies its rounding).
Where v^lambda itself underflows or overflows the route carries no information
left: v^lambda is 0 or inf and so is every quotient, although v^(lambda - m) may
be an ordinary number. Measured: g^(6) at lambda = 6 is exactly 120 for every v,
and the route returned 0 at v = 1e-60 and 1e-120 and inf at v = 1e60, the
correct 120 only between about 1e-52 and 1e52; g' at lambda = 6 and v = 1e-60 is
1e-300 and came out 0. The rounded exponent is now used where the other route
has nothing left, at a cost of |log v| eps relative, 1.5e-14 at the smallest
double, which is what a result the division route cannot represent at all is
worth. The reference rows are unaffected: the fallback runs only where the
division route gives 0 or inf, and all seven orders keep the bounds measured on
2026-11-12.

**The test** is in `tests/bijector/test_box_cox.cpp`, in a kernel on the default
execution space: for each of the 21 values of lambda above, the fourteen
quantities (the map and its log-Jacobian, orders 0 to 6) at the four ends,
checked first against the function's own approach to the end and then against
the written-out values of the edges, of the signs of the infinities and of the
log-Jacobians. Every end of box_cox has such an approach, the finite ones
included: there the two arguments are the points where w = 1 + lambda u is 1e-30
and 1e-60, which lie inside the domain on whichever side of the end it is. An
infinite limit must grow past 1 in its own sign between the two arguments, a
zero limit fall below 1e-6, and a finite limit be reached to 1e-8, which is what
the approach is worth: it is only as good as v^lambda at the nearer argument,
1e-10 at lambda = 1/6 and v = 1e-60, the slowest of these lambda. One ulp from
a knife edge the limit would be right and out of reach, the exponent being about
1e-16 while the logarithm of the argument reaches 745 at most, so the quantity
sits on the plateau of the edge over the whole range of double; the test tells a
plateau from an approach by the two arguments giving the same value, as
yeo_johnson's does, and then checks only that the plateau is finite and carries
the sign of the limit.

**What was still NaN when the repair above was written, and no longer is.** The
parameter derivatives (the `*_partials` functions, and so `forward_partials_dK`
and the rest) were NaN at every end: at v = 0 and v = +inf a derivative in
lambda brings powers of log v that multiply a power of v which is 0 or inf, and
at an infinite u the Bell polynomials of the lambda-derivatives of S are built
from u^(j+1) log1prel^(j)(lambda u), each term inf times NaN. They were left so
on the reasoning recorded in the next section, which the day then overtook; the
section after it repairs them. `box_cox_inv_lj_partials` forwards to
`yeo_johnson_inv_lj_pos_partials`, which the two maps share, and that function
was not touched: box_cox's guard sits in `box_cox_inv_lj_partials`, before the
shared branch is reached.

**The partials at the ends: the measurement, and a decision that was overtaken
the same day.** Asked after the repair above, answered first on the state of
the sibling map and then again when that state changed; both readings are kept
here, because the measurement holds either way and only the conclusion moved.

What the measurement said, and still says. Nothing outside the bijector layer
and its own tests and benches reads a `*_partials` member today:
`transformed_distrib` holds the map's parameters as fixed values and not as
coordinates of theta, and reads the ends of its own support from the bound
functions and not from the map at an end. And most of the limits are not
numbers a caller could use. Classified with SymPy at the ends, over lambda in
{-3/2, -1/2, 1/2, 1, 2, 3} and to total order four: of the 84 limits, 48 are
infinite, 26 finite and 10 zero, so 58 per cent are infinite, and an infinity
propagates into a criterion exactly as a NaN does. (The run also covered
lambda = 0, whose 14 rows are an artefact of the script, which substitutes
lambda = 0 into the definition before differentiating, leaving an expression
without lambda; the true limits there are those of L^(n+1)/(n+1), L = log v,
hence infinite. Those rows are excluded from the counts above.)

Every one of those 26 is the same quantity: the pure lambda-derivatives of the
forward map at the end where the image is finite, which are the
lambda-derivatives of that very end,

    d^n/dlambda^n (-1/lambda) = (-1)^(n+1) n! / lambda^(n+1),

measured as 4, -16, 96 and -768 at lambda = 1/2 and v = 0, and as 4/9 at
lambda = -3/2 and v = +inf.

That run covered the forward map, one of its mixed partials and the inverse
map, and not the log-Jacobians, so it does not say that the bound's derivatives
are the only finite limits anywhere, and they are not. A second family was
pointed out by the yeo_johnson session and found here too, in the mixed limits
of log|h'| at lambda = 0, where log|h'| is affine in u: the limit is the
constant (-1)^n1 (n1!)^2 wherever the number of argument derivatives is one more
than the number in lambda, an infinity of sign (-1)^n1 below that and zero above
it. Read off the SymPy table at lambda = 0: 1 at (n0, n1) = (1, 0) and -1 at
(2, 1), at u = +inf and at u = -inf alike. Unlike yeo_johnson's, box_cox's two
infinite ends carry the same sign here, the map having one branch and not a
negative branch reflected from the positive one.

**The first conclusion, and why it fell.** On that measurement the partials were
left as they were, on the parity argument that yeo_johnson left its own NaN at
an infinite end. That ceased to be true within the hour: the yeo_johnson session
went on to repair all four of its partials families, adding
`yj_fwd_partials_at_inf`, `yj_inv_partials_at_inf`, `yj_fwd_lj_partials_at_inf`
and `yj_inv_lj_partials_at_edge` to `scalar_yeo_johnson.hpp`. The parity
argument therefore now points the other way, and the cost is lower than the
first estimate: that estimate assumed each limit had to be derived on its own,
while the sibling's derivation shows one rule covering all of them, that a
lambda-derivative multiplies a closed form by a logarithm of the argument, which
loses to a negative power of it and wins against a vanishing exponent, so the
sign of the exponent decides the mixed limits exactly as it decides the pure
ones.

What stays true either way, and was not reopened: the derivatives in lambda of
the four bound functions (`dom_lo`, `dom_hi`, `img_lo`, `img_hi`) are written
only on the day a bijector parameter becomes modelled and something reads them.
They are a closed form of the bound and need no limit of a mixed partial, which
is why they are the route for the caller that wants a moving support bound, and
not the partials at a limit.

**The partials at a large finite argument, measured because yeo_johnson's lose
themselves there.** The inverse ones do, and sooner than the sibling's, which
are clean at 1e40 and lose 28 of 448 entries at 1e60. Counting the NaN among the
27 components with n2 = 0 over orders 1 to 6, at lambda in {-3/2, -1/2, 0, 1/6,
1/2, 2/3, 1, 3/2, 2, 3, 6} and with the argument taken on the unbounded side of
the image: clean at 1e10 and 1e20; 1 entry of the inverse map's at 1e40, at
lambda = 0 alone; 2 to 3 of the inverse map's and 1 of log|h'|'s at 1e60, for
every lambda; 4 to 10 and 3 at 1e100; and 6 to 21 and 6 at 1e300. The forward
map's partials and the forward log-Jacobian's are clean at every argument up to
1e300. This is a loss inside the forms at a finite point and not a missing limit
at an end; its consequence for any test of the partials is that the arguments
that approach an end must stop at 1e20 for the inverse, where nothing is lost
yet.

**The log-Jacobian's half of that was repaired later the same day, by the
yeo_johnson session and inside the function the two maps share, with one hole
left.** It put the guard in `yeo_johnson_inv_lj_pos_partials`, which
`box_cox_inv_lj_partials` forwards to, so above |lambda u| = 1e3 the
lambda-derivatives come from their structure instead of from the forms, where
psi^(j)(z) underflows while z^(j+1) overflows. Re-measured here against the
column above: log|h'|'s partials are now clean at 1e60, 1e100 and 1e300 for
every lambda tested EXCEPT lambda = 0, where 6 of the 7 entries are still NaN at
1e300. The guard keys on |lambda u|, which is identically zero at lambda = 0, so
that one column never takes the rescaled route; it is clean at 1e100 and below
because the forms themselves still hold there. The inverse MAP's partials are
unchanged by that work, 3 NaN at 1e60, 10 at 1e100 and 21 at 1e300, their route
being a different function.

One function name, checked because the two are easy to confuse.
`box_cox_inv_lj_partials` forwards to `yeo_johnson_inv_lj_pos_partials`, the
inner positive branch the two maps share, and that function was not touched by
the yeo_johnson repair: its guards went into the public
`yeo_johnson_inv_lj_partials`, a different function of the same prefix. Every
hunk of that file's diff is an insertion and none falls inside the shared
function, so box_cox's inverse log-Jacobian partials did not move with it.


**The partials repaired too (Giovanni, the same day, after the measurement
above).** All six families now return their limits: the forward map's and its
log-Jacobian's at v = 0 and v = +inf, the inverse map's and its log-Jacobian's
at an infinite end of the image and at the finite one. Measured before, over the
same 21 values of lambda: 4 to 35 NaN components per lambda among the 27 with
n2 = 0 over orders 1 to 6, at every end. After: none, on gcc and on CUDA.

**What the rules are.** The rule of the pure limits carries over wherever the
argument runs to an infinity: a lambda-derivative multiplies the closed form by
a logarithm of the argument, which loses to a negative power of it and wins
against a vanishing exponent, so the exponent's three-way verdict is unchanged
and only the knife-edge value moves, from the prefactor at n1 = 0 to an infinity
above it. The six rules, with n0 = K - n1 argument derivatives and n1 in lambda:

- *g at both ends.* The sum over p of e_(m-1-p)(c) n!/(n-p)! L^(n-p) is dominated
  by the smallest p whose coefficient does not vanish, a smaller p carrying a
  higher power of L = log v; if every coefficient in range vanishes the sum is
  identically zero. At n0 = 0 the value tends to an end of the image, whose
  lambda-derivatives are (-1)^(n1+1) n1!/lambda^(n1+1) where that end is finite.
- *log|g'| at both ends.* It is linear in lambda, so only n1 = 0 and n1 = 1 are
  not identically zero; the column n1 = 1 is d^n0_v log v, free of lambda, and so
  is not zero at lambda = 1 where the column n1 = 0 is.
- *h at an infinite end.* P_n0(lambda) with the sign (-1)^n1 from the
  lambda-derivatives, and the exponent's sign from 1 - n0 lambda; at u = -inf
  everything dies with h.
- *log|h'| at an infinite end.* An argument derivative carries w^(-n0) and dies,
  so only n0 = 0 survives, and there the quantity is c_n1 log w with
  c_n1 = d^n1_lambda[(1 - lambda)/lambda]. At lambda = 0 it is affine in u and a
  second family appears, the constant (-1)^n1 (n1!)^2 wherever n0 = n1 + 1.
- *log|h'| at the finite end.* The most singular term is C u^n1 w^(-n0-n1), so
  every entry with n1 >= 1 is an infinity, of the sign of
  -(1 - lambda)(-1)^n0/lambda^(n1+1-n0); at lambda = 1 the factor 1 - lambda
  vanishes and the next term, carrying -1/lambda^2, leaves (-1)^n0.
- *h at the finite end,* the one that is not the pure rule in disguise. There a
  lambda-derivative brings u/w and not log w, so it lowers the power of w by one:
  the exponent is 1/lambda - K and depends on the total order alone, not on how
  it splits. Above the edge the sign is that of
  P_n0 (-1)^n1 prod_(i=0..n1-1)(1 - (n0+i) lambda), the sign of lambda cancelling
  between u^n1 and the product, and below 0 that product and P_n0 are both
  positive, which leaves (-1)^n1. On the edge itself, K lambda = 1, the power is 1
  and no logarithm is left, so every n1 is finite and the column is
  P_K(lambda) (-1/lambda^2)^n1, consecutive entries in the ratio -1/lambda^2,
  which is -K^2 there; at K = 6 and lambda = 1/6 it is 5/324, -5/9, 20, -720,
  25920, -933120, 33592320. Two degeneracies, both at lambda = 1/M with M an
  integer below K, where either P_n0 or one factor of the product vanishes and the
  next term takes over: the limit is then 0 at n1 = 0, where h^(K) is identically
  zero, and an infinity of sign (-1)^(n0+M) above it.

**Checked**: all 2592 of them, by `check_partials_limits` in
`tools/reference/bijector_box_cox_forms.py`, each rule against SymPy's own limit
of the mixed derivative of the definition, over lambda in {-3/2, -1/2, 0, 1/6,
1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3} and every (K, n1) with K from 1 to 6, with the
free invariant that the column n1 = 0 reproduces the pure rule. With the 1232
pure limits of the section above, 3824 limits of box_cox are now held against
SymPy's own limit of the definition. One defect in the
check itself, found and repaired: `bc_definition` substitutes lambda before
differentiating, which leaves an expression without lambda whose
lambda-derivatives are identically zero and would have passed any rule at all;
`bc_definition_sym` keeps lambda symbolic and reaches the degenerate lambda = 0
as a limit afterwards.

**The test** is the case "box_cox: the partials at the ends of both domains" in
`tests/bijector/test_box_cox.cpp`, in a second kernel: no NaN in any of the 27
entries of any of the four quantities at any of the four ends, the column n1 = 0
against the pure member of the same order (two code paths that must agree), the
knife-edge columns and the degenerate ones written out, the constants of
log|h'| at lambda = 0, the lambda-derivatives of the bound, the vanishing of
log|g'| above n1 = 1, and the approach to every finite limit. The approach is a
weak oracle here and the arguments are chosen for it: a lambda-derivative brings
powers of the logarithm, so the correction decays as v^lambda (log v)^n1 and
1e-20 still leaves 6e-3 relative at lambda = 1/3 and order 6; the forward
arguments are therefore 1e-300 and 1e300, where the forms are clean, while the
inverse ones stop at 1e20 for the reason measured above.

**CUDA, with the same counts.** 7 box_cox cases, 16307 assertions on the
`build_cuda.bat` default build, against 16299 on gcc; the eight extra are the
device's own `REQUIRE` on the mirror. The full suite is 667 cases and 1187260
assertions there, all passing. Measured on the default build and not under
`nofmad`, as the bounds require. An earlier CUDA run of the same code failed 12
assertions, all of them in `test_yeo_johnson.cpp` and none in box_cox, and they
went with that session's own repair of its plateau gates.

**Why box_cox cannot lose an infinity to FMA at a finite end, and the check that
shows it.** The yeo_johnson session measured a defect of that kind in its own map
and asked whether box_cox shares it. It does not, and the reason is structural:
at a finite end every quantity of box_cox is taken by the guard
`bc_inv_end_of`, which asks `u == -1/lambda`, a comparison with the bound the
dispatch itself used, and then returns a literal infinity from the limit
functions. No intermediate has to cancel. The defect needs the opposite shape,
which is what the sibling had: the infinity obtained by letting w = 1 + lambda u
cancel to exactly zero, so that the negative powers produce it by themselves.
The host does cancel there, lambda times the rounded -1/lambda rounding back to
-1; CUDA does not, nvcc contracting 1 + lambda u into an FMA that keeps the
exact product, so w survives at about 4e-17 and the infinity comes out as a
finite 1e100 instead.

What shows it rather than argues it: the test asserts those values as exact
infinities and not merely as large ones, and they pass on the device. The pure
log-Jacobian at the finite end is 48 such assertions (8 values of lambda by
orders 1 to 6), h itself at the finite end 7 more per negative lambda, and the
partials now 54 (two negative lambda by the 27 entries) plus the three of
lambda = 2/3 past its edge. The general rule is worth keeping next to the FMA
entry of CLAUDE.md: an assertion that holds on gcc because an intermediate is
exactly zero is a CUDA failure in waiting, and it does not present as a
tolerance problem but as an infinity turning into a large finite number.



**A test defect that only CUDA could show, and the rule it gives.** Both kernels
of `test_box_cox.cpp` take a pointer into a multi-dimensional View and walk the
last index by hand, which is contiguous only under `LayoutRight`. The views were
declared without a layout, so each took the execution space's own: `LayoutRight`
on the host and `LayoutLeft` on CUDA, where the FIRST index is contiguous. On the
device the writes therefore scattered across the lambda index and every entry
never written read back as the zero Kokkos had initialized it with. Measured on
CUDA before the layout was named: 120 failed assertions in the case of the pure
limits, every one of them an expected infinity read as exactly 0, while gcc was
green throughout; found by the yeo_johnson session, whose own views had the same
shape and the same 54 and 53 failures. Naming `Kokkos::LayoutRight` in the two
aliases is the whole repair, and it is the pattern `param_common.hpp` already
uses.

Two things this is worth keeping for. A green host run says nothing about this
class of defect: a test that writes through a raw pointer into a
multi-dimensional View has to run on CUDA before it is believed. And the rule
generalizes beyond ScatterView, which `CLAUDE.md` already records from the other
direction: wherever a kernel takes a pointer into a View and walks one index by
hand, the layout is named and not left to the default.

## The bound distances of the scalar bijectors at an infinite bound (2026-11-17)

**What was wrong.** `scalar_bijector::inverse_distance_upper` and
`inverse_distance_lower` return the distance of `h(u)` to the two bounds of the
range of `h`, which is the domain of `g`. Six maps compute that distance
directly, to avoid the cancellation of a difference near a finite bound, and
seven more return `+inf` for a bound that is infinite; everything else went
through the generic difference, `dom_hi - h(u)` and `h(u) - dom_lo` for a plain
bijector and `forward_domain_upper() - inverse(u)` and
`inverse(u) - forward_domain_lower()` for an inverted one. Wherever the bound
is infinite and `h(u)` has reached it, that difference is `inf - inf` and gave
NaN. The defect was found by the session that wrote `transformed_distrib`,
which reads the ends of the support of `g(X)` from the bijector.

Measured over the twenty maps in twenty-seven parameter settings, both
directions of each (plain and inverted), at both ends of the closed domain of
`h`, for both distances: 108 cells, of which 58 were NaN.

| where | cells | before | after |
|---|---|---|---|
| plain bijector, the generic difference | 8 | NaN | `+inf` |
| inverted bijector, the generic difference | 45 | NaN | `+inf` |
| `h(u)` itself NaN at the end, `power` | 1 | NaN | the value's limit |
| `h(u)` itself NaN at the end, `box_cox` | 4 | NaN | the value's limit |
| every other cell | 50 | the limit | unchanged |

The eight cells of the plain direction are `identity` at both ends, `log` and
`sqrt` and `asinh` at `u = +inf`, `asinh` at `u = -inf`, and `inverse` and
`inverse_sq` at `u = 0`, where `h(u) = +inf`. The forty-five of the inverted
direction are every end of every map whose image, which is the range of `h`
there, has an infinite bound. That set is not confined to an infinite argument:
inverted `logit`, `probit`, `cauchit`, `cloglog` and `loglog` failed at the
finite ends `u = 0` and `u = 1`, inverted `atanh` at `u = -1` and `u = 1`, and
inverted `bounded(-2, 3)` at `u = -2` and `u = 3`. The defect is that the value
has reached an infinite bound, not that the argument is infinite.

**The convention, decided with Giovanni on 2026-11-17: the distance to an
infinite bound is `+inf` at every point of the closed domain, the bound itself
included.** With `V = (v_min, v_max)` the range of `h`, the upper distance is
`d(u) = v_max - h(u)`. When `v_max` is finite, `h(u) -> v_max` as `u` runs to
the end of its domain and the limit of `d` is 0, which is what the functions
already returned. When `v_max = +inf`, every `h(u)` of the interior is finite
and `d(u) = +inf` identically: `d` is the constant `+inf` on the whole
interior, so `+inf` is also its limit at the end, and the rule of the module,
that a closed bound gives the limit, decides the value without a further
choice. The alternative put to Giovanni was 0, read as the distance of the
bound from itself, which is a value assigned to `inf - inf` by decree rather
than a limit; it would also have meant rewriting the seven per-map functions
that today return `+inf` ignoring `u`. The two conventions differ in those 53
cells and nowhere else.

The repair is two helpers in `detail_scalar` of
`include/laplace/bijector/scalar.hpp`, `to_upper_bound(hi, value)` and
`to_lower_bound(lo, value)`, which return `+inf` for an infinite bound and the
difference otherwise, and NaN wherever the value is NaN, which is how a point
outside the closed domain and an invalid parameter arrive there. They replace
the generic difference in the two `default:` branches of `inv_to_upper` and
`inv_to_lower` and in the `inverted` branch of the two members. No per-map
function needed a change, and no caller was touched.

**A second defect, in `power`, repaired here.** `power_offset(e, x)` forms the
small exponent correction `1 + e log x`, the term that carries the part of the
exponent lost to rounding when `1/lambda` is not representable. Its guard
skipped the term at `x = 0`, where `log x` is infinite, and not at an infinite
`x`, where it is infinite as well: with `e` of the order of 1e-17 the term came
out as an infinity of the sign of `e` and multiplied the power instead of
correcting it. So `h(+inf)` was NaN for `power(-3)` and `power(-2.5)`, where the
power is 0 and the correction `-inf`, and `-inf` for `power(2.5)`, where the
power is `+inf` and the correction again negative; the true limits are 0 and
`+inf`. The guard now skips any non-finite `x`, at which the sign of the whole
exponent is carried by its leading part, the one that `power_scaled` raises
directly. Finite arguments are untouched, the condition being `!isfinite(x)`.
The four `box_cox` cells of the table were repaired in `scalar_box_cox.hpp` by
the session that owns that file, on the same day.

**The test** is in `tests/bijector/test_scalar.cpp`, the case "bijector scalar:
the distances at the ends of every domain": thirty settings covering the twenty
maps, each plain and inverted, in a kernel on the default execution space (a
named function with external linkage, never a lambda inside a `TEST_CASE`). For
each of the two ends of the closed domain of `h` it checks that neither
distance is NaN and that neither is negative; that the distance to an infinite
bound is exactly `+inf`; that the distance to a finite bound equals the
difference, except where `h(u)` has run off to the opposite infinite bound and
the distance is `+inf` there too; and that both distances are NaN for the two
invalid settings, `affine(1, 0)` and `bounded(3, 1)`. 688 assertions, green on
gcc; the CUDA build gives the same counts.

## The improper multivariate Student t: mvstudent_t2 over a singular Q (2026-11-17)

What. The improper (intrinsic) class of `mvstudent_t2_distrib<S>` for a positive
semidefinite structure with a constant null space (`weighted_sum_psdmat`), the
t counterpart of the improper `mvgaussian2`: Q = M(theta) of rank rho < d, r = y - mu,
delta = r' Q r,

    l = c_rho(nu) + 1/2 log|Q|_+ - (nu + rho)/2 log1p(delta/nu),

defined for every y. Writing r = U a + U0 b (U, U0 orthonormal bases of the range and
of the null space of Q), delta depends on a only, so the law is the rho-dimensional
proper t of a times Lebesgue measure on b; it is also the scale mixture, over
w ~ Gamma(nu/2, rate nu/2), of the improper Gaussian with precision w Q and its
range-only constant. Use: a t penalty on a smooth, Q = lambda S.

How. The blocks are the degenerate mvstudent_t1's kernels with W = Q and the sign
+1/2 of the log pseudo-determinant (the far-tail double-double route included);
no support test. The state holds no factor; `constrained_state` adds the factor of
Q + cP for the constrained law U0'(y - mu) = 0: `constrained_rng`
(y = mu + (I - P) L^-T z / sqrt(w)), `constrained_mean` (mu for nu > 1, NaN otherwise),
`constrained_covariance` (nu/(nu - 2) Q^+ for nu > 2, NaN otherwise). The expected
information is that of the constrained law: mu block kappa Q, theta-theta
kappa I^G_ab - t_a t_b/(2 (nu + rho + 2)) with I^G from log|Q|_+ (structures with
d_ab Q = delta_ab d_a Q), theta-nu and nu-nu as the proper t with rho.
The improper `mvgaussian2` gained `constrained_mean` (mu) for symmetry.

One matrix. With a single penalty (one theta) the theta-theta entry is
kappa rho/2 - rho^2/(2 (nu + rho + 2)) = nu rho/(2 (nu + rho + 2)), formed by a
cancellation that loses about (nu + rho)/nu: with exact inputs (mpmath at 50 digits)
1.5-29 ulp at nu = 1 for rho = 8-98, 81 ulp at rho = 498, 4.9e3 ulp at nu = 0.01,
rho = 98. Ttilde = 0 exactly there, so the traceless form with H = 0,
nu t^2/(2 rho (nu + rho + 2)), is used instead, in the route shared with the
degenerate mvstudent_t1: measured 0 ulp against nu rho/(2 (nu + rho + 2)) at
nu = 1e-3 to 30, both classes. Several penalties keep the direct form.

Measured (tests/distrib/test_mvstudent_t_improper.cpp, references
tools/reference/distrib_mvstudent_t_improper.py, mpmath at 60 digits from the
definition, 3866 values; weighted_sum structures d = 4..6, m = 1..3; host only, the
CUDA build compiles and runs the host tests; gcc and MSVC host tables identical
except one cell), worst ulp1N over A + K <= 6 and the information to order 4:

| point | blocks | information |
|---|---|---|
| random, m = 1, 2, 3, two D2 | 4.2 | 3.0 |
| m = 1, nu = 0.05 | 2.1 | 1.9 |
| m = 1, nu = 1e6 | 0.79 | 0.18 |
| two penalties 9 apart | 23 | 0.67 |
| large residual | 2.1 | 0.56 |

The spread point (theta 9 apart) is the worst, as in the degenerate class, but
small: the improper blocks use Q itself, never the dense (Sigma + cP)^-1. A shift
of y along the null space (+0.75 times the constant vector) changes no block beyond
65.3 ulp1N, the rounding of the shifted y. Constrained draws (d = 5, rho = 4, nu = 7,
2e5 draws): null component 1.7e-15 relative, mean 0.0022 sd, covariance 0.0028 of
its largest entry; the constrained covariance equals nu/(nu - 2) times the improper
Gaussian's Q^+ to 1e-14.

Open. statmodels7 has no improper t (its mvstudent_t refuses a rank-deficient
matrix): no frozen references. No bench: the kernels are the degenerate t's.


## The Dirichlet and the multinomial: dirichlet_distrib and multinomial_distrib (2026-11-17)

What. The first laws with a `simplex_vector` parameter: the mean (Dirichlet) or the
probabilities (multinomial) mu on the open simplex of d categories, carried by their
additive log-ratio coordinates eta (reference category the last, eta_d = 0),
log mu_j = eta_j - LSE(eta); the Dirichlet adds the precision phi > 0, alpha_j = phi mu_j;
the multinomial's size N is a data constant (a trailing field, as binomial_distrib's).

    Dirichlet:   l = lgamma(phi) - sum_j lgamma(alpha_j) + sum_j (alpha_j - 1) log y_j
    multinomial: l = log N! - sum_j log y_j! + sum_j y_j log mu_j

The multinomial is the first discrete multivariate law (`kind`, `logpmf`); the
Dirichlet stores d values of y and differentiates in the d - 1 reference coordinates
(`y_coord_size()`). Derivatives: K in phi, A in eta, A + K <= 6; derivatives in y
(Dirichlet); the expected information and its derivatives in (eta, phi) to order 4.

How. The derivatives of mu and log mu in eta, over every subset of the positions of a
tuple, come from alr's subset recursion (d_b mu_c = mu_c (delta_bc - mu_b)), keeping
every subset; -L is the joint cumulant kappa of the categorical indicators.
- The cumulants vanish where their polynomials do (1 - 2 mu = 0 and the like), and in
  double the recursion carries an absolute error of about eps/4 there, which phi or N
  then multiply (measured 2.5e5 ulp1N at N = 1e6). The recursion runs in
  double-double from the unit's probabilities, formed once in double-double
  (mu_j = e^(eta_j - eta_max) / sum, and 1 - mu_j as the sum of the others), and each
  table entry is rounded at the end; afterwards <= 60 ulp1N.
- log mu of the largest category is -log1p(sum of the others): eta_c - LSE loses the
  digits of a probability near one (1e6-1e8 ulp1N at mu = 1 - 1e-8, phi = 1e7).
- Dirichlet, -sum_j lgamma(phi mu_j), per coordinate: below alpha = 1,
  -lgamma(a) = log a - lgamma(1 + a), the sum of log alpha_j in closed form through the
  cumulants and -lgamma(1 + a) through Faa di Bruno with psi_n(1 + a); from alpha = 1,
  -lgamma(a) = -(a log a - a) + 1/2 log a - 1/2 log 2pi - delta(a), Stirling's error
  delta through Faa di Bruno (math/stirling.hpp, to order 6), the leading terms in
  closed form. Measured with exact inputs (scratch, d = 3): the direct route (polygammas
  at alpha, as beta_distrib) loses 2.2e5 ulp at phi = 1e3, 3.9e9 at 1e5 and 4.6e13 at
  1e7 in blocks with two or more phi derivatives (the leading Stirling terms are linear
  in mu_j and cancel across j), and 1e8-3e9 ulp at alpha = 1e-9; this route <= 32 ulp.
- The pieces linear in mu_j of the large set are summed over the small set with the
  sign changed (sum_j d mu_j = 0 exactly), and are exactly zero when the large set is
  empty (else phi^-(K-1) times a rounding: 1e12 ulp1N at phi = 1e-3).
- The data and Stirling's leading term are combined per large coordinate as
  d mu_j log(y_j/mu_j), log(y_j/mu_j) = log1p((y_j - mu_j)/mu_j) with mu_j in
  double-double (y is close to mu at a large phi); the Leibniz terms with a derivative
  on log mu_j sum, over every j, to (A - 1) kappa, so the large set's share is taken
  from the cumulant (the value went from 4.6e5 to 90 ulp1N at phi = 1e7).
- Expected information, E log y_j = psi(alpha_j) - psi(phi):
  I_ab = phi^2 sum_j psi_1(alpha_j) d_a mu_j d_b mu_j, I_a,phi = sum_j d_a mu_j alpha_j r_1(alpha_j)
  (r_1 = psi_1 - 1/x: the 1 of alpha psi_1 drops exactly), I_phi,phi = sum_j mu_j^2 r_1(alpha_j) - r_1(phi);
  their derivatives by the same split, psi_1(a) = 1/a + 1/(2a^2) + delta''(a) from 1 and
  1/a^2 + psi_1(1 + a) below, and phi sum_large d(mu X_a X_b) (X_a = d_a log mu_j) from the
  cumulant kappa_ab (8.5e4 -> <= 60 ulp1N). Multinomial: N kappa_ab and N kappa_abt.
- The multinomial's log-mass in the saddle-point form of binomial_distrib, with
  Stirling's error and the unit deviance (no cancellation at large N).
- Two scores were differences of close terms carrying the rounding of mu: y_a - N mu_a
  and the Dirichlet's (alpha_a - 1)/y_a^J - (alpha_d - 1)/y_d^J; both are formed in
  double-double from the unit's double-double mu (5.2e5 and 1.75e3 -> 0 ulp1N).
- Draws: the Dirichlet as log G_j = log G(alpha_j + 1) + log(U_j)/alpha_j (exact in law;
  the gamma by inversion at draw 2j, the uniform at 2j + 1 of counter_uniform_source),
  y by log-sum-exp, never 0/0 at a tiny shape; the multinomial by conditional binomials,
  y_j ~ Bin(N - sum_(i<j) y_i, mu_j / sum_(i>=j) mu_i), by inversion at draw j.
- The tables (4 KB) and the partition sums (3.5 KB) live in the law's scratch, not on
  the thread's stack: on CUDA they overran the tests' 8 KB stack.

Measured (tests/distrib/test_simplex.cpp, in a kernel; references
tools/reference/distrib_simplex.py, mpmath at 60 and 75 digits from the definition,
11262 values, d = 2, 3, 5, phi from 1e-3 to 1e7, N to 1e6, shapes to 1e-12, the
information's closed forms checked by quadrature over the beta law), worst ulp1N:

| | gcc | CUDA |
|---|---|---|
| Dirichlet blocks, A + K <= 6 | 44.8 | 33.2 |
| Dirichlet y derivatives | 14.3 | 4.82 |
| information (eta, eta), (eta, phi), (phi, phi), to order 4 | 4.06, 4.83, 3.06 | 4.27, 5.95, 11.1 |
| multinomial blocks, A <= 6 | 1.65 | 1.65 |
| multinomial information, to order 4 | 0.717 | 0.717 |

statmodels7 frozen (tools/reference/statmodels7_simplex.R, distributions7 0.94.2,
608 values, orders 0-4 and the order-0 information, d = 2, 3, 5): agreement within
statmodels7's own accuracy, 439 ulp1N in the Dirichlet blocks (its direct routes at
phi = 300) and 628 in the multinomial log-mass (its lgamma form at size 1000), the rest
<= 2.3. Its expected third and fourth derivatives are E[d^3 l] and E[d^4 l], not the
derivatives of the information, and are not frozen. Draws (4e4, d = 3): mean within
4.5 standard errors, covariance within 4 per cent of its largest entry, on the simplex
or summing to the size; at phi = 0.01 every coordinate finite and the sum one.

Speed (benchmarks/distrib_simplex.cpp, 2e4 units, phi = 30, size 50), ns per unit,
orders 0 / 1 / 2:

| | one thread | 16 threads | CUDA |
|---|---|---|---|
| Dirichlet d = 3 | 2924 / 3650 / 5975 | 238 / 297 / 501 | 66 / 89 / 185 |
| Dirichlet d = 10 | 11568 / 15509 / 36658 | 888 / 1229 / 2616 | 207 / 313 / 859 |
| multinomial d = 3 | 503 / 1503 / 1956 | 44 / 118 / 157 | 12 / 32 / 44 |
| multinomial d = 10 | 3532 / 7747 / 11839 | 294 / 588 / 920 | 65 / 109 / 236 |

The double-double tables are most of it: a first version that rebuilt the
double-double probabilities in every table cost 5-40 times more on one thread.

Open. independent_product_distrib does not take a component whose unit is constrained
(y_coord_size() < dim()).

**A rule the two bijector sessions arrived at from opposite directions.** Where a
quantity's limit at a boundary is an infinity and the code reaches it through a
guard, the test asserts that infinity EXACTLY. Every weaker check -- that the
value is not NaN, that it is large, that it is finite or infinite -- passes
whether the guard fires or not, so it cannot tell a working guard from a missing
one. The yeo_johnson session states it that way; one refinement belongs with it,
measured here. The exact assertion distinguishes them only on the DEVICE. With
the guard removed the forms take over, and at a finite end they obtain the
infinity by letting w = 1 + lambda u cancel to exactly zero, which the host does
and CUDA does not: on gcc the forms return the same infinity the guard would
have, so even `== inf` passes there and nothing in a host run can see the
difference, while on CUDA the FMA leaves w at about 4e-17 and the infinity comes
back as a large finite number, which only the exact assertion rejects. So the
rule has two halves: assert the infinity exactly, and do not believe a host run
about it.

How both sessions found it, which is why it is a rule and not an anecdote. Each
had asserted the exact infinities for one half of the partials at the finite end
and left the other half to a no-NaN check, in opposite halves, an hour apart:
yeo_johnson for the log-Jacobian's partials and not the map's, box_cox for the
map's and not the log-Jacobian's. Neither gap was visible in a green run on
either platform, since the guards were in fact firing; what each gap would have
hidden is a guard that stopped firing later. The missing assertions are
`yj_inv_partials_at_edge`'s in `test_yeo_johnson.cpp` and the block (4c) of
`test_box_cox.cpp`.

**The inverse partials at a large finite argument, repaired by adopting the
yeo_johnson session's rescaled route (Giovanni, 2026-11-17).** The measurement
two paragraphs above recorded the defect and left it: above |lambda u| = 1e3 the
forms compute inf times 0, psi^(j)(x) underflowing while the power of u
overflows, and the column came out NaN -- 3 of the 7 entries at u = 1e60, 10 at
1e100 and 21 at 1e300 over orders 1 to 6. That session then wrote a route which
keeps q_j = x^(j+1) psi^(j)(x) bounded, put it in the shared positive branch of
its own header, and offered it. `box_cox_inv_partials` now takes
`yj_inv_partials_rescaled_pos<K>` above that threshold.

**It is Box-Cox's column unchanged, and the instruction to adapt it was
unnecessary.** The session said to drop its trailing
`if (n0 == 0 && n1 == 0) out[n1] -= 1.0;`, which is the -1 of
h = exp(S) - 1 against Box-Cox's exp(S). That line never fires for a partial:
n0 = 0 needs n1 = K, and n1 = 0 then forces K = 0, which `bijector_partials` is
never instantiated for. So the routine is called unchanged rather than copied,
which keeps one source for both maps. It takes lambda and u directly and
performs no Yeo-Johnson reflection, which is what makes it correct here.

**CUDA, from the yeo_johnson session's own build of the whole tree.** Its
build-cuda-yjlarge compiled this tree, my uncommitted box_cox changes included,
and its ctest passed at 100 per cent; run on its binary, the box_cox cases give 8
cases and 16718 assertions against 16710 on gcc. The eight extra are not
device-conditional assertions, which this file has none of: localised per case,
every case matches except "the limits at the ends of both domains", 4700 on gcc
against 4708. That case's approach check branches on the data -- a finite limit
costs one assertion, a limit already attained none, a plateau three, an
approaching infinity two -- so the count follows which branch each entry takes,
and the approach values differ in their last bits between the platforms under
FMA contraction. Eight entries therefore take a branch with more assertions on
the device. Every assertion passes on both, so this is a property of a
data-dependent test and not a difference in the map. So the adoption is device-verified without a
second build of the GPU's time, which is what one build for every session buys.

**Measured against references that share nothing with the C++ but the
definition.** (1 + lambda u)^(1/lambda) differentiated symbolically by SymPy and
only then evaluated by mpmath at 60 digits, 189 values over lambda in {1/2,
-3/2, 1/6, 3, 2/3} and u from 1e4 to 1e300. No NaN, and the worst relative
difference 1.373e-13, at lambda = 1/6 and u = 1e100 in the order-5 column at
n1 = 1; by order, 2.1e-14 at orders 1 and 2 rising to 1.37e-13 at 4 to 6. That is
the floor rather than the route's fault: the column carries
exp((1 - n0 lambda) S), so one ulp of S is amplified by it, and a test bound
tighter than about 2e-13 would be measuring S. A numerical differentiator cannot
produce these references at all, the yeo_johnson session having measured
mpmath's own returning entries wrong by two hundred orders of magnitude at
u = 1e60.

**The threshold, and what it costs.** 1e3, the same constant as the sibling's.
Measured rather than assumed: just above it, at lambda u = 5e3 and 6.7e5, the
forms are the more accurate route by a factor of 1.8, 6.0e-15 against 1.07e-14,
both far inside any bound. Kept all the same, on the ground that above the
threshold the failure is a cliff into NaN while below it the cost is under a
factor of two, and that one constant for both maps is worth more than the
factor. A later choice may move it; the numbers are here for that.

**The band was repaired too, within minutes of the paragraph above being
written, and only one regime is left.** The yeo_johnson session built a second
route for it on Giovanni's word: the band is |lambda x| below about 1e3 together
with |x| large, where the same inf times 0 is reached from the other side, the
power of x standing in front of a bracket in psi^(j) which cancels as its
argument goes to 0. Its route takes the sum of z^m psi^(j+m)(z) first and applies
one power of the argument once, so a sum that is exactly zero returns zero
instead of meeting an overflowed power. It sits in
`yeo_johnson_inv_lj_pos_partials` again, which is what `box_cox_inv_lj_partials`
forwards to, so box_cox's log-Jacobian partials are repaired by it as well.

Measured here over 72 cells, lambda in {0, 1e-310, 1e-305, 1e-300, 1e-200,
1e-100, 1e-50, 1e-20, 1e-3} by u in {1e40, 1e60, 1e100, 1e125, 1e150, 1e200,
1e225, 1e300}: the log-Jacobian's partials are now clean in EVERY cell, 0 NaN of
the 7 entries at order 6 everywhere, where lambda = 0 had lost 6 of them at
1e300 before.

What is left is the inverse MAP's partials, in two pieces that are worth keeping
apart because only one of them is shared.

The shared piece: inside the rescaled route, for a tiny lambda with a large
argument. Measured here as 106 NaN over 56 of those 72 cells at order 6, from
lambda = 1e-310 up to 1e-50 and from u = 1e40 upward. The yeo_johnson session
re-measured its own at 1147 cells and found 370 NaN over 183, for lambda from
1e-30 down to 1e-310 with |x| from 1e50 up, correcting an earlier figure of 36
in 12 cells that a coarse grid had produced. Its cause is not lambda^(j+1)
underflowing, which stays representable: at lambda = 1e-30 and x = 1e60 that
power is 1e-210, while the chain built on it overflows, the exponent's first
derivative being about 1e210 so that a sixth Bell polynomial would be 1e1260, and
an inf minus inf appears. Nothing finite is lost: that session evaluated
exp((1 - n0 lambda) S), which carries the value, for each of its 370 entries and
found it infinite in every one, 0 exceptions, so what is returned is NaN in place
of an infinity of a known sign.

The piece that is box_cox's alone: lambda = 0 exactly. There |lambda u| is 0, so
the rescaled route is never taken and the NaN comes from box_cox's own forms,
`bc_bell` and `bc_inv_mixed`, where the Bell polynomial of
u^(j+1) psi^(j)(0) overflows. Measured from u = 1e40 upward, 1 entry of the 7 at
order 6 and 2 from u = 1e100. The sibling's map has no NaN at lambda = 0 at all,
its own forms handling z = 0 and returning infinities, which is the one place the
two maps differ in this corner. box_cox's truth there is an infinity whatever u
is above about 710, h being exp(u), so again nothing finite is lost.

Nothing was implemented for either piece. Closing them means deriving the signs
of those infinities, which is a separate piece of work; the shared half is with
Giovanni through the yeo_johnson session, under its instruction to build the band
and nothing more, and the lambda = 0 half is box_cox's own and is recorded here
rather than repaired.

**The test** is the case "box_cox: the inverse partials at a large finite
argument" in `tests/bijector/test_box_cox.cpp`, with the references printed by
`tools/reference/bijector_box_cox_large.py`: the two columns whose every entry a
double can hold, lambda = 1/2 at u = 1e60 and lambda = -3/2 at u = -1e60 at
orders 3 and 6, the bound 2e-13; P_6 at lambda = 1/6 surviving the rescaled
route, 5!/6^5; and no NaN in any order over ten values of lambda and four
arguments. Two of the references are below the smallest double, 2.19e-398 and
-2.31e-337, and are written as zero with the entry required to underflow:
a literal of that size is truncated by the compiler with a warning, and a test
file that warns costs every other session the same noise.

## fixed_distrib: parameters held at given values (2026-11-18)

What: `include/laplace/distrib/fixed.hpp` is new: `fixed_distrib<D, C...>`, the
parent distribution D with some of its parameters held at given values and
removed from the parameter vector, specialized for a univariate parent and for a
multivariate one, with the host factory `fixed<C...>(parent, values...)`. The
plan is `note/piano_wrapper.md` section 4; its section 5 had recommended not
carrying `fixed` as a wrapper, and Giovanni asked for it on 2026-11-18, so that
recommendation is superseded.

How, and why it is cheap. Write theta = (theta_F, theta_H) for the free
coordinates and the held ones, with theta_H = c a vector of constants; the law is
l(y; theta_F) = l(y; theta_F, c). This is a restriction of the domain and NOT a
reparametrization: the completion theta_F -> (theta_F, c) is the identity on the
free coordinates and carries no Jacobian. So for 1 <= K <= 6 and free indices
a_1, ..., a_K, the derivative of order K is the parent's own tensor of order K
RESTRICTED to the multi-indices that lie in the free set: a selection of
components, with no chain rule and no new arithmetic, so every surviving
component is bit for bit the parent's and the components involving a held
coordinate are absent rather than present and zero. The same selection carries
the mixed derivatives in y (selected on the theta part, untouched on the y part),
the derivatives of log F and log(1 - F), and the expected information: the law of
Y under the fixed model IS the parent's at (theta_F, c), so I^fix_ab = I_ab for a
and b free, the free BLOCK of the parent's information and not the inverse of a
submatrix of its inverse, so nothing is profiled out. Its derivatives to order
four are the parent's blocks selected twice, on the pair (a, b) and on the K
differentiating indices. The value, the cumulative functions, the quantile, the
atoms, the ends of the support, center, scale, y_kinks and y_regularity are the
parent's, none of them being indexed by a parameter.

The work is the index surface. A free index is mapped back to the parent's before
any member indexed by a parameter: `param_kinks(j, y)` on the univariate side,
and on the multivariate side `param_size`, `param_offset`, `theta_size()`, the
`idx` tuple of `logpdf_dk<K, A...>`, the slots and the index pair of
`expected_info_dk`, `coord_lower`, `coord_upper` and the block descriptors
`n_blocks()` and `block_start(slot, g)`. Two of those were found by reading
`multivariate_core.hpp` rather than from the list the work started with:
`coord_lower(j, i)` and `coord_upper(j, i)`, indexed by parameter AND coordinate,
and the block descriptors, whose boundaries a partially held dense parameter
would move.

Cost by kind of held coordinate. Any subset of the INDEXED coordinates theta (a
whole scalar parameter, a whole matrix parameter, single elements of either)
costs the remap of the at-most-six `idx` integers plus a completed theta written
once into the scratch, and leaves the derivative layout alone, a derivative in
theta being asked for by an explicit index tuple and not as a tensor over theta;
`scratch_size()` grows by the parent's `theta_size()`. A whole DENSE parameter (a
vector or a simplex_vector) is passed to the parent as the pointer the wrapper
holds and the parent is called with order 0 for it, whose Kronecker factor is one
double, so the layout is again unchanged and nothing is added to the scratch.
Both are free: no gather anywhere. On the univariate side the selection reads one
parent tensor on the stack, 28 doubles at order 6 over three parameters.

Two refusals on the multivariate side come from the interface and not from a
choice. `logpdf(st, y, x)` takes no scratch, so there is nowhere to interleave
free and held DENSE coordinates: a dense parameter may be held only whole, and
only when it is the law's only dense parameter, which is the case of every
multivariate law here. Single coordinates of a MATRIX are not affected, being
indexed. Holding part of a dense parameter, or one of several, would need a
logpdf that takes w, which is a change to `multivariate_core.hpp` and to every
multivariate law.

Two decisions taken with Giovanni on 2026-11-18, before the code. The held set is
a template parameter, since `n_params` and `params` are static and only the values
can be data; a held coordinate is named by `coord_of<D>("mu")` or
`coord_of<D>("mu", 2)`, which builds it from the parameter's name in a constant
expression. The two alternatives were a flat index into theta, which does not
exist on the multivariate side where theta and x are separate arrays, and a
compile-time string NTTP, which would add a type the library does not have. The
chosen spelling is what a call site writes; what it builds is an int that encodes
the parameter index and the coordinate inside it, and not a small struct of the
two, for the reason in the next section. And a partially held parameter SHRINKS
to its free coordinates rather than keeping its shape with constants in it: a
kept shape would report an information block over coordinates that are not
parameters, singular once their rows were zeroed, and would push the mask of free
coordinates into every caller. Measured before recommending it, for a dense slot
of n = 10 with m = 9 free at order 6: the shrinking reading gathers C(14, 6) =
3003 components out of the parent's C(15, 6) = 5005, which is 40 KB of added
scratch against a device block's 48 to 100 KB of shared memory; the kept shape
forwards the 5005 with no gather. The reason above decided it, not the figures,
and the figures are moot for a dense parameter, which the refusal above holds
whole.

Measured (`tests/distrib/test_fixed.cpp`, 14 cases and 1144 assertions, the same
counts and none failing on gcc and on CUDA). The comparisons are exact wherever the
member is a selection rather than a computation: the parent is built with the held
value in place and each surviving component must equal the parent's component at
the mapped multi-index bit for bit.

| case | quantity | result |
|---|---|---|
| student_t2, sigma held, free {mu, nu} | value, F, log(1 - F), quantile, support ends, center, scale | exact |
| | d1-d6 of the log-density, every component at every mapped tuple | exact |
| | d1-d6 of log F and of log(1 - F) | exact |
| | `logpdf_dy_dk<J, K>`, J + K <= 6 | exact |
| | information and its d1-d4, pair and tuple both selected | exact |
| enet, mu held | `param_kinks(0, y)` reports no kink (the parent's lambda) | count 0, was 1 unmapped |
| enet, lambda held | `param_kinks(0, y)` reports mu's kink at y | count 1, point == y |
| gaussian2, v held | the start leaves v at the held value; mu equals the unheld law's start | exact |
| fixed over truncated(gaussian2, -1, 2.5), v held | value, log F, d1-d6 of both | exact |
| fixed over zero-adjusted gamma2, sigma2 held | value, d1-d4, atom point and log mass | exact |
| fixed over fixed (student_t2, sigma then nu) | value, d1-d6; the nested start keeps both held values | exact |
| mvgaussian2 over diagonal_pdmat, d = 3, mu held at zero | n_params 1, theta_size and scratch_size the parent's, dense_size 0 | as stated |
| | value and the order-2 blocks in theta against the parent at mu = 0 | exact |
| | the start writes every coordinate of theta | exact |
| mvgaussian2, coordinate 1 of Q held | theta_size d - 1, scratch_size + the parent's theta_size, free coordinate 1 of Q is the parent's 2 | as stated |
| | value and the order-2 block against the parent at the completed theta | exact |
| | the start drops the held coordinate | exact |
| a univariate wrapper built inside a kernel | against the host | <= 1e-12 relative |
| mu held at zero inside a kernel | value and the order-2 block against the parent in the same kernel | exact |

What it says: nothing is recomputed, so there is nothing to lose. Every quantity
that the mathematics says is a selection is measured as one, to the last bit, and
the two cases that are about indices rather than values (the `param_kinks` remap
and the free-to-parent coordinate map of a partially held matrix) are checked
against the value an unmapped implementation would give.

### Four things nvcc does not accept that gcc does (2026-11-18)

The first CUDA build of `test_fixed.cpp` failed four times where gcc had been
green, each time for a reason worth keeping, and none of them about arithmetic.

A requires-expression written inline in a member's constraint evaluates to false
under nvcc where gcc evaluates it to true. All five errors of the first build
were `requires(const D p) { p.template logpdf_dy_dk<1, 0>(0.0); }` on the
member that forwards the mixed derivatives in y, which made the member vanish
from the wrapper over a parent that has it. The optional members of a parent are
now detected through named concepts in `detail_fixed`, which is what
`univariate_as_multivariate.hpp` had already found and said in a comment.

A `static constexpr` table read at a RUNTIME index has no definition in device
code. The second build stopped with nine errors of the form "identifier
`...::params_` is undefined in device code", from the members that asked whether
a parameter is dense by reading `params[j].kind` and `D::params[j].kind` inside
`LAPLACE_HD` functions. The question is now a fold over the parameter indices
(`detail_fixed::dense_slot`), so the descriptor is read only at compile-time
indices. The same reason is why `mv_coord_lower()` and `mv_coord_upper()` are
host functions, which their comment says.

A `static constexpr int` member forwarded from the parent is instantiated
eagerly by nvcc where gcc waits for a use, so `y_regularity = D::y_regularity`
is an error over a discrete parent, which declares none. It is inherited from a
helper specialized on the continuous kind, as `atom_mixture_distrib` already
inherits it.

A template argument of CLASS type cannot cross into an extended lambda's
closure. The third and fourth builds ended with MSVC's `fatal error C1903:
unable to recover from previous error(s)` on nvcc's generated host file, and
with no diagnostic of its own before it, which is what made this one hard to
read. nvcc writes the closure type into that file, and a template argument of
class type is printed there as a braced initializer:

    ::laplace::distrib::fixed_distrib< ::laplace::distrib::student_t2_distrib,
                                       ::laplace::distrib::fixed_coord{(int)1, (int)-1} >

which is not a template argument, so the host compiler cannot parse it. A probe
of eighteen lines settled it: one class template with a `coord{int, int}`
argument and one with an `int` argument, each captured by a `KOKKOS_LAMBDA`; the
first fails and the second compiles. The structural class type had passed an
earlier probe that used it only inside a `__global__` function, which is not
where a distribution goes. So the held coordinates are encoded as ints, the call
site keeping the spelling `coord_of<D>("mu")`, whose return type is the only
thing that changed. A braced list over the pack, `for (const fixed_coord c :
{C...})`, is also out of device code, its `std::initializer_list` having static
storage; the one in a `LAPLACE_HD` member is now a fold over
`std::index_sequence`.

## Frozen R references for wave 2 of the optimizers (2026-11-18)

**What.** Lasso, elastic-net and least-absolute-deviation fits produced in R
once, so that the proximal and bundle optimizers of wave 2 can be compared with
glmnet and quantreg without R at test time (note/optimizers.md section 12,
decisions 7 and 8). Versions: R 4.6.0, glmnet 5.0, quantreg 6.1. The script is
`tools/reference/optimizers_wave2.R`, run from the repository root with
`Rscript --vanilla`; it writes `tests/data/reference/optimizers/wave2/` (about
18 MB; the layout is in the README there).

**The data.** Five linear regressions with a dense design and fixed seeds:
n = 100 with p = 10 (independent columns) and p = 200 (AR(1) correlation
0.9^|j - k|, p > n), n = 1000 with p = 50 (AR(1), 0.5), n = 1e4 with p = 100
(equicorrelated, 0.5) and n = 1e5 with p = 10 (AR(1), 0.8). The true
coefficients are sparse (max(3, p/10) nonzero, from 2 down to 0.5 with
alternating signs) and the intercept is 0.5. Each dataset carries two
responses on the same X, one with N(0, 1) errors for the penalized fits and
one with Laplace(0, 1) errors for LAD. The two smallest are stored as hex CSV,
the others as raw little-endian binary64, column-major.

**The objective in Laplace's units.** glmnet, family "gaussian", with an
unpenalized intercept b0 and `standardize = FALSE`, documents its objective as

  F_g(b0, beta) = RSS / (2 n) + lambda_g (alpha |beta|_1 + (1 - alpha) |beta|_2^2 / 2).

The optimizers work with sums, not means, so the reference objective is
F_L = n F_g:

  F_L = RSS / 2 + a |beta|_1 + c |beta|_2^2 / 2,   a = n lambda_g alpha,   c = n lambda_g (1 - alpha).

RSS / 2 is the gaussian negative log-likelihood with sigma = 1 less
n log(2 pi) / 2. The penalty is, on each coefficient, `enet_distrib`'s with
lambda = n lambda_g and the same alpha (its rates are a = lambda alpha and
c = lambda (1 - alpha)) less its log-normalizer A; at alpha = 1 it is
`laplace2_distrib`'s with rate lambda = n lambda_g (or `laplace1_distrib`'s
with sigma = 1 / (n lambda_g)) less -log(lambda / 2). Both constants are in
`glmnet_fits.csv`, so that a Laplace objective with the penalty's normalizing
constant included can be compared as well.

**glmnet's ridge term.** glmnet divides y by its standard deviation s_y
(divisor n) before fitting and divides lambda by s_y with it; the lasso term is
unchanged by this, but the ridge term comes out divided by s_y: what it
minimizes is RSS / (2 n) + lambda alpha |beta|_1 + lambda (1 - alpha) |beta|^2 / (2 s_y).
Measured on a test problem with s_y = 3.4996, the ridge weight implied by the
KKT conditions at glmnet's point was 0.285745 times lambda (1 - alpha), which
is 1/s_y to seven digits. The script therefore calls glmnet with
lambda' = k lambda_g and alpha' = alpha / k, k = alpha + s_y (1 - alpha), which
makes its objective F_g at (lambda_g, alpha) exactly; k = 1 for the lasso.
Before this correction the elastic-net fits violated the KKT conditions of F_L
by up to 1.4e4 (in F_L's units, at n = 1e5); after it, by at most 7.5e-3,
3.3e-5 relative to a.

**Polishing.** glmnet stops on a relative change (`thresh = 1e-14` here, with
`maxit = 1e8`), and its coefficients still carry errors of up to 8.3e-5 (the
p > n dataset at the smallest lambda). Each fit is therefore polished: with
glmnet's support S and signs s, the KKT system on S, (Xc_S' Xc_S + c I) beta_S
= Xc_S' yc - a s_S with Xc and yc centred and b0 = mean(y) - mean(X)' beta, is
solved directly with one step of iterative refinement. The polished point is
accepted only when its signs agree with s and every coordinate off S satisfies
|Xc_j' r| < a strictly; then it is the minimizer of F_L up to rounding. All 30
fits (5 datasets, alpha = 1 and 0.5, lambda = 0.5, 0.1 and 0.01 times glmnet's
lambda_max) were accepted, with the same support as glmnet's, KKT violations
of 1e-15 to 2e-9 (in F_L's units, against a from 0.70 to 6.6e4) and supports
from 1 to 113 coefficients. The smallest slack off the support, min (a -
|X_j' r|), is 1.0e-3 (p = 200, alpha = 0.5, the smallest lambda): a test that
reads the support from a computed point must resolve the subgradient below
that.

**LAD.** quantreg's `rq.fit(cbind(1, X), y, tau = 0.5, method = "br")` solves
the LAD problem, min sum |y_i - b0 - x_i' beta|, as a linear program by the
simplex method of Barrodale and Roberts, so its solution is a vertex at which
p + 1 residuals are zero. The script takes those p + 1 observations as the
basis B, re-solves the square system X1_B coef = y_B, and certifies optimality
through the dual on the basis, u_B = -(X1_B')^(-1) X1_N' sign(r_N): the point
is optimal when max |u_B| <= 1, and unique when the maximum is below 1. It
was 0.89, 0.991, 0.993 and 0.94 on the four datasets with p < n, so all four
solutions are unique; the re-solved vertex differs from rq's coefficients by
at most 5.6e-15. The dataset with p > n has no LAD reference. `rq_fits.csv`
also gives sigma_hat = sum |r| / n and the `laplace1_distrib` negative
log-likelihood at (beta, sigma_hat), n + n log(2 sigma_hat), the reference for
the Laplace regression with sigma fitted by block alternation.

**What is open.** The SCAD and MCP references from ncvreg (installed, 3.16.0)
are added to the same script when the penalties land them (note/penalties.md,
step 5).

## Brent's method and block alternation (2026-11-18)

**What.** Two optimizers of wave 2 (note/optimizers.md section 12, decisions
2 and 6): `brent_optimizer` (`include/laplace/optimizers/brent.hpp`), which
minimizes a function of one variable from its values alone, and
`block_coordinate_optimizer` (`block_coordinate.hpp`), which splits x into
contiguous blocks and minimizes each in turn with its own sub-optimizer while
the other coordinates are held. With them, one stopping rule for every method
without derivatives, `search_size_stop_rule` in `stop_rules.hpp`, and a field
`search_size` in `stop_rule_input` and in `optimizer_report`.

**Brent.** The bracketing phase evaluates x0 and x0 + h (h = `initial_step`,
default 1), turns round when the second value is the higher, and expands by
the golden ratio until the value no longer falls; a non-finite value is
+infinity, so a refused point ends the expansion on its side. Without a bracket
after `max_expansions` (50) the run stops at the lowest point seen with
`iteration_budget`, or `evaluation_budget` when the budget ran out first. The
reduction is Brent's (1973, chapter 5): the vertex of the parabola through the
three best points when it lies in the bracket and moves less than half the step
before last, a golden-section step into the larger part otherwise. Its
stationarity measure is the half-width of the bracket, in the units of x;
`search_size_stop_rule` declares convergence when it is at most
`absolute_tolerance + relative_tolerance max_i |x_i|` (defaults 1e-8 and 0).
Brent's own test, the bracket within sqrt(eps) |x| + 1e-12 of the best point,
ends the run as converged at resolution, since no further point can be told
apart from x there. On the test functions: (x - 2)^2 + exp(-x) from 0 in 12
evaluations (half-width 3.1e-8, at resolution); sum |y_i - x| over seven points
from -20 in 34 evaluations, the median within the half-width 6.4e-8; x - log x
from 3, whose expansion crosses 0 where the function is NaN, in 16
evaluations; -x ends on the expansion budget, never converged.

**Block alternation.** Each block's sub-optimizer runs on a view of the
objective that evaluates the full objective and returns the block's slice of
the gradient and diagonal block of the curvature, so a block-aware objective
is not needed. The blocks are visited cyclically (Gauss-Seidel), each from the
current values (warm start). Solves are inexact: from the second sweep, a
sub-optimizer whose rule has a tolerance in the objective's units (a rule that
reads the decrement or the proximal gradient mapping) runs at
max(final, 0.1 D), where "final" is its configured tolerance and D is the
decrease of f over the previous sweep; the first sweep runs at the final
tolerances. Convergence is declared only on a sweep run entirely at the final
tolerances in which every block declared its own stationarity at its start
without moving beyond resolution: its sub-run succeeded, and it either left
the block unchanged, or moved it by at most twice its final search size (a
Brent block whose start lies in its final bracket), or moved it along a set
where f falls by at most 64 eps |f| (a flat set of minimizers). A sweep at the
final tolerances that moves nothing without converging ends the run as
stalled; otherwise the run ends after at most 100 sweeps. The report gives the
sweeps as iterations, the sub-runs' counts summed, the sum of the blocks'
decrements of the last sweep (where the sub-optimizers give one) and the
largest search size of the last sweep.

One repair made while testing: the first version ended as stalled any sweep
that moved nothing, and the coupled problem below stopped at sweep 2 with a
decrement of 0.93. At sweep 2 the tolerance 0.1 D, with D the large decrease
of the first sweep, let every block declare convergence at its start; a sweep
that moves nothing at loosened tolerances only says the tolerances were loose,
and the next one runs at the final tolerances because D is then 0.

**The Laplace against its closed form.** laplace1 (mu, sigma) and laplace2
(mu, lambda = 1/sigma), in the free coordinates (mu, log scale), mu by Brent
and the scale by Newton on the expected curvature, from the library start plus
1 on each free coordinate, on data drawn in C++ at mu = 1.5 and scale 2. The
closed form is mu at the median and the scale equal to the mean absolute
deviation from it; at even n every point of the median interval
[y_(n/2), y_(n/2+1)] is a minimizer, with the same mean absolute deviation.
All sixteen runs (both parametrizations; n = 1000, 1001, 1e5, 1e5 + 1) end
converged at resolution in 3 sweeps. At odd n mu is the median bit for bit; at
even n it lies inside the interval; the scale is within 1.6e-6 relative of its
closed form, inside the bound sqrt(2 tol / n) that the decrement tolerance
1e-6 of the Newton block allows. The same counts and outcomes on gcc and CUDA
(5 cases, 98 assertions on each build).

**Smooth problems against Newton.** f(x) = x'Ax/2 - b'x + sum_i log(1 +
exp(x_i)) in four coordinates, blocks solved by Newton and by BFGS: with A
diagonal (separable, blocks 2 + 2) in 4 sweeps; with A coupling every pair of
coordinates in 23 sweeps (blocks 1 + 3) and 12 (blocks 3 + 1), the linear rate
of Gauss-Seidel. The values agree with joint Newton within 1e-5.

**The negative case.** f(x1, x2) = |x1 - x2| + 0.1 (x1 + x2 - 2)^2, minimum 0
at (1, 1), from (0, 0): along each coordinate alone (0, 0) is a minimum, with
one-sided slopes -1.4 and 0.6. With both blocks solved by Newton the kink is
seen by the blocks' line searches, which fail, and the run ends stalled at
f = 0.4 after one sweep. A block solved by a method meant for kinks (Brent)
would declare itself stationary there, and the alternation would be declared
converged at a point that is not jointly stationary: the limit recorded in
decision 6, and the reason why a kink must stay inside one block. A test pins
it: with Brent on both blocks the run is declared converged after one sweep at
(0, 0) within 1e-7, f = 0.4 against the minimum 0.

**A composite block.** Over a composite objective (s + r, as
`penalized_objective`), the view is composite too: its value and derivatives
are those of s and its non-smooth part is r of the full point, so that a
proximal sub-optimizer's value s + r is f of the full point; its non-smooth
blocks are the penalties whose range lies inside the view's block, with the
per-coordinate members of proximal Newton's coordinate descent forwarded. A
penalty that straddles two blocks is refused, and so is a sub-optimizer that
never asks for the penalty of its block (the view counts the calls of the
non-smooth members); a sub-optimizer of a block without penalty reports s,
and r is added at its point. Measured on the lasso of the proximal tests
(gaussian regression, n = 500, p = 21 with the intercept unpenalized, weight
60 on the 20 slopes), the intercept by Newton and the slopes by FISTA in 11
sweeps or by proximal Newton in 12, sub-tolerances 1e-12: the same support as
proximal Newton on the whole of x (6 of 20 slopes) and the same value within
7e-16 relative (662.551358380276); the test checks the point within 1e-5 of
the largest coefficient and the value within 1e-8 relative.

**Measurements** (`benchmarks/optimizers_block_coordinate.cpp`, gcc 14.2,
OpenMP; n even, from the library start plus 1; time of the fastest of repeated
fits; passes are reductions over the data):

| distrib | n | method | sweeps or iterations | passes | 1 thread | 16 threads |
|---|---|---|---|---|---|---|
| laplace1 | 1e3 | Brent + Newton | 3 | 111 | 0.88 ms | 4.9 ms |
| laplace1 | 1e3 | Newton, expected curvature | 6 | 14 | 0.13 ms | 0.64 ms |
| laplace1 | 1e5 | Brent + Newton | 3 | 74 | 38 ms | 7.9 ms |
| laplace1 | 1e5 | Newton, expected curvature | 7 | 16 | 10 ms | 1.9 ms |
| laplace1 | 1e6 | Brent + Newton | 3 | 67 | 351 ms | 43 ms |
| laplace1 | 1e6 | Newton, expected curvature | 7 | 15 | 98 ms | 11 ms |
| laplace2 | 1e6 | Brent + Newton | 3 | 92 | 442 ms | 51 ms |
| laplace2 | 1e6 | Newton, expected curvature | 16 | 41 | 220 ms | 26 ms |

**What it says.** On the Laplace, where the expected information is smooth,
joint Fisher scoring is 2 to 7 times faster: Brent spends 20 to 30 value passes
per sweep shrinking a bracket on a kink, where the scoring step reaches the
median interval in a few passes. Both reach the closed form. Alternation is the
general route for a block with r_j = 0 and no usable expected information, as
decision 6 says, not the fast route here. The count of passes differs by a few
between thread counts, because the reduction's order changes the last bits of
f and with them Brent's path.

**What is open.** The CUDA timings of the bench were not taken (the CUDA
build ran the tests only).

## The proximal methods: FISTA and proximal Newton (2026-11-18)

**What.** The composite form of wave 2 (note/optimizers.md section 12,
decisions 1-5): f(x) = s(x) + sum_b r_b(x_b), s smooth and each r_b a penalty
with a proximal operator. Three headers in `include/laplace/optimizers/`:
`penalized_objective.hpp` (the adapter that attaches each penalty of
laplace::penalty to a contiguous range of x, with its theta held fixed, and
forwards value, prox_beta and weak_convexity), `proximal_gradient.hpp`
(`proximal_gradient_optimizer`) and `proximal_newton.hpp`
(`proximal_newton_optimizer` and the concept `coordinate_composite_objective`).
With them `prox_mapping_stop_rule`, the bit `stop_needs_prox_mapping = 8u` and
the field `prox_mapping` in `stop_rule_input`, and `damped_coordinates` in
`optimizer_report`.

**The adapter.** The penalized ranges must lie in [0, n) and must not overlap;
coordinates outside every range are unpenalized, and the prox is the identity
there. Each call recomputes the penalty's state from theta (a gather), so the
object holds no pointers into its own buffers. For proximal Newton the adapter
also gives the range of each block, whether the block is separable with its
kinks on the coordinates, and the prox of one coordinate. That prox is one
prox_beta of the whole block on a scratch copy, O(d_b) per coordinate: the
same order as one coordinate update with a dense H, so the penalties need no
per-coordinate member yet.

**FISTA.** A step from y with step t = 1/L is z = prox_{t r}(y - t grad s(y)).
It is accepted when s(z) - s(y) - grad s(y)'(z - y) <= |z - y|^2/(2t) + 8 eps
max(1, |s(y)|); otherwise t is halved, and t never grows. The first t comes
from one secant along -grad s(x_0). The momentum is Beck and Teboulle's. The
adaptive restart of O'Donoghue and Candes fires when f(z) > f(x) + 8 eps
max(1, |f(x)|); the step is then retried from x within the same iteration (the
slack is optimizers7's lesson: a bare comparison fires on the rounding of f
near the solution). With some rho_b > 0 the method is monotone FISTA (MFISTA),
with t <= 0.99/max_b rho_b also inside the backtracking.

One repair, found at n = 1e5 on one thread: s is then a serial sum of 1e5
squares, and its rounding exceeds the 8 eps |s| allowance. Near the solution
the test failed on rounding alone, t fell from 2.2e-7 to 1.1e-13 in 4000
iterations, and the run had not finished after 240 s. An excess below sqrt(eps)
max(1, |s(y)|) is now judged on gradients, as TFOCS does: the step is accepted
when (z - y)'(grad s(z) - grad s(y)) <= |z - y|^2/(2t). For a convex s this
implies the inequality above, and a difference of gradients has no rounding at
the scale of |s|. The same run now takes 370 iterations and 4.0 s.

**FISTA's stationarity measure (decided by Giovanni, 2026-11-18).** Decision 6
read the proximal gradient mapping G_t = (x - prox_{t r}(x - t grad s(x)))/t as
(1/2) t |G_t|^2. With t = 1/L this is smaller than f - f* by up to L/mu (mu the
smallest curvature), the defect found in BFGS's (1/2) g'Bg. Measured at a
declared 1e-8, on the lasso with equicorrelated columns (correlation 0.9,
p = 51), f - f* was 3.9e-6 at n = 1e4 and 4.5e-6 at n = 1e5. The measure is
now (1/2) gamma |G_t|^2, gamma = max(t, gamma_max), where gamma_max is the
largest secant ratio s's/s'y over the points at which grad s was evaluated:
the long Barzilai-Borwein step, which estimates 1/mu from below. With it, f -
f* at the declared 1e-8 is 4.2e-9 and 7.0e-9 (263 and 291 iterations, against
219 and 229). On the 30 glmnet fits below, f - f* is at most 1.11 times the
declared measure, so no confirmation on the next iteration (as BFGS has) was
added (decided by Giovanni on this measurement). With momentum
the step is taken from y; when the rule accepts a measure read at y, the
momentum is reset and the step retried from x, and convergence is declared
only on a measure read at the point reported.

A tolerance within a few eps |f| of the resolution of f makes FISTA crawl:
glmnet's d5 (n = 1e5, f = 5.8e4) took 86148 iterations at 1e-10, about 7 eps
|f|, and a resolution multiple of 8 in the rule did not help. The tests keep
the tolerance at max(1e-10, 1e3 eps |f*|).

**Proximal Newton.** At x the step d solves min_d g'd + d'Hd/2 + r(x + d)
inexactly, from d = 0. When every block is separable with its kinks on the
coordinates, the inner solver is coordinate descent on the model: coordinate
j goes to prox_{r_j/a_j}(z_j - (g_j + [H(z - x)]_j)/a_j) with a_j = H_jj,
O(n) per coordinate. A coordinate with H_jj <= rho_j is damped to rho_j/0.99
and counted in damped_coordinates. Otherwise the inner solver is FISTA on the
model, with step 1/L, L the Gershgorin bound on the largest eigenvalue of H,
and the gradient restart. The step is accepted on Armijo's condition on the
full objective, f(x + s d) - f(x) <= 1e-4 s Delta with Delta = g'd + r(x + d)
- r(x), halving s from 1.

The inner solve stops when its measure at the iterate is at most
inner_fraction times the current decrement (1/2) d'Hd. A repair, accepted by
Giovanni: the measure is (1/2) G'H^-1 G, where G is the model's proximal
gradient mapping at the iterate, with the Cholesky factor of H. The cheaper
measures, the last sweep's decrease or (1/2) t |G|^2, are smaller by up to the
condition number of H: on columns equicorrelated at 0.99 they stopped the
inner solve so early that proximal Newton ran 200 outer iterations without
converging; with (1/2) G'H^-1 G it converges in 12.

**The proximal Newton decrement (decided by Giovanni, 2026-11-18).** Decision
4 made the decrement (1/2) d'Hd NaN without a Cholesky factor of H. A lasso
with p > n always has a singular X'X, so proximal Newton could never declare
convergence there: on glmnet's d2 (p = 200, n = 100) it ended stalled or at
the iteration budget on all 6 fits. The factor is now that of H restricted to
the free coordinates: the unpenalized ones, and the penalized ones not held at
a kink at the subproblem's solution. Coordinate j counts as held when its
scalar prox returns z_j for targets moved by sqrt(eps) of their size to either
side of z_j - (g_j + u_j)/a_j. When H has no factor, the same restricted
factor also gives the inner measure, which otherwise falls back to the blind
(1/2) t |G|^2. An inner solve that misses its target inner_fraction (1/2) d'Hd
within max_inner_sweeps gives the stop rule max(decrement, inner measure /
inner_fraction), so a convergence verdict still needs the inner measure below
inner_fraction times the outer tolerance (a repair: discarding the verdict
instead never converged on d2's lasso at 0.01 of lambda_max, where the decrement
fell to 5.7e-17 while the inner measure, read through the factor of an
ill-conditioned block of 91 coordinates with n = 100, kept a rounding floor
above 0.01 times it). On d2, at the decrement tolerance 1e-10 and inner_fraction
0.01, 5 of the 6 fits now converge: the lasso at 0.5, 0.1 and 0.01 of
lambda_max in 4, 7 and 23 iterations, and the enet at 0.5 and 0.1 in 4 and 5.
Their coefficients lie within 2.4e-6 of glmnet's and f - f* is at most 1.8e-10.
The sixth, the enet at 0.01 of lambda_max, has 113 coefficients in its support
and the intercept, more than n. H on those coordinates is singular, and the run
ends stalled at 1.4e-7 from glmnet's point. The enet's ridge curvature c is
part of r, which the composite form does not expose to the optimizer.

**inner_fraction, measured.** Proximal Newton with coordinate descent on the
bench below, 0.1 against 0.01: identical on the independent designs (9
evaluations of s, 5 sweeps). On the equicorrelated design (0.9) the
evaluations of s go from 27 to 13 at n = 1e4 and from 29 to 13 at n = 1e5,
the inner sweeps from 648 to 756 and from 692 to 750, and the time on 8
threads from 5.9 to 5.1 ms and from 22.7 to 12.5 ms (on 1 thread from 11.8 to
8.1 ms and from 132.8 to 61.9 ms). An outer iteration costs a pass over the
data, an inner sweep O(p^2) without data. The default is 0.01 (decided by
Giovanni, 2026-11-18).

**Tests** (`tests/optimizers/test_proximal.cpp`, the objective in
`regression_objective.hpp`, the reader of the references in
`wave2_reference.hpp`). The regression objective is -l of a gaussian or
Laplace likelihood with a dense X, one Kokkos reduction of 1 + p sums per
evaluation, X'X formed once (the Laplace's curvature is the expected
information X'X/sigma^2). Internal cases: the lasso, the elastic net and the
lasso on columns equicorrelated at 0.99, by FISTA, proximal Newton with
coordinate descent and proximal Newton with FISTA inside, which agree on the
point within 1e-6 of max |b| and on the exact support, and satisfy the KKT
conditions and the prox mapping within 1e-5 of the kink weight. MCP, a weakly
convex penalty with rho = 1/gamma written in the test: with rho declared, MFISTA
and proximal Newton converge to stationary points and proximal Newton damps
210 coordinate updates; with rho wrongly declared 0, a run either fails or
stops at a stationary point (on this problem both converge correctly). The
adapter's value, prox, ranges and errors; gradient_stop_rule refused by both
methods at compile time.

Against the 30 glmnet fits frozen from R (section "Frozen R references for
wave 2 of the optimizers"), both methods from zero:

| | proximal Newton (1e-10) | FISTA (max(1e-10, 1e3 eps f*)) |
|---|---|---|
| exact support | 30 of 30 | 30 of 30 |
| converged | 29 of 30 (not d2 enet at 0.01) | 30 of 30 |
| coefficient error / max \|b\| | at most 2.4e-6 (d2), 3.3e-7 elsewhere | at most 9.6e-6 (d2), 1.4e-6 elsewhere |
| f - f* | at most 1.8e-10 | at most 1.11 times the declared measure |

At 1e-12, about 30 eps |f| on d2, proximal Newton ends stalled on d2's lasso at
0.01 of lambda_max (its point within 9e-9 of glmnet's): Armijo cannot show a
decrease below the rounding of f before the decrement reaches the tolerance.

**Bench** (`benchmarks/optimizers_proximal.cpp`, gcc 14.2, OpenMP 8 threads,
p = 51 with an unpenalized intercept, lambda = 0.1 n, or 0.02 n on the
equicorrelated design; tolerance 1e-8; ms):

| problem | n | FISTA | proximal Newton, coordinate descent | proximal Newton, FISTA inside |
|---|---|---|---|---|
| lasso | 1e4 | 3.7 (11 it) | 1.1 (4 it) | 1.5 (5 it) |
| lasso | 1e5 | 20.5 (9 it) | 5.9 (4 it) | 7.3 (5 it) |
| enet | 1e5 | 26.1 (10 it) | 8.0 (4 it) | 8.7 (5 it) |
| lasso, correlation 0.9 | 1e4 | 60.9 (263 it) | 6.4 (6 it) | 4.8 (6 it) |
| lasso, correlation 0.9 | 1e5 | 391 (291 it) | 12.9 (6 it) | 13.3 (7 it) |

Both proximal Newton columns at the default inner_fraction 0.01; final code,
one run.

**What is open.** (1) Tolerances near the resolution of f: within a few eps
|f| of it FISTA crawls (d5 at 1e-10: 86148 iterations) and proximal Newton
ends stalled (d2 at 1e-12), and the stop rule's resolution test (multiple 1, or
8) does not catch it; the tests keep FISTA at max(1e-10, 1e3 eps |f*|) and
proximal Newton at 1e-10. (2) Proximal Newton on the enet
with a support of n coordinates or more (d2 at 0.01 of lambda_max: 113 > 100)
ends stalled at 1.5e-7 from glmnet's point: H on the free coordinates is
singular, and the enet's ridge curvature c, which would make the problem
strongly convex there, lives in r and is not exposed to the optimizer. (3) On
CUDA the regression test objective computes the gradient with a team reduction
per column, since an array reduction of 1 + p sums exceeded CUDA's per-block
scratch at p = 201; host backends keep the single pass. The CUDA timings of the
bench were not taken.

## Nelder-Mead and compass search (2026-11-18)

**What.** The two multidimensional optimizers of wave 2 that use values only
(note/optimizers.md section 12, decision 2): `nelder_mead_optimizer`
(`include/laplace/optimizers/nelder_mead.hpp`) and `compass_optimizer`
(`compass.hpp`). Both report their stationarity measure as `search_size`, in
the units of x, and stop by default on `search_size_stop_rule` (converged
when the search size is at most 1e-8). Both work as block sub-optimizers of
`block_coordinate_optimizer`. Tests in
`tests/optimizers/test_derivative_free.cpp`, bench in
`benchmarks/optimizers_derivative_free.cpp`.

**Nelder-Mead.** The starting simplex is x0 and the n points x0 + h_j e_j,
with h_j = 0.1 max(1, |x0_j|), or a simplex given by the caller. Each iteration
sorts the vertices and then reflects, expands, contracts outside or inside, or
shrinks toward the best vertex (Lagarias, Reeds, Wright and Wright 1998), with
Gao and Han's (2012) coefficients 1, 1 + 2/n, 3/4 - 1/(2n), 1 - 1/n. At n = 2
these are the classical 1, 2, 1/2, 1/2; at n = 1 the classical ones are used,
because Gao and Han's shrink factor would be 0. A non-finite value counts as
+infinity. The search size is the diameter of the simplex from the best vertex,
max_j |x_j - x_0|_2. A shrink that leaves every vertex unchanged means the
simplex has collapsed to rounding, and the run ends as converged at
resolution.

**McKinnon's safeguard.** McKinnon (1998) gave a strictly convex C1 function,
(x <= 0 ? 360 x^2 : 6 x^2) + y + y^2 (tau = 2, theta = 6, phi = 60), on which
Nelder-Mead from his simplex performs inside contractions toward (0, 0), where
the gradient is (0, 1), while the minimum is -1/4 at (0, -1/2). The shape of
the simplex is watched: the conditioning |det E| / prod_j |E_j|, where E holds
the edges from the best vertex (computed by modified Gram-Schmidt), is 1 for a
right-angled simplex and 0 for a flat one, whatever its size. Below 1e-6 the
simplex is rebuilt right-angled at the best vertex, with edges equal to the
diameter reached, at most 3 times per run. Measured: without the safeguard
the run ends converged at (0, 0) with f = 0, after 108 iterations and 219
evaluations; with it the run reaches (0, -1/2) within 4.9e-9, after 142
iterations and 273 evaluations.

**Compass search.** A poll evaluates x +- delta d_j over the n columns d_j of
an orthonormal basis, 2n points in all. The default basis is a new random one
at every poll: the Householder reflection I - 2 w w'/w'w of a vector w drawn
uniformly in [-1, 1]^n from the 64-bit LCG of multistart, seeded with `seed`
(default 1), so that gcc and MSVC give the same polls. Its columns reach every
direction of the sphere over a run, which is the property the convergence
theory for a Lipschitz f needs (the idea of MADS, Audet and Dennis 2006). The
coordinate basis is the other option. With first improvement (the default)
a poll stops at its first lower value. After a success delta is doubled, up to
1e4 times its start; after a failure it is halved. The start is
delta = 0.1 max(1, max_i |x0_i|). The search size is the step of the poll that
just failed, the distance at which no lower value was found in any of the 2n
directions; it is not a number after a successful poll, since x has moved. A
failed poll whose trial points all round to x ends the run as converged at
resolution.

**The battery** (f* = 0; status 0 converged, 4 iteration budget; one thread,
gcc 14.2, fastest of repeats):

| problem | NM evaluations | NM ms | compass random | compass coordinate |
|---|---|---|---|---|
| Rosenbrock n = 2 | 261 | 0.054 | 10574 | 25328 |
| Rosenbrock n = 4 | 737 | 0.215 | 48499 | budget |
| Powell singular | 1444 | 0.408 | budget | budget |
| Beale | 145 | 0.043 | 1133 | 6531 |
| helical valley | 407 | 0.122 | 6878 | 11352 |
| Wood | 872 | 0.336 | 35312 | budget |
| quadratic n = 4, kappa 1e2 | 595 | 0.165 | 1241 | 825 |
| quadratic n = 4, kappa 1e4 | 688 | 0.193 | 32878 | budget |
| quadratic n = 10, kappa 1e2 | 2060 | 1.395 | 8525 | 23744 |

Nelder-Mead converges on every problem at f* = 0, 1, 1e3 and 1e6, with
|x - x*| at most 7.5e-8 at f* = 0. With f* = 1e6 the distance grows to the
resolution of f, up to 8e-6 on the problems whose minimum has a positive
definite Hessian, and 1.5e-3 on Powell's, where f grows as |x - x*|^4. The
random compass converges on all but Powell's singular minimum, where it
reaches the budget of 10000 polls with |x - x*| = 1.1e-2. It needs 2 to 70
times the evaluations of Nelder-Mead. The coordinate poll reaches the budget
on Rosenbrock n = 4, Wood and the kappa 1e4 quadratic (f - f* = 0.21 there),
because its fixed axes descend narrow valleys in small steps. The tests ask a
converged run to lie within max(1e3 search size, C (eps max(1, |f*|))^p), with
p = 1/2 (1/4 for Powell's) and C between 10 and 1e3, and a run that does not
converge to end with a budget status.

**Non-smooth.** sum |x_i| at n = 2 and 4, and sum |y_i - mu| over 7 and
8 points (the median, and an interval of minimizers): both methods converge
with the minimizer within twice the final search size. Nelder-Mead uses 127
and 471 evaluations on sum |x_i|, compass 290 and 977.

**As a block.** The Laplace (laplace1) of the fit_distrib bench by
alternation, mu by the method and log sigma by Newton on the expected
curvature, from the library start plus 1 on each free coordinate: 3 sweeps at
n = 1000 and 1001 (tests) and 1e5 (bench), mu inside the median
interval and sigma within 1.6e-6 relative of the mean absolute deviation. At
n = 1e5 on one thread: Brent 74 reductions in 37 ms, Nelder-Mead 187 in 91 ms,
compass 208 in 101 ms.

**The spread check.** On sum p_i^2 + 100 sin(1e20 p_1)
at n = 4 both methods first reached a converged verdict near the start
(|x| about 1, f about -96): the search size certifies only that no lower value
was found at that distance, and the noise has such points at every scale. A
convergence verdict, from the stop rule or from the method's own rounding
test, now stands only when the values that the final search compared differ by
at most `value_spread_tolerance` + 64 eps |f| (default 1e-6, in the units of
f): over the final simplex, max_j f(v_j) - min_j f(v_j); over the final failed
poll, max |f(x +- delta d_j) - f(x)|. A larger spread ends the run as
`stalled`. It costs no evaluation, since both sets of values are already at
hand. Measured at the verdict: 157 (Nelder-Mead) and 159 (compass) on the noisy
objective, which now ends stalled on both; at most 1.02e-8 on the 94 other runs
of the tests that reached a verdict (1.2e-10 on the smooth problems, up to
1.02e-8 at the kinks of sum |x_i| and of the median, where the spread is about
the search size times the slope); on least absolute deviations,
sum |y_i - b_0 - b_1 x_i| with n = 1e3, 1e4 and 1e5, at most 1.4e-8, all
converged. The term 64 eps |f| keeps the check from refusing a minimum whose
values are flat to rounding when |f| is large (the spread from rounding alone
is about eps |f|, 2.2e-10 measured at f* = 1e6). The default leaves a factor of
about 100 above the largest spread of a converged run and 1.6e8 below the
noisy one; a kink whose slope times the search size exceeds 1e-6 (slope above
about 100 with the default search size 1e-8) would be refused, which no test
problem reaches.

**CUDA.** Both methods run on the host; the whole laplace_tests passes on the
CUDA build, and the 4 derivative-free cases give 205 assertions, none failing,
on gcc and on CUDA.

## The proximal bundle method and least absolute deviations (2026-11-18)

**What.** `bundle_optimizer` (`include/laplace/optimizers/bundle.hpp`), the
method of wave 2 for a convex objective that is not differentiable everywhere
and has no simple proximal operator (note/optimizers.md section 12, decision
7), from the value and one subgradient per point (`gradient_objective`). The
case it was written for is least absolute deviations (LAD),
sum_i |y_i - x_i'beta| / sigma, the negative log-likelihood of the Laplace
distribution in its location with covariates. Tests in
`tests/optimizers/test_bundle.cpp`, bench in `benchmarks/optimizers_bundle.cpp`.

**The method.** Each point y_j visited gives a cut, the affine minorant
f(y_j) + g_j'(z - y_j); the cuts are kept as the subgradient and the
linearization error alpha_j at the stability centre x, updated on each move of
the centre by alpha_j += f(x_new) - f(x) - g_j'(x_new - x). The step is
d = -t p, p = G lambda, with lambda the solution of the dual over the simplex,
min (t/2)|G lambda|^2 + alpha'lambda; the model predicts the decrease
v = t|p|^2 + alpha'lambda, and the trial point is a serious step (the centre
moves) when f falls by at least 0.1 v, a null step otherwise. The proximity t
follows Kiwiel's rule (1990): the interpolated u_int = 2u(1 - fall/v), u = 1/t,
after two serious steps in a row or more than three null steps in a row whose
new cut has a large linearization error, a halving of u after more than three
serious steps, u kept within [u/10, 10u] per step and above 1e-10 u_0.
Kiwiel's variation estimate eps_v reads |p| + alpha_p, which mixes the units of
a gradient and of the objective; here it reads the stationarity measure below.
The first t is 1/|g_0|, a first step of length 1 in x (optimizers7 multiplied
the subgradient by a raw t0, and a first step 233 units long on Rosenbrock
left t unable to recover). A trial point where f is not finite is refused
(no cut, t / 10). When the bundle exceeds max_bundle_size cuts (default
n + 2, enough for a vertex in R^n) the cuts with lambda_j = 0 are dropped
first, then the aggregate cut (p, alpha'lambda), formed at the centre before
the step, replaces all but the newest; its linearization error is moved with
the centre when the step was serious (optimizers7 once stored the aggregate at
the new centre, where it can lie above f).

**The dual.** An active-set method of the kind of Wolfe's minimum-norm-point
algorithm: the working set holds cuts whose (g_j, 1) are linearly independent,
the equality-constrained problem on it is two Cholesky solves of
t G_F'G_F + s^2 11' (s^2 the mean of t|g_j|^2, which puts the two parts on one
scale), and a cut whose (g_j, 1) depends on the working set enters by
replacing one of them along a direction on which the dual falls linearly. A
cut enters when its dual gradient is below the current value by more than
1e-13 (t max_j Q_jj + max_j alpha_j), the rounding of a score that is a sum of
nearly cancelling terms; with 1e-13 max_j |score_j| the solve cycled to its cap
at large t. Measured against FISTA on the dual with the exact projection on
the simplex (step 1/(t trace G'G), the same solver for both dual solves; 16
OpenMP threads, best of three):

| dataset (n, p + 1) | active set: it, s | FISTA dual: it, s | FISTA: distance to the vertex |
|---|---|---|---|
| d1 (100, 11) | 61, 0.0036 | 62, 0.0282 | 4.8e-9 |
| d3 (1000, 51) | 211, 0.030 | 222, 0.72 | 2.8e-8 |
| d4 (10000, 101) | 446, 0.45 | 825, 10.6 | 9.6e-8 |
| d5 (100000, 11) | 74, 0.014 | 84, 0.049 | 5.1e-8 |

The active set is 4 to 24 times faster; FISTA converges, honestly, near the
vertex but not on it (within 5e-9 to 1e-7). The exact active set is the
solver (decided by Giovanni, 2026-11-18), and the FISTA dual, kept for this
measurement only, is not in the library.

**Stationarity.** The model is a minorant of f, so f(x) - f* is at most the
cutting-plane gap f(x) - min_z f_hat(z), which by duality is
min{alpha'lambda : G lambda = 0, lambda in the simplex}, infinite while 0 is
not in the hull of the cuts' subgradients. The measure is the model's
predicted decrease at a large proximity Gamma, v(Gamma) =
Gamma |p_Gamma|^2 + alpha'lambda_Gamma, from a second solve of the dual
warm-started from the first: it increases with Gamma towards the gap and falls
short of it by at most |d*|^2/(2 Gamma), d* the step to the model's minimizer.
Gamma = certificate_ratio max(gamma, t), with gamma the largest of t_0 and the
t of every serious step (the longest proximity at which the model was
confirmed); gamma never falls on a null step. Three measures were set aside on
measurement.
- (1/2) gamma |p_t|^2 + alpha'lambda_t, from the step's own dual, declared
  convergence with f - f* 5 to 54 times above it on d3, d4 and d5
  (f - f* = 9.7e-6 under 1.8e-7 on d3, 2.4e-4 from the vertex): near a vertex
  f - f* behaves like |p| times the distance, not like a quadratic decrement.
- The secant ratios s's/s'y between consecutive centres as a third term of
  gamma, as BFGS's gamma_max: on a polyhedral f they are 0 or infinite, and on
  the CUDA build one pair with s'y at rounding level set gamma to 1.2e6 on d4,
  so that Gamma |p|^2 held the measure at 2.2e-6 on the vertex itself
  (f - f* = 9e-12) for 100000 null steps, the iteration budget.
- A measure carrying the current t (optimizers7's -v) is driven under any
  tolerance by null steps that shrink t; on an objective finite only at its
  start the bundle method ends stalled with a measure of 141.

That CUDA run also showed a null step that cannot be resolved: at the vertex
the step's predicted decrease v was 9.4e-12, below the resolution of f
(64 eps |f| = 1.4e-10), every trial point gave f exactly and Kiwiel's rule left
t unchanged. Such a null step now multiplies t by 10, and ends the run as
stalled once t is at its cap. The ratio, LAD at the default decrement
tolerance 1e-6, 16 OpenMP threads (f - f*, with the measure in brackets):

| certificate_ratio | d1 | d3 | d4 | d5 |
|---|---|---|---|---|
| 10 | 4.3e-14 (5.1e-14) | 3.3e-12 (7.6e-7) | 6.0e-8 (8.9e-7) | 1.9e-10 (2.2e-10) |
| 100 | 4.3e-14 (5.1e-14) | 3.3e-12 (7.9e-7) | 3.2e-9 (6.1e-7) | 1.9e-10 (2.2e-10) |
| 1e3 | 4.3e-14 (5.1e-14) | 3.3e-12 (4.9e-7) | 3.3e-11 (1.1e-7) | 1.9e-10 (2.2e-10) |
| 1e5 | 4.3e-14 (3.4e-11) | 1.3e-12 (1.3e-12) | 6.0e-8 (5.3e-7) | 1.9e-10 (2.2e-10) |

Every run converged and every measure is above f - f*; the default is 100
(decided by Giovanni, 2026-11-18, with the certificate itself, the default
bundle size n + 2, the statuses above and the two repairs of the CUDA run).

**LAD against quantreg.** On the four wave-2 datasets with p < n (G2's frozen
references, the vertex re-solved from rq's basis), from beta = 0, active set,
bundle size n + 2 (8 OpenMP threads; f - f*, measure, max |beta - vertex|):

| dataset | decrement 1e-6: trial points, f - f*, measure, distance | decrement max(1e-10, 1e3 eps f*): the same |
|---|---|---|
| d1 | 61, 1.4e-14, 2.5e-14, 4.4e-15 | 61, 1.4e-14, 2.5e-14, 4.4e-15 |
| d3 | 214, 1.0e-12, 1.1e-12, 2.0e-13 | 214, 1.0e-12, 1.1e-12, 2.0e-13 |
| d4 | 411, 6.0e-8, 7.8e-7, 3.5e-6 | 438, 5.2e-10, 5.2e-10, 2.9e-11 |
| d5 | 74, 3.1e-10, 3.4e-10, 1.4e-10 | 74, 3.1e-10, 3.4e-10, 1.4e-10 |

Once the bundle holds the p + 1 cuts of the basis the model is exact at the
vertex, and the method lands on it; at the default rule it may stop just
before (d4). Over 1 to 16 threads the measure was above f - f* in every run but
two, both on d5 at f - f* about 1e-9 (1.08 and 1.3 times the measure), within
the rounding of a sum of 1e5 terms (sqrt(n) eps |f| about 7e-9); the test
asks f - f* <= 2 measure + 64 eps |f*|. At a decrement of 1e-10, two of eight
runs on d4 and d5 ended stalled: 64 eps |f| is 1.4e-10 and 1.4e-9 there, so
the tight runs use max(1e-10, 1e3 eps |f*|) as G1's FISTA test does. Bundle
sizes (16 threads): 2n + 2 converges as well, at 1.0 to 2.4 times the time of
n + 2; 10 cuts reach the budget of 20000 trial points on d3 and d4 and need
19465 on d1.

**The Laplace with covariates by block alternation.** laplace1's -l in
(beta, u = log sigma), e^-u sum|r| + n u + n log 2, beta by the bundle method
and u by Newton (block_coordinate_optimizer): 3 sweeps on each dataset,
beta within 3.6e-11 of the vertex, sigma within 1.2e-6 relative of
sigma_hat = sum|r|/n (the decrement tolerance bounds it by about
sqrt(2 tol / n)), -l within 1.7e-8 of rq_fits' laplace1 value.

**Newton with the expected information and BFGS on LAD** (decision 7's
measurement, for the engine of phase 7; from beta = 0, default settings, BFGS
with 5000 iterations; status 3 stalled, 4 iteration budget, 6 line search
failed):

| dataset | Newton, X'X: status, it, f - f*, distance | BFGS: status, it, f - f*, distance |
|---|---|---|
| d1 | 3, 111, 0.16, 3.6e-2 | 6, 31, 2.0e-3, 9.0e-4 |
| d3 | 4, 200, 0.24, 2.1e-2 | 6, 195, 7.1e-5, 1.1e-4 |
| d4 | 4, 200, 9.2e-2, 2.9e-3 | 6, 207, 3.2e-3, 5.1e-4 |
| d5 | 3, 56, 1.2e-4, 1.6e-4 | 3, 58, 1.3e-5, 4.5e-5 |

Neither converges on any of the four, and both say so: Newton's step from the
smooth X'X has no descent at the kinks (stalled or the 200-iteration budget,
with 1409 to 2303 values in its line searches), BFGS's Wolfe search fails at a
kink. BFGS gets nearer (f - f* 1e-5 to 3e-3); at n = 1e5 the kinks are dense
enough for both to come within 1e-4 of the vertex. For LAD the route is the
bundle method (or LAD's own linear programme); the expected information is a
route for the scale, not for the location.

**CUDA.** The method runs on the host; the sums over rows are the regression
objective's kernels. The whole laplace_tests passes on the CUDA build, and the
5 bundle cases (with the header case) give 72 assertions, none failing, on gcc
and on CUDA. On the device every LAD fit lands on the vertex at the default
rule (d4: 452 trial points, 0.42 s, f - f* = 9.1e-12; d5: 71, 0.044 s).

## Penalties through a linear map: c(D) and Condat's prox (2026-11-18)

What. `linear_map_penalty<P, Map>` (include/laplace/penalty/linear_map_penalty.hpp)
is pen(beta) = P(D beta) + c(D) for a separable penalty P on u = D beta, with
D = W = diag(w), w_j > 0 (`diagonal_map`), or the forward differences of
order K (`difference_map<K>`, (p - K) x p). Two numerical pieces are new: the
constant c(D) = -1/2 log det(D D^T) of the difference map, and the proximal
operator through first differences of a Laplace penalty (the 1-D total
variation). Everything else is the chain rule over P's derivatives and the
per-coordinate prox of P.

**c(D).** The closed form

  det(D_K D_K^T) = prod_{j=0}^{K-1} binom(p + j, 2j + 1) / binom(2j, j)

(p for K = 1, p^2 (p^2 - 1)/12 for K = 2) holds exactly for K = 1..4 and
p = K + 1..30, 40, 50 (SymPy, Bareiss determinant of the integer matrix
D D^T against the product as a rational; tools/reference/penalty_difference_logdet.py).
Its log is computed as a sum of logs of ratios (p + j - i)/(i + 1), O(K^2)
terms, with no cancellation at large p (differences of lgamma lose about
eps lgamma(p) in absolute terms). Against mpmath at 200 bits on the log of
the exact integer determinant (118 values): worst relative error 3.8e-16, the
same on gcc and CUDA.

The alternative, a Cholesky factorization of D D^T (banded, O(p K^2)), was
measured against the closed form (numpy, dense):

| p | K = 1 | K = 2 | K = 3 | K = 4 |
|---|---|---|---|---|
| 200 | 6e-14 | -2.8e-10 | -5.0e-8 | -1.0e-4 |
| 2000 | 4e-12 | -2.0e-8 | -0.073 | fails |

The condition number of D D^T grows as p^(2K): Cholesky loses the digits and
then breaks. Kept: the closed form; Cholesky only for a general sparse D if
one is ever offered.

Consistency. For P the product of N(0, 1/lambda) on the m = p - K
coordinates, pen = lambda/2 |D beta|^2 - m/2 log lambda + m/2 log 2 pi -
1/2 log det(D D^T), which is the intrinsic mvgaussian2 with Q = lambda D^T D
since |Q|_+ = lambda^m det(D D^T). Tested for K = 1, 2, 3, p up to 12,
lambda 0.3 and 2: value 7.9e-16, beta derivatives of orders 1, 2, 3 4.8e-16,
1.5e-16, 0, the theta derivative (d/dv = -v d/drho) 1.3e-15, mixed 5.6e-16
(relative; host only, weighted_sum_psdmat being a host structure).

**Condat's prox.** For K = 1 and P the lasso product of laplace1 (rate
a = 1/sigma) or laplace2 (a = lambda) at one shared rate, prox_t of
a |D beta - mu0|_1 is b0 + TV_{ta}(v - b0), with b0 the cumulative sum of mu0
(D b0 = mu0) and TV_w(y) = argmin_x 1/2 |x - y|^2 + w sum_i |x_{i+1} - x_i|,
computed by L. Condat's direct algorithm (IEEE Signal Processing Letters
20(11), 2013), transcribed from his reference C code: exact in a finite
number of steps, O(p) in practice, O(p^2) in the worst case. A prototype in
Python agreed with the dual problem solved by FISTA at gap 1e-12 to 1.1e-12
(p = 100) and 2.6e-11 (p = 1000).

Test (kernel, gcc and CUDA identical): the exact certificate. With x the
output and z_i = sum_{j<=i} (v_j - x_j), optimality is |z_i| <= t a for all
i, z_i = -t a sign((Dx)_i - mu0_i) where the jump is not zero, and
z_{p-1} = 0. On p = 2, 3, 5, 17, 100, 1000, 1e4, t = 1e-10, 0.05, 0.5, 3,
1e8, with and without mu0, the violations relative to sum(|v| + |x|) + t a
are 1.1e-16 (bound), 1.1e-16 (jumps), 9.0e-15 (closing). At t = 1e8 the
output is the mean of v within 6.0e-9 (vmin = y_0 - lambda rounds at the
scale of lambda), at t = 1e-10 every coordinate moves by 2 t a at most (to
a relative 1e-5, rounding of v - x at |v| ~ 1). Against the dual solved by
FISTA on the host to gap 1e-13, p = 10 and 100: 6.4e-13. Different rates
across the sets of the product give NaN by design (Condat's algorithm has
one weight); a weighted TV prox is not offered.

Speed (host, one thread, gcc -O2, the algorithm copied into a standalone
program; a step signal of 50 levels plus uniform noise, weight 0.05, 1, 20):

| p | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|
| ns per coordinate | 1.4 - 5.0 | 1.6 - 12 | 9.1 - 11 | 4.9 - 11 |

linear in p over three decades: 11 ms for p = 1e6 at worst, against 0.77 ms
for p = 1000 in the Python prototype (5 us here).

Rejected for K >= 2: an iterative prox by FISTA on the dual box
|z|_inf <= t a with step 1/4^K. Iterations to a gap of 1e-8 (Python
prototype): K = 2, 12476 (p = 100) and 55407 (p = 1000); K = 3, 78920 and
194066; K = 1 needed 7341 and 38804 to 1e-12. Far too many for a prox called
at every step of an outer method; for K >= 2 the prox is `none`, and
f(beta) + P(D beta) is split by a primal-dual optimizer through inner() and
map() (with norm_squared_bound() = 4^K >= ||D||_2^2 for its steps).

**Other checks.** Diagonal map against the scales moved onto beta (gaussian2
with v/w_j^2; laplace1 with sigma/w_j and mu0_j/w_j, an exact identity with
the constant): 7.1e-15 and 1.8e-15 on value and derivatives, the prox
identical. Student t1 on second differences: beta derivatives of orders
1, 2, 3 against central differences 1.3e-10, 6.2e-10, 9.4e-11. rho through
W = diag(0.5, 3): rho_P max w_j^2 = 2.19727 equals max(-pen'') on a grid.
All in tests/penalty/test_linear_map.cpp, 8 cases, 48 assertions, none
failing on gcc and CUDA (the whole laplace_tests green on CUDA).

Open. The dense derivative layout over p for the difference map (binom(p +
J - 1, J) doubles: 4 MB at p = 1000, J = 2; 1.3 GB at J = 3) where the true
Hessian is banded (p (K + 1)); a banded layout comes with the engine. P must
be separable.

## Frozen R references for penalties through a linear map (2026-11-18)

**What.** Adaptive-lasso, fused-lasso, trend-filtering and elastic-net-on-differences
fits produced in R once, so that the proximal optimizers can be tested on
`linear_map_penalty` (note/penalties.md section 5d) without R at test time.
Versions: R 4.6.0, glmnet 5.0, genlasso 1.6.1. The script is
`tools/reference/optimizers_wave2_linear_map.R`, run from the repository root
with `Rscript --vanilla` after `optimizers_wave2.R`; it writes about 2.1 MB to
`tests/data/reference/optimizers/wave2/` (layout in the README there).

**The adaptive lasso and glmnet's penalty factors.** On the five wave-2
datasets (y with N(0, 1) errors), the weights are w_j = 1 / |b_j|, b the OLS
coefficients on centred data, or on the dataset with p > n the ridge
coefficients of (Xc'Xc + 10 I) b = Xc'yc; they range from 0.50 to 6.1e4.
glmnet, called with `penalty.factor = w`, rescales the factors to
pf'_j = w_j p / sum_k w_k and minimizes RSS / (2 n) + lambda_g sum_j pf'_j |beta_j|.
In Laplace's units (times n) this is

  F_L = RSS / 2 + r sum_j w_j |beta_j|,   r = n lambda_g p / sum_k w_k,

the smooth part plus `laplace2_distrib`'s penalty at rate r on u = W beta
(`linear_map_penalty` over `diagonal_map(w)`), less its constants
-p log(r / 2) and c(W) = -sum_j log w_j, which `adaptive_lasso_fits.csv` gives
together. The rescaling was checked through the KKT conditions: glmnet's own
point satisfies them with kink weights r w_j to within 5.5e-3 (in F_L's units,
at n = 1e5; 1e-14 to 2.4e-5 on the other datasets), and violates them by 86 to
8.7e6 when read with the unscaled weights n lambda_g w_j. Each fit is polished as
in the wave-2 references (the KKT system on glmnet's support, accepted only with
the same signs and a strict inequality off the support): all 15 fits (lambda =
0.5, 0.1 and 0.01 times glmnet's lambda_max with these factors) were accepted,
with glmnet's support, supports of 1 to 59 coefficients, KKT violations of at
most 1.9e-9, and a smallest slack off the support of 0.026. glmnet's
coefficients differ from the polished ones by up to 5.4e-6.

**The differences and genlasso.** Two new datasets with no intercept and an
AR(1) design (correlation 0.5): n = 200, p = 50 and n = 1000, p = 200, each with
a response on piecewise-constant coefficients (four levels) and one on
piecewise-linear coefficients (two kinks), N(0, 1) errors. genlasso minimizes
(1/2) |y - X beta|^2 + lambda |D beta|_1 with no intercept, which is already
Laplace's objective: `laplace2_distrib` at rate lambda on each row of
u = D beta (`difference_map<K>`), less -log(lambda / 2) per row and
c(D) = -(1/2) log det(D D^T). Three penalties: the fused lasso (K = 1,
`fusedlasso1d`), trend filtering of order 1 (K = 2, `trendfilter` with
`ord = 1`), and the elastic net on first differences,
RSS / 2 + a |D beta|_1 + c |D beta|^2 / 2 with a = lambda alpha, c = lambda (1 - alpha),
alpha = 0.5, which is genlasso on the augmented data [y; 0], [X; sqrt(c) D] at
lambda = a (the penalty is `enet_distrib` at (lambda, alpha) on each row,
whose constant per row is in the file). lambda is 0.5, 0.1 and 0.01 times the
first knot of the path of the unaugmented problem.

**Polishing and the certificate.** genlasso's path is exact up to its linear
algebra, and its points are polished as well: with Z the rows where
|(D beta)_i| <= 1e-8 max(1, max|beta|) and s the signs on the others, the
objective is minimized exactly on {beta : D_Z beta = 0} (spanned, for K = 1,
by the indicators of the fused groups, so that the fused differences are exact
zeros; for K = 2 by an orthonormal basis of the null space of D_Z), and the
point is certified by the dual: X'(X beta - y) + c D'D beta + a D'u = 0 with
u_A = s_A and u_Z found by least squares. All 18 fits were accepted with
max |u_Z| < 1 strictly (0.929 to 0.9990 for the fused lasso, 0.939 to 0.9997
for the elastic net, 0.9982 to 0.99994 for trend filtering) and residuals of
the dual system of at most 1.1e-9 in gradient units. genlasso's coefficients
differ from the polished ones by at most 1.3e-12, and its objective exceeds
the polished one by at most 9e-12 relative. The fused-lasso fits have 1 to 5
nonzero differences, trend filtering 1 to 6 kinks, the elastic net 11 to 32
nonzero differences.

**The tests** (`tests/optimizers/test_linear_map_fits.cpp`, on the regression
test objective, gcc and CUDA, the whole suite green on CUDA). The adaptive lasso by proximal Newton (coordinate
descent inside, at tolerance 1e-10) and FISTA, 15 fits: the exact support in
every fit by both; proximal Newton in 2 to 7 iterations, coefficients within
1.1e-6 of the reference (relative to max |b|), |f - f*| at most 8.4e-11;
FISTA in 7 to 497 iterations (497 on the p > n dataset at the smallest
lambda), coefficients within 3.0e-6, f - f* at most 1.01 times its
stationarity measure. The fused lasso by FISTA with Condat's prox, 6 fits: the
fusion pattern exact (the prox returns fused groups as equal values, so the
differences off the pattern are zero exactly), 8 to 25 iterations,
coefficients within 6.1e-6, f - f* at most 0.99 times FISTA's measure but on
one fit, where it is 8.1e-10 on gcc and 9.3e-10 on CUDA against a measure of
5.6e-10: 14 and 16 ulps of f = 3.4e5, at the resolution of the objective. gcc and
CUDA give the same iteration counts and statuses, and the same errors to the
digits printed. Trend filtering and the elastic net on
differences have no prox through the map and are left to a primal-dual
method.

## The primal-dual method for a penalty through a linear map (2026-11-18)

**What.** `primal_dual_optimizer` (`include/laplace/optimizers/primal_dual.hpp`)
minimizes s(x) + g(D x), where s is smooth with a gradient, g is a convex
penalty whose prox exists in u = D x, and D is a linear map known through its
action, the action of its transpose and a bound B >= ||D||_2^2. It is the route
for the penalties of `linear_map_penalty<P, Map>` that have no prox in beta:
trend filtering (`difference_map<K>`, K >= 2) and the elastic net on
differences. The objective is `linear_map_penalized_objective<Smooth, P, Map>`,
built from the smooth part and one `linear_map_penalty` attached to a range of
x; it gives the method P's value and prox (`inner()`), and the map's apply,
apply_transpose and norm bound (`map()`). A penalty with rho > 0 is refused,
since the method needs g convex.

**How.** The iteration of Condat (2013) and Vu (2013) with the smooth part
linearized and no third term:

    x+ = x - tau (grad s(x) + D'y)
    w  = y + sigma D(2 x+ - x),  u' = prox_{g/sigma}(w / sigma),  y+ = w - sigma u'

The dual step uses Moreau's identity, so only P's own prox is called, and
y+ lies in the subdifferential of g at u'. The steps satisfy
1/tau - sigma ||D||^2 > L/2 through tau = 0.99 / (L/2 + sigma B), where L is
the Lipschitz constant of grad s. L is estimated from a secant along a short
gradient step at the start, and doubled whenever the descent lemma fails
between consecutive iterates (a non-finite s counts as a failure); it never
decreases, so after finitely many doublings the steps follow the fixed-step
theory. sigma starts at 50 L / B and is balanced between the two parts of the
stationarity measure below: when one part exceeds 1.5 times the other, sigma
is multiplied or divided by 1 + a_k, with a_0 = 0.5 and a_k multiplied by 0.95
at each adjustment, so that the steps settle.

**The stationarity measure.** With r_p = grad s(x) + D'y (exactly the
direction of the next primal step, so it costs nothing) and
e_d = g(D x) - g(u') - y'(D x - u') >= 0 (the linearization error of g at u',
one more value of P per iteration), convexity of s and g gives
f(x) - f* <= e_d + |r_p| |x - x*|. The measure is
V = e_d + max(gamma_max, tau) |r_p|^2, where gamma_max is the largest secant
ratio s's / s'y of grad s seen along the iterates (the same device as FISTA's
measure). It is in the units of the objective and is read by
`primal_dual_stop_rule` (default tolerance 1e-6) through a new field
`primal_dual` of `stop_rule_input`. Measured on a Python prototype first (exact
f* from Condat's prox or from the box-constrained dual solved by BVLS), then in
C++: the largest ratio (f - f*) / V in the tail of the runs is 1.00 on every
problem; with a factor 1/2 on the r_p term it reaches 1.06. The naive measure
(1/2)(tau |r_p|^2 + sigma |r_d|^2), with r_d = u' - D x the dual residual,
understates f - f* by factors of 1e3 to 1e14 and was rejected.

**Choices measured.** Iterations to V <= 1e-6 on the prototype:

| problem | fixed sigma = 50 L/B | adaptive (decay 0.95) | adaptive (decay 0.99) |
|---|---|---|---|
| regression n = 300, p = 50, fused, correlation 0.5 | 32882 | 1041 | 912 |
| same, correlation 0.95 | not converged in 2e5 | 9153 | not converged (f - f* = 305) |
| Laplace log-scale, p = 200, fused | 14899 | 2359 | 7932 |
| Laplace log-scale, p = 200, order 2 | 12799 | 11048 | 22733 |

A fixed sigma is fast only near a problem-dependent constant (sigma_0 = 5, 50
and 500 L/B were each the best on some problem, by factors of 10 to 30). The
adaptive rule is insensitive to sigma_0 within a factor of about 2 over 5 to
500 L/B. Balancing tau |r_p|^2 against sigma |r_d|^2 instead of the two parts of
V was worse on trend filtering (order 2, lambda 50: f - f* = 1e-2 after 5e4
iterations against convergence in 19303). Decaying a_k only when the direction
of the adjustment changes, or adjusting every tenth iteration, was slower on
every problem tried.

**Tests** (`tests/optimizers/test_primal_dual.cpp`, the sums of s in Kokkos
kernels, gcc and CUDA). Against FISTA with the exact prox in beta (Condat's TV
prox through `difference_map<1>`, the weighted soft threshold through
`diagonal_map`): fused lasso on an identity design (p = 200, lambda 2 and 20)
and on regressions (n = 300, 41 columns, intercept unpenalized, correlation 0
and 0.9), the adaptive lasso, the fused lasso of a Laplace log-scale
likelihood s(x) = sum x_i + |y_i| e^{-x_i}: at tolerance 1e-9 the primal-dual
values exceed FISTA's by 8.9e-10 to 1.0e-9, each equal to V to three digits,
and the points agree within 2.1e-7; 628 to 13739 iterations (FISTA 15 to 797).
Without a prox in beta, by the optimality conditions (z solving D'z = -grad s
in least squares, then z_i - c u_i in a d|u_i| coordinate by coordinate):
trend filtering of order 2 and 3 and the elastic net on second differences,
identity designs and a regression, residuals of D'z = -grad s at most 4.0e-10
and subgradient violations at most 1.2e-8 at tolerance 1e-8 (the Laplace
log-scale case of order 2 at tolerance 1e-6: 1.3e-5). The 18 genlasso fits
frozen in `tests/data/reference/optimizers/wave2` (fused lasso, trend
filtering of order 2, elastic net on first differences): on n = 200, p = 50
all nine converge at 1e-8 in 702 to 8104 iterations, with coefficients within
3.7e-8 of the polished reference (relative to max |b|) and f - f* at most
1.01 V. On n = 1000, p = 200 four of nine converge at their tolerance
(1e3 eps |f|) in 2580 to 5564 iterations; the other five do not within 1e5
iterations (trend filtering at fraction 0.01 stops at f - f* = 1.8, V = 2.0),
and end with iteration_budget and f - f* <= V; the test caps them at 1e4 and
checks the honesty of the status. An objective unbounded below
(s = c'x with sum c != 0, fused penalty) ends with iteration_budget and
V = 74 after 2e4 iterations; a non-finite start is refused. gcc and CUDA give
the same statuses, and iteration counts equal or within 7 of each other.

**Speed** (`benchmarks/optimizers_primal_dual.cpp`, gcc, 8 OpenMP threads,
time per run):

| problem | size | iterations | time | status |
|---|---|---|---|---|
| denoising, fused (FISTA with Condat's prox) | p = 200 | 9 | 0.09 ms | converged |
| denoising, fused | p = 200 | 1926 | 6.8 ms | converged |
| denoising, order 2 | p = 200 | 6324 | 23 ms | converged |
| denoising, order 3 | p = 200 | 1e5 | 385 ms | budget, V = 1.6e-3 |
| denoising, fused | p = 1000 | 1e5 | 1.5 s | budget, V = 1.0e-3 |
| regression, fused | n = 1e3, 41 columns | 6372 | 0.45 s | converged (FISTA: 86, 13 ms) |
| regression, order 2 | n = 1e3 | 39385 | 2.8 s | converged |
| regression, fused | n = 1e5 | 30571 | 12.4 s | converged (FISTA: 87, 72 ms) |
| regression, order 2 | n = 1e4, 1e5 | 1e5 | 11.5 s, 40.5 s | budget |

**What it says.** The method is correct and its measure is a tight, honest
bound on f - f* wherever it was checked, converged or not. It is first order,
with a rate of order 1/k, and slow when D'D is ill-conditioned (the condition
number of D_K D_K' grows as p^(2K)) or when s is: trend filtering of order 2 or
3 at p >= 200 and regressions with correlated columns need 1e4 to more than
1e5 iterations. Where a prox in beta exists (the fused lasso, adaptive
weights) FISTA with that prox is 35 to 170 times faster and gives exact zeros
in D x, which the primal-dual point does not (only u' has them).

**Open.** On the genlasso dataset with n = 1000 and p = 200, five of the
nine fits do not converge within 1e5 iterations: the three trend-filtering
fits (fractions 0.5, 0.1 and 0.01, ending at V = 2.7e-6, 2.5e-3 and 2.0) and
the fused lasso and the elastic net on differences at fraction 0.01 (V = 6.5e-3
and 4.2e-6), all with the honest status iteration_budget. A specialized method
for trend filtering, ADMM with the banded solves of (X'X + rho D'D) (Ramdas
and Tibshirani, 2016), is left for later; a
third, separable term h(x) with its own prox (the sparse fused lasso) is not
in the objective; the report gives x, not the exact zeros of u'.

## SCAD and MCP: scad_penalty and mcp_penalty (2026-11-18)

What. `scad_penalty` and `mcp_penalty` (include/laplace/penalty/nonconvex_penalty.hpp),
pen(beta) = sum_j q(|beta_j|) with theta = (lambda, a) or (lambda, gamma)
shared by all coordinates, lambda > 0, a > 1, gamma > 0. MCP (Zhang 2010):
q = lambda u - u^2/(2 gamma) for u <= gamma lambda, gamma lambda^2/2 beyond.
SCAD (Fan and Li 2001): lambda u for u <= lambda, lambda u - (u - lambda)^2
/(2(a - 1)) up to a lambda (Fan and Li's (2 a lambda u - u^2 - lambda^2)
/(2(a - 1)) rewritten), (a + 1) lambda^2/2 beyond. Class unnormalizable, C0 in
beta (kink at 0), C1 at the knots and in theta, rho = 1/gamma and 1/(a - 1).
Derivatives J any (J >= 3 zero), K <= 1; sign(0) = 0 at beta_j = 0, the inner
piece at the knots. No new special functions: piecewise polynomials.

**Prox.** MCP, t <= gamma: 0 for u <= t lambda, s (u - t lambda) gamma/(gamma - t)
up to gamma lambda, v beyond. SCAD, t <= a - 1: s (u - t lambda)_+ up to
lambda (1 + t), s (u - t (a lambda - u)/(a - 1 - t)) up to a lambda, v beyond.
Written with the exact differences gamma - t, a lambda - u and a - 1 - t
(as an independent R implementation does), not 1 - t/gamma or (a - 1) u -
t a lambda, which round first as t approaches the bound. Both forms hold at
t rho = 1 (the middle region is empty), accepted up to a relative 4 eps so that
t = 1/rho computed in floating point passes; t rho > 1 gives NaN. Checked
before writing (Python, 500 random cases per regime against a refined grid):
the closed forms reach the minimum within 5.3e-15 in the objective, and so do
the global minimizers for t rho > 1, not implemented (MCP the hard threshold at
lambda sqrt(t gamma), SCAD the better of two candidates).

**Tests** (tests/penalty/test_nonconvex.cpp, kernels on gcc and CUDA; cases
MCP (0.7, 1.5), (2, 3), (1, 0.4), SCAD (0.7, 2.2), (2, 3.7), (1, 1.3)):

| check | gcc | CUDA | bound |
|---|---|---|---|
| value against the original forms (relative) | 4.2e-16 | 4.2e-16 | 2e-15 |
| J = 1, 2 against central differences (h = 1e-6) | 1.2e-9 | 1.2e-9 | 5e-9 |
| K = 1, J = 0, 1, 2 against central differences | 1.7e-9 | 1.4e-9 | 5e-9 |
| prox optimality residual / max(1, abs(v)) | 2.2e-16 | 2.2e-16 | 1e-15 |
| prox objective above golden section (relative) | 5.9e-16 | 5.9e-16 | 2e-15 |
| rho against the grid maximum of -pen'' | 0 | 0 | exact |
| diagonal map against q(w_j abs(beta_j)) and the scaled prox | 0 | 5.2e-18 | 1e-16 |
| first differences, J = 1 against central differences | 2.4e-9 | 2.4e-9 | 5e-9 |
| 744 frozen R rows (value, gradient, q'', prox) | 1.5e-16 | 1.5e-16 | 5e-16 |

The conventions at 0 and at the knots hold by exact equality; value, J = 1
and d/dtheta are continuous across the knots to 1e-10 at relative distance
1e-12; t rho > 1 gives NaN; the scalar and per-coordinate prox overloads agree
exactly. 5 cases, 7816 assertions, none failing on gcc and CUDA (the whole
laplace_tests green on CUDA). The finite-difference sizes are the truncation
error of the differences.

Open. The properized versions (a Gaussian tail beyond the last knot, proper,
for REML) are distribs, written by the distributions work; their Z needs
Dawson's function F(x) = e^{-x^2} int_0^x e^{s^2} ds, not in Laplace yet.

## Dawson's integral and its derivatives to order 6 (2026-11-18)

**What.** `include/laplace/math/dawson.hpp`: Dawson's integral
F(x) = e^(-x^2) int_0^x e^(s^2) ds (`dawson`) and its derivatives F^(k)(x),
k = 1..6 (`dawson_d1` ... `dawson_d6`), for every real x. F is odd and
bounded, with a single maximum F = 0.5410442246351816 at
x = 0.9241388730045917; F(x) ~ x near the origin, F(x) ~ 1/(2x) as x -> inf,
and F'(x) = 1 - 2 x F(x) exactly. Needed by the normalizing constants of the
properized MCP and SCAD penalties,

  Z_MCP  = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma/2)) + e^(-gamma lambda^2/2) sqrt(pi V/2) ],
  Z_SCAD = 2 [ (1 - e^(-lambda^2))/lambda + e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2))
               + e^(-(a+1) lambda^2/2) sqrt(pi V/2) ],

with lambda > 0 the penalty strength, gamma > 0 the MCP curvature, a > 1 the
SCAD shape, c = a - 1 and V the tail scale: the arguments are positive and
grow with lambda, so the right half line and the large-argument behaviour are
what the use exercises, while the negative side is covered for correctness.
Nothing in `math/` gives F: there is no erfi, `erfcx` is the scaled complement
of the real error function rather than of the imaginary one, and the
incomplete gamma routines do not take the negative argument that
F(x) = (sqrt(pi)/2) e^(-x^2) erfi(x) would need through them.

**How.** Six candidates, in two groups. For the value alone:

- Rybicki's equally spaced sum, F(x) = (1/sqrt(pi)) sum_(n odd) e^(-(x - n h)^2)/n
  in the limit h -> 0, whose error falls like e^(-(pi/(2h))^2). The terms are
  paired around the even multiple of h nearest to x and both exponentials of a
  term follow a geometric recurrence, so the whole sum costs three exp calls.
  At h = 0.22 with 20 terms it reaches 4.2 ulp on [5/2, 8) and 4.2 ulp above 8,
  but 1.5 ulp1 and 3.0e13 ulp below 5/2, where the sum cannot resolve a small
  x, and it costs 138-142 ns on one gcc thread. Discarded: slower than both
  chosen routes and no route to the derivatives;
- the continued fraction of the asymptotic series,
  F = 1/(2 D) with D = x - (1/2)/(x - 1/(x - (3/2)/(x - ...))). At 16 levels it
  is the most accurate candidate above 8, 1.26 ulp at 22 ns on one thread,
  against 6.5 ulp at 25 ns for the chosen series, and 32 or 64 levels change
  neither figure. Below 8 it is useless (3.1e14 ulp on [5/2, 8)) and it gives
  the value only. Discarded for the second reason;
- published rational approximations (Cody, Paciorek and Thacher) were not
  measured: their coefficients are not available here, and a rational is a
  route for the value alone, since each derivative order would need its own
  fit. What a polynomial route can give was measured instead with mpmath's
  Chebyshev fits (`tools/reference/dawson.py --cheb`): on [0, 5/2] a fit of
  F(x)/x in x^2 reaches 3.6e-19 at degree 21, but its coefficients are those
  of the Taylor series of F to sixteen digits (1, -2/3, 4/15, -8/105, ...), so
  evaluating it cancels exactly as the Taylor series does; on [5/2, 7] a fit
  of F in x needs degree 31 for 2.9e-17 and degree 35 for 1.2e-19, while the
  same fit in 1/x^2 reaches only 8.0e-13 at degree 23, the branch point at
  1/x^2 = 0 sitting too close to the interval.

For the value and every derivative order:

- the alternating Taylor series F(x) = sum_n (-2)^n x^(2n+1)/(2n+1)!!,
  differentiated term by term. Its largest term exceeds the sum by a factor
  that grows like e^(x^2): 266 ulp and 33 ulp1 for the value below 5/2 and
  2.7e27 ulp on [5/2, 8); at order 6, 2.5e5 ulp1 below 5/2. Discarded above
  |x| = 2;
- the recurrence F^(1) = 1 - 2 x F, F^(k+1) = -2 (x F^(k) + k F^(k-1)) over an
  accurate value. It is the cheapest route to the derivatives and the one that
  fails: at large x the two terms of a step agree to about log2(x) bits, since
  x F^(k) ~ (-1)^k k!/(2 x^k) and k F^(k-1) ~ (-1)^(k-1) k!/(2 x^k), so order 1
  loses 2 log2(x) bits and each further order another log2(x). Measured, with
  the chosen value as its seed: order 1 reaches 4.85 ulp1 below 5/2, 7.57 ulp1
  and 969 ulp on [5/2, 8), and 3.53 ulp1 with 2.6e16 ulp above 8; order 6
  reaches 419 ulp1 below 5/2, 5.0e6 ulp1 on [5/2, 8) and 2.2e41 ulp1 above 8.
  Discarded;
- **the scaled series** (chosen below |x| = 8). With A_K(x) = e^(x^2) F^(K)(x),
  the identity A_(K+1) = A_K' - 2 x A_K turns the series of
  A_0(x) = int_0^x e^(s^2) ds = sum_m x^(2m+1)/(m! (2m+1)) into

    F^(K)(x) = e^(-x^2) sum_(m >= 0) b_K(m) x^n / m!,

  with n = 2m + 1 for K even and n = 2m for K odd, and

    b_0(m) =    1/(2m+1)                      b_1(m) =   -1/(2m-1)
    b_2(m) =    4/((2m-1)(2m+1))               b_3(m) =  -12/((2m-3)(2m-1))
    b_4(m) =   96/((2m-3)(2m-1)(2m+1))         b_5(m) = -480/((2m-5)(2m-3)(2m-1))
    b_6(m) = 5760/((2m-5)(2m-3)(2m-1)(2m+1)).

  One formula covers every m >= 0, with no exceptional leading coefficient,
  and no denominator vanishes, because 2m - j is odd for odd j. Each b_K keeps
  one sign from the first m at which all its factors are positive on, so the
  tail of every series has one sign and nothing cancels inside it; the leading
  coefficients that break the sign are at most four, and the points at which
  they cancel the tail are the zeros of F^(K) themselves. Every denominator of
  b_K is a window of consecutive odd numbers, so the ratio of two neighbouring
  terms is a single fraction, w (2m - 1 - 2 floor((K+1)/2)) / (m (2m + 1)) for
  K even and the same with 2m - 1 in place of 2m + 1 for K odd, with w = x^2:
  one division per term. The series is summed at the rounded w = fl(x^2) and
  multiplied by e^(-w) at the same w. The sum behaves like e^w, so the error
  that rounding x^2 causes in the two factors cancels to first order; an
  fma-corrected e^(-x^2) against a series summed at fl(x^2) would instead
  leave it in, about x^2 2^-53 relative, 18 ulp at x = 6. The cap is 320
  terms, against the x^2 + 5 |x| the series needs, 104 at |x| = 8;
- **the differentiated asymptotic series** (chosen from |x| = 8),

    F^(K)(x) ~ (-1)^K sum_m ((2m-1)!!/2^(m+1)) (2m+1)_K x^(-(2m+1+K)),

  with (2m+1)_K the rising factorial. Every term carries the sign (-1)^K, so
  nothing cancels here either. The ratio of neighbouring terms is
  w (2m + K - 1)(2m + K)/(4m) with w = 1/x^2, again one division per term, and
  the sum is stopped at its smallest term, where an asymptotic series is most
  accurate. At |x| = 8 that is about 25 terms.

Both series were derived and checked against SymPy's own derivatives in
`tools/reference/dawson_forms.py`: the defining identity, the two recurrences
on b_K, nine coefficients of every order against the series of
e^(x^2) F^(K)(x), nine terms of every order of the asymptotic series against
the term-by-term derivative, the seven values F^(k)(0) and the maximum of F.

**The threshold.** The asymptotic series needs a larger argument at a higher
order, by the rising factorial that multiplies its terms, so the |x| at which
it becomes the better route was swept over 5, 6, 6.5, 7 and 8. The smallest
value at which each order is as accurate as the series is 6 for the value and
for order 1, 6.5 for orders 2 and 3, 7 for orders 4 and 5, and 8 for order 6,
whose worst point at a threshold of 7 is x = 7.0014 with 2988 ulp. One
threshold for every order, 8, was chosen: it costs the value about 20 per cent
on a mixed argument (58 against 48 ns on one gcc thread) and nothing in
accuracy, and it keeps one constant instead of seven.

**Measurement.** References `dawson.csv` and `dawson_d1.csv` ...
`dawson_d6.csv` (`tools/reference/dawson.py`): the value from mpmath's erfi,
the derivatives from the defining recurrence, everything at 600 and at 700
bits and kept only where the two agree to 2^-100 relative, which absorbs the
cancellation of the recurrence (about 190 bits at order 6 at x = 1e8). Two
routes check the value at every point, Kummer's function
F(x) = x e^(-x^2) M(1/2, 3/2, x^2) and, below |x| = 12, the quadrature of the
definition itself: the largest relative difference over the grid is 7.0e-211.
2176 rows per order, none dropped: x log-uniform in [2^-45, 1e8], uniform in
[0, 12], around each candidate threshold, at the maximum of F and near the
first zeros of the derivatives, and the mirror image of a third of the grid.

The derivative of order k has ceil(k/2) simple zeros on the positive axis,
each with its mirror image (0.9241388730045917; 1.501975268; 0.595679356 and
1.955608092; 1.066311887 and 2.340530538; 0.474246676, 1.466192267 and
2.680438697; 0.878711253, 1.819221958 and 2.987984368, counted at 400 bits over
(0, 60]), and the grid holds all twelve, so the relative error is unbounded
there and the absolute error in ulps of 1 (ulp1) is the measure that carries
meaning below the threshold; above it, past every zero,
the relative error does. Maxima of the chosen routes:

| region | measure | k = 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| \|x\| < 8, series | ulp1 | 1.24 | 1.07 | 2.54 | 4.63 | 8.00 | 13.4 | 89.5 |
| \|x\| >= 8, asymptotic | ulp | 6.50 | 5.37 | 6.14 | 6.01 | 5.78 | 8.27 | 7.95 |

gcc/OpenMP and CUDA agree on every figure of the table and on every worst
argument. The value, which has no zero away from the origin, is within 9.68
ulp on gcc and 10.35 ulp on CUDA everywhere, its worst argument being
-5.2231447303876841 on gcc and -5.1502338383858 on CUDA. Split by region on
the grid before the zeros were added to it, the ulp1 of the series was 1.24,
1.07, 2.54, 4.63, 8.00, 12.9 and 89.5 below |x| = 5/2 and 0.61, 0.28, 0.36,
0.37, 0.86, 3.67 and 14.9 on [5/2, 8), which is where the cost of the route
changes rather than its accuracy. In the series region the relative error of each
derivative is read at one of its zeros, where it means nothing: 4.9e14 at
order 1 (x = 0.92413887300459174), 1.1e9 at order 2 (1.501975268), 4.7e9 at
order 3 (-0.595679356), 1.7e9 at order 4 (2.340530538), 1.2e11 at order 5
(1.466192267) and 4.2e9 at order 6 (2.987984368). The 89.5 ulp1 of order 6
is of the same kind: it is reached where F^(6) crosses 1 while its scale in
that region is a few hundred, so the error is below one ulp of the function's
own size. In the asymptotic region the worst arguments are
8.1294951781346469, 8.3084666620398711, -10.440565655072483,
9.0253180852636845, -8.9288483521814008, 9.3274832764819831 and
11.636480101888953 for the value and the orders 1 to 6, all close to the
threshold, where that series is the least converged.

ns per call (`benchmarks/math_dawson.cpp`, 2^18 arguments drawn from each
region; one gcc thread, OpenMP on 8 threads, and the RTX 4070 SUPER):

| route | region | one thread | OpenMP | CUDA |
|---|---|---|---|---|
| series | \|x\| < 5/2 | 48-55 | 7.5-9.0 | 2.9-8.6 |
| series | 5/2 <= \|x\| < 8 | 128-155 | 21-29 | 11-39 |
| asymptotic | \|x\| >= 8 | 24-26 | 3.4-4.2 | 1.2-7.7 |

**What this says.** The value is within 9.7 ulp everywhere and the derivatives
within 13 ulp1 to order 5, which is the accuracy of the elementary functions
they are built on; order 6 is within 90 ulp1, and within one ulp of its own
scale. The cost is set by the number of terms of the series, x^2 + 5 |x|, so
it grows from 50 ns near the origin to 155 ns just below the threshold and
drops to 25 ns above it. Nothing in either route cancels, which is what the
two single-signed tails buy: the recurrence that every textbook gives for the
derivatives loses 3 to 41 orders of magnitude of relative accuracy over the
same grid.

**Open.** The cost on [5/2, 8) is the weak point: 155 ns is three times the
cost of the same series near the origin and six times that of the asymptotic
series. Two ways down, neither taken: a threshold per order, 6 for the value
and for order 1 rising to 8 for order 6, which would serve the value at 25 ns
from |x| = 6 up and cost six more constants; or a polynomial of degree about
31 per order on that interval, which would need seven tables of 32
coefficients and a decision about where its relative accuracy near the zeros
is good enough. The 89.5 ulp1 of order 6 could be brought to a few ulp1 by
summing the scaled series in double-double, at about ten times its cost.

**Chosen.** This version. Tested in `tests/math/test_dawson.cpp` on both
builds: the absolute error below the threshold and the relative error above it
against the bounds of the table with a margin of about half again, the value's
relative error everywhere, the seven values F^(k)(0) exactly, the parity of
every order bit for bit, the two routes meeting at |x| = 8 within 1e-13 in
absolute value, and F'(x) = 1 - 2 x F(x) within 1e-14.

## The properized MCP and SCAD: mcp_distrib and scad_distrib (2026-11-18)

**What.** Two univariate laws continuous on the whole real line, the densities
whose negative logarithms are Zhang's MCP and Fan and Li's SCAD on an interval
around the location and a Gaussian beyond it. With `u = |y - mu|`, `lambda > 0`
and `V` written for `tail_sigma2`,

    MCP   q = lambda u - u^2/(2 gamma)                       u <= gamma lambda
          q = gamma lambda^2/2 + (u - gamma lambda)^2/(2 V)  beyond
    SCAD  q = lambda u                                       u <= lambda
          q = lambda u - (u - lambda)^2/(2 c), c = a - 1      lambda < u <= a lambda
          q = (a + 1) lambda^2/2 + (u - a lambda)^2/(2 V)     beyond

and `f(y) = exp(-q(u))/Z`. The original penalties are flat beyond their last
knot, so `exp(-pen)` has an infinite integral for every parameter value; the
Gaussian tail makes the law proper and leaves the penalty untouched on the inner
pieces, which is why `mcp_penalty` and `scad_penalty` and these two laws share
their inner region exactly. Parameters, named with Giovanni on 2026-11-18:
`(mu, lambda, gamma, tail_sigma2)` and `(mu, lambda, a, tail_sigma2)`, the
fourth the variance of the Gaussian TAIL and not of the law, on `(0, inf)` with
no further constraint.

**How. The normalizing constants.** Every piece has an exponent that is a
polynomial of degree two in `u`, so completing the square turns its integral
into an integral of `e^(s^2)` where the exponent is convex in `u` (the inner
piece of MCP, the middle piece of SCAD) and into the normal integral on the
tail. The first is Dawson's integral `F` (`math/dawson.hpp`), which is why these
laws needed it:

    Z_MCP  = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma/2))
                 + e^(-gamma lambda^2/2) sqrt(pi V/2) ]
    Z_SCAD = 2 [ (1 - e^(-lambda^2))/lambda
                 + e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2))
                 + e^(-(a + 1) lambda^2/2) sqrt(pi V/2) ]

Both are sums of positive terms, so nothing cancels, and `log_z` adds them in
logarithms, which keeps the tail's contribution where `e^(-gamma lambda^2/2)`
underflows (gamma lambda^2 above 1490). Measured against the quadrature of the
definition `Z = 2 int_0^inf e^(-q) du`: worst relative error 3.8e-59 at 60
digits over eight cases and 1.1e-48 at 400 bits over the twelve of the reference
script. That comparison is also the check that the density integrates to one.

**How. The derivatives.** `l = -q(u) - log Z`, `q` a polynomial of degree two in
`u` on each piece whose coefficients are rational in the parameters, so every
mixed partial is closed. `y` and `mu` enter through `u` only, so the component
with `a` derivatives in `mu`, `J` in `y` and `(b, m, n)` in `(lambda, p, V)` is

    -(-1)^a sign(y - mu)^(a + J) d^(a+J)_u d^b_lambda d^m_p d^n_V q
    - [a + J = 0] d^b_lambda d^m_p d^n_V log Z,

with `sign(y - mu)^t` read as 1 for even `t` and as the sign with `sign(0) = 0`
for odd `t`: at the kink the odd orders are the mean of the two one-sided
values, which is zero, and the even orders agree on the two sides and are taken
from the inner piece. At a knot the value and every derivative come from the
inner piece, as in the penalties. `q` is quadratic in `u`, so every component
with `a + J >= 3` is zero inside a piece. The mixed partials were derived with
SymPy and checked against SymPy's own derivatives for every `(j, i, m, n)` of
total order up to six, at sixteen points inside the five pieces
(`tools/reference/distrib_mcp_scad_forms.py`): worst relative error 1.4e-15,
which is the double arithmetic of the forms and not the forms.

The derivatives of `log Z` are not written out. In both laws `V` enters only
through the factor `sqrt(pi V/2)`, so `Z = 2 [A(lambda, p) + B(lambda, p)
sqrt(pi V/2)]` and the table of `Z` over the three parameters is assembled from
two tables over `(lambda, p)` and the derivatives of `V^(1/2)`; the tables come
from the Leibniz product, the exponential and the chain rule over multi-indices
and `log Z` from the logarithm of a table, the same recursions `enet_distrib`
uses for its own constant (`detail/properized_nonconvex.hpp`). Checked in the
same script against SymPy's derivatives of `Z` written with `erfi`: worst
relative error 1.2e-13 over five points at total order up to six, rising to 3e-9
for the sixth pure derivative in `p` at `lambda = 0.01`, where that derivative
is itself 2.7e-11 and the absolute error 8e-20. The whole C++ transcription
agrees with that checked mirror to 5.7e-14 over eleven cases, both laws, every
component of orders one to six and the mixed `(y, theta)` derivatives.

Two sizes, not accuracy. A table over three variables is packed triangularly,
over the multi-indices of total order at most `K` only, 84 entries at order six
instead of the cube's 343, and the chain rule keeps two tables of the order
below instead of `K + 1`: without both, an order-six call of `logpdf_dk` asks a
CUDA thread for about ten kilobytes of stack against the 8192 bytes the tests
set, and with them for about 3.9.

SCAD's `(1 - e^(-lambda^2))/lambda` and its six derivatives are taken from the
power series `sum_(k >= 1) (-1)^(k+1) lambda^(2k-1)/k!` below `lambda = 1` and
by the Leibniz rule over `-expm1(-lambda^2)` and `1/lambda` above it: the
Leibniz route cancels like `lambda^(-2)` per order, its terms reaching 7.2e8 at
`lambda = 0.1` and order six against a result of 21.

**How. The cumulative functions.** Closed, and free of cancellation on every
piece, because the integral from `u` to infinity is a sum of products of
positive factors and never a difference:

    MCP, u <= gamma lambda:  sqrt(2 gamma) e^(-q(u)) F((gamma lambda - u)/sqrt(2 gamma))
                             + e^(-gamma lambda^2/2) sqrt(pi V/2)
    MCP, beyond:             2 e^(-gamma lambda^2/2) sqrt(pi V/2) Phi(-(u - gamma lambda)/sqrt(V))
    SCAD, u <= lambda:       e^(-lambda u) (1 - e^(-lambda (lambda - u)))/lambda + M + T
    SCAD, middle:            sqrt(2 c) e^(-q(u)) F((a lambda - u)/sqrt(2 c)) + T
    SCAD, beyond:            2 T Phi(-(u - a lambda)/sqrt(V))

with `M` the whole middle piece and `T` the whole tail, and `F(y)` that integral
over `Z` below the location and one minus it above. The first piece of SCAD goes
through `expm1`, so nothing cancels as `u` approaches `lambda`. `log_cdf` and
`log_ccdf` take the small side in logarithms, where it reaches below the
smallest double, and the other side as one minus a plain ratio: the logarithm of
the small side carries the rounding error of its own exponent, one ulp of the
result per unit of `|log F|`, which at `u = 25.6` with `lambda = 1.3`,
`gamma = 3` and `V = 2` is 124 ulp and was measured as 86.

The derivatives of `log F` in the parameters, the expected information and the
quantile are left to the generic routes.

**Measured, gcc (Serial and OpenMP) and CUDA (RTX 4070 SUPER).**

| quantity | worst |
|---|---|
| the inner region against mcp_penalty and scad_penalty, the value | exact, bit for bit |
| the same, the derivatives in y and in the hyperparameters | 2 eps of max(value, 1) |
| log f and its orders 1 to 6 against mpmath, in ulp of the order's scale | 3.66, 11.4, 11.5, 8.75, 11.2, 41.7, 162 (orders 0 to 6) |
| log F and log(1 - F) against mpmath | 2.46 ulp of max(abs(log F), 1); 437 ulp relative |
| Z against mpmath | 1.33 ulp |
| cdf and ccdf against the quadrature of the definition | 6.8e-16 relative |
| the mass by the generic panel quadrature, every knot declared | 4.4e-16 |
| the same with the kink at the location as the only cut | 8.7e-11 |
| the normal limit of the log-density (gamma or lambda to 0) | 8.0e-14 absolute |
| cdf(quantile(p)) against p, generic quantile, p from 1e-8 to 1 - 1e-8 | 2.5e-15 relative |
| host against device, the sum of the components of each order | 1.18e-15 at order 1, 2.1e-16 to 5.5e-16 at the others |

**Open.** No closed expected information: the generic quadrature route gives it,
in the score form, which is the form that holds where a parameter's regularity
is finite. No `prox_y`, although the inner region's prox is the penalty's firm
and three-piece thresholds; nothing asks for it yet.

The 437 ulp of the relative measure of `log F` are not a defect of the route but
the conditioning of a Gaussian tail: the argument of `Phi` carries one ulp and
`log Phi` has slope `-z`, so the relative error of a tail probability cannot be
below about `z^2` ulp, `z` the standardized distance beyond the knot. In ulp of
`max(abs(log F), 1)`, which is what a reader of a log-likelihood sees, the same
rows are within 2.46.

**Chosen.** This version, tested in `tests/distrib/test_mcp_scad.cpp`: 15 cases
and 42378 assertions, the same count on the gcc build and on the CUDA build, both
green. The whole `laplace_tests` suite is green on both with the two laws and the
wider `distrib_kinks` in it (758 cases, 1471453 assertions on CUDA).

**Two things found on the way, both outside these two laws.**

`distrib_kinks` held two non-smooth points, and these laws have three and five
in `y`. Raised to eight with the single shared `continuous_order` kept, a law
whose points differ in regularity declaring the worst of them (Giovanni,
2026-11-18); `transformed_distrib`'s merge buffer and cap follow. What the knots
are worth as cuts, measured on the properized MCP: with the kink at the location
as the only cut the mass is out by 8.7e-11 against 4.4e-16, and the generic
expected information does not converge on any component, all ten NaN in both the
score and the hessian form, while with the knots declared the score form
converges on all ten and the hessian form on the nine it marks valid, its
`(mu, mu)` component being invalid where a regularity is zero.

mpmath's `quad` is not to be trusted on `int_u^inf e^(-q)` far out in the tail,
where the integrand spans hundreds of e-folds. At `lambda = 30`, `gamma = 2`,
`V = 2` and `u = 18` it returned 2.2011e-201 with the knot as the only cut and
2.1696e-201 with four more, against the value 2.1732381118224717e-201, on which
three routes agree: the same quadrature with the integrand factored by
`e^(-q(u))` and geometric subdivisions, at 60 and at 120 digits; Dawson's
integral through mpmath's `erfi`; and the asymptotic expansion of Dawson's
integral. The reference script now factors and subdivides, and it uses `log1p`
on the side of the cumulative function that is close to one, where `1 - r` at
400 bits rounds to one as soon as `r` falls below 2^-400 and its logarithm would
read as zero.

## SCAD and MCP fits against ncvreg; proximal Newton's fallback measure (2026-11-18)

**What.** The proximal methods on the non-convex penalties `mcp_penalty`
(gamma = 3) and `scad_penalty` (a = 3.7), against fits of ncvreg 3.16.0 (R
4.6.0) frozen by `tools/reference/optimizers_wave2_ncvreg.R` on the five
gaussian datasets of the wave-2 references, at 0.5, 0.1 and 0.01 of ncvreg's
lambda_max: 30 fits in `tests/data/reference/optimizers/wave2/`
(`ncvreg_fits.csv`, `ncvreg.csv`, `ncvreg_weights.csv`), tested by
`tests/optimizers/test_nonconvex_fits.cpp`. With them one change to
`proximal_newton_optimizer`: a fallback measure when the free block of H has no
Cholesky factor.

**The conversion.** ncvreg centres y, scales each column to sum(x^2)/n = 1
(scale s_j) and minimizes RSS/(2n) + sum_j q(s_j |beta_j|; lambda, shape) in the
original coordinates. Each piece of both q is homogeneous of degree 2 in
(u, lambda) jointly, so n q(u; lambda, shape) = q(sqrt(n) u; sqrt(n) lambda,
shape), and n times ncvreg's objective is

    RSS/2 + sum_j q(w_j |beta_j|; sqrt(n) lambda, shape),   w_j = |x_j - mean(x_j)|_2 = sqrt(n) s_j,

the gaussian test objective (sigma = 1) plus `linear_map_penalty<mcp_penalty
or scad_penalty, diagonal_map>` with weights w and theta = (sqrt(n) lambda,
shape), the intercept unpenalized. An unnormalizable penalty keeps its class
through the map, so the value adds no -sum log w_j.

**The references.** ncvreg runs a path of 100 log-spaced lambdas down to 0.01
lambda_max, the three fractions included, with eps = 1e-14. Its point is then
solved again exactly: on its support and with the piece of q of each u_j, the
stationarity is linear, (Xc_S'Xc_S - diag(d_S)) beta_S = Xc_S'yc - (w s k)_S,
with (k, d) = (lambda, w^2/gamma) on MCP's inner piece, (lambda, 0) on SCAD's
first, (a lambda/(a - 1), w^2/(a - 1)) on SCAD's second and (0, 0) on the flat
tails. All 30 polished points keep ncvreg's signs and pieces, satisfy the KKT
conditions within 2e-9 (in the objective's units) with a strict inequality off
the support, and have a positive smallest eigenvalue of that matrix: they are
strict local minima. The objective is convex (smallest eigenvalue of the
standardized X'X/n above rho = 1/gamma or 1/(a - 1)) on d1 and d4 only.

**The fits.** From x0 = 0: proximal Newton (tolerance 1e-10), MFISTA
(tolerance max(1e-10, 1e3 eps |f|)) and a multistart of MFISTA from 10 Latin
hypercube starts (spread 1). Stationarity is measured as the largest violation
of 0 in the Clarke subdifferential, divided by lambda max_j w_j.

| fits | proximal Newton | MFISTA | multistart |
|---|---|---|---|
| 24 (d1, d3, d4, d5; d2 at 0.5) | ncvreg's support; coefficients within 2.3e-6, abs(f - f*) <= 1.5e-10 | ncvreg's support; within 2.8e-6, f - f* <= 2.95 times its measure | ncvreg's point |
| d2 (p = 200 > n = 100) at 0.1 | MCP +3.29, SCAD -12.18 | MCP -1.35, SCAD -12.70 | MCP -1.35, SCAD -12.70 |
| d2 at 0.01 | MCP +0.555, SCAD +0.750 | MCP +0.639, SCAD +0.911 | MCP +0.639, SCAD +0.911 |

(f - f(ncvreg) in the objective's units in the last two rows.) On the 24 fits
every method reaches ncvreg's minimum, and so do all 10 starts of the
multistart on d2 at 0.5. On d2 below 0.5 there are other local minima (the 10
starts end at 8 (MCP) and 10 (SCAD) distinct optima at 0.1, at 8 and 6 at
0.01): every converged run is stationary (at most
5.6e-6), the multistart reaches ncvreg's value or a lower one at 0.1, and at
0.01 no route from 0 or from random starts reaches it. Measured without being
kept in the test: multistarts of 30 starts with spread 0.1 or 0.3 end between
+0.54 and +0.91 there, and a path warm-started along ncvreg's grid ends +0.045
(MCP) and -0.233 (SCAD); at SCAD 0.1 that path reproduces ncvreg's point
exactly, at MCP 0.1 it ends +4.80. The test bounds: on the 24 fits coefficients
within 5e-6 and abs(f - f*) <= 4e-10 for proximal Newton, within 1.5e-5 and
f - f* <= 4 times the measure + 64 eps |f| for MFISTA and the multistart; on d2
below 0.5 stationarity <= 1e-5 for every converged run, and at 0.1 the
multistart's value <= ncvreg's + 4 times its measure + 64 eps |f|.

The damping of proximal Newton never fires on these problems: in ncvreg's
scaling H_jj is about w_j^2, above the block's rho max_j w_j^2 since rho < 1
and the column norms are close.

**The fallback measure.** On d2 at 0.01, proximal Newton from 0 ended
"stalled" after 85 (MCP) and 98 (SCAD) iterations, with a NaN decrement, at
points stationary within 1e-8: the free coordinates (the unpenalized ones and
the penalized ones not held at a kink, here many in the flat tails) outnumbered
the rows, so their block of H = X'X had no Cholesky factor. From random starts
at 0.1, 9 of 10 runs stalled the same way, at points that interpolate the data
with coefficients in the flat tails. Now, when that factor does not exist and H
is finite, the decrement is the measure of `proximal_gradient_optimizer` at x,
(1/2) max(t, gamma_max) |G_t|^2 with G_t = (x - prox_{t r}(x - t grad s(x)))/t,
t = 1/L the step of the inner model (L the Gershgorin bound of H, at least
rho/0.99) and gamma_max the largest s's/s'y over proximal Newton's accepted
steps; it stays NaN only when H is not finite. The new field
`decrement_from_prox_mapping` of `optimizer_report` (false by default) says that
the decrement of the report is this measure; the tests check that it is true
exactly on the d2 fits at 0.01 here and on the elastic net below, false
elsewhere. Measured: d2 at 0.01 converges in 59 (MCP) and 71 (SCAD) iterations at
the same local minima, with stationarity 2.9e-6 and 2.2e-6; the elastic net on
d2 at 0.01 of lambda_max (support 113 > n), which could not converge before,
converges in 43 iterations with f - f* = 2.6e-12 and coefficients within
5.7e-7. The other 29 glmnet fits and the 28 other SCAD and MCP fits end with a
positive definite factor, so the fallback never stopped them and they are
unchanged.

**What is open.** No method here finds ncvreg's minimum on d2 at 0.01 from
random starts; a warm-started path, which is how ncvreg reaches its point, is a
route of the engine (penalties plan, section 7) and is not tested here. The
damping is exercised only by the synthetic MCP test of `test_proximal.cpp`.

## The univariate power exponential: powerexp_distrib (2026-11-18)

`powerexp_distrib(mu, sigma, p)`, the power exponential law, also called the
generalized normal of the first kind: `mu` real, `sigma > 0` the STANDARD
DEVIATION and `p > 0` the shape. With

    c(p) = sqrt(Gamma(1/p) / Gamma(3/p)),   s = sigma c(p),   z = (y - mu)/s,
    f(y) = p / (2 s Gamma(1/p)) exp(-|z|^p),

so that `Var Y = s^2 Gamma(3/p)/Gamma(1/p) = sigma^2` for every `p`: `sigma`
carries the whole spread and `p` only the shape of the tails, the convention of
`pseudohuber_distrib`. Decided with Giovanni before the work started, together
with the two conventions below.

**The three limits, with the constants.** `c(2) = sqrt(2)`, and the law is
exactly the Gaussian of standard deviation `sigma`; `c(1) = 1/sqrt(2)`, and it
is exactly the Laplace of scale `sigma/sqrt(2)`, whose variance
`2 (sigma/sqrt 2)^2` is `sigma^2`. As `p` grows, `Gamma(1/p) ~ p` and
`Gamma(3/p) ~ p/3`, so `c(p) -> sqrt(3)` and NOT to one: the limit law is the
uniform on `[mu - sigma sqrt(3), mu + sigma sqrt(3)]`, whose variance is
`sigma^2`. Measured: `c(1e7) = 1.732050907546` against
`sqrt(3) = 1.732050807569`.

**The form the derivatives are assembled in.** The log-density is written as one
function of `zeta = (y - mu)/sigma` and `p` plus two functions of one parameter:

    l = Lambda(zeta, p) + g(p) - log sigma,
    Lambda = -|zeta|^p exp(A(p)),   A(p) = -p lambda(p),   lambda(p) = log c(p),
    g(p) = log p - log 2 - (3/2) lgamma(1/p) + (1/2) lgamma(3/p).

`lambda` and `g` are EVALUATED in two equivalent forms, obtained with
`lgamma(v) = lgamma(1 + v) - log v`:

    lambda(p) = (1/2) (lgamma(1 + 1/p) - lgamma(1 + 3/p) + log 3),
    g(p)      = -log 2 - lgamma(1 + 1/p) - lambda(p),

in which the logarithms that cancel (`log(1/p) - log(3/p) = -log 3` inside
`lambda`, and `log p` against the `-log(1/p)` inside `lgamma(1/p)` in `g`) are
cancelled by hand and never formed. The reason and the measurement are below,
under "Where the loss comes from". The same identity, with
`psi(1 + v) = psi(v) + 1/v`, gives the derivative of `lambda` used by the
expected information, `kappa = (3 psi(1 + 3/p) - psi(1 + 1/p)) / (2 p^2)`, in
which the poles `3/(3a)` and `1/a` of `3 psi(3a) - psi(a)` have cancelled.

`Lambda` FACTORIZES over the three parameters,
`Lambda = -|y - mu|^p sigma^-p exp(A(p))`, so it is differentiated in all three
at once rather than through a table of `(zeta, p)` derivatives carried to
`(mu, sigma)`. With `fall(p, a) = p (p-1) ... (p-a+1)`,
`rise(p, b) = p (p+1) ... (p+b-1)`, `r = y - mu`, `L = log|zeta|` and `Y_k` the
complete Bell polynomial of the derivatives of
`V(p) = (p - a) log|r| - (p + b) log sigma + A(p)` (whose first derivative is
`L + A'(p)` and whose higher ones are `A^(m)(p)`),

    d^a/dmu^a d^b/dsigma^b d^c/dp^c Lambda
      = -(-1)^(a+b) sgn(r)^a |zeta|^(p-a) exp(A(p)) sigma^(-a-b)
        sum_(i+j+k=c) c!/(i! j! k!) [d^i fall(p, a)/dp^i] [d^j rise(p, b)/dp^j] Y_k,

where `|zeta|^(p-a) exp(A(p)) sigma^(-a-b) = exp(V)` is computed as
`exp(p (L - lambda(p)) - a L) sigma^(-a-b)`. The derivatives of `lambda` and of
`g` need those of `lgamma(1 + 1/p)` and `lgamma(1 + 3/p)`, one Faa di Bruno
composition each through `v = 1 + a/p` with the polygamma functions of `v`
outside (orders 0 to 5 at order six).

**Why not the usual location-scale route, and the first of two bugs it caused.**
`detail_location_scale::mixed_shape_dk` carries a table of mixed `(zeta, p)`
derivatives to `(mu, sigma, p)` by multiplying each derivative in `zeta` by a
polynomial in `zeta`. At the location a derivative in `zeta` is infinite and the
polynomial carries a factor `zeta = 0`, so the product is a NaN: the test that
asks for no NaN at `y = mu` caught 84 of them, in the `(mu, sigma)` mixed
components at `p = 0.5`, 0.95 and 1.05. The factorized form has no such product,
and its magnitude `|zeta|^(p-a)` depends on the number of derivatives in `mu`
alone, which is what makes the limits at the location elementary. The second bug
was in the assembly itself: a `sym_tensor<3, K>` holds the components of total
order exactly `K`, and `pos3(a, b, c)` with `a + b + c < K` is a position in the
smaller tensor, so a loop over `a + b + c <= K` writes several values into one
slot. At `p = 2`, `K = 4` the `(mu, mu)` entry of order two landed in the slot of
the four `mu` derivatives, which `fall(2, 4) = 0` then never rewrote: the test
read `-0.591716` where the fourth derivative of a parabola is zero. The
assembly now walks the pairs `(a, b)` with `a + b <= K` and takes
`c = K - a - b`, so every component is written once.

**Regularity, the one delicate point.** Since
`d^j/dz^j |z|^p = p (p-1) ... (p-j+1) |z|^(p-j) sgn(z)^j`, the derivative of
order `j` at `z = 0` is finite and zero while `j < p` and infinite once `j > p`,
so the regularity of `l` in `y` (and in `mu`) depends on `p`, a runtime value,
while `y_regularity` is `static constexpr`. Settled with Giovanni:
`y_regularity = 0`, the worst case over `p`, and the true order at runtime in the
`continuous_order` field of `y_kinks()` and `param_kinks(0, y)`. The order
returned is `ceil(p) - 1`, with `infinitely_smooth` at `p == 2.0` exactly and no
test for any other even integer, an exact equality on a runtime double being
fragile while `p` moves under an optimizer. `ceil(p) - 1` is `floor(p)` away from
the integers and `p - 1` at an integer: `floor(p)` everywhere, as the assignment
first put it, would claim one order more than exists at an odd integer, `C^1` at
`p = 1` where the law is the Laplace and the first derivative jumps. The same
formula was reached independently, from the homogeneity of `||r||^(2 beta)`, by
the session writing the multivariate law, where it reads `ceil(2 beta) - 1`
outside the integers. Being conservative at `p = 4, 6, ...` costs panel
granularity in the quadratures and nothing else. One consumer reads the static
field: `independent_product_penalty.hpp` sets `beta_regularity = D::y_regularity`
and takes `penalty_kink_geometry::per_coordinate` unless the law is
`infinitely_smooth`, so a power exponential used as a penalty always takes the
kink geometry, at `p >= 2` as well.

**The limits at the location.** A component with no derivative in `mu` is zero,
`Lambda` vanishing identically in `(sigma, p)` at `r = 0`; an odd number of them
is zero by the convention `sgn(0) = 0` of `laplace1.hpp`, the mean of the two
one-sided derivatives, whose magnitudes are equal; an even number `a` is zero
while `a < p`, `-(-1)^b fall(p, a) rise(p, b) exp(A) sigma^(-a-b)` at `a = p`
with no derivative in `p`, and infinite otherwise, with the sign of the term
carrying the highest power of `L`. Where `p` is an integer below `a`,
`fall(p, a) = 0` and the component is exactly zero, correctly: `|zeta|^p` is then
a polynomial of degree below `a`. One consequence is worth stating because it
surprises: `d^2 l / dzeta^2` at the location is 0 above `p = 2`,
`-2 exp(A)/sigma^2` at `p = 2` and infinite below it, so its derivative in `p` is
infinite at `p = 2`, where `l` itself is a polynomial in `y`. The test checks
those infinities with exact comparisons, on the device as well as on the host, so
that a guard that failed to fire could not pass as "large".

**Numerics.** `|z|^p` is formed as `exp(p (L - lambda(p)))` and never as
`exp(p L + A(p))`: for large `p` the two terms of the second form are huge and
nearly opposite, while in the first the difference is of order one. Each order
`a` carries its own exponential, `exp(p (L - lambda) - a L)`, rather than
`|zeta|^(-a)` times a common one, because the common one underflows near the
location while `|zeta|^(p-a)` with `a > p` is the large number that belongs
there. Where the sum over `(i, j, k)` is exactly zero, which happens when `p` is
an integer below `a`, the component is zero and the exponential is not
multiplied in, which would be zero times infinity.

**The cumulative functions.** Closed, through the regularized incomplete gamma:
with `t = |z|^p`, which is a `Gamma(1/p, 1)` variable under the law,
`F(y) = 1/2 + sgn(z) P(1/p, t)/2`, so the far side of the distribution is half an
upper regularized function and no tail is formed as a difference of numbers near
one half; the near side is `-log 2 + log1p(P(1/p, t))`. The quantile inverts the
same relation (`gamma_q_inv` below the median, `gamma_p_inv` above it). The
derivative TENSORS of `log F` and `log(1 - F)` in the parameters are NOT
supplied: they would need the derivatives of `P(1/p, t)` in its shape mixed with
those in `t`, and `incomplete_gamma_deriv.hpp` carries the shape derivatives to
order five, one short of the six that three parameters need. The generic
quadrature route answers for them.

**The expected information.** Closed in all six entries. With `a = 1/p` and
`q = |z|^p`, which is `Gamma(a, 1)` under the law, the scores in the natural
parametrization `(mu, s, p)` are `(p/s)|z|^(p-1) sgn(z)`, `(p q - 1)/s` and
`a^2 psi(1 + a) - a q log q`, and the moments of `q` and of `q log q` give

    I_mu mu = p^2 Gamma(2 - 1/p) / (s^2 Gamma(1/p)),
    I_s s = p / s^2,   I_s p = -(1 + psi(1 + a)) / (p s),
    I_p p = -a^4 psi(1 + a)^2 + a^3 (1 + a) (psi'(2 + a) + psi(2 + a)^2),

with `I_mu s = I_mu p = 0` by symmetry. The Jacobian of `s = sigma c(p)` carries
them to `(mu, sigma, p)` through `kappa = lambda'(p)`:
`I_sigma sigma = p / sigma^2`,
`I_sigma p = (p kappa - (1 + psi(1 + a))/p) / sigma`,
`I_p p = I^nat_p p - 2 kappa (1 + psi(1 + a))/p + p kappa^2`, and `I_mu mu`
unchanged, evaluated as `p Gamma(2 - a)/(s^2 Gamma(1 + a))` since
`Gamma(a) = Gamma(1 + a)/a`. `I_mu mu` is the integral of `|z|^(2p-2)` against
the density and is finite only for `p > 1/2`; at `p <= 1/2` the member returns
`+infinity` rather than a finite number, while the `(sigma, p)` block stays
finite for every `p > 0`. `I_sigma p` is exactly zero at `p = 2`: the standard
deviation and the shape are orthogonal at the Gaussian. The derivatives of `I` in
the parameters are left to the generic route.

**What was checked symbolically** (`tools/reference/powerexp_forms.py`, SymPy and
mpmath, nothing of the library's own code):

| what | checks | worst relative |
| --- | --- | --- |
| `l` against the logarithm of the density, and `A`, `g` against their definitions | 16 | 0 |
| `d^m/dp^m lgamma(1 + a/p)`, m = 0..6, by Faa di Bruno through v = 1 + a/p, a = 1, 3; p = 1/3, 1, 5/2, 40 | 56 | 0 |
| `A^(m)` and `g^(m)`, m = 0..6, in the shifted forms, against SymPy's derivatives of the original definitions through `lgamma(1/p)` and `lgamma(3/p)`; same four `p` | 56 | 0 |
| `A^(m)` and `g^(m)`, m = 1..6, through `chi` (the route for `p >= 2`), against the same derivatives; p = 2, 5/2, 4, 40 | 48 | 0 |
| the `(zeta, p)` table `d^c/dp^c d^j/dzeta^j Lambda`, every `c + j <= 6`; p = 1/2, 3/2, 2, 9/2; zeta = 3/7, 2, -5/3 | 336 | 0 |
| `d^i/dp^i fall(p, j)` from the integer coefficients; j = 0..6; p = 1/2, 2, 17/4 | 147 | 0 |
| the factorized tensor, all 84 components with `K <= 6`, both sides of the location; p = 1/2, 3/2, 2, 9/2; two `(mu, sigma)` pairs | 16 point sets | 1.6e-166 |
| the reductions at `p = 2` and `p = 1` | 6 | 0 |
| the closed `I` against the quadrature of the score products | p = 0.8, 1, 1.5, 2, 3, 6 | 1.3e-6 at p = 0.8, 9.5e-11 at p = 1, <= 2.9e-19 above |

The Taylor tables of `chi` are checked by their own script
(`tools/reference/powerexp_chi_table.py check`): `chi^(j)`, j = 0..6, evaluated in
double as the header does, at 80 points of each of the three intervals against
mpmath, within 6.3 ulp (order 6 near `x = 1.5`), 3.3 for every lower order.

The figure at `p = 0.8` is the quadrature's, not the closed form's: `I_mu mu` is
the integral of `|z|^(2p-2) = |z|^(-0.4)`, whose endpoint singularity the
quadrature resolves to about six digits there and to eleven at `p = 1`.

**The reference values** (`tools/reference/distrib_powerexp.py`): 180 points,
15120 rows, 0 dropped. The derivatives are SymPy's own symbolic derivatives of
the definition through `lgamma(1/p)` and `lgamma(3/p)`, evaluated through mpmath
(`sp.lambdify`) at 50 and at 80 digits; a row is kept when the two agree to
1e-30 of the largest component of its order, and all 15120 did. The route shares
no step with the library's, which goes through Faa di Bruno, the Bell polynomials
and the shifted forms. `|z|^p` is not differentiable symbolically through an
absolute value, so each point is built with the sign of `y - mu` fixed; the
points have `mu` dyadic, `sigma` a power of two and `zeta = (y - mu)/sigma` a
double of 20 significant bits, so that the library and the reference see the
same `zeta` in double arithmetic. The location itself, where several derivatives
are infinite, is not in the file: the test checks those limits directly. `p`
runs over 0.4, 0.75, 0.95, 1, 1.05, 1.5, 2, 2.5, 3, 4, 7.5 and 30, and `zeta`
over 0.03, 0.3, 1, -1, 2.5, -7 and 30, with `(mu, sigma) = (0.5, 1)`,
`(-2, 2^-10)` and `(1, 2^10)`.

**Worst scaled error against those references**, in ulp of the largest component
of the order at that point, the row of order 0 in ulp of the value, measured on
2026-11-19 with the `chi` route of the next paragraphs. gcc and CUDA (the default
build, not `nofmad`) agree to the digit in every entry; the bounds are those
figures times about one and a half:

| order | p < 2.5: worst (bound) | p >= 2.5: worst (bound) |
| --- | --- | --- |
| 0 | 49.5 (75) | 32.9 (50) |
| 1 | 39.7 (60) | 31.2 (50) |
| 2 | 60.9 (95) | 32.1 (50) |
| 3 | 440 (660) | 29.9 (45) |
| 4 | 102 (155) | 32.8 (50) |
| 5 | 678 (1050) | 355 (535) |
| 6 | 273 (410) | 35.9 (55) |

By `p`, over every order: 0.4: 188; 0.75: 440; 0.95: 107; 1: 78.5; 1.05: 678;
1.5: 27; 2: 25.8; 2.5: 26.3; 3: 10.6; 4: 355; 7.5: 13; 30: 35.9. Before the
`chi` route (2026-11-18, gcc): 2: 88.8; 2.5: 266; 3: 80.6; 4: 2.27e3 (483 on
CUDA); 7.5: 29.7; the region `p >= 2.5` read 266 at order 3, 2.27e3 at order 5
and 80.6 at order 6.

**Where the loss comes from, measured.** At the points of large error the
pure-`p` component of the derivative is the sum of the derivatives in `p` of
`Lambda` and of `g`, which are of opposite sign and nearly equal, both carrying
the derivatives of `lambda(p) = log c(p)`, the factor the standard-deviation
parametrization introduces. Over a grid of 15 values of `p` (0.3 to 30) and 8 of
`zeta` (0.01 to 30), the factor `(|Lambda^(K)| + |g^(K)|)/|Lambda^(K) + g^(K)|`
by which that sum loses significance has medians 1.1, 1.8, 2.9, 4.0, 8.0 and 8.7
at orders 1 to 6, 90th percentiles between 13 and 17, and maxima of 66 (order 1),
420 (order 2, p = 0.4), 486 (order 3), 677 (order 4, p = 0.8), 234 (order 5,
p = 4) and 256 (order 6); no point of the grid loses more than 1000. It is
caused by the rescaling: the same derivatives of the law in its natural scale,
at fixed `s` rather than fixed `sigma`, lose a factor 1 at those points (1.7 at
the order-5 one). The natural route does not escape it, though: the conversion
back to fixed `sigma` adds `lambda'(p) (p |z|^p - 1)` and its derivatives, and the
cancellation moves there.

Inside the shape part there was a second, avoidable loss, which the shifted
forms above remove. The derivatives of `lgamma(v)` at small `v` are dominated by
the pole of `-log v`, and those pole terms cancel exactly between `lgamma(1/p)`
and `lgamma(3/p)` in `lambda`, and between `lgamma(1/p)` and `log p` in `g`, but
only after both have been computed and rounded. Emulating the pure-`p` component
in double with every other quantity as the header computes it and the shape
derivatives taken exact gave 408 ulp at order 2, `p = 0.4` and 314 at order 5,
`p = 4`, against 1.08e3 and 2.92e3 measured on the first version, which went
through `lgamma(a/p)`. The two routes compared over the reference grid (12 `p`
times 6 `|zeta|`), both emulated with the same correctly rounded lgamma and
polygamma values so that only the formula differs, error of the pure-`p`
component in ulp of its exact value:

| order | median: through lgamma(a/p), through lgamma(1 + a/p) | p90 | max | shifted better / worse |
| --- | --- | --- | --- | --- |
| 1 | 2.84, 2.03 | 23.7, 10.6 | 393, 393 | 30 / 17 |
| 2 | 4.68, 2.97 | 11.7, 9.77 | 152, 152 | 35 / 21 |
| 3 | 5.67, 4.54 | 24, 26.9 | 212, 408 | 26 / 28 |
| 4 | 6.8, 5.09 | 28.8, 22.4 | 214, 136 | 33 / 25 |
| 5 | 16.5, 7.74 | 106, 78.5 | 5.7e3, 1.85e3 | 49 / 16 |
| 6 | 25.1, 10 | 126, 65.8 | 287, 213 | 47 / 18 |

Order 3 is a draw and every other order improves, the high ones most. Measured on
the library itself, the shifted forms took order 2 at `p = 0.4` from 1.08e3 to
60.9 ulp, orders 0 to 2 and 4 at `p >= 2.5` from 81-87 to 31-33, and order 5 at
`p = 4` from 2.92e3 to 2.27e3, while three maxima rose by a factor 1.5 to 2.3
(order 3 at `p = 0.75`, 188 to 440; order 3 at `p >= 2.5`, 130 to 266; order 6,
185 to 273), within the spread the table shows for order 3.

**The structural cancellation at large `p`, and the `chi` route that removes it.**
After the shifted forms the worst row was order 5 at `p = 4`, 2.27e3 ulp on gcc
and 483 on CUDA. Emulated with correctly rounded lgamma and polygamma values it
still lost 1.85e3 ulp, so the polygamma's own error accounted for a factor 1.2
only. The rest came from a cancellation inside
`A^(m) = -(p lambda^(m) + m lambda^(m-1))`: for large `p`,
`lambda = (1/2) log 3 + gamma/p + O(p^-2)` with `gamma` Euler's constant, and the
term `gamma/p` contributes `gamma (-1)^m m!/p^m` and its opposite to the two
products, while each next term of the expansion `c_k p^-k` contributes two
products that cancel to a fraction `1/(2m+1)`, `2/(2m+2)`, ... of their size. The
same terms cancel between `lgamma(1 + 1/p)` and `lambda` in `g`.

The cure is to never form the product. With `chi(x) = lgamma(1 + x)/x + gamma`,
which is analytic for `x > -1` and starts at `zeta(2) x/2`,

    A(p) = -(chi(1/p) - 3 chi(3/p))/2 - gamma - (p/2) log 3,
    g(p) = -log 2 - (1/2) log 3 - (3/(2p)) (chi(1/p) - chi(3/p)),

exact identities in which `gamma` has been removed by hand; `chi(a/p)` is
differentiated in `p` by Faa di Bruno through `x = a/p`, whose derivatives
`a (-1)^k k!/p^(k+1)` all have the sign `(-1)^k`, and `g` by Leibniz's rule
against `1/p`. The values `A(p)`, `g(p)` and `log c(p)` keep the shifted forms.
For `p >= 2` both arguments lie in `(0, 3/2]`, and `chi` is evaluated there
without a polygamma, from three Taylor expansions whose coefficients
`chi^(n)(c)/n!` were computed with mpmath at 60 digits: centre 0 on `[0, 0.15]`
(33 coefficients, `(-1)^(n+1) zeta(n+1)/(n+1)`), centre 0.4 on `(0.15, 0.65]`
(37) and centre 1.05 on `(0.65, 1.5]` (41), 111 doubles in
`LAPLACE_DEVICE_GLOBAL_TABLE`, outside the constant bank. Each order `j` is
Horner's rule on `sum_n a_n n!/(n-j)! t^(n-j)`, the seven recurrences advanced
in one loop over `n`; one pass carrying the Taylor coefficients at `x` by
repeated synthetic division is as fast but measured 4.0 ulp against 2.7 near
`x = 0`, and 547 against 355 ulp on the worst row. Below `p = 2` the route of the
shifted forms is kept: there `3/p > 3/2`, and at `p = 1.5` emulating the `chi`
route with exact `chi` values did not improve on it (18.7 against 16.7 ulp at
order 1).

Emulated over the reference grid with correctly rounded inputs (the current
route with correctly rounded polygamma values, the `chi` route with correctly
rounded `chi^(j)`, and the floor with `A^(m)` and `g^(m)` themselves correctly
rounded), worst over `|zeta|` of the pure-`p` component in ulp, order 5:
`p = 2`: 36.8, 7.2, 3.66; `p = 2.5`: 45.8, 13.8, 5.2; `p = 3`: 40.7, 26, 26;
`p = 4`: 1.85e3, 454, 314. Measured on the library, gcc and CUDA alike: order 5
at `p = 4` from 2.27e3 (483 on CUDA) to 355 ulp, order 3 at `p >= 2.5` from 266 to
29.9, order 6 from 80.6 to 35.9, and `p = 2` from 88.8 to 25.8 over every order;
no entry rose. What is left at the worst row is the final cancellation of 234
between the derivatives of `Lambda` and of `g`, on shape derivatives now right to
a few ulp; with every shape derivative exact the row would read 314.

Cost, one thread, gcc `-O3` (the test build), `p = 1.99` (old route) against
`p = 2.01` and `p = 4`: `shape_derivs<6>` 218 ns against 276 and 280 ns, all of
`logpdf_dk<6>` 960 against 1000 and 1030 ns, about 5 per cent.

**The other checks** (`tests/distrib/test_powerexp.cpp`), worst measured
difference, identical on gcc and CUDA except where the table says:

| what | worst |
| --- | --- |
| the value against the definition (`lgamma` and `pow` of the standard library) | 2.83e-15 relative |
| `p = 2` against `gaussian1_distrib`, every order, the `(mu, sigma)` block | 4.12e-16 relative |
| `p = 1` against `laplace1_distrib` of scale `sigma/sqrt 2`, every order | 1.76e-15 relative |
| the mass and the variance `sigma^2` by `panels_quad` with a cut at the location | within 1e-12 and 1e-10, the test's tolerances |
| the closed `I` against the reference file | 2.18e-15 relative, at p = 0.6, pair (p, p) |
| the closed `I` against the variance of the score and against the generic score-form route | 7.4e-16 relative |
| the closed `log F` and `log(1 - F)` against the generic quadrature route | 9.93e-16 relative |
| the quantile against its own cumulative function | 1.79e-14 relative |
| every order inside a Kokkos kernel, host against device, at the exact points y = -3 + i/4, as a fraction of the sum of the absolute values of the components | 0 on gcc; on CUDA 2.4e-16 (order 1) to 7.5e-15 (order 5) |
| the value, a mixed derivative, the cumulative functions, the information and the quantile inside a kernel, host against device | 0 on gcc; 7.0e-15 relative on CUDA |

Every derivative order is also checked against a five-point difference of the
order below, in the parameters, in `y` and mixed (`check_derivs` and the
three-parameter `check_y_derivs3`), at `p` from 0.45 to 5 and away from the
location.

**A test defect found on CUDA.** The first kernel test sampled
`y = -3 + 0.2 i`, which rounds: at `i = 17` it lands a few ulp from the location
on the host, and fused multiply-add puts it a different number of ulp away on the
device. There the derivatives grow like `|zeta|^(p-a)`, so a factor 2 in `zeta`
became a factor `2^(a-p)` in the component, and the host-device comparison read
0.32 at order 2 for `p = 1.6` (`2^0.4 = 1.32`), 20.1 at order 6 (`2^4.4 = 21`)
and 41.2 at order 6 for `p = 0.6` (`2^5.4 = 42`), while gcc, where host and
"device" run the same code, read 0. The kernel bounds of that version had never
been measured on the device. The points are now `y = -3 + i/4`, exactly
representable on both sides and never near the location.

**Builds.** gcc (Kokkos Serial and OpenMP): 13 cases, 25405 assertions, 0
failed. CUDA (sm_89, RTX 4070 SUPER, the default build): 13 cases, 25405
assertions, 0 failed, confirmed by a second build after the bounds were set.
Both again on 2026-11-19 after the `chi` route, with the same counts and 0
failed.

## The multivariate power exponential: mvpowerexp1 and mvpowerexp2 (2026-11-18)

**What.** `mvpowerexp1_distrib<S>` (scale matrix `Sigma = M(theta)`) and
`mvpowerexp2_distrib<S>` (inverse scale `Q = M(theta)`), the law of Gómez,
Gómez-Villegas and Marín (1998), proper class only, in
`include/laplace/distrib/mvpowerexp1.hpp`, `mvpowerexp2.hpp`,
`mvpowerexp_state.hpp` and the shared `detail/mvpowerexp_common.hpp`. With
`r = y - mu`, `delta = r' W r` (`W = Sigma^-1` or `Q`) and the shape `beta > 0`,

    l = -log Z(d, beta) -+ 1/2 log det M - 1/2 delta^beta,
    Z(d, beta) = 2^(1 + d/(2 beta)) pi^(d/2) Gamma(1 + d/(2 beta)) / (d Gamma(d/2)).

The typed parameter list is (mu vector, the matrix, beta scalar), beta last in
"the rest". Value and derivative blocks to order six in (mu, theta, beta) and
mixed, the draw, the mean and the covariance.

**The convention (Giovanni, 2026-11-18).** `Sigma` is the SCALE matrix and is
not rescaled to the covariance, as `mvstudent_t1`'s is not, so that `beta = 1`
is exactly `mvgaussian1` (and `mvgaussian2` for the second form). The
covariance is `Sigma 2^(1/beta) Gamma((d+2)/(2 beta)) / (d Gamma(d/(2 beta)))`,
factor 1 at `beta = 1`.

**How.** The derivative blocks by Faà di Bruno over the set partitions of the
mu and theta positions with at most two mu per part, with the kernel table

    H(n, c) = d_delta^n d_beta^c (-1/2 delta^beta)
            = -1/2 delta^(beta - n) sum_(j=0..c) binom(c, j) P_n^(j)(beta) L^(c-j),

`L = log delta` and `P_n` the falling factorial, whose beta-derivatives come
from `P_(n+1)^(j) = (beta - n) P_n^(j) + j P_n^(j-1)`; the derivatives of
`log Z` through the reciprocal chain rule
`d^n/dbeta^n F(1/beta) = (-1)^n sum_k F^(k)(1/beta) (n!/k!) binom(n-1, k-1) beta^(-n-k)`
with `F(u) = (d/2) u log 2 + lgamma(1 + (d/2) u)` and `polygamma`.
`tools/reference/mvpowerexp_forms.py` derives the constant by radial
integration (equal to the published `Z` in symbols, and to the quadrature of
the definition within 1.2e-60 over `d` in {1, 2, 3, 5} and seven betas) and
checks every closed form against SymPy's own derivative, the Faà di Bruno
assembly included at 5.5e-41. References: `tools/reference/distrib_mvpowerexp.py`,
45252 rows of `mpmath.diff` of the definition at 60 and 75 digits (stops if
they disagree by more than 1e-30), six structures, betas 0.3, 0.5, 0.7, 1, 2
and 6, `r` at the location, at 1e-3 of it, at order one and far in the tail,
and a near-singular log_cholesky point (`log L33 = -7.5`).

**Double-double from three Faà di Bruno positions on (decided by Giovanni).**
For `n = A + K_theta >= 3` the residual, `W_U r`, `r' W_U r`, the kernel table
and the partition sum go in double-double, as `mvstudent_t` does; orders one
and two stay in double. Two losses it removes, both measured: the partition sum
cancels 951-fold at the worst entry of an order-six mu block of form 1 at
`delta = 9.42`, `beta = 0.7` (153 ulp1N of the block's scale in double, 9.03
with double-double); and `exp((beta - n) log delta)` in double carries the
relative error `eps |(beta - n) log delta|` inherent in `exp(log)`, which
forming the product exactly removes. `delta^beta` itself is evaluated by `pow`,
which is exact at `beta = 1`.

The sum over `j` in `H(n, c)` can cancel at moderate `|log delta|`, where terms
of different powers of `L` have opposite signs. The worst entry of the kernel
table over `beta` in [0.3, 6], `delta` in [1e-12, 1e8] and `n + c <= 6` is
3.7e-14 relative to mpmath, at `delta = 1e-3`, `beta = 0.5`, `n = 2`, `c = 4`,
where the `L^4` term (-569) and the `L^2` term (572) leave about 3 (an earlier
note in this work placed it at `delta = 1e8`, `beta = 6` and blamed `exp(log)`
without locating it; that was wrong). It does not reach the blocks: those that
hold it are within 6.7 ulp of `max(|ref|, 1)` over `A <= 2`, up to six
derivatives in beta and `delta` from 1e-4 to 9, so no change was made.

**The kink.** `delta^beta = ||r||_W^(2 beta)` is homogeneous of degree
`2 beta`, so `l` stays continuous in `y` at `y = mu` to every order for a
positive integer beta and to order `ceil(2 beta) - 1` otherwise (checked against
SymPy). The multivariate interface has no runtime kinks member, so the mu
descriptor declares 0, the worst case. At `r = 0` every partition term is
homogeneous of degree `2 beta - A` whatever the partition, so the kernel
contributes exactly 0 when `2 beta > A`; the factor-by-factor product there
would read 0 times infinity.

**The numbers.** Worst ulp1N over every `(A, K)` with `A + K <= 6`, all six
betas and all points, per structure:

| structure | mvpowerexp1 gcc | CUDA | mvpowerexp2 gcc | CUDA |
|---|---|---|---|---|
| diagonal | 58.7 | 58.7 | 106 | 106 |
| ar1 | 4.65 | 4.65 | 8.08 | 8.08 |
| compound_symmetry | 6.02 | 6.02 | 5.67 | 5.67 |
| log_cholesky | 3.96e3 | 3.86e3 | 18.1 | 17.4 |
| scaled_ar1 | 17.4 | 21.7 | 8.27 | 8.59 |
| factor_analytic | 3.94 | 8.26 | 11.3 | 11.3 |

The log_cholesky entry of form 1 is the error of `Sigma^-1` itself, proved and
not inferred: every value above 1e3 is at the near-singular point
(cond(Sigma) = 3.5e7) in the `K = 0` blocks with `A = 2, 3, 4` at
`beta = 0.5`; the double `Sigma^-1` there has relative error 1.008e-9
(4.5e6 ulp), and feeding THAT matrix into an otherwise exact mpmath computation
of the same blocks costs 750, 1441 and 3694 ulp1N at `A = 2, 3, 4`, against
1.46e3, 1.54e3 and 3.96e3 measured. Form 2 takes `Q` directly and never
inverts, and stays within 18.1 at the same point. The diagonal entries are the
far-tail point (`||r|| = 17`) at `beta = 6` and orders five and six at `beta = 0.7`.

Other checks, the same on gcc and CUDA except where two numbers are given:
`beta = 1` against `mvgaussian1/2` within 0.884 ulp1N over 18 blocks per
structure on three structures (0 on several); form 1 at `theta` against form 2
at `-theta` over the diagonal structure 0 ulp1N; `log Z` against a quadrature
of the radial form 2.22e-16 (gcc) and 2.95e-16 (CUDA) relative; the mass
through `logpdf` in one dimension 2.22e-16; the `beta = 1/2` group-lasso form
0.794 ulp1; second derivatives in beta against central differences at
`h = 1e-5` 2.66e-9; draws within 0.00137 standard deviations on the mean and
0.0065 relative on the covariance over `beta` in {0.5, 1, 2}. The test runs
197221 assertions on both builds, all passing; the bounds are the measurement
times 1.5 on the default CUDA build and gcc.

**What is open.** The degenerate and improper classes (a positive semidefinite
structure) are not written. There is no `expected_info_dk`. The radial kink has
no place in the multivariate interface (no `y_regularity`, no runtime kinks),
which would need a change of its own across every multivariate law and wrapper.

## The optimizer benches on the GPU: where CUDA pays (2026-11-19)

**What.** The nine optimizer benches (`benchmarks/optimizers_*.cpp`) timed on the
gcc build (gcc 14.2, Kokkos OpenMP, 1 and 16 threads) and on the CUDA build
(CUDA 13.1 with MSVC 14.44 for the host code, Kokkos with relocatable device
code, RTX 4070 SUPER, whose FP64 rate is 1/64 of FP32), with both build slots
held for the whole run. Three benches are host code only (`foundations`,
`quasi_newton`, `second_order`): on the CUDA build they measure MSVC against gcc,
not the GPU. Six evaluate the objective in Kokkos kernels:

| bench | objective | kernels per call | host part | sizes |
|---|---|---|---|---|
| fit_distrib | -l of one distrib, 1 to 3 parameters | one parallel_reduce | Newton or BFGS step | n = 1e3, 1e5, 1e6 |
| block_coordinate, derivative_free (Laplace rows) | the same for laplace1 and laplace2 | one parallel_reduce | Brent, Nelder-Mead or compass on mu, Newton on the scale | n = 1e3, 1e5, 1e6 |
| proximal | gaussian regression, p = 51 | one pass over X by chunks of rows, then a small kernel adding the chunks (first measured: a reduce writing weights and a team reduce per column) | FISTA, or proximal Newton's coordinate descent on the model | n = 1e3, 1e4, 1e5 |
| primal_dual | gaussian regression, p = 41 | the same | Condat-Vu step with a difference map | n = 1e3 (default), 1e4, 1e5 |
| bundle | least absolute deviations, p = 11 to 101 | the same | dual QP on the bundle | n = 100 to 1e5 |

The split between the time spent inside the objective's calls (kernels,
launches and copies to the host) and the time of the host loop was measured with
a timer around each call of the objective, in a separate program outside the
repository that runs the same problems twice and keeps the second run. On CUDA
every call returns its result to the host, so it ends with the device idle and
the timer includes the whole device work.

**The fit_distrib objective (a few parameters, one reduction per call).** Time
per call measured by the timer, laplace1:

| n | gcc 1 thread | gcc 16 threads | CUDA |
|---|---|---|---|
| 1e3 | 8.6 us | 77 us | 26 us |
| 1e5 | 0.63 ms | 157 to 171 us | 48 to 52 us |
| 1e6 | 6.5 ms | 1.03 to 1.06 ms | 258 to 275 us |

(1 thread: time per fit over the number of passes of the bench.) ms per fit,
fastest of repeats, 1 thread / 16 threads / CUDA:

| problem | n = 1e3 | n = 1e5 | n = 1e6 |
|---|---|---|---|
| laplace1, Brent on mu + Newton on sigma | 0.84 / 5.1 / 2.0 | 38.8 / 7.8 / 2.5 | 363 / 43.5 / 11.5 |
| laplace1, Newton on the expected curvature | 0.12 / 0.67 / 0.28 | 10.2 / 1.9 / 0.63 | 97 / 10.9 / 2.8 |
| laplace2, Brent + Newton | 0.64 / 4.6 / 1.5 | 37.7 / 7.6 / 1.8 | 447 / 53 / 8.2 |
| laplace1, Nelder-Mead on mu + Newton | 1.6 / 10.5 / 4.0 | 92 / 23.2 / 6.9 | |

Over the 28 rows of the fit_distrib bench with a pass time (14 distribs, Newton
on the observed curvature and BFGS), the GPU against the faster of 1 and 16
threads: at n = 1e3 faster in 11 of 28, geometric mean 0.86 (range 0.44 to
2.1); at n = 1e5 faster in 26 of 28, geometric mean 3.1 (0.71 to 9.4); at
n = 1e6 faster in all 28, geometric mean 5.1 (2.5 to 11.6). With Newton on the
observed curvature at n = 1e6 the gain against 16 threads is 8.9 to 11.6 times
for cauchy, logistic, lognormal2 and gumbel and 3.5 to 4.1 times for gamma1,
gamma2 and chisq, whose rows call lgamma and digamma, as the FP64 rate of this
card suggests. From the bench's own pass times, the objective takes 70 to 100
per cent of the time from n = 1e5 (the lowest on fits that stop at iteration 0
after one pass) and 65 to 100 per cent at n = 1e3. BFGS rows can differ in
their counts between backends (finding b below), so their ratios mix speed and
work.

**The regression objective of the tests (p columns).** As first measured, its
device path allocated two Views on the device in every call and computed the
gradient with one team per column, p teams in all (11 to 101 blocks on a card
that runs thousands): the GPU was never faster, 0.59 to 0.64 ms per call at
n = 1e5 and p = 51 against 0.57 to 0.61 ms on 16 threads, where reading X once
(41 MB) at the card's bandwidth takes about 0.1 ms. The path now makes one pass
over X: a team takes 256 rows, computes their residuals, the value and the
weights in team scratch, then sums the weights times each column over its rows,
and writes its 1 + p partial sums to a buffer allocated at construction; a
second small kernel adds the teams' partial sums in a fixed order. The host
path is unchanged. Time per call after the change (timer around each call,
second of two runs):

| bench | n, p | gcc 16 threads | CUDA |
|---|---|---|---|
| proximal | 1e3, 51 | 92 to 102 us | 57 to 71 us |
| proximal | 1e4, 51 | 117 to 129 us | 59 to 66 us |
| proximal | 1e5, 51 | 539 to 585 us | 94 to 98 us |
| primal_dual | 1e3 / 1e4 / 1e5, 41 | 93 / 120 / 460 us | 60 / 64 / 98 us |
| bundle | 100, 11 / 1e3, 51 / 1e4, 101 / 1e5, 11 | 91 / 97 / 194 / 225 us | 44 / 72 / 95 / 63 us |

On 1 thread a call costs about 35 us at n = 1e3, 0.32 ms at 1e4 and 4.6 ms at
1e5 (proximal, p = 51). ms per run, 1 thread / 16 threads / CUDA:

| problem | n = 1e3 | n = 1e4 | n = 1e5 |
|---|---|---|---|
| lasso, FISTA | 1.46 / 3.5 / 2.6 | 10.1 / 4.0 / 1.9 | 139 / 15.8 / 3.0 |
| equicorrelated lasso, FISTA | | | 3975 / 309 / 58 |
| lasso, proximal Newton (CD) | | | 42 / 4.9 / 1.0 |
| equicorrelated lasso, proximal Newton (CD) | | | 61 / 11.1 / 10.7 |
| fused lasso, primal-dual (26606 iterations) | | | 12.3 s on 16 threads / 2.8 s |
| LAD, bundle | 26.5 / 37.9 / 30.6 (p = 51) | 715 / 479 / 393 (p = 101) | 115 / 17.6 / 11.5 (p = 11) |

So with one pass over X the GPU is faster than the best CPU from n = 1e4 at
p = 41 to 51 (about 2 times) and 4 to 5 times at n = 1e5; at n = 1e3 a single
thread is fastest, since the GPU's floor is about 45 to 60 us per call (two
kernels and a copy back). Where the host part dominates the GPU gains little:
proximal Newton's coordinate descent on the equicorrelated lasso (0.7 to 1.2 ms
per iteration with gcc, 1.5 to 2.7 ms with MSVC; the objective takes 5 to 12
per cent of the run on CUDA) and the bundle's dual QP with p = 51 and 101
(about 0.1 and 0.9 ms per iteration on both builds; the objective takes 9 to 50
per cent of the run).

**Host code: MSVC against gcc.** On the host-only benches the CUDA build is
slower by a geometric mean of 2.3 times (second_order, 95 rows; multistart rows
of quasi_newton 3.7 to 3.9 times); iterations, evaluations and results are
identical except on the noisy objective sum(p^2) + 100 sin(1e20 p_1), whose
value differs between the two platforms' sin, as it should. The same factor
shows in the host part of the kernel benches (denoising rows of primal_dual, pure
host code: 1.8 to 2.7 times).

**Where the GPU pays.** For an objective with one reduction per call over
rows of light work, from n of about 1e5 (3 times the best CPU time, 5 times at
1e6); at n = 1e3 a single CPU thread is usually fastest, since the GPU's floor
of about 15 to 26 us per call (launch, reduction and copy back) and the 16
threads' floor of about 38 to 77 us are both larger than the work (one pass of
order 0 at n = 1e3: 7.5 to 127 us on 1 thread, 38 to 62 on 16, 15 to 79 on
CUDA). The crossover lies between n = 1e3 and 1e5; n = 1e4 was not run. For a
regression with p of about 50 read in one pass, from n = 1e4 (2 times) and 4 to
5 times at n = 1e5, unless the host loop dominates.

**The primal-dual status that changed with the thread count, and its repair.**
First measured, the fused lasso at n = 1e3 ended with its budget on 1 thread
(1e5 iterations, V = 2.1e-4) and converged in 6372 iterations on 16 threads and
on CUDA; at n = 1e4 the same; at n = 1e5 it converged only on CUDA (26606
iterations); trend filtering of order 2 at n = 1e3 converged in 39385 iterations
except on 1 thread. Traced per iteration, the 1- and 16-thread runs agree to
1e-14 relative up to iteration 4722 (same V, |r_p|^2, sigma, and L = 3.51e4 after
one doubling at iteration 61). At iteration 4723, on 1 thread, the
descent-lemma test s(x+) - s(x) - grad s(x)'(x+ - x) - (L/2)|x+ - x|^2 <= 8 eps
max(1, |s(x)|) failed on rounding: its first two terms are 8.3e-8 each and
cancel, the computed excess swings by about +-2e-12 between iterations (the
sequential sum of 1000 rows, about 13 eps |s|), above the slack 1.49e-12 plus
(L/2)|x+ - x|^2 = 7.8e-13, while on 16 threads it swings by about +-3e-13. L
only doubles, and each doubling halves tau and makes the quadratic term smaller,
so the next false failure is likelier: 21 more doublings, L = 1.2e12, tau =
1.3e-14, V held at 6.1e-5 until the budget (trend filtering: from iteration
22164, L = 3.7e10, V = 7.2e-4). The repair is the fallback the proximal gradient
method already has (Becker, Candes and Grant, 2011): an excess within sqrt(eps)
max(1, |s(x)|) is decided on gradients, (x+ - x)'(grad s(x+) - grad s(x)) <=
(L/2)|x+ - x|^2, with grad s(x+) already computed, so it costs no evaluation.
After it, 1 and 16 threads give the same status and iterations: fused lasso
6372, 15764 and 26606 iterations at n = 1e3, 1e4 and 1e5, trend filtering 39385
at n = 1e3 (CUDA the same), trend filtering at n = 1e4 still at the budget with
V = 8.4e-2 on both; L doubles once in every run. Other effects, measured on gcc:
denoising with the fused lasso at p = 1000 now converges in 4114 iterations
(was the budget with V = 1.0e-3); of the genlasso fits of the tests, fused
lasso and elastic net on differences at fraction 0.01 on f2 (n = 1000, p = 200)
now converge within the tests' 1e4 iterations (2472 and 3359; were the budget
with V = 1.6e-4 and 1.9e-6), fused at 0.01 on f1 takes 1909 iterations instead
of 1919, and every other row is unchanged. A test runs the fused case at
n = 1e3 with its sum taken on Kokkos::Serial and on the default space and asks
both to converge within 6000 to 6800 iterations (6372 and 6372 on gcc).

**BFGS at iteration 0.** BFGS from a start already at the optimum spends 8 to 39
evaluations confirming convergence, a number that varies with the backend
(gaussian1 at n = 1e3: 23 on 1 thread, 11 on CUDA); its times are those
evaluations.

**Cost of the run.** The primal_dual bench runs n = 1000 by default (the
larger sizes on the command line): 5.8 s on 1 thread, 8.5 s on 16 threads and
11.8 s on CUDA, against 17 minutes on 1 thread with n = 1e4 and 1e5, whose
trend-filtering rows run to the 1e5-iteration budget (4.7 ms per iteration at
n = 1e5 on 1 thread). Every other bench takes seconds.

## Starting values and the fit_distrib bench on the laws added after it (2026-11-19)

**What.** The fit_distrib bench of the optimizers (section "fit_distrib data and
estimates") covered the sixteen continuous and three discrete laws that existed when
it was written. Seventeen cases of the laws added since are now fitted against frozen
references: student_t1, student_t2, skewnormal, skewt, pseudohuber, weibull,
vonmises, beta, betabinom (size 10), negbin1, negbin2, pig, enet, powerexp at
p = 1.5 and p = 3, mcp and scad, each at n = 1e3 and 1e5. Every law already had its
own start_value; the shapes of powerexp (p = 2) and mvpowerexp (beta = 1) were
constants and now come from the kurtosis.

**Starting shapes from the kurtosis.** `powerexp_distrib::start_value(y)` solves
K(p) = b_2, with K(p) = Gamma(5/p) Gamma(1/p) / Gamma(3/p)^2 the kurtosis of the law
(6 at p = 1, 3 at p = 2, 9/5 as p goes to infinity, decreasing) and b_2 = m_4 / m_2^2
the sample's, by bisection on log p over [1/2, 20] (clamped outside
[K(20), K(1/2)] = [1.824, 25.2]); mu and sigma are the sample mean and standard
deviation. `mvpowerexp1/2_distrib::start_value(y, n, theta)` solves Mardia's
multivariate kurtosis of the law, d^2 Gamma(d/(2 beta)) Gamma((d+4)/(2 beta)) /
Gamma((d+2)/(2 beta))^2 (d(d+2) at beta = 1, the univariate K(2 beta) at d = 1),
equated to the sample's (Mardia, 1970), by bisection on log beta over [1/4, 10], and
takes the matrix from S divided by the covariance factor at that beta. Measured
(tests/distrib/test_shape_starts.cpp):

| law | draws | true shape | relative error of the start |
|---|---|---|---|
| powerexp | 1e5 | p = 0.8, 1.5, 2, 3, 6 | 0.0389, 0.0116, 0.0056, 6.4e-5, 0.0094 |
| mvpowerexp1/2, d = 3, log-Cholesky | 4e4 | beta = 0.4, 1, 2.5 | 0.0022, 0.0023, 0.0062 |
| mvpowerexp1, d = 2, ar1 | 4e4 | beta = 0.4, 1, 2.5 | 0.0036, 0.0032, 0.0055 |

With the log-Cholesky structure the covariance of the start equals S to 4.9e-16
relative; ar1 cannot represent every S. Bounds 0.06 and 0.0095. A latent defect was
found on the way: `skewnormal_distrib::start_value` was a template over any summary
type, so `distrib_start_value(prototype, y)`, which prefers a member over the
sample, picked it with the vector and did not compile; it now takes
`distrib_sample_summary`, whose `skewness` it reads.

**References.** `tools/reference/optimizers_fit_distrib_laws.R` simulates the data
in R and writes `fit_distrib_laws.csv` beside the first bench's files (which it does
not touch). The log-density is independent of Laplace's code: distributions7's
`distrib_pdf` for the laws statmodels7 has, gamlss.dist's `dPE` for the power
exponential (its (mu, sigma, nu) is Laplace's (mu, sigma, p)), and for mcp and scad
the definition of the header with the normalizing constant by `integrate()`. The MLE
is optim's BFGS in free coordinates followed by cycles of one-dimensional Brent
searches until a cycle gains less than 1e-11, from the true values and from
statmodels7's estimate, keeping the better; Brent needs no derivative, so it reaches
a maximum at a kink. Laplace's log-likelihood at R's MLE equals R's to at most
1.2e-13 relative on all 34 data sets (bound 1.8e-13): the parametrizations agree.
statmodels7's `fit_distrib()` agrees with R's maximum to 1e-8 on every law it has
except enet, where it reports convergence at alpha = 2.2e-308 (n = 1e3, 10.1 below
the maximum) and at alpha = 1 - 5e-9 (n = 1e5, 2056 below); the test then requires
the fit to reach above statmodels7's value and compares estimates with R's only.

**Fits.** Every run of the thirteen smooth laws (Newton with the observed and the
expected curvature, BFGS, from the library and the +1 start) converges within the
decrement bound of R's MLE and of statmodels7's. enet, mcp and scad have their MLE of
mu on an observation (the nearest one 1.4e-10 to 2.4e-9 from R's estimate), the kink
of l in mu: from the library start every smooth method reaches the maximum (within
4.2e-5 at n = 1e3, enet BFGS, and 1.75e-6 at n = 1e5, scad Newton expected with an
uncapped step, 1.11e-6 with the cap below) but its stopping rule cannot certify a kink,
and most end stalled or with a failed line search. They are asserted on the value
(6.3e-5 and 2.7e-6, the uncapped measurement times 1.5). The certified fit is the
alternation of the Laplace (Brent on mu, Newton on the rest with its step capped at 1):
on all six data sets from the library start it stops converged at the resolution of
Brent's search, within 6.0e-7 of R's log-likelihood (2.2e-7 at n = 1e3). Its mu is
checked through the value only: l has a kink at every observation, and a neighbouring
one can lie within the value bound (scad at n = 1e5 from the +1 start: mu 3.3e-5 from
R's, -l 7.8e-7 above it).

**The +1 start, and why some methods diverge from it.** The maximum is interior in
every case; the failures are the method's.
- pseudohuber: the expected curvature in log nu at the start is 2.33 against a
  gradient of 30.6 (the observed one is -1.74, indefinite). Uncapped Fisher scoring
  takes the full step, accepted because -l falls (2591 to 2244): nu goes to 1.5e-4,
  then 1e28, then 1.3e-159, where l is flat in log nu, and it stalls 19.4 above the
  minimum of -l at n = 1e3. Newton with the observed curvature takes the
  eigen-modified direction, capped at 1, and BFGS a scaled gradient step: both
  converge.
- enet: Fisher scoring takes logit alpha to about 8 in its first step (alpha near 1,
  the Laplace end) and stays there with any cap; the trust region and the capped
  alternation reach the maximum.
- mcp and scad: two basins. The profile log-likelihood at n = 1e3 (maximized over
  the other three parameters, in R) has its maximum at gamma = 3.5 (-2106.99) for mcp
  and at a = 4 (-1985.93) for scad, a valley near gamma = 30 (-2185.33) and a = 10
  (-2038.45), and beyond it a supremum at gamma or a going to infinity, where the tail
  variance goes to 0 and the law becomes the Laplace of rate lambda (-2166.82 and
  -2027.60). From the +1 start (gamma = 8.2) the first Fisher step takes mcp from
  gamma 7 to 71 and the tail variance from 1.9 to 1.3e-4, across the valley; no
  method tried reaches the maximum (Fisher with any cap, trust region, BFGS,
  alternation), and every such fit is flagged on the boundary (infinite free
  standard error). For scad the capped Fisher scoring and the capped alternation
  reach the maximum. A reparametrization of the shape moves the boundary to a finite
  point but cannot remove the second basin, which is in the likelihood.

**Fisher scoring with a capped step.** Over the 36 data sets of the bench at both n,
from both starts (144 runs, the diagnostic "Fisher scoring max_step" of
`tests/optimizers/test_fit_distrib_diagnostics.cpp`), with `max_step` the largest component of a
step in the free coordinates:

| max_step | converged | iterations | value evaluations | gradient evaluations | largest -l above the minimum, converged runs |
|---|---|---|---|---|---|
| infinite | 131 | 1442 | 6089 | 1586 | 1.75e-6 |
| 2 | 133 | 1446 | 6414 | 1590 | 1.75e-6 |
| 1 | 134 | 1153 | 4102 | 1297 | 1.11e-6 |

No run changes to a worse status. With 2, pseudohuber's two +1 runs converge; with 1,
also scad's (n = 1e5 converged, n = 1e3 stalled at the kink within 3.2e-7); enet's +1
runs stay far (882 to 1e5 above) and mcp's unchanged. At equal status, 9 runs take one
more iteration with 1 (scad from the library start at n = 1e3: 8 more, 42 to 50) and
18 take fewer. The bench's Fisher scoring now caps its step at 1
(`fisher_max_step` in `fit_distrib_bench.hpp`); `newton_optimizer`'s own default stays
infinite.

**Reference.** Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis
with applications. Biometrika 57(3), 519-530.

## The multivariate normal rectangle probability, its derivatives and the truncated law: math/mvnormal.hpp, and the cdf members of mvgaussian1 and mvgaussian2 (2026-11-19)

**What.** `log_mvn_rect(k, a, b, Sigma)`, log Pr(a < X <= b) for X ~ N(0, Sigma),
every limit possibly infinite, on the log scale and with a small RELATIVE
error deep in the tail; its derivatives in the shift of the rectangle of every
order (`log_mvn_rect_derivs`); the cumulants of the truncated law T = X | a <
X <= b to order six (`log_mvn_trunc_cumulants`), with `mvn_trunc_mean` and
`mvn_trunc_cov`; the derivatives of log P through those cumulants
(`log_mvn_rect_log_derivs`). On top, the optional members of mvgaussian1 and
mvgaussian2: log_cdf, log_ccdf, log_rect_prob, cdf, ccdf, and the derivatives
log_cdf_dk, log_ccdf_dk, log_rect_prob_dk (location to order six, theta by the
heat equation), log_cdf_dy_dk, log_ccdf_dy_dk, detected by the named concept
`mv_has_cdf`; the class with a singular precision returns quiet NaN.

**References.** mpmath, `tools/reference/mvnormal.py`:
- `bvn_rect`, 2654 rows: lower, upper and mixed orthants on a grid of limits
  (-37 to 8) and correlations (to +-0.9999), and finite rectangles of widths
  1e-6 to 5; log P by the one-dimensional integral of phi(x) times the
  conditional interval probability, a log-concave integrand, cut around its
  mode at multiples of its width, at 40 and 50 digits with different cuts,
  kept when the two agree to 1e-30.
- `mvn_factor`, 700 rows, k = 3..10, one-factor correlations r_ij = l_i l_j
  (loadings to +-0.99), orthants and rectangles: the one-dimensional integral
  over the factor of the product of the conditional intervals.
- `mvn_laml`, 840 rows, k = 2..6: the orthants of the piecewise LAML of the
  lasso (note/penalties.md 5f): one-factor R, t = lam sigma from 0.5 to 20,
  log P to about -1400.
- `tvn_general`: trivariate rectangles with a general correlation by Plackett's
  identity along R(t) = I + t (R - I) and inclusion-exclusion, the precision
  doubled until two runs agree to 1e-30.
- `mvn_trunc_moments`, 359 rows (one-factor, LAML and rectangles, k = 2..6):
  the raw moments of T to order 4 (k <= 3) or 2 by Gauss-Legendre over the
  factor at 60 digits, the conditional truncated moments closed, the cumulants
  from them at that precision; two rules (degree 6 and 7) must agree to 1e-24.
- `tools/reference/mvnormal_lattice.py`: Korobov generators for N = 1021,
  4093, 16381, 65521 and lattice dimensions 1..18, minimizing the weighted P_2
  criterion with gamma_j = 1/j^2 (unweighted, the criterion grows like 3.3^s
  and from s = 13 picks degenerate generators such as 2).

**The bivariate.** Candidates measured on `bvn_rect` (max |error of log P|):

| candidate | log P > -2.3 | > -23 | > -230 | > -2300 |
|---|---|---|---|---|
| Genz's bivariate (Drezner-Wesolowsky) | 3.1e-16 | 3.6e-7 | 1.8e2 | 1.7e3 |
| separation of variables, tanh-sinh 129 nodes | 3.6e-3 | 8.0e-4 | 1.9e-10 | 1.6e-10 |
| SOV, Gaussian-matched tilt | 3.2e-1 | 1.3e-3 | 9.2e-2 | 2.9e-6 |
| principal axes + exponential tilt, 97 nodes per piece | 1.3e-15 | 1.8e-13 | 1.6e-10 | 1.9e-10 |

The plain separation of variables fails for |r| near 1: the inner interval
probability changes over a width sqrt(1 - r^2)/|r| of the outer variable. A
Gaussian tilt matched to the curvature fails because the proposal's tails are
lighter than the integrand's (the ratio grows like exp((1/w^2 - 1) x^2/2)).
What works: X1 = al u + be v, X2 = al u - be v (al = sqrt((1 + r)/2),
be = sqrt((1 - r)/2)), the outer variable v when r >= 0 and u when r < 0, so
that the inner limits move with slope at most 1; the outer axis cut where the
active lines of the inner interval change; an exponential tilt (unit
variance, slope matched at the maximum) on each piece. The last two rows of
the table were then limited by tiny rectangles (widths 1e-6), whose rounded
limits lose eps |a| / width: the widths are now passed exactly
(log_interval(a, b, width)), and a coordinate narrow relative to the slopes is
summed by 16-point Gauss-Legendre. Library route (Genz while its result is
>= 1e-3, else the above with 97 nodes per piece), gcc:

| log P | > -2.3 | > -23 | > -230 | > -2300 | below (relative) |
|---|---|---|---|---|---|
| max error | 3.1e-16 | 9.9e-15 | 1.4e-13 | 8.5e-13 | 1.1e-14 |

65 nodes give 2.4e-13, 1.6e-12 and 6.7e-12 in the middle bins. Cost, gcc one
thread: 0.8 us in the bulk (Genz), 40 to 65 us in the tail.

**Three and more coordinates.** The orthant session's bar (note/penalties.md
5f: tilted log-SOV within 3.4e-13 at k = 2-4, 1e-4 at k = 4-6 with a Korobov
lattice) is met as follows. The separation of variables with the
Gibson-Glasbey-Elston order, every factor through log_Phi and Phi_inv_logp,
tilted by Botev's (2017) minimax saddle point (Newton on the gradient with
its exact Jacobian, 2 (m - 1) unknowns, step halved until the gradient's norm
falls); m <= 4 constrained coordinates by a tensor tanh-sinh rule over the
first m - 2 (193 nodes over one coordinate, 97 per coordinate over two) and the
bivariate for the last two; m >= 5
by the fixed lattice (N = 16381, Giovanni) over the first m - 1, the last by
log_interval (the bivariate for the last two gave the same error at two to
five times the cost). A first trivariate in principal axes of the most
correlated pair, with the bivariate inside, reached only 1e-9 to 3e-5 in the
tail and was dropped. With 49 nodes per coordinate a near-singular trivariate
of `tvn_general` (r_12 = -0.95, det R = 0.0075) lost 3.3e-6 in log P in the
bulk (the reference confirmed by an independent nested integral and by Monte
Carlo); 193 nodes over one coordinate give 9.4e-16 there, and 97 per coordinate
over two were kept for m = 4 (Giovanni; a tail value then costs about 230 ms,
against about 60 ms with 49). Measured, gcc (max |error of log P|):

| set | k | > -2.3 | > -23 | > -230 | > -2300 |
|---|---|---|---|---|---|
| mvn_laml | 2 | 1.8e-16 | 4.8e-15 | 4.4e-14 | 2.0e-13 |
| mvn_laml | 3 | 2.0e-16 | 5.2e-15 | 3.4e-14 | 1.3e-13 |
| mvn_laml | 4 | 5.2e-15 | 1.1e-14 | 3.3e-14 | 2.4e-13 |
| mvn_laml | 5 | 1.6e-6 | 1.6e-5 | 1.1e-5 | 9.3e-6 |
| mvn_laml | 6 | - | 7.3e-5 | 4.9e-5 | 1.5e-5 |
| tvn_general | 3 | 9.4e-16 | 5.5e-14 | 1.1e-12 | 2.2e-11 |
| mvn_factor | 3 | 7.7e-16 | 1.7e-14 | 8.2e-14 | 1.9e-12 |
| mvn_factor | 4 | 7.3e-15 | 4.8e-14 | 6.1e-13 | 2.1e-12 |
| mvn_factor | 5..8 | 1.2e-5 | 4.7e-5 | 2.9e-5 | 8.9e-6 |

The lattice's error does not grow in the tail (the tilt). Lattice size
against error, one-factor rows k = 5..10 (largest over k): N = 1021 1.4e-3,
4093 1.4e-4, 16381 4.7e-5, 65521 7.3e-6; the cost is linear in N
(k = 6: 1.6, 6.5, 26 and 102 ms per value, gcc one thread).

**Derivatives in the shift.** For a set S of coordinates and a choice e of a
finite end for each, G_{S,e} = phi_{Sigma_SS}(c_S) Pr(rest | X_S = c_S); then
d_j G_S = sum_e sgn G_{S+j} for j outside S and d_i G_S = -v_i G_S - sum_j
B_ji sum_e sgn G_{S+j} inside (v = Sigma_SS^-1 c_S, B = Sigma_TS Sigma_SS^-1),
so every derivative is a combination of lower-dimensional rectangle
probabilities. Central differences of the order below agree to 1e-10 at every
order to six, k = 1..4, orthants, rectangles and mixed (test bound 1e-8). In
the far tail the log-derivatives built from these moments cancel: at k = 1
against log_Phi_dk, order 3 lost 1.2e-9 at z = -10 and order 6 0.21 at z = -20.

**The truncated law, and the log-derivatives through it.** One dimension:
W = Z | lo < Z <= hi is c -+ Y with c the end nearer the bulk and Y of density
proportional to exp(-a y - y^2/2); E[Y^j] = (-1)^j B_j(l_1, ..., l_j), l_i
the derivatives of the log Mills ratio (normal.hpp) and B_j the complete Bell
polynomials, whose terms share a sign; a finite window subtracts the part
beyond it, a narrow one is summed by Gauss-Legendre about the mean. Against
mpmath: within 2e-15 to z = -40 (relative, every central moment to six), the
odd moments of windows of width 1e-5 within 5e-11 of their own (nearly zero)
size. Several dimensions: two passes over the tilted nodes (the last
coordinate exact), weights and mean, then the central moments as sums of
products of d = E[X | node] - E[T] and the last coordinate's exact central
moments; cumulants over the set partitions without singleton blocks. Against
mvn_trunc_moments, errors in units of the product of standard deviations
(gcc; the same in the rows with log P < -23):

| k | log P | mean | covariance | kappa_3 | kappa_4 | ms per call |
|---|---|---|---|---|---|---|
| 2 | 5.7e-14 | 2.1e-13 | 2.8e-14 | 6.0e-13 | 4.7e-12 | 0.15 |
| 3 | 1.8e-12 | 1.1e-10 | 3.6e-11 | 1.0e-9 | 2.3e-8 | 16 |
| 4..6, N = 16381 | 1.3e-5..5.7e-5 | 1.6e-4..2.1e-4 | 3.1e-3 | | | 52..65 |
| 4..6, N = 65521 | 2.7e-6..8.2e-6 | 1.9e-5..6.1e-5 | 3.0e-4..6.7e-4 | | | 207..271 |

Tensor rule for the moments to m = 3 with 65 nodes per coordinate (49 gave
cov 4.7e-14 / 1.9e-9 at k = 2 / 3, 97 gave 1.4e-12 at k = 3 for 34 ms); the
moments' lattice defaults to N = 65521 (Giovanni), log P and its derivatives
stay at 16381. With log F(delta) = log P + K_T(-Q delta) - delta' Q delta / 2
(phi_Sigma(u + delta) = phi_Sigma(u) exp(-delta'Q u - delta'Q delta/2),
Q = Sigma_cc^-1), d^n log F = (-Q)^n kappa_n for n >= 3, -Q + Q kappa_2 Q and
-Q kappa_1 below: at k = 1 the six orders of log Phi are within 3.3e-13 of
log_Phi_dk from z = 3 to z = -40. The distrib members take this route for
the blocks without theta (every order in the location and in y).

**Theta in the far tail.** The theta blocks go through the heat equation on
the moments (y-order A + 2K) and the set-partition logarithm, which cancels
at high orders in the tail: against differences in theta of the order below,
K = 1 and 2 (A <= 1) hold to <= 2e-9 at y = -15; K = 3 with A = 2 loses 1.2e-5
at y = -6, 4.4e-4 at -10 and 6e-2 at -15 (bivariate, unstructured
log-Cholesky). Open: the heat equation written for log F, so that the theta
blocks are assembled from the cumulants too.

**Costs (gcc -O2, one thread).** Derivatives with the recursion are dominated
by the conditional probabilities of dimension >= 3 (measured after the tensor
rule's final node counts): k = 3 0.24 ms (order 1) to 0.35 ms (y-order 12);
k = 4 17 to 18 ms; k = 5 48 to 52 ms; k = 6 34 ms (order 1) to 172 ms (lattice
of the conditional probabilities N = 4093). Scratch: 3 KB (k = 2) to 2.9 MB (k = 6, y-order 12).

**CUDA (RTX 4070 SUPER, default build).** The same tests on the device, one
thread per reference row. The lattice routes (k >= 5) and the truncated law at
k >= 4 give the gcc figures to the digit. Elsewhere (max |error of log P|):
bivariate 2.6e-16, 9.5e-15, 1.4e-13, 9.1e-13 (relative below -2300 1.1e-14);
mvn_laml k = 2..4 at most 4.3e-15, 1.1e-14, 4.3e-14, 2.4e-13; mvn_factor k = 3
7.1e-16, 1.7e-14, 9.9e-14, 2.1e-12 and k = 4 6.4e-15, 3.0e-14, 2.1e-13,
1.3e-12; tvn_general 9.4e-16, 3.8e-14, 8.3e-13, 1.0e-11. Truncated law, k = 2:
log P 1.1e-13, mean 4.1e-13, covariance 2.9e-14, kappa_3 1.5e-12, kappa_4
6.4e-12; k = 3: 2.3e-12, 1.1e-10, 3.6e-11, 1.0e-9, 2.3e-8. The distrib members
inside a kernel differ from their host values by at most 1.8e-14 relative. The
test bounds are the larger of the two builds times about 1.5.

**Theta on the log scale (Giovanni, 2026-11-19, after the commit 6065021).**
The theta blocks are assembled on the log scale where the recursion's
logarithm cancels. psi(delta; E) = log F(delta; Sigma + E) satisfies
d_E psi [M] = (1/2) M : (grad^2 psi + grad psi grad psi'), so along the
directions M_c = d_c Sigma of the blocks of a set partition of the theta
indices, G_(S + t) = (1/2) M_t : grad^2 G_S + (1/2) sum over the ordered
splits S1 + S2 = S of grad G_S1' M_t grad G_S2 (G of the empty set is psi),
the derivatives in delta by Leibniz over the positions; only products of
derivatives of psi enter, never the moments of F. The derivatives of psi to
order A + 2K (twelve at most) come from the cumulants: trunc1_moments
extended to order 12 (for a < 4 by 64-point Gauss-Legendre on [0, 12], for
a >= 4 by Miller's backward recurrence of E_(j+1) = -a E_j + j E_(j-1),
started at j = N + 60; central moments within 2.3e-13 of mpmath at every
order to 12, z from 3 to -40) and the cumulants from the central moments by
the recursion over the subsets containing the first position (2^(n-1)
subsets per tuple; the set partitions are 4.2 million at order 12).
Measured: the K = 3, A = 2 block of the bivariate (unstructured
log-Cholesky) against differences of the order below, 6.9e-10 at y = -6 and
1.7e-9 at y = -15 (the recursion route: 1.2e-5 and 6e-2); the cumulant route
against differences of the order below to order 12, 1.1e-7 (bivariate and
trivariate, bulk and tail, step 1e-4). Against the recursion in the bulk the
two agree to 1e-12..1e-11 at k = 2-3 and order 6; at k >= 4 the cumulants
come from the lattice and their high orders are noisy (k = 4 at order 4:
5e-2, order 2: 4.5e-4), while the recursion's pieces are exact there. The
log route also costs more in the bulk (k = 3: 12 to 29 ms against 0.3 ms). So
log_dk takes the log route for at most three coordinates when log F < -6
(heat_log_threshold), and the recursion otherwise (also for the location
blocks, K = 0, which the commit had sent through the cumulants for every k).
Member costs, gcc one thread, K = 1..3 with A <= 2: bivariate 2.8 to 6.3 us in
the bulk and 0.15 to 0.29 ms in the tail (y = -8); trivariate 0.41 to 0.43 ms
and 23 to 38 ms.

**The lattice at four coordinates on request (2026-11-19).** `log_mvn_rect` and
`log_mvn_trunc_cumulants` take `tensor_max_dim` (default `mvn_tensor_max_dim`
= 4 and `mvn_trunc_tensor_max_dim` = 3): the largest number of constrained
coordinates done by the tensor rule. With 3 the lattice (N = 16381) takes four
coordinates: on the 180 LAML rows at k = 4 log P within 4.3e-6, 1.3e-5, 1.3e-5,
9.7e-6 by bin (gcc), against 2.4e-13 with the tensor rule. The tensor rule's
cost depends on the route of its inner bivariate: an equicorrelated orthant
(r = 0.81) at limits -9 to -11 costs 12.4 ms with it and 13.9 ms with the
lattice, while the orthant work measured 0.32 s for deep one-factor orthants
at r = 0.9, where every inner bivariate takes the tail route (about 40 us,
9409 nodes).

## The multivariate normal rectangle: derivatives in the two limits, and the far tail at four and more coordinates (2026-11-20)

**What.** Two additions to the section above. (1) `log_mvn_rect_limit_derivs`,
d^zeta F / F for the tuples zeta over the 2k limits z = (lo_1..lo_k,
hi_1..hi_k), and the members `log_rect_prob_dab_dk<Ja, Jb, K, A...>` of
mvgaussian1 and mvgaussian2: the derivatives of log Pr(a < Y <= b) of order Ja
in a, Jb in b, K in theta and A in mu (Ja + Jb + K + A <= 6), the output the
Kronecker product of the packed tensors of a, b and mu, a outermost
(`cdf_dab_scratch_size`). (2) The theta and location blocks of log_cdf_dk and
the rest in the far tail for four and more coordinates.

**The limits.** The table of the recursion (G_{S,e}, the corner densities)
already holds every end choice e separately; the shift derivative is their
signed sum. A derivative in hi_j alone pins coordinate j at its upper end, one
in lo_j at its lower end with a minus sign, and d lo_j d hi_j F = 0 exactly; so
d^zeta F / F is one term of the sum, at no extra cost. The members contract the
tensors over z with M~ = M (x) [[1, 1], [1, 1]] for the heat equation (the
shift is d_lo + d_hi) and take d_mu = -(d_a + d_b); the logarithm is the
set-partition logarithm over the items, as in the recursion route. For a
coordinate with one infinite end the derivative in its finite end is the shift
derivative of the coordinate, so where log F < -6 the entries whose a and b
items lie on such coordinates come from the log route (the cumulants) and the
others from the recursion. Checks: summed over the two sides, the limit tuples
equal the shift tuples to rounding (k = 1..4, order 5); every block (d = 2, 3,
both forms, orders to (Ja, Jb, K, A) = (3, 0, 1, 1), (1, 3, 1, 0), (2, 0, 3, 1),
(2, 2, 1, 1)) agrees with central differences of the block one order lower to
2.1e-10; Ja = Jb = 0 equals log_rect_prob_dk to 3.6e-15.

**References.** `mvn_limit_derivs` in `tools/reference/mvnormal.py`, 56 rows,
k = 2..4, one-factor correlation (loadings -0.9 to 0.9), centres -12 to 1.5,
finite rectangles (widths 10^-1.5 to 10^0.5), one-sided coordinates and mixes:
given the factor the coordinates are independent and a derivative in b_i acts
on D_i(t) = Phi(u_i) - Phi(l_i) alone, d^m D_i / d b_i^m = s_i^-m (-1)^(m-1)
He_(m-1)(u_i) phi(u_i); every d^zeta P / P by Gauss-Legendre over the factor at
50 digits, the derivatives of log P over the set partitions, two rules
(degree 6 and 7) agreeing to 1e-24. Every order 1..3 over the 2k limits.
Measured, gcc (max relative error, |ref| below 1 read absolute):

| rows | order 1 | order 2 | order 3 |
|---|---|---|---|
| log P > -23 | 1.3e-14 | 3.4e-12 | 1.5e-10 |
| log P < -23, one infinite end (log route) | 5.8e-14 | 2.1e-15 | 1.9e-13 |
| the same entries by the recursion | 9.3e-14 | 3.0e-11 | 2.3e-9 |
| log P < -23, both ends finite (recursion) | 3.7e-13 | 3.2e-10 | 1.6e-8 |

CUDA (RTX 4070 SUPER, default build): 1.3e-14, 3.5e-12, 1.5e-10; 5.4e-14,
2.3e-15, 1.8e-13; 4.5e-13, 3.2e-10, 4.5e-8 (the last the only figure that
moves). The k = 4 tail blocks, the k = 4 cumulants and the gaussian_laplace
moments below give the gcc figures to two digits on CUDA. Test bounds: the
larger of the two builds times 1.5.

Open: an entry in the limits of a coordinate with both ends finite has no
shift relation, so in the far tail it keeps the recursion's cancellation.

**The far tail at four and more coordinates.** Before (recursion, mvgaussian1
log_cdf_dk, unstructured log-Cholesky, y_i = t + 0.3 i, each block against
central differences of the block one order lower, relative):

| k = 4, t | K3 A0 | K3 A2 | location A6 | K2 A4 |
|---|---|---|---|---|
| -3 | 2.7e-10 | 3.4e-7 | 4.2e-6 | 3.8e-6 |
| -6 | 1.0e-9 | 3.0e-5 | 2.4e-4 | 1.2e-3 |
| -10 | 2.8e-8 | 7.6e-4 | 3.7e-3 | 2.8e-2 |
| -15 | 8.1e-8 | 2.6e-2 | 7.3e-2 | 1.4 |

At k = 5 the recursion (its conditional probabilities on the lattice) gave
K3 A2 2.1e-2, 7.6e-1, 8.9e-1 and A6 2.7e-1, 1.0, 1.0 at t = -3, -6, -10. The
log route needs cumulants that do not cancel; at k = 4 they came from the
lattice (5e-2 at order 4). The repair: the truncated law's nodes are a tensor
tanh-sinh rule over three coordinates when four are constrained
(`mvn_trunc_tensor_max_dim` = 4, `mvn_trunc_tensor_half_nodes_3` = 16, 33
nodes per coordinate; Giovanni, 2026-11-20). Against the 98 k = 4 rows of
`mvn_trunc_moments` (units of the standard deviations), by nodes per
coordinate:

| nodes | log P | mean | covariance | ms per call |
|---|---|---|---|---|
| lattice 65521 (before) | 2.7e-6..8.2e-6 | to 6.1e-5 | 3.0e-4..6.7e-4 | 207..271 |
| 25 | 1.2e-6 | 2.2e-6 | 1.4e-5 | 45 |
| 33 | 5.6e-10 | 1.5e-8 | 3.0e-7 | 105 |
| 49 | 1.5e-12 | 1.3e-9 | 5.5e-10 | 337 |
| 65 | 1.4e-12 | 4.3e-9 | 2.6e-11 | 812 |

(timed four at once on one thread each). With 33 nodes every k = 4 block above
is at 1.4e-11..2.5e-8 for t = -3..-15 (49 and 65 nodes give the same, the floor
of the difference check). The route is chosen by log F alone: the log route
below heat_log_threshold (-6) for every k, the recursion above. At k = 5 the
log route (lattice cumulants, N = 16381) gives K3 A2 3.9e-4, 2.7e-4, 1.3e-4
and A6 1.2e-2, 7.4e-4, 4.9e-5 at t = -3, -6, -10, against the recursion's
values above; in the bulk the recursion wins (t = -1: K3 A2 6.5e-5 and A6
5.3e-3 against 1.4e-1 and 1.4e-1), so the threshold stays. Open: at five and
more coordinates the tail blocks are limited by the lattice (1e-4..1e-2 at
high orders).

**Costs (gcc -O2, one thread, k = 4).** Per member call, K = 1..3 with A to 6:
in the bulk (t = -1) the recursion, 20 to 21 ms; in the tail (t = -10) 0.62 to
0.70 s on the log route, against 1.1 to 1.2 s on the recursion before, most of
it log P itself (the tensor rule's inner bivariates take their tail route).
With the tensor rule as the default at four coordinates, mvn_trunc_mean,
mvn_trunc_cov and the moments of gaussian_laplace at k = 4 change too: the
gaussian_laplace test (moments, k = 4) went from log I (moments) 5.0e-6, mean
1.4e-4, covariance 2.7e-3 to 4.1e-11, 3.9e-9 and 3.2e-8 on gcc, and the case
from 26 to 74 s (its moments were on the lattice of 16381 points, about 60 ms
per orthant, against 105 ms now; the default lattice of 65521 cost 207 to
271 ms).

## The Gaussian-lasso integral and its law: math/gaussian_laplace.hpp (2026-11-19)

**What.** `gaussian_laplace_log_integral(k, m, Sigma, lam)`,
log I = log int N(beta; m, Sigma) exp(-sum_i lam_i |beta_i|) dbeta, and the mean
(`gaussian_laplace_mean`) and covariance (`gaussian_laplace_cov`, packed lower)
of the law proportional to the integrand; lam_i >= 0 finite, Sigma packed lower.

**How.** On the orthant of signs s, with ell = lam o s, the integrand is
exp(c_s) N(beta; mu_s, Sigma), mu_s = m - Sigma ell, c_s = -ell' (m + mu_s)/2, so
log I = log sum_s exp(c_s + log P_s) with P_s the orthant probability of
N(mu_s, Sigma), from `log_mvn_rect` (math/mvnormal.hpp). Deep in the tail c_s
and log P_s are both large (about t^2/2 at t = lam sd) and of opposite sign;
only their sum is formed, in a log-sum-exp over the orthants. The moments are
the mixture of the per-orthant truncated means and covariances
(`log_mvn_trunc_cumulants`, orders 1 and 2), with weights
exp(c_s + log P_s - log I), accumulated orthant by orthant as convex
combinations centred on the running mean:
Cov <- g Cov + f C_s + f g d d', mean <- mean + f d, with f the new orthant's
weight, g = 1 - f the weight of those before (both exponentials of
differences of logs) and d its mean minus the running mean. The value asks
log_mvn_rect for the lattice from four coordinates on (`tensor_max_dim = 3`),
as the moments already do: at k = 4 the tensor tanh-sinh rule costs 0.32 s
per orthant and the lattice about 20 ms.

**References.** mpmath, `tools/reference/gaussian_laplace.py`, 195 rows,
k = 1..5, one-factor Sigma (loadings to 0.99, standard deviations 0.5 to 2),
t = lam sd from 0.5 to 20 (rates jittered by 0.7-1.3), the mode at the kink,
random within 1.5 sd, or off the kink by lam sd^2 + 4 sd on half the
coordinates. Given the factor z the coordinates are independent and each
one-dimensional integral is closed, J = exp(lam^2 tau^2/2)(exp(-lam mu)
Phi(mu/tau - lam tau) + exp(lam mu) Phi(-mu/tau - lam tau)), with the
conditional moments of its two truncated-normal pieces closed too; I and the
moments are then 1-D integrals over z by tanh-sinh, cut around the mode of
the log-concave integrand, at 40 digits and at 50 digits on cuts 1.3 times
wider; all 195 rows agree (1e-28 for log I, 1e-24 for the moments). The
formulas were checked against a 2-D quadrature in double (k = 2) and the
closed form (k = 1), to one ulp.

**Accuracy** (`tests/math/test_gaussian_laplace.cpp`; log I absolute, the
mean in units of the law's standard deviations, the covariance in units of
their products; the moments on the lattice N = 16381 in the test, the default
is 65521; max over the rows, gcc and CUDA):

| k | log I | log I (moment functions) | mean | covariance |
|---|---|---|---|---|
| 1 | 3.6e-14 | 3.6e-14 | 1.4e-14 | 5.0e-14 |
| 2 | 6.5e-14 | 5.5e-14 | 2.1e-13 | 1.4e-13 |
| 3 | 1.1e-13 | 1.1e-13 | 1.4e-12 | 4.5e-12 |
| 4 | 3.3e-6 | 3.3e-6 | 9.5e-5 | 1.8e-3 |
| 5 | 5.8e-6 | 5.8e-6 | 7.9e-5 | 1.2e-3 |

gcc and CUDA agree to the digit from k = 3 on; at k = 1-2 CUDA's log I differs
in the last digits (3.6e-14 against 9.3e-15 at k = 1). Bounds: these with a
margin of about half again.

**Speed** (gcc -O3, one thread, one-factor r = 0.9, t = 10, mode at the kink;
`benchmarks/math_gaussian_laplace.cpp`; seconds per evaluation):

| k | orthants | log I | mean | covariance |
|---|---|---|---|---|
| 1 | 2 | 2e-7 | 7e-7 | 1.6e-6 |
| 2 | 4 | 1.9e-4 | 1.4e-4 | 3.5e-4 |
| 3 | 8 | 0.061 | 0.030 | 0.070 |
| 4 | 16 | 0.33 | 1.1 | 2.5 |
| 5 | 32 | 0.61 | 2.8 | 6.3 |
| 6 | 64 | 1.5 | 7.0 | 16 |
| 7 | 128 | 4.0 | 18 | 38 |
| 8 | 256 | 9.1 | 38 | 82 |

(k = 4 with the tensor rule: 5.1 s for log I.) The orthants are independent,
so the sum divides over threads.

**Open.** Pruning the orthants whose saddle-point bound is 40 below the
largest removes some only at correlation 0.99 (202 of 256 at k = 8): not
implemented.

## The power exponential penalties: the prox of a|x|^p, powerexp and the isotropic mvpowerexp (2026-11-19)

What. The penalty side of powerexp_distrib and mvpowerexp1/2 (note/penalties.md
section 5g). One scalar problem,

  x*(u, a, p) = argmin_x a|x|^p + (x - u)^2/2,   a > 0, p > 0,

in include/laplace/distrib/detail/power_prox.hpp, behind three members:
powerexp_distrib::prox_y(v, t) = mu + s x*((v - mu)/s, t/s^2, p) with s the
natural scale sigma c(p) (written in units of s so that s^-p never overflows);
mvpowerexp1/2::prox_y, present only over the new one-parameter structure
scaled_identity_pdmat (M = e^(2 theta) I), where the minimizer lies along
u = v - mu with norm c x*(||u||/c, t q/2, 2 beta), q the diagonal of Sigma^-1
or Q and c = q^(-1/2); at beta = 1/2 the block soft threshold. General Sigma
has no prox (a secular equation with a factorization per step). Descriptors:
proper, beta_regularity 0, kinks per coordinate (powerexp) or radial
(mvpowerexp), prox closed_form, rho = 0 for p >= 1 and beta >= 1/2, infinity
below.

How. p = 1 is the soft threshold and p = 2 is u/(1 + 2a), exactly. Otherwise a
nonzero |x*| solves phi(x) = x + a p x^(p-1) - |u| = 0, by Newton from a bound
on the side where the iteration is monotone, stopped when a step no longer
moves x in its direction (a test that FMA cannot fool, unlike a tolerance):
1 < p < 2, phi concave, from the lower bound (|u|/(|u|^(2-p) + a p))^(1/(p-1));
p > 2, phi convex, from min(|u|, (|u|/(a p))^(1/(p-1))); p < 1 (the bridge,
nonconvex) zero when |u| <= tau = x_b + a p x_b^(p-1), x_b =
(2a(1-p))^(1/(2-p)), zero on the tie, otherwise the larger root from |u| on the
convex branch. The global minimizer is returned for every t: rho is infinite
for p < 1, so no step satisfies t rho < 1, and the optimizers that require it
(MFISTA, proximal Newton) refuse the penalty; a plain iterative thresholding
may use it, without a convergence guarantee.

Measured before writing (Python against a 200-bit bisection, 400 random (u, a)
per p in 1e-8..1e8): at most 12 Newton steps, mean 1.2 to 4.1; C++ -O3 on one
core with std::pow, 4 ns for the soft threshold, 250 ns for p < 1, 360-510 ns
for 1 < p < 2, 67-126 ns for p > 2.

References. tools/reference/power_prox.py, mpmath: bisection in log x (400
halvings), the bracket from the monotone branch, for p < 1 the comparison with
the objective at 0; at 200 and 260 bits, kept when they agree to 2^-120. 2798
rows, 21 values of p from 0.1 to 50 (0.99, 1.01, 1.05, 1.99, 2.01 among them),
(u, a) log-uniform in 1e-8..1e8, and for p < 1 the threshold approached at
relative 1e-6 and 1e-10 on both sides.

Error measure. The error in ulps of x* is divided by the condition number of
the problem, kappa = |u|/(|x*| phi'(x*)), phi'(x) = 1 + a p (p - 1) |x|^(p-2):
rounding the data moves x* by kappa ulps. kappa tends to 1/(p - 1) near p = 1
when x* is tiny and is large for p < 1 just above the threshold, where phi'
vanishes at the edge of the branch.

**Tests** (tests/penalty/test_power_prox.cpp, kernels on gcc and CUDA):

| check | gcc | CUDA | bound |
|---|---|---|---|
| x* against mpmath, ulp / max(1, kappa) | 7.17 | 7.27 | 11 |
| x* against mpmath, plain ulp (printed) | 164 | 192 | - |
| zeros of the bridge where the reference is zero | exact | exact | exact |
| powerexp prox_y: objective above a 4001-point grid + golden (relative) | 2.51e-15 | 2.51e-15 | 4e-15 |
| powerexp prox_y: optimality residual (scaled) | 5.68e-16 | 5.68e-16 | 9e-16 |
| mvpowerexp1 at beta = 1/2 against the block soft threshold | 2.22e-16 | 2.22e-16 | 4e-16 |
| mvpowerexp1: optimality residual of the gradient (scaled) | 2.22e-16 | 2.37e-16 | 4e-16 |
| mvpowerexp1: objective above the least along the ray (relative) | 3.99e-16 | 4.25e-16 | 7e-16 |
| mvpowerexp2 at Q = I/sigma^2 against mvpowerexp1 | 0 | 0 | exact |

The worst ulp/kappa point is p = 0.9 just above the threshold
(u = 1.25e-3, a = 4.92e-4). p = 1 and p = 2 equal the closed forms bit for bit,
the prox is odd in u bit for bit, NaN gives NaN and u = 0 gives 0; mvpowerexp at
u = 0 and in the zero region of beta = 1/2 returns mu exactly; the group-lasso
product (mvpowerexp2 over scaled_identity_pdmat) and the powerexp product with
mu_0 forward prox_y copy by copy exactly. scaled_identity_pdmat equals
diagonal_pdmat with equal parameters bit for bit (matrix, inverse, factor), and
its derivatives agree with central differences to 1e-9.

Open. A prox for a general Sigma (beta >= 1/2: a secular equation in gamma,
x = (I + gamma Sigma^-1)^-1 u; beta < 1/2: several stationary points), if a
consumer asks.

## Residues of the optimizers: BFGS's reset on a stall, the rounding floor of the primal-dual measure, the two basins of mcp and scad (2026-11-19)

**BFGS at kappa 1e10, why it stalled.** On the quadratic f = (1/2)(x - x*)'A(x - x*)
with n = 10, the eigenvalues of A log-spaced from 1 to kappa = 1e10 and x0 = 0,
BFGS with the Wolfe search ended stalled after 130 iterations at f - f* = 1.13e-2.
Traced per iteration: no curvature pair was skipped (cos(s, y) >= 3e-3 against
the threshold 1e-10) and the pairs were exact (|y - A s| / |A s| from 1e-11 to
7e-7). The error stays on the eigenvector of eigenvalue 1, |e_weak| = 0.15 from
iteration 90 on, which is the whole of f - f* = (1/2) 0.15^2. No pair sampled
that direction: the largest s'y/y'y seen is 1.05e-4 against its true 1, so B is
1e-4 of the truth there, and so is the floor gamma_max |g|^2 of the decrement.
In the last search the step had |d| = 6.5e-8 and the slope -3.7e-5 (the computed
slope exact to three digits), a predicted decrease of 2.4e-12 against a rounding
of f of +-2e-10: Armijo's test was decided by the rounding and the zoom ended
below rounding. The cause is the loss of curvature information along the weak
direction, then the resolution of f. Armijo's backtracking converged at n = 10
along a different path, not at n = 20.

**The repair.** When the search ends below rounding or below resolution while the
model holds pairs, B is reset once to the scaled steepest descent (gamma_max is
kept) and the search is repeated; a second such search without an accepted step
in between ends stalled, as before. Measured on the whole quasi-Newton battery
(the seven More-Garbow-Hillstrom problems at f* = 0, 1, 1e3, 1e6 under both
searches, the quadratics n = 10, 20, 50, 100 with kappa from 1e2 to 1e12, and
the negative cases):

| quadratic | before | with the reset |
|---|---|---|
| n = 10, kappa 1e10, Wolfe | stalled, 130 it, f - f* 1.1e-2 | converged, 169 it, 9.2e-7 |
| n = 20, kappa 1e10, Wolfe | stalled, 228 it, 1.2e2 | converged, 485 it, 2.3e-12 |
| n = 50, kappa 1e10, Wolfe | stalled, 543 it, 1.9e3 | converged, 722 it, 2.3e-9 |
| n = 50, kappa 1e11 / 1e12, Wolfe | stalled | converged, 1.2e-11 / 1.2e-18 |
| n = 10, kappa 1e11, Armijo | stalled, f - f* 1.9e2 | converged, f - f* 2.1e-5, decrement 6.3e-7 |

Every row where no stall occurred is unchanged (all of kappa <= 1e9 and the
More-Garbow-Hillstrom problems); the other rows at kappa 1e11 and 1e12 still end
stalled, line search failed or at the budget; no negative case converges (abs_sum
with n = 4 under Wolfe goes from stalled to line search failed). Keeping gamma_max
matters: clearing it with B gave f - f* = 3.7e-6 at n = 10, kappa 1e10, a false
success at the tolerance 1e-6. The last row of the table is a false success that
the reset creates: a limit, documented and not tested.

L-BFGS with the same reset, measured and not applied: at memories 5, 10 and 20 it
changes no status and no f - f* except at kappa 1e11 and 1e12 with n = 10 and 20
(stalled becomes line search failed, or the same status with more evaluations)
and Brown's badly scaled function under Armijo (stalled either way). Its false
successes at kappa 1e8 to 1e10 (converged 1e-2 above f*) come with no stall, so
a reset cannot reach them.

**The rounding floor of the primal-dual measure.** On the genlasso dataset f2
(n = 1000, p = 200), trend filtering at 0.5 of lambda_max (lambda = 1.08e6) ran to
1e6 iterations without converging at the tolerance 1e-8, while its coefficients
were within 7.5e-13 of the reference: V reached 3.1e-8 at iteration 2e5 and
then wandered between 2.8e-8 and 3.5e-8, with f changing by 6e-12 (about eps |f|,
f = 18877.4). Each of the 18 genlasso fits was run past its convergence with the
tolerance set to 0, and the measure recorded over the second half of the extra
iterations (as many as the run took to converge, at least 2e4). The floor is
entirely in e_d (the primal part was below 3e-18). Its scale is the rounding of
D x: lambda |D x|_1 is not it (V/(eps lambda |D x|_1) ran from 0 to 2.1e4), while
against S = |y|_inf sqrt(B) |x|_1, with B >= |D|_2^2 (sqrt(B) = 2^K bounds the
absolute row sums of D_K) and |y|_inf = lambda at the optimum, the largest
V/(eps S) over the 18 fits is 0.69 (f2 trend 0.5; 0.26 and 0.22 at 0.1 and 0.01,
0.28 on f1, at most 0.07 on f2's fused and elastic-net fits). The optimizer now
passes the floor 2 eps |y|_inf sqrt(B) |x|_1 to the stop rule, and
primal_dual_stop_rule declares converged at resolution when V is at most
resolution_multiple times the larger of eps |f| and that floor. f2's fits at a
budget of 1e6 (gcc, 4 threads):

| fit | before | with the floor |
|---|---|---|
| trend 0.5 | budget, 1e6 it, V 3.1e-8, coef err 7.5e-13 | at resolution, 108466 it, V 1.2e-7, f - f* 1.3e-8, 1.5e-12 |
| trend 0.1 | converged, 170807 it | at resolution, 164753 it, coef err 6.0e-11 |
| trend 0.01 | converged, 830577 it | converged, 830586 it |
| fused and fused_enet, 0.5, 0.1, 0.01 | converged, 2473 to 5564 it | the same |

On f1 (n = 200, p = 50) the floor is 2e-12 to 3e-10, below the tolerance, and no
status changes.

**Trend filtering by ADMM, for later.** A prototype (NumPy, one thread) of the
ADMM of Ramdas and Tibshirani (2016) on f2's three trend fits: the split
alpha = D_1 beta, the beta step a solve with the cached Cholesky factor of
X'X + rho D_1'D_1, the alpha step the fused-lasso prox of D_1 (Condat's algorithm,
checked on 300 random cases against the optimality conditions), stopped when
f - f* <= 1e-8 against the reference. The plain ADMM on z = D_2 beta is given for
comparison.

| rho / lambda | specialized, trend 0.5 / 0.1 / 0.01 | plain ADMM, the same |
|---|---|---|
| 1e-2 | 2100 / 5965 / not within 2e4 | |
| 1e-1 | 540 / 600 / 8280 | |
| 1 | 3590 / 3290 / 825 | 23050 / 40485 / not within 1e5 |
| 10 | not within 2e4 / not within 2e4 / 2995 | 59610 / 66095 / 73785 |

The specialized ADMM needs 100 to 1500 times fewer iterations than primal_dual
(108466, 164753, 830586), and each costs one pair of triangular solves of order p
and the O(p) prox, against a pass over X. Its costs are the choice of rho (a
factor of 10 away from the best costs 5 to 10 times the iterations, or does not
converge), a factorization for each rho (banded only when X = I), and a
certificate: the framework declares success only on a measure that bounds
f - f*, and ADMM's primal and dual residuals do not, so a duality gap would be
built from its dual variable. Not implemented; the route if trend filtering
becomes a model term.

**mcp and scad from the worsened start.** From the +1 start (every free
coordinate of the library start moved by 1) every single method tried ends in the
far basin of mcp, the Laplace limit at gamma -> infinity (section "Starting values
and the fit_distrib bench on the laws added after it"). A multistart over the
bench's Fisher scoring (step capped at 1), with starts by Latin hypercube within
3 spread max(1, |u0_j|) of the +1 start:

| law, n | 5 starts, spread 0.5 | 5, 1 | 10, 0.5 | 10, 1 | 20, 0.5 | 20, 1 |
|---|---|---|---|---|---|---|
| mcp 1e3: -l above R's | 6.5e-7 | 1.7e-5 | 9.7e-6 | 1.7e-5 | 1.5e-6 | 2.0e-5 |
| mcp 1e3: f evaluations | 1685 | 1538 | 6189 | 3778 | 8560 | 7826 |
| mcp 1e5: -l above R's | -2e-9 | boundary (4.2e3) | 1.7e-7 | 5.7e-7 | | |
| scad 1e3: -l above R's | 3.2e-7 | 3.2e-7 | 3.2e-7 | 3.2e-7 | | |
| scad 1e5: -l above R's | 1.1e-6 | 1.1e-6 | 1.1e-6 | 1.1e-6 | | |

With 10 starts and spread 0.5 the best of the runs is at R's maximum for both laws
at both sizes, within the kink bounds of the bench (6.3e-5 at n = 1e3, 2.7e-6 at
n = 1e5). 1 to 4 of the 10 runs reach the maximum, and the others reach the
boundary. Under the selection rule of the time, where a converged run beat a
non-converged one, the BFGS multistart kept a boundary run in 15 of the 18 mcp
and scad settings tried, although up to 5 of its runs reached the maximum. Those
runs ended stalled on the kink in mu, while the boundary runs ended converged.
The alternation (Brent on mu, Newton capped at 1) reaches the maximum in every
setting, at 1.6e4 to 4.1e5 evaluations of f.

**The multistart's selection rule: values first.** The old rule failed in the
same way under Fisher scoring once the gaussian_tail derivatives of mcp and scad
moved by a few ulp (a different route through the same formulas, on 2026-11-19).
At n = 1e3, one boundary run of the ten then ended converged (-l 2186.84 for mcp,
2033.30 for scad), and the multistart kept it over the three stalled runs at R's
maximum (-l 2106.98 and 1985.93). Three rules were compared, with each start's
report logged:

- R0, the old rule: a converged run beats a non-converged one; between runs of
  the same kind, a later run replaces the kept one only when its value is lower
  by more than distinct_tolerance.
- R1: values first. A later run replaces the kept one when its value is lower by
  more than distinct_tolerance, whatever the statuses, or when its value is within
  the tolerance, it converged, and the kept run did not.
- R4: R1 applied only to the runs that ended on their own (converged, stalled, or
  failed line search), so that a run cut by a budget is skipped.

The settings were the 17 laws of the bench at n = 1e3, from the library start and
from the +1 start, plus mcp and scad at n = 1e5, each with a multistart of 10
starts at spread 0.5 over Fisher scoring and over BFGS: 76 multistarts in all.
R1 and R4 keep the same run in all 76, and that run is at R's maximum in all 76.
R0 keeps the same run as R1 in 67. In the other 9, all mcp or scad, R0 keeps a
converged boundary run at -l 2157.03 to 2186.84 for mcp at n = 1e3 (R's 2106.98),
2027.60 to 2033.30 for scad (R's 1985.93), and 215022.96 for mcp at n = 1e5
(R's 210796.13). The Styblinski-Tang bench and the 30 nonconvex fits keep the same
run under all three rules. R1 was chosen (Giovanni, 2026-11-19): it equals R4
wherever it was measured, and it needs no fallback for the case where no run ended
on its own. The test asserts both the best run and the kept run at R's maximum
and off the boundary.

## The properizing tails of the MCP and SCAD laws: Student, Gaussian and logistic (2026-11-19)

**What.** `mcp_distrib` and `scad_distrib` take the tail that makes them proper
as a template parameter, whose default is `student_tail`; the Gaussian tail of
2026-11-18 is `gaussian_tail`, and `logistic_tail` is the third. The laws'
own parameters are unchanged, the tail's follow them, so a law has four
parameters with a one-parameter tail and five with the Student one. Decided with
Giovanni on 2026-11-19 after the question of what else could replace the flat
tail: three tails, not five, with the Student one as the default.

A tail is an exponent in `x = u - K`, the distance beyond the law's last knot,
of the form

    H(x; theta) = kappa(theta) Psi(x / varsigma(theta)),   x >= 0,

with `H(0) = H'(0) = 0`, so that `q` and `q'` stay continuous at the knot, and
with `varsigma` a MONOMIAL in the tail's own parameters:

    gaussian   Psi(z) = z^2/2          varsigma = V^(1/2)      kappa = 1
    student    Psi(z) = log(1 + z^2)   varsigma = s nu^(1/2)   kappa = (nu + 1)/2
    logistic   Psi(z) = 2 log cosh z   varsigma = s            kappa = 1

**Why three, and why the Student one is the default.** The penalties exist
because `q' = 0` beyond the knot leaves large deviations unpenalized, which is
what makes them asymptotically unbiased. The Gaussian tail destroys exactly
that: `q'(x) = x/V` grows without bound, so beyond `x = lambda V` the law
shrinks MORE than a rate-`lambda` Laplace, the opposite of what the MCP and the
SCAD are for. Measured at `s = 1` and `nu = 3`, with `x` the distance beyond the
knot:

    x        gaussian q'   logistic q'   student q'    x times the student q'
    10       10            2             0.3883495     3.8834951
    100      100           2             0.03998800    3.9988004
    1000     1000          2             0.003999988   3.999988
    10000    10000         2             0.000399999988 3.9999999

The logistic score is bounded by `2/s`, so with `2/s <= lambda` that tail never
shrinks more than a rate-`lambda` Laplace; the Student score is
`(nu + 1) x/(nu s^2 + x^2)`, which tends to zero, so `x q'` climbs to `nu + 1`
and the original behaviour survives up to `O(1/x)`. That is the whole argument
for the default. Three rather than more: the pure Pareto tail
`(1 + x^2/s^2)^(-c)` is the Student tail reparametrized, with `c = (nu + 1)/2`
and the scale `s sqrt(nu)`, and the pure exponential tail `x/s` has
`H'(0) = 1/s`, which breaks the `C^1` join and would add two knots to each law.
The power exponential tail `|x/s|^p/p` was left out: its score grows without
bound like the Gaussian one, so it buys nothing, and for `1 < p < 2` its second
derivative at the knot is infinite.

**How. Two identities carry every mixed partial.** Both derived and checked
against SymPy's own derivatives in `tools/reference/properized_tail_forms.py`.

(1) Through the knot. Both laws have `K = p lambda` (`p` the concavity gamma of
MCP and the shape a of SCAD), so for any `F` of `x = u - p lambda`

    d^j_u d^i_lambda d^m_p F(x) = sum_r c_r(i, m) F^(j + r)(x),
    c_r(i, m) = C(i, m + i - r) (-1)^m m!/(r - i)! lambda^(r - i) (-p)^(r - m),

over `max(i, m) <= r <= i + m`; the derivatives in `u` only shift the order, the
coefficients being free of `u`. This is `x_chain_coef`, and it replaces the
closed `q_tail_square` of the Gaussian-only version, which was the same
statement for a parabola.

(2) In the tail's own parameters. With `D = varsigma d/dvarsigma` and
`E = z d/dz`, every `theta_j` enters through `varsigma` alone, so
`theta_j d/dtheta_j = e_j D`, and the Stirling numbers of the first kind turn
`d^n` into `(theta d/dtheta)^k`:

    d^n1_theta1 d^n2_theta2 G = theta1^(-n1) theta2^(-n2)
        sum_(k1,k2) s1(n1,k1) s1(n2,k2) e1^k1 e2^k2 D^(k1 + k2) G,
    D^m [varsigma^(-r) Psi^(r)(z)] = (-1)^m varsigma^(-r)
        sum_q C(m,q) r^(m-q) sum_j S2(q,j) z^j Psi^(r+j)(z),

the second line because `D` acts on a function of `z` as `-E` and on
`varsigma^(-r)` as multiplication by `-r`, with `E^q f = sum_j S2(q,j) z^j f^(j)`.
So one component of the tail piece of `q` is a finite sum of values of `Psi`,
with integer coefficients and powers of `z`, `r`, `e_j` and `1/theta_j`: no Bell
polynomials, nothing on the stack beyond a few doubles, and one code path for
every tail. The Stirling values are generated where they are used, `s1` from the
falling factorial into a local array of seven doubles and `s2` from its closed
alternating sum, rather than from a `static constexpr` table, which in device
code has no definition at a runtime index.

Measured against SymPy's own derivatives, at the exact rationals of the doubles
the transcribed route uses: the two Stirling identities reproduce the falling
factorial and `E^n` to order six exactly, identity (1) is within 1.0e-15 over
every `(j, i, m)` of total order up to six, the components of `H` within
4.4e-16, 2.7e-14 and 3.0e-13 for the Gaussian, Student and logistic tails, and
the two identities composed - one component of the tail piece of `q` - within
8.9e-16, 8.1e-13 and 1.1e-14. The reference points must be exact: with `sp.Float`
substitutions, which carry fifteen digits, `evalf` cannot reach the thirty digits
asked of it, and the comparison then measures SymPy's own loss rather than the
transcription. That showed as 2.8e-10 on the logistic tail piece at order six,
where the transcribed route is right to 1.3e-14 (checked against mpmath's own
numerical derivative at 60 digits) while SymPy's unsimplified derivative at
`sp.N(..., 30)` was out by 3.7e-10.

The highest derivative of `Psi` reached is `r + j` with `r <= J + K` and
`j <= K`, so eighteen at `J = K = 6`; both transcriptions were measured against
mpmath at 60 digits to that order, over `z` from zero to 200, and are within
4.6e-14 and 3.5e-13 relative there.

**The two derivative sequences, and what was rejected.** For the Student tail
`Psi^(k) = N_k(z) omega^k` with `omega = 1/(1 + z^2)`, `N_1 = 2z` and
`N_(k+1) = N_k'(1 + z^2) - 2 k z N_k`, each term assembled as
`c_j (z omega)^j omega^(k-j)`, where `z omega <= 1/2` and `omega <= 1` for every
`z`, so nothing overflows at any order. The closed form
`(-1)^(k-1) (k-1)! 2 cos(k (pi/2 - arctan z))/(1 + z^2)^(k/2)` was written
first, is a single term and cancels nowhere, and was rejected on measurement:
where the odd orders vanish at `z = 0` it returns `(k-1)!` times the rounding of
the angle, 1.5e-14 at `k = 5` against an exact zero, which the polynomial gives.
For the logistic tail `Psi^(k) = 2 d^(k-1) tanh`, written in
`g = sech^2 z` as `R_n(g)` for odd `n` and `t R_n(g)` for even, with `R_0 = 1`,
`R_(n+1) = -2 g R_n'` from odd to even and `R_(n+1) = g R_n - 2 g (1 - g) R_n'`
from even to odd. Every `R_n` with `n >= 1` vanishes at `g = 0`, so the leading
term carries the value in the far tail and nothing cancels there; `g` itself is
built from `4 e^(-2|z|)/(1 + e^(-2|z|))^2` and not from `1 - tanh^2`, which
would lose every digit for large `|z|`.

**The constants and the partial integrals.** `Z = 2 [A(lambda, p) + B(lambda, p) T]`
in both laws, with `T = int_0^inf exp(-H)` the tail's own constant, so the tail's
parameters enter through `T` alone:

    T = sqrt(pi V/2)                                    the Gaussian tail
    T = s sqrt(pi nu) Gamma(nu/2)/(2 Gamma((nu+1)/2))   the Student tail
    T = s                                               the logistic tail

and the partial integrals, which the cumulative functions need, are

    int_x^inf e^(-H) = 2 sqrt(pi V/2) Phi(-x/sqrt(V))     the Gaussian tail
    int_x^inf e^(-H) = T I_w(nu/2, 1/2)                   the Student tail
    int_x^inf e^(-H) = s (1 - tanh(x/s))                  the logistic tail

with `w = varsigma^2/(varsigma^2 + x^2)` and `I` the regularized incomplete beta
(`math/incomplete_beta.hpp`). The Student constant comes from `v = varsigma t`
and then `w = 1/(1 + t^2)`, which turns the tail integral into a beta integral;
the Cauchy case `nu = 1` gives `T = s pi/2`. Three things were measured here.

- The closed `T` against the quadrature of `exp(-H)` at 40 digits: 1.4e-41 for
  the Gaussian tail and exact for the other two. Its mixed partials to total
  order six against SymPy: 1.4e-16, 1.6e-16 and exact, the Student one through
  `log T = log s + log G(nu)`, whose derivatives in `nu` are elementary plus one
  polygamma difference per order.
- The partial integrals against the quadrature of the definition, factored by
  the value at the left end so that the integrand starts at one: 3.9e-40,
  4.1e-41 and 1.3e-27 over `x` from zero to twenty.
- **Whichever of the two beta arguments is small must be formed directly**, never
  by subtracting the other from one. Just beyond the knot `1 - w` is of the
  order `x^2/varsigma^2`, which the quotient `v2/(v2 + x^2)` rounds away: at the
  second knot of SCAD with `lambda = 3`, `a = 1.1`, `s = 0.5` and `nu = 40`, at
  `x = 3.3e-7`, `log F` through `w` alone was out by 1.3e5 ulp of
  `max(|log F|, 1)` — the whole size of the increment of `F` across that step —
  and through `ibetac(1/2, nu/2, x^2/(v2 + x^2))` it is within two. The logistic
  tail needs the same care for the opposite reason: `s (1 - tanh(x/s))`
  underflows to zero from `x/s` of about nineteen, so `log_upper` is written as
  `log(2 s) - 2 z - log1p(e^(-2 z))`.

**The table of log Z.** With a one-parameter tail the table runs over
`(lambda, p, theta)` and the logarithm follows by the existing `log3`; with the
Student tail it runs over `(lambda, p, s, nu)` and needed `log4`, the same
recursion with one index more, and `idx4`, the triangular packing over four
variables, which at order six holds 210 entries against the 2401 of the square
box. `log4` was checked against SymPy's derivatives of `log Z` for MCP with the
Student tail, taking `A` and `B` from SymPy so that only the assembly and the
four-variable recursion were under test: 6.2e-15 at total order up to six.

**What was measured.** References from `tools/reference/distrib_mcp_scad.py`,
extended to carry a tail column and two tail parameters, with six parameter sets
per tail (the Student ones including `nu = 1`, the Cauchy tail, `nu = 0.7`,
which has no mean, and `nu = 40`, close to the Gaussian tail). The closed `Z`
agrees with the quadrature of the definition `Z = 2 int_0^inf e^(-q) du` to
1.1e-48 at 400 bits over all 36 cases, which is also the check that each density
integrates to one.

Over 74184 values, with 97668 rows in the file for the three tails together, the
worst errors of the log-density and its derivatives, in ulp of the largest
component of their order at that point:

    order    0    1    2    3    4    5     6
    ulp      2    6    6   10    9   25   163

and the logarithms of the cumulative functions within 2 ulp of
`max(|log F|, 1)`, `Z` within 2 ulp. The orders rise because `log Z` goes
through the multi-index recursions and the tail's components through the
Stirling expansion of its parameters, whose conditioning at a small lambda is
what the sixth derivative sees; the worst point at order six is SCAD with the
logistic tail at `lambda = 3`, `a = 1.1`, `s = 0.5`. The reference case runs on
the host in both builds, so the gcc and the CUDA figures are the same. The
Gaussian tail, measured again through the new route, is where it was on
2026-11-18 or slightly better: 3.66, 8, 6.3, 4.45, 11.2, 40.2 and 163 ulp for
the orders 0 to 6, 2.46 ulp of `max(|log F|, 1)` for the cumulative functions
and 1.33 ulp for `Z`.

Besides the references: the inner pieces equal `mcp_penalty` and `scad_penalty`
bit for bit in the value for every tail, and within 2 eps of `max(|value|, 1)`
in the derivatives, the two sides writing the same quantity differently; the
derivatives in the tail's own parameters are exactly zero there. The mass by the
panel quadrature with every knot as a cut is out by 3.3e-16, the increment of
`F` against the quadrature of the density by 8.7e-17, and every order against
central differences within 2e-6 of the order's scale, the tolerance the Gaussian
tail uses. The Gaussian tail is the Student tail's limit as `nu` grows at a fixed
scale: at `y` 5.1 beyond the knot the difference of the log-densities is 9.15e-1,
1.046e-2, 1.047e-4 and 1.049e-6 at `nu = 1e2, 1e4, 1e6, 1e8`, so `nu` times it
settles at 104.9 and the difference falls like `1/nu`.

**What is open.** No closed `expected_info` and no `prox_y` for either law with
any tail; the generic score-form route serves the information, and it converges
because `y_kinks()` declares every knot. No speed benchmark for these laws. The
degrees of freedom of the Student tail are weakly identified when only the tail
informs them, so in practice they will often be held with `fixed_distrib`; the
law admits them as a free parameter and nothing stops a fit from estimating them.

## The nonlinear conjugate gradient: five betas and the floor of its stationarity measure (2026-11-20)

**What.** `conjugate_gradient_optimizer` (include/laplace/optimizers/conjugate_gradient.hpp)
minimizes a C1 objective with p_k = -g_k + beta_k p_{k-1}, y = g_k - g_{k-1}, and
beta chosen by a member: Fletcher-Reeves g_k'g_k / g_{k-1}'g_{k-1}; Polak-Ribiere+
max(0, g_k'y / g_{k-1}'g_{k-1}); Hestenes-Stiefel g_k'y / p_{k-1}'y; Dai-Yuan
g_k'g_k / p_{k-1}'y; and Hager-Zhang (the default),
beta = max(beta_N, eta_k) with beta_N = (y - 2 p_{k-1} |y|^2 / p_{k-1}'y)'g_k / p_{k-1}'y
(Hager and Zhang 2005) and eta_k = -1 / (|p_{k-1}| min(0.01, |g_{k-1}|)) (Hager and
Zhang 2006). beta is set to 0 (a restart along -g_k) by Powell's test
|g_k'g_{k-1}| >= 0.2 g_k'g_k, every n directions, and whenever p_{k-1}'y <= 0 for
the betas that divide by it. CG enters the loop of BFGS and L-BFGS
(detail::quasi_newton_run) as a model of the direction: the strong Wolfe search,
the two-iteration confirmation of a convergence verdict and the statuses are
theirs; a direction that is not a descent direction restarts the model, and so
does a secant pair that fails the curvature test (which a strong Wolfe step along
a descent direction always passes). The loop searches from step 1, so the model
returns d_k = a_k p_k with the initial step of Nocedal and Wright (3.60),
a_k = tau_{k-1} g_{k-1}'p_{k-1} / g_k'p_k, tau_{k-1} the step accepted along
p_{k-1}; after a restart d = -min(1, 1/|g|_inf) g as in BFGS, with p = -g
unscaled (FR and PR+ are not invariant to the scale of p_{k-1}; a first version
stored the scaled direction and their recursions were wrong).

**The curvature constant of the Wolfe search.** Total value evaluations over the
battery below, all five betas: 160833 at c2 = 0.1, 211886 at 0.4, 756169 at 0.9.
The default is 0.1.

**The stationarity measure.** CG has no model B of the inverse curvature; with d
the next direction (scaled as above), three candidates were run along every
trajectory of the battery (the seven More-Garbow-Hillstrom problems at
f* = 0, 1, 1e3, 1e6; quadratics n = 10 with kappa 1e2 to 1e12 and n = 100 with
kappa 1e6; tolerance 1e-6 and the two-pass confirmation):

| measure | false successes (converged with f - f* > 1e-5) |
|---|---|
| (1/2) max(\|g'd\|, gamma g'g), gamma = max s'y/y'y (BFGS's) | every beta: kappa 1e8, n = 10, f - f* 3.4e-5 to 3.9e-3; kappa 1e6, n = 100, 1.1e-3 to 8.8e-3; FR, DY, HZ at kappa 1e6, n = 10, 2.0e-5 to 2.9e-5 |
| (1/2) tau g'g, tau the last step along p (the line-search step, as for B = I) | from kappa 1e4, up to 0.18 at kappa 1e8 |
| (1/2) max(\|g'd\|, gamma g'g), gamma = max s's/s'y | none; the largest f - f* of a converged run 7.2e-7 (HS, n = 100, kappa 1e6) |

s'y/y'y and s's/s'y are the inverse curvatures of Barzilai and Borwein's two
steps; on a quadratic both lie in [1/lambda_max, 1/lambda_min] and
s's/s'y >= s'y/y'y. The bound (1/2) g'g / lambda_min on f - f* is approached by
the larger one. The third measure is chosen (decided by Giovanni on 2026-11-20; the same
comparison was measured for gradient descent and BB). Its cost
on the More-Garbow-Hillstrom problems, evaluations summed over the four values of
f*, first measure against third: HZ 2344 and 2381, PR+ 2304 and 2605, HS 2426
and 2564, DY 2509 and 2671, FR 2746 and 2902. The quadratics take longer, because
the first measure stopped them early (HZ, kappa 1e8, n = 10: 1300 iterations at
f - f* 3.9e-3 against 3393 at 9.6e-8). It enters the loop through a declaration
of the model (floor_from_step); BFGS and L-BFGS keep s'y/y'y.

**Where it does not converge.** At kappa 1e10 and 1e12 (n = 10) every beta ends
stalled or line search failed, 1e2 to 1.5e4 above f*. Brown's badly scaled
function with f* = 1, 1e3, 1e6 ends stalled or line search failed within 2e-5 of
x* for most betas: its condition number is about 1e12, the attainable gradient at
x* is about sqrt(2 lambda_max eps |f|), and the measure, whose floor carries
1/lambda_min, stays near kappa eps |f|, above 1e-6 (FR and DY at f* = 1 and HZ at
f* = 1 and 1e3 converge). From the saddle start (1, 0) of x1^2 + x2^4/4 - x2^2/2
the gradient's second component is 0 on the whole line x2 = 0, so every beta lands
on the saddle (0, 0), a stationary point, in one step and reports converged; from
(1, 1e-3) every beta reaches a minimum.

**Linear CG as a check of the formulas.** With an exact line search on a
quadratic, no Powell restart and no periodic restart, every beta reduces to
linear CG; the values along the run agree with a textbook linear CG (step
r'r / d'Ad, beta = r_k'r_k / r_{k-1}'r_{k-1}) to a relative 1e-9 and f - f* is
below 1e-13 of its start after n steps (n = 5 at kappa 1e3, n = 10 at kappa 10).
At n = 10 and kappa 1e3 neither the textbook routine nor these betas terminate at
step n: both stall near f - f* = 1.6e-3 at steps 9 and 10 and end two steps later,
a loss of conjugacy in rounding.

**Against optimizers7 0.22.0.** Its cg() with the betas fr, pr, hs, dy (strong
Wolfe c2 = 0.1, step 1 first, no Powell restart), default settings, frozen on the
battery by tools/reference/optimizers_optimizers7_cg.R into
tests/data/reference/optimizers/optimizers7_cg_counts.csv (212 runs). On the 40
rows per beta that are More-Garbow-Hillstrom problems or the quadratics at
kappa 1e2 and 1e6, optimizers7 reaches f - f* <= 1e-5 with converged = TRUE in 21
(FR) and 24 (PR, HS, DY); the same beta here reaches it in 38, 37, 37 and 38 (the
rest are Brown with f* > 0), including every row where optimizers7 did. On the
rows where both succeed, value evaluations here against optimizers7: FR 1707 and
13950, PR+ 2054 and 8403, HS 2017 and 10762, DY 4072 and 54823; the difference
is mostly the initial step (3.60) against optimizers7's step 1.

**Cost.** O(n) work and 3n doubles of state per iteration; each Wolfe trial is
one value and gradient evaluation. Extended Rosenbrock with n = 100 from the
standard start, gcc 14.2 -O3, one thread, best of 50: FR 46.0 us (30 iterations,
88 evaluations), PR+ 50.0 (33, 94), HS 45.1 (29, 88), DY 48.5 (32, 92), HZ 47.7
(31, 93); L-BFGS with 10 pairs 82.3 us (36, 43). CG spends about three
evaluations per iteration at c2 = 0.1 against L-BFGS's 1.2 at c2 = 0.9: where an
evaluation dominates the cost (a large reduction on a GPU), L-BFGS needs about
half the evaluations on this problem.

## Gradient descent, Barzilai-Borwein, Adam and the nonmonotone line search (2026-11-20)

What: the first-order methods of wave 3 other than conjugate gradient
(note/optimizers.md, section 13): `gradient_descent_optimizer`
(`gradient_descent.hpp`), `barzilai_borwein_optimizer` (`barzilai_borwein.hpp`),
`adam_optimizer` (`adam.hpp`), and `nonmonotone_line_search` with
`nonmonotone_reference` in `line_search.hpp`. Tests:
`tests/optimizers/test_first_order.cpp`. References frozen from optimizers7
0.22.0: `tools/reference/optimizers_optimizers7_gd_bb.R` ->
`tests/data/reference/optimizers/optimizers7_gd_bb_counts.csv` (212 runs: gd()
and bb() with its three variants, the battery at f* = 0, 1, 1e3, 1e6).

### How

- The nonmonotone search (Zhang and Hager 2004) accepts a step s when
  f(x + s d) - C_k <= c1 s g'd, with C_0 = f_0, Q_0 = 1, and after each accepted
  f_{k+1}: Q_{k+1} = eta Q_k + 1, C_{k+1} = (eta Q_k C_k + f_{k+1}) / Q_{k+1}
  (eta = 0.85). Its backtracking is Armijo's: the body of `armijo_line_search`
  became `detail::backtracking_search`, which tests against a reference value;
  Armijo passes f(x) and is unchanged (a test checks that eta = 0 gives the same
  outcome, step, value and evaluations as Armijo, bit for bit).
- Gradient descent and Barzilai-Borwein share one loop (`detail::first_order_run`):
  d = -alpha_k g, searched from step 1. Scaling d rather than the initial step
  keeps the slope g'd finite where |g|^2 overflows: with d = -g and the step
  alpha_k, the start of cosh at 700 gave g'd = -inf and the run ended
  `line_search_failed` at iteration 0. Gradient descent: alpha_0 = min(1, 1/|g|_inf),
  then alpha_{k+1} = t_k g_k'g_k / g_{k+1}'g_{k+1} (Nocedal and Wright 2006,
  section 3.5), t_k the accepted step along -g. Barzilai-Borwein: BB1 = s's/s'y,
  BB2 = s'y/y'y or the two alternating, clipped to [1e-10, 1e10].
- Adam (Kingma and Ba 2015), deterministic: m, v, bias correction, the step
  u = -B m_hat with B = diag(lr / (sqrt(v_hat) + eps)). No line search: a
  trial point with a non-finite value or gradient is refused and the step
  halved, up to 60 times, then `line_search_failed` (decision 3, never a clamp).
- Stationarity (decision 5): (1/2) max(b, gamma_max) g'g, with b the last
  accepted step for gradient descent and the next alpha for Barzilai-Borwein;
  (1/2) max(g'Bg, gamma_max g'g) for Adam. A convergence verdict must hold at
  two consecutive iterations (V3).

### The floor gamma_max: s'y/y'y declares false successes, s's/s'y does not

Decision 5 takes gamma_max as the largest s'y/y'y over the accepted pairs.
First-order steps are dominated by the stiff directions, so s'y/y'y stays
near 1/lambda_max and the weak directions are never seen. Converged runs with
s'y/y'y, tolerance 1e-6, gcc, one thread:

| method | problem | f - f* | decrement | ratio |
|---|---|---|---|---|
| BB alternating | quadratic n = 10, kappa 1e6 | 1.15e-3 | 5.7e-7 | 2000 |
| BB alternating | quadratic n = 20, kappa 1e6 | 4.7e-3 | 9.8e-7 | 4800 |
| BB alternating | quadratic n = 50, kappa 1e4 | 9.1e-5 | 9.9e-7 | 92 |
| BB alternating | Rosenbrock n = 100 | 6.3e-5 | 4.0e-7 | 160 |
| gradient descent | Wood | 1.0e-4 | 9.1e-7 | 110 |
| gradient descent | Powell singular | 2.1e-5 | 9.2e-7 | 23 |

With gamma_max the largest s's/s'y (the BB1 step, the inverse Rayleigh
quotient of the average Hessian along s, at most 1/lambda_min on a quadratic),
every converged run of gradient descent (Armijo and nonmonotone) and of the
three BB variants, on the More-Garbow-Hillstrom problems at f* = 0, 1e3, 1e6
and the quadratics n = 10 to 100, kappa 1e2 to 1e10, has f - f* <= 2.0 times
the decrement (worst: Powell singular, 1.97). The cost: gradient descent needs
1.3 to 4 times more iterations (Wood 403 -> 1617, Powell singular 2137 -> 6425,
Rosenbrock n = 2 4842 -> 6596), and on Rosenbrock with f* = 1e6 it ends
`stalled` (decrement 3e-6, f - f* 6e-8, its predicted decreases below the
rounding of f, 2.2e-10) instead of converged. The same finding holds for
conjugate gradient (the section above). The floor is one function,
`detail::secant_floor` in `objective.hpp`, used by these methods and by
conjugate gradient (decided by Giovanni on 2026-11-20). Adam keeps
s'y/y'y: with it every converged run has f - f* <= 1.5 times the decrement
(Powell singular 1.52), no false success.

### Barzilai-Borwein after a refused pair: reset, not the previous alpha

When s'y <= 1e-10 |s| |y|, keeping the previous alpha is an absorbing state: a
short step samples the curvature over a short interval, on a valley floor that
curvature is not positive, and the next pair is refused too. Measured on
Rosenbrock (n = 2, f* = 0): BB2 held alpha = 9.3e-4 for 650 iterations, 759 in
all. Resetting alpha to min(1, 1/|g|_inf) gives 88 (alternating 253 -> 53,
BB1 80 -> 38); no other problem of the battery got worse. optimizers7's notes
record the same trap and the same remedy.

### Adam: an overflowing g^2 made the measure 0

At cosh(700) g_i^2 overflows, v_i = inf, B_ii = 0, so g'Bg = 0, and with
gamma_max = 0 the measure read 0: the run ended `converged` at the start. The
measure is now infinite whenever g'g is not finite; the run then ends
`stalled` (the step u is 0).

### The battery (gcc, one thread, f* = 0, defaults, tolerance 1e-6)

Iterations / value evaluations; "sad." is the stationary point near Wood's
saddle at f = 7.877, where L-BFGS also stops; "budget" the iteration budget.

| problem | GD Armijo | GD nonmonotone | BB alternating | BB1 | BB2 | Adam lr 1e-3 |
|---|---|---|---|---|---|---|
| Rosenbrock n = 2 | 6596/13224 | 5481/10979 | 53/110 | 38/87 | 88/189 | 11241/11242 |
| Rosenbrock n = 100 | 8114/16265 | 7050/14119 | 53/110 | 39/89 | 94/201 | 11772/11773 |
| Powell singular | 6425/12881 | 5260/10535 | 85/179 | 135/290 | 98/201 | 18061/18062 |
| Beale | 230/477 | 232/475 | 26/54 | 39/82 | 27/55 | 7740/7741 |
| helical valley | 1377/2782 | 1305/2625 | 68/139 | 81/173 | 115/233 | 15324/15325 |
| Wood | 1617/3261 | 3911/7839 | 22/45 sad. | 99/207 | 22/45 sad. | 12775 sad. |
| Brown badly scaled | budget | budget | budget | budget | budget | budget |
| quadratic kappa 1e2 | 197/417 | 172/355 | 62/125 | 55/113 | 62/125 | 18105 |
| quadratic kappa 1e4 | budget | budget | 662/1361 | 1047/2477 | 454/923 | 25056 |
| quadratic kappa 1e6 | budget | budget | 5484/11378 | budget | 5185/10445 | 74762 |
| quadratic kappa 1e8, 1e10 | budget | budget | budget | budget | budget | budget |

Every converged run has f - f* <= 2.8e-6 (gradient descent, nonmonotone, on Powell singular). The budgets are 1e4 iterations
(gradient descent, BB) and 1e5 (Adam). Against optimizers7 0.22.0: its gd()
(step 1 offered at every iteration, 500 iterations) converged on 12 of 53
runs; its bb() takes 51-58 iterations on Rosenbrock and reaches Wood's minimum
(its alpha_0 = 0.01 and Grippo-Lampariello-Lucidi memory 10), where the
alternating and BB2 runs here stop at the stationary point at 7.877. Its
successes are mostly change rules (|df| < 1e-10); the test requires that
wherever its gd() or bb() reached the minimum, the same method here succeeds
with f - f* <= 1e-5 (Wood excepted as above).

### Negative cases

No false success on the noisy objective (gradient descent stalled, BB
`line_search_failed`, Adam at its budget), on the quadratics at kappa 1e8
and 1e10 (budget), on the exploded start (not converged). On sum |x_i| from
integer starts BB lands exactly on 0 (f = 0, the minimum, a subgradient 0
there); gradient descent and Adam do not converge. On the saddle problem from
(1, 0), where g_2 = 0 exactly, every method reaches the saddle (0, 0), a
stationary point (Adam to within 2e-3, its measure 9.8e-7).

### The host vectors against a Kokkos gradient (benchmarks/optimizers_first_order_vectors.cpp)

Microseconds per call or per iteration. The objective is
f(x) = sum_i log cosh(x_i - c_i), with one row per coordinate; its value and
gradient are one parallel_reduce, and x goes down and g comes back at every
call. "host GD" is the host work of one gradient descent or Barzilai-Borwein
iteration on one thread (8 passes over p: the direction, the trial point, four
sums, three copies). "host Adam" is the same for Adam. "dev GD" is the same
work as host GD written as Kokkos kernels, with no copy between spaces. The
last three columns are a full iteration of the optimizer (30 iterations), with
the value evaluations per iteration in brackets. gcc 14.2 with OpenMP, -O3;
CUDA 13.1 on an RTX 4070 SUPER.

| backend | p | value | val+grad | host GD | host Adam | dev GD | GD it | BB it | Adam it |
|---|---|---|---|---|---|---|---|---|---|
| OpenMP, 1 thread | 1e4 | 627 | 661 | 14 | 48 | 21 | 2235 [5.3] | 1008 [2.2] | 727 [1.0] |
| OpenMP, 1 thread | 1e5 | 6291 | 7249 | 144 | 492 | 163 | 20516 [5.0] | 9904 [2.2] | 7261 [1.0] |
| OpenMP, 1 thread | 1e6 | 61659 | 70378 | 2158 | 5504 | 2221 | 197356 [4.7] | 102158 [2.2] | 73563 [1.0] |
| OpenMP, 16 threads | 1e4 | 111 | 119 | 15 | 49 | 104 | 654 [5.6] | 269 [2.2] | 192 [1.0] |
| OpenMP, 16 threads | 1e5 | 585 | 769 | 275 | 653 | 386 | 3211 [4.9] | 1813 [2.2] | 1548 [1.0] |
| OpenMP, 16 threads | 1e6 | 4719 | 6147 | 2481 | 5757 | 1722 | 21227 [4.0] | 14206 [2.2] | 13741 [1.0] |
| CUDA | 1e4 | 37 | 54 | 20 | 50 | 64 | 321 [6.3] | 136 [2.2] | 125 [1.0] |
| CUDA | 1e5 | 97 | 186 | 193 | 495 | 72 | 1111 [4.4] | 740 [2.2] | 868 [1.0] |
| CUDA | 1e6 | 619 | 1194 | 2697 | 5651 | 129 | 10047 [5.2] | 6714 [2.2] | 8629 [1.0] |

What it says. On one thread the host vectors cost 3 to 8 per cent of a
gradient, so they do not matter. With 16 OpenMP threads at p = 1e6 they cost
40 per cent of a gradient (GD, BB) and 94 per cent (Adam). On CUDA they
dominate from p = 1e5: at p = 1e6 one GD or BB iteration spends 2.7 ms on the
host against 1.2 ms for the gradient with its copies (2.3 times), and Adam
5.7 ms (4.7 times). The same work as kernels on the GPU takes 0.13 ms, 21
times less than on the host, and it would also remove the two copies per
evaluation. Below p = 1e4 the kernel launches cost more than the host loops
(dev GD 64 us on CUDA and 104 us with 16 threads, against 20 and 15 us on the
host). The state moves into Kokkos Views (decided by Giovanni on
2026-11-20 on this measurement).

## The prox of the MCP and SCAD laws, over the three properizing tails (2026-11-20)

**What it is.** `prox_y(v, t)` is the minimizer of `-log f(y) + (y - v)^2/(2t)`, which with
`U = |v - mu|` and `x = |y - mu|` is

    min over x >= 0 of  g(x) = t q(x) + (x - U)^2 / 2,

`q` the law's properized penalty. The normalizer does not depend on y, so it does not
enter. The minimizer carries the sign of `v - mu` and does not exceed `U` in modulus,
since `q` does not decrease. `g` is smooth on each piece of `q` and grows at infinity, so
its global minimum sits at `x = 0`, at a knot, or at a stationary point of one piece: the
implementation enumerates those candidates, compares `g`, and keeps the smallest `x` of a
tie, the rule the prox of `a|x|^p` already follows. An infinite point is its own prox,
whatever the step, and NaN passes through.

**Inside the knots** each piece is a quadratic and its stationary point is the classical
threshold:

| law, piece | curvature of g | stationary point |
|---|---|---|
| mcp, 0 <= x <= K = gamma lambda | 1 - t/gamma | (U - t lambda)/(1 - t/gamma), the firm threshold |
| scad, 0 <= x <= lambda | 1 | U - t lambda, the soft threshold |
| scad, lambda < x <= K = a lambda | 1 - t/(a - 1) | (U - t a lambda/(a-1))/(1 - t/(a-1)) |

A curvature at most zero leaves no interior minimum on that piece, so only its ends
compete; that is the nonconvex regime t >= gamma for mcp and t >= a - 1 for scad. At
t = gamma with U = t lambda, and at the matching point of scad's middle piece, the slope
vanishes together with the curvature and the objective is flat over the whole piece: every
point of it minimizes, and the convention of the smallest one is what makes the returned
value well defined. 168 of the 12243 reference rows are that case.

**Beyond the last knot the tail decides.** With w = U - K and z = x - K the stationary
equation is t H'(z) + z - w = 0, and the three tails differ in kind:

| tail | equation | roots | shrinkage w - z as w grows |
|---|---|---|---|
| gaussian | linear, H' = z/V | one, wV/(V + t) | w t/(V + t), grows without bound |
| student | the cubic z^3 - w z^2 + (varsigma^2 + 2 t kappa) z - w varsigma^2 | one or three, all positive | 2 t kappa / w, falls to zero |
| logistic | (2t/s) tanh(z/s) + z - w | one | 2t/s, a constant |

Each has a root with z > 0 only for w > 0. The Student tail's H' rises to kappa/varsigma
at z = varsigma and falls back, so the objective can hold two local minima beyond the
knot and the comparison of values is what picks between them; every real root of the cubic
is positive by Descartes' rule, so none is discarded. The roots come from the
trigonometric form when the discriminant is positive and from Cardano's otherwise, then a
Newton refinement of each root on t H' + z - w, which near a root is better conditioned
than the cubic, run until the step no longer moves the iterate. A fixed pair of steps is
not enough: the root the trigonometric form supplies was measured 4.4e6 ulp from the
converged one, which surfaced only once the roots were returned in ascending order and
the caller's tie rule began to prefer that copy over the converged one.
The logistic equation is increasing and concave, so Newton from max(0, w - 2t/s), where it
is at most zero, rises monotonically to the single root and is stopped when the step no
longer rises, which is rounding.

That table is the prox's side of the argument that made the Student tail the default. A
point far beyond the knot carries no information about the penalty's shape, and the
classical MCP and SCAD prox leaves it alone, their penalty being flat there. The Gaussian
tail pulls it back by a fixed fraction of its distance however far out it lies, the
logistic one by a fixed amount, and the Student one by less and less: only the Student
tail agrees with the classical operator in the limit.

**The discriminant cancels far out.** The cubic's discriminant is a difference of two
terms that both grow like w^6 and agree to leading order. Measured: at w = 1e11 its
computed sign comes out positive where the cubic has one real root, which sends the branch
test to the trigonometric form and fills the buffer with three numbers that are not roots.
The root near w is therefore added whatever that test decided, by Newton from w, where the
equation is increasing; a duplicate of a root already found costs one objective
evaluation. A test checks the stationarity and the knot comparison out to U = 1e15. The
Python mirror over 10080 cases out to U = 1.6e5 does not separate the two versions, so
this is a hazard the ordinary grid does not reach.

**Measured.** 12243 reference rows from `tools/reference/distrib_mcp_scad_prox.py`: two
laws by three tails, a grid in lambda, in the second shape, in U across the knots and the
thresholds and out to 10000 times the last knot, and in t through the convex, flat and
concave regimes. The reference comes from the definition rather than from the closed
candidates: at 200 bits, g' is bisected on every sign change of a scan whose nodes include
the knots, so that no bracket straddles one, with a geometric refinement on both sides of
each knot, since a stationary point of the tail hugs the knot whenever the point lies just
beyond it. Each row carries the objective's gap to the runner-up.

| what | figure |
|---|---|
| minimizer against the reference, 12075 rows with a separated minimum | 13 ulp, at scad's knot with t = 20 |
| objective at the computed point above the reference's, every row | 3.95e-16 relative at worst |
| first-order optimality where the minimizer is interior, 369 points | 7.01e-16 relative |
| host against device, 24 points for each of four law-and-tail pairs | 2.28e-16 relative |
| objective above the reference's, on the CUDA build | 5.38e-16 relative |
| smallest positive gap in the reference | 4.13e-8 |

No row has a gap between zero and 4.13e-8, so a flat optimum is never mistaken for a near
tie, and a near tie never decides a branch by rounding.

**What is open.** No expected information in closed form for these laws: `expected_info`
goes through `generic_info.hpp`. The order-zero derivation is settled and checked at 60
digits, and is written down where the work can be picked up: `I_mu_mu` needs the kink's
point mass, as the elastic net's does, and equals `(2/Z)[lambda + int q'' e^-q]`, the
integration by parts of `E[(q')^2]`; `I_mu_theta = 0` for every smooth parameter by
symmetry; one new closed constant per tail, `J = int H'' e^-H`, is `sqrt(pi/(2V))`,
`4/(3s)` and `(kappa/varsigma)[2 B(1/2, kappa + 3/2) - B(1/2, kappa + 1/2)]`, exact to
3e-61 against the quadrature of its definition; the smooth block is
`E[d2 q] + d2 log Z`, whose second term is already in the log Z table, and the inner
moments follow from the Dawson value by a recursion per piece, within 1.85e-54 of their
quadrature. What is not derived is the tail's second moments, `int (d_t H)^2 e^-H` and
their relatives, which reduce to `int Psi^2 e^-H`, `int Psi z Psi' e^-H` and
`int (z Psi')^2 e^-H`, closed in each tail. Two identities come free:
`int (H')^2 e^-H = J` and `int d_t H e^-H = -d_t T`.

## The generic prox in y of the continuous univariate laws (2026-11-20)

What. A proximal operator in y for every continuous univariate distrib that has
no closed form of its own (note/penalties.md section 5h), in
include/laplace/distrib/generic_prox.hpp:

  x*(v, t) = argmin_y  -t log f(y) + (y - v)^2/2,   t >= 0,

prox_y_by_newton(d, v, t), with distrib_prox_y(d, v, t) taking the member
prox_y where a law has one (laplace1/2, enet, powerexp, mcp, scad) and this
route otherwise. It needs log f to be C^2 in y and the law's weak-convexity
constant rho = max(0, sup_y l''(y)), l = log f, which
include/laplace/distrib/weak_convexity.hpp declares per law (0 for the
log-concave ones; finite for Student, Cauchy, lognormal, inverse Gaussian,
mcp and scad; infinite for gamma, Weibull, chi-square and beta with a shape
below 1 and for powerexp with p < 1). fixed_distrib and truncated_distrib read
their parent's rho; skewt, vonmises and transformed_distrib declare none and
have no prox. univariate_as_multivariate offers prox_y through
distrib_prox_y, so the independent product and distrib_penalty have it for
every such law; the penalty's prox descriptor is closed_form.

How. x* is the root of g(y) = y - v - t l'(y), g' = 1 - t l''. For
t rho < 1, g' >= c = 1 - t rho > 0: one root, at most |g(x)|/c from any point
x, so one evaluation at v brackets it without the mode; at t rho = 1 the
bracket comes from doubling steps; t rho > 1 returns NaN (several local
minima are possible; MFISTA and proximal Newton keep t rho < 1). The bracket
is clipped to the support: at a finite end the point used is the nearest one
at least the least normal number inside (next to 0 a subnormal y turns
(k - 1)/y into inf - inf), and when g has the interior's sign there the end
itself is the minimizer (the objective is convex on the closed support). A
density unbounded at an end has rho = infinity, hence NaN: the objective goes
to -infinity there. Inside the bracket Newton's step is taken unless it
leaves the bracket or fails to halve the step of two iterations before
(rtsafe), else the bracket is bisected, geometrically towards a finite end
when it is four times nearer to it. An infinite g' (l'' overflowing next to
an end) counts as no Newton step. A step below 4 eps |y| is accepted only
when g changes sign within one more such step: next to a singular end (beta
at y -> 1, where g' reaches 1e36) Newton crawls by steps of an ulp far from
the root, and the plain test stopped at 1 - 2.2e-16 where the root was 0.25.

Measured before writing (Python against a 200-bit minimization, 13 cases,
note/penalties.md 5h): within 2.1 ulp of the condition, 2.4 to 10.5
evaluations of g on average.

References. tools/reference/generic_prox.py, mpmath: log f from the definition
of each law in Laplace's parametrization (constants dropped), phi' by mpmath's
numerical differentiation, a bracket widened from v (or closed at an end of
the support, which is the minimizer when phi' keeps its sign up to it), then
mpmath's Anderson-Bjorck root finder, or bisection to the working precision
where that fails (phi' grows like exp(e^-z) in the left tail of the Gumbel
law); at 200 and 260 bits, kept when they agree to 2^-120 (every point did).
2400 rows: 20 cases (student_t1, student_t2, cauchy, two pseudohuber, logistic,
gumbel, two skewnormal, gaussian1, two gamma1, two weibull, lognormal1,
invgauss, beta, truncated student_t1 on [-1, 2], truncated gaussian1 on
[0.5, 3], exponential), 120 points each, v = c + s 10^U(-4, 4) with either
sign and below the support for one in ten, t = s^2 10^U(-6, 6) capped at
U(0, 0.999)/rho.

Error measure. ulps of x* over the condition number of the problem,
kappa = (|x*| + |v| + t |l'(x*)|)/(|x*| g'(x*)): rounding the terms of g at
their own size moves x* by kappa ulps. The optimality residual |g(x)| is
read in units of eps (|x| + |v| + t |l'(x)| + g'(x) |x|), the rounding of its
terms plus the change of g over one ulp of x, which dominates next to a
singular end (beta at x = 1 - 1.4e-9: 2.2e7 units of the terms alone).

**Tests** (tests/distrib/test_generic_prox.cpp, kernels on gcc and CUDA):

| check | gcc | CUDA | bound |
|---|---|---|---|
| x* against mpmath, ulp / max(1, kappa) | 5.53 | 3.45 | 8 |
| x* against mpmath, plain ulp (printed) | 19.4 | 18.4 | - |
| points at an end of the support equal to the reference's end | exact | exact | exact |
| optimality residual, units above | 11.9 | 7.92 | 18 |
| sign of g next to a returned end | right | right | exact |

Per case (ulp/kappa, gcc): Student, Cauchy, pseudo-Huber, logistic, gaussian,
lognormal, inverse Gaussian, beta and the truncated laws 0.25-1.5; Gumbel 2.24,
skew normal 3.17, Weibull 3.73, gamma1 with shape 1.25 5.53 (its l' carries
the partial cancellation of (s - 1)/y against s/y near 0). t = 0 projects v on
the support, t rho = 1.01 and rho = infinity (gamma, Weibull, beta with a shape
below 1) give NaN, NaN v and negative t give NaN; fixed_distrib gives its
parent's prox bit for bit; univariate_as_multivariate gives the member prox_y
bit for bit where it exists (laplace1) and prox_y_by_newton bit for bit
otherwise (student_t1).

Cost (C++ -O3, one thread, 4000 points per law, same ranges): evaluations of
g mean / max, ns per call. gaussian1 3.2 / 11, 31; exponential 2.2 / 3, 39;
student_t1 4.2 / 40, 1540; cauchy 4.1 / 43, 820; logistic 5.4 / 39, 1230;
gumbel 12.2 / 104, 630; skewnormal 6.0-6.5 / 13-87, 3200-3500; gamma1 9.6-14.5 /
51-73, 150-200; weibull 8.4-14.1 / 55-72, 900-1470; lognormal1 10.8 / 49, 310;
invgauss 11.5 / 58, 190; beta 12.9 / 54, 190; pseudohuber 6.3-7.5 / 50-53,
3500-27000. The cost of one evaluation is the law's logpdf_dy_dk<1, 0> and
<2, 0>: pseudo-Huber recomputes its Bessel ratio at every call.

Found on the way. gamma1_distrib::logpdf_dy_dk<1, 0> near y = 0 at shape
s = 1/phi = 1 forms s (mu - y)/(mu y) - 1/y, two terms of size 1/y that cancel:
at mu = 1.5, phi = 1, where l' = -2/3 exactly, it gives 0 at y = 1e-16 and
16384 at y = 1e-20. Reported to the distributions' owners; the test uses
gamma1(1.5, 0.8) meanwhile, end clipping being covered by exponential and the
truncated laws.

Open. A prox for t rho > 1 (the global minimizer among several stationary
points, which needs the mode and the shape of l''), and declarations of rho
for skewt and transformed_distrib, if a consumer asks.

## The expected information of the MCP and SCAD laws at order zero, in closed form (2026-11-20)

**What it is.** `mcp_distrib::expected_info()` and `scad_distrib::expected_info()` return
the whole matrix \(I(\theta)\) in closed form, for every properizing tail, where the two
laws went through `generic_info.hpp` before. The dispatch of `distrib_expected_info<K>`
is per derivative order, so order zero is replaced and the orders one to four still go
through the quadrature.

**The location needs the kink's point mass.** `-E[l_mu_mu]` misses it, so the location's
row comes from the score form, and one integration by parts turns it into an integral of
the a.e. second derivative plus a boundary term:

    I_mu_mu = E[(q')^2] = (2/Z) [ lambda + int_0^inf q'' e^-q du ],

the boundary term `lambda` being the jump of `q'` at the location, as `enet_distrib`
already does. `I_mu_theta = 0` for every other parameter, by the symmetry of the law in
`y - mu`, and is returned as an exact zero. The rest of the matrix is the hessian form,

    I_ab = E[q_ab] + d2_ab log Z,

with `log Z` from the order-two table the laws already build for their log-density and
`E[q_ab]` from the pieces below.

**What the inner pieces contribute.** Both laws have one quadratic piece, in `u` for MCP
and in `u - lambda` for SCAD, whose exponent's derivative vanishes at its upper end, so
the moments

    N_k = int_0^(p lambda) t^k exp(-lambda t + t^2/(2p)) dt,   k = 0, 1, 2,

with `p = gamma` or `a - 1`, carry every inner contribution. `N_0` is the Dawson value the
normalizing constant already holds. `N_1` and `N_2` follow by differentiating `N_0` in
`lambda`, whose upper end moves with it: with `x = lambda sqrt(p/2)`, `w = p lambda^2`
and `E = e^(-w/2)`,

    N_1 = p [E - F'(x)],   N_2 = 2 p^2 lambda E + p sqrt(p/2) F''(x).

Every term is positive, so nothing cancels however large `w` is, while the plain forms
`lambda p N_0 - p(1 - E)` and `(p^2 lambda^2 - p) N_0 + p^2 lambda (2E - 1)` lose about a
digit per decade of `w`: 7.2e-5 relative at `w = 1e6`, measured against mpmath. Below
`w = 1` both forms cancel the other way, by eps/w, and the double series in `w` serves
instead,

    N_k = (p lambda)^(k+1) sum_(m,n) (w/2)^m (-w)^n / [m! n! (k + 2m + n + 1)],

from expanding both exponentials on `t = p lambda s`. The three evaluations measured
against mpmath over `w` from 1e-6 to 1e6:

| w | plain | Dawson form | series |
|---|---|---|---|
| 1e-6 | 4.9e-10 | 1.5e-09 | 1.0e-16 |
| 1e-3 | 5.3e-13 | 5.3e-13 | 3.0e-16 |
| 0.1 | 4.8e-15 | 4.8e-15 | 2.5e-16 |
| 1 | 7.4e-16 | 6.5e-16 | 4.7e-16 |
| 5 | 4.7e-16 | 2.7e-16 | 8.2e-07 |
| 30 | 2.8e-14 | 3.8e-16 | 1.3e+20 |
| 1e3 | 1.1e-10 | 3.2e-16 | 1.3e+93 |
| 1e6 | 7.2e-05 | 2.9e-16 | 1.2e+240 |

so the implementation takes the series below `w = 1` and the Dawson form above it.

**What the tail contributes.** Beyond the last knot every second derivative of `q` is a
combination of `1`, `H'`, `H''` and the derivatives of `H` in the tail's own parameters,
so the tail supplies three integrals against its own weight,

    info_xx         = int_0^inf H''                   e^-H dx
    info_xt(j)      = int_0^inf d_x d_theta_j H       e^-H dx
    info_tt(i, j)   = int_0^inf d_theta_i d_theta_j H e^-H dx,

and the two identities `int H' e^-H = 1` and `int Psi' e^(-kappa Psi) dz = 1/kappa`
follow from `H(0) = 0`. Writing `H = kappa Psi(x/varsigma)` with `varsigma` a monomial,
every one of the three reduces to `T`, to `S_0 = T/varsigma` and to the three moments

    P_k = int_0^inf z^k Psi''(z) e^(-kappa Psi(z)) dz,   k = 0, 1, 2,

which is the tail's new `psi_curvature_moment`. No integral of `Psi` itself appears,
because `kappa` is affine in its parameter and its second derivative vanishes. The three
tails have them in elementary form:

| tail | P_0 | P_1 | P_2 |
|---|---|---|---|
| gaussian | sqrt(pi/2) | 1 | sqrt(pi/2) |
| logistic | 4/3 | (4 log 2 - 1)/3 | pi^2/9 - 2/3 |
| student | 2 nu S_0/(nu + 3) | 2(nu - 1)/[(nu+1)(nu+3)] | 2(nu - 3) S_0/[(nu+1)(nu+3)] |

the Student tail's being Beta integrals reduced to `S_0` times a rational function of
`nu`, so that no new special function is evaluated anywhere in the matrix: the whole
closed form reads Dawson's integral and its first two derivatives, the tail's own
constant, and rational arithmetic. The constants `4 log 2 - 1` and `pi^2/9 - 2/3` were
identified from 50-digit quadrature with mpmath's `identify` and confirmed to 1.3e-51.

**Measured.** The references are 1200 values over 100 settings, two laws by three tails
by five pairs `(lambda, p)` by three or four tail settings, from
`tools/reference/distrib_mcp_scad_info.py` at 200 bits. The reference comes from the
definition rather than from the closed forms: `I_mu_mu` is the quadrature of
`E[(q')^2]`, the arrangement that carries the point mass, and the smooth block is the
quadrature of `E[q_a q_b]` minus the product of the first derivatives of `log Z`. Against
that reference the closed assembly of the formulas, evaluated at 200 bits, agrees to
2.0e-41, so the derivation is settled before any rounding enters.

| what | figure |
|---|---|
| closed assembly against the reference, both at 200 bits | 2.0e-41 |
| the matrix in double | 58.7 ulp of the largest component |
| the generic score-form route against the same reference | 3.84 ulp, 4.14 on CUDA |
| the two routes against each other | 1.3e-14 relative |
| the two branches of the moments across w = 1, 5 crossings | 2.46e-15 relative |
| host against device, 6 law-and-tail pairs | 0 ulp (OpenMP), 1.65 ulp (CUDA) |

**What the closed form buys is speed and not accuracy.** The quadrature it replaces is
the more accurate of the two on this grid, 3.84 ulp against 58.7: a sum of a dozen closed
terms carries its own roundings, where an adaptive quadrature of one integrand accumulates
less, and the score form the generic route uses is besides self-consistent in the
normalizing constant, since the identity `E[q_a] = -d_a log Z` holds with an inaccurate
`Z` as well, so an error there enters twice with opposite signs. Both figures are below
1e-14 relative, so neither decides anything statistical; the figure that does decide is
the one in "Speed" below. The 58.7 ulp sit at scad with the logistic tail at
`lambda = 0.2`, `a = 1.4`, `s = 6`, where the assembly's condition number, the sum of the
moduli of its terms over the modulus of the result, is 7.4; that number stays between 1
and 9 over every setting of the grid and is the same for the three tails, so no single
term cancels and what is left is the rounding of the terms themselves, amplified by it.

**The first measurement was 706 ulp and the cause was not the assembly.** The Student
tail's own constant carried it: `log T` was a difference of two log-gammas of size
`nu/2`, hence an absolute error of about `nu eps/2`, which is 468 ulp of `T` at
`nu = 250`, and every component reads `T` through `Z`. That is why the two routes then
agreed with each other to the same order at which both left the reference, 1.84e-13: they
shared the constant. It is now evaluated through the half-step log-gamma ratio of
`math/stirling.hpp` (the next section), which took the whole matrix to the figures above
and the generic quadrature from 404 to 3.84 ulp.

**Speed.** `benchmarks/distrib_generic.cpp`, ns per call on one Serial thread of the gcc
build, both lock slots held:

| law and tail | member | generic, score form | generic, hessian form |
|---|---|---|---|
| mcp, Student (lambda = 1.3, gamma = 3, s = 0.9, nu = 3.5) | 1179 | 1392550 | 7068537 |
| mcp, Gaussian (lambda = 1.3, gamma = 3, V = 0.8) | 370 | 664464 | 2304021 |
| scad, Student (lambda = 1.3, a = 3.7, s = 0.9, nu = 3.5) | 1252 | 1671959 | 8467176 |

so the closed form is 1181, 1796 and 1335 times faster than the route it replaces, which
is the score form. For scale, `gaussian1`'s generic information costs 92 microseconds
against these 1.4 to 1.7 milliseconds: a kinked density with a heavy tail is where the
quadrature is most expensive, since the panels have to resolve the kink at the location
and the two knots, and the tail integrates slowly. The OpenMP column is not the
comparison to read, its member figures being 539 to 887 ns against 1.6 ns on one thread:
at this cost per call the launch is the whole measurement.

**What is open.** The derivatives of the information, orders one to four, still go through
the generic quadrature, which the per-order dispatch of `distrib_expected_info<K>` allows.
There the generic route has a gap that order zero has not: `I` does not depend on the
location, so every derivative component carrying `mu` is zero, and those are exactly the
components the generic route returns as NaN, the regularity rule of the score form
excluding them. A closed form would return the zeros, and that, rather than speed or
accuracy, is the argument for writing it. The route is short: the same independence of
`mu` kills those components outright, and every factor of the closed form depends either
on `(lambda, p)` or on the tail's parameters and never on both, so the derivative tables
are outer products of two small ones instead of one table over four variables. What it
needs that is not there yet is a reciprocal over four indices and the tables of the
tail's three integrals, which for the Gaussian and the logistic tails are monomials, their
`P_k` being constants.

## The half-step log-gamma ratio: math/stirling.hpp (2026-11-20)

**What it is.** `lgamma_half_ratio(a)` and its derivatives to order six are

    R(a) = log Gamma(a + 1/2) - log Gamma(a),   a > 0,
    R^(k)(a) = psi^(k-1)(a + 1/2) - psi^(k-1)(a),

which every law whose shape enters through `Gamma((nu + 1)/2)/Gamma(nu/2)` reads. The
plain difference of the two log-gammas loses one digit per decade of `a`, each of them
being of size `a log a` where `R` is only `(log a)/2`: measured against mpmath, 0.5 ulp at
`a = 0.15`, 28 at `a = 8`, 210 at `a = 125`, 3.6e8 at `a = 1e8` and 2.3e12 at the top of
the reference grid. The Student
properizing tail carried exactly that, 468 ulp of its own constant `T` at `nu = 250`, and
through `Z` it set the accuracy of the whole law.

**The route.** Stirling's error `delta`, which the same header already evaluates for every
`a > 0` with its derivatives to order six, takes the growth out:

    R(a) = (1/2) log a + S(a) + delta(a + 1/2) - delta(a),
    S(a) = a log(1 + 1/(2a)) - 1/2 = sum_(j >= 1) c_j a^-j,  c_j = (-1)^j/((j+1) 2^(j+1)),

the series being the one of `log(1 + u) - u` at `u = 1/(2a)`, differentiated term by term.
The two deltas are each of order `1/(12a)` and their difference smaller still, so nothing
cancels there; the head `(1/2) log a` carries the growth exactly.

Below the threshold the recurrence `R(a) = R(a + 1) - log((a + 1/2)/a)` brings the
argument up, every term of a step being elementary and exact and every term of one sum
having one sign. Where to put that threshold was the one choice here, and three were
measured. One is not available at all: at `a = 1` the series needs 84 terms at order six,
and the cap of 60 it is summed with left 6027 ulp there (476 at order five), which is how
the cap was found. Ten looked attractive, since at `a >= 10` both deltas are in their own
asymptotic series and the series needs 18 terms instead of 39, and the first comparison
seemed to favour it by a wide margin. That comparison was wrong: it put the cost of one
threshold in one region against the cost of the other in another region. Region by region,
on one Serial thread at order zero,

| region of a | threshold 2 | threshold 10 |
|---|---|---|
| (1e-9, 0.5] | 322 ns | 369 ns |
| (1, 2] | 297 ns | 345 ns |
| (10, 125] | 66.7 ns | 56.0 ns |

so the nine steps of the longer walk cost more than the twenty-one series terms they save,
and above the thresholds the two are the same code. The threshold is therefore 2, which
also measures better: 1.42 ulp1 for the value and at most 3.96 ulp for the derivatives,
against 4.88 and 9.81 at ten, where the sum of a longer walk, 1.25 against
`R(1) = -0.12`, weighs against a value that is small near its own zero. A variant that
used the plain difference below `a = 1` instead of the recurrence was measured and
rejected as well: 13 ulp at `a = 0.75` against 0.6.

**Measured.** 1386 reference rows per order at 400 and 500 bits
(`tools/reference/lgamma_half_ratio.py`, kept only where the two precisions agree to
2^-100), `a` log-uniform in [2^-30, 1e12] with a uniform part in [1/4, 40] and clusters at
the thresholds:

| order | this route | the plain difference |
|---|---|---|
| 0 | 1.42 ulp1, which is 198 ulp beside the zero | 2.3e12 ulp at a = 9.9e11 |
| 1 | 2.16 ulp | 2.6e13 ulp at a = 9.9e11 |
| 2 | 2.75 ulp | 1.6e12 ulp at a = 8.3e11 |
| 3 | 3.25 ulp | 1.0e12 ulp at a = 9.9e11 |
| 4 | 3.65 ulp | 1.2e12 ulp at a = 8.3e11 |
| 5 | 3.96 ulp | 1.3e12 ulp at a = 8.3e11 |
| 6 | 3.41 ulp | 1.2e12 ulp at a = 8.3e11 |

The CUDA build measures 1.97 ulp1 for the value, the 198 ulp beside the zero and the
derivatives being the same to the digits printed.

`R` vanishes at `a = 1.21163...`, where `Gamma(a + 1/2) = Gamma(a)` astride the minimum of
`Gamma`, so the value is reported in ulp1, the absolute error in units of the ulp of
`max(|R|, 1)`; the derivatives have no zero in `a > 0` and are reported in ulp.

**What it changed.** `student_tail::log_g_derivs` now reads `R` at `a = nu/2` instead of
differencing two log-gammas and two polygammas per order, since
`log Gamma(nu/2) - log Gamma((nu+1)/2) = -R(nu/2)` and the `k`-th derivative of it in
`nu` is `-2^-k R^(k)(nu/2)`. The expected information of the two properized laws, whose
whole matrix reads the tail constant through `Z`, went from 706 to 58.7 ulp of its largest
component over the reference grid, and the generic quadrature on the same grid from 404 to
3.84 ulp.

## The derivatives of the expected information of the MCP and SCAD laws, orders one to four (2026-11-20)

**What it is.** `expected_info_dk<K>` on `mcp_distrib` and `scad_distrib`, for `K` from one
to four, returns the derivatives of the whole matrix in closed form over every properizing
tail, where the per-order dispatch of `distrib_expected_info<K>` sent them to the generic
quadrature before. With the order zero of the previous section the two laws no longer use
that route at all.

**What the closed form buys first is the exact zeros.** The information of these laws does
not depend on the location, so every derivative component whose multi-index carries `mu`
is zero. Those are exactly the components the generic route cannot produce: the regularity
rule of the score form refuses them and returns NaN, the location being the parameter the
kink moves with. Of the multi-indices of a five-parameter law one in five carries `mu` at
order one and one in two at order four. The closed form returns the zeros as zeros, and
that, rather than speed, is the reason the orders were written.

**The structure that makes it short.** Every factor of the closed form of the previous
section depends either on `(lambda, p)`, with `p = gamma` for MCP and `a - 1` for SCAD, or
on the tail's own parameters, and never on both. So no table over four or five variables is
ever built: each factor carries a table over `(lambda, p)` of total order `K` or a table
over the tail's parameters, and the product of the two is an outer product. The helpers are
in `detail/properized_info.hpp`: `st_mul` for a product of two such split tables, `st_recip`
for a reciprocal, `st_add_outer` for a term that is a product of one of each, and
`st_add_lp` for a term that only moves with `(lambda, p)`. The reciprocal over all the
indices is needed once per entry, for the division by `Z`, and `assemble_info_dk` does the
rest with products and sums.

**The tables the factors need.** `monomial2` builds the table of `lambda^beta p^alpha` for
real `beta` and `alpha`, the falling factorials cancelling every slot past a whole exponent
on their own, so the same function serves the monomials of the numerators and the negative
powers of `p` that the matrix divides by. `moment_tables` builds the tables of the three
inner moments `N_0`, `N_1` and `N_2`, by the composition `dawson_term` already uses above
`w = 1` and term by term from the double series below it, where the series is a polynomial
in `lambda` and `p` and its table is therefore exact. The tail's three integrals are
tabulated by the new `const_table` on each tail; for the Gaussian and the logistic tails
their `P_k` are constants, so those tables are monomials in the tail's own parameter.

**Measured, three independent ways.** Each order against five-point differences of the
order below, which checks the whole chain down to the 200-bit references of order zero; the
generic quadrature at orders one and two, which is a second opinion on the values and not
only on their consistency; and mpmath differentiating the 40-digit quadrature on the
components where the two routes disagreed.

| what | figure |
|---|---|
| orders one to four against five-point differences | 3.15e-11, 2.02e-10, 6.99e-10, 1.29e-09 relative |
| orders one and two against the generic quadrature | 1.05e-15 relative over 474 components |
| the tables of the inner moments against differences | 9.6e-10 relative |
| host against device at order four | 0 ulp (OpenMP), 6.69 ulp (CUDA) |

The components the generic route is not asked for are of two kinds. The ones carrying `mu`
it returns as NaN and the closed form as exact zeros, checked as zeros against the
differences. The other kind found a defect in the generic route's own regularity rule: it
asked for the counts of one parameter at a time, so it allowed two derivatives in a
parameter of regularity one, while the knots of these laws move with `mu`, with `lambda`
and with `p` together. The first derivative of such an integral carries no boundary term,
the integrand being continuous at the knot, but the second one does, the first derivative
of the integrand jumping there. Measured on MCP with the Student tail at
`(lambda, gamma, s, nu) = (1.3, 3, 0.9, 3.5)`: the generic second derivative of
`I_lambda_lambda` in `gamma` was 2.32 times the closed one, while the first derivatives of
the two routes agree, the five-point differences confirm the closed second, and mpmath
differentiating the whole 40-digit quadrature agrees with the closed one to 0.0. The rule
now sums the counts over every parameter that may move the same kink, so the route refuses
those 28 components itself.

**Speed.** `benchmarks/distrib_generic.cpp`, ns per call on one Serial thread of the gcc
build, both lock slots held. The generic route is not timed past order two, where it already
costs 25 milliseconds a call; at order four it integrates 1050 components of a
five-parameter law at once and costs seconds.

| law and tail | order | member | generic, score form |
|---|---|---|---|
| mcp, Student | 0 | 3272 | 2898241 |
| mcp, Student | 1 | 7805 | 7568466 |
| mcp, Student | 2 | 21021 | 25199750 |
| mcp, Student | 3 | 48240 | not timed |
| mcp, Student | 4 | 107501 | not timed |
| mcp, Gaussian | 0 | 373 | 691606 |
| mcp, Gaussian | 1 | 1485 | 1591708 |
| mcp, Gaussian | 4 | 18283 | not timed |
| scad, Student | 0 | 3390 | 3116954 |
| scad, Student | 1 | 8668 | 8182121 |
| scad, Student | 4 | 107547 | not timed |

so the ratio grows with the order, 886 times at zero, 970 at one and 1199 at two for the
Student tail, the member's cost growing by a factor of about two and a half an order while
the quadrature's grows by three.

**The repair of the half-step ratio's cost.** The order-zero figures above are not the ones
first measured. The half-step log-gamma ratio that fixed the Student tail's constant cost
the member call 1179 to 7258 ns, 6.2 times, because `const_mixed` returns one component of
the tail's derivative table at a time and rebuilds everything behind it, which for the
Student tail is the derivatives of `G` and with them one value of the ratio an order. Over
the multi-indices of a table of order `K` that was the whole cost of the call. Each tail now
builds that table once, in `const_table`, and `log_z_table` reads it: 7258 back to 3272 ns
for the Student tail and 471 back to 373 for the Gaussian one, which is the figure the
difference of log-gammas had. Every printed figure of every check is unchanged by the
repair. What is left between 3272 and the 1179 of the difference of log-gammas is the
ratio's own cost, about 300 ns a call, asked three or four times per `const_table`.
Building the array of `R` and its derivatives in one pass would recover part of it,
and it does not fall out of the present route, the derivatives of Stirling's error
being a separate function per order. Decided by Giovanni on 2026-11-20 not to do it:
the ratio stays as one function per order, and the 2 microseconds stay where they are.

## Proximal Newton with an indefinite Hessian: the eigenvalues corrected (2026-11-25)

**What.** `proximal_newton_optimizer` takes the curvature H of the smooth part as
given. When H is indefinite the quadratic model of the subproblem,
g'd + d'Hd/2 + r(x + d), has no minimum for a penalty of linear growth such as the
lasso, the step it finds gives no decrease, and the run ended at iteration 0 with
`line_search_failed`. This happens with the observed Hessian of a Gaussian whose
scale is modelled too: per row, with r = y - mu and e = sigma^-2, minus the
log-likelihood has second derivatives e, 2 r e and 2 r^2 e in (mu, log sigma),
whose determinant -2 r^2 e^2 is negative, so the summed Hessian is indefinite
wherever the residuals are large against sigma (from a start at zero, for
example). `newton_optimizer` passes on the same model without the penalty because
it corrects the eigenvalues; the change brings the same correction into proximal
Newton.

**How.** At each iteration the Cholesky factor of H is attempted as before. Only
when it fails, the eigendecomposition H = V diag(w) V' is taken
(`laplace::dense::symmetric_eigen_decomposition`), and when the smallest
eigenvalue is below -floor, floor = eigen_floor max_i |w_i| (eigen_floor = 1e-8,
the default of `newton_optimizer`), H is replaced by
V diag(max(|w_i|, floor)) V'. The coordinate descent or FISTA on the model, the
inner measure and the decrement then read the corrected matrix. Three choices:

- The test is "indefinite", not "not positive definite". A positive semidefinite
  singular H, such as X'X with p > n, is left as it is: the penalty bounds the
  model there, and the Cholesky factor on the free coordinates (with the fallback
  measure of the section on SCAD and MCP) already handles it. Correcting it would
  add a ridge of 1e-8 of the largest eigenvalue and move those runs.
- The decrement of a corrected iteration is NaN, as in `newton_optimizer`, so no
  point where the observed H is indefinite is declared converged (at a saddle the
  step is 0 and so would be the decrement). The report keeps the factorization of
  the observed H in `final_factorization`.
- The damping of a coordinate with H_jj <= rho_j (a weakly convex penalty) acts
  on the corrected diagonal, as before on the observed one; the damping of an
  unpenalized coordinate with H_jj <= 0 cannot fire on a corrected iteration,
  whose diagonal is at least the floor.

The new report field `eigen_modified_iterations` counts the corrected iterations;
an infinite `eigen_floor` turns the correction off.

**Measured.** The test (`tests/optimizers/test_proximal.cpp`, "an indefinite
Hessian (the scale modelled) gets its eigenvalues corrected"): y_i ~
N(mu_i, sigma_i^2), mu_i = b_0 + x_i'b over 10 covariates (three slopes not
zero), log sigma_i = c_0 + c_1 w_i, 200 rows, a lasso with weight 20 on the ten
slopes, 13 coordinates, from zero, where H has no Cholesky factor.

| run | status | iterations | corrected |
|---|---|---|---|
| correction off | line search failed | 0 | 0 |
| correction on | converged | 9 | 3 |
| FISTA (proximal gradient) | converged | 261 | |

The corrected run reaches FISTA's point within 1e-5 with the same support (6 of
10 slopes), its final iteration is plain (positive definite factor, finite
decrement 8.1e-13), and the prox mapping at the point is below 1e-5 of the weight.
On the same kind of model at 10^5 and 10^6 rows (50 covariates, 53 coordinates)
the correction converged in 6 to 7 iterations against 9 to 10 with the expected
information as the curvature, with H corrected at the first assembly only.

No existing proximal test is touched: every proximal Newton run of
`test_proximal.cpp` and `test_nonconvex_fits.cpp` reports
`eigen_modified_iterations == 0`, so H is never changed and the iterations, points
and statuses are those of before (their curvature is X'X, positive definite or
positive semidefinite). The CUDA build gives the same counts on the new test
(9 iterations, 3 corrected, FISTA 261), and the whole suite, slow cases included,
ends with 0 failed assertions on gcc and on CUDA.

**Cost.** The decomposition is dense and cubic and is paid only on the iterations
whose Cholesky attempt fails. Eigen, one thread, a decomposition against a
Cholesky factor of the same matrix:

| n | Cholesky | eigendecomposition | ratio |
|---|---|---|---|
| 53 | 0.007 ms | 0.088 ms | 12 |
| 201 | 0.21 ms | 3.5 ms | 16 |
| 500 | 2.4 ms | 48 ms | 20 |
| 1000 | 15 ms | 363 ms | 24 |

The p > n fits (201 coordinates, X'X singular) pay one decomposition per
iteration without being corrected, about 3.5 ms.

**Open.** A large sparse curvature (many latent coefficients) cannot take a dense
decomposition, and its correction is not settled. The expected information as a
fallback when the correction fails (a failed decomposition, or a failed line
search after a corrected step) is not in the optimizer, which receives one
curvature; it is left to the caller.

## The engine's row kernels and inner fit (2026-11-25)

**What.** The first milestone of the fitting engine (`include/laplace/engine/`): a model as
data for the general route (stored design rows of fixed width for the linear nodes, the
nonlinear pointwise nodes `link` and `logistic_curve` walked with a switch on their kind, the
distrib a template argument chosen by a switch on the host over a list given by the caller,
the number $R \le 4$ of linear predictors a constant of the kernel); four row kernels, each
computing its quantity and nothing above it (the value $f = -\sum_i w_i \ell_i$, also its
gradient, also the observed information, or also the expected information
$\sum_i w_i J_i^\top I(\theta_i) J_i$, which needs the nodes to order one only); values that
carry their support, with derivatives packed on the sorted index tuples; the Hessian of a
model with one grouping as an arrowhead, factorized by a sparse $LDL^\top$
(`laplace::sparse::sparse_ldlt`, Eigen's `SimplicialLDLT` with the approximate minimum degree
ordering, in the static library `laplace_sparse`) behind `sparse_curvature_solver`; and the
inner objective `row_model_objective` for `newton_optimizer`, with quadratic penalties at
fixed hyperparameters, held coordinates, the cache of the last assembly and three
curvatures (observed, expected, observed replaced by expected where it is not positive
definite). The derivatives of the logistic curve were derived and checked with SymPy
(`tools/reference/engine_nodes.py`).

**Correctness** (`tests/engine/`, gcc and CUDA).

- Three simulated models, 300 units, at a point away from the optimum: a Gaussian with a
  random intercept (10 groups) and a log link on the scale; a gamma whose mean is the logistic
  curve of three predictors, one of them through a log link (a nested node), with a random
  intercept in the asymptote; a Student t with location, scale and degrees of freedom
  modelled (separable). Against a twin written in the test (theta and its derivatives in the
  predictors by hand, composed with dense tensors over the unsorted units): value, gradient,
  observed and expected information within $10^{-12}$ of the largest entry. Gradient against
  central differences of the value, observed information against central differences of the
  gradient: within $10^{-7}$ of the scale.
- Two models with a reference inner fit from hand-written engines written before the library's
  engine (the reference coefficients and data are frozen in the tests, `tests/data/engine/`):
  a Gaussian with a cubic B-spline and its roughness penalty, a random intercept and the scale
  modelled and held, 2000 rows, reproduced to $10^{-9}$ in every coordinate; a gamma whose mean
  is the logistic curve with a random asymptote, 2000 rows, to $10^{-9}$ relative.
- Dense and sparse curvature reach the same point ($10^{-7}$); the fallback to the expected
  information is counted; held coordinates keep their value; asking again at the fitted
  point runs no kernel.

- CUDA: the same 9 test cases and 1498 assertions as gcc, 0 failed.

**Open.** Order three and four of the kernels and the outer criterion (the next milestone); the
start; more than one grouping; local slots above 32 (team scratch memory); the block size on the
GPU, not tuned.

## The engine's marginal criterion (2026-11-25)

**What.** The second milestone of the fitting engine (`include/laplace/engine/`): derivatives
of the row kernels to order four, and the Laplace marginal criterion with its exact gradient and
Hessian. A value now carries its third and fourth derivatives in the linear predictors, packed
on the sorted triples and quadruples (`predictor_value<R, K>`, positions in closed form checked
at compile time against `distrib::sym_tensor_position`), and the chain rule has one function
for each of the orders three and four (one term for each partition of the index set, summed only
where the blocks lie in the supports of the inputs). The logistic curve has its derivatives to
order four, written through $f = A\,h(x_m, s)$, $h = g(z)$, $g = 1/(1 + e^z)$, and derived and
checked with SymPy (`tools/reference/engine_nodes.py`, 34 closed forms against SymPy's own
derivatives of $f$). Two passes over the units (`outer_rows.hpp`): of order three, the vector
$\tau_c = \sum_{rs} (H^{-1})_{rs} T_{rsc}$; of order four, also $T[v_a]$ on the arrowhead,
$T[v_a, v_b]$ and $q_{ab} = \sum (H^{-1})_{rs} Q_{rstu} v_{a,t} v_{b,u}$ for the outer directions
$v_a$, with $T$ and $Q$ the third and fourth derivatives of $-\sum_i w_i \ell_i$ and every
contraction taken once per unit. The criterion (`laplace_marginal_criterion`) is
$V(u) = F(\hat x; \rho) + \tfrac12\log|H| - \tfrac12\log|S_\lambda|_+$ over the log
hyperparameters of the penalty terms and the coordinates estimated on it, with every other free
coordinate integrated; $H$ is the observed Hessian, $V = +\infty$ where it is not positive
definite or where the inner fit fails (both counted), and $H^{-1}$ is a dense matrix formed from
the sparse factorization. The outer fit is `newton_optimizer` on the exact Hessian.

**Correctness** (`tests/engine/test_marginal_criterion.cpp`, gcc and CUDA).

- Three simulated models, away from the optimum of the criterion: a Gaussian with a random
  intercept (400 units, 10 groups; $\log\lambda$ of the group prior and the two coefficients of
  the log scale on the criterion), a gamma whose mean is the logistic curve fed by a log link
  (400 units, 10 groups; $\log\lambda$ and $\log\phi$), a Student t with three modelled parameters
  (300 units; $\log\nu$). The exact gradient against central differences of $V$ at steps $h$ and
  $h/2$ ($h = 10^{-3}$) extrapolated by Richardson, the exact Hessian against those of the
  gradient:

  | model | gradient error / scale | Hessian error / scale |
  |---|---|---|
  | Gaussian, gcc | $7.5\cdot10^{-11}$ / 22 | $1.7\cdot10^{-11}$ / 735 |
  | gamma with the logistic curve, gcc | $6.4\cdot10^{-11}$ / 62 | $2.7\cdot10^{-11}$ / 144 |
  | Student t, gcc | $1.3\cdot10^{-10}$ / 4.2 | $1.5\cdot10^{-10}$ / 16 |
  | the three, CUDA | at most $2.3\cdot10^{-10}$ | at most $1.5\cdot10^{-10}$ |

  The bound of the test is $10^{-8}$ of the scale. Each evaluation of the differences starts its
  inner fit at the inner minimum of the centre: started from the previous point, the inner
  residual (about the square of the distance moved) enters $\tfrac12\log|H|$ at first order and
  left an error proportional to the step ($10^{-6}$ at $h = 10^{-6}$ on the gamma).
- Two models with a frozen reference fit (the data of the inner-fit tests, 2000 rows, frozen in
  `tests/data/engine/`). A Gaussian with a cubic B-spline and its roughness penalty, a random
  intercept and the log scale modelled (four outer coordinates): the criterion at the
  reference optimum matches the reference to $10^{-12}$ once the constant $\tfrac{n}{2}\log 2\pi$
  that the reference leaves out is added, and a complete fit from $u = 0$ reaches its point to
  $3.8\cdot10^{-8}$ in 8 outer iterations. A gamma whose mean is the logistic curve with a random
  asymptote (two outer coordinates): the criterion at the reference optimum matches to
  $4\cdot10^{-12}$ relative, and a complete fit from the reference's start takes the same five
  Newton steps and ends on the same point.
- Where the inner fit started next to the minimum stalls (its step below the rounding of the
  coordinates before the decrement falls below its tolerance) its point is used and the stall is
  counted (`marginal_criterion_counts::inner_stalls`); this happened on $10^6$ rows near the
  optimum.

**Open.** The expected information in $\log|H|$ with its derivatives; $H^{-1}$ by selected
inversion and the arrowhead block formula for $\operatorname{tr}(H^{-1}\dot H_a H^{-1}\dot H_b)$
in place of dense $N \times N$ matrices; a polishing Newton step at the inner minimum (a cold inner
fit left $3.8\cdot10^{-8}$ in $V$ on the gamma); more than 16 local slots or 4 outer directions in
the order-four pass; more than one grouping.
