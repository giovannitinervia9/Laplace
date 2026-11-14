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
OpenMP (4 threads), taken while another session's CUDA build (nvcc, ptxas) was
running: log P 685 / 1044 / 1283 and 179 / 281 / 338; log Q 687 / 1178 / 1473
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

## Piecewise Chebyshev expansions of the derivatives of log Phi on [-1, 4)

**What.** `log_Phi_d1..d5` in `include/laplace/math/normal.hpp` on t in
[-1, 4) (written by an agent, reviewed and verified by the session):
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
`exp`, `expm1` and `log` (written by an agent, reviewed and corrected by the
session). sqrt by Karp's method, scaled by an even power of two outside
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

The agent's version failed three ways, each fixed by the session: sqrt gave
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
elsewhere (written by an agent, reviewed and corrected by the session).
Q = erfc(s)/2 + R with s = sqrt(a phi), R = e^-(a phi) S(a, eta) / sqrt(2 pi a),
S = sum_(k<=9) c_k(eta) a^-k, the c_k Boost.Math's 53-bit polynomials
(igamma_large.hpp, Boost licence); erfc as e^-y erfcx(s) so that nothing
underflows, and the logarithm of the small tail as -y + log(bracket). The
classic route needs O(sqrt(a)) terms near x = a; this one is O(1).

**Measurement.** New references `gamma_{p,q}_large.csv` and
`log_gamma_{p,q}_large.csv` (2400 rows: a log-uniform on [1e2, 1e15], x =
a (1 + d), |d| log-uniform on [2^-40, 1], both signs). Two defects in the
agent's references, both repaired: mpmath.quad on [0, x] without breakpoints
near x, where the mass sits (wrong by a factor 0.58 at a = 1e12, x 24 sqrt(a)
below the mode), and the integrand not scaled (values like e^-288 defeat
quad's absolute error control; scaled by its maximum on the interval, the
values agree to 30 digits across breakpoints and degrees); and the logarithm
of the complement taken as log(1 - p) at 200 bits instead of log1p(-p). The
agent's switch at |eta| <= 3 was wrong: Boost's c_k are polynomials truncated
about eta = 0, and the route, measured on the logarithm of the small tail, is
within 6 ulp for |eta| < 0.5 at every a, 11-38 ulp for |eta| in [0.5, 1) and
up to 1.7e3 beyond; the switch is now 0.5 (beyond, the classic route needs a
few tens of terms). Result, gcc (CUDA within the same test bounds): P and Q
within 0.82 ulp of max(|f|, 1) and 2.9-3.4 ulpL (the small tail carries
about a phi ulps from its argument: 1.0e3-1.3e3 plain ulps where a phi is
near 500); log P and log Q within 5.9-6.4 ulp of max(|f|, 1). Every call
within 5 us on one thread, against O(sqrt(a)) terms of the classic route
(1e6 at a = 1e12). The agent's bench ran for hours: Boost outside its own
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
plain relative ulp is the measure. Written by two agents (Haiku) from a brief,
reviewed and optimized in this session.

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

ns per call on one gcc thread (`benchmarks/math_rel.cpp`, the other session
building CUDA at the same time): exprel 16-44 (the value 16-31, the
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

**How.** Written by five agents (Haiku) from one brief, reviewed and
optimized in this session. The forms, each checked against SymPy's
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

**Accuracy.** Each cause measured with mpmath (scripts in the session's
scratch directory, not kept): the ratio of the sum of the absolute values
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

ns per call on one gcc thread (`benchmarks/math_stirling.cpp`, another
session's agent building at the same time): direct 5-37; recurrence 59-307
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
x = a e^(z/2)), one gcc thread / OpenMP with 8 threads, measured while another
session's CUDA test was running: P orders 1-6: 1197/160, 1400/182, 1581/212,
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
error of every mass that no sum can remove. The log F session then wrote the
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
test, fixed by the log F session (LAPLACE_HD).

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
polynomial of 1/phi and of y/mu of one sign. In the same session the Poisson
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

**Open.** The CUDA build was not run for these routes (no CUDA build in this
session). The gaussian1 route below x = 4 is unchanged: there the order-6
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

**Measurement.** References `pdmat_pilot.csv` (`tools/reference/pdmat_pilot.py`,
written by a Haiku agent): the matrices built from their definitions in mpmath
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
(survey in the session notes). At S = 0 an entry of the order-K derivative
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

## Univariate distributions, batch 1: the location-scale families in closed form

**What.** `include/laplace/distrib/`: `gaussian2.hpp` (mu, v = sigma^2),
`gaussian3.hpp` (mu, tau = 1/sigma^2), `cauchy.hpp`, `logistic.hpp`,
`gumbel.hpp` (all three in mu and sigma) and `laplace2.hpp` (mu, lambda = 1/sigma),
each with the whole interface of `core.hpp`: the parameter descriptors, the
log-density and its derivatives to order 6, the mixed (y, theta) derivatives,
the expected information and its derivatives to order 4, cdf, ccdf, log F,
log(1 - F) with their derivatives to order 6, and the quantile. All six are
continuous with a fixed support; only laplace2 has kinks, those of laplace1
(mu is C^0 in log f and C^1 in log F at mu = y). Written by three Haiku 5.5
subagents, one pair each, and reviewed here. Tests in
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
caught by the agent writing gaussian2 and gaussian3 against SymPy.
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
measured. Every other test file of the project that writes a `KOKKOS_LAMBDA`
in a TEST_CASE body carries the latent collision above and passes only because
its counters do not collide today.

**Chosen.** The six distributions as written, with the bounds of each test at
about 1.5 times the measured maximum, on gcc and on CUDA.

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

It has been now, by the session that owns `include/laplace/math/normal.hpp`,
which added the relative error of each order on [-1, 4) to `check_deriv` in
`tests/math/test_normal.cpp`, away from the zeros of the component (|g| at
least a hundredth of the largest |g| of that order, since a component crosses
zero inside the band and nothing can be relatively accurate there). Orders 4,
5 and 6 carry 43.2, 65.3 and 36.8 ulp where the earlier measure saw 0.177,
0.228 and 0.429. That is the size of the 90 ulp above, so gaussian2's 117 is
accounted for by a chain costing at most 28 and inputs carrying tens, with
nothing wrong in the composition.

**What is open.** Whether those 43 to 65 ulp on [-1, 4) can be reduced, which
is a question about how the Chebyshev pieces of that band are generated and
whether the expansion is in the right variable for the high orders. It bounds
what any chain built on those derivatives can achieve, gaussian1, gaussian2
and gaussian3 alike.

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
  M^-1[F] = sum_(T in F) Y[T]^T Y[F \ T] by Leibniz; one call of
  `cholesky_dK` per sub-tuple, about 3^K p^3/6 + 2^K p^3/3 operations, and a
  derivative is zero by structure when every product has a zero factor.
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
coordinates:

| B, p | M^-1, K = 0 | M^-1, K = 6, pure | M^-1, K = 6, distinct | log det, K = 0 |
|---|---|---|---|---|
| log_cholesky, 3 | 46 / 15 / 2.1 | 16100 / 2560 / 782 | 6130 / 1136 / 232 | 2.3 / 3.5 / 1.0 |
| log_cholesky, 4 | 105 / 27 / 2.8 | 26582 / 3970 / 1379 | 9982 / 1702 / 456 | 2.5 / 3.0 / 1.1 |
| partial_corr, 3 | 104 / 26 / 2.7 | 18677 / 3483 / 756 | (q = 3) | 83 / 14 / 2.1 |
| partial_corr, 4 | 185 / 39 / 4.2 | 29347 / 5003 / 1402 | 13710 / 2531 / 616 | 171 / 27 / 3.4 |
| matrix_log, 3 | 474 / 111 / 55 | 9493 / 1832 / 1469 | 169617 / 35381 / 29282 | 4.6 / 26 / 9.0 |
| matrix_log, 4 | 1123 / 166 / 127 | 11915 / 2937 / 3308 | 296256 / 64124 / 61653 | 3.0 / 32 / 10 |

The pure tuple is the dear one for the generic route: every one of its 63
sub-tuples is a nonzero derivative of L (for log_cholesky the diagonal
coordinate (0, 0) moves the whole column 0), while six distinct coordinates
leave most L[T] structurally zero. Equal sub-tuples are computed again (64
calls where prod (m_v + 1) = 7 would do for the pure tuple); merging them is
the known ceiling, marked in the code. For matrix_log the order of cost is
reversed (the bijector's Taylor arithmetic grows with the number of distinct
coordinates). log det costs 1-171 ns, partial_corr's being the q
logarithms of cosh.

**Open.** The sub-tuple merging above; the generic route needs the
Cholesky factor (`closed_factor`): a structure with neither M^-1 nor L in
closed form would need the factor computed from M first.

**Chosen.** This version; tested in `tests/pdmat/test_unstructured.cpp` on
gcc and CUDA, bounds 1.5 times the larger measured error.
