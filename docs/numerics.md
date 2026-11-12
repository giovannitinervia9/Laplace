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

## Shape derivatives of log P(a, x) and log Q(a, x), orders 1 to 5

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
listed above were not reduced.

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
