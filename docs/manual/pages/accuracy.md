# How accuracy is measured {#accuracy}

Every numerical module in Laplace documents the measured accuracy of its
functions, accompanied by error bounds enforced in automated tests. On module
pages, readers encounter figures such as "4.8 ulpC", "measured max error 44.4 ulp",
or "test bound 66.0 ulp". This page explains how every
accuracy figure in this manual is obtained, how the error metrics are defined,
and what guarantees the test bounds provide.

Laplace uses a four-step accuracy verification pipeline:
1. **Reference generation:** High-precision reference tables are generated using
   arbitrary-precision arithmetic in Python with `mpmath` (Johansson and others).
   Values are stored in hexadecimal format as double-double numbers of about 106
   bits of precision (Dekker 1971), rendering reference error negligible compared
   to standard 53-bit double precision.
2. **Backend evaluation:** Candidate functions are evaluated inside Kokkos
   parallel kernels on both CPU and GPU execution spaces.
3. **Error quantification:** The computed double-precision values are compared
   against the reference values to evaluate error in units in the last place (ulp)
   across multiple metrics (relative ulp, absolute ulp1, logarithmic ulpL, and
   conditioned ulpC; Goldberg 1991; Muller et al. 2018).
4. **Automated regression testing:** doctest test suites execute inside Kokkos
   kernels across both backends, asserting that every function remains strictly
   within its prescribed error bound and reproduces exact non-finite values.

The complete record of all numerical comparisons, candidate implementations,
and algorithmic choices is maintained in the public log
[`docs/numerics.md`](https://github.com/giovannitinervia9/Laplace/blob/main/docs/numerics.md).

## Reference generation

Reference values are produced by standalone Python scripts located in
`tools/reference/`, executed in isolated mode with `python -I` (Python 3.13 and
mpmath 1.3.0 as of 2026-11-11; Johansson and others).

### Precision and representations

Most reference scripts set the working precision to `mpmath.mp.prec = 200`
bits, well beyond the 53 bits of IEEE 754 binary64 double precision. Specific
functions where cancellation or large arguments demand extra headroom raise the
precision to 256 or 300 bits.

For example, when generating references for `bessel_k`
(`tools/reference/bessel_k.py`), mpmath's standard `besselk` function fails
for orders near \f$\nu = 10^6\f$ due to instability in its hypergeometric series
(`docs/numerics.md`). The reference script instead evaluates the integral
representation (DLMF 10.32.9):
\f[
K_\nu(x) = \int_0^\infty e^{-x \cosh t} \cosh(\nu t) \, \mathrm{d}t
\f]
where \f$\nu \in \mathbb{R}_{\ge 0}\f$ is the real order, \f$x \in (0, \infty)\f$
is the positive real argument, and \f$t \in [0, \infty)\f$ is the real
integration variable. Defining the positive weight function
\f$w(t) = e^{-x \cosh t} \cosh(\nu t) \in (0, \infty)\f$, the logarithm is
\f$\log K_\nu(x) = \log \int_0^\infty w(t) \, \mathrm{d}t\f$, and its derivatives
with respect to \f$x\f$ of order \f$n \in \{1, \dots, 7\}\f$ are:
\f[
\frac{\mathrm{d}^n}{\mathrm{d}x^n} \log K_\nu(x) = (-1)^n \kappa_n
\f]
where \f$\kappa_n \in \mathbb{R}\f$ is the \f$n\f$-th cumulant of \f$C = \cosh T \in [1, \infty)\f$
under the probability density \f$w(t) / \int_0^\infty w(u) \, \mathrm{d}u\f$.
The moments are computed by double-exponential quadrature centred at the peak
\f$t^* = \mathrm{asinh}(\nu / x) \in [0, \infty)\f$ at precision
\f$P = 256 + 8 \log_2(2 + \nu + x + 1/x)\f$ bits, and verified at \f$P + 96\f$
bits (all rows agree to \f$2^{-110}\f$).

### Reference file format

Reference values are written to CSV files in `tests/data/reference/` (such as
`tests/data/reference/Phi.csv`). Each file contains a header line specifying
up to three input arguments followed by `hi,lo`. Every number is printed in
exact hexadecimal floating-point format (`%a` in C and Python) to prevent any
decimal conversion error.

The two fields `hi` and `lo` represent the exact mathematical value as an
unevaluated double-double sum \f$hi + lo\f$ (Dekker 1971):
- \f$hi \in \mathbb{R}\f$ is the exact value rounded to IEEE 754 double precision (53 bits).
- \f$lo \in \mathbb{R}\f$ is the residual difference rounded to double precision,
  providing approximately 106 bits of effective significand.

The sum \f$hi + lo\f$ carries about 106 bits, so the rounding of the stored
reference is about \f$2^{-106}\f$ relative, far below a double's \f$2^{-53}\f$;
the working precision of the computation (200 bits or more) is what makes
those 106 bits correct.

### Sampling strategy

Reference points are not chosen uniformly at random. Instead, test grids are
deliberately concentrated in numerically sensitive regions:
- Asymptotic tails (deep underflow and overflow regimes),
- Neighbourhoods of zeros and turning points where catastrophic cancellation can occur,
- Region boundaries and transition points between different asymptotic or series expansions.

## Backend evaluation

Laplace is designed to run on diverse hardware execution spaces via Kokkos.
Candidate functions are evaluated using Kokkos parallel kernels
(`bench::eval_kernel` in `benchmarks/measure.hpp`):
- **CPU build:** compiled with gcc (Serial and OpenMP execution spaces).
- **GPU build:** compiled with NVIDIA nvcc for CUDA execution spaces.

### Backend differences in the last bit

CPU and GPU outputs can differ in the least significant bit for two reasons:
1. **Fused multiply-add (FMA) contraction:** nvcc routinely contracts
   expressions of the form \f$a \cdot b + c\f$ into a single fused multiply-add
   instruction (`fma`) with a single final rounding (Muller et al. 2018). While
   FMA increases accuracy in general, it alters intermediate rounding and can
   break identities that rely on uncontracted arithmetic (such as Dekker's split
   or Knuth's `two_sum`).
2. **Platform math libraries:** CUDA device code executes NVIDIA `libdevice`
   elementary transcendentals, whereas CPU host code executes the host standard
   library (`libm`) or Laplace's own host transcendentals (`elementary.hpp`).

Because of these differences, performance and accuracy tables throughout this
manual report results for both CPU (gcc) and GPU (CUDA) backends separately.

## Error metrics and definitions

The accuracy harness (`benchmarks/measure.hpp`) defines several complementary
metrics to characterize floating-point error.

### Definition of ulp

The unit in the last place of a real number \f$a \in \mathbb{R}\f$, denoted
\f$\mathrm{ulp}(a)\f$, is the distance between the two closest floating-point
numbers straddling \f$a\f$ (Goldberg 1991; Muller et al. 2018). In IEEE 754
double precision with 52 explicit significand bits, \f$\mathrm{ulp}(a)\f$ is
computed by `bench::ulp_of(a)`:
\f[
\mathrm{ulp}(a) = 2^{e - 52}
\f]
where the exponent \f$e \in \mathbb{Z}\f$ is:
\f[
e = \begin{cases}
-1022 & \text{if } |a| < 2.2250738585072014 \times 10^{-308} \text{ (subnormal)}, \\
\lfloor \log_2 |a| \rfloor & \text{otherwise (normal)}.
\end{cases}
\f]
For subnormal numbers, the spacing remains fixed at \f$2^{-1022 - 52} = 2^{-1074}\f$.

### Error calculation

Let \f$y \in \mathbb{R}\f$ be the computed double-precision value and
\f$hi, lo \in \mathbb{R}\f$ be the reference double-double components. For a
given scaling factor \f$s \in \mathbb{R}_{> 0}\f$, the signed error in units of
the last place of \f$s\f$ is computed by `bench::error_in(y, hi, lo, s)`:
\f[
\text{error}(y, hi, lo, s) = \frac{(y - hi) - lo}{\mathrm{ulp}(s)}
\f]
Special values are verified strictly:
- If \f$hi\f$ is NaN, \f$y\f$ must also be NaN (error 0).
- If \f$hi\f$ is \f$\pm\infty\f$, \f$y\f$ must match \f$hi\f$ exactly with identical sign (error 0).
- If \f$y\f$ produces an unexpected NaN or infinity when \f$hi\f$ is finite, or if the sign of infinity differs, the point is classified as an invalid result (`n_bad`).

Points flagged as `n_bad` are excluded from numeric summary statistics and
recorded separately. In automated tests, `n_bad` must be strictly zero.

### Error scales: ulp, ulp1, ulpL, and ulpC

Different applications require different notions of error:

#### 1. Relative error: ulp

When the scale is \f$s = |hi|\f$, the error is the classical relative error in
units in the last place:
\f[
\text{ulp} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(|hi|)}
\f]
An error of 0.5 ulp or less indicates correct rounding under round-to-nearest.
However, relative ulp error diverges to infinity whenever \f$hi \to 0\f$, even
when the computed result differs from zero by only a few units of \f$2^{-52}\f$.

#### 2. Order-one absolute error: ulp1

For functions where values can cross or approach zero, or where the output is
subsequently added to quantities of order one (such as a log-density or a
score), the appropriate scale is \f$s = \max(|hi|, 1.0)\f$:
\f[
\text{ulp1} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(\max(|hi|, 1.0))}
\f]
When \f$|hi| \ge 1\f$, ulp1 coincides with standard ulp. When \f$|hi| < 1\f$,
ulp1 measures absolute error in units of \f$\mathrm{ulp}(1.0) = 2^{-52} \approx 2.22 \times 10^{-16}\f$.

#### 3. Inherent logarithmic error: ulpL

When a quantity \f$f \in \mathbb{R}_{> 0}\f$ is evaluated in exponential form,
say \f$f = \exp(\log f)\f$ or via a small tail representation, an uncertainty of
1 ulp in the exponent \f$\log f\f$ induces a relative error of \f$|\log f|\f$
ulps in \f$f\f$. The metric ulpL scales the relative error by the conditioning
of exponentiation:
\f[
\text{ulpL} = \frac{\text{ulp}}{\max(1.0, |\log(hi)|)}
\f]
A bounded ulpL demonstrates that the implementation does not introduce errors
beyond what the inherent conditioning of exponentiation entails.

#### 4. Conditioned scale: ulpC

General numerical principles dictate that an algorithm's error should be judged
against what the mathematical conditioning of the function allows (Higham 2002).
On module pages where a function is inherently ill-conditioned near certain
arguments, error is scaled by a function-specific conditioning scale \f$s \in \mathbb{R}_{> 0}\f$:
\f[
\text{ulpC} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(s)}
\f]
where \f$s\f$ reflects the sensitivity of the exact value to perturbations in
its inputs.

**Example in `bessel_k`:**
For \ref laplace::math::own::log_bessel_k "log_bessel_k()", a 1-ulp variation in the argument \f$x\f$ shifts
\f$\log K_\nu(x)\f$ by approximately \f$|x g|\f$ ulps of 1, where
\f$g = \frac{\mathrm{d}}{\mathrm{d}x} \log K_\nu(x)\f$ (computed by \ref laplace::math::own::log_bessel_k_dx1 "log_bessel_k_dx1()").
Near the turning point \f$x \approx 0.66\nu\f$, where \f$\log K_\nu(x)\f$ crosses
zero, \f$|x g| \approx 1.2\nu\f$. The module therefore defines the conditioning scale:
\f[
s = \max(|\log K_\nu(x)|, |x g|, 1.0)
\f]
In units of ulpC, \ref laplace::math::own::log_bessel_k "log_bessel_k()" exhibits a measured maximum error of only
**4.8 ulpC** on all reference points on both gcc and CUDA backends.

### Summary metrics

Across a test dataset of \f$N \in \mathbb{Z}_{\ge 1}\f$ points where
\f$N - n_{\mathrm{bad}} > 0\f$ with \f$n_{\mathrm{bad}} \in \mathbb{Z}_{\ge 0}\f$,
Laplace reports:
- **Maximum error:** the peak error observed over all valid points (e.g. `max_ulp` or `max_ulp1`).
- **Mean error:** the arithmetic average of the error across all valid points.
- **Correctly rounded fraction (CR):** the fraction of valid evaluations whose error satisfies \f$\text{error} \le 0.5\f$ ulp (Muller et al. 2018). A CR share near 1.0 confirms that the function returns the correctly rounded IEEE 754 float in almost all cases.
- **Worst point:** the exact input argument tuple \f$\mathbf{x}\f$ where the maximum error was recorded.
- **Non-finite errors** (`n_bad`): the count of evaluations producing mismatched non-finite results.

## Automated accuracy tests

Every numerical component is backed by regression test suites in `tests/math/`
built with doctest (such as `tests/math/test_bessel_k.cpp`).

### Execution inside Kokkos kernels

Tests load the precomputed reference CSV files from `tests/data/reference/`, copy
the arguments into Kokkos views, and evaluate the library functions inside a
parallel kernel on the active execution space (`bench::eval_kernel`). This
verifies that functions compile, inline, and run correctly both on host CPU
threads and on device GPU threads.

### Test bounds and safety margins

In each test, the maximum observed error is compared against an explicit test bound:
\f[
\text{max\_error} \le \text{bound}
\f]
The test bound is chosen as the measured maximum error plus a margin chosen per
function. For example, in `tests/math/test_bessel_k.cpp`:
- For \ref laplace::math::own::log_bessel_k "log_bessel_k()", the measured maximum is 4.8 ulpC on both backends, and the test bound is set to **7.5 ulpC**.
- For the derivatives \ref laplace::math::own::log_bessel_k_dx2 "log_bessel_k_dx2()" and \ref laplace::math::own::log_bessel_k_dx3 "log_bessel_k_dx3()", the measured maximums on gcc are 44.4 ulp and 44.0 ulp (where Temme's first coefficient cancels near \f$x = 1\f$ for \f$\nu = 1/4\f$), and the test bounds are set to **66.0 ulp**.

The margin ensures that test suites do not fail spuriously due to minor,
benign 1-ulp variations introduced by differing compiler optimization passes,
microarchitectures, or updated math runtimes, while firmly catching any genuine
numerical regression.

### Boundary and special value checks

Tests verify that edge cases, boundaries, and exceptional values satisfy exact
mathematical requirements:
- Invalid inputs (such as \f$x \le 0\f$ for functions defined on \f$(0, \infty)\f$) or NaN inputs return quiet NaNs.
- Infinities propagate with mathematically correct signs (for instance, \f$\lim_{x \to \infty} \log K_\nu(x) = -\infty\f$ and \f$\lim_{x \to \infty} \frac{\mathrm{d}}{\mathrm{d}x} \log K_\nu(x) = -1.0\f$).
- Known mathematical symmetries are tested to exact bitwise equality (such as \f$\log K_{-\nu}(x) = \log K_\nu(x)\f$).
- Closed-form solutions at special parameter values are verified (such as half-integer orders \f$K_{1/2}(x) = \sqrt{\frac{\pi}{2x}} e^{-x}\f$).

## References

- F. Johansson and others, mpmath: a Python library for arbitrary-precision floating-point arithmetic, version 1.3.0, https://mpmath.org/
- D. Goldberg (1991), What every computer scientist should know about floating-point arithmetic, ACM Computing Surveys 23(1), 5-48
- J.-M. Muller and others (2018), Handbook of Floating-Point Arithmetic, 2nd edition, Birkhäuser
- N. J. Higham (2002), Accuracy and Stability of Numerical Algorithms, 2nd edition, SIAM
- T. J. Dekker (1971), A floating-point technique for extending the available precision, Numerische Mathematik 18, 224-242
