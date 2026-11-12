# How accuracy is measured {#accuracy}

Every numerical module in Laplace documents the measured accuracy of its
functions, accompanied by error bounds enforced in automated tests. On module
pages, readers encounter figures such as "4.8 ulpC", "measured max error 44.4 ulp",
or "test bound 66.0 ulp". This page explains how every
accuracy figure in this manual is obtained, how the error metrics are defined,
and what guarantees the test bounds provide. See \ref derivation "How formulas are derived" for how the analytic formulas inside Laplace are obtained.

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

Some functions need a different route to their references, because mpmath's
own implementation is not accurate enough over the whole range: the module
pages say so where it applies (for example, the references of
\ref bessel_k "the Bessel function of the second kind" come from an integral
representation evaluated at a precision that grows with the arguments).

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

### Error scales

Different applications require different notions of error:

\anchor accuracy_ulp
#### Relative error

When the scale is \f$s = |hi|\f$, the error is the classical relative error in
units in the last place:
\f[
\text{ulp} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(|hi|)}
\f]
An error of 0.5 ulp or less indicates correct rounding under round-to-nearest.
However, relative ulp error diverges to infinity whenever \f$hi \to 0\f$, even
when the computed result differs from zero by only a few units of \f$2^{-52}\f$.

\anchor accuracy_ulp1
#### Order-one absolute error

For functions whose values can cross or approach zero, or whose output is
of interest only to an absolute accuracy of order one, the scale is \f$s = \max(|hi|, 1.0)\f$:
\f[
\text{ulp1} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(\max(|hi|, 1.0))}
\f]
When \f$|hi| \ge 1\f$, ulp1 coincides with standard ulp. When \f$|hi| < 1\f$,
ulp1 measures absolute error in units of \f$\mathrm{ulp}(1.0) = 2^{-52} \approx 2.22 \times 10^{-16}\f$.

\anchor accuracy_ulpL
#### Inherent logarithmic error

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

\anchor accuracy_conditioned
#### Conditioned scale

General numerical principles dictate that an algorithm's error should be judged
against what the mathematical conditioning of the function allows (Higham 2002).
On module pages where a function is inherently ill-conditioned near certain
arguments, error is scaled by a function-specific conditioning scale \f$s \in \mathbb{R}_{> 0}\f$:
\f[
\text{ulpC} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(s)}
\f]
where \f$s\f$ reflects the sensitivity of the exact value to perturbations in
its inputs.

Each module page that uses ulpC gives its own scale \f$s\f$; for example,
the page \ref bessel_k "Modified Bessel function of the second kind" scales
the error of \f$\log K_\nu(x)\f$ by its sensitivity to the argument.

\anchor accuracy_inverse
#### Conditioned error of inverse functions

For a function that solves \f$F(x) = s\f$ for \f$x\f$, where \f$s\f$ is the
input (a probability or its logarithm), let \f$x^*\f$ be the exact root and
\f$x\f$ the computed one. One ulp of the input moves the exact root by
\f$\mathrm{d}x = \mathrm{ulp}(s) / |\mathrm{d}s/\mathrm{d}x|\f$, and the error
is measured against the larger of this shift and one ulp of the root:
\f[
\text{ulpC} = \frac{|x - x^*|}{\max(\mathrm{ulp}(x^*), \mathrm{d}x)}
\f]
Where the problem is well conditioned (\f$\mathrm{d}x\f$ below one ulp of
\f$x^*\f$), this is the relative error in ulps of the root.

\anchor accuracy_mixed
#### Mixed partial derivatives

A function of two variables with mixed partial derivatives of total order
\f$K\f$ returns the \f$K + 1\f$ components \f$f_{K-j,j}\f$,
\f$j = 0, \dots, K\f$, where \f$f_{i,j}\f$ is the derivative of order \f$i\f$ in
the first variable and order \f$j\f$ in the second. A mixed component can be
small against the pure components \f$f_{K,0}\f$ and \f$f_{0,K}\f$ of the
same order. Its error is measured in ulps of the larger of its own size and
the geometric mean of the pure components:
\f[
\text{ulpN} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(\max(|f_{i,j}|, N_{i,j}))},
\quad N_{i,j} = |f_{K,0}|^{i/K} |f_{0,K}|^{j/K}
\f]
For the pure components, and wherever \f$|f_{i,j}| \ge N_{i,j}\f$, ulpN is
the relative error in ulps. Its logarithmic variant divides by the
conditioning of exponentiation as ulpL does:
\f$\text{ulpN} / \max(1, |\log |f_{i,j}||)\f$, also written ulpL on the module
pages.

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

\anchor accuracy_bounds
### Test bounds and safety margins

In each test, the maximum error \f$E\f$ observed over the reference points must satisfy an explicit test bound \f$B\f$:
\f[
E \le B
\f]
The test bound \f$B\f$ is chosen as the measured maximum error plus a margin chosen per
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
