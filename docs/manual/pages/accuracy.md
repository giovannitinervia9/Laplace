# How accuracy is measured {#accuracy}

Every numerical module in Laplace documents measured accuracy alongside test error bounds. This page defines the error metrics and explains references, backend evaluation, and test bounds. See \ref derivation "How formulas are derived" for derivations, and [`docs/numerics.md`](https://github.com/giovannitinervia9/Laplace/blob/main/docs/numerics.md) for measurement records.

## Reference generation

Python scripts in `tools/reference/` generate reference values with `mpmath` (2023) at 200 bits, raising precision to 256 or 300 bits where cancellation or large arguments need headroom. Where `mpmath` lacks accuracy, scripts use alternative routes such as numerical integration at growing precision, noted on module pages.

Datasets are CSV files in `tests/data/reference/` with up to three inputs followed by `hi` and `lo` in hexadecimal (`%a`). Together, \f$hi, lo \in \mathbb{R}\f$ store the double-double sum \f$hi + lo\f$ (Dekker 1971): \f$hi\f$ is the 53-bit rounding and \f$lo\f$ is the rounded residual, giving about 106 bits of significand (\f$2^{-106}\f$ storage rounding). Grids concentrate points in asymptotic tails, near zeros and turning points, and at expansion boundaries.

## Backend evaluation

Functions run in Kokkos parallel kernels on CPU via gcc with Serial and OpenMP, and on GPU via NVIDIA nvcc for CUDA. Backends can differ in the last bit for two reasons: nvcc contracts \f$a \cdot b + c\f$ into `fma` (Muller et al. 2018), altering intermediate rounding and breaking identities like Dekker's split or Knuth's `two_sum` (Knuth, 1997); and CUDA calls `libdevice` whereas CPU calls `libm` or `elementary.hpp`. Accuracy and speed tables report CPU and GPU separately.

## Error metrics and definitions

The harness (`benchmarks/measure.hpp`) measures error in ulps against the double-double reference.

### Definition of ulp

For \f$a \in \mathbb{R}\f$, \f$\mathrm{ulp}(a)\f$ is the spacing between the two nearest binary64 numbers straddling \f$a\f$ (Goldberg 1991; Muller et al. 2018):
\f[
\mathrm{ulp}(a) = 2^{e - 52}, \qquad
e = \begin{cases}
-1022 & \text{if } |a| < 2.2250738585072014 \times 10^{-308} \text{ (subnormal)}, \\
\lfloor \log_2 |a| \rfloor & \text{otherwise (normal)}.
\end{cases}
\f]
In the subnormal range, the spacing stays fixed at \f$2^{-1022 - 52} = 2^{-1074}\f$.

### Error calculation and special values

Given computed \f$y \in \mathbb{R}\f$, reference \f$hi + lo\f$, and scale \f$s \in \mathbb{R}_{> 0}\f$, the signed error in units of \f$\mathrm{ulp}(s)\f$ is:
\f[
\mathrm{error}(y, hi, lo, s) = \frac{(y - hi) - lo}{\mathrm{ulp}(s)}.
\f]
Non-finite values must match: NaN for NaN, and \f$\pm\infty\f$ with identical sign. Unexpected non-finite values or wrong signs are invalid results, excluded from summaries and required to be zero in tests.

### Error scales

Module pages choose the scale \f$s\f$ suited to each function.

\anchor accuracy_ulp
#### Relative error

Setting \f$s = |hi|\f$ gives the relative error in ulps:
\f[
\mathrm{ulp} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(|hi|)}.
\f]
Errors of at most 0.5 ulp indicate correct rounding under round-to-nearest. Relative ulp diverges as \f$hi \to 0\f$.

\anchor accuracy_ulp1
#### Order-one absolute error

For functions crossing zero or entering order-one sums, the scale is \f$s = \max(|hi|, 1.0)\f$:
\f[
\mathrm{ulp1} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(\max(|hi|, 1.0))}.
\f]
This matches relative ulp for \f$|hi| \ge 1\f$ and measures absolute error in units of \f$2^{-52} \approx 2.22 \times 10^{-16}\f$ for \f$|hi| < 1\f$.

\anchor accuracy_ulpL
#### Inherent logarithmic error

When \f$f \in \mathbb{R}_{> 0}\f$ is formed via \f$\exp(\log f)\f$ or a tail log form, 1 ulp in \f$\log f\f$ induces \f$|\log f|\f$ ulps in \f$f\f$:
\f[
\mathrm{ulpL} = \frac{\mathrm{ulp}}{\max(1.0, |\log(hi)|)}.
\f]
Bounded ulpL shows no error beyond exponentiation conditioning.

\anchor accuracy_conditioned
#### Conditioned scale

Where a function is ill-conditioned, error is scaled by its sensitivity (Higham 2002) using a function-specific \f$s \in \mathbb{R}_{> 0}\f$:
\f[
\mathrm{ulpC} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(s)}.
\f]
Module pages using ulpC define \f$s\f$, such as argument sensitivity for \ref bessel_k "modified Bessel functions of the second kind".

\anchor accuracy_inverse
#### Conditioned error of inverse functions

When solving \f$F(x) = s\f$ for \f$x\f$ with exact root \f$x^*\f$, one ulp in \f$s\f$ shifts the root by \f$\mathrm{d}x = \mathrm{ulp}(s) / |\mathrm{d}s/\mathrm{d}x|\f$:
\f[
\mathrm{ulpC} = \frac{|x - x^*|}{\max(\mathrm{ulp}(x^*), \mathrm{d}x)}.
\f]
Where \f$\mathrm{d}x \le \mathrm{ulp}(x^*)\f$, this reduces to the relative ulp error of \f$x^*\f$.

\anchor accuracy_mixed
#### Mixed partial derivatives and matrix normwise error

For a bivariate function differentiated to total order \f$K\f$, mixed component \f$f_{i,j}\f$ of order \f$i\f$ in the first variable and \f$j = K - i\f$ in the second can be small next to pure components \f$f_{K,0}\f$ and \f$f_{0,K}\f$. Error uses the larger of \f$|f_{i,j}|\f$ and their geometric mean:
\f[
\mathrm{ulpN} = \frac{|(y - hi) - lo|}{\mathrm{ulp}(\max(|f_{i,j}|, N_{i,j}))},
\qquad N_{i,j} = |f_{K,0}|^{i/K} |f_{0,K}|^{j/K}.
\f]
Wherever \f$|f_{i,j}| \ge N_{i,j}\f$, including pure components, ulpN equals relative ulp. Module pages also write ulpL for its logarithmic variant \f$\mathrm{ulpN} / \max(1, |\log |f_{i,j}||)\f$.

\anchor accuracy_matrix
For matrix-valued quantities with computed entries \f$y_{ij}\f$ and reference entries \f$r_{ij}\f$, error is measured normwise using \f$\varepsilon = 2^{-52}\f$:
\f[
\mathrm{ulpN} = \frac{\max_{i,j} |y_{ij} - r_{ij}|}{\varepsilon \max\left(\max_{i,j} |r_{ij}|,\, 10^{-20}\right)}.
\f]
Here the \f$10^{-20}\f$ floor avoids dividing by difference noise when derivatives vanish.

### Summary metrics

Across \f$N \in \mathbb{Z}_{\ge 1}\f$ points with valid evaluations, summaries report peak error, arithmetic mean, worst-error input \f$\mathbf{x}\f$, and mismatched non-finite count. They also report the correctly rounded fraction (CR), the share of valid points within 0.5 ulp (Muller et al. 2018).

## Automated accuracy tests

Regression test suites are built with doctest. They load CSV files from `tests/data/reference/` into Kokkos views and evaluate functions inside parallel kernels on CPU and GPU execution spaces.

\anchor accuracy_bounds
### Test bounds and safety margins

Each test asserts that the observed maximum error \f$E\f$ over the reference set satisfies an explicit bound \f$B\f$:
\f[
E \le B.
\f]
The bound \f$B\f$ adds a safety margin above the measured maximum on both backends, preventing spurious failures from 1-ulp compiler, hardware, or runtime variations while catching regressions.

### Boundary and special value checks

Tests also verify invalid or NaN inputs returning quiet NaNs, signed infinities, mathematical symmetries, and closed-form special cases.

## References

- T. J. Dekker (1971), A floating-point technique for extending the available precision, Numerische Mathematik, 18(3), 224–242. https://doi.org/10.1007/bf01397083
- D. Goldberg (1991), What every computer scientist should know about floating-point arithmetic, ACM Computing Surveys, 23(1), 5–48. https://doi.org/10.1145/103162.103163
- N. J. Higham (2002), Accuracy and Stability of Numerical Algorithms. https://doi.org/10.1137/1.9780898718027
- D. E. Knuth (1997), The Art of Computer Programming, Volume 2: Seminumerical Algorithms, 3rd edition, Addison-Wesley.
- The mpmath development team (2023), mpmath: a Python library for arbitrary-precision floating-point arithmetic, version 1.3.0, https://mpmath.org
- J. Muller and others (2018), Handbook of Floating-Point Arithmetic. https://doi.org/10.1007/978-3-319-76526-6
