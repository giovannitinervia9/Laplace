# fit_distrib reference data

Written by `tools/reference/optimizers_fit_distrib.R` (R, fixed seeds); see the
section "Frozen R references for the optimizers" of `docs/numerics.md`.

- `<distrib>_n1000.csv`: 1000 observations, a header line `y`, then one value
  per line as a C99 hexadecimal double (`%a`, e.g. `0x1.0c8p+1`), exact.
- `<distrib>_n100000.f64`: 100000 observations as raw IEEE 754 binary64,
  little-endian, 8 bytes each, no header and no count (the file is exactly
  800000 bytes), in the order in which R drew them.
- `fit_distrib.csv`: one row per (distrib, n, parameter) with the true value,
  statmodels7's `fit_distrib()` estimate, standard error, log-likelihood,
  iterations, counts and convergence, and the closed-form MLE where one exists.
  Comment lines start with `#`.

`<distrib>` is the Laplace name (`laplace1`, `invgauss`, ...); the column
`distributions7` of `fit_distrib.csv` gives the matching statmodels7
constructor.
