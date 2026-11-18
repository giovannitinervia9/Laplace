# Wave 2 reference data (lasso, elastic net, LAD)

Written by `tools/reference/optimizers_wave2.R` (R, fixed seeds); see the
section "Frozen R references for wave 2 of the optimizers" of `docs/numerics.md`.
About 18 MB in all.

## Data

Five datasets, listed in `datasets.csv` (n, p, design, correlation, seed, file,
true coefficients). Each holds the n x p design X (no intercept column) and two
responses on the same X: `y_gaussian = 0.5 + X beta + N(0, 1)` and
`y_lad = 0.5 + X beta + Laplace(0, 1)`.

| id | n | p | design of X |
|---|---|---|---|
| d1_n100_p10 | 100 | 10 | independent N(0, 1) columns |
| d2_n100_p200 | 100 | 200 | AR(1), corr 0.9^\|j-k\| (p > n) |
| d3_n1000_p50 | 1000 | 50 | AR(1), corr 0.5^\|j-k\| |
| d4_n10000_p100 | 10000 | 100 | equicorrelated, corr 0.5 |
| d5_n100000_p10 | 100000 | 10 | AR(1), corr 0.8^\|j-k\| |

- `<id>.csv` (d1, d2): a header `x1,...,xp,y_gaussian,y_lad`, then one row per
  observation, every value a C99 hexadecimal double (`%a`), exact.
- `<id>.f64` (d3, d4, d5): the same n x (p + 2) matrix as raw IEEE 754
  binary64, little-endian, COLUMN-MAJOR (x1 for all rows, then x2, ...,
  then y_gaussian, then y_lad), no header; the size is 8 n (p + 2) bytes.

## Fits

Coefficients are C99 hex doubles; `b0` is the unpenalized intercept, `beta<j>`
the coefficient of `x<j>` (1-based). All CSV files start with `#` comment lines.

- `glmnet_fits.csv`: one row per (dataset, alpha in {1, 0.5}, lambda_fraction
  in {0.5, 0.1, 0.01} of glmnet's lambda_max), y = y_gaussian. Columns:
  `lambda_glmnet`, `lambda_laplace` (= n lambda_glmnet), the rates `a`, `c`,
  the exact support (1-based indices of beta), the objective in Laplace's units
  `RSS/2 + a |beta|_1 + c |beta|^2/2` and in glmnet's (divided by n), the
  penalty's normalizing constant per coefficient, the KKT violation and the
  slack off the support.
- `glmnet.csv`: one row per coefficient; `polished` is the reference (the KKT
  system on glmnet's support solved directly), `glmnet` glmnet's own output.
- `rq_fits.csv`: one row per dataset with p < n, y = y_lad: the LAD objective
  `sum |r|`, `sigma_hat = sum |r| / n`, the laplace1 negative log-likelihood at
  (beta, sigma_hat), the dual certificate `dual_max` (optimal if <= 1, unique if
  < 1), and the basis (the p + 1 rows with zero residual).
- `rq.csv`: one row per coefficient; `vertex` is the reference (the basis
  system re-solved), `rq` quantreg's own output.
