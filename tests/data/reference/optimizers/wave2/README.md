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

## Penalties through a linear map

Written by `tools/reference/optimizers_wave2_linear_map.R` (run after
`optimizers_wave2.R`, with R 4.6.0, glmnet 5.0 and genlasso 1.6.1); see the section "Frozen R references for penalties
through a linear map" of `docs/numerics.md`.

### Adaptive lasso (glmnet with penalty.factor, datasets d1 to d5, y_gaussian)

- `adaptive_lasso_weights.csv`: one row per (dataset, coefficient), the weight
  `w_j = 1 / |b_j|` (hex), b from OLS on centred data when p < n, from the
  ridge `(Xc'Xc + 10 I) b = Xc'yc` on d2 (p > n).
- `adaptive_lasso_fits.csv`: one row per (dataset, lambda_fraction in {0.5,
  0.1, 0.01} of glmnet's lambda_max with these factors). glmnet rescales
  penalty.factor to `pf' = w p / sum(w)`, so in Laplace's units the objective is
  `RSS/2 + r sum_j w_j |beta_j|` with `rate_laplace = r = n lambda_glmnet p /
  sum_weights`: laplace2 at rate r on `u = W beta`, whose value adds
  `penalty_constant = -p log(r/2) - sum_j log w_j`. Columns as in
  `glmnet_fits.csv`, plus `kkt_violation_glmnet_unscaled` (glmnet's point read
  with kink weights `n lambda_glmnet w_j`, which it violates by orders of
  magnitude: the check that the rescaling is real).
- `adaptive_lasso.csv`: one row per coefficient (`b0`, `beta<j>`); `polished`
  is the reference (KKT system on glmnet's support), `glmnet` glmnet's own.

### Fused lasso, trend filtering, elastic net on differences (genlasso)

- `difference_datasets.csv`: two datasets with no intercept, X AR(1) with
  correlation 0.5: `f1_n200_p50` (n = 200, p = 50, `f1_n200_p50.csv`, hex) and
  `f2_n1000_p200` (n = 1000, p = 200, `f2_n1000_p200.f64`, raw little-endian
  column-major). Columns `x1..xp, y_fused, y_trend`: `y_fused = X beta_fused +
  N(0, 1)` with piecewise-constant coefficients, `y_trend = X beta_trend +
  N(0, 1)` with piecewise-linear ones (both listed in the file).
- `difference_fits.csv`: one row per (dataset, penalty, lambda_fraction in
  {0.5, 0.1, 0.01} of the first knot of genlasso's path). The objective, with no
  intercept, is `RSS/2 + a |D beta|_1 + c |D beta|^2/2`, D the differences of
  order `order`, `a = lambda alpha`, `c = lambda (1 - alpha)`:
  - `fused`: y_fused, order 1, alpha 1, `fusedlasso1d(y, X)`;
  - `trend`: y_trend, order 2, alpha 1, `trendfilter(y, X, ord = 1)`;
  - `fused_enet`: y_fused, order 1, alpha 0.5, `genlasso` on the augmented data
    `[y; 0]`, `[X; sqrt(c) D]` at lambda = a.
  Laplace's penalty (laplace2 at rate lambda, or enet at (lambda, alpha), on
  each row of D through `difference_map<order>`) has the value
  `a |D beta|_1 + c |D beta|^2/2 + (p - order) penalty_constant_per_row -
  log_det_ddt / 2`. `nonzero_rows` lists the 1-based rows i with
  `(D beta)_i != 0`; `dual_max` (< 1: strict) and `dual_residual` certify the
  polished point (the KKT system solved on genlasso's fusion pattern).
- `difference.csv`: one row per coefficient (`beta<j>`); `polished` is the
  reference, `genlasso` genlasso's own.
