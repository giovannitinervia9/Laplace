# Frozen references for penalties through a linear map (note/penalties.md
# section 5d, note/optimizers.md section 12 decision 8): the adaptive lasso by
# glmnet with penalty.factor on the wave-2 datasets, and the fused lasso,
# trend filtering and the elastic net on differences by genlasso on new
# datasets with ordered coefficients.
#
# Run from the repository root, after optimizers_wave2.R has written the
# datasets: Rscript --vanilla tools/reference/optimizers_wave2_linear_map.R
#
# Output in tests/data/reference/optimizers/wave2/ (layout in its README.md):
#   adaptive_lasso_weights.csv  one row per (dataset, coefficient): w_j
#   adaptive_lasso_fits.csv     one row per (dataset, lambda): rates, support, objective, KKT
#   adaptive_lasso.csv          one row per (dataset, lambda, coefficient)
#   <id>.csv or <id>.f64        the data of the difference problems (columns x1..xp, y)
#   difference_datasets.csv     one row per difference dataset
#   difference_fits.csv         one row per (dataset, penalty, lambda): objective, support, certificate
#   difference.csv              one row per (dataset, penalty, lambda, coefficient)
#
# Adaptive lasso. glmnet (alpha = 1, standardize = FALSE, intercept unpenalized)
# rescales penalty.factor pf to pf' = pf p / sum(pf) and minimizes
#   RSS / (2 n) + lambda_g sum_j pf'_j |beta_j|.
# Times n, in Laplace's units:
#   F_L = RSS / 2 + r sum_j w_j |beta_j|,   w_j = pf_j,   r = n lambda_g p / sum_j w_j,
# which is laplace2's penalty with rate r on u = W beta (linear_map_penalty
# over diagonal_map(w)), less its constants: -log(r / 2) per coefficient and
# c(W) = -sum_j log w_j. The conversion is checked through the KKT conditions
# with the kink weights a_j = r w_j: a glmnet point that satisfies them for
# these a_j, and only for these, confirms the rescaling.
#
# Differences. genlasso minimizes (1/2) |y - X beta|^2 + lambda |D beta|_1, with
# no intercept, which is F_L = RSS / 2 + lambda |D beta|_1 as it stands: laplace2
# with rate lambda on u = D beta (linear_map_penalty over difference_map<K>),
# less -log(lambda / 2) per row of D and c(D) = -(1/2) log det(D D^T). K = 1 is
# the fused lasso (fusedlasso1d), K = 2 trend filtering of order 1 (trendfilter,
# ord = 1). The elastic net on first differences,
#   RSS / 2 + a |D beta|_1 + c |D beta|^2 / 2,   a = lambda alpha, c = lambda (1 - alpha),
# is genlasso on the augmented data [X; sqrt(c) D], [y; 0] at lambda = a.
#
# Polish and certificate (as in optimizers_wave2.R). With genlasso's pattern,
# Z the rows with (D beta)_i = 0 and s the signs on the others A, beta is
# solved exactly on the subspace {D_Z beta = 0} = range(B):
#   B' (X'X + c D'D) B theta = B' X'y - a B' D_A' s_A,   beta = B theta,
# and certified by the dual u: X'(X beta - y) + c D'D beta + a D' u = 0 with
# u_A = s_A and u_Z solved by least squares; the point is optimal iff the
# residual of that system is at rounding and max |u_Z| <= 1 (strict: < 1).

suppressMessages({ library(glmnet) })
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
define_cases_only <- TRUE
source("tools/reference/optimizers_wave2.R")  # datasets, write_data, h, g17, out_dir

load_wave2 <- function(ds) {
  f <- file.path(out_dir, paste0(ds$id, if (ds$n * (ds$p + 2) <= hex_limit) ".csv" else ".f64"))
  m <- if (grepl("\\.csv$", f)) {
    b <- as.matrix(read.csv(f, colClasses = "character")); matrix(as.numeric(b), nrow(b))
  } else matrix(readBin(f, "double", n = ds$n * (ds$p + 2), size = 8, endian = "little"), ds$n)
  list(X = m[, seq_len(ds$p)], y = m[, ds$p + 1])
}

write_with_header <- function(d, f, header) {
  con <- file(file.path(out_dir, f), "w")
  writeLines(paste("#", header), con)
  write.csv(d, con, row.names = FALSE)
  close(con)
}

# ---------------------------------------------------------------------------
# Adaptive lasso

# KKT of RSS/2 + sum a_j |beta_j| with a free intercept
weighted_kkt <- function(X, y, b0, beta, a) {
  r <- y - b0 - drop(X %*% beta)
  g <- -drop(crossprod(X, r))
  on <- beta != 0
  v <- numeric(length(beta))
  v[on] <- abs(g[on] + a[on] * sign(beta[on]))
  v[!on] <- pmax(0, abs(g[!on]) - a[!on])
  list(max_violation = max(abs(sum(r)), v),
       slack = if (any(!on)) min(a[!on] - abs(g[!on])) else NA_real_,
       objective = sum(r^2) / 2 + sum(a * abs(beta)))
}

weighted_polish <- function(X, y, beta, a) {
  S <- which(beta != 0); s <- sign(beta[S])
  xm <- colMeans(X); ym <- mean(y)
  b <- numeric(ncol(X))
  if (length(S)) {
    Xs <- sweep(X[, S, drop = FALSE], 2, xm[S])
    A <- crossprod(Xs)
    rhs <- drop(crossprod(Xs, y - ym)) - a[S] * s
    b[S] <- solve(A, rhs)
    b[S] <- b[S] + solve(A, rhs - drop(A %*% b[S]))
  }
  list(b0 = ym - sum(xm * b), beta = b, signs_ok = all(sign(b[S]) == s))
}

ridge_for_weights <- 10  # c of the ridge fit giving the weights when p >= n (d2)
w_rows <- list(); al_rows <- list(); alc_rows <- list()
timing <- list()
for (k in seq_along(datasets)) {
  ds <- datasets[[k]]
  dd <- load_wave2(ds); X <- dd$X; y <- dd$y; n <- ds$n; p <- ds$p
  Xc <- sweep(X, 2, colMeans(X)); yc <- y - mean(y)
  # initial fit: OLS when p < n, ridge with c = ridge_for_weights otherwise; w_j = 1 / |b_j|
  A <- crossprod(Xc) + if (p >= n) diag(ridge_for_weights, p) else 0
  binit <- solve(A, crossprod(Xc, yc))
  w <- 1 / abs(drop(binit))
  stopifnot(all(is.finite(w)), all(w > 0))
  w_rows[[k]] <- data.frame(dataset = ds$id, coef = paste0("beta", seq_len(p)), weight = h(w),
                            initial_fit = if (p >= n) "ridge" else "ols")
  pf_scaled <- w * p / sum(w)                       # glmnet's internal factors
  lambda_max <- max(abs(drop(crossprod(Xc, yc))) / (n * pf_scaled))
  lambdas <- lambda_max * lambda_fractions
  t0 <- Sys.time()
  f <- glmnet(X, y, family = "gaussian", alpha = 1, lambda = lambdas, penalty.factor = w,
              standardize = FALSE, intercept = TRUE, control = list(thresh = 1e-14, maxit = 1e8))
  timing[[length(timing) + 1]] <- sprintf("%-16s adaptive glmnet %.2f s", ds$id, as.numeric(Sys.time() - t0, units = "secs"))
  cat(timing[[length(timing)]], "\n")
  for (l in seq_along(lambdas)) {
    r_rate <- n * lambdas[l] * p / sum(w)
    a <- r_rate * w
    b0 <- unname(f$a0[l]); beta <- unname(as.numeric(f$beta[, l]))
    kg <- weighted_kkt(X, y, b0, beta, a)
    # the unscaled reading a_j = n lambda_g w_j, for contrast: glmnet's point must violate it
    k_unscaled <- weighted_kkt(X, y, b0, beta, n * lambdas[l] * w)
    pl <- weighted_polish(X, y, beta, a)
    kp <- weighted_kkt(X, y, pl$b0, pl$beta, a)
    accepted <- pl$signs_ok && (is.na(kp$slack) || kp$slack > 0)
    cat(sprintf("   l %d  support %3d  kkt glmnet %.3g (unscaled reading %.3g)  polished %.3g  slack %.3g  diff %.3g\n",
                l, sum(pl$beta != 0), kg$max_violation, k_unscaled$max_violation, kp$max_violation, kp$slack,
                max(abs(c(b0 - pl$b0, beta - pl$beta)))))
    stopifnot(accepted)
    al_rows[[length(al_rows) + 1]] <- data.frame(
      dataset = ds$id, lambda_fraction = lambda_fractions[l], lambda_glmnet = h(lambdas[l]),
      rate_laplace = h(r_rate), sum_weights = h(sum(w)),
      support_size = sum(pl$beta != 0), support = paste(which(pl$beta != 0), collapse = " "),
      support_glmnet_same = identical(which(beta != 0), which(pl$beta != 0)),
      objective_laplace = h(kp$objective),
      penalty_constant = h(-p * log(r_rate / 2) - sum(log(w))),
      kkt_violation_polished = g17(kp$max_violation), kkt_violation_glmnet = g17(kg$max_violation),
      kkt_violation_glmnet_unscaled = g17(k_unscaled$max_violation),
      kkt_slack_off_support = g17(kp$slack),
      max_abs_diff_glmnet_polished = g17(max(abs(c(b0 - pl$b0, beta - pl$beta)))),
      polished_accepted = accepted)
    alc_rows[[length(alc_rows) + 1]] <- data.frame(
      dataset = ds$id, lambda_fraction = lambda_fractions[l],
      coef = c("b0", paste0("beta", seq_len(p))), polished = h(c(pl$b0, pl$beta)), glmnet = h(c(b0, beta)))
  }
}
ver <- sprintf("glmnet %s, R %s", packageVersion("glmnet"), getRversion())
write_with_header(do.call(rbind, w_rows), "adaptive_lasso_weights.csv",
  c(sprintf("written by tools/reference/optimizers_wave2_linear_map.R (%s)", ver),
    "w_j = 1 / |b_j|, b from OLS on centred data (p < n) or ridge (X'X + 10 I) b = X'y on centred data (p >= n); hex doubles"))
write_with_header(do.call(rbind, al_rows), "adaptive_lasso_fits.csv",
  c(sprintf("glmnet(X, y_gaussian, alpha = 1, lambda, penalty.factor = w, standardize = FALSE, thresh = 1e-14) (%s)", ver),
    "glmnet rescales pf to pf' = w p / sum(w); objective_laplace = RSS/2 + rate_laplace sum_j w_j |beta_j| at the polished point,",
    "rate_laplace = n lambda_glmnet p / sum_weights (laplace2's rate on u = W beta); penalty_constant = -p log(rate/2) - sum log w_j",
    "(Laplace's penalty value is rate |W beta|_1 + penalty_constant); kkt values in objective units, kink weights a_j = rate w_j;",
    "kkt_violation_glmnet_unscaled = glmnet's point read with a_j = n lambda_glmnet w_j (wrong reading, must be large)"))
write_with_header(do.call(rbind, alc_rows), "adaptive_lasso.csv",
  "coefficients as C99 hex doubles: polished = exact KKT solution on glmnet's support (the reference), glmnet = glmnet's own")

# ---------------------------------------------------------------------------
# Differences: fused lasso (K = 1), trend filtering (K = 2), enet on first differences

suppressMessages(library(genlasso))
diff_matrix <- function(p, K) { D <- diag(p); for (s in seq_len(K)) D <- diff(D); D }
log_det_ddt <- function(p, K) sum(sapply(0:(K - 1), function(j) lchoose(p + j, 2 * j + 1) - lchoose(2 * j, j)))

difference_datasets <- list(
  list(id = "f1_n200_p50",   n = 200,  p = 50,  rho = 0.5),
  list(id = "f2_n1000_p200", n = 1000, p = 200, rho = 0.5)
)
# piecewise constant (fused) and piecewise linear (trend) coefficients over j = 1..p
beta_fused <- function(p) { t <- seq_len(p) / p; ifelse(t <= 0.2, 0, ifelse(t <= 0.5, 2, ifelse(t <= 0.8, -1, 0.5))) }
beta_trend <- function(p) { t <- seq_len(p) / p; 2 * pmin(4 * t, 2 - 2 * t, 0.5 + 0 * t) - 0.5 }
diff_fractions <- c(0.5, 0.1, 0.01)  # of the first knot of genlasso's path

# exact solution on {D_Z beta = 0} and the dual certificate; a, c the l1 and ridge
# weights on D beta; Z found from genlasso's beta
polish_differences <- function(X, y, D, K, beta, a, c) {
  p <- ncol(X); Db <- drop(D %*% beta)
  Z <- which(abs(Db) <= 1e-8 * max(1, max(abs(beta))))
  A <- setdiff(seq_len(nrow(D)), Z); s <- sign(Db[A])
  if (K == 1) {  # indicators of the fused groups: exact zeros in D beta
    grp <- cumsum(c(1, as.integer(!(seq_len(p - 1) %in% Z))))
    B <- outer(seq_len(p), seq_len(max(grp)), function(i, g) as.numeric(grp[i] == g))
  } else if (length(Z)) {  # orthonormal basis of the null space of D_Z
    q <- qr(t(D[Z, , drop = FALSE]))
    B <- qr.Q(q, complete = TRUE)[, (q$rank + 1):p, drop = FALSE]
  } else B <- diag(p)
  H <- crossprod(X) + c * crossprod(D)
  rhs <- drop(crossprod(B, crossprod(X, y) - a * crossprod(D[A, , drop = FALSE], s)))
  M <- crossprod(B, H %*% B)
  th <- solve(M, rhs); th <- th + solve(M, rhs - drop(M %*% th))
  b <- drop(B %*% th)
  Dbp <- drop(D %*% b)
  g <- drop(crossprod(X, X %*% b - y)) + c * drop(crossprod(D, Dbp))
  target <- -g / a - drop(crossprod(D[A, , drop = FALSE], s))
  if (length(Z)) {
    DZt <- t(D[Z, , drop = FALSE])
    uZ <- qr.solve(DZt, target)
    res <- max(abs(DZt %*% uZ - target))
  } else { uZ <- numeric(0); res <- max(abs(target)) }
  r <- y - drop(X %*% b); rg <- y - drop(X %*% beta)
  list(beta = b, Z = Z, A = A, signs_ok = all(sign(Dbp[A]) == s), dual_max = if (length(uZ)) max(abs(uZ)) else 0,
       dual_residual = res * a, objective = sum(r^2) / 2 + a * sum(abs(Dbp)) + c * sum(Dbp^2) / 2,
       objective_genlasso_point = sum(rg^2) / 2 + a * sum(abs(Db)) + c * sum(Db^2) / 2)
}

dd_rows <- list(); df_rows <- list(); dc_rows <- list()
for (k in seq_along(difference_datasets)) {
  ds <- difference_datasets[[k]]
  seed <- 20261118 + 1000 + 100 * k
  set.seed(seed)
  X <- make_x(ds$n, ds$p, "ar1", ds$rho)
  bf <- beta_fused(ds$p); bt <- beta_trend(ds$p)
  y_fused <- drop(X %*% bf) + rnorm(ds$n)
  y_trend <- drop(X %*% bt) + rnorm(ds$n)
  m <- cbind(X, y_fused, y_trend)
  colnames(m) <- c(paste0("x", seq_len(ds$p)), "y_fused", "y_trend")
  file <- write_data(ds, m)
  dd_rows[[k]] <- data.frame(dataset = ds$id, n = as.integer(ds$n), p = ds$p, design = "ar1", rho = ds$rho, seed = seed,
                             file = file, beta_fused = paste(g17(bf), collapse = " "), beta_trend = paste(g17(bt), collapse = " "))
  cases <- list(list(penalty = "fused", K = 1, y = y_fused, alpha = 1),
                list(penalty = "trend", K = 2, y = y_trend, alpha = 1),
                list(penalty = "fused_enet", K = 1, y = y_fused, alpha = 0.5))
  for (cs in cases) {
    D <- diff_matrix(ds$p, cs$K)
    t0 <- Sys.time()
    path <- if (cs$K == 1) fusedlasso1d(cs$y, X = X) else trendfilter(cs$y, X = X, ord = cs$K - 1)
    lam_top <- path$lambda[1]
    timing[[length(timing) + 1]] <- sprintf("%-16s %-10s genlasso path %.2f s", ds$id, cs$penalty,
                                            as.numeric(Sys.time() - t0, units = "secs"))
    cat(timing[[length(timing)]], "\n")
    for (fr in diff_fractions) {
      lam <- fr * lam_top
      a <- lam * cs$alpha; c <- lam * (1 - cs$alpha)
      if (c > 0) {  # enet: genlasso on [X; sqrt(c) D], [y; 0] at lambda = a
        t0 <- Sys.time()
        pe <- genlasso(c(cs$y, numeric(nrow(D))), X = rbind(X, sqrt(c) * D), D = D)
        beta <- drop(coef(pe, lambda = a)$beta)
        timing[[length(timing) + 1]] <- sprintf("%-16s %-10s genlasso augmented path %.2f s", ds$id, cs$penalty,
                                                as.numeric(Sys.time() - t0, units = "secs"))
      } else beta <- drop(coef(path, lambda = lam)$beta)
      pl <- polish_differences(X, cs$y, D, cs$K, beta, a, c)
      ok <- pl$signs_ok && pl$dual_max < 1 && pl$dual_residual <= 1e-8 * max(1, abs(pl$objective))
      cat(sprintf("   %-10s frac %.2f  nonzero diffs %3d of %3d  dual max %.6f  dual residual %.3g  diff %.3g\n",
                  cs$penalty, fr, length(pl$A), nrow(D), pl$dual_max, pl$dual_residual, max(abs(beta - pl$beta))))
      stopifnot(ok, pl$objective <= pl$objective_genlasso_point * (1 + 1e-12))
      const_row <- if (c == 0) -log(lam / 2) else {
        x <- a / sqrt(c); log(2) + pnorm(-x, log.p = TRUE) - dnorm(x, log = TRUE) - log(c) / 2
      }
      df_rows[[length(df_rows) + 1]] <- data.frame(
        dataset = ds$id, penalty = cs$penalty, order = cs$K, alpha = cs$alpha, lambda_fraction = fr,
        lambda = h(lam), a = h(a), c = h(c), lambda_path_top = h(lam_top),
        nonzero_differences = length(pl$A), nonzero_rows = paste(pl$A, collapse = " "),
        objective_laplace = h(pl$objective), objective_genlasso_point = h(pl$objective_genlasso_point),
        penalty_constant_per_row = h(const_row), log_det_ddt = h(log_det_ddt(ds$p, cs$K)),
        dual_max = g17(pl$dual_max), dual_residual = g17(pl$dual_residual),
        max_abs_diff_genlasso_polished = g17(max(abs(beta - pl$beta))), polished_accepted = ok)
      dc_rows[[length(dc_rows) + 1]] <- data.frame(
        dataset = ds$id, penalty = cs$penalty, lambda_fraction = fr, coef = paste0("beta", seq_len(ds$p)),
        polished = h(pl$beta), genlasso = h(beta))
    }
  }
}
ver <- sprintf("genlasso %s, R %s", packageVersion("genlasso"), getRversion())
write_with_header(do.call(rbind, dd_rows), "difference_datasets.csv",
  c(sprintf("written by tools/reference/optimizers_wave2_linear_map.R (%s)", ver),
    "y_fused = X beta_fused + N(0, 1), y_trend = X beta_trend + N(0, 1); no intercept; X AR(1)"))
write_with_header(do.call(rbind, df_rows), "difference_fits.csv",
  c(sprintf("fused: fusedlasso1d(y_fused, X); trend: trendfilter(y_trend, X, ord = 1) (D of order 2); fused_enet: genlasso([y; 0], [X; sqrt(c) D1], D1) at lambda = a (%s)", ver),
    "objective_laplace = RSS/2 + a |D beta|_1 + c |D beta|^2/2 at the polished point, a = lambda alpha, c = lambda (1 - alpha), no intercept;",
    "lambda = lambda_fraction * lambda_path_top (the first knot of the path of the unaugmented problem);",
    "Laplace's penalty value is a |D beta|_1 + c |D beta|^2/2 + (p - order) penalty_constant_per_row - log_det_ddt / 2",
    "(laplace2 at rate lambda, or enet at (lambda, alpha), on each row of D; c(D) = -(1/2) log det(D D^T));",
    "nonzero_rows = 1-based rows i of D with (D beta)_i != 0; dual_max = max |u_i| over the zero rows (< 1: strict certificate);",
    "dual_residual = largest residual of a D'u = -grad, in gradient units"))
write_with_header(do.call(rbind, dc_rows), "difference.csv",
  "coefficients as C99 hex doubles: polished = exact solution on genlasso's fusion pattern (the reference), genlasso = genlasso's own")
writeLines(unlist(timing), file.path(tempdir(), "linear_map_timing.txt"))
