# Frozen references for wave 2 of the optimizers (note/optimizers.md section 12,
# decisions 7 and 8): gaussian linear regressions with dense X fitted by
# glmnet (lasso and elastic net at fixed lambda) and least absolute deviation
# fits by quantreg::rq (method "br", the exact simplex LP), on data simulated
# with fixed seeds.
#
# Run from the repository root: Rscript --vanilla tools/reference/optimizers_wave2.R
#
# Output in tests/data/reference/optimizers/wave2/ (layout in its README.md):
#   <id>.csv or <id>.f64   the data: columns x1..xp, y_gaussian, y_lad
#   datasets.csv           one row per dataset (n, p, design, seed, file)
#   glmnet.csv             one row per (dataset, alpha, lambda, coefficient)
#   glmnet_fits.csv        one row per (dataset, alpha, lambda): objective, support, KKT
#   rq.csv                 one row per (dataset, coefficient)
#   rq_fits.csv            one row per dataset: objective, optimality certificate
#
# Objectives. glmnet (family "gaussian", standardize = FALSE, intercept b0
# unpenalized) minimizes
#   F_g(b0, beta) = RSS / (2 n) + lambda_g (alpha |beta|_1 + (1 - alpha) |beta|_2^2 / 2).
# In Laplace's units (sums, not means) n F_g is
#   F_L = RSS / 2 + a |beta|_1 + c |beta|_2^2 / 2,   a = n lambda_g alpha,  c = n lambda_g (1 - alpha),
# which is the smooth part RSS / 2 (the gaussian -l with sigma = 1, less
# n log(2 pi) / 2) plus, on each coefficient, enet_distrib's penalty with
# lambda = n lambda_g and the same alpha, less its log-normalizer A per
# coefficient; at alpha = 1, laplace2_distrib's penalty with rate
# lambda = n lambda_g (laplace1 with sigma = 1 / (n lambda_g)), less
# -log(lambda / 2) per coefficient. Both constants are written in glmnet_fits.csv.
#
# glmnet stops on a relative change, so its coefficients carry an error of
# order thresh^(1/2) in places. Each fit is therefore also "polished": with
# glmnet's support S and signs s, the KKT system on S,
#   (Xc_S' Xc_S + c I) beta_S = Xc_S' yc - a s_S,   b0 = mean(y) - mean(X)' beta,
# (Xc, yc centred) is solved directly, and the polished point is accepted only
# if its signs agree with s and every coordinate off S has |Xc_j' r| < a (the
# strict KKT inequality), which makes it the exact minimizer up to rounding.
#
# LAD: rq minimizes sum |y - b0 - X beta| with method "br". The solution is a
# vertex: p + 1 residuals are zero. The script re-solves the p + 1 by p + 1
# basis system exactly and certifies optimality by solving for the dual on the
# basis, u_B = -(X1_B')^(-1) X1_N' sign(r_N): the point is optimal iff
# max |u_B| <= 1 (and the solution unique if < 1).

suppressMessages({ library(glmnet); library(quantreg) })
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

out_dir <- "tests/data/reference/optimizers/wave2"
hex_limit <- 25000  # n * (p + 2) up to this goes to hex CSV, above to raw .f64

# design: "independent", "ar1" (corr rho^|j - k|) or "equicorrelated" (corr rho)
datasets <- list(
  list(id = "d1_n100_p10",      n = 100,   p = 10,  design = "independent",    rho = 0),
  list(id = "d2_n100_p200",     n = 100,   p = 200, design = "ar1",            rho = 0.9),
  list(id = "d3_n1000_p50",     n = 1000,  p = 50,  design = "ar1",            rho = 0.5),
  list(id = "d4_n10000_p100",   n = 10000, p = 100, design = "equicorrelated", rho = 0.5),
  list(id = "d5_n100000_p10",   n = 1e5,   p = 10,  design = "ar1",            rho = 0.8)
)
alphas <- c(1, 0.5)
lambda_fractions <- c(0.5, 0.1, 0.01)  # of lambda_max(alpha) = max |Xc' yc| / (n alpha)
intercept <- 0.5

make_x <- function(n, p, design, rho) {
  z <- matrix(rnorm(n * p), n, p)
  if (design == "ar1") {
    for (j in seq_len(p)[-1]) z[, j] <- rho * z[, j - 1] + sqrt(1 - rho^2) * z[, j]
  } else if (design == "equicorrelated") {
    z <- sqrt(1 - rho) * z + sqrt(rho) * rnorm(n)
  }
  z
}
true_beta <- function(p) {
  k <- max(3, p %/% 10)
  b <- numeric(p)
  b[seq_len(k)] <- seq(2, 0.5, length.out = k) * (-1)^(seq_len(k) - 1)
  b
}

write_data <- function(ds, m) {
  if (length(m) <= hex_limit) {
    f <- paste0(ds$id, ".csv")
    lines <- apply(m, 1, function(r) paste(sprintf("%a", r), collapse = ","))
    writeLines(c(paste(colnames(m), collapse = ","), lines), file.path(out_dir, f))
    back <- as.matrix(read.csv(file.path(out_dir, f), colClasses = "character"))
    back <- matrix(as.numeric(back), nrow(m))
  } else {
    f <- paste0(ds$id, ".f64")
    writeBin(as.vector(m), file.path(out_dir, f), size = 8, endian = "little")
    back <- matrix(readBin(file.path(out_dir, f), "double", n = length(m), size = 8, endian = "little"), nrow(m))
  }
  stopifnot(identical(unname(back), unname(m)))
  f
}

h <- function(x) sprintf("%a", x)
g17 <- function(x) sprintf("%.17g", x)

# KKT of F_L at (b0, beta): stationarity in b0, and for each j
# |g_j + a s_j| on the support, max(0, |g_j| - a) off it, g = -X'r + c beta.
lasso_kkt <- function(X, y, b0, beta, a, c) {
  r <- y - b0 - drop(X %*% beta)
  g <- -drop(crossprod(X, r)) + c * beta
  on <- beta != 0
  v <- numeric(length(beta))
  v[on] <- abs(g[on] + a * sign(beta[on]))
  v[!on] <- pmax(0, abs(g[!on]) - a)
  list(max_violation = max(abs(sum(r)), v), slack = if (any(!on)) min(a - abs(g[!on])) else NA_real_,
       objective = sum(r^2) / 2 + a * sum(abs(beta)) + c * sum(beta^2) / 2, rss = sum(r^2))
}

polish <- function(X, y, beta, a, c) {
  S <- which(beta != 0)
  s <- sign(beta[S])
  xm <- colMeans(X); ym <- mean(y)
  b <- numeric(ncol(X))
  if (length(S)) {
    Xs <- sweep(X[, S, drop = FALSE], 2, xm[S])
    A <- crossprod(Xs) + diag(c, length(S))
    rhs <- drop(crossprod(Xs, y - ym)) - a * s
    b[S] <- solve(A, rhs)
    b[S] <- b[S] + solve(A, rhs - drop(A %*% b[S]))  # one step of iterative refinement
  }
  list(b0 = ym - sum(xm * b), beta = b, signs_ok = all(sign(b[S]) == s))
}

lad_certificate <- function(X1, y, coef) {
  r <- y - drop(X1 %*% coef)
  B <- order(abs(r))[seq_len(ncol(X1))]
  coef_v <- solve(X1[B, , drop = FALSE], y[B])
  r_v <- y - drop(X1 %*% coef_v)
  N <- setdiff(seq_along(y), B)
  u <- -solve(t(X1[B, , drop = FALSE]), drop(crossprod(X1[N, , drop = FALSE], sign(r_v[N]))))
  list(coef = coef_v, objective = sum(abs(r_v)), objective_rq = sum(abs(r)), dual_max = max(abs(u)),
       basis = sort(B))
}

if (!isTRUE(get0("define_cases_only"))) {
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
ds_rows <- list(); coef_rows <- list(); fit_rows <- list(); rq_rows <- list(); rqfit_rows <- list()
for (k in seq_along(datasets)) {
  ds <- datasets[[k]]
  seed <- 20261118 + 100 * k
  set.seed(seed)
  X <- make_x(ds$n, ds$p, ds$design, ds$rho)
  beta_true <- true_beta(ds$p)
  eta <- intercept + drop(X %*% beta_true)
  y_gaussian <- eta + rnorm(ds$n)
  y_lad <- eta + (rexp(ds$n) - rexp(ds$n))  # Laplace errors, scale 1
  m <- cbind(X, y_gaussian, y_lad)
  colnames(m) <- c(paste0("x", seq_len(ds$p)), "y_gaussian", "y_lad")
  file <- write_data(ds, m)
  ds_rows[[k]] <- data.frame(dataset = ds$id, n = as.integer(ds$n), p = ds$p, design = ds$design,
                             rho = ds$rho, seed = seed, file = file,
                             true_beta = paste(g17(beta_true), collapse = " "))
  n <- ds$n; y <- y_gaussian
  yc <- y - mean(y); Xc <- sweep(X, 2, colMeans(X))
  for (alpha in alphas) {
    lambda_max <- max(abs(crossprod(Xc, yc))) / (n * alpha)
    lambdas <- lambda_max * lambda_fractions
    t0 <- Sys.time()
    # glmnet divides y by sy (its sd with divisor n) and lambda by sy, which
    # leaves the lasso term in place and divides the ridge term by sy: it
    # minimizes RSS/(2n) + lambda' alpha' |beta|_1 + lambda' (1 - alpha') |beta|^2 / (2 sy)
    # (measured: the ridge weight its KKT point implies is 1/sy to 7 digits).
    # Calling it with lambda' = lambda k, alpha' = alpha / k, k = alpha + sy (1 - alpha),
    # makes that F_g at (lambda, alpha) exactly (k = 1 for the lasso).
    sy <- sqrt(mean(yc^2))
    kk <- alpha + sy * (1 - alpha)
    f <- glmnet(X, y, family = "gaussian", alpha = alpha / kk, lambda = lambdas * kk, standardize = FALSE,
                intercept = TRUE, control = list(thresh = 1e-14, maxit = 1e8))
    cat(sprintf("%-16s alpha %.1f  glmnet %.2f s\n", ds$id, alpha, as.numeric(Sys.time() - t0, units = "secs")))
    for (l in seq_along(lambdas)) {
      a <- n * lambdas[l] * alpha; c <- n * lambdas[l] * (1 - alpha)
      b0 <- unname(f$a0[l]); beta <- unname(as.numeric(f$beta[, l]))
      kg <- lasso_kkt(X, y, b0, beta, a, c)
      pl <- polish(X, y, beta, a, c)
      kp <- lasso_kkt(X, y, pl$b0, pl$beta, a, c)
      accepted <- pl$signs_ok && (is.na(kp$slack) || kp$slack > 0)
      lam_L <- n * lambdas[l]
      pen_const <- if (alpha == 1) -log(lam_L / 2) else {
        x <- a / sqrt(c); log(2) + pnorm(-x, log.p = TRUE) - dnorm(x, log = TRUE) - log(c) / 2
      }
      fit_rows[[length(fit_rows) + 1]] <- data.frame(
        dataset = ds$id, alpha = alpha, lambda_fraction = lambda_fractions[l],
        lambda_glmnet = h(lambdas[l]), lambda_laplace = h(lam_L),
        a = h(a), c = h(c),
        support_size = sum(pl$beta != 0), support = paste(which(pl$beta != 0), collapse = " "),
        support_glmnet_same = identical(which(beta != 0), which(pl$beta != 0)),
        objective_laplace = h(kp$objective), objective_glmnet_units = h(kp$objective / n),
        objective_laplace_glmnet_point = h(kg$objective),
        penalty_constant_per_coef = h(pen_const),
        kkt_violation_polished = g17(kp$max_violation), kkt_violation_glmnet = g17(kg$max_violation),
        kkt_slack_off_support = g17(kp$slack),
        max_abs_diff_glmnet_polished = g17(max(abs(c(b0 - pl$b0, beta - pl$beta)))),
        polished_accepted = accepted)
      cat(sprintf("   l %d  support %3d  kkt glmnet %.3g  polished %.3g  slack %.3g  diff %.3g
", l, sum(pl$beta != 0), kg$max_violation, kp$max_violation, kp$slack, max(abs(beta - pl$beta))))
      stopifnot(accepted)
      coef_rows[[length(coef_rows) + 1]] <- data.frame(
        dataset = ds$id, alpha = alpha, lambda_fraction = lambda_fractions[l],
        coef = c("b0", paste0("beta", seq_len(ds$p))),
        polished = h(c(pl$b0, pl$beta)), glmnet = h(c(b0, beta)))
    }
  }
  if (ds$p + 1 > ds$n) next  # LAD has no unique vertex with more coefficients than rows
  X1 <- cbind(1, X)
  t0 <- Sys.time()
  fr <- suppressWarnings(rq.fit(X1, y_lad, tau = 0.5, method = "br"))
  cat(sprintf("%-16s rq br %.2f s\n", ds$id, as.numeric(Sys.time() - t0, units = "secs")))
  cert <- lad_certificate(X1, y_lad, fr$coefficients)
  stopifnot(cert$dual_max <= 1 + 1e-9, cert$objective <= cert$objective_rq * (1 + 1e-14))
  sig <- cert$objective / n
  rqfit_rows[[length(rqfit_rows) + 1]] <- data.frame(
    dataset = ds$id, n = as.integer(n), p = ds$p,
    objective = h(cert$objective), objective_rq = h(cert$objective_rq),
    sigma_hat = h(sig), negloglik_laplace1 = h(n + n * log(2 * sig)),
    dual_max = g17(cert$dual_max), unique = cert$dual_max < 1 - 1e-9,
    max_abs_diff_rq_vertex = g17(max(abs(fr$coefficients - cert$coef))),
    basis = paste(cert$basis, collapse = " "))
  rq_rows[[length(rq_rows) + 1]] <- data.frame(dataset = ds$id, coef = c("b0", paste0("beta", seq_len(ds$p))),
                             vertex = h(cert$coef), rq = h(unname(fr$coefficients)))
}

write_with_header <- function(d, f, header) {
  con <- file(file.path(out_dir, f), "w")
  writeLines(paste("#", header), con)
  write.csv(d, con, row.names = FALSE)
  close(con)
}
ver <- sprintf("glmnet %s, quantreg %s, R %s", packageVersion("glmnet"), packageVersion("quantreg"), getRversion())
write_with_header(do.call(rbind, ds_rows), "datasets.csv",
  c(sprintf("written by tools/reference/optimizers_wave2.R (%s)", ver),
    "y_gaussian = 0.5 + X true_beta + N(0, 1) errors; y_lad = 0.5 + X true_beta + Laplace(0, 1) errors"))
write_with_header(do.call(rbind, fit_rows), "glmnet_fits.csv",
  c(sprintf("glmnet(X, y_gaussian, alpha, lambda = lambda_glmnet path, standardize = FALSE, intercept = TRUE, thresh = 1e-14) (%s)", ver),
    "objective_laplace = RSS/2 + a |beta|_1 + c |beta|^2/2 at the polished point, a = n lambda_glmnet alpha, c = n lambda_glmnet (1 - alpha);",
    "lambda_laplace = n lambda_glmnet is enet's lambda (laplace2's rate at alpha = 1); objective_glmnet_units = objective_laplace / n;",
    "penalty_constant_per_coef = what Laplace's penalty adds per coefficient (enet: its A; laplace2: -log(lambda/2));",
    "kkt values in objective_laplace units; slack = min over j off the support of a - |X_j' r| (> 0: strict)."))
write_with_header(do.call(rbind, coef_rows), "glmnet.csv",
  "coefficients as C99 hex doubles: polished = exact KKT solution on glmnet's support (the reference), glmnet = glmnet's own")
write_with_header(do.call(rbind, rqfit_rows), "rq_fits.csv",
  c(sprintf("quantreg::rq.fit(cbind(1, X), y_lad, tau = 0.5, method = \"br\") (%s)", ver),
    "objective = sum |r| at the vertex; sigma_hat = objective / n; negloglik_laplace1 = n + n log(2 sigma_hat), the laplace1 -l at (beta, sigma_hat);",
    "dual_max = max |u_B| of the dual on the basis (<= 1: optimal, < 1: unique); basis = the p + 1 observations (1-based) with zero residual"))
write_with_header(do.call(rbind, rq_rows), "rq.csv",
  "coefficients as C99 hex doubles: vertex = basis system re-solved (the reference), rq = rq.fit's own")

# ncvreg (SCAD, MCP): to be added when the penalties land (note/penalties.md, step 5).
# The fits will go to ncvreg.csv / ncvreg_fits.csv on the same data, with the map
# between ncvreg's (1/(2n)) RSS + sum P(|beta_j|; lambda, gamma) and Laplace's penalties.
}
