# Frozen references for the non-convex penalties of wave 2 of the optimizers
# (note/optimizers.md section 12, point 16): SCAD (a = 3.7) and MCP
# (gamma = 3) fitted by ncvreg on the gaussian datasets of
# tools/reference/optimizers_wave2.R (read back from their files, which that
# script must have written first).
#
# Run from the repository root: Rscript --vanilla tools/reference/optimizers_wave2_ncvreg.R
#
# Output in tests/data/reference/optimizers/wave2/:
#   ncvreg_fits.csv   one row per (dataset, penalty, lambda_fraction): objective,
#                     support, KKT, local and global convexity
#   ncvreg.csv        one row per (dataset, penalty, lambda_fraction, coefficient)
#   ncvreg_weights.csv one row per (dataset, coefficient): the weight w_j
#
# The conversion. ncvreg (family "gaussian") centres y and standardizes each
# column of X to mean 0 and sum(x^2)/n = 1 (scale s_j = sqrt(sum((x_j - mean)^2) / n)),
# and minimizes, in the original coordinates beta_j = beta~_j / s_j,
#   F_n(b0, beta) = RSS / (2 n) + sum_j q(s_j |beta_j|; lambda, shape),
# q the MCP or SCAD of laplace::penalty (nonconvex_penalty.hpp). Both q satisfy
#   n q(u; lambda, shape) = q(sqrt(n) u; sqrt(n) lambda, shape)
# (each piece is homogeneous of degree 2 in (u, lambda) jointly), so n F_n is
#   F_L = RSS / 2 + sum_j q(w_j |beta_j|; lambda_L, shape),
#   w_j = sqrt(n) s_j = |x_j - mean(x_j)|_2,  lambda_L = sqrt(n) lambda,
# the gaussian -l with sigma = 1 (less n log(2 pi) / 2) plus
# linear_map_penalty<mcp_penalty or scad_penalty, diagonal_map> with weights w
# and theta = (lambda_L, shape). The penalty is unnormalizable, so the map adds
# no constant c(W) = -sum_j log w_j to its value (sum_log_weights in
# ncvreg_fits.csv is informative only); the intercept b0 is unpenalized.
#
# Polishing. ncvreg stops on a relative change of the coefficients, so each fit
# is re-solved exactly: with ncvreg's support S, signs s and the piece of q on
# which each u_j = w_j |beta_j| lies, the stationarity on S is linear,
#   (Xc_S' Xc_S - diag(d_S)) beta_S = Xc_S' yc - (w s k)_S,  b0 = mean(y) - mean(X)' beta,
# with (k, d) = (lambda_L, w^2 / gamma) on MCP's inner piece, (0, 0) on the flat
# tail, (lambda_L, 0) on SCAD's first piece and (a lambda_L / (a - 1),
# w^2 / (a - 1)) on its second. The polished point is accepted only if the
# signs and pieces are unchanged and each coordinate off S satisfies
# |Xc_j' r| < lambda_L w_j strictly. min_eigen_support, the smallest eigenvalue
# of Xc_S' Xc_S - diag(d_S), tells whether it is a strict local minimum (> 0);
# globally_convex says whether F_L is convex, which holds when the smallest
# eigenvalue of W^-1 Xc' Xc W^-1 exceeds rho (1/gamma, 1/(a - 1)); then the
# minimum is unique.

suppressMessages(library(ncvreg))
out_dir <- "tests/data/reference/optimizers/wave2"
h <- function(x) sprintf("%a", x)
g17 <- function(x) sprintf("%.17g", x)

penalties <- list(list(name = "MCP", shape = 3), list(name = "SCAD", shape = 3.7))
lambda_fractions <- c(0.5, 0.1, 0.01)  # of ncvreg's lambda_max = max |Xs' yc| / n

read_dataset <- function(row) {
  n <- row$n; p <- row$p
  f <- file.path(out_dir, row$file)
  if (grepl("\\.f64$", row$file)) {
    m <- matrix(readBin(f, "double", n = n * (p + 2), size = 8, endian = "little"), n)
  } else {
    m <- as.matrix(read.csv(f, colClasses = "character"))
    m <- matrix(as.numeric(m), n)
  }
  list(X = m[, seq_len(p), drop = FALSE], y = m[, p + 1])
}

q_pen <- function(u, l, a, pen) {
  if (pen == "MCP") ifelse(u <= a * l, l * u - u^2 / (2 * a), a * l^2 / 2)
  else ifelse(u <= l, l * u, ifelse(u <= a * l, l * u - (u - l)^2 / (2 * (a - 1)), (a + 1) * l^2 / 2))
}
# piece index of u: 0 first, 1 middle (SCAD), 2 flat; MCP has 0 and 2
piece_of <- function(u, l, a, pen) {
  if (pen == "MCP") ifelse(u <= a * l, 0L, 2L) else ifelse(u <= l, 0L, ifelse(u <= a * l, 1L, 2L))
}
k_and_d <- function(pc, w, l, a, pen) {
  if (pen == "MCP") list(k = ifelse(pc == 0, l, 0), d = ifelse(pc == 0, w^2 / a, 0))
  else list(k = ifelse(pc == 0, l, ifelse(pc == 1, a * l / (a - 1), 0)), d = ifelse(pc == 1, w^2 / (a - 1), 0))
}

# KKT of F_L at (b0, beta): stationarity on the support, the subgradient bound off it
kkt <- function(X, y, b0, beta, w, l, a, pen) {
  r <- y - b0 - drop(X %*% beta)
  g <- -drop(crossprod(X, r))  # gradient of RSS/2
  u <- w * abs(beta)
  on <- beta != 0
  pc <- piece_of(u, l, a, pen)
  kd <- k_and_d(pc, w, l, a, pen)
  dq <- w * sign(beta) * kd$k - kd$d * beta  # derivative of q(w |beta|) on the support
  v <- numeric(length(beta))
  v[on] <- abs(g[on] + dq[on])
  v[!on] <- pmax(0, abs(g[!on]) - l * w[!on])
  list(max_violation = max(abs(sum(r)), v), slack = if (any(!on)) min(l * w[!on] - abs(g[!on])) else NA_real_,
       objective = sum(r^2) / 2 + sum(q_pen(u, l, a, pen)), pieces = pc)
}

polish <- function(X, y, beta, w, l, a, pen) {
  S <- which(beta != 0)
  s <- sign(beta[S])
  xm <- colMeans(X); ym <- mean(y)
  pc <- piece_of(w * abs(beta), l, a, pen)
  b <- numeric(ncol(X)); min_eig <- NA_real_
  if (length(S)) {
    kd <- k_and_d(pc[S], w[S], l, a, pen)
    Xs <- sweep(X[, S, drop = FALSE], 2, xm[S])
    A <- crossprod(Xs) - diag(kd$d, length(S))
    rhs <- drop(crossprod(Xs, y - ym)) - w[S] * s * kd$k
    b[S] <- solve(A, rhs)
    b[S] <- b[S] + solve(A, rhs - drop(A %*% b[S]))  # one step of iterative refinement
    min_eig <- min(eigen(A, symmetric = TRUE, only.values = TRUE)$values)
  }
  pc_new <- piece_of(w * abs(b), l, a, pen)
  list(b0 = ym - sum(xm * b), beta = b, signs_ok = all(sign(b[S]) == s), pieces_ok = all(pc_new[S] == pc[S]),
       min_eig = min_eig)
}

ds_tab <- read.csv(file.path(out_dir, "datasets.csv"), comment.char = "#", stringsAsFactors = FALSE)
fit_rows <- list(); coef_rows <- list(); weight_rows <- list(); timing <- list()
for (k in seq_len(nrow(ds_tab))) {
  row <- ds_tab[k, ]
  d <- read_dataset(row)
  X <- d$X; y <- d$y; n <- row$n; p <- row$p
  xm <- colMeans(X); Xc <- sweep(X, 2, xm); yc <- y - mean(y)
  w <- sqrt(colSums(Xc^2))
  weight_rows[[k]] <- data.frame(dataset = row$dataset, coef = paste0("beta", seq_len(p)), weight = h(w))
  # smallest eigenvalue of W^-1 Xc'Xc W^-1 (ncvreg's X'X / n): global convexity
  eig_std <- if (p > n) 0 else min(eigen(crossprod(sweep(Xc, 2, w, "/")), symmetric = TRUE, only.values = TRUE)$values)
  lambda_max <- max(abs(crossprod(Xc, yc)) / w) / sqrt(n)  # ncvreg's: max |Xs' yc| / n, Xs = Xc sqrt(n) / w
  # the path: 100 log-spaced values from lambda_max to 0.01 lambda_max, the fractions included
  grid <- exp(seq(0, log(0.01), length.out = 100))
  grid <- grid[sapply(grid, function(g) all(abs(g - lambda_fractions) > 1e-12))]
  path_fr <- sort(c(grid, lambda_fractions), decreasing = TRUE)
  for (pz in penalties) {
    a <- pz$shape; rho <- if (pz$name == "MCP") 1 / a else 1 / (a - 1)
    t0 <- Sys.time()
    f <- suppressWarnings(ncvreg(X, y, family = "gaussian", penalty = pz$name, gamma = a,
                                 lambda = lambda_max * path_fr, eps = 1e-14, max.iter = 1e7, warn = FALSE))
    timing[[length(timing) + 1]] <- data.frame(dataset = row$dataset, penalty = pz$name,
                                               seconds = as.numeric(Sys.time() - t0, units = "secs"))
    for (fr in lambda_fractions) {
      li <- which(abs(path_fr - fr) < 1e-15)
      stopifnot(length(li) == 1)
      lam <- lambda_max * fr; lL <- sqrt(n) * lam
      b0 <- unname(f$beta[1, li]); beta <- unname(f$beta[-1, li])
      # the conversion checked: ncvreg's own objective at its point, times n, equals F_L
      Fn <- sum((y - b0 - drop(X %*% beta))^2) / (2 * n) + sum(q_pen(w / sqrt(n) * abs(beta), lam, a, pz$name))
      kg <- kkt(X, y, b0, beta, w, lL, a, pz$name)
      stopifnot(abs(n * Fn - kg$objective) <= 1e-12 * kg$objective)
      pl <- polish(X, y, beta, w, lL, a, pz$name)
      kp <- kkt(X, y, pl$b0, pl$beta, w, lL, a, pz$name)
      accepted <- pl$signs_ok && pl$pieces_ok && (is.na(kp$slack) || kp$slack > 0)
      S <- which(pl$beta != 0)
      cat(sprintf("%-16s %-4s fr %-5g support %3d  kkt ncvreg %.3g polished %.3g  slack %.3g  min_eig %.3g  convex %s  diff %.3g  %s\n",
                  row$dataset, pz$name, fr, length(S), kg$max_violation, kp$max_violation, kp$slack, pl$min_eig,
                  eig_std > rho, max(abs(c(b0 - pl$b0, beta - pl$beta))), if (accepted) "ok" else "REJECTED"))
      stopifnot(accepted)
      fit_rows[[length(fit_rows) + 1]] <- data.frame(
        dataset = row$dataset, penalty = pz$name, shape = a, lambda_fraction = fr,
        lambda_ncvreg = h(lam), lambda_laplace = h(lL),
        support_size = length(S), support = paste(S, collapse = " "),
        pieces = paste(kp$pieces[S], collapse = " "),
        objective_laplace = h(kp$objective), objective_ncvreg_units = h(kp$objective / n),
        objective_laplace_ncvreg_point = h(kg$objective),
        sum_log_weights = h(sum(log(w))),
        kkt_violation_polished = g17(kp$max_violation), kkt_violation_ncvreg = g17(kg$max_violation),
        kkt_slack_off_support = g17(kp$slack), min_eigen_support = g17(pl$min_eig),
        min_eigen_standardized = g17(eig_std), rho = g17(rho), globally_convex = eig_std > rho,
        max_abs_diff_ncvreg_polished = g17(max(abs(c(b0 - pl$b0, beta - pl$beta)))))
      coef_rows[[length(coef_rows) + 1]] <- data.frame(
        dataset = row$dataset, penalty = pz$name, lambda_fraction = fr,
        coef = c("b0", paste0("beta", seq_len(p))),
        polished = h(c(pl$b0, pl$beta)), ncvreg = h(c(b0, beta)))
    }
  }
}

write_with_header <- function(d, f, header) {
  con <- file(file.path(out_dir, f), "w")
  writeLines(paste("#", header), con)
  write.csv(d, con, row.names = FALSE)
  close(con)
}
ver <- sprintf("ncvreg %s, R %s", packageVersion("ncvreg"), getRversion())
write_with_header(do.call(rbind, fit_rows), "ncvreg_fits.csv",
  c(sprintf("ncvreg(X, y_gaussian, penalty, gamma = shape, lambda = path to 0.01 lambda_max, eps = 1e-14) (%s)", ver),
    "objective_laplace = RSS/2 + sum_j q(w_j |beta_j|; lambda_laplace, shape) at the polished point = n times ncvreg's objective;",
    "lambda_laplace = sqrt(n) lambda_ncvreg, w_j = |x_j - mean(x_j)|_2 (ncvreg_weights.csv); Laplace's linear_map_penalty with",
    "diagonal_map adds nothing to the value (unnormalizable; sum_log_weights informative only); pieces: 0 first, 1 middle (SCAD), 2 flat, one per support index;",
    "kkt values in objective_laplace units; slack = min over j off the support of lambda_laplace w_j - |X_j' r| (> 0: strict);",
    "min_eigen_support > 0: strict local minimum; globally_convex: min_eigen_standardized > rho, a unique minimum."))
write_with_header(do.call(rbind, coef_rows), "ncvreg.csv",
  "coefficients as C99 hex doubles: polished = exact stationary point on ncvreg's support and pieces (the reference), ncvreg = ncvreg's own")
write_with_header(do.call(rbind, weight_rows), "ncvreg_weights.csv",
  "w_j = |x_j - mean(x_j)|_2 as C99 hex doubles: sqrt(n) times ncvreg's column scale")
# timings: local only (note/optimizers.md section 10), never in the reference files
tf <- Sys.getenv("LAPLACE_NCVREG_TIMING")
if (nzchar(tf)) write.csv(do.call(rbind, timing), tf, row.names = FALSE)
