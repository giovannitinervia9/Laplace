# Frozen statmodels7 results for Laplace's multivariate part: the multivariate
# Gaussian over the log-Cholesky structure (distributions7 0.94.2) and the
# structured and additive penalties (penalties7 0.34.0, parameters7 0.27.0).
# statmodels7 is used as an oracle only: this script calls its exported
# functions and writes their results; no code of it enters Laplace.
#
# Usage (from the repository root, with statmodels7 installed):
#   Rscript --vanilla tools/reference/statmodels7_multivariate.R
#
# Files written (hex doubles as by sprintf("%a"); lo is always 0, the values
# are statmodels7's doubles, not exact ones):
#
# 1. tests/data/reference/mvgaussian_s7.csv, in the format of mvgaussian.csv
#    (tools/reference/distrib_mvgaussian.py), structure 3 only:
#      param,structure,d,theta,y,mu,A,idx,entry,hi,lo
#    param 1 is mvgaussian1 (Sigma = M(theta)), 2 is mvgaussian2 (Q = M(theta)).
#    A = 0, 1, 2 is the block of the log-density with A derivatives in mu and
#    the sorted theta tuple idx (A + K <= 4, the orders statmodels7 has: the
#    log-density, distrib_gradient, distrib_hessian, distrib_deriv3,
#    distrib_deriv4); entry 0 for A = 0, i for A = 1, i (i + 1)/2 + j (i >= j)
#    for A = 2. A = -1 is the expected information in theta at order 0,
#    -distrib_expected_hessian, at entry a (a + 1)/2 + b (a >= b).
#
# 2. tests/data/reference/penalty_s7.csv, one row per value:
#      penalty,case,n_coef,theta,beta,matrices,quantity,idx,entry,hi,lo
#    penalty is structured (structured_penalty(log_cholesky(n_coef)), theta in
#    Laplace's order, matrices '-') or additive (additive_penalty(list(P_1,
#    ..., P_m)), theta = (lambda_1, ..., lambda_m), matrices the P_k row-major,
#    entries joined by ';', matrices by '|'). The penalty is
#      rho(beta; theta) = 1/2 beta' Omega(theta) beta - 1/2 log pdet Omega(theta)
#                         + r/2 log 2 pi,
#    r the rank of Omega; structured: Omega = M(theta) (the precision);
#    additive: Omega = sum_k lambda_k P_k. All derivatives are on the parameter
#    scale (in theta and lambda themselves, not in a link of them).
#    quantity, idx and entry:
#      value        rho                                        idx -,     entry 0
#      logpdet      log pdet Omega                              idx -,     entry 0
#      grad_beta    d rho / d beta_i                            idx -,     entry i
#      grad_theta   d rho / d theta_k                           idx k,     entry 0
#      hess_theta   d^2 rho / d theta_k d theta_l (k <= l)      idx k;l,   entry 0
#      cross        d^2 rho / d beta_i d theta_k                idx k,     entry i
#
# Parametrization map (log-Cholesky). Both libraries use L_ii = exp(t_ii),
# L_ij = t_ij (i > j), M = L L' (checked on param_factor). They differ in the
# order of t:
#   Laplace (unstructured_pdmat<log_cholesky_bijector>): the packed lower
#     triangle by rows, t_ij at i (i + 1)/2 + j (0-based, i >= j), so the
#     diagonal is interleaved: (log L11, L21, log L22, L31, L32, log L33, ...).
#   statmodels7 (parameters7::log_cholesky): the p log-diagonal entries first,
#     then the strict lower triangle by columns: (log_L1, ..., log_Lp, L2.1,
#     L3.1, ..., Lp.1, L3.2, ..., Lp.(p-1)).
# Every theta, idx and entry in the files is in Laplace's order (0-based); the
# map is lap_of_s7 below. mu and y are the same in both.
#
# Points: d = 2, 3, 4 with theta, y, mu from set.seed(20261116 + d) (theta
# N(0, 0.5^2), y and mu N(0, 1)), and a near-singular one at d = 3 with
# log L33 = -8 (Sigma, or Q, has a smallest eigenvalue near e^-16).

suppressPackageStartupMessages({
  library(distributions7)
  library(penalties7)
})
stopifnot(packageVersion("distributions7") == "0.94.2",
          packageVersion("penalties7") == "0.34.0",
          packageVersion("parameters7") == "0.27.0")

hx <- function(x) sprintf("%a", x)
hxl <- function(v) paste(hx(v), collapse = ";")

# s7 index (1-based) of Laplace's 0-based packed position, for dimension p
lap_of_s7 <- function(p) {
  # s7 order -> (i, j) 0-based
  ij <- rbind(cbind(0:(p - 1), 0:(p - 1)),
              do.call(rbind, lapply(seq_len(p - 1) - 1, function(j)
                cbind((j + 1):(p - 1), j))))
  ij[, 1] * (ij[, 1] + 1) / 2 + ij[, 2]  # Laplace position of each s7 index
}

out_dir <- file.path("tests", "data", "reference")
stopifnot(dir.exists(out_dir))

## 1. mvgaussian -------------------------------------------------------------

points <- list()
for (d in 2:4) {
  set.seed(20261116 + d)
  q <- d * (d + 1) / 2
  points[[length(points) + 1]] <- list(d = d, theta = rnorm(q, 0, 0.5),
                                       y = rnorm(d), mu = rnorm(d))
}
# near singular, theta in Laplace's order: (log L11, L21, log L22, L31, L32, log L33)
points[[length(points) + 1]] <- list(d = 3, theta = c(0.2, 0.7, -0.1, -0.4, 0.9, -8),
                                     y = c(0.3, -1.2, 0.8), mu = c(0.1, -0.5, 0.4))

rows <- character(0)
for (pt in points) {
  d <- pt$d
  q <- d * (d + 1) / 2
  lap <- lap_of_s7(d)                 # lap[k] = Laplace position of s7 theta k
  th_s7 <- pt$theta[lap + 1]          # s7 order from Laplace order
  for (param in 1:2) {
    s <- parameters7::log_cholesky(d)
    D <- if (param == 1) mvgaussian1_distrib(d, s) else mvgaussian2_distrib(d, s)
    pn <- D@params                    # d means then q structure values, s7 order
    stopifnot(length(pn) == d + q)
    th <- as.list(setNames(c(pt$mu, th_s7), pn))
    y <- matrix(pt$y, 1)
    lists <- list(
      list(`0` = distrib_pdf(D, y, th, log = TRUE)),
      distrib_gradient(D, y, th),
      distrib_hessian(D, y, th),
      distrib_deriv3(D, y, th),
      distrib_deriv4(D, y, th))
    key <- paste(param, 3, d, hxl(pt$theta), hxl(pt$y), hxl(pt$mu), sep = ",")
    for (ord in 0:4) {
      tuples <- if (ord == 0) list(integer(0)) else {
        # multisets of parameter indices (1-based, s7 order), nondecreasing
        g <- as.matrix(expand.grid(rep(list(seq_along(pn)), ord)))
        g <- g[apply(g, 1, function(r) all(diff(r) >= 0)), , drop = FALSE]
        lapply(seq_len(nrow(g)), function(r) unname(g[r, ]))
      }
      for (tu in tuples) {
        m <- tu[tu <= d]               # mean indices (1-based)
        t <- tu[tu > d] - d            # s7 theta indices (1-based)
        A <- length(m)
        if (A > 2) next
        nm <- if (ord == 0) "0" else paste(pn[tu], collapse = "_")
        v <- lists[[ord + 1]][[nm]]
        if (is.null(v)) stop("missing ", nm)
        v <- v[1]
        idx <- sort(lap[t])
        entry <- if (A == 0) 0 else if (A == 1) m - 1 else {
          i <- max(m) - 1; j <- min(m) - 1; i * (i + 1) / 2 + j }
        rows <- c(rows, paste(key, A, if (length(idx)) paste(idx, collapse = ";") else "-",
                              entry, hx(v), hx(0), sep = ","))
      }
    }
    # expected information in theta: -E[d^2 l / dtheta_a dtheta_b]
    eh <- distrib_expected_hessian(D, y, th)
    for (a in 1:q) for (b in a:q) {
      v <- eh[[paste(pn[d + a], pn[d + b], sep = "_")]]
      if (is.null(v)) stop("missing information ", a, " ", b)
      la <- max(lap[a], lap[b]); lb <- min(lap[a], lap[b])
      rows <- c(rows, paste(key, -1, "-", la * (la + 1) / 2 + lb, hx(-v[1]), hx(0), sep = ","))
    }
  }
}
writeLines(c("param,structure,d,theta,y,mu,A,idx,entry,hi,lo", rows),
           file.path(out_dir, "mvgaussian_s7.csv"))
cat("mvgaussian_s7.csv:", length(rows), "rows\n")

## 2. penalties --------------------------------------------------------------

prow <- character(0)
emit <- function(kind, case, n, theta, beta, mats, quantity, idx, entry, v) {
  prow <<- c(prow, paste(kind, case, n, hxl(theta), hxl(beta), mats, quantity,
                         idx, entry, hx(v), hx(0), sep = ","))
}
freeze_penalty <- function(kind, case, pen, theta_lap, theta_list, beta, mats, perm) {
  # perm[k] = Laplace index (0-based) of penalty parameter k (pen@params order)
  n <- length(beta)
  emit(kind, case, n, theta_lap, beta, mats, "value", "-", 0, penalty_value(pen, beta, theta_list))
  emit(kind, case, n, theta_lap, beta, mats, "logpdet", "-", 0, penalty_logpdet(pen, theta_list)$value)
  gb <- penalty_gradient(pen, beta, theta_list)
  for (i in seq_len(n)) emit(kind, case, n, theta_lap, beta, mats, "grad_beta", "-", i - 1, gb[i])
  gt <- penalty_grad_theta(pen, beta, theta_list)
  ht <- penalty_hess_theta(pen, beta, theta_list)
  cr <- penalty_cross(pen, beta, theta_list)
  pn <- pen@params
  for (k in seq_along(pn)) {
    emit(kind, case, n, theta_lap, beta, mats, "grad_theta", perm[k], 0, gt[[pn[k]]])
    for (i in seq_len(n)) emit(kind, case, n, theta_lap, beta, mats, "cross", perm[k], i - 1, cr[[pn[k]]][i])
    for (l in k:length(pn)) {
      v <- ht[[paste(pn[k], pn[l], sep = "_")]]
      if (is.null(v)) v <- ht[[paste(pn[l], pn[k], sep = "_")]]
      if (is.null(v)) stop("missing hess_theta ", pn[k], " ", pn[l])
      emit(kind, case, n, theta_lap, beta, mats, "hess_theta",
           paste(sort(c(perm[k], perm[l])), collapse = ";"), 0, v)
    }
  }
}

# structured: Omega = log-Cholesky M(theta), d = 2, 3, 4 and the near-singular point
case <- 0
for (pt in points) {
  d <- pt$d
  lap <- lap_of_s7(d)
  pen <- structured_penalty(parameters7::log_cholesky(d))
  th <- as.list(setNames(pt$theta[lap + 1], pen@params))
  freeze_penalty("structured", case, pen, pt$theta, th, pt$y, "-", lap)
  case <- case + 1
}

# additive: Omega = sum_k lambda_k P_k
D1 <- diff(diag(4))
P_a <- list(diag(4), crossprod(D1))                     # full rank sum
D2 <- crossprod(diff(diag(3), differences = 2))
P_b <- list(kronecker(diag(3), D2), kronecker(D2, diag(3)))  # tensor, rank 8 of 9
set.seed(20261117)
add_cases <- list(list(P = P_a, lambda = c(0.7, 2.5), beta = rnorm(4)),
                  list(P = P_b, lambda = c(1.3, 0.4), beta = rnorm(9)),
                  list(P = P_b, lambda = c(1, 1e6), beta = rnorm(9)))
case <- 0
for (ac in add_cases) {
  pen <- additive_penalty(ac$P)
  th <- as.list(setNames(ac$lambda, pen@params))
  mats <- paste(vapply(ac$P, function(P) hxl(t(P)), ""), collapse = "|")  # row-major
  freeze_penalty("additive", case, pen, ac$lambda, th, ac$beta, mats, seq_along(ac$lambda) - 1)
  case <- case + 1
}
writeLines(c("penalty,case,n_coef,theta,beta,matrices,quantity,idx,entry,hi,lo", prow),
           file.path(out_dir, "penalty_s7.csv"))
cat("penalty_s7.csv:", length(prow), "rows\n")
