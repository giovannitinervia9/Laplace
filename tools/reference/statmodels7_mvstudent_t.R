# Frozen statmodels7 results for Laplace's multivariate Student t: mvstudent_t1
# (scale matrix Sigma = M(theta)) and mvstudent_t2 (Q = M(theta)) over the
# log-Cholesky structure (distributions7 0.94.2, parameters7 0.27.0).
# statmodels7 is used as an oracle only: this script calls its exported
# functions and writes their results; no code of it enters Laplace.
#
# Usage (from the repository root, with statmodels7 installed):
#   Rscript --vanilla tools/reference/statmodels7_mvstudent_t.R
#
# File written (hex doubles as by sprintf("%a"); lo is always 0, the values are
# statmodels7's doubles, not exact ones): tests/data/reference/mvstudent_t_s7.csv,
# in the format of mvstudent_t.csv (tools/reference/distrib_mvstudent_t.py),
# structure 3 only:
#   param,structure,d,theta,y,mu,A,idx,entry,hi,lo
# theta is the rest, the log-Cholesky parameters in Laplace's order followed by
# nu; idx the sorted tuple of rest indices (nu is the last, q). A >= 0 is the
# block with A derivatives in mu (A + K <= 4, the orders statmodels7 has in
# closed form: the log-density, distrib_gradient, distrib_hessian, and
# distrib_deriv3 and distrib_deriv4 with expected = FALSE); entry 0 for A = 0,
# i for A = 1, i (i + 1)/2 + j (i >= j) for A = 2, and for A >= 3 the rank of
# the sorted mu tuple in lexicographic order. A = -1 is the expected information
# in the rest at order 0, -distrib_expected_hessian (closed form), at entry
# a (a + 1)/2 + b (a >= b). Nothing of statmodels7's sampled expected third and
# fourth derivatives is used. Every derivative is on the parameter scale (in nu
# itself, not in its log).
#
# Parametrization map: as in statmodels7_multivariate.R (lap_of_s7).
#
# Points: d = 2 and 3 with theta, y, mu from set.seed(20261118 + d) (theta
# N(0, 0.5^2), y and mu N(0, 1)) at nu = 4.5, and the d = 2 point again at
# nu = 0.8 and nu = 400.

suppressPackageStartupMessages(library(distributions7))
stopifnot(packageVersion("distributions7") == "0.94.2",
          packageVersion("parameters7") == "0.27.0")

hx <- function(x) sprintf("%a", x)
hxl <- function(v) paste(hx(v), collapse = ";")

lap_of_s7 <- function(p) {
  ij <- rbind(cbind(0:(p - 1), 0:(p - 1)),
              do.call(rbind, lapply(seq_len(p - 1) - 1, function(j)
                cbind((j + 1):(p - 1), j))))
  ij[, 1] * (ij[, 1] + 1) / 2 + ij[, 2]
}

# lexicographic rank (0-based) of a sorted tuple of 0-based indices over d
sorted_rank <- function(t, d) {
  A <- length(t)
  g <- as.matrix(expand.grid(rep(list(0:(d - 1)), A)))[, A:1, drop = FALSE]
  g <- g[apply(g, 1, function(r) all(diff(r) >= 0)), , drop = FALSE]
  g <- g[do.call(order, as.data.frame(g)), , drop = FALSE]
  which(apply(g, 1, function(r) all(r == t))) - 1
}

out_dir <- file.path("tests", "data", "reference")
stopifnot(dir.exists(out_dir))

points <- list()
for (d in 2:3) {
  set.seed(20261118 + d)
  q <- d * (d + 1) / 2
  points[[length(points) + 1]] <- list(d = d, theta = rnorm(q, 0, 0.5), nu = 4.5,
                                       y = rnorm(d), mu = rnorm(d))
}
for (nu in c(0.8, 400)) {
  p <- points[[1]]
  p$nu <- nu
  points[[length(points) + 1]] <- p
}

rows <- character(0)
for (pt in points) {
  d <- pt$d
  q <- d * (d + 1) / 2
  lap <- c(lap_of_s7(d), q)           # Laplace position of each s7 rest index (nu last in both)
  th_s7 <- pt$theta[lap_of_s7(d) + 1]
  for (param in 1:2) {
    s <- parameters7::log_cholesky(d)
    D <- if (param == 1) mvstudent_t1_distrib(d, s) else mvstudent_t2_distrib(d, s)
    pn <- D@params                    # d locations, q structure values, nu
    stopifnot(length(pn) == d + q + 1, pn[length(pn)] == "nu")
    th <- as.list(setNames(c(pt$mu, th_s7, pt$nu), pn))
    y <- matrix(pt$y, 1)
    lists <- list(
      list(`0` = distrib_pdf(D, y, th, log = TRUE)),
      distrib_gradient(D, y, th),
      distrib_hessian(D, y, th),
      distrib_deriv3(D, y, th),
      distrib_deriv4(D, y, th))
    key <- paste(param, 3, d, hxl(c(pt$theta, pt$nu)), hxl(pt$y), hxl(pt$mu), sep = ",")
    for (ord in 0:4) {
      tuples <- if (ord == 0) list(integer(0)) else {
        g <- as.matrix(expand.grid(rep(list(seq_along(pn)), ord)))
        g <- g[apply(g, 1, function(r) all(diff(r) >= 0)), , drop = FALSE]
        lapply(seq_len(nrow(g)), function(r) unname(g[r, ]))
      }
      for (tu in tuples) {
        m <- tu[tu <= d]
        t <- tu[tu > d] - d
        A <- length(m)
        nm <- if (ord == 0) "0" else paste(pn[tu], collapse = "_")
        v <- lists[[ord + 1]][[nm]]
        if (is.null(v)) stop("missing ", nm)
        v <- v[1]
        idx <- sort(lap[t])
        entry <- if (A == 0) 0 else if (A == 1) m - 1 else if (A == 2) {
          i <- max(m) - 1; j <- min(m) - 1; i * (i + 1) / 2 + j } else sorted_rank(sort(m - 1), d)
        rows <- c(rows, paste(key, A, if (length(idx)) paste(idx, collapse = ";") else "-",
                              entry, hx(v), hx(0), sep = ","))
      }
    }
    eh <- distrib_expected_hessian(D, y, th)
    for (a in 1:(q + 1)) for (b in a:(q + 1)) {
      v <- eh[[paste(pn[d + a], pn[d + b], sep = "_")]]
      if (is.null(v)) stop("missing information ", a, " ", b)
      la <- max(lap[a], lap[b]); lb <- min(lap[a], lap[b])
      rows <- c(rows, paste(key, -1, "-", la * (la + 1) / 2 + lb, hx(-v[1]), hx(0), sep = ","))
    }
  }
}
writeLines(c("param,structure,d,theta,y,mu,A,idx,entry,hi,lo", rows),
           file.path(out_dir, "mvstudent_t_s7.csv"))
cat("mvstudent_t_s7.csv:", length(rows), "rows\n")
