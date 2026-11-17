# Frozen statmodels7 results for Laplace's Dirichlet (dirichlet_distrib) and
# multinomial (multinomial_distrib), both over parameters7::simplex on its
# additive log-ratio chart, reference category the last, as Laplace's alr
# (distributions7 0.94.2, parameters7 0.27.0). statmodels7 is used as an oracle
# only: this script calls its exported functions and writes their results; no
# code of it enters Laplace.
#
# Usage (from the repository root, with statmodels7 installed):
#   Rscript --vanilla tools/reference/statmodels7_simplex.R
#
# File written (hex doubles as by sprintf("%a"); lo is always 0, the values are
# statmodels7's doubles, not exact ones): tests/data/reference/distrib_simplex_s7.csv,
# in the format of distrib_simplex.csv (tools/reference/distrib_simplex.py):
#   law,d,x,theta,y,size,kind,J,A,idx,entry,hi,lo
# law 1 the Dirichlet, 2 the multinomial; x the alr coordinates eta; theta phi.
# kind 0, J = 0: the block with A derivatives in eta and K in phi, A + K <= 4 (the
# orders statmodels7 has in closed form: the log-density, distrib_gradient,
# distrib_hessian, distrib_deriv3 and distrib_deriv4 with expected = FALSE);
# kinds 1, 2, 3 at A = 0 and no idx: the expected information (eta, eta),
# (eta, phi), (phi, phi), minus distrib_expected_hessian (closed form there).
# statmodels7's expected third and fourth derivatives, E[d^3 l] and E[d^4 l], are
# not frozen: Laplace returns the derivatives of the information instead, which
# differ from them by E[d^2 l x score] (the Dirichlet's y term enters every eta
# order). Every derivative is on the parameter scale (in phi itself).
#
# Points: d = 2, 3, 5 with eta from set.seed(20261117 + d) (N(0, 0.7^2)); the
# Dirichlet at phi = 4.5 and 300 with y a draw from distrib_rng; the multinomial
# at size 10 and 1000 with y a draw from distrib_rng.

suppressPackageStartupMessages(library(distributions7))
stopifnot(packageVersion("distributions7") == "0.94.2",
          packageVersion("parameters7") == "0.27.0")

hx <- function(x) sprintf("%a", x)
hxl <- function(v) paste(hx(v), collapse = ";")

# lexicographic rank (0-based) of a sorted tuple of 0-based indices over m
sorted_rank <- function(t, m) {
  A <- length(t)
  g <- as.matrix(expand.grid(rep(list(0:(m - 1)), A)))[, A:1, drop = FALSE]
  g <- g[apply(g, 1, function(r) all(diff(r) >= 0)), , drop = FALSE]
  g <- g[do.call(order, as.data.frame(g)), , drop = FALSE]
  which(apply(g, 1, function(r) all(r == t))) - 1
}
entry_of <- function(e, m) {
  A <- length(e)
  if (A == 0) 0 else if (A == 1) e else if (A == 2) max(e) * (max(e) + 1) / 2 + min(e) else sorted_rank(sort(e), m)
}

out_dir <- file.path("tests", "data", "reference")
stopifnot(dir.exists(out_dir))

rows <- character(0)
for (d in c(2, 3, 5)) {
  set.seed(20261117 + d)
  m <- d - 1
  eta <- rnorm(m, 0, 0.7)
  for (law in 1:2) {
    for (par in if (law == 1) c(4.5, 300) else c(10, 1000)) {
      D <- if (law == 1) dirichlet_distrib(d) else multinomial_distrib(d, size = par)
      pn <- D@params
      stopifnot(length(pn) == m + (law == 1))
      th <- as.list(setNames(if (law == 1) c(eta, par) else eta, pn))
      y <- distrib_rng(D, 1, th)
      lists <- list(
        list(`0` = distrib_pdf(D, y, th, log = TRUE)),
        distrib_gradient(D, y, th),
        distrib_hessian(D, y, th),
        distrib_deriv3(D, y, th),
        distrib_deriv4(D, y, th))
      key <- paste(law, d, hxl(eta), if (law == 1) hx(par) else "-", hxl(y),
                   if (law == 2) hx(par) else hx(0), sep = ",")
      for (ord in 0:4) {
        tuples <- if (ord == 0) list(integer(0)) else {
          g <- as.matrix(expand.grid(rep(list(seq_along(pn)), ord)))
          g <- g[apply(g, 1, function(r) all(diff(r) >= 0)), , drop = FALSE]
          lapply(seq_len(nrow(g)), function(r) unname(g[r, ]))
        }
        for (tu in tuples) {
          e <- tu[tu <= m] - 1
          K <- sum(tu > m)
          nm <- if (ord == 0) "0" else paste(pn[tu], collapse = "_")
          v <- lists[[ord + 1]][[nm]]
          if (is.null(v)) stop("missing ", nm)
          rows <- c(rows, paste(key, 0, 0, length(e), if (K) paste(rep(0, K), collapse = ";") else "-",
                                entry_of(e, m), hx(v[1]), hx(0), sep = ","))
        }
      }
      eh <- distrib_expected_hessian(D, y, th)
      for (a in seq_along(pn)) for (b in seq_len(a)) {
        v <- eh[[paste(pn[b], pn[a], sep = "_")]]
        if (is.null(v)) v <- eh[[paste(pn[a], pn[b], sep = "_")]]
        if (is.null(v)) stop("missing information ", a, " ", b)
        kind <- if (a <= m) 1 else if (b <= m) 2 else 3
        ent <- if (kind == 1) (a - 1) * a / 2 + (b - 1) else if (kind == 2) b - 1 else 0
        rows <- c(rows, paste(key, kind, 0, 0, "-", ent, hx(-v[1]), hx(0), sep = ","))
      }
    }
  }
}
writeLines(c("law,d,x,theta,y,size,kind,J,A,idx,entry,hi,lo", rows),
           file.path(out_dir, "distrib_simplex_s7.csv"))
cat("distrib_simplex_s7.csv:", length(rows), "rows\n")
