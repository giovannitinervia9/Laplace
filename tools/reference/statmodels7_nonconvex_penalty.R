# Frozen statmodels7 results for Laplace's scad_penalty and mcp_penalty
# (include/laplace/penalty/nonconvex_penalty.hpp), from penalties7 0.34.0:
# penalty_value, penalty_gradient, the diagonal of penalty_hessian and
# penalty_prox of scad_penalty() and mcp_penalty() at unit curvature, without a
# map. statmodels7 is used as an oracle only: this script calls its exported
# functions and writes their results; no code of it enters Laplace.
#
# Usage (from the repository root, with statmodels7 installed):
#   Rscript --vanilla tools/reference/statmodels7_nonconvex_penalty.R
#
# File written (hex doubles as by sprintf("%a"); lo is always 0, the values are
# statmodels7's doubles): tests/data/reference/penalty_nonconvex_s7.csv,
#   law,kind,lambda,shape,t,x,hi,lo
# law 0 MCP (shape gamma), 1 SCAD (shape a); kind 0 the value q(|x|), 1 the
# gradient, 2 the second derivative (the diagonal of the Hessian), 3 the prox of
# x with step t (t = 0 for kinds 0-2). One coordinate per row (n_coef = 1).
#
# Points: lambda 0.7 and 2; gamma 1.5 and 3, a 2.2 and 3.7; x = s u lambda with
# s = +-1 and u in 0, 0.3, 1, 1.5, the shape, the shape + 0.5, 5 (the knots
# u = 1 (SCAD) and u = shape exactly). The prox at t = 0.1, 0.5 and 0.99 times
# the bound (gamma, a - 1), on x = s u lambda with u in 0, 0.05, 0.5, 1.2,
# 1.05 (1 + t), between the knots, the shape, 1.5 shape, 6.

suppressPackageStartupMessages(library(penalties7))
stopifnot(packageVersion("penalties7") == "0.34.0")

hx <- function(x) sprintf("%a", x)
rows <- character()
add <- function(law, kind, lam, shape, t, x, val) {
  rows <<- c(rows, paste(law, kind, hx(lam), hx(shape), hx(t), hx(x), hx(val), hx(0), sep = ","))
}

for (law in 0:1) {
  shapes <- if (law == 0) c(1.5, 3) else c(2.2, 3.7)
  for (lam in c(0.7, 2)) for (shape in shapes) {
    th <- if (law == 0) list(lambda = lam, gamma = shape) else list(lambda = lam, a = shape)
    p <- if (law == 0) mcp_penalty(n_coef = 1L) else scad_penalty(n_coef = 1L)
    for (s in c(-1, 1)) for (u in c(0, 0.3, 1, 1.5, shape, shape + 0.5, 5)) {
      x <- s * u * lam
      if (s < 0 && u == 0) next
      add(law, 0, lam, shape, 0, x, penalty_value(p, x, th))
      add(law, 1, lam, shape, 0, x, penalty_gradient(p, x, th)[1])
      add(law, 2, lam, shape, 0, x, as.matrix(penalty_hessian(p, x, th))[1, 1])
    }
    bound <- if (law == 0) shape else shape - 1
    for (t in c(0.1, 0.5, 0.99 * bound)) {
      mid <- (1 + t + shape) / 2
      for (s in c(-1, 1)) for (u in c(0, 0.05, 0.5, 1.2, 1.05 * (1 + t), mid, shape, 1.5 * shape, 6)) {
        x <- s * u * lam
        add(law, 3, lam, shape, t, x, penalty_prox(p, x, t, th))
      }
    }
  }
}

out <- file.path("tests", "data", "reference", "penalty_nonconvex_s7.csv")
writeLines(c("law,kind,lambda,shape,t,x,hi,lo", rows), out)
cat(length(rows), "rows written to", out, "\n")
