# Frozen reference counts of optimizers7's gd() and bb() on the battery of
# test problems (note/optimizers.md, decision 10 and section 13). Runs gd()
# (Armijo, step 1 offered at every iteration) and bb() with each of its three
# variants ("alternate", "bb1", "bb2"; nonmonotone search with memory 10,
# alpha0 = 0.01), otherwise with their default settings, from the battery's
# starts at the added constants f* = 0, 1, 1e3, 1e6, and writes
# tests/data/reference/optimizers/optimizers7_gd_bb_counts.csv. The counts are a
# reference for comparison, not targets; the columns are those of
# optimizers7_counts.csv (written by optimizers_optimizers7.R), with
# method = "gd" or "bb_<variant>".
#
# Run from the repository root: Rscript --vanilla tools/reference/optimizers_optimizers7_gd_bb.R

suppressMessages(library(optimizers7))
source("tools/reference/optimizers_problems.R")
options(optimizers7.check_gradient = FALSE)  # the gradient check at the start would spend evaluations

builders <- list(
  gd = function(...) gd(...),
  bb_alternate = function(...) bb(variant = "alternate", ...),
  bb_bb1 = function(...) bb(variant = "bb1", ...),
  bb_bb2 = function(...) bb(variant = "bb2", ...))

run_one <- function(p, method, maxit = NULL) {
  opt <- if (is.null(maxit)) builders[[method]]() else builders[[method]](maxit = maxit)
  suppressWarnings(minimize(opt, p$fn, p$x0, gr = p$gr))
}

# the rule that fired, found as in optimizers_optimizers7.R: the previous
# iterate from a rerun with maxit = iterations - 1 (the runs are deterministic)
rules_fired <- function(p, method, r) {
  if (!identical(r@criterion_met, builders[[method]]()@criterion@label)) return(r@criterion_met)
  if (r@iterations < 1) return("at start")
  prev <- if (r@iterations == 1) list(value = p$fn(p$x0), par = p$x0)
          else { q <- run_one(p, method, maxit = r@iterations - 1); list(value = q@value, par = q@par) }
  state <- list(f_new = r@value, f_old = prev$value, x_new = r@par, x_old = prev$par,
                gradient = r@gradient)
  rules <- builders[[method]]()@criterion@criteria
  fired <- vapply(rules, function(cr) isTRUE(crit_met(cr, state)), NA)
  if (!any(fired)) return("criterion (no single rule)")
  paste(vapply(rules[fired], function(cr) cr@label, ""), collapse = " & ")
}

rows <- list()
for (p in battery()) {
  for (m in names(builders)) {
    r <- tryCatch(run_one(p, m), error = function(e) e)
    if (inherits(r, "error")) {
      rows[[length(rows) + 1]] <- data.frame(problem = p$name, n = p$n, offset = p$offset, method = m,
        iterations = NA, f_count = NA, g_count = NA, h_count = NA, value = NA, value_minus_f_star = NA,
        gradient_max_norm = NA, distance_max = NA, converged = NA,
        rule_fired = paste("error:", conditionMessage(r)), point = NA)
      next
    }
    g <- if (is.null(r@gradient)) p$gr(r@par) else r@gradient
    rows[[length(rows) + 1]] <- data.frame(
      problem = p$name, n = p$n, offset = p$offset, method = m,
      iterations = r@iterations, f_count = r@counts[["f"]], g_count = r@counts[["g"]],
      h_count = r@counts[["h"]],
      value = sprintf("%.17g", r@value), value_minus_f_star = sprintf("%.6g", r@value - p$f_star),
      gradient_max_norm = sprintf("%.6g", max(abs(g))),
      distance_max = sprintf("%.6g", if (p$name == "saddle") min(max(abs(r@par - c(0, 1))), max(abs(r@par - c(0, -1))))
                                    else max(abs(r@par - p$x_star))),
      converged = r@converged, rule_fired = rules_fired(p, m, r),
      point = paste(sprintf("%.17g", r@par), collapse = " "))
  }
}
out <- do.call(rbind, rows)
dir.create("tests/data/reference/optimizers", recursive = TRUE, showWarnings = FALSE)
con <- file("tests/data/reference/optimizers/optimizers7_gd_bb_counts.csv", "w")
writeLines(c(
  sprintf("# optimizers7 %s (R %s) gd() and bb() (variants alternate, bb1, bb2) on the battery of tests/optimizers/test_problems.hpp, default settings;",
          packageVersion("optimizers7"), getRversion()),
  "# written by tools/reference/optimizers_optimizers7_gd_bb.R. distance_max = max |x - x*| (saddle: nearest of (0, +-1));",
  "# point = final x, space-separated. value and point at full precision; the rest at 6 digits."), con)
write.csv(out, con, row.names = FALSE)
close(con)
cat(nrow(out), "runs written\n")
print(table(out$method, out$converged, useNA = "ifany"))
