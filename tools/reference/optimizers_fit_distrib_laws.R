# Frozen references for the fit_distrib bench on the univariate laws added after
# the first bench (note/optimizers.md, decision 8; the first sixteen are in
# optimizers_fit_distrib.R, whose files this script does not touch). For each
# law it simulates data in R with a fixed seed at n = 1e3 and 1e5, writes them,
# and writes the maximum likelihood estimate computed in R from a log-density
# that is independent of Laplace's code:
#   - distributions7's distrib_pdf for the laws statmodels7 has, together with
#     statmodels7's own fit_distrib() estimate (default settings);
#   - gamlss.dist's dPE for the power exponential, whose (mu, sigma, nu) is
#     Laplace's (mu, sigma, p), sigma the standard deviation;
#   - for the properized MCP and SCAD, the definition of the header with its
#     normalizing constant by integrate() rather than the closed form.
# The MLE is reached by optim (BFGS in free coordinates, reltol 1e-15) and
# polished by cycles of one-dimensional Brent searches (optimize, tol 1e-13 in
# the free coordinate) until a cycle gains less than 1e-11 in the
# log-likelihood; Brent needs no derivative, so a kink at the MLE (the location
# of enet, mcp and scad) is reached as well as a smooth optimum. The score is
# reported for the smooth parameters only (central differences).
#
# Run from the repository root:
#   Rscript --vanilla tools/reference/optimizers_fit_distrib_laws.R
#
# Output in tests/data/reference/optimizers/fit_distrib/:
#   <case>_n1000.csv     the data at n = 1e3, one column y as hex doubles (C99 %a)
#   <case>_n100000.f64   the data at n = 1e5, raw little-endian IEEE doubles
#   fit_distrib_laws.csv one row per (case, n, parameter)

suppressMessages(library(statmodels7))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

out_dir <- "tests/data/reference/optimizers/fit_distrib"
sizes <- c(1e3, 1e5)

# --- the properized MCP and SCAD (include/laplace/distrib/mcp.hpp, scad.hpp)
mcp_q <- function(u, t) {
  k <- t$gamma * t$lambda
  ifelse(u <= k, t$lambda * u - u^2 / (2 * t$gamma), t$gamma * t$lambda^2 / 2 + (u - k)^2 / (2 * t$tail_sigma2))
}
scad_q <- function(u, t) {
  l <- t$lambda; a <- t$a; c <- a - 1
  ifelse(u <= l, l * u,
         ifelse(u <= a * l, l * u - (u - l)^2 / (2 * c), (a + 1) * l^2 / 2 + (u - a * l)^2 / (2 * t$tail_sigma2)))
}
# Z = 2 int_0^inf exp(-q(u)) du, piece by piece (the integrand is smooth on each)
log_z <- function(q, knots, t) {
  e <- c(0, knots, Inf)
  s <- 0
  for (i in seq_len(length(e) - 1))
    s <- s + integrate(function(u) exp(-q(u, t)), e[i], e[i + 1], rel.tol = 1e-12, abs.tol = 0)$value
  log(2 * s)
}
mcp_logpdf <- function(y, t) -mcp_q(abs(y - t$mu), t) - log_z(mcp_q, t$gamma * t$lambda, t)
scad_logpdf <- function(y, t) -scad_q(abs(y - t$mu), t) - log_z(scad_q, c(t$lambda, t$a * t$lambda), t)
# draws: a side, then the inner region (rejection from the uniform, exp(-q) being
# decreasing there with maximum 1) or the Gaussian tail, by their masses
rng_properized <- function(n, t, q, inner_end) {
  zi <- integrate(function(u) exp(-q(u, t)), 0, inner_end, rel.tol = 1e-12)$value
  zt <- exp(-q(inner_end, t)) * sqrt(pi * t$tail_sigma2 / 2)
  u <- numeric(n)
  inner <- runif(n) < zi / (zi + zt)
  for (i in which(inner)) repeat {
    v <- runif(1, 0, inner_end)
    if (runif(1) <= exp(-q(v, t))) { u[i] <- v; break }
  }
  u[!inner] <- inner_end + abs(rnorm(sum(!inner), 0, sqrt(t$tail_sigma2)))
  t$mu + ifelse(runif(n) < 0.5, -u, u)
}

# free coordinate maps by domain: identity, log, logit, log(x - lower)
dom_real <- list(to = identity, from = identity)
dom_pos <- list(to = log, from = exp)
dom_unit <- list(to = qlogis, from = plogis)
dom_above <- function(a) list(to = function(x) log(x - a), from = function(u) a + exp(u))

d7_case <- function(name, d7, theta, domains, to_d7 = identity, from_d7 = identity, size = NULL) {
  dist <- function() if (is.null(size)) get(paste0(d7, "_distrib"))() else get(paste0(d7, "_distrib"))(size = size)
  list(name = name, reference = paste0("distributions7 ", d7), theta = theta, domains = domains,
       rng = function(n, t) as.numeric(distrib_rng(dist(), n, to_d7(t))),
       logpdf = function(y, t) log(distrib_pdf(dist(), y, to_d7(t))),
       statmodels7 = function(y) { set.seed(1); f <- fit_distrib(dist(), y); list(est = from_d7(as.list(f@coefficients)),
                                                                                     converged = f@converged, loglik = f@loglik) })
}
pe_case <- function(name, theta) list(
  name = name, reference = "gamlss.dist dPE", theta = theta, domains = list(dom_real, dom_pos, dom_pos),
  rng = function(n, t) gamlss.dist::rPE(n, t$mu, t$sigma, t$p),
  logpdf = function(y, t) gamlss.dist::dPE(y, t$mu, t$sigma, t$p, log = TRUE), statmodels7 = NULL)

vm_to <- function(t) list(mu = t$mu, rho = exp(-t$s^2 / 2))
vm_from <- function(t) list(mu = t$mu, s = sqrt(-2 * log(t$rho)))
cases <- list(
  d7_case("student_t1", "student_t1", list(mu = 1, sigma = 2, nu = 4), list(dom_real, dom_pos, dom_pos)),
  d7_case("student_t2", "student_t2", list(mu = 1, sigma = 2, nu = 5), list(dom_real, dom_pos, dom_above(2))),
  d7_case("skewnormal", "skewnormal2", list(mu = 1, sigma = 2, gamma = 0.5), list(dom_real, dom_pos, NULL),
          to_d7 = function(t) list(mu = t$mu, sigma = t$sigma, gamma1 = t$gamma),
          from_d7 = function(t) list(mu = t$mu, sigma = t$sigma, gamma = t$gamma1)),
  d7_case("skewt", "skewt", list(mu = 1, sigma = 2, alpha = 2, nu = 5), list(dom_real, dom_pos, dom_real, dom_pos)),
  d7_case("pseudohuber", "pseudohuber2", list(mu = 1, sigma = 2, nu = 2), list(dom_real, dom_pos, dom_pos)),
  d7_case("weibull", "weibull3", list(m = 2, sigma = 1.5), list(dom_pos, dom_pos),
          to_d7 = function(t) list(mean = t$m, sigma = t$sigma), from_d7 = function(t) list(m = t$mean, sigma = t$sigma)),
  d7_case("vonmises", "vonmises2", list(mu = 0.5, s = 0.7), list(dom_real, dom_pos), to_d7 = vm_to, from_d7 = vm_from),
  d7_case("beta", "beta1", list(mu = 0.3, phi = 5), list(dom_unit, dom_pos)),
  d7_case("betabinom", "betabinom1", list(mu = 0.3, sigma = 0.5), list(dom_unit, dom_pos), size = 10),
  d7_case("negbin1", "negbin1", list(mu = 3, theta = 0.8), list(dom_pos, dom_pos)),
  d7_case("negbin2", "negbin2", list(mu = 3, theta = 2), list(dom_pos, dom_pos)),
  d7_case("pig", "pig2", list(mu = 3, alpha = 2), list(dom_pos, dom_pos)),
  d7_case("enet", "enet", list(mu = 1, lambda = 1, alpha = 0.5), list(dom_real, dom_pos, dom_unit)),
  pe_case("powerexp_p1.5", list(mu = 1, sigma = 2, p = 1.5)),
  pe_case("powerexp_p3", list(mu = 1, sigma = 2, p = 3)),
  list(name = "mcp", reference = "definition by integrate()", theta = list(mu = 1, lambda = 1, gamma = 3, tail_sigma2 = 1),
       domains = list(dom_real, dom_pos, dom_pos, dom_pos),
       rng = function(n, t) rng_properized(n, t, mcp_q, t$gamma * t$lambda), logpdf = mcp_logpdf, statmodels7 = NULL),
  list(name = "scad", reference = "definition by integrate()", theta = list(mu = 1, lambda = 1, a = 3.7, tail_sigma2 = 1),
       domains = list(dom_real, dom_pos, dom_above(1), dom_pos),
       rng = function(n, t) rng_properized(n, t, scad_q, t$a * t$lambda), logpdf = scad_logpdf, statmodels7 = NULL)
)
# skewnormal's gamma lives in (-g, g), g the largest skewness of the family
g_max <- 0.5 * (4 - pi) * (2 / (pi - 2))^1.5
cases[[3]]$domains[[3]] <- list(to = function(x) qlogis((x + g_max) / (2 * g_max)),
                                from = function(u) 2 * g_max * plogis(u) - g_max)

# the MLE: optim then Brent cycles, in free coordinates
mle_of <- function(cs, y, start) {
  nm <- names(cs$theta)
  th <- function(u) setNames(Map(function(d, v) d$from(v), cs$domains, u), nm)
  # a trial point where the density cannot be evaluated (integrate() fails at extreme
  # parameters) is refused like a point outside the domain
  nll <- function(u) {
    v <- tryCatch(-sum(cs$logpdf(y, th(u))), error = function(e) NaN)
    if (is.finite(v)) v else 1e300
  }
  u <- mapply(function(d, v) d$to(v), cs$domains, start)
  o <- optim(u, nll, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  u <- o$par
  f <- nll(u)
  for (cycle in 1:200) {
    f0 <- f
    for (j in seq_along(u)) {
      g <- function(x) { v <- u; v[j] <- x; nll(v) }
      r <- optimize(g, c(u[j] - 0.5, u[j] + 0.5), tol = 1e-13)
      if (r$objective < f) { u[j] <- r$minimum; f <- r$objective }
    }
    if (f0 - f < 1e-11) break
  }
  list(theta = unlist(th(u)), loglik = -f, cycles = cycle, u = u, nll = nll)
}
# max |d(-l)/du_j| / n over the smooth coordinates, by central differences
score_of <- function(m, smooth, n) {
  s <- 0
  for (j in which(smooth)) {
    h <- 1e-5 * max(1, abs(m$u[j]))
    up <- m$u; dn <- m$u; up[j] <- up[j] + h; dn[j] <- dn[j] - h
    s <- max(s, abs(m$nll(up) - m$nll(dn)) / (2 * h) / n)
  }
  s
}
kinked <- c(enet = 1, mcp = 1, scad = 1)

if (!isTRUE(get0("define_cases_only"))) {
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
rows <- list()
for (k in seq_along(cases)) {
  cs <- cases[[k]]
  for (j in seq_along(sizes)) {
    n <- sizes[j]
    seed <- 20261119 + 100 * k + j
    set.seed(seed)
    y <- cs$rng(n, cs$theta)
    stem <- file.path(out_dir, sprintf("%s_n%d", cs$name, as.integer(n)))
    if (n <= 1e3) {
      writeLines(c("y", sprintf("%a", y)), paste0(stem, ".csv"))
      y <- as.numeric(readLines(paste0(stem, ".csv"))[-1])
    } else {
      writeBin(y, paste0(stem, ".f64"), size = 8, endian = "little")
      y <- readBin(paste0(stem, ".f64"), "double", n = n, size = 8, endian = "little")
    }
    sm <- list(est = rep(NA_real_, length(cs$theta)), converged = NA, loglik = NA_real_)
    if (!is.null(cs$statmodels7)) {
      r <- tryCatch(cs$statmodels7(y), error = function(e) NULL)
      if (!is.null(r)) sm <- list(est = unlist(r$est)[names(cs$theta)], converged = r$converged, loglik = r$loglik)
    }
    # from the true values, and from statmodels7's estimate where it has one; the better
    # of the two (statmodels7 stopped at alpha = 2.2e-308 on enet at n = 1e3, where the
    # Brent cycles cannot leave a logit of -708, 10 below the interior maximum)
    t0 <- Sys.time()
    m <- mle_of(cs, y, cs$theta)
    if (!is.null(cs$statmodels7) && all(is.finite(sm$est))) {
      m2 <- mle_of(cs, y, as.list(sm$est))
      if (m2$loglik > m$loglik) m <- m2
    }
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    smooth <- rep(TRUE, length(cs$theta))
    if (cs$name %in% names(kinked)) smooth[1] <- FALSE
    if (cs$name %in% c("mcp", "scad")) smooth[2:3] <- FALSE
    sc <- score_of(m, smooth, n)
    rows[[length(rows) + 1]] <- data.frame(
      case = cs$name, reference = cs$reference, n = as.integer(n), seed = seed, param = names(cs$theta),
      true_value = unlist(cs$theta), mle = sprintf("%.17g", m$theta), loglik_mle = sprintf("%.17g", m$loglik),
      statmodels7 = sprintf("%.17g", sm$est), loglik_statmodels7 = sprintf("%.17g", sm$loglik),
      converged_statmodels7 = sm$converged, score_smooth = sprintf("%.3g", sc), cycles = m$cycles)
    cat(sprintf("%-14s n = %6d  loglik %.12g  statmodels7 %.12g  score %.2g  cycles %d  %.1f s\n",
                cs$name, as.integer(n), m$loglik, sm$loglik, sc, m$cycles, elapsed))
  }
}
out <- do.call(rbind, rows)
con <- file(file.path(out_dir, "fit_distrib_laws.csv"), "w")
writeLines(c(
  sprintf("# R %s, statmodels7 %s (distributions7 %s), gamlss.dist %s; written by tools/reference/optimizers_fit_distrib_laws.R.",
          getRversion(), packageVersion("statmodels7"), packageVersion("distributions7"), packageVersion("gamlss.dist")),
  "# mle = maximum likelihood estimate of R's log-density (optim BFGS, then Brent cycles until a cycle gains < 1e-11);",
  "# statmodels7 = fit_distrib() with its defaults (set.seed(1)), NA where statmodels7 has no such law;",
  "# score_smooth = max |d(-l)/du| / n over the smooth parameters at the MLE (central differences, free coordinates)."), con)
write.csv(out, con, row.names = FALSE)
close(con)
}
