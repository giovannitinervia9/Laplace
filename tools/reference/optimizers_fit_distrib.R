# Frozen references for the fit_distrib bench of the optimizers
# (note/optimizers.md, decision 8). For each of the 16 parametrizations of
# include/laplace/distrib it simulates data in R with a fixed seed at n = 1e3
# and 1e5, writes the data, fits them with statmodels7's fit_distrib()
# (default settings) and writes the estimates, standard errors, log-likelihood,
# iterations, counts and convergence, together with the closed-form MLEs where
# they exist (an exact oracle).
#
# Run from the repository root: Rscript --vanilla tools/reference/optimizers_fit_distrib.R
#
# Output in tests/data/reference/optimizers/fit_distrib/:
#   <distrib>_n1000.csv     the data at n = 1e3, one column y as hex doubles (C99 %a)
#   <distrib>_n100000.f64   the data at n = 1e5, raw little-endian IEEE doubles
#   fit_distrib.csv         one row per (distrib, n, parameter)
#
# Each Laplace distrib is matched to the distributions7 constructor with the
# same parameters; the log-densities agree to 4.4e-16 at the check points of
# check_parametrizations() below, which the script runs first.

suppressMessages(library(statmodels7))
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

out_dir <- "tests/data/reference/optimizers/fit_distrib"
sizes <- c(1e3, 1e5)

# Inverse Gaussian by Michael, Schucany and Haas (1976), mean mu, dispersion phi.
rinvgauss_msh <- function(n, mu, phi) {
  lambda <- 1 / phi
  v <- rnorm(n)^2
  x <- mu + mu^2 * v / (2 * lambda) - mu / (2 * lambda) * sqrt(4 * mu * lambda * v + mu^2 * v^2)
  ifelse(runif(n) <= mu / (mu + x), x, mu^2 / x)
}

# Laplace name, distributions7 constructor, true parameters (names as in
# distributions7, meanings as in the Laplace header), simulator, log-density
# written from the Laplace header, closed-form MLE (or NULL).
gaussian_mle <- function(y) c(mean(y), sqrt(mean((y - mean(y))^2)))
lognormal_mle <- function(y) { l <- log(y); m <- mean(l); c(m, mean((l - m)^2)) }
laplace_mle <- function(y) { m <- median(y); c(m, mean(abs(y - m))) }
cases <- list(
  list(name = "cauchy", d7 = "cauchy", theta = list(mu = 1, sigma = 2),
       rng = function(n, t) rcauchy(n, t$mu, t$sigma),
       logpdf = function(y, t) dcauchy(y, t$mu, t$sigma, log = TRUE), mle = NULL),
  list(name = "chisq", d7 = "chisq", theta = list(mu = 4.5),
       rng = function(n, t) rchisq(n, t$mu),
       logpdf = function(y, t) dchisq(y, t$mu, log = TRUE), mle = NULL),
  list(name = "exponential", d7 = "exponential", theta = list(mu = 2),
       rng = function(n, t) rexp(n, 1 / t$mu),
       logpdf = function(y, t) dexp(y, 1 / t$mu, log = TRUE), mle = function(y) mean(y)),
  list(name = "gamma1", d7 = "gamma1", theta = list(mu = 2, phi = 0.4),
       rng = function(n, t) rgamma(n, shape = 1 / t$phi, scale = t$mu * t$phi),
       logpdf = function(y, t) dgamma(y, shape = 1 / t$phi, scale = t$mu * t$phi, log = TRUE), mle = NULL),
  list(name = "gamma2", d7 = "gamma2", theta = list(mu = 2, sigma2 = 1.5),
       rng = function(n, t) rgamma(n, shape = t$mu^2 / t$sigma2, scale = t$sigma2 / t$mu),
       logpdf = function(y, t) dgamma(y, shape = t$mu^2 / t$sigma2, scale = t$sigma2 / t$mu, log = TRUE), mle = NULL),
  list(name = "gaussian1", d7 = "gaussian1", theta = list(mu = 1, sigma = 2),
       rng = function(n, t) rnorm(n, t$mu, t$sigma),
       logpdf = function(y, t) dnorm(y, t$mu, t$sigma, log = TRUE), mle = gaussian_mle),
  list(name = "gaussian2", d7 = "gaussian2", theta = list(mu = 1, sigma2 = 4),
       rng = function(n, t) rnorm(n, t$mu, sqrt(t$sigma2)),
       logpdf = function(y, t) dnorm(y, t$mu, sqrt(t$sigma2), log = TRUE),
       mle = function(y) { m <- gaussian_mle(y); c(m[1], m[2]^2) }),
  list(name = "gaussian3", d7 = "gaussian3", theta = list(mu = 1, tau = 0.25),
       rng = function(n, t) rnorm(n, t$mu, 1 / sqrt(t$tau)),
       logpdf = function(y, t) dnorm(y, t$mu, 1 / sqrt(t$tau), log = TRUE),
       mle = function(y) { m <- gaussian_mle(y); c(m[1], 1 / m[2]^2) }),
  list(name = "gumbel", d7 = "gumbel", theta = list(mu = 1, sigma = 2),
       rng = function(n, t) t$mu - t$sigma * log(rexp(n)),
       logpdf = function(y, t) { z <- (y - t$mu) / t$sigma; -log(t$sigma) - z - exp(-z) }, mle = NULL),
  list(name = "invgauss", d7 = "invgauss1", theta = list(mu = 1.5, phi = 0.5),
       rng = function(n, t) rinvgauss_msh(n, t$mu, t$phi),
       logpdf = function(y, t) -0.5 * log(2 * pi * t$phi * y^3) - (y - t$mu)^2 / (2 * t$phi * t$mu^2 * y),
       mle = function(y) c(mean(y), mean(1 / y - 1 / mean(y)))),
  list(name = "laplace1", d7 = "laplace", theta = list(mu = 1, sigma = 2),
       rng = function(n, t) t$mu + t$sigma * (rexp(n) - rexp(n)),
       logpdf = function(y, t) -log(2 * t$sigma) - abs(y - t$mu) / t$sigma, mle = laplace_mle),
  list(name = "laplace2", d7 = "laplace2", theta = list(mu = 1, lambda = 0.5),
       rng = function(n, t) t$mu + (rexp(n) - rexp(n)) / t$lambda,
       logpdf = function(y, t) log(t$lambda / 2) - t$lambda * abs(y - t$mu),
       mle = function(y) { m <- laplace_mle(y); c(m[1], 1 / m[2]) }),
  list(name = "logistic", d7 = "logistic", theta = list(mu = 1, sigma = 2),
       rng = function(n, t) rlogis(n, t$mu, t$sigma),
       logpdf = function(y, t) dlogis(y, t$mu, t$sigma, log = TRUE), mle = NULL),
  list(name = "lognormal1", d7 = "lognormal1", theta = list(mu = 0.5, sigma2 = 0.36),
       rng = function(n, t) rlnorm(n, t$mu, sqrt(t$sigma2)),
       logpdf = function(y, t) dlnorm(y, t$mu, sqrt(t$sigma2), log = TRUE), mle = lognormal_mle),
  list(name = "lognormal2", d7 = "lognormal2", theta = list(mean = 2, var = 1.5),
       rng = function(n, t) { s <- log(1 + t$var / t$mean^2); rlnorm(n, log(t$mean) - s / 2, sqrt(s)) },
       logpdf = function(y, t) { s <- log(1 + t$var / t$mean^2); dlnorm(y, log(t$mean) - s / 2, sqrt(s), log = TRUE) },
       mle = function(y) { m <- lognormal_mle(y); c(exp(m[1] + m[2] / 2), (exp(m[2]) - 1) * exp(2 * m[1] + m[2])) }),
  list(name = "poisson", d7 = "poisson", theta = list(mu = 3.5),
       rng = function(n, t) as.numeric(rpois(n, t$mu)),
       logpdf = function(y, t) dpois(y, t$mu, log = TRUE), mle = function(y) mean(y))
)

d7 <- function(cs) get(paste0(cs$d7, "_distrib"))()

# Laplace's log-density (written from its header) against distributions7's.
check_parametrizations <- function() {
  for (cs in cases) {
    y <- if (cs$name == "poisson") c(0, 2, 7) else c(0.3, 1.7, 4.2)
    a <- log(distributions7::distrib_pdf(d7(cs), y, cs$theta))
    err <- max(abs(a - cs$logpdf(y, cs$theta)))
    if (err > 1e-14) stop(cs$name, ": log-densities differ by ", err)
  }
}
check_parametrizations()

# Sourced with define_cases_only = TRUE (the timing script), it stops here.
if (!isTRUE(get0("define_cases_only"))) {
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
rows <- list()
for (k in seq_along(cases)) {
  cs <- cases[[k]]
  for (j in seq_along(sizes)) {
    n <- sizes[j]
    set.seed(20261115 + 100 * k + j)
    y <- cs$rng(n, cs$theta)
    stem <- file.path(out_dir, sprintf("%s_n%d", cs$name, as.integer(n)))
    if (n <= 1e3) {
      writeLines(c("y", sprintf("%a", y)), paste0(stem, ".csv"))
      y <- as.numeric(readLines(paste0(stem, ".csv"))[-1])  # what a reader gets back
    } else {
      writeBin(y, paste0(stem, ".f64"), size = 8, endian = "little")
      y <- readBin(paste0(stem, ".f64"), "double", n = n, size = 8, endian = "little")
    }
    set.seed(1)  # fit_distrib's starts after the first are random draws
    t0 <- Sys.time()
    f <- fit_distrib(d7(cs), y)
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    est <- f@coefficients
    mle <- if (is.null(cs$mle)) rep(NA_real_, length(est)) else cs$mle(y)
    ll_mle <- if (is.null(cs$mle)) NA_real_ else sum(cs$logpdf(y, setNames(as.list(mle), names(est))))
    rows[[length(rows) + 1]] <- data.frame(
      distrib = cs$name, distributions7 = cs$d7, n = as.integer(n), seed = 20261115 + 100 * k + j,
      param = names(est), true_value = unlist(cs$theta),
      estimate = sprintf("%.17g", est), se = sprintf("%.17g", f@se),
      closed_form_mle = sprintf("%.17g", mle),
      loglik = sprintf("%.17g", f@loglik), loglik_closed_form = sprintf("%.17g", ll_mle),
      iterations = f@iterations, converged = f@converged, method = f@method, criterion = f@criterion,
      f_count = f@counts[[1]], g_count = f@counts[[2]], h_count = f@counts[[3]],
      score = sprintf("%.6g", f@score))
    cat(sprintf("%-12s n = %6d  iterations %2d  converged %s  loglik %.10g  %.3f s\n",
                cs$name, as.integer(n), f@iterations, f@converged, f@loglik, elapsed))
  }
}
out <- do.call(rbind, rows)
con <- file(file.path(out_dir, "fit_distrib.csv"), "w")
writeLines(c(
  sprintf("# statmodels7 %s (distributions7 %s, optimizers7 %s, R %s): fit_distrib() with its defaults",
          packageVersion("statmodels7"), packageVersion("distributions7"), packageVersion("optimizers7"), getRversion()),
  "# (Fisher scoring, n_start = 5, set.seed(1) before each fit) on the data files beside this one;",
  "# written by tools/reference/optimizers_fit_distrib.R. closed_form_mle = exact MLE where one exists (NA otherwise);",
  "# for laplace1/laplace2 at even n the median is the midpoint of the two middle values (the likelihood is flat between them).",
  "# score = fit_distrib's max-norm of the score per observation (link scale) at the estimate, the quantity its",
  "# stopping rule tests (< 1e-6)."), con)
write.csv(out, con, row.names = FALSE)
close(con)
}
