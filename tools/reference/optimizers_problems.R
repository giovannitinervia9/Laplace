# The battery of test problems of the optimizers, written in R from
# tests/optimizers/test_problems.hpp (same functions, starts, minimizers and
# added constants). Sourced by optimizers_optimizers7.R and by
# tools/validation/optimizers_r_timing.R. Each problem is a list with name, n,
# offset, x0, x_star, f_star, fn, gr, hess.

# A problem from f, g and H written without the offset.
battery_problem <- function(name, x0, x_star, f_star0, fn, gr, hess, offset = 0) {
  list(name = name, n = length(x0), offset = offset, x0 = x0, x_star = x_star,
       f_star = f_star0 + offset,
       fn = function(x) offset + fn(x), gr = gr, hess = hess)
}

# f = sum r_k^2 from residual functions returning list(r, grad, hess).
least_squares <- function(res) {
  list(
    fn = function(x) sum(vapply(res(x), function(t) t$r^2, 0)),
    gr = function(x) Reduce(`+`, lapply(res(x), function(t) 2 * t$r * t$grad)),
    hess = function(x) Reduce(`+`, lapply(res(x), function(t) 2 * (tcrossprod(t$grad) + t$r * t$hess)))
  )
}
residual <- function(r, grad, hess = matrix(0, length(grad), length(grad))) list(r = r, grad = grad, hess = hess)

extended_rosenbrock_problem <- function(n, offset = 0) {
  odd <- seq(1, n, by = 2); even <- odd + 1
  battery_problem(paste0("extended Rosenbrock n = ", n), rep(c(-1.2, 1), n / 2), rep(1, n), 0,
    fn = function(x) { a <- x[odd]; b <- x[even]; sum(100 * (b - a^2)^2 + (1 - a)^2) },
    gr = function(x) {
      a <- x[odd]; b <- x[even]; t <- b - a^2; g <- numeric(n)
      g[odd] <- -400 * a * t - 2 * (1 - a); g[even] <- 200 * t; g
    },
    hess = function(x) {
      a <- x[odd]; b <- x[even]; H <- matrix(0, n, n)
      H[cbind(odd, odd)] <- 1200 * a^2 - 400 * b + 2
      H[cbind(odd, even)] <- H[cbind(even, odd)] <- -400 * a
      H[cbind(even, even)] <- 200
      H
    }, offset = offset)
}

powell_singular_problem <- function(offset = 0) {
  s5 <- sqrt(5); s10 <- sqrt(10)
  ls <- least_squares(function(x) {
    t3 <- x[2] - 2 * x[3]; t4 <- x[1] - x[4]
    h3 <- matrix(0, 4, 4); h3[2, 2] <- 2; h3[2, 3] <- h3[3, 2] <- -4; h3[3, 3] <- 8
    h4 <- matrix(0, 4, 4); h4[1, 1] <- 2 * s10; h4[1, 4] <- h4[4, 1] <- -2 * s10; h4[4, 4] <- 2 * s10
    list(residual(x[1] + 10 * x[2], c(1, 10, 0, 0)),
         residual(s5 * (x[3] - x[4]), c(0, 0, s5, -s5)),
         residual(t3^2, c(0, 2 * t3, -4 * t3, 0), h3),
         residual(s10 * t4^2, c(2 * s10 * t4, 0, 0, -2 * s10 * t4), h4))
  })
  battery_problem("Powell singular", c(3, -1, 0, 1), c(0, 0, 0, 0), 0, ls$fn, ls$gr, ls$hess, offset)
}

beale_problem <- function(offset = 0) {
  yb <- c(1.5, 2.25, 2.625)
  ls <- least_squares(function(x) lapply(1:3, function(i) {
    p <- x[2]^i; p1 <- i * x[2]^(i - 1)
    h <- matrix(c(0, p1, p1, if (i > 1) x[1] * i * (i - 1) * x[2]^(i - 2) else 0), 2, 2)
    residual(yb[i] - x[1] * (1 - p), c(-(1 - p), x[1] * p1), h)
  }))
  battery_problem("Beale", c(1, 1), c(3, 0.5), 0, ls$fn, ls$gr, ls$hess, offset)
}

brown_badly_scaled_problem <- function(offset = 0) {
  ls <- least_squares(function(x) list(
    residual(x[1] - 1e6, c(1, 0)),
    residual(x[2] - 2e-6, c(0, 1)),
    residual(x[1] * x[2] - 2, c(x[2], x[1]), matrix(c(0, 1, 1, 0), 2, 2))))
  battery_problem("Brown badly scaled", c(1, 1), c(1e6, 2e-6), 0, ls$fn, ls$gr, ls$hess, offset)
}

helical_valley_problem <- function(offset = 0) {
  twopi <- 2 * pi
  ls <- least_squares(function(x) {
    x1 <- x[1]; x2 <- x[2]; x3 <- x[3]; r2 <- x1^2 + x2^2; rho <- sqrt(r2)
    theta <- if (x1 > 0) atan(x2 / x1) / twopi else if (x1 < 0) atan(x2 / x1) / twopi + 0.5
             else if (x2 >= 0) 0.25 else -0.25
    r4 <- twopi * r2^2; r3 <- rho * r2
    h1 <- matrix(0, 3, 3)
    h1[1, 1] <- -100 * (2 * x1 * x2 / r4)
    h1[1, 2] <- h1[2, 1] <- -100 * ((x2^2 - x1^2) / r4)
    h1[2, 2] <- -100 * (-2 * x1 * x2 / r4)
    h2 <- matrix(0, 3, 3)
    h2[1, 1] <- 10 * x2^2 / r3; h2[1, 2] <- h2[2, 1] <- -10 * x1 * x2 / r3; h2[2, 2] <- 10 * x1^2 / r3
    list(residual(10 * (x3 - 10 * theta), c(-100 * (-x2 / (twopi * r2)), -100 * (x1 / (twopi * r2)), 10), h1),
         residual(10 * (rho - 1), c(10 * x1 / rho, 10 * x2 / rho, 0), h2),
         residual(x3, c(0, 0, 1)))
  })
  battery_problem("helical valley", c(-1, 0, 0), c(1, 0, 0), 0, ls$fn, ls$gr, ls$hess, offset)
}

wood_problem <- function(offset = 0) {
  s90 <- sqrt(90); s10 <- sqrt(10)
  ls <- least_squares(function(x) {
    h1 <- matrix(0, 4, 4); h1[1, 1] <- -20
    h3 <- matrix(0, 4, 4); h3[3, 3] <- -2 * s90
    list(residual(10 * (x[2] - x[1]^2), c(-20 * x[1], 10, 0, 0), h1),
         residual(1 - x[1], c(-1, 0, 0, 0)),
         residual(s90 * (x[4] - x[3]^2), c(0, 0, -2 * s90 * x[3], s90), h3),
         residual(1 - x[3], c(0, 0, -1, 0)),
         residual(s10 * (x[2] + x[4] - 2), c(0, s10, 0, s10)),
         residual((x[2] - x[4]) / s10, c(0, 1 / s10, 0, -1 / s10)))
  })
  battery_problem("Wood", c(-3, -1, -3, -1), c(1, 1, 1, 1), 0, ls$fn, ls$gr, ls$hess, offset)
}

# The 64-bit LCG of test_uniform (Knuth's MMIX constants), in 16-bit limbs held
# in doubles (every partial sum below 2^53, so exact); returns the top 53 bits
# times 2^-53, bit for bit the C++ values.
test_uniform <- function(seed) {
  limbs <- function(hex) {
    d <- strtoi(substring(hex, c(1, 5, 9, 13), c(4, 8, 12, 16)), 16L)
    rev(as.numeric(d))  # least significant first
  }
  a <- limbs("5851F42D4C957F2D"); c <- limbs("14057B7EF767814F")
  s <- c(seed %% 65536, (seed %/% 65536) %% 65536, 0, 0)
  function() {
    r <- numeric(4); carry <- 0
    for (k in 1:4) {
      acc <- carry + c[k]
      for (i in 1:k) acc <- acc + s[i] * a[k - i + 1]
      r[k] <- acc %% 65536; carry <- acc %/% 65536
    }
    s <<- r
    (s[4] * 2^37 + s[3] * 2^21 + s[2] * 2^5 + s[1] %/% 2^11) * 2^-53
  }
}

ill_conditioned_quadratic_problem <- function(n, kappa, offset = 0) {
  u <- test_uniform(12345 + n)
  v <- vapply(seq_len(n), function(i) u() - 0.5, 0)
  lambda <- if (n > 1) kappa^((seq_len(n) - 1) / (n - 1)) else 1
  Q <- diag(n) - 2 * tcrossprod(v) / sum(v^2)
  A <- Q %*% diag(lambda, n) %*% Q
  xs <- seq_len(n)
  battery_problem(sprintf("quadratic n = %d, kappa = %g", n, kappa), rep(0, n), xs, 0,
    fn = function(x) 0.5 * sum((x - xs) * (A %*% (x - xs))),
    gr = function(x) as.numeric(A %*% (x - xs)),
    hess = function(x) A, offset = offset)
}

saddle_problem <- function(offset = 0) {
  battery_problem("saddle", c(1, 0), c(0, 1), -0.25,
    fn = function(x) x[1]^2 + 0.25 * x[2]^4 - 0.5 * x[2]^2,
    gr = function(x) c(2 * x[1], x[2]^3 - x[2]),
    hess = function(x) matrix(c(2, 0, 0, 3 * x[2]^2 - 1), 2, 2), offset = offset)
}

noisy_problem <- function(n) {
  battery_problem("noisy", rep(1, n), rep(0, n), NaN,
    fn = function(x) 100 * sin(1e20 * x[1]) + sum(x^2),
    gr = function(x) { g <- 2 * x; g[1] <- g[1] + 100 * 1e20 * cos(1e20 * x[1]); g },
    hess = function(x) { H <- diag(2, n); H[1, 1] <- H[1, 1] - 100 * 1e40 * sin(1e20 * x[1]); H })
}

exploded_cosh_problem <- function(n, offset = 0) {
  battery_problem("exploded cosh", rep(700, n), rep(0, n), n,
    fn = function(x) sum(cosh(x)), gr = function(x) sinh(x),
    hess = function(x) diag(cosh(x), n), offset = offset)
}

# The battery: every problem at every added constant (the noisy one at 0 only).
# Sizes as in benchmarks/optimizers_foundations.cpp and the doctest cases.
battery <- function(offsets = c(0, 1, 1e3, 1e6)) {
  out <- list()
  for (o in offsets) {
    out <- c(out, list(
      extended_rosenbrock_problem(2, o), extended_rosenbrock_problem(10, o),
      extended_rosenbrock_problem(100, o), powell_singular_problem(o), beale_problem(o),
      brown_badly_scaled_problem(o), helical_valley_problem(o), wood_problem(o),
      ill_conditioned_quadratic_problem(10, 1e2, o), ill_conditioned_quadratic_problem(10, 1e6, o),
      ill_conditioned_quadratic_problem(10, 1e12, o), saddle_problem(o), exploded_cosh_problem(3, o)))
  }
  c(out, list(noisy_problem(2)))
}

# Self-check of the transcription: gradients and Hessians against central
# differences at a generic point, as check_problem_derivatives does in C++.
check_battery_derivatives <- function(problems = battery(0)) {
  for (p in problems) {
    if (p$name %in% c("noisy", "exploded cosh")) next
    x <- p$x0 + 0.1 * sin(seq_len(p$n))
    h <- 1e-6
    gd <- vapply(seq_len(p$n), function(i) { e <- replace(numeric(p$n), i, h); (p$fn(x + e) - p$fn(x - e)) / (2 * h) }, 0)
    Hd <- sapply(seq_len(p$n), function(i) { e <- replace(numeric(p$n), i, h); (p$gr(x + e) - p$gr(x - e)) / (2 * h) })
    g <- p$gr(x); H <- p$hess(x)
    eg <- max(abs(g - gd)) / max(1, abs(g)); eH <- max(abs(H - Hd)) / max(1, abs(H))
    # 1e-4: Brown badly scaled has f near 1e12, so the differences lose about 1e-5
    stopifnot(eg < 1e-4, eH < 1e-4, isTRUE(all.equal(H, t(H))))
    stopifnot(abs(p$fn(p$x_star) - p$f_star) <= 1e-12 * max(1, abs(p$f_star)) || p$name == "saddle")
  }
  invisible(TRUE)
}
