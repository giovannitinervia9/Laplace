"""Reference values of the Mills ratio on the log scale, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/mills.py

Format as in normal.py: the argument, then hi, lo (hexadecimal), with hi + lo the
exact value as a double-double.

    log_mills              x    log M(x), M(x) = Phi(-x)/phi(x) the Mills ratio
    log_mills_d1 .. _d6    x    d^k log M(x) / dx^k, k = 1..6

Everything comes from the integral representation

    M(x) = int_0^inf exp(-x s - s^2/2) ds,
    m_k  = int_0^inf s^k exp(-x s - s^2/2) ds = (-1)^k M^(k)(x),

whose integrands are positive, so no quadrature here cancels. The raw moments
mu_k = m_k / m_0 of the density proportional to exp(-x s - s^2/2) on s > 0 give
the cumulants kappa_k by kappa_n = mu_n - sum_(i=1)^(n-1) C(n-1, i-1) kappa_i
mu_(n-i), and

    d^k log M / dx^k = (-1)^k kappa_k,     log M = log m_0.

For x > 0 the substitution s = u/x puts the mass of every integrand at u of
order one, which is what lets the same quadrature serve x = 10^-6 and x = 10^9;
for x <= 0 the integration is in s, whose mass sits between 0 and -x + 20. The
moment-to-cumulant step is the only one that subtracts: at large x the terms are
of size (k-1)! / x^k against a result of the same size, so it costs a factor k
and no more. Every value is computed at two precisions, 220 and 290 bits, and
the two must agree to 2^-140 relative or the script stops.

This route shares nothing with the C++ implementation, which takes the value
from erfcx and the derivatives from those of log Phi and from the downward ratio
recursion of normal.hpp.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261115
N = 2000
KMAX = 6
NAMES = ["log_mills"] + [f"log_mills_d{k}" for k in range(1, KMAX + 1)]


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def moments(x, prec):
    """log m_0 and the raw moments mu_1 .. mu_KMAX at the given precision."""
    with mpmath.workprec(prec):
        X = mpmath.mpf(x)
        if X >= 1:
            # The mass of exp(-x s - s^2/2) sits at s of order 1/x, so s = u/x
            # brings it to u of order one: m_k = x^-(k+1) n_k with
            # n_k = int_0^inf u^k exp(-u - u^2/(2 x^2)) du.
            w = 1 / (2 * X * X)
            pts = [0, 1, 5, 20, mpmath.inf]
            n = [mpmath.quad(lambda u, k=k: u**k * mpmath.exp(-u - w * u * u), pts)
                 for k in range(KMAX + 1)]
            log_m0 = mpmath.log(n[0]) - mpmath.log(X)
            mu = [n[k] / (n[0] * X**k) for k in range(KMAX + 1)]
        else:
            # Below x = 1 the mass sits between 0 and -x + 20, and the integration
            # is in s itself; the substitution above would push it to u ~ 1/x.
            peak = max(mpmath.mpf(0), -X)
            pts = sorted({mpmath.mpf(0), peak, peak + 1, peak + 5, peak + 20})
            pts = pts + [mpmath.inf]
            m = [mpmath.quad(lambda s, k=k: s**k * mpmath.exp(-X * s - s * s / 2), pts)
                 for k in range(KMAX + 1)]
            log_m0 = mpmath.log(m[0])
            mu = [m[k] / m[0] for k in range(KMAX + 1)]
        kap = [mpmath.mpf(0)] * (KMAX + 1)
        for n_ in range(1, KMAX + 1):
            s = mu[n_]
            binom = mpmath.mpf(1)  # C(n - 1, i - 1)
            for i in range(1, n_):
                s -= binom * kap[i] * mu[n_ - i]
                binom = binom * (n_ - i) / i
            kap[n_] = s
        # d^k log M / dx^k = (-1)^k kappa_k
        out = [log_m0] + [(-1) ** k * kap[k] for k in range(1, KMAX + 1)]
        return [+v for v in out]  # copies at this precision


def log_value(x):
    """log M(x) = log Phi(-x) + x^2/2 + log(2 pi)/2, the definition, for x far
    below zero, where the moment quadrature's integrand would reach exp(x^2/2).
    log1p(-ncdf(x)) keeps the digits of Phi(-x) as it approaches one."""
    with mpmath.workprec(300):
        X = mpmath.mpf(x)
        return X * X / 2 + mpmath.log(2 * mpmath.pi) / 2 + mpmath.log1p(-mpmath.ncdf(X))


def values(x):
    a = moments(x, 220)
    b = moments(x, 290)
    for k, (u, v) in enumerate(zip(a, b)):
        if v != 0 and abs(u - v) > mpmath.mpf(2) ** -140 * abs(v):
            raise RuntimeError(f"precision check failed at x = {x!r}, order {k}")
    return b


# Where the derivatives are taken. Below x = -5 the cumulants of order three and
# above fall like exp(-x^2/2) (the density is a normal barely touched by the
# truncation at s = 0), while the raw moments grow like |x|^k, so the
# moment-to-cumulant recursion cancels without limit and no working precision
# recovers it: at x = -40 the result is below the double range. The C++ route
# there is exactly d^k log Phi / dt^k at a positive argument, which has its own
# references in normal.py, so nothing is lost. The value log M = log m_0 has no
# cancellation anywhere and keeps the whole negative range.
DERIV_LOWER = -5.0


def grid(rng):
    xs = [log_uniform(rng, -20, 30) for _ in range(2 * N // 5)]          # x > 0, 1e-6 to 1e9
    xs += [rng.uniform(0.4, 2.5) for _ in range(N // 5)]                 # around the split at 1
    xs += [rng.uniform(DERIV_LOWER, 1.0) for _ in range(N // 5)]
    xs += [rng.uniform(25.0, 45.0) for _ in range(N // 5)]               # where Phi(-x) underflows
    xs += [0.0, 1.0, -1.0, 0.5, 2.0, 1e-8, 1e8, 30.0, 38.0, -5.0, 1e-300]
    return xs


def value_grid(rng):
    """x below DERIV_LOWER, for log_mills alone."""
    xs = [-log_uniform(rng, 2, 10) for _ in range(N // 10)]              # -4 down to -2048
    xs += [rng.uniform(-40.0, DERIV_LOWER) for _ in range(N // 10)]
    xs += [-37.0, -38.5, -40.0, -100.0, -1e3, -1e8]
    return [x for x in xs if x < DERIV_LOWER]


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{SEED}-mills")
    xs = grid(rng)
    rows = {name: [] for name in NAMES}
    for i, x in enumerate(xs):
        v = values(x)
        for name, val in zip(NAMES, v):
            rows[name].append((x, val))
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(xs)}", flush=True)
    for x in value_grid(rng):
        rows["log_mills"].append((x, log_value(x)))
    for name in NAMES:
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("x,hi,lo\n")
            for x, val in rows[name]:
                hi, lo = dd(val)
                fh.write(f"{float.hex(float(x))},{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows[name])} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
