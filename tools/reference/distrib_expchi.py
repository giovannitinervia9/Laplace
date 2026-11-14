"""Reference values of the derivatives of log F and log(1 - F) for exponential_distrib (mean mu)
and chisq_distrib (degrees of freedom mu), orders 1 to 6.

Usage (from the repository root):  python -I tools/reference/distrib_expchi.py

Files tests/data/reference/distrib_exponential.csv and tests/data/reference/distrib_chisq.csv,
one row per value:
    upper (0 log F, 1 log(1 - F)), y, K, hi, lo, mu
with y and mu in hexadecimal (the double values used by the tests), K the order in mu, and hi
and lo the high and low parts of the mpmath value.

Method: mpmath.diff (high-precision finite differences on the analytic function, with its own
step control) of the DEFINITION, in mu, at 300 and 400 bits:
    exponential:  log(1 - exp(-y/mu))  (upper = 0),   log(exp(-y/mu))  (upper = 1),
    chisq:        log P(mu/2, y/2)     (upper = 0),   log Q(mu/2, y/2)  (upper = 1),
with P and Q the regularized incomplete gamma functions (mpmath.gammainc). A row is kept
if the two precisions agree to 2^-80 relative. The route shares nothing with the library's:
no expansion of the logarithm, no Faa di Bruno sum, no polygamma.

The grid: exponential at mu = 1e-3, 1 and 1e4 (the derivative tensor scales as mu^-K, but the
three values are checked independently), y = mu u for u in U_EXP, which covers both
sides of the switch u = 1 of the library, the tails and the small-u regime; chisq at
mu in MU_CHI and y = mu r for r in R_CHI.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpmath.mpf(2) ** -80

U_EXP = [1e-8, 1e-4, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0, 1.01, 1.2, 2.0, 3.5, 6.0, 12.0, 25.0, 40.0, 80.0, 300.0]
MU_EXP = [1e-3, 1.0, 1e4]
MU_CHI = [1e-3, 0.05, 1.0, 3.0, 17.0, 150.0, 1e4]
R_CHI = [1e-4, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0, 20.0]


def log_cdf(dist, upper, y, mu):
    """log F (upper = 0) or log(1 - F) (upper = 1) at observation y and parameter mu."""
    if dist == 0:
        if upper:
            return -y / mu
        return mpmath.log(1 - mpmath.exp(-y / mu))
    a, x = mu / 2, y / 2
    if upper:
        return mpmath.log(mpmath.gammainc(a, x, mpmath.inf, regularized=True))
    return mpmath.log(mpmath.gammainc(a, 0, x, regularized=True))


def value(dist, upper, y, K, mu0, prec):
    with mpmath.workprec(prec):
        f = lambda mu: log_cdf(dist, upper, mpmath.mpf(y), mu)
        return mpmath.diff(f, mpmath.mpf(mu0), K)


def rows_for(dist, upper, y, K, mu0):
    a = value(dist, upper, y, K, mu0, 300)
    c = value(dist, upper, y, K, mu0, 400)
    if c == 0 or not mpmath.isfinite(c) or abs(a - c) > AGREE * abs(c):
        return None
    hi = float(c)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    lo = float(c - mpmath.mpf(hi))
    return f"{upper},{float(y).hex()},{K},{hi.hex()},{lo.hex()},{float(mu0).hex()}\n"


def task(job):
    dist, mu0, ys = job
    lines, dropped = [], 0
    for upper in (0, 1):
        for y in ys:
            for K in range(1, 7):
                line = rows_for(dist, upper, mpmath.mpf(y), K, mpmath.mpf(mu0))
                if line is None:
                    dropped += 1
                else:
                    lines.append(line)
    return lines, dropped


def jobs():
    out = []
    for mu in MU_EXP:
        m = float(mu)
        ys = [float(m * u) for u in U_EXP]  # the double y of the test; u = y/mu is then close to the grid
        out.append((0, m, ys))
    for mu in MU_CHI:
        m = float(mu)
        ys = [float(m * r) for r in R_CHI]
        out.append((1, m, ys))
    return out


def main():
    with Pool(4) as pool:
        res = pool.map(task, jobs())
    names = {0: "distrib_exponential.csv", 1: "distrib_chisq.csv"}
    for dist, name in names.items():
        n = dropped = 0
        with open(REF / name, "w", newline="\n") as f:
            f.write("upper,y,K,hi,lo,mu\n")
            for (d, _, _), (lines, dr) in zip(jobs(), res):
                if d != dist:
                    continue
                f.writelines(lines)
                n += len(lines)
                dropped += dr
        print(f"{name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
