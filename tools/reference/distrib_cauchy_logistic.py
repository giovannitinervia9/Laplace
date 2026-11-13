"""Reference values of the derivatives of log F and log(1 - F) in (mu, sigma) for the
Cauchy and the logistic distributions (cauchy_distrib, logistic_distrib), orders 1 to 6.

Usage (from the repository root):  python -I tools/reference/distrib_cauchy_logistic.py

File tests/data/reference/distrib_cauchy_logistic.csv, one row per value:
    dist (0 cauchy, 1 logistic), upper (0 log F, 1 log(1 - F)), y, K, b, hi, lo, mu, sigma
with b the number of derivatives in sigma (K - b in mu); y, hi, lo, mu and sigma in
hexadecimal. The reference point (mu, sigma) is one of three pairs; y = mu + sigma z for
the grid of z values below (the row stores y itself, so the library and the reference see
the same double).

Method, as in distrib_log_cdf.py: mpmath.diff of the DEFINITION, at 300 and 400 bits,
a row kept when the two agree to 2^-80 relative. The definitions are
    Cauchy:    F = 1/2 + atan(z)/pi,  1 - F = 1/2 - atan(z)/pi,        z = (y - mu)/sigma
    logistic:  F = 1/(1 + e^(-z)),    1 - F = 1/(1 + e^z) (the same function, written so
               that 1 - F keeps its digits at z = 700).
Nothing of the library's route (log atan2, the Riccati recurrence, the series in 1/z^2, the
logistic p recurrence) is used here.

Grid: the Cauchy out to |z| = 1e8 (where the tail is algebraic, log(1 - F) = -log(pi z)
+ O(1/z^2)); the logistic out to |z| = 700 (where e^(-|z|) is 1e-304).
"""
import math
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_cauchy_logistic.csv"
AGREE = mpmath.mpf(2) ** -80
PARAMS = [(0.4, 1.3), (-2.0, 0.25), (1.0, 4.0)]
ZS = {
    0: [-1e8, -1e4, -100.0, -30.0, -7.5, -2.0, -0.7, -0.05, 0.0, 0.3, 1.0, 2.5, 6.0, 25.0, 100.0, 1e4, 1e8],
    1: [-700.0, -300.0, -40.0, -10.0, -3.0, -0.7, -0.05, 0.0, 0.3, 1.0, 2.5, 6.0, 20.0, 40.0, 300.0, 700.0],
}


def log_cdf(dist, upper, y, mu, s):
    z = (y - mu) / s
    if dist == 0:
        val = mpmath.mpf(1) / 2 - mpmath.atan(z) / mpmath.pi if upper else mpmath.mpf(1) / 2 + mpmath.atan(z) / mpmath.pi
        return mpmath.log(val)
    if upper:
        return -mpmath.log(1 + mpmath.exp(z))
    return -mpmath.log(1 + mpmath.exp(-z))


def value(dist, upper, y, K, b, prec, mu0, s0):
    with mpmath.workprec(prec):
        f = lambda mu, s: log_cdf(dist, upper, mpmath.mpf(y), mu, s)
        return mpmath.diff(f, (mpmath.mpf(mu0), mpmath.mpf(s0)), (K - b, b))


def task(args):
    dist, upper, mu0, s0 = args
    lines, dropped = [], 0
    for zz in ZS[dist]:
        y = float(mu0 + s0 * zz)
        for K in range(1, 7):
            for b in range(K + 1):
                a, c = value(dist, upper, y, K, b, 300, mu0, s0), value(dist, upper, y, K, b, 400, mu0, s0)
                if c == 0 or abs(a - c) > AGREE * abs(c):
                    dropped += 1
                    continue
                hi = float(c)
                if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                    dropped += 1
                    continue
                lo = float(c - mpmath.mpf(hi))
                lines.append(f"{dist},{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{mu0.hex()},{s0.hex()}\n")
    return lines, dropped


def main():
    jobs = [(d, u, m, s) for d in (0, 1) for u in (0, 1) for (m, s) in PARAMS]
    with Pool(4) as pool:
        results = pool.map(task, jobs)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("dist,upper,y,K,b,hi,lo,mu,sigma\n")
        for lines, d in results:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
