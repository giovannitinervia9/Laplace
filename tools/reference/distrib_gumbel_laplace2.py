"""Reference values of the derivatives of log F and log(1 - F) for gumbel_distrib (in (mu, sigma))
and laplace2_distrib (in (mu, lambda)), orders 1 to 6, both tails and the centre.

Usage (from the repository root):  python -I tools/reference/distrib_gumbel_laplace2.py

File tests/data/reference/distrib_gumbel_laplace2.csv, one row per value:
    dist (4 gumbel, 5 laplace2), upper (0 log F, 1 log(1 - F)), y, K, b, hi, lo, mu, second
with b the number of derivatives in the second parameter (K - b in mu), second = sigma for the
Gumbel and lambda for laplace2; y, hi, lo, mu and second in hexadecimal. The parameters are
mu = 0.4 with sigma = 1.3 (Gumbel) and lambda = 0.8 (laplace2).

Method: mpmath.diff (high-precision finite differences with its own step control) of the
DEFINITIONS, not of the library's formulas:
  Gumbel:    F = exp(-exp(-z)), z = (y - mu)/sigma; log F = log F, log(1 - F) = log(-expm1(-exp(-z)));
  laplace2:  F = exp(lambda (y - mu))/2 for y < mu, 1 - exp(-lambda (y - mu))/2 for y > mu;
             log F and log(1 - F) of these piecewise definitions.
Each value is computed at 300 and 400 bits; a row is kept when the two agree to 2^-80 relative.
Points of the laplace2 grid with |y - mu| < 0.05 are not used (the kink at y = mu). The
Gumbel grid runs over z from -5 to 40, where 1 - F underflows slowly; the laplace2 grid over
|z| = lambda |y - mu| from 0.05 to 30 on both sides of mu.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_gumbel_laplace2.csv"
MU = 0.4
SIGMA = 1.3
LAMBDA = 0.8
AGREE = mpmath.mpf(2) ** -80

GUMBEL_Z = [-5.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 40.0]
LAPLACE_Z = [-30.0, -12.0, -5.0, -2.0, -1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]


def gumbel_log_cdf(upper, y, mu, s):
    z = (y - mu) / s
    if upper:
        return mpmath.log(-mpmath.expm1(-mpmath.exp(-z)))
    return mpmath.log(mpmath.exp(-mpmath.exp(-z)))


def laplace2_log_cdf(upper, y, mu, lam):
    d = lam * (y - mu)
    if y < mu:
        F = mpmath.exp(d) / 2
        return mpmath.log(1 - F) if upper else mpmath.log(F)
    F = 1 - mpmath.exp(-d) / 2
    return mpmath.log(mpmath.exp(-d) / 2) if upper else mpmath.log(F)


def value(dist, upper, y, K, b, prec):
    with mpmath.workprec(prec):
        if dist == 4:
            f = lambda mu, s: gumbel_log_cdf(upper, mpmath.mpf(y), mu, s)
            return mpmath.diff(f, (mpmath.mpf(MU), mpmath.mpf(SIGMA)), (K - b, b))
        f = lambda mu, lam: laplace2_log_cdf(upper, mpmath.mpf(y), mu, lam)
        return mpmath.diff(f, (mpmath.mpf(MU), mpmath.mpf(LAMBDA)), (K - b, b))


def task(args):
    dist, upper = args
    lines, dropped = [], 0
    if dist == 4:
        ys = [MU + SIGMA * z for z in GUMBEL_Z]
        second = SIGMA
    else:
        ys = [MU + z / LAMBDA for z in LAPLACE_Z]
        second = LAMBDA
    for y in ys:
        for K in range(1, 7):
            for b in range(K + 1):
                a, c = value(dist, upper, y, K, b, 300), value(dist, upper, y, K, b, 400)
                if c == 0 or abs(a - c) > AGREE * abs(c):
                    dropped += 1
                    continue
                hi = float(c)
                if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                    dropped += 1
                    continue
                lo = float(c - mpmath.mpf(hi))
                lines.append(f"{dist},{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{MU.hex()},{second.hex()}\n")
    return lines, dropped


def main():
    jobs = [(d, u) for d in (4, 5) for u in (0, 1)]
    with Pool(4) as pool:
        res = pool.map(task, jobs)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("dist,upper,y,K,b,hi,lo,mu,second\n")
        for lines, d in res:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
