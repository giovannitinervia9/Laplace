"""Reference values of the derivatives of log F and log(1 - F) for gaussian2 (mean mu,
variance v) and gaussian3 (mean mu, precision tau), orders 1 to 6, on a grid of
standardized points z that covers both tails and the centre.

Usage (from the repository root):  python -I tools/reference/distrib_gaussian23.py

File tests/data/reference/distrib_gaussian23.csv, one row per value:
    dist (0 gaussian2, second parameter v; 1 gaussian3, second parameter tau), upper
    (0 log F, 1 log(1 - F)), y, K, b, hi, lo, mu, w
with b the number of derivatives in the second parameter (K - b in mu); y, hi, lo, mu
and w in hexadecimal. The mean is mu = 0.4 throughout; w is v or tau, two values per
distribution (a moderate and a small or large one, see PAIRS).

Method: mpmath.diff (high-precision finite differences on the analytic function, with
its own step control) of the DEFINITION
    log ncdf((y - mu)/sigma)                 (upper = 0, log F),
    log ncdf((mu - y)/sigma)                 (upper = 1, log(1 - F)),
with sigma = sqrt(v) for gaussian2 and sigma = 1/sqrt(tau) for gaussian3, at 300 and
400 bits. A row is kept if the two agree to 2^-80 relative. The route shares nothing
with the library's: no z-derivatives of log Phi and no polynomial recursion; the
library's closed forms of the tail parts are not used here either.

The observation points are y = mu + sigma z for z in GRID (rounded to double), so
that the tails |z| >= 4 and the centre are both sampled. Points where the reference is
zero or not finite, or where the two precisions disagree, are dropped.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_gaussian23.csv"
MU = 0.4
AGREE = mpmath.mpf(2) ** -80
GRID = [-35.0, -22.0, -13.0, -8.0, -6.0, -4.5, -4.0, -3.5, -2.5, -1.5, -0.5, 0.0,
        0.5, 1.5, 2.5, 3.5, 4.0, 4.5, 6.0, 8.0, 13.0, 22.0, 35.0]
# (dist, w): gaussian2 with v = 1.69 (sigma 1.3) and v = 0.09 (sigma 0.3);
# gaussian3 with tau = 0.6 and tau = 16 (sigma 0.258 and 0.25)
PAIRS = [(0, 1.69), (0, 0.09), (1, 0.6), (1, 16.0)]


def log_cdf(dist, upper, y, mu, w):
    """log F (upper = 0) or log(1 - F) (upper = 1) of the normal with the second parameter w."""
    if dist == 0:
        sigma = mpmath.sqrt(w)
    else:
        sigma = 1 / mpmath.sqrt(w)
    z = (y - mu) / sigma
    if upper:
        z = -z
    return mpmath.log(mpmath.ncdf(z))


def value(dist, upper, y, K, b, w0, prec):
    with mpmath.workprec(prec):
        f = lambda mu, w: log_cdf(dist, upper, mpmath.mpf(y), mu, w)
        return mpmath.diff(f, (mpmath.mpf(MU), mpmath.mpf(w0)), (K - b, b))


def task(pair):
    dist, w0 = pair
    sigma = math.sqrt(w0) if dist == 0 else 1.0 / math.sqrt(w0)
    lines, dropped = [], 0
    for upper in (0, 1):
        for zz in GRID:
            y = MU + sigma * zz
            for K in range(1, 7):
                for b in range(K + 1):
                    a, c = value(dist, upper, y, K, b, w0, 300), value(dist, upper, y, K, b, w0, 400)
                    if c == 0 or not mpmath.isfinite(c) or abs(a - c) > AGREE * abs(c):
                        dropped += 1
                        continue
                    hi = float(c)
                    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                        dropped += 1
                        continue
                    lo = float(c - mpmath.mpf(hi))
                    lines.append(f"{dist},{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{MU.hex()},{w0.hex()}\n")
    return lines, dropped


def main():
    with Pool(4) as pool:
        res = pool.map(task, PAIRS)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("dist,upper,y,K,b,hi,lo,mu,w\n")
        for lines, d in res:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
