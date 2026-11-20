"""Reference values of the half-step log-gamma ratio R(a) and of its derivatives to
order 6, computed with mpmath.

    R(a) = log Gamma(a + 1/2) - log Gamma(a),   a > 0,

which behaves as (log a)/2 as a -> inf and as -log a as a -> 0.

Usage (from the repository root):  python -I tools/reference/lgamma_half_ratio.py [--workers=N]

Files (format as in stirling.py: the argument a, then hi, lo in hexadecimal):
    lgamma_half_ratio.csv        a   R(a)
    lgamma_half_ratio_d{k}.csv   a   R^(k)(a) = psi^(k-1)(a + 1/2) - psi^(k-1)(a), k = 1..6

Method: from the definition, with mpmath's loggamma and polygamma. The difference
cancels heavily at large a (about a relative, 1e8 at a = 1e8); it is computed at 400 and
500 bits and kept only if the two agree to 2^-100 relative and the value is a finite
normal double. The library goes through Stirling's error and a convergent series, so the
two routes are independent.

Points: a log-uniform in [2^-30, 1e12]; a uniform in [1/4, 40]; around a = 1, where the
library's recurrence step starts, and around the thresholds of Stirling's error, which it
calls at a and a + 1/2.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261120
KMAX = 6
PRECS = (400, 500)
AGREE = mpmath.mpf(2) ** -100
TINY = 2.2250738585072014e-308
THRESH = (0.5, 1.0, 10.0)  # the step of the recurrence, and the routes of delta


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def ratio_k(k, a):
    half = mpmath.mpf(1) / 2
    if k == 0:
        return mpmath.loggamma(a + half) - mpmath.loggamma(a)
    return mpmath.polygamma(k - 1, a + half) - mpmath.polygamma(k - 1, a)


def row(a):
    av = float(a)
    vals = {}
    for p in PRECS:
        with mpmath.workprec(p):
            ma = mpmath.mpf(av)
            vals[p] = [ratio_k(k, ma) for k in range(KMAX + 1)]
    out = []
    for k in range(KMAX + 1):
        x, y = vals[PRECS[0]][k], vals[PRECS[1]][k]
        if abs(x - y) > AGREE * abs(y):
            out.append(None)
            continue
        hi, lo = dd(y)
        if not (math.isfinite(hi) and abs(hi) >= TINY):
            out.append(None)
            continue
        out.append((hi, lo))
    return av, out


def sample():
    rng = random.Random(SEED)
    xs = [0.5, 1.0, 2.0, 10.0, 0.25, 125.0]
    for _ in range(700):  # log-uniform in [2^-30, 1e12]
        xs.append(math.exp(rng.uniform(-30 * math.log(2.0), math.log(1e12))))
    for _ in range(500):  # uniform in [1/4, 40]
        xs.append(rng.uniform(0.25, 40.0))
    for t in THRESH:
        for _ in range(60):
            xs.append(t * 2.0 ** rng.uniform(-1.0, 1.0))
    return sorted(set(xs))


def main():
    workers = 4
    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            workers = int(arg.split("=")[1])
    xs = sample()
    with Pool(workers) as pool:
        out = pool.map(row, xs, chunksize=8)
    for k in range(KMAX + 1):
        name = "lgamma_half_ratio" if k == 0 else f"lgamma_half_ratio_d{k}"
        path = OUT / f"{name}.csv"
        n = dropped = 0
        with open(path, "w", newline="\n") as f:
            f.write("x,hi,lo\n")
            for x, vals in out:
                if vals[k] is None:
                    dropped += 1
                    continue
                hi, lo = vals[k]
                f.write(f"{x.hex()},{hi.hex()},{lo.hex()}\n")
                n += 1
        print(f"{path.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
