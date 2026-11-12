"""Reference values of Stirling's error delta(a) and of its derivatives to order 6,
computed with mpmath.

    delta(a) = log Gamma(a) - (a - 1/2) log a + a - log sqrt(2 pi),   a > 0,

the remainder of Stirling's series for log Gamma (delta(a) ~ 1/(12 a) as
a -> inf, ~ -log(a)/2 as a -> 0).

Usage (from the repository root):  python -I tools/reference/stirling.py [--workers=N]

Files (format as in exprel.py: the argument a, then hi, lo in hexadecimal):
    stirling_error.csv        a   delta(a)
    stirling_error_d{k}.csv   a   delta^(k)(a), k = 1..6

Method: from the definition, with mpmath's loggamma and polygamma
(delta^(k)(a) = psi^(k-1)(a) - d^k/da^k [(a - 1/2) log a - a], the elementary
part differentiated in closed form). The difference cancels heavily at large a
(about a^2 relative at a = 1e15, 2^100); it is computed at 400 and 500 bits and
kept only if the two agree to 2^-100 relative and the value is a finite normal
double. The library uses other routes (Binet's integral, the asymptotic
series), so the two are independent.

Points: a log-uniform in [2^-40, 1e15]; a uniform in [1/4, 40] (where the
library changes route); around each threshold of the library T,
a = T 2^u with u uniform in [-1, 1].
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261112
KMAX = 6
PRECS = (400, 500)
AGREE = mpmath.mpf(2) ** -100
TINY = 2.2250738585072014e-308
THRESH = (0.5, 8.0, 12.0)  # candidate route boundaries, refined by the measurement


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def elementary_dk(k, a):
    """d^k/da^k [(a - 1/2) log a - a]"""
    if k == 0:
        return (a - mpmath.mpf(1) / 2) * mpmath.log(a) - a
    if k == 1:
        return mpmath.log(a) - 1 / (2 * a)
    # d^k [a log a] = (-1)^k (k-2)! / a^(k-1); d^k [-(1/2) log a] = (-1)^k (k-1)! / (2 a^k)
    return (-1) ** k * (mpmath.factorial(k - 2) / a ** (k - 1) + mpmath.factorial(k - 1) / (2 * a ** k))


def delta_k(k, a):
    if k == 0:
        return mpmath.loggamma(a) - elementary_dk(0, a) - mpmath.log(2 * mpmath.pi) / 2
    return mpmath.polygamma(k - 1, a) - elementary_dk(k, a)


def row(a):
    av = float(a)
    vals = {}
    for p in PRECS:
        with mpmath.workprec(p):
            ma = mpmath.mpf(av)
            vals[p] = [delta_k(k, ma) for k in range(KMAX + 1)]
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
    xs = [0.5, 1.0, 2.0, 10.0]
    for _ in range(700):  # log-uniform in [2^-40, 1e15]
        xs.append(math.exp(rng.uniform(-40 * math.log(2.0), math.log(1e15))))
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
        name = "stirling_error" if k == 0 else f"stirling_error_d{k}"
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
