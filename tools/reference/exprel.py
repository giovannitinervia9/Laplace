"""Reference values of exprel(x) = (e^x - 1)/x and of its derivatives to order 6,
computed with mpmath.

Usage (from the repository root):  python -I tools/reference/exprel.py [--workers=N]

Files (format as in bessel_k.py: the argument x, then hi, lo in hexadecimal):
    exprel.csv          x   phi(x)
    exprel_d{k}.csv     x   phi^(k)(x), k = 1..6

Method. phi^(k)(x) = int_0^1 t^k e^(x t) dt (the k-th derivative of phi, for
k = 0 the integral is phi itself), evaluated by mpmath.quad (tanh-sinh on
subintervals of [0, 1], with the breakpoints clustered at both ends, where the
integrand has its peak for large |x|). This route is independent of the
library, which uses the power series and the forward recursion. Every value is
computed at two precisions, 200 and 300 bits, and kept only if the two agree to
2^-100 relative and the exact value is a finite normal double; the other rows
are dropped and counted.

Points: x = 0; |x| log-uniform in [2^-60, 1] with both signs (the series and
the switch to the recursion); x uniform in [-30, 30]; |x| log-uniform in [1, 745]
with both signs (the tail for x -> -inf); around each threshold T_k of the library
(set from the measurements in docs/numerics.md), |x| = T_k 2^u with u
uniform in [-2, 2] and both signs; x uniform in [600, 745] (the scaled range
above 700); and x uniform in [709, 720] (the overflow of the results, about
716.5).
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
PRECS = (200, 300)
AGREE = mpmath.mpf(2) ** -100
TINY = 2.2250738585072014e-308  # the smallest normal double
X_MAX = 745.0

# Thresholds of the library, per order k: (T for x > 0, T for x < 0). The power
# series below T for x > 0, Kummer's form above -T for x < 0, the recursion
# elsewhere; measured in docs/numerics.md.
THRESH = {1: (1.0, 2.0), 2: (2.0, 2.0), 3: (2.0, 4.0), 4: (4.0, 4.0), 5: (4.0, 8.0), 6: (4.0, 8.0)}
BAND_PER_SIGN = 40


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def phi_k(k, x):
    """int_0^1 t^k e^(x t) dt at the working precision, by tanh-sinh on pieces
    of [0, 1] with the breakpoints clustered at both ends"""
    f = lambda t: t ** k * mpmath.exp(x * t)
    return mpmath.quad(f, [0, mpmath.mpf(2) ** -20, mpmath.mpf(2) ** -8, mpmath.mpf(1) / 8, mpmath.mpf(1) / 2,
                           mpmath.mpf(7) / 8, 1 - mpmath.mpf(2) ** -8, 1 - mpmath.mpf(2) ** -20, 1])


def row(x):
    """[(hi, lo) or None for k = 0..KMAX] for one argument x"""
    xv = float(x)
    vals = {}
    for p in PRECS:
        with mpmath.workprec(p):
            mx = mpmath.mpf(xv)
            vals[p] = [phi_k(k, mx) for k in range(KMAX + 1)]
    out = []
    for k in range(KMAX + 1):
        a, b = vals[PRECS[0]][k], vals[PRECS[1]][k]
        if abs(a - b) > AGREE * abs(b):
            out.append(None)
            continue
        hi, lo = dd(b)
        if not (math.isfinite(hi) and abs(hi) >= TINY):
            out.append(None)
            continue
        out.append((hi, lo))
    return xv, out


def sample():
    rng = random.Random(SEED)
    xs = [0.0]
    for _ in range(400):  # |x| log-uniform in [2^-60, 1], both signs
        m = math.ldexp(1.0 + rng.random(), rng.randint(-60, -1))
        xs.append(math.copysign(m, rng.random() - 0.5))
    for _ in range(400):  # x uniform in [-30, 30]
        xs.append(rng.uniform(-30.0, 30.0))
    for _ in range(300):  # |x| log-uniform in [1, 745], both signs
        m = math.exp(rng.uniform(0.0, math.log(X_MAX)))
        xs.append(math.copysign(m, rng.random() - 0.5))
    for k in sorted(THRESH):  # the thresholds of each order, both signs
        for sign, t in zip((1.0, -1.0), THRESH[k]):
            for _ in range(BAND_PER_SIGN):
                xs.append(sign * t * 2.0 ** rng.uniform(-2.0, 2.0))
    for _ in range(150):  # the scaled range above 700 (e^x e^-700 and its product with e^700)
        xs.append(rng.uniform(600.0, X_MAX))
    for _ in range(50):  # the overflow boundary of the results, about 716.5
        xs.append(rng.uniform(709.0, 720.0))
    return sorted(set(xs))


def main():
    workers = 4
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
    xs = sample()
    with Pool(workers) as pool:
        out = pool.map(row, xs, chunksize=4)
    dropped = [0] * (KMAX + 1)
    for k in range(KMAX + 1):
        name = "exprel" if k == 0 else f"exprel_d{k}"
        path = OUT / f"{name}.csv"
        n = 0
        with open(path, "w", newline="\n") as f:
            f.write("x,hi,lo\n")
            for x, vals in out:
                if vals[k] is None:
                    dropped[k] += 1
                    continue
                hi, lo = vals[k]
                f.write(f"{x.hex()},{hi.hex()},{lo.hex()}\n")
                n += 1
        print(f"{path.name}: {n} rows ({dropped[k]} dropped)")


if __name__ == "__main__":
    main()
