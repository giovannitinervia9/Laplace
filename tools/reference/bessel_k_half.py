"""Reference values of log K_nu(x) and of its derivatives in x to order 7 at
half-integer orders nu = n + 1/2, computed with mpmath.besselk and mpmath.diffs.

Usage (from the repository root):
    python -I tools/reference/bessel_k_half.py [--workers=N]

Files (format as in bessel_k.py: the arguments nu, x, then hi, lo):
    log_bessel_k_half.csv          nu, x   log K_nu(x)
    log_bessel_k_half_dx{m}.csv    nu, x   d^m/dx^m log K_nu(x), m = 1..7

Method. log K_nu(x) = log besselk(nu, x), and its derivatives in x by
mpmath.diffs (finite differences at raised working precision) of that function.
Neither step uses the closed form K_(n+1/2)(x) = sqrt(pi/(2x)) e^-x P_n(1/(2x))
that the library evaluates. Every row is computed at P = 320 bits and P + 128,
and kept only if all eight values agree to 2^-110 relative (dropped rows are
counted). The points: n = 0..40, x concentrated in [0.3, 3] (where Temme's
series meets reduced order -1/2 and its first coefficient cancels near x = 1),
the rest log-uniform over [2^-14, 2^12].
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261116
NMAX = 7
NHALF = 40
AGREE = mpmath.mpf(2) ** -110


def dd(v):
    hi = float(v)
    lo = float(v - mpmath.mpf(hi))
    return hi, lo


def values(nu, x, prec):
    with mpmath.workprec(prec):
        n_, x_ = mpmath.mpf(nu), mpmath.mpf(x)
        f = lambda t: mpmath.log(mpmath.besselk(n_, t))
        return [+v for v in mpmath.diffs(f, x_, NMAX)]


def agree(u, v):
    return abs(u - v) <= AGREE * max(abs(u), abs(v), mpmath.mpf(2) ** -1070)


def row(args):
    nu, x = args
    u, v = values(nu, x, 320), values(nu, x, 448)
    if all(agree(a, b) for a, b in zip(u, v)):
        return (nu, x, v)
    return None


def sample(rng, count):
    pts = []
    for _ in range(count):
        n = rng.randint(0, NHALF)
        if rng.random() < 0.6:
            x = rng.uniform(0.3, 3.0)
        else:
            x = 2.0 ** rng.uniform(-14, 12)
        pts.append((n + 0.5, x))
    return pts


def main():
    workers = 4
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
    pts = sample(random.Random(SEED), 1500)
    pts += [(n + 0.5, x) for n in (0, 1, 2, 3, 5, 10, 19) for x in (0.9, 1.0, 1.05, 1.12)]
    with Pool(workers) as pool:
        out = pool.map(row, pts, chunksize=4)
    rows = sorted(r for r in out if r is not None)
    print(f"{len(pts) - len(rows)} rows dropped (precisions disagree)")
    for m in range(NMAX + 1):
        name = "log_bessel_k_half" if m == 0 else f"log_bessel_k_half_dx{m}"
        with open(OUT / f"{name}.csv", "w", newline="\n") as f:
            f.write("nu,x,hi,lo\n")
            for nu, x, v in rows:
                hi, lo = dd(v[m])
                f.write(f"{float(nu).hex()},{float(x).hex()},{hi.hex()},{lo.hex()}\n")
    print(f"{len(rows)} rows written")


if __name__ == "__main__":
    main()
