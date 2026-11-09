"""Reference values of the incomplete beta function in the regions where it is
hard to compute, with mpmath at 200 bits.

Usage (from the repository root):  python -I tools/reference/ibeta_hard.py [processes]

Writes tests/data/reference/ibeta_hard.csv (I_x(a, b)) and ibetac_hard.csv
(1 - I_x(a, b)) on the same arguments, in the format of incomplete.py: a, b, x,
then hi, lo in hexadecimal, hi + lo being the exact value as a double-double.

The regions (s = a + b, mean m = a/s, standard deviation sd = sqrt(a b / (s^2 (s + 1)))):
  1. one shape small (2^-8 to 40), the other large (40 to 2^14), x = m + k sd
     with k uniform in (-4, 4), a quarter of them next to the switching point
     (a + 1)/(a + b + 2) of the old continued fraction, a tenth with k up to 30;
  2. both shapes between 1 and 40, x = m + k sd, k in (-5, 5);
  3. both shapes large (40 to 2^14), ratio of the shapes log-uniform up to 2^8,
     x = m + k sd, k in (-5, 5) and a tenth with k up to 30;
  4. both shapes below 1, x uniform or log-uniform near 0 and 1;
  5. the worst arguments of the first own version (docs/numerics.md).

The value is computed on the side of the mean where the hypergeometric series
has positive terms (as in incomplete.py, whose formulas are copied here so that
that file stays untouched), and the other side as 1 minus it at 200 bits, which
is exact enough because the side below the mean is at most about one half.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261109


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    # continuous log-uniform between 2^lo_exp and 2^hi_exp
    return 2.0 ** rng.uniform(lo_exp, hi_exp)


def _I_series(a, b, x):
    # I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) 2F1(a + b, 1; a + 1; x)   (copied from incomplete.py)
    lb = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    return mpmath.exp(a * mpmath.log(x) + b * mpmath.log1p(-x) - mpmath.log(a) - lb) * \
        mpmath.hyp2f1(a + b, 1, a + 1, x, maxterms=10**7)


def I_and_Ic(args):
    a, b, x = args
    A, B, X = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(x)
    if X <= A / (A + B):
        i = _I_series(A, B, X)
        return i, 1 - i
    ic = _I_series(B, A, 1 - X)
    return 1 - ic, ic


def near(rng, a, b, kmax):
    s = a + b
    m = a / s
    sd = math.sqrt(a * b / (s * s * (s + 1)))
    return m + rng.uniform(-kmax, kmax) * sd


def args_hard():
    rng = random.Random(SEED)
    out = []

    def add(a, b, x):
        if 0.0 < x < 1.0:
            out.append((a, b, x))

    # 1. one small, one large
    for i in range(3000):
        small, large = log_uniform(rng, -8, math.log2(40)), log_uniform(rng, math.log2(40), 14)
        a, b = (small, large) if i % 2 == 0 else (large, small)
        r = i % 20
        if r < 5:
            sw = (a + 1) / (a + b + 2)
            add(a, b, sw * (1 + rng.uniform(-0.05, 0.05)))
        elif r < 7:
            add(a, b, near(rng, a, b, 30.0))
        else:
            add(a, b, near(rng, a, b, 4.0))
    # 2. both moderate
    for i in range(1500):
        a, b = log_uniform(rng, 0, math.log2(40)), log_uniform(rng, 0, math.log2(40))
        add(a, b, near(rng, a, b, 5.0))
    # 3. both large
    for i in range(2500):
        a = log_uniform(rng, math.log2(40), 14)
        b = a * log_uniform(rng, -8, 8)
        if not (40 <= b <= 2.0**14):
            b = log_uniform(rng, math.log2(40), 14)
        if i % 10 == 0:
            add(a, b, near(rng, a, b, 30.0))
        elif i % 10 == 1:
            add(a, a, near(rng, a, a, 5.0))
        else:
            add(a, b, near(rng, a, b, 5.0))
    # 4. both small
    for i in range(800):
        a, b = log_uniform(rng, -8, 0), log_uniform(rng, -8, 0)
        k = i % 3
        if k == 0:
            add(a, b, rng.random())
        elif k == 1:
            add(a, b, log_uniform(rng, -40, -1))
        else:
            add(a, b, 1.0 - log_uniform(rng, -50, -1))
    # 5. worst arguments of the first own version
    for a, b, x in [(1.149, 3599.17, 0.00061), (6111.26, 17.29, 0.99689), (4902.22, 7.29, 0.9983),
                    (44.2, 0.0042, 0.982), (0.064, 7678, 0.00016)]:
        for k in range(-20, 21):
            add(a, b, x * (1 + k * 2e-3))
    return out


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    args = args_hard()
    with Pool(procs) as pool:
        vals = pool.map(I_and_Ic, args, chunksize=16)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, idx in (("ibeta_hard", 0), ("ibetac_hard", 1)):
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("a,b,x,hi,lo\n")
            for (a, b, x), v in zip(args, vals):
                hi, lo = dd(v[idx])
                fh.write(",".join(float.hex(float(t)) for t in (a, b, x)) + f",{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(args)} rows", flush=True)


if __name__ == "__main__":
    main()
