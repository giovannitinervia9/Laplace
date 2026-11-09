"""Reference values of the special functions, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/special.py [name ...]

Same format as elementary.py: tests/data/reference/<name>.csv with columns x, hi, lo
in hexadecimal, hi + lo the value as a double-double. The arguments cover the
domain the distributions use (x > 0), with extra points near the zeros of the
function (lgamma at 1 and 2, digamma at its positive root) where the relative
error is hardest, near 0, and at large x where the asymptotic forms take over.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
N_RANDOM = 20000
SEED = 20261108
DIGAMMA_ROOT = 1.4616321449683623


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def near(rng, c, n, lo_exp=-50, hi_exp=-2):
    return [c + math.copysign(log_uniform(rng, lo_exp, hi_exp), rng.random() - 0.5) for _ in range(n)]


def positive_sample(rng, max_exp):
    xs = [log_uniform(rng, -60, max_exp) for _ in range(N_RANDOM // 2)]
    xs += [rng.uniform(0.0, 30.0) for _ in range(N_RANDOM // 4)]
    return xs


FUNCTIONS = {
    # name: (mpmath function, extra arguments)
    "lgamma": (mpmath.loggamma,
               lambda rng: positive_sample(rng, 1000) + near(rng, 1.0, N_RANDOM // 8)
               + near(rng, 2.0, N_RANDOM // 8) + [1.0, 2.0, 0.5, 3.0, 1e-300, 5e-324, 1e300]),
    "digamma": (mpmath.digamma,
                lambda rng: positive_sample(rng, 1000) + near(rng, DIGAMMA_ROOT, N_RANDOM // 4)
                + [1.0, 2.0, 0.5, DIGAMMA_ROOT, 1e-300, 1e300]),
    "trigamma": (lambda x: mpmath.polygamma(1, x),
                 lambda rng: positive_sample(rng, 1000) + [rng.uniform(0.0, 3.0) for _ in range(N_RANDOM // 4)]
                 + [1.0, 2.0, 0.5, 1e-150, 1e300]),
}
for _n in range(2, 7):
    FUNCTIONS[f"polygamma{_n}"] = (
        (lambda n: (lambda x: mpmath.polygamma(n, x)))(_n),
        lambda rng: positive_sample(rng, 1000) + [rng.uniform(0.0, 3.0) for _ in range(N_RANDOM // 4)]
        + [1.0, 2.0, 0.5, 1e-40, 1e300])


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        f, sampler = FUNCTIONS[name]
        rng = random.Random(SEED)
        xs = [x for x in sampler(rng) if x > 0]
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("x,hi,lo\n")
            for x in xs:
                hi, lo = dd(f(mpmath.mpf(x)))
                fh.write(f"{float.hex(x)},{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(xs)} rows")


if __name__ == "__main__":
    main(sys.argv[1:] or list(FUNCTIONS))
