"""Reference values of the elementary functions, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/elementary.py

For every function it writes tests/data/reference/<name>.csv with one row per
argument x and three columns, all doubles written in hexadecimal so that they
are read back exactly:

    x, hi, lo

where hi + lo is the value of the function at x as a double-double
(hi = the correctly rounded double, lo = the rounded remainder). The error of
a computed y in units in the last place is then ((y - hi) - lo) / ulp(hi).

The arguments are a fixed-seed sample over the whole domain (uniform in the
argument or in its binary exponent, depending on the function) plus the edge
cases: zero, values near 1 or 0 where cancellation happens, the overflow and
underflow thresholds, subnormal arguments and results.
"""
import math
import random
from pathlib import Path

import mpmath

mpmath.mp.prec = 200          # 200 bits: far beyond the 106 of a double-double
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
N_RANDOM = 20000
SEED = 20261108


def dd(v):
    """mpmath value -> (hi, lo) doubles."""
    if mpmath.isinf(v):
        return (math.inf if v > 0 else -math.inf), 0.0
    hi = float(v)
    if math.isinf(hi):
        return hi, 0.0
    lo = float(v - mpmath.mpf(hi))
    return hi, lo


def log_uniform(rng, lo_exp, hi_exp):
    """A double whose binary exponent is uniform in [lo_exp, hi_exp]."""
    e = rng.randint(lo_exp, hi_exp)
    return math.ldexp(1.0 + rng.random(), e)


def edges_exp():
    xs = [0.0, -0.0, 1.0, -1.0, 0.5, math.log(2), 2 ** -28, -(2 ** -28), 2 ** -60,
          709.0, 709.78, 709.782712893383973096, -708.39, -708.4, -744.0,
          -745.0, -745.133219101941108420, 1e-300, -1e-300]
    return xs


def edges_log():
    xs = [1.0, 2.0, 0.5, math.e, 1.0 + 2 ** -52, 1.0 - 2 ** -53, 1.0 + 2 ** -20,
          1.0 - 2 ** -20, math.nextafter(1.0, 2.0), math.nextafter(1.0, 0.0),
          5e-324, 2.2250738585072014e-308, 2.2250738585072009e-308, 1e-310,
          1.7976931348623157e308, 0.7071067811865476, 1.4142135623730951]
    return xs


def sample(name, rng):
    xs = []
    if name == "exp":
        xs += edges_exp()
        xs += [rng.uniform(-745.13, 709.78) for _ in range(N_RANDOM // 2)]
        xs += [rng.uniform(-1.0, 1.0) for _ in range(N_RANDOM // 4)]
        xs += [math.copysign(log_uniform(rng, -60, -1), rng.random() - 0.5)
               for _ in range(N_RANDOM // 4)]
        f = mpmath.exp
    elif name == "log":
        xs += edges_log()
        xs += [log_uniform(rng, -1074, 1023) for _ in range(N_RANDOM // 2)]
        xs += [rng.uniform(0.5, 2.0) for _ in range(N_RANDOM // 4)]
        xs += [1.0 + math.copysign(log_uniform(rng, -52, -2), rng.random() - 0.5)
               for _ in range(N_RANDOM // 4)]
        xs = [x for x in xs if x > 0]
        f = mpmath.log
    else:
        raise ValueError(name)
    rows = []
    for x in xs:
        hi, lo = dd(f(mpmath.mpf(x)))
        rows.append((x, hi, lo))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["exp", "log"]:
        rng = random.Random(SEED)
        rows = sample(name, rng)
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("x,hi,lo\n")
            for x, hi, lo in rows:
                fh.write(f"{float.hex(x)},{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
