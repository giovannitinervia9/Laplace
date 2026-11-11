"""Reference values of the double-double square root, exponential and logarithm,
computed with mpmath at 300 bits.

Usage (from the repository root):  python -I tools/reference/double_double.py

For every function it writes tests/data/reference/dd_<name>.csv (name = sqrt,
exp, log) with one row per argument and four columns, all doubles written in
hexadecimal so that they are read back exactly:

    x_hi, x_lo, hi, lo

The argument is the double-double x_hi + x_lo, with |x_lo| <= ulp(x_hi) / 2
(a random fraction of that bound, except for the arguments 1 +- 2^-k, which are
given exactly as the double-double nearest to 1 +- 2^-k). The value is computed
from this exact argument, so hi + lo is the function at x_hi + x_lo to about
2^-250 relative: the reference is never the function at x_hi alone.

Points. exp: uniform on [-745.2, 709.8], uniform on [-1, 1], log-uniform
magnitudes from 2^-60 to 2^9, points around k ln 2 and (k + 1/2) ln 2 (the
reduction boundaries), and the overflow and underflow thresholds. log:
log-uniform over 2^-1000 .. 2^1000, subnormal arguments, uniform on [0.5, 2],
and 1 +- 2^-k for k = 1 .. 100 (both signs). sqrt: log-uniform over
2^-1000 .. 2^1000, subnormal arguments, uniform on [0.5, 2] and squares near
the perfect squares. About 3000 points per function.
"""
import math
import random
from pathlib import Path

import mpmath

mpmath.mp.prec = 300  # 300 bits: far beyond the 106 of a double-double
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261111
N = 3000


def dd_of(v):
    """mpmath value -> (hi, lo) doubles, hi the rounded value, lo the remainder."""
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    """A positive double whose binary exponent is uniform in [lo_exp, hi_exp]."""
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def dd_argument(rng, x_hi):
    """A double-double with leading part x_hi and a random tail of at most ulp / 2."""
    return x_hi, rng.uniform(-1.0, 1.0) * math.ulp(x_hi) / 2.0


def exp_points(rng):
    xs = [0.0, 1.0, -1.0, 0.5, -0.5, math.log(2.0), -math.log(2.0), 2.0 ** -30, -(2.0 ** -30),
          2.0 ** -60, -(2.0 ** -60), 709.0, 709.78, 709.782712893383973096, 709.7827128933840,
          -708.39, -708.4, -745.0, -745.133219101941108420, -745.14, 1e-300, -1e-300]
    xs += [rng.uniform(-745.2, 709.8) for _ in range(N // 3)]
    xs += [rng.uniform(-1.0, 1.0) for _ in range(N // 6)]
    xs += [math.copysign(log_uniform(rng, -60, 9), rng.random() - 0.5) for _ in range(N // 6)]
    for _ in range(N // 6):
        k = rng.randint(-1075, 1024)
        xs.append(k * math.log(2.0) + rng.uniform(-2.0 ** -12, 2.0 ** -12))
        xs.append((k + 0.5) * math.log(2.0) + rng.uniform(-2.0 ** -12, 2.0 ** -12))
    return [x for x in xs if -746.0 <= x <= 710.0]


def log_points(rng):
    xs = [1.0, 2.0, 0.5, math.e, 1.0 + 2.0 ** -52, 1.0 - 2.0 ** -53, 1.0 + 2.0 ** -20,
          1.0 - 2.0 ** -20, math.nextafter(1.0, 2.0), math.nextafter(1.0, 0.0),
          5e-324, 2.2250738585072014e-308, 2.2250738585072009e-308, 1e-310,
          1.7976931348623157e308, 0.7071067811865476, 1.4142135623730951]
    xs += [log_uniform(rng, -1000, 1000) for _ in range(N // 2)]
    xs += [math.ldexp(1.0 + rng.random(), rng.randint(-1074, -1023)) for _ in range(N // 12)]
    xs += [rng.uniform(0.5, 2.0) for _ in range(N // 6)]
    rest = N - len(xs)
    xs += [log_uniform(rng, -10, 10) for _ in range(max(rest, 0))]
    return xs


def near_one(rng):
    """1 +- 2^-k for k = 1 .. 100, both signs, as exact double-doubles."""
    rows = []
    for k in range(1, 101):
        for s in (1.0, -1.0):
            t = mpmath.mpf(1) + s * mpmath.mpf(2) ** -k
            hi, lo = dd_of(t)
            rows.append((hi, lo))
    for _ in range(N // 12):
        k = rng.randint(1, 100)
        s = rng.choice((1.0, -1.0))
        t = mpmath.mpf(1) + s * mpmath.mpf(2) ** -k * mpmath.mpf(1.0 + rng.random())
        rows.append(dd_of(t))
    return rows


def sqrt_points(rng):
    xs = [2.0, 4.0, 0.25, 1e-300, 1.7976931348623157e308, 5e-324, 2.2250738585072014e-308,
          0.5, 3.0, 9.0, 1e-10, 1e10]
    xs += [log_uniform(rng, -1000, 1000) for _ in range(N // 2)]
    xs += [math.ldexp(1.0 + rng.random(), rng.randint(-1074, -1023)) for _ in range(N // 12)]
    xs += [rng.uniform(0.5, 2.0) for _ in range(N // 6)]
    for _ in range(N // 12):
        r = math.ldexp(1.0 + rng.random(), rng.randint(-500, 500))
        xs.append(r * r)
    return xs


def rows_for(name, rng):
    if name == "exp":
        f = mpmath.exp
        pairs = [dd_argument(rng, x) for x in exp_points(rng)]
    elif name == "log":
        f = mpmath.log
        pairs = [dd_argument(rng, x) for x in log_points(rng) if x > 0]
        pairs += near_one(rng)
    elif name == "sqrt":
        f = mpmath.sqrt
        pairs = [dd_argument(rng, x) for x in sqrt_points(rng)]
    elif name == "expm1":
        f = mpmath.expm1
        xs = [math.copysign(log_uniform(rng, -80, -2), rng.random() - 0.5) for _ in range(2000)]
        xs += [rng.uniform(-0.35, 0.35) for _ in range(500)] + [rng.uniform(-40, 40) for _ in range(500)]
        pairs = [dd_argument(rng, x) for x in xs]
    else:
        raise ValueError(name)
    rows = []
    for x_hi, x_lo in pairs:
        v = f(mpmath.mpf(x_hi) + mpmath.mpf(x_lo))
        hi, lo = dd_of(v)
        rows.append((x_hi, x_lo, hi, lo))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["sqrt", "exp", "log", "expm1"]:
        rng = random.Random(SEED)
        rows = rows_for(name, rng)
        path = OUT / f"dd_{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("x_hi,x_lo,hi,lo\n")
            for x_hi, x_lo, hi, lo in rows:
                fh.write(f"{float.hex(x_hi)},{float.hex(x_lo)},{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
