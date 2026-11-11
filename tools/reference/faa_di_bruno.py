"""Reference values for Faa di Bruno's formula to order 7, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/faa_di_bruno.py

File tests/data/reference/faa_di_bruno.csv, one row per case:
    f1..f7, g1..g7, then d1_hi, d1_lo, ..., d7_hi, d7_lo
where f_k = f^(k)(g(x)), g_j = g^(j)(x) (all doubles, in hexadecimal) and
d_n = d^n/dx^n f(g(x)) computed exactly from those doubles (at 300 bits).

Method. The composition is done on truncated Taylor series, independently of
the Bell polynomials of the library: with F(t) = sum_k f_k t^k / k! and
G(e) = sum_j g_j e^j / j!, the coefficient of e^n in F(G(e)) times n! is d_n.

Cases: (a) derivatives drawn at random, magnitudes log-uniform on
(2^-10, 2^10) with random signs (cancellations included); (b) compositions met
in practice: outer f in {log, exp, lgamma, log Phi, x^-1/2} (log-densities and
their parts) and inner g in {exp, logistic, Phi, x^2, sqrt, log}
(inverse links and transformations), at random points, their derivatives
computed by mpmath.diff and rounded to doubles.
"""
import math
import random
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261111
N = 7
mpmath.mp.prec = 300


def dd(v):
    hi = float(v)
    return hi, float(v - mpmath.mpf(hi))


def compose(f, g):
    """d_1..d_N of f(g(x)) from f[1..N], g[1..N] by series composition"""
    F = [mpmath.mpf(0)] + [mpmath.mpf(f[k]) / mpmath.factorial(k) for k in range(1, N + 1)]
    G = [mpmath.mpf(0)] + [mpmath.mpf(g[j]) / mpmath.factorial(j) for j in range(1, N + 1)]
    out = [mpmath.mpf(0)] * (N + 1)
    power = [mpmath.mpf(1)] + [mpmath.mpf(0)] * N  # G^k
    for k in range(1, N + 1):
        new = [mpmath.mpf(0)] * (N + 1)
        for i in range(N + 1):
            if power[i] == 0:
                continue
            for j in range(1, N + 1 - i):
                new[i + j] += power[i] * G[j]
        power = new
        for n in range(N + 1):
            out[n] += F[k] * power[n]
    return [out[n] * mpmath.factorial(n) for n in range(1, N + 1)]


def random_case(rng):
    def one():
        return math.copysign(math.ldexp(1.0 + rng.random(), rng.randint(-10, 10)), rng.random() - 0.5)
    return [0.0] + [one() for _ in range(N)], [0.0] + [one() for _ in range(N)]


OUTER = {
    "log": mpmath.log,
    "exp": mpmath.exp,
    "lgamma": mpmath.loggamma,
    "logPhi": lambda t: mpmath.log(mpmath.ncdf(t)),
    "rsqrt": lambda t: 1 / mpmath.sqrt(t),
}
INNER = {
    "exp": (mpmath.exp, (-3, 3)),
    "logistic": (lambda x: 1 / (1 + mpmath.exp(-x)), (-6, 6)),
    "Phi": (mpmath.ncdf, (-4, 4)),
    "square": (lambda x: x * x, (0.1, 5)),
    "sqrt": (mpmath.sqrt, (0.05, 20)),
    "log": (mpmath.log, (1.05, 30)),
}


def practical_case(rng, fo, gi):
    gfun, (lo, hi) = INNER[gi]
    ffun = OUTER[fo]
    for _ in range(20):
        x = mpmath.mpf(rng.uniform(lo, hi))
        y = gfun(x)
        if fo in ("log", "rsqrt", "lgamma") and y <= 0:
            continue
        g = [0.0] + [float(mpmath.diff(gfun, x, j)) for j in range(1, N + 1)]
        f = [0.0] + [float(mpmath.diff(ffun, y, k)) for k in range(1, N + 1)]
        return f, g
    return None


def main():
    rng = random.Random(SEED)
    cases = [random_case(rng) for _ in range(1500)]
    for fo in OUTER:
        for gi in INNER:
            for _ in range(15):
                c = practical_case(rng, fo, gi)
                if c is not None:
                    cases.append(c)
    path = OUT / "faa_di_bruno.csv"
    with open(path, "w", newline="\n") as fh:
        cols = [f"f{k}" for k in range(1, N + 1)] + [f"g{k}" for k in range(1, N + 1)]
        cols += [f"d{n}_{p}" for n in range(1, N + 1) for p in ("hi", "lo")]
        fh.write(",".join(cols) + "\n")
        for f, g in cases:
            d = compose(f, g)
            vals = [v.hex() for v in f[1:]] + [v.hex() for v in g[1:]]
            for v in d:
                hi, lo = dd(v)
                vals += [hi.hex(), lo.hex()]
            fh.write(",".join(vals) + "\n")
    print(f"{path.name}: {len(cases)} rows")


if __name__ == "__main__":
    main()
