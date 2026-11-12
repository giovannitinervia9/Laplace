"""Reference values for the bounded scalar bijector (include/laplace/bijector/detail/
scalar_bounded.hpp): g(v) = log(v - a) - log(b - v) on (a, b), h(u) = a + (b - a) expit(u).

Usage (from the repository root):  python -I tools/reference/bijector_bounded.py

Writes tests/data/reference/bijector_bounded.csv (through bijector_param_common.py).
Before writing, the closed forms of the header are checked against SymPy: for every
multi-index (n0, n1, n2) of total order K = n0 + n1 + n2 in 1..6, the header's formula
for each of the four quantities (forward, inverse, forward and inverse log-Jacobian)
is compared with SymPy's derivative of the defining expression, at a few rational
points, to 40 digits. A mismatch stops the script.

The parameters are (a, b) = (p1, p2) in this order; the bounds are the grid
(0, 1), (-1, 3), (2, 5), (-1e6, 1e6), (1 - 2^-20, 1) and (-1e-8, 1e-8). The arguments
v are log-uniform towards each bound (down to about 2^-60 of b - a, where a double
allows it) and uniform inside; u runs over 0, powers of 2 and the integers up to
+-750, with random values near 0. The distances (which 4, 5) are included.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp
import sympy as sp

from bijector_param_common import X, P1, P2, write_references, _default_extra

SEED = 20261112
KMAX = 6
DPS = 40

# the defining expressions in SymPy: g in v = X, h in u = X
G = sp.log(X - P1) - sp.log(P2 - X)
H = P1 + (P2 - P1) / (1 + sp.exp(-X))
LJG = sp.log(P2 - P1) - sp.log(X - P1) - sp.log(P2 - X)  # log g'(v)
LJH = sp.log(P2 - P1) + sp.log(1 / (1 + sp.exp(-X))) + sp.log(1 / (1 + sp.exp(X)))  # log h'(u)


def fact(n):
    return mp.factorial(n)


def sgn(k):
    return 1 if k % 2 == 0 else -1


# ---------------------------------------------------------------- closed forms of the header
# the same formulas as scalar_bounded.hpp, in exact (mpmath) arithmetic

def sig_poly(k, U, D):
    """d^k expit / du^k from U = p q and D = q - p, k >= 1 (as in the header)"""
    if k == 1:
        return U
    if k == 2:
        return U * D
    if k == 3:
        return U * (1 - 6 * U)
    if k == 4:
        return U * D * (1 - 12 * U)
    if k == 5:
        return U * (1 - 30 * U + 120 * U**2)
    if k == 6:
        return U * D * (1 - 60 * U + 360 * U**2)
    raise ValueError(k)


def cf_fwd(n0, n1, n2, v, a, b):
    K = n0 + n1 + n2
    p, q = v - a, b - v
    if K == 0:
        return mp.log(p) - mp.log(q)
    A, B = 1 / p, 1 / q
    r = 0
    if n2 == 0:
        r += sgn(n1 + K - 1) * fact(K - 1) * A**K
    if n1 == 0:
        r += sgn(n2) * fact(K - 1) * B**K
    return r


def cf_fwd_lj(n0, n1, n2, v, a, b):
    K = n0 + n1 + n2
    p, q, s = v - a, b - v, b - a
    if K == 0:
        return mp.log(s) - mp.log(p) - mp.log(q)
    A, B, S = 1 / p, 1 / q, 1 / s
    r = 0
    if n2 == 0:
        r += sgn(n0) * fact(K - 1) * A**K
    if n1 == 0:
        r += sgn(n2) * fact(K - 1) * B**K
    if n0 == 0:
        r += sgn(n1 + K - 1) * fact(K - 1) * S**K
    return r


def cf_inv(n0, n1, n2, u, a, b):
    K = n0 + n1 + n2
    p = 1 / (1 + mp.exp(-u))
    q = 1 - p
    U, D = p * q, q - p
    if K == 0:
        return a * q + b * p
    if n1 + n2 == 0:
        return (b - a) * sig_poly(n0, U, D)
    if n1 + n2 == 1:
        if n0 == 0:
            return q if n1 == 1 else p
        sg = sig_poly(n0, U, D)
        return -sg if n1 == 1 else sg
    return 0


def cf_inv_lj(n0, n1, n2, u, a, b):
    K = n0 + n1 + n2
    p = 1 / (1 + mp.exp(-u))
    q = 1 - p
    U, D = p * q, q - p
    s = b - a
    if K == 0:
        return mp.log(s) + mp.log(U)
    if n1 + n2 == 0:
        if n0 == 1:
            return D
        return -2 * sig_poly(n0 - 1, U, D)
    if n0 == 0:
        S = 1 / s
        return sgn(n1 + K - 1) * fact(K - 1) * S**K
    return 0


# ---------------------------------------------------------------- the SymPy check

def multi(k):
    for n2 in range(k + 1):
        for n1 in range(k - n2 + 1):
            yield k - n1 - n2, n1, n2


def check_closed_forms():
    points = [(sp.Rational(2, 11), sp.Rational(-3, 7), sp.Rational(5, 3)),
              (sp.Rational(-1, 13), sp.Rational(-1, 2), sp.Rational(7, 4)),
              (sp.Rational(9, 10), sp.Rational(1, 9), sp.Rational(11, 8))]
    upoints = [(sp.Rational(3, 10), sp.Rational(-3, 7), sp.Rational(5, 3)),
               (sp.Rational(-5, 2), sp.Rational(-1, 2), sp.Rational(7, 4)),
               (sp.Rational(1, 7), sp.Rational(1, 9), sp.Rational(11, 8))]
    checked = 0
    for K in range(0, KMAX + 1):
        for n0, n1, n2 in multi(K):
            d = {}
            cases = [("g", G, cf_fwd, points), ("log g'", LJG, cf_fwd_lj, points),
                     ("h", H, cf_inv, upoints), ("log h'", LJH, cf_inv_lj, upoints)]
            for name, expr, cf, pts in cases:
                e = expr
                if n0:
                    e = sp.diff(e, X, n0)
                if n1:
                    e = sp.diff(e, P1, n1)
                if n2:
                    e = sp.diff(e, P2, n2)
                for xv, p1v, p2v in pts:
                    sym = sp.N(e.subs({X: xv, P1: p1v, P2: p2v}), DPS)
                    mine = cf(n0, n1, n2, mp.mpf(xv.p) / xv.q, mp.mpf(p1v.p) / p1v.q, mp.mpf(p2v.p) / p2v.q)
                    ref = mp.mpf(str(sym))
                    if abs(mine - ref) > mp.mpf(10) ** (-30) * max(1, abs(ref)):
                        sys.exit(f"closed form of {name} differs at order {(n0, n1, n2)}: "
                                 f"{mine} against SymPy {ref}")
                    checked += 1
    print(f"closed forms checked against SymPy: {checked} values, all agree to 1e-30", flush=True)


# ---------------------------------------------------------------- the points

PAIRS = [(0.0, 1.0), (-1.0, 3.0), (2.0, 5.0), (-1e6, 1e6), (1 - 2.0**-20, 1.0), (-1e-8, 1e-8)]


def v_points(rng, a, b):
    s = b - a
    pts = set()
    for k in range(2, 61, 3):  # log-uniform towards each bound
        pts.add(a + s * 2.0**-k)
        pts.add(b - s * 2.0**-k)
    for _ in range(16):
        pts.add(a + s * rng.random())
    return sorted(v for v in pts if a < v < b)


def u_points(rng):
    base = [0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 37.0, -37.0, 40.0, -40.0,
            100.0, -100.0, 300.0, -300.0, 700.0, -700.0, 750.0, -750.0]
    pts = set(base)
    for k in (5, 15, 30, 45, 60):
        pts.add(2.0**-k)
        pts.add(-(2.0**-k))
    for _ in range(12):
        pts.add(rng.uniform(-10.0, 10.0))
    return sorted(pts)


def extra_bits(which, x, p1, p2):
    """working bits beyond the default: near a bound the derivatives of log(g') cancel
    to about K log2(s / distance) bits (the log-Jacobian is built as the log of the
    SymPy derivative of g), so the bits grow with the octaves to the nearer bound"""
    b = _default_extra(which, x, p1, p2)
    if which in (0, 2):
        d = min(x - p1, p2 - x)
        if d > 0:
            b += int(10 * abs(math.log2(d)))
    return b


def main():
    mp.mp.dps = 60
    check_closed_forms()
    rng = random.Random(SEED)
    points_v, points_u = [], []
    for a, b in PAIRS:
        for v in v_points(rng, a, b):
            points_v.append((v, a, b))
        for u in u_points(rng):
            points_u.append((u, a, b))
    print(f"{len(points_v)} points in v, {len(points_u)} in u", flush=True)
    write_references("bounded", g=G, h=H, nparams=2, decreasing=lambda p1, p2: False,
                     points_v=points_v, points_u=points_u, distances=True, bounds=(P1, P2),
                     extra_bits=extra_bits, workers=2)


if __name__ == "__main__":
    main()
