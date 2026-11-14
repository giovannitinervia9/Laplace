"""Reference values for the factor-analytic structure M(theta) = Lambda Lambda^T
+ Psi (factor_analytic_pdmat) and the derivatives of M, M^-1 and log det M in
theta to order 6, computed with mpmath at 640 bits from the definition of M.

Usage (from the repository root):  python -I tools/reference/pdmat_factor_analytic.py

File: tests/data/reference/pdmat_factor_analytic.csv, one row per entry, header
    p,d,theta,quantity,order,idx,i,j,hi,lo
theta is the point, its coordinates as float.hex joined by ';'; idx is the
sorted tuple of the derivative, joined by ';' ('-' at order 0). hi is the
nearest double to the value and lo the nearest double to the remainder.

theta: the loadings of the lower trapezoid of Lambda (p x d) by rows, row i
holding Lambda_i0, ..., Lambda_i,min(i, d-1), with Lambda_ii = exp(theta) on
the diagonal and the others equal to their theta; then t_0, ..., t_(p-1),
Psi = diag(exp(2 t_i)). M = Lambda Lambda^T + Psi.
Quantities: 0 M; 1 the inverse of M (mp.inverse); 2 log(det M) (mp.det; one
row, i = j = 0). The entries i >= j. None uses Woodbury's formulas or the
closed forms of the C++ code.

Method. A derivative along a tuple whose distinct coordinates v have
multiplicities m_v is the tensor product of central differences
    prod_v sum_(k=0..m_v) (-1)^k C(m_v, k) f(theta + (m_v/2 - k) h e_v) / h^m_v,
h = 2^-60. Every derivative is repeated with h = 2^-70, and the script stops
if the two disagree by more than 1e-30 times (1 + the largest absolute entry
of that quantity and tuple). The worst disagreement is printed.

Shapes (p, d): (1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (5, 2) (d = 0 is
diagonal_pdmat, checked against it in the test). For each, a random theta (entries N(0, 0.5^2),
random.Random(20261114 + 10 p + d)) and a spread theta (off-diagonal loadings
+-1.5 alternating, diagonal loadings 0.4, t_i = -2 for even i and 1 for odd
i: psi from e^-4 to e^2). Tuples: order 0; all tuples of order 1; all of
order 2 when they are at most 40, else 40 drawn at random; for
orders 3 to 6 the pure tuple of every coordinate, 6 random mixed sorted
tuples and 4 random mixed sorted tuples within one row of M (the loadings and
the t of that row), or as many as 1000 draws find.
"""
import math
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pdmat_factor_analytic.csv"
PREC = 640
KMAX = 6
H = mp.mpf(2) ** -60
H_CHECK = mp.mpf(2) ** -70
TOL = mp.mpf(10) ** -30
SHAPES = [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (5, 2)]

HEADER = "p,d,theta,quantity,order,idx,i,j,hi,lo"


def layout(p, d):
    """(row, column) of each loading, then (row, -1) for each t."""
    out = [(i, j) for i in range(p) for j in range(min(i + 1, d))]
    return out + [(i, -1) for i in range(p)]


def quantities(p, d, theta):
    """[M, M^-1 (dicts (i, j) -> value, i >= j), {(0, 0): log det M}]."""
    lay = layout(p, d)
    L = mp.matrix(p, max(d, 1))
    M = mp.matrix(p, p)
    for x, (i, j) in zip(theta, lay):
        if j < 0:
            M[i, i] += mp.exp(2 * x)
        else:
            L[i, j] = mp.exp(x) if i == j else x
    M += L * L.T
    Mi = mp.inverse(M)
    lower = [(i, j) for i in range(p) for j in range(i + 1)]
    return [{k: M[k] for k in lower}, {k: Mi[k] for k in lower}, {(0, 0): mp.log(mp.det(M))}]


def derivative(p, d, theta, tup, h, cache):
    """The quantities differentiated along tup (a sorted tuple), as dicts."""
    mult = {}
    for v in tup:
        mult[v] = mult.get(v, 0) + 1
    vars_ = sorted(mult)
    acc = [None, None, None]

    def rec(k0, coef, shift):
        if k0 == len(vars_):
            key = tuple(sorted(shift.items()))
            if key not in cache:
                x = list(theta)
                for v, s in shift.items():
                    x[v] += s * h / 2
                cache[key] = quantities(p, d, x)
            vals = cache[key]
            for q in range(3):
                if acc[q] is None:
                    acc[q] = {k: mp.mpf(0) for k in vals[q]}
                for k in vals[q]:
                    acc[q][k] += coef * vals[q][k]
            return
        v = vars_[k0]
        mv = mult[v]
        for k in range(mv + 1):
            s = dict(shift)
            s[v] = mv - 2 * k
            rec(k0 + 1, coef * (-1) ** k * math.comb(mv, k), s)

    rec(0, mp.mpf(1), {})
    scale = h ** len(tup)
    return [{k: x / scale for k, x in a.items()} for a in acc]


def hex_pair(v):
    if v == 0:
        return "0x0.0p+0", "0x0.0p+0"
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return float.hex(hi), float.hex(lo)


def points(p, d):
    lay = layout(p, d)
    rng = random.Random(20261114 + 10 * p + d)
    out = [[rng.gauss(0.0, 0.5) for _ in lay]]
    s, n_off = [], 0
    for i, j in lay:
        if j < 0:
            s.append(-2.0 if i % 2 == 0 else 1.0)
        elif i == j:
            s.append(0.4)
        else:
            s.append(1.5 if n_off % 2 == 0 else -1.5)
            n_off += 1
    out.append(s)
    return out


def tuples(p, d, rng):
    lay = layout(p, d)
    n = len(lay)
    out = [()] + [(v,) for v in range(n)]
    pairs = list(combinations_with_replacement(range(n), 2))
    out += pairs if len(pairs) <= 40 else sorted(rng.sample(pairs, 40))
    rows = [[v for v, (i, _) in enumerate(lay) if i == r] for r in range(p)]
    rows = [r for r in rows if len(r) > 1]
    for K in range(3, KMAX + 1):
        out += [(v,) * K for v in range(n)]
        mixed, same = set(), set()
        for _ in range(1000):  # small q has fewer than 6 mixed tuples
            if len(mixed) == 6:
                break
            t = tuple(sorted(rng.randrange(n) for _ in range(K)))
            if len(set(t)) > 1:
                mixed.add(t)
        for _ in range(1000):
            if not rows or len(same) == 4:
                break
            r = rng.choice(rows)
            t = tuple(sorted(rng.choice(r) for _ in range(K)))
            if len(set(t)) > 1:
                same.add(t)
        out += sorted(mixed) + sorted(same - mixed)
    return out


def main():
    mp.mp.prec = PREC
    t0 = time.time()
    rows = []
    worst_ratio, worst_where = mp.mpf(0), None
    for p, d in SHAPES:
        for pt, tf in enumerate(points(p, d)):
            theta = [mp.mpf(x) for x in tf]
            tstr = ";".join(float.hex(x) for x in tf)
            c1, c2 = {}, {}
            rng = random.Random(11 + 100 * p + 10 * d + pt)
            for tup in tuples(p, d, rng):
                d1 = derivative(p, d, theta, tup, H, c1)
                d2 = derivative(p, d, theta, tup, H_CHECK, c2)
                for q in range(3):
                    scale = 1 + max(abs(x) for x in d1[q].values())
                    for key in sorted(d1[q]):
                        diff = abs(d1[q][key] - d2[q][key])
                        if diff > TOL * scale:
                            raise SystemExit("h-check failed: p %d d %d point %d tuple %r quantity %d key %r: %s"
                                             % (p, d, pt, tup, q, key, mp.nstr(diff / scale, 5)))
                        if diff / scale > worst_ratio:
                            worst_ratio, worst_where = diff / scale, (p, d, pt, tup, q, key)
                        hi, lo = hex_pair(d1[q][key])
                        rows.append("%d,%d,%s,%d,%d,%s,%d,%d,%s,%s" % (
                            p, d, tstr, q, len(tup), ";".join(map(str, tup)) if tup else "-",
                            key[0], key[1], hi, lo))
            print("p %d d %d point %d: %d rows, %.1f s" % (p, d, pt, len(rows), time.time() - t0))
            sys.stdout.flush()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(r + "\n")
    print("rows written: %d" % len(rows))
    print("worst disagreement relative to (1 + max entry): %s at %s" % (mp.nstr(worst_ratio, 5), worst_where))
    print("run time: %.1f s, output %s" % (time.time() - t0, OUT))


if __name__ == "__main__":
    main()
