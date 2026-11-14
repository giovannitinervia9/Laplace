"""Reference values for the unstructured structures M(u) (unstructured_pdmat<B>)
and the derivatives of M^-1 and log det M in u to order 6, computed with
mpmath at 640 bits from the definition of M.

Usage (from the repository root):  python -I tools/reference/pdmat_unstructured.py

File: tests/data/reference/pdmat_unstructured.csv, one row per entry, header
    map,p,u,quantity,order,idx,i,j,hi,lo
u is the point, its coordinates as float.hex joined by ';'; idx is the sorted
tuple of the derivative, joined by ';' ('-' at order 0). hi is the nearest
double to the value and lo the nearest double to the remainder.

Maps (code: bijector, M from u), u packed as in include/laplace/bijector/:
    0 log_cholesky: L_ii = exp(u_ii), L_ij = u_ij (i > j), M = L L^T;
      u_ij at i(i+1)/2 + j.
    1 matrix_log: S symmetric with S_ij = S_ji = u_ij (u_ij moves both),
      M = expm(S); u_ij at i(i+1)/2 + j.
    2 partial_corr: z_ab = tanh(u_ab), L_ab = z_ab prod_(l<b) sqrt(1 - z_al^2),
      L_aa = prod_(l<a) sqrt(1 - z_al^2), M = L L^T; u_ab (a > b) at
      a(a-1)/2 + b.
Quantities: 0 the inverse of M (mp.inverse; the entries i >= j); 1 log(det M)
(mp.det; one row, i = j = 0). Neither uses the closed forms of the C++ code.

Method. A derivative along a tuple whose distinct coordinates v have
multiplicities m_v is the tensor product of central differences
    prod_v sum_(k=0..m_v) (-1)^k C(m_v, k) f(u + (m_v/2 - k) h e_v) / h^m_v,
h = 2^-60. Every derivative is repeated with h = 2^-70, and the script stops
if the two disagree by more than 1e-30 times (1 + the largest absolute entry
of that quantity and tuple). The worst disagreement is printed.

Points: p in {1, 2, 3, 4} (partial_corr: p in {2, 3, 4}); for each, a random
u (entries N(0, 0.5^2), random.Random(20261113 + 100 map + p)) and a spread u
(diagonal +-2 alternating and off-diagonal 0.7 for maps 0 and 1; every u_ab
= +-1.5 alternating for map 2). Tuples: order 0; all tuples of orders 1 and
2; for orders 3 to 6 the pure tuple of every coordinate and 6 random mixed
sorted tuples.
"""
import math
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pdmat_unstructured.csv"
PREC = 640
KMAX = 6
H = mp.mpf(2) ** -60
H_CHECK = mp.mpf(2) ** -70
TOL = mp.mpf(10) ** -30

HEADER = "map,p,u,quantity,order,idx,i,j,hi,lo"


def n_free(m, p):
    return p * (p - 1) // 2 if m == 2 else p * (p + 1) // 2


def build(m, p, u):
    """M(u) as an mp.matrix."""
    if m == 1:
        S = mp.matrix(p, p)
        for i in range(p):
            for j in range(i + 1):
                S[i, j] = S[j, i] = u[i * (i + 1) // 2 + j]
        return mp.expm(S)
    L = mp.matrix(p, p)
    if m == 0:
        for i in range(p):
            for j in range(i + 1):
                x = u[i * (i + 1) // 2 + j]
                L[i, j] = mp.exp(x) if i == j else x
    else:
        for a in range(p):
            pr = mp.mpf(1)
            for b in range(a):
                z = mp.tanh(u[a * (a - 1) // 2 + b])
                L[a, b] = z * pr
                pr *= mp.sqrt(1 - z * z)
            L[a, a] = pr
    return L * L.T


def quantities(m, p, u):
    """[dict (i, j) -> M^-1_ij for i >= j, {(0, 0): log det M}]."""
    M = build(m, p, u)
    Mi = mp.inverse(M)
    return [{(i, j): Mi[i, j] for i in range(p) for j in range(i + 1)}, {(0, 0): mp.log(mp.det(M))}]


def derivative(m, p, u, tup, h, cache):
    """Both quantities differentiated along tup (a sorted tuple), as two dicts."""
    mult = {}
    for v in tup:
        mult[v] = mult.get(v, 0) + 1
    vars_ = sorted(mult)
    acc = [None, None]

    def rec(d, coef, shift):
        if d == len(vars_):
            key = tuple(sorted(shift.items()))
            if key not in cache:
                x = list(u)
                for v, s in shift.items():
                    x[v] += s * h / 2
                cache[key] = quantities(m, p, x)
            vals = cache[key]
            for q in range(2):
                if acc[q] is None:
                    acc[q] = {k: mp.mpf(0) for k in vals[q]}
                for k in vals[q]:
                    acc[q][k] += coef * vals[q][k]
            return
        v = vars_[d]
        mv = mult[v]
        for k in range(mv + 1):
            s = dict(shift)
            s[v] = mv - 2 * k
            rec(d + 1, coef * (-1) ** k * math.comb(mv, k), s)

    rec(0, mp.mpf(1), {})
    scale = h ** len(tup)
    return [{k: x / scale for k, x in a.items()} for a in acc]


def hex_pair(v):
    if v == 0:
        return "0x0.0p+0", "0x0.0p+0"
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return float.hex(hi), float.hex(lo)


def points(m, p):
    n = n_free(m, p)
    rng = random.Random(20261113 + 100 * m + p)
    out = [[rng.gauss(0.0, 0.5) for _ in range(n)]]
    if m == 2:
        out.append([1.5 if k % 2 == 0 else -1.5 for k in range(n)])
    else:
        s = []
        for i in range(p):
            for j in range(i + 1):
                s.append((2.0 if i % 2 == 0 else -2.0) if i == j else 0.7)
        out.append(s)
    return out


def tuples(n, rng):
    out = [()]
    for K in (1, 2):
        out += list(combinations_with_replacement(range(n), K))
    for K in range(3, KMAX + 1):
        out += [(v,) * K for v in range(n)]
        mixed = set()
        while n > 1 and len(mixed) < 6:
            t = tuple(sorted(rng.randrange(n) for _ in range(K)))
            if len(set(t)) > 1:
                mixed.add(t)
        out += sorted(mixed)
    return out


def main():
    mp.mp.prec = PREC
    t0 = time.time()
    rows = []
    worst_ratio, worst_where = mp.mpf(0), None
    for m in range(3):
        for p in (1, 2, 3, 4):
            if m == 2 and p == 1:
                continue
            n = n_free(m, p)
            for pt, uf in enumerate(points(m, p)):
                u = [mp.mpf(x) for x in uf]
                ustr = ";".join(float.hex(x) for x in uf)
                c1, c2 = {}, {}
                rng = random.Random(7 + 1000 * m + 10 * p + pt)
                for tup in tuples(n, rng):
                    d1 = derivative(m, p, u, tup, H, c1)
                    d2 = derivative(m, p, u, tup, H_CHECK, c2)
                    for q in range(2):
                        scale = 1 + max(abs(x) for x in d1[q].values())
                        for key in sorted(d1[q]):
                            diff = abs(d1[q][key] - d2[q][key])
                            if diff > TOL * scale:
                                raise SystemExit("h-check failed: map %d p %d point %d tuple %r quantity %d key %r: %s"
                                                 % (m, p, pt, tup, q, key, mp.nstr(diff / scale, 5)))
                            if diff / scale > worst_ratio:
                                worst_ratio, worst_where = diff / scale, (m, p, pt, tup, q, key)
                            hi, lo = hex_pair(d1[q][key])
                            rows.append("%d,%d,%s,%d,%d,%s,%d,%d,%s,%s" % (
                                m, p, ustr, q, len(tup), ";".join(map(str, tup)) if tup else "-",
                                key[0], key[1], hi, lo))
                print("map %d p %d point %d: %d rows, %.1f s" % (m, p, pt, len(rows), time.time() - t0))
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
