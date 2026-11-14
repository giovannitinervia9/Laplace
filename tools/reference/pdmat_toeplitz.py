"""Reference values for toeplitz_pdmat, the p x p Toeplitz correlation matrix
M(theta) parametrized by its partial autocorrelations phi_k = tanh(theta_(k-1)),
and the derivatives of M, M^-1, L and log det M in theta to order 6, computed
with mpmath at 640 bits from the definition.

Usage (from the repository root):  python -I tools/reference/pdmat_toeplitz.py

File: tests/data/reference/pdmat_toeplitz.csv, one row per entry, header
    p,theta,quantity,order,idx,i,j,hi,lo
theta is the point, its coordinates as float.hex joined by ';' ('-' for p = 1);
idx is the sorted tuple of the derivative, joined by ';' ('-' at order 0). hi
is the nearest double to the value and lo the nearest double to the remainder.

Definition. The partial autocorrelation of lag k is the last coefficient of
the best linear predictor of x_k from x_(k-1), ..., x_0, that is the last entry
of the solution c of T_k c = (m_1, ..., m_k), with T_k the k x k Toeplitz matrix
of m_0, ..., m_(k-1). The entry is affine in m_k, so m_k is found from two
linear solves (mp.lu_solve): with b the right side at m_k = 0,
    m_k = (phi_k - (T_k^-1 b)_k) / (T_k^-1)_kk.
The Durbin-Levinson recursion of the C++ code is not used.
Quantities: 0 M; 1 M^-1 (mp.inverse); 2 the lower Cholesky factor L
(mp.cholesky); 3 log det M (mp.det; one row, i = j = 0). Quantities 0 to 2
write the entries with i >= j.

Method. A derivative along a tuple whose distinct coordinates v have
multiplicities m_v is the tensor product of central differences
    prod_v sum_(k=0..m_v) (-1)^k C(m_v, k) f(theta + (m_v/2 - k) h e_v) / h^m_v,
h = 2^-60. Every derivative is repeated with h = 2^-70, and the script stops
if the two disagree by more than 1e-30 times (1 + the largest absolute entry
of that quantity and tuple). The worst disagreement is printed.

Points: p in {1, 2, 3, 4, 6}; for p >= 2 a random theta (entries N(0, 0.6^2),
random.Random(20261114 + p)), a spread theta (+-1.5 alternating) and a steep
theta (theta_0 = 6, then -2.5, +2.5, ... alternating). Condition numbers of M
at p = 2, 3, 4, 6: random 1.2, 1.3, 34, 16; spread 20, 410, 8.2e3, 3.1e6;
steep 1.6e5, 2.7e7, 4.5e9, 1.2e14. Tuples: order 0; all tuples of orders 1
and 2; for orders 3 to 6 the pure tuple of every coordinate and 6 random mixed
sorted tuples (all of them when there are fewer).
"""
import math
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pdmat_toeplitz.csv"
PREC = 640
KMAX = 6
H = mp.mpf(2) ** -60
H_CHECK = mp.mpf(2) ** -70
TOL = mp.mpf(10) ** -30

HEADER = "p,theta,quantity,order,idx,i,j,hi,lo"


def autocorrelations(theta):
    """m_0..m_(p-1) from the partial autocorrelations tanh(theta), by definition."""
    m = [mp.mpf(1)]
    for k in range(1, len(theta) + 1):
        T = mp.matrix(k, k)
        for i in range(k):
            for j in range(k):
                T[i, j] = m[abs(i - j)]
        b = mp.matrix([m[i + 1] if i + 1 < k else 0 for i in range(k)])
        e = mp.matrix([1 if i == k - 1 else 0 for i in range(k)])
        x0 = mp.lu_solve(T, b)
        x1 = mp.lu_solve(T, e)
        m.append((mp.tanh(theta[k - 1]) - x0[k - 1]) / x1[k - 1])
    return m


def quantities(p, theta):
    """[M, M^-1, L as dicts (i, j) -> value for i >= j, {(0, 0): log det M}]."""
    m = autocorrelations(theta)
    M = mp.matrix(p, p)
    for i in range(p):
        for j in range(p):
            M[i, j] = m[abs(i - j)]
    Mi = mp.inverse(M)
    L = mp.cholesky(M)
    low = lambda X: {(i, j): X[i, j] for i in range(p) for j in range(i + 1)}
    return [low(M), low(Mi), low(L), {(0, 0): mp.log(mp.det(M))}]


def derivative(p, theta, tup, h, cache):
    """The four quantities differentiated along tup (a sorted tuple)."""
    mult = {}
    for v in tup:
        mult[v] = mult.get(v, 0) + 1
    vars_ = sorted(mult)
    acc = [None] * 4

    def rec(d, coef, shift):
        if d == len(vars_):
            key = tuple(sorted(shift.items()))
            if key not in cache:
                x = list(theta)
                for v, s in shift.items():
                    x[v] += s * h / 2
                cache[key] = quantities(p, x)
            vals = cache[key]
            for q in range(4):
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


def points(p):
    n = p - 1
    if n == 0:
        return [[]]
    rng = random.Random(20261114 + p)
    out = [[rng.gauss(0.0, 0.6) for _ in range(n)]]
    out.append([1.5 if k % 2 == 0 else -1.5 for k in range(n)])
    out.append([6.0] + [-2.5 if k % 2 == 1 else 2.5 for k in range(1, n)])
    return out


def tuples(n, rng):
    out = [()]
    if n == 0:
        return out
    for K in (1, 2):
        out += list(combinations_with_replacement(range(n), K))
    for K in range(3, KMAX + 1):
        out += [(v,) * K for v in range(n)]
        mixed = set()
        # at most 6, and never more than there are
        want = min(6, math.comb(n + K - 1, K) - n)
        while len(mixed) < want:
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
    for p in (1, 2, 3, 4, 6):
        for pt, tf in enumerate(points(p)):
            theta = [mp.mpf(x) for x in tf]
            tstr = ";".join(float.hex(x) for x in tf) if tf else "-"
            c1, c2 = {}, {}
            rng = random.Random(11 + 10 * p + pt)
            for tup in tuples(p - 1, rng):
                d1 = derivative(p, theta, tup, H, c1)
                d2 = derivative(p, theta, tup, H_CHECK, c2)
                for q in range(4):
                    scale = 1 + max(abs(x) for x in d1[q].values())
                    for key in sorted(d1[q]):
                        diff = abs(d1[q][key] - d2[q][key])
                        if diff > TOL * scale:
                            raise SystemExit("h-check failed: p %d point %d tuple %r quantity %d key %r: %s"
                                             % (p, pt, tup, q, key, mp.nstr(diff / scale, 5)))
                        if diff / scale > worst_ratio:
                            worst_ratio, worst_where = diff / scale, (p, pt, tup, q, key)
                        hi, lo = hex_pair(d1[q][key])
                        rows.append("%d,%s,%d,%d,%s,%d,%d,%s,%s" % (
                            p, tstr, q, len(tup), ";".join(map(str, tup)) if tup else "-",
                            key[0], key[1], hi, lo))
            print("p %d point %d: %d rows, %.1f s" % (p, pt, len(rows), time.time() - t0))
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
