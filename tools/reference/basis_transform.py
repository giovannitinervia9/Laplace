"""Reference eigenvalues for the basis transforms (include/laplace/basis/transform.hpp).

Usage (from the repository root):  python -I tools/reference/basis_transform.py

Cubic B-spline bases with uniform clamped knots on [0, 1] (K columns, K - 3
knot intervals), penalty S = G_mm (the operator D^m) for m = 2, 3, Gram matrix
G = G_00, with and without the zero-integral constraint C beta = 0,
C_j = int b_j = (t_(j+4) - t_j)/4.

G and S are computed exactly (fractions.Fraction): on each knot interval the
B-splines are polynomials obtained from the Cox-de Boor recursion on
polynomials, and their products are integrated exactly; nothing is taken from
the C++ code. With Z an exact basis of the null space of C (Z = I without the
constraint; with it, Z = [I; -c_(0..K-2)/c_(K-1)]), the transform's eigenvalues
are those of the pencil (Z' S Z, Z' G Z), invariant under the choice of Z.
They are computed with mpmath at 200 bits (Cholesky of Z' G Z, then eigsy of
L^-1 Z' S Z L^-T) and rounded to double; the null eigenvalues come out below
1e-55 of the largest and are written as 0.

File (tests/data/reference/basis_transform.csv):
  case,K,m,constrained,j,lambda      lambda in increasing order, float.hex
"""
import os
from fractions import Fraction as F

import mpmath as mp

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
mp.mp.prec = 200
D = 3


def knots(K):
    n = K - D
    return [F(0)] * (D + 1) + [F(q, n) for q in range(1, n)] + [F(1)] * (D + 1)


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def pmul(a, b):
    if not a or not b:
        return []
    o = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            o[i + j] += x * y
    return o


def pder(a, k):
    for _ in range(k):
        a = [i * a[i] for i in range(1, len(a))]
    return a


def pint(a, lo, hi):
    return sum(c * (hi ** (i + 1) - lo ** (i + 1)) / (i + 1) for i, c in enumerate(a))


def interval_polys(t, K, s):
    """B_(i,D) on [t_s, t_(s+1)] as polynomials in x, for i = s - D .. s."""
    b = {i: ([F(1)] if i == s else []) for i in range(len(t) - 1)}
    for p in range(1, D + 1):
        nb = {}
        for i in range(len(t) - p - 1):
            r = []
            d1 = t[i + p] - t[i]
            if d1 != 0 and b[i]:
                r = padd(r, pmul([-t[i] / d1, 1 / d1], b[i]))
            d2 = t[i + p + 1] - t[i + 1]
            if d2 != 0 and b[i + 1]:
                r = padd(r, pmul([t[i + p + 1] / d2, -1 / d2], b[i + 1]))
            nb[i] = r
        b = nb
    return {i: b[i] for i in range(s - D, s + 1)}


def gram(K, k, l):
    t = knots(K)
    g = [[F(0)] * K for _ in range(K)]
    for s in range(D, K):
        if t[s] == t[s + 1]:
            continue
        p = interval_polys(t, K, s)
        for i, pi in p.items():
            for j, pj in p.items():
                g[i][j] += pint(pmul(pder(pi, k), pder(pj, l)), t[s], t[s + 1])
    return g


def mpf(x):
    return mp.mpf(x.numerator) / x.denominator


def eigenvalues(K, m, constrained):
    t = knots(K)
    G, S = gram(K, 0, 0), gram(K, m, m)
    if constrained:
        c = [(t[j + D + 1] - t[j]) / (D + 1) for j in range(K)]
        Z = [[F(0)] * (K - 1) for _ in range(K)]
        for i in range(K - 1):
            Z[i][i] = F(1)
            Z[K - 1][i] = -c[i] / c[K - 1]
    else:
        Z = [[F(int(i == j)) for j in range(K)] for i in range(K)]
    q = len(Z[0])

    def cong(A):
        AZ = [[sum(A[i][l] * Z[l][j] for l in range(K) if Z[l][j] != 0) for j in range(q)] for i in range(K)]
        return mp.matrix([[mpf(sum(Z[l][i] * AZ[l][j] for l in range(K) if Z[l][i] != 0)) for j in range(q)]
                          for i in range(q)])

    A, B = cong(S), cong(G)
    Li = mp.inverse(mp.cholesky(B))
    w = sorted(mp.eigsy(Li * A * Li.T, eigvals_only=True))
    lmax = w[-1]
    return [mp.mpf(0) if abs(x) < mp.mpf(10) ** -55 * lmax else x for x in w]


def main():
    rows = ["case,K,m,constrained,j,lambda"]
    case = 0
    for K in (10, 40):
        for m in (2, 3):
            for constrained in (0, 1):
                w = eigenvalues(K, m, constrained)
                for j, x in enumerate(w):
                    rows.append(f"{case},{K},{m},{constrained},{j},{float(x).hex()}")
                print(f"case {case}: K={K} m={m} constrained={constrained} lambda_max={float(w[-1]):.6g}")
                case += 1
    path = os.path.join(OUT, "basis_transform.csv")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
