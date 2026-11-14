"""Reference values for the weighted sum of penalties M(theta) = sum_j e^theta_j S_j
(weighted_sum_psdmat): the log pseudo-determinant log|M|_+ and its derivatives
in theta to order 6, computed with mpmath at 256 bits from the definition.

Usage (from the repository root):  python -I tools/reference/pdmat_weighted_sum.py

Files, in tests/data/reference/:
  pdmat_weighted_sum_cases.csv   case,p,m,rank,j,i,k,value
      the matrices S_j, every entry (i, k) of the lower triangle, k <= i, as
      float.hex; every S_j is exact in double (integers or integers times
      powers of two), so its null space is exact.
  pdmat_weighted_sum.csv         case,theta,order,idx,hi,lo
      theta as float.hex joined by ';'; idx the sorted tuple joined by ';'
      ('-' at order 0); hi the nearest double to the value, lo the nearest
      double to the remainder.

Method. N, an exact rational basis of the null space of sum_j S_j, and Z, an
exact rational basis of its orthogonal complement, are computed by SymPy.
The null space of a sum of positive semidefinite matrices is the
intersection of theirs, so Z spans the range of M at every theta and
    log|M|_+ = log det(Z^T M Z) - log det(Z^T Z).
With A = Z^T M Z and B_j = A^-1 (e^theta_j Z^T S_j Z) (similar to
e^theta_j M^+ S_j on the range), d A / dtheta_j = e^theta_j Z^T S_j Z on any
tuple of one repeated index and 0 otherwise, so the derivative of log det A
along a sorted tuple of K positions is
    sum over the set partitions pi of the positions whose blocks each repeat
    one index, (-1)^(|pi| - 1) times the sum over the (|pi| - 1)! cyclic
    orders of the blocks of tr(B_(j_1) ... B_(j_|pi|)).
A and B_j are formed in exact rational arithmetic up to the factors e^theta_j
and evaluated at 256 bits; nothing here uses the levels, the scaling or the
eigendecompositions of the C++ code. The partition formula itself is checked,
once, against tensor central differences of log det A at 640 bits (h = 2^-60
and 2^-70 must agree to 1e-30) on the case lowrank, every sorted tuple of
orders 1 to 6, at two points.

Cases (name: m components, p, construction):
  te2:        P6 (x) I5, I6 (x) P5; P_k the integrated squared second
              derivative of the k cubic B-splines on the knots 0..k+3 over
              [3, k], times the least common multiple of its denominators
  te3:        D3 (x) I3 (x) I4, I3 (x) D3 (x) I4, I3 (x) I3 (x) D4; D_k the
              second-difference penalty D^T D of k coefficients
  lowrank:    three A A^T, A 8 x 3, 8 x 2, 8 x 2 with entries in -4..4
  lowrank_ill: the same with the columns of A scaled by 1, 2^-4, 2^-8 (or
              by 1, 2^-6), condition of a component up to 2^16
  ridge:      P10 and I10 (full rank)
  single:     P8 alone
Points: theta_0 = s ln 10, theta_1 = 0, theta_2 = (s/2) ln 10 (m = 3), with
s in -20, -12, -6, 0, 3, 6, 9, 12, 15, 20 (te3: -12, 0, 6, 12, 20), and one
point N(0, 2^2) (random.Random(20261114)). Tuples: every sorted tuple of
orders 0 to 6.
"""
import math
import random
import sys
import time
from fractions import Fraction
from itertools import combinations_with_replacement, permutations
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
OUT = ROOT / "pdmat_weighted_sum.csv"
OUT_CASES = ROOT / "pdmat_weighted_sum_cases.csv"
PREC = 256
KMAX = 6


def bspline_penalty(k):
    """Integrated squared second derivative of the k cubic B-splines on the
    knots 0..k+3 over [3, k], scaled to integers (exact rational first)."""
    x = sp.Symbol("x")
    knots = list(range(k + 4))
    basis = sp.bspline_basis_set(3, knots, x)
    d2 = [sp.diff(b, x, 2) for b in basis]
    P = sp.zeros(k, k)
    for a in range(k):
        for b in range(a, k):
            P[a, b] = P[b, a] = sp.integrate(sp.piecewise_fold(d2[a] * d2[b]), (x, 3, k))
    den = sp.ilcm(*[sp.fraction(sp.Rational(e))[1] for e in P])
    return P * den


def diff_penalty(k):
    D = sp.zeros(k - 2, k)
    for i in range(k - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1, -2, 1
    return D.T * D


def kron(*Ms):
    out = Ms[0]
    for M in Ms[1:]:
        out = sp.kronecker_product(out, M)
    return out


def lowrank(cols, scales, seed):
    rng = random.Random(seed)
    out = []
    for c, sc in zip(cols, scales):
        A = sp.Matrix(8, c, lambda i, j: rng.randint(-4, 4) * sp.Rational(1, 2 ** sc[j]))
        out.append(A * A.T)
    return out


def cases():
    P6, P5, P10, P8 = (bspline_penalty(k) for k in (6, 5, 10, 8))
    I = sp.eye
    return {
        "te2": [kron(P6, I(5)), kron(I(6), P5)],
        "te3": [kron(diff_penalty(3), I(3), I(4)), kron(I(3), diff_penalty(3), I(4)), kron(I(3), I(3), diff_penalty(4))],
        "lowrank": lowrank([3, 2, 2], [[0, 0, 0], [0, 0], [0, 0]], 1),
        "lowrank_ill": lowrank([3, 2, 2], [[0, 4, 8], [0, 6], [0, 3]], 2),
        "ridge": [P10, I(10)],
        "single": [P8],
    }


def range_basis(Ss):
    total = Ss[0]
    for S in Ss[1:]:
        total = total + S
    N = total.nullspace()
    if not N:
        return sp.eye(total.rows)
    Nm = sp.Matrix.hstack(*N)
    return sp.Matrix.hstack(*(Nm.T.nullspace()))


def partition_words(tup):
    """{word: coefficient}, the words reduced to their least rotation."""
    K = len(tup)
    coef = {}

    def rec(q, blocks):
        if q == K:
            labels = [tup[b[0]] for b in blocks]
            nb = len(blocks)
            sign = 1 if nb % 2 == 1 else -1
            for perm in permutations(range(1, nb)):
                word = [labels[0]] + [labels[i] for i in perm]
                w = min(tuple(word[s:] + word[:s]) for s in range(nb))
                coef[w] = coef.get(w, 0) + sign
            return
        for b in blocks:
            if tup[b[0]] == tup[q]:
                b.append(q)
                rec(q + 1, blocks)
                b.pop()
        blocks.append([q])
        rec(q + 1, blocks)
        blocks.pop()

    rec(0, [])
    return {w: c for w, c in coef.items() if c != 0}


def evaluator(Zs, theta):
    """log det A and the trace of a word, at the current precision."""
    lam = [mp.exp(t) for t in theta]
    A = sum((l * S for l, S in zip(lam[1:], Zs[1:])), lam[0] * Zs[0])
    Ai = mp.inverse(A)
    B = [Ai * (l * S) for l, S in zip(lam, Zs)]
    cache = {}

    def trace(word):
        if word not in cache:
            P = B[word[0]]
            for j in word[1:-1]:
                P = P * B[j]
            if len(word) == 1:
                cache[word] = sum(P[i, i] for i in range(P.rows))
            else:
                L = B[word[-1]]
                cache[word] = mp.fsum(P[i, k] * L[k, i] for i in range(P.rows) for k in range(P.rows))
        return cache[word]

    return mp.log(mp.det(A)), trace


def tuples(m):
    for K in range(1, KMAX + 1):
        yield from combinations_with_replacement(range(m), K)


def hex_pair(v):
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return float.hex(hi), float.hex(lo)


def check_formula(Zs, m):
    """Partition formula against tensor central differences at 640 bits."""
    worst = 0
    with mp.workprec(640):
        Zm = [mp.matrix(Z.tolist()) for Z in Zs]

        def logdet(th):
            A = sum((mp.exp(t) * S for t, S in zip(th[1:], Zm[1:])), mp.exp(th[0]) * Zm[0])
            return mp.log(mp.det(A))

        for theta in ([mp.mpf("0.3"), mp.mpf("-0.7"), mp.mpf("1.1")], [mp.mpf(4), mp.mpf(0), mp.mpf(-3)]):
            _, trace = evaluator(Zm, theta)
            for tup in tuples(m):
                f = sum(c * trace(w) for w, c in partition_words(tup).items())
                vals = []
                for h in (mp.mpf(2) ** -60, mp.mpf(2) ** -70):
                    mult = {v: tup.count(v) for v in set(tup)}
                    keys = sorted(mult)
                    tot = mp.mpf(0)

                    def rec(i, coef, shift):
                        nonlocal tot
                        if i == len(keys):
                            th = [theta[v] + shift.get(v, 0) for v in range(m)]
                            tot += coef * logdet(th)
                            return
                        v, n = keys[i], mult[keys[i]]
                        for k in range(n + 1):
                            s = dict(shift)
                            s[v] = (mp.mpf(n) / 2 - k) * h
                            rec(i + 1, coef * (-1) ** k * math.comb(n, k) / h ** n, s)

                    rec(0, mp.mpf(1), {})
                    vals.append(tot)
                if abs(vals[0] - vals[1]) > mp.mpf(10) ** -30 * (1 + abs(f)):
                    sys.exit(f"differences disagree on {tup}")
                worst = max(worst, abs(f - vals[1]) / (1 + abs(f)))
    print(f"partition formula vs differences: worst relative {mp.nstr(worst, 3)}")
    if worst > mp.mpf(10) ** -25:
        sys.exit("the partition formula disagrees with the differences")


def main():
    mp.mp.prec = PREC
    t0 = time.time()
    cs = cases()
    rng = random.Random(20261114)
    rows, case_rows = [], []
    for name, Ss in cs.items():
        m, p = len(Ss), Ss[0].rows
        Z = range_basis(Ss)
        r = Z.cols
        for j, S in enumerate(Ss):
            for i in range(p):
                for k in range(i + 1):
                    v = Fraction(int(sp.fraction(S[i, k])[0]), int(sp.fraction(S[i, k])[1]))
                    f = float(v)
                    assert Fraction(f) == v, "a matrix entry is not exact in double"
                    case_rows.append(f"{name},{p},{m},{r},{j},{i},{k},{float.hex(f)}")
        Zs = [Z.T * S * Z for S in Ss]
        if name == "lowrank":
            check_formula(Zs, m)
        logZ = mp.log(mp.det(mp.matrix((Z.T * Z).tolist())))
        Zm = [mp.matrix(M.tolist()) for M in Zs]
        spreads = [-12, 0, 6, 12, 20] if name == "te3" else [-20, -12, -6, 0, 3, 6, 9, 12, 15, 20]
        points = []
        for s in spreads:
            th = [s * math.log(10), 0.0, s / 2 * math.log(10)][:m]
            points.append(th)
        points.append([rng.gauss(0, 2) for _ in range(m)])
        for th in points:
            theta = [mp.mpf(t) for t in th]  # the doubles exactly
            key = ";".join(float.hex(t) for t in th)
            ld, trace = evaluator(Zm, theta)
            hi, lo = hex_pair(ld - logZ)
            rows.append(f"{name},{key},0,-,{hi},{lo}")
            for tup in tuples(m):
                v = mp.fsum(c * trace(w) for w, c in partition_words(tup).items())
                hi, lo = hex_pair(v)
                rows.append(f"{name},{key},{len(tup)},{';'.join(map(str, tup))},{hi},{lo}")
        print(f"{name}: p = {p}, m = {m}, rank {r}, {len(points)} points, {time.time() - t0:.0f} s")
    OUT.write_text("case,theta,order,idx,hi,lo\n" + "\n".join(rows) + "\n")
    OUT_CASES.write_text("case,p,m,rank,j,i,k,value\n" + "\n".join(case_rows) + "\n")
    print(f"{len(rows)} values, {len(case_rows)} matrix entries")


if __name__ == "__main__":
    main()
