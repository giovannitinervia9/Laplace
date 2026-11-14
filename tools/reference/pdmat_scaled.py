"""Reference values for unit_diagonal_scaled_pdmat<R>, M = D R D with
D = diag(exp(s_i)), and its derivatives in theta = (s, r) to order 6,
computed with mpmath at 640 bits from the definition of M.

Usage (from the repository root):  python -I tools/reference/pdmat_scaled.py

File: tests/data/reference/pdmat_scaled.csv, one row per entry, header
    structure,p,common,theta,quantity,order,idx,i,j,hi,lo
theta is the point, its coordinates as float.hex joined by ';'; idx is the
sorted tuple of the derivative, joined by ';' ('-' at order 0). hi is the
nearest double to the value and lo the nearest double to the remainder.

Structures R (code: name, R from its parameters r):
    0 compound_symmetry: a = -1/(p-1), rho = a + (1 - a)/(1 + exp(-r)),
      R = (1 - rho) I + rho ones ones^T; R = [[1]] for p = 1.
    1 ar1: R_ij = tanh(r)^|i-j|.
    2 partial_corr (unstructured_corr_pdmat): z_ab = tanh(r_ab),
      L_ab = z_ab prod_(l<b) sqrt(1 - z_al^2), L_aa = prod_(l<a) sqrt(1 - z_al^2),
      R = L L^T; r_ab (a > b) at a(a-1)/2 + b.
theta = (s_0..s_(p-1), r), or (s, r) with common = 1 (D = exp(s) I).

Quantities: 0 M; 1 its inverse (mp.inverse); 2 its lower Cholesky factor
(mp.cholesky); 3 log(det M) (mp.det); 4 the log-Jacobian of theta -> the
lower triangle of M with its diagonal, log|det J| with J computed by central
differences at 2000 bits, h = 2^-600 (structure 2 with common = 0 only, the
one case with as many parameters as entries). Matrices: the entries i >= j;
scalars: one row, i = j = 0. None uses the closed forms of the C++ code.

Method. A derivative along a tuple whose distinct coordinates v have
multiplicities m_v is the tensor product of central differences
    prod_v sum_(k=0..m_v) (-1)^k C(m_v, k) f(theta + (m_v/2 - k) h e_v) / h^m_v,
h = 2^-60. Every derivative is repeated with h = 2^-70, and the script stops
if the two disagree by more than 1e-30 times (1 + the largest absolute entry
of that quantity and tuple). The worst disagreement is printed.

Points: structures 0 and 1 with p in {1, 3}, structure 2 with p in {2, 3};
common in {0, 1}; for each, a random theta (scales N(0, 1), r N(0, 0.5^2),
random.Random(20261113 + 100 structure + 10 p + common)) and a spread theta
(scales +-3 alternating, r = 1.5 alternating in sign). Tuples: order 0; all
tuples of orders 1 and 2; for orders 3 to 6 the pure tuple of every
coordinate and 6 random mixed sorted tuples (all of them when there are fewer).
"""
import math
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pdmat_scaled.csv"
PREC = 640
JAC_PREC = 2000
JAC_H_EXP = -600
KMAX = 6
H = mp.mpf(2) ** -60
H_CHECK = mp.mpf(2) ** -70
TOL = mp.mpf(10) ** -30

HEADER = "structure,p,common,theta,quantity,order,idx,i,j,hi,lo"


def n_r(st, p):
    return p * (p - 1) // 2 if st == 2 else 1


def n_scales(p, common):
    return 1 if common else p


def build_r(st, p, r):
    R = mp.matrix(p, p)
    if st == 0:
        if p == 1:
            R[0, 0] = mp.mpf(1)
            return R
        a = mp.mpf(-1) / (p - 1)
        rho = a + (1 - a) / (1 + mp.exp(-r[0]))
        for i in range(p):
            for j in range(p):
                R[i, j] = (1 - rho) * (1 if i == j else 0) + rho
        return R
    if st == 1:
        rho = mp.tanh(r[0])
        for i in range(p):
            for j in range(p):
                R[i, j] = rho ** abs(i - j)
        return R
    L = mp.matrix(p, p)
    for a in range(p):
        pr = mp.mpf(1)
        for b in range(a):
            z = mp.tanh(r[a * (a - 1) // 2 + b])
            L[a, b] = z * pr
            pr *= mp.sqrt(1 - z * z)
        L[a, a] = pr
    return L * L.T


def build(st, p, common, th):
    ns = n_scales(p, common)
    R = build_r(st, p, th[ns:])
    M = mp.matrix(p, p)
    for i in range(p):
        for j in range(p):
            si = th[0 if common else i]
            sj = th[0 if common else j]
            M[i, j] = mp.exp(si + sj) * R[i, j]
    return M


def log_jacobian(st, p, common, th):
    """log|det d vech(M) / d theta| by central differences at JAC_PREC bits."""
    with mp.workprec(JAC_PREC):
        h = mp.mpf(2) ** JAC_H_EXP
        n = len(th)
        J = mp.matrix(n, n)
        for c in range(n):
            up = [mp.mpf(x) for x in th]
            dn = [mp.mpf(x) for x in th]
            up[c] += h
            dn[c] -= h
            Mu, Md = build(st, p, common, up), build(st, p, common, dn)
            row = 0
            for i in range(p):
                for j in range(i + 1):
                    J[row, c] = (Mu[i, j] - Md[i, j]) / (2 * h)
                    row += 1
        v = mp.log(abs(mp.det(J)))
    return +v


def quantities(st, p, common, th, with_jac):
    M = build(st, p, common, th)
    Mi = mp.inverse(M)
    L = mp.cholesky(M)
    low = lambda X: {(i, j): X[i, j] for i in range(p) for j in range(i + 1)}
    out = [low(M), low(Mi), low(L), {(0, 0): mp.log(mp.det(M))}]
    if with_jac:
        out.append({(0, 0): log_jacobian(st, p, common, th)})
    return out


def derivative(st, p, common, th, tup, h, cache, with_jac):
    mult = {}
    for v in tup:
        mult[v] = mult.get(v, 0) + 1
    vars_ = sorted(mult)
    acc = None

    def rec(d, coef, shift):
        nonlocal acc
        if d == len(vars_):
            key = tuple(sorted(shift.items()))
            if key not in cache:
                x = list(th)
                for v, s in shift.items():
                    x[v] += s * h / 2
                cache[key] = quantities(st, p, common, x, with_jac)
            vals = cache[key]
            if acc is None:
                acc = [{k: mp.mpf(0) for k in q} for q in vals]
            for q in range(len(vals)):
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


def points(st, p, common):
    ns, nr = n_scales(p, common), n_r(st, p)
    rng = random.Random(20261113 + 100 * st + 10 * p + common)
    out = [[rng.gauss(0.0, 1.0) for _ in range(ns)] + [rng.gauss(0.0, 0.5) for _ in range(nr)]]
    out.append([3.0 if k % 2 == 0 else -3.0 for k in range(ns)] + [1.5 if k % 2 == 0 else -1.5 for k in range(nr)])
    return out


def tuples(n, rng):
    out = [()]
    for K in (1, 2):
        out += list(combinations_with_replacement(range(n), K))
    for K in range(3, KMAX + 1):
        out += [(v,) * K for v in range(n)]
        mixed = set()
        want = min(6, math.comb(n + K - 1, K) - n)  # every mixed tuple when there are fewer
        while len(mixed) < want:
            t = tuple(sorted(rng.randrange(n) for _ in range(K)))
            if len(set(t)) > 1:
                mixed.add(t)
        out += sorted(mixed)
    return out


def cases():
    for st in range(3):
        for p in ((2, 3) if st == 2 else (1, 3)):
            for common in (0, 1):
                yield st, p, common


def main():
    mp.mp.prec = PREC
    t0 = time.time()
    rows = []
    worst_ratio, worst_where = mp.mpf(0), None
    for st, p, common in cases():
        with_jac = st == 2 and common == 0
        for pt, tf in enumerate(points(st, p, common)):
            th = [mp.mpf(x) for x in tf]
            tstr = ";".join(float.hex(x) for x in tf)
            c1, c2 = {}, {}
            rng = random.Random(7 + 1000 * st + 100 * p + 10 * common + pt)
            for tup in tuples(len(tf), rng):
                d1 = derivative(st, p, common, th, tup, H, c1, with_jac)
                d2 = derivative(st, p, common, th, tup, H_CHECK, c2, with_jac)
                for q in range(len(d1)):
                    scale = 1 + max(abs(x) for x in d1[q].values())
                    for key in sorted(d1[q]):
                        diff = abs(d1[q][key] - d2[q][key])
                        if diff > TOL * scale:
                            raise SystemExit("h-check failed: structure %d p %d common %d point %d tuple %r "
                                             "quantity %d key %r: %s"
                                             % (st, p, common, pt, tup, q, key, mp.nstr(diff / scale, 5)))
                        if diff / scale > worst_ratio:
                            worst_ratio, worst_where = diff / scale, (st, p, common, pt, tup, q, key)
                        hi, lo = hex_pair(d1[q][key])
                        rows.append("%d,%d,%d,%s,%d,%d,%s,%d,%d,%s,%s" % (
                            st, p, common, tstr, q, len(tup), ";".join(map(str, tup)) if tup else "-",
                            key[0], key[1], hi, lo))
            print("structure %d p %d common %d point %d: %d rows, %.1f s"
                  % (st, p, common, pt, len(rows), time.time() - t0))
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
