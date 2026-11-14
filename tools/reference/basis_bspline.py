"""Exact reference values for the B-spline bases (include/laplace/basis/bspline.hpp).

Usage (from the repository root):  python -I tools/reference/basis_bspline.py

Everything is computed in exact rational arithmetic (fractions.Fraction) from
the definition and rounded to the nearest double at the end, so the files hold
the correctly rounded values.

Definition. For knots t_0 <= ... <= t_{K+d} and a point x of the domain
[t_d, t_K], let s be the largest index in [d, K - 1] with t_s <= x (the basis
is right-continuous; x = t_K belongs to the last interval). Then
B_{i,0}(x) = 1 for i = s, 0 otherwise, and the Cox-de Boor recursion
    B_{i,p} = (x - t_i)/(t_{i+p} - t_i) B_{i,p-1} + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) B_{i+1,p-1}
with 0/0 = 0. Derivatives:
    D B_{i,p} = p (B_{i,p-1}/(t_{i+p} - t_i) - B_{i+1,p-1}/(t_{i+p+1} - t_{i+1})).
A SymPy check (bspline_basis_set, symbolic differentiation) confirms the
values and derivatives on one configuration; it prints the largest difference.

The integrals and the Gram matrices do not use the integral formula of the
C++ code: on each knot interval [t_s, t_{s+1}] every B_j is the polynomial
sum_m B_j^(m)(t_s) (x - t_s)^m / m!, whose exact coefficients come from the
derivatives above, and these polynomials are integrated exactly.

Files (tests/data/reference/):
  basis_bspline_knots.csv   case,name,degree,dim,kind,j,knot
      kind: 0 uniform_clamped(lower, upper, dim, degree), 1 uniform_extended,
      2 the knot vector as given (the C++ test rebuilds kinds 0 and 1 with
      their constructors and checks the knots); knot written with float.hex.
  basis_bspline.csv         case,quantity,k,l,x,first,i,value
      quantity 0: row of order k at x, i = 0..d (the column first + i);
      quantity 1: integral from lower to x, window i = 0..d;
      quantity 2: total integral of column i over the domain (x, first unused);
      quantity 3: Gram entry G_kl[first][i] = int B_first^(k) B_i^(l)
      (x unused). x and value with float.hex.
"""
import math
import os
import random
import sys
from fractions import Fraction as F

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
rng = random.Random(20261114)
KMAX = 6


def interval(t, K, d, x):
    s = d
    for j in range(d, K):
        if t[j] <= x:
            s = j
    return s


def all_orders(t, K, d, x, s=None):
    """exact B^(k)_{s-d..s,d}(x) for k = 0..d; x exact (Fraction)"""
    if s is None:
        s = interval(t, K, d, x)
    T = [F(v) for v in t]
    B = {0: {s: F(1)}}
    for p in range(1, d + 1):
        cur = {}
        for i in range(s - p, s + 1):
            v = F(0)
            a = B[p - 1].get(i, F(0)); den = T[i + p] - T[i]
            if a and den:
                v += (x - T[i]) / den * a
            b = B[p - 1].get(i + 1, F(0)); den = T[i + p + 1] - T[i + 1]
            if b and den:
                v += (T[i + p + 1] - x) / den * b
            cur[i] = v
        B[p] = cur
    res = []
    for k in range(0, d + 1):
        v = dict(B[d - k])
        for p in range(d - k + 1, d + 1):
            w = {}
            for i in range(s - p, s + 1):
                acc = F(0)
                a = v.get(i, F(0)); den = T[i + p] - T[i]
                if a and den:
                    acc += p * a / den
                b = v.get(i + 1, F(0)); den = T[i + p + 1] - T[i + 1]
                if b and den:
                    acc -= p * b / den
                w[i] = acc
            v = w
        res.append([v.get(i, F(0)) for i in range(s - d, s + 1)])
    return s, res


def taylor(t, K, d, s):
    """coefficients c[q][m] of B_{s-d+q} on interval s in powers of (x - t_s)"""
    _, der = all_orders(t, K, d, F(t[s]), s)
    return [[der[m][q] / math.factorial(m) for m in range(d + 1)] for q in range(d + 1)]


def poly_int(c, y):
    """int_0^y sum c_m u^m du"""
    return sum(cm * y ** (m + 1) / (m + 1) for m, cm in enumerate(c))


def configs(d):
    K = 12
    out = []
    nint = K - d
    for lo, hi, name in ((0.0, 1.0, "uniform_clamped"), (1000.0, 1001.0, "shifted1000")):
        h = (hi - lo) / nint
        out.append((name, 0, [lo] * (d + 1) + [lo + q * h for q in range(1, nint)] + [hi] * (d + 1), K))
    h = 1.0 / nint
    t = [0.0 + (j - d) * h if j < K else 1.0 + (j - K) * h for j in range(K + d + 1)]
    t[d] = 0.0
    out.append(("uniform_extended", 1, t, K))
    inner = sorted(rng.uniform(0.0, 1.0) for _ in range(8))
    out.append(("random", 2, [0.0] * (d + 1) + inner + [1.0] * (d + 1), len(inner) + d + 1))
    base = [q / 9.0 for q in range(1, 9)]
    for mu in sorted({2, d, d + 1}):
        if mu < 2 or mu > d + 1:
            continue
        inner = sorted(base + [0.5] * mu)
        out.append((f"multiple_mu{mu}", 2, [0.0] * (d + 1) + inner + [1.0] * (d + 1), len(inner) + d + 1))
    inner = sorted(base + [0.4, 0.4 + 1e-12])
    out.append(("tiny12", 2, [0.0] * (d + 1) + inner + [1.0] * (d + 1), len(inner) + d + 1))
    return out


def points(t, K, d):
    lo, hi = t[d], t[K]
    xs = [rng.uniform(lo, hi) for _ in range(40)]
    for v in sorted(set(t)):
        if lo <= v <= hi:
            for y in (v, math.nextafter(v, -math.inf), math.nextafter(v, math.inf)):
                if lo <= y <= hi:
                    xs.append(y)
    return sorted(set(xs))


def main():
    kf = open(os.path.join(OUT, "basis_bspline_knots.csv"), "w", newline="\n")
    rf = open(os.path.join(OUT, "basis_bspline.csv"), "w", newline="\n")
    kf.write("case,name,degree,dim,kind,j,knot\n")
    rf.write("case,quantity,k,l,x,first,i,value\n")
    case = 0
    for d in (0, 1, 2, 3, 5, 10):
        for name, kind, t, K in configs(d):
            for j, v in enumerate(t):
                kf.write(f"{case},{name},{d},{K},{kind},{j},{v.hex()}\n")
            lower = F(t[d])
            tays = {s: taylor(t, K, d, s) for s in range(d, K) if t[s + 1] > t[s]}
            # integral of column j over each full interval
            full = {}
            for s, c in tays.items():
                y = F(t[s + 1]) - F(t[s])
                for q in range(d + 1):
                    full[(s - d + q, s)] = poly_int(c[q], y)
            for x in points(t, K, d):
                X = F(x)
                s, der = all_orders(t, K, d, X)
                for k in range(0, min(d, KMAX) + 1):
                    for i, v in enumerate(der[k]):
                        rf.write(f"{case},0,{k},0,{x.hex()},{s - d},{i},{float(v).hex()}\n")
                for q in range(d + 1):
                    j = s - d + q
                    acc = sum(full.get((j, ss), F(0)) for ss in range(d, s))
                    acc += poly_int(tays[s][q], X - F(t[s]))
                    rf.write(f"{case},1,0,0,{x.hex()},{s - d},{q},{float(acc).hex()}\n")
            for j in range(K):
                tot = sum(full.get((j, ss), F(0)) for ss in range(d, K))
                rf.write(f"{case},2,0,0,0x0.0p+0,0,{j},{float(tot).hex()}\n")
            for (k, l) in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (3, 3)):
                if k > d or l > d:
                    continue
                G = [[F(0)] * K for _ in range(K)]
                for s, c in tays.items():
                    y = F(t[s + 1]) - F(t[s])
                    for a in range(d + 1):
                        pa = [c[a][m + k] * math.factorial(m + k) / math.factorial(m) for m in range(d + 1 - k)]
                        for b in range(d + 1):
                            pb = [c[b][m + l] * math.factorial(m + l) / math.factorial(m) for m in range(d + 1 - l)]
                            prod = [F(0)] * (len(pa) + len(pb))
                            for m1, c1 in enumerate(pa):
                                for m2, c2 in enumerate(pb):
                                    prod[m1 + m2] += c1 * c2
                            G[s - d + a][s - d + b] += poly_int(prod, y)
                for a in range(K):
                    for b in range(K):
                        rf.write(f"{case},3,{k},{l},0x0.0p+0,{a},{b},{float(G[a][b]).hex()}\n")
            case += 1
    kf.close()
    rf.close()

    # SymPy check of the definition on one configuration
    import sympy as sp
    xs = sp.Symbol("x")
    d = 3
    name, kind, t, K = configs(d)[3]
    tr = [sp.Rational(F(v).numerator, F(v).denominator) for v in t]
    basis = sp.bspline_basis_set(d, tr, xs)
    worst = 0
    for x in points(t, K, d)[::3]:
        if x in t:  # on a knot SymPy's Piecewise takes the left interval, Laplace the right one
            continue
        X = F(x)
        s, der = all_orders(t, K, d, X)
        for k in range(d + 1):
            for q, v in enumerate(der[k]):
                e = sp.diff(basis[s - d + q], xs, k).subs(xs, sp.Rational(X.numerator, X.denominator))
                worst = max(worst, abs(sp.Rational(v.numerator, v.denominator) - e))
    print(f"{case} cases; SymPy check, largest |exact - sympy| = {worst}")


if __name__ == "__main__":
    sys.exit(main())
