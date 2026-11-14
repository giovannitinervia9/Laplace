"""Exact reference values for the cyclic B-spline bases (include/laplace/basis/cyclic_bspline.hpp).

Usage (from the repository root):  python -I tools/reference/basis_cyclic_bspline.py

Everything is computed in exact rational arithmetic (fractions.Fraction) from
the definition and rounded to the nearest double at the end.

Definition. The K knots t_0 <= ... <= t_{K-1} of one period (doubles) and the
period omega (a double) define the bi-infinite sequence
T_e = t_(e mod K) + floor(e/K) omega, taken exactly. Column c of the cyclic
basis is b_c(x) = sum_m B_{c+mK}(x), B_e the B-spline of degree d on
T_e..T_{e+d+1}. At a point x (any double, not reduced to one period) let s be
the index with T_s <= x < T_{s+1}; the nonzero B-splines are B_{s-d..s},
computed by the Cox-de Boor recursion on the exact knots, and B_e belongs to
column e mod K. Derivatives by the derivative recursion (as in
basis_bspline.py). This route never reduces x modulo the period and never
folds an extended basis, which is what the C++ code does.

Integrals from lower = t_0 to x: on each interval [T_s, T_{s+1}] every b_c is
a polynomial in (x - T_s), whose exact coefficients come from the derivatives
at T_s; the integral is the sum of the exact integrals over the intervals
between T_0 and x (with a minus sign for x < lower). The Gram matrices
integrate the products over the K intervals of one period.

Files (tests/data/reference/):
  basis_cyclic_bspline_cases.csv  case,name,degree,dim,kind,lower,period,j,knot
      kind 0: uniform(lower, period, dim, degree) (the C++ test rebuilds it and
      checks the knots), kind 1: from_knots(knots, period, degree).
  basis_cyclic_bspline.csv  case,quantity,k,l,x,first,i,value
      quantity 0: row of order k at x, value i in column (first + i) mod K;
      quantity 1: int_lower^x b_i (all K columns; first unused);
      quantity 2: integral of column i over one period (x, first unused);
      quantity 3: Gram entry G_kl[first][i] = int over one period of
      b_first^(k) b_i^(l) (x unused).
"""
import math
import os
import random
import sys
from fractions import Fraction as F

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
rng = random.Random(20261114)
KMAX = 6


class Knots:
    def __init__(self, t, period):
        self.t = [F(v) for v in t]
        self.K = len(t)
        self.w = F(period)

    def __call__(self, e):
        m, j = divmod(e, self.K)
        return self.t[j] + m * self.w

    def interval(self, x):
        # largest s with T_s <= x (then T_{s+1} > x)
        m = math.floor((x - self.t[0]) / self.w)
        s = m * self.K - 1
        while self(s) > x:
            s -= 1
        while self(s + 1) <= x:
            s += 1
        return s


def all_orders(T, d, x, s):
    """exact B^(k)_{s-d..s}(x) for k = 0..d on interval s"""
    B = {0: {s: F(1)}}
    for p in range(1, d + 1):
        cur = {}
        for i in range(s - p, s + 1):
            v = F(0)
            a = B[p - 1].get(i, F(0)); den = T(i + p) - T(i)
            if a and den:
                v += (x - T(i)) / den * a
            b = B[p - 1].get(i + 1, F(0)); den = T(i + p + 1) - T(i + 1)
            if b and den:
                v += (T(i + p + 1) - x) / den * b
            cur[i] = v
        B[p] = cur
    res = []
    for k in range(0, d + 1):
        v = dict(B[d - k])
        for p in range(d - k + 1, d + 1):
            w = {}
            for i in range(s - p, s + 1):
                acc = F(0)
                a = v.get(i, F(0)); den = T(i + p) - T(i)
                if a and den:
                    acc += p * a / den
                b = v.get(i + 1, F(0)); den = T(i + p + 1) - T(i + 1)
                if b and den:
                    acc -= p * b / den
                w[i] = acc
            v = w
        res.append([v.get(i, F(0)) for i in range(s - d, s + 1)])
    return res


def poly_int(c, y):
    return sum(cm * y ** (m + 1) / (m + 1) for m, cm in enumerate(c))


class Integrals:
    """exact int_lower^x b_c; intervals s and s + K carry the same polynomials"""

    def __init__(self, T, d):
        self.T, self.d, self.K = T, d, T.K
        self.tay = {}
        self.full = {}
        for s in range(self.K):
            y = T(s + 1) - T(s)
            if y == 0:
                continue
            der = all_orders(T, d, T(s), s)
            c = [[der[m][q] / math.factorial(m) for m in range(d + 1)] for q in range(d + 1)]
            self.tay[s] = c
            self.full[s] = [poly_int(c[q], y) for q in range(d + 1)]

    def column_sums(self, lo, hi):
        """sum over intervals lo..hi - 1 of their full integrals, per column"""
        acc = [F(0)] * self.K
        for i in range(lo, hi):
            j = i % self.K
            if j in self.full:
                for q in range(self.d + 1):
                    acc[(i - self.d + q) % self.K] += self.full[j][q]
        return acc

    def at(self, x):
        s = self.T.interval(x)
        res = self.column_sums(0, s) if s >= 0 else [-v for v in self.column_sums(s, 0)]
        c = self.tay[s % self.K]
        y = x - self.T(s)
        for q in range(self.d + 1):
            res[(s - self.d + q) % self.K] += poly_int(c[q], y)
        return res


def configs(d):
    K = 12
    out = []
    for lo, w, name in ((0.0, 1.0, "uniform_unit"), (1000.0, 1.0, "uniform_shift1000"), (0.0, 365.25, "uniform_doy")):
        h = ((lo + w) - lo) / K
        out.append((name, 0, [lo + j * h for j in range(K)], w))
    h = 1.0 / (d + 1)
    out.append(("uniform_min", 0, [0.0 + j * h for j in range(d + 1)], 1.0))
    knots = [0.0] + sorted(rng.uniform(0.0, 365.0) for _ in range(K - 1))
    out.append(("random_doy", 1, knots, 365.0))
    base = [q / 9.0 for q in range(1, 9)]
    for mu in sorted({2, d + 1}):
        if mu < 2 or mu > d + 1:
            continue
        out.append((f"interior_mu{mu}", 1, sorted([0.0] + base + [0.5] * mu), 1.0))
        out.append((f"lower_mu{mu}", 1, [0.0] * mu + base, 1.0))
    return out


def points(t, w):
    lo, hi = t[0], t[0] + w
    xs = [rng.uniform(lo - 2 * w, lo + 3 * w) for _ in range(20)]
    for v in sorted(set(t)) + [hi]:
        for y in (v, math.nextafter(v, -math.inf), math.nextafter(v, math.inf)):
            if lo <= y <= hi:
                xs.append(y)
    return sorted(set(xs))


def main():
    cf = open(os.path.join(OUT, "basis_cyclic_bspline_cases.csv"), "w", newline="\n")
    rf = open(os.path.join(OUT, "basis_cyclic_bspline.csv"), "w", newline="\n")
    cf.write("case,name,degree,dim,kind,lower,period,j,knot\n")
    rf.write("case,quantity,k,l,x,first,i,value\n")
    case = 0
    for d in (0, 1, 2, 3, 5, 10):
        for name, kind, t, w in configs(d):
            K = len(t)
            if K < d + 1:
                continue
            for j, v in enumerate(t):
                cf.write(f"{case},{name},{d},{K},{kind},{t[0].hex()},{w.hex()},{j},{v.hex()}\n")
            T = Knots(t, w)
            I = Integrals(T, d)
            for x in points(t, w):
                X = F(x)
                s = T.interval(X)
                der = all_orders(T, d, X, s)
                first = (s - d) % K
                for k in range(0, min(d, KMAX) + 1):
                    for i, v in enumerate(der[k]):
                        rf.write(f"{case},0,{k},0,{x.hex()},{first},{i},{float(v).hex()}\n")
                for c, v in enumerate(I.at(X)):
                    rf.write(f"{case},1,0,0,{x.hex()},0,{c},{float(v).hex()}\n")
            tot = I.column_sums(0, K)
            for c in range(K):
                rf.write(f"{case},2,0,0,0x0.0p+0,0,{c},{float(tot[c]).hex()}\n")
            for (k, l) in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (3, 3)):
                if k > d or l > d:
                    continue
                G = [[F(0)] * K for _ in range(K)]
                for s, c in I.tay.items():
                    y = T(s + 1) - T(s)
                    for a in range(d + 1):
                        pa = [c[a][m + k] * math.factorial(m + k) / math.factorial(m) for m in range(d + 1 - k)]
                        for b in range(d + 1):
                            pb = [c[b][m + l] * math.factorial(m + l) / math.factorial(m) for m in range(d + 1 - l)]
                            prod = [F(0)] * (len(pa) + len(pb))
                            for m1, c1 in enumerate(pa):
                                for m2, c2 in enumerate(pb):
                                    prod[m1 + m2] += c1 * c2
                            G[(s - d + a) % K][(s - d + b) % K] += poly_int(prod, y)
                for a in range(K):
                    for b in range(K):
                        rf.write(f"{case},3,{k},{l},0x0.0p+0,{a},{b},{float(G[a][b]).hex()}\n")
            case += 1
    cf.close()
    rf.close()
    print(f"{case} cases")


if __name__ == "__main__":
    sys.exit(main())
