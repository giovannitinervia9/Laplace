"""Reference values for the tensor product bases (include/laplace/basis/tensor.hpp).

Usage (from the repository root):  python -I tools/reference/basis_tensor.py

The product's values are computed from the definitions of the margins, never
from the product code: the B-spline margins in exact rational arithmetic
(Cox-de Boor and the Taylor polynomials of basis_bspline.py and
basis_cyclic_bspline.py, whose functions are imported), the Legendre margins
from their integer coefficients (basis_legendre.py) and the Fourier margins
with mpmath, everything multiplied at 400 bits and rounded to the nearest
double at the end.

Margins (the C++ test builds the same ones; the B-spline knots are the
doubles the C++ constructors compute, taken as exact rationals):
  bspline(lower, upper, K, d)  bspline_basis_owner::uniform_clamped
  cyclic(lower, period, K, d)  cyclic_bspline_basis_owner::uniform
  legendre(lower, upper, n)    make_legendre_basis
  fourier(lower, period, H)    make_fourier_basis (one full period)
Cases:
  0  bspline(0, 1, 7, 3) x bspline(0, 2, 5, 2)
  1  bspline(0, 1, 6, 3) x cyclic(0, 1, 5, 2)        (x_2 over several periods)
  2  legendre(0, 2, 3) x fourier(0, 1, 2)
  3  bspline(0, 1, 5, 2) x cyclic(0, 1, 6, 3) x legendre(-1, 1, 2)

File tests/data/reference/basis_tensor.csv, columns
  case,quantity,k,l,p,x,index,value
k and l are multi-indices joined by ';' (k_j = -1: the integral from lower_j),
x the point joined by ';', p its number; x and value with float.hex.
  quantity 0: row of multi-index k at point p, value of column index
              (only the nonzero columns; the others are 0);
  quantity 1: expansion sum_c beta_c d^k B_c(x) at point p (index unused);
  quantity 2: Gram entry G_kl[r][c], index = r K + c (nonzero entries);
  quantity 3: the thin plate penalty of case 0,
              G_(2,0)(2,0) + 2 G_(1,1)(1,1) + G_(0,2)(0,2), index = r K + c;
  quantity 4: beta_c for the expansions, index = c (k, l, p, x unused).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractions import Fraction as F  # noqa: E402

import mpmath as mp  # noqa: E402

import basis_bspline as bsp  # noqa: E402
import basis_cyclic_bspline as cyc  # noqa: E402
import basis_legendre as leg  # noqa: E402

mp.mp.prec = 400
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
rng = random.Random(20261114)


def mpf(v):
    if isinstance(v, F):
        return mp.mpf(v.numerator) / v.denominator
    return mp.mpf(v)


def pderiv(c, k):
    for _ in range(k):
        c = [c[m] * m for m in range(1, len(c))] or [F(0)]
    return c


def pint_product(a, b, y):
    """int_0^y a(u) b(u) du for coefficient lists in u"""
    s = F(0)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    s += ai * bj * y ** (i + j + 1) / (i + j + 1)
    return s


class BSpline:
    def __init__(self, lo, hi, K, d):
        n = K - d
        h = (hi - lo) / n  # the knots in double, as uniform_clamped builds them
        self.t = [F(lo)] * (d + 1) + [F(lo + q * h) for q in range(1, n)] + [F(hi)] * (d + 1)
        self.K, self.d = K, d
        # Taylor polynomials of the columns on each interval, in u = x - t_s
        self.pieces = []
        for s in range(d, K):
            y = self.t[s + 1] - self.t[s]
            if y > 0:
                self.pieces.append((s, y, bsp.taylor(self.t, K, d, s)))

    def knots(self):
        return sorted(set(float(v) for v in self.t))

    def domain(self):
        return float(self.t[self.d]), float(self.t[self.K])

    def vec(self, k, x):
        x = F(x)
        out = [F(0)] * self.K
        d = self.d
        if k >= 0:
            if k > d:
                return out
            s, res = bsp.all_orders(self.t, self.K, d, x)
            for q in range(d + 1):
                out[s - d + q] = res[k][q]
            return out
        s = bsp.interval(self.t, self.K, d, x)
        for (s2, y, c) in self.pieces:
            if s2 > s:
                break
            yy = y if s2 < s else x - self.t[s]
            for q in range(d + 1):
                out[s2 - d + q] += bsp.poly_int(c[q], yy)
        return out

    def gram(self, k, l):
        G = [[F(0)] * self.K for _ in range(self.K)]
        d = self.d
        for (s, y, c) in self.pieces:
            for a in range(d + 1):
                pa = pderiv(c[a], k)
                for b in range(d + 1):
                    G[s - d + a][s - d + b] += pint_product(pa, pderiv(c[b], l), y)
        return G


class Cyclic:
    def __init__(self, lo, period, K, d):
        h = ((lo + period) - lo) / K  # the knots in double, as uniform builds them
        self.T = cyc.Knots([lo + j * h for j in range(K)], period)
        self.K, self.d = K, d
        self.ints = cyc.Integrals(self.T, d)

    def knots(self):
        return [float(v) for v in self.T.t]

    def domain(self):
        return float(self.T.t[0]), float(self.T.t[0] + self.T.w)

    def vec(self, k, x):
        x = F(x)
        if k < 0:
            return self.ints.at(x)
        out = [F(0)] * self.K
        d = self.d
        if k > d:
            return out
        s = self.T.interval(x)
        res = cyc.all_orders(self.T, d, x, s)
        for q in range(d + 1):
            out[(s - d + q) % self.K] += res[k][q]
        return out

    def gram(self, k, l):
        G = [[F(0)] * self.K for _ in range(self.K)]
        d = self.d
        for s, c in self.ints.tay.items():
            y = self.T(s + 1) - self.T(s)
            for a in range(d + 1):
                pa = pderiv(c[a], k)
                for b in range(d + 1):
                    G[(s - d + a) % self.K][(s - d + b) % self.K] += pint_product(pa, pderiv(c[b], l), y)
        return G


class Legendre:
    def __init__(self, lo, hi, n):
        self.lo, self.hi, self.n, self.K = F(lo), F(hi), n, n + 1
        self.w = self.hi - self.lo

    def knots(self):
        return []

    def domain(self):
        return float(self.lo), float(self.hi)

    def vec(self, k, x):
        w = mpf(self.w)
        t = 2 * (mpf(F(x)) - mpf(self.lo)) / w - 1
        out = []
        for m in range(self.n + 1):
            c = mp.sqrt(mp.mpf(2 * m + 1) / w) / mp.mpf(2) ** m
            if k >= 0:
                out.append(c * (2 / w) ** k * leg.horner(leg.derivative(m, k), t))
            else:
                C, c0, den = leg.antiderivative(m)
                out.append(c * (w / 2) * (leg.horner([0] + C, t) - c0) / den)
        return out

    def gram(self, k, l):
        w = mpf(self.w)
        g = leg.exact_gram_int(self.n, k, l)
        return [[mp.sqrt(mp.mpf(2 * i + 1) / w) * mp.sqrt(mp.mpf(2 * j + 1) / w) * (2 / w) ** (k + l) * (w / 2) *
                 g[i][j] for j in range(self.K)] for i in range(self.K)]


class Fourier:
    def __init__(self, lo, period, H):
        self.lo, self.period, self.H, self.K = F(lo), F(period), H, 2 * H + 1

    def knots(self):
        return []

    def domain(self):
        return float(self.lo), float(self.lo + self.period)

    def vec(self, k, x):
        u = mpf(F(x) - self.lo)
        out = [u if k < 0 else (mp.mpf(1) if k == 0 else mp.mpf(0))]
        for j in range(1, self.H + 1):
            nu = 2 * mp.pi * j / mpf(self.period)
            if k < 0:
                out += [(1 - mp.cos(nu * u)) / nu, mp.sin(nu * u) / nu]
            else:
                ph = k * mp.pi / 2
                out += [nu ** k * mp.sin(nu * u + ph), nu ** k * mp.cos(nu * u + ph)]
        return out

    def gram(self, k, l):
        """closed form on one full period"""
        P = mpf(self.period)
        G = [[mp.mpf(0)] * self.K for _ in range(self.K)]
        if k == 0 and l == 0:
            G[0][0] = P
        for j in range(1, self.H + 1):
            nu = 2 * mp.pi * j / P
            f = nu ** (k + l) * P / 2
            # sin(a + k pi/2), cos(a + k pi/2) = sin(a + (k + 1) pi/2)
            ph = {(2 * j - 1): 0, (2 * j): 1}
            for r, pr in ph.items():
                for c, pc in ph.items():
                    G[r][c] = f * mp.cos((k + pr - l - pc) * mp.pi / 2)
        return G


CASES = [
    ([BSpline(0, 1, 7, 3), BSpline(0, 2, 5, 2)],
     [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 0), (-1, 0), (0, -1), (-1, -1), (-1, 2)],
     [((0, 0), (0, 0)), ((1, 1), (1, 1)), ((2, 0), (2, 0)), ((1, 0), (0, 1)), ((2, 0), (0, 2))]),
    ([BSpline(0, 1, 6, 3), Cyclic(0, 1, 5, 2)],
     [(0, 0), (1, 1), (0, 2), (-1, -1), (0, -1), (2, -1)],
     [((0, 0), (0, 0)), ((1, 1), (1, 1)), ((0, 2), (0, 2))]),
    ([Legendre(0, 2, 3), Fourier(0, 1, 2)],
     [(0, 0), (2, 1), (1, 3), (-1, 0), (-1, -1)],
     [((0, 0), (0, 0)), ((1, 1), (1, 1)), ((2, 0), (0, 2))]),
    ([BSpline(0, 1, 5, 2), Cyclic(0, 1, 6, 3), Legendre(-1, 1, 2)],
     [(0, 0, 0), (1, 0, 2), (-1, 0, 0), (0, -1, 1), (-1, -1, -1)],
     [((0, 0, 0), (0, 0, 0)), ((1, 1, 1), (1, 1, 1))]),
]


def margin_points(m, extra):
    lo, hi = m.domain()
    xs = [rng.uniform(lo, hi) for _ in range(4)]
    for v in m.knots():
        for y in (v, math.nextafter(v, -math.inf), math.nextafter(v, math.inf)):
            if lo <= y <= hi:
                xs.append(y)
    if isinstance(m, Cyclic):  # several periods away
        w = hi - lo
        xs += [rng.uniform(lo - 2 * w, lo + 3 * w) for _ in range(4)]
    return xs


def points(margins):
    """14 points: random draws from each margin's own list, which holds the knots
    and one ulp on either side"""
    lists = [margin_points(m, None) for m in margins]
    return [tuple(rng.choice(L) for L in lists) for _ in range(14)]


def kron_vec(vs):
    out = [mp.mpf(1)]
    for v in vs:
        out = [a * mpf(b) for a in out for b in v]
    return out


def kron_mat(gs):
    out = [[mp.mpf(1)]]
    for g in gs:
        n = len(g)
        out = [[out[i][j] * mpf(g[a][b]) for j in range(len(out)) for b in range(n)]
               for i in range(len(out)) for a in range(n)]
    return out


def hexf(v):
    return float(v).hex()


def main():
    f = open(os.path.join(OUT, "basis_tensor.csv"), "w", newline="\n")
    f.write("case,quantity,k,l,p,x,index,value\n")
    js = lambda t: ";".join(str(v) for v in t)  # noqa: E731
    nrows = 0
    for c, (margins, korders, gpairs) in enumerate(CASES):
        K = 1
        for m in margins:
            K *= m.K
        beta = [F(rng.randint(-9, 9), 7) for _ in range(K)]
        for i, b in enumerate(beta):
            f.write(f"{c},4,,,,,{i},{hexf(b)}\n")
        pts = points(margins)
        for k in korders:
            for p, x in enumerate(pts):
                xs = js(hexf(v) for v in x)
                row = kron_vec([m.vec(kj, xj) for m, kj, xj in zip(margins, k, x)])
                for i, v in enumerate(row):
                    if v != 0:
                        f.write(f"{c},0,{js(k)},,{p},{xs},{i},{hexf(v)}\n")
                        nrows += 1
                e = mp.fsum(mpf(b) * v for b, v in zip(beta, row))
                f.write(f"{c},1,{js(k)},,{p},{xs},,{hexf(e)}\n")
        grams = {}
        for k, l in gpairs:
            G = kron_mat([m.gram(kj, lj) for m, kj, lj in zip(margins, k, l)])
            grams[(k, l)] = G
            for r in range(K):
                for cc in range(K):
                    if G[r][cc] != 0:
                        f.write(f"{c},2,{js(k)},{js(l)},,,{r * K + cc},{hexf(G[r][cc])}\n")
                        nrows += 1
        if c == 0:
            g20 = kron_mat([m.gram(kj, kj) for m, kj in zip(margins, (2, 0))])
            g11 = kron_mat([m.gram(kj, kj) for m, kj in zip(margins, (1, 1))])
            g02 = kron_mat([m.gram(kj, kj) for m, kj in zip(margins, (0, 2))])
            for r in range(K):
                for cc in range(K):
                    v = g20[r][cc] + 2 * g11[r][cc] + g02[r][cc]
                    if v != 0:
                        f.write(f"{c},3,,,,,{r * K + cc},{hexf(v)}\n")
    f.close()
    print(f"{len(CASES)} cases, {nrows} row and Gram values")


if __name__ == "__main__":
    main()
