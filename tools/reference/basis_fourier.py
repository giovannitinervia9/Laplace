"""Reference values for the Fourier bases (include/laplace/basis/fourier.hpp).

Usage (from the repository root):  python -I tools/reference/basis_fourier.py

Everything is computed with mpmath at 300 bits from the definition and rounded
to the nearest double at the end. The basis on [lower, upper] with period
omega and H harmonics has the columns
    b_0 = 1,  b_{2j-1}(x) = sin(j nu x'),  b_{2j}(x) = cos(j nu x'),
with x' = x - lower, nu = 2 pi / omega; lower, upper, omega and x are the
doubles of the files, taken exactly.

Quantities (tests/data/reference/basis_fourier.csv, columns
case,quantity,k,l,p,i,j,value):
  0  b_i^(k)(x_p): the derivative of order k, written by differentiating the
     definition term by term (the cycle sin -> cos -> -sin -> -cos, times (j nu)^k;
     a SymPy check at the end compares it with sympy.diff);
  1  int_lower^{x_p} b_i, from the antiderivatives (x' for the constant,
     (1 - cos(j nu x'))/(j nu) and sin(j nu x')/(j nu));
  3  the Gram entry G_kl[i][j] = int_lower^upper b_i^(k) b_j^(l) dx. It does not
     use the product-to-sum formulas of the C++ code: each derivative is written
     as a sum of two complex exponentials (sin and cos through Im and Re of
     (i a)^k e^{i a t}), the four products are integrated exactly
     (e^{i n nu L} - 1)/(i n nu), and the real part is kept.
For H = 100 (three domains only) the rows and integrals are given for a subset
of the columns, and the Gram entries for a subset of the rows i (see keep_column).
The case number counts only the cases written.

Files: basis_fourier_cases.csv  case,lower,period,upper,H   (doubles as float.hex)
       basis_fourier_points.csv case,p,x                    (float.hex)
       basis_fourier.csv        as above                    (value as float.hex)
"""
import math
import os
import random
import sys

import mpmath as mp

mp.mp.prec = 300

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
rng = random.Random(20261114)
I = mp.mpc(0, 1)

DOMAINS = [
    ("full_0_1", 0.0, 1.0, 1.0),
    ("full_m3_4.5", -3.0, 7.5, 4.5),
    ("short_period", 0.0, 7.0, 10.0),
    ("long_period", 0.0, 12.0, 9.0),
    ("shifted_1000", 1000.0, 1.0, 1001.0),
    ("shifted_1000_two", 1000.0, 0.7, 1003.0),
]
HS = (0, 1, 4, 20, 100)
GRAM_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (3, 3), (6, 6), (0, 6), (2, 5))
GRAM_PAIRS_20 = ((0, 0), (1, 1), (2, 3), (6, 6), (0, 2))
BIG_DOMAINS = (0, 2, 5)  # the domains that carry H = 100


def keep_column(i, H):
    """for H = 100 only a subset of the columns is written (file size): the
    first harmonics, the ones after each re-anchoring of the rotation
    (j = 1 mod 8 and its neighbours), and the last ones"""
    if H < 100 or i == 0:
        return True
    j = (i + 1) // 2
    return j <= 10 or j % 4 == 1 or j % 8 == 0 or j >= 98
GRAM_PAIRS_BIG = ((0, 0), (2, 2), (1, 3), (6, 6))


def points(lo, om, hi, H):
    xs = [rng.uniform(lo, hi) for _ in range(14 if H >= 20 else 28)]
    xs += [lo, hi, lo - 1.3 * om, hi + 2.1 * om, lo - 0.01, hi + 0.01, lo + om, lo + 0.5 * om]
    for j in (1, 2, 7, 8, 9, 17, 33, 50, 100):
        if j > H:
            continue
        for _ in range(2):
            n = rng.randrange(0, j)
            d = rng.choice((0.0, 1e-9, -1e-9, 0.5, 0.5 + 1e-9, 0.5 - 1e-9))
            xs.append(lo + (n + d) / j * om)
    return xs


def columns(H):
    """(j, kind) for each column: kind 0 constant, 1 sin, 2 cos"""
    c = [(0, 0)]
    for j in range(1, H + 1):
        c += [(j, 1), (j, 2)]
    return c


def deriv_value(j, kind, k, nu, xp):
    """k-th derivative of the column at x' (mp), by the cycle of derivatives"""
    if kind == 0:
        return mp.mpf(1) if k == 0 else mp.mpf(0)
    a = j * nu
    t = a * xp
    s, c = mp.sin(t), mp.cos(t)
    # d/dx: sin -> a cos, cos -> -a sin
    cycle_sin = [s, c, -s, -c]
    cycle_cos = [c, -s, -c, s]
    v = (cycle_sin if kind == 1 else cycle_cos)[k % 4]
    return v * a ** k


def integral_value(j, kind, nu, xp):
    if kind == 0:
        return xp
    a = j * nu
    t = a * xp
    return (1 - mp.cos(t)) / a if kind == 1 else mp.sin(t) / a


def exp_terms(j, kind, k, nu):
    """the derivative of order k of the column as [(coefficient, n)]: sum c e^{i n nu t}"""
    if kind == 0:
        return [(mp.mpc(1), 0)] if k == 0 else []
    z0 = (I * j * nu) ** k
    z0 = mp.mpc(z0)
    if kind == 2:  # Re Z = (Z + conj Z)/2
        return [(z0 / 2, j), (mp.conj(z0) / 2, -j)]
    return [(z0 / (2 * I), j), (-mp.conj(z0) / (2 * I), -j)]  # Im Z = (Z - conj Z)/(2i)


def main():
    cf = open(os.path.join(OUT, "basis_fourier_cases.csv"), "w", newline="\n")
    pf = open(os.path.join(OUT, "basis_fourier_points.csv"), "w", newline="\n")
    rf = open(os.path.join(OUT, "basis_fourier.csv"), "w", newline="\n")
    cf.write("case,name,lower,period,upper,H\n")
    pf.write("case,p,x\n")
    rf.write("case,quantity,k,l,p,i,j,value\n")
    case = 0
    for di, (name, lo, om, hi) in enumerate(DOMAINS):
        for H in HS:
            if H == 100 and di not in BIG_DOMAINS:
                continue
            cf.write(f"{case},{name},{lo.hex()},{om.hex()},{hi.hex()},{H}\n")
            LO, OM, HI = mp.mpf(lo), mp.mpf(om), mp.mpf(hi)
            nu = 2 * mp.pi / OM
            L = HI - LO
            cols = columns(H)
            K = len(cols)
            for p, x in enumerate(points(lo, om, hi, H)):
                pf.write(f"{case},{p},{x.hex()}\n")
                xp = mp.mpf(x) - LO
                for k in range(7):
                    for i, (j, kind) in enumerate(cols):
                        if not keep_column(i, H):
                            continue
                        v = deriv_value(j, kind, k, nu, xp)
                        rf.write(f"{case},0,{k},0,{p},{i},0,{float(v).hex()}\n")
                for i, (j, kind) in enumerate(cols):
                    if not keep_column(i, H):
                        continue
                    v = integral_value(j, kind, nu, xp)
                    rf.write(f"{case},1,0,0,{p},{i},0,{float(v).hex()}\n")
            # Gram matrices: exact integrals of the exponentials, n = -2H..2H
            E = {}
            for n in range(-2 * H, 2 * H + 1):
                E[n] = L if n == 0 else (mp.exp(I * n * nu * L) - 1) / (I * n * nu)
            rows = list(range(K)) if H < 100 else sorted(set(list(range(8)) + [50, 51, 100, 101, 150, 151, K - 2, K - 1]))
            for (k, l) in (GRAM_PAIRS if H < 20 else GRAM_PAIRS_20 if H < 100 else GRAM_PAIRS_BIG):
                tb = [exp_terms(j, kind, l, nu) for (j, kind) in cols]
                for i in rows:
                    ta = exp_terms(*cols[i][:1], cols[i][1], k, nu)
                    for jj in range(K):
                        acc = mp.mpc(0)
                        for c1, n1 in ta:
                            for c2, n2 in tb[jj]:
                                acc += c1 * c2 * E[n1 + n2]
                        rf.write(f"{case},3,{k},{l},0,{i},{jj},{float(mp.re(acc)).hex()}\n")
            case += 1
    cf.close()
    pf.close()
    rf.close()

    # SymPy check of the derivative cycle
    import sympy as sp
    xs = sp.Symbol("x")
    worst = 0
    nu0 = 2 * mp.pi / 7
    for j in (1, 3):
        for kind, f in ((1, sp.sin), (2, sp.cos)):
            expr = f(j * sp.Float(str(nu0), 60) * xs)
            for k in range(7):
                d = sp.diff(expr, xs, k).subs(xs, sp.Float("0.8123456789", 60))
                v = deriv_value(j, kind, k, nu0, mp.mpf("0.8123456789"))
                worst = max(worst, abs(mp.mpf(str(d)) - v))
    print(f"{case} cases; SymPy check of the derivatives, largest difference = {mp.nstr(worst, 3)}")


if __name__ == "__main__":
    sys.exit(main())
