"""Exact reference values for the Legendre bases (include/laplace/basis/legendre.hpp).

Usage (from the repository root):  python -I tools/reference/basis_legendre.py

Definition. On [lower, upper] of width w = upper - lower, with t = 2 (x - lower)/w - 1,
the orthonormal Legendre basis of maximum degree n has the columns
    b_m(x) = sqrt((2 m + 1)/w) P_m(t),   m = 0..n,
and  int_lower^upper b_i b_j dx = delta_ij.

The references do not use the recurrences or the closed forms of the C++ code.
P_m has the exact integer coefficients
    P_m(t) = 2^-m sum_i (-1)^i C(m, i) C(2 m - 2 i, m) t^(m - 2 i),
which are differentiated exactly (integers) and evaluated with mpmath at 400
bits at the double x of each point (t computed in mpmath from the exact x, not
rounded), then scaled and rounded to the nearest double:
    b_m^(k)(x) = sqrt((2 m + 1)/w) (2/w)^k P_m^(k)(t).
The integral from lower, int_lower^x b_m, comes from the exact antiderivative of
the coefficients (an integer polynomial over 2^m lcm(1..m + 1)), vanishing at
t = -1:  sqrt((2 m + 1)/w) (w/2) int_{-1}^t P_m.
The cross Gram matrices G_kl[i][j] = int_lower^upper b_i^(k) b_j^(l) dx are
    sqrt((2 i + 1)(2 j + 1))/w (2/w)^(k + l) (w/2) int_{-1}^1 P_i^(k) P_j^(l) dt,
the last integral computed exactly (integers) from the coefficients. The
script checks that the (0, 0) matrix is the identity to 1e-100.

Files (tests/data/reference/):
  basis_legendre_cases.csv  case,lower,upper,n   (lower, upper written with float.hex)
  basis_legendre.csv        case,quantity,k,l,x,i,j,value
      quantity 0: b_i^(k)(x), i = 0..n (j unused, 0), k = 0..min(n, 7) (order 7 only
      supplies the conditioning of order 6 in the test);
      quantity 1: int_lower^x b_i, i = 0..n (j unused);
      quantity 3: Gram entry G_kl[i][j] (x unused).
      x and value with float.hex.
"""
import math
import os
import random
import sys
from math import comb

import mpmath as mp

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "reference")
mp.mp.prec = 400
rng = random.Random(20261114)

NS = (0, 1, 2, 5, 10, 20, 50)
INTERVALS = ((0.0, 1.0), (-1.0, 1.0), (-3.0, 7.0), (1000.0, 1001.0))
GRAM_INTERVALS = ((0.0, 1.0), (-3.0, 7.0))
GRAM_PAIRS_SMALL = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (3, 3), (1, 3))
GRAM_PAIRS_BIG = ((1, 1), (3, 3), (0, 2), (6, 6))
GRAM_NMAX_ALL = 20


def legendre_int(m):
    """integer coefficients A[p], p = 0..m, with P_m = 2^-m sum A[p] t^p"""
    a = [0] * (m + 1)
    for i in range(m // 2 + 1):
        a[m - 2 * i] = (-1) ** i * comb(m, i) * comb(2 * m - 2 * i, m)
    return a


_der = {}


def derivative(m, k):
    """integer coefficients of 2^m P_m^(k)"""
    if (m, k) not in _der:
        a = legendre_int(m)
        for _ in range(k):
            a = [a[p] * p for p in range(1, len(a))] or [0]
        _der[(m, k)] = a if k <= m else [0]
    return _der[(m, k)]


def horner(c, t):
    r = mp.mpf(0)
    for v in reversed(c):
        r = r * t + v
    return r


def lcm_upto(n):
    r = 1
    for i in range(2, n + 1):
        r = r * i // math.gcd(r, i)
    return r


_anti = {}


def antiderivative(m):
    """(C, c0, den): int_{-1}^t 2^m P_m dt = (sum C[p] t^p - c0)/den, with C[p] at power p + 1"""
    if m not in _anti:
        a = legendre_int(m)
        D = lcm_upto(m + 1)
        c = [a[p] * (D // (p + 1)) for p in range(m + 1)]
        c0 = sum(c[p] * (-1) ** (p + 1) for p in range(m + 1))
        _anti[m] = (c, c0, D)
    return _anti[m]


def exact_gram_int(n, k, l):
    """int_{-1}^1 P_i^(k) P_j^(l) dt for i, j = 0..n as (numerator S, denominator) with
    value = 2 S / (2^(i + j) L), L = lcm(1..2 n + 1)"""
    L = lcm_upto(2 * n + 1)
    G = [[None] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        A = derivative(i, k)
        for j in range(n + 1):
            B = derivative(j, l)
            S = 0
            for a, ca in enumerate(A):
                if ca == 0:
                    continue
                for b, cb in enumerate(B):
                    if cb and (a + b) % 2 == 0:
                        S += ca * cb * (L // (a + b + 1))
            G[i][j] = mp.mpf(2 * S) / (mp.mpf(2) ** (i + j) * L)
    return G


def hexf(v):
    return float(v).hex()


def points(lo, hi):
    w = hi - lo
    xs = {lo, hi, 0.5 * (lo + hi)}
    for _ in range(8):
        xs.add(rng.uniform(lo, hi))
    for k in (1, 2, 3, 4, 6, 8, 10, 12):
        xs.add(lo + w * 10.0 ** -k)
        xs.add(hi - w * 10.0 ** -k)
    return sorted(v for v in xs if lo <= v <= hi)


def main():
    cf = open(os.path.join(OUT, "basis_legendre_cases.csv"), "w", newline="\n")
    rf = open(os.path.join(OUT, "basis_legendre.csv"), "w", newline="\n")
    cf.write("case,lower,upper,n\n")
    rf.write("case,quantity,k,l,x,i,j,value\n")
    gram_cache = {}
    case = 0
    worst_id = mp.mpf(0)
    for n in NS:
        for lo, hi in INTERVALS:
            cf.write(f"{case},{lo.hex()},{hi.hex()},{n}\n")
            LO, HI = mp.mpf(lo), mp.mpf(hi)
            w = HI - LO
            for x in points(lo, hi):
                t = 2 * (mp.mpf(x) - LO) / w - 1
                for k in range(0, min(n, 7) + 1):
                    for m in range(k, n + 1):
                        v = mp.sqrt((2 * m + 1) / w) * (2 / w) ** k * horner(derivative(m, k), t) / mp.mpf(2) ** m
                        rf.write(f"{case},0,{k},0,{x.hex()},{m},0,{hexf(v)}\n")
                    for m in range(0, k):
                        rf.write(f"{case},0,{k},0,{x.hex()},{m},0,{hexf(0)}\n")
                for m in range(n + 1):
                    c, c0, D = antiderivative(m)
                    q = (horner([0] + c, t) - c0) / (mp.mpf(2) ** m * D)
                    v = mp.sqrt((2 * m + 1) / w) * (w / 2) * q
                    rf.write(f"{case},1,0,0,{x.hex()},{m},0,{hexf(v)}\n")
            if (lo, hi) in GRAM_INTERVALS:
                pairs = GRAM_PAIRS_SMALL if n <= GRAM_NMAX_ALL else GRAM_PAIRS_BIG
                for (k, l) in pairs:
                    if n == 0 and (k > 0 or l > 0):
                        continue
                    if (n, k, l) not in gram_cache:
                        gram_cache[(n, k, l)] = exact_gram_int(n, k, l)
                    I = gram_cache[(n, k, l)]
                    if (k, l) == (0, 0):
                        for i in range(n + 1):
                            for j in range(n + 1):
                                idv = mp.mpf(1) if i == j else mp.mpf(0)
                                val = mp.sqrt((2 * i + 1) * (2 * j + 1)) / w * (w / 2) * I[i][j]
                                worst_id = max(worst_id, abs(val - idv))
                    for i in range(n + 1):
                        for j in range(n + 1):
                            v = mp.sqrt((2 * i + 1) * (2 * j + 1)) / w * (2 / w) ** (k + l) * (w / 2) * I[i][j]
                            rf.write(f"{case},3,{k},{l},0x0.0p+0,{i},{j},{hexf(v)}\n")
            case += 1
    cf.close()
    rf.close()
    print(f"{case} cases; largest |G_00 - I| = {mp.nstr(worst_id, 3)}")
    if worst_id > mp.mpf(10) ** -100:
        print("G_00 is not the identity")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
