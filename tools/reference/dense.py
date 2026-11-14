"""Test matrices and reference values for laplace::dense (include/laplace/dense/dense.hpp):
eigenvalues of symmetric matrices, generalized eigenvalues of symmetric pairs
(A, B) with B positive definite, singular values and numerical ranks of
rectangular matrices, computed with mpmath at 200 bits.

Usage (from the repository root):  python -I tools/reference/dense.py

File: tests/data/reference/dense.csv, one row per value, header
    case,kind,m,n,quantity,i,j,hi,lo
kind is symmetric, generalized or rectangular; m x n is the shape of A.
Quantities: A and B, the matrices themselves (every entry, i row and j column,
lo = 0); w, the eigenvalues in increasing order (symmetric and generalized,
j = 0); s, the singular values in decreasing order (rectangular, j = 0); rank,
the numerical rank expected from the rule of laplace::dense, the number of
s_i above max(m, n) eps s_1 (rectangular, one row, i = j = 0, hi the rank).
hi is the nearest double to a value and lo the nearest double to the
remainder, both written with float.hex.

Construction. Each matrix is built at 200 bits from a chosen spectrum and
random orthogonal factors (mp.qr of a matrix of normal deviates drawn with
random.Random(20261114 + case number)), then rounded to double; the rounded
matrix (symmetric by construction: its upper triangle is copied from the
lower) is the test matrix. The references are the eigenvalues or singular
values of the rounded matrix itself, from mp.eigsy and mp.svd_r at 200 bits,
not the chosen spectrum: rounding moves them by about eps times the norm.
Generalized pairs: with B = L L^T (mp.cholesky), the eigenvalues of
L^-1 A L^-T. A matrix of exact rank r (chosen singular values zero, or a row
that is the exact sum of two rows of integers) keeps singular values of order
eps s_1 after rounding, and its expected numerical rank is r; the script stops
if a computed s_i lies within a factor 10 of the threshold, where the rank
would depend on rounding.
"""
import random
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "dense.csv"
PREC = 200
EPS = 2.0 ** -52

HEADER = "case,kind,m,n,quantity,i,j,hi,lo"


def hex_pair(x):
    """hi, lo as float.hex of the double-double nearest to the mpf x."""
    hi = float(x)
    lo = float(x - mp.mpf(hi))
    return hi.hex(), lo.hex()


def orthogonal(rng, n):
    g = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            g[i, j] = mp.mpf(rng.gauss(0.0, 1.0))
    q, _ = mp.qr(g)
    return q


def rounded(a):
    """The matrix rounded to double, entries kept as exact mpf."""
    out = mp.matrix(a.rows, a.cols)
    for i in range(a.rows):
        for j in range(a.cols):
            out[i, j] = mp.mpf(float(a[i, j]))
    return out


def symmetric_from_spectrum(rng, values):
    n = len(values)
    q = orthogonal(rng, n)
    a = q * mp.diag([mp.mpf(v) for v in values]) * q.T
    a = rounded(a)
    for i in range(n):
        for j in range(i + 1, n):
            a[i, j] = a[j, i]
    return a


def rectangular_from_singular_values(rng, m, n, values):
    u = orthogonal(rng, m)
    v = orthogonal(rng, n)
    s = mp.matrix(m, n)
    for k, x in enumerate(values):
        s[k, k] = mp.mpf(x)
    return rounded(u * s * v.T)


def rows_of_matrix(case, kind, name, a, m, n):
    rows = []
    for i in range(a.rows):
        for j in range(a.cols):
            rows.append(f"{case},{kind},{m},{n},{name},{i},{j},{float(a[i, j]).hex()},{(0.0).hex()}")
    return rows


def rows_of_values(case, kind, name, values, m, n):
    rows = []
    for i, x in enumerate(values):
        hi, lo = hex_pair(x)
        rows.append(f"{case},{kind},{m},{n},{name},{i},0,{hi},{lo}")
    return rows


def symmetric_eigenvalues(a):
    w = mp.eigsy(a, eigvals_only=True)
    return sorted(w[i] for i in range(a.rows))


def main():
    mp.mp.prec = PREC
    out = [HEADER]
    case = 0

    # Symmetric: (n, spectrum)
    symmetric = [
        (5, [1, 2, 3, 4, 5]),
        (20, [-3 + 10 * k / 19 for k in range(20)]),             # indefinite
        (20, [10.0 ** (-12 * k / 19) for k in range(20)]),       # positive definite, condition 1e12
        (20, [1 + 1e-9 * k for k in range(10)] + [2 + k for k in range(10)]),  # a cluster of ten
        (12, [0.0] * 4 + [1 + k for k in range(8)]),             # semidefinite, four zeros
        (60, None),                                             # 60 values uniform on (0.1, 10)
    ]
    for n, spectrum in symmetric:
        case += 1
        rng = random.Random(20261114 + case)
        if spectrum is None:
            spectrum = [rng.uniform(0.1, 10.0) for _ in range(n)]
        a = symmetric_from_spectrum(rng, spectrum)
        out += rows_of_matrix(case, "symmetric", "A", a, n, n)
        out += rows_of_values(case, "symmetric", "w", symmetric_eigenvalues(a), n, n)

    # Generalized: (n, spectrum of A or 'difference', spectrum of B)
    generalized = [
        (5, [-2, -1, 0.5, 1, 3], [10.0 ** (-2 * k / 4) for k in range(5)]),         # cond(B) = 1e2
        (20, "difference", [10.0 ** (-6 * k / 19) for k in range(20)]),             # penalty-like A, cond(B) = 1e6
        (40, [-5 + 10 * k / 39 for k in range(40)], [10.0 ** (-3 * k / 39) for k in range(40)]),
    ]
    for n, spectrum_a, spectrum_b in generalized:
        case += 1
        rng = random.Random(20261114 + case)
        if spectrum_a == "difference":
            # D^T D for the second differences D ((n - 2) x n): integers, rank n - 2
            a = mp.matrix(n, n)
            for r in range(n - 2):
                coef = {r: 1, r + 1: -2, r + 2: 1}
                for i, ci in coef.items():
                    for j, cj in coef.items():
                        a[i, j] += ci * cj
        else:
            a = symmetric_from_spectrum(rng, spectrum_a)
        b = symmetric_from_spectrum(rng, spectrum_b)
        l = mp.cholesky(b)
        li = mp.inverse(l)
        c = li * a * li.T
        for i in range(n):
            for j in range(i + 1, n):
                c[i, j] = c[j, i] = (c[i, j] + c[j, i]) / 2
        out += rows_of_matrix(case, "generalized", "A", a, n, n)
        out += rows_of_matrix(case, "generalized", "B", b, n, n)
        out += rows_of_values(case, "generalized", "w", symmetric_eigenvalues(c), n, n)

    # Rectangular: (m, n, singular values, or 'integers' for constraint rows)
    rectangular = [
        (8, 5, [5, 4, 3, 2, 1]),
        (5, 8, [10.0 ** (-10 * k / 4) for k in range(5)]),
        (40, 25, [1 + k for k in range(18)][::-1] + [0.0] * 7),      # rank 18
        (25, 40, [10.0 ** (-k / 3) for k in range(10)] + [0.0] * 15),  # rank 10
        (30, 30, [10.0 ** (-12 * k / 29) for k in range(30)]),       # full rank, condition 1e12
        (1, 20, "ones"),                                             # sum-to-zero constraint
        (3, 20, None),                                               # three random constraints
        (3, 20, "integers"),                                         # third row = first + second, rank 2
    ]
    for m, n, values in rectangular:
        case += 1
        rng = random.Random(20261114 + case)
        if values == "ones":
            a = mp.matrix([[1] * n])
        elif values == "integers":
            r1 = [rng.randint(-5, 5) for _ in range(n)]
            r2 = [rng.randint(-5, 5) for _ in range(n)]
            a = mp.matrix([r1, r2, [x + y for x, y in zip(r1, r2)]])
        elif values is None:
            a = mp.matrix(m, n)
            for i in range(m):
                for j in range(n):
                    a[i, j] = mp.mpf(rng.gauss(0.0, 1.0))
        else:
            a = rectangular_from_singular_values(rng, m, n, values)
        s = mp.svd_r(a, compute_uv=False)
        s = sorted((s[i] for i in range(min(m, n))), reverse=True)
        threshold = max(m, n) * EPS * s[0]
        rank = sum(1 for x in s if x > threshold)
        near = [x for x in s if threshold / 10 < x < threshold * 10]
        if near:
            raise SystemExit(f"case {case}: singular value {mp.nstr(near[0], 5)} near the threshold "
                             f"{mp.nstr(threshold, 5)}")
        out += rows_of_matrix(case, "rectangular", "A", a, m, n)
        out += rows_of_values(case, "rectangular", "s", s, m, n)
        out.append(f"{case},rectangular,{m},{n},rank,0,0,{float(rank).hex()},{(0.0).hex()}")
        print(f"case {case}: {m} x {n}, rank {rank}, s_min/s_1 = {mp.nstr(s[-1] / s[0], 3)}")

    OUT.write_text("\n".join(out) + "\n")
    print(f"{OUT}: {len(out) - 1} rows, {case} cases")


if __name__ == "__main__":
    main()
