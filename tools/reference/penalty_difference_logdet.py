"""Reference values of log det(D D^T) for the difference matrices of
laplace::penalty::difference_map (include/laplace/penalty/linear_map_penalty.hpp).

Usage (from the repository root):  python -I tools/reference/penalty_difference_logdet.py

File: tests/data/reference/penalty_difference_logdet.csv, header k,p,hi,lo:
k the order of the differences, p the number of columns of D ((p - k) x p),
hi and lo the double-double nearest to log det(D D^T) (float.hex).

Route: from the definition, independent of the closed form used in C++.
D is built explicitly with integer entries (the k-th forward differences of
the identity), D D^T is formed in exact integer arithmetic and its
determinant computed exactly with SymPy (Bareiss elimination); the log of
that integer is taken with mpmath at 200 bits. The script also checks, as
exact integers, the closed form
    det(D_k D_k^T) = prod_{j=0}^{k-1} binom(p + j, 2j + 1) / binom(2j, j)
for every (k, p) it writes, and stops if it ever fails.
"""
from math import comb
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "penalty_difference_logdet.csv"
mp.mp.prec = 200


def difference_matrix(p, k):
    rows = []
    for i in range(p - k):
        r = [0] * p
        for j in range(k + 1):
            r[i + j] = (-1) ** (k - j) * comb(k, j)
        rows.append(r)
    return sp.Matrix(rows)


def closed_form(p, k):
    v = Fraction(1)
    for j in range(k):
        v *= Fraction(comb(p + j, 2 * j + 1), comb(2 * j, j))
    return v


def hex_pair(x):
    hi = float(x)
    lo = float(x - mp.mpf(hi))
    return hi.hex(), lo.hex()


lines = ["k,p,hi,lo"]
for k in (1, 2, 3, 4):
    for p in list(range(k + 1, 31)) + [40, 50]:
        d = difference_matrix(p, k)
        det = int((d * d.T).det(method="bareiss"))
        if Fraction(det) != closed_form(p, k):
            raise SystemExit(f"closed form fails at k = {k}, p = {p}")
        hi, lo = hex_pair(mp.log(det))
        lines.append(f"{k},{p},{hi},{lo}")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {len(lines) - 1} rows to {OUT}; closed form exact on every row")
