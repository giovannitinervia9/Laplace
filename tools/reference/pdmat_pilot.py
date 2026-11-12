"""Reference values for three structured p x p matrices M(theta) and their
derivatives in theta to order 6, computed with mpmath at 640 bits.

Usage (from the repository root):  python -I tools/reference/pdmat_pilot.py

File: tests/data/reference/pdmat_pilot.csv, one row per entry, with the header
    structure,p,param,theta,quantity,order,i,j,hi,lo
Integers are decimal; theta, hi and lo are written with float.hex. hi is the
nearest double to the value and lo the nearest double to the remainder.

Structures (code: name, definition). The matrices are built from these
definitions, not simplified by hand.
    0 diagonal: p parameters theta_k = theta + 0.25 k (k = 0..p-1), M = diag(exp(2 theta_k)).
      The derivatives are taken along theta_param only, the others fixed.
    1 compound_symmetry: a = -1/(p-1), rho = a + (1 - a)/(1 + exp(-theta)),
      M = (1 - rho) I + rho ones ones^T (unit diagonal, rho off the diagonal).
      For p = 1, M = [[1]].
    2 ar1: rho = tanh(theta), M[i][j] = rho^|i-j|.

Quantities: 0 the matrix M; 1 its inverse; 2 the lower Cholesky factor L with
M = L L^T; 3 log(det M). Quantities 0 to 2 write the entries with i >= j;
quantity 3 writes one row with i = j = 0.

Method. The order-n derivative of each entry of f(theta), with f a quantity
(M, its inverse, its Cholesky factor, or the log determinant), is the central
difference
    f^(n)(theta) = sum_{k=0..n} (-1)^k C(n,k) f(theta + (n/2 - k) h) / h^n,
with h = 2^-60, so the offsets are the multiples m h/2 for m = n - 2k in
{-6..6}. Each quantity is evaluated once per offset and the values are reused
for every entry and order. The check repeats every derivative with h = 2^-70
and stops with an error if the two disagree by more than 1e-30 times
(1 + the largest absolute entry of that quantity and order at that point).
The worst disagreement is printed.

Points: ar1, p in {1, 2, 3, 6}, theta in {-4, -0.8, 0, 0.35, 1.7, 5, 10};
compound_symmetry, p in {1, 2, 3, 5, 9}, theta in {-8, -2, -0.3, 0, 0.6, 2.5, 9};
diagonal, p in {1, 3}, theta in {-2, 0.3, 3}, param in 0..p-1.
"""
import math
import sys
import time
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pdmat_pilot.csv"
PREC = 640
KMAX = 6
H = mp.mpf(2) ** -60
H_CHECK = mp.mpf(2) ** -70
TOL = mp.mpf(10) ** -30
DIAG_STEP = mp.mpf("0.25")

HEADER = "structure,p,param,theta,quantity,order,i,j,hi,lo"


def build(structure, p, theta, param, shift):
    """The p x p matrix M for the parameter theta, shifted by shift along theta
    (for the diagonal structure, shifted along theta_param only)."""
    M = mp.matrix(p, p)
    if structure == 0:
        for k in range(p):
            t = theta + DIAG_STEP * k
            if k == param:
                t = t + shift
            M[k, k] = mp.exp(2 * t)
        return M
    t = theta + shift
    if structure == 1:
        if p == 1:
            M[0, 0] = mp.mpf(1)
            return M
        a = mp.mpf(-1) / (p - 1)
        rho = a + (1 - a) / (1 + mp.exp(-t))
        for i in range(p):
            for j in range(p):
                M[i, j] = (1 - rho) * (1 if i == j else 0) + rho
        return M
    rho = mp.tanh(t)
    for i in range(p):
        for j in range(p):
            M[i, j] = rho ** abs(i - j)
    return M


def quantity_entries(quantity, M):
    """Dict (i, j) -> value of the quantity, for the entries that are written."""
    p = M.rows
    if quantity == 0:
        src = M
    elif quantity == 1:
        src = mp.inverse(M)
    elif quantity == 2:
        src = mp.cholesky(M)
    else:
        return {(0, 0): mp.log(mp.det(M))}
    return {(i, j): src[i, j] for i in range(p) for j in range(i + 1)}


def offset_cache(structure, p, theta, param, quantity, h):
    """The quantity at theta + m h/2 for m = -2 KMAX .. 2 KMAX, keyed by m."""
    cache = {}
    for m in range(-2 * KMAX, 2 * KMAX + 1):
        M = build(structure, p, theta, param, m * h / 2)
        cache[m] = quantity_entries(quantity, M)
    return cache


def derivative(cache, n, h, key):
    """The n-th derivative of the entry key by the central difference."""
    s = mp.mpf(0)
    for k in range(n + 1):
        s += (-1) ** k * math.comb(n, k) * cache[n - 2 * k][key]
    return s / h ** n


def hex_pair(v):
    """hi and lo of a value as float.hex strings; an exact zero gives 0x0.0p+0."""
    if v == 0:
        return "0x0.0p+0", "0x0.0p+0"
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return float.hex(hi), float.hex(lo)


def points():
    """(structure, p, param, theta as a double) for every point of the pilot."""
    out = []
    for p in (1, 3):
        for theta in (-2.0, 0.3, 3.0):
            for param in range(p):
                out.append((0, p, param, theta))
    for p in (1, 2, 3, 5, 9):
        for theta in (-8.0, -2.0, -0.3, 0.0, 0.6, 2.5, 9.0):
            out.append((1, p, 0, theta))
    for p in (1, 2, 3, 6):
        for theta in (-4.0, -0.8, 0.0, 0.35, 1.7, 5.0, 10.0):
            out.append((2, p, 0, theta))
    return out


def main():
    mp.mp.prec = PREC
    t0 = time.time()
    rows = []
    worst_abs = mp.mpf(0)
    worst_ratio = mp.mpf(0)
    worst_where = None
    n_checked = 0

    for structure, p, param, theta_f in points():
        theta = mp.mpf(theta_f)
        for quantity in range(4):
            c1 = offset_cache(structure, p, theta, param, quantity, H)
            c2 = offset_cache(structure, p, theta, param, quantity, H_CHECK)
            keys = sorted(c1[0].keys())
            for order in range(KMAX + 1):
                vals = {}
                for key in keys:
                    v1 = derivative(c1, order, H, key)
                    v2 = derivative(c2, order, H_CHECK, key)
                    vals[key] = (v1, v2)
                scale = mp.mpf(1) + max(abs(v1) for v1, _ in vals.values())
                for key in keys:
                    v1, v2 = vals[key]
                    diff = abs(v1 - v2)
                    ratio = diff / scale
                    n_checked += 1
                    if diff > TOL * scale:
                        raise SystemExit(
                            "h-check failed: structure %d p %d param %d theta %r "
                            "quantity %d order %d key %r: difference %s, scale %s"
                            % (structure, p, param, theta_f, quantity, order, key,
                               mp.nstr(diff, 5), mp.nstr(scale, 5)))
                    if diff > worst_abs:
                        worst_abs = diff
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_where = (structure, p, param, theta_f, quantity, order, key)
                    hi, lo = hex_pair(v1)
                    rows.append("%d,%d,%d,%s,%d,%d,%d,%d,%s,%s" % (
                        structure, p, param, float.hex(theta_f), quantity, order,
                        key[0], key[1], hi, lo))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(r + "\n")

    elapsed = time.time() - t0
    print("rows written: %d" % len(rows))
    print("h-check entries compared: %d" % n_checked)
    print("worst absolute disagreement (h = 2^-60 vs 2^-70): %s" % mp.nstr(worst_abs, 5))
    print("worst disagreement relative to (1 + max entry): %s at %s"
          % (mp.nstr(worst_ratio, 5), worst_where))
    print("run time: %.2f s" % elapsed)
    print("output: %s" % OUT)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
