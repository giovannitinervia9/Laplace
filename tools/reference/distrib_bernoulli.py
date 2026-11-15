"""Reference values for bernoulli_distrib (mean mu in (0, 1), observation y in {0, 1}), orders 1 to 6.

Usage (from the repository root):  python -I tools/reference/distrib_bernoulli.py

Writes tests/data/reference/distrib_bernoulli.csv, one row per value:

    quantity, y, K, hi, lo, mu

quantity   logpmf   d^K l / dmu^K of l = log p(y; mu), y in {0, 1}
           info     d^K I / dmu^K of the expected information I(mu) (y is written as 0)
           logcdf   d^K log F / dmu^K at y in [0, 1), where F(y) = P(Y <= y)
           logccdf  d^K log(1 - F) / dmu^K at y in [0, 1)
y, mu      the doubles of the test, in hexadecimal
K          the order in mu, 1 to 6 (info: 1 to 4)
hi, lo     the high and the low part of the mpmath value

Method. Every value is the mpmath derivative (mpmath.diff, its own step control) of
the DEFINITION of the quantity, not of the library's formula:

    logpmf:  log p(y; mu), with p(y; mu) = mu^y (1 - mu)^(1 - y);
    info:    sum over y in {0, 1} of p(y; mu) (d l(y; mu) / dmu)^2, the inner derivative
             taken by mpmath.diff, which is the expected information E[(d l / dmu)^2];
    logcdf:  log of the sum of p(k; mu) over k in {0, 1} with k <= floor(y);
    logccdf: log of the sum of p(k; mu) over k in {0, 1} with k > floor(y).

Each value is computed at 300 and at 400 bits and kept when the two agree to 2^-80 of
the magnitude S of the largest term of its derivative (S = |value| for the single-term
quantities, S = K! (mu^-(K+1) + (1 - mu)^-(K+1)) for info). That criterion, not a
relative one, keeps the exact zeros of the odd orders of info at mu = 1/2.

The SymPy part of the script checks each closed form that the library writes
(bernoulli.hpp) against SymPy's own symbolic derivative, for every order 1 to 6, and
exits with an error when one differs. The mpmath references are independent of that
check: they come from the definitions above.

Grid: mu in MU_GRID, from 1e-15 to 1 - 1e-15, the hard points where log(1 - mu) needs
log1p and where the odd orders of info vanish (mu = 1/2). y = 0 and y = 1 for the mass,
y = 0 and y = 0.5 for the cdf family (both in [0, 1), where F = 1 - mu).
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpmath.mpf(2) ** -80

MU_GRID = [1e-15, 1e-8, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-8, 1 - 1e-15]
Y_MASS = [0.0, 1.0]
Y_CDF = [0.0, 0.5]


def sympy_check():
    """Each closed form of bernoulli.hpp against SymPy's derivative, orders 1 to 6."""
    m = sp.symbols("mu", positive=True)
    for y in (0, 1):
        ell = y * sp.log(m) + (1 - y) * sp.log(1 - m)
        for K in range(1, 7):
            closed = (
                sp.Integer(-1) ** (K - 1) * sp.factorial(K - 1) / m**K
                if y == 1
                else -sp.factorial(K - 1) / (1 - m) ** K
            )
            assert sp.simplify(sp.diff(ell, m, K) - closed) == 0, ("logpmf", y, K)
    for K in range(1, 5):
        info = sp.Integer(1) / m + sp.Integer(1) / (1 - m)
        closed = sp.Integer(-1) ** K * sp.factorial(K) / m ** (K + 1) + sp.factorial(K) / (1 - m) ** (K + 1)
        assert sp.simplify(sp.diff(info, m, K) - closed) == 0, ("info", K)
    # the expected information as E[(d l / dmu)^2] symbolically
    score = {1: 1 / m, 0: -1 / (1 - m)}
    assert sp.simplify(m * score[1] ** 2 + (1 - m) * score[0] ** 2 - (1 / m + 1 / (1 - m))) == 0
    for K in range(1, 7):
        assert sp.simplify(sp.diff(sp.log(1 - m), m, K) + sp.factorial(K - 1) / (1 - m) ** K) == 0, ("logcdf", K)
        closed = sp.Integer(-1) ** (K - 1) * sp.factorial(K - 1) / m**K
        assert sp.simplify(sp.diff(sp.log(m), m, K) - closed) == 0, ("logccdf", K)
    print("sympy: every closed form agrees with SymPy's derivative")


def p(y, mu):
    """The mass mu^y (1 - mu)^(1 - y) at y in {0, 1}, as an mpmath number."""
    return mu**y * (1 - mu) ** (1 - y)


def logpmf(y, mu):
    return mpmath.log(p(y, mu))


def info_at(mu):
    """The expected information E[(d l / dmu)^2] at mu, the inner derivative by mpmath.diff."""
    total = mpmath.mpf(0)
    for y in (0, 1):
        dl = mpmath.diff(lambda m: logpmf(y, m), mu)
        total += p(y, mu) * dl**2
    return total


def log_cdf_at(upper, y, mu):
    """log F(y) (upper = False) or log(1 - F(y)) (upper = True) for y in [0, 1)."""
    k_max = int(mpmath.floor(y))
    ks = [k for k in (0, 1) if (k > k_max if upper else k <= k_max)]
    return mpmath.log(sum(p(k, mu) for k in ks))


def value(what, y, K, mu0, prec):
    with mpmath.workprec(prec):
        mu = mpmath.mpf(mu0)
        yv = mpmath.mpf(y)
        if what == "logpmf":
            f = lambda m: logpmf(yv, m)
        elif what == "info":
            f = info_at
        elif what == "logcdf":
            f = lambda m: log_cdf_at(False, yv, m)
        else:
            f = lambda m: log_cdf_at(True, yv, m)
        return mpmath.diff(f, mu, K)


def term_scale(what, y, K, mu):
    """The magnitude of the largest term of d^K of the quantity, for the agreement test."""
    mu = mpmath.mpf(mu)
    fk = mpmath.factorial(K - 1)
    if what == "info":
        return mpmath.factorial(K) * (mu ** (-(K + 1)) + (1 - mu) ** (-(K + 1)))
    if what == "logpmf":
        return fk / (mu**K if y == 1.0 else (1 - mu) ** K)
    if what == "logcdf":
        return fk / (1 - mu) ** K
    return fk / mu**K


def row(what, y, K, mu0):
    s = term_scale(what, y, K, mu0)
    a = value(what, y, K, mu0, 300)
    c = value(what, y, K, mu0, 400)
    if not mpmath.isfinite(c) or abs(a - c) > AGREE * s:
        return None
    hi = float(c)
    lo = float(c - mpmath.mpf(hi))
    return f"{what},{float(y).hex()},{K},{hi.hex()},{lo.hex()},{float(mu0).hex()}\n"


def task(job):
    what, y, mu0, orders = job
    lines, dropped = [], 0
    for K in orders:
        line = row(what, y, K, mu0)
        if line is None:
            dropped += 1
        else:
            lines.append(line)
    return lines, dropped


def jobs():
    out = []
    for mu in MU_GRID:
        for y in Y_MASS:
            out.append(("logpmf", y, mu, range(1, 7)))
        out.append(("info", 0.0, mu, range(1, 5)))
        for y in Y_CDF:
            out.append(("logcdf", y, mu, range(1, 7)))
            out.append(("logccdf", y, mu, range(1, 7)))
    return out


def main():
    sympy_check()
    work = jobs()
    with Pool(4) as pool:
        res = pool.map(task, work)
    n = sum(len(lines) for lines, _ in res)
    dropped = sum(dr for _, dr in res)
    with open(REF / "distrib_bernoulli.csv", "w", newline="\n") as f:
        f.write("quantity,y,K,hi,lo,mu\n")
        for lines, _ in res:
            f.writelines(lines)
    print(f"distrib_bernoulli.csv: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
