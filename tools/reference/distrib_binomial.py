"""Reference values for binomial_distrib (mean proportion mu, known size n): the log-mass, its
derivatives in mu to order 6, the expected information and its derivatives to order 4, and the
cdf family with its logarithms.

Usage (from the repository root):  python -I tools/reference/distrib_binomial.py

File tests/data/reference/distrib_binomial.csv, one row per value:
    what, n, mu, y, hi, lo
with n, mu and y in hexadecimal (the double values used by the tests; n and y are integers),
hi and lo the high and low parts of the mpmath value at 300 bits, and what the quantity:
    0       log p(y)
    1..6    d^K log p / dmu^K
    10      I(mu) = n / (mu (1 - mu)),   11..14  d^K I / dmu^K, K = 1..4   (y is 0 there)
    20      F(y),  21  1 - F(y),  22  log F(y),  23  log(1 - F(y))          (0 <= y < n)

Method. The log-mass comes from the definition, log(C(n, y) mu^y (1 - mu)^(n - y)), with the exact
integer binomial coefficient math.comb. The derivatives are evaluated from SymPy's own derivative
of the log-mass (sympy.diff) in the variable mu, lambdified to mpmath; the closed forms the header
transcribes,
    d^K l = (K-1)! [ (-1)^(K-1) y / mu^K - (n - y) / (1 - mu)^K ],
    d^K I = n [ (-1)^K K! / mu^(K+1) + K! / (1 - mu)^(K+1) ],
are checked against SymPy's derivative of the expressions as symbolic identities before any value
is written. I = n / (mu (1 - mu)) is the Fisher information of the binomial; for n <= 100 it is
also checked here as -E[d^2 l / dmu^2] by summing the mass. The cdf is the sum of the mass over the
smaller tail, from the tail's largest term outward until a term falls below 2^-290 of the sum,
and the other side is one minus it (the tail summed is always the one of probability below
about one half, so the subtraction loses nothing); for n <= 100 the sums are also checked against
the regularized incomplete beta function of mpmath at a few points. The route shares nothing with
the library's (no incomplete beta, no Stirling error, no saddle point).

The grid: n in 1, 2, 10, 100, 1e4, 1e6; mu from 1e-12 to 1 - 1e-12; y at both ends of the
support, around n mu (the mode, one and three standard deviations either side) and in both tails.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
PREC = 300
N_LIST = [1, 2, 10, 100, 10000, 1000000]
MU_LIST = [1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4, 1 - 1e-8, 1 - 1e-12]

# ---- the closed forms against SymPy ----
S_MU, S_N, S_Y = sp.symbols("mu n y", positive=True)
S_L = (S_Y * sp.log(S_MU) + (S_N - S_Y) * sp.log(1 - S_MU))  # the part of l that depends on mu
S_I = S_N / (S_MU * (1 - S_MU))


def closed_l(K):
    return sp.factorial(K - 1) * ((-1) ** (K - 1) * S_Y / S_MU**K - (S_N - S_Y) / (1 - S_MU) ** K)


def closed_i(K):
    return S_N * ((-1) ** K * sp.factorial(K) / S_MU ** (K + 1) + sp.factorial(K) / (1 - S_MU) ** (K + 1))


def check_forms():
    for K in range(1, 7):
        assert sp.simplify(sp.diff(S_L, S_MU, K) - closed_l(K)) == 0, ("l", K)
    # I from the definition -E[d^2 l]: E[y] = n mu, and d^2 l is linear in y
    d2 = sp.diff(S_L, S_MU, 2)
    assert sp.simplify(-d2.subs(S_Y, S_N * S_MU) - S_I) == 0
    for K in range(1, 5):
        assert sp.simplify(sp.diff(S_I, S_MU, K) - closed_i(K)) == 0, ("I", K)


# lambdified SymPy derivatives (the values are evaluated from SymPy's own derivative)
L_D = {K: sp.lambdify((S_MU, S_N, S_Y), sp.diff(S_L, S_MU, K), "mpmath") for K in range(1, 7)}
I_D = {K: sp.lambdify((S_MU, S_N), sp.diff(S_I, S_MU, K), "mpmath") for K in range(0, 5)}


# ---- the mass and the cdf by summing ----
def log_pmf(n, y, mu):
    return mpmath.log(mpmath.mpf(math.comb(n, y)) * mu**y * (1 - mu) ** (n - y))


def pmf(n, y, mu):
    return mpmath.mpf(math.comb(n, y)) * mu**y * (1 - mu) ** (n - y)


def tail_sums(n, k, mu):
    """(F(k), 1 - F(k)) for 0 <= k < n, summing the tail of probability below about one half."""
    q = 1 - mu
    eps = mpmath.mpf(2) ** -290
    if k < n * mu:  # lower tail, downwards from k
        t = pmf(n, k, mu)
        s = t
        i = k
        while i > 0:
            t = t * i / (n - i + 1) * q / mu
            s += t
            i -= 1
            if t <= eps * s:
                break
        return s, 1 - s
    t = pmf(n, k + 1, mu)  # upper tail, upwards from k + 1
    s = t
    i = k + 1
    while i < n:
        t = t * (n - i) / (i + 1) * mu / q
        s += t
        i += 1
        if t <= eps * s:
            break
    return 1 - s, s


def check_beta(n, k, mu, F, G):
    """The sums against mpmath's regularized incomplete beta, F(k) = I_(1-mu)(n - k, k + 1)."""
    f = mpmath.betainc(n - k, k + 1, 0, 1 - mu, regularized=True)
    g = mpmath.betainc(k + 1, n - k, 0, mu, regularized=True)
    assert abs(f - F) <= mpmath.mpf(2) ** -150 * F
    assert abs(g - G) <= mpmath.mpf(2) ** -150 * G


def y_grid(n, mu):
    m = n * mu
    sd = math.sqrt(n * mu * (1 - mu))
    cand = {0, 1, n - 1, n, round(m), round(m) - 1, round(m) + 1}
    for z in (3.0, 1.0, -1.0, -3.0):
        cand.add(round(m + z * sd))
    for fr in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        cand.add(round(n * fr))
    cand.add(round(m * 0.1))
    cand.add(round(m * 10))
    cand.add(round(m * 2))
    return sorted(c for c in cand if 0 <= c <= n)


def split(x):
    hi = float(x)
    return hi, float(x - mpmath.mpf(hi))


def row(what, n, mu, y, x):
    if x == 0 or not mpmath.isfinite(x):
        return None
    hi, lo = split(x)
    if not math.isfinite(hi) or abs(hi) < 1e-300:
        return None
    return f"{what},{float(n).hex()},{float(mu).hex()},{float(y).hex()},{hi.hex()},{lo.hex()}\n"


def task(job):
    n, mu0 = job
    mpmath.mp.prec = PREC
    mu = mpmath.mpf(mu0)
    out = []

    def add(what, y, x):
        r = row(what, n, mu0, y, x)
        if r is not None:
            out.append(r)

    # expected information, order 0..4
    for K in range(0, 5):
        add(10 + K, 0, I_D[K](mu, mpmath.mpf(n)))
    if n <= 100:  # I = -E[d^2 l] by summing the mass
        e = sum(pmf(n, y, mu) * L_D[2](mu, mpmath.mpf(n), mpmath.mpf(y)) for y in range(n + 1))
        assert abs(-e - I_D[0](mu, mpmath.mpf(n))) <= mpmath.mpf(2) ** -200 * abs(e) * max(1, n)
    for y in y_grid(n, mu0):
        ym = mpmath.mpf(y)
        add(0, y, log_pmf(n, y, mu))
        for K in range(1, 7):
            add(K, y, L_D[K](mu, mpmath.mpf(n), ym))
        if y < n:
            F, G = tail_sums(n, y, mu)
            if n <= 100:
                check_beta(n, y, mu, F, G)
            add(20, y, F)
            add(21, y, G)
            add(22, y, mpmath.log(F))
            add(23, y, mpmath.log(G))
    return out


def main():
    mpmath.mp.prec = PREC
    check_forms()
    jobs = [(n, mu) for n in N_LIST for mu in MU_LIST]
    with Pool(4) as pool:
        res = pool.map(task, jobs, chunksize=1)
    n_rows = 0
    with open(REF / "distrib_binomial.csv", "w", newline="\n") as f:
        f.write("what,n,mu,y,hi,lo\n")
        for lines in res:
            f.writelines(lines)
            n_rows += len(lines)
    print(f"distrib_binomial.csv: {n_rows} rows")


if __name__ == "__main__":
    sys.exit(main())
