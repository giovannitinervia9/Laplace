"""Check of the closed forms of the Box-Cox bijector (include/laplace/bijector/detail/scalar_box_cox.hpp)
against SymPy derivatives of the defining expressions, for every (order in the argument K or m,
order in lambda n) with total order at most 6.

Usage (from the repository root): python -I tools/reference/bijector_box_cox_forms.py

For each form, the closed form is evaluated in mpmath at 60 digits at random points and compared
with the SymPy derivative of the defining expression, evaluated at the same point. The forms are
those of the header's top comment: g = L phi(lambda L) (phi = (e^y - 1)/y, SymPy-differentiated),
h = exp(S), S = u psi(lambda u) with psi = log(1 + y)/y, the Bell recursion for the lambda
derivatives of h, the Leibniz form of the P_K jets, and the two log-Jacobians. Prints the largest
relative difference per form; a value below 1e-40 means the closed form is right.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402
import sympy as sp  # noqa: E402

mp.mp.dps = 60
X, LAM = sp.symbols("x lam", real=True)  # X = v for g, u for h
Y = sp.Symbol("y", real=True)
KMAX = 6

# defining expressions and their SymPy derivatives
L = sp.log(X)
g_def = L * (sp.exp(LAM * L) - 1) / (LAM * L)     # (v^lambda - 1)/lambda with the removable point cancelled
h_def = (1 + LAM * X) ** (1 / LAM)
phi = (sp.exp(Y) - 1) / Y                          # exprel
psi = sp.log(1 + Y) / Y                            # log1prel
S_def = sp.log(1 + LAM * X) / LAM                  # log h
LJg_def = (LAM - 1) * L                            # log g'
LJh_def = (1 - LAM) * S_def                        # log h'

phi_d = [sp.lambdify(Y, sp.diff(phi, Y, k), "mpmath") for k in range(7)]
psi_d = [sp.lambdify(Y, sp.diff(psi, Y, k), "mpmath") for k in range(7)]


def rnd(rng, lo, hi):
    return mp.mpf(rng.uniform(lo, hi))


def binom(n, k):
    return mp.binomial(n, k)


def falling(a, k):
    r = mp.mpf(1)
    for i in range(k):
        r *= a - i
    return r


def rising(a, k):
    r = mp.mpf(1)
    for i in range(k):
        r *= a + i
    return r


def sgn(k):
    return -1 if k % 2 else 1


# ---------------------------------------------------------------- forward g
def G_closed(m, n, v, lam):
    Lv = mp.log(v)
    if m == 0 and n == 0:
        y = lam * Lv
        return Lv * mp.expm1(y) / y if y != 0 else Lv
    if n == 0:
        return falling(lam - 1, m - 1) * v ** (lam - m)
    if m == 0:
        return Lv ** (n + 1) * phi_d[n](lam * Lv)
    j = m - 1
    e = [mp.mpf(1)] + [mp.mpf(0)] * 6
    for r in range(j):
        c = lam - 1 - r
        for k in range(r + 1, 0, -1):
            e[k] += c * e[k - 1]
    s = mp.mpf(0)
    for p in range(0, min(j, n) + 1):
        s += e[j - p] * falling(n, p) * Lv ** (n - p)
    return v ** (lam - m) * s


def LJg_closed(m, n, v, lam):
    Lv = mp.log(v)
    if n >= 2:
        return mp.mpf(0)
    if m == 0:
        return (lam - 1) * Lv if n == 0 else Lv
    d = sgn(m - 1) * mp.factorial(m - 1) * v ** (-m)
    return (lam - 1) * d if n == 0 else d


# ---------------------------------------------------------------- inverse h
def S_jet_closed(K, n, u, lam):
    x = lam * u
    if K == 0:
        return u ** (n + 1) * psi_d[n](x)
    t = 1 / (1 + x)
    s = mp.mpf(0)
    for j in range(0, min(n, K - 1) + 1):
        q = n - j
        s += (binom(n, j) * falling(K - 1, j) * lam ** (K - 1 - j) * sgn(q) * rising(K, q) * u ** q * t ** (K + q))
    return sgn(K - 1) * mp.factorial(K - 1) * s


def PK_jets(K, lam):
    P = [mp.mpf(1)] + [mp.mpf(0)] * 6
    for k in range(1, K):
        f = 1 - k * lam
        for j in range(6, -1, -1):
            P[j] = f * P[j] - (k * j * P[j - 1] if j > 0 else 0)
    return P


def H_closed(K, n, u, lam):
    x = lam * u
    S = u * psi_d[0](x)
    A = [mp.mpf(1)] + [mp.mpf(0)] * 7
    xs = [mp.mpf(0)] * 8
    for j in range(1, n + 1):
        xs[j] = S_jet_closed(0, j, u, lam)
    for i in range(n):
        A[i + 1] = sum(binom(i, j) * xs[j + 1] * A[i - j] for j in range(0, i + 1))
    if K == 0:
        return mp.exp(S) * A[n]
    P = PK_jets(K, lam)
    t = 1 / (1 + x)
    s = mp.mpf(0)
    for j in range(0, n + 1):
        for i in range(0, n - j + 1):
            q = n - j - i
            s += binom(n, j) * binom(n - j, i) * P[j] * A[i] * sgn(q) * rising(K, q) * u ** q * t ** q
    return mp.exp((1 - K * lam) * S) * s


def LJh_closed(K, n, u, lam):
    if n == 0:
        return (1 - lam) * S_jet_closed(K, 0, u, lam)
    return (1 - lam) * S_jet_closed(K, n, u, lam) - n * S_jet_closed(K, n - 1, u, lam)


# ---------------------------------------------------------------- the checks
def sympy_partial(expr, var, n_var, n_lam):
    e = expr
    if n_var:
        e = sp.diff(e, var, n_var)
    if n_lam:
        e = sp.diff(e, LAM, n_lam)
    return sp.lambdify((X, LAM), e, "mpmath")


def check(name, defn, domain, closed, pairs):
    """domain 'v': the argument is positive, log-uniform over 2^-30..2^30; 'u': inside the image of h."""
    worst = mp.mpf(0)
    rng = random.Random(7)
    for (m, n) in pairs:
        f = sympy_partial(defn, X, m, n)
        for _ in range(12):
            if domain == "v":
                v = mp.mpf(2) ** rng.uniform(-30, 30)
                lam = rnd(rng, -3, 3)
            else:
                lam = rnd(rng, -2, 2)
                v = rnd(rng, -2, 2)
                if 1 + lam * v <= 0:
                    continue
            ref = f(v, lam)
            got = closed(m, n, v, lam)
            scale = max(abs(ref), mp.mpf(10) ** -50)
            rel = abs(got - ref) / scale
            worst = max(worst, rel)
    print(f"{name:14s} order pairs {len(pairs):2d}  largest relative difference {mp.nstr(worst, 3)}")
    return worst


def main():
    pairs = [(m, n) for m in range(0, KMAX + 1) for n in range(0, KMAX + 1 - m) if m + n >= 1]
    ok = True
    ok &= check("g", g_def, "v", G_closed, pairs) < mp.mpf("1e-40")
    ok &= check("log g'", LJg_def, "v", LJg_closed, pairs) < mp.mpf("1e-40")
    ok &= check("h", h_def, "u", H_closed, pairs) < mp.mpf("1e-40")
    ok &= check("log h'", LJh_def, "u", LJh_closed, pairs) < mp.mpf("1e-40")
    print("all closed forms agree" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
