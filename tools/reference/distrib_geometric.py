"""Reference values and closed forms for geometric_distrib (mean mu, y = 0, 1, 2, ...).

Usage (from the repository root):  python -I tools/reference/distrib_geometric.py

Writes tests/data/reference/distrib_geometric.csv, one row per value:
    kind, y, K, hi, lo, mu
with kind 0 = log p(y; mu), 1 = expected information (y unused, written 0), 2 = log F(y),
3 = log(1 - F(y)); y and mu in hexadecimal (the double values the test uses); K the derivative
order in mu (0 the value itself); hi and lo the high and low parts of the mpmath value.

Method. The derivative of the DEFINITION is taken symbolically with SymPy and evaluated by mpmath at
400 bits (a row is kept when 300 and 400 bits agree to 2^-100 relative), so the route shares no
formula with the header:
    l(mu; y)  = y log mu - (y + 1) log(1 + mu)
    I(mu)     = 1 / (mu (1 + mu))                   (the variance of the score: Var(Y) / (mu (1 + mu))^2)
    log F     = log(1 - (mu / (1 + mu))^n),  n = floor(y) + 1
    log(1-F)  = n log(mu / (1 + mu))
with n kept symbolic in the derivative, so one derivative serves every y.

Closed forms used by the header, derived and checked here (the check runs on every invocation):
  1. Brackets:  B_K(mu) = 1/mu^K - 1/(1 + mu)^K = S_K(mu) / (mu (1 + mu))^K,
     S_K = sum_(j < K) C(K, j) mu^j, so
       d^K l  = (-1)^(K-1) (K-1)! [y S_K - mu^K] / (mu (1 + mu))^K,
       d^K I  = (-1)^K K! B_(K+1),
       d^K log(1 - F) = (-1)^(K-1) (K-1)! n B_K,
     and u(mu) = log(1 - F) has d^K u = that last expression.
  2. g(u) = log(-expm1(u)), q = 1 / expm1(-u) > 0:   g^(k)(u) = -R_(k-1)(q), k >= 1,
     with R_0 = q and R_(j+1) = (q + q^2) R_j'(q); every coefficient of R_j is positive (they are
     k! S(j + 1, k) with S the Stirling numbers of the second kind), so nothing cancels for u < 0.
     log F = g(u(mu)) is composed with laplace::math::chain_rule_d1..d6.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpmath.mpf(2) ** -100
KMAX = 6

mu_s, y_s, n_s, u_s, q_s = sp.symbols("mu y n u q", positive=True)

MU = [1e-6, 0.01, 0.5, 1.0, 3.5, 40.0, 1e3, 1e6]


def check_forms():
    """Check the closed forms against SymPy's own derivatives (assertions) and print R_j."""
    for K in range(1, KMAX + 2):
        S = sum(sp.binomial(K, j) * mu_s ** j for j in range(K))
        B = 1 / mu_s ** K - 1 / (1 + mu_s) ** K
        assert sp.simplify(B - S / (mu_s * (1 + mu_s)) ** K) == 0, K
    l = y_s * sp.log(mu_s) - (y_s + 1) * sp.log(1 + mu_s)
    info = 1 / (mu_s * (1 + mu_s))
    u = n_s * (sp.log(mu_s) - sp.log(1 + mu_s))
    for K in range(1, KMAX + 1):
        S = sum(sp.binomial(K, j) * mu_s ** j for j in range(K))
        form = (-1) ** (K - 1) * sp.factorial(K - 1) * (y_s * S - mu_s ** K) / (mu_s * (1 + mu_s)) ** K
        assert sp.simplify(form - sp.diff(l, mu_s, K)) == 0, ("l", K)
        Sb = sum(sp.binomial(K, j) * mu_s ** j for j in range(K))
        formu = (-1) ** (K - 1) * sp.factorial(K - 1) * n_s * Sb / (mu_s * (1 + mu_s)) ** K
        assert sp.simplify(formu - sp.diff(u, mu_s, K)) == 0, ("u", K)
    for K in range(1, 5):
        S = sum(sp.binomial(K + 1, j) * mu_s ** j for j in range(K + 1))
        form = (-1) ** K * sp.factorial(K) * S / (mu_s * (1 + mu_s)) ** (K + 1)
        assert sp.simplify(form - sp.diff(info, mu_s, K)) == 0, ("I", K)
    # g(u) = log(1 - e^u), q = 1/(e^(-u) - 1):  g^(k) = -R_(k-1)(q)
    g = sp.log(1 - sp.exp(u_s))
    qexpr = 1 / (sp.exp(-u_s) - 1)
    R = q_s
    print("R_j(q), coefficients from degree 1, g^(j+1)(u) = -R_j(q):")
    for j in range(0, KMAX):
        check = sp.simplify((-R).subs(q_s, qexpr) - sp.diff(g, u_s, j + 1))
        assert check == 0, (j, check)
        p = sp.Poly(sp.expand(R), q_s)
        coeffs = [int(c) for c in reversed(p.all_coeffs())]
        assert all(c >= 0 for c in coeffs)
        print(f"  j={j}: {coeffs[1:]}")
        R = sp.expand((q_s + q_s ** 2) * sp.diff(R, q_s))
    print("forms checked against SymPy")


def derivative_functions():
    """lambdified (mu, y or n) derivatives of the definitions, orders 0..KMAX."""
    l = y_s * sp.log(mu_s) - (y_s + 1) * sp.log(1 + mu_s)
    info = 1 / (mu_s * (1 + mu_s))
    ratio = mu_s / (1 + mu_s)
    logF = sp.log(1 - ratio ** n_s)
    logG = n_s * sp.log(ratio)
    out = {}
    for kind, expr, var, kmax in ((0, l, y_s, KMAX), (1, info, None, 4), (2, logF, n_s, KMAX), (3, logG, n_s, KMAX)):
        fs = []
        for K in range(0, kmax + 1):
            d = expr if K == 0 else sp.diff(expr, mu_s, K)
            args = (mu_s,) if var is None else (mu_s, var)
            fs.append(sp.lambdify(args, d, modules="mpmath"))
        out[kind] = fs
    return out


FUNCS = None


def value(kind, K, mu0, arg, prec):
    with mpmath.workprec(prec):
        f = FUNCS[kind][K]
        m = mpmath.mpf(mu0)
        if kind == 1:
            return f(m)
        return f(m, mpmath.mpf(arg))


def row(kind, K, mu0, y):
    arg = y if kind in (0,) else (math.floor(y) + 1 if kind in (2, 3) else 0)
    try:
        a = value(kind, K, mu0, arg, 300)
        c = value(kind, K, mu0, arg, 400)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    if c == 0 or not mpmath.isfinite(c) or abs(a - c) > AGREE * abs(c):
        return None
    hi = float(c)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    lo = float(c - mpmath.mpf(hi))
    return f"{kind},{float(0 if kind == 1 else y).hex()},{K},{hi.hex()},{lo.hex()},{float(mu0).hex()}\n"


def ys_for(mu):
    """Integer observations: the ends, the zeros of the derivatives (y near mu^K / S_K), the
    centre and both tails; one non-integer value to exercise the floor."""
    ys = {0, 1, 2, 3, 7}
    for K in range(1, KMAX + 1):
        S = sum(math.comb(K, j) * mu ** j for j in range(K))
        z = mu ** K / S
        for d in (-1, 0, 1):
            ys.add(max(0, int(round(z)) + d))
    for r in (0.25, 0.5, 1.0, 1.5, 2.0, 5.0, 20.0, 100.0):
        ys.add(int(round(r * mu)))
    ys = sorted(v for v in ys if v < 3e8)
    return [float(v) for v in ys] + [2.5]


def task(mu):
    lines, dropped = [], 0
    for y in ys_for(mu):
        for kind in (0, 2, 3):
            for K in range(0, KMAX + 1):
                r = row(kind, K, mu, y)
                if r is None:
                    dropped += 1
                else:
                    lines.append(r)
    for K in range(0, 5):
        r = row(1, K, mu, 0.0)
        if r is None:
            dropped += 1
        else:
            lines.append(r)
    return lines, dropped


def init():
    global FUNCS
    FUNCS = derivative_functions()


def main():
    check_forms()
    init()
    with Pool(4, initializer=init) as pool:
        res = pool.map(task, MU)
    n = dropped = 0
    with open(REF / "distrib_geometric.csv", "w", newline="\n") as f:
        f.write("kind,y,K,hi,lo,mu\n")
        for lines, dr in res:
            f.writelines(lines)
            n += len(lines)
            dropped += dr
    print(f"distrib_geometric.csv: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
