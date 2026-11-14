"""SymPy check of the closed forms of gamma2 (mean mu, variance sigma2) used in
include/laplace/distrib/gamma2.hpp, and mpmath references of its log-density, log F and
log(1 - F) derivatives in (mu, sigma2) to total order 6.

Usage (from the repository root):  python -I tools/reference/distrib_gamma2.py

Part 1, SymPy (printed; the script exits with status 1 on any disagreement):
  - the outer partials of G(s, c) = -s D(c) + (1/2) log s - delta(s), D(c) = c - 1 - log c,
    in (s, c) to total order 6, against SymPy's own derivatives (delta is a chain of
    symbols D_k = delta^(k)(s); the total derivative in s is the chain rule on the D_k);
  - the expected information: I_new = J' I_old J with the Jacobian J of (mu, phi) in
    (mu, sigma2), phi = sigma2 / mu^2 = 1/s, and I_old of gamma1; its entries against the
    closed forms H_mm, H_ms, H_ss of gamma2.hpp; and the s-derivatives of H against the
    Leibniz formula of stirling_power_term;
  - the direct closed forms of the log-density in (mu, sigma2) at orders 1 and 2 (kind 2
    in the CSV: they are evaluated at 250 bits and used by the C++ test as a second route).

Part 2, mpmath (tests/data/reference/distrib_gamma2.csv), one row per value:
    kind (0 log-density, 1 log F or log(1 - F), 2 the direct SymPy form of orders 1 and 2),
    upper (0 log F or density, 1 log(1 - F)), y, mu, sigma2, K, b, hi, lo
with b the number of derivatives in sigma2 (K - b in mu), and y, mu, sigma2, hi, lo as
hexadecimal doubles. Values are mpmath.diff of the DEFINITIONS
    log f = (s - 1) log y - s y / mu - s log(mu/s) - loggamma(s),   s = mu^2/sigma2,
    log F = log P(s, mu y / sigma2),  log(1 - F) = log Q(s, mu y / sigma2)
(mpmath.gammainc, regularized), at 200 and 260 bits; a row is kept if the two agree to
2^-80 relative. No Stirling series and no closed form of this file's C++ is used.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_gamma2.csv"
MAXK = 6
AGREE = mpmath.mpf(2) ** -80
# (mu, phi = sigma2 / mu^2) pairs: shapes 1/phi from 1e-4 to 1e8
MUS = [0.4, 2.5]
# Shapes 1/phi from 1e-4 to 1e4. The shape 1e8 (phi = 1e-8) is not in the mpmath table:
# mpmath.gammainc raises NoConvergence there for every c >= 1 (measured 2026-11-14 at
# c = 1, 1.05 and 1.5, both values of mu), it takes no maxterms argument, and the
# quadrature that replaces it can only carry whichever of P and Q is the small one, so
# the other side would have to come from log1p of it. The same range as the gamma1 table.
# The shape 1e8 is covered instead by the SymPy checks of part 1 and, for the expected
# information, by the comparison with the exact transform at 300 bits recorded in
# docs/numerics.md.
PHIS = [1e-4, 0.02, 1.0, 1e4]
CS = [0.01, 0.7, 0.95, 1.0, 1.05, 1.5, 10.0]  # y = mu c


# ---------------------------------------------------------------- SymPy part

def total_d(expr, var, s_expr, dvar_s):
    """Total derivative in var of an expression in var and the symbols D_k = delta^(k)(s), where
    s = s_expr(var) and d s / d var = dvar_s (chain rule on the D_k)."""
    out = sp.diff(expr, var)
    for k in range(0, MAXK):
        out += sp.diff(expr, D[k]) * D[k + 1] * dvar_s
    return sp.expand(out)


D = [sp.Symbol(f"D{k}") for k in range(0, MAXK + 2)]  # D[k] = delta^(k)(s)
s, c, mu, sig2, y, m = sp.symbols("s c mu sig2 y m", positive=True)


def check_outer():
    """G(s, c): closed forms of the (i, j) partials, i + j <= 6, against SymPy's."""
    G = -s * (c - 1 - sp.log(c)) + sp.log(s) / 2 - D[0]
    ok = True
    # closed forms as in gamma2.hpp: j = 0: i = 0 (value), i = 1: -D(c) + h', i >= 2: h^(i);
    # j >= 1: i = 0: -s D^(j)(c), i = 1: -D^(j)(c); i >= 2: 0
    Dc = lambda j: (1 - 1 / c) if j == 1 else (-1) ** j * sp.factorial(j - 1) / c**j
    h = lambda i: (1 / (2 * s) - D[1]) if i == 1 else (-1) ** (i - 1) * sp.factorial(i - 1) / (2 * s**i) - D[i]
    # derivatives of G: the s-derivative by total_d (only the D_k depend on s beyond the explicit s)
    dG = {}
    cur = {(0, 0): G}
    for i in range(0, MAXK + 1):
        for j in range(0, MAXK + 1 - i):
            if (i, j) == (0, 0):
                continue
            if i >= 1:
                e = total_d(cur[(i - 1, j)], s, s, 1)
                cur[(i, j)] = e
            else:
                e = sp.diff(cur[(i, j - 1)], c)
                cur[(i, j)] = sp.expand(e)
            if i == 0 and j >= 1:
                closed = -s * Dc(j)
            elif i == 1 and j >= 1:
                closed = -Dc(j)
            elif j == 0 and i == 1:
                closed = -(c - 1 - sp.log(c)) + h(1)
            elif j == 0 and i >= 2:
                closed = h(i)
            else:
                closed = sp.Integer(0)  # i >= 2 and j >= 1
            diff_ = sp.simplify(sp.expand(cur[(i, j)] - closed))
            if diff_ != 0:
                print("outer", (i, j), "FAILS:", diff_)
                ok = False
    print("outer partials of G(s, c), i + j <= 6:", "ok" if ok else "FAILED")
    return ok


def check_info():
    """Expected information: I_new = J' I_old J against H_ab(m, s); H against the Leibniz form."""
    phi = sp.Symbol("phi", positive=True)
    ss = sp.Symbol("ss", positive=True)  # s = 1/phi
    mm = sp.Symbol("mm", positive=True)  # mu
    # gamma1 (mu, phi): I_mumu = s/mu^2, I_phiphi = s^2/2 + s^4 delta''(s), I_muphi = 0
    s_of_phi = 1 / phi
    Iold = sp.Matrix([[s_of_phi / mm**2, 0], [0, s_of_phi**2 / 2 + s_of_phi**4 * D[2]]])
    # Jacobian d(mu, phi)/d(mu, sigma2), with phi = sigma2 / mu^2
    J = sp.Matrix([[1, 0], [-2 * phi / mm, 1 / mm**2]])
    Inew = (J.T * Iold * J).subs(phi, 1 / ss)
    H = {
        (0, 0): (ss + 2 + 4 * ss**2 * D[2]) / mm**2,
        (0, 1): -(ss + 2 * ss**3 * D[2]) / mm**3,
        (1, 1): (ss**2 / 2 + ss**4 * D[2]) / mm**4,
    }
    ok = True
    for (a, b), h in H.items():
        d = sp.simplify(sp.expand(Inew[a, b] - h))
        print(f"I_new[{a},{b}] = J' I_old J against H: {'ok' if d == 0 else 'FAILED ' + str(d)}")
        ok &= d == 0
    # s-derivatives of H against the Leibniz form F^(j) of stirling_power_term
    def leib(n, j):
        return sum(sp.binomial(j, t) * sp.ff(n, t) * ss ** (n - t) * D[2 + j - t] for t in range(0, min(j, n) + 1))
    F = {
        0: lambda j: (ss + 2 if j == 0 else (1 if j == 1 else 0)) + 4 * leib(2, j),
        1: lambda j: -(ss if j == 0 else (1 if j == 1 else 0)) - 2 * leib(3, j),
        2: lambda j: sp.Rational(1, 2) * (ss**2 if j == 0 else (2 * ss if j == 1 else (2 if j == 2 else 0)))
        + leib(4, j),
    }
    for p, ((a, b), h) in enumerate(H.items()):
        k = 2 + p  # power of mu in H_ab
        Fp = {0: ss + 2 + 4 * ss**2 * D[2], 1: -(ss + 2 * ss**3 * D[2]), 2: ss**2 / 2 + ss**4 * D[2]}[p]
        assert sp.simplify(Fp * mm ** (-k) - h) == 0
        cur = Fp
        for i in range(0, 5):
            # the mu-derivatives: d^i mu^-k = (-1)^i (k+i-1)!/(k-1)! mu^-(k+i) times F(s)
            lhs_mu = sp.diff(h, mm, i)
            rhs = (-1) ** i * sp.factorial(k + i - 1) / sp.factorial(k - 1) / mm ** (k + i) * Fp
            if sp.simplify(sp.expand(lhs_mu - rhs)) != 0:
                print("mu-derivative of H FAILS", (a, b), i)
                ok = False
        for j in range(0, 5):
            # the s-derivatives of F, by the chain rule on the D_k, against the Leibniz form of gamma2.hpp
            cur_j = Fp
            for _ in range(j):
                cur_j = total_d(cur_j, ss, ss, 1)
            if sp.simplify(sp.expand(cur_j - F[p](j))) != 0:
                print("s-derivative of F FAILS", (a, b), j, sp.simplify(sp.expand(cur_j - F[p](j))))
                ok = False
    print("expected information, closed forms in (mu, s):", "ok" if ok else "FAILED")
    return ok


def direct_logpdf_forms(K_max=2):
    """The log-density in (mu, sigma2) as a chain of the symbols D_k, with derivatives to order K_max.
    Returns {(a, b): expression} with a derivatives in mu and b in sigma2."""
    sh = mu**2 / sig2
    l0 = -sh * (y / mu - 1 - sp.log(y / mu)) + sp.log(sh) / 2 - D[0] - sp.log(y)
    out = {}
    cur = {(0, 0): l0}
    for n in range(1, K_max + 1):
        for a in range(0, n + 1):
            b = n - a
            if a >= 1:
                src = cur[(a - 1, b)]
                cur[(a, b)] = sp.expand(total_d(src, mu, sh, sp.diff(sh, mu)))
            else:
                src = cur[(a, b - 1)]
                cur[(a, b)] = sp.expand(total_d(src, sig2, sh, sp.diff(sh, sig2)))
            out[(a, b)] = cur[(a, b)]
    return out


def check_direct():
    forms = direct_logpdf_forms(2)
    # the D_k with k >= 1 appear linearly in the orders 1 and 2; they are used numerically below
    print("direct closed forms of the log-density, orders 1 and 2, in (mu, sigma2): computed")
    return forms


# ---------------------------------------------------------------- mpmath part

def delta_mp(sv):
    return mpmath.loggamma(sv) - ((sv - mpmath.mpf(1) / 2) * mpmath.log(sv) - sv + mpmath.log(2 * mpmath.pi) / 2)


def logpdf_def(yv, muv, s2v):
    sv = muv * muv / s2v
    th = s2v / muv
    return (sv - 1) * mpmath.log(yv) - yv / th - sv * mpmath.log(th) - mpmath.loggamma(sv)


def log_cdf_def(upper, yv, muv, s2v):
    sv = muv * muv / s2v
    x = muv * yv / s2v
    if upper:
        return mpmath.log(mpmath.gammainc(sv, x, mpmath.inf, regularized=True))
    return mpmath.log(mpmath.gammainc(sv, 0, x, regularized=True))


def value_at(fun, yv, mu0, s0, K, b, prec):
    with mpmath.workprec(prec):
        f = lambda mv, sv: fun(mpmath.mpf(yv), mv, sv)
        return mpmath.diff(f, (mpmath.mpf(mu0), mpmath.mpf(s0)), (K - b, b))


def hx(v):
    return float(v).hex()


def task(pair):
    muv, phi = pair
    s2v = phi * muv * muv
    lines = []
    dropped = 0
    # kind 0: log-density; kind 1: log F and log(1 - F)
    for yc in CS:
        yv = muv * yc
        for kind, fun, upper in [(0, logpdf_def, 0), (1, lambda yy, mv, sv: log_cdf_def(0, yy, mv, sv), 0),
                                 (1, lambda yy, mv, sv: log_cdf_def(1, yy, mv, sv), 1)]:
            for K in range(1, MAXK + 1):
                for b in range(K + 1):
                    a = value_at(fun, yv, muv, s2v, K, b, 200)
                    cc = value_at(fun, yv, muv, s2v, K, b, 260)
                    if cc == 0 or not mpmath.isfinite(cc) or abs(a - cc) > AGREE * abs(cc):
                        dropped += 1
                        continue
                    hi = float(cc)
                    if not math.isfinite(hi):
                        dropped += 1
                        continue
                    lo = float(cc - mpmath.mpf(hi))
                    lines.append(f"{kind},{upper},{hx(yv)},{hx(muv)},{hx(s2v)},{K},{b},{hx(hi)},{hx(lo)}\n")
    return lines, dropped


def direct_rows(pair, forms):
    """kind 2: the SymPy direct closed forms of orders 1 and 2 at the same points, evaluated at 250 bits."""
    muv, phi = pair
    s2v = phi * muv * muv
    lines = []
    with mpmath.workprec(250):
        dl = [None] + [mpmath.diff(delta_mp, mpmath.mpf(muv * muv / s2v), k) for k in range(1, MAXK + 1)]
        d0 = delta_mp(mpmath.mpf(muv * muv / s2v))
        for yc in CS:
            yv = muv * yc
            subs = {mu: mpmath.mpf(muv), sig2: mpmath.mpf(s2v), y: mpmath.mpf(yv), D[0]: d0}
            for k in range(1, MAXK + 2):
                subs[D[k]] = dl[k] if k <= MAXK else mpmath.mpf(0)
            keys = list(subs.keys())
            for (a, b), e in forms.items():
                if a + b > 2:
                    continue
                f = sp.lambdify(keys, e, "mpmath")
                val = mpmath.mpf(f(*[subs[k] for k in keys]))
                hi = float(val)
                lo = float(val - mpmath.mpf(hi))
                lines.append(f"2,0,{hx(yv)},{hx(muv)},{hx(s2v)},{a + b},{b},{hx(hi)},{hx(lo)}\n")
    return lines


def main():
    ok = check_outer()
    ok &= check_info()
    forms = check_direct()
    if not ok:
        return 1
    with Pool(4) as pool:
        res = pool.map(task, [(mm_, ph) for mm_ in MUS for ph in PHIS])
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("kind,upper,y,mu,sigma2,K,b,hi,lo\n")
        for lines, dr in res:
            f.writelines(lines)
            n += len(lines)
            dropped += dr
        for pair in [(mm_, ph) for mm_ in MUS for ph in PHIS]:
            rows = direct_rows(pair, forms)
            f.writelines(rows)
            n += len(rows)
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
