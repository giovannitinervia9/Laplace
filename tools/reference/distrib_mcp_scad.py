"""Reference values for mcp_distrib and scad_distrib (include/laplace/distrib/mcp.hpp,
scad.hpp), the properized MCP and SCAD: the log-density and its derivatives in
(mu, lambda, p, V), the logarithms of the cumulative functions, and the normalizing
constant itself.

Usage (from the repository root):  python -I tools/reference/distrib_mcp_scad.py [quick]

File tests/data/reference/distrib_mcp_scad.csv, one row per value:
    kind, law, K, pos, y, hi, lo, mu, lam, p, V
law 0 is MCP, whose third parameter is gamma, and law 1 SCAD, whose third is a.
kind 0 is the derivative of order K of log f at position pos of the packed
sym_tensor<4, K> (K = 0 is the value, pos 0); kind 1 is log F, kind 2 log(1 - F) and
kind 3 the normalizing constant Z itself (K = 0, pos 0, y unused). y, hi, lo, mu, lam,
p and V are in hexadecimal, and the row stores y itself so that the library and the
reference see the same double (u = |y - mu| is formed the same way in both).

Method. The derivatives are SymPy's own, of the expressions below, evaluated by mpmath
at 300 and 400 bits, a row kept when the two agree to 2^-80 relative; no finite
difference and no series of the library's. With u = |y - mu|, the penalty q is the
piecewise polynomial

    MCP   q = lambda u - u^2/(2 gamma)                       u <= gamma lambda
          q = gamma lambda^2/2 + (u - gamma lambda)^2/(2 V)  beyond
    SCAD  q = lambda u                                       u <= lambda
          q = lambda u - (u - lambda)^2/(2 c), c = a - 1      lambda < u <= a lambda
          q = (a + 1) lambda^2/2 + (u - a lambda)^2/(2 V)     beyond

and log f = -q(u) - log Z. The derivatives of q are SymPy's, taken on the piece the
point lies in, with d/dmu carried through u = |y - mu| by its sign; those of log Z are
SymPy's on

    Z_MCP  = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma/2)) + e^(-gamma lambda^2/2) sqrt(pi V/2) ]
    Z_SCAD = 2 [ (1 - e^(-lambda^2))/lambda + e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2))
                 + e^(-(a + 1) lambda^2/2) sqrt(pi V/2) ],

written with F(x) = sqrt(pi)/2 e^(-x^2) erfi(x), Dawson's integral. These two closed
constants are compared with the quadrature of the definition,
Z = 2 int_0^inf e^(-q) du, at the start of main(), at every case and at 400 bits: that
comparison is what ties the reference to the definition, and it is also the check that
the density integrates to one. log F and log(1 - F) come from the quadrature of the
definition directly, not from a closed form.

Derivative rows are written only where u stands clear of every knot, since a
derivative in lambda or in p is one-sided there; the value, log F and log(1 - F) are
written at the knots as well, and the conventions at a knot and at the kink u = 0 are
checked by exact equality in tests/distrib/test_mcp_scad.cpp instead.
"""
import itertools
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_mcp_scad.csv"
AGREE = mpmath.mpf(2) ** -80

# (mu, lambda, p, V); p is gamma for MCP and a for SCAD, so it exceeds one here
PARAMS = [
    (0.5, 1.3, 3.0, 2.0),
    (0.0, 0.2, 7.0, 1.0),
    (-2.0, 4.0, 1.25, 9.0),
    (0.0, 0.01, 3.0, 1.0),
    (0.0, 3.0, 1.1, 0.5),
    (0.0, 30.0, 2.0, 2.0),
]

MU, LAM, P, VV, UU = sp.symbols("mu lam p V u", real=True)


def q_pieces(law):
    """the pieces of q in (u, lambda, p, V), with the upper end of each in u."""
    if law == 0:
        return [(P * LAM, LAM * UU - UU**2 / (2 * P)),
                (sp.oo, P * LAM**2 / 2 + (UU - P * LAM) ** 2 / (2 * VV))]
    return [(LAM, LAM * UU),
            (P * LAM, LAM * UU - (UU - LAM) ** 2 / (2 * (P - 1))),
            (sp.oo, (P + 1) * LAM**2 / 2 + (UU - P * LAM) ** 2 / (2 * VV))]


def z_closed_sym(law):
    daw = lambda x: sp.sqrt(sp.pi) / 2 * sp.exp(-x**2) * sp.erfi(x)
    root = sp.sqrt(sp.pi * VV / 2)
    if law == 0:
        x = LAM * sp.sqrt(P / 2)
        return 2 * (sp.sqrt(2 * P) * daw(x) + sp.exp(-P * LAM**2 / 2) * root)
    c = P - 1
    return 2 * ((1 - sp.exp(-(LAM**2))) / LAM + sp.exp(-(LAM**2)) * sp.sqrt(2 * c) * daw(LAM * sp.sqrt(c / 2))
                + sp.exp(-(P + 1) * LAM**2 / 2) * root)


# --------------------------------------------------------------------------- numeric q, for the definition
def q_num(law, u, lam, p, V):
    if law == 0:
        if u <= p * lam:
            return lam * u - u * u / (2 * p)
        return p * lam * lam / 2 + (u - p * lam) ** 2 / (2 * V)
    if u <= lam:
        return lam * u
    c = p - 1
    if u <= p * lam:
        return lam * u - (u - lam) ** 2 / (2 * c)
    return (p + 1) * lam * lam / 2 + (u - p * lam) ** 2 / (2 * V)


def knots(law, lam, p):
    return [p * lam] if law == 0 else [lam, p * lam]


def z_quad(law, lam, p, V):
    pts = [mpmath.mpf(0)] + knots(law, lam, p) + [mpmath.inf]
    return 2 * mpmath.quad(lambda u: mpmath.e ** (-q_num(law, u, lam, p, V)), pts)


def upper_quad(law, u, lam, p, V):
    """int_u^inf exp(-q) dt, factored by exp(-q(u)) so that the integrand starts at one,
    with the knots beyond u as cuts and subdivisions inside each stretch.

    The plain quadrature of exp(-q) over [u, inf) is not to be trusted far out in the
    tail, where the integrand spans hundreds of e-folds: at lambda = 30, gamma = 2,
    V = 2 and u = 18 it returns 2.2011e-201 with the knot as the only cut and
    2.1696e-201 with four more, while the value is 2.1732381118224717e-201 (the factored
    quadrature here at 60 and at 120 digits, Dawson's integral through mpmath's erfi, and
    the asymptotic expansion of Dawson's integral all agree on it)."""
    q0 = q_num(law, u, lam, p, V)
    g = lambda v: mpmath.e ** (-(q_num(law, u + v, lam, p, V) - q0))
    pts = [mpmath.mpf(0)]
    prev = mpmath.mpf(0)
    for k in knots(law, lam, p):
        if k <= u:
            continue
        end = k - u
        for j in range(1, 13):
            pts.append(prev + (end - prev) * mpmath.mpf(j) / 12)
        prev = end
    step = max(mpmath.sqrt(V), 1 / mpmath.mpf(lam))
    for j in range(-3, 8):
        pts.append(prev + step * mpmath.mpf(2) ** j)
    pts.append(mpmath.inf)
    return mpmath.e ** (-q0) * mpmath.quad(g, pts)


def y_grid(law, mu, lam, p):
    """the kink, every knot and its two sides, the middle of each piece, both tails."""
    ks = knots(law, lam, p)
    us = [mpmath.mpf(0)]
    for k in ks:
        us += [k * mpmath.mpf("0.3"), k * (1 - mpmath.mpf("1e-7")), k, k * (1 + mpmath.mpf("1e-7"))]
    big = ks[-1]
    us += [big * mpmath.mpf("1.5"), big + 3, big * 4 + 10]
    out = []
    for u in sorted(set(us)):
        out.append(mu + u)
        if u > 0:
            out.append(mu - u)
    return out


def clear_of_knots(law, u, lam, p):
    for k in knots(law, lam, p):
        if abs(u - k) <= mpmath.mpf("1e-6") * k:
            return False
    return u > mpmath.mpf("1e-6") * (1 + lam)


def piece_of(law, u, lam, p):
    for i, k in enumerate(knots(law, lam, p)):
        if u <= k:
            return i
    return len(knots(law, lam, p))


def sym_pos(P_, K, idx):
    pos, prev = 0, 0
    for j in range(K):
        for v in range(prev, idx[j]):
            n, k = P_ - v + K - j - 2, K - j - 1
            pos += math.comb(n, k) if 0 <= k <= n else 0
        prev = idx[j]
    return pos


def hexrow(kind, law, K, pos, y, v, mu, lam, p, V):
    hi = float(v)
    lo = float(v - mpmath.mpf(hi))
    return "{},{},{},{},{},{},{},{},{},{},{}".format(
        kind, law, K, pos, float(y).hex(), hi.hex(), lo.hex(), float(mu).hex(), float(lam).hex(),
        float(p).hex(), float(V).hex())


def ev(expr, subs, prec):
    old = mpmath.mp.prec
    mpmath.mp.prec = prec
    try:
        return mpmath.mpf(sp.N(expr.subs(subs), int(prec * 0.302) + 10))
    finally:
        mpmath.mp.prec = old


def one_case(arg):
    law, (mu0, lam0, p0, V0), kmax, lz_sym, q_sym = arg
    rows = []
    base = {LAM: sp.Float(lam0, 50), P: sp.Float(p0, 50), VV: sp.Float(V0, 50)}
    lmu, lp, lv = mpmath.mpf(lam0), mpmath.mpf(p0), mpmath.mpf(V0)

    # the derivatives of log Z, which do not depend on y
    lz = {}
    for key, expr in lz_sym.items():
        a = ev(expr, base, 300)
        b = ev(expr, base, 400)
        lz[key] = (a, b)

    mpmath.mp.prec = 400
    zq = z_quad(law, lmu, lp, lv)

    for y in y_grid(law, mpmath.mpf(mu0), lmu, lp):
        yf = float(y)
        yv = mpmath.mpf(yf)
        u = abs(yv - mpmath.mpf(mu0))
        vals = {}
        for prec in (300, 400):
            mpmath.mp.prec = prec
            zz = z_quad(law, lmu, lp, lv)
            up = upper_quad(law, u, lmu, lp, lv)
            r = up / zz  # the probability beyond u, the small side when u is large
            # log1p on the side that is close to one: at 400 bits, 1 - r rounds to one
            # as soon as r falls below 2^-400, and the logarithm of the rounded value
            # would be zero where the true one is -r
            below = yv <= mpmath.mpf(mu0)
            l_lower = mpmath.log(r) if below else mpmath.log1p(-r)
            l_upper = mpmath.log1p(-r) if below else mpmath.log(r)
            vals[prec] = (-q_num(law, u, lmu, lp, lv) - mpmath.log(zz), l_lower, l_upper, zz)
        mpmath.mp.prec = 400
        for kind in (0, 1, 2, 3):
            a, b = vals[300][kind], vals[400][kind]
            if abs(a - b) <= AGREE * max(abs(b), 1):
                rows.append(hexrow(kind, law, 0, 0, yf, b, mu0, lam0, p0, V0))
        if not clear_of_knots(law, u, lmu, lp):
            continue
        pc = piece_of(law, u, lmu, lp)
        su = {UU: sp.Float(float(u), 50)}
        su.update(base)
        sgn = 1 if yv > mpmath.mpf(mu0) else -1
        for K in range(1, kmax + 1):
            for a in range(K + 1):
                for b in range(K + 1 - a):
                    for m in range(K + 1 - a - b):
                        n = K - a - b - m
                        expr = q_sym[(pc, a, b, m, n)]
                        out = []
                        for prec in (300, 400):
                            # d/dmu = -sign(y - mu) d/du, carried a times
                            r = -mpmath.mpf((-sgn) ** a) * ev(expr, su, prec)
                            if a == 0:
                                r -= lz[(b, m, n)][0 if prec == 300 else 1]
                            out.append(r)
                        if abs(out[0] - out[1]) > AGREE * max(abs(out[1]), 1):
                            continue
                        idx = [0] * a + [1] * b + [2] * m + [3] * n
                        rows.append(hexrow(0, law, K, sym_pos(4, K, idx), yf, out[1], mu0, lam0, p0, V0))
    return rows


def build_symbolic(law, kmax):
    """the symbolic derivatives, differentiated once and evaluated at every case."""
    lz_sym = {}
    logz = sp.log(z_closed_sym(law))
    for tot in range(kmax + 1):
        for i, m, n in itertools.product(range(tot + 1), repeat=3):
            if i + m + n != tot:
                continue
            d = logz
            for var, o in ((LAM, i), (P, m), (VV, n)):
                if o:
                    d = sp.diff(d, var, o)
            lz_sym[(i, m, n)] = sp.simplify(d) if tot <= 1 else d
    q_sym = {}
    for pc, (_end, expr) in enumerate(q_pieces(law)):
        for tot in range(kmax + 1):
            for a, b, m, n in itertools.product(range(tot + 1), repeat=4):
                if a + b + m + n != tot:
                    continue
                d = expr
                for var, o in ((UU, a), (LAM, b), (P, m), (VV, n)):
                    if o:
                        d = sp.diff(d, var, o)
                q_sym[(pc, a, b, m, n)] = d
    return lz_sym, q_sym


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    kmax = 2 if quick else 6
    mpmath.mp.prec = 400
    worst = mpmath.mpf(0)
    for law in (0, 1):
        zc = z_closed_sym(law)
        for mu, lam, p, V in PARAMS:
            a = ev(zc, {LAM: sp.Float(lam, 50), P: sp.Float(p, 50), VV: sp.Float(V, 50)}, 400)
            b = z_quad(law, mpmath.mpf(lam), mpmath.mpf(p), mpmath.mpf(V))
            worst = max(worst, abs(a - b) / b)
    print("closed Z against the quadrature of the definition: worst relative error", mpmath.nstr(worst, 3))
    assert worst < mpmath.mpf(2) ** -90, "the closed normalizing constant disagrees with the definition"

    jobs = []
    for law in (0, 1):
        lz_sym, q_sym = build_symbolic(law, kmax)
        for pr in PARAMS:
            jobs.append((law, pr, kmax, lz_sym, q_sym))
    with Pool(min(4, len(jobs))) as pool:
        parts = pool.map(one_case, jobs)
    rows = [r for part in parts for r in part]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("kind,law,K,pos,y,hi,lo,mu,lam,p,V\n")
        for r in rows:
            f.write(r + "\n")
    print(f"{len(rows)} rows written to {OUT}")


if __name__ == "__main__":
    main()
