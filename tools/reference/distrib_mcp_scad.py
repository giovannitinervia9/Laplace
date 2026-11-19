"""Reference values for mcp_distrib and scad_distrib (include/laplace/distrib/mcp.hpp,
scad.hpp), the properized MCP and SCAD, with each of the three properizing tails
(include/laplace/distrib/properizing_tail.hpp): the log-density and its derivatives in
(mu, lambda, p, theta_t), the logarithms of the cumulative functions, and the
normalizing constant itself.

Usage (from the repository root):  python -I tools/reference/distrib_mcp_scad.py [quick]

File tests/data/reference/distrib_mcp_scad.csv, one row per value:
    kind, law, tail, K, pos, y, hi, lo, mu, lam, p, t1, t2
law 0 is MCP, whose third parameter is gamma, and law 1 SCAD, whose third is a.
tail 0 is the Gaussian tail, whose only parameter t1 is the variance V; tail 1 is the
Student tail, whose t1 is the scale s and t2 the degrees of freedom nu; tail 2 is the
logistic tail, whose only parameter t1 is the scale s. t2 is written as one for a tail
that has no second parameter, and the law then has four parameters rather than five.
kind 0 is the derivative of order K of log f at position pos of the packed
sym_tensor<n_params, K> (K = 0 is the value, pos 0); kind 1 is log F, kind 2 log(1 - F)
and kind 3 the normalizing constant Z itself (K = 0, pos 0, y unused). y, hi, lo, mu,
lam, p, t1 and t2 are in hexadecimal, and the row stores y itself so that the library
and the reference see the same double (u = |y - mu| is formed the same way in both).

Method. The derivatives are SymPy's own, of the expressions below, evaluated by mpmath
at 300 and 400 bits, a row kept when the two agree to 2^-80 relative; no finite
difference, and none of the library's own identities or recursions. With u = |y - mu|,
K the last knot and H the tail's exponent, the penalty q is

    MCP   q = lambda u - u^2/(2 gamma)                  u <= K = gamma lambda
          q = gamma lambda^2/2 + H(u - K)               beyond
    SCAD  q = lambda u                                  u <= lambda
          q = lambda u - (u - lambda)^2/(2 c), c = a - 1 lambda < u <= K = a lambda
          q = (a + 1) lambda^2/2 + H(u - K)             beyond

    H = x^2/(2 V)                        the Gaussian tail
    H = (nu + 1)/2 log(1 + x^2/(nu s^2)) the Student tail
    H = 2 log cosh(x/s)                  the logistic tail

and log f = -q(u) - log Z. The derivatives of q are SymPy's, taken on the piece the
point lies in, with d/dmu carried through u = |y - mu| by its sign; those of log Z are
SymPy's on

    Z_MCP  = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma/2)) + e^(-gamma lambda^2/2) T ]
    Z_SCAD = 2 [ (1 - e^(-lambda^2))/lambda + e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2))
                 + e^(-(a + 1) lambda^2/2) T ],

written with F(x) = sqrt(pi)/2 e^(-x^2) erfi(x), Dawson's integral, and with the tail's
own constant T = int_0^inf exp(-H),

    T = sqrt(pi V / 2)                                  the Gaussian tail
    T = s sqrt(pi nu) Gamma(nu/2) / (2 Gamma((nu+1)/2))  the Student tail
    T = s                                               the logistic tail.

These closed constants are compared with the quadrature of the definition,
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

MU, LAM, P, T1, T2, UU = sp.symbols("mu lam p t1 t2 u", real=True)

# the number of parameters each tail adds, and the parameter sets to measure at:
# (mu, lambda, p, t1, t2); p is gamma for MCP and a for SCAD, so it exceeds one here
N_TAIL_PARAMS = {0: 1, 1: 2, 2: 1}
TAIL_NAMES = {0: "gaussian", 1: "student", 2: "logistic"}

PARAMS = {
    0: [(0.5, 1.3, 3.0, 2.0, 1.0),
        (0.0, 0.2, 7.0, 1.0, 1.0),
        (-2.0, 4.0, 1.25, 9.0, 1.0),
        (0.0, 0.01, 3.0, 1.0, 1.0),
        (0.0, 3.0, 1.1, 0.5, 1.0),
        (0.0, 30.0, 2.0, 2.0, 1.0)],
    1: [(0.5, 1.3, 3.0, 1.1, 5.0),
        (0.0, 0.2, 7.0, 1.0, 1.0),      # nu = 1, the Cauchy tail
        (-2.0, 4.0, 1.25, 3.0, 0.7),    # nu < 1, no mean
        (0.0, 0.01, 3.0, 1.0, 2.5),
        (0.0, 3.0, 1.1, 0.5, 40.0),     # nu large, close to the Gaussian tail
        (0.0, 30.0, 2.0, 2.0, 5.0)],
    2: [(0.5, 1.3, 3.0, 1.1, 1.0),
        (0.0, 0.2, 7.0, 1.0, 1.0),
        (-2.0, 4.0, 1.25, 3.0, 1.0),
        (0.0, 0.01, 3.0, 1.0, 1.0),
        (0.0, 3.0, 1.1, 0.5, 1.0),
        (0.0, 30.0, 2.0, 2.0, 1.0)],
}


# --------------------------------------------------------------------------- the tails
def h_sym(tail, x):
    """the tail's exponent H as a SymPy expression in x and the tail's parameters."""
    if tail == 0:
        return x**2 / (2 * T1)
    if tail == 1:
        return (T2 + 1) / 2 * sp.log(1 + x**2 / (T2 * T1**2))
    return 2 * sp.log(sp.cosh(x / T1))


def t_sym(tail):
    """the tail's constant T = int_0^inf exp(-H)."""
    if tail == 0:
        return sp.sqrt(sp.pi * T1 / 2)
    if tail == 1:
        return T1 * sp.sqrt(sp.pi * T2) * sp.gamma(T2 / 2) / (2 * sp.gamma((T2 + 1) / 2))
    return T1


def h_num(tail, x, t1, t2):
    """H in mpmath, for the quadratures of the definition."""
    if tail == 0:
        return x * x / (2 * t1)
    if tail == 1:
        return (t2 + 1) / 2 * mpmath.log1p(x * x / (t2 * t1 * t1))
    z = x / t1
    return 2 * (abs(z) - mpmath.log(2) + mpmath.log1p(mpmath.e ** (-2 * abs(z))))


# --------------------------------------------------------------------------- q and Z
def q_pieces(law, tail):
    """the pieces of q, with the upper end of each in u."""
    x = UU - P * LAM
    if law == 0:
        return [(P * LAM, LAM * UU - UU**2 / (2 * P)),
                (sp.oo, P * LAM**2 / 2 + h_sym(tail, x))]
    return [(LAM, LAM * UU),
            (P * LAM, LAM * UU - (UU - LAM) ** 2 / (2 * (P - 1))),
            (sp.oo, (P + 1) * LAM**2 / 2 + h_sym(tail, x))]


def z_closed_sym(law, tail):
    daw = lambda x: sp.sqrt(sp.pi) / 2 * sp.exp(-x**2) * sp.erfi(x)
    T = t_sym(tail)
    if law == 0:
        x = LAM * sp.sqrt(P / 2)
        return 2 * (sp.sqrt(2 * P) * daw(x) + sp.exp(-P * LAM**2 / 2) * T)
    c = P - 1
    return 2 * ((1 - sp.exp(-(LAM**2))) / LAM + sp.exp(-(LAM**2)) * sp.sqrt(2 * c) * daw(LAM * sp.sqrt(c / 2))
                + sp.exp(-(P + 1) * LAM**2 / 2) * T)


def q_num(law, tail, u, lam, p, t1, t2):
    if law == 0:
        if u <= p * lam:
            return lam * u - u * u / (2 * p)
        return p * lam * lam / 2 + h_num(tail, u - p * lam, t1, t2)
    if u <= lam:
        return lam * u
    c = p - 1
    if u <= p * lam:
        return lam * u - (u - lam) ** 2 / (2 * c)
    return (p + 1) * lam * lam / 2 + h_num(tail, u - p * lam, t1, t2)


def knots(law, lam, p):
    return [p * lam] if law == 0 else [lam, p * lam]


def z_quad(law, tail, lam, p, t1, t2):
    pts = [mpmath.mpf(0)] + knots(law, lam, p) + [mpmath.inf]
    return 2 * mpmath.quad(lambda u: mpmath.e ** (-q_num(law, tail, u, lam, p, t1, t2)), pts)


def upper_quad(law, tail, u, lam, p, t1, t2):
    """int_u^inf exp(-q) dt, factored by exp(-q(u)) so that the integrand starts at one,
    with the knots beyond u as cuts and subdivisions inside each stretch.

    The plain quadrature of exp(-q) over [u, inf) is not to be trusted far out in the
    tail, where the integrand spans hundreds of e-folds: with the Gaussian tail at
    lambda = 30, gamma = 2, V = 2 and u = 18 it returns 2.2011e-201 with the knot as
    the only cut and 2.1696e-201 with four more, while the value is
    2.1732381118224717e-201 (the factored quadrature here at 60 and at 120 digits,
    Dawson's integral through mpmath's erfi, and the asymptotic expansion of Dawson's
    integral all agree on it)."""
    q0 = q_num(law, tail, u, lam, p, t1, t2)
    g = lambda v: mpmath.e ** (-(q_num(law, tail, u + v, lam, p, t1, t2) - q0))
    pts = [mpmath.mpf(0)]
    prev = mpmath.mpf(0)
    for k in knots(law, lam, p):
        if k <= u:
            continue
        end = k - u
        for j in range(1, 13):
            pts.append(prev + (end - prev) * mpmath.mpf(j) / 12)
        prev = end
    step = max(t1, 1 / mpmath.mpf(lam))
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


def hexrow(kind, law, tail, K, pos, y, v, mu, lam, p, t1, t2):
    hi = float(v)
    lo = float(v - mpmath.mpf(hi))
    return "{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
        kind, law, tail, K, pos, float(y).hex(), hi.hex(), lo.hex(), float(mu).hex(),
        float(lam).hex(), float(p).hex(), float(t1).hex(), float(t2).hex())


def ev(expr, subs, prec):
    old = mpmath.mp.prec
    mpmath.mp.prec = prec
    try:
        return mpmath.mpf(sp.N(expr.subs(subs), int(prec * 0.302) + 10))
    finally:
        mpmath.mp.prec = old


def one_case(arg):
    law, tail, (mu0, lam0, p0, t10, t20), kmax, lz_sym, q_sym = arg
    nt = N_TAIL_PARAMS[tail]
    rows = []
    base = {LAM: sp.Float(lam0, 50), P: sp.Float(p0, 50), T1: sp.Float(t10, 50),
            T2: sp.Float(t20, 50)}
    lmu, lp = mpmath.mpf(lam0), mpmath.mpf(p0)
    lt1, lt2 = mpmath.mpf(t10), mpmath.mpf(t20)

    # the derivatives of log Z, which do not depend on y
    lz = {}
    for key, expr in lz_sym.items():
        lz[key] = (ev(expr, base, 300), ev(expr, base, 400))

    for y in y_grid(law, mpmath.mpf(mu0), lmu, lp):
        yf = float(y)
        yv = mpmath.mpf(yf)
        u = abs(yv - mpmath.mpf(mu0))
        vals = {}
        for prec in (300, 400):
            mpmath.mp.prec = prec
            zz = z_quad(law, tail, lmu, lp, lt1, lt2)
            up = upper_quad(law, tail, u, lmu, lp, lt1, lt2)
            r = up / zz  # the probability beyond u, the small side when u is large
            # log1p on the side that is close to one: at 400 bits, 1 - r rounds to one
            # as soon as r falls below 2^-400, and the logarithm of the rounded value
            # would be zero where the true one is -r
            below = yv <= mpmath.mpf(mu0)
            l_lower = mpmath.log(r) if below else mpmath.log1p(-r)
            l_upper = mpmath.log1p(-r) if below else mpmath.log(r)
            vals[prec] = (-q_num(law, tail, u, lmu, lp, lt1, lt2) - mpmath.log(zz),
                          l_lower, l_upper, zz)
        mpmath.mp.prec = 400
        for kind in (0, 1, 2, 3):
            a, b = vals[300][kind], vals[400][kind]
            if abs(a - b) <= AGREE * max(abs(b), 1):
                rows.append(hexrow(kind, law, tail, 0, 0, yf, b, mu0, lam0, p0, t10, t20))
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
                        for n1 in range(K + 1 - a - b - m):
                            n2 = K - a - b - m - n1
                            if nt == 1 and n2 != 0:
                                continue
                            expr = q_sym[(pc, a, b, m, n1, n2)]
                            out = []
                            for prec in (300, 400):
                                # d/dmu = -sign(y - mu) d/du, carried a times
                                r = -mpmath.mpf((-sgn) ** a) * ev(expr, su, prec)
                                if a == 0:
                                    r -= lz[(b, m, n1, n2)][0 if prec == 300 else 1]
                                out.append(r)
                            if abs(out[0] - out[1]) > AGREE * max(abs(out[1]), 1):
                                continue
                            idx = [0] * a + [1] * b + [2] * m + [3] * n1 + [4] * n2
                            rows.append(hexrow(0, law, tail, K, sym_pos(3 + nt, K, idx), yf,
                                               out[1], mu0, lam0, p0, t10, t20))
    return rows


def build_symbolic(law, tail, kmax):
    """the symbolic derivatives, differentiated once and evaluated at every case."""
    nt = N_TAIL_PARAMS[tail]
    lz_sym = {}
    logz = sp.log(z_closed_sym(law, tail))
    for tot in range(kmax + 1):
        for i, m, n1, n2 in itertools.product(range(tot + 1), repeat=4):
            if i + m + n1 + n2 != tot or (nt == 1 and n2 != 0):
                continue
            d = logz
            for var, o in ((LAM, i), (P, m), (T1, n1), (T2, n2)):
                if o:
                    d = sp.diff(d, var, o)
            lz_sym[(i, m, n1, n2)] = sp.simplify(d) if tot <= 1 else d
    q_sym = {}
    for pc, (_end, expr) in enumerate(q_pieces(law, tail)):
        for tot in range(kmax + 1):
            for a, b, m, n1, n2 in itertools.product(range(tot + 1), repeat=5):
                if a + b + m + n1 + n2 != tot or (nt == 1 and n2 != 0):
                    continue
                d = expr
                for var, o in ((UU, a), (LAM, b), (P, m), (T1, n1), (T2, n2)):
                    if o:
                        d = sp.diff(d, var, o)
                q_sym[(pc, a, b, m, n1, n2)] = d
    return lz_sym, q_sym


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    kmax = 2 if quick else 6
    tails = (0, 1, 2)
    mpmath.mp.prec = 400
    worst = mpmath.mpf(0)
    for law in (0, 1):
        for tail in tails:
            zc = z_closed_sym(law, tail)
            for mu, lam, p, t1, t2 in PARAMS[tail]:
                a = ev(zc, {LAM: sp.Float(lam, 50), P: sp.Float(p, 50), T1: sp.Float(t1, 50),
                            T2: sp.Float(t2, 50)}, 400)
                b = z_quad(law, tail, mpmath.mpf(lam), mpmath.mpf(p), mpmath.mpf(t1),
                           mpmath.mpf(t2))
                worst = max(worst, abs(a - b) / b)
    print("closed Z against the quadrature of the definition: worst relative error",
          mpmath.nstr(worst, 3))
    assert worst < mpmath.mpf(2) ** -90, "the closed normalizing constant disagrees with the definition"

    jobs = []
    for law in (0, 1):
        for tail in tails:
            lz_sym, q_sym = build_symbolic(law, tail, kmax)
            for pr in PARAMS[tail]:
                jobs.append((law, tail, pr, kmax, lz_sym, q_sym))
    with Pool(min(8, len(jobs))) as pool:
        parts = pool.map(one_case, jobs)
    rows = [r for part in parts for r in part]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("kind,law,tail,K,pos,y,hi,lo,mu,lam,p,t1,t2\n")
        for r in rows:
            f.write(r + "\n")
    print(f"{len(rows)} rows written to {OUT}")


if __name__ == "__main__":
    main()
