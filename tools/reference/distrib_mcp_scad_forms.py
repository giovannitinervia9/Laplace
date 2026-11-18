"""Derivation and check of the closed forms of mcp_distrib and scad_distrib
(include/laplace/distrib/mcp.hpp, scad.hpp), the properized MCP and SCAD.

Usage (from the repository root):  python -I tools/reference/distrib_mcp_scad_forms.py

Nothing is written: the script derives with SymPy and checks, as
tools/reference/bijector_forms.py does. Two things are checked.

1. The mixed partial derivatives of the penalty q. With u = |y - mu|, theta =
   (mu, lambda, p, V) and p = gamma (MCP) or a (SCAD), the log-density is

       l = -q(u; lambda, p, V) - log Z(lambda, p, V),

   and q is, piece by piece, a polynomial of degree two in u whose coefficients
   are rational in the parameters:

       MCP,  u <= gamma lambda:   q = lambda u - u^2/(2 gamma)
       MCP,  u >  gamma lambda:   q = gamma lambda^2/2 + (u - gamma lambda)^2/(2 V)
       SCAD, u <= lambda:         q = lambda u
       SCAD, lambda < u <= a lambda: q = lambda u - (u - lambda)^2/(2 c),  c = a - 1
       SCAD, u >  a lambda:       q = (a + 1) lambda^2/2 + (u - a lambda)^2/(2 V).

   The functions q_mixed_mcp and q_mixed_scad below are the Python mirror of
   the C++ component functions: they return one mixed partial
   d^j_u d^i_lambda d^m_p d^n_V q, given the four orders. Each is compared with
   SymPy's own derivative of the piece, at points inside every piece, for every
   (j, i, m, n) of total order up to six.

2. The derivatives of log Z. The normalizing constants are

       Z_MCP  = 2 [ sqrt(2 gamma) F(x)   + e^(-gamma lambda^2/2)   sqrt(pi V/2) ]
       Z_SCAD = 2 [ (1 - e^(-lambda^2))/lambda + e^(-lambda^2) sqrt(2 c) F(x_c)
                    + e^(-(a + 1) lambda^2/2) sqrt(pi V/2) ]

   with F Dawson's integral, x = lambda sqrt(gamma/2) and x_c = lambda sqrt(c/2)
   (derived by completing the square in each piece; checked against mpmath
   quadrature of the definition in tools/reference/distrib_mcp_scad.py). In both,
   V appears only through the factor sqrt(pi V/2) multiplying a function of
   (lambda, p), so the table of Z over the three parameters is built from two
   tables over (lambda, p) and the derivatives of V^(1/2). The tables themselves
   are the Leibniz and chain-rule recursions over multi-indices (mul2, exp2,
   comp2, log3 here, the same functions in the header), and the script checks
   the finished log Z table against SymPy's derivatives of the closed Z written
   with erfi, F(x) = sqrt(pi)/2 e^(-x^2) erfi(x).
"""
import itertools
import math
import sys

import mpmath
import sympy as sp

mpmath.mp.dps = 50
KMAX = 6


# ---------------------------------------------------------------------------
# small helpers, mirrored in the header
# ---------------------------------------------------------------------------
def binom(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def d2(j, x):
    """d^j/dx^j of x^2."""
    return x * x if j == 0 else (2.0 * x if j == 1 else (2.0 if j == 2 else 0.0))


def negfac(m, r):
    """d^m/dr^m of 1/r = (-1)^m m! / r^(m+1)."""
    return (-1.0) ** m * math.factorial(m) / r ** (m + 1)


def lin(j, x):
    """d^j/dx^j of x: x, 1, 0, ..."""
    return x if j == 0 else (1.0 if j == 1 else 0.0)


def half_pow(m, r):
    """d^m/dr^m of r^(1/2) = c_m r^(1/2 - m), c_m = prod_(t<m) (1/2 - t)."""
    c = 1.0
    for t in range(m):
        c *= 0.5 - t
    return c * r ** (0.5 - m)


# ---------------------------------------------------------------------------
# 1. the mixed partials of q, one component at a time
# ---------------------------------------------------------------------------
def q_rate_term(j, i, u, lam):
    """d^j_u d^i_lambda of lambda u."""
    if j > 1 or i > 1:
        return 0.0
    return lin(j, u) * lin(i, lam)


def q_tail_square(j, i, m, u, lam, p):
    """d^j_u d^i_lambda d^m_p of (u - p lambda)^2 / 2, the tail of both laws."""
    s = 0.0
    if i == 0 and m == 0:
        s += 0.5 * d2(j, u)
    if j <= 1 and i <= 1 and m <= 1:
        s -= lin(j, u) * lin(i, lam) * lin(m, p)
    if j == 0:
        s += 0.5 * d2(i, lam) * d2(m, p)
    return s


def q_mixed_mcp(j, i, m, n, u, lam, gam, V):
    """d^j_u d^i_lambda d^m_gamma d^n_V q for the properized MCP."""
    if u <= gam * lam:  # inner piece: lambda u - u^2/(2 gamma), no V
        if n > 0:
            return 0.0
        s = q_rate_term(j, i, u, lam) if m == 0 else 0.0
        if i == 0:
            s += -0.5 * d2(j, u) * negfac(m, gam)
        return s
    # tail piece: gamma lambda^2/2 + (u - gamma lambda)^2/(2 V)
    s = 0.0
    if n == 0 and j == 0:
        if m == 0:
            s += 0.5 * d2(i, lam) * gam
        elif m == 1:
            s += 0.5 * d2(i, lam)
    s += q_tail_square(j, i, m, u, lam, gam) * negfac(n, V)
    return s


def q_mixed_scad(j, i, m, n, u, lam, a, V):
    """d^j_u d^i_lambda d^m_a d^n_V q for the properized SCAD."""
    if u <= lam:  # lambda u
        return q_rate_term(j, i, u, lam) if (m == 0 and n == 0) else 0.0
    if u <= a * lam:  # lambda u - (u - lambda)^2/(2 c), c = a - 1
        if n > 0:
            return 0.0
        s = q_rate_term(j, i, u, lam) if m == 0 else 0.0
        v = u - lam
        tot = j + i
        if tot == 0:
            w = 0.5 * v * v
        elif tot == 1:
            w = (-1.0) ** i * v
        elif tot == 2:
            w = (-1.0) ** i
        else:
            w = 0.0
        s += -w * negfac(m, a - 1.0)
        return s
    # tail piece: (a + 1) lambda^2/2 + (u - a lambda)^2/(2 V)
    s = 0.0
    if n == 0 and j == 0:
        if m == 0:
            s += 0.5 * d2(i, lam) * (a + 1.0)
        elif m == 1:
            s += 0.5 * d2(i, lam)
    s += q_tail_square(j, i, m, u, lam, a) * negfac(n, V)
    return s


# ---------------------------------------------------------------------------
# 2. tables of partial derivatives over multi-indices, mirrored in the header
# ---------------------------------------------------------------------------
def mul2(f, g, K):
    """Leibniz product of two tables over two variables."""
    h = {}
    for i in range(K + 1):
        for m in range(K + 1 - i):
            s = 0.0
            for a in range(i + 1):
                for b in range(m + 1):
                    s += binom(i, a) * binom(m, b) * f.get((a, b), 0.0) * g.get((i - a, m - b), 0.0)
            h[(i, m)] = s
    return h


def add2(f, g, K):
    return {(i, m): f.get((i, m), 0.0) + g.get((i, m), 0.0) for i in range(K + 1) for m in range(K + 1 - i)}


def first_index(i, m):
    """alpha - e_r and r, r the first nonzero coordinate of alpha = (i, m)."""
    return (i - 1, m, 0) if i > 0 else (i, m - 1, 1)


def exp2(phi, K):
    """table of exp(phi) from the table of phi: D^(gamma + e_r) e^phi =
    sum_(beta <= gamma) C(gamma, beta) D^beta e^phi D^(gamma - beta + e_r) phi."""
    E = {(0, 0): math.exp(phi[(0, 0)])}
    for tot in range(1, K + 1):
        for i in range(tot + 1):
            m = tot - i
            gi, gm, r = first_index(i, m)
            s = 0.0
            for bi in range(gi + 1):
                for bm in range(gm + 1):
                    di = gi - bi + (1 if r == 0 else 0)
                    dm = gm - bm + (1 if r == 1 else 0)
                    s += binom(gi, bi) * binom(gm, bm) * E[(bi, bm)] * phi.get((di, dm), 0.0)
            E[(i, m)] = s
    return E


def comp2(fd, x, K):
    """table of f(x) from the values f^(k)(x_0), k = 0..K, and the table of x.
    The tables of f^(k)(x) are built downward from k = K, each to total order
    K - k: D^(gamma + e_r) f^(k)(x) = sum_(beta <= gamma) C(gamma, beta)
    D^beta f^(k+1)(x) D^(gamma - beta + e_r) x."""
    tabs = [{(0, 0): fd[k]} for k in range(K + 1)]
    for k in range(K - 1, -1, -1):
        for tot in range(1, K - k + 1):
            for i in range(tot + 1):
                m = tot - i
                gi, gm, r = first_index(i, m)
                s = 0.0
                for bi in range(gi + 1):
                    for bm in range(gm + 1):
                        di = gi - bi + (1 if r == 0 else 0)
                        dm = gm - bm + (1 if r == 1 else 0)
                        s += binom(gi, bi) * binom(gm, bm) * tabs[k + 1].get((bi, bm), 0.0) * x.get((di, dm), 0.0)
                tabs[k][(i, m)] = s
    return tabs[0]


def log3(F, K):
    """table of log(F) over three variables from the table of F (F_0 > 0):
    F_(gamma + e_r) = F_0 U_(gamma + e_r) + sum_(0 != beta <= gamma)
    C(gamma, beta) F_beta U_(gamma - beta + e_r)."""
    U = {(0, 0, 0): math.log(F[(0, 0, 0)])}
    f0 = F[(0, 0, 0)]
    for tot in range(1, K + 1):
        for i in range(tot + 1):
            for m in range(tot - i + 1):
                n = tot - i - m
                if i > 0:
                    gi, gm, gn, r = i - 1, m, n, 0
                elif m > 0:
                    gi, gm, gn, r = i, m - 1, n, 1
                else:
                    gi, gm, gn, r = i, m, n - 1, 2
                s = F[(i, m, n)]
                for bi in range(gi + 1):
                    for bm in range(gm + 1):
                        for bn in range(gn + 1):
                            if bi + bm + bn == 0:
                                continue
                            di = gi - bi + (1 if r == 0 else 0)
                            dm = gm - bm + (1 if r == 1 else 0)
                            dn = gn - bn + (1 if r == 2 else 0)
                            s -= (
                                binom(gi, bi)
                                * binom(gm, bm)
                                * binom(gn, bn)
                                * F[(bi, bm, bn)]
                                * U[(di, dm, dn)]
                            )
                U[(i, m, n)] = s / f0
    return U


# ---------------------------------------------------------------------------
# Dawson's F and its derivatives, here from mpmath (the header uses math/dawson.hpp)
# ---------------------------------------------------------------------------
def dawson_derivs(x, K):
    """F^(k)(x), k = 0..K, from the definition at 50 digits."""
    f = lambda t: mpmath.e ** (-t * t) * mpmath.quad(lambda s: mpmath.e ** (s * s), [0, t])
    out = [float(f(mpmath.mpf(x)))]
    for k in range(1, K + 1):
        out.append(float(mpmath.diff(f, mpmath.mpf(x), k)))
    return out


# ---------------------------------------------------------------------------
# the tables of Z and of log Z, mirrored in the header
# ---------------------------------------------------------------------------
def z_tables_mcp(lam, gam, V, K):
    """tables over (lambda, gamma) of A = sqrt(2 gamma) F(x) and B = e^(-gamma lambda^2/2)."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    x = {}
    for m in range(K + 1):
        h = inv_sqrt2 * half_pow(m, gam)
        x[(0, m)] = lam * h
        if K >= 1:
            x[(1, m)] = h
    fd = dawson_derivs(x[(0, 0)], K)
    G = comp2(fd, x, K)
    sq = {(0, m): math.sqrt(2.0) * half_pow(m, gam) for m in range(K + 1)}
    A = mul2(sq, G, K)
    phi = {
        (0, 0): -0.5 * gam * lam * lam,
        (1, 0): -gam * lam,
        (2, 0): -gam,
        (0, 1): -0.5 * lam * lam,
        (1, 1): -lam,
        (2, 1): -1.0,
    }
    B = exp2(phi, K)
    return A, B


def g_derivs_scad(lam, K):
    """d^i/dlambda^i of (1 - e^(-lambda^2))/lambda, by the series for small lambda
    (single-signed tail) and by Leibniz over (1 - e^(-lambda^2)) and 1/lambda above."""
    out = []
    if lam < 1.0:
        for i in range(K + 1):
            s = 0.0
            k = 1
            while True:
                e = 2 * k - 1
                if e >= i:
                    t = (-1.0) ** (k + 1) * math.factorial(e) / (math.factorial(e - i) * math.factorial(k)) * lam ** (e - i)
                    s += t
                    if abs(t) <= 1e-22 * abs(s) and e - i > 2:
                        break
                k += 1
                if k > 200:
                    break
            out.append(s)
        return out
    e = [math.exp(-lam * lam)]  # e[j] = d^j e^(-lambda^2)
    for j in range(1, K + 1):
        prev = e[j - 2] if j >= 2 else 0.0
        e.append(-2.0 * lam * e[j - 1] - 2.0 * (j - 1) * prev)
    for i in range(K + 1):
        s = 0.0
        for j in range(i + 1):  # negfac(k, lambda) is d^k/dlambda^k of 1/lambda
            num = -mpmath.expm1(-lam * lam) if j == 0 else -e[j]
            s += binom(i, j) * float(num) * negfac(i - j, lam)
        out.append(s)
    return out


def z_tables_scad(lam, a, V, K):
    """tables over (lambda, a) of A = (1 - e^(-lambda^2))/lambda
    + e^(-lambda^2) sqrt(2 c) F(x_c) and B = e^(-(a + 1) lambda^2/2), c = a - 1."""
    c = a - 1.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    g = {(i, 0): v for i, v in enumerate(g_derivs_scad(lam, K))}
    phi_e = {(0, 0): -lam * lam, (1, 0): -2.0 * lam, (2, 0): -2.0}
    E = exp2(phi_e, K)
    sc = {(0, m): math.sqrt(2.0) * half_pow(m, c) for m in range(K + 1)}
    x = {}
    for m in range(K + 1):
        h = inv_sqrt2 * half_pow(m, c)
        x[(0, m)] = lam * h
        if K >= 1:
            x[(1, m)] = h
    fd = dawson_derivs(x[(0, 0)], K)
    G = comp2(fd, x, K)
    A = add2(g, mul2(mul2(E, sc, K), G, K), K)
    phi_b = {
        (0, 0): -0.5 * (a + 1.0) * lam * lam,
        (1, 0): -(a + 1.0) * lam,
        (2, 0): -(a + 1.0),
        (0, 1): -0.5 * lam * lam,
        (1, 1): -lam,
        (2, 1): -1.0,
    }
    B = exp2(phi_b, K)
    return A, B


def log_z_table(A, B, V, K):
    """table of log Z over (lambda, p, V) from the two tables over (lambda, p):
    Z = 2 [A + B sqrt(pi V/2)]."""
    root = math.sqrt(math.pi / 2.0)
    Z = {}
    for i in range(K + 1):
        for m in range(K + 1 - i):
            for n in range(K + 1 - i - m):
                v = 2.0 * root * B.get((i, m), 0.0) * half_pow(n, V)
                if n == 0:
                    v += 2.0 * A.get((i, m), 0.0)
                Z[(i, m, n)] = v
    return log3(Z, K)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
U, LAM, P, VV = sp.symbols("u lam p V", positive=True)

Q_PIECES = {
    "mcp inner": (LAM * U - U**2 / (2 * P), "mcp", "inner"),
    "mcp tail": (P * LAM**2 / 2 + (U - P * LAM) ** 2 / (2 * VV), "mcp", "tail"),
    "scad piece 0": (LAM * U, "scad", "0"),
    "scad piece 1": (LAM * U - (U - LAM) ** 2 / (2 * (P - 1)), "scad", "1"),
    "scad piece 2": ((P + 1) * LAM**2 / 2 + (U - P * LAM) ** 2 / (2 * VV), "scad", "2"),
}

# (u, lambda, p, V) inside each piece
Q_POINTS = {
    "mcp inner": [(0.7, 1.3, 3.0, 2.0), (2.9, 1.0, 3.0, 0.5), (1e-4, 0.2, 7.0, 1.0), (0.3, 4.0, 0.25, 9.0)],
    "mcp tail": [(5.0, 1.3, 3.0, 2.0), (12.0, 1.0, 3.0, 0.5), (3.0, 0.2, 7.0, 1.0), (2.0, 4.0, 0.25, 9.0)],
    "scad piece 0": [(0.6, 1.3, 3.7, 2.0), (0.05, 0.2, 2.5, 1.0)],
    "scad piece 1": [(2.0, 1.3, 3.7, 2.0), (0.4, 0.2, 2.5, 1.0), (1.1, 1.0, 1.2, 4.0)],
    "scad piece 2": [(9.0, 1.3, 3.7, 2.0), (1.0, 0.2, 2.5, 1.0), (2.0, 1.0, 1.2, 4.0)],
}


def check_q():
    worst = 0.0
    worst_at = None
    for name, (expr, law, _piece) in Q_PIECES.items():
        for pt in Q_POINTS[name]:
            u, lam, p, V = pt
            subs = {U: sp.Float(u, 40), LAM: sp.Float(lam, 40), P: sp.Float(p, 40), VV: sp.Float(V, 40)}
            for tot in range(0, KMAX + 1):
                for j, i, m, n in itertools.product(range(tot + 1), repeat=4):
                    if j + i + m + n != tot:
                        continue
                    d = expr
                    for var, order in ((U, j), (LAM, i), (P, m), (VV, n)):
                        if order:
                            d = sp.diff(d, var, order)
                    want = float(sp.N(d.subs(subs), 30))
                    got = q_mixed_mcp(j, i, m, n, u, lam, p, V) if law == "mcp" else q_mixed_scad(j, i, m, n, u, lam, p, V)
                    err = abs(got - want) / max(abs(want), 1.0)
                    if err > worst:
                        worst = err
                        worst_at = (name, pt, (j, i, m, n), got, want)
    print(f"q mixed partials, all (j, i, m, n) of total order <= {KMAX}: worst relative error {worst:.3e}")
    if worst_at and worst > 1e-12:
        print("  worst at", worst_at)
    return worst


def dawson_sym(x):
    return sp.sqrt(sp.pi) / 2 * sp.exp(-x**2) * sp.erfi(x)


def z_sym(law):
    """Z in closed form with SymPy's erfi, for the derivative check."""
    if law == "mcp":
        x = LAM * sp.sqrt(P / 2)
        return 2 * (sp.sqrt(2 * P) * dawson_sym(x) + sp.exp(-P * LAM**2 / 2) * sp.sqrt(sp.pi * VV / 2))
    c = P - 1
    xc = LAM * sp.sqrt(c / 2)
    return 2 * (
        (1 - sp.exp(-LAM**2)) / LAM
        + sp.exp(-LAM**2) * sp.sqrt(2 * c) * dawson_sym(xc)
        + sp.exp(-(P + 1) * LAM**2 / 2) * sp.sqrt(sp.pi * VV / 2)
    )


Z_POINTS = [(1.3, 3.0, 2.0), (0.2, 7.0, 1.0), (4.0, 0.25, 9.0), (0.01, 3.0, 1.0), (3.0, 1.1, 0.5)]


def check_log_z():
    worst = {"mcp": 0.0, "scad": 0.0}
    where = {"mcp": None, "scad": None}
    for law in ("mcp", "scad"):
        logz = sp.log(z_sym(law))
        for lam, p, V in Z_POINTS:
            if law == "scad" and p <= 1.0:
                p = p + 1.0
            subs = {LAM: sp.Float(lam, 40), P: sp.Float(p, 40), VV: sp.Float(V, 40)}
            A, B = z_tables_mcp(lam, p, V, KMAX) if law == "mcp" else z_tables_scad(lam, p, V, KMAX)
            tab = log_z_table(A, B, V, KMAX)
            for tot in range(0, KMAX + 1):
                for i, m, n in itertools.product(range(tot + 1), repeat=3):
                    if i + m + n != tot:
                        continue
                    d = logz
                    for var, order in ((LAM, i), (P, m), (VV, n)):
                        if order:
                            d = sp.diff(d, var, order)
                    want = float(sp.N(d.subs(subs), 30))
                    got = tab[(i, m, n)]
                    err = abs(got - want) / max(abs(want), 1e-8)
                    if err > worst[law]:
                        worst[law] = err
                        where[law] = ((lam, p, V), (i, m, n), got, want)
        print(f"log Z table, {law}, all (i, m, n) of total order <= {KMAX}: worst relative error {worst[law]:.3e}")
        if where[law] and worst[law] > 1e-9:
            print("  worst at", where[law])
    return max(worst.values())


def main():
    e1 = check_q()
    e2 = check_log_z()
    ok = e1 < 1e-12 and e2 < 1e-9
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
