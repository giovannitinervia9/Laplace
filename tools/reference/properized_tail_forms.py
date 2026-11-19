"""The closed forms behind the properizing tails of mcp_distrib and scad_distrib,
each checked against SymPy's own derivatives (python -I tools/reference/properized_tail_forms.py).

A tail replaces the flat part of the penalty beyond the knot k by an exponent

    H(x; theta) = kappa(theta) Psi(x / varsigma(theta)),   x = u - k >= 0,

with H(0) = H'(0) = 0, so that the join at the knot stays C^1, and with varsigma a
MONOMIAL in the tail's own parameters, varsigma = prod_j theta_j^(e_j). Three tails:

    gaussian   Psi(z) = z^2/2,          varsigma = V^(1/2),      kappa = 1
    student    Psi(z) = log(1 + z^2),   varsigma = s nu^(1/2),   kappa = (nu + 1)/2
    logistic   Psi(z) = 2 log cosh z,   varsigma = s,            kappa = 1

Two identities carry every mixed partial the laws need.

(1) The knot. Both laws have k = p lambda (p = gamma for MCP, a for SCAD), so for any
    F of x = u - p lambda,

      d^j_u d^i_lambda d^m_p F(x) = sum_r c_r(i, m) F^(j + r)(x),
      c_r(i, m) = C(i, m + i - r) (-1)^m m!/(r - i)! lambda^(r - i) (-p)^(r - m),

    over max(i, m) <= r <= i + m. The derivatives in u only shift the order, since the
    coefficients do not depend on u.

(2) The tail's own parameters. With D = varsigma d/dvarsigma and E = z d/dz, every
    theta_j enters through varsigma only, so theta_j d/dtheta_j = e_j D, and the
    Stirling numbers of the first kind turn d^n into (theta d/dtheta)^k:

      d^n1_theta1 d^n2_theta2 G = theta1^(-n1) theta2^(-n2)
          sum_(k1, k2) s1(n1, k1) s1(n2, k2) e1^k1 e2^k2 D^(k1 + k2) G,

      D^m [varsigma^(-r) Psi^(r)(z)] = (-1)^m varsigma^(-r)
          sum_q C(m, q) r^(m - q) sum_j S2(q, j) z^j Psi^(r + j)(z),

    the second line because D acts on a function of z as -E and on varsigma^(-r) as
    multiplication by -r, and E^q f = sum_j S2(q, j) z^j f^(j).

So one component of the tail piece of q is a finite sum of values of Psi, with integer
coefficients and powers of z, r, e_j and 1/theta_j: no Bell polynomials, no tables.
"""

import math
from fractions import Fraction

import sympy as sp

ORDER = 6  # the total order the laws need in the parameters
RMAX = 18  # the highest derivative of Psi reached: r + j with r <= J + K and j <= K


# ---------------------------------------------------------------------------
# Stirling numbers
# ---------------------------------------------------------------------------

def stirling1_table(n):
    """s1[n][k], signed, from x(x-1)...(x-n+1) = sum_k s1(n, k) x^k."""
    t = [[0] * (n + 1) for _ in range(n + 1)]
    t[0][0] = 1
    for i in range(1, n + 1):
        for k in range(i + 1):
            t[i][k] = (t[i - 1][k - 1] if k else 0) - (i - 1) * t[i - 1][k]
    return t


def stirling2_table(n):
    """S2[n][k], from E^n f = sum_k S2(n, k) z^k f^(k)."""
    t = [[0] * (n + 1) for _ in range(n + 1)]
    t[0][0] = 1
    for i in range(1, n + 1):
        for k in range(1, i + 1):
            t[i][k] = t[i - 1][k - 1] + k * t[i - 1][k]
    return t


S1 = stirling1_table(ORDER)
S2 = stirling2_table(ORDER)


def check_stirling():
    x = sp.Symbol("x")
    for n in range(ORDER + 1):
        fall = sp.prod([x - i for i in range(n)]) if n else sp.Integer(1)
        poly = sum(S1[n][k] * x**k for k in range(n + 1))
        assert sp.expand(fall - poly) == 0, n
    z = sp.Symbol("z", positive=True)
    f = sp.Function("f")
    for n in range(1, ORDER + 1):
        lhs = f(z)
        for _ in range(n):
            lhs = sp.expand(z * sp.diff(lhs, z))
        rhs = sum(S2[n][k] * z**k * sp.diff(f(z), z, k) for k in range(n + 1))
        assert sp.simplify(lhs - rhs) == 0, n
    print("  Stirling tables: falling factorial and E^n both reproduced to order %d" % ORDER)


# ---------------------------------------------------------------------------
# (1) the chain through the knot, x = u - p lambda
# ---------------------------------------------------------------------------

def x_chain_coef(r, i, m, lam, p):
    """c_r(i, m) of identity (1); zero outside max(i, m) <= r <= i + m."""
    if r < max(i, m) or r > i + m:
        return 0.0
    t = m + i - r
    c = math.comb(i, t) * ((-1.0) ** m) * math.factorial(m) / math.factorial(r - i)
    return c * lam ** (r - i) * (-p) ** (r - m)


def check_x_chain():
    u, lam, p = sp.symbols("u lam p", positive=True)
    F = sp.Function("F")
    x = u - p * lam
    worst = 0.0
    at = {u: exact(7 / 3), lam: exact(1.25), p: exact(2.2)}
    # concrete F: a function whose derivatives are all distinct and nonzero
    w = sp.Symbol("w")
    Fc = sp.exp(sp.Rational(3, 7) * w) * sp.sin(sp.Rational(2, 5) * w)
    for j in range(ORDER + 1):
        for i in range(ORDER + 1 - j):
            for m in range(ORDER + 1 - j - i):
                lhs = sp.diff(Fc.subs(w, u - p * lam), u, j, lam, i, p, m).subs(at)
                rhs = sum(
                    x_chain_coef(r, i, m, float(at[lam]), float(at[p]))
                    * float(sp.diff(Fc, w, j + r).subs(w, (u - p * lam).subs(at)))
                    for r in range(max(i, m), i + m + 1)
                )
                a, b = float(lhs), float(rhs)
                err = abs(a - b) / max(abs(a), 1e-300)
                worst = max(worst, err)
    print("  knot chain d^j_u d^i_lambda d^m_p F(u - p lambda): worst relative error %.3e" % worst)
    assert worst < 1e-11


# ---------------------------------------------------------------------------
# (2) the tail's parameters, and the three tails
# ---------------------------------------------------------------------------

class Tail:
    """name, Psi as a SymPy expression in z, varsigma's exponents, kappa."""

    def __init__(self, name, psi, pnames, exps, varsigma, kappa):
        self.name = name
        self.psi = psi
        self.pnames = pnames
        self.exps = exps          # e_j, the exponent of theta_j in varsigma
        self.varsigma = varsigma  # callable(theta) -> varsigma
        self.kappa = kappa        # callable(theta, b) -> d^b kappa / dtheta_2^b (b = 0, 1)


Z = sp.Symbol("z")


def sqrt_of(v):
    """the square root in the precision of v, so that a quadrature run in mpmath does
    not build its scale as a double and then measure that rounding as an error."""
    import mpmath as mp
    return mp.sqrt(v) if isinstance(v, mp.mpf) else math.sqrt(v)


def one_of(v):
    import mpmath as mp
    return mp.mpf(1) if isinstance(v, mp.mpf) else 1.0


TAILS = [
    Tail("gaussian", Z**2 / 2, ["V"], [sp.Rational(1, 2)],
         lambda t: sqrt_of(t[0]), lambda t, b: one_of(t[0]) if b == 0 else 0 * t[0]),
    Tail("student", sp.log(1 + Z**2), ["s", "nu"], [sp.Integer(1), sp.Rational(1, 2)],
         lambda t: t[0] * sqrt_of(t[1]),
         lambda t, b: (t[1] + one_of(t[1])) / 2 if b == 0 else (one_of(t[1]) / 2 if b == 1 else 0 * t[1])),
    Tail("logistic", 2 * sp.log(sp.cosh(Z)), ["s"], [sp.Integer(1)],
         lambda t: t[0], lambda t, b: one_of(t[0]) if b == 0 else 0 * t[0]),
]

PSI_D = {}  # (tail name, k) -> a float callable for Psi^(k)


def psi_derivs(tail):
    if tail.name in PSI_D:
        return PSI_D[tail.name]
    fs = [sp.lambdify(Z, sp.diff(tail.psi, Z, k), "mpmath") for k in range(RMAX + 1)]
    PSI_D[tail.name] = fs
    return fs


def psi_chain(tail, r, n, z, theta):
    """d^r_x d^n_theta of Psi(x / varsigma) at the point, identity (2)."""
    fs = psi_derivs(tail)
    vs = tail.varsigma(theta)
    nt = len(tail.pnames)
    e = [float(x) for x in tail.exps]
    total = 0.0
    k_ranges = [range(n[j] + 1) for j in range(nt)]

    def walk(j, ks):
        nonlocal total
        if j == nt:
            coef = 1.0
            for v in range(nt):
                coef *= S1[n[v]][ks[v]] * e[v] ** ks[v]
            if coef == 0.0:
                return
            m = sum(ks)
            dm = 0.0
            for q in range(m + 1):
                inner = 0.0
                for jj in range(q + 1):
                    if S2[q][jj] == 0:
                        continue
                    inner += S2[q][jj] * z**jj * float(fs[r + jj](z))
                dm += math.comb(m, q) * (r ** (m - q) if (m - q) or r else 1.0) * inner
            total += coef * ((-1.0) ** m) * dm
            return
        for k in k_ranges[j]:
            walk(j + 1, ks + [k])

    walk(0, [])
    den = 1.0
    for v in range(nt):
        den *= theta[v] ** n[v]
    return total / (den * vs**r)


def h_component(tail, r, n, x, theta):
    """d^r_x d^n_theta H(x; theta), H = kappa Psi(x / varsigma)."""
    vs = tail.varsigma(theta)
    z = x / vs
    if tail.name != "student":
        return tail.kappa(theta, 0) * psi_chain(tail, r, n, z, theta)
    s = 0.0
    for b in range(min(n[1], 1) + 1):
        s += math.comb(n[1], b) * tail.kappa(theta, b) * psi_chain(tail, r, [n[0], n[1] - b], z, theta)
    return s


def exact(v):
    """the exact rational of the double v, so that the reference and the transcribed
    route are evaluated at literally the same point and sp.N can reach its precision."""
    return sp.Rational(float(v))


def check_h():
    """every component of H against SymPy's own derivative of H."""
    x = sp.Symbol("x", positive=True)
    pts = {"gaussian": [(2.3, [1.7]), (0.4, [0.3])],
           "student": [(2.3, [1.1, 3.0]), (0.4, [0.7, 1.5]), (9.0, [2.0, 7.5])],
           "logistic": [(2.3, [1.1]), (0.4, [0.7]), (9.0, [2.0])]}
    for tail in TAILS:
        syms = sp.symbols(" ".join(tail.pnames), positive=True)
        syms = (syms,) if not isinstance(syms, tuple) else syms
        vs = sp.prod([syms[j] ** tail.exps[j] for j in range(len(syms))])
        kap = (syms[1] + 1) / 2 if tail.name == "student" else sp.Integer(1)
        H = kap * tail.psi.subs(Z, x / vs)
        # H(0) = H'(0) = 0, which is what makes the join C^1
        assert sp.simplify(H.subs(x, 0)) == 0, tail.name
        assert sp.simplify(sp.diff(H, x).subs(x, 0)) == 0, tail.name
        worst, worst_at = 0.0, None
        nt = len(syms)
        for xv, tv in pts[tail.name]:
            at = {x: exact(xv)}
            for j in range(nt):
                at[syms[j]] = exact(tv[j])
            for r in range(ORDER + 1):
                for n1 in range(ORDER + 1 - r):
                    for n2 in range(ORDER + 1 - r - n1 if nt == 2 else 1):
                        n = [n1, n2][:nt]
                        d = H
                        if r:
                            d = sp.diff(d, x, r)
                        for j in range(nt):
                            if n[j]:
                                d = sp.diff(d, syms[j], n[j])
                        want = float(sp.N(d.subs(at), 30))
                        got = h_component(tail, r, n, xv, tv)
                        err = abs(got - want) / max(abs(want), 1.0)
                        if err > worst:
                            worst, worst_at = err, (xv, tuple(tv), r, tuple(n), want, got)
        print("  %-9s H components to total order %d: worst scaled error %.3e  at %s"
              % (tail.name, ORDER, worst, worst_at[:4]))
        assert worst < 1e-12, worst_at


# ---------------------------------------------------------------------------
# the tail constant T = int_0^inf exp(-H) and its partials
# ---------------------------------------------------------------------------

T_mp = {
    "gaussian": lambda t: __import__("mpmath").sqrt(__import__("mpmath").pi * t[0] / 2),
    "student": lambda t: (lambda mp: t[0] / 2 * mp.sqrt(mp.pi * t[1]) * mp.gamma(t[1] / 2)
                          / mp.gamma((t[1] + 1) / 2))(__import__("mpmath")),
    "logistic": lambda t: t[0],
}


def check_T():
    """T in closed form, and its partials in the tail's parameters, against SymPy."""
    import mpmath as mp
    mp.mp.dps = 40
    x = sp.Symbol("x", positive=True)
    forms = {}
    V, s, nu = sp.symbols("V s nu", positive=True)
    forms["gaussian"] = (sp.sqrt(sp.pi * V / 2), [V], [1.7])
    forms["student"] = (s / 2 * sp.sqrt(sp.pi * nu) * sp.gamma(nu / 2) / sp.gamma((nu + 1) / 2),
                        [s, nu], [1.1, 3.0])
    forms["logistic"] = (s, [s], [1.1])
    for tail in TAILS:
        T, syms, tv = forms[tail.name]
        # the closed T against the quadrature of exp(-H)
        tvm = [mp.mpf(str(v)) for v in tv]
        vsn = tail.varsigma(tvm)
        kap = tail.kappa(tvm, 0)
        psi = sp.lambdify(Z, tail.psi, "mpmath")
        num = mp.quad(lambda t: mp.e ** (-kap * psi(t / vsn)), [0, vsn, 10 * vsn, mp.inf])
        at = {syms[j]: exact(tv[j]) for j in range(len(syms))}
        cl = T_mp[tail.name](tvm)
        print("  %-9s T = %-34s  %s  rel to quadrature %.2e"
              % (tail.name, sp.srepr(T)[:0] or str(T), mp.nstr(cl, 12), abs(cl - num) / num))
        assert abs(cl - num) / num < 1e-25
        # and its partials, which the laws need to total order ORDER
        worst = 0.0
        for n1 in range(ORDER + 1):
            for n2 in range(ORDER + 1 - n1 if len(syms) == 2 else 1):
                d = T
                if n1:
                    d = sp.diff(d, syms[0], n1)
                if len(syms) == 2 and n2:
                    d = sp.diff(d, syms[1], n2)
                want = float(sp.N(d.subs(at), 30))
                got = t_component(tail.name, [n1, n2][: len(syms)], tv)
                worst = max(worst, abs(got - want) / max(abs(want), 1.0))
        print("            partials of T to total order %d: worst scaled error %.3e" % (ORDER, worst))
        assert worst < 1e-11


def sqrt_deriv(m, r):
    c = 1.0
    for t in range(m):
        c *= 0.5 - t
    return c * r ** (0.5 - m)


def inv_deriv(m, r):
    return (1.0 if m % 2 == 0 else -1.0) * math.factorial(m) / r ** (m + 1)


def t_component(name, n, theta):
    """d^n T, the planned C++ route: gaussian sqrt(pi/2) d^n V^(1/2); logistic d^n s;
    student d^n1 s times d^n2 G(nu), G = sqrt(pi nu) Gamma(nu/2)/(2 Gamma((nu+1)/2)),
    through log G, whose derivatives are elementary plus polygamma."""
    if name == "gaussian":
        return math.sqrt(math.pi / 2) * sqrt_deriv(n[0], theta[0])
    if name == "logistic":
        return theta[0] if n[0] == 0 else (1.0 if n[0] == 1 else 0.0)
    s, nu = theta
    if n[0] > 1:
        return 0.0
    sfac = s if n[0] == 0 else 1.0
    # log G = (1/2) log(pi nu) + lgamma(nu/2) - lgamma((nu+1)/2) - log 2
    K = n[1]
    g = [0.0] * (K + 1)
    g[0] = 0.5 * math.log(math.pi * nu) + math.lgamma(nu / 2) - math.lgamma((nu + 1) / 2) - math.log(2.0)
    import mpmath as mp
    for k in range(1, K + 1):
        g[k] = 0.5 * (-1.0) ** (k - 1) * math.factorial(k - 1) / nu**k
        g[k] += (0.5**k) * float(mp.polygamma(k - 1, nu / 2) - mp.polygamma(k - 1, (nu + 1) / 2))
    # G = exp(g) by the scalar exponential recursion
    G = [0.0] * (K + 1)
    G[0] = math.exp(g[0])
    for k in range(1, K + 1):
        G[k] = sum(math.comb(k - 1, b) * G[b] * g[k - b] for b in range(k))
    return sfac * G[K]


# ---------------------------------------------------------------------------
# the partial integral of exp(-H), for the cumulative functions
# ---------------------------------------------------------------------------

def check_upper():
    """int_x^inf exp(-H) dv in closed form, against quadrature."""
    import mpmath as mp
    mp.mp.dps = 40
    cases = {
        # gaussian: sqrt(pi V / 2) erfc(x / sqrt(2 V))
        "gaussian": (lambda x, t: mp.sqrt(mp.pi * t[0] / 2) * mp.erfc(x / mp.sqrt(2 * t[0])), [1.7]),
        # student: the Student t upper tail, s sqrt(nu) B(1/2, nu/2)/2 times the
        # regularized incomplete beta at nu/(nu + z^2)
        # varsigma/2 B(nu/2, 1/2) I_w(nu/2, 1/2), w = varsigma^2/(varsigma^2 + x^2),
        # from v = varsigma t and then w = 1/(1 + t^2)
        "student": (lambda x, t: (lambda vs: vs / 2 * mp.beta(t[1] / 2, mp.mpf(1) / 2)
                    * mp.betainc(t[1] / 2, mp.mpf(1) / 2, 0, vs**2 / (vs**2 + x**2),
                                 regularized=True))(t[0] * mp.sqrt(t[1])), [1.1, 3.0]),
        # logistic: s (1 - tanh(x/s))
        "logistic": (lambda x, t: t[0] * (1 - mp.tanh(x / t[0])), [1.1]),
    }
    for tail in TAILS:
        f, tv = cases[tail.name]
        tvm = [mp.mpf(str(v)) for v in tv]
        vsn = tail.varsigma(tvm)
        kap = tail.kappa(tvm, 0)
        psi = sp.lambdify(Z, tail.psi, "mpmath")
        worst = 0.0
        for xv in [0.0, 0.3, 1.0, 4.0, 20.0]:
            # the value at the left end is factored out, or quad loses the deep tail:
            # at x = 20 the gaussian integrand is e^(-117) and a plain quad was out by
            # 1.8e-6 (the same trap as the reference of 2026-11-18)
            x0 = mp.mpf(xv)
            h0 = kap * psi(x0 / vsn)
            g = lambda v: mp.e ** (-(kap * psi((x0 + v) / vsn) - h0))
            num = mp.e ** (-h0) * mp.quad(g, [0, vsn, 10 * vsn, 100 * vsn, mp.inf])
            cl = f(x0, tvm)
            worst = max(worst, abs(cl - num) / num)
        print("  %-9s int_x^inf exp(-H): worst relative error %.2e over x = 0 .. 20" % (tail.name, worst))
        assert worst < 1e-20


# ---------------------------------------------------------------------------
# the two pieces the laws add on top of the identities above
# ---------------------------------------------------------------------------

def idx4(i, m, n, o):
    """the triangular packing over four variables of properized_nonconvex.hpp."""
    t = i + m + n + o
    pos = math.comb(t + 3, 4)
    for j in range(i):
        pos += math.comb(t - j + 2, 2)
    for j in range(m):
        pos += math.comb(t - i - j + 1, 1)
    return pos + n


def tab4_size(K):
    return (K + 1) * (K + 2) * (K + 3) * (K + 4) // 24


def log4(F, K):
    """the table of log(F) over four variables, the recursion of log4 in the header."""
    U = [0.0] * tab4_size(K)
    f0 = F[idx4(0, 0, 0, 0)]
    U[idx4(0, 0, 0, 0)] = math.log(f0)
    for tot in range(1, K + 1):
        for i in range(tot + 1):
            for m in range(tot - i + 1):
                for n in range(tot - i - m + 1):
                    o = tot - i - m - n
                    gi, gm, gn, go, r = i, m, n, o, 3
                    if i > 0:
                        gi, r = i - 1, 0
                    elif m > 0:
                        gm, r = m - 1, 1
                    elif n > 0:
                        gn, r = n - 1, 2
                    else:
                        go = o - 1
                    s = F[idx4(i, m, n, o)]
                    for bi in range(gi + 1):
                        for bm in range(gm + 1):
                            for bn in range(gn + 1):
                                for bo in range(go + 1):
                                    if bi + bm + bn + bo == 0:
                                        continue
                                    di = gi - bi + (1 if r == 0 else 0)
                                    dm = gm - bm + (1 if r == 1 else 0)
                                    dn = gn - bn + (1 if r == 2 else 0)
                                    dd = go - bo + (1 if r == 3 else 0)
                                    s -= (math.comb(gi, bi) * math.comb(gm, bm) * math.comb(gn, bn)
                                          * math.comb(go, bo) * F[idx4(bi, bm, bn, bo)]
                                          * U[idx4(di, dm, dn, dd)])
                    U[idx4(i, m, n, o)] = s / f0
    return U


def check_log4():
    """Z = 2 [A(lambda, p) + B(lambda, p) T(s, nu)] for mcp with the Student tail: the
    table of log Z assembled from A, B and T and then log4, against SymPy's own
    derivatives of log Z. A and B come from SymPy here, so that only the assembly and
    the four-variable recursion are under test."""
    K = ORDER
    lam, gam, s, nu = sp.symbols("lam gam s nu", positive=True)
    x = lam * sp.sqrt(gam / 2)
    A = sp.sqrt(2 * gam) * sp.sqrt(sp.pi) / 2 * sp.exp(-x**2) * sp.erfi(x)  # sqrt(2 g) F(x)
    B = sp.exp(-gam * lam**2 / 2)
    T = s / 2 * sp.sqrt(sp.pi * nu) * sp.gamma(nu / 2) / sp.gamma((nu + 1) / 2)
    at = {lam: exact(1.3), gam: exact(3.0), s: exact(1.1), nu: exact(5.0)}
    Ad = [[float(sp.N(sp.diff(A, lam, i, gam, m).subs(at), 30)) for m in range(K + 1)]
          for i in range(K + 1)]
    Bd = [[float(sp.N(sp.diff(B, lam, i, gam, m).subs(at), 30)) for m in range(K + 1)]
          for i in range(K + 1)]
    Td = [[float(sp.N(sp.diff(T, s, n, nu, o).subs(at), 30)) for o in range(K + 1)]
          for n in range(K + 1)]
    Z = [0.0] * tab4_size(K)
    for i in range(K + 1):
        for m in range(K + 1 - i):
            for n in range(K + 1 - i - m):
                for o in range(K + 1 - i - m - n):
                    v = 2.0 * Bd[i][m] * Td[n][o]
                    if n == 0 and o == 0:
                        v += 2.0 * Ad[i][m]
                    Z[idx4(i, m, n, o)] = v
    U = log4(Z, K)
    logZ = sp.log(2 * (A + B * T))
    worst, worst_at = 0.0, None
    for i in range(K + 1):
        for m in range(K + 1 - i):
            for n in range(K + 1 - i - m):
                for o in range(K + 1 - i - m - n):
                    want = float(sp.N(sp.diff(logZ, lam, i, gam, m, s, n, nu, o).subs(at), 30))
                    got = U[idx4(i, m, n, o)]
                    err = abs(got - want) / max(abs(want), 1.0)
                    if err > worst:
                        worst, worst_at = err, (i, m, n, o, want, got)
    print("  log Z over (lambda, gamma, s, nu) by log4: worst scaled error %.3e  at %s"
          % (worst, worst_at[:4]))
    assert worst < 1e-12, worst_at


def check_q_tail_piece():
    """the tail piece of q, H(u - p lambda; theta), one component at a time: identity (1)
    over the components of H, against SymPy's own derivative of the whole piece."""
    u, lam, p = sp.symbols("u lam p", positive=True)
    for tail in TAILS:
        syms = sp.symbols(" ".join(tail.pnames), positive=True)
        syms = (syms,) if not isinstance(syms, tuple) else syms
        nt = len(syms)
        vs = sp.prod([syms[j] ** tail.exps[j] for j in range(nt)])
        kap = (syms[1] + 1) / 2 if tail.name == "student" else sp.Integer(1)
        H = kap * tail.psi.subs(Z, (u - p * lam) / vs)
        tv = {"gaussian": [1.21], "student": [1.1, 5.0], "logistic": [1.1]}[tail.name]
        at = {u: exact(9.0), lam: exact(1.3), p: exact(3.0)}
        for j in range(nt):
            at[syms[j]] = exact(tv[j])
        uv, lv, pv = 9.0, 1.3, 3.0
        worst, worst_at = 0.0, None
        for j in range(ORDER + 1):
            for i in range(ORDER + 1 - j):
                for m in range(ORDER + 1 - j - i):
                    for n1 in range(ORDER + 1 - j - i - m):
                        for n2 in range(ORDER + 1 - j - i - m - n1 if nt == 2 else 1):
                            n = [n1, n2][:nt]
                            d = H
                            if j:
                                d = sp.diff(d, u, j)
                            if i:
                                d = sp.diff(d, lam, i)
                            if m:
                                d = sp.diff(d, p, m)
                            for v in range(nt):
                                if n[v]:
                                    d = sp.diff(d, syms[v], n[v])
                            want = float(sp.N(d.subs(at), 30))
                            got = 0.0
                            for r in range(max(i, m), i + m + 1):
                                c = x_chain_coef(r, i, m, lv, pv)
                                if c != 0.0:
                                    got += c * h_component(tail, j + r, n, uv - pv * lv, tv)
                            err = abs(got - want) / max(abs(want), 1.0)
                            if err > worst:
                                worst, worst_at = err, (j, i, m, tuple(n), want, got)
        print("  %-9s tail piece of q to total order %d: worst scaled error %.3e  at %s"
              % (tail.name, ORDER, worst, worst_at[:4]))
        assert worst < 1e-12, worst_at


if __name__ == "__main__":
    print("Stirling identities")
    check_stirling()
    print("\nidentity (1), the chain through the knot")
    check_x_chain()
    print("\nidentity (2), the components of H")
    check_h()
    print("\nthe tail constant T")
    check_T()
    print("\nthe partial integral of exp(-H)")
    check_upper()
    print("\nthe tail piece of q, identities (1) and (2) composed")
    check_q_tail_piece()
    print("\nthe four-variable logarithm of Z")
    check_log4()
    print("\nall closed forms agree with SymPy")