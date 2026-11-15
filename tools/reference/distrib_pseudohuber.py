"""Reference values and checks of the closed forms for pseudohuber_distrib
(include/laplace/distrib/pseudohuber.hpp).

Usage (from the repository root):
    python -I tools/reference/distrib_pseudohuber.py forms
    python -I tools/reference/distrib_pseudohuber.py logpdf [workers]
    python -I tools/reference/distrib_pseudohuber.py small
    python -I tools/reference/distrib_pseudohuber.py rder <file>

The density, with z = (y - mu)/sigma, x = sqrt(nu) and K_n the modified Bessel function of the
second kind,
    f(y) = sqrt(R) / (2 sigma x K_1(x)) exp(-sqrt(nu + R z^2)),    R(nu) = x K_2(x) / K_1(x),
so that
    l = -sqrt(nu + R z^2) + (1/2) log R - (1/2) log nu - log K_1(x) - log 2 - log sigma.

File in tests/data/reference (every number in hexadecimal; (a, b, c) is the number of derivatives in
(mu, sigma, nu), K = a + b + c, K = 0 the value):
    distrib_pseudohuber.csv   y, K, a, b, c, hi, lo, mu, sigma, nu       d^K log f

"forms" checks, with SymPy and mpmath, what the header transcribes:
  * the exact identity R = 1 - x g'(x) and d^k R / dx^k = -(x g^(k+1) + k g^(k)), g = log K_1,
    against mpmath's derivatives of x K_2(x)/K_1(x), at three values of x (the two sides of the
    switch of the Bessel derivatives at x = 20 included);
  * the exact form H_(1/2)'' = q H_(-3/2) for H_p = (q + z^2)^p, and the recurrence
    (q + z^2) H_p^(m+1) = 2 (p - m) z H_p^(m) + m (2 p - m + 1) H_p^(m-1), against SymPy;
  * the table D[c][j] = d^c/dnu^c d^j/dz^j Lambda, Lambda = -sqrt(nu + R(nu) z^2), assembled as the
    header does (Faa di Bruno through rho = sqrt(R), q = nu/R, the Bell polynomials of sympy and
    Leibniz) for a test function R(nu), against SymPy's own derivative, every c + j <= 6;
  * the Gaussian and Laplace limits of the density (value of the normalizer at nu = 1e-8 and 1e8).

Method for the file. Nothing of the library's routes is used. l is the definition above. Its
derivatives in (mu, sigma, nu) are SymPy's symbolic derivatives of l with R and h = (1/2) log R -
(1/2) log nu - log K_1 left as undefined functions of nu, whose derivatives at the point are
mpmath.diff of their definitions (besselk), evaluated at 400 and 600 bits; a row is kept when the two
precisions agree to 2^-100 of the largest component of the order at that point, and is otherwise
dropped. The points have mu dyadic, sigma a power of two and z a short double, so that
(y - mu)/sigma is exactly z in double arithmetic and the library and the reference see the same z.
"""
import itertools
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpmath.mpf(2) ** -100

NU = [1e-6, 1e-3, 0.1, 1.0, 2.0, 100.0, 399.0, 401.0, 1e4, 1e6, 1e8]
COMBOS = [(0.5, 1.0), (-2.0, 2.0 ** -20), (1.0, 2.0 ** 20)]


def short(v, bits=20):
    if v == 0:
        return 0.0
    m, e = math.frexp(v)
    return math.ldexp(round(m * 2 ** bits) / 2 ** bits, e)


ZS = [short(v) for v in [0.0, 0.3, 1.0, -1.0, 4.0, 30.0, -30.0, 1000.0, 1e8]]
ZS_EXTREME = [short(v) for v in [0.0, 0.3, -4.0, 30.0, 1e8]]


def hexs(v):
    return float(v).hex()


def split(v):
    hi = float(v)
    return hi, float(v - mpmath.mpf(hi))


def comps(K):
    return [(K - b - c, b, c) for b in range(K + 1) for c in range(K - b + 1)]


# ------------------------------------------------------------------ the definitions (mpmath)

NSER = 8  # truncated power series of order 7: coefficients 0..7


def Rdef(nu):
    """R(nu) = x K_2(x) / K_1(x) from the definition (the Bessel functions of mpmath)"""
    x = mpmath.sqrt(nu)
    return x * mpmath.besselk(2, x) / mpmath.besselk(1, x)


def hdef(nu):
    """h(nu) = (1/2) log R - (1/2) log nu - log K_1(x)"""
    x = mpmath.sqrt(nu)
    return mpmath.log(Rdef(nu)) / 2 - mpmath.log(nu) / 2 - mpmath.log(mpmath.besselk(1, x))


# Truncated power series (lists of NSER mpf) in a variable t: the reference's own arithmetic, used to
# differentiate the definitions in nu without numerical differences (mpmath.diff at order six on K_n of a
# non-integer argument takes minutes per point).
def s_mul(a, b):
    r = [mpmath.mpf(0)] * NSER
    for i in range(NSER):
        for j in range(NSER - i):
            r[i + j] += a[i] * b[j]
    return r


def s_inv(a):
    r = [mpmath.mpf(0)] * NSER
    r[0] = 1 / a[0]
    for n in range(1, NSER):
        r[n] = -sum(a[k] * r[n - k] for k in range(1, n + 1)) / a[0]
    return r


def s_log(a):
    """log a = log a0 + int a'/a"""
    da = [(k + 1) * a[k + 1] for k in range(NSER - 1)] + [mpmath.mpf(0)]
    q = s_mul(da, s_inv(a))
    r = [mpmath.log(a[0])] + [q[k - 1] / k for k in range(1, NSER)]
    return r


def s_compose(c, sv):
    """sum_n c[n] sv^n for a series sv with sv[0] = 0"""
    r = [mpmath.mpf(0)] * NSER
    pw = [mpmath.mpf(1)] + [mpmath.mpf(0)] * (NSER - 1)
    for n in range(NSER):
        r = [r[k] + c[n] * pw[k] for k in range(NSER)]
        pw = s_mul(pw, sv)
    return r


def bessel_series(x0):
    """Taylor coefficients in s = x - x0 of K_0 and K_1, from the values at x0 and the system
    K_0' = -K_1, K_1' = -K_0 - K_1/x (which both functions satisfy: the recurrences of the modified
    Bessel functions of the second kind)"""
    c0 = [mpmath.mpf(0)] * NSER
    c1 = [mpmath.mpf(0)] * NSER
    c0[0] = mpmath.besselk(0, x0)
    c1[0] = mpmath.besselk(1, x0)
    e = [(-1) ** m / x0 ** (m + 1) for m in range(NSER)]  # 1/(x0 + s)
    for n in range(NSER - 1):
        c0[n + 1] = -c1[n] / (n + 1)
        c1[n + 1] = (-c0[n] - sum(c1[i] * e[n - i] for i in range(n + 1))) / (n + 1)
    return c0, c1


def nu_series(nu0):
    """R and h as truncated series in t = nu - nu0, and the series of x"""
    nu0 = mpmath.mpf(nu0)
    x0 = mpmath.sqrt(nu0)
    # x = sqrt(nu0 + t) = x0 (1 + t/nu0)^(1/2)
    xs = [x0 * mpmath.binomial(mpmath.mpf(1) / 2, n) / nu0 ** n for n in range(NSER)]
    sv = [mpmath.mpf(0)] + xs[1:]
    c0, c1 = bessel_series(x0)
    K0 = s_compose(c0, sv)
    K1 = s_compose(c1, sv)
    # R = x K_2 / K_1 with K_2 = K_0 + (2/x) K_1, so R = 2 + x K_0 / K_1
    Rs = s_mul(xs, s_mul(K0, s_inv(K1)))
    Rs[0] += 2
    # h = (1/2) log R - (1/2) log nu - log K_1
    lnu = [mpmath.log(nu0)] + [(-1) ** (n - 1) / (n * nu0 ** n) for n in range(1, NSER)]
    lR = s_log(Rs)
    lK1 = s_log(K1)
    hs = [lR[n] / 2 - lnu[n] / 2 - lK1[n] for n in range(NSER)]
    return Rs, hs


def nu_derivs(which, nu, n):
    """f, f', ..., f^(n) at nu for f = R (which = 0) or h (which = 1), from the series"""
    ser = nu_series(nu)[which]
    return [ser[k] * mpmath.factorial(k) for k in range(n + 1)]


# ------------------------------------------------------------------ the symbolic derivatives of l

MU, SG, NUS, Y = sp.symbols("mu sigma nu y", real=True)
RF = sp.Function("Rf")(NUS)
HF = sp.Function("hf")(NUS)
RSYM = sp.symbols("R0:7")
HSYM = sp.symbols("h0:7")


def build_exprs():
    """l and its derivatives for every (a, b, c), K <= 6, with the functions replaced by symbols"""
    z = (Y - MU) / SG
    l = -sp.sqrt(NUS + RF * z ** 2) + HF - sp.log(2) - sp.log(SG)
    cache = {(0, 0, 0): l}
    out = {}
    for K in range(0, 7):
        for (a, b, c) in comps(K):
            if (a, b, c) in cache:
                e = cache[(a, b, c)]
            else:
                if a > 0:
                    e = sp.diff(cache[(a - 1, b, c)], MU)
                elif b > 0:
                    e = sp.diff(cache[(a, b - 1, c)], SG)
                else:
                    e = sp.diff(cache[(a, b, c - 1)], NUS)
                cache[(a, b, c)] = e
            rep = {}
            for k in range(6, 0, -1):
                rep[sp.Derivative(RF, (NUS, k))] = RSYM[k]
                rep[sp.Derivative(HF, (NUS, k))] = HSYM[k]
            e2 = e.subs(rep).subs({RF: RSYM[0], HF: HSYM[0]})
            out[(a, b, c)] = sp.lambdify((MU, SG, NUS, Y) + RSYM + HSYM, e2, modules="mpmath")
    return out


_FUNCS = None


def eval_point(args):
    """all the components K = 0..6 at (mu, sigma, nu, y) at the given precision, as mpf"""
    global _FUNCS
    mu, sg, nu, y, prec = args
    if _FUNCS is None:
        _FUNCS = build_exprs()
    mpmath.mp.prec = prec
    nuv = mpmath.mpf(nu)
    Rd = nu_derivs(0, nuv, 6)
    hd = nu_derivs(1, nuv, 6)
    vals = {}
    for key, f in _FUNCS.items():
        vals[key] = f(mpmath.mpf(mu), mpmath.mpf(sg), nuv, mpmath.mpf(y), *Rd, *hd)
    return vals


def point_y(mu, sg, z0):
    y = mu + sg * z0
    assert (y - mu) / sg == z0, (mu, sg, z0)
    return y


def one_point(pt):
    mu, sg, nu, y = pt
    lo = eval_point((mu, sg, nu, y, 400))
    hi = eval_point((mu, sg, nu, y, 600))
    rows = []
    for K in range(0, 7):
        cs = comps(K)
        scale = max(abs(hi[c]) for c in cs)
        ok = all(abs(hi[c] - lo[c]) <= AGREE * scale for c in cs)
        if not ok:
            continue
        for (a, b, c) in cs:
            h, l = split(hi[(a, b, c)])
            rows.append((y, K, a, b, c, h, l, mu, sg, nu))
    return pt, rows


def make_logpdf(workers):
    pts = []
    for nu in NU:
        for ci, (mu, sg) in enumerate(COMBOS):
            for z0 in (ZS if ci == 0 else ZS_EXTREME):
                pts.append((mu, sg, nu, point_y(mu, sg, z0)))
    kept = dropped = 0
    path = OUT / "distrib_pseudohuber.csv"
    with open(path, "w") as f, Pool(workers) as pool:
        f.write("y,K,a,b,c,hi,lo,mu,sigma,nu\n")
        for pt, rows in pool.imap(one_point, pts, chunksize=1):
            if len(rows) == 0:
                dropped += 1
            for (y, K, a, b, c, h, l, mu, sg, nu) in rows:
                f.write(",".join([hexs(y), str(K), str(a), str(b), str(c), hexs(h), hexs(l), hexs(mu), hexs(sg), hexs(nu)]) + "\n")
                kept += 1
            f.flush()
    print("rows", kept, "points with no row", dropped, "->", path)


# ------------------------------------------------------------------ forms

def check_forms():
    mpmath.mp.dps = 60
    bad = 0

    # (1) the identity for R and its x-derivatives: the series of K_0, K_1 and R in s = x - x0 against the
    # formula, with g = log K_1; and the series route against mpmath's own derivative at low order
    for x0 in [mpmath.mpf("0.05"), mpmath.mpf("1.3"), mpmath.mpf("19.5"), mpmath.mpf("20.5"), mpmath.mpf("300")]:
        c0, c1 = bessel_series(x0)
        gs = s_log(c1)
        Rs = s_mul([x0, mpmath.mpf(1)] + [mpmath.mpf(0)] * (NSER - 2), s_mul(c0, s_inv(c1)))
        Rs[0] += 2
        gd = [None] + [gs[k] * mpmath.factorial(k) for k in range(1, NSER)]
        for k in range(0, 7):
            exact = Rs[k] * mpmath.factorial(k)
            form = 1 - x0 * gd[1] if k == 0 else -(x0 * gd[k + 1] + k * gd[k])
            rel = abs(form - exact) / max(abs(exact), mpmath.mpf(10) ** -30)
            if rel > mpmath.mpf(10) ** -20:
                bad += 1
                print("R identity", float(x0), k, float(rel))
        for k in (1, 2):
            ex = mpmath.diff(lambda x: mpmath.log(mpmath.besselk(1, x)), x0, k)
            if abs(ex - gd[k]) > mpmath.mpf(10) ** -30 * abs(ex):
                bad += 1
                print("series of g against mpmath.diff", float(x0), k)
    print("R identity checked")

    # (2) the derivatives of H_p = (q + z^2)^p, at exact rational points, evaluated to 50 digits
    z, q = sp.symbols("z q", positive=True)
    A = q + z ** 2
    pts = [{q: sp.Rational(3, 7), z: sp.Rational(5, 3)}, {q: sp.Rational(1, 1000), z: sp.Rational(400)}, {q: sp.Rational(9), z: sp.Rational(0)}]
    for p in [sp.Rational(1, 2), sp.Rational(-1, 2), sp.Rational(-3, 2), sp.Rational(-5, 2), sp.Rational(-9, 2)]:
        H = [A ** p]
        for m in range(1, 8):
            H.append(sp.diff(H[-1], z))
        for m in range(0, 6):
            lower = H[m - 1] if m > 0 else 0
            rhs = 2 * (p - m) * z * H[m] + m * (2 * p - m + 1) * lower
            for pt in pts:
                lhs_v = sp.N((A * H[m + 1]).subs(pt), 50)
                rhs_v = sp.N(rhs.subs(pt), 50)
                if abs(lhs_v - rhs_v) > sp.Float(10) ** -35 * max(abs(lhs_v), abs(rhs_v), 1):
                    bad += 1
                    print("H recurrence", p, m, pt)
    Hh = A ** sp.Rational(1, 2)
    Hm = A ** sp.Rational(-3, 2)
    for m in range(2, 8):
        for pt in pts:
            l_v = sp.N(sp.diff(Hh, z, m).subs(pt), 50)
            r_v = sp.N((q * sp.diff(Hm, z, m - 2)).subs(pt), 50)
            if abs(l_v - r_v) > sp.Float(10) ** -35 * max(abs(l_v), abs(r_v), sp.Float(10) ** -30):
                bad += 1
                print("H(1/2)^(m) = q H(-3/2)^(m-2) fails at", m, pt)
    print("H recurrence and the identity H_(1/2)^(m) = q H_(-3/2)^(m-2) checked")

    # (3) the table D, assembled as the header does, against SymPy's derivative, for a test R(nu)
    nu = sp.symbols("nu", positive=True)
    zz = sp.symbols("zz", real=True)
    Rtest = sp.sqrt(nu) + sp.Rational(3, 2) + 1 / (nu + 2) + 1 / (nu + sp.Rational(1, 3))
    Lam = -sp.sqrt(nu + Rtest * zz ** 2)
    for (zv, nv) in [(sp.Rational(7, 10), sp.Rational(9, 4)), (sp.Rational(29, 10), sp.Rational(4)),
                     (sp.Rational(51, 10), sp.Rational(1, 9)), (sp.Rational(0), sp.Rational(1, 4)),
                     (sp.Rational(400), sp.Rational(25))]:
        K = 6
        Rn = [sp.N(sp.diff(Rtest, nu, k).subs(nu, nv), 60) for k in range(K + 1)]
        R = Rn[0]
        xs = sp.symbols("x1:8")

        def faa(f_of_R, order):
            """d^order/dnu^order f(R(nu)) given f's derivatives as a function of k"""
            if order == 0:
                return f_of_R(0)
            return sum(f_of_R(k) * sp.bell(order, k, Rn[1:order - k + 2]) for k in range(1, order + 1))

        def falling_half(k):
            r = sp.Integer(1)
            for i in range(k):
                r *= sp.Rational(1, 2) - i
            return r

        rho = [faa(lambda k: falling_half(k) * R ** (sp.Rational(1, 2) - k) if k else sp.sqrt(R), c) for c in range(K + 1)]
        u = [faa(lambda k: (-1) ** k * sp.factorial(k) * R ** (-1 - k), c) for c in range(K + 1)]
        qd = [nv * u[0]] + [nv * u[k] + k * u[k - 1] for k in range(1, K + 1)]
        q0 = qd[0]
        a2 = q0 + zv ** 2
        Hs = {}
        for i in range(1, K + 1):
            p = sp.Rational(1, 2) - i
            seq = [a2 ** p]
            for m in range(0, K - i):
                lower = seq[m - 1] if m > 0 else 0
                seq.append((2 * (p - m) * zv * seq[m] + m * (2 * p - m + 1) * lower) / a2)
            Hs[i] = seq
        Gt = {}
        for j in range(0, K + 1):
            if j == 0:
                Gt[(0, 0)] = sp.sqrt(a2)
            elif j == 1:
                Gt[(0, 1)] = zv / sp.sqrt(a2)
            else:
                Gt[(0, j)] = q0 * Hs[2][j - 2]
        for i in range(1, K + 1):
            for j in range(0, K - i + 1):
                Gt[(i, j)] = falling_half(i) * Hs[i][j]
        worst = 0
        for c in range(K + 1):
            for j in range(0, K - c + 1):
                if c == 0 and j == 0:
                    continue
                gam = lambda a: Gt[(0, j)] if a == 0 else sum(Gt[(i, j)] * sp.bell(a, i, qd[1:a - i + 2]) for i in range(1, a + 1))
                mine = -sum(sp.binomial(c, a) * rho[c - a] * gam(a) for a in range(c + 1))
                ex = Lam
                if j:
                    ex = sp.diff(ex, zz, j)
                if c:
                    ex = sp.diff(ex, nu, c)
                ex = sp.N(ex.subs({zz: zv, nu: nv}), 50)
                err = abs(mine - ex) / max(abs(ex), sp.Float(10) ** -40)
                worst = max(worst, err)
                if err > sp.Float(10) ** -25:
                    bad += 1
                    print("table D", c, j, zv, nv, float(err))
        print("table D at z =", zv, "nu =", nv, "worst relative", float(worst))

    # (4) the limits of the normalizer: the density integrates to one and has variance sigma^2
    for nuv in [mpmath.mpf("1e-8"), mpmath.mpf("1e-3"), mpmath.mpf(1), mpmath.mpf(50), mpmath.mpf("1e8")]:
        R = Rdef(nuv)
        dens = lambda t: mpmath.exp(-mpmath.sqrt(nuv + R * t ** 2) + mpmath.log(R) / 2 - mpmath.log(nuv) / 2 - mpmath.log(mpmath.besselk(1, mpmath.sqrt(nuv))) - mpmath.log(2))
        width = 1 / mpmath.sqrt(R) * 40 + 40 / mpmath.sqrt(R) if nuv < 1 else 12 + 40 / mpmath.sqrt(R)
        pts = [-width, -5, 0, 5, width]
        total = mpmath.quad(dens, [-width, -mpmath.mpf(5), 0, 5, width])
        var = mpmath.quad(lambda t: t ** 2 * dens(t), [-width, -mpmath.mpf(5), 0, 5, width])
        e1, e2 = abs(total - 1), abs(var - 1)
        print("nu", float(nuv), "mass", mpmath.nstr(total, 15), "variance", mpmath.nstr(var, 15))
        if e1 > mpmath.mpf(10) ** -10 or e2 > mpmath.mpf(10) ** -10:
            bad += 1
    print("FAILED" if bad else "all forms agree", bad)
    return bad



# ------------------------------------------------------------------ the small-nu series for R and h

SMALL_TERMS = 14  # terms of the series in nu (the header's small_nu_terms)


def small_coefs(M=SMALL_TERMS, exact=False):
    """Coefficients (in nu) of P0 = I_0(x), Hs = sum H_k u^k/(k!)^2, J1 = nu I_1(x)/(x/2) / ..., W1: the header's tables.
    With u = nu/4, x = sqrt(nu), L = log(x/2) and gamma Euler's constant:
        K_0 = -(L + gamma) P0 + Hs,                   P0 = sum u^k/(k!)^2,  Hs = sum_(k>=1) H_k u^k/(k!)^2,
        x K_1 = B = 1 + (1/2) L J1 - W1,              J1 = sum_k nu^(k+1)/(4^k k! (k+1)!),
                                                      W1 = sum_k (H_k + H_(k+1) - 2 gamma) nu^(k+1)/(4^(k+1) k! (k+1)!),
    (DLMF 10.31.1 for n = 0 and n = 1, H_k the harmonic numbers, psi(k + 1) = H_k - gamma), and
        R = 2 + nu K_0 / B"""
    one = sp.Integer(1) if exact else mpmath.mpf(1)
    g = sp.EulerGamma if exact else mpmath.euler
    H = [0 * one]
    for k in range(1, M + 2):
        H.append(H[-1] + one / k)
    fact = [sp.factorial(k) if exact else mpmath.factorial(k) for k in range(M + 2)]
    p0 = [one / (4 ** k * fact[k] ** 2) for k in range(M)]
    hs = [H[k] * p0[k] for k in range(M)]
    j1 = [0 * one] + [one / (4 ** k * fact[k] * fact[k + 1]) for k in range(M - 1)]
    w1 = [0 * one] + [(H[k] + H[k + 1] - 2 * g) / (4 ** (k + 1) * fact[k] * fact[k + 1]) for k in range(M - 1)]
    return p0, hs, j1, w1


def small_poly(c, nu, i):
    """d^i/dnu^i of sum_k c[k] nu^k"""
    r = mpmath.mpf(0)
    for k in range(i, len(c)):
        f = mpmath.mpf(1)
        for m in range(i):
            f *= (k - m)
        r += c[k] * f * nu ** (k - i)
    return r


def small_derivs(nu, K=6):
    """R^(k) and h^(k), k = 0..K, from the series by the formulas of the header (Leibniz, Faa di Bruno)"""
    nu = mpmath.mpf(nu)
    p0, hs, j1, w1 = small_coefs()
    gam = mpmath.euler
    L = [mpmath.log(nu) / 2 - mpmath.log(2)] + [mpmath.mpf(1) / 2 * (-1) ** (j - 1) * mpmath.factorial(j - 1) / nu ** j for j in range(1, K + 1)]
    A, B = [], []
    for i in range(K + 1):
        a = -(L[0] + gam) * small_poly(p0, nu, i) + small_poly(hs, nu, i)
        b = (1 if i == 0 else 0) - small_poly(w1, nu, i)
        for j in range(1, i + 1):
            a -= mpmath.binomial(i, j) * L[j] * small_poly(p0, nu, i - j)
        for j in range(0, i + 1):
            b += mpmath.mpf(1) / 2 * mpmath.binomial(i, j) * L[j] * small_poly(j1, nu, i - j)
        A.append(a)
        B.append(b)
    # W = 1/B: W^(m) = sum_j (-1)^j j! B^(-1-j) Bell_(m,j)(B', ...)
    W = [1 / B[0]]
    for m in range(1, K + 1):
        W.append(sum((-1) ** j * mpmath.factorial(j) / B[0] ** (j + 1) * bell_num(m, j, B[1:]) for j in range(1, m + 1)))
    Q = [sum(mpmath.binomial(k, i) * A[i] * W[k - i] for i in range(k + 1)) for k in range(K + 1)]
    R = [2 + nu * Q[0]] + [nu * Q[k] + k * Q[k - 1] for k in range(1, K + 1)]
    # h = (1/2) log R - log B
    def logd(F, k):
        if k == 0:
            return mpmath.log(F[0])
        return sum((-1) ** (j - 1) * mpmath.factorial(j - 1) / F[0] ** j * bell_num(k, j, F[1:]) for j in range(1, k + 1))
    h = [logd(R, k) / 2 - logd(B, k) for k in range(K + 1)]
    return R, h


def bell_num(n, k, g):
    """partial Bell polynomial B_(n,k)(g_1, ...) with g a list g[0] = g_1"""
    if n == 0 and k == 0:
        return mpmath.mpf(1)
    if n == 0 or k == 0:
        return mpmath.mpf(0)
    return sum(mpmath.binomial(n - 1, i - 1) * g[i - 1] * bell_num(n - i, k - 1, g) for i in range(1, n - k + 2))


def check_small():
    bad = 0
    # (1) the coefficients against SymPy's series of K_0 and x K_1 (exact), through nu^6
    x = sp.symbols("x", positive=True)
    Ls = sp.symbols("L")
    p0, hs, j1, w1 = small_coefs(8, exact=True)
    nu = x ** 2
    K0 = -(Ls + sp.EulerGamma) * sum(p0[k] * nu ** k for k in range(8)) + sum(hs[k] * nu ** k for k in range(8))
    B = 1 + sp.Rational(1, 2) * Ls * sum(j1[k] * nu ** k for k in range(8)) - sum(w1[k] * nu ** k for k in range(8))
    for expr, ref in ((K0, sp.besselk(0, x)), (B, x * sp.besselk(1, x))):
        ser = sp.series(ref, x, 0, 14).removeO()
        mine = sp.expand(expr.subs(Ls, sp.log(x / 2)))
        d = sp.simplify(sp.expand(sp.expand_log(sp.expand(ser - mine), force=True)))
        # drop the terms beyond x^13
        d = sum(t for t in sp.Add.make_args(sp.expand(d)) if sp.degree(t.subs(sp.log(x), 1), x) < 14) if d != 0 else 0
        if sp.simplify(d) != 0:
            bad += 1
            print("series differs", d)
    print("series coefficients against SymPy checked")
    # (2) R, h and their derivatives against the definition (series jets of the Bessel ODE system)
    mpmath.mp.prec = 400
    worst = 0
    for nu_v in [1e-6, 1e-3, 0.05, 0.3, 1.0]:  # SMALL_TERMS = 14 holds to 1e-22 relative here (1e-15 at nu = 3.5)
        R, h = small_derivs(nu_v)
        Rr = nu_derivs(0, nu_v, 6)
        hr = nu_derivs(1, nu_v, 6)
        for k in range(7):
            for a, b, name in ((R[k], Rr[k], "R"), (h[k], hr[k], "h")):
                rel = abs(a - b) / max(abs(b), mpmath.mpf(10) ** -30)
                worst = max(worst, rel)
                if rel > mpmath.mpf(10) ** -20:
                    bad += 1
                    print("small series", name, k, nu_v, mpmath.nstr(rel, 5))
        print("nu", nu_v, "worst relative so far", mpmath.nstr(worst, 5))
    print("FAILED" if bad else "small-nu series agree", bad)
    return bad


def make_rder(path, nus):
    """R^(k) and h^(k), k = 0..6, at the given nu from the definition (series jets), hex hi/lo, one nu per line"""
    mpmath.mp.prec = 400
    with open(path, "w") as f:
        f.write("nu," + ",".join("R%d_hi,R%d_lo" % (k, k) for k in range(7)) + "," + ",".join("h%d_hi,h%d_lo" % (k, k) for k in range(7)) + "\n")
        for nu_v in nus:
            Rr = nu_derivs(0, nu_v, 6)
            hr = nu_derivs(1, nu_v, 6)
            cells = [hexs(nu_v)]
            for v in Rr + hr:
                hi, lo = split(v)
                cells += [hexs(hi), hexs(lo)]
            f.write(",".join(cells) + "\n")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forms"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2
    if cmd == "forms":
        sys.exit(1 if check_forms() else 0)
    elif cmd == "logpdf":
        make_logpdf(workers)
    elif cmd == "small":
        sys.exit(1 if check_small() else 0)
    elif cmd == "rder":
        make_rder(sys.argv[2], [10 ** (-6 + 0.25 * i) for i in range(0, 32)])
    else:
        print(__doc__)
