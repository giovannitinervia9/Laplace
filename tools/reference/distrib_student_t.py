"""Reference values and checks of the closed forms for student_t1_distrib and student_t2_distrib
(include/laplace/distrib/student_t1.hpp, student_t2.hpp).

Usage (from the repository root):
    python -I tools/reference/distrib_student_t.py forms
    python -I tools/reference/distrib_student_t.py logpdf [workers]
    python -I tools/reference/distrib_student_t.py info [workers]
    python -I tools/reference/distrib_student_t.py logcdf [workers]

Files in tests/data/reference (every number in hexadecimal; fam is 1 for student_t1 and 2 for
student_t2; (a, b, c) is the number of derivatives in (mu, sigma, nu), K = a + b + c):
    distrib_student_t_logpdf.csv   fam, y, K, a, b, c, hi, lo, mu, sigma, nu       d^K log f
    distrib_student_t_info.csv     fam, K, pair, b, c, hi, lo, mu, sigma, nu       d^K I_pair / dsigma^b dnu^c
    distrib_student_t_logcdf.csv   fam, upper, y, K, a, b, c, hi, lo, mu, sigma, nu  d^K log F (upper 0) or
                                                                                    d^K log(1 - F) (upper 1)
pair is the position in sym_tensor<3, 2> (0 mumu, 1 musigma, 2 munu, 3 sigmasigma, 4 sigmanu, 5 nunu);
the entries that vanish (pairs 1 and 2, and every derivative in mu) are not in the file.

"forms" checks, with SymPy, the closed forms that the headers transcribe:
  * the table D[c][j] = d^c/dnu^c d^j/dz^j Lambda, Lambda = -(nu + 1)/2 log(1 + z^2/nu), for every
    c + j <= 6 (the Leibniz form over P(z) = z (1 - z^2), and the pure nu-derivatives in r = z^2/(nu + z^2)),
    each against SymPy's own derivative at three points in exact rational arithmetic;
  * the information of student_t2 as the Jacobian transform J' I1 J of student_t1's through
    (s, nu) = (sigma sqrt((nu - 2)/nu), nu), against the simplified closed forms of the header;
  * the information entries of both families against E[l_a l_b] by mpmath quadrature (at 30 digits,
    two parameter points each), and the series in 1/nu of I_nunu (coefficients printed, the series
    against the exact value at the cut nu = 16);
  * the recurrence for the derivatives of r(nu) = sqrt((nu - 2)/nu).

Method for the files. Nothing of the library's routes is used.
  logpdf:  l is the definition (log Gamma, logs); its derivatives are SymPy's symbolic derivatives of
           that expression, evaluated by mpmath at 400 and 600 bits, a row kept when the two agree to
           2^-100 relative (otherwise once more at 800 and 1000 bits, then dropped). For student_t2 the
           scale s = sigma sqrt((nu - 2)/nu) is substituted before differentiating.
  info:    the closed forms above (verified by "forms" against the definition) and mpmath.diff of them.
  logcdf:  log F = log T_nu(z), T_nu(-|z|) = I_x(nu/2, 1/2)/2, x = nu/(nu + z^2) (the side of the
           incomplete beta that does not cancel), log T = log1p(-T(-z)) for z > 0. The table
           d^i/dz^i d^j/dnu^j log T_nu(z), i + j <= 6, is mpmath.diff of that definition at 280 and 360
           bits (agree to 2^-100, else 440 and 560, else dropped), at the exact z of the row; the
           derivatives in (mu, sigma, nu) come from composing the bivariate Taylor polynomial of that
           table with the series of z(mu + dmu, sigma + dsigma, nu + dnu) = (y - mu - dmu) /
           ((sigma + dsigma) rho(nu + dnu)), rho = 1 (student_t1) or sqrt((nu - 2)/nu) (student_t2),
           in exact mpmath arithmetic. log(1 - F) is the same with the table at -z. The composition is
           checked against mpmath.diff of the three-variable definition on a point at orders up to 3.
  The points of student_t1 have mu dyadic and sigma a power of two, and z a short double, so that
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

NU1 = [0.01, 0.1, 0.5, 1.0, 2.0, 3.5, 10.0, 100.0, 1e4, 1e8, 1e12]
NU2 = [2.000001, 2.01, 2.5, 3.0, 5.0, 10.0, 100.0, 1e4, 1e8, 1e12]
COMBOS = [(0.5, 1.0), (-2.0, 2.0 ** -20), (1.0, 2.0 ** 20)]


def short(v, bits=20):
    """v rounded to a double with `bits` significant bits"""
    if v == 0:
        return 0.0
    m, e = math.frexp(v)
    return math.ldexp(round(m * 2 ** bits) / 2 ** bits, e)


ZS = [short(v) for v in [0.0, 0.3, 1.0, -1.0, 4.0, 30.0, -30.0, 1000.0, 1e8]]


def hexs(v):
    return float(v).hex()


def split(v):
    hi = float(v)
    return hi, float(v - mpmath.mpf(hi))


def sorted_comps(K):
    return [(K - b - c, b, c) for b in range(K + 1) for c in range(K - b + 1)]


def point_y(mu, sg, z0):
    y = mu + sg * z0
    assert (y - mu) / sg == z0, (mu, sg, z0)
    return y


# ------------------------------------------------------------------ forms

def check_forms():
    z, nu = sp.symbols("z nu", positive=True)
    A = nu + z ** 2
    Lam = -(nu + 1) / 2 * sp.log(1 + z ** 2 / nu)
    P = z * (1 - z ** 2)
    pts = [(sp.Rational(7, 10), sp.Rational(33, 10)), (sp.Rational(29, 10), sp.Rational(4, 10)), (sp.Rational(51, 10), sp.Rational(17))]
    bad = 0
    for c in range(0, 7):
        for j in range(0, 7 - c):
            if c == 0 and j == 0:
                continue
            ex = sp.diff(Lam, nu, c, z, j) if (c and j) else (sp.diff(Lam, nu, c) if c else sp.diff(Lam, z, j))
            if j >= 1:
                g = sp.diff(P * A ** (-(c + 1)), z, j - 1) if j > 1 else P * A ** (-(c + 1))
                f = -(-1) ** c * sp.factorial(c) * g
                if c == 0:
                    f -= {1: z, 2: 1}.get(j, 0)
            else:
                r = z ** 2 / A
                if c == 1:
                    f = sp.Rational(1, 2) * (sp.log(1 - r) + r) + sp.Rational(1, 2) * (1 / nu - 1 / A)
                else:
                    m = c - 1
                    B = (1 + m * r) * (1 - r) ** m - 1
                    f = (sp.Rational(1, 2) * (-1) ** m * sp.factorial(m - 1) * nu ** (-m) * B
                         + sp.Rational(1, 2) * (-1) ** m * sp.factorial(m) * (nu ** (-m - 1) - A ** (-m - 1)))
            for zz, nn in pts:
                a, b = sp.N(ex.subs({z: zz, nu: nn}), 30), sp.N(f.subs({z: zz, nu: nn}), 30)
                if abs(a - b) > 1e-20 * (1 + abs(a)):
                    bad += 1
                    print("BAD table", c, j, zz, nn, a, b)
    print("Lambda table: %d mismatches" % bad)
    # the polynomial form of B_m(r) against the product
    rr = sp.symbols("rr")
    for m in range(1, 6):
        poly = rr ** 2 * sum((-1) ** k * (sp.binomial(m, k) - m * sp.binomial(m, k - 1)) * rr ** (k - 2) for k in range(2, m + 2))
        assert sp.expand(poly - ((1 + m * rr) * (1 - rr) ** m - 1)) == 0
    print("B_m polynomial: ok")

    # the information of student_t2 as J' I1 J
    sg, n = sp.symbols("sigma nu", positive=True)
    s = sg * sp.sqrt((n - 2) / n)
    a1 = (n + 1) / (n + 3)
    b1 = 2 * n / (n + 3)
    e1 = -2 / ((n + 1) * (n + 3))
    f1 = sp.Function("f1")(n)
    I1 = sp.Matrix([[a1 / s ** 2, 0, 0], [0, b1 / s ** 2, e1 / s], [0, e1 / s, f1]])
    J = sp.Matrix([[1, 0, 0], [0, sp.diff(s, sg), sp.diff(s, n)], [0, 0, 1]])
    I2 = sp.simplify(J.T * I1 * J)
    mine = {(0, 0): n * (n + 1) / ((n + 3) * (n - 2) * sg ** 2), (1, 1): 2 * n / ((n + 3) * sg ** 2),
            (1, 2): 6 / ((n + 3) * (n - 2) * (n + 1) * sg),
            (2, 2): f1 + 2 / ((n + 3) * n * (n - 2) ** 2) - 4 / (n * (n - 2) * (n + 1) * (n + 3))}
    ok = all(sp.simplify(I2[k] - v) == 0 for k, v in mine.items()) and I2[0, 1] == 0 and I2[0, 2] == 0
    print("student_t2 information as J' I1 J:", "ok" if ok else "MISMATCH")

    # the information entries against E[l_a l_b] by quadrature
    mpmath.mp.dps = 30
    for fam, nu0 in [(1, mpmath.mpf("3.7")), (1, mpmath.mpf("0.6")), (2, mpmath.mpf("4.2")), (2, mpmath.mpf("2.3"))]:
        sg0 = mpmath.mpf("1.3")
        mu0 = mpmath.mpf("0.4")
        sc = lambda sg_, nu_: sg_ * mpmath.sqrt((nu_ - 2) / nu_) if fam == 2 else sg_

        def logf(mu_, sg_, nu_, y):
            zz = (y - mu_) / sc(sg_, nu_)
            return (mpmath.loggamma((nu_ + 1) / 2) - mpmath.loggamma(nu_ / 2) - mpmath.log(nu_ * mpmath.pi) / 2
                    - mpmath.log(sc(sg_, nu_)) - (nu_ + 1) / 2 * mpmath.log(1 + zz ** 2 / nu_))

        def score(y, i):
            o = [0, 0, 0]
            o[i] = 1
            return density_fn(fam, tuple(o))(mu0, sg0, nu0, y)

        closed = info_values(fam, sg0, nu0)
        for (i, jx), pos in [((0, 0), 0), ((1, 1), 3), ((1, 2), 4), ((2, 2), 5), ((0, 1), 1), ((0, 2), 2)]:
            val = mpmath.quad(lambda y: mpmath.exp(logf(mu0, sg0, nu0, y)) * score(y, i) * score(y, jx),
                              [-mpmath.inf, mu0 - 6 * sg0, mu0, mu0 + 6 * sg0, mpmath.inf])
            ref = closed[pos]
            err = abs(val - ref) / (abs(ref) if ref != 0 else mpmath.mpf(1))
            print("  fam %d nu %s I[%d,%d]: closed %s quadrature %s rel %s" % (fam, nu0, i, jx, mpmath.nstr(ref, 15), mpmath.nstr(val, 15), mpmath.nstr(err, 3)))
            assert err < 1e-14, "information mismatch"
    print("information against quadrature: ok")

    # the series of I_nunu in 1/nu
    N = 44
    c = {}
    for k in range(1, N):
        d = (-1) ** k * (sp.bernoulli(k, 0) - sp.bernoulli(k, sp.Rational(1, 2))) * 2 ** (k + 1)
        c[k + 1] = c.get(k + 1, 0) + d / 4
    for k in range(1, N + 1):
        t = sp.Rational(0)
        if k == 1:
            t += sp.Rational(5, 3)
        t += -2 * (-1) ** (k - 1) + sp.Rational(1, 3) * (-3) ** (k - 1)
        c[k] = c.get(k, 0) - t / 2
    print("series coefficients c_n, n = 1..8:", [str(c[k]) for k in range(1, 9)])
    mpmath.mp.dps = 40
    for nu0 in (16, 20, 50):
        ex = (mpmath.psi(1, mpmath.mpf(nu0) / 2) - mpmath.psi(1, mpmath.mpf(nu0 + 1) / 2)) / 4 - mpmath.mpf(nu0 + 5) / (2 * nu0 * (nu0 + 1) * (nu0 + 3))
        s_ = sum(mpmath.mpf(int(c[k].p)) / int(c[k].q) * mpmath.mpf(nu0) ** (-k) for k in range(4, 39))
        print("  series at nu = %d: relative error %s" % (nu0, mpmath.nstr(abs(s_ - ex) / ex, 3)))
        assert abs(s_ - ex) / ex < 1e-16
    print("table for the header (n = 4..38):")
    print(", ".join(repr(float(c[k])) for k in range(4, 39)))

    # r(nu) recurrence
    mpmath.mp.dps = 40
    nu0 = mpmath.mpf("2.7")
    rr_ = [mpmath.sqrt((nu0 - 2) / nu0)]
    for q in range(1, 7):
        acc = -2 * (-1) ** q * mpmath.factorial(q) * nu0 ** (-q - 1)
        for i in range(1, q):
            acc -= mpmath.binomial(q, i) * rr_[i] * rr_[q - i]
        rr_.append(acc / (2 * rr_[0]))
    for q in range(1, 7):
        d = mpmath.diff(lambda x: mpmath.sqrt((x - 2) / x), nu0, q)
        assert abs(d - rr_[q]) < mpmath.mpf(10) ** -25 * abs(d), q
    print("r(nu) recurrence: ok")


# ------------------------------------------------------------------ the definitions

def info_values(fam, sg, nu):
    """the six entries (mumu, musigma, munu, sigmasigma, sigmanu, nunu) of the information, by the closed forms"""
    inn = (mpmath.psi(1, nu / 2) - mpmath.psi(1, (nu + 1) / 2)) / 4 - (nu + 5) / (2 * nu * (nu + 1) * (nu + 3))
    if fam == 1:
        return [(nu + 1) / ((nu + 3) * sg ** 2), 0, 0, 2 * nu / ((nu + 3) * sg ** 2), -2 / (sg * (nu + 1) * (nu + 3)), inn]
    return [nu * (nu + 1) / ((nu + 3) * (nu - 2) * sg ** 2), 0, 0, 2 * nu / ((nu + 3) * sg ** 2),
            6 / (sg * (nu + 3) * (nu - 2) * (nu + 1)),
            inn + 2 / ((nu + 3) * nu * (nu - 2) ** 2) - 4 / (nu * (nu - 2) * (nu + 1) * (nu + 3))]


_SYM = {}


def density_fn(fam, comp):
    """lambdified symbolic derivative of l of multi-index comp = (a, b, c) in (mu, sigma, nu)"""
    key = (fam, comp)
    if key not in _SYM:
        mu, sg, nu, y = sp.symbols("mu sigma nu y")
        s = sg * sp.sqrt((nu - 2) / nu) if fam == 2 else sg
        z = (y - mu) / s
        d = (sp.loggamma((nu + 1) / 2) - sp.loggamma(nu / 2) - sp.log(nu * sp.pi) / 2 - sp.log(s)
             - (nu + 1) / 2 * sp.log(1 + z ** 2 / nu))
        for var, n_ in ((mu, comp[0]), (sg, comp[1]), (nu, comp[2])):
            if n_:
                d = sp.diff(d, var, n_)
        _SYM[key] = sp.lambdify((mu, sg, nu, y), d, modules="mpmath")
    return _SYM[key]


def logpdf_task(args):
    fam, y, mu, sg, nu = args
    out, dropped = [], 0
    for K in range(1, 7):
        for comp in sorted_comps(K):
            vals = None
            for p1, p2 in ((400, 600), (800, 1000)):
                with mpmath.workprec(p1):
                    a = density_fn(fam, comp)(mpmath.mpf(mu), mpmath.mpf(sg), mpmath.mpf(nu), mpmath.mpf(y))
                with mpmath.workprec(p2):
                    b = density_fn(fam, comp)(mpmath.mpf(mu), mpmath.mpf(sg), mpmath.mpf(nu), mpmath.mpf(y))
                if a == b or abs(a - b) <= AGREE * abs(b):
                    vals = b
                    break
            if vals is None or vals == 0 or not math.isfinite(float(vals)) or abs(float(vals)) < 2.2250738585072014e-308:
                dropped += 1
                continue
            hi, lo = split(vals)
            out.append("%d,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s\n" % (fam, hexs(y), K, comp[0], comp[1], comp[2], hexs(hi), hexs(lo),
                                                                 hexs(mu), hexs(sg), hexs(nu)))
    return out, dropped


def logpdf_points():
    pts = []
    for fam, nus in ((1, NU1), (2, NU2)):
        for nu in nus:
            zs = ZS + [short(math.sqrt(nu)), -short(math.sqrt(nu))] if fam == 1 else ZS
            for z0 in zs:
                mu, sg = COMBOS[0]
                pts.append((fam, point_y(mu, sg, z0), mu, sg, nu))
        for nu in (nus[1], nus[3], nus[5], nus[7], nus[9]):
            for z0 in [short(v) for v in (0.0, 0.3, -1.0, 4.0, 30.0, -1000.0, 1e8)]:
                for mu, sg in COMBOS[1:]:
                    pts.append((fam, point_y(mu, sg, z0), mu, sg, nu))
    return pts


# ------------------------------------------------------------------ the information

def info_task(args):
    fam, mu, sg, nu = args
    out, dropped = [], 0
    names = {3: 3, 4: 4, 5: 5}
    for pos in (0, 3, 4, 5):
        for K in range(0, 5):
            for b in range(K + 1):
                c = K - b
                if pos == 5 and b > 0:
                    continue  # I_nunu does not depend on sigma
                vals = None
                for p1, p2 in ((400, 600), (800, 1000)):
                    res = []
                    for prec in (p1, p2):
                        with mpmath.workprec(prec):
                            f = lambda s_, n_: info_values(fam, s_, n_)[pos]
                            res.append(mpmath.diff(f, (mpmath.mpf(sg), mpmath.mpf(nu)), (b, c)))
                    if res[0] == res[1] or abs(res[0] - res[1]) <= AGREE * abs(res[1]):
                        vals = res[1]
                        break
                if vals is None or vals == 0 or not math.isfinite(float(vals)) or abs(float(vals)) < 2.2250738585072014e-308:
                    dropped += 1
                    continue
                hi, lo = split(vals)
                out.append("%d,%d,%d,%d,%d,%s,%s,%s,%s,%s\n" % (fam, K, pos, b, c, hexs(hi), hexs(lo), hexs(mu), hexs(sg), hexs(nu)))
    return out, dropped


def info_points():
    pts = []
    for sg in (2.0 ** -20, 1.3, 2.0 ** 20):
        for nu in [0.01, 0.1, 0.5, 1.0, 2.0, 3.5, 10.0, 15.9, 16.0, 16.1, 17.9, 18.0, 18.1, 19.9, 20.0, 20.1, 30.0, 100.0, 1e3, 1e4, 1e6, 1e8]:
            pts.append((1, 0.0, sg, nu))
        for nu in [2.000001, 2.01, 2.5, 3.0, 5.0, 10.0, 16.0, 17.9, 18.0, 18.1, 19.9, 20.0, 20.1, 30.0, 100.0, 1e4, 1e8]:
            pts.append((2, 0.0, sg, nu))
    return pts


# ------------------------------------------------------------------ log F

def log_cdf_def(z, nu):
    half = mpmath.mpf(1) / 2
    x = nu / (nu + z * z)
    y = z * z / (nu + z * z)
    if z < 0:
        t = mpmath.betainc(nu / 2, half, 0, x, regularized=True) / 2 if x <= 0.5 else (1 - mpmath.betainc(half, nu / 2, 0, y, regularized=True)) / 2
        return mpmath.log(t)
    z = -z
    x = nu / (nu + z * z)
    y = z * z / (nu + z * z)
    t = mpmath.betainc(nu / 2, half, 0, x, regularized=True) / 2 if x <= 0.5 else (1 - mpmath.betainc(half, nu / 2, 0, y, regularized=True)) / 2
    return mpmath.log1p(-t)


KEYS = [(i, k - i) for k in range(1, 7) for i in range(k, -1, -1)]


def table_at(z, nu, prec):
    with mpmath.workprec(prec):
        U, V = mpmath.mpf(z), mpmath.mpf(nu)
        out = {(0, 0): log_cdf_def(U, V)}
        for k in KEYS:
            out[k] = mpmath.diff(log_cdf_def, (U, V), k)
        return out


def table_agree(p, q):
    return all(abs(p[k] - q[k]) <= AGREE * abs(q[k]) or p[k] == q[k] for k in q)


def table_task(args):
    """the (z, nu) table at an exact z given as a (mu, sigma, nu, y, fam) description"""
    key, z, nu = args
    try:
        for p1, p2 in ((280, 360), (440, 560)):
            a, b = table_at(z, nu, p1), table_at(z, nu, p2)
            if table_agree(a, b):
                return key, b
        return key, None
    except Exception as e:  # noqa: BLE001
        return key, repr(e)


# multivariate truncated polynomials in (dmu, dsigma, dnu) as dict {(a, b, c): coefficient}, total degree <= 6
DEG = 6


def pmul(p, q):
    r = {}
    for (a1, b1, c1), v1 in p.items():
        for (a2, b2, c2), v2 in q.items():
            if a1 + a2 + b1 + b2 + c1 + c2 > DEG:
                continue
            k = (a1 + a2, b1 + b2, c1 + c2)
            r[k] = r.get(k, 0) + v1 * v2
    return r


def padd(p, q, s=1):
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + s * v
    return r


def pconst(v):
    return {(0, 0, 0): v}


def pscale(p, s):
    return {k: v * s for k, v in p.items()}


def ppow_series(p0, p, coeffs):
    """sum_k coeffs[k] (p/p0)^k for p without constant term, p0 the constant: returns the polynomial"""
    r = pconst(coeffs[0])
    pw = pconst(mpmath.mpf(1))
    q = pscale(p, 1 / p0)
    for k in range(1, DEG + 1):
        pw = pmul(pw, q)
        r = padd(r, pscale(pw, coeffs[k]))
    return r


def compose_cdf(table, z0, mu0, sg0, nu0, y, rho_family, sgn=1):
    """the Taylor polynomial in (dmu, dsigma, dnu) of f(sgn z(delta), nu0 + dnu), where f has the table of
    its (z, nu) derivatives at z0 = sgn z(0) and z(delta) = (y - mu0 - dmu)/((sigma0 + dsigma) rho(nu0 + dnu))"""
    mp = mpmath
    dnu = {(0, 0, 1): mp.mpf(1)}
    if rho_family == 2:
        # 1/nu = (1/nu0) sum (-dnu/nu0)^k; rho^2 = 1 - 2/nu; rho = sqrt(rho0^2 + p), p without constant term
        inv = ppow_series(nu0, dnu, [(-1) ** k / nu0 for k in range(DEG + 1)])
        rho2 = padd(pconst(mp.mpf(1)), inv, -2)
        r0sq = rho2[(0, 0, 0)]
        p = {k: v for k, v in rho2.items() if k != (0, 0, 0)}
        coeffs = [mp.binomial(mp.mpf(1) / 2, k) for k in range(DEG + 1)]
        rho = pscale(ppow_series(r0sq, p, coeffs), mp.sqrt(r0sq))
    else:
        rho = pconst(mp.mpf(1))
    sig = {(0, 0, 0): sg0, (0, 1, 0): mp.mpf(1)}
    den = pmul(sig, rho)
    d0 = den[(0, 0, 0)]
    pden = {k: v for k, v in den.items() if k != (0, 0, 0)}
    invden = ppow_series(d0, pden, [(-1) ** k / d0 for k in range(DEG + 1)])
    num = {(0, 0, 0): mp.mpf(y) - mu0, (1, 0, 0): mp.mpf(-1)}
    zpoly = pscale(pmul(num, invden), sgn)
    dz = padd(zpoly, pconst(z0), -1)  # without constant term
    res = {}
    pw = pconst(mp.mpf(1))
    for i in range(DEG + 1):
        if i > 0:
            pw = pmul(pw, dz)
        for j in range(DEG + 1 - i):
            coef = (table[(i, j)] if (i + j) > 0 else table[(0, 0)]) / (mp.factorial(i) * mp.factorial(j))
            res = padd(res, pmul(pw, {(0, 0, j): coef}) if j > 0 else pscale(pw, coef))
    return res


def logcdf_points():
    """(fam, mu, sigma, nu, y, z, key) with the table keys (z, nu) shared between points"""
    pts = []
    nus1 = [0.01, 0.5, 1.0, 2.0, 3.5, 10.0, 100.0, 1e4, 1e8]
    for nu in nus1:
        zs = [short(v) for v in [0.0, 0.3, 1.0, -1.0, 4.0, -4.0, 30.0, -30.0, 1000.0, -1000.0, 1e8, -1e8]] + [short(math.sqrt(nu)), -short(math.sqrt(nu))]
        for z0 in zs:
            mu, sg = COMBOS[0]
            pts.append((1, mu, sg, nu, point_y(mu, sg, z0), z0))
    for nu in (0.5, 3.5, 100.0):
        for z0 in (0.0, 1.0, -4.0, 30.0):
            for mu, sg in COMBOS[1:]:
                pts.append((1, mu, sg, nu, point_y(mu, sg, z0), z0))
    nus2 = [2.000001, 2.5, 5.0, 30.0, 1e4, 1e8]
    for nu in nus2:
        for z0 in [short(v) for v in [0.0, 0.3, -0.3, 1.0, -1.0, 4.0, -4.0, 30.0, -30.0]]:
            mu, sg = COMBOS[0]
            pts.append((2, mu, sg, nu, point_y(mu, sg, z0), z0))
    for nu in (5.0, 30.0):
        for z0 in (1.0, -4.0):
            for mu, sg in COMBOS[1:]:
                pts.append((2, mu, sg, nu, point_y(mu, sg, z0), z0))
    return pts


def exact_z(fam, mu, sg, nu, y):
    with mpmath.workprec(600):
        s = mpmath.mpf(sg) * (mpmath.sqrt((mpmath.mpf(nu) - 2) / mpmath.mpf(nu)) if fam == 2 else 1)
        return (mpmath.mpf(y) - mpmath.mpf(mu)) / s


def table_key(zt, nu):
    return (mpmath.nstr(zt, 60), nu)


def logcdf_rows(point, tabs):
    fam, mu, sg, nu, y, z0 = point
    rows, dropped = [], 0
    for upper in (0, 1):
        with mpmath.workprec(600):
            zexact = exact_z(fam, mu, sg, nu, y)
            zt = -zexact if upper else zexact
            tab = tabs[table_key(zt, nu)]
        if tab is None or isinstance(tab, str):
            dropped += 83
            continue
        with mpmath.workprec(500):
            poly = compose_cdf(tab, zt, mpmath.mpf(mu), mpmath.mpf(sg), mpmath.mpf(nu), y, fam, -1 if upper else 1)
            for K in range(1, 7):
                for (a, b, c) in sorted_comps(K):
                    v = poly.get((a, b, c), 0) * mpmath.factorial(a) * mpmath.factorial(b) * mpmath.factorial(c)
                    if v == 0 or not math.isfinite(float(v)) or abs(float(v)) < 2.2250738585072014e-308:
                        dropped += 1
                        continue
                    hi, lo = split(v)
                    rows.append("%d,%d,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s\n" % (fam, upper, hexs(y), K, a, b, c, hexs(hi), hexs(lo), hexs(mu), hexs(sg), hexs(nu)))
    return rows, dropped


def check_composition():
    """compose_cdf against mpmath.diff of the three-variable definition, orders up to 3"""
    for fam, nu0, z0 in [(1, 3.5, 1.2), (2, 5.0, -0.8)]:
        mu0, sg0, y = 0.5, 1.0, 0.5 + 1.0 * z0 * (math.sqrt((nu0 - 2) / nu0) if fam == 2 else 1.0)
        with mpmath.workprec(400):
            zex = exact_z(fam, mu0, sg0, nu0, y)
            tab = table_at(zex, nu0, 400)
            poly = compose_cdf(tab, zex, mpmath.mpf(mu0), mpmath.mpf(sg0), mpmath.mpf(nu0), y, fam)

            def f(m, s, n):
                sc = s * mpmath.sqrt((n - 2) / n) if fam == 2 else s
                return log_cdf_def((mpmath.mpf(y) - m) / sc, n)

            worst = 0
            for K in range(1, 4):
                for (a, b, c) in sorted_comps(K):
                    d = mpmath.diff(f, (mpmath.mpf(mu0), mpmath.mpf(sg0), mpmath.mpf(nu0)), (a, b, c))
                    v = poly.get((a, b, c), 0) * mpmath.factorial(a) * mpmath.factorial(b) * mpmath.factorial(c)
                    worst = max(worst, abs(d - v) / abs(d))
            print("composition fam %d: worst relative difference against mpmath.diff (orders 1-3): %s" % (fam, mpmath.nstr(worst, 3)))
            assert worst < mpmath.mpf(2) ** -90


# ------------------------------------------------------------------ main

def write(path, header, rows):
    with open(path, "w", newline="\n") as f:
        f.write(header + "\n")
        f.writelines(rows)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "forms"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if what == "forms":
        check_forms()
        return
    if what == "logpdf":
        pts = logpdf_points()
        with Pool(workers) as pool:
            results = pool.map(logpdf_task, pts, chunksize=4)
        rows = [r for rs, _ in results for r in rs]
        dropped = sum(d for _, d in results)
        write(OUT / "distrib_student_t_logpdf.csv", "fam,y,K,a,b,c,hi,lo,mu,sigma,nu", rows)
        print("logpdf: %d points, %d rows, %d dropped" % (len(pts), len(rows), dropped))
    elif what == "info":
        pts = info_points()
        with Pool(workers) as pool:
            results = pool.map(info_task, pts)
        rows = [r for rs, _ in results for r in rs]
        write(OUT / "distrib_student_t_info.csv", "fam,K,pair,b,c,hi,lo,mu,sigma,nu", rows)
        print("info: %d points, %d rows, %d dropped" % (len(pts), len(rows), sum(d for _, d in results)))
    elif what == "check":
        check_composition()
    elif what == "logcdf":
        pts = logcdf_points()
        jobs = {}
        for (fam, mu, sg, nu, y, z0) in pts:
            for upper in (0, 1):
                with mpmath.workprec(600):
                    ze = exact_z(fam, mu, sg, nu, y)
                    zt = -ze if upper else ze
                    key = table_key(zt, nu)
                    jobs[key] = (key, zt, nu)
        print("%d points, %d tables" % (len(pts), len(jobs)), flush=True)
        with Pool(workers) as pool:
            tabs = dict(pool.imap_unordered(table_task, list(jobs.values())))
        rows, dropped = [], 0
        for p in pts:
            r, d = logcdf_rows(p, tabs)
            rows += r
            dropped += d
        write(OUT / "distrib_student_t_logcdf.csv", "fam,upper,y,K,a,b,c,hi,lo,mu,sigma,nu", rows)
        print("logcdf: %d points, %d rows, %d dropped" % (len(pts), len(rows), dropped))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
