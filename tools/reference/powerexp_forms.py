"""Symbolic checks of the closed forms transcribed into powerexp_distrib
(include/laplace/distrib/powerexp.hpp).

Usage (from the repository root):
    python -I tools/reference/powerexp_forms.py

The law. With mu real, sigma > 0 (the standard deviation) and p > 0,

    c(p) = sqrt(Gamma(1/p) / Gamma(3/p)),   s = sigma c(p),
    zeta = (y - mu) / sigma,                z = (y - mu) / s = zeta / c(p),
    f(y) = p / (2 s Gamma(1/p)) exp(-|z|^p),

so that Var Y = s^2 Gamma(3/p) / Gamma(1/p) = sigma^2 for every p. The log-density is written
as one function of zeta and p plus two functions of one parameter,

    l = Lambda(zeta, p) + g(p) - log sigma,
    Lambda(zeta, p) = -|zeta|^p exp(A(p)),
    A(p) = -p log c(p) = -(p/2) (lgamma(1/p) - lgamma(3/p)),
    g(p) = log p - log 2 - (3/2) lgamma(1/p) + (1/2) lgamma(3/p),

which is the form the header transcribes: Lambda goes to (mu, sigma, p) through
detail_location_scale::mixed_shape_dk, and g and -log sigma add their own derivatives to the
pure-p and pure-sigma components.

What this checks, each against SymPy's own derivative of the definition:
  1. the identities A(p) = -p log c(p) and g(p) = log p - log 2 - log c(p) - lgamma(1/p), and that
     l above is the log of the density;
  2. the derivatives of lgamma(a/p) in p to order six, assembled by Faa di Bruno through v = a/p
     with the polygamma functions of v, as laplace::distrib::detail_powerexp::lgamma_over_p_derivs
     does;
  3. the derivatives of A and of g to order six, assembled from those;
  4. the table D[c][j] = d^c/dp^c d^j/dzeta^j Lambda, every c + j <= 6, assembled as the header
     does: the Leibniz rule between the falling factorial fall(p, j) = p (p-1) ... (p-j+1) and
     exp(w_j(p)), w_j = (p - j) log|zeta| + A(p), whose p-derivatives are
     w_j' = log|zeta| + A'(p) and w_j^(m) = A^(m)(p) for m >= 2, the exponential differentiated by
     the complete Bell polynomials of those;
  5. the derivatives in p of the falling factorial, from the coefficients of the polynomial;
  6. the two reductions, p = 2 to the Gaussian of standard deviation sigma and p = 1 to the
     Laplace of scale sigma / sqrt(2), and the limit c(p) -> sqrt(3) as p -> infinity, which makes
     the limit law the uniform on [mu - sigma sqrt(3), mu + sigma sqrt(3)];
  7. the expected information in closed form, I(mu, sigma, p), against the integrals of the score
     products over the density (mpmath quadrature), and the divergence of I_mu mu at p <= 1/2.

Nothing of the library's code is used here; the point is that two independent routes to the same
number agree.
"""
import math
import sys

import mpmath
import sympy as sp

MU, SG, P, Y, ZETA = sp.symbols("mu sigma p y zeta", real=True, positive=False)
TOL = 1e-14

fails = []


def report(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fails.append(name)


def close(a, b, tol=TOL):
    a, b = sp.N(a, 40), sp.N(b, 40)
    scale = max(abs(a), abs(b), sp.Float(1e-300))
    return abs(a - b) / scale <= tol, f"relative {float(abs(a - b) / scale):.3g}"


# ---------------------------------------------------------------- 1. the form of l

def c_of_p(p):
    return sp.sqrt(sp.loggamma(1 / p).rewrite(sp.gamma) if False else sp.gamma(1 / p) / sp.gamma(3 / p))


A_EXPR = -(P / 2) * (sp.loggamma(1 / P) - sp.loggamma(3 / P))
G_EXPR = sp.log(P) - sp.log(2) - sp.Rational(3, 2) * sp.loggamma(1 / P) + sp.Rational(1, 2) * sp.loggamma(3 / P)


def check_form():
    print("1. the form of l")
    for pv in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Integer(2), sp.Rational(7, 2)]:
        logc = sp.log(c_of_p(pv))
        ok, d = close(A_EXPR.subs(P, pv), -pv * logc)
        report(f"A(p) = -p log c(p) at p = {pv}", ok, d)
        ok, d = close(G_EXPR.subs(P, pv), sp.log(pv) - sp.log(2) - logc - sp.loggamma(1 / pv))
        report(f"g(p) = log p - log 2 - log c(p) - lgamma(1/p) at p = {pv}", ok, d)
    # l against the logarithm of the density, at points on both sides of mu
    for pv in [sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Rational(9, 2)]:
        for muv, sgv, yv in [(sp.Rational(1, 4), sp.Integer(2), sp.Integer(3)),
                             (sp.Integer(-1), sp.Rational(1, 2), sp.Rational(-7, 3))]:
            cv = c_of_p(pv)
            sv = sgv * cv
            zv = (yv - muv) / sv
            dens = pv / (2 * sv * sp.gamma(1 / pv)) * sp.exp(-abs(zv) ** pv)
            zeta = (yv - muv) / sgv
            l = -abs(zeta) ** pv * sp.exp(A_EXPR.subs(P, pv)) + G_EXPR.subs(P, pv) - sp.log(sgv)
            ok, d = close(l, sp.log(dens))
            report(f"l = log f at p = {pv}, mu = {muv}, sigma = {sgv}, y = {yv}", ok, d)


# ---------------------------------------------------------------- 2. the derivatives of lgamma(a/p)

def bell_complete(m, w):
    """the complete Bell polynomial Y_m(w_1, ..., w_m), so that d^m/dp^m e^w = e^w Y_m"""
    if m == 0:
        return sp.Integer(1)
    return sum(sp.bell(m, k, w[1:m - k + 2]) for k in range(1, m + 1))


def chain(m, f, g):
    """d^m/dp^m f(g(p)) by Faa di Bruno, f[1..m] and g[1..m] the derivatives"""
    return sum(f[k] * sp.bell(m, k, g[1:m - k + 2]) for k in range(1, m + 1))


def lgamma1p_over_p_derivs(a, p, K):
    """[d^m/dp^m lgamma(1 + a/p)] for m = 0..K, as the header assembles it: Faa di Bruno through
    v = 1 + a/p, whose p-derivatives are those of a/p"""
    v = 1 + (sp.Rational(a) / p if not isinstance(p, sp.Expr) else a / p)
    f = [sp.Integer(0)] + [sp.polygamma(k - 1, v) for k in range(1, K + 1)]
    g = [sp.Integer(0)] + [a * (-1) ** k * sp.factorial(k) / p ** (k + 1) for k in range(1, K + 1)]
    return [sp.loggamma(v)] + [chain(m, f, g) for m in range(1, K + 1)]


def check_lgamma_over_p():
    print("2. d^m/dp^m lgamma(1 + a/p), m = 0..6")
    for a in [1, 3]:
        exact = [sp.diff(sp.loggamma(1 + sp.Integer(a) / P), P, m) for m in range(7)]
        for pv in [sp.Rational(1, 3), sp.Integer(1), sp.Rational(5, 2), sp.Integer(40)]:
            got = lgamma1p_over_p_derivs(a, pv, 6)
            for m in range(7):
                ok, d = close(got[m], exact[m].subs(P, pv))
                report(f"a = {a}, p = {pv}, order {m}", ok, d)


# ---------------------------------------------------------------- 3. the derivatives of A and g

def lambda_derivs(p, K):
    """[lambda^(m)(p)], lambda = log c(p) = (1/2) (lgamma(1 + 1/p) - lgamma(1 + 3/p) + log 3), the form
    of the header's shape_derivs, in which the logarithms that cancel are never formed"""
    g1 = lgamma1p_over_p_derivs(1, p, K)
    g3 = lgamma1p_over_p_derivs(3, p, K)
    return [(g1[0] - g3[0] + sp.log(3)) / 2] + [(g1[m] - g3[m]) / 2 for m in range(1, K + 1)]


def a_derivs(p, K):
    """[A^(m)(p)] for m = 0..K: A = -p lambda, Leibniz against the factor p"""
    lam = lambda_derivs(p, K)
    return [-p * lam[0]] + [-(p * lam[m] + m * lam[m - 1]) for m in range(1, K + 1)]


def g_derivs(p, K):
    """[g^(m)(p)] for m = 0..K: g = -log 2 - lgamma(1 + 1/p) - lambda"""
    g1 = lgamma1p_over_p_derivs(1, p, K)
    lam = lambda_derivs(p, K)
    return [-sp.log(2) - g1[0] - lam[0]] + [-g1[m] - lam[m] for m in range(1, K + 1)]


def check_ag():
    print("3. the derivatives of A and of g, orders 0..6, in the shifted forms against SymPy's own")
    print("   derivatives of the ORIGINAL definitions through lgamma(1/p) and lgamma(3/p)")
    exA = [sp.diff(A_EXPR, P, m) for m in range(7)]
    exG = [sp.diff(G_EXPR, P, m) for m in range(7)]
    for pv in [sp.Rational(1, 3), sp.Integer(1), sp.Rational(5, 2), sp.Integer(40)]:
        a, g = a_derivs(pv, 6), g_derivs(pv, 6)
        for m in range(7):
            ok, d = close(a[m], exA[m].subs(P, pv))
            report(f"A at p = {pv}, order {m}", ok, d)
            ok, d = close(g[m], exG[m].subs(P, pv))
            report(f"g at p = {pv}, order {m}", ok, d)


# ---------------------------------------------------------------- 4, 5. the table D and fall(p, j)

def fall_coefs(K):
    """coefficients in p of fall(p, j) = p (p-1) ... (p-j+1), j = 0..K"""
    co = [[sp.Integer(1)]]
    for j in range(K):
        prev = co[j]
        new = [sp.Integer(0)] * (j + 2)
        for n, v in enumerate(prev):
            new[n + 1] += v      # p * prev
            new[n] -= j * v      # -j * prev
        co.append(new)
    return co


def fall_derivs(p, j, K, co):
    """[d^i/dp^i fall(p, j)] for i = 0..K, from the coefficients"""
    out = []
    for i in range(K + 1):
        s = sp.Integer(0)
        for n in range(i, len(co[j])):
            s += co[j][n] * sp.factorial(n) / sp.factorial(n - i) * p ** (n - i)
        out.append(s)
    return out


def check_fall():
    print("5. the derivatives in p of fall(p, j)")
    co = fall_coefs(6)
    for j in range(7):
        exact = sp.prod([P - k for k in range(j)])
        for pv in [sp.Rational(1, 2), sp.Integer(2), sp.Rational(17, 4)]:
            got = fall_derivs(pv, j, 6, co)
            for i in range(7):
                ok, d = close(got[i], sp.diff(exact, P, i).subs(P, pv))
                report(f"j = {j}, p = {pv}, order {i}", ok, d)


def table_D(zeta, p, K, co):
    """D[c][j] = d^c/dp^c d^j/dzeta^j Lambda, assembled as the header does"""
    L = sp.log(abs(zeta))
    A = a_derivs(p, K)
    w = [sp.Integer(0), L + A[1]] + [A[m] for m in range(2, K + 1)]
    Yv = [bell_complete(m, w) for m in range(K + 1)]
    D = [[None] * (K + 1) for _ in range(K + 1)]
    for j in range(K + 1):
        sj = sp.Integer(1) if j % 2 == 0 else sp.sign(zeta)
        F = fall_derivs(p, j, K, co)
        Tj = sp.exp(p * (L - sp.log(c_of_p(p))) - j * L)
        for c in range(K + 1 - j):
            s = sum(sp.binomial(c, i) * F[i] * Yv[c - i] for i in range(c + 1))
            D[c][j] = -sj * Tj * s
    return D


def check_table():
    print("4. the table D[c][j] = d^c/dp^c d^j/dzeta^j Lambda, c + j <= 6")
    co = fall_coefs(6)
    lam = -abs(ZETA) ** P * sp.exp(A_EXPR)
    for zv in [sp.Rational(3, 7), sp.Integer(2), sp.Rational(-5, 3)]:
        pos = zv > 0
        lam_s = (-(ZETA if pos else -ZETA) ** P * sp.exp(A_EXPR))
        for pv in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Integer(2), sp.Rational(9, 2)]:
            D = table_D(zv, pv, 6, co)
            for j in range(7):
                dz = sp.diff(lam_s, ZETA, j)
                for c in range(7 - j):
                    exact = sp.diff(dz, P, c).subs({ZETA: zv, P: pv})
                    ok, d = close(D[c][j], exact, 1e-12)
                    report(f"zeta = {zv}, p = {pv}, (c, j) = ({c}, {j})", ok, d)


# ---------------------------------------------------------------- 6. the reductions and the limit

def check_reductions():
    print("6. the reductions at p = 2 and p = 1, and the limit of c(p)")
    ok, d = close(c_of_p(sp.Integer(2)), sp.sqrt(2))
    report("c(2) = sqrt(2)", ok, d)
    ok, d = close(c_of_p(sp.Integer(1)), 1 / sp.sqrt(2))
    report("c(1) = 1/sqrt(2)", ok, d)
    for muv, sgv, yv in [(sp.Rational(1, 4), sp.Integer(2), sp.Integer(3)), (sp.Integer(-1), sp.Rational(1, 2), sp.Rational(-7, 3))]:
        l2 = -abs((yv - muv) / sgv) ** 2 * sp.exp(A_EXPR.subs(P, 2)) + G_EXPR.subs(P, 2) - sp.log(sgv)
        gauss = -sp.log(sgv * sp.sqrt(2 * sp.pi)) - (yv - muv) ** 2 / (2 * sgv ** 2)
        ok, d = close(l2, gauss)
        report(f"p = 2 is the Gaussian(mu, sigma) at y = {yv}", ok, d)
        l1 = -abs((yv - muv) / sgv) * sp.exp(A_EXPR.subs(P, 1)) + G_EXPR.subs(P, 1) - sp.log(sgv)
        b = sgv / sp.sqrt(2)
        lap = -sp.log(2 * b) - abs(yv - muv) / b
        ok, d = close(l1, lap)
        report(f"p = 1 is the Laplace(mu, sigma/sqrt 2) at y = {yv}", ok, d)
    # c(p) -> sqrt(3), so s -> sigma sqrt(3) and the limit law is uniform on mu +- sigma sqrt(3)
    big = c_of_p(sp.Integer(10 ** 7))
    ok, d = close(big, sp.sqrt(3), 1e-6)
    report("c(p) -> sqrt(3) as p -> infinity", ok, d)
    print("     c(1e7) = %.12f, sqrt(3) = %.12f" % (float(sp.N(big)), float(sp.N(sp.sqrt(3)))))


# ---------------------------------------------------------------- 7. the expected information

def info_closed(p):
    """the closed form of the header, as a 3x3 list in (mu, sigma, p) with sigma = 1"""
    a = mpmath.mpf(1) / p
    psi = lambda x: mpmath.digamma(x)
    psi1 = lambda x: mpmath.polygamma(1, x)
    logc = (mpmath.loggamma(a) - mpmath.loggamma(3 * a)) / 2
    c = mpmath.exp(logc)
    s = c  # sigma = 1
    kappa = (3 * psi(3 * a) - psi(a)) / (2 * p ** 2)
    i_mumu = mpmath.inf if p <= mpmath.mpf(1) / 2 else p ** 2 * mpmath.exp(mpmath.loggamma(2 - a) - mpmath.loggamma(a)) / s ** 2
    i_sgsg = p
    i_sgp = p * kappa - (1 + psi(1 + a)) / p
    nat_pp = -a ** 4 * psi(1 + a) ** 2 + a ** 3 * (1 + a) * (psi1(2 + a) + psi(2 + a) ** 2)
    i_pp = nat_pp - 2 * kappa * (1 + psi(1 + a)) / p + p * kappa ** 2
    return [[i_mumu, 0, 0], [0, i_sgsg, i_sgp], [0, i_sgp, i_pp]]


def info_quadrature(p):
    """E[l_a l_b] by quadrature of the scores at sigma = 1, mu = 0"""
    p = mpmath.mpf(p)
    a = 1 / p
    logc = (mpmath.loggamma(a) - mpmath.loggamma(3 * a)) / 2
    c = mpmath.exp(logc)
    lg = mpmath.loggamma(a)

    def logf(y, mu, sg, pp):
        aa = 1 / pp
        lc = (mpmath.loggamma(aa) - mpmath.loggamma(3 * aa)) / 2
        z = abs(y - mu) / (sg * mpmath.exp(lc))
        return mpmath.log(pp) - mpmath.log(2) - mpmath.log(sg) - lc - mpmath.loggamma(aa) - z ** pp

    def score(y):
        h = mpmath.mpf(10) ** -10
        g = [0, 0, 0]
        g[0] = (logf(y, h, 1, p) - logf(y, -h, 1, p)) / (2 * h)
        g[1] = (logf(y, 0, 1 + h, p) - logf(y, 0, 1 - h, p)) / (2 * h)
        g[2] = (logf(y, 0, 1, p + h) - logf(y, 0, 1, p - h)) / (2 * h)
        return g

    out = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            f = lambda y: score(y)[i] * score(y)[j] * mpmath.exp(logf(y, 0, 1, p))
            v = 2 * mpmath.quad(f, [0, c, 4 * c, mpmath.inf]) if (i, j) != (0, 1) else 0
            out[i][j] = out[j][i] = v
    # the odd components vanish by symmetry
    out[0][1] = out[1][0] = 0
    out[0][2] = out[2][0] = 0
    return out


def check_info():
    print("7. the expected information in closed form against the quadrature of the scores")
    mpmath.mp.dps = 30
    for pv in [0.8, 1.0, 1.5, 2.0, 3.0, 6.0]:
        cl = info_closed(mpmath.mpf(pv))
        qu = info_quadrature(pv)
        # relative to the largest entry of the matrix: I_sigma p is exactly zero at p = 2, where
        # the standard deviation and the shape are orthogonal, and a ratio to itself says nothing
        scale = max(abs(cl[i][j]) for i in range(3) for j in range(3) if cl[i][j] != mpmath.inf)
        worst, where = 0.0, ""
        for i in range(3):
            for j in range(i, 3):
                if (i, j) in ((0, 1), (0, 2)):
                    continue
                e = abs(cl[i][j] - qu[i][j]) / max(abs(cl[i][j]), abs(qu[i][j]), scale * mpmath.mpf("1e-6"))
                if e > worst:
                    worst, where = float(e), f"({i},{j})"
        report(f"p = {pv}", worst <= 2e-5, f"worst relative {worst:.3g} at {where}")
    # the orthogonality of sigma and p at the Gaussian, in closed form
    report("I_sigma p = 0 at p = 2", info_closed(mpmath.mpf(2))[1][2] == 0)
    for pv in [0.2, 0.4, 0.5]:
        report(f"I_mu mu infinite at p = {pv}", info_closed(mpmath.mpf(pv))[0][0] == mpmath.inf)
    # the Gaussian: I = diag(1/sigma^2, 2/sigma^2, I_pp) at p = 2 with sigma = 1
    cl = info_closed(mpmath.mpf(2))
    report("I_mu mu = 1 at p = 2, sigma = 1", abs(cl[0][0] - 1) < 1e-12, f"{float(cl[0][0]):.15f}")
    report("I_sigma sigma = 2 at p = 2, sigma = 1", abs(cl[1][1] - 2) < 1e-12, f"{float(cl[1][1]):.15f}")


def main():
    check_form()
    check_lgamma_over_p()
    check_ag()
    check_table()
    check_fall()
    check_reductions()
    check_info()
    check_tensor()
    print()
    if fails:
        print(f"{len(fails)} checks FAILED:")
        for f in fails[:20]:
            print("  " + f)
        sys.exit(1)
    print("every check passed")



# ---------------------------------------------------------------- 8. the factorized tensor

def comps(K):
    """the sorted multi-indices of order K as (a, b, c), in the packing order of sym_tensor<3, K>"""
    return [(K - b - c, b, c) for b in range(K + 1) for c in range(K - b + 1)]


def rise_coefs(K):
    """coefficients in p of rise(p, b) = p (p+1) ... (p+b-1), b = 0..K"""
    co = [[sp.Integer(1)]]
    for b in range(K):
        prev = co[b]
        new = [sp.Integer(0)] * (b + 2)
        for n, v in enumerate(prev):
            new[n + 1] += v
            new[n] += b * v
        co.append(new)
    return co


def rise_derivs(p, b, K, co):
    out = []
    for j in range(K + 1):
        s = sp.Integer(0)
        for n in range(j, len(co[b])):
            s += co[b][n] * sp.factorial(n) / sp.factorial(n - j) * p ** (n - j)
        out.append(s)
    return out


def tensor_component(mu, sg, pv, yv, a, b, c, fco, rco):
    """the component with a derivatives in mu, b in sigma and c in p of
    Lambda = -|y - mu|^p sigma^-p exp(A(p)), as the header assembles it"""
    K = a + b + c
    r = yv - mu
    L = sp.log(abs(r / sg))
    A = a_derivs(pv, max(K, 1))
    w = [sp.Integer(0), L + A[1]] + [A[m] for m in range(2, max(K, 1) + 1)]
    Y = [bell_complete(m, w) for m in range(K + 1)]
    F = fall_derivs(pv, a, K, fco)
    G = rise_derivs(pv, b, K, rco)
    s = sp.Integer(0)
    for i in range(c + 1):
        for j in range(c - i + 1):
            k = c - i - j
            s += sp.factorial(c) / (sp.factorial(i) * sp.factorial(j) * sp.factorial(k)) * F[i] * G[j] * Y[k]
    sgn = sp.Integer(1) if a % 2 == 0 else sp.sign(r)
    expV = sp.exp(pv * (L - sp.log(c_of_p(pv))) - a * L) * sg ** (-a - b)
    return -(-1) ** (a + b) * sgn * expV * s


def check_tensor():
    print("8. the factorized tensor of Lambda in (mu, sigma, p), every (a, b, c) with K <= 6")
    fco, rco = fall_coefs(6), rise_coefs(6)
    for (muv, sgv) in [(sp.Rational(1, 4), sp.Integer(2)), (sp.Integer(-1), sp.Rational(1, 2))]:
        for yv in [muv + sgv * sp.Rational(3, 7), muv - sgv * sp.Rational(5, 3)]:
            pos = yv > muv
            rr = (Y - MU) if pos else (MU - Y)
            lam = -rr ** P * SG ** (-P) * sp.exp(A_EXPR)
            cache = {(0, 0, 0): lam}
            for pv in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Integer(2), sp.Rational(9, 2)]:
                worst, where = 0.0, ""
                for K in range(0, 7):
                    for (a, b, c) in comps(K):
                        if (a, b, c) not in cache:
                            if a > 0:
                                e = sp.diff(cache[(a - 1, b, c)], MU)
                            elif b > 0:
                                e = sp.diff(cache[(a, b - 1, c)], SG)
                            else:
                                e = sp.diff(cache[(a, b, c - 1)], P)
                            cache[(a, b, c)] = e
                        exact = cache[(a, b, c)].subs({MU: muv, SG: sgv, Y: yv, P: pv})
                        got = tensor_component(muv, sgv, pv, yv, a, b, c, fco, rco)
                        ok, d = close(got, exact, 1e-12)
                        if not ok:
                            report(f"mu {muv} sigma {sgv} y {yv} p {pv} (a,b,c) = ({a},{b},{c})", False, d)
                        else:
                            e2 = float(abs(sp.N(got - exact, 40)) / max(abs(sp.N(exact, 40)), sp.Float("1e-300")))
                            if e2 > worst:
                                worst, where = e2, f"({a},{b},{c})"
                report(f"mu {muv} sigma {sgv} y {yv} p {pv}: all 84 components", True, f"worst relative {worst:.3g} at {where}")

if __name__ == "__main__":
    main()
