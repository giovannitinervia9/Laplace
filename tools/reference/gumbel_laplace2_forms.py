"""Checks of the closed forms of gumbel_distrib and laplace2_distrib (headers
include/laplace/distrib/gumbel.hpp and laplace2.hpp), each against an independent route:
SymPy's own derivatives of the definition, mpmath quadrature, or series expansions.

Usage (from the repository root):  python -I tools/reference/gumbel_laplace2_forms.py

Gumbel: l = -log(s) - z - exp(-z), z = (y - mu)/s, s = sigma. log F = -exp(-z). With
V = exp(-z) the forms checked are, for a derivatives in mu and b in s (K = a + b):
  d^a_mu d^b_s l      = lin + exp + logs, where
      lin  = -(-1)^b b! z s^-b          (a = 0),  (-1)^b b! s^-(b+1)  (a = 1),  0 (a >= 2)
      exp  = -(-1)^b b! s^-K e^-z L_b^(a-1)(z)              (the same as d^a d^b of -e^-z)
      logs = (-1)^b (b-1)! s^-b         (a = 0, b >= 1),   0 otherwise
  d^a_mu d^b_s log F  = -(-1)^b b! s^-K e^-z L_b^(a-1)(z)
where L_b^(alpha) is the generalized Laguerre polynomial, here by its forward recurrence
(the one the C++ uses).
log(1 - F) = -z + H(e^-z) for z >= 0, with H(u) = log((1 - e^-u)/u) = sum_k h_k u^k
(checked against SymPy's series), and = -sum_n exp(-n e^-z)/n for z < 0; the z-derivatives
of exp(-w), w = n e^-z, are exp(-w) P_j(w) with P_0 = 1, P_(j+1)(w) = w (P_j - P_j').
Expected information (score (1 - V)/s and (-1 + z (1 - V))/s, V ~ Exp(1)): closed forms
1/s^2, (gamma - 1)/s^2 and ((1 - gamma)^2 + pi^2/6)/s^2, checked by quadrature.

laplace2: l = log(lambda) - log 2 - lambda |y - mu|. With r = y - mu and s = sign(r)
(s = 0 at the kink), the components (a, b) with a derivatives in mu, b in lambda are:
  a = 0: b = 1: 1/lambda - |r|;  b >= 2: (-1)^(b-1) (b-1)!/lambda^b;  (b = 0 is the value)
  a = 1: b = 0: lambda s;  b = 1: s;  b >= 2 or a >= 2: 0 (away from the kink)
The series of log F and log(1 - F) in lambda: for the term exp(-n lambda q), with
q = y - mu (log F) or mu - y (log(1 - F)), eps = d q / d mu and u = n lambda q,
  d^a_mu d^b_lambda exp(-n lambda q) = (-n eps)^a lambda^(a-b) exp(-u) b! L_b^(a-b)(u).
"""
import math
from fractions import Fraction

import mpmath
import sympy as sp

mpmath.mp.dps = 50


def lag(n, alpha, x):
    """Generalized Laguerre L_n^(alpha)(x) by the forward recurrence, any real alpha."""
    if n == 0:
        return mpmath.mpf(1)
    l0, l1 = mpmath.mpf(1), 1 + alpha - x
    for k in range(1, n):
        l0, l1 = l1, ((2 * k + 1 + alpha - x) * l1 - (k + alpha) * l0) / (k + 1)
    return l1


def fact(n):
    return mpmath.factorial(n)


def rel_err(a, b):
    return abs(a - b) / max(abs(b), mpmath.mpf(10) ** -30)


def check_gumbel_components():
    mu, s, y = sp.symbols("mu s y", real=True)
    z = (y - mu) / s
    lpdf = -sp.log(s) - z - sp.exp(-z)
    logF = -sp.exp(-z)
    pts = [(mpmath.mpf("0.4"), mpmath.mpf("1.3"), mpmath.mpf("0.7")),
           (mpmath.mpf("0.4"), mpmath.mpf("1.3"), mpmath.mpf("-3.1")),
           (mpmath.mpf("-0.2"), mpmath.mpf("0.6"), mpmath.mpf("4.5"))]
    worst_pdf = worst_F = 0.0
    for m0, s0, y0 in pts:
        zv = (y0 - m0) / s0
        e = mpmath.exp(-zv)
        for K in range(1, 7):
            for b in range(K + 1):
                a = K - b
                sg = (-1) ** b
                lin = 0
                if a == 0:
                    lin = -sg * fact(b) * zv * s0 ** (-b)
                elif a == 1:
                    lin = sg * fact(b) * s0 ** (-b - 1)
                logs = sg * fact(b - 1) * s0 ** (-b) if (a == 0 and b >= 1) else 0
                ex = -sg * fact(b) * s0 ** (-K) * e * lag(b, a - 1, zv)
                closed = lin + ex + logs
                d = lpdf
                dF = logF
                if a:
                    d = sp.diff(d, mu, a)
                    dF = sp.diff(dF, mu, a)
                if b:
                    d = sp.diff(d, s, b)
                    dF = sp.diff(dF, s, b)
                sub = {mu: sp.Float(str(m0), 60), s: sp.Float(str(s0), 60), y: sp.Float(str(y0), 60)}
                ref = mpmath.mpf(str(sp.N(d.subs(sub), 50)))
                reffF = mpmath.mpf(str(sp.N(dF.subs(sub), 50)))
                # the closed log F form
                closedF = -sg * fact(b) * s0 ** (-K) * e * lag(b, a - 1, zv)
                worst_pdf = max(worst_pdf, float(abs(closed - ref) / max(abs(ref), 1e-30)))
                worst_F = max(worst_F, float(abs(closedF - reffF) / max(abs(reffF), 1e-30)))
                if float(abs(closed - ref)) > 1e-30 + 1e-25 * abs(float(ref)):
                    print("  logpdf mismatch", K, a, b, float(closed), float(ref))
                if float(abs(closedF - reffF)) > 1e-30 + 1e-25 * abs(float(reffF)):
                    print("  log F mismatch", K, a, b, float(closedF), float(reffF))
    print(f"gumbel logpdf components (K 1..6, all (a,b)): max relative difference {worst_pdf:.3g}")
    print(f"gumbel log F components (K 1..6, all (a,b)):  max relative difference {worst_F:.3g}")


def check_gumbel_h_series():
    u = sp.symbols("u")
    ser = sp.series(sp.log((1 - sp.exp(-u)) / u), u, 0, 13).removeO()
    coef = {k: sp.nsimplify(ser.coeff(u, k)) for k in range(0, 13)}
    # the Bernoulli form: h_1 = -1/2, h_(2n) = B_(2n) / (2n (2n)!)
    ok = True
    for k in range(1, 13):
        if k == 1:
            h = sp.Rational(-1, 2)
        elif k % 2 == 0:
            h = sp.bernoulli(k) / (k * sp.factorial(k))
        else:
            h = 0
        if sp.simplify(h - coef[k]) != 0:
            ok = False
            print("  h mismatch at k", k, h, coef[k])
    print("gumbel H series: Bernoulli form equals SymPy's series up to u^12:", ok)
    # the truncation error of the sum to k = 20 at u = 1 (the largest u used on the z >= 0 route)
    trunc = mpmath.mpf(0)
    for k in range(22, 80, 2):
        trunc += mpmath.mpf(sp.N(sp.bernoulli(k) / (k * sp.factorial(k)), 40))
    print(f"  tail of the series beyond k = 20 at u = 1 (approximate, sum over k <= 79): {float(trunc):.3g}")
    # the z-route: log(1 - exp(-exp(-z))) = -z + H(exp(-z)) at a few z >= 0
    for zv in (0.0, 0.3, 1.0, 2.5):
        zz = mpmath.mpf(zv)
        uu = mpmath.exp(-zz)
        direct = mpmath.log(1 - mpmath.exp(-uu))
        hs = sum(mpmath.mpf(str(sp.N(coef[k], 50))) * uu ** k for k in range(1, 13))
        print(f"  z = {zv}: -z + H(u) - log(1 - F) = {float(-zz + hs - direct):.3g}")


def check_gumbel_touchard():
    # d^j/dz^j exp(-n e^-z) = exp(-w) P_j(w), w = n e^-z, P_(j+1) = w (P_j - P_j')
    z, w = sp.symbols("z w")
    for n in (1, 3):
        f = sp.exp(-n * sp.exp(-z))
        P = sp.Integer(1)
        worst = 0.0
        for j in range(0, 7):
            d = sp.diff(f, z, j) if j else f
            zv = sp.Rational(7, 10)
            ref = sp.N(d.subs(z, zv), 40)
            wv = n * sp.exp(-zv)
            mine = sp.N(sp.exp(-wv) * P.subs(w, wv), 40)
            worst = max(worst, float(abs(ref - mine) / abs(ref)))
            P = sp.expand(w * (P - sp.diff(P, w)))
        print(f"gumbel Touchard recursion, n = {n}: max relative difference (j <= 6) {worst:.3g}")
    P = sp.Integer(1)
    out = []
    for j in range(1, 7):
        P = sp.expand(w * (P - sp.diff(P, w)))
        out.append(str(P))
    print("  P_j(w):", out)


def check_gumbel_info():
    gamma = mpmath.euler
    pi2_6 = mpmath.pi ** 2 / 6
    f = lambda zz: mpmath.exp(-zz - mpmath.exp(-zz))
    V = lambda zz: mpmath.exp(-zz)
    smu = lambda zz: 1 - V(zz)
    ss = lambda zz: -1 + zz * (1 - V(zz))
    I = {}
    I["mm"] = mpmath.quad(lambda zz: smu(zz) ** 2 * f(zz), [-mpmath.inf, 0, mpmath.inf])
    I["ms"] = mpmath.quad(lambda zz: smu(zz) * ss(zz) * f(zz), [-mpmath.inf, 0, mpmath.inf])
    I["ss"] = mpmath.quad(lambda zz: ss(zz) ** 2 * f(zz), [-mpmath.inf, 0, mpmath.inf])
    print("gumbel expected information (multiply by 1/sigma^2):")
    print(f"  mu mu  quad {float(I['mm']):.17g}  closed 1")
    print(f"  mu s   quad {float(I['ms']):.17g}  closed gamma - 1 = {float(gamma - 1):.17g}")
    print(f"  s s    quad {float(I['ss']):.17g}  closed (1-gamma)^2 + pi^2/6 = {float((1 - gamma) ** 2 + pi2_6):.17g}")
    print(f"  gamma = {gamma}, pi^2/6 = {pi2_6}")
    # derivatives of the closed constants in sigma: d^K sigma^-2 = (-1)^K (K+1)! sigma^-(K+2)
    s = sp.symbols("s", positive=True)
    for K in range(1, 5):
        d = sp.diff(s ** -2, s, K)
        form = (-1) ** K * sp.factorial(K + 1) * s ** (-(K + 2))
        print(f"  d^{K}/ds^{K} s^-2 - (-1)^{K} {K + 1}! s^-{K + 2} = {sp.simplify(d - form)}")


def check_laplace2():
    mu, lam, y = sp.symbols("mu lam y", real=True)
    pts = [(mpmath.mpf("0.4"), mpmath.mpf("0.8"), mpmath.mpf("1.1")),
           (mpmath.mpf("0.4"), mpmath.mpf("0.8"), mpmath.mpf("-0.9")),
           (mpmath.mpf("-0.2"), mpmath.mpf("2.5"), mpmath.mpf("0.05"))]
    worst = 0.0
    for sign_region in (1, -1):
        l = sp.log(lam) - sp.log(2) - lam * (sign_region * (y - mu))
        for m0, l0, y0 in pts:
            r = y0 - m0
            if sign_region * r <= 0:
                continue
            sg = sp.sign(sign_region * r)
            sgn = float(sign_region) if r != 0 else 0.0
            for K in range(1, 7):
                for b in range(K + 1):
                    a = K - b
                    d = l
                    if a:
                        d = sp.diff(d, mu, a)
                    if b:
                        d = sp.diff(d, lam, b)
                    sub = {mu: sp.Float(str(m0), 60), lam: sp.Float(str(l0), 60), y: sp.Float(str(y0), 60)}
                    ref = mpmath.mpf(str(sp.N(d.subs(sub), 50)))
                    rr = abs(r)
                    sgnr = 1 if r > 0 else -1
                    if a == 0 and b == 0:
                        closed = mpmath.log(l0) - mpmath.log(2) - l0 * rr
                    elif a == 0 and b == 1:
                        closed = 1 / l0 - rr
                    elif a == 0 and b >= 2:
                        closed = (-1) ** (b - 1) * fact(b - 1) / l0 ** b
                    elif a == 1 and b == 0:
                        closed = l0 * sgnr
                    elif a == 1 and b == 1:
                        closed = mpmath.mpf(sgnr)
                    else:
                        closed = mpmath.mpf(0)
                    worst = max(worst, float(abs(closed - ref)))
    print(f"laplace2 logpdf components (away from the kink): max absolute difference {worst:.3g}")
    # series in lambda: d^a_mu d^b_lambda exp(-n lambda q)
    worst = 0.0
    for n in (1, 2, 3):
        for eps, qexpr in ((-1, y - mu), (1, mu - y)):  # eps = d q / d mu
            T = sp.exp(-n * lam * qexpr)
            for (m0, l0, y0) in pts:
                q = (y0 - m0) if eps == -1 else (m0 - y0)
                if q <= 0:
                    continue
                for K in range(1, 7):
                    for b in range(K + 1):
                        a = K - b
                        d = T
                        if a:
                            d = sp.diff(d, mu, a)
                        if b:
                            d = sp.diff(d, lam, b)
                        sub = {mu: sp.Float(str(m0), 60), lam: sp.Float(str(l0), 60), y: sp.Float(str(y0), 60)}
                        ref = mpmath.mpf(str(sp.N(d.subs(sub), 50)))
                        u = n * l0 * q
                        closed = (-n * eps) ** a * l0 ** (a - b) * mpmath.exp(-u) * fact(b) * lag(b, a - b, u)
                        scale = max(abs(ref), mpmath.mpf(10) ** -40)
                        worst = max(worst, float(abs(closed - ref) / scale))
    print(f"laplace2 series term exp(-n lambda q), all (a,b) with K <= 6: max relative difference {worst:.3g}")
    # expected information by quadrature with the density (lambda/2) exp(-lambda |r|)
    l0 = mpmath.mpf("0.8")
    dens = lambda r: l0 / 2 * mpmath.exp(-l0 * abs(r))
    Imm = mpmath.quad(lambda r: (l0 * mpmath.sign(r)) ** 2 * dens(r), [-mpmath.inf, 0, mpmath.inf])
    Iml = mpmath.quad(lambda r: (l0 * mpmath.sign(r)) * (1 / l0 - abs(r)) * dens(r), [-mpmath.inf, 0, mpmath.inf])
    Ill = mpmath.quad(lambda r: (1 / l0 - abs(r)) ** 2 * dens(r), [-mpmath.inf, 0, mpmath.inf])
    print(f"laplace2 expected information at lambda = 0.8: I_mm quad {float(Imm):.17g} vs lambda^2 {float(l0 ** 2):.17g}")
    print(f"  I_ml quad {float(Iml):.3g} (0);  I_ll quad {float(Ill):.17g} vs lambda^-2 {float(1 / l0 ** 2):.17g}")
    lam_s = sp.symbols("lam", positive=True)
    for K in range(1, 5):
        d = sp.diff(lam_s ** -2, lam_s, K)
        form = (-1) ** K * sp.factorial(K + 1) * lam_s ** (-(K + 2))
        print(f"  d^{K}/dlam^{K} lam^-2 - (-1)^{K} {K + 1}! lam^-{K + 2} = {sp.simplify(d - form)}")
        dm = sp.diff(lam_s ** 2, lam_s, K)
        print(f"  d^{K}/dlam^{K} lam^2 = {dm}")


if __name__ == "__main__":
    check_gumbel_components()
    check_gumbel_h_series()
    check_gumbel_touchard()
    check_gumbel_info()
    check_laplace2()
