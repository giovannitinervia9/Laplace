"""SymPy checks of the formulas of include/laplace/math/incomplete_gamma_mixed.hpp.

Usage (from the repository root):
    python -I tools/reference/incomplete_mixed_forms.py

Nothing is written; each check prints ok or raises. The checks:

1. The recursions of the bivariate logarithm and exponential (Euler operator
   theta = h d/dh + e d/de, theta F = F theta log F), against SymPy's series
   of log F and exp L for polynomials with symbolic coefficients, to total
   degree 4.
2. The lower side. With nu = (1 - pi) delta_0 + pi g(d) dd (g a density) and
   y = x - t, the integral I_L(a + h, x (1 + e)) / I_L equals
   E_nu[exp(h d + e y) (1 + c e)] with c = 1 on the density part; the header
   writes it as exp(K) (1 + e (pi + (1 - pi)(1 - exp(-K)))), K = log E_nu[exp(h d + e y)].
   Checked on a measure with a point mass and two atoms in place of the
   density, to total degree 4.
3. The explicit terms in e: with the prefactor x^a e^-x of both sides,
   (a + h) log(x (1 + e)) - x (1 + e) has the e-coefficients a - x (e),
   a (-1)^(m-1)/m (e^m, m >= 2) and (-1)^(m-1)/m (h e^m); the upper series
   adds -log(1 + e); Temme's -a phi = -(x - a) + a log(x/a) gives the same
   e-terms.
4. The first derivatives in closed form: x d log P/dx = a / I_L and
   x d log Q/dx = -1 / I_U, from Gamma(a + 1) P = x^a e^-x I_L and
   Gamma(a) Q = x^a e^-x I_U with I_L, I_U written as integrals in t.
"""
import sympy as sp

h, e = sp.symbols("h e")
DEG = 4


def trunc(expr, deg=DEG):
    """the terms of total degree <= deg of a polynomial in h, e"""
    p = sp.Poly(sp.expand(expr), h, e)
    return sum(c * h ** i * e ** j for (i, j), c in p.terms() if i + j <= deg)


def series2(expr, deg=DEG):
    """bivariate Taylor polynomial at (0, 0) through a scaling variable"""
    s = sp.Symbol("s")
    ser = sp.series(expr.subs({h: s * h, e: s * e}), s, 0, deg + 1).removeO()
    return sp.expand(ser.subs(s, 1))


def coef(p, i, j):
    return sp.Poly(sp.expand(p), h, e).coeff_monomial(h ** i * e ** j)


def check_log_exp():
    F = {(i, j): sp.Symbol(f"F{i}{j}") for i in range(DEG + 1) for j in range(DEG + 1 - i)}
    F[(0, 0)] = sp.Integer(1)
    poly = sum(c * h ** i * e ** j for (i, j), c in F.items())
    ref = series2(sp.log(poly))
    L = {(0, 0): sp.Integer(0)}
    for n in range(1, DEG + 1):
        for i in range(n + 1):
            j = n - i
            acc = n * F[(i, j)]
            for u in range(i + 1):
                for w in range(j + 1):
                    s = u + w
                    if s in (0, n):
                        continue
                    acc -= s * L[(u, w)] * F[(i - u, j - w)]
            L[(i, j)] = sp.expand(acc / n)
    for (i, j), v in L.items():
        assert sp.simplify(v - coef(ref, i, j)) == 0, ("log", i, j)
    Lpoly = sum(v * h ** i * e ** j for (i, j), v in L.items())
    E = {(0, 0): sp.Integer(1)}
    for n in range(1, DEG + 1):
        for i in range(n + 1):
            j = n - i
            acc = 0
            for u in range(i + 1):
                for w in range(j + 1):
                    if u + w == 0:
                        continue
                    acc += (u + w) * L[(u, w)] * E[(i - u, j - w)]
            E[(i, j)] = sp.expand(acc / n)
    refE = series2(sp.exp(Lpoly))
    for (i, j), v in E.items():
        assert sp.simplify(v - coef(refE, i, j)) == 0, ("exp", i, j)
    # exp(log F) = F
    for (i, j), v in E.items():
        assert sp.simplify(v - F[(i, j)]) == 0, ("exp log", i, j)
    print("1. bivariate log and exp recursions: ok")


def check_lower_mixture():
    pi, w1 = sp.symbols("pi w1", positive=True)
    d1, d2, y1, y2 = sp.symbols("d1 d2 y1 y2")
    # nu: mass 1 - pi at (0, 0), pi w1 at (d1, y1), pi (1 - w1) at (d2, y2)
    atoms = [(1 - pi, 0, 0, 0), (pi * w1, d1, y1, 1), (pi * (1 - w1), d2, y2, 1)]
    lhs = sum(m * sp.exp(h * d + e * y) * (1 + c * e) for m, d, y, c in atoms)
    M = sum(m * sp.exp(h * d + e * y) for m, d, y, c in atoms)
    K = sp.log(M)
    rhs = sp.exp(K) * (1 + e * (pi + (1 - pi) * (1 - sp.exp(-K))))
    diff = series2(sp.log(lhs) - sp.log(rhs), 3)
    assert sp.simplify(diff) == 0
    print("2. lower side, the factor 1 + e of the density part: ok")


def check_explicit():
    a, x = sp.symbols("a x", positive=True)
    pref = (a + h) * sp.log(x * (1 + e)) - x * (1 + e)
    ser = series2(pref - pref.subs(e, 0))
    assert sp.simplify(coef(ser, 0, 1) - (a - x)) == 0
    for m in range(2, DEG + 1):
        assert sp.simplify(coef(ser, 0, m) - a * sp.Integer(-1) ** (m - 1) / m) == 0
    for m in range(1, DEG):
        assert sp.simplify(coef(ser, 1, m) - sp.Integer(-1) ** (m - 1) / m) == 0
    for i in range(2, DEG + 1):
        for m in range(1, DEG + 1 - i):
            assert coef(ser, i, m) == 0
    # Temme: -a phi = -(x - a) + a log(x/a) at (a + h, x (1 + e))
    A, X = a + h, x * (1 + e)
    aphi = -(X - A) + A * sp.log(X / A)
    ser2 = series2(aphi - aphi.subs(e, 0))
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2)]:
        assert sp.simplify(coef(ser2, i, j) - coef(ser, i, j)) == 0, ("temme", i, j)
    print("3. explicit terms in e: ok")


def check_first_x():
    a, x, t = sp.symbols("a x t", positive=True)
    # I_L = x^-a e^x int_0^x t^a e^-t dt (a Gamma(a) P = x^a e^-x I_L)
    gl = sp.lowergamma(a + 1, x)
    IL = 1 + x ** (-a) * sp.exp(x) * gl
    # check P' / P = f / P = a / (x I_L) through Gamma(a + 1) P = x^a e^-x I_L
    P = x ** a * sp.exp(-x) * IL / sp.gamma(a + 1)
    f = x ** (a - 1) * sp.exp(-x) / sp.gamma(a)
    val = {a: sp.Rational(7, 3), x: sp.Rational(5, 4)}
    lhs = (x * sp.diff(sp.log(P), x)).subs(val).evalf(40)
    rhs = (a / IL).subs(val).evalf(40)
    assert abs(lhs - rhs) < 1e-30
    assert abs((x * f / P).subs(val).evalf(40) - rhs) < 1e-30
    gu = sp.uppergamma(a, x)
    IU = x ** (-a) * sp.exp(x) * gu
    Q = x ** a * sp.exp(-x) * IU / sp.gamma(a)
    lhs = (x * sp.diff(sp.log(Q), x)).subs(val).evalf(40)
    rhs = (-1 / IU).subs(val).evalf(40)
    assert abs(lhs - rhs) < 1e-30
    print("4. first derivatives in x in closed form: ok")


if __name__ == "__main__":
    check_log_exp()
    check_lower_mixture()
    check_explicit()
    check_first_x()
