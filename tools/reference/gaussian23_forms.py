"""SymPy check of the closed forms of gaussian2 (mean mu, variance v) and gaussian3
(mean mu, precision tau) used in include/laplace/distrib/gaussian2.hpp and gaussian3.hpp.

Usage (from the repository root):  python -I tools/reference/gaussian23_forms.py

Each closed form is written here as a Python function with the same arguments as
its C++ counterpart, and compared with SymPy's own derivative of the definition:
  - the log-density l(mu, w; y) = -log(2 pi)/2 - log(v)/2 - (y - mu)^2/(2 v)
    (gaussian2) and -log(2 pi)/2 + log(tau)/2 - tau (y - mu)^2/2 (gaussian3), components
    (a, b) with a derivatives in mu and b in the second parameter, a + b <= 6;
  - the three closed parts of the tail route of log Phi: -t^2/2 = -(y - mu)^2/(2 v) (and
    -tau (y - mu)^2/2), -log x with x = |y - mu| / sigma, and -w with w = sigma^2/(y - mu)^2
    (gaussian3: w = 1/(tau (y - mu)^2)); each differentiated in (mu, w) in closed form;
  - the expected information and its derivatives to order 4.
The remainder of the tail route, S_2, goes through detail_location_scale::mixed_power_dk
and is not checked here: tests/distrib check it against mpmath.

Prints one line per form and exits with status 1 when a form disagrees.
"""
import sys

import sympy as sp

MAXK = 6
mu, y, v, tau = sp.symbols("mu y v tau", real=True)


def fact(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def c_a(a, r):
    """D_a (r^2) with respect to mu, r = y - mu: r^2, -2 r, 2, 0 for a = 0, 1, 2, > 2."""
    if a == 0:
        return r * r
    if a == 1:
        return -2 * r
    if a == 2:
        return 2
    return 0


# ---- gaussian2: l = -log(2 pi)/2 - log(v)/2 - r^2/(2 v), (a, b) component, a + b >= 1
def g2_logpdf_comp(a, b, r, w):
    if a == 0:  # b >= 1
        return (-1) ** b * (sp.Rational(fact(b - 1), 2) / w**b - sp.Rational(fact(b), 2) * r * r / w ** (b + 1))
    if a == 1:
        return r * (-1) ** b * fact(b) / w ** (b + 1)
    if a == 2:
        return -((-1) ** b) * fact(b) / w ** (b + 1)
    return 0


# ---- gaussian3: l = -log(2 pi)/2 + log(tau)/2 - tau r^2/2, (a, b) component, a + b >= 1
def g3_logpdf_comp(a, b, r, t):
    ta = {0: t, 1: 1}.get(b, 0)  # T_b: d^b/dtau^b tau
    out = -sp.Rational(1, 2) * c_a(a, r) * ta
    if a == 0:  # b >= 1
        out += sp.Rational(1, 2) * (-1) ** (b - 1) * fact(b - 1) / t**b
    return out


# ---- tail parts of log Phi, each in (mu, w); r = y - mu
def g2_tailA(a, b, r, w):  # -r^2/(2 v)
    return -sp.Rational(1, 2) * c_a(a, r) * (-1) ** b * fact(b) / w ** (b + 1)


def g2_tailB(a, b, r, w):  # +log(sigma) = (1/2) log v, and -log|r|
    if a == 0 and b >= 1:
        return sp.Rational(1, 2) * (-1) ** (b - 1) * fact(b - 1) / w**b
    if a >= 1 and b == 0:
        return fact(a - 1) / r**a
    return 0


def g2_tailC(a, b, r, w):  # -w = -v/r^2
    if b == 0:
        return -w * fact(a + 1) / r ** (a + 2)
    if b == 1:
        return -fact(a + 1) / r ** (a + 2)
    return 0


def g3_tailA(a, b, r, t):  # -tau r^2/2
    ta = {0: t, 1: 1}.get(b, 0)
    return -sp.Rational(1, 2) * c_a(a, r) * ta


def g3_tailB(a, b, r, t):  # -log x = -log|r| - (1/2) log tau
    if a == 0 and b >= 1:
        return -sp.Rational(1, 2) * (-1) ** (b - 1) * fact(b - 1) / t**b
    if a >= 1 and b == 0:
        return fact(a - 1) / r**a
    return 0


def g3_tailC(a, b, r, t):  # -w = -1/(tau r^2)
    return -fact(a + 1) / r ** (a + 2) * (-1) ** b * fact(b) / t ** (b + 1)


# ---- expected information (diagonal) and its derivatives in the second parameter
def g2_info_K(K, w):  # I_mumu = 1/v and I_vv = 1/(2 v^2) at order K in v
    return [(-1) ** K * fact(K) / w ** (K + 1), sp.Rational((-1) ** K * fact(K + 1), 2) / w ** (K + 2)]


def g3_info_K(K, t):  # I_mumu = tau, I_tautau = 1/(2 tau^2)
    mm = 1 if K == 1 else 0  # I_mumu = tau: its first derivative in tau is 1, the rest 0
    return [mm, sp.Rational((-1) ** K * fact(K + 1), 2) / t ** (K + 2)]


def check(name, expr_sympy, expr_formula):  # True when the difference simplifies to zero
    diff = sp.simplify(sp.together(sp.expand(expr_sympy - expr_formula)))
    ok = diff == 0
    print(("ok   " if ok else "FAIL ") + name)
    return ok


def main():
    all_ok = True
    r_expr = y - mu
    # gaussian2 log-density, (a, b) with a + b = K, K = 1..6 (the value is checked as K = 0 in the test)
    l2 = -sp.log(2 * sp.pi) / 2 - sp.log(v) / 2 - (y - mu) ** 2 / (2 * v)
    for K in range(1, MAXK + 1):
        for a in range(K + 1):
            b = K - a
            if K >= 1 and a == 0 and b == 0:
                continue
            d = sp.diff(l2, mu, a, v, b) if (a or b) else l2
            f = g2_logpdf_comp(a, b, r_expr, v)
            all_ok &= check(f"gaussian2 logpdf K={K} (a={a}, b={b})", d, f)
    # gaussian3
    l3 = -sp.log(2 * sp.pi) / 2 + sp.log(tau) / 2 - tau * (y - mu) ** 2 / 2
    for K in range(1, MAXK + 1):
        for a in range(K + 1):
            b = K - a
            d = sp.diff(l3, mu, a, tau, b)
            f = g3_logpdf_comp(a, b, r_expr, tau)
            all_ok &= check(f"gaussian3 logpdf K={K} (a={a}, b={b})", d, f)
    # the three closed parts of the tail route, differentiated in (mu, w) (r = y - mu, > 0 on the tail side)
    parts2 = {
        "A": -(y - mu) ** 2 / (2 * v),
        "B": sp.log(sp.sqrt(v)) - sp.log(y - mu),
        "C": -v / (y - mu) ** 2,
    }
    forms2 = {"A": g2_tailA, "B": g2_tailB, "C": g2_tailC}
    parts3 = {
        "A": -tau * (y - mu) ** 2 / 2,
        "B": -sp.log(sp.sqrt(tau)) - sp.log(y - mu),
        "C": -1 / (tau * (y - mu) ** 2),
    }
    forms3 = {"A": g3_tailA, "B": g3_tailB, "C": g3_tailC}
    for K in range(1, MAXK + 1):
        for a in range(K + 1):
            b = K - a
            for name in ("A", "B", "C"):
                d2 = sp.diff(parts2[name], mu, a, v, b) if (a or b) else parts2[name]
                f2 = forms2[name](a, b, y - mu, v)
                all_ok &= check(f"gaussian2 tail {name} K={K} (a={a}, b={b})", d2, f2)
                d3 = sp.diff(parts3[name], mu, a, tau, b) if (a or b) else parts3[name]
                f3 = forms3[name](a, b, y - mu, tau)
                all_ok &= check(f"gaussian3 tail {name} K={K} (a={a}, b={b})", d3, f3)
    # expected information derivatives
    for K in range(1, 5):
        d_mm = sp.diff(1 / v, v, K)
        d_vv = sp.diff(1 / (2 * v**2), v, K)
        f = g2_info_K(K, v)
        all_ok &= check(f"gaussian2 expected information K={K}", d_mm, f[0])
        all_ok &= check(f"gaussian2 expected information K={K} (vv)", d_vv, f[1])
        d_mm = sp.diff(tau, tau, K)
        d_tt = sp.diff(1 / (2 * tau**2), tau, K)
        f = g3_info_K(K, tau)
        all_ok &= check(f"gaussian3 expected information K={K} (mm)", d_mm, f[0])
        all_ok &= check(f"gaussian3 expected information K={K} (tt)", d_tt, f[1])
    print("all forms agree with SymPy" if all_ok else "SOME FORMS DISAGREE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
