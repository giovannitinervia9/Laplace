"""SymPy check of the closed forms of exponential_distrib (mean mu) and chisq_distrib (df mu)
used in include/laplace/distrib/exponential.hpp and chisq.hpp.

Usage (from the repository root):  python -I tools/reference/distrib_expchi_forms.py

Each closed form is compared with SymPy's own derivative of the definition, at rational
points (exact arithmetic, so a mismatch is a wrong formula, not rounding):
  - exponential: the log-density -log(mu) - y/mu, its mu-derivatives of order 1 to 6, the
    expected information 1/mu^2 (from -E[l_mumu] by integrating against the density) and
    its derivatives to order 4, log(1 - F) = -y/mu, and the mixed derivatives in y;
  - the derivatives G^(k)(u) of G(u) = log(1 - e^(-u)), k = 1 to 6, against the Eulerian
    form (-1)^(k+1) q A_(k-1)(q)/(1 - q)^k with q = e^(-u) used for u >= 1;
  - chisq: the log-density through lgamma(mu/2) against the polygamma forms of its
    mu-derivatives (orders 1 to 6), the saddle-point value against the lgamma form, the
    expected information psi'(mu/2)/4 and its derivatives, and the mixed y-derivatives.

Prints one line per check and exits with status 1 when a check fails.
"""
import sys

import sympy as sp

MAXK = 6
mu, y, u, q = sp.symbols("mu y u q", positive=True)
fails = []


def check(name, a, b, points):
    """a and b agree at every point of the list (dict of substitutions)."""
    ok = True
    for pt in points:
        d = sp.nsimplify(sp.simplify((a - b).subs(pt)))
        if d != 0:
            ok = False
            break
    print(("ok   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def pts_mu_y(n=3):
    vals = [(sp.Rational(3, 2), sp.Rational(7, 5)), (sp.Rational(1, 3), sp.Rational(2, 9)), (sp.Rational(5), sp.Rational(11, 2))]
    return [{mu: m, y: v} for m, v in vals[:n]]


def fact(n):
    return sp.factorial(n)


# ---------------------------------------------------------------- exponential
l_exp = -sp.log(mu) - y / mu
for K in range(1, MAXK + 1):
    closed = (-1) ** K * fact(K - 1) / mu ** K * (1 - K * y / mu)
    check(f"exponential logpdf_dk<{K}>", sp.diff(l_exp, mu, K), closed, pts_mu_y())
for J in range(1, MAXK + 1):
    for K in range(0, MAXK + 1 - J):
        if J == 1:
            closed = -(-1) ** K * fact(K) / mu ** (K + 1)
            expr = sp.diff(sp.diff(l_exp, y), mu, K)
            check(f"exponential logpdf_dy_dk<1,{K}>", expr, closed, pts_mu_y())
check("exponential d^2/dy^2 log f = 0", sp.diff(l_exp, y, 2), 0, pts_mu_y())

# expected information: -E[l_mumu] over the density exp(-y/mu)/mu
f_exp = sp.exp(-y / mu) / mu
Ellmm = sp.integrate(sp.diff(l_exp, mu, 2) * f_exp, (y, 0, sp.oo))
Ellmm = sp.simplify(Ellmm)
check("exponential I = -E[l_mumu] = 1/mu^2", -Ellmm, 1 / mu ** 2, pts_mu_y())
for K in range(1, 5):
    closed = (-1) ** K * fact(K + 1) / mu ** (K + 2)
    check(f"exponential I^(K), K={K}", sp.diff(1 / mu ** 2, mu, K), closed, pts_mu_y())

# log(1 - F) = -y/mu and its mu-derivatives
for K in range(1, MAXK + 1):
    closed = (-1) ** (K + 1) * fact(K) * (y / mu) / mu ** K
    check(f"exponential log(1-F) derivative {K}", sp.diff(-y / mu, mu, K), closed, pts_mu_y())

# G(u) = log(1 - e^-u): derivatives against the Eulerian form in q = e^-u
G = sp.log(1 - sp.exp(-u))
A = {
    0: sp.Integer(1),
    1: sp.Integer(1),
    2: 1 + q,
    3: 1 + 4 * q + q ** 2,
    4: 1 + 11 * q + 11 * q ** 2 + q ** 3,
    5: 1 + 26 * q + 66 * q ** 2 + 26 * q ** 3 + q ** 4,
}
u_pts = [{u: sp.Rational(3, 2)}, {u: sp.Rational(7, 2)}, {u: sp.Integer(1)}]
for k in range(1, MAXK + 1):
    closed = (-1) ** (k + 1) * q * A[k - 1] / (1 - q) ** k
    closed = closed.subs(q, sp.exp(-u))
    check(f"G^({k}) Eulerian form (u >= 1)", sp.diff(G, u, k), closed, u_pts)

# the pole of G^(k) at u = 0 is that of log u (the rest is L = log((1-e^-u)/u))
L = sp.log((1 - sp.exp(-u)) / u)
for k in range(1, MAXK + 1):
    pole = (-1) ** (k - 1) * fact(k - 1) / u ** k
    check(f"G^({k}) = pole + L^({k})", sp.diff(G, u, k), pole + sp.diff(L, u, k),
          [{u: sp.Rational(1, 3)}, {u: sp.Rational(4, 5)}])

# ---------------------------------------------------------------- chisq
a = sp.symbols("a", positive=True)
ch_mu = mu  # the degrees of freedom
lgam = sp.loggamma(mu / 2)
l_chi = (mu / 2 - 1) * sp.log(y) - y / 2 - lgam - (mu / 2) * sp.log(2)
for K in range(1, MAXK + 1):
    if K == 1:
        closed = (sp.log(y / 2) - sp.polygamma(0, mu / 2)) / 2
    else:
        closed = -sp.polygamma(K - 1, mu / 2) / 2 ** K
    check(f"chisq logpdf_dk<{K}>", sp.diff(l_chi, mu, K), closed, [{mu: m, y: v} for m, v in
                                                              [(sp.Rational(3, 2), sp.Rational(7, 5)),
                                                               (sp.Rational(1, 3), sp.Rational(2, 9)),
                                                               (sp.Rational(5), sp.Rational(11, 2))]])
for K in range(1, 5):
    closed = sp.polygamma(K + 1, mu / 2) / 2 ** (K + 2)
    check(f"chisq I^(K), K={K}", sp.diff(sp.polygamma(1, mu / 2) / 4, mu, K), closed,
          [{mu: sp.Rational(3, 2)}, {mu: sp.Integer(5)}])

# the saddle-point value: -a D(y/mu) + log(a)/2 - delta(a) - log y - log(2 pi)/2 with a = mu/2
# delta(a) = lgamma(a) - (a - 1/2) log a + a - log(2 pi)/2 (Stirling's error)
a_ = mu / 2
delta = sp.loggamma(a_) - (a_ - sp.Rational(1, 2)) * sp.log(a_) + a_ - sp.log(2 * sp.pi) / 2
D = (y / mu) - 1 - sp.log(y / mu)
saddle = -a_ * D + sp.log(a_) / 2 - delta - sp.log(y) - sp.log(2 * sp.pi) / 2
check("chisq saddle-point value equals the log-density", saddle, l_chi, pts_mu_y())

# mixed y-derivatives
for J in range(1, MAXK + 1):
    for K in range(0, MAXK + 1 - J):
        if K > 1:
            closed = 0
        else:
            c = (-1) ** (J - 1) * fact(J - 1) / y ** J
            if K == 0:
                closed = (mu / 2 - 1) * c - (sp.Rational(1, 2) if J == 1 else 0)
            else:
                closed = c / 2
        expr = sp.diff(sp.diff(l_chi, y, J), mu, K)
        check(f"chisq logpdf_dy_dk<{J},{K}>", expr, closed, [{mu: sp.Rational(3, 2), y: sp.Rational(7, 5)},
                                                              {mu: sp.Integer(5), y: sp.Rational(11, 2)}])

if fails:
    print("FAILED:", ", ".join(fails))
    sys.exit(1)
print("all forms agree")
