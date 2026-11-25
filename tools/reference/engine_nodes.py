"""The derivatives of the pointwise nodes of include/laplace/engine/pointwise_node.hpp.

The logistic growth curve f = A / (1 + exp((xm - t) / s)) in its inputs (A, xm, s):
each closed form written in the header (in the variables z = (xm - t) / s,
p = 1 / (1 + e^z), q = p (1 - p), w = 1 - 2 p) is checked against SymPy's own
derivative of f. Orders three and four are written through f = A h(xm, s),
h = g(z), g = 1 / (1 + e^z), whose derivatives are g1 = -q, g2 = w q,
g3 = 2 q^2 - w^2 q, g4 = w^3 q - 8 w q^2; the derivatives of h in (xm, s) are
first derived here from a generic g (SymPy's chain rule) and printed, then the
closed forms of the header are checked against the derivatives of f itself.
Run with python -I; prints the forms and "ok".
"""

import sympy as sp

A, xm, s, t, Z = sp.symbols("A x_m s t z", real=True)
f = A / (1 + sp.exp((xm - t) / s))

z = (xm - t) / s
p = 1 / (1 + sp.exp(z))
q = p * (1 - p)
w = 1 - 2 * p
g1, g2, g3, g4 = -q, w * q, 2 * q**2 - w**2 * q, w**3 * q - 8 * w * q**2

# the derivatives of h = G((xm - t) / s) for a generic G, in z, s and G^(k)
G = sp.Function("G")
gs = sp.symbols("g1:5")
h_generic = G((xm - t) / s)
for k in range(2, 5):
    for nb in range(k + 1):
        d = sp.diff(h_generic, *([xm] * (k - nb) + [s] * nb))
        d = d.replace(lambda e: isinstance(e, sp.Subs), lambda e: e.doit())
        d = d.replace(lambda e: isinstance(e, sp.Derivative), lambda e: gs[sum(c for _, c in e.variable_count) - 1])
        d = sp.factor(sp.simplify(sp.expand(d.subs(xm, Z * s + t))) * s**k)
        print("h_" + "x" * (k - nb) + "s" * nb, "= (%s) / s^%d" % (d, k))

h = {  # the forms of the header
    (xm, xm): g2 / s**2,
    (xm, s): -(g1 + g2 * z) / s**2,
    (s, s): z * (2 * g1 + g2 * z) / s**2,
    (xm, xm, xm): g3 / s**3,
    (xm, xm, s): -(2 * g2 + g3 * z) / s**3,
    (xm, s, s): (2 * g1 + 4 * g2 * z + g3 * z**2) / s**3,
    (s, s, s): -z * (6 * g1 + 6 * g2 * z + g3 * z**2) / s**3,
    (xm, xm, xm, xm): g4 / s**4,
    (xm, xm, xm, s): -(3 * g3 + g4 * z) / s**4,
    (xm, xm, s, s): (6 * g2 + 6 * g3 * z + g4 * z**2) / s**4,
    (xm, s, s, s): -(6 * g1 + 18 * g2 * z + 9 * g3 * z**2 + g4 * z**3) / s**4,
    (s, s, s, s): z * (24 * g1 + 36 * g2 * z + 12 * g3 * z**2 + g4 * z**3) / s**4,
}

closed = {
    (A,): p,
    (xm,): -A * q / s,
    (s,): A * q * z / s,
    (A, A): sp.Integer(0),
    (A, xm): -q / s,
    (A, s): q * z / s,
    (xm, xm): A * q * w / s**2,
    (xm, s): A * q * (1 - w * z) / s**2,
    (s, s): A * q * z * (w * z - 2) / s**2,
}
for var, form in h.items():
    if len(var) >= 3:
        closed[var] = A * form  # no A: A times a derivative of h
    if len(var) <= 3:
        closed[(A,) + var] = form  # one A: a derivative of h of one order less
for k in (3, 4):
    for nA in range(2, k + 1):
        for nb in range(k - nA + 1):
            closed[(A,) * nA + (xm,) * (k - nA - nb) + (s,) * nb] = sp.Integer(0)

R_ = sp.Rational
rng = [(R_(13, 10), R_(7, 10), R_(21, 10), R_(2, 5)), (R_(-1, 2), 3, R_(3, 5), R_(19, 10)),
       (R_(11, 5), R_(-11, 10), R_(17, 10), R_(-4, 5)), (R_(9, 10), 5, R_(4, 5), -2)]
for var, form in closed.items():
    d = sp.diff(f, *var)
    for a_, x_, s_, t_ in rng:
        val = (d - form).subs({A: a_, xm: x_, s: s_, t: t_}).evalf(50)
        scale = 1 + abs(d.subs({A: a_, xm: x_, s: s_, t: t_}).evalf(50))
        assert abs(val) < 1e-40 * scale, (var, val)
    print(var, "=", form)
print("ok")
