"""Closed forms used by the coupled bijectors (include/laplace/bijector/), derived
and checked with SymPy.

Usage (from the repository root):  python -I tools/reference/bijector_coupled_forms.py

1. The derivatives of sech u to order 6, written as sech(u) P_K(T) with
   T = tanh u, and again with T^2 replaced by 1 - S, S = sech^2 u = 1 - T^2, so
   that the code can use the form that does not cancel (T^2 < 1/2: the T form;
   beyond: the S form, as tanh_dk in scalar_helpers.hpp). Each form is checked
   against SymPy's own derivative at random points.

2. The inverse log-Jacobian of partial_corr, -sum_(a>b) (p - b) log cosh u_ab
   (0-based b), against log |det| of the Jacobian of u -> strict lower
   triangle of R computed symbolically for p = 2, 3, 4.

3. The inverse log-Jacobian of log_cholesky, p log 2 + sum_i (p - i + 1) u_ii
   (0-based i), against the symbolic Jacobian for p = 2, 3.

4. The inverse log-Jacobian of alr, sum_(a=1..K) log pi_a, against the
   symbolic Jacobian in the coordinates pi_1..pi_(K-1) for K = 2, 3, 4.
"""
import random

import sympy as sp

u = sp.Symbol("u", real=True)
T, S = sp.symbols("T S", real=True)


def sech_forms(kmax=6):
    # (sech P(T))' = sech (-T P + (1 - T^2) P'(T)), since sech' = -sech T and T' = 1 - T^2
    print("sech^(K)(u) = sech(u) P_K, P_K in T = tanh u; then with T^2 = 1 - S")
    p = sp.Integer(1)
    for k in range(1, kmax + 1):
        p = sp.expand(-T * p + (1 - T ** 2) * sp.diff(p, T))
        d = sp.diff(sp.sech(u), u, k)
        # put the odd orders as T times a polynomial in T^2
        odd = k % 2 == 1
        q = sp.expand(sp.cancel(p / T)) if odd else p
        qs = sp.expand(q.subs(T ** 2, 1 - S))
        print(f"  K={k}: P = {'T*(' if odd else '('}{q}) = {'T*(' if odd else '('}{qs})")
        for _ in range(5):
            x = sp.Float(random.uniform(-4, 4), 50)
            exact = d.subs(u, x).evalf(40)
            x = x.evalf(50)
            t = sp.tanh(x).evalf(40)
            form1 = (sp.sech(x) * (t if odd else 1) * q.subs(T, t)).evalf(40)
            form2 = (sp.sech(x) * (t if odd else 1) * qs.subs(S, 1 - t ** 2)).evalf(40)
            assert abs(exact - form1) < 1e-30 * (1 + abs(exact)), (k, x)
            assert abs(exact - form2) < 1e-30 * (1 + abs(exact)), (k, x)
    print("  checked against SymPy's derivative at 5 random points per order")


def jac_logdet(outs, vs):
    J = sp.Matrix([[sp.diff(o, v) for v in vs] for o in outs])
    return sp.log(sp.Abs(J.det()))


def check_partial_corr():
    for p in (2, 3, 4):
        us = {(a, b): sp.Symbol(f"u{a}{b}", real=True) for a in range(p) for b in range(a)}
        L = sp.zeros(p, p)
        for a in range(p):
            pr = sp.Integer(1)
            for b in range(a):
                L[a, b] = sp.tanh(us[a, b]) * pr
                pr *= sp.sech(us[a, b])
            L[a, a] = pr
        R = L * L.T
        keys = sorted(us)
        lj = jac_logdet([R[a, b] for a, b in keys], [us[k] for k in keys])
        closed = -sum((p - b) * sp.log(sp.cosh(us[a, b])) for a, b in keys)
        for _ in range(3):
            pt = {us[k]: sp.Float(random.uniform(-2, 2), 50) for k in keys}
            d = (lj.subs(pt) - closed.subs(pt)).evalf(30)
            assert abs(d) < 1e-20, (p, d)
            diag = max(abs((R[i, i] - 1).subs(pt).evalf(30)) for i in range(p))
            assert diag < 1e-20
    print("partial_corr: inverse log-Jacobian -sum (p - b) log cosh u_ab checked, p = 2, 3, 4")


def check_log_cholesky():
    for p in (2, 3):
        us = {(i, j): sp.Symbol(f"u{i}{j}", real=True) for i in range(p) for j in range(i + 1)}
        L = sp.zeros(p, p)
        for (i, j), s in us.items():
            L[i, j] = sp.exp(s) if i == j else s
        Sg = L * L.T
        keys = sorted(us)
        lj = jac_logdet([Sg[i, j] for i, j in keys], [us[k] for k in keys])
        closed = p * sp.log(2) + sum((p - i + 1) * us[i, i] for i in range(p))
        for _ in range(3):
            pt = {us[k]: sp.Float(random.uniform(-2, 2), 50) for k in keys}
            assert abs((lj.subs(pt) - closed.subs(pt)).evalf(30)) < 1e-20
    print("log_cholesky: inverse log-Jacobian p log 2 + sum (p - i + 1) u_ii checked, p = 2, 3")


def check_alr():
    for K in (2, 3, 4):
        us = sp.symbols(f"u0:{K - 1}", real=True)
        D = 1 + sum(sp.exp(v) for v in us)
        pis = [sp.exp(v) / D for v in us] + [1 / D]
        lj = jac_logdet(pis[:-1], us)
        closed = sum(sp.log(x) for x in pis)
        for _ in range(3):
            pt = {v: sp.Float(random.uniform(-3, 3), 50) for v in us}
            assert abs((lj.subs(pt) - closed.subs(pt)).evalf(30)) < 1e-20
    print("alr: inverse log-Jacobian sum log pi_a checked, K = 2, 3, 4")


if __name__ == "__main__":
    random.seed(20261112)
    print("SymPy", sp.__version__)
    sech_forms()
    check_partial_corr()
    check_log_cholesky()
    check_alr()
