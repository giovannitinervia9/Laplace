"""Reference values for the parameter-derivatives of the inverse of Box-Cox at a large finite
argument, where the generated forms lose the value and the rescaled route supplies it instead.

They come from the definition of the inverse, differentiated SYMBOLICALLY by SymPy and only then
evaluated at 60 digits by mpmath, so the route to them shares nothing with the C++ beyond the
definition itself:

    h(u; lambda) = (1 + lambda u)^(1/lambda),     1 + lambda u > 0.

Numerical differentiation is not usable here, and the reason is measured rather than assumed: the
yeo_johnson session found mpmath's own differentiator returning entries wrong by two hundred
orders of magnitude at u = 1e60, several of them exactly zero, when its own script was first
written that way on 2026-11-17.

Where the forms fail: above |lambda u| = 1e3 the inverse partials of scalar_box_cox.hpp take
yj_inv_partials_rescaled_pos, which keeps q_j = (lambda u)^(j+1) psi^(j)(lambda u) bounded where
psi^(j) underflows and the power overflows. That routine is the shared positive branch of
Yeo-Johnson and is already Box-Cox's column: its one Yeo-Johnson line, the -1 of h = exp(S) - 1,
is guarded by n0 == 0 && n1 == 0, which needs K = 0 and so never fires for a partial.

The printed numbers are the ones written into tests/bijector/test_box_cox.cpp, at seventeen
significant digits. Run with

    python -I tools/reference/bijector_box_cox_large.py
"""
import mpmath as mp
import sympy as sp

mp.mp.dps = 60

u, lam = sp.symbols("u lam", real=True)
H = (1 + lam * u) ** (1 / lam)

# (lambda, u): every one has 1 + lambda u > 0 and |lambda u| above the threshold 1e3, and covers
# a positive lambda, a negative one with a negative u, a small lambda and a large one
CASES = [
    (sp.Rational(1, 2), sp.Integer(10) ** 60),
    (-sp.Rational(3, 2), -sp.Integer(10) ** 60),
    (sp.Rational(1, 6), sp.Integer(10) ** 100),
    (sp.Integer(3), sp.Integer(10) ** 300),
]
ORDERS = [1, 3, 6]  # the full column of each


def main():
    print("// lambda, u, K, n1 = 0..K: d^(K-n1)_u d^n1_lambda h, SymPy-differentiated, mpmath at 60 digits")
    for l, uv in CASES:
        for K in ORDERS:
            vals = []
            for n1 in range(0, K + 1):
                n0 = K - n1
                e = sp.diff(H, u, n0) if n0 else H
                e = sp.diff(e, lam, n1) if n1 else e
                f = sp.lambdify(u, e.subs(lam, l), "mpmath")
                vals.append(mp.nstr(f(mp.mpf(uv)), 17))
            print(f"// lambda = {l}, u = 1e{sp.log(abs(uv), 10)}, K = {K}")
            print("    " + ", ".join(vals))


if __name__ == "__main__":
    main()
