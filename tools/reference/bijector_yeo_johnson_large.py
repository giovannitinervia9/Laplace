"""Reference values for the parameter-derivatives of the inverse of Yeo-Johnson, and of its
log-Jacobian, at a large finite argument, where the generated forms lose the value and the
rescaled route in scalar_yeo_johnson.hpp supplies it instead.

They come from the definition of the inverse, differentiated SYMBOLICALLY by SymPy and then
evaluated at 60 digits by mpmath, so the route to them shares nothing with the C++ beyond the
definition itself:

    h(u; lambda) = exp(log(1 + lambda u)/lambda) - 1                          for u >= 0
                 = 1 - exp(log(1 + (2 - lambda)(-u))/(2 - lambda))            for u < 0

Numerical differentiation is not usable here: a finite difference in u at u = 1e60 loses every
digit of the result, and mpmath's own differentiator returned entries wrong by two hundred orders
of magnitude, some of them exactly zero, when this was first written that way on 2026-11-17.

The printed numbers are the ones written into tests/bijector/test_yeo_johnson.cpp, at seventeen
significant digits, with the ones no double can hold printed as infinities. Run with

    python -I tools/reference/bijector_yeo_johnson_large.py
"""
import mpmath as mp
import sympy as sp

mp.mp.dps = 60

DBL_MAX = mp.mpf("1.7976931348623157e308")
KMAX = 6

u, lam = sp.symbols("u lam", positive=True)

# (lambda, x): a pair on each branch at 1e60, one at 1e300, and one whose value overflows
CASES = [("0.5", "1e60"), ("0.5", "-1e60"), ("1.3", "1e300"), ("0.3", "-1e60")]

# The two quantities on each branch, in the variable u of that branch's own formula: on the
# negative branch u stands for -x, and a derivative in x then carries (-1)^n0.
POS = {"h": sp.exp(sp.log(1 + lam * u) / lam) - 1,
       "lj": (1 - lam) * sp.log(1 + lam * u) / lam}
NEG = {"h": 1 - sp.exp(sp.log(1 + (2 - lam) * u) / (2 - lam)),
       "lj": (1 - (2 - lam)) * sp.log(1 + (2 - lam) * u) / (2 - lam)}


def cxx(v):
    if abs(v) > DBL_MAX:
        return "inf" if v > 0 else "-inf"
    return mp.nstr(v, 17, strip_zeros=False)


def pure_lambda(which, lams, xs):
    """d^K_lambda of the quantity, K = 1..KMAX: the only entries the rescaled route supplies.

    Every entry that also carries a derivative in the argument is smaller by a factor
    (1 + lambda u)^(-1) per order, so at these arguments it is below the smallest double and
    underflows to zero; there is nothing there to check against."""
    x = mp.mpf(xs)
    expr = (POS if x >= 0 else NEG)[which]
    uv = sp.Rational(xs if x >= 0 else xs.lstrip("-"))
    lv = sp.Rational(lams)
    return [mp.mpf(str(sp.N(sp.diff(expr, lam, K).subs({u: uv, lam: lv}), 60)))
            for K in range(1, KMAX + 1)]


for which, name in (("h", "the inverse"), ("lj", "its log-Jacobian")):
    print(f"// {name}: d^K_p1, K = 1..6")
    for lams, xs in CASES:
        print(f"    // p1 = {lams}, x = {xs}")
        print("    {" + ", ".join(cxx(v) for v in pure_lambda(which, lams, xs)) + "},")
    print()


# ---------------------------------------------------------------------------------------------
# The band on the other side of the forms' loss: lambda so small that lambda u stays small even
# at a huge u. There the log-Jacobian's partials are the ones at stake, and at lambda = 0 exactly
# every entry has a closed form, so the whole column is printed rather than the pure one.
BAND_X = ["1e103", "1e150", "1e300"]


def band_column(K, xs):
    """d^(K-n1)_x d^n1_lambda log|h'| at lambda = 0, n1 = 0..K, on the positive branch"""
    x = sp.Rational(xs)
    expr = POS["lj"]
    vals = []
    for n1 in range(K + 1):
        n0 = K - n1
        e = sp.diff(expr, u, n0) if n0 else expr
        if n1:
            e = sp.diff(e, lam, n1)
        v = sp.limit(e.subs(u, x), lam, 0)
        if v.is_infinite:
            vals.append("inf" if v > 0 else "-inf")
            continue
        vals.append(cxx(mp.mpf(str(sp.N(v, 60)))))
    return vals


# flattened in the layout the test uses: the orders K = 1..6 one after another, the entries of
# order K at yj_part_offset(K) + n1, 27 in all
print("// the log-Jacobian at lambda = 0 in the band, flat in the test's own layout:")
print("// the orders 1 to 6 one after another, entry n1 of order K at (K-1)(K+2)/2 + n1")
for xs in BAND_X:
    flat = []
    for K in range(1, KMAX + 1):
        flat += band_column(K, xs)
    print(f"    // x = {xs}")
    for start in range(0, len(flat), 5):
        print("    " + ", ".join(flat[start:start + 5]) + ("," if start + 5 < len(flat) else "},"))
