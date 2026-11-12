"""Derivations behind the closed forms of the bijectors power and softplus
(include/laplace/bijector/detail/scalar_power.hpp, scalar_softplus.hpp), with SymPy.

Usage (from the repository root):  python -I tools/reference/bijector_powsoft_forms.py

Notation for softplus with sharpness a > 0. For the forward map the argument is
v > 0 and x = a v > 0; for the inverse it is u and x = a u.

  t(x) = 1/expm1(x), so that (1 + t) = 1/(1 - e^-x) is the derivative of Z;
  Z(x) = log(e^x - 1) = x + M(x), M(x) = log(1 - e^-x); forward g = Z(a v)/a = v + M(x)/a;
  S(x) = log(1 + e^x) = x + L(x), L(x) = log(1 + e^-x) for x > 0; inverse h = S(a u)/a;
  sigma(x) = expit(x), eps = sigma(-x) = 1 - sigma(x) for x > 0.

Checked here:

 1. t' = -(t + t^2), and the polynomials Q_k with t^(k) = Q_k(t), k = 1..5, have
    coefficients of one sign per k, so they do not cancel for t > 0.
 2. The x-linear part of the pure a-derivatives of Z(x)/a and S(x)/a cancels
    exactly, since Z' = 1 + t and S' = sigma: the coefficient of x is (-1)^m m!
    from the value and C(m, 1) (-1)^(m-1) (m-1)! from the first derivative, which add to 0.
 3. psi(z) = log(exprel(z)) = log(phi(z)), phi(z) = (e^z - 1)/z. With R_k = phi^(k)/phi,
    psi^(r) = D^(r-1) R_1 for the derivation D R_k = R_(k+1) - R_k R_1 (the rule
    used for lambda(x) = -psi(-x) = -log exprel(-x) at small x, the regular part
    of the forward log-Jacobian -log(1 - e^-x) = -log x + lambda(x)).
 4. The Leibniz formulas for mixed derivatives in (argument, parameter a):
      d^n0/dv^n0 d^n1/da^n1 [Phi(a v)/a] = a^(n0-n1-1) sum_j C(n1,j) ff(n0-1,j) x^(n1-j) Phi^(n0+n1-j)(x),
      d^n0/dv^n0 d^n1/da^n1 [Psi(a v)]   = a^(n0-n1)   sum_i C(n0,i) (n1!/(n1-i)!) x^(n1-i) Psi^(n0+n1-i)(x),
    with ff(m, j) = m (m-1) ... (m-j+1) and ff(-1, j) = (-1)^j j!. Checked against
    SymPy's derivatives of log(e^(a v) - 1)/a for n0 >= 1 (the header uses the
    pure a-derivatives, n0 = 0, in the form of point 2).
"""
import sympy as sp

x, a, v, z, t, R1, R2, R3, R4, R5, R6 = sp.symbols("x a v z t R1 R2 R3 R4 R5 R6", real=True)
Rs = [R1, R2, R3, R4, R5, R6]


def show(name, expr, var):
    p = sp.Poly(sp.expand(expr), var)
    terms = " + ".join(f"({c})*{var}^{k}" for (k,), c in sorted(p.terms()))
    # the caller passes the symbol; print it by name
    print(f"{name} = {terms}")


# 1. t^(k) = Q_k(t)
Q = {0: t}
for k in range(1, 6):
    Q[k] = sp.expand(sp.diff(Q[k - 1], t) * (-(t + t**2)))
for k in range(1, 6):
    show(f"Q_{k}", Q[k], t)
    assert all(c * (-1) ** k >= 0 for c in sp.Poly(Q[k], t).all_coeffs()), k
xv = sp.Rational(7, 10)
for k in range(1, 6):
    direct = sp.diff(1 / (sp.exp(x) - 1), x, k).subs(x, xv).evalf(30)
    via = Q[k].subs(t, 1 / (sp.exp(xv) - 1)).evalf(30)
    assert abs(direct - via) < 1e-20 * max(1, abs(direct)), k
print("check 1 passed: t^(k) = Q_k(t), k = 1..5")


# 3. psi^(r) = D^(r-1) R_1 with D R_k = R_(k+1) - R_k R_1
def D(P):
    out = 0
    for k in range(1, 6):
        out += sp.diff(P, Rs[k - 1]) * (Rs[k] - Rs[k - 1] * R1)
    return sp.expand(out)


psi = {1: R1}
for r in range(2, 7):
    psi[r] = D(psi[r - 1])
for r in range(1, 7):
    print(f"psi^({r}) =", psi[r])
zv = sp.Rational(-13, 10)
phi = (sp.exp(z) - 1) / z
subsR = {Rs[k - 1]: (sp.diff(phi, z, k) / phi).subs(z, zv).evalf(40) for k in range(1, 7)}
for r in range(1, 7):
    direct = sp.diff(sp.log(phi), z, r).subs(z, zv).evalf(40)
    via = psi[r].subs(subsR).evalf(40)
    assert abs(direct - via) < 1e-25 * max(1, abs(direct)), r
print("check 3 passed: psi^(r) from the R_k recursion, r = 1..6")

# 2. the x-linear cancellation of the pure a-derivatives of Z(x)/a and S(x)/a
for m in range(1, 7):
    coef_value = (-1) ** m * sp.factorial(m)
    coef_first = sp.binomial(m, 1) * (-1) ** (m - 1) * sp.factorial(m - 1)
    assert sp.simplify(coef_value + coef_first) == 0, m
print("check 2 passed: the x-linear coefficients of the pure a-derivatives cancel")

# 4. Leibniz form against SymPy, g(v; a) = log(e^(a v) - 1)/a, n0 >= 1
def ff(m, j):
    out = 1
    for i in range(j):
        out *= m - i
    return out


g = sp.log(sp.exp(a * v) - 1) / a
av, vv = sp.Rational(3, 2), sp.Rational(1, 3)
xval = av * vv
tv = 1 / (sp.exp(xval) - 1)
for n0 in range(1, 5):
    for n1 in range(0, 4):
        if n0 + n1 > 5:
            continue
        direct = sp.diff(g, v, n0, a, n1).subs({a: av, v: vv}).evalf(30)
        Zd = lambda r: (1 + tv) if r == 1 else Q[r - 1].subs(t, tv)
        s = sum(sp.binomial(n1, j) * ff(n0 - 1, j) * xval ** (n1 - j) * Zd(n0 + n1 - j)
                for j in range(0, min(n0 - 1, n1) + 1))
        via = (av ** (n0 - n1 - 1) * s).evalf(30)
        assert abs(direct - via) < 1e-20 * max(1, abs(direct)), (n0, n1, direct, via)
print("check 4 passed: Leibniz form of the mixed derivatives of g, n0 >= 1, n0 + n1 <= 5")


# 5. Taylor series of lambda(x) = -log exprel(-x) = -log((1 - e^-x)/x) at x = 0, used for
# 0 < x < 1/4 (the forms of point 3 cancel to an absolute error of order eps there, while
# lambda^(r)(x) = O(x) for some r). lambda^(r)(x) = sum_(n >= r) c_n n!/(n-r)! x^(n-r).
NSERIES = 16
phi_series = sum((-x) ** n / sp.factorial(n + 1) for n in range(NSERIES + 2))
lam_series = sp.series(-sp.log(phi_series), x, 0, NSERIES + 1).removeO()
coeffs = [sp.nsimplify(lam_series.coeff(x, n)) for n in range(NSERIES + 1)]
print("lambda series coefficients c_n, n = 0..%d:" % NSERIES)
print([str(c) for c in coeffs])
for r in range(1, 7):
    direct = sp.diff(-sp.log((1 - sp.exp(-x)) / x), x, r).subs(x, sp.Rational(1, 5)).evalf(30)
    via = sum(coeffs[n] * sp.ff(n, r) * sp.Rational(1, 5) ** (n - r) for n in range(r, NSERIES + 1)).evalf(30)
    assert abs(direct - via) < 1e-15 * max(1, abs(direct)), r
print("check 5 passed: the series of lambda^(r) at x = 1/5, r = 1..6")
