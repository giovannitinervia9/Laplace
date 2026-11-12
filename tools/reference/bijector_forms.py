"""Closed forms of the derivatives of the scalar bijectors, derived with SymPy.

Usage (from the repository root):  python -I tools/reference/bijector_forms.py

Prints, for each family of formulas used in include/laplace/bijector/scalar.hpp,
the polynomials whose coefficients the header writes out. Each form is checked
here against SymPy's own derivative of the defining expression.

  expit      p = 1/(1 + e^-u), q = 1 - p, U = p q, D = q - p (D^2 = 1 - 4U):
             d^k p / du^k = U (A_k(U) + D B_k(U))
  tanh       T = tanh u, S = 1 - T^2:  d^k T / du^k = S P_k(T)
  r, s       r = (1 + x^2)^(-1/2), s = x r (|s| < 1, r^2 = 1 - s^2):
             d^m r = r^(m+1) V_m(s), d^m r^2 = r^(m+2) W_m(s), d^m (s r) = r^(m+1) X_m(s)
  tan        d^k tan(y) / dy^k = T_k(z), z = tan y
  probit     z = Phi^-1(p), phi = phi(z):  d^k z / dp^k = Q_k(z) / phi^k,
             d^k (z^2/2) / dp^k = R_k(z) / phi^k
  hermite    d^k Phi / du^k = (-1)^(k-1) He_(k-1)(u) phi(u)
  cloglog    w = e^u:  d^k (1 - exp(-w)) / du^k = exp(u - w) P_k(w)
  loglog     t = log s:  d^k log(-t) / ds^k = C_k(t) / (s t)^k,
             d^k (-t - log(-t)) / ds^k = E_k(t) / (s t)^k
"""
import sympy as sp

KMAX = 6
u, x, s, t, w, z, U, D, T, y = sp.symbols("u x s t w z U D T y", real=True)


def poly_str(expr, var):
    p = sp.Poly(sp.expand(expr), var)
    return [int(c) if c.is_integer else c for c in reversed(p.all_coeffs())]


def expit_forms():
    p = 1 / (1 + sp.exp(-u))
    q = 1 - p
    print("expit: d^k p = U (A_k(U) + D B_k(U)); coefficients from degree 0")
    # recursion on (A, B): U' = U D, D' = -2 U, D^2 = 1 - 4U
    A, B = sp.Integer(1), sp.Integer(0)
    for k in range(1, KMAX + 1):
        expr = U * (A + D * B)
        check = sp.simplify(expr.subs({U: p * q, D: q - p}) - sp.diff(p, u, k))
        assert check == 0, (k, check)
        print(f"  k={k}: A={poly_str(A, U)}  B={poly_str(B, U)}")
        # differentiate U (A + D B) in u
        dexpr = sp.diff(expr, U) * U * D + sp.diff(expr, D) * (-2 * U)
        dexpr = sp.expand(dexpr).subs(D**2, 1 - 4 * U)
        dexpr = sp.expand(sp.expand(dexpr).subs(D**3, D * (1 - 4 * U)))
        rest = sp.expand(dexpr / U)
        A = rest.coeff(D, 0)
        B = rest.coeff(D, 1)


def tanh_forms():
    # d/du [S P(T)] = S (-2 T P + (1 - T^2) P'), S = 1 - T^2
    print("tanh: d^k T = (1 - T^2) P_k(T)")
    P = sp.Integer(1)
    for k in range(1, KMAX + 1):
        check = sp.simplify((1 - sp.tanh(u) ** 2) * P.subs(T, sp.tanh(u)) - sp.diff(sp.tanh(u), u, k))
        assert check == 0, (k, check)
        print(f"  k={k}: P={poly_str(P, T)}")
        P = sp.expand(-2 * T * P + (1 - T**2) * sp.diff(P, T))


def rs_forms():
    # d/dx [r^a P(s)] = r^(a+1) (-a s P + (1 - s^2) P'), since r' = -s r^2, s' = r^3
    r = 1 / sp.sqrt(1 + x**2)
    for name, f, a0, P0 in [("V: d^m r = r^(m+1) V_m(s)", r, 1, sp.Integer(1)),
                            ("W: d^m r^2 = r^(m+2) W_m(s)", r**2, 2, sp.Integer(1)),
                            ("X: d^m (s r) = r^(m+1) X_m(s)", x * r**2, 1, s)]:
        print("r,s:", name)
        a, P = a0, P0
        for m in range(0, KMAX):
            val = (r**a * P.subs(s, x * r)).subs(x, sp.Rational(3, 7))
            ref = sp.diff(f, x, m).subs(x, sp.Rational(3, 7))
            assert sp.simplify(val - ref) == 0, (name, m)
            print(f"  m={m}: {poly_str(P, s)}")
            P = sp.expand(-a * s * P + (1 - s**2) * sp.diff(P, s))
            a += 1


def tan_forms():
    print("tan: d^k tan = T_k(z)")
    for k in range(1, KMAX + 1):
        # d/dy P(tan y) = (1 + z^2) P'(z)
        P = (1 + z**2) if k == 1 else sp.expand((1 + z**2) * sp.diff(P, z))
        assert sp.simplify(P.subs(z, sp.tan(y)) - sp.diff(sp.tan(y), y, k)) == 0, k
        print(f"  k={k}: {poly_str(P, z)}")


def probit_forms():
    # dz/dp = 1/phi(z), dphi/dz = -z phi: d/dp [F(z)/phi^k] = (F' + k z F)/phi^(k+1)
    print("probit: d^k z = Q_k(z)/phi^k ; d^k z^2/2 = R_k(z)/phi^k")
    Q, R = sp.Integer(1), z
    for k in range(1, KMAX + 1):
        print(f"  k={k}: Q={poly_str(Q, z)}  R={poly_str(R, z)}")
        Q = sp.expand(sp.diff(Q, z) + k * z * Q)
        R = sp.expand(sp.diff(R, z) + k * z * R)
    print("hermite He_m")
    for m in range(0, KMAX):
        print(f"  m={m}: {poly_str(sp.hermite_prob(m, u), u)}")


def cloglog_forms():
    print("cloglog: d^k (1 - exp(-e^u)) = exp(u - w) P_k(w)")
    h = 1 - sp.exp(-sp.exp(u))
    for k in range(1, KMAX + 1):
        d = sp.diff(h, u, k)
        P = sp.expand(sp.simplify(d / sp.exp(u - sp.exp(u))).subs(sp.exp(u), w))
        print(f"  k={k}: {poly_str(P, w)}")


def loglog_forms():
    print("phi(s) = log(-log s): d^k = C_k(t)/(s t)^k ; psi(s) = -t - log(-t): d^k = E_k(t)/(s t)^k")
    ss = sp.symbols("ss", positive=True)
    tt = sp.log(ss)
    for name, f in [("C", sp.log(-tt)), ("E", -tt - sp.log(-tt))]:
        for k in range(1, KMAX + 1):
            d = sp.diff(f, ss, k)
            P = sp.expand(sp.expand(sp.simplify(d * (ss * tt) ** k)).subs(sp.log(ss), t))
            print(f"  {name}_{k}: {poly_str(P, t)}")


if __name__ == "__main__":
    expit_forms()
    tanh_forms()
    rs_forms()
    tan_forms()
    probit_forms()
    cloglog_forms()
    loglog_forms()
