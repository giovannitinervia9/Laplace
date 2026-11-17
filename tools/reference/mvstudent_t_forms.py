"""The closed forms of the multivariate t (distrib/detail/mvstudent_t_common.hpp),
checked against SymPy's own derivatives, and its expected information checked
against a quadrature over the law.

Usage (from the repository root):  python -I tools/reference/mvstudent_t_forms.py

Notation: rho the dimension (the rank in the degenerate class), nu the degrees of
freedom, b = nu/2, h = rho/2, delta = r' W r the quadratic form, and
    l = c_rho(nu) + s log|M| + g(delta, nu),
    c_rho(nu) = lgamma(b + h) - lgamma(b) - h log(2 b pi),
    g(delta, nu) = -(nu + rho)/2 log(1 + delta/nu).

Checked here:
1. G(n, c) = d^n/d delta^n d^c/d nu^c g for n >= 1:
       -1/2 (-1)^(n-1) (n-1)! (nu + rho) (nu + delta)^-n                       (c = 0)
       -1/2 (-1)^(n-1) (n-1)! f_(c-1) (nu + delta)^-(n+c)
                 [(1 - n) nu - (n + c - 1) rho + c delta]                          (c >= 1)
   with f_k = (-1)^k n (n + 1) ... (n + k - 1).
2. G(0, c), c >= 1: student_t1's pure nu derivative of -(nu + 1)/2 log1p(delta/nu)
   (checked in distrib_student_t.py) plus -(rho - 1)/2 d^c log1p(delta/nu), with
       d^c log1p(delta/nu) = (-1)^(c-1) (c-1)! nu^-c expm1(-c log1p(delta/nu)).
3. c_rho by integer shifts, m = floor(rho/2), f(x) = lgamma(x + 1/2) - lgamma(x) - 1/2 log x:
       rho even: sum_(j=1..m-1) log1p(j/b) - m log 2 pi,
       rho odd:  sum_(j=1..m-1) log1p(j/b) + f(b + m) + 1/2 log1p(m/b) - (m + 1/2) log 2 pi,
   and d^k log1p(j/b)/db^k = (-1)^(k-1) (k-1)! b^-k expm1(-k log1p(j/b)).
4. I_nunu(nu, rho) = 1/4 [psi'(b) - psi'(b + h)] - rho (nu + rho + 4)/(2 nu (nu + rho)(nu + rho + 2))
   by the recurrence in rho (each term positive):
       I(nu, rho) - I(nu, rho - 2) = (h + 1)/4 P1 + (h - 1)^2/4 P1/b,
       P1 = 1/((b + h)(b + h - 1)^2 (b + h + 1)),   I(nu, 2) = 1/(2 b^2 (b + 1)(b + 2)).
5. The expected information (Lange, Little and Taylor 1989), kappa = (nu + rho)/(nu + rho + 2),
   X_a = Sigma^-1 d_a Sigma, T_ab = tr(X_a X_b), t_a = tr X_a:
       I_mumu = kappa Sigma^-1, I_mu,rest = 0,
       I_ab = [(nu + rho) T_ab - t_a t_b] / (2 (nu + rho + 2))
            = kappa Ttilde_ab / 2 + nu t_a t_b / (2 rho (nu + rho + 2)),
       Ttilde_ab = tr(Xtilde_a Xtilde_b), Xtilde = X - tr(X)/rho I,
       I_a,nu = -t_a / ((nu + rho)(nu + rho + 2)),   I_nunu as in 4,
   checked as E[d_i l d_j l] by quadrature over the law, in the proper class (d = 2,
   Sigma = diag(e^(2 t_1), e^(2 t_2)) rotated) and in the degenerate class (d = 3,
   Sigma = U diag(e^(2 t_1), e^(2 t_2)) U' with a fixed 3 x 2 orthonormal U, rho = 2,
   the law on mu + col(U)): the information on the range is that of the proper t of
   dimension rho.
"""
import sympy as sp
import mpmath as mp

nu, delta, rho, b, h, x = sp.symbols("nu delta rho b h x", positive=True)


def check(name, expr):
    e = sp.simplify(expr)
    if e != 0:
        raise SystemExit(f"{name}: residual {e}")
    print(f"ok  {name}")


def ffact(n, k):  # f_k = (-1)^k n (n + 1) ... (n + k - 1)
    return (-1) ** k * sp.rf(n, k)


g = -(nu + rho) / 2 * sp.log(1 + delta / nu)

# 1
for n in range(1, 7):
    for c in range(0, 7 - n):
        lhs = sp.diff(g, delta, n, nu, c) if c > 0 else sp.diff(g, delta, n)
        pre = -sp.Rational(1, 2) * (-1) ** (n - 1) * sp.factorial(n - 1)
        if c == 0:
            rhs = pre * (nu + rho) * (nu + delta) ** -n
        else:
            rhs = pre * ffact(n, c - 1) * (nu + delta) ** (-n - c) * ((1 - n) * nu - (n + c - 1) * rho + c * delta)
        check(f"G({n},{c})", lhs - rhs)

# 2
L = sp.log(1 + delta / nu)
for c in range(1, 7):
    rhs = (-1) ** (c - 1) * sp.factorial(c - 1) * nu ** -c * ((1 + delta / nu) ** -c - 1)
    check(f"d^{c} log1p(delta/nu)", sp.diff(L, nu, c) - rhs)
    split = sp.diff(-(nu + 1) / 2 * L, nu, c) - (rho - 1) / 2 * rhs
    check(f"G(0,{c}) split", sp.diff(g, nu, c) - split)

# 3 (numerically at 50 digits: lgamma identities over integer shifts)
mp.mp.dps = 50
for r in range(1, 12):
    for nv in (mp.mpf("0.3"), mp.mpf(7), mp.mpf("1234.5")):
        bb, m = nv / 2, r // 2
        direct = mp.loggamma(bb + mp.mpf(r) / 2) - mp.loggamma(bb) - mp.mpf(r) / 2 * mp.log(2 * bb * mp.pi)
        s = mp.fsum(mp.log1p(mp.mpf(j) / bb) for j in range(1, m))
        if r % 2 == 0:
            s -= m * mp.log(2 * mp.pi)
        else:
            f = mp.loggamma(bb + m + mp.mpf(1) / 2) - mp.loggamma(bb + m) - mp.log(bb + m) / 2
            s += f + mp.log1p(m / bb) / 2 - (m + mp.mpf(1) / 2) * mp.log(2 * mp.pi)
        if abs(s - direct) > mp.mpf(10) ** -45:
            raise SystemExit(f"c_rho shift: rho {r}, nu {nv}: {s - direct}")
print("ok  c_rho by integer shifts, rho = 1..11")
jj = sp.symbols("j", positive=True)
for k in range(1, 7):
    rhs = (-1) ** (k - 1) * sp.factorial(k - 1) * b ** -k * ((1 + jj / b) ** -k - 1)
    check(f"d^{k} log1p(j/b)", sp.diff(sp.log(1 + jj / b), b, k) - rhs)

# 4
def R(hh):
    n_, d_ = 2 * b, 2 * hh
    return d_ * (n_ + d_ + 4) / (2 * n_ * (n_ + d_) * (n_ + d_ + 2))


P1 = 1 / ((b + h) * (b + h - 1) ** 2 * (b + h + 1))
delta_I = sp.Rational(1, 4) / (b + h - 1) ** 2 - (R(h) - R(h - 1))  # the polygamma part telescopes
check("I_nunu recurrence", delta_I - ((h + 1) / 4 * P1 + (h - 1) ** 2 / 4 * P1 / b))
check("I_nunu(nu, 2)", sp.Rational(1, 4) / b**2 - R(1) - 1 / (2 * b**2 * (b + 1) * (b + 2)))
mp.mp.dps = 80  # the polygamma form cancels by nu^2 at nu = 1e6
for r in range(1, 9):  # the recurrence against the polygamma form, numerically
    for nv in (mp.mpf("0.4"), mp.mpf(9), mp.mpf(1e6)):
        bb = nv / 2
        exact = (mp.psi(1, bb) - mp.psi(1, bb + mp.mpf(r) / 2)) / 4 - r * (nv + r + 4) / (2 * nv * (nv + r) * (nv + r + 2))
        if r % 2:
            acc = (mp.psi(1, bb) - mp.psi(1, bb + mp.mpf(1) / 2)) / 4 - (nv + 5) / (2 * nv * (nv + 1) * (nv + 3))
            hs = [mp.mpf(k) + mp.mpf(1) / 2 for k in range(1, (r - 1) // 2 + 1)]
        else:
            acc = 1 / (2 * bb**2 * (bb + 1) * (bb + 2))
            hs = [mp.mpf(k) for k in range(2, r // 2 + 1)]
        for hh in hs:
            p1 = 1 / ((bb + hh) * (bb + hh - 1) ** 2 * (bb + hh + 1))
            acc += (hh + 1) / 4 * p1 + (hh - 1) ** 2 / 4 * p1 / bb
        if abs(acc - exact) > mp.mpf(10) ** -40 * abs(exact):
            raise SystemExit(f"I_nunu recurrence: rho {r}, nu {nv}: {acc - exact}")
print("ok  I_nunu recurrence against the polygamma form, rho = 1..8")

# 5: the information against E[score score'] by quadrature
mp.mp.dps = 20


def quad_info(t1, t2, nv, U, rho_):
    """E[d_i l d_j l] for (t1, t2, nu), l the density of y = U L x on the range
    (L = diag(e^t1, e^t2)), x the standard rho-dimensional t; integrated over
    x = R (cos phi, sin phi), R^2/rho ~ F(rho, nu)."""
    d = len(U)

    def score(y, th):
        # mpmath.diff of the log-density at the point y of R^d, coordinates on the range
        def ll(a, bq, n_):
            s1, s2 = mp.exp(2 * a), mp.exp(2 * bq)
            g1 = mp.fsum(U[k][0] * y[k] for k in range(d))
            g2 = mp.fsum(U[k][1] * y[k] for k in range(d))
            dl = g1**2 / s1 + g2**2 / s2
            return (mp.loggamma((n_ + rho_) / 2) - mp.loggamma(n_ / 2) - rho_ / 2 * mp.log(n_ * mp.pi)
                    - mp.log(s1 * s2) / 2 - (n_ + rho_) / 2 * mp.log1p(dl / n_))
        return [mp.diff(ll, th, tuple(1 if k == i else 0 for k in range(3))) for i in range(3)]

    th = (mp.mpf(t1), mp.mpf(t2), mp.mpf(nv))
    e1, e2 = mp.exp(t1), mp.exp(t2)

    def dens_R(R):  # density of R = |x|, x ~ t_2(0, I, nu)
        return 2 * mp.pi * R * mp.gamma((nv + 2) / 2) / (mp.gamma(nv / 2) * nv * mp.pi) * (1 + R**2 / nv) ** (-(nv + 2) / 2)

    out = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            def f(R, phi):
                x1, x2 = R * mp.cos(phi) * e1, R * mp.sin(phi) * e2
                y = [U[k][0] * x1 + U[k][1] * x2 for k in range(d)]
                s = score(y, th)
                return s[i] * s[j] * dens_R(R) / (2 * mp.pi)
            out[i][j] = mp.quad(f, [0, 1, 4, mp.inf], [0, mp.pi / 2, mp.pi, 3 * mp.pi / 2, 2 * mp.pi])
    return out


def closed_info(nv, rho_):
    # X_1 = diag(2, 0), X_2 = diag(0, 2) on the range: T = diag(4, 4), t = (2, 2)
    T = [[4, 0], [0, 4]]
    t = [2, 2]
    out = [[0] * 3 for _ in range(3)]
    for i in range(2):
        for j in range(i + 1):
            out[i][j] = ((nv + rho_) * T[i][j] - t[i] * t[j]) / (2 * (nv + rho_ + 2))
        out[2][i] = -mp.mpf(t[i]) / ((nv + rho_) * (nv + rho_ + 2))
    bb = mp.mpf(nv) / 2
    out[2][2] = (mp.psi(1, bb) - mp.psi(1, bb + mp.mpf(rho_) / 2)) / 4 - rho_ * (nv + rho_ + 4) / (
        2 * nv * (nv + rho_) * (nv + rho_ + 2))
    return out


s2 = 1 / mp.sqrt(2)
for U, label in (([[1, 0], [0, 1]], "proper d = 2"),
                 ([[s2, 0], [-s2, 0], [0, 1]], "degenerate d = 3, rho = 2")):
    nv = mp.mpf("4.5")
    q = quad_info(0.3, -0.4, nv, U, 2)
    c = closed_info(nv, 2)
    worst = max(abs(q[i][j] - c[i][j]) for i in range(3) for j in range(i + 1))
    if worst > mp.mpf(10) ** -10:
        raise SystemExit(f"information, {label}: {worst}")
    print(f"ok  information by quadrature, {label} (worst {mp.nstr(worst, 3)})")

# the traceless split of I_ab
Ta, Tb, Tab, ta, tb = sp.symbols("Ta Tb Tab ta tb")
check("traceless split", ((nu + rho) * Tab - ta * tb) / (2 * (nu + rho + 2))
      - ((nu + rho) / (nu + rho + 2) * (Tab - ta * tb / rho) / 2 + nu * ta * tb / (2 * rho * (nu + rho + 2))))
