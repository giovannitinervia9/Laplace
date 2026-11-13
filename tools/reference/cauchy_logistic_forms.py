"""Checks, with SymPy and mpmath, of the closed forms of cauchy_distrib and logistic_distrib
(include/laplace/distrib/cauchy.hpp, logistic.hpp).

Usage (from the repository root):  python -I tools/reference/cauchy_logistic_forms.py

Each recurrence and closed form that the headers use is compared with SymPy's own
derivative of the defining function, at points that cover the centre and both tails.
The script prints the largest relative difference of each check; a failure stops it.

Notation: z = (y - mu)/sigma, w = 1/(1 + z^2), theta(z) = atan2(1, z) = pi/2 - atan(z),
u_j(z) = Re((z + i)^(-j)) = (1 + z^2)^(-j/2) cos(j theta(z)), p(z) = 2 z w (the logistic
p is written P below to keep the two apart).
"""
import sympy as sp
import mpmath

mpmath.mp.dps = 50
z = sp.symbols("z", real=True)
POINTS = [-30.0, -7.5, -2.0, -0.7, -0.05, 0.0, 0.3, 1.0, 2.5, 6.0, 25.0]
JMAX = 6
worst = {}


def record(name, got, want):
    # a component that is exactly 0 (odd derivatives of -log(1 + z^2) at z = 0) is checked absolutely
    if want == 0:
        if abs(got) > 1e-40:
            raise SystemExit(f"{name}: got {got} where the exact value is 0")
        return
    err = abs(got - want) / abs(want)
    worst[name] = max(worst.get(name, 0.0), err)
    if err > 1e-13:
        raise SystemExit(f"{name}: relative difference {err:.3e} (got {got}, want {want})")


def num(expr, x):
    # sp.Rational(x) is the exact binary value of the double x (a Python float would be read at 53 bits)
    return mpmath.mpf(sp.N(expr.subs(z, sp.Rational(x)), 60))


def u_rec(x):
    """u_j by the recurrence u_0 = 1, u_1 = z w, u_(j+1) = w (2 z u_j - u_(j-1))."""
    w = 1 / (1 + x * x)
    u = [mpmath.mpf(1), x * w]
    for j in range(1, JMAX + 1):
        u.append(w * (2 * x * u[j] - u[j - 1]))
    return u  # u[0..JMAX+1]


def u_closed(x, j):
    """u_j = (1 + z^2)^(-j/2) cos(j theta), theta = atan2(1, z)."""
    th = mpmath.atan2(1, x)
    return (1 + x * x) ** (-mpmath.mpf(j) / 2) * mpmath.cos(j * th)


# --- Cauchy log f: Lambda_f = -log(1 + z^2), L_j = 2 (-1)^j (j-1)! u_j ----------------------
Lf = -sp.log(1 + z**2)
for x in POINTS:
    u = u_rec(mpmath.mpf(x))
    for j in range(1, JMAX + 1):
        want = num(sp.diff(Lf, z, j), x)
        got = 2 * (-1) ** j * mpmath.factorial(j - 1) * u[j]
        record("cauchy logf L_j (u recurrence)", float(got), float(want))
        record("cauchy logf L_j (closed u_j)", float(2 * (-1) ** j * mpmath.factorial(j - 1) * u_closed(mpmath.mpf(x), j)), float(want))

# --- Cauchy log(1 - F): Lambda = log theta (the -log pi is a constant), q = Lambda' = -w/theta,
# q^(j+1) = -sum_i C(j,i) (q^(i) + p^(i)) q^(j-i), p^(i) = 2 (-1)^i i! u_(i+1), L_j = q^(j-1) ---
theta_sym = sp.pi / 2 - sp.atan(z)  # equal to atan2(1, z) for all real z
Lup = sp.log(theta_sym)
for x in POINTS:
    xm = mpmath.mpf(x)
    u = u_rec(xm)
    w = 1 / (1 + xm * xm)
    th = mpmath.atan2(1, xm)
    q = [-w / th]
    p = [2 * (-1) ** i * mpmath.factorial(i) * u[i + 1] for i in range(JMAX)]
    for j in range(JMAX):
        s = 0
        for i in range(j + 1):
            s += mpmath.binomial(j, i) * (q[i] + p[i]) * q[j - i]
        q.append(-s)
    for j in range(1, JMAX + 1):
        want = num(sp.diff(Lup, z, j), x)
        record("cauchy log(1-F) L_j (Riccati)", float(q[j - 1]), float(want))

# p(z) = 2 z w = 1/(z + i) + 1/(z - i): its derivatives are 2 (-1)^i i! u_(i+1)
for x in POINTS:
    u = u_rec(mpmath.mpf(x))
    for i in range(JMAX):
        want = num(sp.diff(2 * z / (1 + z**2), z, i), x)
        record("cauchy p^(i) = 2 (-1)^i i! u_(i+1)", float(2 * (-1) ** i * mpmath.factorial(i) * u[i + 1]), float(want))

# --- Logistic: P = p(z) = 1/(1 + e^-z), s_0 = P, s_1 = P (1 - P),
# s_(k+1) = (1 - 2P) s_k - sum_(i=1)^(k-1) C(k,i) s_i s_(k-i) equals d^k P / dz^k -----------
P = 1 / (1 + sp.exp(-z))
for x in POINTS:
    xv = mpmath.mpf(x)
    Pv = 1 / (1 + mpmath.exp(-xv))
    s = [Pv, Pv * (1 - Pv)]
    for k in range(1, JMAX + 1):
        acc = (1 - 2 * Pv) * s[k]
        for i in range(1, k):
            acc -= mpmath.binomial(k, i) * s[i] * s[k - i]
        s.append(acc)
    for k in range(1, JMAX + 1):
        record("logistic s_k recurrence", float(s[k]), float(num(sp.diff(P, z, k), x)))
    # log f: Lambda_f = log(P (1 - P)); L_1 = 1 - 2P, L_j = -2 s_(j-1)
    Lf_log = sp.log(P * (1 - P))
    record("logistic logf L_1", float(1 - 2 * Pv), float(num(sp.diff(Lf_log, z, 1), x)))
    for j in range(2, JMAX + 1):
        record("logistic logf L_j", float(-2 * s[j - 1]), float(num(sp.diff(Lf_log, z, j), x)))
    # log F = log P: L_1 = 1 - P, L_j = -s_(j-1); log(1 - F) = log(1 - P): L_1 = -P, L_j = -s_(j-1)
    record("logistic logF L_1", float(1 - Pv), float(num(sp.diff(sp.log(P), z, 1), x)))
    record("logistic log(1-F) L_1", float(-Pv), float(num(sp.diff(sp.log(1 - P), z, 1), x)))
    for j in range(2, JMAX + 1):
        record("logistic logF L_j", float(-s[j - 1]), float(num(sp.diff(sp.log(P), z, j), x)))
        record("logistic log(1-F) L_j", float(-s[j - 1]), float(num(sp.diff(sp.log(1 - P), z, j), x)))

# --- Expected information. Cauchy: I = diag(1/(2 sigma^2), 1/(2 sigma^2)); logistic:
# I_mumu = 1/(3 sigma^2), I_sigmasigma = (pi^2 + 3)/(9 sigma^2). Their sigma-derivatives:
# d^K/dsigma^K sigma^-2 = (-1)^K (K+1)! sigma^-(K+2). Check against SymPy and quadrature. --
sig = sp.symbols("sigma", positive=True)
for c in (sp.Rational(1, 2), sp.Rational(1, 3), (sp.pi**2 + 3) / 9):
    for K in range(1, 5):
        want = sp.diff(c / sig**2, sig, K)
        mine = (-1) ** K * sp.factorial(K + 1) * c / sig ** (K + 2)
        if sp.simplify(mine - want) != 0:
            raise SystemExit(f"expected information derivative of order {K} differs")
print("expected information derivatives of order 1 to 4: symbolic identity holds")
# Quadrature of the score variances at sigma = 1 (the constants above).
mpmath.mp.dps = 40
cauchy_I_mu = mpmath.quad(lambda x: (2 * x / (1 + x * x)) ** 2 / (mpmath.pi * (1 + x * x)), [-mpmath.inf, 0, mpmath.inf])
logistic_I_mu = mpmath.quad(lambda x: (1 - 2 / (1 + mpmath.exp(-x))) ** 2 * mpmath.exp(-x) / (1 + mpmath.exp(-x)) ** 2, [-mpmath.inf, 0, mpmath.inf])
logistic_I_sig = mpmath.quad(
    lambda x: (1 + x * (1 - 2 / (1 + mpmath.exp(-x)))) ** 2 * mpmath.exp(-x) / (1 + mpmath.exp(-x)) ** 2,
    [-mpmath.inf, 0, mpmath.inf])
cauchy_I_sig = mpmath.quad(
    lambda x: (1 - 2 * x * x / (1 + x * x)) ** 2 / (mpmath.pi * (1 + x * x)), [-mpmath.inf, 0, mpmath.inf])
print("Cauchy I_mumu quad", cauchy_I_mu, "(want 1/2)")
print("Cauchy I_sigmasigma quad", cauchy_I_sig, "(want 1/2)")
print("logistic I_mumu quad", logistic_I_mu, "(want 1/3)")
print("logistic I_sigmasigma quad", logistic_I_sig, "(want (pi^2+3)/9 =", (mpmath.pi**2 + 3) / 9, ")")
if abs(logistic_I_sig - (mpmath.pi**2 + 3) / 9) > mpmath.mpf(10) ** -25:
    raise SystemExit("logistic I_sigmasigma differs from (pi^2 + 3)/9")
if abs(cauchy_I_mu - mpmath.mpf(1) / 2) > mpmath.mpf(10) ** -25 or abs(cauchy_I_sig - mpmath.mpf(1) / 2) > mpmath.mpf(10) ** -25:
    raise SystemExit("Cauchy information differs from 1/2")

d_coef = [mpmath.mpf(x) for x in ['-0.3183098861837906715377675', '-0.05066059182116888572193973', '0.09535278391686372745111516', '0.03120723231710817335067163', '-0.05356501848763564834060514', '-0.02264455753591862746639991', '0.03648098562136395317256342', '0.01779855358096177898071911', '-0.02733511667641241239426563', '-0.01467206806028395606950955', '0.02168999525747323033848866', '0.0124844036344296403699575']]

# --- Far-tail coefficients d_n of log(1 - atan(s)/pi) = sum_n d_n s^n (cauchy.hpp, far_tail_dk) ---
# Checked against SymPy's own series for the first 12 terms; the header table holds 60.
s_ = sp.symbols("s")
ser = sp.series(sp.log(1 - sp.atan(s_) / sp.pi), s_, 0, 13).removeO()
for n in range(1, 13):
    want = sp.N(ser.coeff(s_, n), 30)
    got = d_coef[n - 1]
    if abs(got - want) > mpmath.mpf(10) ** -20 * max(abs(want), 1):
        raise SystemExit(f"far-tail coefficient d_{n} differs: {got} vs {want}")
print("far-tail coefficients d_1..d_12 agree with SymPy's series")

for name in sorted(worst):
    print(f"{name}: max relative difference {worst[name]:.3e}")
print("all closed forms agree with SymPy")
