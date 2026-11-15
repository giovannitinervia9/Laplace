"""SymPy check of the closed forms of negbin1 and negbin2 and mpmath references of their
log-mass, log F and log(1 - F) derivatives in (mu, theta) to total order 6.

Usage (from the repository root):  python -I tools/reference/distrib_negbin.py

Both families are the same negative binomial mass written in a shape r and a mean m,

    l(r, m; y) = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
                 + r log(r / (r + m)) + y log(m / (r + m)),

with (r, m) = (theta, mu) for negbin2 (variance mu + mu^2/theta) and (r, m) = (mu/theta, mu)
for negbin1 (variance mu (1 + theta)).

Part 1, SymPy (printed; the script exits with status 1 on any disagreement), checks the
forms the header uses:
  - the decomposition that removes every cancellation of the Poisson limit,
        l = V(r, y - 1/2, y) + delta(r + y) - delta(r) - V(r, y, m) - m + y log m - loggamma(y + 1),
    with V(r, c, h) = (r + c) log1p(h / r) - h and delta Stirling's error;
  - V_0 = c h / r - (r + c) psi1(h / r),  psi1(x) = x - log1p(x);
  - the direct forms of V_p = d^p V / dr^p, p = 1..6, through E_p = 1 - (r/(r+h))^p;
  - the direct forms of W_pq = d^(p+q) [(r + c) log1p(h/r)] / dr^p dh^q for q >= 1;
  - the series of V_p in powers of h / r, used below x = 1/8;
  - the partials of the inner maps r = mu/theta and m = mu of negbin1.

Part 2, mpmath (tests/data/reference/distrib_negbin.csv), one row per value:
    fam (1 negbin1, 2 negbin2), kind (0 log p, 1 log F, 2 log(1 - F)), y, mu, theta, K, b, hi, lo
with b the number of derivatives in theta (K - b in mu) and y, mu, theta, hi, lo as
hexadecimal doubles. The values are mpmath.diff of the DEFINITIONS above (log F and
log(1 - F) from the mass summed over the support, never from an incomplete beta), taken at
200 and 260 bits; a row is kept when the two agree to 2^-80 relative. No form of part 1
enters part 2.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_negbin.csv"
MAXK = 6
AGREE = mpmath.mpf(2) ** -80

# (mu, theta) pairs. negbin2: theta -> inf is the Poisson limit, theta -> 0 heavy overdispersion.
# negbin1: theta -> 0 is the Poisson limit (the shape mu/theta grows), and (2242, 0.028) is the
# corner statmodels7 could not repair (note/inventario_distrib.md, 2.2e-9 on d34e).
PARAMS = {
    2: [(0.5, 0.5), (3.5, 2.0), (40.0, 1e-2), (40.0, 1e3), (1e3, 1e6), (2.0, 1e-3), (1e3, 0.5), (1e5, 1e8)],
    1: [(0.5, 0.5), (3.5, 2.0), (40.0, 1e-2), (40.0, 1e3), (1e3, 1e-6), (2242.0, 0.028), (2.0, 1e3), (1e5, 1e-3)],
}
YMAX_CDF = 1e5  # log F and log(1 - F) cost a raised working precision per component: few points

r_s, m_s, c_s, h_s, x_s, y_s, mu_s, th_s = sp.symbols("r m c h x y mu theta", positive=True)


def stirling(a):
    """delta(a) = loggamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)."""
    return sp.loggamma(a) - (a - sp.Rational(1, 2)) * sp.log(a) + a - sp.log(sp.sqrt(2 * sp.pi))


def poch(j, p):
    """(j)_p = j (j+1) ... (j+p-1), with (0)_p = 0 for p >= 1."""
    out = sp.Integer(1)
    for i in range(p):
        out *= j + i
    return out


def check_forms():
    core = (sp.loggamma(y_s + r_s) - sp.loggamma(r_s) - sp.loggamma(y_s + 1)
            + r_s * sp.log(r_s / (r_s + m_s)) + y_s * sp.log(m_s / (r_s + m_s)))
    V = (r_s + c_s) * sp.log(1 + h_s / r_s) - h_s

    # 1. the decomposition
    VT = V.subs({c_s: y_s - sp.Rational(1, 2), h_s: y_s})
    VA = V.subs({c_s: y_s, h_s: m_s})
    decomp = VT + stirling(r_s + y_s) - stirling(r_s) - VA - m_s + y_s * sp.log(m_s) - sp.loggamma(y_s + 1)
    assert sp.simplify(sp.expand_log(core - decomp, force=True)) == 0, "decomposition"

    # 2. V_0 through psi1
    psi1 = x_s - sp.log(1 + x_s)
    v0 = c_s * h_s / r_s - (r_s + c_s) * psi1.subs(x_s, h_s / r_s)
    assert sp.simplify(V - v0) == 0, "V_0"

    # 3. V_p, p >= 1, through E_p = 1 - (r/(r+h))^p
    def E(p):
        return 1 - (r_s / (r_s + h_s)) ** p

    for p in range(1, MAXK + 1):
        if p == 1:
            form = -(r_s + c_s) * E(1) / r_s + sp.log(1 + h_s / r_s)
        else:
            form = ((-1) ** p * sp.factorial(p - 2) * r_s ** (1 - p)
                    * ((1 + c_s / r_s) * (p - 1) * E(p) - p * E(p - 1)))
        assert sp.simplify(form - sp.diff(V, r_s, p)) == 0, ("V_p", p)

    # 4. W_pq, q >= 1
    W = (r_s + c_s) * sp.log(1 + h_s / r_s)
    for q in range(1, MAXK + 1):
        for p in range(0, MAXK + 1 - q):
            if p == 0:
                form = (r_s + c_s) * (-1) ** (q - 1) * sp.factorial(q - 1) / (r_s + h_s) ** q
            else:
                form = ((-1) ** (p + q - 1) * sp.factorial(p + q - 2)
                        * (r_s * (q - 1) + c_s * (p + q - 1) - p * h_s) / (r_s + h_s) ** (p + q))
            assert sp.simplify(form - sp.diff(W, r_s, p, h_s, q)) == 0, ("W_pq", p, q)

    # 5. the series of V_p in powers of h/r: V_p = (-1)^p r^-p sum_n (-1)^(n-1) x^n/n [(n-1)_p r + c (n)_p]
    NS = 9
    for p in range(1, MAXK + 1):
        ser = sum((-1) ** (n - 1) * (h_s / r_s) ** n / n * (poch(n - 1, p) * r_s + c_s * poch(n, p))
                  for n in range(1, NS + 1))
        ser = (-1) ** p * r_s ** (-p) * ser
        d = sp.diff(V, r_s, p)
        tail = sp.series(sp.expand(d - ser), h_s, 0, NS + 1).removeO()
        assert sp.simplify(tail) == 0, ("series V_p", p, tail)

    # 6. the inner maps of negbin1: r = mu/theta, m = mu
    rin = mu_s / th_s
    for p in range(0, MAXK + 1):
        for q in range(0, MAXK + 1 - p):
            if p + q == 0:
                continue
            if p == 0:
                form = mu_s * (-1) ** q * sp.factorial(q) / th_s ** (q + 1)
            elif p == 1:
                form = (-1) ** q * sp.factorial(q) / th_s ** (q + 1)
            else:
                form = sp.Integer(0)
            assert sp.simplify(form - sp.diff(rin, mu_s, p, th_s, q)) == 0, ("r map", p, q)
    # 7. Loader's saddle-point identity, for y >= 1:
    #   l = (1/2) log(r / (2 pi y (y + r))) + delta(y+r) - delta(y) - delta(r)
    #       - bd0(y, b1) - bd0(r, b2)
    # with bd0(a, b) = a log(a/b) - a + b, b1 = m (y+r)/(r+m), b2 = r (y+r)/(r+m). The identity
    # uses delta(y) = stirlerr(y), that is loggamma(y + 1) = (y + 1/2) log y - y
    # + (1/2) log(2 pi) + delta(y), which is checked first.
    # SymPy does not reduce loggamma(y + 1) - loggamma(y) on its own; the recurrence is applied
    # by hand wherever loggamma(y + 1) appears
    shift = {sp.loggamma(y_s + 1): sp.loggamma(y_s) + sp.log(y_s)}
    assert sp.simplify(sp.expand_log(
        (sp.loggamma(y_s + 1) - ((y_s + sp.Rational(1, 2)) * sp.log(y_s) - y_s
                                 + sp.log(sp.sqrt(2 * sp.pi)) + stirling(y_s))).subs(shift),
        force=True)) == 0, "stirlerr"
    b1 = m_s * (y_s + r_s) / (r_s + m_s)
    b2 = r_s * (y_s + r_s) / (r_s + m_s)
    assert sp.simplify(b1 + b2 - (y_s + r_s)) == 0, "b1 + b2"
    delta_num = r_s * (y_s - m_s) / (r_s + m_s)
    assert sp.simplify(y_s - b1 - delta_num) == 0, "y - b1"
    assert sp.simplify(r_s - b2 + delta_num) == 0, "r - b2"

    def bd0(a, b):
        return a * sp.log(a / b) - a + b

    loader = (sp.Rational(1, 2) * sp.log(r_s / (2 * sp.pi * y_s * (y_s + r_s)))
              + stirling(y_s + r_s) - stirling(y_s) - stirling(r_s) - bd0(y_s, b1) - bd0(r_s, b2))
    assert sp.simplify(sp.expand_log((core - loader).subs(shift), force=True)) == 0, "Loader"

    # 8. negbin1's rationals: with r = mu/theta and m = mu, every inner map is regular at
    # theta = 0, and the second deviance has no 1/theta left in it.
    sub1 = {r_s: mu_s / th_s, m_s: mu_s}
    u = y_s * th_s / mu_s
    kappa = (sp.log(1 + x_s) - x_s) / x_s
    assert sp.simplify(sp.Rational(1, 2) * sp.log(r_s / (y_s + r_s)).subs(sub1)
                       + sp.Rational(1, 2) * sp.log(1 + u)) == 0, "negbin1 prefactor"
    assert sp.simplify(b1.subs(sub1) - (y_s * th_s + mu_s) / (1 + th_s)) == 0, "negbin1 b1"
    assert sp.simplify((delta_num / b1).subs(sub1) - (y_s - mu_s) / (y_s * th_s + mu_s)) == 0, "negbin1 w1"
    # bd0(r, b2) as a product: b2 = b1 / theta and w2 = -theta w1 exactly, so with
    # psi(v) = D(1 + v) / v the factor 1/theta cancels and bd0(r, b2) = -b1 w1 psi(-theta w1),
    # which does not cancel where y is near mu (the form through kappa did: its terms are of
    # the size of mu while the result is of the size of (y - mu)^2 theta / mu)
    b1n, w1n = (y_s * th_s + mu_s) / (1 + th_s), (y_s - mu_s) / (y_s * th_s + mu_s)
    assert sp.simplify(b2.subs(sub1) - b1n / th_s) == 0, "negbin1 b2 = b1 / theta"
    assert sp.simplify((-delta_num / b2).subs(sub1) + th_s * w1n) == 0, "negbin1 w2 = -theta w1"
    v_s = sp.symbols("v")
    psi_v = ((1 + v_s) * sp.log(1 + v_s) - v_s) / v_s
    assert sp.simplify(sp.expand_log(
        bd0(r_s, b2).subs(sub1) + b1n * w1n * psi_v.subs(v_s, -th_s * w1n), force=True)) == 0, \
        "negbin1 bd0(r, b2) as a product"
    ser_psi = sum((-1) ** (j + 1) * v_s ** j / (j * (j + 1)) for j in range(1, 12))
    assert sp.simplify(sp.series(psi_v - ser_psi, v_s, 0, 12).removeO()) == 0, "psi series"
    for q in range(1, MAXK + 1):
        lhs = v_s * sp.diff(psi_v, v_s, q) + q * sp.diff(psi_v, v_s, q - 1)
        rhs = sp.diff((1 + v_s) * sp.log(1 + v_s) - v_s, v_s, q)
        assert sp.simplify(lhs - rhs) == 0, ("psi recursion", q)

    # 9. negbin2's rationals, with r = theta and m = mu
    sub2 = {r_s: th_s, m_s: mu_s}
    assert sp.simplify(b1.subs(sub2) - mu_s * (y_s + th_s) / (th_s + mu_s)) == 0, "negbin2 b1"
    assert sp.simplify((delta_num / b1).subs(sub2) - th_s * (y_s - mu_s) / (mu_s * (y_s + th_s))) == 0, "negbin2 w1"
    assert sp.simplify((-delta_num / b2).subs(sub2) - (mu_s - y_s) / (y_s + th_s)) == 0, "negbin2 w2"

    # 10. kappa: the recursion z kappa^(q) + q kappa^(q-1) = d^q [log1p(z) - z] used above 1/2,
    # and the series coefficients used below it
    kz = kappa.subs(x_s, x_s)
    for q in range(1, MAXK + 1):
        lhs = x_s * sp.diff(kz, x_s, q) + q * sp.diff(kz, x_s, q - 1)
        assert sp.simplify(lhs - sp.diff(sp.log(1 + x_s) - x_s, x_s, q)) == 0, ("kappa recursion", q)
    ser = sum((-1) ** n * x_s ** n / (n + 1) for n in range(1, 12))
    assert sp.simplify(sp.series(kz - ser, x_s, 0, 12).removeO()) == 0, "kappa series"

    # 11. the difference of Stirling's errors as a function of s = 1/r: the asymptotic series
    # of stirling.hpp at r and at y + r, the second through t = s / (1 + y s)
    s_s = sp.symbols("s", positive=True)
    t_of_s = s_s / (1 + y_s * s_s)
    assert sp.simplify(t_of_s - 1 / (y_s + 1 / s_s)) == 0, "t(s)"
    for m in range(1, MAXK + 1):
        form = (-1) ** (m - 1) * sp.factorial(m) * y_s ** (m - 1) / (1 + y_s * s_s) ** (m + 1)
        assert sp.simplify(form - sp.diff(t_of_s, s_s, m)) == 0, ("t derivatives", m)

    print("forms checked against SymPy: Loader's identity, both parametrizations' rationals,")
    print("negbin1's second deviance, kappa, t(s), and V_0, V_p, W_pq with their series")


# ------------------------------------------------------------------ mpmath references


def log_mass(fam, y, mu, theta):
    r = theta if fam == 2 else mu / theta
    m = mu
    return (mpmath.loggamma(y + r) - mpmath.loggamma(r) - mpmath.loggamma(y + 1)
            + r * mpmath.log(r / (r + m)) + y * mpmath.log(m / (r + m)))


MASS_FUNCS = {}


def mass_derivative_functions():
    """Lambdified (mu, theta, y) derivatives of the log-mass, indexed by (fam, K, b).

    The derivative of the definition is taken symbolically, so the mass rows need no
    difference quotient and no raised working precision.
    """
    out = {}
    for fam in (1, 2):
        r = th_s if fam == 2 else mu_s / th_s
        l = (sp.loggamma(y_s + r) - sp.loggamma(r) - sp.loggamma(y_s + 1)
             + r * sp.log(r / (r + mu_s)) + y_s * sp.log(mu_s / (r + mu_s)))
        for K in range(0, MAXK + 1):
            for b in range(0, K + 1):
                d = sp.diff(l, mu_s, K - b, th_s, b)
                out[(fam, K, b)] = sp.lambdify((mu_s, th_s, y_s), d, modules="mpmath")
    return out


def log_cdf(fam, k, mu, theta, upper):
    """log F(k) or log(1 - F(k)) through the incomplete beta, F(k) = I_p(r, k + 1).

    The identity is the classical one between the negative binomial's cumulative mass and
    the regularized incomplete beta function; mpmath evaluates it from its own
    hypergeometric series, so the route shares nothing with the library's.
    """
    r = theta if fam == 2 else mu / theta
    p = r / (r + mu)
    a, b = (mpmath.mpf(0), p) if not upper else (p, mpmath.mpf(1))
    return mpmath.log(mpmath.betainc(r, k + 1, a, b, regularized=True))


def value(fam, kind, y, mu, theta, K, b, prec):
    with mpmath.workprec(prec if kind != 0 else 2 * prec):
        # the mass rows are symbolic derivatives, so the precision costs almost nothing, and the
        # high orders in theta need it: the expression for the sixth derivative of
        # loggamma(y + mu/theta) carries terms of the size of theta^-6 while the result is of the
        # size of mu, so at theta = 1e-6 about 31 digits are lost and 200 bits leave too few for
        # the two precisions to agree
        mu0, th0 = mpmath.mpf(mu), mpmath.mpf(theta)
        if kind == 0:
            return MASS_FUNCS[(fam, K, b)](mu0, th0, mpmath.mpf(y))
        f = lambda a, t: log_cdf(fam, math.floor(y), a, t, kind == 2)
        if K == 0:
            return f(mu0, th0)
        return mpmath.diff(f, (mu0, th0), (K - b, b))


def row(fam, kind, y, mu, theta, K, b):
    try:
        a = value(fam, kind, y, mu, theta, K, b, 200)
        d = value(fam, kind, y, mu, theta, K, b, 260)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    if d == 0 or not mpmath.isfinite(d) or abs(a - d) > AGREE * abs(d):
        return None
    hi = float(d)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    lo = float(d - mpmath.mpf(hi))
    return (f"{fam},{kind},{float(y).hex()},{float(mu).hex()},{float(theta).hex()},"
            f"{K},{b},{hi.hex()},{lo.hex()}\n")


def ys_for(mu, theta, fam):
    sd = math.sqrt(mu + mu * mu / theta) if fam == 2 else math.sqrt(mu * (1.0 + theta))
    ys = {0.0, 1.0, 2.0, 7.0}
    for f in (0.5, 1.0, 2.0):
        ys.add(float(round(max(0.0, mu + f * sd))))
    ys.add(float(round(mu)))
    ys.add(2.5)  # one non-integer, to exercise the floor of log F
    return sorted(v for v in ys if v < 1e8)


def task(job):
    """One (family, mu, theta, kind): every y of the grid and every component to order 6.

    The mass rows come from the lambdified symbolic derivatives; the two cumulative kinds
    from mpmath.diff on the summed mass, which costs a raised working precision per
    component, so they take the first few points of the grid only.
    """
    fam, mu, theta, kind = job
    ys = ys_for(mu, theta, fam)
    if kind != 0:
        ys = [v for v in ys if v <= YMAX_CDF][:4]
    lines, dropped = [], 0
    for y in ys:
        for K in range(0, MAXK + 1):
            for b in range(0, K + 1):
                r = row(fam, kind, y, mu, theta, K, b)
                if r is None:
                    dropped += 1
                else:
                    lines.append(r)
    return lines, dropped


def init():
    global MASS_FUNCS
    MASS_FUNCS = mass_derivative_functions()


def main():
    check_forms()
    init()
    jobs = [(fam, mu, th, kind) for fam, ps in PARAMS.items() for kind in (0, 1, 2) for (mu, th) in (ps if kind == 0 else ps[:5])]
    with Pool(8, initializer=init) as pool:
        res = pool.map(task, jobs)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("fam,kind,y,mu,theta,K,b,hi,lo\n")
        for lines, dr in res:
            f.writelines(lines)
            n += len(lines)
            dropped += dr
    print(f"distrib_negbin.csv: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
