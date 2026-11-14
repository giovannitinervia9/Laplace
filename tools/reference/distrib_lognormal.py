"""Closed forms and reference values for lognormal1_distrib and lognormal2_distrib.

Usage (from the repository root):
    python -I tools/reference/distrib_lognormal.py forms       checks the closed forms, prints them
    python -I tools/reference/distrib_lognormal.py reference   writes tests/data/reference/distrib_lognormal.csv

Parametrizations (see include/laplace/distrib/lognormal1.hpp and lognormal2.hpp):
  lognormal1: log Y ~ N(mu, v), so l(mu, v; y) = gaussian2(mu, v; log y) - log y;
  lognormal2: mean m > 0 and variance v > 0 of Y, with s = log(1 + v/m^2) and mu = log m - s/2.

Forms (mode "forms"), each checked against SymPy's own derivative:
  1. the y-derivatives of a function of x = log y: D^J f(log y) = y^(-J) sum_k s(J, k) f^(k)(log y),
     s(J, k) the signed Stirling numbers of the first kind (theta = y d/dy, y^J D^J = theta (theta - 1)
     ... (theta - J + 1));
  2. the derivative of -log y in y, (-1)^J (J - 1)! y^(-J);
  3. the partial derivatives of s(m, v) and mu(m, v) to total order 6, in the factored form that
     is transcribed into lognormal2.hpp.

The reference (mode "reference") is mpmath.diff of the DEFINITION of log F and log(1 - F) in the
parameters, at 300 and 400 bits, kept when the two agree to 2^-80 relative. Each row:
    dist (1 lognormal1 in (mu, v), 2 lognormal2 in (m, v)), upper (0 log F, 1 log(1 - F)),
    K, b (b derivatives in the second parameter, K - b in the first), y, t0, t1, hi, lo,
with y, t0, t1, hi and lo in hexadecimal floats (hi + lo is the value with about 30 digits
beyond double). y is exp(mu + sigma z) for z in GRID, rounded to double, so the tails are sampled.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_lognormal.csv"
AGREE = mpmath.mpf(2) ** -80
GRID = [-35.0, -13.0, -6.0, -4.0, -2.5, -1.0, 0.0, 1.0, 2.5, 4.0, 6.0, 13.0, 35.0]

# (dist, first parameter, second parameter)
PARAMS_LN1 = [(0.3, 0.25), (0.3, 4.0), (-1.2, 0.01), (1.0, 25.0)]  # (mu, v): mean of log Y, variance of log Y
# the (mu, s) that each PARAMS_LN2 pair maps to: the same law, so lognormal1 at these points and lognormal2 at the
# pair share their observation points (same centre and scale in the y grid) and compare the composition's cost
PARAMS_LN1 += [(math.log(m0) - 0.5 * math.log1p(v0 / m0**2), math.log1p(v0 / m0**2))
               for (m0, v0) in [(40.0, 0.001), (0.5, 50.0), (2.0, 0.5), (0.3, 0.0025)]]
PARAMS_LN2 = [(40.0, 0.001), (0.5, 50.0), (2.0, 0.5), (0.3, 0.0025)]  # (m, v): mean and variance of Y


# ---------------------------------------------------------------------------------------------
# 1. y-derivatives of a function of x = log y (Stirling numbers of the first kind, signed)
# ---------------------------------------------------------------------------------------------

STIRLING = {
    1: {1: 1},
    2: {1: -1, 2: 1},
    3: {1: 2, 2: -3, 3: 1},
    4: {1: -6, 2: 11, 3: -6, 4: 1},
    5: {1: 24, 2: -50, 3: 35, 4: -10, 5: 1},
    6: {1: -120, 2: 274, 3: -225, 4: 85, 5: -15, 6: 1},
}


def check_stirling_and_minus_log():
    y = sp.symbols("y", positive=True)
    x = sp.log(y)
    # an explicit test function of x with all its derivatives nonzero
    f = lambda t: sp.exp(sp.Rational(3, 10) * t) * t**4 + sp.sin(t) / (2 + t**2)
    worst = 0
    for J in range(1, 7):
        lhs = sp.diff(f(x), y, J)
        rhs = 0
        for k, s_jk in STIRLING[J].items():
            rhs += s_jk * sp.diff(f(sp.Symbol("t")), sp.Symbol("t"), k).subs(sp.Symbol("t"), x)
        rhs = rhs / y**J
        for yv in [sp.Rational(7, 10), sp.Rational(5, 2), sp.Rational(13, 3)]:
            d = sp.N((lhs - rhs).subs(y, yv), 40)
            worst = max(worst, abs(float(d)))
        # the -log y term: D^J (-log y) = (-1)^J (J - 1)! y^(-J)
        lhs2 = sp.diff(-sp.log(y), y, J)
        rhs2 = (-1) ** J * sp.factorial(J - 1) / y**J
        assert sp.simplify(lhs2 - rhs2) == 0, J
    print(f"[stirling] D^J f(log y) and D^J(-log y), J = 1..6: max |difference| {worst:.3e}")
    assert worst < 1e-30


# ---------------------------------------------------------------------------------------------
# 2. the inner maps s(m, v) = log(1 + v/m^2) and mu(m, v) = log m - s/2
# ---------------------------------------------------------------------------------------------

m, v = sp.symbols("m v", positive=True)
S_INNER = sp.log(1 + v / m**2)  # log of 1 + v/m^2, the variance of log Y
MU_INNER = sp.log(m) - S_INNER / 2


def inner_forms(N):
    """Factored partials d^N/dm^(N-q) dv^q of s and of mu, q = 0..N (packed b = q)."""
    out = {"s": [], "mu": []}
    for q in range(N + 1):
        p = N - q
        for name, expr in (("s", S_INNER), ("mu", MU_INNER)):
            e = sp.diff(expr, m, p, v, q) if (p + q) > 0 else expr
            out[name].append(sp.factor(sp.together(sp.simplify(e))))
    return out


def check_inner_forms():
    for N in range(1, 7):
        forms = inner_forms(N)
        for q in range(N + 1):
            p = N - q
            for name, expr in (("s", S_INNER), ("mu", MU_INNER)):
                exact = sp.diff(expr, m, p, v, q)
                diff = sp.simplify(sp.together(exact - forms[name][q]))
                assert diff == 0, (name, p, q, diff)
    print("[inner] closed forms of s and mu, orders 1..6, checked against sympy.diff")


def print_inner():
    for N in range(1, 7):
        forms = inner_forms(N)
        for name in ("s", "mu"):
            for q in range(N + 1):
                print(f"{name} order {N}, dv^{q}: {forms[name][q]}")


# ---------------------------------------------------------------------------------------------
# 3. the expected information of lognormal2 in closed form at order 0 (the composition is in C++)
# ---------------------------------------------------------------------------------------------

def check_info_jacobian():
    """I_ab = sum_c I_c(mu, s) d_a theta_c d_b theta_c with I_mu = 1/s, I_s = 1/(2 s^2)."""
    sym_mu, sym_s = sp.symbols("mu_ s_", positive=True)
    Ic = [1 / sym_s, 1 / (2 * sym_s**2)]
    J = [[sp.diff(MU_INNER, m), sp.diff(MU_INNER, v)], [sp.diff(S_INNER, m), sp.diff(S_INNER, v)]]
    I_new = sp.zeros(2, 2)
    for a in range(2):
        for b in range(2):
            I_new[a, b] = sum(Ic[c].subs(sym_s, S_INNER) * J[c][a] * J[c][b] for c in range(2))
    # the off-diagonal entry is not zero: its leading term at small v/m^2 is -3/(2 m v)
    lead = sp.limit(sp.simplify(I_new[0, 1] * m * v), v, 0)
    print("[info] I_mv * m v at v -> 0:", lead)
    print("[info] I_mm =", sp.factor(sp.together(sp.simplify(I_new[0, 0]))))
    print("[info] I_mv =", sp.factor(sp.together(sp.simplify(I_new[0, 1]))))
    print("[info] I_vv =", sp.factor(sp.together(sp.simplify(I_new[1, 1]))))


def forms():
    check_stirling_and_minus_log()
    check_inner_forms()
    check_info_jacobian()
    print_inner()


# ---------------------------------------------------------------------------------------------
# reference values (mpmath, from the definition)
# ---------------------------------------------------------------------------------------------

def log_cdf_def(dist, upper, y, t0, t1):
    """The definition: log Phi(z) (upper = 0) or log Phi(-z), with z = (log y - mu)/sqrt(v)."""
    if dist == 1:
        mu, var = t0, t1
    else:
        mean, var_y = t0, t1
        s = mpmath.log1p(var_y / mean**2)
        mu = mpmath.log(mean) - s / 2
        var = s
    z = (mpmath.log(y) - mu) / mpmath.sqrt(var)
    if upper:
        z = -z
    return mpmath.log(mpmath.ncdf(z))


def value(dist, upper, y, K, b, t0, t1, prec):
    with mpmath.workprec(prec):
        f = lambda a, c: log_cdf_def(dist, upper, mpmath.mpf(y), a, c)
        return mpmath.diff(f, (mpmath.mpf(t0), mpmath.mpf(t1)), (K - b, b))


def hexf(x):
    return float(x).hex()


def row_list(job):
    dist, upper, t0, t1 = job
    rows = []
    mu_or_m = t0
    if dist == 1:
        centre, scale = mu_or_m, math.sqrt(t1)
    else:
        s = math.log1p(t1 / t0**2)
        centre, scale = math.log(t0) - s / 2, math.sqrt(s)
    for z in GRID:
        y = math.exp(centre + scale * z)
        if not (y > 0.0 and math.isfinite(y)):
            continue
        yd = float(y)  # the observation point is a double
        for K in range(1, 7):
            for b in range(K + 1):
                hi_prec = value(dist, upper, yd, K, b, t0, t1, 300)
                lo_prec = value(dist, upper, yd, K, b, t0, t1, 400)
                if not (mpmath.isfinite(lo_prec) and lo_prec != 0):
                    continue
                if abs(hi_prec - lo_prec) > AGREE * abs(lo_prec):
                    continue
                hi = float(lo_prec)
                lo = float(lo_prec - hi)
                rows.append(f"{dist},{upper},{K},{b},{hexf(yd)},{hexf(t0)},{hexf(t1)},{hexf(hi)},{hexf(lo)}")
    return rows


def reference():
    mpmath.mp.dps = 120
    jobs = []
    for dist, pars in ((1, PARAMS_LN1), (2, PARAMS_LN2)):
        for t0, t1 in pars:
            for upper in (0, 1):
                jobs.append((dist, upper, t0, t1))
    with Pool(4) as pool:
        results = pool.map(row_list, jobs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("dist,upper,K,b,y,t0,t1,hi,lo\n")
        n = 0
        for rows in results:
            for r in rows:
                f.write(r + "\n")
                n += 1
    print(f"wrote {n} rows to {OUT}")


YDK_OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_lognormal_ydk.csv"
YDK_POINTS = [(1.5, 0.8, 0.4), (1.5, 0.8, 2.5), (1.5, 0.8, 9.0), (40.0, 0.001, 40.0), (0.5, 50.0, 2.0),
              (2.0, 0.5, 0.9)]


def ydk_reference():
    """Mixed (y, theta) derivatives of the lognormal2 log-density, from the definition in SymPy.

    The log-density l(m, v; y) = -log(2 pi)/2 - log(s)/2 - (log y - mu)^2/(2 s) - log y, with
    s = log(1 + v/m^2), mu = log m - s/2, is differentiated J times in y and p times in m and q times
    in v symbolically (p + q = K, J + K <= 6), and the result is evaluated with mpmath at 300 bits.
    Row: m, v, y, J, K, b (the number of v derivatives), hi, lo.
    """
    y, m_s, v_s = sp.symbols("y m v", positive=True)
    s_expr = sp.log(1 + v_s / m_s**2)
    mu_expr = sp.log(m_s) - s_expr / 2
    x = sp.log(y)
    L = -sp.log(2 * sp.pi) / 2 - sp.log(s_expr) / 2 - (x - mu_expr) ** 2 / (2 * s_expr) - x
    mpmath.mp.dps = 300
    rows = []
    for (m0, v0, y0) in YDK_POINTS:
        for J in range(0, 7):
            for K in range(0, 7 - J):
                for q in range(K + 1):
                    p = K - q
                    expr = L
                    if J:
                        expr = sp.diff(expr, y, J)
                    if p:
                        expr = sp.diff(expr, m_s, p)
                    if q:
                        expr = sp.diff(expr, v_s, q)
                    f = sp.lambdify((y, m_s, v_s), expr, "mpmath")
                    val = f(mpmath.mpf(y0), mpmath.mpf(m0), mpmath.mpf(v0))
                    hi = float(val)
                    lo = float(val - hi)
                    rows.append(f"{hexf(m0)},{hexf(v0)},{hexf(y0)},{J},{K},{q},{hexf(hi)},{hexf(lo)}")
    YDK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(YDK_OUT, "w") as fh:
        fh.write("m,v,y,J,K,b,hi,lo\n")
        for r in rows:
            fh.write(r + "\n")
    print(f"wrote {len(rows)} rows to {YDK_OUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "forms"
    if mode == "forms":
        forms()
    elif mode == "reference":
        reference()
    elif mode == "ydk":
        ydk_reference()
    else:
        raise SystemExit(__doc__)
