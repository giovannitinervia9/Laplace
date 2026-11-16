"""References for beta_distrib (mean mu, precision phi; include/laplace/distrib/beta.hpp):
the log-density and its derivatives in (mu, phi) to order 6, the expected information and its
derivatives, the cdf family, and the derivatives of log F and log(1 - F) to order 6.

Usage (from the repository root):  python -I tools/reference/distrib_beta.py [part ...]
with the parts  check logpdf info cdf logcdf  (all of them when none is named). Files written under
tests/data/reference/:

  beta_logpdf.csv   y,mu,phi,K,b,hi,lo        d^K l / dmu^(K-b) dphi^b at (mu, phi), K = 0 the value
  beta_info.csv     kind,K,pair,b,mu,phi,hi,lo
                    kind 0: I_ab by quadrature of the squared score (K = 0 only);
                    kind 1: d^K I_ab / dmu^(K-b) dphi^b from SymPy's derivatives of the closed form,
                    pair 0 = (mu, mu), 1 = (mu, phi), 2 = (phi, phi)
  beta_cdf.csv      kind,y,mu,phi,hi,lo       kind 0 F, 1 1 - F, 2 log F, 3 log(1 - F)
  beta_logcdf.csv   upper,y,mu,phi,K,b,hi,lo  d^K log F (upper = 0) or log(1 - F) (upper = 1)
                    / dmu^(K-b) dphi^b by mpmath.diff of the definition

y, mu, phi, hi and lo are hexadecimal doubles (hi + lo is the value as a double-double).

What each reference is made from (nothing of beta.hpp's own formulas):
  * the log-density: SymPy's derivatives, in (mu, phi), of the DEFINITION
        l = loggamma(phi) - loggamma(mu phi) - loggamma(nu phi) + (mu phi - 1) log y + (nu phi - 1) log(1 - y),
    nu = 1 - mu, with log y and log(1 - y) as two symbols (l is linear in them, so a derivative is
    F + G log y + H log(1 - y)), evaluated by mpmath at 200 and 300 bits and kept where the two agree;
  * the expected information: the integral of S_a S_b f over (0, 1) by mpmath.quad in u = logit y, with S
    the scores that SymPy takes from the definition;
  * the cdf: x^a / (a B(a, b)) 2F1(a, 1 - b; a + 1; x) of mpmath, each tail from the series that
    converges to it (the upper tail by the exchange (a, b, x) -> (b, a, 1 - x)), at 260 and 340 bits;
  * the derivatives of log F and log(1 - F): mpmath.diff of that, in (mu, phi), at 200 and 260 bits.

Part `check` also verifies, against SymPy, the Leibniz form that beta.hpp uses for every component,
the derivatives in y, and the closed forms of the information against its quadrature.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
MAXK = 6
PHIS = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 2e3, 3e3, 5e3, 1e4, 1e5, 1e6, 1e7, 1e8]

mu_s, phi_s = sp.symbols("mu phi", positive=True)
LY, L1 = sp.symbols("LY L1", real=True)
nu_s = 1 - mu_s
L_DEF = (
    sp.loggamma(phi_s)
    - sp.loggamma(mu_s * phi_s)
    - sp.loggamma(nu_s * phi_s)
    + (mu_s * phi_s - 1) * LY
    + (nu_s * phi_s - 1) * L1
)
I_MM = phi_s**2 * (sp.polygamma(1, mu_s * phi_s) + sp.polygamma(1, nu_s * phi_s))
I_MP = phi_s * (mu_s * sp.polygamma(1, mu_s * phi_s) - nu_s * sp.polygamma(1, nu_s * phi_s))
I_PP = mu_s**2 * sp.polygamma(1, mu_s * phi_s) + nu_s**2 * sp.polygamma(1, nu_s * phi_s) - sp.polygamma(1, phi_s)


def hx(v):
    return float(v).hex()


def split(v):
    hi = float(v)
    return hi, float(v - mpmath.mpf(hi))


# ------------------------------------------------------------------ SymPy derivatives of the definition

def derivative_table(expr, kmax):
    """{(i, j): d^(i+j) expr / dmu^i dphi^j}, i + j <= kmax."""
    tab = {(0, 0): expr}
    for n in range(1, kmax + 1):
        for i in range(n + 1):
            j = n - i
            tab[(i, j)] = sp.diff(tab[(i - 1, j)], mu_s) if i > 0 else sp.diff(tab[(0, j - 1)], phi_s)
    return tab


_L_TAB = None
_L_FUN = None


def logpdf_functions():
    """Lambdified F (the part free of y), G (coefficient of log y), H (coefficient of log(1 - y))."""
    global _L_TAB, _L_FUN
    if _L_FUN is None:
        _L_TAB = derivative_table(L_DEF, MAXK)
        _L_FUN = {}
        for key, e in _L_TAB.items():
            f0 = e.subs({LY: 0, L1: 0})
            g = sp.diff(e, LY)
            h = sp.diff(e, L1)
            args = [mu_s, phi_s]
            _L_FUN[key] = (
                sp.lambdify(args, f0, "mpmath"),
                sp.lambdify(args, g, "mpmath"),
                sp.lambdify(args, h, "mpmath"),
            )
    return _L_FUN


def free_parts(mu, phi, dps):
    fun = logpdf_functions()
    with mpmath.workdps(dps):
        m, p = mpmath.mpf(mu), mpmath.mpf(phi)
        return {key: (f(m, p), g(m, p), h(m, p)) for key, (f, g, h) in fun.items()}


def logpdf_rows_for(mu, phi, ys):
    """Rows of beta_logpdf.csv for one (mu, phi), or None where the 200 and 300 bit values disagree."""
    a = free_parts(mu, phi, 60)
    b = free_parts(mu, phi, 90)
    rows = []
    with mpmath.workdps(90):
        for key in a:
            if key[0] + key[1] > MAXK:
                continue
            fa, fb = a[key][0], b[key][0]
            if abs(fa - fb) > max(mpmath.mpf(2) ** -100 * abs(fb), mpmath.mpf(10) ** -45):
                return None, key
        for y in ys:
            ly = mpmath.log(mpmath.mpf(y))
            l1 = mpmath.log(1 - mpmath.mpf(y))
            for key, (f, g, h) in b.items():
                K = key[0] + key[1]
                if K >= 3 and y != 0.5:
                    continue
                v = f + g * ly + h * l1
                if abs(v) < mpmath.mpf(10) ** -50:
                    v = mpmath.mpf(0)  # an exact zero of the symmetric case (mu = 1/2), not noise
                hi, lo = split(v)
                rows.append(f"{hx(y)},{hx(mu)},{hx(phi)},{K},{key[1]},{hx(hi)},{hx(lo)}\n")
    return rows, None


def mus_for(phi):
    s = {0.5, 0.1, 0.9, 0.01, 0.99}
    for a in (0.5, 0.05):
        m = a / phi
        if 1e-14 < m < 0.45:
            s.add(m)
            s.add(1.0 - m)
    return sorted(s)


def ys_for(mu, phi):
    ys = {1e-12, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-4, 1.0 - 1e-12, mu}
    sd = math.sqrt(mu * (1.0 - mu) / (phi + 1.0))
    for z in (-2.0, 2.0):
        y = mu + z * sd
        if 1e-9 < y < 1.0 - 1e-9:
            ys.add(y)
    return sorted(ys)


def part_logpdf():
    out = REF / "beta_logpdf.csv"
    n = dropped = 0
    with open(out, "w", newline="\n") as f:
        f.write("y,mu,phi,K,b,hi,lo\n")
        for phi in PHIS:
            for mu in mus_for(phi):
                rows, bad = logpdf_rows_for(mu, phi, ys_for(mu, phi))
                if rows is None:
                    dropped += 1
                    print(f"dropped (mu {mu!r}, phi {phi!r}): precisions disagree at {bad}")
                    continue
                f.writelines(rows)
                n += len(rows)
    print(f"{out.name}: {n} rows, {dropped} points dropped")


# ------------------------------------------------------------------ the expected information

def info_functions():
    tabs = {name: derivative_table(e, 4) for name, e in (("mm", I_MM), ("mp", I_MP), ("pp", I_PP))}
    return {
        name: {k: sp.lambdify([mu_s, phi_s], e, "mpmath") for k, e in tab.items()} for name, tab in tabs.items()
    }


def info_closed_rows(funs, mu, phi):
    rows = []
    for pair, name in enumerate(("mm", "mp", "pp")):
        for (i, j), fn in funs[name].items():
            K = i + j
            vals = []
            for dps in (60, 90):
                with mpmath.workdps(dps):
                    vals.append(fn(mpmath.mpf(mu), mpmath.mpf(phi)))
            if abs(vals[0] - vals[1]) > max(mpmath.mpf(2) ** -100 * abs(vals[1]), mpmath.mpf(10) ** -45):
                print(f"info closed form: precisions disagree at mu {mu!r} phi {phi!r} pair {pair} ({i}, {j})")
                continue
            hi, lo = split(vals[1] if abs(vals[1]) >= mpmath.mpf(10) ** -50 else mpmath.mpf(0))
            rows.append(f"1,{K},{pair},{j},{hx(mu)},{hx(phi)},{hx(hi)},{hx(lo)}\n")
    return rows


def scores():
    s_mu = sp.lambdify([mu_s, phi_s, LY, L1], sp.diff(L_DEF, mu_s), "mpmath")
    s_ph = sp.lambdify([mu_s, phi_s, LY, L1], sp.diff(L_DEF, phi_s), "mpmath")
    return s_mu, s_ph


def info_quadrature(s_mu, s_ph, mu, phi, dps=30):
    """I_ab = integral of S_a S_b f dy over (0, 1), in u = logit y (dy = y (1 - y) du)."""
    with mpmath.workdps(dps + 10):
        m = mpmath.mpf(mu)
        p = mpmath.mpf(phi)
        n = 1 - m
        a, b = m * p, n * p
        lbeta = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(p)

        def parts(u):
            ly = -mpmath.log1p(mpmath.exp(-u))
            l1 = -mpmath.log1p(mpmath.exp(u))
            dens = mpmath.exp(a * ly + b * l1 - lbeta)
            return dens, s_mu(m, p, ly, l1), s_ph(m, p, ly, l1)

        u0 = mpmath.log(m / n)
        w = 1 / mpmath.sqrt(p * m * n)
        pts = {u0 + k * w for k in (-64, -16, -4, -1, 0, 1, 4, 16, 64)}
        for c in (1, 10, 40, 120):
            pts.add(u0 - c / a)
            pts.add(u0 + c / b)
        pts = sorted(pts)
        pts = [-mpmath.inf] + pts + [mpmath.inf]
        out = []
        for fa, fb in ((0, 0), (0, 1), (1, 1)):
            def integrand(u, fa=fa, fb=fb):
                d, x1, x2 = parts(u)
                sc = (x1, x2)
                return sc[fa] * sc[fb] * d

            out.append(mpmath.quad(integrand, pts, maxdegree=9))
        return out


def part_info():
    funs = info_functions()
    s_mu, s_ph = scores()
    out = REF / "beta_info.csv"
    n = 0
    with open(out, "w", newline="\n") as f:
        f.write("kind,K,pair,b,mu,phi,hi,lo\n")
        for phi in PHIS:
            for mu in mus_for(phi):
                rows = info_closed_rows(funs, mu, phi)
                f.writelines(rows)
                n += len(rows)
        # the quadrature, at the points where it is quick and meaningful (both shapes at least 0.03)
        for phi in (0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1e3, 1e4, 1e6, 1e8):
            for mu in mus_for(phi):
                if mu * phi < 0.03 or (1 - mu) * phi < 0.03:
                    continue
                q = info_quadrature(s_mu, s_ph, mu, phi)
                for pair in range(3):
                    hi, lo = split(q[pair])
                    f.write(f"0,0,{pair},0,{hx(mu)},{hx(phi)},{hx(hi)},{hx(lo)}\n")
                    n += 1
    print(f"{out.name}: {n} rows")


# ------------------------------------------------------------------ the cdf family

def lower_series(a, b, x):
    """I_x(a, b) = x^a / (a B(a, b)) 2F1(a, 1 - b; a + 1; x): the series of mpmath, regularized."""
    return x**a * mpmath.hyp2f1(a, 1 - b, a + 1, x, maxterms=10**6) / (a * mpmath.beta(a, b))


def tails(a, b, x):
    """(F, 1 - F), each from the series that converges to it: the side of the mean of x is direct."""
    if x <= a / (a + b):
        lo = lower_series(a, b, x)
        return lo, 1 - lo
    up = lower_series(b, a, 1 - x)
    return 1 - up, up


def log_tails(a, b, x):
    """(log F, log(1 - F)) with the small side from its own value and the large one by log1p."""
    f, c = tails(a, b, x)
    if x <= a / (a + b):
        return mpmath.log(f), mpmath.log1p(-f)
    return mpmath.log1p(-c), mpmath.log(c)


def cdf_points():
    pts = []
    for phi in (0.05, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
        for mu in mus_for(phi):
            if mu * phi < 1e-3 or (1 - mu) * phi < 1e-3:
                continue
            for y in ys_for(mu, phi):
                pts.append((y, mu, phi))
    return pts


def cdf_rows_for(pt):
    y, mu, phi = pt
    res = []
    for prec in (260, 340):
        with mpmath.workprec(prec):
            m = mpmath.mpf(mu)
            a, b = m * phi, (1 - m) * phi
            try:
                f, c = tails(a, b, mpmath.mpf(y))
                lf, lc = log_tails(a, b, mpmath.mpf(y))
            except mpmath.libmp.libhyper.NoConvergence:
                return []
            res.append((f, c, lf, lc))
    rows = []
    for kind in range(4):
        v0, v1 = res[0][kind], res[1][kind]
        with mpmath.workprec(340):
            if abs(v0 - v1) > mpmath.mpf(2) ** -120 * abs(v1) and abs(v0 - v1) > mpmath.mpf(2) ** -120:
                continue
            if kind < 2 and not (mpmath.mpf(10) ** -300 < abs(v1) < 1e300):
                continue
            hi, lo = split(v1)
        rows.append(f"{kind},{hx(y)},{hx(mu)},{hx(phi)},{hx(hi)},{hx(lo)}\n")
    return rows


def part_cdf():
    out = REF / "beta_cdf.csv"
    with Pool(1) as pool:
        res = pool.map(cdf_rows_for, cdf_points(), chunksize=8)
    n = 0
    with open(out, "w", newline="\n") as f:
        f.write("kind,y,mu,phi,hi,lo\n")
        for rows in res:
            f.writelines(rows)
            n += len(rows)
    print(f"{out.name}: {n} rows")


# derivatives of log F and log(1 - F)

LOGCDF_POINTS = [
    (0.05, 0.5, [0.3, 0.9]),
    (0.5, 0.1, [0.05, 0.5]),
    (1.0, 0.5, [0.01, 0.5, 0.99]),
    (1.0, 0.9, [0.8]),
    (3.0, 0.3, [0.1, 0.3, 0.8]),
    (10.0, 0.1, [0.02, 0.1, 0.4]),
    (10.0, 0.5, [0.35]),
    (30.0, 0.7, [0.5, 0.7, 0.9]),
    (100.0, 0.2, [0.12, 0.2, 0.3]),
    (300.0, 0.5, [0.45, 0.5]),
    (1000.0, 0.4, [0.37, 0.4]),
]


def logcdf_def(upper, yv, mv, pv):
    a, b = mv * pv, (1 - mv) * pv
    lf, lc = log_tails(a, b, yv)
    return lc if upper else lf


def logcdf_value(upper, y, mu, phi, K, j, prec):
    with mpmath.workprec(prec):
        yv = mpmath.mpf(y)
        f = lambda mv, pv: logcdf_def(upper, yv, mv, pv)
        return mpmath.diff(f, (mpmath.mpf(mu), mpmath.mpf(phi)), (K - j, j))


def logcdf_task(task):
    upper, y, mu, phi = task
    rows = []
    for K in range(1, MAXK + 1):
        for j in range(K + 1):
            try:
                a = logcdf_value(upper, y, mu, phi, K, j, 200)
                c = logcdf_value(upper, y, mu, phi, K, j, 260)
            except mpmath.libmp.libhyper.NoConvergence:
                continue
            with mpmath.workprec(260):
                if c == 0 or abs(a - c) > mpmath.mpf(2) ** -80 * abs(c):
                    continue
                hi, lo = split(c)
            rows.append(f"{upper},{hx(y)},{hx(mu)},{hx(phi)},{K},{j},{hx(hi)},{hx(lo)}\n")
    return rows


def part_logcdf():
    tasks = [(u, y, mu, phi) for phi, mu, ys in LOGCDF_POINTS for y in ys for u in (0, 1)]
    out = REF / "beta_logcdf.csv"
    with Pool(3) as pool:
        res = pool.map(logcdf_task, tasks, chunksize=1)
    n = 0
    with open(out, "w", newline="\n") as f:
        f.write("upper,y,mu,phi,K,b,hi,lo\n")
        for rows in res:
            f.writelines(rows)
            n += len(rows)
    print(f"{out.name}: {n} rows")


# ------------------------------------------------------------------ the check part

def fact(n):
    return math.factorial(n)


def leibniz(i, j, m, phi, f):
    """d_mu^i d_phi^j F(m phi) = sum_r C(i, r) j!/(j-r)! m^(j-r) phi^(i-r) F^(i+j-r)(m phi): beta.hpp's shape_deriv."""
    s = mpmath.mpf(0)
    for r in range(min(i, j) + 1):
        s += math.comb(i, r) * (fact(j) // fact(j - r)) * m ** (j - r) * phi ** (i - r) * f(i + j - r)
    return s


def free_component(i, j, mu, phi):
    """The part of d_mu^i d_phi^j l free of y, by the form of beta.hpp (i + j >= 1)."""
    nu = 1 - mu
    a, b = mu * phi, nu * phi
    fa = lambda m: -mpmath.polygamma(m - 1, a)
    fb = lambda m: -mpmath.polygamma(m - 1, b)
    v = leibniz(i, j, mu, phi, fa) + (-1) ** i * leibniz(i, j, nu, phi, fb)
    if i == 0:
        v += mpmath.polygamma(i + j - 1, phi)
    return v


def mixed_info_component(i, j, mu, phi):
    """d_mu^i d_phi^j I_mu,phi by the form of beta.hpp: phi mu psi_1(alpha) - phi nu psi_1(beta), Leibniz over (phi m) and G."""
    nu = 1 - mu
    out = mpmath.mpf(0)
    for m, s, pm in ((mu, 1, 1), (nu, -1, -1)):
        w = m * phi
        g = lambda k, w=w: mpmath.polygamma(k + 1, w)
        tot = mpmath.mpf(0)
        for p in (0, 1):
            for q in (0, 1):
                if p > i or q > j:
                    continue
                P = {(0, 0): m * phi, (1, 0): s * phi, (0, 1): m, (1, 1): s}[(p, q)]
                a_, b_ = i - p, j - q
                tot += math.comb(i, p) * math.comb(j, q) * P * (s**a_) * leibniz(a_, b_, m, phi, g)
        out += pm * tot
    return out


def part_check():
    mpmath.mp.dps = 60
    tab = derivative_table(L_DEF, MAXK)
    fun = logpdf_functions()
    ok = True
    pts = [(0.3, 2.5), (0.7, 0.4), (0.05, 40.0), (0.5, 7.0), (0.93, 1e3)]
    for mu, phi in pts:
        m, p = mpmath.mpf(mu), mpmath.mpf(phi)
        for (i, j), (f0, g, h) in fun.items():
            if i + j == 0:
                continue
            ref = f0(m, p)
            got = free_component(i, j, m, p)
            err = abs(got - ref) / max(abs(ref), 1)
            if err > mpmath.mpf(10) ** -40:
                print("free component FAILS", (i, j), mu, phi, mpmath.nstr(err, 3))
                ok = False
    print("the Leibniz form of every y-free component of order 1 to 6:", "ok" if ok else "FAILED")
    # the information: closed derivatives against the forms of beta.hpp
    funs = info_functions()
    ok2 = True
    for mu, phi in pts:
        m, p = mpmath.mpf(mu), mpmath.mpf(phi)
        for K in range(0, 5):
            for j in range(K + 1):
                i = K - j
                ref_mm = funs["mm"][(i, j)](m, p)
                ref_mp = funs["mp"][(i, j)](m, p)
                ref_pp = funs["pp"][(i, j)](m, p)
                got_mm = -free_component(i + 2, j, m, p)
                got_pp = -free_component(i, j + 2, m, p)
                got_mp = mixed_info_component(i, j, m, p)
                for nm, g, r in (("mm", got_mm, ref_mm), ("mp", got_mp, ref_mp), ("pp", got_pp, ref_pp)):
                    err = abs(g - r) / max(abs(r), 1)
                    if err > mpmath.mpf(10) ** -35:
                        print("information FAILS", nm, (i, j), mu, phi, mpmath.nstr(err, 3))
                        ok2 = False
    print("the information components by the forms of beta.hpp (orders 0 to 4):", "ok" if ok2 else "FAILED")
    # the derivatives in y: (J - 1)! [(-1)^(J-1) (alpha - 1)/y^J - (beta - 1)/(1 - y)^J], with the theta derivatives
    y = sp.symbols("y", positive=True)
    ly = L_DEF.subs({LY: sp.log(y), L1: sp.log(1 - y)})
    ok3 = True
    for J in range(1, 7):
        dj = sp.diff(ly, y, J)
        for K in range(0, 7 - J):
            for jb in range(K + 1):
                e = dj
                for _ in range(K - jb):
                    e = sp.diff(e, mu_s)
                for _ in range(jb):
                    e = sp.diff(e, phi_s)
                fn = sp.lambdify([mu_s, phi_s, y], e, "mpmath")
                mu, phi, yy = 0.3, 2.5, 0.37
                m, p = mpmath.mpf(mu), mpmath.mpf(phi)
                yv = mpmath.mpf(yy)
                nu = 1 - m
                ya = (-1) ** (J - 1) * fact(J - 1) / yv**J
                yb = -fact(J - 1) / (1 - yv) ** J
                alpha_ij = {(0, 0): m * p, (1, 0): p, (0, 1): m, (1, 1): mpmath.mpf(1)}.get((K - jb, jb), mpmath.mpf(0))
                beta_ij = {(0, 0): nu * p, (1, 0): -p, (0, 1): nu, (1, 1): mpmath.mpf(-1)}.get((K - jb, jb), mpmath.mpf(0))
                if K == 0:
                    got = (alpha_ij - 1) * ya + (beta_ij - 1) * yb
                else:
                    got = alpha_ij * ya + beta_ij * yb
                ref = fn(m, p, yv)
                if abs(got - ref) > mpmath.mpf(10) ** -40 * max(abs(ref), 1):
                    print("y derivative FAILS", J, K, jb)
                    ok3 = False
    print("derivatives in y and (y, theta), J + K <= 6:", "ok" if ok3 else "FAILED")
    # the closed information against its quadrature
    s_mu, s_ph = scores()
    ok4 = True
    for mu, phi in [(0.3, 2.5), (0.5, 7.0), (0.1, 30.0), (0.8, 0.6)]:
        q = info_quadrature(s_mu, s_ph, mu, phi, dps=25)
        m, p = mpmath.mpf(mu), mpmath.mpf(phi)
        cl = [funs["mm"][(0, 0)](m, p), funs["mp"][(0, 0)](m, p), funs["pp"][(0, 0)](m, p)]
        for k in range(3):
            err = abs(q[k] - cl[k]) / max(abs(cl[k]), mpmath.mpf(10) ** -10)
            if err > mpmath.mpf(10) ** -18 and not (abs(cl[k]) < 1e-15 and abs(q[k]) < 1e-15):
                print("information vs quadrature FAILS", (mu, phi), k, mpmath.nstr(err, 3))
                ok4 = False
    print("the closed expected information against its quadrature:", "ok" if ok4 else "FAILED")
    return ok and ok2 and ok3 and ok4


PARTS = {
    "check": part_check,
    "logpdf": part_logpdf,
    "info": part_info,
    "cdf": part_cdf,
    "logcdf": part_logcdf,
}


def main():
    names = sys.argv[1:] or list(PARTS)
    for nm in names:
        if nm not in PARTS:
            print("unknown part", nm, "; parts:", " ".join(PARTS))
            return 2
    ok = True
    for nm in names:
        r = PARTS[nm]()
        if nm == "check" and r is False:
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
