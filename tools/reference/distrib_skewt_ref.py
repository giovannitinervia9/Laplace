"""The reference values of log f and its derivatives for skewt_distrib
(include/laplace/distrib/skewt.hpp); the closed-form checks are in distrib_skewt.py.

Usage (from the repository root):
    python -I tools/reference/distrib_skewt_ref.py check
    python -I tools/reference/distrib_skewt_ref.py logpdf [workers]

Writes tests/data/reference/distrib_skewt_logpdf.csv with the columns
    y, K, a, b, c, d, hi, lo, mu, sigma, alpha, nu
every number in hexadecimal, (a, b, c, d) the number of derivatives in (mu, sigma, alpha, nu)
and K = a + b + c + d; hi + lo is the value as a sum of two doubles.

Method. Nothing of the library's routes and no closed form of the header enters. With
M = nu + 1 the log-density is

    log f = log 2 + log Gamma(M/2) - log Gamma((M-1)/2) - log((M-1) pi)/2 - log sigma
            - (M/2) log(1 + z^2/(M-1)) + LT(w, M),
    z = (y - mu)/sigma,   w = alpha z sqrt(M/(M - 1 + z^2)),

where LT(x, M) = log T_M(x) is carried as an unknown function of two variables. SymPy
differentiates the whole expression in (mu, sigma, alpha, M) for every multi-index of total
order one to six, which is its own chain rule and not the header's, and each derivative of LT
that appears is replaced by a symbol A_ij standing for

    A_ij = d^i/dx^i d^j/dM^j log T_M(x)   at   (w, M).

Those are computed from the Student density t_M by differentiating under the integral sign,
which is legitimate because d^j/dM^j t_M(u) decays like |u|^(-M-1) times powers of
log(1 + u^2/(M-1)):

    d^i/dx^i d^j/dM^j T_M(x) = d^(i-1)/dx^(i-1) [d^j/dM^j t_M(x)]      for i >= 1 (SymPy, exact),
    d^j/dM^j T_M(x) = integral of d^j/dM^j t_M over the side of x that does not cancel
                      (mpmath quadrature; for x > 0 the upper side, with
                      T = 1 - U at j = 0 and d^j T = -d^j U above),

and the derivatives of the logarithm follow from those of T by the same unknown-function
device in two variables. Every point is evaluated twice, at 60 and at 90 digits; a row is kept
when the two agree to 2^-100 relative and dropped otherwise.
"""

import csv
import os
import sys
import time

import mpmath as mp
import sympy as sp

# ---------------------------------------------------------------- symbols

MU, SG, AL, MM, Y = sp.symbols("mu sg al M y", real=True)
X = sp.Symbol("x", real=True)
NU = MM - 1
Z = (Y - MU) / SG
W = AL * Z * sp.sqrt(MM / (NU + Z ** 2))
LT = sp.Function("LT")
TF = sp.Function("TF")
A = {(i, j): sp.Symbol("A_%d_%d" % (i, j)) for i in range(7) for j in range(7)}
TS = {(i, j): sp.Symbol("T_%d_%d" % (i, j)) for i in range(7) for j in range(7)}


def resolve(e, fun, sym, arg1, arg2):
    """Replace every derivative of `fun`(arg1, arg2) in e by sym[(i, j)]."""

    def one(s):
        d = s.expr if isinstance(s, sp.Subs) else s
        i = j = 0
        a1 = d.expr.args[0]
        for v, n in d.variable_count:
            if v == a1 or str(v).startswith("_xi_1"):
                i += n
            else:
                j += n
        return sym[(i, j)]

    e = e.replace(lambda x: isinstance(x, sp.Subs), one)
    e = e.replace(lambda x: isinstance(x, sp.Derivative) and x.expr.func == fun, one)
    return e.replace(fun(arg1, arg2), sym[(0, 0)])


# ---------------------------------------------------------------- the pieces, once

_BUILT_DENSITY = False
_BUILT = False


def binom(n, k):
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def build_density():
    """The density's derivatives and the derivatives of a logarithm: a few seconds, and all the
    tail quadrature needs.

    The Student density at M degrees of freedom is t_M(u) = C(M) g(u, M) with
      C(M) = Gamma((M+1)/2) / (sqrt(M pi) Gamma(M/2)),   g(u, M) = (1 + u^2/M)^(-(M+1)/2),
    and the normalizer does not depend on u, so the quadratures integrate combinations of the
    M-derivatives of g alone, which carry no gamma function: by Leibniz's rule
      d^j/dM^j t_M(u) = sum_i C(j, i) C^(i)(M) d^(j-i)/dM^(j-i) g(u, M).
    """
    global _BUILT_DENSITY, CNORM, GMIX, LOGT, TKEYS
    if _BUILT_DENSITY:
        return
    cexpr = sp.gamma((MM + 1) / 2) / (sp.sqrt(MM * sp.pi) * sp.gamma(MM / 2))
    gexpr = (1 + X ** 2 / MM) ** (-(MM + 1) / 2)
    CNORM = [sp.lambdify(MM, sp.diff(cexpr, MM, i) if i else cexpr, "mpmath") for i in range(7)]
    GMIX = {}
    for q in range(7):
        e = sp.diff(gexpr, MM, q) if q else gexpr
        for pp in range(7 - q):
            GMIX[(pp, q)] = sp.lambdify((X, MM), sp.diff(e, X, pp) if pp else e, "mpmath")
    LOGT = {}
    for n in range(1, 7):
        for i in range(n + 1):
            j = n - i
            LOGT[(i, j)] = sp.lambdify(
                tuple(TS[(pp, q)] for pp in range(7) for q in range(7) if pp + q <= 6),
                resolve(sp.diff(sp.log(TF(X, MM)), X, i, MM, j), TF, TS, X, MM), "mpmath")
    TKEYS = [(pp, q) for pp in range(7) for q in range(7) if pp + q <= 6]
    _BUILT_DENSITY = True


def build():
    """The derivatives of log f in the four parameters, with LT left unknown: about two minutes,
    so the modes that do not need them (check) never pay for it.

    The derivatives are built one variable at a time from the order below (so each entry costs one
    differentiation, not K), resolved, and the whole set goes through one common-subexpression
    elimination and one lambdify: the 209 expressions share nearly everything.
    """
    global _BUILT, AKEYS, ARGS, KEYS, ALL
    if _BUILT:
        return
    build_density()
    print("building the symbolic derivatives", flush=True)
    t0 = time.time()
    c = sp.loggamma(MM / 2) - sp.loggamma((MM - 1) / 2) - sp.log((MM - 1) * sp.pi) / 2
    logf = sp.log(2) + c - sp.log(SG) - MM / 2 * sp.log(1 + Z ** 2 / NU) + LT(W, MM)
    AKEYS = [(pp, q) for pp in range(7) for q in range(7) if pp + q <= 6]
    ARGS = (MU, SG, AL, MM, Y) + tuple(A[k] for k in AKEYS)
    variables = (MU, SG, AL, MM)
    raw = {(0, 0, 0, 0): logf}
    KEYS = [(0, 0, 0, 0)]
    for K in range(1, 7):
        for a in range(K + 1):
            for b in range(K - a + 1):
                for cc in range(K - a - b + 1):
                    d = K - a - b - cc
                    key = (a, b, cc, d)
                    for i, n in enumerate(key):      # differentiate the first lower entry found
                        if n > 0:
                            low = list(key)
                            low[i] -= 1
                            raw[key] = sp.diff(raw[tuple(low)], variables[i])
                            break
                    KEYS.append(key)
        print("  order %d differentiated (%.0f s)" % (K, time.time() - t0), flush=True)
    expr = [resolve(raw[k], LT, A, W, MM) for k in KEYS]
    print("  resolved (%.0f s), lambdifying with cse" % (time.time() - t0), flush=True)
    ALL = sp.lambdify(ARGS, expr, "mpmath", cse=True)
    print("  lambdified (%.0f s)" % (time.time() - t0), flush=True)
    _BUILT = True


def tail_panels(x, m):
    """Geometric panels from x outwards, the first of the width over which t_M decays by e.

    The decay length of t_M at u = x is 1/|d log t/du| = (m + x^2)/((m + 1)|x|), which at large m
    and large |x| is a small fraction of the distance to the next round number: panels of fixed
    width miss the mass entirely (measured below).
    """
    ax = abs(x)
    d = (m + x * x) / ((m + 1) * max(ax, mp.mpf("0.1")))
    pts, step, far = [mp.mpf(0)], d, mp.mpf(0)
    for _ in range(80):
        far = far + step
        pts.append(far)
        step *= 2
        if far > 100 * d and far > 100:
            break
    pts.append(mp.inf)
    return pts


def log_t_partials(x, m, dps):
    """A_ij = d^i/dx^i d^j/dM^j log T_M(x) at (x, m).

    T and its M-derivatives come from the integral of the density's M-derivatives over the side of
    x that does not cancel, in the variable s = |u - x| and scaled by the density at x, so that
    the integrand is of size one at s = 0: without that scaling mpmath's quadrature of a tail of
    size 1e-274 loses everything but three digits (2.5e-3 relative at m = 1001, x = -50, against
    mpmath's own incomplete beta; 1e-59 with it). The x-derivatives of T are the density's own
    derivatives, and the derivatives of the logarithm follow from those of T.
    """
    build_density()
    with mp.workprec(int(dps * 3.33)):
        x, m = mp.mpf(x), mp.mpf(m)
        cn = [CNORM[i](m) for i in range(7)]
        upper = x > 0
        sgn = 1 if upper else -1          # u = x + sgn * s stays on the side of x we integrate
        lx = mp.log(cn[0]) + mp.log(GMIX[(0, 0)](x, m))   # log t_M(x)
        at_x = mp.e ** lx

        def integrand(j):
            return lambda s, j=j: sum(binom(j, i) * cn[i] * GMIX[(0, j - i)](x + sgn * s, m)
                                      for i in range(j + 1)) / at_x

        pts = tail_panels(x, m)
        vals = [at_x * mp.quad(integrand(j), pts) for j in range(7)]
        T = {}
        for j in range(7):
            v = vals[j]
            T[(0, j)] = ((1 - v) if j == 0 else -v) if upper else v
        for i in range(1, 7):
            for j in range(7 - i):
                T[(i, j)] = sum(binom(j, q) * cn[q] * GMIX[(i - 1, j - q)](x, m)
                                for q in range(j + 1))
        tv = [T[k] for k in TKEYS]
        out = {(0, 0): mp.log(T[(0, 0)])}
        for k, f in LOGT.items():
            out[k] = f(*tv)
        return out


def check_tail(x, m, dps=60):
    """T_M(x) from the quadrature above against mpmath's incomplete beta: the arbiter."""
    with mp.workprec(int(dps * 3.33)):
        x, m = mp.mpf(x), mp.mpf(m)
        q = m / (m + x * x)
        lower = mp.betainc(m / 2, mp.mpf(1) / 2, 0, q, regularized=True) / 2
        beta = lower if x <= 0 else 1 - lower
        got = mp.e ** log_t_partials(x, m, dps)[(0, 0)]
        return float(abs(got - beta) / abs(beta))


def row_values(mu, sigma, alpha, nu, y, dps):
    """every derivative of log f of order 0..6 at one point, keyed (K, a, b, c, d)"""
    build()
    with mp.workprec(int(dps * 3.33)):
        m = mp.mpf(nu) + 1
        z = (mp.mpf(y) - mp.mpf(mu)) / mp.mpf(sigma)
        w = mp.mpf(alpha) * z * mp.sqrt(m / (mp.mpf(nu) + z * z))
        Av = log_t_partials(w, m, dps)
        a = tuple(Av[k] for k in AKEYS)
        red = ALL(mp.mpf(mu), mp.mpf(sigma), mp.mpf(alpha), m, mp.mpf(y), *a)
        out = {}
        for key, v in zip(KEYS, red):
            out[(sum(key),) + key] = v
        return out


# the hard points of the family: alpha = 0 (the Student), nu large (the skew normal), |alpha|
# large (the half t), small nu, both tails, and several scales
PARAMS = [
    (0.0, 1.0, 0.0, 3.5),
    (0.0, 1.0, 0.0, 0.7),
    (-2.0, 0.5, 0.75, 3.5),
    (-2.0, 0.5, -0.75, 3.5),
    (1.5, 2.0, 2.0, 1.0),
    (1.5, 2.0, 2.0, 12.0),
    (0.0, 1.0, 5.0, 2.5),
    (0.0, 1.0, -5.0, 2.5),
    (0.0, 1.0, 50.0, 4.0),
    (0.0, 1.0, 0.5, 200.0),
    (0.0, 1.0, 2.0, 1000.0),
    (0.3, 0.25, 1.0, 0.3),
    (0.0, 1.0, 1.0, 1e4),
    (0.0, 1.0, -2.0, 1e6),
    (7.0, 0.0009765625, 1.5, 2.0),   # sigma = 2^-10
    (-7.0, 1024.0, -1.5, 6.0),       # sigma = 2^10
    (0.0, 1.0, 20.0, 0.5),
    (0.0, 1.0, -20.0, 30.0),
]
ZS = [-1e8, -1e4, -30.0, -3.0, -0.4, 0.0, 0.9, 4.0, 25.0, 1e4]


def hexd(v):
    return float(v).hex()


def split(v):
    hi = float(v)
    return hi, float(mp.mpf(v) - mp.mpf(hi))


def one_point(job):
    """one (mu, sigma, alpha, nu, y) at two precisions; returns the rows that agree"""
    mu, sigma, alpha, nu, y = job
    t1 = time.time()
    lo = row_values(mu, sigma, alpha, nu, y, 60)
    hi = row_values(mu, sigma, alpha, nu, y, 90)
    rows, dropped = [], 0
    for key in sorted(lo):
        K = key[0]
        if K == 0:
            continue
        v1, v2 = mp.mpf(lo[key]), mp.mpf(hi[key])
        if abs(v1 - v2) > mp.mpf(2) ** -100 * max(abs(v2), mp.mpf(1)):
            dropped += 1
            continue
        _, a, b, c, d = key
        h, l = split(v2)
        rows.append([hexd(y), K, a, b, c, d, hexd(h), hexd(l), hexd(mu), hexd(sigma), hexd(alpha),
                     hexd(nu)])
    print("  mu %g sigma %g alpha %g nu %g y %g: %d rows, %d dropped (%.0f s)" %
          (mu, sigma, alpha, nu, y, len(rows), dropped, time.time() - t1), flush=True)
    return rows, dropped


def main(workers):
    jobs = [(mu, sigma, alpha, nu, mu + sigma * zz)
            for (mu, sigma, alpha, nu) in PARAMS for zz in ZS]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(one_point, jobs))
    else:
        out = [one_point(j) for j in jobs]
    rows = [r for part, _ in out for r in part]
    dropped = sum(d for _, d in out)
    path = os.path.join("tests", "data", "reference", "distrib_skewt_logpdf.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "K", "a", "b", "c", "d", "hi", "lo", "mu", "sigma", "alpha", "nu"])
        w.writerows(rows)
    print("%s: %d rows, %d dropped" % (path, len(rows), dropped))


def check_all():
    """the quadrature of the tail against the incomplete beta, at the (w, M) of the whole grid"""
    worst = 0.0
    for (mu, sigma, alpha, nu) in PARAMS:
        for zz in ZS:
            z = mp.mpf(zz)
            m = mp.mpf(nu) + 1
            w = mp.mpf(alpha) * z * mp.sqrt(m / (mp.mpf(nu) + z * z))
            try:
                e = check_tail(w, m)
            except Exception as exc:
                print("  alpha %g nu %g z %g (w %g, m %g): %s: %s" %
                      (alpha, nu, float(zz), float(w), float(m), type(exc).__name__, exc))
                worst = float("inf")
                continue
            worst = max(worst, e)
            if e > 1e-30:
                print("  w %-12g m %-12g relative %8.1e  <-- above 1e-30" % (float(w), float(m), e))
    print("the tail quadrature against the incomplete beta: worst %.1e over %d points" %
          (worst, len(PARAMS) * len(ZS)))
    return worst <= 1e-30


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        sys.exit(0 if check_all() else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "logpdf":
        main(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        print(__doc__)
        sys.exit(2)
