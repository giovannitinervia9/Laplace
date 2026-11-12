"""Reference values for the scalar bijectors without parameters
(include/laplace/bijector/scalar.hpp), computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/bijector_scalar.py [map ...] [--workers N]

One file per map, tests/data/reference/bijector_<map>.csv, with three
arguments and the value as hi, lo (hex doubles, as in incomplete.py):
    x       the point (v for the forward map, u for the inverse);
    which   0 forward g, 1 inverse h, 2 forward log-Jacobian log|g'|,
            3 inverse log-Jacobian log|h'|, 4 the distance b - h(u) to the
            upper bound b of the range of h, 5 the distance h(u) - a to its
            lower bound a (maps whose range is (0, 1) or (-1, 1); order 0
            only, evaluated as written, 1 - h, with 1200 more bits);
    k       the order of the derivative, 0 the value itself; order 7 is
            not tested, it sets the scale of the error of order 6 (see the test).
Functions that are constant in x (a log-Jacobian of order 2 of probit, the
orders above 2 of the inverse of sqrt, ...) have no rows: the test checks them
exactly.

Method. Every map is written in SymPy by its definition (logit as
log(v/(1 - v)), its inverse as 1/(1 + exp(-u)), ...), differentiated
symbolically, and evaluated with mpmath; nothing here uses the closed forms of
the header. The exception is the forward probit z = Phi^-1(v): z is found by
Newton's method on Phi(z) = v (on Phi(-z) = 1 - v above 1/2), its derivatives
by reversion of the Taylor series of Phi at z, and the forward log-Jacobian
z^2/2 + log sqrt(2 pi) by composition of that series. The working precision
grows with the point (bits for the distance to a bound, for exp(|x|) and for
log2|x|), and every row is computed at two precisions, 120 bits apart, and kept
only if the two agree to 2^-100 relative; rows whose value is zero, not finite or
below 2^-1022 in magnitude are dropped (where values under- or overflow is
the subject of the clamping study).
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261112
KMAX = 7  # order 7 is written as the scale of the error of order 6 (the test)
TINY = 2.0**-1022
X = sp.Symbol("x", real=True)


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mp.mpf(hi))


def hexs(x):
    return float(x).hex()


# map name: (forward g(x), inverse h(x), decreasing, domain sampler, image sampler)
def unit_points(rng, n):
    pts = [0.5, 0.25, 0.75]
    for _ in range(n):
        e = rng.randint(2, 1000)
        pts.append(math.ldexp(1.0 + rng.random(), -e))
    for _ in range(n):
        e = rng.randint(2, 53)
        pts.append(1.0 - math.ldexp(1.0 + rng.random(), -e - 1))
    pts += [rng.uniform(0.02, 0.98) for _ in range(n)]
    return pts


def sym_unit_points(rng, n):  # (-1, 1)
    pts = [0.0, 0.5, -0.5]
    for _ in range(n):
        e = rng.randint(2, 1000)
        pts.append(rng.choice([-1, 1]) * math.ldexp(1.0 + rng.random(), -e))
    for _ in range(n):
        e = rng.randint(2, 53)
        pts.append(rng.choice([-1, 1]) * (1.0 - math.ldexp(1.0 + rng.random(), -e - 1)))
    pts += [rng.uniform(-0.98, 0.98) for _ in range(n)]
    return pts


def real_points(rng, n, emax):
    pts = [0.0, 1.0, -1.0]
    for _ in range(2 * n):
        e = rng.randint(-30, emax)
        pts.append(rng.choice([-1, 1]) * math.ldexp(1.0 + rng.random(), e))
    pts += [rng.uniform(-8, 8) for _ in range(n)]
    return pts


def pos_points(rng, n):
    pts = [1.0, 0.5, 2.0]
    for _ in range(3 * n):
        pts.append(math.ldexp(1.0 + rng.random(), rng.randint(-1000, 1000)))
    return pts


MAPS = {
    "log": (sp.log(X), sp.exp(X), False, pos_points, lambda r, n: real_points(r, n, 10)),
    "logit": (sp.log(X / (1 - X)), 1 / (1 + sp.exp(-X)), False, unit_points, lambda r, n: real_points(r, n, 10)),
    "probit": (None, (1 + sp.erf(X / sp.sqrt(2))) / 2, False, unit_points, lambda r, n: real_points(r, n, 5)),
    "cauchit": (sp.tan(sp.pi * (X - sp.Rational(1, 2))), sp.Rational(1, 2) + sp.atan(X) / sp.pi, False,
                unit_points, lambda r, n: real_points(r, n, 1000)),
    "cloglog": (sp.log(-sp.log(1 - X)), 1 - sp.exp(-sp.exp(X)), False, unit_points,
                lambda r, n: real_points(r, n, 10)),
    "loglog": (-sp.log(-sp.log(X)), sp.exp(-sp.exp(-X)), False, unit_points, lambda r, n: real_points(r, n, 10)),
    "inverse": (1 / X, 1 / X, True, pos_points, pos_points),
    "inverse_sq": (X**-2, X ** sp.Rational(-1, 2), True, pos_points, pos_points),
    "sqrt": (sp.sqrt(X), X**2, False, pos_points, pos_points),
    "atanh": (sp.atanh(X), sp.tanh(X), False, sym_unit_points, lambda r, n: real_points(r, n, 10)),
    "asinh": (sp.asinh(X), sp.sinh(X), False, lambda r, n: real_points(r, n, 1000),
              lambda r, n: real_points(r, n, 10)),
}


def derivative_table(expr, decreasing):
    """[(which, k, lambdified or None if constant)] for an expression of X."""
    out = {}
    d = expr
    for k in range(KMAX + 1):
        out[k] = d
        d = sp.diff(d, X)
    return out


# finite bounds of the range of h, for the distances (which 4, 5)
BOUNDS = {"logit": (0, 1), "probit": (0, 1), "cauchit": (0, 1), "cloglog": (0, 1), "loglog": (0, 1),
          "atanh": (-1, 1)}


def build(name):
    g, h, dec, _, _ = MAPS[name]
    table = {}
    for which, expr in ((0, g), (1, h)):
        if expr is None:
            continue
        der = derivative_table(expr, dec)
        for k, e in der.items():
            table[(which, k)] = e
        d1 = der[1]
        lj = sp.log(-d1) if dec else sp.log(d1)
        lder = derivative_table(lj, dec)
        for k, e in lder.items():
            table[(which + 2, k)] = e
    funcs = {}
    for key, e in table.items():
        if not e.free_symbols:
            continue  # constant: checked exactly by the test
        funcs[key] = sp.lambdify(X, e, modules="mpmath")
    if name in BOUNDS:  # distances of h(u) to the bounds of its range, written as 1 - h
        lo, hi = BOUNDS[name]
        funcs[(4, 0)] = sp.lambdify(X, hi - h, modules="mpmath")
        funcs[(5, 0)] = sp.lambdify(X, h - lo, modules="mpmath")
    if g is None:  # probit forward: series reversion
        for k in range(KMAX + 1):
            funcs[(0, k)] = ("probit", k)
            if k >= 0:
                funcs[(2, k)] = ("probit_lj", k)
    return funcs


def extra_bits(x, which):
    ax = abs(x)
    b = 0
    if ax > 0:
        b += int(2 * abs(math.log2(ax)))
    if ax < 3000:
        b += int(3 * ax)  # exp(2|x|): 2 log2(e) |x| bits
    if which in (0, 2) and 0 < ax < 1:
        b += int(-math.log2(1 - ax)) if ax < 1 else 0
    if which in (4, 5):
        b += 1200  # bound - h(u) cancels down to 2^-1022 (smaller values are dropped)
    return b + 64


# ---- probit forward by Newton and series reversion
def probit_z(v):
    v = mp.mpf(v)
    if v < mp.mpf(0.5):
        target, sign = v, -1
    else:
        target, sign = 1 - v, 1
    # z0 for Phi(-|z|) = target
    z = -mp.sqrt(-2 * mp.log(target)) if target < mp.mpf("1e-3") else mp.mpf(float(mp.sqrt(2) * mp.erfinv(2 * target - 1)))
    for _ in range(200):
        f = mp.ncdf(z) - target
        dz = f / mp.npdf(z)
        z -= dz
        if abs(dz) <= abs(z) * mp.mpf(2) ** (-mp.mp.prec + 8) + mp.mpf(2) ** (-mp.mp.prec):
            break
    return -z if sign > 0 else z


def series_mul(a, b, n):
    c = [mp.mpf(0)] * (n + 1)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            c[i + j] += a[i] * b[j]
    return c


def probit_series(v):
    """Taylor coefficients of z(v + t) and of z^2/2, to order KMAX."""
    z = probit_z(v)
    n = KMAX
    # a_j = Phi^(j)(z)/j! = (-1)^(j-1) He_(j-1)(z) phi(z)/j!
    he = [mp.mpf(1), z]
    for m in range(2, n):
        he.append(z * he[-1] - (m - 1) * he[-2])
    a = [mp.mpf(0)] + [(-1) ** (j - 1) * he[j - 1] * mp.npdf(z) / mp.factorial(j) for j in range(1, n + 1)]
    # reversion: b with a(b(t)) = t, b_0 = 0
    b = [mp.mpf(0)] * (n + 1)
    b[1] = 1 / a[1]
    for m in range(2, n + 1):
        # coefficient of t^m in sum_j a_j b(t)^j with b_m = 0, then solve a_1 b_m + that = 0
        acc = mp.mpf(0)
        p = b[:]  # b(t)^1
        for j in range(2, m + 1):
            p = series_mul(p, b, n)
            acc += a[j] * p[m]
        b[m] = -acc / a[1]
    zs = [z] + b[1:]
    sq = series_mul(zs, zs, n)
    lj = [x / 2 for x in sq]
    lj[0] += mp.log(mp.sqrt(2 * mp.pi))
    return zs, lj


def evaluate(fn, x):
    if isinstance(fn, tuple):
        zs, lj = probit_series(x)
        k = fn[1]
        c = zs[k] if fn[0] == "probit" else lj[k]
        return c * mp.factorial(k)
    return fn(mp.mpf(x))


def row(args):
    name, which, k, x = args
    funcs = build_cached(name)
    fn = funcs[(which, k)]
    base = 200 + extra_bits(x, which)
    vals = []
    for prec in (base, base + 120):
        with mp.workprec(prec):
            try:
                vals.append(evaluate(fn, x))
            except (ZeroDivisionError, ValueError, OverflowError):
                return None
    v1, v2 = vals
    if not (mp.isfinite(v1) and mp.isfinite(v2)) or mp.im(v1) != 0:
        return None
    v1, v2 = mp.re(v1), mp.re(v2)
    if v2 == 0:
        return None  # an exact zero at both precisions may be a cancellation; not kept
    elif abs(v1 - v2) > abs(v2) * mp.mpf(2) ** -100:
        return "disagree"
    with mp.workprec(base + 120):
        hi, lo = dd(v2)
    if not math.isfinite(hi) or (hi != 0 and abs(hi) < TINY):
        return None
    return f"{hexs(x)},{hexs(which)},{hexs(k)},{hexs(hi)},{hexs(lo)}"


_cache = {}


def build_cached(name):
    if name not in _cache:
        _cache[name] = build(name)
    return _cache[name]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = 6
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
        args = [a for a in args if a != str(workers)]
    names = args or list(MAPS)
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for name in names:
            _, _, _, dom, img = MAPS[name]
            funcs = build_cached(name)
            pts_dom = sorted(set(dom(rng, 60)))
            pts_img = sorted(set(img(rng, 60)))
            jobs = []
            for (which, k) in sorted(funcs):
                pts = pts_dom if which in (0, 2) else pts_img
                jobs += [(name, which, k, x) for x in pts]
            res = pool.map(row, jobs, chunksize=8)
            rows = [r for r in res if r and r != "disagree"]
            bad = sum(1 for r in res if r == "disagree")
            path = OUT / f"bijector_{name}.csv"
            with path.open("w", encoding="utf-8") as fh:
                fh.write("x,which,k,hi,lo\n")
                fh.write("\n".join(rows) + "\n")
            print(f"{name}: {len(rows)} rows, {len(jobs) - len(rows) - bad} dropped (not finite or subnormal), "
                  f"{bad} dropped (precisions disagree)", flush=True)


if __name__ == "__main__":
    main()
