"""Reference values for the Student t distribution function with real degrees
of freedom, Owen's T function and the skew normal distribution function, with
their derivatives, computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/student_owen.py student|sn|owen [n_points] [workers]

Files (format as in incomplete.py: the arguments, then hi, lo):
    log_student_cdf.csv, log_student_cdf_d{i}{j}.csv   z, nu      log T_nu(z) and d^(i+j)/dz^i dnu^j, 1 <= i + j <= 5
    log_sn_cdf.csv, log_sn_cdf_d{i}{j}.csv             z, alpha   log F(z; alpha) of the standard skew normal, likewise in (z, alpha)
    owens_t.csv                                         h, a       T(h, a)

The survival functions need no files of their own: 1 - T_nu(z) = T_nu(-z) and
1 - F(z; alpha) = F(-z; -alpha).

Method. T_nu(z) = I_x(nu/2, 1/2) / 2 for z < 0, x = nu/(nu + z^2), with the side
of the incomplete beta that does not cancel (x <= 1/2 directly, otherwise
1 - I_(1-x)(1/2, nu/2), whose value is of order one), and log T = log1p(-T(-z))
for z > 0. The skew normal F(z; alpha) = 2 int_(-inf)^z phi(t) Phi(alpha t) dt
by quadrature in u = z - t (a positive integrand), and log F = log1p(-S) with
S = 2 int_z^inf phi(t) Phi(alpha t) dt for z > 0. Owen's
T(h, a) = (1/(2 pi)) int_0^a exp(-h^2 (1 + s^2)/2) / (1 + s^2) ds by quadrature,
cut at multiples of 1/|h| where the integrand decays like a Gaussian.
Derivatives of the Student cdf by mpmath.diff (finite differences at raised
precision on an analytic function). For the skew normal every derivative of F
of order one or more is elementary, dF/dz = 2 phi(z) Phi(alpha z) and
dF/dalpha = -exp(-(1 + alpha^2) z^2 / 2) / (pi (1 + alpha^2)), so the
quadrature gives only the value; the derivatives of F come from mpmath.diff of
these two, and those of log F from the bivariate Taylor series of F composed
with log (the constant term as log1p(-S) for z > 0, where F rounds to 1). The quadratures use the double-exponential rule of incomplete_deriv.py
(mpmath.quad did not reach the tolerance on these integrands). Every row is
computed at two precisions (280 and 360 bits for the bivariate functions,
whose high orders lose many digits in the composition with log; 200 and 280
for Owen's T) and kept only if
the two agree to 2^-100 relative (dropped rows are counted).
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261109
KMAX = 5
KEYS = [(i, k - i) for k in range(1, KMAX + 1) for i in range(k, -1, -1)]


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


# ---------------------------------------------------------------- quadrature (as in incomplete_deriv.py)

def _piece(F, lo, hi, nf, tol):
    """int_lo^hi F(v) dv for a vector-valued F (nf components), by the
    double-exponential rule: tanh-sinh on a finite piece, exp-sinh on a
    half-line; the step is halved until every component changes by less than
    tol times the largest of them."""
    halfpi = mpmath.pi / 2
    if lo == -mpmath.inf or hi == mpmath.inf:
        end, sign = (hi, -1) if lo == -mpmath.inf else (lo, 1)

        def node(s):
            e = mpmath.exp(halfpi * mpmath.sinh(s))
            return end + sign * e, halfpi * mpmath.cosh(s) * e
    else:
        c, d = (lo + hi) / 2, (hi - lo) / 2

        def node(s):
            u = halfpi * mpmath.sinh(s)
            ch = mpmath.cosh(u)
            return c + d * mpmath.tanh(u), d * halfpi * mpmath.cosh(s) / (ch * ch)

    # half-lines reach 1e7 from their end (the slowest tail, e^(a v) with
    # a = 2^-10, is below 2^-400 there); finite pieces s in [-6, 6]
    # on a half-line s runs from -6 (distance e^-316 from the end) up to the cap
    half = lo == -mpmath.inf or hi == mpmath.inf
    smin = -6
    smax = mpmath.asinh(mpmath.log(10 ** 7) / halfpi) if half else 6
    total = [mpmath.mpf(0)] * nf

    def add(s):
        v, w = node(s)
        if not (lo < v < hi) and lo != v and hi != v:
            return
        if not mpmath.isfinite(v):
            return
        f = F(v)
        for i in range(nf):
            total[i] += w * f[i]

    h = mpmath.mpf(1) / 2
    k = 0
    while True:
        s = smin + (h if k > 0 else 0)
        step = 2 * h if k > 0 else h
        while s <= smax:
            add(s)
            s += step
        est = [t * h for t in total]
        if k > 0:
            # relative to the piece, or absolute below 2^-64 (the integrand is
            # scaled to order one at its peak, so a piece that small does not count)
            # (no absolute floor here: the values themselves can be as small as e^-800)
            scale = max(abs(t) for t in est)
            if scale == 0:
                return est
            if max(abs(u - w) for u, w in zip(est, prev)) <= tol * scale:
                return est
        prev = est
        if k >= 12:
            raise RuntimeError("quadrature did not converge")
        h /= 2
        k += 1


def integrate_all(F, pts, nf):
    tol = mpmath.mpf(2) ** (-(mpmath.mp.prec - 30))
    total = [mpmath.mpf(0)] * nf
    for lo, hi in zip(pts[:-1], pts[1:]):
        r = _piece(F, lo, hi, nf, tol)
        total = [t + u for t, u in zip(total, r)]
    return total



def de_quad(f, pts):
    return integrate_all(lambda v: [f(v)], pts, 1)[0]


# ---------------------------------------------------------------- functions

def student_tail(z, nu):
    """P(T < -|z|) = I_x(nu/2, 1/2) / 2"""
    x = nu / (nu + z * z)
    y = z * z / (nu + z * z)
    if x <= 0.5:
        return mpmath.betainc(nu / 2, mpmath.mpf(1) / 2, 0, x, regularized=True) / 2
    return (1 - mpmath.betainc(mpmath.mpf(1) / 2, nu / 2, 0, y, regularized=True)) / 2


def log_student_cdf(z, nu):
    if z < 0:
        return mpmath.log(student_tail(z, nu))
    return mpmath.log1p(-student_tail(z, nu))


def sn_lower(z, alpha):
    """F(z; alpha) for z <= 0 as a sum of positive terms: for alpha >= 0,
    F = int_alpha^inf exp(-(1 + b^2) z^2 / 2) / (pi (1 + b^2)) db (F(z; inf) = 0
    for z < 0, and dF/dalpha is minus the integrand); for alpha < 0,
    F = Phi(z) + 2 T(z, |alpha|)."""
    if alpha < 0:
        return mpmath.ncdf(z) + 2 * owens_t(z, -alpha)
    sc = 1 / max(abs(z), mpmath.mpf(1) / 16)
    pts = [alpha + k * sc for k in (0, mpmath.mpf(1) / 4, 1, 4, 16, 64)] + [mpmath.inf]

    def f(b):
        return mpmath.exp(-(1 + b * b) * z * z / 2) / (mpmath.pi * (1 + b * b))

    return de_quad(f, pts)


def sn_side(z, alpha, lower):
    """F(z; alpha) (lower) or 1 - F(z; alpha) = F(-z; -alpha), for the side with z <= 0"""
    return sn_lower(z, alpha) if lower else sn_lower(-z, -alpha)


def owens_t(h, a):
    if a == 0:
        return mpmath.mpf(0)
    sa = 1 if a > 0 else -1
    A = abs(a)
    sc = 1 / max(abs(h), mpmath.mpf(1) / 16)
    pts = [0] + [p for p in (sc / 4, sc, 4 * sc, 16 * sc) if p < A] + [A]

    def f(s):
        return mpmath.exp(-h * h * (1 + s * s) / 2) / (1 + s * s)

    return sa * de_quad(f, pts) / (2 * mpmath.pi)


# ---------------------------------------------------------------- rows

def s_mul(p, q):
    r = {}
    for (i1, j1), c1 in p.items():
        for (i2, j2), c2 in q.items():
            if i1 + i2 + j1 + j2 <= KMAX:
                key = (i1 + i2, j1 + j2)
                r[key] = r.get(key, 0) + c1 * c2
    return r


def sn_bivariate(z, alpha, prec):
    with mpmath.workprec(prec):
        Z, A = mpmath.mpf(z), mpmath.mpf(alpha)
        # d^j/dalpha^j Phi(alpha z) = z^j (-1)^(j-1) He_(j-1)(alpha z) phi(alpha z), j >= 1,
        # He the probabilists' Hermite polynomials; written out so that no finite
        # difference is taken of a value much larger than the derivative

        def he(n, w):
            h0, h1 = mpmath.mpf(1), w
            if n == 0:
                return h0
            for m in range(1, n):
                h0, h1 = h1, w * h1 - m * h0
            return h1

        def gz(j):
            if j == 0:
                return lambda u: 2 * mpmath.npdf(u) * mpmath.ncdf(A * u)
            return lambda u: 2 * mpmath.npdf(u) * u ** j * (-1) ** (j - 1) * he(j - 1, A * u) * mpmath.npdf(A * u)

        ga = lambda v: -mpmath.exp(-(1 + v * v) * Z * Z / 2) / (mpmath.pi * (1 + v * v))
        D = {}
        for (i, j) in KEYS:
            D[(i, j)] = mpmath.diff(gz(j), Z, i - 1) if i >= 1 else mpmath.diff(ga, A, j - 1)
        if z < 0:
            F0 = sn_side(Z, A, True)
            c0, logc0 = F0, mpmath.log(F0)
        else:
            S0 = sn_side(Z, A, False)
            c0, logc0 = 1 - S0, mpmath.log1p(-S0)
        # log(c0 + sum D_ij h^i k^j / (i! j!)) = log c0 + log(1 + u), u = (F - c0) / c0
        u = {k: D[k] / (mpmath.factorial(k[0]) * mpmath.factorial(k[1]) * c0) for k in KEYS}
        out = {(0, 0): logc0}
        pw = {(0, 0): mpmath.mpf(1)}
        for n in range(1, KMAX + 1):
            pw = s_mul(pw, u)
            for k, c in pw.items():
                out[k] = out.get(k, 0) + c * mpmath.mpf((-1) ** (n + 1)) / n
        for k in KEYS:
            out[k] = out.get(k, 0) * mpmath.factorial(k[0]) * mpmath.factorial(k[1])
        return out


def bivariate(fun, u, v, prec):
    with mpmath.workprec(prec):
        U, V = mpmath.mpf(u), mpmath.mpf(v)
        out = {(0, 0): fun(U, V)}
        for k in KEYS:
            out[k] = mpmath.diff(fun, (U, V), k)
        return out


def agree(p, q):
    return all(abs(p[k] - q[k]) <= mpmath.mpf(2) ** -100 * abs(q[k]) or p[k] == q[k] for k in q)


def work_bivariate(args):
    kind, u, v = args
    try:
        if kind == "student":
            r1 = bivariate(log_student_cdf, u, v, 280)
            r2 = bivariate(log_student_cdf, u, v, 360)
            if not agree(r1, r2):  # large nu in the deep tails: once more, higher
                r1 = bivariate(log_student_cdf, u, v, 440)
                r2 = bivariate(log_student_cdf, u, v, 560)
        else:
            r1 = sn_bivariate(u, v, 280)
            r2 = sn_bivariate(u, v, 360)
            # the z-derivative against the elementary density ratio, and the value against Phi - 2T
            with mpmath.workprec(280):
                Z, A = mpmath.mpf(u), mpmath.mpf(v)
                F = mpmath.exp(r2[(0, 0)])
                d1 = 2 * mpmath.npdf(Z) * mpmath.ncdf(A * Z) / F
                assert abs(d1 - r2[(1, 0)]) <= mpmath.mpf(2) ** -150 * abs(d1)
                if abs(u) < 3:
                    c = mpmath.ncdf(Z) - 2 * owens_t(Z, A)
                    # Phi - 2T cancels where F << Phi: compare on the scale of Phi
                    assert abs(c - F) <= mpmath.mpf(2) ** -150 * mpmath.ncdf(Z)
    except Exception as e:  # noqa: BLE001
        return ("error", (u, v), repr(e))
    return ("ok" if agree(r1, r2) else "bad", (u, v), r2)


def work_owen(args):
    h, a = args
    try:
        with mpmath.workprec(200):
            t1 = owens_t(mpmath.mpf(h), mpmath.mpf(a))
        with mpmath.workprec(280):
            t2 = owens_t(mpmath.mpf(h), mpmath.mpf(a))
            # check against the closed form at a = 1: T(h, 1) = Phi(h)(1 - Phi(h))/2
            if a == 1.0:
                c = mpmath.ncdf(h) * mpmath.ncdf(-h) / 2
                assert abs(t2 - c) <= mpmath.mpf(2) ** -150 * abs(c)
    except Exception as e:  # noqa: BLE001
        return ("error", (h, a), repr(e))
    ok = abs(t1 - t2) <= mpmath.mpf(2) ** -100 * abs(t2) or t1 == t2
    return ("ok" if ok else "bad", (h, a), {(0, 0): t2})


def student_args(rng, n):
    out = []
    for i in range(n):
        nu = log_uniform(rng, -4, 19)
        k = i % 3
        if k == 0:
            z = rng.uniform(-6.0, 6.0)
        elif k == 1:
            z = -log_uniform(rng, -20, 20)   # lower tail, up to |z| = 2^21
        else:
            z = log_uniform(rng, -20, 6)    # upper side
        out.append(("student", z, nu))
    return out


def sn_args(rng, n):
    out = []
    for i in range(n):
        alpha = math.copysign(log_uniform(rng, -8, 7), rng.random() - 0.5)
        k = i % 3
        if k == 0:
            z = rng.uniform(-6.0, 6.0)
        elif k == 1:
            z = -log_uniform(rng, -6, 5)     # lower tail, |z| up to 64
        else:
            z = log_uniform(rng, -6, 5)
        out.append(("sn", z, alpha))
    return out


def owen_args(rng, n):
    out = []
    for i in range(n):
        h = math.copysign(log_uniform(rng, -10, 5), rng.random() - 0.5)
        if i % 4 == 0:
            h = rng.uniform(-8.0, 8.0)
        a = math.copysign(log_uniform(rng, -10, 10), rng.random() - 0.5)
        out.append((h, a))
    out += [(0.0, 1.0), (1.0, 1.0), (-3.0, 1.0), (10.0, 1.0), (0.0, 0.5), (5.0, 100.0)]
    return out


def write(path, header, rows):
    with path.open("w", newline="\n") as fh:
        fh.write(header + ",hi,lo\n")
        for args, v in rows:
            hi, lo = dd(v)
            fh.write(",".join(float.hex(float(t)) for t in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
    print(f"{path}: {len(rows)} rows", flush=True)


def main():
    kind = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (3000 if kind == "owen" else 800)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rng = random.Random(f"{SEED}-{kind}")
    if kind == "owen":
        args, work = owen_args(rng, n), work_owen
    else:
        args = student_args(rng, n) if kind == "student" else sn_args(rng, n)
        work = work_bivariate
    with Pool(workers) as pool:
        res = pool.map(work, args, chunksize=4)
    counts = {}
    for r in res:
        counts[r[0]] = counts.get(r[0], 0) + 1
        if r[0] != "ok" and counts[r[0]] <= 5:
            print("dropped", r[0], r[1], r[2] if r[0] == "error" else "", flush=True)
    print("counts", counts, flush=True)
    good = [r for r in res if r[0] == "ok"]
    OUT.mkdir(parents=True, exist_ok=True)
    if kind == "owen":
        write(OUT / "owens_t.csv", "h,a", [(r[1], r[2][(0, 0)]) for r in good])
        return
    stem, header = ("log_student_cdf", "z,nu") if kind == "student" else ("log_sn_cdf", "z,alpha")
    write(OUT / f"{stem}.csv", header, [(r[1], r[2][(0, 0)]) for r in good])
    for k in KEYS:
        write(OUT / f"{stem}_d{k[0]}{k[1]}.csv", header, [(r[1], r[2][k]) for r in good])


if __name__ == "__main__":
    main()
