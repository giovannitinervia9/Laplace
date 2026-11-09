"""Reference values of the derivatives of log P(a, x), log Q(a, x),
log I_x(a, b) and log(1 - I_x(a, b)) with respect to the shapes, orders 1 to 5,
computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/incomplete_deriv.py gamma|beta [n_points] [workers]

Files (format as in incomplete.py: the arguments, then hi, lo):
    log_gamma_p_da{k}.csv, log_gamma_q_da{k}.csv        a, x       d^k/da^k, k = 1..5
    log_ibeta_d{i}{j}.csv, log_ibetac_d{i}{j}.csv       a, b, x    d^(i+j)/da^i db^j, 1 <= i + j <= 5

Method. With v = log t (gamma) the incomplete integral of one side is
    J(a) = int t^(a-1) e^-t dt = int exp(a v - e^v) dv
over v < log x (lower) or v > log x (upper), and its derivatives in a are the
moments m_n = int v^n exp(a v - e^v) dv. For the beta, with u = logit t,
    J(a, b) = int t^a (1-t)^b du,  m_ij = int (log t)^i (log(1-t))^j t^a (1-t)^b du,
over u < logit x or u > logit x, with log t = -log1p(e^-u) and
log(1 - t) = -log1p(e^u) exact on the whole line. In these variables the
endpoint singularities of t^(a-1) and (1-t)^(b-1) become tails decaying like
e^(a v), e^(a u), e^(-b u), which tanh-sinh on a half-line reaches (statmodels7,
statmodels7 notes: a tanh-sinh reference in t lost the mass at the singularity).
The integrals are cut at the mode, at several of its widths on both sides, and
near the end point, at multiples of the scale of the integrand there.

The side integrated is the smaller one (x below or above the mean). Its
regularized value S = J / Gamma(a) (or J / B(a, b)) is expanded as a truncated
Taylor series in the shape increments h, from the moments and from the series
of log Gamma (polygamma); then log S and log(1 - S) are formed as series
(log(1 + u), exp(u) by their power series, u without constant term), and the
derivative is the coefficient times i! j!. Nothing cancels except in
log(1 - S) where S is near 1, which the working precision covers.

Checks. Every row is computed at two precisions (240 and 320 bits) with
different quadrature degrees, and kept only if the two agree to 2^-100
relative (the dropped rows are counted). The value S itself is compared with
mpmath's gammainc and betainc on every row (the upper beta side as
I_(1-x)(b, a)).
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


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


# ---------------------------------------------------------------- truncated bivariate series
# a series is a dict {(i, j): coefficient} with i + j <= KMAX

def s_mul(p, q):
    r = {}
    for (i1, j1), c1 in p.items():
        for (i2, j2), c2 in q.items():
            if i1 + i2 + j1 + j2 <= KMAX:
                key = (i1 + i2, j1 + j2)
                r[key] = r.get(key, 0) + c1 * c2
    return r


def s_add(p, q, sq=1):
    r = dict(p)
    for k, c in q.items():
        r[k] = r.get(k, 0) + sq * c
    return r


def s_scale(p, c):
    return {k: c * v for k, v in p.items()}


def s_compose(u, coeffs):
    """sum_n coeffs[n] u^n, u without constant term, n = 0..KMAX"""
    r = {(0, 0): coeffs[0]}
    pw = {(0, 0): mpmath.mpf(1)}
    for n in range(1, KMAX + 1):
        pw = s_mul(pw, u)
        r = s_add(r, s_scale(pw, coeffs[n]))
    return r


def s_log(p):
    """log of a series with positive constant term"""
    c0 = p[(0, 0)]
    u = {k: v / c0 for k, v in p.items() if k != (0, 0)}
    coeffs = [mpmath.log(c0)] + [mpmath.mpf((-1) ** (n + 1)) / n for n in range(1, KMAX + 1)]
    return s_compose(u, coeffs)


def s_exp(p):
    c0 = p.get((0, 0), 0)
    u = {k: v for k, v in p.items() if k != (0, 0)}
    e = mpmath.exp(c0)
    coeffs = [e / mpmath.factorial(n) for n in range(KMAX + 1)]
    return s_compose(u, coeffs)


def derivatives(series):
    return {k: v * mpmath.factorial(k[0]) * mpmath.factorial(k[1]) for k, v in series.items() if k != (0, 0)}


def lgamma_series(z, which):
    """log Gamma(z + h) as a series in h_1 (which = 0), h_2 (1), or h_1 + h_2 (2)"""
    r = {(0, 0): mpmath.loggamma(z)}
    for n in range(1, KMAX + 1):
        c = mpmath.polygamma(n - 1, z) / mpmath.factorial(n)
        if which == 0:
            r[(n, 0)] = c
        elif which == 1:
            r[(0, n)] = c
        else:
            for i in range(n + 1):
                r[(i, n - i)] = r.get((i, n - i), 0) + c * mpmath.binomial(n, i)
    return r


# ---------------------------------------------------------------- quadrature

def cut_points(lo, hi, centre, wl, wr, end, w_end):
    """breakpoints in (lo, hi): around the mode and before the finite end"""
    pts = {centre}
    for k in (0.25, 1, 4, 16, 64, 256, 1024):
        pts.add(centre - k * wl)
        pts.add(centre + k * wr)
    if end is not None:
        for k in (0.25, 1, 4, 16, 64):
            pts.add(end - k * w_end if end == hi else end + k * w_end)
    inside = sorted(p for p in pts if lo < p < hi)
    return [lo] + inside + [hi]


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
            scale = max(max(abs(t) for t in est), mpmath.mpf(2) ** -64)
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


# ---------------------------------------------------------------- gamma

def gamma_row(a, x, prec, degree):
    with mpmath.workprec(prec):
        A, X = mpmath.mpf(a), mpmath.mpf(x)
        lx = mpmath.log(X)
        lower = x < a
        c = mpmath.log(A)
        wl = 1 / A if a < 1 else 1 / mpmath.sqrt(A)
        wr = 1 if a < 1 else 1 / mpmath.sqrt(A)
        slope = abs(A - X)
        w_end = 1 / max(slope, mpmath.mpf(1) / 64)
        lo, hi = (-mpmath.inf, lx) if lower else (lx, mpmath.inf)
        pts = cut_points(lo, hi, c, wl, wr, lx, w_end)
        # scale the integrand by exp(-a v_ref + e^v_ref) at the larger of the
        # mode and the end point inside the side, so that it is of order one
        vref = lx if (lower and lx < c) or (not lower and lx > c) else c
        lref = A * vref - mpmath.exp(vref)
        cutoff = -(prec + 200)

        def F(v):
            expo = A * v - mpmath.exp(v) - lref
            if expo < cutoff:
                return [mpmath.mpf(0)] * (KMAX + 1)
            e = mpmath.exp(expo)
            out = [e]
            for _ in range(KMAX):
                out.append(out[-1] * v)
            return out

        m = integrate_all(F, pts, KMAX + 1)
        J = {(n, 0): m[n] / mpmath.factorial(n) for n in range(KMAX + 1)}
        logS = s_add(s_log(J), lgamma_series(A, 0), -1)
        logS[(0, 0)] += lref
        S = mpmath.exp(logS[(0, 0)])
        logB = s_log(s_add({(0, 0): mpmath.mpf(1)}, s_exp(logS), -1))
        check = mpmath.gammainc(A, 0, X, regularized=True) if lower else mpmath.gammainc(A, X, mpmath.inf, regularized=True)
        dS, dB = derivatives(logS), derivatives(logB)
        p_d, q_d = (dS, dB) if lower else (dB, dS)
        return [p_d[(k, 0)] for k in range(1, KMAX + 1)], [q_d[(k, 0)] for k in range(1, KMAX + 1)], S, check


def gamma_args(rng, n):
    out = []
    for i in range(n):
        a = log_uniform(rng, -10, 13)
        if i % 2 == 0:
            x = a + rng.uniform(-30.0, 30.0) * math.sqrt(a)
            if x <= 0:
                x = log_uniform(rng, -30, 0) * a
        else:
            x = log_uniform(rng, -30, 14)
        out.append((a, x))
    return out


# ---------------------------------------------------------------- beta

def beta_row(a, b, x, prec, degree):
    with mpmath.workprec(prec):
        A, B, X = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(x)
        ux = mpmath.log(X) - mpmath.log1p(-X)
        lower = x < a / (a + b)
        c = mpmath.log(A) - mpmath.log(B)
        w = mpmath.sqrt((A + B) / (A * B))
        wl = max(w, 1 / A) if a < 1 else w
        wr = max(w, 1 / B) if b < 1 else w
        slope = abs(A - (A + B) * X)
        w_end = 1 / max(slope, mpmath.mpf(1) / 64)
        lo, hi = (-mpmath.inf, ux) if lower else (ux, mpmath.inf)
        pts = cut_points(lo, hi, c, wl, wr, ux, w_end)

        def logs(u):
            return -mpmath.log1p(mpmath.exp(-u)), -mpmath.log1p(mpmath.exp(u))

        uref = ux if (lower and ux < c) or (not lower and ux > c) else c
        l1, l2 = logs(uref)
        lref = A * l1 + B * l2
        keys = [(i, j) for i in range(KMAX + 1) for j in range(KMAX + 1 - i)]

        cutoff = -(prec + 200)

        def F(u):
            p, q = logs(u)
            expo = A * p + B * q - lref
            if expo < cutoff:
                return [mpmath.mpf(0)] * len(keys)
            e = mpmath.exp(expo)
            pp = [mpmath.mpf(1)]
            qq = [mpmath.mpf(1)]
            for _ in range(KMAX):
                pp.append(pp[-1] * p)
                qq.append(qq[-1] * q)
            return [e * pp[i] * qq[j] for i, j in keys]

        vals = integrate_all(F, pts, len(keys))
        J = {k: v / (mpmath.factorial(k[0]) * mpmath.factorial(k[1])) for k, v in zip(keys, vals)}
        lbeta = s_add(s_add(lgamma_series(A, 0), lgamma_series(B, 1)), lgamma_series(A + B, 2), -1)
        logS = s_add(s_log(J), lbeta, -1)
        logS[(0, 0)] += lref
        S = mpmath.exp(logS[(0, 0)])
        logBig = s_log(s_add({(0, 0): mpmath.mpf(1)}, s_exp(logS), -1))
        # the upper side as I_(1-x)(b, a): mpmath's betainc(a, b, x, 1) forms 1 - I
        # and returned 0 for 2.7e-280 at (a, b, x) = (0.0153, 2632, 0.214)
        check = mpmath.betainc(A, B, 0, X, regularized=True) if lower else mpmath.betainc(B, A, 0, 1 - X, regularized=True)
        dS, dB = derivatives(logS), derivatives(logBig)
        i_d, c_d = (dS, dB) if lower else (dB, dS)
        return i_d, c_d, S, check


def beta_args(rng, n):
    out = []
    for i in range(n):
        a, b = log_uniform(rng, -8, 11), log_uniform(rng, -8, 11)
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        k = i % 4
        if k == 0:
            x = rng.random()
        elif k == 1:
            x = m + rng.uniform(-30.0, 30.0) * sd
        elif k == 2:
            x = log_uniform(rng, -60, -1)
        else:
            x = 1.0 - log_uniform(rng, -53, -1)
        if 0.0 < x < 1.0:
            out.append((a, b, x))
    return out


# ---------------------------------------------------------------- driver

def agree(u, v):
    return abs(u - v) <= mpmath.mpf(2) ** -100 * abs(v) or u == v


def work_gamma(args):
    a, x = args
    try:
        p1, q1, S1, chk = gamma_row(a, x, 240, 8)
        p2, q2, S2, _ = gamma_row(a, x, 320, 10)
    except Exception as e:  # noqa: BLE001
        return ("error", (a, x), repr(e))
    ok = all(agree(u, v) for u, v in zip(p1 + q1, p2 + q2))
    sok = abs(S2 - chk) <= mpmath.mpf(2) ** -90 * abs(chk)
    return ("ok" if ok and sok else ("bad_s" if not sok else "bad"), (a, x), (p2, q2))


def work_beta(args):
    a, b, x = args
    try:
        i1, c1, S1, chk = beta_row(a, b, x, 240, 8)
        i2, c2, S2, _ = beta_row(a, b, x, 320, 10)
    except Exception as e:  # noqa: BLE001
        return ("error", (a, b, x), repr(e))
    keys = sorted(i2)
    ok = all(agree(i1[k], i2[k]) and agree(c1[k], c2[k]) for k in keys)
    sok = abs(S2 - chk) <= mpmath.mpf(2) ** -90 * abs(chk)
    return ("ok" if ok and sok else ("bad_s" if not sok else "bad"), (a, b, x), (i2, c2))


def write(path, header, rows):
    with path.open("w", newline="\n") as fh:
        fh.write(header + ",hi,lo\n")
        for args, v in rows:
            hi, lo = dd(v)
            fh.write(",".join(float.hex(float(t)) for t in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
    print(f"{path}: {len(rows)} rows", flush=True)


def main():
    kind = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (1500 if kind == "gamma" else 600)
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rng = random.Random(f"{SEED}-{kind}")
    args = gamma_args(rng, n) if kind == "gamma" else beta_args(rng, n)
    with Pool(workers) as pool:
        res = pool.map(work_gamma if kind == "gamma" else work_beta, args, chunksize=4)
    counts = {}
    for r in res:
        counts[r[0]] = counts.get(r[0], 0) + 1
        if r[0] != "ok" and counts[r[0]] <= 5:
            print("dropped", r[0], r[1], r[2] if r[0] == "error" else "", flush=True)
    print("counts", counts, flush=True)
    good = [r for r in res if r[0] == "ok"]
    OUT.mkdir(parents=True, exist_ok=True)
    if kind == "gamma":
        for k in range(1, KMAX + 1):
            write(OUT / f"log_gamma_p_da{k}.csv", "a,x", [(r[1], r[2][0][k - 1]) for r in good])
            write(OUT / f"log_gamma_q_da{k}.csv", "a,x", [(r[1], r[2][1][k - 1]) for r in good])
    else:
        for i in range(KMAX + 1):
            for j in range(KMAX + 1 - i):
                if i + j == 0:
                    continue
                write(OUT / f"log_ibeta_d{i}{j}.csv", "a,b,x", [(r[1], r[2][0][(i, j)]) for r in good])
                write(OUT / f"log_ibetac_d{i}{j}.csv", "a,b,x", [(r[1], r[2][1][(i, j)]) for r in good])


if __name__ == "__main__":
    main()
