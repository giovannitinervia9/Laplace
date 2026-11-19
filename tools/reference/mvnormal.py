"""mpmath references for the multivariate normal rectangle probabilities
(include/laplace/math/mvnormal.hpp).

Run with python -I. Writes tests/data/reference/<name>.csv: the arguments, then
hi, lo of log P (hexadecimal), as in elementary.py.

  bvn_rect   log Pr(a1 < X1 <= b1, a2 < X2 <= b2), unit variances, correlation r:
             columns a1, b1, a2, b2, r. Reference: the one-dimensional integral
             int_{a1}^{b1} phi(x) (Phi(u(x)) - Phi(l(x))) dx, l, u the
             conditional limits, by mpmath's tanh-sinh at 40 digits on pieces
             cut around the mode of the (log-concave) integrand at multiples of
             its width; repeated at 50 digits on a shifted set of cuts and
             kept only when the two agree to 1e-30 relative.
"""
import sys
import random
import multiprocessing
import mpmath as mp

OUT = "tests/data/reference"


def dd(v):
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return hi, lo


def Phi(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def interval(a, b):
    """Phi(b) - Phi(a) without cancellation in either tail"""
    if b <= 0:
        return Phi(b) - Phi(a)
    if a >= 0:
        return Phi(-a) - Phi(-b)
    return 1 - Phi(a) - Phi(-b)


def cuts_near(h, sign, shift):
    """points on the side `sign` of a far end h where the integrand changes scale"""
    s = max(1, abs(h))
    return [h + sign * (2 ** j) * shift / s for j in range(-4, 8)]


def bvn_rect(a1, b1, a2, b2, r, dps, shift):
    with mp.workdps(dps):
        a1, b1, a2, b2, r = (mp.mpf(v) if v not in ("-inf", "inf") else mp.mpf(v) for v in (a1, b1, a2, b2, r))
        s = mp.sqrt((1 - r) * (1 + r))

        def f(x):
            lo = (a2 - r * x) / s if a2 != -mp.inf else -mp.inf
            hi = (b2 - r * x) / s if b2 != mp.inf else mp.inf
            return mp.npdf(x) * interval(lo, hi)

        return log_integral(f, a1, b1, shift)


def log_integral(f, a, b, shift):
    """log int_a^b f for a log-concave positive f (a product of log-concave
    factors, Prekopa): the mode x* of log f by ternary search, its width from the
    curvature (or the slope at an end), cuts at x* -+ 2^j w, then tanh-sinh."""
    lo = a if mp.isfinite(a) else mp.mpf(-200)
    hi = b if mp.isfinite(b) else mp.mpf(200)
    g = lambda x: mp.log(f(x))
    x0, x1 = lo, hi
    for _ in range(400):
        m1 = x0 + (x1 - x0) / 3
        m2 = x1 - (x1 - x0) / 3
        if g(m1) < g(m2):
            x0 = m1
        else:
            x1 = m2
        if x1 - x0 < mp.mpf(10) ** (-mp.mp.dps // 2):
            break
    xs = (x0 + x1) / 2
    eps = mp.mpf(10) ** (-mp.mp.dps // 4)
    d2 = (g(xs + eps) - 2 * g(xs) + g(xs - eps)) / eps ** 2 if lo < xs - eps and xs + eps < hi else 0
    if d2 < 0:
        w = 1 / mp.sqrt(-d2)
    else:
        e = lo if xs - lo < hi - xs else hi
        d1 = abs(mp.diff(g, e + (eps if e == lo else -eps)))
        w = 1 / max(d1, mp.mpf(1e-3))
    pts = {xs}
    for j in range(-6, 12):
        pts.add(xs + (2 ** j) * shift * w)
        pts.add(xs - (2 ** j) * shift * w)
    inner = sorted(p for p in pts if lo < p < hi)
    nodes = [a if mp.isfinite(a) else -mp.inf] + inner + [b if mp.isfinite(b) else mp.inf]
    # the far pieces factor out the scale of f at the mode
    gm = g(xs)
    v = mp.quad(lambda x: mp.exp(g(x) - gm) if f(x) > 0 else 0, nodes, maxdegree=10)
    return gm + mp.log(v)


def bvn_rows(rng):
    rows = []
    hs = [-37, -30, -20, -12, -8, -5, -3, -1.5, -0.5, 0, 0.5, 1.5, 3, 5, 8]
    rs = [-0.9999, -0.999, -0.99, -0.95, -0.9, -0.7, -0.4, -0.1, 0, 0.1, 0.4, 0.7, 0.9, 0.925, 0.95, 0.99, 0.999, 0.9999]
    for h in hs:
        for k in hs:
            for r in rs:
                if rng.random() < 0.45:
                    rows.append(("-inf", h, "-inf", k, r))
    # upper and mixed orthants
    for _ in range(300):
        h, k, r = rng.choice(hs), rng.choice(hs), rng.choice(rs)
        kind = rng.randrange(3)
        if kind == 0:
            rows.append((h, "inf", k, "inf", r))
        elif kind == 1:
            rows.append(("-inf", h, k, "inf", r))
        else:
            rows.append((h, "inf", "-inf", k, r))
    # finite rectangles, narrow and wide
    for _ in range(500):
        c1, c2 = rng.uniform(-12, 6), rng.uniform(-12, 6)
        w1 = 10 ** rng.uniform(-6, 0.7)
        w2 = 10 ** rng.uniform(-6, 0.7)
        r = rng.choice(rs)
        rows.append((c1, c1 + w1, c2, c2 + w2, r))
    return rows


def fl(v):
    return -float("inf") if v == "-inf" else float("inf") if v == "inf" else float(v)


def bvn_row(args):
    args = tuple(fl(a) for a in args)
    margs = tuple("-inf" if a == -float("inf") else "inf" if a == float("inf") else a for a in args)
    v1 = bvn_rect(*margs, 40, 1)
    v2 = bvn_rect(*margs, 50, 1.37)
    with mp.workdps(50):
        if abs(v1 - v2) > mp.mpf("1e-30") * max(1, abs(v2)):
            print("disagree", args, mp.nstr(v1, 20), mp.nstr(v2, 20), file=sys.stderr)
            return None
        hi, lo = dd(v2)
    return ",".join(float.hex(a) for a in args) + f",{float.hex(hi)},{float.hex(lo)}"


KF = 10  # the largest dimension of the one-factor rows


def factor_rect(a, b, lam, dps, shift):
    """log Pr(a < X <= b), X_i = lam_i T + sqrt(1 - lam_i^2) E_i: the one-factor
    correlation r_ij = lam_i lam_j, as int phi(t) prod_i (conditional interval) dt,
    a log-concave integrand"""
    with mp.workdps(dps):
        lam = [mp.mpf(v) for v in lam]
        s = [mp.sqrt((1 - l) * (1 + l)) for l in lam]
        A = [mp.mpf(v) if v != -float("inf") else -mp.inf for v in a]
        B = [mp.mpf(v) if v != float("inf") else mp.inf for v in b]

        def f(t):
            v = mp.npdf(t)
            for i in range(len(lam)):
                lo = (A[i] - lam[i] * t) / s[i] if A[i] != -mp.inf else -mp.inf
                hi = (B[i] - lam[i] * t) / s[i] if B[i] != mp.inf else mp.inf
                v *= interval(lo, hi)
            return v

        return log_integral(f, -mp.inf, mp.inf, shift)


def factor_rows(rng):
    rows = []
    lams = [-0.99, -0.95, -0.9, -0.7, -0.4, 0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99]
    for _ in range(700):
        k = rng.randrange(3, KF + 1)
        lam = [rng.choice(lams) for _ in range(k)]
        kind = rng.randrange(4)
        a, b = [], []
        for i in range(k):
            if kind == 0:    # lower orthant, from the bulk to the far tail
                a.append(-float("inf")), b.append(rng.choice([-12, -6, -3, -1.5, -0.5, 0, 0.5, 1.5, 3]))
            elif kind == 1:  # mixed orthant
                h = rng.choice([-6, -3, -1, 0, 1, 3])
                if rng.random() < 0.5:
                    a.append(-float("inf")), b.append(h)
                else:
                    a.append(h), b.append(float("inf"))
            else:            # rectangles, wide and narrow
                c = rng.uniform(-6, 3)
                w = 10 ** rng.uniform(-4 if kind == 3 else -1, 0.7)
                a.append(c), b.append(c + w)
        rows.append((k, a, b, lam))
    return rows


def laml_rows(rng):
    """The orthants of the piecewise LAML of the lasso (note/penalties.md 5f): a
    one-factor correlation R with loadings l in [-c, c] (c = 0.5, 0.9, 0.99),
    standard deviations sig, t = lam * mean(sig) from 0.5 to 20, mode m zero or
    random; the orthant s in {-1, 1}^k is the lower orthant at
    a = s o (m - lam Sigma s) / sig with loadings s o l, deep in the tail at large t."""
    import numpy as np
    nrng = np.random.default_rng(20261119)
    rows = []
    for k in range(2, 7):
        for c in (0.5, 0.9, 0.99):
            l = nrng.uniform(-c, c, k)
            l[0] = c
            sig = nrng.uniform(0.5, 2.0, k)
            R = np.outer(l, l)
            np.fill_diagonal(R, 1.0)
            Sig = R * np.outer(sig, sig)
            for t in (0.5, 2.0, 5.0, 10.0, 20.0):
                lam = t / np.mean(sig)
                for mode in ("zero", "random"):
                    m = np.zeros(k) if mode == "zero" else sig * nrng.uniform(-1.5, 1.5, k)
                    signs = [np.array(s, float) for s in __import__("itertools").product((-1, 1), repeat=k)]
                    if len(signs) > 6:
                        pick = nrng.choice(len(signs), 6, replace=False)
                        signs = [signs[0], signs[-1]] + [signs[i] for i in pick[:4]]
                    for s in signs:
                        a = s * (m - lam * Sig @ s) / sig
                        rows.append((k, [-float("inf")] * k, [float(v) for v in a], [float(v) for v in s * l]))
    return rows


def factor_row(row):
    k, a, b, lam = row
    v1 = factor_rect(a, b, lam, 40, 1)
    v2 = factor_rect(a, b, lam, 50, 1.37)
    with mp.workdps(50):
        if abs(v1 - v2) > mp.mpf("1e-30") * max(1, abs(v2)):
            print("disagree", row, mp.nstr(v1, 20), mp.nstr(v2, 20), file=sys.stderr)
            return None
        hi, lo = dd(v2)
    pad = lambda v, f: list(v) + [f] * (KF - len(v))
    cols = [float(k)] + pad(a, 0.0) + pad(b, 0.0) + pad(lam, 0.0)
    return ",".join(float.hex(float(x)) for x in cols) + f",{float.hex(hi)},{float.hex(lo)}"


def tvn_general(a, b, R, dps, shift):
    """the trivariate rectangle for a general correlation R by two nested
    log-concave integrals over z1, z2 of X = L z (L the Cholesky factor of R),
    the third coordinate's conditional interval in closed form"""
    with mp.workdps(dps):
        L = mp.cholesky(mp.matrix(R))
        A = [mp.mpf(v) if v != -float("inf") else -mp.inf for v in a]
        B = [mp.mpf(v) if v != float("inf") else mp.inf for v in b]

        def lim(c, num, den):
            return c if not mp.isfinite(c) else (c - num) / den

        def inner(z1):
            def f(z2):
                m = L[2, 0] * z1 + L[2, 1] * z2
                return mp.npdf(z2) * interval(lim(A[2], m, L[2, 2]), lim(B[2], m, L[2, 2]))
            m2 = L[1, 0] * z1
            return log_integral(f, lim(A[1], m2, L[1, 1]), lim(B[1], m2, L[1, 1]), shift)

        def g(z1):
            return mp.npdf(z1) * mp.exp(inner(z1))

        return log_integral(g, lim(A[0], 0, L[0, 0]), lim(B[0], 0, L[0, 0]), shift)


def tvn_orthant_plackett(h, R, dps):
    """P(X <= h), X ~ N(0, R), k = 3, by Plackett's identity along R(t) = I + t (R - I):
    dP/dt = sum_{i<j} r_ij phi2(h_i, h_j; t r_ij) Phi((h_l - m_l)/s_l), the
    conditional of the third at X_i = h_i, X_j = h_j under R(t). Infinite h_i
    drop their terms (phi2 = 0 there). Returns P (not its log)."""
    with mp.workdps(dps):
        h = [mp.mpf(v) if mp.isfinite(v) else (mp.inf if v > 0 else -mp.inf) for v in h]
        if any(v == -mp.inf for v in h):
            return mp.mpf(0)
        P0 = mp.fprod(Phi(v) if mp.isfinite(v) else mp.mpf(1) for v in h)

        def dP(t):
            tot = mp.mpf(0)
            for (i, j, l) in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                if not (mp.isfinite(h[i]) and mp.isfinite(h[j])):
                    continue
                rij, ril, rjl = t * R[i][j], t * R[i][l], t * R[j][l]
                d = 1 - rij * rij
                q = (h[i] ** 2 - 2 * rij * h[i] * h[j] + h[j] ** 2) / d
                f2 = mp.exp(-q / 2) / (2 * mp.pi * mp.sqrt(d))
                if mp.isfinite(h[l]):
                    bi = (ril - rij * rjl) / d
                    bj = (rjl - rij * ril) / d
                    m = bi * h[i] + bj * h[j]
                    v = 1 - (bi * ril + bj * rjl)
                    c = Phi((h[l] - m) / mp.sqrt(v))
                else:
                    c = mp.mpf(1)
                tot += R[i][j] * f2 * c
            return tot

        return P0 + mp.quad(dP, [0, mp.mpf(1) / 2, mp.mpf(9) / 10, mp.mpf(99) / 100, 1])


def tvn_rect_plackett(a, b, R, dps):
    """the rectangle by inclusion-exclusion over its 8 corners, each a lower
    orthant by Plackett's identity, at a precision raised until two runs agree"""
    def run(d):
        with mp.workdps(d):
            tot = mp.mpf(0)
            for mask in range(8):
                h, sg, skip = [], 1, False
                for i in range(3):
                    if (mask >> i) & 1:
                        if a[i] == -float("inf"):
                            skip = True
                        h.append(a[i]),
                        sg = -sg
                    else:
                        h.append(b[i])
                if skip:
                    continue
                tot += sg * tvn_orthant_plackett(h, R, d)
            return tot
    d = dps
    prev = run(d)
    while True:
        d = 2 * d
        cur = run(d)
        with mp.workdps(d):
            if cur > 0 and abs(cur - prev) <= mp.mpf(10) ** (-30) * cur:
                return mp.log(cur)
        prev = cur
        if d > 320:
            return None


def tvn_rows(rng):
    rows = []
    while len(rows) < 160:
        r = [rng.choice([-0.999, -0.95, -0.8, -0.5, -0.2, 0.0, 0.3, 0.6, 0.9, 0.98, 0.999]) for _ in range(3)]
        R = [[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1]]
        if mp.det(mp.matrix(R)) < 1e-3:
            continue
        kind = rng.randrange(3)
        a, b = [], []
        for i in range(3):
            if kind == 0:
                a.append(-float("inf")), b.append(rng.choice([-8, -5, -3, -1.5, -0.5, 0.5, 2, 4]))
            elif kind == 1:
                h = rng.choice([-4, -1, 0, 1, 3])
                if rng.random() < 0.5:
                    a.append(-float("inf")), b.append(h)
                else:
                    a.append(h), b.append(float("inf"))
            else:
                c = rng.uniform(-4, 2)
                a.append(c), b.append(c + 10 ** rng.uniform(-2, 0.5))
        rows.append((a, b, r))
    return rows


def tvn_row(row):
    a, b, r = row
    R = [[1, r[0], r[1]], [r[0], 1, r[2]], [r[1], r[2], 1]]
    v = tvn_rect_plackett(a, b, R, 40)
    if v is None:
        return None
    hi, lo = dd(v)
    cols = a + b + r
    return ",".join(float.hex(float(x)) for x in cols) + f",{float.hex(hi)},{float.hex(lo)}"


KM = 6  # the largest dimension of the moment rows


def gl_nodes(n_deg):
    """Gauss-Legendre nodes and weights on [-1, 1] at the working precision"""
    g = mp.calculus.quadrature.GaussLegendre(mp.mp)
    return g.calc_nodes(n_deg, mp.mp.prec)


def factor_trunc_moments(a, b, lam, dps, n_deg):
    """log P and the cumulants of orders 1..4 (4 only for k <= 3) of T = X | a < X <= b
    for the one-factor correlation r_ij = lam_i lam_j: given the factor t the
    coordinates are independent truncated normals, whose moments are closed;
    E[g(T)] = int phi(t) prod_i D_i(t) E_t[g] dt / P by Gauss-Legendre on pieces
    cut around the mode of the log-concave weight; raw moments, then cumulants,
    at the working precision (dps 60, so that the cancellation of raw moments in
    the tail costs nothing in double)."""
    import itertools
    with mp.workdps(dps):
        k = len(lam)
        lam = [mp.mpf(v) for v in lam]
        s = [mp.sqrt((1 - l) * (1 + l)) for l in lam]
        A = [mp.mpf(v) if v != -float("inf") else -mp.inf for v in a]
        B = [mp.mpf(v) if v != float("inf") else mp.inf for v in b]
        order = 4 if k <= 3 else 2

        def cond(t):
            """per coordinate: D_i(t) and the raw moments 1..order of X_i given t, truncated"""
            out = []
            for i in range(k):
                m, sd = lam[i] * t, s[i]
                lo = (A[i] - m) / sd if A[i] != -mp.inf else -mp.inf
                hi = (B[i] - m) / sd if B[i] != mp.inf else mp.inf
                D = interval(lo, hi)
                # standard truncated raw moments by the recursion E[Z^j] = (j-1) E[Z^(j-2)] + (l^(j-1) phi(l) - u^(j-1) phi(u)) / D
                pl = mp.npdf(lo) if lo != -mp.inf else mp.mpf(0)
                pu = mp.npdf(hi) if hi != mp.inf else mp.mpf(0)
                Ez = [mp.mpf(1), (pl - pu) / D]
                for j in range(2, order + 1):
                    tl = lo ** (j - 1) * pl if lo != -mp.inf else mp.mpf(0)
                    tu = hi ** (j - 1) * pu if hi != mp.inf else mp.mpf(0)
                    Ez.append((j - 1) * Ez[j - 2] + (tl - tu) / D)
                # X = m + sd Z
                Ex = [sum(mp.binomial(j, q) * m ** (j - q) * sd ** q * Ez[q] for q in range(j + 1)) for j in range(order + 1)]
                out.append((D, Ex))
            return out

        def logw(t):
            return -t * t / 2 + sum(mp.log(D) for D, _ in cond(t))

        # the mode of the weight, then pieces at multiples of its width
        x0, x1 = mp.mpf(-60), mp.mpf(60)
        for _ in range(200):
            m1, m2 = x0 + (x1 - x0) / 3, x1 - (x1 - x0) / 3
            if logw(m1) < logw(m2):
                x0 = m1
            else:
                x1 = m2
        xs = (x0 + x1) / 2
        e = mp.mpf(10) ** -8
        d2 = (logw(xs + e) - 2 * logw(xs) + logw(xs - e)) / e ** 2
        wd = 1 / mp.sqrt(-d2) if d2 < 0 else mp.mpf(1)
        cuts = sorted(set([xs + q * wd for q in (-40, -20, -10, -6, -3, -1.5, -0.5, 0, 0.5, 1.5, 3, 6, 10, 20, 40)]))
        nodes = gl_nodes(n_deg)
        tuples = {o: list(itertools.combinations_with_replacement(range(k), o)) for o in range(1, order + 1)}
        sums = {o: [mp.mpf(0)] * len(tuples[o]) for o in tuples}
        lw0 = logw(xs)
        S0 = mp.mpf(0)
        for pa, pb in zip(cuts[:-1], cuts[1:]):
            h, c = (pb - pa) / 2, (pa + pb) / 2
            for (xn, wn) in nodes:
                t = c + h * xn
                cd = cond(t)
                wt = wn * h * mp.exp(-t * t / 2 + sum(mp.log(D) for D, _ in cd) - lw0)
                S0 += wt
                for o in tuples:
                    for q, tp in enumerate(tuples[o]):
                        # E_t[prod X] with repeated indices: product over distinct of E_t[X_i^count]
                        v = mp.mpf(1)
                        for i in set(tp):
                            v *= cd[i][1][tp.count(i)]
                        sums[o][q] += wt * v
        logP = lw0 + mp.log(S0) - mp.log(2 * mp.pi) / 2
        raw = {o: {tp: sums[o][q] / S0 for q, tp in enumerate(tuples[o])} for o in tuples}

        def rawm(tp):
            tp = tuple(sorted(tp))
            return mp.mpf(1) if not tp else raw[len(tp)][tp]

        # cumulants from raw moments over the set partitions
        def partitions(seq):
            if not seq:
                yield []
                return
            first, rest = seq[0], seq[1:]
            for p in partitions(rest):
                yield [[first]] + p
                for i in range(len(p)):
                    yield p[:i] + [[first] + p[i]] + p[i + 1:]

        cum = {}
        for o in tuples:
            cum[o] = []
            for tp in tuples[o]:
                v = mp.mpf(0)
                for p in partitions(list(range(o))):
                    nb = len(p)
                    term = (-1) ** (nb - 1) * mp.factorial(nb - 1)
                    for blk in p:
                        term *= rawm([tp[q] for q in blk])
                    v += term
                cum[o].append(v)
        return logP, cum


def trunc_rows(rng):
    rows = [r for r in laml_rows(rng) if r[0] <= KM]
    rng.shuffle(rows)
    rows = rows[:260]
    rows += [r for r in factor_rows(rng) if r[0] <= 5][:140]
    return rows


def trunc_row(row):
    k, a, b, lam = row
    lp1, c1 = factor_trunc_moments(a, b, lam, 60, 6)
    lp2, c2 = factor_trunc_moments(a, b, lam, 60, 7)
    with mp.workdps(60):
        bad = abs(lp1 - lp2) > mp.mpf(10) ** -28 * max(1, abs(lp2))
        for o in c1:
            for x, y in zip(c1[o], c2[o]):
                if abs(x - y) > mp.mpf(10) ** -24 * max(abs(y), mp.mpf(10) ** -300):
                    bad = True
        if bad:
            print("disagree", row, file=sys.stderr)
            return None
    pad = lambda v, f, n: list(v) + [f] * (n - len(v))
    vals = [float(v) for o in sorted(c2) for v in c2[o]]
    ncum = 6 + 21 + 56 + 126  # orders 1..4 at k = 6 at most
    cols = [float(k)] + pad(a, 0.0, KM) + pad(b, 0.0, KM) + pad(lam, 0.0, KM) + [float(lp2)] + pad(vals, 0.0, ncum)
    return ",".join(float.hex(float(x)) for x in cols)


def main(names):
    rng = random.Random(20261119)
    for name in names:
        if name == "mvn_trunc_moments":
            rows = trunc_rows(rng)
            with multiprocessing.Pool(8) as pool:
                out = [o for o in pool.map(trunc_row, rows, chunksize=2) if o is not None]
            with open(f"{OUT}/{name}.csv", "w") as fh:
                fh.write("k,a[6],b[6],lam[6],logP,cumulants by order then sorted tuple (orders 1-2, and 3-4 when k <= 3)\n")
                fh.write("\n".join(out) + "\n")
            print(name, len(out), "rows")
            continue
        if name in ("mvn_factor", "tvn_general", "mvn_laml"):
            rows = factor_rows(rng) if name == "mvn_factor" else laml_rows(rng) if name == "mvn_laml" else tvn_rows(rng)
            fn = tvn_row if name == "tvn_general" else factor_row
            with multiprocessing.Pool(8) as pool:
                out = [o for o in pool.map(fn, rows, chunksize=2) if o is not None]
            head = ("k," + ",".join(f"a{i}" for i in range(KF)) + "," + ",".join(f"b{i}" for i in range(KF)) + ","
                    + ",".join(f"l{i}" for i in range(KF)) + ",hi,lo") if name != "tvn_general" else \
                "a0,a1,a2,b0,b1,b2,r01,r02,r12,hi,lo"
            with open(f"{OUT}/{name}.csv", "w") as fh:
                fh.write(head + "\n" + "\n".join(out) + "\n")
            print(name, len(out), "rows")
            continue
        if name == "bvn_rect":
            rows = bvn_rows(rng)
            with multiprocessing.Pool(8) as pool:
                out = [o for o in pool.map(bvn_row, rows, chunksize=4) if o is not None]
            with open(f"{OUT}/{name}.csv", "w") as fh:
                fh.write("a1,b1,a2,b2,r,hi,lo\n")
                fh.write("\n".join(out) + "\n")
            print(name, len(out), "rows")


if __name__ == "__main__":
    main(sys.argv[1:] or ["bvn_rect"])
