"""Reference values for the Dirichlet (dirichlet_distrib) and the multinomial
(multinomial_distrib), the two laws whose dense parameter mu lives on the simplex
through its additive log-ratio coordinates eta (reference category the last):
blocks of derivatives of the log-density in (eta, phi), derivatives in y for the
Dirichlet, and derivatives of the expected information, computed with mpmath at
60 digits from the definition.

Usage (from the repository root):  python -I tools/reference/distrib_simplex.py

File: tests/data/reference/distrib_simplex.csv, one row per entry, header
    law,d,x,theta,y,size,kind,J,A,idx,entry,hi,lo
law is 1 (Dirichlet) or 2 (multinomial). x is eta (d - 1 values), theta is phi
('-' for the multinomial), y the d stored values of the unit, size the number of
trials (multinomial; 0 for the Dirichlet); x, theta and y are float.hex joined by
';'. idx is the sorted tuple of theta indices (all 0, phi; '-' when empty).
kind 0 is a block of the log-density (log-mass) with J derivatives in y and A in
eta; kind 1, 2, 3 is a derivative with A in eta and idx in phi of the expected
information entry (eta, eta), (eta, phi), (phi, phi), J = 0. Entries: a tensor of
order n over m coordinates is ranked as in multivariate_core.hpp (order 0: 0;
order 1: i; order 2: i (i + 1)/2 + j for i >= j; order >= 3: the rank of the
sorted tuple in lexicographic order); a block is the Kronecker product, the first
factor (y for kind 0, the information entry for kinds 1-3) times the size of the
eta tensor plus the eta rank. hi is the nearest double to the value and lo the
nearest double to the remainder.

Definitions, mu_j = e^(eta_j) / (1 + sum_(b<d-1) e^(eta_b)) (eta_(d-1) = 0),
alpha_j = phi mu_j:
    Dirichlet:   l = lgamma(phi) - sum_j lgamma(alpha_j) + sum_j (alpha_j - 1) log y_j,
                 the y derivatives in y_1..y_(d-1) with y_d = y_d0 - sum_a (y_a - y_a0)
                 (the reference coordinates; y is used as given at the point);
    multinomial: l = lgamma(N + 1) - sum_j lgamma(y_j + 1) + sum_j y_j log mu_j.
Expected information, with E log y_j = psi(alpha_j) - psi(phi) (Dirichlet):
    I_ab = phi^2 sum_j psi_1(alpha_j) d_a mu_j d_b mu_j,
    I_a,phi = sum_j d_a mu_j alpha_j psi_1(alpha_j),
    I_phi,phi = sum_j mu_j^2 psi_1(alpha_j) - psi_1(phi),
    d_a mu_j = mu_j (delta_ja - mu_a);
multinomial I_ab = N mu_a (delta_ab - mu_b). The Dirichlet's closed forms are
checked once against E[score score'] by quadrature over the beta law (d = 2) at the
start. The blocks are mpmath.diff of these functions; every value is computed at
60 and at 75 digits, and the script stops if the two disagree by more than 1e-40
times (1 + |value|).

Points: d = 2, 3, 5; a moderate point (phi = 4.5) and hard ones: phi = 1e-3,
1e3, 1e7 (the Stirling cancellations of the mixed blocks), one tiny mean
(mu = 1e-9, alpha = 1e-9 and, at phi = 1e7, 1e-2), a mean near one
(1 - 1e-8), alpha = 1e-12; the multinomial at N = 1, 10, 1000, 1e6 with the same
means. y: a Dirichlet draw on the log scale (log G(alpha + 1) + log U / alpha);
multinomial counts N mu rounded so that they sum to N. Tuples: A = 0..6 in eta
with K = 0..6 - A in phi (a sample of entries for A >= 3), y blocks J = 1..4 with
A + K <= 2, the information to total order 4.
"""
import random
import sys
import time
from itertools import combinations_with_replacement
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp

DPS, DPS_CHECK, TOL = 60, 75, mp.mpf(10) ** -40
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_simplex.csv"
HEADER = "law,d,x,theta,y,size,kind,J,A,idx,entry,hi,lo"


def probs(eta):
    e = [mp.mpf(v) for v in eta] + [mp.mpf(0)]
    m = max(e)
    s = mp.fsum(mp.exp(v - m) for v in e)
    return [mp.exp(v - m) / s for v in e]


def dirichlet_l(eta, phi, y):
    mu = probs(eta)
    return mp.loggamma(phi) + mp.fsum(-mp.loggamma(phi * m) + (phi * m - 1) * mp.log(v) for m, v in zip(mu, y))


def multinomial_l(eta, y, n):
    mu = probs(eta)
    return mp.loggamma(n + 1) + mp.fsum(-mp.loggamma(v + 1) + v * mp.log(m) for m, v in zip(mu, y))


def dmu(mu, a, j):
    return mu[j] * ((1 if j == a else 0) - mu[a])


def dirichlet_info(kind, eta, phi, a, b):
    mu = probs(eta)
    al = [phi * m for m in mu]
    t1 = [mp.polygamma(1, v) for v in al]
    if kind == 1:
        return phi**2 * mp.fsum(t1[j] * dmu(mu, a, j) * dmu(mu, b, j) for j in range(len(mu)))
    if kind == 2:
        return mp.fsum(dmu(mu, a, j) * al[j] * t1[j] for j in range(len(mu)))
    return mp.fsum(m * m * t for m, t in zip(mu, t1)) - mp.polygamma(1, phi)


def multinomial_info(eta, n, a, b):
    mu = probs(eta)
    return n * mu[a] * ((1 if a == b else 0) - mu[b])


def orders(nv, positions):
    o = [0] * nv
    for p in positions:
        o[p] += 1
    return tuple(o)


def diff(f, x, positions):
    x = [mp.mpf(v) for v in x]
    if not positions:
        return f(x)
    return mp.diff(lambda *t: f(list(t)), x, orders(len(x), positions))


def checked(fn):
    with mp.workdps(DPS):
        v = fn()
    with mp.workdps(DPS_CHECK):
        w = fn()
    if abs(v - w) > TOL * (1 + abs(w)):
        raise SystemExit(f"disagreement {mp.nstr(abs(v - w), 5)} at value {mp.nstr(w, 20)}")
    return w


def hexs(v):
    return ";".join(float(x).hex() for x in v)


def split(v):
    hi = float(v)
    return hi, float(v - mp.mpf(hi))


def tensor_size(m, n):
    return len(list(combinations_with_replacement(range(m), n)))


def rank(t, m):
    if len(t) == 0:
        return 0
    if len(t) == 1:
        return t[0]
    if len(t) == 2:
        return t[1] * (t[1] + 1) // 2 + t[0]
    for r, u in enumerate(combinations_with_replacement(range(m), len(t))):
        if u == t:
            return r
    raise ValueError(t)


def entries(m, n, rng):
    allt = list(combinations_with_replacement(range(m), n))
    if n <= 2 or len(allt) <= 4:
        return allt
    return [allt[0], allt[-1]] + rng.sample(allt[1:-1], 2)


def mean_from(mu_head):
    """eta for a mean whose first coordinates are mu_head, the rest shared equally"""
    d = len(mu_head[1])
    head = [mp.mpf(v) for v in mu_head[0]]
    rest = (1 - mp.fsum(head)) / (d - len(head))
    mu = head + [rest] * (d - len(head))
    return [float(mp.log(v / mu[-1])) for v in mu[:-1]]


def dirichlet_draw(eta, phi, rng):
    mu = probs(eta)
    lg = [mp.log(mp.mpf(rng.gammavariate(float(phi * m) + 1, 1))) + mp.log(mp.mpf(rng.random())) / (phi * m) for m in mu]
    top = max(lg)
    e = [mp.exp(v - top) for v in lg]
    s = mp.fsum(e)
    return [max(float(v / s), 1e-300) for v in e]


def points():
    rng = random.Random(20261117)
    pts = []
    for d in (2, 3, 5):
        z = [0] * d
        means = [
            ("moderate", [float(rng.gauss(0, 0.7)) for _ in range(d - 1)]),
            ("tiny", mean_from(([1e-9], z))),
            ("near one", mean_from(([1 - 1e-8], z))),
        ]
        for name, eta in means:
            for phi in (4.5, 1e-3, 1e3, 1e7) if name == "moderate" else (1.0, 1e7):
                pts.append((1, d, eta, phi, dirichlet_draw(eta, phi, rng), 0))
            for n in (1, 10, 1000, 1e6) if name == "moderate" else (10, 1e6):
                mu = [float(v) for v in probs(eta)]
                y = [float(round(n * v)) for v in mu[:-1]]
                y.append(n - sum(y))
                if y[-1] < 0:
                    y[-1] = 0.0
                    y[0] = n - sum(y[1:])
                pts.append((2, d, eta, 0.0, y, n))
        eta12 = mean_from(([1e-12 / 1.0], z))
        pts.append((1, d, eta12, 1.0, dirichlet_draw(eta12, 1.0, rng), 0))
    return pts


def task(pt):
    law, d, eta, phi, y, n = pt
    m = d - 1
    rng = random.Random(1000 * law + 10 * d + int(abs(eta[0]) * 1000) + int(phi) % 997)
    head = f"{law},{d},{hexs(eta)},{float(phi).hex() if law == 1 else '-'},{hexs(y)},{float(n).hex() if law == 2 else '0x0.0p+0'}"
    rows = []

    def put(kind, J, A, K, ent, v):
        hi, lo = split(v)
        rows.append(f"{head},{kind},{J},{A},{';'.join(['0'] * K) or '-'},{ent},{hi.hex()},{lo.hex()}")

    if law == 1:
        x0 = list(eta) + [phi]

        def l_par(x):
            return dirichlet_l(x[:m], x[m], y)

        for A in range(7):
            for e in entries(m, A, rng):
                for K in range(7 - A):
                    v = checked(lambda: diff(l_par, x0, list(e) + [m] * K))
                    put(0, 0, A, K, rank(e, m), v)
        if min(y) >= 1e-4:
            xy = list(eta) + [phi] + list(y[:-1])

            def l_y(x):
                yy = [mp.mpf(v) for v in x[m + 1:]]
                yy.append(mp.mpf(y[-1]) - mp.fsum(v - mp.mpf(w) for v, w in zip(yy, y[:-1])))
                return dirichlet_l(x[:m], x[m], yy)

            for J in range(1, 5):
                for ey in entries(m, J, rng):
                    for A in range(0, 3 - (J > 2) * 2):
                        for e in entries(m, A, rng)[:2]:
                            for K in range(0, 3 - A - (J > 2) * 2):
                                pos = [m + 1 + i for i in ey] + list(e) + [m] * K
                                v = checked(lambda: diff(l_y, xy, pos))
                                put(0, J, A, K, rank(ey, m) * tensor_size(m, A) + rank(e, m), v)
        for kind in (1, 2, 3):
            ab = [(a, b) for a in range(m) for b in range(a + 1)] if kind == 1 else [(a, 0) for a in range(m)] if kind == 2 else [(0, 0)]
            ib = (lambda a, b: a * (a + 1) // 2 + b) if kind == 1 else (lambda a, b: a) if kind == 2 else (lambda a, b: 0)
            for (a, b) in ab:
                for A in range(5):
                    for e in entries(m, A, rng)[:3]:
                        for K in range(5 - A):
                            v = checked(lambda: diff(lambda x: dirichlet_info(kind, x[:m], x[m], a, b), x0, list(e) + [m] * K))
                            put(kind, 0, A, K, ib(a, b) * tensor_size(m, A) + rank(e, m), v)
    else:
        for A in range(7):
            for e in entries(m, A, rng):
                v = checked(lambda: diff(lambda x: multinomial_l(x, y, n), eta, list(e)))
                put(0, 0, A, 0, rank(e, m), v)
        for (a, b) in [(a, b) for a in range(m) for b in range(a + 1)]:
            for A in range(5):
                for e in entries(m, A, rng)[:3]:
                    v = checked(lambda: diff(lambda x: multinomial_info(x, n, a, b), eta, list(e)))
                    put(1, 0, A, 0, (a * (a + 1) // 2 + b) * tensor_size(m, A) + rank(e, m), v)
    return rows


def check_information():
    """the closed Dirichlet information against E[score score'] over the beta law, d = 2"""
    with mp.workdps(30):
        eta, phi = [mp.mpf("0.4")], mp.mpf("4.5")
        mu = probs(eta)
        a1, a2 = phi * mu[0], phi * mu[1]
        lb = mp.loggamma(phi) - mp.loggamma(a1) - mp.loggamma(a2)

        def score(t):
            y = [t, 1 - t]
            s_eta = mp.diff(lambda e: dirichlet_l([e], phi, y), eta[0])
            s_phi = mp.diff(lambda p: dirichlet_l(eta, p, y), phi)
            return s_eta, s_phi

        def expect(g):
            return mp.quad(lambda t: g(*score(t)) * mp.exp(lb + (a1 - 1) * mp.log(t) + (a2 - 1) * mp.log(1 - t)), [0, 0.5, 1])

        for kind, g in ((1, lambda u, v: u * u), (2, lambda u, v: u * v), (3, lambda u, v: v * v)):
            q = expect(g)
            c = dirichlet_info(kind, eta, phi, 0, 0)
            if abs(q - c) > mp.mpf(10) ** -15 * (1 + abs(c)):
                raise SystemExit(f"information kind {kind}: quadrature {q} closed {c}")
        print("information closed forms agree with the quadrature (d = 2)")


def main():
    check_information()
    pts = points()
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(task, pts, chunksize=1)
    rows = [r for rs in res for r in rs]
    OUT.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows from {len(pts)} points to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
