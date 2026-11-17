"""Reference values for the improper (intrinsic) class of mvstudent_t2: the
multivariate Student t over an inverse scale Q = M(theta) of rank rho < d with a
constant null space (weighted_sum_psdmat), flat along the null space. Blocks of
derivatives of the log-density in (mu, theta, nu) and derivatives of the
expected information of the constrained law U0'(y - mu) = 0 in (theta, nu),
computed with mpmath at 60 digits from the definition.

Usage (from the repository root):
    python -I tools/reference/distrib_mvstudent_t_improper.py

File: tests/data/reference/mvstudent_t_improper.csv, the format of
mvstudent_t.csv (see distrib_mvstudent_t.py), param = 4.

Definitions (r = y - mu, |Q|_+ the product of the rho nonzero eigenvalues,
delta = r' Q r, which ignores the null component of r):
    l = lgamma((nu + rho)/2) - lgamma(nu/2) - rho/2 log(nu pi) + 1/2 log|Q|_+ - (nu + rho)/2 log(1 + delta/nu),
the rho-dimensional proper t of the range component times Lebesgue measure on
the null one. Its expected information is that of the constrained law, the
rho-dimensional proper t on the range, written as in distrib_mvstudent_t.py
with I^G the improper Gaussian's (the F of distrib_mvgaussian_rank_deficient.py,
param 2), t_a = d_a log|Sigma| = -d_a log|Q|_+ and kappa = (nu + rho)/(nu + rho + 2):
    I_ab = kappa I^G_ab - t_a t_b / (2 (nu + rho + 2)),
    I_a,nu = -t_a / ((nu + rho)(nu + rho + 2)),
    I_nunu = 1/4 [psi'(nu/2) - psi'((nu + rho)/2)] - rho (nu + rho + 4) / (2 nu (nu + rho)(nu + rho + 2)),
    I_mumu = kappa Q.
Every value is computed at 60 and 75 digits and checked as in distrib_mvstudent_t.py.

Structures: the weighted_sum structures 0..3 of distrib_mvgaussian_rank_deficient.py
(structure 0 has one matrix, m = 1). Points: per structure a random one (theta
N(0, 1), nu 3.5, y and mu N(0, 1), so r has a null component) and hard ones: m = 1
at nu = 0.05 (the small-nu information) and nu = 1e6 (towards the Gaussian), two
penalties 9 apart, a large residual.
"""
import importlib.util
import random
import sys
import time
from itertools import combinations_with_replacement
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "tests" / "data" / "reference" / "mvstudent_t_improper.csv"
PARAM = 4


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MT = load("distrib_mvstudent_t")
RD = MT.RD


def logpdf(s, th_all, y, mu):
    d, rho, S = RD.structure(s)
    q = len(S)
    nu = th_all[q]
    Q = RD.build(s, th_all[:q])
    ld = RD.range_part(s, Q)[0]
    r = [y[i] - mu[i] for i in range(d)]
    delta = mp.fsum(r[i] * Q[i, j] * r[j] for i in range(d) for j in range(d))
    c = mp.loggamma((nu + rho) / 2) - mp.loggamma(nu / 2) - mp.mpf(rho) / 2 * mp.log(nu * mp.pi)
    return c + ld / 2 - (nu + rho) / 2 * mp.log1p(delta / nu)


def info_entry(s, th_all, a, b):
    _, rho, S = RD.structure(s)
    q = len(S)
    th, nu = th_all[:q], th_all[q]
    if a > b:
        a, b = b, a
    if a == q:
        h = mp.mpf(rho) / 2
        return (mp.psi(1, nu / 2) - mp.psi(1, nu / 2 + h)) / 4 - rho * (nu + rho + 4) / (2 * nu * (nu + rho) * (nu + rho + 2))
    x = [mp.mpf(v) for v in th]

    def t(k):  # d_k log|Sigma| = -d_k log|Q|_+
        return -mp.diff(lambda *tp: RD.range_part(s, RD.build(s, list(tp)))[0], x, RD.orders(q, [k]))

    if b == q:
        return -t(a) / ((nu + rho) * (nu + rho + 2))
    ig = mp.diff(lambda *tp: RD.info_F(2, s, list(tp), x), x, RD.orders(q, [a, b]))
    return (nu + rho) / (nu + rho + 2) * ig - t(a) * t(b) / (2 * (nu + rho + 2))


def info_mumu(s, th_all, i, j):
    _, rho, S = RD.structure(s)
    q = len(S)
    nu = th_all[q]
    return (nu + rho) / (nu + rho + 2) * RD.build(s, th_all[:q])[i, j]


def block_value(s, d, th, y, mu, mu_pos, idx):
    """the mixed partial by polarization, as distrib_mvstudent_t.block_value"""
    q = len(th)
    x = [mp.mpf(v) for v in mu] + [mp.mpf(v) for v in th]
    pos = list(mu_pos) + [d + k for k in idx]
    yy = [mp.mpf(v) for v in y]

    def f(v):
        return logpdf(s, list(v[d:]), yy, list(v[:d]))

    n = len(pos)
    if n == 0:
        return f(x)
    acc = mp.mpf(0)
    for mask in range(1 << (n - 1)):
        eps = [1] + [-1 if mask & (1 << (i - 1)) else 1 for i in range(1, n)]
        u = [mp.mpf(0)] * (d + q)
        for e, p in zip(eps, pos):
            u[p] += e
        sign = 1
        for e in eps:
            sign *= e
        acc += sign * mp.diff(lambda t: f([x[k] + t * u[k] for k in range(d + q)]), 0, n)
    return acc / (mp.factorial(n) * 2 ** (n - 1))


def points():
    pts = []
    for s in range(4):
        rng = random.Random(20261118 + 10 * s)
        d, _, S = RD.structure(s)
        th = [round(rng.gauss(0, 1), 6) for _ in S] + [3.5]
        y = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        pts.append((s, d, th, y, mu))
    pts += [
        (0, 4, [0.4, 0.05], [0.3, -0.8, 1.1, 0.2], [0.0, 0.0, 0.0, 0.0]),          # m = 1, small nu
        (0, 4, [-0.7, 1e6], [1.0, 2.0, -1.0, 0.5], [0.0, 0.5, 0.0, 0.0]),          # m = 1, near the Gaussian
        (1, 5, [-4.5, 4.5, 4.0], [0.5, -0.4, 0.6, 0.1, -0.2], [0.0] * 5),          # two penalties 9 apart
        (3, 5, [1.5, -1.0, 2.0], [20.0, -1.0, 30.0, 0.5, -12.0], [0.0] * 5),      # large residual
    ]
    return pts


def task(pt):
    s, d, th, y, mu = pt
    q = len(th)  # the rest: structure + nu
    rng = random.Random(4000 + 10 * s + int(th[-1] * 7))
    head = f"{PARAM},{s},{d},{MT.hexs(th)},{MT.hexs(y)},{MT.hexs(mu)}"
    rows = []

    def put(A, t, ent, v):
        hi, lo = MT.split(v)
        rows.append(f"{head},{A},{';'.join(map(str, t)) or '-'},{ent},{hi.hex()},{lo.hex()}")

    for A, kmax, all_up in ((0, 6, 2), (1, 5, 1), (2, 4, 1), (3, 3, 1), (4, 2, 0), (5, 1, 0), (6, 0, 0)):
        if A <= 2:
            ents = [()] if A == 0 else [(i,) for i in range(d)] if A == 1 else [(i, j) for i in range(d) for j in range(i + 1)]
        else:
            allt = list(combinations_with_replacement(range(d), A))
            ents = [allt[0], allt[-1]] + rng.sample(allt[1:-1], min(2, len(allt) - 2))
        for K in range(kmax + 1):
            for t in MT.tuples(q, K, rng, all_up, q - 1):
                for e in ents:
                    v = MT.checked(lambda: block_value(s, d, th, y, mu, e, t))
                    ent = 0 if A == 0 else e[0] if A == 1 else e[0] * (e[0] + 1) // 2 + e[1] if A == 2 else MT.sorted_rank(e, d)
                    put(A, t, ent, v)
    pairs = [(a, b) for a in range(q) for b in range(a + 1)]
    for K in range(5):
        ps = pairs if K == 0 else rng.sample(pairs, min(3, len(pairs))) + [(q - 1, 0), (q - 1, q - 1)]
        for (a, b) in ps:
            for t in MT.tuples(q, K, rng, 0, q - 1)[:2]:
                v = MT.checked(lambda: MT.diff_over_rest(lambda x: info_entry(s, x, a, b), th, list(t)))
                put(-1, t, a * (a + 1) // 2 + b, v)
    for K in range(3):
        for t in MT.tuples(q, K, rng, 0, q - 1)[:2]:
            for (i, j) in ((0, 0), (d - 1, 0)):
                v = MT.checked(lambda: MT.diff_over_rest(lambda x: info_mumu(s, x, i, j), th, list(t)))
                put(-2, t, i * (i + 1) // 2 + j, v)
    return rows


def main():
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(task, points(), chunksize=1)
    rows = [r for rs in res for r in rs]
    OUT.write_text(MT.HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
