"""Reference values for the multivariate Student t distribs mvstudent_t1 (scale
matrix Sigma = M(theta)) and mvstudent_t2 (inverse scale Q = M(theta)), proper
and degenerate: blocks of derivatives of the log-density in (mu, theta, nu) and
derivatives of the expected information in (theta, nu), computed with mpmath at
60 digits from the definition.

Usage (from the repository root):  python -I tools/reference/distrib_mvstudent_t.py

File: tests/data/reference/mvstudent_t.csv, one row per entry, header
    param,structure,d,theta,y,mu,A,idx,entry,hi,lo
param is 1 (mvstudent_t1), 2 (mvstudent_t2) or 3 (mvstudent_t1, degenerate, over
weighted_sum_psdmat). theta is the point of "the rest", the structure's
parameters followed by nu; theta, y and mu are their coordinates as float.hex
joined by ';'. idx is the sorted tuple of rest indices (nu is the last index,
q), joined by ';' ('-' when empty). A = 0..6 is a block of the log-density with
A derivatives in mu: entry 0 for A = 0, entry i for A = 1, entry i (i + 1)/2 + j
(i >= j) for A = 2, and for A >= 3 the rank of the sorted tuple of mu indices
in lexicographic order. A = -1 is the expected information in the rest,
d_idx I_ab at entry a (a + 1)/2 + b (a >= b); A = -2 the (mu, mu) block of the
information, d_idx I_mu_i mu_j at entry i (i + 1)/2 + j. hi is the nearest
double to the value and lo the nearest double to the remainder.

Structures: param 1 and 2, the structures 0..5 of distrib_mvgaussian.py
(diagonal, ar1, compound_symmetry, log_cholesky, scaled ar1, factor_analytic);
param 3, the weighted_sum structures 0..3 of distrib_mvgaussian_rank_deficient.py.

Definitions (r = y - mu, rho = d or the rank, |.| the determinant or the
pseudo-determinant, W = Sigma^-1, Sigma^+ or Q, delta = r' W r):
    l = lgamma((nu + rho)/2) - lgamma(nu/2) - rho/2 log(nu pi) -+ 1/2 log|M| - (nu + rho)/2 log(1 + delta/nu)
with mpmath's Cholesky (proper) or eigendecomposition (degenerate, r in the
range). The blocks are mpmath.diff of l. The expected information is that of
Lange, Little and Taylor (1989), checked against a quadrature over the law,
proper and degenerate, in tools/reference/mvstudent_t_forms.py; here it is
written with the Gaussian information I^G of the same matrix (the second
derivative of F, as in distrib_mvgaussian.py, not the C++ route),
t_a = d_a log|Sigma| and kappa = (nu + rho)/(nu + rho + 2):
    I_ab = kappa I^G_ab - t_a t_b / (2 (nu + rho + 2)),
    I_a,nu = -t_a / ((nu + rho)(nu + rho + 2)),
    I_nunu = 1/4 [psi'(nu/2) - psi'((nu + rho)/2)] - rho (nu + rho + 4) / (2 nu (nu + rho)(nu + rho + 2)),
    I_mumu = kappa W,
and its derivatives are mpmath.diff of these functions of (theta, nu). Every
value is computed at 60 and at 75 digits, and the script stops if the two
disagree by more than 1e-40 times (1 + |value|).

Points: for each structure a random point (theta N(0, 0.5^2), nu 4.5, y and mu
N(0, 1); for the degenerate law y = mu + a vector of the range) and hard ones:
nu = 1e6 (the cancellation towards the Gaussian), nu = 0.05 (heavy tails, the
small-nu information), a large residual, the near-singular points of the
Gaussian references. Tuples: A = 0 orders 0 to 6, A = 1 0 to 5, A = 2 0 to 4,
A = 3..6 to 6 - A (a sample of entries), the information 0 to 4.
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
OUT = HERE.parents[1] / "tests" / "data" / "reference" / "mvstudent_t.csv"
HEADER = "param,structure,d,theta,y,mu,A,idx,entry,hi,lo"
DPS, DPS_CHECK = 60, 75
TOL = mp.mpf(10) ** -40


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load("distrib_mvgaussian")
RD = load("distrib_mvgaussian_rank_deficient")


def n_struct_params(param, s, d):
    return len(RD.structure(s)[2]) if param == 3 else G.n_params(s, d)


def matrix_parts(param, s, d, th):
    """rho, log|M|, W (the matrix of the quadratic form)"""
    if param == 3:
        M = RD.build(s, th)
        ld, Mp = RD.range_part(s, M)
        return RD.structure(s)[1], ld, Mp
    M = G.build(s, d, th)
    ld = G.log_det(M)
    return d, ld, (mp.inverse(M) if param == 1 else M)


def logpdf(param, s, d, th, y, mu):
    q = n_struct_params(param, s, d)
    nu = th[q]
    rho, ld, W = matrix_parts(param, s, d, th[:q])
    r = [y[i] - mu[i] for i in range(d)]
    delta = mp.fsum(r[i] * W[i, j] * r[j] for i in range(d) for j in range(d))
    c = mp.loggamma((nu + rho) / 2) - mp.loggamma(nu / 2) - mp.mpf(rho) / 2 * mp.log(nu * mp.pi)
    sgn = 1 if param == 2 else -1
    return c + sgn * ld / 2 - (nu + rho) / 2 * mp.log1p(delta / nu)


def log_det_sigma(param, s, d, th):
    """log|Sigma| as a function of the structure's parameters"""
    if param == 3:
        return RD.range_part(s, RD.build(s, th))[0]
    ld = G.log_det(G.build(s, d, th))
    return -ld if param == 2 else ld


def info_gauss(param, s, d, th, a, b):
    """I^G_ab of the Gaussian with the same matrix, by the F of the Gaussian scripts"""
    q = len(th)
    x = [mp.mpf(v) for v in th]
    if param == 3:
        return mp.diff(lambda *tp: RD.info_F(1, s, list(tp), x), x, G.orders(q, [a, b]))
    return mp.diff(lambda *tp: G.info_F(param, s, d, list(tp), x), x, G.orders(q, [a, b]))


def info_entry(param, s, d, th_all, a, b):
    """I_ab over the rest (nu at index q), from the definition above"""
    q = n_struct_params(param, s, d)
    th, nu = th_all[:q], th_all[q]
    rho = RD.structure(s)[1] if param == 3 else d
    if a > b:
        a, b = b, a
    if a == q:
        h = mp.mpf(rho) / 2
        return (mp.psi(1, nu / 2) - mp.psi(1, nu / 2 + h)) / 4 - rho * (nu + rho + 4) / (2 * nu * (nu + rho) * (nu + rho + 2))
    x = [mp.mpf(v) for v in th]

    def t(k):
        return mp.diff(lambda *tp: log_det_sigma(param, s, d, list(tp)), x, G.orders(q, [k]))

    if b == q:
        return -t(a) / ((nu + rho) * (nu + rho + 2))
    kappa = (nu + rho) / (nu + rho + 2)
    return kappa * info_gauss(param, s, d, th, a, b) - t(a) * t(b) / (2 * (nu + rho + 2))


def info_mumu(param, s, d, th_all, i, j):
    q = n_struct_params(param, s, d)
    rho, _, W = matrix_parts(param, s, d, th_all[:q])
    nu = th_all[q]
    return (nu + rho) / (nu + rho + 2) * W[i, j]


def block_value(param, s, d, th, y, mu, mu_pos, idx):
    """The mixed partial by the polarization identity: with v_1..v_n the unit
    vectors of the n positions, D^n f[v_1..v_n] = 1/(n! 2^(n-1)) sum over the
    signs eps with eps_1 = +1 of prod(eps) d^n/dt^n f(x + t sum eps_i v_i) at
    t = 0 (the terms with eps_1 = -1 repeat those with +1); each term one
    one-dimensional mpmath.diff, instead of a nested multivariate one."""
    q = len(th)
    x = [mp.mpf(v) for v in mu] + [mp.mpf(v) for v in th]
    pos = list(mu_pos) + [d + k for k in idx]
    yy = [mp.mpf(v) for v in y]

    def f(v):
        return logpdf(param, s, d, list(v[d:]), yy, list(v[:d]))

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


def diff_over_rest(fn, th, idx):
    x = [mp.mpf(v) for v in th]
    if not idx:
        return fn(x)
    return mp.diff(lambda *t: fn(list(t)), x, G.orders(len(th), idx))


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


def sorted_rank(t, d):
    """lexicographic rank of the sorted tuple t among the sorted tuples over d"""
    for r, u in enumerate(combinations_with_replacement(range(d), len(t))):
        if u == t:
            return r
    raise ValueError(t)


def range_vector(s, rng):
    """a vector of the range of the weighted_sum structure s (projection of a normal vector)"""
    d = RD.structure(s)[0]
    M = RD.build(s, [mp.mpf(0)] * len(RD.structure(s)[2]))
    _, Mp = RD.range_part(s, M)
    P = M * Mp  # the projector on the range
    g = [mp.mpf(round(rng.gauss(0, 1), 6)) for _ in range(d)]
    return [float(mp.fsum(P[i, j] * g[j] for j in range(d))) for i in range(d)]


def points():
    pts = []
    for s, d in ((0, 2), (1, 3), (2, 4), (3, 2), (3, 3), (4, 3), (5, 3)):
        rng = random.Random(20261116 + 10 * s + d)
        q = G.n_params(s, d)
        th = [round(rng.gauss(0, 0.5), 6) for _ in range(q)] + [4.5]
        y = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        for param in (1, 2):
            pts.append((param, s, d, th, y, mu))
    # hard points, both parametrizations
    hard = [
        (3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -0.4, 1e6], [1.0, 2.0, -1.0], [0.0, 0.5, 0.0]),     # near the Gaussian
        (1, 3, [0.6, 0.05], [1.5, -2.0, 0.3], [0.0, 0.0, 0.0]),                                 # heavy tails
        (4, 3, [0.3, -0.2, 0.1, 0.5, 0.05], [0.4, 0.2, -0.3], [0.0, 0.0, 0.0]),                # small nu, a common scale
        (0, 4, [3.0, -3.0, 2.5, -2.0, 2.5], [20.0, -1.0, 30.0, 0.5], [0.0, 0.0, -5.0, 0.0]),   # large residuals
        (3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -11.5, 6.0], [1.0, 2.0, -1.0], [0.0, 0.5, 0.0]),   # near singular
        (2, 3, [8.0, 3.0], [0.5, -0.4, 0.6], [0.0, 0.1, 0.0]),
    ]
    for s, d, th, y, mu in hard:
        for param in (1, 2):
            pts.append((param, s, d, th, y, mu))
    # degenerate, weighted_sum
    for s in range(4):
        rng = random.Random(20261117 + 10 * s)
        d, _, S = RD.structure(s)
        th = [round(rng.gauss(0, 1), 6) for _ in range(len(S))] + [3.5]
        mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        v = range_vector(s, rng)
        y = [mu[i] + v[i] for i in range(d)]
        pts.append((3, s, d, th, y, mu))
    rng = random.Random(7)
    d, _, S = RD.structure(1)
    mu = [0.0] * d
    v = range_vector(1, rng)
    pts.append((3, 1, d, [-4.5, 4.5, 1e5], [mu[i] + v[i] for i in range(d)], mu))
    return pts


def tuples(q, K, rng, all_up_to, nu_index):
    if K == 0:
        return [()]
    if K <= all_up_to:
        return list(combinations_with_replacement(range(q), K))
    out = [tuple([c] * K) for c in sorted(set([nu_index] + rng.sample(range(q), min(1, q))))]
    for _ in range(3):
        t = tuple(sorted(rng.randrange(q) for _ in range(K)))
        if t not in out:
            out.append(t)
    return out


def task(pt):
    param, s, d, th, y, mu = pt
    q = len(th)  # the rest: structure + nu
    rng = random.Random(1000 * param + 10 * s + d + int(th[-1] * 7))
    head = f"{param},{s},{d},{hexs(th)},{hexs(y)},{hexs(mu)}"
    rows = []

    def put(A, t, ent, v):
        hi, lo = split(v)
        rows.append(f"{head},{A},{';'.join(map(str, t)) or '-'},{ent},{hi.hex()},{lo.hex()}")

    for A, kmax, all_up in ((0, 6, 2), (1, 5, 1), (2, 4, 1), (3, 3, 1), (4, 2, 0), (5, 1, 0), (6, 0, 0)):
        if A <= 2:
            ents = [()] if A == 0 else [(i,) for i in range(d)] if A == 1 else [(i, j) for i in range(d) for j in range(i + 1)]
        else:
            allt = list(combinations_with_replacement(range(d), A))
            ents = [allt[0], allt[-1]] + rng.sample(allt[1:-1], min(2, len(allt) - 2))
        for K in range(kmax + 1):
            for t in tuples(q, K, rng, all_up, q - 1):
                for e in ents:
                    v = checked(lambda: block_value(param, s, d, th, y, mu, e, t))
                    ent = 0 if A == 0 else e[0] if A == 1 else e[0] * (e[0] + 1) // 2 + e[1] if A == 2 else sorted_rank(e, d)
                    put(A, t, ent, v)
    pairs = [(a, b) for a in range(q) for b in range(a + 1)]
    for K in range(5):
        ps = pairs if K == 0 else rng.sample(pairs, min(3, len(pairs))) + [(q - 1, 0), (q - 1, q - 1)]
        for (a, b) in ps:
            for t in tuples(q, K, rng, 0, q - 1)[:2]:
                v = checked(lambda: diff_over_rest(lambda x: info_entry(param, s, d, x, a, b), th, list(t)))
                put(-1, t, a * (a + 1) // 2 + b, v)
    for K in range(3):
        for t in tuples(q, K, rng, 0, q - 1)[:2]:
            for (i, j) in ((0, 0), (d - 1, 0)):
                v = checked(lambda: diff_over_rest(lambda x: info_mumu(param, s, d, x, i, j), th, list(t)))
                put(-2, t, i * (i + 1) // 2 + j, v)
    return rows


def main():
    pts = points()
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(task, pts, chunksize=1)
    rows = [r for rs in res for r in rs]
    OUT.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
