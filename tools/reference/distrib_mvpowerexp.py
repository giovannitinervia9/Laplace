"""Reference values for the multivariate power exponential distribs mvpowerexp1
(scale matrix Sigma = M(theta)) and mvpowerexp2 (inverse scale Q = M(theta)):
blocks of derivatives of the log-density in (mu, theta, beta), computed with
mpmath at 60 digits from the definition.

Usage (from the repository root):  python -I tools/reference/distrib_mvpowerexp.py

File: tests/data/reference/mvpowerexp.csv, one row per entry, header
    param,structure,d,theta,y,mu,A,idx,entry,hi,lo
param is 1 (mvpowerexp1) or 2 (mvpowerexp2). theta is the point of "the rest",
the structure's parameters followed by beta; theta, y and mu are their
coordinates as float.hex joined by ';'. idx is the sorted tuple of rest indices
(beta is the last index, q), joined by ';' ('-' when empty). A = 0..6 is a block
of the log-density with A derivatives in mu: entry 0 for A = 0, entry i for
A = 1, entry i (i + 1)/2 + j (i >= j) for A = 2, and for A >= 3 the rank of the
sorted tuple of mu indices in lexicographic order. hi is the nearest double to
the value and lo the nearest double to the remainder.

Structures: the structures 0..5 of distrib_mvgaussian.py (diagonal, ar1,
compound_symmetry, log_cholesky, scaled ar1, factor_analytic).

The definition (Gomez, Gomez-Villegas and Marin 1998; r = y - mu, |.| the
determinant, W = Sigma^-1 or Q, delta = r' W r):
    l = -log Z(d, beta) -+ 1/2 log|M| - 1/2 delta^beta,
    Z(d, beta) = 2^(1 + d/(2 beta)) pi^(d/2) Gamma(1 + d/(2 beta)) / (d Gamma(d/2)),
with mpmath's Cholesky for the matrix. The blocks are mpmath.diff of l, by the
polarization identity, so that each mixed partial is a sum of one-dimensional
derivatives. Z is derived by radial integration and checked against the
quadrature of the definition in tools/reference/mvpowerexp_forms.py. Every
value is computed at 60 and at 75 digits, and the script stops if the two
disagree by more than 1e-30 times (1 + |value|).

Points: for each structure a random point (theta N(0, 0.5^2), y and mu N(0, 1))
at several betas, and hard ones: beta in (0, 1/2), beta = 1/2 (the group lasso),
beta in (1/2, 1), beta = 1 (the Gaussian), beta = 2 and beta = 6 (the light
tails), r at the location, at 1e-3 of it, at order one and far in the tail, and
a near-singular matrix. At r = 0 exactly only the value and the derivatives in
beta alone are written: the others do not exist there, the kink being radial.
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
OUT = HERE.parents[1] / "tests" / "data" / "reference" / "mvpowerexp.csv"
HEADER = "param,structure,d,theta,y,mu,A,idx,entry,hi,lo"
DPS, DPS_CHECK = 60, 75
TOL = mp.mpf(10) ** -30


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load("distrib_mvgaussian")


def log_z(d, beta):
    m = mp.mpf(d) / (2 * beta)
    return (1 + m) * mp.log(2) + mp.mpf(d) / 2 * mp.log(mp.pi) + mp.loggamma(1 + m) - mp.log(d) - mp.loggamma(mp.mpf(d) / 2)


def logpdf(param, s, d, th, y, mu):
    q = len(th) - 1
    beta = th[q]
    M = G.build(s, d, th[:q])
    ld = G.log_det(M)
    W = mp.inverse(M) if param == 1 else M
    r = [y[i] - mu[i] for i in range(d)]
    delta = mp.fsum(r[i] * W[i, j] * r[j] for i in range(d) for j in range(d))
    sgn = 1 if param == 2 else -1
    kernel = mp.mpf(0) if delta == 0 else delta**beta / 2
    return -log_z(d, beta) + sgn * ld / 2 - kernel


def block_value(param, s, d, th, y, mu, mu_pos, idx):
    """The mixed partial by the polarization identity: with v_1..v_n the unit
    vectors of the n positions, D^n f[v_1..v_n] = 1/(n! 2^(n-1)) sum over the
    signs eps with eps_1 = +1 of prod(eps) d^n/dt^n f(x + t sum eps_i v_i) at
    t = 0 (the terms with eps_1 = -1 repeat those with +1); each term is one
    one-dimensional mpmath.diff instead of a nested multivariate one."""
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


# beta covering the regimes: below 1/2, at 1/2 (the group lasso), between 1/2
# and 1, at 1 (the Gaussian), and above (the light tails)
BETAS = [0.3, 0.5, 0.7, 1.0, 2.0, 6.0]


def points():
    pts = []
    for s, d in ((0, 1), (0, 2), (1, 3), (2, 4), (3, 2), (3, 3), (4, 3), (5, 5)):
        rng = random.Random(20261118 + 10 * s + d)
        q = G.n_params(s, d)
        base = [round(rng.gauss(0, 0.5), 6) for _ in range(q)]
        y = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        for beta in BETAS:
            for param in (1, 2):
                pts.append((param, s, d, base + [beta], y, mu, 6))
    # hard points
    hard = []
    for beta in BETAS:
        # r at the location exactly: only the value and the pure beta derivatives
        hard.append((3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -0.4, beta], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0], 6))
        # r at 1e-3 of the location, where the kink is near
        hard.append((3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -0.4, beta], [1e-3, 0.5 - 2e-3, 1e-3], [0.0, 0.5, 0.0], 3))
        # far in the tail
        hard.append((0, 4, [0.4, -0.3, 0.25, -0.2, beta], [9.0, -7.0, 11.0, 0.5], [0.0, 0.0, -1.0, 0.0], 4))
        # a near-singular matrix
        hard.append((3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -7.5, beta], [1.0, 2.0, -1.0], [0.0, 0.5, 0.0], 4))
        # a common scale with a factor structure
        hard.append((4, 3, [0.3, -0.2, 0.1, 0.5, beta], [0.4, 0.2, -0.3], [0.0, 0.0, 0.0], 4))
    for s, d, th, y, mu, kmax in hard:
        for param in (1, 2):
            pts.append((param, s, d, th, y, mu, kmax))
    return pts


def tuples(q, K, rng, all_up_to, beta_index):
    """The sorted tuples of K rest indices: all of them up to all_up_to, then a
    sample that always includes the pure beta tuple and one pure theta tuple."""
    if K == 0:
        return [()]
    if K <= all_up_to:
        return list(combinations_with_replacement(range(q), K))
    out = [tuple([beta_index] * K), tuple([0] * K)]
    if q > 1:
        out.append(tuple(sorted([0] * (K - 1) + [beta_index])))
    for _ in range(3):
        t = tuple(sorted(rng.randrange(q) for _ in range(K)))
        if t not in out:
            out.append(t)
    return out


def task(pt):
    param, s, d, th, y, mu, kink_order = pt
    q = len(th)  # the rest: the structure's parameters and beta
    at_kink = all(y[i] == mu[i] for i in range(d))
    rng = random.Random(1000 * param + 10 * s + d + int(th[-1] * 71))
    head = f"{param},{s},{d},{hexs(th)},{hexs(y)},{hexs(mu)}"
    rows = []

    def put(A, t, ent, v):
        hi, lo = split(v)
        rows.append(f"{head},{A},{';'.join(map(str, t)) or '-'},{ent},{hi.hex()},{lo.hex()}")

    for A, kmax, all_up in ((0, 6, 2), (1, 5, 1), (2, 4, 1), (3, 3, 1), (4, 2, 0), (5, 1, 0), (6, 0, 0)):
        if A > kink_order:
            continue
        if A <= 2:
            ents = (
                [()]
                if A == 0
                else [(i,) for i in range(d)]
                if A == 1
                else [(i, j) for i in range(d) for j in range(i + 1)]
            )
        else:
            allt = list(combinations_with_replacement(range(d), A))
            ents = [allt[0], allt[-1]] + rng.sample(allt[1:-1], min(2, max(0, len(allt) - 2)))
        for K in range(kmax + 1):
            if A + K > kink_order:
                continue
            for t in tuples(q, K, rng, all_up, q - 1):
                # at the location exactly only the value and the derivatives in
                # beta alone exist: the kink is radial
                if at_kink and (A > 0 or any(i != q - 1 for i in t)):
                    continue
                for e in ents:
                    v = checked(lambda: block_value(param, s, d, th, y, mu, e, t))
                    ent = (
                        0
                        if A == 0
                        else e[0]
                        if A == 1
                        else e[0] * (e[0] + 1) // 2 + e[1]
                        if A == 2
                        else sorted_rank(e, d)
                    )
                    put(A, t, ent, v)
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
