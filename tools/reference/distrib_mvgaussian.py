"""Reference values for the multivariate Gaussian distribs mvgaussian1 (covariance
Sigma = M(theta)) and mvgaussian2 (precision Q = M(theta)): blocks of
derivatives of the log-density in (mu, theta) and derivatives of the expected
information in theta, computed with mpmath at 60 digits from the definition.

Usage (from the repository root):  python -I tools/reference/distrib_mvgaussian.py

File: tests/data/reference/mvgaussian.csv, one row per entry, header
    param,structure,d,theta,y,mu,A,idx,entry,hi,lo
param is 1 (mvgaussian1) or 2 (mvgaussian2). theta, y and mu are points, their
coordinates as float.hex joined by ';'. idx is the sorted tuple of theta
indices, joined by ';' ('-' when empty). A = 0, 1, 2 is a block of the
log-density with A derivatives in mu: entry 0 for A = 0, entry i (the
component mu_i) for A = 1, entry i (i + 1)/2 + j (i >= j) for A = 2.
A = -1 is the expected information in theta, d_idx I_ab, at entry
a (a + 1)/2 + b (a >= b). hi is the nearest double to the value and lo the
nearest double to the remainder.

Structures M(theta) (code: name, definition, as the pdmat references):
    0 diagonal: M = diag(exp(2 theta_i)).
    1 ar1: rho = tanh(theta), M_ij = rho^|i-j|.
    2 compound_symmetry: a = -1/(d-1), rho = a + (1 - a)/(1 + exp(-theta)),
      M = (1 - rho) I + rho 1 1'.
    3 unstructured, log_cholesky: L_ii = exp(theta_ii), L_ij = theta_ij (i > j),
      theta_ij at i(i+1)/2 + j, M = L L'.
    4 unit_diagonal_scaled<ar1>: theta = (s_0..s_(d-1), r),
      M_ij = exp(s_i + s_j) tanh(r)^|i-j|.
    5 factor_analytic with one factor: theta = (l_0, ..., l_(d-1), t_0, ...,
      t_(d-1)), Lambda_0 = exp(l_0), Lambda_i = l_i (i >= 1),
      M = Lambda Lambda' + diag(exp(2 t_i)).

Definitions (r = y - mu, log det and the solve by mpmath's Cholesky):
    l1 = -d/2 log 2 pi - 1/2 log det Sigma - 1/2 r' Sigma^-1 r
    l2 = -d/2 log 2 pi + 1/2 log det Q - 1/2 r' Q r
The blocks are mpmath.diff of l over (mu, theta) with the orders of the
tuple. The expected information is I_ab(theta) = d^2 F(t', theta) / dt'_a dt'_b
at t' = theta, with F = E_theta[-l(t')] up to a constant,
    F1 = 1/2 log det Sigma(t') + 1/2 tr(Sigma(t')^-1 Sigma(theta)),
    F2 = -1/2 log det Q(t') + 1/2 tr(Q(t') Q(theta)^-1),
and its derivatives are mpmath.diff of that function of theta. None of this
uses the formulas of the C++ code (Leibniz's rule over the structure's
derivatives). Every value is computed twice, at 60 and at 75 digits, and the
script stops if the two disagree by more than 1e-40 times (1 + |value|).

Points: for each structure a random point (theta N(0, 0.5^2), y and mu
N(0, 1); random.Random(20261115 + 10 structure + d)) and a hard one (near
singular, correlations near +-1, large residuals; listed in POINTS). Tuples:
A = 0, orders 0 to 6; A = 1, orders 0 to 5; A = 2, orders 0 to 4; the
information, orders 0 to 4. All tuples of order 1 (and of order 2 for A = 0),
then the pure tuples of two coordinates and random mixed ones.
"""
import math
import random
import sys
import time
from itertools import combinations_with_replacement
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "mvgaussian.csv"
HEADER = "param,structure,d,theta,y,mu,A,idx,entry,hi,lo"
DPS, DPS_CHECK = 60, 75
TOL = mp.mpf(10) ** -40


def n_params(s, d):
    return {0: d, 1: 1, 2: 1, 3: d * (d + 1) // 2, 4: d + 1, 5: 2 * d}[s]


def build(s, d, th):
    M = mp.matrix(d, d)
    if s == 0:
        for i in range(d):
            M[i, i] = mp.exp(2 * th[i])
    elif s == 1:
        rho = mp.tanh(th[0])
        for i in range(d):
            for j in range(d):
                M[i, j] = rho ** abs(i - j)
    elif s == 2:
        a = mp.mpf(-1) / (d - 1)
        rho = a + (1 - a) / (1 + mp.exp(-th[0]))
        for i in range(d):
            for j in range(d):
                M[i, j] = 1 if i == j else rho
    elif s == 3:
        L = mp.matrix(d, d)
        for i in range(d):
            for j in range(i + 1):
                x = th[i * (i + 1) // 2 + j]
                L[i, j] = mp.exp(x) if i == j else x
        M = L * L.T
    elif s == 4:
        rho = mp.tanh(th[d])
        for i in range(d):
            for j in range(d):
                M[i, j] = mp.exp(th[i] + th[j]) * rho ** abs(i - j)
    else:
        lam = [mp.exp(th[0])] + [th[i] for i in range(1, d)]
        for i in range(d):
            for j in range(d):
                M[i, j] = lam[i] * lam[j] + (mp.exp(2 * th[d + i]) if i == j else 0)
    return M


def log_det(M):
    L = mp.cholesky(M)
    return 2 * mp.fsum(mp.log(L[i, i]) for i in range(M.rows))


def logpdf(param, s, d, th, y, mu):
    M = build(s, d, th)
    r = mp.matrix([y[i] - mu[i] for i in range(d)])
    ld = log_det(M)
    if param == 1:
        x = mp.cholesky_solve(M, r)
        q = mp.fsum(r[i] * x[i] for i in range(d))
        return -mp.mpf(d) / 2 * mp.log(2 * mp.pi) - ld / 2 - q / 2
    q = mp.fsum(r[i] * M[i, j] * r[j] for i in range(d) for j in range(d))
    return -mp.mpf(d) / 2 * mp.log(2 * mp.pi) + ld / 2 - q / 2


def info_F(param, s, d, tp, th):
    Mp, M = build(s, d, tp), build(s, d, th)
    if param == 1:
        X = mp.inverse(Mp) * M
        return log_det(Mp) / 2 + mp.fsum(X[i, i] for i in range(d)) / 2
    X = Mp * mp.inverse(M)
    return -log_det(Mp) / 2 + mp.fsum(X[i, i] for i in range(d)) / 2


def orders(nv, positions):
    n = [0] * nv
    for p in positions:
        n[p] += 1
    return tuple(n)


def block_value(param, s, d, th, y, mu, mu_pos, idx):
    """d/dmu_(mu_pos) d/dtheta_idx of l."""
    q = len(th)
    x = [mp.mpf(v) for v in mu] + [mp.mpf(v) for v in th]
    n = orders(d + q, list(mu_pos) + [d + k for k in idx])
    yy = [mp.mpf(v) for v in y]

    def f(*v):
        return logpdf(param, s, d, list(v[d:]), yy, list(v[:d]))

    if sum(n) == 0:
        return f(*x)
    return mp.diff(f, x, n)


def info_value(param, s, d, th, a, b, idx):
    q = len(th)
    x = [mp.mpf(v) for v in th]

    def g(*t):
        t = list(t)
        return mp.diff(lambda *tp: info_F(param, s, d, list(tp), t), t, orders(q, [a, b]))

    if not idx:
        return g(*x)
    return mp.diff(g, x, orders(q, idx))


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


# hard points: structure, d, theta, y, mu
def hard_points():
    pts = []
    pts.append((0, 4, [3.0, -3.0, 2.5, -2.0], [20.0, -1.0, 30.0, 0.5], [0.0, 0.0, -5.0, 0.0]))
    pts.append((1, 5, [3.5], [1.0, 1.2, 0.9, 1.1, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]))
    pts.append((2, 3, [8.0], [0.5, -0.4, 0.6], [0.0, 0.1, 0.0]))
    # log_cholesky, log L22 = -11.5
    pts.append((3, 3, [0.2, 0.7, -0.3, -0.5, 1.1, -11.5], [1.0, 2.0, -1.0], [0.0, 0.5, 0.0]))
    pts.append((4, 3, [2.0, -2.0, 1.0, -2.5], [8.0, -0.1, 3.0], [0.0, 0.0, 0.0]))
    pts.append((5, 4, [0.4, 1.5, -1.5, 1.5, -3.0, -3.0, 1.0, -3.0], [2.0, 3.0, -3.0, 2.5], [0.0, 0.0, 0.0, 0.0]))
    return pts


def random_point(s, d):
    rng = random.Random(20261115 + 10 * s + d)
    q = n_params(s, d)
    th = [round(rng.gauss(0, 0.5), 6) for _ in range(q)]
    y = [round(rng.gauss(0, 1), 6) for _ in range(d)]
    mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
    return (s, d, th, y, mu)


def tuples(q, K, rng, all_up_to):
    if K == 0:
        return [()]
    if K <= all_up_to:
        return list(combinations_with_replacement(range(q), K))
    out = [tuple([c] * K) for c in sorted(rng.sample(range(q), min(2, q)))]
    for _ in range(3):
        t = tuple(sorted(rng.randrange(q) for _ in range(K)))
        if t not in out:
            out.append(t)
    return out


def task(args):
    param, (s, d, th, y, mu) = args
    q = len(th)
    rng = random.Random(1000 * param + 10 * s + d)
    head = f"{param},{s},{d},{hexs(th)},{hexs(y)},{hexs(mu)}"
    rows = []
    for A, kmax, all_up in ((0, 6, 2), (1, 5, 1), (2, 4, 1)):
        entries = [()] if A == 0 else [(i,) for i in range(d)] if A == 1 else [(i, j) for i in range(d) for j in range(i + 1)]
        for K in range(kmax + 1):
            for t in tuples(q, K, rng, all_up):
                for e in entries:
                    v = checked(lambda: block_value(param, s, d, th, y, mu, e, t))
                    ent = 0 if A == 0 else e[0] if A == 1 else e[0] * (e[0] + 1) // 2 + e[1]
                    hi, lo = split(v)
                    rows.append(f"{head},{A},{';'.join(map(str, t)) or '-'},{ent},{hi.hex()},{lo.hex()}")
    pairs = [(a, b) for a in range(q) for b in range(a + 1)]
    for K in range(5):
        ps = pairs if K == 0 else rng.sample(pairs, min(3, len(pairs)))
        for (a, b) in ps:
            for t in tuples(q, K, rng, 0)[:2]:
                v = checked(lambda: info_value(param, s, d, th, a, b, list(t)))
                hi, lo = split(v)
                rows.append(f"{head},-1,{';'.join(map(str, t)) or '-'},{a * (a + 1) // 2 + b},{hi.hex()},{lo.hex()}")
    return rows


def main():
    pts = [random_point(s, d) for s, d in ((0, 2), (1, 3), (2, 4), (3, 2), (3, 3), (4, 3), (5, 3))] + hard_points()
    jobs = [(param, p) for param in (1, 2) for p in pts]
    t0 = time.time()
    with Pool(4) as pool:
        res = pool.map(task, jobs, chunksize=1)
    rows = [r for rs in res for r in rs]
    OUT.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
