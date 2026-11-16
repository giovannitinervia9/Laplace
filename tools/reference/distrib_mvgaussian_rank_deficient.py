"""Reference values for the rank-deficient multivariate Gaussians: mvgaussian1
over a singular covariance Sigma = M(theta) (degenerate) and mvgaussian2 over a
singular precision Q = M(theta) (improper), M(theta) = sum_j exp(theta_j) S_j
(weighted_sum_psdmat). Blocks of derivatives of the log-density in
(mu, theta) and derivatives of the expected information in theta, computed
with mpmath at 60 digits from the definition.

Usage (from the repository root):
    python -I tools/reference/distrib_mvgaussian_rank_deficient.py

File: tests/data/reference/mvgaussian_rank_deficient.csv, the format of
mvgaussian.csv (tools/reference/distrib_mvgaussian.py):
    param,structure,d,theta,y,mu,A,idx,entry,hi,lo

Structures (D_k the (d - k) x d matrix of k-th differences):
    0  d = 4, S = (D2'D2),                         rank 2
    1  d = 5, S = (D1'D1, D2'D2),                  rank 4
    2  d = 6, S = (D1'D1, D2'D2, D3'D3),           rank 5
    3  d = 5, S = (D2'D2, D2' diag(1, 2, 3) D2),   rank 3
The S_j are integer matrices; the C++ test builds the same.

Definitions, from the eigendecomposition of M by mpmath (eigsy), its rho
largest eigenvalues lambda_k and their eigenvectors v_k (rho the rank, the
other d - rho eigenvalues are zero):
    log|M|_+ = sum_k log lambda_k,   M^+ = sum_k v_k v_k' / lambda_k,
    l1 = -rho/2 log 2 pi - 1/2 log|Sigma|_+ - 1/2 r' Sigma^+ r   (r = y - mu)
    l2 = -rho/2 log 2 pi + 1/2 log|Q|_+    - 1/2 r' Q r
For the degenerate law, r lies in the range of Sigma (the point is built so;
l1 is the density on mu + col(Sigma)). The blocks are mpmath.diff of l over
(mu, theta). The expected information is I_ab = d^2 F(t', theta)/dt'_a dt'_b
at t' = theta with F = E_theta[-l(t')] up to a constant,
    F1 = 1/2 log|Sigma(t')|_+ + 1/2 tr(Sigma(t')^+ Sigma(theta)),
    F2 = -1/2 log|Q(t')|_+ + 1/2 tr(Q(t') Q(theta)^+),
the expectation under the degenerate law, or under the constrained law of the
improper one (U0'r = 0, covariance Q^+); its derivatives are mpmath.diff of
that function of theta. Nothing here uses the route of the C++ code
((M + cP)^-1 and its recursion, log_pdet from Wood's transform). Every value
is computed at 60 and at 75 digits, and the script stops if they disagree by
more than 1e-40 times (1 + |value|).

Points: for each structure a random point (theta N(0, 1), y and mu N(0, 1),
for the degenerate law y = mu + the projection of a N(0, 1) vector on the
range; random.Random(20261116 + 10 structure)) and a spread one (theta with
components 9 apart). Tuples as in distrib_mvgaussian.py.
"""
import random
import sys
import time
from itertools import combinations_with_replacement
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "mvgaussian_rank_deficient.csv"
HEADER = "param,structure,d,theta,y,mu,A,idx,entry,hi,lo"
DPS, DPS_CHECK = 60, 75
TOL = mp.mpf(10) ** -40


def diff_matrix(d, k):
    rows = [[1 if i == j else 0 for j in range(d)] for i in range(d)]
    for _ in range(k):
        rows = [[rows[i + 1][j] - rows[i][j] for j in range(d)] for i in range(len(rows) - 1)]
    return rows


def gram(D, w=None):
    d = len(D[0])
    w = w or [1] * len(D)
    return [[sum(w[k] * D[k][i] * D[k][j] for k in range(len(D))) for j in range(d)] for i in range(d)]


def structure(s):
    """(d, rank, list of S_j as integer lists)"""
    if s == 0:
        return 4, 2, [gram(diff_matrix(4, 2))]
    if s == 1:
        return 5, 4, [gram(diff_matrix(5, 1)), gram(diff_matrix(5, 2))]
    if s == 2:
        return 6, 5, [gram(diff_matrix(6, k)) for k in (1, 2, 3)]
    D2 = diff_matrix(5, 2)
    return 5, 3, [gram(D2), gram(D2, [1, 2, 3])]


def build(s, th):
    d, _, S = structure(s)
    M = mp.matrix(d, d)
    for j, Sj in enumerate(S):
        e = mp.exp(th[j])
        for a in range(d):
            for b in range(d):
                M[a, b] += e * Sj[a][b]
    return M


def range_part(s, M):
    """log|M|_+ and M^+ from the rho largest eigenvalues"""
    d, rho, _ = structure(s)
    E, V = mp.eigsy(M)
    order = sorted(range(d), key=lambda k: E[k], reverse=True)[:rho]
    ld = mp.fsum(mp.log(E[k]) for k in order)
    Mp = mp.matrix(d, d)
    for k in order:
        for a in range(d):
            for b in range(d):
                Mp[a, b] += V[a, k] * V[b, k] / E[k]
    return ld, Mp


def logpdf(param, s, th, y, mu):
    d, rho, _ = structure(s)
    M = build(s, th)
    r = [y[i] - mu[i] for i in range(d)]
    ld, Mp = range_part(s, M)
    W = Mp if param == 1 else M
    q = mp.fsum(r[i] * W[i, j] * r[j] for i in range(d) for j in range(d))
    c = -mp.mpf(rho) / 2 * mp.log(2 * mp.pi)
    return c - ld / 2 - q / 2 if param == 1 else c + ld / 2 - q / 2


def info_F(param, s, tp, th):
    d = structure(s)[0]
    Mp_, M = build(s, tp), build(s, th)
    ldp, pinv_p = range_part(s, Mp_)
    if param == 1:
        X = pinv_p * M
        return ldp / 2 + mp.fsum(X[i, i] for i in range(d)) / 2
    _, pinv = range_part(s, M)
    X = Mp_ * pinv
    return -ldp / 2 + mp.fsum(X[i, i] for i in range(d)) / 2


def orders(nv, positions):
    n = [0] * nv
    for p in positions:
        n[p] += 1
    return tuple(n)


def block_value(param, s, th, y, mu, mu_pos, idx):
    d, q = structure(s)[0], len(th)
    x = [mp.mpf(v) for v in mu] + [mp.mpf(v) for v in th]
    n = orders(d + q, list(mu_pos) + [d + k for k in idx])
    yy = [mp.mpf(v) for v in y]

    def f(*v):
        return logpdf(param, s, list(v[d:]), yy, list(v[:d]))

    if sum(n) == 0:
        return f(*x)
    return mp.diff(f, x, n)


def info_value(param, s, th, a, b, idx):
    q = len(th)
    x = [mp.mpf(v) for v in th]

    def g(*t):
        t = list(t)
        return mp.diff(lambda *tp: info_F(param, s, list(tp), t), t, orders(q, [a, b]))

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


def point(param, s, kind):
    d, rho, S = structure(s)
    rng = random.Random(20261116 + 10 * s + kind)
    if kind == 0:
        th = [round(rng.gauss(0, 1), 6) for _ in S]
    else:
        th = [round(9.0 * (j - (len(S) - 1) / 2) + rng.gauss(0, 0.3), 6) for j in range(len(S))]
    mu = [round(rng.gauss(0, 1), 6) for _ in range(d)]
    if param == 2:
        y = [round(rng.gauss(0, 1), 6) for _ in range(d)]
        return th, y, mu
    # the degenerate law: y - mu in the range, in double (the null-space basis
    # in exact arithmetic is that of the S_j at theta = 0)
    with mp.workdps(DPS_CHECK):
        _, Mp = range_part(s, build(s, [0] * len(S)))
        M = build(s, [0] * len(S))
        Pr = M * Mp  # the projector on the range
        z = [rng.gauss(0, 1) for _ in range(d)]
        r = [mp.fsum(Pr[i, j] * z[j] for j in range(d)) for i in range(d)]
    y = [float(mp.mpf(mu[i]) + r[i]) for i in range(d)]
    return th, y, mu


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
    param, s, kind = args
    d = structure(s)[0]
    th, y, mu = point(param, s, kind)
    q = len(th)
    rng = random.Random(1000 * param + 10 * s + kind)
    head = f"{param},{s},{d},{hexs(th)},{hexs(y)},{hexs(mu)}"
    rows = []
    for A, kmax, all_up in ((0, 6, 2), (1, 5, 1), (2, 4, 1)):
        entries = [()] if A == 0 else [(i,) for i in range(d)] if A == 1 else [(i, j) for i in range(d) for j in range(i + 1)]
        for K in range(kmax + 1):
            for t in tuples(q, K, rng, all_up):
                for e in entries:
                    v = checked(lambda: block_value(param, s, th, y, mu, e, t))
                    ent = 0 if A == 0 else e[0] if A == 1 else e[0] * (e[0] + 1) // 2 + e[1]
                    hi, lo = split(v)
                    rows.append(f"{head},{A},{';'.join(map(str, t)) or '-'},{ent},{hi.hex()},{lo.hex()}")
    pairs = [(a, b) for a in range(q) for b in range(a + 1)]
    for K in range(5):
        ps = pairs if K == 0 else rng.sample(pairs, min(3, len(pairs)))
        for (a, b) in ps:
            for t in tuples(q, K, rng, 0)[:2]:
                v = checked(lambda: info_value(param, s, th, a, b, list(t)))
                hi, lo = split(v)
                rows.append(f"{head},-1,{';'.join(map(str, t)) or '-'},{a * (a + 1) // 2 + b},{hi.hex()},{lo.hex()}")
    return rows


def main():
    jobs = [(param, s, kind) for param in (1, 2) for s in range(4) for kind in (0, 1)]
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(task, jobs, chunksize=1)
    rows = [r for rs in res for r in rs]
    OUT.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    print(f"{len(rows)} rows to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
