"""Reference values for the coupled bijectors (alr, log_cholesky, partial_corr,
matrix_log; include/laplace/bijector/) and their derivatives to order 6.

Usage (from the repository root):  python -I tools/reference/bijector_coupled.py

Files in tests/data/reference/:
  bijector_coupled_points.csv  map,dim,point,side,index,value
      side 0: the free vector u of the point; side 1: the constrained value v
      (alr: pi, K entries; matrices: packed lower triangle by rows), the
      double nearest to h(u).
  bijector_coupled.csv  map,dim,point,direction,quantity,order,t0,...,t5,out,hi,lo
      direction 0 inverse (at u), 1 forward (at v); quantity 0 the map,
      1 the log-Jacobian, 2 the Cholesky factor L(u) (inverse only);
      t0..t5 the sorted tuple (-1 beyond the order); out the output entry
      (0 for the log-Jacobian). hi is the nearest double to the value, lo
      the nearest double to the remainder (float.hex).
Maps: 0 alr, 1 log_cholesky, 2 partial_corr, 3 matrix_log.

Definitions (independent of the C++ formulas):
  alr          pi_a = e^(u_a) / (1 + sum_b e^(u_b)), pi_K = 1 / (...); forward
               u_a = log pi_a - log pi_K with pi_K = v_K - sum_b (pi_b - v_b)
               (the double pi_K as given, moved as 1 - sum of the others).
  log_cholesky L_ii = e^(u_ii), L_ij = u_ij, Sigma = L L^T; forward the
               Cholesky factor of Sigma (mpmath), u_ii = log L_ii.
  partial_corr z = tanh u; row a of L built from w = 1 - sum_(l<b) L_al^2:
               L_ab = z_ab sqrt(w), L_aa = sqrt(w_a) (the onion construction
               as written in the literature, not the sech product of the
               C++); forward u_ab = atanh(L_ab / sqrt(1 - sum_(l<b) L_al^2))
               from the mpmath Cholesky factor of R.
  matrix_log   Sigma = exp(S) (mp.expm); forward log Sigma = Q diag(log e) Q^T
               from mp.eigsy (Sigma = Q diag(e) Q^T).
  log-Jacobians: inverse  alr sum_a log pi_a; log_cholesky
               p log 2 + sum_i (p - i + 1) u_ii; partial_corr
               -sum_(a>b) (p - b) log cosh u_ab; matrix_log
               sum_(i<=j) log f[s_i, s_j], f[x, y] = (e^x - e^y)/(x - y),
               f[x, x] = e^x, s the eigenvalues of S (mp.eigsy); forward the
               negatives at g(v). The first three are checked against the
               determinant of the symbolic Jacobian in bijector_coupled_forms.py;
               the matrix_log one comes from the eigenvalues of the derivative
               of exp (note/bijectors.md section 4).

Method. Inverse directions of alr, log_cholesky, partial_corr (and the alr
forward): exact SymPy derivatives of the definitions above, evaluated by
mpmath at 300 bits. Forward directions of log_cholesky and partial_corr, and
both directions of matrix_log: mixed central differences of the definitions
at 1200 bits, step h = 2^-130, repeated with h = 2^-140; the run stops if the
two differ by more than 1e-40 times (1 + the largest entry of that tuple and
quantity). Tuples: all of orders 1 and 2; for orders 3..6 every pure tuple
and up to 24 mixed ones drawn at random (seeded), 12 for matrix_log. The
matrix_log points include S = 0 (Sigma = I, every eigenvalue equal).
"""
import itertools
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
OUT_POINTS = ROOT / "bijector_coupled_points.csv"
OUT = ROOT / "bijector_coupled.csv"
KMAX = 6
MIXED = 24
EXACT_PREC = 300
FD_PREC = 1200
H1 = mp.mpf(2) ** -130
H2 = mp.mpf(2) ** -140
TOL = mp.mpf(10) ** -40


def packed(i, j):
    return i * (i + 1) // 2 + j


def strict(i, j):
    return i * (i - 1) // 2 + j


def hex_pair(v):
    if v == 0:
        return "0x0.0p+0", "0x0.0p+0"
    hi = float(v)
    lo = float(v - mp.mpf(hi))
    return float.hex(hi), float.hex(lo)


def tuples_for(n, rng, mixed_ok=None, mixed=MIXED):
    """Sorted tuples per order: all for orders 1-2, pure + a sample beyond."""
    out = []
    for k in range(1, KMAX + 1):
        allt = list(itertools.combinations_with_replacement(range(n), k))
        if k <= 2:
            out += allt
            continue
        pure = [t for t in allt if t[0] == t[-1]]
        mixed_t = [t for t in allt if t[0] != t[-1] and (mixed_ok is None or mixed_ok(t))]
        rng.shuffle(mixed_t)
        out += pure + sorted(mixed_t[:mixed])
    return out


# ------------------------------------------------------------ symbolic maps
def sym_alr(K):
    us = sp.symbols(f"u0:{K - 1}", real=True)
    D = 1 + sum(sp.exp(v) for v in us)
    pis = [sp.exp(v) / D for v in us] + [1 / D]
    lj = sum(us) - K * sp.log(D)  # sum_a log pi_a, written without log of a quotient
    return list(us), pis, lj, None


def sym_log_cholesky(p):
    keys = [(i, j) for i in range(p) for j in range(i + 1)]
    us = [sp.Symbol(f"u{i}_{j}", real=True) for i, j in keys]
    L = sp.zeros(p, p)
    for (i, j), s in zip(keys, us):
        L[i, j] = sp.exp(s) if i == j else s
    S = L * L.T
    outs = [S[i, j] for i, j in keys]
    lj = p * sp.log(2) + sum((p - i + 1) * us[packed(i, i)] for i in range(p))
    return us, outs, lj, [L[i, j] for i, j in keys]


def sym_partial_corr(p):
    keys = [(a, b) for a in range(p) for b in range(a)]
    us = [sp.Symbol(f"u{a}_{b}", real=True) for a, b in keys]
    z = {k: sp.tanh(s) for k, s in zip(keys, us)}
    L = sp.zeros(p, p)
    for a in range(p):
        acc = sp.Integer(0)
        for b in range(a):
            L[a, b] = z[a, b] * sp.sqrt(1 - acc)
            acc += L[a, b] ** 2
        L[a, a] = sp.sqrt(1 - acc)
    R = L * L.T
    outs = [sp.Integer(1) if i == j else R[i, j] for i in range(p) for j in range(i + 1)]
    lj = -sum((p - b) * sp.log(sp.cosh(us[strict(a, b)])) for a, b in keys)
    return us, outs, lj, [L[i, j] for i in range(p) for j in range(i + 1)]


SYM = {0: sym_alr, 1: sym_log_cholesky, 2: sym_partial_corr}


def exact_rows(mp_id, dim, pt, upt, rng, mixed_ok):
    """Inverse direction: exact derivatives of the symbolic definitions."""
    us, outs, lj, chol = SYM[mp_id](dim)
    quants = [(0, outs), (1, [lj])] + ([(2, chol)] if chol is not None else [])
    tups = tuples_for(len(us), rng, mixed_ok)
    rows = []
    for q, exprs in quants:
        memo = {(): exprs}

        def deriv(t):
            if t not in memo:
                memo[t] = [sp.diff(e, us[t[-1]]) for e in deriv(t[:-1])]
            return memo[t]

        for t in [()] + tups:
            d = deriv(t)
            f = sp.lambdify(us, d, "mpmath")
            rows += emit(mp_id, dim, pt, 0, q, len(t), t, f(*upt))
    return rows


def alr_forward_rows(dim, pt, vpt, rng):
    """alr forward and its log-Jacobian at the double pi: exact derivatives."""
    ps = sp.symbols(f"p0:{dim - 1}", real=True)
    vK = vpt[-1]
    piK = sp.Float(vK, 120) - sum(p - sp.Float(v, 120) for p, v in zip(ps, vpt[:-1]))
    outs = [sp.log(p) - sp.log(piK) for p in ps]
    lj = -sum(sp.log(p) for p in ps) - sp.log(piK)
    at = [mp.mpf(v) for v in vpt[:-1]]
    rows = []
    for q, exprs in ((0, outs), (1, [lj])):
        for t in [()] + tuples_for(dim - 1, rng):
            d = exprs
            for v in t:
                d = [sp.diff(e, ps[v]) for e in d]
            rows += emit(0, dim, pt, 1, q, len(t), t, sp.lambdify(ps, d, "mpmath")(*at))
    return rows


# ------------------------------------------------------------ forward by differences
def fwd_log_cholesky(p, x):
    """x: the packed lower triangle; returns (u, log-Jacobian)."""
    S = mp.matrix(p, p)
    for i in range(p):
        for j in range(i + 1):
            S[i, j] = S[j, i] = x[packed(i, j)]
    L = mp.cholesky(S)
    u = [mp.log(L[i, j]) if i == j else L[i, j] for i in range(p) for j in range(i + 1)]
    lj = -(p * mp.log(2) + sum((p - i + 1) * mp.log(L[i, i]) for i in range(p)))
    return u, lj


def fwd_partial_corr(p, x):
    """x: the strict lower triangle; returns (u, log-Jacobian)."""
    R = mp.matrix(p, p)
    for i in range(p):
        R[i, i] = 1
        for j in range(i):
            R[i, j] = R[j, i] = x[strict(i, j)]
    L = mp.cholesky(R)
    u = [None] * (p * (p - 1) // 2)
    lj = mp.mpf(0)
    for a in range(1, p):
        acc = mp.mpf(0)
        for b in range(a):
            ua = mp.atanh(L[a, b] / mp.sqrt(1 - acc))
            u[strict(a, b)] = ua
            lj += (p - b) * mp.log(mp.cosh(ua))
            acc += L[a, b] ** 2
    return u, lj


def sym_of(p, x):
    M = mp.matrix(p, p)
    for i in range(p):
        for j in range(i + 1):
            M[i, j] = M[j, i] = x[packed(i, j)]
    return M


def log_dd_sum(lam):
    """sum_(i<=j) log f[l_i, l_j], f[x, y] = (e^x - e^y)/(x - y), f[x, x] = e^x,
    as (x + y)/2 + log(sinh(z)/z), z = (x - y)/2: no cancellation when two
    eigenvalues coincide up to rounding (repeated eigenvalues at S = 0)."""
    s = mp.mpf(0)
    for i in range(len(lam)):
        for j in range(i, len(lam)):
            x, y = lam[i], lam[j]
            z = (x - y) / 2
            s += (x + y) / 2 + (mp.log(mp.sinh(z) / z) if z != 0 else 0)
    return s


def inv_matrix_log(p, x):
    """x: the packed lower triangle of S; returns (exp(S) packed, inverse log-Jacobian)."""
    S = sym_of(p, x)
    E = mp.expm(S)
    lam = mp.eigsy(S)[0]
    return [E[i, j] for i in range(p) for j in range(i + 1)], log_dd_sum([lam[i] for i in range(p)])


def fwd_matrix_log(p, x):
    """x: the packed lower triangle of Sigma; returns (log(Sigma) packed, forward log-Jacobian)."""
    Sg = sym_of(p, x)
    E, Q = mp.eigsy(Sg)
    lam = [mp.log(E[i]) for i in range(p)]
    X = Q * mp.diag(lam) * Q.T
    return [X[i, j] for i in range(p) for j in range(i + 1)], -log_dd_sum(lam)


# (map, direction) -> function of the input returning (outputs, log-Jacobian)
FWD = {(1, 1): fwd_log_cholesky, (2, 1): fwd_partial_corr, (3, 0): inv_matrix_log, (3, 1): fwd_matrix_log}
MIXED_BY_MAP = {3: 12}


def stencil(f, x0, t, h, cache):
    """Mixed central difference of f along the tuple t at x0 (vector outputs)."""
    vars_ = sorted(set(t))
    mult = [t.count(v) for v in vars_]
    acc = None
    for ks in itertools.product(*[range(m + 1) for m in mult]):
        coef = 1
        x = list(x0)
        key = []
        for v, m, k in zip(vars_, mult, ks):
            coef *= (-1) ** k * math.comb(m, k)
            off = m - 2 * k  # offset (m/2 - k) h, in units of h/2
            x[v] = x[v] + off * h / 2
            key.append((v, off))
        key = tuple(key)
        if key not in cache:
            cache[key] = f(x)
        val = cache[key]
        acc = [coef * y for y in val] if acc is None else [a + coef * y for a, y in zip(acc, val)]
    return [a / h ** len(t) for a in acc]


def fd_task(args):
    mp_id, dim, pt, x0f, seed, direction = args
    mp.mp.prec = FD_PREC
    rng = random.Random(seed)
    x0 = [mp.mpf(v) for v in x0f]
    fwd = FWD[(mp_id, direction)]

    def f(x):
        u, lj = fwd(dim, x)
        return u + [lj]

    n = len(x0)
    val = f(x0)
    rows = emit(mp_id, dim, pt, direction, 0, 0, (), val[:-1]) + emit(mp_id, dim, pt, direction, 1, 0, (), val[-1:])
    worst = mp.mpf(0)
    c1, c2 = {}, {}
    for t in tuples_for(n, rng, mixed=MIXED_BY_MAP.get(mp_id, MIXED)):
        d1 = stencil(f, x0, t, H1, c1)
        d2 = stencil(f, x0, t, H2, c2)
        for q, sl in ((0, slice(0, n)), (1, slice(n, n + 1))):
            a, b = d1[sl], d2[sl]
            scale = 1 + max(abs(y) for y in a)
            dev = max(abs(y - z) for y, z in zip(a, b)) / scale
            worst = max(worst, dev)
            if dev > TOL:
                raise SystemExit(f"h-check failed: map {mp_id} p {dim} point {pt} tuple {t} quantity {q}: {mp.nstr(dev, 5)}")
            rows += emit(mp_id, dim, pt, direction, q, len(t), t, a)
    return rows, worst


def emit(mp_id, dim, pt, direction, q, order, t, vals):
    tt = list(t) + [-1] * (KMAX - len(t))
    ts = ",".join(str(x) for x in tt)
    out = []
    for e, v in enumerate(vals):
        hi, lo = hex_pair(mp.mpf(v))
        out.append(f"{mp_id},{dim},{pt},{direction},{q},{order},{ts},{e},{hi},{lo}")
    return out


# ------------------------------------------------------------ points
def points():
    """(map, dim, point, u as mpf strings) for every point."""
    rng = random.Random(20261112)
    out = []
    for K in (2, 3, 5):
        us = [[rng.gauss(0, 1.5) for _ in range(K - 1)] for _ in range(3)]
        us.append([(-30.0 + 5 * (a // 2) if a % 2 == 0 else 20.0 - 5 * (a // 2)) for a in range(K - 1)])  # spread, no ties
        us.append([25.0 + a for a in range(K - 1)])  # tiny pi_K
        out += [(0, K, i, u) for i, u in enumerate(us)]
    for p in (1, 2, 3, 4):
        n = p * (p + 1) // 2
        us = [[rng.gauss(0, 0.7) for _ in range(n)] for _ in range(3)]
        spread = [0.3 * rng.gauss(0, 1) for _ in range(n)]
        for i in range(p):
            spread[packed(i, i)] = 4.0 * (-1) ** i
        us.append(spread)
        out += [(1, p, i, u) for i, u in enumerate(us)]
    for p in (2, 3, 4):
        n = p * (p - 1) // 2
        us = [[rng.gauss(0, 1.0) for _ in range(n)] for _ in range(3)]
        # |z| near 1, with R kept moderately conditioned (p = 4 at 4.0: condition 4e8)
        amp = {2: 4.0, 3: 2.5, 4: 1.5}[p]
        us.append([amp * (-1) ** k for k in range(n)])
        out += [(2, p, i, u) for i, u in enumerate(us)]
    for p in (1, 2, 3, 4):
        n = p * (p + 1) // 2
        us = [[rng.gauss(0, 0.5) for _ in range(n)] for _ in range(3)]
        us.append([0.0] * n)  # S = 0: Sigma = I, every eigenvalue equal
        spread = [0.2 * rng.gauss(0, 1) for _ in range(n)]
        for i in range(p):
            spread[packed(i, i)] = 2.0 * (-1) ** i
        us.append(spread)
        out += [(3, p, i, u) for i, u in enumerate(us)]
    return out


def value_of(mp_id, dim, u):
    """h(u) at EXACT_PREC as the stored value v."""
    if mp_id == 3:
        return inv_matrix_log(dim, u)[0]
    us, outs, _, _ = SYM[mp_id](dim)
    return [sp.lambdify(us, e, "mpmath")(*u) for e in outs]


def two_row_ok(dim):
    """partial_corr: mixed tuples on at most two rows (the others are zero)."""
    def ok(t):
        rows = set()
        for k in t:
            a = 1
            while (a + 1) * a // 2 <= k:
                a += 1
            rows.add(a)
        return len(rows) <= 2
    return ok


def main():
    t0 = time.time()
    mp.mp.prec = EXACT_PREC
    rows, prows = [], []
    fd_jobs = []
    for mp_id, dim, pt, uf in points():
        u = [mp.mpf(x) for x in uf]
        v = value_of(mp_id, dim, u)
        for k, x in enumerate(uf):
            prows.append(f"{mp_id},{dim},{pt},0,{k},{float.hex(float(x))}")
        vd = [float(x) for x in v]
        for k, x in enumerate(vd):
            prows.append(f"{mp_id},{dim},{pt},1,{k},{float.hex(x)}")
        rng = random.Random(1000 * mp_id + 10 * dim + pt)
        if mp_id != 3:
            rows += exact_rows(mp_id, dim, pt, u, rng, two_row_ok(dim) if mp_id == 2 else None)
        if mp_id == 3:
            fd_jobs.append((3, dim, pt, uf, 11 * pt + dim, 0))
            fd_jobs.append((3, dim, pt, vd, 13 * pt + dim, 1))
        elif mp_id == 0:
            rows += alr_forward_rows(dim, pt, vd, rng)
        elif mp_id == 1:
            fd_jobs.append((1, dim, pt, vd, 7 * pt + dim, 1))
        else:
            fd_jobs.append((2, dim, pt, [vd[packed(a, b)] for a in range(dim) for b in range(a)], 7 * pt + dim, 1))
        print(f"map {mp_id} dim {dim} point {pt}: exact rows done ({time.time() - t0:.0f} s)")
        sys.stdout.flush()
    worst = mp.mpf(0)
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r, w in ex.map(fd_task, fd_jobs):
            rows += r
            worst = max(worst, w)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_POINTS, "w", encoding="utf-8", newline="\n") as f:
        f.write("map,dim,point,side,index,value\n")
        f.write("\n".join(prows) + "\n")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("map,dim,point,direction,quantity,order,t0,t1,t2,t3,t4,t5,out,hi,lo\n")
        f.write("\n".join(rows) + "\n")
    print(f"rows: {len(rows)}, points: {len(prows)}")
    print(f"worst h-check disagreement (relative to 1 + max entry): {mp.nstr(worst, 5)}")
    print(f"run time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
