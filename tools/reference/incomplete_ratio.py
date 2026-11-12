"""Reference values of the partial derivatives of log P(a, a c) and
log Q(a, a c) in the shape a (at fixed c) and the ratio c = x/a, total
orders 1 to 6, and of the derivatives in x of log P and log Q at integer a
near the mean (the Poisson cdf), computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/incomplete_ratio.py [n_ratio] [workers]

Files (format as in incomplete.py: the arguments, then hi, lo):
    log_gamma_p_ratio_mixed_d{K}.csv, log_gamma_q_ratio_mixed_d{K}.csv   a, c, j
        d^K / da^(K-j) dc^j of log P(a, a c) and log Q(a, a c), j = 0..K
    log_gamma_p_mixed_poisson_d{K}.csv, log_gamma_q_mixed_poisson_d{K}.csv   a, x, j
        d^K / da^(K-j) dx^j of log P(a, x) and log Q(a, x) at integer a from
        10 to 1000 and x = a + t sqrt(a), t in {-4, -3, -2, -1, -1/2, 1/2, 1, 2,
        3, 4} (x > 0), x = a - 1 (the Poisson cdf at k = mu) and x = a; and
        at a = 300, 1000, 3000, 1e4, 3e4, 1e5 with x = a + t sqrt(a), t from
        -3 to 3 in steps of 1/2 (the pure derivatives in x of orders 5 and 6
        near the mean, added on 2026-11-12, night)

Method. The bivariate Taylor series of log S(a + h, x (1 + e)) in the shape
increment h and the relative increment e of x is that of
incomplete_mixed.mixed_row (from the definition: the moments of
int exp(a v - e^v) dv by quadrature and the moving end point; nothing of the
C++ transcription). At fixed c the point (a + h, c (1 + g)) has
x' = x (1 + h/a)(1 + g), so e = h/a + g + h g/a; the series is composed with
this e in mpmath at the working precision (240 or 320 bits), where the
cancellation between the terms in h and in e that the C++ routes avoid costs
nothing, and d^(i+j) / da^i dc^j = i! j! R_ij / c^j.

Independent check: on 12 points the composed values are compared with
mpmath.diff of log gammainc(a, a c) (finite differences on the closed form,
at 300 bits), which shares nothing with the series.

Checks. Every row at 240 and 320 bits (480 and 640 where these disagree),
kept if the two agree to 2^-90 of the largest component of its order at that
point.

Points (n_ratio, default 1200): a from 1/8 to 3.3e4 (the gamma distribution's
a = 1/phi) and c from 1e-3 to 64 (c = y/mu): a quarter with c within
4/sqrt(a) of 1, a quarter log-uniform in c, a quarter at the boundaries of
the routes (a = 2, 20, 300, 1000; |c - 1| = 0.3, 0.4, 0.5; x = 36;
x = a - 1/3), a quarter with a from 16 to 1.6e4 and c from 0.05 to 4; then
a grid at large a near the mean (added on 2026-11-12, night, for the pure
derivatives in c of orders 5 and 6): a = 300, 1000, 3000, 1e4, 3e4, 1e5 and
c = 1 + t/sqrt(a), t from -4 to 4 in steps of 1/2.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incomplete_deriv as idv  # noqa: E402
import incomplete_mixed as im  # noqa: E402

OUT = idv.OUT
K = 6
SEED = "2026-11-12-ratio"


def compose(d, a, c, x):
    """d: {(i, j): d^(i+j) log S / da^i dx^j}; returns {(i, j): d^(i+j) log S(a, a c) / da^i dc^j}"""
    A, X, C = mpmath.mpf(a), mpmath.mpf(x), mpmath.mpf(c)
    # coefficients of h^i e^j
    B = {(i, j): v * X ** j / (mpmath.factorial(i) * mpmath.factorial(j)) for (i, j), v in d.items()}
    # powers of e = h/a + g + h g/a as series {(i, j): coef}, truncated to degree K
    e1 = {(1, 0): 1 / A, (0, 1): mpmath.mpf(1), (1, 1): 1 / A}

    def mul(p, q):
        r = {}
        for (i1, j1), u in p.items():
            for (i2, j2), w in q.items():
                if i1 + i2 + j1 + j2 <= K:
                    r[(i1 + i2, j1 + j2)] = r.get((i1 + i2, j1 + j2), 0) + u * w
        return r

    epow = [{(0, 0): mpmath.mpf(1)}]
    for _ in range(K):
        epow.append(mul(epow[-1], e1))
    R = {}
    for (i, j), b in B.items():
        for (p, q), w in epow[j].items():
            if i + p + q <= K:
                R[(i + p, q)] = R.get((i + p, q), 0) + b * w
    return {(i, j): v * mpmath.factorial(i) * mpmath.factorial(j) / C ** j for (i, j), v in R.items() if i + j >= 1}


def ratio_row(a, c, prec):
    with mpmath.workprec(prec):
        x = mpmath.mpf(a) * mpmath.mpf(c)
        p_d, q_d, S, chk = im.mixed_row(a, x, prec)
        return compose(p_d, a, c, x), compose(q_d, a, c, x), S, chk


def agree(u, v, scale):
    return abs(u - v) <= mpmath.mpf(2) ** -90 * scale or u == v


def order_scale(d, k):
    return max(abs(d[(k - j, j)]) for j in range(k + 1))


def run_rows(fn, args):
    a, c = args
    for lo, hi in ((240, 320), (480, 640)):
        try:
            p1, q1, S1, chk = fn(a, c, lo)
            p2, q2, S2, _ = fn(a, c, hi)
        except Exception as e:  # noqa: BLE001
            return ("error", (a, c), repr(e))
        ok = True
        for k in range(1, K + 1):
            sp, sq = order_scale(p2, k), order_scale(q2, k)
            for j in range(k + 1):
                key = (k - j, j)
                if not (agree(p1[key], p2[key], sp) and agree(q1[key], q2[key], sq)):
                    ok = False
        if ok:
            break
    sok = abs(S2 - chk) <= mpmath.mpf(2) ** -90 * abs(chk)
    return ("ok" if ok and sok else ("bad_s" if not sok else "bad"), (a, c), (p2, q2))


def work_ratio(args):
    return run_rows(ratio_row, args)


def work_poisson(args):
    return run_rows(lambda a, x, prec: im.mixed_row(a, x, prec), args)


def ratio_args(rng, n):
    out = []
    lu = idv.log_uniform
    for i in range(n):
        k = i % 4
        if k == 0:  # c near 1
            a = lu(rng, -3, 13)
            c = 1.0 + rng.uniform(-4.0, 4.0) / math.sqrt(a)
        elif k == 1:  # c anywhere
            a = lu(rng, -3, 13)
            c = lu(rng, -10, 5)
        elif k == 2:  # the boundaries of the routes
            a = rng.choice([2.0, 20.0, 300.0, 1000.0, lu(rng, 1, 13)]) * (1.0 + rng.uniform(-0.02, 0.02))
            u = rng.randrange(4)
            if u == 0:
                c = 1.0 + rng.choice([-1.0, 1.0]) * rng.choice([0.3, 0.4, 0.5]) * (1.0 + rng.uniform(-0.03, 0.03))
            elif u == 1:
                c = 36.0 / a * (1.0 + rng.uniform(-0.03, 0.03))
            elif u == 2:
                c = (a - 1.0 / 3.0) / a * (1.0 + rng.uniform(-0.01, 0.01))
            else:
                c = lu(rng, -3, 3)
        else:  # large a
            a = lu(rng, 4, 13)
            c = math.exp(rng.uniform(math.log(0.05), math.log(4.0)))
        if c > 0.0:
            out.append((a, c))
    return out


LARGE_A = (300.0, 1000.0, 3000.0, 1e4, 3e4, 1e5)


def large_ratio_args():
    """large a, c within 4/sqrt(a) of 1"""
    return [(a, 1.0 + 0.5 * i / math.sqrt(a)) for a in LARGE_A for i in range(-8, 9)]


def poisson_args():
    out = []
    for a in (10, 11, 12, 20, 25, 39, 40, 41, 42, 60, 100, 150, 299, 300, 301, 500, 700, 1000):
        for t in (-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4):
            x = a + t * math.sqrt(a)
            if t == 0:
                x = a - 1.0  # the Poisson cdf at k = mu (a = k + 1, x = mu = a - 1)
            if x > 0.0:
                out.append((float(a), float(x)))
        out.append((float(a), float(a)))
    for a in LARGE_A:
        for i in range(-6, 7):
            out.append((a, a + 0.5 * i * math.sqrt(a)))
    return out


def write(path, rows):
    with path.open("w", newline="\n") as fh:
        fh.write("a,x,j,hi,lo\n")
        for args, v in rows:
            hi, lo = idv.dd(v)
            fh.write(",".join(float.hex(float(t)) for t in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
    print(f"{path}: {len(rows)} rows", flush=True)


def independent_check():
    """composed values against mpmath.diff of the closed form"""
    worst = mpmath.mpf(0)
    for a, c in ((50.0, 1.0), (50.0, 1.08), (0.25, 0.4), (3.3, 0.7), (1000.0, 0.97), (20.5, 1.3),
                 (2.0, 0.01), (7.0, 3.0), (300.0, 1.45), (0.05, 2.0), (120.0, 0.6), (9.0, 9.0)):
        p, q, _, _ = ratio_row(a, c, 300)
        with mpmath.workprec(300):
            for upper, d in ((0, p), (1, q)):
                def f(A, C):
                    X = A * C
                    if upper:
                        return mpmath.log(mpmath.gammainc(A, X, mpmath.inf, regularized=True))
                    return mpmath.log(mpmath.gammainc(A, 0, X, regularized=True))
                for k in (1, 3, 6):
                    sc = order_scale(d, k)
                    for j in range(k + 1):
                        v = mpmath.diff(f, (mpmath.mpf(a), mpmath.mpf(c)), (k - j, j))
                        worst = max(worst, abs(v - d[(k - j, j)]) / sc)
    print(f"independent check (mpmath.diff, 12 points, orders 1, 3, 6): max |diff| / order scale = "
          f"{mpmath.nstr(worst, 3)}", flush=True)


def main():
    n_ratio = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    independent_check()
    args = ratio_args(random.Random(SEED), n_ratio) + large_ratio_args()
    pargs = poisson_args()
    with Pool(workers) as pool:
        res = pool.map(work_ratio, args, chunksize=4)
        pres = pool.map(work_poisson, pargs, chunksize=2)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, rs, fname in (("ratio", res, "log_gamma_{}_ratio_mixed_d{}.csv"),
                           ("poisson", pres, "log_gamma_{}_mixed_poisson_d{}.csv")):
        counts = {}
        for r in rs:
            counts[r[0]] = counts.get(r[0], 0) + 1
            if r[0] != "ok" and counts[r[0]] <= 5:
                print("dropped", tag, r[0], r[1], r[2] if r[0] == "error" else "", flush=True)
        print(tag, "counts", counts, flush=True)
        good = [r for r in rs if r[0] == "ok"]
        for side, name in ((0, "p"), (1, "q")):
            for k in range(1, K + 1):
                rows = []
                for r in good:
                    for j in range(k + 1):
                        rows.append(((r[1][0], r[1][1], float(j)), r[2][side][(k - j, j)]))
                write(OUT / fname.format(name, k), rows)


if __name__ == "__main__":
    main()
