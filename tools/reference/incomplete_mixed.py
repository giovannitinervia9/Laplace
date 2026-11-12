"""Reference values of the mixed partial derivatives of log P(a, x) and
log Q(a, x) in the shape a and the argument x, total orders 1 to 6, computed
with mpmath.

Usage (from the repository root):
    python -I tools/reference/incomplete_mixed.py [n_extra] [workers] [n_poisson]

Files (format as in incomplete.py: the arguments, then hi, lo):
    log_gamma_p_mixed_d{K}.csv, log_gamma_q_mixed_d{K}.csv   a, x, j
holding d^K / da^(K-j) dx^j of log P and log Q, j = 0..K (j derivatives in
x), K = 1..6. The third column is the component j, written as a double.

Method (from the definition; independent of the C++ transcription, which
works with the cumulants of d = log(t/x) under a measure that depends on x).
With v = log t the incomplete integral of one side is
    J(a, x) = int exp(a v - e^v) dv  over v < log x (lower) or v > log x (upper),
its Taylor series in the shape increment h is formed from the moments
int v^n exp(a v - e^v) dv by the quadrature of incomplete_deriv.py (the same
cuts and checks), and the increment of x by eps moves only the end point:
    J(a + h, x + eps) = J(a + h, x) +/- int_0^eps (x + s)^(a + h - 1) e^-(x + s) ds,
+ for the lower side and - for the upper one. The integrand of the last term
is exp((a - 1 + h) log(x + s) - x - s), whose bivariate Taylor series in
(h, s) is exp of a series with known coefficients
(log(x + s) = log x + sum_m (-1)^(m-1) (s/x)^m / m); it is integrated in s term
by term. Then log S = log J - log Gamma(a + h) and log(1 - S) are formed as
bivariate series in h and the relative increment e = eps/x (in eps itself
the coefficients grow like (a/x)^j and the cumulants with i >= 2 cancelled
beyond 240 bits in the far lower tail), and the derivative is the
coefficient times i! j! / x^j.

Points: the 1500 points of incomplete_deriv.py (kind gamma), the 400 points
at the route boundaries of its order 6 (edge_args), and n_extra points
(default 600) chosen for the mixed partials: small x (down to 1e-30) for
a < 2, a near 1 (where log Q = -x at a = 1 and its x-derivatives vanish), x
near a for a from 1 to 1e4, the far tails, the closed form at small x and the
boundaries x = 3/2 and x = 36; and n_poisson points (default 600) at integer
a from 1 to 1000 with x in [0.05, 1000] (the Poisson cdf F(k) = Q(k + 1, mu)),
a quarter each with x near a, x far below a (down to a/1e4), x far above a,
and x log-uniform.

Checks. Every row is computed at two precisions (240 and 320 bits; again at
480 and 640 bits where these disagree: in the far lower tail, x/a below about
1e-6, the cumulants with i >= 2 cancel by 1e40 and more), and kept only if
the two agree to 2^-100 of max(|f|, 1) (the dropped rows are
counted). The
value S itself is compared with mpmath's gammainc on every row.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incomplete_deriv as idv  # noqa: E402

OUT = idv.OUT
K = 6


def boundary_series(A, X, lref):
    """int_0^eps (x + s)^(a + h - 1) e^-(x + s) ds e^-lref as a bivariate
    series {(i, j): c} in (h, e), e = eps/x the relative increment, i + j <= K"""
    lx = mpmath.log(X)
    # l(s) = log(1 + s/x) as a series in s: (0, m) -> (-1)^(m-1) / (m x^m)
    ell = {(0, m): mpmath.mpf((-1) ** (m - 1)) / (m * X ** m) for m in range(1, K)}
    # U = (a - 1) l(s) + h log x + h l(s) - s
    U = idv.s_scale(ell, A - 1)
    U[(1, 0)] = lx
    for (i, m), c in ell.items():
        if 1 + m <= K - 1:
            U[(1, m)] = U.get((1, m), 0) + c
    U[(0, 1)] = U.get((0, 1), 0) - 1
    U[(0, 0)] = (A - 1) * lx - X - lref
    phi = idv.s_exp(U)
    # integrate in s: (i, m) -> (i, m + 1) / (m + 1); then the relative
    # increment e = eps / x: (i, m + 1) times x^(m + 1)
    return {(i, m + 1): c * X ** (m + 1) / (m + 1) for (i, m), c in phi.items() if i + m + 1 <= K}


def mixed_row(a, x, prec):
    idv.KMAX = K
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
        pts = idv.cut_points(lo, hi, c, wl, wr, lx, w_end)
        vref = lx if (lower and lx < c) or (not lower and lx > c) else c
        lref = A * vref - mpmath.exp(vref)
        cutoff = -(prec + 200)

        def F(v):
            expo = A * v - mpmath.exp(v) - lref
            if expo < cutoff:
                return [mpmath.mpf(0)] * (K + 1)
            e = mpmath.exp(expo)
            out = [e]
            for _ in range(K):
                out.append(out[-1] * v)
            return out

        m = idv.integrate_all(F, pts, K + 1)
        J = {(n, 0): m[n] / mpmath.factorial(n) for n in range(K + 1)}
        bnd = boundary_series(A, X, lref)
        J = idv.s_add(J, bnd, 1 if lower else -1)
        logS = idv.s_add(idv.s_log(J), idv.lgamma_series(A, 0), -1)
        logS[(0, 0)] += lref
        S = mpmath.exp(logS[(0, 0)])
        logB = idv.s_log(idv.s_add({(0, 0): mpmath.mpf(1)}, idv.s_exp(logS), -1))
        check = mpmath.gammainc(A, 0, X, regularized=True) if lower else mpmath.gammainc(A, X, mpmath.inf, regularized=True)
        # derivatives in (h, e), then d/dx = (1/x) d/de
        dS, dB = idv.derivatives(logS), idv.derivatives(logB)
        dS = {k: v / X ** k[1] for k, v in dS.items()}
        dB = {k: v / X ** k[1] for k, v in dB.items()}
        p_d, q_d = (dS, dB) if lower else (dB, dS)
        return p_d, q_d, S, check


def agree1(u, v):
    """agreement to 2^-100 of max(|v|, 1): the figure of the tests is the
    smaller of the relative error and the error in units of max(|f|, 1), and
    components far below 1 (the cumulants with i >= 2 in the far lower tail,
    1e-14 next to terms of order (a/x)^j) cancel beyond 240 bits relative"""
    return abs(u - v) <= mpmath.mpf(2) ** -100 * max(abs(v), 1) or u == v


def work(args):
    a, x = args
    for lo, hi in ((240, 320), (480, 640)):
        try:
            p1, q1, S1, chk = mixed_row(a, x, lo)
            p2, q2, S2, _ = mixed_row(a, x, hi)
        except Exception as e:  # noqa: BLE001
            return ("error", (a, x), repr(e))
        keys = sorted(p2)
        ok = all(agree1(p1[k], p2[k]) and agree1(q1[k], q2[k]) for k in keys)
        if ok:
            break
    sok = abs(S2 - chk) <= mpmath.mpf(2) ** -90 * abs(chk)
    return ("ok" if ok and sok else ("bad_s" if not sok else "bad"), (a, x), (p2, q2))


def mixed_args(rng, n):
    """points for the derivatives in x"""
    out = []
    lu = idv.log_uniform
    for i in range(n):
        k = i % 8
        if k == 0:  # small x, a < 2: the closed form, the lower series, the pole of a
            a = lu(rng, -12, 1)
            x = lu(rng, -100, 0)
        elif k == 1:  # a near 1, x anywhere above 1/2 (pure x derivatives near zero)
            a = 1.0 + rng.choice([-1.0, 1.0]) * lu(rng, -40, -2)
            x = lu(rng, -1, 9)
        elif k == 2:  # x near a
            a = lu(rng, 0, 13)
            x = a + rng.uniform(-3.0, 3.0) * math.sqrt(a)
        elif k == 3:  # far upper tail
            a = lu(rng, -8, 13)
            x = a + math.sqrt(a) * rng.uniform(5.0, 60.0) + rng.uniform(0.0, 300.0)
        elif k == 4:  # far lower tail
            a = lu(rng, 0, 13)
            x = a * lu(rng, -40, -2)
        elif k == 5:  # the boundaries x = 3/2, 36 and x = a +/- 0.75 sqrt(a)
            a = lu(rng, -6, 5)
            x = rng.choice([1.5, 36.0, a + 0.75 * math.sqrt(a), a - 1.0 / 3.0]) * (1.0 + rng.uniform(-0.03, 0.03))
        elif k == 6:  # Temme's region and its boundaries
            a = lu(rng, 4, 13)
            x = a * (1.0 + rng.uniform(-0.55, 0.55))
        else:  # tiny a
            a = lu(rng, -12, -4)
            x = lu(rng, -30, 7)
        if x > 0.0:
            out.append((a, x))
    return out


def poisson_args(rng, n):
    """integer a = k + 1 (the Poisson cdf F(k) = Q(k + 1, mu) at x = mu), a
    from 1 to 1000, x from 0.05 to 1000"""
    out = []
    for i in range(n):
        a = float(min(1000, max(1, int(round(math.exp(rng.uniform(0.0, math.log(1000.0))))))))
        k = i % 4
        if k == 0:  # x near a (k near mu)
            x = a + rng.uniform(-4.0, 4.0) * math.sqrt(a)
        elif k == 1:  # x much smaller than a (F near 1, 1 - F ~ x^a / a!)
            x = a * 10.0 ** -rng.uniform(0.3, 4.0)
        elif k == 2:  # x much larger than a
            x = a + math.sqrt(a) * rng.uniform(4.0, 40.0) + rng.uniform(0.0, 50.0)
        else:
            x = math.exp(rng.uniform(math.log(0.05), math.log(1000.0)))
        if 0.05 <= x <= 1000.0:
            out.append((a, x))
    return out


def write(path, rows):
    with path.open("w", newline="\n") as fh:
        fh.write("a,x,j,hi,lo\n")
        for args, v in rows:
            hi, lo = idv.dd(v)
            fh.write(",".join(float.hex(float(t)) for t in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
    print(f"{path}: {len(rows)} rows", flush=True)


def main():
    n_extra = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_poisson = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    common = idv.gamma_args(random.Random(f"{idv.SEED}-gamma"), 1500)
    edges = idv.edge_args(random.Random(f"{idv.SEED}-gamma6-edges"), 400)
    extra = mixed_args(random.Random(f"{idv.SEED}-gamma-mixed"), n_extra)
    poisson = poisson_args(random.Random(f"{idv.SEED}-gamma-mixed-poisson"), n_poisson)
    args = common + edges + extra + poisson
    with Pool(workers) as pool:
        res = pool.map(work, args, chunksize=4)
    counts = {}
    for r in res:
        counts[r[0]] = counts.get(r[0], 0) + 1
        if r[0] != "ok" and counts[r[0]] <= 5:
            print("dropped", r[0], r[1], r[2] if r[0] == "error" else "", flush=True)
    print("counts", counts, flush=True)
    good = [r for r in res if r[0] == "ok"]
    # the pure-a components against the files of incomplete_deriv.py on the common points
    for side, name in ((0, "p"), (1, "q")):
        for k in range(1, K + 1):
            old = idv.read_ref(OUT / f"log_gamma_{name}_da{k}.csv")
            worst, cnt = mpmath.mpf(0), 0
            for r in good:
                key = (r[1][0], r[1][1])
                if key in old:
                    v = r[2][side][(k, 0)]
                    worst = max(worst, abs(v - old[key]) / max(abs(v), 1))
                    cnt += 1
            print(f"order {k} {name} pure a: {cnt} common rows, max |new - old| / max(|f|, 1) = {mpmath.nstr(worst, 3)}",
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for side, name in ((0, "p"), (1, "q")):
        for k in range(1, K + 1):
            rows = []
            for r in good:
                for j in range(k + 1):
                    rows.append(((r[1][0], r[1][1], float(j)), r[2][side][(k - j, j)]))
            write(OUT / f"log_gamma_{name}_mixed_d{k}.csv", rows)


if __name__ == "__main__":
    main()
