"""Reference values for enet_distrib (include/laplace/distrib/enet.hpp), the elastic-net
distribution in (mu, lambda, alpha): derivatives of the log-density, of log F and log(1 - F),
of the expected information, and the quantile.

Usage (from the repository root):  python -I tools/reference/distrib_enet.py [quick]

File tests/data/reference/distrib_enet.csv, one row per value:
    kind, K, p, pos, y, hi, lo, mu, lam, alp
kind 0 derivative of order K of log f, kind 1 of log F, kind 2 of log(1 - F), kind 3 the K-th
derivative of the expected information (K = 0: the information; p the pair 0..5 of
sym_tensor<3, 2> = (mu,mu), (mu,lam), (mu,alpha), (lam,lam), (lam,alpha), (alpha,alpha)),
kind 4 the quantile (y holds the probability p, hi the quantile). pos is the position in the
packed sym_tensor<3, K>: the component with i derivatives in mu, j in lambda, k in alpha
(i + j + k = K, sorted index (0^i, 1^j, 2^k)) is at (K - i)(K - i + 1)/2 + k. y, hi, lo, mu,
lam and alp are in hexadecimal; the row stores y itself so that the library and the
reference see the same double (z = y - mu is formed exactly in both).

Method, as in distrib_log_cdf.py: mpmath.diff of the DEFINITION at 300 and 400 bits, a row kept
when the two agree to 2^-80 relative. With a = lam alpha, c = lam (1 - alpha), s = sqrt(c),
x = a/s and z = y - mu the definitions are
    f(y)   = exp(-a |z| - c z^2/2) / Z,   Z = 2 int_0^inf exp(-a v - c v^2/2) dv
           = 2 Phi(-x) / (phi(x) s)        (mpmath.ncdf, mpmath.npdf)
    P(Z > D) = int_D^inf f = Phi(-(s D + x)) / (2 Phi(-x)),   D = |z|
    F = P(Z > D) for z < 0 and 1 - P(Z > D) for z > 0 (log1p for the near side),
    the information I_ab = E[score_a score_b] from the moments of |z|,
         E|z|^k = J_k / J_0,  a J_k + c J_(k+1) = k J_(k-1),  a J_0 + c J_1 = 1
    (integration by parts of int_0^inf z^k e^(-a z - c z^2/2) dz): the score of mu is
    a sign(z) + c z; the scores of (a, c) are -|z| - A_a and -z^2/2 - A_c, whose covariance is
    the covariance of (|z|, z^2/2), carried to (lambda, alpha) by the Jacobian of the rates.
Nothing of the library's route (log Mills ratio derivatives, the moment recursion in t = 1/x^2, the
chain rule through the rates, exp3 and log3) is used here. The closed forms of Z, of the tail
and of the moments are checked against mpmath.quad of the definitions at the start of main().
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_enet.csv"
AGREE = mpmath.mpf(2) ** -80
NEAR1 = 1.0 - 1e-12  # the doubles nearest the stated alphas, exactly as the test reads them
PARAMS = [  # (lambda, alpha)
    (1.7, 0.4), (0.3, 0.9), (5.0, 0.1), (1.0, 0.5),
    (1.0, 0.6180339887498949), (1.0, 0.8284271247461903),  # x = 1 and x = 2 at lambda = 1
    (2.0, 1e-12), (2.0, 1e-8), (2.0, 1e-4),
    (2.0, 1.0 - 1e-12), (2.0, 1.0 - 1e-8), (2.0, 1.0 - 1e-4),
    (1e-6, 0.5), (1e6, 0.5), (1e-6, 0.9), (1e6, 0.01), (1e6, 1.0 - 1e-5),
]
MUS = [0.4, -2.0]
ZS_CDF = [-300.0, -30.0, -6.0, -1.0, -0.1, -1e-3, 1e-3, 0.1, 1.0, 6.0, 30.0, 300.0]
ZS_PDF = [-6.0, -0.1, 0.1, 6.0]
PS = [1e-300, 1e-100, 1e-12, 1e-3, 0.3, 0.5, 0.7, 1.0 - 1e-3, 1.0 - 2.0 ** -40]


def mpf(v):
    return mpmath.mpf(v)


def rates(lam, alp):
    lam, alp = mpf(lam), mpf(alp)
    return lam * alp, lam * (1 - alp)


def log_z(lam, alp):
    a, c = rates(lam, alp)
    s = mpmath.sqrt(c)
    x = a / s
    return mpmath.log(2 * mpmath.ncdf(-x) / mpmath.npdf(x) / s)


def logpdf(y, mu, lam, alp):
    a, c = rates(lam, alp)
    z = y - mu
    return -a * abs(z) - c * z * z / 2 - log_z(lam, alp)


def log_tail(D, lam, alp):
    a, c = rates(lam, alp)
    s = mpmath.sqrt(c)
    x = a / s
    return mpmath.log(mpmath.ncdf(-(s * D + x)) / (2 * mpmath.ncdf(-x)))


def log_cdf(y, mu, lam, alp, upper):
    z = y - mu
    near = (z > 0) != bool(upper)  # log F on the right of mu, log(1 - F) on the left
    t = log_tail(abs(z), lam, alp)
    return mpmath.log1p(-mpmath.exp(t)) if near else t


def moments(lam, alp, n):
    a, c = rates(lam, alp)
    s = mpmath.sqrt(c)
    x = a / s
    j = [mpmath.ncdf(-x) / mpmath.npdf(x) / s]
    j.append((1 - a * j[0]) / c)
    for k in range(1, n):
        j.append((k * j[k - 1] - a * j[k]) / c)
    return [v / j[0] for v in j]  # E|z|^k


def info_entries(lam, alp):
    a, c = rates(lam, alp)
    m = moments(lam, alp, 4)
    imm = a * a + 2 * a * c * m[1] + c * c * m[2]
    v11 = m[2] - m[1] ** 2
    v12 = m[3] / 2 - m[1] * m[2] / 2
    v22 = m[4] / 4 - m[2] ** 2 / 4
    jac = [[alp, lam], [1 - alp, -lam]]  # rows (a, c), columns (lambda, alpha)
    v = [[v11, v12], [v12, v22]]
    blk = [[sum(jac[r][p] * v[r][q] * jac[q][s] for r in range(2) for q in range(2)) for s in range(2)] for p in range(2)]
    return imm, blk[0][0], blk[0][1], blk[1][1]


def pos3(K, i, k):
    return (K - i) * (K - i + 1) // 2 + k


def components(K):
    """(i, j, k, pos) for the sym_tensor<3, K>"""
    r = []
    for i in range(K + 1):
        for k in range(K - i + 1):
            r.append((i, K - i - k, k, pos3(K, i, k)))
    return r


_WEIGHTS = {}


def fd_weights(prec, npts=9, max_order=6):
    """weights[m][o] (o = -4..4, unit step) of the central finite-difference formula for the m-th
    derivative: sum_o w[m][o] o^q = m! delta(q, m) for q = 0..npts-1."""
    key = (prec, npts, max_order)
    if key not in _WEIGHTS:
        with mpmath.workprec(prec):
            half = npts // 2
            offs = list(range(-half, half + 1))
            vand = mpmath.matrix(npts, npts)
            for q in range(npts):
                for n, o in enumerate(offs):
                    vand[q, n] = mpf(o) ** q
            w = []
            for m in range(max_order + 1):
                rhs = mpmath.matrix(npts, 1)
                rhs[m] = mpmath.factorial(m)
                sol = mpmath.lu_solve(vand, rhs)
                w.append([sol[n] for n in range(npts)])
            _WEIGHTS[key] = w
    return _WEIGHTS[key]


def all_partials(f, base, scales, prec, hbits, max_total=6):
    """All partial derivatives of f(mu, lam, alp) of total order <= max_total at base, by tensor-product
    central differences on a 9 x 9 x 9 grid with steps scales[j] 2^-hbits (a variable the function does not
    depend on is given scale None and left out). Returns {(i, j, k): value}."""
    with mpmath.workprec(prec):
        w = fd_weights(prec)
        npts = 9
        half = npts // 2
        h = [None if sc is None else mpf(sc) * mpf(2) ** -hbits for sc in scales]
        b = [mpf(v) for v in base]
        ranges = [range(-half, half + 1) if hj is not None else range(0, 1) for hj in h]
        grid = {}
        for oa in ranges[0]:
            for ob in ranges[1]:
                for oc in ranges[2]:
                    x = [b[0] + (oa * h[0] if h[0] is not None else 0), b[1] + ob * h[1], b[2] + oc * h[2]]
                    grid[(oa, ob, oc)] = f(*x)
        out = {}
        # contract the alpha axis for every order k, then lambda, then mu
        step_a = {}
        for k in range(max_total + 1):
            for oa in ranges[0]:
                for ob in ranges[1]:
                    step_a[(k, oa, ob)] = mpmath.fsum(w[k][oc + half] * grid[(oa, ob, oc)] for oc in ranges[2])
        step_b = {}
        for k in range(max_total + 1):
            for j in range(max_total + 1 - k):
                for oa in ranges[0]:
                    step_b[(k, j, oa)] = mpmath.fsum(w[j][ob + half] * step_a[(k, oa, ob)] for ob in ranges[1])
        for k in range(max_total + 1):
            for j in range(max_total + 1 - k):
                for i in range(max_total + 1 - k - j):
                    if h[0] is None and i > 0:
                        continue
                    v = mpmath.fsum(w[i][oa + half] * step_b[(k, j, oa)] for oa in ranges[0]) if h[0] is not None else step_b[(k, j, 0)]
                    d = (h[0] ** i if h[0] is not None else 1) * h[1] ** j * h[2] ** k
                    out[(i, j, k)] = v / d
        return out


def param_scales(mu, lam, alp):
    """the step scales of (mu, lambda, alpha): the parameter itself, min(alpha, 1 - alpha) for alpha"""
    return [max(abs(mu), 1.0), lam, min(alp, 1.0 - alp)]


def two_runs(fun):
    """fun(prec, hbits) -> value; the two runs differ in the working precision and in the step"""
    a, c = fun(500, 40), fun(600, 48)
    if c == 0 or abs(a - c) > AGREE * abs(c):
        return None
    hi = float(c)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    return hi, float(c - mpf(hi))


def row(kind, K, p, pos, y, vals, mu, lam, alp):
    hi, lo = vals
    return f"{kind},{K},{p},{pos},{y.hex()},{hi.hex()},{lo.hex()},{mu.hex()},{lam.hex()},{alp.hex()}\n"


def scale_of(lam, alp):
    a, c = rates(lam, alp)
    return float(1 / (a + mpmath.sqrt(c)))


def tables(f, mu0, lam0, alp0, with_mu=True):
    """the partial derivatives of f at (mu0, lam0, alp0) from two runs that differ in precision and in step"""
    sc = param_scales(mu0, lam0, alp0)
    if not with_mu:
        sc[0] = None
    return (all_partials(f, (mu0, lam0, alp0), sc, 500, 40), all_partials(f, (mu0, lam0, alp0), sc, 600, 48))


def keep(t1, t2, key):
    a, c = t1[key], t2[key]
    if c == 0 or abs(a - c) > AGREE * abs(c):
        return None
    hi = float(c)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    return hi, float(c - mpf(hi))


def task(args):
    kind, mu0, lam0, alp0 = args
    lines, dropped = [], 0
    sc = scale_of(lam0, alp0)
    if kind in (0, 1, 2):
        zs = ZS_PDF if kind == 0 else ZS_CDF
        for zz in zs:
            y = mu0 + zz * sc
            if kind == 0:
                f = lambda m, l, a: logpdf(mpf(y), m, l, a)
            else:
                f = lambda m, l, a: log_cdf(mpf(y), m, l, a, kind == 2)
            t1, t2 = tables(f, mu0, lam0, alp0)
            for K in range(0, 7):
                for (i, j, k, pos) in ([(0, 0, 0, 0)] if K == 0 else components(K)):
                    r = keep(t1, t2, (i, j, k))
                    if r is None:
                        dropped += 1
                        continue
                    lines.append(row(kind, K, 0, pos, y, r, mu0, lam0, alp0))
    elif kind == 3:
        pairs = {0: 0, 3: 1, 4: 2, 5: 3}  # the nonzero pairs of sym_tensor<3, 2> and their entry in info_entries
        for p, idx in pairs.items():
            f = lambda m, l, a: info_entries(l, a)[idx]
            t1, t2 = tables(f, mu0, lam0, alp0, with_mu=False)
            for K in range(0, 5):
                for k in range(K + 1):
                    r = keep(t1, t2, (0, K - k, k))
                    if r is None:
                        dropped += 1
                        continue
                    lines.append(row(3, K, p, pos3(K, 0, k), 0.0, r, mu0, lam0, alp0))
    else:
        for pr in PS:
            def solve(prec, hbits):
                with mpmath.workprec(prec):
                    ptail = mpf(pr) if pr < 0.5 else 1 - mpf(pr)
                    target = mpmath.log(ptail)
                    g = lambda D: log_tail(D, lam0, alp0) - target
                    a, c = rates(mpf(lam0), mpf(alp0))
                    unit = 1 / (a + mpmath.sqrt(c))
                    guess = unit * (mpf(10) ** -3 if ptail > 0.4 else (1 - target))
                    D = mpmath.findroot(g, guess, tol=mpf(2) ** (-prec + 30), maxsteps=400)
                    return (mpf(mu0) - D) if pr < 0.5 else (mpf(mu0) + D)
            r = two_runs(solve)
            if r is None:
                dropped += 1
                continue
            lines.append(row(4, 0, 0, 0, pr, r, mu0, lam0, alp0))
    return lines, dropped


def check_definitions():
    """The closed forms against quadrature of the definitions at a few points."""
    with mpmath.workprec(300):
        for lam, alp in [(1.7, 0.4), (2.0, 1.0 - 1e-4), (2.0, 1e-4), (0.3, 0.9)]:
            a, c = rates(mpf(lam), mpf(alp))
            f0 = lambda v: mpmath.exp(-a * v - c * v * v / 2)
            z = 2 * mpmath.quad(f0, [0, 1, 10, mpmath.inf])
            assert abs(mpmath.exp(log_z(lam, alp)) / z - 1) < mpf(2) ** -150, "Z"
            for D in [0.3, 2.0]:
                t = mpmath.quad(f0, [D, D + 1, mpmath.inf]) / z
                assert abs(mpmath.exp(log_tail(mpf(D), lam, alp)) / t - 1) < mpf(2) ** -150, "tail"
            m = moments(lam, alp, 4)
            for k in range(5):
                q = 2 * mpmath.quad(lambda v: v ** k * f0(v), [0, 1, 10, mpmath.inf]) / z
                assert abs(m[k] / q - 1) < mpf(2) ** -150, "moment"
            imm, ill, ila, iaa = info_entries(lam, alp)
            qm = 2 * mpmath.quad(lambda v: (a + c * v) ** 2 * f0(v), [0, 1, 10, mpmath.inf]) / z
            assert abs(imm / qm - 1) < mpf(2) ** -150, "Imm"
            # I_lambda,lambda = E[(d l / d lambda)^2], d l / d lambda = -alp |z| - (1 - alp) z^2/2 - A_lambda
            al = mpmath.diff(lambda l: log_z(l, alp), mpf(lam))
            s1 = lambda v: (-alp * v - (1 - alp) * v * v / 2 - al) ** 2 * f0(v)
            qll = 2 * mpmath.quad(s1, [0, 1, 10, mpmath.inf]) / z
            assert abs(ill / qll - 1) < mpf(2) ** -90, "Ill"
    print("definitions checked against quadrature")


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    check_definitions()
    params = PARAMS[:3] if quick else PARAMS
    jobs = []
    for n, (lam, alp) in enumerate(params):
        mu = MUS[n % 2]
        for kind in (0, 1, 2, 3, 4):
            jobs.append((kind, mu, lam, alp))
    with Pool(2) as pool:
        results = pool.map(task, jobs, chunksize=1)
    n = dropped = 0
    path = OUT if not quick else OUT.with_name("distrib_enet_quick.csv")
    with open(path, "w", newline="\n") as f:
        f.write("kind,K,p,pos,y,hi,lo,mu,lam,alp\n")
        for lines, d in results:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{path.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
