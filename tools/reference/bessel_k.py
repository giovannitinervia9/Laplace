"""Reference values of log K_nu(x), the modified Bessel function of the second
kind of real order nu >= 0 at x > 0, and of its derivatives in x to order 7,
computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/bessel_k.py [--workers=N] [--check]

Files (format as in incomplete.py: the arguments nu, x, then hi, lo):
    log_bessel_k.csv          nu, x   log K_nu(x)
    log_bessel_k_dx{n}.csv    nu, x   d^n/dx^n log K_nu(x), n = 1..7

Method. From K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt (DLMF 10.32.9),
log K_nu(x) = log int w(t) dt with the positive weight
w(t) = exp(-x cosh t) cosh(nu t), and d^n/dx^n log K_nu(x) = (-1)^n kappa_n,
the n-th cumulant of C = cosh T under the density w / int w. The integrand is
centred at its peak t* = asinh(nu / x) (where x sinh t = nu, neglecting the
e^(-2 nu t) part of cosh(nu t)) and scaled by its value there; the moments of
C - cosh t* come from the same quadrature (a double-exponential rule on
[0, t*] and [t*, inf) whose step is halved until every moment, and the
integral of its absolute value, change by less than 2^-(prec - 40) of the
latter), and the cumulants from those moments. Every row is computed at two
precisions, P = 256 + 8 log2(2 + nu + x + 1/x) bits and P + 96, and kept only
if all eight values agree to 2^-110 relative (dropped rows are counted).

This route does not use mpmath.besselk, whose hypergeometric series fails for
orders near 1e6 and above, nor the Riccati identity that the library uses.
`--check` compares it, where besselk works, with log besselk and with the
derivatives obtained from K_(nu+1)/K_nu and the Riccati identity at high
precision.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261111
NMAX = 7
AGREE = mpmath.mpf(2) ** -110


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


# ---------------------------------------------------------------- quadrature

def de_rule(F, lo, hi, nf, prec):
    """int_lo^hi F(v) dv for a vector-valued F with nf components, by the
    double-exponential rule (tanh-sinh on a finite piece, exp-sinh on
    [lo, inf)). F returns 2 nf values: the components and their absolute
    values; the step is halved until every component changes by less than
    2^-(prec - 40) times the integral of its absolute value."""
    halfpi = mpmath.pi / 2
    half = hi == mpmath.inf
    if half:
        def node(s):
            e = mpmath.exp(halfpi * mpmath.sinh(s))
            return lo + e, halfpi * mpmath.cosh(s) * e
        smin, smax = -7, mpmath.asinh(mpmath.log(mpmath.mpf(10) ** 9) / halfpi)
    else:
        c, d = (lo + hi) / 2, (hi - lo) / 2

        def node(s):
            u = halfpi * mpmath.sinh(s)
            ch = mpmath.cosh(u)
            return c + d * mpmath.tanh(u), d * halfpi * mpmath.cosh(s) / (ch * ch)
        smin, smax = -7, 7
    tol = mpmath.mpf(2) ** -(prec - 40)
    total = [mpmath.mpf(0)] * (2 * nf)

    def add(s):
        v, w = node(s)
        if not (lo <= v <= hi) or not mpmath.isfinite(v):
            return
        f = F(v)
        for i in range(2 * nf):
            total[i] += w * f[i]

    h = mpmath.mpf(1) / 2
    k = 0
    prev = None
    while True:
        s = smin + (h if k > 0 else 0)
        step = 2 * h if k > 0 else h
        while s <= smax:
            add(s)
            s += step
        est = [t * h for t in total]
        if prev is not None:
            if all(abs(est[i] - prev[i]) <= tol * est[nf + i] for i in range(nf)):
                return est[:nf], est[nf:]
        prev = est
        if k >= 14:
            raise RuntimeError("quadrature did not converge")
        h /= 2
        k += 1


def log_k_and_derivs(nu, x, prec):
    """[log K_nu(x), d/dx log K, ..., d^7/dx^7 log K] at prec bits"""
    with mpmath.workprec(prec):
        nu, x = mpmath.mpf(nu), mpmath.mpf(x)
        ts = mpmath.asinh(nu / x)
        c0 = mpmath.cosh(ts)
        # phi(t) = -x cosh t + nu t, the log of the dominant part of w
        phi0 = -x * c0 + nu * ts

        def dc(t):  # cosh t - cosh t*, without cancellation
            return 2 * mpmath.sinh((t + ts) / 2) * mpmath.sinh((t - ts) / 2)

        def F(t):
            d = dc(t)
            # cosh(nu t) = e^(nu t) (1 + e^(-2 nu t)) / 2
            lw = -x * d + nu * (t - ts) + mpmath.log1p(mpmath.exp(-2 * nu * t)) - mpmath.log(2)
            w = mpmath.exp(lw)
            vals = [w]
            p = w
            for _ in range(NMAX):
                p = p * d
                vals.append(p)
            return vals + [abs(v) for v in vals]

        nf = NMAX + 1
        # the upper tail ends where the log of the weight has fallen by
        # (prec + 100) log 2 plus the growth of the moments: the exponent
        # -x (cosh t - c0) + nu (t - ts) is concave, so doubling L finds it
        drop = (prec + 100) * mpmath.log(2)
        # width of the peak, at most 1 (for small nu and x the weight is flat up
        # to t ~ log(2/x) and the doubling below finds the end)
        sigma = min(mpmath.mpf(1), (nu * nu + x * x) ** (-mpmath.mpf(1) / 4))

        def logw(t):  # log of the weight times the largest moment factor
            return -x * dc(t) + nu * (t - ts) + NMAX * mpmath.log(abs(dc(t)) + 1)

        L = sigma
        while logw(ts + L) > -drop:
            L *= 2
        parts = []
        if ts > 0:
            Llo = sigma
            while Llo < ts and logw(ts - Llo) > -drop:
                Llo *= 2
            lo = max(mpmath.mpf(0), ts - Llo)
            parts.append(de_rule(F, lo, ts, nf, prec))
        parts.append(de_rule(F, ts, ts + L, nf, prec))
        I = [sum(p[0][i] for p in parts) for i in range(nf)]
        m = [I[i] / I[0] for i in range(nf)]  # raw moments of D = C - c0
        # cumulants of D (those of C for n >= 2; kappa_1(C) = c0 + kappa_1(D))
        kap = [mpmath.mpf(0)] * (nf)
        for n in range(1, nf):
            s = m[n]
            for j in range(1, n):
                s -= mpmath.binomial(n - 1, j - 1) * kap[j] * m[n - j]
            kap[n] = s
        kap[1] += c0
        out = [phi0 + mpmath.log(I[0])]
        for n in range(1, nf):
            out.append((-1) ** n * kap[n])
        return out


def prec_for(nu, x):
    return 256 + 8 * int(math.log2(2 + nu + x + 1 / x))


def agree(u, v):
    return abs(u - v) <= AGREE * max(abs(u), abs(v), mpmath.mpf(2) ** -1070)


def row(args):
    nu, x = args
    p = prec_for(nu, x)
    try:
        u, v = log_k_and_derivs(nu, x, p), log_k_and_derivs(nu, x, p + 96)
    except RuntimeError:
        return None
    if all(agree(a, b) for a, b in zip(u, v)):
        return (nu, x, v)
    return None


# ---------------------------------------------------------------- the sample

def sample(rng, n):
    pts = []
    for _ in range(n):
        r = rng.random()
        if r < 0.25:    # small real orders, wide x
            nu = rng.choice([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, rng.uniform(0.0, 5.0)])
            x = log_uniform(rng, -30, 30)
        elif r < 0.5:   # half-integer orders y - 1/2 (Poisson-inverse Gaussian)
            y = int(log_uniform(rng, 0, 40))
            nu = y - 0.5
            x = log_uniform(rng, -20, 30)
        elif r < 0.75:  # large orders around the turning point x ~ nu (Debye)
            nu = log_uniform(rng, 3, 30)
            x = nu * log_uniform(rng, -8, 8)
        else:           # general
            nu = log_uniform(rng, -10, 20)
            x = log_uniform(rng, -20, 20)
        pts.append((nu, x))
    return pts


def check():
    """the integral route against mpmath.besselk and the Riccati identity"""
    rng = random.Random(5)
    worst = 0
    for nu, x in [(0.0, 1e-6), (0.5, 3.0), (2.0, 50.0), (1.0, 0.1), (3.3, 7.0), (20.5, 30.0), (4.0, 1e-3)] + \
            [(rng.uniform(0, 30), log_uniform(rng, -8, 7)) for _ in range(20)]:
        p = 300
        a = log_k_and_derivs(nu, x, p)
        with mpmath.workprec(p):
            n_, x_ = mpmath.mpf(nu), mpmath.mpf(x)
            lk = mpmath.log(mpmath.besselk(n_, x_))
            g = [n_ / x_ - mpmath.besselk(n_ + 1, x_) / mpmath.besselk(n_, x_)]
            # Riccati: g' = 1 + nu^2/x^2 - g/x - g^2, differentiated
            inv = [(-1) ** m * mpmath.factorial(m) / x_ ** (m + 1) for m in range(NMAX)]   # (1/x)^(m)
            inv2 = [(-1) ** m * mpmath.factorial(m + 1) / x_ ** (m + 2) for m in range(NMAX)]  # (1/x^2)^(m)
            for n in range(NMAX - 1):
                s = n_ ** 2 * inv2[n]
                for j in range(n + 1):
                    c = mpmath.binomial(n, j)
                    s -= c * g[j] * inv[n - j] + c * g[j] * g[n - j]
                g.append((1 if n == 0 else 0) + s)
            b = [lk] + g
            e = max(abs(u - v) / max(abs(v), mpmath.mpf(1e-300)) for u, v in zip(a, b))
            worst = max(worst, e)
            print(f"nu {nu:.4g} x {x:.4g}: max rel diff {mpmath.nstr(e, 3)}")
    print("worst", mpmath.nstr(worst, 3))


def coefs(K=16):
    """The Debye polynomials u_k(t) of the uniform expansion
    K_nu(nu z) ~ sqrt(pi / (2 nu)) e^(-nu eta) (1 + z^2)^(-1/4) sum_k (-1)^k u_k(t) / nu^k,
    t = (1 + z^2)^(-1/2) (DLMF 10.41.4), from the recurrence (DLMF 10.41.9)
    u_(k+1)(t) = t^2 (1 - t^2) u_k'(t) / 2 + (1/8) int_0^t (1 - 5 s^2) u_k(s) ds,
    as exact rationals rounded once to double. u_k(t) = t^k sum_(i=0..k) c_(k,i) t^(2i);
    the table holds c_(k,i) at index k (k + 1) / 2 + i."""
    from fractions import Fraction as Fr
    u = [[Fr(1)]]  # coefficients of t^0 .. t^(3k)
    for k in range(K):
        p = u[-1]
        new = [Fr(0)] * (len(p) + 3)
        for i in range(1, len(p)):  # t^2 (1 - t^2) (i p_i t^(i-1)) / 2
            new[i + 1] += i * p[i] / 2
            new[i + 3] -= i * p[i] / 2
        for i, c in enumerate(p):  # (1/8) int (1 - 5 s^2) c s^i
            new[i + 1] += c / 8 / (i + 1)
            new[i + 3] -= 5 * c / 8 / (i + 3)
        u.append(new[:3 * (k + 1) + 1])
    vals = []
    for k, p in enumerate(u):
        for i in range(k + 1):
            vals.append(p[k + 2 * i])
        assert all(p[e] == 0 for e in range(len(p)) if not (e >= k and (e - k) % 2 == 0)), k
    print(f"// {len(vals)} coefficients, u_0 .. u_{K}")
    for j in range(0, len(vals), 4):
        print("    " + ", ".join(float(v).hex() for v in vals[j:j + 4]) + ",")
    # B_k = max over t in [0, 1] and j <= 7 of |the j-th Taylor coefficient of
    # u_k at t| (on a grid of 4001 points); the terms 0..k-1 suffice for
    # nu >= nu_enough[k-1] = (2^60 B_k)^(1/k), where the first omitted term is
    # below 2^-60 for the value and the first seven derivatives in t
    B = []
    for k in range(1, K + 1):
        p = [float(c) for c in u[k]]
        best = 0.0
        for g in range(4001):
            t = g / 4000
            for j in range(8):
                s = sum(p[e] * math.comb(e, j) * t ** (e - j) for e in range(j, len(p)))
                best = max(best, abs(s))
        B.append(best)
    print("// nu_enough[k - 1], k = 1..%d" % K)
    print("    " + ", ".join(repr(float((2.0 ** 60 * b) ** (1.0 / k))) for k, b in enumerate(B, 1)))


def temme_band(rng, n):
    pts = []
    for _ in range(n):
        nu = rng.choice([0.0, 0.25, 0.5, 0.5, 1.0, 1.5, rng.uniform(0.0, 3.0)])
        x = rng.uniform(0.5, 1.0)
        pts.append((nu, x))
    return pts


def main():
    if sys.argv[1:2] == ["coefs"]:
        coefs()
        return
    workers = 4
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
    if "--check" in sys.argv:
        check()
        return
    n = 2400
    for a in sys.argv[1:]:
        if a.startswith("--n="):
            n = int(a.split("=")[1])
    pts = sample(random.Random(SEED), n)
    pts += [(0.0, 1.0), (1.0, 1.0), (0.5, 1.0), (2.0, 2.0), (20.5, 20.0), (1e6 - 0.5, 1e6), (0.0, 1e-300 ** 0.1)]
    # Temme's region near x = 1 (added 2026-11-11): its first coefficient f_0
    # is a difference of two terms that vanishes near x = 1-1.12 for every
    # reduced order, and below nu = 3/2 at most one recurrence step damps the
    # error of the ratio; the worst case is nu = 1/2 just below x = 1
    pts += temme_band(random.Random(SEED + 1), 200)
    with Pool(workers) as pool:
        out = pool.map(row, pts, chunksize=2)
    rows = sorted(r for r in out if r is not None)
    print(f"{len(pts) - len(rows)} rows dropped (precisions disagree or quadrature failed)")
    for n_ in range(NMAX + 1):
        name = "log_bessel_k" if n_ == 0 else f"log_bessel_k_dx{n_}"
        path = OUT / f"{name}.csv"
        with open(path, "w", newline="\n") as f:
            f.write("nu,x,hi,lo\n")
            for nu, x, v in rows:
                hi, lo = dd(v[n_])
                f.write(f"{float(nu).hex()},{float(x).hex()},{hi.hex()},{lo.hex()}\n")
        print(f"{path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
