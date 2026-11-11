"""Reference values of the incomplete gamma and beta functions, their
inverses, erfc and the normal quantile, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/incomplete.py [name ...]

Format as in elementary.py, with one to three argument columns before hi, lo:

    gamma_p, gamma_q        a, x      regularized lower / upper incomplete gamma
    gamma_p_inv             a, p      x such that P(a, x) = p
    ibeta, ibetac           a, b, x   regularized incomplete beta I_x(a, b), 1 - I_x(a, b)
    ibeta_inv               a, b, p   x such that I_x(a, b) = p
    erfc                    x
    qnorm                   p         standard normal quantile

The arguments reach far into both tails: half the x of the gamma functions are
a + k sqrt(a) with k up to 30 standard deviations, half are log-uniform; the
beta functions use the same idea around the mean a / (a + b). For the inverses
an argument x0 is drawn, p = P(a, x0) is rounded to a double, and the exact
inverse of that p is found by Newton steps at 200 bits from x0.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
N = 10000
SEED = 20261108


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def _P_series(a, x):
    # P(a, x) = x^a e^-x / Gamma(a + 1) 1F1(1; a + 1; x): all terms positive
    return mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a + 1)) *         mpmath.hyp1f1(1, a + 1, x, maxterms=10**7)


def _Q_upper(a, x):
    # Q(a, x) = e^-x x^a / Gamma(a) times Legendre's continued fraction, by the
    # modified Lentz method; for x > a it converges quickly
    tiny = mpmath.mpf(2) ** -1000
    eps = mpmath.mpf(2) ** -190
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10**7):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a)) * h


def P(a, x):
    if x <= a:
        return _P_series(a, x)
    try:
        return 1 - _Q_upper(a, x)
    except Exception:
        return _P_series(a, x)


def Q(a, x):
    if x > a:
        try:
            return _Q_upper(a, x)
        except Exception:
            pass
    return 1 - _P_series(a, x)


def _I_series(a, b, x):
    # I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) 2F1(a + b, 1; a + 1; x)
    lb = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    return mpmath.exp(a * mpmath.log(x) + b * mpmath.log1p(-x) - mpmath.log(a) - lb) *         mpmath.hyp2f1(a + b, 1, a + 1, x, maxterms=10**7)


def I(a, b, x):
    if x <= a / (a + b):
        return _I_series(a, b, x)
    return 1 - _I_series(b, a, 1 - x)


def Ic(a, b, x):
    """1 - I_x(a, b) = I_{1-x}(b, a), computed without the subtraction when it is small."""
    if x <= a / (a + b):
        return 1 - _I_series(a, b, x)
    return _I_series(b, a, 1 - x)


def gamma_args(rng):
    out = []
    for i in range(N):
        a = log_uniform(rng, -10, 14)
        if i % 2 == 0:
            x = a + rng.uniform(-30.0, 30.0) * math.sqrt(a)
            if x <= 0:
                x = log_uniform(rng, -30, 0) * a
        else:
            x = log_uniform(rng, -30, 15)
        out.append((a, x))
    return out


def beta_args(rng):
    out = []
    for i in range(N):
        a, b = log_uniform(rng, -8, 12), log_uniform(rng, -8, 12)
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        k = i % 4
        if k == 0:
            x = rng.random()
        elif k == 1:
            x = m + rng.uniform(-30.0, 30.0) * sd
        elif k == 2:
            x = log_uniform(rng, -60, -1)
        else:
            x = 1.0 - log_uniform(rng, -53, -1)
        if 0.0 < x < 1.0:
            out.append((a, b, x))
    return out


def newton_inverse(F, dF, target, x0, lo, hi):
    x = mpmath.mpf(x0)
    for _ in range(6):
        d = dF(x)
        if d == 0:
            break
        x = x - (F(x) - target) / d
        if not (lo < x < hi):
            return None
    return x


def normal_quantile(p):
    """x with Phi(x) = p, by Newton on log Phi in the lower half."""
    if p > 0.5:
        return -normal_quantile(1 - p)          # 1 - p is exact at 200 bits
    if p == 0.5:
        return mpmath.mpf(0)
    lp = mpmath.log(p)
    x = -mpmath.sqrt(-2 * lp)                   # start in the tail
    for _ in range(60):
        Phi = mpmath.ncdf(x)
        step = (mpmath.log(Phi) - lp) * Phi / mpmath.npdf(x)
        x -= step
        if abs(step) < mpmath.mpf(2) ** -180 * (1 + abs(x)):
            break
    return x


def rows_for(name, rng):
    rows = []
    if name in ("gamma_p", "gamma_q"):
        f = P if name == "gamma_p" else Q
        for a, x in gamma_args(rng):
            rows.append(((a, x), f(mpmath.mpf(a), mpmath.mpf(x))))
    elif name == "gamma_p_inv":
        for a, x0 in gamma_args(rng):
            A = mpmath.mpf(a)
            p = float(P(A, mpmath.mpf(x0)))
            if not (0.0 < p < 1.0):
                continue
            dens = lambda x: mpmath.exp((A - 1) * mpmath.log(x) - x - mpmath.loggamma(A))
            x = newton_inverse(lambda t: P(A, t), dens, mpmath.mpf(p), x0, 0, mpmath.inf)
            if x is not None:
                rows.append(((a, p), x))
    elif name in ("ibeta", "ibetac"):
        for a, b, x in beta_args(rng):
            A, B, X = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(x)
            rows.append(((a, b, x), I(A, B, X) if name == "ibeta" else Ic(A, B, X)))
    elif name == "ibeta_inv":
        for a, b, x0 in beta_args(rng)[: N // 2]:
            A, B = mpmath.mpf(a), mpmath.mpf(b)
            p = float(I(A, B, mpmath.mpf(x0)))
            if not (0.0 < p < 1.0):
                continue
            lbeta = mpmath.log(mpmath.beta(A, B))
            dens = lambda x: mpmath.exp((A - 1) * mpmath.log(x) + (B - 1) * mpmath.log(1 - x) - lbeta)
            x = newton_inverse(lambda t: I(A, B, t), dens, mpmath.mpf(p), x0, 0, 1)
            if x is not None:
                rows.append(((a, b, p), x))
    elif name == "erfc":
        xs = [rng.uniform(-6.0, 28.0) for _ in range(N // 2)]
        xs += [math.copysign(log_uniform(rng, -60, 4), rng.random() - 0.5) for _ in range(N // 2)]
        xs += [0.0, 26.5, 27.2, -6.0, 1e-300]
        for x in xs:
            rows.append(((x,), mpmath.erfc(mpmath.mpf(x))))
    elif name == "qnorm":
        ps = [log_uniform(rng, -1074, -1) for _ in range(N // 2)]
        ps += [1.0 - log_uniform(rng, -53, -2) for _ in range(N // 4)]
        ps += [rng.random() for _ in range(N // 4)]
        ps += [0.5, 0.25, 0.975, 5e-324]
        for p in ps:
            if 0.0 < p < 1.0:
                rows.append(((p,), normal_quantile(mpmath.mpf(p))))
    else:
        raise ValueError(name)
    return rows


ARGNAMES = {"gamma_p": "a,x", "gamma_q": "a,x", "gamma_p_inv": "a,p", "ibeta": "a,b,x",
            "ibetac": "a,b,x", "ibeta_inv": "a,b,p", "erfc": "x", "qnorm": "p"}


# Large shapes (gamma_*_large): a from 1e2 to 1e15, x = a (1 + d) with |d| log-uniform
# from 2^-40 to 1 and of both signs. The series of _P_series and the fraction of
# _Q_upper need O(sqrt(a)) terms there, and mpmath.gammainc gives up above a of
# about 1e8 (its hypergeometric series does not converge in the default number of
# terms). The reference is therefore the integral itself, by mpmath.quad at 200
# bits, split at the points m + k sqrt(a) / 2 (k = -40..40) around the mode
# m = a - 1 of the integrand t^(a-1) e^-t, whose width is sqrt(a). Its prefactor
# exp((a-1) log m - m - log Gamma(a)) is of size a log a and cancels, so it is
# formed at extra precision (about 2 log2(a) more bits). Checked against
# mpmath.gammainc for a <= 1e6 (agreement to 1e-54 relative; note/numerics.md).
LARGE_NAMES = ("gamma_p_large", "gamma_q_large", "log_gamma_p_large", "log_gamma_q_large")
LARGE_N = 2400


def large_args(rng):
    out = []
    for i in range(LARGE_N):
        a = 10.0 ** rng.uniform(2.0, 15.0)
        mag = 2.0 ** rng.uniform(-40.0, 0.0)  # log-uniform in |d|, from 2^-40 to 1
        d = mag if i % 2 == 0 else -mag
        x = a * (1.0 + d)
        out.append((a, x))
    return out


def _integrand_quad(a, x, lower_side):
    """Integral of t^(a-1) e^-t over [x, inf) (lower_side False) or over [0, x] (True),
    times 1 / Gamma(a), returned as an mpf with the prefactor at extra precision."""
    extra = 200 + 2 * int(math.log2(a)) + 40
    with mpmath.workprec(extra):
        A = mpmath.mpf(a)
        X = mpmath.mpf(x)
        m = A - 1
        lg = (A - 1) * mpmath.log(m) - m - mpmath.loggamma(A)
        sd = mpmath.sqrt(A)
        # the integrand divided by its largest value on the interval (at min(X, m)
        # on the lower side, max(X, m) on the upper side): mpmath.quad's error
        # control is absolute, and values like e^-288 defeat it (measured: the
        # unscaled version was off by a factor 0.58 at a = 1e12, x 24 sqrt(a)
        # below the mode, and agreed with itself to 6e-12 only across breakpoints)
        tmax = min(X, m) if lower_side else max(X, m)
        lh_max = (A - 1) * mpmath.log(tmax / m) - (tmax - m)
        h = lambda t: mpmath.exp((A - 1) * mpmath.log(t / m) - (t - m) - lh_max)
        # breakpoints at the local scale of the integrand at X, ell =
        # min(sqrt(a), 1 / |r|) with r = d/dt log h at X = (a - 1)/X - 1, in a
        # geometric progression away from X (the mass sits within a few ell of
        # X when X is far from the mode m, and within a few sqrt(a) of m
        # otherwise), and around the mode by sqrt(a)/2 when it lies inside
        r = (A - 1) / X - 1
        ell = sd if r == 0 else min(sd, 1 / abs(r))
        if lower_side:
            pts = [mpmath.mpf(0), X]
            pts += [X - ell * 2 ** k for k in range(0, 200) if X - ell * 2 ** k > 0]
            pts += [m + k * sd / 2 for k in range(-40, 41) if 0 < m + k * sd / 2 < X]
        else:
            pts = [X, mpmath.inf]
            pts += [X + ell * 2 ** k for k in range(0, 200)]
            pts += [m + k * sd / 2 for k in range(-40, 41) if m + k * sd / 2 > X]
        pts = sorted(set(pts))
        val = mpmath.quad(h, pts)
        v = mpmath.exp(lg + lh_max) * val
    return +v  # back to the working precision of the module (200 bits)


def P_large(a, x):
    return _integrand_quad(a, x, True)


def Q_large(a, x):
    return _integrand_quad(a, x, False)


def write_large(names):
    rng = random.Random(SEED)
    rows = []
    for a, x in large_args(rng):
        A = mpmath.mpf(a)
        X = mpmath.mpf(x)
        if x < a:
            p = P_large(A, X)
            q = 1 - p
        else:
            q = Q_large(A, X)
            p = 1 - q
        rows.append((a, x, p, q))
    for name in names:
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write("a,x,hi,lo\n")
            for a, x, p, q in rows:
                if name == "gamma_p_large":
                    v = p
                elif name == "gamma_q_large":
                    v = q
                elif name == "log_gamma_p_large":
                    v = mpmath.log(p) if x < a else mpmath.log1p(-q)
                else:
                    v = mpmath.log1p(-p) if x < a else mpmath.log(q)
                hi, lo = dd(v)
                fh.write(",".join(float.hex(float(t)) for t in (a, x)) + f",{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows)} rows", flush=True)


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    large = [n for n in names if n in LARGE_NAMES]
    if large:
        write_large(large)
    for name in names:
        if name in LARGE_NAMES:
            continue
        rng = random.Random(SEED)
        rows = rows_for(name, rng)
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write(f"{ARGNAMES[name]},hi,lo\n")
            for args, v in rows:
                hi, lo = dd(v)
                fh.write(",".join(float.hex(float(a)) for a in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(ARGNAMES))
