"""Reference values of the functions of the normal block, computed with mpmath.

Usage (from the repository root):  python -I tools/reference/normal.py [name ...]

Format as in elementary.py: the argument, then hi, lo (hexadecimal), with
hi + lo the exact value as a double-double. Phi is the standard normal
distribution function and phi its density.

    erfcx              x       exp(x^2) erfc(x); overflows below x = -26.6
    Phi                x       Phi(x); underflows to 0 below x = -38.5
    log_Phi            x       log Phi(x), down to x = -2^500
    log_Phi_d1 .. d5   t       d^k log Phi(t) / dt^k, k = 1..5
    Phi_inv_logp       l       x with log Phi(x) = l, -l log-uniform in
                               [2^-60, 2^1000] (p = e^l beyond the double range
                               on one side, near 1 on the other)

The quantile from p (Phi_inv) uses qnorm.csv, written by incomplete.py.

The derivatives of log Phi come from the inverse Mills ratio R = phi/Phi, which
is the first derivative, and its differential equation R' = -R (t + R). By
Leibniz's rule its n-th derivative satisfies
    R^(n+1) = -(t R^(n) + n R^(n-1)) - sum_{i=0}^{n} C(n, i) R^(i) R^(n-i),
and d^k log Phi / dt^k = R^(k-1). For t far below zero the recursion cancels:
t + R is of size 1/|t| while t and R are of size |t|, and the k-th derivative is
of size |t|^-k against terms of size |t|^k. The working precision is raised by
12 bits per binary order of |t| above the 200 bits, and every value is
recomputed with 80 more bits; the two must agree to 2^-120 relative, or the
script stops. This is independent of statmodels7's formulas (pt_logphi.h),
which use the recursion above t = -1 and cumulants below.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261109
N = 10000


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def signed(rng, lo_exp, hi_exp):
    return math.copysign(log_uniform(rng, lo_exp, hi_exp), rng.random() - 0.5)


# ---------------------------------------------------------------- functions

def erfcx(x):
    x = mpmath.mpf(x)
    return mpmath.exp(x * x) * mpmath.erfc(x)


def Phi(x):
    return mpmath.ncdf(mpmath.mpf(x))


def log_Phi(x):
    x = mpmath.mpf(x)
    if x > 0:
        return mpmath.log1p(-mpmath.ncdf(-x))  # 1 - Phi(x) kept, not rounded away
    if x < -mpmath.mpf(10) ** 40:
        # mpmath's erfc fails for huge arguments; there the asymptotic series
        # Phi(x) = phi(x)/|x| sum_k (-1)^k (2k-1)!!/x^(2k) needs few terms
        y = 1 / (x * x)
        s, term = mpmath.mpf(1), mpmath.mpf(1)
        for k in range(1, 8):
            term *= -(2 * k - 1) * y
            s += term
        return -x * x / 2 - mpmath.log(-x) - mpmath.log(2 * mpmath.pi) / 2 + mpmath.log(s)
    return mpmath.log(mpmath.ncdf(x))


def inv_mills(x):
    """R = phi(x)/Phi(x) = d log Phi / dx."""
    x = mpmath.mpf(x)
    if x < -mpmath.mpf(10) ** 40:
        return mpmath.exp(-x * x / 2 - mpmath.log(2 * mpmath.pi) / 2 - log_Phi(x))
    return mpmath.npdf(x) / mpmath.ncdf(x)


def _logPhi_derivs(t, n, prec):
    with mpmath.workprec(prec):
        T = mpmath.mpf(t)
        R = [mpmath.npdf(T) / mpmath.ncdf(T)]
        for m in range(n - 1):
            s = -(T * R[m] + (m * R[m - 1] if m >= 1 else 0))
            for i in range(m + 1):
                s -= mpmath.binomial(m, i) * R[i] * R[m - i]
            R.append(s)
        return [+r for r in R]  # copies at this precision


def logPhi_deriv(t, k):
    """d^k log Phi / dt^k at t, checked at two precisions."""
    extra = 12 * max(0, math.ceil(math.log2(abs(t)))) if t < -1 else 0
    p1 = 200 + extra
    a = _logPhi_derivs(t, k, p1)[k - 1]
    b = _logPhi_derivs(t, k, p1 + 80)[k - 1]
    if b != 0 and abs(a - b) > mpmath.mpf(2) ** -120 * abs(b):
        raise RuntimeError(f"precision check failed at t = {t!r}, k = {k}")
    return b


def Phi_inv_logp(l):
    """x with log Phi(x) = l: Newton's method inside a bracket, bisection when
    a step leaves it (log Phi is increasing and concave)."""
    L = mpmath.mpf(l)
    lo = -mpmath.sqrt(2 * max(-L, 1)) - 2         # log Phi(lo) < l
    hi = mpmath.sqrt(2 * max(-mpmath.log(-L), 1)) + 2  # log Phi(hi) > l
    x = -mpmath.sqrt(-2 * L) if L < -1 else mpmath.mpf(0)
    for _ in range(2000):
        g = log_Phi(x) - L
        if g < 0:
            lo = x
        else:
            hi = x
        xn = x - g / inv_mills(x)
        if not (lo < xn < hi):
            xn = (lo + hi) / 2
        step = xn - x
        x = xn
        if abs(step) < mpmath.mpf(2) ** -180 * (1 + abs(x)):
            return x
    raise RuntimeError(f"no convergence at l = {l!r}")


# ---------------------------------------------------------------- argument sets

def rows_for(name, rng):
    rows = []
    if name == "erfcx":
        xs = [rng.uniform(-26.5, 30.0) for _ in range(N // 2)]
        xs += [signed(rng, -60, 4) for _ in range(N // 4)]
        xs = [x for x in xs if x > -26.5]
        xs += [log_uniform(rng, 4, 500) for _ in range(N // 4)]
        xs += [0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 26.0, 50.0, -1.0, -26.5]
        rows = [((x,), erfcx(x)) for x in xs]
    elif name == "Phi":
        xs = [rng.uniform(-38.5, 9.0) for _ in range(N // 2)]
        xs += [signed(rng, -60, 3) for _ in range(N // 2)]
        xs += [0.0, -1.0, -5.0, -37.0, -38.4, -38.5, 8.3, 9.0]
        rows = [((x,), Phi(x)) for x in xs]
    elif name == "log_Phi":
        xs = [rng.uniform(-40.0, 40.0) for _ in range(N // 2)]
        xs += [-log_uniform(rng, 4, 500) for _ in range(N // 4)]
        xs += [signed(rng, -60, 3) for _ in range(N // 4)]
        xs += [0.0, -1.0, 1.0, -5.0, -38.5, 5.0, 8.3, 9.0, 38.0]
        rows = [((x,), log_Phi(x)) for x in xs]
    elif name.startswith("log_Phi_d") and not name.endswith("_mid"):
        k = int(name[-1])
        ts = [rng.uniform(-40.0, 38.0) for _ in range(N // 4)]
        ts += [rng.uniform(-3.0, 1.0) for _ in range(N // 8)]  # around the switch near t = -1
        ts += [-log_uniform(rng, 4, 60) for _ in range(N // 8)]
        ts += [signed(rng, -40, 2) for _ in range(N // 8)]
        ts += [0.0, -1.0, 1.0, -2.0, -10.0, 10.0, 37.0]
        N_k = len(ts)
        rows = [((t,), logPhi_deriv(t, k)) for t in ts]
        print(f"{name}: {N_k} points", flush=True)
    elif name.startswith("log_Phi_d") and name.endswith("_mid"):
        # dense rows on [-1, 10] for the piecewise Chebyshev fits of
        # log_phi_cheb.py and for the bench's regional errors; written to
        # log_Phi_d{k}_mid.csv, the files above are unchanged
        k = int(name[len("log_Phi_d")])
        ts = [rng.uniform(-1.0, 10.0) for _ in range(16000)]
        ts += [rng.uniform(-1.0, -0.5) for _ in range(2000)]
        ts += [rng.uniform(4.0, 6.0) for _ in range(2000)]
        ts += [-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        rows = [((t,), logPhi_deriv(t, k)) for t in ts]
        print(f"{name}: {len(ts)} points", flush=True)
    elif name == "Phi_inv_logp":
        ls = [-log_uniform(rng, -60, 999) for _ in range(N // 2)]
        ls += [-log_uniform(rng, -10, 10) for _ in range(N // 4)]   # moderate p
        ls += [math.log(rng.random()) for _ in range(N // 4)]       # p uniform
        ls += [-math.log(2.0), -1.0, -1e-300, -745.2, -746.0, -1e300]
        rows = [((l,), Phi_inv_logp(l)) for l in ls if l < 0.0]
    else:
        raise ValueError(name)
    return rows


NAMES = ["erfcx", "Phi", "log_Phi"] + [f"log_Phi_d{k}" for k in range(1, 6)] + ["Phi_inv_logp"]
ARGNAMES = {"erfcx": "x", "Phi": "x", "log_Phi": "x", "Phi_inv_logp": "logp"}


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        rng = random.Random(f"{SEED}-{name}")
        rows = rows_for(name, rng)
        path = OUT / f"{name}.csv"
        with path.open("w", newline="\n") as fh:
            fh.write(f"{ARGNAMES.get(name, 't')},hi,lo\n")
            for args, v in rows:
                hi, lo = dd(v)
                fh.write(",".join(float.hex(float(a)) for a in args) + f",{float.hex(hi)},{float.hex(lo)}\n")
        print(f"{path}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or NAMES)
