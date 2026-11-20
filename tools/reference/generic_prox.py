"""Reference values of the generic proximal operator in y of a continuous univariate law,
computed with mpmath.

    x*(v, t) = argmin_y  -t log f(y) + (y - v)^2 / 2,   t >= 0,

the problem solved by prox_y_by_newton (include/laplace/distrib/generic_prox.hpp;
note/penalties.md section 5h), on the laws and parameters of CASES below, which
tests/distrib/test_generic_prox.cpp mirrors in the same order.

Usage (from the repository root):  python -I tools/reference/generic_prox.py

File generic_prox.csv: case, v, t, then hi, lo of x* (all in hexadecimal).

Method. log f is written from the definition of each law in Laplace's
parametrization (constants dropped, they do not move the minimizer), never
from the library's derivatives. The objective's derivative
phi'(y) = (y - v) - t d/dy log f(y) comes from mpmath's numerical
differentiation (mp.diff) of that expression. Every case keeps t rho < 1,
rho being the law's weak-convexity constant, so phi is convex on the support
and phi' increases: a bracket around v is widened until phi' changes sign,
or closed at an end of the support, in which case the end is the minimizer
when phi' does not change sign before it; the bracket is then solved by
mpmath's Anderson-Bjorck root finder inside it, or bisected to the working precision when that
fails (where phi' grows exponentially, in the left tail of the Gumbel law).
Everything is computed at 200 and at 260 bits and a point is kept only if the
two agree to 2^-120 relative.

Points: per case, v = c + s sign 10^U(-4, 4) and, on a half-line or an
interval, values of v outside the support; t = s^2 10^U(-6, 6), capped at
U(0, 0.999)/rho for laws with rho > 0; c and s are the law's location and
spread.
"""
import random
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261120
PRECS = (200, 260)
N_POINTS = 120
INF = float("inf")


def student(mu, s, nu):
    return lambda y: -(mp.mpf(nu) + 1) / 2 * mp.log(1 + ((y - mu) / s) ** 2 / nu)


def pseudohuber(mu, sigma, nu):
    cache = {}

    def lf(y):
        if mp.mp.prec not in cache:  # R(nu) = x K_2(x)/K_1(x), x = sqrt(nu), once per precision
            x = mp.sqrt(nu)
            cache[mp.mp.prec] = x * mp.besselk(2, x) / mp.besselk(1, x)
        R = cache[mp.mp.prec]
        z = (y - mu) / sigma
        return -mp.sqrt(nu + R * z * z)
    return lf


def skewnormal(mu, sigma, gamma):
    def lf(y):
        b = mp.sqrt(2 / mp.pi)
        cc = mp.sign(gamma) * (2 * abs(mp.mpf(gamma)) / (4 - mp.pi)) ** (mp.mpf(1) / 3)
        omq = 1 - (mp.pi / 2 - 1) * cc * cc
        alpha = cc / (b * mp.sqrt(omq))
        s = mp.sqrt(1 + cc * cc)
        t = ((y - mu) / sigma + cc) / s
        return -t * t / 2 + mp.log(mp.ncdf(alpha * t))
    return lf


# (name, log f up to a constant, lower, upper, rho, centre, spread); the C++ test builds
# the same laws in the same order
CASES = [
    ("student_t1(0.5, 2, 3)", student(0.5, 2, 3), -INF, INF, 4 / (8 * 3 * 4), 0.5, 2),
    ("student_t2(-1, 1.5, 5)", student(-1, 1.5 * (3 / 5) ** 0.5, 5), -INF, INF, 6 / (8 * 3 * 2.25), -1, 1.5),
    ("cauchy(0, 0.7)", student(0, 0.7, 1), -INF, INF, 0.25 / 0.49, 0, 0.7),
    ("pseudohuber(0.3, 1.2, 1)", pseudohuber(0.3, 1.2, 1), -INF, INF, 0, 0.3, 1.2),
    ("pseudohuber(0, 1, 0.01)", pseudohuber(0, 1, 0.01), -INF, INF, 0, 0, 1),
    ("logistic(1, 0.5)", lambda y: -(y - 1) / mp.mpf(0.5) - 2 * mp.log1p(mp.exp(-(y - 1) / mp.mpf(0.5))), -INF, INF, 0, 1, 0.5),
    ("gumbel(-0.5, 2)", lambda y: -(y + mp.mpf(0.5)) / 2 - mp.exp(-(y + mp.mpf(0.5)) / 2), -INF, INF, 0, -0.5, 2),
    ("skewnormal(0, 1, 0.5)", skewnormal(0, 1, 0.5), -INF, INF, 0, 0, 1),
    ("skewnormal(2, 0.5, -0.95)", skewnormal(2, 0.5, -0.95), -INF, INF, 0, 2, 0.5),
    ("gaussian1(0.2, 3)", lambda y: -((y - mp.mpf(0.2)) / 3) ** 2 / 2, -INF, INF, 0, 0.2, 3),
    # gamma1(mu, phi): shape 1/phi, scale mu phi
    ("gamma1(2, 1/3)", lambda y: 2 * mp.log(y) - y / (2 * mp.mpf(1) / 3), 0, INF, 0, 2, 1.2),
    ("gamma1(1.5, 0.8)", lambda y: mp.mpf(0.25) * mp.log(y) - y / (mp.mpf(1.5) * mp.mpf(0.8)), 0, INF, 0, 1.5, 1.3),
    # weibull(m, k): scale b = m / Gamma(1 + 1/k)
    ("weibull(2, 2)", lambda y: mp.log(y) - (y / (2 / mp.gamma(mp.mpf(1.5)))) ** 2, 0, INF, 0, 2, 1),
    ("weibull(1, 1.3)", lambda y: mp.mpf(0.3) * mp.log(y) - (y / (1 / mp.gamma(1 + 1 / mp.mpf(1.3)))) ** mp.mpf(1.3), 0, INF, 0, 1, 0.8),
    ("lognormal1(0.2, 0.5)", lambda y: -mp.log(y) - (mp.log(y) - mp.mpf(0.2)) ** 2 / 1, 0, INF,
     float(mp.exp(2 * 0.5 - 2 * 0.2 - 3) / (2 * 0.5)), 1.5, 1),
    # invgauss(mu, phi): lambda = 1/phi
    ("invgauss(1, 0.5)", lambda y: -mp.mpf(1.5) * mp.log(y) - 2 * (y - 1) ** 2 / (2 * y), 0, INF, 0.125, 1, 0.7),
    # beta(mu, phi): a = mu phi, b = (1 - mu) phi
    ("beta(0.3, 10)", lambda y: 2 * mp.log(y) + 6 * mp.log(1 - y), 0, 1, 0, 0.3, 0.15),
    ("truncated student_t1(0, 1, 3) on [-1, 2]", student(0, 1, 3), -1, 2, 4 / 24, 0, 1),
    ("truncated gaussian1(0, 1) on [0.5, 3]", lambda y: -y * y / 2, 0.5, 3, 0, 1, 1),
    ("exponential(2)", lambda y: -y / 2, 0, INF, 0, 2, 2),
]


def solve(lf, lo_e, hi_e, v, t):
    """argmin of -t lf + (y - v)^2/2 on [lo_e, hi_e] at the current precision (phi convex)."""
    v, t = mp.mpf(v), mp.mpf(t)
    if t == 0:
        return min(max(v, lo_e), hi_e)

    def dphi(y):
        return (y - v) - t * mp.diff(lf, y)

    lo_f, hi_f = lo_e > -INF, hi_e < INF
    lo_e, hi_e = mp.mpf(lo_e), mp.mpf(hi_e)
    span = (hi_e - lo_e) if lo_f and hi_f else mp.mpf(1)
    tiny = mp.mpf(2) ** (-mp.mp.prec + 8) * max(span, abs(lo_e) if lo_f else 1, abs(hi_e) if hi_f else 1)
    in_lo, in_hi = (lo_e + tiny if lo_f else -mp.inf), (hi_e - tiny if hi_f else mp.inf)
    x = min(max(v, in_lo), in_hi)
    d = dphi(x)
    if d == 0:
        return x
    step = max(abs(x), mp.mpf(1))
    a = b = x
    if d < 0:  # the root is above x
        while True:
            b = x + step
            if b >= in_hi:
                if dphi(in_hi) <= 0:
                    return hi_e
                b = in_hi
                break
            if dphi(b) > 0:
                break
            x, step = b, step * 2
        a = x
    else:
        while True:
            a = x - step
            if a <= in_lo:
                if dphi(in_lo) >= 0:
                    return lo_e
                a = in_lo
                break
            if dphi(a) < 0:
                break
            x, step = a, step * 2
        b = x
    try:
        x = mp.findroot(dphi, (a, b), solver="anderson", tol=mp.mpf(2) ** (-2 * mp.mp.prec + 40))
        if a <= x <= b:
            return x
    except ValueError:
        pass
    for _ in range(mp.mp.prec + 60):  # phi' huge in a tail (Gumbel): bisection, which reads only its sign
        m = (a + b) / 2
        if dphi(m) > 0:
            b = m
        else:
            a = m
    return (a + b) / 2


def checked(lf, lo_e, hi_e, v, t):
    vals = []
    for prec in PRECS:
        with mp.workprec(prec):
            vals.append(solve(lf, lo_e, hi_e, v, t))
    x0, x1 = vals
    with mp.workprec(PRECS[1]):
        if abs(x0 - x1) > max(abs(x1), mp.mpf(2) ** -1000) * mp.mpf(2) ** -120:
            return None
    return x1


def hi_lo(x):
    with mp.workprec(PRECS[1]):
        hi = float(x)
        lo = float(x - hi)
    return hi, lo


def main():
    rng = random.Random(SEED)
    rows = []
    for k, (name, lf, lo_e, hi_e, rho, c, s) in enumerate(CASES):
        n0 = len(rows)
        for i in range(N_POINTS):
            v = c + s * rng.choice((-1.0, 1.0)) * 10 ** rng.uniform(-4, 4)
            if i % 10 == 9 and lo_e > -INF:
                v = lo_e - s * 10 ** rng.uniform(-3, 1)  # below the support
            t = s * s * 10 ** rng.uniform(-6, 6)
            if rho > 0:
                t = min(t, rng.uniform(0.0, 0.999) / rho)
            x = checked(lf, lo_e, hi_e, v, t)
            if x is None:
                print(f"  {name}: v = {v!r}, t = {t!r} dropped (precisions disagree)")
                continue
            hi, lo = hi_lo(x)
            rows.append((k, v, t, hi, lo))
        print(f"{name}: {len(rows) - n0} points", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "generic_prox.csv", "w", newline="\n") as f:
        f.write("case,v,t,hi,lo\n")
        for r in rows:
            f.write(",".join(float(x).hex() for x in r) + "\n")
    print(f"generic_prox.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
