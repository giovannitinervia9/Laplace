"""Reference values for the Box-Cox bijector (include/laplace/bijector/detail/scalar_box_cox.hpp),
written to tests/data/reference/bijector_box_cox.csv through bijector_param_common.

Usage (from the repository root):
    python -I tools/reference/bijector_box_cox.py

The map, with v > 0 the argument of the forward map g and u the argument of the inverse h,
lambda the exponent, L = log v:
    g(v) = (v^lambda - 1)/lambda             (log v at lambda = 0)
    h(u) = (1 + lambda u)^(1/lambda)         (exp(u) at lambda = 0)
    range of h = (0, inf); image of g = (-1/lambda, inf) for lambda > 0, (-inf, -1/lambda) for
    lambda < 0, R for lambda = 0.
The SymPy definitions above are differentiated symbolically and evaluated with mpmath, as for
every other map. They are singular at lambda = 0 (0/0), so for lambda = 0 the script passes
`custom`: the Taylor coefficients in lambda, computed with SymPy from the same definitions
(series of exp(S) for h, with S = log h = sum_n (-1)^n lambda^n u^(n+1)/(n+1); the series
sum_n lambda^n L^(n+1)/(n+1)! for g; log|g'| = (lambda - 1) L and log|h'| = (1 - lambda) S
are exact in lambda). At lambda = 0 the derivative d^n/d lambda^n of a function F is n! times
its lambda^n coefficient, and the derivatives in the argument are taken symbolically.

Points: the argument v log-uniform from 2^-60 to 2^60 and uniform around 1; the argument u
inside the image, uniform, at log-uniform distances from the finite bound -1/lambda, and at large
magnitudes. The exponents are the hard values of the task: 0, +-2^-40, +-2^-20, +-1e-3, +-0.3,
0.5, 1, 1.7, -1.5, 3.

The reference file has one row per quantity and derivative as described in
bijector_param_common.py; the distances to the bounds are not written (the range of h is
(0, inf), so the upper distance is +inf and the lower one is h(u); the test checks both exactly).
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp  # noqa: E402
import mpmath as mp  # noqa: E402

from bijector_param_common import X, P1, P2, write_references  # noqa: E402

SEED = 20261112
LAMBDAS = [0.0, -2.0**-40, 2.0**-40, -2.0**-20, 2.0**-20, -1e-3, 1e-3, -0.3, 0.3, 0.5, 1.0, 1.7, -1.5, 3.0]

# g and h as written in SymPy (X is v for g and u for h; P1 is lambda)
G = (X**P1 - 1) / P1
H = (1 + P1 * X) ** (1 / P1)


# ---------------------------------------------------------------- lambda = 0 (limits)

_series = {}


def _lambda_series():
    """h at lambda = 0: the coefficients of lambda^n of exp(S), n = 0..6 (functions of X = u)."""
    if "h" not in _series:
        # orders up to 7: the reference keeps order 7 as the scale of the error of order 6
        s = sum((-1) ** n * P1**n * X ** (n + 1) / (n + 1) for n in range(8))
        ser = sp.series(sp.exp(s), P1, 0, 8).removeO()
        _series["h"] = [sp.expand(ser.coeff(P1, n)) for n in range(8)]
    return _series["h"]


def _S_coeff(n):
    """lambda^n coefficient of S = log h at lambda = 0 (zero for n < 0)."""
    if n < 0:
        return sp.Integer(0)
    return (-1) ** n * X ** (n + 1) / (n + 1)


def limit_expr(which, n1):
    """d^n1/d lambda^n1 at lambda = 0 of the quantity `which` (before the derivatives in X)."""
    L = sp.log(X)
    if which == 0:  # g = sum_n lambda^n L^(n+1)/(n+1)!, so n1! times the coefficient is L^(n1+1)/(n1+1)
        return L ** (n1 + 1) / (n1 + 1)
    if which == 1:  # h
        return sp.factorial(n1) * _lambda_series()[n1]
    if which == 2:  # (lambda - 1) L
        return {0: -L, 1: L}.get(n1, sp.Integer(0))
    if which == 3:  # (1 - lambda) S
        return sp.factorial(n1) * (_S_coeff(n1) - _S_coeff(n1 - 1))
    raise ValueError(which)


_lim_cache = {}


def custom(which, n0, n1, n2, x, p1, p2):
    """The limit at lambda = 0 of the quantity (which, n0, n1); None elsewhere (the generic route)."""
    if p1 != 0:
        return None
    key = (which, n0, n1)
    if key not in _lim_cache:
        e = limit_expr(which, n1)
        if n0:
            e = sp.diff(e, X, n0)
        _lim_cache[key] = sp.lambdify(X, e, modules="mpmath")
    return _lim_cache[key](mp.mpf(x))


def decreasing(p1, p2):
    return False  # g' = v^(lambda - 1) > 0


def extra_bits(which, x, p1, p2):
    """Extra working bits. The cancellation of the lambda derivatives grows like |log2 lambda| per
    order; near v = 1 (log v small) the derivatives in lambda cancel to powers of log v, and near the
    bound of the image (1 + lambda u small) the logarithm of h does."""
    b = 64
    for t in (x, p1):
        at = abs(t)
        if at > 0:
            b += int(2 * abs(math.log2(at)))
        if at < 3000:
            b += int(3 * at)
    if p1 != 0 and abs(p1) < 1:
        b += int(8 * abs(math.log2(abs(p1))))
    with mp.workprec(400):
        if which in (0, 2):
            d = abs(mp.mpf(x) - 1)
        else:
            d = abs(1 + mp.mpf(p1) * mp.mpf(x))
        if 0 < d < 1:
            b += int(8 * -mp.log(d, 2))
    return b


# ---------------------------------------------------------------- points

def v_points(rng, lam):
    pts = [1.0, 0.5, 2.0, 0.1, 10.0, 2.0**-60, 2.0**60, 1 - 2.0**-30, 1 + 2.0**-30, 1 + 2.0**-10]
    for e in range(-60, 61, 6):
        pts.append(2.0**e * (1 + rng.random()))
    pts += [rng.uniform(0.5, 2.0) for _ in range(4)]
    return [(v, lam, 0.0) for v in pts]


def u_points(rng, lam):
    """u inside the image of g, which is the domain of h: (-1/lambda, inf) for lambda > 0,
    (-inf, -1/lambda) for lambda < 0, R for lambda = 0."""
    if lam == 0.0:
        pts = [0.0, 1.0, -1.0, 0.5, -0.5, 30.0, -30.0]
        pts += [s * 2.0**k for s in (1, -1) for k in range(-60, 10, 7)]
        pts += [rng.uniform(-40, 40) for _ in range(8)]
        return [(u, lam, 0.0) for u in pts]
    b = -1.0 / lam  # the finite bound of the image
    pts = []
    for d in (2.0**-60, 2.0**-40, 2.0**-20, 1e-6, 1e-3, 0.1, 1.0):
        pts.append(b + d if lam > 0 else b - d)
    lo = max(b, -30.0) if lam > 0 else -30.0
    hi = 30.0 if lam > 0 else min(b, 30.0)
    pts += [rng.uniform(lo, hi) for _ in range(10)]
    pts += [2.0**k for k in (3, 6, 9)] + [-(2.0**k) for k in (3, 6, 9)]
    out = []
    for u in pts:
        if lam > 0 and not (u > b):
            continue
        if lam < 0 and not (u < b):
            continue
        if not (1 + lam * u > 0):
            continue
        out.append((u, lam, 0.0))
    return out


def main():
    rng = random.Random(SEED)
    points_v, points_u = [], []
    for lam in LAMBDAS:
        points_v += v_points(rng, lam)
        points_u += u_points(rng, lam)
    print(f"points: {len(points_v)} for g, {len(points_u)} for h", flush=True)
    write_references("box_cox", g=G, h=H, nparams=1, decreasing=decreasing,
                     points_v=points_v, points_u=points_u, distances=False,
                     extra_bits=extra_bits, custom=custom, workers=2)


if __name__ == "__main__":
    main()
