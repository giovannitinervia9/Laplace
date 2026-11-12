"""Reference values of the bijector softplus, h(u) = log(1 + e^(a u))/a and
g(v) = log(e^(a v) - 1)/a (include/laplace/bijector/detail/scalar_softplus.hpp),
through bijector_param_common, with the sharpness a as the parameter p1.

Usage (from the repository root):  python -I tools/reference/bijector_softplus.py

Writes tests/data/reference/bijector_softplus.csv with nparams = 1: the pure
derivatives and the mixed ones d^n0/dx^n0 d^n1/da^n1 up to total order 6 (order
7 is written as the scale of the error, see the test). The sharpness takes the
values 1e-3, 0.1, 1, 7 and 1e3. The argument runs over the
magnitudes x = a v (forward) and x = a u (inverse), log-uniform from 2^-60 to
about 2^9.5 = 724 and both signs for the inverse, where the results are normal
doubles.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp  # noqa: E402

from bijector_param_common import X, P1, P2, write_references  # noqa: E402

SEED = 20261112
SHARPNESS = [1e-3, 0.1, 1.0, 7.0, 1e3]


def grid(rng):
    """argument magnitudes x > 0, log-uniform from 2^-60 to 2^9.5 (half-octave steps, random mantissa)"""
    out = []
    e = -60.0
    while e <= 9.5:
        out.append((1.0 + rng.random()) * 2.0 ** e)
        e += 0.5
    return out


def extra_bits(which, x, p1, p2):
    """working bits beyond the default: log(1 + e^z) with |z| = |a x| large needs about |z| / log 2
    bits to keep 1 + e^z distinct from 1, so the precision follows the product a x (the
    default follows x and a only in magnitude, and the two precisions then agree on a wrong value)"""
    z = abs(p1 * x)
    bits = 64 + int(1.5 * z)
    for t in (x, p1, p2):  # the default of bijector_param_common.write_references, kept
        at = abs(t)
        if at > 0:
            bits += int(2 * abs(math.log2(at)))
        if at < 3000:
            bits += int(3 * at)
    return bits


if __name__ == "__main__":
    rng = random.Random(SEED)
    xs = grid(rng)
    # the argument of the forward map is v = x / a > 0, of the inverse u = +-x / a
    points_v = [(x / a, a, 0.0) for a in SHARPNESS for x in xs]
    points_u = [(s * x / a, a, 0.0) for a in SHARPNESS for x in xs for s in (1.0, -1.0)]
    write_references(
        "softplus",
        g=sp.log(sp.exp(P1 * X) - 1) / P1,
        h=sp.log(1 + sp.exp(P1 * X)) / P1,
        nparams=1,
        decreasing=lambda p1, p2: False,
        points_v=points_v,
        points_u=points_u,
        distances=False,
        extra_bits=extra_bits,
        workers=2,
    )
