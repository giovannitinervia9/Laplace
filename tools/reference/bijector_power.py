"""Reference values of the bijector power, g(v) = v^lambda, h(u) = u^(1/lambda)
(include/laplace/bijector/detail/scalar_power.hpp), through bijector_param_common.

Usage (from the repository root):  python -I tools/reference/bijector_power.py

Writes tests/data/reference/bijector_power.csv with nparams = 0: only the pure
derivatives in the argument, as there are no derivatives in lambda. The exponents cover
both signs, values near 0 and integers;
the argument runs over log-uniform magnitudes from 2^-1000 to 2^1000, where the
results are normal doubles (rows outside that range are dropped by the writer).
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp  # noqa: E402

from bijector_param_common import X, P1, P2, write_references  # noqa: E402

SEED = 20261112
LAMBDAS = [3.0, 1.7, 0.3, 1e-3, -0.4, -1.5, -7.0]


def magnitudes(rng, n):
    """positive doubles, log-uniform in magnitude from 2^-1000 to 2^1000"""
    out = []
    for i in range(n):
        e = -1000.0 + 2000.0 * (i + rng.random()) / n
        out.append((1.0 + rng.random()) * 2.0 ** math.floor(e))
    return out


if __name__ == "__main__":
    rng = random.Random(SEED)
    xs = magnitudes(rng, 260) + [0.5, 1.0, 2.0, 1.0 + 2.0**-52, 1.0 - 2.0**-53]
    points = [(x, lam, 0.0) for lam in LAMBDAS for x in xs]
    write_references(
        "power",
        g=X**P1,
        h=X ** (1 / P1),
        nparams=0,
        decreasing=lambda p1, p2: p1 < 0,
        points_v=points,
        points_u=points,
        distances=False,
        workers=2,
    )
