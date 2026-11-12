"""Reference values of the upper_bounded scalar bijector (include/laplace/bijector/detail/scalar_shifted.hpp):
g(v) = -log(p1 - v) on (-inf, p1), increasing, and h(u) = p1 - e^(-u) on R, p1 the upper bound.

Usage (from the repository root):
    python -I tools/reference/bijector_upper_bounded.py

Writes tests/data/reference/bijector_upper_bounded.csv through bijector_param_common.py.
The distance p1 - h(u) = e^(-u) is written in the file as the distance to the upper bound
(which 4); the distance to the lower bound is +inf and has no rows.

Points: the bound in {0, +-2.5, +-1e8, +-1e-8} (7 values). Arguments v = p1 - d, with
d = |p1| 2^-k for k in {1, 2, 5, 10, 20, 35, 45, 60} (d = 2^-k when p1 = 0), so the
distance to the bound runs down to 2^-60 of its magnitude; a v equal to p1 after rounding
gives no row (g = +inf there). Also v = p1 - {0.5, 3, 1e3, 1e6}. Inverse points u are
spread over R, with |u| up to 300.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sympy as sp  # noqa: E402

from bijector_param_common import P1, X, write_references  # noqa: E402

BOUNDS = [0.0, 2.5, -2.5, 1e8, -1e8, 1e-8, -1e-8]
KS = [1, 2, 5, 10, 20, 35, 45, 60]
BELOW = [0.5, 3.0, 1e3, 1e6]
INVERSE = [-300.0, -40.0, -5.0, -1e-3, 0.0, 0.75, 5.0, 40.0, 300.0]


def points_v():
    out = []
    for b in BOUNDS:
        scale = abs(b) if b != 0.0 else 1.0
        for k in KS:
            out.append((float(b - scale * 2.0**-k), b, 0.0))
        for d in BELOW:
            out.append((float(b - d), b, 0.0))
    return out


def points_u():
    return [(u, b, 0.0) for b in BOUNDS for u in INVERSE]


def main():
    write_references(
        "upper_bounded",
        g=-sp.log(P1 - X),
        h=P1 - sp.exp(-X),
        nparams=1,
        decreasing=lambda p1, p2: False,
        points_v=points_v(),
        points_u=points_u(),
        distances=True,
        bounds=(-sp.oo, P1),
        workers=2,
    )


if __name__ == "__main__":
    main()
