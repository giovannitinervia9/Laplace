"""Reference values of the affine scalar bijector (include/laplace/bijector/detail/scalar_affine.hpp):
g(v) = (v - p1)/p2 and h(u) = p1 + p2 u, with p1 the location and p2 the scale (p2 != 0).

Usage (from the repository root):
    python -I tools/reference/bijector_affine.py

Writes tests/data/reference/bijector_affine.csv through bijector_param_common.py, which
defines the file format and the method (SymPy derivatives, mpmath at two precisions).
The derivatives of the affine map are simple enough to check by hand: g is linear in v
and p1 and has 1/p2 as its only nonlinearity in p2; the derivative of order n in p2 of
1/p2 is (-1)^n n! / p2^(n+1), which is the form the header writes out. The reference is
SymPy's own differentiation, so the header and this file agree only if both are right.

Points: the location in {0, +-1.5, +-1e6}, the scale in {+-1e-8, +-0.5, +-3, +-1e8}
(40 pairs). The argument v (forward) and the argument u (inverse) are taken over the same
list: a few values spread over the real line (large magnitudes included), and three
values next to the location, where g is zero and the cancellation in v - p1 is largest.
The domain is the whole real line, so there is no argument close to a bound.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bijector_param_common import P1, P2, X, write_references  # noqa: E402

LOCATIONS = [0.0, 1.5, -1.5, 1e6, -1e6]
SCALES = [1e-8, -1e-8, 0.5, -0.5, 3.0, -3.0, 1e8, -1e8]
SPREAD = [-1e6, -3.5, -1e-3, 0.0, 0.75, 4.25, 2e3, 1e6]
INVERSE = [-300.0, -20.0, -2.5, -1e-3, 0.0, 0.75, 6.0, 300.0]


def points(xs_of):
    """(x, p1, p2) for every parameter pair, with the arguments xs_of(location)"""
    out = []
    for loc in LOCATIONS:
        for sc in SCALES:
            for x in xs_of(loc):
                out.append((float(x), float(loc), float(sc)))
    return out


def near_location(loc):
    return SPREAD + [loc + 0.5, loc - 2.0, loc + 1e-3 * (1.0 + abs(loc))]


def inverse_points(loc):
    return INVERSE + [loc + 0.5, loc - 2.0, loc + 1e-3 * (1.0 + abs(loc))]


def main():
    write_references(
        "affine",
        g=(X - P1) / P2,
        h=P1 + P2 * X,
        nparams=2,
        decreasing=lambda p1, p2: p2 < 0,
        points_v=points(near_location),
        points_u=points(inverse_points),
        distances=False,
        bounds=None,
        workers=2,
    )


if __name__ == "__main__":
    main()
