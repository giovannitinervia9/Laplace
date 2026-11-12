"""Reference values of the Yeo-Johnson bijector (include/laplace/bijector/detail/scalar_yeo_johnson.hpp)
with its exponent lambda as the first parameter, written to tests/data/reference/bijector_yeo_johnson.csv.

Usage (from the repository root):  python -I tools/reference/bijector_yeo_johnson.py

The map, in SymPy, from its definition (nothing from the header):
  g(v) = ((1 + v)^lambda - 1)/lambda                     v >= 0
  g(v) = -((1 - v)^(2 - lambda) - 1)/(2 - lambda)        v < 0
  h(u) = (1 + lambda u)^(1/lambda) - 1                   u >= 0
  h(u) = 1 - (1 - (2 - lambda) u)^(1/(2 - lambda))       u < 0
and the log-Jacobians log g', log h' of each branch. Each branch is differentiated
symbolically, then evaluated with mpmath (see bijector_param_common.py).

The branches are singular at lambda = 0 (the quotient) and at lambda = 2 (the
exponent 2 - lambda = 0). Their limits are taken by evaluating at lambda = 2^-300
and 2 - 2^-300 with extra working precision (330 bits per order in lambda, so
that the cancellation in the quotient is resolved): the error of that is of the
order of 2^-300 relative, far below the 2^-100 that the rows must agree to.
The custom function also sets the branch from the sign of the argument: v = 0 takes
v >= 0, and the derivatives of order 3 and more at v = 0 are not sampled (the
branches differ there: the kink of forward_kinks()).

Points: v and u log-spaced from 2^-60 to 2^60 on both sides, with 0 and some
moderate values; for lambda in {-0.5, 2.5}, also u next to the finite bound of
the image (1 - 2^-k of the bound, k up to 40). The lambda values are the hard
ones: 0, +-2^-30, 2 - 2^-30, 2, 2 + 2^-30, 1, 1.6, 2.5, -0.5, 0.4 and 1e-3.
u is inside the image of h: R for 0 <= lambda <= 2, u < -1/lambda below 0 and
u > 1/(2 - lambda) above 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402
import sympy as sp  # noqa: E402

import bijector_param_common as common  # noqa: E402
from bijector_param_common import X, P1, P2, write_references  # noqa: E402

G = {"pos": ((1 + X) ** P1 - 1) / P1, "neg": -((1 - X) ** (2 - P1) - 1) / (2 - P1)}
H = {"pos": (1 + P1 * X) ** (1 / P1) - 1, "neg": 1 - (1 - (2 - P1) * X) ** (1 / (2 - P1))}
BASE = {}
for b in ("pos", "neg"):
    BASE[(0, b)] = G[b]
    BASE[(1, b)] = H[b]
    BASE[(2, b)] = sp.log(sp.diff(G[b], X))  # g' > 0 on both branches
    BASE[(3, b)] = sp.log(sp.diff(H[b], X))  # h' > 0 on both branches

EPS_BITS = 300
LAMBDAS = [0.0, -2.0 ** -30, 2.0 ** -30, 1e-3, 0.4, 1.0, 1.6, 2.0 - 2.0 ** -30, 2.0, 2.0 + 2.0 ** -30, 2.5, -0.5]
EXPS = [-60, -30, -10, -3, -1, 0, 1, 3, 10, 30, 60]
MID = [0.1, 0.5, 0.9, 3.0, 10.0]

_FN = {}


def _fn(which, branch, n0, n1):
    """the lambdified derivative d^(n0 + n1) / (dx^n0 dp1^n1) of the branch (cached per process)"""
    key = (which, branch, n0, n1)
    if key not in _FN:
        e = BASE[(which, branch)]
        if n0:
            e = sp.diff(e, X, n0)
        if n1:
            e = sp.diff(e, P1, n1)
        _FN[key] = sp.lambdify((X, P1), e, modules="mpmath")
    return _FN[key]


def custom(which, n0, n1, n2, x, p1, p2):
    """every row: the branch of the argument, the limit at lambda = 0 and 2 (2^-300 offsets)"""
    if n0 + n1 + n2 >= 3 and x == 0.0:
        raise ValueError("v = 0 is not sampled at orders >= 3 (the two branches differ there)")
    extra = 0
    if p1 == 0.0:
        lam = mp.mpf(2) ** -EPS_BITS
        extra = 330 * (n1 + 1)
    elif p1 == 2.0:
        lam = 2 - mp.mpf(2) ** -EPS_BITS
        extra = 330 * (n1 + 1)
    else:
        lam = mp.mpf(p1)
    branch = "pos" if x >= 0.0 else "neg"
    with mp.workprec(mp.mp.prec + extra):
        v = _fn(which, branch, n0, n1)(mp.mpf(x), lam)
        # a quantity that vanishes at lambda = 0 or 2 comes out as the residual of the offset
        # (of the order of 2^-300): the row is dropped; the same values are checked at 2^-30 and 1e-3
        if extra and abs(v) < mp.mpf(2) ** -200:
            raise ValueError("zero at lambda = 0 or 2: the offset residual")
        return v


def extra_bits(which, x, p1, p2):
    return 600 + common._default_extra(which, x, p1, p2)


def v_points(lam):
    vals = {0.0}
    for e in EXPS:
        vals.add(2.0 ** e)
        vals.add(-(2.0 ** e))
    for m in MID:
        vals.add(m)
        vals.add(-m)
    return sorted(vals)


def u_points(lam):
    lo = 1.0 / (2.0 - lam) if lam > 2.0 else float("-inf")
    hi = -1.0 / lam if lam < 0.0 else float("inf")
    cand = {0.0}
    for e in [-60, -30, -10, -3, -1, 0, 1, 3, 9]:
        cand.add(2.0 ** e)
        cand.add(-(2.0 ** e))
    cand |= {0.1, -0.1, 0.5, -0.5, 0.9, -0.9, 2.0, -2.0, 3.0, -3.0}
    if lam in (-0.5, 2.5):
        bound = hi if lam < 0.0 else lo
        for k in (2, 5, 10, 20, 30, 40):
            cand.add(bound * (1.0 - 2.0 ** -k))
    return sorted(u for u in cand if lo < u < hi)


def main():
    points_v = [(v, lam, 0.0) for lam in LAMBDAS for v in v_points(lam)]
    points_u = [(u, lam, 0.0) for lam in LAMBDAS for u in u_points(lam)]
    write_references("yeo_johnson", g=G["pos"], h=H["pos"], nparams=1,
                     decreasing=lambda p1, p2: False, points_v=points_v, points_u=points_u,
                     distances=False, extra_bits=extra_bits, custom=custom, workers=2)


if __name__ == "__main__":
    main()
