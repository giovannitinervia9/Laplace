"""Shared machinery for the reference values of the scalar bijectors with
parameters (include/laplace/bijector/detail/scalar_<map>.hpp). Each map has a
small script that defines the map in SymPy and calls write_references().

File tests/data/reference/bijector_<map>.csv, header
    x,p1,p2,which,n0,n1,n2,hi,lo
all in hexadecimal doubles (as the other reference files):
    x        the point: v for which 0 and 2, u for which 1, 3, 4, 5
    p1, p2   the parameters, in the order of the map's partial derivatives
    which    0 forward g, 1 inverse h, 2 log|g'|, 3 log|h'|,
             4 the distance dom_hi - h(u), 5 the distance h(u) - dom_lo
    n0, n1, n2   the derivative d^(n0+n1+n2) / (d arg^n0 d p1^n1 d p2^n2);
             total orders 0..7 (order 7 is not tested: it sets the error scale
             of order 6); the distances only at order 0
    hi + lo  the exact value as a double-double

Method: SymPy differentiates the defining expressions symbolically, mpmath
evaluates them at two working precisions (base and base + 120 bits; the base
is 200 bits plus extra_bits(...) from the map's script), and a row is kept only
if the two agree to 2^-100 relative and the value is a nonzero finite normal
double. A map's script may pass `custom(which, n0, n1, n2, x, p1, p2)` to
evaluate some quantities differently (a limit at a removable singularity, for
example); it returns an mpmath number or None to use the default.

Usage from a map's script (run from the repository root with python -I):
    from bijector_param_common import X, P1, P2, write_references
    write_references("affine", g=(X - P1) / P2, h=P1 + P2 * X, nparams=2,
                     decreasing=lambda p1, p2: p2 < 0, points_v=[...], points_u=[...],
                     distances=False)
points_v, points_u: lists of (x, p1, p2) as Python floats (exact doubles).
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
X, P1, P2 = sp.symbols("x p1 p2", real=True)
KMAX = 7
TINY = 2.0**-1022

_state = {}


def multisets(k, nparams):
    """(n0, n1, n2) with n0 + n1 + n2 = k, n1 = 0 unless nparams >= 1, n2 = 0 unless nparams == 2"""
    out = []
    for n2 in range(0, k + 1 if nparams == 2 else 1):
        for n1 in range(0, k - n2 + 1 if nparams >= 1 else 1):
            out.append((k - n1 - n2, n1, n2))
    return out


def _build(g, h, nparams, decreasing_sign, distances, bounds):
    """lambdified functions keyed by (which, n0, n1, n2)"""
    exprs = {}
    for which, base in ((0, g), (1, h)):
        d1 = sp.diff(base, X)
        lj = sp.log(decreasing_sign * d1)
        for w, e in ((which, base), (which + 2, lj)):
            for k in range(KMAX + 1):
                for n0, n1, n2 in multisets(k, nparams):
                    d = e
                    if n0:
                        d = sp.diff(d, X, n0)
                    if n1:
                        d = sp.diff(d, P1, n1)
                    if n2:
                        d = sp.diff(d, P2, n2)
                    exprs[(w, n0, n1, n2)] = d
    if distances:
        lo, hi = bounds
        exprs[(4, 0, 0, 0)] = hi - h
        exprs[(5, 0, 0, 0)] = h - lo
    funcs = {}
    for key, e in exprs.items():
        if not e.free_symbols:
            continue  # constant: the map's test checks it exactly
        funcs[key] = sp.lambdify((X, P1, P2), e, modules="mpmath")
    return funcs


def _hexs(x):
    return float(x).hex()


def _dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mp.mpf(hi))


def _row(job):
    key, x, p1, p2 = job
    funcs = _state["funcs"]
    custom = _state["custom"]
    extra = _state["extra_bits"]
    base = 200 + extra(key[0], x, p1, p2) + (1200 if key[0] >= 4 else 0)
    vals = []
    for prec in (base, base + 120):
        with mp.workprec(prec):
            try:
                v = custom(*key, x, p1, p2) if custom else None
                if v is None:
                    v = funcs[key](mp.mpf(x), mp.mpf(p1), mp.mpf(p2))
                vals.append(mp.mpmathify(v))
            except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                return None
    v1, v2 = vals
    if mp.im(v1) != 0 or mp.im(v2) != 0:
        return None
    v1, v2 = mp.re(v1), mp.re(v2)
    if not (mp.isfinite(v1) and mp.isfinite(v2)) or v2 == 0:
        return None
    if abs(v1 - v2) > abs(v2) * mp.mpf(2) ** -100:
        return "disagree"
    with mp.workprec(base + 120):
        hi, lo = _dd(v2)
    if not math.isfinite(hi) or abs(hi) < TINY:
        return None
    w, n0, n1, n2 = key
    return f"{_hexs(x)},{_hexs(p1)},{_hexs(p2)},{_hexs(w)},{_hexs(n0)},{_hexs(n1)},{_hexs(n2)},{_hexs(hi)},{_hexs(lo)}"


def _init(state):
    # the lambdified functions cannot be pickled: each worker builds its own
    _state.update(state)
    _state["funcs"] = _build(*state["build_args"])


def _default_extra(which, x, p1, p2):
    b = 64
    for t in (x, p1, p2):
        at = abs(t)
        if at > 0:
            b += int(2 * abs(math.log2(at)))
        if at < 3000:
            b += int(3 * at)
    return b


def write_references(name, g, h, nparams, decreasing, points_v, points_u, distances=False, bounds=None,
                     extra_bits=None, custom=None, workers=3):
    """g, h: SymPy expressions in X, P1, P2 (h written in X as its argument u);
    decreasing(p1, p2) -> bool, the sign of g' for the log-Jacobians; bounds:
    (lo, hi) SymPy expressions of the domain of g, for the distances."""
    sign_cases = {}
    for (x, p1, p2) in points_v + points_u:
        sign_cases.setdefault(decreasing(p1, p2), []).append(None)
    rows, dropped, disagree, total = [], 0, 0, 0
    for dec in sorted(sign_cases):
        build_args = (g, h, nparams, -1 if dec else 1, distances, bounds)
        funcs = _build(*build_args)
        jobs = []
        for key in sorted(funcs):
            pts = points_v if key[0] in (0, 2) else points_u
            jobs += [(key, x, p1, p2) for (x, p1, p2) in pts if decreasing(p1, p2) == dec]
        total += len(jobs)
        state = {"build_args": build_args, "custom": custom, "extra_bits": extra_bits or _default_extra}
        with Pool(workers, initializer=_init, initargs=(state,)) as pool:
            res = pool.map(_row, jobs, chunksize=16)
        rows += [r for r in res if r and r != "disagree"]
        disagree += sum(1 for r in res if r == "disagree")
        dropped += sum(1 for r in res if r is None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"bijector_{name}.csv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("x,p1,p2,which,n0,n1,n2,hi,lo\n")
        fh.write("\n".join(rows) + "\n")
    print(f"{name}: {len(rows)} rows of {total}; {dropped} dropped (zero, not finite or subnormal), "
          f"{disagree} dropped (precisions disagree)", flush=True)
    return len(rows)


if __name__ == "__main__":
    sys.exit("import this module from a map's script")
