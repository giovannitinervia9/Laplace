"""Backward error of a computed inverse, in ulps of p.

Usage: python -I tools/reference/backward.py <name> <results.bin>
    name: gamma_p_inv or ibeta_inv

The forward error of an inverse x(p) is the condition number times the error
in p: for a small shape a, x = (p Gamma(a + 1))^(1/a) moves by 1/a ulps when p
moves by one ulp, so a large forward error is no fault of the algorithm. The
backward error measures the algorithm: for the computed y it evaluates
p' = F(y) at 200 bits and reports |p' - p| / ulp(p), the distance in ulps
between the p that was asked for and the p that y actually solves. A value
near the forward error divided by the condition number is the best possible.
"""
import math
import struct
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"


def load(name, binpath):
    rows = []
    with (REF / f"{name}.csv").open() as fh:
        next(fh)
        for line in fh:
            parts = [float.fromhex(t) for t in line.strip().split(",")]
            rows.append(parts[:-2])
    data = Path(binpath).read_bytes()
    ys = struct.unpack(f"<{len(data) // 8}d", data)
    return rows, ys


def ulp(v):
    v = abs(v)
    if v < 2.2250738585072014e-308:
        return 2.0 ** -1074
    return 2.0 ** (math.frexp(v)[1] - 53)


def main(name, binpath):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import incomplete  # noqa: E402  (same directory; -I removed it from the path)
    rows, ys = load(name, binpath)
    worst, worst_args, errs = 0.0, None, []
    for args, y in zip(rows, ys):
        if not math.isfinite(y) or y <= 0:
            errs.append(math.inf)
            continue
        if name == "gamma_p_inv":
            a, p = args
            pp = incomplete.P(mpmath.mpf(a), mpmath.mpf(y))
        else:
            a, b, p = args
            if y >= 1:
                errs.append(math.inf)
                continue
            pp = incomplete.I(mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(y))
        e = float(abs(pp - p)) / ulp(p)
        errs.append(e)
        if e > worst:
            worst, worst_args = e, args
    finite = sorted(e for e in errs if math.isfinite(e))
    bad = len(errs) - len(finite)
    q = lambda f: finite[min(len(finite) - 1, int(f * len(finite)))]
    print(f"{name}: n {len(errs)}, backward error in ulps of p: median {q(0.5):.3g}, "
          f"99% {q(0.99):.3g}, max {worst:.3g} at {worst_args}, non-finite or out of range {bad}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
