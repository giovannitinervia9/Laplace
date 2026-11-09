"""Backward error of computed inverse incomplete gamma and beta functions.

Usage: python -I tools/reference/backward_inverse.py <results dir> <name>[:<impl>] ...
    name: a reference file of incomplete.py or inverse.py (gamma_p_inv,
          gamma_q_inv, gamma_p_inv_tail, ..., ibeta_inv_logp, ibetac_inv_logq);
    impl: own (default) or boost; the results are <results dir>/<name>_<impl>.bin,
          as written by benchmarks/math_inverse.

For each computed y the forward function is evaluated at 200 bits (formulas of
inverse.py) and two numbers are reported:
    back    |F(y) - s| / ulp(s), s the input (p or q): how far, in ulps of
            the input, the probability that y actually solves lies from the
            one asked for; for the _logp and _logq files
            |log F(y) - l| / ulp(l), in ulps of the input l = log p (a
            relative error of p of |l| 2^-52 per ulp of l);
    backC   back / max(1, c), c = f(y) ulp(y) / ulp(s) the number of ulps of
            s that one ulp of y is worth (f the density; for the log inputs
            c = f(y) ulp(y) / (F(y) ulp(l))). Rounding y to a double alone
            leaves back up to c / 2, so backC near 1 is the best possible.
Rows whose reference value is 0 or 1 and whose result equals it count as 0.
Prints the median, 99th percentile and maximum of both, with the worst
arguments, and the number of non-finite or out-of-range results.
"""
import math
import struct
import sys
from pathlib import Path

import mpmath

mpmath.mp.prec = 200
REF = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"


def load(name, binpath):
    rows, refs = [], []
    with (REF / f"{name}.csv").open() as fh:
        next(fh)
        for line in fh:
            parts = [float.fromhex(t) for t in line.strip().split(",")]
            rows.append(parts[:-2])
            refs.append(parts[-2])
    data = Path(binpath).read_bytes()
    ys = struct.unpack(f"<{len(data) // 8}d", data)
    return rows, refs, ys


def ulp(v):
    v = abs(v)
    if v < 2.2250738585072014e-308:
        return 2.0 ** -1074
    return 2.0 ** (math.frexp(v)[1] - 53)


def forward(name, args, y):
    """(F(y), f(y)) at 200 bits, F the side the input refers to."""
    import inverse  # noqa: E402  (same directory)
    Y = mpmath.mpf(y)
    if name.startswith("gamma"):
        A = mpmath.mpf(args[0])
        lower = name.startswith("gamma_p")
        F = inverse.P(A, Y) if lower else inverse.Q(A, Y)
        f = mpmath.exp((A - 1) * mpmath.log(Y) - Y - mpmath.loggamma(A))
        return F, f
    A, B = mpmath.mpf(args[0]), mpmath.mpf(args[1])
    lower = name.startswith("ibeta_")
    X1 = 1 - Y  # exact at 200 bits
    F = inverse.I(A, B, Y, X1) if lower else inverse.Ic(A, B, Y, X1)
    f = mpmath.exp((A - 1) * mpmath.log(Y) + (B - 1) * mpmath.log(X1) - inverse._lbeta(A, B))
    return F, f


def main(resdir, specs):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for spec in specs:
        name, _, impl = spec.partition(":")
        impl = impl or "own"
        rows, refs, ys = load(name, Path(resdir) / f"{name}_{impl}.bin")
        logs = name.endswith("logp") or name.endswith("logq")
        beta = name.startswith("ibeta")
        back, backc = [], []
        worst, worst_c = (0.0, None), (0.0, None)
        bad = 0
        for args, ref, y in zip(rows, refs, ys):
            s = args[-1]
            if not math.isfinite(y) or y < 0 or (beta and y > 1):
                bad += 1
                continue
            if (y == 0.0 and ref == 0.0) or (beta and y == 1.0 and ref == 1.0):
                back.append(0.0)
                backc.append(0.0)
                continue
            if y == 0.0 or (beta and y == 1.0):
                bad += 1
                continue
            F, f = forward(name, args, y)
            if logs:
                e = float(abs(mpmath.log(F) - s)) / ulp(s)
                c = float(f / F) * ulp(y) / ulp(s)
            else:
                e = float(abs(F - s)) / ulp(s)
                c = float(f) * ulp(y) / ulp(s)
            ec = e / max(1.0, c)
            back.append(e)
            backc.append(ec)
            if e > worst[0]:
                worst = (e, (args, y))
            if ec > worst_c[0]:
                worst_c = (ec, (args, y))
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else math.nan
        fmt = lambda w: f"{w[0]:.3g} at {w[1][0]} -> {w[1][1]!r}" if w[1] else "0"
        print(f"{name:17s} {impl:5s} n {len(ys)}: back median {q(back, 0.5):.3g}, 99% {q(back, 0.99):.3g}, "
              f"max {fmt(worst)} | backC median {q(backc, 0.5):.3g}, 99% {q(backc, 0.99):.3g}, "
              f"max {fmt(worst_c)} | non-finite or out of range {bad}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
