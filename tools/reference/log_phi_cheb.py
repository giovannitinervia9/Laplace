"""Piecewise Chebyshev expansions of g_k(t) = d^k log Phi(t) / dt^k, k = 1..6.

Usage (from the repository root):
    python -I tools/reference/log_phi_cheb.py [--lo -1] [--hi 5] [--width 1] [--update]

The expansions cover [lo, hi) in pieces of length width. On each piece, the
values of g_k at M Chebyshev nodes of the first kind (computed at 300 bits by
the recursion of tools/reference/normal.py, R^(n+1) = -(t R^(n) + n R^(n-1))
- sum_i C(n, i) R^(i) R^(n-i), with g_k = R^(k-1) and R = phi/Phi) give the
coefficients c_0, .., c_(M-1) of

    g_k(t) = c_0 + sum_(n >= 1) c_n T_n(x),   x = (2 (t - a) / width) - 1,

where a is the left end of the piece and T_n the Chebyshev polynomials. The
degree of each order is the smallest n such that the tail sum_(m > n) |c_m|
is below 2^-56 of max |g_k| on the piece; the degree of an order is the largest
of its pieces' degrees, and every piece keeps the coefficients up to it. The
coefficients are rounded to double and written as C++ hexadecimal literals.

The table layout consumed by include/laplace/math/normal.hpp: the orders in
turn; within an order, piece by piece (piece-major); within a piece, c_0 ..
c_deg. Clenshaw's recurrence evaluates the series there.

With --update, the block between the markers BEGIN and END in
include/laplace/math/normal.hpp is replaced by the new tables; without it the
block is printed. The script also reports, against 300-bit values at random
points, the error of the rounded table (absolute, in units of 2^-52) and the
truncation error alone, per order.
"""
import argparse
import math
import random
import sys
from pathlib import Path

import mpmath

ROOT = Path(__file__).resolve().parents[2]
NORMAL_HPP = ROOT / "include" / "laplace" / "math" / "normal.hpp"
BEGIN = "// BEGIN GENERATED: tools/reference/log_phi_cheb.py"
END = "// END GENERATED: tools/reference/log_phi_cheb.py"
M_NODES = 80           # Chebyshev nodes per piece; the coefficients beyond 60 are at the noise level
PREC = 300             # bits for the values and the coefficients
TOL_EXP = -56          # truncation below 2^TOL_EXP max |g_k| on the piece
SEED = 20261111


def derivs(t, n):
    """g_1(t) .. g_n(t) at the current precision."""
    T = mpmath.mpf(t)
    R = [mpmath.npdf(T) / mpmath.ncdf(T)]
    for m in range(n - 1):
        s = -(T * R[m] + (m * R[m - 1] if m >= 1 else 0))
        for i in range(m + 1):
            s -= mpmath.binomial(m, i) * R[i] * R[m - i]
        R.append(s)
    return R  # R[k-1] = g_k


def g_values(t, n):
    """g_1..g_n at t at PREC bits, checked against PREC + 80 bits."""
    with mpmath.workprec(PREC):
        a = derivs(t, n)
        a = [+x for x in a]
    with mpmath.workprec(PREC + 80):
        b = derivs(t, n)
    for x, y in zip(a, b):
        if abs(x - y) > mpmath.mpf(2) ** -(PREC - 20) * max(1, abs(y)):
            raise RuntimeError(f"precision check failed at t = {t!r}")
    return a


def chebyshev_coefficients(samples):
    """c_0 .. c_(M-1) of the interpolant at the M first-kind nodes, c_0 the mean."""
    M = len(samples)
    with mpmath.workprec(PREC):
        theta = [mpmath.pi * (j + mpmath.mpf(1) / 2) / M for j in range(M)]
        c = []
        for n in range(M):
            s = mpmath.fsum(samples[j] * mpmath.cos(n * theta[j]) for j in range(M))
            c.append(s / M if n == 0 else 2 * s / M)
    return c


def clenshaw_float(c, x):
    b1 = b2 = 0.0
    for cn in reversed(c[1:]):
        b1, b2 = 2.0 * x * b1 - b2 + float(cn), b1
    return float(c[0]) + x * b1 - b2


def fit(lo, width, pieces_per_order):
    """For each order k and piece p: the coefficients and the degree."""
    edges = [lo + p * width for p in range(pieces_per_order)]
    results = {k: [] for k in range(1, 7)}
    for p, a in enumerate(edges):
        b = a + width
        c_mid = (a + b) / 2
        half = (b - a) / 2
        with mpmath.workprec(PREC):
            xs = [mpmath.cos(mpmath.pi * (j + mpmath.mpf(1) / 2) / M_NODES) for j in range(M_NODES)]
            ts = [mpmath.mpf(c_mid) + half * x for x in xs]
        vals = [g_values(mpmath.nstr(t, 80), 6) for t in ts]
        for k in range(1, 7):
            samples = [v[k - 1] for v in vals]
            c = chebyshev_coefficients(samples)
            # max |g_k| on the piece, from the interpolant on a fine grid
            cf = [float(x) for x in c]
            gmax = max(abs(clenshaw_float(cf, math.cos(math.pi * i / 2000))) for i in range(2001))
            tol = 2.0 ** TOL_EXP * gmax
            tails = [0.0] * (M_NODES + 1)
            for m in range(M_NODES - 1, 0, -1):
                tails[m] = tails[m + 1] + abs(float(c[m]))
            deg = next((n for n in range(1, M_NODES) if tails[n + 1] <= tol), None)
            if deg is None:
                raise RuntimeError(f"no degree below {M_NODES} for k = {k} on [{a}, {b}]")
            results[k].append({"a": a, "b": b, "c": c, "gmax": gmax, "deg": deg,
                               "tail_tol": tol, "tail_at_deg": tails[deg + 1], "c_last": abs(float(c[-1]))})
    return results


def check(results, lo, width, degs):
    """Errors of the rounded table at random points, against 300-bit values."""
    rng = random.Random(SEED)
    for k in range(1, 7):
        worst_round = worst_trunc = 0.0
        worst_t = None
        for piece in results[k]:
            a, b = piece["a"], piece["b"]
            for _ in range(60):
                t = a + (b - a) * rng.random()
                x_exact = g_values(mpmath.nstr(mpmath.mpf(t), 80), k)[k - 1]
                with mpmath.workprec(PREC):
                    x = 2 * (mpmath.mpf(t) - a) / width - 1
                    c_full = piece["c"]
                    c_round = [mpmath.mpf(float(v)) for v in c_full[: degs[k] + 1]]
                    c_trunc = c_full[: degs[k] + 1]
                    s_round = clenshaw_mp(c_round, x)
                    s_trunc = clenshaw_mp(c_trunc, x)
                    err_round = abs(s_round - x_exact)
                    err_trunc = abs(s_trunc - x_exact)
                scale = max(1.0, abs(float(x_exact)))
                worst_round = max(worst_round, float(err_round) / scale)
                if float(err_trunc) / scale > worst_trunc:
                    worst_trunc = float(err_trunc) / scale
                    worst_t = t
        print(f"g{k}: degree {degs[k]}, table error {worst_round / 2.0 ** -52:.3g} ulp1 "
              f"(absolute / max(|g|, 1), units of 2^-52), truncation alone "
              f"{worst_trunc / 2.0 ** -52:.3g} ulp1 at t = {worst_t:.6g}", file=sys.stderr)


def clenshaw_mp(c, x):
    with mpmath.workprec(PREC):
        b1 = b2 = mpmath.mpf(0)
        for cn in reversed(c[1:]):
            b1, b2 = 2 * x * b1 - b2 + cn, b1
        return c[0] + x * b1 - b2


def hexlist(values, per_line=3):
    items = [float.hex(float(v)) for v in values]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("  " + ", ".join(items[i:i + per_line]))
    body = [ln + ", \\" for ln in lines[:-1]] + [lines[-1]]
    return "\n".join(body)


def render(results, lo, hi, width, degs):
    pieces = int(round((hi - lo) / width))
    offsets = {}
    total = 0
    data = []
    for k in range(1, 7):
        offsets[k] = total
        for piece in results[k]:
            for n in range(degs[k] + 1):
                data.append(piece["c"][n])
        total += pieces * (degs[k] + 1)
    assert len(data) == total
    deg_list = ", ".join(str(degs[k]) for k in range(1, 7))
    off_list = ", ".join(str(offsets[k]) for k in range(1, 7))
    out = []
    out.append(BEGIN)
    out.append("// d^k log Phi / dt^k on [cheb_lo, cheb_hi): piecewise Chebyshev expansions of width")
    out.append("// cheb_width, c_0 + sum_(n >= 1) c_n T_n(x), x = 2 (t - a) / width - 1 on the piece")
    out.append("// [a, a + width); generated by tools/reference/log_phi_cheb.py (do not edit).")
    out.append("namespace detail_normal {")
    out.append("")
    out.append(f"constexpr double cheb_lo = {float.hex(float(lo))};  // the expansions cover [cheb_lo, cheb_hi)")
    out.append(f"constexpr double cheb_hi = {float.hex(float(hi))};")
    out.append(f"constexpr double cheb_width = {float.hex(float(width))};  // pieces of this length")
    out.append(f"constexpr int cheb_pieces = {pieces};")
    out.append(f"constexpr int cheb_size = {total};")
    out.append("")
    out.append("// clang-format off")
    out.append("#define LAPLACE_LOG_PHI_CHEB \\")
    out.append(hexlist(data))
    out.append("// clang-format on")
    out.append("#if defined(__CUDACC__)")
    out.append("static __constant__ double log_phi_cheb_dev[cheb_size] = {LAPLACE_LOG_PHI_CHEB};")
    out.append("#endif")
    out.append("inline constexpr double log_phi_cheb_host[cheb_size] = {LAPLACE_LOG_PHI_CHEB};")
    out.append("// the coefficient i of the table")
    out.append("LAPLACE_HD_FORCEINLINE double cheb_coef(int i) {")
    out.append("  KOKKOS_IF_ON_DEVICE((return log_phi_cheb_dev[i];))")
    out.append("  KOKKOS_IF_ON_HOST((return log_phi_cheb_host[i];))")
    out.append("}")
    out.append("// the degree of the expansion of g_k, and the index of its first coefficient, k = 1..6")
    out.append("LAPLACE_HD_FORCEINLINE int cheb_degree(int k) {")
    out.append(f"  constexpr int d[6] = {{{deg_list}}};")
    out.append("  return d[k - 1];")
    out.append("}")
    out.append("LAPLACE_HD_FORCEINLINE int cheb_offset(int k) {")
    out.append(f"  constexpr int o[6] = {{{off_list}}};")
    out.append("  return o[k - 1];")
    out.append("}")
    out.append("")
    out.append("}  // namespace detail_normal")
    out.append(END)
    return "\n".join(out) + "\n"


def update(block):
    text = NORMAL_HPP.read_text(encoding="utf-8")
    if BEGIN not in text:
        raise RuntimeError("no BEGIN marker in normal.hpp")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    if tail.startswith("\n"):
        tail = tail[1:]  # the block ends with its own newline
    NORMAL_HPP.write_text(head + block + tail, encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=-1.0)
    ap.add_argument("--hi", type=float, default=4.0)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--update", action="store_true", help="write the block into normal.hpp")
    args = ap.parse_args()
    pieces = int(round((args.hi - args.lo) / args.width))
    results = fit(args.lo, args.width, pieces)
    degs = {}
    for k in range(1, 7):
        degs[k] = max(p["deg"] for p in results[k])
        for p in results[k]:
            print(f"k={k} [{p['a']:+.3f}, {p['b']:+.3f}) max|g|={p['gmax']:.6g} "
                  f"degree {p['deg']} tail {p['tail_at_deg']:.3g} <= {p['tail_tol']:.3g} "
                  f"|c_last|={p['c_last']:.3g}", file=sys.stderr)
    check(results, args.lo, args.width, degs)
    block = render(results, args.lo, args.hi, args.width, degs)
    if args.update:
        update(block)
        print(f"wrote {NORMAL_HPP}", file=sys.stderr)
    else:
        sys.stdout.write(block)


if __name__ == "__main__":
    main()
