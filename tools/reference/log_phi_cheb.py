"""Piecewise Chebyshev expansions of g_k(t) = d^k log Phi(t) / dt^k, k = 1..6.

Usage (from the repository root):
    python -I tools/reference/log_phi_cheb.py [--lo -1] [--hi 4] [--width 1] [--update]

The expansions cover [lo, hi). Each order has its own pieces, because the
accuracy of the route is limited by the ratio sum_n |c_n| / |g_k(t)|, the
condition number of Clenshaw's sum at the point, and that ratio is set by two
things which differ from order to order (measured on 2026-11-14,
docs/numerics.md, "The accuracy of the log Phi derivatives on [-1, 4)"):

  * the zeros of g_k inside the band, where |g_k| becomes arbitrarily small
    while the coefficients stay of the size of the largest |g_k| of the piece.
    Each zero z is made the midpoint of its piece and that piece carries the
    expansion of h(t) = g_k(t) / (t - z) instead of g_k, with z written as a
    pair of doubles so that the factor (t - z) is formed to full accuracy. The
    quotient h is analytic wherever g_k is, the nearest singularity being the
    complex zeros of Phi at about 3.08 +- 3.81i, so the expansion of h
    converges as fast as that of g_k.
  * the decay of g_k in the right tail, where g_k falls by an order of
    magnitude across a piece of width one, so that the coefficients of the
    piece are much larger than |g_k| at its right end. Above t = 1 the pieces
    are half as wide, which keeps the ratio near one.

The degree of a piece is the smallest n whose tail sum_(m > n) |c_m| falls
below 2^-56 of the SMALLEST |value| at the nodes of the piece, a relative
criterion; where no degree below the node count satisfies it the largest
|value| is used instead. The degree of an order is the largest of its pieces'
degrees, and every piece keeps its coefficients up to it.

The values of g_k at the M Chebyshev nodes of the first kind are computed at
300 bits by the recursion of tools/reference/normal.py,
R^(n+1) = -(t R^(n) + n R^(n-1)) - sum_i C(n, i) R^(i) R^(n-i), with
g_k = R^(k-1) and R = phi/Phi, and give the coefficients c_0, .., c_(M-1) of

    f(t) = c_0 + sum_(n >= 1) c_n T_n(x),   x = (2 (t - a) / (b - a)) - 1,

where f is g_k, or h on a piece with a zero, [a, b) is the piece and T_n the
Chebyshev polynomials. The coefficients are rounded to double and written as
C++ hexadecimal literals.

The piece of t is found in the header without a scan: 40 bins of width 1/8
cover [lo, hi), and the bin of t gives the piece whose left edge is at or
before the bin's left end, after which one comparison corrects the index. That
is enough as long as no piece is narrower than the bin, which the script
checks.

With --update, the block between the markers BEGIN and END in
include/laplace/math/normal.hpp is replaced by the new tables; without it the
block is printed. The script also reports, against 300-bit values at random
points, the error of each order relative to g_k itself and in ulp of the
largest |g_k| of the band.
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
TOL_EXP = -56          # truncation below 2^TOL_EXP of the smallest |value| of the piece
BINS = 40              # bins of the piece lookup, of width (hi - lo) / BINS
MIN_PIECE = 0.2        # a cut is not placed nearer than this to another cut or to a zero
SEED = 20261111
CONT = " \\"


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


def zeros_of(k, lo, hi):
    """the zeros of g_k in [lo, hi) at 300 bits: a scan for sign changes, then Newton"""
    with mpmath.workprec(PREC):
        n = int(round((hi - lo) / 0.02))
        grid = [mpmath.mpf(lo) + (mpmath.mpf(hi) - lo) * i / n for i in range(n + 1)]
        vals = [derivs(t, k)[k - 1] for t in grid]
        out = []
        for i in range(n):
            if vals[i] == 0 or vals[i] * vals[i + 1] >= 0:
                continue
            t = (grid[i] + grid[i + 1]) / 2
            for _ in range(80):  # Newton on g_k with g_(k+1)
                d = derivs(t, k + 1)
                step = d[k - 1] / d[k]
                t -= step
                if abs(step) < mpmath.mpf(2) ** -280 * max(1, abs(t)):
                    break
            else:
                raise RuntimeError(f"Newton did not converge for a zero of g{k}")
            if not lo <= t < hi:
                raise RuntimeError(f"a zero of g{k} left the band: {mpmath.nstr(t, 20)}")
            out.append(+t)
        return out


def split(v):
    """a 300-bit number as a pair of doubles, the subtraction at full precision"""
    with mpmath.workprec(PREC):
        h = float(v)
        return h, float(mpmath.mpf(v) - mpmath.mpf(h))


def pieces_of(k, lo, hi, width, zeros):
    """the pieces of order k: (a, b, the zeros of that piece)"""
    cuts = {lo, hi}
    for z in zeros:
        cuts |= {float(z) - width / 2, float(z) + width / 2}
    # the halved pieces of the right tail; a cut too near an existing one, or
    # near a zero, would leave a sliver of a piece and is dropped
    keep_away = list(cuts) + [float(z) for z in zeros]
    c = 1.0
    while c < hi:
        if all(abs(c - d) > MIN_PIECE for d in keep_away):
            cuts.add(c)
        c += width / 2
    cuts = sorted(c for c in cuts if lo <= c <= hi)
    edges = []
    for a, b in zip(cuts, cuts[1:]):
        n = max(1, math.ceil((b - a) / (1.05 * width)))
        edges += [a + (b - a) * i / n for i in range(n)]
    edges.append(hi)
    out = [(a, b, [z for z in zeros if a <= float(z) < b]) for a, b in zip(edges, edges[1:])]
    narrowest = min(b - a for a, b, _ in out)
    if narrowest <= (hi - lo) / BINS:
        raise RuntimeError(f"g{k} has a piece of width {narrowest:.4g}, not wider than a bin")
    return out


def fit(lo, hi, width):
    """for each order k: its pieces with coefficients, degrees and zeros"""
    results = {}
    for k in range(1, 7):
        zeros = zeros_of(k, lo, hi)
        out = []
        for a, b, zs in pieces_of(k, lo, hi, width, zeros):
            with mpmath.workprec(PREC):
                mid, half = (mpmath.mpf(a) + b) / 2, (mpmath.mpf(b) - a) / 2
                xs = [mpmath.cos(mpmath.pi * (j + mpmath.mpf(1) / 2) / M_NODES) for j in range(M_NODES)]
                ts = [mid + half * x for x in xs]
            samples = []
            for t in ts:
                v = g_values(mpmath.nstr(t, 80), 6)[k - 1]
                with mpmath.workprec(PREC):
                    for z in zs:
                        v = v / (mpmath.mpf(t) - z)
                samples.append(v)
            c = chebyshev_coefficients(samples)
            vals = [abs(float(v)) for v in samples]
            out.append({"a": a, "b": b, "z": zs, "c": c, "deg": degree_of(c, vals),
                        "vmin": min(vals), "vmax": max(vals)})
        deg = max(p["deg"] for p in out)
        for p in out:
            p["cd"] = [float(v) for v in p["c"][: deg + 1]]
            p["zdd"] = [split(z) for z in p["z"]]
        results[k] = {"pieces": out, "deg": deg, "zeros": zeros}
    return results


def degree_of(c, vals):
    tails = [0.0] * (M_NODES + 1)
    for m in range(M_NODES - 1, 0, -1):
        tails[m] = tails[m + 1] + abs(float(c[m]))
    for ref in (min(vals), max(vals)):
        if ref <= 0:
            continue
        tol = 2.0 ** TOL_EXP * ref
        for n in range(1, M_NODES):
            if tails[n + 1] <= tol:
                return n
    raise RuntimeError("no degree below the node count")


def clenshaw(c, x):
    b1 = b2 = 0.0
    for cn in reversed(c[1:]):
        b1, b2 = 2.0 * x * b1 - b2 + cn, b1
    return c[0] + x * b1 - b2


def evaluate(piece, t):
    """the double route of the header, for the report below"""
    x = 2.0 * (t - piece["a"]) / (piece["b"] - piece["a"]) - 1.0
    v = clenshaw(piece["cd"], x)
    for zhi, zlo in piece["zdd"]:
        v *= (t - zhi) - zlo
    return v


def check(results):
    """the error of each order at random points, against 300-bit values"""
    rng = random.Random(SEED)
    for k in range(1, 7):
        r = results[k]
        scale = max(p["vmax"] * max(1.0, max((abs(p["b"] - float(z)) for z in p["z"]), default=1.0))
                    for p in r["pieces"])
        worst_rel = worst_band = 0.0
        worst_t = None
        for piece in r["pieces"]:
            for _ in range(60):
                t = piece["a"] + (piece["b"] - piece["a"]) * rng.random()
                exact = g_values(mpmath.nstr(mpmath.mpf(t), 80), k)[k - 1]
                err = abs(float(mpmath.mpf(evaluate(piece, t)) - exact))
                worst_band = max(worst_band, err / math.ulp(scale))
                if exact != 0:
                    u = err / math.ulp(abs(float(exact)))
                    if u > worst_rel:
                        worst_rel, worst_t = u, t
        print(f"g{k}: {len(r['pieces'])} pieces, degree {r['deg']}, "
              f"relative to the component {worst_rel:.3g} ulp at t = {worst_t:.6g}, "
              f"{worst_band:.3g} ulp of about the largest |g| ({scale:.4g})", file=sys.stderr)


def hexlist(values, per_line=3):
    items = [float.hex(float(v)) for v in values]
    rows = ["  " + ", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\n".join(r + ("," + CONT if i + 1 < len(rows) else "") for i, r in enumerate(rows))


def intlist(values, per_line=20):
    items = [str(int(v)) for v in values]
    rows = ["  " + ", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\n".join(r + ("," + CONT if i + 1 < len(rows) else "") for i, r in enumerate(rows))


def table(out, name, body, size, kind="double", where="LAPLACE_DEVICE_TABLE"):
    out.append("// clang-format off")
    out.append(f"#define LAPLACE_LOG_PHI_{name.upper()}" + CONT)
    out.append(body)
    out.append("// clang-format on")
    out.append("#if defined(__CUDACC__)")
    out.append(f"{where} {kind} {name}_dev[{size}] = {{LAPLACE_LOG_PHI_{name.upper()}}};")
    out.append("#endif")
    out.append(f"inline constexpr {kind} {name}_host[{size}] = {{LAPLACE_LOG_PHI_{name.upper()}}};")
    out.append(f"LAPLACE_HD_FORCEINLINE {kind} {name}(int i) {{")
    out.append(f"  KOKKOS_IF_ON_DEVICE((return {name}_dev[i];))")
    out.append(f"  KOKKOS_IF_ON_HOST((return {name}_host[i];))")
    out.append("}")
    out.append("")


def small(out, name, values):
    out.append(f"LAPLACE_HD_FORCEINLINE constexpr int {name}(int k) {{")
    out.append(f"  constexpr int v[6] = {{{', '.join(str(int(x)) for x in values)}}};")
    out.append("  return v[k - 1];")
    out.append("}")


def render(results, lo, hi):
    edges, coefs, zeros, flags, bins = [], [], [], [], []
    eoff, coff, foff, npieces, degs = [], [], [], [], []
    for k in range(1, 7):
        r = results[k]
        eoff.append(len(edges))
        coff.append(len(coefs))
        foff.append(len(flags))
        npieces.append(len(r["pieces"]))
        degs.append(r["deg"])
        for p in r["pieces"]:
            edges.append(p["a"])
        edges.append(r["pieces"][-1]["b"])
        for p in r["pieces"]:
            coefs += p["cd"]
            flags.append(1 if p["z"] else 0)
            zeros += list(p["zdd"][0]) if p["z"] else [0.0, 0.0]
    for k in range(1, 7):
        es = [p["a"] for p in results[k]["pieces"]] + [hi]
        for i in range(BINS):
            left = lo + (hi - lo) * i / BINS
            bins.append(max(j for j in range(len(es) - 1) if es[j] <= left))

    out = [BEGIN,
           "// d^k log Phi / dt^k on [cheb_lo, cheb_hi): the piecewise Chebyshev expansions, one",
           "// set of pieces per order, generated by tools/reference/log_phi_cheb.py (do not",
           "// edit). On a piece without a zero of g_k the table holds the coefficients of g_k",
           "// itself; on a piece whose midpoint is a zero z of g_k it holds those of",
           "// g_k(t) / (t - z), and cheb_has_zero marks it so that the factor is restored from",
           "// the pair of doubles in cheb_zero. The bins of cheb_bin find the piece of t.",
           "namespace detail_normal {",
           "",
           f"constexpr double cheb_lo = {float.hex(float(lo))};  // the expansions cover [cheb_lo, cheb_hi)",
           f"constexpr double cheb_hi = {float.hex(float(hi))};",
           f"constexpr int cheb_bins = {BINS};  // bins of the piece lookup, of width (hi - lo) / cheb_bins",
           f"constexpr double cheb_bin_scale = {float.hex(BINS / (hi - lo))};",
           f"constexpr int cheb_size = {len(coefs)};  // coefficients in all",
           ""]
    table(out, "cheb_edge", hexlist(edges), len(edges))
    # the coefficients are the only table of several kilobytes, and their index
    # differs from thread to thread, so they live in device global memory rather
    # than in the constant bank (which is also nearly full)
    table(out, "cheb_coef", hexlist(coefs), len(coefs), where="LAPLACE_DEVICE_GLOBAL_TABLE")
    table(out, "cheb_zero", hexlist(zeros), len(zeros))
    table(out, "cheb_bin", intlist(bins), len(bins), kind="int")
    table(out, "cheb_has_zero", intlist(flags), len(flags), kind="int")
    out.append("// per order k = 1..6: the number of pieces, the degree of the expansions, and the")
    out.append("// first index of the order in cheb_edge, in cheb_coef and in cheb_has_zero")
    small(out, "cheb_pieces", npieces)
    small(out, "cheb_degree", degs)
    small(out, "cheb_edge_offset", eoff)
    small(out, "cheb_coef_offset", coff)
    small(out, "cheb_flag_offset", foff)
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
    ap.add_argument("--width", type=float, default=1.0, help="the width of a piece of the left part")
    ap.add_argument("--update", action="store_true", help="write the block into normal.hpp")
    args = ap.parse_args()
    results = fit(args.lo, args.hi, args.width)
    for k in range(1, 7):
        r = results[k]
        print(f"k={k} zeros {[mpmath.nstr(z, 12) for z in r['zeros']]} degree {r['deg']}", file=sys.stderr)
        for p in r["pieces"]:
            print(f"  [{p['a']:+.4f}, {p['b']:+.4f}) degree {p['deg']:3d} |value| in "
                  f"[{p['vmin']:.4g}, {p['vmax']:.4g}]{' zero' if p['z'] else ''}", file=sys.stderr)
    check(results)
    block = render(results, args.lo, args.hi)
    if args.update:
        update(block)
        print(f"wrote {NORMAL_HPP}", file=sys.stderr)
    else:
        sys.stdout.write(block)


if __name__ == "__main__":
    main()
