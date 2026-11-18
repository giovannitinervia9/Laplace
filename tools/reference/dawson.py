"""Reference values of Dawson's integral F and of its derivatives to order 6,
computed with mpmath.

    F(x) = exp(-x^2) int_0^x exp(s^2) ds,

odd, bounded, with a single maximum F = 0.5410442246351816 at
x = 0.9241388730045917, F(x) ~ x near 0 and F(x) ~ 1/(2x) as x -> inf, and
F'(x) = 1 - 2 x F(x).

Usage (from the repository root):  python -I tools/reference/dawson.py [--workers=N] [--cheb]

Files (format as in stirling.py: the argument x, then hi, lo in hexadecimal):
    dawson.csv        x   F(x)
    dawson_d{k}.csv   x   F^(k)(x), k = 1..6

Method. The value is F(x) = (sqrt(pi)/2) exp(-x^2) erfi(x), from mpmath's own
erfi. Two routes check it on every point of the grid: Kummer's function,
F(x) = x exp(-x^2) M(1/2, 3/2, x^2), and, for |x| <= 12, the quadrature of the
definition itself (mpmath's tanh-sinh rule on int_0^x exp(s^2) ds), which is
the route that shares nothing with the library's. The derivatives come from
the defining recurrence

    F^(1) = 1 - 2 x F,     F^(k+1) = -2 (x F^(k) + k F^(k-1)),

which is exact but cancels: at large x the two terms of each step agree to
about log2(x) bits, so order 6 loses about 2 log2(x) + 5 log2(x) bits, 190 at
x = 1e8. Everything is computed at 600 and at 700 bits and a value is kept
only if the two agree to 2^-100 relative and the result is a finite normal
double. The library uses neither route (it sums exp(-x^2) sum_m b_k(m) x^n/m!
below the threshold and the differentiated asymptotic series above it), so the
two are independent.

Points: x log-uniform in [2^-45, 1e8]; x uniform in [0, 12], where the
maximum, the zeros of the derivatives and the change of route all lie; around
each candidate threshold T, x = T 2^u with u uniform in [-1/4, 1/4]; the
maximum and the zeros of F^(k) themselves; and the mirror image of a third of
the grid, since F is odd and a wrong sign must fail.

--cheb prints the Chebyshev coefficients of the two polynomial candidates of
the measurement (a fit of F(x)/x in x^2 on [0, 5/2] and one of 2 x F(x) - 1 in
1/x^2 on [5/2, 7]), for transcription into benchmarks/math_dawson.cpp.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261118
KMAX = 6
PRECS = (600, 700)
AGREE = mpmath.mpf(2) ** -100
TINY = 2.2250738585072014e-308
THRESH = (2.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0)  # candidate route boundaries
XMAX_QUAD = 12.0  # the quadrature check is run below this


def dawson(x):
    """F(x) through mpmath's erfi."""
    return mpmath.sqrt(mpmath.pi) / 2 * mpmath.exp(-x**2) * mpmath.erfi(x)


def dawson_kummer(x):
    """F(x) = x exp(-x^2) M(1/2, 3/2, x^2)."""
    return x * mpmath.exp(-x**2) * mpmath.hyp1f1(mpmath.mpf(1) / 2, mpmath.mpf(3) / 2, x**2)


def dawson_quad(x):
    """F(x) from the definition, by quadrature."""
    if x == 0:
        return mpmath.mpf(0)
    return mpmath.exp(-x**2) * mpmath.quad(lambda s: mpmath.exp(s**2), [0, x])


def derivatives(x, f):
    """F, F^(1), ..., F^(KMAX) from the defining recurrence."""
    out = [f, 1 - 2 * x * f]
    for k in range(1, KMAX):
        out.append(-2 * (x * out[k] + k * out[k - 1]))
    return out


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def row(xv):
    vals = {}
    checks = []
    for p in PRECS:
        with mpmath.workprec(p):
            mx = mpmath.mpf(xv)
            f = dawson(mx)
            vals[p] = derivatives(mx, f)
            if p == PRECS[1] and f != 0:
                checks.append(abs(dawson_kummer(mx) - f) / abs(f))
                if abs(xv) <= XMAX_QUAD:
                    checks.append(abs(dawson_quad(mx) - f) / abs(f))
    out = []
    for k in range(KMAX + 1):
        a, b = vals[PRECS[0]][k], vals[PRECS[1]][k]
        if b == 0 or abs(a - b) > AGREE * abs(b):
            out.append(None)
            continue
        hi, lo = dd(b)
        if not (math.isfinite(hi) and abs(hi) >= TINY):
            out.append(None)
            continue
        out.append((hi, lo))
    worst = max((float(c) for c in checks), default=0.0)
    return xv, out, worst


def sample():
    rng = random.Random(SEED)
    xs = [0.9241388730045917]  # the maximum of F
    # every zero of F^(k) on the positive axis, ceil(k/2) of them for the order k
    xs += [0.9241388730045917, 1.501975268, 0.595679356, 1.955608092, 1.066311887,
           2.340530538, 0.474246676, 1.466192267, 2.680438697, 0.878711253,
           1.819221958, 2.987984368]
    for _ in range(700):  # log-uniform in [2^-45, 1e8]
        xs.append(math.exp(rng.uniform(-45 * math.log(2.0), 8 * math.log(10.0))))
    for _ in range(500):  # uniform in [0, 12]: the maximum, the zeros, the change of route
        xs.append(rng.uniform(0.0, 12.0))
    for t in THRESH:
        for _ in range(60):
            xs.append(t * 2.0 ** rng.uniform(-0.25, 0.25))
    xs = sorted(set(xs))
    neg = rng.sample(xs, len(xs) // 3)
    return sorted(set(xs + [-v for v in neg]))


def cheb():
    """The polynomial candidates of the measurement."""
    mpmath.mp.prec = 300
    print("# F(x)/x as a polynomial in u = x^2, u in [0, 25/4] (x in [0, 5/2])")
    for n in (14, 16, 18, 20, 22):
        c, err = mpmath.chebyfit(lambda u: dawson(mpmath.sqrt(u)) / mpmath.sqrt(u) if u > 0 else mpmath.mpf(1),
                                 [0, 6.25], n, error=True)
        print(f"#   degree {n - 1}: max error {mpmath.nstr(err, 3)}")
        if err < mpmath.mpf("1e-18"):
            print("    // highest power first")
            print("\n".join(f"    {mpmath.nstr(v, 20)}," for v in c))
            break
    print("# F(x) as a polynomial in x on [5/2, 7], and in v = 1/x^2 on [1/49, 4/25]")
    for n in (16, 20, 24, 28, 32, 36):
        c, err = mpmath.chebyfit(dawson, [2.5, 7.0], n, error=True)
        print(f"#   in x,   degree {n - 1}: max error {mpmath.nstr(err, 3)}")
    for n in (14, 16, 18, 20, 22, 24):
        c, err = mpmath.chebyfit(lambda v: 2 / mpmath.sqrt(v) * dawson(1 / mpmath.sqrt(v)) - 1,
                                 [1.0 / 49.0, 0.16], n, error=True)
        print(f"#   degree {n - 1}: max error {mpmath.nstr(err, 3)}")
        if err < mpmath.mpf("1e-18"):
            print("    // highest power first")
            print("\n".join(f"    {mpmath.nstr(v, 20)}," for v in c))
            break


def main():
    workers = 4
    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            workers = int(arg.split("=")[1])
        elif arg == "--cheb":
            cheb()
            return
    xs = sample()
    with Pool(workers) as pool:
        out = pool.map(row, xs, chunksize=8)
    worst = max(w for _, _, w in out)
    print(f"erfi against Kummer's function and the quadrature: largest relative difference {worst:.3g}")
    for k in range(KMAX + 1):
        name = "dawson" if k == 0 else f"dawson_d{k}"
        path = OUT / f"{name}.csv"
        n = dropped = 0
        with open(path, "w", newline="\n") as f:
            f.write("x,hi,lo\n")
            for x, vals, _ in out:
                if vals[k] is None:
                    dropped += 1
                    continue
                hi, lo = vals[k]
                f.write(f"{x.hex()},{hi.hex()},{lo.hex()}\n")
                n += 1
        print(f"{path.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
