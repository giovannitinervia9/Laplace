"""Taylor tables of chi(x) = lgamma(1 + x)/x + gamma for powerexp_distrib (include/laplace/distrib/
powerexp.hpp), and the check of their accuracy.

For p >= 2 the derivatives of the shape functions A(p) and g(p) are formed from chi at x = 1/p and
x = 3/p, x in (0, 3/2]. chi is analytic for x > -1. It is evaluated from three Taylor expansions:
  centre 0    on [0, 0.15],   coefficients (-1)^k zeta(k)/k (the series of lgamma(1 + x)),
  centre 0.4  on (0.15, 0.65],
  centre 1.05 on (0.65, 1.5],
whose coefficients chi^(n)(c)/n! are computed here with mpmath at 60 digits and rounded once.

  python -I tools/reference/powerexp_chi_table.py table   prints the C++ table (hex doubles)
  python -I tools/reference/powerexp_chi_table.py check   the error of chi^(j), j = 0..6, in double
"""
import math
import sys

import mpmath

mpmath.mp.dps = 60
EG = mpmath.euler
# (centre, upper end of the interval, last power N)
SPEC = [(0.0, 0.15, 32), (0.4, 0.65, 36), (1.05, 1.5, 40)]


def chi(t):
    return mpmath.loggamma(1 + t) / t + EG


def coeffs(c, N):
    if c == 0:
        return [mpmath.mpf(0)] + [(-1) ** k * mpmath.zeta(k) / k for k in range(2, N + 2)]
    return mpmath.taylor(chi, mpmath.mpf(c), N)


TABLES = [(c, hi, [float(v) for v in coeffs(c, N)]) for c, hi, N in SPEC]


def chi_derivs(x):
    """chi^(j)(x), j = 0..6, in double, as the header evaluates them"""
    for c, hi, a in TABLES:
        if x <= hi:
            break
    t = x - c
    # Horner for each order on sum_n a_n n!/(n-j)! t^(n-j); the header advances the seven
    # recurrences in one loop over n, which performs the same operations for each order
    d = [0.0] * 7
    for n in range(len(a) - 1, -1, -1):
        f = 1.0
        for j in range(7):
            if j <= n:
                d[j] = d[j] * t + a[n] * f
            f *= float(n - j)
    return d


def table():
    for (c, hi, a), name in zip(TABLES, ["0", "1", "2"]):
        print(f"// centre {c}, up to x = {hi}: {len(a)} coefficients")
        print(f"#define LAPLACE_POWEREXP_CHI_{name} \\")
        cells = [v.hex() if v != 0 else "0x0.0p+0" for v in a]
        for i in range(0, len(cells), 4):
            end = ", \\" if i + 4 < len(cells) else ""
            print("    " + ", ".join(cells[i:i + 4]) + end)


def check():
    worst = 0.0
    lo = 0.0
    for c, hi, a in TABLES:
        w = [0.0] * 7
        for i in range(1, 81):
            x = lo + (hi - lo) * i / 80
            v = chi_derivs(x)
            for j in range(7):
                e = mpmath.diff(chi, mpmath.mpf(x), j)
                u = math.ldexp(1.0, math.frexp(abs(float(e)))[1] - 53)
                w[j] = max(w[j], float(abs(mpmath.mpf(v[j]) - e) / u))
        print(f"centre {c} on ({lo}, {hi}], N = {len(a) - 1}: worst ulp by order j = 0..6: "
              + " ".join(f"{e:.3g}" for e in w))
        worst = max(worst, max(w))
        lo = hi
    print(f"worst {worst:.3g} ulp")


if __name__ == "__main__":
    {"table": table, "check": check}[sys.argv[1]]()
