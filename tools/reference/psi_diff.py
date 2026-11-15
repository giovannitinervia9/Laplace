"""Reference values of the polygamma differences at a shift,

    psi^(n)(z) - psi^(n)(z + h),   n = 0..5,   z > 0,   h > 0,

computed with mpmath at 200 bits from the definition (mpmath.psi), which is
independent of the route under test (own::detail_ibd::dpsi_all, the asymptotic
expansion plus a recurrence applied to the difference, in
include/laplace/math/incomplete_beta_deriv.hpp). Order 5 is what the sixth
derivative of a log-gamma difference at an integer shift needs (the negative
binomial and the beta-binomial masses).

Usage (from the repository root):  python -I tools/reference/psi_diff.py

File tests/data/reference/psi_diff.csv, one row per (z, h, n):
    z, h, n, hi, lo        (z, h, hi, lo as hexadecimal doubles)
hi is the difference rounded to double and lo the rounded remainder, so that
(y - hi) - lo is the error of a computed y in units of hi.

The grid: z from 1e-3 to 1e8 (the recurrence below 15, the expansion above),
h/z from 1e-12 (where the difference cancels: psi^(n)(z) - psi^(n)(z+h) is
about -h psi^(n+1)(z), tiny against either term) to 1e12 (where the difference
is the first term alone), with h also taken as a small integer, the shift of a
discrete mass.
"""
import itertools
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "psi_diff.csv"

Z = ["1e-3", "0.4", "1.0", "3.5", "12.0", "15.0", "40.0", "1e3", "1e5", "1e8"]
RATIO = ["1e-12", "1e-8", "1e-4", "0.01", "0.5", "1.0", "7.0", "1e4", "1e8", "1e12"]
SHIFT = ["1.0", "2.0", "5.0", "40.0", "1000.0"]
ORDERS = range(6)


def pairs():
    out = []
    for zs, rs in itertools.product(Z, RATIO):
        z = mpmath.mpf(zs)
        out.append((z, z * mpmath.mpf(rs)))
    for zs, hs in itertools.product(Z, SHIFT):
        out.append((mpmath.mpf(zs), mpmath.mpf(hs)))
    return out


def split(v):
    hi = float(v)
    lo = float(v - mpmath.mpf(hi))
    return hi, lo


def main():
    mpmath.mp.prec = 200
    lines = ["z,h,n,hi,lo"]
    for z, h in pairs():
        for n in ORDERS:
            d = mpmath.psi(n, z) - mpmath.psi(n, z + h)
            hi, lo = split(d)
            lines.append(f"{float(z).hex()},{float(h).hex()},{n},{hi.hex()},{lo.hex()}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(lines) - 1} rows")


if __name__ == "__main__":
    main()
