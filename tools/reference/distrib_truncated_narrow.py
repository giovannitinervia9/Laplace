"""Reference values of log Z and its derivatives in (mu, sigma) for a Gaussian truncated
to a narrow window [a, b] (truncated_distrib over gaussian1_distrib), orders 0 to 4.

Usage (from the repository root):  python -I tools/reference/distrib_truncated_narrow.py

File tests/data/reference/distrib_truncated_narrow.csv, one row per value:
    a, b, K, nb, hi, lo
with nb the number of derivatives in sigma (K - nb in mu), all but K and nb in
hexadecimal. b is the double nearest to a + w, stored as it is, so that the library and
the reference see the same window.

Method: mpmath.diff of the DEFINITION, log Z = log(Phi((b - mu)/sigma) - Phi((a - mu)/sigma)),
at 400 and 500 bits (the difference cancels by about 40 bits at w = 1e-12), a row kept
when the two agree to 2^-80 relative.
"""
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_truncated_narrow.csv"
AGREE = mpmath.mpf(2) ** -80
MU, SIGMA = 0.4, 1.3
STARTS = [1.0, 4.0, -3.0]
WIDTHS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


def value(a, b, K, nb, prec):
    with mpmath.workprec(prec):
        A, B = mpmath.mpf(a), mpmath.mpf(b)
        f = lambda mu, s: mpmath.log(mpmath.ncdf((B - mu) / s) - mpmath.ncdf((A - mu) / s))
        return mpmath.diff(f, (mpmath.mpf(MU), mpmath.mpf(SIGMA)), (K - nb, nb))


def main():
    lines, dropped = [], 0
    for a in STARTS:
        for w in WIDTHS:
            b = a + w
            for K in range(0, 5):
                for nb in range(0, K + 1):
                    v1, v2 = value(a, b, K, nb, 400), value(a, b, K, nb, 500)
                    if abs(v1 - v2) > AGREE * abs(v2):
                        dropped += 1
                        continue
                    hi = float(v2)
                    lo = float(v2 - mpmath.mpf(hi))
                    lines.append(f"{a.hex()},{b.hex()},{K},{nb},{hi.hex()},{lo.hex()}")
    OUT.write_text("a,b,K,nb,hi,lo\n" + "\n".join(lines) + "\n")
    print(f"{len(lines)} rows, {dropped} dropped")


if __name__ == "__main__":
    main()
