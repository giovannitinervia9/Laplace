"""Reference values of the polygamma remainder, computed with mpmath.

The remainder is psi^(n)(x) minus the leading term of its asymptotic
expansion,

    mu_0(x) = log x,    mu_n(x) = (-1)^(n+1) (n - 1)! x^-n   (n >= 1),

so r_n(x) = psi^(n)(x) - mu_n(x). It exists because a weighted sum of
polygammas whose monomials cancel algebraically loses those digits when each
term is evaluated on its own; see the documentation of
own::polygamma_monomial_remainder in include/laplace/math/polygamma.hpp.

Usage (from the repository root):
    python -I tools/reference/psi_remainder.py

Files (format as in bessel.py: the argument, then hi and lo in hexadecimal,
the two halves of the double-double of the true value):

    psi_remainder_d{n}.csv   x   r_n(x),  n = 0..6

Method, and why it is independent of the library. mpmath evaluates psi^(n)
itself to arbitrary precision and the monomial is elementary, so the reference
is the subtraction done at a precision where it costs nothing: the remainder is
of relative size 1/(2x) against psi^(n), so the subtraction loses about
log2(x) bits and no more. The library, by contrast, never performs that
subtraction at all — it drops the leading term of the expansion — so the two
routes share no arithmetic.

Every row is computed at two precisions and kept only if they agree to
AGREE relative, and the number dropped is printed. That guard is not
ceremony: a reference written earlier the same day with too little working
precision silently returned a value wrong by a factor of 2.7 and was believed
over a correct library for one measurement.
"""

import math
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
NMAX = 6
AGREE = mpmath.mpf(2) ** -140


def dd(v):
    """The value as a pair of doubles, hi + lo."""
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def monomial(n, x):
    """mu_n(x), the leading term of the asymptotic expansion of psi^(n)."""
    if n == 0:
        return mpmath.log(x)
    sign = 1 if (n % 2 == 1) else -1
    return sign * mpmath.factorial(n - 1) / x**n


def remainder(n, x, prec):
    """r_n(x) = psi^(n)(x) - mu_n(x) at the given working precision."""
    old = mpmath.mp.prec
    mpmath.mp.prec = prec
    try:
        xx = mpmath.mpf(x)
        return mpmath.psi(n, xx) - monomial(n, xx)
    finally:
        mpmath.mp.prec = old


def remainder_checked(n, x):
    """The remainder at two precisions, or None if they disagree.

    The subtraction loses about log2(x) bits, so the precision is sized from x
    and the second run is given half again as much.
    """
    bits = max(abs(math.log2(x)), 1.0)
    p1 = int(bits) + 220
    a = remainder(n, x, p1)
    b = remainder(n, x, p1 + p1 // 2 + 60)
    if b == 0:
        return b if a == 0 else None
    if abs(a - b) / abs(b) > AGREE:
        return None
    return b


def grid():
    """The x values: the whole working range, with the recurrence threshold.

    X0 = 15 is where the library stops shifting the argument up and uses the
    expansion directly, so the points on either side of it are the ones that
    show a seam. The large end reaches 1e12 because that is where the beta's
    precision phi lives when the remainder is worth having at all.
    """
    pts = []
    e = -3.0
    while e <= 12.001:
        pts.append(10.0**e)
        e += 0.25
    for v in [0.5, 1.0, 1.4616321449683623, 2.0, 7.0, 13.0, 14.0, 14.5, 15.0,
              15.5, 16.0, 20.0, 40.0, 50.0, 100.0, 1e3, 3e3, 1e4, 1e6, 1e8]:
        pts.append(v)
    seen, out = set(), []
    for p in sorted(pts):
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xs = grid()
    for n in range(0, NMAX + 1):
        rows = ["x,hi,lo"]
        dropped = 0
        for x in xs:
            v = remainder_checked(n, x)
            if v is None:
                dropped += 1
                continue
            hi, lo = dd(v)
            rows.append("%s,%s,%s" % (float.hex(x), float.hex(hi), float.hex(lo)))
        name = "psi_remainder_d%d.csv" % n
        (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("%s: %d rows%s" % (name, len(rows) - 1, ", %d dropped" % dropped if dropped else ""))


if __name__ == "__main__":
    main()
