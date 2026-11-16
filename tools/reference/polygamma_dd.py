"""Reference values for the double-double polygammas of orders 0..13
(include/laplace/math/polygamma_dd.hpp):

    psi_n(x),
    G_n(x, h) = psi_n(x + h) - psi_n(x),          h > 0,
    r_n(x)    = psi_n(x) - mu_n(x),               mu_0 = log x, mu_n = (-1)^(n+1) (n-1)! x^-n.

Usage (from the repository root):
    python -I tools/reference/polygamma_dd.py

File tests/data/reference/polygamma_dd.csv, one row per value:

    kind, n, xhi, xlo, hhi, hlo, hi, mid, lo

kind 0 = psi_n, 1 = G_n, 2 = r_n; the argument x = xhi + xlo and the shift
h = hhi + hlo are double-doubles (h = 0 for kinds 0 and 2); the value is
hi + mid + lo, three doubles, so that the error of a double-double result can
be measured at 2^-106 and below. All doubles in hexadecimal.

Method, and why it is independent of the library. mpmath evaluates psi_n by
the Hurwitz zeta function, at 300 bits; G_n and r_n are the subtractions done
at that precision, which lose at most log2(x/h) and log2(x) bits, less than
60 of the 300 on this grid. The library never subtracts: it writes G_n as sums
of terms of one sign and drops the leading term of the expansion for r_n.
Every row is computed again at 400 bits and kept only if the two agree to
2^-200 relative. mpmath.psi itself was checked once against the integral
representations psi_n(x) = (-1)^(n+1) int_0^inf t^n e^(-xt) / (1 - e^(-t)) dt
and their analogues for G_n and psi(x) - log x, by quadrature, at 55 points up
to x = 1e13 and n = 13: agreement to 1.2e-75 (docs/numerics.md).

The grid, concentrated where the layer is hard:
  psi_n, r_n: x from 1e-6 to 5 (the pole terms dominate), 0.25 to 60 every
    0.37 plus points at both sides of the shift threshold X0 = 20 (the number
    of recurrence steps changes there), 1e2 to 3.3e15 (one or two series
    terms), and double-double arguments d/p with p = 3 and 7;
  G_n: x from 1e-3 to 1e13 against h from 1e-3 to 333, so x/h from 3e-6 to
    1e16 (the naive difference of two values would lose x/h units), and the
    generalized gamma's pairs x = d/p, h = 1/p for d = 10..1e9, p = 0.003..30,
    both in double-double: the lognormal limit d/p -> inf is x/h = d -> inf.
Values above 1e300 in magnitude are left out (the library declares them).
"""
from pathlib import Path

import mpmath as mp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "polygamma_dd.csv"
ORDERS = range(14)


def split3(v):
    a = float(v)
    r = v - mp.mpf(a)
    b = float(r)
    r -= mp.mpf(b)
    return a, b, float(r)


def dd(v):
    a = float(v)
    return a, float(v - mp.mpf(a))


def xs_psi():
    pts = []
    for e in range(-6, 1):
        for m in ("1", "2.5", "5"):
            pts.append(mp.mpf(m + "e%d" % e))
    v = mp.mpf("0.25")
    while v < 60:
        pts.append(v)
        v += mp.mpf("0.37")
    for s in ("19", "19.999999", "20", "20.000001", "21"):
        pts.append(mp.mpf(s))
    for e in range(2, 16):
        for m in ("1", "3.3"):
            pts.append(mp.mpf(m + "e%d" % e))
    for d, p in (("7", "3"), ("1e3", "7"), ("1e7", "3"), ("2e10", "7"), ("40", "3"), ("59", "3")):
        pts.append(mp.mpf(d) / mp.mpf(p))
    return pts


def xh_G():
    out = []
    X = ["1e-3", "0.3", "1.0", "3.5", "14.7", "19.99", "25.3", "40.1", "1e3", "1e5", "1e7", "1e10", "1e13"]
    H = ["1e-3", "0.01", "0.1", "0.5", "1.0", "2.0", "7.0", "333.3"]
    for xs in X:
        for hs in H:
            out.append((mp.mpf(xs), mp.mpf(hs)))
    for d in ("10", "1e3", "1e5", "1e7", "1e9"):
        for p in ("0.003", "0.3", "1", "3", "30"):
            P = mp.mpf(p)
            out.append((mp.mpf(d) / P, 1 / P))
    return out


def mu(n, x):
    return mp.log(x) if n == 0 else (-1) ** (n + 1) * mp.factorial(n - 1) * x ** (-n)


def values(kind, n, x, h):
    if kind == 0:
        return mp.psi(n, x)
    if kind == 1:
        return mp.psi(n, x + h) - mp.psi(n, x)
    return mp.psi(n, x) - mu(n, x)


def row(kind, n, xs, hs):
    with mp.workprec(300):
        x = mp.mpf(xs[0]) + mp.mpf(xs[1])
        h = mp.mpf(hs[0]) + mp.mpf(hs[1])
        v = values(kind, n, x, h)
    with mp.workprec(400):
        x = mp.mpf(xs[0]) + mp.mpf(xs[1])
        h = mp.mpf(hs[0]) + mp.mpf(hs[1])
        w = values(kind, n, x, h)
        if abs(v - w) > abs(w) * mp.mpf(2) ** -200:
            return None
    if abs(v) > mp.mpf("1e300"):
        return None
    return ",".join([str(kind), str(n), *(t.hex() for t in (*xs, *hs, *split3(v)))])


def main():
    mp.mp.prec = 300
    lines = ["kind,n,xhi,xlo,hhi,hlo,hi,mid,lo"]
    dropped = 0
    jobs = []
    for x0 in xs_psi():
        for n in ORDERS:
            jobs += [(0, n, dd(x0), (0.0, 0.0)), (2, n, dd(x0), (0.0, 0.0))]
    for x0, h0 in xh_G():
        for n in ORDERS:
            jobs.append((1, n, dd(x0), dd(h0)))
    for j in jobs:
        r = row(*j)
        if r is None:
            dropped += 1
        else:
            lines.append(r)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} rows, {dropped} left out, written to {OUT}")


if __name__ == "__main__":
    main()
