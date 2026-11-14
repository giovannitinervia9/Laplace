"""Reference values that separate the conditioning of the log F derivatives from the
accuracy of the routes that compute them.

Usage (from the repository root):  python -I tools/reference/distrib_conditioning.py

The derivatives of log F of a gaussian at large positive z are dominated by
L_1 = R(z) = phi(z)/Phi(z), whose logarithmic derivative is about -z. The argument
z = (y - mu)/sigma carries a rounding error of up to half an ulp, that is about
z eps/2 in absolute terms, so the value of the derivative inherits a relative error
of about z^2 eps/2 whatever route computes it. This file lets the tests see that:
every row has mu = 0 and a scale of 1, so that z = y is exact in double and the only
error left is the route's own.

File tests/data/reference/distrib_conditioning.csv, one row per value:
    dist (0 gaussian1 in sigma, 1 gaussian2 in the variance, 2 gaussian3 in the
    precision, 3 gumbel in sigma), upper (0 log F, 1 log(1 - F)), y, K, b, hi, lo
with b the number of derivatives in the second parameter (K - b in mu); y, hi and lo
in hexadecimal. The scale is 1 in all three parametrizations (sigma = 1, v = 1,
tau = 1), so the three differ only in the chain rule applied to the same z.

Method: mpmath.diff of the definition log(ncdf(+-(y - mu)/s(w))) at 300 and 400 bits,
with s(w) = w, sqrt(w) and 1/sqrt(w) for the three parametrizations; a row is kept
when the two precisions agree to 2^-80 relative. The route under test shares nothing
with this one.
"""
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_conditioning.csv"
AGREE = mpmath.mpf(2) ** -80
MAXK = 6

# z = y exactly, since mu = 0 and every scale is 1
POINTS = [mpmath.mpf(v) for v in (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 24, 27, 30)]


def scale_of(dist, w):
    if dist == 0:
        return w                      # sigma
    if dist == 1:
        return mpmath.sqrt(w)         # variance
    if dist == 3:
        return w                      # gumbel sigma
    return 1 / mpmath.sqrt(w)         # precision


def log_cdf(dist, upper, z):
    if dist == 3:                         # gumbel: F = exp(-e^-z)
        u = mpmath.exp(-z)
        return mpmath.log(-mpmath.expm1(-u)) if upper else -u
    return mpmath.log(mpmath.ncdf(-z if upper else z))


def value(dist, upper, y, K, b, prec):
    """d^K log F / dmu^(K-b) dw^b at mu = 0, w = 1."""
    with mpmath.workprec(prec):
        def f(mu, w):
            z = (y - mu) / scale_of(dist, w)
            return log_cdf(dist, upper, z)
        a = K - b
        if a and b:
            return mpmath.diff(f, (mpmath.mpf(0), mpmath.mpf(1)), (a, b))
        if a:
            return mpmath.diff(lambda mu: f(mu, mpmath.mpf(1)), mpmath.mpf(0), a)
        return mpmath.diff(lambda w: f(mpmath.mpf(0), w), mpmath.mpf(1), b)


def split(x):
    """x as a double plus the remaining double, so that hi + lo is the exact value."""
    hi = float(x)
    lo = float(x - mpmath.mpf(hi))
    return hi, lo


def hexf(v):
    """The C hexadecimal form, which strtod reads back exactly (float.hex)."""
    return float(v).hex()


def main():
    rows, dropped = [], 0
    for dist in (0, 1, 2, 3):
        for upper in (0, 1):
            for y in POINTS:
                yy = -y if upper else y
                for K in range(1, MAXK + 1):
                    for b in range(K + 1):
                        a = value(dist, upper, yy, K, b, 300)
                        c = value(dist, upper, yy, K, b, 400)
                        if a == 0 or abs(a - c) > AGREE * abs(c):
                            dropped += 1
                            continue
                        hi, lo = split(c)
                        rows.append((dist, upper, hexf(float(yy)), K, b, hexf(hi), hexf(lo)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        fh.write("dist,upper,y,K,b,hi,lo\n")
        for r in rows:
            fh.write("%d,%d,%s,%d,%d,%s,%s\n" % r)
    print(f"{len(rows)} rows written to {OUT}, {dropped} dropped")


if __name__ == "__main__":
    main()
