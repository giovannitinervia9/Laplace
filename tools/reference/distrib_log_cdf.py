"""Reference values of the derivatives of log F and log(1 - F) in (mu, sigma)
for the location-scale distributions gaussian1 and laplace, orders 1 to 6.

Usage (from the repository root):  python -I tools/reference/distrib_log_cdf.py

File tests/data/reference/distrib_log_cdf.csv, one row per value:
    dist (0 gaussian1, 1 laplace, 2 poisson, 3 gamma1), upper (0 log F, 1 log(1 - F)),
    y, K, b, hi, lo, mu, phi (sigma for the first two, 0 for the Poisson)
with b the number of derivatives in sigma (K - b in mu); y, hi, lo, mu in
hexadecimal. mu = 0.4 and sigma = 1.3 for the first two; for the Poisson,
derivatives in mu of log Q(y + 1, mu) and log P(y + 1, mu) (mpmath.gammainc),
b = 0, on a grid of (y, mu) around and away from the median.

Method: mpmath.diff (high-precision finite differences on the analytic
function, with its own step control) of the closed forms
log ncdf((y - mu)/sigma) and, for the Laplace, log(e^z/2) or log(1 - e^-z/2),
at 300 and 400 bits; a row is kept if the two agree to 2^-80 relative. This
route shares nothing with the library's (derivatives in z by log_Phi_dK or a
polynomial recursion, carried to (mu, sigma) by the location-scale rules).
For the Laplace, points with |y - mu| < 0.05 are not used (the kink of the
second derivative at y = mu).
"""
import math
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_log_cdf.csv"
MU, SIGMA = 0.4, 1.3
AGREE = mpmath.mpf(2) ** -80


def log_cdf(dist, upper, y, mu, s):
    z = (y - mu) / s
    if upper:
        z = -z
    if dist == 0:
        return mpmath.log(mpmath.ncdf(z))
    if z < 0:
        return z - mpmath.log(2)
    return mpmath.log(1 - mpmath.exp(-z) / 2)


def value(dist, upper, y, K, b, prec):
    with mpmath.workprec(prec):
        f = lambda mu, s: log_cdf(dist, upper, mpmath.mpf(y), mu, s)
        return mpmath.diff(f, (mpmath.mpf(MU), mpmath.mpf(SIGMA)), (K - b, b))


def poisson_value(upper, k, mu, K, prec):
    with mpmath.workprec(prec):
        if upper:
            f = lambda m: mpmath.log(mpmath.gammainc(k + 1, 0, m, regularized=True))
        else:
            f = lambda m: mpmath.log(mpmath.gammainc(k + 1, m, mpmath.inf, regularized=True))
        return mpmath.diff(f, mpmath.mpf(mu), K)


def gamma1_value(upper, y, mu0, phi0, K, b, prec):
    """d^K / dmu^(K-b) dphi^b of log P(1/phi, y/(mu phi)) (upper: log Q)"""
    with mpmath.workprec(prec):
        def f(mu, phi):
            a, x = 1 / phi, mpmath.mpf(y) / (mu * phi)
            if upper:
                return mpmath.log(mpmath.gammainc(a, x, mpmath.inf, regularized=True))
            return mpmath.log(mpmath.gammainc(a, 0, x, regularized=True))
        return mpmath.diff(f, (mpmath.mpf(mu0), mpmath.mpf(phi0)), (K - b, b))


def main():
    ys = {0: [-30.0, -12.0, -3.0, -1.0, 0.2, 0.4, 1.0, 1.7, 3.0, 9.0, 20.0, 35.0],
          1: [-30.0, -6.0, -1.0, 0.2, 0.6, 1.7, 3.0, 9.0, 30.0]}
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("dist,upper,y,K,b,hi,lo,mu,phi\n")
        for dist in (0, 1):
            for upper in (0, 1):
                for y in ys[dist]:
                    for K in range(1, 7):
                        for b in range(K + 1):
                            a, c = value(dist, upper, y, K, b, 300), value(dist, upper, y, K, b, 400)
                            if c == 0 or abs(a - c) > AGREE * abs(c):
                                dropped += 1
                                continue
                            hi = float(c)
                            if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                                dropped += 1
                                continue
                            lo = float(c - mpmath.mpf(hi))
                            f.write(f"{dist},{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{MU.hex()},{SIGMA.hex()}\n")
                            n += 1
        for mu in (0.05, 0.5, 3.0, 10.0, 40.0, 300.0):
            for k in sorted({0, 1, 2, int(mu / 3), int(mu) - 1, int(mu), int(mu) + 1, int(2 * mu) + 3, int(5 * mu) + 10}):
                if k < 0:
                    continue
                for upper in (0, 1):
                    for K in range(1, 7):
                        a, c = poisson_value(upper, k, mu, K, 300), poisson_value(upper, k, mu, K, 400)
                        if c == 0 or abs(a - c) > AGREE * abs(c):
                            dropped += 1
                            continue
                        hi = float(c)
                        if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                            dropped += 1
                            continue
                        lo = float(c - mpmath.mpf(hi))
                        f.write(f"2,{upper},{float(k).hex()},{K},0,{hi.hex()},{lo.hex()},{float(mu).hex()},0x0p+0\n")
                        n += 1
        # gamma1 (mu = 2, phi): F = P(1/phi, y/(mu phi)); every component (K - b in mu, b in phi)
        for phi in (0.02, 0.3, 4.0):
            for y in (0.05, 0.8, 2.0, 3.5, 9.0):
                for upper in (0, 1):
                    for K in range(1, 7):
                        for b in range(K + 1):
                            a, c = gamma1_value(upper, y, 2.0, phi, K, b, 300), gamma1_value(upper, y, 2.0, phi, K, b, 400)
                            if c == 0 or abs(a - c) > AGREE * abs(c):
                                dropped += 1
                                continue
                            hi = float(c)
                            if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                                dropped += 1
                                continue
                            lo = float(c - mpmath.mpf(hi))
                            f.write(f"3,{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{(2.0).hex()},{phi.hex()}\n")
                            n += 1
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
