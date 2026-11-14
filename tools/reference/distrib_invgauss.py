"""Closed forms and reference values for invgauss_distrib (include/laplace/distrib/invgauss.hpp).

Usage (from the repository root):  python -I tools/reference/distrib_invgauss.py

Part 1 (SymPy): every closed form of the header is checked against SymPy's own derivative of
the definition: the parameter derivatives of the log-density (orders 1 to 6), the mixed
(y, theta) derivatives (J + K <= 6), the expected information and its derivatives (orders 1
to 4), and the Riccati and logistic recurrences used for the cdf. Any mismatch stops the script.

Part 2 (mpmath): reference values of log F and log(1 - F) and of their derivatives in
(mu, phi), orders 0 to 6, taken from the DEFINITION
    F(y) = ncdf(z1) + exp(2/(phi mu)) ncdf(-z2),
    S(y) = ncdf(-z1) - exp(2/(phi mu)) ncdf(-z2),
    z1 = (y/mu - 1)/sqrt(phi y), z2 = (y/mu + 1)/sqrt(phi y),
with mpmath.diff at 300 and 400 bits (a row is kept when the two agree to 2^-80 relative).
The route shares nothing with the library: no log Phi derivatives, no chain rule, no
logaddexp. File tests/data/reference/distrib_invgauss.csv, one row per value:
    upper (0 log F, 1 log(1 - F)), y, K, b, hi, lo, mu, phi
with b the number of derivatives in phi (K - b in mu); y, hi, lo, mu, phi in hexadecimal.
K = 0 rows are the values (b = 0).
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_invgauss.csv"

# (mu, phi): a moderate pair, a small phi mu where exp(2/(phi mu)) overflows in double,
# a large dispersion, and a pair with mu far from 1
PAIRS = [(1.0, 0.5), (0.4, 0.05), (2.0, 3.0), (0.3, 0.01), (0.2, 0.002), (1.0, 2.0)]
# y = mu t: both tails, the centre and the crossover where A and B are of the same size
TS = [0.05, 0.2, 0.5, 0.9, 0.97, 1.0, 1.03, 1.1, 1.3, 2.0, 4.0, 10.0, 40.0, 200.0]


# ---------------------------------------------------------------- SymPy checks

def small(expr, scale_expr, pts):
    """True when expr vanishes at the points pts to 1e-30 relative to the magnitude of scale_expr."""
    for pt in pts:
        pt = {k: sp.Rational(str(v)) for k, v in pt.items()}  # exact points, no binary round-off
        e = sp.N(expr.subs(pt), 50)
        m = sp.N(scale_expr.subs(pt), 50)
        if abs(e) > mpmath.mpf(10) ** -30 * (1 + abs(m)):
            return False
    return True


def sympy_checks():
    mu, phi, y = sp.symbols("mu phi y", positive=True)
    ell = -sp.Rational(1, 2) * sp.log(2 * sp.pi * phi * y**3) - (y - mu) ** 2 / (2 * phi * mu**2 * y)
    fact = sp.factorial

    def gd(a):  # closed form of d^a g / dmu^a, g = (y-mu)^2/(mu^2 y)
        if a == 0:
            return (y - mu) ** 2 / (mu**2 * y)
        return (-1) ** a * fact(a) * ((a + 1) * y - 2 * mu) / mu ** (a + 2)

    bad = 0
    # 1. parameter derivatives of the log-density, orders 1..6
    for K in range(1, 7):
        for b in range(K + 1):
            a = K - b
            closed = -sp.Rational(1, 2) * gd(a) * (-1) ** b * fact(b) / phi ** (b + 1)
            if b >= 1 and a == 0:
                closed += (-1) ** b * fact(b - 1) / (2 * phi**b)
            ref = sp.diff(ell, mu, a, phi, b) if a + b > 0 else ell
            pts = [{mu: m, phi: p, y: q} for m, p, q in [(0.7, 1.3, 0.4), (2.0, 0.2, 5.0), (1.1, 3.0, 0.9)]]
            if not small(ref - closed, ref, pts):
                bad += 1
                print("logpdf mismatch", K, a, b)
    # 2. mixed (y, theta) derivatives, J >= 1, J + K <= 6
    for J in range(1, 7):
        for K in range(0, 7 - J):
            for b in range(K + 1):
                a = K - b
                closed = 0
                if a == 0 and b == 0:
                    closed += -sp.Rational(3, 2) * (-1) ** (J - 1) * fact(J - 1) / y**J
                if J == 1:
                    closed += (-sp.Rational(1, 2) * (-1) ** a * fact(a + 1) / mu ** (a + 2)
                               * (-1) ** b * fact(b) / phi ** (b + 1))
                if a == 0:
                    closed += (-sp.Rational(1, 2) * (-1) ** J * fact(J) / y ** (J + 1)
                               * (-1) ** b * fact(b) / phi ** (b + 1))
                ref = sp.diff(ell, y, J, mu, a, phi, b) if (a + b) > 0 else sp.diff(ell, y, J)
                pts = [{mu: m, phi: p, y: q} for m, p, q in [(0.7, 1.3, 0.4), (2.0, 0.2, 5.0)]]
                if not small(ref - closed, ref, pts):
                    bad += 1
                    print("mixed mismatch", J, K, a, b)
    # 3. expected information: I = E[-d2 l] (the closed form is the Fisher information of the family)
    I_mumumu = 1 / (phi * mu**3)
    I_phiphi = 1 / (2 * phi**2)
    for K in range(1, 5):
        for b in range(K + 1):
            a = K - b
            closed_mm = (-1) ** K * sp.factorial(a + 2) / 2 * fact(b) / (mu ** (3 + a) * phi ** (1 + b))
            ref = sp.diff(I_mumumu, mu, a, phi, b)
            if not small(ref - closed_mm, ref, [{mu: 0.7, phi: 1.3}]):
                bad += 1
                print("info mumu mismatch", K, a, b)
        closed_pp = (-1) ** K * sp.factorial(K + 1) / 2 / phi ** (K + 2)
        if not small(sp.diff(I_phiphi, phi, K) - closed_pp, closed_pp, [{phi: 1.3}]):
            bad += 1
            print("info phiphi mismatch", K)
    # 4. the Riccati recurrence of r = e^x/(1 - e^x) and the derivatives of log(1 - e^x) = -R_(n-1)
    x = sp.symbols("x", negative=True)
    r = sp.exp(x) / (1 - sp.exp(x))
    R = [r]
    for k in range(0, 5):
        R.append(sp.expand(R[k] + sum(sp.binomial(k, i) * R[i] * R[k - i] for i in range(k + 1))))
    for n in range(1, 7):
        ref = sp.diff(sp.log(1 - sp.exp(x)), x, n)
        if not small(ref + sp.diff(r, x, n - 1), ref, [{x: -0.7}]):
            bad += 1
            print("nu mismatch", n)
        if not small(sp.diff(r, x, n - 1) - R[n - 1], sp.diff(r, x, n - 1), [{x: -0.7}]):
            bad += 1
            print("Riccati mismatch", n)
    # 5. the softplus recurrence: d^n log(1 + e^x) = sigma^(n-1) with sigma the logistic function
    sig = 1 / (1 + sp.exp(-x))
    for n in range(1, 7):
        if not small(sp.diff(sp.log(1 + sp.exp(x)), x, n) - sp.diff(sig, x, n - 1), sp.diff(sig, x, n - 1), [{x: 0.37}]):
            bad += 1
            print("softplus mismatch", n)
    print(f"sympy checks: {'all agree' if bad == 0 else str(bad) + ' mismatches'}")
    if bad:
        sys.exit(1)


# ---------------------------------------------------------------- mpmath references

def log_value(upper, y, mu, phi):
    """log F (upper = 0) or log(1 - F) (upper = 1) from the definition."""
    z1 = (y / mu - 1) / mpmath.sqrt(phi * y)
    z2 = (y / mu + 1) / mpmath.sqrt(phi * y)
    E = 2 / (phi * mu)
    if upper:
        return mpmath.log(mpmath.ncdf(-z1) - mpmath.exp(E) * mpmath.ncdf(-z2))
    return mpmath.log(mpmath.ncdf(z1) + mpmath.exp(E) * mpmath.ncdf(-z2))


def value(upper, y, K, b, mu0, phi0, prec):
    with mpmath.workprec(prec):
        f = lambda m, p: log_value(upper, mpmath.mpf(y), m, p)
        if K == 0:
            return f(mpmath.mpf(mu0), mpmath.mpf(phi0))
        return mpmath.diff(f, (mpmath.mpf(mu0), mpmath.mpf(phi0)), (K - b, b))


def task(pair):
    mu0, phi0 = pair
    lines, dropped = [], 0
    for upper in (0, 1):
        for t in TS:
            y = mu0 * t
            yr = float(y)  # the grid point is the double
            ym = mpmath.mpf(yr)
            for K in range(0, 7):
                for b in range(K + 1):
                    a, c = value(upper, ym, K, b, mu0, phi0, 300), value(upper, ym, K, b, mu0, phi0, 400)
                    if c == 0 or not mpmath.isfinite(c) or abs(a - c) > mpmath.mpf(2) ** -80 * abs(c):
                        dropped += 1
                        continue
                    hi = float(c)
                    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                        dropped += 1
                        continue
                    lo = float(c - mpmath.mpf(hi))
                    lines.append(f"{upper},{yr.hex()},{K},{b},{hi.hex()},{lo.hex()},{float(mu0).hex()},{float(phi0).hex()}\n")
                    if K == 0:
                        break
    return lines, dropped


def logpdf_value(y, mu, phi):
    """l(mu, phi; y) from the definition: log of the inverse Gaussian density."""
    return -mpmath.log(2 * mpmath.pi * phi * y**3) / 2 - (y - mu) ** 2 / (2 * phi * mu**2 * y)


def logpdf_task(pair):
    mu0, phi0 = pair
    lines, dropped = [], 0
    for t in TS:
        yr = float(mu0 * t)
        for J in range(0, 7):
            for K in range(0, 7 - J):
                for b in range(K + 1):
                    vals = []
                    for prec in (300, 400):
                        with mpmath.workprec(prec):
                            f = lambda yy, m, p: logpdf_value(yy, m, p)
                            if J + K == 0:
                                vals.append(f(mpmath.mpf(yr), mpmath.mpf(mu0), mpmath.mpf(phi0)))
                            else:
                                vals.append(mpmath.diff(f, (mpmath.mpf(yr), mpmath.mpf(mu0), mpmath.mpf(phi0)),
                                                        (J, K - b, b)))
                    a, c = vals
                    if c == 0 or not mpmath.isfinite(c) or abs(a - c) > mpmath.mpf(2) ** -80 * abs(c):
                        dropped += 1
                        continue
                    hi = float(c)
                    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                        dropped += 1
                        continue
                    lo = float(c - mpmath.mpf(hi))
                    lines.append(f"{J},{K},{b},{yr.hex()},{hi.hex()},{lo.hex()},{float(mu0).hex()},{float(phi0).hex()}\n")
    return lines, dropped


LOGPDF_OUT = OUT.with_name("distrib_invgauss_logpdf.csv")


def logpdf_references():
    """mpmath references of the log-density and its derivatives in y and in (mu, phi), J + K <= 6."""
    with Pool(4) as pool:
        res = pool.map(logpdf_task, PAIRS)
    n = dropped = 0
    with open(LOGPDF_OUT, "w", newline="\n") as f:
        f.write("J,K,b,y,hi,lo,mu,phi\n")
        for lines, d in res:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{LOGPDF_OUT.name}: {n} rows ({dropped} dropped)")


# Extra points where the delta series is engaged: delta (1 + |a|) with a = z1/sqrt2 and delta = sqrt(2/(phi y)) between
# 1/16 and 2. Each pair (mu, phi) is scanned on a log grid of y and each target crossed is bisected.
EXTRA_PAIRS = [(1.0, 2.0), (1.0, 8.0), (4.0, 4.0), (1.0, 1.0)]


def delta_product(y, mu, phi):
    z1 = (y / mu - 1) / math.sqrt(phi * y)
    return math.sqrt(2 / (phi * y)) * (1 + abs(z1 / math.sqrt(2)))


def extra_ys():
    """One point per (pair, band): the first y on the log grid whose delta product lies in the band."""
    bands = [(1 / 16, 1 / 8), (1 / 8, 1 / 4), (1 / 4, 1 / 2), (1 / 2, 1.0), (1.0, 2.0)]
    ys = []
    for mu, phi in EXTRA_PAIRS:
        lo0, hi0 = mu * 1.001, mu * 1e7
        grid = [lo0 * (hi0 / lo0) ** (k / 2999) for k in range(3000)]
        for blo, bhi in bands:
            for y in grid:
                v = delta_product(y, mu, phi)
                if blo <= v < bhi:
                    ys.append((mu, phi, y, v))
                    break
    return ys


def extra_task(point):
    mu0, phi0, y0, _tgt = point
    lines = []
    for upper in (0, 1):
        for K in range(0, 7):
            for b in range(K + 1):
                vals = []
                for prec in (300, 400):
                    with mpmath.workprec(prec):
                        vals.append(value(upper, mpmath.mpf(y0), K, b, mu0, phi0, prec))
                a, c = vals
                if c == 0 or not mpmath.isfinite(c) or abs(a - c) > mpmath.mpf(2) ** -80 * abs(c):
                    continue
                hi = float(c)
                if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
                    continue
                lo = float(c - mpmath.mpf(hi))
                lines.append(f"{upper},{float(y0).hex()},{K},{b},{hi.hex()},{lo.hex()},{float(mu0).hex()},{float(phi0).hex()}\n")
                if K == 0:
                    break
    return lines


def extra_references():
    """Adds rows to distrib_invgauss.csv at the extra points (appended; the existing rows are kept)."""
    points = extra_ys()
    print("extra points:", [(m, f, round(y, 6), t) for m, f, y, t in points])
    with Pool(4) as pool:
        res = pool.map(extra_task, points)
    with open(OUT, "a", newline='\n') as f:
        for lines in res:
            f.writelines(lines)
    print(f"appended {sum(len(r) for r in res)} rows to {OUT.name}")


def main():
    sympy_checks()
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        return extra_references()
    if len(sys.argv) > 1 and sys.argv[1] == "logpdf":
        return logpdf_references()
    with Pool(4) as pool:
        res = pool.map(task, PAIRS)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("upper,y,K,b,hi,lo,mu,phi\n")
        for lines, d in res:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
