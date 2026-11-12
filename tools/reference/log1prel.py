"""Reference values of log1prel(x) = log(1 + x)/x and of its derivatives to order 6,
computed with mpmath.

Usage (from the repository root):  python -I tools/reference/log1prel.py

Files (format as in bessel_k.py: the argument x, then hi, lo):
    log1prel.csv        x   psi(x)
    log1prel_d{k}.csv   x   d^k/dx^k psi(x), k = 1..6
where psi(x) = log(1 + x)/x, psi(0) = 1, and hi + lo is the exact value as a
double-double (hex). A row is kept only if its exact value is a finite normal
double and the two precisions below agree.

Method. From psi(x) = int_0^1 dt/(1 + x t) and
psi^(k)(x) = (-1)^k k! int_0^1 t^k (1 + x t)^(-(k+1)) dt, the substitution
s = 1 + x t, a = 1 + x and the expansion of (s - 1)^k give the closed form
psi^(k)(x) = (-1)^k k! x^(-(k+1)) B_k(a), with
    B_k(a) = log a + sum_(j=0..k-1) C(k, j) (-1)^(k-j) (a^(j-k) - 1) / (j - k).
This is independent of the library's series and recursions. It cancels when
|x| is small (B_k is of order x^(k+1)), so the working precision is raised by
(k + 1) log2(1/|x|) + 64 bits. Every row is computed at P = 200 and P + 100
bits, and kept only if they agree to 2^-100 relative. The closed form is
checked against mpmath.quad of the integral in `--check`.

The doubles near -1 have 1 + x >= 2^-53 (the spacing of doubles in [-1, -1/2]
is 2^-53), so the sample near -1 is 1 + x log-uniform on [2^-53, 1]; smaller
values of 1 + x are not representable as x.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261112
NMAX = 6
P1 = 200
P2 = 300
AGREE = mpmath.mpf(2) ** -100
TINY = 2.2250738585072014e-308  # the smallest positive normal double

# The switches of the implementation (docs/numerics.md): the series for
# |x| < 1/4, the quadrature for -4/5 < x <= 6, the recursion elsewhere. Points
# are placed on both sides of each.
THRESHOLDS = [-0.8, -0.25, 0.25, 6.0]


def psi_all(x, prec):
    """[psi(x), psi'(x), ..., psi^(6)(x)] as mpmath numbers at `prec` bits,
    x a double with x != 0 and x > -1, by the closed form above."""
    if x == 0.0:
        out = [mpmath.mpf(1)]
        for k in range(1, NMAX + 1):
            out.append(mpmath.mpf((-1) ** k * math.factorial(k)) / (k + 1))
        return out
    extra = 64 + (NMAX + 1) * max(0, -math.frexp(abs(x))[1]) + 8
    with mpmath.workprec(prec + extra):
        xm = mpmath.mpf(x)
        a = 1 + xm
        la = mpmath.log(a)
        out = []
        for k in range(NMAX + 1):
            b = la
            for j in range(k):
                b += mpmath.binomial(k, j) * (-1) ** (k - j) * (a ** (j - k) - 1) / (j - k)
            val = (-1) ** k * mpmath.factorial(k) * b / xm ** (k + 1)
            out.append(+val)  # rounded to the working precision
        return out


def ref_row(x):
    """For each order, (hi, lo) or None when the two precisions disagree or the
    exact value is not a finite normal double."""
    u = psi_all(x, P1)
    v = psi_all(x, P2)
    res = []
    for k in range(NMAX + 1):
        a, b = u[k], v[k]
        ok = abs(a - b) <= AGREE * max(abs(b), mpmath.mpf(2) ** -1070)
        hi = float(b)
        if not ok or not math.isfinite(hi) or abs(hi) < TINY:
            res.append(None)
            continue
        lo = float(b - mpmath.mpf(hi))
        res.append((hi, lo))
    return res


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def sample(rng):
    xs = [0.0]
    for _ in range(900):  # |x| log-uniform from 2^-60 to 1, both signs
        m = log_uniform(rng, -61, -1)
        xs.append(m if rng.random() < 0.5 else -m)
    for _ in range(600):  # 1 + x log-uniform from 2^-53 to 1 (the doubles near -1)
        d = math.ldexp(1.0, -53 + int(53 * rng.random()))
        x = -1.0 + d
        if x > -1.0:
            xs.append(x)
    for _ in range(700):  # uniform on [-0.99, 30]
        xs.append(-0.99 + 30.99 * rng.random())
    for _ in range(400):  # |x| log-uniform from 1 to 1e300, positive
        xs.append(math.ldexp(1.0 + rng.random(), rng.randint(0, 996)))
    for t in THRESHOLDS:  # around each switch, both sides
        for e in range(2, 22, 2):
            for s in (1.0 - 2.0 ** -e, 1.0 + 2.0 ** -e):
                x = t * s
                if x > -1.0:
                    xs.append(x)
    return sorted(set(xs))


def check():
    """the closed form against mpmath.quad of the integral, at some points"""
    rng = random.Random(7)
    worst = 0.0
    for x in [0.3, -0.7, 2.5, -0.999, 1e-5, 40.0] + [rng.uniform(-0.99, 5.0) for _ in range(10)]:
        mine = psi_all(x, 200)
        with mpmath.workprec(200):
            xm = mpmath.mpf(x)
            for k in range(NMAX + 1):
                q = mpmath.quad(lambda t: t ** k / (1 + xm * t) ** (k + 1), [0, 1])
                q = (-1) ** k * mpmath.factorial(k) * q
                e = abs(q - mine[k]) / abs(q)
                worst = max(worst, float(e))
    print(f"closed form against mpmath.quad: worst relative difference {worst:.3g}")


def main():
    if "--check" in sys.argv:
        check()
        return
    rng = random.Random(SEED)
    xs = sample(rng)
    rows = [ref_row(x) for x in xs]
    for k in range(NMAX + 1):
        name = "log1prel" if k == 0 else f"log1prel_d{k}"
        path = OUT / f"{name}.csv"
        n = 0
        with open(path, "w", newline="\n") as f:
            f.write("x,hi,lo\n")
            for x, r in zip(xs, rows):
                if r[k] is None:
                    continue
                hi, lo = r[k]
                f.write(f"{x.hex()},{hi.hex()},{lo.hex()}\n")
                n += 1
        print(f"{path.name}: {n} rows of {len(xs)} points")


if __name__ == "__main__":
    main()
