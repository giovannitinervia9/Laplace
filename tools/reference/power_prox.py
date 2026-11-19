"""Reference values of the proximal operator of a|x|^p, computed with mpmath.

    x*(u, a, p) = argmin_x  a |x|^p + (x - u)^2 / 2,   a > 0, p > 0,

the scalar problem behind the proxes of powerexp_distrib and of the isotropic
mvpowerexp1/2 (note/penalties.md section 5g; the library's routine is
include/laplace/distrib/detail/power_prox.hpp).

Usage (from the repository root):  python -I tools/reference/power_prox.py

File (format as in dawson.py: the arguments u, a, p, then hi, lo in
hexadecimal): power_prox.csv.

Method. A nonzero minimizer has the sign of u and solves
g(x) = x + a p x^(p-1) - |u| = 0 on (0, |u|]. The root is bracketed and
bisected in log x (400 halvings), not by Newton, which is the library's route:
for p > 1, g is increasing and the bracket is (|u| e^-k, |u|] widened until
g < 0 at its left end; for p < 1 the objective may have two local minima, 0
and the larger root, on the branch where g increases, x > x_m =
(a p (1 - p))^(1/(2-p)); the bracket is (x_m, |u|] when g(x_m) < 0 (otherwise
0 is the only minimizer) and the objective at the root is compared with its
value at 0, 0 winning a tie. Everything is computed at 200 and at 260 bits and
a point is kept only if the two agree to 2^-120 relative.

Points: p in a list from 0.1 to 50 (with p near 1 and near 2 on both sides,
and 1/2, 2/3, the two cases with closed forms in the literature); for each p,
(u, a) log-uniform in [1e-8, 1e8] with a random sign of u; and for p < 1 the
threshold tau = x_b + a p x_b^(p-1), x_b = (2a(1-p))^(1/(2-p)), approached as
|u| = tau (1 + e) with e = +-1e-6, +-1e-10.
"""
import random
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261119
PRECS = (200, 260)
P_LIST = (0.1, 0.3, 0.5, 2.0 / 3.0, 0.75, 0.9, 0.99, 1.01, 1.05, 1.2, 1.5, 1.9, 1.99,
          2.01, 2.1, 2.5, 3.0, 4.0, 6.0, 10.0, 50.0)
N_RANDOM = 120


def prox(u, a, p):
    """argmin_x a|x|^p + (x - u)^2/2 at the current precision."""
    u, a, p = mpmath.mpf(u), mpmath.mpf(a), mpmath.mpf(p)
    U = abs(u)
    if U == 0:
        return mpmath.mpf(0)

    def g(lx):
        return mpmath.exp(lx) + a * p * mpmath.exp((p - 1) * lx) - U

    hi = mpmath.log(U) + mpmath.mpf(2) ** -60  # g > 0 at |u| (a > 0); the margin absorbs log/exp rounding
    if p < 1:
        xm = (a * p * (1 - p)) ** (1 / (2 - p))
        if xm >= U:
            return mpmath.mpf(0)
        lo = mpmath.log(xm)
        if g(lo) >= 0:
            return mpmath.mpf(0)
    else:
        lo = hi - 10
        while g(lo) > 0:
            lo -= 10
    for _ in range(400):
        m = (lo + hi) / 2
        if g(m) > 0:
            hi = m
        else:
            lo = m
    r = mpmath.exp((lo + hi) / 2)
    if p < 1 and a * r**p + (r - U) ** 2 / 2 >= U**2 / 2:
        return mpmath.mpf(0)
    return mpmath.sign(u) * r


def checked(u, a, p):
    vals = []
    for prec in PRECS:
        with mpmath.workprec(prec):
            vals.append(prox(u, a, p))
    x0, x1 = vals
    if x0 == 0 or x1 == 0:
        return x1 if x0 == x1 else None
    with mpmath.workprec(PRECS[1]):
        if abs(x0 - x1) > abs(x1) * mpmath.mpf(2) ** -120:
            return None
    return x1


def hi_lo(x):
    with mpmath.workprec(PRECS[1]):
        hi = float(x)
        lo = float(x - hi)
    return hi, lo


def main():
    rng = random.Random(SEED)
    rows = []
    for p in P_LIST:
        pts = []
        for _ in range(N_RANDOM):
            u = rng.choice((-1.0, 1.0)) * 10 ** rng.uniform(-8, 8)
            a = 10 ** rng.uniform(-8, 8)
            pts.append((u, a))
        if p < 1:
            for _ in range(10):
                a = 10 ** rng.uniform(-4, 4)
                with mpmath.workprec(PRECS[1]):
                    xb = (2 * mpmath.mpf(a) * (1 - mpmath.mpf(p))) ** (1 / (2 - mpmath.mpf(p)))
                    tau = xb + a * p * xb ** (p - 1)
                for e in (1e-6, -1e-6, 1e-10, -1e-10):
                    pts.append((rng.choice((-1.0, 1.0)) * float(tau * (1 + e)), a))
        for u, a in pts:
            x = checked(u, a, p)
            if x is None:
                continue
            hi, lo = hi_lo(x)
            if hi != 0 and abs(hi) < 2.2250738585072014e-308:
                continue  # subnormal reference: dropped
            rows.append((u, a, p, hi, lo))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "power_prox.csv", "w", newline="\n") as f:
        f.write("u,a,p,hi,lo\n")
        for r in rows:
            f.write(",".join(float(v).hex() for v in r) + "\n")
    print(f"power_prox.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
