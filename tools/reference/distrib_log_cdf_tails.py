"""Reference values of the derivatives of log F and log(1 - F) in (mu, sigma),
orders 1 to 6, for gaussian1 and laplace in their tails: the points where the
derivatives in z, carried to (mu, sigma) as z-polynomials, cancel.

Usage (from the repository root):  python -I tools/reference/distrib_log_cdf_tails.py

File tests/data/reference/distrib_log_cdf_tails.csv, with the columns of
distrib_log_cdf.csv (written by distrib_log_cdf.py, whose functions are used
here): dist (0 gaussian1, 1 laplace), upper (0 log F, 1 log(1 - F)), y, K, b,
hi, lo, mu, phi (= sigma), with b the number of derivatives in sigma; y, hi,
lo, mu and phi in hexadecimal. mu = 0.4 and sigma = 1.3.

Points, in z = (y - mu)/sigma: for gaussian1 the lower tail of log F
(z = -1.5 .. -35) and the upper tail of log(1 - F) (z = 1.5 .. 35), where
log Phi(z) is close to -z^2/2; for laplace the side of each function where it
is log(1 - e^(-|z|)/2) (log(1 - F) for z < 0, log F for z > 0), |z| = 0.05 .. 30.
The method is that of distrib_log_cdf.py: mpmath.diff of the closed forms at
300 and 400 bits, a row kept when the two agree to 2^-80 relative.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distrib_log_cdf import AGREE, MU, SIGMA, value  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_log_cdf_tails.csv"

GAUSS_Z = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0, 17.0, 22.0, 28.0, 35.0]
LAPLACE_Z = [0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.77, 4.9, 6.0, 8.0, 11.0, 15.0, 20.0, 30.0]


def task(args):
    dist, upper = args
    zs = GAUSS_Z if dist == 0 else LAPLACE_Z
    # gaussian1: log F in the lower tail, log(1 - F) in the upper; laplace: log(1 - F) for
    # z < 0, log F for z > 0
    if dist == 0:
        sgn = -1.0 if upper == 0 else 1.0
    else:
        sgn = -1.0 if upper == 1 else 1.0
    lines, dropped = [], 0
    for zz in zs:
        y = MU + SIGMA * sgn * zz
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
                lines.append(f"{dist},{upper},{y.hex()},{K},{b},{hi.hex()},{lo.hex()},{MU.hex()},{SIGMA.hex()}\n")
    return lines, dropped


def main():
    jobs = [(d, u) for d in (0, 1) for u in (0, 1)]
    with Pool(4) as pool:
        res = pool.map(task, jobs)
    n = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("dist,upper,y,K,b,hi,lo,mu,phi\n")
        for lines, d in res:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{OUT.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
