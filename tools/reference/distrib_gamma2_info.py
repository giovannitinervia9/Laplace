"""Reference values of the expected information of gamma2 (mean mu, variance sigma2) by quadrature
at 40 digits, for shapes from 1e-4 to 1e8, where the double-precision quadrature of the C++ tests
(check_common's score_variance_continuous) does not resolve the density at y -> 0.

Usage (from the repository root):  python -I tools/reference/distrib_gamma2_info.py

File tests/data/reference/distrib_gamma2_info.csv, one row per component:
    mu, sigma2, a, b, value      (a <= b in {0 mu, 1 sigma2}; mu, sigma2 and value in hexadecimal)
value = E[S_a S_b] with S_a = d l / d theta_a the score of the log-density l(mu, sigma2; y), which
equals the expected information (the variance of the score). The scores are the SymPy closed forms of
distrib_gamma2.py (direct_logpdf_forms, orders 1) with delta'(s) = mpmath.diff of the definition of
Stirling's error; the density is the definition (gamma with shape s = mu^2/sigma2 and scale
sigma2/mu). The integral is mpmath.quad (tanh-sinh) over the pieces 0 to mu, to 8 mu, to 100 mu, to inf,
which resolve the singular behaviour of the shape-1e8 and shape-50 densities near their modes.
"""
import importlib.util
import sys
from pathlib import Path

import mpmath
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "tests" / "data" / "reference" / "distrib_gamma2_info.csv"
# shapes 1, 50 and 1e8. Shape 1e-4 is left out: half of its mass lies below y = 1e-300, which no quadrature in y
# resolves (the closed form tends to 4/mu^2 there, and the C++ tests check the closed form at that shape by differences)
POINTS = [(0.4, 0.16), (2.5, 0.125), (1.0e4, 1.0)]


def load_direct():
    spec = importlib.util.spec_from_file_location("dg", HERE / "distrib_gamma2.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    return dg


def main():
    dg = load_direct()
    forms = dg.direct_logpdf_forms(1)
    mpmath.mp.dps = 40
    lines = []
    for muv, s2v in POINTS:
        mu_, s2_ = mpmath.mpf(muv), mpmath.mpf(s2v)
        s = mu_ * mu_ / s2_
        d1 = mpmath.diff(dg.delta_mp, s, 1)
        th = s2_ / mu_
        # the score in mu is forms[(1, 0)], the score in sigma2 is forms[(0, 1)]
        exprs = {0: sp.lambdify([dg.mu, dg.sig2, dg.y, dg.D[1]], forms[(1, 0)], "mpmath"),
                 1: sp.lambdify([dg.mu, dg.sig2, dg.y, dg.D[1]], forms[(0, 1)], "mpmath")}

        def density(yv):
            return mpmath.exp((s - 1) * mpmath.log(yv) - yv / th - s * mpmath.log(th) - mpmath.loggamma(s))

        def integrand(yv, a, b):
            sa = exprs[a](mu_, s2_, yv, d1)
            sb = exprs[b](mu_, s2_, yv, d1)
            return sa * sb * density(yv)

        cuts = [mpmath.mpf(0), mu_, 8 * mu_, 100 * mu_, mpmath.inf]
        for a, b in [(0, 0), (0, 1), (1, 1)]:
            val = sum(mpmath.quad(lambda yv: integrand(yv, a, b), [cuts[i], cuts[i + 1]]) for i in range(4))
            lines.append(f"{float(mu_).hex()},{float(s2_).hex()},{a},{b},{float(val).hex()}\n")
            print(muv, s2v, a, b, mpmath.nstr(val, 15))
    with open(OUT, "w", newline="\n") as f:
        f.write("mu,sigma2,a,b,value\n")
        f.writelines(lines)
    print(f"{OUT.name}: {len(lines)} rows")


if __name__ == "__main__":
    sys.exit(main())
