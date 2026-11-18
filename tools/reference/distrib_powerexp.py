"""Reference values for powerexp_distrib (include/laplace/distrib/powerexp.hpp).

Usage (from the repository root):
    python -I tools/reference/distrib_powerexp.py logpdf [workers]
    python -I tools/reference/distrib_powerexp.py info [workers]

The law, with mu real, sigma > 0 the standard deviation and p > 0 the shape:

    c(p) = sqrt(Gamma(1/p) / Gamma(3/p)),   s = sigma c(p),   z = (y - mu) / s,
    f(y) = p / (2 s Gamma(1/p)) exp(-|z|^p),

so that Var Y = sigma^2 for every p.

Files written in tests/data/reference (every number in hexadecimal, hi + lo the two halves of the
high-precision value):
    distrib_powerexp.csv        y, K, a, b, c, hi, lo, mu, sigma, p   d^K log f, (a, b, c)
                                derivatives in (mu, sigma, p), K = a + b + c, K = 0 the value
    distrib_powerexp_info.csv   i, j, hi, lo, sigma, p                I_ij(mu, sigma, p) at mu = 0

Method. Nothing of the library's route is used: the derivatives are SymPy's own symbolic
derivatives of the definition above, with loggamma left in place (SymPy differentiates it into the
polygamma functions), and each one is evaluated twice through mpmath (sp.lambdify), at 50 and at 80
digits. Evaluating the symbolic expressions with SymPy's own adaptive evalf instead gives the same
numbers to 1e-50 and costs 52 seconds a point against 9. A row is kept when the two agree to 1e-30 of the largest component of its order at that
point, and is dropped otherwise. The library instead builds the same quantities by one Faa di Bruno
composition through v = a/p and the complete Bell polynomials of w_j(p), so the two routes share no
step; tools/reference/powerexp_forms.py checks the library's assembly against SymPy as well.

|z|^p is not differentiable symbolically through an absolute value, so each point is built with the
sign of y - mu fixed and the expression written with (y - mu)^p or (mu - y)^p accordingly; the
points are chosen on both sides of the location. They have mu dyadic, sigma a power of two and
zeta = (y - mu)/sigma a double of 20 significant bits, so that the library and the reference see
exactly the same zeta in double arithmetic. The location itself (zeta = 0), where several
derivatives are infinite, is not in the file: the test checks those limits directly.

The expected information is the closed form of the header evaluated by mpmath at 60 digits, which
tools/reference/powerexp_forms.py has already checked against the quadrature of the score products.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"

# the shape: inside the cusp region, on both sides of the integers 1, 2 and 3, and far above
PS = [0.4, 0.75, 0.95, 1.0, 1.05, 1.5, 2.0, 2.5, 3.0, 4.0, 7.5, 30.0]
# zeta = (y - mu)/sigma: near the location, of order one on both sides, and in the tails
ZS_MAIN = [0.03, 0.3, 1.0, -1.0, 2.5, -7.0, 30.0]
ZS_EXTREME = [0.03, -1.0, 2.5, -7.0]
COMBOS = [(0.5, 1.0, ZS_MAIN), (-2.0, 2.0 ** -10, ZS_EXTREME), (1.0, 2.0 ** 10, ZS_EXTREME)]

MU, SG, P, Y = sp.symbols("mu sigma p y", real=True)


def short(v, bits=20):
    """v rounded to `bits` significant bits, so that (y - mu)/sigma is exact in double arithmetic"""
    if v == 0:
        return 0.0
    m, e = math.frexp(v)
    return math.ldexp(round(m * 2 ** bits) / 2 ** bits, e)


def hexs(v):
    return float(v).hex()


def comps(K):
    """the sorted multi-indices of order K as (a, b, c), in the packing order of sym_tensor<3, K>"""
    return [(K - b - c, b, c) for b in range(K + 1) for c in range(K - b + 1)]


# ------------------------------------------------------------------ the symbolic derivatives

def build_exprs(positive):
    """l and every derivative d^K l / dmu^a dsigma^b dp^c, K <= 6, with the sign of y - mu fixed"""
    r = (Y - MU) if positive else (MU - Y)
    # l = log p - log 2 - log sigma - log c(p) - lgamma(1/p) - (r / (sigma c(p)))^p
    logc = (sp.loggamma(1 / P) - sp.loggamma(3 / P)) / 2
    l = sp.log(P) - sp.log(2) - sp.log(SG) - logc - sp.loggamma(1 / P) - (r / (SG * sp.exp(logc))) ** P
    cache = {(0, 0, 0): l}
    out = {}
    for K in range(0, 7):
        for (a, b, c) in comps(K):
            if (a, b, c) not in cache:
                if a > 0:
                    e = sp.diff(cache[(a - 1, b, c)], MU)
                elif b > 0:
                    e = sp.diff(cache[(a, b - 1, c)], SG)
                else:
                    e = sp.diff(cache[(a, b, c - 1)], P)
                cache[(a, b, c)] = e
            out[(K, a, b, c)] = cache[(a, b, c)]
    return out


EXPRS = {}


def exprs_for(positive):
    """the derivative expressions, built once per process"""
    if positive not in EXPRS:
        EXPRS[positive] = build_exprs(positive)
    return EXPRS[positive]


FNS = {}


def fns_for(positive):
    """the same expressions as mpmath functions of (mu, sigma, p, y), built once per process"""
    if positive not in FNS:
        FNS[positive] = {k: sp.lambdify((MU, SG, P, Y), v, modules="mpmath")
                         for k, v in exprs_for(positive).items()}
    return FNS[positive]


def eval_all(fns, mu, sg, pv, y, dps):
    mpmath.mp.dps = dps
    args = (mpmath.mpf(mu), mpmath.mpf(sg), mpmath.mpf(pv), mpmath.mpf(y))
    return {k: f(*args) for k, f in fns.items()}


def one_point(arg):
    """every component at one (mu, sigma, p, zeta); returns the rows and how many were dropped"""
    mu, sg, pv, zv = arg
    y = mu + sg * zv
    fns = fns_for(zv > 0)
    hi_prec = eval_all(fns, mu, sg, pv, y, 80)
    lo_prec = eval_all(fns, mu, sg, pv, y, 50)
    mpmath.mp.dps = 80
    rows, dropped = [], 0
    for K in range(0, 7):
        scale = max([abs(hi_prec[(K, a, b, c)]) for (a, b, c) in comps(K)] + [mpmath.mpf(0)])
        if scale == 0:
            scale = mpmath.mpf(1)
        for (a, b, c) in comps(K):
            v = hi_prec[(K, a, b, c)]
            if abs(v - lo_prec[(K, a, b, c)]) > scale * mpmath.mpf("1e-30"):
                dropped += 1
                continue
            hi = float(v)
            lo = float(v - mpmath.mpf(hi))
            if not (math.isfinite(hi) and math.isfinite(lo)):
                dropped += 1
                continue
            rows.append((hexs(y), K, a, b, c, hexs(hi), hexs(lo), hexs(mu), hexs(sg), hexs(pv)))
    return rows, dropped


def write_logpdf(workers):
    args = []
    for (mu, sg, zs) in COMBOS:
        for pv in PS:
            for zv in zs:
                args.append((mu, sg, pv, short(zv)))
    print(f"{len(args)} points, {workers} workers")
    rows, dropped = [], 0
    if workers > 1:
        with Pool(workers) as pool:
            for i, (r, d) in enumerate(pool.imap(one_point, args)):
                rows += r
                dropped += d
                if (i + 1) % 20 == 0:
                    print(f"  {i + 1}/{len(args)} points, {len(rows)} rows")
    else:
        for i, a in enumerate(args):
            r, d = one_point(a)
            rows += r
            dropped += d
    path = OUT / "distrib_powerexp.csv"
    with open(path, "w", newline="\n") as f:
        f.write("y,K,a,b,c,hi,lo,mu,sigma,p\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"{path}: {len(rows)} rows, {dropped} dropped")


# ------------------------------------------------------------------ the expected information

def info_closed(sigma, p):
    """I(mu, sigma, p) as a 3x3 list of mpf, the closed form of the header (mpmath, 60 digits)"""
    a = 1 / p
    psi, psi1 = mpmath.digamma, lambda x: mpmath.polygamma(1, x)
    logc = (mpmath.loggamma(a) - mpmath.loggamma(3 * a)) / 2
    s = sigma * mpmath.exp(logc)
    kappa = (3 * psi(3 * a) - psi(a)) / (2 * p ** 2)
    i_mumu = (mpmath.inf if p <= mpmath.mpf(1) / 2
              else p ** 2 * mpmath.exp(mpmath.loggamma(2 - a) - mpmath.loggamma(a)) / s ** 2)
    nat_pp = -a ** 4 * psi(1 + a) ** 2 + a ** 3 * (1 + a) * (psi1(2 + a) + psi(2 + a) ** 2)
    i_sgsg = p / sigma ** 2
    i_sgp = (p * kappa - (1 + psi(1 + a)) / p) / sigma
    i_pp = nat_pp - 2 * kappa * (1 + psi(1 + a)) / p + p * kappa ** 2
    return [[i_mumu, mpmath.mpf(0), mpmath.mpf(0)], [mpmath.mpf(0), i_sgsg, i_sgp],
            [mpmath.mpf(0), i_sgp, i_pp]]


def write_info(_workers):
    mpmath.mp.dps = 60
    rows = []
    for sg in [0.25, 1.0, 4.0]:
        for pv in [0.6, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.5, 30.0]:
            I = info_closed(mpmath.mpf(sg), mpmath.mpf(pv))
            for i in range(3):
                for j in range(i, 3):
                    v = I[i][j]
                    if v == mpmath.inf:
                        continue
                    hi = float(v)
                    lo = float(v - mpmath.mpf(hi))
                    rows.append((i, j, hexs(hi), hexs(lo), hexs(sg), hexs(pv)))
    path = OUT / "distrib_powerexp_info.csv"
    with open(path, "w", newline="\n") as f:
        f.write("i,j,hi,lo,sigma,p\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"{path}: {len(rows)} rows")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "logpdf"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if what == "logpdf":
        write_logpdf(workers)
    elif what == "info":
        write_info(workers)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
