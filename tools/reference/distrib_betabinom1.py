"""SymPy check of the closed forms of betabinom1 and mpmath references of its log-mass, log F
and log(1 - F) derivatives in (mu, sigma) to total order 6.

Usage (from the repository root):  python -I tools/reference/distrib_betabinom1.py

The beta-binomial of mean proportion mu in (0, 1) and dispersion sigma > 0 with a known size n
has, with a = mu / sigma and b = (1 - mu) / sigma,

    l(mu, sigma; y) = log C(n, y) + log B(y + a, n - y + b) - log B(a, b),  y = 0, ..., n.

Part 1, SymPy (printed; the script exits with status 1 on any disagreement), checks the forms
the header uses:
  - the shift identity: both shifts y and n - y are integers, so the two log-betas never need
    to be formed and the whole difference is, exactly,

        l = log C(n, y) + sum_(i<y) log(mu + i sigma) + sum_(i<n-y) log(1 - mu + i sigma)
                        - sum_(i<n) log(1 + i sigma),

    in which every term is regular at sigma = 0 (the logarithms of sigma cancel, y + (n - y)
    - n = 0) and the value at sigma = 0 is the binomial's;
  - the derivative of total order K = p + q, p in mu and q in sigma, of the same sums,

        d^K l = (-1)^(K-1) (K-1)! [ sum_(i<y) i^q / (mu + i sigma)^K
                                    + (-1)^p sum_(i<n-y) i^q / (1 - mu + i sigma)^K
                                    - [p = 0] sum_(i<n) i^K / (1 + i sigma)^K ],

    each sum of terms of one sign;
  - Loader's form of log C(n, y) with Stirling's error.

Part 2, mpmath (tests/data/reference/distrib_betabinom1.csv), one row per value:
    kind (0 log p, 1 log F, 2 log(1 - F)), y, mu, sigma, size, K, b, hi, lo
with b the number of derivatives in sigma (K - b in mu) and y, mu, sigma, size, hi, lo as
hexadecimal doubles. The mass rows are mpmath evaluations of the symbolic derivatives of the
DEFINITION above (the two log-betas as loggamma, never the sums of part 1), so the two routes
share nothing; the cumulative rows are mpmath.diff of the mass summed over the support.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_betabinom1.csv"
MAXK = 6
AGREE = mpmath.mpf(2) ** -80

# (mu, sigma, size). sigma -> 0 is the binomial limit, sigma large puts the mass on 0 and n;
# mu near 0 and near 1; the size from 1 to 1e4.
PARAMS = [
    (0.5, 0.5, 10.0),
    (0.3, 2.0, 5.0),
    (0.5, 1e-6, 12.0),
    (0.5, 1e-3, 40.0),
    (0.02, 0.25, 50.0),
    (0.98, 0.25, 50.0),
    (0.5, 1e3, 7.0),
    (0.4, 0.05, 1000.0),
    (0.5, 0.01, 10000.0),
    (0.7, 1.0, 1.0),
    # the region of the route of bounded cost (10 sigma <= min(mu, 1 - mu)) and the two sides of
    # its cut, plus one size the sums could not reach and one large size outside the region
    (0.4, 0.002, 200.0),
    (0.5, 0.004, 500.0),
    (0.02, 0.001, 5000.0),
    (0.3, 0.01, 100000.0),
    (0.5, 1.0, 1000.0),
]
# log F and log(1 - F) cost a raised working precision per component: the small sizes only
SIZE_MAX_CDF = 50.0

mu_s, sg_s, y_s, n_s, i_s = sp.symbols("mu sigma y n i", positive=True)


def stirling(a):
    """delta(a) = loggamma(a) - (a - 1/2) log a + a - log sqrt(2 pi)."""
    return sp.loggamma(a) - (a - sp.Rational(1, 2)) * sp.log(a) + a - sp.log(sp.sqrt(2 * sp.pi))


def log_choose(n, y):
    return sp.loggamma(n + 1) - sp.loggamma(y + 1) - sp.loggamma(n - y + 1)


def definition(y=None, n=None):
    """l as a difference of log-betas; y and n may be symbols or integers.

    The shifts enter only as arguments of log-gammas, so the derivatives in (mu, sigma) are taken
    once with y and n symbolic and evaluated afterwards.
    """
    y = y_s if y is None else y
    n = n_s if n is None else n
    a, b = mu_s / sg_s, (1 - mu_s) / sg_s
    lbeta = lambda u, v: sp.loggamma(u) + sp.loggamma(v) - sp.loggamma(u + v)
    return log_choose(n, y) + lbeta(y + a, n - y + b) - lbeta(a, b)


def shift_form(y, n):
    """The same l as the three sums over integer shifts."""
    out = log_choose(n, y)
    out += sum(sp.log(mu_s + i * sg_s) for i in range(y))
    out += sum(sp.log(1 - mu_s + i * sg_s) for i in range(n - y))
    out -= sum(sp.log(1 + i * sg_s) for i in range(n))
    return out


def derivative_form(y, n, p, q):
    """The header's d^(p+q) l / dmu^p dsigma^q."""
    K = p + q
    s1 = sum(sp.Integer(i) ** q / (mu_s + i * sg_s) ** K for i in range(y))
    s2 = sum(sp.Integer(i) ** q / (1 - mu_s + i * sg_s) ** K for i in range(n - y))
    s3 = sum(sp.Integer(i) ** K / (1 + i * sg_s) ** K for i in range(n)) if p == 0 else sp.Integer(0)
    return (-1) ** (K - 1) * sp.factorial(K - 1) * (s1 + (-1) ** p * s2 - s3)


def shift_loggamma(expr):
    """loggamma(z + k) -> loggamma(z) + sum_(i<k) log(z + i) for every positive integer shift k.

    SymPy does not apply the recurrence of the gamma function on its own, and the identity the
    header rests on is exactly that recurrence at the integer shifts y, n - y and n.
    """
    rep = {}
    for t in expr.atoms(sp.loggamma):
        k, rest = t.args[0].as_coeff_Add()
        if rest == 0 or not (k.is_Integer and k > 0):
            continue
        rep[t] = sp.loggamma(rest) + sum(sp.log(rest + i) for i in range(int(k)))
    return expr.xreplace(rep) if rep else expr


def check_forms():
    for n, y in [(10, 3), (5, 0), (5, 5), (7, 4), (1, 0), (1, 1)]:
        d = sp.simplify(sp.expand_log(shift_loggamma(definition(y, n)) - shift_form(y, n), force=True))
        assert d == 0, ("shift identity", n, y, d)
    # the derivatives, against SymPy's own derivative of the shift form
    for n, y in [(6, 2), (4, 0), (4, 4), (1, 1)]:
        base = shift_form(y, n)
        for K in range(1, MAXK + 1):
            for q in range(0, K + 1):
                p = K - q
                lhs = derivative_form(y, n, p, q)
                rhs = sp.diff(base, mu_s, p, sg_s, q)
                assert sp.simplify(lhs - rhs) == 0, ("derivative", n, y, p, q)
    # Loader's form of log C(n, y), for 0 < y < n
    for n, y in [(10, 3), (40, 17), (7, 1)]:
        h = -y * sp.log(sp.Rational(y, n)) - (n - y) * sp.log(1 - sp.Rational(y, n))
        loader = (sp.Rational(1, 2) * sp.log(sp.Rational(n, 2 * y * (n - y)) / sp.pi)
                  + stirling(n) - stirling(y) - stirling(n - y) + h)
        assert sp.simplify(sp.expand_log(log_choose(n, y) - loader, force=True)) == 0, ("log C", n, y)
    check_stirling()
    print("forms checked against SymPy: the shift identity, its derivatives to order 6,")
    print("Loader's form of log C(n, y), and the Stirling decomposition")


def check_stirling():
    """The decomposition the route of bounded cost uses, symbolically and then at 60 digits.

    Substituting loggamma(z) = (z - 1/2) log z - z + log sqrt(2 pi) + delta(z) in the six
    log-gammas of the definition leaves, with Delta = y - n mu, A = mu + y sigma,
    B = 1 - mu + (n - y) sigma and C = 1 + n sigma (so that A + B = C exactly),

        l = log C(n, y) + y log mu + (n - y) log(1 - mu)
            + Delta [rho(w) - rho(v)],  rho(t) = ((1 + t) log1p(t) - t)/t,
              w = Delta sigma/(mu C),  v = -Delta sigma/((1 - mu) C)
            - (1/2) [log A - log mu + log B - log(1 - mu) - log C]
            + delta(A/sigma) + delta(B/sigma) - delta(C/sigma)
            - delta(mu/sigma) - delta((1 - mu)/sigma) + delta(1/sigma).

    The six log-gammas cancel between the two sides, so what is left is an identity among
    logarithms; SymPy proves that one at three rational mu (it does not pull the constant out of
    log(c x + c), so the check is on the ratio of the two sides). The whole identity is then
    checked against the definition at 60 digits, non-integer y and n among the points, which also
    shows that it needs no integer shift.
    """
    def P(z):  # loggamma(z) minus Stirling's error
        return (z - sp.Rational(1, 2)) * sp.log(z) - z + sp.log(sp.sqrt(2 * sp.pi))

    def rho(t):
        return ((1 + t) * sp.log(1 + t) - t) / t

    kk = sp.Symbol("k", positive=True)
    for mu0 in (sp.Rational(1, 2), sp.Rational(3, 7), sp.Rational(49, 50)):
        q0 = 1 - mu0
        a, b, sv = mu0 / sg_s, q0 / sg_s, 1 / sg_s
        C, D = 1 + n_s * sg_s, y_s - n_s * mu0
        w, v = D * sg_s / (mu0 * C), -D * sg_s / (q0 * C)
        lhs = P(y_s + a) + P(n_s - y_s + b) - P(n_s + sv) - P(a) - P(b) + P(sv)
        rhs = (y_s * sp.log(mu0) + (n_s - y_s) * sp.log(q0) + D * (rho(w) - rho(v))
               - sp.Rational(1, 2) * (sp.log(1 + y_s * sg_s / mu0) + sp.log(1 + (n_s - y_s) * sg_s / q0)
                                      - sp.log(1 + n_s * sg_s)))
        d = sp.expand_log(sp.expand((lhs - rhs).subs(n_s, y_s + kk)), force=True)
        r = sp.simplify(sp.radsimp(sp.powsimp(sp.expand(sp.exp(d)), force=True)))
        assert r == 1, ("the Stirling decomposition", mu0, r)

    def delta_mp(z):
        return (mpmath.loggamma(z) - (z - mpmath.mpf(1) / 2) * mpmath.log(z) + z
                - mpmath.log(mpmath.sqrt(2 * mpmath.pi)))

    def rho_mp(t):
        return ((1 + t) * mpmath.log1p(t) - t) / t

    def stirling_mp(mu, sigma, y, n):
        a, b, sv = mu / sigma, (1 - mu) / sigma, 1 / sigma
        C, D = 1 + n * sigma, y - n * mu
        out = (mpmath.loggamma(n + 1) - mpmath.loggamma(y + 1) - mpmath.loggamma(n - y + 1)
               + y * mpmath.log(mu) + (n - y) * mpmath.log1p(-mu))
        if D != 0:
            out += D * (rho_mp(D * sigma / (mu * C)) - rho_mp(-D * sigma / ((1 - mu) * C)))
        out -= (mpmath.log1p(y * sigma / mu) + mpmath.log1p((n - y) * sigma / (1 - mu))
                - mpmath.log1p(n * sigma)) / 2
        return (out + delta_mp(a + y) + delta_mp(b + n - y) - delta_mp(sv + n)
                - delta_mp(a) - delta_mp(b) + delta_mp(sv))

    mpmath.mp.dps = 60
    worst = mpmath.mpf(0)
    pts = [(mu, sg, n) for (mu, sg, n) in PARAMS] + [(0.37, 0.21, 11.3)]
    for (mu, sg, n) in pts:
        mu, sg, n = mpmath.mpf(mu), mpmath.mpf(sg), mpmath.mpf(n)
        for frac in ("0", "0.001", "0.3", "0.5", "0.77", "1"):
            y = mpmath.floor(mpmath.mpf(frac) * n) if n != mpmath.mpf("11.3") else mpmath.mpf("3.7")
            ref = log_mass(y, mu, sg, n)
            worst = max(worst, abs(ref - stirling_mp(mu, sg, y, n)) / abs(ref))
    mpmath.mp.dps = 15
    assert worst < mpmath.mpf(10) ** -50, ("the Stirling decomposition at 60 digits", worst)
    print(f"  the Stirling decomposition: symbolic at three mu, {float(worst):.2g} relative at 60 digits")


# ------------------------------------------------------------------ mpmath references

MASS_FUNCS = {}


def mass_derivative_functions():
    """Lambdified (mu, sigma, y, n) derivatives of the DEFINITION, indexed by (K, b)."""
    out = {}
    l = definition()
    for K in range(0, MAXK + 1):
        for b in range(0, K + 1):
            d = sp.diff(l, mu_s, K - b, sg_s, b)
            out[(K, b)] = sp.lambdify((mu_s, sg_s, y_s, n_s), d, modules="mpmath")
    return out


def log_mass(y, mu, sigma, n):
    a, b = mu / sigma, (1 - mu) / sigma
    lbeta = lambda u, v: mpmath.loggamma(u) + mpmath.loggamma(v) - mpmath.loggamma(u + v)
    return (mpmath.loggamma(n + 1) - mpmath.loggamma(y + 1) - mpmath.loggamma(n - y + 1)
            + lbeta(y + a, n - y + b) - lbeta(a, b))


def log_cdf(k, mu, sigma, n, upper):
    """log F(k) or log(1 - F(k)) by summing the mass over the finite support."""
    if upper:
        terms = [mpmath.e ** log_mass(j, mu, sigma, n) for j in range(int(k) + 1, int(n) + 1)]
    else:
        terms = [mpmath.e ** log_mass(j, mu, sigma, n) for j in range(0, int(k) + 1)]
    return mpmath.log(mpmath.fsum(terms))


def value(kind, y, mu, sigma, n, K, b, prec):
    with mpmath.workprec(prec if kind != 0 else 2 * prec):
        mu0, sg0 = mpmath.mpf(mu), mpmath.mpf(sigma)
        if kind == 0:
            return MASS_FUNCS[(K, b)](mu0, sg0, mpmath.mpf(y), mpmath.mpf(n))
        f = lambda a, t: log_cdf(math.floor(y), a, t, n, kind == 2)
        if K == 0:
            return f(mu0, sg0)
        return mpmath.diff(f, (mu0, sg0), (K - b, b))


def row(kind, y, mu, sigma, n, K, b):
    try:
        a = value(kind, y, mu, sigma, n, K, b, 200)
        d = value(kind, y, mu, sigma, n, K, b, 260)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    if d == 0 or not mpmath.isfinite(d) or abs(a - d) > AGREE * abs(d):
        return None
    hi = float(d)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    lo = float(d - mpmath.mpf(hi))
    return (f"{kind},{float(y).hex()},{float(mu).hex()},{float(sigma).hex()},{float(n).hex()},"
            f"{K},{b},{hi.hex()},{lo.hex()}\n")


def ys_for(mu, sigma, n):
    """The ends of the support, the mean, and the points around it."""
    sd = math.sqrt(n * mu * (1 - mu) * (1 + (n - 1) * sigma / (1 + sigma)))
    ys = {0.0, n, 1.0 if n >= 1 else 0.0, max(0.0, n - 1.0)}
    for f in (-1.0, 0.0, 1.0):
        ys.add(float(min(max(0.0, round(n * mu + f * sd)), n)))
    if n >= 4:
        ys.add(2.5)  # one non-integer, to exercise the floor of log F
    return sorted(set(ys))


def task(job):
    kind, mu, sigma, n = job
    ys = [v for v in ys_for(mu, sigma, n) if v == math.floor(v) or kind != 0]
    if kind != 0:
        ys = ys[:4]
    lines, dropped = [], 0
    for y in ys:
        for K in range(0, MAXK + 1):
            for b in range(0, K + 1):
                r = row(kind, y, mu, sigma, n, K, b)
                if r is None:
                    dropped += 1
                else:
                    lines.append(r)
    return lines, dropped


def init():
    global MASS_FUNCS
    MASS_FUNCS = mass_derivative_functions()


def main():
    check_forms()
    jobs = [(0, mu, sg, n) for (mu, sg, n) in PARAMS]
    jobs += [(k, mu, sg, n) for k in (1, 2) for (mu, sg, n) in PARAMS if n <= SIZE_MAX_CDF]
    with Pool(8, initializer=init) as pool:
        res = pool.map(task, jobs)
    n_rows = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("kind,y,mu,sigma,size,K,b,hi,lo\n")
        for lines, dr in res:
            f.writelines(lines)
            n_rows += len(lines)
            dropped += dr
    print(f"distrib_betabinom1.csv: {n_rows} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
