"""SymPy and mpmath checks of the closed forms of pig, and mpmath references of its log-mass and
cumulative derivatives in (mu, alpha) to total order 6.

Usage (from the repository root):  python -I tools/reference/distrib_pig.py

The Poisson-inverse Gaussian in the mean mu > 0 and alpha > 0, with s = sqrt(mu^2 + alpha^2),

    p(y) = sqrt(2 alpha / pi) e^(s - mu) (mu alpha / (mu + s))^y K_(y - 1/2)(alpha) / y!,

on y = 0, 1, 2, ..., with mean mu and variance mu + mu^2 / (s - mu).

Part 1 (printed; the script exits with status 1 on any disagreement) checks the forms the header
uses:
  - the order of the Bessel function is always half an odd integer, so it is elementary:
    K_(y-1/2)(alpha) = sqrt(pi / (2 alpha)) e^-alpha T_m(alpha) with m = max(y - 1, 0) and
    T_m = sum_(k=0)^m (m+k)! / (k! (m-k)!) (2 alpha)^-k, a sum of positive terms (checked against
    mpmath's own Bessel function at 60 digits);
  - the square root of alpha in the mass cancels the one in that identity, which leaves
        l = (s - mu) - alpha + y log mu - y log(mu + s) - y log 2 + log Q(theta) - loggamma(y + 1),
        Q(theta) = sum_(j=1)^y a_j e^(j theta),  a_j = (m + y - j)! / ((y - j)! (m - y + j)!),
        theta = log(2 alpha),
    (a_j = c_(y-j) with c_k the coefficient above) so that every power of alpha sits inside one
    exponential family in theta whose statistic j is at least 1;
  - d log Q / dtheta^q is the q-th cumulant of j under the weights a_j e^(j theta) / Q, and the
    conversion from the central moments to the cumulants to order 6;
  - the mean's score is exactly dl/dmu = c (y - mu) with c = 1/mu - 1/s, so that the terms of the
    size of y that the naive assembly carries are gone, and every derivative with at least one
    index in mu is (y - mu) c_(p-1,q) - (p-1) c_(p-2,q), c_(i,j) the mixed partials of c;
  - c = h(u) / mu with u = (alpha / mu)^2 and h(u) = 1 - (1 + u)^(-1/2) = u / (w (w + 1)),
    w = sqrt(1 + u), whose derivatives are h^(n)(u) = (-1)^(n+1) (2n)! / (4^n n!) (1 + u)^(-1/2-n),
    and the partials of the two inner maps u and 1/mu;
  - the expected information: I_mu_mu = 1/mu - 1/s = c exactly and I_mu_alpha = 0 exactly
    (checked by summing the mass at 40 digits), so only I_alpha_alpha needs a sum over the support.

Part 2, mpmath (tests/data/reference/distrib_pig.csv), one row per value:
    kind (0 log p; the cumulative kinds 1 and 2 are not generated, see CDF_ROWS), y, mu, alpha, K, b, hi, lo
with b the number of derivatives in alpha (K - b in mu) and y, mu, alpha, hi, lo as hexadecimal
doubles. The mass rows are mpmath evaluations of the symbolic derivatives of the DEFINITION above,
whose Bessel function SymPy differentiates by its own recurrence and mpmath evaluates by its own
series, so nothing of part 1 enters them.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
import sympy as sp

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_pig.csv"
MAXK = 6
AGREE = mpmath.mpf(2) ** -80

# (mu, alpha). alpha -> infinity at fixed mu is the Poisson limit; alpha << mu is the heavy-tailed
# end (the variance mu + mu^2/(s - mu) grows as 2 mu^3 / alpha^2).
PARAMS = [
    (2.0, 3.0),
    (0.5, 0.1),
    (10.0, 1.0),
    (40.0, 1e-2),
    (1.0, 1e3),
    (1e3, 1e-3),
    (1e3, 1e3),
    (1e-3, 1e-3),
    (100.0, 5.0),
    (1e4, 2.0),
]
# Only the mass is referenced here. log F and log(1 - F) have no closed form for this family, so
# every reference value would be a numerical derivative of a sum of masses, each mass a Bessel
# function, at a raised working precision: measured about a second a component at order 4 on the
# light-tailed parameters and far worse on the others, with no bound on the cost at the heavy end.
# What the library computes there is the generic summation route of generic_cdf.hpp, whose accuracy
# is measured in its own section on the families with a closed cumulative function, and
# tests/distrib/test_pig.cpp checks it here against 5-point differences to order 3.
CDF_ROWS = False

mu_s, al_s, u_s, th_s, y_s = sp.symbols("mu alpha u theta y", positive=True)
S_OF = sp.sqrt(mu_s ** 2 + al_s ** 2)


def definition(y=None):
    """l with the Bessel function left as it is; y may be a symbol or an integer.

    The count enters only as the order of the Bessel function and as a multiplier, so the
    derivatives in (mu, alpha) are taken once with y symbolic and evaluated afterwards.
    """
    y = y_s if y is None else y
    return (sp.Rational(1, 2) * sp.log(2 * al_s / sp.pi) + (S_OF - mu_s)
            + y * sp.log(mu_s * al_s / (mu_s + S_OF))
            + sp.log(sp.besselk(y - sp.Rational(1, 2), al_s)) - sp.loggamma(y + 1))


def coef(m, k):
    return sp.factorial(m + k) / (sp.factorial(k) * sp.factorial(m - k))


def q_form(y):
    """l with the Bessel function replaced by the half-integer polynomial, written through Q."""
    m = max(y - 1, 0)
    Q = sum(coef(m, y - j) * sp.exp(j * th_s) for j in range(1, y + 1)) if y >= 1 else sp.Integer(1)
    Q = Q.subs(th_s, sp.log(2 * al_s))
    return ((S_OF - mu_s) - al_s + y * sp.log(mu_s) - y * sp.log(mu_s + S_OF) - y * sp.log(2)
            + sp.log(Q) - sp.loggamma(y + 1))


def check_bessel_identity():
    """The half-integer identity, against mpmath's own Bessel function."""
    mpmath.mp.dps = 60
    worst = mpmath.mpf(0)
    for y in list(range(0, 25)) + [40, 60]:
        m = max(y - 1, 0)
        for a in ("1e-4", "0.01", "0.3", "1", "7", "100", "1e4"):
            al = mpmath.mpf(a)
            T = mpmath.fsum(mpmath.binomial(m + k, k) * mpmath.binomial(m, k) * mpmath.factorial(k)
                            / (2 * al) ** k for k in range(m + 1))
            lhs = mpmath.log(mpmath.besselk(mpmath.mpf(2 * y - 1) / 2, al))
            rhs = mpmath.log(mpmath.pi / (2 * al)) / 2 - al + mpmath.log(T)
            worst = max(worst, abs(lhs - rhs) / abs(lhs) if lhs else abs(lhs - rhs))
    assert worst < mpmath.mpf(10) ** -50, ("half-integer identity", worst)
    mpmath.mp.dps = 15
    return worst


def check_forms():
    worst = check_bessel_identity()
    print(f"  half-integer Bessel identity against mpmath: worst {float(worst):.2g} relative")
    # the Q form of the mass, for the small counts SymPy can reduce
    for y in (0, 1, 2):
        r = sp.besselsimp(sp.exp(definition(y)) / sp.exp(q_form(y)))
        for _ in range(4):
            r = sp.radsimp(sp.expand(sp.besselsimp(r).rewrite(sp.exp)))
            if r == 1:
                break
        assert r == 1, ("Q form", y, r)
    print("  the Q form of the mass: y = 0, 1, 2 reduce to the definition")
    # the exact score in mu, and the Leibniz form of every derivative with an index in mu. The
    # Bessel function does not depend on mu, so these are rational in (mu, alpha): the first two
    # orders are reduced symbolically and the rest compared at 50 digits on a grid, which is as
    # conclusive for rational functions and far cheaper.
    c_s = 1 / mu_s - 1 / S_OF
    l = definition()
    assert sp.simplify(sp.diff(l, mu_s) - c_s * (y_s - mu_s)) == 0, "score in mu"
    print("  the score in mu is exactly c (y - mu), c = 1/mu - 1/s")
    # the points keep |log(alpha/mu)| moderate: 1/mu - 1/s itself cancels like (mu/alpha)^2 per
    # order, which is what the header's h(u) route exists for, and at 200 digits a ratio of 100
    # still leaves 160 of them
    pts = [("2", "3"), ("0.5", "0.1"), ("7", "1"), ("1", "5"), ("100", "1")]
    mpmath.mp.dps = 200
    worst_leib = mpmath.mpf(0)
    for K in range(1, MAXK + 1):
        for p_ in range(1, K + 1):
            q = K - p_
            form = (y_s - mu_s) * sp.diff(c_s, mu_s, p_ - 1, al_s, q)
            if p_ >= 2:
                form -= (p_ - 1) * sp.diff(c_s, mu_s, p_ - 2, al_s, q)
            exact = sp.diff(l, mu_s, p_, al_s, q)
            if K <= 2:
                assert sp.simplify(form - exact) == 0, ("mu derivatives", p_, q)
                continue
            fa = sp.lambdify((mu_s, al_s, y_s), form, modules="mpmath")
            fb = sp.lambdify((mu_s, al_s, y_s), exact, modules="mpmath")
            for (a, b) in pts:
                for yv in (0, 3, 13):  # never equal to one of the mu above, where (y - mu) c_(p-1,q) is exactly 0
                    va = fa(mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(yv))
                    vb = fb(mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(yv))
                    sc = max(abs(va), abs(vb))
                    if sc > 0:
                        worst_leib = max(worst_leib, abs(va - vb) / sc)
    mpmath.mp.dps = 15
    assert worst_leib < mpmath.mpf(10) ** -40, ("Leibniz form", worst_leib)
    print("  the Leibniz form of the derivatives with an index in mu: orders 1 and 2 symbolic,")
    print(f"  orders 3 to 6 within {float(worst_leib):.2g} relative at 200 digits on five (mu, alpha)")
    # h and its derivatives, and c = h(u) / mu
    w_s = sp.sqrt(1 + u_s)
    h = 1 - (1 + u_s) ** sp.Rational(-1, 2)
    assert sp.simplify(h - u_s / (w_s * (w_s + 1))) == 0, "h without cancellation"
    for n in range(1, MAXK + 1):
        form = (-1) ** (n + 1) * sp.factorial(2 * n) / (4 ** n * sp.factorial(n)) * (1 + u_s) ** (sp.Rational(-1, 2) - n)
        assert sp.simplify(form - sp.diff(h, u_s, n)) == 0, ("h derivatives", n)
    assert sp.simplify(c_s - (h.subs(u_s, (al_s / mu_s) ** 2) / mu_s)) == 0, "c through h"
    print("  h = 1 - (1 + u)^(-1/2) = u / (w (w + 1)) with its derivatives, and c = h(u) / mu")
    # the partials of the inner maps u = alpha^2 / mu^2 and 1 / mu
    u_of = al_s ** 2 / mu_s ** 2
    for p_ in range(0, MAXK + 1):
        for q in range(0, MAXK + 1 - p_):
            if p_ + q == 0:
                continue
            da = {0: al_s ** 2, 1: 2 * al_s, 2: sp.Integer(2)}.get(q, sp.Integer(0))
            form = da * (-1) ** p_ * sp.factorial(p_ + 1) / mu_s ** (p_ + 2)
            assert sp.simplify(form - sp.diff(u_of, mu_s, p_, al_s, q)) == 0, ("u map", p_, q)
            form1 = ((-1) ** p_ * sp.factorial(p_) / mu_s ** (p_ + 1)) if q == 0 else sp.Integer(0)
            assert sp.simplify(form1 - sp.diff(1 / mu_s, mu_s, p_, al_s, q)) == 0, ("1/mu map", p_, q)
    print("  the partials of the two inner maps u and 1/mu")
    # the cumulants of j: d^q log Q / dtheta^q, and the conversion from the central moments
    a = sp.symbols("a1:6", positive=True)  # weights on j = 1..5
    Q = sum(a[j - 1] * sp.exp(j * th_s) for j in range(1, 6))
    lq = sp.log(Q)
    wts = [a[j - 1] * sp.exp(j * th_s) / Q for j in range(1, 6)]
    m1 = sum(wts[j - 1] * j for j in range(1, 6))
    cm = [sp.simplify(sum(wts[j - 1] * (j - m1) ** r for j in range(1, 6))) for r in range(0, 7)]
    kap = [None, m1, cm[2], cm[3],
           cm[4] - 3 * cm[2] ** 2,
           cm[5] - 10 * cm[3] * cm[2],
           cm[6] - 15 * cm[4] * cm[2] - 10 * cm[3] ** 2 + 30 * cm[2] ** 3]
    for q in range(1, MAXK + 1):
        d = sp.simplify(sp.diff(lq, th_s, q) - kap[q])
        assert d == 0, ("cumulant", q, d)
    print("  the cumulants of Q to order 6 and their conversion from the central moments")


def check_information():
    """I_mu_mu = 1/mu - 1/s and I_mu_alpha = 0, by summing the mass at 40 digits."""
    mpmath.mp.dps = 40
    out = []
    # alpha >= mu, where the variance mu + mu^2/(s - mu) is modest: the second moment under the
    # score converges with the variance, so a heavy-tailed point would test the cut, not the identity
    for (mu, al) in [(2.0, 3.0), (1.0, 5.0), (0.5, 2.0), (4.0, 4.0)]:
        mu0, al0 = mpmath.mpf(mu), mpmath.mpf(al)
        s = mpmath.sqrt(mu0 ** 2 + al0 ** 2)
        c = 1 / mu0 - 1 / s
        sd = mpmath.sqrt(mu0 + mu0 ** 2 / (s - mu0))
        top = int(mu + 80 * sd) + 40
        # the score in mu is c (y - mu); the score in alpha by a central difference of the
        # definition at high precision
        def lp(y, a, b):
            ss = mpmath.sqrt(a ** 2 + b ** 2)
            return (mpmath.log(2 * b / mpmath.pi) / 2 + (ss - a) + y * mpmath.log(a * b / (a + ss))
                    + mpmath.log(mpmath.besselk(mpmath.mpf(2 * y - 1) / 2, b)) - mpmath.loggamma(y + 1))
        imm = ima = tot = mpmath.mpf(0)
        for y in range(top + 1):
            p = mpmath.e ** lp(y, mu0, al0)
            sm = c * (y - mu0)
            sa = mpmath.diff(lambda b: lp(y, mu0, b), al0)
            imm += p * sm * sm
            ima += p * sm * sa
            tot += p
        out.append((mu, al, float(tot), float(imm), float(c), float(ima)))
    mpmath.mp.dps = 15
    for (mu, al, tot, imm, c, ima) in out:
        # the support sum is cut, so the identities hold to the mass left beyond the cut, which the
        # heavy tail at alpha << mu makes the binding term
        tol = max(1e-18, 1e4 * abs(1.0 - tot))
        assert abs(tot - 1.0) < 1e-5, ("mass", mu, al, tot)
        assert abs(imm - c) < tol * c, ("I_mu_mu", mu, al, imm, c)
        assert abs(ima) < tol * c, ("I_mu_alpha", mu, al, ima)
        print(f"  (mu, alpha) = ({mu}, {al}): mass 1 - {1.0 - tot:.3g}, I_mu_mu {imm:.17g} against "
              f"1/mu - 1/s {c:.17g}, I_mu_alpha {ima:.3g}")


# ------------------------------------------------------------------ mpmath references

MASS_FUNCS = {}


def mass_derivative_functions():
    """Lambdified (mu, alpha, y) derivatives of the DEFINITION, indexed by (K, b)."""
    out = {}
    l = definition()
    for K in range(0, MAXK + 1):
        for b in range(0, K + 1):
            d = sp.diff(l, mu_s, K - b, al_s, b)
            out[(K, b)] = sp.lambdify((mu_s, al_s, y_s), d, modules="mpmath")
    return out


def log_mass(y, mu, al):
    s = mpmath.sqrt(mu ** 2 + al ** 2)
    return (mpmath.log(2 * al / mpmath.pi) / 2 + (s - mu) + y * mpmath.log(mu * al / (mu + s))
            + mpmath.log(mpmath.besselk(mpmath.mpf(2 * int(y) - 1) / 2, al)) - mpmath.loggamma(y + 1))


def log_cdf(k, mu, al, upper):
    """log F(k) by summing the mass; log(1 - F(k)) as its complement, the upper tail being infinite."""
    lower = mpmath.fsum([mpmath.e ** log_mass(j, mu, al) for j in range(0, int(k) + 1)])
    return mpmath.log(1 - lower) if upper else mpmath.log(lower)


def value(kind, y, mu, al, K, b, prec):
    with mpmath.workprec(prec if kind == 0 else 3 * prec):
        mu0, al0 = mpmath.mpf(mu), mpmath.mpf(al)
        if kind == 0:
            return MASS_FUNCS[(K, b)](mu0, al0, mpmath.mpf(y))
        f = lambda a, c: log_cdf(math.floor(y), a, c, kind == 2)
        if K == 0:
            return f(mu0, al0)
        return mpmath.diff(f, (mu0, al0), (K - b, b))


def row(kind, y, mu, al, K, b):
    try:
        a = value(kind, y, mu, al, K, b, 200)
        d = value(kind, y, mu, al, K, b, 260)
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    if d == 0 or not mpmath.isfinite(d) or abs(a - d) > AGREE * abs(d):
        return None
    hi = float(d)
    if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
        return None
    lo = float(d - mpmath.mpf(hi))
    return (f"{kind},{float(y).hex()},{float(mu).hex()},{float(al).hex()},"
            f"{K},{b},{hi.hex()},{lo.hex()}\n")


def ys_for(mu, al):
    s = math.sqrt(mu * mu + al * al)
    sd = math.sqrt(mu + mu * mu / (s - mu)) if s > mu else 1e6
    ys = {0.0, 1.0, 2.0, 5.0}
    for f in (0.0, 1.0, 3.0):
        ys.add(float(min(round(mu + f * sd), 3000)))
    ys.add(float(min(round(mu), 3000)))
    # mpmath's Bessel function is slow where its order and its argument are close (measured at
    # (mu, alpha) = (1e3, 1e3): 6.5 s for the value at y = 1e3 and twice that per order, against
    # 0.03 s at y = 1175). That is a property of the reference route, not of the library's, whose
    # cost does not depend on y at all; the counts in that band are left out.
    return sorted(v for v in ys if v >= 0.0 and not (al > 100.0 and 0.8 * al < v < 1.3 * al))


def task(job):
    kind, mu, al = job
    ys = ys_for(mu, al)
    kmax = MAXK
    lines, dropped = [], 0
    for y in ys:
        for K in range(0, kmax + 1):
            for b in range(0, K + 1):
                r = row(kind, y, mu, al, K, b)
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
    check_information()
    jobs = [(0, mu, al) for (mu, al) in PARAMS]
    if CDF_ROWS:
        light = [(mu, al) for (mu, al) in PARAMS if al >= 0.5 * mu][:4]
        jobs += [(k, mu, al) for k in (1, 2) for (mu, al) in light]
    res = []
    with Pool(8, initializer=init) as pool:
        for job, out in zip(jobs, pool.imap(task, jobs)):
            res.append(out)
            print(f"  job {job}: {len(out[0])} rows, {out[1]} dropped", flush=True)
    n_rows = dropped = 0
    with open(OUT, "w", newline="\n") as f:
        f.write("kind,y,mu,alpha,K,b,hi,lo\n")
        for lines, dr in res:
            f.writelines(lines)
            n_rows += len(lines)
            dropped += dr
    print(f"distrib_pig.csv: {n_rows} rows ({dropped} dropped)")


if __name__ == "__main__":
    sys.exit(main())
