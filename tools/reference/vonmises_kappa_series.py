"""The two series of the von Mises map s -> kappa, derived with SymPy.

kappa is the concentration, s = sqrt(-2 log rho) the circular standard
deviation and rho = A(kappa) = I_1(kappa) / I_0(kappa) the mean resultant
length, so rho = exp(-s^2 / 2) and kappa = A^-1(exp(-s^2 / 2)).

Both ends of the range have a convergent-looking power series, and that is what
makes the map cheap and cancellation-free there:

  small s (concentrated): kappa = 1/s^2 + 1/2 + 5 s^2 / 24 + ..., a series in
      s^2 obtained by inverting u = -log A(kappa) as a series in 1/kappa;
  large s (nearly uniform): kappa = 2 rho + rho^3 + ..., a series in
      rho = exp(-s^2 / 2) obtained by inverting rho = A(kappa) as a series in
      kappa around zero.

Run with `python -I tools/reference/vonmises_kappa_series.py <mode>`:

  small    the coefficients of kappa as a series in s^2, and the check that
           the series satisfies A(kappa(s)) = exp(-s^2/2) to the order kept;
  large    the coefficients of kappa as a series in rho, with the same check;
  verify   both series against mpmath's own root of A(kappa) = rho, to find
           where each one stops being worth its cost.
"""

import sys

from sympy import (
    Rational,
    Symbol,
    besseli,
    exp,
    log,
    nsimplify,
    series,
    simplify,
    solve,
    sqrt,
    symbols,
)

# How many terms of each series to produce. Six derivatives in s of a series in
# s^2 need the terms up to s^(2 N) with 2 N - 6 >= 0, plus a few to see the
# decay.
N_SMALL = 8
N_LARGE = 8


def hankel_coeff(nu, j):
    """a_j(nu) of the Hankel expansion of I_nu, a rational number.

    I_nu(z) ~ e^z / sqrt(2 pi z) sum_j (-1)^j a_j(nu) / z^j, with
    a_j(nu) = prod_{i=1}^{j} (4 nu^2 - (2i-1)^2) / (j! 8^j).
    """
    num = Rational(1)
    for i in range(1, j + 1):
        num *= Rational(4 * nu * nu - (2 * i - 1) ** 2)
    den = Rational(1)
    for i in range(1, j + 1):
        den *= i * 8
    return num / den


def ratio_asymptotic(order):
    """A(kappa) = I_1/I_0 as a series in t = 1/kappa, to the given order.

    The two Hankel expansions share the factor e^kappa / sqrt(2 pi kappa),
    which cancels in the ratio, so the ratio is the quotient of two explicit
    polynomials in t and is expanded as such. SymPy cannot series the Bessel
    quotient itself, so the coefficients are built from hankel_coeff.
    """
    t = Symbol("t", positive=True)
    n = order + 3
    p1 = sum((-1) ** j * hankel_coeff(1, j) * t**j for j in range(n))
    p0 = sum((-1) ** j * hankel_coeff(0, j) * t**j for j in range(n))
    r = series(p1 / p0, t, 0, order + 1).removeO()
    return r.expand(), t


def small_series(n_terms=N_SMALL, show=True):
    """kappa as a series in v = s^2: kappa = 1/v + 1/2 + sum_{j>=1} a_j v^j."""
    t = Symbol("t", positive=True)  # t = 1 / kappa
    r, t = ratio_asymptotic(n_terms + 2)
    if show:
        print("A(kappa) as a series in t = 1/kappa:")
        print("  " + str(r))
        print()
    # u = -log A, as a series in t
    u = series(-log(r), t, 0, n_terms + 2).removeO().expand()
    if show:
        print("u = -log A(kappa) as a series in t = 1/kappa:")
        print("  " + str(u))
        print()
    # Invert u(t) = v / 2 with v = s^2: write t as a series in v and match.
    v = Symbol("v", positive=True)
    coeffs = [Symbol("c%d" % j) for j in range(1, n_terms + 2)]
    t_of_v = sum(c * v**j for j, c in enumerate(coeffs, start=1))
    expr = series(u.subs(t, t_of_v) - v / 2, v, 0, n_terms + 2).removeO().expand()
    sol = {}
    for j in range(1, n_terms + 2):
        c = expr.coeff(v, j).subs(sol)
        s = solve(c, coeffs[j - 1], dict=True)
        if not s:
            break
        sol[coeffs[j - 1]] = simplify(s[0][coeffs[j - 1]])
    t_sol = t_of_v.subs(sol)
    kappa = series(1 / t_sol, v, 0, n_terms).removeO().expand()
    if show:
        print("kappa as a series in v = s^2 (concentrated end):")
        print("  kappa = " + str(kappa))
        print()
        print("  coefficient of v^j, as a rational and as a double:")
        print("    1/v           1")
        for j in range(0, n_terms):
            c = simplify(kappa.coeff(v, j))
            if c == 0:
                continue
            print("    v^%-2d  %18s  %+.17e" % (j, str(nsimplify(c)), float(c)))
        print()
        # The check: A(kappa(v)) must be exp(-v/2) to the order kept.
        chk = series(r.subs(t, 1 / kappa) - exp(-v / 2), v, 0, n_terms - 1).removeO()
        print("  check A(kappa(v)) - exp(-v/2), expanded to v^%d:" % (n_terms - 2))
        print("    " + str(simplify(chk)))
    return kappa, v


def large_series(n_terms=N_LARGE, show=True):
    """kappa as a series in rho: kappa = 2 rho + sum_{j>=2} b_j rho^j."""
    from sympy import factorial, gamma

    k = Symbol("k", positive=True)
    rho = Symbol("rho", positive=True)
    # A(kappa) around kappa = 0, from the power series
    #   I_nu(k) = sum_j (k/2)^(2j+nu) / (j! Gamma(j + nu + 1)),
    # whose common factor (k/2)^nu is kept so that the quotient starts at k/2.
    n = n_terms + 3
    i1 = sum((k / 2) ** (2 * j + 1) / (factorial(j) * gamma(j + 2)) for j in range(n))
    i0 = sum((k / 2) ** (2 * j) / (factorial(j) * gamma(j + 1)) for j in range(n))
    a = series(i1 / i0, k, 0, n_terms + 2).removeO().expand()
    if show:
        print("A(kappa) as a series in kappa around zero:")
        print("  " + str(a))
        print()
    coeffs = [Symbol("b%d" % j) for j in range(1, n_terms + 1)]
    k_of_rho = sum(c * rho**j for j, c in enumerate(coeffs, start=1))
    expr = series(a.subs(k, k_of_rho) - rho, rho, 0, n_terms + 1).removeO().expand()
    sol = {}
    for j in range(1, n_terms + 1):
        c = expr.coeff(rho, j).subs(sol)
        s = solve(c, coeffs[j - 1], dict=True)
        if not s:
            break
        sol[coeffs[j - 1]] = simplify(s[0][coeffs[j - 1]])
    kappa = k_of_rho.subs(sol).expand()
    if show:
        print("kappa as a series in rho (nearly uniform end):")
        print("  kappa = " + str(kappa))
        print()
        print("  coefficient of rho^j, as a rational and as a double:")
        for j in range(1, n_terms + 1):
            c = simplify(kappa.coeff(rho, j))
            if c == 0:
                continue
            print("    rho^%-2d  %18s  %+.17e" % (j, str(nsimplify(c)), float(c)))
        print()
        chk = series(a.subs(k, kappa) - rho, rho, 0, n_terms).removeO()
        print("  check A(kappa(rho)) - rho, expanded to rho^%d:" % (n_terms - 1))
        print("    " + str(simplify(chk)))
    return kappa, rho


def mode_small():
    small_series()


def mode_large():
    large_series()


def mode_verify():
    from mpmath import besseli as mbesseli
    from mpmath import exp as mexp
    from mpmath import findroot, log as mlog, mp, mpf, sqrt as msqrt

    mp.dps = 50

    def mratio(k):
        k = mpf(k)
        return mpf(0) if k == 0 else mbesseli(1, k) / mbesseli(0, k)

    def exact(s):
        rho = mexp(-mpf(s) ** 2 / 2)
        g = 1 / (2 * (1 - rho)) if rho > mpf("0.5") else 2 * rho
        return findroot(lambda k: mratio(k) - rho, g)

    ks, v = small_series(show=False)
    kl, r = large_series(show=False)
    small_c = [(j, float(simplify(ks.coeff(v, j)))) for j in range(0, N_SMALL)]
    large_c = [(j, float(simplify(kl.coeff(r, j)))) for j in range(1, N_LARGE + 1)]

    def small_eval(s):
        vv = s * s
        out = 1.0 / vv
        p = 1.0
        for j, c in small_c:
            if c == 0.0:
                continue
            out += c * (vv**j)
        return out

    def large_eval(s):
        rr = float(mexp(-mpf(s) ** 2 / 2))
        out = 0.0
        for j, c in large_c:
            if c == 0.0:
                continue
            out += c * (rr**j)
        return out

    print("Relative error of each series against the mpmath root, by s.")
    print("%10s  %22s  %14s  %14s" % ("s", "kappa", "series in s^2", "series in rho"))
    grid = [
        "1e-4", "1e-3", "1e-2", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5",
        "0.7", "1.0", "1.3", "1.6", "2.0", "2.3", "2.6", "3.0", "3.5", "4.0",
    ]
    for s_str in grid:
        s = mpf(s_str)
        k = exact(s)
        es = abs(mpf(small_eval(float(s))) / k - 1) if float(s) < 10 else float("nan")
        el = abs(mpf(large_eval(float(s))) / k - 1)
        print("%10s  %22.14e  %14.3e  %14.3e" % (s_str, float(k), float(es), float(el)))


MODES = {"small": mode_small, "large": mode_large, "verify": mode_verify}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "small"
    if mode not in MODES:
        print("modes: " + ", ".join(sorted(MODES)))
        raise SystemExit(2)
    MODES[mode]()
