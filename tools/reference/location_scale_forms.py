"""Checks the two chain rules of include/laplace/distrib/detail/location_scale.hpp
against SymPy's own derivatives.

Usage (from the repository root):  python -I tools/reference/location_scale_forms.py

A location-scale distribution has log F(y) = Lambda(z), z = (y - mu)/sigma, and
the headers need its derivatives in the parameters. Two routines do that:

  mixed_dk        the second parameter is sigma itself,
  mixed_power_dk  the second parameter is w with sigma = w^k (the variance
                  k = 1/2, the precision k = -1/2, the rate k = -1).

Both build a polynomial in z whose coefficients multiply Lambda^(j)(z). This
script rebuilds those polynomials by the recursions written in the header and
compares them with sympy.diff of Lambda((y - mu) / w**k), for every order up to
six, every split between the two parameters, and several k. Lambda stays an
undefined function, so the check is an identity in Lambda, not a check at a
point.

Nothing is written; the script prints the number of forms checked and exits
non-zero if one disagrees.
"""
import sys

import sympy as sp

MAXK = 6


def recursion_terms(a, b, k):
    """The polynomial R_b of the header, as a dict {(j, m): coefficient} meaning
    coefficient * z^m * Lambda^(j)(z)."""
    z = sp.Symbol("z")
    Lam = sp.Function("Lam")
    # R_0 = (-d/dz)^a Lambda
    q = Lam(z)
    for _ in range(a):
        q = -sp.diff(q, z)
    # R_(i+1) = -((k a + i) R_i + k z R_i')
    for i in range(b):
        q = -((k * a + i) * q + k * z * sp.diff(q, z))
    return sp.expand(q)


def sigma_recursion_terms(a, b):
    """The polynomial of mixed_dk: sigma first, b times, with n incrementing,
    then mu, a times."""
    z = sp.Symbol("z")
    Lam = sp.Function("Lam")
    q = Lam(z)
    for n in range(b):  # d/dsigma: -(n Q + z Q')
        q = -(n * q + z * sp.diff(q, z))
    for _ in range(a):  # d/dmu: -Q'
        q = -sp.diff(q, z)
    return sp.expand(q)


def main():
    y, mu, w, sigma = sp.symbols("y mu w sigma", positive=True)
    Lam = sp.Function("Lam")
    z = sp.Symbol("z")
    checked = 0
    bad = []

    # mixed_power_dk, several exponents k
    for k in (sp.Rational(1, 2), sp.Rational(-1, 2), sp.Integer(-1), sp.Integer(1),
              sp.Rational(1, 3)):
        s = w ** k
        expr = Lam((y - mu) / s)
        for K in range(1, MAXK + 1):
            for b in range(K + 1):
                a = K - b
                exact = sp.diff(expr, mu, a, w, b)
                claim = recursion_terms(a, b, k).subs(z, (y - mu) / s) * s ** (-a) * w ** (-b)
                checked += 1
                if sp.simplify(sp.expand(exact - claim)) != 0:
                    bad.append(("mixed_power_dk", k, K, b))

    # mixed_dk, the second parameter sigma itself
    expr = Lam((y - mu) / sigma)
    for K in range(1, MAXK + 1):
        for b in range(K + 1):
            a = K - b
            exact = sp.diff(expr, mu, a, sigma, b)
            claim = sigma_recursion_terms(a, b).subs(z, (y - mu) / sigma) * sigma ** (-K)
            checked += 1
            if sp.simplify(sp.expand(exact - claim)) != 0:
                bad.append(("mixed_dk", 1, K, b))

    print(f"checked {checked} forms, {len(bad)} disagree")
    for item in bad:
        print("  disagrees:", item)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
