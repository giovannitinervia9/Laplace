"""Coefficients of the asymptotic expansion of the polygamma functions, as
double-doubles, for include/laplace/math/polygamma_dd.hpp.

    n = 0:   b_(0,k) = B_2k / (2k)                     psi(z) = log z - 1/(2z) - sum_k b_(0,k) z^-2k
    n >= 1:  b_(n,k) = B_2k (2k + n - 1)! / (2k)!      psi_n(z) = (-1)^(n+1) z^-n [ (n-1)! + n!/(2z)
                                                                  + sum_k b_(n,k) z^-2k ]

for k = 1..36 and n = 0..13. The Bernoulli numbers are SymPy's exact rationals,
so each coefficient is an exact rational number, rounded once to a double (hi)
and once more for the remainder (lo).

Usage (from the repository root):
    python -I tools/reference/polygamma_dd_coefficients.py

prints the body of LAPLACE_POLYGAMMA_DD_COEF, which is pasted into the header.
36 terms: at the shift threshold z = 20 order 13 needs 33 for a truncation
below 2^-107 of its leading term, order 0 needs 16 (see the docstring of
polygamma_dd.py and docs/numerics.md).
"""
import sympy

K = 36
N = 13


def split(q):
    hi = float(q)
    lo = float(q - sympy.Rational(hi))
    return hi, lo


def main():
    for n in range(N + 1):
        row = []
        for k in range(1, K + 1):
            b = sympy.bernoulli(2 * k)
            if n == 0:
                c = b / (2 * k)
            else:
                c = b * sympy.factorial(2 * k + n - 1) / sympy.factorial(2 * k)
            hi, lo = split(sympy.Rational(c))
            row.append("%s, %s" % (hi.hex(), lo.hex()))
        print("  /* n = %d */ " % n + ", ".join(row) + (", \\" if n < N else ""))


if __name__ == "__main__":
    main()
