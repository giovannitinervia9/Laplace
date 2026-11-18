"""Closed forms transcribed into include/laplace/math/dawson.hpp, derived and
checked with SymPy.

Dawson's integral is

    F(x) = exp(-x^2) int_0^x exp(s^2) ds,

odd, bounded, with F'(x) = 1 - 2 x F(x). Two series are used.

1. The scaled series. Write A_K(x) = exp(x^2) F^(K)(x), so that
   F^(K)(x) = exp(-x^2) A_K(x). From A_(K+1) = A_K' - 2 x A_K and
   A_0(x) = int_0^x exp(s^2) ds = sum_m x^(2m+1) / (m! (2m+1)), every A_K is a
   power series in x with powers of one parity (odd for K even, even for K
   odd). Writing the coefficient of x^(2m+1) (K even) or of x^(2m) (K odd) as
   b_K(m)/m!, the recurrence on the coefficients becomes

       K odd :  b_K(m) = (2m + 1) b_(K-1)(m) - 2 m b_(K-1)(m - 1)
       K even:  b_K(m) = 2 b_(K-1)(m + 1) - 2 b_(K-1)(m)

   and it closes in the form (checked here for K <= 6)

       b_0(m) =     1 / (2m+1)
       b_1(m) =    -1 / (2m-1)
       b_2(m) =     4 / ((2m-1)(2m+1))
       b_3(m) =   -12 / ((2m-3)(2m-1))
       b_4(m) =    96 / ((2m-3)(2m-1)(2m+1))
       b_5(m) =  -480 / ((2m-5)(2m-3)(2m-1))
       b_6(m) =  5760 / ((2m-5)(2m-3)(2m-1)(2m+1))

   with no exceptional leading coefficients: one formula covers every m >= 0,
   and no denominator vanishes, since 2m - j is odd for odd j. Every b_K has
   the sign (-1)^K from the first m at which all its factors are positive on,
   so the tail of each series has one sign and nothing cancels inside it.

2. The asymptotic series. F(x) ~ sum_m (2m-1)!! / (2^(m+1) x^(2m+1)),
   differentiated term by term,

       F^(K)(x) ~ (-1)^K sum_m ((2m-1)!! / 2^(m+1)) (2m+1)_K x^(-(2m+1+K)),

   with (2m+1)_K = (2m+1)(2m+2)...(2m+K) the rising factorial. Every term has
   the sign (-1)^K, so again nothing cancels.

Usage (from the repository root):  python -I tools/reference/dawson_forms.py
"""
import sympy as sp

x, s = sp.symbols("x s", real=True)
m = sp.symbols("m", integer=True, nonnegative=True)

# Dawson's integral, from its definition through erfi
F = sp.sqrt(sp.pi) / 2 * sp.exp(-x**2) * sp.erfi(x)

B = {
    0: 1 / (2 * m + 1),
    1: -sp.Integer(1) / (2 * m - 1),
    2: 4 / ((2 * m - 1) * (2 * m + 1)),
    3: -12 / ((2 * m - 3) * (2 * m - 1)),
    4: 96 / ((2 * m - 3) * (2 * m - 1) * (2 * m + 1)),
    5: -480 / ((2 * m - 5) * (2 * m - 3) * (2 * m - 1)),
    6: 5760 / ((2 * m - 5) * (2 * m - 3) * (2 * m - 1) * (2 * m + 1)),
}
KMAX = 6
NTERMS = 9  # coefficients of each series that are compared


def check_defining_identity():
    """F'(x) = 1 - 2 x F(x) and F is odd."""
    assert sp.simplify(sp.diff(F, x) - (1 - 2 * x * F)) == 0
    assert sp.simplify(F.subs(x, -x) + F) == 0
    print("F'(x) = 1 - 2 x F(x) and F(-x) = -F(x): both identities hold")


def check_coefficient_recurrence():
    """The two recurrences on b_K, as they are used to close the forms."""
    for k in range(1, KMAX + 1):
        if k % 2 == 1:
            rec = (2 * m + 1) * B[k - 1] - 2 * m * B[k - 1].subs(m, m - 1)
        else:
            rec = 2 * B[k - 1].subs(m, m + 1) - 2 * B[k - 1]
        assert sp.simplify(rec - B[k]) == 0, k
    print("b_K recurrence: the closed forms of b_1 ... b_6 follow from b_0")


def check_scaled_series():
    """The coefficients of exp(x^2) F^(K)(x) against b_K(m)/m!."""
    for k in range(KMAX + 1):
        a = sp.exp(x**2) * sp.diff(F, x, k)
        ser = sp.series(a, x, 0, 2 * NTERMS + 2).removeO().expand()
        for j in range(NTERMS):
            n = 2 * j + 1 if k % 2 == 0 else 2 * j
            got = sp.nsimplify(sp.simplify(ser.coeff(x, n)))
            want = sp.simplify(B[k].subs(m, j) / sp.factorial(j))
            assert sp.simplify(got - want) == 0, (k, j, got, want)
        print(f"  K = {k}: {NTERMS} coefficients of exp(x^2) F^({k}) match b_{k}(m)/m!")
    print("scaled series: every coefficient checked against SymPy's own derivative")


def check_asymptotic_series():
    """The differentiated asymptotic series, term by term."""

    for k in range(KMAX + 1):
        for j in range(NTERMS):
            # term j of the series for F, differentiated k times
            term = sp.factorial2(2 * j - 1) / (2 ** (j + 1) * x ** (2 * j + 1))
            want = sp.diff(term, x, k)
            got = ((-1) ** k * sp.factorial2(2 * j - 1) / 2 ** (j + 1)
                   * sp.rf(2 * j + 1, k) * x ** (-(2 * j + 1 + k)))
            assert sp.simplify(got - want) == 0, (k, j)
    print(f"asymptotic series: {NTERMS} terms of every order match the term-by-term derivative")

    # The expansion is asymptotic, not convergent, so SymPy cannot read its
    # coefficients off a Taylor series at infinity; the truncated series is
    # compared with F itself instead, at three arguments and to 30 digits.
    for xv in (6, 8, 12, 30):
        approx = sum(sp.factorial2(2 * j - 1) / sp.Integer(2) ** (j + 1) / sp.Integer(xv) ** (2 * j + 1)
                     for j in range(NTERMS))
        exact = sp.N(F.subs(x, xv), 40)
        print(f"  {NTERMS} terms at x = {xv:>2}: relative error {sp.N(abs(approx - exact) / exact, 3)}")


def check_special_values():
    """The values used by the test: F^(K)(0) and the maximum of F."""
    for k in range(KMAX + 1):
        at0 = sp.simplify(sp.diff(F, x, k).subs(x, 0))
        want = 0 if k % 2 == 0 else sp.simplify(B[k].subs(m, 0))
        assert sp.simplify(at0 - want) == 0, (k, at0, want)
        print(f"  F^({k})(0) = {at0}")
    # the single maximum of F on the positive axis: F'(x) = 0, that is 2 x F(x) = 1
    xm = sp.nsolve(sp.diff(F, x), 0.92)
    print(f"maximum of F at x = {sp.N(xm, 16)}, F = {sp.N(F.subs(x, xm), 16)}")


def main():
    check_defining_identity()
    check_coefficient_recurrence()
    check_scaled_series()
    check_asymptotic_series()
    check_special_values()
    print("\nb_K(m), for transcription into dawson.hpp:")
    for k in range(KMAX + 1):
        print(f"  b_{k}(m) = {sp.simplify(B[k])}")


if __name__ == "__main__":
    main()
