"""Check of the closed forms of the Box-Cox bijector (include/laplace/bijector/detail/scalar_box_cox.hpp)
against SymPy derivatives of the defining expressions, for every (order in the argument K or m,
order in lambda n) with total order at most 6.

Usage (from the repository root): python -I tools/reference/bijector_box_cox_forms.py

For each form, the closed form is evaluated in mpmath at 60 digits at random points and compared
with the SymPy derivative of the defining expression, evaluated at the same point. The forms are
those of the header's top comment: g = L phi(lambda L) (phi = (e^y - 1)/y, SymPy-differentiated),
h = exp(S), S = u psi(lambda u) with psi = log(1 + y)/y, the Bell recursion for the lambda
derivatives of h, the Leibniz form of the P_K jets, and the two log-Jacobians. Prints the largest
relative difference per form; a value below 1e-40 means the closed form is right.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402
import sympy as sp  # noqa: E402

mp.mp.dps = 60
X, LAM = sp.symbols("x lam", real=True)  # X = v for g, u for h
Y = sp.Symbol("y", real=True)
KMAX = 6

# defining expressions and their SymPy derivatives
L = sp.log(X)
g_def = L * (sp.exp(LAM * L) - 1) / (LAM * L)     # (v^lambda - 1)/lambda with the removable point cancelled
h_def = (1 + LAM * X) ** (1 / LAM)
phi = (sp.exp(Y) - 1) / Y                          # exprel
psi = sp.log(1 + Y) / Y                            # log1prel
S_def = sp.log(1 + LAM * X) / LAM                  # log h
LJg_def = (LAM - 1) * L                            # log g'
LJh_def = (1 - LAM) * S_def                        # log h'

phi_d = [sp.lambdify(Y, sp.diff(phi, Y, k), "mpmath") for k in range(7)]
psi_d = [sp.lambdify(Y, sp.diff(psi, Y, k), "mpmath") for k in range(7)]


def rnd(rng, lo, hi):
    return mp.mpf(rng.uniform(lo, hi))


def binom(n, k):
    return mp.binomial(n, k)


def falling(a, k):
    r = mp.mpf(1)
    for i in range(k):
        r *= a - i
    return r


def rising(a, k):
    r = mp.mpf(1)
    for i in range(k):
        r *= a + i
    return r


def sgn(k):
    return -1 if k % 2 else 1


# ---------------------------------------------------------------- forward g
def G_closed(m, n, v, lam):
    Lv = mp.log(v)
    if m == 0 and n == 0:
        y = lam * Lv
        return Lv * mp.expm1(y) / y if y != 0 else Lv
    if n == 0:
        return falling(lam - 1, m - 1) * v ** (lam - m)
    if m == 0:
        return Lv ** (n + 1) * phi_d[n](lam * Lv)
    j = m - 1
    e = [mp.mpf(1)] + [mp.mpf(0)] * 6
    for r in range(j):
        c = lam - 1 - r
        for k in range(r + 1, 0, -1):
            e[k] += c * e[k - 1]
    s = mp.mpf(0)
    for p in range(0, min(j, n) + 1):
        s += e[j - p] * falling(n, p) * Lv ** (n - p)
    return v ** (lam - m) * s


def LJg_closed(m, n, v, lam):
    Lv = mp.log(v)
    if n >= 2:
        return mp.mpf(0)
    if m == 0:
        return (lam - 1) * Lv if n == 0 else Lv
    d = sgn(m - 1) * mp.factorial(m - 1) * v ** (-m)
    return (lam - 1) * d if n == 0 else d


# ---------------------------------------------------------------- inverse h
def S_jet_closed(K, n, u, lam):
    x = lam * u
    if K == 0:
        return u ** (n + 1) * psi_d[n](x)
    t = 1 / (1 + x)
    s = mp.mpf(0)
    for j in range(0, min(n, K - 1) + 1):
        q = n - j
        s += (binom(n, j) * falling(K - 1, j) * lam ** (K - 1 - j) * sgn(q) * rising(K, q) * u ** q * t ** (K + q))
    return sgn(K - 1) * mp.factorial(K - 1) * s


def PK_jets(K, lam):
    P = [mp.mpf(1)] + [mp.mpf(0)] * 6
    for k in range(1, K):
        f = 1 - k * lam
        for j in range(6, -1, -1):
            P[j] = f * P[j] - (k * j * P[j - 1] if j > 0 else 0)
    return P


def H_closed(K, n, u, lam):
    x = lam * u
    S = u * psi_d[0](x)
    A = [mp.mpf(1)] + [mp.mpf(0)] * 7
    xs = [mp.mpf(0)] * 8
    for j in range(1, n + 1):
        xs[j] = S_jet_closed(0, j, u, lam)
    for i in range(n):
        A[i + 1] = sum(binom(i, j) * xs[j + 1] * A[i - j] for j in range(0, i + 1))
    if K == 0:
        return mp.exp(S) * A[n]
    P = PK_jets(K, lam)
    t = 1 / (1 + x)
    s = mp.mpf(0)
    for j in range(0, n + 1):
        for i in range(0, n - j + 1):
            q = n - j - i
            s += binom(n, j) * binom(n - j, i) * P[j] * A[i] * sgn(q) * rising(K, q) * u ** q * t ** q
    return mp.exp((1 - K * lam) * S) * s


def LJh_closed(K, n, u, lam):
    if n == 0:
        return (1 - lam) * S_jet_closed(K, 0, u, lam)
    return (1 - lam) * S_jet_closed(K, n, u, lam) - n * S_jet_closed(K, n - 1, u, lam)


# ---------------------------------------------------------------- the checks
def sympy_partial(expr, var, n_var, n_lam):
    e = expr
    if n_var:
        e = sp.diff(e, var, n_var)
    if n_lam:
        e = sp.diff(e, LAM, n_lam)
    return sp.lambdify((X, LAM), e, "mpmath")


def check(name, defn, domain, closed, pairs):
    """domain 'v': the argument is positive, log-uniform over 2^-30..2^30; 'u': inside the image of h."""
    worst = mp.mpf(0)
    rng = random.Random(7)
    for (m, n) in pairs:
        f = sympy_partial(defn, X, m, n)
        for _ in range(12):
            if domain == "v":
                v = mp.mpf(2) ** rng.uniform(-30, 30)
                lam = rnd(rng, -3, 3)
            else:
                lam = rnd(rng, -2, 2)
                v = rnd(rng, -2, 2)
                if 1 + lam * v <= 0:
                    continue
            ref = f(v, lam)
            got = closed(m, n, v, lam)
            scale = max(abs(ref), mp.mpf(10) ** -50)
            rel = abs(got - ref) / scale
            worst = max(worst, rel)
    print(f"{name:14s} order pairs {len(pairs):2d}  largest relative difference {mp.nstr(worst, 3)}")
    return worst


# ------------------------------------------------- the limits at the ends of the two domains
#
# Every closed form above is written from exprel or log1prel so that lambda = 0 needs no branch
# of its own, and at an infinite argument those two primitives are themselves evaluated at an
# infinite argument, where each of them is inf/inf; at a finite end a vanishing prefactor meets
# an infinite power. Either way the form gives NaN although the limit exists. The C++
# bc_fwd_at_end, bc_fwd_lj_at_end, bc_inv_at_end and bc_inv_lj_at_end
# (include/laplace/bijector/detail/scalar_box_cox.hpp) return these limits instead, and the
# rules below are those functions, compared here with SymPy's limit of the definition.
#
# The ends. g has domain (0, inf), so its ends are v = 0 and v = +inf whatever lambda is. The
# domain of h is the image of g: (-1/lambda, inf) for lambda > 0, so the ends are u = -1/lambda
# and u = +inf; (-inf, -1/lambda) for lambda < 0, so they are u = -inf and u = -1/lambda; the
# whole line for lambda = 0, where there is no finite end.

ENDS_FWD = ("zero", "pinf")


def bc_ends_inv(l):
    if l > 0:
        return ("edge", "pinf")
    if l < 0:
        return ("ninf", "edge")
    return ("ninf", "pinf")


def bc_definition(q, l):
    """The definition of q with lambda = l, in the limiting form at lambda = 0, where the
    Box-Cox quotient is 0/0."""
    g = sp.log(X) if l == 0 else (X ** l - 1) / l
    h = sp.exp(X) if l == 0 else (1 + l * X) ** (sp.Integer(1) / l)
    if q == "fwd":
        return g
    if q == "inv":
        return h
    if q == "fwd_lj":
        return sp.log(sp.diff(g, X))
    return sp.log(sp.diff(h, X))


def bc_limit_rule(q, end, n, l):
    """The limit of the n-th argument derivative of q at the end `end`, as the C++ computes it.
    The argument enters every closed form only through a power of v or of w = 1 + lambda u, so
    the sign of that power's exponent decides: the power runs to +inf, dies, or is 1 on the
    knife edge where the exponent vanishes, and there the prefactor is the whole limit."""
    if q == "fwd":
        if n == 0:  # the ends of the image of g
            if end == "zero":
                return -1 / l if l > 0 else -sp.oo
            return -1 / l if l < 0 else sp.oo
        pref = sp.Integer(1)  # F_n = prod_(k=1..n-1) (lambda - k)
        for k in range(1, n):
            pref *= l - k
        e = l - n  # the exponent of v, zero exactly where lambda is the integer n
        grow = end == "pinf"
    elif q == "fwd_lj":
        s = l - 1
        if s == 0:  # the identity: log|g'| is 0 and so is every derivative of it
            return sp.Integer(0)
        if n == 0:
            s0 = -s if end == "zero" else s
            return sp.oo if s0 > 0 else -sp.oo
        if end == "pinf":  # every derivative carries v^(-n) and dies
            return sp.Integer(0)
        s1 = s * (-1) ** (n - 1)
        return sp.oo if s1 > 0 else -sp.oo
    elif q == "inv":
        if end == "ninf":  # w runs to +inf with a negative exponent: h and every order die
            return sp.Integer(0)
        pref = sp.Integer(1)  # P_n = prod_(r=1..n-1) (1 - r lambda)
        for r in range(1, n):
            pref *= 1 - r * l
        if end == "edge" and l < 0:  # the exponent 1/lambda - n is negative and P_n > 0
            return sp.oo
        e = 1 - n * l  # the sign of 1/lambda - n for lambda >= 0, the product compared with 1
        grow = end == "pinf"
    else:  # inv_lj = (1 - lambda) S, S = log(1 + lambda u)/lambda (u itself at lambda = 0)
        if l == 1:  # the identity
            return sp.Integer(0)
        if n == 0:
            if end == "pinf":
                s = 1 - l  # S runs to +inf
            elif end == "ninf":
                s = l - 1  # S runs to -inf, and 1 - lambda > 0 there
            else:
                s = (l - 1) if l > 0 else (1 - l)  # S runs to -inf above 0 and to +inf below
            return sp.oo if s > 0 else -sp.oo
        if end != "edge":  # d^n_u S carries w^(-n) and dies, log|h'| being u at lambda = 0
            if l == 0:
                return sp.Integer(1) if n == 1 else sp.Integer(0)
            return sp.Integer(0)
        s = (1 - l) * (-1) ** (n - 1) * l ** (n - 1)
        return sp.oo if s > 0 else -sp.oo
    if e == 0:
        return pref
    to_inf = (e > 0) if grow else (e < 0)
    if not to_inf:
        return sp.Integer(0)
    return sp.oo if pref > 0 else (-sp.oo if pref < 0 else sp.Integer(0))


def check_limits():
    """Each rule against SymPy's limit of the definition, for every lambda and every order."""
    lams = [sp.Rational(-3, 2), sp.Integer(-1), sp.Rational(-1, 2), sp.Rational(-1, 3),
            sp.Rational(-1, 5), sp.Integer(0), sp.Rational(1, 6), sp.Rational(1, 5),
            sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3),
            sp.Rational(7, 10), sp.Integer(1), sp.Rational(3, 2), sp.Integer(2),
            sp.Rational(5, 2), sp.Integer(3), sp.Integer(4), sp.Integer(5), sp.Integer(6),
            sp.Integer(7)]
    count = 0
    for q in ("fwd", "fwd_lj", "inv", "inv_lj"):
        inverse = q in ("inv", "inv_lj")
        for l in lams:
            for end in (bc_ends_inv(l) if inverse else ENDS_FWD):
                d = bc_definition(q, l)
                point, side = {"zero": (sp.Integer(0), "+"), "pinf": (sp.oo, "-"),
                               "ninf": (-sp.oo, "+")}.get(end, (None, None))
                if end == "edge":
                    point, side = -1 / l, ("+" if l > 0 else "-")
                for n in range(0, KMAX + 1):
                    e = sp.diff(d, X, n) if n else d
                    ref = sp.limit(sp.simplify(e), X, point, side)
                    got = bc_limit_rule(q, end, n, l)
                    same = ref == got if ref.is_infinite or got.is_infinite else sp.simplify(ref - got) == 0
                    if not same:
                        raise SystemExit(f"limit differs: {q} at {end} lambda={l} n={n}: "
                                         f"rule {got} vs SymPy {ref}")
                    count += 1
    print(f"checked {count} limits at the ends of the two domains against SymPy's limit of the definition")


def main():
    pairs = [(m, n) for m in range(0, KMAX + 1) for n in range(0, KMAX + 1 - m) if m + n >= 1]
    ok = True
    ok &= check("g", g_def, "v", G_closed, pairs) < mp.mpf("1e-40")
    ok &= check("log g'", LJg_def, "v", LJg_closed, pairs) < mp.mpf("1e-40")
    ok &= check("h", h_def, "u", H_closed, pairs) < mp.mpf("1e-40")
    ok &= check("log h'", LJh_def, "u", LJh_closed, pairs) < mp.mpf("1e-40")
    print("all closed forms agree" if ok else "MISMATCH")
    check_limits()


# ------------------------------- the partials at the ends of the two domains
#
# out[n1] is d^n0_x d^n1_lambda with n0 = K - n1, the entries of bijector_partials<K> with n2 = 0.
# The rules below are the C++ bc_fwd_partials_at_end, bc_fwd_lj_partials_at_end,
# bc_inv_partials_at_inf, bc_inv_partials_at_edge, bc_inv_lj_partials_at_inf and
# bc_inv_lj_partials_at_edge, compared here with SymPy's own limit of the mixed derivative of the
# definition.


def bc_sym_prod(expr):
    """the elementary symmetric polynomials e_k of c_r = lambda - 1 - r, r = 0..j-1, as a list"""
    def build(j, l):
        e = [sp.Integer(1)] + [sp.Integer(0)] * 7
        for r in range(j):
            c = l - 1 - r
            for k in range(r + 1, 0, -1):
                e[k] += c * e[k - 1]
        return e
    return build


def bc_partials_rule(q, end, K, n1, l):
    n0 = K - n1
    if q == "fwd":
        if n0 == 0:  # the lambda-derivatives of the end of the image
            finite = (l > 0) if end == "zero" else (l < 0)
            s = sp.Integer(1) if (n1 % 2) else sp.Integer(-1)  # (-1)^(n1+1)
            if finite:
                return s * sp.factorial(n1) / l ** (n1 + 1)
            return (sp.oo if s > 0 else -sp.oo) if end == "zero" else sp.oo
        e = bc_sym_prod(None)(n0 - 1, l)
        j = n0 - 1
        p0 = None
        for p in range(0, min(j, n1) + 1):
            if e[j - p] != 0:
                p0 = p
                break
        if p0 is None:
            return sp.Integer(0)
        coef = e[j - p0]
        for i in range(p0):
            coef *= (n1 - i)
        sign = -coef if (end == "zero" and ((n1 - p0) % 2)) else coef
        if l == n0:
            if p0 == n1:
                return coef
            return sp.oo if sign > 0 else -sp.oo
        if l > n0:  # the exponent is positive
            return (sp.oo if sign > 0 else -sp.oo) if end == "pinf" else sp.Integer(0)
        return sp.Integer(0) if end == "pinf" else (sp.oo if sign > 0 else -sp.oo)
    if q == "fwd_lj":
        if n1 >= 2:
            return sp.Integer(0)
        if n1 == 1:
            if n0 == 0:
                return -sp.oo if end == "zero" else sp.oo
            if end == "pinf":
                return sp.Integer(0)
            return sp.oo if ((n0 - 1) % 2 == 0) else -sp.oo
        if l == 1:
            return sp.Integer(0)
        if n0 == 0:
            s = 1 - l if end == "zero" else l - 1
            return sp.oo if s > 0 else -sp.oo
        if end == "pinf":
            return sp.Integer(0)
        s = (l - 1) * (1 if ((n0 - 1) % 2 == 0) else -1)
        return sp.oo if s > 0 else -sp.oo
    if q == "inv":
        if end == "ninf":
            return sp.Integer(0)
        if end == "pinf":
            pref = sp.Integer(1)
            for r in range(1, n0):
                pref *= 1 - r * l
            if n1 % 2:
                pref = -pref
            na = n0 * l
            if na == 1 and n1 == 0:
                return pref
            if na <= 1:
                return sp.oo if pref > 0 else (-sp.oo if pref < 0 else sp.Integer(0))
            return sp.Integer(0)
        # the finite end
        if l < 0:
            es = -1
        else:
            kl = K * l
            es = 0 if kl == 1 else (1 if kl < 1 else -1)
        if es == 0:
            pk = sp.Integer(1)
            for r in range(1, K):
                pk *= 1 - r * l
            return pk * (-1 / l ** 2) ** n1
        if es > 0:
            return sp.Integer(0)
        M = 0
        for r in range(1, K):
            if r * l == 1:
                M = r
                break
        if M > 0:
            if n1 == 0:
                return sp.Integer(0)
            return -sp.oo if ((n0 + M) % 2) else sp.oo
        s = sp.Integer(1)
        for r in range(1, n0):
            s *= 1 - r * l
        if n1 % 2:
            s = -s
        for i in range(n1):
            if (n0 + i) * l > 1:
                s = -s
        return sp.oo if s > 0 else -sp.oo
    # inv_lj
    if end in ("pinf", "ninf"):
        if l == 0:
            deg = n1 + 1 - n0
            if deg < 0:
                return sp.Integer(0)
            if deg == 0:
                f = sp.factorial(n1)
                return (-1 if (n1 % 2) else 1) * f * f
            s = -1 if (n1 % 2) else 1
            if end == "ninf" and (deg % 2):
                s = -s
            return sp.oo if s > 0 else -sp.oo
        if n0 >= 1:
            return sp.Integer(0)
        c = (1 - l) / l if n1 == 0 else ((-1) ** n1) * sp.factorial(n1) / l ** (n1 + 1)
        return sp.oo if c > 0 else (-sp.oo if c < 0 else sp.Integer(0))
    if n1 == 0:  # the pure log-Jacobian at the finite end
        return bc_limit_rule("inv_lj", "edge", n0, l)
    if l == 1:
        return -sp.oo if (n0 % 2) else sp.oo
    s = (l - 1) * (-1 if (n0 % 2) else 1)
    if l < 0 and ((n1 + 1 - n0) % 2):
        s = -s
    return sp.oo if s > 0 else -sp.oo


def bc_definition_sym(q):
    """The definition with lambda left SYMBOLIC, which the partials need: substituting lambda
    before differentiating in it leaves an expression without lambda, whose lambda-derivatives are
    identically zero. The degenerate lambda = 0 is reached afterwards as a limit, the Box-Cox
    quotient being 0/0 there."""
    g = (X ** LAM - 1) / LAM
    h = (1 + LAM * X) ** (1 / LAM)
    if q == "fwd":
        return g
    if q == "inv":
        return h
    if q == "fwd_lj":
        return sp.log(sp.diff(g, X))
    return sp.log(sp.diff(h, X))


def check_partials_limits(kmax=6):
    """Each partials rule against SymPy's limit of the mixed derivative of the definition.
    Free invariant: the n1 = 0 column must agree with the pure rule checked above."""
    lams = [sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Integer(0), sp.Rational(1, 6),
            sp.Rational(1, 4), sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3),
            sp.Integer(1), sp.Rational(3, 2), sp.Integer(2), sp.Integer(3)]
    count = 0
    for q in ("fwd", "fwd_lj", "inv", "inv_lj"):
        inverse = q in ("inv", "inv_lj")
        for l in lams:
            for end in (bc_ends_inv(l) if inverse else ENDS_FWD):
                d = bc_definition_sym(q)
                point, side = {"zero": (sp.Integer(0), "+"), "pinf": (sp.oo, "-"),
                               "ninf": (-sp.oo, "+")}.get(end, (None, None))
                if end == "edge":
                    point, side = -1 / l, ("+" if l > 0 else "-")
                for K in range(1, kmax + 1):
                    for n1 in range(0, K + 1):
                        n0 = K - n1
                        e = sp.diff(d, X, n0) if n0 else d
                        e = sp.diff(e, LAM, n1) if n1 else e
                        e = sp.limit(e, LAM, 0) if l == 0 else e.subs(LAM, l)
                        ref = sp.limit(sp.simplify(e), X, point, side)
                        got = bc_partials_rule(q, end, K, n1, l)
                        same = (ref == got if ref.is_infinite or got.is_infinite
                                else sp.simplify(ref - got) == 0)
                        if not same:
                            raise SystemExit(f"partial differs: {q} at {end} lambda={l} K={K} "
                                             f"n1={n1} (n0={n0}): rule {got} vs SymPy {ref}")
                        if n1 == 0:  # the invariant against the pure rule
                            pure = bc_limit_rule(q, end, K, l)
                            ok = (pure == got if pure.is_infinite or got.is_infinite
                                  else sp.simplify(pure - got) == 0)
                            if not ok:
                                raise SystemExit(f"n1 = 0 disagrees with the pure rule: {q} {end} "
                                                 f"lambda={l} K={K}: {got} vs {pure}")
                        count += 1
    print(f"checked {count} partials limits at the ends against SymPy's limit of the definition")


if __name__ == "__main__":
    # no argument: the closed forms and the pure limits; "partials [kmax]": the partials limits
    if len(sys.argv) > 1 and sys.argv[1] == "partials":
        check_partials_limits(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        main()
