"""Closed forms of the Yeo-Johnson bijector, derived and checked with SymPy, and
the C++ of include/laplace/bijector/detail/scalar_yeo_johnson.hpp written from them.

Usage (from the repository root):  python -I tools/reference/bijector_yeo_johnson_forms.py

Notation: x is the argument (v for the forward map g, u for the inverse h),
p1 = lambda the exponent, c = 2 - lambda, M = log(1 + v) and N = log(1 - v) for
the forward map, and for the inverse, a = -u on the negative branch. Two
functions of one variable are the primitives of the header:
  exprel(z) = (e^z - 1)/z           (laplace::math::exprel, its derivatives exprel_dK)
  log1prel(z) = log(1 + z)/z        (laplace::math::log1prel, its derivatives log1prel_dK)
Each is written with its derivatives as SymPy functions EXj, LRj (EXj' = EX(j+1)).

Definitions (the reference of tools/reference/bijector_yeo_johnson.py):
  g(v) = ((1 + v)^lambda - 1)/lambda                     v >= 0
  g(v) = -((1 - v)^(2 - lambda) - 1)/(2 - lambda)        v < 0
  h(u) = (1 + lambda u)^(1/lambda) - 1                   u >= 0
  h(u) = 1 - (1 - (2 - lambda) u)^(1/(2 - lambda))       u < 0

Closed forms (n >= 1 is the order of the x-derivative; the mixed derivatives
of order (n0, n1) are the n1-th lambda-derivatives of the n0-th x-derivative):
  forward, v >= 0:  g = M exprel(lambda M);  g^(n) = prod_(k=1..n-1) (lambda - k) (1+v)^(lambda - n)
  forward, v < 0:   g = -N exprel(c N);      g^(n) = (-1)^(n-1) prod_(k=0..n-2) (1 - lambda - k) (1-v)^(c - n)
  inverse, u >= 0:  h = expm1(u log1prel(lambda u));
                    h^(n) = prod_(k=1..n-1) (1 - k lambda) exp(u (1 - n lambda) log1prel(lambda u))
  inverse, u < 0:   h = -expm1(a log1prel(c a)), a = -u;
                    h^(n) = prod_(k=0..n-2) (1 - lambda + k c) exp(u (1 - lambda + (n-1) c) log1prel(c a))
  log|g'|, v >= 0:  (lambda - 1) M;  n >= 1: (lambda - 1) (-1)^(n-1) (n-1)! (1+v)^(-n)
  log|g'|, v < 0:   (1 - lambda) N;  n >= 1: -(1 - lambda) (n-1)! (1-v)^(-n)
  log|h'|, u >= 0:  (1 - lambda) u log1prel(lambda u);
                    n >= 1: (1 - lambda) (-1)^(n-1) (n-1)! lambda^(n-1) (1 + lambda u)^(-n)
  log|h'|, u < 0:   (1 - lambda) u log1prel(c a);
                    n >= 1: (1 - lambda) (n-1)! c^(n-1) (1 - c u)^(-n)

Every form is regular at lambda = 0 and at lambda = 2 (the Box-Cox quotients
are inside exprel and log1prel), so no limit is taken anywhere. The lambda
dependence of each form is a polynomial times an exponential of a product of
lambda with exprel or log1prel, so the lambda-derivatives are obtained by
SymPy's chain rule, one primitive derivative per order. Each closed form and
each mixed derivative is checked here numerically against SymPy's derivative of
the definition; the header is then written from these forms (--header, the
default, writes it; --check only checks).
"""
import math
import random
import sys
from functools import reduce
from pathlib import Path

import mpmath as mp
import sympy as sp

x, lam = sp.symbols("x p1", real=True)
ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "laplace" / "bijector" / "detail" / "scalar_yeo_johnson.hpp"
KMAX = 6


def family(prefix, nmax):
    """SymPy functions prefix0 ... prefixN, with prefix_j' = prefix_(j+1)."""
    classes = []

    def make(j):
        def fdiff(self, argindex=1):
            return classes[j + 1](self.args[0])

        return fdiff

    for j in range(nmax):
        classes.append(type(f"{prefix}{j}", (sp.Function,), {"nargs": 1, "fdiff": make(j)}))
    return classes


class LOG1P(sp.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return 1 / (1 + self.args[0])


class EM1(sp.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return sp.exp(self.args[0])


class POW1P(sp.Function):
    """POW1P(z, s) = (1 + z)^s, printed as pow(1 + z, s): the power of 1 + z itself, not
    exp(s log1p z), whose rounding error grows with s log1p z (the argument of the power is
    exact up to the rounding of 1 + z, which is the one-ulp perturbation of z that ulpC measures)"""
    nargs = 2

    def fdiff(self, argindex=1):
        z, s = self.args
        if argindex == 1:
            return s * POW1P(z, s - 1)
        return LOG1P(z) * POW1P(z, s)


EX = family("EX", KMAX + 2)  # EX[j] = exprel^(j)
LR = family("LR", KMAX + 2)  # LR[j] = log1prel^(j)


def pr(terms):
    return reduce(lambda a, b: a * b, terms, sp.Integer(1))


def base(q, b, n):
    """n = 0: the value; n >= 1: the closed form of the n-th x-derivative."""
    c = 2 - lam
    if q == "fwd" and b == "pos":
        M = LOG1P(x)
        if n == 0:
            return M * EX[0](lam * M)
        return pr(lam - k for k in range(1, n)) * POW1P(x, lam - n)
    if q == "fwd" and b == "neg":
        N = LOG1P(-x)
        if n == 0:
            return -N * EX[0](c * N)
        return (-1) ** (n - 1) * pr(1 - lam - k for k in range(0, n - 1)) * POW1P(-x, c - n)
    if q == "inv" and b == "pos":
        Lz = LR[0](lam * x)
        if n == 0:
            return EM1(x * Lz)
        return pr(1 - k * lam for k in range(1, n)) * sp.exp(x * (1 - n * lam) * Lz)
    if q == "inv" and b == "neg":
        a = -x
        if n == 0:
            return -EM1(a * LR[0](c * a))
        return pr(1 - lam + k * c for k in range(0, n - 1)) * sp.exp(
            x * (1 - lam + (n - 1) * c) * LR[0](c * a))
    if q == "fwd_lj" and b == "pos":
        M = LOG1P(x)
        if n == 0:
            return (lam - 1) * M
        return (lam - 1) * (-1) ** (n - 1) * math.factorial(n - 1) * (1 + x) ** (-n)
    if q == "fwd_lj" and b == "neg":
        N = LOG1P(-x)
        if n == 0:
            return (1 - lam) * N
        return -(1 - lam) * math.factorial(n - 1) * (1 - x) ** (-n)
    if q == "inv_lj" and b == "pos":
        if n == 0:
            return (1 - lam) * x * LR[0](lam * x)
        return (1 - lam) * (-1) ** (n - 1) * math.factorial(n - 1) * lam ** (n - 1) * (1 + lam * x) ** (-n)
    if q == "inv_lj" and b == "neg":
        if n == 0:
            return (1 - lam) * x * LR[0](-c * x)
        return (1 - lam) * math.factorial(n - 1) * c ** (n - 1) * (1 - c * x) ** (-n)
    raise ValueError((q, b, n))


# ---------------------------------------------------------------- definitions (reference)

DEF_G = {"pos": ((1 + x) ** lam - 1) / lam, "neg": -((1 - x) ** (2 - lam) - 1) / (2 - lam)}
DEF_H = {"pos": (1 + lam * x) ** (1 / lam) - 1, "neg": 1 - (1 - (2 - lam) * x) ** (1 / (2 - lam))}


def definition(q, b):
    if q == "fwd":
        return DEF_G[b]
    if q == "inv":
        return DEF_H[b]
    if q == "fwd_lj":
        return sp.log(sp.diff(DEF_G[b], x))
    return sp.log(sp.diff(DEF_H[b], x))


# ---------------------------------------------------------------- checks

def numeric_namespace():
    """mpmath versions of the primitives, for the check only: exprel_j(z) = int_0^1 t^j e^(z t) dt
    and log1prel_j(z) = (-1)^j j! int_0^1 t^j (1 + z t)^(-(j+1)) dt (the integral forms of
    docs/numerics.md, section exprel and log1prel)"""
    ns = {"LOG1P": lambda z: mp.log1p(z), "EM1": lambda z: mp.expm1(z),
          "POW1P": lambda z, s: mp.power(1 + z, s)}
    for j in range(len(EX)):
        ns[EX[j].__name__] = (lambda jj: lambda z: mp.quad(lambda t: t ** jj * mp.exp(z * t), [0, 1]))(j)
    for j in range(len(LR)):
        ns[LR[j].__name__] = (lambda jj: lambda z: (-1) ** jj * mp.factorial(jj)
                              * mp.quad(lambda t: t ** jj * (1 + z * t) ** (-(jj + 1)), [0, 1]))(j)
    return ns


def check():
    """numeric check of every closed form against SymPy's derivative of the definition"""
    mp.mp.dps = 60
    ns = [numeric_namespace(), "mpmath"]
    points = {("fwd", "pos"): (0.7, 0.3), ("fwd", "neg"): (-0.6, 1.4), ("inv", "pos"): (0.8, -0.3),
              ("inv", "neg"): (-0.7, 1.3), ("fwd_lj", "pos"): (0.7, -1.2), ("fwd_lj", "neg"): (-0.6, 0.45),
              ("inv_lj", "pos"): (0.8, -0.3), ("inv_lj", "neg"): (-0.7, 1.3)}
    worst = 0.0
    count = 0
    for (q, b), (x0, l0) in points.items():
        d = definition(q, b)
        for n0 in range(0, 5):
            for n1 in range(0, 5 - n0):
                if n0 + n1 == 0:
                    continue
                e_ref = d
                if n0:
                    e_ref = sp.diff(e_ref, x, n0)
                if n1:
                    e_ref = sp.diff(e_ref, lam, n1)
                e_new = base(q, b, n0)
                if n1:
                    e_new = sp.diff(e_new, lam, n1)
                f_ref = sp.lambdify((x, lam), e_ref, modules="mpmath")
                f_new = sp.lambdify((x, lam), e_new, modules=ns)
                r = f_ref(mp.mpf(x0), mp.mpf(l0))
                s = f_new(mp.mpf(x0), mp.mpf(l0))
                err = abs(s - r) / (1 + abs(r))
                worst = max(worst, float(err))
                count += 1
                if err > mp.mpf(10) ** -40:
                    raise SystemExit(f"closed form differs: {q} {b} n0={n0} n1={n1}: {s} vs {r}")
    # the values at two points of the branches
    for (q, b), (x0, l0) in points.items():
        d = definition(q, b)
        f_ref = sp.lambdify((x, lam), d, modules="mpmath")
        f_new = sp.lambdify((x, lam), base(q, b, 0), modules=ns)
        r, s = f_ref(mp.mpf(x0), mp.mpf(l0)), f_new(mp.mpf(x0), mp.mpf(l0))
        err = abs(s - r) / (1 + abs(r))
        worst = max(worst, float(err))
        count += 1
        if err > mp.mpf(10) ** -40:
            raise SystemExit(f"value differs: {q} {b}: {s} vs {r}")
    print(f"checked {count} closed forms (values and mixed derivatives up to order 4): worst relative difference {worst:.3g}")


# ---------------------------------------------------------------- limits at an infinite argument

# The branch an infinite argument belongs to, and the ends that lie in the closed domain:
# the forward map has domain R, so both ends are always there; the domain of h is the image
# of g, bounded above by -1/lambda below 0 and below by 1/(2 - lambda) above 2, so u = +inf
# is an end of it only for lambda >= 0 and u = -inf only for lambda <= 2.
def end_in_domain(q, upper, l):
    if q in ("fwd", "fwd_lj"):
        return True
    return l >= 0 if upper else l <= 2


def definition_num(q, b, l):
    """The definition of q on branch b with lambda = l, in the limiting form where the
    Box-Cox quotient of the definition is 0/0: lambda = 0 on the positive branch and
    lambda = 2 on the negative one."""
    g = {"pos": sp.log(1 + x), "neg": -sp.log(1 - x)}
    h = {"pos": sp.exp(x) - 1, "neg": 1 - sp.exp(-x)}
    degenerate = (b == "pos" and l == 0) or (b == "neg" and l == 2)
    if not degenerate:
        return definition(q, b).subs(lam, l)
    if q == "fwd":
        return g[b]
    if q == "inv":
        return h[b]
    if q == "fwd_lj":
        return sp.log(sp.diff(g[b], x))
    return sp.log(sp.diff(h[b], x))


def limit_rule(q, upper, n, l):
    """The limit of the n-th x-derivative of q as x -> +oo (upper) or -oo, as the C++
    yj_fwd_at_inf, yj_inv_at_inf, yj_fwd_lj_at_inf and yj_inv_lj_at_inf compute it: the
    argument enters each closed form only through a power of 1 + v, 1 - v, 1 + lambda u or
    1 - c u, c = 2 - lambda, so the sign of that power's exponent decides, and the
    prefactor gives the sign and the value on the knife edge where the exponent vanishes."""
    c = 2 - l
    if q == "fwd":
        if n == 0:  # the ends of the image of g
            if upper:
                return -1 / l if l < 0 else sp.oo
            return 1 / c if l > 2 else -sp.oo
        a = l if upper else c
        pref = sp.Integer(1 if (upper or n % 2 == 1) else -1)
        for k in range(1, n):
            pref *= a - k
        e = a - n
    elif q == "inv":
        if n == 0:
            return sp.oo if upper else -sp.oo
        a = l if upper else c
        pref = sp.Integer(1)
        for k in range(1, n):
            pref *= (1 - k * a) if upper else (k * a - 1)
        e = 1 - n * a
    elif q == "fwd_lj":
        if n >= 1:
            return sp.Integer(0)
        s = l - 1 if upper else 1 - l
        return sp.oo if s > 0 else (-sp.oo if s < 0 else sp.Integer(0))
    else:  # inv_lj
        if n >= 2:
            return sp.Integer(0)
        if n == 1:  # log|h'| is affine in u at lambda = 0 above and at lambda = 2 below
            if upper:
                return sp.Integer(1) if l == 0 else sp.Integer(0)
            return sp.Integer(-1) if l == 2 else sp.Integer(0)
        s = 1 - l if upper else l - 1
        return sp.oo if s > 0 else (-sp.oo if s < 0 else sp.Integer(0))
    if e > 0:
        return sp.oo if pref > 0 else -sp.oo
    if e < 0:
        return sp.Integer(0)
    return pref


# ---------------------------------------------------------------- the partials at an infinite argument
#
# The limits of d^n0_x d^n1_lambda of each quantity, n0 + n1 = K. The argument enters each closed
# form through one power of 1 + v, 1 - v, 1 + lambda u or 1 - c u; a lambda-derivative multiplies
# it by a logarithm of that same quantity, which loses to a negative power and wins against a
# vanishing exponent. So the sign of the exponent decides as it does for the pure derivatives, and
# the knife edge, where the power is 1, carries the prefactor for n1 = 0 and an infinity above it.
#
# The inverse's negative branch is the positive one reflected: h(u) = -h_pos(-u; c) and
# log|h'|(u) = log|h'_pos|(-u; c) with c = 2 - lambda, so d^n0_u d^n1_lambda picks up
# (-1)^(n0+n1+1) for h and (-1)^(n0+n1) for its log-Jacobian, with c in place of lambda.

def prod_range(f, lo, hi):
    p = sp.Integer(1)
    for k in range(lo, hi + 1):
        p *= f(k)
    return p


def inf_sign(s):
    """+oo or -oo by the sign of s, which must not be zero."""
    return sp.oo if s > 0 else -sp.oo


def fwd_partial_limit(upper, n0, n1, l):
    """g: prod_(k=1..n0-1)(a - k) w^(a - n0), a = lambda above and c = 2 - lambda below, with
    (-1)^(n0-1) on the negative branch and (-1)^n1 from d^n1_lambda = (-1)^n1 d^n1_c there.
    At n0 = 0 the value tends to the end of the image, whose lambda-derivatives are those of
    -1/lambda (above, lambda < 0) and of 1/c (below, c < 0), and to an infinity otherwise."""
    a = l if upper else 2 - l
    branch = sp.Integer(1) if upper else sp.Integer(-1) ** (n0 - 1 + n1)
    if n0 == 0:
        if upper:
            if l < 0:  # the lambda-derivatives of the end of the image, -1/lambda
                return sp.diff(-1 / lam, lam, n1).subs(lam, l)
            return sp.oo
        if l > 2:  # the lambda-derivatives of the other end, 1/(2 - lambda)
            return sp.diff(1 / (2 - lam), lam, n1).subs(lam, l)
        return inf_sign(sp.Integer(-1) ** (n1 + 1))
    pref = branch * prod_range(lambda k: a - k, 1, n0 - 1)
    if a > n0:
        return inf_sign(pref)
    if a == n0:
        return pref if n1 == 0 else inf_sign(pref)
    return sp.Integer(0)


def inv_partial_limit_pos(n0, n1, l):
    """h at u = +inf: P_n0 (1 + lambda u)^(1/lambda - n0), P_n0 = prod(1 - k lambda), whose
    lambda-derivatives multiply it by powers of -log(1 + lambda u)/lambda^2, hence (-1)^n1.
    The sign of 1 - n0 lambda is the sign of the exponent, lambda = 0 included."""
    pref = prod_range(lambda k: 1 - k * l, 1, n0 - 1)
    a = 1 - n0 * l
    if a > 0:
        return inf_sign(sp.Integer(-1) ** n1 * pref)
    if a == 0:
        return pref if n1 == 0 else inf_sign(sp.Integer(-1) ** n1 * pref)
    return sp.Integer(0)


def inv_partial_limit(upper, n0, n1, l):
    if upper:
        return inv_partial_limit_pos(n0, n1, l)
    r = inv_partial_limit_pos(n0, n1, 2 - l)
    return r * sp.Integer(-1) ** (n0 + n1 + 1)


def fwd_lj_partial_limit(upper, n0, n1, l):
    """log|g'| is (lambda - 1) log(1 + v) above and (1 - lambda) log(1 - v) below: linear in
    lambda, so only n1 = 0 and n1 = 1 are not identically zero, and every v-derivative carries
    a negative power of 1 +- v and dies."""
    if n0 >= 1:
        return sp.Integer(0)
    if n1 == 0:
        s = l - 1 if upper else 1 - l
        return sp.Integer(0) if s == 0 else inf_sign(s)
    if n1 == 1:
        return sp.oo if upper else -sp.oo
    return sp.Integer(0)


def inv_lj_partial_limit_pos(n0, n1, l):
    """log|h'| at u = +inf, from Psi = (1 - lambda) log(1 + lambda u)/lambda.
    For lambda > 0: a u-derivative carries (1 + lambda u)^(-n0) and dies, while the pure
    lambda-derivatives multiply log(1 + lambda u) by d^n1_lambda[(1 - lambda)/lambda], of sign
    (-1)^n1, so they run to (-1)^n1 inf (and n1 = 0 to the sign of 1 - lambda).
    At lambda = 0, Psi is the polynomial sum_j (-1)^j u^(j+1) lambda^j/(j+1) times (1 - lambda),
    so d^n1_lambda Psi is a polynomial in u of degree n1 + 1 with leading coefficient
    (-1)^n1 n1!/(n1 + 1): its n0-th u-derivative runs to (-1)^n1 inf below n1 + 1, is the
    constant (-1)^n1 (n1!)^2 at n0 = n1 + 1 (1 at n1 = 0), and vanishes above."""
    if l == 0:
        if n0 < n1 + 1:
            return sp.oo if n1 == 0 else inf_sign(sp.Integer(-1) ** n1)
        if n0 == n1 + 1:
            return sp.Integer(1) if n1 == 0 else sp.Integer(-1) ** n1 * sp.factorial(n1) ** 2
        return sp.Integer(0)
    if n0 >= 1:
        return sp.Integer(0)
    if n1 == 0:
        return sp.Integer(0) if l == 1 else inf_sign(1 - l)
    return inf_sign(sp.Integer(-1) ** n1)


def inv_lj_partial_limit(upper, n0, n1, l):
    if upper:
        return inv_lj_partial_limit_pos(n0, n1, l)
    r = inv_lj_partial_limit_pos(n0, n1, 2 - l)
    return r * sp.Integer(-1) ** (n0 + n1)


PARTIAL_RULE = {"fwd": fwd_partial_limit, "inv": inv_partial_limit,
                "fwd_lj": fwd_lj_partial_limit, "inv_lj": inv_lj_partial_limit}


def inv_lj_edge_limit(n0, n1, l):
    """log|h'| and its mixed derivatives at the FINITE end of the domain of h: u = -1/lambda
    for lambda < 0 (the positive branch) and u = 1/(2 - lambda) for lambda > 2 (the negative
    one). There w = 1 + lambda u (1 - c u below) vanishes, and
    d^n0_u log|h'| = (1 - lambda)(-1)^(n0-1)(n0-1)! lambda^(n0-1) w^(-n0); each
    lambda-derivative raises the power of 1/w by one and brings a factor -n0 u, so the most
    singular term is C u^n1 w^(-n0-n1) and the limit is an infinity whose sign, with
    u -> -1/lambda, is that of (1 - lambda)(-1)^(n0-1) lambda^(n0-1-n1), namely (-1)^n1 above.
    Below, log|h'|(u) = log|h'_pos|(-u; c) adds (-1)^(n0+n1), so the sign is (-1)^n0."""
    return inf_sign(sp.Integer(-1) ** (n1 if l < 0 else n0))


def inv_edge_limit(n0, n1, l):
    """h and its mixed derivatives at the FINITE end of the domain of h. There w = 1 + lambda u
    (1 - c u below) vanishes and h^(n0) = P_n0 w^(1/lambda - n0) with 1/lambda - n0 < 0, so every
    entry is an infinity; each lambda-derivative multiplies by a term whose dominant part is
    (1/lambda - n0) u/w, negative at the upper end where u = -1/lambda > 0 and lambda < 0, so the
    sign alternates in n1 and P_n0 > 0 there. Below lambda = 2 the reflection
    h(u) = -h_pos(-u; c) adds (-1)^(n0+n1+1), leaving (-1)^(n0+1)."""
    if l < 0:
        return inf_sign(sp.Integer(-1) ** n1)
    return inf_sign(sp.Integer(-1) ** (n0 + 1))


def check_edge_limits():
    """The mixed limits at the finite end of the domain of h, against SymPy, for the map and
    for its log-Jacobian."""
    count = 0
    for l in (sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Rational(5, 2), sp.Integer(4)):
        b = "pos" if l < 0 else "neg"
        end = -1 / l if l < 0 else 1 / (2 - l)
        for q in ("inv", "inv_lj"):
          dsym = definition(q, b)
          rule = inv_edge_limit if q == "inv" else inv_lj_edge_limit
          for n1 in range(0, KMAX + 1):
              e1 = sp.diff(dsym, lam, n1) if n1 else dsym
              e1 = e1.subs(lam, l)
              for n0 in range(0, KMAX + 1 - n1):
                  if n0 + n1 == 0:
                      continue
                  e = sp.diff(e1, x, n0) if n0 else e1
                  # approached from inside the domain: from below at the upper end, from
                  # above at the lower one
                  ref = sp.limit(sp.simplify(e), x, end, "-" if l < 0 else "+")
                  got = rule(n0, n1, l)
                  if ref != got:
                      raise SystemExit(f"edge limit differs: {q} lambda={l} n0={n0} n1={n1}: "
                                       f"rule {got} vs SymPy {ref}")
                  count += 1
    print(f"checked {count} mixed limits at the finite end of the domain of h against SymPy")


def check_partial_limits():
    """Every mixed limit of the rule above against SymPy's limit of the definition."""
    lams = [sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Integer(0), sp.Rational(1, 3), sp.Rational(1, 2),
            sp.Integer(1), sp.Rational(3, 2), sp.Integer(2), sp.Rational(5, 2), sp.Integer(3)]
    count = 0
    for q in ("fwd", "inv", "fwd_lj", "inv_lj"):
        rule = PARTIAL_RULE[q]
        for upper in (True, False):
            b = "pos" if upper else "neg"
            for l in lams:
                if not end_in_domain(q, upper, l):
                    continue
                dl = definition_num(q, b, l)
                dsym = definition(q, b)
                for n1 in range(1, KMAX + 1):
                    # d^n1_lambda first, at symbolic lambda, then the value of lambda
                    e1 = sp.diff(dsym, lam, n1)
                    e1 = sp.simplify(sp.limit(e1, lam, l)) if l in (0, 2) else e1.subs(lam, l)
                    for n0 in range(0, KMAX + 1 - n1):
                        e = sp.diff(e1, x, n0) if n0 else e1
                        ref = sp.limit(sp.simplify(e), x, sp.oo if upper else -sp.oo)
                        got = rule(upper, n0, n1, l)
                        same = (ref == got if ref.is_infinite or got.is_infinite
                                else sp.simplify(ref - got) == 0)
                        if not same:
                            raise SystemExit(f"partial limit differs: {q} {'+' if upper else '-'}inf "
                                             f"lambda={l} n0={n0} n1={n1}: rule {got} vs SymPy {ref}")
                        count += 1
                # n1 = 0 is the pure rule, already checked by check_limits; check the agreement
                for n0 in range(0, KMAX + 1):
                    if rule(upper, n0, 0, l) != limit_rule(q, upper, n0, l):
                        raise SystemExit(f"partial rule disagrees with the pure rule: {q} "
                                         f"{'+' if upper else '-'}inf lambda={l} n0={n0}")
                    count += 1
    print(f"checked {count} mixed limits at an infinite argument against SymPy")


def check_limits():
    """The limits of each quantity at an infinite argument, against SymPy's limit of the
    definition. Every closed form of the header is written from exprel or log1prel, whose
    argument is infinite there, so the form itself gives NaN although the limit exists;
    the C++ yj_*_at_inf return these limits instead."""
    lams = [sp.Rational(-3, 2), sp.Rational(-1, 2), sp.Integer(0), sp.Rational(1, 5), sp.Rational(1, 4),
            sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Integer(1), sp.Rational(3, 2),
            sp.Rational(5, 3), sp.Rational(7, 4), sp.Rational(9, 5), sp.Rational(11, 6), sp.Integer(2),
            sp.Rational(5, 2), sp.Integer(4)]
    count = 0
    for q in ("fwd", "inv", "fwd_lj", "inv_lj"):
        for upper in (True, False):
            b = "pos" if upper else "neg"
            for l in lams:
                if not end_in_domain(q, upper, l):
                    continue
                dl = definition_num(q, b, l)
                for n in range(0, KMAX + 1):
                    e = sp.diff(dl, x, n) if n else dl
                    ref = sp.limit(sp.simplify(e), x, sp.oo if upper else -sp.oo)
                    got = limit_rule(q, upper, n, l)
                    same = ref == got if ref.is_infinite or got.is_infinite else sp.simplify(ref - got) == 0
                    if not same:
                        raise SystemExit(f"limit differs: {q} {'+' if upper else '-'}inf "
                                         f"lambda={l} n={n}: rule {got} vs SymPy {ref}")
                    count += 1
    print(f"checked {count} limits at an infinite argument against SymPy's limit of the definition")


# ---------------------------------------------------------------- C++ output

def yj_pow_cxx(z, sexp):
    """(1 + z)^s with s = k + a, k an integer and a = lambda or -lambda: yj_pow(1 + z, a, k), so that
    the exponent is not rounded (lambda - k and 2 - lambda - k are not doubles)"""
    k = sexp.subs(lam, 0)
    a = sp.expand(sexp - k)
    if not k.is_Integer or a not in (lam, -lam):
        raise ValueError(f"exponent not k +- lambda: {sexp}")
    return f"yj_pow(1.0 + {cxx(z)}, {cxx(a)}, {int(k)})"


def cxx(e):
    if e.is_Symbol:
        return e.name
    if e.is_Integer:
        return f"{int(e)}.0"
    if e.is_Rational:
        return f"({e.p}.0 / {e.q}.0)"
    if e.is_Add:
        return "(" + " + ".join(cxx(a) for a in e.args) + ")"
    if e.is_Mul:
        return "(" + " * ".join(cxx(a) for a in e.args) + ")"
    if e.is_Pow:
        b, p = e.args
        if p.is_Integer:
            n = int(p)
            if n >= 0:
                return f"ipow<{n}>({cxx(b)})"
            return f"(1.0 / ipow<{-n}>({cxx(b)}))"
        raise ValueError(f"non-integer power {e}")
    if isinstance(e, sp.exp):
        return f"m::exp({cxx(e.args[0])})"
    if isinstance(e, EM1):
        return f"Kokkos::expm1({cxx(e.args[0])})"
    if isinstance(e, POW1P):
        return yj_pow_cxx(e.args[0], e.args[1])
    if isinstance(e, LOG1P):
        return f"Kokkos::log1p({cxx(e.args[0])})"
    raise ValueError(f"cannot print {e} ({type(e)})")


def shared_log1prel(exprs, indent):
    """The derivatives of log1prel in the expressions all share one argument (lambda u, or
    (lambda - 2) u on the negative branch). They are replaced by the entries of an array lr[]
    filled by one call of log1prel and one of detail_log1prel::derivs_upto, which gives every
    order in one pass (bit-identical to log1prel_dK), instead of one call per order."""
    atoms = set()
    for e in exprs:
        atoms |= {a for a in e.atoms(sp.Function) if type(a) in LR}
    if not atoms:
        return exprs, []
    args = {a.args[0] for a in atoms}
    z = next(iter(args))
    for a in args:
        if sp.simplify(a - z) != 0:
            raise ValueError(f"log1prel at two different arguments: {args}")
    orders = {LR.index(type(a)) for a in atoms}
    nmax = max(orders)
    sub = {a: sp.Symbol(f"lr[{LR.index(type(a))}]") for a in atoms}
    lines = [f"{indent}const double z = {cxx(sp.factor(z))};", f"{indent}double lr[{nmax + 1}];"]
    if 0 in orders:
        lines.append(f"{indent}lr[0] = m::log1prel(z);")
    if nmax >= 1:
        lines.append(f"{indent}m::detail_log1prel::derivs_upto({nmax}, z, lr);")
    return [e.xreplace(sub) for e in exprs], lines


def shared_exprel(exprs, indent):
    """The derivatives of exprel appear at y = s log1p(X), one per function: log1p(X) and exprel
    are computed first, exprel by yj_exprel_d, which takes the power (1 + X)^s where |y| is large."""
    atoms = set()
    for e in exprs:
        atoms |= {a for a in e.atoms(sp.Function) if type(a) in EX}
    if not atoms:
        return exprs, []
    lines, sub = [], {}
    for n, a in enumerate(sorted(atoms, key=str)):
        y = a.args[0]
        logs = [f for f in sp.Mul.make_args(y) if isinstance(f, LOG1P)]
        if len(logs) != 1:
            raise ValueError(f"exprel argument not s log1p(X): {y}")
        X = logs[0].args[0]
        sexp = sp.factor(y / logs[0])
        j = EX.index(type(a))
        k = sexp.subs(lam, 0)
        lines += [f"{indent}const double lp{n} = Kokkos::log1p({cxx(X)});",
                  f"{indent}const double ex{n} = yj_exprel_d<{j}>(({cxx(sexp)}) * lp{n}, {cxx(X)}, "
                  f"{cxx(sp.expand(sexp - k))}, {int(k)});"]
        sub[a] = sp.Symbol(f"ex{n}")
        sub[logs[0]] = sp.Symbol(f"lp{n}")
    return [e.xreplace(sub) for e in exprs], lines


def shared_pow(exprs, indent):
    """the powers (1 + z)^s, computed first by yj_pow (before the common subexpressions, which would
    hide the form k +- lambda of the exponent)"""
    atoms = set()
    for e in exprs:
        atoms |= e.atoms(POW1P)
    lines, sub = [], {}
    for n, a in enumerate(sorted(atoms, key=str)):
        lines.append(f"{indent}const double pw{n} = {yj_pow_cxx(*a.args)};")
        sub[a] = sp.Symbol(f"pw{n}")
    return [e.xreplace(sub) for e in exprs], lines


def emit(exprs, indent):
    """the statements computing the expressions (common subexpressions named t0, t1, ...)"""
    exprs, lines = shared_log1prel(exprs, indent)
    exprs, more = shared_exprel(exprs, indent)
    lines += more
    exprs, more = shared_pow(exprs, indent)
    lines += more
    reps, red = sp.cse(exprs, symbols=sp.numbered_symbols("t"))
    lines += [f"{indent}const double {s.name} = {cxx(e)};" for s, e in reps]
    return lines, [cxx(e) for e in red]


INV_LJ_SWITCH = 0.5
E = sp.Symbol("e")


def inv_lj_factored(b, K, exprs, indent):
    """The mixed derivatives of log|h'| (at least one derivative in u and one in lambda) are
    rational in u and lambda; expanded by SymPy their terms cancel when |z| is large, z = lambda u
    (u >= 0) or (lambda - 2) u (u < 0): at u = -3, lambda = 0.4, order (3, 3), the terms are 96
    times the result (46 ulpC). Written in e = 1/(1 + z) and factored, the same derivative is a
    power of e times a short polynomial in e, 13 times more accurate there; the factored form
    divides by lambda (or 2 - lambda), so it is used for |z| > INV_LJ_SWITCH only, where those
    are bounded away from 0 unless |u| is large."""
    s = lam if b == "pos" else -(2 - lam)  # z = s u
    x_of_e = (1 / E - 1) / s
    alt = []
    for n1, e in enumerate(exprs):
        n0 = K - n1
        alt.append(sp.factor(sp.simplify(e.subs(x, x_of_e))) if n0 >= 1 and n1 >= 1 else e)
    exprs2, pre = shared_log1prel(alt, indent + "  ")
    reps, red = sp.cse(exprs2, symbols=sp.numbered_symbols("f"))
    lines = [f"{indent}const double zs = {cxx(sp.factor(s * x))};",
             f"{indent}if (Kokkos::fabs(zs) > {INV_LJ_SWITCH}) {{",
             f"{indent}  const double e = 1.0 / (1.0 + zs);"]
    lines += pre
    lines += [f"{indent}  const double {t.name} = {cxx(v)};" for t, v in reps]
    for n1, r in enumerate(cxx(v) for v in red):
        # the pure derivative in the parameter, which the rescaled route supplies above the switch
        if n1 == K:
            lines.append(f"{indent}  out[{n1}] = yj_big ? yj_top : ({r});")
        else:
            lines.append(f"{indent}  out[{n1}] = {r};")
    lines += [f"{indent}  return;", f"{indent}}}"]
    return lines


def branch_body(q, b, K, mode, indent):
    """C++ statements for the order K of quantity q on branch b: mode 'pure' returns the single
    expression, 'partials' writes out[n1] for n1 = 0..K and returns."""
    if mode == "pure":
        exprs = [base(q, b, K)]
    else:
        exprs = []
        for n1 in range(0, K + 1):
            e = base(q, b, K - n1)
            if n1:
                e = sp.diff(e, lam, n1)
            exprs.append(e)
    out = []
    if q == "inv_lj" and mode == "partials":
        # Only the pure derivative in the parameter loses its value at a large finite argument:
        # every entry with a derivative in the argument carries a power of 1 + lambda u rather
        # than log(1 + lambda u)/lambda and stays bounded. On the negative branch the quantity is
        # the positive one at (2 - lambda, -u), whose n1 = K entry picks up (-1)^K.
        zb = "p1 * x" if b == "pos" else "(p1 - 2.0) * x"
        arg = "p1, x" if b == "pos" else "2.0 - p1, -x"
        sgn = "" if (b == "pos" or K % 2 == 0) else "-"
        out.append(f"{indent}const bool yj_big = Kokkos::fabs({zb}) > 1.0e3;")
        out.append(f"{indent}const double yj_top =")
        out.append(f"{indent}    yj_big ? {sgn}yj_inv_lj_pure_lambda_rescaled<{K}>({arg}) : 0.0;")
        # the other side of the loss: a huge argument with lambda u small, where the forms give
        # NaN for values that are mostly finite
        out.append(f"{indent}if (!yj_big && Kokkos::fabs(x) > 1.0e50) {{")
        out.append(f"{indent}  yj_inv_lj_partials_small_z<{K}>({arg}, out);")
        if b == "neg" and K % 2:
            out.append(f"{indent}  for (int n1 = 0; n1 <= {K}; ++n1) out[n1] = -out[n1];")
        out.append(f"{indent}  return;")
        out.append(f"{indent}}}")
    if q == "inv_lj" and mode == "partials" and K >= 2:
        out += inv_lj_factored(b, K, exprs, indent)
    lines, red = emit(exprs, indent)
    out += lines
    if mode == "pure":
        out.append(f"{indent}return {red[0]};")
    else:
        for n1, r in enumerate(red):
            if q == "inv_lj" and n1 == K:
                out.append(f"{indent}out[{n1}] = yj_big ? yj_top : ({r});")
            else:
                out.append(f"{indent}out[{n1}] = {r};")
        out.append(f"{indent}return;")
    return out


def branch_block(q, K, mode, indent):
    """both branches; the positive branch of the partials of log|h'| is a call of
    yeo_johnson_inv_lj_pos_partials, which Box-Cox shares"""
    out = [f"{indent}if (x >= 0.0) {{"]
    if q == "inv_lj" and mode == "partials":
        out += [f"{indent}  yeo_johnson_inv_lj_pos_partials<{K}>(x, p1, out);", f"{indent}  return;"]
    else:
        out += branch_body(q, "pos", K, mode, indent + "  ")
    out.append(f"{indent}}}")
    out.append(f"{indent}{{")
    out += branch_body(q, "neg", K, mode, indent + "  ")
    out.append(f"{indent}}}")
    return out


def inv_lj_pos_function():
    """d^(K - n1)_u d^n1_lambda of (1 - lambda) u log1prel(lambda u) = (1 - lambda) log(1 + lambda u)/lambda,
    n1 = 0..K, in out[0..K]: the positive branch of log|h'| of Yeo-Johnson, and log|h'| of Box-Cox
    (scalar_box_cox.hpp) for every u > -1/lambda. The other entries of out are not written."""
    lines = ["// d^(K - n1)_u d^n1_lambda [(1 - lambda) log(1 + lambda u)/lambda], n1 = 0..K, in out[0..K] (the",
             "// other entries are not written): log|h'| of Yeo-Johnson for u >= 0, and log|h'| of Box-Cox",
             "// (scalar_box_cox.hpp) for every u with 1 + lambda u > 0.",
             "template <int K>",
             "LAPLACE_HD void yeo_johnson_inv_lj_pos_partials(double x, double p1, double* out) {",
             "  static_assert(K >= 1 && K <= 6, \"order 1 to 6\");"]
    for K in range(1, KMAX + 1):
        lines.append(f"  if constexpr (K == {K}) {{")
        lines += branch_body("inv_lj", "pos", K, "partials", "    ")
        lines.append("  }")
    lines += ["}", ""]
    return lines


CPP_NAME = {"fwd": "yeo_johnson_fwd", "inv": "yeo_johnson_inv",
            "fwd_lj": "yeo_johnson_fwd_lj", "inv_lj": "yeo_johnson_inv_lj"}
AT_INF_NAME = {"fwd": "yj_fwd_at_inf", "inv": "yj_inv_at_inf",
               "fwd_lj": "yj_fwd_lj_at_inf", "inv_lj": "yj_inv_lj_at_inf"}
PARTIALS_AT_INF_NAME = {"fwd": "yj_fwd_partials_at_inf", "inv": "yj_inv_partials_at_inf",
                        "fwd_lj": "yj_fwd_lj_partials_at_inf", "inv_lj": "yj_inv_lj_partials_at_inf"}


def generate_functions():
    lines = []
    for q in ("fwd", "inv", "fwd_lj", "inv_lj"):
        name = CPP_NAME[q]
        if q == "inv_lj":
            lines += inv_lj_pos_function()
        # pure derivatives, K = 0..6
        lines.append("template <int K>")
        lines.append(f"LAPLACE_HD double {name}(double x, double p1, double p2) {{")
        lines.append("  static_assert(K >= 0 && K <= 6, \"order 0 to 6\");")
        lines.append("  static_cast<void>(p2);")
        lines.append("  // an infinite x: the forms below give NaN there, the limit is taken instead")
        lines.append(f"  if (Kokkos::isinf(x)) return {AT_INF_NAME[q]}<K>(x > 0.0, p1);")
        if q in ("inv", "inv_lj"):
            edge = "yj_inv_at_edge" if q == "inv" else "yj_inv_lj_at_edge"
            lines.append("  // the finite end of the domain of h, where the forms give a large finite value")
            lines.append("  // under FMA contraction instead of the infinity the limit has")
            lines.append("  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1)))")
            lines.append(f"    return {edge}<K>(p1 < 0.0, p1);")
        for K in range(0, KMAX + 1):
            lines.append(f"  if constexpr (K == {K}) {{")
            lines += branch_block(q, K, "pure", "    ")
            lines.append("  }")
        lines.append("  return nan();")
        lines.append("}")
        lines.append("")
        # partials, K = 1..6
        lines.append("template <int K>")
        lines.append(f"LAPLACE_HD void {name}_partials(double x, double p1, double p2, double* out) {{")
        lines.append("  static_assert(K >= 1 && K <= 6, \"order 1 to 6\");")
        lines.append("  static_cast<void>(p2);")
        lines.append("  for (int i = 0; i < (K + 1) * (K + 2) / 2; ++i) out[i] = 0.0;")
        lines.append("  // an infinite x: the forms below give NaN there, the limits are taken instead")
        lines.append("  if (Kokkos::isinf(x)) {")
        lines.append(f"    {PARTIALS_AT_INF_NAME[q]}<K>(x > 0.0, p1, out);")
        lines.append("    return;")
        lines.append("  }")
        if q == "inv":
            lines.append("  // a large finite argument: the forms lose the value, the rescaled route keeps it")
            lines.append("  if (Kokkos::fabs((x >= 0.0 ? p1 : p1 - 2.0) * x) > 1.0e3) {")
            lines.append("    yj_inv_partials_rescaled<K>(p1, x, out);")
            lines.append("    return;")
            lines.append("  }")
        if q in ("inv", "inv_lj"):
            edge = "yj_inv_partials_at_edge" if q == "inv" else "yj_inv_lj_partials_at_edge"
            lines.append("  // the finite end of the domain of h, where the forms give a large finite")
            lines.append("  // value under FMA contraction instead of the infinity the limit has")
            lines.append("  if ((p1 < 0.0 && x == -1.0 / p1) || (p1 > 2.0 && x == 1.0 / (2.0 - p1))) {")
            lines.append(f"    {edge}<K>(p1 < 0.0, p1, out);")
            lines.append("    return;")
            lines.append("  }")
        for K in range(1, KMAX + 1):
            lines.append(f"  if constexpr (K == {K}) {{")
            lines += branch_block(q, K, "partials", "    ")
            lines.append("  }")
        lines.append("}")
        lines.append("")
    return lines


PREAMBLE = """// Scalar bijector yeo_johnson (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
// Yeo-Johnson, with exponent lambda = p1 (any real; NaN gives NaN), p2 unused:
//   g(v) = ((1 + v)^lambda - 1)/lambda             v >= 0   (Box-Cox of 1 + v)
//   g(v) = -((1 - v)^(2 - lambda) - 1)/(2 - lambda) v < 0   (minus Box-Cox of 1 - v)
//   h = g^-1: u >= 0 gives v = (1 + lambda u)^(1/lambda) - 1, u < 0 gives
//   v = 1 - (1 - (2 - lambda) u)^(1/(2 - lambda)).
// Range of g: R for 0 <= lambda <= 2; for lambda < 0 the positive branch is bounded
// above by -1/lambda (the image is (-inf, -1/lambda)); for lambda > 2 the negative
// branch is bounded below by 1/(2 - lambda) (the image is (1/(2 - lambda), inf)).
// The domain of g is R, so the distances of h(u) to the bounds of its range are +inf.
//
// Every form is written from exprel and log1prel (laplace/math/exprel.hpp,
// laplace/math/log1prel.hpp) and their derivatives, with M = log1p(v) and
// N = log1p(-v) (Kokkos::log1p, so that v tiny is not rounded away), c = 2 - lambda,
// and the inverse on the negative branch in a = -u:
//   forward, v >= 0:  g = M exprel(lambda M); g^(n) = prod_(k=1..n-1)(lambda - k) (1+v)^(lambda - n)
//   forward, v < 0:   g = -N exprel(c N);    g^(n) = (-1)^(n-1) prod_(k=0..n-2)(1 - lambda - k) (1-v)^(c - n)
//   inverse, u >= 0:  h = expm1(u log1prel(lambda u)) (no cancellation: the value is
//                     expm1 of a product, and log1prel is the series near 0);
//                     h^(n) = prod_(k=1..n-1)(1 - k lambda) exp(u (1 - n lambda) log1prel(lambda u))
//   inverse, u < 0:   h = -expm1(a log1prel(c a)); h^(n) = prod_(k=0..n-2)(1 - lambda + k c)
//                     exp(u (1 - lambda + (n-1) c) log1prel(c a))
//   log|g'|: (lambda - 1) M (v >= 0), (1 - lambda) N (v < 0); their x-derivatives
//                     (lambda - 1)(-1)^(n-1)(n-1)! (1+v)^(-n) and -(1 - lambda)(n-1)!(1-v)^(-n)
//   log|h'|: (1 - lambda) u log1prel(lambda u) (u >= 0), (1 - lambda) u log1prel(c a) (u < 0);
//                     their u-derivatives (1 - lambda)(-1)^(n-1)(n-1)! lambda^(n-1)(1+lambda u)^(-n)
//                     and (1 - lambda)(n-1)! c^(n-1)(1 - c u)^(-n)
// The lambda-derivatives are SymPy's chain rule on these forms: the derivative of exprel
// is exprel_d1, that of log1prel is log1prel_d1, and so on to order 6; the derivatives of
// log1prel that one function needs (all at one argument per branch) come from one call of
// m::detail_log1prel::derivs_upto, which gives every order in one pass. Every form is
// regular at lambda = 0 and at lambda = 2, so no limit is taken; the function is
// a function of lambda with no branch at 0 or 2. The forms and the checks are
// in tools/reference/bijector_yeo_johnson_forms.py (SymPy 1.14.0), which writes this file.
//
// At v = 0 (u = 0) the forward and inverse take the branch v >= 0, whose
// derivatives at 0 are the right-hand ones: g''' differs from the left-hand one
// by a jump (the kink, declared by forward_kinks()), g and its first two
// derivatives are continuous. The log-Jacobians are continuous at 0 up to
// their first derivative.
//
// At an infinite argument every form above is NaN, exprel and log1prel being
// evaluated at an infinite argument themselves, while the limit exists; and at
// the finite end of the image, where there is one, the forms reach the limit
// only because 1 + lambda u cancels to exactly zero, which an FMA prevents. So
// every quantity takes its limit from a guard at both kinds of end: the value
// and the x-derivatives from yj_fwd_at_inf, yj_inv_at_inf, yj_fwd_lj_at_inf,
// yj_inv_lj_at_inf, yj_inv_at_edge and yj_inv_lj_at_edge, and the
// lambda-derivatives (the _partials functions) from the four
// yj_*_partials_at_inf and from yj_inv_partials_at_edge and
// yj_inv_lj_partials_at_edge, with the derivation and the limits below.
// g(+inf) and g(-inf) are the two ends of the image of g. Nothing is NaN at
// an end of either domain for a valid lambda.
//
// Invalid parameters (lambda NaN or infinite): every function returns NaN; the
// bounds of the domains are NaN (the dispatch then rejects every argument).
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/log1prel.hpp>

namespace laplace::bijector::detail_scalar {

LAPLACE_HD_FORCEINLINE bool yeo_johnson_valid(double p1) { return p1 - p1 == 0.0; }  // finite

// w^(a + k) for w > 0, a = lambda or -lambda and k an integer (|k| <= 6): pow_pos(w, a) multiplied
// or divided by w |k| times, so that the exponent a + k (lambda - 2 or 2 - lambda, say) is not
// rounded: its rounding error, eps |a + k| at most, would be multiplied by log w (by 41 at
// w = 1e18: 20 ulpC in g'' measured on 2026-11-12). Where the products overflow or underflow
// while the power does not, pow_pos(w, a + k).
LAPLACE_HD_FORCEINLINE double yj_pow(double w, double a, int k) {
  double r = pow_pos(w, a);
  for (int i = 0; i < k; ++i) r *= w;
  for (int i = 0; i < -k; ++i) r /= w;
  if (!(r > 0.0) || !Kokkos::isfinite(r)) return pow_pos(w, a + k);
  return r;
}

// exprel^(J)(y) at y = s log1p(X), s = a + k as in yj_pow. For 8 <= |y| < 700 and |X| >= 1, by the
// forward recursion of exprel (the method of m::exprel_dJ there) started from E = (1 + X)^s by
// yj_pow instead of exp(y): the rounding of log1p(X), of s and of the product s log1p(X) is then
// not amplified by |y| (at v = -1.15e18, lambda = 1e-9 it gave 26 ulpC in d_lambda g). For
// |X| >= 1 the rounding of 1 + X is within the perturbation of X that ulpC allows. Elsewhere
// m::exprel_dJ(y).
template <int J>
LAPLACE_HD double yj_exprel_d(double y, double X, double a, int k) {
  const double ay = Kokkos::fabs(y);
  if (ay >= 8.0 && ay < 700.0 && Kokkos::fabs(X) >= 1.0) {
    const double E = yj_pow(1.0 + X, a, k);
    double p = (E - 1.0) / y;
    for (int j = 1; j <= J; ++j) p = (E - j * p) / y;
    return p;
  }
  if constexpr (J == 0) return m::exprel(y);
  if constexpr (J == 1) return m::exprel_d1(y);
  if constexpr (J == 2) return m::exprel_d2(y);
  if constexpr (J == 3) return m::exprel_d3(y);
  if constexpr (J == 4) return m::exprel_d4(y);
  if constexpr (J == 5) return m::exprel_d5(y);
  if constexpr (J == 6) return m::exprel_d6(y);
}

// ------------------------------------------- a huge argument while lambda u stays small
//
// The rescaled route below starts at |lambda u| = 1e3 and cannot reach the other side of the
// forms' loss: where lambda is so small that lambda u stays small even at a huge u, the forms
// carry a power of the argument in front of a bracket in the psi^(j)(z) which cancels as z goes
// to zero, so the power is infinite, the bracket is zero, and the product is NaN again.
// Measured on 2026-11-17 for the log-Jacobian's partials: at lambda = 0 the entries start to
// fail from |x| = 1e103 at order six and from 1e154 at orders four and five, and at x = 1e300
// order six returns NaN where the exact values are 0 at n1 = 2 and -3.6e301 at n1 = 3, order
// five where it is 4 at n1 = 2. Most of those are finite, so returning signed infinities would
// be wrong, and the rescaled route cannot serve either, its q_j being a cancelling difference at
// small z. The band is narrow: for lambda above about 1e-200 nothing fails anywhere, lambda u
// then reaching 1e3 before a power of u can overflow.
//
// What this route uses is that the power of u in each term does not depend on the summation
// index. Writing S = u psi(lambda u) with psi = log1prel and z = lambda u, Leibniz in u gives
//
//   d^n0_u S^(j) = sum_m C(n0, m) [d^(n0-m)_u u^(j+1)] lambda^m psi^(j+m)(z)
//
// and u^(j+1-n0+m) lambda^m = u^(j+1-n0) z^m, one power for the whole sum:
//
//   d^n0_u S^(j) = u^(j+1-n0) sum_(m=m0..n0) C(n0,m) (j+1)!/(j+1-n0+m)! z^m psi^(j+m)(z)
//
// with m0 = max(0, n0 - j - 1), which keeps every factorial's argument non-negative. The sum is
// bounded at small z, every term being a small power of z times a derivative of psi, so it is
// taken first and the single power applied once at the end. A sum that is exactly zero returns
// zero without that multiplication: at z = 0 it is zero exactly when the m = 0 term is absent,
// and a power of u that has overflowed would otherwise turn an exact zero into NaN. That is the
// whole of the repair; the identity is checked against SymPy's own derivative for every (j, n0)
// with j + n0 <= 6 by check_dS_small_z in this file's script.
//
// The entries of the log-Jacobian follow from log|h'| = (1 - lambda) S:
//
//   d^n0_u d^n1_lambda log|h'| = (1 - lambda) d^n0_u S^(n1) - n1 d^n0_u S^(n1-1).
//
// The route is taken for |u| above 1e50, fifty decades below the measured onset and far above
// any argument a model produces, and below that the forms are kept: over the overlap the two
// agree to 2.3e-16 relative (measured for lambda from 0 to 1.9 and |x| from 1 to 1e140, every
// order and every entry). The map's own partials need nothing here: at lambda = 0 the inverse is
// exp(u) - 1, already an infinity for every |u| above about 710, so the band holds no finite
// value of theirs to lose (measured 0 of 7 NaN at every place in it).

LAPLACE_HD double yj_fact_small(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= double(i);
  return f;
}

// d^n0_u S^(j) by the identity above; lr[k] = psi^(k)(z) for k = 0..j+n0
LAPLACE_HD double yj_dS_small_z(int j, int n0, double z, double u, const double* lr) {
  const int m0 = (n0 - j - 1 > 0) ? (n0 - j - 1) : 0;
  double acc = 0.0;
  for (int mm = m0; mm <= n0; ++mm) {
    double zm = 1.0;
    for (int i = 0; i < mm; ++i) zm *= z;
    const double cm = yj_fact_small(n0) / (yj_fact_small(mm) * yj_fact_small(n0 - mm));
    acc += cm * (yj_fact_small(j + 1) / yj_fact_small(j + 1 - n0 + mm)) * zm * lr[j + mm];
  }
  if (acc == 0.0) return 0.0;  // exact, and a power of u that has overflowed must not reach it
  const int e = j + 1 - n0;
  double p = 1.0;
  for (int i = 0; i < (e >= 0 ? e : -e); ++i) p *= u;
  return (e >= 0) ? acc * p : acc / p;
}

// out[n1] = d^(K-n1)_u d^n1_lambda [(1 - lambda) log(1 + lambda u)/lambda] for n1 = 0..K, valid
// wherever 1 + lambda u > 0, so for the positive branch of Yeo-Johnson and the whole of Box-Cox.
// Yeo-Johnson's negative branch is this at (2 - lambda, -u) with every entry times (-1)^K, the
// sign being (-1)^(n0+n1) and n0 + n1 = K.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_small_z(double lam, double u, double* out) {
  const double z = lam * u;
  double lr[K + 1];
  lr[0] = m::log1prel(z);
  m::detail_log1prel::derivs_upto(K, z, lr);
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    double v = (1.0 - lam) * yj_dS_small_z(n1, n0, z, u, lr);
    if (n1 >= 1) v -= double(n1) * yj_dS_small_z(n1 - 1, n0, z, u, lr);
    out[n1] = v;
  }
}

// ---------------------------------------------------------------- a large finite argument
//
// The lambda-derivatives of the inverse are written below as a polynomial in the derivatives of
// log1prel multiplied by a power of the argument, and above |lambda u| of about 1e3 that shape
// loses the value: psi^(j)(z) decays like j! log(1+z)/z^(j+1) and underflows while the power
// z^(j+1) in front of it overflows. Measured on 2026-11-17: at lambda = 1/2 and u = 1e60,
// log1prel_d6 is exactly 0 and z^7 is infinite, so the form returned inf * 0 = NaN for a value
// whose true size is 7.36e135 (mpmath at 60 digits). Regrouping the polynomial does not fix it,
// because its terms genuinely span hundreds of orders of magnitude: two attempts that only
// moved the powers around left more NaN than they removed, not fewer.
//
// What works is to compute the whole derivative from its structure, with the dominant scale
// factored out and every intermediate bounded:
//
//   h^(n0) = C_n0(lambda) exp(A_n0),   A_n0 = (1 - n0 lambda) S,   S = log(1 + lambda u)/lambda
//   d^n1_lambda [C exp(A)] = exp(A) sum_i binom(n1, i) C^(i) Y_(n1-i)(A', ..., A^(n1-i))
//
// with Y the complete Bell polynomials, A^(j) = (1 - n0 lambda) S^(j) - n0 j S^(j-1), and
//
//   S^(j) = q_j/lambda^(j+1),   q_j = z^(j+1) psi^(j)(z) = (-1)^j j! [log(1+z) - sum_(k=1..j) r^k/k]
//
// where r = z/(1+z). That closed form for q_j is proved equal to the derivative by SymPy for
// j = 0 to 7 (check_scaled_log1prel in this file's script); it is bounded, growing only as a
// logarithm, so S^(j) and the Bell polynomials stay representable and the only large number is
// the exp at the end. At lambda = 1/2 and u = 1e60 the chain runs q_6 ~ 1e5, S^(6) ~ 1e7,
// Y_6 ~ 1e16, exp(A) ~ 1e119, product 7.36e135, which is the value to eight digits.
//
// Below |z| = 1e3 the forms are exact and this route is not taken: q_j there is a difference of
// two nearly equal quantities and would lose about two digits at z = 1. The two agree to 1e-15
// or better over the whole overlap (measured from u = 1 to u = 1e20).

LAPLACE_HD double yj_binom(int n, int k) {
  double b = 1.0;
  for (int i = 0; i < k; ++i) b = b * double(n - i) / double(i + 1);
  return b;
}

// q[j] = z^(j+1) psi^(j)(z) for j = 0..n, by the closed form above
LAPLACE_HD void yj_log1prel_scaled(int n, double z, double* q) {
  const double L = Kokkos::log1p(z);
  const double r = z / (1.0 + z);
  double fact = 1.0, rk = 1.0, sum = 0.0;
  for (int j = 0; j <= n; ++j) {
    if (j >= 1) {
      fact *= double(j);
      rk *= r;
      sum += rk / double(j);
    }
    q[j] = ((j % 2) ? -fact : fact) * (L - sum);
  }
}

// out[n1] = d^(K-n1)_u d^n1_lambda h for n1 = 0..K, on the POSITIVE branch (u >= 0, lambda != 0,
// which |z| > 1e3 guarantees). Every entry carries at least one derivative, K being 1 or more, so
// the -1 of h = exp(S) - 1 never appears here and the same routine serves any map whose inverse
// is exp(S) times a constant: Box-Cox calls it unchanged for its own column. The negative branch is this one reflected, h(u) = -h_pos(-u; c)
// with c = 2 - lambda, so its entries are these at (-u, c) times (-1)^(K+1), the sign being the
// same for every n1 since n0 + n1 = K.
template <int K>
LAPLACE_HD void yj_inv_partials_rescaled_pos(double lam, double u, double* out) {
  double q[K + 1];
  yj_log1prel_scaled(K, lam * u, q);
  const double S = q[0] / lam;
  for (int n1 = 0; n1 <= K; ++n1) {
    const int n0 = K - n1;
    double Sd[K + 2], Ad[K + 2], Y[K + 2], Cd[K + 2];
    for (int j = 0; j <= K + 1; ++j) Sd[j] = Ad[j] = Y[j] = Cd[j] = 0.0;
    double lp = lam;
    for (int j = 1; j <= n1; ++j) {
      lp *= lam;
      Sd[j] = q[j] / lp;
    }
    for (int j = 1; j <= n1; ++j)
      Ad[j] = (1.0 - double(n0) * lam) * Sd[j] - double(n0) * double(j) * (j == 1 ? S : Sd[j - 1]);
    Y[0] = 1.0;
    for (int mm = 0; mm < n1; ++mm) {
      double acc = 0.0;
      for (int j = 0; j <= mm; ++j) acc += yj_binom(mm, j) * Ad[j + 1] * Y[mm - j];
      Y[mm + 1] = acc;
    }
    // C_n0 = prod_(k=1..n0-1)(1 - k lambda) and its lambda-derivatives, one factor at a time
    Cd[0] = 1.0;
    int d = 0;
    for (int k = 1; k <= n0 - 1; ++k) {
      const double f = 1.0 - double(k) * lam, fp = -double(k);
      for (int i = d + 1; i >= 0; --i) {
        double v = (i <= d) ? Cd[i] * f : 0.0;
        if (i >= 1 && i - 1 <= d) v += double(i) * Cd[i - 1] * fp;
        Cd[i] = v;
      }
      ++d;
    }
    double acc = 0.0;
    for (int i = 0; i <= n1; ++i) acc += yj_binom(n1, i) * Cd[i] * Y[n1 - i];
    out[n1] = acc * m::exp((1.0 - double(n0) * lam) * S);
  }
}

template <int K>
LAPLACE_HD void yj_inv_partials_rescaled(double p1, double x, double* out) {
  if (x >= 0.0) {
    yj_inv_partials_rescaled_pos<K>(p1, x, out);
    return;
  }
  yj_inv_partials_rescaled_pos<K>(2.0 - p1, -x, out);
  if ((K + 1) % 2) {
    for (int n1 = 0; n1 <= K; ++n1) out[n1] = -out[n1];
  }
}

// d^K_lambda [(1 - lambda) log(1 + lambda u)/lambda] = (1 - lambda) S^(K) - K S^(K-1): the only
// entry of the log-Jacobian's partials that loses its value at a large finite argument, every
// other carrying a power of 1 + lambda u instead of S and staying bounded. It holds wherever
// 1 + lambda u > 0, so it serves the positive branch of Yeo-Johnson and the whole of Box-Cox;
// Yeo-Johnson's negative branch calls it at (2 - lambda, -u) and takes (-1)^K.
template <int K>
LAPLACE_HD double yj_inv_lj_pure_lambda_rescaled(double lam, double u) {
  double q[K + 1];
  yj_log1prel_scaled(K, lam * u, q);
  double lp = 1.0, sd[K + 2];
  for (int j = 0; j <= K + 1; ++j) sd[j] = 0.0;
  for (int j = 0; j <= K; ++j) {
    lp *= lam;
    sd[j] = q[j] / lp;  // S^(j)
  }
  // d^K_lambda [(1 - lambda) S] = (1 - lambda) S^(K) - K S^(K-1)
  double v = (1.0 - lam) * sd[K];
  if (K >= 1) v -= double(K) * sd[K - 1];
  return v;
}

// ---------------------------------------------------------------- the limits at an infinite argument
//
// Every form above is written from exprel or log1prel, whose argument is itself infinite at an
// infinite x: exprel(inf) and log1prel(inf) are inf/inf, so the form gives NaN where the limit
// exists. The four functions below return the limit instead, and each quantity calls its own at
// the top. The argument enters every closed form only through a power of 1 + v, 1 - v,
// 1 + lambda u or 1 - c u, c = 2 - lambda, so the sign of that power's exponent decides: above
// zero the power runs to +inf and the prefactor gives the sign, at zero the power is 1 and the
// prefactor is the whole limit, below zero the power dies and the limit is 0. The limits and
// their derivation are in tools/reference/bijector_yeo_johnson_forms.py (check_limits), which
// compares each of them with SymPy's limit of the definition.
//
// The exponent is never formed: it vanishes exactly where a product of lambda with a small
// integer equals 1 (or where lambda equals the integer), so the two are compared instead of
// subtracted. A subtraction is what nvcc contracts into an FMA, and the exact product
// 3 (1/3) is 1 - 2^-54, not 1, so the knife edge would be reached on one platform and not on
// the other. Where the comparison fails the limit is +-inf or 0, which is the limit for that
// lambda: only lambda exactly 1/n carries a finite limit of order n.
//
// The derivation and every limit are in tools/reference/bijector_yeo_johnson_forms.py
// (check_limits), which compares each of them with SymPy's limit of the definition.

// The limit of pref w^e as w -> +inf: above is true where e > 0, and pref is then never zero
// in any of the uses below; an exponent known to vanish returns pref before this is called.
LAPLACE_HD_FORCEINLINE double yj_limit_power(double pref, bool above) {
  if (above) return pref > 0.0 ? inf() : -inf();
  return 0.0;
}

// g and its v-derivatives at v = +inf (upper) and at v = -inf. The value is the end of the
// image of g: -1/lambda below 0 and 1/(2 - lambda) above 2, infinite otherwise. For n >= 1
//   g^(n) = prod_(k=1..n-1)(lambda - k) (1 + v)^(lambda - n)                     v > 0
//   g^(n) = (-1)^(n-1) prod_(k=1..n-1)(c - k) (1 - v)^(c - n)                    v < 0
// and the exponent lambda - n (or c - n) decides, the prefactor being (-1)^(n-1) (n-1)! where
// it vanishes.
template <int K>
LAPLACE_HD double yj_fwd_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    return upper ? (p1 < 0.0 ? -1.0 / p1 : inf()) : (p1 > 2.0 ? 1.0 / (2.0 - p1) : -inf());
  }
  const double a = upper ? p1 : 2.0 - p1;  // lambda, or c = 2 - lambda
  double pref = (upper || (K % 2) == 1) ? 1.0 : -1.0;
  for (int k = 1; k <= K - 1; ++k) pref *= a - k;
  if (a == double(K)) return pref;
  return yj_limit_power(pref, a > double(K));
}

// h and its u-derivatives at u = +inf (upper) and at u = -inf. The range of h is R, so the
// value is infinite with the sign of u; u = +inf belongs to the domain of h only for
// lambda >= 0 and u = -inf only for lambda <= 2, the dispatch rejecting the other end. For
// n >= 1
//   h^(n) = prod_(k=1..n-1)(1 - k lambda) (1 + lambda u)^(1/lambda - n)          u > 0
//   h^(n) = prod_(k=1..n-1)(k c - 1) (1 - c u)^(1/c - n)                         u < 0
// and 1 - n lambda (or 1 - n c) has the sign of the exponent, zero included; lambda = 0 above,
// where h = e^u - 1, and lambda = 2 below, where h = 1 - e^(-u), fall under its positive case.
template <int K>
LAPLACE_HD double yj_inv_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) return upper ? inf() : -inf();
  const double a = upper ? p1 : 2.0 - p1;  // lambda, or c = 2 - lambda
  double pref = 1.0;
  for (int k = 1; k <= K - 1; ++k) pref *= upper ? 1.0 - k * a : k * a - 1.0;
  const double ka = double(K) * a;
  if (ka == 1.0) return pref;
  return yj_limit_power(pref, ka < 1.0);
}

// log|g'| and its v-derivatives at an infinite v: (lambda - 1) log(1 + v) above and
// (1 - lambda) log(1 - v) below run to an infinity of the sign of lambda - 1 (of 1 - lambda)
// and are 0 at lambda = 1, where g is the identity. Every v-derivative carries (1 + v)^(-n) or
// (1 - v)^(-n) and dies.
template <int K>
LAPLACE_HD double yj_fwd_lj_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    const double s = upper ? p1 - 1.0 : 1.0 - p1;
    return s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
  }
  static_cast<void>(upper);
  return 0.0;
}

// log|h'| and its u-derivatives at an infinite u: (1/lambda - 1) log(1 + lambda u) above and
// (1/c - 1) log(1 - c u) below run to an infinity of the sign of 1 - lambda (of lambda - 1) and
// are 0 at lambda = 1, where h is the identity; lambda = 0 above, where log|h'| = u, and
// lambda = 2 below, where log|h'| = -u, fall under the positive case. Every u-derivative
// carries (1 + lambda u)^(-n) or (1 - c u)^(-n) and dies, except on those two edges, where
// log|h'| is affine in u: the first derivative is then its slope, 1 above and -1 below.
template <int K>
LAPLACE_HD double yj_inv_lj_at_inf(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if constexpr (K == 0) {
    const double s = upper ? 1.0 - p1 : p1 - 1.0;
    return s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
  }
  if constexpr (K == 1) return upper ? (p1 == 0.0 ? 1.0 : 0.0) : (p1 == 2.0 ? -1.0 : 0.0);
  static_cast<void>(upper);
  return 0.0;
}

// ---- the mixed derivatives at an infinite argument
//
// out[n1] is d^(K - n1)_x d^n1_lambda, n1 = 0..K: the entries of bijector_partials<K> with
// n2 = 0, which are the only ones this map writes (it has no second parameter). A
// lambda-derivative multiplies a closed form by a logarithm of the argument, which loses to a
// negative power of it and wins against a vanishing exponent, so the sign of the exponent
// decides as it does for the pure derivatives: above zero an infinity, below zero zero, and at
// zero the prefactor for n1 = 0 and an infinity above it. The entries at n1 = 0 agree with
// the pure functions above, which the reference script checks.
//
// The inverse's negative branch is its positive branch reflected: h(u) = -h_pos(-u; c) and
// log|h'|(u) = log|h'_pos|(-u; c) with c = 2 - lambda, so d^n0_u d^n1_lambda carries
// (-1)^(n0+n1+1) for h and (-1)^(n0+n1) for its log-Jacobian, with c in place of lambda.
//
// A factorial and an integer power with a run-time order: the loops run over n1, and this
// path is taken only at an infinite argument.
LAPLACE_HD_FORCEINLINE double yj_fact_at(int n) {
  double f = 1.0;
  for (int k = 2; k <= n; ++k) f *= double(k);
  return f;
}
LAPLACE_HD_FORCEINLINE double yj_ipow_at(double z, int n) {
  double p = 1.0;
  for (int k = 0; k < n; ++k) p *= z;
  return p;
}

// g: prod_(k=1..n0-1)(a - k) (1 +- v)^(a - n0), a = lambda above and c = 2 - lambda below,
// with (-1)^(n0-1) on the negative branch and (-1)^n1 from d^n1_lambda = (-1)^n1 d^n1_c there.
// At n0 = 0 the value tends to an end of the image, whose lambda-derivatives are those of
// -1/lambda (above, below lambda = 0) and of 1/(2 - lambda) (below, above lambda = 2).
template <int K>
LAPLACE_HD void yj_fwd_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    if (n0 == 0) {
      if (upper) {
        out[n1] = p1 < 0.0 ? ((n1 % 2 ? 1.0 : -1.0) * yj_fact_at(n1) / yj_ipow_at(p1, n1 + 1)) : inf();
      } else {
        out[n1] = p1 > 2.0 ? (yj_fact_at(n1) / yj_ipow_at(2.0 - p1, n1 + 1))
                           : yj_limit_power(n1 % 2 ? 1.0 : -1.0, true);
      }
      continue;
    }
    double pref = (upper || ((n0 - 1 + n1) % 2) == 0) ? 1.0 : -1.0;
    for (int k = 1; k <= n0 - 1; ++k) pref *= a - double(k);
    if (a == double(n0) && n1 == 0) {
      out[n1] = pref;
      continue;
    }
    out[n1] = yj_limit_power(pref, a >= double(n0));
  }
}

// h: prod_(k=1..n0-1)(1 - k a) (1 + a |u|)^(1/a - n0), a = lambda above and c below, the sign
// of 1 - n0 a being the sign of the exponent (lambda = 0 above and lambda = 2 below included).
template <int K>
LAPLACE_HD void yj_inv_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    double v = 1.0;
    for (int k = 1; k <= n0 - 1; ++k) v *= 1.0 - double(k) * a;
    if (n1 % 2) v = -v;                                 // (-1)^n1 from the lambda-derivatives
    if (!upper && ((n0 + n1) % 2) == 0) v = -v;         // (-1)^(n0+n1+1) from the reflection
    const double na = double(n0) * a;
    out[n1] = (na == 1.0 && n1 == 0) ? v : yj_limit_power(v, na <= 1.0);
  }
}

// log|g'| is (lambda - 1) log(1 + v) above and (1 - lambda) log(1 - v) below: linear in
// lambda, so only n1 = 0 and n1 = 1 are not identically zero, and every v-derivative carries
// a negative power of 1 +- v and dies.
template <int K>
LAPLACE_HD void yj_fwd_lj_partials_at_inf(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    if (K - n1 >= 1) {
      out[n1] = 0.0;
    } else if (n1 == 0) {
      const double s = upper ? p1 - 1.0 : 1.0 - p1;
      out[n1] = s > 0.0 ? inf() : (s < 0.0 ? -inf() : 0.0);
    } else if (n1 == 1) {
      out[n1] = upper ? inf() : -inf();
    } else {
      out[n1] = 0.0;
    }
  }
}

// log|h'| above is Psi = (1 - lambda) log(1 + lambda u)/lambda. For lambda > 0 a u-derivative
// carries (1 + lambda u)^(-n0) and dies, while a pure lambda-derivative multiplies the
// logarithm by d^n1_lambda[(1 - lambda)/lambda], of sign (-1)^n1. At lambda = 0, Psi is the
// series (1 - lambda) sum_j (-1)^j u^(j+1) lambda^j/(j + 1), so d^n1_lambda Psi is a polynomial
// in u of degree n1 + 1 with leading coefficient (-1)^n1 n1!/(n1 + 1): its n0-th u-derivative
// runs to (-1)^n1 inf below n1 + 1, is the constant (-1)^n1 (n1!)^2 at n0 = n1 + 1, and
// vanishes above.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_at_inf(bool upper, double p1, double* out) {
  const double a = upper ? p1 : 2.0 - p1;
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int n0 = K - n1;
    double v;
    if (a == 0.0) {
      if (n0 < n1 + 1) {
        v = (n1 % 2) ? -inf() : inf();
      } else if (n0 == n1 + 1) {
        v = ((n1 % 2) ? -1.0 : 1.0) * yj_fact_at(n1) * yj_fact_at(n1);
      } else {
        v = 0.0;
      }
    } else if (n0 >= 1) {
      v = 0.0;
    } else if (n1 == 0) {
      v = a == 1.0 ? 0.0 : (a < 1.0 ? inf() : -inf());
    } else {
      v = (n1 % 2) ? -inf() : inf();
    }
    if (!upper && ((n0 + n1) % 2)) v = -v;
    out[n1] = v;
  }
}

// h and log|h'| with their u-derivatives at the FINITE end of the domain of h. There
// w = 1 + lambda u (1 - c u below) vanishes and every derivative is a negative power of it,
// hence an infinity. On the host w comes out exactly zero and the powers give that infinity by
// themselves, so these two were right without a branch; nvcc contracts 1 + lambda u into an FMA,
// which multiplies lambda by the ROUNDED -1/lambda exactly instead of rounding the product,
// leaves w at about 4e-17 and turns the infinity into a finite 1e100 (measured 7.8e100 at order
// six, 2026-11-17). The limit is therefore returned here rather than left to the arithmetic.
// h^(n) = P_n w^(1/lambda - n), with P_n > 0 at the upper end (lambda < 0, every factor above 1)
// and of sign (-1)^(n-1) at the lower one (lambda > 2).
template <int K>
LAPLACE_HD double yj_inv_at_edge(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if (upper) return inf();
  return (K % 2) ? inf() : -inf();  // (-1)^(K-1)
}

// log|h'| = (1/lambda - 1) log w and its u-derivatives: +inf at the upper end, (-1)^K inf at
// the lower one, which are the n1 = 0 entries of yj_inv_lj_partials_at_edge below.
template <int K>
LAPLACE_HD double yj_inv_lj_at_edge(bool upper, double p1) {
  if (!yeo_johnson_valid(p1)) return nan();
  if (upper) return inf();
  return (K % 2) ? -inf() : inf();
}

// h and its mixed derivatives at the FINITE end of the domain of h. There
// h^(n0) = P_n0 w^(1/lambda - n0) with w = 1 + lambda u vanishing and the exponent negative, so
// every entry is an infinity; each lambda-derivative multiplies by a term dominated by
// (1/lambda - n0) u/w, which is negative at the upper end (u = -1/lambda > 0 with lambda < 0,
// where P_n0 > 0 as well), so the sign alternates in n1. The reflection adds (-1)^(n0+n1+1)
// below lambda = 2 and leaves (-1)^(n0+1). Without this the entries come from the forms, where
// FMA keeps w at about 4e-17 and an infinity comes out as a finite 1e100 on the device.
template <int K>
LAPLACE_HD void yj_inv_partials_at_edge(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int e = upper ? n1 : (K - n1 + 1);
    out[n1] = (e % 2) ? -inf() : inf();
  }
}

// log|h'| and its mixed derivatives at the FINITE end of the domain of h, which is -1/lambda
// below lambda = 0 and 1/(2 - lambda) above lambda = 2. There w = 1 + lambda u (1 - c u below)
// vanishes, and d^n0_u log|h'| = (1 - lambda)(-1)^(n0-1)(n0-1)! lambda^(n0-1) w^(-n0); each
// lambda-derivative raises the power of 1/w by one and brings a factor -n0 u, so the most
// singular term is C u^n1 w^(-n0-n1) and the limit is an infinity whose sign, with u at the end
// itself, is that of (1 - lambda)(-1)^(n0-1) lambda^(n0-1-n1), namely (-1)^n1 above; below,
// log|h'|(u) = log|h'_pos|(-u; c) adds (-1)^(n0+n1) and leaves (-1)^n0.
template <int K>
LAPLACE_HD void yj_inv_lj_partials_at_edge(bool upper, double p1, double* out) {
  for (int n1 = 0; n1 <= K; ++n1) {
    if (!yeo_johnson_valid(p1)) {
      out[n1] = nan();
      continue;
    }
    const int e = upper ? n1 : K - n1;
    out[n1] = (e % 2) ? -inf() : inf();
  }
}

"""


def write_header():
    body = generate_functions()
    tail = """LAPLACE_HD_FORCEINLINE double yeo_johnson_dom_lo(double p1, double p2) {
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? -inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_dom_hi(double p1, double p2) {
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_img_lo(double p1, double p2) {  // R, or (1/(2 - p1), inf) above 2
  static_cast<void>(p2);
  if (!yeo_johnson_valid(p1)) return nan();
  return p1 > 2.0 ? 1.0 / (2.0 - p1) : -inf();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_img_hi(double p1, double p2) {  // R, or (-inf, -1/p1) below 0
  static_cast<void>(p2);
  if (!yeo_johnson_valid(p1)) return nan();
  return p1 < 0.0 ? -1.0 / p1 : inf();
}
// dom_hi - h(u) and h(u) - dom_lo: the range of h is R (dom = R), so both are +inf
LAPLACE_HD_FORCEINLINE double yeo_johnson_inv_to_upper(double u, double p1, double p2) {
  static_cast<void>(u);
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double yeo_johnson_inv_to_lower(double u, double p1, double p2) {
  static_cast<void>(u);
  static_cast<void>(p2);
  return yeo_johnson_valid(p1) ? inf() : nan();
}

}  // namespace laplace::bijector::detail_scalar
"""
    text = PREAMBLE + "\n".join(body) + "\n" + tail
    HEADER.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {HEADER} ({text.count(chr(10))} lines)")


def check_scaled_log1prel():
    """The closed form the rescaled route rests on, against SymPy's own derivative:

        z^(j+1) psi^(j)(z) = (-1)^j j! [log(1 + z) - sum_(k=1..j) r^k/k],   r = z/(1 + z)

    with psi(z) = log(1 + z)/z. Both sides are bounded where the two factors on the left are not,
    which is what lets the parameter-derivatives of the inverse be computed at a large argument.
    """
    z = sp.Symbol("z", positive=True)
    psi = sp.log(1 + z) / z
    r = z / (1 + z)
    for j in range(0, 8):
        left = z ** (j + 1) * sp.diff(psi, z, j)
        right = (-1) ** j * sp.factorial(j) * (sp.log(1 + z) - sum(r ** k / k for k in range(1, j + 1)))
        if sp.simplify(sp.expand(sp.together(left - right))) != 0:
            raise AssertionError(f"the closed form for z^(j+1) psi^(j)(z) fails at j = {j}")
    print("the closed form for z^(j+1) psi^(j)(z) agrees with the derivative for j = 0 to 7")


def check_dS_small_z():
    """The identity the small-z route rests on, against SymPy's own derivative:

        d^n0_u S^(j) = u^(j+1-n0) sum_(m=m0..n0) C(n0,m) (j+1)!/(j+1-n0+m)! z^m psi^(j+m)(z)

    with S = u psi(lambda u), psi(t) = log(1+t)/t, z = lambda u and m0 = max(0, n0-j-1). The
    point of the grouping is that one power of u carries the whole sum, so nothing is left
    factored in front of a bracket that cancels as z goes to zero.
    """
    u_, l_ = sp.symbols("u_ l_", positive=True)
    t = sp.Symbol("t_", positive=True)
    psi = sp.log(1 + t) / t
    S = u_ * psi.subs(t, l_ * u_)
    for K in range(1, KMAX + 1):
        for n1 in range(0, K + 1):
            j, n0 = n1, K - n1
            left = sp.diff(sp.diff(S, l_, j), u_, n0) if j else sp.diff(S, u_, n0)
            m0 = max(0, n0 - j - 1)
            right = 0
            for mm in range(m0, n0 + 1):
                coef = sp.binomial(n0, mm) * sp.factorial(j + 1) / sp.factorial(j + 1 - n0 + mm)
                right += coef * (l_ * u_) ** mm * sp.diff(psi, t, j + mm).subs(t, l_ * u_)
            right *= u_ ** (j + 1 - n0)
            if sp.simplify(left - right) != 0:
                raise AssertionError(f"the small-z identity fails at j = {j}, n0 = {n0}")
    print(f"the small-z identity agrees with the derivative for every (j, n0) with j + n0 <= {KMAX}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        check_limits()
        check_partial_limits()
        check_edge_limits()
        check_scaled_log1prel()
        check_dS_small_z()
    elif "--write" in sys.argv:  # the checks above are slow; --write skips them
        write_header()
    else:
        check()
        check_limits()
        check_partial_limits()
        check_edge_limits()
        check_scaled_log1prel()
        check_dS_small_z()
        write_header()
