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


# ---------------------------------------------------------------- C++ output

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
        return f"Kokkos::pow(1.0 + {cxx(e.args[0])}, {cxx(e.args[1])})"
    if isinstance(e, LOG1P):
        return f"Kokkos::log1p({cxx(e.args[0])})"
    for j, cls in enumerate(EX):
        if isinstance(e, cls):
            name = "m::exprel" if j == 0 else f"m::exprel_d{j}"
            return f"{name}({cxx(e.args[0])})"
    for j, cls in enumerate(LR):
        if isinstance(e, cls):
            name = "m::log1prel" if j == 0 else f"m::log1prel_d{j}"
            return f"{name}({cxx(e.args[0])})"
    raise ValueError(f"cannot print {e} ({type(e)})")


def emit(exprs, indent):
    """the statements computing the expressions (common subexpressions named t0, t1, ...)"""
    reps, red = sp.cse(exprs, symbols=sp.numbered_symbols("t"))
    lines = [f"{indent}const double {s.name} = {cxx(e)};" for s, e in reps]
    return lines, [cxx(e) for e in red]


def branch_block(q, K, mode, indent):
    """C++ statements for the order K of quantity q, for both branches: mode 'pure' returns
    the single expression, 'partials' writes out[n1] for n1 = 0..K."""
    out = []
    for b in ("pos", "neg"):
        if mode == "pure":
            exprs = [base(q, b, K)]
        else:
            exprs = []
            for n1 in range(0, K + 1):
                e = base(q, b, K - n1)
                if n1:
                    e = sp.diff(e, lam, n1)
                exprs.append(e)
        lines, red = emit(exprs, indent + "  ")
        cond = "if (x >= 0.0) {" if b == "pos" else "{"
        out.append(f"{indent}{cond}" if b == "pos" else f"{indent}{{")
        out += lines
        if mode == "pure":
            out.append(f"{indent}  return {red[0]};")
        else:
            out += [f"{indent}  out[{n1}] = {r};" for n1, r in enumerate(red)]
            out.append(f"{indent}  return;")
        out.append(f"{indent}}}")
    return out


CPP_NAME = {"fwd": "yeo_johnson_fwd", "inv": "yeo_johnson_inv",
            "fwd_lj": "yeo_johnson_fwd_lj", "inv_lj": "yeo_johnson_inv_lj"}


def generate_functions():
    lines = []
    for q in ("fwd", "inv", "fwd_lj", "inv_lj"):
        name = CPP_NAME[q]
        # pure derivatives, K = 0..6
        lines.append("template <int K>")
        lines.append(f"LAPLACE_HD double {name}(double x, double p1, double p2) {{")
        lines.append("  static_assert(K >= 0 && K <= 6, \"order 0 to 6\");")
        lines.append("  static_cast<void>(p2);")
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
// is exprel_d1, that of log1prel is log1prel_d1, and so on to order 6. Every form is
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
// Invalid parameters (lambda NaN or infinite): every function returns NaN; the
// bounds of the domains are NaN (the dispatch then rejects every argument).
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/log1prel.hpp>

namespace laplace::bijector::detail_scalar {

LAPLACE_HD_FORCEINLINE bool yeo_johnson_valid(double p1) { return p1 - p1 == 0.0; }  // finite

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


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        check()
        write_header()
