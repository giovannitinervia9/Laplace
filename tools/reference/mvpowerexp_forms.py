"""Closed forms of the multivariate power exponential, derived and checked with SymPy.

Usage (from the repository root):  python -I tools/reference/mvpowerexp_forms.py

Derives the normalizing constant by radial integration and checks every closed
form that include/laplace/distrib/detail/mvpowerexp_common.hpp transcribes
against SymPy's own derivative of the defining expression.

The law (Gomez, Gomez-Villegas and Marin 1998), with r = y - mu, d the
dimension, Sigma the scale matrix, delta = r' Sigma^-1 r and beta > 0 the shape:

  l = -log Z(d, beta) - 1/2 log det Sigma - 1/2 delta^beta,
  Z(d, beta) = 2^(1 + d/(2 beta)) pi^(d/2) Gamma(1 + d/(2 beta)) / (d Gamma(d/2)).

What is checked here.

1. Z by radial integration, against the closed form above (`check_constant`).
2. The covariance Var Y = Sigma 2^(1/beta) Gamma((d+2)/(2 beta)) / (d Gamma(d/(2 beta)))
   by the same radial integral (`check_covariance`); the factor is 1 at beta = 1.
3. The kernel table

     H(n, c) = d^n/d delta^n d^c/d beta^c [-1/2 delta^beta]
             = -1/2 delta^(beta - n) sum_(j=0..c) binom(c, j) P_n^(j)(beta) L^(c-j),

   L = log delta and P_n(beta) = beta (beta - 1) ... (beta - n + 1) the falling
   factorial, with the recursion P_(n+1)^(j) = (beta - n) P_n^(j) + j P_n^(j-1)
   (`check_kernel_table`).
4. The derivatives of log Z in beta. With u = 1/beta and
   F(u) = (d/2) u log 2 + lgamma(1 + (d/2) u),

     log Z = log 2 + (d/2) log pi - log d - lgamma(d/2) + F(1/beta),
     d^n/d beta^n F(1/beta) = (-1)^n sum_(k=1..n) F^(k)(1/beta) (n!/k!) binom(n-1, k-1) beta^(-n-k),
     F^(1)(u) = (d/2)(log 2 + psi(1 + (d/2) u)),  F^(k)(u) = (d/2)^k psi^(k-1)(1 + (d/2) u),

   both the reciprocal-chain formula and the whole composition (`check_log_z`).
5. The Faa di Bruno assembly of a derivative block against SymPy's direct
   derivative of l, for a concrete two-parameter structure in d = 2 and d = 3
   (`check_blocks`): A positions in mu, the theta tuple T and c positions in
   beta give the sum over the set partitions of the A + |T| positions whose
   parts hold at most two mu positions, of H(n, c) times the product over the
   parts of d_part delta, n the number of parts, with
   d_U delta = r' W_U r, d_mu_i d_U delta = -2 (W_U r)_i,
   d_mu_i d_mu_j d_U delta = 2 (W_U)_ij and zero with three or more mu.
6. The regularity order at r = 0 (`check_regularity`): the highest derivative
   order of delta^beta in the components of r that stays continuous there is
   ceil(2 beta) - 1 for a non-integer beta and infinite for an integer one.
"""
import itertools
import math

import sympy as sp

KMAX = 6

beta, delta, L, bb = sp.symbols("beta delta L b", positive=True)


# ---------------------------------------------------------------- 1. the constant
def check_constant():
    """Integrate exp(-delta^beta/2) in radial coordinates and compare with Z."""
    print("1. the normalizing constant")
    d = sp.Symbol("d", positive=True, integer=True)
    s = sp.Symbol("s", positive=True)
    # the surface of the unit sphere of R^d times int_0^inf s^(d-1) exp(-s^(2 beta)/2) ds,
    # with the substitution t = s^(2 beta)/2
    t = sp.Symbol("t", positive=True)
    s_of_t = (2 * t) ** (1 / (2 * beta))
    jac = sp.diff(s_of_t, t)
    radial = sp.integrate(sp.simplify(s_of_t ** (d - 1) * jac * sp.exp(-t)), (t, 0, sp.oo))
    surface = 2 * sp.pi ** (sp.Rational(1, 2) * d) / sp.gamma(d / 2)
    derived = sp.simplify(surface * radial)
    closed = (
        2 ** (1 + d / (2 * beta)) * sp.pi ** (d / 2) * sp.gamma(1 + d / (2 * beta)) / (d * sp.gamma(d / 2))
    )
    diff = sp.simplify(sp.powsimp(sp.expand_func(derived / closed), force=True))
    print(f"   derived  = {derived}")
    print(f"   closed   = {closed}")
    print(f"   ratio    = {diff}")
    assert diff == 1, diff
    # and numerically, by quadrature of the definition, in three dimensions and several betas
    import mpmath as mp

    mp.mp.prec = 200
    for dv in (1, 2, 3, 5):
        for bv in ("0.3", "0.5", "0.8", "1", "1.7", "2", "4"):
            b = mp.mpf(bv)
            surf = 2 * mp.pi ** (mp.mpf(dv) / 2) / mp.gamma(mp.mpf(dv) / 2)
            quad = surf * mp.quad(lambda s: s ** (dv - 1) * mp.exp(-(s ** (2 * b)) / 2), [0, 1, 4, 20, mp.inf])
            clo = (
                2 ** (1 + mp.mpf(dv) / (2 * b))
                * mp.pi ** (mp.mpf(dv) / 2)
                * mp.gamma(1 + mp.mpf(dv) / (2 * b))
                / (dv * mp.gamma(mp.mpf(dv) / 2))
            )
            rel = abs(quad / clo - 1)
            assert rel < mp.mpf("1e-40"), (dv, bv, rel)
            print(f"   d={dv} beta={bv}: quadrature/closed - 1 = {mp.nstr(rel, 4)}")
    print("   ok")


# ---------------------------------------------------------------- 2. the covariance
def check_covariance():
    """E[u u'] = (E ||u||^2 / d) I with u = Sigma^(-1/2) r."""
    print("2. the covariance factor")
    import mpmath as mp

    mp.mp.prec = 200
    for dv in (1, 2, 3, 5):
        for bv in ("0.3", "0.5", "1", "1.7", "2", "4"):
            b = mp.mpf(bv)
            num = mp.quad(lambda s: s ** (dv + 1) * mp.exp(-(s ** (2 * b)) / 2), [0, 1, 4, 20, mp.inf])
            den = mp.quad(lambda s: s ** (dv - 1) * mp.exp(-(s ** (2 * b)) / 2), [0, 1, 4, 20, mp.inf])
            got = num / den / dv
            clo = 2 ** (1 / b) * mp.gamma((mp.mpf(dv) + 2) / (2 * b)) / (dv * mp.gamma(mp.mpf(dv) / (2 * b)))
            rel = abs(got / clo - 1)
            assert rel < mp.mpf("1e-40"), (dv, bv, rel)
            if bv == "1":
                assert abs(clo - 1) < mp.mpf("1e-50"), clo
            print(f"   d={dv} beta={bv}: factor={mp.nstr(clo, 10)}, quadrature rel {mp.nstr(rel, 4)}")
    print("   ok (factor 1 at beta = 1)")


# ---------------------------------------------------------------- 3. the kernel table
def falling_derivs(n, c):
    """P_n^(j)(beta) for j = 0..c, from the recursion P_(n+1)^(j) = (beta - n) P_n^(j) + j P_n^(j-1)."""
    p = [sp.Integer(1)] + [sp.Integer(0)] * c
    for m in range(n):
        q = [sp.expand((beta - m) * p[j] + (j * p[j - 1] if j else 0)) for j in range(c + 1)]
        p = q
    return p


def kernel_closed(n, c):
    """The closed H(n, c) in delta, L = log delta and beta."""
    P = falling_derivs(n, c)
    acc = sum(sp.binomial(c, j) * P[j] * L ** (c - j) for j in range(c + 1))
    return -sp.Rational(1, 2) * delta ** (beta - n) * acc


def check_kernel_table():
    print("3. the kernel table H(n, c)")
    h = -sp.Rational(1, 2) * delta**beta
    worst = 0
    for n in range(KMAX + 1):
        for c in range(KMAX + 1 - n):
            direct = sp.diff(sp.diff(h, delta, n), beta, c)
            closed = kernel_closed(n, c).subs(L, sp.log(delta))
            diff = sp.simplify(sp.expand(sp.powsimp(direct - closed, force=True)))
            assert diff == 0, (n, c, diff)
            worst = max(worst, n + c)
            # the falling factorial recursion itself
            if c == 0 and n > 0:
                assert sp.expand(falling_derivs(n, 0)[0] - sp.prod([beta - i for i in range(n)])) == 0
    print(f"   every H(n, c) with n + c <= {worst} equals SymPy's derivative; P_n^(j) recursion ok")


# ---------------------------------------------------------------- 4. log Z in beta
def log_z_expr(dv):
    """log Z(dv, beta) as SymPy."""
    m = sp.Integer(dv) / (2 * beta)
    return (
        (1 + m) * sp.log(2)
        + sp.Integer(dv) / 2 * sp.log(sp.pi)
        + sp.loggamma(1 + m)
        - sp.log(dv)
        - sp.loggamma(sp.Integer(dv) / 2)
    )


def check_log_z():
    print("4. the derivatives of log Z in beta")
    u = sp.Symbol("u", positive=True)
    # the reciprocal chain rule, on a generic F
    F = sp.Function("F")
    for n in range(1, KMAX + 1):
        direct = sp.diff(F(1 / beta), beta, n)
        closed = (-1) ** n * sum(
            sp.diff(F(u), u, k).subs(u, 1 / beta)
            * sp.factorial(n)
            / sp.factorial(k)
            * sp.binomial(n - 1, k - 1)
            * beta ** (-n - k)
            for k in range(1, n + 1)
        )
        assert sp.simplify(direct - closed) == 0, n
    print(f"   d^n/dbeta^n F(1/beta) closed form ok for n = 1..{KMAX}")
    # the whole composition, for several dimensions
    for dv in (1, 2, 3, 5):
        lz = log_z_expr(dv)
        const = sp.log(2) + sp.Integer(dv) / 2 * sp.log(sp.pi) - sp.log(dv) - sp.loggamma(sp.Integer(dv) / 2)
        Fu = sp.Integer(dv) / 2 * u * sp.log(2) + sp.loggamma(1 + sp.Integer(dv) / 2 * u)
        assert sp.simplify(lz - (const + Fu.subs(u, 1 / beta))) == 0, dv
        for n in range(1, KMAX + 1):
            closed = (-1) ** n * sum(
                sp.diff(Fu, u, k).subs(u, 1 / beta)
                * sp.factorial(n)
                / sp.factorial(k)
                * sp.binomial(n - 1, k - 1)
                * beta ** (-n - k)
                for k in range(1, n + 1)
            )
            assert sp.simplify(sp.diff(lz, beta, n) - closed) == 0, (dv, n)
        # F^(k)(u) through polygamma
        for k in range(1, KMAX + 1):
            want = (sp.Integer(dv) / 2) ** k * sp.polygamma(k - 1, 1 + sp.Integer(dv) / 2 * u)
            if k == 1:
                want = sp.Integer(dv) / 2 * (sp.log(2) + sp.polygamma(0, 1 + sp.Integer(dv) / 2 * u))
            assert sp.simplify(sp.diff(Fu, u, k) - want) == 0, (dv, k)
        # log Z at beta = 1 is d/2 log(2 pi)
        at1 = sp.simplify(sp.expand_func(lz.subs(beta, 1)))
        assert sp.simplify(at1 - sp.Integer(dv) / 2 * sp.log(2 * sp.pi)) == 0, dv
        print(f"   d={dv}: composition and F^(k) ok; log Z(d, 1) = (d/2) log(2 pi)")
    print("   ok")


# ---------------------------------------------------------------- 5. the blocks
def set_partitions(items):
    """Every set partition of a list, as a list of lists of indices."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1 :]
        yield [[first]] + p


def check_blocks():
    print("5. the Faa di Bruno assembly against SymPy's derivative of l")
    for dv in (2, 3):
        np_ = 2
        mu = sp.symarray("mu", dv)
        yv = sp.symarray("y", dv)
        th = sp.symarray("th", np_)
        # a concrete positive definite Sigma(theta): th0 I + th1 B with B = v v'
        v = sp.Matrix([sp.Integer(i + 1) for i in range(dv)])
        Sig = th[0] * sp.eye(dv) + th[1] * (v * v.T)
        W = Sig.inv()
        r = sp.Matrix([yv[i] - mu[i] for i in range(dv)])
        delta_e = sp.expand((r.T * W * r)[0, 0])
        ell = -log_z_expr(dv) - sp.log(Sig.det()) / 2 - delta_e**beta / 2
        # numeric point
        sub = {th[0]: sp.Rational(7, 5), th[1]: sp.Rational(3, 10), beta: sp.Rational(7, 10)}
        for i in range(dv):
            sub[mu[i]] = sp.Rational(i, 4)
            sub[yv[i]] = sp.Rational(3 + i, 5)
        d_val = sp.nsimplify(delta_e.subs(sub))
        worst = 0.0
        # the derivative blocks, A in mu, the theta tuple, c in beta
        for A in range(0, 4):
            for kt in range(0, 3):
                for c in range(0, 3):
                    if A + kt + c > 4 or A + kt + c == 0:
                        continue
                    T = [0] * kt if kt < 2 else [0, 1]
                    mu_pos = list(range(A))  # all derivatives in mu_0 (one coordinate, repeated)
                    # direct
                    e = ell
                    for _ in range(A):
                        e = sp.diff(e, mu[0])
                    for j in T:
                        e = sp.diff(e, th[j])
                    for _ in range(c):
                        e = sp.diff(e, beta)
                    direct = sp.N(e.subs(sub), 40)
                    # assembled
                    pos = mu_pos + [A + i for i in range(kt)]
                    total = sp.Integer(0)
                    for part in set_partitions(pos):
                        if any(sum(1 for p in blk if p < A) > 2 for blk in part):
                            continue
                        n = len(part)
                        term = kernel_closed(n, c).subs({delta: d_val, L: sp.log(d_val), beta: sub[beta]})
                        ok = True
                        for blk in part:
                            tt = [T[p - A] for p in blk if p >= A]
                            WU = W
                            for j in tt:
                                WU = sp.diff(WU, th[j])
                            nmu = sum(1 for p in blk if p < A)
                            if nmu == 0:
                                f = (r.T * WU * r)[0, 0]
                            elif nmu == 1:
                                f = -2 * (WU * r)[0, 0]
                            else:
                                f = 2 * WU[0, 0]
                            term *= sp.N(f.subs(sub), 40)
                        if ok:
                            total += term
                    if A == 0 and c == 0:
                        ld = -sp.log(Sig.det()) / 2
                        for j in T:
                            ld = sp.diff(ld, th[j])
                        total += sp.N(ld.subs(sub), 40)
                        lz = -log_z_expr(dv)
                        if kt == 0:
                            total += sp.N(lz.subs(sub), 40)
                    if A == 0 and kt == 0 and c > 0:
                        lz = -log_z_expr(dv)
                        for _ in range(c):
                            lz = sp.diff(lz, beta)
                        total += sp.N(lz.subs(sub), 40)
                    scale = max(abs(float(direct)), 1.0)
                    err = abs(float(direct - total)) / scale
                    worst = max(worst, err)
                    assert err < 1e-30, (dv, A, kt, c, direct, total, err)
        print(f"   d={dv}: every block with A + |T| + c <= 4 matches, worst relative {worst:.2e}")
    print("   ok")


# ---------------------------------------------------------------- 6. the regularity
def check_regularity():
    print("6. the regularity order of delta^beta at r = 0")
    # along a ray r = s e, delta^beta = (e' W e)^beta s^(2 beta): the order of |s|^(2 beta)
    s = sp.Symbol("s", real=True)
    for bv in (sp.Rational(3, 10), sp.Rational(1, 2), sp.Rational(7, 10), 1, sp.Rational(3, 2), 2, 3):
        a = 2 * bv
        f = abs(s) ** a
        order = None
        for k in range(0, 9):
            g = sp.diff(f, s, k)
            lim_p = sp.limit(g, s, 0, "+")
            lim_m = sp.limit(g, s, 0, "-")
            if lim_p != lim_m or lim_p in (sp.oo, -sp.oo, sp.zoo) or lim_p is sp.nan:
                order = k - 1
                break
        want = "inf" if sp.Integer(bv).is_Integer and bv == int(bv) else int(math.ceil(float(a)) - 1)
        got = "inf" if order is None else order
        if want == "inf":
            assert got == "inf" or got >= 8, (bv, got)
            print(f"   beta={bv}: smooth (delta^beta is a polynomial in r)")
        else:
            assert got == want, (bv, got, want)
            print(f"   beta={bv}: continuous order {got} = ceil(2 beta) - 1")
    print("   ok")


if __name__ == "__main__":
    check_constant()
    check_covariance()
    check_kernel_table()
    check_log_z()
    check_blocks()
    check_regularity()
    print("all forms checked")
