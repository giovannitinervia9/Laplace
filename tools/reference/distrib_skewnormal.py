"""Reference values for skewnormal_distrib (centred skew normal): the derivatives to order 6 of the
log density, of log F and of log(1 - F) in (mu, sigma, gamma).

Usage (from the repository root):  python -I tools/reference/distrib_skewnormal.py [quick]

File tests/data/reference/distrib_skewnormal.csv, one row per value:
    kind (0 log density, 1 log F, 2 log(1 - F)), y, K, b, c, hi, lo, mu, sigma, gamma
with c the number of derivatives in gamma, b the number in sigma and K - b - c in mu; y, hi, lo, mu,
sigma and gamma in hexadecimal (the row stores the doubles the library sees).

Method. The DEFINITION is evaluated in truncated multivariate Taylor arithmetic (total degree 6 in the
three increments of mu, sigma and gamma) at 300 and 400 bits, so that no step size and no cancellation
of finite differences enters, and a row is kept when the two precisions agree to 2^-80 relative.
Nothing of the library's route (log_Phi_dk, the Bell polynomials, the (z, cc) tables, log_sn_cdf_dk)
is used. The definition is the classic skew normal at the triple implied by the moments,
    cc = sign(g) (2|g|/(4 - pi))^(1/3),  xi = mu - sigma cc,  omega = sigma sqrt(1 + cc^2),
    alpha = cc / (b sqrt(1 - (|g|/G)^(2/3))),   b = sqrt(2/pi),   G = ((4 - pi)/2)(b/sqrt(1 - b^2))^3,
    l = log 2 - log omega - log(2 pi)/2 - t^2/2 + log Phi(alpha t),   t = (y - xi)/omega,
all with the true pi (the library works with the double pi: near |g| = G the map is that sensitive).
The cdf is F(t; alpha) = int_(-inf)^t 2 phi(x) Phi(alpha x) dx, whose partial derivatives are elementary:
    dF/dt = 2 phi(t) Phi(alpha t),    dF/dalpha = -exp(-(1 + alpha^2) t^2/2) / (pi (1 + alpha^2)),
and F(t; alpha) for t <= 0 is the integral of -dF/dalpha over (alpha, inf) (F(t; +inf) = 0 for t < 0),
a positive integrand. The Taylor polynomial of F in (t, alpha) is built from these (the mixed
coefficients are computed in both ways and compared); for t > 0 the identity 1 - F(t; alpha) =
F(-t; -alpha) (the substitution x -> -x in the integral) turns the upper side into a lower one, and
log F = log(1 - S) is evaluated with log1p so that S far below 2^-300 keeps its digits.
"""
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath
from mpmath import mpf

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "distrib_skewnormal.csv"
N = 6  # total order
AGREE = mpf(2) ** -80
PRECS = (300, 400)


# ------------------------------------------------------------------ truncated multivariate series
class Ser:
    nvars = 3

    def __init__(self, d=None):
        self.d = d if d is not None else {}

    @staticmethod
    def const(c):
        return Ser({(0,) * Ser.nvars: mpf(c)})

    @staticmethod
    def var(i, c0):
        e = [0] * Ser.nvars
        d = {(0,) * Ser.nvars: mpf(c0)}
        e[i] = 1
        d[tuple(e)] = mpf(1)
        return Ser(d)

    def c0(self):
        return self.d.get((0,) * Ser.nvars, mpf(0))

    def __add__(self, o):
        if not isinstance(o, Ser):
            o = Ser.const(o)
        r = dict(self.d)
        for k, v in o.d.items():
            r[k] = r.get(k, 0) + v
        return Ser(r)

    __radd__ = __add__

    def __neg__(self):
        return Ser({k: -v for k, v in self.d.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, Ser) else -mpf(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, Ser):
            return Ser({k: v * o for k, v in self.d.items()})
        r = {}
        for ka, va in self.d.items():
            da = sum(ka)
            for kb, vb in o.d.items():
                if da + sum(kb) > N:
                    continue
                k = tuple(x + y for x, y in zip(ka, kb))
                r[k] = r.get(k, 0) + va * vb
        return Ser(r)

    __rmul__ = __mul__

    def apply(self, derivs):
        """f(self) from the derivatives f^(n)(s0), n = 0..N, of f at the constant term s0."""
        s0 = self.c0()
        delta = self - s0
        # Horner: sum f_n/n! delta^n
        res = Ser.const(derivs[N] / mpmath.factorial(N))
        for n in range(N - 1, -1, -1):
            res = res * delta + derivs[n] / mpmath.factorial(n)
        return res

    def inv(self):
        s0 = self.c0()
        return self.apply([(-1) ** n * mpmath.factorial(n) / s0 ** (n + 1) for n in range(N + 1)])

    def __truediv__(self, o):
        if not isinstance(o, Ser):
            return self * (1 / mpf(o))
        return self * o.inv()

    def __rtruediv__(self, o):
        return self.inv() * o

    def pow(self, p):
        s0 = self.c0()
        p = mpf(p)
        out, ff = [], mpf(1)
        for n in range(N + 1):
            out.append(ff * s0 ** (p - n))
            ff *= p - n
        return self.apply(out)

    def sqrt(self):
        return self.pow(mpf(1) / 2)

    def exp(self):
        e = mpmath.exp(self.c0())
        return self.apply([e] * (N + 1))

    def log(self):
        s0 = self.c0()
        return self.apply([mpmath.log(s0)] + [(-1) ** (n - 1) * mpmath.factorial(n - 1) / s0 ** n for n in range(1, N + 1)])

    def log1m(self):
        """log(1 - self) with the constant term through log1p (self's constant term may be tiny)."""
        s0 = self.c0()
        return self.apply([mpmath.log1p(-s0)] + [-mpmath.factorial(n - 1) / (1 - s0) ** n for n in range(1, N + 1)])


def Phi_derivs(u0):
    """Phi^(n)(u0), n = 0..N."""
    phi = mpmath.npdf(u0)
    out = [mpmath.ncdf(u0)]
    he = [mpf(1), u0]
    for n in range(2, N + 1):
        he.append(u0 * he[-1] - (n - 1) * he[-2])
    for n in range(1, N + 1):
        out.append((-1) ** (n - 1) * he[n - 1] * phi)
    return out


def log_Phi(u):
    """log Phi of a series (through the upper tail when u0 > 0, so that Phi near 1 keeps its digits)."""
    u0 = u.c0()
    if u0 > 0:
        return (-u).apply(Phi_derivs(-u0)).log1m()
    return u.apply(Phi_derivs(u0)).log()


# ------------------------------------------------------------------ the density
def constants():
    pi = mpmath.pi
    b = mpmath.sqrt(2 / pi)
    G = (4 - pi) / 2 * (b / mpmath.sqrt(1 - b * b)) ** 3
    return pi, b, G


def implied(sg, g, g0):
    """(t numerator pieces): cc, s, alpha as series in the increments."""
    pi, b, G = constants()
    ag, sgn = (g, 1) if g0 > 0 else (-g, -1)
    cc = (ag * (2 / (4 - pi))).pow(mpf(1) / 3) * sgn
    omq = 1 - (ag / G).pow(mpf(2) / 3)
    alpha = cc / (omq.sqrt() * b)
    s = (1 + cc * cc).sqrt()
    return cc, s, alpha


def tseries(mu, sg, g, y, g0):
    cc, s, alpha = implied(sg, g, g0)
    omega = sg * s
    xi = mu - sg * cc
    t = (y - xi) / omega
    return t, alpha, omega


def density_series(mu0, s0, g0, y):
    mu, sg, g = Ser.var(0, mu0), Ser.var(1, s0), Ser.var(2, g0)
    t, alpha, omega = tseries(mu, sg, g, mpf(y), g0)
    u = alpha * t
    return -omega.log() + (mpmath.log(2) - mpmath.log(2 * mpmath.pi) / 2) - t * t * mpf("0.5") + log_Phi(u)


# ------------------------------------------------------------------ the cdf
def fsn_integral(t0, a0):
    """F(t0; a0) for t0 <= 0: the integral of -dF/dalpha over (a0, inf), cut in geometric pieces at the
    scales of the integrand (the peak at b = 0 of width 1/|t0|, or the endpoint a0 with decay rate
    a0 t0^2 + |t0|) so that tanh-sinh converges on every piece; the error estimate is checked."""
    if t0 == 0:
        return (mpmath.pi / 2 - mpmath.atan(a0)) / mpmath.pi

    # the integrand times exp(E0), E0 the exponent at its maximum, so that its values are of order 1 and
    # the absolute stopping test of mpmath's quadrature is a relative one
    E0 = (1 + max(a0, mpf(0)) ** 2) * t0 * t0 / 2

    def f(bb):
        return mpmath.exp(E0 - (1 + bb * bb) * t0 * t0 / 2) / (mpmath.pi * (1 + bb * bb))

    at = abs(t0)
    pts = {a0}
    if a0 > 0:
        sc = 1 / (a0 * t0 * t0 + at)
        pts |= {a0 + sc * mpf(2) ** k for k in range(-2, 90, 1)}
    else:
        sc = 1 / (2 * at)
        pts |= {mpf(0)} | {s * sc * mpf(2) ** k for k in range(-2, 90) for s in (-1, 1)}
        pts = {p for p in pts if p >= a0}
        sc2 = 1 / at
        pts |= {sc2 * mpf(2) ** k for k in range(-2, 90)}
    pts = sorted(p for p in pts if p >= a0)
    v, err = mpmath.quad(f, pts + [mpmath.inf], error=True, maxdegree=10)
    if err > mpf(2) ** -100 * abs(v):
        raise ArithmeticError(f"quadrature error {err} for F({t0}; {a0})")
    return v * mpmath.exp(-E0)


def fsn_taylor(t0, a0):
    """Taylor coefficients c[(p, q)] of F(t; alpha) at (t0, a0), t0 <= 0, total order <= N."""
    assert t0 <= 0
    Ser.nvars = 2
    dt, da = Ser.var(0, t0), Ser.var(1, a0)
    # F_t = 2 phi(t) Phi(alpha t),  F_alpha = -exp(-(1 + alpha^2) t^2/2)/(pi (1 + alpha^2))
    phi = (dt * dt * mpf("-0.5")).exp() * (2 / mpmath.sqrt(2 * mpmath.pi))
    Ft = phi * da_Phi(dt, da)
    Fa = -((-(1 + da * da) * dt * dt * mpf("0.5")).exp()) / ((1 + da * da) * mpmath.pi)
    c = {(0, 0): fsn_integral(t0, a0)}
    for (p, q), v in Ft.d.items():
        if p + q + 1 <= N:
            c[(p + 1, q)] = v / (p + 1)
    for (p, q), v in Fa.d.items():
        if p + q + 1 <= N and q + 1 >= 1:
            w = v / (q + 1)
            if p >= 1:  # computed in both ways: they must agree
                ref = c[(p, q + 1)]
                assert abs(w - ref) <= mpf(2) ** -60 * max(abs(w), abs(ref)) or abs(w) < mpf(10) ** -300, (t0, a0, p, q, w, ref)
            else:
                c[(p, q + 1)] = w
    Ser.nvars = 3
    return c


def da_Phi(dt, da):
    u = da * dt
    return u.apply(Phi_derivs(u.c0()))


def compose_fsn(tser, aser):
    """F(t(x); alpha(x)) as a series in the three increments, for t0 = tser.c0 <= 0."""
    t0, a0 = tser.c0(), aser.c0()
    c = fsn_taylor(t0, a0)
    Ser.nvars = 3
    dts, das = tser - t0, aser - a0
    pt = [Ser.const(1)]
    pa = [Ser.const(1)]
    for i in range(1, N + 1):
        pt.append(pt[-1] * dts)
        pa.append(pa[-1] * das)
    res = Ser()
    for (p, q), v in c.items():
        res = res + (pt[p] * pa[q]) * v
    return res


def log_cdf_series(mu0, s0, g0, y, upper):
    mu, sg, g = Ser.var(0, mu0), Ser.var(1, s0), Ser.var(2, g0)
    t, alpha, omega = tseries(mu, sg, g, mpf(y), g0)
    sgn = -1 if upper else 1  # F_SN at (sgn t, sgn alpha) is F (lower) or 1 - F (upper)
    ts, als = t * sgn, alpha * sgn
    if ts.c0() <= 0:
        return compose_fsn(ts, als).log()
    return compose_fsn(-ts, -als).log1m()  # the other tail S, F = 1 - S


# ------------------------------------------------------------------ the grid
def G_double_offsets():
    with mpmath.workprec(300):
        G = constants()[2]
        return [float(G - mpf("1e-6")), float(G - mpf("1e-10"))]


def g_list():
    near = G_double_offsets()
    base = [1e-12, 1e-8, 1e-4, 1e-2, 0.5, 0.9] + near
    return [s * g for g in base for s in (1.0, -1.0)]


ZPDF = [-12.0, -5.0, -2.0, -0.5, 0.0, 0.7, 2.5, 6.0, 12.0]
ZCDF = [-12.0, -5.0, -2.0, -0.5, 0.3, 1.5, 4.0, 8.0]
ZSHORT = [-6.0, -1.0, 0.5, 4.0, 9.0]
PARAMS = [(0.4, 1.3), (-2.0, 0.25)]
PARAMS_EXTRA = [(1e5, 1e-6), (3.0, 1e6)]
G_EXTRA = [1e-4, -1e-4, 0.5, -0.5, G_double_offsets()[0], -G_double_offsets()[0]]


def value_rows(kind, mu0, s0, g0, y):
    sers = {}
    for prec in PRECS:
        mpmath.mp.prec = prec
        Ser.nvars = 3
        if kind == 0:
            sers[prec] = density_series(mpf(mu0), mpf(s0), mpf(g0), y)
        else:
            sers[prec] = log_cdf_series(mpf(mu0), mpf(s0), mpf(g0), y, kind == 2)
    lines, dropped = [], 0
    mpmath.mp.prec = PRECS[-1]
    lo_s, hi_s = sers[PRECS[0]], sers[PRECS[-1]]
    for (a, b, c), v in hi_s.d.items():
        K = a + b + c
        if K < 1:
            continue
        fac = mpmath.factorial(a) * mpmath.factorial(b) * mpmath.factorial(c)
        hv = v * fac
        lv = lo_s.d.get((a, b, c), mpf(0)) * fac
        if hv == 0 or abs(hv - lv) > AGREE * abs(hv):
            dropped += 1
            continue
        hi = float(hv)
        if not math.isfinite(hi) or abs(hi) < 2.2250738585072014e-308:
            dropped += 1
            continue
        lo = float(hv - mpf(hi))
        lines.append(f"{kind},{float(y).hex()},{K},{b},{c},{hi.hex()},{lo.hex()},{float(mu0).hex()},{float(s0).hex()},{float(g0).hex()}\n")
    return lines, dropped


def task(args):
    kind, mu0, s0, g0, zs = args
    lines, dropped = [], 0
    for z in zs:
        y = mu0 + s0 * z
        l, d = value_rows(kind, mu0, s0, g0, y)
        lines += l
        dropped += d
    return lines, dropped


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    jobs = []
    gs = g_list()
    if quick:
        gs = gs[::5]
    for g0 in gs:
        for (m, s) in PARAMS:
            jobs.append((0, m, s, g0, ZPDF))
        jobs.append((1, PARAMS[0][0], PARAMS[0][1], g0, ZCDF))
        jobs.append((2, PARAMS[0][0], PARAMS[0][1], g0, ZCDF))
    if not quick:
        for g0 in G_EXTRA:
            for (m, s) in PARAMS_EXTRA:
                jobs.append((0, m, s, g0, ZSHORT))
                jobs.append((1, m, s, g0, ZSHORT[:4]))
                jobs.append((2, m, s, g0, ZSHORT[:4]))
    with Pool(2) as pool:
        results = pool.map(task, jobs, chunksize=1)
    n = dropped = 0
    out = OUT.with_name("distrib_skewnormal_quick.csv") if quick else OUT
    with open(out, "w", newline="\n") as f:
        f.write("kind,y,K,b,c,hi,lo,mu,sigma,gamma\n")
        for lines, d in results:
            f.writelines(lines)
            n += len(lines)
            dropped += d
    print(f"{out.name}: {n} rows ({dropped} dropped)")


if __name__ == "__main__":
    main()
