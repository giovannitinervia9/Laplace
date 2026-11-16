"""The closed form of the von Mises arc mass at large concentration.

The cdf of `vonmises_distrib` needs, for an arc [a, b] that does not hold the
mean direction mu in its interior,

    M(a, b) = integral from a to b of exp(-2 kappa sin^2((t - mu)/2)) dt,

and returns its logarithm. The family computed it by panel quadrature, which
does not converge at large kappa: at kappa = 1e8 the integrand is non-zero over
1.1e-8 of an arc 0.0416 wide, a ratio of four million, and its value is
e^(-5.5e7). Giovanni's decision of 2026-11-17 is to use a closed form there.

The change of variable that makes it closed. With

    w = 2 sin((t - mu)/2),      dt = dw / sqrt(1 - w^2/4),

which is monotone on (mu - pi, mu + pi), the exponent is exactly -kappa w^2/2,
so

    M = integral from wa to wb of exp(-kappa w^2/2) (1 - w^2/4)^(-1/2) dw.

The Gaussian factor is exact, not an approximation, and the remaining factor
g(w) = (1 - w^2/4)^(-1/2) varies on a scale of order one while the Gaussian
varies on 1/sqrt(kappa). So

    M ~ g(w*) sqrt(2 pi / kappa) [Phi(-w0 sqrt kappa) - Phi(-w1 sqrt kappa)]

with w0 = min(|wa|, |wb|), w1 = max, and w* a point at which g is evaluated.
Written with Phi this covers both regimes with one formula: an endpoint far
from mu, where Phi's own asymptotics give the 1/(kappa w0) behaviour of an
endpoint-dominated Laplace integral, and an endpoint at mu itself, where
w0 = 0 and the bracket becomes the half-Gaussian. That matters because the
logarithm of a tiny tail is what the family needs, and `math/normal.hpp`
supplies log Phi to a measured accuracy.

This script decides, by measurement, the two things the implementation cannot
guess: at which point w* to evaluate g, and from which kappa the closed form
beats the quadrature.

Run with `python -I tools/reference/vonmises_arc.py <mode>`:

  check    the closed form against an exact mpmath quadrature of M, over kappa
           and over the arc's geometry, for each candidate choice of w*;
  write    the reference file the test reads.
"""

import sys
from pathlib import Path

from mpmath import mp, mpf, exp, log, sin, sqrt, pi, quad, erfc, ncdf, besseli

mp.dps = 50

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"


def exact_log_arc(kappa, mu, a, b):
    """log M(a, b) by quadrature at high precision, the integral factored by its
    largest endpoint value so that nothing underflows."""
    kappa, mu, a, b = mpf(kappa), mpf(mu), mpf(a), mpf(b)

    def u(t):
        return -2 * kappa * sin((t - mu) / 2) ** 2

    um = max(u(a), u(b))
    val = quad(lambda t: exp(u(t) - um), [a, b])
    return um + log(val)


def log_phi(x):
    """log Phi(x) at high precision, accurate in the far left tail."""
    x = mpf(x)
    return log(ncdf(x))


def closed_log_arc(kappa, mu, a, b, where="near"):
    """The closed form. `where` picks the point at which g is evaluated:
    near  = the endpoint nearest mu (the one that dominates),
    mid   = the midpoint in w,
    eff   = the near endpoint shifted by one Gaussian decay length.
    """
    kappa, mu, a, b = mpf(kappa), mpf(mu), mpf(a), mpf(b)
    wa = 2 * sin((a - mu) / 2)
    wb = 2 * sin((b - mu) / 2)
    w0, w1 = min(abs(wa), abs(wb)), max(abs(wa), abs(wb))
    rk = sqrt(kappa)

    def g(w):
        return 1 / sqrt(1 - w * w / 4)

    if where == "near":
        ws = w0
    elif where == "mid":
        ws = (w0 + w1) / 2
    else:
        # one decay length in from the dominating endpoint: the mean of w under
        # the Gaussian weight restricted to the arc, to first order
        ws = w0 + 1 / (kappa * w0) if w0 > 1 / rk else w0 + 1 / rk

    # the bracket, as a difference of standard normal tails
    lo = log_phi(-w0 * rk)
    hi = log_phi(-w1 * rk)
    # log(e^lo - e^hi) without cancellation
    d = hi - lo
    bracket = lo + log(-(exp(d) - 1)) if d < 0 else mpf("-inf")
    return log(g(ws)) + log(sqrt(2 * pi / kappa)) + bracket


def cases():
    """kappa over the range where the closed form might be used, and arcs of
    several geometries: an endpoint close to mu, far from it, and the whole
    half-circle."""
    out = []
    for ke in [3, 4, 5, 6, 7, 8, 10, 12]:
        kappa = mpf(10) ** ke
        for mu in [mpf(0), mpf(-2), mpf(2)]:
            s = 1 / sqrt(kappa)
            for lo_off, hi_off in [
                (10 * s, 40 * s),       # both ends inside a few widths
                (3 * s, pi - mpf("0.1")),
                (mpf("0.2"), mpf("1.1")),
                (mpf("1.0"), pi - mpf("0.05")),
                (0, mpf("0.5")),        # endpoint exactly at mu
            ]:
                a, b = mu + lo_off, mu + hi_off
                if b - a <= 0 or b - mu >= pi:
                    continue
                out.append((kappa, mu, a, b))
    return out


def mode_check():
    print("Relative error of log M, closed form against the exact quadrature.")
    print("The error is on log M itself, which is what the family returns.")
    print()
    print("%8s %7s %11s %11s   %11s %11s %11s" %
          ("kappa", "mu", "a - mu", "b - mu", "near", "mid", "eff"))
    worst = {"near": mpf(0), "mid": mpf(0), "eff": mpf(0)}
    for kappa, mu, a, b in cases():
        ex = exact_log_arc(kappa, mu, a, b)
        row = []
        for w in ("near", "mid", "eff"):
            try:
                ap = closed_log_arc(kappa, mu, a, b, w)
                rel = abs((ap - ex) / ex) if ex != 0 else abs(ap - ex)
            except Exception:
                rel = mpf("nan")
            row.append(rel)
            if rel == rel and rel > worst[w]:
                worst[w] = rel
        print("%8.1e %7.1f %11.3e %11.3e   %11.3e %11.3e %11.3e" %
              (float(kappa), float(mu), float(a - mu), float(b - mu),
               float(row[0]), float(row[1]), float(row[2])))
    print()
    for w in ("near", "mid", "eff"):
        print("worst relative error on log M, g at the %-4s point: %.3e" % (w, float(worst[w])))
    print()
    print("The choice of w* matters only through g, which is 1 + w^2/8 + ...,")
    print("so it cannot matter at all when the arc is short in w; the rows with")
    print("a long arc are the ones that separate the three.")


def mode_write():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["kappa,mu,a,b,hi,lo"]
    for kappa, mu, a, b in cases():
        v = exact_log_arc(kappa, mu, a, b)
        hi = float(v)
        lo = float(v - mpf(hi))
        rows.append("%s,%s,%s,%s,%s,%s" % (float.hex(float(kappa)), float.hex(float(mu)),
                                           float.hex(float(a)), float.hex(float(b)),
                                           float.hex(hi), float.hex(lo)))
    (OUT / "vonmises_arc.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("vonmises_arc.csv: %d rows" % (len(rows) - 1))


MODES = {"check": mode_check, "write": mode_write}

if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "check"
    if m not in MODES:
        print("modes: " + ", ".join(sorted(MODES)))
        raise SystemExit(2)
    MODES[m]()
