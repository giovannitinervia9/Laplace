"""References of vonmises_distrib (mean direction mu, circular standard deviation s), with mpmath.

Run with `python -I tools/reference/distrib_vonmises.py <mode>`; modes:

  write   all the files below (the default);
  check2d the full two-variable mpmath.diff of log f against the assembled references at a few
          points, which is what makes the assembled route independent of the separation formula;
  info    only the information files.

Files written under tests/data/reference/ (inputs and the mass of the cancelling terms are plain
hexadecimal doubles, the value is hi + lo as in bessel.py):

  vonmises_logpdf.csv   s,mu,y,K,b,hi,lo,mag   d^K l / (dmu^(K-b) ds^b), K = 0 the value; mag is the sum
                        of the absolute values of the terms that add up to the component, the scale of
                        the rounding that cannot be avoided where the component is zero
  vonmises_dy.csv       s,mu,y,J,K,b,hi,lo,mag   J derivatives in y and K in the parameters, J >= 1
  vonmises_cdf.csv      s,mu,y,which,hi,lo       which 0: F(y) from -pi, 1: 1 - F, 2: log F, 3: log(1 - F)
  vonmises_info.csv     s,which,hi,lo            which 0: I_mu,mu, 1: I_mu,s, 2: I_s,s (quadrature of the scores)
  vonmises_info_dk.csv  s,K,which,hi,lo          K-th derivative in s of the closed forms; which 0: I_mu,mu = kappa rho,
                                                 1: I_s,s = s^2 rho^2 / A'(kappa) (A' from Bessel's recurrence)

Method, from the definition and not from the library's formulas. kappa(s) is a root of
I_1(k)/I_0(k) = exp(-s^2/2) (mpmath.besseli). The derivatives of kappa in s and of
L(kappa(s)) = log I_0 - kappa come from reverting the Taylor series of A = I_1/I_0 (Riccati identity,
vonmises_scale.py, already checked there against mpmath.diff to 1e-79) and composing it with the series of
kappa(s) and with L^(j) = A^(j-1); this is NOT the Leibniz identity through q = 1 - rho that the library
uses. The mu and y derivatives of cos(y - mu) are mpmath.diff. Every row is computed at two working
precisions and dropped if they disagree. The cdf is a quadrature of the density from -pi (both arcs, F + G = 1
checked), independent of the Bessel-ratio series. The information is the quadrature of the squared scores.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mpmath import mp, mpf, besseli, exp, log, cos, sin, quad, pi, diff, factorial, binomial  # noqa: E402

import vonmises_scale as vs  # noqa: E402

mp.dps = 60
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
AGREE = mpf(2) ** -120


def dd(v):
    import math

    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpf(hi))


def hx(x):
    return float.hex(float(x))


def wrap(d):
    d = mpf(d)
    k = mp.floor(d / (2 * pi) + mpf("0.5"))
    return d - 2 * pi * k


def jets_once(s, extra):
    """kappa^(0..6) and d^j/ds^j L(kappa(s)), j = 1..6, at working precision prec + extra."""
    n = 6
    kd = vs.kappa_derivs(s, n, extra=extra)
    old = mp.prec
    mp.prec = old + extra
    try:
        k = vs.kappa_of_s(s)
        a = vs.ratio_derivs(k, n - 1)  # A, A', ..., A^(5)
        Lk = [None, a[0] - 1] + [a[j - 1] for j in range(2, n + 1)]  # L^(j)(kappa), j = 1..6
        inner = [mpf(0)] + [kd[i] / factorial(i) for i in range(1, n + 1)]
        out = [mpf(0)] * (n + 1)
        power = [mpf(1)] + [mpf(0)] * n
        for m in range(1, n + 1):
            new = [mpf(0)] * (n + 1)
            for i1 in range(n + 1):
                if power[i1] == 0:
                    continue
                for i2 in range(1, n + 1 - i1):
                    new[i1 + i2] += power[i1] * inner[i2]
            power = new
            c = Lk[m] / factorial(m)
            for i in range(1, n + 1):
                out[i] += c * power[i]
        Ld = [None] + [out[i] * factorial(i) for i in range(1, n + 1)]
        return kd, Ld
    finally:
        mp.prec = old


def jets(s):
    """The jets at two precisions, or None if they disagree."""
    from mpmath import log as mlog

    k0 = vs.kappa_of_s(s)
    bits = abs(float(mlog(k0 + mpf("1e-300")) / mlog(2)))
    e1 = int(7 * bits) + 80
    kd1, L1 = jets_once(s, e1)
    kd2, L2 = jets_once(s, e1 + e1 // 2 + 60)
    for i in range(7):
        for u, v in ((kd1[i], kd2[i]),) + (((L1[i], L2[i]),) if i >= 1 else ()):
            if v == 0:
                if u != 0:
                    return None
            elif abs(u - v) / abs(v) > AGREE:
                return None
    return kd2, L2


def rho_derivs(s, n):
    g = vs.gauss_taylor(s, n)
    return [g[m] * factorial(m) for m in range(n + 1)]


def logpdf_value(kap, z):
    L = log(besseli(0, kap)) - kap
    return kap * (cos(z) - 1) - L - log(2 * pi), L


def s_grid():
    pts = [mpf(10) ** (-e) for e in range(8, 0, -1)] + [mpf(v) for v in ("0.05", "0.2", "0.3", "0.5", "1", "2", "3", "4", "4.5", "6", "10", "20")]
    seen, out = set(), []
    for p in sorted(pts):
        key = float(p)
        if key not in seen:
            seen.add(key)
            out.append(mpf(repr(key)))
    return out


# (mu, y) pairs: z = y - mu wrapped near 0, +-pi/2, +-pi, and the cut crossed in both directions
def points():
    pairs = []
    for mu in (0.0, 2.0, -2.0, 3.1, -3.14):
        for y in (0.0, 0.01, 0.8, 1.9, -1.2, -3.0, 3.1):
            pairs.append((mu, y))
        for z in (1e-3, -1e-3, 1.5707963267948966, -1.5707963267948966):
            yy = mu + z
            if -3.14159 < yy < 3.14159:
                pairs.append((mu, yy))
    return pairs


def mag_pure_s(kd, rho_d, K, z, s):
    q = -(exp(-s * s / 2) - 1)
    e0 = abs(cos(z) - 1) + abs(q)
    tot = abs(kd[K]) * e0
    for j in range(K - 1):
        tot += binomial(K - 1, j) * abs(kd[j + 1]) * abs(rho_d[K - 1 - j])
    return tot


def mode_write():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = points()
    lp = ["s,mu,y,K,b,hi,lo,mag"]
    dy = ["s,mu,y,J,K,b,hi,lo,mag"]
    dropped = 0
    dy_pairs = set(pairs[::3])  # the mixed file is large, a third of the points carry it
    for s in s_grid():
        j = jets(s)
        if j is None:
            dropped += 1
            print("  dropped s = %.6e (the two precisions disagree)" % float(s))
            continue
        kd, Ld = j
        rho_d = rho_derivs(s, 6)
        for mu, y in pairs:
            z = wrap(mpf(y) - mpf(mu))
            val, L = logpdf_value(kd[0], z)
            head = "%s,%s,%s" % (hx(s), hx(mu), hx(y))
            h, l = dd(val)
            lp.append("%s,0,0,%s,%s,%s" % (head, hx(h), hx(l), float(abs(kd[0] * (cos(z) - 1)) + abs(L) + log(2 * pi))))
            for K in range(1, 7):
                for b in range(K + 1):
                    a = K - b
                    if a == 0:
                        v = kd[K] * (cos(z) - 1) - Ld[K]
                        mag = mag_pure_s(kd, rho_d, K, z, s)
                    else:
                        v = diff(lambda m: cos(z - m) - 1, mpf(0), a) * kd[b]
                        mag = abs(kd[b])
                    h, l = dd(v)
                    lp.append("%s,%d,%d,%s,%s,%s" % (head, K, b, hx(h), hx(l), float(mag)))
            if (mu, y) not in dy_pairs:
                continue
            for J in range(1, 7):
                for K in range(0, 7 - J):
                    for b in range(K + 1):
                        a = K - b
                        yv, mv = mpf(y), mpf(mu)
                        v = diff(lambda yy, mm: cos(yy - mm) - 1, (yv, mv), (J, a)) * kd[b]
                        h, l = dd(v)
                        dy.append("%s,%d,%d,%d,%s,%s,%s" % (head, J, K, b, hx(h), hx(l), float(abs(kd[b]))))
    (OUT / "vonmises_logpdf.csv").write_text("\n".join(lp) + "\n", encoding="utf-8")
    (OUT / "vonmises_dy.csv").write_text("\n".join(dy) + "\n", encoding="utf-8")
    print("vonmises_logpdf.csv: %d rows, vonmises_dy.csv: %d rows, %d s values dropped" % (len(lp) - 1, len(dy) - 1, dropped))
    write_cdf()
    write_info()


def arc_integral(kap, s, mu_w, a, b):
    """int_a^b of the density, by quadrature with breakpoints at the peak and its copies."""
    norm = 2 * pi * besseli(0, kap) * exp(-kap)
    f = lambda t: exp(kap * (cos(t - mu_w) - 1)) / norm
    cuts = [a, b]
    for peak in (mu_w - 2 * pi, mu_w, mu_w + 2 * pi):
        for m in (-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100):
            p = peak + m * s
            if a < p < b:
                cuts.append(p)
    cuts = sorted(set(cuts))
    return quad(f, cuts)


def write_cdf():
    mp.dps = 40
    rows = ["s,mu,y,which,hi,lo"]
    dropped = 0
    for sv in ("1e-4", "1e-2", "0.05", "0.3", "1", "2", "3", "4.5", "8"):
        s = mpf(repr(float(sv)))
        kap = vs.kappa_of_s(s)
        for mu in (0.0, 2.0, -2.0, 3.0):
            mu_w = wrap(mu)
            ys = [-3.1, -2.5, -1.0, -0.2, 0.0, 0.1, 0.5, 1.5, 2.9, 3.14]
            for m in (-3, -1, 1, 3, 10):
                ys.append(float(mu_w + m * s))
            for y in sorted(set(ys)):
                if not (-3.14159 < y < 3.14159):
                    continue
                lo_arc = arc_integral(kap, s, mu_w, -pi, mpf(y))
                up_arc = arc_integral(kap, s, mu_w, mpf(y), pi)
                if abs(lo_arc + up_arc - 1) > mpf("1e-25") or lo_arc <= 0 or up_arc <= 0:
                    dropped += 1
                    continue
                head = "%s,%s,%s" % (hx(s), hx(mu), hx(y))
                for w, v in enumerate((lo_arc, up_arc, log(lo_arc), log(up_arc))):
                    h, l = dd(v)
                    rows.append("%s,%d,%s,%s" % (head, w, hx(h), hx(l)))
    (OUT / "vonmises_cdf.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("vonmises_cdf.csv: %d rows, %d points dropped" % (len(rows) - 1, dropped))
    mp.dps = 60


def write_info():
    rows = ["s,which,hi,lo"]
    for sv in ("1e-3", "1e-2", "0.05", "0.1", "0.3", "1", "2", "3", "4.5", "6"):
        s = mpf(repr(float(sv)))
        kap = vs.kappa_of_s(s)
        dk = diff(vs.kappa_of_s, s, 1)
        A = vs.ratio(kap)
        w = [s * m for m in (-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100)]
        cuts = sorted(set([-pi, pi] + [p for p in w if -pi < p < pi]))
        dens = lambda z: vs.density(z, kap)
        i_mm = quad(lambda z: (kap * sin(z)) ** 2 * dens(z), cuts)
        i_ms = quad(lambda z: (kap * sin(z)) * (dk * (cos(z) - A)) * dens(z), cuts)
        i_ss = quad(lambda z: (dk * (cos(z) - A)) ** 2 * dens(z), cuts)
        for w_, v in enumerate((i_mm, i_ms, i_ss)):
            h, l = dd(v)
            rows.append("%s,%d,%s,%s" % (hx(s), w_, hx(h), hx(l)))
    (OUT / "vonmises_info.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("vonmises_info.csv: %d rows" % (len(rows) - 1))

    def imm(t):
        k = vs.kappa_of_s(t)
        return k * vs.ratio(k)

    def iss(t):
        k = vs.kappa_of_s(t)
        A = vs.ratio(k)
        return t * t * A * A / (1 - A / k - A * A)

    rows = ["s,K,which,hi,lo"]
    mp.dps = 100
    for sv in ("0.05", "0.1", "0.3", "1", "2", "3", "4.5"):
        s = mpf(repr(float(sv)))
        for K in range(1, 5):
            for w_, fn in enumerate((imm, iss)):
                h, l = dd(diff(fn, s, K))
                rows.append("%s,%d,%d,%s,%s" % (hx(s), K, w_, hx(h), hx(l)))
    mp.dps = 60
    (OUT / "vonmises_info_dk.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("vonmises_info_dk.csv: %d rows" % (len(rows) - 1))


def mode_check2d():
    """mpmath.diff of the two-variable log f against the assembled components."""
    mp.dps = 50
    worst = 0
    for sv, mu, y in (("0.3", 0.4, 1.1), ("1.0", -2.0, 0.5), ("3.0", 2.0, -1.0)):
        s = mpf(sv)
        j = jets(s)
        kd, Ld = j

        def logf(m, ss):
            k = vs.kappa_of_s(ss)
            return k * (cos(mpf(y) - m) - 1) - (log(besseli(0, k)) - k) - log(2 * pi)

        z = wrap(mpf(y) - mpf(mu))
        for a, b in ((0, 1), (1, 1), (2, 1), (1, 2), (0, 3), (3, 0), (2, 2), (0, 4)):
            ref = diff(logf, (mpf(mu), s), (a, b))
            if a == 0:
                mine = kd[b] * (cos(z) - 1) - Ld[b]
            else:
                mine = diff(lambda m: cos(z - m) - 1, mpf(0), a) * kd[b]
            rel = abs(ref - mine) / max(abs(ref), mpf("1e-30"))
            worst = max(worst, rel)
            print("s=%s a=%d b=%d  2d diff %.12e  assembled %.12e  rel %.2e" % (sv, a, b, float(ref), float(mine), float(rel)))
    print("worst relative difference %.3e" % float(worst))


MODES = {"write": mode_write, "check2d": mode_check2d, "info": write_info}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "write"
    if mode not in MODES:
        print("modes: " + ", ".join(sorted(MODES)))
        raise SystemExit(2)
    MODES[mode]()
