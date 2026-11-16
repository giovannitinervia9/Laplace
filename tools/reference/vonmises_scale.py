"""Checks of the von Mises circular standard deviation s = sqrt(-2 log rho).

Run with `python -I tools/reference/vonmises_scale.py <mode>`:

  mass     P(|Y - mu| <= s) over the range of rho, the number the manual page
           quotes, with mpmath at 60 digits;
  map      the map s -> kappa and its asymptotics, to fix the switch points of
           kappa_from_s;
  deriv    the derivatives of kappa in s to order six, by mpmath's own
           differentiation of the map, as the independent route;
  info     the expected information in (mu, s), to see whether it is diagonal
           and whether I_ss = s^2 rho^2 / A'(kappa) holds;
  write    the reference files of the map itself.

Files written by `write` (format as in bessel.py: the argument, then hi and lo
in hexadecimal, the two halves of the double-double of the true value):

  vonmises_kappa.csv        s   kappa(s)
  vonmises_kappa_d{n}.csv   s   d^n kappa / ds^n, n = 1..6

Method for the derivatives, which must not repeat the library's own route. The
derivatives of A = I_1/I_0 come from the Riccati identity A' = 1 - A/k - A^2
differentiated n times, in mpmath at a working precision that grows with
|log2 kappa| (the identity cancels at both ends); the Taylor series of A at
kappa is reverted to give the series of K = A^-1 at rho = A(kappa); and that
series is composed with the exact Taylor series of rho(s) = exp(-s^2/2), whose
coefficients come from the recurrence of the derivatives of a Gaussian. Reading
the coefficients of the composition gives every order at once. kappa itself is
a root of A(kappa) = exp(-s^2/2) found by mpmath.findroot, started from the
asymptotic guess so that the root is the one wanted at every s.
"""

import sys
from pathlib import Path

from mpmath import mp, besseli, mpf, exp, log, sqrt, cos, sin, quad, pi, diff, findroot

mp.dps = 60


def ratio(k):
    """A(kappa) = I_1(kappa) / I_0(kappa), the mean resultant length."""
    k = mpf(k)
    if k == 0:
        return mpf(0)
    return besseli(1, k) / besseli(0, k)


def kappa_of_rho(rho):
    """A^-1(rho) by a root of A(kappa) = rho."""
    rho = mpf(rho)
    if rho == 0:
        return mpf(0)
    # A good start: 2 rho for small rho, 1 / (2 (1 - rho)) for rho near one.
    g = 2 * rho if rho < mpf("0.5") else 1 / (2 * (1 - rho))
    return findroot(lambda k: ratio(k) - rho, g)


def kappa_of_s(s):
    """kappa from the circular standard deviation s, through rho = exp(-s^2/2)."""
    s = mpf(s)
    return kappa_of_rho(exp(-s * s / 2))


def s_of_kappa(k):
    """s = sqrt(-2 log A(kappa))."""
    return sqrt(-2 * log(ratio(k)))


def density(z, k):
    """The von Mises density at z = y - mu, in (mu, kappa)."""
    return exp(k * cos(z)) / (2 * pi * besseli(0, k))


def mass_within_s(rho):
    """P(|Y - mu| <= s) at the given mean resultant length rho."""
    rho = mpf(rho)
    k = kappa_of_rho(rho)
    s = sqrt(-2 * log(rho))
    half = min(s, pi)  # beyond pi the band is the whole circle
    return 2 * quad(lambda z: density(z, k), [0, half])


def mode_mass():
    print("P(|Y - mu| <= s), s = sqrt(-2 log rho), by mpmath at %d digits" % mp.dps)
    print()
    print("%10s  %16s  %16s  %12s" % ("rho", "kappa", "s", "P(|Y-mu|<=s)"))
    for r in ["0.999", "0.99", "0.9", "0.75", "0.5", "0.25", "0.1", "0.05", "0.01"]:
        rho = mpf(r)
        k = kappa_of_rho(rho)
        s = sqrt(-2 * log(rho))
        print("%10s  %16.8f  %16.8f  %12.6f" % (r, float(k), float(s), float(mass_within_s(rho))))
    print()
    print("s against pi: s > pi below rho = exp(-pi^2/2) = %.8f" % float(exp(-pi * pi / 2)))


def mode_map():
    print("The map s -> kappa, and 1 / s^2 for comparison")
    print()
    print("%12s  %24s  %12s  %12s" % ("s", "kappa", "kappa s^2", "1 - rho"))
    for e in range(-8, 3):
        for m in [1.0, 3.0]:
            s = mpf(m) * mpf(10) ** e
            if s > 20:
                continue
            k = kappa_of_s(s)
            q = -(exp(-s * s / 2) - 1)
            print("%12.3e  %24.12e  %12.8f  %12.4e" % (float(s), float(k), float(k * s * s), float(q)))
    print()
    print("kappa s^2 -> 1 as s -> 0 confirms kappa ~ 1 / s^2.")
    print("The asymptotic series of kappa in u = s^2 / 2, from inverting u = -log A(kappa):")
    # kappa = 1/(2u) + c0 + c1 u + ... ; the coefficients by fitting at small u
    for e in [-3, -4, -5, -6]:
        u = mpf(10) ** e
        s = sqrt(2 * u)
        k = kappa_of_s(s)
        print("  u = 1e%-3d  kappa - 1/(2u) = %+.12f" % (e, float(k - 1 / (2 * u))))


def mode_deriv():
    print("Derivatives of kappa in s to order six, by mpmath's own differentiation")
    print()
    for s_str in ["1e-4", "1e-2", "0.1", "0.5", "1.0", "2.0", "3.0", "5.0"]:
        s = mpf(s_str)
        print("s = %s, kappa = %.12e" % (s_str, float(kappa_of_s(s))))
        for n in range(1, 7):
            d = diff(kappa_of_s, s, n)
            print("   d%d = %+.16e" % (n, float(d)))
        print()


def mode_info():
    print("The expected information in (mu, s): is it diagonal, and is")
    print("I_ss = s^2 rho^2 / A'(kappa)?")
    print()
    print("%10s  %14s  %16s  %16s  %12s" % ("s", "kappa", "I_mumu", "I_ss", "I_mus"))
    for s_str in ["1e-3", "0.1", "0.5", "1.0", "2.0", "3.0"]:
        s = mpf(s_str)
        k = kappa_of_s(s)
        rho = exp(-s * s / 2)
        dk = diff(kappa_of_s, s, 1)
        # log f = kappa(s) (cos z - 1) - (log I0(kappa) - kappa) - log(2 pi)
        def logf(z, kk):
            return kk * cos(z) - log(besseli(0, kk)) - log(2 * pi)

        # scores at the true parameters, by differentiation of log f
        def score_mu(z):
            return diff(lambda m: logf(z - m + 0, k), mpf(0), 1)

        # d log f / ds = kappa'(s) (cos z - A(kappa))
        def score_s(z):
            return dk * (cos(z) - ratio(k))

        i_mumu = quad(lambda z: (k * sin(z)) ** 2 * density(z, k), [-pi, 0, pi])
        i_ss = quad(lambda z: score_s(z) ** 2 * density(z, k), [-pi, 0, pi])
        i_mus = quad(lambda z: (k * sin(z)) * score_s(z) * density(z, k), [-pi, 0, pi])
        print("%10s  %14.6e  %16.10e  %16.10e  %12.3e" % (s_str, float(k), float(i_mumu), float(i_ss), float(i_mus)))
    print()
    print("Closed forms to compare: I_mumu = kappa rho, I_ss = s^2 rho^2 / A'(kappa)")
    print("%10s  %18s  %18s" % ("s", "kappa rho", "s^2 rho^2 / A'"))
    for s_str in ["1e-3", "0.1", "0.5", "1.0", "2.0", "3.0"]:
        s = mpf(s_str)
        k = kappa_of_s(s)
        rho = exp(-s * s / 2)
        ap = diff(ratio, k, 1)
        print("%10s  %18.10e  %18.10e" % (s_str, float(k * rho), float(s * s * rho * rho / ap)))


NMAX = 6
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"


def dd(v):
    """The value as a pair of doubles, hi + lo."""
    import math as _math

    hi = float(v)
    if _math.isinf(hi) or _math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpf(hi))


def ratio_derivs(k, n):
    """A(k) and A^(1..n)(k) from the Riccati identity, at the current precision.

    A' = 1 - A/k - A^2, so differentiating n times,
      A^(n+1) = -sum_j C(n,j) A^(j) (1/k)^(n-j) - sum_j C(n,j) A^(j) A^(n-j),
    with (1/k)^(m) = (-1)^m m! / k^(m+1).
    """
    from mpmath import binomial, factorial

    k = mpf(k)
    a = [ratio(k)]
    for m in range(0, n):
        s1 = mpf(0)
        s2 = mpf(0)
        for j in range(0, m + 1):
            c = binomial(m, j)
            e = m - j
            inv = (-1) ** e * factorial(e) / k ** (e + 1)
            s1 += c * a[j] * inv
            s2 += c * a[j] * a[m - j]
        # the constant 1 of A' = 1 - A/k - A^2 survives only at m = 0
        a.append((mpf(1) if m == 0 else mpf(0)) - s1 - s2)
    return a


def revert(a, n):
    """Coefficients of K = A^-1 around rho, from the Taylor coefficients of A.

    Input a[j] = A^(j)(kappa) for j = 0..n. Writing A(kappa + t) = rho + sum_(j>=1) p_j t^j
    with p_j = a[j]/j!, the series K(rho + u) = kappa + sum_(j>=1) b_j u^j is the one
    with sum_j p_j (sum_i b_i u^i)^j = u. Returns K^(m)(rho) = m! b_m for m = 1..n.
    """
    from mpmath import factorial

    p = [a[j] / factorial(j) for j in range(0, n + 1)]
    b = [mpf(0)] * (n + 1)
    b[1] = 1 / p[1]
    for m in range(2, n + 1):
        # the coefficient of u^m in sum_(j>=1) p_j (sum b_i u^i)^j must vanish;
        # every term but p_1 b_m involves only b_1 .. b_(m-1)
        acc = mpf(0)
        for j in range(2, m + 1):
            acc += p[j] * poly_power_coeff(b, j, m, m)
        b[m] = -acc / p[1]
    return [b[m] * factorial(m) for m in range(1, n + 1)]


def poly_power_coeff(b, j, m, top):
    """Coefficient of u^m in (sum_(i=1..top) b_i u^i)^j, with b_top taken as zero.

    b_top is excluded so that the reversion can solve for it.
    """
    c = [mpf(0)] * (m + 1)
    c[0] = mpf(1)
    for _ in range(j):
        d = [mpf(0)] * (m + 1)
        for i1 in range(0, m + 1):
            if c[i1] == 0:
                continue
            for i2 in range(1, m + 1 - i1):
                if i2 >= top:
                    continue
                d[i1 + i2] += c[i1] * b[i2]
        c = d
    return c[m]


def gauss_taylor(s, n):
    """Taylor coefficients in h of rho(s + h) = exp(-(s+h)^2/2), to order n.

    The derivatives follow from D_(m+1) = -(s D_m + m D_(m-1)), exact algebra at
    the current precision, and the coefficients are those divided by m!.
    """
    from mpmath import factorial

    s = mpf(s)
    d = [exp(-s * s / 2)]
    if n >= 1:
        d.append(-s * d[0])
    for m in range(1, n):
        d.append(-(s * d[m] + mpf(m) * d[m - 1]))
    return [d[m] / factorial(m) for m in range(0, n + 1)]


def kappa_derivs(s, n=NMAX, extra=None):
    """kappa(s) and d^n kappa/ds^n for n = 1..n, by reversion and composition.

    The working precision is raised with |log2 kappa| times the highest order,
    because the Riccati identity cancels at both ends of the range of kappa:
    A^(j) is of size kappa^-(j+1) at large kappa and kappa^(1-j) at small kappa,
    while its terms are of order one, so about (n + 1) |log2 kappa| bits of the
    result are lost before anything else. At kappa = 1e16 and order six that is
    371 bits, and a run with only 166 extra bits returned +1.34e68 for a
    derivative whose exact leading term, -7!/s^8, is -5.04e67: the whole value
    was rounding noise. kappa_derivs_checked() is the entry point that refuses
    such a row instead of returning it.
    """
    from mpmath import factorial, log as mlog

    s = mpf(s)
    old = mp.prec
    if extra is None:
        k0 = kappa_of_s(s)
        bits = abs(float(mlog(k0 + mpf("1e-300")) / mlog(2)))
        extra = int((n + 1) * bits) + 80
    mp.prec = old + extra
    try:
        k = kappa_of_s(s)
        a = ratio_derivs(k, n)
        kd = revert(a, n)  # K^(m)(rho), m = 1..n
        g = gauss_taylor(s, n)  # rho(s + h) coefficients, g[0] = rho
        # compose: kappa(s + h) = kappa + sum_m kd[m-1]/m! (sum_(i>=1) g[i] h^i)^m
        inner = [mpf(0)] * (n + 1)
        for i in range(1, n + 1):
            inner[i] = g[i]
        out = [mpf(0)] * (n + 1)
        out[0] = k
        power = [mpf(0)] * (n + 1)
        power[0] = mpf(1)
        for m in range(1, n + 1):
            new = [mpf(0)] * (n + 1)
            for i1 in range(0, n + 1):
                if power[i1] == 0:
                    continue
                for i2 in range(1, n + 1 - i1):
                    new[i1 + i2] += power[i1] * inner[i2]
            power = new
            c = kd[m - 1] / factorial(m)
            for i in range(1, n + 1):
                out[i] += c * power[i]
        # the coefficient of h^i is the i-th derivative over i!
        return [out[0]] + [out[i] * factorial(i) for i in range(1, n + 1)]
    finally:
        mp.prec = old


AGREE = mpf(2) ** -140


def kappa_derivs_checked(s, n=NMAX):
    """kappa_derivs at two working precisions, returning None if they disagree.

    The second run is given half again as many extra bits. A row is kept only if
    every order agrees to AGREE relative, so that a loss of precision in the
    Riccati identity drops the row rather than writing noise into a reference
    file.
    """
    from mpmath import log as mlog

    k0 = kappa_of_s(s)
    bits = abs(float(mlog(k0 + mpf("1e-300")) / mlog(2)))
    e1 = int((n + 1) * bits) + 80
    a = kappa_derivs(s, n, extra=e1)
    b = kappa_derivs(s, n, extra=e1 + e1 // 2 + 60)
    for i in range(0, n + 1):
        scale = abs(b[i])
        if scale == 0:
            if a[i] != 0:
                return None
            continue
        if abs(a[i] - b[i]) / scale > AGREE:
            return None
    return b


def write_grid():
    """The s values of the reference files: the three regions and their joins.

    Capped at s = 4.5. Beyond it the map's own conditioning, not the algorithm,
    sets the error: kappa is about 2 exp(-s^2/2), so a relative perturbation
    eps of s moves kappa by s^2 eps, which is 178 ulp at s = 17.8 however
    kappa is computed. The region above the cap is checked in the test against
    its asymptote with a tolerance of that size, which is the only honest bound
    there.
    """
    pts = []
    e = -8.0
    while e <= 0.653:  # up to s = 4.5
        pts.append(mpf(10) ** mpf(repr(e)))
        e += 0.25
    for v in ["0.05", "0.09", "0.1", "0.11", "0.15", "0.2", "0.3", "0.5", "1.0",
              "1.1774", "1.5", "2.0", "2.5", "2.9", "3.0", "3.5", "3.9", "4.0",
              "4.1", "4.5"]:
        pts.append(mpf(v))
    seen, out = set(), []
    for p in sorted(pts):
        key = float(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(mpf(repr(key)))  # the nearest double, so the test reads the same s
    return out


def mode_write():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = write_grid()
    rows = [["x,hi,lo"] for _ in range(NMAX + 1)]
    dropped = 0
    for s in grid:
        vals = kappa_derivs_checked(s, NMAX)
        if vals is None:
            dropped += 1
            print("  dropped s = %.6e (the two precisions disagree)" % float(s))
            continue
        for n in range(0, NMAX + 1):
            hi, lo = dd(vals[n])
            rows[n].append("%s,%s,%s" % (float.hex(float(s)), float.hex(hi), float.hex(lo)))
    names = ["vonmises_kappa.csv"] + ["vonmises_kappa_d%d.csv" % n for n in range(1, NMAX + 1)]
    for n, name in enumerate(names):
        (OUT / name).write_text("\n".join(rows[n]) + "\n", encoding="utf-8")
        print("%s: %d rows" % (name, len(rows[n]) - 1))
    if dropped:
        print("%d of %d points dropped" % (dropped, len(grid)))


MODES = {"mass": mode_mass, "map": mode_map, "deriv": mode_deriv, "info": mode_info,
         "write": mode_write}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mass"
    if mode not in MODES:
        print("modes: " + ", ".join(sorted(MODES)))
        raise SystemExit(2)
    MODES[mode]()
