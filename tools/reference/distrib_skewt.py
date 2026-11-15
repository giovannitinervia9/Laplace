"""Reference values and checks of the closed forms for skewt_distrib
(include/laplace/distrib/skewt.hpp).

Usage (from the repository root):
    python -I tools/reference/distrib_skewt.py forms
    python -I tools/reference/distrib_skewt.py logpdf

Files in tests/data/reference (every number in hexadecimal; (a, b, c, d) is the number of
derivatives in (mu, sigma, alpha, nu) and K = a + b + c + d):
    distrib_skewt_logpdf.csv   y, K, a, b, c, d, hi, lo, mu, sigma, alpha, nu    d^K log f

"forms" checks with SymPy the one family of closed forms the header transcribes, the mixed
derivatives of u(z, nu) = z sqrt((nu + 1)/(nu + z^2)):
  * the two identities the recursion of u_table encodes,
        d/dz [z^m B^-s] = m z^(m-1) B^-s - 2 s z^(m+1) B^-(s+1),
        d^q/dnu^q B^(-1/2) = prod_(i<q) (-1/2 - i) B^(-1/2-q),      B = nu + z^2,
    and the rule d^p/dz^p [z W] = z d^p W + p d^(p-1) W that joins the factor z;
  * a reimplementation of the whole recursion in exact rational-plus-sqrt arithmetic against
    SymPy's own derivative of u, for every p + q <= 6 at three points.

Method for the file. Nothing of the library's routes is used and no closed form of the header
enters: log f is the definition

    log f = log 2 + log Gamma((nu+1)/2) - log Gamma(nu/2) - log(nu pi)/2 - log sigma
            - ((nu+1)/2) log(1 + z^2/nu) + log T_(nu+1)(w),
    z = (y - mu)/sigma,   w = alpha z sqrt((nu + 1)/(nu + z^2)),

with T_m the Student cdf from the incomplete beta on the side that does not cancel
(T_m(-|x|) = I_q(m/2, 1/2)/2 with q = m/(m + x^2), and log T_m(x) = log1p(-T_m(-x)) for x > 0),
and the derivatives are mpmath.diff of that function of the four parameters, computed at 400 and
at 600 bits; a row is kept when the two agree to 2^-100 relative, otherwise it is recomputed at
800 and 1000 bits and dropped if they still disagree. The grid covers the three limits of the
family (alpha = 0 the Student, nu large the skew normal, |alpha| large the half t) and both
tails.
"""

import csv
import os
import struct
import sys

import mpmath as mp


def hexd(x):
    return float(x).hex()


def split(x):
    """x as a sum of two doubles, hi + lo."""
    hi = float(x)
    if not mp.isfinite(x) or hi in (float("inf"), float("-inf")):
        return hi, 0.0
    lo = float(x - mp.mpf(hi))
    return hi, lo


# ---------------------------------------------------------------- the definition


def student_cdf(x, m):
    """T_m(x), the Student cdf at m degrees of freedom, from the incomplete beta."""
    x, m = mp.mpf(x), mp.mpf(m)
    a = -abs(x)
    q = m / (m + a * a)
    lower = mp.betainc(m / 2, mp.mpf(1) / 2, 0, q, regularized=True) / 2
    return lower if x <= 0 else 1 - lower


def log_student_cdf(x, m):
    x, m = mp.mpf(x), mp.mpf(m)
    a = -abs(x)
    q = m / (m + a * a)
    lower = mp.betainc(m / 2, mp.mpf(1) / 2, 0, q, regularized=True) / 2
    return mp.log(lower) if x <= 0 else mp.log1p(-lower)


def logpdf(mu, sigma, alpha, nu, y):
    mu, sigma, alpha, nu, y = (mp.mpf(v) for v in (mu, sigma, alpha, nu, y))
    z = (y - mu) / sigma
    w = alpha * z * mp.sqrt((nu + 1) / (nu + z * z))
    c = mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - mp.log(nu * mp.pi) / 2
    return (mp.log(2) + c - mp.log(sigma) - (nu + 1) / 2 * mp.log1p(z * z / nu)
            + log_student_cdf(w, nu + 1))


# ---------------------------------------------------------------- forms


def check_forms():
    import sympy as sp

    z, nu, m, s = sp.symbols("z nu m s", positive=True)
    B = nu + z ** 2
    ok = True

    # identity 1: the z-derivative of one term z^m B^-s
    for mm in range(0, 5):
        for ss in [sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(5, 2)]:
            lhs = sp.diff(z ** mm * B ** (-ss), z)
            rhs = mm * z ** (mm - 1) * B ** (-ss) - 2 * ss * z ** (mm + 1) * B ** (-ss - 1)
            if sp.simplify(lhs - rhs) != 0:
                print("identity 1 fails at m = %d, s = %s" % (mm, ss))
                ok = False

    # identity 2: the nu-derivatives of B^(-1/2)
    for q in range(0, 7):
        lhs = sp.diff(B ** sp.Rational(-1, 2), nu, q)
        phi = sp.Integer(1)
        for i in range(q):
            phi *= sp.Rational(-1, 2) - i
        rhs = phi * B ** (sp.Rational(-1, 2) - q)
        if sp.simplify(lhs - rhs) != 0:
            print("identity 2 fails at q = %d" % q)
            ok = False

    # identity 3: the rule that joins the factor z
    W = sp.Function("W")(z)
    for p in range(1, 7):
        lhs = sp.diff(z * W, z, p)
        rhs = z * sp.diff(W, z, p) + p * sp.diff(W, z, p - 1)
        if sp.simplify(lhs - rhs) != 0:
            print("identity 3 fails at p = %d" % p)
            ok = False

    # the whole recursion of u_table, reimplemented here, against SymPy's derivative of u
    u = z * sp.sqrt((nu + 1) / (nu + z ** 2))
    K = 6
    for zv, nuv in [(sp.Rational(3, 5), sp.Rational(7, 4)),
                    (sp.Rational(-9, 2), sp.Rational(1, 3)),
                    (sp.Rational(21, 10), sp.Rational(17, 2))]:
        # Q[p][q] = d^p/dz^p d^q/dnu^q B^(-1/2) by the recursion of the header
        Q = {}
        for q in range(K + 1):
            phi = sp.Integer(1)
            for i in range(q):
                phi *= sp.Rational(-1, 2) - i
            ss = sp.Rational(1, 2) + q
            c = {(0, 0): phi}
            for p in range(K - q + 1):
                if p > 0:
                    d = {}
                    for (mm, r), v in c.items():
                        if mm > 0:
                            d[(mm - 1, r)] = d.get((mm - 1, r), 0) + mm * v
                        d[(mm + 1, r + 1)] = d.get((mm + 1, r + 1), 0) - 2 * (ss + r) * v
                    c = d
                Bv = nuv + zv ** 2
                Q[(p, q)] = sum(v * zv ** mm * Bv ** (-(ss + r)) for (mm, r), v in c.items())
        P = [None] * (K + 1)
        Av = nuv + 1
        P[0] = sp.sqrt(Av)
        for i in range(1, K + 1):
            P[i] = P[i - 1] * (sp.Rational(1, 2) - (i - 1)) / Av
        for q in range(K + 1):
            for p in range(K - q + 1):
                got = 0
                for i in range(q + 1):
                    zq = zv * Q[(p, q - i)] + (p * Q[(p - 1, q - i)] if p > 0 else 0)
                    got += sp.binomial(q, i) * P[i] * zq
                want = sp.diff(u, z, p, nu, q).subs({z: zv, nu: nuv})
                if sp.simplify(sp.nsimplify(got - want)) != 0:
                    print("u_table fails at p = %d, q = %d, z = %s, nu = %s" % (p, q, zv, nuv))
                    ok = False
    print("forms: " + ("every identity holds" if ok else "FAILURES above"))
    return ok


# ---------------------------------------------------------------- the file

# (mu, sigma, alpha, nu): the limits of the family and the ordinary points
PARAMS = [
    (0.0, 1.0, 0.0, 3.5),        # the Student
    (0.0, 1.0, 0.0, 0.7),
    (-2.0, 0.5, 0.75, 3.5),
    (-2.0, 0.5, -0.75, 3.5),
    (1.5, 2.0, 2.0, 1.0),
    (1.5, 2.0, 2.0, 12.0),
    (0.0, 1.0, 5.0, 2.5),
    (0.0, 1.0, -5.0, 2.5),
    (0.0, 1.0, 50.0, 4.0),       # towards the half t
    (0.0, 1.0, 0.5, 200.0),      # towards the skew normal
    (0.0, 1.0, 2.0, 1000.0),
    (0.3, 0.25, 1.0, 0.3),       # small nu, small scale
]
# y - mu in units of sigma
ZS = [-30.0, -3.0, -0.4, 0.0, 0.9, 4.0, 25.0]


def table(mu, sigma, alpha, nu, y, bits):
    """every mixed partial of log f of order 1..6, by mpmath.diff at `bits` bits"""
    out = {}
    with mp.workprec(bits):
        f = lambda a, b, c, d: logpdf(a, b, c, d, y)
        for K in range(1, 7):
            for a in range(K + 1):
                for b in range(K - a + 1):
                    for c in range(K - a - b + 1):
                        d = K - a - b - c
                        out[(K, a, b, c, d)] = mp.diff(
                            f, (mp.mpf(mu), mp.mpf(sigma), mp.mpf(alpha), mp.mpf(nu)), (a, b, c, d))
    return out


def write_logpdf():
    path = os.path.join("tests", "data", "reference", "distrib_skewt_logpdf.csv")
    rows, dropped = [], 0
    for (mu, sigma, alpha, nu) in PARAMS:
        for zz in ZS:
            y = mu + sigma * zz
            lo_t = table(mu, sigma, alpha, nu, y, 400)
            hi_t = table(mu, sigma, alpha, nu, y, 600)
            for key in sorted(lo_t):
                a1, a2 = lo_t[key], hi_t[key]
                if abs(a1 - a2) > mp.mpf(2) ** -100 * max(abs(a2), mp.mpf(1)):
                    b1 = table(mu, sigma, alpha, nu, y, 800)[key]
                    b2 = table(mu, sigma, alpha, nu, y, 1000)[key]
                    if abs(b1 - b2) > mp.mpf(2) ** -100 * max(abs(b2), mp.mpf(1)):
                        dropped += 1
                        continue
                    a2 = b2
                K, a, b, c, d = key
                h, l = split(a2)
                rows.append([hexd(y), K, a, b, c, d, hexd(h), hexd(l), hexd(mu), hexd(sigma),
                             hexd(alpha), hexd(nu)])
            print("  %s done, %d rows so far" % (str((mu, sigma, alpha, nu, y)), len(rows)))
            sys.stdout.flush()
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "K", "a", "b", "c", "d", "hi", "lo", "mu", "sigma", "alpha", "nu"])
        w.writerows(rows)
    print("%s: %d rows, %d dropped" % (path, len(rows), dropped))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "forms"
    if what == "forms":
        sys.exit(0 if check_forms() else 1)
    elif what == "logpdf":
        write_logpdf()
    else:
        print(__doc__)
        sys.exit(2)
