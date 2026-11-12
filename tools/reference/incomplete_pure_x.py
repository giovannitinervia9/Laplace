"""Independent references for the pure derivatives in x (or in c = x/a at
fixed a) of log P(a, x) and log Q(a, x), orders 1 to 7, computed with mpmath.

Usage (from the repository root):
    python -I tools/reference/incomplete_pure_x.py <points file> <output file>

The points file has one "a c" pair per line. Each output line is
    a c K p q p_r q_r
with a and c as hexadecimal doubles, K the order, p and q the values of
d^K log P(a, a c) / dc^K and d^K log Q(a, a c) / dc^K (at fixed a; a^K times
the derivatives in x) with x = a c formed exactly, and p_r, q_r the same at
x = fl(a c), the product rounded to double (the measure of what the rounding
of x costs).

Method. No quadrature: with D = x^a e^-x / Gamma(a),
    S(a, x (1 + e)) = S(a, x) +- D int_0^e (1 + u)^(a - 1) e^(-x u) du
(+ for P, - for Q), the integrand is exp((a - 1) log(1 + u) - x u) as a power
series in u, S(a, x) is mpmath's gammainc, and log S(a, x (1 + e)) follows as
a power series in e; d^K / dc^K = K! [e^K] / c^K. Every value at 300 and 400
bits, which must agree to 1e-60 relative. Used on 2026-11-12 (night) to
check the files of incomplete_ratio.py at large a near c = 1 (agreement to
the double) and to measure the routes there (docs/numerics.md).
"""
import math
import sys

import mpmath

K = 7


def ser_mul(p, q):
    r = [mpmath.mpf(0)] * (K + 1)
    for i, u in enumerate(p):
        for j, w in enumerate(q):
            if i + j <= K:
                r[i + j] += u * w
    return r


def ser_exp(L):
    # E = exp(L), L[0] = 0
    E = [mpmath.mpf(0)] * (K + 1)
    E[0] = mpmath.mpf(1)
    for n in range(1, K + 1):
        E[n] = sum(m * L[m] * E[n - m] for m in range(1, n + 1)) / n
    return E


def ser_log(F):
    L = [mpmath.mpf(0)] * (K + 1)
    for n in range(1, K + 1):
        L[n] = (n * F[n] - sum(m * L[m] * F[n - m] for m in range(1, n))) / (n * F[0])
    L[0] = mpmath.log(F[0])
    return L


def pure(a, c, prec):
    """x = a c exactly"""
    with mpmath.workprec(prec):
        A = mpmath.mpf(a)
        X = A * mpmath.mpf(c)
        # (1+u)^(a-1) e^(-x u) = exp((a-1) log(1+u) - x u)
        L = [mpmath.mpf(0)] * (K + 1)
        for m in range(1, K + 1):
            L[m] = (A - 1) * mpmath.mpf((-1) ** (m - 1)) / m
        L[1] -= X
        E = ser_exp(L)
        I = [mpmath.mpf(0)] + [E[m - 1] / m for m in range(1, K + 1)]
        logD = A * mpmath.log(X) - X - mpmath.loggamma(A)
        P = mpmath.gammainc(A, 0, X, regularized=True)
        Q = mpmath.gammainc(A, X, mpmath.inf, regularized=True)
        D = mpmath.exp(logD)
        out = {}
        for name, S, sg in (("p", P, 1), ("q", Q, -1)):
            F = [S] + [sg * D * I[m] for m in range(1, K + 1)]
            out[name] = ser_log(F)
        return out


if __name__ == "__main__":
    # arguments: file with lines a c ; writes a c K p_vK q_vK (d^K/dc^K at fixed a) and the x-derivative
    inp, outp = sys.argv[1], sys.argv[2]
    rows = []
    for line in open(inp):
        if not line.strip():
            continue
        a, c = (float(t) for t in line.split())
        r1 = pure(a, c, 300)
        r2 = pure(a, c, 400)
        xr = a * c  # rounded to double
        with mpmath.workprec(400):
            cr = mpmath.mpf(xr) / mpmath.mpf(a)
        r3 = pure(a, cr, 400)
        for k in range(1, K + 1):
            vals = []
            for nm in ("p", "q"):
                v1 = r1[nm][k] * math.factorial(k) / mpmath.mpf(c) ** k
                v2 = r2[nm][k] * math.factorial(k) / mpmath.mpf(c) ** k
                assert abs(v1 - v2) <= 1e-60 * abs(v2) + mpmath.mpf(10) ** -300, (a, c, k, v1, v2)
                vals.append(v2)
            for nm in ("p", "q"):
                with mpmath.workprec(400):
                    vals.append(r3[nm][k] * math.factorial(k) / cr ** k)
            rows.append(f"{float.hex(a)} {float.hex(c)} {k} " + " ".join(mpmath.nstr(v, 25) for v in vals))
    with open(outp, "w") as fh:
        fh.write("\n".join(rows) + "\n")
