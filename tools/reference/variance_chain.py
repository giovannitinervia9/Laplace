"""Candidates for the (mu, v) derivatives of log F of gaussian2, measured in double.

The chain now in the header carries the z-derivatives straight to (mu, v) with
sigma = v^(1/2), whose recursion step -((k a + i) R_i + k z R_i') subtracts its two
terms when k > 0. The alternative is to carry them to (mu, tau) with tau = 1/v, where
k = -1/2 and the step adds, and then convert the tau-derivatives to v-derivatives
through tau = 1/v.

Both routes are run here in Python floats, that is in double, from the same
z-derivatives rounded to double, and compared with the exact value at 300 bits. What
the comparison isolates is the chain, since the inputs are identical.

Usage:  python -I variance_chain.py
"""
import math

import mpmath

mpmath.mp.prec = 300
KMAX = 6


def exact_L(z, kmax):
    """L_j = d^j/dz^j log Phi(z) exactly, and the same rounded to double."""
    g = lambda t: mpmath.log(mpmath.ncdf(t))
    ex = [None] + [mpmath.diff(g, z, j) for j in range(1, kmax + 1)]
    return ex, [None] + [float(v) for v in ex[1:]]


def power_chain(L, z, sigma, w, k, K):
    """mixed_power_dk in double: returns the K + 1 components, index b."""
    out = []
    for b in range(K + 1):
        a = K - b
        q = {(0, 0): 1.0}
        for _ in range(a):                      # d/dmu: -Q'
            d = {}
            for (j, m), c in q.items():
                if m > 0:
                    d[(j, m - 1)] = d.get((j, m - 1), 0.0) + m * c
                d[(j + 1, m)] = d.get((j + 1, m), 0.0) + c
            q = {key: -c for key, c in d.items()}
        for i in range(b):                      # d/dw
            d = {}
            for (j, m), c in q.items():
                if m > 0:
                    d[(j, m - 1)] = d.get((j, m - 1), 0.0) + m * c
                d[(j + 1, m)] = d.get((j + 1, m), 0.0) + c
            r = {}
            for key, c in q.items():
                r[key] = r.get(key, 0.0) - (k * a + i) * c
            for (j, m), c in d.items():
                r[(j, m + 1)] = r.get((j, m + 1), 0.0) - k * c
            q = r
        s = 0.0
        for (j, m), c in q.items():
            if j >= 1 and c != 0.0:
                s += c * z ** m * L[j]
        f = 1.0
        for _ in range(a):
            f /= sigma
        for _ in range(b):
            f /= w
        out.append(s * f)
    return out


def bell_partial(n, j, d):
    """B_(n,j) in the derivatives d[1..] of the inner map, by the recurrence."""
    B = [[0.0] * (j + 1) for _ in range(n + 1)]
    B[0][0] = 1.0
    for m in range(1, n + 1):
        for i in range(1, min(m, j) + 1):
            s = 0.0
            for t in range(1, m - i + 2):
                s += math.comb(m - 1, t - 1) * d[t] * B[m - t][i - 1]
            B[m][i] = s
    return B[n][j]


def via_precision(L, z, v, K):
    """The (mu, tau) chain with k = -1/2, then tau = 1/v by Faa di Bruno, in double."""
    tau = 1.0 / v
    sigma = 1.0 / math.sqrt(tau)
    # the tau-derivatives of the inner map tau(v) = 1/v
    dtau = [0.0] * (K + 2)
    for n in range(1, K + 1):
        dtau[n] = (-1.0) ** n * math.factorial(n) / v ** (n + 1)
    out = []
    for b in range(K + 1):
        a = K - b
        if b == 0:
            out.append(power_chain(L, z, sigma, tau, -0.5, K)[0] if a == K else 0.0)
            continue
        # g^(i)(tau) = d^a/dmu^a d^i/dtau^i, the component at index i of the order a + i
        s = 0.0
        for i in range(1, b + 1):
            gi = power_chain(L, z, sigma, tau, -0.5, a + i)[i]
            s += gi * bell_partial(b, i, dtau)
        out.append(s)
    return out


def exact_component(z, mu, v, a, b):
    y = mu + z * mpmath.sqrt(mpmath.mpf(v))
    def f(m, w):
        return mpmath.log(mpmath.ncdf((y - m) / mpmath.sqrt(w)))
    if a and b:
        return mpmath.diff(f, (mpmath.mpf(mu), mpmath.mpf(v)), (a, b))
    if a:
        return mpmath.diff(lambda m: f(m, mpmath.mpf(v)), mpmath.mpf(mu), a)
    return mpmath.diff(lambda w: f(mpmath.mpf(mu), w), mpmath.mpf(v), b)


def ulp_of(x):
    return math.ldexp(1.0, math.frexp(abs(x))[1] - 53) if x else 0.0


print(f"{'z':>5} {'K':>2} {'current k=1/2':>15} {'via precision':>15}")
for z in (3.0, 4.0, 6.0, 8.0, 13.0):
    ex, Ld = exact_L(mpmath.mpf(z), KMAX)
    for K in (5, 6):
        v = 1.0
        a_now = power_chain(Ld, z, 1.0, v, 0.5, K)
        a_new = via_precision(Ld, z, v, K)
        exact = [exact_component(mpmath.mpf(z), 0.0, v, K - b, b) for b in range(K + 1)]
        scale = max(abs(float(e)) for e in exact)
        u = ulp_of(scale)
        w_now = max(abs(a_now[b] - float(exact[b])) / u for b in range(K + 1))
        w_new = max(abs(a_new[b] - float(exact[b])) / u for b in range(K + 1))
        print(f"{z:>5.1f} {K:>2} {w_now:>15.1f} {w_new:>15.1f}")
