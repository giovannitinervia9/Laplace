"""mpmath references for the Gaussian-lasso integral
(include/laplace/math/gaussian_laplace.hpp).

Run with python -I. Writes tests/data/reference/gaussian_laplace.csv.

  I(m, Sigma, lam) = int N(beta; m, Sigma) exp(-sum_i lam_i |beta_i|) dbeta,

and the mean and covariance of the law proportional to the integrand. Sigma has
one-factor correlation: beta_i = m_i + sd_i (l_i z + sqrt(1 - l_i^2) e_i) with
z, e_i independent standard normals. Given z the coordinates are independent and
each one-dimensional integral is in closed form,

  J_i(z) = exp(lam^2 tau^2 / 2) (exp(-lam mu) Phi(mu/tau - lam tau) + exp(lam mu) Phi(-mu/tau - lam tau)),

mu = m_i + sd_i l_i z, tau = sd_i sqrt(1 - l_i^2), the two terms the positive and
the negative half-line, on which the law is a truncated normal of mean
mu -+ lam tau^2. So I = int phi(z) prod_i J_i(z) dz, and the moments are 1-D
integrals of the conditional moments of the two truncated pieces. The
integrals are taken by mpmath's tanh-sinh on pieces cut around the mode of the
(log-concave) integrand at multiples of its width, at 40 digits and again at
50 digits on cuts 1.3 times wider; a row is kept only when the two agree
(1e-28 relative for log I, 1e-24 for the moments).

Columns (k <= KM, unused slots 0): k, m[KM], sd[KM], l[KM], lam[KM], then
hi, lo of log I, then the mean [KM] and the covariance packed lower
(row i, columns j <= i) [KM (KM + 1) / 2], all hexadecimal.
"""
import sys
import random
import multiprocessing
import numpy as np
import mpmath as mp
from scipy.special import log_ndtr
from scipy import optimize

OUT = "tests/data/reference"
KM = 5


def log_J_double(mu, tau, lam):
    lt = lam * tau
    return 0.5 * lt * lt + np.logaddexp(-lam * mu + log_ndtr(mu / tau - lt), lam * mu + log_ndtr(-mu / tau - lt))


def integrals(m, sd, l, lam, dps, wscale):
    k = len(m)
    tau = [s * np.sqrt(1 - x * x) for s, x in zip(sd, l)]
    fd = lambda z: -0.5 * z * z + sum(log_J_double(m[i] + sd[i] * l[i] * z, tau[i], lam[i]) for i in range(k))
    if all(x == 0 for x in l):
        z0, w = 0.0, 1.0
    else:
        zs = np.linspace(-60, 60, 2401)
        z0 = zs[np.argmax([fd(z) for z in zs])]
        z0 = optimize.minimize_scalar(lambda z: -fd(z), bounds=(z0 - 0.1, z0 + 0.1), method="bounded",
                                      options={"xatol": 1e-12}).x
        h = 1e-4
        w = 1 / np.sqrt(max(-(fd(z0 + h) - 2 * fd(z0) + fd(z0 - h)) / h**2, 1e-6))
    with mp.workdps(dps):
        M = [mp.mpf(x) for x in m]; SD = [mp.mpf(x) for x in sd]; L = [mp.mpf(x) for x in l]
        LA = [mp.mpf(x) for x in lam]; TAU = [SD[i] * mp.sqrt(1 - L[i] ** 2) for i in range(k)]
        Z0 = mp.mpf(z0)
        F0 = mp.mpf(fd(z0))

        def cond(z):
            """log of phi(z) prod J_i (shifted by F0), and per coordinate the conditional E[beta], E[beta^2]"""
            lf = -z * z / 2 - F0
            m1, m2 = [], []
            for i in range(k):
                mu, t, la = M[i] + SD[i] * L[i] * z, TAU[i], LA[i]
                up, um = mu / t - la * t, -mu / t - la * t
                lA = la * la * t * t / 2 - la * mu + mp.log(mp.ncdf(up))
                lB = la * la * t * t / 2 + la * mu + mp.log(mp.ncdf(um))
                lJ = lA + mp.log1p(mp.exp(lB - lA)) if lA > lB else lB + mp.log1p(mp.exp(lA - lB))
                a, b = mp.exp(lA - lJ), mp.exp(lB - lJ)
                rp, rm = mp.npdf(up) / mp.ncdf(up), mp.npdf(um) / mp.ncdf(um)
                m1.append(a * t * (up + rp) - b * t * (um + rm))
                m2.append(a * t * t * (up * up + 1 + up * rp) + b * t * t * (um * um + 1 + um * rm))
                lf += lJ
            return mp.exp(lf), m1, m2

        W = mp.mpf(w) * wscale
        pts = [-mp.inf] + [Z0 + d * W for d in (-40, -12, -4, -1, 0, 1, 4, 12, 40)] + [mp.inf]

        def integ(fun):
            def g(z):
                if abs(z) > 1e6:
                    return mp.mpf(0)
                f, m1, m2 = cond(z)
                return f * fun(m1, m2)
            return mp.quad(g, pts)

        I0 = integ(lambda m1, m2: 1)
        logI = mp.log(I0) + F0 - mp.log(mp.sqrt(2 * mp.pi))
        E = [integ(lambda m1, m2, i=i: m1[i]) / I0 for i in range(k)]
        C = []
        for i in range(k):
            for j in range(i + 1):
                if i == j:
                    v = integ(lambda m1, m2, i=i: m2[i]) / I0 - E[i] ** 2
                else:
                    v = integ(lambda m1, m2, i=i, j=j: m1[i] * m1[j]) / I0 - E[i] * E[j]
                C.append(v)
        return logI, E, C


def rows(rng):
    out = []
    for k in range(1, KM + 1):
        for c in ((0.0,) if k == 1 else (0.5, 0.9, 0.99)):
            for t in (0.5, 2.0, 5.0, 10.0, 20.0):
                for mode in ("zero", "random", "far"):
                    sd = [rng.uniform(0.5, 2.0) for _ in range(k)]
                    l = [rng.uniform(-c, c) for _ in range(k)]
                    if k > 1:
                        l[0] = c
                    lam = [t / s * rng.uniform(0.7, 1.3) for s in sd]
                    if mode == "zero":
                        m = [0.0] * k
                    elif mode == "random":
                        m = [s * rng.uniform(-1.5, 1.5) for s in sd]
                    else:  # the mode of the integrand off the kink on half the coordinates
                        m = [s * (rng.choice((-1, 1)) * (lam[i] * s + 4) if i % 2 == 0 else rng.uniform(-1, 1))
                             for i, s in enumerate(sd)]
                    out.append((k, m, sd, l, lam))
    return out


def row(r):
    k, m, sd, l, lam = r
    a = integrals(m, sd, l, lam, 40, 1.0)
    b = integrals(m, sd, l, lam, 50, 1.3)
    with mp.workdps(50):
        if abs(a[0] - b[0]) > mp.mpf(10) ** -28 * max(1, abs(b[0])):
            print("disagree logI", r, file=sys.stderr)
            return None
        scale = max(abs(x) for x in b[1] + b[2])
        for x, y in zip(a[1] + a[2], b[1] + b[2]):
            if abs(x - y) > mp.mpf(10) ** -24 * scale:
                print("disagree moments", r, file=sys.stderr)
                return None
        hi = float(b[0]); lo = float(b[0] - mp.mpf(hi))
    pad = lambda v, n: list(v) + [0.0] * (n - len(v))
    cols = ([float(k)] + pad(m, KM) + pad(sd, KM) + pad(l, KM) + pad(lam, KM) + [hi, lo]
            + pad([float(x) for x in b[1]], KM) + pad([float(x) for x in b[2]], KM * (KM + 1) // 2))
    return ",".join(float.hex(float(x)) for x in cols)


def main():
    rng = random.Random(20261119)
    rr = rows(rng)
    with multiprocessing.Pool(8) as pool:
        out = [o for o in pool.map(row, rr, chunksize=1) if o is not None]
    with open(f"{OUT}/gaussian_laplace.csv", "w") as fh:
        fh.write(f"k,m[{KM}],sd[{KM}],l[{KM}],lam[{KM}],hi,lo,mean[{KM}],cov packed lower[{KM * (KM + 1) // 2}]\n")
        fh.write("\n".join(out) + "\n")
    print("gaussian_laplace", len(out), "of", len(rr), "rows")


if __name__ == "__main__":
    main()
