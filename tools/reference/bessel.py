"""Reference values of the von Mises block of Bessel functions, computed with
mpmath.

Usage (from the repository root):
    python -I tools/reference/bessel.py [name ...] [--workers=N]
    python -I tools/reference/bessel.py coefs    (prints the coefficient tables
                                                  of include/laplace/math/bessel_i.hpp)

Files (format as in special.py: the argument, then hi, lo in hexadecimal):
    log_bessel_i0.csv           k     log I0(k)
    log_bessel_i0_scaled.csv    k     log I0(k) - k
    bessel_i_ratio.csv          k     A(k) = I1(k)/I0(k)
    bessel_i_ratio_d{n}.csv     k     A^(n)(k), n = 1..6
    bessel_i_ratio_inv.csv      rho   K(rho), the inverse of A (A(K(rho)) = rho)
    bessel_i_ratio_inv_d{n}.csv kappa K^(n)(rho) at rho = A(kappa), n = 1..6

The derivatives of the inverse take kappa, not rho, as their argument: the
caller of a von Mises distribution in (mu, rho) computes kappa = K(rho) once,
and kappa carries the information that rho loses near 1.

Method. I0 and I1 by mpmath.besseli. The derivatives of A follow from the
Riccati identity A' = 1 - A/k - A^2, differentiated n times:
A^(n+1) = -sum_j C(n, j) A^(j) (1/k)^(n-j) - sum_j C(n, j) A^(j) A^(n-j),
with (1/k)^(m) = (-1)^m m! / k^(m+1). The identity cancels heavily near 0 (its
terms are of order k^-n) and at large k (A' ~ 1/(2 k^2) against terms of order
one), so the working precision grows with |log2 k|; every row is computed at
two precisions and kept only if they agree to 2^-110 relative (dropped rows are
counted). The derivatives of the inverse come from the reversion of the Taylor
series of A at kappa (independent of the closed forms used in the library):
with A(kappa + t) = A(kappa) + sum_j a_j t^j, the series K(rho + s) =
kappa + sum_j b_j s^j satisfies A(kappa + sum_j b_j s^j) = rho + s, which gives
b_1 = 1/a_1 and each b_n from the ones below it; K^(n) = n! b_n. The inverse
itself: rho is drawn as a double (rho = A(kappa) rounded, or directly near 0
and 1), and K(rho) is found by mpmath.findroot at high precision.
"""
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261110
NMAX = 6          # highest derivative order of A and of the inverse
N_RANDOM = 4000   # random points per file (plus fixed ones)
AGREE = mpmath.mpf(2) ** -110


def dd(v):
    hi = float(v)
    if math.isinf(hi) or math.isnan(hi):
        return hi, 0.0
    return hi, float(v - mpmath.mpf(hi))


def log_uniform(rng, lo_exp, hi_exp):
    return math.ldexp(1.0 + rng.random(), rng.randint(lo_exp, hi_exp))


def kappa_sample(rng, lo_exp=-60, hi_exp=45):
    """kappa > 0: log-uniform over the range, uniform where the algorithms
    change (the power series below 0.5, the continued fraction up to a few
    tens, the asymptotic series above), and points just around 0.5 and 30."""
    xs = [log_uniform(rng, lo_exp, hi_exp) for _ in range(N_RANDOM // 2)]
    xs += [rng.uniform(0.0, 2.0) for _ in range(N_RANDOM // 8)]
    xs += [rng.uniform(0.0, 60.0) for _ in range(N_RANDOM // 4)]
    xs += [rng.uniform(60.0, 1000.0) for _ in range(N_RANDOM // 8)]
    for c in (0.5, 1.0, 20.0, 30.0, 50.0, 100.0):
        xs += [c * (1.0 + rng.uniform(-1e-3, 1e-3)) for _ in range(20)]
        xs += [c, math.nextafter(c, 0.0), math.nextafter(c, math.inf)]
    xs += [1e-300, 5e-324, 1e-10, 1e-5, 0.1, 2.0, 10.0, 1e3, 1e6, 1e10, 1e15]
    return [x for x in xs if x > 0.0]


# ---------------------------------------------------------------- the functions at a given precision

def prec_for(k, order):
    """bits needed for the Riccati recursion to order `order` at k"""
    lk = abs(math.log2(k)) if k > 0 else 1100
    return 160 + (order + 2) * int(lk + 4)


def ratio_derivs(k, prec, nmax=NMAX):
    """A(k), A'(k), ..., A^(nmax)(k) at `prec` bits"""
    with mpmath.workprec(prec):
        k = mpmath.mpf(k)
        A = [mpmath.besseli(1, k) / mpmath.besseli(0, k)]
        inv = [(-1) ** m * mpmath.factorial(m) / k ** (m + 1) for m in range(nmax)]
        for n in range(nmax):
            s = 0
            for j in range(n + 1):
                c = mpmath.binomial(n, j)
                s += c * A[j] * inv[n - j] + c * A[j] * A[n - j]
            A.append((1 if n == 0 else 0) - s)
        return [+a for a in A]


def inverse_derivs(k, prec, nmax=NMAX):
    """K'(rho), ..., K^(nmax)(rho) at rho = A(k), by series reversion"""
    A = ratio_derivs(k, prec, nmax)
    with mpmath.workprec(prec):
        a = [0] + [A[j] / mpmath.factorial(j) for j in range(1, nmax + 1)]   # a[j], j >= 1

        def mul(p, q):  # truncated product of series without constant term
            r = [mpmath.mpf(0)] * (nmax + 1)
            for i in range(1, nmax + 1):
                if p[i] == 0:
                    continue
                for j in range(1, nmax + 1 - i):
                    r[i + j] += p[i] * q[j]
            return r

        b = [mpmath.mpf(0)] * (nmax + 1)
        b[1] = 1 / a[1]
        for n in range(2, nmax + 1):
            # coefficient of s^n in sum_{j >= 2} a_j g^j with g = b_1 s + ... + b_(n-1) s^(n-1)
            g = b[:n] + [mpmath.mpf(0)] * (nmax + 1 - n)
            power, c = g, 0
            for j in range(2, n + 1):
                power = mul(power, g)
                c += a[j] * power[n]
            b[n] = -c / a[1]
        return [mpmath.factorial(n) * b[n] for n in range(1, nmax + 1)]


def log_i0(k, prec):
    with mpmath.workprec(prec):
        k = mpmath.mpf(k)
        if k < 1:
            return mpmath.log1p(mpmath.besseli(0, k) - 1)
        return mpmath.log(mpmath.besseli(0, k))


def inverse(rho, prec):
    """K(rho) for a double rho in (0, 1)"""
    with mpmath.workprec(prec):
        r = mpmath.mpf(rho)
        A = lambda k: mpmath.besseli(1, k) / mpmath.besseli(0, k) - r
        # start from the approximations k ~ 2 rho (small) and 1/(2(1 - rho)) (large)
        k0 = 2 * r if rho < 0.5 else (1 / (2 * (1 - r)) if rho > 0.9 else 2 * r / (1 - r * r))
        return mpmath.findroot(A, k0, tol=mpmath.mpf(2) ** (-prec + 20))


def agree(u, v):
    return abs(u - v) <= AGREE * max(abs(u), abs(v), mpmath.mpf(2) ** -1070)


# ---------------------------------------------------------------- rows

def row_log_i0(k):
    p = 300
    u, v = log_i0(k, p), log_i0(k, p + 64)
    return (k, v) if agree(u, v) else None


def row_log_i0_scaled(k):
    p = 300 + int(math.log2(k + 1))
    u, v = log_i0(k, p) - k, log_i0(k, p + 64) - k
    return (k, v) if agree(u, v) else None


def row_ratio(k):
    p = prec_for(k, NMAX)
    u, v = ratio_derivs(k, p), ratio_derivs(k, p + 96)
    return (k, v) if all(agree(a, b) for a, b in zip(u, v)) else None


def row_inverse_derivs(k):
    p = prec_for(k, NMAX + 2)
    u, v = inverse_derivs(k, p), inverse_derivs(k, p + 96)
    return (k, v) if all(agree(a, b) for a, b in zip(u, v)) else None


def row_inverse(rho):
    p = 200 + 2 * int(-math.log2(1 - rho)) + 2 * int(-math.log2(rho))
    try:
        u, v = inverse(rho, p), inverse(rho, p + 64)
    except (ValueError, ZeroDivisionError):
        return None
    return (rho, v) if agree(u, v) else None


def rho_sample(rng):
    """rho in (0, 1): A(kappa) rounded for the kappa sample, plus points near 0
    and near 1 (1 - rho down to 2^-52)"""
    ks = [k for k in kappa_sample(rng, -60, 50) if k > 1e-300]
    rhos = []
    for k in ks[: N_RANDOM // 2]:
        with mpmath.workprec(120):
            r = float(mpmath.besseli(1, k) / mpmath.besseli(0, k))
        if 0.0 < r < 1.0:
            rhos.append(r)
    rhos += [log_uniform(rng, -60, -1) for _ in range(N_RANDOM // 4)]
    rhos += [1.0 - log_uniform(rng, -52, -2) for _ in range(N_RANDOM // 4)]
    rhos += [rng.uniform(0.0, 1.0) for _ in range(N_RANDOM // 4)]
    rhos += [0.5, 0.53, 0.85, 0.9, 0.99, 1.0 - 2.0 ** -52, 1.0 - 2.0 ** -53, 1e-300]
    return [r for r in rhos if 0.0 < r < 1.0]


# ---------------------------------------------------------------- driver

def write(name, rows):
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="\n") as f:
        f.write("x,hi,lo\n")
        for x, v in rows:
            hi, lo = dd(v)
            f.write(f"{float(x).hex()},{hi.hex()},{lo.hex()}\n")
    print(f"{path.name}: {len(rows)} rows")


def run(pool, fn, xs):
    out = pool.map(fn, xs, chunksize=8)
    rows = [r for r in out if r is not None]
    dropped = len(xs) - len(rows)
    if dropped:
        print(f"  {fn.__name__}: {dropped} rows dropped (precisions disagree)")
    return rows


def coefs(na=26, nq=60):
    """The coefficient tables of include/laplace/math/bessel_i.hpp, exact
    rationals rounded once to double:
      a_n: A(k) = sum_n a_n k^(2n+1), the power series at 0 (radius 2.405,
           the first zero of I0 on the imaginary axis);
      q_n: A(k) ~ sum_n q_n k^(-n), the quotient of the asymptotic series of
           I1 and I0, whose coefficients are (-1)^n a_n(nu) with
           a_n(nu) = prod_{j=1}^n (4 nu^2 - (2j - 1)^2) / (n! 8^n)."""
    from fractions import Fraction as F
    i0 = [F(1, 4 ** m * math.factorial(m) ** 2) for m in range(na)]
    i1 = [F(1, 2 * 4 ** m * math.factorial(m) * math.factorial(m + 1)) for m in range(na)]
    a = []
    for n in range(na):
        a.append(i1[n] - sum(a[j] * i0[n - j] for j in range(n)))

    def an(nu, n):
        p = F(1)
        for j in range(1, n + 1):
            p *= 4 * nu * nu - (2 * j - 1) ** 2
        return p / (math.factorial(n) * 8 ** n)

    b0 = [(-1) ** n * an(0, n) for n in range(nq)]
    b1 = [(-1) ** n * an(1, n) for n in range(nq)]
    q = []
    for n in range(nq):
        q.append(b1[n] - sum(q[j] * b0[n - j] for j in range(n)))
    for name, vals in (("ratio_series", a), ("ratio_asymptotic", q)):
        print(f"constexpr double {name}[{len(vals)}] = {{")
        for v in vals:
            print(f"    {float(v).hex()},  // {float(v)!r}")
        print("};")

    # The derivatives of the inverse for large kappa, normalized by their
    # leading term: K^(n)(A(k)) = n! 2^n k^(n+1) (1 + sum_(m >= 1) e_(n,m) u^m),
    # u = 1/k. Series in u with exact rational coefficients: d/dk = -u^2 d/du,
    # A^(j) from A(u) = sum q_n u^n, then K^(n) by the inverse function rule
    # (K' = 1/A', K^(n+1) = (d/dk K^(n)) / A').
    N = nq  # order of truncation in u

    def smul(x, y):
        r = [F(0)] * N
        for i, xi in enumerate(x):
            if xi:
                for j in range(N - i):
                    r[i + j] += xi * y[j]
        return r

    # Laurent series as (offset, coefficients): value = u^off * sum c_i u^i
    def lmul(x, y):
        return (x[0] + y[0], smul(x[1], y[1]))

    def linv(x):
        off, c = x
        k0 = next(i for i, v in enumerate(c) if v)
        c = c[k0:] + [F(0)] * k0
        r = [F(0)] * N
        r[0] = 1 / c[0]
        for n in range(1, N):
            r[n] = -sum(c[j] * r[n - j] for j in range(1, n + 1)) / c[0]
        return (-(off + k0), r)

    def lderiv_k(x):  # d/dk = -u^2 d/du
        off, c = x
        r = [-(i + off) * ci for i, ci in enumerate(c)]
        return (off + 1, r)

    A = (0, list(q))
    A1 = lderiv_k(A)
    w = linv(A1)
    Kn = w
    rows = []
    for n in range(1, NMAX + 1):
        if n > 1:
            Kn = lmul(lderiv_k(Kn), w)
        off, c = Kn
        k0 = next(i for i, v in enumerate(c) if v)
        lead = c[k0]
        assert off + k0 == -(n + 1) and lead == math.factorial(n) * 2 ** n, (n, off + k0, lead)
        rows.append([v / lead for v in c[k0 + 1:k0 + 1 + 45]])
    print(f"constexpr double inverse_asymptotic[{NMAX}][45] = {{")
    for r in rows:
        print("    {" + ", ".join(float(v).hex() for v in r) + "},")
    print("};")

    # The same derivatives for small kappa, as power series in k:
    # K^(n)(A(k)) = k^p sum_m g_(n,m) k^(2m), p = 0 for odd n and 1 for even n
    # (A is odd in k, K odd in rho). Plain power series, d/dk term by term.
    M = 2 * na + 1  # degree of truncation in k

    def pmul(x, y):
        r = [F(0)] * M
        for i, xi in enumerate(x):
            if xi:
                for j in range(M - i):
                    r[i + j] += xi * y[j]
        return r

    def pinv(c):
        r = [F(0)] * M
        r[0] = 1 / c[0]
        for n in range(1, M):
            r[n] = -sum(c[j] * r[n - j] for j in range(1, n + 1)) / c[0]
        return r

    def pderiv(c):
        return [(i + 1) * c[i + 1] for i in range(M - 1)] + [F(0)]

    Ak = [F(0)] * M
    for n, an_ in enumerate(a):
        if 2 * n + 1 < M:
            Ak[2 * n + 1] = an_
    wk = pinv(pderiv(Ak))
    Kn = wk
    rows = []
    for n in range(1, NMAX + 1):
        if n > 1:
            Kn = pmul(pderiv(Kn), wk)
        p = 0 if n % 2 else 1
        assert all(Kn[i] == 0 for i in range(1 - p, M - 2, 2)), n
        rows.append([Kn[p + 2 * m] for m in range(24)])
    print(f"constexpr double inverse_series[{NMAX}][24] = {{")
    for r in rows:
        print("    {" + ", ".join(float(v).hex() for v in r) + "},")
    print("};")


def main():
    if sys.argv[1:2] == ["coefs"]:
        coefs()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = 4
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
    want = set(args) or {"log_i0", "ratio", "inverse", "inverse_derivs"}
    rng = random.Random(SEED)
    ks = sorted(set(kappa_sample(rng)))
    with Pool(workers) as pool:
        if "log_i0" in want:
            write("log_bessel_i0", run(pool, row_log_i0, ks + [0.0]))
            write("log_bessel_i0_scaled", run(pool, row_log_i0_scaled, ks + [0.0]))
        if "ratio" in want:
            rows = run(pool, row_ratio, ks)
            write("bessel_i_ratio", [(k, v[0]) for k, v in rows])
            for n in range(1, NMAX + 1):
                write(f"bessel_i_ratio_d{n}", [(k, v[n]) for k, v in rows])
        if "inverse" in want:
            rhos = sorted(set(rho_sample(random.Random(SEED + 1))))
            write("bessel_i_ratio_inv", run(pool, row_inverse, rhos))
        if "inverse_derivs" in want:
            rows = run(pool, row_inverse_derivs, ks)
            for n in range(1, NMAX + 1):
                write(f"bessel_i_ratio_inv_d{n}", [(k, v[n - 1]) for k, v in rows])


if __name__ == "__main__":
    main()
