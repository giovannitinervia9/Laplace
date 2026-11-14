"""Reference values for the two-into-two chain rule to order 6, with mpmath.

Usage (from the repository root):  python -I tools/reference/chain_pair.py

File tests/data/reference/chain_pair.csv, one row per case and order:
    case, K, then G[0..T(K)-1], u0[1..T(K)-1], u1[1..T(K)-1],
    then for b = 0..K the pair d_hi, d_lo
where T(s) = (s+1)(s+2)/2, every input is a double written in hexadecimal, the
multi-index (i, j) sits at position (i+j)(i+j+1)/2 + j (by total order, then by
the number of derivatives in the second variable), and d_b is
d^K H / dx_0^(K-b) dx_1^b for H(x) = G(u_0(x), u_1(x)), computed at 300 bits
from exactly those doubles.

Method, independent of the recursion of include/laplace/math/chain_pair.hpp:
the composition is done on truncated bivariate Taylor series. With
    U_r(e) = sum_(1 <= p + q <= K) u_r[p][q] e_0^p e_1^q / (p! q!),
    H(e)   = sum_(i + j <= K) G[i][j] U_0(e)^i U_1(e)^j / (i! j!),
the derivative d^(p,q) H is p! q! times the coefficient of e_0^p e_1^q in H,
every product truncated at total order K.

Cases: (a) derivatives drawn at random, magnitudes log-uniform on (2^-8, 2^8)
with random signs, so that the sums cancel; (b) the two maps this batch needs,
the shape and ratio of gamma2 (u_0 = mu^2/v, u_1 = y/mu, in (mu, v)) and the
map of lognormal2 (u_0 = log m - s/2, u_1 = s with s = log(1 + v/m^2), in
(m, v)), with an outer function of the shape of a log-density
(G = -u_0 D(u_1) + log(u_0)/2 with D(z) = z - 1 - log z), every partial
derivative computed by mpmath.diff and rounded to a double.
"""
import random
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261113
KMAX = 6
mpmath.mp.prec = 300


def tri(i, j):
    return ((i + j) * (i + j + 1)) // 2 + j


def tsize(s):
    return ((s + 1) * (s + 2)) // 2


def dd(v):
    hi = float(v)
    return hi, float(mpmath.mpf(v) - mpmath.mpf(hi))


def hx(v):
    return float(v).hex()


# ---- truncated bivariate series, A[(p, q)] with p + q <= K ----------------
def smul(a, b, K):
    out = {}
    for (p1, q1), v1 in a.items():
        for (p2, q2), v2 in b.items():
            if p1 + p2 + q1 + q2 > K:
                continue
            key = (p1 + p2, q1 + q2)
            out[key] = out.get(key, mpmath.mpf(0)) + v1 * v2
    return out


def compose(G, u0, u1, K):
    """d^(K-b, b) of G(u_0, u_1), b = 0..K, from the packed input arrays"""
    series = []
    for u in (u0, u1):
        s = {}
        for n in range(1, K + 1):
            for j in range(n + 1):
                i = n - j
                s[(i, j)] = mpmath.mpf(u[tri(i, j)]) / (mpmath.factorial(i) * mpmath.factorial(j))
        series.append(s)
    powers = []
    for s in series:
        p = [{(0, 0): mpmath.mpf(1)}]
        for _ in range(K):
            p.append(smul(p[-1], s, K))
        powers.append(p)
    H = {}
    for n in range(K + 1):
        for j in range(n + 1):
            i = n - j
            c = mpmath.mpf(G[tri(i, j)]) / (mpmath.factorial(i) * mpmath.factorial(j))
            if c == 0:
                continue
            term = smul(powers[0][i], powers[1][j], K)
            for key, v in term.items():
                H[key] = H.get(key, mpmath.mpf(0)) + c * v
    out = []
    for b in range(K + 1):
        p, q = K - b, b
        out.append(H.get((p, q), mpmath.mpf(0)) * mpmath.factorial(p) * mpmath.factorial(q))
    return out


# ---- the partial derivatives of an expression, by mpmath.diff -------------
def partials(f, x0, x1, K):
    """f(x_0, x_1) -> the packed array of its partials to total order K"""
    v = [0.0] * tsize(K)
    for n in range(K + 1):
        for j in range(n + 1):
            i = n - j
            v[tri(i, j)] = float(mpmath.re(mpmath.diff(f, (x0, x1), (i, j))))
    return v


def main():
    random.seed(SEED)
    rows = []
    # (a) random derivatives
    for case in range(60):
        K = 1 + case % KMAX
        G = [0.0] * tsize(K)
        u0 = [0.0] * tsize(K)
        u1 = [0.0] * tsize(K)
        for n in range(K + 1):
            for j in range(n + 1):
                i = n - j
                for arr in (G, u0, u1):
                    arr[tri(i, j)] = random.choice((-1.0, 1.0)) * 2.0 ** random.uniform(-8.0, 8.0)
        rows.append((case, K, G, u0, u1, compose(G, u0, u1, K)))
    # (b) the two maps of the batch, with a log-density-shaped outer function
    def dev(z):
        return z - 1 - mpmath.log(z)

    def outer(a, c):
        return -a * dev(c) + mpmath.log(a) / 2

    maps = []
    for mu, v, y in ((2.0, 0.5, 1.4), (0.3, 9.0, 0.05), (40.0, 0.001, 41.0)):
        maps.append(("gamma2", lambda m, w, y=y: m * m / w, lambda m, w, y=y: y / m, mu, v))
    for m, v in ((2.0, 0.5), (40.0, 0.001), (0.3, 9.0)):
        s = lambda a, b: mpmath.log(1 + b / (a * a))
        maps.append(("lognormal2", lambda a, b, s=s: mpmath.log(a) - s(a, b) / 2, s, m, v))
    case = 100
    for name, f0, f1, x0, x1 in maps:
        a0, a1 = f0(mpmath.mpf(x0), mpmath.mpf(x1)), f1(mpmath.mpf(x0), mpmath.mpf(x1))
        for K in range(1, KMAX + 1):
            G = partials(lambda a, c: outer(a, c), a0, a1, K)
            u0 = partials(f0, mpmath.mpf(x0), mpmath.mpf(x1), K)
            u1 = partials(f1, mpmath.mpf(x0), mpmath.mpf(x1), K)
            rows.append((case, K, G, u0, u1, compose(G, u0, u1, K)))
        case += 1

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "chain_pair.csv").open("w", newline="") as fh:
        fh.write("case,K,inputs...,values(hi,lo)...\n")
        for case, K, G, u0, u1, d in rows:
            cells = [str(case), str(K)]
            cells += [hx(x) for x in G]
            cells += [hx(x) for x in u0[1:]]
            cells += [hx(x) for x in u1[1:]]
            for v in d:
                hi, lo = dd(v)
                cells += [hx(hi), hx(lo)]
            fh.write(",".join(cells) + "\n")
    print(f"chain_pair.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
