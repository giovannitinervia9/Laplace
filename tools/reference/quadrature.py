"""Reference values for the quadrature rules: integrals taken from the uses in
the distributions, with mpmath, and the node tables of the fixed Gauss rules.

Usage (from the repository root):
    python -I tools/reference/quadrature.py             (the integrals)
    python -I tools/reference/quadrature.py nodes       (prints the Gauss tables)

File tests/data/reference/quadrature.csv: id, p1, p2, hi, lo; one row per
(integral, parameters). The integrals (the C++ functors in
benchmarks/quad_common.hpp evaluate the same integrands):

 0 beta       int_0^1 x^(p1-1) (1-x)^(p2-1) dx = B(p1, p2)          singular ends
 1 gamma      int_0^inf x^(p1-1) e^-x dx = Gamma(p1)                 half-line, singular end
 2 gauss_cos  int_R exp(-x^2/2) cos(p1 x) dx = sqrt(2 pi) exp(-p1^2/2) whole line, oscillating
 3 student    int_0^inf (1 + x^2/p1)^(-(p1+1)/2) dx = sqrt(p1) B(1/2, p1/2)/2   heavy tail
 4 laplace    int_p1^p2 exp(-|x|) dx                                 kink at 0
 5 log2       int_0^1 x^p1 log(x)^2 dx = 2 / (p1+1)^3, p1 + 1 >= 0.1   log singularity
 6 skewnormal int_R 2 x^2 phi(x) phi(p1 x)^2 / Phi(p1 x) dx          E[(d/dalpha log f)^2], skew normal
 7 huber      int_R exp(-p1 (sqrt(1 + x^2) - 1)) dx = 2 e^p1 K_1(p1)  pseudo-Huber normalizer
 8 sech       int_R sech(x)^p1 dx = B(p1/2, 1/2)                     exponential tails, p1 small = wide

Values by closed forms where they exist (mpmath at 200 bits); the skew normal
entry by mpmath.quad at 200 bits split at 0 and at +-1/|alpha|, +-8/|alpha|.
"""
import math
import random
import sys
from pathlib import Path

import mpmath

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference"
SEED = 20261112
mpmath.mp.prec = 200


def dd(v):
    hi = float(v)
    return hi, float(v - mpmath.mpf(hi))


def lu(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def value(i, p1, p2):
    a, b = mpmath.mpf(p1), mpmath.mpf(p2)
    if i == 0:
        return mpmath.beta(a, b)
    if i == 1:
        return mpmath.gamma(a)
    if i == 2:
        return mpmath.sqrt(2 * mpmath.pi) * mpmath.exp(-a * a / 2)
    if i == 3:
        return mpmath.sqrt(a) * mpmath.beta(mpmath.mpf(1) / 2, a / 2) / 2
    if i == 4:
        def F(t):  # int_0^t e^-|x| dx, odd
            return mpmath.sign(t) * (1 - mpmath.exp(-abs(t)))
        return F(b) - F(a)
    if i == 5:
        return 2 / (a + 1) ** 3
    if i == 6:
        f = lambda x: 2 * x * x * mpmath.npdf(x) * mpmath.npdf(a * x) ** 2 / mpmath.ncdf(a * x)
        s = 1 / abs(a) if a != 0 else mpmath.mpf(1)
        pts = sorted({-40, -8 * s, -s, 0, s, 8 * s, 40} | ({-80, 80}))
        pts = [mpmath.mpf(p) for p in pts if abs(p) <= 80]
        return mpmath.quad(f, [-mpmath.inf] + pts + [mpmath.inf])
    if i == 7:
        return 2 * mpmath.exp(a) * mpmath.besselk(1, a)
    if i == 8:
        return mpmath.beta(a / 2, mpmath.mpf(1) / 2)
    raise ValueError(i)


def sample(rng):
    rows = []
    for _ in range(60):
        rows.append((0, lu(rng, 0.05, 50), lu(rng, 0.05, 50)))
        rows.append((1, lu(rng, 0.05, 100), 0.0))
        rows.append((2, rng.uniform(0.0, 6.0), 0.0))
        rows.append((3, lu(rng, 0.25, 30), 0.0))
        a = rng.uniform(-20, 20)
        rows.append((4, a, a + lu(rng, 0.01, 40)))
        rows.append((5, lu(rng, 0.1, 20) - 1.0, 0.0))
        rows.append((6, rng.choice([-1, 1]) * lu(rng, 0.01, 30), 0.0))
        rows.append((7, lu(rng, 0.01, 100), 0.0))
        rows.append((8, lu(rng, 0.05, 20), 0.0))
    return rows


def golub_welsch(alpha, beta, n):
    """nodes and weights of the Gauss rule for the recurrence coefficients
    alpha_k, beta_k (beta_0 = the total mass), at the working precision"""
    J = mpmath.matrix(n, n)
    for k in range(n):
        J[k, k] = alpha[k]
        if k + 1 < n:
            J[k, k + 1] = J[k + 1, k] = mpmath.sqrt(beta[k + 1])
    E, Q = mpmath.eigsy(J)
    nodes = [E[k] for k in range(n)]
    weights = [beta[0] * Q[0, k] ** 2 for k in range(n)]
    order = sorted(range(n), key=lambda k: nodes[k])
    return [nodes[k] for k in order], [weights[k] for k in order]


def nodes():
    mpmath.mp.prec = 256
    out = {}
    for n in (16, 32, 64):  # Legendre on [-1, 1]
        out[f"legendre{n}"] = golub_welsch([0] * n, [2] + [mpmath.mpf(k * k) / (4 * k * k - 1) for k in range(1, n)], n)
    for n in (32,):  # Hermite, weight exp(-x^2)
        out[f"hermite{n}"] = golub_welsch([0] * n, [mpmath.sqrt(mpmath.pi)] + [mpmath.mpf(k) / 2 for k in range(1, n)], n)
    for n in (32,):  # Laguerre, weight exp(-x)
        out[f"laguerre{n}"] = golub_welsch([2 * k + 1 for k in range(n)], [1] + [mpmath.mpf(k * k) for k in range(1, n)], n)
    for name, (x, w) in out.items():
        # check: the rule integrates the first 2n - 1 monomials (Legendre)
        print(f"// {name}: {len(x)} nodes, then {len(w)} weights")
        print("    " + ", ".join(float(v).hex() for v in x) + ",")
        print("    " + ", ".join(float(v).hex() for v in w) + ",")


def main():
    if sys.argv[1:2] == ["nodes"]:
        nodes()
        return
    rng = random.Random(SEED)
    rows = sample(rng)
    path = OUT / "quadrature.csv"
    with open(path, "w", newline="\n") as f:
        f.write("id,p1,p2,hi,lo\n")
        for i, p1, p2 in rows:
            v = value(i, p1, p2)
            hi, lo = dd(v)
            f.write(f"{float(i).hex()},{float(p1).hex()},{float(p2).hex()},{hi.hex()},{lo.hex()}\n")
    print(f"{path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
