"""Korobov generators for the rank-1 lattice rules of include/laplace/math/mvnormal.hpp.

Run with python -I. For each prime N and each dimension s = 1 .. S_MAX the
generator a (points k (1, a, a^2, ..., a^(s-1)) / N mod 1) minimizes the
worst-case error of the weighted Korobov space of smoothness alpha = 2,

    P_2(a) = -1 + (1/N) sum_k prod_j (1 + gamma_j 2 pi^2 B_2({k a^j / N})),  B_2(x) = x^2 - x + 1/6,

with product weights gamma_j = 1/j^2 in front of B_2, so that the leading
coordinates, which the Gibson-Glasbey-Elston order makes the important ones,
count most (unweighted, the criterion grows like 3.3^s and from s = 13 picks
degenerate generators such as 2), over every a in [1, (N - 1)/2] (the
criterion is symmetric in a -> N - a). Prints the C++ table.
"""
import numpy as np

S_MAX = 18
PRIMES = [1021, 4093, 16381, 65521]


def best_generators(N):
    k = np.arange(N, dtype=np.int64)
    best = [None] * (S_MAX + 1)
    best_val = [np.inf] * (S_MAX + 1)
    for a in range(1, (N - 1) // 2 + 1):
        prod = np.ones(N)
        z = 1
        for s in range(1, S_MAX + 1):
            x = (k * z % N) / N
            prod *= 1.0 + 2.0 * np.pi ** 2 / s ** 2 * (x * x - x + 1.0 / 6.0)
            v = prod.mean() - 1.0
            if v < best_val[s]:
                best_val[s], best[s] = v, a
            z = z * a % N
    return best[1:], best_val[1:]


if __name__ == "__main__":
    for N in PRIMES:
        g, v = best_generators(N)
        print(f"// N = {N}: P_2 = " + ", ".join(f"{x:.2e}" for x in v))
        print(f"{{{', '.join(str(x) for x in g)}}},")
