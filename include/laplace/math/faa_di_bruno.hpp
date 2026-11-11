// Faa di Bruno's formula and the partial Bell polynomials, to order 7
// (laplace::math):
//
//   bell_partial(n, k, g)   B_(n,k)(g_1, ..., g_(n-k+1)), 1 <= k <= n <= 7
//   compose_d1 ... _d7(f, g)
//                           d^N/dx^N f(g(x)) = sum_(k=1..N) f^(k)(g(x)) B_(N,k)(g'(x), ..., g^(N-k+1)(x))
//
// The arrays are indexed by the derivative order: f[k] = f^(k) at g(x) and
// g[j] = g^(j) at x, k, j = 1..N (element 0, the value, is not read). The
// distributions use this to carry their derivatives in a parameter theta to
// the linear predictor eta of a link, theta = h(eta) (f the log-density in
// theta, g = h); mixed derivatives in several parameters, each with its own
// link, are sums of products of the B_(n,k) of each parameter.
//
// The polynomials are written out (from sympy's bell(n, k, ...)), each a sum
// of monomials with positive integer coefficients; compose_dN sums
// f[k] B_(N,k) from k = N down to 1.
#pragma once

#include <laplace/core/config.hpp>

namespace laplace::math {

// B_(n,k)(g_1, ..., g_(n-k+1)); g[j] = g_j (g[0] is not read); 0 outside 1 <= k <= n <= 7
LAPLACE_HD double bell_partial(int n, int k, const double* g) {
  const double g1 = g[1];
  switch (n * 8 + k) {
    case 1 * 8 + 1: return g1;
    case 2 * 8 + 1: return g[2];
    case 2 * 8 + 2: return g1 * g1;
    case 3 * 8 + 1: return g[3];
    case 3 * 8 + 2: return 3.0 * g1 * g[2];
    case 3 * 8 + 3: return g1 * g1 * g1;
    case 4 * 8 + 1: return g[4];
    case 4 * 8 + 2: return 4.0 * g1 * g[3] + 3.0 * g[2] * g[2];
    case 4 * 8 + 3: return 6.0 * g1 * g1 * g[2];
    case 4 * 8 + 4: return (g1 * g1) * (g1 * g1);
    case 5 * 8 + 1: return g[5];
    case 5 * 8 + 2: return 5.0 * g1 * g[4] + 10.0 * g[2] * g[3];
    case 5 * 8 + 3: return 10.0 * g1 * g1 * g[3] + 15.0 * g1 * g[2] * g[2];
    case 5 * 8 + 4: return 10.0 * (g1 * g1 * g1) * g[2];
    case 5 * 8 + 5: return (g1 * g1) * (g1 * g1) * g1;
    case 6 * 8 + 1: return g[6];
    case 6 * 8 + 2: return 6.0 * g1 * g[5] + 15.0 * g[2] * g[4] + 10.0 * g[3] * g[3];
    case 6 * 8 + 3: return 15.0 * g1 * g1 * g[4] + 60.0 * g1 * g[2] * g[3] + 15.0 * g[2] * g[2] * g[2];
    case 6 * 8 + 4: return 20.0 * (g1 * g1 * g1) * g[3] + 45.0 * (g1 * g1) * (g[2] * g[2]);
    case 6 * 8 + 5: return 15.0 * ((g1 * g1) * (g1 * g1)) * g[2];
    case 6 * 8 + 6: return (g1 * g1 * g1) * (g1 * g1 * g1);
    case 7 * 8 + 1: return g[7];
    case 7 * 8 + 2: return 7.0 * g1 * g[6] + 21.0 * g[2] * g[5] + 35.0 * g[3] * g[4];
    case 7 * 8 + 3:
      return 21.0 * g1 * g1 * g[5] + 105.0 * g1 * g[2] * g[4] + 70.0 * g1 * g[3] * g[3] + 105.0 * g[2] * g[2] * g[3];
    case 7 * 8 + 4:
      return 35.0 * (g1 * g1 * g1) * g[4] + 210.0 * (g1 * g1) * g[2] * g[3] + 105.0 * g1 * (g[2] * g[2] * g[2]);
    case 7 * 8 + 5: return 35.0 * ((g1 * g1) * (g1 * g1)) * g[3] + 105.0 * (g1 * g1 * g1) * (g[2] * g[2]);
    case 7 * 8 + 6: return 21.0 * ((g1 * g1) * (g1 * g1) * g1) * g[2];
    case 7 * 8 + 7: return (g1 * g1 * g1) * (g1 * g1 * g1) * g1;
    default: return 0.0;
  }
}

namespace detail_faa {
template <int N>
LAPLACE_HD_FORCEINLINE double compose(const double* f, const double* g) {
  double s = 0.0;
  for (int k = N; k >= 1; --k) s += f[k] * bell_partial(N, k, g);
  return s;
}
}  // namespace detail_faa

LAPLACE_HD double compose_d1(const double* f, const double* g) { return f[1] * g[1]; }
LAPLACE_HD double compose_d2(const double* f, const double* g) { return detail_faa::compose<2>(f, g); }
LAPLACE_HD double compose_d3(const double* f, const double* g) { return detail_faa::compose<3>(f, g); }
LAPLACE_HD double compose_d4(const double* f, const double* g) { return detail_faa::compose<4>(f, g); }
LAPLACE_HD double compose_d5(const double* f, const double* g) { return detail_faa::compose<5>(f, g); }
LAPLACE_HD double compose_d6(const double* f, const double* g) { return detail_faa::compose<6>(f, g); }
LAPLACE_HD double compose_d7(const double* f, const double* g) { return detail_faa::compose<7>(f, g); }

}  // namespace laplace::math
