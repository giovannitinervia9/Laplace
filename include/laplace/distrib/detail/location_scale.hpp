// Derivatives in (mu, sigma) of a function of z = (y - mu)/sigma alone, for the
// location-scale distributions (laplace::distrib, internal).
//
// If Lambda is a function of z with derivatives L_j = Lambda^(j)(z), then
//   d/dmu    [sigma^-n Q(z)] = sigma^-(n+1) (-Q'(z)),
//   d/dsigma [sigma^-n Q(z)] = sigma^-(n+1) (-(n Q(z) + z Q'(z))),
// so d^a/dmu^a d^b/dsigma^b Lambda = sigma^-(a+b) sum_(j,m) c_jm z^m L_j, with
// the integer coefficients c_jm built by applying the two rules (sigma first,
// b times, then mu, a times; they commute). Used for log F and log(1 - F),
// whose derivatives in z are those of the standardized distribution.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>

namespace laplace::distrib::detail_location_scale {

// Q = sum c[j][m] z^m L_j, j, m = 0..6
struct z_poly {
  double c[8][8] = {};
};

// Q -> Q' (in z)
LAPLACE_HD_FORCEINLINE z_poly dz(const z_poly& q) {
  z_poly r;
  for (int j = 0; j < 7; ++j)
    for (int m = 0; m < 7; ++m) {
      const double v = q.c[j][m];
      if (v == 0.0) continue;
      if (m > 0) r.c[j][m - 1] += m * v;
      r.c[j + 1][m] += v;
    }
  return r;
}

// the mixed derivatives of order K, packed (b derivatives in sigma at position b);
// L[j] = Lambda^(j)(z), j = 1..K
template <int K>
LAPLACE_HD sym_tensor<2, K> mixed_dk(const double* L, double z, double sigma) {
  sym_tensor<2, K> t;
  double isk = 1.0;
  for (int i = 0; i < K; ++i) isk /= sigma;
  double zp[8];
  zp[0] = 1.0;
  for (int m = 1; m < 8; ++m) zp[m] = zp[m - 1] * z;
  for (int b = 0; b <= K; ++b) {
    z_poly q;
    q.c[0][0] = 1.0;
    int n = 0;
    for (int i = 0; i < b; ++i, ++n) {  // d/dsigma: -(n Q + z Q')
      const z_poly d = dz(q);
      z_poly r;
      for (int j = 0; j < 8; ++j)
        for (int m = 0; m < 8; ++m) {
          double v = -n * q.c[j][m];
          if (m > 0) v -= d.c[j][m - 1];
          r.c[j][m] = v;
        }
      q = r;
    }
    for (int i = 0; i < K - b; ++i) {  // d/dmu: -Q'
      const z_poly d = dz(q);
      for (int j = 0; j < 8; ++j)
        for (int m = 0; m < 8; ++m) q.c[j][m] = -d.c[j][m];
    }
    double s = 0.0;
    for (int j = 1; j <= K; ++j)
      for (int m = 0; m <= K; ++m)
        if (q.c[j][m] != 0.0) s += q.c[j][m] * zp[m] * L[j];
    t[b] = s * isk;
  }
  return t;
}

}  // namespace laplace::distrib::detail_location_scale
