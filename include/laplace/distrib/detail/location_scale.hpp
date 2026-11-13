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
// mixed_power_dk does the same when the second parameter carries the scale as a
// power, sigma = w^k (the variance, the precision, the rate).
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

// The same for a second parameter w that carries the scale as a power,
// sigma = w^k: the variance w = sigma^2 (k = 1/2), the precision w = tau
// (k = -1/2), the rate w = lambda (k = -1). Since dsigma/dw = k sigma / w,
//   d/dw [sigma^-n Q(z)] = (k/w) sigma^-n (-(n Q(z) + z Q'(z))),
// so a derivative in w leaves the power of sigma where it is, while a
// derivative in mu gives sigma^-(n+1) (-Q'(z)) and raises it by one. The two
// operators commute, so the a derivatives in mu are applied first and n is then
// the constant a through the b derivatives in w. The factor k/w is itself a
// function of w, so it is differentiated too: writing the state after i
// derivatives in w as sigma^-a w^-i R_i(z),
//   d/dw [sigma^-a w^-i R_i] = sigma^-a w^-(i+1) (-((k a + i) R_i + k z R_i')),
// which gives
//   d^a/dmu^a d^b/dw^b Lambda = sigma^-a w^-b R_b,   R_0 = (-d/dz)^a Lambda,
//   R_(i+1) = -((k a + i) R_i + k z R_i').
// With k = 1 and w = sigma this is mixed_dk, which is kept as it stands because
// its bounds for gaussian1 and laplace1 were measured on that code.
template <int K>
LAPLACE_HD sym_tensor<2, K> mixed_power_dk(const double* L, double z, double sigma, double w,
                                           double k) {
  sym_tensor<2, K> t;
  double zp[8];
  zp[0] = 1.0;
  for (int m = 1; m < 8; ++m) zp[m] = zp[m - 1] * z;
  for (int b = 0; b <= K; ++b) {
    const int a = K - b;
    z_poly q;
    q.c[0][0] = 1.0;
    for (int i = 0; i < a; ++i) {  // d/dmu: -Q'
      const z_poly d = dz(q);
      for (int j = 0; j < 8; ++j)
        for (int m = 0; m < 8; ++m) q.c[j][m] = -d.c[j][m];
    }
    for (int i = 0; i < b; ++i) {  // d/dw: -((k a + i) R + k z R')
      const z_poly d = dz(q);
      z_poly r;
      for (int j = 0; j < 8; ++j)
        for (int m = 0; m < 8; ++m) {
          double v = -(k * a + i) * q.c[j][m];
          if (m > 0) v -= k * d.c[j][m - 1];
          r.c[j][m] = v;
        }
      q = r;
    }
    double s = 0.0;
    for (int j = 1; j <= K; ++j)
      for (int m = 0; m <= K; ++m)
        if (q.c[j][m] != 0.0) s += q.c[j][m] * zp[m] * L[j];
    double f = 1.0;
    for (int i = 0; i < a; ++i) f /= sigma;
    for (int i = 0; i < b; ++i) f /= w;
    t[b] = s * f;
  }
  return t;
}

}  // namespace laplace::distrib::detail_location_scale
