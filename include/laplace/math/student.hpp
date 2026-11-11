// The distribution function of the standard Student t with real degrees of
// freedom nu > 0, on the log scale, and its mixed partial derivatives in
// (z, nu) of total order 1 to 5, Laplace's own.
//
//   own::log_student_cdf(z, nu)          log T_nu(z), by own::beta_inc (below)
//   own::log_student_cdf_d1 .. _d5        one function per order K, returning
//                                         shape_deriv<K> with c[j] = d^K / dz^(K-j) dnu^j
//
// The survival function needs no function of its own: 1 - T_nu(z) = T_nu(-z).
//
// Notation. t_nu(s) = exp(c(nu) + k(s, nu)) is the density, with
//     c(nu) = log Gamma((nu + 1)/2) - log Gamma(nu/2) - log(nu pi)/2,
//     k(s, nu) = -(nu + 1)/2 log(1 + s^2/nu);
// a = |z|, A = nu + z^2, x = nu / A. Every function of (z, nu) is carried as a
// truncated bivariate Taylor polynomial in (dz, dnu): an array of 21 doubles,
// the coefficient of dz^i dnu^j (divided by i! j!) at index
// ix(i, j) = (i + j)(i + j + 1)/2 + j (detail_ibd::ix). The derivatives of
// order K are the coefficients of total order K times i! j!.
//
// Why not the density over the distribution function. d log T / dz = t / T,
// and the higher z-derivatives follow from those of log t; but
// d^2 log T / dz^2 = (t/T) (d log t/dz - t/T) cancels: by a factor z^2 in the
// Gaussian tail and nu in the power tail (z^2 >> nu). The same happens with the
// beta route below in its x-derivatives. The candidate chosen here therefore
// writes every derivative as an integral of derivatives of the density, whose
// parts do not cancel; the beta composition is kept as the second candidate.
//
// Candidate 1, quadrature (chosen; derivs below). Three regions:
//   centre, T(-a) > t_switch = 1/5: T = 1/2 + G, G = int_0^z t(s) ds. The
//     z-derivatives of G are those of t at z (closed form); the nu-derivatives
//     are int_0^z d^j t / dnu^j ds, by Gauss-Legendre in s = sqrt(nu) sinh(tau),
//     where t ds = e^c sqrt(nu) cosh(tau)^(-nu) dtau, on pieces [0, 1], [1, 3],
//     [3, 9], ... (cosh vanishes only at tau = i pi/2, so pieces away from 0 may
//     grow geometrically). Then log T by the logarithm of the polynomial.
//   Gaussian tail, z <= 0 with z^2 < 4 nu (shift_ratio): by the shift s = z - u, u >= 0,
//         log T = c(nu) + k(z, nu) + log int_0^inf exp(Dk(u)) du,
//     Dk(u) = k(z - u + dz, nu + dnu) - k(z + dz, nu + dnu) as a polynomial in
//     (dz, dnu); the derivatives of the last term are the cumulants of the
//     polynomial-valued variable Dk under the weight exp(Dk_00), from the sum
//     of exp(Dk) over the nodes (one exponential of a polynomial per node).
//     Every coefficient of Dk is computed as an exact difference: with
//     A_u = nu + (z - u)^2 and delta = A_u - A = u (u + 2a), the pure
//     nu-coefficients come from log(A_u + d) - log(A + d) (pure_nu below), and
//     those with i >= 1 from d k / ds = -s - R, R = s (1 - s^2)/(nu + s^2):
//     the pure z-coefficients from -(nu + 1) s / (nu + s^2), the mixed ones
//     from -R (whose nu-dependence carries the factor 1 - s^2 explicitly; the
//     other split loses a factor nu as nu grows). Differences of the
//     polynomials are formed from those of their pieces (Dq = -q Dd q_u for a
//     reciprocal), so that nothing of size |z| cancels to leave O(1/|z|).
//     Nodes in zeta = log(A_u / A), in which exp(Dk_00) = exp(-(nu + 1) zeta/2)
//     exactly, by Takahasi and Mori's map zeta = sigma exp(alpha (t - e^-t))
//     for exponentially decaying integrands (exp-sinh converged like e^(-2.7/h)).
//   power tail, z <= 0 with z^2 >= 4 nu: by the scaling s = z e^v, v >= 0,
//         log T = c(nu) + h(z, nu) + log int_0^inf exp(Dk(v) + v) dv,
//     h = log|z| + k(z, nu) (whose z-derivative nu (1 - z^2)/(z A) is written
//     without the cancellation of 1/z against -(nu + 1) z / A), and
//         Dk(v) = -(nu + 1 + dnu) v - (nu + 1 + dnu)/2 log(1 - x m_v),
//     m_v = 1 - e^(-2v), x the polynomial (nu + dnu)/(nu + dnu + (z + dz)^2);
//     everything bounded (the shift would need e^zeta beyond the double range
//     for small nu). Nodes in v by the same map. For nu < 1 the integral is
//     taken at fixed w = nu v instead, with the pole of c(nu) at nu = 0 removed
//     analytically (small_nu_anchor below).
//   upper side z > 0 outside the centre: log T(z) = log(1 - S), S = T(-z)
//     from the lower tail (its polynomial with dz -> -dz), composed as
//     log(1 - S0) + log(1 - r (exp(L - L0) - 1)), r = S0/(1 - S0).
// The region is chosen with own::beta_inc (T(-a) against t_switch).
//
// Candidate 2, beta composition (derivs_beta below). For z < 0,
// T = I_x(nu/2, 1/2)/2, or (x > 1/2) T = (1 - I_y(1/2, nu/2))/2, y = 1 - x.
// The pure shape derivatives come from log_ibeta_dK / log_ibetac_dK
// (incomplete_beta_deriv.hpp, one call per order); those in the variable xi of
// the frame (x or y) from d log I / d xi = +-exp(log density - log I), order by
// order; the result is composed with a = nu/2 and xi(z, nu). Its x-derivatives
// have the cancellation described above.
//
// c(nu): with a2 = nu/2, f(a2) = log Gamma(a2 + 1/2) - log Gamma(a2) - log(a2)/2,
// c = f(a2) - log(2 pi)/2; f^(j)(a2) = (-1)^(j-1) int_0^inf t^(j-1) e^(-a2 t) tanh(t/4)/2 dt,
// so that for large a2 the asymptotic series from the Taylor series of tanh,
// f^(j) = (-1)^(j-1) sum_k c_k (j + 2k - 2)! a2^-(j + 2k - 1), c_k = 2 (1 - 4^-k) B_2k / (2k)!,
// and below 15 the recurrence f(b) = f(b + 1) + log1p(-1/(2b + 1)^2)/2, whose
// derivatives are second differences of b^-j, written as polynomials in
// 1/(2b + 1)^2 without cancellation (fder below).
//
// The value. log_student_cdf uses own::beta_inc (log I_x(nu/2, 1/2) - log 2 on
// the lower side, log1p(-T(-z)) on the upper side, with x or 1 - x whichever is
// below 1/2): as accurate as the quadrature against the references and in the
// deep tails, and 4.5 times faster on one CPU thread.
//
// Derived from the formulas above; no code copied (R's pt.c and pbeta.c are GPL).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_beta_deriv.hpp>

#if defined(__CUDACC__)
#define LAPLACE_ST_NOINLINE __noinline__
#else
#define LAPLACE_ST_NOINLINE
#endif

namespace laplace::math::own {

namespace detail_st {

namespace ibd = detail_ibd;
namespace ibx = detail_ibx;
using ibd::ix;
constexpr int NC = ibd::NC;
constexpr double halfpi = 1.57079632679489661923;
constexpr double half_log_2pi = 0.91893853320467274178;
constexpr double ln2 = 0.69314718055994530942;

// ---------------------------------------------------------------- polynomials

// number of coefficients of total order <= K
template <int K>
LAPLACE_HD_FORCEINLINE constexpr int nk() {
  return (K + 1) * (K + 2) / 2;
}

LAPLACE_HD_FORCEINLINE void pzero(double* p) {
  for (int c = 0; c < NC; ++c) p[c] = 0.0;
}

// r = p q, total order <= K (r may not alias p or q)
template <int K>
LAPLACE_HD void pmul(const double* p, const double* q, double* r) {
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      for (int k = 0; k <= i; ++k)
        for (int l = 0; l <= j; ++l) v += p[ix(k, l)] * q[ix(i - k, j - l)];
      r[ix(i, j)] = v;
    }
}

// 1/n, n = 1..6
LAPLACE_HD_FORCEINLINE constexpr double rec(int n) {
  return n == 1 ? 1.0 : n == 2 ? 0.5 : n == 3 ? 1.0 / 3.0 : n == 4 ? 0.25 : n == 5 ? 0.2 : 1.0 / 6.0;
}

// q = 1 / d for d = d00 + d10 ds + d20 ds^2 + d01 dnu (the denominators met here)
template <int K>
LAPLACE_HD void pinv_sparse(double d00, double d10, double d20, double d01, double* q) {
  const double q0 = 1.0 / d00;
  q[0] = q0;
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      if (i >= 1) v += d10 * q[ix(i - 1, j)];
      if (i >= 2) v += d20 * q[ix(i - 2, j)];
      if (j >= 1) v += d01 * q[ix(i, j - 1)];
      q[ix(i, j)] = -v * q0;
    }
}

// q = 1 / d (d_00 != 0)
template <int K>
LAPLACE_HD void pinv(const double* d, double* q) {
  const double q0 = 1.0 / d[0];
  q[0] = q0;
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      for (int k = 0; k <= i; ++k)
        for (int l = 0; l <= j; ++l)
          if (k + l > 0) v += d[ix(k, l)] * q[ix(i - k, j - l)];
      q[ix(i, j)] = -v * q0;
    }
}

// p = exp(q - q_00): p_00 = 1, then i p_ij = sum_{k>=1} k q_kl p_(i-k)(j-l)
// (i >= 1) and j p_0j = sum_{l>=1} l q_0l p_0(j-l)
template <int K>
LAPLACE_HD void pexp(const double* q, double* p) {
  p[0] = 1.0;
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      if (i >= 1) {
        for (int k = 1; k <= i; ++k)
          for (int l = 0; l <= j; ++l) v += k * q[ix(k, l)] * p[ix(i - k, j - l)];
        v *= rec(i);
      } else {
        for (int l = 1; l <= j; ++l) v += l * q[ix(0, l)] * p[ix(0, j - l)];
        v *= rec(j);
      }
      p[ix(i, j)] = v;
    }
}

// q = log(p / p_00), q_00 = 0 (the same recursion solved for q)
template <int K>
LAPLACE_HD void plog(const double* p, double* q) {
  const double ip0 = 1.0 / p[0];
  q[0] = 0.0;
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double v = 0.0;
      if (i >= 1) {
        for (int k = 1; k <= i; ++k)
          for (int l = 0; l <= j; ++l)
            if (k != i || l != j) v += k * q[ix(k, l)] * p[ix(i - k, j - l)];
        q[ix(i, j)] = (p[ix(i, j)] - v * rec(i)) * ip0;
      } else {
        for (int l = 1; l < j; ++l) v += l * q[ix(0, l)] * p[ix(0, j - l)];
        q[ix(0, j)] = (p[ix(0, j)] - v * rec(j)) * ip0;
      }
    }
}

// univariate: p = exp(q - q_0), p_0 = 1
LAPLACE_HD_FORCEINLINE void uexp(int K, const double* q, double* p) {
  p[0] = 1.0;
  for (int j = 1; j <= K; ++j) {
    double v = 0.0;
    for (int l = 1; l <= j; ++l) v += l * q[l] * p[j - l];
    p[j] = v * rec(j);
  }
}

// ---------------------------------------------------------------- pure nu-coefficients
//
// The normalized Taylor coefficients C_j, j = 1..K, in d of
//     -(m + d)/2 [log(Q + d) - log(P + d)],   0 < P <= Q,
// given e = (Q - P)/Q, rho = P/Q = 1 - e, G1 = -log(1 - e) - e and mP = m - P.
// With F_j = ((-1)^j / j) P^-j (1 - rho^j) (1 - rho^j = e sum_{i<j} rho^i) the
// coefficients of the logarithm difference, C_j = -(m F_j + F_(j-1))/2, and
//     m F_j + F_(j-1) = (P F_j + F_(j-1)) + mP F_j,
//     P F_j + F_(j-1) = G1 (j = 1), (-1)^j P^(1-j) G_j (j >= 2),
//     G_j = (1 - rho^j)/j - (1 - rho^(j-1))/(j-1) = -e^2 int_0^1 (1 - e v)^(j-2) v dv,
// the integral exact by 3-point Gauss-Legendre (degree <= 4). For the density
// (P = nu, m = nu + 1, mP = 1) the two parts are each of order nu^-(j+1) as
// nu grows; formed from F_j and F_(j-1) directly they would cancel to that from
// nu^-j.
LAPLACE_HD void pure_nu(int K, double P, double e, double rho, double G1, double mP, double* C) {
  const double iP = 1.0 / P;
  // 3-point Gauss-Legendre on [0, 1]
  constexpr double gv0 = 0.11270166537925831148, gv2 = 0.88729833462074168852;
  constexpr double gw0 = 0.27777777777777777778, gw1 = 0.44444444444444444444;
  const double b0 = 1.0 - e * gv0, b1 = 1.0 - e * 0.5, b2 = 1.0 - e * gv2;
  double p0 = 1.0, p1 = 1.0, p2 = 1.0;  // (1 - e v)^(j-2)
  double Pj = 1.0, Pm = 1.0;            // P^-j, P^(1-j)
  double rj = 1.0, sr = 0.0;            // rho^j, sum_{i<j} rho^i
  const double e2 = e * e;
  for (int j = 1; j <= K; ++j) {
    sr += rj;
    rj *= rho;
    Pm = Pj;
    Pj *= iP;
    const double sgn = (j % 2 == 0) ? 1.0 : -1.0;
    const double F = sgn * e * sr * Pj * rec(j);
    double H;
    if (j == 1) {
      H = G1;
    } else {
      const double Gj = -e2 * (gw0 * (p0 * gv0 + p2 * gv2) + gw1 * p1 * 0.5);
      H = sgn * Pm * Gj;
      p0 *= b0;
      p1 *= b1;
      p2 *= b2;
    }
    C[j] = -0.5 * (H + mP * F);
  }
}

// G1(e) = -log(1 - e) - e, with F0 = -log(1 - e) given (used for e > 1/2)
LAPLACE_HD_FORCEINLINE double g1_of(double e, double F0) { return e <= 0.5 ? -ibx::log1pmx(-e) : F0 - e; }

// ---------------------------------------------------------------- c(nu)
//
// f[j] = f^(j)(b), j = 0..K, f(b) = log Gamma(b + 1/2) - log Gamma(b) - log(b)/2
LAPLACE_HD void fder(double b, int K, double* f) {
  for (int j = 0; j <= K; ++j) f[j] = 0.0;
  const int M = b < 15.0 ? static_cast<int>(15.0 - b) + 1 : 0;
  // recurrence terms g^(j)(b + k), k < M
  // recurrence terms g^(j)(b + k), k < M: g = log1p(-eps^2)/2 (the atanh series
  // of detail_ibx::log1p_poly where eps^2 <= 1/2); for j >= 1, with p = w + 1/2 and eps = 1/(2p),
  //   w^-j + (w + 1)^-j - 2 p^-j = p^-j [(1 - eps)^-j + (1 + eps)^-j - 2]
  //     = 2 p^-j sum_{l=1..j} (C(j, 2l) - (-1)^l C(j, l)) eps^(2l) / (1 - eps^2)^j,
  // whose leading term j (j + 1)/2 eps^2 dominates (eps <= 1/3 for w >= 1)
  for (int k = 0; k < M; ++k) {
    const double w = b + k, eps = 1.0 / (2.0 * w + 1.0), eps2 = eps * eps, q = 1.0 - eps2;
    f[0] += 0.5 * (eps2 <= 0.5 ? ibx::log1p_poly(-eps2) : Kokkos::log1p(-eps2));
    if (K == 0) continue;
    const double ip = 1.0 / (w + 0.5), iq = 1.0 / q;
    double ipj = 1.0, iqj = 1.0, fct = 1.0;  // p^-j, (1 - eps^2)^-j, (j-1)!
    for (int j = 1; j <= K; ++j) {
      ipj *= ip;
      iqj *= iq;
      if (j > 1) fct *= (j - 1);
      double br;  // w^-j + (w + 1)^-j - 2 p^-j
      if (w >= 0.5) {
        double num = 0.0, e2l = 1.0;
        for (int l = 1; l <= j; ++l) {
          e2l *= eps2;
          num += ((2 * l <= j ? ibd::binom(j, 2 * l) : 0.0) - ((l % 2) ? -1.0 : 1.0) * ibd::binom(j, l)) * e2l;
        }
        br = 2.0 * ipj * num * iqj;
      } else {
        br = Kokkos::pow(w, -double(j)) + Kokkos::pow(w + 1.0, -double(j)) - 2.0 * ipj;
      }
      f[j] += ((j % 2 == 1) ? 0.5 : -0.5) * fct * br;
    }
  }
  // asymptotic series at Z = b + M: c_k = 2 (1 - 4^-k) B_2k / (2k)!
  constexpr double ck[10] = {0.125, -0.0026041666666666665, 6.510416666666667e-05, -1.6469804067460318e-06,
                             4.171273905974427e-08, -1.0565800106408831e-09, 2.6763439450953015e-11,
                             -6.779257147811904e-13, 1.717205860610079e-14, -4.349733248883285e-16};
  const double Z = b + M, iZ = 1.0 / Z, iZ2 = iZ * iZ;
  double zj = iZ;  // Z^-(j + 1)
  for (int j = 0; j <= K; ++j) {
    // sum_k c_k (j + 2k - 2)! Z^-(j + 2k - 1), from k = 1: j! Z^-(j + 1)
    double s = 0.0, zp = zj, fct = ibd::factorial(j);
    for (int k = 1; k <= 10; ++k) {
      s += ck[k - 1] * fct * zp;
      zp *= iZ2;
      fct *= (j + 2.0 * k - 1.0) * (j + 2.0 * k);
    }
    f[j] += (j == 0) ? -s : ((j % 2 == 1) ? s : -s);
    zj *= iZ;
  }
}

// normalized Taylor coefficients of c(nu + d), j = 0..K
LAPLACE_HD void cpoly(double nu, int K, double* c) {
  double f[6];
  fder(0.5 * nu, K, f);
  c[0] = f[0] - half_log_2pi;
  double s = 1.0;
  for (int j = 1; j <= K; ++j) {
    s *= 0.5 / j;
    c[j] = f[j] * s;
  }
}

// ---------------------------------------------------------------- the anchor k(z + dz, nu + dnu)
//
// KA: k at (z, nu), all coefficients to order K. Pure nu: pure_nu with P = nu,
// Q = A. With i >= 1 from k_s = -(nu + 1) s Q (pure z) and -N Q (mixed), Q the
// polynomial 1 / (A + 2 z ds + ds^2 + dnu), N = s (1 - s^2) expanded in ds.
// Qa, N returned for the nodes.
template <int K>
LAPLACE_HD void anchor_k(double z, double nu, double* KA, double* Qa, double* N) {
  constexpr int K1 = K > 0 ? K - 1 : 0;
  const double z2 = z * z, A = nu + z2, m = nu + 1.0;
  pzero(KA);
  const double F0 = Kokkos::log1p(z2 / nu), e = z2 / A;
  KA[0] = -0.5 * m * F0;
  double C[6];
  pure_nu(K, nu, e, nu / A, g1_of(e, F0), 1.0, C);
  for (int j = 1; j <= K; ++j) KA[ix(0, j)] = C[j];
  N[0] = z * (1.0 - z) * (1.0 + z);
  N[1] = 1.0 - 3.0 * z2;
  N[2] = -3.0 * z;
  N[3] = -1.0;
  if (K == 0) return;
  pinv_sparse<K1>(A, 2.0 * z, 1.0, 1.0, Qa);
  for (int n = 0; n <= K1; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      double ks;
      if (j == 0) {
        ks = -m * (z * Qa[ix(i, 0)] + (i >= 1 ? Qa[ix(i - 1, 0)] : 0.0));
      } else {
        double v = 0.0;
        for (int k = 0; k <= 3 && k <= i; ++k) v += N[k] * Qa[ix(i - k, j)];
        ks = -v;
      }
      KA[ix(i + 1, j)] = ks / (i + 1);
    }
}

// ---------------------------------------------------------------- quadrature parameters

struct par {
  // tails: nodes w = sigma exp(alpha (t - e^-t)), t = tlo + k h, tlo = -log(start/alpha)
  // (Takahasi and Mori's map for exponentially decaying integrands)
  double h_g = 0.25;          // step, Gaussian tail (w = zeta)
  double alpha_g = 0.75;
  double sig_g = 1.0;         // zeta scale = sig_g * 2/(nu + 1)
  double h_p = 0.25;          // step, power tail (w = v)
  double alpha_p = 0.75;
  double sig_p = 1.0;         // v scale = sig_p / (nu + 1)
  double start = 42.0;
  double tol = 1e-18;         // stop when W (1 + node/scale)^K < tol sum W
  int gl_n = 16;              // Gauss-Legendre nodes per centre piece (8, 12, 16, 20, 24)
  double piece0 = 1.0;        // first centre piece [0, piece0], then x3
  double t_switch = 0.2;      // centre when T(-|z|) > t_switch
  double nu_w = 1.0;          // power tail at fixed w = nu v below this nu
  double shift_ratio = 4.0;   // lower tail by the shift when z^2 < shift_ratio nu, else by the scaling
  int* count = nullptr;       // nodes (host, measurement)
};

// Gauss-Legendre rules on [-1, 1], n = 8, 12, 16, 20, 24: the positive half,
// pairs (x_i, w_i), written by mpmath at 200 bits
// clang-format off
#define LAPLACE_ST_GLEG_DATA \
  0.9602898564975363, 0.10122853629037626, 0.7966664774136267, 0.22238103445337448, \
  0.525532409916329, 0.31370664587788727, 0.1834346424956498, 0.362683783378362, \
  0.9815606342467192, 0.04717533638651183, 0.9041172563704749, 0.10693932599531843, \
  0.7699026741943047, 0.16007832854334622, 0.5873179542866175, 0.20316742672306592, \
  0.3678314989981802, 0.2334925365383548, 0.1252334085114689, 0.24914704581340277, \
  0.9894009349916499, 0.027152459411754096, 0.9445750230732326, 0.062253523938647894, \
  0.8656312023878318, 0.09515851168249279, 0.755404408355003, 0.12462897125553388, \
  0.6178762444026438, 0.14959598881657674, 0.45801677765722737, 0.16915651939500254, \
  0.2816035507792589, 0.18260341504492358, 0.09501250983763744, 0.1894506104550685, \
  0.9931285991850949, 0.017614007139152118, 0.9639719272779138, 0.04060142980038694, \
  0.912234428251326, 0.06267204833410907, 0.8391169718222188, 0.08327674157670475, \
  0.7463319064601508, 0.10193011981724044, 0.636053680726515, 0.11819453196151841, \
  0.5108670019508271, 0.13168863844917664, 0.37370608871541955, 0.14209610931838204, \
  0.22778585114164507, 0.14917298647260374, 0.07652652113349734, 0.15275338713072584, \
  0.9951872199970213, 0.0123412297999872, 0.9747285559713095, 0.028531388628933663, \
  0.9382745520027328, 0.04427743881741981, 0.8864155270044011, 0.05929858491543678, \
  0.820001985973903, 0.0733464814110803, 0.7401241915785544, 0.08619016153195327, \
  0.6480936519369755, 0.09761865210411388, 0.5454214713888396, 0.10744427011596563, \
  0.4337935076260451, 0.1155056680537256, 0.3150426796961634, 0.12167047292780339, \
  0.1911188674736163, 0.1258374563468283, 0.06405689286260563, 0.12793819534675216
// clang-format on
#if defined(__CUDACC__)
static __constant__ double gleg_dev[80] = {LAPLACE_ST_GLEG_DATA};
#endif
inline constexpr double gleg_host[80] = {LAPLACE_ST_GLEG_DATA};
LAPLACE_HD_FORCEINLINE double gleg(int i) {
  KOKKOS_IF_ON_DEVICE((return gleg_dev[i];))
  KOKKOS_IF_ON_HOST((return gleg_host[i];))
}
LAPLACE_HD_FORCEINLINE int gleg_offset(int n) { return n == 8 ? 0 : n == 12 ? 4 : n == 16 ? 10 : n == 20 ? 18 : 28; }

// The default nodes of the tails, t = tlo + k/4, alpha = 3/4, start 42, k = 0..80:
// pairs (E_k, D_k), E_k = exp(alpha (t - e^-t)), D_k = h alpha (1 + e^-t) E_k
// (node = sigma E_k, weight = sigma D_k), written by mpmath at 200 bits
// clang-format off
#define LAPLACE_ST_TM_DATA \
  2.80860548598085e-20, 3.0016971131420337e-19, 3.6700850304365626e-16, 3.069992444745453e-15, \
  6.142947850812895e-13, 4.027360795363283e-12, 2.0749601460316544e-10, 1.0680543625568569e-09, \
  2.0152241838432272e-08, 8.16212977363854e-08, 7.414076439107956e-07, 2.369390820530448e-06, \
  1.2807488817106019e-05, 3.240764297723833e-05, 0.0001227919137979723, 0.0002470728522243779, \
  0.0007442782032028779, 0.0011971867281921754, 0.0031565323874562794, 0.004085158515042225, \
  0.010136791913112175, 0.010637473261213427, 0.02621341598615498, 0.02251057549602841, \
  0.057264714971662675, 0.040673077990471075, 0.10969730302763715, 0.06522921759249836, \
  0.18970142849426436, 0.09571812894870435, 0.30292831246082036, 0.13160306476935837, \
  0.4546303529211899, 0.17267506757832718, 0.6501113700509317, 0.21926595870226753, \
  0.8953042962725303, 0.272301844994732, 1.1973550910481712, 0.3332751681968942, \
  1.5651668911000427, 0.40420191342929035, 2.009911033564162, 0.48760229265140925, \
  2.5455375137500234, 0.5865200988988198, 3.189325455546419, 0.7045832229591458, \
  3.9625138740256, 0.8461032952280718, 4.891050752555578, 1.0162124822397618, \
  6.006497415189098, 1.2210375034263505, 7.347126525519671, 1.4679135310960434, \
  8.959256014070071, 1.7656432344703097, 10.89886774917651, 2.124808718770491, \
  13.233568748188116, 2.5581465806622186, 16.04496423064281, 3.0809989041351353, \
  19.431526056906957, 3.7118559190197185, 23.51205744988706, 4.473009397043613, \
  28.429875897273313, 5.391339825778675, 34.35786147189909, 6.4992651446793746, \
  41.50454835625363, 7.835884532701745, 50.12147417789325, 9.448357614068378, \
  60.512046147516116, 11.393567745195373, 73.04223650335355, 13.740128056837483, \
  88.15348429058452, 16.57080100543326, 106.37825831821274, 19.98541676293388, \
  128.35882998266226, 24.10439335916348, 154.86991783703556, 29.072982704995873, \
  186.84600230927825, 35.066392216246115, 225.4142736457984, 42.29596262939146, \
  271.9343747861915, 51.016619839438235, 328.04634046356074, 61.535863510369694, \
  395.7284228215683, 74.22460939341673, 477.36684243387975, 89.5302676483441, \
  575.839924095477, 107.99251830253921, 694.6195839545462, 130.26234008172239, \
  837.8937463519145, 157.12496355766942, 1010.7140067040202, 189.52755792708444, \
  1219.1737469255168, 228.6126276404598, 1470.6229836253708, 275.7582964253541, \
  1773.9275244850523, 332.6268990936106, 2139.781570506201, 401.22359444918595, \
  2581.084786282966, 483.9670659508841, 3113.3971335562014, 583.7748029908465, \
  3755.487505191038, 704.1659697529154, 4529.995504063097, 849.3854887419076, \
  5464.2297007566885, 1024.5537140906986, 6591.130516135063, 1235.8469720313892, \
  7950.431679411752, 1490.715334273494, 9590.061213998757, 1798.1453028408434, \
  11567.831349005328, 2168.9766684685137, 13953.476941596917, 2616.284714786554, \
  16831.114283696486, 3155.841244568403, 20302.20698899645, 3806.6706835380187, \
  24489.14353558075, 4591.720869604359, 29539.55260604651, 5538.672179127458, \
  35631.50838128815, 6680.913519496612, 42979.80932331014, 8058.719600901817, \
  51843.55183297894, 9720.670997156656, 62535.26582509615, 11725.367066019253, \
  75431.93433539198, 14143.492125498613, 90988.28570381229, 17060.30773821628, \
  109752.8270080752, 20578.65898019385, 132387.1840762246, 24822.600693202647, \
  159689.42999479198, 29941.77158004019
// clang-format on
constexpr int tm_n = 81;
#if defined(__CUDACC__)
static __constant__ double tm_dev[2 * tm_n] = {LAPLACE_ST_TM_DATA};
#endif
inline constexpr double tm_host[2 * tm_n] = {LAPLACE_ST_TM_DATA};
LAPLACE_HD_FORCEINLINE double tm_tab(int i) {
  KOKKOS_IF_ON_DEVICE((return tm_dev[i];))
  KOKKOS_IF_ON_HOST((return tm_host[i];))
}

// node k of a tail rule (step h, map parameter alpha): E and D as above
LAPLACE_HD_FORCEINLINE void tm_node(int k, double h, double alpha, double start, double& E, double& D) {
  if (h == 0.25 && alpha == 0.75 && start == 42.0 && k < tm_n) {
    E = tm_tab(2 * k);
    D = tm_tab(2 * k + 1);
    return;
  }
  const double t = -math::log(start / alpha) + k * h, emt = math::exp(-t);
  E = math::exp(alpha * (t - emt));
  D = h * alpha * (1.0 + emt) * E;
}

// ---------------------------------------------------------------- centre
//
// log T(z) for T(-|z|) > t_switch: T = 1/2 + G (see the head of the file)
template <int K>
LAPLACE_ST_NOINLINE LAPLACE_HD void centre(double z, double nu, const par& pp, double* L) {
  const double a = Kokkos::fabs(z), sgn = z < 0.0 ? -1.0 : 1.0, m = nu + 1.0;
  double cp[6];
  cpoly(nu, K, cp);
  // H_j = int_0^a d^j exp(k(s, nu)) / dnu^j ds / j!
  double H[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double sq = Kokkos::sqrt(nu), tz = Kokkos::asinh(a / sq);
  const int off = gleg_offset(pp.gl_n), nh = pp.gl_n / 2;
  double lo = 0.0;
  while (lo < tz) {
    const double hi = Kokkos::fmin(tz, lo == 0.0 ? pp.piece0 : 3.0 * lo);
    const double mid = 0.5 * (lo + hi), hw = 0.5 * (hi - lo);
    for (int i = 0; i < 2 * nh; ++i) {
      const double xg = gleg(2 * (off + i % nh)), wg = gleg(2 * (off + i % nh) + 1);
      const double tau = mid + (i < nh ? -hw * xg : hw * xg);
      const double sh = Kokkos::sinh(tau), ch = Kokkos::sqrt(1.0 + sh * sh);
      const double q = sh * sh, F0 = Kokkos::log1p(q), e = q / (1.0 + q);
      double C[6], E[6];
      pure_nu(K, nu, e, 1.0 / (1.0 + q), g1_of(e, F0), 1.0, C);
      uexp(K, C, E);
      const double val = math::exp(-0.5 * m * F0) * wg * hw * sq * ch;
      for (int j = 0; j <= K; ++j) H[j] += val * E[j];
      if (pp.count) ++*pp.count;
    }
    lo = hi;
  }
  // G's pure nu-coefficients: sgn exp(c) * H
  double ec[6], Gp[6];
  uexp(K, cp, ec);
  const double e0 = math::exp(cp[0]);
  for (int j = 0; j <= K; ++j) {
    double v = 0.0;
    for (int l = 0; l <= j; ++l) v += ec[l] * H[j - l];
    Gp[j] = sgn * e0 * v;
  }
  // t's polynomial at z
  double KA[NC], Qa[NC], N[4], T[NC];
  pzero(T);
  T[0] = 0.5 + Gp[0];
  for (int j = 1; j <= K; ++j) T[ix(0, j)] = Gp[j];
  if (K >= 1) {
    anchor_k<K>(z, nu, KA, Qa, N);
    const double t0 = math::exp(cp[0] + KA[0]);
    for (int j = 1; j <= K; ++j) KA[ix(0, j)] += cp[j];
    double tp[NC];
    pexp<K>(KA, tp);
    for (int n = 1; n <= K; ++n)
      for (int j = 0; j < n; ++j) {
        const int i = n - j;
        T[ix(i, j)] = t0 * tp[ix(i - 1, j)] / i;
      }
  }
  plog<K>(T, L);
  L[0] = math::log(T[0]);
}

// ---------------------------------------------------------------- Gaussian tail (shift)

template <int K>
LAPLACE_ST_NOINLINE LAPLACE_HD void tail_shift(double a, double nu, const par& pp, double* L) {
  constexpr int K1 = K > 0 ? K - 1 : 0;
  const double z = -a, z2 = a * a, A = nu + z2, m = nu + 1.0;
  double KA[NC], Qa[NC], N[4];
  anchor_k<K>(z, nu, KA, Qa, N);
  const double mP = (1.0 - a) * (1.0 + a), c3 = 3.0 * z2 - 1.0;
  double M[NC];
  pzero(M);
  const double sigma = pp.sig_g * 2.0 / m, h = pp.h_g, tlo = -math::log(pp.start / pp.alpha_g);
  for (int k = 0; k < 400; ++k) {
    double Ek, Dk;
    tm_node(k, h, pp.alpha_g, pp.start, Ek, Dk);
    const double t = tlo + k * h, zeta = sigma * Ek, wr = sigma * Dk;
    const double ez = ibd::node_expm1(zeta, false), Eu = 1.0 + ez;
    const double Au = A * Eu, delta = A * ez, r = Kokkos::sqrt(z2 + delta), u = delta / (a + r);
    const double W = wr * math::exp(-0.5 * m * zeta) * Au / (2.0 * r);
    if (pp.count) ++*pp.count;
    if (k > 0 && t > 0.0) {
      const double g = 1.0 + zeta / sigma;
      double gk = 1.0;
      for (int i = 0; i < K; ++i) gk *= g;
      if (W * gk < pp.tol * M[0]) break;
    }
    double P[NC], E[NC];
    pzero(P);
    // pure nu: P = A, Q = A_u
    {
      const double e = ez / Eu, rho = 1.0 / Eu;
      const double G1 = zeta <= 1.0 ? ibd::expm1mx(-zeta) : zeta - e;
      double C[6];
      pure_nu(K, A, e, rho, G1, mP, C);
      for (int j = 1; j <= K; ++j) P[ix(0, j)] = C[j];
    }
    if (K >= 1) {
      double Qu[NC], tq[NC], DQ[NC];
      pinv_sparse<K1>(Au, -2.0 * r, 1.0, 1.0, Qu);
      // tq = Dd Qu, Dd = delta - 2u ds
      for (int n = 0; n <= K1; ++n)
        for (int j = 0; j <= n; ++j) {
          const int i = n - j;
          tq[ix(i, j)] = delta * Qu[ix(i, j)] - (i >= 1 ? 2.0 * u * Qu[ix(i - 1, j)] : 0.0);
        }
      pmul<K1>(Qa, tq, DQ);
      const double dN[3] = {u * (c3 + u * (3.0 * a + u)), -3.0 * delta, 3.0 * u};
      for (int n = 0; n <= K1; ++n)
        for (int j = 0; j <= n; ++j) {
          const int i = n - j;
          double ks;
          if (j == 0) {
            // D(s Q) = -u Qu + s DQ, DQ = -Qa Dd Qu
            const double dsq = -u * Qu[ix(i, 0)] - (z * DQ[ix(i, 0)] + (i >= 1 ? DQ[ix(i - 1, 0)] : 0.0));
            ks = -m * dsq;
          } else {
            double v = 0.0;
            for (int kk = 0; kk <= 2 && kk <= i; ++kk) v += dN[kk] * Qu[ix(i - kk, j)];
            for (int kk = 0; kk <= 3 && kk <= i; ++kk) v -= N[kk] * DQ[ix(i - kk, j)];
            ks = -v;
          }
          P[ix(i + 1, j)] = ks / (i + 1);
        }
    }
    pexp<K>(P, E);
    for (int c = 0; c < nk<K>(); ++c) M[c] += W * E[c];
  }
  double cp[6], lm[NC];
  cpoly(nu, K, cp);
  plog<K>(M, lm);
  pzero(L);
  for (int c = 0; c < nk<K>(); ++c) L[c] = KA[c] + lm[c];
  for (int j = 1; j <= K; ++j) L[ix(0, j)] += cp[j];
  L[0] = cp[0] + KA[0] + math::log(M[0]);
}

// ---------------------------------------------------------------- power tail (scale)
//
// For small nu (nu < nu_w) the integral is taken at fixed w = nu v, so that
//     log T = [c(nu) - log nu + h(z, nu)] + log int_0^inf exp(-w) (1 - x m_v)^(-(nu+1)/2) dw:
// at fixed v the weight exp(-nu v) makes the nu-cumulants of the integral of
// size (j-1)!/nu^j, cancelling against the pole of c(nu) to leave L's own
// nu^-(j-1) (a loss growing like 1/nu, measured 1300-2400 ulps at nu = 0.07).
// With a2 = nu/2 and R(a2) = log Gamma(a2 + 1/2) - log Gamma(a2 + 1) (regular at 0),
//     c - log nu + h = R(a2) + (nu/2) log nu - log(2)/2 - log(2 pi)/2 + log a - ((nu + 1)/2) log A,
// whose term (nu/2) log nu is the genuine singular part of log T as nu -> 0.
// Hh: its pure nu-coefficients and constant (the z-coefficients are set by the caller).
LAPLACE_HD void small_nu_anchor(double a, double nu, int K, double* Hh) {
  const double A = nu + a * a, m = nu + 1.0, a2 = 0.5 * nu, lnu = math::log(nu), lA = math::log(A);
  double dp[5], dD;
  ibd::dpsi_all(a2 + 0.5, 0.5, K > 1 ? K - 1 : 0, dp, dD);  // psi^(n)(a2 + 1/2) - psi^(n)(a2 + 1)
  Hh[0] = (math::lgamma(a2 + 0.5) - math::lgamma(a2 + 1.0)) + 0.5 * nu * lnu - 0.5 * ln2 - half_log_2pi + math::log(a) -
          0.5 * m * lA;
  double s = 1.0, iAj = 1.0, inj = 1.0, lam_prev = lA;
  for (int j = 1; j <= K; ++j) {
    s *= 0.5 / j;  // 2^-j / j!
    iAj /= A;
    const double lam = ((j % 2 == 1) ? 1.0 : -1.0) * iAj / j;  // coefficients of log(A + d)
    if (j >= 2) inj /= nu;  // nu^(1-j)
    const double Q = j == 1 ? 0.5 * (lnu + 1.0) : 0.5 * ((j % 2 == 0) ? 1.0 : -1.0) * inj / (j * (j - 1.0));
    Hh[ix(0, j)] = s * dp[j - 1] + Q - 0.5 * (m * lam + lam_prev);
    lam_prev = lam;
  }
}


template <int K>
LAPLACE_ST_NOINLINE LAPLACE_HD void tail_scale(double a, double nu, const par& pp, double* L) {
  constexpr int K1 = K > 0 ? K - 1 : 0;
  const double z = -a, z2 = a * a, A = nu + z2, m = nu + 1.0, x = nu / A;
  // anchor h = log|z| + k(z, nu)
  double Hh[NC];
  pzero(Hh);
  const bool fw = nu < pp.nu_w;
  if (!fw) {
    const double F0 = Kokkos::log1p(z2 / nu), e = z2 / A;
    Hh[0] = math::log(a) - 0.5 * m * F0;
    double C[6];
    pure_nu(K, nu, e, x, g1_of(e, F0), 1.0, C);
    for (int j = 1; j <= K; ++j) Hh[ix(0, j)] = C[j];
  } else {
    // c(nu) - log(nu) + h, its pole in nu written out (see small_nu_anchor)
    small_nu_anchor(a, nu, K, Hh);
  }
  double D[NC], xp[NC];
  pzero(D);
  D[ix(0, 0)] = A;
  D[ix(1, 0)] = 2.0 * z;
  D[ix(0, 1)] = 1.0;
  if (K >= 2) D[ix(2, 0)] = 1.0;
  {
    double Q[NC];
    pinv<K>(D, Q);
    // x = (nu + dnu) Q
    for (int n = 0; n <= K; ++n)
      for (int j = 0; j <= n; ++j) {
        const int i = n - j;
        xp[ix(i, j)] = nu * Q[ix(i, j)] + (j >= 1 ? Q[ix(i, j - 1)] : 0.0);
      }
  }
  if (K >= 1) {
    // dh/dz = (nu + dnu)(1 - (z + dz)^2) / ((z + dz) D)
    double ZD[NC], iZD[NC], O[NC], Hs[NC];
    pzero(ZD);
    for (int n = 0; n <= K1; ++n)
      for (int j = 0; j <= n; ++j) {
        const int i = n - j;
        ZD[ix(i, j)] = z * D[ix(i, j)] + (i >= 1 ? D[ix(i - 1, j)] : 0.0);
      }
    pinv<K1>(ZD, iZD);
    pzero(O);
    O[ix(0, 0)] = nu * (1.0 - a) * (1.0 + a);
    O[ix(0, 1)] = (1.0 - a) * (1.0 + a);
    if (K1 >= 1) {
      O[ix(1, 0)] = -2.0 * z * nu;
      if (K1 >= 2) O[ix(1, 1)] = -2.0 * z;
      if (K1 >= 2) O[ix(2, 0)] = -nu;
      if (K1 >= 3) O[ix(2, 1)] = -1.0;
    }
    pmul<K1>(O, iZD, Hs);
    for (int n = 0; n <= K1; ++n)
      for (int j = 0; j <= n; ++j) {
        const int i = n - j;
        Hh[ix(i + 1, j)] = Hs[ix(i, j)] / (i + 1);
      }
  }
  double M[NC];
  pzero(M);
  const double sigma = pp.sig_p / m, h = pp.h_p, tlo = -math::log(pp.start / pp.alpha_p);
  for (int k = 0; k < 400; ++k) {
    double Ek, Dk;
    tm_node(k, h, pp.alpha_p, pp.start, Ek, Dk);
    const double t = tlo + k * h, v = sigma * Ek, wr = sigma * Dk;
    const double mv = -ibd::node_expm1(-2.0 * v, false);
    const double w0 = -x * mv, lw = Kokkos::log1p(w0);
    const double W = (fw ? nu : 1.0) * wr * math::exp(-nu * v - 0.5 * m * lw);
    if (pp.count) ++*pp.count;
    if (k > 0 && t > 0.0) {
      const double g = 1.0 + v / sigma;
      double gk = 1.0;
      for (int i = 0; i < K; ++i) gk *= g;
      if (W * gk < pp.tol * M[0]) break;
    }
    double P[NC], E[NC];
    if (K >= 1) {
      double Wp[NC], Lg[NC];
      if (!fw) {
        for (int c = 0; c < NC; ++c) Wp[c] = -mv * xp[c];
      } else {
        // at fixed w = nu v: m_v = 1 - exp(-2 w/(nu + dnu)) is a polynomial in dnu,
        // phi = -2w/(nu + dnu) = -2v sum_j (-dnu/nu)^j
        double phi[6], ep[6], mh[6];
        double r = 1.0;
        phi[0] = 0.0;
        for (int j = 1; j <= K; ++j) {
          r *= -1.0 / nu;
          phi[j] = -2.0 * v * r;
        }
        uexp(K, phi, ep);
        const double e2v = 1.0 - mv;
        for (int j = 1; j <= K; ++j) mh[j] = -e2v * ep[j];
        for (int n = 0; n <= K; ++n)
          for (int j = 0; j <= n; ++j) {
            const int i = n - j;
            double sm = mv * xp[ix(i, j)];
            for (int l = 1; l <= j; ++l) sm += mh[l] * xp[ix(i, j - l)];
            Wp[ix(i, j)] = -sm;
          }
      }
      Wp[0] = 1.0 + w0;
      plog<K>(Wp, Lg);
      Lg[0] = lw;
      P[0] = 0.0;
      for (int n = 1; n <= K; ++n)
        for (int j = 0; j <= n; ++j) {
          const int i = n - j;
          P[ix(i, j)] = -0.5 * (m * Lg[ix(i, j)] + (j >= 1 ? Lg[ix(i, j - 1)] : 0.0));
        }
      if (!fw) P[ix(0, 1)] -= v;
      pexp<K>(P, E);
    } else {
      E[0] = 1.0;
    }
    for (int c = 0; c < nk<K>(); ++c) M[c] += W * E[c];
  }
  double cp[6], lm[NC];
  if (fw) {
    for (int j = 0; j <= K; ++j) cp[j] = 0.0;
  } else {
    cpoly(nu, K, cp);
  }
  plog<K>(M, lm);
  pzero(L);
  for (int c = 0; c < nk<K>(); ++c) L[c] = Hh[c] + lm[c];
  for (int j = 1; j <= K; ++j) L[ix(0, j)] += cp[j];
  L[0] = cp[0] + Hh[0] + math::log(M[0]);
}

// ---------------------------------------------------------------- the upper side

// L (normalized coefficients of log S at -z) -> those of log(1 - S) at z
template <int K>
LAPLACE_HD void complement(double* L) {
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j)
      if ((n - j) % 2 == 1) L[ix(n - j, j)] = -L[ix(n - j, j)];
  const double l0 = L[0];
  double E[NC];
  pexp<K>(L, E);
  if (l0 < -620.0) {
    // S below 2^-894: log(1 - S) = -S E to first order, formed in logarithms
    // so that a subnormal S keeps its digits
    for (int c = 1; c < nk<K>(); ++c)
      L[c] = E[c] == 0.0 ? 0.0 : (E[c] > 0.0 ? -1.0 : 1.0) * math::exp(l0 + math::log(Kokkos::fabs(E[c])));
    L[0] = -math::exp(l0);
    return;
  }
  const double S = math::exp(l0), oneS = -Kokkos::expm1(l0), r = S / oneS;
  double p[NC];
  for (int c = 0; c < nk<K>(); ++c) p[c] = -r * E[c];
  p[0] = 1.0;
  plog<K>(p, L);
  L[0] = Kokkos::log1p(-S);
}

// ---------------------------------------------------------------- the driver

// log T(-a) by own::beta_inc: T(-a) = I_x(nu/2, 1/2)/2 or (1 - I_y(1/2, nu/2))/2,
// with the low part of the logarithm (beta_inc_result::log_i_lo) in *lo
LAPLACE_HD_FORCEINLINE double log_lower_beta(double a, double nu, double* lo = nullptr) {
  const double z2 = a * a, A = nu + z2, x = nu / A, y = z2 / A;
  double li, li_lo;
  if (x <= 0.5) {
    const auto r = own::beta_inc(0.5 * nu, 0.5, x);
    li = r.log_i;
    li_lo = r.log_i_lo;
  } else {
    const auto r = own::beta_inc(0.5, 0.5 * nu, y);
    li = r.log_ic;
    li_lo = r.log_ic_lo;
  }
  if (lo) *lo = li_lo;
  return li - ln2;
}

template <int K>
LAPLACE_HD void derivs(double z, double nu, const par& pp, double* L) {
  const double nan = math::detail::quiet_nan();
  pzero(L);
  if (!(nu > 0.0) || !(nu < math::detail::pos_inf()) || z != z) {
    for (int c = 0; c < NC; ++c) L[c] = nan;
    return;
  }
  if (z == -math::detail::pos_inf()) {
    for (int c = 1; c < NC; ++c) L[c] = nan;
    L[0] = -math::detail::pos_inf();
    return;
  }
  if (z == math::detail::pos_inf()) return;
  const double a = Kokkos::fabs(z);
  if (a == 0.0 || log_lower_beta(a, nu) > math::log(pp.t_switch)) {
    centre<K>(z, nu, pp, L);
    return;
  }
  if (a * a < pp.shift_ratio * nu) tail_shift<K>(a, nu, pp, L);
  else tail_scale<K>(a, nu, pp, L);
  if (z > 0.0) complement<K>(L);
}

// ---------------------------------------------------------------- candidate 2: beta composition

// d^k log I / da^k in the frame (pure in the shape a = nu/2)
template <int k>
LAPLACE_HD double pure_a(bool upper, double a, double xi) {
  if (!upper) return ibd::order<k>(a, 0.5, xi, false).c[0];
  return ibd::order<k>(0.5, a, xi, true).c[k];
}

template <int K>
LAPLACE_ST_NOINLINE LAPLACE_HD void lower_beta(double a, double nu, double* Lz) {
  const double z = -a, z2 = a * a, A = nu + z2, x = nu / A, y = z2 / A, sa = 0.5 * nu;
  const bool upper = x > 0.5;
  const double xi = upper ? y : x, sxi = upper ? -1.0 : 1.0;  // dx = sxi dxi
  // L(a + da, xi + dxi): Lab[ix(i, m)], i in a, m in xi
  double Lab[NC], Gd[NC], E[NC], P[NC];
  pzero(Lab);
  pzero(Gd);
  const auto bi = upper ? own::beta_inc(0.5, sa, y) : own::beta_inc(sa, 0.5, x);
  Lab[0] = upper ? bi.log_ic : bi.log_i;
  if (K >= 1) Lab[ix(1, 0)] = pure_a<1>(upper, sa, xi);
  if constexpr (K >= 2) Lab[ix(2, 0)] = pure_a<2>(upper, sa, xi) / 2.0;
  if constexpr (K >= 3) Lab[ix(3, 0)] = pure_a<3>(upper, sa, xi) / 6.0;
  if constexpr (K >= 4) Lab[ix(4, 0)] = pure_a<4>(upper, sa, xi) / 24.0;
  if constexpr (K >= 5) Lab[ix(5, 0)] = pure_a<5>(upper, sa, xi) / 120.0;
  // log density (a - 1) log x - log(y)/2 - log B(a, 1/2), log B = -f(a) - log(a)/2 + log(pi)/2
  double f[6];
  fder(sa, K, f);
  const double lx = x < 0.5 ? math::log(x) : Kokkos::log1p(-y), ly = x < 0.5 ? Kokkos::log1p(-x) : math::log(y);
  Gd[0] = (sa - 1.0) * lx - 0.5 * ly + f[0] + 0.5 * math::log(sa) - 0.5723649429247001;
  double fct = 1.0, sap = 1.0;
  for (int i = 1; i <= K; ++i) {
    fct *= i;
    sap /= sa;
    // d^i log B / da^i = -(f^(i) + (-1)^(i-1) (i-1)!/(2 a^i))
    const double dlb = -(f[i] + ((i % 2 == 1) ? 0.5 : -0.5) * (fct / i) * sap);
    Gd[ix(i, 0)] = -dlb / fct;
  }
  Gd[ix(1, 0)] += lx;
  // log x and log y in dxi: x + sxi dxi, y - sxi dxi
  for (int mm = 1; mm <= K; ++mm) {
    const double sg = (mm % 2 == 1) ? 1.0 : -1.0;
    const double lxm = sg * Kokkos::pow(sxi / x, mm) / mm, lym = sg * Kokkos::pow(-sxi / y, mm) / mm;
    Gd[ix(0, mm)] = (sa - 1.0) * lxm - 0.5 * lym;
    if (mm + 1 <= K) Gd[ix(1, mm)] = lxm;
  }
  // order by order: E = Gd - L, P = sxi exp(E) = dL/dxi
  for (int n = 0; n < K; ++n) {
    for (int mm = 0; mm <= n; ++mm) E[ix(n - mm, mm)] = Gd[ix(n - mm, mm)] - Lab[ix(n - mm, mm)];
    if (n == 0) {
      P[0] = sxi * math::exp(E[0]);
    } else {
      for (int mm = 0; mm <= n; ++mm) {
        const int i = n - mm;
        double v = 0.0;
        if (mm >= 1) {
          for (int k = 0; k <= i; ++k)
            for (int l = 1; l <= mm; ++l) v += l * E[ix(k, l)] * P[ix(i - k, mm - l)];
          v /= mm;
        } else {
          for (int k = 1; k <= i; ++k) v += k * E[ix(k, 0)] * P[ix(i - k, 0)];
          v /= i;
        }
        P[ix(i, mm)] = v;
      }
    }
    for (int mm = 0; mm <= n; ++mm) Lab[ix(n - mm, mm + 1)] = P[ix(n - mm, mm)] / (mm + 1);
  }
  // compose with a = nu/2 + dnu/2 and xi(z + dz, nu + dnu)
  double D[NC], Q[NC], X[NC], Xp[NC], T[NC];
  pzero(D);
  D[ix(0, 0)] = A;
  D[ix(1, 0)] = 2.0 * z;
  D[ix(0, 1)] = 1.0;
  if (K >= 2) D[ix(2, 0)] = 1.0;
  pinv<K>(D, Q);
  pzero(X);
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      if (!upper) {
        X[ix(i, j)] = nu * Q[ix(i, j)] + (j >= 1 ? Q[ix(i, j - 1)] : 0.0);
      } else {
        X[ix(i, j)] = z2 * Q[ix(i, j)] + (i >= 1 ? 2.0 * z * Q[ix(i - 1, j)] : 0.0) + (i >= 2 ? Q[ix(i - 2, j)] : 0.0);
      }
    }
  X[0] = 0.0;
  pzero(Lz);
  pzero(Xp);
  Xp[0] = 1.0;  // dxi^0
  for (int mm = 0; mm <= K; ++mm) {
    if (mm > 0) {
      pmul<K>(Xp, X, T);
      for (int c = 0; c < NC; ++c) Xp[c] = T[c];
    }
    for (int i = 0; i + mm <= K; ++i) {
      const double cf = Lab[ix(i, mm)] * Kokkos::pow(0.5, i);
      // add cf dnu^i Xp
      for (int n = 0; n + i <= K; ++n)
        for (int j = 0; j <= n; ++j) Lz[ix(n - j, j + i)] += cf * Xp[ix(n - j, j)];
    }
  }
}

template <int K>
LAPLACE_HD void derivs_beta(double z, double nu, double* L) {
  const double nan = math::detail::quiet_nan();
  pzero(L);
  if (!(nu > 0.0) || !(nu < math::detail::pos_inf()) || z != z || Kokkos::fabs(z) == math::detail::pos_inf()) {
    for (int c = 0; c < NC; ++c) L[c] = nan;
    return;
  }
  const double a = Kokkos::fmax(Kokkos::fabs(z), 1e-150);  // T(0) = 1/2: the limit
  lower_beta<K>(a, nu, L);
  L[0] -= ln2;
  if (z > 0.0) complement<K>(L);
}

// ---------------------------------------------------------------- public helpers

template <int K>
LAPLACE_HD_FORCEINLINE shape_deriv<K> pick(const double* L) {
  shape_deriv<K> r;
  for (int j = 0; j <= K; ++j) r.c[j] = L[ix(K - j, j)] * ibd::factorial(K - j) * ibd::factorial(j);
  return r;
}

template <int K>
LAPLACE_HD shape_deriv<K> order(double z, double nu) {
  double L[NC];
  derivs<K>(z, nu, par{}, L);
  return pick<K>(L);
}

// the value by own::beta_inc (a candidate for log_student_cdf)
LAPLACE_HD double log_cdf_beta(double z, double nu) {
  if (!(nu > 0.0) || z != z) return math::detail::quiet_nan();
  if (z == 0.0) return -ln2;
  double lo = 0.0;
  const double lS = log_lower_beta(Kokkos::fabs(z), nu, &lo);
  if (z < 0.0) return lS;
  // T(-|z|) = e^(lS + lo): far in the tail |lS| / 2 ulps are lost without lo
  double S = math::exp(lS);
  S += S * lo;
  return Kokkos::log1p(-S);
}

// the value by the quadrature
LAPLACE_HD double log_cdf_quad(double z, double nu) {
  double L[NC];
  derivs<0>(z, nu, par{}, L);
  return L[0];
}

}  // namespace detail_st

LAPLACE_HD double log_student_cdf(double z, double nu) { return detail_st::log_cdf_beta(z, nu); }

LAPLACE_HD shape_deriv<1> log_student_cdf_d1(double z, double nu) { return detail_st::order<1>(z, nu); }
LAPLACE_HD shape_deriv<2> log_student_cdf_d2(double z, double nu) { return detail_st::order<2>(z, nu); }
LAPLACE_HD shape_deriv<3> log_student_cdf_d3(double z, double nu) { return detail_st::order<3>(z, nu); }
LAPLACE_HD shape_deriv<4> log_student_cdf_d4(double z, double nu) { return detail_st::order<4>(z, nu); }
LAPLACE_HD shape_deriv<5> log_student_cdf_d5(double z, double nu) { return detail_st::order<5>(z, nu); }

}  // namespace laplace::math::own

namespace laplace::math {

// What the rest of Laplace calls (docs/numerics.md, 2026-11-09): the value by
// own::beta_inc, the derivatives by the three-region quadrature.
LAPLACE_HD_FORCEINLINE double log_student_cdf(double z, double nu) { return own::log_student_cdf(z, nu); }
LAPLACE_HD_FORCEINLINE shape_deriv<1> log_student_cdf_d1(double z, double nu) { return own::log_student_cdf_d1(z, nu); }
LAPLACE_HD_FORCEINLINE shape_deriv<2> log_student_cdf_d2(double z, double nu) { return own::log_student_cdf_d2(z, nu); }
LAPLACE_HD_FORCEINLINE shape_deriv<3> log_student_cdf_d3(double z, double nu) { return own::log_student_cdf_d3(z, nu); }
LAPLACE_HD_FORCEINLINE shape_deriv<4> log_student_cdf_d4(double z, double nu) { return own::log_student_cdf_d4(z, nu); }
LAPLACE_HD_FORCEINLINE shape_deriv<5> log_student_cdf_d5(double z, double nu) { return own::log_student_cdf_d5(z, nu); }

}  // namespace laplace::math
