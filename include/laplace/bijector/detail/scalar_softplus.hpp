// Scalar bijector softplus (laplace::bijector::detail_scalar). Internal:
// the public interface is the member functions of scalar_bijector (scalar.hpp).
//
//   softplus: (0, inf) -> R, g(v) = log(e^(p1 v) - 1)/p1, p1 > 0; p1 = sharpness a, p2 unused
//
// Written 2026-11-12; the derivation and its checks are in
// tools/reference/bijector_powsoft_forms.py. Notation, with a = p1 > 0:
//
//   forward: x = a v > 0, t = 1/expm1(x), M = log(1 - e^-x), so that g = v + M/a.
//            Z = log(e^x - 1) = x + M has Z' = 1 + t and Z^(r) = t^(r-1) for r >= 2,
//            and t^(k) = Q_k(t) with Q_k polynomials of one sign (Q_1 = -(t + t^2), ...).
//   inverse: x = a u, sigma = expit, S = log(1 + e^x) = x + L (x > 0, L = log(1 + e^-x)),
//            so that h = S/a, h^(K) = a^(K-1) sigma^(K-1)(x), and S^(r) = sigma^(r-1).
//   log-Jacobians: log g'(v) = -M(x) (since g' = 1/(1 - e^-x)), with derivatives
//            -a^K t^(K-1); log h'(u) = log sigma(x), with derivatives -a^K sigma^(K-1).
//
// Mixed derivatives in (argument, a), of total order K, with n0 derivatives in the
// argument and n1 in a. With ff(m, j) = m (m-1) ... (m-j+1), ff(-1, j) = (-1)^j j!:
//
//   Phi-type (g = Z(a v)/a, h = S(a u)/a):
//     d^n0 d^n1 [Phi(a v)/a] = a^(n0-n1-1) sum_(j <= min(n0-1, n1)) C(n1,j) ff(n0-1,j) x^(n1-j) Phi^(n0+n1-j)(x),
//   pure a-derivatives (n0 = 0) are written without Phi itself, whose x-linear part
//   cancels: for the forward map T = (-1)^n1 n1! M + (-1)^(n1-1) n1! x t + sum_(k>=2) ...;
//   for the inverse map, x > 0: T = (-1)^n1 n1! (L + x eps) + sum_(k>=2) ..., eps = sigma(-x),
//   and x <= 0: T = sum_(k=0..n1) C(n1,k)(-1)^(n1-k)(n1-k)! x^k S^(k)(x).
//
//   Psi-type (log-Jacobians, Psi(a v)):
//     d^n0 d^n1 [Psi(a v)] = a^(n0-n1) sum_(i <= min(n0,n1)) C(n0,i) (n1!/(n1-i)!) x^(n1-i) Psi^(n0+n1-i)(x).
//   For the forward log-Jacobian, Psi = log(1/(1 - e^-x)) = -log x + lambda(x) with
//   lambda(x) = -log exprel(-x), and the log x part contributes nothing to the mixed
//   derivatives: they are taken from lambda, from its Bernoulli series for 0 < x < 3 (the
//   derivatives of lambda are O(x) for some orders, so a closed form cancels there). For
//   x >= 3 the derivatives of Psi itself are used: lambda is then of order log x, and
//   its derivatives would cancel to the exponentially small log-Jacobian.
//
// Cancellation, and how each is avoided: t and sigma are computed from expm1 and from e = e^-|x|;
// M and L use log1p where e^-x is small; the x-linear parts of the pure a-derivatives of
// Z and S are cancelled analytically (above); the derivatives of expit are written in U = p q
// and D = q - p, as in scalar.hpp.
//
// Invalid sharpness (a <= 0, NaN, infinite): every function returns NaN.
#pragma once

#include <laplace/bijector/detail/scalar_helpers.hpp>
#include <laplace/math/exprel.hpp>

namespace laplace::bijector::detail_scalar {

LAPLACE_HD_FORCEINLINE bool softplus_valid(double a) { return a > 0.0 && Kokkos::isfinite(a); }

// ---------------------------------------------------------------- integer helpers, orders <= 6

LAPLACE_HD_FORCEINLINE double sp_binom(int n, int k) {  // C(n, k), 0 <= k <= n <= 6, exact
  // integer arithmetic: the double division of the device is slow (FP64 at 1/64 rate)
  int c = 1;
  for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
  return double(c);
}
LAPLACE_HD_FORCEINLINE double sp_ff(int m, int j) {  // m (m - 1) ... (m - j + 1)
  double f = 1.0;
  for (int i = 0; i < j; ++i) f *= double(m - i);
  return f;
}
LAPLACE_HD_FORCEINLINE double sp_fact(int n) {  // n!, 0 <= n <= 6
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= double(i);
  return f;
}
LAPLACE_HD_FORCEINLINE double sp_sgn(int n) { return (n & 1) ? -1.0 : 1.0; }  // (-1)^n, n >= 0
LAPLACE_HD_FORCEINLINE double sp_ipow(double x, int k) {                      // x^k, k >= 0
  double r = 1.0;
  for (int i = 0; i < k; ++i) r *= x;
  return r;
}

// ---------------------------------------------------------------- expit, inverse side

// p = sigma(x), q = sigma(-x), U = p q, D = q - p, and s[j] = sigma^(j)(x), j = 0..5,
// with sigma = expit. q and D are exact for x > 0 (the small one is formed directly).
struct sp_expit {
  double p, q, U, D, s[6];
};
LAPLACE_HD_FORCEINLINE sp_expit sp_expit_of(double x) {
  const double ax = Kokkos::fabs(x), e = m::exp(-ax), d = 1.0 + e;
  const double big = 1.0 / d, small = e / d;
  sp_expit w{};
  w.p = x >= 0.0 ? big : small;
  w.q = x >= 0.0 ? small : big;
  w.U = small * big;
  w.D = Kokkos::copysign(Kokkos::expm1(-ax) / d, -x);
  w.s[0] = w.p;
  w.s[1] = w.U;
  w.s[2] = w.U * w.D;
  w.s[3] = w.U * (1.0 - 6.0 * w.U);
  w.s[4] = w.U * w.D * (1.0 - 12.0 * w.U);
  w.s[5] = w.U * (1.0 - 30.0 * w.U + 120.0 * w.U * w.U);
  return w;
}

// log(1 + e^x) for x <= 0 and log(1 + e^-x) for x > 0, both without overflow
LAPLACE_HD_FORCEINLINE double sp_softplus_neg(double x) {  // log(1 + e^(-|x|)) = L(|x|)
  return Kokkos::log1p(m::exp(-Kokkos::fabs(x)));
}

// ---------------------------------------------------------------- forward side

// t = 1/expm1(x) and q[k] = t^(k)(x), k = 0..5, with t^(k) = Q_k(t) (the forms script)
LAPLACE_HD_FORCEINLINE void sp_t_derivs(double t, double (&q)[6]) {
  constexpr double c1[] = {-1.0, -1.0};
  constexpr double c2[] = {1.0, 3.0, 2.0};
  constexpr double c3[] = {-1.0, -7.0, -12.0, -6.0};
  constexpr double c4[] = {1.0, 15.0, 50.0, 60.0, 24.0};
  constexpr double c5[] = {-1.0, -31.0, -180.0, -390.0, -360.0, -120.0};
  q[0] = t;
  q[1] = t * poly(c1, t);
  q[2] = t * poly(c2, t);
  q[3] = t * poly(c3, t);
  q[4] = t * poly(c4, t);
  q[5] = t * poly(c5, t);
}

// M(x) = log(1 - e^-x), x > 0, through log(-expm1(-x)) for small x and log1p(-e^-x) beyond log 2
LAPLACE_HD_FORCEINLINE double sp_M(double x) {
  if (x < 0.6931471805599453) return m::log(-Kokkos::expm1(-x));
  return Kokkos::log1p(-m::exp(-x));
}

// lam[r] = lambda^(r)(x), r = 1..6, for 0 < x < 3, from the Bernoulli series
//   lambda(x) = -log exprel(-x) = x/2 - sum_(m >= 1) e_m x^(2m),  e_m = B_(2m)/(2m (2m)!),
// so that lambda^(r)(x) = [r = 1]/2 - sum_(m: 2m >= r) e_m ff(2m, r) x^(2m - r). The series
// has radius 2 pi; forty terms leave less than 1e-18 at x = 3 for every order. The
// coefficients are checked in tools/reference/bijector_powsoft_forms.py (point 5).
// Above x = 3 the log-Jacobian's own derivatives are used (sp_fwd_lj_mixed).
LAPLACE_HD_FORCEINLINE void sp_lam_derivs(double x, double (&lam)[7]) {
  constexpr double e[41] = {
      0.0,
      0.041666666666666664,
      -0.00034722222222222224,
      5.5114638447971785e-06,
      -1.033399470899471e-07,
      2.08767569878681e-09,
      -4.403491782239578e-11,
      9.55895466477477e-13,
      -2.1185501852016142e-14,
      4.770034475709914e-16,
      -1.087434349279031e-17,
      2.5040921947091955e-19,
      -5.814360285755218e-21,
      1.3595027075497952e-22,
      -3.1976847953705525e-24,
      7.559841507792277e-26,
      -1.7952470840225633e-27,
      4.279919045926073e-29,
      -1.0238874835181417e-30,
      2.4570353308144855e-32,
      -5.912556039251575e-34,
      1.4263504196386035e-35,
      -3.448761101064296e-37,
      8.355995924900409e-39,
      -2.0284068901347272e-40,
      4.932494088401362e-42,
      -1.2013609118886046e-43,
      2.93037597123054e-45,
      -7.157631582051671e-47,
      1.7505303199257856e-48,
      -4.286340263736453e-50,
      1.0507187153602927e-51,
      -2.578329672952582e-53,
      6.333076709908297e-55,
      -1.5570050981916042e-56,
      3.831255987515387e-58,
      -9.435110209871418e-60,
      2.325348326578568e-61,
      -5.735171485985387e-63,
      1.4154862697031623e-64,
      -3.495831891722668e-66};
  const double x2 = x * x;
  for (int r = 1; r <= 6; ++r) {
    const int mstart = (r + 1) / 2;  // the smallest m with 2m >= r
    double s = 0.0;
    for (int m = 40; m >= mstart; --m) s = s * x2 + e[m] * sp_ff(2 * m, r);
    // the terms are x^(2m - r) = x^(2 (m - mstart)) x^(2 mstart - r)
    lam[r] = (r == 1 ? 0.5 : 0.0) - s * sp_ipow(x, 2 * mstart - r);
  }
}

struct sp_fwd {
  double v, a, x, t, M, q[6], lam[7];
};
LAPLACE_HD_FORCEINLINE sp_fwd sp_fwd_prepare(double v, double a, bool need_lam) {
  sp_fwd s{};
  s.v = v;
  s.a = a;
  s.x = a * v;
  s.t = 1.0 / Kokkos::expm1(s.x);
  sp_t_derivs(s.t, s.q);
  s.M = sp_M(s.x);
  if (need_lam && s.x < 3.0) sp_lam_derivs(s.x, s.lam);
  return s;
}
// Z^(r)(x) for r >= 1: 1 + t for r = 1, t^(r-1) beyond
LAPLACE_HD_FORCEINLINE double sp_fwd_Zd(const sp_fwd& s, int r) { return r == 1 ? 1.0 + s.t : s.q[r - 1]; }

// d^n0 d^n1 g at (v, a), with n0 + n1 >= 1 or the value when both are 0
LAPLACE_HD_FORCEINLINE double sp_fwd_mixed(const sp_fwd& s, int n0, int n1) {
  const double x = s.x;
  if (n0 == 0) {
    if (n1 == 0) return s.v + s.M / s.a;  // g = v + M(x)/a
    double T = sp_sgn(n1) * sp_fact(n1) * s.M;
    T += sp_sgn(n1 - 1) * sp_fact(n1) * x * s.t;  // the x-linear part of Z has cancelled
    for (int k = 2; k <= n1; ++k)
      T += sp_binom(n1, k) * sp_sgn(n1 - k) * sp_fact(n1 - k) * sp_ipow(x, k) * s.q[k - 1];
    return T * pow_int(s.a, -n1 - 1);
  }
  double sum = 0.0;
  const int jmax = n0 - 1 < n1 ? n0 - 1 : n1;
  for (int j = 0; j <= jmax; ++j)
    sum += sp_binom(n1, j) * sp_ff(n0 - 1, j) * sp_ipow(x, n1 - j) * sp_fwd_Zd(s, n0 + n1 - j);
  return sum * pow_int(s.a, n0 - n1 - 1);
}

// d^n0 d^n1 log g'(v) = d^n0 d^n1 log F(a v), F = 1/(1 - e^-x), at (v, a)
LAPLACE_HD_FORCEINLINE double sp_fwd_lj_mixed(const sp_fwd& s, int n0, int n1) {
  if (n1 == 0) {
    if (n0 == 0) return -s.M;
    return -pow_int(s.a, n0) * s.q[n0 - 1];
  }
  if (n0 == 0) return -sp_ipow(s.v, n1) * s.q[n1 - 1];  // a^(-n1) x^n1 = v^n1
  double sum = 0.0;
  const int imax = n0 < n1 ? n0 : n1;
  for (int i = 0; i <= imax; ++i) {
    const double w = sp_binom(n0, i) * sp_fact(n1) / sp_fact(n1 - i) * sp_ipow(s.x, n1 - i);
    sum += s.x < 3.0 ? w * s.lam[n0 + n1 - i] : -w * s.q[n0 + n1 - i - 1];
  }
  return sum * pow_int(s.a, n0 - n1);
}

// ---------------------------------------------------------------- inverse side

struct sp_inv {
  double u, a, x, S0, L, f;  // S0 = log(1 + e^x), L = log(1 + e^-x) (x > 0), f = log sigma(x)
  sp_expit w;
};
LAPLACE_HD_FORCEINLINE sp_inv sp_inv_prepare(double u, double a) {
  sp_inv s{};
  s.u = u;
  s.a = a;
  s.x = a * u;
  s.w = sp_expit_of(s.x);
  s.L = sp_softplus_neg(s.x);
  if (s.x > 0.0) {
    s.S0 = 0.0;  // not used: S = x + L is taken in the form u + L/a
    s.f = -s.L;
  } else {
    s.S0 = Kokkos::log1p(m::exp(s.x));
    s.f = s.x - s.S0;
  }
  return s;
}

// d^n0 d^n1 h at (u, a)
LAPLACE_HD_FORCEINLINE double sp_inv_mixed(const sp_inv& s, int n0, int n1) {
  const double x = s.x, a = s.a;
  if (n0 == 0 && n1 == 0) return x > 0.0 ? s.u + s.L / a : s.S0 / a;  // h = S(x)/a
  if (n0 >= 1) {
    double sum = 0.0;
    const int jmax = n0 - 1 < n1 ? n0 - 1 : n1;
    for (int j = 0; j <= jmax; ++j)
      sum += sp_binom(n1, j) * sp_ff(n0 - 1, j) * sp_ipow(x, n1 - j) * s.w.s[n0 + n1 - j - 1];
    return sum * pow_int(a, n0 - n1 - 1);
  }
  // pure a-derivative, n1 >= 1
  double T = 0.0;
  if (x > 0.0) {
    T = sp_sgn(n1) * sp_fact(n1) * (s.L + x * s.w.q);  // the x-linear part of S has cancelled
    for (int k = 2; k <= n1; ++k)
      T += sp_binom(n1, k) * sp_sgn(n1 - k) * sp_fact(n1 - k) * sp_ipow(x, k) * s.w.s[k - 1];
  } else {
    T = sp_sgn(n1) * sp_fact(n1) * s.S0;
    for (int k = 1; k <= n1; ++k)
      T += sp_binom(n1, k) * sp_sgn(n1 - k) * sp_fact(n1 - k) * sp_ipow(x, k) * s.w.s[k - 1];
  }
  return T * pow_int(a, -n1 - 1);
}

// d^n0 d^n1 log h'(u) = d^n0 d^n1 log sigma(a u) at (u, a)
LAPLACE_HD_FORCEINLINE double sp_inv_lj_mixed(const sp_inv& s, int n0, int n1) {
  if (n0 == 0 && n1 == 0) return s.f;
  double sum = 0.0;
  const int imax = n0 < n1 ? n0 : n1;
  for (int i = 0; i <= imax; ++i) {
    const int r = n0 + n1 - i;  // order of the derivative of log sigma, r >= 1
    const double w = sp_binom(n0, i) * sp_fact(n1) / sp_fact(n1 - i) * sp_ipow(s.x, n1 - i);
    // (log sigma)' = 1 - sigma = q (exact for x > 0); d^r log sigma = -sigma^(r-1) for r >= 2
    sum += w * (r == 1 ? s.w.q : -s.w.s[r - 1]);
  }
  return sum * pow_int(s.a, n0 - n1);
}

// ---------------------------------------------------------------- the maps

template <int K>
LAPLACE_HD double softplus_fwd(double v, double p1, double p2) {
  if (!softplus_valid(p1)) return nan();
  return sp_fwd_mixed(sp_fwd_prepare(v, p1, false), K, 0);
}
template <int K>
LAPLACE_HD double softplus_inv(double u, double p1, double p2) {
  if (!softplus_valid(p1)) return nan();
  return sp_inv_mixed(sp_inv_prepare(u, p1), K, 0);
}
template <int K>
LAPLACE_HD double softplus_fwd_lj(double v, double p1, double p2) {
  if (!softplus_valid(p1)) return nan();
  return sp_fwd_lj_mixed(sp_fwd_prepare(v, p1, false), K, 0);
}
template <int K>
LAPLACE_HD double softplus_inv_lj(double u, double p1, double p2) {
  if (!softplus_valid(p1)) return nan();
  return sp_inv_lj_mixed(sp_inv_prepare(u, p1), K, 0);
}

// The partials of total order K in (argument, a): component n1 (the index of
// (n1, n2 = 0)) is d^(K-n1) / (d arg^(K-n1) d a^n1); the components with n2 > 0 are 0.
template <int K>
LAPLACE_HD void softplus_fwd_partials(double v, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!softplus_valid(p1)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const sp_fwd s = sp_fwd_prepare(v, p1, true);
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = sp_fwd_mixed(s, K - n1, n1);
}
template <int K>
LAPLACE_HD void softplus_inv_partials(double u, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!softplus_valid(p1)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const sp_inv s = sp_inv_prepare(u, p1);
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = sp_inv_mixed(s, K - n1, n1);
}
template <int K>
LAPLACE_HD void softplus_fwd_lj_partials(double v, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!softplus_valid(p1)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const sp_fwd s = sp_fwd_prepare(v, p1, true);
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = sp_fwd_lj_mixed(s, K - n1, n1);
}
template <int K>
LAPLACE_HD void softplus_inv_lj_partials(double u, double p1, double p2, double* out) {
  constexpr int n = (K + 1) * (K + 2) / 2;
  for (int i = 0; i < n; ++i) out[i] = 0.0;
  if (!softplus_valid(p1)) {
    for (int i = 0; i < n; ++i) out[i] = nan();
    return;
  }
  const sp_inv s = sp_inv_prepare(u, p1);
  for (int n1 = 0; n1 <= K; ++n1) out[n1] = sp_inv_lj_mixed(s, K - n1, n1);
}

// The domains: g on (0, inf), h on R; the range of h is (0, inf).
LAPLACE_HD_FORCEINLINE double softplus_dom_lo(double p1, double p2) { return softplus_valid(p1) ? 0.0 : nan(); }
LAPLACE_HD_FORCEINLINE double softplus_dom_hi(double p1, double p2) { return softplus_valid(p1) ? inf() : nan(); }
LAPLACE_HD_FORCEINLINE double softplus_img_lo(double p1, double p2) { return softplus_valid(p1) ? -inf() : nan(); }
LAPLACE_HD_FORCEINLINE double softplus_img_hi(double p1, double p2) { return softplus_valid(p1) ? inf() : nan(); }
// dom_hi - h(u) = inf; h(u) - dom_lo = h(u) itself (no cancellation)
LAPLACE_HD_FORCEINLINE double softplus_inv_to_upper(double u, double p1, double p2) {
  return softplus_valid(p1) ? inf() : nan();
}
LAPLACE_HD_FORCEINLINE double softplus_inv_to_lower(double u, double p1, double p2) {
  return softplus_inv<0>(u, p1, p2);
}

}  // namespace laplace::bijector::detail_scalar
