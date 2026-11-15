// The negative binomial log-mass in Loader's saddle-point form, and the small algebra of
// packed pairs of derivatives that assembles it (laplace::distrib::detail_negbin). The two
// families of the library are the same mass in a shape r > 0 and a mean m > 0,
//
//   p(y) = Gamma(y + r) / (Gamma(r) y!) (r / (r + m))^r (m / (r + m))^y,   y = 0, 1, 2, ...
//
//   negbin2_distrib (mu, theta), Var = mu + mu^2 / theta:   (r, m) = (theta, mu)
//   negbin1_distrib (mu, theta), Var = mu (1 + theta):      (r, m) = (mu / theta, mu)
//
// Why Loader's form. Every grouping of that expression keeps terms of the size of y log y
// while the log-mass is of the size of log y, so at large counts the mass loses about
// log10(y) digits (measured on a first version written with Stirling's error alone: 1.4e5 ulp
// at y = 1e5). The identity that removes it is the one poisson.hpp and binomial.hpp already
// use, with delta Stirling's error (math/stirling.hpp, delta(y) = stirlerr(y)) and
// bd0(a, b) = a log(a/b) - a + b = b D(a/b), D(1 + w) = (1 + w) log1p(w) - w: for y >= 1,
//
//   l = (1/2) log( r / (2 pi y (y + r)) ) + delta(y + r) - delta(y) - delta(r)
//       - bd0(y, b1) - bd0(r, b2),
//   b1 = m (y + r) / (r + m),  b2 = r (y + r) / (r + m),  b1 + b2 = y + r,
//
// and both deviances are taken at the one numerator Delta = r (y - m) / (r + m), since
// b1 - y = -Delta and b2 - r = Delta exactly. Every term is then of the size of the result at
// any y. At y = 0 the form has log y in it and the mass is written directly as
// r log(r/(r+m)) = -(V(r, 0, m) + m) instead.
//
// How a deviance is differentiated. With psi1(z) = z - log1p(z),
//
//   bd0(a, b) = a psi1((b - a) / a),
//
// which is exact and, for the deviance at the shape (a = r, where b2 - r = Delta always), a
// product of a and one composition: z2 = Delta / r is a plain rational of the parameters and
// carries no large factor. For the deviance at the count (a = y) the two routes measured are
//   |z| < 1/2 (near the mode): the same composition, y psi1(z) with z = (b1 - y) / y;
//   |z| >= 1/2:                d^alpha bd0 = d^alpha b1 - y d^alpha log b1, both tensors closed.
// The composition's Bell sum alternates and at order 6 its terms are far larger than the result
// when the count is far from the mean (measured: 11 digits lost at y = 2.5 against a mean of
// 1000), while the two-term form cancels only near the mode, where the composition is exact.
// For the same reason every rational of the parameters is built from a closed formula with one
// numerator rather than from a product of tensors: theta / (theta + mu) and mu / (theta + mu)
// have d^(p+q) = (-1)^(p+q) (p+q-1)! (p theta - q mu) / (theta + mu)^(p+q+1) up to sign, whose
// pure derivatives in either parameter are single terms, and Leibniz's rule on
// mu (y + theta) * (theta + mu)^-1 instead loses mu/theta in the mu direction.
//
// Why each family composes it itself. The parameters enter b1, b2, Delta and r as rational
// functions, which differ between the two parametrizations, and a composition through the
// shape is not allowed for negbin1: with r = mu/theta the chain rule's terms are of the size
// of theta^(1-K) while d^K l / dtheta^K is of the size of one, a relative loss of
// theta^(K-1) (measured: order 6 at theta = 1e-6 was noise, 6e23 ulp). Written as rational
// functions of (mu, theta) directly, every inner map of both families is regular at both
// Poisson limits (theta -> infinity for negbin2, theta -> 0 for negbin1).
//
// What this file provides: the algebra of packed pairs of derivatives (the layout of
// math/chain_pair.hpp: P[pair_index(p, q)] = d^(p+q) f / dx_0^p dx_1^q, p + q <= K) for the
// constants, affine forms, their reciprocals and logarithms, products by Leibniz's rule and
// the composition of a function of one variable; the derivatives of D, of Stirling's error
// difference and of kappa(z) = (log1p(z) - z) / z; and V(r, c, h) = (r + c) log1p(h/r) - h
// with its derivatives, which the two masses at y = 0 are written with.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/binomial.hpp>  // detail_binomial::deviance_unit, D(1 + w) by its series
#include <laplace/distrib/core.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_negbin {

using math::detail_chain_pair::binom;
using math::detail_chain_pair::pair_count;
using math::detail_chain_pair::pair_index;

// ------------------------------------------------------------------ small scalar helpers

LAPLACE_HD_FORCEINLINE double factorial_of(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

// the switch from a series to a direct form, on a ratio of two arguments
inline constexpr double series_cut = 0.125;

// psi1(x) = x - log1p(x) = sum_(n >= 2) (-1)^n x^n / n, x > -1
LAPLACE_HD_FORCEINLINE double psi1(double x) {
  if (Kokkos::fabs(x) >= series_cut) return x - Kokkos::log1p(x);
  double p = x * x, s = 0.0;  // (-1)^n x^n at n = 2
  for (int n = 2; n <= 60; ++n, p *= -x) {
    const double t = p / n;
    s += t;
    if (Kokkos::fabs(t) <= 0x1p-56 * Kokkos::fabs(s)) break;
  }
  return s;
}

// kappa(z) = (log1p(z) - z) / z = -z/2 + z^2/3 - ..., z > -1: the series below 1/2, above it
// the recursion z kappa^(q) + q kappa^(q-1) = d^q [log1p(z) - z], which divides by z only
// where z is not small. Fills out[0..K].
template <int K>
LAPLACE_HD void kappa_derivs(double z, double* out) {
  if (Kokkos::fabs(z) < 0.5) {
    // the k-th derivative of the term (-1)^n z^n / (n + 1) is
    // (-1)^n n!/(n-k)! z^(n-k) / (n + 1); one running power and one running falling factorial
    // per order
    for (int k = 0; k <= K; ++k) {
      double s = 0.0;
      const int n0k = k > 1 ? k : 1;
      double p = (k == 0) ? z : 1.0;  // z^(n - k), from n = max(k, 1)
      double fall = 1.0;         // n (n-1) ... (n - k + 1)
      const int n0 = n0k;
      for (int i = 0; i < k; ++i) fall *= (n0 - i);
      if (k == 0) fall = 1.0;
      for (int n = n0; n <= 200; ++n) {
        const double t = ((n % 2) ? -1.0 : 1.0) * fall * p / (n + 1);
        s += t;
        if (n > k + 1 && Kokkos::fabs(t) <= 0x1p-56 * Kokkos::fabs(s)) break;
        p *= z;
        fall *= static_cast<double>(n + 1) / (n + 1 - k);
      }
      out[k] = s;
    }
    return;
  }
  out[0] = Kokkos::log1p(z) / z - 1.0;
  double prev = out[0];
  for (int q = 1; q <= K; ++q) {
    // d^q [log1p(z) - z] = (-1)^(q-1) (q-1)! / (1+z)^q, minus 1 at q = 1
    double p = 1.0;
    for (int i = 0; i < q; ++i) p *= (1.0 + z);
    double d = ((q % 2) ? 1.0 : -1.0) * factorial_of(q - 1) / p;
    if (q == 1) d -= 1.0;
    out[q] = (d - q * prev) / z;
    prev = out[q];
  }
}

// the derivatives of D(1 + w) = (1 + w) log1p(w) - w in w: D' = log1p(w), D'' = 1/(1 + w),
// D^(k) = (-1)^k (k-2)! / (1 + w)^(k-1) for k >= 2
template <int K>
LAPLACE_HD void deviance_derivs(double w, double* out) {
  out[0] = detail_binomial::deviance_unit(w);
  if constexpr (K >= 1) out[1] = Kokkos::log1p(w);
  double p = 1.0;  // (1 + w)^(k - 1)
  for (int k = 2; k <= K; ++k) {
    p *= (1.0 + w);
    out[k] = ((k % 2) ? -1.0 : 1.0) * factorial_of(k - 2) / p;
  }
}

// psi1(z) = z - log1p(z) and its derivatives, z > -1: psi1'(z) = z / (1 + z) (exact in z, and
// of the size of z near 0, where a difference 1 - 1/(1+z) would not be) and
// psi1^(k)(z) = (-1)^k (k-1)! / (1 + z)^k for k >= 2. The deviance at a count is
// bd0(a, b) = a psi1((b - a) / a), so these are its outer derivatives. Fills out[0..K].
template <int K>
LAPLACE_HD void psi1_derivs(double z, double* out) {
  out[0] = psi1(z);
  if constexpr (K >= 1) out[1] = z / (1.0 + z);
  double p = 1.0 + z;  // (1 + z)^k
  for (int k = 2; k <= K; ++k) {
    p *= (1.0 + z);
    out[k] = ((k % 2) ? -1.0 : 1.0) * factorial_of(k - 1) / p;
  }
}

// psi(v) = D(1 + v) / v = sum_(j >= 1) (-1)^(j+1) v^j / (j (j + 1)), v > -1: the series below
// |v| = 1/2, above it the recursion v psi^(k) + k psi^(k-1) = D^(k)(1 + v). A deviance divided
// by its argument appears wherever bd0 is taken at a b that carries a factor the parameter
// cannot hold: bd0(r, b2) = -b1 w1 psi(-theta w1) for negbin1, which has no 1/theta in it and
// is a product, so nothing cancels where y is near mu. Fills out[0..K].
template <int K>
LAPLACE_HD void dev_ratio_derivs(double v, double* out) {
  if (Kokkos::fabs(v) < 0.5) {
    for (int k = 0; k <= K; ++k) {
      double s = 0.0;
      const int j0 = k > 1 ? k : 1;
      double p = (k == 0) ? v : 1.0;  // v^(j - k)
      double fall = 1.0;              // j (j-1) ... (j - k + 1)
      for (int i = 0; i < k; ++i) fall *= (j0 - i);
      for (int j = j0; j <= 200; ++j) {
        const double t = ((j % 2) ? 1.0 : -1.0) * fall * p / (static_cast<double>(j) * (j + 1));
        s += t;
        if (j > k + 1 && Kokkos::fabs(t) <= 0x1p-56 * Kokkos::fabs(s)) break;
        p *= v;
        fall *= static_cast<double>(j + 1) / (j + 1 - k);
      }
      out[k] = s;
    }
    return;
  }
  double d[K + 2];
  deviance_derivs<(K + 1 > 6 ? 6 : K + 1)>(v, d);
  out[0] = d[0] / v;
  for (int k = 1; k <= K; ++k) out[k] = (d[k] - k * out[k - 1]) / v;
}

// delta^(p)(a), p = 0..6 chosen at run time
LAPLACE_HD_FORCEINLINE double stirling_dp(int p, double a) {
  switch (p) {
    case 0: return math::stirling_error(a);
    case 1: return math::stirling_error_d1(a);
    case 2: return math::stirling_error_d2(a);
    case 3: return math::stirling_error_d3(a);
    case 4: return math::stirling_error_d4(a);
    case 5: return math::stirling_error_d5(a);
    default: return math::stirling_error_d6(a);
  }
}

// ------------------------------------------------------------- packed pairs of derivatives
//
// P[pair_index(p, q)] = d^(p+q) f / dx_0^p dx_1^q for p + q <= K, the layout of chain_pair.hpp.

template <int K>
LAPLACE_HD_FORCEINLINE void pair_zero(double* P) {
  for (int i = 0; i < pair_count(K); ++i) P[i] = 0.0;
}

// an affine form of value v with gradient (a0, a1)
template <int K>
LAPLACE_HD_FORCEINLINE void pair_affine(double v, double a0, double a1, double* P) {
  pair_zero<K>(P);
  P[pair_index(0, 0)] = v;
  if constexpr (K >= 1) {
    P[pair_index(1, 0)] = a0;
    P[pair_index(0, 1)] = a1;
  }
}

// 1 / L with L an affine form of value L and gradient (a0, a1)
template <int K>
LAPLACE_HD void pair_recip_affine(double L, double a0, double a1, double* P) {
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      double t = factorial_of(p + q) / L;
      for (int i = 0; i < p; ++i) t *= a0 / L;
      for (int i = 0; i < q; ++i) t *= a1 / L;
      P[pair_index(p, q)] = ((p + q) % 2 ? -1.0 : 1.0) * t;
    }
}

// log L with L an affine form of value L and gradient (a0, a1)
template <int K>
LAPLACE_HD void pair_log_affine(double L, double a0, double a1, double* P) {
  P[pair_index(0, 0)] = math::log(L);
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      if (p + q == 0) continue;
      double t = factorial_of(p + q - 1);
      for (int i = 0; i < p; ++i) t *= a0 / L;
      for (int i = 0; i < q; ++i) t *= a1 / L;
      P[pair_index(p, q)] = ((p + q) % 2 ? 1.0 : -1.0) * t;  // (p+q-1)! a0^p a1^q / L^(p+q)
    }
}

// C = A B by Leibniz's rule; C must not alias A or B
template <int K>
LAPLACE_HD void pair_mul(const double* A, const double* B, double* C) {
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      double s = 0.0;
      for (int i = 0; i <= p; ++i)
        for (int j = 0; j <= q; ++j)
          s += binom(p, i) * binom(q, j) * A[pair_index(i, j)] * B[pair_index(p - i, q - j)];
      C[pair_index(p, q)] = s;
    }
}

// C += a A
template <int K>
LAPLACE_HD_FORCEINLINE void pair_axpy(double a, const double* A, double* C) {
  for (int i = 0; i < pair_count(K); ++i) C[i] += a * A[i];
}

// theta / (theta + mu): d^(p+q) = (-1)^(p+q) (p+q-1)! (p theta - q mu) / L^(p+q+1), one exact
// numerator, so the pure derivatives in either parameter are single terms. mu / (theta + mu) is
// its complement, hence the same with the opposite sign away from the value.
template <int K>
LAPLACE_HD void pair_share(double mu, double th, bool first, double* P) {
  const double L = th + mu;
  P[pair_index(0, 0)] = (first ? mu : th) / L;
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      const int n = p + q;
      if (n == 0) continue;
      double d = factorial_of(n - 1) * (p * th - q * mu);
      for (int i = 0; i <= n; ++i) d /= L;
      P[pair_index(p, q)] = ((n % 2) ? -1.0 : 1.0) * (first ? -d : d);
    }
}

// log(mu / (theta + mu)) = -log1p(theta / mu): the derivatives in mu alone are
// (-1)^(p-1) (p-1)! mu^-p E_p with E_p = 1 - (mu / L)^p built by terms of one sign, so the
// difference mu^-p - L^-p is never formed; with one derivative in theta or more only -log L
// is left.
template <int K>
LAPLACE_HD void pair_log_share(double mu, double th, double* P) {
  const double L = th + mu;
  P[pair_index(0, 0)] = -Kokkos::log1p(th / mu);
  const double rho = mu / L;
  double E = 0.0, rp = 1.0, mp = 1.0;
  for (int p = 1; p <= K; ++p) {
    E += (th / L) * rp;  // E_p = 1 - rho^p
    rp *= rho;
    mp *= mu;
    P[pair_index(p, 0)] = ((p % 2) ? 1.0 : -1.0) * factorial_of(p - 1) * E / mp;
  }
  for (int q = 1; q <= K; ++q)
    for (int p = 0; p + q <= K; ++p) {
      const int n = p + q;
      double d = factorial_of(n - 1);
      for (int i = 0; i < n; ++i) d /= L;
      P[pair_index(p, q)] = ((n % 2) ? -1.0 : 1.0) * d;
    }
}

template <int K, int N>
LAPLACE_HD void compose_orders(const double* G, const double* u0, const double* u1, double* C) {
  if constexpr (N <= K) {
    double out[N + 1];
    math::chain_rule_pair_dk<N>(G, u0, u1, out);
    for (int b = 0; b <= N; ++b) C[pair_index(N - b, b)] = out[b];
    compose_orders<K, N + 1>(G, u0, u1, C);
  }
}

// C = g(u(x_0, x_1)) from the derivatives g[0..K] of g in one variable and the packed
// derivatives of u; C must not alias u
template <int K>
LAPLACE_HD void pair_compose(const double* g, const double* u, double* C) {
  double G[pair_count(K)], zero[pair_count(K)];
  pair_zero<K>(G);
  pair_zero<K>(zero);
  for (int i = 0; i <= K; ++i) G[pair_index(i, 0)] = g[i];
  C[pair_index(0, 0)] = g[0];
  compose_orders<K, 1>(G, u, zero, C);
}

// ----------------------------------------------------------------- V, for the mass at y = 0
//
// V(r, c, h) = (r + c) log1p(h / r) - h. Its derivatives in r cancel below x = h / r = 1/8
// (the two terms of (r + c) Lambda_p + p Lambda_(p-1), Lambda = log1p(h/r), are both of size
// p! x r^(1-p) while V_p is of size x^2), and are taken there from the series
//
//   V_p = (-1)^p r^-p sum_(n >= 1) (-1)^(n-1) x^n / n [ (n-1)_p r + c (n)_p ],
//   (j)_p = j (j+1) ... (j+p-1),  (0)_p = 0 for p >= 1,
//
// whose terms decay by the factor x (n + p) / n. Above the cut the direct form is used,
// through E_p = 1 - (r/(r+h))^p built by a recursion of terms of one sign, so that
// (r + h)^-p - r^-p is never formed as a difference. The value takes the series of
// psi1(x) = x - log1p(x) below the cut, V_0 = c x - (r + c) psi1(x), and the direct form
// above it, where psi1 is of the size of x and c x - (r + c) psi1(x) would cancel.
// The forms are derived and checked against SymPy in tools/reference/distrib_negbin.py.
template <int K>
LAPLACE_HD void v_derivs(double r, double c, double h, double* v) {
  const double x = h / r;
  v[0] = Kokkos::fabs(x) < series_cut ? c * x - (r + c) * psi1(x) : (r + c) * Kokkos::log1p(x) - h;
  if (h == 0.0) {
    for (int p = 1; p <= K; ++p) v[p] = 0.0;
    return;
  }
  if (Kokkos::fabs(x) < series_cut) {
    double acc[K + 1] = {}, big[K + 1] = {};
    double xn = x;
    for (int n = 1; n <= 80; ++n, xn *= x) {
      double pa = 1.0, pb = 1.0;  // (n-1)_p and (n)_p, built up in p
      bool done = true;
      const double sn = (n % 2) ? 1.0 : -1.0;
      for (int p = 1; p <= K; ++p) {
        pa *= static_cast<double>(n - 1 + p - 1);
        pb *= static_cast<double>(n + p - 1);
        const double t = sn * xn / n * (pa * r + c * pb);
        acc[p] += t;
        const double a = Kokkos::fabs(t);
        if (a > big[p]) big[p] = a;
        if (a > 0x1p-56 * big[p]) done = false;
      }
      if (n > 2 && done) break;
    }
    double rp = 1.0;
    for (int p = 1; p <= K; ++p) {
      rp *= r;
      v[p] = ((p % 2) ? -1.0 : 1.0) * acc[p] / rp;
    }
    return;
  }
  if constexpr (K >= 1) {
    double E[K + 1];
    const double rho = r / (r + h), E1 = h / (r + h);
    E[0] = 0.0;
    double rp = 1.0;
    for (int p = 1; p <= K; ++p, rp *= rho) E[p] = E[p - 1] + E1 * rp;
    v[1] = -(r + c) * E[1] / r + Kokkos::log1p(x);
    double r1p = 1.0;  // r^(p-1)
    for (int p = 2; p <= K; ++p) {
      r1p *= r;
      v[p] = ((p % 2) ? -1.0 : 1.0) * factorial_of(p - 2) / r1p *
             ((1.0 + c / r) * (p - 1) * E[p] - p * E[p - 1]);
    }
  }
}

// d^(p+q) [(r + c) log1p(h / r)] / dr^p dh^q for q >= 1: the two terms of Leibniz's rule
// combine into the one numerator r (q - 1) + c (p + q - 1) - p h, exact in its inputs.
LAPLACE_HD_FORCEINLINE double w_mixed(double r, double c, double h, int p, int q) {
  const double s = r + h;
  double sp = 1.0;  // s^(p+q)
  for (int i = 0; i < p + q; ++i) sp *= s;
  if (p == 0) return (r + c) * ((q % 2) ? 1.0 : -1.0) * factorial_of(q - 1) / sp;
  const double num = r * (q - 1) + c * (p + q - 1) - p * h;
  return (((p + q) % 2) ? 1.0 : -1.0) * factorial_of(p + q - 2) * num / sp;
}

// ------------------------------------------------- Stirling's error difference for negbin1
//
// Delta_delta(s) = delta(y + r) - delta(r) as a function of s = 1 / r, needed because
// negbin1's shape r = mu / theta is not a regular variable at its Poisson limit. Above
// r = 10 both arguments are in the range where stirling.hpp uses its asymptotic series
// delta(a) = sum_j sigma_j a^(1-2j), so delta(r) = sum_j sigma_j s^(2j-1) and
// delta(y + r) = sum_j sigma_j t^(2j-1) with t = 1 / (y + r) = s / (1 + y s), and both are
// explicit functions of s. Fills out[0..K] with the s-derivatives.
template <int K>
LAPLACE_HD void stirling_diff_s(double s, double y, double* out) {
  double hs[K + 1], ts[K + 1], tt[K + 1];
  for (int k = 0; k <= K; ++k) hs[k] = 0.0, ts[k] = 0.0, tt[k] = 0.0;
  const double t = s / (1.0 + y * s);
  // delta(r) = sum_j sigma_j s^(2j-1) and the same series at t, both differentiated in place
  for (int j = 1; j <= 20; ++j) {
    const double sig = math::detail_stirling::asym_coef(j - 1);
    const int e = 2 * j - 1;
    double keep = 0.0;
    for (int k = 0; k <= K; ++k) {
      if (k > e) continue;
      double fall = 1.0;
      for (int i = 0; i < k; ++i) fall *= (e - i);
      double ps = 1.0, pt = 1.0;
      for (int i = 0; i < e - k; ++i) ps *= s, pt *= t;
      ts[k] += sig * fall * ps;
      hs[k] += sig * fall * pt;
      keep = Kokkos::fmax(keep, Kokkos::fabs(sig * fall * ps));
    }
    if (j > 2 && keep <= 0x1p-54 * Kokkos::fabs(ts[0])) break;
  }
  // the chain rule for the term at t = s / (1 + y s): t^(m) = (-1)^(m-1) m! y^(m-1) / (1+ys)^(m+1)
  double tin[K + 1];
  const double d = 1.0 + y * s;
  double pd = d * d, py = 1.0;  // (1+ys)^(m+1), y^(m-1)
  for (int m = 1; m <= K; ++m) {
    tin[m] = ((m % 2) ? 1.0 : -1.0) * factorial_of(m) * py / pd;
    py *= y;
    pd *= d;
  }
  tt[0] = hs[0];
  if constexpr (K >= 1) tt[1] = math::chain_rule_d1(hs, tin);
  if constexpr (K >= 2) tt[2] = math::chain_rule_d2(hs, tin);
  if constexpr (K >= 3) tt[3] = math::chain_rule_d3(hs, tin);
  if constexpr (K >= 4) tt[4] = math::chain_rule_d4(hs, tin);
  if constexpr (K >= 5) tt[5] = math::chain_rule_d5(hs, tin);
  if constexpr (K >= 6) tt[6] = math::chain_rule_d6(hs, tin);
  for (int k = 0; k <= K; ++k) out[k] = tt[k] - ts[k];
}

// ------------------------------------------------------------------ the two log-masses
//
// Both fill the packed partials of l in (x_0, x_1) = (mu, theta) to total order K. Every
// inner map is an affine form of (mu, theta), a reciprocal of one, or a product of those, so
// each is regular where its family's Poisson limit is.

// negbin2: (r, m) = (theta, mu). b1 = (y + theta) mu / (theta + mu) from the closed share
// tensor, Delta = theta (y - mu) / (theta + mu), and the two deviances are
//   bd0(y, b1) = y psi1(z),      z = (b1 - y) / y = -Delta / y,
//   bd0(theta, b2) = theta psi1(z2),  z2 = Delta / theta = (y - mu) / (theta + mu),
// the second a product of theta and one composition, with b2 never formed.
template <int K>
LAPLACE_HD void negbin2_partials(double mu, double th, double y, double* G) {
  constexpr int N = pair_count(K);
  if (!(y > 0.0)) {  // l = theta log(theta / (theta + mu)) = -(V(theta, 0, mu) + mu)
    double v[K + 1];
    v_derivs<K>(th, 0.0, mu, v);
    pair_zero<K>(G);
    for (int q = 0; q <= K; ++q)
      for (int p = 1; p + q <= K; ++p) G[pair_index(p, q)] = -w_mixed(th, 0.0, mu, q, p);
    for (int q = 1; q <= K; ++q) G[pair_index(0, q)] = -v[q];
    // the value directly: -(V(theta, 0, mu) + mu) would subtract mu from a quantity of its own
    // size (measured: 239 ulp at mu = 40, theta = 0.01), while one log1p cancels nothing
    G[pair_index(0, 0)] = -th * Kokkos::log1p(mu / th);
    return;
  }
  const double L1 = y + th, L2 = th + mu;
  double a[N], b[N], c[N], T[N], Dv[K + 1];
  pair_zero<K>(G);
  // the part that depends on theta alone: (1/2) log(theta / (y + theta)) - (1/2) log(2 pi y)
  // + delta(y + theta) - delta(y) - delta(theta), the first written as -(1/2) log1p(y/theta)
  // and its derivatives through E_q = 1 - (theta / (y + theta))^q
  G[pair_index(0, 0)] = -0.5 * Kokkos::log1p(y / th) - 0.5 * math::log(6.283185307179586 * y) +
                        math::stirling_error(y + th) - math::stirling_error(y) - math::stirling_error(th);
  {
    const double rho = th / L1;
    double E = 0.0, rp = 1.0, tq = 1.0;
    for (int q = 1; q <= K; ++q) {
      E += (y / L1) * rp;  // E_q = 1 - rho^q by terms of one sign
      rp *= rho;
      tq *= th;
      G[pair_index(0, q)] = 0.5 * ((q % 2) ? 1.0 : -1.0) * factorial_of(q - 1) * E / tq +
                            stirling_dp(q, y + th) - stirling_dp(q, th);
    }
  }
  // - bd0(y, b1) = - y psi1(z), z = (b1 - y) / y = -Delta / y = -theta (y - mu) / (y (theta + mu)),
  // with b1 = (y + theta) mu / (theta + mu) from the closed share tensor (its pure derivatives in
  // either parameter are then single terms). Near the mode, where |z| is small, the composition
  // of psi1 with z is the accurate route; away from it the two terms of
  // d^a bd0 = d^a b1 - y d^a log b1 are of the size of the result and the logarithm's own
  // tensor, which is closed here, avoids the alternating sum of a composition.
  const double zval = -th * (y - mu) / (y * L2);
  pair_affine<K>(L1, 0.0, 1.0, a);
  pair_share<K>(mu, th, true, b);
  pair_mul<K>(a, b, T);  // b1
  if (Kokkos::fabs(zval) < 0.5) {
    for (int i = 0; i < N; ++i) c[i] = T[i] / y;  // z = b1 / y - 1
    c[pair_index(0, 0)] = zval;
    psi1_derivs<K>(zval, Dv);
    pair_compose<K>(Dv, c, a);
    pair_axpy<K>(-y, a, G);
  } else {
    pair_log_share<K>(mu, th, b);                     // log(mu / (theta + mu))
    pair_log_affine<K>(L1, 0.0, 1.0, c);              // log(y + theta)
    pair_axpy<K>(1.0, c, b);                          // log b1
    pair_axpy<K>(-1.0, T, G);
    pair_axpy<K>(y, b, G);
    G[pair_index(0, 0)] += T[pair_index(0, 0)] - y * b[pair_index(0, 0)] - y * psi1(zval);
  }
  // - bd0(theta, b2) = - theta psi1(z2), z2 = (b2 - theta) / theta = (y - mu) / (theta + mu):
  // a product of theta and a composition, with no 1/theta and no large factor
  pair_affine<K>(y - mu, -1.0, 0.0, a);
  pair_recip_affine<K>(L2, 1.0, 1.0, b);
  pair_mul<K>(a, b, c);  // z2
  psi1_derivs<K>(c[pair_index(0, 0)], Dv);
  pair_compose<K>(Dv, c, a);
  pair_affine<K>(th, 0.0, 1.0, b);
  pair_mul<K>(a, b, c);
  pair_axpy<K>(-1.0, c, G);
}

// negbin1: (r, m) = (mu / theta, mu). Then (y + r) / r = 1 + u with u = y theta / mu,
// b1 = (y theta + mu) / (1 + theta) = y - (y - mu) / (1 + theta) (whose partials are closed and
// of one term each), w1 = (y - mu) / (y theta + mu), and the deviance at the shape is
//
//   bd0(r, b2) = -b1 w1 psi(-theta w1),   psi(v) = D(1 + v) / v,
//
// because b2 = b1 / theta and w2 = -theta w1 exactly, so the factor 1/theta cancels and what is
// left is a product. Every one of those is regular at theta = 0.
template <int K>
LAPLACE_HD void negbin1_partials(double mu, double th, double y, double* G) {
  constexpr int N = pair_count(K);
  double a[N], b[N], c[N], T[N], U[N], g[K + 1];
  pair_zero<K>(G);
  if (!(y > 0.0)) {  // l = -(mu / theta) log1p(theta) = -mu (1 + kappa(theta))
    double kt[K + 1];
    kappa_derivs<K>(th, kt);
    pair_affine<K>(mu, 1.0, 0.0, a);
    pair_zero<K>(b);
    for (int q = 0; q <= K; ++q) b[pair_index(0, q)] = kt[q];
    b[pair_index(0, 0)] += 1.0;
    pair_mul<K>(a, b, c);
    pair_axpy<K>(-1.0, c, G);
    return;
  }
  const double M1 = y * th + mu;
  // u = y theta / mu and its packed partials
  pair_affine<K>(y * th, 0.0, y, a);
  pair_recip_affine<K>(mu, 1.0, 0.0, b);
  pair_mul<K>(a, b, U);
  const double u = U[pair_index(0, 0)];
  // - (1/2) log1p(u) - (1/2) log(2 pi y) + Delta_delta - delta(y)
  g[0] = Kokkos::log1p(u);
  {
    double p = 1.0;
    for (int k = 1; k <= K; ++k) {
      p *= (1.0 + u);
      g[k] = ((k % 2) ? 1.0 : -1.0) * factorial_of(k - 1) / p;
    }
  }
  pair_compose<K>(g, U, a);
  pair_axpy<K>(-0.5, a, G);
  G[pair_index(0, 0)] += -0.5 * math::log(6.283185307179586 * y) - math::stirling_error(y);
  {
    const double r = mu / th;
    if (r >= 10.0) {  // both Stirling errors are in the range of their asymptotic series
      pair_affine<K>(th, 0.0, 1.0, a);
      pair_recip_affine<K>(mu, 1.0, 0.0, b);
      pair_mul<K>(a, b, T);  // s = theta / mu
      stirling_diff_s<K>(T[pair_index(0, 0)], y, g);
      pair_compose<K>(g, T, a);
    } else {  // theta is of the size of mu: the shape is a regular variable here
      pair_affine<K>(mu, 1.0, 0.0, a);
      pair_recip_affine<K>(th, 0.0, 1.0, b);
      pair_mul<K>(a, b, T);  // r = mu / theta
      for (int k = 0; k <= K; ++k) g[k] = stirling_dp(k, y + r) - stirling_dp(k, r);
      pair_compose<K>(g, T, a);
    }
    pair_axpy<K>(1.0, a, G);
  }
  // b1 = (y theta + mu) / (1 + theta) = y - (y - mu) / (1 + theta), whose partials are closed and
  // of one term each, and w1 = (y - mu) / (y theta + mu)
  double B1[N], W1[N];
  pair_zero<K>(B1);
  {
    double p = 1.0 + th;  // (1 + theta)^(q+1)
    for (int q = 0; q <= K; ++q) {
      const double s = ((q % 2) ? -1.0 : 1.0) * factorial_of(q) / p;  // (-1)^q q! / (1+theta)^(q+1)
      B1[pair_index(0, q)] = -(y - mu) * s;
      if (1 + q <= K) B1[pair_index(1, q)] = s;
      p *= (1.0 + th);
    }
    B1[pair_index(0, 0)] += y;
  }
  pair_affine<K>(y - mu, -1.0, 0.0, a);
  pair_recip_affine<K>(M1, 1.0, y, b);
  pair_mul<K>(a, b, W1);
  // - bd0(y, b1) = - y psi1(z), z = (b1 - y) / y = -(y - mu) / (y (1 + theta)): the composition
  // near the mode, the two closed tensors d^a b1 - y d^a log b1 away from it
  // (log b1 = log(y theta + mu) - log1p(theta), both of an affine form)
  const double zval = -(y - mu) / (y * (1.0 + th));
  if (Kokkos::fabs(zval) < 0.5) {
    for (int i = 0; i < N; ++i) c[i] = B1[i] / y;
    c[pair_index(0, 0)] = zval;
    psi1_derivs<K>(zval, g);
    pair_compose<K>(g, c, a);
    pair_axpy<K>(-y, a, G);
  } else {
    pair_log_affine<K>(M1, 1.0, y, b);
    pair_log_affine<K>(1.0 + th, 0.0, 1.0, c);
    pair_axpy<K>(-1.0, c, b);  // log b1
    pair_axpy<K>(-1.0, B1, G);
    pair_axpy<K>(y, b, G);
    G[pair_index(0, 0)] += B1[pair_index(0, 0)] - y * b[pair_index(0, 0)] - y * psi1(zval);
  }
  // - bd0(r, b2) = b1 w1 psi(-theta w1), psi(v) = D(1 + v) / v: b2 = b1 / theta and
  // w2 = -theta w1 exactly, so the factor 1/theta cancels and what is left is a product
  pair_affine<K>(-th, 0.0, -1.0, a);
  pair_mul<K>(a, W1, T);  // v = -theta w1
  dev_ratio_derivs<K>(T[pair_index(0, 0)], g);
  pair_compose<K>(g, T, a);
  pair_mul<K>(B1, W1, c);
  pair_mul<K>(c, a, b);
  pair_axpy<K>(1.0, b, G);
}

}  // namespace detail_negbin

}  // namespace laplace::distrib
