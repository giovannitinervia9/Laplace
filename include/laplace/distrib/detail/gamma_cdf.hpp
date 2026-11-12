// Derivatives of log F and log(1 - F) for the distributions whose cdf is a
// regularized incomplete gamma function (laplace::distrib, internal).
//
// S(a, x) is P(a, x) or Q(a, x). laplace::math::log_gamma_{p,q}_mixed_dN give
// d^N log S / da^(N-j) dx^j (the Poisson cdf, a = k + 1, x = mu);
// laplace::math::log_gamma_{p,q}_ratio_mixed_dN give R_ij = d^(i+j) log S(a, a c)
// / da^i dc^j, the derivatives in a at fixed ratio c = x/a.
//
// gamma1 (mu, phi): F(y) = P(a, a c) with a = 1/phi and c = y/mu: a depends on
// phi alone and c on mu alone, so by Faa di Bruno's formula for each inner map
//   d^K log S / dmu^r dphi^s = sum_(i <= s, j <= r) R_ij B_(s,i)(a', a'', ...) B_(r,j)(c', c'', ...),
// with a^(n)(phi) = (-1)^n n!/phi^(n+1), c^(n)(mu) = (-1)^n n! y/mu^(n+1) and
// B_(n,k) the partial Bell polynomials (laplace::math::bell_partial;
// B_(0,0) = 1, B_(n,0) = 0 for n >= 1). Every B_(s,i) has the sign (-1)^s, so
// the sum has no cancellation beyond that of the R_ij themselves. (Until
// 2026-11-12 the composition went over the partials in (a, x), where along phi
// with x/a fixed d/da and c d/dx are large and of opposite sign: 4e7 ulp lost
// at phi = 0.02, y = mu.)
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/incomplete_gamma_mixed.hpp>
#include <laplace/math/incomplete_gamma_ratio.hpp>

namespace laplace::distrib::detail_gamma_cdf {

// d^N log S / da^(N-j) dx^j, j = 0..N, into v[0..N]
template <int N>
LAPLACE_HD_FORCEINLINE void mixed_order(bool lower, double a, double x, double* v) {
  namespace m = laplace::math;
  if constexpr (N == 1) { const auto r = lower ? m::log_gamma_p_mixed_d1(a, x) : m::log_gamma_q_mixed_d1(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
  if constexpr (N == 2) { const auto r = lower ? m::log_gamma_p_mixed_d2(a, x) : m::log_gamma_q_mixed_d2(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
  if constexpr (N == 3) { const auto r = lower ? m::log_gamma_p_mixed_d3(a, x) : m::log_gamma_q_mixed_d3(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
  if constexpr (N == 4) { const auto r = lower ? m::log_gamma_p_mixed_d4(a, x) : m::log_gamma_q_mixed_d4(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
  if constexpr (N == 5) { const auto r = lower ? m::log_gamma_p_mixed_d5(a, x) : m::log_gamma_q_mixed_d5(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
  if constexpr (N == 6) { const auto r = lower ? m::log_gamma_p_mixed_d6(a, x) : m::log_gamma_q_mixed_d6(a, x); for (int j = 0; j <= N; ++j) v[j] = r.v[j]; }
}

// R_ij = d^N log S(a, a c) / da^(N-j) dc^j, j = 0..N, into R[N - j][j]
template <int N>
LAPLACE_HD_FORCEINLINE void ratio_order(bool lower, double a, double c, double (*R)[7]) {
  namespace m = laplace::math;
  if constexpr (N == 1) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d1(a, c) : m::log_gamma_q_ratio_mixed_d1(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
  if constexpr (N == 2) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d2(a, c) : m::log_gamma_q_ratio_mixed_d2(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
  if constexpr (N == 3) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d3(a, c) : m::log_gamma_q_ratio_mixed_d3(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
  if constexpr (N == 4) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d4(a, c) : m::log_gamma_q_ratio_mixed_d4(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
  if constexpr (N == 5) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d5(a, c) : m::log_gamma_q_ratio_mixed_d5(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
  if constexpr (N == 6) { const auto r = lower ? m::log_gamma_p_ratio_mixed_d6(a, c) : m::log_gamma_q_ratio_mixed_d6(a, c); for (int j = 0; j <= N; ++j) R[N - j][j] = r.v[j]; }
}

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// B_(n,k)(g_1, ...) with B_(0,0) = 1 and B_(n,0) = 0 for n >= 1
LAPLACE_HD_FORCEINLINE double bell0(int n, int k, const double* g) {
  if (k == 0) return n == 0 ? 1.0 : 0.0;
  return laplace::math::bell_partial(n, k, g);
}

// d^K log S(1/phi, y/(mu phi)) / dmu^(K-b) dphi^b, b = 0..K (b in phi at position b)
template <int K>
LAPLACE_HD sym_tensor<2, K> gamma1_dk(bool lower, double y, double mu, double phi) {
  const double a = 1.0 / phi, c = y / mu;
  double R[7][7] = {};  // R[i][j] = d^(i+j) log S(a, a c) / da^i dc^j, i + j <= K
  ratio_order<1>(lower, a, c, R);
  if constexpr (K >= 2) ratio_order<2>(lower, a, c, R);
  if constexpr (K >= 3) ratio_order<3>(lower, a, c, R);
  if constexpr (K >= 4) ratio_order<4>(lower, a, c, R);
  if constexpr (K >= 5) ratio_order<5>(lower, a, c, R);
  if constexpr (K >= 6) ratio_order<6>(lower, a, c, R);
  // the derivatives of the inner maps, g[n] for n = 1..K
  double ga[7], gc[7];
  {
    double tp = -a * a, tm = -c / mu;  // a' = -1/phi^2, c' = -y/mu^2
    for (int n = 1; n <= K; ++n) {
      ga[n] = tp;
      gc[n] = tm;
      tp *= -(n + 1) * a;
      tm *= -(n + 1) / mu;
    }
  }
  sym_tensor<2, K> t;
  for (int b = 0; b <= K; ++b) {
    const int r = K - b;
    double acc = 0.0;
    for (int i = 0; i <= b; ++i) {
      const double ba = bell0(b, i, ga);
      if (ba == 0.0) continue;
      for (int j = 0; j <= r; ++j) {
        if (i + j == 0) continue;
        const double bc = bell0(r, j, gc);
        if (bc == 0.0) continue;
        acc += R[i][j] * (ba * bc);
      }
    }
    t[b] = acc;
  }
  return t;
}

}  // namespace laplace::distrib::detail_gamma_cdf
