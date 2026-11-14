// lognormal2_distrib (laplace::distrib): the lognormal distribution in the mean m > 0 and the
// variance v > 0 of Y, the same law as lognormal1_distrib written through
//
//   s = log(1 + v/m^2),     mu = log m - s/2,
//
// where s is the variance of log Y and mu its mean (so lognormal1_distrib(mu, s) is the same law).
// s is formed with log1p(v/m^2), never as the log of a sum.
//
// Parameter derivatives. The log-density is a composition H(m, v) = G(mu(m, v), s(m, v)) of
// lognormal1_distrib's log-density G(mu, s; y) with the two-into-two map (m, v) -> (mu, s), in which
// both inner variables depend on both parameters. The composition is
// laplace::math::chain_rule_pair_dk<K> (include/laplace/math/chain_pair.hpp): it takes the partial
// derivatives of G in (mu, s), which are lognormal1_distrib's own derivatives at (mu, s), and the
// partial derivatives of mu and s in (m, v) to order K, whose closed forms are in
// detail_lognormal2::{s,mu}_d1 ... _d6 (derived and checked with SymPy in
// tools/reference/distrib_lognormal.py). The derivatives in y are the same composition with the
// y-derivatives of G (lognormal1_distrib::logpdf_dy_dk) in place of its parameter derivatives.
//
// Expected information. The information is a tensor: I_new = J' I_old J, with J = d(mu, s)/d(m, v)
// and I_old = diag(1/s, 1/(2 s^2)) (gaussian2's diagonal in its variance s). Written out,
//
//   I_ab = sum_c I_c(s) d_a theta_c d_b theta_c,    theta = (mu, s), I_mu = 1/s, I_s = 1/(2 s^2),
//
// which has a nonzero off-diagonal entry (at small v/m^2 it is -3/(2 m v) to leading order). Its
// derivatives to order 4 are computed by composing each factor and multiplying: I_c(s) by the chain
// rule of chain_rule_pair_dk on the outer function 1/s or 1/(2 s^2) (whose s-derivatives are the
// closed forms (-1)^j j!/s^(j+1) and (-1)^j (j+1)!/(2 s^(j+2)), the same as gaussian2's expected
// information derivatives), the products d_a theta_c d_b theta_c by the same routine with the
// outer function u_0 u_1 (its partials are 1 at (1, 1), u_1 at (1, 0), u_0 at (0, 1) and zero above),
// and the sum over c. There is no separate Leibniz recursion: the product rule is the chain rule of
// the product function.
//
// Quantile, cdf, ccdf: lognormal1's at (mu, s), with the same identities.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/gaussian2.hpp>
#include <laplace/distrib/lognormal1.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>

namespace laplace::distrib {

namespace detail_lognormal2 {

using laplace::math::detail_chain_pair::pair_count;
using laplace::math::detail_chain_pair::pair_index;

// Largest packed size used here: orders 0 to 6 of a function of two variables.
inline constexpr int packed_max = pair_count(6);

// s(m, v) = log(1 + v/m^2) and mu(m, v) = log m - s/2 (their values)
LAPLACE_HD_FORCEINLINE double s_value(double m, double v) { return Kokkos::log1p(v / (m * m)); }
LAPLACE_HD_FORCEINLINE double mu_value(double m, double v) { return math::log(m) - 0.5 * s_value(m, v); }

// The partial derivatives of s in (m, v) of order N, the component q at position q (q derivatives in v,
// N - q in m). With S = m^2 + v; checked against SymPy's own derivative by distrib_lognormal.py (mode forms).
LAPLACE_HD_FORCEINLINE sym_tensor<2, 1> s_d1(double m, double v) {
  const double S = m * m + v;
  sym_tensor<2, 1> t;
  t[0] = -2.0 * v / (m * S);
  t[1] = 1.0 / S;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 2> s_d2(double m, double v) {
  const double S = m * m + v, S2 = S * S;
  sym_tensor<2, 2> t;
  t[0] = 2.0 * v * (3.0 * m * m + v) / (m * m * S2);
  t[1] = -2.0 * m / S2;
  t[2] = -1.0 / S2;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 3> s_d3(double m, double v) {
  const double S = m * m + v, S3 = S * S * S;
  sym_tensor<2, 3> t;
  t[0] = -4.0 * v * (6.0 * m * m * m * m + 3.0 * m * m * v + v * v) / (m * m * m * S3);
  t[1] = -2.0 * (v - 3.0 * m * m) / S3;
  t[2] = 4.0 * m / S3;
  t[3] = 2.0 / S3;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 4> s_d4(double m, double v) {
  const double m2 = m * m, S = m2 + v, S4 = S * S * S * S;
  sym_tensor<2, 4> t;
  t[0] = 12.0 * v * (10.0 * m2 * m2 * m2 + 5.0 * m2 * m2 * v + 4.0 * m2 * v * v + v * v * v) / (m2 * m2 * S4);
  t[1] = 24.0 * m * (v - m2) / S4;
  t[2] = 4.0 * (v - 5.0 * m2) / S4;
  t[3] = -12.0 * m / S4;
  t[4] = -6.0 / S4;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 5> s_d5(double m, double v) {
  const double m2 = m * m, S = m2 + v, S5 = S * S * S * S * S;
  sym_tensor<2, 5> t;
  t[0] = -48.0 * v * (15.0 * m2 * m2 * m2 * m2 + 5.0 * m2 * m2 * m2 * v + 10.0 * m2 * m2 * v * v + 5.0 * m2 * v * v * v +
                      v * v * v * v) /
         (m2 * m2 * m * S5);
  t[1] = 24.0 * (5.0 * m2 * m2 - 10.0 * m2 * v + v * v) / S5;
  t[2] = -24.0 * m * (3.0 * v - 5.0 * m2) / S5;
  t[3] = -12.0 * (v - 7.0 * m2) / S5;
  t[4] = 48.0 * m / S5;
  t[5] = 24.0 / S5;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 6> s_d6(double m, double v) {
  const double m2 = m * m, S = m2 + v, S2 = S * S, S6 = S2 * S2 * S2;
  sym_tensor<2, 6> t;
  t[0] = 240.0 * v *
         (21.0 * m2 * m2 * m2 * m2 * m2 + 21.0 * m2 * m2 * m2 * v * v + 15.0 * m2 * m2 * v * v * v +
          6.0 * m2 * v * v * v * v + v * v * v * v * v) /
         (m2 * m2 * m2 * S6);
  t[1] = -240.0 * m * (v - 3.0 * m2) * (3.0 * v - m2) / S6;
  t[2] = -24.0 * (35.0 * m2 * m2 - 42.0 * m2 * v + 3.0 * v * v) / S6;
  t[3] = 96.0 * m * (3.0 * v - 7.0 * m2) / S6;
  t[4] = 48.0 * (v - 9.0 * m2) / S6;
  t[5] = -240.0 * m / S6;
  t[6] = -120.0 / S6;
  return t;
}

// The partial derivatives of mu in (m, v) of order N (mu = log m - s/2, so mu_m = 1/m - s_m/2 and
// mu_v = -s_v/2 at the first order).
LAPLACE_HD_FORCEINLINE sym_tensor<2, 1> mu_d1(double m, double v) {
  const double S = m * m + v;
  sym_tensor<2, 1> t;
  t[0] = (m * m + 2.0 * v) / (m * S);
  t[1] = -1.0 / (2.0 * S);
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 2> mu_d2(double m, double v) {
  const double m2 = m * m, S = m2 + v, S2 = S * S;
  sym_tensor<2, 2> t;
  t[0] = -(m2 * m2 + 5.0 * m2 * v + 2.0 * v * v) / (m2 * S2);
  t[1] = m / S2;
  t[2] = 1.0 / (2.0 * S2);
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 3> mu_d3(double m, double v) {
  const double m2 = m * m, S = m2 + v, S3 = S * S * S;
  sym_tensor<2, 3> t;
  t[0] = 2.0 * (m2 * m2 * m2 + 9.0 * m2 * m2 * v + 6.0 * m2 * v * v + 2.0 * v * v * v) / (m * m2 * S3);
  t[1] = (v - 3.0 * m2) / S3;
  t[2] = -2.0 * m / S3;
  t[3] = -1.0 / S3;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 4> mu_d4(double m, double v) {
  const double m2 = m * m, S = m2 + v, S4 = S * S * S * S;
  sym_tensor<2, 4> t;
  t[0] = -6.0 * (m2 * m2 * m2 * m2 + 14.0 * m2 * m2 * m2 * v + 11.0 * m2 * m2 * v * v + 8.0 * m2 * v * v * v +
                 2.0 * v * v * v * v) /
         (m2 * m2 * S4);
  t[1] = -12.0 * m * (v - m2) / S4;
  t[2] = -2.0 * (v - 5.0 * m2) / S4;
  t[3] = 6.0 * m / S4;
  t[4] = 3.0 / S4;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 5> mu_d5(double m, double v) {
  const double m2 = m * m, S = m2 + v, S5 = S * S * S * S * S;
  sym_tensor<2, 5> t;
  t[0] = 24.0 *
         (m2 * m2 * m2 * m2 * m2 + 20.0 * m2 * m2 * m2 * m2 * v + 15.0 * m2 * m2 * m2 * v * v +
          20.0 * m2 * m2 * v * v * v + 10.0 * m2 * v * v * v * v + 2.0 * v * v * v * v * v) /
         (m2 * m2 * m * S5);
  t[1] = -12.0 * (5.0 * m2 * m2 - 10.0 * m2 * v + v * v) / S5;
  t[2] = 12.0 * m * (3.0 * v - 5.0 * m2) / S5;
  t[3] = 6.0 * (v - 7.0 * m2) / S5;
  t[4] = -24.0 * m / S5;
  t[5] = -12.0 / S5;
  return t;
}
LAPLACE_HD_FORCEINLINE sym_tensor<2, 6> mu_d6(double m, double v) {
  const double m2 = m * m, S = m2 + v, S2 = S * S, S6 = S2 * S2 * S2;
  sym_tensor<2, 6> t;
  t[0] = -120.0 *
         (m2 * m2 * m2 * m2 * m2 * m2 + 27.0 * m2 * m2 * m2 * m2 * m2 * v + 15.0 * m2 * m2 * m2 * m2 * v * v +
          41.0 * m2 * m2 * m2 * v * v * v + 30.0 * m2 * m2 * v * v * v * v + 12.0 * m2 * v * v * v * v * v +
          2.0 * v * v * v * v * v * v) /
         (m2 * m2 * m2 * S6);
  t[1] = 120.0 * m * (v - 3.0 * m2) * (3.0 * v - m2) / S6;
  t[2] = 12.0 * (35.0 * m2 * m2 - 42.0 * m2 * v + 3.0 * v * v) / S6;
  t[3] = -48.0 * m * (3.0 * v - 7.0 * m2) / S6;
  t[4] = -24.0 * (v - 9.0 * m2) / S6;
  t[5] = 120.0 * m / S6;
  t[6] = 60.0 / S6;
  return t;
}

// The order-N partials of s and of mu, N = 1 .. 6 (dispatch on the order).
template <int N>
LAPLACE_HD sym_tensor<2, N> s_dk(double m, double v) {
  if constexpr (N == 1) return s_d1(m, v);
  else if constexpr (N == 2) return s_d2(m, v);
  else if constexpr (N == 3) return s_d3(m, v);
  else if constexpr (N == 4) return s_d4(m, v);
  else if constexpr (N == 5) return s_d5(m, v);
  else return s_d6(m, v);
}
template <int N>
LAPLACE_HD sym_tensor<2, N> mu_dk(double m, double v) {
  if constexpr (N == 1) return mu_d1(m, v);
  else if constexpr (N == 2) return mu_d2(m, v);
  else if constexpr (N == 3) return mu_d3(m, v);
  else if constexpr (N == 4) return mu_d4(m, v);
  else if constexpr (N == 5) return mu_d5(m, v);
  else return mu_d6(m, v);
}

// The packed inputs of chain_rule_pair_dk<K>: u[pair_index(p, q)] = d^(p+q) u / dm^p dv^q for
// 1 <= p + q <= K (the order-N block is the packing of sym_tensor<2, N>, its component q at N(N+1)/2 + q),
// and the values at position 0 (not read by the chain rule).
template <int N>
LAPLACE_HD void put_inner(double m, double v, double* U_mu, double* U_s) {
  const auto a = mu_dk<N>(m, v), b = s_dk<N>(m, v);
  for (int q = 0; q <= N; ++q) {
    U_mu[pair_index(N - q, q)] = a[q];
    U_s[pair_index(N - q, q)] = b[q];
  }
}
template <int K>
LAPLACE_HD void inner_arrays(double m, double v, double* U_mu, double* U_s) {
  U_mu[0] = mu_value(m, v);
  U_s[0] = s_value(m, v);
  if constexpr (K >= 1) put_inner<1>(m, v, U_mu, U_s);
  if constexpr (K >= 2) put_inner<2>(m, v, U_mu, U_s);
  if constexpr (K >= 3) put_inner<3>(m, v, U_mu, U_s);
  if constexpr (K >= 4) put_inner<4>(m, v, U_mu, U_s);
  if constexpr (K >= 5) put_inner<5>(m, v, U_mu, U_s);
  if constexpr (K >= 6) put_inner<6>(m, v, U_mu, U_s);
}

// The packed outer tables of chain_rule_pair_dk<K>: G[pair_index(i, j)] = d^(i+j) G / d mu^i d s^j at
// the point (mu, s), with G = lognormal1_distrib's function (log-density, its y-derivatives, log F or
// log(1 - F)). Position 0 holds the value; the order-N block is the packing of the order-N derivative.
template <int N>
LAPLACE_HD void put_block(const sym_tensor<2, N>& t, double* G) {
  for (int q = 0; q <= N; ++q) G[pair_index(N - q, q)] = t[q];
}
// The outer functions depend on (mu, s) through the residual r = log y - mu only, so they are evaluated from r
// (see residual() below): the parameter mu is never formed in the outer tables.
template <int J, int N>
LAPLACE_HD void put_dy(double y, double r, double s, double* G) {
  put_block<N>(detail_lognormal1::dy_dk_residual<J, N>(y, r, s), G);
}
template <int N>
LAPLACE_HD void put_cdf(double r, double s, double* G) {
  put_block<N>(gaussian2_distrib{0.0, s}.log_cdf_dk<N>(r), G);
}
template <int N>
LAPLACE_HD void put_ccdf(double r, double s, double* G) {
  put_block<N>(gaussian2_distrib{0.0, s}.log_ccdf_dk<N>(r), G);
}

// The residual r = log y - mu = log(y/m) + s/2. Near y = m it is formed as log1p((y - m)/m) + s/2: forming
// log y - mu directly loses the digits of s/2 there (mu = log m - s/2 is about 4 at m = 40, s/2 about 3e-7 at
// v = 0.001: a relative error of 6e-11 in the J = 1 derivative, measured in tools/reference/distrib_lognormal.py,
// mode ydk). Far from y = m, (y - m)/m is not small and may round to -1 (log1p of -1 is -infinity), so there
// r is log y - log m + s/2, where log y - log m is at least 0.4 in size and nothing cancels.
LAPLACE_HD_FORCEINLINE double residual(double m, double v, double y) {
  const double s = s_value(m, v);
  if (Kokkos::fabs(y - m) < 0.5 * m) return Kokkos::log1p((y - m) / m) + 0.5 * s;
  return math::log(y) - math::log(m) + 0.5 * s;
}

// The orders 1 .. K of a composition written with chain_rule_pair_dk, packed as the inputs of the
// next composition: h[pair_index(p, q)] for the composed function (position 0 is G[0]).
template <int n>
LAPLACE_HD void compose_order(const double* G, const double* u0, const double* u1, double* h) {
  double out[n + 1];
  laplace::math::chain_rule_pair_dk<n>(G, u0, u1, out);
  for (int b = 0; b <= n; ++b) h[pair_index(n - b, b)] = out[b];
}
template <int K>
LAPLACE_HD void compose_orders(const double* G, const double* u0, const double* u1, double* h) {
  h[0] = G[0];
  if constexpr (K >= 1) compose_order<1>(G, u0, u1, h);
  if constexpr (K >= 2) compose_order<2>(G, u0, u1, h);
  if constexpr (K >= 3) compose_order<3>(G, u0, u1, h);
  if constexpr (K >= 4) compose_order<4>(G, u0, u1, h);
  if constexpr (K >= 5) compose_order<5>(G, u0, u1, h);
  if constexpr (K >= 6) compose_order<6>(G, u0, u1, h);
}

// The packed derivatives of the product f g of two functions of (m, v), orders 0 .. K, from their
// packed derivatives (the chain rule of the outer function u_0 u_1).
template <int K>
LAPLACE_HD void product_packed(const double* f, const double* g, double* h) {
  double G[packed_max] = {};
  G[pair_index(0, 0)] = f[0] * g[0];
  G[pair_index(1, 0)] = g[0];
  G[pair_index(0, 1)] = f[0];
  G[pair_index(1, 1)] = 1.0;
  compose_orders<K>(G, f, g, h);
}

// The packed derivatives of an outer function of (mu, s) that depends on s alone with partials
// G_j = (-1)^j c_j / s^(e_j) at the point, for the factors 1/s and 1/(2 s^2) of the information.
template <int K>
LAPLACE_HD void inverse_power_factor(double s, double scale, int power, double* G) {
  // scale * d^j/ds^j s^(-power) = scale * (-1)^j (power)(power+1)...(power+j-1) s^(-power-j)
  double c = scale;  // scale * (-1)^j * rising factorial, updated per j
  double inv = 1.0;  // s^(-power), by repeated division (power <= 2)
  for (int i = 0; i < power; ++i) inv /= s;
  for (int j = 0; j <= K; ++j) {
    G[pair_index(0, j)] = c * inv;
    // next order: multiply by -(power + j) and by 1 / s
    c *= -static_cast<double>(power + j);
    inv /= s;
  }
}

// The packed expected information of order K: pair[p] for (m,m), (m,v), (v,v), block b = the
// component with b derivatives in v (I_ab = sum_c I_c(s) d_a theta_c d_b theta_c).
template <int K>
LAPLACE_HD expected_info_tensor<2, K> info_tensor(double m, double v) {
  double U_mu[packed_max] = {}, U_s[packed_max] = {};
  inner_arrays<K + 1>(m, v, U_mu, U_s);
  const double s = s_value(m, v);
  double Gm[packed_max] = {}, Gs[packed_max] = {};
  inverse_power_factor<K>(s, 1.0, 1, Gm);   // I_mu = 1/s
  inverse_power_factor<K>(s, 0.5, 2, Gs);   // I_s = 1/(2 s^2)
  double A_mu[packed_max] = {}, A_s[packed_max] = {};
  compose_orders<K>(Gm, U_mu, U_s, A_mu);
  compose_orders<K>(Gs, U_mu, U_s, A_s);
  // F[c][a]: the first derivative of theta_c in a (a = 0: m, a = 1: v) as a packed function
  double F_mu_m[packed_max] = {}, F_mu_v[packed_max] = {}, F_s_m[packed_max] = {}, F_s_v[packed_max] = {};
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      F_mu_m[pair_index(p, q)] = U_mu[pair_index(p + 1, q)];
      F_mu_v[pair_index(p, q)] = U_mu[pair_index(p, q + 1)];
      F_s_m[pair_index(p, q)] = U_s[pair_index(p + 1, q)];
      F_s_v[pair_index(p, q)] = U_s[pair_index(p, q + 1)];
    }
  expected_info_tensor<2, K> out;
  const double* F_mu[2] = {F_mu_m, F_mu_v};
  const double* F_s[2] = {F_s_m, F_s_v};
  const int ia[3] = {0, 0, 1}, ib[3] = {0, 1, 1};  // (m, m), (m, v), (v, v)
  for (int p = 0; p < 3; ++p) {
    double Q_mu[packed_max] = {}, Q_s[packed_max] = {}, T_mu[packed_max] = {}, T_s[packed_max] = {};
    product_packed<K>(F_mu[ia[p]], F_mu[ib[p]], Q_mu);
    product_packed<K>(A_mu, Q_mu, T_mu);
    product_packed<K>(F_s[ia[p]], F_s[ib[p]], Q_s);
    product_packed<K>(A_s, Q_s, T_s);
    for (int b = 0; b <= K; ++b) out.pair[p][b] = T_mu[pair_index(K - b, b)] + T_s[pair_index(K - b, b)];
  }
  return out;
}

}  // namespace detail_lognormal2

/**
 * \ingroup distrib_lognormal2
 * \brief Lognormal distribution parameterized by the mean \f$m > 0\f$ and the variance \f$v > 0\f$ of \f$Y\f$.
 *
 * The same law as lognormal1_distrib with \f$s = \log(1 + v/m^2)\f$ (the variance of \f$\log Y\f$) and
 * \f$\mu = \log m - s/2\f$ (its mean); the log-density is \f$\ell(m, v; y) = \ell_{\mathrm{N}}(\mu, s; \log y) - \log y\f$.
 * Its parameter derivatives, log F and log(1 - F) are compositions of lognormal1's with the map (m, v) to (mu, s)
 * (chain_rule_pair_dk), and its expected information is the tensor I' J transformation of gaussian2's.
 * \see lognormal1_distrib, gaussian2_distrib, distrib_lognormal2
 */
struct lognormal2_distrib {
  /// Mean of \f$Y\f$, \f$m \in (0, \infty)\f$.
  double m;
  /// Variance of \f$Y\f$, \f$v \in (0, \infty)\f$.
  double v;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Differentiability order of the log-density in \f$y\f$ on \f$(0, \infty)\f$ (\c infinitely_smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c m and \c v.
  static constexpr distrib_param_info params[2] = {
      {"m", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"v", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// The lognormal1_distrib with the same law: \f$(\mu, s)\f$ of this parametrization.
  LAPLACE_HD lognormal1_distrib base() const {
    return lognormal1_distrib{detail_lognormal2::mu_value(m, v), detail_lognormal2::s_value(m, v)};
  }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates the log-density \f$\log f(y; m, v)\f$ for \f$y > 0\f$.
   * \param y Observation point \f$y \in (0, \infty)\f$.
   * \return \f$\log f(y; m, v)\f$ (the same value as lognormal1_distrib at \f$(\mu, s)\f$).
   */
  LAPLACE_HD double logpdf(double y) const {
    return detail_lognormal1::dy_dk_residual<0, 0>(y, detail_lognormal2::residual(m, v, y), detail_lognormal2::s_value(m, v))[0];
  }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates \f$\partial_y^J \partial_\theta^K \log f(y; m, v)\f$, \f$J + K \le 6\f$, for \f$y > 0\f$.
   *
   * The composition of lognormal1's \f$\partial_y^J \partial_{(\mu, s)}^N\f$ log-density, for \f$N \le K\f$, with the map
   * (m, v) to (mu, s): chain_rule_pair_dk<K> on the packed outer and inner derivatives.
   *
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in \f$(m, v)\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    static_assert(J >= 0 && K >= 0 && J + K <= 6, "lognormal2_distrib: J + K must be at most 6");
    const double s = detail_lognormal2::s_value(m, v), r = detail_lognormal2::residual(m, v, y);
    sym_tensor<2, K> t;
    if constexpr (K == 0) {
      t[0] = detail_lognormal1::dy_dk_residual<J, 0>(y, r, s)[0];
    } else {
      double G[detail_lognormal2::packed_max] = {}, U_mu[detail_lognormal2::packed_max] = {},
             U_s[detail_lognormal2::packed_max] = {};
      G[0] = detail_lognormal1::dy_dk_residual<J, 0>(y, r, s)[0];
      if constexpr (K >= 1) detail_lognormal2::put_dy<J, 1>(y, r, s, G);
      if constexpr (K >= 2) detail_lognormal2::put_dy<J, 2>(y, r, s, G);
      if constexpr (K >= 3) detail_lognormal2::put_dy<J, 3>(y, r, s, G);
      if constexpr (K >= 4) detail_lognormal2::put_dy<J, 4>(y, r, s, G);
      if constexpr (K >= 5) detail_lognormal2::put_dy<J, 5>(y, r, s, G);
      if constexpr (K >= 6) detail_lognormal2::put_dy<J, 6>(y, r, s, G);
      detail_lognormal2::inner_arrays<K>(m, v, U_mu, U_s);
      laplace::math::chain_rule_pair_dk<K>(G, U_mu, U_s, t.v);
    }
    return t;
  }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log f(y; m, v)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    return logpdf_dy_dk<0, K>(y);
  }
  /// First-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates the expected Fisher information \f$I'\f$ in \f$(m, v)\f$ as a tensor, \f$I_{ab} = \sum_c I_c \partial_a\theta_c \partial_b\theta_c\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2> with \f$I_{mm}\f$, \f$I_{mv}\f$, \f$I_{vv}\f$ at positions 0, 1, 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const auto t = detail_lognormal2::info_tensor<0>(m, v);
    sym_tensor<2, 2> out;
    for (int p = 0; p < 3; ++p) out[p] = t.pair[p][0];
    return out;
  }
  /// First-order derivatives of the expected information in \f$(m, v)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return detail_lognormal2::info_tensor<1>(m, v); }
  /// Second-order derivatives of the expected information in \f$(m, v)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return detail_lognormal2::info_tensor<2>(m, v); }
  /// Third-order derivatives of the expected information in \f$(m, v)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return detail_lognormal2::info_tensor<3>(m, v); }
  /// Fourth-order derivatives of the expected information in \f$(m, v)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return detail_lognormal2::info_tensor<4>(m, v); }

  /// \f$F(y)\f$ (lognormal1's at \f$(\mu, s)\f$; 0 for \f$y \le 0\f$).
  LAPLACE_HD double cdf(double y) const {
    if (y <= 0.0) return 0.0;
    return gaussian2_distrib{0.0, detail_lognormal2::s_value(m, v)}.cdf(detail_lognormal2::residual(m, v, y));
  }
  /// \f$1 - F(y)\f$ (1 for \f$y \le 0\f$).
  LAPLACE_HD double ccdf(double y) const {
    if (y <= 0.0) return 1.0;
    return gaussian2_distrib{0.0, detail_lognormal2::s_value(m, v)}.ccdf(detail_lognormal2::residual(m, v, y));
  }
  /// \f$\log F(y)\f$.
  LAPLACE_HD double log_cdf(double y) const {
    if (y <= 0.0) return -distrib_infinity;
    return gaussian2_distrib{0.0, detail_lognormal2::s_value(m, v)}.log_cdf(detail_lognormal2::residual(m, v, y));
  }
  /// \f$\log(1 - F(y))\f$.
  LAPLACE_HD double log_ccdf(double y) const {
    if (y <= 0.0) return 0.0;
    return gaussian2_distrib{0.0, detail_lognormal2::s_value(m, v)}.log_ccdf(detail_lognormal2::residual(m, v, y));
  }
  /// Quantile \f$\exp(\mu + \sqrt{s}\,\Phi^{-1}(p))\f$ with \f$\mu\f$ and \f$s\f$ of this parametrization.
  LAPLACE_HD double quantile(double p) const { return base().quantile(p); }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ by composition with the map (m, v) to (mu, s).
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "lognormal2_distrib: K must be between 1 and 6");
    const double s = detail_lognormal2::s_value(m, v), r = detail_lognormal2::residual(m, v, y);
    double G[detail_lognormal2::packed_max] = {}, U_mu[detail_lognormal2::packed_max] = {},
           U_s[detail_lognormal2::packed_max] = {};
    G[0] = gaussian2_distrib{0.0, s}.log_cdf(r);
    if constexpr (K >= 1) detail_lognormal2::put_cdf<1>(r, s, G);
    if constexpr (K >= 2) detail_lognormal2::put_cdf<2>(r, s, G);
    if constexpr (K >= 3) detail_lognormal2::put_cdf<3>(r, s, G);
    if constexpr (K >= 4) detail_lognormal2::put_cdf<4>(r, s, G);
    if constexpr (K >= 5) detail_lognormal2::put_cdf<5>(r, s, G);
    if constexpr (K >= 6) detail_lognormal2::put_cdf<6>(r, s, G);
    detail_lognormal2::inner_arrays<K>(m, v, U_mu, U_s);
    sym_tensor<2, K> t;
    laplace::math::chain_rule_pair_dk<K>(G, U_mu, U_s, t.v);
    return t;
  }

  /**
   * \ingroup distrib_lognormal2
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ by composition with the map (m, v) to (mu, s).
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const {
    static_assert(K >= 1 && K <= 6, "lognormal2_distrib: K must be between 1 and 6");
    const double s = detail_lognormal2::s_value(m, v), r = detail_lognormal2::residual(m, v, y);
    double G[detail_lognormal2::packed_max] = {}, U_mu[detail_lognormal2::packed_max] = {},
           U_s[detail_lognormal2::packed_max] = {};
    G[0] = gaussian2_distrib{0.0, s}.log_ccdf(r);
    if constexpr (K >= 1) detail_lognormal2::put_ccdf<1>(r, s, G);
    if constexpr (K >= 2) detail_lognormal2::put_ccdf<2>(r, s, G);
    if constexpr (K >= 3) detail_lognormal2::put_ccdf<3>(r, s, G);
    if constexpr (K >= 4) detail_lognormal2::put_ccdf<4>(r, s, G);
    if constexpr (K >= 5) detail_lognormal2::put_ccdf<5>(r, s, G);
    if constexpr (K >= 6) detail_lognormal2::put_ccdf<6>(r, s, G);
    detail_lognormal2::inner_arrays<K>(m, v, U_mu, U_s);
    sym_tensor<2, K> t;
    laplace::math::chain_rule_pair_dk<K>(G, U_mu, U_s, t.v);
    return t;
  }
  /// First-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /// First-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order parameter derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }

  /// Lower bound of the support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of the support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location: the mean \f$m\f$.
  LAPLACE_HD double center() const { return m; }
  /// Typical spread \f$\sqrt{v}\f$ (the standard deviation of \f$Y\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(v); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
