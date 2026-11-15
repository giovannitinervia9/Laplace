// Azzalini's skew t (laplace::distrib), in the direct parametrization:
//
//   f(y) = (2 / sigma) t_nu(z) T_(nu+1)(w),   z = (y - mu)/sigma,
//   w = alpha z sqrt((nu + 1)/(nu + z^2)),
//
// with t_nu the Student density and T_m the Student cdf at m degrees of
// freedom. On the log scale
//
//   log f = log 2 + c(nu) - log sigma + Lambda(z, nu) + log T_(nu+1)(w),
//   Lambda(z, nu) = -((nu + 1)/2) log(1 + z^2/nu),
//
// so the family is the Student t of student_t1.hpp with one factor more. The
// two logarithms are taken separately: own::log_student_cdf returns the second
// on the log scale, so the light tail gives a large negative number and not
// -inf.
//
// The derivatives to order six. The parameters are (mu, sigma, alpha, nu) and
// y enters only through z, so the assembly is the location-scale one with two
// smooth shape parameters, detail_location_scale::mixed_two_shape_dk, fed with
// the table
//
//   D[(c1, c2, j)] = d^c1/dalpha^c1 d^c2/dnu^c2 d^j/dz^j
//                    [Lambda(z, nu) + log T_(nu+1)(w)],   c1 + c2 + j <= K,
//
// to which c(nu) and -log sigma add their own derivatives in the components
// with every index in nu and in sigma. The first term is student_t1's table.
// The second is a composition in three variables, which two facts reduce to
// compositions in two:
//
//   1. alpha enters w = alpha u(z, nu) linearly, with u independent of alpha,
//      so d^c1/dalpha^c1 log T_(nu+1)(w) = u^c1 (d^c1/dw^c1 log T)(w, nu + 1)
//      exactly: no Bell sum in alpha, only a power of u;
//   2. what is left is a function of the two arguments (w, m) of the Student
//      cdf composed with the two inner maps w = alpha u(z, nu) and m = nu + 1
//      in the two variables (z, nu), which is math::chain_rule_pair_dk, and
//      the second map is linear.
//
// So, for each order c1 in alpha, the table of the (z, nu) derivatives of
// u^c1 (chain_rule_pair_scalar_dk on a power) and of
// (d^c1/dw^c1 log T)(alpha u, nu + 1) (chain_rule_pair_dk with the outer table
// shifted by c1 in its first argument) are formed and multiplied by Leibniz's
// rule. The only new closed forms the family needs are the mixed (z, nu)
// derivatives of u, below.
//
// alpha = 0 is student_t1, nu -> infinity the direct skew normal, and
// |alpha| -> infinity the half t. The cdf, which has no simple closed form at
// real nu, is the generic route (quadrature of the density), measured against
// the chi-square mixture of skew normal cdfs and chosen on 2026-11-15; this
// file writes no cdf member, and the dispatchers of generic_cdf.hpp answer for
// it. The expected information has no closed form either and comes from the
// generic quadrature.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/location_scale.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta_deriv.hpp>
#include <laplace/math/student.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_skewt {

// Number of multi-indices (p, q) of total order at most six, in the packing of
// math::chain_rule_pair_dk: (p, q) sits at pair(p, q), by total order and then
// by the order in the second variable.
constexpr int n_pair = 28;
LAPLACE_HD_FORCEINLINE constexpr int pair(int p, int q) { return ((p + q) * (p + q + 1)) / 2 + q; }

// the packed position of the multi-index with a derivatives in mu, b in
// sigma, c in alpha and d in nu, a + b + c + d = K
template <int K>
LAPLACE_HD_FORCEINLINE int pos4(int a, int b, int c, int d) {
  if constexpr (K == 0) {
    return 0;
  } else {
    int idx[K];
    int p = 0;
    for (int i = 0; i < a; ++i) idx[p++] = 0;
    for (int i = 0; i < b; ++i) idx[p++] = 1;
    for (int i = 0; i < c; ++i) idx[p++] = 2;
    for (int i = 0; i < d; ++i) idx[p++] = 3;
    return sym_tensor_position<4, K>(idx);
  }
}

// row of the (z, nu) tables below: entry (j, c) at j * 7 + c
LAPLACE_HD_FORCEINLINE constexpr int tab(int j, int c) { return j * 7 + c; }

LAPLACE_HD_FORCEINLINE constexpr double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE constexpr double binom(int n, int k) {
  double r = 1.0;
  for (int i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
  return r;
}

// The mixed derivatives of u(z, nu) = z sqrt((nu + 1)/(nu + z^2)), packed as
// U[pair(p, q)] = d^p/dz^p d^q/dnu^q u, p + q <= K, the value at pair(0, 0).
//
// With A = nu + 1, B = nu + z^2 and u = z A^(1/2) B^(-1/2):
//   d^q/dnu^q A^(1/2) = (1/2)(1/2 - 1)...(1/2 - q + 1) A^(1/2 - q),
//   d^q/dnu^q B^(-1/2) = phi_q B^(-1/2 - q),  phi_q = prod_(i<q) (-1/2 - i),
// since dB/dnu = 1, and the z-derivatives of z^m B^(-s) follow from
//   d/dz [z^m B^(-s)] = m z^(m-1) B^(-s) - 2 s z^(m+1) B^(-s-1),
// so one derivative in z maps the pair (m, s) to (m - 1, s) and (m + 1, s + 1)
// with those two coefficients: the table c[m][r] of the coefficient of
// z^m B^(-s-r) is carried forward one z-derivative at a time, every exponent
// staying a half-integer, and B^(-s-r) is evaluated as B^(-1/2) times a power
// of 1/B. Finally z and A^(1/2) are joined by
//   d^p/dz^p [z W] = z d^p W/dz^p + p d^(p-1) W/dz^(p-1)
// and by Leibniz's rule in nu.
template <int K>
LAPLACE_HD void u_table(double z, double nu, double* U) {
  const double A = nu + 1.0, B = nu + z * z;
  const double iB = 1.0 / B, isB = 1.0 / Kokkos::sqrt(B);
  double Q[49] = {};  // Q[tab(p, q)] = d^p/dz^p d^q/dnu^q B^(-1/2)
  for (int q = 0; q <= K; ++q) {
    double phi = 1.0;
    for (int i = 0; i < q; ++i) phi *= -0.5 - i;
    const double s = 0.5 + q;
    double c[8][8] = {}, d[8][8] = {};
    c[0][0] = phi;
    for (int p = 0; p + q <= K; ++p) {
      if (p > 0) {
        for (int m = 0; m < 8; ++m)
          for (int r = 0; r < 8; ++r) d[m][r] = 0.0;
        for (int m = 0; m <= K; ++m)
          for (int r = 0; r <= K; ++r) {
            const double v = c[m][r];
            if (v == 0.0) continue;
            if (m > 0) d[m - 1][r] += m * v;
            d[m + 1][r + 1] += -2.0 * (s + r) * v;
          }
        for (int m = 0; m < 8; ++m)
          for (int r = 0; r < 8; ++r) c[m][r] = d[m][r];
      }
      double acc = 0.0;
      double ipow[8];
      ipow[0] = 1.0;
      for (int r = 1; r < 8; ++r) ipow[r] = ipow[r - 1] * iB;
      double zpow[8];
      zpow[0] = 1.0;
      for (int m = 1; m < 8; ++m) zpow[m] = zpow[m - 1] * z;
      for (int m = 0; m <= K + 1; ++m)
        for (int r = 0; r <= K + 1; ++r)
          if (c[m][r] != 0.0) acc += c[m][r] * zpow[m] * ipow[q + r];
      Q[tab(p, q)] = acc * isB;
    }
  }
  double P[7];  // P[i] = d^i/dnu^i A^(1/2)
  P[0] = Kokkos::sqrt(A);
  for (int i = 1; i <= K; ++i) P[i] = P[i - 1] * (0.5 - (i - 1)) / A;
  for (int q = 0; q <= K; ++q)
    for (int p = 0; p + q <= K; ++p) {
      double acc = 0.0;
      for (int i = 0; i <= q; ++i) {
        const double zq = z * Q[tab(p, q - i)] + (p > 0 ? p * Q[tab(p - 1, q - i)] : 0.0);
        acc += binom(q, i) * P[i] * zq;
      }
      U[pair(p, q)] = acc;
    }
}

// u(z, nu) itself, written so that z^2 does not overflow and the value is
// exact at large |z|, where u tends to sign(z) sqrt(nu + 1).
LAPLACE_HD_FORCEINLINE double u_value(double z, double nu) {
  const double a = Kokkos::fabs(z);
  if (a <= 1.0) return z * Kokkos::sqrt((nu + 1.0) / (nu + z * z));
  const double r = nu / (a * a);
  return (z < 0.0 ? -1.0 : 1.0) * Kokkos::sqrt((nu + 1.0) / (1.0 + r));
}

// The (z, nu) derivatives of order N of the composition
// G(alpha u(z, nu), nu + 1), written into R[tab(j, c)] with j in z and c in nu,
// one call of chain_rule_pair_dk per order from one to NMAX.
template <int N, int NMAX>
LAPLACE_HD void pair_orders(const double* G, const double* u0, const double* u1, double* R) {
  if constexpr (N >= 1 && N <= NMAX) {
    double out[N + 1];
    math::chain_rule_pair_dk<N>(G, u0, u1, out);
    for (int b = 0; b <= N; ++b) R[tab(N - b, b)] = out[b];
    pair_orders<N + 1, NMAX>(G, u0, u1, R);
  }
}

// the same for a power of u, an outer function of one variable
template <int N, int NMAX>
LAPLACE_HD void scalar_orders(const double* f, const double* u, double* W) {
  if constexpr (N >= 1 && N <= NMAX) {
    double out[N + 1];
    math::chain_rule_pair_scalar_dk<N>(f, u, out);
    for (int b = 0; b <= N; ++b) W[tab(N - b, b)] = out[b];
    scalar_orders<N + 1, NMAX>(f, u, W);
  }
}

// One order C1 in alpha of the table: the rows D[(C1, c2, j)] of
// d^C1/dalpha^C1 log T_(nu+1)(alpha u) = u^C1 (d^C1 log T / dw^C1)(alpha u, nu + 1),
// then the next order down.
template <int K, int C1>
LAPLACE_HD void alpha_row(const double* H, const double* U, const double* u0, const double* u1,
                          double u, double* D) {
  constexpr int n1 = K - C1;
  double R[49] = {}, W[49] = {};
  double G[n_pair] = {};
  for (int n = 0; n <= n1; ++n)
    for (int j = 0; j <= n; ++j) G[pair(n - j, j)] = H[pair(n - j + C1, j)];
  R[tab(0, 0)] = H[pair(C1, 0)];
  pair_orders<1, n1>(G, u0, u1, R);
  if constexpr (C1 == 0) {
    W[tab(0, 0)] = 1.0;
  } else {
    // f[i] = d^i/dx^i x^C1 at x = u, that is C1!/(C1 - i)! u^(C1 - i)
    double f[7] = {};
    for (int i = 0; i <= C1 && i <= K; ++i) {
      double v = 1.0;
      for (int k = 0; k < i; ++k) v *= C1 - k;
      for (int k = 0; k < C1 - i; ++k) v *= u;
      f[i] = v;
    }
    W[tab(0, 0)] = f[0];
    scalar_orders<1, n1>(f, U, W);
  }
  for (int c2 = 0; c2 <= n1; ++c2)
    for (int j = 0; j + c2 <= n1; ++j) {
      double acc = 0.0;
      for (int jp = 0; jp <= j; ++jp)
        for (int cp = 0; cp <= c2; ++cp)
          acc += binom(j, jp) * binom(c2, cp) * W[tab(jp, cp)] * R[tab(j - jp, c2 - cp)];
      D[(C1 * (K + 1) + c2) * (K + 1) + j] += acc;
    }
  if constexpr (C1 > 0) alpha_row<K, C1 - 1>(H, U, u0, u1, u, D);
}

// The whole table of the skew t's Lambda(z, alpha, nu) = log t_nu(z) - c(nu)
// + log T_(nu+1)(w), in the layout mixed_two_shape_dk reads.
template <int K>
LAPLACE_HD void lambda_table(double z, double alpha, double nu, double* D) {
  double U[n_pair] = {};
  u_table<K>(z, nu, U);
  const double u = u_value(z, nu);
  const double w = alpha * u;
  double L[math::own::detail_st::NC];
  math::own::detail_st::derivs<K>(w, nu + 1.0, math::own::detail_st::par{}, L);
  double H[n_pair] = {};  // H[pair(i, j)] = d^i/dw^i d^j/dm^j log T_m(w)
  for (int n = 0; n <= K; ++n)
    for (int j = 0; j <= n; ++j) {
      const int i = n - j;
      H[pair(i, j)] = L[math::own::detail_ibd::ix(i, j)] * fact(i) * fact(j);
    }
  double u0[n_pair] = {}, u1[n_pair] = {};
  for (int n = 1; n <= K; ++n)
    for (int j = 0; j <= n; ++j) u0[pair(n - j, j)] = alpha * U[pair(n - j, j)];
  u1[pair(0, 1)] = 1.0;
  alpha_row<K, K>(H, U, u0, u1, u, D);
  // the Student part, which carries no alpha
  double S[(K + 1) * (K + 1)] = {};
  detail_student_t1::lambda_table<K>(z, nu, S);
  for (int c2 = 0; c2 <= K; ++c2)
    for (int j = 0; c2 + j <= K; ++j) D[c2 * (K + 1) + j] += S[c2 * (K + 1) + j];
}

}  // namespace detail_skewt

/**
 * \ingroup distrib_skewt
 * \brief Azzalini's skew \f$t\f$ distribution on \f$\mathbb{R}\f$ in direct parametrization.
 *
 * Models a continuous outcome \f$y \in (-\infty, \infty)\f$ with location
 * \f$\mu \in \mathbb{R}\f$, scale \f$\sigma > 0\f$, shape \f$\alpha \in \mathbb{R}\f$, and
 * degrees of freedom \f$\nu > 0\f$ via the density
 * \f[
 *   f(y \mid \mu, \sigma, \alpha, \nu)
 *     = \frac{2}{\sigma}\, t_\nu(z)\, T_{\nu+1}\!\left(\alpha\, z \sqrt{\frac{\nu + 1}{\nu + z^2}}\right),
 *   \qquad z = \frac{y - \mu}{\sigma},
 * \f]
 * where \f$t_\nu\f$ is the standard Student \f$t\f$ density with \f$\nu\f$ degrees of freedom
 * and \f$T_{\nu+1}\f$ is the standard Student \f$t\f$ cumulative distribution function with
 * \f$\nu + 1\f$ degrees of freedom.
 *
 * The log-density is evaluated as
 * \f[
 *   \log f(y) = \log 2 + c(\nu) - \log \sigma - \frac{\nu + 1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right)
 *               + \log T_{\nu+1}(w),
 * \f]
 * with \f$c(\nu) = \log\Gamma((\nu + 1)/2) - \log\Gamma(\nu/2) - \frac{1}{2}\log(\nu\pi)\f$ and
 * \f$w = \alpha\, z \sqrt{(\nu + 1)/(\nu + z^2)}\f$. Reducing \f$\alpha = 0\f$ recovers the
 * Student \f$t\f$ distribution (\f$\nu\f$ degrees of freedom), and letting \f$\nu \to \infty\f$
 * yields the skew normal distribution in direct parametrization. Cumulative probabilities,
 * quantiles, and Fisher information are provided through \ref distrib_generic.
 *
 * \see distrib_skewt, logpdf(), logpdf_dk(), logpdf_dy_dk(), rng(), start_value()
 */
// Azzalini's skew t, with the location mu, the scale sigma, the shape alpha and the
// degrees of freedom nu.
struct skewt_distrib {
  /// \brief Location parameter \f$\mu \in (-\infty, \infty)\f$.
  double mu;
  /// \brief Scale parameter \f$\sigma > 0\f$ (not the standard deviation).
  double sigma;
  /// \brief Shape parameter \f$\alpha \in (-\infty, \infty)\f$; \f$\alpha = 0\f$ gives the Student \f$t\f$.
  double alpha;
  /// \brief Degrees-of-freedom parameter \f$\nu > 0\f$.
  double nu;

  /// \brief Distribution kind tag (continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// \brief Number of scalar parameters in the parameter vector \f$(\mu, \sigma, \alpha, \nu)\f$.
  static constexpr int n_params = 4;
  /// \brief Smoothness order of the log-density with respect to \f$y\f$.
  static constexpr int y_regularity = infinitely_smooth;
  /// \brief Metadata table describing the parameters \f$\mu\f$, \f$\sigma\f$, \f$\alpha\f$, and \f$\nu\f$.
  static constexpr distrib_param_info params[4] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"alpha", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"nu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_skewt
   * \brief Evaluate the log-density \f$\log f(y \mid \mu, \sigma, \alpha, \nu)\f$.
   *
   * Computes
   * \f[
   *   \log f(y) = \log 2 + c(\nu) - \log \sigma
   *               - \frac{\nu + 1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right)
   *               + \log T_{\nu+1}\!\left(\alpha\, u(z, \nu)\right),
   * \f]
   * where \f$z = (y - \mu)/\sigma\f$, \f$u(z, \nu) = z\sqrt{(\nu + 1)/(\nu + z^2)}\f$
   * (evaluated in scaled form when \f$|z| > 1\f$ to avoid overflow), and
   * \f$\log T_{\nu+1}\f$ is supplied by \c log_student_cdf().
   *
   * \param y Observed real value \f$y \in (-\infty, \infty)\f$.
   * \return Log-density \f$\log f(y \mid \mu, \sigma, \alpha, \nu)\f$.
   * \see logpdf_dk(), logpdf_dy_dk()
   */
  LAPLACE_HD double logpdf(double y) const {
    const double z = (y - mu) / sigma;
    double cp[1];
    math::own::detail_st::cpoly(nu, 0, cp);
    const double w = alpha * detail_skewt::u_value(z, nu);
    return 0.6931471805599453 + cp[0] - math::log(sigma) - 0.5 * (nu + 1.0) * Kokkos::log1p(z * z / nu) +
           math::own::log_student_cdf(w, nu + 1.0);
  }

  /**
   * \ingroup distrib_skewt
   * \brief Partial derivatives of order \f$K\f$ of \f$\log f(y \mid \mu, \sigma, \alpha, \nu)\f$
   *        with respect to \f$(\mu, \sigma, \alpha, \nu)\f$.
   *
   * Combines the Student log-CDF derivatives of \f$\log T_{\nu+1}(w)\f$ with the bivariate
   * Faà di Bruno chain rule for \f$w = \alpha\, u(z, \nu)\f$ and the shared location-scale
   * transformation in \f$(z, \sigma)\f$, then adds the \f$\nu\f$-only normalizing-constant
   * derivatives and \f$\sigma\f$-only Jacobian term.
   *
   * \tparam K Derivative order, \f$1 \le K \le 6\f$.
   * \param y Observed real value \f$y \in (-\infty, \infty)\f$.
   * \return Symmetric tensor of order \f$K\f$ over \f$(\mu, \sigma, \alpha, \nu)\f$.
   * \see logpdf(), logpdf_d1(), logpdf_dy_dk()
   */
  template <int K>
  LAPLACE_HD sym_tensor<4, K> logpdf_dk(double y) const {
    const double z = (y - mu) / sigma;
    double D[(K + 1) * (K + 1) * (K + 1)] = {};
    detail_skewt::lambda_table<K>(z, alpha, nu, D);
    sym_tensor<4, K> t = detail_location_scale::mixed_two_shape_dk<K>(D, z, sigma);
    double cp[7];
    math::own::detail_st::cpoly(nu, K, cp);
    t[detail_skewt::pos4<K>(0, 0, 0, K)] += detail_skewt::fact(K) * cp[K];
    double isk = 1.0;
    for (int k = 0; k < K; ++k) isk /= sigma;
    t[detail_skewt::pos4<K>(0, K, 0, 0)] +=
        detail_student_t1::sign_of_power(K) * detail_skewt::fact(K - 1) * isk;
    return t;
  }

  /**
   * \name Fixed-order parameter derivatives
   * \brief Convenience wrappers returning the order-\f$K\f$ derivative tensor of
   *        \f$\log f(y \mid \mu, \sigma, \alpha, \nu)\f$ over \f$(\mu, \sigma, \alpha, \nu)\f$.
   * @{
   */
  /// \brief First-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<1>(y).
  LAPLACE_HD sym_tensor<4, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// \brief Second-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<2>(y).
  LAPLACE_HD sym_tensor<4, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// \brief Third-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<3>(y).
  LAPLACE_HD sym_tensor<4, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// \brief Fourth-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<4>(y).
  LAPLACE_HD sym_tensor<4, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// \brief Fifth-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<5>(y).
  LAPLACE_HD sym_tensor<4, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// \brief Sixth-order parameter derivatives of \f$\log f\f$; returns \c logpdf_dk<6>(y).
  LAPLACE_HD sym_tensor<4, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /// @}

  /**
   * \ingroup distrib_skewt
   * \brief Mixed derivatives of order \f$J\f$ in \f$y\f$ and order \f$K\f$ in \f$(\mu, \sigma, \alpha, \nu)\f$.
   *
   * Because \f$y\f$ and \f$\mu\f$ enter \f$\log f\f$ only through \f$y - \mu\f$,
   * \f$\partial/\partial y = -\partial/\partial\mu\f$, so each mixed derivative equals
   * \f$(-1)^J\f$ times the corresponding component of \c logpdf_dk<J + K>() with \f$J\f$
   * additional differentiations in \f$\mu\f$.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (\f$J \ge 0\f$, \f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to \f$(\mu, \sigma, \alpha, \nu)\f$ (\f$K \ge 0\f$).
   * \param y Observed real value \f$y \in (-\infty, \infty)\f$.
   * \return Symmetric tensor of order \f$K\f$ in \f$(\mu, \sigma, \alpha, \nu)\f$.
   * \see logpdf(), logpdf_dk()
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<4, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<4, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<4, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<4, K> r;
      const double s = detail_student_t1::sign_of_power(J);
      for (int b = 0; b <= K; ++b)
        for (int c = 0; b + c <= K; ++c)
          for (int d = 0; b + c + d <= K; ++d) {
            const int a = K - b - c - d;
            r[detail_skewt::pos4<K>(a, b, c, d)] = s * t[detail_skewt::pos4<J + K>(a + J, b, c, d)];
          }
      return r;
    }
  }

  /**
   * \ingroup distrib_skewt
   * \brief Draw a variate from the exact skew \f$t\f$ representation using a uniform generator.
   *
   * With \f$\delta = \alpha / \sqrt{1 + \alpha^2}\f$, independent standard normal draws
   * \f$N_1, N_2 \sim \mathcal{N}(0, 1)\f$, and \f$G \sim \chi^2_\nu\f$ independent of both,
   * returns
   * \f[
   *   Y = \mu + \sigma\, \frac{\delta |N_1| + \sqrt{1 - \delta^2}\, N_2}{\sqrt{G / \nu}},
   * \f]
   * which is Azzalini and Capitanio's scale-mixture representation of the skew \f$t\f$:
   * the numerator is a standard skew normal of shape \f$\alpha\f$ and the denominator is
   * Student's own scale mixture. The three draws come from \c standard_normal_by_inversion()
   * and \c standard_gamma_by_inversion(), each consuming one uniform variate from \p gen, so
   * the draw is reproducible across threads and backends. \f$G / \nu\f$ is \f$2 / \nu\f$ times
   * a unit-scale gamma draw of shape \f$\nu / 2\f$.
   *
   * \tparam Gen Uniform source type satisfying the counter-based generator interface.
   * \param gen Uniform variate source.
   * \return A single draw \f$Y \sim \text{ST}(\mu, \sigma^2, \alpha, \nu)\f$.
   * \see distrib_skewt
   */
  template <class Gen>
  LAPLACE_HD double rng(Gen& gen) const {
    const double delta = alpha / Kokkos::sqrt(1.0 + alpha * alpha);
    const double n1 = standard_normal_by_inversion(gen);
    const double n2 = standard_normal_by_inversion(gen);
    const double w = 2.0 * standard_gamma_by_inversion(0.5 * nu, gen) / nu;
    return mu + sigma * (delta * Kokkos::fabs(n1) + Kokkos::sqrt(1.0 - delta * delta) * n2) /
                    Kokkos::sqrt(w);
  }

  /**
   * \ingroup distrib_skewt
   * \brief Draw a variate from the counter-based default stream for observation \p index under \p seed.
   *
   * \param seed 64-bit random seed.
   * \param index Observation index defining the independent counter stream.
   * \return A single draw \f$Y \sim \text{ST}(\mu, \sigma^2, \alpha, \nu)\f$.
   * \see rng(Gen&) const
   */
  LAPLACE_HD double rng(std::uint64_t seed, std::uint64_t index) const {
    counter_uniform_source src{seed, index};
    return rng(src);
  }

  /// \brief Lower bound of the support, \f$-\infty\f$.
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// \brief Upper bound of the support, \f$+\infty\f$.
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \brief Central location \f$\mu\f$ used by generic quadrature and root-finding helpers.
  LAPLACE_HD double center() const { return mu; }
  /// \brief Typical spread \f$\sigma\f$ used by generic quadrature and root-finding helpers.
  LAPLACE_HD double scale() const { return sigma; }
  /// \brief Return non-smooth points in parameter \p j (empty: smooth everywhere in \f$(\mu, \sigma, \alpha, \nu)\f$).
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// \brief Return non-smooth points in \f$y\f$ (empty: infinitely smooth on \f$\mathbb{R}\f$).
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }

  /**
   * \ingroup distrib_skewt
   * \brief Construct robust starting values for \f$(\mu, \sigma, \alpha, \nu)\f$ from a sample summary.
   *
   * Uses the sample median for \f$\mu\f$, half the interquartile range for \f$\sigma\f$,
   * the sign of sample skewness (\f$-1, 0, +1\f$) for \f$\alpha\f$, and \f$\nu = 5\f$,
   * nudged strictly inside parameter bounds via \c start_inside().
   *
   * \param s Empirical sample summary computed from the data.
   * \return Initial \c skewt_distrib parameter tuple inside the interior of the parameter space.
   * \see distrib_start, start_inside()
   */
  // Starting value from a sample (generic_start.hpp): the median, half the interquartile range
  // (robust, the law may have no moments), the sign of the sample skewness for alpha and nu = 5.
  skewt_distrib start_value(const distrib_sample_summary& s) const {
    const double a = s.skewness > 0.0 ? 1.0 : (s.skewness < 0.0 ? -1.0 : 0.0);
    return {start_inside(s.median, params[0]),
            start_inside(0.5 * (s.upper_quartile - s.lower_quartile), params[1]),
            start_inside(a, params[2]), start_inside(5.0, params[3])};
  }
};

}  // namespace laplace::distrib
