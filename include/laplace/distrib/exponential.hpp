// exponential_distrib (laplace::distrib): the exponential distribution in its mean
// mu > 0 (rate 1/mu),
//
//   l(mu; y) = -log mu - u,   u = y/mu,   y > 0.
//
// Parameter derivatives, with u = y/mu and u^(n) = d^n u/dmu^n = (-1)^n n! u/mu^n:
//   K >= 1:  l^(K) = (-1)^K (K - 1)!/mu^K (1 - K u)
// (the terms -log mu give (-1)^K (K - 1)!/mu^K, the term -u gives -(-1)^K K! u/mu^K).
// Expected information I = 1/mu^2 (the score has variance Var(u)/mu^2 = 1/mu^2), with
//   I^(K) = (-1)^K (K + 1)!/mu^(K+2).
//
// The cdf is F(y) = 1 - exp(-u), so log(1 - F) = -u exactly, with
//   d^K log(1 - F)/dmu^K = (-1)^(K+1) K! u/mu^K,
// and log F = G(u) with G(u) = log(1 - e^(-u)). The derivatives of log F in mu are
// the chain rule of G at u(mu) (laplace::math::chain_rule_dK, the shared Faa di Bruno
// machinery), and G is split at u = u0 = 1 for the derivatives in u:
//
//   u >= 1:  G^(k)(u) = (-1)^(k+1) q A_(k-1)(q)/(1 - q)^k, with q = e^(-u) and the Eulerian
//            polynomials A_m (A_0 = 1, A_1 = 1, A_2 = 1 + q, A_3 = 1 + 4q + q^2, ...), from
//            G^(k) = (-1)^(k+1) sum_(n >= 1) n^(k-1) q^n = (-1)^(k+1) Li_(1-k)(q); every
//            coefficient is positive, so nothing cancels, and 1 - q = -expm1(-u) is exact.
//   0 < u < 1: G(u) = log u + L(u), L(u) = log((1 - e^(-u))/u) = log exprel(-u), and the
//            pole of G^(k) is the exact derivative of log u, so
//              d^K log F/dmu^K = (-1)^K (K - 1)!/mu^K + sum_k L^(k)(u) B_(K,k)(u^(1), ...),
//            L^(k) by the chain rule of log at z = exprel(-u) (laplace::math::exprel_dK).
//            There L^(k) = O(1) and every term carries a factor u, so the sum is O(u/mu^K)
//            and the pole cancels by an exact identity, not by subtraction.
//
// The value is log F = log(-expm1(-u)): expm1 keeps the relative accuracy of 1 - e^(-u), so the
// value needs no split.
//
// Expected information derivatives and every other closed form are checked with SymPy in
// tools/reference/distrib_expchi_forms.py.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/exprel.hpp>
#include <laplace/math/faa_di_bruno.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_exponential {

// the switch between the two routes for log F in u (see the file comment)
inline constexpr double u_split = 1.0;

LAPLACE_HD_FORCEINLINE double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

// (-1)^n as a double
LAPLACE_HD_FORCEINLINE double sgn(int n) { return (n % 2 == 0) ? 1.0 : -1.0; }

// d^K f(g(x)) by laplace::math::chain_rule_dK, K = 1..6
template <int K>
LAPLACE_HD_FORCEINLINE double chain_order(const double* f, const double* g) {
  if constexpr (K == 1) return math::chain_rule_d1(f, g);
  if constexpr (K == 2) return math::chain_rule_d2(f, g);
  if constexpr (K == 3) return math::chain_rule_d3(f, g);
  if constexpr (K == 4) return math::chain_rule_d4(f, g);
  if constexpr (K == 5) return math::chain_rule_d5(f, g);
  if constexpr (K == 6) return math::chain_rule_d6(f, g);
}

// the derivatives of u(mu) = y/mu with respect to mu: g[j] = u^(j) = (-1)^j j! u/mu^j, j = 1..K
template <int K>
LAPLACE_HD_FORCEINLINE void u_derivs(double u, double mu, double* g) {
  g[0] = 0.0;
  double p = u;  // u/mu^j
  for (int j = 1; j <= K; ++j) {
    p /= mu;
    g[j] = sgn(j) * fact(j) * p;
  }
}

// G^(k)(u) = (-1)^(k+1) q A_(k-1)(q)/(1 - q)^k for k = 1..K, u >= u_split (q = e^(-u)), into G[1..K]
template <int K>
LAPLACE_HD_FORCEINLINE void log_one_minus_exp_derivs_large(double u, double* G) {
  const double q = Kokkos::exp(-u);
  const double omq = -Kokkos::expm1(-u);  // 1 - q, exact in its leading digits
  const double A[6] = {
      1.0,                                              // A_0
      1.0,                                              // A_1
      1.0 + q,                                          // A_2
      1.0 + q * (4.0 + q),                              // A_3 = 1 + 4q + q^2
      1.0 + q * (11.0 + q * (11.0 + q)),                // A_4 = 1 + 11q + 11q^2 + q^3
      1.0 + q * (26.0 + q * (66.0 + q * (26.0 + q)))};  // A_5 = 1 + 26q + 66q^2 + 26q^3 + q^4
  double denom = 1.0;
  for (int k = 1; k <= K; ++k) {
    denom /= omq;  // (1 - q)^-k
    G[k] = -sgn(k) * q * A[k - 1] * denom;  // (-1)^(k+1) = -(-1)^k
  }
}

// L^(k)(u) = d^k/du^k log exprel(-u) for k = 1..K, u in (0, u_split), into L[1..K]
template <int K>
LAPLACE_HD void log_exprel_derivs_small(double u, double* L) {
  const double x = -u;
  const double z = math::exprel(x);
  double e[7] = {0, 0, 0, 0, 0, 0, 0};  // e[j] = exprel^(j)(x)
  if constexpr (K >= 1) e[1] = math::exprel_d1(x);
  if constexpr (K >= 2) e[2] = math::exprel_d2(x);
  if constexpr (K >= 3) e[3] = math::exprel_d3(x);
  if constexpr (K >= 4) e[4] = math::exprel_d4(x);
  if constexpr (K >= 5) e[5] = math::exprel_d5(x);
  if constexpr (K >= 6) e[6] = math::exprel_d6(x);
  double f[K + 1], g[K + 1];
  f[0] = 0.0;
  g[0] = 0.0;
  double iz = 1.0;  // z^-k
  for (int k = 1; k <= K; ++k) {
    iz /= z;
    f[k] = sgn(k - 1) * fact(k - 1) * iz;  // d^k log z = (-1)^(k-1) (k - 1)!/z^k
    g[k] = sgn(k) * e[k];                  // d^k/du^k exprel(-u) = (-1)^k exprel^(k)(x)
  }
  if constexpr (K >= 1) L[1] = chain_order<1>(f, g);
  if constexpr (K >= 2) L[2] = chain_order<2>(f, g);
  if constexpr (K >= 3) L[3] = chain_order<3>(f, g);
  if constexpr (K >= 4) L[4] = chain_order<4>(f, g);
  if constexpr (K >= 5) L[5] = chain_order<5>(f, g);
  if constexpr (K >= 6) L[6] = chain_order<6>(f, g);
}

}  // namespace detail_exponential

/**
 * \ingroup distrib_exponential
 * \brief Univariate exponential distribution parameterized by its mean \f$\mu > 0\f$.
 *
 * Implements the continuous density \f$\ell(\mu; y) = -\log\mu - y/\mu\f$ on \f$(0, \infty)\f$, its parameter
 * derivatives to order 6 (closed form), the expected information \f$1/\mu^2\f$ with its derivatives to order 4,
 * and the cumulative distribution functions: \f$\log(1 - F) = -y/\mu\f$ exactly, and \f$\log F\f$ split at
 * \f$u = y/\mu = 1\f$ as described in the file comment.
 * \see distrib_exponential, distrib_core
 */
struct exponential_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (1).
  static constexpr int n_params = 1;
  /// Differentiability order of the log-density in the observation \f$y\f$ (\c infinitely_smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_exponential
   * \brief Evaluates the log-density \f$\log f(y; \mu) = -\log\mu - y/\mu\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-density value, or \f$-\infty\f$ if \f$y \le 0\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    return -math::log(mu) - y / mu;
  }

  /**
   * \ingroup distrib_exponential
   * \brief Evaluates the \f$K\f$-th derivative of \f$\log f\f$ in \f$\mu\f$, \f$(-1)^K (K-1)!\,(1 - K u)/\mu^K\f$ with \f$u = y/\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> (one component).
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpdf_dk(double y) const {
    using detail_exponential::fact;
    using detail_exponential::sgn;
    sym_tensor<1, K> t;
    const double u = y / mu;
    double ip = 1.0;  // mu^-K
    for (int k = 0; k < K; ++k) ip /= mu;
    t[0] = sgn(K) * fact(K - 1) * ip * (1.0 - K * u);
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log f(y; \mu)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth derivative of \f$\log f\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_exponential
   * \brief Mixed partial derivatives \f$\partial_y^J \partial_\mu^K \log f(y; \mu)\f$ with \f$J + K \le 6\f$.
   *
   * The log-density is linear in \f$y\f$, so \f$\partial_y^J\f$ vanishes for \f$J \ge 2\f$, and
   * \f$\partial_y \log f = -1/\mu\f$, whose \f$K\f$-th derivative in \f$\mu\f$ is \f$-(-1)^K K!/\mu^{K+1}\f$.
   * For \f$J = 0\f$, returns the parameter derivatives (\f$K \ge 1\f$) or the value (\f$K = 0\f$).
   * \tparam J Derivative order in \f$y\f$.
   * \tparam K Derivative order in \f$\mu\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<1, K> logpdf_dy_dk(double y) const {
    sym_tensor<1, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
    } else if constexpr (J == 1) {
      double ip = 1.0;  // mu^-(K+1)
      for (int k = 0; k <= K; ++k) ip /= mu;
      t[0] = -detail_exponential::sgn(K) * detail_exponential::fact(K) * ip;
    } else {
      t[0] = 0.0;
    }
    return t;
  }

  /**
   * \ingroup distrib_exponential
   * \brief Expected Fisher information \f$I(\mu) = 1/\mu^2\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = 1.0 / (mu * mu);
    return t;
  }
  /**
   * \ingroup distrib_exponential
   * \brief \f$K\f$-th derivative of the expected information, \f$(-1)^K (K+1)!/\mu^{K+2}\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    double ip = 1.0 / (mu * mu);  // mu^-(K+2)
    for (int k = 0; k < K; ++k) ip /= mu;
    expected_info_tensor<1, K> t;
    t.pair[0][0] = detail_exponential::sgn(K) * detail_exponential::fact(K + 1) * ip;
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} I(\mu)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth derivative of the expected information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_exponential
   * \brief Cumulative distribution function \f$F(y) = 1 - e^{-y/\mu}\f$ (0 for \f$y \le 0\f$).
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const { return y <= 0.0 ? 0.0 : -Kokkos::expm1(-y / mu); }
  /**
   * \ingroup distrib_exponential
   * \brief Complementary cdf \f$1 - F(y) = e^{-y/\mu}\f$ (1 for \f$y \le 0\f$).
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const { return y <= 0.0 ? 1.0 : Kokkos::exp(-y / mu); }
  /**
   * \ingroup distrib_exponential
   * \brief Logarithm of the cdf, \f$\log F(y) = \log(-\operatorname{expm1}(-y/\mu))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability (\f$-\infty\f$ for \f$y \le 0\f$).
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y <= 0.0 ? -distrib_infinity : math::log(-Kokkos::expm1(-y / mu));
  }
  /**
   * \ingroup distrib_exponential
   * \brief Logarithm of the complementary cdf, \f$\log(1 - F(y)) = -y/\mu\f$ exactly.
   * \param y Observation point \f$y\f$.
   * \return Log-probability.
   */
  LAPLACE_HD double log_ccdf(double y) const { return y <= 0.0 ? 0.0 : -y / mu; }

  /**
   * \ingroup distrib_exponential
   * \brief \f$K\f$-th derivative in \f$\mu\f$ of \f$\log(1 - F(y))\f$, \f$(-1)^{K+1} K!\,u/\mu^K\f$ with \f$u = y/\mu\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_ccdf_dk(double y) const {
    using detail_exponential::fact;
    using detail_exponential::sgn;
    sym_tensor<1, K> t;
    const double u = y / mu;
    double ip = 1.0;
    for (int k = 0; k < K; ++k) ip /= mu;
    t[0] = -sgn(K) * fact(K) * u * ip;
    return t;
  }

  /**
   * \ingroup distrib_exponential
   * \brief \f$K\f$-th derivative in \f$\mu\f$ of \f$\log F(y)\f$, for \f$y > 0\f$.
   *
   * With \f$u = y/\mu\f$ and \f$u^{(j)} = (-1)^j j!\,u/\mu^j\f$: for \f$u \ge 1\f$ the chain rule of
   * \f$G^{(k)}(u)\f$ (closed form, `log_one_minus_exp_derivs_large`), for \f$u < 1\f$ the pole term
   * \f$(-1)^K (K-1)!/\mu^K\f$ plus the chain rule of \f$L^{(k)}(u)\f$. Both chain rules are
   * `laplace::math::chain_rule_dK`.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_cdf_dk(double y) const {
    using detail_exponential::chain_order;
    sym_tensor<1, K> t;
    if (!(y > 0.0)) {
      t[0] = math::detail::quiet_nan();
      return t;
    }
    const double u = y / mu;
    double g[K + 1];
    detail_exponential::u_derivs<K>(u, mu, g);
    if (u >= detail_exponential::u_split) {
      double G[K + 1];
      G[0] = 0.0;
      detail_exponential::log_one_minus_exp_derivs_large<K>(u, G);
      t[0] = chain_order<K>(G, g);
    } else {
      double L[K + 1];
      L[0] = 0.0;
      detail_exponential::log_exprel_derivs_small<K>(u, L);
      double ip = 1.0;  // mu^-K
      for (int k = 0; k < K; ++k) ip /= mu;
      t[0] = detail_exponential::sgn(K) * detail_exponential::fact(K - 1) * ip + chain_order<K>(L, g);
    }
    return t;
  }

  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_exponential
   * \brief Quantile function \f$F^{-1}(p) = -\mu \log(1 - p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return -mu * Kokkos::log1p(-p); }

  /// Lower bound of support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\mu\f$).
  LAPLACE_HD double scale() const { return mu; }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
