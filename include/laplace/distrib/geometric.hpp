// geometric_distrib (laplace::distrib): the geometric distribution in its mean
// mu > 0, on y = 0, 1, 2, ... (the number of failures before the first success).
//
//   P(Y = y) = (1 / (1 + mu)) (mu / (1 + mu))^y,
//   l(mu; y) = y log mu - (y + 1) log1p(mu) = -y t - log1p(mu),   t = log1p(1/mu) > 0.
//
// Derivatives of l: (-1)^(K-1) (K-1)! [y / mu^K - (y + 1) / (1 + mu)^K], written as one fraction
//   [y S_K - mu^K] / (mu (1 + mu))^K,   S_K = sum_(j < K) C(K, j) mu^j = (1 + mu)^K - mu^K,
// so no difference of two large terms is formed (only the genuine zero of the derivative remains).
// Expected information 1 / (mu (1 + mu)), its K-th derivative (-1)^K K! B_(K+1) with
// B_K = 1/mu^K - 1/(1 + mu)^K = S_K / (mu (1 + mu))^K.
//
// With n = floor(y) + 1:  1 - F(y) = (mu / (1 + mu))^n = e^u, u = -n t, and F(y) = -expm1(u).
// Derivatives of log(1 - F) = u: (-1)^(K-1) (K-1)! n B_K. Derivatives of log F = g(u), g(u) = log(-expm1(u)):
// by the chain rule (laplace::math::chain_rule_dK) from those of u and
//   g^(k)(u) = -R_(k-1)(q),  q = 1 / expm1(-u) > 0,  R_0 = q,  R_(j+1) = (q + q^2) R_j'(q),
// polynomials in q with positive integer coefficients (derived and checked with SymPy in
// tools/reference/distrib_geometric.py), so nothing cancels as u -> -inf or as u -> 0-.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_geometric {

// S_K(mu) = sum_(j < K) C(K, j) mu^j by Horner (K <= 7)
template <int K>
LAPLACE_HD_FORCEINLINE double s_poly(double mu) {
  double c[K + 1];
  c[0] = 1.0;
  for (int j = 0; j < K; ++j) c[j + 1] = c[j] * (K - j) / (j + 1);  // C(K, j + 1), exact integers
  double s = c[K - 1];
  for (int j = K - 2; j >= 0; --j) s = s * mu + c[j];
  return s;
}

// B_K = 1/mu^K - 1/(1 + mu)^K = S_K / (mu (1 + mu))^K
template <int K>
LAPLACE_HD_FORCEINLINE double bracket(double mu) {
  const double m = mu * (1.0 + mu);
  double p = 1.0;
  for (int k = 0; k < K; ++k) p *= m;
  return s_poly<K>(mu) / p;
}

// log(1 - e^x), x <= 0 (Maechler's switch at -log 2)
LAPLACE_HD_FORCEINLINE double log1mexp(double x) {
  return x > -0.69314718055994530942 ? math::log(-Kokkos::expm1(x)) : Kokkos::log1p(-math::exp(x));
}

// R_j(q), j = 0..5: g^(j+1)(u) = -R_j(q)
LAPLACE_HD_FORCEINLINE double r_poly(int j, double q) {
  switch (j) {
    case 0: return q;
    case 1: return q * (1.0 + q);
    case 2: return q * (1.0 + q * (3.0 + 2.0 * q));
    case 3: return q * (1.0 + q * (7.0 + q * (12.0 + 6.0 * q)));
    case 4: return q * (1.0 + q * (15.0 + q * (50.0 + q * (60.0 + 24.0 * q))));
    default: return q * (1.0 + q * (31.0 + q * (180.0 + q * (390.0 + q * (360.0 + 120.0 * q)))));
  }
}

template <int K>
LAPLACE_HD_FORCEINLINE double chain(const double* f, const double* g) {
  if constexpr (K == 1) return math::chain_rule_d1(f, g);
  else if constexpr (K == 2) return math::chain_rule_d2(f, g);
  else if constexpr (K == 3) return math::chain_rule_d3(f, g);
  else if constexpr (K == 4) return math::chain_rule_d4(f, g);
  else if constexpr (K == 5) return math::chain_rule_d5(f, g);
  else return math::chain_rule_d6(f, g);
}

// the chain rule of order m (1..6) chosen at run time
LAPLACE_HD_FORCEINLINE double chain_at(int m, const double* f, const double* g) {
  switch (m) {
    case 1: return math::chain_rule_d1(f, g);
    case 2: return math::chain_rule_d2(f, g);
    case 3: return math::chain_rule_d3(f, g);
    case 4: return math::chain_rule_d4(f, g);
    case 5: return math::chain_rule_d5(f, g);
    default: return math::chain_rule_d6(f, g);
  }
}

// B_m for m = 1..K into b[m] (b[0] unused)
template <int K>
LAPLACE_HD_FORCEINLINE void brackets(double mu, double* b) {
  b[1] = bracket<1>(mu);
  if constexpr (K >= 2) b[2] = bracket<2>(mu);
  if constexpr (K >= 3) b[3] = bracket<3>(mu);
  if constexpr (K >= 4) b[4] = bracket<4>(mu);
  if constexpr (K >= 5) b[5] = bracket<5>(mu);
  if constexpr (K >= 6) b[6] = bracket<6>(mu);
}

// d^m u / dmu^m = (-1)^(m-1) (m-1)! n B_m into g[m], m = 1..K
template <int K>
LAPLACE_HD_FORCEINLINE void u_derivs(double n, double mu, double* g) {
  double b[K + 1];
  brackets<K>(mu, b);
  double fac = 1.0;  // (-1)^(m-1) (m-1)!
  for (int m = 1; m <= K; ++m) {
    g[m] = fac * n * b[m];
    fac *= -m;
  }
}

// the smallest k >= 0 with log F(k) >= lt (upper = false) or log(1 - F(k)) <= lt (upper = true),
// lt < 0 finite; t = log1p(1/mu). Starts from the closed form and corrects by the predicates.
LAPLACE_HD double quantile_core(double t, double lt, bool upper) {
  // log F(k) >= lt  <=>  u <= log(1 - e^lt);  log(1 - F(k)) <= lt  <=>  u <= lt;  u = -(k + 1) t
  const double target = upper ? lt : log1mexp(lt);
  const double x = target / -t;  // k + 1 >= x
  if (!(x < 1e300)) return distrib_infinity;
  double k = Kokkos::ceil(x) - 1.0;
  if (!(k > 0.0)) k = 0.0;
  auto pred = [&](double kk) {
    const double u = -(kk + 1.0) * t;
    return upper ? u <= lt : log1mexp(u) >= lt;
  };
  for (int it = 0; it < 8; ++it) {
    if (!pred(k)) k += 1.0;
    else if (k > 0.0 && pred(k - 1.0)) k -= 1.0;
    else break;
  }
  return k;
}

}  // namespace detail_geometric

/**
 * \ingroup distrib_geometric
 * \brief Univariate geometric discrete distribution parameterized by mean \f$\mu > 0\f$.
 *
 * Counts the failures \f$y = 0, 1, 2, \dots\f$ before the first success, with mass
 * \f$p(y; \mu) = \frac{1}{1+\mu}\left(\frac{\mu}{1+\mu}\right)^y\f$ and variance \f$\mu(1+\mu)\f$.
 * Every member is closed: the log-mass and its derivatives to order 6, the expected information and its
 * derivatives to order 4, the cumulative distribution function with the derivatives of \f$\log F\f$ and
 * \f$\log(1 - F)\f$ to order 6, and the quantile.
 * \see distrib_geometric, distrib_core
 */
struct geometric_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (1).
  static constexpr int n_params = 1;
  /// Descriptor of \c mu.
  static constexpr distrib_param_info params[1] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_geometric
   * \brief Evaluates the log-mass \f$\log p(y; \mu) = -y\log(1 + 1/\mu) - \log(1 + \mu)\f$.
   *
   * Both terms are negative, so nothing cancels.
   * \param y Observation point \f$y \ge 0\f$.
   * \return Log-mass value.
   * \see distrib_geometric, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const { return -y * Kokkos::log1p(1.0 / mu) - Kokkos::log1p(mu); }

  /**
   * \ingroup distrib_geometric
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu)\f$ with respect to \f$\mu\f$.
   *
   * \f$(-1)^{K-1}(K-1)!\,[\,y S_K - \mu^K\,]/(\mu(1+\mu))^K\f$ with \f$S_K = (1+\mu)^K - \mu^K\f$ expanded
   * as a polynomial with positive coefficients.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \ge 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> logpmf_dk(double y) const {
    const double m = mu * (1.0 + mu);
    double p = 1.0, mk = 1.0;  // (mu (1 + mu))^K, mu^K
    for (int k = 0; k < K; ++k) p *= m, mk *= mu;
    double d = (y * detail_geometric::s_poly<K>(mu) - mk) / p;
    for (int k = 1; k < K; ++k) d *= -k;
    sym_tensor<1, K> t;
    t[0] = d;
    return t;
  }
  /**
   * \name Derivatives of the log-mass
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log p(y; \mu)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> logpmf_d1(double y) const { return logpmf_dk<1>(y); }
  /// Second-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> logpmf_d2(double y) const { return logpmf_dk<2>(y); }
  /// Third-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> logpmf_d3(double y) const { return logpmf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> logpmf_d4(double y) const { return logpmf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> logpmf_d5(double y) const { return logpmf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log p\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> logpmf_d6(double y) const { return logpmf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_geometric
   * \brief Evaluates the expected Fisher information \f$I(\mu) = 1/(\mu(1+\mu))\f$.
   * \return Symmetric tensor \c sym_tensor<1, 2> with \f$I_{\mu\mu}\f$ at index 0.
   */
  LAPLACE_HD sym_tensor<1, 2> expected_info() const {
    sym_tensor<1, 2> t;
    t[0] = detail_geometric::bracket<1>(mu);
    return t;
  }
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates the \f$K\f$-th derivative of the expected Fisher information with respect to \f$\mu\f$.
   *
   * \f$(-1)^K K!\,B_{K+1}\f$ with \f$B_m = \mu^{-m} - (1+\mu)^{-m}\f$ written as one fraction.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<1, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<1, K> expected_info_dk() const {
    double d = detail_geometric::bracket<K + 1>(mu);
    for (int k = 1; k <= K; ++k) d *= -k;
    expected_info_tensor<1, K> t;
    t.pair[0][0] = d;
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} I(\mu)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivative of expected Fisher information in \f$\mu\f$.
  LAPLACE_HD expected_info_tensor<1, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  // F(y) = P(Y <= y) for any real y (0 below 0); 1 - F(y) = e^(-(floor(y) + 1) t)
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates the cumulative distribution function \f$F(y) = \mathbb{P}(Y \le y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$F(y) = 1 - (\mu/(1+\mu))^{\lfloor y \rfloor + 1}\f$ for \f$y \ge 0\f$, 0 below.
   */
  LAPLACE_HD double cdf(double y) const {
    return y < 0.0 ? 0.0 : -Kokkos::expm1(-(Kokkos::floor(y) + 1.0) * Kokkos::log1p(1.0 / mu));
  }
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = \mathbb{P}(Y > y)\f$.
   * \param y Observation point \f$y\f$.
   * \return \f$(\mu/(1+\mu))^{\lfloor y \rfloor + 1}\f$ for \f$y \ge 0\f$, 1 below.
   */
  LAPLACE_HD double ccdf(double y) const {
    return y < 0.0 ? 1.0 : math::exp(-(Kokkos::floor(y) + 1.0) * Kokkos::log1p(1.0 / mu));
  }
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates \f$\log F(y) = \log(-\mathrm{expm1}(-(\lfloor y \rfloor + 1)t))\f$, \f$t = \log(1 + 1/\mu)\f$.
   *
   * Returns \f$-\infty\f$ when \f$y < 0\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   * \see distrib_geometric, cdf(), log_cdf_dk()
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y < 0.0 ? -distrib_infinity
                   : detail_geometric::log1mexp(-(Kokkos::floor(y) + 1.0) * Kokkos::log1p(1.0 / mu));
  }
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates \f$\log(1 - F(y)) = -(\lfloor y \rfloor + 1)\log(1 + 1/\mu)\f$.
   *
   * Returns \f$0\f$ when \f$y < 0\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   * \see distrib_geometric, ccdf(), log_ccdf_dk()
   */
  LAPLACE_HD double log_ccdf(double y) const {
    return y < 0.0 ? 0.0 : -(Kokkos::floor(y) + 1.0) * Kokkos::log1p(1.0 / mu);
  }

  // d^K log F / dmu^K and d^K log(1 - F) / dmu^K at y (floored); below 0, log F = -inf
  // (NaN derivatives) and log(1 - F) = 0
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ with respect to \f$\mu\f$.
   *
   * Composes the derivatives of \f$g(u) = \log(-\mathrm{expm1}(u))\f$ (positive-coefficient polynomials in
   * \f$q = 1/\mathrm{expm1}(-u)\f$) with those of \f$u = -(\lfloor y \rfloor + 1)\log(1 + 1/\mu)\f$.
   * Fills quiet NaN when \f$y < 0\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   * \see distrib_geometric, log_cdf()
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_cdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y < 0.0) {
      t[0] = math::detail::quiet_nan();
      return t;
    }
    const double n = Kokkos::floor(y) + 1.0;
    const double lt = Kokkos::log1p(1.0 / mu);  // t
    const double s = n * lt;                    // -u
    if (s <= 0.69314718055994530942) {
      // F <= 1/2: log F = g(u), u = -n t; g^(k) = -R_(k-1)(q), q = 1/expm1(s)
      const double q = 1.0 / Kokkos::expm1(s);
      double f[K + 1], g[K + 1];
      for (int k = 1; k <= K; ++k) f[k] = -detail_geometric::r_poly(k - 1, q);
      detail_geometric::u_derivs<K>(n, mu, g);
      t[0] = detail_geometric::chain<K>(f, g);
      return t;
    }
    // F > 1/2: log F = log(1 - w), w = r^n, r = mu/(1 + mu). The derivatives of w come from the chain rule
    // of x^n at r (r^(m) = (-1)^(m-1) m! v^(m+1), v = 1/(1 + mu): every Bell polynomial of these has terms of
    // one sign), those of log(1 - w) from -(k-1)!/F^k; with u in place of w the sum would cancel by a
    // factor 1/mu^K at small mu.
    const double v = 1.0 / (1.0 + mu);
    double gr[K + 1], fx[K + 1], wd[K + 1], hk[K + 1];
    double vp = v * v, fact = 1.0, sgn = 1.0, falling = 1.0;
    for (int m = 1; m <= K; ++m) {
      gr[m] = sgn * fact * vp;  // (-1)^(m-1) m! v^(m+1)
      vp *= v, fact *= (m + 1), sgn = -sgn;
      falling *= (n - (m - 1));  // n (n - 1) ... (n - m + 1), 0 once m > n
      fx[m] = falling == 0.0 ? 0.0 : falling * math::exp(-(n - m) * lt);  // (n)_m r^(n-m)
    }
    for (int m = 1; m <= K; ++m) wd[m] = detail_geometric::chain_at(m, fx, gr);
    const double F = -Kokkos::expm1(-s);
    hk[1] = -1.0 / F;
    for (int k = 2; k <= K; ++k) hk[k] = hk[k - 1] * (k - 1) / F;
    t[0] = detail_geometric::chain<K>(hk, wd);
    return t;
  }
  /**
   * \ingroup distrib_geometric
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ with respect to \f$\mu\f$.
   *
   * \f$(-1)^{K-1}(K-1)!\,(\lfloor y \rfloor + 1)\,B_K\f$ with \f$B_K = \mu^{-K} - (1+\mu)^{-K}\f$ written as one fraction.
   * Returns \f$0\f$ when \f$y < 0\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y\f$.
   * \return Packed symmetric tensor \c sym_tensor<1, K> of derivatives.
   * \see distrib_geometric, log_ccdf()
   */
  template <int K>
  LAPLACE_HD sym_tensor<1, K> log_ccdf_dk(double y) const {
    sym_tensor<1, K> t;
    if (y < 0.0) return t;
    double d = (Kokkos::floor(y) + 1.0) * detail_geometric::bracket<K>(mu);
    for (int k = 1; k < K; ++k) d *= -k;
    t[0] = d;
    return t;
  }
  /**
   * \name Derivatives of the log-CDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log F(y)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log F\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Derivatives of the log-CCDF
   * Parameter derivatives \f$\frac{\mathrm{d}^K}{\mathrm{d}\mu^K} \log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order derivative of \f$\log(1 - F)\f$ in \f$\mu\f$.
  LAPLACE_HD sym_tensor<1, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_geometric
   * \brief Quantile: the smallest integer \f$k \ge 0\f$ with \f$F(k) \ge p\f$.
   *
   * Starts from \f$\lceil \log(1 - p)/\log(\mu/(1+\mu)) \rceil - 1\f$ and corrects it by the predicates the
   * generic search uses (\f$\log F(k) \ge \log p\f$ for \f$p \le 1/2\f$, \f$\log(1 - F(k)) \le \log(1 - p)\f$ above),
   * so it returns the same integer as quantile_by_search().
   * \param p Probability \f$p \in [0, 1]\f$; NaN outside.
   * \return Quantile; 0 at \f$p = 0\f$ and \f$+\infty\f$ at \f$p = 1\f$.
   */
  LAPLACE_HD double quantile(double p) const {
    if (!(p > 0.0)) return p == 0.0 ? 0.0 : math::detail::quiet_nan();
    if (!(p < 1.0)) return p == 1.0 ? distrib_infinity : math::detail::quiet_nan();
    const double t = Kokkos::log1p(1.0 / mu);
    return p <= 0.5 ? detail_geometric::quantile_core(t, math::log(p), false)
                    : detail_geometric::quantile_core(t, Kokkos::log1p(-p), true);
  }
  /**
   * \ingroup distrib_geometric
   * \brief Quantile from \f$\log p\f$, the same integer as quantile_by_search_from_log().
   * \param log_p Natural logarithm of the probability, in \f$(-\infty, 0]\f$; NaN above 0.
   * \return Quantile.
   */
  LAPLACE_HD double quantile_from_log(double log_p) const {
    if (!(log_p < 0.0)) return log_p == 0.0 ? distrib_infinity : math::detail::quiet_nan();
    if (!(log_p > -distrib_infinity)) return 0.0;
    const double t = Kokkos::log1p(1.0 / mu);
    return log_p <= -0.69314718055994530942 ? detail_geometric::quantile_core(t, log_p, false)
                                            : detail_geometric::quantile_core(t, detail_geometric::log1mexp(log_p), true);
  }
  /**
   * \ingroup distrib_geometric
   * \brief Upper quantile from \f$\log q = \log(1 - p)\f$: the smallest integer \f$k\f$ with \f$\log(1 - F(k)) \le \log q\f$.
   * \param log_q Natural logarithm of the upper tail probability, in \f$(-\infty, 0]\f$; NaN above 0.
   * \return Quantile.
   */
  LAPLACE_HD double quantile_from_log_upper(double log_q) const {
    if (!(log_q < 0.0)) return log_q == 0.0 ? 0.0 : math::detail::quiet_nan();
    if (!(log_q > -distrib_infinity)) return distrib_infinity;
    const double t = Kokkos::log1p(1.0 / mu);
    return log_q <= -0.69314718055994530942 ? detail_geometric::quantile_core(t, log_q, true)
                                            : detail_geometric::quantile_core(t, detail_geometric::log1mexp(log_q), false);
  }

  /// \ingroup distrib_geometric
  /// \brief Lower bound of discrete support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_geometric
  /// \brief Upper bound of discrete support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \ingroup distrib_geometric
  /// \brief Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_geometric
  /// \brief Typical spread (\f$\sqrt{\mu(1+\mu)}\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu * (1.0 + mu)); }
  /// \ingroup distrib_geometric
  /// \brief Non-smooth points in parameter \f$\mu\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /**
   * \ingroup distrib_geometric
   * \brief Starting value from a sample summary: \f$\mu = \bar{y}\f$ (`s.mean`), the maximum likelihood estimate, kept strictly inside \f$(0, \infty)\f$.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c geometric_distrib with parameters strictly inside their open domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  // Starting value from a sample (generic_start.hpp): the sample mean, the MLE; strictly inside the domain.
  geometric_distrib start_value(const distrib_sample_summary& s) const { return {start_inside(s.mean, params[0])}; }

};

}  // namespace laplace::distrib
