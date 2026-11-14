// gamma2_distrib (laplace::distrib): the gamma distribution in its mean mu > 0
// and its variance sigma2 > 0 (shape s = mu^2/sigma2, scale sigma2/mu, rate
// mu/sigma2), so that the parameters are (mu, sigma2) and phi = sigma2/mu^2 = 1/s.
//
//   l(mu, sigma2; y) = -s D(c) + (1/2) log s - delta(s) - log y - (1/2) log(2 pi),
//   D(c) = c - 1 - log c >= 0,  c = y/mu,  s = mu^2/sigma2,
//
// with delta Stirling's error (laplace::math::stirling_error). This is gamma1's
// form (gamma1_distrib in gamma1.hpp) with the inner map s(mu, sigma2) in place of
// 1/phi; it has no cancellation at large s. The deviance D and the Stirling error
// are shared with gamma1 (detail_gamma1::deviance_unit, detail_gamma1::stirling_upto).
//
// Composition. Every quantity below is G(u_0, u_1) with an outer function G of two
// arguments and two inner maps of (mu, sigma2) (laplace::math::chain_rule_pair_dk):
//
//   log density:      G(s, c) = -s D(c) + h(s),  h(s) = (1/2) log s - delta(s),
//                     inner maps s(mu, sigma2) and c(mu) = y/mu;
//   log F, log(1 - F): G(a, c) = log P(a, a c) (or Q), the mixed partials R_ij of
//                     detail_gamma_cdf::ratio_order, inner maps s and c as above;
//   y-derivatives:    G(s, mu) with the terms -s y/mu + (s - 1) log y, inner maps s and mu;
//   information:      H_ab(mu, s) (below), inner maps mu and s.
//
// The inner map s depends on both parameters, so a one-variable Bell sum per
// parameter (gamma1's route) does not apply; chain_rule_pair_dk does the mixed
// composition for both inner maps at once.
//
// Expected information. Gamma1's (mu, phi) information is I_mumu = s/mu^2,
// I_muphi = 0, I_phiphi = s^2/2 + s^4 delta''(s). Carried to (mu, sigma2) by the
// Jacobian J = d(mu, phi)/d(mu, sigma2), J_(mu,mu) = 1, J_(phi,mu) = -2 phi/mu,
// J_(phi,sigma2) = 1/mu^2, J_(mu,sigma2) = 0, one gets I_new = J' I_old J, whose
// entries are, with phi = 1/s,
//
//   H_mm(mu, s) = (s + 2 + 4 s^2 delta''(s)) / mu^2,
//   H_ms(mu, s) = -(s + 2 s^3 delta''(s)) / mu^3,
//   H_ss(mu, s) = (s^2/2 + s^4 delta''(s)) / mu^4,
//
// functions of (mu, s) alone. The derivatives of H_ab in mu and s are written in
// closed form (power of mu times a function of s, the s-derivatives of that
// function by Leibniz's rule over s^n delta''(s)) and composed with s(mu, sigma2).
// This is the Leibniz expansion of J' I_old J over the entries of J, with the
// chain rule on the entries of I_old, carried out once.
//
// F(y) = P(s, s c) = P(mu^2/sigma2, mu y/sigma2), the regularized lower incomplete
// gamma function, with the scale sigma2/mu and the quantile (sigma2/mu) P^-1(s, p).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/gamma_cdf.hpp>
#include <laplace/distrib/gamma1.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_gamma.hpp>
#include <laplace/math/incomplete_inverse.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_gamma2 {

// number of multi-indices (i, j) with i + j <= 6, the size of an outer table of total order 6
inline constexpr int n_pairs6 = 28;

LAPLACE_HD_FORCEINLINE double ipow(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}

// u[pair_index(p, q)] = d^(p+q) s / dmu^p dsigma2^q, 1 <= p + q <= K, with s = mu^2 / sigma2:
// the product of the derivatives of mu^2 in mu and of sigma2^-1 in sigma2
template <int K>
LAPLACE_HD_FORCEINLINE void shape_partials(double mu, double sigma2, double* u) {
  using laplace::math::detail_chain_pair::pair_index;
  double qs[K + 1];  // d^q (sigma2^-1) / dsigma2^q = (-1)^q q! sigma2^-(q+1)
  qs[0] = 1.0 / sigma2;
  for (int q = 1; q <= K; ++q) qs[q] = qs[q - 1] * (-q) / sigma2;
  for (int p = 0; p <= K; ++p)
    for (int q = 0; p + q <= K; ++q) {
      if (p + q == 0) continue;
      const double pm = p == 0 ? mu * mu : p == 1 ? 2.0 * mu : p == 2 ? 2.0 : 0.0;  // d^p (mu^2) / dmu^p
      u[pair_index(p, q)] = pm * qs[q];
    }
}

// u[pair_index(p, 0)] = d^p c / dmu^p for c = y/mu, p >= 1 (no dependence on sigma2)
template <int K>
LAPLACE_HD_FORCEINLINE void ratio_partials(double y, double mu, double* u) {
  using laplace::math::detail_chain_pair::pair_index;
  double t = y / mu;  // (-1)^p p! y / mu^(p+1), built by t_p = t_(p-1) (-p) / mu
  for (int p = 1; p <= K; ++p) {
    t *= -p / mu;
    u[pair_index(p, 0)] = t;
  }
}

// G(s, c) = -s D(c) + h(s) of the log-density: G[pair_index(i, j)] = d^(i+j) G / ds^i dc^j, i + j <= K, K >= 1.
// The c-derivatives of -s D(c) are -s D^(j)(c) (i = 0) and -D^(j)(c) (i = 1); above i = 1 the mixed
// terms vanish. D^(1)(c) = (y - mu)/y and D^(j)(c) = (-1)^j (j-1)! (mu/y)^j for j >= 2.
template <int K>
LAPLACE_HD_FORCEINLINE void logpdf_outer(double s, double y, double mu, double* G) {
  using detail_gamma1::fact;
  using laplace::math::detail_chain_pair::pair_index;
  double dl[K + 1];  // delta^(k)(s), k = 1..K
  detail_gamma1::stirling_upto<K>(s, dl);
  double h[K + 1];  // h^(i)(s), i = 1..K: h' = 1/(2s) - delta', h^(i) = (-1)^(i-1) (i-1)! / (2 s^i) - delta^(i)
  double si = 1.0;
  for (int i = 1; i <= K; ++i) {
    si /= s;
    h[i] = ((i % 2 == 1) ? 1.0 : -1.0) * 0.5 * fact(i - 1) * si - dl[i];
  }
  double Dj[K + 1];
  const double r = mu / y;
  Dj[1] = (y - mu) / y;
  double rp = r;  // r^j at step j: starts at r, multiplied before use
  for (int j = 2; j <= K; ++j) {
    rp *= r;
    Dj[j] = ((j % 2 == 0) ? 1.0 : -1.0) * fact(j - 1) * rp;
  }
  const double Dv = detail_gamma1::deviance_unit(y, mu);
  for (int i = 0; i <= K; ++i)
    for (int j = 0; i + j <= K; ++j) {
      double g = 0.0;
      if (j == 0) {
        if (i == 0) g = -s * Dv + 0.5 * math::log(s) - math::stirling_error(s);  // the value
        else if (i == 1) g = -Dv + h[1];
        else g = h[i];
      } else if (i == 0) {
        g = -s * Dj[j];
      } else if (i == 1) {
        g = -Dj[j];
      }
      G[pair_index(i, j)] = g;
    }
}

// G(a, c) = log S(a, a c) of the incomplete gamma ratio, with the mixed partials R[i][j] of
// detail_gamma_cdf::ratio_order: G[pair_index(i, j)] = R[i][j] for i + j = 1..K
template <int K>
LAPLACE_HD_FORCEINLINE void cdf_outer(bool lower, double a, double c, double* G) {
  using laplace::math::detail_chain_pair::pair_index;
  double R[7][7] = {};
  detail_gamma_cdf::ratio_order<1>(lower, a, c, R);
  if constexpr (K >= 2) detail_gamma_cdf::ratio_order<2>(lower, a, c, R);
  if constexpr (K >= 3) detail_gamma_cdf::ratio_order<3>(lower, a, c, R);
  if constexpr (K >= 4) detail_gamma_cdf::ratio_order<4>(lower, a, c, R);
  if constexpr (K >= 5) detail_gamma_cdf::ratio_order<5>(lower, a, c, R);
  if constexpr (K >= 6) detail_gamma_cdf::ratio_order<6>(lower, a, c, R);
  for (int i = 0; i <= K; ++i)
    for (int j = 0; i + j <= K; ++j)
      if (i + j >= 1) G[pair_index(i, j)] = R[i][j];
}

// d^j/ds^j [s^n delta''(s)], n <= 4, j <= 4, from dl[k] = delta^(k)(s), k <= 6, by Leibniz's rule:
// sum_t C(j, t) (n! / (n-t)!) s^(n-t) delta^(2 + j - t)(s)
LAPLACE_HD_FORCEINLINE double stirling_power_term(int n, int j, double s, const double* dl) {
  double r = 0.0, binom = 1.0;  // binom = C(j, t)
  for (int t = 0; t <= j && t <= n; ++t) {
    double falling = 1.0;  // n! / (n - t)!
    for (int k = 0; k < t; ++k) falling *= (n - k);
    r += binom * falling * ipow(s, n - t) * dl[2 + j - t];
    binom = binom * (j - t) / (t + 1);
  }
  return r;
}

// d^j/ds^j of the three coefficients of H (see the header): F_mm = s + 2 + 4 s^2 delta'',
// F_ms = -(s + 2 s^3 delta''), F_ss = s^2/2 + s^4 delta''
LAPLACE_HD_FORCEINLINE double info_mm_s(int j, double s, const double* dl) {
  double r = 4.0 * stirling_power_term(2, j, s, dl);
  if (j == 0) r += s + 2.0;
  else if (j == 1) r += 1.0;
  return r;
}
LAPLACE_HD_FORCEINLINE double info_ms_s(int j, double s, const double* dl) {
  double r = -2.0 * stirling_power_term(3, j, s, dl);
  if (j == 0) r -= s;
  else if (j == 1) r -= 1.0;
  return r;
}
LAPLACE_HD_FORCEINLINE double info_ss_s(int j, double s, const double* dl) {
  double r = stirling_power_term(4, j, s, dl);
  if (j == 0) r += 0.5 * s * s;
  else if (j == 1) r += s;
  else if (j == 2) r += 1.0;
  return r;
}

// (-1)^(J-1) (J-1)! / y^J, the J-th derivative of log y, J >= 1
LAPLACE_HD_FORCEINLINE double log_y_deriv(int J, double y) {
  using detail_gamma1::fact;
  return ((J % 2 == 1) ? 1.0 : -1.0) * fact(J - 1) / ipow(y, J);
}

}  // namespace detail_gamma2

/**
 * \ingroup distrib_gamma2
 * \brief Univariate gamma distribution parameterized by mean \f$\mu > 0\f$ and variance \f$\sigma^2 > 0\f$.
 *
 * Shape \f$s = \mu^2/\sigma^2\f$, scale \f$\sigma^2/\mu\f$. The log-density is
 * \f$\ell = -s D(y/\mu) + \frac{1}{2}\log s - \delta(s) - \log y - \frac{1}{2}\log(2\pi)\f$, with
 * \f$D(c) = c - 1 - \log c\f$ and \f$\delta\f$ Stirling's error. Derivatives in \f$(\mu, \sigma^2)\f$
 * to order 6 come from the outer partials in \f$(s, c)\f$ and the inner maps \f$s(\mu, \sigma^2)\f$,
 * \f$c(\mu)\f$ composed by laplace::math::chain_rule_pair_dk; the expected information is
 * \f$I = J' I_{\mathrm{old}} J\f$ in closed form in \f$(\mu, s)\f$.
 * \see gamma1_distrib, chain_rule_pair_dk, stirling_error
 */
struct gamma2_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Variance parameter \f$\sigma^2 \in (0, \infty)\f$.
  double sigma2;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Differentiability order of the log-density in \f$y\f$ (\c infinitely_smooth on \f$(0, \infty)\f$).
  static constexpr int y_regularity = infinitely_smooth;
  /// Descriptors of \c mu and \c sigma2 (both smooth; the support does not move).
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma2", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the log-density \f$\log f(y; \mu, \sigma^2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-density value, or \f$-\infty\f$ if \f$y \le 0\f$.
   */
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    const double s = mu * mu / sigma2;
    return -s * detail_gamma1::deviance_unit(y, mu) + 0.5 * math::log(s) - math::stirling_error(s) - math::log(y) -
           0.91893853320467274178;
  }

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>; component \f$b\f$ has \f$b\f$ derivatives in \f$\sigma^2\f$.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    const double s = mu * mu / sigma2;
    double G[detail_gamma2::n_pairs6] = {}, u0[detail_gamma2::n_pairs6] = {}, u1[detail_gamma2::n_pairs6] = {};
    double out[K + 1];
    detail_gamma2::logpdf_outer<K>(s, y, mu, G);
    detail_gamma2::shape_partials<K>(mu, sigma2, u0);
    detail_gamma2::ratio_partials<K>(y, mu, u1);
    math::chain_rule_pair_dk<K>(G, u0, u1, out);
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = out[b];
    return t;
  }

  /**
   * \name Derivatives of the log-density
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log f(y; \mu, \sigma^2)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates mixed partial derivatives of \f$\log f\f$ with \f$J\f$ derivatives in \f$y\f$ and order \f$K\f$ in parameters.
   *
   * The terms of \f$\ell\f$ with \f$y\f$ are \f$-s y/\mu + (s - 1)\log y\f$, with \f$s = \mu^2/\sigma^2\f$ a function of both
   * parameters. For \f$J = 1\f$ the function is \f$-s/\mu + (s - 1)/y\f$, whose \f$s\f$-derivative \f$(\mu - y)/(\mu y)\f$ is
   * written in that form; for \f$J \ge 2\f$ it is \f$(s - 1)(-1)^{J-1}(J-1)!/y^J\f$. Both are composed with \f$s(\mu, \sigma^2)\f$
   * and, for \f$J = 1\f$, with \f$\mu\f$ itself, by laplace::math::chain_rule_pair_dk.
   *
   * \tparam J Derivative order with respect to \f$y\f$ (\f$J + K \le 6\f$).
   * \tparam K Derivative order with respect to \f$(\mu, \sigma^2)\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<2, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const double s = mu * mu / sigma2;
      sym_tensor<2, K> t;
      if constexpr (K == 0) {
        if constexpr (J == 1) t[0] = -s / mu + (s - 1.0) / y;
        else t[0] = (s - 1.0) * detail_gamma2::log_y_deriv(J, y);
      } else {
        using detail_gamma2::n_pairs6;
        using laplace::math::detail_chain_pair::pair_index;
        double G[n_pairs6] = {}, u0[n_pairs6] = {}, u1[n_pairs6] = {}, out[K + 1];
        detail_gamma2::shape_partials<K>(mu, sigma2, u0);
        u1[pair_index(1, 0)] = 1.0;  // the second inner map is mu itself
        if constexpr (J == 1) {
          // G(s, m) = -s/m + (s - 1)/y, m = mu
          G[pair_index(0, 0)] = -s / mu + (s - 1.0) / y;
          G[pair_index(1, 0)] = (mu - y) / (mu * y);
          double fm = 1.0 / mu;  // m^-(j+1)
          double sg = 1.0;       // (-1)^j j!
          for (int j = 1; j <= K; ++j) {
            fm /= mu;
            sg *= -j;
            G[pair_index(0, j)] = -s * sg * fm;
            G[pair_index(1, j)] = -sg * fm;
          }
        } else {
          const double k = detail_gamma2::log_y_deriv(J, y);
          G[pair_index(0, 0)] = (s - 1.0) * k;
          G[pair_index(1, 0)] = k;
        }
        math::chain_rule_pair_dk<K>(G, u0, u1, out);
        for (int b = 0; b <= K; ++b) t[b] = out[b];
      }
      return t;
    }
  }

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the expected Fisher information matrix \f$I(\mu, \sigma^2)\f$.
   * \return Symmetric tensor \c sym_tensor<2, 2>: \f$I_{\mu\mu}\f$ at index 0, \f$I_{\mu\sigma^2}\f$ at 1, \f$I_{\sigma^2\sigma^2}\f$ at 2.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    const double s = mu * mu / sigma2, d2 = math::stirling_error_d2(s);
    const double m2 = mu * mu, m3 = m2 * mu, m4 = m2 * m2;
    sym_tensor<2, 2> t;
    t[0] = (s + 2.0 + 4.0 * s * s * d2) / m2;
    t[1] = -(s + 2.0 * s * s * s * d2) / m3;
    t[2] = (0.5 * s * s + s * s * s * s * d2) / m4;
    return t;
  }

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates partial derivatives of order \f$K\f$ of the expected Fisher information in \f$(\mu, \sigma^2)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    using detail_gamma2::n_pairs6;
    using detail_gamma2::ipow;
    using laplace::math::detail_chain_pair::pair_index;
    using detail_gamma1::fact;
    const double s = mu * mu / sigma2;
    double dl[7] = {};
    detail_gamma1::stirling_upto<6>(s, dl);
    double G[3][n_pairs6] = {}, u0[n_pairs6] = {}, u1[n_pairs6] = {};
    u0[pair_index(1, 0)] = 1.0;  // the first inner map is mu itself
    detail_gamma2::shape_partials<K>(mu, sigma2, u1);
    for (int p = 0; p < 3; ++p) {
      const int k = 2 + p;  // H_ab = F_ab(s) mu^-k: k = 2, 3, 4 for mm, ms, ss
      for (int i = 0; i <= K; ++i) {
        // d^i mu^-k = (-1)^i (k + i - 1)! / (k - 1)! mu^-(k + i)
        const double ci = ((i % 2 == 1) ? -1.0 : 1.0) * fact(k + i - 1) / fact(k - 1) / ipow(mu, k + i);
        for (int j = 0; i + j <= K; ++j) {
          const double F = p == 0   ? detail_gamma2::info_mm_s(j, s, dl)
                           : p == 1 ? detail_gamma2::info_ms_s(j, s, dl)
                                    : detail_gamma2::info_ss_s(j, s, dl);
          G[p][pair_index(i, j)] = ci * F;
        }
      }
    }
    expected_info_tensor<2, K> t;
    for (int p = 0; p < 3; ++p) {
      double out[K + 1];
      math::chain_rule_pair_dk<K>(G[p], u0, u1, out);
      for (int b = 0; b <= K; ++b) t.pair[p][b] = out[b];
    }
    return t;
  }

  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} I_{ab}(\mu, \sigma^2)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the cumulative distribution function \f$F(y) = P(s, \mu y / \sigma^2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y \le y)\f$.
   */
  LAPLACE_HD double cdf(double y) const {
    return y <= 0.0 ? 0.0 : math::gamma_p(mu * mu / sigma2, mu * y / sigma2);
  }
  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = Q(s, \mu y / \sigma^2)\f$.
   * \param y Observation point \f$y\f$.
   * \return Probability \f$\mathbb{P}(Y > y)\f$.
   */
  LAPLACE_HD double ccdf(double y) const {
    return y <= 0.0 ? 1.0 : math::gamma_q(mu * mu / sigma2, mu * y / sigma2);
  }
  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the logarithm of the cumulative distribution function, \f$\log F(y)\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log F(y)\f$.
   */
  LAPLACE_HD double log_cdf(double y) const {
    return y <= 0.0 ? -distrib_infinity : math::log_gamma_p(mu * mu / sigma2, mu * y / sigma2);
  }
  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the logarithm of the complementary cumulative distribution function, \f$\log(1 - F(y))\f$.
   * \param y Observation point \f$y\f$.
   * \return Log-probability \f$\log(1 - F(y))\f$.
   */
  LAPLACE_HD double log_ccdf(double y) const {
    return y <= 0.0 ? 0.0 : math::log_gamma_q(mu * mu / sigma2, mu * y / sigma2);
  }

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$ in \f$(\mu, \sigma^2)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param lower true for \f$\log F\f$, false for \f$\log(1 - F)\f$.
   * \param y Observation point \f$y > 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K>.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(bool lower, double y) const {
    const double a = mu * mu / sigma2, c = y / mu;
    double G[detail_gamma2::n_pairs6] = {}, u0[detail_gamma2::n_pairs6] = {}, u1[detail_gamma2::n_pairs6] = {};
    double out[K + 1];
    detail_gamma2::cdf_outer<K>(lower, a, c, G);
    detail_gamma2::shape_partials<K>(mu, sigma2, u0);
    detail_gamma2::ratio_partials<K>(y, mu, u1);
    math::chain_rule_pair_dk<K>(G, u0, u1, out);
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = out[b];
    return t;
  }
  /**
   * \name Derivatives of the log-CDF and log-CCDF
   * Parameter derivatives \f$\frac{\partial^K}{\partial\theta^K} \log F(y)\f$ and \f$\frac{\partial^K}{\partial\theta^K}\log(1 - F(y))\f$, orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(true, y); }
  /// Second-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(true, y); }
  /// Third-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(true, y); }
  /// Fourth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(true, y); }
  /// Fifth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(true, y); }
  /// Sixth-order partial derivatives of \f$\log F\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(true, y); }
  /// First-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return log_cdf_dk<1>(false, y); }
  /// Second-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return log_cdf_dk<2>(false, y); }
  /// Third-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return log_cdf_dk<3>(false, y); }
  /// Fourth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return log_cdf_dk<4>(false, y); }
  /// Fifth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return log_cdf_dk<5>(false, y); }
  /// Sixth-order partial derivatives of \f$\log(1 - F)\f$ in \f$(\mu, \sigma^2)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return log_cdf_dk<6>(false, y); }
  /** @} */

  /**
   * \ingroup distrib_gamma2
   * \brief Evaluates the quantile function \f$F^{-1}(p) = (\sigma^2/\mu)\, P^{-1}(s, p)\f$.
   * \param p Probability \f$p \in (0, 1)\f$.
   * \return Quantile value \f$y\f$.
   */
  LAPLACE_HD double quantile(double p) const { return (sigma2 / mu) * math::gamma_p_inv(mu * mu / sigma2, p); }

  /// Lower bound of support (0.0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (\f$\sigma\f$).
  LAPLACE_HD double scale() const { return Kokkos::sqrt(sigma2); }
  /// Kinks in parameter \f$\theta_j\f$ for observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// Kinks in observation \f$y\f$ (none).
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }
};

}  // namespace laplace::distrib
