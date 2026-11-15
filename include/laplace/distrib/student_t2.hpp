// student_t2_distrib (laplace::distrib): the Student t distribution in its location mu, its
// standard deviation sigma > 0 and its degrees of freedom nu > 2, so that Var Y = sigma^2 exactly
// (for every nu > 2; the mean is mu for nu > 1). It is student_t1 at the scale
//
//   s = sigma r(nu),   r(nu) = sqrt((nu - 2)/nu),
//
// l(mu, sigma, nu; y) = l1(mu, s(sigma, nu), nu; y), F(y) = T_nu((y - mu)/s). No kinks.
//
// Composition. The derivatives of l, of log F and of log(1 - F) in (mu, sigma, nu) are those of
// student_t1 in (mu, s, nu) carried through the inner map (s, nu) = (sigma r(nu), nu). mu is not
// touched by the map: for each number a of derivatives in mu, the remaining order K - a goes
// through one math::chain_rule_pair_dk<K - a> call, whose outer array is
// G[(i, j)] = d^(a + i + j) l1 / dmu^a ds^i dnu^j (the components of student_t1's tensors of order
// a + i + j), whose inner maps are u0 = s and u1 = nu in the variables (sigma, nu):
//   d^p/dsigma^p d^q/dnu^q s = sigma r^(q) (p = 0), r^(q) (p = 1), 0 (p >= 2),
// and r^(q) comes from r^2 = 1 - 2/nu by Leibniz's rule,
//   r^(q) = [-2 (-1)^q q! nu^-(q+1) - sum_(i=1)^(q-1) C(q, i) r^(i) r^(q-i)] / (2 r),
// whose terms all have one sign (no cancellation, also as nu tends to 2, where r tends to 0 and
// the derivatives in nu carry (nu - 2)^(1/2 - q)).
//
// The expected information is the Jacobian transform J' I1 J of student_t1's, which simplifies
// (r cancels in the entries with sigma) to closed forms in nu alone times powers of sigma:
//   I_mumu = nu (nu + 1)/((nu + 3)(nu - 2) sigma^2) = (1 + 6/((nu + 3)(nu - 2)))/sigma^2,
//   I_sigmasigma = 2 nu/((nu + 3) sigma^2),
//   I_sigmanu = 6/((nu + 3)(nu - 2)(nu + 1) sigma),
//   I_nunu = I1_nunu + 2/((nu + 3) nu (nu - 2)^2) - 4/(nu (nu - 2)(nu + 1)(nu + 3)),
// I_musigma = I_munu = 0 (tools/reference/distrib_student_t.py derives them with SymPy from
// J' I1 J). The nu-derivatives of the rational entries are sums of positive products of powers of
// 1/(nu + alpha) (detail_student_t1::inv_prod_deriv), and I1_nunu is student_t1's.
//
// No closed quantile: the generic route applies, as for student_t1.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/student_t1.hpp>
#include <laplace/math/chain_pair.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_student_t2 {

using detail_student_t1::fact;
using detail_student_t1::pos3;
using detail_student_t1::sign_of_power;

// size of the packed tensors of order up to 6 in three parameters
inline constexpr int tensor_max = 28;

// r[q] = d^q/dnu^q sqrt((nu - 2)/nu), q = 0..K
template <int K>
LAPLACE_HD void sqrt_ratio_derivs(double nu, double* r) {
  r[0] = Kokkos::sqrt((nu - 2.0) / nu);
  double inu = 1.0 / nu;  // nu^-(q+1)
  for (int q = 1; q <= K; ++q) {
    inu /= nu;
    double acc = -2.0 * sign_of_power(q) * fact(q) * inu;
    for (int i = 1; i < q; ++i) acc -= binomial_coef(q, i) * r[i] * r[q - i];
    r[q] = acc / (2.0 * r[0]);
  }
}

// u0[pair_index(p, q)] = d^p/dsigma^p d^q/dnu^q (sigma r(nu)), 1 <= p + q <= K; u1 = nu
template <int K>
LAPLACE_HD void inner_maps(double sigma, double nu, double* u0, double* u1) {
  using math::detail_chain_pair::pair_index;
  double r[K + 1];
  sqrt_ratio_derivs<K>(nu, r);
  for (int q = 1; q <= K; ++q) u0[pair_index(0, q)] = sigma * r[q];
  for (int q = 0; q + 1 <= K; ++q) u0[pair_index(1, q)] = r[q];
  u1[pair_index(0, 1)] = 1.0;
}

// the tensors of student_t1 at (mu, s, nu), orders 0..K, one per call of get<M>
template <int M, class Src>
LAPLACE_HD void load(const Src& src, double* T) {
  const sym_tensor<3, M> t = src.template get<M>();
  for (int i = 0; i < sym_tensor<3, M>::size; ++i) T[M * tensor_max + i] = t[i];
}

template <int K, class Src>
LAPLACE_HD void load_all(const Src& src, double* T) {
  load<1>(src, T);
  if constexpr (K >= 2) load<2>(src, T);
  if constexpr (K >= 3) load<3>(src, T);
  if constexpr (K >= 4) load<4>(src, T);
  if constexpr (K >= 5) load<5>(src, T);
  if constexpr (K >= 6) load<6>(src, T);
}

// the components with A derivatives in mu, A = 0 .. K, written into t
template <int K, int A>
LAPLACE_HD void compose_block(const double* T, double sigma, double nu, sym_tensor<3, K>& t) {
  using math::detail_chain_pair::pair_index;
  if constexpr (A == K) {
    t[pos3(K, 0, 0)] = T[K * tensor_max + pos3(K, 0, 0)];
  } else {
    constexpr int n = K - A;
    double G[tensor_max] = {}, u0[tensor_max] = {}, u1[tensor_max] = {}, out[n + 1];
    for (int i = 0; i <= n; ++i)
      for (int j = 0; i + j <= n; ++j)
        if (i + j >= 1) G[pair_index(i, j)] = T[(A + i + j) * tensor_max + pos3(A, i, j)];
    inner_maps<n>(sigma, nu, u0, u1);
    math::chain_rule_pair_dk<n>(G, u0, u1, out);
    for (int c = 0; c <= n; ++c) t[pos3(A, n - c, c)] = out[c];
  }
  if constexpr (A < K) compose_block<K, A + 1>(T, sigma, nu, t);
}

template <int K, class Src>
LAPLACE_HD sym_tensor<3, K> compose(const Src& src, double sigma, double nu) {
  double T[(K + 1) * tensor_max];
  load_all<K>(src, T);
  sym_tensor<3, K> t;
  compose_block<K, 0>(T, sigma, nu, t);
  return t;
}

struct logpdf_source {
  student_t1_distrib b;
  double y;
  template <int M>
  LAPLACE_HD sym_tensor<3, M> get() const { return b.logpdf_dk<M>(y); }
};

struct log_cdf_source {
  student_t1_distrib b;
  double y;
  bool upper;
  template <int M>
  LAPLACE_HD sym_tensor<3, M> get() const { return b.log_cdf_tail_dk<M>(y, upper); }
};

// the nu-derivatives, orders 0..Q, of the entries of the information in nu alone
template <int Q>
LAPLACE_HD void info_derivs(double nu, double* a2, double* b, double* e2, double* f2) {
  double a1[Q + 1], e1[Q + 1], f1[Q + 1];
  detail_student_t1::info_ab_e_derivs<Q>(nu, a1, b, e1);
  detail_student_t1::info_nunu_derivs<Q>(nu, f1);
  const double al_a[2] = {-2.0, 3.0};
  const double al_e[3] = {-2.0, 3.0, 1.0};
  const double al_1[4] = {3.0, 0.0, -2.0, -2.0};
  const double al_2[4] = {0.0, -2.0, 1.0, 3.0};
  for (int q = 0; q <= Q; ++q) {
    a2[q] = 6.0 * detail_student_t1::inv_prod_deriv<2>(q, al_a, nu);
    if (q == 0) a2[q] += 1.0;
    e2[q] = 6.0 * detail_student_t1::inv_prod_deriv<3>(q, al_e, nu);
    f2[q] = f1[q] + 2.0 * detail_student_t1::inv_prod_deriv<4>(q, al_1, nu) -
            4.0 * detail_student_t1::inv_prod_deriv<4>(q, al_2, nu);
  }
}

}  // namespace detail_student_t2

/**
 * \ingroup distrib_student_t2
 * \brief Student t distribution parametrized by location, standard deviation, and degrees of freedom.
 *
 * Models a real observation \f$y \in \mathbb{R}\f$ with location (mean) \f$\mu \in \mathbb{R}\f$,
 * standard deviation \f$\sigma > 0\f$, and degrees of freedom \f$\nu > 2\f$. It reparametrizes
 * `student_t1_distrib` with scale \f$s = \sigma\sqrt{(\nu - 2)/\nu}\f$, so the log-density is
 * \f[
 * \ell(\mu, \sigma, \nu; y) = c(\nu) - \log s - \frac{\nu + 1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right),
 * \qquad z = \frac{y - \mu}{s}.
 * \f]
 * Near \f$\nu = 2^+\f$, derivatives of `log_cdf` and `log_ccdf` amplify because
 * \f$\sqrt{(\nu - 2)/\nu}\f$ becomes steep.
 *
 * \see distrib_student_t2, student_t1_distrib, gaussian1_distrib
 */
struct student_t2_distrib {
  /// Location parameter (mean) \f$\mu \in \mathbb{R}\f$.
  double mu;
  /// Standard deviation \f$\sigma > 0\f$.
  double sigma;
  /// Degrees of freedom \f$\nu > 2\f$.
  double nu;

  /// Distribution kind (continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of distribution parameters (\f$3\f$).
  static constexpr int n_params = 3;
  /// Regularity of the log-density with respect to \f$y\f$ (infinitely smooth).
  static constexpr int y_regularity = infinitely_smooth;
  /// Parameter metadata descriptors for `mu`, `sigma`, and `nu`.
  static constexpr distrib_param_info params[3] = {
      {"mu", -distrib_infinity, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"sigma", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"nu", 2.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_student_t2
   * \brief Compute the scale \f$s = \sigma\sqrt{(\nu - 2)/\nu}\f$ of the equivalent `student_t1_distrib`.
   * \return Scale parameter \f$s > 0\f$ for \f$\nu > 2\f$.
   * \see as_t1
   */
  LAPLACE_HD double t1_scale() const { return sigma * Kokkos::sqrt((nu - 2.0) / nu); }
  /**
   * \ingroup distrib_student_t2
   * \brief Construct the equivalent `student_t1_distrib` at \f$(\mu, s, \nu)\f$.
   * \return Scale-parametrized `student_t1_distrib{mu, t1_scale(), nu}`.
   * \see t1_scale, student_t1_distrib
   */
  LAPLACE_HD student_t1_distrib as_t1() const { return student_t1_distrib{mu, t1_scale(), nu}; }

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the log-density \f$\ell(\mu, \sigma, \nu; y)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log-density value at `y` (\f$-\infty\f$ when `y` is \f$\pm\infty\f$).
   * \see logpdf_dk, log_cdf, expected_info
   */
  LAPLACE_HD double logpdf(double y) const { return as_t1().logpdf(y); }

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log f(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see logpdf, logpdf_d1, logpdf_dy_dk
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dk(double y) const {
    return detail_student_t2::compose<K>(detail_student_t2::logpdf_source{as_t1(), y}, sigma, nu);
  }
  /**
   * \name Log-density parameter derivatives by order
   * \ingroup distrib_student_t2
   * \brief Shorthand accessors calling `logpdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see logpdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<3, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate \f$\partial^J / \partial y^J\f$ composed with order-\f$K\f$ parameter
   *        derivatives of \f$\log f(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam J Observation derivative order (\f$J \ge 0\f$).
   * \tparam K Parameter derivative order (\f$K \ge 0\f$, with \f$J + K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of order \f$K\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \see logpdf_dk
   */
  // l depends on y and mu only through y - mu, so d/dy = -d/dmu (as in student_t1).
  template <int J, int K>
  LAPLACE_HD sym_tensor<3, K> logpdf_dy_dk(double y) const {
    if constexpr (J == 0) {
      if constexpr (K == 0) {
        sym_tensor<3, 0> t;
        t[0] = logpdf(y);
        return t;
      } else {
        return logpdf_dk<K>(y);
      }
    } else {
      const sym_tensor<3, J + K> t = logpdf_dk<J + K>(y);
      sym_tensor<3, K> r;
      const double s = detail_student_t2::sign_of_power(J);
      for (int b = 0; b <= K; ++b)
        for (int c = 0; b + c <= K; ++c)
          r[detail_student_t2::pos3(K - b - c, b, c)] = s * t[detail_student_t2::pos3(K - b - c + J, b, c)];
      return r;
    }
  }

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the expected Fisher information matrix in \f$(\mu, \sigma, \nu)\f$.
   * \return Symmetric \f$3 \times 3\f$ Fisher information matrix stored in `sym_tensor<3, 2>`.
   * \see expected_info_dk, logpdf_dk
   */
  LAPLACE_HD sym_tensor<3, 2> expected_info() const {
    double a[1], b[1], e[1], f[1];
    detail_student_t2::info_derivs<0>(nu, a, b, e, f);
    const double is2 = 1.0 / (sigma * sigma);
    sym_tensor<3, 2> t;
    t[0] = a[0] * is2;
    t[3] = b[0] * is2;
    t[4] = e[0] / sigma;
    t[5] = f[0];
    return t;
  }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate order-\f$K\f$ parameter derivatives of the expected Fisher information in
   *        \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Expected-information tensor of order \f$K\f$.
   * \see expected_info, expected_info_d1
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<3, K> expected_info_dk() const {
    double a[K + 1], b[K + 1], e[K + 1], f[K + 1];
    detail_student_t2::info_derivs<K>(nu, a, b, e, f);
    expected_info_tensor<3, K> t;
    detail_student_t1::fill_info<K>(a, b, e, f, sigma, t);
    return t;
  }
  /**
   * \name Expected-information parameter derivatives by order
   * \ingroup distrib_student_t2
   * \brief Shorthand accessors calling `expected_info_dk<K>()` for \f$K = 1, \dots, 4\f$.
   * \return Expected-information tensor of the corresponding order.
   * \see expected_info_dk
   * @{
   */
  /// First-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected information.
  LAPLACE_HD expected_info_tensor<3, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the cumulative distribution function \f$F(y) = T_\nu((y - \mu)/s)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Lower-tail probability in \f$[0, 1]\f$ (\f$0\f$ at \f$-\infty\f$, \f$1\f$ at \f$+\infty\f$).
   * \see ccdf, log_cdf
   */
  LAPLACE_HD double cdf(double y) const { return as_t1().cdf(y); }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the complementary cumulative distribution function \f$1 - F(y) = T_\nu(-(y - \mu)/s)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Upper-tail probability in \f$[0, 1]\f$ (\f$1\f$ at \f$-\infty\f$, \f$0\f$ at \f$+\infty\f$).
   * \see cdf, log_ccdf
   */
  LAPLACE_HD double ccdf(double y) const { return as_t1().ccdf(y); }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the log cumulative distribution function \f$\log F(y)\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log lower-tail probability in \f$[-\infty, 0]\f$ (\f$-\infty\f$ at \f$-\infty\f$, \f$0\f$ at \f$+\infty\f$).
   * \see log_ccdf, log_cdf_dk, cdf
   */
  LAPLACE_HD double log_cdf(double y) const { return as_t1().log_cdf(y); }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate the log complementary cumulative distribution function \f$\log(1 - F(y))\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Log upper-tail probability in \f$[-\infty, 0]\f$ (\f$0\f$ at \f$-\infty\f$, \f$-\infty\f$ at \f$+\infty\f$).
   * \see log_cdf, log_ccdf_dk, ccdf
   */
  LAPLACE_HD double log_ccdf(double y) const { return as_t1().log_ccdf(y); }

  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$
   *        in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \param upper If `false`, differentiate \f$\log F(y)\f$; if `true`, differentiate \f$\log(1 - F(y))\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_dk, log_ccdf_dk
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_tail_dk(double y, bool upper) const {
    return detail_student_t2::compose<K>(detail_student_t2::log_cdf_source{as_t1(), y, upper}, sigma, nu);
  }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log F(y)\f$ in \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_tail_dk, log_cdf, log_cdf_d1
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_cdf_dk(double y) const { return log_cdf_tail_dk<K>(y, false); }
  /**
   * \ingroup distrib_student_t2
   * \brief Evaluate order-\f$K\f$ parameter derivatives of \f$\log(1 - F(y))\f$ in
   *        \f$(\mu, \sigma, \nu)\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of order \f$K\f$.
   * \see log_cdf_tail_dk, log_ccdf, log_ccdf_d1
   */
  template <int K>
  LAPLACE_HD sym_tensor<3, K> log_ccdf_dk(double y) const { return log_cdf_tail_dk<K>(y, true); }
  /**
   * \name Log-CDF parameter derivatives by order
   * \ingroup distrib_student_t2
   * \brief Shorthand accessors calling `log_cdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see log_cdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 1> log_cdf_d1(double y) const { return log_cdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 2> log_cdf_d2(double y) const { return log_cdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 3> log_cdf_d3(double y) const { return log_cdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 4> log_cdf_d4(double y) const { return log_cdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 5> log_cdf_d5(double y) const { return log_cdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log F\f$.
  LAPLACE_HD sym_tensor<3, 6> log_cdf_d6(double y) const { return log_cdf_dk<6>(y); }
  /** @} */
  /**
   * \name Log-CCDF parameter derivatives by order
   * \ingroup distrib_student_t2
   * \brief Shorthand accessors calling `log_ccdf_dk<K>(y)` for \f$K = 1, \dots, 6\f$.
   * \param y Observation value \f$y \in \mathbb{R}\f$.
   * \return Symmetric tensor of partial derivatives of the corresponding order.
   * \see log_ccdf_dk
   * @{
   */
  /// First-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 1> log_ccdf_d1(double y) const { return log_ccdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 2> log_ccdf_d2(double y) const { return log_ccdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 3> log_ccdf_d3(double y) const { return log_ccdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 4> log_ccdf_d4(double y) const { return log_ccdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 5> log_ccdf_d5(double y) const { return log_ccdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log(1 - F)\f$.
  LAPLACE_HD sym_tensor<3, 6> log_ccdf_d6(double y) const { return log_ccdf_dk<6>(y); }
  /** @} */

  /// Lower bound of the support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of the support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Typical spread (standard deviation \f$\sigma\f$).
  LAPLACE_HD double scale() const { return sigma; }
  /**
   * \ingroup distrib_student_t2
   * \brief Return parameter-domain non-smooth points (none for Student t).
   * \return Empty kink list.
   * \see y_kinks
   */
  LAPLACE_HD distrib_kinks param_kinks(int /*j*/, double /*y*/) const { return distrib_kinks{}; }
  /// Return observation-domain non-smooth points (none for Student t).
  LAPLACE_HD distrib_kinks y_kinks() const { return distrib_kinks{}; }
  /**
   * \ingroup distrib_student_t2
   * \brief Construct a starting parameter vector inside the domain from sample summaries.
   * \param s Sample summary statistics (`mean`, `variance`).
   * \return Valid starting `student_t2_distrib` with sample mean, standard deviation, and \f$\nu = 5\f$.
   * \see distrib_start
   */
  // Starting value from a sample (generic_start.hpp): the sample mean and standard deviation (the
  // parameters are the mean and the standard deviation themselves) and nu = 5; strictly inside the domain.
  student_t2_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.mean, params[0]), start_inside(Kokkos::sqrt(s.variance), params[1]),
            start_inside(5.0, params[2])};
  }
};

}  // namespace laplace::distrib
