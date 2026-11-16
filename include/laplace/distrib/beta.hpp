// beta_distrib (laplace::distrib): the beta distribution in its mean mu in (0, 1) and its
// precision phi > 0, on the support (0, 1),
//
//   alpha = mu phi,  beta = nu phi,  nu = 1 - mu,   alpha + beta = phi,
//   E[Y] = mu,  Var Y = mu nu / (phi + 1),
//   l(mu, phi; y) = lgamma(phi) - lgamma(alpha) - lgamma(beta) + (alpha - 1) log y + (beta - 1) log(1 - y).
//
// The shape maps are linear: d alpha/d mu = phi, d beta/d mu = -phi, d alpha/d phi = mu,
// d beta/d phi = nu, and every second derivative of them vanishes except
// d^2 alpha/dmu dphi = 1 and d^2 beta/dmu dphi = -1. Hence:
//
// * The data enter l linearly in the shapes, so they survive in the derivatives in theta = (mu, phi) only
//   to order two (order 1: phi log(y/(1 - y)) in the mu component, mu log y + nu log(1 - y) in the phi
//   component; order 2: log(y/(1 - y)) in the mixed component). From order three on the derivatives of l
//   are free of y, and they are the observed and the expected derivatives at once.
//
// * Derivatives of order K. With i derivatives in mu and j in phi (i + j = K >= 1), writing
//   F^(m)(alpha) for the m-th derivative of F(alpha) at alpha = mu phi, Leibniz' rule gives
//
//     d_mu^i d_phi^j F(mu phi) = sum_(r = 0)^(min(i, j)) C(i, r) j!/(j - r)! mu^(j-r) phi^(i-r) F^(i+j-r)(alpha)
//
//   (detail_beta::shape_deriv). For F = -lgamma, F^(m) = -psi^(m-1). The beta term is the same with
//   mu -> nu, alpha -> beta and the factor (-1)^i, since d/dmu = -d/dnu; the term lgamma(phi) adds
//   psi^(K-1)(phi) to the pure phi component. The components, derived by SymPy from the definition and
//   checked against that derivation by tools/reference/distrib_beta.py, are therefore sums of polygamma
//   functions at alpha, beta and phi times powers of mu, nu and phi.
//
// * The expected information is the closed form
//     I_mu,mu   = phi^2 (psi_1(alpha) + psi_1(beta)),
//     I_mu,phi  = phi (mu psi_1(alpha) - nu psi_1(beta)),
//     I_phi,phi = mu^2 psi_1(alpha) + nu^2 psi_1(beta) - psi_1(phi).
//   I_mu,mu and I_phi,phi are minus the y-free second derivatives of l, so their derivatives in theta are
//   minus the y-free components of order K + 2 (detail_beta::free_component). I_mu,phi is not: the
//   observed mixed second derivative is R - I_mu,phi with the residual R = log(y/(1 - y)) - psi(alpha) +
//   psi(beta), whose expectation is zero, and its derivatives are those of phi mu psi_1(alpha) and of
//   phi nu psi_1(beta), by Leibniz' rule over the product (detail_beta::mixed_info_term).
//
// * F(y) = I_y(alpha, beta), the regularized incomplete beta function: the value and both tails from
//   math::ibeta, ibetac and their logarithms. The derivatives of log F and log(1 - F) in (mu, phi) are
//   the mixed partials of log I_y(a, b) and log(1 - I_y(a, b)) in the shapes (math::log_ibeta_dK and
//   log_ibetac_dK) composed with the two linear shape maps by math::chain_rule_pair_dk.
//
// * The quantile is the inverse incomplete beta (math::ibeta_inv and its logarithmic forms); the draw is
//   the generic rng_by_inversion over it.
//
// Not finite at the ends: the score in y, (alpha - 1)/y - (beta - 1)/(1 - y), is not finite at y = 0 or
// y = 1. Those points are outside the open support, and logpdf returns -infinity there, but a draw can
// land on them when a shape is small (alpha or beta below about 0.06).
//
// Accuracy. The derivative components are assembled from the polygamma functions as written above, with
// no rearrangement. Two corners lose digits in the components that are small compared with their terms:
//   * large phi: the leading monomials of psi_m(alpha), psi_m(beta) and psi_m(phi) cancel exactly in the
//     pure phi components and nearly so in the mixed ones;
//   * small alpha or beta: the pole terms (-1)^(m+1) m!/alpha^(m+1) cancel in the mixed components.
// The pure phi components are the one case rearranged here: they are assembled from the polygamma
// remainders psi_n - mu_n (math::polygamma_monomial_remainder, detail_beta::pure_phi_combination), so that
// the three leading monomials cancel algebraically and nothing of the result is formed by subtraction.
// That took their error from about eps phi relative, 1.6e8 to 3.9e8 ulp of themselves at phi = 1e8, to at
// most 13.4 ulp at every phi measured. The mixed components and the small-shape corner are not rearranged:
// they would need the same treatment of the pole terms, which no family has asked for yet.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/incomplete_beta.hpp>
#include <laplace/math/incomplete_beta_deriv.hpp>
#include <laplace/math/incomplete_inverse.hpp>
#include <laplace/math/special.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_beta {

// number of multi-indices (i, j) with i + j <= 6, the size of an outer table of total order 6
inline constexpr int n_pairs6 = 28;

LAPLACE_HD_FORCEINLINE double fact(int n) {
  double r = 1.0;
  for (int i = 2; i <= n; ++i) r *= i;
  return r;
}

LAPLACE_HD_FORCEINLINE double ipow(double x, int n) {
  double r = 1.0;
  for (int i = 0; i < n; ++i) r *= x;
  return r;
}

// C(i, r) j!/(j - r)!, the coefficient of Leibniz' rule for d_mu^i d_phi^j of a function of mu phi
LAPLACE_HD_FORCEINLINE double leibniz_coef(int i, int j, int r) {
  double c = 1.0;
  for (int t = 1; t <= r; ++t) c = c * static_cast<double>(i - r + t) / static_cast<double>(t) * static_cast<double>(j - t + 1);
  return c;
}

// d_mu^i d_phi^j F(m phi) as a function of mu with m = mu (or m = nu with d/dmu = -d/dnu applied by the
// caller): sum_r C(i, r) j!/(j - r)! m^(j-r) phi^(i-r) f[i + j - r], where f[k] = F^(k)(m phi)
LAPLACE_HD_FORCEINLINE double shape_deriv(int i, int j, double m, double phi, const double* f) {
  double s = 0.0;
  const int rmax = i < j ? i : j;
  for (int r = 0; r <= rmax; ++r) s += leibniz_coef(i, j, r) * ipow(m, j - r) * ipow(phi, i - r) * f[i + j - r];
  return s;
}

// The pure phi combination mu^(n+1) psi_n(alpha) + nu^(n+1) psi_n(beta) - psi_n(phi) for n >= 1, built
// from the remainders psi_n - mu_n instead of the polygamma functions themselves, with mu_0(x) = log x
// and mu_n(x) = (-1)^(n+1) (n - 1)! x^-n.
//
// The three leading monomials cancel exactly: mu^(n+1) mu_n(mu phi) = (-1)^(n+1) (n - 1)! mu phi^-n and
// the same with nu, so their sum is mu_n(phi) (mu + nu) = mu_n(phi), which is the third term. Subtracted
// in floating point they leave the combination, of size phi^-(n+1), as the difference of terms of size
// phi^-n, and so with about log10(phi) digits gone; dropped algebraically they cost nothing. The identity
// used is mu + nu = 1 with nu the stored 1 - mu, which also defines alpha and beta, so no new
// approximation enters here.
LAPLACE_HD_FORCEINLINE double pure_phi_combination(int n, double mu, double nu, double phi) {
  const double ra = math::polygamma_monomial_remainder(n, mu * phi);
  const double rb = math::polygamma_monomial_remainder(n, nu * phi);
  const double rp = math::polygamma_monomial_remainder(n, phi);
  return ipow(mu, n + 1) * ra + ipow(nu, n + 1) * rb - rp;
}

// the y-free part of d_mu^i d_phi^j l (i + j >= 1) from the tables fa[k] = -psi_(k-1)(alpha) and
// fb[k] = -psi_(k-1)(beta), and psi_phi = psi_(i+j-1)(phi), added when i = 0
LAPLACE_HD_FORCEINLINE double free_component(int i, int j, double mu, double nu, double phi, const double* fa,
                                             const double* fb, double psi_phi) {
  // the pure phi component, where the three terms cancel to one order of phi, from the remainders
  if (i == 0 && j >= 2 && j <= 7) return -pure_phi_combination(j - 1, mu, nu, phi);
  const double sign = (i % 2 == 0) ? 1.0 : -1.0;
  double v = shape_deriv(i, j, mu, phi, fa) + sign * shape_deriv(i, j, nu, phi, fb);
  if (i == 0) v += psi_phi;
  return v;
}

// d_mu^i d_phi^j [ phi q G(q phi) ] with q = mu (s = +1) or q = nu (s = -1), so that dq/dmu = s, from the
// table g[k] = G^(k)(q phi): Leibniz' rule over the product phi q, whose only nonzero derivatives are
// phi q, s phi (one derivative in mu), q (one in phi) and s (one in each)
LAPLACE_HD_FORCEINLINE double mixed_info_term(int i, int j, double q, double s, double phi, const double* g) {
  double tot = 0.0;
  for (int p = 0; p <= 1 && p <= i; ++p)
    for (int u = 0; u <= 1 && u <= j; ++u) {
      const double P = p == 0 ? (u == 0 ? q * phi : q) : (u == 0 ? s * phi : s);
      const double cp = p == 0 ? 1.0 : static_cast<double>(i);
      const double cu = u == 0 ? 1.0 : static_cast<double>(j);
      const int a = i - p, b = j - u;
      const double sa = (s < 0.0 && a % 2 == 1) ? -1.0 : 1.0;  // d_mu^a of G(q phi) carries s^a
      tot += cp * cu * P * sa * shape_deriv(a, b, q, phi, g);
    }
  return tot;
}

// psi_m(x) for m in [lo, hi] into p[m]
LAPLACE_HD_FORCEINLINE void psi_range(double x, int lo, int hi, double* p) {
  for (int m = lo; m <= hi; ++m) p[m] = math::polygamma(m, x);
}

// The y-free parts of the order-K derivatives, K >= 2: out[b], b indices equal to phi
template <int K>
LAPLACE_HD_FORCEINLINE void free_tensor(double mu, double nu, double phi, double* out) {
  const double alpha = mu * phi, beta = nu * phi;
  // psi_(m-1) for the orders m = K - r met by r <= min(i, j) <= K/2: m - 1 from K - 1 - K/2 to K - 1
  constexpr int lo = K - 1 - K / 2;
  double pa[8] = {}, pb[8] = {}, fa[8] = {}, fb[8] = {};
  psi_range(alpha, lo, K - 1, pa);
  psi_range(beta, lo, K - 1, pb);
  for (int m = lo + 1; m <= K; ++m) {
    fa[m] = -pa[m - 1];
    fb[m] = -pb[m - 1];
  }
  const double psi_phi = math::polygamma(K - 1, phi);
  for (int b = 0; b <= K; ++b) out[b] = free_component(K - b, b, mu, nu, phi, fa, fb, psi_phi);
}

// The components of the expected information and of its derivatives of order K (K = 0..4): mm[b], mp[b],
// pp[b] with K - b derivatives in mu and b in phi, extra to the two indices of the entry.
template <int K>
LAPLACE_HD void info_components(double mu, double phi, double* mm, double* mp, double* pp) {
  const double nu = 1.0 - mu;
  const double alpha = mu * phi, beta = nu * phi;
  double pa[8] = {}, pb[8] = {}, fa[8] = {}, fb[8] = {}, ga[8] = {}, gb[8] = {};
  psi_range(alpha, 1, K + 1, pa);
  psi_range(beta, 1, K + 1, pb);
  for (int m = 2; m <= K + 2; ++m) {  // fa[m] = -psi_(m-1), the table of F = -lgamma
    fa[m] = -pa[m - 1];
    fb[m] = -pb[m - 1];
  }
  for (int m = 0; m <= K; ++m) {  // ga[m] = psi_(m+1), the table of G = psi_1
    ga[m] = pa[m + 1];
    gb[m] = pb[m + 1];
  }
  const double psi_phi = math::polygamma(K + 1, phi);  // the term of order K + 2 of lgamma(phi), pure phi only
  for (int b = 0; b <= K; ++b) {
    const int i = K - b, j = b;
    mm[b] = -free_component(i + 2, j, mu, nu, phi, fa, fb, 0.0);
    pp[b] = -free_component(i, j + 2, mu, nu, phi, fa, fb, psi_phi);
    mp[b] = mixed_info_term(i, j, mu, 1.0, phi, ga) - mixed_info_term(i, j, nu, -1.0, phi, gb);
  }
}

// d^n/du^n log(u) at u = y or u = 1 - y with the sign of d^n/dy^n: ya = d^J log y, yb = d^J log(1 - y)
LAPLACE_HD_FORCEINLINE void log_y_derivs(int J, double y, double* ya, double* yb) {
  const double iy = 1.0 / y, iz = 1.0 / (1.0 - y);
  *ya = ((J % 2 == 1) ? 1.0 : -1.0) * fact(J - 1) * ipow(iy, J);
  *yb = -fact(J - 1) * ipow(iz, J);
}

// G[n (n + 1)/2 + j] = d^n / da^(n-j) db^j log I_x(a, b) (upper = false) or log(1 - I_x(a, b)), n = N
template <int N>
LAPLACE_HD_FORCEINLINE void put_order(const math::mixed_partials<N>& m, double* G) {
  for (int j = 0; j <= N; ++j) G[N * (N + 1) / 2 + j] = m.c[j];
}

// the outer table of the cdf: every order from 1 to K
template <int K>
LAPLACE_HD void cdf_outer(bool upper, double a, double b, double x, double* G) {
  if (!upper) {
    put_order<1>(math::log_ibeta_d1(a, b, x), G);
    if constexpr (K >= 2) put_order<2>(math::log_ibeta_d2(a, b, x), G);
    if constexpr (K >= 3) put_order<3>(math::log_ibeta_d3(a, b, x), G);
    if constexpr (K >= 4) put_order<4>(math::log_ibeta_d4(a, b, x), G);
    if constexpr (K >= 5) put_order<5>(math::log_ibeta_d5(a, b, x), G);
    if constexpr (K >= 6) put_order<6>(math::log_ibeta_d6(a, b, x), G);
  } else {
    put_order<1>(math::log_ibetac_d1(a, b, x), G);
    if constexpr (K >= 2) put_order<2>(math::log_ibetac_d2(a, b, x), G);
    if constexpr (K >= 3) put_order<3>(math::log_ibetac_d3(a, b, x), G);
    if constexpr (K >= 4) put_order<4>(math::log_ibetac_d4(a, b, x), G);
    if constexpr (K >= 5) put_order<5>(math::log_ibetac_d5(a, b, x), G);
    if constexpr (K >= 6) put_order<6>(math::log_ibetac_d6(a, b, x), G);
  }
}

}  // namespace detail_beta

/**
 * \ingroup distrib_beta
 * \brief Beta distribution on \f$(0, 1)\f$ in its mean \f$\mu \in (0, 1)\f$ and precision \f$\phi > 0\f$.
 *
 * Density
 * \f[
 * f(y; \mu, \phi) = \frac{\Gamma(\phi)}{\Gamma(\mu\phi)\,\Gamma((1 - \mu)\phi)}\, y^{\mu\phi - 1} (1 - y)^{(1 - \mu)\phi - 1}, \qquad y \in (0, 1),
 * \f]
 * with shapes \f$\alpha = \mu\phi\f$, \f$\beta = (1 - \mu)\phi\f$, mean \f$\mu\f$, and variance
 * \f$\mu(1 - \mu)/(\phi + 1)\f$. Evaluates the log-density, parameter derivatives to order \f$6\f$,
 * mixed observation-parameter derivatives, closed expected Fisher information and its derivatives to
 * order \f$4\f$, cumulative functions and their log-derivatives to order \f$6\f$, quantiles, and host
 * starting values; random draws use \ref distrib_generic "rng_by_inversion()".
 * \see distrib_beta, distrib_core
 */
// The beta distribution with mean mu and precision phi (see the top of this file).
struct beta_distrib {
  /// Mean parameter \f$\mu \in (0, 1)\f$.
  // Mean, in (0, 1).
  double mu;
  /// Precision parameter \f$\phi \in (0, \infty)\f$ with \f$\alpha = \mu\phi\f$ and \f$\beta = (1 - \mu)\phi\f$.
  // Precision, in (0, infinity): alpha = mu phi, beta = (1 - mu) phi.
  double phi;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (\f$2\f$).
  static constexpr int n_params = 2;
  /// Differentiability order of \f$\log f\f$ in \f$y\f$ (\c infinitely_smooth).
  // Regularity of log f in y: C^infinity on the open support (0, 1).
  static constexpr int y_regularity = infinitely_smooth;
  /// Parameter descriptors for \c mu on \f$(0, 1)\f$ and \c phi on \f$(0, \infty)\f$.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, 1.0, false, infinitely_smooth, false, infinitely_smooth},
      {"phi", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /**
   * \ingroup distrib_beta
   * \brief Evaluates the log-density \f$\ell(\mu, \phi; y)\f$.
   *
   * Computes \f$\log\Gamma(\phi) - \log\Gamma(\alpha) - \log\Gamma(\beta) + (\alpha - 1)\log y + (\beta - 1)\log(1 - y)\f$
   * with \f$\alpha = \mu\phi\f$ and \f$\beta = (1 - \mu)\phi\f$. Returns \f$-\infty\f$ for
   * \f$y \le 0\f$, \f$y \ge 1\f$, or NaN \f$y\f$.
   * \param y Observation point \f$y \in (0, 1)\f$.
   * \return Log-density value, or \f$-\infty\f$ outside \f$(0, 1)\f$.
   * \par Accuracy
   * Max error \f$1.38\,\varepsilon\f$ of the sum of terms on `gcc` (test bound \f$8\,\varepsilon\f$).
   * \see distrib_beta, logpdf_dk(), logpdf_dy_dk()
   */
  // The log-density; -infinity outside the open support (0, 1).
  LAPLACE_HD double logpdf(double y) const {
    if (!(y > 0.0 && y < 1.0)) return -distrib_infinity;
    const double nu = 1.0 - mu;
    const double alpha = mu * phi, beta = nu * phi;
    return (math::lgamma(phi) - math::lgamma(alpha) - math::lgamma(beta)) + (alpha - 1.0) * math::log(y) +
           (beta - 1.0) * Kokkos::log1p(-y);
  }

  /**
   * \ingroup distrib_beta
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log f(y; \mu, \phi)\f$ in \f$(\mu, \phi)\f$.
   *
   * Component \f$j\f$ holds \f$\partial^K \ell / \partial\mu^{K - j}\partial\phi^j\f$: order \f$1\f$
   * keeps the residual \f$R = \log(y/(1 - y)) - \psi(\alpha) + \psi(\beta)\f$ grouped as
   * \f$\phi R\f$, order \f$2\f$ adds \f$\log(y/(1 - y))\f$ to the mixed entry only, and orders
   * \f$3\f$ to \f$6\f$ are independent of \f$y\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \in (0, 1)\f$ (unused for \f$K \ge 3\f$).
   * \return Packed symmetric tensor `sym_tensor<2, K>` of derivatives.
   * \par Accuracy
   * Order \f$1\f$ within \f$1.38\,\varepsilon\f$ of terms and orders \f$2\f$ to \f$6\f$ within
   * \f$12.2\f$ ulp of the order's scale on `gcc`; pure-\f$\phi\f$ entries stay within \f$13.4\f$ ulp of themselves.
   * \see distrib_beta, logpdf(), logpdf_dy_dk()
   */
  // The derivatives of order K (1 <= K <= 6) of log f in (mu, phi): component b has b derivatives in phi.
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dk(double y) const {
    const double nu = 1.0 - mu;
    sym_tensor<2, K> t;
    if constexpr (K == 1) {
      // l_mu = phi R with R = log(y/(1 - y)) - psi(alpha) + psi(beta), the residual, kept as one group;
      // l_phi = psi(phi) + mu (log y - psi(alpha)) + nu (log(1 - y) - psi(beta))
      const double ly = math::log(y), l1y = Kokkos::log1p(-y);
      const double psa = math::polygamma(0, mu * phi), psb = math::polygamma(0, nu * phi);
      t[0] = phi * ((ly - l1y) - psa + psb);
      t[1] = math::polygamma(0, phi) + mu * (ly - psa) + nu * (l1y - psb);
    } else {
      double out[K + 1];
      detail_beta::free_tensor<K>(mu, nu, phi, out);
      for (int b = 0; b <= K; ++b) t[b] = out[b];
      if constexpr (K == 2) t[1] += math::log(y) - Kokkos::log1p(-y);  // the only y term of order 2
    }
    return t;
  }
  /**
   * \name Derivatives of the log-density
   * Parameter derivatives of \f$\log f(y; \mu, \phi)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  // First- to sixth-order derivatives of log f in (mu, phi).
  /// First-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$, independent of \f$y\f$.
  // From order 3 the derivatives do not depend on y.
  LAPLACE_HD sym_tensor<2, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$, independent of \f$y\f$.
  LAPLACE_HD sym_tensor<2, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$, independent of \f$y\f$.
  LAPLACE_HD sym_tensor<2, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log f\f$ in \f$(\mu, \phi)\f$, independent of \f$y\f$.
  LAPLACE_HD sym_tensor<2, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_beta
   * \brief Evaluates \f$J\f$ derivatives in \f$y\f$ and the order-\f$K\f$ parameter tensor of \f$\log f(y; \mu, \phi)\f$.
   *
   * Uses \f$y_a = \mathrm{d}^J\log y / \mathrm{d}y^J\f$ and \f$y_b = \mathrm{d}^J\log(1 - y) / \mathrm{d}y^J\f$
   * for \f$J \ge 1\f$; returns \f$(\alpha - 1)y_a + (\beta - 1)y_b\f$ at \f$K = 0\f$,
   * \f$(\phi(y_a - y_b),\, \mu y_a + \nu y_b)\f$ at \f$K = 1\f$, \f$y_a - y_b\f$ at the mixed entry
   * for \f$K = 2\f$, and zero for \f$K \ge 3\f$. Not finite at \f$y = 0\f$ or \f$y = 1\f$.
   * \tparam J Observation derivative order (\f$J \ge 0\f$).
   * \tparam K Parameter derivative order (\f$J + K \le 6\f$).
   * \param y Observation point \f$y \in (0, 1)\f$.
   * \return Packed symmetric tensor `sym_tensor<2, K>` of mixed derivatives.
   * \par Accuracy
   * Checked against 5-point differences within \f$10^{-7}\f$ relative (and \f$10^{-14}\f$ for \f$J = 1, 2\f$ at \f$K = 0\f$).
   * \see distrib_beta, logpdf(), logpdf_dk()
   */
  // d^(J+K) l / dy^J dtheta^K, J + K <= 6: J derivatives in y and the symmetric tensor of order K in theta.
  // The y terms of l are (alpha - 1) log y + (beta - 1) log(1 - y), linear in the shapes, so
  //   d^J / dy^J = (alpha - 1) ya + (beta - 1) yb,  ya = d^J log y,  yb = d^J log(1 - y)  (J >= 1),
  // one derivative in theta turns (alpha - 1) and (beta - 1) into d alpha, d beta (phi, -phi in mu; mu,
  // nu in phi), the mixed second one into 1 and -1, and every higher one vanishes. J = 0 is logpdf_dk.
  // The score is not finite at y = 0 and y = 1, which are outside the open support.
  template <int J, int K>
  LAPLACE_HD sym_tensor<2, K> logpdf_dy_dk(double y) const {
    sym_tensor<2, K> t;
    if constexpr (J == 0) {
      if constexpr (K == 0) t[0] = logpdf(y);
      else t = logpdf_dk<K>(y);
      return t;
    } else if constexpr (K >= 3) {
      return t;  // every derivative in theta of order 3 or more of the y terms vanishes
    } else {
      const double nu = 1.0 - mu;
      double ya, yb;
      detail_beta::log_y_derivs(J, y, &ya, &yb);
      if constexpr (K == 0) {
        t[0] = (mu * phi - 1.0) * ya + (nu * phi - 1.0) * yb;
      } else if constexpr (K == 1) {
        t[0] = phi * (ya - yb);
        t[1] = mu * ya + nu * yb;
      } else {
        t[1] = ya - yb;  // K == 2: t[0] and t[2] stay 0
      }
      return t;
    }
  }

  /**
   * \ingroup distrib_beta
   * \brief Evaluates the closed expected Fisher information \f$I(\mu, \phi)\f$.
   *
   * Returns \f$I_{\mu\mu} = \phi^2(\psi_1(\alpha) + \psi_1(\beta))\f$,
   * \f$I_{\mu\phi} = \phi(\mu\psi_1(\alpha) - \nu\psi_1(\beta))\f$, and
   * \f$I_{\phi\phi} = \mu^2\psi_1(\alpha) + \nu^2\psi_1(\beta) - \psi_1(\phi)\f$
   * in `t[0]`, `t[1]`, and `t[2]`, where \f$I_{\mu\phi} = 0\f$ when \f$\mu = 1/2\f$.
   * \return Symmetric tensor `sym_tensor<2, 2>` of the expected information.
   * \par Accuracy
   * Max error \f$5.66\,\varepsilon\f$ of the scale on `gcc` (\f$4.89\,\varepsilon(\phi + 1)\f$ on \f$I_{\phi\phi}\f$); test bound \f$64\,\varepsilon\f$ of scale.
   * \see distrib_beta, expected_info_dk(), logpdf_dk()
   */
  // The expected information: t[0] = I_mu,mu, t[1] = I_mu,phi, t[2] = I_phi,phi.
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    double mm[1], mp[1], pp[1];
    detail_beta::info_components<0>(mu, phi, mm, mp, pp);
    sym_tensor<2, 2> t;
    t[0] = mm[0];
    t[1] = mp[0];
    t[2] = pp[0];
    return t;
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates parameter derivatives of order \f$K\f$ of the expected Fisher information.
   *
   * Components of \f$I_{\mu\mu}\f$ and \f$I_{\phi\phi}\f$ come from minus the \f$y\f$-free
   * components of `logpdf_dk()` of order \f$K + 2\f$, while derivatives of \f$I_{\mu\phi}\f$
   * differentiate \f$\phi\mu\psi_1(\alpha) - \phi\nu\psi_1(\beta)\f$ by Leibniz's rule.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container `expected_info_tensor<2, K>` with `pair[0]` for \f$(\mu, \mu)\f$, `pair[1]` for \f$(\mu, \phi)\f$, and `pair[2]` for \f$(\phi, \phi)\f$.
   * \par Accuracy
   * Max error \f$5.66\,\varepsilon\f$ of the order's scale on `gcc`; pure-\f$\phi\f$ entries stay within \f$7.68\,\varepsilon\f$ of themselves.
   * \see distrib_beta, expected_info()
   */
  // Derivatives of order K (1 <= K <= 4) of the expected information: pair[p][b], p = 0 (mu, mu),
  // 1 (mu, phi), 2 (phi, phi), b derivatives in phi among the K.
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    double mm[K + 1], mp[K + 1], pp[K + 1];
    detail_beta::info_components<K>(mu, phi, mm, mp, pp);
    expected_info_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) {
      t.pair[0][b] = mm[b];
      t.pair[1][b] = mp[b];
      t.pair[2][b] = pp[b];
    }
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives of \f$I(\mu, \phi)\f$ for orders \f$K = 1 \dots 4\f$.
   * @{
   */
  /// First-order derivatives of the expected Fisher information.
  LAPLACE_HD expected_info_tensor<2, 1> expected_info_d1() const { return expected_info_dk<1>(); }
  /// Second-order derivatives of the expected Fisher information.
  LAPLACE_HD expected_info_tensor<2, 2> expected_info_d2() const { return expected_info_dk<2>(); }
  /// Third-order derivatives of the expected Fisher information.
  LAPLACE_HD expected_info_tensor<2, 3> expected_info_d3() const { return expected_info_dk<3>(); }
  /// Fourth-order derivatives of the expected Fisher information.
  LAPLACE_HD expected_info_tensor<2, 4> expected_info_d4() const { return expected_info_dk<4>(); }
  /** @} */

  /**
   * \ingroup distrib_beta
   * \brief Evaluates the cumulative distribution function \f$F(y) = I_y(\alpha, \beta)\f$.
   *
   * Calls \ref laplace::math::ibeta "ibeta()". Returns \f$0\f$ for \f$y \le 0\f$ (and NaN) and
   * \f$1\f$ for \f$y \ge 1\f$.
   * \param y Observation point.
   * \return Probability \f$F(y) \in [0, 1]\f$.
   * \par Accuracy
   * Max relative error \f$6.2 \times 10^{-14}\f$ on `gcc` (test bound \f$5 \times 10^{-12}\f$).
   * \see distrib_beta, ccdf(), log_cdf(), quantile()
   */
  // F(y) = I_y(alpha, beta), 1 - F(y) and their logarithms; 0, 1, -infinity, 0 at y <= 0 and
  // 1, 0, 0, -infinity at y >= 1.
  LAPLACE_HD double cdf(double y) const {
    if (!(y > 0.0)) return 0.0;
    if (!(y < 1.0)) return 1.0;
    return math::ibeta(mu * phi, (1.0 - mu) * phi, y);
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates the complementary cumulative distribution function \f$1 - F(y) = 1 - I_y(\alpha, \beta)\f$.
   *
   * Calls \ref laplace::math::ibetac "ibetac()". Returns \f$1\f$ for \f$y \le 0\f$ (and NaN) and
   * \f$0\f$ for \f$y \ge 1\f$.
   * \param y Observation point.
   * \return Upper-tail probability \f$1 - F(y) \in [0, 1]\f$.
   * \par Accuracy
   * Max relative error \f$6.6 \times 10^{-14}\f$ on `gcc` (test bound \f$5 \times 10^{-12}\f$).
   * \see distrib_beta, cdf(), log_ccdf()
   */
  LAPLACE_HD double ccdf(double y) const {
    if (!(y > 0.0)) return 1.0;
    if (!(y < 1.0)) return 0.0;
    return math::ibetac(mu * phi, (1.0 - mu) * phi, y);
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates \f$\log F(y) = \log I_y(\alpha, \beta)\f$.
   *
   * Calls \ref laplace::math::log_ibeta "log_ibeta()". Returns \f$-\infty\f$ for \f$y \le 0\f$
   * (and NaN) and \f$0\f$ for \f$y \ge 1\f$.
   * \param y Observation point.
   * \return Log-probability \f$\log F(y) \in [-\infty, 0]\f$.
   * \par Accuracy
   * Max error \f$1.9 \times 10^{-15}\f$ divided by \f$\max(1, |\log F|)\f$ on `gcc` (test bound \f$5 \times 10^{-12}\f$).
   * \see distrib_beta, cdf(), log_ccdf(), log_cdf_dk()
   */
  LAPLACE_HD double log_cdf(double y) const {
    if (!(y > 0.0)) return -distrib_infinity;
    if (!(y < 1.0)) return 0.0;
    return math::log_ibeta(mu * phi, (1.0 - mu) * phi, y);
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates \f$\log(1 - F(y)) = \log(1 - I_y(\alpha, \beta))\f$.
   *
   * Calls \ref laplace::math::log_ibetac "log_ibetac()". Returns \f$0\f$ for \f$y \le 0\f$
   * (and NaN) and \f$-\infty\f$ for \f$y \ge 1\f$.
   * \param y Observation point.
   * \return Log upper-tail probability \f$\log(1 - F(y)) \in [-\infty, 0]\f$.
   * \par Accuracy
   * Max error \f$1.4 \times 10^{-15}\f$ divided by \f$\max(1, |\log(1 - F)|)\f$ on `gcc` (test bound \f$5 \times 10^{-12}\f$).
   * \see distrib_beta, ccdf(), log_cdf(), log_ccdf_dk()
   */
  LAPLACE_HD double log_ccdf(double y) const {
    if (!(y > 0.0)) return 0.0;
    if (!(y < 1.0)) return -distrib_infinity;
    return math::log_ibetac(mu * phi, (1.0 - mu) * phi, y);
  }

  /**
   * \ingroup distrib_beta
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ or \f$\log(1 - F(y))\f$.
   *
   * Evaluates the shape derivatives via log_ibeta_d1()..log_ibeta_d6() or
   * log_ibetac_d1()..log_ibetac_d6() (\ref incomplete_beta "incomplete beta") and composes them with the linear
   * maps \f$\alpha = \mu\phi\f$ and \f$\beta = (1 - \mu)\phi\f$ using chain_rule_pair_dk()
   * (\ref chain_pair "chain_pair"). Returns zeros at \f$y = 1\f$ (`upper = false`) or \f$y = 0\f$ (`upper = true`),
   * and quiet NaN at the opposite endpoint or outside \f$[0, 1]\f$.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param upper Selects \f$\log(1 - F(y))\f$ when `true` and \f$\log F(y)\f$ when `false`.
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>` of parameter derivatives.
   * \par Accuracy
   * On `gcc`, max scaled errors across orders \f$1\f$ to \f$6\f$ are \f$4.3 \times 10^{-14}\f$ to
   * \f$2.8 \times 10^{-10}\f$ for \f$\log F\f$ and \f$2.9 \times 10^{-15}\f$ to \f$1.8 \times 10^{-10}\f$ for \f$\log(1 - F)\f$.
   * \see distrib_beta, log_cdf_dk(), log_ccdf_dk(), incomplete_beta, chain_pair
   */
  // The derivatives of order K (1 <= K <= 6) of log F (upper = false) or log(1 - F) (upper = true) in
  // (mu, phi): the mixed partials in the shapes (a, b) = (alpha, beta), composed with the linear maps
  // alpha(mu, phi) and beta(mu, phi) by chain_rule_pair_dk. u0 = alpha and u1 = beta are filled at the
  // packed positions pair_index(1, 0) = 1 (d/dmu), pair_index(0, 1) = 2 (d/dphi), pair_index(1, 1) = 4.
  template <int K>
  LAPLACE_HD sym_tensor<2, K> cdf_dk(bool upper, double y) const {
    const double nu = 1.0 - mu;
    double G[detail_beta::n_pairs6] = {}, u0[detail_beta::n_pairs6] = {}, u1[detail_beta::n_pairs6] = {};
    detail_beta::cdf_outer<K>(upper, mu * phi, nu * phi, y, G);
    u0[1] = phi;
    u0[2] = mu;
    u0[4] = 1.0;
    u1[1] = -phi;
    u1[2] = nu;
    u1[4] = -1.0;
    double out[K + 1];
    math::chain_rule_pair_dk<K>(G, u0, u1, out);
    sym_tensor<2, K> t;
    for (int b = 0; b <= K; ++b) t[b] = out[b];
    return t;
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log F(y)\f$ via `cdf_dk<K>(false, y)`.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>` of derivatives of \f$\log F(y)\f$.
   * \par Accuracy
   * See cdf_dk() (test bounds \f$7 \times 10^{-14}\f$ to \f$4.5 \times 10^{-10}\f$ of the order's scale).
   * \see distrib_beta, cdf_dk(), log_ccdf_dk()
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_cdf_dk(double y) const { return cdf_dk<K>(false, y); }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates parameter derivatives of order \f$K\f$ of \f$\log(1 - F(y))\f$ via `cdf_dk<K>(true, y)`.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point.
   * \return Packed symmetric tensor `sym_tensor<2, K>` of derivatives of \f$\log(1 - F(y))\f$.
   * \par Accuracy
   * See cdf_dk() (test bounds \f$7 \times 10^{-14}\f$ to \f$4.5 \times 10^{-10}\f$ of the order's scale).
   * \see distrib_beta, cdf_dk(), log_cdf_dk()
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> log_ccdf_dk(double y) const { return cdf_dk<K>(true, y); }
  /**
   * \name Parameter derivatives of the log cumulative functions
   * Parameter derivatives of \f$\log F(y)\f$ and \f$\log(1 - F(y))\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_cdf_d1(double y) const { return cdf_dk<1>(false, y); }
  /// Second-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_cdf_d2(double y) const { return cdf_dk<2>(false, y); }
  /// Third-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_cdf_d3(double y) const { return cdf_dk<3>(false, y); }
  /// Fourth-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_cdf_d4(double y) const { return cdf_dk<4>(false, y); }
  /// Fifth-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_cdf_d5(double y) const { return cdf_dk<5>(false, y); }
  /// Sixth-order derivatives of \f$\log F(y)\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_cdf_d6(double y) const { return cdf_dk<6>(false, y); }
  /// First-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 1> log_ccdf_d1(double y) const { return cdf_dk<1>(true, y); }
  /// Second-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 2> log_ccdf_d2(double y) const { return cdf_dk<2>(true, y); }
  /// Third-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 3> log_ccdf_d3(double y) const { return cdf_dk<3>(true, y); }
  /// Fourth-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 4> log_ccdf_d4(double y) const { return cdf_dk<4>(true, y); }
  /// Fifth-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 5> log_ccdf_d5(double y) const { return cdf_dk<5>(true, y); }
  /// Sixth-order derivatives of \f$\log(1 - F(y))\f$ in \f$(\mu, \phi)\f$.
  LAPLACE_HD sym_tensor<2, 6> log_ccdf_d6(double y) const { return cdf_dk<6>(true, y); }
  /** @} */

  /**
   * \ingroup distrib_beta
   * \brief Evaluates the quantile \f$F^{-1}(p)\f$ via \ref laplace::math::ibeta_inv "ibeta_inv()".
   *
   * Returns \f$0\f$ at \f$p = 0\f$, \f$1\f$ at \f$p = 1\f$, and quiet NaN outside \f$[0, 1]\f$ or on
   * NaN; roots below \f$2^{-1075}\f$ flush to \f$0\f$ and complements at or below \f$2^{-1074}\f$
   * flush to \f$1\f$.
   * \param p Probability \f$p \in [0, 1]\f$.
   * \return Quantile \f$y \in [0, 1]\f$.
   * \par Accuracy
   * Matches \ref laplace::math::ibeta_inv "ibeta_inv()" (\f$7.05\f$ ulpC bulk, \f$5.38\f$ ulpC tail); checked against `cdf()` to \f$10^{-9}\f$ relative.
   * \see distrib_beta, quantile_from_log(), quantile_from_log_upper(), cdf()
   */
  // The quantile F^-1(p), from p, from log p and from log(1 - p), the inverse incomplete beta; the last
  // two never form 1 - p, so a quantile in the far tail is found to the accuracy of the log cdf there.
  LAPLACE_HD double quantile(double p) const { return math::ibeta_inv(mu * phi, (1.0 - mu) * phi, p); }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates the quantile from lower-tail log-probability \f$\log p\f$ via \ref laplace::math::ibeta_inv_logp "ibeta_inv_logp()".
   *
   * Returns \f$0\f$ at \f$\log p = -\infty\f$, \f$1\f$ at \f$\log p = 0\f$, and quiet NaN for
   * \f$\log p > 0\f$ or NaN.
   * \param log_p Log-probability \f$\log p \in [-\infty, 0]\f$.
   * \return Quantile \f$y \in [0, 1]\f$.
   * \par Accuracy
   * Matches \ref laplace::math::ibeta_inv_logp "ibeta_inv_logp()" (\f$18.6\f$ ulpC max); checked against `log_cdf()` to \f$10^{-9}|\log p| + 4\alpha\varepsilon\f$.
   * \see distrib_beta, quantile(), quantile_from_log_upper(), log_cdf()
   */
  LAPLACE_HD double quantile_from_log(double log_p) const {
    return math::ibeta_inv_logp(mu * phi, (1.0 - mu) * phi, log_p);
  }
  /**
   * \ingroup distrib_beta
   * \brief Evaluates the quantile from upper-tail log-probability \f$\log(1 - p)\f$ via \ref laplace::math::ibetac_inv_logq "ibetac_inv_logq()".
   *
   * Returns \f$0\f$ at \f$\log q = 0\f$, \f$1\f$ at \f$\log q = -\infty\f$, and quiet NaN for
   * \f$\log q > 0\f$ or NaN.
   * \param log_q Log complement probability \f$\log q \in [-\infty, 0]\f$.
   * \return Quantile \f$y \in [0, 1]\f$.
   * \par Accuracy
   * Matches \ref laplace::math::ibetac_inv_logq "ibetac_inv_logq()" (\f$5.82\f$ ulpC max); checked against `log_ccdf()` to \f$10^{-9}|\log q| + 4\beta\varepsilon/(1 - y)\f$.
   * \see distrib_beta, quantile(), quantile_from_log(), log_ccdf()
   */
  LAPLACE_HD double quantile_from_log_upper(double log_q) const {
    return math::ibetac_inv_logq(mu * phi, (1.0 - mu) * phi, log_q);
  }

  /// \ingroup distrib_beta
  /// \brief Lower bound of the open support (\f$0\f$).
  /// \return `0.0`.
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_beta
  /// \brief Upper bound of the open support (\f$1\f$).
  /// \return `1.0`.
  LAPLACE_HD double support_upper() const { return 1.0; }
  /// \ingroup distrib_beta
  /// \brief Central location (the mean \f$\mu\f$).
  /// \return `mu`.
  // The mean, and the standard deviation sqrt(mu nu / (phi + 1)).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_beta
  /// \brief Typical spread (the standard deviation \f$\sqrt{\mu(1 - \mu)/(\phi + 1)}\f$).
  /// \return Standard deviation.
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu * (1.0 - mu) / (phi + 1.0)); }
  /// \ingroup distrib_beta
  /// \brief Non-smooth points in the parameters (none).
  /// \return Empty `distrib_kinks`.
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }
  /// \ingroup distrib_beta
  /// \brief Non-smooth points in \f$y\f$ (none).
  /// \return Empty `distrib_kinks`.
  LAPLACE_HD distrib_kinks y_kinks() const { return {}; }

  /**
   * \ingroup distrib_beta
   * \brief Starting value from a sample summary by the method of moments.
   *
   * Clamps the sample mean \f$\bar{y}\f$ to \f$[10^{-6}, 1 - 10^{-6}]\f$ (defaulting NaN to
   * \f$1/2\f$) and sets \f$\phi = \max(\mu_0(1 - \mu_0)/s_y^2 - 1, 10^{-3})\f$ when
   * \f$\bar{y} \in (0, 1)\f$ and \f$0 < s_y^2 < \infty\f$, or \f$\phi = 1\f$ otherwise, moving both
   * strictly inside their open domains with start_inside().
   * \param s Sample summary computed on the host by summarize_sample().
   * \return A `beta_distrib` strictly inside \f$(0, 1) \times (0, \infty)\f$.
   * \par Accuracy
   * Closed-form formula in the summary moments.
   * \see distrib_beta, distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  // Starting value by the method of moments, inverting Var = mu nu / (1 + phi): mu is the sample mean
  // clamped into (1e-6, 1 - 1e-6), phi = max(mu (1 - mu) / variance - 1, 1e-3), each then moved strictly
  // inside its domain. A sample with no spread, with a non-finite variance, or with a mean outside
  // (0, 1) has no moment estimate of phi: phi = 1 then (the uniform law at mu = 1/2), and a NaN mean
  // gives mu = 1/2.
  beta_distrib start_value(const distrib_sample_summary& s) const {
    double m = s.mean;
    if (!(m == m)) m = 0.5;
    const double mu0 = m < 1e-6 ? 1e-6 : (m > 1.0 - 1e-6 ? 1.0 - 1e-6 : m);
    double phi0 = 1.0;
    if (s.mean > 0.0 && s.mean < 1.0 && s.variance > 0.0 && s.variance < distrib_infinity) {
      const double est = mu0 * (1.0 - mu0) / s.variance - 1.0;
      phi0 = est > 1e-3 ? est : 1e-3;
    }
    return {start_inside(mu0, params[0]), start_inside(phi0, params[1])};
  }
};

}  // namespace laplace::distrib
