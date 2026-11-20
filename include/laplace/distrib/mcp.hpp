// mcp_distrib (laplace::distrib): the properized minimax concave penalty, the
// density whose negative logarithm is Zhang's MCP on an interval around the
// location and a tail of the user's choice beyond it. Parameters: the location mu
// (real), the rate lambda > 0, the concavity gamma > 0 and the parameters of the
// tail. With u = |y - mu| and K = gamma lambda the knot,
//
//   q(u) = lambda u - u^2 / (2 gamma)              u <= K
//   q(u) = gamma lambda^2 / 2 + H(u - K)           u >  K
//
//   l(mu, lambda, gamma, theta_t; y) = -q(u) - log Z,
//   Z = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma / 2)) + e^(-gamma lambda^2/2) T ],
//
// F being Dawson's integral (math/dawson.hpp), H the tail's exponent and
// T = int_0^inf exp(-H) its constant (distrib/properizing_tail.hpp). The original
// MCP has a flat tail, which makes exp(-MCP) unnormalizable for every parameter;
// replacing the flat part by a tail that grows makes the law proper and leaves the
// penalty untouched on u <= K, so mcp_penalty and this law share their inner region
// exactly. The inner part of Z comes from completing the square, the inner exponent
// being convex in u, which is where Dawson's integral enters; it was checked against
// mpmath quadrature of the definition (tools/reference/distrib_mcp_scad.py).
//
// The tail is a template parameter whose default is student_tail. The three tails
// differ in the score far from the location: the Gaussian one has q' growing without
// bound, so beyond u = K + lambda V it shrinks more than a rate-lambda Laplace, which
// is the opposite of what the MCP is for; the logistic one has q' bounded by 2/s; the
// Student one has q' -> 0, the MCP's own behaviour up to O(1/u). The Gaussian tail is
// the limit of the Student one as its degrees of freedom grow at a fixed scale.
//
// Shape. K is the knot. q and q' are continuous there, q' being zero on both sides,
// and q'' jumps from -1/gamma to H''(0), so the law is C^1 in y at the knot and C^0 at
// u = 0, where q' jumps from -lambda to lambda. The kink at the location is the only
// point of regularity 0, so y_regularity is 0, as for laplace2_distrib. The limits are
// the laws one expects: gamma -> 0 leaves no inner region and gives exp(-H) itself,
// and gamma lambda^2 -> infinity makes the tail's mass vanish and gives the Laplace of
// rate lambda.
//
// Derivatives. l is -q - log Z. y and mu enter through u only, with du/dy = sign(y - mu)
// and du/dmu = -sign(y - mu), so the component with a derivatives in mu, J in y and
// (b, m, n1, n2) in (lambda, gamma, theta_t) is
//
//   -(-1)^a sign(y - mu)^(a + J) d^(a+J)_u d^b_lambda d^m_gamma d^n_theta q
//   - [a + J = 0] d^b_lambda d^m_gamma d^n_theta log Z,
//
// where sign(y - mu)^t is read as 1 for even t and as sign(y - mu) with sign(0) = 0 for
// odd t: at the kink the odd orders are the mean of the two one-sided values, which is
// zero, and the even orders agree on the two sides and are taken from the inner piece.
// The inner piece is quadratic in u, so every component with a + J >= 3 vanishes
// inside it; at a knot the value and every derivative come from the inner piece, as in
// mcp_penalty. The mixed partials of q were derived and checked against SymPy in
// tools/reference/distrib_mcp_scad_forms.py and
// tools/reference/properized_tail_forms.py; those of log Z are built by the recursions
// of detail/properized_nonconvex.hpp, which the first script checks.
//
// Cumulative functions. The integral of exp(-q) from u to infinity is closed and free
// of cancellation on both pieces, with G(x) = int_x^inf exp(-H) the tail's own partial
// integral,
//
//   u <= K:  sqrt(2 gamma) e^(-q(u)) F((K - u)/sqrt(2 gamma)) + e^(-gamma lambda^2/2) T
//   u >  K:  e^(-gamma lambda^2/2) G(u - K),
//
// so F(y) is that integral over Z below the location and one minus it above. Its
// derivatives in the parameters, the expected information and the quantile are left to
// the generic routes (generic_cdf.hpp, generic_info.hpp, generic_quantile.hpp,
// generic_rng.hpp).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/properized_nonconvex.hpp>
#include <laplace/distrib/properizing_tail.hpp>
#include <laplace/math/dawson.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

#include <Kokkos_Core.hpp>

#include <array>

namespace laplace::distrib {

namespace detail_mcp {

using namespace detail_properized;

// q(u)
template <class Tail>
LAPLACE_HD double q_value(double u, double lambda, double gamma, const Tail& tail) {
  const double knot = gamma * lambda;
  if (u <= knot) return lambda * u - u * u / (2.0 * gamma);
  return 0.5 * gamma * lambda * lambda + detail_tail::h_mixed(tail, 0, 0, 0, u - knot);
}

// one mixed partial d^j_u d^i_lambda d^m_gamma d^n1 d^n2 q, the last two in the tail's
// own parameters
template <class Tail>
LAPLACE_HD double q_mixed(int j, int i, int m, int n1, int n2, double u, double lambda, double gamma,
                          const Tail& tail) {
  if (u <= gamma * lambda) {  // lambda u - u^2/(2 gamma); the tail does not appear
    if (n1 > 0 || n2 > 0) return 0.0;
    double s = (m == 0) ? q_rate(j, i, u, lambda) : 0.0;
    if (i == 0) s -= 0.5 * d2_pow(j, u) * inv_deriv(m, gamma);
    return s;
  }
  // gamma lambda^2/2 + H(u - gamma lambda)
  double s = 0.0;
  if (n1 == 0 && n2 == 0 && j == 0) {
    if (m == 0) s += 0.5 * d2_pow(i, lambda) * gamma;
    else if (m == 1) s += 0.5 * d2_pow(i, lambda);
  }
  return s + q_tail_piece(j, i, m, n1, n2, u, lambda, gamma, tail);
}

// the tables over (lambda, gamma) of A and B in Z = 2 [A + B T]
template <int K>
LAPLACE_HD void z_tables(double lambda, double gamma, double* A, double* B) {
  dawson_term<K>(lambda, gamma, A);  // sqrt(2 gamma) F(lambda sqrt(gamma/2))
  double phi[tab2_size(K)];          // -gamma lambda^2/2
  zero2<K>(phi);
  phi[idx2<K>(0, 0)] = -0.5 * gamma * lambda * lambda;
  if constexpr (K >= 1) {
    phi[idx2<K>(1, 0)] = -gamma * lambda;
    phi[idx2<K>(0, 1)] = -0.5 * lambda * lambda;
  }
  if constexpr (K >= 2) {
    phi[idx2<K>(2, 0)] = -gamma;
    phi[idx2<K>(1, 1)] = -lambda;
  }
  if constexpr (K >= 3) phi[idx2<K>(2, 1)] = -1.0;
  exp2<K>(phi, B);
}

// log Z, the two positive terms added in logs so that a knot large enough to
// underflow e^(-gamma lambda^2/2) still leaves the tail's logarithm
template <class Tail>
LAPLACE_HD double log_z(double lambda, double gamma, const Tail& tail) {
  const double x = lambda * Kokkos::sqrt(0.5 * gamma);
  const double inner = Kokkos::sqrt(2.0 * gamma) * math::dawson(x);
  return 0.69314718055994530942 +
         log_add(math::log(inner), -0.5 * gamma * lambda * lambda + tail.log_tail_const());
}

// Z itself, a sum of two positive terms. Where the second underflows the first is
// the whole constant, which is the Laplace limit 2/lambda.
template <class Tail>
LAPLACE_HD double z_value(double lambda, double gamma, const Tail& tail) {
  const double x = lambda * Kokkos::sqrt(0.5 * gamma);
  return 2.0 * (Kokkos::sqrt(2.0 * gamma) * math::dawson(x) +
                math::exp(-0.5 * gamma * lambda * lambda) * tail.tail_const());
}

// int_u^inf exp(-q(t)) dt, u >= 0
template <class Tail>
LAPLACE_HD double upper_integral(double u, double lambda, double gamma, const Tail& tail) {
  const double knot = gamma * lambda, C = 0.5 * gamma * lambda * lambda;
  if (u > knot) return math::exp(-C) * tail.upper(u - knot);
  const double root = Kokkos::sqrt(2.0 * gamma);
  return root * math::exp(-q_value(u, lambda, gamma, tail)) * math::dawson((knot - u) / root) +
         math::exp(-C) * tail.tail_const();
}

template <class Tail>
LAPLACE_HD double log_upper_integral(double u, double lambda, double gamma, const Tail& tail) {
  const double knot = gamma * lambda, C = 0.5 * gamma * lambda * lambda;
  if (u > knot) return -C + tail.log_upper(u - knot);
  const double root = Kokkos::sqrt(2.0 * gamma);
  const double a = math::log(root * math::dawson((knot - u) / root)) - q_value(u, lambda, gamma, tail);
  return log_add(a, -C + tail.log_tail_const());
}

// the parameter descriptors: the law's three, then the tail's own (host, read at a
// runtime index, as everywhere)
template <class Tail, int N>
constexpr std::array<distrib_param_info, N> param_list() {
  std::array<distrib_param_info, N> p{};
  p[0] = {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1};
  p[1] = {"lambda", 0.0, distrib_infinity, false, 1, false, 2};
  p[2] = {"gamma", 0.0, distrib_infinity, false, 1, false, 2};
  for (int j = 0; j < Tail::n_tail_params; ++j) p[3 + j] = Tail::tail_params[j];
  return p;
}

}  // namespace detail_mcp

/**
 * \ingroup distrib_mcp
 * \brief The properized minimax concave penalty, continuous on the real line, in the
 * location \f$\mu\f$, the rate \f$\lambda > 0\f$, the concavity \f$\gamma > 0\f$ and the
 * parameters of the properizing tail.
 *
 * \tparam Tail The tail beyond the knot: \c student_tail (the default), \c gaussian_tail
 * or \c logistic_tail.
 * \see distrib_mcp, scad_distrib, distrib_properizing_tail
 */
template <class Tail = student_tail>
struct mcp_distrib {
  /// Location \f$\mu \in (-\infty, \infty)\f$, the mode and the median.
  double mu;
  /// Rate \f$\lambda \in (0, \infty)\f$, the slope of \f$q\f$ at the location.
  double lambda;
  /// Concavity \f$\gamma \in (0, \infty)\f$; the knot sits at \f$u = \gamma\lambda\f$.
  double gamma;
  /// The properizing tail, carrying its own parameters.
  Tail tail;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (three, plus the tail's own).
  static constexpr int n_params = 3 + Tail::n_tail_params;
  /// Differentiability order of the log-density in \f$y\f$ (0, a kink at \f$y = \mu\f$).
  static constexpr int y_regularity = 0;
  /// Descriptors of the parameters, the law's three followed by the tail's.
  static constexpr std::array<distrib_param_info, n_params> params_ =
      detail_mcp::param_list<Tail, n_params>();
  /// Descriptors of the parameters.
  static constexpr const distrib_param_info* params = params_.data();

  /// The log-density \f$-q(|y - \mu|) - \log Z\f$.
  LAPLACE_HD double logpdf(double y) const {
    return -detail_mcp::q_value(Kokkos::fabs(y - mu), lambda, gamma, tail) -
           detail_mcp::log_z(lambda, gamma, tail);
  }

  /**
   * \brief Proximal operator in \f$y\f$: \f$\arg\min_y \{-\log f(y) + (y - v)^2/(2t)\}\f$.
   *
   * With \f$U = |v - \mu|\f$ the problem is \f$\min_{x \ge 0} t\,q(x) + (x - U)^2/2\f$, and
   * the minimizer has the sign of \f$v - \mu\f$ with \f$|x| \le U\f$, since \f$q\f$ does not
   * decrease. The objective is smooth on each of the two pieces of \f$q\f$ and grows at
   * infinity, so its global minimum is at \f$x = 0\f$, at the knot \f$K = \gamma\lambda\f$, or
   * at a stationary point of one piece. Inside the knot the objective is a quadratic of
   * curvature \f$1 - t/\gamma\f$, with the firm threshold \f$(U - t\lambda)/(1 - t/\gamma)\f$
   * as its stationary point when that curvature is positive, and no interior minimum
   * otherwise; beyond the knot the stationary points come from the tail
   * (\c prox_roots(), up to three with the Student tail). The candidates are compared on
   * the objective, the smallest \f$x\f$ winning a tie, as in the prox of \f$a|x|^p\f$.
   * \param v Point \f$v\f$.
   * \param t Step \f$t > 0\f$.
   * \return The minimizer.
   * \see penalty_prox, properizing_tail
   */
  LAPLACE_HD double prox_y(double v, double t) const {
    const double u = v - mu, U = Kokkos::fabs(u);
    // the location itself passes through, and so do NaN and an infinite point,
    // the last of them being its own prox however large the step
    if (!(U > 0.0) || !Kokkos::isfinite(U)) return v;
    const double knot = gamma * lambda;
    double xb = 0.0, gb = 0.5 * U * U;  // x = 0, where q vanishes
    auto offer = [&](double x) {
      const double e = x - U;
      const double g = t * detail_mcp::q_value(x, lambda, gamma, tail) + 0.5 * e * e;
      if (g < gb) gb = g, xb = x;
    };
    const double curv = 1.0 - t / gamma;
    if (curv > 0.0) {
      const double x = (U - t * lambda) / curv;
      if (x > 0.0 && x < knot) offer(x);
    }
    offer(knot);
    double z[4];
    const int n = tail.prox_roots(U - knot, t, z);
    for (int j = 0; j < n; ++j) offer(knot + z[j]);
    return mu + (u < 0.0 ? -xb : xb);
  }

  /// Derivatives of order \f$K\f$ of the log-density in \f$(\mu, \lambda, \gamma, \theta_t)\f$.
  template <int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dk(double y) const {
    return logpdf_dy_dk<0, K>(y);
  }
  /**
   * \name Derivatives of the log-density
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<n_params, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /// \f$J\f$ derivatives in \f$y\f$ and the symmetric tensor of order \f$K\f$ in the parameters.
  template <int J, int K>
  LAPLACE_HD sym_tensor<n_params, K> logpdf_dy_dk(double y) const {
    using namespace detail_properized;
    constexpr int nt = Tail::n_tail_params;
    const double u = Kokkos::fabs(y - mu);
    const double s = y > mu ? 1.0 : (y < mu ? -1.0 : 0.0);
    sym_tensor<n_params, K> t;
    if constexpr (K == 0) {
      if constexpr (J == 0) t[0] = logpdf(y);
      else
        t[0] = -u_chain_factor(0, J, s) * detail_mcp::q_mixed(J, 0, 0, 0, 0, u, lambda, gamma, tail);
      return t;
    } else {
      double A[tab2_size(K)], B[tab2_size(K)], LZ[z_tab_size(K, nt)];
      if constexpr (J == 0) {
        detail_mcp::z_tables<K>(lambda, gamma, A, B);
        log_z_table<K>(A, B, tail, LZ);
      }
      for (int a = 0; a <= K; ++a)
        for (int b = 0; a + b <= K; ++b)
          for (int m = 0; a + b + m <= K; ++m)
            for (int n1 = 0; a + b + m + n1 <= K; ++n1) {
              const int n2 = K - a - b - m - n1;
              if (nt == 1 && n2 != 0) continue;
              int idx[K];
              int k = 0;
              for (int w = 0; w < a; ++w) idx[k++] = 0;
              for (int w = 0; w < b; ++w) idx[k++] = 1;
              for (int w = 0; w < m; ++w) idx[k++] = 2;
              for (int w = 0; w < n1; ++w) idx[k++] = 3;
              for (int w = 0; w < n2; ++w) idx[k++] = 4;
              double v = -u_chain_factor(a, a + J, s) *
                         detail_mcp::q_mixed(a + J, b, m, n1, n2, u, lambda, gamma, tail);
              if constexpr (J == 0) {
                if (a == 0) v -= LZ[z_tab_index<Tail>(b, m, n1, n2)];
              }
              t[sym_tensor_position<n_params, K>(idx)] = v;
            }
      return t;
    }
  }

  /**
   * \brief The expected Fisher information in closed form.
   *
   * The location is handled by the score form, \f$I_{\mu\mu} = \mathbb{E}[q'(u)^2]\f$,
   * which integration by parts turns into \f$(2/Z)[\lambda + \int_0^\infty q''e^{-q}]\f$:
   * the boundary term \f$\lambda\f$ is the jump of \f$q'\f$ at the kink, the point mass
   * that \f$-\mathbb{E}[l_{\mu\mu}]\f$ misses, as in enet_distrib. By symmetry
   * \f$I_{\mu\theta} = 0\f$ for every other parameter. The rest is the hessian form
   * \f$I_{ab} = \mathbb{E}[q_{ab}] + \partial^2_{ab}\log Z\f$, with \f$\log Z\f$ from the
   * order-two table and \f$\mathbb{E}[q_{ab}]\f$ from the moments of the inner piece
   * (\c dawson_moments) and the three integrals the tail supplies (\c info_xx,
   * \c info_xt, \c info_tt). Nothing beyond Dawson's integral and its first two
   * derivatives, the tail's own constant and one rational function of the tail's
   * parameters is evaluated.
   * \return Packed symmetric matrix \c sym_tensor<n_params, 2>.
   * \see distrib_mcp, properizing_tail
   */
  LAPLACE_HD sym_tensor<n_params, 2> expected_info() const {
    using namespace detail_properized;
    constexpr int nt = Tail::n_tail_params, ns = 2 + nt;
    double A[tab2_size(2)], B[tab2_size(2)], LZ[z_tab_size(2, nt)];
    detail_mcp::z_tables<2>(lambda, gamma, A, B);
    log_z_table<2>(A, B, tail, LZ);
    double N[3];
    dawson_moments(lambda, gamma, N);
    const double E = B[idx2<2>(0, 0)], T = tail.tail_const();
    const double Z2 = N[0] + E * T, J = detail_tail::info_xx(tail);
    sym_tensor<n_params, 2> t;
    int idx[2] = {0, 0};
    t[sym_tensor_position<n_params, 2>(idx)] = (lambda - N[0] / gamma + E * J) / Z2;
    for (int b = 1; b < n_params; ++b) {
      idx[0] = 0, idx[1] = b;
      t[sym_tensor_position<n_params, 2>(idx)] = 0.0;
    }
    for (int a = 0; a < ns; ++a)
      for (int b = a; b < ns; ++b) {
        double v;  // int_0^inf q_ab e^-q du
        if (a == 0 && b == 0) v = E * gamma * (T + gamma * J);
        else if (a == 0 && b == 1) v = E * (lambda * T - 1.0 + gamma * lambda * J);
        else if (a == 1 && b == 1) v = -N[2] / (gamma * gamma * gamma) + E * lambda * lambda * J;
        else if (a == 0) v = -gamma * E * detail_tail::info_xt(tail, b - 2);
        else if (a == 1) v = -lambda * E * detail_tail::info_xt(tail, b - 2);
        else v = E * detail_tail::info_tt(tail, a - 2, b - 2);
        idx[0] = a + 1, idx[1] = b + 1;
        t[sym_tensor_position<n_params, 2>(idx)] =
            v / Z2 + LZ[z_tab_index<Tail>(int(a == 0) + int(b == 0), int(a == 1) + int(b == 1),
                                          int(a == 2) + int(b == 2), int(a == 3) + int(b == 3))];
      }
    return t;
  }

  /// The cumulative distribution function \f$F(y)\f$.
  LAPLACE_HD double cdf(double y) const {
    const double r = detail_mcp::upper_integral(Kokkos::fabs(y - mu), lambda, gamma, tail) /
                     detail_mcp::z_value(lambda, gamma, tail);
    return y <= mu ? r : 1.0 - r;
  }
  /// The complementary cumulative distribution function \f$1 - F(y)\f$.
  LAPLACE_HD double ccdf(double y) const {
    const double r = detail_mcp::upper_integral(Kokkos::fabs(y - mu), lambda, gamma, tail) /
                     detail_mcp::z_value(lambda, gamma, tail);
    return y >= mu ? r : 1.0 - r;
  }
  // The side on which the probability is small is taken in logarithms, where it can
  // reach below the smallest double; the other side is one minus that probability,
  // taken as a plain ratio. The logarithm of the small side would lose one ulp per
  // unit of its own size there: at u = 25.6 with lambda = 1.3 and gamma = 3 the
  // exponent is -124, and exp of it carries the exponent's rounding error, so about
  // 124 ulp of the result, of which 86 were measured on 2026-11-18; the plain ratio
  // has no such term.
  /// \f$\log F(y)\f$.
  LAPLACE_HD double log_cdf(double y) const {
    const double u = Kokkos::fabs(y - mu);
    if (y <= mu)
      return detail_mcp::log_upper_integral(u, lambda, gamma, tail) - detail_mcp::log_z(lambda, gamma, tail);
    return Kokkos::log1p(-detail_mcp::upper_integral(u, lambda, gamma, tail) /
                         detail_mcp::z_value(lambda, gamma, tail));
  }
  /// \f$\log(1 - F(y))\f$.
  LAPLACE_HD double log_ccdf(double y) const {
    const double u = Kokkos::fabs(y - mu);
    if (y >= mu)
      return detail_mcp::log_upper_integral(u, lambda, gamma, tail) - detail_mcp::log_z(lambda, gamma, tail);
    return Kokkos::log1p(-detail_mcp::upper_integral(u, lambda, gamma, tail) /
                         detail_mcp::z_value(lambda, gamma, tail));
  }

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Rough spread for the quadratures: the Laplace scale plus the tail's own.
  LAPLACE_HD double scale() const { return 1.0 / lambda + tail.scale_hint(); }

  /// Non-smooth points in parameter \f$\theta_j\f$ for observation \f$y\f$.
  LAPLACE_HD distrib_kinks param_kinks(int j, double y) const {
    distrib_kinks k;
    const double u = Kokkos::fabs(y - mu);
    if (j == 0) {
      k.count = 1;
      k.point[0] = y;  // the kink of the log-density, continuous_order 0
    } else if (j == 1 && u > 0.0) {
      k.count = 1;
      k.point[0] = u / gamma;  // the knot u = gamma lambda, C^1
      k.continuous_order = 1;
    } else if (j == 2 && u > 0.0) {
      k.count = 1;
      k.point[0] = u / lambda;
      k.continuous_order = 1;
    }
    return k;  // the tail is smooth in its own parameters
  }

  /**
   * \ingroup distrib_mcp
   * \brief Non-smooth points in \f$y\f$: the kink at \f$y = \mu\f$ and the two knots
   * \f$y = \mu \pm \gamma\lambda\f$, with \c continuous_order 0.
   *
   * The kink is \f$C^0\f$ and the knots \f$C^1\f$; one order covers the three, so the
   * least of them is declared. The knots belong here although they are not kinks,
   * because the quadratures place a panel end at every point declared, and the second
   * derivative of the density jumps at them.
   */
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    const double knot = gamma * lambda;
    k.count = 3;
    k.point[0] = mu - knot;
    k.point[1] = mu;
    k.point[2] = mu + knot;
    return k;
  }

  /// Starting value from a sample summary.
  mcp_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.median, params[0]), start_inside(1.0 / s.mean_abs_dev_median, params[1]),
            start_inside(3.0, params[2]), tail.start_value(s)};
  }
};

}  // namespace laplace::distrib
