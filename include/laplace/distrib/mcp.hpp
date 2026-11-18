// mcp_distrib (laplace::distrib): the properized minimax concave penalty, the
// density whose negative logarithm is Zhang's MCP on an interval around the
// location and a Gaussian beyond it. Parameters: the location mu (real), the
// rate lambda > 0, the concavity gamma > 0 and the variance tail_sigma2 > 0 of
// the Gaussian tail. With u = |y - mu|,
//
//   q(u) = lambda u - u^2 / (2 gamma)                            u <= gamma lambda
//   q(u) = gamma lambda^2 / 2 + (u - gamma lambda)^2 / (2 V)     u >  gamma lambda
//
//   l(mu, lambda, gamma, V; y) = -q(u) - log Z,
//   Z = 2 [ sqrt(2 gamma) F(lambda sqrt(gamma / 2)) + e^(-gamma lambda^2/2) sqrt(pi V / 2) ],
//
// V written for tail_sigma2 and F for Dawson's integral (math/dawson.hpp). The
// original MCP has a flat tail, which makes exp(-MCP) unnormalizable for every
// parameter; replacing the flat part by a Gaussian of variance V makes the law
// proper and leaves the penalty untouched on u <= gamma lambda, so mcp_penalty
// and this law share their inner region exactly. Z comes from completing the
// square in each piece, the inner exponent being convex in u, which is where
// Dawson's integral enters; it was checked against mpmath quadrature of the
// definition (tools/reference/distrib_mcp_scad.py).
//
// Shape. gamma lambda is the knot. q and q' are continuous there, q' being zero
// on both sides, and q'' jumps from -1/gamma to 1/V, so the law is C^1 in y at
// the knot and C^0 at u = 0, where q' jumps from -lambda to lambda. The kink at
// the location is the only point of regularity 0, so y_regularity is 0, as for
// laplace2_distrib. The limits are the laws one expects: gamma -> 0 leaves no
// inner region and gives the normal of variance V, and gamma lambda^2 -> infinity
// makes the tail's mass vanish and gives the Laplace of rate lambda.
//
// Derivatives. l is -q - log Z, and q is a polynomial of degree two in u on each
// piece whose coefficients are rational in the parameters, so every mixed partial
// is closed. y and mu enter through u only, with du/dy = sign(y - mu) and
// du/dmu = -sign(y - mu), so the component with a derivatives in mu, J in y and
// (b, m, n) in (lambda, gamma, V) is
//
//   -(-1)^a sign(y - mu)^(a + J) d^(a+J)_u d^b_lambda d^m_gamma d^n_V q
//   - [a + J = 0] d^b_lambda d^m_gamma d^n_V log Z,
//
// where sign(y - mu)^t is read as 1 for even t and as sign(y - mu) with
// sign(0) = 0 for odd t: at the kink the odd orders are the mean of the two
// one-sided values, which is zero, and the even orders agree on the two sides and
// are taken from the inner piece. q is quadratic in u, so every component with
// a + J >= 3 vanishes inside a piece; at a knot the value and every derivative
// come from the inner piece, as in mcp_penalty. The mixed partials of q were
// derived and checked against SymPy in tools/reference/distrib_mcp_scad_forms.py;
// those of log Z are built by the recursions of
// detail/properized_nonconvex.hpp, which the same script checks.
//
// Cumulative functions. The integral of exp(-q) from u to infinity is closed and
// free of cancellation on both pieces,
//
//   u <= gamma lambda:  sqrt(2 gamma) e^(-q(u)) F((gamma lambda - u)/sqrt(2 gamma))
//                       + e^(-gamma lambda^2/2) sqrt(pi V / 2)
//   u >  gamma lambda:  2 e^(-gamma lambda^2/2) sqrt(pi V / 2) Phi(-(u - gamma lambda)/sqrt(V)),
//
// so F(y) is that integral over Z below the location and one minus it above. Its
// derivatives in the parameters, the expected information and the quantile are
// left to the generic routes (generic_cdf.hpp, generic_info.hpp,
// generic_quantile.hpp, generic_rng.hpp).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/properized_nonconvex.hpp>
#include <laplace/math/dawson.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/normal.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::distrib {

namespace detail_mcp {

using namespace detail_properized;

// q(u), V written for tail_sigma2
LAPLACE_HD double q_value(double u, double lambda, double gamma, double V) {
  const double knot = gamma * lambda;
  if (u <= knot) return lambda * u - u * u / (2.0 * gamma);
  const double w = u - knot;
  return 0.5 * gamma * lambda * lambda + w * w / (2.0 * V);
}

// one mixed partial d^j_u d^i_lambda d^m_gamma d^n_V q
LAPLACE_HD double q_mixed(int j, int i, int m, int n, double u, double lambda, double gamma, double V) {
  if (u <= gamma * lambda) {  // lambda u - u^2/(2 gamma); V does not appear
    if (n > 0) return 0.0;
    double s = (m == 0) ? q_rate(j, i, u, lambda) : 0.0;
    if (i == 0) s -= 0.5 * d2_pow(j, u) * inv_deriv(m, gamma);
    return s;
  }
  // gamma lambda^2/2 + (u - gamma lambda)^2/(2 V)
  double s = 0.0;
  if (n == 0 && j == 0) {
    if (m == 0) s += 0.5 * d2_pow(i, lambda) * gamma;
    else if (m == 1) s += 0.5 * d2_pow(i, lambda);
  }
  s += q_tail_square(j, i, m, u, lambda, gamma) * inv_deriv(n, V);
  return s;
}

// the tables over (lambda, gamma) of A and B in Z = 2 [A + B sqrt(pi V / 2)]
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
LAPLACE_HD double log_z(double lambda, double gamma, double V) {
  const double x = lambda * Kokkos::sqrt(0.5 * gamma);
  const double inner = Kokkos::sqrt(2.0 * gamma) * math::dawson(x);
  return 0.69314718055994530942 + log_add(math::log(inner), log_tail_integral(0.5 * gamma * lambda * lambda, V));
}

// Z itself, a sum of two positive terms. Where the second underflows the first is
// the whole constant, which is the Laplace limit 2/lambda.
LAPLACE_HD double z_value(double lambda, double gamma, double V) {
  const double x = lambda * Kokkos::sqrt(0.5 * gamma);
  return 2.0 * (Kokkos::sqrt(2.0 * gamma) * math::dawson(x) +
                math::exp(-0.5 * gamma * lambda * lambda) * 1.25331413731550025121 * Kokkos::sqrt(V));
}

// int_u^inf exp(-q(t)) dt, u >= 0
LAPLACE_HD double upper_integral(double u, double lambda, double gamma, double V) {
  const double knot = gamma * lambda, C = 0.5 * gamma * lambda * lambda;
  if (u > knot) return tail_upper(u - knot, C, V);
  const double root = Kokkos::sqrt(2.0 * gamma);
  return root * math::exp(-q_value(u, lambda, gamma, V)) * math::dawson((knot - u) / root) +
         math::exp(-C) * 1.25331413731550025121 * Kokkos::sqrt(V);
}

LAPLACE_HD double log_upper_integral(double u, double lambda, double gamma, double V) {
  const double knot = gamma * lambda, C = 0.5 * gamma * lambda * lambda;
  if (u > knot) return log_tail_upper(u - knot, C, V);
  const double root = Kokkos::sqrt(2.0 * gamma);
  const double a = math::log(root * math::dawson((knot - u) / root)) - q_value(u, lambda, gamma, V);
  return log_add(a, log_tail_integral(C, V));
}

}  // namespace detail_mcp

/**
 * \ingroup distrib_mcp
 * \brief The properized minimax concave penalty, continuous on the real line, in the
 * location \f$\mu\f$, the rate \f$\lambda > 0\f$, the concavity \f$\gamma > 0\f$ and the
 * variance \c tail_sigma2 \f$> 0\f$ of the Gaussian tail.
 * \see distrib_mcp, scad_distrib
 */
struct mcp_distrib {
  /// Location \f$\mu \in (-\infty, \infty)\f$, the mode and the median.
  double mu;
  /// Rate \f$\lambda \in (0, \infty)\f$, the slope of \f$q\f$ at the location.
  double lambda;
  /// Concavity \f$\gamma \in (0, \infty)\f$; the knot sits at \f$u = \gamma\lambda\f$.
  double gamma;
  /// Variance \f$V \in (0, \infty)\f$ of the Gaussian tail beyond the knot.
  double tail_sigma2;

  /// Distribution kind (\c distrib_kind::continuous).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// Number of natural parameters (4).
  static constexpr int n_params = 4;
  /// Differentiability order of the log-density in \f$y\f$ (0, a kink at \f$y = \mu\f$).
  static constexpr int y_regularity = 0;
  /// Descriptors of the four parameters.
  static constexpr distrib_param_info params[4] = {
      {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1},
      {"lambda", 0.0, distrib_infinity, false, 1, false, 2},
      {"gamma", 0.0, distrib_infinity, false, 1, false, 2},
      {"tail_sigma2", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// The log-density \f$-q(|y - \mu|) - \log Z\f$.
  LAPLACE_HD double logpdf(double y) const {
    return -detail_mcp::q_value(Kokkos::fabs(y - mu), lambda, gamma, tail_sigma2) -
           detail_mcp::log_z(lambda, gamma, tail_sigma2);
  }

  /// Derivatives of order \f$K\f$ of the log-density in \f$(\mu, \lambda, \gamma, V)\f$.
  template <int K>
  LAPLACE_HD sym_tensor<4, K> logpdf_dk(double y) const {
    return logpdf_dy_dk<0, K>(y);
  }
  /**
   * \name Derivatives of the log-density
   * @{
   */
  /// First-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 1> logpdf_d1(double y) const { return logpdf_dk<1>(y); }
  /// Second-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 2> logpdf_d2(double y) const { return logpdf_dk<2>(y); }
  /// Third-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 3> logpdf_d3(double y) const { return logpdf_dk<3>(y); }
  /// Fourth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 4> logpdf_d4(double y) const { return logpdf_dk<4>(y); }
  /// Fifth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 5> logpdf_d5(double y) const { return logpdf_dk<5>(y); }
  /// Sixth-order partial derivatives of \f$\log f\f$.
  LAPLACE_HD sym_tensor<4, 6> logpdf_d6(double y) const { return logpdf_dk<6>(y); }
  /** @} */

  /// \f$J\f$ derivatives in \f$y\f$ and the symmetric tensor of order \f$K\f$ in the parameters.
  template <int J, int K>
  LAPLACE_HD sym_tensor<4, K> logpdf_dy_dk(double y) const {
    using namespace detail_properized;
    const double u = Kokkos::fabs(y - mu);
    const double s = y > mu ? 1.0 : (y < mu ? -1.0 : 0.0);
    sym_tensor<4, K> t;
    if constexpr (K == 0) {
      if constexpr (J == 0) t[0] = logpdf(y);
      else
        t[0] = -u_chain_factor(0, J, s) * detail_mcp::q_mixed(J, 0, 0, 0, u, lambda, gamma, tail_sigma2);
      return t;
    } else {
      double A[tab2_size(K)], B[tab2_size(K)], LZ[tab3_size(K)];
      if constexpr (J == 0) {
        detail_mcp::z_tables<K>(lambda, gamma, A, B);
        log_z_table<K>(A, B, tail_sigma2, LZ);
      }
      for (int a = 0; a <= K; ++a)
        for (int b = 0; a + b <= K; ++b)
          for (int m = 0; a + b + m <= K; ++m) {
            const int n = K - a - b - m;
            int idx[K];
            int k = 0;
            for (int w = 0; w < a; ++w) idx[k++] = 0;
            for (int w = 0; w < b; ++w) idx[k++] = 1;
            for (int w = 0; w < m; ++w) idx[k++] = 2;
            for (int w = 0; w < n; ++w) idx[k++] = 3;
            double v = -u_chain_factor(a, a + J, s) *
                       detail_mcp::q_mixed(a + J, b, m, n, u, lambda, gamma, tail_sigma2);
            if constexpr (J == 0) {
              if (a == 0) v -= LZ[idx3(b, m, n)];
            }
            t[sym_tensor_position<4, K>(idx)] = v;
          }
      return t;
    }
  }

  /// The cumulative distribution function \f$F(y)\f$.
  LAPLACE_HD double cdf(double y) const {
    const double r = detail_mcp::upper_integral(Kokkos::fabs(y - mu), lambda, gamma, tail_sigma2) /
                     detail_mcp::z_value(lambda, gamma, tail_sigma2);
    return y <= mu ? r : 1.0 - r;
  }
  /// The complementary cumulative distribution function \f$1 - F(y)\f$.
  LAPLACE_HD double ccdf(double y) const {
    const double r = detail_mcp::upper_integral(Kokkos::fabs(y - mu), lambda, gamma, tail_sigma2) /
                     detail_mcp::z_value(lambda, gamma, tail_sigma2);
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
      return detail_mcp::log_upper_integral(u, lambda, gamma, tail_sigma2) -
             detail_mcp::log_z(lambda, gamma, tail_sigma2);
    return Kokkos::log1p(-detail_mcp::upper_integral(u, lambda, gamma, tail_sigma2) /
                         detail_mcp::z_value(lambda, gamma, tail_sigma2));
  }
  /// \f$\log(1 - F(y))\f$.
  LAPLACE_HD double log_ccdf(double y) const {
    const double u = Kokkos::fabs(y - mu);
    if (y >= mu)
      return detail_mcp::log_upper_integral(u, lambda, gamma, tail_sigma2) -
             detail_mcp::log_z(lambda, gamma, tail_sigma2);
    return Kokkos::log1p(-detail_mcp::upper_integral(u, lambda, gamma, tail_sigma2) /
                         detail_mcp::z_value(lambda, gamma, tail_sigma2));
  }

  /// Lower bound of support (\f$-\infty\f$).
  LAPLACE_HD double support_lower() const { return -distrib_infinity; }
  /// Upper bound of support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// Central location (\f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// Rough spread for the quadratures: the Laplace scale plus the tail's standard deviation.
  LAPLACE_HD double scale() const { return 1.0 / lambda + Kokkos::sqrt(tail_sigma2); }

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
    return k;
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
            start_inside(3.0, params[2]), start_inside(s.variance, params[3])};
  }
};

}  // namespace laplace::distrib
