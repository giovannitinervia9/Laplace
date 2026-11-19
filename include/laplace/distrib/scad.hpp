// scad_distrib (laplace::distrib): the properized smoothly clipped absolute
// deviation penalty, the density whose negative logarithm is Fan and Li's SCAD on
// an interval around the location and a tail of the user's choice beyond it.
// Parameters: the location mu (real), the rate lambda > 0, the shape a > 1 and the
// parameters of the tail. With u = |y - mu|, c = a - 1 and K = a lambda the second
// knot,
//
//   q(u) = lambda u                                            u <= lambda
//   q(u) = lambda u - (u - lambda)^2 / (2 c)                    lambda < u <= K
//   q(u) = (a + 1) lambda^2 / 2 + H(u - K)                      u > K
//
//   l(mu, lambda, a, theta_t; y) = -q(u) - log Z,
//   Z = 2 [ (1 - e^(-lambda^2))/lambda + e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2))
//           + e^(-(a + 1) lambda^2/2) T ],
//
// F being Dawson's integral (math/dawson.hpp), H the tail's exponent and
// T = int_0^inf exp(-H) its constant (distrib/properizing_tail.hpp). The original
// SCAD has a flat tail, which makes exp(-SCAD) unnormalizable for every parameter;
// replacing the flat part by a tail that grows makes the law proper and leaves the
// penalty untouched on u <= K, so scad_penalty and this law share their inner region
// exactly. The inner part of Z comes from completing the square in each piece, the
// middle exponent being convex in u, which is where Dawson's integral enters; it was
// checked against mpmath quadrature of the definition
// (tools/reference/distrib_mcp_scad.py).
//
// The tail is a template parameter whose default is student_tail. The three tails
// differ in the score far from the location: the Gaussian one has q' growing without
// bound, so beyond u = K + lambda V it shrinks more than a rate-lambda Laplace, which
// is the opposite of what the SCAD is for; the logistic one has q' bounded by 2/s; the
// Student one has q' -> 0, the SCAD's own behaviour up to O(1/u). The Gaussian tail is
// the limit of the Student one as its degrees of freedom grow at a fixed scale.
//
// Shape. The two knots are lambda and K. q and q' are continuous at both, q' being
// lambda on the two sides of the first and zero on the two sides of the second, while
// q'' jumps from 0 to -1/c and then from -1/c to H''(0), so the law is C^1 in y at both
// knots and C^0 at u = 0, where q' jumps from -lambda to lambda. The kink at the
// location is the only point of regularity 0, so y_regularity is 0, as for
// laplace2_distrib. The limits are the laws one expects: lambda -> 0 leaves no inner
// region and gives exp(-H) itself, and a lambda^2 -> infinity makes the tail's mass
// vanish and gives the Laplace of rate lambda.
//
// Derivatives. l is -q - log Z. y and mu enter through u only, with du/dy = sign(y - mu)
// and du/dmu = -sign(y - mu), so the component with a derivatives in mu, J in y and
// (b, m, n1, n2) in (lambda, a, theta_t) is
//
//   -(-1)^a sign(y - mu)^(a + J) d^(a+J)_u d^b_lambda d^m_a d^n_theta q
//   - [a + J = 0] d^b_lambda d^m_a d^n_theta log Z,
//
// where sign(y - mu)^t is read as 1 for even t and as sign(y - mu) with sign(0) = 0 for
// odd t: at the kink the odd orders are the mean of the two one-sided values, which is
// zero, and the even orders agree on the two sides and are taken from the inner piece.
// The inner pieces are quadratic in u, so every component with a + J >= 3 vanishes
// inside one of them; at a knot the value and every derivative come from the inner
// piece, as in scad_penalty. The mixed partials of q were derived and checked against
// SymPy in tools/reference/distrib_mcp_scad_forms.py and
// tools/reference/properized_tail_forms.py; those of log Z are built by the recursions
// of detail/properized_nonconvex.hpp, which the first script checks.
//
// Cumulative functions. The integral of exp(-q) from u to infinity is closed and free
// of cancellation on all three pieces, with the first piece written through expm1 so
// that nothing cancels as u approaches lambda and G(x) = int_x^inf exp(-H) the tail's
// own partial integral,
//
//   u <= lambda:        e^(-lambda u) (1 - e^(-lambda (lambda - u)))/lambda + M + T'
//   lambda < u <= K:    sqrt(2 c) e^(-q(u)) F((K - u)/sqrt(2 c)) + T'
//   u > K:              e^(-(a + 1) lambda^2/2) G(u - K),
//
// where M = e^(-lambda^2) sqrt(2 c) F(lambda sqrt(c/2)) is the whole middle piece and
// T' = e^(-(a + 1) lambda^2/2) T the whole tail, so F(y) is that integral over Z below
// the location and one minus it above. Its derivatives in the parameters, the expected
// information and the quantile are left to the generic routes (generic_cdf.hpp,
// generic_info.hpp, generic_quantile.hpp, generic_rng.hpp).
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

namespace detail_scad {

using namespace detail_properized;

// q(u)
template <class Tail>
LAPLACE_HD double q_value(double u, double lambda, double a, const Tail& tail) {
  if (u <= lambda) return lambda * u;
  if (u <= a * lambda) {
    const double v = u - lambda;
    return lambda * u - v * v / (2.0 * (a - 1.0));
  }
  return 0.5 * (a + 1.0) * lambda * lambda + detail_tail::h_mixed(tail, 0, 0, 0, u - a * lambda);
}

// one mixed partial d^j_u d^i_lambda d^m_a d^n1 d^n2 q, the last two in the tail's own
// parameters
template <class Tail>
LAPLACE_HD double q_mixed(int j, int i, int m, int n1, int n2, double u, double lambda, double a,
                          const Tail& tail) {
  if (u <= lambda) return (m == 0 && n1 == 0 && n2 == 0) ? q_rate(j, i, u, lambda) : 0.0;
  if (u <= a * lambda) {  // lambda u - (u - lambda)^2/(2 c), c = a - 1, dc/da = 1
    if (n1 > 0 || n2 > 0) return 0.0;
    double s = (m == 0) ? q_rate(j, i, u, lambda) : 0.0;
    const double v = u - lambda;
    const double sign = (i % 2 == 0) ? 1.0 : -1.0;  // (-1)^i, one factor per derivative in lambda
    const int tot = j + i;
    double w = 0.0;
    if (tot == 0) w = 0.5 * v * v;
    else if (tot == 1) w = sign * v;
    else if (tot == 2) w = sign;
    s -= w * inv_deriv(m, a - 1.0);
    return s;
  }
  // (a + 1) lambda^2/2 + H(u - a lambda)
  double s = 0.0;
  if (n1 == 0 && n2 == 0 && j == 0) {
    if (m == 0) s += 0.5 * d2_pow(i, lambda) * (a + 1.0);
    else if (m == 1) s += 0.5 * d2_pow(i, lambda);
  }
  return s + q_tail_piece(j, i, m, n1, n2, u, lambda, a, tail);
}

// d^i/dlambda^i of (1 - e^(-lambda^2))/lambda, i = 0..K. Below lambda = 1 the power
// series sum_(k >= 1) (-1)^(k+1) lambda^(2k-1)/k! is differentiated term by term,
// which is where it converges quickly and where its alternation costs nothing.
// Above it the Leibniz rule over (1 - e^(-lambda^2)) and 1/lambda is used, both
// factors being of size one there. The split is at lambda = 1 because the Leibniz
// route cancels like lambda^(-2) per order: at lambda = 0.1 and order six its terms
// reach 7.2e8 against a result of 21.
template <int K>
LAPLACE_HD void g_derivs(double lambda, double* g) {
  if (lambda < 1.0) {
    for (int i = 0; i <= K; ++i) {
      double s = 0.0, kf = 1.0;  // kf = k!
      for (int k = 1; k <= 60; ++k) {
        kf *= double(k);
        const int e = 2 * k - 1;  // the power of lambda before differentiating
        if (e < i) continue;
        double num = 1.0;  // e! / (e - i)!
        for (int t = 0; t < i; ++t) num *= double(e - t);
        double p = 1.0;
        for (int t = 0; t < e - i; ++t) p *= lambda;
        const double term = ((k % 2 == 1) ? 1.0 : -1.0) * num * p / kf;
        s += term;
        if (e - i > 2 && Kokkos::fabs(term) <= 0x1p-54 * Kokkos::fabs(s)) break;
      }
      g[i] = s;
    }
    return;
  }
  double e[K + 1];  // e[j] = d^j/dlambda^j of e^(-lambda^2), by d e = -2 lambda e
  e[0] = math::exp(-lambda * lambda);
  for (int j = 1; j <= K; ++j)
    e[j] = -2.0 * lambda * e[j - 1] - 2.0 * double(j - 1) * (j >= 2 ? e[j - 2] : 0.0);
  for (int i = 0; i <= K; ++i) {
    double s = 0.0;
    for (int j = 0; j <= i; ++j) {
      const double num = (j == 0) ? -Kokkos::expm1(-lambda * lambda) : -e[j];
      s += binomial_coef(i, j) * num * inv_deriv(i - j, lambda);
    }
    g[i] = s;
  }
}

// the tables over (lambda, a) of A and B in Z = 2 [A + B T]
template <int K>
LAPLACE_HD void z_tables(double lambda, double a, double* A, double* B) {
  const double c = a - 1.0;
  double dterm[tab2_size(K)], ee[tab2_size(K)], mid[tab2_size(K)], phi[tab2_size(K)];
  dawson_term<K>(lambda, c, dterm);  // sqrt(2 c) F(lambda sqrt(c/2)); dc/da = 1
  zero2<K>(phi);
  phi[idx2<K>(0, 0)] = -lambda * lambda;
  if constexpr (K >= 1) phi[idx2<K>(1, 0)] = -2.0 * lambda;
  if constexpr (K >= 2) phi[idx2<K>(2, 0)] = -2.0;
  exp2<K>(phi, ee);  // e^(-lambda^2)
  mul2<K>(ee, dterm, mid);
  double g[K + 1];
  g_derivs<K>(lambda, g);
  zero2<K>(A);
  for (int i = 0; i <= K; ++i) A[idx2<K>(i, 0)] = g[i];
  for (int i = 0; i <= K; ++i)
    for (int m = 0; i + m <= K; ++m) A[idx2<K>(i, m)] += mid[idx2<K>(i, m)];
  zero2<K>(phi);  // -(a + 1) lambda^2/2
  phi[idx2<K>(0, 0)] = -0.5 * (a + 1.0) * lambda * lambda;
  if constexpr (K >= 1) {
    phi[idx2<K>(1, 0)] = -(a + 1.0) * lambda;
    phi[idx2<K>(0, 1)] = -0.5 * lambda * lambda;
  }
  if constexpr (K >= 2) {
    phi[idx2<K>(2, 0)] = -(a + 1.0);
    phi[idx2<K>(1, 1)] = -lambda;
  }
  if constexpr (K >= 3) phi[idx2<K>(2, 1)] = -1.0;
  exp2<K>(phi, B);
}

// log Z, the three positive terms added in logs so that a knot large enough to
// underflow e^(-(a + 1) lambda^2/2) still leaves the tail's logarithm
template <class Tail>
LAPLACE_HD double log_z(double lambda, double a, const Tail& tail) {
  const double c = a - 1.0;
  const double first = -Kokkos::expm1(-lambda * lambda) / lambda;
  const double middle =
      math::exp(-lambda * lambda) * Kokkos::sqrt(2.0 * c) * math::dawson(lambda * Kokkos::sqrt(0.5 * c));
  return 0.69314718055994530942 +
         log_add(math::log(first + middle),
                 -0.5 * (a + 1.0) * lambda * lambda + tail.log_tail_const());
}

// Z itself, a sum of three positive terms. Where the last underflows the first two
// are the whole constant, which is the Laplace limit 2/lambda.
template <class Tail>
LAPLACE_HD double z_value(double lambda, double a, const Tail& tail) {
  const double c = a - 1.0;
  const double first = -Kokkos::expm1(-lambda * lambda) / lambda;
  const double middle =
      math::exp(-lambda * lambda) * Kokkos::sqrt(2.0 * c) * math::dawson(lambda * Kokkos::sqrt(0.5 * c));
  return 2.0 * (first + middle +
                math::exp(-0.5 * (a + 1.0) * lambda * lambda) * tail.tail_const());
}

// int_u^inf exp(-q(t)) dt, u >= 0
template <class Tail>
LAPLACE_HD double upper_integral(double u, double lambda, double a, const Tail& tail) {
  const double knot = a * lambda, C = 0.5 * (a + 1.0) * lambda * lambda;
  if (u > knot) return math::exp(-C) * tail.upper(u - knot);
  const double root = Kokkos::sqrt(2.0 * (a - 1.0));
  const double from = u <= lambda ? lambda : u;  // the middle piece starts at lambda
  double r = root * math::exp(-q_value(from, lambda, a, tail)) * math::dawson((knot - from) / root) +
             math::exp(-C) * tail.tail_const();
  if (u < lambda) r += math::exp(-lambda * u) * (-Kokkos::expm1(-lambda * (lambda - u))) / lambda;
  return r;
}

template <class Tail>
LAPLACE_HD double log_upper_integral(double u, double lambda, double a, const Tail& tail) {
  const double knot = a * lambda, C = 0.5 * (a + 1.0) * lambda * lambda;
  if (u > knot) return -C + tail.log_upper(u - knot);
  const double root = Kokkos::sqrt(2.0 * (a - 1.0));
  const double from = u <= lambda ? lambda : u;
  double r = log_add(math::log(root * math::dawson((knot - from) / root)) - q_value(from, lambda, a, tail),
                     -C + tail.log_tail_const());
  if (u < lambda)
    r = log_add(r, math::log(-Kokkos::expm1(-lambda * (lambda - u))) - lambda * u - math::log(lambda));
  return r;
}

// the parameter descriptors: the law's three, then the tail's own (host, read at a
// runtime index, as everywhere)
template <class Tail, int N>
constexpr std::array<distrib_param_info, N> param_list() {
  std::array<distrib_param_info, N> p{};
  p[0] = {"mu", -distrib_infinity, distrib_infinity, false, 0, false, 1};
  p[1] = {"lambda", 0.0, distrib_infinity, false, 1, false, 2};
  p[2] = {"a", 1.0, distrib_infinity, false, 1, false, 2};
  for (int j = 0; j < Tail::n_tail_params; ++j) p[3 + j] = Tail::tail_params[j];
  return p;
}

}  // namespace detail_scad

/**
 * \ingroup distrib_scad
 * \brief The properized smoothly clipped absolute deviation penalty, continuous on the
 * real line, in the location \f$\mu\f$, the rate \f$\lambda > 0\f$, the shape \f$a > 1\f$
 * and the parameters of the properizing tail.
 *
 * \tparam Tail The tail beyond the second knot: \c student_tail (the default),
 * \c gaussian_tail or \c logistic_tail.
 * \see distrib_scad, mcp_distrib, distrib_properizing_tail
 */
template <class Tail = student_tail>
struct scad_distrib {
  /// Location \f$\mu \in (-\infty, \infty)\f$, the mode and the median.
  double mu;
  /// Rate \f$\lambda \in (0, \infty)\f$, the slope of \f$q\f$ up to the first knot.
  double lambda;
  /// Shape \f$a \in (1, \infty)\f$; the knots sit at \f$u = \lambda\f$ and \f$u = a\lambda\f$.
  double a;
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
      detail_scad::param_list<Tail, n_params>();
  /// Descriptors of the parameters.
  static constexpr const distrib_param_info* params = params_.data();

  /// The log-density \f$-q(|y - \mu|) - \log Z\f$.
  LAPLACE_HD double logpdf(double y) const {
    return -detail_scad::q_value(Kokkos::fabs(y - mu), lambda, a, tail) -
           detail_scad::log_z(lambda, a, tail);
  }

  /// Derivatives of order \f$K\f$ of the log-density in \f$(\mu, \lambda, a, \theta_t)\f$.
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
      else t[0] = -u_chain_factor(0, J, s) * detail_scad::q_mixed(J, 0, 0, 0, 0, u, lambda, a, tail);
      return t;
    } else {
      double A[tab2_size(K)], B[tab2_size(K)], LZ[z_tab_size(K, nt)];
      if constexpr (J == 0) {
        detail_scad::z_tables<K>(lambda, a, A, B);
        log_z_table<K>(A, B, tail, LZ);
      }
      for (int i = 0; i <= K; ++i)
        for (int b = 0; i + b <= K; ++b)
          for (int m = 0; i + b + m <= K; ++m)
            for (int n1 = 0; i + b + m + n1 <= K; ++n1) {
              const int n2 = K - i - b - m - n1;
              if (nt == 1 && n2 != 0) continue;
              int idx[K];
              int k = 0;
              for (int w = 0; w < i; ++w) idx[k++] = 0;
              for (int w = 0; w < b; ++w) idx[k++] = 1;
              for (int w = 0; w < m; ++w) idx[k++] = 2;
              for (int w = 0; w < n1; ++w) idx[k++] = 3;
              for (int w = 0; w < n2; ++w) idx[k++] = 4;
              double v = -u_chain_factor(i, i + J, s) *
                         detail_scad::q_mixed(i + J, b, m, n1, n2, u, lambda, a, tail);
              if constexpr (J == 0) {
                if (i == 0) v -= LZ[z_tab_index<Tail>(b, m, n1, n2)];
              }
              t[sym_tensor_position<n_params, K>(idx)] = v;
            }
      return t;
    }
  }

  /// The cumulative distribution function \f$F(y)\f$.
  LAPLACE_HD double cdf(double y) const {
    const double r = detail_scad::upper_integral(Kokkos::fabs(y - mu), lambda, a, tail) /
                     detail_scad::z_value(lambda, a, tail);
    return y <= mu ? r : 1.0 - r;
  }
  /// The complementary cumulative distribution function \f$1 - F(y)\f$.
  LAPLACE_HD double ccdf(double y) const {
    const double r = detail_scad::upper_integral(Kokkos::fabs(y - mu), lambda, a, tail) /
                     detail_scad::z_value(lambda, a, tail);
    return y >= mu ? r : 1.0 - r;
  }
  // The side on which the probability is small is taken in logarithms, where it can
  // reach below the smallest double; the other side is one minus that probability,
  // taken as a plain ratio, which carries no error from the size of the exponent (see
  // the same pair in mcp.hpp).
  /// \f$\log F(y)\f$.
  LAPLACE_HD double log_cdf(double y) const {
    const double u = Kokkos::fabs(y - mu);
    if (y <= mu)
      return detail_scad::log_upper_integral(u, lambda, a, tail) - detail_scad::log_z(lambda, a, tail);
    return Kokkos::log1p(-detail_scad::upper_integral(u, lambda, a, tail) /
                         detail_scad::z_value(lambda, a, tail));
  }
  /// \f$\log(1 - F(y))\f$.
  LAPLACE_HD double log_ccdf(double y) const {
    const double u = Kokkos::fabs(y - mu);
    if (y >= mu)
      return detail_scad::log_upper_integral(u, lambda, a, tail) - detail_scad::log_z(lambda, a, tail);
    return Kokkos::log1p(-detail_scad::upper_integral(u, lambda, a, tail) /
                         detail_scad::z_value(lambda, a, tail));
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
      k.count = 2;  // the knots u = lambda and u = a lambda, both C^1
      k.point[0] = u / a;
      k.point[1] = u;
      k.continuous_order = 1;
    } else if (j == 2 && u > lambda) {
      k.count = 1;  // the knot u = a lambda, inside the domain a > 1 only when u > lambda
      k.point[0] = u / lambda;
      k.continuous_order = 1;
    }
    return k;  // the tail is smooth in its own parameters
  }

  /**
   * \ingroup distrib_scad
   * \brief Non-smooth points in \f$y\f$: the kink at \f$y = \mu\f$ and the four knots
   * \f$y = \mu \pm \lambda\f$ and \f$y = \mu \pm a\lambda\f$, with \c continuous_order 0.
   *
   * The kink is \f$C^0\f$ and the knots \f$C^1\f$; one order covers the five, so the
   * least of them is declared. The knots belong here although they are not kinks,
   * because the quadratures place a panel end at every point declared, and the second
   * derivative of the density jumps at them.
   */
  LAPLACE_HD distrib_kinks y_kinks() const {
    distrib_kinks k;
    k.count = 5;
    k.point[0] = mu - a * lambda;
    k.point[1] = mu - lambda;
    k.point[2] = mu;
    k.point[3] = mu + lambda;
    k.point[4] = mu + a * lambda;
    return k;
  }

  /// Starting value from a sample summary.
  scad_distrib start_value(const distrib_sample_summary& s) const {
    return {start_inside(s.median, params[0]), start_inside(1.0 / s.mean_abs_dev_median, params[1]),
            start_inside(3.7, params[2]), tail.start_value(s)};
  }
};

}  // namespace laplace::distrib
