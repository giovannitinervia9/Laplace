// The map from the von Mises circular standard deviation s to the concentration
// kappa, with its derivatives in s to order six (laplace::distrib).
//
// The von Mises is carried in the mean direction mu and the circular standard
// deviation
//
//     s = sqrt(-2 log rho),    rho = A(kappa) = I_1(kappa) / I_0(kappa),
//
// where rho is the mean resultant length and A the Bessel function ratio of
// math/bessel_i.hpp. So rho = exp(-s^2/2) and kappa = A^-1(exp(-s^2/2)), and s
// is the standard deviation of the wrapped normal with the same rho: s^2 is
// about 1/kappa when the law is concentrated, and about seventy per cent of the
// mass lies within s of mu while s stays well below pi.
//
// The map must be computed from s and must never form 1 - rho by subtraction.
// As s goes to zero, kappa grows like 1/s^2 while 1 - rho falls like s^2/2, so
// a complement obtained as 1 - rho carries an absolute error of half an ulp of
// one, which is a relative error of about 2 kappa eps on the quantity that
// determines kappa; below s = 1.5e-8 even rho itself rounds to one and the
// information is gone, while the true kappa, about 4e16, is representable.
//
// Three regions, each cancellation-free, with v = s^2 and
// q = 1 - rho = -expm1(-v/2), which keeps full relative accuracy for every s.
//
//   s <= s_series_small: kappa has a power series in v, got by inverting
//       u = -log A(kappa) as a series in 1/kappa,
//
//         kappa = 1/v + 1/2 + 5 v/24 + 3 v^2/16 + 1867 v^3/5760 + 199 v^4/256
//                 + 449083 v^5/193536 + 251821 v^6/30720
//                 + 5147579047 v^7/154828800,
//
//       whose derivatives in s are the series differentiated term by term, with
//       no cancellation at all. Measured relative error against an independent
//       root of A(kappa) = rho: 2.1e-17 at s = 1e-4, 4.5e-16 at s = 0.1 and
//       4.9e-11 at s = 0.2, which is what sets the switch point.
//
//   s >= s_series_large: kappa has a power series in rho, got by inverting
//       rho = A(kappa) around kappa = 0,
//
//         kappa = 2 rho + rho^3 + 5 rho^5/6 + 19 rho^7/24,
//
//       again differentiated term by term. Measured 2.4e-16 at s = 3, 7.2e-13
//       at s = 2.6 and 4.5e-8 at s = 2.0.
//
//   in between, where kappa runs from about 0.022 to 100 and nothing is small:
//       the value from bessel_i_ratio_inv_c(q), the inverse read from the
//       complement, and the derivatives by Faa di Bruno over
//       kappa = K(rho(s)). The outer derivatives K^(n) are taken from
//       bessel_i_ratio_inv_dn(kappa), which takes kappa and not rho, so the
//       ill-conditioned variable never enters; the inner ones are the exact
//       derivatives of exp(-s^2/2).
//
// Both series were derived with SymPy from the Hankel expansions of I_0 and
// I_1 and from the power series at zero, each checked by substitution
// (tools/reference/vonmises_kappa_series.py, modes small and large, which
// return an exact zero for the identity to the order kept). The switch points
// are the ones the measurements above give and the bench confirms.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/math/bessel_i.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>

namespace laplace::distrib::detail_vonmises {

/// Below this s the series in v = s^2 is used.
///
/// Measured at 0.1 the series carried the value to 4.5e-16 but not the high
/// orders, for the same reason as the other switch point: its truncation is of
/// relative size c_8 v^9, and the n-th derivative weights that term by a
/// falling factorial about a thousand times the leading term's, which at
/// s = 0.09 gave 33 ulp at order three and 137 at order six. At 0.05,
/// v^9 = 3.8e-24, which leaves the truncation far below eps at every order.
inline constexpr double s_series_small = 0.05;
/// From this s upwards the series in rho is used.
///
/// Measured at 3.0 the series carried the value to 2.4e-16 but not the high
/// orders: its truncation is of relative size rho^8, and differentiating it n
/// times multiplies that by about 9^n, which at s = 3 gave 699 ulp at order
/// three and 2.9e6 at order six. Moving the switch to 4.0, where
/// rho = 3.4e-4 and rho^8 = 1.6e-28, leaves 9^6 rho^8 = 8.5e-23 against an eps
/// of 1.1e-16, with room to spare, and costs nothing but one constant; the
/// alternative, two more terms of the series, would have bought the same thing
/// at more arithmetic.
inline constexpr double s_series_large = 4.0;

/// Number of terms kept of the series in v, which carries powers v^0 .. v^7.
inline constexpr int n_terms_small = 8;
/// Highest odd power of rho kept of the series in rho.
inline constexpr int n_terms_large = 7;

LAPLACE_HD_FORCEINLINE double nan() { return laplace::math::detail::quiet_nan(); }

// Coefficient of v^j in kappa = 1/v + sum_(j>=0) c_j v^j. Eight doubles, so a
// function-local constexpr array rather than a device table.
LAPLACE_HD_FORCEINLINE double coeff_small(int j) {
  constexpr double c[n_terms_small] = {
      0.5,                     // 1/2
      5.0 / 24.0,              // 0.208333...
      3.0 / 16.0,              // 0.1875
      1867.0 / 5760.0,         // 0.324131944...
      199.0 / 256.0,           // 0.77734375
      449083.0 / 193536.0,     // 2.320410672...
      251821.0 / 30720.0,      // 8.197298177...
      5147579047.0 / 154828800.0  // 33.246909147...
  };
  return c[j];
}

// Coefficient of rho^j in kappa = sum_j b_j rho^j, zero for even j.
LAPLACE_HD_FORCEINLINE double coeff_large(int j) {
  constexpr double b[n_terms_large + 1] = {0.0, 2.0, 0.0, 1.0, 0.0, 5.0 / 6.0, 0.0, 19.0 / 24.0};
  return b[j];
}

// The derivatives of order 0 .. N of E(s) = exp(-a s^2 / 2), from
// E' = -a s E and hence D_(n+1) = -a (s D_n + n D_(n-1)). Exact in the sense
// that it introduces no cancellation of its own: the alternation of signs is
// the function's, not the method's.
template <int N>
LAPLACE_HD_FORCEINLINE void gauss_dk(double s, double a, double (&d)[N + 1]) {
  d[0] = laplace::math::exp(-0.5 * a * s * s);
  if constexpr (N >= 1) d[1] = -a * s * d[0];
  for (int n = 1; n < N; ++n) d[n + 1] = -a * (s * d[n] + double(n) * d[n - 1]);
}

// kappa by the series in v = s^2.
LAPLACE_HD_FORCEINLINE double value_small(double s) {
  const double v = s * s;
  double p = 1.0, acc = 0.0;
  for (int j = 0; j < n_terms_small; ++j) {
    acc += coeff_small(j) * p;
    p *= v;
  }
  return 1.0 / v + acc;
}

// kappa by the series in rho.
LAPLACE_HD_FORCEINLINE double value_large(double s) {
  const double rho = laplace::math::exp(-0.5 * s * s);
  const double r2 = rho * rho;
  double p = rho, acc = 0.0;
  for (int j = 1; j <= n_terms_large; j += 2) {
    acc += coeff_large(j) * p;
    p *= r2;
  }
  return acc;
}

// kappa by the inverse of the ratio, each side of the band reading whichever of
// rho and its complement is the small one and therefore the one that carries
// kappa's relative accuracy. For rho above 1/2 (s below 1.1774) kappa is fixed
// by q = 1 - rho, which -expm1 gives exactly; below it kappa is fixed by rho
// itself, which exp gives exactly, and forming the complement there would be
// the wasteful direction.
LAPLACE_HD_FORCEINLINE double value_middle(double s) {
  const double u = 0.5 * s * s;
  const double rho = laplace::math::exp(-u);
  if (rho <= 0.5) return laplace::math::own::bessel_i_ratio_inv(rho);
  return laplace::math::own::bessel_i_ratio_inv_c(-Kokkos::expm1(-u));
}

// Derivative of order N of the series in v, term by term. The leading term
// 1/v = s^-2 gives (-1)^N (N+1)! s^-(N+2), and v^j = s^(2j) gives the falling
// factorial (2j)(2j-1)...(2j-N+1) times s^(2j-N), which vanishes for 2j < N.
template <int N>
LAPLACE_HD_FORCEINLINE double deriv_small(double s) {
  static_assert(N >= 1 && N <= 6, "orders one to six");
  double fact = 1.0;
  for (int i = 2; i <= N + 1; ++i) fact *= double(i);
  double pw = s * s;
  for (int i = 0; i < N; ++i) pw *= s;  // s^(N+2)
  double acc = ((N % 2) ? -fact : fact) / pw;
  for (int j = 0; j < n_terms_small; ++j) {
    const int m = 2 * j;
    if (m < N) continue;
    double ff = 1.0;
    for (int i = 0; i < N; ++i) ff *= double(m - i);
    double sp = 1.0;
    for (int i = 0; i < m - N; ++i) sp *= s;
    acc += coeff_small(j) * ff * sp;
  }
  return acc;
}

// Derivative of order N of the series in rho, term by term: rho^j is
// exp(-j s^2 / 2), whose derivatives come from gauss_dk with a = j.
template <int N>
LAPLACE_HD_FORCEINLINE double deriv_large(double s) {
  static_assert(N >= 1 && N <= 6, "orders one to six");
  double acc = 0.0;
  for (int j = 1; j <= n_terms_large; j += 2) {
    double d[N + 1];
    gauss_dk<N>(s, double(j), d);
    acc += coeff_large(j) * d[N];
  }
  return acc;
}

// The outer derivatives K^(k)(rho) at rho = A(kappa), from kappa.
LAPLACE_HD_FORCEINLINE double inverse_dk(int k, double kappa) {
  namespace own = laplace::math::own;
  switch (k) {
    case 1: return own::bessel_i_ratio_inv_d1(kappa);
    case 2: return own::bessel_i_ratio_inv_d2(kappa);
    case 3: return own::bessel_i_ratio_inv_d3(kappa);
    case 4: return own::bessel_i_ratio_inv_d4(kappa);
    case 5: return own::bessel_i_ratio_inv_d5(kappa);
    case 6: return own::bessel_i_ratio_inv_d6(kappa);
    default: return nan();
  }
}

// Derivative of order N in the middle region, by Faa di Bruno over
// kappa = K(rho(s)).
template <int N>
LAPLACE_HD_FORCEINLINE double deriv_middle(double s, double kappa) {
  static_assert(N >= 1 && N <= 6, "orders one to six");
  double f[N + 1], g[N + 1];
  f[0] = kappa;
  for (int k = 1; k <= N; ++k) f[k] = inverse_dk(k, kappa);
  gauss_dk<N>(s, 1.0, g);
  namespace m = laplace::math;
  if constexpr (N == 1) return m::chain_rule_d1(f, g);
  else if constexpr (N == 2) return m::chain_rule_d2(f, g);
  else if constexpr (N == 3) return m::chain_rule_d3(f, g);
  else if constexpr (N == 4) return m::chain_rule_d4(f, g);
  else if constexpr (N == 5) return m::chain_rule_d5(f, g);
  else return m::chain_rule_d6(f, g);
}

}  // namespace detail_vonmises

namespace laplace::distrib {

/**
 * \ingroup distrib_vonmises
 * \brief The von Mises concentration \f$\kappa\f$ from the circular standard deviation \f$s\f$.
 *
 * Solves \f$A(\kappa) = e^{-s^2/2}\f$ for \f$\kappa\f$, where
 * \f$A(\kappa) = I_1(\kappa)/I_0(\kappa)\f$ is the mean resultant length. Computed from
 * \f$s\f$ throughout: an 8-term Laurent series in \f$s^2\f$ for \f$s < 0.05\f$, a 4-term odd
 * power series in \f$\rho = e^{-s^2/2}\f$ for \f$s \ge 4\f$, and the inverse ratio read from
 * \f$-\operatorname{expm1}(-s^2/2)\f$ when \f$\rho > 0.5\f$ or \f$\rho\f$ otherwise on
 * \f$0.05 \le s < 4\f$, so that \f$1 - \rho\f$ is never formed by subtraction.
 *
 * \param s Circular standard deviation \f$s \in (0, \infty)\f$.
 * \return The value \f$\kappa \in (0, \infty)\f$, \f$+\infty\f$ at \f$s = 0\f$, and quiet \c NaN
 *         for \f$s < 0\f$ or \c NaN.
 *
 * \par Measured accuracy
 * Maximum error \f$5.32\f$ ulp on \f$s \in [10^{-8}, 4.5]\f$ (test bound \f$8\f$ ulp); above
 * \f$s = 4.5\f$, conditioned to \f$s^2\f$ ulp.
 *
 * \see distrib_vonmises, vonmises_kappa_from_s_dk(), vonmises_kappa_from_s_upto(),
 *      bessel_i_ratio_inv(), bessel_i_ratio_inv_c()
 */
LAPLACE_HD double vonmises_kappa_from_s(double s) {
  namespace d = detail_vonmises;
  if (!(s >= 0.0)) return d::nan();
  if (s == 0.0) return laplace::math::detail::pos_inf();
  if (s < d::s_series_small) return d::value_small(s);
  if (s >= d::s_series_large) return d::value_large(s);
  return d::value_middle(s);
}

/**
 * \ingroup distrib_vonmises
 * \brief Derivative of order \f$N\f$ of \f$\kappa(s)\f$, one function per order.
 *
 * For \f$s < 0.05\f$ and \f$s \ge 4\f$ the series is differentiated term by term, which carries no
 * cancellation of its own. On \f$0.05 \le s < 4\f$ the derivative is a Faà di Bruno composition
 * of \f$K^{(n)}\f$, evaluated at \f$\kappa\f$ rather than at \f$\rho\f$, with the exact derivatives
 * of \f$\rho(s) = e^{-s^2/2}\f$. Returns quiet \c NaN for \f$s \le 0\f$ or \c NaN.
 *
 * \tparam N Derivative order, \f$1 \le N \le 6\f$.
 * \param s Circular standard deviation \f$s \in (0, \infty)\f$.
 * \return The value \f$\mathrm{d}^N \kappa / \mathrm{d}s^N\f$, or quiet \c NaN outside \f$(0, \infty)\f$.
 *
 * \par Measured accuracy
 * Maximum error \f$6.12\f$ to \f$25.7\f$ ulp across orders \f$1\f$ to \f$6\f$ on
 * \f$s \in [10^{-8}, 4.5]\f$ (test bounds \f$10\f$ to \f$50\f$ ulp).
 *
 * \see distrib_vonmises, vonmises_kappa_from_s(), vonmises_kappa_from_s_upto(), bessel_i_ratio_inv_d1()
 */
template <int N>
LAPLACE_HD double vonmises_kappa_from_s_dk(double s) {
  namespace d = detail_vonmises;
  if (!(s > 0.0)) return d::nan();
  if (s < d::s_series_small) return d::deriv_small<N>(s);
  if (s >= d::s_series_large) return d::deriv_large<N>(s);
  return d::deriv_middle<N>(s, d::value_middle(s));
}

/**
 * \ingroup distrib_vonmises
 * \brief \f$\kappa(s)\f$ and its derivatives of orders \f$1\f$ to \f$N\f$ from one evaluation.
 *
 * Fills `out[0]` with \f$\kappa\f$ and `out[n]` with \f$\mathrm{d}^n \kappa/\mathrm{d}s^n\f$ for
 * \f$n = 1, \dots, N\f$. In the middle region \f$0.05 \le s < 4\f$ all orders share one solve for
 * \f$\kappa\f$ and one set of outer and inner derivatives. At \f$s = 0\f$, sets `out[0]` to
 * \f$+\infty\f$ and `out[1..N]` to quiet \c NaN; for \f$s < 0\f$ or \c NaN, sets `out[0..N]` to
 * quiet \c NaN.
 *
 * \tparam N Highest order, \f$1 \le N \le 6\f$.
 * \param s Circular standard deviation \f$s \in (0, \infty)\f$.
 * \param out Output array of at least \f$N + 1\f$ doubles.
 *
 * \par Measured accuracy
 * Agrees bit for bit with vonmises_kappa_from_s() and vonmises_kappa_from_s_dk() (maximum errors
 * \f$5.32\f$ to \f$25.7\f$ ulp, test bounds \f$8\f$ to \f$50\f$ ulp on \f$s \in [10^{-8}, 4.5]\f$).
 *
 * \par Scope
 * This joint form is an addition to the one function per derivative order, never a replacement for
 * them: it exists here because the saving was measured on this map, and a map elsewhere gets one only
 * when the same measurement is made for it.
 *
 * \see distrib_vonmises, vonmises_kappa_from_s(), vonmises_kappa_from_s_dk()
 */
template <int N>
LAPLACE_HD void vonmises_kappa_from_s_upto(double s, double* out) {
  namespace d = detail_vonmises;
  static_assert(N >= 1 && N <= 6, "orders one to six");
  if (!(s > 0.0)) {
    const double bad = (s == 0.0) ? laplace::math::detail::pos_inf() : d::nan();
    out[0] = bad;
    for (int n = 1; n <= N; ++n) out[n] = d::nan();
    return;
  }
  if (s < d::s_series_small) {
    out[0] = d::value_small(s);
    if constexpr (N >= 1) out[1] = d::deriv_small<1>(s);
    if constexpr (N >= 2) out[2] = d::deriv_small<2>(s);
    if constexpr (N >= 3) out[3] = d::deriv_small<3>(s);
    if constexpr (N >= 4) out[4] = d::deriv_small<4>(s);
    if constexpr (N >= 5) out[5] = d::deriv_small<5>(s);
    if constexpr (N >= 6) out[6] = d::deriv_small<6>(s);
    return;
  }
  if (s >= d::s_series_large) {
    out[0] = d::value_large(s);
    if constexpr (N >= 1) out[1] = d::deriv_large<1>(s);
    if constexpr (N >= 2) out[2] = d::deriv_large<2>(s);
    if constexpr (N >= 3) out[3] = d::deriv_large<3>(s);
    if constexpr (N >= 4) out[4] = d::deriv_large<4>(s);
    if constexpr (N >= 5) out[5] = d::deriv_large<5>(s);
    if constexpr (N >= 6) out[6] = d::deriv_large<6>(s);
    return;
  }
  // One solve for kappa, one set of outer and inner derivatives, then the
  // compositions of every order off the same two tables.
  const double kappa = d::value_middle(s);
  double f[7], g[7];
  f[0] = kappa;
  for (int k = 1; k <= N; ++k) f[k] = d::inverse_dk(k, kappa);
  d::gauss_dk<6>(s, 1.0, g);
  namespace m = laplace::math;
  out[0] = kappa;
  if constexpr (N >= 1) out[1] = m::chain_rule_d1(f, g);
  if constexpr (N >= 2) out[2] = m::chain_rule_d2(f, g);
  if constexpr (N >= 3) out[3] = m::chain_rule_d3(f, g);
  if constexpr (N >= 4) out[4] = m::chain_rule_d4(f, g);
  if constexpr (N >= 5) out[5] = m::chain_rule_d5(f, g);
  if constexpr (N >= 6) out[6] = m::chain_rule_d6(f, g);
}

}  // namespace laplace::distrib
