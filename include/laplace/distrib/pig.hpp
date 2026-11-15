// pig_distrib (laplace::distrib): the Poisson-inverse Gaussian in the mean mu > 0 and the shape
// alpha > 0, on y = 0, 1, 2, ..., with s = sqrt(mu^2 + alpha^2),
//
//   p(y) = sqrt(2 alpha / pi) e^(s - mu) (mu alpha / (mu + s))^y K_(y - 1/2)(alpha) / y!,
//
// mean mu and variance mu + mu^2 / (s - mu). It is the mixture of a Poisson of mean mu Z with Z an
// inverse Gaussian of mean 1, written in the parametrization whose expected information is
// diagonal. The Poisson distribution is the limit alpha -> infinity at fixed mu; alpha -> 0 is the
// heavy-tailed end.
//
// The order of the Bessel function is always half an odd integer, so it is elementary: with
// m = max(y - 1, 0),
//
//   K_(y-1/2)(alpha) = sqrt(pi / (2 alpha)) e^-alpha sum_(k=0)^m (m+k)! / (k! (m-k)!) (2 alpha)^-k,
//
// a sum of positive terms. Its square root of alpha cancels the one in the mass, and the powers of
// alpha of the sum join the (mu alpha)^y, which leaves one exponential family in theta = log(2 alpha)
// over the shifted index j = y - k, which runs from 1 to y:
//
//   l = (s - mu) - alpha + y log mu - y log(mu + s) - y log 2 + log Q(theta) - loggamma(y + 1),
//   Q(theta) = sum_(j=1)^y a_j e^(j theta),   a_j = (2y - 1 - j)! / ((y - j)! (j - 1)!),
//
// with Q = 1 at y = 0, where the mass is e^(s - mu - alpha) exactly. So no Bessel routine, no Debye
// expansion and no table enter this family: everything is the logarithm of a positive sum and its
// cumulants.
//
// How the sum is evaluated. The terms a_j e^(j theta) are walked outwards from their largest one,
// whose index j* is the root in [1, y] of j^2 - (2y - 1 + 2 alpha) j + 2 alpha y = 0, by the exact
// ratio a_(j+1) e^theta / a_j = 2 alpha (y - j) / (j (2y - 1 - j)), and the walk stops when a term
// falls below 2^-80 of the largest. The cost is the effective width of the weights, not y: at a
// count of 1e6 the value costs what it costs at a count of 10.
//
// The value in a saddle-point form. Written as log Q - loggamma(y + 1) the mass keeps terms of the
// size of y log y while it is itself of the size of y log((mu + s) / (2 mu)), which the heavy-tailed
// end makes small: measured 2.6e5 ulp at (mu, alpha, y) = (1e3, 1e-3, 1e6). The four log-gammas of
// a_(j*) and of y! are therefore taken through Stirling's error, delta(a) = lgamma(a) - (a - 1/2)
// log a + a - log sqrt(2 pi) (math/stirling.hpp): their linear parts sum to j* + 2, the
// coefficients of log y to -3/2, and the coefficient of log 2 to -(j* + 1/2) once 2y log 2 is
// taken out, which is exactly what the remaining y log(2 mu / (mu + s)) needs. What is left of the
// four arguments' logarithms is three log1p whose arguments are of the size of j*/y, and the one
// term of the size of y is -y log1p(alpha^2 / (2 mu (s + mu))), written through
// s - mu = alpha^2 / (s + mu) so that the difference is never formed:
//
//   l = (s - mu - alpha) + (j* + 2) - log(2 pi)
//       + delta(2y - j*) - delta(y - j* + 1) - delta(j*) - delta(y + 1)
//       - (3/2) log y - (j* + 1/2) log 2 - (j* - 1/2) log j*
//       + (2y - j* - 1/2) log1p(-j*/(2y)) - (y - j* + 1/2) log1p((1 - j*)/y)
//       - (y + 1/2) log1p(1/y) + j* log(2 alpha) - y log1p(alpha^2/(2 mu (s + mu))) + log W,
//
// with W the sum of the terms scaled by the largest. Every term is then of the size of the result
// or of j*, which is itself about alpha. The constant s - mu - alpha is the log-mass at y = 0 and
// cancels at both ends, so it is taken as mu^2/(s + alpha) - mu when alpha >= mu and as
// alpha^2/(s + mu) - alpha otherwise: in each the positive term is at most half the negative one.
//
// Derivatives in alpha. d^q log Q / dtheta^q is the q-th cumulant of j under the weights
// a_j e^(j theta) / Q, taken from the central moments of the same walk (kappa_1 = <j> >= 1, so
// nothing cancels inside it), and theta = log(2 alpha) composes it into alpha
// (faa_di_bruno.hpp). The rest of the pure derivatives in alpha are the derivatives of s and of
// log(mu + s): d s / dalpha = alpha / s, and above the first order d^q s / dalpha^q =
// mu^2 d^(q-2) (mu^2 + alpha^2)^(-3/2) / dalpha^(q-2), which is free of the cancellation that
// 1/s - alpha^2/s^3 has when alpha is much larger than mu.
//
// Derivatives in the mean. The naive assembly carries terms of the size of y where the result is of
// the size of one, because the score in mu is exactly
//
//   dl/dmu = c (y - mu),   c = 1/mu - 1/s,
//
// (the terms of the size of y cancel identically, which is also why I_mu_mu = c). Every derivative
// with at least one index in mu is then that product differentiated, that is
//
//   d^(p+q) l / dmu^p dalpha^q = (y - mu) c_(p-1,q) - (p-1) c_(p-2,q),   p >= 1,
//
// with c_(i,j) the mixed partials of c. Those are taken from c = h(u) / mu with u = (alpha / mu)^2
// and h(u) = 1 - (1 + u)^(-1/2), whose value is written u / (w (w + 1)), w = sqrt(1 + u), and whose
// derivatives are h^(n)(u) = (-1)^(n+1) (2n)! / (4^n n!) (1 + u)^(-1/2-n): the difference
// 1/mu - 1/s loses everything when alpha is much smaller than mu, and h(u) does not. The two inner
// maps u and 1/mu are composed by chain_pair.hpp.
//
// The expected information is diagonal in this parametrization: I_mu_mu = c = 1/mu - 1/s and
// I_mu_alpha = 0 exactly, so the member supplies both of those closed (and the derivatives of
// I_mu_mu are the partials of c), and takes I_alpha_alpha and its derivatives from the generic sum
// over the support (generic_info.hpp). The cumulative function has no closed form: it and the
// derivatives of log F and log(1 - F) come from the generic summation route (generic_cdf.hpp).
// start_value inverts the moments: mu = mean, and from Var = mu + mu^2 / (s - mu),
// alpha = sqrt(d (d + 2 mu)) with d = s - mu = mu^2 / (Var - mu), the denominator floored at
// mu / 100 so that an underdispersed sample starts near the Poisson limit.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/generic_info.hpp>
#include <laplace/math/chain_pair.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/faa_di_bruno.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace laplace::distrib {

namespace detail_pig {

using math::detail_chain_pair::pair_count;
using math::detail_chain_pair::pair_index;

// s - mu - alpha, the log-mass at y = 0 and the constant of every other count, written so that
// neither difference cancels: s - alpha = mu^2 / (s + alpha) and s - mu = alpha^2 / (s + mu) are
// both exact, and the one whose small term is at most half the other is taken, so the result is a
// sum of a positive term below half and a negative one
LAPLACE_HD_FORCEINLINE double excess_minus_shape(double mu, double alpha, double s) {
  return alpha >= mu ? mu * mu / (s + alpha) - mu : alpha * alpha / (s + mu) - alpha;
}

// the largest index of a_j e^(j theta): the root in [1, y] of j^2 - (2y - 1 + 2 alpha) j + 2 alpha y,
// rounded to the neighbour with the larger term (the ratio of the two is the exact ratio below)
LAPLACE_HD double mode_index(double y, double alpha) {
  const double b = 2.0 * y - 1.0 + 2.0 * alpha, c = 2.0 * alpha * y;
  const double disc = b * b - 4.0 * c;
  double j = disc > 0.0 ? 0.5 * (b - Kokkos::sqrt(disc)) : 1.0;
  j = Kokkos::floor(j);
  if (!(j >= 1.0)) j = 1.0;
  if (j > y) j = y;
  // the ratio t_(j+1) / t_j; the root is already within a step, so a few corrections suffice
  for (int k = 0; k < 4 && j < y; ++k) {
    if (!(2.0 * alpha * (y - j) > j * (2.0 * y - 1.0 - j))) break;
    j += 1.0;
  }
  return j;
}

// The walk over the weights a_j e^(j theta), scaled by their largest term. Returns the logarithm
// of the scaled sum, log W = log(sum_j t_j / t_(j*)) >= 0, writes the index of the largest term to
// j_max and the central moments of j of orders 2..N to central[2..N], the first moment to mean_j.
// N <= 6; at y = 0 there is no term and everything is zero.
template <int N>
LAPLACE_HD double log_sum_and_moments(double y, double alpha, double& j_max, double& mean_j, double* central) {
  for (int r = 0; r <= N; ++r) central[r] = 0.0;
  mean_j = 0.0;
  j_max = 0.0;
  if (!(y >= 1.0)) return 0.0;
  const double jm = mode_index(y, alpha);
  j_max = jm;
  constexpr double cut = 0x1p-80;
  constexpr long max_steps = 1L << 24;
  // first pass: the scaled sum and the first moment about j*
  double sum = 1.0, m1 = 0.0;
  for (int dir = 0; dir < 2; ++dir) {
    double t = 1.0, j = jm;
    for (long n = 0; n < max_steps; ++n) {
      // the ratio towards the next index in this direction
      double r;
      if (dir == 0) {  // upwards: t_(j+1) / t_j
        if (j >= y) break;
        r = 2.0 * alpha * (y - j) / (j * (2.0 * y - 1.0 - j));
        j += 1.0;
      } else {  // downwards: t_(j-1) / t_j
        if (j <= 1.0) break;
        r = (j - 1.0) * (2.0 * y - j) / (2.0 * alpha * (y - j + 1.0));
        j -= 1.0;
      }
      t *= r;
      if (!(t > cut)) break;
      sum += t;
      m1 += t * (j - jm);
    }
  }
  mean_j = jm + m1 / sum;
  // second pass: the central moments, the same walk repeated. The term at j* is added once, before
  // the two directions.
  if constexpr (N >= 2) {
  {
    const double d0 = jm - mean_j;
    double p = 1.0;
    for (int r = 2; r <= N; ++r) {
      p = (r == 2) ? d0 * d0 : p * d0;
      central[r] += p;
    }
  }
  for (int dir = 0; dir < 2; ++dir) {
    double t = 1.0, j = jm, d, p;
    for (long n = 0; n < max_steps; ++n) {
      double r;
      if (dir == 0) {
        if (j >= y) break;
        r = 2.0 * alpha * (y - j) / (j * (2.0 * y - 1.0 - j));
        j += 1.0;
      } else {
        if (j <= 1.0) break;
        r = (j - 1.0) * (2.0 * y - j) / (2.0 * alpha * (y - j + 1.0));
        j -= 1.0;
      }
      t *= r;
      if (!(t > cut)) break;
      d = j - mean_j;
      p = 1.0;
      for (int q = 2; q <= N; ++q) {
        p = (q == 2) ? d * d : p * d;
        central[q] += t * p;
      }
    }
  }
  }
  for (int r = 2; r <= N; ++r) central[r] /= sum;
  return math::log(sum);
}

// the cumulants of j from its mean and central moments, orders 1..N (N <= 6)
template <int N>
LAPLACE_HD void cumulants_of(double mean_j, const double* m, double* kap) {
  kap[0] = 0.0;
  if constexpr (N >= 1) kap[1] = mean_j;
  if constexpr (N >= 2) kap[2] = m[2];
  if constexpr (N >= 3) kap[3] = m[3];
  if constexpr (N >= 4) kap[4] = m[4] - 3.0 * m[2] * m[2];
  if constexpr (N >= 5) kap[5] = m[5] - 10.0 * m[3] * m[2];
  if constexpr (N >= 6) kap[6] = m[6] - 15.0 * m[4] * m[2] - 10.0 * m[3] * m[3] + 30.0 * m[2] * m[2] * m[2];
}

// g(x) = (1 + x)^(-1/2) and its derivatives of orders 0..N:
// g^(n)(x) = (-1)^n (2n)! / (4^n n!) (1 + x)^(-1/2-n), so h^(n) = -g^(n) for n >= 1
template <int N>
LAPLACE_HD void recip_sqrt_derivs(double x, double* out) {
  const double w = Kokkos::sqrt(1.0 + x);
  out[0] = 1.0 / w;
  double p = out[0], coef = 1.0;
  for (int n = 1; n <= N; ++n) {
    p /= (1.0 + x);
    coef *= (2.0 * n - 1.0) / 2.0;  // (2n)!/(4^n n!) = (2n-1)!! / 2^n
    out[n] = ((n & 1) ? -coef : coef) * p;
  }
}

// The mixed partials of c = 1/mu - 1/s and of m = mu / s, for every total order up to K (K <= 5):
// C[pair_index(i, j)] = d^(i+j) c / dmu^i dalpha^j and M likewise for m.
//
// Both are functions of the one ratio alpha / mu, and which of its two powers is the small one
// decides the composition, because an inner map that is large makes every term of the chain large
// while the result is not. The split is at alpha = mu:
//
//   alpha <= mu:  u = (alpha / mu)^2 <= 1,  c = h(u) / mu,  m = 1 - h(u),
//   alpha >  mu:  v = (mu / alpha)^2 < 1,   c = 1/mu - g(v) / alpha,  m = (mu / alpha) g(v),
//
// with h(x) = 1 - (1 + x)^(-1/2) and g(x) = (1 + x)^(-1/2). Taking the first form everywhere costs
// the Poisson limit: at mu = 1, alpha = 1e3 every term of the chain for d^5 m / dmu^5 is of the
// size of 1e-3 while the result is 4.5e-14, measured 6.8e12 ulp on the order-6 tensor at y = 0. In
// each branch the inner maps are products of a power of one parameter and a power of the other,
// so their partials are closed, and the outer map is the product G(u_0, u_1) = u_1 g(u_0) or
// u_1 h(u_0), whose partials are g^(i)(u_0) u_1 at no derivative in u_1, g^(i)(u_0) at one and zero
// beyond. chain_pair.hpp composes them, once per order.
template <int K>
LAPLACE_HD void c_partials(double mu, double alpha, double* C, double* M) {
  constexpr int n = pair_count(K);
  for (int i = 0; i < n; ++i) C[i] = M[i] = 0.0;
  const double s = Kokkos::hypot(mu, alpha);
  M[0] = mu / s;
  const bool low = alpha <= mu;
  const double r = low ? alpha / mu : mu / alpha, x = r * r;
  double Gx[K + 1];
  recip_sqrt_derivs<K>(x, Gx);
  const double w = Kokkos::sqrt(1.0 + x);
  C[0] = low ? (x / (w * (w + 1.0))) / mu : 1.0 / mu - Gx[0] / alpha;
  if constexpr (K >= 1) {
    double G[pair_count(K)], GM[pair_count(K)], U0[pair_count(K)], U1[pair_count(K)], U1m[pair_count(K)];
    for (int i = 0; i < n; ++i) G[i] = GM[i] = U0[i] = U1[i] = U1m[i] = 0.0;
    // the inner map x: (alpha^2) (mu^-2) when alpha <= mu, (mu^2) (alpha^-2) otherwise. The partials
    // of a product of a function of mu and a function of alpha are the product of theirs.
    const double big = low ? mu : alpha, small = low ? alpha : mu;
    double fb = 1.0 / (big * big);  // (-1)^p (p+1)! / big^(p+2)
    for (int p = 0; p <= K; ++p) {
      const double cb = ((p & 1) ? -fb : fb);
      const double ds[3] = {small * small, 2.0 * small, 2.0};
      for (int q = 0; q <= 2; ++q)
        if (p + q <= K) U0[low ? pair_index(p, q) : pair_index(q, p)] = cb * ds[q];
      fb *= (p + 2.0) / big;
    }
    // u_1 = 1/mu (alpha <= mu) or 1/alpha (alpha > mu), and u_1m = mu/s's own second factor
    double f1 = 1.0 / big;  // (-1)^p p! / big^(p+1)
    for (int p = 0; p <= K; ++p) {
      const double c1 = ((p & 1) ? -f1 : f1);
      U1[low ? pair_index(p, 0) : pair_index(0, p)] = c1;
      if (low) {
        U1m[pair_index(p, 0)] = c1;  // unused: m = 1 - h(u) has no factor
      } else {
        // u_1m = mu / alpha: (mu or 1) times (-1)^p p! / alpha^(p+1)
        U1m[pair_index(0, p)] = c1 * mu;
        if (p + 1 <= K) U1m[pair_index(1, p)] = c1;
      }
      f1 *= (p + 1.0) / big;
    }
    for (int i = 0; i <= K; ++i) {
      // h^(i) = -g^(i) for i >= 1, and the value is only ever C[0] or M[0]
      const double hi = i == 0 ? (x / (w * (w + 1.0))) : -Gx[i];
      if (low) {
        G[pair_index(i, 0)] = hi / mu;
        if (i + 1 <= K) G[pair_index(i, 1)] = hi;
        GM[pair_index(i, 0)] = -hi;
      } else {
        G[pair_index(i, 0)] = -Gx[i] / alpha;
        if (i + 1 <= K) G[pair_index(i, 1)] = -Gx[i];
        GM[pair_index(i, 0)] = Gx[i] * (mu / alpha);
        if (i + 1 <= K) GM[pair_index(i, 1)] = Gx[i];
      }
    }
    double out[K + 1];
#define LAPLACE_PIG_CHAIN(ORD)                                                      \
  if constexpr (K >= ORD) {                                                         \
    math::chain_rule_pair_dk<ORD>(G, U0, U1, out);                                  \
    for (int b = 0; b <= ORD; ++b) C[pair_index(ORD - b, b)] = out[b];              \
    math::chain_rule_pair_dk<ORD>(GM, U0, low ? U1 : U1m, out);                     \
    for (int b = 0; b <= ORD; ++b) M[pair_index(ORD - b, b)] = out[b];              \
  }
    LAPLACE_PIG_CHAIN(1)
    LAPLACE_PIG_CHAIN(2)
    LAPLACE_PIG_CHAIN(3)
    LAPLACE_PIG_CHAIN(4)
    LAPLACE_PIG_CHAIN(5)
#undef LAPLACE_PIG_CHAIN
    // above the split c also carries the pure derivatives of its own 1/mu
    if (!low) {
      double fm = 1.0 / mu;
      for (int p = 1; p <= K; ++p) {
        fm *= p / mu;  // p! / mu^(p+1)
        C[pair_index(p, 0)] += ((p & 1) ? -fm : fm);
      }
    }
  }
}

// d^q s / dalpha^q, q = 0..Q (Q <= 6), s = sqrt(mu^2 + alpha^2): the value, alpha / s at the first
// order, and above it mu^2 d^(q-2) (mu^2 + alpha^2)^(-3/2) / dalpha^(q-2), which is exact where
// 1/s - alpha^2/s^3 cancels
template <int Q>
LAPLACE_HD void s_alpha_derivs(double mu, double alpha, double* out) {
  // v is taken as s^2 rather than mu^2 + alpha^2, so that 1/(v s) is exactly the cube of the same
  // 1/s the first order uses and neither square overflows before s does
  const double s = Kokkos::hypot(mu, alpha), v = s * s;
  out[0] = s;
  if constexpr (Q >= 1) out[1] = alpha / s;
  if constexpr (Q >= 2) {
    // g(alpha) = v^(-3/2) composed from f(v) = v^(-3/2) and v(alpha) = mu^2 + alpha^2
    double f[Q + 1], g[Q + 1];
    double pw = 1.0 / (v * s);  // v^(-3/2)
    double coef = 1.0;
    f[0] = pw;
    for (int k = 1; k <= Q; ++k) {
      coef *= -(1.5 + (k - 1));
      pw /= v;
      f[k] = coef * pw;
    }
    g[0] = v;
    g[1] = 2.0 * alpha;
    g[2] = 2.0;
    for (int k = 3; k <= Q; ++k) g[k] = 0.0;
    const double mu2 = mu * mu;
    out[2] = mu2 * f[0];
    if constexpr (Q >= 3) out[3] = mu2 * math::chain_rule_d1(f, g);
    if constexpr (Q >= 4) out[4] = mu2 * math::chain_rule_d2(f, g);
    if constexpr (Q >= 5) out[5] = mu2 * math::chain_rule_d3(f, g);
    if constexpr (Q >= 6) out[6] = mu2 * math::chain_rule_d4(f, g);
  }
}

// d^q log(mu + s) / dalpha^q, q = 0..Q (Q <= 6), from the derivatives of s
template <int Q>
LAPLACE_HD void log_mu_plus_s_derivs(double mu, double alpha, double* out) {
  double sd[Q + 1];
  s_alpha_derivs<Q>(mu, alpha, sd);
  const double A = mu + sd[0];
  double f[Q + 1];
  f[0] = math::log(A);
  double pw = 1.0;
  double fact = 1.0;
  for (int k = 1; k <= Q; ++k) {
    pw *= A;
    if (k >= 2) fact *= (k - 1);
    f[k] = ((k & 1) ? fact : -fact) / pw;
  }
  out[0] = f[0];
  if constexpr (Q >= 1) out[1] = math::chain_rule_d1(f, sd);
  if constexpr (Q >= 2) out[2] = math::chain_rule_d2(f, sd);
  if constexpr (Q >= 3) out[3] = math::chain_rule_d3(f, sd);
  if constexpr (Q >= 4) out[4] = math::chain_rule_d4(f, sd);
  if constexpr (Q >= 5) out[5] = math::chain_rule_d5(f, sd);
  if constexpr (Q >= 6) out[6] = math::chain_rule_d6(f, sd);
}

}  // namespace detail_pig

/**
 * \ingroup distrib_pig
 * \brief Poisson-inverse Gaussian distribution with mean \f$\mu > 0\f$ and shape \f$\alpha > 0\f$.
 *
 * Mass \f$p(y) = \sqrt{2\alpha/\pi}\, e^{s - \mu} \left(\mu\alpha/(\mu + s)\right)^y
 * K_{y - 1/2}(\alpha) / y!\f$ on \f$y = 0, 1, 2, \dots\f$ with \f$s = \sqrt{\mu^2 + \alpha^2}\f$,
 * mean \f$\mu\f$ and variance \f$\mu + \mu^2/(s - \mu)\f$; the Poisson distribution is the limit
 * \f$\alpha \to \infty\f$ at fixed \f$\mu\f$. The order of the Bessel function is always half an odd
 * integer, so the mass is elementary, and the expected information is diagonal.
 * \see distrib_pig, distrib_core
 */
struct pig_distrib {
  /// Mean parameter \f$\mu \in (0, \infty)\f$.
  double mu;
  /// Shape parameter \f$\alpha \in (0, \infty)\f$; the Poisson distribution is the limit \f$\alpha \to \infty\f$.
  double alpha;

  /// Distribution kind (\c distrib_kind::discrete).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// Number of natural parameters (2).
  static constexpr int n_params = 2;
  /// Descriptors of \c mu and \c alpha.
  static constexpr distrib_param_info params[2] = {
      {"mu", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth},
      {"alpha", 0.0, distrib_infinity, false, infinitely_smooth, false, infinitely_smooth}};

  /// \ingroup distrib_pig
  /// \brief The radius \f$s = \sqrt{\mu^2 + \alpha^2}\f$ that every quantity is written with.
  LAPLACE_HD double radius() const { return Kokkos::hypot(mu, alpha); }
  /// \ingroup distrib_pig
  /// \brief The excess \f$s - \mu = \alpha^2/(s + \mu)\f$, free of the cancellation of the difference.
  LAPLACE_HD double excess() const { return alpha * alpha / (radius() + mu); }

  /**
   * \ingroup distrib_pig
   * \brief Evaluates the log-mass \f$\log p(y; \mu, \alpha)\f$.
   *
   * Returns \f$-\infty\f$ when \f$y\f$ is not a non-negative integer. At \f$y = 0\f$ the value is
   * \f$s - \mu - \alpha\f$ exactly; above it the powers of \f$\alpha\f$ of the half-integer Bessel
   * polynomial join the \f$(\mu\alpha)^y\f$ and the sum is walked outwards from its largest term.
   * \param y Observation point.
   * \return Log-mass value.
   * \see distrib_pig, logpmf_dk()
   */
  LAPLACE_HD double logpmf(double y) const {
    if (!(y >= 0.0) || y != Kokkos::floor(y)) return -distrib_infinity;
    const double s = radius();
    const double base = detail_pig::excess_minus_shape(mu, alpha, s);
    if (y == 0.0) return base;
    constexpr double log2 = 0.69314718055994530942, log_2pi = 1.8378770664093454836;
    double j, mean_j, m[2];
    const double logW = detail_pig::log_sum_and_moments<1>(y, alpha, j, mean_j, m);
    // Stirling's error at the four log-gammas of a_(j*) and of y!, the linear parts summing to
    // j + 2 and the logarithms of y to -(3/2) log y exactly; what is left of the arguments'
    // logarithms is three log1p of the size of j, and the one term of the size of y is
    // -y log((mu + s) / (2 mu)), taken through s - mu = alpha^2 / (s + mu)
    const double L = Kokkos::log1p(alpha * alpha / (2.0 * mu * (s + mu)));
    return base + (j + 2.0) - log_2pi + math::stirling_error(2.0 * y - j) -
           math::stirling_error(y - j + 1.0) - math::stirling_error(j) - math::stirling_error(y + 1.0) -
           1.5 * math::log(y) - (j + 0.5) * log2 - (j - 0.5) * math::log(j) +
           (2.0 * y - j - 0.5) * Kokkos::log1p(-j / (2.0 * y)) -
           (y - j + 0.5) * Kokkos::log1p((1.0 - j) / y) - (y + 0.5) * Kokkos::log1p(1.0 / y) +
           j * (log2 + math::log(alpha)) - y * L + logW;
  }

  /**
   * \ingroup distrib_pig
   * \brief Evaluates partial derivatives of order \f$K\f$ of \f$\log p(y; \mu, \alpha)\f$.
   *
   * Component \f$b\f$ holds \f$\partial^K \log p / \partial\mu^{K-b} \partial\alpha^{b}\f$. The
   * components with at least one derivative in \f$\mu\f$ come from the exact score
   * \f$\partial\ell/\partial\mu = c\,(y - \mu)\f$, \f$c = 1/\mu - 1/s\f$, differentiated once more;
   * the pure derivative in \f$\alpha\f$ from the cumulants of the Bessel sum.
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param y Observation point \f$y \ge 0\f$.
   * \return Packed symmetric tensor \c sym_tensor<2, K> of derivatives.
   */
  template <int K>
  LAPLACE_HD sym_tensor<2, K> logpmf_dk(double y) const {
    sym_tensor<2, K> t;
    // the components with p = K - b >= 1 derivatives in mu. The score is c (y - mu), whose product
    // is differentiated as y c - (1 - m), m = mu / s: the first order takes the factored form, which
    // is exactly 0 at y = mu, and every higher order the expanded one, because
    // (y - mu) c_(p-1,q) - (p-1) c_(p-2,q) splits a result of the size of alpha^-(p-1) into terms of
    // the size of one at the Poisson limit (measured 4.1e15 ulp at order 6, mu = 1, alpha = 1e3,
    // y = 0, where the whole tensor is d^K s / dmu^K), while y c_(p-1,q) + m_(p-1,q) is one term
    // there and never worse than the factored form elsewhere.
    {
      constexpr int N = K - 1 > 5 ? 5 : (K - 1 > 0 ? K - 1 : 0);
      double C[math::detail_chain_pair::pair_count(N)], Mp[math::detail_chain_pair::pair_count(N)];
      detail_pig::c_partials<N>(mu, alpha, C, Mp);
      using math::detail_chain_pair::pair_index;
      if constexpr (K == 1) {
        t[0] = (y - mu) * C[0];
      } else {
        for (int b = 0; b < K; ++b) {
          const int p = K - b, q = b;
          t[b] = y * C[pair_index(p - 1, q)] + Mp[pair_index(p - 1, q)];
        }
      }
    }
    // the pure derivative in alpha
    {
      double sd[K + 1], ld[K + 1];
      detail_pig::s_alpha_derivs<K>(mu, alpha, sd);
      detail_pig::log_mu_plus_s_derivs<K>(mu, alpha, ld);
      double j_max, mean_j, m[7], kap[8];
      detail_pig::log_sum_and_moments<K>(y, alpha, j_max, mean_j, m);
      detail_pig::cumulants_of<K>(mean_j, m, kap);
      // theta = log(2 alpha): d^k theta / dalpha^k = (-1)^(k-1) (k-1)! / alpha^k
      double th[K + 1];
      th[0] = 0.0;
      double pw = 1.0, fact = 1.0;
      for (int k = 1; k <= K; ++k) {
        pw *= alpha;
        if (k >= 2) fact *= (k - 1);
        th[k] = ((k & 1) ? fact : -fact) / pw;
      }
      double q_term = 0.0;
      if (y >= 1.0) {
        if constexpr (K == 1) q_term = math::chain_rule_d1(kap, th);
        else if constexpr (K == 2) q_term = math::chain_rule_d2(kap, th);
        else if constexpr (K == 3) q_term = math::chain_rule_d3(kap, th);
        else if constexpr (K == 4) q_term = math::chain_rule_d4(kap, th);
        else if constexpr (K == 5) q_term = math::chain_rule_d5(kap, th);
        else q_term = math::chain_rule_d6(kap, th);
      }
      // at the first order alpha/s - 1 is -mu^2 / (s (s + alpha)), which does not cancel when
      // alpha is much larger than mu
      const double lead = K == 1 ? -mu * mu / (sd[0] * (sd[0] + alpha)) : sd[K];
      t[K] = lead - y * ld[K] + q_term;
    }
    return t;
  }
  /**
   * \name Derivatives of the log-mass
   * Parameter derivatives of \f$\log p(y; \mu, \alpha)\f$ for orders \f$K = 1 \dots 6\f$.
   * @{
   */
  /// First-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 1> logpmf_d1(double y) const { return logpmf_dk<1>(y); }
  /// Second-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 2> logpmf_d2(double y) const { return logpmf_dk<2>(y); }
  /// Third-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 3> logpmf_d3(double y) const { return logpmf_dk<3>(y); }
  /// Fourth-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 4> logpmf_d4(double y) const { return logpmf_dk<4>(y); }
  /// Fifth-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 5> logpmf_d5(double y) const { return logpmf_dk<5>(y); }
  /// Sixth-order derivatives of \f$\log p\f$ in \f$(\mu, \alpha)\f$.
  LAPLACE_HD sym_tensor<2, 6> logpmf_d6(double y) const { return logpmf_dk<6>(y); }
  /** @} */

  /**
   * \ingroup distrib_pig
   * \brief Evaluates the expected Fisher information, diagonal in this parametrization.
   *
   * \f$I_{\mu\mu} = 1/\mu - 1/s\f$ and \f$I_{\mu\alpha} = 0\f$ are closed; \f$I_{\alpha\alpha}\f$
   * comes from the generic sum over the support.
   * \return Symmetric tensor \c sym_tensor<2, 2> of the information.
   */
  LAPLACE_HD sym_tensor<2, 2> expected_info() const {
    sym_tensor<2, 2> t = expected_info_by_integral<0, info_form::score>(*this);
    double C[1], Mp[1];
    detail_pig::c_partials<0>(mu, alpha, C, Mp);
    t[0] = C[0];
    t[1] = 0.0;
    return t;
  }
  /**
   * \ingroup distrib_pig
   * \brief Evaluates the \f$K\f$-th derivative of the expected Fisher information.
   *
   * The \f$(\mu, \mu)\f$ block is the derivative tensor of \f$c = 1/\mu - 1/s\f$ and the
   * \f$(\mu, \alpha)\f$ block is zero; the \f$(\alpha, \alpha)\f$ block comes from the generic sum.
   * \tparam K Derivative order (\f$1 \le K \le 4\f$).
   * \return Container \c expected_info_tensor<2, K> of information derivatives.
   */
  template <int K>
  LAPLACE_HD expected_info_tensor<2, K> expected_info_dk() const {
    expected_info_tensor<2, K> t = expected_info_by_integral<K, info_form::score>(*this);
    double C[math::detail_chain_pair::pair_count(K)], Mp[math::detail_chain_pair::pair_count(K)];
    detail_pig::c_partials<K>(mu, alpha, C, Mp);
    for (int b = 0; b <= K; ++b) {
      t.pair[0][b] = C[math::detail_chain_pair::pair_index(K - b, b)];
      t.pair[1][b] = 0.0;
    }
    return t;
  }
  /**
   * \name Derivatives of the expected Fisher information
   * Parameter derivatives of \f$I(\mu, \alpha)\f$ for orders \f$K = 1 \dots 4\f$.
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

  /// \ingroup distrib_pig
  /// \brief Lower bound of the discrete support (0).
  LAPLACE_HD double support_lower() const { return 0.0; }
  /// \ingroup distrib_pig
  /// \brief Upper bound of the discrete support (\f$+\infty\f$).
  LAPLACE_HD double support_upper() const { return distrib_infinity; }
  /// \ingroup distrib_pig
  /// \brief Central location (the mean \f$\mu\f$).
  LAPLACE_HD double center() const { return mu; }
  /// \ingroup distrib_pig
  /// \brief Typical spread \f$\sqrt{\mu + \mu^2/(s - \mu)}\f$.
  LAPLACE_HD double scale() const { return Kokkos::sqrt(mu + mu * mu / excess()); }
  /// \ingroup distrib_pig
  /// \brief Non-smooth points in the parameters (none).
  LAPLACE_HD distrib_kinks param_kinks(int, double) const { return {}; }

  /**
   * \ingroup distrib_pig
   * \brief Starting value from a sample summary: \f$\mu = \bar{y}\f$ (`s.mean`, both the moment and
   * the maximum likelihood estimate) and the moment estimate of \f$\alpha\f$ from the sample
   * variance (`s.variance`).
   *
   * With \f$d = \mu^2/(s^2_y - \mu)\f$ inverted from \f$\mathrm{Var} = \mu + \mu^2/(s - \mu)\f$,
   * the estimate is \f$\alpha = \sqrt{d(d + 2\mu)}\f$. The denominator is floored at \f$\mu/100\f$,
   * so a sample with no overdispersion starts at \f$d = 100\mu\f$, near this family's Poisson limit,
   * instead of at an end of the domain.
   * \param s Sample summary computed by summarize_sample().
   * \return A \c pig_distrib with both parameters strictly inside their open domains.
   * \see distrib_start_value(), distrib_sample_summary, start_inside(), distrib_start
   */
  pig_distrib start_value(const distrib_sample_summary& s) const {
    const double m = start_inside(s.mean, params[0]);
    const double floor_d = 0.01 * m, e = s.variance - m;
    const double d = m * m / (e > floor_d ? e : floor_d);
    return {m, start_inside(std::sqrt(d * (d + 2.0 * m)), params[1])};
  }
};

}  // namespace laplace::distrib
