// The integral of a Gaussian density against the lasso's exponential, on the log scale,
//
//   I(m, Sigma, lam) = int N(beta; m, Sigma) exp(-sum_i lam_i |beta_i|) dbeta,
//
// and the mean and covariance of the law proportional to its integrand.
// On the orthant of signs s (s_i beta_i > 0), with ell = lam o s, the integrand is
// exp(c_s) N(beta; mu_s, Sigma), mu_s = m - Sigma ell, c_s = ell' Sigma ell / 2 - ell' m
// = -ell' (m + mu_s) / 2, so
//
//   log I = log sum_s exp(c_s + log P_s),  P_s = Pr(s o beta > 0), beta ~ N(mu_s, Sigma),
//
// summed as log-sum-exp: in the tail c_s and log P_s are large and of opposite sign,
// and only their sum is formed. P_s and the moments of the truncated law on each
// orthant come from math/mvnormal.hpp (log_mvn_rect, log_mvn_trunc_cumulants); the
// moments of the whole law are their mixture with weights exp(c_s + log P_s) / I,
// accumulated one orthant at a time as convex combinations centred on the running
// mean, never as second moments minus the square of the mean.
//
// The value takes log_mvn_rect's lattice from four coordinates on, as the moments do.
// The cost is 2^k orthant probabilities. All functions are LAPLACE_HD, allocate
// nothing, and take their workspace from the caller.
#pragma once

#include <laplace/math/mvnormal.hpp>

namespace laplace::math {

namespace detail_gaussian_laplace {

// the orthant whose bit i is set where beta_i > 0: writes mu_s and the limits of
// Y = beta - mu_s (Y ~ N(0, Sigma)), returns c_s
LAPLACE_HD double orthant(int k, unsigned s, const double* m, const double* S, const double* lam, double* mu,
                          double* lo, double* hi) {
  using detail_mvnormal::inf;
  using detail_mvnormal::pk;
  double c = 0.0;
  for (int i = 0; i < k; ++i) {
    double v = m[i];
    for (int j = 0; j < k; ++j) v -= S[pk(i, j)] * ((s >> j) & 1u ? lam[j] : -lam[j]);
    mu[i] = v;
    const bool pos = (s >> i) & 1u;
    c -= (pos ? lam[i] : -lam[i]) * (m[i] + v) / 2;
    lo[i] = pos ? -v : -inf();
    hi[i] = pos ? inf() : -v;
  }
  return c;
}

LAPLACE_HD bool bad_arguments(int k, const double* m, const double* lam) {
  if (k < 1 || k > mvn_max_dim) return true;
  for (int i = 0; i < k; ++i)
    if (m[i] != m[i] || !(lam[i] >= 0.0) || lam[i] == detail_mvnormal::inf()) return true;
  return false;
}

// Order 1 writes the mean, 2 the covariance packed lower; returns log I
template <int Order>
LAPLACE_HD double moments(int k, const double* m, const double* S, const double* lam, double* out, double* w,
                          int n_lattice) {
  using namespace detail_mvnormal;
  const int no = Order == 1 ? k : k * (k + 1) / 2;
  if (bad_arguments(k, m, lam)) {
    for (int q = 0; q < no; ++q) out[q] = nan();
    return nan();
  }
  double* mu = w;
  double* lo = w + k;
  double* hi = w + 2 * k;
  double* mean = w + 3 * k;
  int* keep = reinterpret_cast<int*>(w + 4 * k);  // k ints in k doubles
  double* kap = w + 5 * k;
  double* ws = kap + tuples_upto(k, Order);
  for (int q = 0; q < no; ++q) out[q] = 0.0;
  for (int i = 0; i < k; ++i) mean[i] = 0.0;
  double L = -inf();
  for (unsigned s = 0; s < (1u << k); ++s) {
    const double c = orthant(k, s, m, S, lam, mu, lo, hi);
    int mm = 0;
    const double lP = log_mvn_trunc_cumulants(k, Order, lo, hi, S, mm, keep, kap, ws, n_lattice);
    if (lP != lP) {
      for (int q = 0; q < no; ++q) out[q] = nan();
      return nan();
    }
    if (!(lP > -inf())) continue;
    // every coordinate has one finite limit, so mm == k; the weights f of this
    // orthant and g = 1 - f of those before, both from the log scale
    const double lw = c + lP, Ln = logaddexp(L, lw);
    const double f = math::exp(lw - Ln), g = math::exp(L - Ln);
    L = Ln;
    // d (in mu): this orthant's mean minus the running mean
    for (int p = 0; p < mm; ++p) {
      const int i = keep[p];
      mu[i] += kap[1 + p] - mean[i];
    }
    if (Order == 2) {
      for (int p = 0; p < mm; ++p)
        for (int q = 0; q <= p; ++q) {
          const int i = keep[p], j = keep[q];
          const int t[2] = {q, p};
          const double kv = kap[1 + mm + tuple_rank(mm, 2, t)];
          double& o = out[pk(i, j)];
          o = g * o + f * kv + f * g * mu[i] * mu[j];
        }
    }
    for (int i = 0; i < k; ++i) mean[i] += f * mu[i];
  }
  if (Order == 1)
    for (int i = 0; i < k; ++i) out[i] = mean[i];
  return L;
}

}  // namespace detail_gaussian_laplace

/**
 * \ingroup gaussian_laplace
 * \brief Workspace size in doubles for the Gaussian-lasso functions.
 *
 * Returns `3 * k + mvn_rect_scratch(k)` when `order` is `0`, and the workspace size needed by
 * gaussian_laplace_mean() (`order` `1`) or gaussian_laplace_cov() (`order` `2`), comprising
 * \f$5k\f$ doubles, the cumulant array of order `order`, and `mvn_trunc_scratch(k, order)`.
 *
 * \param k Dimension, \f$1 \le k \le\f$ `mvn_max_dim`.
 * \param order `0` for gaussian_laplace_log_integral(), `1` for gaussian_laplace_mean(), or
 *        `2` for gaussian_laplace_cov().
 * \return Number of `double` elements required in `w`.
 *
 * \see gaussian_laplace_log_integral(), gaussian_laplace_mean(), gaussian_laplace_cov()
 */
LAPLACE_HD int gaussian_laplace_scratch(int k, int order) {
  if (order == 0) return 3 * k + mvn_rect_scratch(k);
  return 5 * k + detail_mvnormal::tuples_upto(k, order) + mvn_trunc_scratch(k, order);
}

/**
 * \ingroup gaussian_laplace
 * \brief Log of the Gaussian-lasso integral \f$\log \int N(\beta; m, \Sigma) \exp(-\sum_i \lambda_i |\beta_i|) \, \mathrm{d}\beta\f$.
 *
 * Sums the \f$2^k\f$ orthants on the log scale as \f$\log \sum_s \exp(c_s + \log P_s)\f$, where
 * \f$s \in \{-1, 1\}^k\f$, \f$\ell = \lambda \circ s\f$, \f$\mu_s = m - \Sigma \ell\f$,
 * \f$c_s = -\ell^\top (m + \mu_s) / 2\f$, and \f$P_s = \Pr(s \circ \beta > 0)\f$ for
 * \f$\beta \sim N(\mu_s, \Sigma)\f$, computed by log_mvn_rect() using its lattice rule from four
 * coordinates onward.
 *
 * \param k Dimension, \f$1 \le k \le\f$ `mvn_max_dim` (19); cost is \f$2^k\f$ orthant probabilities.
 * \param m Mean vector of the Gaussian (\f$k\f$ elements).
 * \param S Covariance matrix \f$\Sigma\f$ (\f$k(k+1)/2\f$ elements), positive definite, packed
 *        lower-triangular as in log_mvn_rect().
 * \param lam Non-negative finite rates \f$\lambda_i \ge 0\f$ (\f$k\f$ elements).
 * \param w Workspace of at least `gaussian_laplace_scratch(k, 0)` doubles.
 * \param n_lattice Lattice size passed to log_mvn_rect() for \f$k \ge 4\f$ (default
 *        `mvn_default_lattice`, 16381).
 * \return \f$\log I\f$ (equal to \f$0\f$ when every \f$\lambda_i = 0\f$); quiet NaN for \f$k\f$ out
 *         of range, a NaN entry in `m`, a negative, NaN, or infinite rate in `lam`, or a NaN from
 *         log_mvn_rect().
 *
 * Measured max error across 195 one-factor cases (gcc/OpenMP and CUDA): \f$3.6 \times 10^{-14}\f$
 * (\f$k = 1\f$), \f$6.5 \times 10^{-14}\f$ (\f$k = 2\f$), \f$1.1 \times 10^{-13}\f$ (\f$k = 3\f$),
 * \f$3.3 \times 10^{-6}\f$ (\f$k = 4\f$), and \f$5.8 \times 10^{-6}\f$ (\f$k = 5\f$).
 *
 * \see gaussian_laplace_scratch(), gaussian_laplace_mean(), gaussian_laplace_cov(), log_mvn_rect()
 */
LAPLACE_HD double gaussian_laplace_log_integral(int k, const double* m, const double* S, const double* lam, double* w,
                                                int n_lattice = mvn_default_lattice) {
  using namespace detail_mvnormal;
  if (detail_gaussian_laplace::bad_arguments(k, m, lam)) return nan();
  double* mu = w;
  double* lo = w + k;
  double* hi = w + 2 * k;
  double* ws = w + 3 * k;
  double L = -inf();
  for (unsigned s = 0; s < (1u << k); ++s) {
    const double c = detail_gaussian_laplace::orthant(k, s, m, S, lam, mu, lo, hi);
    // the lattice from four coordinates on (the tensor rule costs 20 times more there)
    const double lP = log_mvn_rect(k, lo, hi, S, ws, n_lattice, 3);
    if (lP != lP) return nan();
    L = logaddexp(L, c + lP);
  }
  return L;
}

/**
 * \ingroup gaussian_laplace
 * \brief Mean of the law proportional to \f$N(\beta; m, \Sigma) \exp(-\sum_i \lambda_i |\beta_i|)\f$, returning \f$\log I\f$.
 *
 * Accumulates the mixture of shifted truncated orthant means from log_mvn_trunc_cumulants() as
 * convex combinations \f$\bar{\beta} \leftarrow \bar{\beta} + f (\mu_s + \kappa_{1,s} - \bar{\beta})\f$,
 * with orthant weight ratio \f$f\f$ computed from log weights.
 *
 * \param k Dimension, as in gaussian_laplace_log_integral().
 * \param m Mean vector of the Gaussian, as in gaussian_laplace_log_integral().
 * \param S Packed lower-triangular covariance matrix, as in gaussian_laplace_log_integral().
 * \param lam Non-negative finite rates, as in gaussian_laplace_log_integral().
 * \param mean Output array receiving \f$k\f$ mean coordinates (filled with quiet NaN on invalid input).
 * \param w Workspace of at least `gaussian_laplace_scratch(k, 1)` doubles.
 * \param n_lattice Lattice size passed to log_mvn_trunc_cumulants() for \f$k \ge 5\f$ (default
 *        `mvn_trunc_default_lattice`, 65521).
 * \return \f$\log I\f$, or quiet NaN under the conditions of gaussian_laplace_log_integral().
 *
 * Measured max error in standard-deviation units (16381-point lattice): \f$1.4 \times 10^{-14}\f$
 * (\f$k = 1\f$), \f$2.1 \times 10^{-13}\f$ (\f$k = 2\f$), \f$1.4 \times 10^{-12}\f$ (\f$k = 3\f$),
 * \f$9.5 \times 10^{-5}\f$ (\f$k = 4\f$), and \f$7.9 \times 10^{-5}\f$ (\f$k = 5\f$).
 *
 * \see gaussian_laplace_log_integral(), gaussian_laplace_cov(), gaussian_laplace_scratch(), log_mvn_trunc_cumulants()
 */
LAPLACE_HD double gaussian_laplace_mean(int k, const double* m, const double* S, const double* lam, double* mean,
                                        double* w, int n_lattice = mvn_trunc_default_lattice) {
  return detail_gaussian_laplace::moments<1>(k, m, S, lam, mean, w, n_lattice);
}

/**
 * \ingroup gaussian_laplace
 * \brief Covariance of the law proportional to \f$N(\beta; m, \Sigma) \exp(-\sum_i \lambda_i |\beta_i|)\f$, packed lower-triangular, returning \f$\log I\f$.
 *
 * Accumulates the mixture covariance one orthant at a time from log_mvn_trunc_cumulants() via the
 * centred convex update \f$C \leftarrow (1 - f) C + f C_s + f(1 - f) d d^\top\f$, where \f$f\f$
 * is the new orthant's relative weight, \f$C_s\f$ is its truncated covariance, and \f$d\f$ is its
 * mean minus the running mean.
 *
 * \param k Dimension, as in gaussian_laplace_log_integral().
 * \param m Mean vector of the Gaussian, as in gaussian_laplace_log_integral().
 * \param S Packed lower-triangular covariance matrix, as in gaussian_laplace_log_integral().
 * \param lam Non-negative finite rates, as in gaussian_laplace_log_integral().
 * \param cov Output array receiving \f$k(k+1)/2\f$ packed lower-triangular entries (filled with
 *        quiet NaN on invalid input).
 * \param w Workspace of at least `gaussian_laplace_scratch(k, 2)` doubles.
 * \param n_lattice Lattice size passed to log_mvn_trunc_cumulants() for \f$k \ge 5\f$ (default
 *        `mvn_trunc_default_lattice`, 65521).
 * \return \f$\log I\f$, or quiet NaN under the conditions of gaussian_laplace_log_integral().
 *
 * Measured max error in standard-deviation-product units (16381-point lattice): \f$5.0 \times 10^{-14}\f$
 * (\f$k = 1\f$), \f$1.4 \times 10^{-13}\f$ (\f$k = 2\f$), \f$4.5 \times 10^{-12}\f$ (\f$k = 3\f$),
 * \f$1.8 \times 10^{-3}\f$ (\f$k = 4\f$), and \f$1.2 \times 10^{-3}\f$ (\f$k = 5\f$).
 *
 * \see gaussian_laplace_log_integral(), gaussian_laplace_mean(), gaussian_laplace_scratch(), log_mvn_trunc_cumulants()
 */
LAPLACE_HD double gaussian_laplace_cov(int k, const double* m, const double* S, const double* lam, double* cov,
                                       double* w, int n_lattice = mvn_trunc_default_lattice) {
  return detail_gaussian_laplace::moments<2>(k, m, S, lam, cov, w, n_lattice);
}

}  // namespace laplace::math
