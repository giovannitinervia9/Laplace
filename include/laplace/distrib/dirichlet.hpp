// dirichlet_distrib (laplace::distrib): the Dirichlet law on the open simplex of d
// categories in its mean mu (a simplex_vector carried by its additive log-ratio
// coordinates eta, reference category the last; bijector/alr.hpp) and its
// precision phi > 0; design and measurements in note/distrib_multivariate.md
// section 18.
//
//   alpha_j = phi mu_j,  E[Y] = mu,  Cov Y = (diag mu - mu mu')/(phi + 1),
//   l(eta, phi; y) = lgamma(phi) - sum_j lgamma(alpha_j) + sum_j (alpha_j - 1) log y_j.
//
// y holds the d coordinates as given (y_d is not recomputed from the others) and
// must satisfy y_j > 0 and |sum_j y_j - 1| <= 1e-8; elsewhere logpdf is -inf.
//
// Routes. The data enter linearly in alpha. -sum_j lgamma(alpha_j) is split per
// coordinate, so that nothing large is formed and then cancelled:
//   alpha_j < 1:   -lgamma(a) = log a - lgamma(1 + a); sum log alpha_j over these
//                  coordinates has closed derivatives in eta (detail_simplex: L,
//                  the cumulants), and -lgamma(1 + a) goes through Faa di Bruno
//                  with psi_n(1 + a), regular at a tiny shape;
//   alpha_j >= 1:  -lgamma(a) = -(a log a - a) + 1/2 log a - 1/2 log 2pi - delta(a),
//                  delta Stirling's error (math/stirling.hpp) through Faa di Bruno;
//                  -(a log a - a) and 1/2 log a have closed derivatives in eta and
//                  phi, and their parts linear in mu_j are summed over the small set
//                  with the sign changed (sum_j d mu_j = 0 exactly).
// lgamma(phi) is written as (phi - 1/2) log phi - phi + 1/2 log 2pi + delta(phi) in
// the value, so that phi log phi cancels algebraically; the pure phi derivatives
// are formed from the polygamma remainders r_n = psi_n - mu_n
// (math::polygamma_monomial_remainder), whose leading monomials cancel exactly.
//
// Derivatives: logpdf_dk<K, A> with K in phi (theta index 0) and A in eta, the
// packed symmetric tensor of order A over the d - 1 coordinates of eta (by rows
// at order 2, lexicographic at order >= 3); K + A <= 6.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/detail/simplex_common.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/special.hpp>
#include <laplace/math/stirling.hpp>

#include <cmath>
#include <cstdint>

namespace laplace::distrib {

namespace detail_dirichlet {

inline constexpr double half_log_two_pi = 0.91893853320467274178;

// delta^(n)(a), n = 0..6, run-time order
LAPLACE_HD double stirling_dn(int n, double a) {
  switch (n) {
    case 0: return math::stirling_error(a);
    case 1: return math::stirling_error_d1(a);
    case 2: return math::stirling_error_d2(a);
    case 3: return math::stirling_error_d3(a);
    case 4: return math::stirling_error_d4(a);
    case 5: return math::stirling_error_d5(a);
    default: return math::stirling_error_d6(a);
  }
}

LAPLACE_HD_FORCEINLINE double binom(int n, int k) {
  double r = 1.0;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

LAPLACE_HD_FORCEINLINE double falling(int c, int i) {  // c!/(c - i)!
  double r = 1.0;
  for (int q = 0; q < i; ++q) r *= c - q;
  return r;
}

LAPLACE_HD_FORCEINLINE double factorial(int n) {
  double r = 1.0;
  for (int q = 2; q <= n; ++q) r *= q;
  return r;
}

}  // namespace detail_dirichlet

/**
 * \ingroup distrib_dirichlet
 * \brief Precomputed state for `dirichlet_distrib` at fixed precision \f$\phi\f$.
 *
 * \see dirichlet_distrib, distrib_dirichlet
 */
struct dirichlet_state {
  /// Precision parameter \f$\phi > 0\f$.
  double phi;
  /// Natural logarithm \f$\log\phi\f$.
  double log_phi;
};

/**
 * \ingroup distrib_dirichlet
 * \brief Dirichlet distribution on the open \f$d\f$-simplex parametrized by mean \f$\mu\f$ and precision \f$\phi\f$.
 *
 * Models \f$Y \sim \operatorname{Dirichlet}(\phi\mu_1, \dots, \phi\mu_d)\f$, where
 * \f$\mu\f$ is a `simplex_vector` carried by its \f$d - 1\f$ additive log-ratio coordinates
 * \f$\eta_a = \log(\mu_a / \mu_d)\f$ (last category as reference) and \f$\phi > 0\f$ is
 * an indexed scalar parameter at index `0` of \f$\theta\f$. Satisfies `multivariate_distrib`.
 *
 * \see dirichlet_state, distrib_dirichlet, distrib_multivariate
 */
struct dirichlet_distrib {
  /// \ingroup distrib_dirichlet
  /// \brief Number of categories \f$d \ge 2\f$.
  int categories = 2;  // d >= 2

  /// \ingroup distrib_dirichlet
  /// \brief Normalization regime (`distrib_normalization::proper`).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// \ingroup distrib_dirichlet
  /// \brief Distribution kind (`distrib_kind::continuous`: a density with derivatives in \f$y\f$).
  static constexpr distrib_kind kind = distrib_kind::continuous;
  /// \ingroup distrib_dirichlet
  /// \brief Number of declared parameters: \f$\mu\f$ (`simplex_vector`) and \f$\phi\f$ (`scalar`).
  static constexpr int n_params = 2;
  /// \ingroup distrib_dirichlet
  /// \brief Descriptors of `mu` (dense, \f$d - 1\f$ coordinates in `x`) and `phi` (indexed scalar on \f$(0, \infty)\f$).
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::simplex_vector, 0.0, 1.0, infinitely_smooth},
      {"phi", distrib_mv_param_kind::scalar, 0.0, distrib_infinity, infinitely_smooth}};
  /// \ingroup distrib_dirichlet
  /// \brief Precomputed state type returned by `theta_state()`.
  using state_type = dirichlet_state;

  // the per-unit scratch: mu, log mu, alpha, log y (d each), F^(n)(alpha_j) for
  // n = 0..6 (7 d), the partition sums (64 x 7 per category of a tuple and one
  // more for the outside categories)
  static constexpr int fn_width = 7;
  static constexpr int psum_size = detail_simplex::max_masks * (detail_simplex::max_order + 1);

  /// \ingroup distrib_dirichlet
  /// \brief Default constructor (\f$d = 2\f$ categories).
  LAPLACE_HD dirichlet_distrib() = default;
  /**
   * \ingroup distrib_dirichlet
   * \brief Construct a Dirichlet distribution with `d` categories.
   * \param d Number of categories (\f$d \ge 2\f$).
   */
  LAPLACE_HD explicit dirichlet_distrib(int d) : categories(d) {}

  /**
   * \ingroup distrib_dirichlet
   * \brief Return the observation dimension \f$d\f$ (`categories`).
   * \return Number of categories \f$d\f$.
   */
  LAPLACE_HD int dim() const { return categories; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the number of open reference coordinates for derivatives in \f$y\f$.
   * \return `categories - 1` (since \f$y_d = 1 - \sum_{a=1}^{d-1} y_a\f$).
   */
  LAPLACE_HD int y_coord_size() const { return categories - 1; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the number of coordinates of parameter `j`.
   * \param j Parameter index in `params` (`0` for `mu`, `1` for `phi`).
   * \return `categories - 1` for `mu`, and `1` for `phi`.
   */
  LAPLACE_HD int param_size(int j) const { return j == 0 ? categories - 1 : 1; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the starting index of parameter `j` in its array (`x` or \f$\theta\f$).
   * \return `0` (both `mu` in `x` and `phi` in \f$\theta\f$ start at index `0`).
   */
  LAPLACE_HD int param_offset(int) const { return 0; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the length of the indexed parameter array \f$\theta\f$.
   * \return `1` (holds \f$\phi\f$).
   */
  LAPLACE_HD int theta_size() const { return 1; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the number of `double` elements stored in caller memory by `theta_state()`.
   * \return `0` (`dirichlet_state` is held by value).
   */
  LAPLACE_HD int state_size() const { return 0; }
  /**
   * \ingroup distrib_dirichlet
   * \brief Return the per-unit working scratch size in `double` elements.
   * \return Number of `double` elements required in `w` by derivative and information calls.
   */
  LAPLACE_HD int scratch_size() const {
    return (9 + fn_width) * categories + 2 * psum_size + detail_simplex::subset_tables::scratch_doubles;
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Precompute the state holding \f$\phi\f$ and \f$\log\phi\f$.
   *
   * The second workspace argument is unused because `state_size()` is `0`.
   *
   * \param theta Parameter array of length `theta_size()` with \f$\phi = \theta_0 > 0\f$.
   * \return Populated `dirichlet_state`.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double*) const { return {theta[0], math::log(theta[0])}; }

  /**
   * \ingroup distrib_dirichlet
   * \brief Check whether \f$y\f$ lies on the open \f$d\f$-simplex within tolerance.
   *
   * Returns `true` when every \f$y_j > 0\f$ and \f$\lvert \sum_{j=0}^{d-1} y_j - 1 \rvert \le 10^{-8}\f$.
   *
   * \param y Observation vector of length `dim()`.
   * \return `true` if `y` is in the support, `false` otherwise.
   */
  LAPLACE_HD bool in_support(const double* y) const {
    double s = 0.0;
    for (int j = 0; j < categories; ++j) {
      if (!(y[j] > 0.0)) return false;
      s += y[j];
    }
    return Kokkos::fabs(s - 1.0) <= 1e-8;
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Compute category probabilities \f$\mu_j\f$ and log-probabilities \f$\log\mu_j\f$ from additive log-ratio coordinates `x`.
   *
   * Factors out the category with the largest \f$\eta_c\f$ (including \f$\eta_d = 0\f$) so a probability near `1` keeps its digits.
   *
   * \param x Additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param mu Output probability vector of length `dim()`.
   * \param lmu Output log-probability vector of length `dim()`.
   */
  // mu and log mu from eta (detail_simplex::log_probabilities)
  LAPLACE_HD void probabilities(const double* x, double* mu, double* lmu) const {
    detail_simplex::log_probabilities(x, categories, lmu);
    for (int j = 0; j < categories; ++j) mu[j] = math::exp(lmu[j]);
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Evaluate the Dirichlet log-density \f$\ell(\eta, \phi; y)\f$.
   *
   * Evaluates \f$\log\Gamma(\phi) - \sum_{j=1}^d \log\Gamma(\alpha_j) + \sum_{j=1}^d (\alpha_j - 1)\log y_j\f$
   * with \f$\alpha_j = \phi\mu_j\f$, splitting categories at \f$\alpha_j = 1\f$ and combining data and Stirling leading terms
   * as \f$\log(y_j/\mu_j)\f$ via double-double probabilities. Returns \f$-\infty\f$ when `in_support(y)` is `false`.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \return Log-density value, or \f$-\infty\f$ off the support.
   * \see in_support(), logpdf_dk(), logpdf_dy_dk(), distrib_dirichlet
   */
  LAPLACE_HD double logpdf(const state_type& st, const double* y, const double* x) const {
    if (!in_support(y)) return -distrib_infinity;
    namespace dd = detail_dirichlet;
    const double phi = st.phi, lphi = st.log_phi;
    // lgamma(phi) = (phi - 1/2) log phi - phi + 1/2 log 2pi + delta(phi)
    double v = -0.5 * lphi + dd::half_log_two_pi + math::stirling_error(phi);
    double small_mass = 0.0, kl = 0.0, sly = 0.0;
    // the probabilities in double-double for log(y_j/mu_j): the normalizer once
    namespace dm = math::detail::ddm;
    double xm = 0.0;
    for (int a = 0; a + 1 < categories; ++a) xm = x[a] > xm ? x[a] : xm;
    dm::dd norm(0.0);
    for (int b = 0; b < categories; ++b) norm = norm + dm::exp(dm::two_sum(b + 1 < categories ? x[b] : 0.0, -xm));
    // log mu_j as in detail_simplex::log_probabilities, without the array
    int cm = categories - 1;
    for (int q = 0; q + 1 < categories; ++q)
      if (x[q] >= xm && x[q] > 0.0) cm = q;
    const dm::dd others = norm - 1.0;  // the largest term is exactly 1
    const double l1 = Kokkos::log1p(others.hi + others.lo);
    for (int j = 0; j < categories; ++j) {
      const double lm_j = j == cm ? -l1 : ((j + 1 < categories ? x[j] : 0.0) - xm) - l1, mu_j = math::exp(lm_j);
      const double a = phi * mu_j, ly = math::log(y[j]);
      sly += ly;
      if (a < 1.0) {
        small_mass += mu_j;
        v += lphi + lm_j - math::lgamma(1.0 + a) + a * ly;
      } else {
        v += 0.5 * (lphi + lm_j) - dd::half_log_two_pi - math::stirling_error(a);
        const dm::dd mud = dm::exp(dm::two_sum(j + 1 < categories ? x[j] : 0.0, -xm)) / norm;
        kl += mu_j * detail_simplex::log_ratio_dd(mud, y[j]);
      }
    }
    // phi log phi - phi times the small mass, phi sum_large mu log(y/mu)
    return v + phi * ((lphi - 1.0) * small_mass + kl) - sly;
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Evaluate partial derivatives of the log-density of order `K` in \f$\phi\f$ and `A` in \f$\eta\f$ (\f$K + A \le 6\f$).
   *
   * Writes the packed symmetric tensor of order `A` over the \f$d - 1\f$ coordinates of \f$\eta\f$
   * (scalar for `A = 0`, vector for `A = 1`, lower-triangular packed by rows for `A = 2`,
   * and `sym_tensor` lexicographic order for `A >= 3`). The `idx` argument is unused because
   * \f$\phi\f$ is the single indexed parameter at index `0`. Matches 60-digit `mpmath` references
   * within `44.8` ulp1N on gcc and `33.2` ulp1N on CUDA.
   *
   * \tparam K Derivative order in \f$\phi\f$ (`K >= 0`).
   * \tparam A Derivative order in \f$\eta\f$ (at most one order; omitted means `0`, `K + A <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output buffer of length `sym_tensor_size(categories - 1, A)` doubles.
   * \param w Caller working scratch of at least `scratch_size()` doubles.
   * \return Always `true`.
   * \see logpdf(), logpdf_dy_dk(), distrib_dirichlet
   */
  template <int K, int... A>
  LAPLACE_HD bool logpdf_dk(const state_type& st, const double* y, const double* x, const int*, double* out,
                            double* w) const {
    static_assert(sizeof...(A) <= 1, "dirichlet_distrib has one dense parameter");
    constexpr int An = (0 + ... + A);
    static_assert(K + An <= 6, "orders up to six");
    if constexpr (K == 0 && An == 0) {
      out[0] = logpdf(st, y, x);
      return true;
    } else {
      const int d = categories;
      double* mu = w;
      double* lm = mu + d;
      double* al = lm + d;
      double* ly = al + d;
      double* lr = ly + d;  // log(y_j/mu_j), detail_simplex::log_ratio
      double* pt = lr + d;  // the unit's probabilities in double-double (4 d)
      double* F = pt + detail_simplex::point_size(d);  // F[j * 7 + n] = F^(n)(alpha_j)
      double* P = F + fn_width * d;
      double* Q = P + psum_size;
      probabilities(x, mu, lm);
      detail_simplex::fill_point(x, d, pt);
      const double phi = st.phi;
      for (int j = 0; j < d; ++j) {
        al[j] = phi * mu[j];
        ly[j] = math::log(y[j]);
        lr[j] = detail_simplex::log_ratio(pt, j, y[j]);
        fill_outer(al[j], K + An, F + fn_width * j);
      }
      if constexpr (An == 0) {
        out[0] = pure_phi<K>(st, mu, lm, al, lr);
      } else {
        detail_simplex::tuple_walk e(An, d - 1);
        e.first();
        int p = 0;
        do {
          out[p++] = eta_block<K, An>(st, e.t, pt, mu, lm, al, ly, lr, F, P, Q, Q + psum_size);
        } while (e.next());
      }
      return true;
    }
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Evaluate partial derivatives of order `J` in \f$y\f$, `K` in \f$\phi\f$, and `A` in \f$\eta\f$ (\f$J + K + A \le 6\f$).
   *
   * Differentiates in the \f$d - 1\f$ open reference coordinates \f$y_1, \dots, y_{d-1}\f$ with
   * \f$y_d = 1 - \sum_{a=1}^{d-1} y_a\f$, writing the Kronecker product (with \f$y\f$ first) of the
   * order-`J` tensor and the order-`A` tensor over `categories - 1` coordinates. Delegates to `logpdf_dk()`
   * when `J == 0`, and returns `false` leaving `out` untouched when `J >= 1` and `K >= 2`. Matches
   * 60-digit `mpmath` references within `14.3` ulp1N on gcc and `4.82` ulp1N on CUDA.
   *
   * \tparam J Derivative order in \f$y\f$ (`J >= 0`).
   * \tparam K Derivative order in \f$\phi\f$ (`K >= 0`).
   * \tparam A Derivative order in \f$\eta\f$ (at most one order; omitted means `0`, `J + K + A <= 6`).
   * \param st Precomputed state from `theta_state()`.
   * \param y Observation vector of length `dim()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param idx Sorted parameter index tuple for \f$\phi\f$ (unused).
   * \param out Output buffer of length `sym_tensor_size(y_coord_size(), J) * sym_tensor_size(categories - 1, A)` doubles.
   * \param w Caller working scratch of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when `J >= 1` and `K >= 2`.
   * \see logpdf_dk(), y_coord_size(), distrib_dirichlet
   */
  // J derivatives in y along the reference coordinates y_1..y_(d-1) (y_d = 1 - sum, y as given at
  // the point), K in phi and A in eta. y enters as sum_j (alpha_j - 1) log y_j, so with
  // h_J(v) = (-1)^(J-1) (J-1)!/v^J the entry at the y tuple s and the eta tuple t is
  //   [s constant = a] d_t^K(alpha_a - 1) h_J(y_a) - d_t^K(alpha_d - 1) (J-1)!/y_d^J,
  // d_t^K alpha_j being phi d_t mu_j (K = 0), d_t mu_j (K = 1) and 0 (K >= 2) for A >= 1, and
  // alpha_j - 1 (K = 0), mu_j (K = 1), 0 (K >= 2) for A = 0. out: the y tensor of order J
  // over d - 1, Kronecker with the eta tensor of order A (y first).
  template <int J, int K, int... A>
  LAPLACE_HD bool logpdf_dy_dk(const state_type& st, const double* y, const double* x, const int* idx, double* out,
                               double* w) const {
    static_assert(sizeof...(A) <= 1, "dirichlet_distrib has one dense parameter");
    constexpr int An = (0 + ... + A);
    static_assert(J + K + An <= 6, "orders up to six");
    if constexpr (J == 0) {
      return logpdf_dk<K, A...>(st, y, x, idx, out, w);
    } else if constexpr (K >= 2) {
      return false;
    } else {
      const int d = categories, m = d - 1;
      double* mu = w;
      double* lm = mu + d;
      probabilities(x, mu, lm);
      const double fj = detail_dirichlet::factorial(J - 1);
      const double sgn = J % 2 == 1 ? 1.0 : -1.0;
      // the coefficient of each category, for one eta tuple: dc[j] for j < d - 1, and the last
      double* dc = lm + d;
      detail_simplex::fill_point(x, d, dc + d);
      const int ne = sym_tensor_size(m, An);
      detail_simplex::tuple_walk e(An, m);
      e.first();
      int pe = 0;
      do {
        if constexpr (An == 0) {
          for (int j = 0; j < d; ++j) dc[j] = K == 0 ? st.phi * mu[j] - 1.0 : mu[j];
        } else {
          detail_simplex::subset_tables tb;
          tb.fill(dc + d, d, e.t, An, dc + d + detail_simplex::point_size(d));
          constexpr int full = (1 << An) - 1;
          for (int j = 0; j < d; ++j) {
            int c = -1;
            for (int q = 0; q < tb.nc; ++q)
              if (tb.cats[q] == j) c = q;
            const double dm = c < 0 ? mu[j] * tb.G[full] : tb.D[c][full];
            dc[j] = K == 0 ? st.phi * dm : dm;
          }
        }
        double last = -dc[d - 1] * fj;
        for (int q = 0; q < J; ++q) last /= y[d - 1];
        detail_simplex::tuple_walk s(J, m);
        s.first();
        int ps = 0;
        do {
          double v = last;
          if constexpr (K == 0 && An == 0) {
            // the pure y derivatives in double-double from the unit's point: at odd J the two
            // terms (alpha_a - 1)/y_a^J and (alpha_d - 1)/y_d^J are close near the mode
            namespace dm = math::detail::ddm;
            const double* pt = dc + d;
            auto term = [&](int j) {
              dm::dd t = dm::dd(pt[2 * j], pt[2 * j + 1]) * st.phi - 1.0;
              for (int q = 0; q < J; ++q) t = t / y[j];
              return t * fj;
            };
            dm::dd r = -term(d - 1);
            if (s.t[0] == s.t[J - 1]) r = r + term(s.t[0]) * sgn;
            v = r.hi + r.lo;
          } else if (s.t[0] == s.t[J - 1]) {
            double h = sgn * fj;
            for (int q = 0; q < J; ++q) h /= y[s.t[0]];
            v += dc[s.t[0]] * h;
          }
          out[ps * ne + pe] = v;
          ++ps;
        } while (s.next());
        ++pe;
      } while (e.next());
      return true;
    }
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Evaluate an expected Fisher information block or its derivative of order `K` in \f$\phi\f$ and `A` in \f$\eta\f$ (\f$K + A \le 4\f$).
   *
   * Slot `0` selects \f$\eta\f$ (`d - 1` coordinates) and `theta_slot` selects \f$\phi\f$ (`1` coordinate).
   * Writes the information block (packed `(d - 1) x (d - 1)` for `(0, 0)`, `(d - 1)`-vector for `(0, theta_slot)`
   * or `(theta_slot, 0)`, scalar for `(theta_slot, theta_slot)`) followed in Kronecker product by the order-`A`
   * symmetric tensor over \f$\eta\f$. Arguments `ab` and `idx` are unused because \f$\phi\f$ has one coordinate.
   * Matches 60-digit `mpmath` references within `4.83` ulp1N on gcc and `11.1` ulp1N on CUDA.
   *
   * \tparam K Derivative order in \f$\phi\f$ (`K >= 0`).
   * \tparam A Derivative order in \f$\eta\f$ (at most one order; omitted means `0`, `K + A <= 4`).
   * \param st Precomputed state from `theta_state()`.
   * \param slot_a Row parameter slot (`0` or `theta_slot`).
   * \param slot_b Column parameter slot (`0` or `theta_slot`).
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output buffer receiving the information block and its order-`A` \f$\eta\f$-tensor.
   * \param w Caller working scratch of at least `scratch_size()` doubles.
   * \return Always `true`.
   * \see distrib_dirichlet, distrib_multivariate
   */
  // The expected information (planned core signature, note/distrib_multivariate.md 18.4.2), with
  // E log y_j = psi(alpha_j) - psi(phi) and d_a mu_j = mu_j (delta_ja - mu_a):
  //   (eta, eta)  phi^2 sum_j psi_1(alpha_j) d_a mu_j d_b mu_j, packed over d - 1;
  //   (eta, phi)  sum_j d_a mu_j alpha_j r_1(alpha_j), r_1 = psi_1 - 1/x (the 1 of alpha psi_1
  //               drops exactly, sum_j d_a mu_j = 0), a vector over d - 1;
  //   (phi, phi)  sum_j mu_j^2 r_1(alpha_j) - r_1(phi) (the monomials cancel exactly).
  // slot 0 is eta, theta_slot phi. Derivatives: K in phi, A in eta, the information entry first
  // (Kronecker with the eta tensor). Each entry is split per coordinate as the blocks are:
  //   alpha_j >= 1:  psi_1(a) = 1/a + 1/(2 a^2) + delta2(a)  (delta2 the second derivative of
  //                  Stirling's error);  alpha_j < 1:  psi_1(a) = 1/a^2 + psi_1(1 + a);
  // the 1/a and 1/a^2 parts are closed in d log mu (X_a = d_a log mu_j) and phi:
  //   (eta, eta)  large: phi mu X_a X_b + 1/2 X_a X_b,  small: X_a X_b;
  //   (eta, phi)  large: X_a/(2 phi),                    small: X_a/phi - d_a mu;
  //   (phi, phi)  large: 1/(2 phi^2),  small: 1/phi^2 - mu/phi,  and -1/(2 phi^2) - delta2(phi);
  // the rest, U_j phi^p f(phi mu_j) with f = delta2 or psi_1(1 + .), U_j = d_a mu d_b mu (p = 2),
  // d_a mu mu (p = 1), mu^2 (p = 0), by Leibniz over the tuple and Faa di Bruno over mu_j.
  template <int K, int... A>
  LAPLACE_HD bool expected_info_dk(const state_type& st, int slot_a, int slot_b, const int*, const double* x,
                                   const int*, double* out, double* w) const {
    static_assert(sizeof...(A) <= 1, "dirichlet_distrib has one dense parameter");
    constexpr int An = (0 + ... + A);
    static_assert(K + An <= 4, "information derivatives up to order four");
    const int d = categories, m = d - 1;
    double* mu = w;
    double* lm = mu + d;
    probabilities(x, mu, lm);
    double* pt = lm + d;
    detail_simplex::fill_point(x, d, pt);
    double* ws = pt + detail_simplex::point_size(d);
    const int kind = slot_a == 0 && slot_b == 0 ? 1 : slot_a == 0 || slot_b == 0 ? 2 : 3;
    const int ne = sym_tensor_size(m, An);
    detail_simplex::tuple_walk e(An, m);
    if (kind == 1) {
      detail_simplex::tuple_walk pab(2, m);
      pab.first();
      int pa = 0;
      do {
        e.first();
        int pe = 0;
        do {
          out[pa * ne + pe] = info_entry<K, An>(st, 1, pab.t[1], pab.t[0], e.t, pt, mu, lm, ws);
          ++pe;
        } while (e.next());
        ++pa;
      } while (pab.next());
    } else if (kind == 2) {
      for (int a = 0; a < m; ++a) {
        e.first();
        int pe = 0;
        do {
          out[a * ne + pe] = info_entry<K, An>(st, 2, a, -1, e.t, pt, mu, lm, ws);
          ++pe;
        } while (e.next());
      }
    } else {
      e.first();
      int pe = 0;
      do {
        out[pe] = info_entry<K, An>(st, 3, -1, -1, e.t, pt, mu, lm, ws);
        ++pe;
      } while (e.next());
    }
    return true;
  }

  // one information entry (kind 1: (a, b); 2: (a, phi); 3: (phi, phi)) differentiated An times
  // in eta along t and K times in phi
  template <int K, int An>
  LAPLACE_HD double info_entry(const state_type& st, int kind, int a, int b, const int* t, const double* pt,
                               const double* mu, const double* lm, double* ws) const {
    namespace dd = detail_dirichlet;
    namespace ds = detail_simplex;
    constexpr int W = ds::max_order + 1;
    const int d = categories;
    const double phi = st.phi;
    // the positions: the fixed indices of the entry, then t; sorted for the tables, with masks
    int pos[6], np = 0;
    if (kind <= 2) pos[np++] = a;
    if (kind == 1) pos[np++] = b;
    for (int q = 0; q < An; ++q) pos[np++] = t[q];
    int order[6];
    for (int q = 0; q < np; ++q) order[q] = q;
    for (int i = 1; i < np; ++i)  // positions sorted by index, stable
      for (int k = i; k > 0 && pos[order[k]] < pos[order[k - 1]]; --k) {
        const int s = order[k];
        order[k] = order[k - 1];
        order[k - 1] = s;
      }
    int sorted[6], where[6];
    for (int q = 0; q < np; ++q) {
      sorted[q] = pos[order[q]];
      where[order[q]] = q;
    }
    ds::subset_tables tb;
    tb.fill(pt, d, sorted, np, ws);
    const int ma = kind <= 2 ? 1 << where[0] : 0, mb = kind == 1 ? 1 << where[1] : 0;
    const int full = (1 << np) - 1, tm = full ^ ma ^ mb;
    auto pinv = [&](int n) {
      double r = 1.0;
      for (int q = 0; q < n; ++q) r /= phi;
      return r;
    };
    const double sgnK = K % 2 == 0 ? 1.0 : -1.0;
    const int p = kind == 1 ? 2 : kind == 2 ? 1 : 0;
    double* P = ws + ds::subset_tables::scratch_doubles;  // the partition sums, after the tables
    double f[ds::max_order + 1];  // f^(n)(alpha_j), n = 0..K + An
    double total = 0.0;
    // with every shape below one, sum_j d mu_j/phi over all j is exactly zero (An >= 1)
    bool any_large = false;
    for (int j = 0; j < d; ++j) any_large = any_large || phi * mu[j] >= 1.0;
    double small_mxx = 0.0;
    for (int j = 0; j < d; ++j) {
      int c = -1;
      for (int q = 0; q < tb.nc; ++q)
        if (tb.cats[q] == j) c = q;
      auto dm = [&](int X) { return c < 0 ? mu[j] * tb.G[X] : tb.D[c][X]; };
      auto dlog = [&](int X) { return X == 0 ? lm[j] : tb.dlog(X, c); };
      const double al = phi * mu[j];
      const bool small = al < 1.0;
      for (int n = 0; n <= K + An; ++n) f[n] = small ? math::polygamma(1 + n, 1.0 + al) : dd::stirling_dn(2 + n, al);
      if (c < 0) ds::partition_sums(tb.G, np, P);
      else ds::partition_sums(tb.D[c], np, P);
      const double mpow = c < 0 ? mu[j] : 1.0;  // outside categories: P holds the sums of G, times mu^k
      // d_V d_phi^K [phi^p f(phi mu)], V a subset of tm
      auto dh = [&](int V) {
        double s = 0.0;
        for (int i = 0; i <= K; ++i) {
          const int r = K - i;  // derivatives on phi^p
          if (r > p) continue;
          double pp = dd::falling(p, r);
          for (int q = 0; q < p - r; ++q) pp *= phi;
          double gv = 0.0;  // d_V [mu^i f^(i)(phi mu)]
          if (V == 0) {
            double mi = 1.0;
            for (int q = 0; q < i; ++q) mi *= mu[j];
            gv = mi * f[i];
          } else {
            double mk = 1.0;
            for (int k = 1; k <= np; ++k) {
              mk *= mpow;
              const double Pk = P[V * W + k];
              if (Pk == 0.0) continue;
              double gk = 0.0;  // d^k/dmu^k [mu^i f^(i)(phi mu)]
              for (int rr = 0; rr <= (k < i ? k : i); ++rr) {
                double mm = 1.0, ph = 1.0;
                for (int q = 0; q < i - rr; ++q) mm *= mu[j];
                for (int q = 0; q < k - rr; ++q) ph *= phi;
                gk += dd::binom(k, rr) * dd::falling(i, rr) * mm * ph * f[i + k - rr];
              }
              gv += gk * mk * Pk;
            }
          }
          s += dd::binom(K, i) * pp * gv;
        }
        return s;
      };
      // d_X U_j, X a subset of tm
      auto dU = [&](int X) {
        double s = 0.0;
        for (int V = X;; V = (V - 1) & X) {
          if (kind == 1) s += dm(ma | V) * dm(mb | (X ^ V));
          else if (kind == 2) s += dm(ma | V) * dm(X ^ V);
          else s += dm(V) * dm(X ^ V);
          if (V == 0) break;
        }
        return s;
      };
      for (int V = tm;; V = (V - 1) & tm) {  // the regular part, Leibniz over the subsets of tm
        total += dU(tm ^ V) * dh(V);
        if (V == 0) break;
      }
      // the closed part
      if (kind == 1) {
        double xx = 0.0, mxx = 0.0;  // d_tm (X_a X_b), d_tm (mu X_a X_b)
        for (int V = tm;; V = (V - 1) & tm) {
          xx += dlog(ma | V) * dlog(mb | (tm ^ V));
          if (V == 0) break;
        }
        for (int U = tm;; U = (U - 1) & tm) {
          const int R = tm ^ U;
          double s = 0.0;
          for (int V = R;; V = (V - 1) & R) {
            s += dlog(ma | V) * dlog(mb | (R ^ V));
            if (V == 0) break;
          }
          mxx += dm(U) * s;
          if (U == 0) break;
        }
        // phi mu X_a X_b of the large set: sum_j mu_j X_a X_b = kappa_ab over all j, so the
        // large set's sum is kappa_(ab tm) = -L minus the small set's (added after the loop)
        if (small) {
          total += K == 0 ? xx : 0.0;
          small_mxx += mxx;
        } else {
          total += K == 0 ? 0.5 * xx : 0.0;
        }
      } else if (kind == 2) {
        const double ck = sgnK * dd::factorial(K) * pinv(K + 1);  // d_phi^K (1/phi)
        total += (small ? 1.0 : 0.5) * ck * dlog(ma | tm);
        if (small && K == 0) total -= dm(ma | tm);
      } else {
        if (An == 0) total += (small ? 1.0 : 0.5) * sgnK * dd::factorial(K + 1) * pinv(K + 2);
        if (small && (any_large || An == 0)) total -= sgnK * dd::factorial(K) * pinv(K + 1) * dm(tm);
      }
    }
    if (kind == 1 && any_large && K <= 1) {
      const double large_mxx = -tb.L[full] - small_mxx;
      total += K == 0 ? phi * large_mxx : large_mxx;
    }
    if (kind == 3 && An == 0) total += -0.5 * sgnK * dd::factorial(K + 1) * pinv(K + 2) - dd::stirling_dn(2 + K, phi);
    return total;
  }

  // F^(n)(alpha) for n = 1..nmax: -psi_(n-1)(1 + alpha) below one, -delta^(n)(alpha) above
  LAPLACE_HD static void fill_outer(double a, int nmax, double* f) {
    for (int n = 1; n <= nmax; ++n)
      f[n] = a < 1.0 ? -math::polygamma(n - 1, 1.0 + a) : -detail_dirichlet::stirling_dn(n, a);
  }

  // d^K/dphi^K l, K >= 1
  template <int K>
  LAPLACE_HD double pure_phi(const state_type& st, const double* mu, const double* lm, const double* al,
                             const double* lr) const {
    // psi_(K-1)(phi) - sum_j mu_j^K psi_(K-1)(alpha_j) [+ sum_j mu_j log y_j at K = 1], from the
    // remainders r = psi - mu_n: the monomials cancel algebraically (sum_j mu_j = 1)
    double s = math::polygamma_monomial_remainder(K - 1, st.phi);
    for (int j = 0; j < categories; ++j) {
      double mk = 1.0;
      for (int q = 0; q < K; ++q) mk *= mu[j];
      s -= mk * math::polygamma_monomial_remainder(K - 1, al[j]);
      if constexpr (K == 1) s += mu[j] * lr[j];
    }
    return s;
  }

  // one entry: An eta derivatives along the sorted tuple t, K in phi
  template <int K, int An>
  LAPLACE_HD double eta_block(const state_type& st, const int* t, const double* pt, const double* mu,
                              const double* lm, const double* al, const double* ly, const double* lr,
                              const double* F, double* P, double* Q, double* ws) const {
    namespace dd = detail_dirichlet;
    namespace ds = detail_simplex;
    constexpr int W = ds::max_order + 1;
    constexpr int full = (1 << An) - 1;
    const int d = categories;
    const double phi = st.phi, lphi = st.log_phi;
    ds::subset_tables tb;
    tb.fill(pt, d, t, An, ws);
    // the outer derivative g_j^(k)(mu_j) of g(m) = m^K F^(K)(phi m)
    auto g = [&](int j, int k) {
      double s = 0.0;
      const int lo = 0, hi = k < K ? k : K;
      for (int i = lo; i <= hi; ++i) {
        double mp = 1.0, pp = 1.0;
        for (int q = 0; q < K - i; ++q) mp *= mu[j];
        for (int q = 0; q < k - i; ++q) pp *= phi;
        s += dd::binom(k, i) * dd::falling(K, i) * mp * pp * F[fn_width * j + K + k - i];
      }
      return s;
    };
    // the categories of the tuple
    auto in_tuple = [&](int j) {
      for (int c = 0; c < tb.nc; ++c)
        if (tb.cats[c] == j) return c;
      return -1;
    };
    // the outside categories share Q (partition sums of G)
    ds::partition_sums(tb.G, An, Q);
    bool any_large = false;
    // y enters as phi sum_j d mu_j log y_j; for the large set it is combined with the
    // -phi d(mu_j log mu_j) of Stirling's leading term, d mu_j log(y_j/mu_j) - H_j (H_j the
    // Leibniz terms with a derivative on log mu_j), since y is close to mu at a large phi
    // sum_j H_j over every category is (An - 1) kappa = -(An - 1) L (from d(sum mu log mu) =
    // d(sum mu eta - LSE)), so the large set's share is that minus the small set's
    double regular = 0.0, yterm = 0.0, small_dm = 0.0, small_dlog = 0.0, large_dlog = 0.0, small_h = 0.0;
    // H = sum_(B strictly inside full) G[B] L[full ^ B]: d_full(mu log mu)/mu - G log mu for outside ones
    double H = 0.0;
    for (int B = 0; B < full; ++B) H += tb.G[B] * tb.L[full ^ B];
    for (int j = 0; j < d; ++j) {
      const int c = in_tuple(j);
      const bool small = al[j] < 1.0;
      double dm, hj, dlg;
      if (c < 0) {
        double s = 0.0, mk = 1.0;
        for (int k = 1; k <= An; ++k) {
          mk *= mu[j];
          s += g(j, k) * mk * Q[full * W + k];
        }
        regular += s;
        dm = mu[j] * tb.G[full];
        hj = mu[j] * H;
        dlg = tb.L[full];
      } else {
        ds::partition_sums(tb.D[c], An, P);
        for (int k = 1; k <= An; ++k) regular += g(j, k) * P[full * W + k];
        dm = tb.D[c][full];
        hj = 0.0;
        for (int B = 0; B < full; ++B) hj += tb.D[c][B] * tb.dlog(full ^ B, c);
        dlg = tb.dlog(full, c);
      }
      if (small) {
        yterm += dm * ly[j];
        small_dm += dm;
        small_dlog += dlg;
        small_h += hj;
      } else {
        any_large = true;
        yterm += dm * lr[j];
        large_dlog += dlg;
      }
    }
    // the linear pieces belong to the large set; summed over the small set with the sign
    // changed (sum_j d mu_j = 0), and exactly zero when the large set is empty
    if (!any_large) small_dm = 0.0;
    const double large_h = any_large ? -(An - 1) * tb.L[full] - small_h : 0.0;
    double v = regular;
    if constexpr (K == 0) {
      v += phi * (yterm - large_h) + small_dlog + phi * (lphi - 1.0) * small_dm + 0.5 * large_dlog;
    } else if constexpr (K == 1) {
      v += yterm - large_h + lphi * small_dm;
    } else {
      // -(-1)^K (K-2)!/phi^(K-1) sum_large d mu = (-1)^K (K-2)!/phi^(K-1) sum_small d mu
      double s = dd::factorial(K - 2);
      for (int q = 0; q < K - 1; ++q) s /= phi;
      v += (K % 2 == 0 ? s : -s) * small_dm;
    }
    return v;
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Compute \f$\log\mu_j\f$ for category `j` from additive log-ratio coordinates `x`.
   *
   * \param x Additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param j Zero-based category index (`0 <= j < categories`).
   * \return Value of \f$\log\mu_j\f$.
   */
  // log mu_j from eta
  LAPLACE_HD double log_mu(const double* x, int j) const { return detail_simplex::log_probability(x, categories, j); }

  /**
   * \ingroup distrib_dirichlet
   * \brief Draw one sample \f$y\f$ on the \f$d\f$-simplex from \f$\operatorname{Dirichlet}(\phi\mu)\f$.
   *
   * Evaluates \f$\log G_j = \log G(\alpha_j + 1) + \log(U_j)/\alpha_j\f$ using `standard_gamma_by_inversion()`
   * at draw `2j` and `open_unit_uniform()` at draw `2j + 1` of `counter_uniform_source{seed, unit}`, then
   * normalizes \f$y_j = G_j / \sum_k G_k\f$ by log-sum-exp so tiny shapes never divide `0/0`.
   *
   * \param st Precomputed state from `theta_state()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param seed Random seed.
   * \param unit Unit identifier.
   * \param y Output sample vector of length `dim()`.
   */
  // draw: log G_j = log G(alpha_j + 1) + log(U_j)/alpha_j, the gamma at draw 2j and the
  // uniform at 2j + 1 of counter_uniform_source{seed, unit}; y by log-sum-exp. w holds d doubles.
  LAPLACE_HD void rng(const state_type& st, const double* x, std::uint64_t seed, std::uint64_t unit, double* y) const {
    counter_uniform_source src{seed, unit};
    double top = -distrib_infinity;
    for (int j = 0; j < categories; ++j) {
      const double a = st.phi * math::exp(log_mu(x, j));
      const double g = standard_gamma_by_inversion(a + 1.0, src);
      const double u = open_unit_uniform(src);
      y[j] = math::log(g) + math::log(u) / a;
      top = y[j] > top ? y[j] : top;
    }
    double s = 0.0;
    for (int j = 0; j < categories; ++j) {
      y[j] = math::exp(y[j] - top);
      s += y[j];
    }
    for (int j = 0; j < categories; ++j) y[j] /= s;
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Write the distribution mean \f$\mathbb{E}[Y] = \mu\f$ into `out`.
   *
   * The first parameter (`theta`) is unused because the mean depends only on \f$\eta\f$.
   *
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output mean vector of length `dim()`.
   */
  LAPLACE_HD void mean(const double*, const double* x, double* out) const {
    for (int j = 0; j < categories; ++j) out[j] = math::exp(log_mu(x, j));
  }
  /**
   * \ingroup distrib_dirichlet
   * \brief Write the packed lower-triangular covariance matrix \f$(\operatorname{diag}(\mu) - \mu\mu^\top)/(\phi + 1)\f$.
   *
   * \param theta Parameter array of length `theta_size()` holding \f$\phi = \theta_0\f$.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output packed lower-triangular buffer of length `pdmat::packed_size(dim())`.
   * \param w Caller scratch buffer of at least `dim()` doubles.
   */
  // (diag mu - mu mu')/(phi + 1), packed by rows of the lower triangle; w holds d doubles
  LAPLACE_HD void covariance(const double* theta, const double* x, double* out, double* w) const {
    for (int j = 0; j < categories; ++j) w[j] = math::exp(log_mu(x, j));
    const double f = 1.0 / (theta[0] + 1.0);
    for (int i = 0; i < categories; ++i)
      for (int j = 0; j <= i; ++j) out[i * (i + 1) / 2 + j] = f * ((i == j ? w[i] : 0.0) - w[i] * w[j]);
  }

  /**
   * \ingroup distrib_dirichlet
   * \brief Compute starting values for \f$\phi\f$ and \f$\eta\f$ on the host from a sample summary.
   *
   * Sets \f$\eta_a = \log(\bar{y}_a / \bar{y}_d)\f$ from the sample mean and estimates \f$\phi\f$ from
   * \f$\phi + 1 = \sum_j \bar{y}_j (1 - \bar{y}_j) / \sum_j \widehat{\operatorname{Var}}(Y_j)\f$
   * (defaulting to `1e6` when the denominator is not positive and floored at `1e-3`).
   *
   * \param s Host-side sample summary from `summarize_mv_sample()`.
   * \param theta Output indexed parameter array of length `theta_size()` receiving \f$\phi\f$.
   * \param x Output dense parameter array of length `categories - 1` receiving \f$\eta\f$.
   */
  // x = alr(sample mean), phi + 1 = sum_j m_j (1 - m_j) / sum_j Var y_j (host)
  void start_value(const distrib_mv_sample_summary& s, double* theta, double* x) const {
    const double lk = std::log(s.mean[categories - 1]);
    for (int a = 0; a + 1 < categories; ++a) x[a] = std::log(s.mean[a]) - lk;
    double num = 0.0, den = 0.0;
    for (int j = 0; j < categories; ++j) {
      num += s.mean[j] * (1.0 - s.mean[j]);
      den += s.covariance[j * (j + 1) / 2 + j];
    }
    const double phi = den > 0.0 ? num / den - 1.0 : 1e6;
    theta[0] = phi > 1e-3 ? phi : 1e-3;  // ponytail: floor 1e-3 for a degenerate sample
  }
};

}  // namespace laplace::distrib
