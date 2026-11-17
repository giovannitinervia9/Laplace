// multinomial_distrib (laplace::distrib): the multinomial law of N trials over d
// categories, its probabilities mu on the simplex carried by their additive
// log-ratio coordinates eta (reference category the last; bijector/alr.hpp),
// N a data constant (a trailing field, as binomial_distrib's size); design in
// note/distrib_multivariate.md section 18.
//
//   l(eta; y) = log N! - sum_j log y_j! + sum_j y_j log mu_j,  sum_j y_j = N,
//
// written, as binomial_distrib's log-mass, in the saddle-point form with
// Stirling's error delta (math/stirling.hpp) and the unit deviance
// D(1 + w) = (1 + w) log1p(w) - w, which has no cancellation at large N:
//
//   l = 1/2 log(2 pi N) + delta(N) - sum_(y_j > 0) [1/2 log(2 pi y_j) + delta(y_j)
//       + N mu_j D(1 + (y_j - N mu_j)/(N mu_j))] - N sum_(y_j = 0) mu_j.
//
// Derivatives in eta: sum_j y_j log mu_j = sum_(a<d-1) y_a eta_a - N LSE(eta), so
// order 1 is y_a - N mu_a and order K >= 2 is -N kappa_t, kappa_t the joint
// cumulant of the categorical indicators (detail_simplex::subset_tables, -L).
// There is no indexed parameter (theta_size() = 0). The expected information is
// N kappa_ab, and its derivatives N kappa_abt are the observed ones without y.
//
// Draw: conditional binomials, y_j ~ Bin(N - sum_(i<j) y_i, mu_j / sum_(i>=j) mu_i)
// for j < d - 1 by inversion, the uniform of category j the j-th of
// counter_uniform_source{seed, unit}; the last category takes the rest.
#pragma once

#include <laplace/bijector/alr.hpp>
#include <laplace/core/config.hpp>
#include <laplace/distrib/binomial.hpp>
#include <laplace/distrib/detail/simplex_common.hpp>
#include <laplace/distrib/generic_quantile.hpp>
#include <laplace/distrib/generic_rng.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/math/elementary.hpp>
#include <laplace/math/stirling.hpp>

#include <cmath>
#include <cstdint>

namespace laplace::distrib {

/**
 * \ingroup distrib_multinomial
 * \brief Empty state of `multinomial_distrib`.
 *
 * The law has no indexed parameter, so `theta_state()` returns this type with no members.
 *
 * \see multinomial_distrib, distrib_multinomial
 */
struct multinomial_state {};

/**
 * \ingroup distrib_multinomial
 * \brief Multinomial distribution of \f$N\f$ trials over \f$d\f$ categories, with probabilities \f$\mu\f$ in additive log-ratio coordinates.
 *
 * Models counts \f$Y = (Y_1, \dots, Y_d)\f$ with \f$\sum_{j=1}^d Y_j = N\f$. The probabilities \f$\mu\f$ form a `simplex_vector` carried by its \f$d - 1\f$ coordinates \f$\eta_a = \log(\mu_a / \mu_d)\f$, the last category being the reference. The trial count \f$N\f$ is the data constant `size`, not a parameter. Satisfies `multivariate_distrib`.
 *
 * \see multinomial_state, distrib_multinomial, distrib_multivariate
 */
struct multinomial_distrib {
  /// \ingroup distrib_multinomial
  /// \brief Number of categories \f$d \ge 2\f$.
  int categories = 2;  // d >= 2
  /// \ingroup distrib_multinomial
  /// \brief Number of trials \f$N \ge 1\f$, an integer held in a `double`; a data constant, not a parameter.
  double size = 1.0;   // N >= 1, an integer held in a double

  /// \ingroup distrib_multinomial
  /// \brief Normalization regime (`distrib_normalization::proper`).
  static constexpr distrib_normalization normalization = distrib_normalization::proper;
  /// \ingroup distrib_multinomial
  /// \brief Distribution kind (`distrib_kind::discrete`: a mass function, with no derivatives in \f$y\f$).
  static constexpr distrib_kind kind = distrib_kind::discrete;
  /// \ingroup distrib_multinomial
  /// \brief Number of declared parameters: \f$\mu\f$ (`simplex_vector`) only.
  static constexpr int n_params = 1;
  /// \ingroup distrib_multinomial
  /// \brief Descriptor of `mu` (dense, \f$d - 1\f$ coordinates in `x`, `infinitely_smooth`).
  static constexpr distrib_mv_param_info params[n_params] = {
      {"mu", distrib_mv_param_kind::simplex_vector, 0.0, 1.0, infinitely_smooth}};
  /// \ingroup distrib_multinomial
  /// \brief Precomputed state type returned by `theta_state()`, here `multinomial_state`.
  using state_type = multinomial_state;

  /**
   * \ingroup distrib_multinomial
   * \brief Default constructor (\f$d = 2\f$ categories, \f$N = 1\f$).
   */
  LAPLACE_HD multinomial_distrib() = default;
  /**
   * \ingroup distrib_multinomial
   * \brief Construct a multinomial distribution with `d` categories and `n` trials.
   * \param d Number of categories (\f$d \ge 2\f$).
   * \param n Number of trials \f$N \ge 1\f$, an integer.
   */
  LAPLACE_HD multinomial_distrib(int d, double n) : categories(d), size(n) {}

  /**
   * \ingroup distrib_multinomial
   * \brief Return the observation dimension \f$d\f$ (`categories`).
   * \return Number of categories \f$d\f$.
   */
  LAPLACE_HD int dim() const { return categories; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the number of observation coordinates, the counts \f$Y_1, \dots, Y_d\f$.
   *
   * The law is discrete, so it has no derivatives in \f$y\f$.
   *
   * \return `categories`.
   */
  // no derivatives in y (discrete); the counts are the d coordinates
  LAPLACE_HD int y_coord_size() const { return categories; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the number of coordinates of the parameter `mu`.
   *
   * The index is not read, since `mu` is the only parameter.
   *
   * \return `categories - 1`.
   */
  LAPLACE_HD int param_size(int) const { return categories - 1; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the starting index of `mu` in `x`.
   * \return `0`.
   */
  LAPLACE_HD int param_offset(int) const { return 0; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the length of the indexed parameter array \f$\theta\f$.
   * \return `0`: the law has no indexed parameter.
   */
  LAPLACE_HD int theta_size() const { return 0; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the number of `double` elements that `theta_state()` stores in caller memory.
   * \return `0`, since `multinomial_state` has no members.
   */
  LAPLACE_HD int state_size() const { return 0; }
  /**
   * \ingroup distrib_multinomial
   * \brief Return the per-unit working scratch size in `double` elements.
   * \return Five blocks of \f$d\f$ doubles, plus the scratch that the cumulant tables need.
   */
  // mu and log mu, d each
  LAPLACE_HD int scratch_size() const { return 5 * categories + detail_simplex::subset_tables::scratch_doubles; }

  /**
   * \ingroup distrib_multinomial
   * \brief Return the empty state of the law.
   *
   * Neither argument is read, since the law has no indexed parameter.
   *
   * \return Empty `multinomial_state`.
   */
  LAPLACE_HD state_type theta_state(const double*, double*) const { return {}; }

  /**
   * \ingroup distrib_multinomial
   * \brief Check whether \f$y\f$ is a point of the support.
   *
   * Returns `true` when every \f$y_j\f$ is a non-negative integer and the \f$d\f$ counts sum to \f$N\f$ exactly in double arithmetic.
   *
   * \param y Observation vector of length `dim()`.
   * \return `true` if `y` is in the support, `false` otherwise.
   */
  // whether y is a point of the support: non-negative integers summing to size
  LAPLACE_HD bool in_support(const double* y) const {
    double s = 0.0;
    for (int j = 0; j < categories; ++j) {
      if (!(y[j] >= 0.0) || y[j] != Kokkos::floor(y[j])) return false;
      s += y[j];
    }
    return s == size;
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Evaluate the multinomial log-mass \f$\ell(\eta; y)\f$.
   *
   * Evaluates the saddle-point form
   * \f[
   * \ell = \tfrac{1}{2}\log(2\pi N) + \delta(N) - \sum_{j : y_j > 0}\Bigl[\tfrac{1}{2}\log(2\pi y_j) + \delta(y_j) + N\mu_j\, D\Bigl(1 + \frac{y_j - N\mu_j}{N\mu_j}\Bigr)\Bigr] - N\sum_{j : y_j = 0}\mu_j,
   * \f]
   * where \f$\delta\f$ is Stirling's error (\ref stirling "stirling") and \f$D(1 + w) = (1 + w)\operatorname{log1p}(w) - w\f$ is the unit deviance (\ref distrib_multinomial "the page of the multinomial"). The probabilities \f$\mu_j\f$ are those of \f$\eta\f$. The form has no cancellation at large \f$N\f$. The value is \f$-\infty\f$ when `in_support(y)` is `false`.
   *
   * \param y Observation vector of length `dim()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \return Log-mass, or \f$-\infty\f$ off the support.
   * \see in_support(), logpmf_dk(), distrib_multinomial
   */
  LAPLACE_HD double logpmf(const state_type&, const double* y, const double* x) const {
    if (!in_support(y)) return -distrib_infinity;
    constexpr double two_pi = 6.28318530717958647693;
    const double n = size;
    double v = 0.5 * math::log(two_pi * n) + math::stirling_error(n), zero_mass = 0.0;
    for (int j = 0; j < categories; ++j) {
      const double lm = detail_simplex::log_probability(x, categories, j);
      const double np = n * math::exp(lm);
      if (y[j] == 0.0) {
        zero_mass += np;
        continue;
      }
      const double e = y[j] - np;
      v -= 0.5 * math::log(two_pi * y[j]) + math::stirling_error(y[j]) + np * detail_binomial::deviance_unit(e / np);
    }
    return v - zero_mass;
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Evaluate the derivatives of the log-mass of order `K` in the indexed parameters and of order `A` in \f$\eta\f$.
   *
   * The law has no indexed parameter, so only \f$K = 0\f$ is defined. For \f$K > 0\f$ the function returns `false` and leaves `out` untouched. The index tuple is not read.
   *
   * For \f$A = 0\f$, `out` receives the log-mass of `logpmf()`. For \f$A = 1\f$, it receives \f$y_a - N\mu_a\f$ for \f$a = 1, \dots, d - 1\f$, formed in double-double because the two terms are close near the mode. For \f$A \ge 2\f$, it receives \f$-N\kappa_t\f$, where \f$\kappa_t\f$ is the joint cumulant of the categorical indicators over the tuple \f$t\f$ of coordinates of \f$\eta\f$, computed as described on \ref distrib_dirichlet_cumulants "the cumulant tables of the Dirichlet page".
   *
   * `out` holds the packed symmetric tensor of order \f$A\f$ over the \f$d - 1\f$ coordinates of \f$\eta\f$, using the storage of \ref distrib_multivariate "Multivariate". The measured maximum error of these blocks is \f$1.65\f$ `ulp1N` on `gcc` and on CUDA, see \ref accuracy "How accuracy is measured".
   *
   * \tparam K Derivative order in the indexed parameters (only \f$K = 0\f$ is defined).
   * \tparam A Derivative order in \f$\eta\f$ (at most one order; `K + A <= 6`).
   * \param st Precomputed state from `theta_state()`, not used.
   * \param y Observation vector of length `dim()`.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output buffer of length `sym_tensor_size(categories - 1, A)` doubles.
   * \param w Caller working scratch of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when \f$K > 0\f$.
   * \see logpmf(), expected_info_dk(), distrib_multinomial
   */
  // K derivatives in theta (none: only K = 0) and A in eta; out holds
  // sym_tensor_size(d - 1, A) doubles in the multivariate layout
  template <int K, int... A>
  LAPLACE_HD bool logpmf_dk(const state_type& st, const double* y, const double* x, const int*, double* out,
                            double* w) const {
    static_assert(sizeof...(A) <= 1, "multinomial_distrib has one dense parameter");
    constexpr int An = (0 + ... + A);
    static_assert(K + An <= 6, "orders up to six");
    if constexpr (K > 0) {
      return false;
    } else if constexpr (An == 0) {
      out[0] = logpmf(st, y, x);
      return true;
    } else {
      const int m = categories - 1;
      double* mu = w;
      probabilities(x, mu);
      if constexpr (An == 1) {
        // y_a - N mu_a in double-double, mu_a from the unit's point: the two terms are close
        // near the mode, and N times the rounding of mu_a would be the whole error
        namespace dm = math::detail::ddm;
        double* pt = w + categories;
        detail_simplex::fill_point(x, categories, pt);
        for (int a = 0; a < m; ++a) {
          const dm::dd r = dm::dd(y[a]) - dm::dd(pt[2 * a], pt[2 * a + 1]) * size;
          out[a] = r.hi + r.lo;
        }
      } else {
        detail_simplex::fill_point(x, categories, w + categories);
        cumulant_tensor<An>(w + categories, size, out, w + 5 * categories);  // -N kappa = N L
      }
      return true;
    }
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Evaluate the expected information in \f$\eta\f$, or its derivative of order `K` in the indexed parameters and of order `A` in \f$\eta\f$ (\f$K + A \le 4\f$).
   *
   * Only the block of \f$\eta\f$ with itself exists, at slot `(0, 0)`. For \f$K > 0\f$, or for a slot other than `0`, the function returns `false` and leaves `out` untouched. The entries are \f$N\kappa_{ab}\f$, where \f$\kappa_{ab}\f$ is the joint cumulant of the categorical indicators for the coordinates \f$a, b = 1, \dots, d - 1\f$ of \f$\eta\f$. Their derivatives of order \f$A\f$ in \f$\eta\f$, the \f$N\kappa_{abt}\f$, do not depend on \f$y\f$.
   *
   * The integer arrays `ab` and `idx` are not read. `out` holds one block of `sym_tensor_size(categories - 1, A)` doubles for each packed entry \f$(a, b)\f$. Within a block, the information entry comes first and the derivatives follow in the packed order of the tensor of order \f$A\f$. The measured maximum error of the information is \f$0.717\f$ `ulp1N` on `gcc` and on CUDA, see \ref accuracy "How accuracy is measured".
   *
   * \tparam K Derivative order in the indexed parameters (only \f$K = 0\f$ is defined).
   * \tparam A Derivative order in \f$\eta\f$ (at most one order; `K + A <= 4`).
   * \param slot_a Row parameter slot; only `0`, the dense \f$\mu\f$, is accepted.
   * \param slot_b Column parameter slot; only `0` is accepted.
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output buffer receiving the information block and its order-\f$A\f$ tensor in \f$\eta\f$.
   * \param w Caller working scratch of at least `scratch_size()` doubles.
   * \return `true` if evaluated; `false` when \f$K > 0\f$ or a slot is not `0`.
   * \see logpmf_dk(), distrib_multinomial, distrib_multivariate
   */
  // the expected information in eta, N kappa_ab, and its derivatives in eta:
  // slot_a = slot_b = 0 (the only dense parameter), out holds
  // packed_size(d - 1) * sym_tensor_size(d - 1, A) doubles, the information
  // entry first (Kronecker)
  template <int K, int... A>
  LAPLACE_HD bool expected_info_dk(const state_type&, int slot_a, int slot_b, const int*, const double* x, const int*,
                                   double* out, double* w) const {
    static_assert(sizeof...(A) <= 1, "multinomial_distrib has one dense parameter");
    constexpr int An = (0 + ... + A);
    static_assert(K + An <= 4, "information derivatives up to order four");
    if constexpr (K > 0) {
      return false;
    } else {
      if (slot_a != 0 || slot_b != 0) return false;
      double* mu = w;
      probabilities(x, mu);
      detail_simplex::fill_point(x, categories, w + categories);
      const int m = categories - 1;
      const int ne = sym_tensor_size(m, An);
      detail_simplex::tuple_walk ab(2, m), e(An, m);
      ab.first();
      int pa = 0;
      do {
        e.first();
        int pe = 0;
        do {
          int t[6];
          int n = 0;
          t[n++] = ab.t[0];
          t[n++] = ab.t[1];
          for (int q = 0; q < An; ++q) t[n++] = e.t[q];
          sort_small(t, n);
          detail_simplex::subset_tables tb;
          tb.fill(w + categories, categories, t, n, w + 5 * categories);
          out[pa * ne + pe] = -size * tb.L[(1 << n) - 1];
          ++pe;
        } while (e.next());
        ++pa;
      } while (ab.next());
      return true;
    }
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Draw one count vector \f$y\f$ from the multinomial law.
   *
   * The counts of the categories \f$j = 1, \dots, d - 1\f$ are drawn in turn as conditional binomials, \f$Y_j \sim \operatorname{Bin}\bigl(N - \sum_{i < j} Y_i,\, \mu_j / \sum_{i \ge j} \mu_i\bigr)\f$, by inversion of the binomial distribution with distrib_quantile(). The uniform of category \f$j\f$ is the \f$j\f$-th draw of `counter_uniform_source{seed, unit}`. The last category takes the remaining trials, and a category takes zero once no trials are left. The probabilities are held in a buffer of 64 doubles, so \f$d \le 64\f$.
   *
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param seed Seed of the counter-based uniform source.
   * \param unit Unit identifier that selects the uniform stream.
   * \param y Output counts of length `dim()`.
   */
  LAPLACE_HD void rng(const state_type&, const double* x, std::uint64_t seed, std::uint64_t unit, double* y) const {
    double mu[64];  // ponytail: d <= 64 on the device; a scratch argument if larger d appear
    probabilities(x, mu);
    counter_uniform_source src{seed, unit};
    double left = size;
    for (int j = 0; j + 1 < categories; ++j) {
      const double u = open_unit_uniform(src);
      double r = 0.0;
      for (int i = j; i < categories; ++i) r += mu[i];  // the remaining mass as a sum
      if (left > 0.0) {
        const double p = mu[j] / r;
        y[j] = p >= 1.0 ? left : distrib_quantile(binomial_distrib{p, left}, u);
      } else {
        y[j] = 0.0;
      }
      left -= y[j];
    }
    y[categories - 1] = left;
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Write the mean \f$\mathbb{E}[Y] = N\mu\f$ into `out`.
   *
   * The first argument (`theta`) is not read.
   *
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output mean vector of length `dim()`.
   */
  // E[y] = size mu
  LAPLACE_HD void mean(const double*, const double* x, double* out) const {
    probabilities(x, out);
    for (int j = 0; j < categories; ++j) out[j] *= size;
  }
  /**
   * \ingroup distrib_multinomial
   * \brief Write the packed lower-triangular covariance \f$N(\operatorname{diag}\mu - \mu\mu^\top)\f$.
   *
   * The entries are stored by rows of the lower triangle. The first argument (`theta`) is not read.
   *
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param out Output packed lower-triangular buffer of length \f$d(d+1)/2\f$.
   * \param w Caller scratch buffer of at least `dim()` doubles.
   */
  // size (diag mu - mu mu'), packed by rows of the lower triangle; w holds d doubles
  LAPLACE_HD void covariance(const double*, const double* x, double* out, double* w) const {
    probabilities(x, w);
    for (int i = 0; i < categories; ++i)
      for (int j = 0; j <= i; ++j) out[i * (i + 1) / 2 + j] = size * ((i == j ? w[i] : 0.0) - w[i] * w[j]);
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Compute the starting value of \f$\eta\f$ on the host from a sample summary.
   *
   * Divides the sample mean of the counts by \f$N\f$ to get the pooled proportions \f$\bar{p}_j\f$, then sets \f$\eta_a = \log(\bar{p}_a / \bar{p}_d)\f$ for \f$a = 1, \dots, d - 1\f$. The array `theta` is empty and is not written. This member runs on the host only.
   *
   * \param s Host-side sample summary from `summarize_mv_sample()`.
   * \param x Output dense parameter array of length `categories - 1` receiving \f$\eta\f$.
   */
  // x from the pooled proportions mean/size, alr's forward map (host); theta is empty
  void start_value(const distrib_mv_sample_summary& s, double*, double* x) const {
    const double lk = std::log(s.mean[categories - 1] / size);
    for (int a = 0; a + 1 < categories; ++a) x[a] = std::log(s.mean[a] / size) - lk;
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Compute the category probabilities \f$\mu\f$ from the additive log-ratio coordinates.
   *
   * Each \f$\mu_j\f$ is the exponential of its log-probability, formed with the largest category factored out; see \ref distrib_dirichlet_coords "the coordinates on the Dirichlet page".
   *
   * \param x Dense additive log-ratio coordinates \f$\eta\f$ of length `categories - 1`.
   * \param mu Output probability vector of length `dim()`.
   */
  // mu from eta (detail_simplex::log_probability)
  LAPLACE_HD void probabilities(const double* x, double* mu) const {
    for (int j = 0; j < categories; ++j) mu[j] = math::exp(detail_simplex::log_probability(x, categories, j));
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Helper of `logpmf_dk()`: writes `scale` times the cumulant table entries of order `An` in the multivariate layout.
   */
  // scale times L (= -kappa) of order An over eta, multivariate layout
  template <int An>
  LAPLACE_HD void cumulant_tensor(const double* pt, double scale, double* out, double* ws) const {
    detail_simplex::tuple_walk e(An, categories - 1);
    e.first();
    int p = 0;
    do {
      detail_simplex::subset_tables tb;
      tb.fill(pt, categories, e.t, An, ws);
      out[p++] = scale * tb.L[(1 << An) - 1];
    } while (e.next());
  }

  /**
   * \ingroup distrib_multinomial
   * \brief Helper of `expected_info_dk()`: sorts the first `n` entries of `t` into increasing order.
   */
  LAPLACE_HD static void sort_small(int* t, int n) {
    for (int i = 1; i < n; ++i) {
      const int v = t[i];
      int k = i - 1;
      while (k >= 0 && t[k] > v) {
        t[k + 1] = t[k];
        --k;
      }
      t[k + 1] = v;
    }
  }
};

}  // namespace laplace::distrib
