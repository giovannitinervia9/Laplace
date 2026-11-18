// A penalty through a linear map, pen(beta) = P(D beta) + c(D) (note/penalties.md
// section 5d, decided with Giovanni on 2026-11-18): the adaptive weights (D = W
// diagonal) and the fused lasso and trend filtering (D the differences of order
// k). The map is not a distribution: with m < p rows there is no law of beta.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/penalty/core.hpp>
#include <laplace/penalty/distrib_penalty.hpp>
#include <laplace/penalty/independent_product_penalty.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace laplace::penalty {

/**
 * \ingroup penalty_linear_map
 * \brief The diagonal map \f$u = W\beta\f$, \f$W = \operatorname{diag}(w)\f$ with every \f$w_j > 0\f$.
 *
 * Square and invertible, so a proper penalty stays proper, with
 * \f$c(W) = -\sum_j \log w_j\f$. The weights are read through a pointer (not
 * copied), which must be readable where the map is constructed and used.
 */
struct diagonal_map {
  /// The map is diagonal: the penalty keeps the structure and the kinks of \f$P\f$.
  static constexpr bool is_diagonal = true;
  /// The map is square and invertible: a proper \f$P\f$ gives a proper penalty.
  static constexpr bool preserves_proper = true;
  /// Order of the differences (0: not a difference map).
  static constexpr int difference_order = 0;

  /// The weights \f$w\f$, length `size`.
  const double* weights = nullptr;
  /// The dimension \f$p = m\f$.
  int size = 0;
  /// \f$c(W) = -\sum_j \log w_j\f$, set by the constructor.
  double log_normalizer_value = 0.0;
  /// \f$\max_j w_j^2 = \lVert W\rVert_2^2\f$, set by the constructor.
  double norm_squared = 0.0;

  diagonal_map() = default;
  /**
   * \brief Construct from the weights.
   * \param w Weights, length `n`, every \f$w_j > 0\f$ (not copied).
   * \param n Dimension.
   */
  LAPLACE_HD diagonal_map(const double* w, int n) : weights(w), size(n) {
    for (int j = 0; j < n; ++j) {
      log_normalizer_value -= Kokkos::log(w[j]);
      norm_squared = Kokkos::fmax(norm_squared, w[j] * w[j]);
    }
  }
  /// \return The number of rows \f$m\f$.
  LAPLACE_HD int m() const { return size; }
  /// \return The number of columns \f$p\f$.
  LAPLACE_HD int p() const { return size; }
  /// \brief \f$u = W\beta\f$. \param beta Length \f$p\f$. \param u Length \f$m\f$.
  LAPLACE_HD void apply(const double* beta, double* u) const {
    for (int j = 0; j < size; ++j) u[j] = weights[j] * beta[j];
  }
  /// \brief \f$W^\top g\f$. \param g Length \f$m\f$. \param out Length \f$p\f$.
  LAPLACE_HD void apply_transpose(const double* g, double* out) const {
    for (int j = 0; j < size; ++j) out[j] = weights[j] * g[j];
  }
  /// \return The bound \f$\max_j w_j^2\f$ on \f$\lVert W\rVert_2^2\f$ (here equal to it).
  LAPLACE_HD double norm_squared_bound() const { return norm_squared; }
  /// \return \f$c(W) = -\sum_j \log w_j\f$.
  LAPLACE_HD double log_normalizer() const { return log_normalizer_value; }
  /// \return The coefficient of row \f$i\f$ at column \f$i\f$, \f$w_i\f$.
  LAPLACE_HD double coefficient(int i, int) const { return weights[i]; }
};

/**
 * \ingroup penalty_linear_map
 * \brief The differences of order \f$K\f$, \f$u_i = \sum_{j=0}^{K} (-1)^{K-j}\binom{K}{j}\beta_{i+j}\f$, \f$i < p - K\f$.
 *
 * \f$D\f$ is \f$(p - K) \times p\f$ of full row rank; its null space, the
 * polynomials of degree below \f$K\f$, does not depend on the hyperparameters,
 * so a proper penalty on \f$D\beta\f$ is intrinsic, with
 * \f$c(D) = -\tfrac12\log\det(DD^\top)\f$ from the closed form
 * \f$\det(DD^\top) = \prod_{j=0}^{K-1} \binom{p+j}{2j+1}/\binom{2j}{j}\f$.
 *
 * \tparam K Order of the differences, \f$K \ge 1\f$.
 */
template <int K>
struct difference_map {
  static_assert(K >= 1, "difference_map: the order is at least 1");
  /// The map is not diagonal.
  static constexpr bool is_diagonal = false;
  /// \f$m < p\f$: a proper \f$P\f$ gives an intrinsic penalty.
  static constexpr bool preserves_proper = false;
  /// Order of the differences.
  static constexpr int difference_order = K;

  /// The dimension \f$p\f$ of \f$\beta\f$, at least \f$K + 1\f$.
  int columns = 0;
  /// \f$c(D) = -\tfrac12\log\det(DD^\top)\f$, set by the constructor.
  double log_normalizer_value = 0.0;

  difference_map() = default;
  /**
   * \brief Construct for \f$\beta\f$ of dimension `p`.
   * \param p Dimension, at least \f$K + 1\f$.
   */
  LAPLACE_HD explicit difference_map(int p) : columns(p) {
    // log det(D D^T) = sum_j [log binom(p + j, 2j + 1) - log binom(2j, j)], each binomial
    // a product of ratios, exact to rounding (lgamma differences cancel at large p)
    double s = 0.0;
    for (int j = 0; j < K; ++j) {
      for (int i = 0; i <= 2 * j; ++i) s += Kokkos::log(double(p + j - i) / double(i + 1));
      for (int i = 0; i < j; ++i) s -= Kokkos::log(double(2 * j - i) / double(i + 1));
    }
    log_normalizer_value = -0.5 * s;
  }
  /// \return The number of rows \f$m = p - K\f$.
  LAPLACE_HD int m() const { return columns - K; }
  /// \return The number of columns \f$p\f$.
  LAPLACE_HD int p() const { return columns; }
  /// \return The coefficient \f$(-1)^{K-j}\binom{K}{j}\f$ of row \f$i\f$ at column \f$i + j\f$.
  LAPLACE_HD static double coefficient(int, int j) {
    const double c = double(distrib::binomial_coef(K, j));
    return (K - j) % 2 == 0 ? c : -c;
  }
  /// \brief \f$u = D\beta\f$. \param beta Length \f$p\f$. \param u Length \f$m\f$.
  LAPLACE_HD void apply(const double* beta, double* u) const {
    for (int i = 0; i < m(); ++i) {
      double s = 0.0;
      for (int j = 0; j <= K; ++j) s += coefficient(i, j) * beta[i + j];
      u[i] = s;
    }
  }
  /// \brief \f$D^\top g\f$. \param g Length \f$m\f$. \param out Length \f$p\f$.
  LAPLACE_HD void apply_transpose(const double* g, double* out) const {
    for (int l = 0; l < columns; ++l) out[l] = 0.0;
    for (int i = 0; i < m(); ++i)
      for (int j = 0; j <= K; ++j) out[i + j] += coefficient(i, j) * g[i];
  }
  /// \return The bound \f$4^K \ge \lVert D\rVert_2^2\f$.
  LAPLACE_HD double norm_squared_bound() const { return double(1 << (2 * K)); }
  /// \return \f$c(D) = -\tfrac12\log\det(DD^\top)\f$.
  LAPLACE_HD double log_normalizer() const { return log_normalizer_value; }
};

namespace detail_linear_map {

// Whether P is the lasso of a product of one univariate law with a kink term
// rate |y - mu| (laplace1, laplace2): its prox on first differences is the 1-D
// total-variation prox, computed by Condat's algorithm.
template <class P>
struct tv_reading {
  static constexpr bool eligible = false;
};
template <class D>
concept has_l1_rate = requires(const double* t) { univariate_penalty_reading<D>::l1_rate(t); };
template <class D>
struct tv_reading<distrib_penalty<distrib::independent_product_distrib<distrib::univariate_as_multivariate<D>>>> {
  static constexpr bool eligible = has_l1_rate<D>;
  using law = D;
};

template <class P>
concept has_prox_beta_steps = requires(const P p, const typename P::state_type& s, const double* v, double* q) {
  p.prox_beta(s, v, v, q, q);
};

// Condat's direct algorithm for 1-D total-variation denoising (L. Condat, "A
// direct algorithm for 1-D total variation denoising", IEEE Signal Processing
// Letters 20(11), 2013): x = argmin (1/2)|x - y|^2 + lam sum_i |x_{i+1} - x_i|.
// O(n) in practice, O(n^2) in the worst case; exact, with no tolerance.
LAPLACE_HD void tv1d_denoise(const double* y, int n, double lam, double* x) {
  if (n <= 0) return;
  int k = 0, k0 = 0, kplus = 0, kminus = 0;
  double umin = lam, umax = -lam, vmin = y[0] - lam, vmax = y[0] + lam;
  for (;;) {
    while (k == n - 1) {
      if (umin < 0.0) {
        do x[k0++] = vmin;
        while (k0 <= kminus);
        k = kminus = k0;
        vmin = y[k];
        umin = lam;
        umax = vmin + umin - vmax;
      } else if (umax > 0.0) {
        do x[k0++] = vmax;
        while (k0 <= kplus);
        k = kplus = k0;
        vmax = y[k];
        umax = -lam;
        umin = vmax + umax - vmin;
      } else {
        vmin += umin / (k - k0 + 1);
        do x[k0++] = vmin;
        while (k0 <= k);
        return;
      }
    }
    if ((umin += y[k + 1] - vmin) < -lam) {
      do x[k0++] = vmin;
      while (k0 <= kminus);
      k = kplus = kminus = k0;
      vmin = y[k];
      vmax = vmin + 2.0 * lam;
      umin = lam;
      umax = -lam;
    } else if ((umax += y[k + 1] - vmax) > lam) {
      do x[k0++] = vmax;
      while (k0 <= kplus);
      k = kplus = kminus = k0;
      vmax = y[k];
      vmin = vmax - 2.0 * lam;
      umin = lam;
      umax = -lam;
    } else {
      ++k;
      if (umin >= lam) {
        kminus = k;
        vmin += (umin - lam) / (k - k0 + 1);
        umin = lam;
      }
      if (umax <= -lam) {
        kplus = k;
        vmax += (umax + lam) / (k - k0 + 1);
        umax = -lam;
      }
    }
  }
}

}  // namespace detail_linear_map

/**
 * \ingroup penalty_linear_map
 * \brief State of `linear_map_penalty`: that of \f$P\f$ and the total-variation rate.
 * \tparam S State type of \f$P\f$.
 */
template <class S>
struct linear_map_penalty_state {
  /// State of \f$P\f$.
  S inner;
  /// Rate \f$a\f$ of the kink terms \f$a\lvert u_i - \mu_{0i}\rvert\f$ when all of them share one rate (Condat's prox), NaN otherwise.
  double tv_rate;
};

/**
 * \ingroup penalty_linear_map
 * \brief The penalty \f$\mathrm{pen}(\beta;\theta) = P(D\beta;\theta) + c(D)\f$ of a separable penalty \f$P\f$ through a map \f$D\f$.
 *
 * Derivatives in \f$\beta\f$ by the chain rule, the order-\f$J\f$ tensor
 * \f$\sum_i T_i\, d_i^{\otimes J}\f$ with \f$T_i\f$ the derivative of \f$P\f$
 * in \f$u_i\f$ and \f$d_i\f$ row \f$i\f$ of \f$D\f$: through `diagonal_map` in
 * the layout of \f$P\f$ (one double per coordinate), through `difference_map`
 * in the dense layout over \f$p\f$ (\f$\binom{p+J-1}{J}\f$ doubles).
 * Derivatives in \f$\theta\f$ are those of \f$P\f$; \f$c(D)\f$ is constant in
 * \f$\theta\f$ and enters the value only (not at all for an `unnormalizable` \f$P\f$).
 *
 * Class: a proper \f$P\f$ stays proper through `diagonal_map` and becomes
 * intrinsic through `difference_map`; an unnormalizable \f$P\f$ stays so; an
 * intrinsic \f$P\f$ is refused (a Gaussian on a map is `mvgaussian2` with
 * \f$Q = \lambda D^\top D\f$). Proximal operator: through `diagonal_map` in
 * closed form by a step per coordinate, \f$[\operatorname{prox}_{t f(W\cdot)}(v)]_j
 * = \operatorname{prox}_{t w_j^2 f_j}(w_j v_j)/w_j\f$; through `difference_map<1>`
 * with laplace1 or laplace2 (the 1-D total variation) by Condat's direct
 * algorithm, which needs one rate shared by all differences; none otherwise.
 *
 * \tparam P Separable penalty on \f$u\f$, not intrinsic.
 * \tparam Map `diagonal_map` or `difference_map<K>`.
 */
template <penalty P, class Map>
struct linear_map_penalty {
  static_assert(P::normalization != penalty_normalization::intrinsic,
                "linear_map_penalty: an intrinsic P is refused; a Gaussian on a map is mvgaussian2 with Q = lambda D^T D");
  // ponytail: separable P only; a diagonal map over a grouped P (group lasso with
  // weights) scales tuples within blocks, added with its first caller
  static_assert(P::structure == penalty_structure::separable, "linear_map_penalty: P must be separable");
  using reading = detail_linear_map::tv_reading<P>;
  static constexpr bool uses_condat = Map::difference_order == 1 && reading::eligible;

  /// The penalty on \f$u = D\beta\f$.
  P inner_penalty;
  /// The map \f$D\f$.
  Map linear_map;

  static constexpr penalty_normalization normalization =
      P::normalization == penalty_normalization::unnormalizable ? penalty_normalization::unnormalizable
      : Map::preserves_proper                                 ? P::normalization
                                                              : penalty_normalization::intrinsic;
  static constexpr int beta_regularity = P::beta_regularity;
  static constexpr penalty_kink_geometry kinks =
      P::kinks == penalty_kink_geometry::none ? penalty_kink_geometry::none
      : (Map::is_diagonal && P::kinks == penalty_kink_geometry::per_coordinate) ? penalty_kink_geometry::per_coordinate
                                                                                : penalty_kink_geometry::linear_map;
  static constexpr penalty_prox_kind prox =
      (Map::is_diagonal && detail_linear_map::has_prox_beta_steps<P>) || uses_condat ? penalty_prox_kind::closed_form
                                                                                     : penalty_prox_kind::none;
  static constexpr penalty_structure structure = Map::is_diagonal ? P::structure : penalty_structure::coupled;
  static constexpr int theta_regularity = P::theta_regularity;
  static constexpr bool q_linear_in_theta = P::q_linear_in_theta;
  static constexpr penalty_scale_mixture scale_mixture = P::scale_mixture;
  using state_type = linear_map_penalty_state<typename P::state_type>;

  linear_map_penalty() = default;
  /**
   * \brief Construct from a penalty on \f$u\f$ and a map.
   * \param inner Penalty \f$P\f$ of dimension `map.m()`.
   * \param map The map \f$D\f$.
   */
  LAPLACE_PENALTY_FORWARD_HD linear_map_penalty(const P& inner, const Map& map) : inner_penalty(inner), linear_map(map) {}

  /// \return The penalty \f$P\f$ on \f$u = D\beta\f$, for a primal-dual method splitting \f$f(\beta) + P(D\beta)\f$.
  LAPLACE_HD const P& inner() const { return inner_penalty; }
  /// \return The map \f$D\f$: `m()`, `p()`, `apply()`, `apply_transpose()`, `norm_squared_bound()`.
  LAPLACE_HD const Map& map() const { return linear_map; }

  LAPLACE_PENALTY_FORWARD_HD int dim() const { return linear_map.p(); }
  LAPLACE_PENALTY_FORWARD_HD int theta_size() const { return inner_penalty.theta_size(); }
  LAPLACE_PENALTY_FORWARD_HD int state_size() const { return inner_penalty.state_size(); }
  /// \return Doubles of scratch: that of \f$P\f$, then \f$3p\f$.
  LAPLACE_PENALTY_FORWARD_HD int scratch_size() const { return inner_penalty.scratch_size() + 3 * linear_map.p(); }

  /**
   * \brief Compute what depends on \f$\theta\f$ only.
   * \param theta Hyperparameters of \f$P\f$, length `theta_size()`.
   * \param w Caller workspace of `state_size() + scratch_size()` doubles.
   * \return The state.
   */
  LAPLACE_PENALTY_FORWARD_HD state_type theta_state(const double* theta, double* w) const {
    state_type s{inner_penalty.theta_state(theta, w), Kokkos::Experimental::quiet_NaN_v<double>};
    if constexpr (uses_condat) {
      // theta of the product is one rate parameter per set; Condat needs them equal
      const int h = inner_penalty.distribution.components.head.n_sets;
      bool shared = true;
      for (int i = 1; i < h; ++i) shared = shared && theta[i] == theta[0];
      if (shared) s.tv_rate = univariate_penalty_reading<typename reading::law>::l1_rate(theta);
    }
    return s;
  }

  /**
   * \brief Return the weak-convexity constant, the bound \f$\rho_P \lVert D\rVert_2^2\f$.
   * \param theta Hyperparameters, length `theta_size()`.
   * \return \f$\rho_P\f$ times `map().norm_squared_bound()`.
   */
  // ponytail: rho_P max_j w_j^2 bounds max_j rho_j w_j^2 from above; per set when a
  // non-convex P with weights needs the sharper constant
  LAPLACE_HD double weak_convexity(const double* theta) const {
    return inner_penalty.weak_convexity(theta) * linear_map.norm_squared_bound();
  }

  /**
   * \brief The penalty value \f$P(D\beta) + c(D)\f$.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \param w Scratch of `scratch_size()` doubles.
   * \return Penalty value.
   */
  LAPLACE_PENALTY_FORWARD_HD double value(const state_type& st, const double* beta, double* w) const {
    double* u = w + inner_penalty.scratch_size();
    linear_map.apply(beta, u);
    return inner_penalty.value(st.inner, u, w) + constant();
  }

  /**
   * \brief Derivatives of the penalty: `J` in \f$\beta\f$ and `K` in \f$\theta\f$ along the sorted tuple `idx`.
   *
   * Through `diagonal_map`: one double per coordinate,
   * \f$w_j^J\f$ times that of \f$P\f$ at \f$u = W\beta\f$. Through
   * `difference_map`: the dense layout over \f$p\f$ of
   * `distrib::mv_logpdf_dy_dk_blocks` (scalar, vector, packed matrix by rows
   * of the lower triangle, sorted tuples), \f$\binom{p+J-1}{J}\f$ doubles.
   * The value (\f$J = K = 0\f$) includes \f$c(D)\f$.
   *
   * \tparam J Derivative order in \f$\beta\f$.
   * \tparam K Derivative order in \f$\theta\f$.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \param idx Sorted tuple of `K` parameter indices (unused when `K == 0`).
   * \param out Output buffer.
   * \param w Scratch of `scratch_size()` doubles.
   * \return `true` if written; `false` and `out` untouched when structurally zero.
   */
  LAPLACE_PENALTY_NO_EXEC_CHECK
  template <int J, int K>
  LAPLACE_HD bool value_dbeta_dk(const state_type& st, const double* beta, const int* idx, double* out,
                                 double* w) const {
    const int m = linear_map.m(), p = linear_map.p();
    double* u = w + inner_penalty.scratch_size();
    double* tu = u + p;
    linear_map.apply(beta, u);
    if constexpr (J == 0) {
      if (!inner_penalty.template value_dbeta_dk<0, K>(st.inner, u, idx, out, w)) return false;
      if constexpr (K == 0) out[0] += constant();
      return true;
    } else if constexpr (Map::is_diagonal) {
      if (!inner_penalty.template value_dbeta_dk<J, K>(st.inner, u, idx, out, w)) return false;
      for (int i = 0; i < m; ++i) {
        double s = 1.0;
        for (int r = 0; r < J; ++r) s *= linear_map.weights[i];
        out[i] *= s;
      }
      return true;
    } else {
      if (!inner_penalty.template value_dbeta_dk<J, K>(st.inner, u, idx, tu, w)) return false;
      constexpr int order = Map::difference_order;
      const int n = distrib::binomial_coef(p + J - 1, J);
      for (int k = 0; k < n; ++k) out[k] = 0.0;
      for (int i = 0; i < m; ++i) {
        // every sorted J-tuple over the support {i, ..., i + D} of row i
        int t[J], g[J];
        for (int r = 0; r < J; ++r) t[r] = 0;
        do {
          double c = tu[i];
          for (int r = 0; r < J; ++r) {
            c *= Map::coefficient(i, t[r]);
            g[r] = i + t[r];
          }
          out[distrib::detail_mv::block_position(p, J, g)] += c;
        } while (distrib::detail_mv::next_block_tuple(order + 1, J, t));
      }
      return true;
    }
  }

  /**
   * \brief Proximal operator \f$\arg\min_b \{\mathrm{pen}(b; \theta) + \lVert b - v\rVert^2/(2t)\}\f$.
   *
   * Through `diagonal_map`: \f$P\f$'s prox at \f$W v\f$ with steps
   * \f$t w_j^2\f$, divided by \f$w\f$. Through `difference_map<1>`: Condat's
   * algorithm at rate \f$t a\f$ on \f$v - b_0\f$, \f$Db_0 = \mu_0\f$, plus
   * \f$b_0\f$; NaN when the rates of the differences are not all equal.
   * \param st State from `theta_state()`.
   * \param v Point, length `dim()`.
   * \param t Step \f$t > 0\f$.
   * \param out Minimizer, length `dim()`.
   * \param w Scratch of `scratch_size()` doubles.
   */
  LAPLACE_PENALTY_FORWARD_HD void prox_beta(const state_type& st, const double* v, double t, double* out,
                                            double* w) const
    requires(prox != penalty_prox_kind::none)
  {
    const int p = linear_map.p();
    double* a = w + inner_penalty.scratch_size();
    double* b = a + p;
    double* c = b + p;
    if constexpr (Map::is_diagonal) {
      const double* wt = linear_map.weights;
      for (int j = 0; j < p; ++j) {
        a[j] = wt[j] * v[j];
        b[j] = t * wt[j] * wt[j];
      }
      inner_penalty.prox_beta(st.inner, a, b, c, w);
      for (int j = 0; j < p; ++j) out[j] = c[j] / wt[j];
    } else {
      // pen = rate |D beta - mu0|_1: beta = beta' + b0 with b0 the cumulative sum of mu0
      const double* mu0 = inner_penalty.mu0;
      b[0] = 0.0;
      for (int j = 1; j < p; ++j) b[j] = b[j - 1] + (mu0 != nullptr ? mu0[j - 1] : 0.0);
      for (int j = 0; j < p; ++j) a[j] = v[j] - b[j];
      if (Kokkos::isnan(st.tv_rate)) {
        for (int j = 0; j < p; ++j) out[j] = st.tv_rate;
        return;
      }
      detail_linear_map::tv1d_denoise(a, p, t * st.tv_rate, c);
      for (int j = 0; j < p; ++j) out[j] = c[j] + b[j];
    }
  }

 private:
  LAPLACE_HD double constant() const {
    return normalization == penalty_normalization::unnormalizable ? 0.0 : linear_map.log_normalizer();
  }
};

}  // namespace laplace::penalty
