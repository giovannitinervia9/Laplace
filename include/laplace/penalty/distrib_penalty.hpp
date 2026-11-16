// distrib_penalty<M>: the penalty of a multivariate distribution M,
//
//   pen(beta; theta) = -log p_M(beta | mu_0, theta),
//
// beta playing the role of y, mu_0 a fixed prior location (zero by default) and
// theta the hyperparameters of M's structure; note/penalties.md section 5a.
// The value is -logpdf and the derivatives in beta (order J), in theta (order K)
// and the mixed ones are -logpdf_dy_dk<J, K>, in the layout of the
// multivariate interface (distrib/multivariate_core.hpp). The normalization
// class comes from M::normalization: proper -> proper, improper -> intrinsic;
// a degenerate M is refused (its -log p is +infinity off a subspace: a
// constraint, not a penalty). The members forward to M and are LAPLACE_HD, so
// the adapter runs on the device when M does and stays on the host when M is
// host only.
//
// The descriptors of the penalty are those of M, written in
// distrib_penalty_traits<M>, specialized for the laws that have a penalty
// reading (so far mvgaussian2_distrib<S>); another M does not compile.
#pragma once

#include <laplace/distrib/multivariate_core.hpp>
#include <laplace/distrib/mvgaussian2.hpp>
#include <laplace/penalty/core.hpp>

#include <type_traits>

// The members forward to M, which may be host only (the improper mvgaussian2 over
// weighted_sum_psdmat): nvcc would warn (#20014) for such an instantiation although
// it is never called on the device, so the check is switched off on these members.
#if defined(__CUDACC__)
#define LAPLACE_PENALTY_NO_EXEC_CHECK _Pragma("nv_exec_check_disable")
#define LAPLACE_PENALTY_FORWARD_HD LAPLACE_PENALTY_NO_EXEC_CHECK LAPLACE_HD
#else
#define LAPLACE_PENALTY_NO_EXEC_CHECK
#define LAPLACE_PENALTY_FORWARD_HD LAPLACE_HD
#endif

namespace laplace::pdmat {
class weighted_sum_psdmat;
}

namespace laplace::penalty {

/**
 * \ingroup penalty
 * \brief Descriptors of the penalty of a multivariate distribution `M`, other than its normalization.
 *
 * Undefined for a general `M`: it is specialized for each law that has a
 * penalty reading. For `distrib::mvgaussian2_distrib<S>` the penalty is
 * quadratic: \f$C^\infty\f$ in \f$\beta\f$ and in \f$\theta\f$, convex
 * (`weak_convexity` is 0), without kinks or proximal operator, coupled
 * through \f$Q\f$, and with \f$Q\f$ linear in the weights \f$\lambda_k\f$
 * exactly when `S` is `pdmat::weighted_sum_psdmat`.
 *
 * \tparam M Multivariate distribution type.
 */
template <class M>
struct distrib_penalty_traits;

/**
 * \ingroup penalty
 * \brief Descriptors of the penalty of `distrib::mvgaussian2_distrib<S>`.
 * \tparam S Matrix structure.
 */
template <class S>
struct distrib_penalty_traits<distrib::mvgaussian2_distrib<S>> {
  /// Regularity in \f$\beta\f$: \f$C^\infty\f$.
  static constexpr int beta_regularity = distrib::infinitely_smooth;
  /// Kinks: none.
  static constexpr penalty_kink_geometry kinks = penalty_kink_geometry::none;
  /// Proximal operator: none.
  static constexpr penalty_prox_kind prox = penalty_prox_kind::none;
  /// Structure: coupled through the precision matrix.
  static constexpr penalty_structure structure = penalty_structure::coupled;
  /// Regularity in \f$\theta\f$: \f$C^\infty\f$.
  static constexpr int theta_regularity = distrib::infinitely_smooth;
  /// \f$Q = \sum_k \lambda_k S_k\f$ exactly for `pdmat::weighted_sum_psdmat`.
  static constexpr bool q_linear_in_theta = std::is_same_v<S, pdmat::weighted_sum_psdmat>;
  /// Scale mixture: none declared.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// Weak-convexity constant: 0, the penalty is convex.
  LAPLACE_HD static double weak_convexity(const double*) { return 0.0; }
};

/**
 * \ingroup penalty
 * \brief The penalty \f$-\log p_M(\beta \mid \mu_0, \theta)\f$ of a multivariate distribution `M`.
 *
 * \f$\beta\f$ plays the role of \f$y\f$, \f$\mu_0\f$ is a fixed prior location
 * (zero when `nullptr`) and \f$\theta\f$ the hyperparameters of `M`. The value is
 * \f$-\log p_M\f$ with its normalizing constant, and the derivatives in
 * \f$\beta\f$, in \f$\theta\f$ and the mixed ones are those of
 * `M::logpdf_dy_dk<J, K>` with the sign flipped. The class is `proper` for a
 * proper `M` and `intrinsic` for an improper one; a degenerate `M` does not compile.
 *
 * \tparam M Type satisfying `distrib::multivariate_distrib` with a specialization of `distrib_penalty_traits`.
 */
template <distrib::multivariate_distrib M>
struct distrib_penalty {
  static_assert(M::normalization != distrib::distrib_normalization::degenerate,
                "distrib_penalty: a degenerate distribution is a constraint, not a penalty");
  /// Compile-time penalty descriptor traits of `M`.
  using traits = distrib_penalty_traits<M>;

  /// The distribution whose negative log-density is the penalty.
  M distribution;
  /// Prior location \f$\mu_0\f$ (non-owning, length `dim()`), or `nullptr` for zero.
  const double* mu0 = nullptr;

  /// Normalization class derived from `M::normalization`.
  static constexpr penalty_normalization normalization = M::normalization == distrib::distrib_normalization::proper
                                                             ? penalty_normalization::proper
                                                             : penalty_normalization::intrinsic;
  /// Regularity in \f$\beta\f$.
  static constexpr int beta_regularity = traits::beta_regularity;
  /// Kink geometry.
  static constexpr penalty_kink_geometry kinks = traits::kinks;
  /// Proximal operator availability.
  static constexpr penalty_prox_kind prox = traits::prox;
  /// Coupling of the coordinates.
  static constexpr penalty_structure structure = traits::structure;
  /// Regularity in \f$\theta\f$.
  static constexpr int theta_regularity = traits::theta_regularity;
  /// Whether \f$Q\f$ is linear in the weights.
  static constexpr bool q_linear_in_theta = traits::q_linear_in_theta;
  /// Scale-mixture representation.
  static constexpr penalty_scale_mixture scale_mixture = traits::scale_mixture;
  /// Precomputed state type of `M`, valid while `w` and `theta` are.
  using state_type = typename M::state_type;

  /// Default constructor.
  distrib_penalty() = default;
  /**
   * \brief Construct from a distribution and an optional prior location.
   * \param m The distribution `M`.
   * \param location Prior location \f$\mu_0\f$ of length `m.dim()` (not copied), or `nullptr` for zero.
   */
  LAPLACE_PENALTY_FORWARD_HD explicit distrib_penalty(const M& m, const double* location = nullptr) : distribution(m), mu0(location) {}

  /// \return The dimension \f$d\f$ of \f$\beta\f$.
  LAPLACE_PENALTY_FORWARD_HD int dim() const { return distribution.dim(); }
  /// \return The number of hyperparameters.
  LAPLACE_PENALTY_FORWARD_HD int n_params() const { return distribution.n_params(); }
  /// \return Doubles of the state written by `theta_state()`.
  LAPLACE_PENALTY_FORWARD_HD int state_size() const { return distribution.state_size(); }
  /// \return Doubles of scratch `w` after the state: that of `M` plus `dim()` for the zero location.
  LAPLACE_PENALTY_FORWARD_HD int scratch_size() const { return distribution.scratch_size() + distribution.dim(); }

  /**
   * \brief Compute what depends on \f$\theta\f$ only (log determinant, matrix, factor).
   * \param theta Hyperparameters, length `n_params()`.
   * \param w Caller workspace of `state_size() + scratch_size()` doubles.
   * \return The state of `M`.
   */
  LAPLACE_PENALTY_FORWARD_HD state_type theta_state(const double* theta, double* w) const { return distribution.theta_state(theta, w); }

  /**
   * \brief Return the weak-convexity constant \f$\sup(-\mathrm{pen}'')\f$ at `theta`.
   * \param theta Hyperparameters, length `n_params()`.
   * \return Weak-convexity constant (0 for a convex penalty).
   */
  LAPLACE_HD double weak_convexity(const double* theta) const { return traits::weak_convexity(theta); }

  /**
   * \brief The penalty value \f$-\log p_M(\beta \mid \mu_0, \theta)\f$.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \param w Scratch of `scratch_size()` doubles.
   * \return Penalty value.
   */
  LAPLACE_PENALTY_FORWARD_HD double value(const state_type& st, const double* beta, double* w) const {
    return -distribution.logpdf(st, beta, location(w));
  }

  /**
   * \brief Derivatives of the penalty: `J` in \f$\beta\f$ and `K` in \f$\theta\f$ along the sorted tuple `idx`.
   *
   * Equal to \f$-\f$ `M::logpdf_dy_dk<J, K>`; the output layout is that of `M`
   * (scalar, `dim()`-vector, packed symmetric matrix for \f$J = 0, 1, 2\f$) and
   * `false` is returned, `out` untouched, when the block is zero by structure
   * or not provided (\f$J \ge 3\f$).
   *
   * \tparam J Derivative order in \f$\beta\f$.
   * \tparam K Derivative order in \f$\theta\f$.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \param idx Sorted tuple of `K` parameter indices (unused when `K == 0`).
   * \param out Output buffer.
   * \param w Scratch of `scratch_size()` doubles.
   * \return `true` if written; `false` and `out` untouched when structurally zero or \f$J \ge 3\f$.
   */
  LAPLACE_PENALTY_NO_EXEC_CHECK
  template <int J, int K>
  LAPLACE_HD bool value_dbeta_dk(const state_type& st, const double* beta, const int* idx, double* out,
                                 double* w) const {
    if (!distribution.template logpdf_dy_dk<J, K>(st, beta, location(w), idx, out, w)) return false;
    const int n = J == 0 ? 1 : (J == 1 ? dim() : pdmat::packed_size(dim()));
    for (int k = 0; k < n; ++k) out[k] = -out[k];
    return true;
  }

 private:
  // mu_0, or zeros in the tail of w after the scratch of M
  LAPLACE_PENALTY_FORWARD_HD const double* location(double* w) const {
    if (mu0 != nullptr) return mu0;
    double* z = w + distribution.scratch_size();
    for (int k = 0; k < distribution.dim(); ++k) z[k] = 0.0;
    return z;
  }
};

}  // namespace laplace::penalty
