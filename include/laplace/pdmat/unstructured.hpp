// unstructured_pdmat<B>: any symmetric positive definite M, or any correlation
// matrix, through a coupled bijector B (note/pdmat_structures.md section 5;
// interface in core.hpp). theta is the bijector's free vector u, with its
// packing, and M = h(theta) is the bijector's inverse map:
//
//   B = log_cholesky_bijector   q = p(p+1)/2, M = L L^T, L_ii = e^(theta_ii)
//   B = matrix_log_bijector     q = p(p+1)/2, M = exp(S), S the symmetric
//                               matrix whose packed lower triangle is theta
//   B = partial_corr_bijector   q = p(p-1)/2, M a correlation matrix
//                               (unstructured_corr_pdmat)
//
// log det M in closed form, written here from the map:
//   log_cholesky  2 sum_i theta_ii; d/dtheta_ii = 2, nothing above order 1
//   matrix_log    tr S = sum_i theta_ii; d/dtheta_ii = 1, nothing above
//   partial_corr  2 sum_a log L_aa = -2 sum_(a>b) log cosh theta_ab;
//                 order K: -2 tanh^(K-1)(theta_ab) on a tuple of the one
//                 variable (a, b), 0 on a mixed tuple
// M^-1: matrix_log in closed form, exp(-S), with d^K exp(-S) = (-1)^K times
// the bijector's d^K exp at -theta; the other two by the generic route of
// core.hpp from the bijector's Cholesky factor. The Cholesky factor and its
// derivatives are the bijector's (not for matrix_log). The log-Jacobian of
// theta -> M is the bijector's inverse log-Jacobian, in its reference
// coordinates: the lower triangle with the diagonal for log_cholesky and
// matrix_log, the strict lower triangle for partial_corr.
#pragma once

#include <laplace/bijector/log_cholesky.hpp>
#include <laplace/bijector/matrix_log.hpp>
#include <laplace/bijector/partial_corr.hpp>
#include <laplace/pdmat/core.hpp>

#include <type_traits>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix or correlation matrix parameterized by a coupled bijector.
 *
 * Maps an unconstrained parameter vector \f$\theta \in \mathbb{R}^q\f$ to a matrix \f$M(\theta) \in \mathbb{R}^{p \times p}\f$
 * via the inverse map \f$h(\theta)\f$ of coupled bijector \c B (\ref bijector_coupled).
 *
 * The template parameter \c B must be one of:
 * - \ref laplace::bijector::log_cholesky_bijector "log_cholesky_bijector": \f$q = p(p+1)/2\f$, parameterizing any symmetric
 *   positive definite matrix \f$M = L L^\top\f$ with \f$L_{ii} = e^{\theta_{ii}}\f$ and \f$L_{ij} = \theta_{ij}\f$ (\f$i > j\f$).
 * - \ref laplace::bijector::matrix_log_bijector "matrix_log_bijector": \f$q = p(p+1)/2\f$, parameterizing any symmetric
 *   positive definite matrix \f$M = \exp(S)\f$, \f$S\f$ the symmetric matrix with \f$S_{ij} = S_{ji} = \theta_{ij}\f$.
 * - \ref laplace::bijector::partial_corr_bijector "partial_corr_bijector": \f$q = p(p-1)/2\f$, parameterizing any
 *   correlation matrix (unit diagonal \f$M_{ii} = 1\f$) via canonical partial correlations.
 *
 * \tparam B One of the three coupled bijectors listed above (\ref bijector_coupled); any other type is rejected by a static assertion.
 *
 * \see pdmat, pdmat_orders, pdmat_structure, unstructured_corr_pdmat, bijector_coupled
 */
template <class B>
struct unstructured_pdmat : pdmat_orders<unstructured_pdmat<B>> {
  static_assert(std::is_same_v<B, bijector::log_cholesky_bijector> || std::is_same_v<B, bijector::matrix_log_bijector> ||
                    std::is_same_v<B, bijector::partial_corr_bijector>,
                "unstructured_pdmat takes log_cholesky_bijector, matrix_log_bijector or partial_corr_bijector");

  static constexpr bool is_log_cholesky = std::is_same_v<B, bijector::log_cholesky_bijector>;
  static constexpr bool is_matrix_log = std::is_same_v<B, bijector::matrix_log_bijector>;
  static constexpr bool is_partial_corr = std::is_same_v<B, bijector::partial_corr_bijector>;

  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;
  /// \brief Coupled bijector instance of type \c B.
  B b{1};

  /// \brief Compile-time descriptor: \c true if \c B is \ref laplace::bijector::partial_corr_bijector "partial_corr_bijector" (unit diagonal entries \f$M_{ii} = 1\f$).
  static constexpr bool unit_diagonal = is_partial_corr;
  /// \brief Compile-time descriptor: \c true if \c B is \ref laplace::bijector::matrix_log_bijector "matrix_log_bijector" (exact closed-form inverse is implemented).
  static constexpr bool closed_inverse = is_matrix_log;
  /// \brief Compile-time descriptor: \c true if \c B is not \ref laplace::bijector::matrix_log_bijector "matrix_log_bijector" (exact Cholesky factor is implemented).
  static constexpr bool closed_factor = !is_matrix_log;
  /// \brief Compile-time descriptor: \c true, the parameterization \f$\theta \mapsto M\f$ is a bijection onto its image and supplies \c log_jacobian() and its derivatives.
  static constexpr bool has_density = true;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  unstructured_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs an unstructured matrix structure of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit unstructured_pdmat(int p) : dim(p), b(p) {}

  /**
   * \ingroup pdmat
   * \brief Number of unconstrained parameters \f$q\f$.
   * \return \f$p(p+1)/2\f$ for log-Cholesky and matrix logarithm, or \f$p(p-1)/2\f$ for partial correlation.
   */
  LAPLACE_HD int n_params() const { return b.free_size(); }
  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch buffer \c w required by functions of this structure.
   * \return Number of doubles: \f$64p^2 + 64p(p+1)/2 + p^2 + q\f$ plus the bijector's scratch size.
   */
  // the generic route's buffers (or -theta for matrix_log), then the bijector's
  LAPLACE_HD int scratch_size() const {
    return detail_pdmat::generic_inverse_scratch(dim) + n_params() + b.scratch_size();
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed positive definite matrix \f$M(\theta)\f$.
   *
   * Evaluates the bijector's inverse map \f$M(\theta) = h(\theta)\f$:
   * - log_cholesky: \f$M = L L^\top\f$, \f$L_{ii} = e^{\theta_{ii}}\f$, \f$L_{ij} = \theta_{ij}\f$ (\f$i > j\f$).
   * - matrix_log: \f$M = \exp(S)\f$, \f$S_{ij} = S_{ji} = \theta_{ij}\f$.
   * - partial_corr: \f$M = L L^\top\f$, \f$L_{ab} = \tanh(\theta_{ab})\prod_{l<b}\operatorname{sech}(\theta_{al})\f$ (\f$a > b\f$), \f$L_{aa} = \prod_{l<a}\operatorname{sech}(\theta_{al})\f$.
   *
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param m Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse map; see \ref bijector_coupled.
   *
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const { b.inverse(theta, m, w); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * Evaluated in closed form:
   * - log_cholesky: \f$\log \det M = 2\sum_{i=0}^{p-1} \theta_{ii}\f$.
   * - matrix_log: \f$\log \det M = \operatorname{tr} S = \sum_{i=0}^{p-1} \theta_{ii}\f$.
   * - partial_corr: \f$\log \det M = -2\sum_{a>b} \log\cosh\theta_{ab}\f$.
   *
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles (unused).
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
   * - log_cholesky: 0.28 ulp (\f$K=0\f$, gcc) / 0.28 ulp (\f$K=0\f$, CUDA)
   * - matrix_log: 0.06 ulp (\f$K=0\f$, gcc) / 0.06 ulp (\f$K=0\f$, CUDA)
   * - partial_corr: 28.4 ulp (\f$K=6\f$, gcc) / 7.38 ulp (\f$K=6\f$, CUDA)
   *
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const {
    double s = 0.0;
    if constexpr (is_partial_corr) {
      for (int k = 0; k < n_params(); ++k) s += log_cosh(theta[k]);
      return -2.0 * s;
    } else {
      for (int i = 0; i < dim; ++i) s += theta[packed_index(i, i)];
      return is_log_cholesky ? 2.0 * s : s;
    }
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * - matrix_log: evaluated in closed form as \f$M^{-1} = \exp(-S)\f$ via the bijector at \f$-\theta\f$.
   * - log_cholesky and partial_corr: evaluated via forward substitution \f$Y = L^{-1}\f$ and Gram summation \f$M^{-1} = Y^\top Y\f$.
   *
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param m Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
   * - log_cholesky: 28.2 ulpN (\f$K=4\f$, gcc) / 23.9 ulpN (\f$K=4\f$, CUDA)
   * - matrix_log: 24.2 ulpN (\f$K=1\f$, gcc) / 21.7 ulpN (\f$K=1\f$, CUDA)
   * - partial_corr: 59.7 ulpN (\f$K=6\f$, gcc) / 52.2 ulpN (\f$K=6\f$, CUDA)
   *
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, double* w) const {
    if constexpr (is_matrix_log) {
      double* neg = w;
      for (int k = 0; k < n_params(); ++k) neg[k] = -theta[k];
      b.inverse(neg, m, w + n_params());
    } else {
      detail_pdmat::generic_inverse(*this, theta, m, w);
    }
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Evaluates the lower-triangular factor \f$L\f$ satisfying \f$M = L L^\top\f$.
   * Available only when \c closed_factor is \c true (log_cholesky and partial_corr).
   *
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param l Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse Cholesky factor; see \ref bijector_coupled.
   *
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, double* w) const
    requires(closed_factor)
  {
    b.inverse_cholesky(theta, l, w);
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-Jacobian determinant of \f$\theta \mapsto M(\theta)\f$.
   *
   * Returns \f$\log\lvert\det J\rvert\f$ of \f$\theta \mapsto M\f$ in reference coordinates: the lower triangle
   * with the diagonal for log_cholesky and matrix_log, and the strict lower triangle for partial_corr.
   *
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Log-Jacobian determinant.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse log-Jacobian; see \ref bijector_coupled.
   *
   * \see log_jacobian_dk(), pdmat_orders::log_jacobian_d1()
   */
  // log |det J| of theta -> M, in the bijector's reference coordinates
  LAPLACE_HD double log_jacobian(const double* theta, double* w) const { return b.inverse_log_jacobian(theta, w); }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates \f$\frac{\partial^K M}{\partial\theta_{i_1}\cdots\partial\theta_{i_K}}\f$
   * via the bijector's inverse derivative.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero for that tuple.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse derivative; see \ref bijector_coupled.
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    return b.template inverse_dk<K>(theta, idx, m, w);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$.
   *
   * Evaluated in closed form:
   * - log_cholesky: at \f$K = 1\f$, 2 on diagonal coordinates \f$(i, i)\f$, 0 otherwise; 0 for \f$K \ge 2\f$.
   * - matrix_log: at \f$K = 1\f$, 1 on diagonal coordinates \f$(i, i)\f$, 0 otherwise; 0 for \f$K \ge 2\f$.
   * - partial_corr: \f$-2\tanh^{(K-1)}(\theta_{ab})\f$ if every index in \c idx is that of the one coordinate \f$\theta_{ab}\f$, and 0 on mixed tuples.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles (unused).
   * \return Value of the derivative, or \f$0.0\f$ if structurally zero for that tuple.
   *
   * \par Measured accuracy
   * Exact (0 ulp) for log_cholesky and matrix_log. For partial_corr, maximum error over orders 0–6:
   * 28.4 ulp (\f$K=6\f$, gcc) / 7.38 ulp (\f$K=6\f$, CUDA).
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], [[maybe_unused]] double* w) const {
    if constexpr (is_partial_corr) {
      for (int k = 1; k < K; ++k)
        if (idx[k] != idx[0]) return 0.0;
      return -2.0 * bijector::detail_coupled::tanh_d(K - 1, theta[idx[0]]);
    } else {
      if (K != 1) return 0.0;
      int i, j;
      bijector::detail_coupled::unpack(idx[0], i, j);
      if (i != j) return 0.0;
      return is_log_cholesky ? 2.0 : 1.0;
    }
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * - matrix_log: \f$(-1)^K\f$ times the bijector's derivative \f$\partial^K \exp(-S)\f$ evaluated at \f$-\theta\f$.
   * - log_cholesky and partial_corr: evaluated by the generic subset recursion on \f$Y = L^{-1}\f$ and Leibniz's rule for \f$M^{-1} = Y^\top Y\f$ (described on the module page, \ref pdmat).
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero for that tuple.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
   * - log_cholesky: 28.2 ulpN (\f$K=4\f$, gcc) / 23.9 ulpN (\f$K=4\f$, CUDA)
   * - matrix_log: 24.2 ulpN (\f$K=1\f$, gcc) / 21.7 ulpN (\f$K=1\f$, CUDA)
   * - partial_corr: 59.7 ulpN (\f$K=6\f$, gcc) / 52.2 ulpN (\f$K=6\f$, CUDA)
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    if constexpr (is_matrix_log) {
      double* neg = w;
      for (int k = 0; k < n_params(); ++k) neg[k] = -theta[k];
      b.template inverse_dk<K>(neg, idx, m, w + n_params());
      if (K % 2)
        for (int k = 0; k < packed_size(dim); ++k) m[k] = -m[k];
      return true;
    } else {
      return detail_pdmat::generic_inverse_dk(*this, theta, idx, m, w);
    }
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates \f$\frac{\partial^K L}{\partial\theta_{i_1}\cdots\partial\theta_{i_K}}\f$
   * via the bijector's inverse Cholesky derivative.
   * Available only when \c closed_factor is \c true (log_cholesky and partial_corr).
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param l Output buffer of at least \f$p(p+1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero for that tuple.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse Cholesky derivative; see \ref bijector_coupled.
   *
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, const int (&idx)[K], double* l, double* w) const
    requires(closed_factor)
  {
    return b.template inverse_cholesky_dk<K>(theta, idx, l, w);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the log-Jacobian of order \f$K\f$.
   *
   * Evaluates \f$\frac{\partial^K \log\lvert\det J\rvert}{\partial\theta_{i_1}\cdots\partial\theta_{i_K}}\f$
   * in reference coordinates via the bijector's inverse log-Jacobian derivative.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \c n_params() unconstrained parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero for that tuple.
   *
   * \par Measured accuracy
   * Bit for bit identical to the bijector's inverse log-Jacobian derivative; see \ref bijector_coupled.
   *
   * \see log_jacobian(), pdmat_orders::log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double log_jacobian_dk(const double* theta, const int (&idx)[K], double* w) const {
    return b.template inverse_log_jacobian_dk<K>(theta, idx, w);
  }

 private:
  // log cosh x = |x| + log1p(e^(-2|x|)) - log 2, without overflow
  LAPLACE_HD static double log_cosh(double x) {
    const double a = x < 0.0 ? -x : x;
    return a + Kokkos::log1p(detail_pdmat::m::exp(-2.0 * a)) - 0.69314718055994530942;
  }
};

/**
 * \ingroup pdmat
 * \brief Alias for unstructured correlation matrix structure using the partial correlation bijector.
 *
 * Parameterizes any symmetric positive definite correlation matrix (\f$M_{ii} = 1\f$) of dimension \f$p\f$
 * via \f$q = p(p-1)/2\f$ unconstrained parameters mapping to canonical partial correlations.
 *
 * \see unstructured_pdmat, bijector::partial_corr_bijector
 */
using unstructured_corr_pdmat = unstructured_pdmat<bijector::partial_corr_bijector>;

static_assert(pdmat_structure<unstructured_pdmat<bijector::log_cholesky_bijector>>);
static_assert(pdmat_structure<unstructured_pdmat<bijector::matrix_log_bijector>>);
static_assert(pdmat_structure<unstructured_corr_pdmat>);

}  // namespace laplace::pdmat
