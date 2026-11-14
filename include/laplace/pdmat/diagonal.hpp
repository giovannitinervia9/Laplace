// diagonal_pdmat: M = diag(sigma_i^2) with theta_i = log sigma_i, i = 0..p-1
// (note/pdmat_structures.md; interface in core.hpp).
//
//   M_ii = e^(2 theta_i), log det M = 2 sum_i theta_i, M^-1_ii = e^(-2 theta_i),
//   L_ii = e^(theta_i).
// A derivative is nonzero only along a tuple whose indices are all equal to
// one i, and then only at (i, i): 2^K e^(2 theta_i), (-2)^K e^(-2 theta_i),
// e^(theta_i); log det has derivative 2 in each theta_i and none above.
#pragma once

#include <laplace/pdmat/core.hpp>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with diagonal structure \f$M(\theta) = \operatorname{diag}(e^{2\theta_i})\f$.
 *
 * Maps \f$p\f$ unconstrained real parameters \f$\theta_i = \log \sigma_i\f$ to a diagonal matrix
 * with entries \f$M_{ii} = \sigma_i^2 = e^{2\theta_i} > 0\f$.
 *
 * \par Measured accuracy
 * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
 * - \f$M\f$: 0.52 ulpN (gcc) / 0.52 ulpN (CUDA)
 * - \f$M^{-1}\f$: 0.89 ulpN (gcc) / 0.66 ulpN (CUDA)
 * - \f$L\f$: 0.59 ulpN (gcc) / 0.59 ulpN (CUDA)
 * - \f$\log \det M\f$: 0.45 ulp (gcc) / 0.45 ulp (CUDA)
 *
 * \see pdmat, pdmat_orders, pdmat_structure
 */
struct diagonal_pdmat : pdmat_orders<diagonal_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;

  /// \brief Compile-time descriptor: \c false, diagonal entries \f$e^{2\theta_i}\f$ are not identically 1.
  static constexpr bool unit_diagonal = false;
  /// \brief Compile-time descriptor: \c true, exact closed-form inverse is implemented.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Cholesky factor is implemented.
  static constexpr bool closed_factor = true;
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  diagonal_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs a diagonal structured matrix of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit diagonal_pdmat(int p) : dim(p) {}

  /**
   * \ingroup pdmat
   * \brief Number of parameters \f$q = p\f$.
   * \return Matrix dimension \c dim.
   */
  LAPLACE_HD int n_params() const { return dim; }

  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch any function needs: 0 for this structure.
   * \return 0.
   */
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed diagonal matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ii} = e^{2\theta_i}\f$ on the diagonal and \f$M_{ij} = 0\f$ for \f$i \ne j\f$.
   *
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 0.52 ulpN (gcc) / 0.52 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, [[maybe_unused]] double* w) const { fill(theta, 2.0, m); }
  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Computes \f$M^{-1}_{ii} = e^{-2\theta_i}\f$ on the diagonal and \f$M^{-1}_{ij} = 0\f$ for \f$i \ne j\f$.
   *
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 0.89 ulpN (gcc) / 0.66 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, [[maybe_unused]] double* w) const { fill(theta, -2.0, m); }
  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Computes \f$L_{ii} = e^{\theta_i}\f$ on the diagonal and \f$L_{ij} = 0\f$ for \f$i \ne j\f$, satisfying \f$M = L L^\top\f$.
   *
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 0.59 ulpN (gcc) / 0.59 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, [[maybe_unused]] double* w) const { fill(theta, 1.0, l); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * Computes:
   * \f[
   * \log \det M(\theta) = 2 \sum_{i=0}^{p-1} \theta_i
   * \f]
   *
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * \par Measured accuracy
   * Maximum error 0.45 ulp (gcc) / 0.45 ulp (CUDA); test bound 1.0 ulp.
   *
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += theta[i];
    return 2.0 * s;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$ along parameter index tuple \c idx.
   *
   * For a pure tuple with \f$i_1 = \dots = i_K = i\f$:
   * \f[
   * \frac{\partial^K M_{ii}}{\partial\theta_i^K} = 2^K e^{2\theta_i}
   * \f]
   * and all other entries are zero.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if computed; \c false and buffer left untouched if \c idx contains mixed indices.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.52 ulpN (gcc) / 0.52 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    return one_entry(theta, idx, 2.0, m);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$ along parameter index tuple \c idx.
   *
   * For \f$K = 1\f$, returns \f$2.0\f$; for \f$K \ge 2\f$, returns \f$0.0\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Value of the derivative, or \f$0.0\f$ if \f$K \ge 2\f$ or \c idx is mixed.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.45 ulp (gcc) / 0.45 ulp (CUDA); test bound 1.0 ulp.
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk([[maybe_unused]] const double* theta, [[maybe_unused]] const int (&idx)[K], [[maybe_unused]] double* w) const {
    return K == 1 ? 2.0 : 0.0;
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$ along parameter index tuple \c idx.
   *
   * For a pure tuple with \f$i_1 = \dots = i_K = i\f$:
   * \f[
   * \frac{\partial^K M^{-1}_{ii}}{\partial\theta_i^K} = (-2)^K e^{-2\theta_i}
   * \f]
   * and all other entries are zero.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if computed; \c false and buffer left untouched if \c idx contains mixed indices.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.89 ulpN (gcc) / 0.66 ulpN (CUDA); test bounds 1.0 to 2.0 ulpN.
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    return one_entry(theta, idx, -2.0, m);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$ along parameter index tuple \c idx.
   *
   * For a pure tuple with \f$i_1 = \dots = i_K = i\f$:
   * \f[
   * \frac{\partial^K L_{ii}}{\partial\theta_i^K} = e^{\theta_i}
   * \f]
   * and all other entries are zero.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$p\f$ parameters \f$\theta_i\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if computed; \c false and buffer left untouched if \c idx contains mixed indices.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.59 ulpN (gcc) / 0.59 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, const int (&idx)[K], double* l, [[maybe_unused]] double* w) const {
    return one_entry(theta, idx, 1.0, l);
  }

 private:
  // the diagonal e^(c theta_i)
  LAPLACE_HD void fill(const double* theta, double c, double* m) const {
    detail_pdmat::fill_zero(m, packed_size(dim));
    for (int i = 0; i < dim; ++i) m[packed_index(i, i)] = detail_pdmat::m::exp(c * theta[i]);
  }
  // c^K e^(c theta_i) at (i, i) when every index of the tuple is i
  template <int K>
  LAPLACE_HD bool one_entry(const double* theta, const int (&idx)[K], double c, double* m) const {
    for (int k = 1; k < K; ++k)
      if (idx[k] != idx[0]) return false;
    const int i = idx[0];
    detail_pdmat::fill_zero(m, packed_size(dim));
    m[packed_index(i, i)] = detail_pdmat::ipow(c, K) * detail_pdmat::m::exp(c * theta[i]);
    return true;
  }
};

static_assert(pdmat_structure<diagonal_pdmat>);

}  // namespace laplace::pdmat
