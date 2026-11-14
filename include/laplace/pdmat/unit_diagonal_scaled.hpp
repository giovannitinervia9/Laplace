// unit_diagonal_scaled_pdmat<R>: M = D R D with D = diag(e^(s_0), ..., e^(s_(p-1))),
// s_i = log sigma_i, and R a structure with unit diagonal
// (note/pdmat_structures.md section 6, "The scaling composition"; interface in
// core.hpp). theta = (s, r): the p scales (one with common_scale, D = e^s I),
// then the q_R parameters r of R.
//
//   M_ij = e^(s_i + s_j) R_ij, M^-1_ij = e^(-s_i - s_j) R^-1_ij, L = D L_R,
//   log det M = 2 sum_i s_i + log det R,
//   log-Jacobian = p log 2 + (p + 1) sum_i s_i + log-Jacobian of R
//   (reference coordinates: the lower triangle of M with its diagonal; R's
//   are its strict lower triangle), only when R has a density and the scales
//   are not common (else NaN).
// Derivatives: a sorted tuple is a run A of scale indices followed by the
// indices B of R. With c_ij(A) = prod_(k in A) (delta_ik + delta_jk),
//   d M_ij = c_ij(A) e^(s_i + s_j) d^B R_ij,
//   d M^-1_ij = (-1)^|A| c_ij(A) e^(-s_i - s_j) d^B R^-1_ij,
//   d L_ij = prod_(k in A) delta_ik e^(s_i) d^B L_R,ij;
// with common_scale c = 2^|A| for M and M^-1 and 1 for L. R writes d^B into
// the output buffer, which is rescaled in place: no scratch beyond R's.
#pragma once

#include <laplace/pdmat/core.hpp>

namespace laplace::pdmat {

namespace detail_pdmat {

enum class scaled_part { matrix, inverse, cholesky, log_det, log_jacobian };

// one quantity of R of order B along sub[0..B-1]; the matrix quantities
// return 1 (computed) or 0 (identically zero, m untouched)
template <int B, class R>
LAPLACE_HD double scaled_part_dk(const R& r, scaled_part q, const double* t, const int* sub, double* m, double* w) {
  int idx[B];
  for (int k = 0; k < B; ++k) idx[k] = sub[k];
  switch (q) {
    case scaled_part::matrix: return r.template matrix_dk<B>(t, idx, m, w) ? 1.0 : 0.0;
    case scaled_part::inverse: return r.template inverse_dk<B>(t, idx, m, w) ? 1.0 : 0.0;
    case scaled_part::cholesky:
      if constexpr (R::closed_factor) return r.template cholesky_dk<B>(t, idx, m, w) ? 1.0 : 0.0;
      else return 0.0;
    case scaled_part::log_det: return r.template log_det_dk<B>(t, idx, w);
    default:
      if constexpr (R::has_density) return r.template log_jacobian_dk<B>(t, idx, w);
      else return 0.0;
  }
}

// the same with the order b = 0..6 known at run time (b = 0: the value)
template <class R>
LAPLACE_HD double scaled_part_runtime(const R& r, scaled_part q, int b, const double* t, const int* sub, double* m, double* w) {
  switch (b) {
    case 0:
      switch (q) {
        case scaled_part::matrix: r.matrix(t, m, w); return 1.0;
        case scaled_part::inverse: r.inverse(t, m, w); return 1.0;
        case scaled_part::cholesky:
          if constexpr (R::closed_factor) r.cholesky(t, m, w);
          return 1.0;
        case scaled_part::log_det: return r.log_det(t, w);
        default:
          if constexpr (R::has_density) return r.log_jacobian(t, w);
          else return 0.0;
      }
    case 1: return scaled_part_dk<1>(r, q, t, sub, m, w);
    case 2: return scaled_part_dk<2>(r, q, t, sub, m, w);
    case 3: return scaled_part_dk<3>(r, q, t, sub, m, w);
    case 4: return scaled_part_dk<4>(r, q, t, sub, m, w);
    case 5: return scaled_part_dk<5>(r, q, t, sub, m, w);
    default: return scaled_part_dk<6>(r, q, t, sub, m, w);
  }
}

}  // namespace detail_pdmat

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix formed by diagonal scaling of a unit-diagonal structure: \f$M = D R D\f$.
 *
 * Scales an underlying structure \f$R\f$ having unit diagonal by a diagonal scale matrix
 * \f$D = \operatorname{diag}(e^{s_0}, \dots, e^{s_{p-1}})\f$, with log-scales \f$s_i = \log\sigma_i\f$.
 * The template argument \c R must satisfy \ref pdmat_structure and have compile-time descriptor \c unit_diagonal set to \c true;
 * the compiler rejects any other structure.
 * Without a unit diagonal the scales would not be identifiable: \f$DRD = (cD)(R/c^2)(cD)\f$ for every \f$c > 0\f$.
 *
 * The parameter vector is \f$\theta = (s_0, \dots, s_{p-1}, r)\f$, where \f$r\f$ comprises the \f$q_R\f$ parameters
 * of \f$R\f$. When \c common_scale is \c true, a single scale \f$s\f$ is shared across all rows (\f$D = e^s I\f$),
 * giving \f$q = 1 + q_R\f$ parameters; otherwise \f$q = p + q_R\f$.
 *
 * Products \f$e^{s_i + s_j}\f$ are evaluated as \f$e^{s_i} e^{s_j}\f$ (\f$e^{2s_i}\f$ on the diagonal) to avoid
 * magnifying the rounding error of \f$s_i + s_j\f$ by \f$|s|\f$.
 *
 * The underlying structure \f$R\f$ writes its quantity directly into the caller's output buffer, which this
 * composition rescales in place; caller scratch required is therefore exactly `scratch_size()` of \f$R\f$.
 *
 * \par Measured accuracy
 * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
 * - Compound symmetry (compound_symmetry_pdmat):
 *   \f$M\f$: 83.4 / 76.3 ulpN (\f$K=6\f$), \f$M^{-1}\f$: 2.8 / 2.8 ulpN, \f$L\f$: 5.1 / 4.8 ulpN, \f$\log\det M\f$: 3.4 / 3.1 ulp
 * - Autoregressive AR(1) (ar1_pdmat):
 *   \f$M\f$: 7.4 / 6.3 ulpN, \f$M^{-1}\f$: 1.1 / 0.73 ulpN, \f$L\f$: 5.2 / 5.8 ulpN, \f$\log\det M\f$: 5.3 / 6.1 ulp
 * - Unstructured correlation (unstructured_corr_pdmat):
 *   \f$M\f$: 40.2 / 17.3 ulpN (\f$K=6\f$), \f$M^{-1}\f$: 16250 / 27504 ulpN (\f$K=6\f$; 31 / 28 ulpN with scaling undone),
 *   \f$L\f$: 16.0 / 16.0 ulpN, \f$\log\det M\f$: 9.4 / 7.4 ulp, \f$\log\lvert\det J\rvert\f$: 14.1 / 11.1 ulp
 *
 * The 16250 / 27504 ulpN error on \f$M^{-1}\f$ with unstructured correlation and heterogeneous scales comes from
 * the normwise measure, not from a loss in the code: with scaling undone the error is at most 31 / 28 ulpN (the error of \f$R^{-1}\f$ itself),
 * magnified by \f$e^{-s_i - s_j}\f$ in the normwise measure. Reference values were computed from the mathematical
 * definition in mpmath at 640 bits (13906 values).
 *
 * \tparam R An underlying structure satisfying \ref pdmat_structure with \c unit_diagonal == \c true.
 *
 * \see pdmat, pdmat_orders, pdmat_structure
 */
template <pdmat_structure R>
  requires(R::unit_diagonal)
struct unit_diagonal_scaled_pdmat : pdmat_orders<unit_diagonal_scaled_pdmat<R>> {
  /// \brief Underlying structure with unit diagonal \f$R\f$.
  R unit_structure{};
  /// \brief Matrix dimension \f$p\f$, identical to that of \f$R\f$.
  int dim = 1;
  /// \brief Whether one scale \f$s\f$ is shared across all rows (\f$D = e^{s} I\f$) instead of one scale per row.
  bool common_scale = false;

  /// \brief Compile-time descriptor: \c false, diagonal entries are \f$e^{2s_i}\f$.
  static constexpr bool unit_diagonal = false;
  /// \brief Compile-time descriptor: indicates whether \f$R\f$ provides an exact closed-form inverse (\c R::closed_inverse).
  static constexpr bool closed_inverse = R::closed_inverse;
  /// \brief Compile-time descriptor: indicates whether \f$R\f$ provides an exact closed-form Cholesky factor (\c R::closed_factor).
  static constexpr bool closed_factor = R::closed_factor;
  /// \brief Compile-time descriptor: indicates whether \f$R\f$ provides a density and log-Jacobian (\c R::has_density; returns NaN with \c common_scale).
  static constexpr bool has_density = R::has_density;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  unit_diagonal_scaled_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs a scaled structure from an underlying unit-diagonal structure \p r.
   * \param r Underlying structure with unit diagonal.
   * \param common Whether to share a single scale parameter \f$s\f$ across all rows (\f$D = e^{s} I\f$).
   */
  LAPLACE_HD explicit unit_diagonal_scaled_pdmat(const R& r, bool common = false)
      : unit_structure(r), dim(r.dim), common_scale(common) {}

  /**
   * \ingroup pdmat
   * \brief Number of scale parameters: 1 if \c common_scale is \c true, otherwise \f$p\f$.
   * \return Number of scale parameters.
   */
  LAPLACE_HD int n_scales() const { return common_scale ? 1 : dim; }
  /**
   * \ingroup pdmat
   * \brief Total number of parameters \f$q\f$: the scale parameters plus \f$R\f$'s parameters.
   * \return Total parameter count (\f$1 + q_R\f$ with \c common_scale, \f$p + q_R\f$ otherwise).
   */
  LAPLACE_HD int n_params() const { return n_scales() + unit_structure.n_params(); }
  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch required: identical to \f$R\f$'s scratch size.
   *
   * The underlying structure \f$R\f$ writes its quantity into the output buffer,
   * which is then rescaled in place, requiring no additional scratch buffer.
   *
   * \return Number of scratch doubles required.
   */
  LAPLACE_HD int scratch_size() const { return unit_structure.scratch_size(); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed scaled positive definite matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ij} = e^{s_i + s_j} R_{ij}\f$ (with \f$s_i = s\f$ when \c common_scale is \c true).
   * Evaluates \f$e^{s_i + s_j}\f$ as \f$e^{s_i} e^{s_j}\f$ (\f$e^{2s_i}\f$ on the diagonal) to avoid
   * magnifying the rounding error of \f$s_i + s_j\f$ by \f$|s|\f$.
   *
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 83.4 / 76.3 ulpN (\f$K=6\f$) for compound symmetry, 7.4 / 6.3 ulpN for AR(1), 40.2 / 17.3 ulpN (\f$K=6\f$) for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const {
    unit_structure.matrix(theta + n_scales(), m, w);
    rescale(theta, m, 0, nullptr, 0);
  }
  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * Computes:
   * \f[
   * \log \det M(\theta) = 2 \sum_{i=0}^{p-1} s_i + \log \det R
   * \f]
   * where \f$\sum_{i=0}^{p-1} s_i = p s\f$ when \c common_scale is \c true.
   *
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 3.4 / 3.1 ulp for compound symmetry, 5.3 / 6.1 ulp for AR(1), 9.4 / 7.4 ulp for unstructured correlation; test bounds 1.5 times measured errors.
   *
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, double* w) const {
    return 2.0 * scale_sum(theta) + unit_structure.log_det(theta + n_scales(), w);
  }
  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Computes \f$M^{-1}_{ij} = e^{-s_i - s_j} R^{-1}_{ij}\f$ (with \f$s_i = s\f$ when \c common_scale is \c true).
   *
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 2.8 / 2.8 ulpN for compound symmetry, 1.1 / 0.73 ulpN for AR(1), 16250 / 27504 ulpN (\f$K=6\f$; 31 / 28 ulpN with scaling undone) for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, double* w) const {
    unit_structure.inverse(theta + n_scales(), m, w);
    rescale(theta, m, 1, nullptr, 0);
  }
  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Computes \f$L = D L_R\f$, with entries \f$L_{ij} = e^{s_i} (L_R)_{ij}\f$ (with \f$s_i = s\f$ when \c common_scale is \c true),
   * satisfying \f$M = L L^\top\f$.
   *
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 5.1 / 4.8 ulpN for compound symmetry, 5.2 / 5.8 ulpN for AR(1), 16.0 / 16.0 ulpN for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, double* w) const
    requires(R::closed_factor)
  {
    unit_structure.cholesky(theta + n_scales(), l, w);
    rescale(theta, l, 2, nullptr, 0);
  }
  /**
   * \ingroup pdmat
   * \brief Evaluates the log-Jacobian \f$\log\lvert\det J\rvert\f$ of the transformation \f$\theta \mapsto M\f$.
   *
   * The reference coordinates are the lower triangle of \f$M\f$ including the diagonal
   * (\f$R\f$'s reference coordinates being its strict lower triangle).
   * Computes:
   * \f[
   * \log\lvert\det J\rvert = p \log 2 + (p + 1) \sum_{i=0}^{p-1} s_i + \log\lvert J_R\rvert
   * \f]
   * Available only when \f$R\f$ has compile-time descriptor `has_density` set to \c true.
   * When \c common_scale is \c true, there are fewer parameters than entries in \f$M\f$ and no density exists, returning NaN.
   *
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Log-Jacobian determinant, or NaN if \c common_scale is \c true.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 14.1 / 11.1 ulp for unstructured correlation; test bounds 1.5 times measured errors.
   *
   * \see log_jacobian_dk()
   */
  LAPLACE_HD double log_jacobian(const double* theta, double* w) const
    requires(R::has_density)
  {
    if (common_scale) return laplace::math::detail::quiet_nan();
    return dim * 0.69314718055994530942 + (dim + 1) * scale_sum(theta) + unit_structure.log_jacobian(theta + n_scales(), w);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * A sorted tuple of parameter indices splits into a run \f$A\f$ of scale indices followed by indices \f$B\f$ of \f$R\f$.
   * Computes:
   * \f[
   * \partial M_{ij} = c_{ij}(A)\,e^{s_i + s_j}\,\partial^B R_{ij}
   * \f]
   * where \f$c_{ij}(A) = \prod_{k \in A} (\delta_{ik} + \delta_{jk})\f$ with Kronecker delta \f$\delta\f$
   * (or \f$2^{|A|}\f$ when \c common_scale is \c true).
   * If the tuple contains more than two distinct scale indices, or if \f$R\f$'s derivative along \f$B\f$
   * is structurally zero, returns \c false leaving buffer \p m untouched.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 83.4 / 76.3 ulpN (\f$K=6\f$) for compound symmetry, 7.4 / 6.3 ulpN for AR(1), 40.2 / 17.3 ulpN (\f$K=6\f$) for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    return matrix_part<K>(detail_pdmat::scaled_part::matrix, 0, theta, idx, m, w);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * A sorted tuple of parameter indices splits into a run \f$A\f$ of scale indices followed by indices \f$B\f$ of \f$R\f$.
   * Computes:
   * \f[
   * \partial M^{-1}_{ij} = (-1)^{|A|} c_{ij}(A)\,e^{-s_i - s_j}\,\partial^B R^{-1}_{ij}
   * \f]
   * where \f$c_{ij}(A) = \prod_{k \in A} (\delta_{ik} + \delta_{jk})\f$ with Kronecker delta \f$\delta\f$
   * (or \f$(-2)^{|A|}\f$ including sign when \c common_scale is \c true).
   * If the tuple contains more than two distinct scale indices, or if \f$R\f$'s derivative along \f$B\f$
   * is structurally zero, returns \c false leaving buffer \p m untouched.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 2.8 / 2.8 ulpN for compound symmetry, 1.1 / 0.73 ulpN for AR(1), 16250 / 27504 ulpN (\f$K=6\f$; 31 / 28 ulpN with scaling undone) for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, const int (&idx)[K], double* m, double* w) const {
    return matrix_part<K>(detail_pdmat::scaled_part::inverse, 1, theta, idx, m, w);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$.
   *
   * A sorted tuple of parameter indices splits into a run \f$A\f$ of scale indices followed by indices \f$B\f$ of \f$R\f$.
   * Computes:
   * \f[
   * \partial L_{ij} = \left(\prod_{k \in A} \delta_{ik}\right) e^{s_i}\,\partial^B (L_R)_{ij}
   * \f]
   * with Kronecker delta \f$\delta\f$ (the scale coefficient is 1 when \c common_scale is \c true).
   * If the tuple contains more than one distinct scale index, or if \f$R\f$'s derivative along \f$B\f$
   * is structurally zero, returns \c false leaving buffer \p l untouched.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 5.1 / 4.8 ulpN for compound symmetry, 5.2 / 5.8 ulpN for AR(1), 16.0 / 16.0 ulpN for unstructured correlation; test bounds 1.5 times the errors with the scaling undone.
   *
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, const int (&idx)[K], double* l, double* w) const
    requires(R::closed_factor)
  {
    return matrix_part<K>(detail_pdmat::scaled_part::cholesky, 2, theta, idx, l, w);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log\det M(\theta)\f$ of order \f$K\f$.
   *
   * A sorted tuple of parameter indices splits into a run \f$A\f$ of scale indices followed by indices \f$B\f$ of \f$R\f$.
   * Evaluates to 2 (or \f$2p\f$ when \c common_scale is \c true) along a single scale index (\f$K=1\f$, \f$|A|=1\f$),
   * to \f$R\f$'s derivative along \f$B\f$ when \f$|A|=0\f$, and to \f$0.0\f$ otherwise.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 3.4 / 3.1 ulp for compound symmetry, 5.3 / 6.1 ulp for AR(1), 9.4 / 7.4 ulp for unstructured correlation; test bounds 1.5 times measured errors.
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], double* w) const {
    return scalar_part<K>(detail_pdmat::scaled_part::log_det, common_scale ? 2.0 * dim : 2.0, theta, idx, w);
  }
  /**
   * \ingroup pdmat
   * \brief Partial derivative of the log-Jacobian of order \f$K\f$.
   *
   * A sorted tuple of parameter indices splits into a run \f$A\f$ of scale indices followed by indices \f$B\f$ of \f$R\f$.
   * Evaluates to \f$p + 1\f$ along a single scale index (\f$K=1\f$, \f$|A|=1\f$),
   * to \f$R\f$'s derivative along \f$B\f$ when \f$|A|=0\f$, and to \f$0.0\f$ otherwise.
   * Returns NaN when \c common_scale is \c true.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of parameters: scale parameters followed by the parameters of \f$R\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, \f$0.0\f$ if identically zero, or NaN if \c common_scale is \c true.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6 on gcc/OpenMP / CUDA (RTX 4070 SUPER):
   * 14.1 / 11.1 ulp for unstructured correlation; test bounds 1.5 times measured errors.
   *
   * \see log_jacobian(), pdmat_orders::log_jacobian_d1()
   */
  template <int K>
  LAPLACE_HD double log_jacobian_dk(const double* theta, const int (&idx)[K], double* w) const
    requires(R::has_density)
  {
    if (common_scale) return laplace::math::detail::quiet_nan();
    return scalar_part<K>(detail_pdmat::scaled_part::log_jacobian, dim + 1.0, theta, idx, w);
  }

 private:
  LAPLACE_HD double scale_of(const double* theta, int i) const { return theta[common_scale ? 0 : i]; }
  LAPLACE_HD double scale_sum(const double* theta) const {
    if (common_scale) return dim * theta[0];
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += theta[i];
    return s;
  }

  // the tuple as a = |A| scale indices, then the indices of R shifted to 0
  template <int K>
  LAPLACE_HD int split(const int (&idx)[K], int* sub) const {
    const int ns = n_scales();
    int a = 0;
    while (a < K && idx[a] < ns) ++a;
    for (int k = a; k < K; ++k) sub[k - a] = idx[k] - ns;
    return a;
  }

  // the scalar quantities: R's along R alone, d1 along one scale, 0 otherwise
  template <int K>
  LAPLACE_HD double scalar_part(detail_pdmat::scaled_part q, double d1, const double* theta, const int (&idx)[K],
                                double* w) const {
    int sub[K];
    const int a = split(idx, sub);
    if (a == 0) return detail_pdmat::scaled_part_runtime(unit_structure, q, K, theta + n_scales(), sub, nullptr, w);
    return K == 1 ? d1 : 0.0;
  }

  // kind 0: M, 1: M^-1, 2: L
  template <int K>
  LAPLACE_HD bool matrix_part(detail_pdmat::scaled_part q, int kind, const double* theta, const int (&idx)[K], double* m,
                              double* w) const {
    int sub[K];
    const int a = split(idx, sub);
    if (!common_scale && a > 0) {  // zero by structure: more than 2 scales (1 for L)
      int distinct = 1;
      for (int k = 1; k < a; ++k) distinct += idx[k] != idx[k - 1];
      if (distinct > (kind == 2 ? 1 : 2)) return false;
    }
    if (detail_pdmat::scaled_part_runtime(unit_structure, q, K - a, theta + n_scales(), sub, m, w) == 0.0) return false;
    rescale(theta, m, kind, idx, a);
    return true;
  }

  // m_ij *= c_ij(A) e^(s_i + s_j) (kind 0), (-1)^a c_ij(A) e^(-s_i - s_j)
  // (kind 1), prod_(k in A) delta_ik e^(s_i) (kind 2); A = idx[0..a-1]
  LAPLACE_HD void rescale(const double* theta, double* m, int kind, const int* A, int a) const {
    namespace dp = detail_pdmat;
    const double sg = kind == 1 ? -1.0 : 1.0;
    double cc = 1.0;  // the coefficient with common_scale
    if (common_scale && kind != 2) cc = dp::ipow(2.0 * sg, a);
    const bool odd = a % 2 == 1;
    for (int i = 0; i < dim; ++i) {
      const double si = scale_of(theta, i), ei = dp::m::exp(sg * si);
      for (int j = 0; j <= i; ++j) {
        double c = cc;
        if (!common_scale && a > 0) {
          bool in = true;  // every index of A is i (or j, for M and M^-1)
          for (int k = 0; k < a; ++k) in = in && (A[k] == i || (kind != 2 && A[k] == j));
          if (!in) c = 0.0;
          else if (kind != 2 && i == j) c = dp::ipow(2.0 * sg, a);
          else c = kind == 1 && odd ? -1.0 : 1.0;
        }
        double e;
        if (kind == 2) e = ei;
        else if (i == j) e = dp::m::exp(2.0 * sg * si);
        else e = ei * dp::m::exp(sg * scale_of(theta, j));
        m[packed_index(i, j)] = c == 0.0 ? 0.0 : c * e * m[packed_index(i, j)];
      }
    }
  }
};

}  // namespace laplace::pdmat
