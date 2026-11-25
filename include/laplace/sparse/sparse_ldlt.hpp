#pragma once

// A sparse symmetric LDL' factorization on the host (laplace::sparse), for the
// curvature of an objective whose Hessian is sparse (sparse_curvature_solver in
// optimizers/sparse_curvature_solver.hpp). The matrix is given by its lower
// triangle, diagonal included, in compressed sparse columns:
//
//   col_ptr  n + 1 offsets into row_idx and values
//   row_idx  the row of each stored entry, row >= column, sorted within a column
//   values   the entries
//
// The pattern is analyzed (fill-reducing ordering, symbolic factorization)
// the first time and again only when it changes; factorize() reads the values.
// The Eigen code (SimplicialLDLT with its approximate minimum degree ordering)
// lives in src/sparse/eigen_sparse_backend.cpp, compiled as C++ and never by
// nvcc, so this header holds no Eigen type.
//
// positive_definite means that every pivot of D is positive and finite.

#include <memory>

namespace laplace::sparse {

/**
 * \ingroup sparse
 * \brief Outcome of sparse_ldlt::factorize().
 *
 * \see sparse_ldlt, \ref sparse
 */
enum class sparse_ldlt_outcome : int {
  positive_definite,      ///< Every pivot of \f$D\f$ is positive and finite.
  not_positive_definite,  ///< The factorization failed or some pivot is zero or negative.
  non_finite,             ///< An entry of the matrix or of the factor \f$D\f$ is not finite.
};

/**
 * \ingroup sparse
 * \brief Host-only sparse \f$L D L^\top\f$ factorization of a symmetric matrix given by its lower triangle.
 *
 * Wraps Eigen's `SimplicialLDLT` with approximate minimum degree (AMD) ordering inside the compiled
 * static library `laplace_sparse` (`Laplace::sparse`), keeping Eigen types out of the header.
 * Analyzes the sparsity pattern once and reuses the symbolic factorization until the pattern changes.
 *
 * \see sparse_ldlt_outcome, \ref sparse
 */
class sparse_ldlt {
 public:
  /**
   * \ingroup sparse
   * \brief Constructs an uninitialized sparse factorization object.
   */
  sparse_ldlt();
  /**
   * \ingroup sparse
   * \brief Destroys the factorization object and its internal Eigen state.
   */
  ~sparse_ldlt();
  /**
   * \ingroup sparse
   * \brief Move constructor.
   */
  sparse_ldlt(sparse_ldlt&&) noexcept;
  /**
   * \ingroup sparse
   * \brief Move assignment operator.
   */
  sparse_ldlt& operator=(sparse_ldlt&&) noexcept;
  sparse_ldlt(const sparse_ldlt&) = delete;
  sparse_ldlt& operator=(const sparse_ldlt&) = delete;

  /**
   * \ingroup sparse
   * \brief Factorizes \f$A + \lambda I\f$ with \f$A\f$ stored in compressed sparse columns (lower triangle).
   *
   * Checks that all `col_ptr[n]` entries of `values` are finite, re-runs AMD symbolic analysis only if
   * `n`, `col_ptr`, or `row_idx` changed since the previous call, and computes the numerical \f$L D L^\top\f$
   * factorization. When `shift != 0.0`, the diagonal entry of each column \f$c\f$ must be stored first
   * at `col_ptr[c]`.
   *
   * \param n Matrix dimension (\f$n \ge 0\f$).
   * \param col_ptr Array of \f$n + 1\f$ column offsets into `row_idx` and `values`.
   * \param row_idx Array of `col_ptr[n]` row indices (`row >= column`), sorted within each column.
   * \param values Array of `col_ptr[n]` lower-triangular numerical entries.
   * \param shift Diagonal shift \f$\lambda\f$ added to each diagonal entry before factorization (default `0.0`).
   * \return Outcome indicating whether all pivots of \f$D\f$ are positive and finite.
   */
  sparse_ldlt_outcome factorize(int n, const int* col_ptr, const int* row_idx, const double* values,
                                double shift = 0.0);
  /**
   * \ingroup sparse
   * \brief Solves \f$(A + \lambda I) z = b\f$ in place using the last factorization.
   *
   * \param b Host array of length \f$n\f$ holding the right-hand side on entry and overwritten with \f$z\f$ on success.
   * \return `true` if a factorization was stored and the solve ran; `false` (leaving `b` untouched) otherwise.
   */
  bool solve(double* b) const;
  /**
   * \ingroup sparse
   * \brief Returns the smallest diagonal pivot of \f$D\f$ from the last successful factorization.
   * \return Minimum pivot (\f$0.0\f$ if \f$n = 0\f$ or no factorization is stored).
   */
  double min_pivot() const;
  /**
   * \ingroup sparse
   * \brief Returns the largest diagonal pivot of \f$D\f$ from the last successful factorization.
   * \return Maximum pivot (\f$0.0\f$ if \f$n = 0\f$ or no factorization is stored).
   */
  double max_pivot() const;
  /**
   * \ingroup sparse
   * \brief Returns \f$\log|\det(A + \lambda I)|\f$, the sum of \f$\log|D_{ii}|\f$, from the last successful factorization.
   * \return The logarithm of the absolute determinant (\f$0.0\f$ if \f$n = 0\f$ or no factorization is stored).
   */
  double log_abs_determinant() const;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace laplace::sparse
