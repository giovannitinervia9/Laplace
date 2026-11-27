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
// lives in src/sparse/eigen_sparse_backend.cpp, the supernodal route in
// src/sparse/supernodal_ldlt.cpp with its dense kernels in src/sparse/*_dense_kernels.cpp,
// all compiled as C++ and never by nvcc, so this header holds no Eigen type.
//
// positive_definite means that every pivot of D is positive and finite. After an outcome other than
// positive_definite, the pivots and log |det| are not available on either route (the simplicial route
// leaves those of its factor, the supernodal route NaN); factor_nonzeros() stays defined.

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
 * \brief The algorithm that sparse_ldlt runs on a pattern.
 *
 * `simplicial` is Eigen's column-by-column \f$L D L^\top\f$. `supernodal` factorizes the leading columns
 * as sparse \f$L D L^\top\f$ and the trailing dense supernode of the factor through dense kernels.
 * `automatic` chooses `supernodal` when the trailing dense supernode has at least 64 columns and
 * `simplicial` otherwise.
 *
 * \see sparse_ldlt, \ref sparse
 */
enum class sparse_ldlt_route : int {
  automatic,   ///< Chosen from the symbolic structure of the factor.
  simplicial,  ///< Eigen's simplicial \f$L D L^\top\f$.
  supernodal,  ///< Sparse leading columns and a dense trailing supernode.
};

/**
 * \ingroup sparse
 * \brief Host-only sparse \f$L D L^\top\f$ factorization of a symmetric matrix given by its lower triangle.
 *
 * Compiled inside `laplace_sparse` (`Laplace::sparse`) so no Eigen or LAPACK types appear in the header,
 * it orders the pattern once by approximate minimum degree (AMD) and reuses the symbolic analysis until
 * the pattern or requested route changes. It runs either `simplicial` (Eigen's `SimplicialLDLT`) or
 * `supernodal` (an up-looking sparse \f$L D L^\top\f$ on the leading columns and a dense Cholesky
 * factorization on the trailing dense supernode, parallelized with OpenMP). Under `automatic`, the
 * default, factorize() takes `supernodal` when the trailing dense supernode has at least 64 columns and
 * `simplicial` otherwise.
 *
 * \see sparse_ldlt_outcome, sparse_ldlt_route, \ref sparse
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
   * \brief Sets the route requested for the next analysis; a change forces the pattern to be analyzed again.
   * \param route The requested route (`automatic` by default).
   */
  void set_route(sparse_ldlt_route route);
  /**
   * \ingroup sparse
   * \brief The route of the last analysis: `simplicial` or `supernodal` (`automatic` before the first analysis or after set_route() changes the requested route).
   * \return The route in use.
   */
  sparse_ldlt_route route() const;
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
  /**
   * \ingroup sparse
   * \brief Returns the number of stored entries of the factor \f$L\f$ below its unit diagonal, from the last symbolic analysis.
   * \return The count (\f$0\f$ before the first factorization).
   */
  long factor_nonzeros() const;
  /**
   * \ingroup sparse
   * \brief Computes the selected inverse: the entries of \f$Z = (A + \lambda I)^{-1}\f$ on the pattern of \f$L + L^\top\f$.
   *
   * Takahashi's recursion on the last factorization \f$P (A + \lambda I) P^\top = L D L^\top\f$, column by
   * column from the last: with \f$J_j\f$ the rows of the entries of column \f$j\f$ of \f$L\f$ below the diagonal,
   * \f$Z_{ij} = -\sum_{k \in J_j} Z_{ik} L_{kj}\f$ for \f$i \in J_j\f$ and
   * \f$Z_{jj} = 1/D_j - \sum_{k \in J_j} L_{kj} Z_{kj}\f$ (in the permuted ordering). Every \f$Z_{ik}\f$ it reads
   * lies on the pattern, since the rows of \f$J_j\f$ are pairwise joined in the pattern of \f$L\f$. The time is of
   * the order of one numeric factorization and the memory that of \f$L\f$. A later factorize() discards it.
   * \return `true` if a factorization was stored and the entries were computed; `false` otherwise.
   */
  bool compute_selected_inverse();
  /**
   * \ingroup sparse
   * \brief The entry \f$Z_{ij}\f$ of the selected inverse, in the original ordering of the matrix.
   *
   * Every pair \f$(i, j)\f$ at which \f$A\f$ has a stored entry (or its transpose) is on the pattern.
   * Throws `std::out_of_range` when \f$(i, j)\f$ is outside the pattern of \f$L + L^\top\f$ or when
   * compute_selected_inverse() has not run since the last factorization.
   * \param i Row, in \f$[0, n)\f$.
   * \param j Column, in \f$[0, n)\f$.
   * \return \f$Z_{ij}\f$.
   */
  double selected_inverse_entry(int i, int j) const;
  /**
   * \ingroup sparse
   * \brief Computes the derivative of the selected inverse along a direction \f$\dot A\f$ on the pattern of \f$A\f$.
   *
   * \f$\dot A\f$ is symmetric and given by its lower triangle on the pattern of the last factorized matrix
   * (`dvalues` aligned with its `col_ptr` and `row_idx`). The derivative of the factorization along
   * \f$\dot A\f$, \f$\dot L\f$ and \f$\dot D\f$ on the pattern of \f$L\f$, follows from
   * \f$\dot L_{ij} D_j + L_{ij} \dot D_j = \dot A_{ij} - \sum_{k<j} (\dot L_{ik} L_{jk} D_k + L_{ik} \dot L_{jk} D_k
   * + L_{ik} L_{jk} \dot D_k)\f$ for \f$i \ge j\f$ (with \f$L_{jj} = 1\f$, \f$\dot L_{jj} = 0\f$), and the derivative
   * of Takahashi's recursion gives \f$\dot Z = -Z \dot A Z\f$ on the pattern of \f$L + L^\top\f$:
   * \f$\dot Z_{ij} = -\sum_{k \in J_j} (\dot Z_{ik} L_{kj} + Z_{ik} \dot L_{kj})\f$ and
   * \f$\dot Z_{jj} = -\dot D_j / D_j^2 - \sum_{k \in J_j} (\dot L_{kj} Z_{kj} + L_{kj} \dot Z_{kj})\f$ (permuted ordering).
   * The time is of the order of one numeric factorization and one selected inverse; the memory that of
   * \f$L\f$ three times. Needs compute_selected_inverse(); a later factorize() or another call replaces it.
   * \param dvalues Array of `col_ptr[n]` lower-triangular entries of \f$\dot A\f$.
   * \return `true` if the selected inverse was available and the derivative was computed; `false` otherwise.
   */
  bool compute_selected_inverse_derivative(const double* dvalues);
  /**
   * \ingroup sparse
   * \brief The entry \f$\dot Z_{ij}\f$ of the last derivative of the selected inverse, in the original ordering.
   *
   * Throws `std::out_of_range` outside the pattern of \f$L + L^\top\f$ or when no derivative is stored.
   * \param i Row, in \f$[0, n)\f$.
   * \param j Column, in \f$[0, n)\f$.
   * \return \f$\dot Z_{ij}\f$.
   */
  double selected_inverse_derivative_entry(int i, int j) const;
  /**
   * \ingroup sparse
   * \brief The sum \f$\sum_{i,j} \dot Z_{ij} V_{ij}\f$ for a symmetric \f$V\f$ on the pattern of \f$A\f$.
   *
   * \f$V\f$ is given by its lower triangle, aligned like `dvalues`; each entry below the diagonal counts
   * twice. With \f$\dot Z = -Z \dot A Z\f$ this is \f$-\operatorname{tr}(Z \dot A Z V)\f$.
   * Throws `std::out_of_range` when no derivative is stored.
   * \param values Array of `col_ptr[n]` lower-triangular entries of \f$V\f$.
   * \return The sum.
   */
  double selected_inverse_derivative_dot(const double* values) const;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace laplace::sparse
