#pragma once

// The status and the options of laplace::dense (dense.hpp). This header
// includes neither Kokkos nor Eigen: the Eigen backend includes it as well.

namespace laplace::dense {

/**
 * \ingroup dense
 * \brief Failure reasons reported when a decomposition does not produce usable outputs.
 *
 * Low numerical rank is not a failure and is reported in dense_decomposition_status::rank.
 * See \ref dense for the conditions that produce each code.
 *
 * \see \ref dense, dense_decomposition_status
 */
// Why a decomposition did not produce usable outputs. A low rank is not a
// failure: the pivoted QR, the SVD and the null space report it in the rank.
enum class dense_failure {
  none,                   // success
  not_positive_definite,  // Cholesky of A, or of B in the generalized problem
  singular,               // a zero on the diagonal of a triangular matrix
  no_convergence,         // an iterative SVD or eigensolver did not converge
  non_finite_input        // a NaN or an infinity among the entries read
};

/**
 * \ingroup dense
 * \brief Status returned by every function of \ref dense.
 *
 * Holds whether the computation succeeded (`success`), the numerical rank or matrix dimension \f$n\f$ (`rank`),
 * and the failure code (`reason`, `dense_failure::none` on success).
 * Numerical outcomes are reported here rather than thrown, and output Views are unspecified when `success` is \c false.
 *
 * \see \ref dense, dense_failure
 */
// Returned by every function of laplace::dense; numerical outcomes are reported
// here, never by exceptions. rank is the numerical rank for the pivoted QR,
// the SVD and the null space (of C), and the dimension n otherwise.
struct dense_decomposition_status {
  bool success = true;
  int rank = 0;
  dense_failure reason = dense_failure::none;
};

/**
 * \ingroup dense
 * \brief Selects the lower or upper triangle of a matrix.
 *
 * Only entries in the chosen triangle are read by triangular_solve().
 *
 * \see \ref dense, triangular_solve()
 */
// Which triangle of a matrix holds the triangular factor.
enum class dense_triangle { lower, upper };

/**
 * \ingroup dense
 * \brief Selects whether triangular_solve() applies \f$T\f$ (`identity`) or \f$T^\top\f$ (`transpose`).
 *
 * \see \ref dense, triangular_solve()
 */
// Whether the triangular factor or its transpose is applied.
enum class dense_operation { identity, transpose };

}  // namespace laplace::dense
