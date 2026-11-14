#pragma once

// The status and the options of laplace::dense (dense.hpp). This header
// includes neither Kokkos nor Eigen: the Eigen backend includes it as well.

namespace laplace::dense {

/**
 * \ingroup dense
 * \brief Failure reasons reported when a decomposition did not produce usable outputs.
 *
 * Values:
 * - `none`: the decomposition succeeded.
 * - `not_positive_definite`: input matrix \f$A\f$, or \f$B\f$ in the generalized eigenproblem, is not positive definite.
 * - `singular`: a diagonal entry of a triangular matrix is zero during a triangular solve.
 * - `no_convergence`: an iterative SVD or eigensolver did not converge.
 * - `non_finite_input`: a NaN or an infinity was encountered among the entries read.
 *
 * A low numerical rank is not considered a failure and is reported in the rank field of dense_decomposition_status.
 *
 * \see dense_decomposition_status
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
 * Fields:
 * - `success`: \c true if the decomposition succeeded and outputs are valid, \c false otherwise.
 * - `rank`: numerical rank for pivoted_qr_decomposition(), svd_decomposition(), and null_space_basis(); matrix dimension \f$n\f$ otherwise.
 * - `reason`: reason code of type dense_failure when `success` is \c false (`dense_failure::none` on success).
 *
 * Numerical outcomes are reported here and are never thrown as exceptions.
 * When `success` is \c false, output Views are left in an unspecified state.
 *
 * \see dense_failure
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
 * \brief Which triangle of a matrix holds the triangular factor.
 *
 * Values:
 * - `lower`: lower triangle (entries above the main diagonal are not read).
 * - `upper`: upper triangle (entries below the main diagonal are not read).
 *
 * \see triangular_solve()
 */
// Which triangle of a matrix holds the triangular factor.
enum class dense_triangle { lower, upper };

/**
 * \ingroup dense
 * \brief Whether the triangular factor or its transpose is applied in a triangular solve.
 *
 * Values:
 * - `identity`: solve \f$T X = B\f$.
 * - `transpose`: solve \f$T^\top X = B\f$.
 *
 * \see triangular_solve()
 */
// Whether the triangular factor or its transpose is applied.
enum class dense_operation { identity, transpose };

}  // namespace laplace::dense
