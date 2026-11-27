#pragma once

// The dense kernels of the supernodal route of laplace::sparse::sparse_ldlt: the operations on a dense
// block of the factor, column-major with a leading dimension. One realization is compiled into
// laplace_sparse, chosen at configure time by LAPLACE_DENSE_KERNELS: the environment's LAPACK and BLAS
// (lapack_dense_kernels.cpp) or kernels of our own on Eigen (eigen_dense_kernels.cpp). Internal: no
// public header includes this file.

namespace laplace::sparse::detail {

// A = R R' with R lower triangular, in place on the lower triangle of a (LAPACK dpotrf, 'L'). Returns 0,
// or k > 0 when the leading minor of order k is not positive definite (the factor is then incomplete).
int dense_cholesky_lower(int m, double* a, int lda);

// From the factor R of dense_cholesky_lower, A^-1 in place on the lower triangle of a (LAPACK dpotri,
// 'L'). Returns 0, or k > 0 when R_kk is zero.
int dense_cholesky_inverse_lower(int m, double* a, int lda);

// Solves A X = B in place for nrhs right-hand sides, from the factor R in the lower triangle of a (LAPACK
// dpotrs, 'L').
void dense_cholesky_solve_lower(int m, int nrhs, const double* a, int lda, double* b, int ldb);

// C = B Z for a general m x m block B and a symmetric Z given by its lower triangle (BLAS dsymm, 'R',
// 'L').
void dense_symmetric_product_right(int m, const double* z, int ldz, const double* b, int ldb, double* c, int ldc);

// The name of the compiled realization, for reports.
const char* dense_kernels_realization();

}  // namespace laplace::sparse::detail
