#pragma once

// The boundary between the dense templates of dense.hpp, which take Kokkos
// Views, and the Eigen backend in src/dense/eigen_backend.cpp. This header
// includes neither Kokkos nor Eigen, so that the backend is compiled as plain
// C++ in every build, the CUDA build included, and nvcc never sees Eigen.

#include "laplace/dense/dense_status.hpp"

namespace laplace::dense {
namespace detail {

// Column-major matrices: entry (i, j) at data[i + j * ld], ld >= rows.
struct const_matrix_ref {
  const double* data;
  int rows, cols, ld;
};
struct matrix_ref {
  double* data;
  int rows, cols, ld;
};
// Vectors: entry i at data[i * inc].
struct vector_ref {
  double* data;
  int size, inc;
};
struct int_vector_ref {
  int* data;
  int size, inc;
};

// Wrong dimensions throw std::invalid_argument; the rest is in the status.
dense_decomposition_status cholesky(const_matrix_ref A, matrix_ref L);
dense_decomposition_status triangular_solve(const_matrix_ref T, matrix_ref B,
                                            dense_triangle triangle, dense_operation operation);
dense_decomposition_status pivoted_qr(const_matrix_ref A, matrix_ref Q, matrix_ref R,
                                      int_vector_ref perm, double relative_tolerance);
dense_decomposition_status svd(const_matrix_ref A, matrix_ref U, vector_ref s, matrix_ref V,
                               double relative_tolerance);
dense_decomposition_status symmetric_eigen(const_matrix_ref A, vector_ref w, matrix_ref V);
dense_decomposition_status generalized_symmetric_eigen(const_matrix_ref A, const_matrix_ref B,
                                                       vector_ref w, matrix_ref V);
dense_decomposition_status cholesky_solve(const_matrix_ref L, matrix_ref B);
// The 1-norm condition of A from its Cholesky factor L; inf when the factor
// has a zero or non-finite diagonal entry, NaN when A is not finite.
double cholesky_condition_estimate(const_matrix_ref A, const_matrix_ref L);
// V (n x n, column-major, ld n) receives the right singular vectors of C; the
// status gives the rank r of C, and columns r..n-1 of V span its null space.
dense_decomposition_status null_space(const_matrix_ref C, double* V, double relative_tolerance);

}  // namespace detail
}  // namespace laplace::dense
