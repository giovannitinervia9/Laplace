// The dense kernels of the supernodal route on Eigen (MPL2), used when no LAPACK is linked. Compiled as
// C++, never by nvcc, and without EIGEN_DONT_PARALLELIZE, so that Eigen's products use OpenMP.
// ponytail: unblocked Eigen LLT and a triangular solve against the identity; the blocked kernels measured
// in note/engine/step2b_factorization.md (section 3) replace them once x86-64-v3 or runtime dispatch is chosen.
#include "detail/dense_kernels.hpp"

#include <Eigen/Dense>

#include <cmath>

namespace laplace::sparse::detail {

namespace {
using block_map = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>;
using const_block_map = Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>;
}  // namespace

int dense_cholesky_lower(int m, double* a, int lda) {
  if (m <= 0) return 0;
  block_map A(a, m, m, Eigen::OuterStride<>(lda));
  // column by column in place (the LAPACK convention: on failure at column k, return k + 1)
  for (int k = 0; k < m; ++k) {
    double d = A(k, k) - A.row(k).head(k).squaredNorm();
    if (!(d > 0.0)) return k + 1;
    d = std::sqrt(d);
    A(k, k) = d;
    const int r = m - k - 1;
    if (r > 0) {
      A.col(k).tail(r).noalias() -= A.bottomLeftCorner(r, k) * A.row(k).head(k).transpose();
      A.col(k).tail(r) /= d;
    }
  }
  return 0;
}

int dense_cholesky_inverse_lower(int m, double* a, int lda) {
  if (m <= 0) return 0;
  block_map A(a, m, m, Eigen::OuterStride<>(lda));
  for (int k = 0; k < m; ++k)
    if (A(k, k) == 0.0) return k + 1;
  Eigen::MatrixXd Li = Eigen::MatrixXd::Identity(m, m);
  A.triangularView<Eigen::Lower>().solveInPlace(Li);
  Eigen::MatrixXd Z(m, m);
  Z.noalias() = Li.transpose() * Li;
  A.triangularView<Eigen::Lower>() = Z;
  return 0;
}

void dense_cholesky_solve_lower(int m, int nrhs, const double* a, int lda, double* b, int ldb) {
  if (m <= 0 || nrhs <= 0) return;
  const_block_map A(a, m, m, Eigen::OuterStride<>(lda));
  block_map B(b, m, nrhs, Eigen::OuterStride<>(ldb));
  A.triangularView<Eigen::Lower>().solveInPlace(B);
  A.transpose().triangularView<Eigen::Upper>().solveInPlace(B);
}

void dense_symmetric_product_right(int m, const double* z, int ldz, const double* b, int ldb, double* c, int ldc) {
  if (m <= 0) return;
  const_block_map Z(z, m, m, Eigen::OuterStride<>(ldz)), B(b, m, m, Eigen::OuterStride<>(ldb));
  block_map C(c, m, m, Eigen::OuterStride<>(ldc));
  C.noalias() = B * Z.selfadjointView<Eigen::Lower>();
}

const char* dense_kernels_realization() { return "eigen"; }

}  // namespace laplace::sparse::detail
