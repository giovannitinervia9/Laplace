// The Eigen code of laplace::sparse (include/laplace/sparse/sparse_ldlt.hpp):
// SimplicialLDLT with the approximate minimum degree ordering (MPL2, adapted
// from CSparse). Compiled as C++, never by nvcc.
#include <laplace/sparse/sparse_ldlt.hpp>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <vector>

namespace laplace::sparse {

using sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct sparse_ldlt::impl {
  Eigen::SimplicialLDLT<sparse_matrix, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
  sparse_matrix A;
  std::vector<int> col_ptr, row_idx;  // the pattern last analyzed
  bool analyzed = false, factorized = false;
  double min_pivot = 0.0, max_pivot = 0.0, log_abs_det = 0.0;
};

sparse_ldlt::sparse_ldlt() : impl_(std::make_unique<impl>()) {}
sparse_ldlt::~sparse_ldlt() = default;
sparse_ldlt::sparse_ldlt(sparse_ldlt&&) noexcept = default;
sparse_ldlt& sparse_ldlt::operator=(sparse_ldlt&&) noexcept = default;

sparse_ldlt_outcome sparse_ldlt::factorize(int n, const int* col_ptr, const int* row_idx, const double* values,
                                           double shift) {
  impl& m = *impl_;
  m.factorized = false;
  const int nnz = col_ptr[n];
  for (int k = 0; k < nnz; ++k)
    if (!std::isfinite(values[k])) return sparse_ldlt_outcome::non_finite;
  const bool same_pattern = m.analyzed && static_cast<int>(m.col_ptr.size()) == n + 1 &&
                            std::equal(col_ptr, col_ptr + n + 1, m.col_ptr.begin()) &&
                            std::equal(row_idx, row_idx + nnz, m.row_idx.begin());
  if (!same_pattern) {
    m.col_ptr.assign(col_ptr, col_ptr + n + 1);
    m.row_idx.assign(row_idx, row_idx + nnz);
    m.A = sparse_matrix(n, n);
    m.A.resizeNonZeros(nnz);
    std::copy(col_ptr, col_ptr + n + 1, m.A.outerIndexPtr());
    std::copy(row_idx, row_idx + nnz, m.A.innerIndexPtr());
  }
  std::copy(values, values + nnz, m.A.valuePtr());
  if (shift != 0.0)
    for (int c = 0; c < n; ++c) m.A.valuePtr()[col_ptr[c]] += shift;  // the diagonal is first in its column
  if (!same_pattern) {
    m.ldlt.analyzePattern(m.A);
    m.analyzed = true;
  }
  m.ldlt.factorize(m.A);
  if (m.ldlt.info() != Eigen::Success) return sparse_ldlt_outcome::not_positive_definite;
  const auto D = m.ldlt.vectorD();
  if (!D.allFinite()) return sparse_ldlt_outcome::non_finite;
  m.min_pivot = n > 0 ? D.minCoeff() : 0.0;
  m.max_pivot = n > 0 ? D.maxCoeff() : 0.0;
  m.log_abs_det = n > 0 ? D.array().abs().log().sum() : 0.0;
  m.factorized = true;
  return m.min_pivot > 0.0 ? sparse_ldlt_outcome::positive_definite : sparse_ldlt_outcome::not_positive_definite;
}

bool sparse_ldlt::solve(double* b) const {
  if (!impl_->factorized) return false;
  const int n = static_cast<int>(impl_->A.rows());
  Eigen::Map<Eigen::VectorXd> v(b, n);
  v = impl_->ldlt.solve(Eigen::VectorXd(v));
  return true;
}

double sparse_ldlt::min_pivot() const { return impl_->min_pivot; }
double sparse_ldlt::max_pivot() const { return impl_->max_pivot; }
double sparse_ldlt::log_abs_determinant() const { return impl_->log_abs_det; }

}  // namespace laplace::sparse
