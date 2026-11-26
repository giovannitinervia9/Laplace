// The Eigen code of laplace::sparse (include/laplace/sparse/sparse_ldlt.hpp):
// SimplicialLDLT with the approximate minimum degree ordering (MPL2, adapted
// from CSparse). Compiled as C++, never by nvcc.
#include <laplace/sparse/sparse_ldlt.hpp>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace laplace::sparse {

using sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct sparse_ldlt::impl {
  Eigen::SimplicialLDLT<sparse_matrix, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
  sparse_matrix A;
  std::vector<int> col_ptr, row_idx;  // the pattern last analyzed
  bool analyzed = false, factorized = false, have_selected = false;
  std::vector<double> z_lower, z_diag;  // the selected inverse: z_lower on the pattern of L, permuted ordering
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
  m.have_selected = false;
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
long sparse_ldlt::factor_nonzeros() const {
  return impl_->analyzed ? static_cast<long>(impl_->ldlt.matrixL().nestedExpression().nonZeros()) : 0;
}

bool sparse_ldlt::compute_selected_inverse() {
  impl& m = *impl_;
  if (!m.factorized) return false;
  const auto& L = m.ldlt.matrixL().nestedExpression();  // strictly lower (LDL' keeps D apart), rows sorted
  const int n = static_cast<int>(L.rows());
  const int* Lp = L.outerIndexPtr();
  const int* Li = L.innerIndexPtr();
  const double* Lx = L.valuePtr();
  const auto D = m.ldlt.vectorD();
  m.z_lower.assign(Lp[n], 0.0);
  m.z_diag.assign(n, 0.0);
  std::vector<double> w(n, 0.0), lj(n, 0.0);
  std::vector<char> in(n, 0);
  for (int j = n - 1; j >= 0; --j) {
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) in[Li[p]] = 1, lj[Li[p]] = Lx[p], w[Li[p]] = 0.0;
    // w_i = sum_{k in J} Z_ik L_kj for i in J, reading Z_ik from column min(i, k)
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
      const int k = Li[p];
      const double lkj = Lx[p];
      w[k] += m.z_diag[k] * lkj;
      for (int q = Lp[k]; q < Lp[k + 1]; ++q) {
        const int r = Li[q];
        if (!in[r]) continue;
        w[r] += m.z_lower[q] * lkj;    // i = r > k
        w[k] += m.z_lower[q] * lj[r];  // i = k < r
      }
    }
    double d = 1.0 / D[j];
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
      m.z_lower[p] = -w[Li[p]];
      d -= Lx[p] * m.z_lower[p];
      in[Li[p]] = 0;
    }
    m.z_diag[j] = d;
  }
  m.have_selected = true;
  return true;
}

double sparse_ldlt::selected_inverse_entry(int i, int j) const {
  const impl& m = *impl_;
  if (!m.have_selected) throw std::out_of_range("sparse_ldlt::selected_inverse_entry: no selected inverse computed");
  const auto& idx = m.ldlt.permutationP().indices();
  int a = idx.size() > 0 ? idx[i] : i, b = idx.size() > 0 ? idx[j] : j;
  if (a == b) return m.z_diag[a];
  if (a < b) std::swap(a, b);
  const auto& L = m.ldlt.matrixL().nestedExpression();
  const int* first = L.innerIndexPtr() + L.outerIndexPtr()[b];
  const int* last = L.innerIndexPtr() + L.outerIndexPtr()[b + 1];
  const int* it = std::lower_bound(first, last, a);
  if (it == last || *it != a) throw std::out_of_range("sparse_ldlt::selected_inverse_entry: outside the pattern of L + L'");
  return m.z_lower[it - L.innerIndexPtr()];
}

}  // namespace laplace::sparse
