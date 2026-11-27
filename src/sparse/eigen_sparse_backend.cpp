// The Eigen code of laplace::sparse (include/laplace/sparse/sparse_ldlt.hpp):
// SimplicialLDLT with the approximate minimum degree ordering (MPL2, adapted
// from CSparse). Compiled as C++, never by nvcc.
#include <laplace/sparse/sparse_ldlt.hpp>

#include "detail/supernodal_factor.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace laplace::sparse {

using sparse_matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The width of the trailing dense supernode from which `automatic` takes the supernodal route. Measured
// on 2026-11-27 (prototipi/step2b/bench_w0.cpp, crossed models, OpenBLAS, 1 and 12 threads): the routes
// cost the same at a width of about 40 (factorization, selected inverse and one derivative direction
// within 0.95-1.2 of each other); at about 80 the supernodal route is 1.6-2.7 times faster; at about 5
// (one grouping, four globals) it is 0.5-0.9 times as fast. Near the crossover the simplicial route,
// the reference, is kept.
constexpr int supernodal_route_min_width = 64;

struct sparse_ldlt::impl {
  Eigen::SimplicialLDLT<sparse_matrix, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
  sparse_matrix A;
  std::vector<int> col_ptr, row_idx;  // the pattern last analyzed
  bool analyzed = false, factorized = false, have_selected = false, have_derivative = false;
  std::vector<double> z_lower, z_diag;  // the selected inverse: z_lower on the pattern of L, permuted ordering
  // the derivative along a direction: of the factor (dl on the pattern of L, dd) and of the selected inverse
  std::vector<double> dl, dd, dz_lower, dz_diag;
  // for the derivative, built once per pattern: where each stored entry of A sits in the factor's storage
  // (p >= 0 an entry of L, -1 - j the diagonal j), and the rows of L (for row j the columns k < j with
  // L_jk != 0 and the position of L_jk)
  bool structure_ready = false;
  std::vector<int> a_position, row_ptr, row_col, row_pos;
  double min_pivot = 0.0, max_pivot = 0.0, log_abs_det = 0.0;
  // the route: requested by set_route(), in use since the last analysis; the supernodal factor
  sparse_ldlt_route requested = sparse_ldlt_route::automatic, in_use = sparse_ldlt_route::automatic;
  std::unique_ptr<detail::supernodal_factor> supernodal;
  bool supernodal_route() const { return in_use == sparse_ldlt_route::supernodal; }
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
  m.have_derivative = false;
  const int nnz = col_ptr[n];
  for (int k = 0; k < nnz; ++k)
    if (!std::isfinite(values[k])) return sparse_ldlt_outcome::non_finite;
  const bool same_pattern = m.analyzed && m.in_use != sparse_ldlt_route::automatic &&
                            static_cast<int>(m.col_ptr.size()) == n + 1 &&
                            std::equal(col_ptr, col_ptr + n + 1, m.col_ptr.begin()) &&
                            std::equal(row_idx, row_idx + nnz, m.row_idx.begin());
  if (!same_pattern) {
    m.col_ptr.assign(col_ptr, col_ptr + n + 1);
    m.row_idx.assign(row_idx, row_idx + nnz);
    m.A = sparse_matrix(n, n);
    m.A.resizeNonZeros(nnz);
    std::copy(col_ptr, col_ptr + n + 1, m.A.outerIndexPtr());
    std::copy(row_idx, row_idx + nnz, m.A.innerIndexPtr());
    // the route: forced, or chosen from the width of the trailing dense supernode of the factor
    // (supernodal_route_min_width, measured)
    m.in_use = m.requested;
    if (m.requested != sparse_ldlt_route::simplicial) {
      if (!m.supernodal) m.supernodal = std::make_unique<detail::supernodal_factor>();
      m.supernodal->analyze(n, col_ptr, row_idx);
      if (m.requested == sparse_ldlt_route::automatic)
        m.in_use = m.supernodal->root_width() >= supernodal_route_min_width ? sparse_ldlt_route::supernodal
                                                                              : sparse_ldlt_route::simplicial;
      if (m.supernodal_route()) m.analyzed = true;
      else m.supernodal.reset();
    }
  }
  if (m.supernodal_route()) {
    const auto out = m.supernodal->factorize(values, shift);
    m.min_pivot = m.supernodal->min_pivot();
    m.max_pivot = m.supernodal->max_pivot();
    m.log_abs_det = m.supernodal->log_abs_determinant();
    m.factorized = out == sparse_ldlt_outcome::positive_definite;
    return out;
  }
  std::copy(values, values + nnz, m.A.valuePtr());
  if (shift != 0.0)
    for (int c = 0; c < n; ++c) m.A.valuePtr()[col_ptr[c]] += shift;  // the diagonal is first in its column
  if (!same_pattern) {
    m.ldlt.analyzePattern(m.A);
    m.analyzed = true;
    m.structure_ready = false;
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

void sparse_ldlt::set_route(sparse_ldlt_route route) {
  if (route != impl_->requested) impl_->requested = route, impl_->in_use = sparse_ldlt_route::automatic;
}

sparse_ldlt_route sparse_ldlt::route() const { return impl_->in_use; }

bool sparse_ldlt::solve(double* b) const {
  if (!impl_->factorized) return false;
  if (impl_->supernodal_route()) return impl_->supernodal->solve(b);
  const int n = static_cast<int>(impl_->A.rows());
  Eigen::Map<Eigen::VectorXd> v(b, n);
  v = impl_->ldlt.solve(Eigen::VectorXd(v));
  return true;
}

double sparse_ldlt::min_pivot() const { return impl_->min_pivot; }
double sparse_ldlt::max_pivot() const { return impl_->max_pivot; }
double sparse_ldlt::log_abs_determinant() const { return impl_->log_abs_det; }
long sparse_ldlt::factor_nonzeros() const {
  if (impl_->analyzed && impl_->supernodal_route()) return impl_->supernodal->factor_nonzeros();
  return impl_->analyzed ? static_cast<long>(impl_->ldlt.matrixL().nestedExpression().nonZeros()) : 0;
}

bool sparse_ldlt::compute_selected_inverse() {
  impl& m = *impl_;
  if (!m.factorized) return false;
  if (m.supernodal_route()) return m.have_selected = m.supernodal->compute_selected_inverse();
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
  if (m.supernodal_route()) return m.supernodal->selected_inverse_entry(i, j);
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

namespace {

// the position in the factor's storage of the permuted entry (a, b): -1 - a on the diagonal, else the
// index of L_ab (a > b) in its column; -2 - n when outside the pattern
int factor_position(const sparse_matrix& L, int a, int b) {
  if (a == b) return -1 - a;
  if (a < b) std::swap(a, b);
  const int* first = L.innerIndexPtr() + L.outerIndexPtr()[b];
  const int* last = L.innerIndexPtr() + L.outerIndexPtr()[b + 1];
  const int* it = std::lower_bound(first, last, a);
  if (it == last || *it != a) return -2 - static_cast<int>(L.rows());
  return static_cast<int>(it - L.innerIndexPtr());
}

}  // namespace

bool sparse_ldlt::compute_selected_inverse_derivative(const double* dvalues) {
  impl& m = *impl_;
  m.have_derivative = false;
  if (!m.have_selected) return false;
  if (m.supernodal_route()) return m.have_derivative = m.supernodal->compute_selected_inverse_derivative(dvalues);
  const auto& L = m.ldlt.matrixL().nestedExpression();
  const int n = static_cast<int>(L.rows());
  const int* Lp = L.outerIndexPtr();
  const int* Li = L.innerIndexPtr();
  const double* Lx = L.valuePtr();
  const auto D = m.ldlt.vectorD();
  const auto& idx = m.ldlt.permutationP().indices();
  if (!m.structure_ready) {
    const int nnz = m.col_ptr[n];
    m.a_position.resize(nnz);
    for (int c = 0; c < n; ++c)
      for (int k = m.col_ptr[c]; k < m.col_ptr[c + 1]; ++k) {
        const int r = m.row_idx[k];
        m.a_position[k] = factor_position(L, idx.size() > 0 ? idx[r] : r, idx.size() > 0 ? idx[c] : c);
      }
    m.row_ptr.assign(n + 1, 0);
    for (int p = 0; p < Lp[n]; ++p) ++m.row_ptr[Li[p] + 1];
    for (int j = 0; j < n; ++j) m.row_ptr[j + 1] += m.row_ptr[j];
    m.row_col.resize(Lp[n]);
    m.row_pos.resize(Lp[n]);
    std::vector<int> next(m.row_ptr.begin(), m.row_ptr.end() - 1);
    for (int k = 0; k < n; ++k)  // columns in increasing order, so each row lists its columns sorted
      for (int p = Lp[k]; p < Lp[k + 1]; ++p) m.row_col[next[Li[p]]] = k, m.row_pos[next[Li[p]]++] = p;
    m.structure_ready = true;
  }
  // the direction on the factor's storage (permuted ordering, lower triangle)
  std::vector<double> xl(Lp[n], 0.0), xd(n, 0.0);
  for (int k = 0; k < m.col_ptr[n]; ++k) {
    const int q = m.a_position[k];
    if (q < -1 - n) throw std::logic_error("sparse_ldlt: an entry of A outside the pattern of its factor");
    (q >= 0 ? xl[q] : xd[-1 - q]) += dvalues[k];
  }
  // the derivative of the factorization, left-looking: column j from the columns k < j with L_jk != 0
  m.dl.assign(Lp[n], 0.0);
  m.dd.assign(n, 0.0);
  std::vector<double> x(n, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) x[Li[p]] = xl[p];
    double dj = xd[j];
    for (int t = m.row_ptr[j]; t < m.row_ptr[j + 1]; ++t) {
      const int k = m.row_col[t], pjk = m.row_pos[t];
      const double ljk = Lx[pjk], dljk = m.dl[pjk], Dk = D[k], dDk = m.dd[k];
      dj -= 2.0 * ljk * dljk * Dk + ljk * ljk * dDk;
      const double c1 = ljk * Dk, c2 = dljk * Dk + ljk * dDk;
      for (int q = pjk + 1; q < Lp[k + 1]; ++q) x[Li[q]] -= m.dl[q] * c1 + Lx[q] * c2;  // rows i > j of column k
    }
    m.dd[j] = dj;
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
      m.dl[p] = (x[Li[p]] - Lx[p] * dj) / D[j];
      x[Li[p]] = 0.0;
    }
  }
  // the derivative of Takahashi's recursion, the same sweep as compute_selected_inverse()
  m.dz_lower.assign(Lp[n], 0.0);
  m.dz_diag.assign(n, 0.0);
  std::vector<double> w(n, 0.0), lj(n, 0.0), dlj(n, 0.0);
  std::vector<char> in(n, 0);
  for (int j = n - 1; j >= 0; --j) {
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) in[Li[p]] = 1, lj[Li[p]] = Lx[p], dlj[Li[p]] = m.dl[p], w[Li[p]] = 0.0;
    // w_i = sum_{k in J} (dZ_ik L_kj + Z_ik dL_kj) for i in J
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
      const int k = Li[p];
      const double lkj = Lx[p], dlkj = m.dl[p];
      w[k] += m.dz_diag[k] * lkj + m.z_diag[k] * dlkj;
      for (int q = Lp[k]; q < Lp[k + 1]; ++q) {
        const int r = Li[q];
        if (!in[r]) continue;
        w[r] += m.dz_lower[q] * lkj + m.z_lower[q] * dlkj;    // i = r > k
        w[k] += m.dz_lower[q] * lj[r] + m.z_lower[q] * dlj[r];  // i = k < r
      }
    }
    double d = -m.dd[j] / (D[j] * D[j]);
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
      m.dz_lower[p] = -w[Li[p]];
      d -= m.dl[p] * m.z_lower[p] + Lx[p] * m.dz_lower[p];
      in[Li[p]] = 0;
    }
    m.dz_diag[j] = d;
  }
  m.have_derivative = true;
  return true;
}

double sparse_ldlt::selected_inverse_derivative_entry(int i, int j) const {
  const impl& m = *impl_;
  if (!m.have_derivative)
    throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_entry: no derivative computed");
  if (m.supernodal_route()) return m.supernodal->selected_inverse_derivative_entry(i, j);
  const auto& L = m.ldlt.matrixL().nestedExpression();
  const auto& idx = m.ldlt.permutationP().indices();
  const int q = factor_position(L, idx.size() > 0 ? idx[i] : i, idx.size() > 0 ? idx[j] : j);
  if (q < -static_cast<int>(L.rows()) - 1)
    throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_entry: outside the pattern of L + L'");
  return q >= 0 ? m.dz_lower[q] : m.dz_diag[-1 - q];
}

double sparse_ldlt::selected_inverse_derivative_dot(const double* values) const {
  const impl& m = *impl_;
  if (!m.have_derivative)
    throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_dot: no derivative computed");
  if (m.supernodal_route()) return m.supernodal->selected_inverse_derivative_dot(values);
  const int n = static_cast<int>(m.col_ptr.size()) - 1;
  double s = 0.0;
  for (int c = 0; c < n; ++c)
    for (int k = m.col_ptr[c]; k < m.col_ptr[c + 1]; ++k) {
      const int q = m.a_position[k];
      s += (q >= 0 ? 2.0 * m.dz_lower[q] : m.dz_diag[-1 - q]) * values[k];
    }
  return s;
}

}  // namespace laplace::sparse
