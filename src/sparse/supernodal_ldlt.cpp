// The supernodal route of laplace::sparse::sparse_ldlt (detail/supernodal_factor.hpp). Compiled as C++,
// never by nvcc; Eigen is used only for the AMD ordering (MPL2, adapted from CSparse), the dense blocks
// go through detail/dense_kernels.hpp.
#include "detail/supernodal_factor.hpp"

#include "detail/dense_kernels.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace laplace::sparse::detail {

void supernodal_factor::analyze(int n, const int* col_ptr, const int* row_idx) {
  n_ = n;
  block_.reset();
  factorized_ = false;
  have_selected_ = false;
  const int nnz = col_ptr[n];
  // the ordering: Eigen's AMD on the pattern of A + A', as the simplicial route computes it
  {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> lower(n, n);
    lower.resizeNonZeros(nnz);
    std::copy(col_ptr, col_ptr + n + 1, lower.outerIndexPtr());
    std::copy(row_idx, row_idx + nnz, lower.innerIndexPtr());
    std::fill(lower.valuePtr(), lower.valuePtr() + nnz, 1.0);
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> full = lower.selfadjointView<Eigen::Lower>();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pinv;
    Eigen::AMDOrdering<int>()(full, pinv);
    order_.assign(pinv.indices().data(), pinv.indices().data() + n);
  }
  inv_.resize(n);
  for (int k = 0; k < n; ++k) inv_[order_[k]] = k;
  // the permuted matrix by columns of its upper triangle
  up_ptr_.assign(n + 1, 0);
  a_slot_.resize(nnz);
  a_diag_.assign(nnz, 0);
  a_hi_.resize(nnz), a_lo_.resize(nnz);
  for (int c = 0; c < n; ++c)
    for (int q = col_ptr[c]; q < col_ptr[c + 1]; ++q) ++up_ptr_[std::max(inv_[row_idx[q]], inv_[c]) + 1];
  for (int k = 0; k < n; ++k) up_ptr_[k + 1] += up_ptr_[k];
  up_row_.resize(nnz);
  {
    std::vector<int> next(up_ptr_.begin(), up_ptr_.end() - 1);
    for (int c = 0; c < n; ++c)
      for (int q = col_ptr[c]; q < col_ptr[c + 1]; ++q) {
        const int a = inv_[row_idx[q]], b = inv_[c];
        const int s = next[std::max(a, b)]++;
        up_row_[s] = std::min(a, b);
        a_slot_[q] = s;
        a_diag_[q] = a == b;
        a_hi_[q] = std::max(a, b), a_lo_[q] = std::min(a, b);
      }
  }
  // the elimination tree (Liu's algorithm with path compression)
  parent_.assign(n, -1);
  {
    std::vector<int> ancestor(n, -1);
    for (int k = 0; k < n; ++k)
      for (int s = up_ptr_[k]; s < up_ptr_[k + 1]; ++s)
        for (int i = up_row_[s]; i != -1 && i < k;) {
          const int next = ancestor[i];
          ancestor[i] = k;
          if (next == -1) parent_[i] = k;
          i = next;
        }
  }
  // column counts by row subtrees, then the trailing dense supernode: the last m columns whose counts
  // are m, m - 1, ..., 1 (each holds every row below it)
  std::vector<int> mark(n, -1);
  {
    std::vector<long> count(n, 1);
    for (int k = 0; k < n; ++k) {
      mark[k] = k;
      for (int s = up_ptr_[k]; s < up_ptr_[k + 1]; ++s)
        for (int i = up_row_[s]; i < k && mark[i] != k; i = parent_[i]) mark[i] = k, ++count[i];
    }
    m_ = 0;
    while (m_ < n && count[n - 1 - m_] == m_ + 1) ++m_;
  }
  nB_ = n - m_;
  // the rows of L on the B columns, by the same row subtrees, then the columns
  std::fill(mark.begin(), mark.end(), -1);
  row_ptr_.assign(n + 1, 0);
  row_col_.clear();
  for (int k = 0; k < n; ++k) {
    const int limit = std::min(k, nB_);
    mark[k] = k;
    const std::size_t first = row_col_.size();
    for (int s = up_ptr_[k]; s < up_ptr_[k + 1]; ++s)
      for (int i = up_row_[s]; i < limit && mark[i] != k; i = parent_[i]) mark[i] = k, row_col_.push_back(i);
    std::sort(row_col_.begin() + static_cast<long>(first), row_col_.end());
    row_ptr_[k + 1] = static_cast<long>(row_col_.size());
  }
  Lp_.assign(nB_ + 1, 0);
  for (int c : row_col_) ++Lp_[c + 1];
  for (int j = 0; j < nB_; ++j) Lp_[j + 1] += Lp_[j];
  Li_.resize(Lp_[nB_]);
  row_pos_.resize(row_col_.size());
  {
    std::vector<long> next(Lp_.begin(), Lp_.end() - 1);
    for (int k = 0; k < n; ++k)  // rows in increasing order, so each column lists its rows sorted
      for (long t = row_ptr_[k]; t < row_ptr_[k + 1]; ++t) {
        const long p = next[row_col_[t]]++;
        Li_[p] = k;
        row_pos_[t] = p;
      }
  }
  t_start_.resize(nB_);
  for (int j = 0; j < nB_; ++j)
    t_start_[j] = static_cast<int>(std::lower_bound(Li_.begin() + Lp_[j], Li_.begin() + Lp_[j + 1], nB_) - Li_.begin());
  // the subtrees within B: a B column whose parent is in T (or none) is a root
  {
    std::vector<int> root(nB_);
    subtree_ptr_.assign(1, 0);
    std::vector<int> index(nB_, -1);
    for (int j = nB_ - 1; j >= 0; --j) {
      const int p = parent_[j];
      root[j] = (p == -1 || p >= nB_) ? j : root[p];
      if (root[j] == j) index[j] = static_cast<int>(subtree_ptr_.size()) - 1, subtree_ptr_.push_back(0);
    }
    for (int j = 0; j < nB_; ++j) ++subtree_ptr_[index[root[j]] + 1];
    for (std::size_t s = 1; s < subtree_ptr_.size(); ++s) subtree_ptr_[s] += subtree_ptr_[s - 1];
    subtree_nodes_.resize(nB_);
    std::vector<int> next(subtree_ptr_.begin(), subtree_ptr_.end() - 1);
    for (int j = nB_ - 1; j >= 0; --j) subtree_nodes_[next[index[root[j]]]++] = j;
  }
  // groups of consecutive B columns whose rows in T are the same (a supernode of B, such as the pair of
  // coordinates of one level of a grouping): the derivative reads the columns of Q and Z_TT once per group
  group_ptr_.assign(1, 0);
  for (int j = 0; j < nB_; ++j) {
    const bool joins = j > 0 && parent_[j - 1] == j && Lp_[j] - t_start_[j - 1] == Lp_[j + 1] - t_start_[j] &&
                       std::equal(Li_.begin() + t_start_[j - 1], Li_.begin() + Lp_[j], Li_.begin() + t_start_[j]);
    if (j > 0 && !joins) group_ptr_.push_back(j);
  }
  if (nB_ > 0) group_ptr_.push_back(nB_);
  Lx_.assign(Li_.size(), 0.0);
  D_.assign(nB_, 0.0);
  scatter_.assign(nnz, 0.0);
  dscatter_.assign(nnz, 0.0);
  // ponytail: one fixed size below which the loops run serially (the cost of an OpenMP region and its
  // per-thread scratch); measured in prototipi/step2b/bench_w0.cpp, tuned with the threshold w0
  parallel_ = static_cast<long>(Li_.size()) + static_cast<long>(m_) * m_ >= 100000;
  a_position_.resize(nnz);
  for (int q = 0; q < nnz; ++q)
    a_position_[q] = a_lo_[q] >= nB_ ? -1 - static_cast<long>(n) : a_hi_[q] == a_lo_[q] ? -1 - a_lo_[q] : b_position(a_hi_[q], a_lo_[q]);
}

sparse_ldlt_outcome supernodal_factor::factorize(const double* values, double shift) {
  factorized_ = false;
  have_selected_ = false;
  have_derivative_ = false;
  // not available after a factorization that fails, on either route (sparse_ldlt.hpp)
  min_pivot_ = max_pivot_ = log_abs_det_ = std::numeric_limits<double>::quiet_NaN();
  const int n = n_, nB = nB_, m = m_;
  for (std::size_t q = 0; q < a_slot_.size(); ++q) scatter_[a_slot_[q]] = values[q] + (a_diag_[q] ? shift : 0.0);
  S_.assign(static_cast<std::size_t>(m) * m, 0.0);
  double* S = S_.data();
  const auto s_at = [&](int row, int col) -> double& {  // S at permuted (row, col), row >= col, both in T
    return S[static_cast<std::size_t>(col - nB) * m + (row - nB)];
  };
  // up-looking over the rows: the B rows complete, the T rows their B part
  std::vector<double> x(n, 0.0);
  for (int k = 0; k < n; ++k) {
    double d = 0.0;
    for (int s = up_ptr_[k]; s < up_ptr_[k + 1]; ++s) {
      const int i = up_row_[s];
      if (i == k) d += scatter_[s];
      else if (i >= nB) s_at(k, i) += scatter_[s];
      else x[i] += scatter_[s];
    }
    for (long t = row_ptr_[k]; t < row_ptr_[k + 1]; ++t) {
      const int i = row_col_[t];
      const long p = row_pos_[t];
      const double yi = x[i];
      x[i] = 0.0;
      for (long q = Lp_[i]; q < p; ++q) {
        const int r = Li_[q];
        if (r >= nB) break;
        x[r] -= Lx_[q] * yi;
      }
      const double lki = yi / D_[i];
      if (k < nB) d -= lki * yi;  // for a T row the Schur update subtracts it
      Lx_[p] = lki;
    }
    if (k < nB) {
      if (!std::isfinite(d)) return sparse_ldlt_outcome::non_finite;
      if (!(d > 0.0)) return sparse_ldlt_outcome::not_positive_definite;
      D_[k] = d;
    } else {
      s_at(k, k) += d;
    }
  }
  // the Schur complement, S -= L_TB D_B L_TB', one column of S per iteration
#pragma omp parallel for schedule(dynamic, 16) if (parallel_)
  for (int k = nB; k < n; ++k) {
    double* col = S + static_cast<std::size_t>(k - nB) * m;  // column k of S, indexed by row - nB
    for (long t = row_ptr_[k]; t < row_ptr_[k + 1]; ++t) {
      const int i = row_col_[t];
      const long p = row_pos_[t];
      const double v = D_[i] * Lx_[p];
      for (long q = p; q < Lp_[i + 1]; ++q) col[Li_[q] - nB] -= Lx_[q] * v;  // rows >= k of column i, all in T
    }
  }
  for (int k = 0; k < m; ++k)
    if (!std::isfinite(S[static_cast<std::size_t>(k) * m + k])) return sparse_ldlt_outcome::non_finite;
  if (!block_) block_ = make_dense_root_block(m);
  if (block_->factorize(m, S) != 0) return sparse_ldlt_outcome::not_positive_definite;
  min_pivot_ = n > 0 ? std::numeric_limits<double>::infinity() : 0.0;
  max_pivot_ = n > 0 ? -std::numeric_limits<double>::infinity() : 0.0;
  log_abs_det_ = 0.0;
  for (int j = 0; j < nB; ++j) {
    min_pivot_ = std::min(min_pivot_, D_[j]), max_pivot_ = std::max(max_pivot_, D_[j]);
    log_abs_det_ += std::log(D_[j]);
  }
  for (int k = 0; k < m; ++k) {
    const double r = S[static_cast<std::size_t>(k) * m + k];
    if (!std::isfinite(r)) return sparse_ldlt_outcome::non_finite;
    min_pivot_ = std::min(min_pivot_, r * r), max_pivot_ = std::max(max_pivot_, r * r);
    log_abs_det_ += 2.0 * std::log(r);
  }
  factorized_ = true;
  return sparse_ldlt_outcome::positive_definite;
}

bool supernodal_factor::solve(double* b) const {
  if (!factorized_) return false;
  const int n = n_, nB = nB_, m = m_;
  std::vector<double> x(n);
  for (int r = 0; r < n; ++r) x[inv_[r]] = b[r];
  for (int j = 0; j < nB; ++j)
    for (long q = Lp_[j]; q < Lp_[j + 1]; ++q) x[Li_[q]] -= Lx_[q] * x[j];
  block_->solve(m, S_.data(), x.data() + nB);
  for (int j = 0; j < nB; ++j) x[j] /= D_[j];
  for (int j = nB - 1; j >= 0; --j)
    for (long q = Lp_[j]; q < Lp_[j + 1]; ++q) x[j] -= Lx_[q] * x[Li_[q]];
  for (int r = 0; r < n; ++r) b[r] = x[inv_[r]];
  return true;
}

long supernodal_factor::factor_nonzeros() const {
  return static_cast<long>(Li_.size()) + static_cast<long>(m_) * (m_ - 1) / 2;
}

bool supernodal_factor::compute_selected_inverse() {
  if (!factorized_) return false;
  const int n = n_, nB = nB_, m = m_;
  have_derivative_ = false;
  Z_.resize(static_cast<std::size_t>(m) * m);
  if (block_->invert(m, S_.data(), Z_.data()) != 0) return false;
  // Z_TT held whole (both triangles), so that its columns can be read contiguously
#pragma omp parallel for schedule(dynamic, 16) if (parallel_)
  for (int c = 0; c < m; ++c)
    for (int r = c + 1; r < m; ++r) Z_[static_cast<std::size_t>(r) * m + c] = Z_[static_cast<std::size_t>(c) * m + r];
  const double* Z = Z_.data();
  const auto zt = [&](int a, int b) {  // Z_TT at permuted (a, b), both in T
    return a >= b ? Z[static_cast<std::size_t>(b - nB) * m + (a - nB)] : Z[static_cast<std::size_t>(a - nB) * m + (b - nB)];
  };
  zl_.assign(Li_.size(), 0.0);
  zd_.assign(nB, 0.0);
  const int n_subtrees = static_cast<int>(subtree_ptr_.size()) - 1;
#pragma omp parallel if (parallel_)
  {
    std::vector<double> w(n, 0.0), lj(n, 0.0);
    std::vector<char> in(n, 0);
#pragma omp for schedule(dynamic, 4)
    for (int s = 0; s < n_subtrees; ++s)
      for (int u = subtree_ptr_[s]; u < subtree_ptr_[s + 1]; ++u) {  // from the subtree's root down
        const int j = subtree_nodes_[u];
        const long p0 = Lp_[j], p1 = Lp_[j + 1], pT = t_start_[j];
        for (long p = p0; p < p1; ++p) in[Li_[p]] = 1, lj[Li_[p]] = Lx_[p], w[Li_[p]] = 0.0;
        // w_i = sum_{k in J} Z_ik L_kj for i in J; the pairs with k in B read column k (rows i > k, both
        // in B or i in T) and its diagonal, the pairs with i and k in T read Z_TT
        for (long p = p0; p < pT; ++p) {
          const int k = Li_[p];
          const double lkj = Lx_[p];
          w[k] += zd_[k] * lkj;
          for (long q = Lp_[k]; q < Lp_[k + 1]; ++q) {
            const int r = Li_[q];
            if (!in[r]) continue;
            w[r] += zl_[q] * lkj;
            w[k] += zl_[q] * lj[r];
          }
        }
        for (long p = pT; p < p1; ++p) {
          const int k = Li_[p];
          const double lkj = Lx_[p];
          for (long q = pT; q < p1; ++q) w[Li_[q]] += zt(Li_[q], k) * lkj;
        }
        double d = 1.0 / D_[j];
        for (long p = p0; p < p1; ++p) {
          zl_[p] = -w[Li_[p]];
          d -= Lx_[p] * zl_[p];
          in[Li_[p]] = 0;
        }
        zd_[j] = d;
      }
  }
  have_selected_ = true;
  return true;
}

double supernodal_factor::dz_tt(int a, int b) const {
  const std::size_t m = m_;
  const double* z = Z_.data() + static_cast<std::size_t>(a - nB_) * m;
  const double* q = Q_.data() + static_cast<std::size_t>(b - nB_) * m;
  double s = 0.0;
  for (std::size_t c = 0; c < m; ++c) s += z[c] * q[c];
  return -s;
}

bool supernodal_factor::compute_selected_inverse_derivative(const double* dvalues) {
  have_derivative_ = false;
  if (!have_selected_) return false;
  const int n = n_, nB = nB_, m = m_;
  for (std::size_t q = 0; q < a_slot_.size(); ++q) dscatter_[a_slot_[q]] = dvalues[q];
  dl_.assign(Li_.size(), 0.0);
  dd_.assign(nB, 0.0);
  dS_.assign(static_cast<std::size_t>(m) * m, 0.0);
  double* dS = dS_.data();
  const auto ds_at = [&](int row, int col) -> double& { return dS[static_cast<std::size_t>(col - nB) * m + (row - nB)]; };
  // the derivative of the up-looking factorization: with y_i = L_ki D_i the value before the division,
  // dy follows the same triangular recursion, dL_ki = (dy_i - L_ki dD_i) / D_i, and
  // dD_k = dA_kk - sum_i (dL_ki y_i + L_ki dy_i)
  std::vector<double> dx(n, 0.0);
  for (int k = 0; k < n; ++k) {
    double dd = 0.0;
    for (int s = up_ptr_[k]; s < up_ptr_[k + 1]; ++s) {
      const int i = up_row_[s];
      if (i == k) dd += dscatter_[s];
      else if (i >= nB) ds_at(k, i) += dscatter_[s];
      else dx[i] += dscatter_[s];
    }
    for (long t = row_ptr_[k]; t < row_ptr_[k + 1]; ++t) {
      const int i = row_col_[t];
      const long p = row_pos_[t];
      const double dyi = dx[i], lki = Lx_[p], yi = lki * D_[i];
      dx[i] = 0.0;
      for (long q = Lp_[i]; q < p; ++q) {
        const int r = Li_[q];
        if (r >= nB) break;
        dx[r] -= dl_[q] * yi + Lx_[q] * dyi;
      }
      const double dlki = (dyi - lki * dd_[i]) / D_[i];
      dl_[p] = dlki;
      if (k < nB) dd -= dlki * yi + lki * dyi;
    }
    if (k < nB) dd_[k] = dd;
    else ds_at(k, k) += dd;
  }
  // dS = dA_TT - d(L_TB D_B L_TB'), one column per iteration, then both triangles
#pragma omp parallel for schedule(dynamic, 16) if (parallel_)
  for (int k = nB; k < n; ++k) {
    double* col = dS + static_cast<std::size_t>(k - nB) * m;
    for (long t = row_ptr_[k]; t < row_ptr_[k + 1]; ++t) {
      const int i = row_col_[t];
      const long p = row_pos_[t];
      const double a1 = D_[i] * Lx_[p], a2 = dd_[i] * Lx_[p] + D_[i] * dl_[p];
      for (long q = p; q < Lp_[i + 1]; ++q) col[Li_[q] - nB] -= Lx_[q] * a2 + dl_[q] * a1;
    }
  }
#pragma omp parallel for schedule(dynamic, 16) if (parallel_)
  for (int c = 0; c < m; ++c)
    for (int r = c + 1; r < m; ++r) dS[static_cast<std::size_t>(r) * m + c] = dS[static_cast<std::size_t>(c) * m + r];
  // Q = dS Z_TT, so that dZ_TT = -Z_TT dS Z_TT = -Z_TT Q
  Q_.resize(static_cast<std::size_t>(m) * m);
  block_->symmetric_product(m, Z_.data(), dS, Q_.data());
  // the pairs with i and k in T, for every B column j: t_ij = sum_k (dZ_ik L_kj + Z_ik dL_kj)
  // = -Z_TT(:, i)' u_j + sum_k Z_ik dL_kj with u_j = sum_k Q(:, k) L_kj; the columns of a group share
  // their rows k, i in T, so each column of Q and of Z_TT is read once per group
  const double* Z = Z_.data();
  const double* Q = Q_.data();
  dtt_.assign(Li_.size(), 0.0);
  const int n_groups = static_cast<int>(group_ptr_.size()) - 1;
#pragma omp parallel if (parallel_)
  {
    std::vector<double> ug;
#pragma omp for schedule(dynamic, 8)
    for (int g = 0; g < n_groups; ++g) {
      const int j0 = group_ptr_[g], width = group_ptr_[g + 1] - j0;
      const long c = Lp_[j0 + 1] - t_start_[j0];  // rows in T, the same for every column of the group
      if (c == 0) continue;
      ug.assign(static_cast<std::size_t>(width) * m, 0.0);
      for (long r = 0; r < c; ++r) {
        const double* qk = Q + static_cast<std::size_t>(Li_[t_start_[j0] + r] - nB) * m;
        for (int v = 0; v < width; ++v) {
          const double lkj = Lx_[t_start_[j0 + v] + r];
          double* u = ug.data() + static_cast<std::size_t>(v) * m;
          for (int a = 0; a < m; ++a) u[a] += qk[a] * lkj;
        }
      }
      for (long r = 0; r < c; ++r) {
        const int i = Li_[t_start_[j0] + r];
        const double* zi = Z + static_cast<std::size_t>(i - nB) * m;
        for (int v = 0; v < width; ++v) {
          const long pT = t_start_[j0 + v];
          const double* u = ug.data() + static_cast<std::size_t>(v) * m;
          double acc = 0.0;
          for (int a = 0; a < m; ++a) acc -= zi[a] * u[a];
          for (long t = 0; t < c; ++t) acc += zi[Li_[pT + t] - nB] * dl_[pT + t];
          dtt_[pT + r] = acc;
        }
      }
    }
  }
  // Takahashi's recursion differentiated on the B columns, in parallel over the subtrees
  dzl_.assign(Li_.size(), 0.0);
  dzd_.assign(nB, 0.0);
  const int n_subtrees = static_cast<int>(subtree_ptr_.size()) - 1;
#pragma omp parallel if (parallel_)
  {
    std::vector<double> w(n, 0.0), lj(n, 0.0), dlj(n, 0.0);
    std::vector<char> in(n, 0);
#pragma omp for schedule(dynamic, 4)
    for (int s = 0; s < n_subtrees; ++s)
      for (int v = subtree_ptr_[s]; v < subtree_ptr_[s + 1]; ++v) {
        const int j = subtree_nodes_[v];
        const long p0 = Lp_[j], p1 = Lp_[j + 1], pT = t_start_[j];
        for (long p = p0; p < p1; ++p) in[Li_[p]] = 1, lj[Li_[p]] = Lx_[p], dlj[Li_[p]] = dl_[p], w[Li_[p]] = 0.0;
        // w_i = sum_{k in J} (dZ_ik L_kj + Z_ik dL_kj) for i in J
        for (long p = p0; p < pT; ++p) {
          const int k = Li_[p];
          const double lkj = Lx_[p], dlkj = dl_[p];
          w[k] += dzd_[k] * lkj + zd_[k] * dlkj;
          for (long q = Lp_[k]; q < Lp_[k + 1]; ++q) {
            const int r = Li_[q];
            if (!in[r]) continue;
            w[r] += dzl_[q] * lkj + zl_[q] * dlkj;
            w[k] += dzl_[q] * lj[r] + zl_[q] * dlj[r];
          }
        }
        for (long q = pT; q < p1; ++q) w[Li_[q]] += dtt_[q];  // the pairs in T, computed by groups above
        double d = -dd_[j] / (D_[j] * D_[j]);
        for (long p = p0; p < p1; ++p) {
          dzl_[p] = -w[Li_[p]];
          d -= dl_[p] * zl_[p] + Lx_[p] * dzl_[p];
          in[Li_[p]] = 0;
        }
        dzd_[j] = d;
      }
  }
  have_derivative_ = true;
  return true;
}

double supernodal_factor::selected_inverse_derivative_entry(int i, int j) const {
  if (!have_derivative_)
    throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_entry: no derivative computed");
  int a = inv_[i], b = inv_[j];
  if (a < b) std::swap(a, b);
  if (b >= nB_) return dz_tt(a, b);
  if (a == b) return dzd_[b];
  const long p = b_position(a, b);
  if (p < 0) throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_entry: outside the pattern of L + L'");
  return dzl_[p];
}

double supernodal_factor::selected_inverse_derivative_dot(const double* values) const {
  if (!have_derivative_) throw std::out_of_range("sparse_ldlt::selected_inverse_derivative_dot: no derivative computed");
  const int nnz = static_cast<int>(a_position_.size());
  double s = 0.0;
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : s) if (parallel_)
  for (int q = 0; q < nnz; ++q) {
    const long p = a_position_[q];
    const double dz = a_lo_[q] >= nB_ ? dz_tt(a_hi_[q], a_lo_[q]) : p < 0 ? dzd_[-1 - p] : dzl_[p];
    s += (a_hi_[q] == a_lo_[q] ? 1.0 : 2.0) * dz * values[q];
  }
  return s;
}

long supernodal_factor::b_position(int a, int b) const {
  const auto first = Li_.begin() + Lp_[b], last = Li_.begin() + Lp_[b + 1];
  const auto it = std::lower_bound(first, last, a);
  return it == last || *it != a ? -1 : static_cast<long>(it - Li_.begin());
}

double supernodal_factor::selected_inverse_entry(int i, int j) const {
  if (!have_selected_) throw std::out_of_range("sparse_ldlt::selected_inverse_entry: no selected inverse computed");
  int a = inv_[i], b = inv_[j];
  if (a < b) std::swap(a, b);
  if (b >= nB_) return Z_[static_cast<std::size_t>(b - nB_) * m_ + (a - nB_)];
  if (a == b) return zd_[b];
  const long p = b_position(a, b);
  if (p < 0) throw std::out_of_range("sparse_ldlt::selected_inverse_entry: outside the pattern of L + L'");
  return zl_[p];
}

}  // namespace laplace::sparse::detail
