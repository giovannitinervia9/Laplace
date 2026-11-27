#pragma once

// The supernodal route of laplace::sparse::sparse_ldlt (internal; no public header includes this file).
//
// The matrix is ordered by approximate minimum degree (Eigen's AMD, the same ordering as the simplicial
// route) and analyzed: elimination tree, column counts, and the trailing supernode T, the last m columns,
// whose lower triangle in the factor is dense. The leading columns B are factorized as L D L' with unit
// L by an up-looking sparse recursion (rows of T get their B part too); the Schur complement
// S = A_TT - L_TB D_B L_TB' is formed dense and factorized as S = R R' by the dense kernels
// (detail/dense_kernels.hpp). The selected inverse is Z_TT = S^-1, dense, and Takahashi's recursion on
// the B columns reading Z_TT, run in parallel over the subtrees of the elimination tree that hang below T.

#include <laplace/sparse/sparse_ldlt.hpp>

#include "dense_root_block.hpp"

#include <memory>
#include <vector>

namespace laplace::sparse::detail {

class supernodal_factor {
 public:
  // symbolic analysis of the lower triangle (col_ptr, row_idx)
  void analyze(int n, const int* col_ptr, const int* row_idx);
  // the number of columns of the trailing dense supernode, after analyze()
  int root_width() const { return m_; }
  sparse_ldlt_outcome factorize(const double* values, double shift);
  bool solve(double* b) const;
  double min_pivot() const { return min_pivot_; }
  double max_pivot() const { return max_pivot_; }
  double log_abs_determinant() const { return log_abs_det_; }
  long factor_nonzeros() const;
  bool compute_selected_inverse();
  // throws std::out_of_range outside the pattern of L + L' or when no selected inverse is stored
  double selected_inverse_entry(int i, int j) const;
  // the derivative of the selected inverse along dA (aligned with the analyzed lower triangle): dL, dD on
  // the B columns, dS dense, Q = dS Z_TT by one symmetric product, so that dZ_TT = -Z_TT Q is read entry
  // by entry (-Z_TT(:, a)' Q(:, b)) where it is needed; then Takahashi's recursion differentiated on B
  bool compute_selected_inverse_derivative(const double* dvalues);
  double selected_inverse_derivative_entry(int i, int j) const;
  double selected_inverse_derivative_dot(const double* values) const;

 private:
  // the position of the permuted entry (a, b), a >= b, in the B columns' storage, or -1
  long b_position(int a, int b) const;
  // dZ_TT at permuted (a, b), both in T
  double dz_tt(int a, int b) const;

  int n_ = 0, m_ = 0, nB_ = 0;
  std::vector<int> order_, inv_;  // order_[new] = old, inv_[old] = new
  // the matrix permuted, by columns of its upper triangle (rows i <= k in column k); a_slot_ maps each
  // stored entry of the user's lower triangle to its place there, a_diag_ marks the diagonal entries
  std::vector<int> up_ptr_, up_row_, a_slot_;
  std::vector<char> a_diag_;
  std::vector<int> a_hi_, a_lo_;  // each stored entry at permuted (hi, lo), hi >= lo
  std::vector<int> parent_;
  // the B columns of L (strictly lower, rows sorted, B rows first then T rows), D of B
  std::vector<long> Lp_;
  std::vector<int> Li_, t_start_;  // t_start_[j]: the first position of a T row in column j
  std::vector<double> Lx_, D_;
  // the rows of L restricted to the B columns: for row k, the columns i < k with L_ki != 0 (ascending)
  // and the position of L_ki
  std::vector<long> row_ptr_, row_pos_;
  std::vector<int> row_col_;
  // the subtrees of the elimination tree within B, each listed from its root down (descending)
  std::vector<int> subtree_ptr_, subtree_nodes_;
  // groups of consecutive B columns with the same rows in T, as first columns (the last entry is nB)
  std::vector<int> group_ptr_;
  // the dense block: S, then its factor R (lower triangle), m x m column-major; Z_TT (lower triangle)
  std::vector<double> scatter_, S_, Z_;
  std::vector<double> zl_, zd_;  // the selected inverse on the B columns, aligned with Lx_ and D_
  // the derivative: dL, dD, dZ on the B columns; dS, then Q = dS Z_TT (m x m); for each stored entry of
  // the user's lower triangle its place in the factor: >= 0 a position in the B columns, -1 - j the
  // diagonal j of B, -1 - n an entry of T x T (built once per pattern)
  std::vector<double> dscatter_, dl_, dd_, dzl_, dzd_, dS_, Q_;
  std::vector<double> dtt_;  // the derivative's sums over pairs in T of each B column, aligned with Li_
  std::vector<long> a_position_;
  bool factorized_ = false, have_selected_ = false, have_derivative_ = false;
  bool parallel_ = false;
  // the root block's dense operations (on the host or the device), created at the first factorization
  std::unique_ptr<dense_root_block> block_;  // whether the loops run in OpenMP regions (a large enough factor)
  double min_pivot_ = 0.0, max_pivot_ = 0.0, log_abs_det_ = 0.0;
};

}  // namespace laplace::sparse::detail
