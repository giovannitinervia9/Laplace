#pragma once

// The curvature solver of Newton for a sparse H (note/optimizers.md, decision
// 7; the dense one is dense_curvature_solver.hpp). The matrix type is
// sparse_curvature_matrix: the lower triangle of a symmetric n x n matrix,
// diagonal included and first in each column, in compressed sparse columns.
// Its pattern is set by the objective that fills it (the constructor that
// Newton calls gives an empty pattern) and stays the same between calls, so
// the ordering and the symbolic factorization are done once.
//
//   factorize(H)                  H = L D L' (laplace::sparse::sparse_ldlt);
//                                 positive_definite when every pivot is > 0
//   factorize_shifted(H, lambda)  the same for H + lambda I
//   newton_direction(g, d)        d = -(H + lambda I)^-1 g with the last factor
//   condition_estimate()          the ratio of the largest to the smallest pivot
//                                 of D, a lower bound of kappa_2 for a positive
//                                 definite H, not Higham's estimate of kappa_1
//
// There is no eigenvalue route and no negative curvature search for a sparse
// H: Newton falls back to steepest descent when the factorization is not
// positive definite. An objective with
// a sparse H that can be indefinite (an observed information) is expected to
// return a positive definite matrix itself (row_model_objective in
// engine/inner_fit.hpp replaces it by the expected information).

#include "laplace/optimizers/dense_curvature_solver.hpp"
#include "laplace/sparse/sparse_ldlt.hpp"

#include <vector>

namespace laplace::optimizers {

/**
 * \ingroup optimizers_second_order
 * \brief Lower triangle of a symmetric sparse \f$n \times n\f$ matrix in compressed sparse columns.
 *
 * The diagonal entry is stored first in each column (`row_idx[col_ptr[c]] == c`), and row indices
 * within each column are sorted (`row >= column`). The sparsity pattern is populated
 * by the objective on the first curvature evaluation and kept constant across calls.
 *
 * \see sparse_curvature_solver, \ref sparse "Sparse linear algebra", \ref optimizers_second_order
 */
struct sparse_curvature_matrix {
  int n = 0;
  std::vector<int> col_ptr;    ///< \f$n + 1\f$ offsets
  std::vector<int> row_idx;    ///< row of each entry, sorted within a column, diagonal first
  std::vector<double> values;  ///< the entries

  /**
   * \ingroup optimizers_second_order
   * \brief Constructs an empty \f$0 \times 0\f$ sparse curvature matrix.
   */
  sparse_curvature_matrix() = default;
  /**
   * \ingroup optimizers_second_order
   * \brief Allocates an empty pattern of size `rows` in the constructor shape expected by `newton_optimizer`.
   * \param rows Matrix dimension \f$n\f$.
   */
  sparse_curvature_matrix(const char*, int rows, int) : n(rows), col_ptr(rows + 1, 0) {}
};

/**
 * \ingroup optimizers_second_order
 * \brief Sparse \f$L D L^\top\f$ curvature solver for sparse_curvature_matrix on the host.
 *
 * Delegates to `laplace::sparse::sparse_ldlt` (requiring `Laplace::sparse`), analyzing the sparsity
 * pattern once per pattern change. Unlike `dense_curvature_solver`, `newton_optimizer` has no
 * eigendecomposition route or negative-curvature search for sparse matrices and falls back to
 * steepest descent when a matrix is not positive definite.
 *
 * \see sparse_curvature_matrix, dense_curvature_solver, \ref sparse "Sparse linear algebra", \ref optimizers_second_order
 */
class sparse_curvature_solver {
 public:
  /**
   * \ingroup optimizers_second_order
   * \brief Curvature matrix type accepted by this solver.
   */
  using matrix_type = sparse_curvature_matrix;

  /**
   * \ingroup optimizers_second_order
   * \brief Constructs a sparse curvature solver for \f$n \times n\f$ matrices.
   * \param n Problem dimension.
   */
  explicit sparse_curvature_solver(int n) : n_(n), rhs_(n) {}
  /**
   * \ingroup optimizers_second_order
   * \brief Returns the dimension \f$n\f$ this solver was constructed for.
   * \return Problem dimension \f$n\f$.
   */
  int size() const { return n_; }

  /**
   * \ingroup optimizers_second_order
   * \brief Factorizes \f$H = L D L^\top\f$ from the lower triangle of `H`.
   * \param H Sparse symmetric curvature matrix.
   * \return Factorization outcome (`positive_definite`, `not_positive_definite`, or `non_finite`).
   */
  curvature_factorization factorize(const matrix_type& H) { return factorize_shifted(H, 0.0); }

  /**
   * \ingroup optimizers_second_order
   * \brief Factorizes \f$H + \lambda I = L D L^\top\f$ from the lower triangle of `H`.
   * \param H Sparse symmetric curvature matrix (left unchanged).
   * \param lambda Diagonal shift \f$\lambda\f$ added to each diagonal entry.
   * \return Factorization outcome (`positive_definite`, `not_positive_definite`, or `non_finite`).
   */
  curvature_factorization factorize_shifted(const matrix_type& H, double lambda) {
    const auto r = ldlt_.factorize(H.n, H.col_ptr.data(), H.row_idx.data(), H.values.data(), lambda);
    outcome_ = r == sparse::sparse_ldlt_outcome::positive_definite       ? curvature_factorization::positive_definite
               : r == sparse::sparse_ldlt_outcome::not_positive_definite ? curvature_factorization::not_positive_definite
                                                                         : curvature_factorization::non_finite;
    return outcome_;
  }

  /**
   * \ingroup optimizers_second_order
   * \brief Returns the outcome of the most recent factorization call.
   * \return Last `curvature_factorization` status (`not_evaluated` initially).
   */
  curvature_factorization last_factorization() const { return outcome_; }

  /**
   * \ingroup optimizers_second_order
   * \brief Solves \f$(H + \lambda I) d = -g\f$ using the last positive-definite \f$L D L^\top\f$ factor.
   *
   * Leaves `d` untouched and returns `false` if the last factorization was not `positive_definite`,
   * the solve failed, or any solution entry is non-finite.
   *
   * \param g Gradient vector \f$g\f$ of length \f$n\f$.
   * \param d Output direction vector \f$d\f$ of length \f$n\f$.
   * \return `true` on success, `false` otherwise.
   */
  bool newton_direction(const optimizer_point& g, const optimizer_vector& d) {
    if (outcome_ != curvature_factorization::positive_definite) return false;
    for (int i = 0; i < n_; ++i) rhs_[i] = -g(i);
    if (!ldlt_.solve(rhs_.data())) return false;
    for (int i = 0; i < n_; ++i) {
      if (!std::isfinite(rhs_[i])) return false;
      d(i) = rhs_[i];
    }
    return true;
  }

  /**
   * \ingroup optimizers_second_order
   * \brief Returns the ratio of the largest to the smallest pivot of \f$D\f$ from the last factorization.
   *
   * Computes `max_pivot() / min_pivot()`, which is a lower bound on \f$\kappa_2\f$ for positive
   * definite \f$H\f$ rather than Higham's estimate of \f$\kappa_1\f$ used by `dense_curvature_solver`.
   *
   * \return Pivot ratio \f$d_{\max} / d_{\min}\f$, or \f$+\infty\f$ when not `positive_definite`.
   */
  double condition_estimate() const {
    if (outcome_ != curvature_factorization::positive_definite) return std::numeric_limits<double>::infinity();
    return ldlt_.max_pivot() / ldlt_.min_pivot();
  }

 private:
  int n_;
  std::vector<double> rhs_;
  sparse::sparse_ldlt ldlt_;
  curvature_factorization outcome_ = curvature_factorization::not_evaluated;
};

/**
 * \ingroup optimizers_second_order
 * \brief Specialization of `curvature_solver_for` selecting `sparse_curvature_solver` for `sparse_curvature_matrix`.
 *
 * \see sparse_curvature_solver, sparse_curvature_matrix, \ref optimizers_second_order
 */
template <>
struct curvature_solver_for<sparse_curvature_matrix> {
  /**
   * \ingroup optimizers_second_order
   * \brief Solver type `sparse_curvature_solver`.
   */
  using type = sparse_curvature_solver;
};

}  // namespace laplace::optimizers
