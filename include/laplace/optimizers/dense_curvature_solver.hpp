#pragma once

// The curvature solver of Newton and the trust region for a dense H
// (note/optimizers.md, decision 7): the methods never touch H directly, only
// these operations. The solver keeps the Cholesky factor of the last matrix
// it factorized; the matrix is read from its lower triangle and not modified.
//
//   factorize(H)                     H = L L^T, or the reason it failed
//   factorize_shifted(H, lambda)     the same for H + lambda I (trust region,
//                                    Levenberg); H itself is not modified
//   newton_direction(g, d)           d = -(H + lambda I)^-1 g with the last factor
//   condition_estimate()             kappa_1 of the factorized matrix (Higham's
//                                    estimate, never above the true value
//                                    beyond rounding)
//
// The eigenvalue modification of Newton and the hard case of More-Sorensen
// use laplace::dense::symmetric_eigen_decomposition directly. The solver for
// an objective's type of H is curvature_solver_for<H>; a sparse solver
// joins it in phase 7.

#include "laplace/dense/dense.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace laplace::optimizers {

/**
 * \ingroup optimizers
 * \brief Cholesky-based solver and condition estimator for dense symmetric curvature matrices.
 *
 * Reads the lower triangle of a `dense_curvature_matrix` \f$H\f$ without modifying \f$H\f$, computes the Cholesky factor of \f$H\f$ or \f$H + \lambda I\f$ via \ref dense, solves \f$(H + \lambda I) d = -g\f$, and estimates \f$\kappa_1(H + \lambda I)\f$.
 * Programs using this class must link `Laplace::dense`.
 *
 * \see \ref optimizers, \ref dense, curvature_solver_for, curvature_factorization
 */
class dense_curvature_solver {
 public:
  /**
   * \ingroup optimizers
   * \brief Curvature matrix type accepted by this solver.
   */
  using matrix_type = dense_curvature_matrix;

  /**
   * \ingroup optimizers
   * \brief Constructs a solver with internal buffers for \f$n \times n\f$ matrices.
   *
   * \param n Problem dimension.
   */
  explicit dense_curvature_solver(int n)
      : shifted_("laplace::optimizers::dense_curvature_solver::shifted", n, n),
        factor_("laplace::optimizers::dense_curvature_solver::factor", n, n),
        rhs_("laplace::optimizers::dense_curvature_solver::rhs", n, 1) {}

  /**
   * \ingroup optimizers
   * \brief Returns the dimension \f$n\f$ this solver was constructed for.
   *
   * \return Problem dimension \f$n\f$.
   */
  int size() const { return static_cast<int>(factor_.extent(0)); }

  /**
   * \ingroup optimizers
   * \brief Factorizes \f$H = L L^\top\f$ from the lower triangle of `H`.
   *
   * Does not modify `H`. Throws `std::invalid_argument` if `H` is not \f$n \times n\f$.
   *
   * \param H Symmetric curvature matrix (lower triangle read).
   * \return Outcome of the factorization (`positive_definite`, `not_positive_definite`, or `non_finite`).
   */
  curvature_factorization factorize(const matrix_type& H) { return factorize_shifted(H, 0.0); }

  /**
   * \ingroup optimizers
   * \brief Factorizes \f$H + \lambda I = L L^\top\f$ from the lower triangle of `H`.
   *
   * Does not modify `H`. Throws `std::invalid_argument` if `H` is not \f$n \times n\f$.
   *
   * \param H Symmetric curvature matrix (lower triangle read).
   * \param lambda Diagonal shift \f$\lambda\f$ added to the diagonal of `H`.
   * \return Outcome of the factorization (`positive_definite`, `not_positive_definite`, or `non_finite`).
   */
  curvature_factorization factorize_shifted(const matrix_type& H, double lambda) {
    const int n = size();
    if (static_cast<int>(H.extent(0)) != n || static_cast<int>(H.extent(1)) != n)
      throw std::invalid_argument("laplace::optimizers::dense_curvature_solver: H does not match the solver's size");
    for (int j = 0; j < n; ++j)
      for (int i = j; i < n; ++i) shifted_(i, j) = H(i, j) + (i == j ? lambda : 0.0);
    const laplace::dense::dense_decomposition_status status = laplace::dense::cholesky_decomposition(shifted_, factor_);
    outcome_ = status.success ? curvature_factorization::positive_definite
               : status.reason == laplace::dense::dense_failure::non_finite_input
                   ? curvature_factorization::non_finite
                   : curvature_factorization::not_positive_definite;
    return outcome_;
  }

  /**
   * \ingroup optimizers
   * \brief Returns the outcome of the most recent factorization call.
   *
   * \return Last `curvature_factorization` status (`not_evaluated` before any factorization).
   */
  curvature_factorization last_factorization() const { return outcome_; }

  /**
   * \ingroup optimizers
   * \brief Solves \f$(H + \lambda I) d = -g\f$ using the last positive-definite Cholesky factor.
   *
   * Leaves `d` untouched and returns `false` if the last factorization did not succeed or if the triangular solve failed.
   *
   * \param g Gradient vector \f$g\f$ of length \f$n\f$.
   * \param d Output direction vector \f$d\f$ of length \f$n\f$.
   * \return `true` on success, `false` otherwise.
   */
  // d = -(H + lambda I)^-1 g with the last successful factor; false (d
  // untouched) when the last factorization did not succeed or the solve
  // failed.
  bool newton_direction(const optimizer_point& g, const optimizer_vector& d) {
    if (outcome_ != curvature_factorization::positive_definite) return false;
    const int n = size();
    for (int i = 0; i < n; ++i) rhs_(i, 0) = -g(i);
    if (!laplace::dense::cholesky_solve(factor_, rhs_).success) return false;
    for (int i = 0; i < n; ++i) d(i) = rhs_(i, 0);
    return true;
  }

  /**
   * \ingroup optimizers
   * \brief Returns the \f$\kappa_1\f$ condition estimate of the last factorized matrix \f$H + \lambda I\f$.
   *
   * Calls `laplace::dense::cholesky_condition_estimate()`.
   *
   * \return Estimated \f$\kappa_1\f$, or \f$+\infty\f$ when the last factorization was not `positive_definite`.
   */
  // kappa_1 of the last factorized matrix; inf when it was not positive definite.
  double condition_estimate() const {
    if (outcome_ != curvature_factorization::positive_definite) return std::numeric_limits<double>::infinity();
    return laplace::dense::cholesky_condition_estimate(shifted_, factor_);
  }

 private:
  matrix_type shifted_;  // the lower triangle of H + lambda I
  matrix_type factor_;
  matrix_type rhs_;
  curvature_factorization outcome_ = curvature_factorization::not_evaluated;
};

/**
 * \ingroup optimizers
 * \brief Trait mapping a curvature matrix type `H` to its curvature solver type.
 *
 * \tparam H Curvature matrix type.
 * \see \ref optimizers, dense_curvature_solver, dense_curvature_matrix
 */
// The curvature solver for a type of H.
template <class H>
struct curvature_solver_for;

/**
 * \ingroup optimizers
 * \brief Specialization of `curvature_solver_for` selecting `dense_curvature_solver` for `dense_curvature_matrix`.
 *
 * \see \ref optimizers, dense_curvature_solver, dense_curvature_matrix
 */
template <>
struct curvature_solver_for<dense_curvature_matrix> {
  /**
   * \ingroup optimizers
   * \brief Solver type `dense_curvature_solver`.
   */
  using type = dense_curvature_solver;
};

}  // namespace laplace::optimizers
