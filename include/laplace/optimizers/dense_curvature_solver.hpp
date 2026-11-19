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
 * `dense_curvature_solver` operates on host views of type
 * `dense_curvature_matrix` for a fixed dimension \f$n\f$. Constructing
 * `dense_curvature_solver(n)` allocates three buffers: the shifted matrix
 * (\f$n \times n\f$), its Cholesky factor (\f$n \times n\f$), and a
 * right-hand-side vector (\f$n\f$ entries). All operations are host functions,
 * and a program using this class must link `Laplace::dense`. Used by
 * newton_optimizer, trust_region_optimizer, and proximal_newton_optimizer
 * (\ref optimizers_second_order, \ref optimizers_proximal).
 *
 * ### Factorization and solves
 *
 * Calling `factorize(H)` is equivalent to `factorize_shifted(H, 0.0)`.
 * `factorize_shifted(H, lambda)` copies the lower triangle of
 * \f$H + \lambda I\f$ without modifying `H`, computes the Cholesky
 * decomposition \f$H + \lambda I = L L^\top\f$ via \ref dense, and stores and
 * returns a `curvature_factorization` outcome (`positive_definite`,
 * `non_finite` if an input entry is non-finite, or `not_positive_definite`
 * otherwise). Passing a matrix whose dimensions are not \f$n \times n\f$
 * throws `std::invalid_argument`. `last_factorization()` returns the most
 * recent outcome (`not_evaluated` before the first call).
 *
 * After a `positive_definite` factorization, `newton_direction(g, d)` solves
 * \f$(H + \lambda I) d = -g\f$ using two triangular solves with the stored
 * factor and returns `true`; when the last factorization was not
 * `positive_definite` or the solve fails, it returns `false` and leaves `d`
 * untouched.
 *
 * ### Condition estimate
 *
 * `condition_estimate()` estimates \f$\kappa_1(A) = \lVert A \rVert_1 \lVert A^{-1} \rVert_1\f$
 * for the last factorized matrix \f$A = H + \lambda I\f$, or returns
 * \f$+\infty\f$ when `last_factorization()` is not `positive_definite`.
 * \f$\lVert A \rVert_1\f$ is computed exactly from the lower triangle, while
 * \f$\lVert A^{-1} \rVert_1\f$ uses Higham's estimate (Higham, 1988, Algorithm
 * 4.1) refining Hager's method (Hager, 1984): an iteration on sign vectors
 * with at most five solves, followed by one solve with
 * \f$x_i = (-1)^i (1 + i / (n - 1))\f$, keeping the largest norm found. Every
 * norm computed is \f$\lVert A^{-1} x \rVert_1\f$ for some \f$x\f$ with
 * \f$\lVert x \rVert_1 = 1\f$, so the estimate never exceeds \f$\kappa_1(A)\f$
 * beyond rounding; each solve costs \f$O(n^2)\f$.
 *
 * ### Measured accuracy and cost
 *
 * Measured with gcc on one thread (see `docs/numerics.md`), the ratio of the
 * condition estimate to the exact \f$\kappa_1\f$ lies in \f$[0.609, 1.0001]\f$
 * for \f$n\f$ from \f$5\f$ to \f$300\f$ and \f$\kappa\f$ from \f$10^2\f$ to
 * \f$10^{13}\f$, where the excess over \f$1\f$ comes from rounding of the
 * solves at \f$\kappa = 10^{13}\f$. At \f$n = 1000\f$, the estimate takes
 * \f$1.21\f$ ms, against \f$16.0\f$ ms for the Cholesky factorization and
 * \f$82.4\f$ ms for the inverse by \f$n\f$ solves.
 *
 * The test (`tests/optimizers/test_optimizers.cpp`, case `"dense curvature solver"`,
 * \f$n = 3\f$) checks \f$H d = -g\f$ and \f$(H + \lambda I) d = -g\f$ with a
 * relative tolerance of \f$10^{-14}\f$, a condition estimate between \f$1\f$
 * and \f$10\f$ on its positive-definite matrix, `not_positive_definite` with
 * `newton_direction()` returning `false` and an infinite estimate on an
 * indefinite matrix, `positive_definite` after a shift \f$\lambda = 2\f$ with
 * `H` unchanged, and `non_finite` on an infinite entry.
 *
 * ### Example
 *
 * Here `H` is a `dense_curvature_matrix`, `g` and `d` are `optimizer_vector`
 * objects of size `n`, and `lambda > 0` is a shift chosen by the caller:
 *
 * \code
 * laplace::optimizers::dense_curvature_solver solver(n);
 * if (solver.factorize(H) == laplace::optimizers::curvature_factorization::positive_definite) {
 *   solver.newton_direction(g, d);            // d = -H^{-1} g
 *   const double kappa = solver.condition_estimate();
 * } else {
 *   solver.factorize_shifted(H, lambda);      // try H + lambda I
 * }
 * \endcode
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
 * Exposes `using type = ...` for each supported curvature matrix type. Its one
 * specialization maps `dense_curvature_matrix` to `dense_curvature_solver`.
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
