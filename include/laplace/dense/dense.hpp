#pragma once

// Dense linear algebra on host Kokkos Views: Cholesky, triangular solve,
// pivoted QR, SVD, symmetric and generalized symmetric eigendecompositions,
// null space basis. Matrices are View<double**, LayoutLeft> (column-major, as
// Eigen and LAPACK), in a memory space the host can read; a View in CudaSpace
// is rejected at compile time. Symmetric inputs are read from their lower
// triangle only. The work is done by Eigen on one thread, in the static
// library laplace_dense (link Laplace::dense); this header does not include
// Eigen.

#include <Kokkos_Core.hpp>

#include <type_traits>

#include "laplace/dense/dense_status.hpp"
#include "laplace/dense/detail/dense_backend.hpp"

namespace laplace::dense {
namespace detail {

template <class V, unsigned R>
constexpr bool is_host_double_view() {
  if constexpr (!Kokkos::is_view_v<V>) {
    return false;
  } else {
    return V::rank() == R && std::is_same_v<typename V::non_const_value_type, double> &&
           Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename V::memory_space>::accessible;
  }
}

template <class V>
constexpr void require_input_matrix() {
  static_assert(is_host_double_view<V, 2>(),
                "laplace::dense: a matrix must be a rank-2 View of double that the host can read");
  static_assert(std::is_same_v<typename V::array_layout, Kokkos::LayoutLeft>,
                "laplace::dense: a matrix must have Kokkos::LayoutLeft");
}

template <class V>
constexpr void require_output_matrix() {
  require_input_matrix<V>();
  static_assert(!std::is_const_v<typename V::value_type>, "laplace::dense: an output matrix must not be const");
}

template <class V>
constexpr void require_output_vector() {
  static_assert(is_host_double_view<V, 1>(),
                "laplace::dense: a vector must be a rank-1 View of double that the host can read");
  static_assert(!std::is_const_v<typename V::value_type>, "laplace::dense: an output vector must not be const");
}

template <class V>
int leading_dimension(const V& a) {
  const int ld = static_cast<int>(a.stride(1));
  return ld > 0 ? ld : 1;
}

template <class V>
const_matrix_ref input_ref(const V& a) {
  return {a.data(), static_cast<int>(a.extent(0)), static_cast<int>(a.extent(1)), leading_dimension(a)};
}

template <class V>
matrix_ref output_ref(const V& a) {
  return {a.data(), static_cast<int>(a.extent(0)), static_cast<int>(a.extent(1)), leading_dimension(a)};
}

template <class V>
vector_ref output_vector_ref(const V& v) {
  return {v.data(), static_cast<int>(v.extent(0)), static_cast<int>(v.stride(0))};
}

}  // namespace detail

/**
 * \ingroup dense
 * \brief Cholesky decomposition of a symmetric positive definite matrix.
 *
 * Computes \f$A = L L^\top\f$ where \f$A\f$ is \f$n \times n\f$ symmetric positive definite and \f$L\f$ is \f$n \times n\f$ lower triangular with its strictly upper part zeroed.
 * Only the lower triangle of \f$A\f$ is read.
 *
 * \param A Input \f$n \times n\f$ symmetric matrix (lower triangle read).
 * \param L Output \f$n \times n\f$ lower-triangular matrix.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::not_positive_definite` if \f$A\f$ is not positive definite, or `dense_failure::non_finite_input` if the lower triangle of \f$A\f$ is non-finite.
 * Throws `std::invalid_argument` if \f$A\f$ is not square or \f$L\f$ does not match its shape; see \ref dense for View conventions.
 *
 * Measured residual \f$\|L L^\top - A\|_F / (\|A\|_F \varepsilon)\f$ is at most 0.62 \f$\varepsilon\f$ across backends (test bound 1 \f$\varepsilon\f$).
 *
 * \see \ref dense, triangular_solve(), dense_decomposition_status, dense_failure
 */
// A = L L^T with L lower triangular (n x n, its upper part set to zero).
// Fails with not_positive_definite when A is not positive definite.
template <class AView, class LView>
dense_decomposition_status cholesky_decomposition(const AView& A, const LView& L) {
  detail::require_input_matrix<AView>();
  detail::require_output_matrix<LView>();
  return detail::cholesky(detail::input_ref(A), detail::output_ref(L));
}

/**
 * \ingroup dense
 * \brief In-place triangular system solve.
 *
 * Solves \f$\operatorname{op}(T) X = B\f$ in place on the \f$n \times k\f$ matrix \f$B\f$, where \f$T\f$ is \f$n \times n\f$ triangular and \f$\operatorname{op}(T)\f$ is \f$T\f$ or \f$T^\top\f$.
 *
 * \param T Input \f$n \times n\f$ triangular matrix (only the triangle chosen by `triangle` is read).
 * \param B Input and output \f$n \times k\f$ matrix overwritten with \f$X\f$.
 * \param triangle Triangle of \f$T\f$ holding the factor (`dense_triangle::lower` or `dense_triangle::upper`).
 * \param operation Whether to apply \f$T\f$ (`dense_operation::identity`) or \f$T^\top\f$ (`dense_operation::transpose`).
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::singular` if any diagonal entry of \f$T\f$ is zero, or `dense_failure::non_finite_input` if any read entry is non-finite.
 * Throws `std::invalid_argument` if \f$T\f$ is not square or \f$B\f$ does not have \f$n\f$ rows; see \ref dense for View conventions.
 *
 * Measured residual \f$\|\operatorname{op}(T) X - B\|_F / (\|T\|_F \|X\|_F \varepsilon)\f$ is at most 0.44 \f$\varepsilon\f$ across backends (test bound 1 \f$\varepsilon\f$).
 *
 * \see \ref dense, cholesky_decomposition(), dense_decomposition_status, dense_failure, dense_triangle, dense_operation
 */
// Solves op(T) X = B in place on B (n x k), with T (n x n) lower or upper
// triangular (the other triangle is not read) and op the identity or the
// transpose. Fails with singular when a diagonal entry of T is zero.
template <class TView, class BView>
dense_decomposition_status triangular_solve(const TView& T, const BView& B, dense_triangle triangle,
                                            dense_operation operation) {
  detail::require_input_matrix<TView>();
  detail::require_output_matrix<BView>();
  return detail::triangular_solve(detail::input_ref(T), detail::output_ref(B), triangle, operation);
}

/**
 * \ingroup dense
 * \brief In-place linear solve from an existing Cholesky factor.
 *
 * Solves \f$A X = B\f$ in place on the \f$n \times k\f$ matrix \f$B\f$ for \f$A = L L^\top\f$ by two triangular solves: \f$L Y = B\f$ followed by \f$L^\top X = Y\f$.
 * Only the lower triangle of \f$L\f$ is read.
 *
 * \param L Input \f$n \times n\f$ lower-triangular Cholesky factor (lower triangle read).
 * \param B Input and output \f$n \times k\f$ matrix overwritten with \f$X\f$.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::singular` if any diagonal entry of \f$L\f$ is zero, or `dense_failure::non_finite_input` if any read entry of \f$L\f$ or \f$B\f$ is non-finite.
 * Throws `std::invalid_argument` if \f$L\f$ is not square or \f$B\f$ does not have \f$n\f$ rows; see \ref dense for View conventions.
 *
 * \see \ref dense, cholesky_decomposition(), triangular_solve(), spd_solve(), dense_decomposition_status, dense_failure
 */
// Solves A X = B in place on B (n x k) for A = L L^T, L the factor of
// cholesky_decomposition: two triangular solves. Fails with singular when a
// diagonal entry of L is zero.
template <class LView, class BView>
dense_decomposition_status cholesky_solve(const LView& L, const BView& B) {
  detail::require_input_matrix<LView>();
  detail::require_output_matrix<BView>();
  return detail::cholesky_solve(detail::input_ref(L), detail::output_ref(B));
}

/**
 * \ingroup dense
 * \brief In-place linear solve for a symmetric positive definite matrix.
 *
 * Factorizes \f$A = L L^\top\f$ via cholesky_decomposition() and solves \f$A X = B\f$ in place on \f$B\f$ via cholesky_solve().
 * Only the lower triangle of \f$A\f$ is read.
 *
 * \param A Input \f$n \times n\f$ symmetric positive definite matrix (lower triangle read).
 * \param L Output \f$n \times n\f$ lower-triangular Cholesky factor.
 * \param B Input and output \f$n \times k\f$ matrix overwritten with \f$X\f$.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::not_positive_definite` if \f$A\f$ is not positive definite, or `dense_failure::non_finite_input` if any read entry is non-finite.
 * Throws `std::invalid_argument` for incompatible shapes; see \ref dense for View conventions.
 *
 * Measured worst forward error over \f$n \le 200\f$ and \f$\kappa_2 \le 10^{12}\f$ is \f$0.336\,\kappa_2 \varepsilon\f$ on gcc and CUDA (test bound \f$0.5\,\kappa_2 \varepsilon\f$).
 *
 * \see \ref dense, cholesky_decomposition(), cholesky_solve(), cholesky_condition_estimate(), dense_decomposition_status, dense_failure
 */
// Solves A X = B in place on B (n x k) for symmetric positive definite A
// (lower triangle read): the factor goes to L (n x n, as
// cholesky_decomposition), then cholesky_solve. Fails with
// not_positive_definite when A is not positive definite.
template <class AView, class LView, class BView>
dense_decomposition_status spd_solve(const AView& A, const LView& L, const BView& B) {
  const dense_decomposition_status status = cholesky_decomposition(A, L);
  if (!status.success) return status;
  return cholesky_solve(L, B);
}

/**
 * \ingroup dense
 * \brief Estimate of the 1-norm condition number from a Cholesky factor.
 *
 * Estimates \f$\kappa_1(A) = \|A\|_1 \|A^{-1}\|_1\f$ for an \f$n \times n\f$ symmetric positive definite matrix \f$A\f$ (lower triangle read) and its Cholesky factor \f$L\f$ in \f$O(n^2)\f$ operations.
 * Computes \f$\|A\|_1\f$ exactly from the lower triangle and \f$\|A^{-1}\|_1\f$ by Higham's refinement of Hager's method using at most five sign-vector solves and, for \f$n > 1\f$, one solve with \f$x_i = (-1)^i (1 + i / (n - 1))\f$ (0-based \f$i\f$).
 *
 * \param A Input \f$n \times n\f$ symmetric matrix (lower triangle read).
 * \param L Input \f$n \times n\f$ lower-triangular Cholesky factor of \f$A\f$.
 * \return Estimate of \f$\kappa_1(A)\f$, at most \f$\kappa_1(A)\f$ up to rounding; `0.0` when \f$n = 0\f$, quiet NaN if the lower triangle of \f$A\f$ is non-finite, or \f$+\infty\f$ if \f$L\f$ has a zero or non-finite diagonal entry.
 *
 * Throws `std::invalid_argument` if \f$A\f$ is not square or \f$L\f$ does not match its shape; see \ref dense for View conventions.
 *
 * Measured ratio to exact \f$\kappa_1(A)\f$ lies in \f$[0.609, 1.0001]\f$ on gcc (\f$n \in [5, 300]\f$, \f$\kappa_2 \in [10^2, 10^{13}]\f$) and \f$[0.859, 1.00002]\f$ in tests on gcc and CUDA.
 *
 * \see \ref dense, cholesky_decomposition(), spd_solve()
 */
// An estimate of the 1-norm condition number kappa_1(A) = |A|_1 |A^-1|_1 of a
// symmetric positive definite A (lower triangle read) from its Cholesky factor
// L: |A|_1 is exact, |A^-1|_1 is Higham's estimate (Higham 1988, the method of
// LAPACK's dlacn2) from at most six solves with L, O(n^2) after the
// factorization. The estimate never exceeds kappa_1(A) beyond rounding.
// Returns inf when L has a zero or non-finite diagonal entry, NaN when the
// lower triangle of A is not finite, 0 when n = 0.
template <class AView, class LView>
double cholesky_condition_estimate(const AView& A, const LView& L) {
  detail::require_input_matrix<AView>();
  detail::require_input_matrix<LView>();
  return detail::cholesky_condition_estimate(detail::input_ref(A), detail::input_ref(L));
}

/**
 * \ingroup dense
 * \brief Column-pivoted QR decomposition.
 *
 * Computes \f$A P = Q R\f$ for an \f$m \times n\f$ matrix \f$A\f$ with \f$p = \min(m, n)\f$, where column \f$j\f$ of \f$Q R\f$ equals column \f$\operatorname{perm}[j]\f$ of \f$A\f$.
 * Numerical rank counts \f$|R_{kk}| > \text{tol} \cdot |R_{11}|\f$ (see \ref dense for default tolerance).
 *
 * \param A Input \f$m \times n\f$ matrix.
 * \param Q Output orthogonal matrix of shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full).
 * \param R Output upper-triangular matrix of shape \f$p \times n\f$ (thin) or \f$m \times n\f$ (full) matching `Q`.
 * \param perm Output rank-1 View of \c int of length \f$n\f$ receiving the column permutation.
 * \param relative_tolerance Threshold factor for rank determination; negative values (default `-1.0`) select \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank of \f$A\f$.
 *
 * Fails with `dense_failure::non_finite_input` if \f$A\f$ is non-finite; throws `std::invalid_argument` for incompatible shapes.
 *
 * Measured errors are at most 1.56 \f$\varepsilon\f$ for residual (bound 2.4 \f$\varepsilon\f$) and 14.12 \f$\varepsilon\f$ for orthonormality (bound 22 \f$\varepsilon\f$).
 *
 * \see \ref dense, svd_decomposition(), dense_decomposition_status, dense_failure
 */
// A P = Q R for A (m x n), p = min(m, n): Q is m x p (thin) or m x m (full),
// with orthonormal columns; R is upper triangular, p x n or m x n to match Q;
// perm (n entries) gives the permutation, column j of Q R being column perm[j]
// of A. The rank counts the |R_kk| above tol |R_11|, with
// tol = max(m, n) eps, or relative_tolerance when it is not negative.
template <class AView, class QView, class RView, class PermView>
dense_decomposition_status pivoted_qr_decomposition(const AView& A, const QView& Q, const RView& R,
                                                    const PermView& perm, double relative_tolerance = -1.0) {
  detail::require_input_matrix<AView>();
  detail::require_output_matrix<QView>();
  detail::require_output_matrix<RView>();
  static_assert(Kokkos::is_view_v<PermView> && PermView::rank() == 1 &&
                    std::is_same_v<typename PermView::value_type, int> &&
                    Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename PermView::memory_space>::accessible,
                "laplace::dense: perm must be a rank-1 View of int that the host can write");
  const detail::int_vector_ref p{perm.data(), static_cast<int>(perm.extent(0)), static_cast<int>(perm.stride(0))};
  return detail::pivoted_qr(detail::input_ref(A), detail::output_ref(Q), detail::output_ref(R), p,
                            relative_tolerance);
}

/**
 * \ingroup dense
 * \brief Singular value decomposition of a rectangular matrix.
 *
 * Computes \f$A = U \operatorname{diag}(s) V^\top\f$ for an \f$m \times n\f$ matrix \f$A\f$ with \f$p = \min(m, n)\f$ and decreasing singular values \f$s_1 \ge s_2 \ge \dots \ge s_p \ge 0\f$.
 * Numerical rank counts \f$s_i > \text{tol} \cdot s_1\f$ (see \ref dense for default tolerance).
 *
 * \param A Input \f$m \times n\f$ matrix.
 * \param U Output left singular vectors of shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full).
 * \param s Output rank-1 View of \c double of length \f$p\f$ receiving singular values.
 * \param V Output right singular vectors of shape \f$n \times p\f$ (thin) or \f$n \times n\f$ (full), independent of `U`.
 * \param relative_tolerance Threshold factor for rank determination; negative values (default `-1.0`) select \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank of \f$A\f$.
 *
 * Fails with `dense_failure::non_finite_input` or `dense_failure::no_convergence`; throws `std::invalid_argument` for incompatible shapes.
 *
 * Measured errors are at most 7.93 \f$\varepsilon\f$ for singular values (bound 12 \f$\varepsilon\f$), 14.39 \f$\varepsilon\f$ for residual (bound 22 \f$\varepsilon\f$), and 44.53 \f$\varepsilon\f$ for orthonormality (bound 67 \f$\varepsilon\f$).
 *
 * \see \ref dense, pivoted_qr_decomposition(), null_space_basis(), dense_decomposition_status, dense_failure
 */
// A = U diag(s) V^T for A (m x n), p = min(m, n): s has p entries in
// decreasing order; U is m x p or m x m and V is n x p or n x n, with
// orthonormal columns. The rank counts the s_i above tol s_1, with
// tol = max(m, n) eps, or relative_tolerance when it is not negative.
template <class AView, class UView, class SView, class VView>
dense_decomposition_status svd_decomposition(const AView& A, const UView& U, const SView& s, const VView& V,
                                             double relative_tolerance = -1.0) {
  detail::require_input_matrix<AView>();
  detail::require_output_matrix<UView>();
  detail::require_output_vector<SView>();
  detail::require_output_matrix<VView>();
  return detail::svd(detail::input_ref(A), detail::output_ref(U), detail::output_vector_ref(s),
                     detail::output_ref(V), relative_tolerance);
}

/**
 * \ingroup dense
 * \brief Symmetric eigendecomposition of a square symmetric matrix.
 *
 * Computes \f$A = V \operatorname{diag}(w) V^\top\f$ for an \f$n \times n\f$ symmetric matrix \f$A\f$ (lower triangle read), with increasing eigenvalues \f$w_1 \le w_2 \le \dots \le w_n\f$ and orthogonal \f$V\f$.
 *
 * \param A Input \f$n \times n\f$ symmetric matrix (lower triangle read).
 * \param w Output rank-1 View of \c double of length \f$n\f$ receiving eigenvalues.
 * \param V Output \f$n \times n\f$ orthogonal eigenvector matrix.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::non_finite_input` if the lower triangle of \f$A\f$ is non-finite, or `dense_failure::no_convergence` if the eigensolver fails to converge; throws `std::invalid_argument` for incompatible shapes.
 *
 * Measured errors are at most 12.2 \f$\varepsilon\f$ for eigenvalues (bound 19 \f$\varepsilon\f$), 9.92 \f$\varepsilon\f$ for residual (bound 15 \f$\varepsilon\f$), and 72.2 \f$\varepsilon\f$ for orthonormality at \f$n = 60\f$ (bound 110 \f$\varepsilon\f$).
 *
 * \see \ref dense, generalized_symmetric_eigen_decomposition(), cholesky_decomposition(), dense_decomposition_status, dense_failure
 */
// A = V diag(w) V^T for symmetric A (n x n), w in increasing order, V
// orthogonal.
template <class AView, class WView, class VView>
dense_decomposition_status symmetric_eigen_decomposition(const AView& A, const WView& w, const VView& V) {
  detail::require_input_matrix<AView>();
  detail::require_output_vector<WView>();
  detail::require_output_matrix<VView>();
  return detail::symmetric_eigen(detail::input_ref(A), detail::output_vector_ref(w), detail::output_ref(V));
}

/**
 * \ingroup dense
 * \brief Generalized symmetric-definite eigendecomposition.
 *
 * Solves \f$A V = B V \operatorname{diag}(w)\f$ with \f$V^\top B V = I\f$ and increasing eigenvalues \f$w\f$ by Cholesky reduction of \f$B\f$; only the lower triangles of \f$A\f$ and \f$B\f$ are read.
 *
 * \param A Input \f$n \times n\f$ symmetric matrix (lower triangle read).
 * \param B Input \f$n \times n\f$ symmetric positive definite matrix (lower triangle read).
 * \param w Output rank-1 View of \c double of length \f$n\f$ receiving generalized eigenvalues.
 * \param V Output \f$n \times n\f$ matrix of \f$B\f$-orthonormal eigenvectors.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * Fails with `dense_failure::not_positive_definite` if \f$B\f$ is not positive definite, `dense_failure::non_finite_input` if either lower triangle is non-finite, or `dense_failure::no_convergence`; throws `std::invalid_argument` for incompatible shapes.
 *
 * Measured eigenvalue error reaches 1270.3 \f$\varepsilon \kappa_i\f$ at condition \f$10^6\f$ (bound 1910 \f$\varepsilon \kappa_i\f$), with residual at most 0.19 \f$\varepsilon\f$ (bound 0.3 \f$\varepsilon\f$) and orthonormality at most 0.10 \f$\varepsilon\f$ (bound 0.15 \f$\varepsilon\f$); see \ref dense.
 *
 * \see \ref dense, symmetric_eigen_decomposition(), cholesky_decomposition(), dense_decomposition_status, dense_failure
 */
// A V = B V diag(w) with V^T B V = I, for symmetric A and symmetric positive
// definite B (n x n), w in increasing order. Fails with
// not_positive_definite when B is not positive definite.
template <class AView, class BView, class WView, class VView>
dense_decomposition_status generalized_symmetric_eigen_decomposition(const AView& A, const BView& B,
                                                                     const WView& w, const VView& V) {
  detail::require_input_matrix<AView>();
  detail::require_input_matrix<BView>();
  detail::require_output_vector<WView>();
  detail::require_output_matrix<VView>();
  return detail::generalized_symmetric_eigen(detail::input_ref(A), detail::input_ref(B),
                                             detail::output_vector_ref(w), detail::output_ref(V));
}

/**
 * \ingroup dense
 * \brief Orthonormal basis of the null space of a rectangular matrix.
 *
 * Computes an orthonormal basis \f$T\f$ satisfying \f$C T = 0\f$ and \f$T^\top T = I\f$ via Jacobi SVD of \f$C\f$ (\f$m \times n\f$), reallocating the managed View `T` to \f$n \times (n - r)\f$.
 * When \f$m = 0\f$, sets \f$T = I_{n \times n}\f$ and returns `rank = 0`; unmanaged `T` fails at compile time.
 *
 * \param C Input \f$m \times n\f$ constraint matrix.
 * \param T Output managed matrix View reallocated to shape \f$n \times (n - r)\f$.
 * \param relative_tolerance Threshold factor for rank determination; negative values (default `-1.0`) select \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank \f$r\f$.
 *
 * Fails with `dense_failure::non_finite_input` or `dense_failure::no_convergence`; see \ref dense for View and rank conventions.
 *
 * Measured errors are at most 1.74 \f$\varepsilon\f$ for residual (bound 2.7 \f$\varepsilon\f$) and 13.07 \f$\varepsilon\f$ for orthonormality (bound 20 \f$\varepsilon\f$).
 *
 * \see \ref dense, svd_decomposition(), dense_decomposition_status, dense_failure
 */
// An orthonormal basis T of the null space of C (m x n): C T = 0, T^T T = I.
// T is reallocated to n x (n - r), with r the rank of C (returned in the
// status) from its SVD with the rule of svd_decomposition. T must be a
// managed View; m = 0 gives T = I.
template <class CView, class TView>
dense_decomposition_status null_space_basis(const CView& C, TView& T, double relative_tolerance = -1.0) {
  detail::require_input_matrix<CView>();
  detail::require_output_matrix<TView>();
  static_assert(!TView::memory_traits::is_unmanaged, "laplace::dense: null_space_basis reallocates T, so T must be managed");
  const int n = static_cast<int>(C.extent(1));
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("laplace::dense::null_space_basis", n, n);
  const dense_decomposition_status status = detail::null_space(detail::input_ref(C), v.data(), relative_tolerance);
  if (!status.success) return status;
  const int r = status.rank;
  Kokkos::realloc(T, n, n - r);
  for (int j = 0; j < n - r; ++j)
    for (int i = 0; i < n; ++i) T(i, j) = v(i, r + j);
  return status;
}

}  // namespace laplace::dense
