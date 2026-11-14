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
 * Computes the Cholesky factorization
 * \f[
 * A = L L^\top
 * \f]
 * where \f$A\f$ is an \f$n \times n\f$ symmetric positive definite matrix and
 * \f$L\f$ is an \f$n \times n\f$ lower-triangular matrix.
 * Only the lower triangle of \f$A\f$ is read (its upper triangle is not accessed).
 * The strictly upper triangular part of \f$L\f$ is set to zero.
 *
 * \param A Input symmetric matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory. Only the lower triangle is read.
 * \param L Output lower-triangular matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * If \f$A\f$ is not positive definite, returns `success = false` with `reason = dense_failure::not_positive_definite`.
 * If a NaN or an infinity is found in the lower triangle of \f$A\f$, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * Non-finite entries in the strictly upper triangle are not read.
 * If \f$A\f$ is not square or \f$L\f$ does not match \f$A\f$'s shape, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: residual \f$\|L L^\top - A\|_F / (\|A\|_F \varepsilon)\f$ is \f$0.56\,\varepsilon\f$ on gcc and \f$0.62\,\varepsilon\f$ on CUDA (test bound 1 \f$\varepsilon\f$).
 *
 * \see triangular_solve(), dense_decomposition_status, dense_failure
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
 * Solves the linear system
 * \f[
 * \operatorname{op}(T) X = B
 * \f]
 * in place on \f$B\f$ (\f$n \times k\f$), where \f$T\f$ is an \f$n \times n\f$ triangular matrix.
 * When `triangle` is `dense_triangle::lower`, \f$T\f$ is lower triangular (upper triangle not read).
 * When `triangle` is `dense_triangle::upper`, \f$T\f$ is upper triangular (lower triangle not read).
 * When `operation` is `dense_operation::identity`, \f$\operatorname{op}(T) = T\f$.
 * When `operation` is `dense_operation::transpose`, \f$\operatorname{op}(T) = T^\top\f$.
 *
 * \param T Input triangular matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param B Input/output matrix of shape \f$n \times k\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory, overwritten with solution \f$X\f$.
 * \param triangle Which triangle of \f$T\f$ holds the factor (`dense_triangle::lower` or `dense_triangle::upper`).
 * \param operation Whether to apply \f$T\f$ or its transpose (`dense_operation::identity` or `dense_operation::transpose`).
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * If any diagonal entry of \f$T\f$ is zero, returns `success = false` with `reason = dense_failure::singular`.
 * If a non-finite entry is found in the referenced triangle of \f$T\f$ or in \f$B\f$, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * If \f$T\f$ is not square or \f$B\f$ does not have \f$n\f$ rows, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: residual \f$\|\operatorname{op}(T) X - B\|_F / (\|T\|_F \|X\|_F \varepsilon)\f$ across all four triangle and operation combinations is \f$0.33\,\varepsilon\f$ on gcc and \f$0.44\,\varepsilon\f$ on CUDA (test bound 1 \f$\varepsilon\f$).
 *
 * \see cholesky_decomposition(), dense_decomposition_status, dense_failure, dense_triangle, dense_operation
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
 * \brief Column-pivoted QR decomposition.
 *
 * Computes the column-pivoted QR decomposition
 * \f[
 * A P = Q R
 * \f]
 * for an \f$m \times n\f$ matrix \f$A\f$, with \f$p = \min(m, n)\f$.
 * Column \f$j\f$ of \f$Q R\f$ equals column \f$\operatorname{perm}[j]\f$ of \f$A\f$.
 * \f$Q\f$ has orthonormal columns and shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full).
 * \f$R\f$ is upper triangular of shape \f$p \times n\f$ (matching thin \f$Q\f$) or \f$m \times n\f$ (matching full \f$Q\f$).
 *
 * The numerical rank counts the diagonal entries \f$|R_{kk}|\f$ satisfying
 * \f[
 * |R_{kk}| > \text{tol} \cdot |R_{11}|
 * \f]
 * with \f$\text{tol} = \max(m, n)\,\varepsilon\f$ when `relative_tolerance < 0`, or `relative_tolerance` otherwise.
 *
 * \param A Input matrix of shape \f$m \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param Q Output orthogonal matrix of shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full), rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param R Output upper-triangular matrix of shape \f$p \times n\f$ or \f$m \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param perm Output column permutation vector of length \f$n\f$, rank-1 View of \c int in host memory.
 * \param relative_tolerance Threshold factor for rank determination; if negative (default -1.0), uses \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank of \f$A\f$.
 *
 * If \f$A\f$ contains non-finite entries, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * A low numerical rank is reported in `rank` and is not a failure.
 * If dimensions do not match the expected thin or full shapes, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: residual \f$\|Q R - A P\|_F / (\|A\|_F \varepsilon)\f$ is \f$1.55\,\varepsilon\f$ on gcc and \f$1.56\,\varepsilon\f$ on CUDA (test bound 2.4 \f$\varepsilon\f$). Loss of orthonormality \f$\|Q^\top Q - I\|_F / \varepsilon\f$ is \f$12.33\,\varepsilon\f$ on gcc and \f$14.12\,\varepsilon\f$ on CUDA (test bound 22 \f$\varepsilon\f$).
 *
 * \see svd_decomposition(), dense_decomposition_status, dense_failure
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
 * Computes the singular value decomposition
 * \f[
 * A = U \operatorname{diag}(s) V^\top
 * \f]
 * for an \f$m \times n\f$ matrix \f$A\f$, with \f$p = \min(m, n)\f$.
 * Singular values \f$s\f$ are in decreasing order \f$s_1 \ge s_2 \ge \dots \ge s_p \ge 0\f$.
 * \f$U\f$ has orthonormal columns and shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full).
 * \f$V\f$ has orthonormal columns and shape \f$n \times p\f$ (thin) or \f$n \times n\f$ (full).
 * Thin and full shapes for \f$U\f$ and \f$V\f$ may be chosen independently.
 *
 * The numerical rank counts the singular values satisfying
 * \f[
 * s_i > \text{tol} \cdot s_1
 * \f]
 * with \f$\text{tol} = \max(m, n)\,\varepsilon\f$ when `relative_tolerance < 0`, or `relative_tolerance` otherwise.
 *
 * \param A Input matrix of shape \f$m \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param U Output left singular vectors of shape \f$m \times p\f$ (thin) or \f$m \times m\f$ (full), rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param s Output singular values vector of length \f$p\f$, rank-1 View of \c double in host memory.
 * \param V Output right singular vectors of shape \f$n \times p\f$ (thin) or \f$n \times n\f$ (full), rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param relative_tolerance Threshold factor for rank determination; if negative (default -1.0), uses \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank of \f$A\f$.
 *
 * If \f$A\f$ contains non-finite entries, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * If the SVD algorithm fails to converge, returns `success = false` with `reason = dense_failure::no_convergence`.
 * A low numerical rank is reported in `rank` and is not a failure.
 * If dimensions do not match the expected thin or full shapes, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: singular values error \f$\max_i |s_i - s_{i,\text{ref}}| / (\varepsilon \max_i |s_{i,\text{ref}}|)\f$ is \f$7.93\,\varepsilon\f$ on gcc and CUDA (test bound 12 \f$\varepsilon\f$). Residual \f$\|U_p \operatorname{diag}(s) V_p^\top - A\|_F / (\|A\|_F \varepsilon)\f$ is \f$14.38\,\varepsilon\f$ on gcc and \f$14.39\,\varepsilon\f$ on CUDA (test bound 22 \f$\varepsilon\f$). Loss of orthonormality \f$\max(\|U^\top U - I\|_F, \|V^\top V - I\|_F) / \varepsilon\f$ is \f$44.17\,\varepsilon\f$ on gcc and \f$44.53\,\varepsilon\f$ on CUDA (test bound 67 \f$\varepsilon\f$).
 *
 * \see pivoted_qr_decomposition(), null_space_basis(), dense_decomposition_status, dense_failure
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
 * Computes the spectral decomposition
 * \f[
 * A = V \operatorname{diag}(w) V^\top
 * \f]
 * for an \f$n \times n\f$ real symmetric matrix \f$A\f$, where eigenvalues \f$w\f$ are in increasing order
 * \f$w_1 \le w_2 \le \dots \le w_n\f$ and \f$V\f$ is an \f$n \times n\f$ orthogonal matrix of eigenvectors (\f$V^\top V = I\f$).
 * Only the lower triangle of \f$A\f$ is read (its upper triangle is not accessed).
 *
 * \param A Input symmetric matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory. Only the lower triangle is read.
 * \param w Output eigenvalues vector of length \f$n\f$, rank-1 View of \c double in host memory.
 * \param V Output eigenvectors matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * If a non-finite entry is found in the lower triangle of \f$A\f$, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * Non-finite entries in the strictly upper triangle are not read.
 * If the eigensolver fails to converge, returns `success = false` with `reason = dense_failure::no_convergence`.
 * If \f$A\f$ or \f$V\f$ is not square or dimensions do not match, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: eigenvalues error \f$\max_i |w_i - w_{i,\text{ref}}| / (\varepsilon \max_i |w_{i,\text{ref}}|)\f$ is \f$12.2\,\varepsilon\f$ on gcc and CUDA (test bound 19 \f$\varepsilon\f$). Residual \f$\|A V - V \operatorname{diag}(w)\|_F / (\|A\|_F \varepsilon)\f$ is \f$9.85\,\varepsilon\f$ on gcc and \f$9.92\,\varepsilon\f$ on CUDA (test bound 15 \f$\varepsilon\f$). Loss of orthonormality \f$\|V^\top V - I\|_F / \varepsilon\f$ is \f$71.0\,\varepsilon\f$ on gcc and \f$72.2\,\varepsilon\f$ on CUDA at \f$n = 60\f$ (test bound 110 \f$\varepsilon\f$).
 *
 * \see generalized_symmetric_eigen_decomposition(), cholesky_decomposition(), dense_decomposition_status, dense_failure
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
 * Solves the generalized symmetric-definite eigenvalue problem
 * \f[
 * A V = B V \operatorname{diag}(w) \quad \text{with} \quad V^\top B V = I
 * \f]
 * for an \f$n \times n\f$ symmetric matrix \f$A\f$ and an \f$n \times n\f$ symmetric positive definite matrix \f$B\f$.
 * Eigenvalues \f$w\f$ are in increasing order \f$w_1 \le w_2 \le \dots \le w_n\f$.
 * Only the lower triangles of \f$A\f$ and \f$B\f$ are read.
 *
 * The problem is solved by Cholesky reduction using \f$B = L L^\top\f$, forming \f$\tilde{A} = L^{-1} A L^{-\top}\f$,
 * solving \f$\tilde{A} Y = Y \operatorname{diag}(w)\f$, and computing \f$V = L^{-\top} Y\f$.
 *
 * \param A Input symmetric matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory. Only the lower triangle is read.
 * \param B Input symmetric positive definite matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory. Only the lower triangle is read.
 * \param w Output generalized eigenvalues vector of length \f$n\f$, rank-1 View of \c double in host memory.
 * \param V Output \f$B\f$-orthonormal eigenvectors matrix of shape \f$n \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \return dense_decomposition_status with `rank = n` on success.
 *
 * If \f$B\f$ is not positive definite, returns `success = false` with `reason = dense_failure::not_positive_definite`.
 * If a non-finite entry is found in the lower triangle of \f$A\f$ or \f$B\f$, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * Non-finite entries in upper triangles are not read.
 * If the eigensolver fails to converge, returns `success = false` with `reason = dense_failure::no_convergence`.
 * If any matrix is not square or dimensions do not match, throws `std::invalid_argument`.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: in units of \f$\varepsilon \kappa_i\f$, eigenvalues error is within \f$1.4\,\varepsilon \kappa_i\f$ when \f$B\f$ has condition number \f$10^2\f$ or \f$10^3\f$, and reaches \f$1270.3\,\varepsilon \kappa_i\f$ on gcc and CUDA at zero eigenvalues when \f$B\f$ has condition number \f$10^6\f$ (test bound 1910 \f$\varepsilon \kappa_i\f$). Residual \f$\|A V - B V \operatorname{diag}(w)\|_F / ((\|A\|_F + \|B\|_F \max_i |w_i|) \|V\|_F \varepsilon)\f$ is \f$0.16\,\varepsilon\f$ on gcc and \f$0.19\,\varepsilon\f$ on CUDA (test bound 0.3 \f$\varepsilon\f$). Generalized orthonormality \f$\|V^\top B V - I\|_F / (\|V\|_F^2 \|B\|_F \varepsilon)\f$ is \f$0.07\,\varepsilon\f$ on gcc and \f$0.10\,\varepsilon\f$ on CUDA (test bound 0.15 \f$\varepsilon\f$).
 *
 * \see symmetric_eigen_decomposition(), cholesky_decomposition(), dense_decomposition_status, dense_failure
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
 * Computes an orthonormal basis \f$T\f$ of the null space of an \f$m \times n\f$ matrix \f$C\f$, satisfying
 * \f[
 * C T = 0 \quad \text{with} \quad T^\top T = I
 * \f]
 * The basis is formed from the right singular vectors of \f$C\f$ via Jacobi SVD.
 * The numerical rank \f$r\f$ of \f$C\f$ counts singular values satisfying \f$s_i > \text{tol} \cdot s_1\f$,
 * with \f$\text{tol} = \max(m, n)\,\varepsilon\f$ when `relative_tolerance < 0`, or `relative_tolerance` otherwise.
 * The output View \f$T\f$ is reallocated to shape \f$n \times (n - r)\f$.
 * \c T must be a managed View. When \f$m = 0\f$, \f$T\f$ is set to the \f$n \times n\f$ identity matrix and rank is 0.
 *
 * \param C Input constraint matrix of shape \f$m \times n\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory.
 * \param T Output managed View of shape \f$n \times (n - r)\f$, rank-2 View of \c double with layout `Kokkos::LayoutLeft` in host memory, reallocated by this function.
 * \param relative_tolerance Threshold factor for rank determination; if negative (default -1.0), uses \f$\max(m, n)\,\varepsilon\f$.
 * \return dense_decomposition_status with `rank` equal to the numerical rank \f$r\f$ of \f$C\f$.
 *
 * If \f$C\f$ contains non-finite entries, returns `success = false` with `reason = dense_failure::non_finite_input`.
 * If the SVD fails to converge, returns `success = false` with `reason = dense_failure::no_convergence`.
 * If \c TView is an unmanaged View, fails at compile time via static assertion.
 * When \f$m = 0\f$, returns `success = true` with `rank = 0` and \f$T = I_{n \times n}\f$.
 * Outputs are unspecified when `success` is \c false.
 *
 * Measured accuracy: residual \f$\|C T\|_F / (\|C\|_F \varepsilon)\f$ is \f$0.94\,\varepsilon\f$ on gcc and \f$1.74\,\varepsilon\f$ on CUDA (test bound 2.7 \f$\varepsilon\f$). Orthonormality \f$\|T^\top T - I\|_F / \varepsilon\f$ is \f$12.09\,\varepsilon\f$ on gcc and \f$13.07\,\varepsilon\f$ on CUDA (test bound 20 \f$\varepsilon\f$). Null space column count matched \f$n - r\f$ exactly in all test cases.
 *
 * \see svd_decomposition(), dense_decomposition_status, dense_failure
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
