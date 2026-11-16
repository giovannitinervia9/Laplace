#pragma once

// Linear transforms of a univariate basis, B -> B T.
//
// A basis with K columns b_0, ..., b_(K-1) is replaced by the q columns of
// b(x)' T, where T is a K x q matrix, and its coefficients by beta = T gamma.
// The basis itself is not changed and keeps its sparse rows: T acts on the
// coefficient side. With T constant,
//   rows and derivatives:  T' b^(k)(x),    integrals: T' int b,
//   Gram matrices:         T' G_kl T,      penalties: T' S T,
//   gradients:             g_gamma = T' g_beta.
//
// make_basis_transform builds T from up to three inputs, always applied in
// this order (the canonical order), so that every property holds at the end:
//   1. constraints C (rows of C beta = 0): T1 = orthonormal basis of the null
//      space of C, so that C T = 0;
//   2. Gram matrix G: T2 = T1 L^-T with T1' G T1 = L L' (Cholesky), so that
//      T' G T = I;
//   3. penalty S: T = T2 W with T2' S T2 = W diag(lambda) W' (symmetric
//      eigendecomposition in the coordinates of step 2, which is the
//      Demmler-Reinsch basis when G is given), so that T' S T = diag(lambda),
//      lambda increasing. The caller declares r, the dimension of the null
//      space of S within the constraints; the r smallest eigenvalues are set to
//      exactly 0, and the build fails if one of them exceeds 1e-8 lambda_max.
// A step whose input is an empty View (0 rows) is skipped. A T built in any
// other way is passed to basis_transform_from_matrix, which declares no
// property.
//
// T is built on the host (laplace::dense is host only) and copied to the
// default device; expand_coefficients and reduce_gradient run in the execution
// space of the Views they are given, congruence on the host.

#include <laplace/dense/dense.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace laplace::basis {

/**
 * \ingroup basis_transform
 * \brief Step of make_basis_transform() at which a numerical failure occurred.
 *
 * Identifies which stage of the canonical build produced a failure status, or `none` on success.
 * \see \ref basis_transform, basis_transform_status, make_basis_transform()
 */
// The step of make_basis_transform that failed.
enum class basis_transform_step : int { none = 0, constraint = 1, orthonormalization = 2, demmler_reinsch = 3 };

/**
 * \ingroup basis_transform
 * \var basis_transform_step::none
 * \brief No failure occurred.
 */
/**
 * \ingroup basis_transform
 * \var basis_transform_step::constraint
 * \brief Failure in the constraint null-space SVD step.
 */
/**
 * \ingroup basis_transform
 * \var basis_transform_step::orthonormalization
 * \brief Failure in the Gram matrix Cholesky orthonormalization step.
 */
/**
 * \ingroup basis_transform
 * \var basis_transform_step::demmler_reinsch
 * \brief Failure in penalty eigendecomposition or the declared null-dimension check.
 */

/**
 * \ingroup basis_transform
 * \brief Status and diagnostics returned by basis_transform::status().
 *
 * Reports whether building the transform succeeded, which step failed, the underlying
 * `laplace::dense` failure reason, the numerical Gram rank on Cholesky failure, and the
 * maximum relative magnitude of the declared null eigenvalues before zeroing.
 * \see \ref basis_transform, basis_transform_step, make_basis_transform()
 */
// Outcome of make_basis_transform. On failure the transform is empty (q = 0).
//   dense_reason: the failure reported by laplace::dense in the failed step
//     (none when the declared null dimension did not hold);
//   gram_rank: when the Cholesky factorization of the Gram matrix failed, the
//     number of eigenvalues of T1' G T1 above q eps times the largest, out of
//     q (otherwise -1);
//   null_eigenvalue_ratio: the largest |lambda_j| / lambda_max over the r
//     declared null eigenvalues, before they are set to 0 (0 without a penalty).
struct basis_transform_status {
  /**
   * \ingroup basis_transform
   * \brief Whether building the transform succeeded (`false` leaves an empty transform with \f$q = 0\f$).
   */
  bool success = true;
  /**
   * \ingroup basis_transform
   * \brief Step of make_basis_transform() at which failure occurred (`basis_transform_step::none` on success).
   */
  basis_transform_step failed_step = basis_transform_step::none;
  /**
   * \ingroup basis_transform
   * \brief Failure reason reported by `laplace::dense`, or `none` on success or null-dimension mismatch.
   */
  laplace::dense::dense_failure dense_reason = laplace::dense::dense_failure::none;
  /**
   * \ingroup basis_transform
   * \brief Numerical rank of \f$T_1^\top G T_1\f$ above \f$q\varepsilon\f$ times the largest eigenvalue when Cholesky fails, otherwise \f$-1\f$.
   */
  int gram_rank = -1;
  /**
   * \ingroup basis_transform
   * \brief Maximum \f$\lvert\lambda_j\rvert / \lambda_{\max}\f$ over the \f$r\f$ declared null eigenvalues before zeroing (0 without a penalty).
   */
  double null_eigenvalue_ratio = 0.0;
};

namespace detail_transform {

using host_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

inline host_matrix identity(int n) {
  host_matrix o("laplace::basis::transform_identity", n, n);
  for (int i = 0; i < n; ++i) o(i, i) = 1.0;
  return o;
}

// a' b (ta) or a b, plain loops: K is at most a few hundred and this runs once.
template <class A, class B>
host_matrix product(const A& a, const B& b, bool ta) {
  const int r = ta ? a.extent(1) : a.extent(0), k = ta ? a.extent(0) : a.extent(1), c = b.extent(1);
  host_matrix o("laplace::basis::transform_product", r, c);
  for (int j = 0; j < c; ++j)
    for (int l = 0; l < k; ++l) {
      const double blj = b(l, j);
      if (blj == 0.0) continue;
      if (ta)
        for (int i = 0; i < r; ++i) o(i, j) += a(l, i) * blj;
      else
        for (int i = 0; i < r; ++i) o(i, j) += a(i, l) * blj;
    }
  return o;
}

// t' a t
template <class A>
host_matrix congruence(const host_matrix& t, const A& a) {
  return product(t, product(a, t, false), true);
}

template <class V>
constexpr void require_host_matrix() {
  static_assert(Kokkos::is_view_v<V> && V::rank() == 2 &&
                    std::is_same_v<typename V::non_const_value_type, double> &&
                    Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename V::memory_space>::accessible,
                "laplace::basis: a matrix must be a rank-2 View of double that the host can read");
}

// rows of t (K x q) times gamma (q) into beta (K), in the execution space of beta
template <class TView, class GammaView, class BetaView>
void expand(const TView& t, const GammaView& gamma, const BetaView& beta) {
  using exec = typename BetaView::execution_space;
  const int q = t.extent(1);
  Kokkos::parallel_for(
      "laplace::basis::expand_coefficients", Kokkos::RangePolicy<exec>(0, t.extent(0)), KOKKOS_LAMBDA(int i) {
        double s = 0.0;
        for (int j = 0; j < q; ++j) s += t(i, j) * gamma(j);
        beta(i) = s;
      });
}

// t' g_beta into g_gamma, in the execution space of g_gamma
template <class TView, class GView, class OutView>
void reduce(const TView& t, const GView& g, const OutView& out) {
  using exec = typename OutView::execution_space;
  const int K = t.extent(0);
  Kokkos::parallel_for(
      "laplace::basis::reduce_gradient", Kokkos::RangePolicy<exec>(0, t.extent(1)), KOKKOS_LAMBDA(int j) {
        double s = 0.0;
        for (int i = 0; i < K; ++i) s += t(i, j) * g(i);
        out(j) = s;
      });
}

}  // namespace detail_transform

/**
 * \ingroup basis_transform
 * \brief Linear coefficient transform \f$T \in \mathbb{R}^{K \times q}\f$ for a univariate basis.
 *
 * Stores the transformation matrix \f$T\f$ on the host and in the default Kokkos device memory space,
 * along with build status and penalty eigenvalues when built by make_basis_transform().
 * Provides coefficient expansion \f$\beta = T\gamma\f$, gradient reduction \f$g_\gamma = T^\top g_\beta\f$,
 * and host-side congruence \f$T^\top A T\f$.
 * \see \ref basis_transform, make_basis_transform(), basis_transform_from_matrix()
 */
class basis_transform {
 public:
  /**
   * \ingroup basis_transform
   * \brief Default Kokkos device memory space storing the device copy of \f$T\f$.
   */
  using device_memory = Kokkos::DefaultExecutionSpace::memory_space;
  /**
   * \ingroup basis_transform
   * \brief Host column-major View type (`Kokkos::LayoutLeft`, `Kokkos::HostSpace`) storing \f$T\f$.
   */
  using host_matrix = detail_transform::host_matrix;
  /**
   * \ingroup basis_transform
   * \brief Device matrix View type in `device_memory` storing \f$T\f$.
   */
  using device_matrix = Kokkos::View<double**, device_memory>;

  /**
   * \ingroup basis_transform
   * \brief Constructs an empty transform with \f$K = 0\f$, \f$q = 0\f$, and default success status.
   */
  basis_transform() = default;

  /**
   * \ingroup basis_transform
   * \brief Number of rows \f$K\f$ of \f$T\f$ (original basis dimension).
   * \return Row count \f$K\f$ (0 for a default-constructed or failed transform).
   * \see \ref basis_transform, columns()
   */
  int rows() const { return static_cast<int>(host_.extent(0)); }     // K
  /**
   * \ingroup basis_transform
   * \brief Number of columns \f$q\f$ of \f$T\f$ (transformed basis dimension).
   * \return Column count \f$q\f$ (0 for a default-constructed or failed transform).
   * \see \ref basis_transform, rows()
   */
  int columns() const { return static_cast<int>(host_.extent(1)); }  // q
  /**
   * \ingroup basis_transform
   * \brief Outcome and diagnostics from constructing the transform.
   * \return Const reference to the basis_transform_status struct.
   * \see \ref basis_transform, basis_transform_status
   */
  const basis_transform_status& status() const { return status_; }
  /**
   * \ingroup basis_transform
   * \brief Penalty eigenvalues \f$\lambda_1 \le \dots \le \lambda_q\f$ from the penalty step.
   *
   * The first \f$r\f$ entries are set to exactly zero.
   * \return Const reference to the vector of \f$q\f$ eigenvalues, or empty when no penalty was given or constructed via basis_transform_from_matrix().
   * \par Measured accuracy
   * Matches 200-bit mpmath references within \f$54.7\varepsilon\lambda_{\max}\f$ across eight exact-rational cubic B-spline cases (test bound \f$83\varepsilon\lambda_{\max}\f$); see \ref basis_transform.
   * \see \ref basis_transform, make_basis_transform()
   */
  // lambda, increasing, the r first exactly 0; empty when no penalty was given
  const std::vector<double>& penalty_eigenvalues() const { return eigenvalues_; }
  /**
   * \ingroup basis_transform
   * \brief Host copy of the transformation matrix \f$T\f$.
   * \return Const reference to the \f$K \times q\f$ `host_matrix` View.
   * \see \ref basis_transform, device_matrix_view()
   */
  const host_matrix& matrix() const { return host_; }
  /**
   * \ingroup basis_transform
   * \brief Device copy of the transformation matrix \f$T\f$.
   * \return Const reference to the \f$K \times q\f$ `device_matrix` View.
   * \see \ref basis_transform, matrix()
   */
  const device_matrix& device_matrix_view() const { return device_; }

  /**
   * \ingroup basis_transform
   * \brief Expands transformed coefficients into original basis coefficients, \f$\beta = T\gamma\f$.
   *
   * Computes \f$\beta_i = \sum_{j=0}^{q-1} T_{ij} \gamma_j\f$ for \f$i = 0, \dots, K - 1\f$ via a Kokkos kernel
   * in the execution space of `beta`, using the device copy of \f$T\f$ when accessible to `beta`'s memory space and the host copy otherwise.
   * \tparam GammaView Rank-1 Kokkos View type for input coefficients \f$\gamma\f$.
   * \tparam BetaView Rank-1 Kokkos View type for output coefficients \f$\beta\f$.
   * \param gamma Input vector of length \f$q\f$.
   * \param beta Output vector of length \f$K\f$.
   * \throw std::invalid_argument If `gamma.extent(0) != columns()` or `beta.extent(0) != rows()`.
   * \par Measured accuracy
   * Matches long-double reference summation within \f$4 q \varepsilon \sum_j \lvert T_{ij}\gamma_j\rvert\f$ in tests on host and device Views.
   * \see \ref basis_transform, reduce_gradient()
   */
  // beta = T gamma (gamma: q, beta: K), in the execution space of beta.
  template <class GammaView, class BetaView>
  void expand_coefficients(const GammaView& gamma, const BetaView& beta) const {
    check_extent(gamma.extent(0), columns(), "expand_coefficients: gamma must have q entries");
    check_extent(beta.extent(0), rows(), "expand_coefficients: beta must have K entries");
    using mem = typename BetaView::memory_space;
    if constexpr (Kokkos::SpaceAccessibility<mem, device_memory>::accessible)
      detail_transform::expand(device_, gamma, beta);
    else
      detail_transform::expand(host_, gamma, beta);
  }

  /**
   * \ingroup basis_transform
   * \brief Reduces a gradient with respect to \f$\beta\f$ into the transformed coordinates, \f$g_\gamma = T^\top g_\beta\f$.
   *
   * Computes \f$(g_\gamma)_j = \sum_{i=0}^{K-1} T_{ij} (g_\beta)_i\f$ for \f$j = 0, \dots, q - 1\f$ via a Kokkos kernel
   * in the execution space of `g_gamma`, using the device copy of \f$T\f$ when accessible to `g_gamma`'s memory space and the host copy otherwise.
   * \tparam GView Rank-1 Kokkos View type for input gradient \f$g_\beta\f$.
   * \tparam OutView Rank-1 Kokkos View type for output gradient \f$g_\gamma\f$.
   * \param g_beta Input gradient vector of length \f$K\f$.
   * \param g_gamma Output gradient vector of length \f$q\f$.
   * \throw std::invalid_argument If `g_beta.extent(0) != rows()` or `g_gamma.extent(0) != columns()`.
   * \par Measured accuracy
   * Matches long-double reference summation within \f$4 K \varepsilon \sum_i \lvert T_{ij}(g_\beta)_i\rvert\f$ in tests on device Views.
   * \see \ref basis_transform, expand_coefficients()
   */
  // g_gamma = T' g_beta (g_beta: K, g_gamma: q), in the execution space of g_gamma.
  template <class GView, class OutView>
  void reduce_gradient(const GView& g_beta, const OutView& g_gamma) const {
    check_extent(g_beta.extent(0), rows(), "reduce_gradient: g_beta must have K entries");
    check_extent(g_gamma.extent(0), columns(), "reduce_gradient: g_gamma must have q entries");
    using mem = typename OutView::memory_space;
    if constexpr (Kokkos::SpaceAccessibility<mem, device_memory>::accessible)
      detail_transform::reduce(device_, g_beta, g_gamma);
    else
      detail_transform::reduce(host_, g_beta, g_gamma);
  }

  /**
   * \ingroup basis_transform
   * \brief Computes the host congruence transform \f$T^\top A T\f$ of a \f$K \times K\f$ matrix.
   *
   * Reads the column-major \f$K \times K\f$ array `a` on the host and writes the column-major \f$q \times q\f$ result \f$T^\top A T\f$ into `out`.
   * \param a Pointer to the input column-major \f$K \times K\f$ matrix (for example from `cross_gram()` or `operator_penalty_matrix()`).
   * \param out Pointer to the output column-major \f$q \times q\f$ matrix.
   * \par Measured accuracy
   * In tests, \f$T^\top G T\f$ matches \f$I_q\f$ within \f$68\varepsilon\f$ and \f$T^\top S T\f$ matches \f$\operatorname{diag}(\lambda)\f$ within \f$64\varepsilon\lambda_{\max}\f$.
   * \see \ref basis_transform, make_basis_transform()
   */
  // out = T' A T on the host: A is K x K (a Gram matrix G_kl, a penalty S),
  // out is q x q, column-major (as cross_gram and operator_penalty_matrix write).
  void congruence(const double* a, double* out) const {
    const int K = rows(), q = columns();
    Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> av(a, K, K);
    const host_matrix c = detail_transform::congruence(host_, av);
    for (int j = 0; j < q; ++j)
      for (int i = 0; i < q; ++i) out[i + j * q] = c(i, j);
  }

  template <class CView, class GView, class SView>
  friend basis_transform make_basis_transform(int K, const CView& C, const GView& G, const SView& S, int r);
  template <class TView>
  friend basis_transform basis_transform_from_matrix(const TView& T);

 private:
  host_matrix host_;
  device_matrix device_;
  std::vector<double> eigenvalues_;
  basis_transform_status status_;

  static void check_extent(std::size_t got, int want, const char* what) {
    if (static_cast<int>(got) != want) throw std::invalid_argument(std::string("basis_transform: ") + what);
  }
  void set_matrix(const host_matrix& t) {
    host_ = host_matrix("laplace::basis::transform_matrix", t.extent(0), t.extent(1));
    Kokkos::deep_copy(host_, t);
    device_ = device_matrix("laplace::basis::transform_matrix_device", t.extent(0), t.extent(1));
    auto m = Kokkos::create_mirror_view(device_);
    for (std::size_t i = 0; i < t.extent(0); ++i)
      for (std::size_t j = 0; j < t.extent(1); ++j) m(i, j) = t(i, j);
    Kokkos::deep_copy(device_, m);
  }
  static basis_transform failed(basis_transform_step step, laplace::dense::dense_failure reason) {
    basis_transform b;
    b.status_.success = false;
    b.status_.failed_step = step;
    b.status_.dense_reason = reason;
    return b;
  }
};

/**
 * \ingroup basis_transform
 * \brief Constructs a basis_transform on the host in canonical order from constraints, Gram matrix, and penalty.
 *
 * Applies up to three steps in fixed canonical order: (1) null space \f$T_1\f$ of \f$C\f$ via SVD (\f$C T_1 = 0\f$),
 * (2) Gram orthonormalization \f$T_2 = T_1 L^{-\top}\f$ from \f$T_1^\top G T_1 = L L^\top\f$ (\f$T_2^\top G T_2 = I\f$), and
 * (3) symmetric eigendecomposition \f$T_2^\top S T_2 = W \operatorname{diag}(\lambda) W^\top\f$, setting \f$T = T_2 W\f$
 * with the \f$r\f$ smallest eigenvalues zeroed. Any input View with zero rows skips that step.
 * \tparam CView Host-readable rank-2 `Kokkos::LayoutLeft` View type of `double` for \f$C\f$.
 * \tparam GView Host-readable rank-2 `Kokkos::LayoutLeft` View type of `double` for \f$G\f$.
 * \tparam SView Host-readable rank-2 `Kokkos::LayoutLeft` View type of `double` for \f$S\f$.
 * \param K Dimension of the original univariate basis (\f$K \ge 1\f$).
 * \param C Constraint matrix of shape \f$m_c \times K\f$, or empty View to skip constraints.
 * \param G Symmetric positive definite Gram matrix of shape \f$K \times K\f$ (every entry is read), or empty View to skip orthonormalization.
 * \param S Symmetric positive semidefinite penalty matrix of shape \f$K \times K\f$ (every entry is read), or empty View to skip diagonalization.
 * \param r Declared null-space dimension of \f$S\f$ within the constraints (\f$0 \le r \le q\f$; ignored when `S` is empty).
 * \return Constructed basis_transform; numerical failures return an empty transform (\f$q = 0\f$) with diagnostics in `status()`.
 * \throw std::invalid_argument If \f$K < 1\f$, `C` does not have \f$K\f$ columns, `G` or `S` is non-empty and not \f$K \times K\f$, or `S` is non-empty and \f$r \notin [0, q]\f$.
 * \par Measured accuracy
 * Across eight exact-rational cubic B-spline cases, the worst errors over the gcc/OpenMP and CUDA builds are \f$54.7\varepsilon\lambda_{\max}\f$ for eigenvalues, \f$45.0\varepsilon\f$ for \f$T^\top G T - I\f$, \f$6.7\varepsilon\lambda_{\max}\f$ off-diagonal, and \f$75.1\varepsilon\max\lvert C\rvert\f$ for \f$C T\f$; see \ref basis_transform.
 * \see \ref basis_transform, basis_transform_from_matrix(), basis_transform_status
 */
// Canonical order: constraints C (m x K), Gram G (K x K), penalty S (K x K),
// each skipped when it is an empty View; r is the dimension of the null space
// of S within the constraints (ignored without S). Matrices are host-readable
// rank-2 Views of double, LayoutLeft; G and S are read whole and must be
// symmetric. Numerical failures come back in status(); wrong shapes throw
// std::invalid_argument.
template <class CView, class GView, class SView>
basis_transform make_basis_transform(int K, const CView& C, const GView& G, const SView& S, int r) {
  namespace ld = laplace::dense;
  using detail_transform::host_matrix;
  detail_transform::require_host_matrix<CView>();
  detail_transform::require_host_matrix<GView>();
  detail_transform::require_host_matrix<SView>();
  if (K < 1) throw std::invalid_argument("make_basis_transform: K must be >= 1");
  const bool has_c = C.extent(0) > 0, has_g = G.extent(0) > 0, has_s = S.extent(0) > 0;
  if (has_c && static_cast<int>(C.extent(1)) != K)
    throw std::invalid_argument("make_basis_transform: C must have K columns");
  if (has_g && (static_cast<int>(G.extent(0)) != K || static_cast<int>(G.extent(1)) != K))
    throw std::invalid_argument("make_basis_transform: G must be K x K");
  if (has_s && (static_cast<int>(S.extent(0)) != K || static_cast<int>(S.extent(1)) != K))
    throw std::invalid_argument("make_basis_transform: S must be K x K");

  // 1. constraints
  host_matrix t;
  if (has_c) {
    host_matrix c("laplace::basis::transform_C", C.extent(0), K);
    for (std::size_t i = 0; i < C.extent(0); ++i)
      for (int j = 0; j < K; ++j) c(i, j) = C(i, j);
    const ld::dense_decomposition_status s = ld::null_space_basis(c, t);
    if (!s.success) return basis_transform::failed(basis_transform_step::constraint, s.reason);
  } else {
    t = detail_transform::identity(K);
  }
  const int q = t.extent(1);
  if (has_s && (r < 0 || r > q)) throw std::invalid_argument("make_basis_transform: need 0 <= r <= q");

  // 2. orthonormalization: t <- t L^-T, with t' G t = L L'
  if (has_g && q > 0) {
    const host_matrix g1 = detail_transform::congruence(t, G);
    host_matrix l("laplace::basis::transform_L", q, q);
    const ld::dense_decomposition_status s = ld::cholesky_decomposition(g1, l);
    if (!s.success) {
      basis_transform b = basis_transform::failed(basis_transform_step::orthonormalization, s.reason);
      Kokkos::View<double*, Kokkos::HostSpace> w("laplace::basis::transform_gram_eigenvalues", q);
      host_matrix v("laplace::basis::transform_gram_vectors", q, q);
      if (ld::symmetric_eigen_decomposition(g1, w, v).success) {
        int rank = 0;
        const double tol = q * std::numeric_limits<double>::epsilon() * w(q - 1);
        for (int j = 0; j < q; ++j) rank += w(j) > tol;
        b.status_.gram_rank = rank;
      }
      return b;
    }
    host_matrix tt("laplace::basis::transform_Tt", q, K);  // t' -> L^-1 t'
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < q; ++j) tt(j, i) = t(i, j);
    ld::triangular_solve(l, tt, ld::dense_triangle::lower, ld::dense_operation::identity);
    for (int i = 0; i < K; ++i)
      for (int j = 0; j < q; ++j) t(i, j) = tt(j, i);
  }

  // 3. penalty: t <- t W, with t' S t = W diag(lambda) W'
  basis_transform b;
  if (has_s && q > 0) {
    const host_matrix s2 = detail_transform::congruence(t, S);
    Kokkos::View<double*, Kokkos::HostSpace> w("laplace::basis::transform_lambda", q);
    host_matrix v("laplace::basis::transform_W", q, q);
    const ld::dense_decomposition_status s = ld::symmetric_eigen_decomposition(s2, w, v);
    if (!s.success) return basis_transform::failed(basis_transform_step::demmler_reinsch, s.reason);
    double lmax = 0.0, ratio = 0.0;
    for (int j = 0; j < q; ++j) lmax = std::fmax(lmax, std::fabs(w(j)));
    for (int j = 0; j < r; ++j)
      if (lmax > 0.0) ratio = std::fmax(ratio, std::fabs(w(j)) / lmax);
    if (ratio > 1e-8) {
      basis_transform f = basis_transform::failed(basis_transform_step::demmler_reinsch, ld::dense_failure::none);
      f.status_.null_eigenvalue_ratio = ratio;
      return f;
    }
    b.eigenvalues_.resize(q);
    for (int j = 0; j < q; ++j) b.eigenvalues_[j] = j < r ? 0.0 : w(j);
    b.status_.null_eigenvalue_ratio = ratio;
    t = detail_transform::product(t, v, false);
  }
  b.set_matrix(t);
  return b;
}

/**
 * \ingroup basis_transform
 * \brief Wraps a caller-supplied transformation matrix \f$T\f$ in a basis_transform.
 *
 * Copies the \f$K \times q\f$ matrix `T` onto host and device storage with no structural properties declared,
 * `status().success` equal to `true`, and empty `penalty_eigenvalues()`.
 * \tparam TView Host-readable rank-2 View type of `double` for \f$T\f$.
 * \param T Caller-supplied \f$K \times q\f$ transformation matrix.
 * \return Initialized basis_transform holding a copy of `T`.
 * \throw std::invalid_argument If `T` has zero rows (\f$K < 1\f$).
 * \par Measured accuracy
 * Copies the entries of `T` to `matrix()` identically (exact entrywise match in tests).
 * \see \ref basis_transform, make_basis_transform()
 */
// A T given by the caller (host-readable K x q View of double); no property is declared.
template <class TView>
basis_transform basis_transform_from_matrix(const TView& T) {
  detail_transform::require_host_matrix<TView>();
  if (T.extent(0) < 1) throw std::invalid_argument("basis_transform_from_matrix: T must have at least one row");
  detail_transform::host_matrix t("laplace::basis::transform_given", T.extent(0), T.extent(1));
  for (std::size_t i = 0; i < T.extent(0); ++i)
    for (std::size_t j = 0; j < T.extent(1); ++j) t(i, j) = T(i, j);
  basis_transform b;
  b.set_matrix(t);
  return b;
}

}  // namespace laplace::basis
