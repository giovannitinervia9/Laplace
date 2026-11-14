// weighted_sum_psdmat: M(theta) = sum_j e^(theta_j) S_j, S_j fixed p x p
// positive semidefinite (note/pdmat_structures.md section 10, interface
// decided with Giovanni on 2026-11-14). Host only: it calls laplace::dense
// (link Laplace::dense). theta_j belongs to S_j; lambda_j = e^(theta_j).
//
// Supplied: M and its derivatives (e^(theta_j) S_j on a tuple of one repeated
// index j, zero otherwise); the log pseudo-determinant log|M|_+, the sum of
// the logarithms of the nonzero eigenvalues, with its derivatives of orders
// 1-6; the rank r and an orthonormal basis of the null space, constant in
// theta (the null space of a sum of positive semidefinite matrices is the
// intersection of theirs), from the null space of sum_j S_j / max|S_j| at
// construction.
//
// log|M|_+ by a graded similarity transform (Wood 2011, appendix B, with the
// partition by lambda_j ||S_j||): with U the basis of the subspace not yet
// resolved (at first I_p), each level
//   - drops the components whose restriction R_j = U^T S_j U is zero
//     (max|R_j| <= zero_tolerance max|S_j|): exactly zero on what is left;
//   - takes as dominant the components with
//     theta_j + log max|R_j| >= top + log split_tolerance, top the largest;
//   - splits U by the eigendecomposition of the dominant sum normalized,
//     sum_dom R_j / max|R_j|: U_+ its range, U_0 its null space; U_+ goes into
//     Q, the next level works on U U_0, and the dominant components are
//     exactly zero there (the structural zeros, set and not computed: a
//     computed zero costs lambda_dom eps, which swamps the subordinate
//     terms).
// The columns of Q span the range of M; level l has scale d_l = e^(top). With
// G_j = D^-1/2 Q^T (lambda_j S_j) Q D^-1/2 (D the diagonal of the d_l, the
// structural zeros set) and H = sum_j G_j, whose entries are of order one,
//   log|M|_+ = log det H + sum_l r_l log d_l,   r_l the columns of level l,
// with log det H from a Cholesky factorization. B_j = H^-1 G_j is similar to
// lambda_j M^+ S_j on the range, so d log|M|_+ / dtheta_j = tr B_j, and
// d B_j / dtheta_k = delta_jk B_j - B_k B_j; along a sorted tuple of K
// positions,
//   d^K log|M|_+ = sum over the set partitions pi of the positions whose
//     blocks each repeat one index, (-1)^(|pi| - 1) times the sum over the
//     (|pi| - 1)! cyclic orders of the blocks of tr(B_(j_1) ... B_(j_|pi|)).
// The route forms H from the scaled G_j and subtracts nothing, so a deeper
// recursion adds no rounding; what costs accuracy is a spread left unscaled
// inside one level. Hence the narrow levels (split_tolerance eps^0.05, where a
// route by Schur complements measured eps^0.35 best; docs/numerics.md).
// sum_j tr B_j = r is an exact identity (the tests check it). If the levels
// do not find r columns, or H is not positive definite, the log_pdet
// functions return NaN.
#pragma once

#include <laplace/dense/dense.hpp>
#include <laplace/pdmat/core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace laplace::pdmat {

namespace detail_weighted_sum {

using host_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>;
using scratch_matrix =
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using scratch_vector = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

constexpr double eps = std::numeric_limits<double>::epsilon();

// C (a x c) = A^T (a' x a, read as its transpose) times B (a' x c), column-major,
// leading dimensions lda, ldb, ldc
inline void mul_tn(int a, int c, int k, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
  for (int j = 0; j < c; ++j)
    for (int i = 0; i < a; ++i) {
      double s = 0.0;
      for (int l = 0; l < k; ++l) s += A[l + i * lda] * B[l + j * ldb];
      C[i + j * ldc] = s;
    }
}
// C (a x c) = A (a x k) B (k x c)
inline void mul_nn(int a, int c, int k, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
  for (int j = 0; j < c; ++j) {
    for (int i = 0; i < a; ++i) C[i + j * ldc] = 0.0;
    for (int l = 0; l < k; ++l) {
      const double b = B[l + j * ldb];
      for (int i = 0; i < a; ++i) C[i + j * ldc] += A[i + l * lda] * b;
    }
  }
}

// tr(B_(w_0) ... B_(w_(n-1))), B_j r x r column-major, P and P2 r x r scratch
inline double trace_of_word(const std::vector<int>& word, int r, const double* const* B, double* P, double* P2) {
  const int n = static_cast<int>(word.size());
  const double* left = B[word[0]];
  double tr = 0.0;
  if (n == 1) {
    for (int a = 0; a < r; ++a) tr += left[a + a * r];
    return tr;
  }
  if (n > 2) {
    mul_nn(r, r, r, left, r, B[word[1]], r, P, r);
    for (int s = 2; s < n - 1; ++s) {
      mul_nn(r, r, r, P, r, B[word[s]], r, P2, r);
      std::swap(P, P2);
    }
    left = P;
  }
  const double* last = B[word[n - 1]];
  for (int a = 0; a < r; ++a)
    for (int b = 0; b < r; ++b) tr += left[a + b * r] * last[b + a * r];
  return tr;
}

inline std::vector<int> least_rotation(const std::vector<int>& w) {
  std::vector<int> best = w, r = w;
  for (size_t s = 1; s < w.size(); ++s) {
    std::rotate(r.begin(), r.begin() + 1, r.end());
    if (r < best) best = r;
  }
  return best;
}

// d^K log det along the sorted tuple idx, from B_j = A^-1 dA/dtheta_j (any
// derivative of A along a tuple of one repeated index j equal to dA/dtheta_j,
// zero along a mixed tuple): the set partitions of the positions by
// restricted growth strings, the cyclic orders of their blocks by the
// permutations of all blocks but the first; each word of indices is reduced
// to its least rotation, and its trace computed once.
template <int K>
double log_det_derivative(const int (&idx)[K], int r, const double* const* B, double* P, double* P2) {
  std::map<std::vector<int>, double> coef;
  int a[K];  // the block of each position
  for (int q = 0; q < K; ++q) a[q] = 0;
  while (true) {
    int nb = 0;
    for (int q = 0; q < K; ++q) nb = std::max(nb, a[q] + 1);
    std::vector<int> label(nb, -1);
    bool uniform = true;
    for (int q = 0; q < K && uniform; ++q) {
      if (label[a[q]] < 0) label[a[q]] = idx[q];
      uniform = label[a[q]] == idx[q];
    }
    if (uniform) {
      const double sign = nb % 2 == 1 ? 1.0 : -1.0;
      std::vector<int> perm(nb);
      for (int b = 0; b < nb; ++b) perm[b] = b;
      do {
        std::vector<int> word(nb);
        for (int b = 0; b < nb; ++b) word[b] = label[perm[b]];
        coef[least_rotation(word)] += sign;
      } while (std::next_permutation(perm.begin() + 1, perm.end()));
    }
    // the next restricted growth string: a[0] = 0, a[q] <= 1 + max(a[0..q-1])
    int q = K - 1;
    for (; q > 0; --q) {
      int mxp = 0;
      for (int s = 0; s < q; ++s) mxp = std::max(mxp, a[s]);
      if (a[q] <= mxp) {
        ++a[q];
        for (int s = q + 1; s < K; ++s) a[s] = 0;
        break;
      }
    }
    if (q == 0) break;
  }
  double total = 0.0;
  for (const auto& [word, c] : coef)
    if (c != 0.0) total += c * trace_of_word(word, r, B, P, P2);
  return total;
}

}  // namespace detail_weighted_sum

/**
 * \ingroup psdmat
 * \brief Host-only positive semidefinite matrix formed as a weighted sum of fixed penalties.
 *
 * Maps \f$q = m\f$ unconstrained real parameters \f$\theta\f$ to
 * \f[
 * M(\theta) = \sum_{j=1}^{m} e^{\theta_j} S_j,
 * \f]
 * where \f$S_1, \dots, S_m\f$ are fixed \f$p \times p\f$ symmetric positive semidefinite matrices.
 * The rank \f$r\f$ and the null space do not depend on \f$\theta\f$ and are computed once at
 * construction. Every operation runs on the host via `laplace::dense` (link `Laplace::dense`).
 * No pseudo-inverse is supplied.
 *
 * \see psdmat, psdmat_structure, reduced_rank_psdmat
 */
// See the comment at the top of the file.
class weighted_sum_psdmat {
 public:
  /// \brief Host column-major matrix View type (`Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>`).
  using host_matrix = detail_weighted_sum::host_matrix;

  /// \brief Matrix dimension \f$p\f$.
  int dim = 0;
  /**
   * \ingroup psdmat
   * \brief Relative level-grouping threshold for the graded similarity transform.
   *
   * Defaults to \f$\epsilon^{0.05} \approx 0.165\f$, grouping active components whose
   * \f$\theta_j + \log\max_{ik}\lvert(R_j)_{ik}\rvert\f$ lies within \f$\log\f$ `split_tolerance`
   * of the level maximum.
   */
  // the partition threshold of the levels (eps^0.05, about 0.165: a level
  // holds the components within a factor of about 6 of the largest) and the
  // test of a zero component (relative to max|S_j|); both measured in
  // docs/numerics.md, the threshold chosen by Giovanni on 2026-11-14
  double split_tolerance = std::pow(detail_weighted_sum::eps, 0.05);
  /**
   * \ingroup psdmat
   * \brief Relative threshold below which a restricted component is dropped as zero.
   *
   * Initialized by the constructor to \f$8p\epsilon\f$ and applied relative to
   * \f$\max_{ik}\lvert(S_j)_{ik}\rvert\f$.
   */
  double zero_tolerance = 0.0;  // set by the constructor: 8 p eps

  /**
   * \ingroup psdmat
   * \brief Constructs a weighted-sum structure from fixed positive semidefinite matrices.
   *
   * Reads the lower triangles of \f$S_1, \dots, S_m\f$, copies the matrices, sets
   * `zero_tolerance` to \f$8p\epsilon\f$, and computes the constant rank \f$r\f$ and orthonormal
   * null-space basis via `laplace::dense::null_space_basis()` on
   * \f$\sum_{j=1}^m S_j / \max_{ik}\lvert(S_j)_{ik}\rvert\f$. Throws `std::invalid_argument` if `S`
   * is empty, the matrices are not square of one size, any matrix is zero or non-finite, any matrix
   * has an eigenvalue below \f$-4p\epsilon\max_{ik}\lvert(S_j)_{ik}\rvert\f$, or the null-space
   * decomposition fails.
   *
   * \tparam V Rank-2 `Kokkos::LayoutLeft` host-accessible View type of `double`.
   * \param S Non-empty vector of \f$p \times p\f$ symmetric positive semidefinite matrices.
   */
  // S: m >= 1 host Views of double, LayoutLeft, p x p; only their lower
  // triangles are read, and they are copied. Throws std::invalid_argument when
  // they are not square of one size or one of them is not positive
  // semidefinite (an eigenvalue below -4 p eps max|S_j|).
  template <class V>
  explicit weighted_sum_psdmat(const std::vector<V>& S) {
    static_assert(Kokkos::is_view_v<V> && V::rank() == 2 &&
                      std::is_same_v<typename V::non_const_value_type, double> &&
                      std::is_same_v<typename V::array_layout, Kokkos::LayoutLeft> &&
                      Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename V::memory_space>::accessible,
                  "weighted_sum_psdmat: each S_j must be a rank-2 LayoutLeft View of double the host can read");
    namespace d = laplace::dense;
    if (S.empty()) throw std::invalid_argument("weighted_sum_psdmat: at least one matrix is needed");
    const int p = static_cast<int>(S[0].extent(0));
    dim = p;
    zero_tolerance = 8.0 * p * detail_weighted_sum::eps;
    host_matrix sum("weighted_sum_psdmat::normalized_sum", p, p), vec("weighted_sum_psdmat::vectors", p, p);
    Kokkos::View<double*, Kokkos::HostSpace> val("weighted_sum_psdmat::values", p);
    for (const auto& s : S) {
      if (static_cast<int>(s.extent(0)) != p || static_cast<int>(s.extent(1)) != p)
        throw std::invalid_argument("weighted_sum_psdmat: the matrices must be square and of one size");
      host_matrix c("weighted_sum_psdmat::S", p, p);
      double mx = 0.0;
      for (int j = 0; j < p; ++j)
        for (int i = j; i < p; ++i) {
          c(i, j) = c(j, i) = s(i, j);
          mx = std::max(mx, std::abs(s(i, j)));
        }
      if (!(mx > 0.0) || !std::isfinite(mx))
        throw std::invalid_argument("weighted_sum_psdmat: a matrix is zero or not finite");
      const auto st = d::symmetric_eigen_decomposition(c, val, vec);
      if (!st.success || val(0) < -4.0 * p * detail_weighted_sum::eps * mx)
        throw std::invalid_argument("weighted_sum_psdmat: a matrix is not positive semidefinite");
      for (int j = 0; j < p; ++j)
        for (int i = 0; i < p; ++i) sum(i, j) += c(i, j) / mx;
      S_.push_back(c);
      max_abs_.push_back(mx);
    }
    const auto st = d::null_space_basis(sum, null_);
    if (!st.success) throw std::invalid_argument("weighted_sum_psdmat: the null space was not found");
    rank_ = st.rank;
  }

  /**
   * \ingroup psdmat
   * \brief Total number of parameters \f$q = m\f$.
   * \return Number of component matrices \f$m\f$.
   */
  int n_params() const { return static_cast<int>(S_.size()); }
  /**
   * \ingroup psdmat
   * \brief Constant matrix rank \f$r\f$ of \f$M(\theta)\f$.
   * \return Rank of \f$\sum_{j=1}^m S_j / \max_{ik}\lvert(S_j)_{ik}\rvert\f$ computed at construction.
   */
  int rank() const { return rank_; }
  /**
   * \ingroup psdmat
   * \brief Orthonormal basis of the constant null space of \f$M(\theta)\f$.
   *
   * Accuracy: tests check orthonormality within \f$64 \times 2.2 \times 10^{-16} p\f$ and
   * annihilation \f$\lvert S_j N\rvert \le 64 \times 2.2 \times 10^{-16} p \max_{ik}\lvert(S_j)_{ik}\rvert\f$.
   *
   * \return Host column-major View of shape \f$p \times (p - r)\f$ spanning the common null space.
   * \see rank(), psdmat
   */
  // orthonormal basis of the null space of M, p x (p - r)
  const host_matrix& null_space_basis() const { return null_; }
  /**
   * \ingroup psdmat
   * \brief Doubles of caller scratch buffer required by functions of this structure.
   *
   * Returns \f$(2m + 10)p^2 + 4p + 2m\f$ doubles.
   *
   * \return Required scratch size in doubles.
   */
  // the levels, the matrices G_j, B_j and H, and the products of the traces
  int scratch_size() const {
    const int p = dim, m = n_params();
    return (2 * m + 10) * p * p + 4 * p + 2 * m;
  }

  /**
   * \ingroup psdmat
   * \brief Evaluates the packed matrix \f$M(\theta) = \sum_{j=1}^m e^{\theta_j} S_j\f$.
   *
   * The third scratch argument is unused.
   *
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param out Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower-triangular storage.
   *
   * Accuracy: checked against the sum within relative tolerance \f$10^{-15}\f$.
   *
   * \see matrix_dk(), log_pdet(), psdmat
   */
  void matrix(const double* t, double* out, double* /*w*/) const {
    const int p = dim;
    for (int i = 0; i < packed_size(p); ++i) out[i] = 0.0;
    for (int j = 0; j < n_params(); ++j) {
      const double l = std::exp(t[j]);
      for (int i = 0; i < p; ++i)
        for (int k = 0; k <= i; ++k) out[packed_index(i, k)] += l * S_[j](i, k);
    }
  }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Writes \f$e^{\theta_j} S_j\f$ when all \f$K\f$ indices equal \f$j\f$, and returns \c false
   * without touching `out` along any mixed tuple. The fourth scratch argument is unused.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param idx Sorted tuple of \f$K\f$ 0-based parameter indices in \f$\{0, \dots, m - 1\}\f$.
   * \param out Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower-triangular storage.
   * \return \c true if written; \c false and buffer untouched if structurally zero.
   *
   * Accuracy: checked exact at order 3 along repeated indices in `tests/pdmat/test_weighted_sum.cpp`.
   *
   * \see matrix(), matrix_d1(), psdmat
   */
  template <int K>
  bool matrix_dk(const double* t, const int (&idx)[K], double* out, double* /*w*/) const {
    for (int q = 1; q < K; ++q)
      if (idx[q] != idx[0]) return false;
    const int p = dim, j = idx[0];
    const double l = std::exp(t[j]);
    for (int i = 0; i < p; ++i)
      for (int k = 0; k <= i; ++k) out[packed_index(i, k)] = l * S_[j](i, k);
    return true;
  }

  /**
   * \ingroup psdmat
   * \brief Evaluates the log pseudo-determinant \f$\log\lvert M(\theta)\rvert_+\f$.
   *
   * Uses the graded similarity transform described on \ref psdmat "the positive semidefinite structures page".
   * Returns quiet \c NaN if the transform fails to find \f$r\f$ columns or the Cholesky
   * factorization of the scaled sum fails.
   *
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles.
   * \return Sum of logarithms of the \f$r\f$ nonzero eigenvalues of \f$M(\theta)\f$, or quiet \c NaN on failure.
   *
   * Accuracy: up to 1.96 ulp on the tensor-product cases (test bound 8 ulp), 16.2 ulp on the
   * single-penalty case (bound 64 ulp), and \f$1.31 \times 10^4\f$ ulp on the ill-conditioned
   * low-rank case (bound \f$5.5 \times 10^4\f$ ulp) on gcc; the CUDA build gives the same errors.
   *
   * \see matrix(), log_pdet_dk(), rank_identity(), psdmat
   */
  double log_pdet(const double* t, double* w) const {
    double logdet;
    if (!transform(t, w, logdet, false)) return std::numeric_limits<double>::quiet_NaN();
    return logdet;
  }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$\log\lvert M(\theta)\rvert_+\f$ of order \f$K\f$.
   *
   * Evaluates the partition trace sum over \f$B_j = H^{-1} G_j\f$ described on
   * \ref psdmat "the positive semidefinite structures page". Returns quiet \c NaN if the graded
   * transform or Cholesky factorization fails.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param idx Sorted tuple of \f$K\f$ 0-based parameter indices in \f$\{0, \dots, m - 1\}\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles.
   * \return Partial derivative value, or quiet \c NaN on failure.
   *
   * Accuracy: over orders 1 to 6, up to \f$1.73 \times 10^4\f$ ulp on the tensor-product cases
   * (test bounds 4 to 70000 ulp) and \f$2.19 \times 10^4\f$ ulp on the ill-conditioned low-rank
   * case (test bounds \f$4.4 \times 10^4\f$ to \f$9 \times 10^4\f$ ulp) on gcc; the CUDA build gives the same errors.
   *
   * \see log_pdet(), log_pdet_d1(), rank_identity(), psdmat
   */
  template <int K>
  double log_pdet_dk(const double* t, const int (&idx)[K], double* w) const {
    double logdet;
    if (!transform(t, w, logdet, true)) return std::numeric_limits<double>::quiet_NaN();
    return derivative(idx, w);
  }

  /// \ingroup psdmat
  /// \brief Order-1 partial derivative \f$\partial M/\partial\theta_{i_1}\f$; forwards to `matrix_dk<1>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted index \f$i_1\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d1()
  bool matrix_d1(const double* t, const int (&i)[1], double* m, double* w) const { return matrix_dk<1>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-2 partial derivative \f$\partial^2 M/\partial\theta_{i_1}\partial\theta_{i_2}\f$; forwards to `matrix_dk<2>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d2()
  bool matrix_d2(const double* t, const int (&i)[2], double* m, double* w) const { return matrix_dk<2>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-3 partial derivative \f$\partial^3 M/\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}\f$; forwards to `matrix_dk<3>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2 \le i_3\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d3()
  bool matrix_d3(const double* t, const int (&i)[3], double* m, double* w) const { return matrix_dk<3>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-4 partial derivative \f$\partial^4 M/(\partial\theta_{i_1}\cdots\partial\theta_{i_4})\f$; forwards to `matrix_dk<4>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_4\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d4()
  bool matrix_d4(const double* t, const int (&i)[4], double* m, double* w) const { return matrix_dk<4>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-5 partial derivative \f$\partial^5 M/(\partial\theta_{i_1}\cdots\partial\theta_{i_5})\f$; forwards to `matrix_dk<5>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_5\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d5()
  bool matrix_d5(const double* t, const int (&i)[5], double* m, double* w) const { return matrix_dk<5>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-6 partial derivative \f$\partial^6 M/(\partial\theta_{i_1}\cdots\partial\theta_{i_6})\f$; forwards to `matrix_dk<6>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_6\f$. \param m Packed output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see matrix(), matrix_dk(), log_pdet_d6()
  bool matrix_d6(const double* t, const int (&i)[6], double* m, double* w) const { return matrix_dk<6>(t, i, m, w); }
  /// \ingroup psdmat
  /// \brief Order-1 partial derivative \f$\partial\log\lvert M\rvert_+/\partial\theta_{i_1}\f$; forwards to `log_pdet_dk<1>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted index \f$i_1\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d1()
  double log_pdet_d1(const double* t, const int (&i)[1], double* w) const { return log_pdet_dk<1>(t, i, w); }
  /// \ingroup psdmat
  /// \brief Order-2 partial derivative \f$\partial^2\log\lvert M\rvert_+/\partial\theta_{i_1}\partial\theta_{i_2}\f$; forwards to `log_pdet_dk<2>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d2()
  double log_pdet_d2(const double* t, const int (&i)[2], double* w) const { return log_pdet_dk<2>(t, i, w); }
  /// \ingroup psdmat
  /// \brief Order-3 partial derivative \f$\partial^3\log\lvert M\rvert_+/\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}\f$; forwards to `log_pdet_dk<3>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2 \le i_3\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d3()
  double log_pdet_d3(const double* t, const int (&i)[3], double* w) const { return log_pdet_dk<3>(t, i, w); }
  /// \ingroup psdmat
  /// \brief Order-4 partial derivative \f$\partial^4\log\lvert M\rvert_+/(\partial\theta_{i_1}\cdots\partial\theta_{i_4})\f$; forwards to `log_pdet_dk<4>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_4\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d4()
  double log_pdet_d4(const double* t, const int (&i)[4], double* w) const { return log_pdet_dk<4>(t, i, w); }
  /// \ingroup psdmat
  /// \brief Order-5 partial derivative \f$\partial^5\log\lvert M\rvert_+/(\partial\theta_{i_1}\cdots\partial\theta_{i_5})\f$; forwards to `log_pdet_dk<5>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_5\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d5()
  double log_pdet_d5(const double* t, const int (&i)[5], double* w) const { return log_pdet_dk<5>(t, i, w); }
  /// \ingroup psdmat
  /// \brief Order-6 partial derivative \f$\partial^6\log\lvert M\rvert_+/(\partial\theta_{i_1}\cdots\partial\theta_{i_6})\f$; forwards to `log_pdet_dk<6>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_6\f$. \param w Scratch buffer.
  /// \return Derivative value, or quiet \c NaN on failure. \see log_pdet(), log_pdet_dk(), matrix_d6()
  double log_pdet_d6(const double* t, const int (&i)[6], double* w) const { return log_pdet_dk<6>(t, i, w); }

  /**
   * \ingroup psdmat
   * \brief Evaluates the trace identity \f$\sum_{j=1}^m \operatorname{tr} B_j\f$.
   *
   * Equals `rank()` in exact arithmetic and provides a reference-free accuracy check. Returns
   * quiet \c NaN where `log_pdet()` returns \c NaN.
   *
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles.
   * \return Trace sum \f$\sum_{j=1}^m \operatorname{tr} B_j\f$, or quiet \c NaN on failure.
   *
   * Accuracy: agrees with `rank()` within \f$6.1 \times 10^{-14}\f$ across reference points (test bound \f$10^{-9} r\f$).
   *
   * \see rank(), log_pdet(), log_pdet_dk(), psdmat
   */
  // sum_j tr(B_j), equal to rank() in exact arithmetic: the check of the
  // derivatives that needs no reference (NaN where log_pdet is NaN)
  double rank_identity(const double* t, double* w) const {
    double logdet;
    if (!transform(t, w, logdet, true)) return std::numeric_limits<double>::quiet_NaN();
    const int r = rank_;
    double s = 0.0;
    for (int j = 0; j < n_params(); ++j)
      for (int a = 0; a < r; ++a) s += B(w, j)[a + a * r];
    return s;
  }

 private:
  std::vector<host_matrix> S_;
  std::vector<double> max_abs_;
  host_matrix null_;
  int rank_ = 0;

  // layout of w
  double* G(double* w, int j) const { return w + static_cast<size_t>(j) * dim * dim; }
  double* B(double* w, int j) const { return w + static_cast<size_t>(n_params() + j) * dim * dim; }
  double* tail(double* w, int k) const { return w + static_cast<size_t>(2 * n_params() + k) * dim * dim; }

  // Fills G_j (r x r, leading dimension r) and, with with_b, B_j; returns
  // false (NaN for the caller) when the levels or the Cholesky factor fail.
  bool transform(const double* t, double* w, double& logdet, bool with_b) const {
    namespace d = laplace::dense;
    using detail_weighted_sum::scratch_matrix;
    using detail_weighted_sum::scratch_vector;
    const int p = dim, m = n_params(), r = rank_;
    double* U = tail(w, 0);    // p x u, the subspace not yet resolved
    double* U2 = tail(w, 1);   // the next U
    double* Q = tail(w, 2);    // p x (columns so far)
    double* T = tail(w, 3);    // p x u, S_j U; then the normalized dominant sum
    double* V = tail(w, 4);    // eigenvectors
    double* R = tail(w, 5);    // the restrictions R_j, one at a time
    double* H = tail(w, 6);
    double* L = tail(w, 7);
    double* logd = tail(w, 10);     // p: the log scale of each column of Q
    double* ev = logd + p;          // p: eigenvalues
    double* mx = ev + p;            // m: max|R_j|
    double* cut = mx + m;           // m: the columns after which G_j is zero
    std::vector<char> active(m, 1);
    for (int i = 0; i < p * p; ++i) U[i] = 0.0;
    for (int i = 0; i < p; ++i) U[i + i * p] = 1.0;
    for (int j = 0; j < m; ++j) cut[j] = p;
    int u = p, nq = 0;
    while (u > 0) {
      // max|R_j| of the active components; a zero restriction drops j
      double top = -std::numeric_limits<double>::infinity();
      std::vector<double> scale(m, -std::numeric_limits<double>::infinity());
      for (int j = 0; j < m; ++j) {
        if (!active[j]) continue;
        restrict_to(j, U, u, T, R);
        double a = 0.0;
        for (int i = 0; i < u * u; ++i) a = std::max(a, std::abs(R[i]));
        mx[j] = a;
        if (!(a > zero_tolerance * max_abs_[j])) {
          active[j] = 0;
          cut[j] = nq;
          continue;
        }
        scale[j] = t[j] + std::log(a);
        top = std::max(top, scale[j]);
      }
      if (!(top > -std::numeric_limits<double>::infinity())) break;
      if (!std::isfinite(top)) return false;
      const double lim = top + std::log(split_tolerance);
      // the normalized dominant sum, u x u in T
      for (int i = 0; i < u * u; ++i) T[i] = 0.0;
      std::vector<int> dom;
      for (int j = 0; j < m; ++j) {
        if (!active[j] || scale[j] < lim) continue;
        dom.push_back(j);
        restrict_to(j, U, u, H, R);  // H as a temporary here
        for (int i = 0; i < u * u; ++i) T[i] += R[i] / mx[j];
      }
      scratch_matrix Tn(T, u, u), Vv(V, u, u);
      scratch_vector e(ev, u);
      if (!d::symmetric_eigen_decomposition(Tn, e, Vv).success) return false;
      int k = 0;  // the rank of the group, rule of laplace::dense: lambda > u eps lambda_max
      for (int i = 0; i < u; ++i)
        if (ev[i] > u * detail_weighted_sum::eps * ev[u - 1]) ++k;
      if (nq + k > r) return false;
      // U_+ = the last k eigenvectors (increasing order), U_0 the first u - k
      detail_weighted_sum::mul_nn(p, k, u, U, p, V + (u - k) * u, u, Q + nq * p, p);
      for (int c = 0; c < k; ++c) logd[nq + c] = top;
      nq += k;
      detail_weighted_sum::mul_nn(p, u - k, u, U, p, V, u, U2, p);
      std::copy(U2, U2 + p * (u - k), U);
      u -= k;
      for (int j : dom) {
        active[j] = 0;
        cut[j] = nq;
      }
    }
    if (nq != r) return false;
    // G_j = e^(theta_j - (logd_a + logd_b)/2) (Q^T S_j Q)_ab for a, b < cut_j
    for (int j = 0; j < m; ++j) {
      double* g = G(w, j);
      const int c = static_cast<int>(cut[j]) < r ? static_cast<int>(cut[j]) : r;
      for (int i = 0; i < r * r; ++i) g[i] = 0.0;
      if (c == 0) continue;
      detail_weighted_sum::mul_nn(p, c, p, S_[j].data(), p, Q, p, T, p);
      detail_weighted_sum::mul_tn(c, c, p, Q, p, T, p, R, c);
      for (int b = 0; b < c; ++b)
        for (int a = 0; a < c; ++a) g[a + b * r] = std::exp(t[j] - 0.5 * (logd[a] + logd[b])) * R[a + b * c];
    }
    for (int i = 0; i < r * r; ++i) H[i] = 0.0;
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < r * r; ++i) H[i] += G(w, j)[i];
    scratch_matrix Hm(H, r, r), Lm(L, r, r);
    if (!d::cholesky_decomposition(Hm, Lm).success) return false;
    logdet = 0.0;
    for (int a = 0; a < r; ++a) logdet += 2.0 * std::log(L[a + a * r]) + logd[a];
    if (!with_b) return true;
    for (int j = 0; j < m; ++j) {
      std::copy(G(w, j), G(w, j) + r * r, B(w, j));
      scratch_matrix Bm(B(w, j), r, r);
      d::triangular_solve(Lm, Bm, d::dense_triangle::lower, d::dense_operation::identity);
      d::triangular_solve(Lm, Bm, d::dense_triangle::lower, d::dense_operation::transpose);
    }
    return true;
  }

  // R (u x u) = U^T S_j U, with T (p x u) as a temporary
  void restrict_to(int j, const double* U, int u, double* T, double* R) const {
    const int p = dim;
    detail_weighted_sum::mul_nn(p, u, p, S_[j].data(), p, U, p, T, p);
    detail_weighted_sum::mul_tn(u, u, p, U, p, T, p, R, u);
  }

  template <int K>
  double derivative(const int (&idx)[K], double* w) const {
    std::vector<const double*> b(n_params());
    for (int j = 0; j < n_params(); ++j) b[j] = B(w, j);
    return detail_weighted_sum::log_det_derivative<K>(idx, rank_, b.data(), tail(w, 8), tail(w, 9));
  }
};

}  // namespace laplace::pdmat

static_assert(laplace::pdmat::psdmat_structure<laplace::pdmat::weighted_sum_psdmat>);
