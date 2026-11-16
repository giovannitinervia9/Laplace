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
//
// The derivatives are not computed from the B_j in double: the alternating
// signs of the partition sum lose digits that grow with the order (1.7e4 ulp
// at order 6 on te2), and the entries of a double B_j are already wrong by eps.
// log_pdet_state and log_pdet_dk rebuild B_j = H^-1 G_j in double-double from
// the G_j of the transform (H = sum_j G_j, its Cholesky factor and two
// triangular solves), and the products of the words, the traces and their
// weighted sum run in double-double too, which brings order 6 to 1.5 ulp on
// te2 and 2.5 on te3 (docs/numerics.md). The words come in lexicographic
// order, so the product of the first k matrices of a word is kept for the
// next word when it shares that prefix; the last factor only enters the trace.
// The double-double matrices live in the scratch buffer after the double ones
// (dd_region), hence the extra 2 (m + 4) r^2 doubles of scratch_size().
// log_pdet (order 0) and rank_identity do not build them.
#pragma once

#include <laplace/dense/dense.hpp>
#include <laplace/math/detail/double_double.hpp>
#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>

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
std::map<std::vector<int>, double> partition_words(const int (&idx)[K]) {
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
  return coef;
}

template <int K>
double log_det_derivative(const int (&idx)[K], int r, const double* const* B, double* P, double* P2) {
  double total = 0.0;
  for (const auto& [word, c] : partition_words<K>(idx))
    if (c != 0.0) total += c * trace_of_word(word, r, B, P, P2);
  return total;
}

using dd = laplace::math::detail::ddm::dd;

// C (r x r) = A B, double-double entries, column-major
inline void mul_dd(int r, const dd* A, const dd* B, dd* C) {
  for (int j = 0; j < r; ++j)
    for (int i = 0; i < r; ++i) {
      dd s(0.0);
      for (int l = 0; l < r; ++l) s = s + A[i + l * r] * B[l + j * r];
      C[i + j * r] = s;
    }
}

// d^K log det along the sorted tuple idx as log_det_derivative, with the matrices B_j, their
// products, the traces and the weighted sum of the words in double-double (B_j: m matrices of
// r x r double-double entries, one after the other). S holds four r x r buffers for the
// products: stack k (k = 1 to 4) is the product of the first k + 1 matrices of the word. The
// words come in lexicographic order, so the products of the first matrices of the previous
// word that are also a prefix of the next one are kept; the last factor only enters the trace.
template <int K>
double log_det_derivative_dd(const int (&idx)[K], int r, const dd* B, dd* S) {
  const size_t rr = static_cast<size_t>(r) * r;
  std::vector<int> prev;
  int have = 0;  // stack 1 to have are valid for the prefix of prev
  dd total(0.0);
  for (const auto& [word, c] : partition_words<K>(idx)) {
    if (c == 0.0) continue;
    const int n = static_cast<int>(word.size());
    const dd* left = B + word[0] * rr;
    dd tr(0.0);
    if (n == 1) {
      for (int a = 0; a < r; ++a) tr = tr + left[a + a * r];
    } else {
      if (n > 2) {
        int cp = 0;
        while (cp < n && cp < static_cast<int>(prev.size()) && prev[cp] == word[cp]) ++cp;
        have = std::max(0, std::min(have, cp - 1));
        for (int k = have + 1; k <= n - 2; ++k)
          mul_dd(r, k == 1 ? B + word[0] * rr : S + (k - 2) * rr, B + word[k] * rr, S + (k - 1) * rr);
        have = n - 2;
        prev = word;
        left = S + (n - 3) * rr;
      }
      const dd* last = B + word[n - 1] * rr;
      for (int a = 0; a < r; ++a)
        for (int b = 0; b < r; ++b) tr = tr + left[a + b * r] * last[b + a * r];
    }
    total = total + tr * c;
  }
  return total.hi + total.lo;
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
  /// \brief Compile-time descriptor: \c true, the common null space of \f$M(\theta)\f$ does not depend on \f$\theta\f$.
  // the null space, the intersection of those of the S_j, does not depend on
  // theta, and null_space_basis() returns an orthonormal basis of it
  static constexpr bool constant_null_space = true;
  /// \brief Compile-time descriptor: \c true, `null_space_basis()` returns an orthonormal basis of the constant null space.
  static constexpr bool supplies_null_space = true;
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
   * Returns \f$(2m + 10)p^2 + 4p + 2m + 2(m + 4)r^2\f$ doubles, where \f$p\f$ is `dim`,
   * \f$m\f$ is `n_params()`, and \f$r\f$ is `rank()`. The \f$2(m + 4)r^2\f$ tail holds the
   * \f$m\f$ double-double matrices \f$B_j\f$ and four \f$r \times r\f$ double-double product
   * buffers after the `double` workspace.
   *
   * \return Required scratch size in doubles.
   */
  // the levels, the matrices G_j, B_j and H, and the products of the traces,
  // then m + 4 double-double matrices r x r (B_j, four products; the first
  // product buffer holds the Cholesky factor of H while B_j is built)
  int scratch_size() const {
    const int p = dim, m = n_params(), r = rank_;
    return (2 * m + 10) * p * p + 4 * p + 2 * m + 2 * (m + 4) * r * r;
  }

  /**
   * \ingroup psdmat
   * \brief Computes starting parameters \f$\theta\f$ from a target packed matrix \f$\hat{M}\f$.
   *
   * Host-only function. Solves the normal equations in the Frobenius inner product for
   * \f$\lambda\f$ minimizing \f$\lVert \hat{M} - \sum_{j=1}^m \lambda_j S_j\rVert_F\f$,
   * floors each \f$\lambda_j \max_{ik}\lvert(S_j)_{ik}\rvert\f$ at \f$10^{-6}\f$ times the
   * largest (setting \f$\theta_j = 0\f$ if none is positive), and sets \f$\theta_j = \log\lambda_j\f$.
   * Exact up to rounding on matrices of the structure when the \f$S_j\f$ are linearly independent.
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output array of \c n_params() doubles inside the domain.
   *
   * Accuracy: recovers \f$\theta\f$ from `matrix()` within \f$2.1 \times 10^{-13}\f$ on three \f$6 \times 6\f$ difference penalties on gcc 14.2/OpenMP (test bound \f$10^{-12}\f$).
   *
   * \see matrix(), psdmat
   */
  // host: lambda_j by least squares on the entries of m_hat (Frobenius inner
  // product), each floored so that lambda_j max|S_j| >= start_eigen_floor
  // times the largest of these (all ones when none is positive), theta_j =
  // log lambda_j; exact, up to rounding, on matrix(theta) when the S_j are
  // linearly independent
  void params_from_matrix(const double* m_hat, double* theta) const {
    namespace dp = detail_pdmat;
    const int p = dim, m = n_params();
    auto dot = [&](auto&& a, auto&& b) {
      double s = 0.0;
      for (int i = 0; i < p; ++i)
        for (int k = 0; k <= i; ++k) s += (i == k ? 1.0 : 2.0) * a(i, k) * b(i, k);
      return s;
    };
    auto hat = [&](int i, int k) { return m_hat[packed_index(i, k)]; };
    std::vector<double> G(m * m), b(m);
    for (int j = 0; j < m; ++j) {
      b[j] = dot(S_[j], hat);
      for (int k = 0; k <= j; ++k) G[j * m + k] = G[k * m + j] = dot(S_[j], S_[k]);
    }
    const auto lam = dp::spd_solve_small(m, G, b);
    double top = 0.0;
    for (int j = 0; j < m; ++j) top = std::max(top, lam[j] * max_abs_[j]);
    for (int j = 0; j < m; ++j)
      theta[j] = top > 0.0 ? std::log(std::max(lam[j] * max_abs_[j], dp::start_eigen_floor * top) / max_abs_[j]) : 0.0;
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
   * Uses the graded similarity transform in `double` described on
   * \ref psdmat "the positive semidefinite structures page"; does not build the double-double
   * matrices \f$B_j\f$. Returns quiet \c NaN if the transform fails to find \f$r\f$ columns or
   * the Cholesky factorization of the scaled sum fails.
   *
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles.
   * \return Sum of logarithms of the \f$r\f$ nonzero eigenvalues of \f$M(\theta)\f$, or quiet \c NaN on failure.
   *
   * Accuracy: up to 1.96 ulp on the tensor-product cases (test bounds 1.7 and 3 ulp), 0.71 ulp on
   * the well-conditioned low-rank case (bound 1.1 ulp), 5.95 ulp on the ridge case (bound 9 ulp),
   * 16.2 ulp on the single-penalty case (bound 24.4 ulp), and \f$1.31 \times 10^4\f$ ulp on the
   * ill-conditioned low-rank case (bound \f$2 \times 10^4\f$ ulp) on gcc and the CUDA build.
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
   * Runs the graded similarity transform in `double`, rebuilds \f$B_j = H^{-1} G_j\f$ in
   * double-double from \f$G_j\f$ (forming \f$H = \sum_j G_j\f$, its Cholesky factor, and two
   * triangular solves in double-double), and evaluates the partition trace sum (word products in
   * lexicographic order with prefix sharing, traces, and weighted sum) in double-double before
   * rounding to `double`. Returns quiet \c NaN if the graded transform or either the `double` or
   * double-double Cholesky factorization of \f$H\f$ fails.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param idx Sorted tuple of \f$K\f$ 0-based parameter indices in \f$\{0, \dots, m - 1\}\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles.
   * \return Partial derivative value, or quiet \c NaN on failure.
   *
   * Accuracy: over orders 1 to 6 on gcc and the CUDA build, up to 2.5 ulp on the tensor-product
   * cases (test bounds 1 to 3.8 ulp), up to 10.9 ulp on the well-conditioned low-rank case
   * (bounds 6.8 to 16.4 ulp), up to 0.63 ulp on the ridge case and \f$2 \times 10^{-15}\f$ ulp on
   * the single-penalty case (bound 1 ulp), and \f$1.09 \times 10^4\f$ to \f$2.18 \times 10^4\f$
   * ulp on the ill-conditioned low-rank case (bounds \f$1.64 \times 10^4\f$ to \f$3.29 \times 10^4\f$ ulp).
   *
   * \see log_pdet(), log_pdet_d1(), rank_identity(), psdmat
   */
  template <int K>
  double log_pdet_dk(const double* t, const int (&idx)[K], double* w) const {
    double logdet;
    if (!transform(t, w, logdet, true) || !build_b_dd(w)) return std::numeric_limits<double>::quiet_NaN();
    return derivative(idx, w);
  }

  /**
   * \ingroup psdmat
   * \brief Evaluates \f$\log\lvert M(\theta)\rvert_+\f$ and stores the shared state at \f$\theta\f$ in `w`.
   *
   * Host only. Runs the graded similarity transform in `double`, builds \f$B_j = H^{-1} G_j\f$ in
   * double-double from \f$G_j\f$ (via \f$H = \sum_j G_j\f$, its Cholesky factor, and two
   * triangular solves in double-double; see \ref psdmat "the positive semidefinite structures page"),
   * and stores the double-double \f$B_j\f$ matrices in `w`. The state lives in the scratch buffer
   * itself, using no new memory. Any derivative at the same \f$\theta\f$ can then be read via
   * `log_pdet_dk_at()`, while `log_pdet()`, `log_pdet_dk()`, and `log_pdet_d1()` to
   * `log_pdet_d6()` do not use the state and recompute the transform on each call. Do not modify
   * `w` or reuse it for another \f$\theta\f$ while reading the state. Returns quiet \c NaN if the
   * transform fails to find \f$r\f$ columns or either the `double` or double-double Cholesky
   * factorization of \f$H\f$ fails; the state is then invalid.
   *
   * \param t Array of \f$m\f$ parameters \f$\theta\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles, holding the state on success.
   * \return Sum of logarithms of the \f$r\f$ nonzero eigenvalues of \f$M(\theta)\f$, or quiet \c NaN on failure.
   *
   * Accuracy: equals `log_pdet()` bit for bit on every reference point.
   *
   * \see log_pdet_dk_at(), log_pdet_dk(), log_pdet(), psdmat
   */
  double log_pdet_state(const double* t, double* w) const {
    double logdet;
    if (!transform(t, w, logdet, true) || !build_b_dd(w)) return std::numeric_limits<double>::quiet_NaN();
    return logdet;
  }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$\log\lvert M(\theta)\rvert_+\f$ of order \f$K\f$ from an evaluated state.
   *
   * Host only. Computes the partial derivative
   * \f$\partial^K\log\lvert M(\theta)\rvert_+/(\partial\theta_{i_1}\cdots\partial\theta_{i_K})\f$
   * along the sorted tuple `idx` \f$= (i_1, \dots, i_K)\f$ from the double-double matrices
   * \f$B_j\f$ stored in `w` by `log_pdet_state()`. Evaluates the word products (in lexicographic
   * order with prefix sharing), traces, and weighted partition sum in double-double, then rounds
   * to `double`. Reading does not modify the stored \f$B_j\f$ matrices, so any number of tuples
   * can be read while `w` is not modified or reused for another \f$\theta\f$. If `log_pdet_state()`
   * failed and returned quiet \c NaN, the state in `w` is invalid and `log_pdet_dk_at()` must not
   * be called on it.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param idx Sorted tuple of \f$K\f$ 0-based parameter indices in \f$\{0, \dots, m - 1\}\f$.
   * \param w Caller host scratch buffer of at least `scratch_size()` doubles from a successful `log_pdet_state()`.
   * \return Partial derivative value along `idx`.
   *
   * Accuracy: equals `log_pdet_dk()` bit for bit on every reference point, with the same accuracy and test bounds.
   *
   * \see log_pdet_state(), log_pdet_dk(), log_pdet_d1(), psdmat
   */
  template <int K>
  double log_pdet_dk_at(const int (&idx)[K], double* w) const {
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

  // the double-double region of w, after the double one: B_j (m matrices), then four product
  // buffers (the first one holds the Cholesky factor of H while B_j is built), each r x r with
  // double-double entries
  using dd = detail_weighted_sum::dd;
  dd* dd_region(double* w) const {
    const size_t used = static_cast<size_t>(2 * n_params() + 10) * dim * dim + 4 * dim + 2 * n_params();
    return reinterpret_cast<dd*>(w + used);
  }

  // B_j = H^-1 G_j in double-double from the G_j that transform left in w (the B_j in double
  // are not read): H = sum_j G_j, its Cholesky factor, two triangular solves. False if H is not
  // positive definite in double-double.
  bool build_b_dd(double* w) const {
    namespace ddm = laplace::math::detail::ddm;
    const int m = n_params(), r = rank_;
    const size_t rr = static_cast<size_t>(r) * r;
    dd* Bd = dd_region(w);
    dd* L = Bd + m * rr;  // the first product buffer, free until the derivatives are read
    for (size_t i = 0; i < rr; ++i) {
      dd h(0.0);
      for (int j = 0; j < m; ++j) h = h + G(w, j)[i];
      L[i] = h;
    }
    for (int j = 0; j < r; ++j) {
      dd d = L[j + j * r];
      for (int k = 0; k < j; ++k) d = d - L[j + k * r] * L[j + k * r];
      if (!(d.hi > 0.0)) return false;
      d = ddm::sqrt(d);
      L[j + j * r] = d;
      for (int i = j + 1; i < r; ++i) {
        dd s = L[i + j * r];
        for (int k = 0; k < j; ++k) s = s - L[i + k * r] * L[j + k * r];
        L[i + j * r] = s / d;
      }
    }
    for (int jm = 0; jm < m; ++jm) {
      dd* X = Bd + jm * rr;
      const double* g = G(w, jm);
      for (size_t i = 0; i < rr; ++i) X[i] = dd(g[i]);
      for (int c = 0; c < r; ++c) {
        for (int i = 0; i < r; ++i) {
          dd s = X[i + c * r];
          for (int k = 0; k < i; ++k) s = s - L[i + k * r] * X[k + c * r];
          X[i + c * r] = s / L[i + i * r];
        }
        for (int i = r - 1; i >= 0; --i) {
          dd s = X[i + c * r];
          for (int k = i + 1; k < r; ++k) s = s - L[k + i * r] * X[k + c * r];
          X[i + c * r] = s / L[i + i * r];
        }
      }
    }
    return true;
  }

  template <int K>
  double derivative(const int (&idx)[K], double* w) const {
    const size_t rr = static_cast<size_t>(rank_) * rank_;
    dd* Bd = dd_region(w);
    return detail_weighted_sum::log_det_derivative_dd<K>(idx, rank_, Bd, Bd + n_params() * rr);
  }
};

}  // namespace laplace::pdmat

static_assert(laplace::pdmat::psdmat_structure<laplace::pdmat::weighted_sum_psdmat>);
