// factor_analytic_pdmat: M = Lambda Lambda^T + Psi (note/pdmat_structures.md
// section 9; interface in core.hpp). Lambda is p x d, lower trapezoidal
// (Lambda_ij = 0 for j > i) with Lambda_jj = e^(theta) on its diagonal and the
// other loadings free; Psi = diag(psi_i), psi_i = e^(2 t_i). theta holds the
// loadings of the trapezoid by rows (row i has min(i + 1, d) of them, the
// last one the log of Lambda_ii when i < d), then t_0, ..., t_(p-1):
// q = pd - d(d-1)/2 + p. Precondition 0 <= d <= p, not checked; d = 0 gives
// M = Psi, and at d = p Psi is not identifiable (Ledermann's bound,
// factor_analytic_max_factors below).
//
// Values through Woodbury, with the d x d matrix
// C = I + Lambda^T Psi^-1 Lambda = I + sum_i e^(-2 t_i) lambda_i lambda_i^T
// (lambda_i the row i of Lambda) and C = R R^T:
//   log det M = 2 sum_i t_i + log det C,
//   M^-1 = Psi^-1 - Z^T Z, Z = R^-1 B^T, B = Psi^-1 Lambda;
// O(p d^2 + d^3) instead of the p^3 of a Cholesky factorization of M. At
// d = p, where Psi^-1 and B C^-1 B^T cancel (every row is explained by the
// factors), M^-1 comes from a Cholesky factorization of M.
// Derivatives along a sorted tuple F, with [S] the derivative along the
// positions in the mask S:
//   M: closed and sparse, as log_cholesky: a tuple of t_i alone reaches M_ii;
//     loadings reach M only on at most two variables of one column; a tuple
//     that mixes a t with a loading is zero;
//   log det M at d < p: C[T] is zero unless every index of T belongs to one
//     row r (its loadings or t_r), and then e^(-2 t_r) (-2)^(number of t_r)
//     times the Leibniz sum over the loadings of lambda_r lambda_r^T; with
//     X[S] = -X[0] sum_(T nonempty subset of S) C[T] X[S \ T] (X = C^-1),
//     d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)]),
//     f the first position; about 3^K d^3;
//   M^-1, and log det M at d = p (where Woodbury's terms cancel): by the
//     recursion on the augmented (p + d) x (p + d)
//     matrix A = [[Psi, Lambda], [Lambda^T, -I_d]], whose inverse is
//     G = A^-1 = [[M^-1, U], [U^T, -C^-1]], U = Psi^-1 Lambda C^-1 = M^-1 Lambda,
//     and log|det A| = log det M:
//       G[S] = -G[0] sum_(T nonempty subset of S) A[T] G[S \ T],
//       d^F M^-1 = the top-left p x p block of G[F],
//       d^F log det M = sum_(T subset of F \ f) tr(G[T] A[f + (F \ f \ T)]),
//     f the first position. A is linear in each loading, so A[T] is nonzero
//     only along one repeated parameter: 2^k psi_i at (i, i) for t_i, the
//     derivative of the loading (r, c) at (r, p + c) and (p + c, r); each
//     term costs O(n^2) and each G[S] one product with G[0] on at most
//     2 |T| rows. G[0] from its blocks: M^-1 as the value above, U and C^-1
//     as products, so that neither M^-1 Lambda nor C^-1 is formed by a
//     subtraction. The recursion on M itself (G[0] M[T] with M close to rank
//     d) lost up to 3e8 ulpN at psi = e^-8, Woodbury's derivatives up to
//     3e12 (docs/numerics.md, Giovanni 2026-11-16).
// The formulas were checked against mpmath's derivatives of the definition
// before the transcription (orders 0 to 6, docs/numerics.md).
#pragma once

#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>
#include <laplace/pdmat/detail/loadings.hpp>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with factor-analytic structure.
 *
 * Maps \f$q = pd - d(d - 1)/2 + p\f$ unconstrained real parameters to a \f$p \times p\f$ symmetric positive
 * definite matrix \f$M(\theta) = \Lambda\Lambda^\top + \Psi\f$, where \f$\Lambda\f$ is a \f$p \times d\f$ lower-trapezoidal
 * loading matrix with strictly positive diagonal entries \f$\Lambda_{jj} = e^\theta\f$ (\f$j < d\f$) and free
 * off-diagonal entries, and \f$\Psi = \operatorname{diag}(e^{2t_0}, \dots, e^{2t_{p-1}})\f$ is a diagonal matrix of
 * specific variances.
 *
 * Precondition: \f$0 \le d \le p\f$. For \f$d = 0\f$, \f$M = \Psi\f$ matches \ref laplace::pdmat::diagonal_pdmat "diagonal_pdmat".
 * At \f$d = p\f$, \f$\Psi\f$ is not identifiable, and \f$M - \Psi\f$ matches the log-Cholesky parameterization.
 *
 * \see pdmat, pdmat_orders, pdmat_structure, diagonal_pdmat, factor_analytic_max_factors()
 */
struct factor_analytic_pdmat : pdmat_orders<factor_analytic_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;        // p
  /// \brief Number of latent factors \f$d\f$ (\f$0 \le d \le p\f$).
  int n_factors = 0;  // d

  /// \brief Compile-time descriptor: \c false, diagonal entries are \f$\sum_k \Lambda_{ik}^2 + e^{2t_i}\f$.
  static constexpr bool unit_diagonal = false;
  /// \brief Compile-time descriptor: \c true, \f$M^{-1}\f$ is closed by Woodbury's identity for \f$d < p\f$ and by a Cholesky factorization of \f$M\f$ for \f$d = p\f$.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c false, no exact closed-form Cholesky factor is implemented.
  static constexpr bool closed_factor = false;
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1 and factors to 0.
   */
  factor_analytic_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs a factor-analytic structured matrix of dimension \f$p\f$ with \f$d\f$ factors.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   * \param d Number of latent factors (\f$0 \le d \le p\f$).
   */
  LAPLACE_HD factor_analytic_pdmat(int p, int d) : dim(p), n_factors(d) {}

  /**
   * \ingroup pdmat
   * \brief Number of free loading parameters \f$pd - d(d - 1)/2\f$.
   * \return Number of loading parameters in the lower trapezoid.
   */
  LAPLACE_HD int n_loadings() const { return dim * n_factors - n_factors * (n_factors - 1) / 2; }
  /**
   * \ingroup pdmat
   * \brief Total number of parameters \f$q = pd - d(d - 1)/2 + p\f$.
   * \return Number of loading parameters plus \f$p\f$ specific variance parameters.
   */
  LAPLACE_HD int n_params() const { return n_loadings() + dim; }

  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch buffer required by functions of this structure.
   *
   * Computes the total capacity:
   * \f[
   * 2pd + p + 2d^2 + 128d^2 + 2d + 64n^2 + n^2 + n
   * \f]
   * doubles with \f$n = p + d\f$, accommodating Woodbury's part and the workspace for the recursion on the augmented matrix \f$A\f$ (64 masks \f$G[S]\f$, one sum matrix, and \f$n\f$ row flags).
   *
   * \return Required scratch size in doubles.
   */
  // Woodbury's part: Lambda (p d), e^(-2 t) (p), a p x d temporary, the
  // factor R of C and C^-1 (d^2 each), C[T] and X[T] (64 d^2 each) and two
  // d-vectors; then the recursion on A (n = p + d): G[S] (64 n^2), a sum
  // (n^2) and n row flags
  LAPLACE_HD int scratch_size() const {
    const int n = dim + n_factors;
    return woodbury_size() + detail_pdmat::generic_masks * n * n + n * n + n;
  }

  /**
   * \ingroup pdmat
   * \brief Computes starting parameters \f$\theta\f$ from a target packed matrix \f$\hat{M}\f$.
   *
   * Host-only function. Initializes \f$\psi_i\f$ to the mean of the \f$p - d\f$ trailing
   * eigenvalues of \f$\hat{M}\f$ (probabilistic PCA fit, floored at \f$10^{-6}\f$ times
   * the largest diagonal entry), runs \c start_fa_iterations steps of principal axis
   * factoring updating \f$\Lambda\f$ from the \f$d\f$ leading eigenpairs of \f$\hat{M} - \Psi\f$
   * and \f$\Psi = \operatorname{diag}(\hat{M} - \Lambda\Lambda^\top)\f$ (floored), and
   * converts the loadings to lower-trapezoidal form via the first \f$d\f$ Cholesky columns
   * of \f$AA^\top\f$.
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output array of \c n_params() doubles inside the domain.
   *
   * Accuracy: on \f$(p, d) = (8, 2)\f$, relative matrix distance \f$\max|M(\hat\theta) - \hat{M}|/\max|\hat{M}|\f$ reaches \f$1.1 \times 10^{-3}\f$ on a matrix of its own structure (test bound 0.05) and 0.17 on a generic positive definite matrix on gcc 14.2/OpenMP. These errors are expected: the result is a starting value for the optimizer, which refines it, and not a fit of the factor model to \f$\hat{M}\f$.
   *
   * \see matrix(), start_fa_iterations, pdmat
   */
  // host: principal axis factoring, start_fa_iterations steps from the
  // probabilistic PCA fit (psi the mean of the trailing eigenvalues): Lambda
  // the leading d eigenvectors of M_hat - Psi scaled by the square roots of
  // their eigenvalues, then Psi = diag(M_hat - Lambda Lambda^T); eigenvalues
  // and psi_i floored; Lambda to the trapezoidal convention
  void params_from_matrix(const double* m_hat, double* theta) const {
    namespace dp = detail_pdmat;
    const int p = dim, d = n_factors;
    const auto dg = dp::floored_diagonal(p, m_hat);
    const double top = *std::max_element(dg.begin(), dg.end()), low = dp::start_eigen_floor * top;
    std::vector<double> lam, V, A(p * d), e(m_hat, m_hat + packed_size(p)), psi(p);
    dp::eigen_decreasing(p, m_hat, lam, V);
    double s2 = 0.0;
    for (int k = d; k < p; ++k) s2 += lam[k] / (p - d);
    for (int i = 0; i < p; ++i) psi[i] = std::max(d < p ? s2 : 0.0, low);
    for (int it = 0; it < start_fa_iterations && d > 0; ++it) {
      for (int i = 0; i < p; ++i) e[packed_index(i, i)] = m_hat[packed_index(i, i)] - psi[i];
      dp::eigen_decreasing(p, e.data(), lam, V);
      for (int i = 0; i < p; ++i)
        for (int k = 0; k < d; ++k) A[i * d + k] = V[i * p + k] * std::sqrt(std::max(lam[k], low));
      for (int i = 0; i < p; ++i) {
        double r = m_hat[packed_index(i, i)];
        for (int k = 0; k < d; ++k) r -= A[i * d + k] * A[i * d + k];
        psi[i] = std::max(r, low);
      }
    }
    if (d > 0) dp::trapezoid_params(p, d, A.data(), theta);
    for (int i = 0; i < p; ++i) theta[n_loadings() + i] = 0.5 * std::log(psi[i]);
  }
  /// \brief Number of principal axis factoring iterations used by `params_from_matrix()` (100).
  // the steps of principal axis factoring in params_from_matrix
  static constexpr int start_fa_iterations = 100;

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed factor-analytic matrix \f$M(\theta)\f$.
   *
   * Computes \f$M = \Lambda\Lambda^\top + \Psi\f$ in packed lower-triangular storage.
   * For \f$d = 0\f$, evaluates \f$M = \Psi\f$, matching \ref laplace::pdmat::diagonal_pdmat "diagonal_pdmat".
   *
   * \param theta Array of \f$q\f$ parameters (loadings followed by \f$t_0, \dots, t_{p-1}\f$).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Accuracy: up to 1.01 ulpN on gcc and CUDA (test bound 2 ulpN).
   *
   * \see inverse(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const {
    const int p = dim, d = n_factors, nl = n_loadings();
    double* lam = w;
    load_lambda(theta, lam);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) s += lam[i * d + k] * lam[j * d + k];
        if (i == j) s += detail_pdmat::m::exp(2.0 * theta[nl + i]);
        m[packed_index(i, j)] = s;
      }
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * Computes \f$\log \det M = 2\sum_{i=0}^{p-1} t_i + \log \det C\f$ via Woodbury's identity,
   * where \f$C = I_d + \Lambda^\top\Psi^{-1}\Lambda = R R^\top\f$ (\f$d \times d\f$).
   * For \f$d = 0\f$, computes \f$2\sum_i t_i\f$.
   *
   * \param theta Array of \f$q\f$ parameters.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * Accuracy on gcc and CUDA: up to 3.5 ulp for \f$d < p\f$ (test bound 6 ulp), and 35 ulp on gcc and 32 ulp on CUDA for \f$d = p\f$ (test bound 53 ulp).
   *
   * \see matrix(), inverse(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, double* w) const {
    const int p = dim, d = n_factors, nl = n_loadings();
    const work s = carve(w);
    load_lambda(theta, s.lam);
    load_em2t(theta, s.em2t);
    build_c(s, s.R);
    detail_pdmat::fa_cholesky(d, s.R);
    double ld = 0.0, lc = 0.0;
    for (int i = 0; i < p; ++i) ld += theta[nl + i];
    for (int k = 0; k < d; ++k) lc += detail_pdmat::m::log(s.R[k * d + k]);
    return 2.0 * ld + 2.0 * lc;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Computes \f$M^{-1} = \Psi^{-1} - Z^\top Z\f$ with \f$Z = R^{-1}\Lambda^\top\Psi^{-1}\f$ and
   * \f$C = I_d + \Lambda^\top\Psi^{-1}\Lambda = R R^\top\f$ via Woodbury's identity for \f$d < p\f$, and via a Cholesky
   * factorization of \f$M\f$ for \f$d = p\f$, in packed lower-triangular storage.
   * For \f$d = 0\f$, computes \f$M^{-1} = \Psi^{-1}\f$.
   *
   * \param theta Array of \f$q\f$ parameters.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Accuracy: up to 178 ulpN on gcc and 374 ulpN on CUDA (test bound 561 ulpN).
   *
   * \see matrix(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, double* w) const {
    const int p = dim, d = n_factors;
    if (d == p && d > 0) return cholesky_inverse(theta, m, w);
    const work s = carve(w);
    load_lambda(theta, s.lam);
    load_em2t(theta, s.em2t);
    build_c(s, s.R);
    detail_pdmat::fa_cholesky(d, s.R);
    double* Z = s.tmp;  // row i: R^-1 b_i
    double* b = s.u;
    for (int i = 0; i < p; ++i) {
      for (int k = 0; k < d; ++k) b[k] = s.em2t[i] * s.lam[i * d + k];
      detail_pdmat::fa_forward(d, s.R, b, Z + i * d);
    }
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) {
        double r = 0.0;
        for (int k = 0; k < d; ++k) r += Z[i * d + k] * Z[j * d + k];
        m[packed_index(i, j)] = (i == j ? s.em2t[i] : 0.0) - r;
      }
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates closed-form sparse derivatives. Returns \c true when computed, or \c false
   * (leaving buffer untouched) when structurally zero:
   * - A tuple consisting of a single index \f$t_i\f$ repeated \f$K\f$ times reaches only entry \f$M_{ii}\f$ (\f$2^K e^{2t_i}\f$).
   * - Loadings reach \f$M\f$ only along at most two variables of the same column.
   * - Mixed tuples between a loading and \f$t\f$, tuples across different columns, or with
   *   three distinct variables are structurally zero.
   * - Derivatives of off-diagonal loadings of order 2 or higher vanish; diagonal loadings \f$e^\theta\f$ have derivative \f$e^\theta\f$ at all orders.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles (unused).
   * \return \c true if computed; \c false and buffer left untouched if structurally zero.
   *
   * Accuracy over orders 1–6: up to 0.49 ulpN on gcc and 0.41 ulpN on CUDA (test bound 1 ulpN).
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    const int p = dim, nl = n_loadings();
    if (idx[0] >= nl) {  // a tuple of t alone (sorted: the loadings come first)
      for (int k = 1; k < K; ++k)
        if (idx[k] != idx[0]) return false;
      const int r = idx[0] - nl;
      detail_pdmat::fill_zero(m, packed_size(p));
      m[packed_index(r, r)] = detail_pdmat::ipow(2.0, K) * detail_pdmat::m::exp(2.0 * theta[idx[0]]);
      return true;
    }
    if (idx[K - 1] >= nl) return false;  // a t with a loading
    return tz().template gram_dk<K>(theta, idx, m);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$.
   *
   * For \f$d < p\f$, evaluates derivatives via Woodbury's identity, with \f$C = I_d + \Lambda^\top\Psi^{-1}\Lambda\f$, \f$X = C^{-1}\f$,
   * \f$C[T]\f$ and \f$X[T]\f$ their derivatives along the positions of the tuple \f$F\f$ in the subset \f$T\f$, \f$f\f$ the
   * first position of \f$F\f$, and \f$X[S] = -X[\emptyset]\sum_{\emptyset \ne T \subseteq S} C[T]\,X[S \setminus T]\f$:
   * \f[
   * \partial^F \log\det C = \sum_{T \subseteq F \setminus \{f\}} \operatorname{tr}\bigl(X[T]\,C[\{f\} \cup (F \setminus \{f\} \setminus T)]\bigr)
   * \f]
   * plus 2 for a single parameter \f$t_i\f$ at order 1. For \f$d = p\f$, evaluates the same sum of traces on the augmented
   * \f$(p + d) \times (p + d)\f$ matrix \f$A = \bigl[\begin{smallmatrix}\Psi & \Lambda \\ \Lambda^\top & -I_d\end{smallmatrix}\bigr]\f$,
   * \f[
   * \partial^F \log\det M = \sum_{T \subseteq F \setminus \{f\}} \operatorname{tr}\bigl(G[T]\,A[\{f\} \cup (F \setminus \{f\} \setminus T)]\bigr),
   * \f]
   * with \f$G = A^{-1} = \bigl[\begin{smallmatrix}M^{-1} & U \\ U^\top & -C^{-1}\end{smallmatrix}\bigr]\f$ (\f$U = \Psi^{-1}\Lambda C^{-1} = M^{-1}\Lambda\f$)
   * and its derivatives \f$G[T]\f$ from \f$G[S] = -G[\emptyset]\sum_{\emptyset \ne T \subseteq S} A[T]\,G[S \setminus T]\f$ (forming only masks within \f$F \setminus \{f\}\f$).
   * For \f$d = 0\f$, returns 2 for a single \f$t_i\f$ at \f$K = 1\f$ and \f$0.0\f$ otherwise.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero.
   *
   * Accuracy over orders 1–6: for \f$d < p\f$, up to 795 ulp on gcc and 926 ulp on CUDA (test bounds 7 to 1390 ulp); for \f$d = p\f$, up to 4141 ulp on gcc and 794 ulp on CUDA (test bounds 692 to 6211 ulp).
   *
   * \see log_det(), inverse_dk(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], double* w) const {
    const int nl = n_loadings(), d = n_factors, n = dim + d, nn = n * n;
    const double psi_part = (K == 1 && idx[0] >= nl) ? 2.0 : 0.0;
    if (d == 0) return psi_part;
    constexpr int NM = 1 << K;
    if (d < dim) {  // Woodbury: the recursion on C and its traces
      const work s = carve(w);
      bool cnz[NM], xnz[NM];
      const detail_pdmat::trapezoid_loadings z = tz();
      z.load_lambda(theta, s.lam);
      load_em2t(theta, s.em2t);
      detail_pdmat::loadings_gram_masks<K>(z, theta, idx, s.lam, s.em2t, true, [&](int v) { return param_row(v); }, s.C, cnz, s.u, s.v);
      detail_pdmat::log_det_masks_inverse<K>(d, s.C, cnz, s.R, s.tmp, s.X, xnz);
      return psi_part + detail_pdmat::log_det_masks_sum<K>(d, s.C, cnz, s.X, xnz);
    }
    // d = p: Woodbury's terms cancel, the traces on A
    const int rest = (NM - 1) ^ 1;
    bool gz[NM];
    augmented_recursion<K>(theta, idx, rest, w, gz);
    const double* G = w + woodbury_size();
    double r = 0.0;
    for (int T = rest;; T = (T - 1) & rest) {
      int i, j;
      double x;
      if (gz[T] && augmented_dk<K>(theta, idx, 1 | (rest ^ T), i, j, x)) {
        const double* g = G + T * nn;
        r += x * (j == i ? g[i * n + i] : g[i * n + j] + g[j * n + i]);
      }
      if (T == 0) break;
    }
    return r;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * For \f$d \ge 1\f$, evaluates derivatives via the recursion on the augmented \f$(p + d) \times (p + d)\f$ matrix
   * \f$A = \bigl[\begin{smallmatrix}\Psi & \Lambda \\ \Lambda^\top & -I_d\end{smallmatrix}\bigr]\f$, whose inverse is
   * \f$G = A^{-1} = \bigl[\begin{smallmatrix}M^{-1} & U \\ U^\top & -C^{-1}\end{smallmatrix}\bigr]\f$ with \f$U = \Psi^{-1}\Lambda C^{-1} = M^{-1}\Lambda\f$
   * and \f$\log\lvert\det A\rvert = \log\det M\f$. With \f$A[T]\f$ and \f$G[T]\f$ the derivatives along the positions of the tuple \f$F\f$ in the subset \f$T\f$,
   * \f[
   * G[S] = -G[\emptyset]\sum_{\emptyset \ne T \subseteq S} A[T]\,G[S \setminus T],
   * \f]
   * and \f$\partial^F M^{-1}\f$ is the top-left \f$p \times p\f$ block of \f$G[F]\f$. Matrix \f$A[T]\f$ is nonzero only along one repeated parameter:
   * \f$2^k \psi_i\f$ (\f$\psi_i = e^{2t_i}\f$) at \f$(i, i)\f$ for \f$t_i\f$ (\f$k\f$ its multiplicity), and the \f$k\f$-th derivative of \f$\Lambda_{rc}\f$ at \f$(r, p + c)\f$ and \f$(p + c, r)\f$.
   * Block \f$G[\emptyset]\f$ is assembled from \f$M^{-1}\f$ via inverse() and products \f$U\f$ and \f$C^{-1}\f$, so neither \f$M^{-1}\Lambda\f$ nor \f$C^{-1}\f$ comes from a subtraction.
   * For \f$d = 0\f$, returns \c true only for a pure tuple of a single repeated \f$t_i\f$ (\f$(-2)^K e^{-2t_i}\f$ at entry \f$(i, i)\f$),
   * and \c false otherwise.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param out Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return \c true if computed; \c false and buffer left untouched if structurally zero.
   *
   * Accuracy over orders 1–6: up to \f$2.1 \times 10^4\f$ ulpN on gcc and \f$2.3 \times 10^4\f$ ulpN on CUDA (test bounds 2155 to 35027 ulpN); the peak occurs at the (3, 1) spread point near an inflection along \f$t_2\f$, where the even derivatives nearly vanish and one ulp of \f$\theta_k\f$ moves the exact derivative by \f$1.6 \times 10^4\f$ ulpN.
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, const int (&idx)[K], double* out, double* w) const {
    const int p = dim, nl = n_loadings();
    if (n_factors == 0) {  // M^-1 = Psi^-1
      for (int k = 1; k < K; ++k)
        if (idx[k] != idx[0]) return false;
      const int r = idx[0] - nl;
      detail_pdmat::fill_zero(out, packed_size(p));
      out[packed_index(r, r)] = detail_pdmat::ipow(-2.0, K) * detail_pdmat::m::exp(-2.0 * theta[idx[0]]);
      return true;
    }
    constexpr int NM = 1 << K;
    const int n = p + n_factors;
    bool gz[NM];
    augmented_recursion<K>(theta, idx, NM - 1, w, gz);
    if (!gz[NM - 1]) return false;
    const double* GF = w + woodbury_size() + (NM - 1) * n * n;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) out[packed_index(i, j)] = GF[i * n + j];
    return true;
  }

 private:
  // M^-1 into m (packed) from a Cholesky factorization of M, M = L L^T,
  // M^-1 = Y^T Y with Y = L^-1; the buffers after Woodbury's part of w
  LAPLACE_HD void cholesky_inverse(const double* theta, double* m, double* w) const {
    const int p = dim, np = packed_size(p);
    double *acc = w + woodbury_size(), *Lp = acc + p * p, *Y = Lp + np;
    matrix(theta, m, w);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) acc[i * p + j] = m[packed_index(i, j)];
    detail_pdmat::fa_cholesky(p, acc);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) Lp[packed_index(i, j)] = acc[i * p + j];
    detail_pdmat::lower_inverse(p, Lp, Y);
    detail_pdmat::fill_zero(m, np);
    detail_pdmat::add_gram(p, Y, Y, m);
  }

  // A[T] along the positions in the mask T of idx: false when zero, else its
  // entry (i, j) (and (j, i)) with value x; nonzero only along one repeated
  // parameter
  template <int K>
  LAPLACE_HD bool augmented_dk(const double* theta, const int (&idx)[K], int T, int& i, int& j, double& x) const {
    const int nl = n_loadings();
    int v = -1, k = 0;
    for (int q = 0; q < K; ++q)
      if (T & (1 << q)) {
        if (v >= 0 && idx[q] != v) return false;
        v = idx[q], ++k;
      }
    if (v >= nl) {
      i = j = v - nl;
      x = detail_pdmat::ipow(2.0, k) * detail_pdmat::m::exp(2.0 * theta[v]);
      return true;
    }
    int c;
    tz().loading_position(v, i, c);
    j = dim + c;
    x = tz().loading_d(theta, v, k);
    return x != 0.0;
  }

  // The recursion on A along idx: G[0] = A^-1 from its blocks, then
  // G[S] = -G[0] sum_T A[T] G[S \ T] (full n x n, after Woodbury's part of w)
  // for the masks S within lim; gz: nonzero by structure
  template <int K>
  LAPLACE_HD void augmented_recursion(const double* theta, const int (&idx)[K], int lim, double* w, bool* gz) const {
    constexpr int NM = 1 << K;
    const int p = dim, d = n_factors, n = p + d, nn = n * n;
    double *G = w + woodbury_size(), *acc = G + detail_pdmat::generic_masks * nn, *rows = acc + nn;
    // M^-1 into the packed lower triangle of acc, then the blocks of G[0]
    inverse(theta, acc, w);
    for (int i = p - 1; i >= 0; --i)
      for (int j = i; j >= 0; --j) G[i * n + j] = G[j * n + i] = acc[packed_index(i, j)];
    const work s = carve(w);
    load_lambda(theta, s.lam);
    load_em2t(theta, s.em2t);
    build_c(s, s.R);
    detail_pdmat::fa_cholesky(d, s.R);
    for (int b = 0; b < d; ++b) {  // column b of C^-1: forward, then back substitution
      for (int i = 0; i < d; ++i) {
        double t = i == b ? 1.0 : 0.0;
        for (int k = 0; k < i; ++k) t -= s.R[i * d + k] * s.u[k];
        s.u[i] = t / s.R[i * d + i];
      }
      for (int i = d - 1; i >= 0; --i) {
        double t = s.u[i];
        for (int k = i + 1; k < d; ++k) t -= s.R[k * d + i] * s.Ci[k * d + b];
        s.Ci[i * d + b] = t / s.R[i * d + i];
      }
    }
    for (int i = 0; i < p; ++i)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int a = 0; a < d; ++a) t += s.em2t[i] * s.lam[i * d + a] * s.Ci[a * d + b];
        G[i * n + p + b] = G[(p + b) * n + i] = t;
      }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) G[(p + a) * n + p + b] = -s.Ci[a * d + b];
    gz[0] = true;
    for (int S = 1; S < NM; ++S) {
      gz[S] = false;
      if (S & ~lim) continue;
      for (int e = 0; e < nn; ++e) acc[e] = 0.0;
      for (int a = 0; a < n; ++a) rows[a] = 0.0;
      for (int T = S; T != 0; T = (T - 1) & S) {
        const int R = S ^ T;
        int i, j;
        double x;
        if (!gz[R] || !augmented_dk<K>(theta, idx, T, i, j, x)) continue;
        const double* gr = G + R * nn;
        for (int b = 0; b < n; ++b) acc[i * n + b] += x * gr[j * n + b];
        if (j != i)
          for (int b = 0; b < n; ++b) acc[j * n + b] += x * gr[i * n + b];
        rows[i] = rows[j] = 1.0;
        gz[S] = true;
      }
      if (!gz[S]) continue;
      double* gs = G + S * nn;
      for (int e = 0; e < nn; ++e) gs[e] = 0.0;
      for (int k = 0; k < n; ++k)
        if (rows[k] != 0.0)
          for (int a = 0; a < n; ++a) {
            const double g = G[a * n + k];
            if (g == 0.0) continue;
            for (int b = 0; b < n; ++b) gs[a * n + b] -= g * acc[k * n + b];
          }
    }
  }
  struct work {
    double *lam, *em2t, *tmp, *R, *Ci, *C, *X, *u, *v;
  };
  LAPLACE_HD int woodbury_size() const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    return 2 * p * d + p + 2 * d * d + 2 * nm * d * d + 2 * d;
  }
  LAPLACE_HD work carve(double* w) const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    work s;
    s.lam = w;
    s.em2t = s.lam + p * d;
    s.tmp = s.em2t + p;
    s.R = s.tmp + p * d;
    s.Ci = s.R + d * d;
    s.C = s.Ci + d * d;
    s.X = s.C + nm * d * d;
    s.u = s.X + nm * d * d;
    s.v = s.u + d;
    return s;
  }

  LAPLACE_HD detail_pdmat::trapezoid_loadings tz() const { return {dim, n_factors}; }
  // the row of parameter v (a loading of that row, or its t)
  LAPLACE_HD int param_row(int v) const {
    const int nl = n_loadings();
    if (v >= nl) return v - nl;
    int r, c;
    tz().loading_position(v, r, c);
    return r;
  }
  LAPLACE_HD void load_lambda(const double* theta, double* lam) const { tz().load_lambda(theta, lam); }
  LAPLACE_HD void load_em2t(const double* theta, double* em2t) const {
    const int nl = n_loadings();
    for (int i = 0; i < dim; ++i) em2t[i] = detail_pdmat::m::exp(-2.0 * theta[nl + i]);
  }
  // C = I + sum_i e^(-2 t_i) lambda_i lambda_i^T into c (full d x d)
  LAPLACE_HD void build_c(const work& s, double* c) const {
    const int p = dim, d = n_factors;
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int i = 0; i < p; ++i) t += s.em2t[i] * s.lam[i * d + a] * s.lam[i * d + b];
        c[a * d + b] = (a == b ? 1.0 : 0.0) + t;
      }
  }
};

static_assert(pdmat_structure<factor_analytic_pdmat>);

/**
 * \ingroup pdmat
 * \brief Largest number of factors \f$d\f$ satisfying Ledermann's bound \f$(p - d)^2 \ge p + d\f$.
 *
 * Beyond this bound, \f$\Lambda\Lambda^\top\f$ and \f$\Psi\f$ cannot be distinguished from \f$M\f$.
 * Host-only function (not marked \c LAPLACE_HD). Returns 0 for \f$p = 1\f$, 1 for \f$p = 3\f$,
 * 2 for \f$p = 5\f$, and 6 for \f$p = 10\f$.
 *
 * \param p Matrix dimension (\f$p \ge 1\f$).
 * \return Maximum identifiable number of factors \f$d \le p\f$.
 *
 * \see factor_analytic_pdmat
 */
// The largest d with (p - d)^2 >= p + d (Ledermann's bound): beyond it
// Lambda Lambda^T and Psi cannot be told apart from M. Host only.
inline int factor_analytic_max_factors(int p) {
  int d = p;
  while (d > 0 && (p - d) * (p - d) < p + d) --d;
  return d;
}

}  // namespace laplace::pdmat
