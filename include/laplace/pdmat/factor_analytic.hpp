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
// O(p d^2 + d^3) instead of the p^3 of a Cholesky factorization of M.
// Derivatives along a sorted tuple F, with [S] the derivative along the
// positions in the mask S:
//   M: closed and sparse, as log_cholesky: a tuple of t_i alone reaches M_ii;
//     loadings reach M only on at most two variables of one column; a tuple
//     that mixes a t with a loading is zero;
//   log det M: C[T] is zero unless every index of T belongs to one row r (its
//     loadings or t_r), and then e^(-2 t_r) (-2)^(number of t_r) times the
//     Leibniz sum over the loadings of lambda_r lambda_r^T; with
//     X[S] = -X[0] sum_(T nonempty subset of S) C[T] X[S \ T] (X = C^-1),
//     d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)]),
//     f the first position; about 3^K d^3. At d = p, where these terms cancel
//     (every row is explained by the factors), the same sum of traces on M:
//     d^F log det M = sum_(T subset of F \ f) tr(G[T] M[f + (F \ f \ T)]),
//     with the G[T] of the recursion below;
//   M^-1: not by Woodbury, whose derivatives cancel (Psi^-1 against
//     B C^-1 B^T) when psi_i is small against the loadings of row i, up to
//     7e7 ulpN (note/pdmat_structures.md section 9, Giovanni 2026-11-14), but
//     by the recursion on M itself, with G = M^-1,
//     G[S] = -G[0] sum_(T nonempty subset of S) M[T] G[S \ T] and G[0] from a
//     Cholesky factorization of M, skipping the M[T] that are
//     zero by structure (most of them); about 2^K p^3 plus the products with
//     the nonzero M[T].
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
  /// \brief Compile-time descriptor: \c true, \f$M^{-1}\f$ is computed by Woodbury's identity, without a factorization of \f$M\f$.
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
   * 2pd + p + d^2 + 128d^2 + 2d + 64\,\frac{p(p + 1)}{2} + 64p^2 + p^2
   * \f]
   * doubles, accommodating Woodbury temporary matrices and the workspace for the \f$M^{-1}\f$ derivative recursion.
   *
   * \return Required scratch size in doubles.
   */
  // Woodbury's part: Lambda (p d), e^(-2 t) (p), a p x d temporary, the
  // factor R of C (d^2), C[T] and X[T] (64 d^2 each) and two d-vectors; then
  // the recursion of the derivatives of M^-1: M[T] (64 packed), X[T]
  // (64 p^2) and a p x p sum
  LAPLACE_HD int scratch_size() const {
    const int p = dim, nm = detail_pdmat::generic_masks;
    return woodbury_size() + nm * packed_size(p) + nm * p * p + p * p;
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
   * Accuracy: on \f$(p, d) = (8, 2)\f$, relative matrix distance \f$\max|M(\hat\theta) - \hat{M}|/\max|\hat{M}|\f$ reaches \f$1.1 \times 10^{-3}\f$ on a matrix of its own structure (test bound 0.05) and 0.17 on a generic positive definite matrix on gcc 14.2/OpenMP.
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
   * Accuracy: up to 0.76 ulpN on gcc and CUDA (test bound 2 ulpN).
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
   * Accuracy on gcc and CUDA: up to 2.0 ulp for \f$d < p\f$ (test bound 4 ulp) and 3.2 ulp for \f$d = p\f$ (test bound 5 ulp).
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
   * \f$C = I_d + \Lambda^\top\Psi^{-1}\Lambda = R R^\top\f$ via Woodbury's identity in packed lower-triangular storage.
   * For \f$d = 0\f$, computes \f$M^{-1} = \Psi^{-1}\f$.
   *
   * \param theta Array of \f$q\f$ parameters.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * Accuracy: up to 56 ulpN on gcc and 90 ulpN on CUDA (test bound 136 ulpN).
   *
   * \see matrix(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, double* w) const {
    const int p = dim, d = n_factors;
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
   * plus 2 for a single parameter \f$t_i\f$ at order 1. For \f$d = p\f$, evaluates the same sum of traces directly on \f$M\f$,
   * \f[
   * \partial^F \log\det M = \sum_{T \subseteq F \setminus \{f\}} \operatorname{tr}\bigl(G[T]\,M[\{f\} \cup (F \setminus \{f\} \setminus T)]\bigr),
   * \f]
   * with \f$G = M^{-1}\f$ and its derivatives \f$G[T]\f$ from the recursion on \f$M\f$ (forming only masks within \f$F \setminus \{f\}\f$).
   * For \f$d = 0\f$, returns 2 for a single \f$t_i\f$ at \f$K = 1\f$ and \f$0.0\f$ otherwise.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero.
   *
   * Accuracy over orders 1–6: for \f$d < p\f$, up to 795 ulp on gcc and 926 ulp on CUDA (test bounds 5 to 1390 ulp); for \f$d = p\f$, up to 820 ulp on gcc and 840 ulp on CUDA (test bounds 2 to 1260 ulp).
   *
   * \see log_det(), inverse_dk(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], double* w) const {
    const int nl = n_loadings(), d = n_factors;
    const double psi_part = (K == 1 && idx[0] >= nl) ? 2.0 : 0.0;
    if (d == 0) return psi_part;
    // at d = p Woodbury's terms cancel (every row is explained by the factors):
    // the traces on M instead (docs/numerics.md)
    if (d == dim) return log_det_dk_on_m<K>(theta, idx, w);
    constexpr int NM = 1 << K;
    const work s = carve(w);
    bool cnz[NM], xnz[NM];
    const detail_pdmat::trapezoid_loadings z = tz();
    z.load_lambda(theta, s.lam);
    load_em2t(theta, s.em2t);
    detail_pdmat::loadings_gram_masks<K>(z, theta, idx, s.lam, s.em2t, true, [&](int v) { return param_row(v); }, s.C, cnz, s.u, s.v);
    detail_pdmat::log_det_masks_inverse<K>(d, s.C, cnz, s.R, s.tmp, s.X, xnz);
    return psi_part + detail_pdmat::log_det_masks_sum<K>(d, s.C, cnz, s.X, xnz);
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates derivatives via the recurrence on \f$M\f$, with \f$G = M^{-1}\f$ and \f$M[T]\f$, \f$G[T]\f$ the derivatives
   * along the positions of the tuple in the subset \f$T\f$:
   * \f[
   * G[S] = -G[\emptyset]\sum_{\emptyset \ne T \subseteq S} M[T]\,G[S \setminus T]
   * \f]
   * with \f$G[\emptyset] = M^{-1}\f$ initialized from a Cholesky factorization of \f$M\f$ and skipping structurally zero \f$M[T]\f$.
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
   * Accuracy over orders 1–6: up to \f$2.4 \times 10^5\f$ ulpN on gcc and \f$3.6 \times 10^5\f$ ulpN on CUDA (test bounds 275 to 545993 ulpN); the peak occurs at the (3, 1) spread point near an inflection along \f$t_2\f$, where the even derivatives nearly vanish and condition reaches \f$1.6 \times 10^4\f$ ulpN per ulp of \f$\theta_k\f$, while errors elsewhere remain within \f$2.5 \times 10^4\f$ ulpN.
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
    const int pp = p * p;
    double* X = w + woodbury_size() + detail_pdmat::generic_masks * packed_size(p);
    bool mz[NM], xz[NM];
    recursion_on_m<K>(theta, idx, NM - 1, w, mz, xz);
    if (!xz[NM - 1]) return false;
    const double* XF = X + (NM - 1) * pp;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) out[packed_index(i, j)] = XF[i * p + j];
    return true;
  }

 private:
  // The recursion on M along idx: M[T] for every mask T (packed, after
  // Woodbury's part of w), G[0] = M^-1 from a Cholesky factorization of M and
  // G[Q] = -G[0] sum_T M[T] G[Q \ T] (full p x p) for the masks Q within lim;
  // mz, xz: nonzero by structure
  template <int K>
  LAPLACE_HD void recursion_on_m(const double* theta, const int (&idx)[K], int lim, double* w, bool* mz,
                                 bool* xz) const {
    const int p = dim;
    constexpr int NM = 1 << K;
    const int np = packed_size(p), pp = p * p, nm = detail_pdmat::generic_masks;
    double* Md = w + woodbury_size();  // M[T], packed
    double* X = Md + nm * np;          // X[T], full p x p
    double* acc = X + nm * pp;
    for (int T = 1; T < NM; ++T) {
      int sub[K], k = 0;
      for (int q = 0; q < K; ++q)
        if (T & (1 << q)) sub[k++] = idx[q];
      mz[T] = matrix_dk_runtime(k, theta, sub, Md + T * np, w);
    }
    // X[0] = M^-1 = Y^T Y, Y = L^-1, from a Cholesky factorization of M (the
    // recursion amplifies its errors less than those of Woodbury's M^-1: at
    // the spread reference point with d = p, 933 against 1.2e5 ulpN)
    matrix(theta, Md, w);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) acc[i * p + j] = Md[packed_index(i, j)];
    detail_pdmat::fa_cholesky(p, acc);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) Md[packed_index(i, j)] = acc[i * p + j];
    double* Y = X + (NM - 1) * pp;  // free until X[NM - 1] is formed
    detail_pdmat::lower_inverse(p, Md, Y);
    detail_pdmat::fill_zero(acc, np);
    detail_pdmat::add_gram(p, Y, Y, acc);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) X[i * p + j] = X[j * p + i] = acc[packed_index(i, j)];
    xz[0] = true;
    for (int Q = 1; Q < NM; ++Q) {
      xz[Q] = false;
      if (Q & ~lim) continue;
      for (int e = 0; e < pp; ++e) acc[e] = 0.0;
      for (int T = Q; T != 0; T = (T - 1) & Q) {
        if (!mz[T] || !xz[Q ^ T]) continue;
        xz[Q] = true;
        const double* MT = Md + T * np;
        const double* XR = X + (Q ^ T) * pp;
        for (int i = 0; i < p; ++i)
          for (int k = 0; k < p; ++k) {
            const double a = MT[i >= k ? packed_index(i, k) : packed_index(k, i)];
            if (a == 0.0) continue;
            for (int j = 0; j < p; ++j) acc[i * p + j] += a * XR[k * p + j];
          }
      }
      if (!xz[Q]) continue;
      double* XQ = X + Q * pp;
      for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) {
          double r = 0.0;
          for (int k = 0; k < p; ++k) r += X[i * p + k] * acc[k * p + j];
          XQ[i * p + j] = -r;
        }
    }
  }

  // d^F log det M = sum_(T subset of F \ f) tr(G[T] M[f + (F \ f \ T)]), with
  // the G[T] of the recursion on M
  template <int K>
  LAPLACE_HD double log_det_dk_on_m(const double* theta, const int (&idx)[K], double* w) const {
    const int p = dim;
    constexpr int NM = 1 << K;
    const int np = packed_size(p), pp = p * p;
    const double* Md = w + woodbury_size();
    const double* X = Md + detail_pdmat::generic_masks * np;
    bool mz[NM], xz[NM];
    const int rest = (NM - 1) ^ 1;
    recursion_on_m<K>(theta, idx, rest, w, mz, xz);
    double ld = 0.0;
    for (int T = rest;; T = (T - 1) & rest) {
      const int U = 1 | (rest ^ T);
      if (xz[T] && mz[U]) {
        const double* XT = X + T * pp;
        const double* MU = Md + U * np;
        for (int i = 0; i < p; ++i)
          for (int j = 0; j < p; ++j) ld += XT[i * p + j] * MU[i >= j ? packed_index(i, j) : packed_index(j, i)];
      }
      if (T == 0) break;
    }
    return ld;
  }

  struct work {
    double *lam, *em2t, *tmp, *R, *C, *X, *u, *v;
  };
  LAPLACE_HD int woodbury_size() const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    return 2 * p * d + p + d * d + 2 * nm * d * d + 2 * d;
  }
  // d^k M along sub[0..k-1], k = 1..6 known at run time
  template <int k>
  LAPLACE_HD bool matrix_dk_of(const double* theta, const int* sub, double* m, double* w) const {
    int idx[k];
    for (int q = 0; q < k; ++q) idx[q] = sub[q];
    return matrix_dk<k>(theta, idx, m, w);
  }
  LAPLACE_HD bool matrix_dk_runtime(int k, const double* theta, const int* sub, double* m, double* w) const {
    switch (k) {
      case 1: return matrix_dk_of<1>(theta, sub, m, w);
      case 2: return matrix_dk_of<2>(theta, sub, m, w);
      case 3: return matrix_dk_of<3>(theta, sub, m, w);
      case 4: return matrix_dk_of<4>(theta, sub, m, w);
      case 5: return matrix_dk_of<5>(theta, sub, m, w);
      default: return matrix_dk_of<6>(theta, sub, m, w);
    }
  }
  LAPLACE_HD work carve(double* w) const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    work s;
    s.lam = w;
    s.em2t = s.lam + p * d;
    s.tmp = s.em2t + p;
    s.R = s.tmp + p * d;
    s.C = s.R + d * d;
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
