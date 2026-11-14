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
//     f the first position; about 3^K d^3;
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
 * \par Measured accuracy
 * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
 * - \f$M\f$: 0.76 ulpN (gcc) / 0.76 ulpN (CUDA)
 * - \f$M^{-1}\f$: 2.4e5 ulpN (\f$K=4\f$, gcc) / 3.6e5 ulpN (\f$K=2, 4, 6\f$, CUDA)
 * - \f$\log \det M\f$ (\f$d < p\f$): 795 ulp (\f$K=6\f$, gcc) / 926 ulp (\f$K=6\f$, CUDA)
 * - \f$\log \det M\f$ (\f$d = p\f$): 7.3e5 ulp (\f$K=5\f$, gcc) / 1.0e6 ulp (\f$K=5\f$, CUDA)
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
   * \brief Evaluates the packed factor-analytic matrix \f$M(\theta)\f$.
   *
   * Computes \f$M = \Lambda\Lambda^\top + \Psi\f$ in packed lower-triangular storage.
   * For \f$d = 0\f$, evaluates \f$M = \Psi\f$, matching \ref laplace::pdmat::diagonal_pdmat "diagonal_pdmat".
   *
   * \param theta Array of \f$q\f$ parameters (loadings followed by \f$t_0, \dots, t_{p-1}\f$).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   *
   * \par Measured accuracy
   * Maximum error 0.76 ulpN (gcc) / 0.76 ulpN (CUDA); test bound 2 ulpN.
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
   * \par Measured accuracy
   * Maximum error: for \f$d < p\f$, 2.0 ulp (gcc) / 2.0 ulp (CUDA); for \f$d = p\f$, 3.2 ulp (gcc) / 3.2 ulp (CUDA). Test bounds 4 ulp (\f$d < p\f$) and 5 ulp (\f$d = p\f$).
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
   * \par Measured accuracy
   * Maximum error 56 ulpN (gcc) / 90 ulpN (CUDA); test bound 136 ulpN.
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
   * \par Measured accuracy
   * Maximum error over orders 1–6: 0.49 ulpN (gcc) / 0.41 ulpN (CUDA); test bound 1 ulpN.
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
   * Evaluates derivatives via Woodbury's identity, with \f$C = I_d + \Lambda^\top\Psi^{-1}\Lambda\f$, \f$X = C^{-1}\f$,
   * \f$C[T]\f$ and \f$X[T]\f$ their derivatives along the positions of the tuple \f$F\f$ in the subset \f$T\f$, \f$f\f$ the
   * first position of \f$F\f$, and \f$X[S] = -X[\emptyset]\sum_{\emptyset \ne T \subseteq S} C[T]\,X[S \setminus T]\f$:
   * \f[
   * \partial^F \log\det C = \sum_{T \subseteq F \setminus \{f\}} \operatorname{tr}\bigl(X[T]\,C[\{f\} \cup (F \setminus \{f\} \setminus T)]\bigr)
   * \f]
   * plus 2 for a single parameter \f$t_i\f$ at order 1. For \f$d = 0\f$, returns 2 for a single \f$t_i\f$ at \f$K = 1\f$ and \f$0.0\f$ otherwise.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ parameters.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param w Caller scratch buffer of at least \c scratch_size() doubles.
   * \return Value of the derivative, or \f$0.0\f$ if identically zero.
   *
   * \par Measured accuracy
   * Maximum error over orders 1–6:
   * - \f$d < p\f$: 795 ulp (\f$K=6\f$, gcc) / 926 ulp (\f$K=6\f$, CUDA); test bounds 5 to 1390 ulp.
   * - \f$d = p\f$: 7.3e5 ulp (\f$K=5\f$, gcc) / 1.0e6 ulp (\f$K=5\f$, CUDA); test bounds 161 to 1507089 ulp.
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, const int (&idx)[K], double* w) const {
    const int nl = n_loadings(), d = n_factors;
    const double psi_part = (K == 1 && idx[0] >= nl) ? 2.0 : 0.0;
    if (d == 0) return psi_part;
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
   * \par Measured accuracy
   * Maximum error over orders 1–6: 2.4e5 ulpN (\f$K=4\f$, gcc) / 3.6e5 ulpN (\f$K=2, 4, 6\f$, CUDA); test bounds 275 to 545993 ulpN.
   * At the (3, 1) spread point, derivatives along \f$t_2\f$ have high condition number (one ulp of \f$\theta_k\f$ moves the exact
   * derivative by \f$1.6 \times 10^4\f$ ulpN); elsewhere \f$M^{-1}\f$ remains within \f$2.5 \times 10^4\f$ ulpN.
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
    const int np = packed_size(p), pp = p * p, nm = detail_pdmat::generic_masks;
    double* Md = w + woodbury_size();  // M[T], packed
    double* X = Md + nm * np;          // X[T], full p x p
    double* acc = X + nm * pp;
    bool mz[NM], xz[NM];  // nonzero by structure
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
    if (!xz[NM - 1]) return false;
    const double* XF = X + (NM - 1) * pp;
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) out[packed_index(i, j)] = XF[i * p + j];
    return true;
  }

 private:
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
