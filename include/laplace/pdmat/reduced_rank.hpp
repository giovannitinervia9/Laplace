// reduced_rank_psdmat: M = Lambda Lambda^T of rank d (note/pdmat_structures.md
// section 10; interface in core.hpp, psdmat_structure). Lambda is p x d with
// the convention of factor_analytic_pdmat (detail/loadings.hpp): lower
// trapezoidal, Lambda_jj = e^(theta) on its diagonal, the other loadings free,
// theta the loadings by rows, q = pd - d(d-1)/2. Precondition 1 <= d <= p, not
// checked.
//
// log|M|_+ = log det C, C = Lambda^T Lambda (d x d): the nonzero eigenvalues
// of Lambda Lambda^T are those of Lambda^T Lambda. At d = p it is
// 2 sum_j theta_jj exactly (Lambda square and triangular); otherwise the value
// from a Cholesky factorization of C and the derivatives by the recursion on
// C of factor_analytic_pdmat, with C[T] zero unless every index of T is a
// loading of one row r, and then the Leibniz sum over those loadings of
// lambda_r lambda_r^T. The derivatives run in double-double from C onward
// (C, C[T], C^-1, X[T], the traces), Lambda in double. M and Lambda have
// closed sparse derivatives.
#pragma once

#include <laplace/math/detail/double_double.hpp>
#include <laplace/pdmat/core.hpp>
#include <laplace/pdmat/detail/from_matrix.hpp>
#include <laplace/pdmat/detail/loadings.hpp>

namespace laplace::pdmat {

/**
 * \ingroup psdmat
 * \brief Structured positive semidefinite matrix with reduced-rank factor loadings.
 *
 * Maps \f$q = pd - d(d - 1)/2\f$ unconstrained real parameters \f$\theta\f$ to a symmetric positive
 * semidefinite \f$p \times p\f$ matrix \f$M(\theta) = \Lambda\Lambda^\top\f$ of fixed rank \f$d\f$,
 * where \f$\Lambda\f$ is a \f$p \times d\f$ lower-trapezoidal loading matrix with diagonal entries
 * \f$\Lambda_{jj} = e^\theta\f$ (\f$0 \le j < d\f$) and free off-diagonal entries stored by rows in \f$\theta\f$.
 *
 * Precondition: \f$1 \le d \le p\f$ (unchecked).
 *
 * \see psdmat, psdmat_orders, psdmat_structure, factor_analytic_pdmat
 */
struct reduced_rank_psdmat : psdmat_orders<reduced_rank_psdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;        // p
  /// \brief Matrix rank and number of latent factors \f$d\f$ (\f$1 \le d \le p\f$).
  int n_factors = 1;  // d, the rank

  /**
   * \ingroup psdmat
   * \brief Default constructor initializing dimension and rank to 1.
   */
  reduced_rank_psdmat() = default;
  /**
   * \ingroup psdmat
   * \brief Constructs a reduced-rank positive semidefinite matrix of dimension \f$p\f$ and rank \f$d\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   * \param d Matrix rank and number of factors (\f$1 \le d \le p\f$).
   */
  LAPLACE_HD reduced_rank_psdmat(int p, int d) : dim(p), n_factors(d) {}

  /**
   * \ingroup psdmat
   * \brief Total number of parameters \f$q = pd - d(d - 1)/2\f$.
   * \return Number of loading parameters in the lower trapezoid.
   */
  LAPLACE_HD int n_params() const { return dim * n_factors - n_factors * (n_factors - 1) / 2; }
  /**
   * \ingroup psdmat
   * \brief Fixed matrix rank \f$d\f$.
   * \return Rank of \f$M(\theta)\f$, equal to `n_factors`.
   */
  LAPLACE_HD int rank() const { return n_factors; }
  /// \brief Compile-time descriptor: \c false, the null space of \f$\Lambda\Lambda^\top\f$ depends on \f$\theta\f$.
  // the null space of Lambda Lambda^T moves with theta, and no basis of it is supplied
  static constexpr bool constant_null_space = false;
  /// \brief Compile-time descriptor: \c false, no null-space basis is supplied (`null_space_basis()` is not provided).
  static constexpr bool supplies_null_space = false;
  /**
   * \ingroup psdmat
   * \brief Doubles of caller scratch buffer required by functions of this structure.
   *
   * Returns \f$pd + 265d^2 + 2d\f$ doubles.
   *
   * \return Required scratch size in doubles.
   */
  // Lambda (p d), R (d^2), two d-vectors, then in double-double (two doubles
  // each) C[T] and X[T] (64 d^2 each) and four d x d matrices
  LAPLACE_HD int scratch_size() const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    return p * d + d * d + 2 * d + 2 * (2 * nm + 4) * d * d;
  }

  /**
   * \ingroup psdmat
   * \brief Computes starting parameters \f$\theta\f$ from a target packed matrix \f$\hat{M}\f$.
   *
   * Host-only function. Forms \f$A\f$ from the \f$d\f$ leading eigenpairs of \f$\hat{M}\f$
   * (columns scaled by square roots of eigenvalues floored at \f$10^{-6}\f$ times the
   * largest, or \f$1\f$ if none is positive), and converts \f$AA^\top\f$ to lower-trapezoidal
   * loadings via the first \f$d\f$ Cholesky columns of \f$AA^\top\f$ (pivots floored).
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output array of \c n_params() doubles inside the domain.
   *
   * Accuracy: recovers \f$\theta\f$ from `matrix()` within \f$1.7 \times 10^{-16}\f$ at \f$(p, d) = (5, 2)\f$ on gcc 14.2/OpenMP (test bound \f$10^{-12}\f$).
   *
   * \see matrix(), psdmat
   */
  // host: Lambda the leading d eigenvectors of m_hat scaled by the square
  // roots of their eigenvalues (floored), to the trapezoidal convention
  // (exact, up to rounding, on matrix(theta))
  void params_from_matrix(const double* m_hat, double* theta) const {
    namespace dp = detail_pdmat;
    const int p = dim, d = n_factors;
    std::vector<double> lam, V, A(p * d);
    dp::eigen_decreasing(p, m_hat, lam, V);
    const double low = lam[0] > 0.0 ? dp::start_eigen_floor * lam[0] : 1.0;
    for (int i = 0; i < p; ++i)
      for (int k = 0; k < d; ++k) A[i * d + k] = V[i * p + k] * std::sqrt(std::max(lam[k], low));
    dp::trapezoid_params(p, d, A.data(), theta);
  }

  /**
   * \ingroup psdmat
   * \brief Evaluates the packed reduced-rank matrix \f$M(\theta) = \Lambda\Lambda^\top\f$.
   *
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower-triangular storage.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   *
   * Accuracy: up to 0.63 ulpN on gcc/OpenMP and 0.55 ulpN on CUDA (test bound 1 ulpN).
   *
   * \see factor(), log_pdet(), matrix_dk()
   */
  // M = Lambda Lambda^T, packed
  LAPLACE_HD void matrix(const double* theta, double* m, double* w) const {
    const int p = dim, d = n_factors;
    double* lam = w;
    tz().load_lambda(theta, lam);
    for (int i = 0; i < p; ++i)
      for (int j = 0; j <= i; ++j) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) s += lam[i * d + k] * lam[j * d + k];
        m[packed_index(i, j)] = s;
      }
  }

  /**
   * \ingroup psdmat
   * \brief Evaluates the loading matrix \f$\Lambda(\theta)\f$ in dense row-major storage.
   *
   * Writes the \f$p \times d\f$ matrix with zeros above the diagonal (not packed).
   * Not part of \ref psdmat_structure.
   *
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param lam Output buffer of at least \f$pd\f$ doubles for row-major \f$p \times d\f$ storage.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles (unused).
   *
   * Accuracy: up to 0.34 ulpN on gcc/OpenMP and CUDA (test bound 1 ulpN).
   *
   * \see matrix(), factor_dk()
   */
  // Lambda, p x d row-major with zeros above the diagonal
  LAPLACE_HD void factor(const double* theta, double* lam, [[maybe_unused]] double* w) const { tz().load_lambda(theta, lam); }

  /**
   * \ingroup psdmat
   * \brief Evaluates the log pseudo-determinant \f$\log\lvert M(\theta)\rvert_+ = \log\det(\Lambda^\top\Lambda)\f$.
   *
   * Computes \f$2\sum_j \theta_{jj}\f$ directly when \f$d = p\f$, and \f$2\sum_k \log R_{kk}\f$
   * from a Cholesky factorization \f$\Lambda^\top\Lambda = R R^\top\f$ when \f$d < p\f$.
   *
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \return Sum of logarithms of the \f$d\f$ nonzero eigenvalues of \f$M(\theta)\f$.
   *
   * Accuracy: up to 1.27 ulp on gcc/OpenMP and 1.61 ulp on CUDA (test bound 3 ulp); within 0.42 ulp at \f$d = p\f$.
   *
   * \see matrix(), log_pdet_dk()
   */
  // log|M|_+ = log det(Lambda^T Lambda)
  LAPLACE_HD double log_pdet(const double* theta, double* w) const {
    const int d = n_factors;
    if (d == dim) {
      double s = 0.0;
      for (int j = 0; j < d; ++j) s += theta[j * (j + 1) / 2 + j];
      return 2.0 * s;
    }
    const work s = carve(w);
    tz().load_lambda(theta, s.lam);
    build_c(s.lam, s.R);
    detail_pdmat::fa_cholesky(d, s.R);
    double lc = 0.0;
    for (int k = 0; k < d; ++k) lc += detail_pdmat::m::log(s.R[k * d + k]);
    return 2.0 * lc;
  }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates the closed sparse derivative of \f$\Lambda\Lambda^\top\f$; returns \c false and
   * leaves `m` untouched when structurally zero.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles (unused).
   * \return \c true if written; \c false and buffer untouched if structurally zero.
   *
   * Accuracy over orders 1 to 6: up to 0.49 ulpN on gcc/OpenMP and 0.43 ulpN on CUDA (test bound 1 ulpN).
   *
   * \see matrix(), psdmat_orders::matrix_d1()
   */
  // d^K M along a sorted tuple: closed and sparse (false, m untouched, when
  // zero by structure)
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    return tz().template gram_dk<K>(theta, idx, m);
  }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$\Lambda(\theta)\f$ of order \f$K\f$ in dense row-major storage.
   *
   * Nonzero only when all \f$K\f$ indices repeat one loading: `1.0` at order 1 for a free loading
   * (zero for \f$K \ge 2\f$) and \f$e^\theta\f$ at every order for a diagonal loading. Not part
   * of \ref psdmat_structure.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param lam Output buffer of at least \f$pd\f$ doubles for row-major \f$p \times d\f$ storage.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles (unused).
   * \return \c true if written; \c false and buffer untouched if structurally zero.
   *
   * Accuracy over orders 1 to 6: up to 0.49 ulpN on gcc/OpenMP and 0.43 ulpN on CUDA (test bound 1 ulpN).
   *
   * \see factor(), factor_d1()
   */
  // d^K Lambda: nonzero only along one repeated loading (1 at order 1 for a
  // free loading, e^theta at every order for a diagonal one)
  template <int K>
  LAPLACE_HD bool factor_dk(const double* theta, const int (&idx)[K], double* lam, [[maybe_unused]] double* w) const {
    for (int k = 1; k < K; ++k)
      if (idx[k] != idx[0]) return false;
    const detail_pdmat::trapezoid_loadings z = tz();
    const double x = z.loading_d(theta, idx[0], K);
    if (x == 0.0) return false;
    int r, c;
    z.loading_position(idx[0], r, c);
    detail_pdmat::fill_zero(lam, dim * n_factors);
    lam[r * n_factors + c] = x;
    return true;
  }
  /// \ingroup psdmat
  /// \brief Order-1 partial derivative \f$\partial\Lambda/\partial\theta_{i_1}\f$; forwards to `factor_dk<1>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted index \f$i_1\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d1(const double* t, const int (&i)[1], double* l, double* w) const { return factor_dk<1>(t, i, l, w); }
  /// \ingroup psdmat
  /// \brief Order-2 partial derivative \f$\partial^2\Lambda/\partial\theta_{i_1}\partial\theta_{i_2}\f$; forwards to `factor_dk<2>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d2(const double* t, const int (&i)[2], double* l, double* w) const { return factor_dk<2>(t, i, l, w); }
  /// \ingroup psdmat
  /// \brief Order-3 partial derivative \f$\partial^3\Lambda/\partial\theta_{i_1}\partial\theta_{i_2}\partial\theta_{i_3}\f$; forwards to `factor_dk<3>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le i_2 \le i_3\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d3(const double* t, const int (&i)[3], double* l, double* w) const { return factor_dk<3>(t, i, l, w); }
  /// \ingroup psdmat
  /// \brief Order-4 partial derivative \f$\partial^4\Lambda/(\partial\theta_{i_1}\cdots\partial\theta_{i_4})\f$; forwards to `factor_dk<4>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_4\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d4(const double* t, const int (&i)[4], double* l, double* w) const { return factor_dk<4>(t, i, l, w); }
  /// \ingroup psdmat
  /// \brief Order-5 partial derivative \f$\partial^5\Lambda/(\partial\theta_{i_1}\cdots\partial\theta_{i_5})\f$; forwards to `factor_dk<5>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_5\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d5(const double* t, const int (&i)[5], double* l, double* w) const { return factor_dk<5>(t, i, l, w); }
  /// \ingroup psdmat
  /// \brief Order-6 partial derivative \f$\partial^6\Lambda/(\partial\theta_{i_1}\cdots\partial\theta_{i_6})\f$; forwards to `factor_dk<6>()`.
  /// \param t Parameter array \f$\theta\f$. \param i Sorted indices \f$i_1 \le \dots \le i_6\f$. \param l Dense \f$p \times d\f$ output buffer. \param w Scratch buffer.
  /// \return \c true if written; \c false (buffer untouched) if structurally zero. \see factor(), factor_dk()
  LAPLACE_HD bool factor_d6(const double* t, const int (&i)[6], double* l, double* w) const { return factor_dk<6>(t, i, l, w); }

  /**
   * \ingroup psdmat
   * \brief Partial derivative of \f$\log\lvert M(\theta)\rvert_+\f$ of order \f$K\f$.
   *
   * Returns `2.0` at order 1 for a diagonal loading and `0.0` otherwise when \f$d = p\f$.
   * For \f$d < p\f$, evaluates the sub-tuple recursion on \f$C = \Lambda^\top\Lambda\f$ described on
   * \ref psdmat "the positive semidefinite structures page", computing in double-double arithmetic
   * from \f$C\f$ onward (with \f$\Lambda\f$ in `double` and the final result rounded to `double`).
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of \f$q\f$ loading parameters by rows.
   * \param idx Sorted tuple of \f$K\f$ parameter indices in \f$\{0, \dots, q - 1\}\f$.
   * \param w Caller scratch buffer of at least `scratch_size()` doubles.
   * \return Derivative value, or \f$0.0\f$ if identically zero.
   *
   * Accuracy over orders 1 to 6: up to 12.6 ulp on gcc/OpenMP and CUDA (test bounds 1 to 19 ulp); within 0.42 ulp at \f$d = p\f$.
   *
   * \see log_pdet(), psdmat_orders::log_pdet_d1()
   */
  // d^K log|M|_+ along a sorted tuple; 0 when identically zero
  template <int K>
  LAPLACE_HD double log_pdet_dk(const double* theta, const int (&idx)[K], double* w) const {
    const int d = n_factors;
    const detail_pdmat::trapezoid_loadings z = tz();
    if (d == dim) {  // 2 sum_j theta_jj
      if (K > 1) return 0.0;
      int r, c;
      z.loading_position(idx[0], r, c);
      return r == c ? 2.0 : 0.0;
    }
    // The recursion on C in double-double: at the (5, 4) spread point the
    // traces of order 6 add terms of 1.2e4 to a result of 0.006, and every
    // stage after Lambda (C, C[T], C^-1, X[T], the traces) loses digits on its
    // own; Lambda and its derivatives are exact in double
    // (docs/numerics.md, "Reduced-rank matrices").
    constexpr int NM = 1 << K;
    const int dd2 = d * d;
    const work s = carve(w);
    bool cnz[NM], xnz[NM];
    z.load_lambda(theta, s.lam);
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        dd_t t(0.0);
        for (int i = 0; i < dim; ++i) t = t + math::detail::ddm::two_prod(s.lam[i * d + a], s.lam[i * d + b]);
        s.C[a * d + b] = t;
      }
    // C[T]: zero unless every index of T is a loading of one row r, then the
    // Leibniz sum over those loadings of lambda_r lambda_r^T
    cnz[0] = true;
    for (int T = 1; T < NM; ++T) {
      int r = -1, n = 0, sl[K];
      bool one_row = true;
      for (int q = 0; q < K; ++q)
        if (T & (1 << q)) {
          int rq, cq;
          z.loading_position(idx[q], rq, cq);
          if (r < 0) r = rq;
          one_row = one_row && rq == r;
          sl[n++] = idx[q];
        }
      cnz[T] = false;
      if (!one_row) continue;
      dd_t* ct = s.C + T * dd2;
      for (int e = 0; e < dd2; ++e) ct[e] = dd_t(0.0);
      for (int A = 0; A < (1 << n); ++A) {
        int sa[K], sb[K], na = 0, nb = 0;
        for (int k = 0; k < n; ++k) (A & (1 << k) ? sa[na++] : sb[nb++]) = sl[k];
        if (!z.lambda_row_d(theta, s.lam, r, sa, na, s.u) || !z.lambda_row_d(theta, s.lam, r, sb, nb, s.v)) continue;
        cnz[T] = true;
        for (int a = 0; a < d; ++a)
          for (int b = 0; b < d; ++b) ct[a * d + b] = ct[a * d + b] + math::detail::ddm::two_prod(s.u[a], s.v[b]);
      }
    }
    // X[0] = C^-1 = Y^T Y, Y = R^-1, C = R R^T
    for (int j = 0; j < d; ++j) {
      dd_t t = s.C[j * d + j];
      for (int k = 0; k < j; ++k) t = t - s.Rd[j * d + k] * s.Rd[j * d + k];
      s.Rd[j * d + j] = math::detail::ddm::sqrt(t);
      for (int i = j + 1; i < d; ++i) {
        dd_t u = s.C[i * d + j];
        for (int k = 0; k < j; ++k) u = u - s.Rd[i * d + k] * s.Rd[j * d + k];
        s.Rd[i * d + j] = u / s.Rd[j * d + j];
      }
    }
    for (int j = 0; j < d; ++j) {
      for (int i = 0; i < j; ++i) s.Yd[i * d + j] = dd_t(0.0);
      s.Yd[j * d + j] = dd_t(1.0) / s.Rd[j * d + j];
      for (int i = j + 1; i < d; ++i) {
        dd_t t(0.0);
        for (int k = j; k < i; ++k) t = t + s.Rd[i * d + k] * s.Yd[k * d + j];
        s.Yd[i * d + j] = -t / s.Rd[i * d + i];
      }
    }
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        dd_t t(0.0);
        for (int k = a > b ? a : b; k < d; ++k) t = t + s.Yd[k * d + a] * s.Yd[k * d + b];
        s.X[a * d + b] = t;
      }
    // X[Q] = -X[0] sum_T C[T] X[Q \ T], for the masks without the first index
    xnz[0] = true;
    for (int Q = 2; Q < NM; Q += 2) {
      xnz[Q] = false;
      for (int e = 0; e < dd2; ++e) s.acc[e] = dd_t(0.0);
      for (int T = Q; T != 0; T = (T - 1) & Q) {
        if (!cnz[T] || !xnz[Q ^ T]) continue;
        xnz[Q] = true;
        const dd_t* ct = s.C + T * dd2;
        const dd_t* xr = s.X + (Q ^ T) * dd2;
        for (int a = 0; a < d; ++a)
          for (int k = 0; k < d; ++k) {
            const dd_t c = ct[a * d + k];
            for (int b = 0; b < d; ++b) s.acc[a * d + b] = s.acc[a * d + b] + c * xr[k * d + b];
          }
      }
      if (!xnz[Q]) continue;
      dd_t* xq = s.X + Q * dd2;
      for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
          dd_t t(0.0);
          for (int k = 0; k < d; ++k) t = t + s.X[a * d + k] * s.acc[k * d + b];
          xq[a * d + b] = -t;
        }
    }
    // d^F log det C = sum_(T subset of F \ f) tr(X[T] C[f + (F \ f \ T)])
    const int rest = (NM - 1) ^ 1;
    dd_t r(0.0);
    for (int T = rest;; T = (T - 1) & rest) {
      const int U = 1 | (rest ^ T);
      if (xnz[T] && cnz[U]) {
        const dd_t* x = s.X + T * dd2;
        const dd_t* c = s.C + U * dd2;
        for (int a = 0; a < d; ++a)
          for (int b = 0; b < d; ++b) r = r + x[a * d + b] * c[b * d + a];
      }
      if (T == 0) break;
    }
    return r.hi + r.lo;
  }

 private:
  using dd_t = math::detail::ddm::dd;
  // the double-double part of the scratch buffer is used only as dd_t
  struct work {
    double *lam, *R, *u, *v;
    dd_t *C, *X, *acc, *Rd, *Yd;
  };
  LAPLACE_HD work carve(double* w) const {
    const int p = dim, d = n_factors, nm = detail_pdmat::generic_masks;
    work s;
    s.lam = w;
    s.R = s.lam + p * d;
    s.u = s.R + d * d;
    s.v = s.u + d;
    s.C = reinterpret_cast<dd_t*>(s.v + d);
    s.X = s.C + nm * d * d;
    s.acc = s.X + nm * d * d;
    s.Rd = s.acc + d * d;
    s.Yd = s.Rd + d * d;
    return s;
  }
  LAPLACE_HD detail_pdmat::trapezoid_loadings tz() const { return {dim, n_factors}; }

  // C = Lambda^T Lambda into c (full d x d)
  LAPLACE_HD void build_c(const double* lam, double* c) const {
    const int p = dim, d = n_factors;
    for (int a = 0; a < d; ++a)
      for (int b = 0; b < d; ++b) {
        double t = 0.0;
        for (int i = 0; i < p; ++i) t += lam[i * d + a] * lam[i * d + b];
        c[a * d + b] = t;
      }
  }
};

static_assert(psdmat_structure<reduced_rank_psdmat>);

}  // namespace laplace::pdmat
