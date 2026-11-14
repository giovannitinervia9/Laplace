// compound_symmetry_pdmat: M = (1 - rho) I + rho 1 1^T, unit diagonal, with
// rho = a + (1 - a) expit(theta), a = -1/(p - 1), the interval on which M is
// positive definite (note/pdmat_structures.md; interface in core.hpp).
//
// With s = expit(theta), q = 1 - s and E = e^theta:
//   1 - rho = p q/(p - 1),  1 + (p - 1) rho = p s,
//   d^K rho = p/(p - 1) expit^(K)(theta);
//   log det M = (p - 1) log(1 - rho) + log(1 + (p - 1) rho)
//             = (p - 1) log(p/(p - 1)) + log p + (p - 1) log q + log s,
//   whose first derivative is q - (p - 1) s and the K-th -p expit^(K-1)(theta);
//   M^-1 (Sherman-Morrison) has diagonal [(p^2 - 2p + 2) + (p - 1)^2 E + 1/E]/p^2
//   and off-diagonal -[(p - 2) + (p - 1) E - 1/E]/p^2;
//   L_00 = 1, L_i0 = rho, and for j >= 1, with u_j = A_j + B_j E,
//   A_j = p - 1 - j, B_j = (p - 1)(1 + j) (so that 1 + j rho is proportional to
//   u_j/(1 + E), with no cancellation since A_j, B_j >= 0):
//     L_jj = exp(g_j),  g_j = [log(p/(p - 1)) + log u_j - log u_(j-1) - log(1 + E)]/2,
//     L_ij = rho exp(r_j) (i > j),
//     r_j = [log(p/(p - 1)) + 2 log(p - 1) + log(1 + E) - log u_(j-1) - log u_j]/2.
// The derivatives of log u = log(A + B E) are expit^(n-1)(theta + log(B/A));
// those of exp(g) are exp(g) times the complete Bell polynomial of g', g'', ...
// and those of rho exp(r) follow by Leibniz. For p = 1, M = [1] and every
// derivative is zero.
#pragma once

#include <laplace/pdmat/core.hpp>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with compound symmetry (equicorrelation) structure.
 *
 * Maps 1 unconstrained real parameter \f$\theta_0\f$ to an equicorrelation matrix \f$M(\theta) = (1 - \rho) I + \rho \mathbf{1}\mathbf{1}^\top\f$
 * with natural correlation parameter \f$\rho = a + (1 - a)\operatorname{expit}(\theta_0)\f$, \f$a = -1/(p - 1)\f$.
 *
 * \par Measured accuracy
 * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
 * - \f$M\f$: 8.01 ulpN (\f$K=4\f$, gcc) / 6.67 ulpN (\f$K=4\f$, CUDA)
 * - \f$M^{-1}\f$: 0.81 ulpN (gcc) / 0.81 ulpN (CUDA)
 * - \f$L\f$: 9.34 ulpN (\f$K=5\f$, gcc) / 9.77 ulpN (\f$K=6\f$, CUDA)
 * - \f$\log \det M\f$: 8.01 ulp (\f$K=5\f$, gcc) / 7.91 ulp1 (\f$K=0\f$, CUDA)
 *
 * \see pdmat, pdmat_orders, pdmat_structure
 */
struct compound_symmetry_pdmat : pdmat_orders<compound_symmetry_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;

  /// \brief Compile-time descriptor: \c true, diagonal entries are identically 1.
  static constexpr bool unit_diagonal = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Sherman-Morrison inverse is implemented.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Cholesky factor is implemented.
  static constexpr bool closed_factor = true;
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  compound_symmetry_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs a compound symmetry structured matrix of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit compound_symmetry_pdmat(int p) : dim(p) {}

  /**
   * \ingroup pdmat
   * \brief Number of parameters \f$q = 1\f$.
   * \return Always 1.
   */
  LAPLACE_HD int n_params() const { return 1; }

  /**
   * \ingroup pdmat
   * \brief Doubles of caller scratch any function needs: 0 for this structure.
   * \return 0.
   */
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed compound symmetry matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ii} = 1\f$ on the diagonal and \f$M_{ij} = \rho\f$ for \f$i \ne j\f$,
   * where \f$\rho = a + (1 - a)\operatorname{expit}(\theta_0)\f$ with \f$a = -1/(p - 1)\f$ (or \f$M = [1]\f$ for \f$p = 1\f$).
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 8.01 ulpN (\f$K=4\f$, gcc) / 6.67 ulpN (\f$K=4\f$, CUDA); test bounds 1 to 13 ulpN.
   *
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, [[maybe_unused]] double* w) const {
    const double rho = dim == 1 ? 0.0 : rho_of(theta[0]);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) m[packed_index(i, j)] = i == j ? 1.0 : rho;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * For \f$p = 1\f$, returns \f$0.0\f$. For \f$p \ge 2\f$, computes:
   * \f[
   * \log \det M(\theta) = (p - 1)\log\left(\frac{p}{p - 1}\right) + \log p + (p - 1)\log(1 - s) + \log s
   * \f]
   * where \f$s = \operatorname{expit}(\theta_0)\f$.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * \par Measured accuracy
   * Maximum error 8.01 ulp (\f$K=5\f$, gcc) / 7.91 ulp1 (\f$K=0\f$, CUDA); test bounds 3 to 13 ulp.
   *
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const {
    if (dim == 1) return 0.0;
    namespace m = detail_pdmat::m;
    const double p = dim, x = theta[0];
    // log s = -log(1 + e^-x), log q = log s - x
    const double log_s = x > 0.0 ? -Kokkos::log1p(m::exp(-x)) : x - Kokkos::log1p(m::exp(x));
    return -(p - 1.0) * Kokkos::log1p(-1.0 / p) + m::log(p) + (p - 1.0) * (log_s - x) + log_s;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Evaluates the Sherman-Morrison closed form in \f$E = e^{\theta_0}\f$.
   * For \f$p = 1\f$, \f$M^{-1} = [1]\f$.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 0.81 ulpN (gcc) / 0.81 ulpN (CUDA); test bounds 1 to 2 ulpN.
   *
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, [[maybe_unused]] double* w) const { inverse_entries(theta[0], 0, m); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Evaluates the closed \f$L D L^\top\f$ representation, satisfying \f$M = L L^\top\f$.
   * For \f$p = 1\f$, \f$L = [1]\f$.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   *
   * \par Measured accuracy
   * Maximum error 9.34 ulpN (\f$K=5\f$, gcc) / 9.77 ulpN (\f$K=6\f$, CUDA); test bounds 3 to 13 ulpN.
   *
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, [[maybe_unused]] double* w) const { cholesky_entries(theta[0], 0, l); }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Computes \f$\frac{p}{p - 1}\operatorname{expit}^{(K)}(\theta_0)\f$ for off-diagonal entries
   * and zero for diagonal entries.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 8.01 ulpN (\f$K=4\f$, gcc) / 6.67 ulpN (\f$K=4\f$, CUDA); test bounds 1 to 13 ulpN.
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    if (dim == 1) return false;
    const double d = dim / (dim - 1.0) * detail_pdmat::expit_d(K, theta[0]);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) m[packed_index(i, j)] = i == j ? 0.0 : d;
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$.
   *
   * Computes \f$q - (p - 1)s\f$ for \f$K = 1\f$, and \f$-p \operatorname{expit}^{(K-1)}(\theta_0)\f$ for \f$K \ge 2\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Value of the derivative, or \f$0.0\f$ if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 8.01 ulp (\f$K=5\f$, gcc) / 7.91 ulp1 (\f$K=0\f$, CUDA); test bounds 3 to 13 ulp.
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], [[maybe_unused]] double* w) const {
    if (dim == 1) return 0.0;
    if constexpr (K == 1) {
      const double s = detail_pdmat::expit_d(0, theta[0]), q = detail_pdmat::expit_d(0, -theta[0]);
      return q - (dim - 1.0) * s;
    } else {
      return -dim * detail_pdmat::expit_d(K - 1, theta[0]);
    }
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * Computes the analytic derivative of the Sherman-Morrison form in \f$E = e^{\theta_0}\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.81 ulpN (gcc) / 0.81 ulpN (CUDA); test bounds 1 to 2 ulpN.
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    if (dim == 1) return false;
    inverse_entries(theta[0], K, m);
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$.
   *
   * Computes derivatives of diagonal entries via Bell polynomials of \f$g_j\f$ and off-diagonal entries
   * via Leibniz expansion of \f$\rho \exp(r_j)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 9.34 ulpN (\f$K=5\f$, gcc) / 9.77 ulpN (\f$K=6\f$, CUDA); test bounds 3 to 13 ulpN.
   *
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* l, [[maybe_unused]] double* w) const {
    if (dim == 1) return false;
    cholesky_entries(theta[0], K, l);
    return true;
  }

 private:
  LAPLACE_HD double rho_of(double x) const {
    const double p = dim, s = detail_pdmat::expit_d(0, x), q = detail_pdmat::expit_d(0, -x);
    return s < 0.5 ? (p * s - 1.0) / (p - 1.0) : 1.0 - p * q / (p - 1.0);
  }

  // M^-1 or its derivative of order K
  LAPLACE_HD void inverse_entries(double x, int K, double* m) const {
    if (dim == 1) {
      m[0] = K == 0 ? 1.0 : 0.0;
      return;
    }
    const double p = dim, E = detail_pdmat::m::exp(x), Ei = detail_pdmat::m::exp(-x);
    const double sgn = K % 2 == 0 ? 1.0 : -1.0, p2 = p * p;
    const double diag = ((K == 0 ? p * p - 2.0 * p + 2.0 : 0.0) + (p - 1.0) * (p - 1.0) * E + sgn * Ei) / p2;
    const double off = -((K == 0 ? p - 2.0 : 0.0) + (p - 1.0) * E - sgn * Ei) / p2;
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j <= i; ++j) m[packed_index(i, j)] = i == j ? diag : off;
  }

  // L or its derivative of order K
  LAPLACE_HD void cholesky_entries(double x, int K, double* l) const {
    namespace dp = detail_pdmat;
    detail_pdmat::fill_zero(l, packed_size(dim));
    const double p = dim, c = dp::m::log(p / (p - 1.0)), lpm1 = dp::m::log(p - 1.0);
    l[0] = K == 0 ? 1.0 : 0.0;
    // derivatives of rho, orders 0..K
    double rho[7];
    dp::expit_all(x, K, rho);
    rho[0] = dim == 1 ? 0.0 : rho_of(x);
    for (int n = 1; n <= K; ++n) rho[n] *= p / (p - 1.0);
    for (int i = 1; i < dim; ++i) l[packed_index(i, 0)] = rho[K];
    double le[7], l0[7], l1[7];  // log(1 + E), log u_(j-1), log u_j and their derivatives
    dp::log_lin_exp_all(1.0, 1.0, x, K, le);
    dp::log_lin_exp_all(p - 1.0, p - 1.0, x, K, l1);  // u_0, carried to u_(j-1) below
    for (int j = 1; j < dim; ++j) {
      for (int n = 0; n <= K; ++n) l0[n] = l1[n];
      dp::log_lin_exp_all(p - 1.0 - j, (p - 1.0) * (1.0 + j), x, K, l1);  // u_j = A_j + B_j E
      // g = log L_jj, r = log(L_ij / rho), and their derivatives 1..K
      double g[7], r[7];
      const double g0 = 0.5 * (c + l1[0] - l0[0] - le[0]), r0 = 0.5 * (c + 2.0 * lpm1 + le[0] - l0[0] - l1[0]);
      for (int n = 1; n <= K; ++n) {
        g[n] = 0.5 * (l1[n] - l0[n] - le[n]);
        r[n] = 0.5 * (le[n] - l0[n] - l1[n]);
      }
      l[packed_index(j, j)] = dp::exp_chain(K, dp::m::exp(g0), g);
      if (j + 1 == dim) continue;
      // L_ij = rho R, R = exp(r): Leibniz over the orders of rho
      const double R0 = dp::m::exp(r0);
      double v = 0.0, binom = 1.0;
      for (int n = 0; n <= K; ++n) {
        v += binom * rho[n] * dp::exp_chain(K - n, R0, r);
        binom = binom * (K - n) / (n + 1);
      }
      for (int i = j + 1; i < dim; ++i) l[packed_index(i, j)] = v;
    }
  }
};

static_assert(pdmat_structure<compound_symmetry_pdmat>);

}  // namespace laplace::pdmat
