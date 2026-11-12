// ar1_pdmat: the AR(1) correlation matrix M_ij = rho^|i - j|, with
// rho = tanh(theta) (note/pdmat_structures.md; interface in core.hpp).
//
// With T = tanh(theta), S = 1 - T^2 = sech^2(theta):
//   M_ij = T^|i-j|, log det M = (p - 1) log(1 - rho^2) = -2 (p - 1) log cosh(theta),
//   M^-1 tridiagonal: cosh^2(theta) at (0, 0) and (p-1, p-1), cosh(2 theta) on
//   the rest of the diagonal, -sinh(2 theta)/2 next to it;
//   L_i0 = T^i, L_ij = T^(i-j) sech(theta) for 1 <= j <= i.
// Derivatives: d^K T^k sech^h by the recursion of detail_pdmat::tanh_power_d,
// d^K log det = -2 (p - 1) tanh^(K-1)(theta), and those of cosh and sinh.
// For p = 1, M = [1] and every derivative is zero.
#pragma once

#include <laplace/pdmat/core.hpp>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with first-order autoregressive AR(1) structure.
 *
 * Maps 1 unconstrained real parameter \f$\theta_0\f$ to a correlation matrix \f$M(\theta)\f$ with
 * entries \f$M_{ij}(\theta) = \rho^{|i - j|}\f$ and natural correlation parameter \f$\rho = \tanh\theta_0 \in (-1, 1)\f$.
 *
 * \par Measured accuracy
 * Maximum error over orders 0–6 on gcc/OpenMP and CUDA (RTX 4070 SUPER):
 * - \f$M\f$: 13.0 ulpN (\f$K=6\f$, gcc) / 11.1 ulpN (\f$K=6\f$, CUDA)
 * - \f$M^{-1}\f$: 0.52 ulpN (gcc) / 0.58 ulpN (CUDA)
 * - \f$L\f$: 8.22 ulpN (\f$K=4\f$, gcc) / 8.22 ulpN (\f$K=4\f$, CUDA)
 * - \f$\log \det M\f$: 3.92 ulp (\f$K=6\f$, gcc) / 4.83 ulp (\f$K=6\f$, CUDA)
 *
 * \see pdmat, pdmat_orders, pdmat_structure
 */
struct ar1_pdmat : pdmat_orders<ar1_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;

  /// \brief Compile-time descriptor: \c true, diagonal entries are identically 1.
  static constexpr bool unit_diagonal = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form tridiagonal inverse is implemented.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Cholesky factor is implemented.
  static constexpr bool closed_factor = true;

  /**
   * \ingroup pdmat
   * \brief Default constructor initializing dimension to 1.
   */
  ar1_pdmat() = default;
  /**
   * \ingroup pdmat
   * \brief Constructs an AR(1) structured matrix of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit ar1_pdmat(int p) : dim(p) {}

  /**
   * \ingroup pdmat
   * \brief Number of parameters \f$q = 1\f$.
   * \return Always 1.
   */
  LAPLACE_HD int n_params() const { return 1; }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed AR(1) correlation matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ij} = (\tanh\theta_0)^{|i - j|}\f$ (or \f$M = [1]\f$ for \f$p = 1\f$).
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   *
   * \par Measured accuracy
   * Maximum error 13.0 ulpN (\f$K=6\f$, gcc) / 11.1 ulpN (\f$K=6\f$, CUDA); test bounds 5 to 20 ulpN.
   *
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m) const {
    const double T = detail_pdmat::tanh_of(theta[0]).T;
    for (int k = 0; k < dim; ++k) {
      const double c = detail_pdmat::ipow(T, k);
      for (int j = 0; j + k < dim; ++j) m[packed_index(j + k, j)] = c;
    }
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the log-determinant \f$\log \det M(\theta)\f$.
   *
   * For \f$p = 1\f$, returns \f$0.0\f$. For \f$p \ge 2\f$, computes:
   * \f[
   * \log \det M(\theta) = -2(p - 1)\log\cosh\theta_0
   * \f]
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \return Log-determinant of \f$M(\theta)\f$.
   *
   * \par Measured accuracy
   * Maximum error 3.92 ulp (\f$K=6\f$, gcc) / 4.83 ulp (\f$K=6\f$, CUDA); test bounds 2 to 8 ulp.
   *
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta) const {
    const double a = theta[0] < 0.0 ? -theta[0] : theta[0];
    const double log_cosh = a + Kokkos::log1p(detail_pdmat::m::exp(-2.0 * a)) - 0.69314718055994530942;
    return -2.0 * (dim - 1) * log_cosh;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed tridiagonal inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Computes \f$\cosh^2\theta_0\f$ at \f$(0, 0)\f$ and \f$(p - 1, p - 1)\f$, \f$\cosh(2\theta_0)\f$
   * on inner diagonal entries, and \f$-\frac{1}{2}\sinh(2\theta_0)\f$ on the subdiagonal.
   * For \f$p = 1\f$, \f$M^{-1} = [1]\f$.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   *
   * \par Measured accuracy
   * Maximum error 0.52 ulpN (gcc) / 0.58 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m) const { inverse_entries(theta[0], 0, m); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Computes \f$L_{i0} = T^i\f$ for column 0, and \f$L_{ij} = T^{i - j}\operatorname{sech}\theta_0\f$
   * for \f$j \ge 1\f$, where \f$T = \tanh\theta_0\f$, satisfying \f$M = L L^\top\f$.
   * For \f$p = 1\f$, \f$L = [1]\f$.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   *
   * \par Measured accuracy
   * Maximum error 8.22 ulpN (\f$K=4\f$, gcc) / 8.22 ulpN (\f$K=4\f$, CUDA); test bounds 5 to 13 ulpN.
   *
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l) const {
    const auto t = detail_pdmat::tanh_of(theta[0]);
    for (int k = 0; k < dim; ++k) {
      const double c = detail_pdmat::ipow(t.T, k);
      for (int j = 0; j + k < dim; ++j) l[packed_index(j + k, j)] = j == 0 ? c : c * t.sech;
    }
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates \f$\frac{\mathrm{d}^K}{\mathrm{d}\theta^K} T^{|i - j|}\f$ band by band
   * using the recurrence preserving factor \f$S = 1 - T^2\f$. Diagonal entries are 0.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 13.0 ulpN (\f$K=6\f$, gcc) / 11.1 ulpN (\f$K=6\f$, CUDA); test bounds 5 to 20 ulpN.
   *
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m) const {
    if (dim == 1) return false;
    const auto t = detail_pdmat::tanh_of(theta[0]);
    // M_ij depends on i - j only: one diagonal band at a time
    for (int k = 0; k < dim; ++k) {
      const double c = k == 0 ? 0.0 : detail_pdmat::tanh_power_d(K, k, 0, t);
      for (int j = 0; j + k < dim; ++j) m[packed_index(j + k, j)] = c;
    }
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$\log \det M(\theta)\f$ of order \f$K\f$.
   *
   * Computes \f$-2(p - 1)\tanh^{(K-1)}(\theta_0)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \return Value of the derivative, or \f$0.0\f$ if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 3.92 ulp (\f$K=6\f$, gcc) / 4.83 ulp (\f$K=6\f$, CUDA); test bounds 2 to 8 ulp.
   *
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, [[maybe_unused]] const int (&idx)[K]) const {
    if (dim == 1) return 0.0;
    return -2.0 * (dim - 1) * detail_pdmat::tanh_power_d(K - 1, 1, 0, detail_pdmat::tanh_of(theta[0]));
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * Computes derivatives of the tridiagonal inverse using \f$2^K\cosh(2\theta_0)\f$
   * and \f$2^K\sinh(2\theta_0)\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 0.52 ulpN (gcc) / 0.58 ulpN (CUDA); test bound 1.0 ulpN.
   *
   * \see inverse(), pdmat_orders::inverse_d1()
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m) const {
    if (dim == 1) return false;
    inverse_entries(theta[0], K, m);
    return true;
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of the Cholesky factor \f$L(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates derivatives of \f$T^k \operatorname{sech}^h\theta_0\f$ with \f$h = 0\f$ for column 0
   * and \f$h = 1\f$ for columns \f$j \ge 1\f$ via the recurrence preserving \f$S\f$.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0 (the structure has one parameter).
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   *
   * \par Measured accuracy
   * Maximum error over orders 0–6: 8.22 ulpN (\f$K=4\f$, gcc) / 8.22 ulpN (\f$K=4\f$, CUDA); test bounds 5 to 13 ulpN.
   *
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* l) const {
    if (dim == 1) return false;
    const auto t = detail_pdmat::tanh_of(theta[0]);
    // L_ij depends on i - j and on whether j = 0: one column of each kind
    for (int k = 0; k < dim; ++k) {
      const double c0 = k == 0 ? 0.0 : detail_pdmat::tanh_power_d(K, k, 0, t);
      const double c1 = detail_pdmat::tanh_power_d(K, k, 1, t);
      for (int j = 0; j + k < dim; ++j) l[packed_index(j + k, j)] = j == 0 ? c0 : c1;
    }
    return true;
  }

 private:
  // M^-1 or its derivative of order K: d^K cosh(2 theta) = 2^K cosh or sinh,
  // and cosh^2(theta) = (1 + cosh(2 theta))/2.
  LAPLACE_HD void inverse_entries(double theta, int K, double* m) const {
    detail_pdmat::fill_zero(m, packed_size(dim));
    if (dim == 1) {
      m[0] = 1.0;
      return;
    }
    const double c2 = Kokkos::cosh(2.0 * theta), s2 = Kokkos::sinh(2.0 * theta);
    const double scale = detail_pdmat::ipow(2.0, K);
    const double dc = scale * (K % 2 == 0 ? c2 : s2), ds = scale * (K % 2 == 0 ? s2 : c2);
    const double end = K == 0 ? 0.5 * (1.0 + c2) : 0.5 * dc;
    for (int i = 0; i < dim; ++i) {
      m[packed_index(i, i)] = (i == 0 || i == dim - 1) ? end : dc;
      if (i > 0) m[packed_index(i, i - 1)] = -0.5 * ds;
    }
  }
};

static_assert(pdmat_structure<ar1_pdmat>);

}  // namespace laplace::pdmat
