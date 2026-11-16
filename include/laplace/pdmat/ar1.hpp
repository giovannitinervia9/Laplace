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
#include <laplace/pdmat/detail/from_matrix.hpp>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix with first-order autoregressive AR(1) structure.
 *
 * Maps 1 unconstrained parameter \f$\theta_0\f$ to correlation matrix \f$M_{ij}(\theta) = \rho^{|i - j|}\f$
 * (\f$p \ge 1\f$, \f$\rho = \tanh\theta_0 \in (-1, 1)\f$; \f$M = [1]\f$ at \f$p = 1\f$).
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
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

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
   * \brief Doubles of caller scratch any function needs: 0 for this structure.
   * \return 0.
   */
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \ingroup pdmat
   * \brief Computes starting parameters \f$\theta\f$ from a target packed covariance matrix \f$\hat{M}\f$.
   *
   * Host-only function. Forms the correlation matrix of \f$\hat{M}\f$ after flooring
   * diagonal entries at \f$10^{-6}\f$ times the largest, averages the lag-1 correlations,
   * clamps \f$\rho\f$ to \f$[-0.99, 0.99]\f$, and sets \f$\theta_0 = \operatorname{atanh}\rho\f$
   * (or \f$0.0\f$ for \f$p = 1\f$).
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output array of \c n_params() doubles inside the domain.
   *
   * Accuracy: recovers \f$\theta\f$ from `matrix()` within \f$8.9 \times 10^{-16}\f$ on gcc 14.2/OpenMP (test bounds \f$10^{-14}\f$ and \f$10^{-13}\f$).
   *
   * \see matrix(), pdmat
   */
  // host: rho the mean lag-1 correlation of m_hat, clamped
  void params_from_matrix(const double* m_hat, double* theta) const {
    const auto r = detail_pdmat::correlation_of(dim, m_hat);
    double s = 0.0;
    for (int i = 1; i < dim; ++i) s += r[packed_index(i, i - 1)];
    theta[0] = dim == 1 ? 0.0 : std::atanh(detail_pdmat::clamp_rho(s / (dim - 1)));
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed AR(1) correlation matrix \f$M(\theta)\f$.
   *
   * Computes \f$M_{ij} = (\tanh\theta_0)^{|i - j|}\f$ (\f$M = [1]\f$ for \f$p = 1\f$).
   * Maximum error 13.0 ulpN on gcc and 11.1 ulpN on CUDA; test bounds 5 to 20 ulpN.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \see inverse(), cholesky(), log_det(), matrix_dk()
   */
  LAPLACE_HD void matrix(const double* theta, double* m, [[maybe_unused]] double* w) const {
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
   * Computes \f$\log \det M(\theta) = -2(p - 1)\log\cosh\theta_0\f$ (\f$0.0\f$ for \f$p = 1\f$).
   * Maximum error 3.92 ulp on gcc and 4.83 ulp on CUDA; test bounds 2 to 8 ulp.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Log-determinant of \f$M(\theta)\f$.
   * \see matrix(), log_det_dk()
   */
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const {
    const double a = theta[0] < 0.0 ? -theta[0] : theta[0];
    const double log_cosh = a + Kokkos::log1p(detail_pdmat::m::exp(-2.0 * a)) - 0.69314718055994530942;
    return -2.0 * (dim - 1) * log_cosh;
  }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed tridiagonal inverse matrix \f$M^{-1}(\theta)\f$.
   *
   * Computes \f$\cosh^2\theta_0\f$ at endpoints \f$(0, 0)\f$ and \f$(p - 1, p - 1)\f$, \f$\cosh(2\theta_0)\f$
   * on interior diagonal entries, and \f$-\frac{1}{2}\sinh(2\theta_0)\f$ on the subdiagonal (\f$M^{-1} = [1]\f$ for \f$p = 1\f$).
   * Maximum error 0.52 ulpN on gcc and 0.58 ulpN on CUDA; test bound 1 ulpN.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \see matrix(), cholesky(), log_det(), inverse_dk()
   */
  LAPLACE_HD void inverse(const double* theta, double* m, [[maybe_unused]] double* w) const { inverse_entries(theta[0], 0, m); }

  /**
   * \ingroup pdmat
   * \brief Evaluates the packed lower Cholesky factor \f$L(\theta)\f$.
   *
   * Computes \f$L_{i0} = T^i\f$ and \f$L_{ij} = T^{i - j}\operatorname{sech}\theta_0\f$ (\f$j \ge 1\f$, \f$T = \tanh\theta_0\f$; \f$L = [1]\f$ for \f$p = 1\f$).
   * Maximum error 8.22 ulpN (gcc and CUDA); test bounds 5 to 13 ulpN.
   *
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \see matrix(), inverse(), log_det(), cholesky_dk()
   */
  LAPLACE_HD void cholesky(const double* theta, double* l, [[maybe_unused]] double* w) const {
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
   * Evaluates \f$\frac{\mathrm{d}^K}{\mathrm{d}\theta^K} T^{|i - j|}\f$ band by band keeping factor \f$S = 1 - T^2\f$
   * (zero diagonal; returns \c false for \f$p = 1\f$). Maximum error over orders 1–6: 13.0 ulpN on gcc and 11.1 ulpN on CUDA; test bounds 5 to 20 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   * \see matrix(), pdmat_orders::matrix_d1()
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
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
   * Computes \f$-2(p - 1)\tanh^{(K-1)}(\theta_0)\f$ (\f$0.0\f$ for \f$p = 1\f$).
   * Maximum error over orders 1–6: 3.92 ulp on gcc and 4.83 ulp on CUDA; test bounds 2 to 8 ulp.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return Value of the derivative, or \f$0.0\f$ if \f$p = 1\f$.
   * \see log_det(), pdmat_orders::log_det_d1()
   */
  template <int K>
  LAPLACE_HD double log_det_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], [[maybe_unused]] double* w) const {
    if (dim == 1) return 0.0;
    return -2.0 * (dim - 1) * detail_pdmat::tanh_power_d(K - 1, 1, 0, detail_pdmat::tanh_of(theta[0]));
  }

  /**
   * \ingroup pdmat
   * \brief Partial derivative of \f$M^{-1}(\theta)\f$ of order \f$K\f$.
   *
   * Evaluates derivatives of the tridiagonal inverse via \f$2^K\cosh(2\theta_0)\f$ and \f$2^K\sinh(2\theta_0)\f$ (returns \c false for \f$p = 1\f$).
   * Maximum error over orders 1–6: 0.52 ulpN on gcc and 0.58 ulpN on CUDA; test bound 1 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0.
   * \param m Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
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
   * Differentiates \f$T^k \operatorname{sech}^h\theta_0\f$ with \f$h = 0\f$ on column 0 and \f$h = 1\f$ for \f$j \ge 1\f$
   * (returns \c false for \f$p = 1\f$). Maximum error over orders 1–6: 8.22 ulpN (gcc and CUDA); test bounds 5 to 13 ulpN.
   *
   * \tparam K Derivative order (\f$1 \le K \le 6\f$).
   * \param theta Array of 1 parameter \f$\theta_0\f$.
   * \param idx Sorted tuple of \f$K\f$ parameter indices, all 0.
   * \param l Output buffer of at least \f$p(p + 1)/2\f$ doubles for packed lower triangle.
   * \param w Caller scratch buffer of \c scratch_size() doubles (unused: the size is 0).
   * \return \c true if \f$p \ge 2\f$; \c false and buffer left untouched if \f$p = 1\f$.
   * \see cholesky(), pdmat_orders::cholesky_d1()
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* l, [[maybe_unused]] double* w) const {
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
