// scaled_identity_pdmat: M = sigma^2 I with theta = log sigma, one parameter
// for any dimension p (note/penalties.md section 5g: the matrix of the group
// lasso, an mvpowerexp with one scale; interface in core.hpp).
//
//   M_ii = e^(2 theta), log det M = 2 p theta, M^-1_ii = e^(-2 theta), L_ii = e^theta.
// Every derivative is along theta alone: c^K e^(c theta) on the diagonal with
// c = 2, -2, 1 for M, M^-1, L; log det has derivative 2 p and none above.
#pragma once

#include <laplace/pdmat/core.hpp>

#include <cmath>

namespace laplace::pdmat {

/**
 * \ingroup pdmat
 * \brief Structured positive definite matrix \f$M(\theta) = e^{2\theta} I_p\f$, one parameter for any dimension.
 *
 * Maps one unconstrained real parameter \f$\theta = \log\sigma\f$ to \f$\sigma^2 I_p\f$.
 *
 * \see pdmat, pdmat_orders, pdmat_structure, diagonal_pdmat
 */
struct scaled_identity_pdmat : pdmat_orders<scaled_identity_pdmat> {
  /// \brief Matrix dimension \f$p\f$.
  int dim = 1;

  /// \brief Compile-time descriptor: \c false, the diagonal \f$e^{2\theta}\f$ is not identically 1.
  static constexpr bool unit_diagonal = false;
  /// \brief Compile-time descriptor: \c true, exact closed-form inverse.
  static constexpr bool closed_inverse = true;
  /// \brief Compile-time descriptor: \c true, exact closed-form Cholesky factor.
  static constexpr bool closed_factor = true;
  /// \brief Compile-time descriptor: \c false, the map \f$\theta \mapsto M\f$ has no density on the matrices.
  static constexpr bool has_density = false;

  /// \brief Default constructor, dimension 1.
  scaled_identity_pdmat() = default;
  /**
   * \brief Constructs \f$\sigma^2 I_p\f$ of dimension \f$p\f$.
   * \param p Matrix dimension (\f$p \ge 1\f$).
   */
  LAPLACE_HD explicit scaled_identity_pdmat(int p) : dim(p) {}

  /// \brief Number of parameters, 1. \return 1.
  LAPLACE_HD int n_params() const { return 1; }
  /// \brief Doubles of caller scratch any function needs: 0. \return 0.
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \brief Starting parameter from a target packed matrix \f$\hat{M}\f$: \f$\theta = \frac{1}{2}\log\bar{m}\f$.
   *
   * Host-only. \f$\bar{m}\f$ is the mean of the diagonal, 1 when it is not positive. Exact up to
   * rounding on matrices of the structure.
   *
   * \param m_hat Target symmetric \f$p \times p\f$ matrix in packed lower-triangular storage.
   * \param theta Output, one double.
   */
  void params_from_matrix(const double* m_hat, double* theta) const {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += m_hat[packed_index(i, i)];
    theta[0] = s > 0.0 ? 0.5 * std::log(s / dim) : 0.0;
  }

  /// \brief Packed \f$M = e^{2\theta} I\f$. \param theta \f$\theta\f$. \param m Output, `packed_size(dim)`. \param w Unused.
  LAPLACE_HD void matrix(const double* theta, double* m, [[maybe_unused]] double* w) const { fill(1.0, 2.0 * theta[0], m); }
  /// \brief Packed \f$M^{-1} = e^{-2\theta} I\f$. \param theta \f$\theta\f$. \param m Output, `packed_size(dim)`. \param w Unused.
  LAPLACE_HD void inverse(const double* theta, double* m, [[maybe_unused]] double* w) const { fill(1.0, -2.0 * theta[0], m); }
  /// \brief Packed Cholesky factor \f$L = e^{\theta} I\f$. \param theta \f$\theta\f$. \param l Output, `packed_size(dim)`. \param w Unused.
  LAPLACE_HD void cholesky(const double* theta, double* l, [[maybe_unused]] double* w) const { fill(1.0, theta[0], l); }
  /// \brief \f$\log\det M = 2p\theta\f$. \param theta \f$\theta\f$. \param w Unused. \return The log-determinant.
  LAPLACE_HD double log_det(const double* theta, [[maybe_unused]] double* w) const { return 2.0 * dim * theta[0]; }

  /**
   * \brief \f$\partial^K M / \partial\theta^K = 2^K e^{2\theta} I\f$.
   * \tparam K Order (\f$1 \le K \le 6\f$). \param theta \f$\theta\f$. \param idx Indices, all 0.
   * \param m Output, `packed_size(dim)`. \param w Unused. \return \c true.
   */
  template <int K>
  LAPLACE_HD bool matrix_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    fill(detail_pdmat::ipow(2.0, K), 2.0 * theta[0], m);
    return true;
  }
  /**
   * \brief \f$\partial^K \log\det M / \partial\theta^K\f$: \f$2p\f$ at \f$K = 1\f$, 0 above.
   * \tparam K Order. \param theta Unused. \param idx Unused. \param w Unused. \return The derivative.
   */
  template <int K>
  LAPLACE_HD double log_det_dk([[maybe_unused]] const double* theta, [[maybe_unused]] const int (&idx)[K],
                               [[maybe_unused]] double* w) const {
    return K == 1 ? 2.0 * dim : 0.0;
  }
  /**
   * \brief \f$\partial^K M^{-1} / \partial\theta^K = (-2)^K e^{-2\theta} I\f$.
   * \tparam K Order. \param theta \f$\theta\f$. \param idx Indices, all 0.
   * \param m Output, `packed_size(dim)`. \param w Unused. \return \c true.
   */
  template <int K>
  LAPLACE_HD bool inverse_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* m, [[maybe_unused]] double* w) const {
    fill(detail_pdmat::ipow(-2.0, K), -2.0 * theta[0], m);
    return true;
  }
  /**
   * \brief \f$\partial^K L / \partial\theta^K = e^{\theta} I\f$.
   * \tparam K Order. \param theta \f$\theta\f$. \param idx Indices, all 0.
   * \param l Output, `packed_size(dim)`. \param w Unused. \return \c true.
   */
  template <int K>
  LAPLACE_HD bool cholesky_dk(const double* theta, [[maybe_unused]] const int (&idx)[K], double* l, [[maybe_unused]] double* w) const {
    fill(1.0, theta[0], l);
    return true;
  }

 private:
  // c e^e on the diagonal, 0 elsewhere
  LAPLACE_HD void fill(double c, double e, double* m) const {
    detail_pdmat::fill_zero(m, packed_size(dim));
    const double v = c * detail_pdmat::m::exp(e);
    for (int i = 0; i < dim; ++i) m[packed_index(i, i)] = v;
  }
};

static_assert(pdmat_structure<scaled_identity_pdmat>);

}  // namespace laplace::pdmat
