// SCAD and MCP, the non-convex penalties of step 5 (note/penalties.md section
// 5e, decided with Giovanni on 2026-11-18). Not minus the log of a prior: the
// flat tail makes exp(-pen) unnormalizable for every theta. With u = |beta_j|,
// pen(beta) = sum_j q(u), theta = (lambda, gamma) or (lambda, a) shared by all
// coordinates:
//
//   MCP   q = lambda u - u^2/(2 gamma)                 u <= gamma lambda
//         q = gamma lambda^2 / 2                        beyond
//   SCAD  q = lambda u                                  u <= lambda
//         q = lambda u - (u - lambda)^2 / (2 (a - 1))   lambda < u <= a lambda
//         q = (a + 1) lambda^2 / 2                      beyond
//
// (Fan and Li's middle piece (2 a lambda u - u^2 - lambda^2)/(2(a - 1))
// rewritten). The knots are C1 in beta and in theta; at a knot the value and
// every derivative come from the inner piece, at u = 0 the first derivative is
// sign(0) = 0. One function per derivative order, in the shapes below.
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>
#include <laplace/penalty/core.hpp>

#include <Kokkos_Core.hpp>

namespace laplace::penalty {

/**
 * \ingroup penalty_nonsmooth
 * \brief State of a non-convex penalty: \f$\lambda\f$ and the shape parameter (\f$\gamma\f$ or \f$a\f$).
 */
struct nonconvex_penalty_state {
  /// The level \f$\lambda\f$.
  double lambda;
  /// The concavity \f$\gamma\f$ of MCP or the parameter \f$a\f$ of SCAD.
  double shape;
};

namespace detail_nonconvex {

// MCP: p = gamma. u >= 0; q1 and q2 are dq/du and d2q/du2; ql*, qp* the same in lambda and gamma.
struct mcp_shape {
  LAPLACE_HD static bool inner(double u, double l, double p) { return u <= p * l; }
  LAPLACE_HD static double q0(double u, double l, double p) { return inner(u, l, p) ? l * u - u * u / (2.0 * p) : 0.5 * p * l * l; }
  LAPLACE_HD static double q1(double u, double l, double p) { return inner(u, l, p) ? l - u / p : 0.0; }
  LAPLACE_HD static double q2(double u, double l, double p) { return inner(u, l, p) ? -1.0 / p : 0.0; }
  LAPLACE_HD static double ql0(double u, double l, double p) { return inner(u, l, p) ? u : p * l; }
  LAPLACE_HD static double ql1(double u, double l, double p) { return inner(u, l, p) ? 1.0 : 0.0; }
  LAPLACE_HD static double ql2(double, double, double) { return 0.0; }
  LAPLACE_HD static double qp0(double u, double l, double p) { return inner(u, l, p) ? u * u / (2.0 * p * p) : 0.5 * l * l; }
  LAPLACE_HD static double qp1(double u, double l, double p) { return inner(u, l, p) ? u / (p * p) : 0.0; }
  LAPLACE_HD static double qp2(double u, double l, double p) { return inner(u, l, p) ? 1.0 / (p * p) : 0.0; }
  LAPLACE_HD static double rho(double p) { return 1.0 / p; }
  // firm threshold, t <= gamma (the middle region is empty at t = gamma);
  // gamma - t is an exact difference near t = gamma, where 1 - t/gamma rounds first
  LAPLACE_HD static double prox(double v, double t, double l, double p) {
    const double u = Kokkos::fabs(v);
    if (u <= t * l) return 0.0;
    if (u <= p * l) return Kokkos::copysign((u - t * l) * p / (p - t), v);
    return v;
  }
};

// SCAD: p = a, c = a - 1.
struct scad_shape {
  LAPLACE_HD static int piece(double u, double l, double p) { return u <= l ? 0 : (u <= p * l ? 1 : 2); }
  LAPLACE_HD static double q0(double u, double l, double p) {
    const int k = piece(u, l, p);
    return k == 0 ? l * u : (k == 1 ? l * u - (u - l) * (u - l) / (2.0 * (p - 1.0)) : 0.5 * (p + 1.0) * l * l);
  }
  LAPLACE_HD static double q1(double u, double l, double p) {
    const int k = piece(u, l, p);
    return k == 0 ? l : (k == 1 ? (p * l - u) / (p - 1.0) : 0.0);
  }
  LAPLACE_HD static double q2(double u, double l, double p) { return piece(u, l, p) == 1 ? -1.0 / (p - 1.0) : 0.0; }
  LAPLACE_HD static double ql0(double u, double l, double p) {
    const int k = piece(u, l, p);
    return k == 0 ? u : (k == 1 ? (p * u - l) / (p - 1.0) : (p + 1.0) * l);
  }
  LAPLACE_HD static double ql1(double u, double l, double p) {
    const int k = piece(u, l, p);
    return k == 0 ? 1.0 : (k == 1 ? p / (p - 1.0) : 0.0);
  }
  LAPLACE_HD static double ql2(double, double, double) { return 0.0; }
  LAPLACE_HD static double qp0(double u, double l, double p) {
    const int k = piece(u, l, p);
    const double c = p - 1.0;
    return k == 0 ? 0.0 : (k == 1 ? (u - l) * (u - l) / (2.0 * c * c) : 0.5 * l * l);
  }
  LAPLACE_HD static double qp1(double u, double l, double p) {
    const double c = p - 1.0;
    return piece(u, l, p) == 1 ? (u - l) / (c * c) : 0.0;
  }
  LAPLACE_HD static double qp2(double u, double l, double p) {
    const double c = p - 1.0;
    return piece(u, l, p) == 1 ? 1.0 / (c * c) : 0.0;
  }
  LAPLACE_HD static double rho(double p) { return 1.0 / (p - 1.0); }
  // three pieces, t <= a - 1 (the middle region is empty at t = a - 1); the
  // middle as u less a correction, a lambda - u and a - 1 - t exact differences
  LAPLACE_HD static double prox(double v, double t, double l, double p) {
    const double u = Kokkos::fabs(v);
    if (u <= l * (1.0 + t)) return Kokkos::copysign(Kokkos::fmax(u - t * l, 0.0), v);
    if (u <= p * l) return Kokkos::copysign(u - t * (p * l - u) / (p - 1.0 - t), v);
    return v;
  }
};

}  // namespace detail_nonconvex

/**
 * \ingroup penalty_nonsmooth
 * \brief A separable non-convex penalty \f$\sum_j q(|\beta_j|; \lambda, \cdot)\f$ with a flat tail: SCAD or MCP.
 *
 * Use the aliases `scad_penalty` and `mcp_penalty`. Class `unnormalizable`,
 * \f$C^0\f$ in \f$\beta\f$ with a kink at each \f$\beta_j = 0\f$, \f$C^1\f$
 * in \f$\theta\f$, separable, closed-form proximal operator. The
 * hyperparameters are shared by all coordinates; weights per coordinate come
 * through `linear_map_penalty` with `diagonal_map`.
 *
 * \tparam Shape `detail_nonconvex::mcp_shape` or `detail_nonconvex::scad_shape`.
 */
template <class Shape>
struct separable_nonconvex_penalty {
  /// \f$\int \exp(-\mathrm{pen})\,\mathrm{d}\beta = \infty\f$ for every \f$\theta\f$.
  static constexpr penalty_normalization normalization = penalty_normalization::unnormalizable;
  /// Kink at \f$\beta_j = 0\f$; the knots are \f$C^1\f$.
  static constexpr int beta_regularity = 0;
  /// A kink on each hyperplane \f$\beta_j = 0\f$.
  static constexpr penalty_kink_geometry kinks = penalty_kink_geometry::per_coordinate;
  /// Firm threshold (MCP) or three-piece threshold (SCAD).
  static constexpr penalty_prox_kind prox = penalty_prox_kind::closed_form;
  /// A sum over coordinates.
  static constexpr penalty_structure structure = penalty_structure::separable;
  /// \f$C^1\f$ in \f$\theta\f$: the knots move with \f$\theta\f$.
  static constexpr int theta_regularity = 1;
  /// Not a quadratic penalty.
  static constexpr bool q_linear_in_theta = false;
  /// Not a scale mixture of normals.
  static constexpr penalty_scale_mixture scale_mixture = penalty_scale_mixture::none;
  /// State: \f$\lambda\f$ and the shape parameter.
  using state_type = nonconvex_penalty_state;

  /// The dimension \f$d\f$ of \f$\beta\f$.
  int size = 0;

  /// \return The dimension \f$d\f$ of \f$\beta\f$.
  LAPLACE_HD int dim() const { return size; }
  /// \return 2: \f$\lambda\f$ and the shape parameter.
  LAPLACE_HD int theta_size() const { return 2; }
  /// \return 0: the state is returned by value.
  LAPLACE_HD int state_size() const { return 0; }
  /// \return 0: no scratch.
  LAPLACE_HD int scratch_size() const { return 0; }

  /**
   * \brief Read the hyperparameters.
   * \param theta \f$(\lambda, \gamma)\f$ or \f$(\lambda, a)\f$.
   * \return The state.
   */
  LAPLACE_HD state_type theta_state(const double* theta, double*) const { return {theta[0], theta[1]}; }

  /**
   * \brief The weak-convexity constant \f$\rho = \sup(-\mathrm{pen}'')\f$: \f$1/\gamma\f$ (MCP), \f$1/(a - 1)\f$ (SCAD).
   * \param theta Hyperparameters.
   * \return \f$\rho\f$.
   */
  LAPLACE_HD double weak_convexity(const double* theta) const { return Shape::rho(theta[1]); }

  /**
   * \brief The penalty value \f$\sum_j q(|\beta_j|)\f$.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \return Penalty value.
   */
  LAPLACE_HD double value(const state_type& st, const double* beta, double*) const {
    double r = 0.0;
    for (int j = 0; j < size; ++j) r += Shape::q0(Kokkos::fabs(beta[j]), st.lambda, st.shape);
    return r;
  }

  /**
   * \brief Derivatives: `J` in \f$\beta\f$ (one double per coordinate) and `K <= 1` in \f$\theta\f$.
   *
   * `J = 0` writes one double (the value, or its derivative in
   * \f$\theta_{\mathrm{idx}[0]}\f$, summed over coordinates); `J = 1, 2`
   * write `dim()` doubles. At \f$\beta_j = 0\f$ the odd order uses
   * \f$\operatorname{sign}(0) = 0\f$; at a knot every derivative is that of
   * the inner piece. `J >= 3` is zero almost everywhere: `false`, `out`
   * untouched.
   *
   * \tparam J Derivative order in \f$\beta\f$.
   * \tparam K Derivative order in \f$\theta\f$, 0 or 1.
   * \param st State from `theta_state()`.
   * \param beta Coefficients, length `dim()`.
   * \param idx `idx[0]` is 0 for \f$\lambda\f$, 1 for the shape parameter (unused when `K == 0`).
   * \param out Output buffer.
   * \return `true` if written; `false` when zero by structure.
   */
  template <int J, int K>
  LAPLACE_HD bool value_dbeta_dk(const state_type& st, const double* beta, const int* idx, double* out, double*) const {
    static_assert(K == 0 || K == 1, "separable_nonconvex_penalty: theta derivatives of order 0 or 1 only (C1 in theta)");
    if constexpr (J >= 3) {
      return false;
    } else {
      const double l = st.lambda, p = st.shape;
      const int which = K == 0 ? -1 : idx[0];
      if constexpr (J == 0) out[0] = 0.0;
      for (int j = 0; j < size; ++j) {
        const double u = Kokkos::fabs(beta[j]);
        const double s = beta[j] > 0.0 ? 1.0 : (beta[j] < 0.0 ? -1.0 : 0.0);
        if constexpr (J == 0) {
          out[0] += which < 0 ? Shape::q0(u, l, p) : (which == 0 ? Shape::ql0(u, l, p) : Shape::qp0(u, l, p));
        } else if constexpr (J == 1) {
          out[j] = s * (which < 0 ? Shape::q1(u, l, p) : (which == 0 ? Shape::ql1(u, l, p) : Shape::qp1(u, l, p)));
        } else {
          out[j] = which < 0 ? Shape::q2(u, l, p) : (which == 0 ? Shape::ql2(u, l, p) : Shape::qp2(u, l, p));
        }
      }
      return true;
    }
  }

  /**
   * \brief Proximal operator \f$\arg\min_b \{\mathrm{pen}(b) + \lVert b - v\rVert^2/(2t)\}\f$, coordinate by coordinate.
   *
   * Closed form for \f$t\rho \le 1\f$, the limit form at \f$t\rho = 1\f$
   * (the middle region of the threshold is then empty), accepted up to a
   * relative \f$4\epsilon\f$ so that \f$t = 1/\rho\f$ computed in floating point
   * passes. NaN for \f$t\rho > 1\f$, where the minimizer is not unique.
   * \param st State from `theta_state()`.
   * \param v Point, length `dim()`.
   * \param t Step \f$t > 0\f$.
   * \param out Minimizer, length `dim()`.
   */
  LAPLACE_HD void prox_beta(const state_type& st, const double* v, double t, double* out, double*) const {
    for (int j = 0; j < size; ++j) out[j] = prox1(st, v[j], t);
  }

  /**
   * \brief Proximal operator with a step per coordinate, \f$t_j\rho \le 1\f$ for each, NaN otherwise.
   * \param st State from `theta_state()`.
   * \param v Point, length `dim()`.
   * \param t Step per coordinate, length `dim()`.
   * \param out Minimizer, length `dim()`.
   */
  LAPLACE_HD void prox_beta(const state_type& st, const double* v, const double* t, double* out, double*) const {
    for (int j = 0; j < size; ++j) out[j] = prox1(st, v[j], t[j]);
  }

 private:
  LAPLACE_HD static double prox1(const state_type& st, double v, double t) {
    if (t * Shape::rho(st.shape) > 1.0 + 4.0 * Kokkos::Experimental::epsilon_v<double>)
      return Kokkos::Experimental::quiet_NaN_v<double>;
    return Shape::prox(v, t, st.lambda, st.shape);
  }
};

/**
 * \ingroup penalty_nonsmooth
 * \brief MCP (Zhang 2010): \f$q(u) = \lambda u - u^2/(2\gamma)\f$ for \f$u \le \gamma\lambda\f$, \f$\gamma\lambda^2/2\f$ beyond.
 *
 * \f$\theta = (\lambda, \gamma)\f$, \f$\lambda > 0\f$, \f$\gamma > 0\f$;
 * \f$\rho = 1/\gamma\f$.
 */
using mcp_penalty = separable_nonconvex_penalty<detail_nonconvex::mcp_shape>;

/**
 * \ingroup penalty_nonsmooth
 * \brief SCAD (Fan and Li 2001): \f$q(u) = \lambda u\f$, then \f$\lambda u - (u - \lambda)^2/(2(a-1))\f$, then \f$(a+1)\lambda^2/2\f$.
 *
 * \f$\theta = (\lambda, a)\f$, \f$\lambda > 0\f$, \f$a > 1\f$;
 * \f$\rho = 1/(a - 1)\f$.
 */
using scad_penalty = separable_nonconvex_penalty<detail_nonconvex::scad_shape>;

/// \ingroup penalty_nonsmooth
/// \brief Domains and regularity of the hyperparameters of `mcp_penalty`: \f$\lambda > 0\f$, \f$\gamma > 0\f$, both \f$C^1\f$.
inline constexpr distrib::distrib_param_info mcp_penalty_theta_params[2] = {
    {"lambda", 0.0, distrib::distrib_infinity, false, 1, false, 1},
    {"gamma", 0.0, distrib::distrib_infinity, false, 1, false, 1}};

/// \ingroup penalty_nonsmooth
/// \brief Domains and regularity of the hyperparameters of `scad_penalty`: \f$\lambda > 0\f$, \f$a > 1\f$, both \f$C^1\f$.
inline constexpr distrib::distrib_param_info scad_penalty_theta_params[2] = {
    {"lambda", 0.0, distrib::distrib_infinity, false, 1, false, 1},
    {"a", 1.0, distrib::distrib_infinity, false, 1, false, 1}};

static_assert(penalty<mcp_penalty> && penalty<scad_penalty>);

}  // namespace laplace::penalty
