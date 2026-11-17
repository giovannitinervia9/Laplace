// The pieces shared by the penalties (laplace::penalty); the design is in
// note/penalties.md (sections 2 and 4), decided with Giovanni on 2026-11-16.
//
// A penalty is a function pen(beta; theta) of a coefficient vector beta in R^d
// and of hyperparameters theta; a proper penalty is minus the log of a prior,
// normalizing constant included. It declares, at compile time, the properties
// from which the engine chooses its algorithms (never from the name):
//
//   normalization       proper, intrinsic or unnormalizable      (class of section 2)
//   beta_regularity     infinitely_smooth, or the order r of C^r in beta
//   kinks               none, per coordinate, radial, or on a linear map of beta
//   prox                none, closed form or iterative
//   structure           separable, by group, or coupled
//   theta_regularity    infinitely_smooth, or the order r in theta
//   q_linear_in_theta   the precision of a quadratic penalty is sum_k lambda_k S_k
//   scale_mixture       the law of the variance when pen is a scale mixture of normals
//
// and one function, weak_convexity(theta), the constant rho = sup(-pen'') (0
// for a convex penalty, infinity when pen'' is unbounded below), a function
// because rho depends on the hyperparameters.
//
// The value and its derivatives are written by each penalty; the shapes are
// those of the multivariate distributions (distrib/multivariate_core.hpp).
#pragma once

#include <laplace/core/config.hpp>
#include <laplace/distrib/core.hpp>

#include <concepts>

namespace laplace::penalty {

/**
 * \ingroup penalty_framework
 * \brief Class of a penalty by normalizability of \f$\exp(-\mathrm{pen})\f$ in \f$\beta\f$.
 *
 * - `proper`: \f$\int \exp(-\mathrm{pen})\,\mathrm{d}\beta\f$ is finite and the
 *   normalizing constant is part of the value (a prior).
 * - `intrinsic`: the penalty is singular with a constant null space; the
 *   constant counts only the range dimensions and the pseudo-determinant
 *   (an improper prior flat along the null space).
 * - `unnormalizable`: the integral is infinite for every \f$\theta\f$ (SCAD, MCP).
 */
enum class penalty_normalization : int { proper, intrinsic, unnormalizable };

/**
 * \ingroup penalty_framework
 * \brief Geometry of the points at which a penalty is not smooth in \f$\beta\f$.
 *
 * - `none`: smooth.
 * - `per_coordinate`: a kink on each hyperplane \f$\beta_j = c\f$ (lasso).
 * - `radial`: a kink at \f$\lVert\beta_g\rVert = 0\f$ (group lasso).
 * - `linear_map`: a kink on \f$D\beta = 0\f$ (fused lasso).
 */
enum class penalty_kink_geometry : int { none, per_coordinate, radial, linear_map };

/**
 * \ingroup penalty_framework
 * \brief Availability of the proximal operator of a penalty.
 *
 * - `none`: no proximal operator is offered.
 * - `closed_form`: a closed-form proximal operator.
 * - `iterative`: a proximal operator computed by an iterative algorithm.
 */
enum class penalty_prox_kind : int { none, closed_form, iterative };

/**
 * \ingroup penalty_framework
 * \brief Coupling between the coordinates of \f$\beta\f$ in a penalty.
 *
 * - `separable`: a sum over coordinates.
 * - `grouped`: a sum over independent blocks of coordinates.
 * - `coupled`: the coordinates interact through a matrix or a map.
 */
enum class penalty_structure : int { separable, grouped, coupled };

/**
 * \ingroup penalty_framework
 * \brief Law of the variance when a penalty is a scale mixture of normals.
 *
 * Named by the law of the variance \f$\tau\f$ in \f$\beta \mid \tau \sim
 * \mathcal{N}(0, \tau)\f$: `inverse_gamma_variance` (Student's t),
 * `exponential_variance` (Laplace); `none` when no such representation is declared.
 */
enum class penalty_scale_mixture : int { none, inverse_gamma_variance, exponential_variance };

/**
 * \ingroup penalty_framework
 * \brief Concept satisfied by penalty types in \c laplace::penalty.
 *
 * Requires the compile-time descriptors `normalization`, `beta_regularity`,
 * `kinks`, `prox`, `structure`, `theta_regularity`, `q_linear_in_theta` and
 * `scale_mixture`, the dimension queries `dim()` and `theta_size()`, and
 * `weak_convexity(theta)`.
 *
 * \tparam P Penalty type to check.
 */
template <class P>
concept penalty = requires(const P p, const double* theta) {
  { P::normalization } -> std::convertible_to<penalty_normalization>;
  { P::beta_regularity } -> std::convertible_to<int>;
  { P::kinks } -> std::convertible_to<penalty_kink_geometry>;
  { P::prox } -> std::convertible_to<penalty_prox_kind>;
  { P::structure } -> std::convertible_to<penalty_structure>;
  { P::theta_regularity } -> std::convertible_to<int>;
  { P::q_linear_in_theta } -> std::convertible_to<bool>;
  { P::scale_mixture } -> std::convertible_to<penalty_scale_mixture>;
  { p.dim() } -> std::convertible_to<int>;
  { p.theta_size() } -> std::convertible_to<int>;
  { p.weak_convexity(theta) } -> std::convertible_to<double>;
};

}  // namespace laplace::penalty
