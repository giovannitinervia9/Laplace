#pragma once

// Trust region with the exact subproblem, dense (note/optimizers.md,
// decision 4). At x with value f, gradient g and symmetric curvature H, the
// step p minimizes the model m(p) = g' p + p' H p / 2 over |p|_2 <= radius:
//
//   (H + lambda I) p = -g,  H + lambda I positive semidefinite,  lambda >= 0,
//   lambda (radius - |p|) = 0                       (More and Sorensen 1983).
//
// When H is positive definite (Cholesky) and the Newton step fits, p is the
// Newton step and lambda = 0. Otherwise H = V diag(w) V' (w increasing) and,
// with gamma = V' g, |p(lambda)|^2 = sum_i gamma_i^2 / (w_i + lambda)^2: lambda
// solves 1/|p(lambda)| = 1/radius (More and Sorensen's secular equation) by
// Newton's iteration safeguarded by bisection on
// [max(0, -w_1), max(0, -w_1) + |g| / radius], in the eigenbasis, so that every
// iteration costs O(n) after one decomposition per accepted point. The hard
// case (g orthogonal to the eigenvectors of w_1 < 0, the step at
// lambda = -w_1 inside the region) adds tau v_1 so that |p| = radius.
//
// The step is tried: rho = (f(x) - f(x + p)) / (m(0) - m(p)). rho < 1/4 or a
// non-finite f(x + p) (a refused point, decision 3) shrinks the radius to
// |p| / 4; rho > 3/4 with p on the boundary doubles it, up to max_radius;
// rho > acceptance moves x (Nocedal and Wright 2006, Algorithm 4.1). The
// decrement (1/2) g' H^-1 g read by the stop rule is NaN where H is not
// positive definite, so no such point is declared converged by the decrement.
// A trial point equal to x in every coordinate ends the run as stalled.

#include "laplace/dense/dense.hpp"
#include "laplace/optimizers/dense_curvature_solver.hpp"
#include "laplace/optimizers/minimize.hpp"
#include "laplace/optimizers/newton.hpp"
#include "laplace/optimizers/objective.hpp"
#include "laplace/optimizers/report.hpp"
#include "laplace/optimizers/stop_rules.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace laplace::optimizers {

namespace detail {

// The exact subproblem at one point: set() once per point (Cholesky and the
// Newton step), solve() once per radius (the eigendecomposition on first
// need, then O(n) per secular iteration).
class trust_region_subproblem {
 public:
  explicit trust_region_subproblem(int n)
      : solver_(n),
        newton_("laplace::optimizers::trust_region_subproblem::newton", n),
        eigenvalues_("laplace::optimizers::trust_region_subproblem::w", n),
        eigenvectors_("laplace::optimizers::trust_region_subproblem::V", n, n),
        gamma_("laplace::optimizers::trust_region_subproblem::gamma", n),
        coefficients_("laplace::optimizers::trust_region_subproblem::c", n) {}

  // Factorizes H; returns the decrement (1/2) g' H^-1 g, NaN when H is not
  // positive definite.
  double set(const dense_curvature_matrix& H, const optimizer_point& g) {
    H_ = H, g_ = g;
    have_eigen_ = false;
    have_newton_ = solver_.factorize(H) == curvature_factorization::positive_definite &&
                   solver_.newton_direction(g, newton_);
    const double slope = have_newton_ ? dot(g, newton_) : 0.0;
    have_newton_ = have_newton_ && slope <= 0.0;  // 0 only at g = 0
    newton_norm_ = have_newton_ ? std::sqrt(dot(newton_, newton_)) : 0.0;
    newton_decrement_ = have_newton_ ? 0.5 * std::abs(slope) : std::numeric_limits<double>::quiet_NaN();
    return newton_decrement_;
  }
  curvature_factorization factorization() const { return solver_.last_factorization(); }

  // Writes the step p for this radius; returns the predicted decrease
  // m(0) - m(p), NaN when H has no finite eigendecomposition.
  double solve(double radius, const optimizer_vector& p) {
    const int n = static_cast<int>(p.extent(0));
    if (have_newton_ && newton_norm_ <= radius) {
      for (int i = 0; i < n; ++i) p(i) = newton_(i);
      return newton_decrement_;
    }
    if (!have_eigen_) {
      if (!laplace::dense::symmetric_eigen_decomposition(H_, eigenvalues_, eigenvectors_).success)
        return std::numeric_limits<double>::quiet_NaN();
      for (int k = 0; k < n; ++k) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += eigenvectors_(i, k) * g_(i);
        gamma_(k) = s;
      }
      have_eigen_ = true;
    }
    secular(radius);
    double predicted = 0.0;
    for (int k = 0; k < n; ++k) {
      const double c = coefficients_(k);
      predicted -= gamma_(k) * c + 0.5 * eigenvalues_(k) * c * c;
    }
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int k = 0; k < n; ++k) s += eigenvectors_(i, k) * coefficients_(k);
      p(i) = s;
    }
    return predicted;
  }

 private:
  // |p(lambda)|^2 / gscale^2 and its derivative term sum (gamma_k / gscale)^2 / (w_k + lambda)^3,
  // skipping the terms with w_k + lambda <= 0; gamma is scaled by gscale = max |gamma_k|
  // so that the squares cannot overflow
  void scaled_norms(double lambda, double gscale, double& s2, double& s3) const {
    s2 = 0.0, s3 = 0.0;
    for (std::size_t k = 0; k < gamma_.extent(0); ++k) {
      const double den = eigenvalues_(k) + lambda;
      if (!(den > 0.0)) continue;
      const double t = gamma_(k) / gscale / den;
      s2 += t * t;
      s3 += t * t / den;
    }
  }

  // coefficients_ = V' p for the solution of the subproblem.
  void secular(double radius) {
    const int n = static_cast<int>(gamma_.extent(0));
    const double w1 = eigenvalues_(0);
    double gscale = 0.0, gnorm = 0.0, wmax = 0.0;
    for (int k = 0; k < n; ++k) gscale = std::max(gscale, std::abs(gamma_(k))), wmax = std::max(wmax, std::abs(eigenvalues_(k)));
    if (gscale == 0.0) gscale = 1.0;
    for (int k = 0; k < n; ++k) gnorm += (gamma_(k) / gscale) * (gamma_(k) / gscale);
    gnorm = gscale * std::sqrt(gnorm);
    const double scaled_radius = radius / gscale;
    double s2 = 0.0, s3 = 0.0;
    const double lower = std::max(0.0, -w1);
    // the rounding of the computed eigenvalues: w_k + lambda below it is not resolved
    const double delta = 8.0 * n * std::numeric_limits<double>::epsilon() * wmax + std::numeric_limits<double>::min();
    // the smallest lambda at which every w_k + lambda is resolved
    const double start = w1 > delta ? 0.0 : lower + delta;

    auto set_coefficients = [&](double lambda) {
      for (int k = 0; k < n; ++k) {
        const double den = eigenvalues_(k) + lambda;
        coefficients_(k) = den > delta ? -gamma_(k) / den : 0.0;
      }
    };

    scaled_norms(start, gscale, s2, s3);
    if (s2 <= scaled_radius * scaled_radius) {
      // the Newton step of a positive definite H, the minimum-norm step of a
      // singular semidefinite H, or the hard case
      set_coefficients(start == 0.0 ? 0.0 : lower);
      if (w1 < -delta) {
        double inside = 0.0;
        for (int k = 0; k < n; ++k) inside += coefficients_(k) * coefficients_(k);
        const double tau = std::sqrt(std::max(0.0, radius * radius - inside));
        coefficients_(0) += gamma_(0) > 0.0 ? -tau : tau;
      }
      return;
    }

    // phi(lambda) = 1/|p| - 1/radius, increasing and concave on (lower, inf);
    // root in [lo, hi] with phi(lo) < 0 <= phi(hi)
    double lo = start, hi = lower + gnorm / radius + delta, lambda = lo;
    for (int it = 0; it < 200; ++it) {
      scaled_norms(lambda, gscale, s2, s3);
      const double norm = std::sqrt(s2);  // |p| / gscale
      const double phi = 1.0 / norm - 1.0 / scaled_radius;
      if (std::abs(norm - scaled_radius) <= 1e-12 * scaled_radius) break;
      (phi < 0.0 ? lo : hi) = lambda;
      double next = lambda - phi * norm * s2 / s3;  // phi' = s3 / |p|^3 in the scaled units
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (next == lambda) break;
      lambda = next;
    }
    set_coefficients(lambda);
  }

  dense_curvature_solver solver_;
  optimizer_vector newton_;
  Kokkos::View<double*, Kokkos::HostSpace> eigenvalues_;
  dense_curvature_matrix eigenvectors_;
  Kokkos::View<double*, Kokkos::HostSpace> gamma_, coefficients_;
  dense_curvature_matrix H_;
  optimizer_point g_;
  bool have_newton_ = false, have_eigen_ = false;
  double newton_norm_ = 0.0, newton_decrement_ = 0.0;
};

}  // namespace detail

/**
 * \ingroup optimizers_second_order
 * \brief Trust-region method with the exact dense subproblem (Moré and Sorensen).
 *
 * Minimizes an objective modeling `curvature_objective` with `dense_curvature_matrix` on the host
 * by solving the quadratic trust-region subproblem \f$\min_{\lVert p\rVert_2 \le \Delta} g^\top p + \tfrac12 p^\top H p\f$
 * at each trial step.
 *
 * \tparam StopRule Stopping rule accepting the gradient, decrement, and previous point and value;
 *   `decrement_stop_rule` by default.
 * \see \ref optimizers_second_order, \ref optimizers, newton_optimizer, minimize()
 */
template <class StopRule = decrement_stop_rule>
  requires stop_rule_for<StopRule, detail::second_order_provides>
struct trust_region_optimizer {
  /**
   * \ingroup optimizers_second_order
   * \brief Stopping rule instance queried at the start and at every accepted point.
   */
  StopRule stop_rule{};
  /**
   * \ingroup optimizers_second_order
   * \brief Maximum number of trial steps (accepted or not) before returning `optimizer_status::iteration_budget` (default 1000).
   */
  int max_iterations = 1000;  // trial steps, accepted or not
  /**
   * \ingroup optimizers_second_order
   * \brief Value-evaluation budget checked before each trial step, default \f$10^5\f$.
   */
  long max_value_evaluations = 100000;
  /**
   * \ingroup optimizers_second_order
   * \brief Initial trust-region radius \f$\Delta\f$ (default \f$1\f$).
   */
  double initial_radius = 1.0;
  /**
   * \ingroup optimizers_second_order
   * \brief Upper bound on the trust-region radius \f$\Delta\f$ (default \f$+\infty\f$).
   */
  double max_radius = std::numeric_limits<double>::infinity();
  /**
   * \ingroup optimizers_second_order
   * \brief Minimum ratio \f$\rho\f$ of actual to predicted decrease required to accept a trial step (default \f$10^{-4}\f$).
   */
  double acceptance = 1e-4;  // a step is accepted when rho > acceptance
  /**
   * \ingroup optimizers_second_order
   * \brief Optional trace callback invoked at every accepted point with the Euclidean step length \f$\lVert p\rVert_2\f$.
   */
  optimizer_trace trace;

  /**
   * \ingroup optimizers_second_order
   * \brief Minimizes `objective` starting from `x0` using the dense trust-region method.
   *
   * Returns `optimizer_status::start_not_admissible` if \f$f(x_0)\f$ is not finite, `stalled` if the
   * model predicts no decrease or the trial point equals the current point in every coordinate,
   * `iteration_budget` or `evaluation_budget` on budget exhaustion, or the status from `stop_rule`.
   * Sets `decrement` to quiet NaN wherever \f$H\f$ has no Cholesky factor and counts every trial step
   * in `iterations`.
   *
   * \tparam O Objective type satisfying `curvature_objective` with `curvature_type` equal to `dense_curvature_matrix`.
   * \param objective Objective to minimize.
   * \param x0 Starting point \f$x_0 \in \mathbb{R}^n\f$.
   * \return Summary `optimizer_report` of the run.
   * \see \ref optimizers_second_order, minimize(), optimizer_report
   */
  template <curvature_objective O>
    requires std::same_as<typename O::curvature_type, dense_curvature_matrix>
  optimizer_report run(O& objective, const optimizer_point& x0) const {
    const auto start = std::chrono::steady_clock::now();
    const int n = static_cast<int>(x0.extent(0));
    optimizer_report report;
    optimizer_vector x("x", n), g("g", n), p("p", n), x_trial("x_trial", n), x_previous("x_previous", n);
    dense_curvature_matrix H("H", n, n);
    detail::trust_region_subproblem subproblem(n);
    Kokkos::deep_copy(x, x0);

    double f = objective.value_gradient_curvature(x, g, H);
    ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
    double f_previous = f, radius = initial_radius, last_step = 0.0;
    if (!std::isfinite(f)) {
      report.status = optimizer_status::start_not_admissible;
      report.point = x, report.value = f, report.gradient = g;
      report.elapsed_seconds = detail::seconds_since(start);
      return report;
    }

    bool moved = true;  // x changed since the last stop check
    for (int k = 0;; ++k) {
      if (moved) {
        report.decrement = subproblem.set(H, g);
        report.final_factorization = subproblem.factorization();
        if (k > 0 && trace) trace(optimizer_trace_point{k, x, f, g, report.decrement, last_step});
        stop_rule_input in;
        in.iteration = k, in.value = f, in.point = x, in.gradient = g, in.decrement = report.decrement;
        in.has_previous = k > 0, in.previous_value = f_previous, in.previous_point = x_previous;
        const stop_verdict verdict = stop_rule.verdict(in);
        if (verdict != stop_verdict::none) {
          report.status = detail::status_from_verdict(verdict);
          break;
        }
        moved = false;
      }
      if (k >= max_iterations) {
        report.status = optimizer_status::iteration_budget;
        break;
      }
      if (report.value_evaluations >= max_value_evaluations) {
        report.status = optimizer_status::evaluation_budget;
        break;
      }
      report.iterations = k + 1;

      const double predicted = subproblem.solve(radius, p);
      if (!(predicted > 0.0)) {  // no decrease predicted (g = 0, H semidefinite) or no decomposition
        report.status = optimizer_status::stalled;
        break;
      }
      if (!detail::trial_point(x, p, 1.0, x_trial)) {
        report.status = optimizer_status::stalled;
        break;
      }
      const double f_trial = objective.value(optimizer_point(x_trial));
      ++report.value_evaluations;
      const double rho = std::isfinite(f_trial) ? (f - f_trial) / predicted : -std::numeric_limits<double>::infinity();
      const double step = std::sqrt(detail::dot(p, p));
      if (rho < 0.25)
        radius = 0.25 * step;
      else if (rho > 0.75 && step >= 0.99 * radius)
        radius = std::min(2.0 * radius, max_radius);
      if (rho > acceptance) {
        Kokkos::deep_copy(x_previous, x);
        Kokkos::deep_copy(x, x_trial);
        f_previous = f;
        f = objective.value_gradient_curvature(x, g, H);
        ++report.value_evaluations, ++report.gradient_evaluations, ++report.curvature_evaluations;
        last_step = step;
        moved = true;
      }
    }

    report.point = x, report.value = f, report.gradient = g;
    report.elapsed_seconds = detail::seconds_since(start);
    return report;
  }
};

}  // namespace laplace::optimizers
