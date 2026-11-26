#pragma once

// The Laplace marginal criterion of a smooth model (laplace::engine; plan,
// sections 7.3, 7.11 and 7.12), with its exact gradient and Hessian.
//
// The inner objective (inner_fit.hpp) is
//
//   F(x; u) = -sum_i w_i l_i(x) + sum_t sum_r pen_t(x[C_tr]; theta_t(u)),
//
// each penalty minus the log of a prior with its normalizing constant (proper,
// or intrinsic with the pseudo-determinant over its range). The outer vector
// is u = (u_0, ..., u_{K-1}, psi): u_a is the free value of an outer
// hyperparameter, whose natural value phi_a(u_a) (the inverse of the bijector
// of penalty_hyperparameter) every hyperparameter of a penalty that reads a
// reads; psi are coordinates of x estimated on the criterion (held in the
// inner fit). Every other coordinate is free, except those the caller fixes,
// and is integrated by the Laplace approximation. With x^ the inner minimum at
// u, H the Hessian of F in the free coordinates at x^ and q their number,
//
//   V(u) = F(x^; u) + (1/2) log|H| - (q/2) log 2 pi,
//
// minus the logarithm of the Laplace approximation of the marginal likelihood
// (an intrinsic penalty is flat along its null space, so those directions are
// integrated against Lebesgue measure).
//
// Its derivatives. Write P_a = dF/du_a at fixed x, P_ab = d2F/du_a du_b at
// fixed x (both zero for psi), s_a = grad_x P_a, s_ab = grad_x P_ab,
// E_a = Hess_x P_a, E_ab = Hess_x P_ab, and tau^(a)_c = sum_rs Hi_rs
// d3 P_a/dx_r dx_s dx_c: through the bijectors, P_a = sum_{j: a} phi'_a
// dpen/dtheta_j and P_ab = sum_{j: a, k: b} phi'_a phi'_b d2pen/dtheta_j
// dtheta_k + delta_ab phi''_a sum_{j: a} dpen/dtheta_j, summed over the terms
// and their replicates. T and Q are the third and fourth derivatives of F in
// x, the likelihood's from the passes of outer_rows.hpp and the penalties' on
// the host; tau_c = sum_rs Hi_rs T_rsc. Let v_a = dx^/du_a: v_a = -Hi s_a for
// a hyperparameter; v_a = e_c - Hi H_full e_c for psi at coordinate c, where
// Hi is the inverse of H padded with zeros on held rows and columns and
// H_full the Hessian of F in all coordinates. Then
//
//   dH_a = E_a + T[v_a]
//   g_a  = P_a + grad_x F . e_c [psi] + (1/2) (tr(Hi E_a) + tau' v_a)
//   H_ab = F2_ab + (1/2) (-tr(Hi dH_a Hi dH_b) + tr(Hi E_ab) + tau^(a)' v_b
//                         + tau^(b)' v_a + Q[Hi; v_a, v_b] + tau' x^_ab)
//
// with F2_ab = P_ab + s_a' v_b + s_b' v_a + v_a' H_full v_b and
// x^_ab = -Hi (T[v_a, v_b] + E_b v_a + E_a v_b + s_ab).
//
// The Hessian H in log|H| is the observed one (decision E1 of 2026-11-25):
// where it is not positive definite at the inner minimum the criterion is +inf
// and the evaluation is counted.
//
// Hi is formed by one of two routes, at the user's choice
// (latent_inverse_route). Dense: the N x N matrix, one solve with the factor
// of H for every free coordinate. Selected: Hi only on the pattern of the
// factor of H (Takahashi's recursion, sparse_ldlt::compute_selected_inverse),
// which holds every pair of coordinates that meet in one unit or in one
// replicate of a penalty; these are all the entries that the value, the
// gradient and the passes read, and every product Hi w is a solve. The term
// tr(Hi dH_a Hi dH_b) of the Hessian still forms the dense Hi on both routes,
// so the default (automatic, provisional) takes the selected route for the
// value and the gradient and the dense one when the Hessian is asked.
// The full Hessian H_full is a list of entries.
//
// Refused at construction: an unnormalizable penalty (SCAD, MCP), whose
// exp(-pen) has no integral, and a penalty with kinks (the hyperparameters of a
// kinked penalty, by a path or by exact integration across the kink, are not
// in this version). The Hessian needs every estimated hyperparameter of a
// penalty to be at least C2 in theta.

#include <laplace/dense/dense.hpp>
#include <laplace/engine/inner_fit.hpp>
#include <laplace/engine/outer_rows.hpp>
#include <laplace/sparse/sparse_ldlt.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace laplace::engine {

namespace detail_marginal {

// X_a = Hi dH_a (N x N, dense) for the A matrices dH_a given in compressed columns (cp: A blocks of
// N + 1 offsets into rows and vals), then tr(X_a X_b) = tr(Hi dH_a Hi dH_b) for the pairs a <= b in
// lexicographic order, as Kokkos kernels (free functions: nvcc refuses an extended lambda in a private
// member function)
inline void trace_products(const Kokkos::View<double**, Kokkos::LayoutLeft, row_model_view::mem>& Hi, int N, int A,
                           const std::vector<int>& cp, const std::vector<int>& rows, const std::vector<double>& vals,
                           std::vector<double>& out) {
  using mem = row_model_view::mem;
  using exec = row_model_view::exec;
  auto to_dev = [](const auto& v, const char* name) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(v.data(), v.size());
    Kokkos::View<T*, mem> d(name, v.size());
    Kokkos::deep_copy(d, h);
    return d;
  };
  const auto dcp = to_dev(cp, "laplace::engine::dH_cols");
  const auto drow = to_dev(rows, "laplace::engine::dH_rows");
  const auto dval = to_dev(vals, "laplace::engine::dH_vals");
  Kokkos::View<double***, Kokkos::LayoutLeft, mem> X("laplace::engine::X", N, N, A);
  Kokkos::parallel_for(
      "laplace::engine::hi_times_dH", Kokkos::MDRangePolicy<exec, Kokkos::Rank<3>>({0, 0, 0}, {N, N, A}),
      KOKKOS_LAMBDA(const int i, const int j, const int a) {
        const int* c = &dcp(a * (N + 1));
        double s = 0.0;
        for (int q = c[j]; q < c[j + 1]; ++q) s += Hi(i, drow(q)) * dval(q);
        X(i, j, a) = s;
      });
  out.assign(A * (A + 1) / 2, 0.0);
  int ab = 0;
  for (int a = 0; a < A; ++a)
    for (int b = a; b < A; ++b, ++ab) {
      double t = 0.0;
      Kokkos::parallel_reduce(
          "laplace::engine::trace_XX", Kokkos::MDRangePolicy<exec, Kokkos::Rank<2>>({0, 0}, {N, N}),
          KOKKOS_LAMBDA(const int i, const int j, double& acc) { acc += X(i, j, a) * X(j, i, b); }, t);
      out[ab] = t;
    }
}

}  // namespace detail_marginal

/**
 * \ingroup engine
 * \brief How laplace_marginal_criterion forms \f$H^{-1}\f$, at the user's choice; the route used is reported in marginal_criterion_counts.
 * \see laplace_marginal_criterion::set_latent_inverse_route(), \ref engine
 */
enum class latent_inverse_route : int {
  automatic,  ///< Per evaluation: selected for the value and the gradient, dense when the outer Hessian is asked (which forms the dense inverse on either route). Provisional: revisited when the outer Hessian has a route without the dense inverse.
  dense,      ///< The dense \f$N \times N\f$ inverse from one solve per free coordinate (memory \f$8N^2\f$ bytes).
  selected,   ///< \f$H^{-1}\f$ on the pattern of the factor of \f$H\f$ only (Takahashi's recursion); the outer Hessian still forms the dense inverse.
};

/**
 * \ingroup engine
 * \brief Counts of a laplace_marginal_criterion: evaluations, inner fits, refusals, outer passes and time.
 * \see laplace_marginal_criterion, \ref engine
 */
struct marginal_criterion_counts {
  long evaluations[3] = {0, 0, 0};  ///< value, value and gradient, value gradient and Hessian
  long inner_fits = 0;
  long inner_failures = 0;           ///< inner fits that did not converge: the criterion is +inf there
  long inner_stalls = 0;             ///< inner fits whose step fell below rounding: their point is used
  optimizers::optimizer_status last_inner_failure = optimizers::optimizer_status::converged;  ///< why the last one failed
  long not_positive_definite = 0;    ///< points where the observed H was not positive definite: +inf
  long polishing_steps = 0;          ///< Newton steps taken with the factor of H after an inner fit
  long outer_passes[2] = {0, 0};     ///< order 3, order 4
  row_kernel_counts inner_kernels;   ///< the inner objective's kernels, summed over every inner fit
  double seconds_inner = 0.0, seconds_algebra = 0.0, seconds_passes = 0.0, seconds_penalties = 0.0;  ///< wall-clock seconds per phase (inner fits, algebra, passes, host penalties)
  double seconds_observed = 0.0;        ///< assembling the observed \f$H\f$ at \f$\hat x\f$ after each inner fit (outside the four phases)
  double seconds_factorization = 0.0;   ///< factorizations of \f$H\f$ at \f$\hat x\f$ (part of `seconds_algebra`)
  double seconds_inverse = 0.0;         ///< forming \f$H^{-1}\f$: the dense solves or the selected inverse (part of `seconds_algebra`)
  double seconds_to_device = 0.0;       ///< copying \f$H^{-1}\f$ on the information layout to the device (part of `seconds_passes`)
  double seconds_trace_products = 0.0;  ///< the dense \f$H^{-1}\f$ on the device, the products \f$H^{-1} \dot H_a\f$ and their traces (part of `seconds_algebra`)
  long factor_nonzeros = 0;             ///< entries of the factor of \f$H\f$ below its diagonal, at the last factorization
  latent_inverse_route latent_inverse_used = latent_inverse_route::dense;  ///< the route of the last evaluation of the derivatives (dense or selected)
};

/**
 * \ingroup engine
 * \brief The Laplace marginal criterion \f$V(u)\f$ of a smooth model, with exact gradient and Hessian.
 *
 * A `curvature_objective` (dense) over the outer vector \f$u\f$ (the free values of the outer
 * hyperparameters, then the coordinates \f$\psi\f$ estimated on the criterion) for the library's
 * optimizers; each new \f$u\f$ runs an inner fit from the previous inner minimum.
 *
 * \tparam List distrib_list of the distribs the model may name.
 * \see fit_marginal(), marginal_criterion_counts, penalty_term, penalty_hyperparameter, \ref engine
 */
template <class List>
class laplace_marginal_criterion {
 public:
  /**
   * \ingroup engine
   * \brief Outer curvature matrix type exposed to `optimizers::curvature_objective`.
   */
  using curvature_type = optimizers::dense_curvature_matrix;
  /**
   * \ingroup engine
   * \brief Sparse inner objective type minimized at each outer vector \f$u\f$.
   */
  using inner_objective = row_model_objective<List, optimizers::sparse_curvature_matrix>;

  /**
   * \ingroup engine
   * \brief Builds the criterion.
   * \param m Built model (must outlive the criterion).
   * \param penalties Penalty terms; a hyperparameter with source \f$a \ge 0\f$ reads outer coordinate \f$a\f$.
   * \param hyperparameters One per outer hyperparameter: the bijector from its natural value to \f$u_a\f$.
   * \param outer_coordinates Coordinates estimated on the criterion (\f$\psi\f$), held in the inner fit.
   * \param fixed Coordinates held at their value in `x0` and not estimated.
   * \param x0 Start of the first inner fit, length \f$N\f$.
   * \param inner_newton Options of the inner Newton fits.
   */
  laplace_marginal_criterion(const row_model& m, std::vector<penalty_term> penalties,
                             std::vector<penalty_hyperparameter> hyperparameters, std::vector<int> outer_coordinates,
                             std::vector<int> fixed, std::vector<double> x0, optimizers::newton_optimizer<> inner_newton = {})
      : model_(&m), n_(m.n_coordinates()), hyper_(std::move(hyperparameters)), psi_(std::move(outer_coordinates)),
        inner_(m, penalties, natural_at_zero(hyper_), row_curvature::observed_else_expected, held_list(psi_, fixed)),
        newton_(inner_newton), xhat_(std::move(x0)), held_(n_, 0) {
    if (static_cast<int>(xhat_.size()) != n_) throw std::invalid_argument("laplace_marginal_criterion: x0 must have N entries");
    for (int c : psi_) held_[c] = 1;
    for (int c : fixed) held_[c] = 1;
    q_ = 0;
    for (int j = 0; j < n_; ++j) q_ += held_[j] ? 0 : 1;
    const auto& pen = inner_.penalties();
    std::vector<int> seen(K(), 0);
    for (int t = 0; t < static_cast<int>(pen.terms().size()); ++t) {
      const auto& e = pen.evaluator(t);
      if (e.normalization() == penalty::penalty_normalization::unnormalizable)
        throw std::invalid_argument("laplace_marginal_criterion: an unnormalizable penalty has no marginal likelihood");
      if (e.kinks() != penalty::penalty_kink_geometry::none)
        throw std::invalid_argument("laplace_marginal_criterion: a penalty with kinks (its hyperparameters are not in this version)");
      for (int a : pen.terms()[t].theta_source)
        if (a >= 0) seen[a] = 1;
    }
    for (int a = 0; a < K(); ++a)
      if (!seen[a]) throw std::invalid_argument("laplace_marginal_criterion: an outer hyperparameter that no penalty reads");
    if (n_outer() < 1) throw std::invalid_argument("laplace_marginal_criterion: an empty outer vector");
  }

  /**
   * \ingroup engine
   * \brief Length \f$K + |\psi|\f$ of the outer vector \f$u\f$.
   * \return Number of outer coordinates.
   */
  int n_outer() const { return K() + static_cast<int>(psi_.size()); }
  /**
   * \ingroup engine
   * \brief Number \f$K\f$ of outer hyperparameters.
   * \return Number of hyperparameters.
   */
  int n_hyperparameters() const { return K(); }
  /**
   * \ingroup engine
   * \brief Last inner minimum \f$\hat x\f$ computed by the criterion.
   * \return Const reference to the inner coordinate vector of length \f$N\f$.
   */
  const std::vector<double>& inner_point() const { return xhat_; }
  /**
   * \ingroup engine
   * \brief Accumulated evaluation, inner-fit, refusal, outer-pass, and timing counters.
   * \return Const reference to `marginal_criterion_counts`.
   */
  const marginal_criterion_counts& counts() const { return counts_; }
  /**
   * \ingroup engine
   * \brief Inner sparse objective holding the penalty terms at the last hyperparameters, and the last assembly.
   * \return Const reference to `inner_objective`.
   */
  const inner_objective& inner() const { return inner_; }
  /**
   * \ingroup engine
   * \brief Whether each inner fit ends with one Newton step taken with the factor of \f$H\f$ (default `true`).
   *
   * The inner residual \f$d\f$ enters \f$\tfrac12\log|H|\f$ at first order; the step leaves a
   * residual of the order of \f$\lVert d\rVert^2\f$, at the cost of one more assembly and factorization.
   * \param on `false` to skip the step.
   */
  void set_polishing(bool on) { polish_ = on; }
  /**
   * \ingroup engine
   * \brief How \f$H^{-1}\f$ is formed for the gradient, the passes and the Hessian (default latent_inverse_route::automatic).
   *
   * Both routes give the same criterion and derivatives up to rounding; they differ in memory and time.
   * \param route The route; the one used is reported in marginal_criterion_counts::latent_inverse_used.
   */
  void set_latent_inverse_route(latent_inverse_route route) { route_ = route; }
  /**
   * \ingroup engine
   * \brief The natural values \f$\phi_a(u_a)\f$ of the outer hyperparameters at `u`.
   * \param u Outer vector (its first `n_hyperparameters()` entries are read).
   * \return One natural value per outer hyperparameter.
   */
  std::vector<double> natural_hyperparameters(const optimizers::optimizer_point& u) const {
    std::vector<double> h(K());
    for (int a = 0; a < K(); ++a) h[a] = hyper_[a].map.inverse(u(a));
    return h;
  }

  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$, returning \f$+\infty\f$ if the inner fit fails or \f$H\f$ is not positive definite.
   * \param u Outer coordinate vector of length `n_outer()`.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value(const optimizers::optimizer_point& u) {
    ++counts_.evaluations[0];
    evaluate(u, 0);
    return V_;
  }
  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$ and writes its exact gradient into `g`.
   *
   * If \f$V(u) = +\infty\f$, fills `g` with quiet NaN. Reuses cached results when called at the
   * same `u` as the previous evaluation.
   *
   * \param u Outer coordinate vector of length `n_outer()`.
   * \param g Writable host vector of length `n_outer()` receiving \f$\nabla V(u)\f$.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value_gradient(const optimizers::optimizer_point& u, const optimizers::optimizer_vector& g) {
    ++counts_.evaluations[1];
    evaluate(u, 1);
    for (int a = 0; a < n_outer(); ++a) g(a) = std::isfinite(V_) ? grad_[a] : std::numeric_limits<double>::quiet_NaN();
    return V_;
  }
  /**
   * \ingroup engine
   * \brief Evaluates \f$V(u)\f$, its exact gradient `g`, and its exact dense Hessian `H`.
   *
   * Throws `std::invalid_argument` if `n_outer() > row_outer_direction_capacity` or if a penalty
   * with an estimated hyperparameter is not \f$C^2\f$ in \f$\theta\f$. If \f$V(u) = +\infty\f$,
   * fills `g` and `H` with quiet NaN. Reuses cached results when called at the same `u` as the
   * previous evaluation.
   *
   * \param u Outer coordinate vector of length `n_outer()`.
   * \param g Writable host vector of length `n_outer()` receiving \f$\nabla V(u)\f$.
   * \param H Dense outer curvature matrix of size `n_outer()` by `n_outer()` receiving \f$\nabla^2 V(u)\f$.
   * \return Marginal criterion value \f$V(u)\f$, or \f$+\infty\f$.
   */
  double value_gradient_curvature(const optimizers::optimizer_point& u, const optimizers::optimizer_vector& g,
                                  curvature_type& H) {
    ++counts_.evaluations[2];
    evaluate(u, 2);
    const int A = n_outer();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int a = 0; a < A; ++a) {
      g(a) = std::isfinite(V_) ? grad_[a] : nan;
      for (int b = 0; b < A; ++b) H(a, b) = std::isfinite(V_) ? hess_[a + A * b] : nan;
    }
    return V_;
  }

 private:
  using clock = std::chrono::steady_clock;
  using dev_matrix = Kokkos::View<double**, Kokkos::LayoutLeft, row_model_view::mem>;
  using vec = std::vector<double>;

  const row_model* model_;
  int n_;
  std::vector<penalty_hyperparameter> hyper_;
  std::vector<int> psi_;
  inner_objective inner_;
  optimizers::newton_optimizer<> newton_;
  std::vector<double> xhat_;
  std::vector<int> held_;
  int q_ = 0;  // integrated coordinates
  bool polish_ = true;
  marginal_criterion_counts counts_;
  // the state at the last point
  std::vector<double> u_last_;
  int level_ = -1;  // -1 nothing, 0 value, 1 gradient, 2 Hessian
  double V_ = 0.0;
  std::vector<double> grad_, hess_;
  sparse::sparse_ldlt ldlt_;
  optimizers::sparse_curvature_matrix Hs_;
  latent_inverse_route route_ = latent_inverse_route::automatic;
  bool selected_ = false;   // the route of the current point
  std::vector<double> Hi_;  // the dense Hi, N x N, column-major
  bool have_hi_ = false, have_selected_ = false, hid_current_ = false, hinfo_current_ = false;
  dev_matrix Hid_;                // the dense Hi on the device (the trace products)
  row_model_view::dview Hinfo_;  // Hi on the information layout, on the device (the passes)

  int K() const { return static_cast<int>(hyper_.size()); }

  static std::vector<int> held_list(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> h = a;
    h.insert(h.end(), b.begin(), b.end());
    return h;
  }
  static std::vector<double> natural_at_zero(const std::vector<penalty_hyperparameter>& h) {
    std::vector<double> v;
    for (const auto& p : h) v.push_back(p.map.inverse(0.0));
    return v;
  }

  void evaluate(const optimizers::optimizer_point& u, int level) {
    const int A = n_outer();
    bool same = level_ >= 0 && static_cast<int>(u_last_.size()) == A;
    for (int a = 0; same && a < A; ++a) same = u(a) == u_last_[a];
    if (same && level_ >= level) return;
    if (!same) {
      u_last_.assign(u.data(), u.data() + A);
      have_hi_ = false, have_selected_ = false, hid_current_ = false, hinfo_current_ = false;
      level_ = 0;
      fit_value(u);
    }
    if (level >= 1 && std::isfinite(V_)) derivatives(level);
    level_ = std::max(level_, level);
  }

  void fit_value(const optimizers::optimizer_point& u) {
    const auto theta = natural_hyperparameters(u);
    for (double t : theta)
      if (!std::isfinite(t)) {
        V_ = std::numeric_limits<double>::infinity();
        return;
      }
    inner_.set_hyperparameters(theta);
    optimizers::optimizer_vector x0("x0", n_);
    for (int j = 0; j < n_; ++j) x0(j) = xhat_[j];
    for (std::size_t k = 0; k < psi_.size(); ++k) x0(psi_[k]) = u_last_[K() + k];
    auto t0 = clock::now();
    const auto r = fit_inner(inner_, x0, newton_);
    counts_.seconds_inner += std::chrono::duration<double>(clock::now() - t0).count();
    ++counts_.inner_fits;
    for (int k = 0; k < 4; ++k) counts_.inner_kernels.calls[k] += r.kernels.calls[k];
    counts_.inner_kernels.seconds += r.kernels.seconds;
    counts_.inner_kernels.expected_fallbacks += r.kernels.expected_fallbacks;
    counts_.inner_kernels.reused += r.kernels.reused;
    // a stall (the step no longer moves the point) is the inner minimum to the precision of x; it
    // happens from a start very near the minimum, before the decrement falls below its tolerance
    const bool stalled = r.optimizer.status == optimizers::optimizer_status::stalled;
    if (stalled) ++counts_.inner_stalls;
    if (!optimizers::optimizer_succeeded(r.optimizer.status) && !stalled) {
      ++counts_.inner_failures;
      counts_.last_inner_failure = r.optimizer.status;
      V_ = std::numeric_limits<double>::infinity();
      return;
    }
    for (int j = 0; j < n_; ++j) xhat_[j] = r.optimizer.point(j);
    optimizers::optimizer_vector g("g", n_);
    double F = 0.0;
    // the observed H at x^, then one Newton step with its factor (the polishing step): the inner
    // residual d enters (1/2) log|H| at first order, through tau' d, and the step leaves its square
    for (int step = 0; step < 2; ++step) {
      const auto xv = point_view();
      t0 = clock::now();
      F = inner_.observed_value_gradient_curvature(xv, g, Hs_);
      counts_.seconds_observed += std::chrono::duration<double>(clock::now() - t0).count();
      t0 = clock::now();
      const auto out = ldlt_.factorize(n_, Hs_.col_ptr.data(), Hs_.row_idx.data(), Hs_.values.data());
      const double tf = std::chrono::duration<double>(clock::now() - t0).count();
      counts_.seconds_algebra += tf, counts_.seconds_factorization += tf;
      counts_.factor_nonzeros = ldlt_.factor_nonzeros();
      if (out != sparse::sparse_ldlt_outcome::positive_definite) {
        ++counts_.not_positive_definite;
        V_ = std::numeric_limits<double>::infinity();
        return;
      }
      if (step == 1 || !polish_) break;
      std::vector<double> d(n_);
      for (int j = 0; j < n_; ++j) d[j] = held_[j] ? 0.0 : -g(j);
      ldlt_.solve(d.data());
      for (int j = 0; j < n_; ++j)
        if (!held_[j]) xhat_[j] += d[j];
      ++counts_.polishing_steps;
    }
    V_ = F + 0.5 * ldlt_.log_abs_determinant() - 0.5 * q_ * std::log(2.0 * std::numbers::pi);
  }

  optimizers::optimizer_vector point_view() const {
    optimizers::optimizer_vector x("xhat", n_);
    for (int j = 0; j < n_; ++j) x(j) = xhat_[j];
    return x;
  }


  // Hi_ij, zero on held rows and columns, from the route of the current point
  double hi(int i, int j) const {
    if (!selected_) return Hi_[i + static_cast<std::size_t>(n_) * j];
    return held_[i] || held_[j] ? 0.0 : ldlt_.selected_inverse_entry(i, j);
  }
  // Hi w, Hi padded with zeros on held rows and columns: one solve (H is the identity on them)
  std::vector<double> hi_times(std::vector<double> w) const {
    for (int j = 0; j < n_; ++j)
      if (held_[j]) w[j] = 0.0;
    ldlt_.solve(w.data());
    for (int j = 0; j < n_; ++j)
      if (held_[j]) w[j] = 0.0;
    return w;
  }
  // the dense Hi: one solve for each free coordinate
  // ponytail: N^2 memory and N solves; the term tr(Hi dH_a Hi dH_b) of the Hessian is the last
  // reader on the selected route, until it has a form of its own.
  void dense_inverse() {
    const int N = n_;
    Hi_.assign(static_cast<std::size_t>(N) * N, 0.0);
    std::vector<double> b(N);
    for (int j = 0; j < N; ++j) {
      if (held_[j]) continue;
      std::fill(b.begin(), b.end(), 0.0);
      b[j] = 1.0;
      ldlt_.solve(b.data());
      for (int i = 0; i < N; ++i) Hi_[i + static_cast<std::size_t>(N) * j] = held_[i] ? 0.0 : b[i];
    }
    have_hi_ = true;
  }

  // a sparse symmetric matrix as a list of entries (both triangles)
  struct entry_list {
    std::vector<int> r, c;
    std::vector<double> v;
    void add(int i, int j, double x) { r.push_back(i), c.push_back(j), v.push_back(x); }
    vec times(const vec& w, int N) const {
      vec o(N, 0.0);
      for (std::size_t k = 0; k < v.size(); ++k) o[r[k]] += v[k] * w[c[k]];
      return o;
    }
  };

  // what the penalties give to the derivatives at x^ (section comment at the top)
  struct penalty_parts {
    vec P;                          // P_a
    std::vector<vec> s;             // s_a
    std::vector<entry_list> E;      // E_a
    std::vector<vec> tau_a;         // tau^(a)
    vec tau;                        // the penalties' share of tau
    std::vector<entry_list> Tv;     // T_pen[v_a]
    vec Pab;                        // P_ab, pairs a <= b of hyperparameters
    std::vector<vec> sab;           // s_ab
    std::vector<entry_list> Eab;    // E_ab
    std::vector<vec> Tvv;           // T_pen[v_a, v_b], pairs a <= b of all outer coordinates
    vec Qvv;                        // Q_pen[Hi; v_a, v_b]
  };

  // the outer hyperparameter of each hyperparameter of a term, with phi' and phi''
  void chain(const optimizers::optimizer_point& u, vec& d1, vec& d2) const {
    d1.assign(K(), 0.0), d2.assign(K(), 0.0);
    for (int a = 0; a < K(); ++a) d1[a] = hyper_[a].map.inverse_d1(u(a)), d2[a] = hyper_[a].map.inverse_d2(u(a));
  }

  // first-order parts (P_a, s_a, E_a, tau^(a)) and the penalties' tau (needs Hi)
  void penalty_first(penalty_parts& pp, int level) const {
    const int N = n_, Kh = K();
    const auto& pen = inner_.penalties();
    optimizers::optimizer_vector uv("u", n_outer());
    for (int a = 0; a < n_outer(); ++a) uv(a) = u_last_[a];
    vec d1, d2;
    chain(uv, d1, d2);
    pp.P.assign(Kh, 0.0), pp.s.assign(Kh, vec(N, 0.0)), pp.E.assign(Kh, {}), pp.tau_a.assign(Kh, vec(N, 0.0));
    pp.tau.assign(N, 0.0);
    std::vector<double> b, buf;
    auto x = [&](int j) { return xhat_[j]; };
    for (int t = 0; t < static_cast<int>(pen.terms().size()); ++t) {
      const auto& term = pen.terms()[t];
      auto& e = pen.evaluator(t);
      buf.resize(std::max({e.tensor_size(3), e.tensor_size(2), e.tensor_size(1), 1}));
      for (const auto& rep : term.replicates) {
        penalty_set::gather(x, rep, b);
        if (e.derivative(3, 0, nullptr, b.data(), buf.data()))
          for_each_penalty_entry(e, 3, buf.data(), [&](const int* c, double v) {
            pp.tau[rep[c[2]]] += hi(rep[c[0]], rep[c[1]]) * v;
          });
        for (int j = 0; j < static_cast<int>(term.theta_source.size()); ++j) {
          const int a = term.theta_source[j];
          if (a < 0) continue;
          const double f = d1[a];
          if (e.derivative(0, 1, &j, b.data(), buf.data())) pp.P[a] += f * buf[0];
          if (e.derivative(1, 1, &j, b.data(), buf.data()))
            for_each_penalty_entry(e, 1, buf.data(), [&](const int* c, double v) { pp.s[a][rep[c[0]]] += f * v; });
          if (e.derivative(2, 1, &j, b.data(), buf.data()))
            for_each_penalty_entry(e, 2, buf.data(), [&](const int* c, double v) { pp.E[a].add(rep[c[0]], rep[c[1]], f * v); });
          if (e.derivative(3, 1, &j, b.data(), buf.data()))
            for_each_penalty_entry(e, 3, buf.data(), [&](const int* c, double v) {
              pp.tau_a[a][rep[c[2]]] += f * hi(rep[c[0]], rep[c[1]]) * v;
            });
        }
      }
    }
    (void)level;
  }

  // second-order parts: P_ab, s_ab, E_ab, T_pen[v_a], T_pen[v_a, v_b], Q_pen[Hi; v_a, v_b]
  void penalty_second(penalty_parts& pp, const std::vector<vec>& v) const {
    const int N = n_, Kh = K(), A = n_outer();
    const auto& pen = inner_.penalties();
    optimizers::optimizer_vector uv("u", A);
    for (int a = 0; a < A; ++a) uv(a) = u_last_[a];
    vec d1, d2;
    chain(uv, d1, d2);
    const int npair = Kh * (Kh + 1) / 2, apair = A * (A + 1) / 2;
    pp.Pab.assign(npair, 0.0), pp.sab.assign(npair, vec(N, 0.0)), pp.Eab.assign(npair, {});
    pp.Tv.assign(A, {}), pp.Tvv.assign(apair, vec(N, 0.0)), pp.Qvv.assign(apair, 0.0);
    auto pair = [&](int a, int b) { return a * Kh - a * (a - 1) / 2 + (b - a); };  // a <= b
    std::vector<double> b, buf;
    auto x = [&](int j) { return xhat_[j]; };
    for (int t = 0; t < static_cast<int>(pen.terms().size()); ++t) {
      const auto& term = pen.terms()[t];
      auto& e = pen.evaluator(t);
      const int h = static_cast<int>(term.theta_source.size());
      bool estimated = false;
      for (int a : term.theta_source) estimated = estimated || a >= 0;
      if (estimated && e.theta_regularity() < 2)
        throw std::invalid_argument("laplace_marginal_criterion: an estimated hyperparameter of a penalty that is not C2 in theta");
      buf.resize(std::max({e.tensor_size(4), e.tensor_size(3), e.tensor_size(2), e.tensor_size(1), 1}));
      for (const auto& rep : term.replicates) {
        penalty_set::gather(x, rep, b);
        if (e.derivative(3, 0, nullptr, b.data(), buf.data()))
          for_each_penalty_entry(e, 3, buf.data(), [&](const int* c, double w) {
            const int r0 = rep[c[0]], r1 = rep[c[1]], r2 = rep[c[2]];
            for (int a = 0; a < A; ++a) pp.Tv[a].add(r0, r1, w * v[a][r2]);
            for (int a = 0, ab = 0; a < A; ++a)
              for (int bb = a; bb < A; ++bb, ++ab) pp.Tvv[ab][r0] += w * v[a][r1] * v[bb][r2];
          });
        if (e.derivative(4, 0, nullptr, b.data(), buf.data()))
          for_each_penalty_entry(e, 4, buf.data(), [&](const int* c, double w) {
            const double hw = w * hi(rep[c[0]], rep[c[1]]);
            const int r2 = rep[c[2]], r3 = rep[c[3]];
            for (int a = 0, ab = 0; a < A; ++a)
              for (int bb = a; bb < A; ++bb, ++ab) pp.Qvv[ab] += hw * v[a][r2] * v[bb][r3];
          });
        for (int j = 0; j < h; ++j) {
          const int a = term.theta_source[j];
          if (a < 0) continue;
          // the phi'' term of P_aa, s_aa, E_aa
          const double f2 = d2[a];
          const int aa = pair(a, a);
          if (e.derivative(0, 1, &j, b.data(), buf.data())) pp.Pab[aa] += f2 * buf[0];
          if (e.derivative(1, 1, &j, b.data(), buf.data()))
            for_each_penalty_entry(e, 1, buf.data(), [&](const int* c, double w) { pp.sab[aa][rep[c[0]]] += f2 * w; });
          if (e.derivative(2, 1, &j, b.data(), buf.data()))
            for_each_penalty_entry(e, 2, buf.data(), [&](const int* c, double w) { pp.Eab[aa].add(rep[c[0]], rep[c[1]], f2 * w); });
          for (int k = j; k < h; ++k) {
            const int bh = term.theta_source[k];
            if (bh < 0) continue;
            const int idx[2] = {j, k};
            // (j, k) and, for j != k, (k, j): both orders of the pair of outer coordinates
            const double f = d1[a] * d1[bh];
            const int p = pair(std::min(a, bh), std::max(a, bh));
            // a pair (j, k) with j != k reading the same a counts twice in d2/du_a^2
            const double mult = (j != k && a == bh) ? 2.0 : 1.0;
            if (e.derivative(0, 2, idx, b.data(), buf.data())) pp.Pab[p] += mult * f * buf[0];
            if (e.derivative(1, 2, idx, b.data(), buf.data()))
              for_each_penalty_entry(e, 1, buf.data(), [&](const int* c, double w) { pp.sab[p][rep[c[0]]] += mult * f * w; });
            if (e.derivative(2, 2, idx, b.data(), buf.data()))
              for_each_penalty_entry(e, 2, buf.data(), [&](const int* c, double w) { pp.Eab[p].add(rep[c[0]], rep[c[1]], mult * f * w); });
          }
        }
      }
    }
  }

  void derivatives(int level) {
    const int N = n_, A = n_outer(), Kh = K();
    if (level == 2 && A > row_outer_direction_capacity)
      throw std::invalid_argument("laplace_marginal_criterion: more outer coordinates than row_outer_direction_capacity");
    auto t0 = clock::now();
    // H^-1, padded with zeros on held rows and columns, by the route of this point
    // automatic (provisional): the Hessian forms the dense inverse anyway, so a selected one would be extra
    selected_ = route_ == latent_inverse_route::selected || (route_ == latent_inverse_route::automatic && level < 2);
    counts_.latent_inverse_used = selected_ ? latent_inverse_route::selected : latent_inverse_route::dense;
    if (selected_ ? !have_selected_ : !have_hi_) {
      if (selected_) have_selected_ = ldlt_.compute_selected_inverse();
      else dense_inverse();
      counts_.seconds_inverse += std::chrono::duration<double>(clock::now() - t0).count();
    }
    // the full Hessian (a list of entries, both triangles) and gradient of F at x^, held coordinates included
    const row_assembly& as = inner_.last_assembly();
    entry_list Hf;
    vec gf = as.gradient;
    for (std::size_t q = 0; q < as.information.size(); ++q) {
      const int r = model_->information_row[q], c = model_->information_column[q];
      Hf.add(r, c, as.information[q]);
      if (r != c) Hf.add(c, r, as.information[q]);
    }
    const auto& pen = inner_.penalties();
    auto x = [&](int j) { return xhat_[j]; };
    pen.for_each_hessian_entry(x, [&](int r, int c, double v) { Hf.add(r, c, v); });
    pen.add_gradient(x, [&](int j, double v) { gf[j] += v; });
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    t0 = clock::now();
    penalty_parts pp;
    penalty_first(pp, level);
    counts_.seconds_penalties += std::chrono::duration<double>(clock::now() - t0).count();
    t0 = clock::now();
    vec trE(Kh, 0.0);
    for (int a = 0; a < Kh; ++a)
      for (std::size_t k = 0; k < pp.E[a].v.size(); ++k)
        trE[a] += hi(pp.E[a].c[k], pp.E[a].r[k]) * pp.E[a].v[k];
    // the directions v_a
    std::vector<vec> v(A);
    for (int a = 0; a < A; ++a) {
      if (a < Kh) {
        v[a] = hi_times(pp.s[a]);
        for (double& z : v[a]) z = -z;
      } else {
        const int c = psi_[a - Kh];
        vec col(N, 0.0);
        for (std::size_t k = 0; k < Hf.v.size(); ++k)
          if (Hf.c[k] == c) col[Hf.r[k]] += Hf.v[k];
        v[a] = hi_times(col);
        for (double& z : v[a]) z = -z;
        v[a][c] += 1.0;
      }
    }
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    // the pass over the units
    t0 = clock::now();
    dev_matrix Vd("laplace::engine::V", N, level == 2 ? A : 1);
    if (!hinfo_current_) {
      const auto td = clock::now();
      const int I = model_->view.information_size;
      if (static_cast<int>(Hinfo_.extent(0)) != I) Hinfo_ = row_model_view::dview("laplace::engine::Hi_information", I);
      auto h = Kokkos::create_mirror_view(Hinfo_);
      for (int q = 0; q < I; ++q) h(q) = hi(model_->information_row[q], model_->information_column[q]);
      Kokkos::deep_copy(Hinfo_, h);
      hinfo_current_ = true;
      counts_.seconds_to_device += std::chrono::duration<double>(clock::now() - td).count();
    }
    {
      auto hv = Kokkos::create_mirror_view(Vd);
      for (int a = 0; level == 2 && a < A; ++a)
        for (int i = 0; i < N; ++i) hv(i, a) = v[a][i];
      Kokkos::deep_copy(Vd, hv);
    }
    row_model_view::dview xd("laplace::engine::xhat", N);
    {
      auto h = Kokkos::create_mirror_view(xd);
      for (int j = 0; j < N; ++j) h(j) = xhat_[j];
      Kokkos::deep_copy(xd, h);
    }
    std::vector<double> pass;
    assemble_outer_row_model<List>(*model_, level == 2 ? 4 : 3, xd, Hinfo_, Vd, level == 2 ? A : 0, pass);
    ++counts_.outer_passes[level == 2 ? 1 : 0];
    counts_.seconds_passes += std::chrono::duration<double>(clock::now() - t0).count();
    t0 = clock::now();
    const outer_pass_layout lo{N, level == 2 ? A : 0, model_->view.information_size};
    vec tau(pass.begin() + lo.tau(), pass.begin() + lo.tau() + N);
    for (int j = 0; j < N; ++j) tau[j] += pp.tau[j];
    auto dot = [&](const vec& a, const vec& b) {
      double z = 0.0;
      for (int i = 0; i < N; ++i) z += a[i] * b[i];
      return z;
    };
    grad_.assign(A, 0.0);
    for (int a = 0; a < A; ++a) {
      const double tv = dot(tau, v[a]);
      grad_[a] = a < Kh ? pp.P[a] + 0.5 * (trE[a] + tv) : gf[psi_[a - Kh]] + 0.5 * tv;
    }
    counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    if (level == 2) {
      t0 = clock::now();
      penalty_second(pp, v);
      counts_.seconds_penalties += std::chrono::duration<double>(clock::now() - t0).count();
      t0 = clock::now();
      hessian(pass, lo, Hf, tau, pp, v);
      counts_.seconds_algebra += std::chrono::duration<double>(clock::now() - t0).count();
    }
  }

  void hessian(const std::vector<double>& pass, const outer_pass_layout& lo, const entry_list& Hf, const vec& tau,
               const penalty_parts& pp, const std::vector<vec>& v) {
    const int N = n_, A = n_outer(), Kh = K();
    auto kpair = [&](int a, int b) { return a * Kh - a * (a - 1) / 2 + (b - a); };  // a <= b < Kh
    // dH_a = E_a + T[v_a] in compressed columns (both triangles), for X_a = Hi dH_a and its traces on the device
    // ponytail: A dense N x N products (A N^2 memory, N nnz(dH) time); the arrowhead block formula
    // when a model with many groups is measured (decision D1).
    std::vector<int> cp(static_cast<std::size_t>(A) * (N + 1), 0), rows;
    std::vector<double> vals;
    {
      std::vector<std::vector<std::pair<int, double>>> col(N);
      for (int a = 0; a < A; ++a) {
        for (auto& c : col) c.clear();
        const double* T = pass.data() + lo.third_direction(a);
        for (int q = 0; q < model_->view.information_size; ++q) {
          if (T[q] == 0.0) continue;
          const int r = model_->information_row[q], c = model_->information_column[q];
          col[c].push_back({r, T[q]});
          if (r != c) col[r].push_back({c, T[q]});
        }
        const auto& tv = pp.Tv[a];
        for (std::size_t k = 0; k < tv.v.size(); ++k) col[tv.c[k]].push_back({tv.r[k], tv.v[k]});
        if (a < Kh) {
          const auto& E = pp.E[a];
          for (std::size_t k = 0; k < E.v.size(); ++k) col[E.c[k]].push_back({E.r[k], E.v[k]});
        }
        int* cpa = cp.data() + static_cast<std::size_t>(a) * (N + 1);
        cpa[0] = static_cast<int>(rows.size());
        for (int j = 0; j < N; ++j) {
          for (const auto& e : col[j]) rows.push_back(e.first), vals.push_back(e.second);
          cpa[j + 1] = static_cast<int>(rows.size());
        }
      }
    }
    std::vector<double> trxx_all;
    if (!have_hi_) {
      const auto ti = clock::now();
      dense_inverse();
      counts_.seconds_inverse += std::chrono::duration<double>(clock::now() - ti).count();
    }
    const auto tt = clock::now();
    if (!hid_current_) {
      if (static_cast<int>(Hid_.extent(0)) != N) Hid_ = dev_matrix("laplace::engine::Hi", N, N);
      Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(Hi_.data(), N, N);
      Kokkos::deep_copy(Hid_, h);
      hid_current_ = true;
    }
    detail_marginal::trace_products(Hid_, N, A, cp, rows, vals, trxx_all);
    counts_.seconds_trace_products += std::chrono::duration<double>(clock::now() - tt).count();
    const vec ht = hi_times(tau);  // Hi tau
    std::vector<vec> Hv(A);        // H_full v_b
    for (int b = 0; b < A; ++b) Hv[b] = Hf.times(v[b], N);
    hess_.assign(static_cast<std::size_t>(A) * A, 0.0);
    int ab = 0;
    for (int a = 0; a < A; ++a)
      for (int b = a; b < A; ++b, ++ab) {
        const bool ha = a < Kh, hb = b < Kh;
        const int kp = ha && hb ? kpair(a, b) : -1;
        // F2
        double f2 = kp >= 0 ? pp.Pab[kp] : 0.0;
        for (int i = 0; i < N; ++i) {
          f2 += v[a][i] * Hv[b][i];
          if (ha) f2 += pp.s[a][i] * v[b][i];
          if (hb) f2 += pp.s[b][i] * v[a][i];
        }
        const double trxx = trxx_all[ab];  // tr(Hi dH_a Hi dH_b)
        // tr(Hi E_ab)
        double trEab = 0.0;
        if (kp >= 0) {
          const auto& E = pp.Eab[kp];
          for (std::size_t k = 0; k < E.v.size(); ++k) trEab += hi(E.c[k], E.r[k]) * E.v[k];
        }
        // tau^(a)' v_b + tau^(b)' v_a
        double ta = 0.0;
        for (int i = 0; i < N; ++i) {
          if (ha) ta += pp.tau_a[a][i] * v[b][i];
          if (hb) ta += pp.tau_a[b][i] * v[a][i];
        }
        // x^_ab: rhs = T[v_a, v_b] + E_b v_a + E_a v_b + s_ab
        vec rhs(pass.begin() + lo.third_pair(ab), pass.begin() + lo.third_pair(ab) + N);
        for (int i = 0; i < N; ++i) rhs[i] += pp.Tvv[ab][i];
        if (hb) {
          const auto e = pp.E[b].times(v[a], N);
          for (int i = 0; i < N; ++i) rhs[i] += e[i];
        }
        if (ha) {
          const auto e = pp.E[a].times(v[b], N);
          for (int i = 0; i < N; ++i) rhs[i] += e[i];
        }
        if (kp >= 0)
          for (int i = 0; i < N; ++i) rhs[i] += pp.sab[kp][i];
        double tx = 0.0;  // tau' x^_ab = -(Hi tau)' rhs
        for (int j = 0; j < N; ++j) tx -= ht[j] * rhs[j];
        const double q = pass[lo.fourth_pair(ab)] + pp.Qvv[ab];
        const double h = f2 + 0.5 * (-trxx + trEab + ta + q + tx);
        hess_[a + A * b] = hess_[b + A * a] = h;
      }
  }
};

/**
 * \ingroup engine
 * \brief Result of fit_marginal(): the outer optimizer's report, the counts and the final inner point.
 * \see fit_marginal(), \ref engine
 */
struct marginal_fit_report {
  optimizers::optimizer_report optimizer;
  marginal_criterion_counts counts;
  std::vector<double> inner_point;
};

/**
 * \ingroup engine
 * \brief Minimizes the Laplace marginal criterion from `u0` with Newton's method on its exact Hessian.
 * \tparam Criterion Marginal criterion type (typically laplace_marginal_criterion).
 * \tparam Newton Outer Newton optimizer type (defaults to `optimizers::newton_optimizer<>`).
 * \param criterion The criterion (its counts accumulate).
 * \param u0 Start of the outer vector.
 * \param newton Options of the outer Newton optimizer.
 * \return marginal_fit_report.
 * \see laplace_marginal_criterion, \ref engine
 */
template <class Criterion, class Newton = optimizers::newton_optimizer<>>
marginal_fit_report fit_marginal(Criterion& criterion, const optimizers::optimizer_point& u0, const Newton& newton = {}) {
  marginal_fit_report r;
  r.optimizer = optimizers::minimize(newton, criterion, u0);
  r.counts = criterion.counts();
  r.inner_point = criterion.inner_point();
  return r;
}

}  // namespace laplace::engine
